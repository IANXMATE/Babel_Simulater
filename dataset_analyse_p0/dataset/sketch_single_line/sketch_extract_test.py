import os
import random
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image, ImageDraw, ImageFont
from skimage import morphology
from skan import Skeleton, summarize
from scipy.optimize import least_squares
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.signal import savgol_filter
import json

# ==========================================
# ⚙️ 全局配置
# ==========================================
CANVAS_SIZE = 256
MAX_BEZIER_ERROR = 1.5 

# ==========================================
# 🎯 0. 数据源 (精准指定字体与 Unicode)
# ==========================================
def render_unicode_glyph(font_path, unicode_hex, size=CANVAS_SIZE):
    if unicode_hex.startswith("U+"):
        unicode_hex = unicode_hex[2:]

    char = chr(int(unicode_hex, 16))
    font = ImageFont.truetype(font_path, int(size * 0.8))

    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)

    bbox = font.getbbox(char)
    if bbox is None:
        raise ValueError(f"无法渲染字符 {char} (Unicode: {unicode_hex})")
        
    left, top, right, bottom = bbox
    w, h = right - left, bottom - top

    x = (size - w) / 2 - left
    y = (size - h) / 2 - top

    draw.text((x, y), char, font=font, fill=0)

    # 转换为 PyTorch 训练管线所需的 Tensor 格式 [H, W, 4]
    arr = np.array(img).astype(np.float32) / 255.0
    binary = arr < 0.5
    target_tensor = torch.from_numpy(arr).float().unsqueeze(-1).repeat(1, 1, 4)
    
    return target_tensor, binary, char

# ==========================================
# 🧠 1. 图提取与闭环/主干分离 (Semantic Topology)
# ==========================================
def collapse_degree2_nodes(G):
    nodes = list(G.nodes())
    for n in nodes:
        if G.degree(n) == 2:
            edges = list(G.edges(n, keys=True, data=True))
            if len(edges) == 2:
                u, v1, k1, d1 = edges[0]
                u, v2, k2, d2 = edges[1]
                if v1 == v2 and k1 == k2: continue
                p1 = d1['path'] if v1 == n else d1['path'][::-1]
                p2 = d2['path'] if v2 == n else d2['path'][::-1]
                new_path = np.vstack([p1[:-1], p2])
                G.remove_edge(n, v1, key=k1)
                G.remove_edge(n, v2, key=k2)
                G.add_edge(v1, v2, path=new_path)
                G.remove_node(n)
    return G

def extract_loops_and_remove(G):
    """🌟 提取闭环(Loop)，并从原图中删除，防止破坏长路径提取"""
    loops = []
    simple_G = nx.Graph(G)
    cycles = nx.cycle_basis(simple_G)
    
    for cycle in cycles:
        loop_pixels = []
        edges_to_remove = []
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            if G.has_edge(u, v):
                edge_dict = G.get_edge_data(u, v)
                best_k = max(edge_dict, key=lambda k: len(edge_dict[k]['path']))
                p = edge_dict[best_k]['path']
                
                if len(loop_pixels) == 0:
                    loop_pixels.append(p)
                else:
                    last_pt = loop_pixels[-1][-1]
                    if np.linalg.norm(p[0] - last_pt) < np.linalg.norm(p[-1] - last_pt):
                        loop_pixels.append(p[1:])
                    else:
                        loop_pixels.append(p[::-1][1:])
                edges_to_remove.append((u, v, best_k))
        
        if loop_pixels:
            full_loop = np.vstack(loop_pixels)
            # 强制首尾相连
            full_loop = np.vstack([full_loop, full_loop[0]])
            loops.append(full_loop)
            for u, v, k in edges_to_remove:
                if G.has_edge(u, v, key=k):
                    G.remove_edge(u, v, key=k)
                    
    G.remove_nodes_from(list(nx.isolates(G)))
    return loops

def extract_longest_paths(G):
    """🌟 提取最长主干 (语义笔画)"""
    strokes = []
    while G.edges():
        for u, v, k, d in G.edges(keys=True, data=True):
            d['weight'] = len(d['path'])

        lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
        max_len = -1
        best_u, best_v = None, None
        
        for u in lengths:
            for v in lengths[u]:
                if lengths[u][v] > max_len:
                    max_len, best_u, best_v = lengths[u][v], u, v

        if best_u is None or best_u == best_v:
            u, v, k, d = list(G.edges(keys=True, data=True))[0]
            strokes.append(d['path'])
            G.remove_edge(u, v, key=k)
            continue

        path_nodes = nx.dijkstra_path(G, best_u, best_v, weight='weight')
        stroke_pixels = []
        for i in range(len(path_nodes)-1):
            n1, n2 = path_nodes[i], path_nodes[i+1]
            edge_dict = G.get_edge_data(n1, n2)
            best_k = max(edge_dict, key=lambda k: len(edge_dict[k]['path']))
            p = edge_dict[best_k]['path']
            if len(stroke_pixels) == 0:
                stroke_pixels.append(p)
            else:
                last_pt = stroke_pixels[-1][-1]
                if np.linalg.norm(p[0] - last_pt) < np.linalg.norm(p[-1] - last_pt):
                    stroke_pixels.append(p[1:])
                else:
                    stroke_pixels.append(p[::-1][1:])
            G.remove_edge(n1, n2, key=best_k)
            
        strokes.append(np.vstack(stroke_pixels))
        G.remove_nodes_from(list(nx.isolates(G)))
    return strokes

# ==========================================
# 🧠 2. 拟合、G1 连续与平滑回归 (Curve Fitting)
# ==========================================
def adaptive_curve_segments(pixel_path, k=4):
    """仅作拟合辅助，不破坏 Stroke 语义"""
    num_pts = len(pixel_path)
    if num_pts < k * 2 + 1: return [pixel_path]
    split_indices = [0]
    for i in range(k, num_pts - k):
        v1 = pixel_path[i] - pixel_path[i - k]
        v2 = pixel_path[i + k] - pixel_path[i]
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 < 1e-5 or norm2 < 1e-5: continue
        cos_theta = np.dot(v1, v2) / (norm1 * norm2)
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
        if angle > 110:   
            if i - split_indices[-1] > k * 2:
                split_indices.append(i)
    split_indices.append(num_pts)
    return [pixel_path[split_indices[i]:split_indices[i+1]] for i in range(len(split_indices)-1) if split_indices[i+1] - split_indices[i] > 3]

def cubic_bezier_np(P, t):
    mt = 1 - t
    return mt**3 * P[0] + 3*mt**2*t * P[1] + 3*mt*t**2 * P[2] + t**3 * P[3]

def fit_bezier_basic(pixel_path):
    """无递归基础拟合 (专门用于被等分的闭环)"""
    num_pts = len(pixel_path)
    if num_pts < 2: return None
    P0, P3 = pixel_path[0], pixel_path[-1]
    t_vals = np.linspace(0, 1, num_pts)[:, np.newaxis]
    P1_init, P2_init = P0 + (P3 - P0) * 0.33, P0 + (P3 - P0) * 0.66
    def residuals(controls):
        return (cubic_bezier_np(np.array([P0, controls[0:2], controls[2:4], P3]), t_vals) - pixel_path).flatten()
    res = least_squares(residuals, x0=np.concatenate([P1_init, P2_init]))
    return np.array([P0, res.x[0:2], res.x[2:4], P3])

def fit_bezier_adaptive(pixel_path, max_error=MAX_BEZIER_ERROR):
    """自适应多段拟合 (用于开放长路径)"""
    num_pts = len(pixel_path)
    if num_pts < 2: return []
    seg = np.linalg.norm(np.diff(pixel_path, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] == 0: return []
    t_vals = (s / s[-1])[:, np.newaxis]
    P0, P3 = pixel_path[0], pixel_path[-1]
    P1_init, P2_init = P0 + (P3 - P0) * 0.33, P0 + (P3 - P0) * 0.66
    def residuals(controls):
        return (cubic_bezier_np(np.array([P0, controls[0:2], controls[2:4], P3]), t_vals) - pixel_path).flatten()
    res = least_squares(residuals, x0=np.concatenate([P1_init, P2_init]))
    P_opt = np.array([P0, res.x[0:2], res.x[2:4], P3])
    error = np.mean(np.linalg.norm(cubic_bezier_np(P_opt, t_vals) - pixel_path, axis=1))
    if error > max_error and num_pts > 10:
        mid = num_pts // 2
        return fit_bezier_adaptive(pixel_path[:mid+1], max_error) + fit_bezier_adaptive(pixel_path[mid:], max_error)
    return [P_opt]

def enforce_g1_continuity(beziers, is_closed=False):
    """🌟 终极版连续性：保证同一 Semantic Stroke 内部平滑不断带"""
    if len(beziers) <= 1: return beziers
    fixed = [beziers[0].copy()]
    for i in range(1, len(beziers)):
        prev, curr = fixed[-1], beziers[i].copy()
        curr[0] = prev[-1] 
        tangent = prev[-1] - prev[-2]
        t_norm = np.linalg.norm(tangent)
        if t_norm > 1e-5:
            curr[1] = curr[0] + (tangent / t_norm) * (np.linalg.norm(curr[-1] - curr[0]) * 0.3)
        fixed.append(curr)
        
    if is_closed and len(fixed) > 1:
        first, last = fixed[0], fixed[-1]
        last[-1] = first[0] # 首尾硬缝合
        tangent = last[-1] - last[-2]
        t_norm = np.linalg.norm(tangent)
        if t_norm > 1e-5:
            first[1] = first[0] + (tangent / t_norm) * (np.linalg.norm(first[-1] - first[0]) * 0.3)
            
    return fixed

def regress_width_dt_smooth(mother_bezier, dt_map):
    """基于距离场 (DT) 提取亚像素宽度，并应用 SG 滤波"""
    t_vals = np.linspace(0, 1, 50)[:, np.newaxis]
    m_pts = cubic_bezier_np(mother_bezier, t_vals)
    coords = np.vstack([m_pts[:, 1], m_pts[:, 0]])
    local_radii = map_coordinates(dt_map, coords, order=1)
    if len(local_radii) >= 9:
        local_radii = savgol_filter(local_radii, window_length=9, polyorder=2)
    local_radii = np.maximum(local_radii, 0.1) 
    def width_residuals(W):
        mt = 1 - t_vals.flatten()
        t = t_vals.flatten()
        return (mt**3*W[0] + 3*mt**2*t*W[1] + 3*mt*t**2*W[2] + t**3*W[3]) - local_radii
    res = least_squares(width_residuals, x0=np.array([4.0, 4.0, 4.0, 4.0]))
    return np.abs(res.x), float(np.mean(local_radii))

# ==========================================
# 🚀 3. 终极组装管线 (主入口)
# ==========================================
def run_ultimate_tokenizer(font_path, unicode_hex):
    # 🌟 调用指定渲染函数
    target_img, binary_img, char = render_unicode_glyph(font_path, unicode_hex)
    dt_map = distance_transform_edt(binary_img)

    skel_img = morphology.skeletonize(binary_img)
    skel_obj = Skeleton(skel_img)
    
    G = nx.MultiGraph()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        branch_data = summarize(skel_obj)
        
    for index, row in branch_data.iterrows():
        coords = skel_obj.path_coordinates(index)
        path_xy = np.column_stack([coords[:, 1], coords[:, 0]])
        src, dst = int(row['node-id-src']), int(row['node-id-dst'])
        if np.linalg.norm(path_xy[0] - skel_obj.coordinates[src][::-1]) > 1.0:
            path_xy = path_xy[::-1]
        G.add_edge(src, dst, key=index, path=path_xy)

    G = collapse_degree2_nodes(G)
    
    # 🌟 两步走：先提取完美的环，再提取剩余的树状主干
    loops = extract_loops_and_remove(G)
    longest_paths = extract_longest_paths(G)

    final_tokens = []
    bezier_global_id, stroke_id = 0, 0
    
    # === 处理 1：闭环 (强制 4 等分 + 闭合 G1 连续) ===
    for loop_path in loops:
        n = len(loop_path)
        step = n // 4
        beziers = []
        for i in range(4):
            start_idx = i * step
            end_idx = (i + 1) * step + 1 if i < 3 else n
            seg = loop_path[start_idx:end_idx]
            if len(seg) >= 4:
                b = fit_bezier_basic(seg)
                if b is not None: beziers.append(b)
                
        beziers = enforce_g1_continuity(beziers, is_closed=True)
        
        for p_opt in beziers:
            w_opt, avg_w = regress_width_dt_smooth(p_opt, dt_map)
            final_tokens.append({
                "unicode": unicode_hex, "stroke_type": "loop", "stroke_id": stroke_id, "bezier_id": bezier_global_id, 
                "mother_bezier": p_opt.tolist(), "width_bezier": w_opt.tolist()
            })
            bezier_global_id += 1
        stroke_id += 1

    # === 处理 2：开放主干 (辅助断点拟合 + 开放 G1 连续) ===
    for path in longest_paths:
        if len(path) < 5: continue
        curve_segments = adaptive_curve_segments(path)
        beziers = []
        for seg in curve_segments:
            beziers += fit_bezier_adaptive(seg)
            
        beziers = enforce_g1_continuity(beziers, is_closed=False)
        
        for p_opt in beziers:
            w_opt, avg_w = regress_width_dt_smooth(p_opt, dt_map)
            final_tokens.append({
                "unicode": unicode_hex, "stroke_type": "path", "stroke_id": stroke_id, "bezier_id": bezier_global_id, 
                "mother_bezier": p_opt.tolist(), "width_bezier": w_opt.tolist()
            })
            bezier_global_id += 1
        stroke_id += 1

    print(f"✅ 解析 '{char}' ({unicode_hex}): 提取 {len(loops)} 个闭环, {len(longest_paths)} 笔主干。")

    # ================= 📊 渲染对比 =================
    import matplotlib.cm as cm
    device = torch.device('cpu')
    grid_y, grid_x = torch.meshgrid(torch.linspace(0, CANVAS_SIZE - 1, CANVAS_SIZE, device=device),
                                    torch.linspace(0, CANVAS_SIZE - 1, CANVAS_SIZE, device=device), indexing='ij')
    clean_pred_img = torch.ones(CANVAS_SIZE, CANVAS_SIZE, 3, device=device)
    t_plot = torch.linspace(0, 1, 200, device=device).unsqueeze(1) 
    cmap = cm.get_cmap('Set1', max(1, stroke_id))

    for token in final_tokens:
        P_tensor = torch.tensor(token["mother_bezier"], dtype=torch.float32, device=device)
        W_tensor = torch.tensor(token["width_bezier"], dtype=torch.float32, device=device)
        # 根据 stroke_id 上色，展示语义分层结果
        color = torch.tensor(cmap(token["stroke_id"])[:3], dtype=torch.float32, device=device)
        
        mt = 1 - t_plot
        M_t = mt**3*P_tensor[0] + 3*mt**2*t_plot*P_tensor[1] + 3*mt*t_plot**2*P_tensor[2] + t_plot**3*P_tensor[3]
        radii = torch.abs(mt**3*W_tensor[0] + 3*mt**2*t_plot*W_tensor[1] + 3*mt*t_plot**2*W_tensor[2] + t_plot**3*W_tensor[3]).squeeze()
        dist = torch.sqrt((grid_x.unsqueeze(-1) - M_t[:, 0])**2 + (grid_y.unsqueeze(-1) - M_t[:, 1])**2)
        stroke_alpha = torch.max(torch.clamp(radii.view(1, 1, -1) - dist + 0.5, 0, 1), dim=-1)[0].unsqueeze(-1)
        clean_pred_img = clean_pred_img * (1 - stroke_alpha) + color * stroke_alpha

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=120)
    axes[0].imshow(target_img[:,:,0].cpu().numpy(), cmap='gray')
    axes[0].set_title(f"Target: {unicode_hex} ('{char}')", fontsize=16)
    axes[0].axis("off")
    axes[1].imshow(clean_pred_img.cpu().numpy())
    axes[1].set_title(f"Reconstructed ({stroke_id} Semantic Strokes)", fontsize=16)
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

# ==========================================
# 🚀 启动！
# ==========================================
if __name__ == "__main__":
    # 🌟 请确保 FONT_PATH 路径正确，指向你的 Tamil 字体
    FONT_PATH = "../alien_tensors_raw/NotoSansTamil[wdth,wght].ttf"
    UNICODE_HEX = "U+0BB4"
    
    run_ultimate_tokenizer(FONT_PATH, UNICODE_HEX)