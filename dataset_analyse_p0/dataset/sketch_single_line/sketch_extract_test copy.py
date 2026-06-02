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
import matplotlib.cm as cm

# ==========================================
# ⚙️ 全局配置
# ==========================================
CANVAS_SIZE = 256
MAX_BEZIER_ERROR = 1.5 

# ==========================================
# 🎯 0. 渲染生成器
# ==========================================
def render_unicode_glyph(font_path, unicode_hex, size=CANVAS_SIZE):
    if unicode_hex.startswith("U+"): unicode_hex = unicode_hex[2:]
    char = chr(int(unicode_hex, 16))
    font = ImageFont.truetype(font_path, int(size * 0.8))

    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    bbox = font.getbbox(char)
    if bbox is None: raise ValueError(f"无法渲染字符 {char}")
        
    left, top, right, bottom = bbox
    w, h = right - left, bottom - top
    x, y = (size - w) / 2 - left, (size - h) / 2 - top
    draw.text((x, y), char, font=font, fill=0)

    arr = np.array(img).astype(np.float32) / 255.0
    binary = arr < 0.5
    target_tensor = torch.from_numpy(arr).float().unsqueeze(-1).repeat(1, 1, 4)
    return target_tensor, binary, char

# ==========================================
# 🧠 1. 图论拓扑核心引擎 (修复了交点与共线问题)
# ==========================================
def collapse_degree2_nodes(G):
    """清理多余节点，将单线段合并"""
    nodes = list(G.nodes())
    for n in nodes:
        if G.degree(n) == 2:
            edges = list(G.edges(n, keys=True, data=True))
            if len(edges) == 2:
                u, v1, k1, d1 = edges[0]
                u, v2, k2, d2 = edges[1]
                if v1 == v2 and k1 == k2: continue # 防自环崩溃
                
                p1 = d1['path'] if v1 == n else d1['path'][::-1]
                p2 = d2['path'] if v2 == n else d2['path'][::-1]
                new_path = np.vstack([p1[:-1], p2])
                
                G.remove_edge(n, v1, key=k1)
                G.remove_edge(n, v2, key=k2)
                G.add_edge(v1, v2, path=new_path)
                G.remove_node(n)
    return G

def resolve_junctions(G, angle_tolerance=45):
    """🌟 灵魂修复：处理所有十字路口和T字路口，缝合共线分支"""
    def get_outward_tangent(path, at_start=True, step=6):
        step = min(step, len(path) - 1)
        if step == 0: return np.array([0.0, 0.0])
        # 获取向外射出的切线向量
        vec = (path[step] - path[0]) if at_start else (path[-1 - step] - path[-1])
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-6 else np.array([0.0, 0.0])

    # 点积阈值：允许直线有 45 度的扭动误差 (135° ~ 180°)
    threshold_dot = np.cos(np.radians(180 - angle_tolerance))

    merged = True
    while merged:
        merged = False
        for node in list(G.nodes()):
            if G.degree(node) < 3: continue # 只处理真实交点
            
            edges = list(G.edges(node, keys=True, data=True))
            best_pair, best_dot = None, threshold_dot

            for i in range(len(edges)):
                for j in range(i+1, len(edges)):
                    u1, v1, k1, d1 = edges[i]
                    u2, v2, k2, d2 = edges[j]
                    if (u1==v1) or (u2==v2) or (k1==k2 and ((u1,v1)==(u2,v2) or (u1,v1)==(v2,u2))):
                        continue

                    vec1 = get_outward_tangent(d1['path'], at_start=(u1 == node))
                    vec2 = get_outward_tangent(d2['path'], at_start=(u2 == node))
                    
                    dot = np.dot(vec1, vec2)
                    if dot < best_dot:
                        best_dot = dot
                        best_pair = (edges[i], edges[j])

            if best_pair:
                e1, e2 = best_pair
                u1, v1, k1, d1 = e1
                u2, v2, k2, d2 = e2

                p1 = d1['path'] if v1 == node else d1['path'][::-1]
                p2 = d2['path'] if u2 == node else d2['path'][::-1]
                new_path = np.vstack([p1[:-1], p2])

                G.remove_edge(u1, v1, key=k1)
                G.remove_edge(u2, v2, key=k2)
                G.add_edge(u1 if v1 == node else v1, v2 if u2 == node else u2, path=new_path)
                merged = True
                break 
    return G

def extract_semantic_strokes(G):
    """分离闭环与主干"""
    strokes = []
    
    # 1. 直接提取完美的独立闭环 (Self-Loops)
    self_loops = list(nx.selfloop_edges(G, keys=True, data=True))
    for u, v, k, d in self_loops:
        strokes.append({'path': d['path'], 'is_closed': True})
        G.remove_edge(u, v, key=k)

    G.remove_nodes_from(list(nx.isolates(G)))

    # 2. 提取剩余的穿透长线
    while G.edges():
        for u, v, k, d in G.edges(keys=True, data=True): d['weight'] = len(d['path'])
        lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
        max_len, best_u, best_v = -1, None, None
        
        for u in lengths:
            for v in lengths[u]:
                if lengths[u][v] > max_len: max_len, best_u, best_v = lengths[u][v], u, v

        if best_u is None or best_u == best_v:
            u, v, k, d = list(G.edges(keys=True, data=True))[0]
            strokes.append({'path': d['path'], 'is_closed': False})
            G.remove_edge(u, v, key=k)
            continue

        path_nodes = nx.dijkstra_path(G, best_u, best_v, weight='weight')
        stroke_pixels = []
        for i in range(len(path_nodes)-1):
            n1, n2 = path_nodes[i], path_nodes[i+1]
            edge_dict = G.get_edge_data(n1, n2)
            best_k = max(edge_dict, key=lambda x: len(edge_dict[x]['path']))
            p = edge_dict[best_k]['path']
            if len(stroke_pixels) == 0: stroke_pixels.append(p)
            else:
                last_pt = stroke_pixels[-1][-1]
                if np.linalg.norm(p[0] - last_pt) < np.linalg.norm(p[-1] - last_pt): stroke_pixels.append(p[1:])
                else: stroke_pixels.append(p[::-1][1:])
            G.remove_edge(n1, n2, key=best_k)
            
        strokes.append({'path': np.vstack(stroke_pixels), 'is_closed': False})
        G.remove_nodes_from(list(nx.isolates(G)))
        
    return strokes

# ==========================================
# 🧠 2. 贝塞尔拟合与 G1 连续
# ==========================================
def adaptive_curve_segments(pixel_path, k=4):
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
        last[-1] = first[0] 
        tangent = last[-1] - last[-2]
        t_norm = np.linalg.norm(tangent)
        if t_norm > 1e-5:
            first[1] = first[0] + (tangent / t_norm) * (np.linalg.norm(first[1] - first[0]))
    return fixed

def regress_width_dt_smooth(mother_bezier, dt_map):
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
# 📊 3. 终极六宫格绘图面板 (保留原汁原味)
# ==========================================
def draw_bezier_on_canvas(canvas, P, width, color):
    ts = np.linspace(0, 1, 100)[:, None]
    mt = 1 - ts
    curve = (mt**3 * P[0] + 3 * mt**2 * ts * P[1] + 3 * mt * ts**2 * P[2] + ts**3 * P[3])
    r = max(1, int(width * 1.2))
    for x, y in curve.astype(int):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                xx, yy = x + dx, y + dy
                if 0 <= xx < canvas.shape[1] and 0 <= yy < canvas.shape[0]:
                    if dx*dx + dy*dy <= r*r: canvas[yy, xx] = color

def plot_six_panel_dashboard(target_img, binary_img, skel_img, dt_map, final_tokens):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(target_img[:,:,0].cpu().numpy(), cmap="gray")
    axes[0, 0].set_title("1. Glyph Target", fontsize=14)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(binary_img, cmap="gray")
    axes[0, 1].set_title("2. Binary Mask", fontsize=14)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(skel_img, cmap="gray")
    axes[0, 2].set_title("3. Raw Skeleton", fontsize=14)
    axes[0, 2].axis("off")

    axes[1, 0].imshow(dt_map, cmap="magma")
    axes[1, 0].set_title("4. Distance Transform (DT)", fontsize=14)
    axes[1, 0].axis("off")

    recon = np.ones((*binary_img.shape, 3))
    cmap_tokens = cm.get_cmap("tab20", max(1, len(final_tokens)))
    
    for i, t in enumerate(final_tokens):
        P = np.array(t["mother_bezier"])
        # 同一个 stroke 用相近的颜色，不同 bezier 段显示微小色差
        base_color = cmap_tokens(t["stroke_id"])[:3]
        draw_bezier_on_canvas(recon, P, width=3.0, color=base_color)
        draw_bezier_on_canvas(recon, np.array([P[0], P[0], P[0], P[0]]), width=1.0, color=[0,0,0])
        
    axes[1, 1].imshow(recon)
    axes[1, 1].set_title(f"5. Bezier Tokens ({len(final_tokens)} Segments)", fontsize=14)
    axes[1, 1].axis("off")

    heat = np.zeros(binary_img.shape)
    for t in final_tokens:
        P, W = np.array(t["mother_bezier"]), np.array(t["width_bezier"])
        ts = np.linspace(0, 1, 80)[:, None]
        mt = 1 - ts
        curve = (mt**3 * P[0] + 3 * mt**2 * ts * P[1] + 3 * mt * ts**2 * P[2] + ts**3 * P[3])
        radii = (mt**3 * W[0] + 3 * mt**2 * ts * W[1] + 3 * mt * ts**2 * W[2] + ts**3 * W[3]).flatten()
        for idx, (x, y) in enumerate(curve.astype(int)):
            if 0 <= x < heat.shape[1] and 0 <= y < heat.shape[0]:
                heat[y, x] = max(heat[y, x], radii[idx]) 
                
    axes[1, 2].imshow(heat, cmap="hot")
    axes[1, 2].set_title("6. Continuous Width Heatmap", fontsize=14)
    axes[1, 2].axis("off")

    plt.tight_layout(pad=3.0)
    plt.show()

# ==========================================
# 🚀 4. 主控核心
# ==========================================
def run_ultimate_tokenizer(font_path, unicode_hex):
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

    # 🌟 完美三步走：清理 -> 共线缝合交点 -> 再清理
    G = collapse_degree2_nodes(G)
    G = resolve_junctions(G, angle_tolerance=45) 
    G = collapse_degree2_nodes(G)
    
    semantic_strokes = extract_semantic_strokes(G)

    final_tokens = []
    bezier_global_id = 0
    
    for stroke_id, stroke in enumerate(semantic_strokes):
        path, is_closed = stroke['path'], stroke['is_closed']
        if len(path) < 5: continue

        if is_closed:
            n = len(path)
            step = n // 4
            beziers = []
            for i in range(4):
                start_idx = i * step
                end_idx = (i + 1) * step + 1 if i < 3 else n
                seg = path[start_idx:end_idx]
                if len(seg) >= 4:
                    b = fit_bezier_basic(seg)
                    if b is not None: beziers.append(b)
            beziers = enforce_g1_continuity(beziers, is_closed=True)
            stroke_type = "loop"
        else:
            curve_segments = adaptive_curve_segments(path)
            beziers = []
            for seg in curve_segments: beziers += fit_bezier_adaptive(seg)
            beziers = enforce_g1_continuity(beziers, is_closed=False)
            stroke_type = "path"

        for p_opt in beziers:
            w_opt, avg_w = regress_width_dt_smooth(p_opt, dt_map)
            final_tokens.append({
                "unicode": unicode_hex, "stroke_type": stroke_type, "stroke_id": stroke_id, "bezier_id": bezier_global_id, 
                "mother_bezier": p_opt.tolist(), "width_bezier": w_opt.tolist()
            })
            bezier_global_id += 1

    print(f"✅ 解析 '{char}' ({unicode_hex}): 共 {len(semantic_strokes)} 笔 Semantic Strokes, {len(final_tokens)} 个 Bezier Tokens。")
    plot_six_panel_dashboard(target_img, binary_img, skel_img, dt_map, final_tokens)

if __name__ == "__main__":
    FONT_PATH = "../alien_tensors_raw/NotoSansTamil[wdth,wght].ttf"
    UNICODE_HEX = "U+0BB4"
    run_ultimate_tokenizer(FONT_PATH, UNICODE_HEX)