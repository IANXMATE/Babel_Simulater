import numpy as np
import networkx as nx
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
from fontTools.ttLib import TTFont

# ==========================================
# 🎯 字体解析与渲染 (Font & Rendering)
# ==========================================
def extract_all_real_chars(font_path, canvas_size=400):
    try: 
        ttfont = TTFont(font_path)
    except Exception: 
        return [chr(c) for c in range(33, 126)]
        
    valid_chars = set()
    for table in ttfont['cmap'].tables:
        for codepoint, glyph_name in table.cmap.items():
            char = chr(codepoint)
            if char.isprintable() and not char.isspace(): 
                valid_chars.add(char)
                
    pil_font = ImageFont.truetype(font_path, int(canvas_size * 0.8))
    final_chars = [c for c in valid_chars if pil_font.getbbox(c)]
    return final_chars if final_chars else [chr(c) for c in range(33, 126)]

def render_unicode_glyph(font_path, char, canvas_size=400):
    """两步动态自适应缩放，保证大字符不越界，小标点放大"""
    test_size = 100
    try:
        test_font = ImageFont.truetype(font_path, test_size)
    except OSError:
        return np.zeros((canvas_size, canvas_size), dtype=bool)

    test_bbox = test_font.getbbox(char)
    if test_bbox is None: 
        return np.zeros((canvas_size, canvas_size), dtype=bool)
        
    test_w = test_bbox[2] - test_bbox[0]
    test_h = test_bbox[3] - test_bbox[1]
    
    max_dim = max(test_w, test_h)
    if max_dim == 0: max_dim = 1 

    safe_scale = (canvas_size * 0.8) / max_dim
    target_size = int(test_size * safe_scale)
    
    font = ImageFont.truetype(font_path, target_size)
    bbox = font.getbbox(char)
    if bbox is None:
        return np.zeros((canvas_size, canvas_size), dtype=bool)
        
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    
    offset_x = (canvas_size - w) / 2 - bbox[0]
    offset_y = (canvas_size - h) / 2 - bbox[1]
    
    draw.text((offset_x, offset_y), char, font=font, fill=0)
    return (np.array(img).astype(np.float32) / 255.0) < 0.5

# ==========================================
# 🕸️ 拓扑图清洗引擎 (Topology Graph Cleaning)
# ==========================================
def collapse_degree2_nodes(G):
    nodes = list(G.nodes())
    for n in nodes:
        if G.degree(n) == 2:
            edges = list(G.edges(n, keys=True, data=True))
            if len(edges) == 2:
                u, v1, k1, d1 = edges[0]
                _, v2, k2, d2 = edges[1]
                if v1 == v2 and k1 == k2: continue 
                p1 = d1['path'] if v1 == n else d1['path'][::-1]
                p2 = d2['path'] if v2 == n else d2['path'][::-1]
                new_path = np.vstack([p1[:-1], p2])
                G.remove_edge(n, v1, key=k1)
                G.remove_edge(n, v2, key=k2)
                G.add_edge(v1, v2, path=new_path)
                G.remove_node(n)
    return G

def prune_spurs(G, max_length=20.0, dist_threshold=5.0):
    """安全非递归版：先剪毛刺，再清洗冗余重叠边"""
    changed = True
    while changed:
        changed = False
        endpoints = [n for n, d in G.degree() if d == 1]
        for n in endpoints:
            if n not in G: continue
            edges = list(G.edges(n, keys=True, data=True))
            if not edges: continue
            u, v, k, d = edges[0]
            path = d['path']
            length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
            if length < max_length and G.degree(v if u == n else u) >= 3:
                G.remove_edge(u, v, key=k)
                G.remove_node(n)
                changed = True

    to_merge = []
    edges = list(G.edges(keys=True, data=True))
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            u1, v1, k1, d1 = edges[i]
            u2, v2, k2, d2 = edges[j]
            p1, p2 = d1['path'], d2['path']
            
            n_samples = min(len(p1), len(p2), 5)
            if n_samples < 2: continue
            
            idx1 = np.linspace(0, len(p1)-1, n_samples).astype(int)
            idx2 = np.linspace(0, len(p2)-1, n_samples).astype(int)
            dist = np.mean(np.linalg.norm(p1[idx1] - p2[idx2], axis=1))
            
            if dist < dist_threshold:
                to_merge.append((u1, v1, k1, u2, v2, k2))
                break 

    for u1, v1, k1, u2, v2, k2 in to_merge:
        if G.has_edge(u1, v1, k1) and G.has_edge(u2, v2, k2):
            p1 = G[u1][v1][k1]['path']
            p2 = G[u2][v2][k2]['path']
            new_path = stitch_paths([p1, p2])
            G.add_edge(u1, v2, key=max(k1, k2)+1000, path=new_path)
            G.remove_edge(u1, v1, k1)
            G.remove_edge(u2, v2, k2)
    return G

def stitch_paths(paths):
    """方向感知物理缝合"""
    if not paths: return np.array([])
    if len(paths) == 1: return paths[0]
    stitched = paths[0].copy()
    remaining = paths[1:]
    
    while remaining:
        best_dist = float('inf')
        best_idx, best_orient = -1, -1
        s_head, s_tail = stitched[0], stitched[-1]
        
        for i, p in enumerate(remaining):
            p_head, p_tail = p[0], p[-1]
            dists = [
                (np.linalg.norm(s_tail - p_head), 0), 
                (np.linalg.norm(s_tail - p_tail), 1), 
                (np.linalg.norm(s_head - p_head), 2), 
                (np.linalg.norm(s_head - p_tail), 3)  
            ]
            for d, orient in dists:
                if d < best_dist: 
                    best_dist, best_idx, best_orient = d, i, orient
                    
        best_p = remaining.pop(best_idx)
        if best_orient == 0: stitched = np.vstack([stitched, best_p])
        elif best_orient == 1: stitched = np.vstack([stitched, best_p[::-1]])
        elif best_orient == 2: stitched = np.vstack([best_p[::-1], stitched])
        elif best_orient == 3: stitched = np.vstack([best_p, stitched])
    return stitched

# ==========================================
# 📐 贝塞尔数学与回归拟合 (Bezier Regression)
# ==========================================
def cubic_bezier_np(P, t):
    mt = 1 - t
    return mt**3 * P[0] + 3*mt**2*t * P[1] + 3*mt*t**2 * P[2] + t**3 * P[3]

def fit_bezier_basic_with_error(pixel_path):
    if len(pixel_path) < 2: return None, float('inf')
    P0, P3 = pixel_path[0], pixel_path[-1]
    t_vals = np.linspace(0, 1, len(pixel_path))[:, np.newaxis]
    
    def residuals(controls): 
        return (cubic_bezier_np(np.array([P0, controls[0:2], controls[2:4], P3]), t_vals) - pixel_path).flatten()
        
    res = least_squares(residuals, x0=np.concatenate([P0 + (P3-P0)*0.33, P0 + (P3-P0)*0.66]))
    P_opt = np.array([P0, res.x[0:2], res.x[2:4], P3])
    error = np.mean(np.linalg.norm(cubic_bezier_np(P_opt, t_vals) - pixel_path, axis=1))
    return P_opt, error

def split_pixel_path_adaptively(pixel_path, max_error=2.0):
    num_pts = len(pixel_path)
    if num_pts < 4: return [pixel_path]
    _, error = fit_bezier_basic_with_error(pixel_path)
    if error > max_error and num_pts > 10:
        mid = num_pts // 2
        return split_pixel_path_adaptively(pixel_path[:mid+1], max_error) + split_pixel_path_adaptively(pixel_path[mid:], max_error)
    return [pixel_path]

def regress_width_dt_fast(mother_bezier, dt_map):
    t_vals = np.linspace(0, 1, 30)[:, np.newaxis]
    m_pts = cubic_bezier_np(mother_bezier, t_vals)
    coords = np.vstack([m_pts[:, 1], m_pts[:, 0]])
    local_radii = np.maximum(map_coordinates(dt_map, coords, order=1), 0.1)
    
    def width_residuals(W):
        mt, t = 1 - t_vals.flatten(), t_vals.flatten()
        return (mt**3*W[0] + 3*mt**2*t*W[1] + 3*mt*t**2*W[2] + t**3*W[3]) - local_radii
        
    res = least_squares(width_residuals, x0=np.array([4.0, 4.0, 4.0, 4.0]))
    return np.abs(res.x).tolist()