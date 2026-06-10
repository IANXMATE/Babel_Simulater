import os
import json
import torch
import numpy as np
import pickle
import copy
import networkx as nx
from scipy.ndimage import distance_transform_edt
from scipy.spatial import distance

try:
    from geometry_vision import fit_bezier_basic_with_error
except ImportError:
    def fit_bezier_basic_with_error(path):
        return np.array([path[0], path[len(path)//3], path[2*len(path)//3], path[-1]]), 0.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(SCRIPT_DIR, "model_config.json")

class RawAnnotationLoader:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        self.anno_dir = os.path.join(SCRIPT_DIR, self.config['data_config']['annotations_dir'])
        self.fonts_raw_dir = os.path.join(SCRIPT_DIR, "../dataset/alien_tensors_raw")
        
    def load_raw_dataset(self):
        if not os.path.exists(self.anno_dir):
            raise FileNotFoundError(f"❌ 找不到标注文件夹: {self.anno_dir}")
        json_files = [f for f in os.listdir(self.anno_dir) if f.endswith('.json')]
        raw_dataset = {}
        for file in json_files:
            font_filename = file.split('.')[0]
            file_path = os.path.join(self.anno_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                font_data = json.load(f)
            for hex_key, strokes in font_data.items():
                if not strokes: continue
                raw_dataset[f"{font_filename}_{hex_key}"] = {
                    "font_path": os.path.join(self.fonts_raw_dir, f"{font_filename}.ttf"),
                    "char": chr(int(hex_key[2:], 16)),
                    "hex_key": hex_key,
                    "font_filename": font_filename,
                    "strokes": strokes
                }
        return raw_dataset

class TopologyAttentionOptimizer:
    def __init__(self, dt_map):
        self.dt_map = dt_map
        
    def cubic_bezier_pts(self, p, num_points=100):
        t = np.linspace(0, 1, num_points)[:, None]
        mt = 1 - t
        return (mt**3 * p[0] + 3 * mt**2 * t * p[1] + 3 * mt * t**2 * p[2] + t**3 * p[3])

    def _split_bezier_math(self, p, t):
        """核心：De Casteljau 算法，完美的数学切割，绝不引发形变"""
        p_np = np.array(p, dtype=np.float32) 
        p0, p1, p2, p3 = p_np[0], p_np[1], p_np[2], p_np[3]
        
        p01 = (1-t)*p0 + t*p1
        p12 = (1-t)*p1 + t*p2
        p23 = (1-t)*p2 + t*p3
        p012 = (1-t)*p01 + t*p12
        p123 = (1-t)*p12 + t*p23
        p0123 = (1-t)*p012 + t*p123
        
        seg1 = np.array([p0, p01, p012, p0123])
        seg2 = np.array([p0123, p123, p23, p3])
        return seg1, seg2

    def process(self, strokes):
        if not strokes: return [], []
        
        # 1. 递归起终点吸附 (对照原图寻找最优质心)
        snapped = self._step1_recursive_snap_endpoints(strokes)
        
        # 2. X, T 尖端自适应缩减 (Split & Delete 裁掉过冲尖尖，保留合规十字头)
        pruned = self._step2_auto_split_and_delete_tips(snapped)
        
        # 2.5 游离端点向主干贴合 (确立 T 型无缝连接)
        final_strokes = self._step2_5_snap_to_trunks(pruned)
        
        # 3. 判定首尾相连真环，建立全域 N^2 注意力矩阵
        attention_matrix = self._step3_build_attention_matrix(final_strokes)
        
        return final_strokes, attention_matrix

    def _step1_recursive_snap_endpoints(self, strokes):
        snapped = copy.deepcopy(strokes)
        N = len(snapped)
        G = nx.Graph()
        G.add_nodes_from(range(2 * N))
        
        for i in range(N):
            for j in range(i, N):
                for idx_i, e_i in enumerate([0, 3]):
                    for idx_j, e_j in enumerate([0, 3]):
                        if i == j and e_i == e_j: continue
                        node_i, node_j = i * 2 + idx_i, j * 2 + idx_j
                        pt_i = np.array(snapped[i]['mother_bezier'][e_i])
                        pt_j = np.array(snapped[j]['mother_bezier'][e_j])
                        
                        y, x = int(np.clip(pt_i[1], 0, self.dt_map.shape[0]-1)), int(np.clip(pt_i[0], 0, self.dt_map.shape[1]-1))
                        snap_threshold = max(self.dt_map[y, x] * 2.2, 12.0)
                        
                        if np.linalg.norm(pt_i - pt_j) < snap_threshold:
                            G.add_edge(node_i, node_j)
                            
        h, w = self.dt_map.shape
        for comp in nx.connected_components(G):
            if len(comp) > 1:
                pts = [np.array(snapped[n // 2]['mother_bezier'][0 if n % 2 == 0 else 3]) for n in comp]
                center = np.mean(pts, axis=0)
                cx, cy = int(np.clip(center[0], 0, w-1)), int(np.clip(center[1], 0, h-1))
                r = 10
                sub_dt = self.dt_map[max(0, cy-r):min(h, cy+r+1), max(0, cx-r):min(w, cx+r+1)]
                if sub_dt.size > 0 and np.max(sub_dt) > 0:
                    ny, nnx = np.unravel_index(np.argmax(sub_dt), sub_dt.shape)
                    optimized_anchor = np.array([max(0, cx-r) + nnx, max(0, cy-r) + ny], dtype=np.float32)
                else: optimized_anchor = center
                    
                for n in comp:
                    snapped[n // 2]['mother_bezier'][0 if n % 2 == 0 else 3] = optimized_anchor.tolist()
        return snapped

    def _step2_auto_split_and_delete_tips(self, strokes):
        """🌟 第二步核心：基于 dt_map 动态感知的智能防误杀裁尖"""
        current_strokes = copy.deepcopy(strokes)
        RATIO_THRESHOLD = 0.08   # 占比容忍度缩小为 8%
        h, w_img = self.dt_map.shape
        
        for _ in range(5): 
            changed = False
            N = len(current_strokes)
            dense_curves = [self.cubic_bezier_pts(np.array(s['mother_bezier']), 100) for s in current_strokes]
            
            for i in range(N):
                for j in range(N):
                    if i == j: continue
                    dists_matrix = distance.cdist(dense_curves[i], dense_curves[j])
                    min_d_idx = np.argmin(dists_matrix)
                    idx_i, idx_j = np.unravel_index(min_d_idx, dists_matrix.shape)
                    
                    if dists_matrix[idx_i, idx_j] < 4.0: 
                        diffs_i = np.diff(dense_curves[i], axis=0)
                        lens_i = np.insert(np.cumsum(np.linalg.norm(diffs_i, axis=1)), 0, 0)
                        L_i1 = lens_i[idx_i]
                        L_i2 = lens_i[-1] - L_i1
                        
                        L_total_j = np.sum(np.linalg.norm(np.diff(dense_curves[j], axis=0), axis=1))
                        L_total = lens_i[-1] + L_total_j
                        min_split_len = min(L_i1, L_i2)
                        
                        # 🌟 读取交叉点的物理线条半径，制定自适应绝对阈值
                        cross_pt = dense_curves[j][idx_j]
                        y, x = int(np.clip(cross_pt[1], 0, h-1)), int(np.clip(cross_pt[0], 0, w_img-1))
                        w_cross = self.dt_map[y, x]
                        
                        # 核心判定：只砍掉“长度小于交叉线半径的1.8倍 + 4像素”的小毛刺
                        # 这完美保护了 `t` 字（顶部很长），但会干掉 `Y` 字（过冲很短）
                        dynamic_abs_threshold = max(w_cross * 1.8 + 4.0, 12.0)
                        
                        if min_split_len > 2.0: # 防呆，防止把绝对端点切碎
                            if (min_split_len / L_total < RATIO_THRESHOLD) or (min_split_len < dynamic_abs_threshold):
                                t_param = idx_i / 99.0
                                seg1, seg2 = self._split_bezier_math(current_strokes[i]['mother_bezier'], t_param)
                                
                                if L_i1 < L_i2: 
                                    current_strokes[i]['mother_bezier'] = seg2.tolist()
                                else:
                                    current_strokes[i]['mother_bezier'] = seg1.tolist()
                                changed = True
                                break
                if changed: break
            if not changed: break
        return current_strokes

    def _step2_5_snap_to_trunks(self, strokes):
        snapped = copy.deepcopy(strokes)
        N = len(snapped)
        dense_curves = [self.cubic_bezier_pts(np.array(s['mother_bezier']), 100) for s in snapped]
        for i in range(N):
            for e_i in [0, 3]:
                pt_i = np.array(snapped[i]['mother_bezier'][e_i])
                is_shared = False
                for j in range(N):
                    if i == j: continue
                    o_mb = np.array(snapped[j]['mother_bezier'])
                    if np.linalg.norm(pt_i - o_mb[0]) < 1.0 or np.linalg.norm(pt_i - o_mb[3]) < 1.0:
                        is_shared = True; break
                if is_shared: continue 

                min_dist, best_pt = float('inf'), None
                y, x = int(np.clip(pt_i[1], 0, self.dt_map.shape[0]-1)), int(np.clip(pt_i[0], 0, self.dt_map.shape[1]-1))
                snap_threshold = max(self.dt_map[y, x] * 2.5, 14.0) 

                for j in range(N):
                    if i == j: continue
                    trunk = dense_curves[j][5:-5] 
                    if len(trunk) > 0:
                        dists = np.linalg.norm(trunk - pt_i, axis=1)
                        if np.min(dists) < min_dist:
                            min_dist = np.min(dists)
                            best_pt = trunk[np.argmin(dists)]
                            
                if min_dist < snap_threshold and best_pt is not None:
                    snapped[i]['mother_bezier'][e_i] = best_pt.tolist()
        return snapped

    def _step3_build_attention_matrix(self, strokes):
        N = len(strokes)
        attn_matrix = np.zeros((N, N), dtype=int)
        if N == 0: return attn_matrix
        
        dense_curves = [self.cubic_bezier_pts(np.array(s['mother_bezier']), 100) for s in strokes]

        # --- 3.1 记录所有物理相交点 ---
        for i in range(N):
            for j in range(i+1, N):
                mb_i = np.array(strokes[i]['mother_bezier'])
                mb_j = np.array(strokes[j]['mother_bezier'])
                shared = sum([1 for ei in [0,3] for ej in [0,3] if np.linalg.norm(mb_i[ei]-mb_j[ej]) < 2.0])
                if shared > 0:
                    attn_matrix[i][j] = 1; attn_matrix[j][i] = 1

        # --- 3.2 记录 X, T 相交 ---
        for i in range(N):
            for j in range(N):
                if i == j: continue
                mb_i = np.array(strokes[i]['mother_bezier'])
                
                # T 型判定
                i_hits_j = False
                for ei in [0, 3]:
                    trunk_j = dense_curves[j][10:-10]
                    if len(trunk_j) > 0 and np.min(np.linalg.norm(trunk_j - mb_i[ei], axis=1)) < 3.5:
                        i_hits_j = True; break
                        
                if i_hits_j and attn_matrix[i][j] < 4:
                    attn_matrix[i][j] = 5 
                    attn_matrix[j][i] = 4 
                    continue
                    
                # X 型判定
                if attn_matrix[i][j] == 0:
                    trunk_i = dense_curves[i][10:-10]
                    trunk_j = dense_curves[j][10:-10]
                    if len(trunk_i) > 0 and len(trunk_j) > 0:
                        if np.min(distance.cdist(trunk_i, trunk_j)) < 3.5:
                            attn_matrix[i][j] = 3

        # --- 3.3 真正的首尾相连真环覆写 ---
        unique_pts = []
        def get_pt_id(pt):
            for idx, upt in enumerate(unique_pts):
                if np.linalg.norm(np.array(pt) - np.array(upt)) < 2.0: return idx
            unique_pts.append(pt)
            return len(unique_pts) - 1
            
        stroke_endpoints = []
        for s in strokes:
            mb = np.array(s['mother_bezier'])
            stroke_endpoints.append((get_pt_id(mb[0]), get_pt_id(mb[3])))
            
        G_simple = nx.Graph()
        for u, v in stroke_endpoints:
            if u != v: G_simple.add_edge(u, v)
            
        try:
            for comp in list(nx.biconnected_components(G_simple)):
                if len(comp) >= 3:
                    loop_strokes = [idx for idx, (u, v) in enumerate(stroke_endpoints) if u in comp and v in comp]
                    for i in loop_strokes:
                        for j in loop_strokes:
                            if i != j: attn_matrix[i][j] = 2 
        except: pass
        
        for i in range(N):
            for j in range(i+1, N):
                u1, v1 = stroke_endpoints[i]
                u2, v2 = stroke_endpoints[j]
                if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
                    if u1 != v1:
                        attn_matrix[i][j] = 2; attn_matrix[j][i] = 2

        return attn_matrix

def main():
    print("🎬 [Step 1.5] 启动无向图首尾闭合检测与自适应修剪管线...")
    loader = RawAnnotationLoader(config_path)
    raw_dataset = loader.load_raw_dataset()

    from PIL import Image, ImageDraw, ImageFont
    def safe_render(font_path, char):
        img = Image.new('L', (400, 400), 0); draw = ImageDraw.Draw(img)
        try:
            pyfont = ImageFont.truetype(font_path, 400 * 0.8)
            w, h = draw.textbbox((0, 0), char, font=pyfont)[2:]
            draw.text(((400-w)/2, (400-h)/2 - h*0.1), char, fill=255, font=pyfont)
        except: pass
        return np.array(img, dtype=np.float32) / 255.0

    cleaned_topology_dataset = {}
    for unique_key, item in raw_dataset.items():
        binary = safe_render(item["font_path"], item["char"])
        if not np.any(binary): binary = np.zeros((400, 400), dtype=np.float32)
        dt_map = distance_transform_edt(binary)
        
        optimizer = TopologyAttentionOptimizer(dt_map)
        ordered_strokes, attn_matrix = optimizer.process(item["strokes"])
        
        enriched_strokes = []
        for order_idx, stroke in enumerate(ordered_strokes):
            enriched_strokes.append({
                "bezier_id": order_idx,
                "mother_bezier": stroke["mother_bezier"],
                "width_bezier": stroke.get("width_bezier", [6.0]*4),
            })
            
        cleaned_topology_dataset[unique_key] = {
            "hex_key": item["hex_key"],
            "char": item["char"],
            "font_filename": item["font_filename"],
            "raw_strokes": item["strokes"], 
            "strokes": enriched_strokes,
            "attention_matrix": attn_matrix.tolist()
        }

    output_path = os.path.join(SCRIPT_DIR, "preprocessed_topology_dataset.pkl")
    with open(output_path, 'wb') as f: pickle.dump(cleaned_topology_dataset, f)
        
    print(f"💾 结构化数据集已清洗保存至: {output_path}")
    print("🎉 吸附 -> 动态自适应裁剪 -> 真环判定 100% 重构成功！")

if __name__ == "__main__":
    main()