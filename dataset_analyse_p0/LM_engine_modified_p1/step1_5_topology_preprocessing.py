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
        valid_widths = dt_map[dt_map > 0]
        self.global_base_width = np.median(valid_widths) if len(valid_widths) > 0 else 5.0
        
    def cubic_bezier_pts(self, p, num_points=100):
        t = np.linspace(0, 1, num_points)[:, None]
        mt = 1 - t
        return (mt**3 * p[0] + 3 * mt**2 * t * p[1] + 3 * mt * t**2 * p[2] + t**3 * p[3])

    def _split_bezier_math(self, p, t):
        p_np = np.array(p, dtype=np.float32) 
        p0, p1, p2, p3 = p_np[0], p_np[1], p_np[2], p_np[3]
        p01 = (1-t)*p0 + t*p1
        p12 = (1-t)*p1 + t*p2
        p23 = (1-t)*p2 + t*p3
        p012 = (1-t)*p01 + t*p12
        p123 = (1-t)*p12 + t*p23
        p0123 = (1-t)*p012 + t*p123
        return np.array([p0, p01, p012, p0123]), np.array([p0123, p123, p23, p3])

    def _is_converged(self, old_strokes, new_strokes):
        if len(old_strokes) != len(new_strokes): return False
        for s1, s2 in zip(old_strokes, new_strokes):
            diff = np.max(np.abs(np.array(s1['mother_bezier']) - np.array(s2['mother_bezier'])))
            if diff > 0.5: return False
        return True

    def process(self, strokes):
        if not strokes: return [], []
        current_strokes = copy.deepcopy(strokes)
        
        for macro_iter in range(4):
            old_strokes = copy.deepcopy(current_strokes)
            
            # 1. 严格质心吸附 (无形变)
            current_strokes = self._step1_strict_endpoint_snap(current_strokes)
            # 2. 尖端修剪
            current_strokes = self._step2_auto_split_and_delete_tips(current_strokes)
            # 3. 互斥锁主干贴合 (T型确立)
            current_strokes = self._step2_5_snap_to_trunks_with_lock(current_strokes)
            
            if self._is_converged(old_strokes, current_strokes): break
                
        attention_matrix = self._step3_build_attention_matrix(current_strokes)
        return current_strokes, attention_matrix

    def _step1_strict_endpoint_snap(self, strokes):
        """🌟 修复点 1 & 2：抛弃黑洞漂移，严格约束网络聚类的质心"""
        snapped = copy.deepcopy(strokes)
        N = len(snapped)
        G = nx.Graph()
        G.add_nodes_from(range(2 * N))
        
        # 构建极其严格的相近端点图
        for i in range(N):
            for j in range(i, N):
                for idx_i, e_i in enumerate([0, 3]):
                    for idx_j, e_j in enumerate([0, 3]):
                        if i == j and e_i == e_j: continue
                        node_i, node_j = i * 2 + idx_i, j * 2 + idx_j
                        pt_i = np.array(snapped[i]['mother_bezier'][e_i])
                        pt_j = np.array(snapped[j]['mother_bezier'][e_j])
                        
                        y, x = int(np.clip(pt_i[1], 0, self.dt_map.shape[0]-1)), int(np.clip(pt_i[0], 0, self.dt_map.shape[1]-1))
                        # 收紧阈值，防止胡乱链式吸附 (最大不超过 8 像素)
                        snap_threshold = min(max(self.dt_map[y, x] * 1.5, 4.0), 8.0) 
                        
                        if np.linalg.norm(pt_i - pt_j) < snap_threshold:
                            G.add_edge(node_i, node_j)
                            
        for comp in nx.connected_components(G):
            if len(comp) > 1:
                pts = [np.array(snapped[n // 2]['mother_bezier'][0 if n % 2 == 0 else 3]) for n in comp]
                # 🌟 直接取最纯净的几何中心，禁止再去 DT_map 里瞎找最大值导致形变！
                centroid = np.mean(pts, axis=0) 
                
                for n in comp:
                    s_idx = n // 2
                    e_idx = 0 if n % 2 == 0 else 3
                    orig_pt = np.array(snapped[s_idx]['mother_bezier'][e_idx])
                    
                    # 🌟 绝对漂移限制 (Max Drift)：如果偏离原点超过 6.0 像素，则限制移动
                    drift_vec = centroid - orig_pt
                    drift_dist = np.linalg.norm(drift_vec)
                    if drift_dist > 6.0:
                        safe_pt = orig_pt + (drift_vec / drift_dist) * 6.0
                    else:
                        safe_pt = centroid
                        
                    snapped[s_idx]['mother_bezier'][e_idx] = safe_pt.tolist()
        return snapped

    def _step2_auto_split_and_delete_tips(self, strokes):
        current_strokes = copy.deepcopy(strokes)
        RATIO_THRESHOLD = 0.15   
        for _ in range(3): 
            changed = False
            N = len(current_strokes)
            dense_curves = [self.cubic_bezier_pts(np.array(s['mother_bezier']), 150) for s in current_strokes]
            for i in range(N):
                for j in range(N):
                    if i == j: continue
                    dists_matrix = distance.cdist(dense_curves[i], dense_curves[j])
                    min_d_idx = np.argmin(dists_matrix)
                    idx_i, idx_j = np.unravel_index(min_d_idx, dists_matrix.shape)
                    
                    cross_pt = dense_curves[j][idx_j]
                    y, x = int(np.clip(cross_pt[1], 0, self.dt_map.shape[0]-1)), int(np.clip(cross_pt[0], 0, self.dt_map.shape[1]-1))
                    w_cross = self.dt_map[y, x]
                    dynamic_abs_threshold = max(w_cross * 1.5, self.global_base_width * 1.5, 8.0)
                    
                    if dists_matrix[idx_i, idx_j] < max(w_cross, 3.0): 
                        diffs_i = np.diff(dense_curves[i], axis=0)
                        dists_i = np.linalg.norm(diffs_i, axis=1)
                        cum_dists = np.insert(np.cumsum(dists_i), 0, 0)
                        L_total = cum_dists[-1]
                        if L_total <= 0: continue
                        
                        L_i1, L_i2 = cum_dists[idx_i], L_total - cum_dists[idx_i]
                        min_split_len = min(L_i1, L_i2)
                        
                        if (min_split_len / L_total < RATIO_THRESHOLD) or (min_split_len < dynamic_abs_threshold):
                            if min_split_len > 2.0: 
                                t_param = L_i1 / L_total
                                seg1, seg2 = self._split_bezier_math(current_strokes[i]['mother_bezier'], t_param)
                                if L_i1 < L_i2: current_strokes[i]['mother_bezier'] = seg2.tolist()
                                else: current_strokes[i]['mother_bezier'] = seg1.tolist()
                                changed = True
                                break
                if changed: break
            if not changed: break
        return current_strokes

    def _step2_5_snap_to_trunks_with_lock(self, strokes):
        """🌟 修复点 3：带端点状态互斥锁的主干贴合（划清 1 和 5 的界限）"""
        snapped = copy.deepcopy(strokes)
        N = len(snapped)
        dense_curves = [self.cubic_bezier_pts(np.array(s['mother_bezier']), 100) for s in snapped]
        
        for i in range(N):
            for e_i in [0, 3]:
                pt_i = np.array(snapped[i]['mother_bezier'][e_i])
                
                # 🌟 状态互斥锁 (Status Lock)
                # 检查该端点是否已经和【其他笔画的端点】完美贴合（误差<1.0）
                is_endpoint_locked = False
                for j in range(N):
                    if i == j: continue
                    o_mb = np.array(snapped[j]['mother_bezier'])
                    if np.linalg.norm(pt_i - o_mb[0]) < 1.0 or np.linalg.norm(pt_i - o_mb[3]) < 1.0:
                        is_endpoint_locked = True; break
                        
                # 如果已经是端点相接关系，绝对禁止它再去贴合主干！直接跳过！
                if is_endpoint_locked: continue 

                min_dist, best_pt = float('inf'), None
                # 收紧主干贴合的判定阈值，防止胡乱拉扯
                snap_threshold = min(self.global_base_width * 2.0, 8.0) 

                for j in range(N):
                    if i == j: continue
                    total_len = len(dense_curves[j])
                    # 严格掐头去尾，防止把另一个端点误认为主干
                    exclude_pts = max(5, int(total_len * 0.10))
                    trunk = dense_curves[j][exclude_pts:-exclude_pts]
                    
                    if len(trunk) > 0:
                        dists = np.linalg.norm(trunk - pt_i, axis=1)
                        if np.min(dists) < min_dist:
                            min_dist = np.min(dists)
                            best_pt = trunk[np.argmin(dists)]
                            
                if min_dist < snap_threshold and best_pt is not None:
                    snapped[i]['mother_bezier'][e_i] = best_pt.tolist()
        return snapped

    def _step3_build_attention_matrix(self, strokes):
        """绝对无歧义矩阵构建"""
        N = len(strokes)
        attn_matrix = np.zeros((N, N), dtype=int)
        if N == 0: return attn_matrix
        
        dense_curves = [self.cubic_bezier_pts(np.array(s['mother_bezier']), 100) for s in strokes]

        # 3.1 端点相交优先确立 (最高优先级，容差 < 1.0)
        for i in range(N):
            for j in range(i+1, N):
                mb_i = np.array(strokes[i]['mother_bezier'])
                mb_j = np.array(strokes[j]['mother_bezier'])
                shared = sum([1 for ei in [0,3] for ej in [0,3] if np.linalg.norm(mb_i[ei]-mb_j[ej]) < 1.5])
                if shared > 0:
                    attn_matrix[i][j] = 1; attn_matrix[j][i] = 1

        # 3.2 T型和X型扫描 (互斥检查)
        for i in range(N):
            for j in range(N):
                if i == j: continue
                # 如果已经是端点相交(1)，则绝对不可能是 T 或 X
                if attn_matrix[i][j] == 1: continue 
                
                mb_i = np.array(strokes[i]['mother_bezier'])
                exclude_pts_j = max(5, int(len(dense_curves[j]) * 0.1))
                trunk_j = dense_curves[j][exclude_pts_j:-exclude_pts_j]
                
                # T 型 (i 顶 j)
                i_hits_j = False
                for ei in [0, 3]:
                    if len(trunk_j) > 0 and np.min(np.linalg.norm(trunk_j - mb_i[ei], axis=1)) < 2.0:
                        i_hits_j = True; break
                        
                if i_hits_j and attn_matrix[i][j] == 0:
                    attn_matrix[i][j] = 5 
                    attn_matrix[j][i] = 4 
                    continue
                    
                # X 型
                if attn_matrix[i][j] == 0:
                    exclude_pts_i = max(5, int(len(dense_curves[i]) * 0.1))
                    trunk_i = dense_curves[i][exclude_pts_i:-exclude_pts_i]
                    if len(trunk_i) > 0 and len(trunk_j) > 0:
                        if np.min(distance.cdist(trunk_i, trunk_j)) < 3.0:
                            attn_matrix[i][j] = 3
                            
        # 3.3 首尾相连真环覆写 (只检查由端点1构成的网络)
        unique_pts = []
        def get_pt_id(pt):
            for idx, upt in enumerate(unique_pts):
                if np.linalg.norm(np.array(pt) - np.array(upt)) < 1.5: return idx
            unique_pts.append(pt)
            return len(unique_pts) - 1
            
        stroke_endpoints = []
        for s in strokes:
            mb = np.array(s['mother_bezier'])
            stroke_endpoints.append((get_pt_id(mb[0]), get_pt_id(mb[3])))
            
        G_simple = nx.MultiGraph()
        for edge_idx, (u, v) in enumerate(stroke_endpoints):
            if u != v: G_simple.add_edge(u, v, key=edge_idx)
            
        cycles = nx.cycle_basis(nx.Graph(G_simple))
        for cycle_nodes in cycles:
            loop_strokes = set()
            for k in range(len(cycle_nodes)):
                u = cycle_nodes[k]
                v = cycle_nodes[(k+1) % len(cycle_nodes)]
                for edge_idx, (su, sv) in enumerate(stroke_endpoints):
                    if (u == su and v == sv) or (u == sv and v == su):
                        loop_strokes.add(edge_idx)
            for i in loop_strokes:
                for j in loop_strokes:
                    if i != j: attn_matrix[i][j] = 2

        # 补齐两线 O 型环
        for i in range(N):
            for j in range(i+1, N):
                u1, v1 = stroke_endpoints[i]
                u2, v2 = stroke_endpoints[j]
                if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
                    if u1 != v1:
                        attn_matrix[i][j] = 2; attn_matrix[j][i] = 2

        return attn_matrix

def main():
    print("🎬 [Step 1.5] 启动防形变、带互斥锁的终极结构化清洗管线...")
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
    print("🎉 防形变漂移、端点状态互斥锁 100% 实装成功！")

if __name__ == "__main__":
    main()