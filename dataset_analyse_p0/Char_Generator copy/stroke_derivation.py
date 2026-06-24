import math
import numpy as np
from collections import defaultdict

# ==========================================
# ⚙️ 几何与归一化基础算法
# ==========================================
CANVAS_SIZE = 400.0

def get_bezier_point(pts, t):
    mt = 1 - t
    return (mt**3)*pts[0] + 3*(mt**2)*t*pts[1] + 3*mt*(t**2)*pts[2] + (t**3)*pts[3]

def normalize_and_sample_function(mother_bezier, N=50):
    """将笔画归一化到 X 轴定积分空间，用于变体 0123 比对"""
    pts = np.array(mother_bezier)
    if np.isnan(pts).any() or np.isinf(pts).any(): return None
    p0 = pts[0]
    pts_t = pts - p0
    dists = np.linalg.norm(pts_t, axis=1)
    max_idx = np.argmax(dists)
    vec = pts_t[max_idx]
    L = dists[max_idx]
    if L < 1e-5: return None
    cos_t, sin_t = vec[0]/L, vec[1]/L
    R = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    pts_r = pts_t @ R.T
    pts_norm = pts_r / L
    ts_dense = np.linspace(0, 1, 300)[:, None]
    curve_dense = get_bezier_point(pts_norm, ts_dense)
    x_dense = np.maximum.accumulate(curve_dense[:, 0])
    y_dense = curve_dense[:, 1]
    if x_dense[-1] < 1e-5: return None
    x_dense = x_dense / x_dense[-1] 
    return np.interp(np.linspace(0, 1.0, N), x_dense, y_dense)

def get_4_isomorphisms_y_only(Y):
    # 变体: 0(原样), 1(起点终点倒转), 2(上下翻转), 3(中心对称倒转)
    return [Y.copy(), -Y[::-1], -Y, Y[::-1]]

def transform_bezier(pts, rot_angle, mirror_x, mirror_y):
    """对贝塞尔坐标进行全局旋转和镜像变换"""
    c = CANVAS_SIZE / 2.0
    new_pts = np.array(pts).copy()
    if mirror_x: new_pts[:, 0] = CANVAS_SIZE - new_pts[:, 0]
    if mirror_y: new_pts[:, 1] = CANVAS_SIZE - new_pts[:, 1]
    rad = math.radians(rot_angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    for i in range(4):
        x, y = new_pts[i][0] - c, new_pts[i][1] - c
        new_pts[i][0], new_pts[i][1] = x * cos_a - y * sin_a + c, x * sin_a + y * cos_a + c
    return new_pts.tolist()

def transform_point(pt, rot_angle, mirror_x, mirror_y):
    """对单个坐标点进行同步的全局旋转和镜像变换"""
    if not pt or len(pt) < 2: return pt
    c = CANVAS_SIZE / 2.0
    x, y = pt[0], pt[1]
    
    if mirror_x: x = CANVAS_SIZE - x
    if mirror_y: y = CANVAS_SIZE - y
        
    rad = math.radians(rot_angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    nx = (x - c) * cos_a - (y - c) * sin_a + c
    ny = (x - c) * sin_a + (y - c) * cos_a + c
    
    return [nx, ny]

# ==========================================
# 🧠 运笔顺序引擎 (基于图与梯度的 DFS)
# ==========================================
class StrokeGraph:
    def __init__(self, strokes, topo_events):
        self.strokes = {s['bezier_id']: s for s in strokes}
        self.bids = list(self.strokes.keys())
        self.adj = defaultdict(list)
        self.degree = defaultdict(int)
        
        for ev in topo_events:
            t = ev.get('type')
            if t in ['E2E', 'X']:
                self._add_edge(ev.get('stroke_a'), ev.get('stroke_b'), ev)
            elif t == 'T':
                self._add_edge(ev.get('guest'), ev.get('host'), ev)
                
    def _add_edge(self, u, v, ev):
        if u in self.strokes and v in self.strokes:
            self.adj[u].append((v, ev))
            self.adj[v].append((u, ev))
            self.degree[u] += 1
            self.degree[v] += 1

    def get_gradient_vector(self, bid, point_coord):
        """计算笔画在某一交点处的切线向量方向"""
        pts = np.array(self.strokes[bid]['mother_bezier'])
        d_p0 = np.linalg.norm(pts[0] - point_coord)
        d_p3 = np.linalg.norm(pts[3] - point_coord)
        v = (pts[1] - pts[0]) if d_p0 < d_p3 else (pts[3] - pts[2])
        n = np.linalg.norm(v)
        return v / n if n > 1e-5 else np.array([1.0, 0.0])

    def generate_stroke_orders(self, max_orders=5):
        """
        规则实现：
        1. 存在关联的笔画通过并查集与 DFS 保证在顺序上相邻。
        2. 存在关联的笔画组中，开头的笔画必须为拓扑度数最小的一笔。
        3. 构成环的笔画同样通过图遍历保证相邻。
        4. 遇到分叉口 (度>=3)，通过梯度向量点乘，选择与上一笔最平滑的一笔。
        """
        visited = set()
        ccs = []
        for bid in self.bids:
            if bid not in visited:
                cc = []; queue = [bid]; visited.add(bid)
                while queue:
                    curr = queue.pop(0); cc.append(curr)
                    for nxt, _ in self.adj[curr]:
                        if nxt not in visited: visited.add(nxt); queue.append(nxt)
                ccs.append(cc)
                
        all_cc_paths = []
        for cc in ccs:
            if len(cc) == 1:
                all_cc_paths.append([[cc[0]]]); continue
            
            # 规则 2: 选择拓扑结构最少的笔画作为入口
            min_deg = min(self.degree[n] for n in cc)
            start_nodes = [n for n in cc if self.degree[n] == min_deg]
            
            cc_paths = []
            for start_node in start_nodes:
                stack = [(start_node, [start_node], None)]
                while stack:
                    curr, path, last_vec = stack.pop()
                    if len(path) == len(cc):
                        cc_paths.append(path)
                        if len(cc_paths) >= max_orders: break # 限制单个连通块内的派生爆炸
                        continue
                        
                    unvisited = [nxt for nxt, _ in self.adj[curr] if nxt not in path]
                    if not unvisited:
                        cc_paths.append(path + list(set(cc) - set(path))); continue
                        
                    # 规则 4: 分叉口平滑度引导
                    if last_vec is not None and len(unvisited) >= 2:
                        def angle_diff(nxt_tuple):
                            nxt_node, ev = nxt_tuple
                            pt = np.array(ev.get('position', [0,0]))
                            nxt_vec = self.get_gradient_vector(nxt_node, pt)
                            return -np.dot(last_vec, nxt_vec) # 点乘越大角度越小，取负号让最顺滑的排前面
                            
                        sorted_neighbors = sorted([(n, ev) for n, ev in self.adj[curr] if n in unvisited], key=angle_diff)
                        unvisited = [n for n, _ in sorted_neighbors]
                    
                    for nxt in reversed(unvisited):
                        pt = next(ev.get('position', [0,0]) for n, ev in self.adj[curr] if n == nxt)
                        exit_vec = self.get_gradient_vector(curr, pt)
                        stack.append((nxt, path + [nxt], exit_vec))
            all_cc_paths.append(cc_paths if cc_paths else [cc])

        # 拼接各个连通块
        final_orders = []
        def combine_paths(cc_idx, curr_path):
            if cc_idx == len(all_cc_paths):
                final_orders.append(curr_path)
                return
            for p in all_cc_paths[cc_idx]:
                if len(final_orders) >= max_orders: return
                combine_paths(cc_idx + 1, curr_path + p)
                
        combine_paths(0, [])
        return final_orders


# ==========================================
# 🚀 对外接口：执行样本组合派生 (含拓扑坐标同步变换)
# ==========================================
def generate_derived_sequences(strokes, topo_events, shape_dict, hex_key, cluster_refs, 
                               max_order_samples, rot_mode, mirror_mode):
    if not strokes: return []
    
    graph = StrokeGraph(strokes, topo_events)
    valid_orders = graph.generate_stroke_orders(max_orders=max_order_samples)
    stroke_map = {s["bezier_id"]: s for s in strokes}
    
    transforms = [(0, False, False)]
    if rot_mode >= 2: transforms.append((180, False, False))
    if rot_mode == 3: transforms.extend([(90, False, False), (270, False, False)])
    if mirror_mode:
        mirrored = [(r, True, False) for r, mx, my in transforms]
        transforms.extend(mirrored)
        
    results = []
    for o_idx, order in enumerate(valid_orders):
        for rot, mx, my in transforms:
            derived_strokes = []
            order_set = set(order) # 当前连通图涉及的笔画
            
            # 1. 变换贝塞尔曲线
            for bid in order:
                orig_s = stroke_map[bid]
                cid = shape_dict[hex_key][bid]
                new_bezier = transform_bezier(orig_s["mother_bezier"], rot, mx, my)
                
                new_Y = normalize_and_sample_function(new_bezier)
                variant_id = 0
                if new_Y is not None and cid in cluster_refs:
                    variants = get_4_isomorphisms_y_only(new_Y)
                    dists = [np.mean(np.abs(cluster_refs[cid] - v)) for v in variants]
                    variant_id = int(np.argmin(dists))
                    
                derived_strokes.append({
                    "bezier_id": bid,
                    "shape_token": cid,
                    "variant_id": variant_id,  # 🌟 这一行你原来就有，非常棒！
                    "mother_bezier": new_bezier,
                    "width_mean": float(np.mean(orig_s["width_bezier"]))
                })
            
            # 🌟 2. 同步变换并筛选当前子图的拓扑事件
            derived_events = []
            for ev in topo_events:
                # 只保留存在于当前连通图序列中的拓扑事件
                involved_strokes = [v for k, v in ev.items() if "stroke" in k or k in ["host", "guest"]]
                if not any(bid in order_set for bid in involved_strokes): continue
                
                new_ev = ev.copy()
                # 提取并同步旋转镜像相交点坐标 (确保你的 json 中叫 "position")
                if "position" in new_ev:
                    new_ev["position"] = transform_point(new_ev["position"], rot, mx, my)
                
                derived_events.append(new_ev)
                
            rule_name = f"order_{o_idx}_rot{rot}_mx{mx}_my{my}"
            results.append((rule_name, derived_strokes, derived_events)) # 返回扩增后的事件
            
    return results