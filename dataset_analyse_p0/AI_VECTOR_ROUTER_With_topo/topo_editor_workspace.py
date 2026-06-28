import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea, QFrame, QMessageBox, QTextBrowser
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# 导入基础库中的贝塞尔渲染函数
from geometry_vision import cubic_bezier_np, regress_width_dt_fast

# ==========================================
# 📐 数学几何辅助函数
# ==========================================
def get_bezier_derivative(pts, t):
    """计算贝塞尔曲线在参数 t 处的切线导数向量"""
    mt = 1 - t
    d = 3*mt**2*(pts[1]-pts[0]) + 6*mt*t*(pts[2]-pts[1]) + 3*t**2*(pts[3]-pts[2])
    return d

def get_angle(v1, v2):
    """计算两向量之间的夹角 (度数)"""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-5 or n2 < 1e-5: return 0.0
    cos_th = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_th)))

def get_polygon_orientation(pts):
    """计算多边形的旋向 (基于多边形符号面积)"""
    area = 0.0
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        area += (pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1])
    return "ccw" if area > 0 else "cw"


# ==========================================
# 🌟 稳定拓扑检测参数
# ==========================================
# 原代码使用 50 个采样点 + 最近采样点距离判定 X，拖拽时容易因为采样点错开而时好时坏。
# 新逻辑：仍然把 Bézier 离散为 polyline，但 X 用“线段相交”判断，而不是“采样点最近距离”。
TOPO_SAMPLE_N = 120
X_ENDPOINT_MARGIN = 2
SEG_EPS = 1e-8


def _cross2d(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def segment_intersection(p0, p1, q0, q1, eps=SEG_EPS):
    """
    判断线段 p0-p1 与 q0-q1 是否相交。
    返回:
        None
        或 (point, local_t_on_p, local_t_on_q)
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)

    r = p1 - p0
    s = q1 - q0
    denom = _cross2d(r, s)

    # 平行/近似平行时不当作 X。E2E/T 已经处理端点接触。
    if abs(denom) < eps:
        return None

    qp = q0 - p0
    t = _cross2d(qp, s) / denom
    u = _cross2d(qp, r) / denom

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        pt = p0 + t * r
        return pt, float(t), float(u)

    return None


def find_polyline_x_intersection(c1, c2, endpoint_margin=X_ENDPOINT_MARGIN):
    """
    在两条离散 polyline 之间寻找真正的线段交点。
    返回:
        hit, point, t1, t2

    t1/t2 是原 Bézier 参数的近似值。
    endpoint_margin 用来避免靠近端点的接触被误标为 X；
    端点附近应该优先由 E2E/T 逻辑处理。
    """
    n1 = len(c1)
    n2 = len(c2)
    if n1 < 2 or n2 < 2:
        return False, None, None, None

    i_start = max(0, endpoint_margin)
    i_end = max(i_start, n1 - 1 - endpoint_margin)
    j_start = max(0, endpoint_margin)
    j_end = max(j_start, n2 - 1 - endpoint_margin)

    best = None

    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            hit = segment_intersection(c1[i], c1[i + 1], c2[j], c2[j + 1])
            if hit is None:
                continue

            pt, lt1, lt2 = hit
            t1 = (i + lt1) / (n1 - 1)
            t2 = (j + lt2) / (n2 - 1)

            # 只接受内部交叉，避免靠近端点的接触被当成 X
            if t1 <= 0.02 or t1 >= 0.98 or t2 <= 0.02 or t2 >= 0.98:
                continue

            best = (pt, t1, t2)
            break
        if best is not None:
            break

    if best is None:
        return False, None, None, None

    pt, t1, t2 = best
    return True, pt, float(t1), float(t2)


class TopoAnnotationWorkspace(QWidget):
    def __init__(self, main_workspace, hex_key, char, binary, dt_map, phase1_edges):
        super().__init__()
        self.main_ws = main_workspace
        self.hex_key = hex_key
        self.char = char
        self.binary = binary
        self.dt_map = dt_map
        
        self.initial_edges = copy.deepcopy(phase1_edges)
        self.edges = copy.deepcopy(phase1_edges)
        
        self.history_stack = []
        self.selected_edge_ids = []
        
        # 拖拽与参数化约束状态
        self.dragging_point = None 
        self.co_dragged_points = [] 
        self.t_constraints = []     
        self.width_cache = {}       
        
        # Layer 5: 专家模仿学习数据集 (Action Log)
        self.edit_history = []
        self.drag_start_pos = None  
        
        try: self.cmap = plt.colormaps['tab20']
        except AttributeError: self.cmap = plt.get_cmap('tab20')
        
        self.init_ui()
        self.save_state()
        self.force_recompute_all_widths() 
        self.update_canvas()
        self.update_palette()
        self.update_topology_text() # 🌟 恢复初始化时的文本渲染

    # ==========================================
    # 界面初始化
    # ==========================================
    def init_ui(self):
        layout = QHBoxLayout(self)
        
        control_panel = QVBoxLayout(); control_panel.setSpacing(10)
        title = QLabel(f"Phase 2: Topo Tune\nTarget: '{self.char}'")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #E91E63;")
        control_panel.addWidget(title)
        
        line = QFrame(); line.setFrameShape(QFrame.HLine); control_panel.addWidget(line)
        
        for btn_text, handler, color in [
            ("Undo Last Move (U)", self.action_undo, "#757575"),
            ("Reset to Phase 1 (R)", self.action_reset, "#795548"),
            ("↩️ Return to P1 Editing", self.action_return_to_phase1, "#00BCD4")
        ]:
            btn = QPushButton(btn_text)
            btn.clicked.connect(handler)
            btn.setStyleSheet(f"padding: 10px; background-color: {color}; color: white; font-weight: bold; border-radius: 4px;")
            control_panel.addWidget(btn)
            
        control_panel.addStretch()
        
        btn_complete = QPushButton("✅ FINISH TOPO (Enter)")
        btn_complete.clicked.connect(self.action_complete_topo)
        btn_complete.setStyleSheet("padding: 14px; background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px;")
        control_panel.addWidget(btn_complete)

        right_panel = QVBoxLayout()
        self.palette_container = QWidget()
        self.palette_layout = QHBoxLayout(self.palette_container)
        self.palette_layout.setContentsMargins(5, 5, 5, 5)
        self.palette_layout.setAlignment(Qt.AlignLeft)
        scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True); scroll_area.setWidget(self.palette_container)
        scroll_area.setMaximumHeight(60)
        
        self.fig = plt.figure(figsize=(15, 18.5)) 
        self.fig.patch.set_facecolor('#FFFFFF')
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01)
        gs = self.fig.add_gridspec(3, 3, height_ratios=[1.2, 1.5, 1.0], hspace=0.1, wspace=0.02)
        
        self.ax_top = self.fig.add_subplot(gs[0, 1])
        self.ax_main = self.fig.add_subplot(gs[1, :])
        self.ax_orig = self.fig.add_subplot(gs[2, 0])
        self.ax_color = self.fig.add_subplot(gs[2, 1])
        self.ax_bw = self.fig.add_subplot(gs[2, 2])
        
        for ax in [self.ax_top, self.ax_main, self.ax_orig, self.ax_color, self.ax_bw]:
            ax.set_facecolor('#FFFFFF')
            ax.set_aspect('equal')
            ax.axis('off')
            
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(1500) 
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('key_press_event', self.on_key)
        self.canvas.setFocusPolicy(Qt.StrongFocus)

        canvas_scroll = QScrollArea()
        canvas_scroll.setWidgetResizable(True); canvas_scroll.setWidget(self.canvas)
        canvas_scroll.setStyleSheet("border: none; background-color: #FFFFFF;")

        self.topo_text_box = QTextBrowser()
        self.topo_text_box.setStyleSheet("background-color: #F8F9FA; border: 1px solid #CCC; padding: 10px; font-size: 14px;")
        self.topo_text_box.setMaximumHeight(140)

        right_panel.addWidget(scroll_area)
        right_panel.addWidget(canvas_scroll, stretch=6)
        right_panel.addWidget(self.topo_text_box, stretch=1)

        layout.addLayout(control_panel, 1)
        layout.addLayout(right_panel, 5)

    def get_display_map(self): return {edge['id']: str(i + 1) for i, edge in enumerate(self.edges)}
    def get_hex_color(self, eid): c = self.cmap((eid % 20)); return f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
    def force_recompute_all_widths(self):
        for edge in self.edges: self.width_cache[edge['id']] = regress_width_dt_fast(np.array(edge['path']), self.dt_map)
    
    # ==========================================
    # 🌟 恢复：实时拓扑文本反馈引擎
    # ==========================================
    def update_topology_text(self):
        id_map = self.get_display_map()
        G = nx.Graph()
        for edge in self.edges: G.add_node(edge['id'])
            
        end_to_end, t_junctions, x_junctions = [], [], []
        connection_points = {} # 🌟 新增：记录两根线之间所有的交点坐标

        for i, e1 in enumerate(self.edges):
            for j, e2 in enumerate(self.edges):
                if i >= j: continue
                p1, p2 = np.array(e1['path']), np.array(e2['path'])
                topo_ts = np.linspace(0, 1, TOPO_SAMPLE_N)[:, None]
                c1 = cubic_bezier_np(p1, topo_ts)
                c2 = cubic_bezier_np(p2, topo_ts)

                is_e2e, is_x = False, False
                t_relations = []
                pts = [] # 记录 e1 和 e2 之间的所有碰撞点

                # 1. 端点对接
                for pt1_idx in [0, 3]:
                    for pt2_idx in [0, 3]:
                        if np.linalg.norm(p1[pt1_idx] - p2[pt2_idx]) < 2.0: 
                            is_e2e = True
                            pts.append(p1[pt1_idx])

                # 2. T型搭接 (主客体判定)
                if not is_e2e:
                    for pt1_idx in [0, 3]:
                        dists = np.linalg.norm(c2 - p1[pt1_idx], axis=1)
                        if np.min(dists) < 2.0: 
                            t_relations.append((e1['id'], e2['id']))
                            pts.append(p1[pt1_idx])
                    for pt2_idx in [0, 3]:
                        dists = np.linalg.norm(c1 - p2[pt2_idx], axis=1)
                        if np.min(dists) < 2.0: 
                            t_relations.append((e2['id'], e1['id']))
                            pts.append(p2[pt2_idx])

                # 3. X型交叉
                # 原逻辑：50 个采样点之间最近距离 < 2.0，容易因为交点落在采样间隙而时好时坏。
                # 新逻辑：polyline 线段相交，判断真正的中心线交叉。
                if not is_e2e and not t_relations:
                    hit_x, x_pt, x_t1, x_t2 = find_polyline_x_intersection(c1, c2)
                    if hit_x:
                        is_x = True
                        pts.append(x_pt)

                if is_e2e or t_relations or is_x: 
                    u, v = min(e1['id'], e2['id']), max(e1['id'], e2['id'])
                    connection_points[(u, v)] = pts
                    G.add_edge(u, v)

                def _span(eid): return f"<span style='color:{self.get_hex_color(eid)}; font-weight:bold;'>{id_map[eid]}</span>"

                if is_e2e: end_to_end.append(f"{_span(e1['id'])}-{_span(e2['id'])}")
                elif t_relations:
                    for guest_id, host_id in list(set(t_relations)):
                        t_junctions.append(f"{_span(guest_id)} 搭在 {_span(host_id)} 上")
                elif is_x: x_junctions.append(f"{_span(e1['id'])} 交叉 {_span(e2['id'])}")

        html_lines = ["<b style='color:#333; font-size:14px;'>📊 实时拓扑状态反馈</b><br>"]
        if end_to_end: html_lines.append(f"<div style='margin-bottom:4px;'><b>[端点对接]：</b> {' 、 '.join(end_to_end)}</div>")
        if t_junctions: html_lines.append(f"<div style='margin-bottom:4px;'><b>[T型搭接]：</b> {' 、 '.join(t_junctions)}</div>")
        if x_junctions: html_lines.append(f"<div style='margin-bottom:4px;'><b>[X型交叉]：</b> {' 、 '.join(x_junctions)}</div>")
        if not (end_to_end or t_junctions or x_junctions):
            html_lines.append("<div style='margin-bottom:4px; color:#777;'>当前无曲线发生物理碰撞。</div>")

        try:
            valid_cycles = []
            # 🌟 核心拦截 1：探测 2-Stroke 闭环 (例如字母 'o')
            for (u, v), pts_list in connection_points.items():
                if len(pts_list) >= 2:
                    for idx1 in range(len(pts_list)):
                        for idx2 in range(idx1+1, len(pts_list)):
                            # 如果两根线在两个不同的物理坐标(距离>5)相撞，它必定是个闭环！
                            if np.linalg.norm(pts_list[idx1] - pts_list[idx2]) >= 5.0:
                                valid_cycles.append([u, v])
                                break
                        else: continue
                        break

            # 🌟 核心拦截 2：探测常规的 >= 3-Stroke 闭环
            cycles = nx.cycle_basis(G)
            for cycle in cycles:
                k = len(cycle)
                if k < 3: continue
                is_valid = True
                for i in range(k):
                    u, v, w = cycle[i-1], cycle[i], cycle[(i+1)%k]
                    p_in = connection_points[(min(u,v), max(u,v))][0]
                    p_out = connection_points[(min(v,w), max(v,w))][0]
                    if np.linalg.norm(p_in - p_out) < 5.0:
                        is_valid = False; break
                if is_valid: valid_cycles.append(cycle)

            if valid_cycles:
                cycle_strs = []
                for cycle in valid_cycles:
                    styled_nodes = [_span(n) for n in cycle]
                    cycle_strs.append(f"{' '.join(styled_nodes)} 属同一环")
                html_lines.append(f"<div style='margin-bottom:4px;'><b>[闭环结构]：</b> {' &nbsp;|&nbsp; '.join(cycle_strs)}</div>")
        except: pass

        self.topo_text_box.setHtml("".join(html_lines))

    def update_canvas(self):
        for ax in [self.ax_top, self.ax_main, self.ax_orig, self.ax_color, self.ax_bw]:
            ax.clear()
            ax.axis('off')

        self.ax_top.imshow(self.binary, cmap='gray', alpha=0.15)
        self.ax_main.imshow(self.binary, cmap='gray', alpha=0.15)
        self.ax_orig.imshow(self.binary, cmap='gray')
        self.ax_color.imshow(self.binary, cmap='gray', alpha=0.05)
        self.ax_bw.imshow(np.ones_like(self.binary), cmap='gray', vmin=0, vmax=1)

        id_map = self.get_display_map()

        for edge in self.edges:
            eid = edge['id']
            is_sel = eid in self.selected_edge_ids
            color_hex = self.get_hex_color(eid)
            color = self.cmap((eid % 20))
            pts = np.array(edge['path'])
            w_opt = self.width_cache.get(eid, np.array([5.0, 5.0, 5.0, 5.0]))
            
            curve_pts = cubic_bezier_np(pts, np.linspace(0, 1, 50)[:, None])
            
            self.ax_top.plot(curve_pts[:, 0], curve_pts[:, 1], color=color, linewidth=2.5 if is_sel else 1.5, alpha=0.9)
            if is_sel:
                self.ax_top.plot(pts[[0, 1, 2, 3], 0], pts[[0, 1, 2, 3], 1], 'k.', markersize=3, alpha=0.4)
            mid_pt = curve_pts[25]
            self.ax_top.text(mid_pt[0], mid_pt[1], id_map[eid], color=color, fontsize=12, fontweight='bold', 
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

            ts_dense = np.linspace(0, 1, 50)[:, None]
            curve_dense = cubic_bezier_np(pts, ts_dense)
            mt = 1 - ts_dense
            w_vals = mt**3 * w_opt[0] + 3*mt**2*ts_dense * w_opt[1] + 3*mt*ts_dense**2 * w_opt[2] + ts_dense**3 * w_opt[3]
            dp = np.gradient(curve_dense, axis=0)
            n = np.zeros_like(dp)
            n[:, 0], n[:, 1] = -dp[:, 1], dp[:, 0]
            n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-5)
            upper, lower = curve_dense + n * w_vals, curve_dense - n * w_vals
            poly = np.vstack([upper, lower[::-1]])

            fill_color = color_hex if not is_sel else '#FF1744'
            self.ax_main.fill(poly[:, 0], poly[:, 1], color=fill_color, alpha=0.35, linewidth=0)
            self.ax_main.plot(curve_pts[:, 0], curve_pts[:, 1], color='black', linewidth=1.5 if is_sel else 0.8, alpha=0.8)
            
            if is_sel:
                self.ax_main.plot([pts[0,0], pts[1,0]], [pts[0,1], pts[1,1]], 'k--', alpha=0.5)
                self.ax_main.plot([pts[2,0], pts[3,0]], [pts[2,1], pts[3,1]], 'k--', alpha=0.5)
                self.ax_main.plot(pts[0,0], pts[0,1], 's', color=color, markersize=9, markeredgecolor='black')
                self.ax_main.plot(pts[3,0], pts[3,1], 's', color=color, markersize=9, markeredgecolor='black')
                self.ax_main.plot(pts[1,0], pts[1,1], 'o', color=color, markersize=7, markeredgecolor='black')
                self.ax_main.plot(pts[2,0], pts[2,1], 'o', color=color, markersize=7, markeredgecolor='black')

            self.ax_color.fill(poly[:, 0], poly[:, 1], color=fill_color, alpha=0.8, linewidth=0)
            self.ax_bw.fill(poly[:, 0], poly[:, 1], color='#E91E63' if is_sel else '#000000', alpha=1.0, linewidth=0)

        self.canvas.draw()

    # ==========================================
    # 🌟 轨迹拦截与操作事件记录
    # ==========================================
    def on_mouse_press(self, event):
        if event.inaxes != self.ax_main or not event.xdata or not event.ydata: return
        min_dist = 15.0; clicked_pt = None
        for i, edge in enumerate(self.edges):
            if edge['id'] not in self.selected_edge_ids: continue
            pts = np.array(edge['path'])
            for p_idx, pt in enumerate(pts):
                dist = np.hypot(pt[0] - event.xdata, pt[1] - event.ydata)
                if dist < min_dist:
                    min_dist, clicked_pt = dist, (i, p_idx)
                    
        if clicked_pt:
            self.dragging_point = clicked_pt
            self.co_dragged_points, self.t_constraints = [], []
            e_idx, p_idx = clicked_pt
            self.drag_start_pos = copy.deepcopy(self.edges[e_idx]['path'][p_idx])
            
            if p_idx in [0, 3]:
                target_pos = np.array(self.edges[e_idx]['path'][p_idx])
                for j, edge in enumerate(self.edges):
                    for ep_idx in [0, 3]:
                        if np.linalg.norm(np.array(edge['path'][ep_idx]) - target_pos) < 1.0:
                            self.co_dragged_points.append((j, ep_idx))
            else:
                self.co_dragged_points.append((e_idx, p_idx))
                
            host_curves = set([e for e, p in self.co_dragged_points])
            for h_idx in host_curves:
                c_host = cubic_bezier_np(np.array(self.edges[h_idx]['path']), np.linspace(0, 1, TOPO_SAMPLE_N)[:, None])
                for o_idx, other_edge in enumerate(self.edges):
                    if o_idx in host_curves: continue
                    for ep_idx in [0, 3]:
                        ep = np.array(other_edge['path'][ep_idx])
                        dists = np.linalg.norm(c_host - ep, axis=1)
                        min_t = np.argmin(dists)
                        if dists[min_t] < 1.5:
                            self.t_constraints.append((o_idx, ep_idx, h_idx, min_t))

    def on_mouse_move(self, event):
        if self.dragging_point and event.inaxes == self.ax_main and event.xdata and event.ydata:
            affected_edge_indices = set()
            for e_idx, p_idx in self.co_dragged_points:
                self.edges[e_idx]['path'][p_idx] = [event.xdata, event.ydata]
                affected_edge_indices.add(e_idx)
            for o_idx, ep_idx, h_idx, min_t in self.t_constraints:
                c_host_updated = cubic_bezier_np(np.array(self.edges[h_idx]['path']), np.linspace(0, 1, TOPO_SAMPLE_N)[:, None])
                self.edges[o_idx]['path'][ep_idx] = c_host_updated[min_t].tolist()
                affected_edge_indices.add(o_idx)
            for e_idx in affected_edge_indices:
                p_opt = np.array(self.edges[e_idx]['path'])
                self.width_cache[self.edges[e_idx]['id']] = regress_width_dt_fast(p_opt, self.dt_map)
            self.update_canvas()

    def on_mouse_release(self, event):
        if not self.dragging_point: return
            
        e_idx, p_idx = self.dragging_point
        id_map = self.get_display_map()
        guest_strk_id = int(id_map[self.edges[e_idx]['id']])
        start_p = self.drag_start_pos
        
        snapped_action_type = None
        host_strk_id = None
        host_t_val = None
        host_ep_val = None
        
        if p_idx in [0, 3]:
            pt = np.array(self.edges[e_idx]['path'][p_idx])
            min_ep_dist, best_ep_pos, best_ep_host, best_ep_idx = float('inf'), None, None, None
            min_curve_dist, best_curve_pos, best_curve_host, best_curve_t = float('inf'), None, None, None
            
            for i, edge in enumerate(self.edges):
                if i == e_idx: continue
                for end_idx in [0, 3]:
                    ep = np.array(edge['path'][end_idx])
                    dist = np.linalg.norm(pt - ep)
                    if dist < min_ep_dist:
                        min_ep_dist, best_ep_pos, best_ep_host, best_ep_idx = dist, ep.tolist(), i, end_idx
                
                curve = cubic_bezier_np(np.array(edge['path']), np.linspace(0, 1, TOPO_SAMPLE_N)[:, None])
                dists = np.linalg.norm(curve - pt, axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] < min_curve_dist:
                    min_curve_dist, best_curve_pos, best_curve_host, best_curve_t = dists[min_idx], curve[min_idx].tolist(), i, min_idx / (TOPO_SAMPLE_N - 1)

            snap_threshold = 12.0
            new_pos = None
            
            def _confirm_snap(title, msg):
                return QMessageBox.question(self, title, msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) == QMessageBox.Yes
                
            if min_ep_dist < snap_threshold and _confirm_snap('端点吸附确认', '是否将该点吸附到其他线的端点？'):
                new_pos = best_ep_pos
                snapped_action_type = "SNAP"
                host_strk_id = int(id_map[self.edges[best_ep_host]['id']])
                host_ep_val = "P3" if best_ep_idx == 3 else "P0"
                
            elif min_curve_dist < snap_threshold and _confirm_snap('T型搭接确认', '是否将该点吸附到其他线上（T型搭接）？'):
                new_pos = best_curve_pos
                snapped_action_type = "T_ATTACH"
                host_strk_id = int(id_map[self.edges[best_curve_host]['id']])
                host_t_val = best_curve_t

            if new_pos:
                for j, ep_idx in self.co_dragged_points:
                    self.edges[j]['path'][ep_idx] = copy.deepcopy(new_pos)
                    self.width_cache[self.edges[j]['id']] = regress_width_dt_fast(np.array(self.edges[j]['path']), self.dt_map)
                    
        # Action Log 记录
        final_p = self.edges[e_idx]['path'][p_idx]
        if np.linalg.norm(np.array(start_p) - np.array(final_p)) > 0.5:
            before_coord = [round(float(start_p[0]), 2), round(float(start_p[1]), 2)]
            after_coord = [round(float(final_p[0]), 2), round(float(final_p[1]), 2)]
            
            if snapped_action_type == "SNAP":
                self.edit_history.append({
                    "action": "SNAP",
                    "stroke": guest_strk_id,
                    "endpoint": f"P{p_idx}",
                    "host_stroke": host_strk_id,
                    "host_endpoint": host_ep_val,
                    "before": before_coord,
                    "after": after_coord
                })
            elif snapped_action_type == "T_ATTACH":
                self.edit_history.append({
                    "action": "T_ATTACH",
                    "guest": guest_strk_id,
                    "guest_endpoint": f"P{p_idx}",
                    "host": host_strk_id,
                    "host_t": round(float(host_t_val), 3),
                    "before": before_coord,
                    "after": after_coord
                })
            else:
                self.edit_history.append({
                    "action": "CONTROL_MOVE",
                    "stroke": guest_strk_id,
                    "control": f"P{p_idx}",
                    "before": before_coord,
                    "after": after_coord
                })

        self.save_state()
        self.dragging_point = None
        self.co_dragged_points = []
        self.t_constraints = []
        self.update_canvas()
        self.update_topology_text() # 🌟 恢复：每次拖拽释放后，自动刷新文本！

    # ==========================================
    # 🌟 落盘：5 层数据架构 (完全依照你的需求)
    # ==========================================
    def action_complete_topo(self):
        id_map = self.get_display_map()
        
        # 🟢 Layer 1: Geometry & Derived Features
        strokes_tokens = []
        for edge in self.edges:
            eid = edge['id']
            p_opt = np.array(edge['path'])
            w_opt = self.width_cache.get(eid, regress_width_dt_fast(p_opt, self.dt_map))
            
            c_pts = cubic_bezier_np(p_opt, np.linspace(0, 1, 50)[:, None])
            length = float(np.sum(np.linalg.norm(np.diff(c_pts, axis=0), axis=1)))
            xmin, ymin = np.min(c_pts, axis=0)
            xmax, ymax = np.max(c_pts, axis=0)
            
            s_type = "closed" if np.linalg.norm(p_opt[0] - p_opt[3]) < 2.0 else "open"
            
            strokes_tokens.append({
                "bezier_id": int(id_map[eid]), 
                "stroke_type": s_type,
                "length": round(length, 2),
                "bbox": [round(float(xmin), 1), round(float(ymin), 1), round(float(xmax), 1), round(float(ymax), 1)],
                "mother_bezier": p_opt.tolist(), 
                "width_bezier": w_opt.tolist() if hasattr(w_opt, 'tolist') else list(w_opt)
            })

        # 🟢 Layer 2 & 3: Parameter Space & Topology Events
        topology_events = []
        G_cycles = nx.Graph()
        
        for i, e1 in enumerate(self.edges):
            G_cycles.add_node(e1['id'])
            for j, e2 in enumerate(self.edges):
                if i >= j: continue
                p1, p2 = np.array(e1['path']), np.array(e2['path'])
                topo_ts = np.linspace(0, 1, TOPO_SAMPLE_N)[:, None]
                c1 = cubic_bezier_np(p1, topo_ts)
                c2 = cubic_bezier_np(p2, topo_ts)
                
                id1, id2 = int(id_map[e1['id']]), int(id_map[e2['id']])
                is_e2e = False
                
                # --- E2E ---
                for pt1_idx, t1 in [(0, 0.0), (3, 1.0)]:
                    for pt2_idx, t2 in [(0, 0.0), (3, 1.0)]:
                        if np.linalg.norm(p1[pt1_idx] - p2[pt2_idx]) < 2.0:
                            is_e2e = True
                            topology_events.append({
                                "type": "E2E",
                                "stroke_a": id1, "t_a": t1,
                                "stroke_b": id2, "t_b": t2,
                                "position": [round(float(p1[pt1_idx][0]), 1), round(float(p1[pt1_idx][1]), 1)]
                            })
                            
                # --- T 搭接 ---
                if not is_e2e:
                    for pt1_idx, t1 in [(0, 0.0), (3, 1.0)]:
                        dists = np.linalg.norm(c2 - p1[pt1_idx], axis=1)
                        m_idx = np.argmin(dists)
                        if dists[m_idx] < 2.0:
                            t2 = m_idx / (TOPO_SAMPLE_N - 1)
                            ang = get_angle(get_bezier_derivative(p1, t1), get_bezier_derivative(p2, t2))
                            topology_events.append({
                                "type": "T",
                                "guest": id1, "guest_t": t1,
                                "host": id2, "host_t": round(t2, 3),
                                "angle": round(ang, 1),
                                "position": [round(float(c2[m_idx][0]), 1), round(float(c2[m_idx][1]), 1)]
                            })
                    for pt2_idx, t2 in [(0, 0.0), (3, 1.0)]:
                        dists = np.linalg.norm(c1 - p2[pt2_idx], axis=1)
                        m_idx = np.argmin(dists)
                        if dists[m_idx] < 2.0:
                            t1 = m_idx / (TOPO_SAMPLE_N - 1)
                            ang = get_angle(get_bezier_derivative(p1, t1), get_bezier_derivative(p2, t2))
                            topology_events.append({
                                "type": "T",
                                "guest": id2, "guest_t": t2,
                                "host": id1, "host_t": round(t1, 3),
                                "angle": round(ang, 1),
                                "position": [round(float(c1[m_idx][0]), 1), round(float(c1[m_idx][1]), 1)]
                            })
                            
                # --- X 交叉 ---
                # 原逻辑：采样点最近距离 < 2.0。现在改成 polyline 线段相交，避免交点落入采样间隙。
                if not is_e2e and len([ev for ev in topology_events if ev['type'] == 'T' and ev['guest'] in (id1, id2)]) == 0:
                    hit_x, x_pt, t1, t2 = find_polyline_x_intersection(c1, c2)
                    if hit_x:
                        ang = get_angle(get_bezier_derivative(p1, t1), get_bezier_derivative(p2, t2))
                        topology_events.append({
                            "type": "X",
                            "stroke_a": id1, "t_a": round(float(t1), 3),
                            "stroke_b": id2, "t_b": round(float(t2), 3),
                            "angle": round(float(ang), 1),
                            "position": [round(float(x_pt[0]), 1), round(float(x_pt[1]), 1)]
                        })

        # 🟢 Layer 4: Cycles & Orientations
        # 🟢 Layer 4: Cycles & Orientations
        # 🟢 Layer 4: Cycles & Orientations
        cycles_tokens = []
        G_cycles = nx.Graph()
        connection_points = {}

        for ev in topology_events:
            ev_type = ev['type']
            if ev_type in ['E2E', 'T', 'X']:
                u = ev.get('stroke_a', ev.get('guest'))
                v = ev.get('stroke_b', ev.get('host'))
                if u is not None and v is not None:
                    u, v = min(u, v), max(u, v)
                    G_cycles.add_edge(u, v)
                    if (u, v) not in connection_points: connection_points[(u, v)] = []
                    connection_points[(u, v)].append(np.array(ev['position']))
            
        try:
            valid_cycle_lists = []
            
            # 🌟 1. 拦截识别 2-stroke cycles (如字母 'o')
            for (u, v), pts_list in connection_points.items():
                if len(pts_list) >= 2:
                    for idx1 in range(len(pts_list)):
                        for idx2 in range(idx1+1, len(pts_list)):
                            if np.linalg.norm(pts_list[idx1] - pts_list[idx2]) >= 5.0:
                                valid_cycle_lists.append([u, v])
                                break
                        else: continue
                        break
            
            # 🌟 2. 拦截识别 >=3 stroke cycles
            basis = nx.cycle_basis(G_cycles)
            for cycle_nodes in basis:
                members = [int(n) for n in cycle_nodes]
                k = len(members)
                if k < 3: continue
                is_valid = True
                for i in range(k):
                    u, v, w = members[i-1], members[i], members[(i+1)%k]
                    p_in = connection_points[(min(u,v), max(u,v))][0]
                    p_out = connection_points[(min(v,w), max(v,w))][0]
                    if np.linalg.norm(p_in - p_out) < 5.0:
                        is_valid = False; break
                if is_valid: valid_cycle_lists.append(members)
                
            # 🌟 3. 统一组装落盘数据
            for members in valid_cycle_lists:
                pts = [np.mean(next(e['path'] for e in self.edges if int(id_map[e['id']]) == n), axis=0) for n in members]
                orient = get_polygon_orientation(pts)
                cycles_tokens.append({"cycle_id": len(cycles_tokens), "members": members, "orientation": orient})
        except: pass

        self.main_ws.save_phase2_topo_data(self.hex_key, {
            "glyph_info": {"hex_key": self.hex_key, "char": self.char},
            "strokes": strokes_tokens,
            "topology_events": topology_events,
            "cycles": cycles_tokens,
            "edit_history": self.edit_history
        })

    def action_undo(self):
        if len(self.history_stack) > 1:
            self.history_stack.pop()
            self.edges = copy.deepcopy(self.history_stack[-1])
            self.force_recompute_all_widths()
            self.update_canvas()
            self.update_topology_text() # 🌟 恢复：撤销后更新拓扑文本

    def action_reset(self):
        self.edges = copy.deepcopy(self.initial_edges)
        self.history_stack.clear()
        self.save_state()
        self.force_recompute_all_widths()
        self.update_canvas()
        self.update_topology_text() # 🌟 恢复：重置后更新拓扑文本
        
    def action_return_to_phase1(self):
        self.main_ws.inner_stack.setCurrentIndex(0)

    def save_state(self):
        self.history_stack.append(copy.deepcopy(self.edges))
        if len(self.history_stack) > 30: self.history_stack.pop(0)

    def update_palette(self):
        id_map = self.get_display_map()
        while self.palette_layout.count():
            child = self.palette_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        for edge in self.edges:
            eid = edge['id']
            c = self.cmap((eid % 20))
            r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
            btn = QPushButton(f" {id_map[eid]} ")
            btn.setFixedSize(40, 30); btn.setCursor(Qt.PointingHandCursor)
            if eid in self.selected_edge_ids:
                btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 2px solid #E91E63; color: #FFF; font-weight: bold; border-radius: 4px;")
            else:
                btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #333; color: #000; border-radius: 4px;")
            btn.clicked.connect(lambda checked, e=eid: self.toggle_selection(e))
            self.palette_layout.addWidget(btn)
        self.palette_layout.addStretch()

    def toggle_selection(self, eid):
        if eid not in self.selected_edge_ids: self.selected_edge_ids.append(eid)
        else: self.selected_edge_ids.remove(eid)
        self.update_canvas(); self.update_palette()

    def on_key(self, event):
        key = event.key.lower() if event.key else ""
        if key == 'u': self.action_undo()
        elif key == 'r': self.action_reset()
        elif key == 'enter': self.action_complete_topo()