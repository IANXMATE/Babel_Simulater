import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea, QFrame, QMessageBox, QTextBrowser
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# 导入基础库中的贝塞尔渲染函数
from geometry_vision import cubic_bezier_np, regress_width_dt_fast

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
        
        # 拖拽联动状态
        self.dragging_point = None 
        self.co_dragged_points = [] 
        self.width_cache = {}       
        
        try: self.cmap = plt.colormaps['tab20']
        except AttributeError: self.cmap = plt.get_cmap('tab20')
        
        self.init_ui()
        self.save_state()
        self.force_recompute_all_widths() 
        self.update_canvas()
        self.update_palette()
        self.update_topology_text() 

    def init_ui(self):
        layout = QHBoxLayout(self)
        
        # ==========================================
        # 左侧控制面板
        # ==========================================
        control_panel = QVBoxLayout()
        control_panel.setSpacing(10)
        
        title = QLabel(f"Phase 2: Topo Tune\nTarget: '{self.char}'")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #E91E63;")
        control_panel.addWidget(title)
        
        line = QFrame(); line.setFrameShape(QFrame.HLine); control_panel.addWidget(line)
        
        btn_undo = QPushButton("Undo Last Move (U)")
        btn_undo.clicked.connect(self.action_undo)
        btn_undo.setStyleSheet("padding: 10px; background-color: #757575; color: white; font-weight: bold; border-radius: 4px;")
        control_panel.addWidget(btn_undo)
        
        btn_reset = QPushButton("Reset to Phase 1 (R)")
        btn_reset.clicked.connect(self.action_reset)
        btn_reset.setStyleSheet("padding: 10px; background-color: #795548; color: white; font-weight: bold; border-radius: 4px;")
        control_panel.addWidget(btn_reset)
        
        control_panel.addStretch()
        
        btn_complete = QPushButton("✅ FINISH TOPO (Enter)")
        btn_complete.clicked.connect(self.action_complete_topo)
        btn_complete.setStyleSheet("padding: 14px; background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px;")
        control_panel.addWidget(btn_complete)

        # ==========================================
        # 右侧核心工作区（3行5图结构 + 底部富文本）
        # ==========================================
        right_panel = QVBoxLayout()
        
        self.palette_container = QWidget()
        self.palette_layout = QHBoxLayout(self.palette_container)
        self.palette_layout.setContentsMargins(5, 5, 5, 5)
        self.palette_layout.setAlignment(Qt.AlignLeft)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True); scroll_area.setWidget(self.palette_container)
        scroll_area.setMaximumHeight(60)
        
        # 🌟 布局彻底放大撑满：调高 figsize，强制消除白边
        self.fig = plt.figure(figsize=(20, 18))
        self.fig.patch.set_facecolor('#FFFFFF')
        # left/right/top/bottom 控制绝对边距，极大地榨干屏幕空间
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.02)
        
        # 🌟 调整比例：第一第三行 1.8，第二行 4.5 (相当于原脚本的等比拉伸)
        gs = self.fig.add_gridspec(3, 3, height_ratios=[1.8, 4.5, 1.8], hspace=0.15, wspace=0.1)
        
        # 行 1：只居中放置一张带编号的小图 (通过占据全部列配合 equal 自动居中居大)
        self.ax_top = self.fig.add_subplot(gs[0, 1])
        
        # 行 2：极大的主编辑图
        self.ax_main = self.fig.add_subplot(gs[1, :])
        
        # 行 3：三张无编号辅助渲染图
        self.ax_orig = self.fig.add_subplot(gs[2, 0])
        self.ax_color = self.fig.add_subplot(gs[2, 1])
        self.ax_bw = self.fig.add_subplot(gs[2, 2])
        
        for ax in [self.ax_top, self.ax_main, self.ax_orig, self.ax_color, self.ax_bw]:
            ax.set_facecolor('#FFFFFF')
            ax.set_aspect('equal')
            ax.axis('off')
            
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('key_press_event', self.on_key)
        self.canvas.setFocusPolicy(Qt.StrongFocus)

        self.topo_text_box = QTextBrowser()
        self.topo_text_box.setStyleSheet("background-color: #F8F9FA; border: 1px solid #CCC; padding: 10px; font-size: 14px;")
        self.topo_text_box.setMaximumHeight(120)

        right_panel.addWidget(scroll_area)
        right_panel.addWidget(self.canvas, stretch=5)
        right_panel.addWidget(self.topo_text_box, stretch=1)

        layout.addLayout(control_panel, 1)
        layout.addLayout(right_panel, 5)

    # 🌟 映射函数：实时将底层错乱的 eid 映射为 1, 2, 3...
    def get_display_map(self):
        return {edge['id']: str(i + 1) for i, edge in enumerate(self.edges)}

    def get_hex_color(self, eid):
        c = self.cmap((eid % 20))
        return f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"

    def force_recompute_all_widths(self):
        for edge in self.edges:
            self.width_cache[edge['id']] = regress_width_dt_fast(np.array(edge['path']), self.dt_map)

    def update_topology_text(self):
        id_map = self.get_display_map()
        G = nx.Graph()
        for edge in self.edges: G.add_node(edge['id'])
            
        for i, e1 in enumerate(self.edges):
            for j, e2 in enumerate(self.edges):
                if i >= j: continue
                connected = False
                p1, p2 = np.array(e1['path']), np.array(e2['path'])
                
                for end1 in [p1[0], p1[3]]:
                    for end2 in [p2[0], p2[3]]:
                        if np.linalg.norm(end1 - end2) < 2.0: connected = True
                
                c1 = cubic_bezier_np(p1, np.linspace(0, 1, 30)[:, None])
                c2 = cubic_bezier_np(p2, np.linspace(0, 1, 30)[:, None])
                for end in [p1[0], p1[3]]:
                    if np.min(np.linalg.norm(c2 - end, axis=1)) < 2.0: connected = True
                for end in [p2[0], p2[3]]:
                    if np.min(np.linalg.norm(c1 - end, axis=1)) < 2.0: connected = True
                        
                if connected: G.add_edge(e1['id'], e2['id'])

        html_lines = ["<b style='color:#333; font-size:15px;'>📊 实时拓扑分析报告</b><br>"]
        
        edges_list = list(G.edges())
        if edges_list:
            html_lines.append("<b>[相交搭接]：</b> ")
            conn_strs = []
            for u, v in edges_list:
                cu, cv = self.get_hex_color(u), self.get_hex_color(v)
                # 使用 id_map 渲染纯数字 1, 2, 3...
                conn_strs.append(f"<span style='color:{cu}; font-weight:bold;'>{id_map[u]}</span> 与 <span style='color:{cv}; font-weight:bold;'>{id_map[v]}</span>")
            html_lines.append(" 、 ".join(conn_strs) + "<br>")
        else:
            html_lines.append("<span style='color:#777;'>当前无曲线相交。</span><br>")

        try:
            cycles = nx.cycle_basis(G)
            if cycles:
                html_lines.append("<b>[闭环结构]：</b> ")
                cycle_strs = []
                for cycle in cycles:
                    styled_nodes = [f"<span style='color:{self.get_hex_color(n)}; font-weight:bold;'>{id_map[n]}</span>" for n in cycle]
                    cycle_strs.append(f"{' '.join(styled_nodes)} 属同一环")
                html_lines.append(" &nbsp;|&nbsp; ".join(cycle_strs))
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

        self.ax_top.set_title("1. Index Reference (Non-editable)", fontsize=10, fontweight='bold')
        self.ax_main.set_title("2. Main Editor Canvas", fontsize=12, fontweight='bold')
        self.ax_orig.set_title("3. Raw Origin", fontsize=10, fontweight='bold')
        self.ax_color.set_title("4. Width Mesh", fontsize=10, fontweight='bold')
        self.ax_bw.set_title("5. TTF Preview", fontsize=10, fontweight='bold')

        id_map = self.get_display_map()

        for edge in self.edges:
            eid = edge['id']
            is_sel = eid in self.selected_edge_ids
            color_hex = self.get_hex_color(eid)
            color = self.cmap((eid % 20))
            pts = np.array(edge['path'])
            w_opt = self.width_cache.get(eid, np.array([5.0, 5.0, 5.0, 5.0]))
            
            curve_pts = cubic_bezier_np(pts, np.linspace(0, 1, 50)[:, None])
            
            # ==========================================
            # 视图 1 (上)：小图带数字编号
            # ==========================================
            self.ax_top.plot(curve_pts[:, 0], curve_pts[:, 1], color=color, linewidth=2.5 if is_sel else 1.5, alpha=0.9)
            if is_sel:
                self.ax_top.plot(pts[[0, 1, 2, 3], 0], pts[[0, 1, 2, 3], 1], 'k.', markersize=3, alpha=0.4)
            # 纯数字编号，不带前缀
            mid_pt = curve_pts[25]
            self.ax_top.text(mid_pt[0], mid_pt[1], id_map[eid], color=color, fontsize=12, fontweight='bold', 
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

            # ==========================================
            # 核心计算多边形宽度
            # ==========================================
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

            # ==========================================
            # 视图 2 (中)：交互编辑巨图
            # ==========================================
            # 1. 极其微弱的实时宽度填充（不干扰视线）
            fill_color = color_hex if not is_sel else '#FF1744'
            self.ax_main.fill(poly[:, 0], poly[:, 1], color=fill_color, alpha=0.12, linewidth=0)
            
            # 2. 细黑线渲染母线骨架
            self.ax_main.plot(curve_pts[:, 0], curve_pts[:, 1], color='black', linewidth=1.5 if is_sel else 0.8, alpha=0.8)
            
            # 3. 选中时才渲染交互锚点
            if is_sel:
                self.ax_main.plot([pts[0,0], pts[1,0]], [pts[0,1], pts[1,1]], 'k--', alpha=0.5)
                self.ax_main.plot([pts[2,0], pts[3,0]], [pts[2,1], pts[3,1]], 'k--', alpha=0.5)
                self.ax_main.plot(pts[0,0], pts[0,1], 's', color=color, markersize=9, markeredgecolor='black')
                self.ax_main.plot(pts[3,0], pts[3,1], 's', color=color, markersize=9, markeredgecolor='black')
                self.ax_main.plot(pts[1,0], pts[1,1], 'o', color=color, markersize=7, markeredgecolor='black')
                self.ax_main.plot(pts[2,0], pts[2,1], 'o', color=color, markersize=7, markeredgecolor='black')

            # ==========================================
            # 视图 3/4/5 (下)：无编号宽度渲染
            # ==========================================
            self.ax_color.fill(poly[:, 0], poly[:, 1], color=fill_color, alpha=0.8, linewidth=0)
            self.ax_bw.fill(poly[:, 0], poly[:, 1], color='#E91E63' if is_sel else '#000000', alpha=1.0, linewidth=0)

        self.canvas.draw()

    # ==========================================
    # 交互与吸附拦截逻辑（保持锁定 ax_main）
    # ==========================================
    def get_line_count_at_point(self, target_pos, threshold=2.0):
        count = 0
        target_pt = np.array(target_pos)
        for edge in self.edges:
            pts = edge['path']
            if np.linalg.norm(np.array(pts[0]) - target_pt) < threshold: count += 1
            if np.linalg.norm(np.array(pts[3]) - target_pt) < threshold: count += 1
        return count

    def confirm_snap(self, snap_type, target_pos):
        x = self.get_line_count_at_point(target_pos)
        if snap_type == 'endpoint':
            title = "端点吸附确认"
            msg = f"是否将该点吸附到其他线的端点？\n\n📌 待吸附端点共有 {x} 个线\n"
            msg += "(正常对接)" if x == 1 else "(⚠️ 提示：x > 1，说明已经是交点了！)"
        elif snap_type == 't_junction':
            title = "T型搭接确认"
            msg = f"是否将该点吸附到其他线上（T型搭接）？\n\n📌 目标位置目前端点数为 {x}\n"
            msg += "(正常搭接)" if x == 0 else "(⚠️ 提示：该线段位置已经存在其他交点了！)"
        else:
            return True
            
        reply = QMessageBox.question(self, title, msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        return reply == QMessageBox.Yes

    def on_mouse_press(self, event):
        if event.inaxes != self.ax_main or not event.xdata or not event.ydata: return
        min_dist = 15.0
        clicked_pt = None
        
        for i, edge in enumerate(self.edges):
            if edge['id'] not in self.selected_edge_ids: continue
            pts = np.array(edge['path'])
            for p_idx, pt in enumerate(pts):
                dist = np.hypot(pt[0] - event.xdata, pt[1] - event.ydata)
                if dist < min_dist:
                    min_dist = dist
                    clicked_pt = (i, p_idx)
                    
        if clicked_pt:
            self.dragging_point = clicked_pt
            self.co_dragged_points = []
            e_idx, p_idx = clicked_pt
            
            if p_idx in [0, 3]:
                target_pos = np.array(self.edges[e_idx]['path'][p_idx])
                for j, edge in enumerate(self.edges):
                    for ep_idx in [0, 3]:
                        if np.linalg.norm(np.array(edge['path'][ep_idx]) - target_pos) < 1.0:
                            self.co_dragged_points.append((j, ep_idx))
            else:
                self.co_dragged_points.append((e_idx, p_idx))

    def on_mouse_move(self, event):
        if self.dragging_point and event.inaxes == self.ax_main and event.xdata and event.ydata:
            affected_edge_indices = set()
            for e_idx, p_idx in self.co_dragged_points:
                self.edges[e_idx]['path'][p_idx] = [event.xdata, event.ydata]
                affected_edge_indices.add(e_idx)
                
            for e_idx in affected_edge_indices:
                p_opt = np.array(self.edges[e_idx]['path'])
                self.width_cache[self.edges[e_idx]['id']] = regress_width_dt_fast(p_opt, self.dt_map)
                
            self.update_canvas()

    def on_mouse_release(self, event):
        if not self.dragging_point: return
            
        e_idx, p_idx = self.dragging_point
        if p_idx in [0, 3]:
            pt = np.array(self.edges[e_idx]['path'][p_idx])
            min_ep_dist, best_ep_pos = float('inf'), None
            min_curve_dist, best_curve_pos = float('inf'), None
            
            for i, edge in enumerate(self.edges):
                if i == e_idx: continue
                for end_idx in [0, 3]:
                    ep = np.array(edge['path'][end_idx])
                    dist = np.linalg.norm(pt - ep)
                    if dist < min_ep_dist: min_ep_dist, best_ep_pos = dist, ep.tolist()
                
                curve = cubic_bezier_np(np.array(edge['path']), np.linspace(0, 1, 50)[:, None])
                dists = np.linalg.norm(curve - pt, axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] < min_curve_dist: min_curve_dist, best_curve_pos = dists[min_idx], curve[min_idx].tolist()

            snap_threshold = 12.0
            new_pos = None
            if min_ep_dist < snap_threshold and self.confirm_snap('endpoint', best_ep_pos):
                new_pos = best_ep_pos
            elif min_curve_dist < snap_threshold and self.confirm_snap('t_junction', best_curve_pos):
                new_pos = best_curve_pos

            if new_pos:
                for j, ep_idx in self.co_dragged_points:
                    self.edges[j]['path'][ep_idx] = copy.deepcopy(new_pos)
                    self.width_cache[self.edges[j]['id']] = regress_width_dt_fast(np.array(self.edges[j]['path']), self.dt_map)

        self.save_state()
        self.dragging_point = None
        self.co_dragged_points = []
        self.update_canvas()
        self.update_topology_text()

    def save_state(self):
        self.history_stack.append(copy.deepcopy(self.edges))
        if len(self.history_stack) > 30: self.history_stack.pop(0)

    def action_undo(self):
        if len(self.history_stack) > 1:
            self.history_stack.pop()
            self.edges = copy.deepcopy(self.history_stack[-1])
            self.force_recompute_all_widths()
            self.update_canvas()
            self.update_topology_text()

    def action_reset(self):
        self.edges = copy.deepcopy(self.initial_edges)
        self.history_stack.clear()
        self.save_state()
        self.force_recompute_all_widths()
        self.update_canvas()
        self.update_topology_text()

    def action_complete_topo(self):
        # 1. 提取重整后的纯数字映射字典
        id_map = self.get_display_map()
        
        # 2. 序列化贝塞尔线条几何与宽度数据
        strokes_tokens = []
        for edge in self.edges:
            eid = edge['id']
            p_opt = np.array(edge['path'])
            w_opt = self.width_cache.get(eid, regress_width_dt_fast(p_opt, self.dt_map))
            w_opt_serializable = w_opt.tolist() if hasattr(w_opt, 'tolist') else list(w_opt)
            
            strokes_tokens.append({
                "bezier_id": int(id_map[eid]), # 存储重整后的纯数字序列标签
                "mother_bezier": p_opt.tolist(), 
                "width_bezier": w_opt_serializable
            })

        # 3. 在落盘前重新构建图论拓扑结构
        G = nx.Graph()
        for edge in self.edges: 
            G.add_node(edge['id'])
            
        for i, e1 in enumerate(self.edges):
            for j, e2 in enumerate(self.edges):
                if i >= j: continue
                connected = False
                p1, p2 = np.array(e1['path']), np.array(e2['path'])
                
                # 终点相交探测
                for end1 in [p1[0], p1[3]]:
                    for end2 in [p2[0], p2[3]]:
                        if np.linalg.norm(end1 - end2) < 2.0: connected = True
                
                # T型搭接探测
                c1 = cubic_bezier_np(p1, np.linspace(0, 1, 30)[:, None])
                c2 = cubic_bezier_np(p2, np.linspace(0, 1, 30)[:, None])
                for end in [p1[0], p1[3]]:
                    if np.min(np.linalg.norm(c2 - end, axis=1)) < 2.0: connected = True
                for end in [p2[0], p2[3]]:
                    if np.min(np.linalg.norm(c1 - end, axis=1)) < 2.0: connected = True
                        
                if connected: 
                    G.add_edge(e1['id'], e2['id'])

        # 4. 提取交点与闭环数据（统一转化为纯数字格式标签）
        intersections = [[int(id_map[u]), int(id_map[v])] for u, v in G.edges()]
        try:
            cycles = [[int(id_map[n]) for n in cycle] for cycle in nx.cycle_basis(G)]
        except:
            cycles = []

        # 5. 打包几何 + 拓扑的大组合包
        char_bundle = {
            "strokes": strokes_tokens,
            "topology": {
                "intersections": intersections,
                "cycles": cycles
            }
        }
        
        # 将组合包交付主窗口的保存方法
        self.main_ws.save_phase2_topo_data(self.hex_key, char_bundle)

    def update_palette(self):
        id_map = self.get_display_map()
        while self.palette_layout.count():
            child = self.palette_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        for edge in self.edges:
            eid = edge['id']
            c = self.cmap((eid % 20))
            r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
            # 按钮上同样显示 1, 2, 3...
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