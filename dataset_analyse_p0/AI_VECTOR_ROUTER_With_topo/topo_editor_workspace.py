import copy
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea, QFrame, QMessageBox
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# 导入你基础库中的贝塞尔渲染函数
from geometry_vision import cubic_bezier_np, regress_width_dt_fast

class TopoAnnotationWorkspace(QWidget):
    def __init__(self, main_workspace, hex_key, char, binary, dt_map, phase1_edges):
        super().__init__()
        self.main_ws = main_workspace # 保存主界面的引用，方便保存后退回或切题
        self.hex_key = hex_key
        self.char = char
        self.binary = binary
        self.dt_map = dt_map
        
        # 深度拷贝，作为本阶段的初始状态 (用于 Reset)
        self.initial_edges = copy.deepcopy(phase1_edges)
        self.edges = copy.deepcopy(phase1_edges)
        
        self.history_stack = []
        self.selected_edge_ids = []
        
        # 拖拽交互状态
        self.dragging_point = None # (edge_idx, point_idx)
        
        try: self.cmap = plt.colormaps['tab20']
        except AttributeError: self.cmap = plt.get_cmap('tab20')
        
        self.init_ui()
        self.save_state()
        self.update_canvas()
        self.update_palette()

    def init_ui(self):
        layout = QHBoxLayout(self)
        
        # 左侧控制面板
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

        # 右侧画布面板
        right_panel = QVBoxLayout()
        self.palette_container = QWidget()
        self.palette_layout = QHBoxLayout(self.palette_container)
        self.palette_layout.setContentsMargins(5, 5, 5, 5)
        self.palette_layout.setAlignment(Qt.AlignLeft)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True); scroll_area.setWidget(self.palette_container)
        scroll_area.setMaximumHeight(60) 
        
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
        self.fig.patch.set_facecolor('#FFFFFF') 
        self.ax.set_facecolor('#FFFFFF')
            
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('key_press_event', self.on_key)
        self.canvas.setFocusPolicy(Qt.StrongFocus)

        right_panel.addWidget(scroll_area)
        right_panel.addWidget(self.canvas)

        layout.addLayout(control_panel, 1)
        layout.addLayout(right_panel, 4)

    def save_state(self):
        self.history_stack.append(copy.deepcopy(self.edges))
        if len(self.history_stack) > 30: self.history_stack.pop(0)

    def action_undo(self):
        if len(self.history_stack) > 1:
            self.history_stack.pop() 
            self.edges = copy.deepcopy(self.history_stack[-1])
            self.update_canvas()

    def action_reset(self):
        self.edges = copy.deepcopy(self.initial_edges)
        self.history_stack.clear()
        self.save_state()
        self.update_canvas()

    def action_complete_topo(self):
        # 将最终微调好的点存入 annotations_topo 数据库
        final_tokens = []
        for edge in self.edges:
            eid = edge['id']
            p_opt = np.array(edge['path']) # 在第二阶段，path已经直接是优化后的 4 个控制点了！
            w_opt = regress_width_dt_fast(p_opt, self.dt_map)
            final_tokens.append({"bezier_id": eid, "mother_bezier": p_opt.tolist(), "width_bezier": w_opt})
        
        # 通知主界面保存 Topo 数据，并切换到下一个字符
        self.main_ws.save_phase2_topo_data(self.hex_key, final_tokens)

    def update_palette(self):
        while self.palette_layout.count():
            child = self.palette_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        for edge in self.edges:
            eid = edge['id']
            c = self.cmap((eid % 20))
            r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
            btn = QPushButton(f" {eid} ")
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

    def update_canvas(self):
        self.ax.clear()
        self.ax.imshow(self.binary, cmap='gray', alpha=0.15)
        self.ax.set_title("Drag Control Points to Fine-Tune", color='black')
        self.ax.axis('off')

        for i, edge in enumerate(self.edges):
            eid = edge['id']
            is_sel = eid in self.selected_edge_ids
            color = self.cmap((eid % 20))
            pts = np.array(edge['path'])
            
            # 画曲线
            curve = cubic_bezier_np(pts, np.linspace(0, 1, 50)[:, None])
            self.ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=4 if is_sel else 2, alpha=1.0 if is_sel else 0.6)
            
            # 绘制交互控制点 (两头端点为方形，中间控制点为圆形)
            if is_sel:
                self.ax.plot([pts[0,0], pts[1,0]], [pts[0,1], pts[1,1]], 'k--', alpha=0.5)
                self.ax.plot([pts[2,0], pts[3,0]], [pts[2,1], pts[3,1]], 'k--', alpha=0.5)
                self.ax.plot(pts[0,0], pts[0,1], 's', color=color, markersize=8, markeredgecolor='black')
                self.ax.plot(pts[3,0], pts[3,1], 's', color=color, markersize=8, markeredgecolor='black')
                self.ax.plot(pts[1,0], pts[1,1], 'o', color=color, markersize=6, markeredgecolor='black')
                self.ax.plot(pts[2,0], pts[2,1], 'o', color=color, markersize=6, markeredgecolor='black')

        self.canvas.draw()

    # --- 拖拽交互逻辑 ---
    def on_mouse_press(self, event):
        if event.inaxes != self.ax or not event.xdata or not event.ydata: return
        # 寻找最近的选中曲线的控制点
        min_dist = 15.0 # 鼠标点击的容差范围
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

    def on_mouse_move(self, event):
        if self.dragging_point and event.inaxes == self.ax and event.xdata and event.ydata:
            e_idx, p_idx = self.dragging_point
            self.edges[e_idx]['path'][p_idx] = [event.xdata, event.ydata]
            self.update_canvas()

    def on_mouse_release(self, event):
        if self.dragging_point:
            self.save_state()
            self.dragging_point = None
            
    def on_key(self, event):
        key = event.key.lower() if event.key else ""
        if key == 'u': self.action_undo()
        elif key == 'r': self.action_reset()
        elif key == 'enter': self.action_complete_topo()