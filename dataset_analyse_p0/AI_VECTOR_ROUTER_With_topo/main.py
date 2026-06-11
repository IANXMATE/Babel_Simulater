import os
import sys
import random
import copy
import io
import qdarktheme
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFrame, QMessageBox, 
                             QScrollArea, QStackedWidget, QGridLayout, QListWidget, QSplitter)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.ndimage import distance_transform_edt
from skimage import morphology
from skimage.morphology import medial_axis
from skan import Skeleton, summarize

# 🌟 导入我们剥离的模块
from data_manager import DatasetManager
from geometry_vision import (
    extract_all_real_chars, render_unicode_glyph, collapse_degree2_nodes,
    prune_spurs, stitch_paths, cubic_bezier_np, fit_bezier_basic_with_error,
    split_pixel_path_adaptively, regress_width_dt_fast
)

# 常量
CANVAS_SIZE = 400
MAX_BEZIER_ERROR = 2
MAX_SPUR_LENGTH = 20.0
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR = os.path.join(SCRIPT_DIR, "../dataset/alien_tensors_raw")
os.makedirs(FONTS_DIR, exist_ok=True)

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# ==========================================
# 🎨 右侧：核心工作区 (从原来的 QMainWindow 降级为 QWidget)
# ==========================================
class AnnotationWorkspace(QWidget):
    def __init__(self, font_path):
        super().__init__()
        self.font_path = font_path
        self.font_filename = os.path.basename(font_path).split('.')[0]
        
        # 🌟 实例化你的数据中枢
        self.db = DatasetManager(SCRIPT_DIR, self.font_filename)
        
        self.font_charset = extract_all_real_chars(self.font_path, CANVAS_SIZE)
        self.history_stack = [] 
        self.selected_edge_ids = [] 
        self.last_click_coord = None 
        self.char = None
        self.thumbnail_cache = {}
        
        try: self.cmap = plt.colormaps['tab20']
        except AttributeError: self.cmap = plt.get_cmap('tab20')
        self.bezier_cache = {}
        
        self.init_ui()
        self.action_next_char()
        self.update_stats_display()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 顶部导航栏
        nav_bar = QHBoxLayout()
        self.btn_anno = QPushButton("✍️ Annotation")
        self.btn_anno.setCheckable(True)
        self.btn_anno.setChecked(True)
        self.btn_anno.clicked.connect(lambda: self.switch_tab(0))
        
        self.btn_comp = QPushButton("✅ Completed")
        self.btn_comp.setCheckable(True)
        self.btn_comp.clicked.connect(lambda: self.switch_tab(1))
        
        self.btn_ban = QPushButton("🚫 Banned")
        self.btn_ban.setCheckable(True)
        self.btn_ban.clicked.connect(lambda: self.switch_tab(2))
        
        nav_bar.addWidget(self.btn_anno)
        nav_bar.addWidget(self.btn_comp)
        nav_bar.addWidget(self.btn_ban)
        nav_bar.addStretch()

        # 🌟 新增：右上角的进度仪表盘
        self.stats_label = QLabel("📊 Loading Stats...")
        self.stats_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #90CAF9; padding-right: 15px;")
        nav_bar.addWidget(self.stats_label)

        layout.addLayout(nav_bar)

        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)
        
        self.init_page_annotation()
        self.init_page_gallery("completed")
        self.init_page_gallery("banned")
        
        self.stacked_widget.addWidget(self.page_annotation)
        self.stacked_widget.addWidget(self.page_completed)
        self.stacked_widget.addWidget(self.page_banned)
    
    def update_stats_display(self):
        """🌟 实时计算并刷新右上角的进度仪表盘"""
        total = len(self.font_charset)
        completed = len(self.db.annotated_outlines)
        banned = len(self.db.meta_data.get("banned", []))
        
        # 剩余数量 = 总数 - 已完成 - 封禁
        remaining = total - completed - banned
        remaining = max(0, remaining) # 防止极端情况出现负数
        
        # 更新文本
        self.stats_label.setText(
            f"📊 Remaining: {remaining}   |   ✅ Completed: {completed}   |   🚫 Banned: {banned}"
        )
    

    def switch_tab(self, index):
        self.btn_anno.setChecked(index == 0)
        self.btn_comp.setChecked(index == 1)
        self.btn_ban.setChecked(index == 2)
        
        if index == 1:
            self.refresh_gallery(self.completed_grid, list(self.db.annotated_outlines.keys()), "completed")
        elif index == 2:
            self.refresh_gallery(self.banned_grid, self.db.meta_data["banned"], "banned")
            
        self.stacked_widget.setCurrentIndex(index)

    def init_page_annotation(self):
        self.page_annotation = QWidget()
        layout = QHBoxLayout(self.page_annotation)

        # 左侧控制面板
        control_panel = QVBoxLayout()
        control_panel.setSpacing(10)
        
        self.title_label = QLabel("Loading...")
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        control_panel.addWidget(self.title_label)
        
        self.stroke_info_label = QLabel("Beziers: 0")
        self.stroke_info_label.setStyleSheet("font-size: 15px; color: #E91E63; font-weight: bold;")
        control_panel.addWidget(self.stroke_info_label)
        
        line1 = QFrame(); line1.setFrameShape(QFrame.HLine); control_panel.addWidget(line1)
        
        # 操作按钮
        actions = [
            ("Merge Strokes (M)", "#4CAF50", self.action_merge),
            ("Delete Stroke (D)", "#F44336", self.action_delete),
            ("Split at Point (B)", "#FF9800", self.action_breakpoint),
            ("Prune Redundant (C)", "#9C27B0", self.action_prune_parallel),
            ("Add Missing Dot (A)", "#00BCD4", self.action_add_dot),
            ("Undo Last Action (U)", "#757575", self.action_undo),
            ("Reset Current Char (R)", "#795548", self.action_reset_char),
            ("Next Random Char (N)", "#2196F3", self.action_next_char)
        ]
        
        for text, color, func in actions:
            btn = QPushButton(text)
            btn.clicked.connect(func)
            btn.setStyleSheet(f"padding: 10px; background-color: {color}; color: white; font-weight: bold; border-radius: 4px;")
            control_panel.addWidget(btn)
            
        control_panel.addStretch()
        
        btn_ban = QPushButton("🚫 BAN Character")
        btn_ban.clicked.connect(self.action_ban_char)
        btn_ban.setStyleSheet("padding: 12px; background-color: #000000; color: white; font-weight: bold;")
        control_panel.addWidget(btn_ban)

        btn_complete = QPushButton("✅ COMPLETE (Enter)")
        btn_complete.clicked.connect(self.action_complete_annotation)
        btn_complete.setStyleSheet("padding: 14px; background-color: #E91E63; color: white; font-weight: bold; font-size: 14px;")
        control_panel.addWidget(btn_complete)

        # 右侧画布面板
        right_panel = QVBoxLayout()
        self.palette_container = QWidget()
        self.palette_layout = QHBoxLayout(self.palette_container)
        self.palette_layout.setContentsMargins(5, 5, 5, 5); self.palette_layout.setSpacing(8)
        self.palette_layout.setAlignment(Qt.AlignLeft)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True); scroll_area.setWidget(self.palette_container)
        scroll_area.setMaximumHeight(60) 
        
        # 🌟 修改为 4 个子图，并加大画布宽度
        self.fig, (self.ax_ref, self.ax_main, self.ax_prev, self.ax_final) = plt.subplots(1, 4, figsize=(18, 5))
        
        # 🌟 画布和子图背景改为纯白
        self.fig.patch.set_facecolor('#FFFFFF') 
        for ax in (self.ax_ref, self.ax_main, self.ax_prev, self.ax_final):
            ax.set_facecolor('#FFFFFF')
            
        self.canvas = FigureCanvas(self.fig)

        self.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.mpl_connect('button_press_event', self.on_click_canvas)
        self.canvas.mpl_connect('key_press_event', self.on_key)

        right_panel.addWidget(scroll_area)
        right_panel.addWidget(self.canvas)

        layout.addLayout(control_panel, 1)
        layout.addLayout(right_panel, 6)

    def init_page_gallery(self, mode):
        page = QWidget()
        layout = QVBoxLayout(page)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        grid = QGridLayout(container)
        scroll.setWidget(container)
        layout.addWidget(scroll)
        
        if mode == "completed":
            self.page_completed = page
            self.completed_grid = grid
        else:
            self.page_banned = page
            self.banned_grid = grid

    # ---------------- 业务逻辑 (调用 geometry_vision) ----------------
    def load_char_topology(self):
        skel, distance = medial_axis(self.binary, return_distance=True)
        skel_obj = Skeleton(skel)
        branch_data = summarize(skel_obj, separator='-')
        G = nx.MultiGraph()
        for index, row in branch_data.iterrows():
            coords = skel_obj.path_coordinates(index)
            if len(coords) > 2:
                path = np.column_stack([coords[:, 1], coords[:, 0]])
                src, dst = int(row['node-id-src']), int(row['node-id-dst'])
                if np.linalg.norm(path[0] - skel_obj.coordinates[src][::-1]) > 1.0: path = path[::-1]
                G.add_edge(src, dst, key=index, path=path)
                
        G = prune_spurs(G, max_length=MAX_SPUR_LENGTH) 
        G = collapse_degree2_nodes(G)
        
        self.edges = []
        self.bezier_cache.clear() 
        global_id = 0
        for u, v, k, d in G.edges(keys=True, data=True):
            path = d['path']
            sub_paths = split_pixel_path_adaptively(path, MAX_BEZIER_ERROR)
            for sp in sub_paths:
                self.edges.append({'id': global_id, 'path': sp})
                global_id += 1
                            
        self.selected_edge_ids = []
        self.history_stack.clear()
        self.last_click_coord = None
        self.save_state()
        self.update_canvas()
        self.update_palette()

    def action_next_char(self):
        max_attempts = 500
        valid_char_found = False
        for _ in range(max_attempts):
            candidate = random.choice(self.font_charset)
            if candidate in self.db.meta_data["banned"] or candidate in self.db.annotated_outlines:
                continue
                
            b_mask = render_unicode_glyph(self.font_path, candidate, CANVAS_SIZE)
            if not np.any(b_mask): continue
            skel = morphology.skeletonize(b_mask)
            if not np.any(skel): continue 
                
            try:
                skel_obj = Skeleton(skel)
                if len(summarize(skel_obj, separator='-')) == 0: continue
            except ValueError: continue
                
            self.char, self.binary = candidate, b_mask
            valid_char_found = True
            break
            
        if not valid_char_found: 
            self.title_label.setText("🎉 All done!")
            return

        unicode_hex = f"U+{ord(self.char):04X}"
        self.title_label.setText(f"Target: '{self.char}' ({unicode_hex})")
        self.dt_map = distance_transform_edt(self.binary)
        self.load_char_topology()

    def action_complete_annotation(self):
        if not self.edges: return
        final_tokens = []
        for edge in self.edges:
            eid = edge['id']
            if eid in self.bezier_cache:
                p_opt, w_opt = self.bezier_cache[eid]
                final_tokens.append({
                    "bezier_id": eid, "mother_bezier": p_opt.tolist(), "width_bezier": w_opt
                })
        
        hex_key = f"U+{ord(self.char):04X}"
        self.db.annotated_outlines[hex_key] = final_tokens
        
        raw_edges_serializable = [{"id": e['id'], "path": e['path'].tolist()} for e in self.edges]
        self.db.meta_data["raw_edges"][hex_key] = raw_edges_serializable
        
        if hex_key in self.db.meta_data['banned']:
            self.db.meta_data['banned'].remove(hex_key)
            
        self.db.save_data()
        self.thumbnail_cache.pop(f"{hex_key}_completed", None) 
        self.update_stats_display()
        self.action_next_char()

    def action_ban_char(self):
        if self.char:
            hex_key = f"U+{ord(self.char):04X}"
            if hex_key not in self.db.meta_data["banned"]:
                self.db.meta_data["banned"].append(hex_key)
            if hex_key in self.db.annotated_outlines: del self.db.annotated_outlines[hex_key]
            self.db.save_data()
            self.update_stats_display()
            self.action_next_char()

    # --- 交互与画布刷新 (略过重复的冗长代码，直接复用你之前的完美逻辑) ---
    def update_canvas(self):
            # --- 1. 原始图 ---
            self.ax_ref.clear(); self.ax_ref.imshow(self.binary, cmap='gray'); self.ax_ref.set_title("1. Original", color='black'); self.ax_ref.axis('off')
            
            # --- 2. 拓扑图 ---
            self.ax_main.clear(); self.ax_main.imshow(self.binary, cmap='gray', alpha=0.15)
            self.stroke_info_label.setText(f"Bezier Strokes: {len(self.edges)}")
            for i, edge in enumerate(self.edges):
                eid = edge['id']
                is_sel = eid in self.selected_edge_ids
                color = self.cmap((eid % 20))
                lw = 6 if is_sel else 3; alpha = 1.0 if is_sel else 0.6
                line, = self.ax_main.plot(edge['path'][:, 0], edge['path'][:, 1], c=color, lw=lw, alpha=alpha, picker=5)
                line.edge_idx = i 
            if self.last_click_coord: self.ax_main.plot(self.last_click_coord[0], self.last_click_coord[1], 'rX', markersize=10)
            self.ax_main.set_title("2. Topology", color='black'); self.ax_main.axis('off')

            # --- 3. 贝塞尔单线预览 ---
            self.ax_prev.clear(); self.ax_prev.imshow(self.binary, cmap='gray', alpha=0.05); self.ax_prev.set_title("3. Bezier", color='black'); self.ax_prev.axis('off')
            
            # --- 🌟 4. 新增：物理轮廓反解渲染 ---
            self.ax_final.clear()
            self.ax_final.imshow(np.ones_like(self.binary), cmap='gray', vmin=0, vmax=1) # 撑开空白的等大底板
            self.ax_final.set_title("4. Reconstructed TTF", color='black')
            self.ax_final.axis('off')

            for edge in self.edges:
                eid = edge['id']
                is_sel = eid in self.selected_edge_ids
                color = self.cmap((eid % 20))
                if eid not in self.bezier_cache:
                    p_opt, _ = fit_bezier_basic_with_error(edge['path'])
                    w_opt = regress_width_dt_fast(p_opt, self.dt_map)
                    self.bezier_cache[eid] = (p_opt, w_opt)
                p_opt, w_opt = self.bezier_cache[eid]
                
                # --- 画第三张图 ---
                mean_w = max(np.mean(w_opt), 1.0)
                ts = np.linspace(0, 1, 50)[:, None]
                curve = cubic_bezier_np(p_opt, ts)
                final_lw = mean_w * 2 * (1.5 if is_sel else 1.0)
                final_alpha = 0.9 if is_sel else 0.4
                self.ax_prev.plot(curve[:, 0], curve[:, 1], color=color, linewidth=final_lw, solid_capstyle='round', alpha=final_alpha)
                self.ax_prev.plot(curve[:, 0], curve[:, 1], color='black', linewidth=1.5, alpha=final_alpha+0.1)

                # ====================================================
                # 🌟 画第四张图：从张量反解为真实闭合轮廓
                # ====================================================
                ts_dense = np.linspace(0, 1, 100)[:, None] # 提升点密度保证边缘平滑
                curve_dense = cubic_bezier_np(p_opt, ts_dense)

                # 1. 解算宽度方程 W(t)
                mt = 1 - ts_dense
                w_vals = mt**3 * w_opt[0] + 3*mt**2*ts_dense * w_opt[1] + 3*mt*ts_dense**2 * w_opt[2] + ts_dense**3 * w_opt[3]

                # 2. 计算当前路径上每个点的“法向量 (Normal Vector)”
                dp = np.gradient(curve_dense, axis=0) # 求导获取切线
                n = np.zeros_like(dp)
                n[:, 0], n[:, 1] = -dp[:, 1], dp[:, 0] # 切线顺时针转90度即为法线
                n_norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-5
                n = n / n_norm # 归一化为单位法向量

                # 3. 沿法向量向外扩张，生成上下边缘，合并为闭合多边形
                upper = curve_dense + n * w_vals
                lower = curve_dense - n * w_vals
                poly = np.vstack([upper, lower[::-1]]) # 尾部需要反转拼接，形成一个环

                # 4. 在画布上使用无边框的多边形填充
                # 选中时高亮为粉红色，否则为纯黑色，模拟真实墨水叠加
                fill_color = '#E91E63' if is_sel else '#000000'
                fill_alpha = 0.8 if is_sel else 1.0
                self.ax_final.fill(poly[:, 0], poly[:, 1], color=fill_color, alpha=fill_alpha, linewidth=0)

            self.canvas.draw()
        
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
                btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 2px solid white; color: #FFF; font-weight: bold; border-radius: 4px;")
            else:
                btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #333; color: #000; border-radius: 4px;")
            btn.clicked.connect(lambda checked, e=eid: self.toggle_selection(e))
            self.palette_layout.addWidget(btn)
        self.palette_layout.addStretch()

    def toggle_selection(self, eid):
        if eid not in self.selected_edge_ids:
            self.selected_edge_ids.append(eid)
            if len(self.selected_edge_ids) > 2: self.selected_edge_ids.pop(0)
        else: self.selected_edge_ids.remove(eid)
        self.update_canvas(); self.update_palette()

    def on_click_canvas(self, event):
        if event.xdata and event.ydata and event.inaxes == self.ax_main:
            self.last_click_coord = (event.xdata, event.ydata); self.update_canvas()

    def on_pick(self, event):
        idx = event.artist.edge_idx
        self.toggle_selection(self.edges[idx]['id'])
        
    def on_key(self, event):
        key = event.key.lower() if event.key else ""
        if key == 'm': self.action_merge()
        elif key == 'd': self.action_delete()
        elif key == 'c': self.action_prune_parallel()
        elif key == 'b': self.action_breakpoint()
        elif key == 'a': self.action_add_dot()
        elif key == 'u': self.action_undo()
        elif key == 'r': self.action_reset_char()
        elif key == 'n': self.action_next_char()
        elif key == 'enter': self.action_complete_annotation()

    def save_state(self):
        self.history_stack.append(copy.deepcopy(self.edges))
        if len(self.history_stack) > 30: self.history_stack.pop(0)

    # 包含合并、断点、删除的动作逻辑完全沿用上一版...
    def action_merge(self):
        if len(self.selected_edge_ids) == 2:
            self.save_state()
            id_a, id_b = self.selected_edge_ids
            paths_to_stitch = [e['path'] for e in self.edges if e['id'] in (id_a, id_b)]
            new_unified_path = stitch_paths(paths_to_stitch)
            _, error = fit_bezier_basic_with_error(new_unified_path)
            
            sub_paths = []
            if error > MAX_BEZIER_ERROR:
                sub_paths = split_pixel_path_adaptively(new_unified_path, MAX_BEZIER_ERROR)
                QMessageBox.warning(self, "Fitting Alert", f"High deviation. Split into {len(sub_paths)} parts.")
            else: sub_paths = [new_unified_path]
            
            self.edges = [e for e in self.edges if e['id'] not in (id_a, id_b)]
            new_id = max([e['id'] for e in self.edges] + [0]) + 1
            for sp in sub_paths:
                self.edges.append({'id': new_id, 'path': sp})
                new_id += 1
            self.selected_edge_ids.clear(); self.bezier_cache.clear(); self.update_canvas(); self.update_palette()

    def action_delete(self):
        if self.selected_edge_ids:
            self.save_state()
            self.edges = [e for e in self.edges if e['id'] not in self.selected_edge_ids]
            self.selected_edge_ids.clear(); self.bezier_cache.clear(); self.update_canvas(); self.update_palette()

    def action_prune_parallel(self):
        if len(self.selected_edge_ids) == 2:
            self.save_state()
            id_a, id_b = self.selected_edge_ids
            edge_a = next(e for e in self.edges if e['id'] == id_a)
            edge_b = next(e for e in self.edges if e['id'] == id_b)
            len_a = np.sum(np.linalg.norm(np.diff(edge_a['path'], axis=0), axis=1))
            len_b = np.sum(np.linalg.norm(np.diff(edge_b['path'], axis=0), axis=1))
            self.edges = [e for e in self.edges if e['id'] != (id_a if len_a < len_b else id_b)]
            self.selected_edge_ids.clear(); self.bezier_cache.clear(); self.update_canvas(); self.update_palette()

    def action_breakpoint(self):
        if len(self.selected_edge_ids) == 1 and self.last_click_coord:
            self.save_state()
            target_id = self.selected_edge_ids[0]
            click_pt = np.array(self.last_click_coord)
            min_dist, best_i, best_split = float('inf'), -1, -1
            for i, edge in enumerate(self.edges):
                if edge['id'] == target_id:
                    dists = np.linalg.norm(edge['path'] - click_pt, axis=1)
                    idx = np.argmin(dists)
                    if dists[idx] < min_dist: min_dist, best_i, best_split = dists[idx], i, idx
            if best_i != -1 and 3 < best_split < len(self.edges[best_i]['path']) - 3:
                edge = self.edges[best_i]
                path1, path2 = edge['path'][:best_split+1], edge['path'][best_split:]
                new_id = max([e['id'] for e in self.edges] + [0]) + 1
                self.edges.pop(best_i)
                self.edges.append({'id': new_id, 'path': path1})
                self.edges.append({'id': new_id+1, 'path': path2})
                self.last_click_coord = None; self.selected_edge_ids.clear(); self.bezier_cache.clear(); self.update_canvas(); self.update_palette()

    def action_add_dot(self):
        if not self.last_click_coord: return
        cx, cy = self.last_click_coord; x, y = int(cx), int(cy)
        if not (0 <= x < CANVAS_SIZE and 0 <= y < CANVAS_SIZE): return
        self.save_state()
        self.edges.append({'id': max([e['id'] for e in self.edges] + [-1]) + 1, 'path': np.array([[cx, cy-1], [cx, cy], [cx, cy+1]])})
        self.last_click_coord = None; self.update_canvas(); self.update_palette()

    def action_undo(self):
        if len(self.history_stack) > 1:
            self.history_stack.pop() 
            self.edges = copy.deepcopy(self.history_stack[-1])
            self.selected_edge_ids.clear(); self.last_click_coord = None; self.bezier_cache.clear()
            self.update_canvas(); self.update_palette()

    def action_reset_char(self):
        self.load_char_topology()

    # --- 渲染缩略图逻辑 ---
    def refresh_gallery(self, grid_layout, key_list, mode):
        while grid_layout.count():
            item = grid_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        cols = 5
        for idx, hex_key in enumerate(key_list):
            char = chr(int(hex_key[2:], 16))
            card = QFrame()
            card.setStyleSheet("background-color: #FFFFFF; border: 1px solid #CCC; border-radius: 8px;")
            card_layout = QVBoxLayout(card)
            
            if f"{hex_key}_{mode}" not in self.thumbnail_cache:
                self.thumbnail_cache[f"{hex_key}_{mode}"] = self.generate_thumbnail(char, hex_key, mode)
                
            img_label = QLabel()
            img_label.setPixmap(self.thumbnail_cache[f"{hex_key}_{mode}"])
            img_label.setAlignment(Qt.AlignCenter)
            
            title = QLabel(f"Char: '{char}'\n({hex_key})")
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("font-weight: bold; border: none;")

            btn_action = QPushButton("Re-annotate" if mode == "completed" else "Unban")
            btn_action.setStyleSheet("background-color: #2196F3; color: white;" if mode == "completed" else "background-color: #4CAF50; color: white;")
            btn_action.clicked.connect(lambda checked, hk=hex_key: self.action_unban(hk) if mode == "banned" else self.action_reannotate(hk))

            card_layout.addWidget(img_label)
            card_layout.addWidget(title)
            card_layout.addWidget(btn_action)
            grid_layout.addWidget(card, idx // cols, idx % cols)

    def generate_thumbnail(self, char, hex_key, mode):
        fig = plt.figure(figsize=(2, 2), dpi=80)
        fig.patch.set_facecolor('#FFFFFF')
        binary = render_unicode_glyph(self.font_path, char, CANVAS_SIZE)
        ax = fig.add_subplot(111)
        ax.imshow(binary, cmap='gray')
        if mode == "completed" and hex_key in self.db.meta_data.get("raw_edges", {}):
            for e in self.db.meta_data["raw_edges"][hex_key]:
                path = np.array(e["path"])
                if len(path) > 1: ax.plot(path[:, 0], path[:, 1], linewidth=2, alpha=0.8)
        ax.axis('off')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        qimg = QImage()
        qimg.loadFromData(buf.getvalue())
        return QPixmap.fromImage(qimg)

    def action_unban(self, hex_key):
        if hex_key in self.db.meta_data["banned"]:
            self.db.meta_data["banned"].remove(hex_key)
            self.db.save_data()
            self.update_stats_display()
            self.switch_tab(2) 

    def action_reannotate(self, hex_key):
        self.char = chr(int(hex_key[2:], 16))
        self.title_label.setText(f"[RE-EDIT] Target: '{self.char}' ({hex_key})")
        self.binary = render_unicode_glyph(self.font_path, self.char, CANVAS_SIZE)
        self.dt_map = distance_transform_edt(self.binary)
        
        raw = self.db.meta_data["raw_edges"].get(hex_key, [])
        self.edges = [{"id": r["id"], "path": np.array(r["path"])} for r in raw]
        
        self.bezier_cache.clear()
        self.selected_edge_ids.clear()
        self.history_stack.clear()
        self.save_state()
        self.update_canvas()
        self.update_palette()
        self.switch_tab(0)

# ==========================================
# 🏠 左侧：主窗口与文件侧边栏
# ==========================================
class FontFactoryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Vector Router - Multi-Font Factory")
        self.setGeometry(50, 50, 1600, 850)
        
        # 核心分割器：左边是列表，右边是工作区
        self.splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.splitter)
        
        # --- 左侧边栏 (字体列表) ---
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        
        sidebar_title = QLabel("📂 Font Library")
        sidebar_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        sidebar_layout.addWidget(sidebar_title)
        
        self.font_list_widget = QListWidget()
        self.font_list_widget.itemClicked.connect(self.on_font_selected)
        sidebar_layout.addWidget(self.font_list_widget)
        
        btn_refresh = QPushButton("🔄 Refresh Fonts")
        btn_refresh.clicked.connect(self.load_font_list)
        btn_refresh.setStyleSheet("padding: 8px; background-color: #2196F3; color: white;")
        sidebar_layout.addWidget(btn_refresh)
        
        # --- 右侧容器 (用于装载 Workspace) ---
        self.workspace_container = QStackedWidget()
        
        # 默认占位符
        placeholder = QLabel("👈 Select a font from the sidebar to begin annotation.")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("font-size: 24px; color: #888;")
        self.workspace_container.addWidget(placeholder)
        
        self.splitter.addWidget(sidebar)
        self.splitter.addWidget(self.workspace_container)
        
        # 设置左右比例 1:5
        self.splitter.setSizes([250, 1350]) 
        
        self.load_font_list()

    def load_font_list(self):
        self.font_list_widget.clear()
        if not os.path.exists(FONTS_DIR): return
        
        for file in os.listdir(FONTS_DIR):
            if file.lower().endswith(('.ttf', '.otf')):
                self.font_list_widget.addItem(file)

    def on_font_selected(self, item):
        font_filename = item.text()
        full_font_path = os.path.join(FONTS_DIR, font_filename)
        
        # 如果已经有了工作区，清理掉旧的
        if self.workspace_container.count() > 1:
            old_widget = self.workspace_container.widget(1)
            self.workspace_container.removeWidget(old_widget)
            old_widget.deleteLater()
            # 🌟 核心修复：在销毁 UI 之前，彻底释放 Matplotlib 的底层 C++ 内存块
            plt.close(old_widget.fig)
            
        # 实例化新的工作区，并塞入右侧
        new_workspace = AnnotationWorkspace(full_font_path)
        self.workspace_container.addWidget(new_workspace)
        self.workspace_container.setCurrentIndex(1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    qdarktheme.setup_theme("light", custom_colors={"primary": "#2196F3"})
    
    window = FontFactoryApp()
    window.show()
    sys.exit(app.exec_())