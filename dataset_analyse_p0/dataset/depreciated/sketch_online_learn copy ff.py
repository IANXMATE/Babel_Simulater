import os
import sys
import random
import json
import copy
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image, ImageDraw, ImageFont
from skimage import morphology
from skan import Skeleton, summarize
from scipy.optimize import least_squares
from scipy.ndimage import distance_transform_edt, map_coordinates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFrame, QMessageBox, QScrollArea)
from PyQt5.QtCore import Qt
from fontTools.ttLib import TTFont

from skimage.filters import threshold_otsu
from skimage.morphology import medial_axis


warnings.filterwarnings("ignore")
mpl.rcParams['axes.unicode_minus'] = False 

# ==========================================
# ⚙️ Global Config & Paths
# ==========================================
CANVAS_SIZE = 400         
MAX_BEZIER_ERROR = 2   
MAX_SPUR_LENGTH = 20.0    

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(SCRIPT_DIR, './alien_tensors_raw/NotoSansTamil[wdth,wght].ttf')

if not os.path.exists(FONT_PATH):
    FONT_PATH = "arial.ttf"

# ==========================================
# 🧠 Active Learning Model
# ==========================================
class StrokeRouterMLP(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class ReplayBuffer:
    def __init__(self):
        self.features, self.labels = [], []
    def add(self, feature, label):
        self.features.append(feature)
        self.labels.append(label)

def extract_pairwise_features(path_a, path_b, dt_map):
    ends_a, ends_b = [path_a[0], path_a[-1]], [path_b[0], path_b[-1]]
    min_dist, best_a_idx, best_b_idx = float('inf'), 0, 0
    for i, ea in enumerate(ends_a):
        for j, eb in enumerate(ends_b):
            dist = np.linalg.norm(ea - eb)
            if dist < min_dist:
                min_dist, best_a_idx, best_b_idx = dist, i, j
                
    dist_feat = np.clip(min_dist / 10.0, 0, 1)
    step = min(4, len(path_a)-1, len(path_b)-1)
    if step < 1: step = 1
    
    vec_a = path_a[step] - path_a[0] if best_a_idx == 0 else path_a[-1-step] - path_a[-1]
    vec_b = path_b[step] - path_b[0] if best_b_idx == 0 else path_b[-1-step] - path_b[-1]
    norm_a, norm_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
    cos_theta = 0 if norm_a < 1e-5 or norm_b < 1e-5 else np.dot(vec_a, vec_b) / (norm_a * norm_b)
    
    len_ratio = min(len(path_a), len(path_b)) / max(len(path_a), len(path_b))
    w_a = dt_map[int(ends_a[best_a_idx][0]), int(ends_a[best_a_idx][1])]
    w_b = dt_map[int(ends_b[best_b_idx][0]), int(ends_b[best_b_idx][1])]
    w_ratio = min(w_a, w_b) / (max(w_a, w_b) + 1e-5)
    return [dist_feat, cos_theta, len_ratio, w_ratio]

# ==========================================
# 🎯 Vision & Geometry Engine
# ==========================================
def extract_all_real_chars(font_path):
    print(f"⏳ Parsing CMAP tables: {font_path}...")
    try: ttfont = TTFont(font_path)
    except Exception: return [chr(c) for c in range(33, 126)]
        
    valid_chars = set()
    for table in ttfont['cmap'].tables:
        for codepoint, glyph_name in table.cmap.items():
            char = chr(codepoint)
            if char.isprintable() and not char.isspace(): valid_chars.add(char)
                
    pil_font = ImageFont.truetype(font_path, int(CANVAS_SIZE * 0.8))
    final_chars = [c for c in valid_chars if pil_font.getbbox(c)]
    return final_chars if final_chars else [chr(c) for c in range(33, 126)]

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

def clean_graph_topology(G):
    """🌟 修复多条线挤在一起：消灭平行冗余边和微小自环"""
    # 1. 清除平行冗余边 (Parallel Edges) - 只保留两点间最长的一根线
    for u in list(G.nodes()):
        neighbors = list(G.neighbors(u))
        for v in set(neighbors):
            if G.number_of_edges(u, v) > 1:
                keys = list(G[u][v].keys())
                longest_k = max(keys, key=lambda k: len(G[u][v][k]['path']))
                for k in keys:
                    if k != longest_k:
                        G.remove_edge(u, v, key=k)
                        
    # 2. 清除微小自环 (Tiny Self-loops)
    for u in list(G.nodes()):
        if G.has_edge(u, u):
            keys = list(G[u][u].keys())
            for k in keys:
                if len(G[u][u][k]['path']) < 15:
                    G.remove_edge(u, u, key=k)
    return G

def prune_spurs(G, max_length):
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
            neighbor = v if u == n else u
            if length < max_length and G.degree(neighbor) >= 3:
                G.remove_edge(u, v, key=k)
                G.remove_node(n)
                changed = True
    return G

def stitch_paths(paths):
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
                if d < best_dist: best_dist, best_idx, best_orient = d, i, orient
                
        best_p = remaining.pop(best_idx)
        if best_orient == 0: stitched = np.vstack([stitched, best_p])
        elif best_orient == 1: stitched = np.vstack([stitched, best_p[::-1]])
        elif best_orient == 2: stitched = np.vstack([best_p[::-1], stitched])
        elif best_orient == 3: stitched = np.vstack([best_p, stitched])
    return stitched

def render_unicode_glyph(font_path, char):
    font = ImageFont.truetype(font_path, int(CANVAS_SIZE * 0.75))
    img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
    draw = ImageDraw.Draw(img)
    bbox = font.getbbox(char)
    if bbox is None: return np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((CANVAS_SIZE - w) / 2 - bbox[0], (CANVAS_SIZE - h) / 2 - bbox[1]), char, font=font, fill=0)
    return (np.array(img).astype(np.float32) / 255.0) < 0.5

def cubic_bezier_np(P, t):
    mt = 1 - t
    return mt**3 * P[0] + 3*mt**2*t * P[1] + 3*mt*t**2 * P[2] + t**3 * P[3]

def fit_bezier_basic_with_error(pixel_path):
    if len(pixel_path) < 2: return None, float('inf')
    P0, P3 = pixel_path[0], pixel_path[-1]
    t_vals = np.linspace(0, 1, len(pixel_path))[:, np.newaxis]
    def residuals(controls): return (cubic_bezier_np(np.array([P0, controls[0:2], controls[2:4], P3]), t_vals) - pixel_path).flatten()
    res = least_squares(residuals, x0=np.concatenate([P0 + (P3-P0)*0.33, P0 + (P3-P0)*0.66]))
    P_opt = np.array([P0, res.x[0:2], res.x[2:4], P3])
    error = np.mean(np.linalg.norm(cubic_bezier_np(P_opt, t_vals) - pixel_path, axis=1))
    return P_opt, error

def split_pixel_path_adaptively(pixel_path, max_error=MAX_BEZIER_ERROR):
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

# ==========================================
# 🖥️ Modern Annotation Application V11
# ==========================================
class ModernAnnotationApp(QMainWindow):
    def __init__(self, font_path):
        super().__init__()
        self.font_path = font_path
        self.setWindowTitle("AI Vector Router - Multi-Graph Pruning & Palette")
        self.setGeometry(50, 50, 1500, 750) 
        
        self.font_charset = extract_all_real_chars(self.font_path)
        self.history_stack = [] 
        self.selected_edge_ids = [] 
        self.last_click_coord = None 
        
        try: self.cmap = plt.colormaps['tab20']
        except AttributeError: self.cmap = plt.get_cmap('tab20')
        
        self.bezier_cache = {}
        self.model = StrokeRouterMLP()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.05)
        self.criterion = nn.BCELoss()
        self.buffer = ReplayBuffer()
        
        self.init_ui()
        self.action_next_char()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # ---------------- Left Panel ----------------
        control_panel = QVBoxLayout()
        control_panel.setSpacing(12)
        
        self.title_label = QLabel("Loading...")
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #1A1A1A;")
        control_panel.addWidget(self.title_label)
        
        self.stroke_info_label = QLabel("Beziers: 0")
        self.stroke_info_label.setStyleSheet("font-size: 15px; color: #E91E63; font-weight: bold;")
        control_panel.addWidget(self.stroke_info_label)
        
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        control_panel.addWidget(line)
        
        btn_merge = QPushButton("Merge Strokes (M)")
        btn_merge.clicked.connect(self.action_merge)
        btn_merge.setStyleSheet("padding: 10px; background-color: #4CAF50; color: white; font-weight: bold;")
        
        btn_delete = QPushButton("Delete Stroke (D)")
        btn_delete.clicked.connect(self.action_delete)
        btn_delete.setStyleSheet("padding: 10px; background-color: #F44336; color: white; font-weight: bold;")

        btn_break = QPushButton("Split at Point (B)")
        btn_break.clicked.connect(self.action_breakpoint)
        btn_break.setStyleSheet("padding: 10px; background-color: #FF9800; color: white; font-weight: bold;")

        btn_add_dot = QPushButton("Add Missing Dot (A)")
        btn_add_dot.clicked.connect(self.action_add_dot)
        btn_add_dot.setStyleSheet("padding: 10px; background-color: #00BCD4; color: white; font-weight: bold;")

        btn_undo = QPushButton("Undo Last Action (U)")
        btn_undo.clicked.connect(self.action_undo)
        btn_undo.setStyleSheet("padding: 10px; background-color: #757575; color: white; font-weight: bold;")

        btn_reset_char = QPushButton("Reset Current Char (R)")
        btn_reset_char.clicked.connect(self.action_reset_char)
        btn_reset_char.setStyleSheet("padding: 10px; background-color: #607D8B; color: white; font-weight: bold;")

        btn_next_char = QPushButton("Next Random Char (N)")
        btn_next_char.clicked.connect(self.action_next_char)
        btn_next_char.setStyleSheet("padding: 10px; background-color: #2196F3; color: white; font-weight: bold;")

        btn_export = QPushButton("Export JSON (Enter)")
        btn_export.clicked.connect(self.action_export)
        btn_export.setStyleSheet("padding: 14px; background-color: #673AB7; color: white; font-weight: bold; font-size: 13px;")

        control_panel.addWidget(btn_merge)
        control_panel.addWidget(btn_delete)
        control_panel.addWidget(btn_break)
        control_panel.addWidget(btn_add_dot)
        control_panel.addWidget(btn_undo)
        control_panel.addWidget(btn_reset_char)
        control_panel.addWidget(btn_next_char)
        control_panel.addStretch()
        control_panel.addWidget(btn_export)

        # ---------------- Right Panel ----------------
        right_panel = QVBoxLayout()
        
        # 🌟 NEW FEATURE: Interactive Palette Selector
        self.palette_container = QWidget()
        self.palette_layout = QHBoxLayout(self.palette_container)
        self.palette_layout.setContentsMargins(5, 5, 5, 5)
        self.palette_layout.setSpacing(8)
        self.palette_layout.setAlignment(Qt.AlignLeft)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.palette_container)
        scroll_area.setMaximumHeight(60) 
        scroll_area.setStyleSheet("border: 1px solid #E0E0E0; background-color: #FAFAFA; border-radius: 5px;")
        
        self.fig, (self.ax_ref, self.ax_main, self.ax_prev) = plt.subplots(1, 3, figsize=(18, 6))
        self.fig.patch.set_facecolor('#F5F5F5')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.mpl_connect('button_press_event', self.on_click_canvas)
        self.canvas.mpl_connect('key_press_event', self.on_key)

        right_panel.addWidget(scroll_area)
        right_panel.addWidget(self.canvas)

        layout.addLayout(control_panel, 1)
        layout.addLayout(right_panel, 5)

    def load_char_topology(self):
        # 🌟 核心升级：不再使用 skeletonize，改用 medial_axis (中轴线提取)
        # medial_axis 对复杂闭环结构有更好的拓扑保持能力
        skel, distance = medial_axis(self.binary, return_distance=True)
        
        # 将骨架转化为图形对象
        skel_obj = Skeleton(skel)
        branch_data = summarize(skel_obj)
        
        G = nx.MultiGraph()
        for index, row in branch_data.iterrows():
            coords = skel_obj.path_coordinates(index)
            if len(coords) > 2:
                path = np.column_stack([coords[:, 1], coords[:, 0]])
                src, dst = int(row['node-id-src']), int(row['node-id-dst'])
                # 保证路径连贯性
                if np.linalg.norm(path[0] - skel_obj.coordinates[src][::-1]) > 1.0: path = path[::-1]
                G.add_edge(src, dst, key=index, path=path)
                
        # 🌟 在这里进行更激进的拓扑平滑：先剪枝，再折叠
        G = prune_spurs(G, max_length=MAX_SPUR_LENGTH * 1.5) 
        G = collapse_degree2_nodes(G)
        
        self.edges = []
        self.bezier_cache.clear() 
        global_id = 0
        
        for u, v, k, d in G.edges(keys=True, data=True):
            path = d['path']
            sub_paths = split_pixel_path_adaptively(path)
            for sp in sub_paths:
                self.edges.append({'id': global_id, 'path': sp})
                global_id += 1
                            
        self.selected_edge_ids = []
        self.history_stack.clear()
        self.last_click_coord = None
        self.save_state()
        self.update_canvas()
        self.update_palette()

    def save_state(self):
        self.history_stack.append(copy.deepcopy(self.edges))
        if len(self.history_stack) > 30: self.history_stack.pop(0)

    # 🌟 NEW FEATURE: Update Palette Buttons
    def update_palette(self):
        # Clear existing palette
        while self.palette_layout.count():
            child = self.palette_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
            
        for edge in self.edges:
            eid = edge['id']
            c = self.cmap((eid % 20))
            r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
            
            btn = QPushButton(f" {eid} ")
            btn.setFixedSize(40, 30)
            btn.setCursor(Qt.PointingHandCursor)
            
            # Highlight border if selected
            if eid in self.selected_edge_ids:
                btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 3px solid #000; color: #FFF; font-weight: bold; border-radius: 4px;")
            else:
                btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #CCC; color: #000; border-radius: 4px;")
                
            btn.clicked.connect(lambda checked, e=eid: self.toggle_selection(e))
            self.palette_layout.addWidget(btn)
            
        self.palette_layout.addStretch()

    def update_canvas(self):
        self.ax_ref.clear()
        self.ax_ref.imshow(self.binary, cmap='gray')
        self.ax_ref.set_title("1. Original Matrix")
        self.ax_ref.axis('off')

        self.ax_main.clear()
        self.ax_main.imshow(self.binary, cmap='gray', alpha=0.15)
        self.stroke_info_label.setText(f"Bezier Strokes: {len(self.edges)}")
        
        for i, edge in enumerate(self.edges):
            eid = edge['id']
            is_sel = eid in self.selected_edge_ids
            color = self.cmap((eid % 20))
            lw = 6 if is_sel else 3
            alpha = 1.0 if is_sel else 0.6
            line, = self.ax_main.plot(edge['path'][:, 0], edge['path'][:, 1], 
                                      c=color, lw=lw, alpha=alpha, picker=5)
            line.edge_idx = i 
            
        if self.last_click_coord:
            self.ax_main.plot(self.last_click_coord[0], self.last_click_coord[1], 'rX', markersize=10)
            
        self.ax_main.set_title("2. Topology (Click & Edit)")
        self.ax_main.axis('off')

        self.ax_prev.clear()
        self.ax_prev.imshow(self.binary, cmap='gray', alpha=0.05) 
        self.ax_prev.set_title("3. Strict 1:1 Bezier Preview")
        self.ax_prev.axis('off')
        
        for edge in self.edges:
            eid = edge['id']
            is_sel = eid in self.selected_edge_ids
            color = self.cmap((eid % 20))
            
            if eid not in self.bezier_cache:
                p_opt, _ = fit_bezier_basic_with_error(edge['path'])
                w_opt = regress_width_dt_fast(p_opt, self.dt_map)
                self.bezier_cache[eid] = (p_opt, w_opt)
            
            p_opt, w_opt = self.bezier_cache[eid]
            mean_w = max(np.mean(w_opt), 1.0)
            ts = np.linspace(0, 1, 50)[:, None]
            curve = cubic_bezier_np(p_opt, ts)
            
            final_lw = mean_w * 2 * (1.5 if is_sel else 1.0)
            final_alpha = 0.9 if is_sel else 0.4
            
            self.ax_prev.plot(curve[:, 0], curve[:, 1], color=color, 
                              linewidth=final_lw, solid_capstyle='round', alpha=final_alpha)
            self.ax_prev.plot(curve[:, 0], curve[:, 1], color='black', linewidth=1.5, alpha=final_alpha+0.1)

        self.canvas.draw()

    # === Centralized Selection Engine ===
    def toggle_selection(self, eid):
        """🌟 Handle selection from both Canvas Clicks and Palette Clicks"""
        if eid not in self.selected_edge_ids:
            self.selected_edge_ids.append(eid)
            if len(self.selected_edge_ids) > 2: self.selected_edge_ids.pop(0)
        else:
            self.selected_edge_ids.remove(eid)
        self.update_canvas()
        self.update_palette()

    def on_click_canvas(self, event):
        if event.xdata and event.ydata and event.inaxes == self.ax_main:
            self.last_click_coord = (event.xdata, event.ydata)
            self.update_canvas()

    def on_pick(self, event):
        idx = event.artist.edge_idx
        clicked_eid = self.edges[idx]['id']
        self.toggle_selection(clicked_eid)
        
    def on_key(self, event):
        key = event.key.lower() if event.key else ""
        if key == 'm': self.action_merge()
        elif key == 'd': self.action_delete()
        elif key == 'b': self.action_breakpoint()
        elif key == 'a': self.action_add_dot()
        elif key == 'u': self.action_undo()
        elif key == 'r': self.action_reset_char()
        elif key == 'n': self.action_next_char()
        elif key == 'enter': self.action_export()

    # === Pipeline Execution Operations ===
    def action_merge(self):
        if len(self.selected_edge_ids) == 2:
            self.save_state()
            id_a, id_b = self.selected_edge_ids
            paths_to_stitch = [e['path'] for e in self.edges if e['id'] in (id_a, id_b)]
            
            if len(paths_to_stitch) >= 2:
                features = extract_pairwise_features(paths_to_stitch[0], paths_to_stitch[1], self.dt_map)
                self.buffer.add(features, 1.0)
            
            new_unified_path = stitch_paths(paths_to_stitch)
            _, error = fit_bezier_basic_with_error(new_unified_path)
            
            sub_paths = []
            if error > MAX_BEZIER_ERROR:
                sub_paths = split_pixel_path_adaptively(new_unified_path)
                QMessageBox.warning(self, "Fitting Warning", f"Deviation {error:.2f} > Limit. Split into {len(sub_paths)} pure Beziers.")
            else:
                sub_paths = [new_unified_path]
            
            self.edges = [e for e in self.edges if e['id'] not in (id_a, id_b)]
            new_id = max([e['id'] for e in self.edges] + [0]) + 1
            for sp in sub_paths:
                self.edges.append({'id': new_id, 'path': sp})
                new_id += 1
                
            self.selected_edge_ids.clear()
            self.bezier_cache.clear() 
            self.update_canvas()
            self.update_palette()

    def action_delete(self):
        if self.selected_edge_ids:
            self.save_state()
            self.edges = [e for e in self.edges if e['id'] not in self.selected_edge_ids]
            self.selected_edge_ids.clear()
            self.bezier_cache.clear()
            self.update_canvas()
            self.update_palette()

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
                
                self.last_click_coord = None
                self.selected_edge_ids.clear()
                self.bezier_cache.clear()
                self.update_canvas()
                self.update_palette()

    def action_add_dot(self):
        if not self.last_click_coord:
            QMessageBox.warning(self, "Warning", "Please click on the canvas first.")
            return

        cx, cy = self.last_click_coord
        x, y = int(cx), int(cy)

        if not (0 <= x < CANVAS_SIZE and 0 <= y < CANVAS_SIZE): return
        y_min, y_max = max(0, y-2), min(CANVAS_SIZE, y+3)
        x_min, x_max = max(0, x-2), min(CANVAS_SIZE, x+3)
        if not np.any(self.binary[y_min:y_max, x_min:x_max]):
            QMessageBox.critical(self, "Failed", "Clicked outside ink area! Invalid dot.")
            return

        self.save_state()
        tiny_path = np.array([[cx, cy-1], [cx, cy], [cx, cy+1]])
        new_id = max([e['id'] for e in self.edges] + [-1]) + 1
        self.edges.append({'id': new_id, 'path': tiny_path})

        self.last_click_coord = None
        self.update_canvas()
        self.update_palette()

    def action_undo(self):
        if len(self.history_stack) > 1:
            self.history_stack.pop() 
            self.edges = copy.deepcopy(self.history_stack[-1])
            self.selected_edge_ids.clear()
            self.last_click_coord = None
            self.bezier_cache.clear()
            self.update_canvas()
            self.update_palette()

    def action_reset_char(self):
        self.load_char_topology()

    def action_next_char(self):
        max_attempts = 200
        valid_char_found = False
        
        for _ in range(max_attempts):
            candidate = random.choice(self.font_charset)
            b_mask = render_unicode_glyph(self.font_path, candidate)
            
            if not np.any(b_mask): continue
            skel = morphology.skeletonize(b_mask)
            if not np.any(skel): continue 
                
            try:
                skel_obj = Skeleton(skel)
                branch_data = summarize(skel_obj)
                if len(branch_data) == 0: continue
            except ValueError:
                continue
                
            self.char, self.binary = candidate, b_mask
            valid_char_found = True
            break
            
        if not valid_char_found: return

        unicode_hex = f"U+{ord(self.char):04X}"
        self.title_label.setText(f"Target Char: '{self.char}' ({unicode_hex})")
        self.dt_map = distance_transform_edt(self.binary)
        self.load_char_topology()

    def action_export(self):
        final_tokens = []
        for edge in self.edges:
            eid = edge['id']
            if eid in self.bezier_cache:
                p_opt, w_opt = self.bezier_cache[eid]
                final_tokens.append({
                    "char": self.char, "bezier_id": eid,
                    "mother_bezier": p_opt.tolist(), "width_bezier": w_opt
                })
        print(json.dumps(final_tokens, indent=2))
        QMessageBox.information(self, "Success", f"Exported {len(final_tokens)} pure Beziers data!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ModernAnnotationApp(FONT_PATH)
    window.show()
    sys.exit(app.exec_())