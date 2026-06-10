import os
import sys
import copy
import io
import numpy as np
import qdarktheme
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.spatial import distance

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QScrollArea, 
                             QStackedWidget, QGridLayout, QListWidget, QSplitter, QFrame)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from data_manager import DatasetManager
from geometry_vision import (render_unicode_glyph, fit_bezier_basic_with_error, 
                             regress_width_dt_fast, cubic_bezier_np)

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR = os.path.join(SCRIPT_DIR, "../dataset/alien_tensors_raw")
CANVAS_SIZE = 400

# ==========================================
# 🧠 核心工具：通用数据适配器
# ==========================================
def extract_bezier_and_width(stroke, dt_map=None):
    """自适应提取贝塞尔参数，兼容已完成的标注和纯原图像素路径"""
    if 'mother_bezier' in stroke:
        mb = np.array(stroke['mother_bezier'])
        w = np.array(stroke.get('width_bezier', [3.0, 3.0, 3.0, 3.0]))
        return mb, w
    elif 'path' in stroke:
        path = np.array(stroke['path'])
        if len(path) < 2: return None, None
        p_opt, _ = fit_bezier_basic_with_error(path)
        w_opt = regress_width_dt_fast(p_opt, dt_map) if dt_map is not None else np.array([3.0]*4)
        return p_opt, w_opt
    return None, None

def set_endpoint(stroke, is_start, new_pos):
    """自适应覆写端点"""
    if 'mother_bezier' in stroke:
        stroke['mother_bezier'][0 if is_start else 3] = new_pos.tolist()
    if 'path' in stroke:
        stroke['path'][0 if is_start else -1] = new_pos.tolist()

# ==========================================
# 🧠 核心算法：拓扑优化与图论分类器
# ==========================================
class TopologyOptimizer:
    def __init__(self, dt_map):
        self.dt_map = dt_map
        
    def process(self, strokes):
        if not strokes: return [], []
        snapped_strokes = self._snap_endpoints(strokes)
        ordered_strokes, classes = self._reorder_and_classify(snapped_strokes)
        return ordered_strokes, classes
        
    def _snap_endpoints(self, strokes):
        """规则 1：动态宽度阈值吸附，处理相交端点"""
        snapped = copy.deepcopy(strokes)
        eps = []
        for i, s in enumerate(snapped):
            mb, _ = extract_bezier_and_width(s)
            if mb is None: continue
            eps.append({'s_idx': i, 'is_start': True, 'pos': mb[0]})
            eps.append({'s_idx': i, 'is_start': False, 'pos': mb[3]})
            
        clusters = []
        for ep in eps:
            placed = False
            y, x = int(np.clip(ep['pos'][1], 0, self.dt_map.shape[0]-1)), int(np.clip(ep['pos'][0], 0, self.dt_map.shape[1]-1))
            w_ep = self.dt_map[y, x]
            
            for c in clusters:
                dist = np.linalg.norm(ep['pos'] - c['center'])
                # 如果两点距离 < 两者线宽之和的 1.2 倍，强制吸附
                if dist < (w_ep + c['avg_w']) * 1.2 + 2.0: 
                    c['eps'].append(ep)
                    c['center'] = np.mean([e['pos'] for e in c['eps']], axis=0)
                    c['avg_w'] = (c['avg_w'] * (len(c['eps'])-1) + w_ep) / len(c['eps'])
                    placed = True
                    break
            if not placed:
                clusters.append({'center': ep['pos'], 'eps': [ep], 'avg_w': w_ep})
                
        # 覆写吸附后的中心坐标
        for c in clusters:
            if len(c['eps']) > 1:
                for ep in c['eps']:
                    set_endpoint(snapped[ep['s_idx']], ep['is_start'], c['center'])
        return snapped

    def _reorder_and_classify(self, strokes):
        """规则 2 & 3：图论拓扑遍历与分类"""
        def pt2key(pt): return (round(pt[0], 1), round(pt[1], 1))
        
        node_to_strokes = {}
        for i, s in enumerate(strokes):
            mb, _ = extract_bezier_and_width(s)
            if mb is None: continue
            k1, k2 = pt2key(mb[0]), pt2key(mb[3])
            node_to_strokes.setdefault(k1, set()).add(i)
            node_to_strokes.setdefault(k2, set()).add(i)
            
        adj = {i: set() for i in range(len(strokes))}
        for s_set in node_to_strokes.values():
            for s1 in s_set:
                for s2 in s_set:
                    if s1 != s2: adj[s1].add(s2)
                    
        # 1. DFS 寻找环结构
        visited, in_cycle = set(), set()
        
        def find_cycles(u, parent, path_stack):
            visited.add(u); path_stack.append(u)
            for v in adj[u]:
                if v == parent: continue
                if v in visited:
                    idx = path_stack.index(v) if v in path_stack else 0
                    for cycle_node in path_stack[idx:]: in_cycle.add(cycle_node)
                else: find_cycles(v, u, path_stack)
            path_stack.pop()
            
        for i in range(len(strokes)):
            if i not in visited: find_cycles(i, -1, [])
                
        # 2. 剥离链状结构
        non_cycle = set(range(len(strokes))) - in_cycle
        sub_node_strokes = {}
        for i in non_cycle:
            mb, _ = extract_bezier_and_width(strokes[i])
            k1, k2 = pt2key(mb[0]), pt2key(mb[3])
            sub_node_strokes.setdefault(k1, set()).add(i)
            sub_node_strokes.setdefault(k2, set()).add(i)
            
        terminals = set()
        for s_set in sub_node_strokes.values():
            if len(s_set) != 2:
                for s in s_set: terminals.add(s)
                
        paths, visited_nc = [], set()
        for start_s in terminals:
            if start_s in visited_nc: continue
            chain = [start_s]; visited_nc.add(start_s); curr = start_s
            while True:
                mb, _ = extract_bezier_and_width(strokes[curr])
                k1, k2 = pt2key(mb[0]), pt2key(mb[3])
                next_s = None
                for k in (k1, k2):
                    if k in sub_node_strokes and len(sub_node_strokes[k]) == 2:
                        neighbors = list(sub_node_strokes[k])
                        candidate = neighbors[0] if neighbors[1] == curr else neighbors[1]
                        if candidate not in visited_nc:
                            next_s = candidate; break
                if next_s is not None:
                    chain.append(next_s); visited_nc.add(next_s); curr = next_s
                else: break
            paths.append(chain)
            
        for i in non_cycle:
            if i not in visited_nc:
                paths.append([i]); visited_nc.add(i)

        # 3. 相交判定与分类
        final_strokes, final_classes = [], []
        for i in in_cycle:
            final_strokes.append(strokes[i]); final_classes.append("Blue")
            
        def get_intersect_type(curr_idx, prev_idx):
            mb_curr, _ = extract_bezier_and_width(strokes[curr_idx])
            mb_prev, _ = extract_bezier_and_width(strokes[prev_idx])
            # 生成平滑贝塞尔曲线散点用作高精度几何碰撞检测
            p_curr = cubic_bezier_np(mb_curr, np.linspace(0, 1, 20)[:, None])
            p_prev = cubic_bezier_np(mb_prev, np.linspace(0, 1, 20)[:, None])
            
            d1 = np.min(np.linalg.norm(p_prev[1:-1] - p_curr[0], axis=1))
            d2 = np.min(np.linalg.norm(p_prev[1:-1] - p_curr[-1], axis=1))
            if d1 < 5.0 or d2 < 5.0: return "Purple" # T型相交
            
            dists = distance.cdist(p_curr[1:-1], p_prev[1:-1])
            if np.min(dists) < 5.0: return "Brown" # X型横穿
            return "None"

        last_s_idx = list(in_cycle)[-1] if in_cycle else None
            
        for chain in paths:
            for i, s_idx in enumerate(chain):
                cls = "Green" # 默认起点
                if len(chain) > 1:
                    if i == len(chain) - 1: cls = "Red"
                    elif i > 0: cls = "Yellow"
                
                if cls == "Green" and last_s_idx is not None:
                    intersect = get_intersect_type(s_idx, last_s_idx)
                    if intersect != "None": cls = intersect
                        
                final_strokes.append(strokes[s_idx])
                final_classes.append(cls)
            last_s_idx = chain[-1]
            
        return final_strokes, final_classes

# ==========================================
# 🎨 离线渲染引擎 (完全对齐 main.py 的视觉逻辑)
# ==========================================
class StateRenderer:
    def __init__(self):
        try: self.cmap = plt.colormaps['tab20']
        except AttributeError: self.cmap = plt.get_cmap('tab20')

    def render_top_image(self, binary, strokes, dt_map=None, size=(3.5, 3.5)):
        """对应 main.py 中的图 3: 彩色平滑贝塞尔与黑色骨架线"""
        fig = plt.figure(figsize=size, dpi=90)
        fig.patch.set_facecolor('#FFFFFF')
        ax = fig.add_subplot(111)
        ax.imshow(binary, cmap='gray', alpha=0.05) # 使用较淡背景
        
        for i, e in enumerate(strokes):
            p_opt, w_opt = extract_bezier_and_width(e, dt_map)
            if p_opt is None: continue
            
            color = self.cmap((i % 20))
            mean_w = max(np.mean(w_opt), 1.0)
            ts = np.linspace(0, 1, 50)[:, None]
            curve = cubic_bezier_np(p_opt, ts)
            
            # 对齐 main.py 粗体彩色曲线与内部骨架
            ax.plot(curve[:, 0], curve[:, 1], c=color, lw=mean_w * 2, solid_capstyle='round', alpha=0.6)
            ax.plot(curve[:, 0], curve[:, 1], color='black', linewidth=1.5, alpha=0.5)
            
        ax.axis('off')
        return self._fig_to_pixmap(fig)

    def render_bottom_image(self, binary, strokes, dt_map, size=(3.5, 3.5)):
        """对应 main.py 中的图 4: 高精度多边形 TTF 挤出渲染"""
        fig = plt.figure(figsize=size, dpi=90)
        fig.patch.set_facecolor('#FFFFFF')
        ax = fig.add_subplot(111)
        ax.imshow(np.ones_like(binary), cmap='gray', vmin=0, vmax=1) 
        
        for e in strokes:
            p_opt, w_opt = extract_bezier_and_width(e, dt_map)
            if p_opt is None: continue
            
            # 高密度 100 采样点完美还原物理法线
            ts_dense = np.linspace(0, 1, 100)[:, None]
            curve_dense = cubic_bezier_np(p_opt, ts_dense)
            mt = 1 - ts_dense
            w_vals = mt**3 * w_opt[0] + 3*mt**2*ts_dense * w_opt[1] + 3*mt*ts_dense**2 * w_opt[2] + ts_dense**3 * w_opt[3]
            
            dp = np.gradient(curve_dense, axis=0)
            n = np.zeros_like(dp)
            n[:, 0], n[:, 1] = -dp[:, 1], dp[:, 0]
            n_norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-5
            n = n / n_norm
            
            upper = curve_dense + n * w_vals
            lower = curve_dense - n * w_vals
            poly = np.vstack([upper, lower[::-1]])
            ax.fill(poly[:, 0], poly[:, 1], color='#000000', alpha=0.85, linewidth=0)
            
        ax.axis('off')
        return self._fig_to_pixmap(fig)

    def render_classified_image(self, binary, strokes, classes, size=(3.5, 3.5)):
        """状态机优化图谱：基于完美贝塞尔曲线着色"""
        fig = plt.figure(figsize=size, dpi=90)
        fig.patch.set_facecolor('#FFFFFF')
        ax = fig.add_subplot(111)
        ax.imshow(binary, cmap='gray', alpha=0.05)
        
        color_map = {
            "Blue": "#2196F3",   # 环状
            "Green": "#4CAF50",  # 连续起点
            "Yellow": "#FFEB3B", # 中间笔
            "Red": "#F44336",    # 终笔
            "Purple": "#9C27B0", # T型起点
            "Brown": "#795548"   # X型起点
        }
        
        for i, (e, cls) in enumerate(zip(strokes, classes)):
            p_opt, _ = extract_bezier_and_width(e)
            if p_opt is None: continue
            
            color = color_map.get(cls, "#000000")
            ts = np.linspace(0, 1, 50)[:, None]
            curve = cubic_bezier_np(p_opt, ts)
            
            # 画线
            ax.plot(curve[:, 0], curve[:, 1], c=color, lw=5, alpha=0.8, solid_capstyle='round')
            
            # 方向箭头
            dir_vec = curve[-1] - curve[-2]
            norm = np.linalg.norm(dir_vec)
            if norm > 0:
                dir_vec = dir_vec / norm * 12
                ax.arrow(curve[-2, 0], curve[-2, 1], dir_vec[0], dir_vec[1], 
                         head_width=6, head_length=8, fc=color, ec=color)
                
            # 数字标记
            ax.text(curve[len(curve)//2, 0], curve[len(curve)//2, 1], str(i+1), 
                    color='white', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
                        
        ax.axis('off')
        return self._fig_to_pixmap(fig)

    def render_thumbnail(self, binary, size=(2.5, 2.5)):
        fig = plt.figure(figsize=size, dpi=80)
        fig.patch.set_facecolor('#FFFFFF')
        ax = fig.add_subplot(111)
        ax.imshow(binary, cmap='gray')
        ax.axis('off')
        return self._fig_to_pixmap(fig)

    def _fig_to_pixmap(self, fig):
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        qimg = QImage()
        qimg.loadFromData(buf.getvalue())
        return QPixmap.fromImage(qimg)

# ==========================================
# 🏠 后续 UI 及主程序保持不变...
# ==========================================
class PreviewWorkspace(QWidget):
    def __init__(self, font_path):
        super().__init__()
        self.font_path = font_path
        self.font_filename = os.path.basename(font_path).split('.')[0]
        self.db = DatasetManager(SCRIPT_DIR, self.font_filename)
        self.renderer = StateRenderer()
        self.thumbnail_cache = {}
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.stacked = QStackedWidget()
        self.page_gallery = QWidget()
        gal_layout = QVBoxLayout(self.page_gallery)
        title = QLabel(f"✅ Annotated Characters: {self.font_filename}")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #333;")
        gal_layout.addWidget(title)
        
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        container = QWidget(); self.grid = QGridLayout(container)
        scroll.setWidget(container)
        gal_layout.addWidget(scroll)
        
        self.page_detail = QWidget()
        self.detail_layout = QVBoxLayout(self.page_detail)
        
        self.stacked.addWidget(self.page_gallery)
        self.stacked.addWidget(self.page_detail)
        self.layout.addWidget(self.stacked)
        self.load_gallery()

    def load_gallery(self):
        completed_keys = list(self.db.annotated_outlines.keys())
        cols = 5
        for idx, hex_key in enumerate(completed_keys):
            char = chr(int(hex_key[2:], 16))
            card = QFrame()
            card.setStyleSheet("background-color: #FFFFFF; border: 2px solid #E0E0E0; border-radius: 8px;")
            card_layout = QVBoxLayout(card)
            
            if hex_key not in self.thumbnail_cache:
                binary = render_unicode_glyph(self.font_path, char, CANVAS_SIZE)
                self.thumbnail_cache[hex_key] = self.renderer.render_thumbnail(binary)
                
            img_label = QLabel()
            img_label.setPixmap(self.thumbnail_cache[hex_key])
            img_label.setAlignment(Qt.AlignCenter)
            
            title = QLabel(f"'{char}'\n{hex_key}")
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("font-size: 16px; font-weight: bold; color: #424242; border: none;")
            
            btn_view = QPushButton("🔍 Inspect Topology")
            btn_view.setCursor(Qt.PointingHandCursor)
            btn_view.setStyleSheet("background-color: #2196F3; color: white; padding: 8px; border-radius: 4px; font-weight: bold;")
            btn_view.clicked.connect(lambda checked, hk=hex_key: self.open_detail(hk))
            
            card_layout.addWidget(img_label)
            card_layout.addWidget(title)
            card_layout.addWidget(btn_view)
            self.grid.addWidget(card, idx // cols, idx % cols)

    def open_detail(self, hex_key):
        while self.detail_layout.count():
            item = self.detail_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
            
        char = chr(int(hex_key[2:], 16))
        # 直接读取完美标注数据，无需额外转义
        raw_strokes = copy.deepcopy(self.db.annotated_outlines.get(hex_key, []))
        
        top_bar = QHBoxLayout()
        btn_back = QPushButton("🔙 Back")
        btn_back.setFixedWidth(100)
        btn_back.setStyleSheet("padding: 10px; background-color: #607D8B; color: white; font-weight: bold; border-radius: 4px;")
        btn_back.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        top_bar.addWidget(btn_back)
        
        lbl_title = QLabel(f"Topology Analysis: '{char}' ({hex_key})")
        lbl_title.setStyleSheet("font-size: 22px; font-weight: bold;")
        top_bar.addWidget(lbl_title)
        top_bar.addStretch()
        self.detail_layout.addLayout(top_bar)
        
        line = QFrame(); line.setFrameShape(QFrame.HLine); self.detail_layout.addWidget(line)

        binary = render_unicode_glyph(self.font_path, char, CANVAS_SIZE)
        dt_map = distance_transform_edt(binary)
        
        optimizer = TopologyOptimizer(dt_map)
        ordered_strokes, classes = optimizer.process(raw_strokes)

        grid_view = QGridLayout()
        grid_view.setSpacing(20)
        
        def create_view_card(title_html, pixmap):
            container = QFrame()
            container.setStyleSheet("background-color: white; border: 1px solid #E0E0E0; border-radius: 8px;")
            l = QVBoxLayout(container)
            lbl_title = QLabel(title_html)
            lbl_title.setAlignment(Qt.AlignCenter)
            lbl_title.setStyleSheet("border: none; margin-top: 5px;")
            
            lbl_img = QLabel()
            lbl_img.setPixmap(pixmap)
            lbl_img.setAlignment(Qt.AlignCenter)
            lbl_img.setStyleSheet("border: none;")
            
            l.addWidget(lbl_title)
            l.addWidget(lbl_img)
            return container

        card1 = create_view_card("<b>1. Original Binary Mask</b>", self.renderer.render_thumbnail(binary, size=(3.5, 3.5)))
        
        # 传入 dt_map 确保宽度完全还原
        card2 = create_view_card("<b>2. Raw Annotated Strokes</b><br><span style='color:gray'>Without endpoint snapping</span>", 
                                 self.renderer.render_top_image(binary, raw_strokes, dt_map, size=(3.5, 3.5)))
        
        card3 = create_view_card("<b>3. Reconstructed BW Glyph</b>", 
                                 self.renderer.render_bottom_image(binary, raw_strokes, dt_map, size=(3.5, 3.5)))
        
        legend = "<span style='color:#2196F3'>Blue:Loop</span> | <span style='color:#4CAF50'>Grn:Start</span> | <span style='color:#FFEB3B'>Ylw:Mid</span> | <span style='color:#F44336'>Red:End</span> | <span style='color:#9C27B0'>Pur:T-Intersect</span> | <span style='color:#795548'>Brn:X-Intersect</span>"
        card4 = create_view_card(f"<b>4. State-Machine Topology Optimized</b><br><span style='font-size:11px;'>{legend}</span>", 
                                 self.renderer.render_classified_image(binary, ordered_strokes, classes, size=(3.5, 3.5)))
        card4.setStyleSheet("background-color: #FAFAFA; border: 2px solid #2196F3; border-radius: 8px;") 

        grid_view.addWidget(card1, 0, 0)
        grid_view.addWidget(card2, 0, 1)
        grid_view.addWidget(card3, 1, 0)
        grid_view.addWidget(card4, 1, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: transparent; border: none;")
        grid_widget = QWidget()
        grid_widget.setLayout(grid_view)
        scroll.setWidget(grid_widget)
        
        self.detail_layout.addWidget(scroll)
        self.stacked.setCurrentIndex(1)

class PreviewAppMain(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Font Topology Inspector")
        self.setGeometry(50, 50, 1500, 950)
        
        self.splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.splitter)
        
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        
        lbl_annotated = QLabel("✅ Annotated Fonts")
        lbl_annotated.setStyleSheet("font-size: 15px; font-weight: bold; color: #4CAF50;")
        sidebar_layout.addWidget(lbl_annotated)
        
        self.list_annotated = QListWidget()
        self.list_annotated.itemClicked.connect(self.on_font_selected)
        self.list_annotated.setStyleSheet("background-color: #F1F8E9; border: 1px solid #C8E6C9; border-radius: 4px;")
        sidebar_layout.addWidget(self.list_annotated)
        
        lbl_unannotated = QLabel("⏳ Pending Fonts")
        lbl_unannotated.setStyleSheet("font-size: 15px; font-weight: bold; color: #FF9800; margin-top: 10px;")
        sidebar_layout.addWidget(lbl_unannotated)
        
        self.list_unannotated = QListWidget()
        self.list_unannotated.itemClicked.connect(self.on_font_selected)
        self.list_unannotated.setStyleSheet("background-color: #FFF3E0; border: 1px solid #FFE0B2; border-radius: 4px;")
        sidebar_layout.addWidget(self.list_unannotated)

        self.workspace_stack = QStackedWidget()
        self.workspace_stack.addWidget(QLabel("👈 Select a font to begin topology inspection..."))
        
        self.splitter.addWidget(sidebar)
        self.splitter.addWidget(self.workspace_stack)
        self.splitter.setSizes([250, 1250])
        
        self.load_font_list()

    def load_font_list(self):
        self.list_annotated.clear()
        self.list_unannotated.clear()
        
        if not os.path.exists(FONTS_DIR): return
        anno_dir = os.path.join(SCRIPT_DIR, "annotations")
        os.makedirs(anno_dir, exist_ok=True)

        for file in os.listdir(FONTS_DIR):
            if file.lower().endswith(('.ttf', '.otf')):
                font_filename = os.path.splitext(file)[0]
                anno_file_path = os.path.join(anno_dir, f"{font_filename}.json")
                is_annotated = False
                if os.path.exists(anno_file_path):
                    try:
                        import json
                        with open(anno_file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if len(data) > 0: is_annotated = True
                    except: pass
                
                if is_annotated: self.list_annotated.addItem(file)
                else: self.list_unannotated.addItem(file)

    def on_font_selected(self, item):
        sender = self.sender()
        if sender == self.list_annotated: self.list_unannotated.clearSelection()
        elif sender == self.list_unannotated: self.list_annotated.clearSelection()

        font_path = os.path.join(FONTS_DIR, item.text())
        if self.workspace_stack.count() > 1:
            old = self.workspace_stack.widget(1)
            self.workspace_stack.removeWidget(old)
            old.deleteLater()
            
        new_ws = PreviewWorkspace(font_path)
        self.workspace_stack.addWidget(new_ws)
        self.workspace_stack.setCurrentIndex(1)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("light", custom_colors={"primary": "#FF9800"})
    window = PreviewAppMain()
    window.show()
    sys.exit(app.exec_())