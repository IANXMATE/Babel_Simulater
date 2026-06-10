import os
import sys
import pickle
import io
import numpy as np
import qdarktheme
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QScrollArea, 
                             QStackedWidget, QGridLayout, QListWidget, QSplitter, QFrame)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# 引入基础视觉库中的字形渲染器 (用于对照真实的 TTF)
try:
    from geometry_vision import render_unicode_glyph
except ImportError:
    from PIL import Image, ImageDraw, ImageFont
    def render_unicode_glyph(font_path, char, canvas_size=400):
        img = Image.new('L', (canvas_size, canvas_size), 0)
        draw = ImageDraw.Draw(img)
        try:
            pyfont = ImageFont.truetype(font_path, canvas_size * 0.8)
            w, h = draw.textbbox((0, 0), char, font=pyfont)[2:]
            draw.text(((canvas_size-w)/2, (canvas_size-h)/2 - h*0.1), char, fill=255, font=pyfont)
        except: pass
        return np.array(img, dtype=np.float32) / 255.0

def cubic_bezier_np(p, t):
    mt = 1 - t
    return (mt**3 * p[0] + 3 * mt**2 * t * p[1] + 3 * mt * t**2 * p[2] + t**3 * p[3])

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR = os.path.join(SCRIPT_DIR, "../dataset/alien_tensors_raw")
PKL_PATH = os.path.join(SCRIPT_DIR, "preprocessed_topology_dataset.pkl")
CANVAS_SIZE = 400

# ==========================================
# 🎨 离线渲染引擎 (极速版：直接读取 PKL 数据)
# ==========================================
class StateRenderer:
    def __init__(self):
        try: self.cmap = plt.colormaps['tab20']
        except AttributeError: self.cmap = plt.get_cmap('tab20')

    def _extract(self, stroke):
        mb = np.array(stroke['mother_bezier'])
        w = np.array(stroke.get('width_bezier', [6.0]*4))
        return mb, w

    def render_top_image(self, binary, strokes, size=(3.5, 3.5)):
        """图2：基础贝塞尔骨架着色"""
        fig = plt.figure(figsize=size, dpi=90)
        fig.patch.set_facecolor('#FFFFFF')
        ax = fig.add_subplot(111)
        ax.imshow(binary, cmap='gray', alpha=0.05) 
        
        for i, e in enumerate(strokes):
            p_opt, w_opt = self._extract(e)
            color = self.cmap((i % 20))
            mean_w = max(np.mean(w_opt), 1.0)
            ts = np.linspace(0, 1, 50)[:, None]
            curve = cubic_bezier_np(p_opt, ts)
            
            ax.plot(curve[:, 0], curve[:, 1], c=color, lw=mean_w * 2, solid_capstyle='round', alpha=0.6)
            ax.plot(curve[:, 0], curve[:, 1], color='black', linewidth=1.5, alpha=0.5)
            
        ax.axis('off')
        return self._fig_to_pixmap(fig)

    def render_bottom_image(self, binary, strokes, size=(3.5, 3.5)):
        """图3：完美多边形挤出重建"""
        fig = plt.figure(figsize=size, dpi=90)
        fig.patch.set_facecolor('#FFFFFF')
        ax = fig.add_subplot(111)
        ax.imshow(np.ones_like(binary), cmap='gray', vmin=0, vmax=1) 
        
        for e in strokes:
            p_opt, w_opt = self._extract(e)
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

    def render_classified_image(self, binary, strokes, size=(3.5, 3.5)):
        """图4：拓扑状态机可视化 (依据 PKL 的 topo_state)"""
        fig = plt.figure(figsize=size, dpi=90)
        fig.patch.set_facecolor('#FFFFFF')
        ax = fig.add_subplot(111)
        ax.imshow(binary, cmap='gray', alpha=0.05)
        
        color_map = {
            1: "#2196F3",   # Blue: 环状
            2: "#4CAF50",   # Green: 连续起点
            3: "#FFEB3B",   # Yellow: 中间笔
            4: "#F44336",   # Red: 终笔
            5: "#9C27B0",   # Purple: T型相交起点
            6: "#795548"    # Brown: X型交叉起点
        }
        
        for i, e in enumerate(strokes):
            p_opt, _ = self._extract(e)
            state = e.get('topo_state', 2) # 获取 step1_5 计算好的状态
            
            color = color_map.get(state, "#000000")
            ts = np.linspace(0, 1, 50)[:, None]
            curve = cubic_bezier_np(p_opt, ts)
            
            ax.plot(curve[:, 0], curve[:, 1], c=color, lw=5, alpha=0.8, solid_capstyle='round')
            
            dir_vec = curve[-1] - curve[-2]
            norm = np.linalg.norm(dir_vec)
            if norm > 0:
                dir_vec = dir_vec / norm * 12
                ax.arrow(curve[-2, 0], curve[-2, 1], dir_vec[0], dir_vec[1], 
                         head_width=6, head_length=8, fc=color, ec=color)
                
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
# 🏠 工作区：画廊与四宫格详情
# ==========================================
class PreviewWorkspace(QWidget):
    def __init__(self, font_path, dataset):
        super().__init__()
        self.font_path = font_path
        self.font_filename = os.path.basename(font_path).split('.')[0]
        self.dataset = dataset # 直接注入内存中的 PKL 数据集
        self.renderer = StateRenderer()
        self.thumbnail_cache = {}
        
        # 筛选出属于当前字体的所有数据
        self.font_records = [v for k, v in self.dataset.items() if v['font_filename'] == self.font_filename]
        
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.stacked = QStackedWidget()
        
        self.page_gallery = QWidget()
        gal_layout = QVBoxLayout(self.page_gallery)
        title = QLabel(f"✅ Data Inspector: {self.font_filename} ({len(self.font_records)} chars in PKL)")
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
        cols = 5
        for idx, record in enumerate(self.font_records):
            hex_key = record['hex_key']
            char = record['char']
            
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
            
            btn_view = QPushButton("🔍 Inspect PKL Data")
            btn_view.setCursor(Qt.PointingHandCursor)
            btn_view.setStyleSheet("background-color: #2196F3; color: white; padding: 8px; border-radius: 4px; font-weight: bold;")
            btn_view.clicked.connect(lambda checked, rec=record: self.open_detail(rec))
            
            card_layout.addWidget(img_label)
            card_layout.addWidget(title)
            card_layout.addWidget(btn_view)
            self.grid.addWidget(card, idx // cols, idx % cols)

    def open_detail(self, record):
        while self.detail_layout.count():
            item = self.detail_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
            
        char = record['char']
        hex_key = record['hex_key']
        strokes = record['strokes'] # 这是从 PKL 中读出来的 100% 结构化清洗数据
        
        top_bar = QHBoxLayout()
        btn_back = QPushButton("🔙 Back")
        btn_back.setFixedWidth(100)
        btn_back.setStyleSheet("padding: 10px; background-color: #607D8B; color: white; font-weight: bold; border-radius: 4px;")
        btn_back.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        top_bar.addWidget(btn_back)
        
        lbl_title = QLabel(f"PKL Data Audit: '{char}' ({hex_key})")
        lbl_title.setStyleSheet("font-size: 22px; font-weight: bold;")
        top_bar.addWidget(lbl_title)
        top_bar.addStretch()
        self.detail_layout.addLayout(top_bar)
        
        line = QFrame(); line.setFrameShape(QFrame.HLine); self.detail_layout.addWidget(line)

        # 仅渲染底层做对照，不执行拓扑计算！
        binary = render_unicode_glyph(self.font_path, char, CANVAS_SIZE)

        grid_view = QGridLayout()
        grid_view.setSpacing(20)
        
        def create_view_card(title_html, pixmap):
            container = QFrame()
            container.setStyleSheet("background-color: white; border: 1px solid #E0E0E0; border-radius: 8px;")
            l = QVBoxLayout(container)
            lbl_title = QLabel(title_html)
            lbl_title.setAlignment(Qt.AlignCenter)
            lbl_title.setStyleSheet("border: none; margin-top: 5px;")
            lbl_img = QLabel(); lbl_img.setPixmap(pixmap); lbl_img.setAlignment(Qt.AlignCenter); lbl_img.setStyleSheet("border: none;")
            l.addWidget(lbl_title); l.addWidget(lbl_img)
            return container

        card1 = create_view_card("<b>1. Original Binary Mask</b>", self.renderer.render_thumbnail(binary, size=(3.5, 3.5)))
        
        card2 = create_view_card("<b>2. PKL Stored Curves</b><br><span style='color:gray'>Preprocessed Geometry</span>", 
                                 self.renderer.render_top_image(binary, strokes, size=(3.5, 3.5)))
        
        card3 = create_view_card("<b>3. Reconstructed Glyph</b>", 
                                 self.renderer.render_bottom_image(binary, strokes, size=(3.5, 3.5)))
        
        legend = "<span style='color:#2196F3'>1:Loop</span> | <span style='color:#4CAF50'>2:Start</span> | <span style='color:#FFEB3B'>3:Mid</span> | <span style='color:#F44336'>4:End</span> | <span style='color:#9C27B0'>5:T-Intersect</span> | <span style='color:#795548'>6:X-Intersect</span>"
        card4 = create_view_card(f"<b>4. Stored Topo-State</b><br><span style='font-size:11px;'>{legend}</span>", 
                                 self.renderer.render_classified_image(binary, strokes, size=(3.5, 3.5)))
        card4.setStyleSheet("background-color: #FAFAFA; border: 2px solid #2196F3; border-radius: 8px;") 

        grid_view.addWidget(card1, 0, 0)
        grid_view.addWidget(card2, 0, 1)
        grid_view.addWidget(card3, 1, 0)
        grid_view.addWidget(card4, 1, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: transparent; border: none;")
        grid_widget = QWidget(); grid_widget.setLayout(grid_view)
        scroll.setWidget(grid_widget)
        
        self.detail_layout.addWidget(scroll)
        self.stacked.setCurrentIndex(1)

# ==========================================
# 🚪 主窗口 (数据树侧边栏)
# ==========================================
class PKLInspectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Preprocessed PKL Data Inspector")
        self.setGeometry(50, 50, 1500, 950)
        
        # 🌟 启动即加载全局 PKL 数据
        if os.path.exists(PKL_PATH):
            with open(PKL_PATH, 'rb') as f:
                self.dataset = pickle.load(f)
            print(f"✅ 成功加载 PKL 数据集，包含 {len(self.dataset)} 个字形。")
        else:
            self.dataset = {}
            print(f"❌ 未找到 PKL 数据集！请先运行 step1_5 预处理脚本。")
        
        self.splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.splitter)
        
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        
        lbl_annotated = QLabel(f"📦 Fonts in PKL Data")
        lbl_annotated.setStyleSheet("font-size: 15px; font-weight: bold; color: #4CAF50;")
        sidebar_layout.addWidget(lbl_annotated)
        
        self.list_annotated = QListWidget()
        self.list_annotated.itemClicked.connect(self.on_font_selected)
        self.list_annotated.setStyleSheet("background-color: #F1F8E9; border: 1px solid #C8E6C9; border-radius: 4px;")
        sidebar_layout.addWidget(self.list_annotated)
        
        lbl_unannotated = QLabel("🈳 Fonts Missing from PKL")
        lbl_unannotated.setStyleSheet("font-size: 15px; font-weight: bold; color: #FF9800; margin-top: 10px;")
        sidebar_layout.addWidget(lbl_unannotated)
        
        self.list_unannotated = QListWidget()
        self.list_unannotated.itemClicked.connect(self.on_font_selected)
        self.list_unannotated.setStyleSheet("background-color: #FFF3E0; border: 1px solid #FFE0B2; border-radius: 4px;")
        sidebar_layout.addWidget(self.list_unannotated)

        self.workspace_stack = QStackedWidget()
        self.workspace_stack.addWidget(QLabel("👈 Select a font to begin PKL data inspection..."))
        
        self.splitter.addWidget(sidebar)
        self.splitter.addWidget(self.workspace_stack)
        self.splitter.setSizes([250, 1250])
        
        self.load_font_list()

    def load_font_list(self):
        self.list_annotated.clear()
        self.list_unannotated.clear()
        
        if not os.path.exists(FONTS_DIR): return
        
        # 获取存在于 PKL 中的字体名集合
        fonts_in_pkl = set([v['font_filename'] for v in self.dataset.values()])

        for file in os.listdir(FONTS_DIR):
            if file.lower().endswith(('.ttf', '.otf')):
                font_filename = os.path.splitext(file)[0]
                
                if font_filename in fonts_in_pkl:
                    self.list_annotated.addItem(file)
                else:
                    self.list_unannotated.addItem(file)

    def on_font_selected(self, item):
        sender = self.sender()
        if sender == self.list_annotated: self.list_unannotated.clearSelection()
        elif sender == self.list_unannotated: self.list_annotated.clearSelection()

        font_path = os.path.join(FONTS_DIR, item.text())
        font_filename = os.path.splitext(item.text())[0]
        
        if font_filename not in [v['font_filename'] for v in self.dataset.values()]:
            # 如果点击了不在 PKL 里的字体，清空右侧
            if self.workspace_stack.count() > 1:
                old = self.workspace_stack.widget(1)
                self.workspace_stack.removeWidget(old)
                old.deleteLater()
            return
            
        if self.workspace_stack.count() > 1:
            old = self.workspace_stack.widget(1)
            self.workspace_stack.removeWidget(old)
            old.deleteLater()
            
        new_ws = PreviewWorkspace(font_path, self.dataset)
        self.workspace_stack.addWidget(new_ws)
        self.workspace_stack.setCurrentIndex(1)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("light", custom_colors={"primary": "#4CAF50"})
    window = PKLInspectorApp()
    window.show()
    sys.exit(app.exec_())