import os
import sys
import pickle
import io
import numpy as np
import qdarktheme
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QScrollArea, 
                             QStackedWidget, QGridLayout, QListWidget, QSplitter, QFrame, QTextEdit)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

try:
    from geometry_vision import render_unicode_glyph
except ImportError:
    from PIL import Image, ImageDraw, ImageFont
    def render_unicode_glyph(font_path, char, canvas_size=400):
        img = Image.new('L', (canvas_size, canvas_size), 0); draw = ImageDraw.Draw(img)
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

class StateRenderer:
    def __init__(self):
        try: self.cmap = plt.colormaps['tab20']
        except AttributeError: self.cmap = plt.get_cmap('tab20')

    def _extract(self, stroke):
        return np.array(stroke['mother_bezier']), np.array(stroke.get('width_bezier', [6.0]*4))

    def get_hex_color(self, idx):
        return mcolors.to_hex(self.cmap((idx % 20)))

    def render_top_image(self, binary, strokes, size=(3.5, 3.5), draw_arrows=False):
        fig = plt.figure(figsize=size, dpi=90); ax = fig.add_subplot(111)
        ax.imshow(binary, cmap='gray', alpha=0.05) 
        for i, e in enumerate(strokes):
            p_opt, w_opt = self._extract(e)
            color = self.cmap((i % 20))
            mean_w = max(np.mean(w_opt), 1.0)
            curve = cubic_bezier_np(p_opt, np.linspace(0, 1, 50)[:, None])
            ax.plot(curve[:, 0], curve[:, 1], c=color, lw=mean_w * 2, solid_capstyle='round', alpha=0.6)
            ax.plot(curve[:, 0], curve[:, 1], color='black', linewidth=1.5, alpha=0.5)
            if draw_arrows:
                dir_vec = curve[-1] - curve[-2]
                norm = np.linalg.norm(dir_vec)
                if norm > 0:
                    ax.arrow(curve[-2, 0], curve[-2, 1], dir_vec[0]/norm*12, dir_vec[1]/norm*12, head_width=6, head_length=8, fc=color, ec=color)
        ax.axis('off'); return self._fig_to_pixmap(fig)

    def render_skeleton_image(self, binary, strokes, size=(3.5, 3.5)):
        """🌟 专属纯净骨架图：极其细的线条，清晰显示关节交点和尖角裁减情况"""
        fig = plt.figure(figsize=size, dpi=90); ax = fig.add_subplot(111)
        ax.imshow(binary, cmap='gray', alpha=0.05) 
        for i, e in enumerate(strokes):
            p_opt, _ = self._extract(e)
            color = self.cmap((i % 20))
            curve = cubic_bezier_np(p_opt, np.linspace(0, 1, 100)[:, None])
            ax.plot(curve[:, 0], curve[:, 1], c=color, lw=2, alpha=1.0)
            ax.plot(p_opt[0, 0], p_opt[0, 1], 'o', c=color, markersize=5)
            ax.plot(p_opt[3, 0], p_opt[3, 1], 's', c=color, markersize=5)
        ax.axis('off'); return self._fig_to_pixmap(fig)

    def render_bottom_image(self, binary, strokes, size=(3.5, 3.5)):
        fig = plt.figure(figsize=size, dpi=90); ax = fig.add_subplot(111)
        ax.imshow(np.ones_like(binary), cmap='gray', vmin=0, vmax=1) 
        for e in strokes:
            p_opt, w_opt = self._extract(e)
            ts_dense = np.linspace(0, 1, 100)[:, None]
            curve_dense = cubic_bezier_np(p_opt, ts_dense)
            mt = 1 - ts_dense
            w_vals = mt**3 * w_opt[0] + 3*mt**2*ts_dense * w_opt[1] + 3*mt*ts_dense**2 * w_opt[2] + ts_dense**3 * w_opt[3]
            dp = np.gradient(curve_dense, axis=0)
            n = np.zeros_like(dp); n[:, 0], n[:, 1] = -dp[:, 1], dp[:, 0]
            n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-5)
            upper, lower = curve_dense + n * w_vals, curve_dense - n * w_vals
            poly = np.vstack([upper, lower[::-1]])
            ax.fill(poly[:, 0], poly[:, 1], color='#000000', alpha=0.85, linewidth=0)
        ax.axis('off'); return self._fig_to_pixmap(fig)

    def render_numbered_topology(self, binary, strokes, size=(3.5, 3.5)):
        fig = plt.figure(figsize=size, dpi=90); ax = fig.add_subplot(111)
        ax.imshow(binary, cmap='gray', alpha=0.05)
        for i, e in enumerate(strokes):
            p_opt, _ = self._extract(e)
            color = self.cmap((i % 20))
            curve = cubic_bezier_np(p_opt, np.linspace(0, 1, 50)[:, None])
            ax.plot(curve[:, 0], curve[:, 1], c=color, lw=5, alpha=0.8, solid_capstyle='round')
            dir_vec = curve[-1] - curve[-2]
            norm = np.linalg.norm(dir_vec)
            if norm > 0:
                ax.arrow(curve[-2, 0], curve[-2, 1], dir_vec[0]/norm*12, dir_vec[1]/norm*12, head_width=6, head_length=8, fc=color, ec=color)
            cx, cy = curve[len(curve)//2, 0], curve[len(curve)//2, 1]
            ax.text(cx, cy, str(i), color='white', fontsize=11, fontweight='bold',
                    ha='center', va='center', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=0.3))
        ax.axis('off'); return self._fig_to_pixmap(fig)

    def render_thumbnail(self, binary, size=(2.5, 2.5)):
        fig = plt.figure(figsize=size, dpi=80); ax = fig.add_subplot(111)
        ax.imshow(binary, cmap='gray'); ax.axis('off'); return self._fig_to_pixmap(fig)

    def _fig_to_pixmap(self, fig):
        plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0); plt.close(fig)
        buf.seek(0); qimg = QImage(); qimg.loadFromData(buf.getvalue())
        return QPixmap.fromImage(qimg)

class PreviewWorkspace(QWidget):
    def __init__(self, font_path, dataset):
        super().__init__()
        self.font_path = font_path; self.font_filename = os.path.basename(font_path).split('.')[0]
        self.dataset = dataset; self.renderer = StateRenderer(); self.thumbnail_cache = {}
        self.font_records = [v for k, v in self.dataset.items() if v['font_filename'] == self.font_filename]
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self); self.stacked = QStackedWidget()
        self.page_gallery = QWidget(); gal_layout = QVBoxLayout(self.page_gallery)
        title = QLabel(f"✅ Preprocessed Graph Matrix Gallery: {self.font_filename}")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        gal_layout.addWidget(title)
        
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        container = QWidget(); self.grid = QGridLayout(container); scroll.setWidget(container)
        gal_layout.addWidget(scroll)
        
        self.page_detail = QWidget(); self.detail_layout = QVBoxLayout(self.page_detail)
        self.stacked.addWidget(self.page_gallery); self.stacked.addWidget(self.page_detail)
        self.layout.addWidget(self.stacked); self.load_gallery()

    def load_gallery(self):
        cols = 5
        for idx, record in enumerate(self.font_records):
            char, hex_key = record['char'], record['hex_key']
            card = QFrame(); card.setStyleSheet("background-color: white; border: 2px solid #E0E0E0; border-radius: 8px;")
            card_layout = QVBoxLayout(card)
            if hex_key not in self.thumbnail_cache:
                binary = render_unicode_glyph(self.font_path, char, CANVAS_SIZE)
                self.thumbnail_cache[hex_key] = self.renderer.render_thumbnail(binary)
            img_label = QLabel(); img_label.setPixmap(self.thumbnail_cache[hex_key]); img_label.setAlignment(Qt.AlignCenter)
            title = QLabel(f"'{char}'\n{hex_key}"); title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("font-size: 15px; font-weight: bold;")
            btn_view = QPushButton("🔍 Audit N^2 Attention")
            btn_view.setStyleSheet("background-color: #2196F3; color: white; padding: 6px; font-weight: bold;")
            btn_view.clicked.connect(lambda checked, rec=record: self.open_detail(rec))
            card_layout.addWidget(img_label); card_layout.addWidget(title); card_layout.addWidget(btn_view)
            self.grid.addWidget(card, idx // cols, idx % cols)

    def open_detail(self, record):
        while self.detail_layout.count():
            item = self.detail_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
            
        top_bar = QHBoxLayout()
        btn_back = QPushButton("🔙 Back"); btn_back.setStyleSheet("padding: 8px; background-color: #607D8B; color: white; font-weight: bold;")
        btn_back.clicked.connect(lambda: self.stacked.setCurrentIndex(0)); top_bar.addWidget(btn_back)
        top_bar.addWidget(QLabel(f"<b>N^2 Attention Topology Audit</b>: '{record['char']}' ({record['hex_key']})"))
        top_bar.addStretch(); self.detail_layout.addLayout(top_bar)
        
        main_scroll = QScrollArea(); main_scroll.setWidgetResizable(True); scroll_content = QWidget(); scroll_layout = QVBoxLayout(scroll_content)
        binary = render_unicode_glyph(self.font_path, record['char'], CANVAS_SIZE)

        grid_view = QGridLayout()
        def create_view_card(title_html, pixmap):
            container = QFrame(); container.setStyleSheet("background-color: white; border: 1px solid #E0E0E0; border-radius: 8px;")
            l = QVBoxLayout(container); lbl_title = QLabel(title_html); lbl_title.setAlignment(Qt.AlignCenter)
            lbl_img = QLabel(); lbl_img.setPixmap(pixmap); lbl_img.setAlignment(Qt.AlignCenter)
            l.addWidget(lbl_title); l.addWidget(lbl_img); return container

        grid_view.addWidget(create_view_card("<b>1. Original Mask</b>", self.renderer.render_thumbnail(binary)), 0, 0)
        grid_view.addWidget(create_view_card("<b>2. Raw Annotated Strokes</b><br><span style='color:gray; font-size:11px;'>With Overlapping Tips</span>", self.renderer.render_top_image(binary, record['raw_strokes'])), 0, 1)
        grid_view.addWidget(create_view_card("<b>3. Snapped & Pruned Skeletons</b><br><span style='color:gray; font-size:11px;'>Cleaned Geometry (Thin lines)</span>", self.renderer.render_skeleton_image(binary, record['strokes'])), 0, 2)
        grid_view.addWidget(create_view_card("<b>4. Reconstructed TTF Mesh</b>", self.renderer.render_bottom_image(binary, record['strokes'])), 1, 0)
        
        c5 = create_view_card("<b>5. N^2 Attention Map (Nodes Label)</b>", self.renderer.render_numbered_topology(binary, record['strokes']))
        c5.setStyleSheet("background-color: #FAFAFA; border: 2px solid #E91E63; border-radius: 8px;")
        grid_view.addWidget(c5, 1, 1)

        text_log = QTextEdit(); text_log.setReadOnly(True)
        text_log.setStyleSheet("font-size: 14px; line-height: 1.6; background-color: #FAFAFA; border: 2px solid #E0E0E0; border-radius: 8px; padding: 12px;")
        attn_matrix = record.get('attention_matrix', [])
        N = len(record['strokes'])
        log_html = "<h4 style='color: #E91E63; margin-top:0;'>🧠 Graphormer 矩阵分析</h4><hr>"
        has_relation = False
        for i in range(N):
            for j in range(N):
                if i == j: continue
                v_ij = attn_matrix[i][j]; v_ji = attn_matrix[j][i]
                c_i = f"<b style='color:{self.renderer.get_hex_color(i)}'>[边 {i}]</b>"
                c_j = f"<b style='color:{self.renderer.get_hex_color(j)}'>[边 {j}]</b>"
                
                if i < j and v_ij == 1 and v_ji == 1:
                    log_html += f"<p>🔗 {c_i} 和 {c_j} 端点交汇 <b>(1)</b></p>"; has_relation = True
                elif i < j and v_ij == 2 and v_ji == 2:
                    log_html += f"<p>🌀 {c_i} 与 {c_j} 属首尾闭合环 <b>(2)</b></p>"; has_relation = True
                elif i < j and v_ij == 3 and v_ji == 3:
                    log_html += f"<p>⚔️ {c_i} 与 {c_j} 呈 X 型相交 <b>(3)</b></p>"; has_relation = True
                elif v_ij == 5 and v_ji == 4:
                    log_html += f"<p>🔨 {c_i} T型顶入 {c_j} 躯干<br><span style='color:gray; font-size:11px;'>[进攻{i}][防守{j}] = 5/4</span></p>"; has_relation = True

        if not has_relation: log_html += "<p style='color:gray;'>各笔画完全物理独立。</p>"
        text_log.setHtml(log_html)
        grid_view.addWidget(text_log, 1, 2)

        scroll_layout.addLayout(grid_view)
        main_scroll.setWidget(scroll_content); self.detail_layout.addWidget(main_scroll)
        self.stacked.setCurrentIndex(1)

class PKLInspectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Preprocessed N^2 Graph Inspector")
        self.setGeometry(50, 50, 1600, 950)
        self.dataset = pickle.load(open(PKL_PATH, 'rb')) if os.path.exists(PKL_PATH) else {}
        self.splitter = QSplitter(Qt.Horizontal); self.setCentralWidget(self.splitter)
        
        sidebar = QWidget(); sidebar_layout = QVBoxLayout(sidebar)
        lbl = QLabel("📦 Extracted PKL Fonts"); lbl.setStyleSheet("font-weight: bold; color: #4CAF50;")
        sidebar_layout.addWidget(lbl)
        
        self.list_annotated = QListWidget()
        self.list_annotated.itemClicked.connect(self.on_font_selected)
        self.list_annotated.setStyleSheet("background-color: #F1F8E9; border-radius: 4px;")
        sidebar_layout.addWidget(self.list_annotated)
        
        self.workspace_stack = QStackedWidget(); self.workspace_stack.addWidget(QLabel("👈 Select a font..."))
        self.splitter.addWidget(sidebar); self.splitter.addWidget(self.workspace_stack); self.splitter.setSizes([230, 1370])
        self.load_font_list()

    def load_font_list(self):
        self.list_annotated.clear()
        if not self.dataset: return
        for font in sorted(list(set([v['font_filename'] for v in self.dataset.values()]))):
            self.list_annotated.addItem(font)

    def on_font_selected(self, item):
        font_filename = item.text()
        if self.workspace_stack.count() > 1:
            old = self.workspace_stack.widget(1); self.workspace_stack.removeWidget(old); old.deleteLater()
        new_ws = PreviewWorkspace(os.path.join(FONTS_DIR, f"{font_filename}.ttf"), self.dataset)
        self.workspace_stack.addWidget(new_ws); self.workspace_stack.setCurrentIndex(1)

if __name__ == '__main__':
    app = QApplication(sys.argv); qdarktheme.setup_theme("light")
    window = PKLInspectorApp(); window.show(); sys.exit(app.exec_())