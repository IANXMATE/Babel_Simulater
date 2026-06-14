import os
import sys
import copy
import io
import json
import numpy as np
import networkx as nx
import qdarktheme
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QScrollArea, 
                             QStackedWidget, QGridLayout, QListWidget, QSplitter, QFrame, QTextBrowser)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# 导入你现有的基础计算模块
from geometry_vision import (render_unicode_glyph, fit_bezier_basic_with_error, 
                             regress_width_dt_fast, cubic_bezier_np)

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR = os.path.join(SCRIPT_DIR, "../dataset/alien_tensors_storage")
TOPO_OUT_DIR = os.path.join(SCRIPT_DIR, "annotations_topo")
META_DIR = os.path.join(SCRIPT_DIR, "metadata")
CANVAS_SIZE = 400

# ==========================================
# 🎨 离线渲染引擎 (包含三段渲染逻辑)
# ==========================================
class StateRenderer:
    def __init__(self):
        try: self.cmap = plt.colormaps['tab20']
        except AttributeError: self.cmap = plt.get_cmap('tab20')

    def _fig_to_pixmap(self, fig):
        plt.tight_layout(pad=0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        buf.seek(0)
        qimg = QImage()
        qimg.loadFromData(buf.getvalue())
        return QPixmap.fromImage(qimg)

    def render_original(self, binary, size=(2, 2)):
        fig = plt.figure(figsize=size, dpi=80)
        fig.patch.set_facecolor('#FFFFFF')
        ax = fig.add_subplot(111)
        ax.imshow(binary, cmap='gray')
        ax.axis('off')
        return self._fig_to_pixmap(fig)

    def render_phase1(self, binary, raw_edges, dt_map=None, bw=False, size=(2, 2)):
        fig = plt.figure(figsize=size, dpi=80)
        fig.patch.set_facecolor('#FFFFFF')
        ax = fig.add_subplot(111)
        
        if bw:
            ax.imshow(np.ones_like(binary), cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(binary, cmap='gray', alpha=0.15)

        for i, e in enumerate(raw_edges):
            path = np.array(e['path'])
            if len(path) < 2: continue
            
            # P1 需要实时拟合计算
            p_opt, _ = fit_bezier_basic_with_error(path)
            w_opt = regress_width_dt_fast(p_opt, dt_map) if dt_map is not None else np.array([5.0]*4)
            color = self.cmap((e.get('id', i) % 20))

            ts_dense = np.linspace(0, 1, 50)[:, None]
            curve_dense = cubic_bezier_np(p_opt, ts_dense)
            
            # 宽度计算
            mt = 1 - ts_dense
            w_vals = mt**3 * w_opt[0] + 3*mt**2*ts_dense * w_opt[1] + 3*mt*ts_dense**2 * w_opt[2] + ts_dense**3 * w_opt[3]
            dp = np.gradient(curve_dense, axis=0)
            n = np.zeros_like(dp)
            n[:, 0], n[:, 1] = -dp[:, 1], dp[:, 0]
            n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-5)
            upper, lower = curve_dense + n * w_vals, curve_dense - n * w_vals
            poly = np.vstack([upper, lower[::-1]])
            
            if bw:
                ax.fill(poly[:, 0], poly[:, 1], color='#000000', alpha=0.9, linewidth=0)
            else:
                ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.8, linewidth=0)
                ax.plot(curve_dense[:, 0], curve_dense[:, 1], color='black', lw=1.5, alpha=0.7)
                
        ax.axis('off')
        return self._fig_to_pixmap(fig)

    def render_phase2(self, binary, strokes, bw=False, with_id=False, size=(2, 2)):
        fig = plt.figure(figsize=size, dpi=80)
        fig.patch.set_facecolor('#FFFFFF')
        ax = fig.add_subplot(111)
        
        if bw:
            ax.imshow(np.ones_like(binary), cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(binary, cmap='gray', alpha=0.15)

        for e in strokes:
            eid = e['bezier_id']
            p_opt = np.array(e['mother_bezier'])
            w_opt = np.array(e['width_bezier'])
            color = self.cmap((eid % 20))

            ts_dense = np.linspace(0, 1, 50)[:, None]
            curve_dense = cubic_bezier_np(p_opt, ts_dense)
            
            mt = 1 - ts_dense
            w_vals = mt**3 * w_opt[0] + 3*mt**2*ts_dense * w_opt[1] + 3*mt*ts_dense**2 * w_opt[2] + ts_dense**3 * w_opt[3]
            dp = np.gradient(curve_dense, axis=0)
            n = np.zeros_like(dp)
            n[:, 0], n[:, 1] = -dp[:, 1], dp[:, 0]
            n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-5)
            upper, lower = curve_dense + n * w_vals, curve_dense - n * w_vals
            poly = np.vstack([upper, lower[::-1]])
            
            if bw:
                ax.fill(poly[:, 0], poly[:, 1], color='#000000', alpha=0.9, linewidth=0)
            else:
                ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.8, linewidth=0)
                ax.plot(curve_dense[:, 0], curve_dense[:, 1], color='black', lw=1.5, alpha=0.7)
                
                if with_id:
                    mid_pt = curve_dense[25]
                    ax.text(mid_pt[0], mid_pt[1], str(eid), color=color, fontsize=14, fontweight='bold', 
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
                    
        ax.axis('off')
        return self._fig_to_pixmap(fig)


# ==========================================
# 🏠 右侧工作区 (四栏画廊 + 详情页)
# ==========================================
class AuditWorkspace(QWidget):
    def __init__(self, font_path):
        super().__init__()
        self.font_path = font_path
        self.font_filename = os.path.basename(font_path).split('.')[0]
        self.renderer = StateRenderer()
        self.thumbnail_cache = {}

        # 加载所有数据状态
        self.p2_data = {}
        self.p1_raw = {}
        self.banned = []
        self.char_list = []
        
        self.load_font_data()
        self.init_ui()

    def load_font_data(self):
        # 1. 拿全部字符
        try:
            from fontTools.ttLib import TTFont
            tt = TTFont(self.font_path)
            cmap = tt.getBestCmap()
            if cmap: self.char_list = [f"U+{k:04X}" for k in cmap.keys()]
            tt.close()
        except: pass

        # 2. 拿 P2 数据
        topo_file = os.path.join(TOPO_OUT_DIR, f"{self.font_filename}_topo.json")
        if os.path.exists(topo_file):
            try:
                with open(topo_file, 'r', encoding='utf-8') as f: self.p2_data = json.load(f)
            except: pass

        # 3. 拿 P1 & Banned 数据
        meta_file = os.path.join(META_DIR, f"{self.font_filename}_meta.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    self.banned = meta.get("banned", [])
                    self.p1_raw = meta.get("raw_edges", {})
            except: pass

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        
        # 顶部四级分类导航
        nav_bar = QHBoxLayout()
        self.btn_full = QPushButton("✅ Fully Completed")
        self.btn_p1 = QPushButton("⏳ P1 Done Only")
        self.btn_ban = QPushButton("🚫 Banned")
        self.btn_not = QPushButton("⚪ Not Started")
        
        self.btn_full.clicked.connect(lambda: self.switch_gallery_tab("completed"))
        self.btn_p1.clicked.connect(lambda: self.switch_gallery_tab("p1_only"))
        self.btn_ban.clicked.connect(lambda: self.switch_gallery_tab("banned"))
        self.btn_not.clicked.connect(lambda: self.switch_gallery_tab("untouched"))
        
        for btn in [self.btn_full, self.btn_p1, self.btn_ban, self.btn_not]:
            btn.setCheckable(True)
            nav_bar.addWidget(btn)
        nav_bar.addStretch()
        self.layout.addLayout(nav_bar)

        self.stacked = QStackedWidget()
        
        # 画廊页面
        self.page_gallery = QWidget()
        gal_layout = QVBoxLayout(self.page_gallery)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        container = QWidget(); self.grid = QGridLayout(container)
        scroll.setWidget(container)
        gal_layout.addWidget(scroll)
        
        # 详情页面
        self.page_detail = QWidget()
        self.detail_layout = QVBoxLayout(self.page_detail)
        
        self.stacked.addWidget(self.page_gallery)
        self.stacked.addWidget(self.page_detail)
        self.layout.addWidget(self.stacked)
        
        self.btn_full.setChecked(True)
        self.switch_gallery_tab("completed")

    def switch_gallery_tab(self, mode):
        for btn in [self.btn_full, self.btn_p1, self.btn_ban, self.btn_not]: btn.setChecked(False)
        
        keys_to_render = []
        if mode == "completed":
            self.btn_full.setChecked(True)
            keys_to_render = list(self.p2_data.keys())
        elif mode == "p1_only":
            self.btn_p1.setChecked(True)
            keys_to_render = [k for k in self.p1_raw.keys() if k not in self.p2_data and k not in self.banned]
        elif mode == "banned":
            self.btn_ban.setChecked(True)
            keys_to_render = self.banned
        elif mode == "untouched":
            self.btn_not.setChecked(True)
            keys_to_render = [k for k in self.char_list if k not in self.p2_data and k not in self.p1_raw and k not in self.banned]
            keys_to_render = keys_to_render[:200] # 限制数量防卡死

        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget(): item.widget().deleteLater()
            
        cols = 5
        for idx, hex_key in enumerate(keys_to_render):
            char = chr(int(hex_key[2:], 16))
            
            card = QFrame()
            card.setStyleSheet("background-color: #FAFAFA; border: 1px solid #CCC; border-radius: 6px;")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(5, 5, 5, 5)
            card_layout.setSpacing(2)

            # --- 渲染三层缩略图 ---
            if hex_key not in self.thumbnail_cache:
                binary = render_unicode_glyph(self.font_path, char, CANVAS_SIZE)
                
                pix_orig = self.renderer.render_original(binary, size=(1.2, 1.2))
                
                # Phase 1
                if hex_key in self.p1_raw:
                    dt_map = distance_transform_edt(binary)
                    pix_p1 = self.renderer.render_phase1(binary, self.p1_raw[hex_key], dt_map, size=(1.2, 1.2))
                else:
                    pix_p1 = QPixmap(pix_orig.size()); pix_p1.fill(Qt.transparent)
                    
                # Phase 2
                if hex_key in self.p2_data:
                    pix_p2 = self.renderer.render_phase2(binary, self.p2_data[hex_key]["strokes"], size=(1.2, 1.2))
                else:
                    pix_p2 = QPixmap(pix_orig.size()); pix_p2.fill(Qt.transparent)
                    
                self.thumbnail_cache[hex_key] = (pix_orig, pix_p1, pix_p2)
                
            p_orig, p_p1, p_p2 = self.thumbnail_cache[hex_key]
            
            l1 = QLabel(); l1.setPixmap(p_orig); l1.setAlignment(Qt.AlignCenter)
            l2 = QLabel(); l2.setPixmap(p_p1); l2.setAlignment(Qt.AlignCenter)
            l3 = QLabel(); l3.setPixmap(p_p2); l3.setAlignment(Qt.AlignCenter)
            
            card_layout.addWidget(l1)
            card_layout.addWidget(l2)
            card_layout.addWidget(l3)
            
            title = QLabel(f"'{char}'\n{hex_key}")
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("font-weight: bold; font-size: 13px; border:none; margin-top:5px;")
            card_layout.addWidget(title)
            
            if mode in ["completed", "p1_only"]:
                btn = QPushButton("🔍 Inspect Details")
                btn.setStyleSheet("background-color: #2196F3; color: white; padding: 5px; border-radius: 3px;")
                btn.clicked.connect(lambda checked, hk=hex_key: self.open_detail_view(hk))
                card_layout.addWidget(btn)
                
            self.grid.addWidget(card, idx // cols, idx % cols)
            
        self.stacked.setCurrentIndex(0)


    def open_detail_view(self, hex_key):
        while self.detail_layout.count():
            item = self.detail_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
            
        char = chr(int(hex_key[2:], 16))
        
        top_bar = QHBoxLayout()
        btn_back = QPushButton("🔙 Back to Gallery")
        btn_back.setStyleSheet("padding: 8px; background-color: #757575; color: white; font-weight: bold; border-radius: 4px;")
        btn_back.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        top_bar.addWidget(btn_back)
        top_bar.addWidget(QLabel(f"<span style='font-size:20px; font-weight:bold;'>Detail Inspection: '{char}' ({hex_key})</span>"))
        top_bar.addStretch()
        self.detail_layout.addLayout(top_bar)
        
        # 初始化基础底图
        binary = render_unicode_glyph(self.font_path, char, CANVAS_SIZE)
        dt_map = distance_transform_edt(binary)
        
        # ==========================================
        # 🌟 三栏图片区 (占满主要空间)
        # ==========================================
        img_container = QWidget()
        vbox_imgs = QVBoxLayout(img_container)
        
        # 1. 顶层：原始形态 (占宽居中)
        row1 = QHBoxLayout()
        lbl_orig = QLabel()
        lbl_orig.setPixmap(self.renderer.render_original(binary, size=(3.5, 3.5)))
        lbl_orig.setAlignment(Qt.AlignCenter)
        lbl_orig.setStyleSheet("border: 2px dashed #BDBDBD;")
        row1.addStretch(); row1.addWidget(lbl_orig); row1.addStretch()
        vbox_imgs.addLayout(row1)
        
        # 2. 中层：Phase 1 双图 (彩色宽网格 + 黑白实心)
        row2 = QHBoxLayout()
        if hex_key in self.p1_raw:
            lbl_p1_c = QLabel(); lbl_p1_c.setPixmap(self.renderer.render_phase1(binary, self.p1_raw[hex_key], dt_map, bw=False, size=(4.5, 4.5)))
            lbl_p1_bw = QLabel(); lbl_p1_bw.setPixmap(self.renderer.render_phase1(binary, self.p1_raw[hex_key], dt_map, bw=True, size=(4.5, 4.5)))
            lbl_p1_c.setStyleSheet("border: 1px solid #E0E0E0;"); lbl_p1_bw.setStyleSheet("border: 1px solid #E0E0E0;")
            row2.addWidget(lbl_p1_c); row2.addWidget(lbl_p1_bw)
        else:
            row2.addWidget(QLabel("Phase 1 Data Missing."))
        vbox_imgs.addLayout(row2)
        
        # 3. 底层：Phase 2 双图 (彩色+数字ID + 黑白实心)
        row3 = QHBoxLayout()
        if hex_key in self.p2_data:
            strokes = self.p2_data[hex_key]["strokes"]
            lbl_p2_c = QLabel(); lbl_p2_c.setPixmap(self.renderer.render_phase2(binary, strokes, bw=False, with_id=True, size=(4.5, 4.5)))
            lbl_p2_bw = QLabel(); lbl_p2_bw.setPixmap(self.renderer.render_phase2(binary, strokes, bw=True, with_id=False, size=(4.5, 4.5)))
            lbl_p2_c.setStyleSheet("border: 1px solid #E0E0E0;"); lbl_p2_bw.setStyleSheet("border: 1px solid #E0E0E0;")
            row3.addWidget(lbl_p2_c); row3.addWidget(lbl_p2_bw)
        else:
            row3.addWidget(QLabel("Phase 2 Data Missing. Not fully completed yet."))
        vbox_imgs.addLayout(row3)
        
        scroll_imgs = QScrollArea()
        scroll_imgs.setWidgetResizable(True)
        scroll_imgs.setWidget(img_container)
        self.detail_layout.addWidget(scroll_imgs, stretch=4)
        
        # ==========================================
        # 🌟 底部文本区：物理状态再现报告
        # ==========================================
        # ==========================================
        # 🌟 底部文本区：直接无脑读取 JSON 结构，校验数据纯净度
        # ==========================================
        topo_box = QTextBrowser()
        topo_box.setStyleSheet("background-color: #F8F9FA; border: 1px solid #CCC; padding: 10px; font-size: 15px;")
        
        if hex_key in self.p2_data:
            # 🌟 核心修改：适配最新 5 层金字塔的 root 级字段
            char_data = self.p2_data[hex_key]
            topology_events = char_data.get("topology_events", [])
            cycles = char_data.get("cycles", [])
            
            # 分类提取事件
            e2e = [ev for ev in topology_events if ev["type"] == "E2E"]
            t_juncs = [ev for ev in topology_events if ev["type"] == "T"]
            x_juncs = [ev for ev in topology_events if ev["type"] == "X"]
            
            # 配色高亮辅助函数
            def _hex_col(eid): 
                c = self.renderer.cmap(int(eid) % 20)
                return f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
            def _span(eid): 
                return f"<span style='color:{_hex_col(eid)}; font-weight:bold;'>{eid}</span>"

            html = ["<b style='color:#333; font-size:16px;'>📊 模型输入数据纯净度校验 (读取自JSON)</b><hr>"]
            
            if e2e:
                strs = [f"{_span(ev['stroke_a'])}-{_span(ev['stroke_b'])} (t1: {ev['t_a']}, t2: {ev['t_b']})" for ev in e2e]
                html.append(f"<div style='margin-bottom:6px;'><b>[端点对接]：</b> {' 、 '.join(strs)}</div>")
                
            if t_juncs:
                strs = [f"{_span(ev['guest'])} (t:{ev['guest_t']}) 搭在 {_span(ev['host'])} (t:{ev['host_t']}) 上 [夹角:{ev.get('angle',0)}°]" for ev in t_juncs]
                html.append(f"<div style='margin-bottom:6px;'><b>[T型搭接]：</b> {' 、 '.join(strs)}</div>")
                
            if x_juncs:
                strs = [f"{_span(ev['stroke_a'])} (t:{ev['t_a']}) 交叉 {_span(ev['stroke_b'])} (t:{ev['t_b']}) [夹角:{ev.get('angle',0)}°]" for ev in x_juncs]
                html.append(f"<div style='margin-bottom:6px;'><b>[X型交叉]：</b> {' 、 '.join(strs)}</div>")
                
            if not topology_events: 
                html.append("<div style='color:#777; margin-bottom:6px;'>该字符 JSON 无任何相交事件记录。</div>")
                
            if cycles:
                c_strs = []
                for cycle in cycles:
                    # 适配新的 cycle 字典格式 (成员 + 旋向)
                    styled_nodes = [_span(n) for n in cycle.get("members", [])]
                    orient = "顺时针" if cycle.get("orientation") == "cw" else "逆时针"
                    c_strs.append(f"{' '.join(styled_nodes)} (旋向: {orient})")
                html.append(f"<div style='margin-top:5px;'><b>[闭环结构]：</b> {' &nbsp;|&nbsp; '.join(c_strs)}</div>")
                
            topo_box.setHtml("".join(html))
        else:
            topo_box.setHtml("<span style='color:red;'>⚠️ Phase 2 Topo Data missing.</span>")
            
        self.detail_layout.addWidget(topo_box, stretch=1)
        self.stacked.setCurrentIndex(1)


# ==========================================
# 🚪 主窗口 (左侧文件分发)
# ==========================================
class AuditorAppMain(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Auditor: Global Visualization")
        self.setGeometry(50, 50, 1600, 900)
        
        self.splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.splitter)
        
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        
        # 1. 全部完成
        lbl_c = QLabel("✅ Fully Completed Fonts")
        lbl_c.setStyleSheet("font-weight: bold; color: #4CAF50;")
        sidebar_layout.addWidget(lbl_c)
        self.list_completed = QListWidget(); self.list_completed.setStyleSheet("background-color: #F1F8E9;")
        self.list_completed.itemClicked.connect(self.on_font_selected)
        sidebar_layout.addWidget(self.list_completed)
        
        # 2. 部分完成
        lbl_p = QLabel("⏳ Progressing Fonts")
        lbl_p.setStyleSheet("font-weight: bold; color: #FF9800;")
        sidebar_layout.addWidget(lbl_p)
        self.list_progress = QListWidget(); self.list_progress.setStyleSheet("background-color: #FFF3E0;")
        self.list_progress.itemClicked.connect(self.on_font_selected)
        sidebar_layout.addWidget(self.list_progress)
        
        # 3. 未动工
        lbl_u = QLabel("⚪ Untouched Fonts")
        lbl_u.setStyleSheet("font-weight: bold; color: #9E9E9E;")
        sidebar_layout.addWidget(lbl_u)
        self.list_untouched = QListWidget(); self.list_untouched.setStyleSheet("background-color: #F5F5F5;")
        self.list_untouched.itemClicked.connect(self.on_font_selected)
        sidebar_layout.addWidget(self.list_untouched)
        
        self.workspace_stack = QStackedWidget()
        self.workspace_stack.addWidget(QLabel("👈 Select a font to begin audit..."))
        
        self.splitter.addWidget(sidebar)
        self.splitter.addWidget(self.workspace_stack)
        self.splitter.setSizes([280, 1320])
        
        self.load_font_list()

    def load_font_list(self):
        for lw in [self.list_completed, self.list_progress, self.list_untouched]: lw.clear()
        if not os.path.exists(FONTS_DIR): return

        for file in os.listdir(FONTS_DIR):
            if file.lower().endswith(('.ttf', '.otf')):
                font_filename = os.path.splitext(file)[0]
                meta_file = os.path.join(META_DIR, f"{font_filename}_meta.json")
                topo_file = os.path.join(TOPO_OUT_DIR, f"{font_filename}_topo.json")
                
                topo_count = 0
                if os.path.exists(topo_file):
                    try:
                        with open(topo_file, 'r', encoding='utf-8') as f: topo_count = len(json.load(f))
                    except: pass
                    
                banned_count, phase1_count = 0, 0
                if os.path.exists(meta_file):
                    try:
                        with open(meta_file, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                            banned_count = len(meta.get("banned", []))
                            phase1_count = len(meta.get("raw_edges", {}))
                    except: pass
                
                if topo_count == 0 and banned_count == 0 and phase1_count == 0:
                    self.list_untouched.addItem(file)
                else:
                    total_chars = 999999 
                    try:
                        from fontTools.ttLib import TTFont
                        ttfont = TTFont(os.path.join(FONTS_DIR, file))
                        total_chars = len(ttfont.getBestCmap())
                        ttfont.close()
                    except: pass
                    
                    if topo_count + banned_count >= total_chars and total_chars > 0:
                        self.list_completed.addItem(file)
                    else:
                        self.list_progress.addItem(file)

    def on_font_selected(self, item):
        sender = self.sender()
        for lw in [self.list_completed, self.list_progress, self.list_untouched]:
            if lw != sender: lw.clearSelection()

        font_path = os.path.join(FONTS_DIR, item.text())
        if self.workspace_stack.count() > 1:
            old = self.workspace_stack.widget(1)
            self.workspace_stack.removeWidget(old)
            old.deleteLater()
            
        new_ws = AuditWorkspace(font_path)
        self.workspace_stack.addWidget(new_ws)
        self.workspace_stack.setCurrentIndex(1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("light", custom_colors={"primary": "#2196F3"})
    window = AuditorAppMain()
    window.show()
    sys.exit(app.exec_())