import os
import sys
import copy
import io
import numpy as np
import qdarktheme
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QScrollArea, 
                             QStackedWidget, QGridLayout, QListWidget, QSplitter, QFrame)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt

from data_manager import DatasetManager
from geometry_vision import (render_unicode_glyph, fit_bezier_basic_with_error, 
                             regress_width_dt_fast, cubic_bezier_np)

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR = os.path.join(SCRIPT_DIR, "../dataset/alien_tensors_raw")
CANVAS_SIZE = 400

# 🌟 新增：告诉时光机去哪里找独立录像
ACTION_LOG_DIR = os.path.join(SCRIPT_DIR, "action_logs")

# ==========================================
# 🎨 离线渲染引擎 (将张量绘制为 QPixmap)
# ==========================================
class StateRenderer:
    def __init__(self):
        try: self.cmap = plt.colormaps['tab20']
        except AttributeError: self.cmap = plt.get_cmap('tab20')

    def render_top_image(self, binary, edges, size=(2.5, 2.5)):
        """渲染上方图：带顺序和颜色的原始/当前笔画"""
        fig = plt.figure(figsize=size, dpi=80)
        fig.patch.set_facecolor('#FFFFFF')
        ax = fig.add_subplot(111)
        ax.imshow(binary, cmap='gray', alpha=0.15)
        for e in edges:
            color = self.cmap((e['id'] % 20))
            path = np.array(e['path'])
            if len(path) > 1:
                # 添加箭头指示笔画方向
                ax.plot(path[:, 0], path[:, 1], c=color, lw=4, alpha=0.8)
                ax.plot(path[0, 0], path[0, 1], marker='o', color=color, markersize=5) # 起点
        ax.axis('off')
        return self._fig_to_pixmap(fig)

    def render_bottom_image(self, binary, edges, dt_map, size=(2.5, 2.5)):
        """渲染下方图：物理轮廓反解还原的纯黑实体图"""
        fig = plt.figure(figsize=size, dpi=80)
        fig.patch.set_facecolor('#FFFFFF')
        ax = fig.add_subplot(111)
        ax.imshow(np.ones_like(binary), cmap='gray', vmin=0, vmax=1) 
        
        for e in edges:
            path = np.array(e['path'])
            if len(path) < 2: continue
            p_opt, _ = fit_bezier_basic_with_error(path)
            w_opt = regress_width_dt_fast(p_opt, dt_map)
            
            ts_dense = np.linspace(0, 1, 60)[:, None]
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
    
    def render_thumbnail(self, binary, size=(1.5, 1.5)):
        """🌟 新增：渲染纯净的原始形态微缩图"""
        fig = plt.figure(figsize=size, dpi=80)
        fig.patch.set_facecolor('#FFFFFF')
        ax = fig.add_subplot(111)
        # 直接把黑白的字形矩阵画出来
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
# 🏠 右侧工作区 (画廊 + 时光机)
# ==========================================
class PreviewWorkspace(QWidget):
    def __init__(self, font_path):
        super().__init__()
        self.font_path = font_path
        self.font_filename = os.path.basename(font_path).split('.')[0]
        self.db = DatasetManager(SCRIPT_DIR, self.font_filename)
        self.renderer = StateRenderer()

        # 🌟 新增这行：用来把画好的图存进内存，秒加载
        self.thumbnail_cache = {}

        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.stacked = QStackedWidget()
        
        # 页面 1：画廊 (微缩图网格)
        self.page_gallery = QWidget()
        gal_layout = QVBoxLayout(self.page_gallery)
        title = QLabel(f"✅ Completed Characters for: {self.font_filename}")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #333;")
        gal_layout.addWidget(title)
        
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        container = QWidget(); self.grid = QGridLayout(container)
        scroll.setWidget(container)
        gal_layout.addWidget(scroll)
        
        # 页面 2：时光机详情页
        self.page_detail = QWidget()
        self.detail_layout = QVBoxLayout(self.page_detail)
        
        self.stacked.addWidget(self.page_gallery)
        self.stacked.addWidget(self.page_detail)
        self.layout.addWidget(self.stacked)
        
        self.load_gallery()

    def load_gallery(self):
        completed_keys = list(self.db.annotated_outlines.keys())
        cols = 6
        for idx, hex_key in enumerate(completed_keys):
            char = chr(int(hex_key[2:], 16))
            
            # 🌟 创建独立的卡片容器
            card = QFrame()
            card.setStyleSheet("background-color: #FFFFFF; border: 2px solid #E0E0E0; border-radius: 8px;")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 10, 10, 10)
            
            # 1. 顶部：渲染真实字体原貌的微缩图
            if hex_key not in self.thumbnail_cache:
                binary = render_unicode_glyph(self.font_path, char, CANVAS_SIZE)
                self.thumbnail_cache[hex_key] = self.renderer.render_thumbnail(binary)
                
            img_label = QLabel()
            img_label.setPixmap(self.thumbnail_cache[hex_key])
            img_label.setAlignment(Qt.AlignCenter)
            
            # 2. 中间：保留文字与 UXXXX 编码
            title = QLabel(f"'{char}'\n{hex_key}")
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("font-size: 16px; font-weight: bold; color: #424242; border: none;")
            
            # 3. 底部：查看时光机的动作按钮
            btn_view = QPushButton("🔍 View Audit")
            btn_view.setCursor(Qt.PointingHandCursor)
            btn_view.setStyleSheet("background-color: #FF9800; color: white; padding: 8px; border-radius: 4px; font-weight: bold;")
            btn_view.clicked.connect(lambda checked, hk=hex_key: self.open_timeline(hk))
            
            card_layout.addWidget(img_label)
            card_layout.addWidget(title)
            card_layout.addWidget(btn_view)
            
            self.grid.addWidget(card, idx // cols, idx % cols)
        

    def open_timeline(self, hex_key):
        # 1. 清理旧的详情页视图
        while self.detail_layout.count():
            item = self.detail_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
            
        char = chr(int(hex_key[2:], 16))
        # 🌟 核心修改：从独立的 JSON 文件中读取时光机数据
        action_log = []
        action_file_path = os.path.join(ACTION_LOG_DIR, f"{self.font_filename}_actions.json")
        if os.path.exists(action_file_path):
            import json
            with open(action_file_path, 'r', encoding='utf-8') as f:
                action_data = json.load(f)
                action_log = action_data.get(hex_key, [])
        
        # 2. 顶部导航栏
        top_bar = QHBoxLayout()
        btn_back = QPushButton("🔙 Back")
        btn_back.setFixedWidth(100)
        btn_back.setStyleSheet("padding: 10px; background-color: #2196F3; color: white; font-weight: bold; border-radius: 4px;")
        btn_back.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        top_bar.addWidget(btn_back)
        
        lbl_title = QLabel(f"Audit Trail: '{char}' ({hex_key})")
        lbl_title.setStyleSheet("font-size: 22px; font-weight: bold;")
        top_bar.addWidget(lbl_title)
        top_bar.addStretch()
        self.detail_layout.addLayout(top_bar)
        
        line = QFrame(); line.setFrameShape(QFrame.HLine); self.detail_layout.addWidget(line)

        if not action_log:
            self.detail_layout.addWidget(QLabel("⚠️ No action history found. Please annotate with the updated main.py."))
            self.detail_layout.addStretch()
            self.stacked.setCurrentIndex(1)
            return

        # 准备数据底板
        binary = render_unicode_glyph(self.font_path, char, CANVAS_SIZE)
        dt_map = distance_transform_edt(binary)

        # ==========================================
        # 🌟 区域 A：固定在最上方的【原始形态】参考图
        # ==========================================
        ref_layout = QHBoxLayout()
        ref_label = QLabel("🎯 <b>Original Raw Topology</b><br><span style='color:#757575; font-size:12px;'>Reference fixed at top</span>")
        ref_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        initial_edges = action_log[0]["edges"]
        img_original = QLabel()
        pix_orig = self.renderer.render_top_image(binary, initial_edges, size=(3.5, 3.5)) # 稍微大一点
        img_original.setPixmap(pix_orig)
        img_original.setStyleSheet("border: 2px dashed #9E9E9E; background-color: white;")
        
        ref_layout.addWidget(img_original)
        ref_layout.addWidget(ref_label)
        ref_layout.addStretch()
        self.detail_layout.addLayout(ref_layout)

        # ==========================================
        # 🌟 区域 B：带滑块拖拽的【步骤流转图】
        # ==========================================
        lbl_timeline_title = QLabel("⏳ <b>Step-by-Step Transformation</b>")
        lbl_timeline_title.setStyleSheet("margin-top: 15px; font-size: 16px;")
        self.detail_layout.addWidget(lbl_timeline_title)

        scroll_time = QScrollArea()
        scroll_time.setWidgetResizable(True)
        scroll_time.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # 关闭垂直滚动条
        scroll_time.setStyleSheet("background-color: #FAFAFA;")
        
        time_container = QWidget()
        time_layout = QHBoxLayout(time_container)
        time_layout.setAlignment(Qt.AlignLeft)
        time_layout.setSpacing(15)

        for i, step in enumerate(action_log):
            action_name = step["action"]
            edges = step["edges"]
            stroke_count = len(edges)
            
            # --- 渲染箭头与操作名称 ---
            if i > 0:
                arrow_container = QVBoxLayout()
                arrow_container.setAlignment(Qt.AlignCenter)
                
                lbl_action = QLabel(action_name)
                lbl_action.setAlignment(Qt.AlignCenter)
                lbl_action.setStyleSheet("font-size: 14px; font-weight: bold; color: #E91E63; background: #FFCDD2; padding: 4px; border-radius: 4px;")
                
                lbl_arrow = QLabel("➔")
                lbl_arrow.setAlignment(Qt.AlignCenter)
                lbl_arrow.setStyleSheet("font-size: 36px; color: #BDBDBD; font-weight: bold;")
                
                arrow_container.addWidget(lbl_action)
                arrow_container.addWidget(lbl_arrow)
                time_layout.addLayout(arrow_container)

            # --- 渲染 "吕" 字型状态卡片 ---
            card = QFrame()
            card.setStyleSheet("background-color: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 8px;")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 10, 10, 10)
            
            # 吕 - 上半部：带颜色拓扑
            lbl_top = QLabel()
            pix_top = self.renderer.render_top_image(binary, edges)
            lbl_top.setPixmap(pix_top)
            lbl_top.setAlignment(Qt.AlignCenter)
            
            # 吕 - 下半部：纯黑实体还原
            lbl_bot = QLabel()
            pix_bot = self.renderer.render_bottom_image(binary, edges, dt_map)
            lbl_bot.setPixmap(pix_bot)
            lbl_bot.setAlignment(Qt.AlignCenter)
            
            # 底部信息：当前笔画数
            lbl_info = QLabel(f"Strokes: {stroke_count}")
            lbl_info.setAlignment(Qt.AlignCenter)
            lbl_info.setStyleSheet("font-size: 14px; font-weight: bold; color: #424242; padding-top: 5px; border: none;")
            
            if i == 0: lbl_info.setText(f"🚀 Initial ({stroke_count} strokes)")
            if i == len(action_log)-1: lbl_info.setText(f"🏁 Final ({stroke_count} strokes)")
            
            card_layout.addWidget(lbl_top)
            card_layout.addWidget(lbl_bot)
            card_layout.addWidget(lbl_info)
            time_layout.addWidget(card)

        scroll_time.setWidget(time_container)
        self.detail_layout.addWidget(scroll_time, 1) # 给时间轴分配剩余的所有拉伸空间
        
        self.stacked.setCurrentIndex(1)


# ==========================================
# 🚪 主窗口
# ==========================================
class PreviewAppMain(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Font Data Auditor")
        self.setGeometry(50, 50, 1500, 850)
        
        self.splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.splitter)
        
        # 👇👇👇 替换为：上下分组的侧边栏布局
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        
        # 上半部分：已标注 (Annotated)
        lbl_annotated = QLabel("✅ Annotated Fonts")
        lbl_annotated.setStyleSheet("font-size: 15px; font-weight: bold; color: #4CAF50;")
        sidebar_layout.addWidget(lbl_annotated)
        
        self.list_annotated = QListWidget()
        self.list_annotated.itemClicked.connect(self.on_font_selected)
        self.list_annotated.setStyleSheet("background-color: #F1F8E9; border: 1px solid #C8E6C9; border-radius: 4px;")
        sidebar_layout.addWidget(self.list_annotated)
        
        # 下半部分：未标注 (Pending)
        lbl_unannotated = QLabel("⏳ Pending Fonts")
        lbl_unannotated.setStyleSheet("font-size: 15px; font-weight: bold; color: #FF9800; margin-top: 10px;")
        sidebar_layout.addWidget(lbl_unannotated)
        
        self.list_unannotated = QListWidget()
        self.list_unannotated.itemClicked.connect(self.on_font_selected)
        self.list_unannotated.setStyleSheet("background-color: #FFF3E0; border: 1px solid #FFE0B2; border-radius: 4px;")
        sidebar_layout.addWidget(self.list_unannotated)

        
        self.workspace_stack = QStackedWidget()
        self.workspace_stack.addWidget(QLabel("👈 Select a font to begin audit..."))
        
        self.splitter.addWidget(sidebar)
        self.splitter.addWidget(self.workspace_stack)
        self.splitter.setSizes([250, 1250])
        
        self.load_font_list()

    def load_font_list(self):
        self.list_annotated.clear()
        self.list_unannotated.clear()
        
        if not os.path.exists(FONTS_DIR): return
        
        # 获取纯净标注文件夹的路径
        anno_dir = os.path.join(SCRIPT_DIR, "annotations")
        os.makedirs(anno_dir, exist_ok=True)

        for file in os.listdir(FONTS_DIR):
            if file.lower().endswith(('.ttf', '.otf')):
                font_filename = os.path.splitext(file)[0]
                anno_file_path = os.path.join(anno_dir, f"{font_filename}.json")
                
                is_annotated = False
                
                # 🌟 核心逻辑：文件存在，并且里面至少有一个字符的数据，才算作“已标注”
                if os.path.exists(anno_file_path):
                    try:
                        import json
                        with open(anno_file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if len(data) > 0: 
                                is_annotated = True
                    except:
                        pass
                
                # 分发到对应的列表中
                if is_annotated:
                    self.list_annotated.addItem(file)
                else:
                    self.list_unannotated.addItem(file)
        

    def on_font_selected(self, item):
        # 🌟 互斥体验：点击上方列表时，清除下方列表的选中状态，反之亦然
        sender = self.sender()
        if sender == self.list_annotated:
            self.list_unannotated.clearSelection()
        elif sender == self.list_unannotated:
            self.list_annotated.clearSelection()

        font_path = os.path.join(FONTS_DIR, item.text())
        
        # ... 后面的切换工作区代码保持不变 ...
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