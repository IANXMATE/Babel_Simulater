import os
import sys
import random
import json
import copy
import warnings
import io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import traceback
from PIL import Image, ImageDraw, ImageFont
from skimage import morphology
from skan import Skeleton, summarize
from scipy.optimize import least_squares
from scipy.ndimage import distance_transform_edt, map_coordinates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFrame, QMessageBox, 
                             QScrollArea, QStackedWidget, QGridLayout, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from fontTools.ttLib import TTFont
import qdarktheme

from skimage.filters import threshold_otsu
from skimage.morphology import medial_axis

#####

from geometry_vision import (
    extract_all_real_chars, render_unicode_glyph, collapse_degree2_nodes,
    prune_spurs, stitch_paths, cubic_bezier_np, fit_bezier_basic_with_error,
    split_pixel_path_adaptively, regress_width_dt_fast
)

from data_manager import DatasetManager


# ==========================================
# 🕵️ 终极侦探雷达：查出到底是谁在阻止 AI 启动！
# ==========================================
print("\n" + "="*40)
print("🔍 [诊断系统] 正在检查 AI 引擎状态...")

# 1. 检查物理文件
_script_dir = os.path.dirname(os.path.abspath(__file__))
_model_path = os.path.join(_script_dir, "ml_engine", "graph_editor_best.pth")
print(f"📍 寻找权重文件: {_model_path}")
print(f"   -> 物理文件存在吗？: {os.path.exists(_model_path)}")

# 2. 检查代码导入
HAS_AI_MODULE = False
try:
    from ml_engine.ai_auto_initializer import UICompatibleAIExecutor
    HAS_AI_MODULE = True
    print("✅ AI 代码模块导入成功！")
except Exception as e:
    # 🌟 这里是关键：如果是代码导入报错，它会把真实的错误原因打印出来！
    print(f"❌ AI 代码模块导入彻底失败！真实报错原因如下：\n   >>> {type(e).__name__}: {e}")

print("="*40 + "\n")
# ==========================================

warnings.filterwarnings("ignore")
mpl.rcParams['axes.unicode_minus'] = False 

# ==========================================
# ⚙️ Global Config & Paths
# ==========================================
CANVAS_SIZE = 400         
MAX_BEZIER_ERROR = 3   
MAX_SPUR_LENGTH = 20.0    

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(SCRIPT_DIR, '../dataset/alien_tensors_storage/NotoSansTamil[wdth,wght].ttf')

if not os.path.exists(FONT_PATH):
    FONT_PATH = "arial.ttf"

# 🌟 NEW: Data Persistence Directories


# ==========================================
# 🖥️ Multi-Page Annotation Application V12
# ==========================================
class ModernAnnotationApp(QMainWindow):
    def __init__(self, font_path):
        super().__init__()
        self.font_path = font_path
        self.font_filename = os.path.basename(self.font_path)
        self.setWindowTitle(f"AI Vector Router - Data Factory ({self.font_filename})")
        self.setGeometry(50, 50, 1500, 750) 
        
        # 实例化数据中枢
        self.db = DatasetManager(SCRIPT_DIR, self.font_filename)
        self.font_charset = extract_all_real_chars(self.font_path)
        self.history_stack = [] 
        self.selected_edge_ids = [] 
        self.last_click_coord = None 
        self.char = None
        self.thumbnail_cache = {} 
        
        try: self.cmap = plt.colormaps['tab20']
        except AttributeError: self.cmap = plt.get_cmap('tab20')
        
        self.bezier_cache = {}
        
        # ==========================================
        # 🛡️ 严苛模式：AI 初始化逻辑
        # ==========================================
        self.init_mode = "raw" 
        self.pure_raw_edges = []
        self.ai_executor = None
        
        model_dir = os.path.join(SCRIPT_DIR, "ml_engine")
        model_path = os.path.join(model_dir, "graph_editor_best.pth")
        
        # 强制检查
        path_exists = os.path.exists(model_path)
        print(f"DEBUG: 正在寻找模型路径: {model_path}")
        print(f"DEBUG: 检查结果 (os.path.exists): {path_exists}")
        
        # 只有当模块导入成功且模型物理存在时，才标记为 has_ai
        self.has_ai = HAS_AI_MODULE and path_exists
        
        if self.has_ai:
            try:
                print("DEBUG: 正在尝试实例化 UICompatibleAIExecutor...")
                self.ai_executor = UICompatibleAIExecutor(model_filename="graph_editor_best.pth")
                print("DEBUG: AI 引擎已成功挂载。")
            except Exception as e:
                self.has_ai = False
                print(f"❌ AI 引擎实例化失败: {str(e)}")
                # 打印详细堆栈以供排查
                traceback.print_exc()
        else:
            print(f"DEBUG: 无法启用 AI 功能。模块存在: {HAS_AI_MODULE}, 文件存在: {path_exists}")
        # ==========================================
        
        # Build UI
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        self.init_page_annotation()
        self.init_page_completed()
        self.init_page_banned()
        
        self.stacked_widget.addWidget(self.page_annotation)
        self.stacked_widget.addWidget(self.page_completed)
        self.stacked_widget.addWidget(self.page_banned)
        
        self.action_next_char()

    # ==========================================
    # 🌟 NEW: AI 与 Raw 模式切换逻辑
    # ==========================================
    def action_toggle_init(self):
        if not self.has_ai or self.ai_executor is None:
            QMessageBox.warning(self, "AI Offline", "AI 模型未就绪")
            return

        # ==========================================
        # 🛡️ 终极自愈：如果快照为空但画板有线，强制原地同步！
        # ==========================================
        if len(self.pure_raw_edges) == 0 and len(self.edges) > 0:
            print("DEBUG: 🚨 检测到快照脱节！自动从当前画板同步纯净数据...")
            self.pure_raw_edges = copy.deepcopy(self.edges)
        # ==========================================

        # 备份当前画板快照
        backup_edges = copy.deepcopy(self.edges)
        
        try:
            if self.init_mode == "raw":
                # 1. 尝试执行 AI 预初始化 (现在绝对不可能传进去空数组了！)
                ai_output = self.ai_executor.generate_ai_init_graph(copy.deepcopy(self.pure_raw_edges))
                
                if not ai_output:
                    raise ValueError("AI 返回了空路径集，拒绝应用！")
                
                self.edges = ai_output
                self.init_mode = "ai"
                self.btn_toggle_init.setText("🔄 Revert to Raw Init")
            
            else:
                # 2. 物理还原
                self.edges = copy.deepcopy(self.pure_raw_edges)
                self.init_mode = "raw"
                self.btn_toggle_init.setText("🧠 Use AI Init")
                
            # 3. 强制重构渲染环境
            self.selected_edge_ids.clear()
            self.bezier_cache.clear()
            self.history_stack.clear()
            self.save_state()
            
            self.ax_main.clear()
            self.ax_prev.clear()
            self.update_canvas()
            self.update_palette()

        except Exception as e:
            print(f"DEBUG: 发生严重渲染/处理错误: {e}")
            self.edges = backup_edges
            self.init_mode = "raw"
            self.btn_toggle_init.setText("🧠 Use AI Init")
            self.update_canvas()
            QMessageBox.critical(self, "渲染错误", f"操作已撤销。\n原因: {e}")

    # ==========================================
    # 🌟 MODIFIED: 修改原有的图扑加载逻辑，使其感知当前模式
    # ==========================================
    def load_char_topology(self):
        skel, distance = medial_axis(self.binary, return_distance=True)
        skel_obj = Skeleton(skel)
        branch_data = summarize(skel_obj)
        G = nx.MultiGraph()
        
        # ... (保留你原来的图构建和减枝逻辑不变) ...
        for index, row in branch_data.iterrows():
            coords = skel_obj.path_coordinates(index)
            if len(coords) > 2:
                path = np.column_stack([coords[:, 1], coords[:, 0]])
                src, dst = int(row['node-id-src']), int(row['node-id-dst'])
                if np.linalg.norm(path[0] - skel_obj.coordinates[src][::-1]) > 1.0: path = path[::-1]
                G.add_edge(src, dst, key=index, path=path)
                
        G = prune_spurs(G, max_length=MAX_SPUR_LENGTH) 
        G = collapse_degree2_nodes(G)
        
        # 填充 edges...
        self.edges = []
        global_id = 0
        for u, v, k, d in G.edges(keys=True, data=True):
            path = d['path']
            sub_paths = split_pixel_path_adaptively(path)
            for sp in sub_paths:
                # 🛡️ 核心修复：在这里进行深层拷贝，确保没有任何引用残留
                # 必须复制 path 数组，否则 sp 仍会指向原始引用
                self.edges.append({'id': global_id, 'path': sp.copy()})
                global_id += 1
        
        # 🛡️ 最终极防线：将这块内存彻底深拷贝给 pure_raw_edges
        # 这一步之后，无论你对 self.edges 怎么折腾，pure_raw_edges 都会纹丝不动
        self.pure_raw_edges = copy.deepcopy(self.edges)
        
        # 🌟 关键修改点 2：如果当前处于 AI 模式，自动顺延给下一个字进行 AI 初始化
        if self.init_mode == "ai" and self.has_ai:
            if self.ai_executor is None:
                self.title_label.setText("Waking up AI...")
                QApplication.processEvents()
                self.ai_executor = UICompatibleAIExecutor()
            self.edges = self.ai_executor.generate_ai_init_graph(self.pure_raw_edges)

        self.bezier_cache.clear() 
        self.selected_edge_ids = []
        self.history_stack.clear()
        self.last_click_coord = None
        self.save_state()
        self.update_canvas()
        self.update_palette()
    
    # ---------------- Data Persistence ----------------
    def load_data(self):
        if os.path.exists(self.anno_json_path):
            with open(self.anno_json_path, 'r', encoding='utf-8') as f:
                self.db.annotated_outlines = json.load(f)
        if os.path.exists(self.meta_json_path):
            with open(self.meta_json_path, 'r', encoding='utf-8') as f:
                self.db.meta_data = json.load(f)

    def save_data(self):
        with open(self.anno_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.db.annotated_outlines, f, indent=2, ensure_ascii=False)
        with open(self.meta_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.db.meta_data, f, indent=2, ensure_ascii=False)

    # ---------------- Layout Setup ----------------
    def init_page_annotation(self):
        self.page_annotation = QWidget()
        layout = QHBoxLayout(self.page_annotation)

        # Left Panel (Controls)
        control_panel = QVBoxLayout()
        control_panel.setSpacing(10)
        
        # Navigation
        nav_layout = QHBoxLayout()
        btn_nav_comp = QPushButton("🎨 View Completed")
        btn_nav_comp.clicked.connect(self.go_to_completed)
        btn_nav_comp.setStyleSheet("background-color: #673AB7; color: white; font-weight: bold; padding: 6px;")
        
        btn_nav_ban = QPushButton("🚫 View Banned")
        btn_nav_ban.clicked.connect(self.go_to_banned)
        btn_nav_ban.setStyleSheet("background-color: #607D8B; color: white; font-weight: bold; padding: 6px;")
        
        nav_layout.addWidget(btn_nav_comp)
        nav_layout.addWidget(btn_nav_ban)
        control_panel.addLayout(nav_layout)

        line0 = QFrame(); line0.setFrameShape(QFrame.HLine); control_panel.addWidget(line0)

        self.title_label = QLabel("Loading...")
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #1A1A1A;")
        control_panel.addWidget(self.title_label)
        
        self.stroke_info_label = QLabel("Beziers: 0")
        self.stroke_info_label.setStyleSheet("font-size: 15px; color: #E91E63; font-weight: bold;")
        control_panel.addWidget(self.stroke_info_label)
        
        line1 = QFrame(); line1.setFrameShape(QFrame.HLine); control_panel.addWidget(line1)
        
        # ==========================================
        # 🌟 NEW: AI 状态指示器与切换按钮
        # ==========================================
        ai_status_layout = QHBoxLayout()
        
        self.lbl_ai_status = QLabel("🟢 AI Active" if self.has_ai else "🔴 AI Offline")
        self.lbl_ai_status.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {'#4CAF50' if self.has_ai else '#F44336'};")
        
        self.btn_toggle_init = QPushButton("🧠 Use AI Init")
        self.btn_toggle_init.clicked.connect(self.action_toggle_init)
        self.btn_toggle_init.setStyleSheet("padding: 10px; background-color: #673AB7; color: white; font-weight: bold;")
        
        ai_status_layout.addWidget(self.lbl_ai_status)
        ai_status_layout.addWidget(self.btn_toggle_init)
        control_panel.addLayout(ai_status_layout)
        # ==========================================


        btn_merge = QPushButton("Merge Strokes (M)"); btn_merge.clicked.connect(self.action_merge)
        btn_merge.setStyleSheet("padding: 10px; background-color: #4CAF50; color: white; font-weight: bold;")
        
        btn_delete = QPushButton("Delete Stroke (D)"); btn_delete.clicked.connect(self.action_delete)
        btn_delete.setStyleSheet("padding: 10px; background-color: #F44336; color: white; font-weight: bold;")

        btn_break = QPushButton("Split at Point (B)"); btn_break.clicked.connect(self.action_breakpoint)
        btn_break.setStyleSheet("padding: 10px; background-color: #FF9800; color: white; font-weight: bold;")

        btn_prune = QPushButton("Prune Redundant (C)"); btn_prune.clicked.connect(self.action_prune_parallel)
        btn_prune.setStyleSheet("padding: 10px; background-color: #9C27B0; color: white; font-weight: bold;")

        btn_add_dot = QPushButton("Add Missing Dot (A)"); btn_add_dot.clicked.connect(self.action_add_dot)
        btn_add_dot.setStyleSheet("padding: 10px; background-color: #00BCD4; color: white; font-weight: bold;")

        btn_undo = QPushButton("Undo Last Action (U)"); btn_undo.clicked.connect(self.action_undo)
        btn_undo.setStyleSheet("padding: 10px; background-color: #757575; color: white; font-weight: bold;")

        btn_reset_char = QPushButton("Reset Current Char (R)"); btn_reset_char.clicked.connect(self.action_reset_char)
        btn_reset_char.setStyleSheet("padding: 10px; background-color: #795548; color: white; font-weight: bold;")

        btn_next_char = QPushButton("Next Random Char (N)"); btn_next_char.clicked.connect(self.action_next_char)
        btn_next_char.setStyleSheet("padding: 10px; background-color: #2196F3; color: white; font-weight: bold;")

        # 🌟 NEW: Save and Ban Buttons
        btn_ban = QPushButton("🚫 BAN Character"); btn_ban.clicked.connect(self.action_ban_char)
        btn_ban.setStyleSheet("padding: 12px; background-color: #000000; color: white; font-weight: bold; font-size: 13px;")

        btn_complete = QPushButton("✅ COMPLETE Annotation (Enter)"); btn_complete.clicked.connect(self.action_complete_annotation)
        btn_complete.setStyleSheet("padding: 14px; background-color: #E91E63; color: white; font-weight: bold; font-size: 14px;")

        control_panel.addWidget(btn_merge)
        control_panel.addWidget(btn_delete)
        control_panel.addWidget(btn_break)
        control_panel.addWidget(btn_prune)
        control_panel.addWidget(btn_add_dot)
        control_panel.addWidget(btn_undo)
        control_panel.addWidget(btn_reset_char)
        control_panel.addWidget(btn_next_char)
        control_panel.addStretch()
        control_panel.addWidget(btn_ban)
        control_panel.addWidget(btn_complete)

        # Right Panel (Canvas + Palette)
        right_panel = QVBoxLayout()
        self.palette_container = QWidget()
        self.palette_layout = QHBoxLayout(self.palette_container)
        self.palette_layout.setContentsMargins(5, 5, 5, 5); self.palette_layout.setSpacing(8)
        self.palette_layout.setAlignment(Qt.AlignLeft)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True); scroll_area.setWidget(self.palette_container)
        scroll_area.setMaximumHeight(60) 
        
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

    def init_page_completed(self):
        self.page_completed = QWidget()
        layout = QVBoxLayout(self.page_completed)
        
        top_bar = QHBoxLayout()
        btn_back = QPushButton("🔙 Back to Annotation")
        btn_back.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        top_bar.addWidget(btn_back)
        top_bar.addWidget(QLabel("<h2>🎨 Completed Characters Gallery</h2>"))
        top_bar.addStretch()
        layout.addLayout(top_bar)

        self.completed_scroll = QScrollArea()
        self.completed_scroll.setWidgetResizable(True)
        self.completed_container = QWidget()
        self.completed_grid = QGridLayout(self.completed_container)
        self.completed_scroll.setWidget(self.completed_container)
        layout.addWidget(self.completed_scroll)

    def init_page_banned(self):
        self.page_banned = QWidget()
        layout = QVBoxLayout(self.page_banned)
        
        top_bar = QHBoxLayout()
        btn_back = QPushButton("🔙 Back to Annotation")
        btn_back.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        top_bar.addWidget(btn_back)
        top_bar.addWidget(QLabel("<h2>🚫 Banned Characters Gallery</h2>"))
        top_bar.addStretch()
        layout.addLayout(top_bar)

        self.banned_scroll = QScrollArea()
        self.banned_scroll.setWidgetResizable(True)
        self.banned_container = QWidget()
        self.banned_grid = QGridLayout(self.banned_container)
        self.banned_scroll.setWidget(self.banned_container)
        layout.addWidget(self.banned_scroll)

    # ---------------- Multi-Page Navigation ----------------
    def go_to_completed(self):
        self.refresh_gallery(self.completed_grid, list(self.db.annotated_outlines.keys()), mode="completed")
        self.stacked_widget.setCurrentIndex(1)

    def go_to_banned(self):
        self.refresh_gallery(self.banned_grid, self.db.meta_data["banned"], mode="banned")
        self.stacked_widget.setCurrentIndex(2)

    def refresh_gallery(self, grid_layout, char_list, mode):
        # Clear layout
        while grid_layout.count():
            item = grid_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        cols = 4
        for idx, char in enumerate(char_list):
            card = QFrame()
            card.setStyleSheet("background-color: #FFFFFF; border: 1px solid #CCC; border-radius: 8px;")
            card_layout = QVBoxLayout(card)
            
            # Use Off-screen generation
            if f"{char}_{mode}" not in self.thumbnail_cache:
                self.thumbnail_cache[f"{char}_{mode}"] = self.generate_thumbnail(char, mode)
                
            img_label = QLabel()
            img_label.setPixmap(self.thumbnail_cache[f"{char}_{mode}"])
            img_label.setAlignment(Qt.AlignCenter)
            
            hex_code = f"U+{ord(char):04X}"
            title = QLabel(f"Char: '{char}' ({hex_code})")
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("font-weight: bold; border: none;")

            btn_action = QPushButton("Re-annotate" if mode == "completed" else "Unban")
            if mode == "completed":
                btn_action.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
                btn_action.clicked.connect(lambda checked, c=char: self.action_reannotate(c))
            else:
                btn_action.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
                btn_action.clicked.connect(lambda checked, c=char: self.action_unban(c))

            card_layout.addWidget(img_label)
            card_layout.addWidget(title)
            card_layout.addWidget(btn_action)
            grid_layout.addWidget(card, idx // cols, idx % cols)

    def generate_thumbnail(self, char, mode):
        """🌟 Offline rendering to memory buffer to avoid blocking UI with heavy Matplotlib figures"""
        fig = plt.figure(figsize=(4, 2) if mode == "completed" else (2, 2), dpi=80)
        binary = render_unicode_glyph(self.font_path, char)
        
        if mode == "banned":
            ax = fig.add_subplot(111)
            ax.imshow(binary, cmap='gray')
            ax.axis('off')
        else:
            ax1 = fig.add_subplot(121)
            ax1.imshow(binary, cmap='gray')
            ax1.axis('off')
            ax2 = fig.add_subplot(122)
            ax2.imshow(binary, cmap='gray', alpha=0.1)
            raw_edges = self.db.meta_data["raw_edges"].get(char, [])
            
            try:
                for e in raw_edges:
                    path = np.array(e["path"])
                    if len(path) > 1:
                        ax2.plot(path[:, 0], path[:, 1], linewidth=2, alpha=0.8)
            except Exception: pass
            ax2.axis('off')
            
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        qimg = QImage()
        qimg.loadFromData(buf.getvalue())
        return QPixmap.fromImage(qimg)

    # ---------------- Actions for DB / Persistence ----------------
    def action_ban_char(self):
        if self.char:
            if self.char not in self.db.meta_data["banned"]:
                self.db.meta_data["banned"].append(self.char)
            self.save_data()
            self.action_next_char()

    def action_unban(self, char):
        if char in self.db.meta_data["banned"]:
            self.db.meta_data["banned"].remove(char)
            self.save_data()
            self.go_to_banned() # refresh gallery

    def action_complete_annotation(self):
        if not self.edges: return
        
        # 🌟 修改点：只保留纯粹的几何张量数据，干掉无用的 char 字段
        final_tokens = []
        for edge in self.edges:
            eid = edge['id']
            if eid in self.bezier_cache:
                p_opt, w_opt = self.bezier_cache[eid]
                final_tokens.append({
                    "bezier_id": eid,
                    "mother_bezier": p_opt.tolist(), 
                    "width_bezier": w_opt
                })
                
        # 外部依然使用 UXXXX 作为 Key 来保存，但内部已经没有文本属性了
        self.db.annotated_outlines[self.char] = final_tokens
        
        # 记录高精度拓扑历史，用于回滚
        raw_edges_serializable = []
        for e in self.edges:
            raw_edges_serializable.append({"id": e['id'], "path": e['path'].tolist()})
        self.db.meta_data["raw_edges"][self.char] = raw_edges_serializable
        
        # 写入磁盘并清理缓存刷新页面
        self.save_data()
        self.thumbnail_cache.pop(f"{self.char}_completed", None) 
        self.action_next_char()


    def action_reannotate(self, char):
        self.char = char
        unicode_hex = f"U+{ord(self.char):04X}"
        self.title_label.setText(f"[RE-EDIT] Target: '{self.char}' ({unicode_hex})")
        self.binary = render_unicode_glyph(self.font_path, self.char)
        self.dt_map = distance_transform_edt(self.binary)
        
        # Load raw edges from metadata
        raw = self.db.meta_data["raw_edges"].get(self.char, [])
        self.edges = [{"id": r["id"], "path": np.array(r["path"])} for r in raw]
        
        # 🌟 补上这一句，保证重新标注时快照也是满的！
        self.pure_raw_edges = copy.deepcopy(self.edges)
        
        self.bezier_cache.clear()
        self.selected_edge_ids.clear()
        self.history_stack.clear()
        self.save_state()
        self.update_canvas()
        self.update_palette()
        self.stacked_widget.setCurrentIndex(0) # Go to Annotation Page

    def action_next_char(self):
        max_attempts = 500
        valid_char_found = False
        
        for _ in range(max_attempts):
            candidate = random.choice(self.font_charset)
            
            # 🌟 NEW: Skip if already Banned or Completed!
            if candidate in self.db.meta_data["banned"] or candidate in self.db.annotated_outlines:
                continue
                
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
            
        if not valid_char_found: 
            QMessageBox.information(self, "Finished", "All valid characters have been annotated or banned!")
            return

        unicode_hex = f"U+{ord(self.char):04X}"
        self.title_label.setText(f"Target Char: '{self.char}' ({unicode_hex})")
        self.dt_map = distance_transform_edt(self.binary)
        self.load_char_topology()

    # ---------------- Original Canvas Actions ----------------
    def load_char_topology(self):
        skel, distance = medial_axis(self.binary, return_distance=True)
        skel_obj = Skeleton(skel)
        branch_data = summarize(skel_obj)
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
                btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 3px solid #000; color: #FFF; font-weight: bold; border-radius: 4px;")
            else:
                btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #CCC; color: #000; border-radius: 4px;")
            btn.clicked.connect(lambda checked, e=eid: self.toggle_selection(e))
            self.palette_layout.addWidget(btn)
        self.palette_layout.addStretch()

    def update_canvas(self):
        self.ax_ref.clear(); self.ax_ref.imshow(self.binary, cmap='gray'); self.ax_ref.set_title("1. Original Matrix"); self.ax_ref.axis('off')
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
        self.ax_main.set_title("2. Topology (Click & Edit)"); self.ax_main.axis('off')

        self.ax_prev.clear(); self.ax_prev.imshow(self.binary, cmap='gray', alpha=0.05); self.ax_prev.set_title("3. Strict 1:1 Bezier Preview"); self.ax_prev.axis('off')
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
            self.ax_prev.plot(curve[:, 0], curve[:, 1], color=color, linewidth=final_lw, solid_capstyle='round', alpha=final_alpha)
            self.ax_prev.plot(curve[:, 0], curve[:, 1], color='black', linewidth=1.5, alpha=final_alpha+0.1)
        self.canvas.draw()

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
        clicked_eid = self.edges[idx]['id']
        self.toggle_selection(clicked_eid)
        
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


    def action_merge(self):
        if len(self.selected_edge_ids) == 2:
            self.save_state()
            id_a, id_b = self.selected_edge_ids
            paths_to_stitch = [e['path'] for e in self.edges if e['id'] in (id_a, id_b)]
            new_unified_path = stitch_paths(paths_to_stitch)
            _, error = fit_bezier_basic_with_error(new_unified_path)
            
            sub_paths = []
            if error > MAX_BEZIER_ERROR:
                sub_paths = split_pixel_path_adaptively(new_unified_path)
                # 🌟 补回：触发合并失败/自适应断裂的警告提示框
                QMessageBox.warning(self, "Fitting Alert", 
                                    f"Merge resulted in a high deviation (Error: {error:.2f} > Limit: {MAX_BEZIER_ERROR}).\n\n"
                                    f"To maintain pure Bezier quality, the segment has been adaptively split into {len(sub_paths)} parts.")
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
            self.selected_edge_ids.clear(); self.bezier_cache.clear(); self.update_canvas(); self.update_palette()

    def action_prune_parallel(self):
        if len(self.selected_edge_ids) == 2:
            self.save_state()
            id_a, id_b = self.selected_edge_ids
            edge_a = next(e for e in self.edges if e['id'] == id_a)
            edge_b = next(e for e in self.edges if e['id'] == id_b)
            len_a = np.sum(np.linalg.norm(np.diff(edge_a['path'], axis=0), axis=1))
            len_b = np.sum(np.linalg.norm(np.diff(edge_b['path'], axis=0), axis=1))
            remove_id = id_a if len_a < len_b else id_b
            self.edges = [e for e in self.edges if e['id'] != remove_id]
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
        tiny_path = np.array([[cx, cy-1], [cx, cy], [cx, cy+1]])
        new_id = max([e['id'] for e in self.edges] + [-1]) + 1
        self.edges.append({'id': new_id, 'path': tiny_path})
        self.last_click_coord = None; self.update_canvas(); self.update_palette()

    def action_undo(self):
        if len(self.history_stack) > 1:
            self.history_stack.pop() 
            self.edges = copy.deepcopy(self.history_stack[-1])
            self.selected_edge_ids.clear(); self.last_click_coord = None; self.bezier_cache.clear()
            self.update_canvas(); self.update_palette()

    def action_reset_char(self):
        self.load_char_topology()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 🌟 2. 施展魔法：一键应用主题
    qdarktheme.setup_theme("dark") 
    # "auto" 会跟随你的系统设置。
    # 你也可以强制指定为暗色："dark" 或亮色现代风："light"

    window = ModernAnnotationApp(FONT_PATH)
    window.show()
    sys.exit(app.exec_())