import torch
import os
import sys
import random
import copy
import io
import json
import traceback
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import qdarktheme

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

# 🌟 导入底层依赖
from data_manager import DatasetManager
from geometry_vision import (
    extract_all_real_chars, render_unicode_glyph, collapse_degree2_nodes,
    prune_spurs, stitch_paths, cubic_bezier_np, fit_bezier_basic_with_error,
    split_pixel_path_adaptively, regress_width_dt_fast
)

# 🌟 导入第二阶段 Topo Editor 模块
from topo_editor_workspace import TopoAnnotationWorkspace

# ==========================================
# ⚙️ 全局配置与 AI 探测
# ==========================================
CANVAS_SIZE = 400
MAX_BEZIER_ERROR = 4.5
MAX_SPUR_LENGTH = 20.0
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR = os.path.join(SCRIPT_DIR, "../dataset/alien_tensors_storage")
os.makedirs(FONTS_DIR, exist_ok=True)

# 动作日志与拓扑输出目录
ACTION_LOG_DIR = os.path.join(SCRIPT_DIR, "action_logs")
os.makedirs(ACTION_LOG_DIR, exist_ok=True)
TOPO_OUT_DIR = os.path.join(SCRIPT_DIR, "annotations_topo")
os.makedirs(TOPO_OUT_DIR, exist_ok=True)

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings = logging.getLogger("warnings")
warnings.setLevel(logging.ERROR)
mpl.rcParams['axes.unicode_minus'] = False 

print("\n" + "="*40)
print("🔍 [诊断系统] 正在检查 AI 引擎状态...")
_model_path = os.path.join(SCRIPT_DIR, "ml_engine", "graph_editor_best.pth")
HAS_AI_MODULE = False
try:
    from ml_engine.ai_auto_initializer import UICompatibleAIExecutor
    HAS_AI_MODULE = True
    print("✅ AI 代码模块导入成功！")
except Exception as e:
    print(f"❌ AI 代码模块导入彻底失败！真实报错原因如下：\n   >>> {type(e).__name__}: {e}")
print("="*40 + "\n")


# ==========================================
# 🎨 右侧：核心标注工作区 (集成双阶段机制)
# ==========================================
class AnnotationWorkspace(QWidget):
    def __init__(self, font_path):
        super().__init__()
        self.font_path = font_path
        self.font_filename = os.path.basename(font_path).split('.')[0]
        
        self.db = DatasetManager(SCRIPT_DIR, self.font_filename)
        self.font_charset = extract_all_real_chars(self.font_path, CANVAS_SIZE)
        
        self.history_stack = [] 
        self.selected_edge_ids = [] 
        self.last_click_coord = None 
        self.char = None
        self.thumbnail_cache = {}
        self.bezier_cache = {}
        
        try: self.cmap = plt.colormaps['tab20']
        except AttributeError: self.cmap = plt.get_cmap('tab20')
        
        self.init_mode = "raw" 
        self.pure_raw_edges = []
        self.ai_executor = None
        self.has_ai = HAS_AI_MODULE and os.path.exists(_model_path)
        
        if self.has_ai:
            try:
                self.ai_executor = UICompatibleAIExecutor(model_filename="graph_editor_best.pth")
            except Exception as e:
                self.has_ai = False
                print(f"❌ AI 引擎实例化失败: {str(e)}")
        
        self.init_ui()
        self.action_next_char()
        self.update_stats_display()

    # --- 界面构建 ---
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 🌟 修复：顶部导航栏新增「Phase 1 Done」选项卡
        nav_bar = QHBoxLayout()
        self.btn_anno = QPushButton("✍️ Annotation")
        self.btn_anno.setCheckable(True); self.btn_anno.setChecked(True)
        self.btn_anno.clicked.connect(lambda: self.switch_tab(0))
        
        self.btn_p1_done = QPushButton("⏳ Phase 1 Done")
        self.btn_p1_done.setCheckable(True)
        self.btn_p1_done.clicked.connect(lambda: self.switch_tab(1))
        
        self.btn_comp = QPushButton("✅ Fully Completed")
        self.btn_comp.setCheckable(True)
        self.btn_comp.clicked.connect(lambda: self.switch_tab(2))
        
        self.btn_ban = QPushButton("🚫 Banned")
        self.btn_ban.setCheckable(True)
        self.btn_ban.clicked.connect(lambda: self.switch_tab(3))
        
        nav_bar.addWidget(self.btn_anno)
        nav_bar.addWidget(self.btn_p1_done)
        nav_bar.addWidget(self.btn_comp)
        nav_bar.addWidget(self.btn_ban)
        nav_bar.addStretch()

        self.stats_label = QLabel("📊 Loading Stats...")
        self.stats_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #90CAF9; padding-right: 15px;")
        nav_bar.addWidget(self.stats_label)
        layout.addLayout(nav_bar)

        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)
        
        # 内置 Inner Stack 处理 Phase 1 & 2
        self.page_annotation_container = QWidget()
        anno_layout = QVBoxLayout(self.page_annotation_container)
        anno_layout.setContentsMargins(0, 0, 0, 0)
        self.inner_stack = QStackedWidget()
        anno_layout.addWidget(self.inner_stack)
        
        self.init_page_phase1()
        self.init_page_gallery("p1_done")
        self.init_page_gallery("completed")
        self.init_page_gallery("banned")
        
        self.stacked_widget.addWidget(self.page_annotation_container)
        self.stacked_widget.addWidget(self.page_p1_done)
        self.stacked_widget.addWidget(self.page_completed)
        self.stacked_widget.addWidget(self.page_banned)
        
    def init_page_phase1(self):
        self.page_phase1 = QWidget()
        layout = QHBoxLayout(self.page_phase1)

        control_panel = QVBoxLayout()
        control_panel.setSpacing(10)
        
        self.title_label = QLabel("Loading...")
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        control_panel.addWidget(self.title_label)
        
        self.lbl_p1_status = QLabel("Phase 1 Data: None")
        self.lbl_p1_status.setStyleSheet("font-size: 13px; color: #E91E63; font-weight: bold;")
        self.btn_load_p1 = QPushButton("📥 Load Saved Phase 1")
        self.btn_load_p1.clicked.connect(self.action_load_saved_p1)
        self.btn_load_p1.setStyleSheet("padding: 8px; background-color: #00BCD4; color: white; font-weight: bold;")
        self.btn_load_p1.hide()
        
        control_panel.addWidget(self.lbl_p1_status)
        control_panel.addWidget(self.btn_load_p1)
        
        self.stroke_info_label = QLabel("Beziers: 0")
        self.stroke_info_label.setStyleSheet("font-size: 15px; color: #E91E63; font-weight: bold;")
        control_panel.addWidget(self.stroke_info_label)
        
        control_panel.addWidget(QFrame(frameShape=QFrame.HLine))
        
        ai_status_layout = QHBoxLayout()
        self.lbl_ai_status = QLabel("🟢 AI Active" if self.has_ai else "🔴 AI Offline")
        self.lbl_ai_status.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {'#4CAF50' if self.has_ai else '#F44336'};")
        self.btn_toggle_init = QPushButton("🧠 Use AI Init")
        self.btn_toggle_init.clicked.connect(self.action_toggle_init)
        self.btn_toggle_init.setStyleSheet("padding: 10px; background-color: #673AB7; color: white; font-weight: bold; border-radius: 4px;")
        ai_status_layout.addWidget(self.lbl_ai_status)
        ai_status_layout.addWidget(self.btn_toggle_init)
        control_panel.addLayout(ai_status_layout)
        
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

        # 🌟 新增：手动保存第一阶段按钮
        btn_save_p1 = QPushButton("💾 Save Phase 1 Progress")
        btn_save_p1.clicked.connect(self.action_save_phase1_only)
        btn_save_p1.setStyleSheet("padding: 12px; background-color: #FF9800; color: white; font-weight: bold;")
        control_panel.addWidget(btn_save_p1)

        btn_go_phase2 = QPushButton("➡️ Proceed to Topo Phase")
        btn_go_phase2.clicked.connect(self.action_proceed_to_phase2)
        btn_go_phase2.setStyleSheet("padding: 14px; background-color: #E91E63; color: white; font-weight: bold; font-size: 14px;")
        control_panel.addWidget(btn_go_phase2)

        right_panel = QVBoxLayout()
        self.palette_container = QWidget()
        self.palette_layout = QHBoxLayout(self.palette_container)
        self.palette_layout.setContentsMargins(5, 5, 5, 5); self.palette_layout.setSpacing(8)
        self.palette_layout.setAlignment(Qt.AlignLeft)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True); scroll_area.setWidget(self.palette_container)
        scroll_area.setMaximumHeight(60) 
        
        self.fig, (self.ax_ref, self.ax_main, self.ax_prev, self.ax_final) = plt.subplots(1, 4, figsize=(18, 5))
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
        
        self.inner_stack.addWidget(self.page_phase1)

    def init_page_gallery(self, mode):
        page = QWidget()
        layout = QVBoxLayout(page)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        grid = QGridLayout(container)
        scroll.setWidget(container)
        layout.addWidget(scroll)
        
        # 🌟 修复：新增 Phase 1 Done 绑定
        if mode == "completed":
            self.page_completed = page; self.completed_grid = grid
        elif mode == "banned":
            self.page_banned = page; self.banned_grid = grid
        elif mode == "p1_done":
            self.page_p1_done = page; self.p1_done_grid = grid

    # --- 数据跟踪与更新 ---
    def record_step(self, action_name):
        if hasattr(self, 'action_log'):
            self.action_log.append({"action": action_name, "edges": copy.deepcopy(self.edges)})
            
    def update_stats_display(self):
        total = len(self.font_charset)
        banned = len(self.db.meta_data.get("banned", []))
        
        # 读取聚合文件获取已完工数量
        topo_file = os.path.join(TOPO_OUT_DIR, f"{self.font_filename}_topo.json")
        completed = 0
        if os.path.exists(topo_file):
            try:
                with open(topo_file, 'r', encoding='utf-8') as f: completed = len(json.load(f))
            except: pass
            
        p1_only = len([k for k in self.db.meta_data.get("raw_edges", {}).keys() if k not in self.db.annotated_outlines])
        remaining = max(0, total - completed - banned)
        self.stats_label.setText(f"📊 Remaining: {remaining}   |   ⏳ P1 Done: {p1_only}   |   ✅ Full Topo: {completed}   |   🚫 Banned: {banned}")
        
    def save_state(self):
        self.history_stack.append(copy.deepcopy(self.edges))
        if len(self.history_stack) > 30: self.history_stack.pop(0)

    def action_load_saved_p1(self):
        hex_key = f"U+{ord(self.char):04X}"
        if hex_key in self.db.meta_data.get("raw_edges", {}):
            raw = self.db.meta_data["raw_edges"][hex_key]
            self.edges = [{"id": r["id"], "path": np.array(r["path"])} for r in raw]
            
            # 使用 _actions.json 读取
            log_file = os.path.join(ACTION_LOG_DIR, f"{self.font_filename}_actions.json")
            history_logs = []
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        all_logs = json.load(f)
                        history_logs = all_logs.get(hex_key, [])
                except: pass
                
            if history_logs:
                # 🌟 核心变化：纯粹继承！不调用 record_step 注入多余的动作
                self.action_log = history_logs 
            else:
                self.action_log = [{"action": "Load Saved Phase 1", "edges": copy.deepcopy(self.edges)}]
                
            self.save_state()
            self.update_canvas(); self.update_palette()

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
        global_id = 0
        for u, v, k, d in G.edges(keys=True, data=True):
            sub_paths = split_pixel_path_adaptively(d['path'], MAX_BEZIER_ERROR)
            for sp in sub_paths:
                self.edges.append({'id': global_id, 'path': sp.copy()}) 
                global_id += 1
                
        self.pure_raw_edges = copy.deepcopy(self.edges)
        self.init_mode = "raw"
        if hasattr(self, 'btn_toggle_init'):
            self.btn_toggle_init.setText("🧠 Use AI Init")

        self.bezier_cache.clear() 
        self.selected_edge_ids.clear()
        self.history_stack.clear()
        self.last_click_coord = None
        self.save_state()
        self.update_canvas()
        self.update_palette()
        self.action_log = [{"action": "Init (Raw)", "edges": copy.deepcopy(self.edges)}]
    
    def action_toggle_init(self):
        if not self.has_ai or self.ai_executor is None:
            QMessageBox.warning(self, "AI Offline", "AI 模型未就绪")
            return
        if len(self.pure_raw_edges) == 0 and len(self.edges) > 0:
            self.pure_raw_edges = copy.deepcopy(self.edges)

        backup_edges = copy.deepcopy(self.edges)
        try:
            if self.init_mode == "raw":
                ai_output = self.ai_executor.generate_ai_init_graph(copy.deepcopy(self.pure_raw_edges))
                if not ai_output: raise ValueError("AI 返回空路径集。")
                self.edges = ai_output
                self.init_mode = "ai"
                self.btn_toggle_init.setText("🔄 Revert to Raw Init")
                self.record_step("AI Process Applied")
            else:
                self.edges = copy.deepcopy(self.pure_raw_edges)
                self.init_mode = "raw"
                self.btn_toggle_init.setText("🧠 Use AI Init")
                self.record_step("Revert to Raw")
                
            self.selected_edge_ids.clear(); self.bezier_cache.clear(); self.history_stack.clear()
            self.save_state(); self.ax_main.clear(); self.ax_prev.clear()
            self.update_canvas(); self.update_palette()
        except Exception as e:
            self.edges = backup_edges
            self.init_mode = "raw"
            self.btn_toggle_init.setText("🧠 Use AI Init")
            self.update_canvas()

    def action_next_char(self):
        max_attempts = 500
        valid_char_found = False
        for _ in range(max_attempts):
            candidate = random.choice(self.font_charset)
            hex_key = f"U+{ord(candidate):04X}"
            
            # 在 action_next_char 函数的循环体内部，找到相关的检测逻辑，替换为：
            topo_file = os.path.join(TOPO_OUT_DIR, f"{self.font_filename}_topo.json")
            has_completed = False
            if os.path.exists(topo_file):
                try:
                    with open(topo_file, 'r', encoding='utf-8') as f: has_completed = hex_key in json.load(f)
                except: pass
                
            if hex_key in self.db.meta_data.get("banned", []) or has_completed: 
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
            QMessageBox.information(self, "Finished", "All valid characters have been fully Topo-Annotated or banned!")
            return

        unicode_hex = f"U+{ord(self.char):04X}"
        self.title_label.setText(f"Phase 1: Target '{self.char}' ({unicode_hex})")
        self.dt_map = distance_transform_edt(self.binary)
        
        if unicode_hex in self.db.meta_data.get("raw_edges", {}):
            self.lbl_p1_status.setText("Phase 1 Data: ✅ Exists in Record")
            self.btn_load_p1.show()
        else:
            self.lbl_p1_status.setText("Phase 1 Data: ❌ None")
            self.btn_load_p1.hide()
            
        self.load_char_topology()
    
    def action_ban_char(self):
        if self.char:
            hex_key = f"U+{ord(self.char):04X}"
            
            # 1. 存入 banned 列表
            if hex_key not in self.db.meta_data.setdefault("banned", []): 
                self.db.meta_data["banned"].append(hex_key)
                
            # 2. 清理相关的所有阶段残留数据，保证数据的纯净
            if hex_key in self.db.annotated_outlines: 
                del self.db.annotated_outlines[hex_key]
            if hex_key in self.db.meta_data.get("raw_edges", {}):
                del self.db.meta_data["raw_edges"][hex_key]
                
            topo_file = os.path.join(TOPO_OUT_DIR, f"{self.font_filename}_{hex_key}.json")
            if os.path.exists(topo_file): 
                os.remove(topo_file)
                
            # 3. 保存并刷新界面
            self.db.save_data()
            self.update_stats_display()
            
            main_window = self.window()
            if hasattr(main_window, 'load_font_list'): 
                main_window.load_font_list()
                
            self.action_next_char()
    
    def action_save_phase1_only(self):
        if not self.edges: 
            QMessageBox.warning(self, "Empty", "No annotations to save.")
            return
            
        hex_key = f"U+{ord(self.char):04X}"
        data_changed = False  # 标志位：记录本次是否真的发生了物理写入
        
        # ==========================================
        # 🌟 拦截 1：比对基础几何数据 raw_edges
        # ==========================================
        raw_edges_serializable = [
            {
                "id": e['id'], 
                "path": e['path'].tolist() if isinstance(e['path'], np.ndarray) else e['path']
            } 
            for e in self.edges
        ]
        
        current_saved_raw = self.db.meta_data.get("raw_edges", {}).get(hex_key)
        if current_saved_raw != raw_edges_serializable:
            self.db.meta_data.setdefault("raw_edges", {})[hex_key] = raw_edges_serializable
            self.db.save_data()
            data_changed = True
        
        # ==========================================
        # 🌟 拦截 2：比对操作历史 actions_log
        # ==========================================
        if hasattr(self, 'action_log'):
            log_file = os.path.join(ACTION_LOG_DIR, f"{self.font_filename}_actions.json")
            all_logs = {}
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f: all_logs = json.load(f)
                except: pass
            
            serialized_log = []
            for step in self.action_log:
                step_edges = [
                    {
                        "id": e['id'], 
                        "path": e['path'].tolist() if isinstance(e['path'], np.ndarray) else e['path']
                    } 
                    for e in step["edges"]
                ]
                serialized_log.append({"action": step["action"], "edges": step_edges})
            
            # 只有内容不一致时，才执行写硬盘操作
            if all_logs.get(hex_key) != serialized_log:
                all_logs[hex_key] = serialized_log
                with open(log_file, 'w', encoding='utf-8') as f: 
                    json.dump(all_logs, f, ensure_ascii=False, indent=2)
                data_changed = True
                
        # ==========================================
        # UI 状态反馈
        # ==========================================
        self.lbl_p1_status.setText("Phase 1 Data: ✅ Manually Saved")
        self.btn_load_p1.show()
        self.update_stats_display()
        
        if data_changed:
            print(f"💾 {hex_key} 第一阶段进度已手动保存并落盘！")
            QMessageBox.information(self, "Saved", "Phase 1 progress successfully saved!\nYou can safely close the app or continue editing.")
        else:
            print(f"🔒 数据空转拦截：当前没有任何修改，已安全跳过手动保存的硬盘覆写！")
            QMessageBox.information(self, "No Changes", "No modifications detected.\nSave skipped to prevent empty overwrites.")


    def action_proceed_to_phase2(self):
        if not self.edges: return
        
        reply = QMessageBox.question(self, 'Auto-Save & Proceed', 
                                     "System will automatically save your Phase 1 records.\nDo you want to proceed to the Topo Phase?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.No:
            return
            
        hex_key = f"U+{ord(self.char):04X}"
        
        # ==========================================
        # 🌟 拦截 1：比对基础几何数据 raw_edges
        # ==========================================
        # 安全转换：防止 ndarray 报错
        raw_edges_serializable = [
            {
                "id": e['id'], 
                "path": e['path'].tolist() if isinstance(e['path'], np.ndarray) else e['path']
            } 
            for e in self.edges
        ]
        
        # 读取数据库中已存在的老数据
        current_saved_raw = self.db.meta_data.get("raw_edges", {}).get(hex_key)
        
        # 只有新老数据发生实质性差异时，才触发大字典的落盘保存！
        if current_saved_raw != raw_edges_serializable:
            self.db.meta_data.setdefault("raw_edges", {})[hex_key] = raw_edges_serializable
            self.db.save_data()
        
        # ==========================================
        # 🌟 拦截 2：比对操作历史 actions_log
        # ==========================================
        if hasattr(self, 'action_log'):
            log_file = os.path.join(ACTION_LOG_DIR, f"{self.font_filename}_actions.json")
            all_logs = {}
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f: all_logs = json.load(f)
                except: pass
            
            # 将内存中的历史序列化成标准格式
            serialized_log = []
            for step in self.action_log:
                step_edges = [
                    {
                        "id": e['id'], 
                        "path": e['path'].tolist() if isinstance(e['path'], np.ndarray) else e['path']
                    } 
                    for e in step["edges"]
                ]
                serialized_log.append({"action": step["action"], "edges": step_edges})
            
            # 只有内存里的操作历史和硬盘里的 JSON 不一致时，才触发 open('w') 覆写！
            if all_logs.get(hex_key) != serialized_log:
                all_logs[hex_key] = serialized_log
                # 加入 indent=2 保证原有排版格式不变
                with open(log_file, 'w', encoding='utf-8') as f: 
                    json.dump(all_logs, f, ensure_ascii=False, indent=2)
                print(f"💾 数据已发生修改，成功保存 Phase 1 记录。")
            else:
                print(f"🔒 数据空转拦截：当前没有任何修改，已安全跳过硬盘覆写！")
            
        # ==========================================
        # 交付阶段 2
        # ==========================================
        phase2_edges = []
        for edge in self.edges:
            eid = edge['id']
            if eid not in self.bezier_cache:
                p_opt, _ = fit_bezier_basic_with_error(edge['path'])
                self.bezier_cache[eid] = (p_opt, None)
            p_opt, _ = self.bezier_cache[eid]
            phase2_edges.append({'id': eid, 'path': p_opt.tolist()})

        self.topo_widget = TopoAnnotationWorkspace(self, hex_key, self.char, self.binary, self.dt_map, phase2_edges)
        if self.inner_stack.count() > 1:
            w = self.inner_stack.widget(1)
            self.inner_stack.removeWidget(w)
            w.deleteLater()
        self.inner_stack.addWidget(self.topo_widget)
        self.inner_stack.setCurrentIndex(1)

    def save_phase2_topo_data(self, hex_key, char_bundle):
        # 🌟 核心变化：全量聚合到一个大 JSON 文件中
        topo_file = os.path.join(TOPO_OUT_DIR, f"{self.font_filename}_topo.json")
        all_topo_data = {}
        
        if os.path.exists(topo_file):
            try:
                with open(topo_file, 'r', encoding='utf-8') as f: 
                    all_topo_data = json.load(f)
            except: pass
            
        # 以 U+XXXX 作为 Key 注入字典
        all_topo_data[hex_key] = char_bundle
        
        with open(topo_file, 'w', encoding='utf-8') as f:
            json.dump(all_topo_data, f, ensure_ascii=False, indent=2)
            
        # 同步更新局部快速索引数据库
        self.db.annotated_outlines[hex_key] = char_bundle["strokes"]
        print(f"✅ Topo & Geometry Annotation for {hex_key} aggregated successfully!")
        
        self.update_stats_display()
        main_window = self.window()
        if hasattr(main_window, 'load_font_list'): 
            main_window.load_font_list()
            
        self.inner_stack.setCurrentIndex(0)
        self.action_next_char()

    def update_canvas(self):
        self.ax_ref.clear(); self.ax_ref.imshow(self.binary, cmap='gray'); self.ax_ref.set_title("1. Original", color='black'); self.ax_ref.axis('off')
        
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

        self.ax_prev.clear(); self.ax_prev.imshow(self.binary, cmap='gray', alpha=0.05); self.ax_prev.set_title("3. Bezier", color='black'); self.ax_prev.axis('off')
        
        self.ax_final.clear()
        self.ax_final.imshow(np.ones_like(self.binary), cmap='gray', vmin=0, vmax=1) 
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
            
            mean_w = max(np.mean(w_opt), 1.0)
            ts = np.linspace(0, 1, 50)[:, None]
            curve = cubic_bezier_np(p_opt, ts)
            final_lw = mean_w * 2 * (1.5 if is_sel else 1.0)
            final_alpha = 0.9 if is_sel else 0.4
            self.ax_prev.plot(curve[:, 0], curve[:, 1], color=color, linewidth=final_lw, solid_capstyle='round', alpha=final_alpha)
            self.ax_prev.plot(curve[:, 0], curve[:, 1], color='black', linewidth=1.5, alpha=final_alpha+0.1)

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

    def action_merge(self):
        if len(self.selected_edge_ids) == 2:
            self.save_state()
            id_a, id_b = self.selected_edge_ids
            paths_to_stitch = [e['path'] for e in self.edges if e['id'] in (id_a, id_b)]
            new_unified_path = stitch_paths(paths_to_stitch)
            
            if len(new_unified_path) > 2:
                diffs = np.diff(new_unified_path, axis=0)
                dists = np.linalg.norm(diffs, axis=1)
                total_len = np.sum(dists)
                chord_len = np.linalg.norm(new_unified_path[-1] - new_unified_path[0])
                
                if chord_len > 1.0 and (total_len / chord_len) < 1.05:
                    num_pts = len(new_unified_path)
                    t_vals = np.linspace(0, 1, num_pts).reshape(-1, 1)
                    new_unified_path = new_unified_path[0] * (1 - t_vals) + new_unified_path[-1] * t_vals
                else:
                    cum_dist = np.insert(np.cumsum(dists), 0, 0)
                    t_uniform = np.linspace(0, total_len, max(20, int(total_len)))
                    new_x = np.interp(t_uniform, cum_dist, new_unified_path[:, 0])
                    new_y = np.interp(t_uniform, cum_dist, new_unified_path[:, 1])
                    new_unified_path = np.column_stack([new_x, new_y])

            _, error = fit_bezier_basic_with_error(new_unified_path)
            
            sub_paths = []
            if error > MAX_BEZIER_ERROR:
                sub_paths = split_pixel_path_adaptively(new_unified_path, MAX_BEZIER_ERROR)
                QMessageBox.warning(self, "Fitting Alert", f"Merge resulted in a high deviation (Error: {error:.2f}).\nSplit into {len(sub_paths)} parts.")
            else: sub_paths = [new_unified_path]
            
            self.edges = [e for e in self.edges if e['id'] not in (id_a, id_b)]
            new_id = max([e['id'] for e in self.edges] + [0]) + 1
            for sp in sub_paths:
                self.edges.append({'id': new_id, 'path': sp})
                new_id += 1
                
            self.selected_edge_ids.clear(); self.bezier_cache.clear()
            self.update_canvas(); self.update_palette()
            self.record_step("Merge (M)")

    def action_delete(self):
        if self.selected_edge_ids:
            self.save_state()
            self.edges = [e for e in self.edges if e['id'] not in self.selected_edge_ids]
            self.selected_edge_ids.clear(); self.bezier_cache.clear(); self.update_canvas(); self.update_palette()
            self.record_step("Delete (D)")

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
            self.record_step("Prune (C)")

    def action_breakpoint(self):
        if not self.last_click_coord: return
        self.save_state()
        click_pt = np.array(self.last_click_coord)
        target_id = self.selected_edge_ids[0] if len(self.selected_edge_ids) == 1 else None
        min_dist, best_i, best_split = float('inf'), -1, -1
        
        for i, edge in enumerate(self.edges):
            if target_id is not None and edge['id'] != target_id: continue
            dists = np.linalg.norm(edge['path'] - click_pt, axis=1)
            if len(dists) == 0: continue
            idx = np.argmin(dists)
            if dists[idx] < min_dist:
                min_dist = dists[idx]; best_i = i; best_split = idx

        if best_i != -1 and min_dist < 15.0:
            edge_path = self.edges[best_i]['path']
            path_len = len(edge_path)
            if 0 < best_split < path_len - 1:
                edge = self.edges[best_i]
                path1, path2 = edge['path'][:best_split+1], edge['path'][best_split:]
                new_id = max([e['id'] for e in self.edges] + [0]) + 1
                self.edges.pop(best_i)
                self.edges.append({'id': new_id, 'path': path1})
                self.edges.append({'id': new_id+1, 'path': path2})
                self.last_click_coord = None
                self.selected_edge_ids.clear(); self.bezier_cache.clear()
                self.update_canvas(); self.update_palette()
                self.record_step("Split (B)")

    def action_add_dot(self):
        if not self.last_click_coord: return
        cx, cy = self.last_click_coord; x, y = int(cx), int(cy)
        if not (0 <= x < CANVAS_SIZE and 0 <= y < CANVAS_SIZE): return
        self.save_state()
        tiny_path = np.array([[cx, cy-1], [cx, cy], [cx, cy+1]])
        new_id = max([e['id'] for e in self.edges] + [-1]) + 1
        self.edges.append({'id': new_id, 'path': tiny_path})
        self.last_click_coord = None; self.update_canvas(); self.update_palette()
        self.record_step("Add Dot (A)")

    def action_undo(self):
        if len(self.history_stack) > 1:
            self.history_stack.pop() 
            self.edges = copy.deepcopy(self.history_stack[-1])
            self.selected_edge_ids.clear(); self.last_click_coord = None; self.bezier_cache.clear()
            self.update_canvas(); self.update_palette()
            if hasattr(self, 'action_log') and len(self.action_log) > 1:
                self.action_log.pop()

    def action_reset_char(self):
        self.load_char_topology()

    # 🌟 修复：新增 Phase 1 导航逻辑
    def switch_tab(self, index):
        self.btn_anno.setChecked(index == 0)
        self.btn_p1_done.setChecked(index == 1)
        self.btn_comp.setChecked(index == 2)
        self.btn_ban.setChecked(index == 3)
        
        # 从聚合大文件中拉取完工的清单键值
        topo_file = os.path.join(TOPO_OUT_DIR, f"{self.font_filename}_topo.json")
        completed_keys = []
        if os.path.exists(topo_file):
            try:
                with open(topo_file, 'r', encoding='utf-8') as f: completed_keys = list(json.load(f).keys())
            except: pass
        
        if index == 1: 
            raw_keys = list(self.db.meta_data.get("raw_edges", {}).keys())
            p1_only_keys = [k for k in raw_keys if k not in completed_keys]
            self.refresh_gallery(self.p1_done_grid, p1_only_keys, "p1_done")
        elif index == 2: 
            self.refresh_gallery(self.completed_grid, completed_keys, "completed")
        elif index == 3: 
            self.refresh_gallery(self.banned_grid, self.db.meta_data.get("banned", []), "banned")
            
        self.stacked_widget.setCurrentIndex(index)

    def action_delete_character(self, hex_key):
        char = chr(int(hex_key[2:], 16))
        reply = QMessageBox.question(self, 'Confirm Deletion', 
                                     f"Are you sure you want to completely erase ALL annotation history for '{char}' ({hex_key})?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.db.delete_character(hex_key)
            if hex_key in self.db.meta_data.get("raw_edges", {}):
                del self.db.meta_data["raw_edges"][hex_key]
                self.db.save_data()

            # 从聚合的大 JSON 文件中剔除该字形键值
            topo_file = os.path.join(TOPO_OUT_DIR, f"{self.font_filename}_topo.json")
            if os.path.exists(topo_file):
                try:
                    with open(topo_file, 'r', encoding='utf-8') as f: big_data = json.load(f)
                    if hex_key in big_data:
                        del big_data[hex_key]
                        with open(topo_file, 'w', encoding='utf-8') as f:
                            json.dump(big_data, f, ensure_ascii=False, indent=2)
                except: pass
            
            self.thumbnail_cache.pop(f"{hex_key}_completed", None)
            self.thumbnail_cache.pop(f"{hex_key}_p1_done", None)
            
            idx = self.stacked_widget.currentIndex()
            self.switch_tab(idx)
            self.update_stats_display()
            main_window = self.window()
            if hasattr(main_window, 'load_font_list'): main_window.load_font_list()

    def action_unban(self, hex_key):
        if hex_key in self.db.meta_data["banned"]:
            self.db.meta_data["banned"].remove(hex_key)
            self.db.save_data()
            self.update_stats_display()
            self.switch_tab(3)
            main_window = self.window()
            if hasattr(main_window, 'load_font_list'): main_window.load_font_list()
    
    def action_reannotate(self, hex_key):
        self.char = chr(int(hex_key[2:], 16))
        self.binary = render_unicode_glyph(self.font_path, self.char, CANVAS_SIZE)
        self.dt_map = distance_transform_edt(self.binary)
        
        # 🌟 1. 优先探测是否存在第二阶段 (Topo) 聚合大字典数据
        topo_file = os.path.join(TOPO_OUT_DIR, f"{self.font_filename}_topo.json")
        has_phase2 = False
        phase2_data = []
        if os.path.exists(topo_file):
            try:
                with open(topo_file, 'r', encoding='utf-8') as f:
                    big_data = json.load(f)
                    if hex_key in big_data:
                        has_phase2 = True
                        phase2_data = big_data[hex_key].get("strokes", [])
            except: pass

        if has_phase2:
            # ==========================================
            # 🚀 路由 A：空降第二阶段 (恢复贝塞尔与宽度)
            # ==========================================
            self.title_label.setText(f"[RE-EDIT] Phase 2: Target '{self.char}' ({hex_key})")
            
            # 将大 JSON 里的精细数据还原为 Phase 2 需要的格式
            phase2_edges = []
            for e in phase2_data:
                phase2_edges.append({'id': e['bezier_id'], 'path': np.array(e['mother_bezier'])})
                
            self.topo_widget = TopoAnnotationWorkspace(self, hex_key, self.char, self.binary, self.dt_map, phase2_edges)
            
            # 🌟 强行注入保存过的完美宽度缓存，避免乱动时的突然闪烁
            for e in phase2_data:
                self.topo_widget.width_cache[e['bezier_id']] = np.array(e['width_bezier'])
                
            # 切换 UI 堆栈到 Phase 2
            if self.inner_stack.count() > 1:
                w = self.inner_stack.widget(1)
                self.inner_stack.removeWidget(w)
                w.deleteLater()
            self.inner_stack.addWidget(self.topo_widget)
            self.inner_stack.setCurrentIndex(1)
            self.switch_tab(0)
            
        else:
            # ==========================================
            # 🛠️ 路由 B：回到第一阶段 (常规流程)
            # ==========================================
            self.title_label.setText(f"[RE-EDIT] Phase 1: Target '{self.char}' ({hex_key})")
            raw = self.db.meta_data["raw_edges"].get(hex_key, [])
            self.edges = [{"id": r["id"], "path": np.array(r["path"])} for r in raw]
            self.pure_raw_edges = copy.deepcopy(self.edges)
            
            self.init_mode = "raw"
            if hasattr(self, 'btn_toggle_init'):
                self.btn_toggle_init.setText("🧠 Use AI Init")
                
            self.bezier_cache.clear(); self.selected_edge_ids.clear(); self.history_stack.clear()
            self.save_state()
            self.action_log = [{"action": "Re-edit Init", "edges": copy.deepcopy(self.edges)}]
            
            self.lbl_p1_status.setText("Phase 1 Data: ✅ Loaded from Record")
            self.btn_load_p1.show()
            
            self.update_canvas(); self.update_palette()
            self.inner_stack.setCurrentIndex(0)
            self.switch_tab(0)

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
            
            # 🌟 动态更新显示数量信息
            # 找到 refresh_gallery 中 mode == "completed" 的判断分支，替换为：
            if mode == "completed":
                topo_file = os.path.join(TOPO_OUT_DIR, f"{self.font_filename}_topo.json")
                strokes_count = "?"
                if os.path.exists(topo_file):
                    try:
                        with open(topo_file, 'r', encoding='utf-8') as f:
                            strokes_count = len(json.load(f).get(hex_key, {}).get("strokes", []))
                    except: pass
                title_text = f"'{char}' ({hex_key})\nTopo Strokes: {strokes_count}"
            elif mode == "p1_done":
                raw_edges = self.db.meta_data.get("raw_edges", {}).get(hex_key, [])
                title_text = f"'{char}' ({hex_key})\nRaw Strokes: {len(raw_edges)}"
            else:
                title_text = f"'{char}' ({hex_key})"
            
            title = QLabel(title_text)
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("font-weight: bold; border: none; color: black;")

            btn_layout = QHBoxLayout()
            btn_layout.setContentsMargins(0, 0, 0, 0)
            btn_action = QPushButton("Re-annotate" if mode in ["completed", "p1_done"] else "Unban")
            btn_action.setStyleSheet("background-color: #2196F3; color: white; padding: 5px;" if mode in ["completed", "p1_done"] else "background-color: #4CAF50; color: white; padding: 5px;")
            btn_action.clicked.connect(lambda checked, hk=hex_key: self.action_unban(hk) if mode == "banned" else self.action_reannotate(hk))
            btn_layout.addWidget(btn_action)

            if mode in ["completed", "p1_done"]:
                btn_delete = QPushButton("🗑️ Delete")
                btn_delete.setStyleSheet("background-color: #F44336; color: white; padding: 5px;")
                btn_delete.clicked.connect(lambda checked, hk=hex_key: self.action_delete_character(hk))
                btn_layout.addWidget(btn_delete)

            card_layout.addWidget(img_label)
            card_layout.addWidget(title)
            card_layout.addLayout(btn_layout) 
            grid_layout.addWidget(card, idx // cols, idx % cols)

    def generate_thumbnail(self, char, hex_key, mode):
        fig = plt.figure(figsize=(2, 2) if mode == "banned" else (2.5, 7.5), dpi=80)
        fig.patch.set_facecolor('#FFFFFF')
        binary = render_unicode_glyph(self.font_path, char, CANVAS_SIZE)

        if mode == "banned":
            ax = fig.add_subplot(111)
            ax.imshow(binary, cmap='gray'); ax.axis('off')
            
        elif mode == "p1_done":
            # 🌟 专属 Phase 1 Done 渲染
            ax_top = fig.add_subplot(311); ax_top.imshow(binary, cmap='gray'); ax_top.set_title("1. Original", fontsize=10); ax_top.axis('off')
            edges = self.db.meta_data.get("raw_edges", {}).get(hex_key, [])
            
            ax_mid = fig.add_subplot(312); ax_mid.imshow(binary, cmap='gray', alpha=0.15); ax_mid.set_title("2. P1 Topology", fontsize=10); ax_mid.axis('off')
            for e in edges:
                color = self.cmap((e['id'] % 20))
                path = np.array(e['path'])
                if len(path) > 1: ax_mid.plot(path[:, 0], path[:, 1], c=color, lw=3, alpha=0.8)

            ax_bot = fig.add_subplot(313); ax_bot.imshow(np.ones_like(binary), cmap='gray', vmin=0, vmax=1); ax_bot.set_title("3. P1 Bezier", fontsize=10); ax_bot.axis('off')
            dt_map = distance_transform_edt(binary)
            for e in edges:
                path = np.array(e['path'])
                if len(path) < 2: continue
                p_opt, _ = fit_bezier_basic_with_error(path)
                w_opt = regress_width_dt_fast(p_opt, dt_map)
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
                ax_bot.fill(poly[:, 0], poly[:, 1], color='#000000', alpha=0.9, linewidth=0)
                
        elif mode == "completed":
            ax_top = fig.add_subplot(311); ax_top.imshow(binary, cmap='gray'); ax_top.set_title("1. Original", fontsize=10); ax_top.axis('off')
            
            # 🌟 核心修改：从全新的大 JSON 聚合文件中读取该字形的第二阶段完成数据
            topo_file = os.path.join(TOPO_OUT_DIR, f"{self.font_filename}_topo.json")
            edges = []
            if os.path.exists(topo_file):
                try:
                    with open(topo_file, 'r', encoding='utf-8') as f:
                        # 深入结构抓取其中的 strokes 列表
                        edges = json.load(f).get(hex_key, {}).get("strokes", [])
                except: pass
            
            ax_mid = fig.add_subplot(312); ax_mid.imshow(binary, cmap='gray', alpha=0.15); ax_mid.set_title("2. Topo Skeletons", fontsize=10); ax_mid.axis('off')
            for e in edges:
                color = self.cmap((e['bezier_id'] % 20)) # 此时的 id 已经是优化后的 1, 2, 3...
                p_opt = np.array(e['mother_bezier'])
                curve = cubic_bezier_np(p_opt, np.linspace(0, 1, 50)[:, None])
                ax_mid.plot(curve[:, 0], curve[:, 1], c=color, lw=3, alpha=0.8)

            ax_bot = fig.add_subplot(313); ax_bot.imshow(np.ones_like(binary), cmap='gray', vmin=0, vmax=1); ax_bot.set_title("3. TTF Mesh", fontsize=10); ax_bot.axis('off')
            for e in edges:
                p_opt = np.array(e['mother_bezier'])
                w_opt = np.array(e['width_bezier'])
                ts_dense = np.linspace(0, 1, 50)[:, None]
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
                ax_bot.fill(poly[:, 0], poly[:, 1], color='#000000', alpha=0.9, linewidth=0)
                
        plt.tight_layout(pad=0.5)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        qimg = QImage()
        qimg.loadFromData(buf.getvalue())
        return QPixmap.fromImage(qimg)


# ==========================================
# 🏠 左侧：主窗口与文件三段式侧边栏
# ==========================================
class FontFactoryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Vector Router - Multi-Font Factory")
        self.setGeometry(50, 50, 1600, 850)
        
        self.splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.splitter)
        
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        
        sidebar_title = QLabel("📂 Font Library")
        sidebar_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        sidebar_layout.addWidget(sidebar_title)
        
        lbl_progress = QLabel("🔄 In Progress")
        lbl_progress.setStyleSheet("font-size: 14px; font-weight: bold; color: #2196F3; margin-top: 5px;")
        sidebar_layout.addWidget(lbl_progress)
        self.list_progress = QListWidget()
        self.list_progress.itemClicked.connect(self.on_font_selected)
        self.list_progress.setStyleSheet("background-color: #E3F2FD; border: 1px solid #BBDEFB; border-radius: 4px; color: black;")
        sidebar_layout.addWidget(self.list_progress, 2) 
        
        lbl_untouched = QLabel("⏳ Untouched")
        lbl_untouched.setStyleSheet("font-size: 14px; font-weight: bold; color: #FF9800; margin-top: 5px;")
        sidebar_layout.addWidget(lbl_untouched)
        self.list_untouched = QListWidget()
        self.list_untouched.itemClicked.connect(self.on_font_selected)
        self.list_untouched.setStyleSheet("background-color: #FFF3E0; border: 1px solid #FFE0B2; border-radius: 4px; color: black;")
        sidebar_layout.addWidget(self.list_untouched, 3) 
        
        lbl_completed = QLabel("✅ Fully Completed")
        lbl_completed.setStyleSheet("font-size: 14px; font-weight: bold; color: #4CAF50; margin-top: 5px;")
        sidebar_layout.addWidget(lbl_completed)
        self.list_completed = QListWidget()
        self.list_completed.itemClicked.connect(self.on_font_selected)
        self.list_completed.setStyleSheet("background-color: #F1F8E9; border: 1px solid #C8E6C9; border-radius: 4px; color: black;")
        sidebar_layout.addWidget(self.list_completed, 1)
        
        btn_refresh = QPushButton("🔄 Refresh Fonts")
        btn_refresh.clicked.connect(self.load_font_list)
        btn_refresh.setStyleSheet("padding: 8px; background-color: #78909C; color: white; border-radius: 4px; margin-top: 5px; font-weight: bold;")
        sidebar_layout.addWidget(btn_refresh)
        
        self.workspace_container = QStackedWidget()
        placeholder = QLabel("👈 Select a font from the sidebar to begin annotation.")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("font-size: 24px; color: #888;")
        self.workspace_container.addWidget(placeholder)
        
        self.splitter.addWidget(sidebar)
        self.splitter.addWidget(self.workspace_container)
        self.splitter.setSizes([250, 1350]) 
        
        self.load_font_list()

    def load_font_list(self):
        self.list_progress.clear()
        self.list_untouched.clear()
        self.list_completed.clear()
        
        if not os.path.exists(FONTS_DIR): return
        
        meta_dir = os.path.join(SCRIPT_DIR, "metadata")
        os.makedirs(meta_dir, exist_ok=True)

        for file in os.listdir(FONTS_DIR):
            if file.lower().endswith(('.ttf', '.otf')):
                font_filename = os.path.splitext(file)[0]
                meta_file = os.path.join(meta_dir, f"{font_filename}_meta.json")
                
                # 🌟 核心修改：不再去列表里扫零散文件，而是去读每个字体的聚合大 JSON 文件
                topo_file = os.path.join(TOPO_OUT_DIR, f"{font_filename}_topo.json")
                topo_count = 0
                if os.path.exists(topo_file):
                    try:
                        with open(topo_file, 'r', encoding='utf-8') as f:
                            # 字典里的键值数（即 U+XXXX 的数量）就是完工的字形数
                            topo_count = len(json.load(f))
                    except: pass
                    
                banned_count = 0
                phase1_count = 0
                
                if os.path.exists(meta_file):
                    try:
                        with open(meta_file, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                            banned_count = len(meta.get("banned", []))
                            phase1_count = len(meta.get("raw_edges", {}))
                    except: pass
                
                # 只要有过任何操作，就不算 Untouched
                if topo_count == 0 and banned_count == 0 and phase1_count == 0:
                    self.list_untouched.addItem(file)
                else:
                    total_chars = 999999 
                    try:
                        from fontTools.ttLib import TTFont
                        font_path = os.path.join(FONTS_DIR, file)
                        ttfont = TTFont(font_path)
                        cmap = ttfont.getBestCmap()
                        total_chars = len(cmap) if cmap else 0
                        ttfont.close()
                    except Exception as e:
                        print(f"Warning reading cmap for {file}: {e}")
                    
                    # 仅当 Topo 阶段 + 禁用 的总量覆盖了全集，才算完全竣工
                    if topo_count + banned_count >= total_chars and total_chars > 0:
                        self.list_completed.addItem(file)
                    else:
                        self.list_progress.addItem(file)

    def on_font_selected(self, item):
        sender = self.sender()
        if sender != self.list_progress: self.list_progress.clearSelection()
        if sender != self.list_untouched: self.list_untouched.clearSelection()
        if sender != self.list_completed: self.list_completed.clearSelection()

        font_filename = item.text()
        full_font_path = os.path.join(FONTS_DIR, font_filename)
        
        if self.workspace_container.count() > 1:
            old_widget = self.workspace_container.widget(1)
            self.workspace_container.removeWidget(old_widget)
            old_widget.deleteLater()
            plt.close(old_widget.fig)
            
        new_workspace = AnnotationWorkspace(full_font_path)
        self.workspace_container.addWidget(new_workspace)
        self.workspace_container.setCurrentIndex(1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("light", custom_colors={"primary": "#2196F3"})
    window = FontFactoryApp()
    window.show()
    sys.exit(app.exec_())