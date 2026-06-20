import os
import sys
import json
import random
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QListWidget, QLabel, QScrollArea, QSplitter, QPushButton)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# 配置读取路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "clustered_results.json")

# ==========================================
# 📐 1. 基础几何与离散化计算
# ==========================================
def get_bezier_point(pts, t):
    mt = 1 - t
    return (mt**3)*pts[0] + 3*(mt**2)*t*pts[1] + 3*mt*(t**2)*pts[2] + (t**3)*pts[3]

def normalize_and_sample(mother_bezier, N=50):
    pts = np.array(mother_bezier)
    p0 = pts[0]
    pts_t = pts - p0
    vec = pts_t[3]
    L = np.linalg.norm(vec)
    if L < 1e-5: return None
    
    cos_theta, sin_theta = vec[0]/L, vec[1]/L
    R = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
    pts_norm = (pts_t @ R.T) / L
    
    ts_dense = np.linspace(0, 1, 200)[:, None]
    curve_dense = get_bezier_point(pts_norm, ts_dense)
    diffs = np.diff(curve_dense, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dist = np.insert(np.cumsum(dists), 0, 0)
    
    if cum_dist[-1] < 1e-5: return None
    
    target_dists = np.linspace(0, cum_dist[-1], N)
    x_eq = np.interp(target_dists, cum_dist, curve_dense[:, 0])
    y_eq = np.interp(target_dists, cum_dist, curve_dense[:, 1])
    return np.column_stack([x_eq, y_eq])

def get_4_isomorphisms(S):
    S0 = S.copy()
    S1 = np.zeros_like(S); S1[:, 0] = 1.0 - S[::-1, 0]; S1[:, 1] = -S[::-1, 1] 
    S2 = np.zeros_like(S); S2[:, 0] = S[:, 0]; S2[:, 1] = -S[:, 1]             
    S3 = np.zeros_like(S); S3[:, 0] = 1.0 - S[::-1, 0]; S3[:, 1] = S[::-1, 1]  
    return [S0, S1, S2, S3]

def calculate_l1(S_A, S_B):
    return np.sum(np.abs(S_A.flatten() - S_B.flatten()))

# ==========================================
# 🎨 2. GUI 主程序
# ==========================================
class ClusterVisualizerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDBSCAN Stroke Cluster Viewer")
        self.setGeometry(100, 100, 1200, 800)
        
        self.clusters = {}
        self.current_cid = None
        
        self.init_ui()
        self.load_data()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)
        
        # --- 左侧：分桶列表 ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.lbl_info = QLabel("Loading data...")
        self.lbl_info.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(self.lbl_info)
        
        self.list_widget = QListWidget()
        self.list_widget.itemSelectionChanged.connect(self.on_cluster_selected)
        left_layout.addWidget(self.list_widget)
        
        # --- 右侧：图像渲染区 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        top_bar = QHBoxLayout()
        self.lbl_cluster_title = QLabel("👈 请在左侧选择一个聚类桶")
        self.lbl_cluster_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        
        self.btn_resample = QPushButton("🎲 重新抽样 (Resample)")
        self.btn_resample.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 6px;")
        self.btn_resample.clicked.connect(self.render_current_cluster)
        self.btn_resample.setDisabled(True)
        
        top_bar.addWidget(self.lbl_cluster_title)
        top_bar.addStretch()
        top_bar.addWidget(self.btn_resample)
        right_layout.addLayout(top_bar)
        
        # Matplotlib Canvas
        self.fig = plt.figure(figsize=(8, 12))
        self.canvas = FigureCanvas(self.fig)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.canvas)
        right_layout.addWidget(scroll_area)
        
        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(right_panel)
        self.splitter.setSizes([250, 950])

    def load_data(self):
        if not os.path.exists(DATA_FILE):
            self.lbl_info.setText(f"❌ 找不到文件:\n{DATA_FILE}")
            return
            
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            
        for item in data:
            cid = item["cluster_id"]
            if cid == -1: continue # 剔除噪声
            if cid not in self.clusters: self.clusters[cid] = []
            self.clusters[cid].append(item)
            
        self.lbl_info.setText(f"✅ 找到 {len(self.clusters)} 个有效分桶")
        
        # 填充列表
        for cid in sorted(self.clusters.keys()):
            count = len(self.clusters[cid])
            self.list_widget.addItem(f"Cluster {cid}  ({count} strokes)")

    def on_cluster_selected(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items: return
        
        # 提取选中的 cid
        text = selected_items[0].text()
        self.current_cid = int(text.split()[1])
        
        self.btn_resample.setDisabled(False)
        self.render_current_cluster()

    def render_current_cluster(self):
        if self.current_cid is None: return
        
        strokes_info = self.clusters[self.current_cid]
        self.lbl_cluster_title.setText(f"🔍 Cluster ID: {self.current_cid} | 样本总数: {len(strokes_info)}")
        
        # 随机抽取最多 6 个笔画进行展示
        display_strokes = random.sample(strokes_info, min(6, len(strokes_info)))
        
        # 1. 全部采样
        S_list = []
        for s in display_strokes:
            S = normalize_and_sample(s["mother_bezier"])
            if S is not None: S_list.append(S)
            
        if not S_list:
            self.fig.clf()
            self.canvas.draw()
            return
            
        # 2. 找出“桶内标准方向 (Type 0)”
        S_ref = S_list[0]
        variant_assignments = []
        for S in S_list:
            variants = get_4_isomorphisms(S)
            dists = [calculate_l1(S_ref, v) for v in variants]
            variant_assignments.append(np.argmin(dists))
            
        mode_variant = max(set(variant_assignments), key=variant_assignments.count)
        
        # 3. 开始绘图
        self.fig.clf()
        n_samples = len(S_list)
        axs = self.fig.subplots(n_samples, 2)
        
        # 兼容只有 1 行时的 numpy 维度退化
        if n_samples == 1: axs = np.array([axs])
        
        type_texts = {
            0: "0: Original", 
            1: "1: Reversed", 
            2: "2: X-Flipped", 
            3: "3: Rev + Flip"
        }
        
        for i, S in enumerate(S_list):
            # 异或计算真实的样式编号
            true_variant_id = variant_assignments[i] ^ mode_variant
            
            # --- 图1: 归一化后的原图 ---
            ax_orig = axs[i, 0]
            ax_orig.plot(S[:, 0], S[:, 1], 'b-', linewidth=3)
            ax_orig.plot(S[0, 0], S[0, 1], 'ro', markersize=8) # 起点红点
            ax_orig.set_title(f"Orig (Type {true_variant_id}: {type_texts[true_variant_id].split(':')[1]})", fontsize=10)
            ax_orig.axis('equal')
            ax_orig.grid(True, linestyle='--', alpha=0.5)
            ax_orig.set_xticks([]); ax_orig.set_yticks([])
            
            # --- 图2: 转换为标准态 (Type 0) ---
            S_standard = get_4_isomorphisms(S)[true_variant_id]
            ax_std = axs[i, 1]
            ax_std.plot(S_standard[:, 0], S_standard[:, 1], 'g-', linewidth=3)
            ax_std.plot(S_standard[0, 0], S_standard[0, 1], 'ro', markersize=8) 
            ax_std.set_title("Standardized (Type 0)", fontsize=10)
            ax_std.axis('equal')
            ax_std.grid(True, linestyle='--', alpha=0.5)
            ax_std.set_xticks([]); ax_std.set_yticks([])
            
        self.fig.tight_layout()
        # 根据动态行数调整画布高度，防止被挤压重叠
        self.fig.set_size_inches(8, 2.5 * n_samples)
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ClusterVisualizerGUI()
    window.show()
    sys.exit(app.exec_())