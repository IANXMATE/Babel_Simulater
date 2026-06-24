import os
import sys
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QScrollArea, 
                             QStackedWidget, QGridLayout, QListWidget, QSplitter, 
                             QFrame, QTextBrowser, QComboBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# ==========================================
# ⚙️ 全局配置与数据源
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, "../fontgpt_dataset.json"))
TOPO_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../AI_VECTOR_ROUTER_With_topo/annotations_topo"))
CLUSTER_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, "../clustered_results.json")) # 🌟 加载密码本

CANVAS_SIZE = 400.0
SHAPE_CODEBOOK = {} # 🌟 全局存储曲线密码本

T_BINS = 32

try: CMAP = plt.colormaps['tab20']
except AttributeError: CMAP = plt.get_cmap('tab20')

def get_hex_color(idx):
    c = CMAP(idx % 20)
    return f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"

def cubic_bezier_np(pts, ts):
    mt = 1 - ts
    return (mt**3)*pts[0] + 3*(mt**2)*ts*pts[1] + 3*mt*(ts**2)*pts[2] + (ts**3)*pts[3]

# ==========================================
# 🎨 绘图引擎 (含 VQ-VAE 曲线还原)
# ==========================================
class MatplotlibCanvas(FigureCanvas):
    def __init__(self, width=5, height=5, dpi=80):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#FFFFFF')
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        super().__init__(self.fig)

def draw_original_topo(ax, char_data):
    ax.clear()
    ax.set_xlim(0, CANVAS_SIZE)
    ax.set_ylim(CANVAS_SIZE, 0)
    ax.axis('off')
    ax.grid(True, linestyle='--', alpha=0.3)
    if not char_data: return

    strokes = char_data.get("strokes", [])
    for stroke in strokes:
        eid = stroke.get("bezier_id", 0)
        pts = np.array(stroke.get("mother_bezier", []))
        if len(pts) != 4: continue
        color = CMAP(eid % 20)
        ts = np.linspace(0, 1, 50)[:, None]
        curve = cubic_bezier_np(pts, ts)
        ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=4, alpha=0.9)
        ax.scatter(pts[0, 0], pts[0, 1], color='green', s=50, zorder=2)
        ax.scatter(pts[3, 0], pts[3, 1], color='red', s=50, zorder=2)
        mid = curve[25]
        ax.text(mid[0], mid[1], f"S{eid}", color=color, fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))


def draw_skeleton(ax, sequence, with_arrows=False, with_order=False, highlight_strokes=None, junction_mark=None):
    ax.clear()
    ax.axis('off')
    
    drawn_strokes = []
    stroke_idx = 0
    scale = CANVAS_SIZE / 32.0 
    all_x, all_y = [], [] 
    
    for item in sequence:
        if item.get("token_type") == "STROKE":
            if "p0" in item:
                p0, p3 = np.array(item["p0"]), np.array(item["p3"])
            else:
                p0 = np.array([(item["p0_cell"][0] + 0.5 + item["p0_offset"][0]) * scale, 
                               (item["p0_cell"][1] + 0.5 + item["p0_offset"][1]) * scale])
                p3 = np.array([(item["p3_cell"][0] + 0.5 + item["p3_offset"][0]) * scale, 
                               (item["p3_cell"][1] + 0.5 + item["p3_offset"][1]) * scale])
            
            drawn_strokes.append((p0, p3))
            is_highlighted = (highlight_strokes is not None and stroke_idx in highlight_strokes)
            is_faded = (highlight_strokes is not None and stroke_idx not in highlight_strokes)
            color = CMAP(stroke_idx % 20)
            alpha = 0.15 if is_faded else 0.9
            lw = 6 if is_highlighted else 3
            
            shape_code = item.get("shape_code", -1)
            var_id = item.get("variant_id", 0) # 🌟 1. 提取变体 ID (等价于你的 morph_emb)
            
            if shape_code in SHAPE_CODEBOOK:
                canon_pts = np.array(SHAPE_CODEBOOK[shape_code]).copy()
                c0, c3 = canon_pts[0], canon_pts[3]
                
                # ==========================================
                # 🌟 2. 核心魔法：局部坐标系翻转映射
                # ==========================================
                v_c = c3 - c0
                L_c = np.linalg.norm(v_c)
                
                if L_c > 1e-5:
                    u = v_c / L_c
                    n = np.array([-u[1], u[0]]) # 法向量，建立“局部 Y 轴”
                    
                    local_x = np.dot(canon_pts - c0, u)
                    local_y = np.dot(canon_pts - c0, n)
                    
                    # 根据 var_id 强行干预局部形状
                    if var_id == 1: # 起终点倒转
                        local_y = -local_y
                        local_x = (L_c - local_x)[::-1]
                        local_y = local_y[::-1]
                    elif var_id == 2: # 🌟 纯镜像：法线方向直接取反，凹凸性瞬间反转！
                        local_y = -local_y 
                    elif var_id == 3: # 中心对称倒转
                        local_x = (L_c - local_x)[::-1]
                        local_y = local_y[::-1]
                    
                    # 重新将形变后的局部坐标拍回密码本的绝对坐标系
                    canon_pts = c0 + np.outer(local_x, u) + np.outer(local_y, n)
                # ==========================================
                
                # ------ 形态翻转完毕，接下来执行正常的刚体贴合 (缩放 + 旋转) ------
                c0, c3 = canon_pts[0], canon_pts[3]
                v_canon, v_pred = c3 - c0, p3 - p0
                len_canon, len_pred = np.linalg.norm(v_canon), np.linalg.norm(v_pred)
                
                if len_canon > 1e-5 and len_pred > 1e-5:
                    s_factor = len_pred / len_canon
                    theta = np.arctan2(v_pred[1], v_pred[0]) - np.arctan2(v_canon[1], v_canon[0])
                    cos_t, sin_t = np.cos(theta), np.sin(theta)
                    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                    mapped_pts = (canon_pts - c0) @ R.T * s_factor + p0
                    
                    ts = np.linspace(0, 1, 50)[:, None]
                    curve = cubic_bezier_np(mapped_pts, ts)
                    
                    all_x.extend(curve[:, 0])
                    all_y.extend(curve[:, 1])
                    
                    ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=lw, alpha=alpha, zorder=1)
                    
                    if with_arrows and not is_faded:
                        v_tangent = mapped_pts[3] - mapped_pts[2]
                        len_tangent = np.linalg.norm(v_tangent)
                        if len_tangent > 1e-5:
                            v_norm = v_tangent / len_tangent
                            xy_text = mapped_pts[3] - 1.0 * v_norm 
                            ax.annotate('', xy=mapped_pts[3], xytext=xy_text, 
                                        arrowprops=dict(arrowstyle='->', color=color, lw=lw, alpha=alpha, mutation_scale=20, zorder=3))
                else:
                    all_x.extend([p0[0], p3[0]])
                    all_y.extend([p0[1], p3[1]])
                    if with_arrows and not is_faded:
                        ax.annotate('', xy=p3, xytext=p0, arrowprops=dict(arrowstyle='->', color=color, lw=lw, alpha=alpha, mutation_scale=20))
                    else:
                        ax.plot([p0[0], p3[0]], [p0[1], p3[1]], color=color, linewidth=lw, alpha=alpha, zorder=1)
            else:
                all_x.extend([p0[0], p3[0]])
                all_y.extend([p0[1], p3[1]])
                if with_arrows and not is_faded:
                    ax.annotate('', xy=p3, xytext=p0, arrowprops=dict(arrowstyle='->', color=color, lw=lw, alpha=alpha, mutation_scale=20))
                else:
                    ax.plot([p0[0], p3[0]], [p0[1], p3[1]], color=color, linewidth=lw, alpha=alpha, zorder=1)
                
            if not is_faded:
                ax.scatter(p0[0], p0[1], color='green', s=40, zorder=2)
                ax.scatter(p3[0], p3[1], color='red', s=40, zorder=2)
            mid_pt = (p0 + p3) / 2
            ax.text(mid_pt[0], mid_pt[1], f"S{stroke_idx}", color=color, fontsize=12, fontweight='bold', alpha=alpha)
            if with_order and not is_faded:
                ax.text(p0[0]-20, p0[1]-20, f"[{stroke_idx+1}]", color='black', fontsize=14, fontweight='bold',
                        bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none', boxstyle='circle,pad=0.2'))
            stroke_idx += 1
            
    if junction_mark:
        u, v, ta, tb = junction_mark
        if 0 <= u < len(drawn_strokes) and 0 <= v < len(drawn_strokes):
            p0_u, p3_u = drawn_strokes[u]
            p0_v, p3_v = drawn_strokes[v]
            pt_u = p0_u * (1 - ta) + p3_u * ta
            pt_v = p0_v * (1 - tb) + p3_v * tb
            center_pt = (pt_u + pt_v) / 2
            
            all_x.append(center_pt[0])
            all_y.append(center_pt[1])
            
            ax.plot(center_pt[0], center_pt[1], marker='X', color='red', markersize=20, markeredgecolor='black', zorder=10)
            ax.text(center_pt[0]+20, center_pt[1]+20, f"({int(center_pt[0])}, {int(center_pt[1])})", 
                    color='red', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    if all_x and all_y:
        cx = (np.min(all_x) + np.max(all_x)) / 2
        cy = (np.min(all_y) + np.max(all_y)) / 2
        size = max(np.max(all_x) - np.min(all_x), np.max(all_y) - np.min(all_y)) / 2 * 1.2
        if size < 10: size = 50 
        ax.set_xlim(cx - size, cx + size)
        ax.set_ylim(cy + size, cy - size)

# ==========================================
# 🏠 UI 层 (已省略未修改的 UI 代码以防冗长，只保留核心解析更新)
# ==========================================
class InspectionWidget(QWidget):
    # ... (init_ui 保持完全一致) ...
    def __init__(self, main_app, hex_key, orig_data, data_group):
        super().__init__()
        self.main_app = main_app
        self.hex_key = hex_key
        self.orig_data = orig_data
        self.data_group = data_group
        self.derivations = list(self.data_group.keys())
        self.current_seq = None
        self.current_junctions = []
        self.init_ui()
        if self.derivations: self.list_derivations.setCurrentRow(0)

    def init_ui(self):
        layout = QHBoxLayout(self)
        left_panel = QVBoxLayout()
        btn_back = QPushButton("🔙 返回画廊")
        btn_back.setStyleSheet("padding: 10px; background-color: #757575; color: white; font-weight: bold; border-radius: 4px;")
        btn_back.clicked.connect(lambda: self.main_app.workspace_stack.setCurrentIndex(0))
        left_panel.addWidget(btn_back)
        lbl = QLabel(f"字符 {self.hex_key} 的派生组:")
        lbl.setStyleSheet("font-weight: bold; margin-top: 10px;")
        left_panel.addWidget(lbl)
        self.list_derivations = QListWidget()
        self.list_derivations.addItems(self.derivations)
        self.list_derivations.itemClicked.connect(self.on_derivation_selected)
        left_panel.addWidget(self.list_derivations)
        layout.addLayout(left_panel, 1)
        
        right_panel = QScrollArea()
        right_panel.setWidgetResizable(True)
        container = QWidget()
        self.vbox = QVBoxLayout(container)
        
        self.vbox.addWidget(self.create_section_title("🎯 Row 1: 【真理层】原始拓扑标注数据 (Absolute GT)"))
        self.row1_layout = QHBoxLayout()
        self.canvas1 = MatplotlibCanvas(width=4, height=4)
        self.text1 = QTextBrowser()
        self.row1_layout.addWidget(self.canvas1, 1)
        self.row1_layout.addWidget(self.text1, 1)
        self.vbox.addLayout(self.row1_layout)
        
        self.vbox.addWidget(self.create_section_title("🚀 Row 2: 【推断层】1D Token 序列还原测试"))
        self.row2_layout = QHBoxLayout()
        self.canvas2 = MatplotlibCanvas(width=4, height=4)
        self.text2 = QTextBrowser()
        self.row2_layout.addWidget(self.canvas2, 1)
        self.row2_layout.addWidget(self.text2, 1)
        self.vbox.addLayout(self.row2_layout)
        
        self.vbox.addWidget(self.create_section_title("🔦 Row 3: Token 交点探照灯 (Junction Spotlight)"))
        self.combo_junctions = QComboBox()
        self.combo_junctions.setStyleSheet("padding: 5px; font-size: 14px; background-color: #FFF9C4;")
        self.combo_junctions.currentIndexChanged.connect(self.on_junction_changed)
        self.vbox.addWidget(self.combo_junctions)
        
        self.row3_layout = QHBoxLayout()
        self.canvas3 = MatplotlibCanvas(width=4, height=4)
        self.text3 = QTextBrowser()
        self.row3_layout.addWidget(self.canvas3, 1)
        self.row3_layout.addWidget(self.text3, 1)
        self.vbox.addLayout(self.row3_layout)
        
        right_panel.setWidget(container)
        layout.addWidget(right_panel, 4)
        
        draw_original_topo(self.canvas1.ax, self.orig_data)
        self.canvas1.draw()
        self.text1.setHtml(self.generate_orig_topo_html(self.orig_data))

    def create_section_title(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #E0E0E0; padding: 5px;")
        return lbl

    def generate_orig_topo_html(self, char_data):
        if not char_data: return "<div style='color:red;'>缺少原始拓扑数据。</div>"
        events = char_data.get("topology_events", [])
        if not events: return "<div style='color:#777;'>原始数据中无拓扑交点记录。</div>"
        def _span(eid): return f"<span style='color:{get_hex_color(eid)}; font-weight:bold;'>S{eid}</span>"
        html = ["<b style='color:#D32F2F;'>原始人工拓扑真值:</b><hr>"]
        for ev in events:
            t = ev.get('type')
            if t == 'E2E': html.append(f"<div style='margin-bottom:8px;'>• {_span(ev['stroke_a'])} (t={ev['t_a']}) 与 {_span(ev['stroke_b'])} (t={ev['t_b']}) 发生 <b>E2E</b> 端点对接。</div>")
            elif t == 'T': html.append(f"<div style='margin-bottom:8px;'>• {_span(ev['guest'])} (t={ev['guest_t']}) <b>T型搭接</b> 到 {_span(ev['host'])} (t={ev['host_t']}) 上 [夹角:{ev.get('angle',0)}°]。</div>")
            elif t == 'X': html.append(f"<div style='margin-bottom:8px;'>• {_span(ev['stroke_a'])} (t={ev['t_a']}) 与 {_span(ev['stroke_b'])} (t={ev['t_b']}) 发生 <b>X交叉</b> [夹角:{ev.get('angle',0)}°]。</div>")
        return "".join(html)

    def parse_token_junctions(self, sequence):
        junctions = []
        stroke_count = 0
        for item in sequence:
            if item.get("token_type") == "STROKE": stroke_count += 1
            elif item.get("token_type") == "JUNCTION":
                idx_a = stroke_count - 1 - item.get("ref_a_dist", 0)
                idx_b = stroke_count - 1 - item.get("ref_b_dist", 0)
                
                # 🌟 恢复为除以 T_BINS (32.0)
                # 此时 32/32 = 1.0, 16/32 = 0.5, 0/32 = 0.0 完美对称！
                ta = item.get("ta_bin", 0) / float(T_BINS) if "ta_bin" in item else item.get("ta", 0.0)
                tb = item.get("tb_bin", 0) / float(T_BINS) if "tb_bin" in item else item.get("tb", 0.0)
                
                junctions.append({"u": idx_a, "v": idx_b, "ta": ta, "tb": tb, "type": item.get("j_type", "Unknown")})
        return junctions

    def on_derivation_selected(self, item):
        deriv_name = item.text()
        self.current_seq = self.data_group[deriv_name]["sequence"]
        self.current_junctions = self.parse_token_junctions(self.current_seq)
        
        draw_skeleton(self.canvas2.ax, self.current_seq, with_arrows=True, with_order=True)
        self.canvas2.draw()
        
        info = [f"<b style='font-size:16px; color:#1976D2;'>派生规则: {deriv_name}</b><hr>"]
        info.append("<div style='margin-bottom:5px;'><b>Sequence 生成流:</b></div>")
        for i, token in enumerate(self.current_seq):
            if token.get("token_type") == "STROKE":
                info.append(f"<div style='color:{get_hex_color(i)}; font-weight:bold;'>[{i+1}] 生成 Stroke {i} (Shape:{token.get('shape_code')})</div>")
            elif token.get("token_type") == "JUNCTION":
                info.append(f"<div style='margin-left:20px; color:#E91E63;'>↳ 触发 {token.get('j_type')} (引用回溯: -{token.get('ref_a_dist')}, -{token.get('ref_b_dist')})</div>")
        self.text2.setHtml("".join(info))
        
        self.combo_junctions.blockSignals(True)
        self.combo_junctions.clear()
        if not self.current_junctions:
            self.combo_junctions.addItem("无交点信息")
        else:
            for i, j in enumerate(self.current_junctions):
                self.combo_junctions.addItem(f"[{i+1}] S{j['u']} & S{j['v']} - {j['type']} 交点", userData=j)
        self.combo_junctions.blockSignals(False)
        self.on_junction_changed()

    def on_junction_changed(self):
        j_data = self.combo_junctions.currentData()
        if not j_data:
            draw_skeleton(self.canvas3.ax, self.current_seq)
            self.text3.setHtml("<span style='color:grey;'>请选择上方交点查看详情...</span>")
        else:
            u, v = j_data['u'], j_data['v']
            draw_skeleton(self.canvas3.ax, self.current_seq, highlight_strokes=[u, v], junction_mark=(u, v, j_data['ta'], j_data['tb']))
            
            def _span(eid): return f"<span style='color:{get_hex_color(eid)}; font-weight:bold;'>S{eid}</span>"
            html = [f"<b style='font-size:18px; color:#D32F2F;'>微观物理验证</b><hr>"]
            html.append(f"<div style='font-size:15px;'><b>参与者：</b> {_span(u)} & {_span(v)}</div>")
            html.append(f"<div style='font-size:15px; margin-top:10px;'><b>Token 预测类型：</b> {j_data['type']}</div>")
            html.append(f"<div style='font-size:15px; margin-top:10px;'><b>解析还原施力点：</b></div>")
            # 🌟 t值现在会完美显示 1.000 或 0.000 了
            html.append(f"<ul><li>{_span(u)}: t = <b style='color:#E91E63'>{j_data['ta']:.3f}</b></li>")
            html.append(f"<li>{_span(v)}: t = <b style='color:#E91E63'>{j_data['tb']:.3f}</b></li></ul>")
            html.append(f"<div style='margin-top:10px; color:#555;'>* 红色十字 ❌ 代表大模型生成的 1D Token 映射回 2D 空间的绝对坐标落点。对比 Row 1 确认对齐度。</div>")
            self.text3.setHtml("".join(html))
            
        self.canvas3.draw()

# ==========================================
# 🚪 Level 1: 主窗口 (文件分发)
# ==========================================
class AuditorAppMain(QMainWindow):
    def __init__(self, dataset, orig_topo):
        super().__init__()
        self.dataset = dataset
        self.orig_topo = orig_topo  
        self.setWindowTitle("FontGPT Dataset Explorer")
        self.setGeometry(50, 50, 1600, 950)
        
        self.splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.splitter)
        
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        lbl_file = QLabel("📁 源文件池 (Source Files)")
        lbl_file.setStyleSheet("font-weight: bold; color: #333;")
        sidebar_layout.addWidget(lbl_file)
        
        self.list_files = QListWidget()
        self.list_files.addItems(sorted(self.dataset.keys()))
        self.list_files.itemClicked.connect(self.on_file_selected)
        sidebar_layout.addWidget(self.list_files)
        
        self.workspace_stack = QStackedWidget()
        self.page_gallery = QWidget()
        gal_layout = QVBoxLayout(self.page_gallery)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        container = QWidget(); self.grid = QGridLayout(container)
        scroll.setWidget(container)
        gal_layout.addWidget(scroll)
        self.workspace_stack.addWidget(self.page_gallery)
        
        self.splitter.addWidget(sidebar)
        self.splitter.addWidget(self.workspace_stack)
        self.splitter.setSizes([300, 1300])

    def render_original_thumbnail(self, char_data):
        fig = plt.figure(figsize=(1.5, 1.5), dpi=80)
        fig.patch.set_facecolor('#F5F5F5')
        ax = fig.add_subplot(111)
        draw_original_topo(ax, char_data)
        plt.tight_layout(pad=0)
        
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        buf.seek(0)
        qimg = QImage()
        qimg.loadFromData(buf.getvalue())
        return QPixmap.fromImage(qimg)

    def on_file_selected(self, item):
        source_file = item.text()
        file_data = self.dataset[source_file] 
        orig_file_data = self.orig_topo.get(source_file, {}) 
        
        while self.grid.count():
            child = self.grid.takeAt(0)
            if child.widget(): child.widget().deleteLater()
            
        cols = 6
        for idx, (hex_key, derivations_dict) in enumerate(file_data.items()):
            orig_char_data = orig_file_data.get(hex_key, {})
            
            card = QFrame()
            card.setStyleSheet("background-color: #FFF; border: 1px solid #CCC; border-radius: 6px;")
            card_layout = QVBoxLayout(card)
            
            lbl_img = QLabel()
            lbl_img.setPixmap(self.render_original_thumbnail(orig_char_data)) 
            lbl_img.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(lbl_img)
            
            title = QLabel(f"'{chr(int(hex_key[2:], 16))}' ({hex_key})")
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("font-weight: bold; border:none;")
            card_layout.addWidget(title)
            
            btn = QPushButton("🔍 验证架构")
            btn.setStyleSheet("background-color: #2196F3; color: white; padding: 5px; border-radius: 3px;")
            btn.clicked.connect(lambda checked, hk=hex_key, o_data=orig_char_data, d_data=derivations_dict: self.open_inspection(hk, o_data, d_data))
            card_layout.addWidget(btn)
            
            self.grid.addWidget(card, idx // cols, idx % cols)
            
        self.workspace_stack.setCurrentIndex(0)

    def open_inspection(self, hex_key, orig_data, data_group):
        if self.workspace_stack.count() > 1:
            old = self.workspace_stack.widget(1)
            self.workspace_stack.removeWidget(old)
            old.deleteLater()
            
        inspector = InspectionWidget(self, hex_key, orig_data, data_group)
        self.workspace_stack.addWidget(inspector)
        self.workspace_stack.setCurrentIndex(1)

# ==========================================
# 🚀 启动入口 (双轨数据装载)
# ==========================================
def load_datasets():
    if not os.path.exists(DATASET_FILE):
        print(f"❌ 找不到派生数据集 {DATASET_FILE}")
        sys.exit(1)
        
    print(f"📖 [Track 1] 正在加载 Token 派生数据: {DATASET_FILE} ...")
    dataset = defaultdict(lambda: defaultdict(dict))
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        for item in raw_data:
            src = item.get("source_file", "unknown_source.json")
            hk = item["hex_key"]
            rule = item["derivation"]
            dataset[src][hk][rule] = item
            
    print(f"📖 [Track 2] 正在加载 Original Topo 物理数据 (来自 {TOPO_DATA_DIR}) ...")
    orig_topo = {}
    if os.path.exists(TOPO_DATA_DIR):
        for fp in glob.glob(os.path.join(TOPO_DATA_DIR, "*_topo.json")):
            src_name = os.path.basename(fp)
            with open(fp, 'r', encoding='utf-8') as f:
                orig_topo[src_name] = json.load(f)
    else:
        print("⚠️ 找不到 annotations_topo 文件夹，Row 1 可能无法显示原始曲线。")
        
    print(f"📖 [Track 3] 正在加载 VQ-VAE 曲线密码本: {CLUSTER_FILE} ...")
    if os.path.exists(CLUSTER_FILE):
        with open(CLUSTER_FILE, 'r', encoding='utf-8') as f:
            for item in json.load(f):
                cid = int(item["cluster_id"])
                if cid != -1: SHAPE_CODEBOOK[cid] = item["mother_bezier"]
    else:
        print("⚠️ 找不到密码本文件，Row 2 将降级为画直线。")
            
    print(f"✅ 数据加载完毕！")
    return dataset, orig_topo

if __name__ == '__main__':
    app = QApplication(sys.argv)
    try:
        import qdarktheme
        qdarktheme.setup_theme("light", custom_colors={"primary": "#2196F3"})
    except ImportError: pass
        
    data, orig_data = load_datasets()
    window = AuditorAppMain(data, orig_data)
    window.show()
    sys.exit(app.exec_())