import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QScrollArea, 
                             QStackedWidget, QGridLayout, QListWidget, QSplitter, 
                             QFrame, QTextBrowser, QComboBox)
from PyQt5.QtGui import QPixmap, QImage, QFont
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
CANVAS_SIZE = 1000.0

try:
    CMAP = plt.colormaps['tab20']
except AttributeError:
    CMAP = plt.get_cmap('tab20')

def get_hex_color(idx):
    c = CMAP(idx % 20)
    return f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"

# ==========================================
# 🎨 核心绘图引擎 (集成渲染逻辑)
# ==========================================
class MatplotlibCanvas(FigureCanvas):
    def __init__(self, width=5, height=5, dpi=80):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#FFFFFF')
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        super().__init__(self.fig)

def draw_skeleton(ax, sequence, with_arrows=False, with_order=False, highlight_strokes=None, junction_mark=None):
    """绘制字形的骨架图 (支持反解析 Cell 与 Offset)"""
    ax.clear()
    ax.set_xlim(0, CANVAS_SIZE)
    ax.set_ylim(CANVAS_SIZE, 0) # Y轴向下
    ax.axis('off')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    drawn_strokes = []
    stroke_idx = 0
    scale = CANVAS_SIZE / 32.0  # 假设 GRID_BINS = 32
    
    # 解析并绘制笔画
    for item in sequence:
        if item.get("token_type") == "STROKE":
            # 🌟 核心兼容补丁：从离散的 Cell 和 Offset 反推绝对物理坐标
            if "p0" in item:
                # 兼容老数据格式
                p0 = np.array(item["p0"])
                p3 = np.array(item["p3"])
            else:
                # 解析 Vector Language 格式
                p0 = np.array([(item["p0_cell"][0] + 0.5 + item["p0_offset"][0]) * scale, 
                               (item["p0_cell"][1] + 0.5 + item["p0_offset"][1]) * scale])
                p3 = np.array([(item["p3_cell"][0] + 0.5 + item["p3_offset"][0]) * scale, 
                               (item["p3_cell"][1] + 0.5 + item["p3_offset"][1]) * scale])
            
            drawn_strokes.append((p0, p3))
            
            # 状态判定
            is_highlighted = (highlight_strokes is not None and stroke_idx in highlight_strokes)
            is_faded = (highlight_strokes is not None and stroke_idx not in highlight_strokes)
            
            color = CMAP(stroke_idx % 20)
            alpha = 0.15 if is_faded else 0.9
            lw = 6 if is_highlighted else 3
            
            # 画线/箭头
            if with_arrows and not is_faded:
                ax.annotate('', xy=p3, xytext=p0, 
                            arrowprops=dict(arrowstyle='->', color=color, lw=lw, alpha=alpha, mutation_scale=20))
            else:
                ax.plot([p0[0], p3[0]], [p0[1], p3[1]], color=color, linewidth=lw, alpha=alpha, zorder=1)
                
            # 画端点
            if not is_faded:
                ax.scatter(p0[0], p0[1], color='green', s=40, zorder=2)
                ax.scatter(p3[0], p3[1], color='red', s=40, zorder=2)
                
            # 文本标签 (ID)
            mid_pt = (p0 + p3) / 2
            ax.text(mid_pt[0], mid_pt[1], f"S{stroke_idx}", color=color, fontsize=12, fontweight='bold', alpha=alpha)
            
            # 绘制绝对时间顺序 (①, ②...)
            if with_order and not is_faded:
                ax.text(p0[0]-20, p0[1]-20, f"[{stroke_idx+1}]", color='black', fontsize=14, fontweight='bold',
                        bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none', boxstyle='circle,pad=0.2'))
            
            stroke_idx += 1
            
    # 如果传入了特定交点，进行红色十字星高亮标注
    if junction_mark:
        u, v, ta, tb = junction_mark
        if 0 <= u < len(drawn_strokes) and 0 <= v < len(drawn_strokes):
            p0_u, p3_u = drawn_strokes[u]
            p0_v, p3_v = drawn_strokes[v]
            
            # 简易线性插值算交点坐标
            pt_u = p0_u * (1 - ta) + p3_u * ta
            pt_v = p0_v * (1 - tb) + p3_v * tb
            center_pt = (pt_u + pt_v) / 2
            
            ax.plot(center_pt[0], center_pt[1], marker='X', color='red', markersize=20, markeredgecolor='black', zorder=10)
            ax.text(center_pt[0]+20, center_pt[1]+20, f"({int(center_pt[0])}, {int(center_pt[1])})", 
                    color='red', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

# ==========================================
# 🏠 Level 2: 派生检视器主界面 (三段式详情)
# ==========================================
class InspectionWidget(QWidget):
    def __init__(self, main_app, hex_key, data_group):
        super().__init__()
        self.main_app = main_app
        self.hex_key = hex_key
        self.data_group = data_group  # {derivation_name: sequence_dict}
        self.derivations = list(self.data_group.keys())
        
        # 提取原版基准数据 (通常是第一个)
        self.base_seq = self.data_group[self.derivations[0]]["sequence"]
        self.current_seq = None
        self.current_junctions = []
        
        self.init_ui()
        if self.derivations:
            self.list_derivations.setCurrentRow(0)

    def init_ui(self):
        layout = QHBoxLayout(self)
        
        # --- 左侧：派生规则列表 ---
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
        
        # --- 右侧：三段式详细排查面板 ---
        right_panel = QScrollArea()
        right_panel.setWidgetResizable(True)
        container = QWidget()
        self.vbox = QVBoxLayout(container)
        
        # Row 1: 原字体样式与拓扑 (基准)
        self.vbox.addWidget(self.create_section_title("🎯 Row 1: 原字体骨架与拓扑信息 (Baseline)"))
        self.row1_layout = QHBoxLayout()
        self.canvas1 = MatplotlibCanvas(width=4, height=4)
        self.text1 = QTextBrowser()
        self.row1_layout.addWidget(self.canvas1, 1)
        self.row1_layout.addWidget(self.text1, 1)
        self.vbox.addLayout(self.row1_layout)
        
        # Row 2: 派生样式、运笔方向与顺序
        self.vbox.addWidget(self.create_section_title("🚀 Row 2: 当前派生方向与顺序 (Derivation)"))
        self.row2_layout = QHBoxLayout()
        self.canvas2 = MatplotlibCanvas(width=4, height=4)
        self.text2 = QTextBrowser()
        self.row2_layout.addWidget(self.canvas2, 1)
        self.row2_layout.addWidget(self.text2, 1)
        self.vbox.addLayout(self.row2_layout)
        
        # Row 3: 拓扑交点探照灯
        self.vbox.addWidget(self.create_section_title("🔦 Row 3: 拓扑交点精准探照灯 (Junction Spotlight)"))
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
        
        # 预渲染 Row 1 (因为基准是不变的)
        draw_skeleton(self.canvas1.ax, self.base_seq, with_arrows=False, with_order=False)
        self.canvas1.draw()
        self.text1.setHtml(self.generate_topo_html(self.base_seq))

    def create_section_title(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #E0E0E0; padding: 5px;")
        return lbl

    def parse_junctions(self, sequence):
        """解析并提取 Token 序列中的所有交点"""
        junctions = []
        stroke_count = 0
        for item in sequence:
            if item.get("token_type") == "STROKE": stroke_count += 1
            elif item.get("token_type") == "JUNCTION":
                # 相对距离还原绝对索引
                idx_a = stroke_count - 1 - item.get("ref_a_dist", 0)
                idx_b = stroke_count - 1 - item.get("ref_b_dist", 0)
                ta = item.get("ta_bin", 0) / 32.0 if "ta_bin" in item else item.get("ta", 0.0)
                tb = item.get("tb_bin", 0) / 32.0 if "tb_bin" in item else item.get("tb", 0.0)
                junctions.append({
                    "u": idx_a, "v": idx_b, "ta": ta, "tb": tb,
                    "type": item.get("j_type", "Unknown")
                })
        return junctions

    def generate_topo_html(self, sequence):
        """生成富文本拓扑报告"""
        junctions = self.parse_junctions(sequence)
        if not junctions: return "<div style='color:#777;'>当前序列无拓扑交点记录。</div>"
        
        def _span(eid): return f"<span style='color:{get_hex_color(eid)}; font-weight:bold;'>S{eid}</span>"
        html = []
        for j in junctions:
            html.append(f"<div style='margin-bottom:8px;'>• {_span(j['u'])} (t={j['ta']:.2f}) 与 {_span(j['v'])} (t={j['tb']:.2f}) 发生 <b>{j['type']}</b> 相交。</div>")
        return "".join(html)

    def on_derivation_selected(self, item):
        deriv_name = item.text()
        self.current_seq = self.data_group[deriv_name]["sequence"]
        self.current_junctions = self.parse_junctions(self.current_seq)
        
        # 渲染 Row 2
        draw_skeleton(self.canvas2.ax, self.current_seq, with_arrows=True, with_order=True)
        self.canvas2.draw()
        
        info = [f"<b style='font-size:16px; color:#1976D2;'>派生规则: {deriv_name}</b><hr>"]
        info.append("<div style='margin-bottom:5px;'><b>顺序流:</b></div>")
        for i, token in enumerate(self.current_seq):
            if token.get("token_type") == "STROKE":
                info.append(f"<div style='color:{get_hex_color(i)}; font-weight:bold;'>[{i+1}] 绘制 Stroke {i} (Shape:{token.get('shape_code')})</div>")
            elif token.get("token_type") == "JUNCTION":
                info.append(f"<div style='margin-left:20px; color:#E91E63;'>↳ 触发交点: {token.get('j_type')} (回溯: -{token.get('ref_a_dist')}, -{token.get('ref_b_dist')})</div>")
        self.text2.setHtml("".join(info))
        
        # 更新 Row 3 的下拉菜单
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
            # 图形高亮渲染
            draw_skeleton(self.canvas3.ax, self.current_seq, highlight_strokes=[u, v], junction_mark=(u, v, j_data['ta'], j_data['tb']))
            
            # 富文本详细分析
            def _span(eid): return f"<span style='color:{get_hex_color(eid)}; font-weight:bold;'>S{eid}</span>"
            html = [f"<b style='font-size:18px; color:#D32F2F;'>交点微观物理分析</b><hr>"]
            html.append(f"<div style='font-size:15px;'><b>参与者：</b> {_span(u)} & {_span(v)}</div>")
            html.append(f"<div style='font-size:15px; margin-top:10px;'><b>相交类型：</b> {j_data['type']}</div>")
            html.append(f"<div style='font-size:15px; margin-top:10px;'><b>施力位置比例：</b></div>")
            html.append(f"<ul><li>{_span(u)} 的切点位于 t = <b style='color:#E91E63'>{j_data['ta']:.3f}</b></li>")
            html.append(f"<li>{_span(v)} 的切点位于 t = <b style='color:#E91E63'>{j_data['tb']:.3f}</b></li></ul>")
            html.append(f"<div style='margin-top:10px; color:#555;'>* 红色十字 ❌ 已在图中标定绝对欧氏坐标位置。</div>")
            self.text3.setHtml("".join(html))
            
        self.canvas3.draw()

# ==========================================
# 🚪 Level 1: 主窗口 (文件分发与画廊)
# ==========================================
class AuditorAppMain(QMainWindow):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.setWindowTitle("FontGPT Dataset Auditor")
        self.setGeometry(50, 50, 1600, 950)
        
        self.splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.splitter)
        
        # --- 左侧：源文件列表 ---
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        lbl_file = QLabel("📁 数据集源文件 (Source Files)")
        lbl_file.setStyleSheet("font-weight: bold; color: #333;")
        sidebar_layout.addWidget(lbl_file)
        
        self.list_files = QListWidget()
        self.list_files.addItems(sorted(self.dataset.keys()))
        self.list_files.itemClicked.connect(self.on_file_selected)
        sidebar_layout.addWidget(self.list_files)
        
        # --- 右侧：多页工作区 ---
        self.workspace_stack = QStackedWidget()
        
        # Page 0: 画廊页
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

    def render_thumbnail(self, sequence):
        """为画廊生成快速缩略图"""
        fig = plt.figure(figsize=(1.5, 1.5), dpi=80)
        fig.patch.set_facecolor('#F5F5F5')
        ax = fig.add_subplot(111)
        draw_skeleton(ax, sequence)
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
        file_data = self.dataset[source_file] # {hex_key: {derivation: dict}}
        
        while self.grid.count():
            child = self.grid.takeAt(0)
            if child.widget(): child.widget().deleteLater()
            
        cols = 6
        for idx, (hex_key, derivations_dict) in enumerate(file_data.items()):
            # 默认用第一条派生作为缩略图
            first_rule = list(derivations_dict.keys())[0]
            base_seq = derivations_dict[first_rule]["sequence"]
            
            card = QFrame()
            card.setStyleSheet("background-color: #FFF; border: 1px solid #CCC; border-radius: 6px;")
            card_layout = QVBoxLayout(card)
            
            lbl_img = QLabel()
            lbl_img.setPixmap(self.render_thumbnail(base_seq))
            lbl_img.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(lbl_img)
            
            title = QLabel(f"'{chr(int(hex_key[2:], 16))}' ({hex_key})")
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("font-weight: bold; border:none;")
            card_layout.addWidget(title)
            
            btn = QPushButton("🔍 审查拓扑")
            btn.setStyleSheet("background-color: #2196F3; color: white; padding: 5px; border-radius: 3px;")
            btn.clicked.connect(lambda checked, hk=hex_key, data=derivations_dict: self.open_inspection(hk, data))
            card_layout.addWidget(btn)
            
            self.grid.addWidget(card, idx // cols, idx % cols)
            
        self.workspace_stack.setCurrentIndex(0)

    def open_inspection(self, hex_key, data_group):
        if self.workspace_stack.count() > 1:
            old = self.workspace_stack.widget(1)
            self.workspace_stack.removeWidget(old)
            old.deleteLater()
            
        inspector = InspectionWidget(self, hex_key, data_group)
        self.workspace_stack.addWidget(inspector)
        self.workspace_stack.setCurrentIndex(1)


# ==========================================
# 🚀 启动入口与数据加载
# ==========================================
def load_dataset():
    if not os.path.exists(DATASET_FILE):
        print(f"❌ 致命错误：找不到数据集 {DATASET_FILE}")
        sys.exit(1)
        
    print(f"📖 正在加载 FontGPT 数据集: {DATASET_FILE} ...")
    dataset = defaultdict(lambda: defaultdict(dict))
    
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        for item in raw_data:
            src = item.get("source_file", "unknown_source.json")
            hk = item["hex_key"]
            rule = item["derivation"]
            dataset[src][hk][rule] = item
            
    print(f"✅ 加载完毕！共计 {len(dataset)} 个源文件区块。")
    return dataset

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 启用暗黑或高亮 UI 增强 (可选)
    try:
        import qdarktheme
        qdarktheme.setup_theme("light", custom_colors={"primary": "#2196F3"})
    except ImportError:
        pass
        
    data = load_dataset()
    window = AuditorAppMain(data)
    window.show()
    sys.exit(app.exec_())