"""
Graph Transformer 数据集可视化审查工具
数据源: fontgpt_dataset_graph.json (Canonical Complete Graph 格式)

架构说明:
  - 每个样本 = 一个字符的增强派生
  - nodes[i]: { node_id, bezier_id, shape_code, variant_id, p0_cell, p0_offset, p3_cell, p3_offset, width_token }
  - edges[i]: { u, v, j_type, j_type_idx, t_u, t_v, t_diff, t_prod }  (N×N 完备图，含负样本)

展示逻辑:
  Row 1 - 原始真值拓扑（来自 annotations_topo，绝对真理层）
  Row 2 - Graph 节点还原图（从 graph nodes 重建笔画几何）
  Row 3 - 边约束探照灯（选择一条有效边，高亮两个笔画并标注交点）
"""

import os
import sys
import json
import glob
import io
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
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
# Graph Transformer 专用数据集
DATASET_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, "../fontgpt_dataset_graph.json"))
TOPO_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../AI_VECTOR_ROUTER_With_topo/annotations_topo"))
CLUSTER_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, "../clustered_results.json"))

CANVAS_SIZE = 400.0
GRID_BINS = 32
SCALE = CANVAS_SIZE / GRID_BINS  # 每格像素宽度

SHAPE_CODEBOOK = {}
J_TYPE_MAP = {0: "NONE", 1: "E2E", 2: "X", 3: "T"}
J_TYPE_COLORS = {0: "#CCCCCC", 1: "#2196F3", 2: "#FF5722", 3: "#4CAF50"}

try:
    CMAP = plt.colormaps['tab20']
except AttributeError:
    CMAP = plt.get_cmap('tab20')


def get_hex_color(idx):
    c = CMAP(idx % 20)
    return f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"


def cubic_bezier_np(pts, ts):
    mt = 1 - ts
    return (mt**3)*pts[0] + 3*(mt**2)*ts*pts[1] + 3*mt*(ts**2)*pts[2] + (ts**3)*pts[3]


def node_to_coords(node):
    """从 graph node 还原出 p0, p3 实际坐标（画布像素单位）"""
    p0_x = (node["p0_cell"][0] + node["p0_offset"][0]) / GRID_BINS * CANVAS_SIZE
    p0_y = (node["p0_cell"][1] + node["p0_offset"][1]) / GRID_BINS * CANVAS_SIZE
    p3_x = (node["p3_cell"][0] + node["p3_offset"][0]) / GRID_BINS * CANVAS_SIZE
    p3_y = (node["p3_cell"][1] + node["p3_offset"][1]) / GRID_BINS * CANVAS_SIZE
    return np.array([p0_x, p0_y]), np.array([p3_x, p3_y])


class MatplotlibCanvas(FigureCanvas):
    def __init__(self, width=5, height=5, dpi=80):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#FFFFFF')
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        super().__init__(self.fig)


# ==========================================
# 🎨 绘图函数层
# ==========================================

def draw_original_topo(ax, char_data):
    """Row 1: 绘制原始 annotations_topo 数据（绝对真理层）"""
    ax.clear()
    ax.axis('off')
    all_x, all_y = [], []
    if not char_data:
        ax.text(0.5, 0.5, "无原始数据", transform=ax.transAxes,
                ha='center', va='center', color='#999', fontsize=14)
        return

    strokes = char_data.get("strokes", [])
    for stroke in strokes:
        eid = stroke.get("bezier_id", 0)
        pts = np.array(stroke.get("mother_bezier", []))
        if len(pts) != 4:
            continue
        color = CMAP(eid % 20)
        ts = np.linspace(0, 1, 50)[:, None]
        curve = cubic_bezier_np(pts, ts)
        all_x.extend(curve[:, 0])
        all_y.extend(curve[:, 1])
        ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=4, alpha=0.9)
        ax.scatter(pts[0, 0], pts[0, 1], color='green', s=50, zorder=2)
        ax.scatter(pts[3, 0], pts[3, 1], color='red', s=50, zorder=2)
        mid = curve[25]
        ax.text(mid[0], mid[1], f"S{eid}", color=color, fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))

    if all_x and all_y:
        cx = (np.min(all_x) + np.max(all_x)) / 2
        cy = (np.min(all_y) + np.max(all_y)) / 2
        size = max(np.max(all_x) - np.min(all_x), np.max(all_y) - np.min(all_y)) / 2 * 1.2
        if size < 10:
            size = 50
        ax.set_xlim(cx - size, cx + size)
        ax.set_ylim(cy + size, cy - size)


def draw_graph_nodes(ax, nodes, highlight_ids=None, junction_mark=None,
                     with_node_labels=True, title_suffix=""):
    """
    Row 2 / Row 3: 绘制 Graph Transformer 格式的节点图

    Args:
        nodes: graph item 的 nodes 列表
        highlight_ids: 要高亮的 node_id 集合（其他节点淡化）
        junction_mark: (u, v, t_u, t_v) 标注交叉点
        with_node_labels: 是否显示节点编号
    """
    ax.clear()
    ax.axis('off')
    if not nodes:
        ax.text(0.5, 0.5, "无节点数据", transform=ax.transAxes,
                ha='center', va='center', color='#999', fontsize=14)
        return

    drawn_strokes = {}  # node_id -> (p0, p3)
    all_x, all_y = [], []

    for node in nodes:
        nid = node["node_id"]
        p0, p3 = node_to_coords(node)
        drawn_strokes[nid] = (p0, p3)

        is_faded = (highlight_ids is not None and nid not in highlight_ids)
        is_highlighted = (highlight_ids is not None and nid in highlight_ids)
        color = CMAP(nid % 20)
        alpha = 0.12 if is_faded else 0.9
        lw = 6 if is_highlighted else 3

        shape_code = node.get("shape_code", -1)
        var_id = node.get("variant_id", 0)

        # 如果密码本有此形态，用贝塞尔曲线绘制
        if shape_code in SHAPE_CODEBOOK:
            canon_pts = np.array(SHAPE_CODEBOOK[shape_code]).copy()
            # 变体翻转
            if var_id in [2, 3]:
                u_vec = canon_pts[3] - canon_pts[0]
                u_dot_u = np.dot(u_vec, u_vec)
                if u_dot_u > 1e-5:
                    for ci in (1, 2):
                        v_vec = canon_pts[ci] - canon_pts[0]
                        proj = (np.dot(v_vec, u_vec) / u_dot_u) * u_vec
                        perp = v_vec - proj
                        canon_pts[ci] = canon_pts[0] + proj - perp
            if var_id in [1, 3]:
                canon_pts = canon_pts[::-1]

            c0, c3 = canon_pts[0], canon_pts[3]
            v_canon = c3 - c0
            v_pred = p3 - p0
            len_c = np.linalg.norm(v_canon)
            len_p = np.linalg.norm(v_pred)

            if len_c > 1e-5 and len_p > 1e-5:
                s = len_p / len_c
                theta = np.arctan2(v_pred[1], v_pred[0]) - np.arctan2(v_canon[1], v_canon[0])
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                mapped_pts = (canon_pts - c0) @ R.T * s + p0
                ts = np.linspace(0, 1, 50)[:, None]
                curve = cubic_bezier_np(mapped_pts, ts)
                all_x.extend(curve[:, 0])
                all_y.extend(curve[:, 1])
                ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=lw, alpha=alpha, zorder=1)
            else:
                all_x.extend([p0[0], p3[0]])
                all_y.extend([p0[1], p3[1]])
                ax.plot([p0[0], p3[0]], [p0[1], p3[1]], color=color, linewidth=lw, alpha=alpha, zorder=1)
        else:
            # 降级为直线
            all_x.extend([p0[0], p3[0]])
            all_y.extend([p0[1], p3[1]])
            ax.plot([p0[0], p3[0]], [p0[1], p3[1]], color=color, linewidth=lw, alpha=alpha, zorder=1)

        if not is_faded:
            ax.scatter(p0[0], p0[1], color='green', s=40, zorder=2)
            ax.scatter(p3[0], p3[1], color='red', s=40, zorder=2)

        if with_node_labels and not is_faded:
            mid_pt = (p0 + p3) / 2
            ax.text(mid_pt[0], mid_pt[1], f"N{nid}", color=color,
                    fontsize=11, fontweight='bold', alpha=alpha,
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    # 交点标注（物理验证核心）
    if junction_mark is not None:
        u_id, v_id, t_u, t_v = junction_mark
        if u_id in drawn_strokes and v_id in drawn_strokes:
            p0_u, p3_u = drawn_strokes[u_id]
            p0_v, p3_v = drawn_strokes[v_id]
            pt_u = p0_u * (1 - t_u) + p3_u * t_u
            pt_v = p0_v * (1 - t_v) + p3_v * t_v
            center = (pt_u + pt_v) / 2
            all_x.append(center[0])
            all_y.append(center[1])
            ax.plot(center[0], center[1], marker='X', color='crimson',
                    markersize=22, markeredgecolor='black', zorder=10)
            ax.text(center[0] + 15, center[1] - 15,
                    f"({int(center[0])}, {int(center[1])})",
                    color='crimson', fontsize=11, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='crimson', pad=2))

            # 连接两个施力点
            ax.plot([pt_u[0], pt_v[0]], [pt_u[1], pt_v[1]],
                    '--', color='crimson', linewidth=1.5, alpha=0.6, zorder=5)

    if all_x and all_y:
        cx = (np.min(all_x) + np.max(all_x)) / 2
        cy = (np.min(all_y) + np.max(all_y)) / 2
        size = max(np.max(all_x) - np.min(all_x), np.max(all_y) - np.min(all_y)) / 2 * 1.2
        if size < 10:
            size = 50
        ax.set_xlim(cx - size, cx + size)
        ax.set_ylim(cy + size, cy - size)

    if title_suffix:
        ax.set_title(title_suffix, fontsize=10, color='#555', pad=3)


def draw_graph_topology_overlay(ax, nodes, edges, highlight_edge=None):
    """
    在 Row 2 图上叠加拓扑边结构可视化
    highlight_edge: (u, v) 要高亮的边
    """
    drawn_strokes = {}
    for node in nodes:
        nid = node["node_id"]
        p0, p3 = node_to_coords(node)
        drawn_strokes[nid] = (p0 + p3) / 2  # 节点中心点

    # 绘制有效边（j_type_idx > 0）
    for edge in edges:
        u, v = edge["u"], edge["v"]
        if u >= v:  # 只画上三角，避免重复
            continue
        j_idx = edge.get("j_type_idx", 0)
        if j_idx == 0:  # NONE 边不画
            continue
        if u not in drawn_strokes or v not in drawn_strokes:
            continue

        c_u = drawn_strokes[u]
        c_v = drawn_strokes[v]
        is_hl = highlight_edge and (edge["u"] == highlight_edge[0] and edge["v"] == highlight_edge[1])
        edge_color = J_TYPE_COLORS.get(j_idx, "#999999")
        lw = 3.0 if is_hl else 1.0
        alpha = 0.9 if is_hl else 0.35

        ax.plot([c_u[0], c_v[0]], [c_u[1], c_v[1]],
                color=edge_color, linewidth=lw, alpha=alpha,
                linestyle='-', zorder=0)
        mid = (c_u + c_v) / 2
        ax.text(mid[0], mid[1], J_TYPE_MAP.get(j_idx, "?"),
                color=edge_color, fontsize=8, alpha=alpha,
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))


# ==========================================
# 🖥️ 检视面板（Graph Transformer 版）
# ==========================================
class InspectionWidget(QWidget):
    """
    三层展示架构:
    Row 1 - 原始 topo 真理层（贝塞尔曲线，人工标注）
    Row 2 - Graph 节点层（从 graph nodes 重建 + 边拓扑叠加）
    Row 3 - 边约束探照灯（选择一条有效边，验证几何约束）
    """
    def __init__(self, main_app, hex_key, orig_data, data_group):
        super().__init__()
        self.main_app = main_app
        self.hex_key = hex_key
        self.orig_data = orig_data
        self.data_group = data_group
        self.derivations = list(self.data_group.keys())
        self.current_item = None      # 当前选中的 graph 样本
        self.current_edges_valid = [] # 当前有效边列表（j_type > 0）
        self.init_ui()
        if self.derivations:
            self.list_derivations.setCurrentRow(0)
            self.on_derivation_selected(self.list_derivations.item(0))

    def init_ui(self):
        layout = QHBoxLayout(self)

        # 左侧栏
        left_panel = QVBoxLayout()
        btn_back = QPushButton("🔙 返回画廊")
        btn_back.setStyleSheet("padding: 10px; background-color: #757575; color: white; font-weight: bold; border-radius: 4px;")
        btn_back.clicked.connect(lambda: self.main_app.workspace_stack.setCurrentIndex(0))
        left_panel.addWidget(btn_back)

        char_display = chr(int(self.hex_key[2:], 16)) if self.hex_key.startswith("0x") else "?"
        lbl = QLabel(f"字符 {char_display}\n({self.hex_key})\n派生组:")
        lbl.setStyleSheet("font-weight: bold; margin-top: 10px; font-size: 13px;")
        left_panel.addWidget(lbl)

        self.list_derivations = QListWidget()
        self.list_derivations.addItems(self.derivations)
        self.list_derivations.itemClicked.connect(self.on_derivation_selected)
        left_panel.addWidget(self.list_derivations)

        # 图例说明
        legend_html = """
        <div style='font-size:11px; color:#555; padding:6px;'>
        <b>边类型图例:</b><br>
        <span style='color:#2196F3'>■ E2E</span> 端点对接<br>
        <span style='color:#FF5722'>■ X</span> 交叉穿越<br>
        <span style='color:#4CAF50'>■ T</span> T型搭接<br>
        <span style='color:#CCCCCC'>■ NONE</span> 无连接
        </div>"""
        legend = QLabel(legend_html)
        legend.setWordWrap(True)
        left_panel.addWidget(legend)
        layout.addLayout(left_panel, 1)

        # 右侧主区域
        right_panel = QScrollArea()
        right_panel.setWidgetResizable(True)
        container = QWidget()
        self.vbox = QVBoxLayout(container)

        # Row 1: 原始真值
        self.vbox.addWidget(self._section_title(
            "🎯 Row 1: 【真理层】原始拓扑标注数据 (Absolute GT, 人工贝塞尔)"))
        row1 = QHBoxLayout()
        self.canvas1 = MatplotlibCanvas(width=5, height=4)
        self.text1 = QTextBrowser()
        self.text1.setMaximumHeight(300)
        row1.addWidget(self.canvas1, 3)
        row1.addWidget(self.text1, 2)
        self.vbox.addLayout(row1)

        # Row 2: Graph 节点层
        self.vbox.addWidget(self._section_title(
            "🕸️ Row 2: 【Graph 节点层】Canonical Complete Graph 还原 (含拓扑边叠加)"))
        row2 = QHBoxLayout()
        self.canvas2 = MatplotlibCanvas(width=5, height=4)
        self.text2 = QTextBrowser()
        self.text2.setMaximumHeight(300)
        row2.addWidget(self.canvas2, 3)
        row2.addWidget(self.text2, 2)
        self.vbox.addLayout(row2)

        # Row 3: 边约束探照灯
        self.vbox.addWidget(self._section_title(
            "🔦 Row 3: 【几何约束验证】Edge 交点探照灯 (选择边 → 验证物理吸合)"))
        self.combo_edges = QComboBox()
        self.combo_edges.setStyleSheet("padding: 5px; font-size: 13px; background-color: #FFF9C4;")
        self.combo_edges.currentIndexChanged.connect(self.on_edge_changed)
        self.vbox.addWidget(self.combo_edges)
        row3 = QHBoxLayout()
        self.canvas3 = MatplotlibCanvas(width=5, height=4)
        self.text3 = QTextBrowser()
        self.text3.setMaximumHeight(300)
        row3.addWidget(self.canvas3, 3)
        row3.addWidget(self.text3, 2)
        self.vbox.addLayout(row3)

        right_panel.setWidget(container)
        layout.addWidget(right_panel, 5)

        # 初始化 Row 1
        draw_original_topo(self.canvas1.ax, self.orig_data)
        self.canvas1.draw()
        self.text1.setHtml(self._gen_orig_topo_html(self.orig_data))

    def _section_title(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "font-size: 15px; font-weight: bold; background-color: #E3F2FD; "
            "padding: 6px; border-left: 4px solid #1976D2;")
        return lbl

    def _gen_orig_topo_html(self, char_data):
        if not char_data:
            return "<div style='color:red;'>缺少原始拓扑数据。</div>"
        events = char_data.get("topology_events", [])
        if not events:
            return "<div style='color:#777;'>原始数据中无拓扑交点记录。</div>"

        def _span(eid):
            return f"<span style='color:{get_hex_color(eid)}; font-weight:bold;'>S{eid}</span>"

        html = ["<b style='color:#D32F2F;'>原始人工拓扑真值:</b><hr>"]
        for ev in events:
            t = ev.get('type')
            if t == 'E2E':
                html.append(f"<div style='margin-bottom:8px;'>• {_span(ev['stroke_a'])} "
                            f"(t={ev['t_a']:.3f}) 与 {_span(ev['stroke_b'])} (t={ev['t_b']:.3f}) "
                            f"发生 <b style='color:#2196F3'>E2E</b> 端点对接。</div>")
            elif t == 'T':
                html.append(f"<div style='margin-bottom:8px;'>• {_span(ev['guest'])} "
                            f"(t={ev['guest_t']:.3f}) <b style='color:#4CAF50'>T型搭接</b> 到 "
                            f"{_span(ev['host'])} (t={ev['host_t']:.3f}) [夹角:{ev.get('angle', 0)}°]。</div>")
            elif t == 'X':
                html.append(f"<div style='margin-bottom:8px;'>• {_span(ev['stroke_a'])} "
                            f"(t={ev['t_a']:.3f}) 与 {_span(ev['stroke_b'])} (t={ev['t_b']:.3f}) "
                            f"发生 <b style='color:#FF5722'>X交叉</b> [夹角:{ev.get('angle', 0)}°]。</div>")
        return "".join(html)

    def on_derivation_selected(self, item):
        if item is None:
            return
        deriv_name = item.text()
        self.current_item = self.data_group[deriv_name]
        nodes = self.current_item.get("nodes", [])
        edges = self.current_item.get("edges", [])

        # 筛选有效边（非 NONE，且 u < v 去重）
        self.current_edges_valid = [
            e for e in edges
            if e.get("j_type_idx", 0) > 0 and e["u"] < e["v"]
        ]

        # Row 2 绘图
        draw_graph_nodes(self.canvas2.ax, nodes)
        draw_graph_topology_overlay(self.canvas2.ax, nodes, edges)
        self.canvas2.draw()

        # Row 2 文字
        self._update_graph_info(deriv_name, nodes, edges)

        # Row 3 更新候选边下拉框
        self.combo_edges.blockSignals(True)
        self.combo_edges.clear()
        if not self.current_edges_valid:
            self.combo_edges.addItem("⚠️ 此样本无有效拓扑边（全 NONE）")
        else:
            for e in self.current_edges_valid:
                j_name = J_TYPE_MAP.get(e["j_type_idx"], "?")
                self.combo_edges.addItem(
                    f"N{e['u']} ↔ N{e['v']}  [{j_name}]  t_u={e['t_u']:.3f} / t_v={e['t_v']:.3f}",
                    userData=e
                )
        self.combo_edges.blockSignals(False)
        self.on_edge_changed()

    def _update_graph_info(self, deriv_name, nodes, edges):
        """生成 Row 2 的节点/边统计信息面板"""
        n_nodes = len(nodes)
        valid_edges = [e for e in edges if e.get("j_type_idx", 0) > 0 and e["u"] < e["v"]]
        n_e2e = sum(1 for e in valid_edges if e.get("j_type_idx") == 1)
        n_x = sum(1 for e in valid_edges if e.get("j_type_idx") == 2)
        n_t = sum(1 for e in valid_edges if e.get("j_type_idx") == 3)

        html = [f"<b style='font-size:15px; color:#1976D2;'>派生规则: {deriv_name}</b><hr>"]
        html.append(f"<div><b>图统计:</b> {n_nodes} 节点 | {len(edges)} 条完备边 (N²={n_nodes**2})</div>")
        html.append(f"<div style='margin-top:6px;'>有效拓扑边: "
                    f"<span style='color:#2196F3'>E2E×{n_e2e}</span>  "
                    f"<span style='color:#FF5722'>X×{n_x}</span>  "
                    f"<span style='color:#4CAF50'>T×{n_t}</span></div>")
        html.append("<hr><b>节点详情:</b>")
        html.append("<table style='width:100%; font-size:12px;'>")
        html.append("<tr><th>ID</th><th>BezierID</th><th>Shape</th><th>Width</th><th>P0 (cell)</th><th>P3 (cell)</th></tr>")
        for n in nodes:
            color = get_hex_color(n["node_id"])
            html.append(
                f"<tr style='color:{color}; font-weight:bold;'>"
                f"<td>N{n['node_id']}</td>"
                f"<td>B{n['bezier_id']}</td>"
                f"<td>{n['shape_code']}</td>"
                f"<td>{n['width_token']}</td>"
                f"<td>({n['p0_cell'][0]}, {n['p0_cell'][1]})</td>"
                f"<td>({n['p3_cell'][0]}, {n['p3_cell'][1]})</td>"
                f"</tr>"
            )
        html.append("</table>")
        self.text2.setHtml("".join(html))

    def on_edge_changed(self):
        """Row 3: 高亮选中边的两个节点，标注交点"""
        if self.current_item is None:
            return
        nodes = self.current_item.get("nodes", [])
        edges = self.current_item.get("edges", [])

        edge_data = self.combo_edges.currentData()
        if edge_data is None:
            draw_graph_nodes(self.canvas3.ax, nodes,
                             title_suffix="← 请从上方选择一条有效边")
            self.canvas3.draw()
            self.text3.setHtml("<span style='color:#999;'>请从下拉框选择一条有效边查看几何约束验证...</span>")
            return

        u_id = edge_data["u"]
        v_id = edge_data["v"]
        t_u = edge_data["t_u"]
        t_v = edge_data["t_v"]
        j_idx = edge_data.get("j_type_idx", 0)
        j_name = J_TYPE_MAP.get(j_idx, "?")

        draw_graph_nodes(
            self.canvas3.ax, nodes,
            highlight_ids={u_id, v_id},
            junction_mark=(u_id, v_id, t_u, t_v),
            title_suffix=f"高亮 N{u_id} & N{v_id}  [{j_name}]"
        )
        # 叠加边
        draw_graph_topology_overlay(
            self.canvas3.ax, nodes, edges,
            highlight_edge=(edge_data["u"], edge_data["v"])
        )
        self.canvas3.draw()

        # 文字说明
        c_u = get_hex_color(u_id)
        c_v = get_hex_color(v_id)
        j_color = J_TYPE_COLORS.get(j_idx, "#999")

        # 计算实际坐标
        node_map = {n["node_id"]: n for n in nodes}
        p0_u, p3_u = node_to_coords(node_map[u_id]) if u_id in node_map else (np.zeros(2), np.zeros(2))
        p0_v, p3_v = node_to_coords(node_map[v_id]) if v_id in node_map else (np.zeros(2), np.zeros(2))
        pt_u = p0_u * (1 - t_u) + p3_u * t_u
        pt_v = p0_v * (1 - t_v) + p3_v * t_v
        gap = np.linalg.norm(pt_u - pt_v)

        html = [f"<b style='font-size:17px; color:#D32F2F;'>几何约束验证层</b><hr>"]
        html.append(f"<div style='font-size:14px;'>边类型: <b style='color:{j_color}'>{j_name}</b></div>")
        html.append(f"<div style='font-size:14px; margin-top:8px;'>"
                    f"笔画 <span style='color:{c_u}; font-weight:bold;'>N{u_id}</span> "
                    f"(BezierID={node_map[u_id]['bezier_id'] if u_id in node_map else '?'}) "
                    f"↔ 笔画 <span style='color:{c_v}; font-weight:bold;'>N{v_id}</span> "
                    f"(BezierID={node_map[v_id]['bezier_id'] if v_id in node_map else '?'})</div>")

        html.append(f"<div style='margin-top:12px;'><b>GT 施力参数 (来自数据集):</b></div>")
        html.append(f"<ul>")
        html.append(f"<li><span style='color:{c_u}; font-weight:bold;'>N{u_id}</span>: "
                    f"t_u = <b style='color:#E91E63'>{t_u:.4f}</b> "
                    f"→ 坐标 ({pt_u[0]:.1f}, {pt_u[1]:.1f})</li>")
        html.append(f"<li><span style='color:{c_v}; font-weight:bold;'>N{v_id}</span>: "
                    f"t_v = <b style='color:#E91E63'>{t_v:.4f}</b> "
                    f"→ 坐标 ({pt_v[0]:.1f}, {pt_v[1]:.1f})</li>")
        html.append(f"</ul>")

        gap_color = '#4CAF50' if gap < 5.0 else ('#FF9800' if gap < 15.0 else '#F44336')
        html.append(f"<div style='margin-top:10px; padding:6px; background:#F5F5F5; border-radius:4px;'>")
        html.append(f"<b>L_junction 物理验证:</b><br>")
        html.append(f"两施力点距离: <b style='color:{gap_color}'>{gap:.2f} px</b> "
                    f"({'✅ 近似吸合' if gap < 5.0 else '⚠️ 偏差较大' if gap < 15.0 else '❌ 偏差严重'})</div>")
        html.append(f"<div style='margin-top:8px; color:#777; font-size:12px;'>"
                    f"辅助特征: t_diff={edge_data.get('t_diff', 0):.4f}, "
                    f"t_prod={edge_data.get('t_prod', 0):.4f}</div>")
        html.append(f"<div style='margin-top:8px; color:#555; font-size:12px;'>"
                    f"* 红叉 ❌ 为两个施力点的中点坐标，理想情况下两点完全重合。</div>")

        self.text3.setHtml("".join(html))


# ==========================================
# 🖥️ 主窗口
# ==========================================
class AuditorAppMain(QMainWindow):
    def __init__(self, dataset, orig_topo):
        super().__init__()
        self.dataset = dataset
        self.orig_topo = orig_topo
        self.setWindowTitle("Graph Transformer Dataset Explorer  |  fontgpt_dataset_graph.json")
        self.setGeometry(50, 50, 1700, 980)

        self.splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.splitter)

        # 侧边栏
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.addWidget(QLabel("📁 源文件池"))
        self.list_files = QListWidget()
        self.list_files.addItems(sorted(self.dataset.keys()))
        self.list_files.itemClicked.connect(self.on_file_selected)
        sidebar_layout.addWidget(self.list_files)

        # 工作区
        self.workspace_stack = QStackedWidget()
        gallery_page = QWidget()
        gal_layout = QVBoxLayout(gallery_page)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.grid = QGridLayout(container)
        scroll.setWidget(container)
        gal_layout.addWidget(scroll)
        self.workspace_stack.addWidget(gallery_page)

        self.splitter.addWidget(sidebar)
        self.splitter.addWidget(self.workspace_stack)
        self.splitter.setSizes([280, 1420])

    def _render_thumbnail(self, char_data):
        """生成 Row 1 缩略图（原始 topo）"""
        fig = plt.figure(figsize=(1.5, 1.5), dpi=72)
        fig.patch.set_facecolor('#F5F5F5')
        ax = fig.add_subplot(111)
        draw_original_topo(ax, char_data)
        plt.tight_layout(pad=0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        buf.seek(0)
        qimg = QImage()
        qimg.loadFromData(buf.getvalue())
        return QPixmap.fromImage(qimg)

    def _render_graph_thumbnail(self, nodes):
        """生成 Row 2 缩略图（graph 节点结构）"""
        fig = plt.figure(figsize=(1.5, 1.5), dpi=72)
        fig.patch.set_facecolor('#EFF8FF')
        ax = fig.add_subplot(111)
        draw_graph_nodes(ax, nodes, with_node_labels=False)
        plt.tight_layout(pad=0)
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
            if child.widget():
                child.widget().deleteLater()

        cols = 5
        for idx, (hex_key, derivations_dict) in enumerate(file_data.items()):
            orig_char_data = orig_file_data.get(hex_key, {})

            # 获取第一个派生样本用于缩略图
            first_item = next(iter(derivations_dict.values()))
            first_nodes = first_item.get("nodes", [])
            n_deriv = len(derivations_dict)
            n_edges_valid = sum(
                1 for e in first_item.get("edges", [])
                if e.get("j_type_idx", 0) > 0 and e["u"] < e["v"]
            )

            card = QFrame()
            card.setStyleSheet("background-color: #FFF; border: 1px solid #CCC; border-radius: 6px;")
            card_layout = QVBoxLayout(card)

            # 双缩略图（上: 原始topo，下: graph还原）
            thumb_row = QHBoxLayout()
            lbl_t1 = QLabel()
            lbl_t1.setPixmap(self._render_thumbnail(orig_char_data))
            lbl_t1.setAlignment(Qt.AlignCenter)
            lbl_t1.setToolTip("原始 Topo GT")
            thumb_row.addWidget(lbl_t1)

            lbl_t2 = QLabel()
            lbl_t2.setPixmap(self._render_graph_thumbnail(first_nodes))
            lbl_t2.setAlignment(Qt.AlignCenter)
            lbl_t2.setToolTip("Graph 节点还原")
            thumb_row.addWidget(lbl_t2)
            card_layout.addLayout(thumb_row)

            # 字符标题
            try:
                char_str = chr(int(hex_key[2:], 16))
            except Exception:
                char_str = "?"
            title = QLabel(f"'{char_str}' ({hex_key})")
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("font-weight: bold; border:none; font-size: 12px;")
            card_layout.addWidget(title)

            # 统计信息
            stat = QLabel(f"派生×{n_deriv}  |  N={len(first_nodes)}  |  边×{n_edges_valid}")
            stat.setAlignment(Qt.AlignCenter)
            stat.setStyleSheet("color: #666; font-size: 10px; border: none;")
            card_layout.addWidget(stat)

            btn = QPushButton("🔍 深度验证")
            btn.setStyleSheet("background-color: #1565C0; color: white; padding: 5px; border-radius: 3px;")
            btn.clicked.connect(
                lambda checked, hk=hex_key, o=orig_char_data, d=derivations_dict:
                self.open_inspection(hk, o, d)
            )
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
# 📦 数据加载
# ==========================================
def load_datasets():
    if not os.path.exists(DATASET_FILE):
        print(f"❌ 找不到 Graph 数据集: {DATASET_FILE}")
        print("   请先运行 step3_build_transformer_dataset_v0.py 生成数据集。")
        sys.exit(1)

    print(f"📖 [Track 1] 加载 Graph Transformer 数据集: {DATASET_FILE} ...")
    dataset = defaultdict(lambda: defaultdict(dict))
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        for item in raw_data:
            src = item.get("source_file", "unknown_source.json")
            hk = item["hex_key"]
            rule = item["derivation"]
            dataset[src][hk][rule] = item

    print(f"   ✅ 加载完成: {sum(len(v) for v in dataset.values())} 个字符派生组，"
          f"共 {len(raw_data)} 条样本")

    print(f"📖 [Track 2] 加载原始拓扑真值 ({TOPO_DATA_DIR}) ...")
    orig_topo = {}
    if os.path.exists(TOPO_DATA_DIR):
        for fp in glob.glob(os.path.join(TOPO_DATA_DIR, "*_topo.json")):
            src_name = os.path.basename(fp)
            with open(fp, 'r', encoding='utf-8') as f:
                orig_topo[src_name] = json.load(f)
        print(f"   ✅ 加载 {len(orig_topo)} 个 topo 文件")
    else:
        print("   ⚠️ 未找到 annotations_topo 文件夹，Row 1 将显示空白")

    print(f"📖 [Track 3] 加载 VQ-VAE 曲线密码本: {CLUSTER_FILE} ...")
    if os.path.exists(CLUSTER_FILE):
        with open(CLUSTER_FILE, 'r', encoding='utf-8') as f:
            for item in json.load(f):
                cid = int(item["cluster_id"])
                if cid != -1 and cid not in SHAPE_CODEBOOK:
                    SHAPE_CODEBOOK[cid] = item["mother_bezier"]
        print(f"   ✅ 加载 {len(SHAPE_CODEBOOK)} 个形状原型")
    else:
        print("   ⚠️ 未找到密码本，绘图将降级为直线")

    return dataset, orig_topo


if __name__ == '__main__':
    app = QApplication(sys.argv)
    try:
        import qdarktheme
        qdarktheme.setup_theme("light", custom_colors={"primary": "#1565C0"})
    except ImportError:
        pass

    data, orig_data = load_datasets()
    window = AuditorAppMain(data, orig_data)
    window.show()
    sys.exit(app.exec_())
