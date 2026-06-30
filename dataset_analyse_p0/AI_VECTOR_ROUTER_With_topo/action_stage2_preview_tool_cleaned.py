# -*- coding: utf-8 -*-
"""
action_stage2_preview_tool_cleaned.py

Phase 2 拓扑标注结果 Preview Tool（Cleaned edit_history 版本）。
读取 annotations_topo/{字体名}_topo.json，展示每个字符 Phase 2 标注的逐步操作时光机。
每一帧包含：
  - 彩色贝塞尔笔画图
  - 实时拓扑状态文本（仿照 topo_editor_workspace.py 的格式）

放置路径：
    dataset_analyse_p0/AI_VECTOR_ROUTER_With_topo/action_stage2_preview_tool.py
"""

import os
import sys
import copy
import io
import glob
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QScrollArea, QStackedWidget, QGridLayout,
    QListWidget, QSplitter, QFrame, QTextBrowser
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

try:
    import qdarktheme
    HAS_QDARK = True
except ImportError:
    HAS_QDARK = False

import logging

from ml_engine_phase2.phase2_edit_history_cleaner import clean_edit_history

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# ── 路径常量 ────────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR       = os.path.join(SCRIPT_DIR, "../dataset/alien_tensors_raw")
TOPO_ANNO_DIR   = os.path.join(SCRIPT_DIR, "annotations_topo")
ANNO_DIR        = os.path.join(SCRIPT_DIR, "annotations")   # Phase 1 最终形态
CANVAS_SIZE     = 400
TOPO_SAMPLE_N   = 120

# ── Tab20 色彩 ──────────────────────────────────────────────────────────────
try:
    _CMAP = plt.colormaps['tab20']
except AttributeError:
    _CMAP = plt.get_cmap('tab20')


def _tab20_hex(bezier_id: int) -> str:
    """根据 bezier_id 返回 tab20 色板的 hex 颜色字符串"""
    c = _CMAP((bezier_id % 20))
    return f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"


def _tab20_rgb(bezier_id: int):
    """返回 (r, g, b) float 元组"""
    return _CMAP((bezier_id % 20))[:3]


# ── 贝塞尔工具 ──────────────────────────────────────────────────────────────
def cubic_bezier_np(P: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """三次贝塞尔采样。P:[4,2], ts:[N,1] → [N,2]"""
    mt = 1.0 - ts
    return (mt**3 * P[0] + 3*mt**2*ts * P[1] +
            3*mt*ts**2 * P[2] + ts**3 * P[3])


def bezier_derivative(P: np.ndarray, t: float) -> np.ndarray:
    """贝塞尔曲线在参数 t 处的切线向量"""
    mt = 1.0 - t
    return (3*mt**2*(P[1]-P[0]) + 6*mt*t*(P[2]-P[1]) + 3*t**2*(P[3]-P[2]))


def get_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-5 or n2 < 1e-5:
        return 0.0
    cos_th = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_th)))


# ── 线段相交（用于 X 型检测）──────────────────────────────────────────────
def _cross2d(a, b):
    return float(a[0]*b[1] - a[1]*b[0])


def _seg_intersect(p0, p1, q0, q1, eps=1e-8):
    r, s = p1 - p0, q1 - q0
    denom = _cross2d(r, s)
    if abs(denom) < eps:
        return None
    qp = q0 - p0
    t = _cross2d(qp, s) / denom
    u = _cross2d(qp, r) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return p0 + t * r, float(t), float(u)
    return None


def _polyline_x_intersection(c1, c2, margin=2):
    n1, n2 = len(c1), len(c2)
    if n1 < 2 or n2 < 2:
        return False, None, None, None
    i_s, i_e = max(0, margin), max(0, n1 - 1 - margin)
    j_s, j_e = max(0, margin), max(0, n2 - 1 - margin)
    for i in range(i_s, i_e):
        for j in range(j_s, j_e):
            hit = _seg_intersect(c1[i], c1[i+1], c2[j], c2[j+1])
            if hit is None:
                continue
            pt, lt1, lt2 = hit
            t1 = (i + lt1) / (n1 - 1)
            t2 = (j + lt2) / (n2 - 1)
            if t1 <= 0.02 or t1 >= 0.98 or t2 <= 0.02 or t2 >= 0.98:
                continue
            return True, pt, float(t1), float(t2)
    return False, None, None, None


# ── 拓扑计算（从 strokes 重算，仿 topo_editor_workspace）──────────────────
def compute_topo_html(strokes: list) -> str:
    """
    根据当前 strokes（包含 mother_bezier 字段）计算拓扑关系，
    返回与 topo_editor_workspace.py 格式完全一致的 HTML 字符串。
    """
    if not strokes:
        return "<div style='color:#777;'>（无笔画数据）</div>"

    # bezier_id → 颜色
    def _span(bid):
        hx = _tab20_hex(bid)
        return f"<span style='color:{hx}; font-weight:bold;'>{bid}</span>"

    end_to_end, t_junctions, x_junctions = [], [], []
    connection_points = {}

    n = len(strokes)
    for i in range(n):
        s1 = strokes[i]
        p1 = np.array(s1["mother_bezier"], dtype=float)
        id1 = int(s1["bezier_id"])
        ts = np.linspace(0, 1, TOPO_SAMPLE_N)[:, None]
        c1 = cubic_bezier_np(p1, ts)

        for j in range(i + 1, n):
            s2 = strokes[j]
            p2 = np.array(s2["mother_bezier"], dtype=float)
            id2 = int(s2["bezier_id"])
            c2 = cubic_bezier_np(p2, ts)

            is_e2e = False
            t_rels = []
            pts = []

            # 1. 端点对接 E2E
            for pt1_idx in [0, 3]:
                for pt2_idx in [0, 3]:
                    if np.linalg.norm(p1[pt1_idx] - p2[pt2_idx]) < 2.0:
                        is_e2e = True
                        pts.append(np.array(p1[pt1_idx]))

            # 2. T 型搭接
            if not is_e2e:
                for pt1_idx in [0, 3]:
                    dists = np.linalg.norm(c2 - p1[pt1_idx], axis=1)
                    if np.min(dists) < 2.0:
                        t_rels.append((id1, id2))
                        pts.append(np.array(p1[pt1_idx]))
                for pt2_idx in [0, 3]:
                    dists = np.linalg.norm(c1 - p2[pt2_idx], axis=1)
                    if np.min(dists) < 2.0:
                        t_rels.append((id2, id1))
                        pts.append(np.array(p2[pt2_idx]))

            # 3. X 型交叉
            is_x = False
            if not is_e2e and not t_rels:
                hit_x, x_pt, _, _ = _polyline_x_intersection(c1, c2)
                if hit_x:
                    is_x = True
                    pts.append(x_pt)

            if is_e2e or t_rels or is_x:
                uk, vk = min(id1, id2), max(id1, id2)
                connection_points[(uk, vk)] = pts

            if is_e2e:
                end_to_end.append(f"{_span(id1)}-{_span(id2)}")
            elif t_rels:
                for guest_id, host_id in list(set(t_rels)):
                    t_junctions.append(f"{_span(guest_id)} 搭在 {_span(host_id)} 上")
            elif is_x:
                x_junctions.append(f"{_span(id1)} 交叉 {_span(id2)}")

    # 闭环检测
    cycle_strs = []
    try:
        import networkx as nx
        G = nx.Graph()
        for s in strokes:
            G.add_node(int(s["bezier_id"]))
        for (u, v) in connection_points:
            G.add_edge(u, v)

        valid_cycles = []
        # 2-stroke 闭环
        for (u, v), pts_list in connection_points.items():
            if len(pts_list) >= 2:
                for a in range(len(pts_list)):
                    for b in range(a + 1, len(pts_list)):
                        if np.linalg.norm(pts_list[a] - pts_list[b]) >= 5.0:
                            valid_cycles.append([u, v])
                            break
                    else:
                        continue
                    break
        # >=3-stroke 闭环
        for cycle in nx.cycle_basis(G):
            k = len(cycle)
            if k < 3:
                continue
            ok = True
            for ci in range(k):
                u, v, w = cycle[ci-1], cycle[ci], cycle[(ci+1) % k]
                key1, key2 = (min(u,v), max(u,v)), (min(v,w), max(v,w))
                if key1 not in connection_points or key2 not in connection_points:
                    ok = False
                    break
                p_in  = connection_points[key1][0]
                p_out = connection_points[key2][0]
                if np.linalg.norm(p_in - p_out) < 5.0:
                    ok = False
                    break
            if ok:
                valid_cycles.append(cycle)

        for cyc in valid_cycles:
            cycle_strs.append(f"{' '.join(_span(n) for n in cyc)} 属同一环")
    except Exception:
        pass

    html = ["<b style='color:#333; font-size:13px;'>📊 实时拓扑状态反馈</b><br>"]
    if end_to_end:
        html.append(f"<div style='margin-bottom:3px;'><b>[端点对接]：</b>{' 、 '.join(end_to_end)}</div>")
    if t_junctions:
        html.append(f"<div style='margin-bottom:3px;'><b>[T型搭接]：</b>{' 、 '.join(t_junctions)}</div>")
    if x_junctions:
        html.append(f"<div style='margin-bottom:3px;'><b>[X型交叉]：</b>{' 、 '.join(x_junctions)}</div>")
    if cycle_strs:
        html.append(f"<div style='margin-bottom:3px;'><b>[闭环结构]：</b>{' &nbsp;|&nbsp; '.join(cycle_strs)}</div>")
    if not (end_to_end or t_junctions or x_junctions or cycle_strs):
        html.append("<div style='color:#777;'>当前无曲线发生物理碰撞。</div>")

    return "".join(html)


def _fmt_coord(pt) -> str:
    try:
        return f"({float(pt[0]):.1f}, {float(pt[1]):.1f})"
    except Exception:
        return "(?)"


def describe_action_html(op: dict) -> str:
    """生成当前步骤的操作说明 HTML。"""
    if not isinstance(op, dict):
        return "<div style='color:#777; margin-bottom:5px;'>[操作] 初始状态，无移动。</div>"

    action = op.get("action", "?")
    before = op.get("before")
    after = op.get("after")
    merged_sources = op.get("merged_sources") or []
    merged_note = ""
    if merged_sources:
        merged_note = f" <span style='color:#9C27B0;'>(合并 {len(merged_sources)} 次后续微调)</span>"

    if action == "CONTROL_MOVE":
        stroke = op.get("stroke", "?")
        control = op.get("control", "?")
        source = op.get("source_action")
        source_note = ""
        if source == "T_ATTACH":
            source_note = f"；来源：重复 T_ATTACH 到 stroke {op.get('source_host', '?')}"
        elif source == "SNAP":
            source_note = f"；来源：重复 SNAP 到 stroke {op.get('source_host_stroke', '?')} {op.get('source_host_endpoint', '?')}"
        return (
            "<div style='background:#FFF8E1; border:1px solid #FFE082; padding:5px; margin-bottom:5px;'>"
            f"<b style='color:#E65100;'>[本步移动]</b> "
            f"stroke <b>{stroke}</b> 的 <b>{control}</b>："
            f"{_fmt_coord(before)} → {_fmt_coord(after)}{merged_note}{source_note}"
            "</div>"
        )

    if action == "T_ATTACH":
        return (
            "<div style='background:#E8F5E9; border:1px solid #A5D6A7; padding:5px; margin-bottom:5px;'>"
            f"<b style='color:#2E7D32;'>[建立 T 连接]</b> "
            f"guest stroke <b>{op.get('guest', '?')}</b> 的 <b>{op.get('guest_endpoint', '?')}</b> "
            f"搭到 host stroke <b>{op.get('host', '?')}</b>，host_t={float(op.get('host_t', 0.0)):.3f}："
            f"{_fmt_coord(before)} → {_fmt_coord(after)}{merged_note}"
            "</div>"
        )

    if action == "SNAP":
        return (
            "<div style='background:#E3F2FD; border:1px solid #90CAF9; padding:5px; margin-bottom:5px;'>"
            f"<b style='color:#1565C0;'>[建立端点吸附]</b> "
            f"stroke <b>{op.get('stroke', '?')}</b> 的 <b>{op.get('endpoint', '?')}</b> "
            f"吸附到 stroke <b>{op.get('host_stroke', '?')}</b> 的 <b>{op.get('host_endpoint', '?')}</b>："
            f"{_fmt_coord(before)} → {_fmt_coord(after)}{merged_note}"
            "</div>"
        )

    return (
        "<div style='background:#F5F5F5; border:1px solid #DDD; padding:5px; margin-bottom:5px;'>"
        f"<b>[{action}]</b> {_fmt_coord(before)} → {_fmt_coord(after)}{merged_note}"
        "</div>"
    )


# ── edit_history 帧重建 ──────────────────────────────────────────────────────
def _pidx(pname: str) -> int:
    """'P0' -> 0, 'P3' -> 3"""
    try:
        return int(pname[1:])
    except (ValueError, IndexError):
        return 0


def replay_frames(phase1_strokes: list, edit_history: list):
    """
    从 Phase 1 最终形态（annotations/{字体名}.json 中的 strokes）出发，
    正向逐步应用 edit_history 中的每个操作，生成每帧的 strokes 快照序列。

    返回：
        [(strokes_snapshot, action_name, action_op), ...]
        第 0 帧 action_op 为 None，后续每帧对应一次 edit_history 操作。
    """
    # 以 Phase 1 最终形态为起点（深拷贝，防止修改原数据）
    strokes = copy.deepcopy(phase1_strokes)
    sid_map = {int(s["bezier_id"]): s for s in strokes}

    def _apply_op(op):
        """正向应用一个 edit_history 操作（before → after）"""
        kind  = op.get("action", "")
        after = op.get("after")
        if after is None:
            return

        if kind == "CONTROL_MOVE":
            bid  = int(op.get("stroke", 0))
            pidx = _pidx(str(op.get("control", "P0")))
            if bid in sid_map:
                sid_map[bid]["mother_bezier"][pidx] = list(after)

        elif kind == "SNAP":
            bid  = int(op.get("stroke", 0))
            pidx = _pidx(str(op.get("endpoint", "P0")))
            if bid in sid_map:
                sid_map[bid]["mother_bezier"][pidx] = list(after)

        elif kind == "T_ATTACH":
            bid  = int(op.get("guest", 0))
            pidx = _pidx(str(op.get("guest_endpoint", "P0")))
            if bid in sid_map:
                sid_map[bid]["mother_bezier"][pidx] = list(after)

    # 第 0 帧：Phase 1 最终形态（Phase 2 标注的起点）
    frames = [(copy.deepcopy(strokes), "Initial (Phase 1 Final)", None)]

    # 正向逐步 replay
    for op in edit_history:
        _apply_op(op)
        frames.append((copy.deepcopy(strokes), op.get("action", "?"), copy.deepcopy(op)))

    return frames


# ── 渲染引擎 ────────────────────────────────────────────────────────────────
class StrokeRenderer:
    def render_strokes(self, strokes: list, size=(2.5, 2.5), highlight_op: dict = None) -> QPixmap:
        """
        将 strokes（含 mother_bezier）渲染为彩色贝塞尔曲线图，
        每条笔画用 tab20 色板着色，起点用圆点标记。
        若 highlight_op 含 before/after，则额外绘制红色移动箭头。
        """
        fig = plt.figure(figsize=size, dpi=80)
        fig.patch.set_facecolor('#FFFFFF')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#FFFFFF')
        ax.set_xlim(0, CANVAS_SIZE)
        ax.set_ylim(CANVAS_SIZE, 0)   # y 轴翻转与图像坐标一致
        ax.set_aspect('equal')
        ax.axis('off')

        for s in strokes:
            bid = int(s["bezier_id"])
            P = np.array(s["mother_bezier"], dtype=float)
            if P.shape != (4, 2):
                continue
            color = _tab20_rgb(bid)
            ts = np.linspace(0, 1, 60)[:, None]
            curve = cubic_bezier_np(P, ts)
            ax.plot(curve[:, 0], curve[:, 1], c=color, lw=3.0, alpha=0.9)
            # 起点圆点
            ax.plot(P[0, 0], P[0, 1], 'o', color=color, markersize=5, zorder=5)
            # 中点标注 bezier_id
            mid = curve[30]
            ax.text(mid[0], mid[1], str(bid), color=color, fontsize=10,
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1))

        self._draw_move_highlight(ax, highlight_op)

        plt.tight_layout(pad=0)
        return self._fig_to_pixmap(fig)

    def _draw_move_highlight(self, ax, op: dict):
        """在图上标识当前操作的 before → after 移动路径。"""
        if not isinstance(op, dict):
            return
        before = op.get("before")
        after = op.get("after")
        if not isinstance(before, (list, tuple)) or not isinstance(after, (list, tuple)):
            return
        if len(before) < 2 or len(after) < 2:
            return
        try:
            bx, by = float(before[0]), float(before[1])
            axx, ayy = float(after[0]), float(after[1])
        except Exception:
            return

        # 移动箭头：红色虚线 + 起止点标记 + after 光圈
        ax.annotate(
            "",
            xy=(axx, ayy), xytext=(bx, by),
            arrowprops=dict(
                arrowstyle="->",
                color="#D32F2F",
                lw=2.6,
                linestyle="--",
                shrinkA=0,
                shrinkB=0,
                mutation_scale=16,
            ),
            zorder=20,
        )
        ax.scatter([bx], [by], s=42, marker="x", c="#757575", linewidths=2.0, zorder=21)
        ax.scatter([axx], [ayy], s=70, marker="o", facecolors="none", edgecolors="#D32F2F", linewidths=2.5, zorder=22)
        ax.text(bx + 4, by + 4, "before", color="#616161", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1), zorder=23)
        ax.text(axx + 4, ayy - 4, "after", color="#D32F2F", fontsize=8, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1), zorder=23)

    def render_thumbnail_from_strokes(self, strokes: list, size=(1.5, 1.5)) -> QPixmap:
        """微缩图：纯黑白笔画轮廓"""
        return self.render_strokes(strokes, size=size)

    @staticmethod
    def _fig_to_pixmap(fig) -> QPixmap:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        qimg = QImage()
        qimg.loadFromData(buf.getvalue())
        return QPixmap.fromImage(qimg)


# ── 字体字符列表数据加载 ─────────────────────────────────────────────────────
def load_topo_index() -> dict:
    """
    扫描 annotations_topo/ 目录，返回：
    {
        "字体名": {
            "path": "/abs/path/to/字体名_topo.json",
            "chars": ["U+XXXX", ...]     # 已标注字符列表
        },
        ...
    }
    """
    result = {}
    if not os.path.isdir(TOPO_ANNO_DIR):
        return result
    for fp in glob.glob(os.path.join(TOPO_ANNO_DIR, "*_topo.json")):
        fn = os.path.basename(fp)
        font_name = fn[:-len("_topo.json")]
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            chars = [k for k, v in data.items() if isinstance(v, dict) and "strokes" in v]
        except Exception:
            chars = []
        result[font_name] = {"path": fp, "chars": sorted(chars)}
    return result


# ── Phase2 PreviewWorkspace ──────────────────────────────────────────────────
class Phase2PreviewWorkspace(QWidget):
    def __init__(self, font_name: str, topo_path: str):
        super().__init__()
        self.font_name = font_name
        self.topo_path = topo_path
        self.topo_data: dict = {}        # {hex_key: bundle}     来自 annotations_topo/
        self.anno_data: dict = {}        # {hex_key: [strokes]}  来自 annotations/（Phase 1 最终形态）
        self.renderer = StrokeRenderer()
        self.thumbnail_cache: dict = {}  # {hex_key: QPixmap}

        self._load_topo_data()
        self._init_ui()

    def _load_topo_data(self):
        try:
            with open(self.topo_path, 'r', encoding='utf-8') as f:
                self.topo_data = json.load(f)
        except Exception as e:
            print(f"[Phase2Preview] Error loading {self.topo_path}: {e}")
            self.topo_data = {}

        # 同时加载 Phase 1 最终形态（初始状态来源）
        anno_path = os.path.join(ANNO_DIR, f"{self.font_name}.json")
        if os.path.exists(anno_path):
            try:
                with open(anno_path, 'r', encoding='utf-8') as f:
                    self.anno_data = json.load(f)
            except Exception as e:
                print(f"[Phase2Preview] Error loading anno {anno_path}: {e}")
                self.anno_data = {}
        else:
            self.anno_data = {}

    def _init_ui(self):
        layout = QVBoxLayout(self)
        self.stacked = QStackedWidget()

        # 页面 0：画廊
        self.page_gallery = QWidget()
        gal_layout = QVBoxLayout(self.page_gallery)
        title = QLabel(f"✅  Phase 2 Topo Annotations for: {self.font_name}  (Cleaned History)")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        gal_layout.addWidget(title)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        container = QWidget(); self.grid = QGridLayout(container)
        scroll.setWidget(container)
        gal_layout.addWidget(scroll)

        # 页面 1：时光机详情
        self.page_detail = QWidget()
        self.detail_layout = QVBoxLayout(self.page_detail)

        self.stacked.addWidget(self.page_gallery)
        self.stacked.addWidget(self.page_detail)
        layout.addWidget(self.stacked)

        self._load_gallery()

    def _load_gallery(self):
        cols = 6
        keys = sorted(self.topo_data.keys())
        for idx, hex_key in enumerate(keys):
            bundle = self.topo_data[hex_key]
            char_str = bundle.get("glyph_info", {}).get("char", "?")

            card = QFrame()
            card.setStyleSheet(
                "background-color: #FFFFFF; border: 2px solid #E0E0E0; border-radius: 8px;"
            )
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 10, 10, 10)

            # 缩略图（用 strokes 的最终状态渲染）
            if hex_key not in self.thumbnail_cache:
                strokes = bundle.get("strokes", [])
                self.thumbnail_cache[hex_key] = self.renderer.render_thumbnail_from_strokes(
                    strokes, size=(1.5, 1.5)
                )
            img_label = QLabel()
            img_label.setPixmap(self.thumbnail_cache[hex_key])
            img_label.setAlignment(Qt.AlignCenter)

            title_lbl = QLabel(f"'{char_str}'\n{hex_key}")
            title_lbl.setAlignment(Qt.AlignCenter)
            title_lbl.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: #424242; border: none;"
            )

            stroke_cnt = len(bundle.get("strokes", []))
            edit_cnt   = len(bundle.get("edit_history", []))
            clean_cnt = len(clean_edit_history(bundle.get("edit_history", [])))
            info_lbl   = QLabel(f"Strokes:{stroke_cnt}  Edits:{edit_cnt} → {clean_cnt}")
            info_lbl.setAlignment(Qt.AlignCenter)
            info_lbl.setStyleSheet("font-size: 11px; color: #757575; border: none;")

            btn_view = QPushButton("🔍 View Audit")
            btn_view.setCursor(Qt.PointingHandCursor)
            btn_view.setStyleSheet(
                "background-color: #E91E63; color: white; padding: 6px; "
                "border-radius: 4px; font-weight: bold;"
            )
            btn_view.clicked.connect(lambda _, hk=hex_key: self._open_timeline(hk))

            card_layout.addWidget(img_label)
            card_layout.addWidget(title_lbl)
            card_layout.addWidget(info_lbl)
            card_layout.addWidget(btn_view)
            self.grid.addWidget(card, idx // cols, idx % cols)

    def _open_timeline(self, hex_key: str):
        # 清理旧详情页
        while self.detail_layout.count():
            item = self.detail_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        bundle   = self.topo_data.get(hex_key, {})
        char_str = bundle.get("glyph_info", {}).get("char", "?")
        raw_edit_history = bundle.get("edit_history", [])
        edit_history = clean_edit_history(raw_edit_history)

        # 用 annotations/（Phase 1 最终形态）作为 Phase 2 的初始状态
        # 若 Phase 1 数据不存在，则 fallback 为 topo 里的最终 strokes（仅展示终态）
        phase1_strokes = self.anno_data.get(hex_key)
        if not phase1_strokes:
            phase1_strokes = bundle.get("strokes", [])

        # ── 顶部导航栏 ──────────────────────────────────────────────────────
        top_bar = QHBoxLayout()
        btn_back = QPushButton("🔙 Back")
        btn_back.setFixedWidth(100)
        btn_back.setStyleSheet(
            "padding: 8px; background-color: #2196F3; color: white; "
            "font-weight: bold; border-radius: 4px;"
        )
        btn_back.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        top_bar.addWidget(btn_back)

        lbl_title = QLabel(f"Phase 2 Audit Trail (Cleaned): '{char_str}'  ({hex_key})")
        lbl_title.setStyleSheet("font-size: 20px; font-weight: bold;")
        top_bar.addWidget(lbl_title)
        top_bar.addStretch()
        self.detail_layout.addLayout(top_bar)

        line = QFrame(); line.setFrameShape(QFrame.HLine)
        self.detail_layout.addWidget(line)

        if not phase1_strokes:
            self.detail_layout.addWidget(
                QLabel("⚠️  No strokes data found in this bundle.")
            )
            self.detail_layout.addStretch()
            self.stacked.setCurrentIndex(1)
            return

        # ── 构建帧序列（从 Phase 1 最终形态正向 replay）────────────────────────
        frames = replay_frames(phase1_strokes, edit_history)

        # ── 区域 A：固定参考图（初始状态） ─────────────────────────────────
        ref_layout = QHBoxLayout()
        init_strokes, _, _ = frames[0]
        ref_label = QLabel(
            "🎯 <b>Phase 1 Final State (Phase 2 Input)</b><br>"
            "<span style='color:#757575; font-size:11px;'>From annotations/ — before any Phase 2 edit</span>"
        )
        ref_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        img_ref = QLabel()
        img_ref.setPixmap(self.renderer.render_strokes(init_strokes, size=(3.5, 3.5)))
        img_ref.setStyleSheet("border: 2px dashed #9E9E9E; background-color: white;")

        ref_layout.addWidget(img_ref)
        ref_layout.addWidget(ref_label)
        ref_layout.addStretch()
        self.detail_layout.addLayout(ref_layout)

        # ── 区域 B：横向步骤时光机 ─────────────────────────────────────────
        lbl_section = QLabel("⏳ <b>Step-by-Step Transformation</b>")
        lbl_section.setStyleSheet("margin-top: 12px; font-size: 15px;")
        self.detail_layout.addWidget(lbl_section)

        scroll_time = QScrollArea()
        scroll_time.setWidgetResizable(True)
        scroll_time.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_time.setStyleSheet("background-color: #FAFAFA;")

        time_container = QWidget()
        time_layout = QHBoxLayout(time_container)
        time_layout.setAlignment(Qt.AlignLeft)
        time_layout.setSpacing(12)

        for i, (frame_strokes, action_name, action_op) in enumerate(frames):
            # 操作标签 + 箭头
            if i > 0:
                arrow_wrap = QVBoxLayout()
                arrow_wrap.setAlignment(Qt.AlignCenter)

                lbl_act = QLabel(action_name)
                lbl_act.setAlignment(Qt.AlignCenter)
                lbl_act.setWordWrap(True)
                lbl_act.setMaximumWidth(90)
                lbl_act.setStyleSheet(
                    "font-size: 12px; font-weight: bold; color: #E91E63; "
                    "background: #FFCDD2; padding: 3px; border-radius: 4px;"
                )

                lbl_arrow = QLabel("➔")
                lbl_arrow.setAlignment(Qt.AlignCenter)
                lbl_arrow.setStyleSheet(
                    "font-size: 30px; color: #BDBDBD; font-weight: bold;"
                )

                arrow_wrap.addWidget(lbl_act)
                arrow_wrap.addWidget(lbl_arrow)
                time_layout.addLayout(arrow_wrap)

            # 状态卡片
            card = QFrame()
            card.setStyleSheet(
                "background-color: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 8px;"
            )
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 8, 8, 8)
            card_layout.setSpacing(4)

            # 1. 彩色贝塞尔笔画图
            lbl_img = QLabel()
            pix = self.renderer.render_strokes(frame_strokes, size=(2.5, 2.5), highlight_op=action_op)
            lbl_img.setPixmap(pix)
            lbl_img.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(lbl_img)

            # 2. 笔画数 + 帧类型标注
            stroke_cnt = len(frame_strokes)
            if i == 0:
                info_txt = f"🚀 Initial ({stroke_cnt} strokes)"
            elif i == len(frames) - 1:
                info_txt = f"🏁 Final ({stroke_cnt} strokes)"
            else:
                info_txt = f"Step {i} ({stroke_cnt} strokes)"
            lbl_info = QLabel(info_txt)
            lbl_info.setAlignment(Qt.AlignCenter)
            lbl_info.setStyleSheet(
                "font-size: 12px; font-weight: bold; color: #424242; "
                "padding-top: 3px; border: none;"
            )
            card_layout.addWidget(lbl_info)

            # 3. 拓扑文本（仿 topo_editor_workspace.py 的 update_topology_text）
            topo_box = QTextBrowser()
            topo_box.setStyleSheet(
                "background-color: #F8F9FA; border: 1px solid #CCC; "
                "padding: 6px; font-size: 12px;"
            )
            topo_box.setMinimumWidth(260)
            topo_box.setMaximumWidth(320)
            topo_box.setMaximumHeight(185)
            topo_html = describe_action_html(action_op) + compute_topo_html(frame_strokes)
            topo_box.setHtml(topo_html)
            card_layout.addWidget(topo_box)

            time_layout.addWidget(card)

        scroll_time.setWidget(time_container)
        self.detail_layout.addWidget(scroll_time, 1)
        self.stacked.setCurrentIndex(1)


# ── 主窗口 ───────────────────────────────────────────────────────────────────
class PreviewAppMain(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phase 2 Topo Annotation Auditor (Cleaned History)")
        self.setGeometry(50, 50, 1500, 850)
        self._topo_index: dict = {}

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # 左侧字体列表
        sidebar = QWidget()
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(10, 10, 10, 10)

        lbl_anno = QLabel("✅  Annotated Fonts (Phase 2)")
        lbl_anno.setStyleSheet("font-size: 14px; font-weight: bold; color: #4CAF50;")
        sb_layout.addWidget(lbl_anno)

        self.list_annotated = QListWidget()
        self.list_annotated.itemClicked.connect(self._on_font_selected)
        self.list_annotated.setStyleSheet(
            "background-color: #F1F8E9; border: 1px solid #C8E6C9; border-radius: 4px;"
        )
        sb_layout.addWidget(self.list_annotated)

        lbl_empty = QLabel("⏳  No Topo Data")
        lbl_empty.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #FF9800; margin-top: 10px;"
        )
        sb_layout.addWidget(lbl_empty)

        self.list_unannotated = QListWidget()
        self.list_unannotated.setStyleSheet(
            "background-color: #FFF3E0; border: 1px solid #FFE0B2; border-radius: 4px;"
        )
        sb_layout.addWidget(self.list_unannotated)

        # 右侧工作区
        self.workspace_stack = QStackedWidget()
        self.workspace_stack.addWidget(
            QLabel("👈  Select a font from the left panel to begin audit...")
        )

        splitter.addWidget(sidebar)
        splitter.addWidget(self.workspace_stack)
        splitter.setSizes([260, 1240])

        self.load_font_list()

    def load_font_list(self):
        self.list_annotated.clear()
        self.list_unannotated.clear()
        self._topo_index = load_topo_index()

        # 已有 Phase 2 标注的字体
        for font_name, info in sorted(self._topo_index.items()):
            if info["chars"]:
                self.list_annotated.addItem(font_name)
            else:
                self.list_unannotated.addItem(font_name)

        # 扫描 FONTS_DIR 中存在但没有 topo 数据的字体
        if os.path.isdir(FONTS_DIR):
            topo_fonts = set(self._topo_index.keys())
            for fn in os.listdir(FONTS_DIR):
                if fn.lower().endswith(('.ttf', '.otf')):
                    name = os.path.splitext(fn)[0]
                    if name not in topo_fonts:
                        self.list_unannotated.addItem(name)

    def _on_font_selected(self, item):
        font_name = item.text()
        info = self._topo_index.get(font_name)
        if not info or not info["chars"]:
            return

        # 移除旧工作区
        if self.workspace_stack.count() > 1:
            old = self.workspace_stack.widget(1)
            self.workspace_stack.removeWidget(old)
            old.deleteLater()

        ws = Phase2PreviewWorkspace(font_name, info["path"])
        self.workspace_stack.addWidget(ws)
        self.workspace_stack.setCurrentIndex(1)


# ── 入口 ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app = QApplication(sys.argv)
    if HAS_QDARK:
        qdarktheme.setup_theme("light", custom_colors={"primary": "#E91E63"})
    window = PreviewAppMain()
    window.show()
    sys.exit(app.exec_())
