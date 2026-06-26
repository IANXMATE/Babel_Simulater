import os
import sys
import json
import random
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout,
    QVBoxLayout, QListWidget, QLabel, QScrollArea,
    QSplitter, QPushButton, QTextEdit
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# ==========================================
# ⚙️ 配置读取路径
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "clustered_results.json")

SAMPLE_PER_CLUSTER = 8
SAMPLE_N = 50


# ==========================================
# 📐 1. 与聚合代码一致的函数化重采样
# ==========================================
def get_bezier_point(pts, t):
    mt = 1 - t
    return (
        (mt ** 3) * pts[0]
        + 3 * (mt ** 2) * t * pts[1]
        + 3 * mt * (t ** 2) * pts[2]
        + (t ** 3) * pts[3]
    )


def normalize_and_sample_function(mother_bezier, N=SAMPLE_N):
    """
    与 clustered_results 聚合代码保持一致：

    1. 平移到 p0
    2. 找离 p0 最远的控制点方向作为主方向
    3. 旋转到主方向接近 +X
    4. 尺度归一化
    5. 曲线采样
    6. 强制 x 单调
    7. 等 x 轴重采样，得到 y=f(x)
    """
    pts = np.array(mother_bezier, dtype=float)

    if np.isnan(pts).any() or np.isinf(pts).any():
        return None

    p0 = pts[0]
    pts_t = pts - p0

    dists_from_p0 = np.linalg.norm(pts_t, axis=1)
    max_idx = np.argmax(dists_from_p0)

    vec = pts_t[max_idx]
    L = dists_from_p0[max_idx]

    if L < 1e-5:
        return None

    cos_theta = vec[0] / L
    sin_theta = vec[1] / L

    R = np.array([
        [cos_theta, sin_theta],
        [-sin_theta, cos_theta]
    ])

    pts_r = pts_t @ R.T
    pts_norm = pts_r / L

    ts_dense = np.linspace(0, 1, 300)[:, None]
    curve_dense = get_bezier_point(pts_norm, ts_dense)

    x_dense = np.maximum.accumulate(curve_dense[:, 0])
    y_dense = curve_dense[:, 1]

    if x_dense[-1] < 1e-5:
        return None

    x_dense = x_dense / x_dense[-1]

    target_x = np.linspace(0, 1.0, N)
    y_eq = np.interp(target_x, x_dense, y_dense)

    return y_eq


def get_4_isomorphisms_y_only(Y):
    """
    与聚合代码一致的 4 种等价形态。

    0: 原方向
    1: 起终点倒转 + 上下翻转
    2: X 轴翻转
    3: 中心对称 / 倒序
    """
    Y0 = Y.copy()
    Y1 = -Y[::-1]
    Y2 = -Y
    Y3 = Y[::-1]
    return [Y0, Y1, Y2, Y3]


def calculate_l1_y(Y_A, Y_B):
    return float(np.mean(np.abs(Y_A - Y_B)))


def get_best_variant_to_ref(Y, Y_ref):
    variants = get_4_isomorphisms_y_only(Y)
    dists = [calculate_l1_y(Y_ref, v) for v in variants]
    best_id = int(np.argmin(dists))
    return best_id, variants[best_id], float(dists[best_id])


def safe_get(item, key, default=""):
    v = item.get(key, default)
    if v is None:
        return default
    return v


def make_fallback_glyph_uid(item):
    source_file = safe_get(item, "source_file", "UNKNOWN_SOURCE")
    hex_key = safe_get(item, "hex_key", "UNKNOWN_HEX")
    return f"{source_file}::{hex_key}"


def make_fallback_stroke_uid(item):
    glyph_uid = item.get("glyph_uid") or make_fallback_glyph_uid(item)
    bezier_id = safe_get(item, "bezier_id", "UNKNOWN_BID")
    return f"{glyph_uid}::{bezier_id}"


# ==========================================
# 🎨 2. GUI 主程序
# ==========================================
class ClusterVisualizerGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Stroke Shape Cluster Viewer - Source Aware")
        self.setGeometry(100, 100, 1450, 900)

        self.clusters = {}
        self.cluster_order = []
        self.current_cid = None
        self.cluster_ref_y = {}

        self.init_ui()
        self.load_data()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)

        # ------------------------------
        # 左侧 panel
        # ------------------------------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.lbl_info = QLabel("Loading data...")
        self.lbl_info.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(self.lbl_info)

        self.list_widget = QListWidget()
        self.list_widget.itemSelectionChanged.connect(self.on_cluster_selected)
        left_layout.addWidget(self.list_widget)

        self.meta_box = QTextEdit()
        self.meta_box.setReadOnly(True)
        self.meta_box.setMinimumHeight(220)
        self.meta_box.setStyleSheet("font-family: Menlo, Consolas, monospace; font-size: 11px;")
        left_layout.addWidget(self.meta_box)

        # ------------------------------
        # 右侧 panel
        # ------------------------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        top_bar = QHBoxLayout()

        self.lbl_cluster_title = QLabel("👈 请在左侧选择一个聚类桶")
        self.lbl_cluster_title.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #333;"
        )

        self.btn_resample = QPushButton("🎲 重新抽样")
        self.btn_resample.setStyleSheet(
            "background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 6px;"
        )
        self.btn_resample.clicked.connect(self.render_current_cluster)
        self.btn_resample.setDisabled(True)

        top_bar.addWidget(self.lbl_cluster_title)
        top_bar.addStretch()
        top_bar.addWidget(self.btn_resample)

        right_layout.addLayout(top_bar)

        self.fig = plt.figure(figsize=(10, 12))
        self.canvas = FigureCanvas(self.fig)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.canvas)

        right_layout.addWidget(scroll_area)

        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(right_panel)
        self.splitter.setSizes([360, 1090])

    def load_data(self):
        if not os.path.exists(DATA_FILE):
            self.lbl_info.setText(f"❌ 找不到文件:\n{DATA_FILE}")
            return

        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 兼容两种格式：
        # 1. clustered_results.json 是 list
        # 2. 未来可能被包成 {"items": [...]}
        if isinstance(data, dict):
            if "items" in data:
                data = data["items"]
            elif "strokes" in data:
                data = data["strokes"]
            else:
                self.lbl_info.setText("❌ JSON 格式不支持：需要 list 或包含 items/strokes")
                return

        missing_source = 0
        missing_uid = 0
        duplicate_stroke_uid_count = 0
        seen_stroke_uids = set()

        for item in data:
            cid = item.get("cluster_id", -1)

            if cid == -1:
                continue

            cid = int(cid)

            if "source_file" not in item:
                missing_source += 1

            if "glyph_uid" not in item:
                item["glyph_uid"] = make_fallback_glyph_uid(item)
                missing_uid += 1

            if "stroke_uid" not in item:
                item["stroke_uid"] = make_fallback_stroke_uid(item)
                missing_uid += 1

            if item["stroke_uid"] in seen_stroke_uids:
                duplicate_stroke_uid_count += 1
            else:
                seen_stroke_uids.add(item["stroke_uid"])

            if cid not in self.clusters:
                self.clusters[cid] = []

            self.clusters[cid].append(item)

        self.cluster_order = sorted(
            self.clusters.keys(),
            key=lambda c: (-len(self.clusters[c]), c)
        )

        self.lbl_info.setText(
            f"✅ 找到 {len(self.clusters)} 个有效分桶\n"
            f"📦 有效 strokes: {sum(len(v) for v in self.clusters.values())}\n"
            f"⚠️ missing source: {missing_source}\n"
            f"⚠️ uid fallback: {missing_uid}\n"
            f"⚠️ duplicate stroke_uid: {duplicate_stroke_uid_count}"
        )

        self.list_widget.clear()

        for cid in self.cluster_order:
            items = self.clusters[cid]
            count = len(items)
            source_count = len(set(x.get("source_file", "UNKNOWN") for x in items))
            glyph_count = len(set(x.get("glyph_uid", "") for x in items))

            self.list_widget.addItem(
                f"Cluster {cid}  | {count} strokes | {source_count} files | {glyph_count} glyphs"
            )

    def on_cluster_selected(self):
        selected_items = self.list_widget.selectedItems()

        if not selected_items:
            return

        text = selected_items[0].text()
        self.current_cid = int(text.split()[1])

        self.btn_resample.setDisabled(False)
        self.render_current_cluster()

    def summarize_cluster(self, cid):
        items = self.clusters[cid]

        source_hist = Counter(x.get("source_file", "UNKNOWN") for x in items)
        hex_hist = Counter(x.get("hex_key", "UNKNOWN") for x in items)
        width_hist = Counter(str(x.get("width_mean", "NA")) for x in items)

        top_sources = source_hist.most_common(8)
        top_hex = hex_hist.most_common(8)

        lines = []
        lines.append(f"Cluster ID: {cid}")
        lines.append(f"Total strokes: {len(items)}")
        lines.append(f"Source files: {len(source_hist)}")
        lines.append(f"Glyph count: {len(set(x.get('glyph_uid', '') for x in items))}")
        lines.append("")
        lines.append("Top source files:")
        for k, v in top_sources:
            lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append("Top hex keys:")
        for k, v in top_hex:
            lines.append(f"  {k}: {v}")

        # width_mean 是连续值，通常不太适合直接 hist，这里只显示有没有
        has_width = sum(1 for x in items if x.get("width_mean", None) is not None)
        lines.append("")
        lines.append(f"Items with width_mean: {has_width}/{len(items)}")

        return "\n".join(lines)

    def pick_reference_y(self, cid):
        """
        给每个 cluster 选一个 reference。
        简单做法：第一个可 normalize 的 stroke。
        """
        if cid in self.cluster_ref_y:
            return self.cluster_ref_y[cid]

        for item in self.clusters[cid]:
            y = normalize_and_sample_function(item["mother_bezier"], N=SAMPLE_N)
            if y is not None:
                self.cluster_ref_y[cid] = y
                return y

        return None

    def render_current_cluster(self):
        if self.current_cid is None:
            return

        cid = self.current_cid
        strokes_info = self.clusters[cid]

        self.lbl_cluster_title.setText(
            f"🔍 Cluster ID: {cid} | 样本总数: {len(strokes_info)}"
        )

        self.meta_box.setPlainText(self.summarize_cluster(cid))

        display_strokes = random.sample(
            strokes_info,
            min(SAMPLE_PER_CLUSTER, len(strokes_info))
        )

        y_ref = self.pick_reference_y(cid)

        if y_ref is None:
            self.fig.clf()
            self.canvas.draw()
            return

        rows = []

        for item in display_strokes:
            y = normalize_and_sample_function(item["mother_bezier"], N=SAMPLE_N)

            if y is None:
                continue

            variant_id, y_standard, dist_to_ref = get_best_variant_to_ref(y, y_ref)

            rows.append({
                "item": item,
                "y": y,
                "variant_id": variant_id,
                "y_standard": y_standard,
                "dist_to_ref": dist_to_ref,
            })

        if not rows:
            self.fig.clf()
            self.canvas.draw()
            return

        self.fig.clf()

        n_samples = len(rows)

        # 三列：
        # 1. raw normalized y=f(x)
        # 2. standardized to cluster ref
        # 3. original px control polyline / curve preview
        axs = self.fig.subplots(n_samples, 3)

        if n_samples == 1:
            axs = np.array([axs])

        x = np.linspace(0.0, 1.0, SAMPLE_N)

        type_texts = {
            0: "Original",
            1: "Reverse + flip",
            2: "X-flip",
            3: "Reverse",
        }

        for i, row in enumerate(rows):
            item = row["item"]
            y = row["y"]
            y_standard = row["y_standard"]
            variant_id = row["variant_id"]
            dist_to_ref = row["dist_to_ref"]

            source_file = item.get("source_file", "UNKNOWN_SOURCE")
            hex_key = item.get("hex_key", "UNKNOWN_HEX")
            char = item.get("char", "")
            bezier_id = item.get("bezier_id", "UNKNOWN_BID")
            stroke_uid = item.get("stroke_uid", "")

            short_source = source_file
            if len(short_source) > 38:
                short_source = short_source[:35] + "..."

            title_base = (
                f"{short_source}\n"
                f"{hex_key} {char} | bid={bezier_id}"
            )

            # ------------------------------
            # 1. 原始归一化 y=f(x)
            # ------------------------------
            ax_orig = axs[i, 0]
            ax_orig.plot(x, y, linewidth=2.5)
            ax_orig.scatter([x[0]], [y[0]], s=45)
            ax_orig.set_title(
                f"Raw normalized\n"
                f"variant={variant_id} ({type_texts.get(variant_id, '?')})",
                fontsize=9
            )
            ax_orig.axis("equal")
            ax_orig.grid(True, linestyle="--", alpha=0.45)
            ax_orig.set_xticks([])
            ax_orig.set_yticks([])

            # ------------------------------
            # 2. 标准化到 reference 后
            # ------------------------------
            ax_std = axs[i, 1]
            ax_std.plot(x, y_standard, linewidth=2.5)
            ax_std.scatter([x[0]], [y_standard[0]], s=45)

            # 叠加 reference，方便看桶内是否一致
            ax_std.plot(x, y_ref, linewidth=1.0, linestyle="--", alpha=0.55)

            ax_std.set_title(
                f"Standardized\n"
                f"dist={dist_to_ref:.4f}",
                fontsize=9
            )
            ax_std.axis("equal")
            ax_std.grid(True, linestyle="--", alpha=0.45)
            ax_std.set_xticks([])
            ax_std.set_yticks([])

            # ------------------------------
            # 3. 原始像素空间 Bézier 预览
            # ------------------------------
            ax_px = axs[i, 2]

            pts = np.array(item["mother_bezier"], dtype=float)
            ts = np.linspace(0, 1, 80)[:, None]
            curve = get_bezier_point(pts, ts)

            ax_px.plot(curve[:, 0], curve[:, 1], linewidth=2.5)
            ax_px.plot(pts[:, 0], pts[:, 1], linestyle="--", marker="o", linewidth=1.0)
            ax_px.scatter([pts[0, 0]], [pts[0, 1]], s=45)

            ax_px.set_title(title_base, fontsize=8)
            ax_px.axis("equal")
            ax_px.grid(True, linestyle="--", alpha=0.45)
            ax_px.set_xticks([])
            ax_px.set_yticks([])

        self.fig.tight_layout()
        self.fig.set_size_inches(12, 2.8 * n_samples)
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = ClusterVisualizerGUI()
    window.show()

    sys.exit(app.exec_())