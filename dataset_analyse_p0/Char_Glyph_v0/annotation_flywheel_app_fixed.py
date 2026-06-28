# -*- coding: utf-8 -*-
"""
annotation_flywheel_app.py

数据飞轮式 PCG 字体标注器。

功能：
1. 从已有候选 JSON 中抽取 N*N 个候选，显示纯黑带宽度缩略图。
2. 每个候选旁有勾选框：勾选=好样本；未勾选=坏样本。
3. 提供 Good / Bad 样本池管理，可预览并从池中剔除。
4. 提供 Generate Fresh Candidates 按钮，自动串行运行：
      layout_aware_graph_sampler.py
      stroke_primitive_sampler.py
      constraint_solver.py
      score_solved_glyphs.py
   并自动重新加载新候选池。
5. 每轮生成前自动修改生成脚本中的 RANDOM_SEED / NUM_SAMPLES 常量，使候选池刷新。
6. 保存 JSON 内容为兼容人工标注脚本的 topo 格式：
      {
        "U+XXXXX": {
          "glyph_info": {...},
          "strokes": [...],
          "topology_events": [...],
          "cycles": [...],
          "edit_history": [...]
        }
      }
7. 自动按 80MB 分文件，避免 GitHub 100MB 单文件限制。
8. 预留 aesthetic / relation / primitive compatibility 模型接口。

放置位置建议：
    Babel_Simulater/dataset_analyse_p0/Char_Glyph/annotation_tools/annotation_flywheel_app.py

因为脚本位于 Char_Glyph 下的子文件夹，所以通过 ../ 回到 Char_Glyph 目录。
"""

import os
import sys
import re
import json
import math
import copy
import time
import random
import hashlib
import traceback
import subprocess

import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea,
    QLabel, QPushButton, QCheckBox, QSpinBox, QMessageBox,
    QFrame, QGroupBox, QComboBox, QTextEdit, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QPainterPath
from PyQt5.QtCore import Qt, QThread, pyqtSignal


# =========================================================
# ⚙️ 路径配置
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 本脚本放在 Char_Glyph 下的子文件夹中，所以 ../ 是 Char_Glyph
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# 生成链路脚本
LAYOUT_SAMPLER_SCRIPT = os.path.join(PROJECT_DIR, "layout_aware_graph_sampler.py")
PRIMITIVE_SAMPLER_SCRIPT = os.path.join(PROJECT_DIR, "stroke_primitive_sampler.py")
SOLVER_SCRIPT = os.path.join(PROJECT_DIR, "constraint_solver.py")
SCORER_SCRIPT = os.path.join(PROJECT_DIR, "score_solved_glyphs.py")

# 生成链路输出
SCORED_FILE = os.path.join(PROJECT_DIR, "gnn_scored_solved_glyphs.json")
SOLVED_FILE = os.path.join(PROJECT_DIR, "solved_glyph_candidates.json")

# 飞轮状态
FLYWHEEL_DIR = os.path.join(SCRIPT_DIR, "annotation_flywheel")
THUMB_DIR = os.path.join(FLYWHEEL_DIR, "thumb_cache")
STATE_FILE = os.path.join(FLYWHEEL_DIR, "pcg_annotation_state.json")
MANIFEST_FILE = os.path.join(FLYWHEEL_DIR, "pcg_label_manifest.json")
GENERATION_LOG_FILE = os.path.join(FLYWHEEL_DIR, "generation_runs.json")

# 兼容人工 topo 标注输出
TOPO_OUT_DIR = os.path.abspath(
    os.path.join(PROJECT_DIR, "../AI_VECTOR_ROUTER_With_topo/annotations_topo")
)

GOOD_FONT_PREFIX = "PCG_AestheticGood"
BAD_FONT_PREFIX = "PCG_AestheticBad"

MAX_JSON_MB = 80
MAX_JSON_BYTES = int(MAX_JSON_MB * 1024 * 1024)

CANVAS_SIZE = 400
THUMB_SIZE = 220

DEFAULT_GRID_N = 5
DEFAULT_NEW_BATCH_SIZE = 1000
DEFAULT_PREFILTER_MODE = "solver_allowed"

GOOD_PRIVATE_BASE = 0xF0000
BAD_PRIVATE_BASE = 0xF8000


# =========================================================
# 🧠 预留：模型 hook
# =========================================================
class AestheticHooks:
    """
    后续可接入：
      - aesthetic_critic.pt：用户审美模型
      - stroke_relation_model.pt：笔画关系模型
      - primitive_compatibility_model.pt：primitive 与 layout/role 兼容性模型
    """

    def __init__(self):
        self.enabled = False
        self.model = None

    def score_candidate(self, candidate):
        return None

    def should_keep_before_human_label(self, candidate):
        return True

    def sample_weight(self, candidate):
        # 当前没有审美模型时，默认使用 combined_score / gnn_score 做弱引导。
        for key in ["combined_score", "gnn_layout_realness_score"]:
            if key in candidate:
                try:
                    return max(0.05, float(candidate[key]))
                except Exception:
                    pass

        item = candidate.get("gnn_score_item", {})
        if isinstance(item, dict):
            for key in ["combined_score", "gnn_layout_realness_score"]:
                if key in item:
                    try:
                        return max(0.05, float(item[key]))
                    except Exception:
                        pass

        return 1.0


AESTHETIC_HOOKS = AestheticHooks()


# =========================================================
# 🧮 基础工具
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path, required=False, default=None):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(path)
        return {} if default is None else default

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def json_size_bytes(obj):
    return len(json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return int(default)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def stable_hash_int(s, mod=10**9):
    h = hashlib.md5(str(s).encode("utf-8")).hexdigest()
    return int(h[:12], 16) % mod


def make_synthetic_hex(candidate_id, label):
    base = GOOD_PRIVATE_BASE if label == "good" else BAD_PRIVATE_BASE
    offset = stable_hash_int(candidate_id, 0x7000)
    code = min(base + offset, 0x10FFFF)
    return f"U+{code:05X}"


def char_from_hex_key(hex_key):
    try:
        cp = int(hex_key.replace("U+", ""), 16)
        if 0 <= cp <= 0x10FFFF:
            return chr(cp)
    except Exception:
        pass
    return "□"


def weighted_sample_without_replacement(items, weights, k):
    items = list(items)
    weights = [max(1e-8, float(w)) for w in weights]
    out = []

    k = min(k, len(items))

    for _ in range(k):
        total = sum(weights)
        if total <= 0:
            idx = random.randrange(len(items))
        else:
            r = random.random() * total
            acc = 0.0
            idx = 0
            for i, w in enumerate(weights):
                acc += w
                if acc >= r:
                    idx = i
                    break

        out.append(items.pop(idx))
        weights.pop(idx)

    return out


def patch_python_constant(script_path, name, value):
    """
    修改脚本里的顶层常量，例如：
        RANDOM_SEED = 42
        NUM_SAMPLES = 200

    如果找不到常量，静默跳过。
    """
    if not os.path.exists(script_path):
        return False

    with open(script_path, "r", encoding="utf-8") as f:
        text = f.read()

    old_text = text

    if isinstance(value, str):
        value_repr = repr(value)
    elif value is None:
        value_repr = "None"
    else:
        value_repr = str(value)

    pattern = rf"^({re.escape(name)}\s*=\s*)(.+?)$"
    text = re.sub(pattern, rf"\g<1>{value_repr}", text, flags=re.MULTILINE)

    if text != old_text:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(text)
        return True

    return False


# =========================================================
# 📦 Candidate 读取
# =========================================================
def get_candidate_id(candidate, idx=0):
    for k in [
        "generated_glyph_id",
        "glyph_candidate_id",
        "candidate_id",
        "sample_id",
        "grammar_sample_id",
    ]:
        if isinstance(candidate, dict) and candidate.get(k, ""):
            return str(candidate[k])
    return f"glyph_candidate_{idx:05d}"


def get_candidate_list_from_scored(data):
    keys = [
        "ranked_solved_glyph_candidates_by_combined",
        "ranked_solved_glyph_candidates",
        "solved_glyph_candidates",
        "glyph_candidates",
        "candidates",
    ]

    for k in keys:
        if k in data and isinstance(data[k], list) and len(data[k]) > 0:
            return data[k]

    return []


def get_candidate_list_from_solved(data):
    if isinstance(data, list):
        return data

    keys = [
        "solved_glyph_candidates",
        "valid_solved_glyph_candidates",
        "usable_solved_glyph_candidates",
        "glyph_candidates",
        "candidates",
    ]

    out = []
    for k in keys:
        if k in data and isinstance(data[k], list):
            out.extend(data[k])

    if out:
        seen = set()
        dedup = []
        for i, c in enumerate(out):
            cid = get_candidate_id(c, i)
            if cid not in seen:
                seen.add(cid)
                dedup.append(c)
        return dedup

    return []


def load_candidates():
    scored_data = load_json(SCORED_FILE, required=False, default={})
    candidates = get_candidate_list_from_scored(scored_data)

    if len(candidates) == 0:
        solved_data = load_json(SOLVED_FILE, required=False, default={})
        candidates = get_candidate_list_from_solved(solved_data)

    normalized = []

    for idx, c in enumerate(candidates):
        if not isinstance(c, dict):
            continue

        cc = copy.deepcopy(c)
        cid = get_candidate_id(cc, idx)
        cc["candidate_id"] = cid
        cc["generated_glyph_id"] = cc.get("generated_glyph_id", cid)
        cc["glyph_candidate_id"] = cc.get("glyph_candidate_id", cid)

        score_item = cc.get("gnn_score_item", {})
        if isinstance(score_item, dict):
            for k in [
                "gnn_layout_realness_score",
                "solver_feasibility_score",
                "combined_score",
                "quality_status",
                "solver_allowed",
            ]:
                if k in score_item and k not in cc:
                    cc[k] = score_item[k]

        normalized.append(cc)

    return normalized


def candidate_is_solver_allowed(candidate):
    if candidate.get("solver_allowed", None) is not None:
        return bool(candidate.get("solver_allowed"))

    q = candidate.get("quality_report", {})
    if isinstance(q, dict):
        status = q.get("quality_status", q.get("QualityStatus", ""))
    else:
        status = candidate.get("quality_status", "")

    if str(status) in ["good", "usable_but_rough", "usable"]:
        return True

    status = candidate.get("quality_status", "")
    if str(status) in ["good", "usable_but_rough", "usable"]:
        return True

    return False


def prefilter_candidates(candidates, mode=DEFAULT_PREFILTER_MODE):
    out = []

    for c in candidates:
        if not AESTHETIC_HOOKS.should_keep_before_human_label(c):
            continue

        if mode == "all":
            out.append(c)

        elif mode == "solver_allowed":
            if candidate_is_solver_allowed(c):
                out.append(c)

        elif mode == "top_combined":
            score = safe_float(c.get("combined_score", c.get("gnn_layout_realness_score", 0.0)), 0.0)
            if score >= 0.50 and candidate_is_solver_allowed(c):
                out.append(c)
        else:
            out.append(c)

    return out if out else candidates


# =========================================================
# 🎨 Bezier / 拓扑 / 渲染
# =========================================================
def cubic_bezier_np(p, t):
    p = np.asarray(p, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32).reshape(-1, 1)
    mt = 1.0 - t
    return mt**3 * p[0] + 3 * mt**2 * t * p[1] + 3 * mt * t**2 * p[2] + t**3 * p[3]


def get_bezier_derivative(pts, t):
    pts = np.asarray(pts, dtype=np.float32)
    mt = 1.0 - float(t)
    return (
        3 * mt**2 * (pts[1] - pts[0])
        + 6 * mt * float(t) * (pts[2] - pts[1])
        + 3 * float(t)**2 * (pts[3] - pts[2])
    )


def get_angle(v1, v2):
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-5 or n2 < 1e-5:
        return 0.0
    cos_th = np.clip(float(np.dot(v1, v2)) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_th)))


def get_polygon_orientation(pts):
    pts = np.asarray(pts, dtype=np.float32)
    area = 0.0
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return "ccw" if area > 0 else "cw"


def normalize_local_polyline(polyline):
    arr = np.asarray(polyline, dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) < 2:
        return np.asarray([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32)

    arr = arr[:, :2].copy()
    finite = np.all(np.isfinite(arr), axis=1)
    arr = arr[finite]

    if len(arr) < 2:
        return np.asarray([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32)

    xmin, xmax = float(np.min(arr[:, 0])), float(np.max(arr[:, 0]))
    ymin, ymax = float(np.min(arr[:, 1])), float(np.max(arr[:, 1]))
    xr = xmax - xmin

    if xr < 1e-6:
        return np.asarray([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32)

    arr[:, 0] = (arr[:, 0] - 0.5 * (xmin + xmax)) / xr
    arr[:, 1] = (arr[:, 1] - 0.5 * (ymin + ymax)) / xr
    arr[:, 0] = np.clip(arr[:, 0], -1.0, 1.0)
    arr[:, 1] = np.clip(arr[:, 1], -1.0, 1.0)
    return arr.astype(np.float32)


def apply_variant(polyline, variant_id):
    pts = np.asarray(polyline, dtype=np.float32).copy()
    v = int(variant_id) % 4

    if v == 0:
        return pts
    if v == 1:
        out = pts.copy()
        out[:, 1] = pts[::-1, 1]
        return out
    if v == 2:
        out = pts.copy()
        out[:, 1] = -out[:, 1]
        return out
    if v == 3:
        out = pts.copy()
        out[:, 1] = -pts[::-1, 1]
        return out
    return pts


def read_layout_prior(node):
    lp = node.get("layout_prior", {})
    if not isinstance(lp, dict):
        lp = {}

    center = lp.get("center_norm", node.get("center_norm", [0.5, 0.5]))
    center = np.asarray(center, dtype=np.float32)[:2]
    if np.max(np.abs(center)) > 1.5:
        center = center / float(CANVAS_SIZE)

    theta = safe_float(lp.get("rotation_rad", node.get("rotation_rad", 0.0)), 0.0)
    if "rotation_deg" in lp and "rotation_rad" not in lp:
        theta = math.radians(safe_float(lp["rotation_deg"], 0.0))

    length = safe_float(
        lp.get("length_norm", lp.get("scale_norm", node.get("length_norm", node.get("scale_norm", 0.25)))),
        0.25,
    )

    if abs(length) > 1.5:
        length = length / float(CANVAS_SIZE)
    length = clamp(length, 0.02, 1.5)

    return {
        "center_norm": center,
        "rotation_rad": theta,
        "length_norm": length,
    }


def polyline_from_node(node):
    for k in [
        "solved_polyline_px",
        "polyline_px",
        "world_polyline_px",
        "transformed_polyline_px",
        "render_polyline_px",
        "points_px",
        "polyline",
        "points",
    ]:
        if k in node:
            arr = np.asarray(node[k], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) >= 2:
                arr = arr[:, :2]
                if np.max(np.abs(arr)) <= 1.5:
                    arr = arr * CANVAS_SIZE
                return arr

    ref = node.get("primitive_ref", {})
    if isinstance(ref, dict):
        poly = None
        for k in [
            "primitive_polyline_local_norm",
            "prototype_polyline_local_norm",
            "prototype_polyline_norm",
            "polyline_local_norm",
            "polyline_norm",
            "polyline",
            "points",
        ]:
            if k in ref:
                poly = ref[k]
                break

        if poly is not None:
            local = normalize_local_polyline(poly)
            local = apply_variant(local, node.get("variant_id", 0))

            lp = read_layout_prior(node)
            center = lp["center_norm"] * CANVAS_SIZE
            theta = lp["rotation_rad"]
            length = lp["length_norm"] * CANVAS_SIZE

            c = math.cos(theta)
            s = math.sin(theta)
            R = np.asarray([[c, -s], [s, c]], dtype=np.float32)

            pts = local * length
            pts = pts @ R.T
            pts = pts + center[None, :]
            return pts

    lp = read_layout_prior(node)
    center = lp["center_norm"] * CANVAS_SIZE
    theta = lp["rotation_rad"]
    length = lp["length_norm"] * CANVAS_SIZE
    d = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float32) * length * 0.5
    return np.stack([center - d, center + d], axis=0)


def bezier_from_polyline(polyline):
    pts = np.asarray(polyline, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 2:
        return np.asarray([[180, 180], [190, 190], [210, 210], [220, 220]], dtype=np.float32)

    pts = pts[:, :2]
    p0 = pts[0]
    p3 = pts[-1]

    if len(pts) >= 4:
        p1 = pts[max(1, len(pts) // 3)]
        p2 = pts[min(len(pts) - 2, 2 * len(pts) // 3)]
    elif len(pts) == 3:
        p1 = pts[1]
        p2 = pts[1]
    else:
        p1 = p0 + (p3 - p0) / 3.0
        p2 = p0 + 2.0 * (p3 - p0) / 3.0

    return np.stack([p0, p1, p2, p3], axis=0).astype(np.float32)


def choose_nodes(candidate):
    for k in ["solved_nodes", "final_nodes", "optimized_nodes", "layout_nodes", "nodes"]:
        if k in candidate and isinstance(candidate[k], list):
            return candidate[k]
    return []


def get_edges(candidate):
    topo = candidate.get("topology", {})
    if isinstance(topo, dict):
        if "positive_edges_undirected" in topo and isinstance(topo["positive_edges_undirected"], list):
            return topo["positive_edges_undirected"]
        if "edges" in topo and isinstance(topo["edges"], list):
            return topo["edges"]
        if "positive_edges_directed" in topo and isinstance(topo["positive_edges_directed"], list):
            return [e for e in topo["positive_edges_directed"] if e.get("direction", "forward") == "forward"]
    if "edges" in candidate and isinstance(candidate["edges"], list):
        return candidate["edges"]
    return []


def node_id_of(node, fallback):
    return safe_int(node.get("node_id", node.get("source_node_id", fallback)), fallback)


def extract_width_bezier(node):
    for key in ["width_bezier", "solved_width_bezier", "widths"]:
        if key in node:
            arr = np.asarray(node[key], dtype=np.float32).reshape(-1)
            if len(arr) >= 4:
                return arr[:4].astype(float).tolist()
            if len(arr) == 1:
                w = float(arr[0])
                return [w, w, w, w]

    for key in [
        "width_px",
        "stroke_width_px",
        "render_width_px",
        "width_mean_px",
        "actual_width_px",
        "target_width_px",
    ]:
        if key in node:
            w = safe_float(node[key], 5.0)
            return [w, w, w, w]

    ref = node.get("primitive_ref", {})
    if isinstance(ref, dict):
        for key in ["width_px", "width", "stroke_width_px"]:
            if key in ref:
                w = safe_float(ref[key], 5.0)
                return [w, w, w, w]

    width_token = safe_int(node.get("width_token", 0), 0)
    w_map = {0: 5.0, 1: 8.0, 2: 12.0, 3: 16.0}
    w = w_map.get(width_token, 5.0)
    return [w, w, w, w]


def stroke_length_from_bezier(mb):
    curve = cubic_bezier_np(mb, np.linspace(0, 1, 50))
    return float(np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1)))


def candidate_to_strokes(candidate):
    nodes = choose_nodes(candidate)
    strokes = []
    id_to_bezier_id = {}

    for idx, node in enumerate(nodes):
        nid = node_id_of(node, idx)
        bezier_id = idx + 1
        id_to_bezier_id[nid] = bezier_id

        mb = None
        for k in ["mother_bezier", "solved_mother_bezier", "bezier", "control_points"]:
            if k in node:
                arr = np.asarray(node[k], dtype=np.float32)
                if arr.ndim == 2 and arr.shape[0] == 4 and arr.shape[1] >= 2:
                    mb = arr[:, :2]
                    break

        if mb is None:
            poly = polyline_from_node(node)
            mb = bezier_from_polyline(poly)

        w = extract_width_bezier(node)

        c_pts = cubic_bezier_np(mb, np.linspace(0, 1, 50))
        xmin, ymin = np.min(c_pts, axis=0)
        xmax, ymax = np.max(c_pts, axis=0)

        stroke_type = "closed" if float(np.linalg.norm(mb[0] - mb[3])) < 2.0 else "open"

        strokes.append({
            "bezier_id": int(bezier_id),
            "stroke_type": stroke_type,
            "length": round(stroke_length_from_bezier(mb), 2),
            "bbox": [
                round(float(xmin), 1),
                round(float(ymin), 1),
                round(float(xmax), 1),
                round(float(ymax), 1),
            ],
            "mother_bezier": mb.astype(float).tolist(),
            "width_bezier": [float(x) for x in w],
        })

    return strokes, id_to_bezier_id


def point_on_bezier(mb, t):
    return cubic_bezier_np(mb, [t])[0]


def topology_events_from_candidate(candidate, strokes, id_to_bezier_id):
    events = []
    edges = get_edges(candidate)

    bid_to_stroke = {s["bezier_id"]: s for s in strokes}

    for edge in edges:
        if not isinstance(edge, dict):
            continue

        u_raw = safe_int(edge.get("u", edge.get("src", -1)), -1)
        v_raw = safe_int(edge.get("v", edge.get("dst", -1)), -1)

        if u_raw not in id_to_bezier_id or v_raw not in id_to_bezier_id:
            continue

        id1 = int(id_to_bezier_id[u_raw])
        id2 = int(id_to_bezier_id[v_raw])

        s1 = bid_to_stroke.get(id1)
        s2 = bid_to_stroke.get(id2)
        if s1 is None or s2 is None:
            continue

        p1 = np.asarray(s1["mother_bezier"], dtype=np.float32)
        p2 = np.asarray(s2["mother_bezier"], dtype=np.float32)

        jt = str(edge.get("j_type", edge.get("type", "E2E")))

        t_u = clamp(safe_float(edge.get("t_u", edge.get("t_a", 0.0)), 0.0), 0.0, 1.0)
        t_v = clamp(safe_float(edge.get("t_v", edge.get("t_b", 0.0)), 0.0), 0.0, 1.0)

        pos_u = point_on_bezier(p1, t_u)
        pos_v = point_on_bezier(p2, t_v)
        pos = 0.5 * (pos_u + pos_v)

        ang = edge.get("angle", edge.get("angle_deg", None))
        if ang is None:
            ang = get_angle(get_bezier_derivative(p1, t_u), get_bezier_derivative(p2, t_v))
        ang = round(float(ang), 1)

        if jt == "T":
            ev = {
                "type": "T",
                "guest": id1,
                "guest_t": round(float(t_u), 3),
                "host": id2,
                "host_t": round(float(t_v), 3),
                "angle": ang,
                "position": [round(float(pos[0]), 1), round(float(pos[1]), 1)],
            }
        elif jt == "X":
            ev = {
                "type": "X",
                "stroke_a": id1,
                "t_a": round(float(t_u), 3),
                "stroke_b": id2,
                "t_b": round(float(t_v), 3),
                "angle": ang,
                "position": [round(float(pos[0]), 1), round(float(pos[1]), 1)],
            }
        else:
            ev = {
                "type": "E2E",
                "stroke_a": id1,
                "t_a": round(float(t_u), 3),
                "stroke_b": id2,
                "t_b": round(float(t_v), 3),
                "angle": ang,
                "position": [round(float(pos[0]), 1), round(float(pos[1]), 1)],
            }

        events.append(ev)

    return events


def cycles_from_events(strokes, events):
    try:
        import networkx as nx
    except Exception:
        return []

    G = nx.Graph()
    connection_points = {}

    for ev in events:
        ev_type = ev.get("type")
        if ev_type not in ["E2E", "T", "X"]:
            continue

        u = ev.get("stroke_a", ev.get("guest"))
        v = ev.get("stroke_b", ev.get("host"))

        if u is None or v is None:
            continue

        u, v = min(int(u), int(v)), max(int(u), int(v))
        G.add_edge(u, v)
        connection_points.setdefault((u, v), []).append(np.asarray(ev.get("position", [0, 0]), dtype=np.float32))

    valid_cycle_lists = []

    for (u, v), pts_list in connection_points.items():
        if len(pts_list) >= 2:
            for i in range(len(pts_list)):
                for j in range(i + 1, len(pts_list)):
                    if np.linalg.norm(pts_list[i] - pts_list[j]) >= 5.0:
                        valid_cycle_lists.append([u, v])
                        break

    try:
        for members in nx.cycle_basis(G):
            members = [int(n) for n in members]
            if len(members) >= 3:
                valid_cycle_lists.append(members)
    except Exception:
        pass

    stroke_center = {}
    for s in strokes:
        mb = np.asarray(s["mother_bezier"], dtype=np.float32)
        stroke_center[int(s["bezier_id"])] = np.mean(mb, axis=0)

    cycles = []
    seen = set()

    for members in valid_cycle_lists:
        key = tuple(sorted(members))
        if key in seen:
            continue
        seen.add(key)

        pts = [stroke_center[m] for m in members if m in stroke_center]
        orient = get_polygon_orientation(pts) if len(pts) >= 3 else "cw"
        cycles.append({
            "cycle_id": len(cycles),
            "members": [int(x) for x in members],
            "orientation": orient,
        })

    return cycles


def candidate_to_char_bundle(candidate, label):
    cid = get_candidate_id(candidate, 0)
    hex_key = make_synthetic_hex(cid, label)
    char = char_from_hex_key(hex_key)

    strokes, id_to_bezier_id = candidate_to_strokes(candidate)
    events = topology_events_from_candidate(candidate, strokes, id_to_bezier_id)
    cycles = cycles_from_events(strokes, events)

    edit_history = [{
        "action": "PCG_HUMAN_LABEL",
        "label": label,
        "candidate_id": cid,
        "source": "annotation_flywheel_app",
        "combined_score": safe_float(candidate.get("combined_score", 0.0), 0.0),
        "gnn_layout_realness_score": safe_float(candidate.get("gnn_layout_realness_score", 0.0), 0.0),
    }]

    return hex_key, {
        "glyph_info": {
            "hex_key": hex_key,
            "char": char,
        },
        "strokes": strokes,
        "topology_events": events,
        "cycles": cycles,
        "edit_history": edit_history,
    }


def render_candidate_pixmap(candidate, thumb_size=THUMB_SIZE):
    image = QImage(thumb_size, thumb_size, QImage.Format_RGB32)
    image.fill(QColor("white"))

    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)

    strokes, _ = candidate_to_strokes(candidate)
    scale = thumb_size / float(CANVAS_SIZE)

    for s in strokes:
        mb = np.asarray(s["mother_bezier"], dtype=np.float32) * scale
        w = np.asarray(s["width_bezier"], dtype=np.float32)
        width_px = max(2.0, float(np.mean(w)) * 2.0 * scale)

        path = QPainterPath()
        path.moveTo(float(mb[0][0]), float(mb[0][1]))
        path.cubicTo(
            float(mb[1][0]), float(mb[1][1]),
            float(mb[2][0]), float(mb[2][1]),
            float(mb[3][0]), float(mb[3][1]),
        )

        pen = QPen(QColor("black"))
        pen.setWidthF(width_px)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawPath(path)

    painter.end()
    return QPixmap.fromImage(image)


# =========================================================
# 💾 80MB 智能分文件保存
# =========================================================
def remove_old_split_files(prefix):
    if not os.path.exists(TOPO_OUT_DIR):
        return
    for fn in os.listdir(TOPO_OUT_DIR):
        if fn.startswith(prefix + "_part") and fn.endswith("_topo.json"):
            try:
                os.remove(os.path.join(TOPO_OUT_DIR, fn))
            except Exception:
                pass


def write_split_topo_files(prefix, data_dict):
    ensure_dir(TOPO_OUT_DIR)
    remove_old_split_files(prefix)

    items = list(data_dict.items())
    files = []
    current = {}
    part_idx = 0

    def flush(part_data, idx):
        if not part_data:
            return None
        out_path = os.path.join(TOPO_OUT_DIR, f"{prefix}_part{idx:03d}_topo.json")
        save_json(part_data, out_path)
        return out_path

    for key, bundle in items:
        test = dict(current)
        test[key] = bundle

        if current and json_size_bytes(test) > MAX_JSON_BYTES:
            path = flush(current, part_idx)
            if path:
                files.append(path)
            part_idx += 1
            current = {key: bundle}
        else:
            current = test

    path = flush(current, part_idx)
    if path:
        files.append(path)

    return files


def save_annotation_outputs(labels, candidates_by_id):
    good_dict = {}
    bad_dict = {}

    manifest = {
        "schema_version": "pcg_annotation_manifest_v2",
        "good_count": 0,
        "bad_count": 0,
        "items": {},
        "topo_out_dir": TOPO_OUT_DIR,
        "max_json_mb": MAX_JSON_MB,
    }

    for cid, label in labels.items():
        if cid not in candidates_by_id:
            continue
        if label not in ["good", "bad"]:
            continue

        candidate = candidates_by_id[cid]
        hex_key, bundle = candidate_to_char_bundle(candidate, label)

        if label == "good":
            good_dict[hex_key] = bundle
        else:
            bad_dict[hex_key] = bundle

        manifest["items"][cid] = {
            "candidate_id": cid,
            "label": label,
            "hex_key": hex_key,
            "char": bundle["glyph_info"]["char"],
            "source_candidate_file": SCORED_FILE if os.path.exists(SCORED_FILE) else SOLVED_FILE,
        }

    manifest["good_count"] = len(good_dict)
    manifest["bad_count"] = len(bad_dict)

    good_files = write_split_topo_files(GOOD_FONT_PREFIX, good_dict)
    bad_files = write_split_topo_files(BAD_FONT_PREFIX, bad_dict)

    manifest["good_topo_files"] = good_files
    manifest["bad_topo_files"] = bad_files

    save_json(manifest, MANIFEST_FILE)
    return manifest


# =========================================================
# 🔄 生成链路后台线程
# =========================================================
class GenerationWorker(QThread):
    log = pyqtSignal(str)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, num_samples, random_seed, parent=None):
        super().__init__(parent)
        self.num_samples = int(num_samples)
        self.random_seed = int(random_seed)

    def run(self):
        try:
            ensure_dir(FLYWHEEL_DIR)

            self.log.emit(f"Using random seed: {self.random_seed}")
            self.log.emit(f"Using num samples: {self.num_samples}")

            patched = []
            patched.append(("layout RANDOM_SEED", patch_python_constant(LAYOUT_SAMPLER_SCRIPT, "RANDOM_SEED", self.random_seed)))
            patched.append(("layout NUM_SAMPLES", patch_python_constant(LAYOUT_SAMPLER_SCRIPT, "NUM_SAMPLES", self.num_samples)))
            patched.append(("primitive RANDOM_SEED", patch_python_constant(PRIMITIVE_SAMPLER_SCRIPT, "RANDOM_SEED", self.random_seed + 1)))
            patched.append(("solver RANDOM_SEED", patch_python_constant(SOLVER_SCRIPT, "RANDOM_SEED", self.random_seed + 2)))

            for patch_name, ok in patched:
                self.log.emit(f"patch {patch_name}: {'ok' if ok else 'skip'}")

            scripts = [
                ("layout_aware_graph_sampler.py", LAYOUT_SAMPLER_SCRIPT),
                ("stroke_primitive_sampler.py", PRIMITIVE_SAMPLER_SCRIPT),
                ("constraint_solver.py", SOLVER_SCRIPT),
                ("score_solved_glyphs.py", SCORER_SCRIPT),
            ]

            for name, path in scripts:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing script: {path}")

                self.log.emit("=" * 80)
                self.log.emit(f"Running: {name}")
                self.log.emit("=" * 80)

                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                env["PYTHONUTF8"] = "1"

                proc = subprocess.Popen(
                    [sys.executable, path],
                    cwd=PROJECT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )

                assert proc.stdout is not None
                tail_lines = []

                for line in proc.stdout:
                    line = line.rstrip("\n")
                    self.log.emit(line)

                    tail_lines.append(line)
                    if len(tail_lines) > 160:
                        tail_lines = tail_lines[-160:]

                code = proc.wait()

                if code != 0:
                    tail_text = "\n".join(tail_lines)
                    raise RuntimeError(
                        f"{name} failed with exit code {code}\n\n"
                        f"========== 子进程最后日志 ==========\n"
                        f"{tail_text}\n"
                        f"===================================="
                    )

            run_log = load_json(GENERATION_LOG_FILE, required=False, default={"runs": []})
            run_log.setdefault("runs", []).append({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "random_seed": self.random_seed,
                "num_samples": self.num_samples,
                "scored_file": SCORED_FILE,
                "solved_file": SOLVED_FILE,
            })
            save_json(run_log, GENERATION_LOG_FILE)

            self.finished_ok.emit()

        except Exception:
            self.failed.emit(traceback.format_exc())


# =========================================================
# 🧱 UI Widgets
# =========================================================
class CandidateTile(QWidget):
    def __init__(self, candidate, checked=False, parent=None):
        super().__init__(parent)
        self.candidate = candidate
        self.cid = get_candidate_id(candidate, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        pix = render_candidate_pixmap(candidate, THUMB_SIZE)

        self.img_label = QLabel()
        self.img_label.setPixmap(pix)
        self.img_label.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("background: white; border: 1px solid #555;")
        layout.addWidget(self.img_label)

        score_text = self._score_text(candidate)
        self.info_label = QLabel(f"{self.cid}\n{score_text}")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.info_label)

        self.check = QCheckBox("✅ Good")
        self.check.setChecked(checked)
        self.check.setStyleSheet("font-size: 13px; font-weight: bold;")
        layout.addWidget(self.check)

        self.img_label.mousePressEvent = self.toggle_by_image

        self.setStyleSheet("""
        QWidget {
            background-color: #f6f6f6;
            border-radius: 6px;
        }
        """)

    def _score_text(self, c):
        parts = []
        if c.get("combined_score", None) is not None:
            parts.append(f"comb={safe_float(c.get('combined_score')):.3f}")
        if c.get("gnn_layout_realness_score", None) is not None:
            parts.append(f"gnn={safe_float(c.get('gnn_layout_realness_score')):.3f}")
        if c.get("quality_status", None) is not None:
            parts.append(str(c.get("quality_status")))
        return " | ".join(parts)

    def toggle_by_image(self, event):
        self.check.setChecked(not self.check.isChecked())

    def is_good(self):
        return self.check.isChecked()


class PoolManagerDialog(QDialog):
    def __init__(self, label_name, labels, candidates_by_id, on_changed, parent=None):
        super().__init__(parent)
        self.label_name = label_name
        self.labels = labels
        self.candidates_by_id = candidates_by_id
        self.on_changed = on_changed

        self.setWindowTitle(f"{label_name.upper()} Pool Manager")
        self.resize(1200, 850)

        self.main_layout = QVBoxLayout(self)

        header = QHBoxLayout()
        self.title = QLabel()
        self.title.setStyleSheet("font-size: 18px; font-weight: bold;")
        header.addWidget(self.title)
        header.addStretch()

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        header.addWidget(btn_close)
        self.main_layout.addLayout(header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.container = QWidget()
        self.grid = QGridLayout(self.container)
        self.scroll.setWidget(self.container)
        self.main_layout.addWidget(self.scroll)

        self.refresh()

    def clear_grid(self):
        while self.grid.count():
            child = self.grid.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def refresh(self):
        self.clear_grid()

        ids = [cid for cid, lab in self.labels.items() if lab == self.label_name and cid in self.candidates_by_id]
        self.title.setText(f"{self.label_name.upper()} pool: {len(ids)}")

        cols = 4
        for idx, cid in enumerate(ids):
            cand = self.candidates_by_id[cid]

            box = QGroupBox(cid)
            box_layout = QVBoxLayout(box)

            pix = render_candidate_pixmap(cand, THUMB_SIZE)
            img = QLabel()
            img.setPixmap(pix)
            img.setFixedSize(THUMB_SIZE, THUMB_SIZE)
            img.setAlignment(Qt.AlignCenter)
            img.setStyleSheet("background: white; border: 1px solid #555;")
            box_layout.addWidget(img)

            btn_remove = QPushButton("Remove from this pool")
            btn_remove.setStyleSheet("background-color: #B71C1C; color: white; font-weight: bold; padding: 6px;")
            btn_remove.clicked.connect(lambda checked=False, x=cid: self.remove_item(x))
            box_layout.addWidget(btn_remove)

            self.grid.addWidget(box, idx // cols, idx % cols)

    def remove_item(self, cid):
        if cid in self.labels:
            del self.labels[cid]
        self.on_changed()
        self.refresh()


# =========================================================
# 🪟 Main Window
# =========================================================
class AnnotationFlywheelApp(QMainWindow):
    def __init__(self):
        super().__init__()

        ensure_dir(FLYWHEEL_DIR)
        ensure_dir(THUMB_DIR)
        ensure_dir(TOPO_OUT_DIR)

        self.setWindowTitle("PCG Glyph Annotation Flywheel App")
        self.resize(1600, 1000)

        self.candidates = []
        self.candidates_by_id = {}
        self.state = load_json(STATE_FILE, required=False, default={})
        self.labels = self.state.get("labels", {})
        if not isinstance(self.labels, dict):
            self.labels = {}

        self.current_batch = []
        self.tiles = []
        self.worker = None

        self.init_ui()
        self.reload_candidates()
        self.generate_grid()

    def init_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        ctrl = QHBoxLayout()

        ctrl.addWidget(QLabel("Grid N:"))
        self.spin_n = QSpinBox()
        self.spin_n.setRange(2, 12)
        self.spin_n.setValue(DEFAULT_GRID_N)
        ctrl.addWidget(self.spin_n)

        ctrl.addWidget(QLabel("New batch size:"))
        self.spin_new_samples = QSpinBox()
        self.spin_new_samples.setRange(50, 10000)
        self.spin_new_samples.setSingleStep(50)
        self.spin_new_samples.setValue(DEFAULT_NEW_BATCH_SIZE)
        ctrl.addWidget(self.spin_new_samples)

        ctrl.addWidget(QLabel("Prefilter:"))
        self.prefilter_combo = QComboBox()
        self.prefilter_combo.addItems(["solver_allowed", "top_combined", "all"])
        self.prefilter_combo.setCurrentText(DEFAULT_PREFILTER_MODE)
        ctrl.addWidget(self.prefilter_combo)

        ctrl.addWidget(QLabel("Sampling:"))
        self.sampling_combo = QComboBox()
        self.sampling_combo.addItems(["weighted_by_score", "uniform_random"])
        ctrl.addWidget(self.sampling_combo)

        self.btn_fresh = QPushButton("🔄 Generate Fresh Candidates")
        self.btn_fresh.clicked.connect(self.generate_fresh_candidates)
        self.btn_fresh.setStyleSheet("background-color: #1565C0; color: white; font-weight: bold; padding: 8px;")
        ctrl.addWidget(self.btn_fresh)

        self.btn_grid = QPushButton("🎲 Generate N×N Grid")
        self.btn_grid.clicked.connect(self.generate_grid)
        ctrl.addWidget(self.btn_grid)

        self.btn_commit = QPushButton("💾 Commit Grid Labels")
        self.btn_commit.clicked.connect(self.commit_current_batch)
        self.btn_commit.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 8px;")
        ctrl.addWidget(self.btn_commit)

        self.btn_save = QPushButton("💾 Save Now")
        self.btn_save.clicked.connect(self.save_all_outputs)
        ctrl.addWidget(self.btn_save)

        self.btn_good = QPushButton("✅ Manage Good")
        self.btn_good.clicked.connect(lambda: self.open_pool_manager("good"))
        ctrl.addWidget(self.btn_good)

        self.btn_bad = QPushButton("🚫 Manage Bad")
        self.btn_bad.clicked.connect(lambda: self.open_pool_manager("bad"))
        ctrl.addWidget(self.btn_bad)

        ctrl.addStretch()

        self.status_label = QLabel()
        self.status_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #1565C0;")
        ctrl.addWidget(self.status_label)

        layout.addLayout(ctrl)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(180)
        self.log_box.hide()
        layout.addWidget(self.log_box)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        layout.addWidget(line)

        help_text = QLabel(
            "勾选=好样本；未勾选=坏样本。Generate Fresh Candidates 会重新运行生成链路并更新候选池。"
        )
        help_text.setStyleSheet("font-size: 13px; color: #444;")
        layout.addWidget(help_text)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_container = QWidget()
        self.grid = QGridLayout(self.grid_container)
        self.grid.setSpacing(10)
        self.scroll.setWidget(self.grid_container)
        layout.addWidget(self.scroll)

    def append_log(self, text):
        self.log_box.append(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def set_busy(self, busy):
        self.btn_fresh.setEnabled(not busy)
        self.btn_grid.setEnabled(not busy)
        self.btn_commit.setEnabled(not busy)
        self.btn_save.setEnabled(not busy)
        self.progress.setVisible(busy)
        self.log_box.setVisible(busy)

    def reload_candidates(self):
        self.candidates = load_candidates()
        self.candidates_by_id = {get_candidate_id(c, i): c for i, c in enumerate(self.candidates)}
        self.update_status()

    def clear_grid(self):
        while self.grid.count():
            child = self.grid.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.tiles = []

    def available_candidates_for_sampling(self):
        mode = self.prefilter_combo.currentText()
        pool = prefilter_candidates(self.candidates, mode=mode)

        unlabeled = [c for c in pool if get_candidate_id(c, 0) not in self.labels]
        if len(unlabeled) >= 4:
            pool = unlabeled

        return pool

    def generate_grid(self):
        self.clear_grid()

        n = int(self.spin_n.value())
        k = n * n

        pool = self.available_candidates_for_sampling()
        if len(pool) == 0:
            QMessageBox.warning(self, "Empty", "没有可抽样 candidate。")
            return

        if self.sampling_combo.currentText() == "uniform_random":
            weights = [1.0 for _ in pool]
        else:
            weights = [AESTHETIC_HOOKS.sample_weight(c) for c in pool]

        batch = weighted_sample_without_replacement(pool, weights, min(k, len(pool)))
        self.current_batch = batch

        for idx, cand in enumerate(batch):
            cid = get_candidate_id(cand, idx)
            checked = self.labels.get(cid, None) == "good"

            tile = CandidateTile(cand, checked=checked)
            self.tiles.append(tile)

            self.grid.addWidget(tile, idx // n, idx % n)

        self.update_status()

    def commit_current_batch(self):
        if len(self.tiles) == 0:
            return

        good_count = 0
        bad_count = 0

        for tile in self.tiles:
            label = "good" if tile.is_good() else "bad"
            self.labels[tile.cid] = label
            if label == "good":
                good_count += 1
            else:
                bad_count += 1

        self.save_all_outputs(show_message=False)

        QMessageBox.information(
            self,
            "Committed",
            f"当前网格已提交：good={good_count}, bad={bad_count}\n已保存 topo split JSON。"
        )

        self.generate_grid()

    def save_all_outputs(self, show_message=True):
        state = {
            "schema_version": "pcg_annotation_state_v2",
            "labels": self.labels,
            "source_scored_file": SCORED_FILE,
            "source_solved_file": SOLVED_FILE,
            "topo_out_dir": TOPO_OUT_DIR,
        }
        save_json(state, STATE_FILE)

        manifest = save_annotation_outputs(self.labels, self.candidates_by_id)
        self.update_status()

        if show_message:
            QMessageBox.information(
                self,
                "Saved",
                f"Saved.\nGood={manifest['good_count']} Bad={manifest['bad_count']}\nTopo out dir:\n{TOPO_OUT_DIR}"
            )

    def open_pool_manager(self, label_name):
        dlg = PoolManagerDialog(
            label_name=label_name,
            labels=self.labels,
            candidates_by_id=self.candidates_by_id,
            on_changed=lambda: self.save_all_outputs(show_message=False),
            parent=self,
        )
        dlg.exec_()
        self.update_status()

    def generate_fresh_candidates(self):
        missing = [
            p for p in [LAYOUT_SAMPLER_SCRIPT, PRIMITIVE_SAMPLER_SCRIPT, SOLVER_SCRIPT, SCORER_SCRIPT]
            if not os.path.exists(p)
        ]

        if missing:
            QMessageBox.warning(self, "Missing scripts", "缺少脚本：\n" + "\n".join(missing))
            return

        reply = QMessageBox.question(
            self,
            "Generate Fresh Candidates",
            "这会重新运行 layout sampler / primitive sampler / solver / scorer，可能需要一些时间。\n继续吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if reply == QMessageBox.No:
            return

        seed = random.randint(1, 10**9)
        num_samples = int(self.spin_new_samples.value())

        self.log_box.clear()
        self.set_busy(True)

        self.worker = GenerationWorker(num_samples=num_samples, random_seed=seed)
        self.worker.log.connect(self.append_log)
        self.worker.finished_ok.connect(self.on_generation_finished)
        self.worker.failed.connect(self.on_generation_failed)
        self.worker.start()

    def on_generation_finished(self):
        self.append_log("✅ Generation finished.")
        self.reload_candidates()
        self.set_busy(False)
        self.generate_grid()
        QMessageBox.information(self, "Done", f"新候选已生成并加载：{len(self.candidates)} 个。")

    def on_generation_failed(self, err):
        self.append_log("❌ Generation failed.")
        self.append_log(err)
        self.set_busy(False)
        QMessageBox.critical(self, "Generation Failed", err[-4000:])

    def update_status(self):
        good = sum(1 for v in self.labels.values() if v == "good")
        bad = sum(1 for v in self.labels.values() if v == "bad")
        total = len(self.candidates)
        labeled = good + bad
        self.status_label.setText(f"Candidates: {total} | Labeled: {labeled} | Good: {good} | Bad: {bad}")

    def closeEvent(self, event):
        try:
            self.save_all_outputs(show_message=False)
        except Exception:
            traceback.print_exc()
        super().closeEvent(event)


# =========================================================
# 🚀 main
# =========================================================
def main():
    print("=" * 80)
    print("🚀 PCG Annotation Flywheel App")
    print("=" * 80)
    print(f"SCRIPT_DIR : {SCRIPT_DIR}")
    print(f"PROJECT_DIR: {PROJECT_DIR}")
    print(f"SCORED_FILE: {SCORED_FILE}")
    print(f"SOLVED_FILE: {SOLVED_FILE}")
    print(f"TOPO_OUT   : {TOPO_OUT_DIR}")
    print(f"MAX_JSON_MB: {MAX_JSON_MB}")
    print("=" * 80)

    ensure_dir(FLYWHEEL_DIR)
    ensure_dir(THUMB_DIR)
    ensure_dir(TOPO_OUT_DIR)

    app = QApplication(sys.argv)
    win = AnnotationFlywheelApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
