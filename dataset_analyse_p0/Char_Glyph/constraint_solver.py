import os

# =========================================================
# ⚠️ 多进程 + scipy/numpy 时，建议限制每个进程内部 BLAS 线程
# 避免 8 个进程 × 每个进程再开 8 线程，导致反而变慢
# 这些必须尽量放在 numpy/scipy import 前
# =========================================================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import math
import random
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

try:
    import networkx as nx
    HAS_NETWORKX = True
except Exception:
    HAS_NETWORKX = False


# =========================================================
# ⚙️ 全局配置
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(SCRIPT_DIR, "glyph_candidates_with_primitives.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "solved_glyph_candidates.json")

PREVIEW_DIR = os.path.join(SCRIPT_DIR, "solver_previews")
BEST_PREVIEW_DIR = os.path.join(PREVIEW_DIR, "best")
ROUGH_PREVIEW_DIR = os.path.join(PREVIEW_DIR, "rough")
BAD_PREVIEW_DIR = os.path.join(PREVIEW_DIR, "bad_debug")

RANDOM_SEED = 42

CANVAS_SIZE = 400.0
CANVAS_MARGIN = 20.0
GLYPH_CENTER = np.array([CANVAS_SIZE * 0.5, CANVAS_SIZE * 0.5], dtype=float)

# 调试时可以设成 10 / 20；正式可以 None
MAX_SOLVE_COUNT = None

# =========================================================
# 🚀 并行配置
# =========================================================
PARALLEL_SOLVE = True

# Mac M4 Pro 可以先用 6 或 8。
# 如果机器很卡，改成 4。
# 如果想自动：NUM_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 8))
NUM_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 8))

# 每个 worker 处理一个 candidate。
# candidate 本身是小 JSON，进程间传输开销可接受。
CHUNKSIZE = 1

# =========================================================
# 🧩 优化器配置
# =========================================================
NUM_RESTARTS = 3
RUN_TWO_STAGE_SOLVER = True

SOLVER_METHOD = "L-BFGS-B"
SOLVER_MAXITER_STAGE1 = 180
SOLVER_MAXITER_STAGE2 = 320

# 如果 primitive 缺失，退化为一根直线
FALLBACK_LINE = np.array([[-0.5, 0.0], [0.5, 0.0]], dtype=float)

EPS = 1e-8

# 结构性 shape，比如直线
STRUCTURAL_SHAPES = {20}

# ---------------------------------------------------------
# Angle 类型权重
# ---------------------------------------------------------
# 关键修改：
# E2E 只表示端点连接，不一定要求切线 180° 连续。
# 所以 E2E angle 在 objective 中弱化。
ANGLE_TYPE_WEIGHT = {
    "E2E": 0.15,
    "T": 1.0,
    "X": 1.0,
    "UNKNOWN": 0.5,
}

# Stage 1：先强行焊接 topology
LOSS_WEIGHTS_STAGE1 = {
    "junction": 240.0,
    "angle": 18.0,
    "length": 3.0,
    "bbox": 25.0,
    "center": 1.5,
    "extent": 2.0,
    "repulsion": 0.0,
    "compact": 4.0,
    "theta_reg": 0.0,
}

# Stage 2：布局、美学、边界微调
LOSS_WEIGHTS_STAGE2 = {
    "junction": 150.0,
    "angle": 12.0,
    "length": 7.0,
    "bbox": 60.0,
    "center": 8.0,
    "extent": 14.0,
    "repulsion": 10.0,
    "compact": 6.0,
    "theta_reg": 0.02,
}

TARGET_MIN_SPAN_RATIO = 0.22
TARGET_MAX_SPAN_RATIO = 0.88

# ---------------------------------------------------------
# Quality filter 阈值
# ---------------------------------------------------------
# 关键修改：
# quality 主角度指标只看 T/X，不看 E2E。
# E2E angle 保留在 debug 中，但不主导 good/bad。
QUALITY_THRESHOLDS = {
    "good": {
        "mean_junction_px": 2.0,
        "max_junction_px": 6.0,

        # 只针对 T/X angle
        "mean_tx_angle_diff_deg": 15.0,
        "max_tx_angle_diff_deg": 35.0,

        "bbox_overflow_px": 2.0,
    },
    "usable": {
        "mean_junction_px": 5.0,
        "max_junction_px": 16.0,

        # 只针对 T/X angle
        "mean_tx_angle_diff_deg": 30.0,
        "max_tx_angle_diff_deg": 70.0,

        "bbox_overflow_px": 8.0,
    },
}

# 预览图数量
PREVIEW_COUNT_BEST = 16
PREVIEW_COUNT_ROUGH = 8
PREVIEW_COUNT_BAD = 8

# 日志控制
PRINT_WORKER_DETAIL = False
PRINT_MAIN_PROGRESS = True
PROGRESS_EVERY = 5


# =========================================================
# 🧮 基础工具
# =========================================================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_mean(vals, default=0.0):
    return float(np.mean(vals)) if len(vals) > 0 else float(default)


def safe_max(vals, default=0.0):
    return float(np.max(vals)) if len(vals) > 0 else float(default)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def to_float_list(arr, ndigits=6):
    return np.round(np.asarray(arr, dtype=float), ndigits).tolist()


def radians_to_degrees(rad):
    return float(rad * 180.0 / math.pi)


def degrees_to_radians(deg):
    return float(deg * math.pi / 180.0)


def normalize_angle_pi(angle_rad):
    a = float(angle_rad)
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


def smallest_angle_diff(a, b):
    d = normalize_angle_pi(float(a) - float(b))
    return abs(d)


def line_angle_between(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < EPS or n2 < EPS:
        return 0.0

    c = float(np.dot(v1, v2) / (n1 * n2 + EPS))
    c = np.clip(c, -1.0, 1.0)

    return float(math.acos(c))


def edge_type_name(edge):
    if "j_type" in edge:
        return str(edge["j_type"])

    idx = int(edge.get("j_type_idx", 0))

    return {
        1: "E2E",
        2: "X",
        3: "T",
    }.get(idx, "UNKNOWN")


def get_target_angle_rad(edge):
    if "angle_deg" in edge:
        return degrees_to_radians(float(edge["angle_deg"]))

    if "angle" in edge:
        val = float(edge["angle"])

        if abs(val) > math.pi + 1e-4:
            return degrees_to_radians(val)

        return val

    if "angle_sin" in edge and "angle_cos" in edge:
        s = float(edge["angle_sin"])
        c = float(edge["angle_cos"])
        ang = math.atan2(s, c)

        if ang < 0:
            ang += 2 * math.pi

        if ang > math.pi:
            ang = 2 * math.pi - ang

        return ang

    return None


# =========================================================
# 📖 输入读取
# =========================================================
def get_candidate_list(data):
    if isinstance(data, dict):
        if "glyph_candidates" in data:
            return data["glyph_candidates"]
        if "sampled_topologies" in data:
            return data["sampled_topologies"]
        if "solved_glyph_candidates" in data:
            return data["solved_glyph_candidates"]

    if isinstance(data, list):
        return data

    return []


def get_candidate_edges(candidate):
    """
    标准字段：
        candidate["topology"]["positive_edges_undirected"]
    """
    if not isinstance(candidate, dict):
        return []

    topo = candidate.get("topology", {})

    if isinstance(topo, dict):
        if "positive_edges_undirected" in topo and isinstance(topo["positive_edges_undirected"], list):
            return topo["positive_edges_undirected"]

        if "edges" in topo and isinstance(topo["edges"], list):
            return topo["edges"]

        if "positive_edges_directed" in topo and isinstance(topo["positive_edges_directed"], list):
            directed = topo["positive_edges_directed"]
            return [
                e for e in directed
                if e.get("direction", "forward") == "forward"
            ]

    if "edges" in candidate and isinstance(candidate["edges"], list):
        return candidate["edges"]

    return []


# =========================================================
# 📐 Polyline 工具
# =========================================================
def get_local_polyline(node):
    prim_ref = node.get("primitive_ref", {})
    poly = prim_ref.get("primitive_polyline_local_norm", None)

    if poly is None:
        return FALLBACK_LINE.copy()

    arr = np.asarray(poly, dtype=float)

    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return FALLBACK_LINE.copy()

    return arr[:, :2].copy()


def polyline_arc_length(poly):
    poly = np.asarray(poly, dtype=float)

    if len(poly) < 2:
        return 0.0

    diffs = np.diff(poly, axis=0)

    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def polyline_cumulative_lengths(poly):
    poly = np.asarray(poly, dtype=float)

    if len(poly) < 2:
        return np.array([0.0], dtype=float)

    segs = np.linalg.norm(np.diff(poly, axis=0), axis=1)

    return np.concatenate([[0.0], np.cumsum(segs)])


def sample_polyline_point_and_tangent(poly, cumlen, t_frac):
    """
    t_frac ∈ [0,1]。
    当前 primitive 是 y=f(x) 离散形态，这里使用弧长比例近似 junction 位置。
    """
    poly = np.asarray(poly, dtype=float)
    t_frac = float(np.clip(t_frac, 0.0, 1.0))

    if len(poly) == 0:
        return np.array([0.0, 0.0]), np.array([1.0, 0.0])

    if len(poly) == 1:
        return poly[0].copy(), np.array([1.0, 0.0])

    total = float(cumlen[-1]) if len(cumlen) > 0 else 0.0

    if total < EPS:
        return poly[0].copy(), np.array([1.0, 0.0])

    target = t_frac * total

    idx = int(np.searchsorted(cumlen, target, side="right") - 1)
    idx = max(0, min(idx, len(poly) - 2))

    l0, l1 = cumlen[idx], cumlen[idx + 1]
    alpha = (target - l0) / max(l1 - l0, EPS)

    p0 = poly[idx]
    p1 = poly[idx + 1]

    point = (1.0 - alpha) * p0 + alpha * p1

    tangent = p1 - p0
    n = np.linalg.norm(tangent)

    if n < EPS:
        tangent = np.array([1.0, 0.0], dtype=float)
    else:
        tangent = tangent / n

    return point, tangent


def rotate_points(poly, theta):
    c = math.cos(theta)
    s = math.sin(theta)

    R = np.array([[c, -s], [s, c]], dtype=float)

    return poly @ R.T


def transform_polyline(poly, cx, cy, theta, scale):
    poly_r = rotate_points(poly, theta)
    out = poly_r * scale
    out[:, 0] += cx
    out[:, 1] += cy
    return out


def transform_point_and_tangent(local_point, local_tangent, cx, cy, theta, scale):
    c = math.cos(theta)
    s = math.sin(theta)

    R = np.array([[c, -s], [s, c]], dtype=float)

    pt = (R @ local_point) * scale + np.array([cx, cy], dtype=float)

    tg = R @ local_tangent
    n = np.linalg.norm(tg)

    if n < EPS:
        tg = np.array([1.0, 0.0], dtype=float)
    else:
        tg = tg / n

    return pt, tg


def bbox_from_points(points):
    pts = np.asarray(points, dtype=float)

    if len(pts) == 0:
        return [0.0, 0.0, 0.0, 0.0]

    return [
        float(np.min(pts[:, 0])),
        float(np.min(pts[:, 1])),
        float(np.max(pts[:, 0])),
        float(np.max(pts[:, 1])),
    ]


def bbox_union(boxes):
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]

    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def bbox_overflow_px(box):
    x0, y0, x1, y1 = box

    overflow = 0.0
    overflow = max(overflow, CANVAS_MARGIN - x0)
    overflow = max(overflow, CANVAS_MARGIN - y0)
    overflow = max(overflow, x1 - (CANVAS_SIZE - CANVAS_MARGIN))
    overflow = max(overflow, y1 - (CANVAS_SIZE - CANVAS_MARGIN))

    return max(0.0, float(overflow))


def connected_pair_set(edges):
    pairs = set()

    for e in edges:
        u = int(e["u"])
        v = int(e["v"])

        if u == v:
            continue

        a, b = min(u, v), max(u, v)
        pairs.add((a, b))

    return pairs


# =========================================================
# 🧩 初始布局
# =========================================================
def build_initial_layout(nodes, edges, rng=None, restart_idx=0):
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    n = len(nodes)

    if n == 0:
        return np.zeros((0, 4), dtype=float)

    centers = np.zeros((n, 2), dtype=float)

    if HAS_NETWORKX and n > 1:
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for e in edges:
            u, v = int(e["u"]), int(e["v"])

            if 0 <= u < n and 0 <= v < n and u != v:
                G.add_edge(u, v)

        if G.number_of_edges() > 0:
            pos = nx.spring_layout(
                G,
                seed=RANDOM_SEED + restart_idx,
                k=1.0 / math.sqrt(max(1, n)),
            )

            xs = np.array([pos[i][0] for i in range(n)], dtype=float)
            ys = np.array([pos[i][1] for i in range(n)], dtype=float)

            if np.max(xs) - np.min(xs) < EPS:
                xs = np.linspace(-1.0, 1.0, n)

            if np.max(ys) - np.min(ys) < EPS:
                ys = np.linspace(-1.0, 1.0, n)

            xs = (xs - np.min(xs)) / (np.max(xs) - np.min(xs) + EPS)
            ys = (ys - np.min(ys)) / (np.max(ys) - np.min(ys) + EPS)

            centers[:, 0] = CANVAS_SIZE * (0.26 + 0.48 * xs)
            centers[:, 1] = CANVAS_SIZE * (0.26 + 0.48 * ys)

        else:
            angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
            radius = CANVAS_SIZE * 0.18
            centers[:, 0] = GLYPH_CENTER[0] + radius * np.cos(angles)
            centers[:, 1] = GLYPH_CENTER[1] + radius * np.sin(angles)

    else:
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
        radius = CANVAS_SIZE * 0.18
        centers[:, 0] = GLYPH_CENTER[0] + radius * np.cos(angles)
        centers[:, 1] = GLYPH_CENTER[1] + radius * np.sin(angles)

    if restart_idx > 0:
        centers += rng.normal(0.0, 28.0, size=centers.shape)
        centers[:, 0] = np.clip(centers[:, 0], CANVAS_MARGIN, CANVAS_SIZE - CANVAS_MARGIN)
        centers[:, 1] = np.clip(centers[:, 1], CANVAS_MARGIN, CANVAS_SIZE - CANVAS_MARGIN)

    neighbor_map = defaultdict(list)

    for e in edges:
        u, v = int(e["u"]), int(e["v"])

        if 0 <= u < n and 0 <= v < n and u != v:
            neighbor_map[u].append(v)
            neighbor_map[v].append(u)

    params = np.zeros((n, 4), dtype=float)

    for i, node in enumerate(nodes):
        cx, cy = centers[i]

        local_arc = max(float(node["_local_arc_len"]), EPS)
        target_len_px = float(node["_target_length_px"])

        scale0 = target_len_px / local_arc
        scale0 = clamp(scale0, 10.0, CANVAS_SIZE * 0.95)

        theta0 = 0.0

        neighs = neighbor_map.get(i, [])

        if len(neighs) > 0:
            vecs = []

            for j in neighs:
                v = centers[j] - centers[i]
                nn = np.linalg.norm(v)

                if nn > EPS:
                    vecs.append(v / nn)

            if len(vecs) > 0:
                vv = np.mean(np.asarray(vecs), axis=0)
                theta0 = float(math.atan2(vv[1], vv[0]))

        if restart_idx > 0:
            theta0 += rng.normal(0.0, math.radians(35.0))

        theta0 = normalize_angle_pi(theta0)

        params[i] = [cx, cy, theta0, scale0]

    return params


# =========================================================
# ⚖️ Objective
# =========================================================
def unpack_params(x, n):
    return np.asarray(x, dtype=float).reshape(n, 4)


def compute_objective_and_metrics(x, cache, return_debug=False):
    nodes = cache["nodes"]
    edges = cache["edges"]
    n = cache["n"]
    weights = cache["weights"]
    connected_pairs = cache["connected_pairs"]

    params = unpack_params(x, n)

    transformed_polys = []
    transformed_boxes = []
    centers = []
    actual_lengths = []

    for i, node in enumerate(nodes):
        cx, cy, theta, scale = params[i]

        local_poly = node["_local_polyline"]
        local_arc = node["_local_arc_len"]

        poly = transform_polyline(local_poly, cx, cy, theta, scale)

        transformed_polys.append(poly)
        transformed_boxes.append(bbox_from_points(poly))
        centers.append(np.array([cx, cy], dtype=float))
        actual_lengths.append(float(local_arc * scale))

    centers = np.asarray(centers, dtype=float)

    glyph_bbox = bbox_union(transformed_boxes)
    gx0, gy0, gx1, gy1 = glyph_bbox

    g_w = gx1 - gx0
    g_h = gy1 - gy0

    g_cx = 0.5 * (gx0 + gx1)
    g_cy = 0.5 * (gy0 + gy1)

    # ------------------------------
    # junction / angle
    # ------------------------------
    junction_terms = []
    angle_terms = []
    edge_debug = []

    for e in edges:
        u = int(e["u"])
        v = int(e["v"])

        if not (0 <= u < n and 0 <= v < n):
            continue

        jtype = edge_type_name(e)

        t_u = float(e.get("t_u", 0.0))
        t_v = float(e.get("t_v", 0.0))

        p_u_local, tan_u_local = sample_polyline_point_and_tangent(
            nodes[u]["_local_polyline"],
            nodes[u]["_local_cumlen"],
            t_u,
        )

        p_v_local, tan_v_local = sample_polyline_point_and_tangent(
            nodes[v]["_local_polyline"],
            nodes[v]["_local_cumlen"],
            t_v,
        )

        cu = params[u]
        cv = params[v]

        p_u, tan_u = transform_point_and_tangent(
            p_u_local,
            tan_u_local,
            cu[0],
            cu[1],
            cu[2],
            cu[3],
        )

        p_v, tan_v = transform_point_and_tangent(
            p_v_local,
            tan_v_local,
            cv[0],
            cv[1],
            cv[2],
            cv[3],
        )

        dist_px = float(np.linalg.norm(p_u - p_v))
        dist_norm = dist_px / CANVAS_SIZE
        junction_terms.append(dist_norm ** 2)

        target_angle = get_target_angle_rad(e)

        pred_angle = None
        angle_diff = None

        if target_angle is not None:
            pred_angle = line_angle_between(tan_u, tan_v)
            angle_diff = smallest_angle_diff(pred_angle, target_angle)

            angle_type_weight = ANGLE_TYPE_WEIGHT.get(jtype, ANGLE_TYPE_WEIGHT["UNKNOWN"])
            angle_terms.append(angle_type_weight * (angle_diff / math.pi) ** 2)

        if return_debug:
            edge_debug.append({
                "u": int(u),
                "v": int(v),
                "j_type": jtype,
                "t_u": round(float(t_u), 6),
                "t_v": round(float(t_v), 6),
                "junction_distance_px": round(float(dist_px), 6),
                "target_angle_deg": None if target_angle is None else round(radians_to_degrees(target_angle), 6),
                "pred_angle_deg": None if pred_angle is None else round(radians_to_degrees(pred_angle), 6),
                "angle_diff_deg": None if angle_diff is None else round(radians_to_degrees(angle_diff), 6),
                "angle_type_weight": ANGLE_TYPE_WEIGHT.get(jtype, ANGLE_TYPE_WEIGHT["UNKNOWN"]),
                "junction_point_u": to_float_list(p_u, 4),
                "junction_point_v": to_float_list(p_v, 4),
            })

    l_junction = safe_mean(junction_terms, 0.0)
    l_angle = safe_mean(angle_terms, 0.0)

    # ------------------------------
    # length prior
    # ------------------------------
    length_terms = []
    node_debug = []

    for i, node in enumerate(nodes):
        target_len = float(node["_target_length_px"])
        actual_len = float(actual_lengths[i])

        rel = (actual_len - target_len) / (target_len + EPS)
        length_terms.append(rel ** 2)

        if return_debug:
            cx, cy, theta, scale = params[i]

            node_debug.append({
                "node_id": int(node.get("node_id", i)),
                "shape_code": int(node.get("shape_code", -1)),
                "width_token": int(node.get("width_token", 0)),
                "center": [round(float(cx), 4), round(float(cy), 4)],
                "rotation_deg": round(radians_to_degrees(theta), 4),
                "scale_px": round(float(scale), 4),
                "target_length_px": round(float(target_len), 4),
                "actual_length_px": round(float(actual_len), 4),
                "length_error_px": round(float(actual_len - target_len), 4),
            })

    l_length = safe_mean(length_terms, 0.0)

    # ------------------------------
    # bbox overflow
    # ------------------------------
    bbox_terms = []

    for poly in transformed_polys:
        overflow = 0.0

        for p in poly:
            x, y = p

            if x < CANVAS_MARGIN:
                overflow += ((CANVAS_MARGIN - x) / CANVAS_SIZE) ** 2

            if x > CANVAS_SIZE - CANVAS_MARGIN:
                overflow += ((x - (CANVAS_SIZE - CANVAS_MARGIN)) / CANVAS_SIZE) ** 2

            if y < CANVAS_MARGIN:
                overflow += ((CANVAS_MARGIN - y) / CANVAS_SIZE) ** 2

            if y > CANVAS_SIZE - CANVAS_MARGIN:
                overflow += ((y - (CANVAS_SIZE - CANVAS_MARGIN)) / CANVAS_SIZE) ** 2

        bbox_terms.append(overflow / max(1, len(poly)))

    l_bbox = safe_mean(bbox_terms, 0.0)

    # ------------------------------
    # center
    # ------------------------------
    dx = (g_cx - GLYPH_CENTER[0]) / (0.5 * CANVAS_SIZE)
    dy = (g_cy - GLYPH_CENTER[1]) / (0.5 * CANVAS_SIZE)
    l_center = dx * dx + dy * dy

    # ------------------------------
    # extent
    # ------------------------------
    gw_r = g_w / CANVAS_SIZE
    gh_r = g_h / CANVAS_SIZE

    low_pen = (
        max(0.0, TARGET_MIN_SPAN_RATIO - gw_r) ** 2
        + max(0.0, TARGET_MIN_SPAN_RATIO - gh_r) ** 2
    )

    high_pen = (
        max(0.0, gw_r - TARGET_MAX_SPAN_RATIO) ** 2
        + max(0.0, gh_r - TARGET_MAX_SPAN_RATIO) ** 2
    )

    l_extent = low_pen + high_pen

    # ------------------------------
    # repulsion / compactness
    # ------------------------------
    rep_terms = []
    compact_terms = []

    for i in range(n):
        for j in range(i + 1, n):
            ci = centers[i]
            cj = centers[j]

            d = float(np.linalg.norm(ci - cj))

            li = actual_lengths[i]
            lj = actual_lengths[j]

            if (i, j) not in connected_pairs:
                desired_min = 0.16 * (li + lj)

                if desired_min > EPS:
                    rep = max(0.0, desired_min - d) / desired_min
                    rep_terms.append(rep ** 2)

            else:
                desired_max = 0.90 * (li + lj)

                if desired_max > EPS:
                    comp = max(0.0, d - desired_max) / desired_max
                    compact_terms.append(comp ** 2)

    l_repulsion = safe_mean(rep_terms, 0.0)
    l_compact = safe_mean(compact_terms, 0.0)

    # ------------------------------
    # theta regularization
    # ------------------------------
    theta_terms = []

    for i, node in enumerate(nodes):
        theta0 = float(node.get("_init_theta", 0.0))
        theta = float(params[i][2])

        diff = smallest_angle_diff(theta, theta0)
        theta_terms.append((diff / math.pi) ** 2)

    l_theta = safe_mean(theta_terms, 0.0)

    total = (
        weights["junction"] * l_junction
        + weights["angle"] * l_angle
        + weights["length"] * l_length
        + weights["bbox"] * l_bbox
        + weights["center"] * l_center
        + weights["extent"] * l_extent
        + weights["repulsion"] * l_repulsion
        + weights["compact"] * l_compact
        + weights["theta_reg"] * l_theta
    )

    metrics = {
        "total": float(total),
        "junction": float(l_junction),
        "angle": float(l_angle),
        "length": float(l_length),
        "bbox": float(l_bbox),
        "center": float(l_center),
        "extent": float(l_extent),
        "repulsion": float(l_repulsion),
        "compact": float(l_compact),
        "theta_reg": float(l_theta),

        "glyph_bbox": [round(gx0, 4), round(gy0, 4), round(gx1, 4), round(gy1, 4)],
        "glyph_span_px": [round(g_w, 4), round(g_h, 4)],
        "glyph_center_px": [round(g_cx, 4), round(g_cy, 4)],
        "bbox_overflow_px": round(bbox_overflow_px(glyph_bbox), 6),
    }

    if return_debug:
        metrics["node_debug"] = node_debug
        metrics["edge_debug"] = edge_debug

    return float(total), metrics


# =========================================================
# ✅ Quality Report
# =========================================================
def evaluate_solve_quality(final_metrics):
    edge_debug = final_metrics.get("edge_debug", [])

    if not edge_debug:
        return {
            "quality_status": "bad_no_edges",
            "is_valid": False,
            "is_usable": False,

            "mean_junction_px": 999.0,
            "max_junction_px": 999.0,

            "mean_all_angle_diff_deg": 999.0,
            "max_all_angle_diff_deg": 999.0,

            "mean_tx_angle_diff_deg": 999.0,
            "max_tx_angle_diff_deg": 999.0,

            "bbox_overflow_px": 999.0,
            "quality_score": 999999.0,
        }

    junctions = [
        float(e.get("junction_distance_px", 999.0))
        for e in edge_debug
    ]

    all_angles = [
        float(e.get("angle_diff_deg", 0.0))
        for e in edge_debug
        if e.get("angle_diff_deg", None) is not None
    ]

    # 关键：T / X 角度才进入主质量判定
    tx_angles = [
        float(e.get("angle_diff_deg", 0.0))
        for e in edge_debug
        if e.get("angle_diff_deg", None) is not None
        and e.get("j_type", "") in ["T", "X"]
    ]

    mean_j = safe_mean(junctions, 999.0)
    max_j = safe_max(junctions, 999.0)

    mean_all_a = safe_mean(all_angles, 0.0)
    max_all_a = safe_max(all_angles, 0.0)

    # 如果没有 T/X angle，则认为 TX angle 约束满足
    mean_tx_a = safe_mean(tx_angles, 0.0)
    max_tx_a = safe_max(tx_angles, 0.0)

    bbox_ov = float(final_metrics.get("bbox_overflow_px", 999.0))

    good = QUALITY_THRESHOLDS["good"]
    usable = QUALITY_THRESHOLDS["usable"]

    is_valid = (
        mean_j <= good["mean_junction_px"]
        and max_j <= good["max_junction_px"]
        and mean_tx_a <= good["mean_tx_angle_diff_deg"]
        and max_tx_a <= good["max_tx_angle_diff_deg"]
        and bbox_ov <= good["bbox_overflow_px"]
    )

    is_usable = (
        mean_j <= usable["mean_junction_px"]
        and max_j <= usable["max_junction_px"]
        and mean_tx_a <= usable["mean_tx_angle_diff_deg"]
        and max_tx_a <= usable["max_tx_angle_diff_deg"]
        and bbox_ov <= usable["bbox_overflow_px"]
    )

    if is_valid:
        status = "good"
    elif is_usable:
        status = "usable_but_rough"
    else:
        status = "bad"

    # 质量分数仍保留 all angle 的弱影响，避免 E2E 完全乱飞
    quality_score = (
        mean_j * 1.0
        + max_j * 0.35
        + mean_tx_a * 0.10
        + max_tx_a * 0.05
        + mean_all_a * 0.015
        + max_all_a * 0.008
        + bbox_ov * 2.0
        + float(final_metrics.get("center", 0.0)) * 10.0
        + float(final_metrics.get("extent", 0.0)) * 10.0
    )

    return {
        "quality_status": status,
        "is_valid": bool(is_valid),
        "is_usable": bool(is_usable),

        "mean_junction_px": round(mean_j, 4),
        "max_junction_px": round(max_j, 4),

        "mean_all_angle_diff_deg": round(mean_all_a, 4),
        "max_all_angle_diff_deg": round(max_all_a, 4),

        "mean_tx_angle_diff_deg": round(mean_tx_a, 4),
        "max_tx_angle_diff_deg": round(max_tx_a, 4),

        "bbox_overflow_px": round(bbox_ov, 4),
        "quality_score": round(float(quality_score), 6),
    }


def candidate_quality_sort_key(c):
    q = c.get("quality_report", {})

    status_rank = {
        "good": 0,
        "usable_but_rough": 1,
        "bad": 2,
        "bad_no_edges": 3,
    }.get(q.get("quality_status", "bad"), 9)

    return (
        status_rank,
        q.get("quality_score", 999999.0),
        q.get("max_junction_px", 999.0),
        q.get("max_tx_angle_diff_deg", 999.0),
        c.get("solver_debug", {}).get("total", 999999.0),
    )


# =========================================================
# 🔧 单次优化
# =========================================================
def optimize_once(nodes, edges, x0, stage_weights_1, stage_weights_2):
    bounds = []

    for _ in range(len(nodes)):
        bounds.append((CANVAS_MARGIN, CANVAS_SIZE - CANVAS_MARGIN))  # cx
        bounds.append((CANVAS_MARGIN, CANVAS_SIZE - CANVAS_MARGIN))  # cy
        bounds.append((-math.pi, math.pi))                           # theta
        bounds.append((10.0, CANVAS_SIZE * 0.95))                    # scale

    connected_pairs = connected_pair_set(edges)

    # Stage 1
    cache1 = {
        "nodes": nodes,
        "edges": edges,
        "n": len(nodes),
        "weights": stage_weights_1,
        "connected_pairs": connected_pairs,
    }

    def objective_stage1(x):
        val, _ = compute_objective_and_metrics(x, cache1, return_debug=False)
        return val

    if RUN_TWO_STAGE_SOLVER:
        res1 = minimize(
            objective_stage1,
            x0,
            method=SOLVER_METHOD,
            bounds=bounds,
            options={
                "maxiter": SOLVER_MAXITER_STAGE1,
                "disp": False,
            },
        )
        x_mid = res1.x
    else:
        res1 = None
        x_mid = x0

    # Stage 2
    cache2 = {
        "nodes": nodes,
        "edges": edges,
        "n": len(nodes),
        "weights": stage_weights_2,
        "connected_pairs": connected_pairs,
    }

    def objective_stage2(x):
        val, _ = compute_objective_and_metrics(x, cache2, return_debug=False)
        return val

    res2 = minimize(
        objective_stage2,
        x_mid,
        method=SOLVER_METHOD,
        bounds=bounds,
        options={
            "maxiter": SOLVER_MAXITER_STAGE2,
            "disp": False,
        },
    )

    final_total, final_metrics = compute_objective_and_metrics(
        res2.x,
        cache2,
        return_debug=True,
    )

    quality_report = evaluate_solve_quality(final_metrics)

    return {
        "x": res2.x,
        "res_stage1": res1,
        "res_stage2": res2,
        "final_total": final_total,
        "final_metrics": final_metrics,
        "quality_report": quality_report,
    }


# =========================================================
# 🚀 单个 candidate 求解
# =========================================================
def solve_one_candidate_core(candidate, idx, seed):
    rng = np.random.default_rng(seed)

    nodes_raw = candidate.get("nodes", [])
    edges_raw = get_candidate_edges(candidate)

    gid = candidate.get("generated_glyph_id", f"glyph_candidate_{idx:05d}")

    if len(nodes_raw) == 0:
        return {
            "candidate_index": idx,
            "generated_glyph_id": gid,
            "ok": False,
            "error": "empty_nodes",
            "result": None,
            "summary_line": f"{gid}: skipped empty nodes",
        }

    if len(edges_raw) == 0:
        return {
            "candidate_index": idx,
            "generated_glyph_id": gid,
            "ok": False,
            "error": "empty_edges",
            "result": None,
            "summary_line": f"{gid}: skipped empty edges",
        }

    nodes = []

    for node in nodes_raw:
        nn = dict(node)

        local_poly = get_local_polyline(nn)
        local_arc = max(polyline_arc_length(local_poly), EPS)
        local_cum = polyline_cumulative_lengths(local_poly)

        len_prior_norm = float(nn.get("solver_priors", {}).get("length_prior_norm", 0.35))
        target_length_px = float(len_prior_norm * CANVAS_SIZE)
        target_length_px = clamp(target_length_px, 20.0, CANVAS_SIZE * 0.70)

        nn["_local_polyline"] = local_poly
        nn["_local_arc_len"] = local_arc
        nn["_local_cumlen"] = local_cum
        nn["_target_length_px"] = target_length_px

        nodes.append(nn)

    restart_results = []

    for restart_idx in range(NUM_RESTARTS):
        init_params = build_initial_layout(
            nodes,
            edges_raw,
            rng=rng,
            restart_idx=restart_idx,
        )

        for i in range(len(nodes)):
            nodes[i]["_init_theta"] = float(init_params[i][2])

        x0 = init_params.reshape(-1)

        init_cache = {
            "nodes": nodes,
            "edges": edges_raw,
            "n": len(nodes),
            "weights": LOSS_WEIGHTS_STAGE2,
            "connected_pairs": connected_pair_set(edges_raw),
        }

        init_total, init_metrics = compute_objective_and_metrics(
            x0,
            init_cache,
            return_debug=False,
        )

        opt_result = optimize_once(
            nodes=nodes,
            edges=edges_raw,
            x0=x0,
            stage_weights_1=LOSS_WEIGHTS_STAGE1,
            stage_weights_2=LOSS_WEIGHTS_STAGE2,
        )

        opt_result["restart_idx"] = restart_idx
        opt_result["initial_total"] = init_total
        opt_result["initial_metrics"] = init_metrics

        restart_results.append(opt_result)

        q = opt_result["quality_report"]

        # 已经非常好就提前停止
        if q["is_valid"] and q["quality_score"] < 3.0:
            break

    best = sorted(
        restart_results,
        key=lambda r: (
            0 if r["quality_report"]["is_valid"] else 1,
            0 if r["quality_report"]["is_usable"] else 1,
            r["quality_report"]["quality_score"],
            r["final_total"],
        )
    )[0]

    x_star = best["x"]
    final_metrics = best["final_metrics"]
    quality_report = best["quality_report"]

    res_stage1 = best["res_stage1"]
    res_stage2 = best["res_stage2"]

    params = unpack_params(x_star, len(nodes))

    solved_nodes = []

    for i, node in enumerate(nodes):
        cx, cy, theta, scale = params[i]
        poly_px = transform_polyline(node["_local_polyline"], cx, cy, theta, scale)

        solved_nodes.append({
            "node_id": int(node.get("node_id", i)),
            "shape_code": int(node.get("shape_code", -1)),
            "variant_id": int(node.get("variant_id", 0)),
            "width_token": int(node.get("width_token", 0)),

            "grammar_role": node.get("grammar_role", {}),
            "primitive_assignment": node.get("primitive_assignment", {}),
            "primitive_ref": node.get("primitive_ref", {}),
            "solver_priors": node.get("solver_priors", {}),

            "center_px": [round(float(cx), 6), round(float(cy), 6)],
            "rotation_rad": round(float(theta), 8),
            "rotation_deg": round(radians_to_degrees(theta), 6),
            "scale_px": round(float(scale), 6),

            "target_length_px": round(float(node["_target_length_px"]), 6),
            "actual_length_px": round(float(node["_local_arc_len"] * scale), 6),

            "solved_polyline_px": to_float_list(poly_px, 6),
            "solved_bbox_px": to_float_list(bbox_from_points(poly_px), 6),
        })

    # restart summary
    restart_summary = []

    for r in restart_results:
        q = r["quality_report"]
        restart_summary.append({
            "restart_idx": int(r["restart_idx"]),
            "initial_total": round(float(r["initial_total"]), 6),
            "final_total": round(float(r["final_total"]), 6),
            "quality_status": q["quality_status"],
            "quality_score": q["quality_score"],
            "max_junction_px": q["max_junction_px"],
            "max_tx_angle_diff_deg": q["max_tx_angle_diff_deg"],
            "max_all_angle_diff_deg": q["max_all_angle_diff_deg"],
        })

    out_candidate = {
        "candidate_index": int(idx),
        "generated_glyph_id": gid,
        "generation_stage": "constraint_solved",

        "source_signature": candidate.get("source_signature", ""),
        "source_file": candidate.get("source_file", ""),
        "glyph_uid": candidate.get("glyph_uid", ""),
        "hex_key": candidate.get("hex_key", ""),
        "char": candidate.get("char", ""),
        "derivation": candidate.get("derivation", ""),

        "topology": candidate.get("topology", {}),
        "edges": edges_raw,

        "solved_nodes": solved_nodes,

        "solve_report": {
            "best_restart_idx": int(best["restart_idx"]),
            "restart_summary": restart_summary,

            "stage1_success": None if res_stage1 is None else bool(res_stage1.success),
            "stage1_status": None if res_stage1 is None else int(res_stage1.status),
            "stage1_message": None if res_stage1 is None else str(res_stage1.message),
            "stage1_nit": None if res_stage1 is None else int(getattr(res_stage1, "nit", -1)),

            "stage2_success": bool(res_stage2.success),
            "stage2_status": int(res_stage2.status),
            "stage2_message": str(res_stage2.message),
            "stage2_nit": int(getattr(res_stage2, "nit", -1)),

            "initial_total": float(best["initial_total"]),
            "final_total": float(best["final_total"]),
        },

        "quality_report": quality_report,
        "solver_debug": final_metrics,
    }

    summary_line = (
        f"{gid}: "
        f"N={len(nodes_raw)} E={len(edges_raw)} "
        f"status={quality_report['quality_status']} "
        f"score={quality_report['quality_score']} "
        f"maxJ={quality_report['max_junction_px']}px "
        f"maxTXA={quality_report['max_tx_angle_diff_deg']}° "
        f"best_restart={best['restart_idx']}"
    )

    return {
        "candidate_index": idx,
        "generated_glyph_id": gid,
        "ok": True,
        "error": None,
        "result": out_candidate,
        "summary_line": summary_line,
    }


def worker_solve_one(args):
    idx, candidate, seed = args

    try:
        return solve_one_candidate_core(candidate, idx, seed)

    except Exception as e:
        return {
            "candidate_index": idx,
            "generated_glyph_id": candidate.get("generated_glyph_id", f"glyph_candidate_{idx:05d}"),
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "result": None,
            "summary_line": f"candidate #{idx}: ERROR {str(e)}",
        }


# =========================================================
# 📊 输入诊断
# =========================================================
def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def corpus_diagnostics(candidates):
    print_header("📊 Constraint Solver Input Diagnostics")

    print("[Input File]")
    print(f"  {INPUT_FILE}")

    print("\n[Basic Info]")
    print(f"  candidate_count: {len(candidates)}")

    first_keys = []
    first_topo_keys = []

    if len(candidates) > 0:
        first_keys = list(candidates[0].keys())

        if isinstance(candidates[0].get("topology", {}), dict):
            first_topo_keys = list(candidates[0]["topology"].keys())

    print("\n[Key Check]")
    print(f"  first candidate keys: {first_keys}")
    print(f"  first topology keys:  {first_topo_keys}")

    stroke_hist = Counter()
    edge_hist = Counter()
    edge_type_hist = Counter()
    primitive_ok_count = 0
    total_edges = 0

    for c in candidates:
        nodes = c.get("nodes", [])
        edges = get_candidate_edges(c)

        stroke_hist[len(nodes)] += 1
        edge_hist[len(edges)] += 1
        total_edges += len(edges)

        ok = True

        for n in nodes:
            if "primitive_ref" not in n:
                ok = False
                break

            if "primitive_polyline_local_norm" not in n.get("primitive_ref", {}):
                ok = False
                break

        if ok:
            primitive_ok_count += 1

        for e in edges:
            edge_type_hist[edge_type_name(e)] += 1

    print(f"\n[Topology Edge Check]")
    print(f"  total readable edges: {total_edges}")

    if total_edges == 0:
        raise RuntimeError(
            "❌ 所有 candidate 的 edge 都读成了 0。"
            "请检查 edge 字段位置，当前标准为 topology.positive_edges_undirected。"
        )

    print(f"\n[Primitive Check]")
    print(f"  candidates with primitive polyline: {primitive_ok_count}/{len(candidates)}")

    print("\n[Stroke Count Distribution]")
    for k, v in sorted(stroke_hist.items()):
        print(f"  {k}: {v}")

    print("\n[Edge Count Distribution]")
    for k, v in sorted(edge_hist.items()):
        print(f"  {k}: {v}")

    print("\n[Edge Type Distribution]")
    for k, v in edge_type_hist.items():
        print(f"  {k}: {v}")

    print("\n[Feasibility Check]")
    print("  ✅ 已检测到 topology.positive_edges_undirected，solver 可以进行拓扑约束求解。")


def print_solver_config():
    print_header("🎚️ Constraint Solver Config")

    print(f"  parallel_solve: {PARALLEL_SOLVE}")
    print(f"  num_workers:    {NUM_WORKERS}")
    print(f"  num_restarts:   {NUM_RESTARTS}")
    print(f"  method:         {SOLVER_METHOD}")
    print(f"  stage1_iter:    {SOLVER_MAXITER_STAGE1}")
    print(f"  stage2_iter:    {SOLVER_MAXITER_STAGE2}")
    print(f"  canvas_size:    {CANVAS_SIZE}")

    print("\n[ANGLE_TYPE_WEIGHT]")
    for k, v in ANGLE_TYPE_WEIGHT.items():
        print(f"  {k}: {v}")

    print("\n[Quality Thresholds]")
    print(json.dumps(QUALITY_THRESHOLDS, indent=2, ensure_ascii=False))


# =========================================================
# 📦 输出汇总
# =========================================================
def summarize_solved_outputs(solved):
    print_header("📦 Constraint Solver Output Summary")

    if len(solved) == 0:
        print("  ⚠️ 没有 solved 输出。")
        return {}

    status_hist = Counter()
    valid_count = 0
    usable_count = 0
    scipy_success_count = 0

    q_scores = []
    mean_js = []
    max_js = []
    mean_tx_as = []
    max_tx_as = []
    mean_all_as = []
    max_all_as = []
    totals = []

    edge_count_hist = Counter()
    node_count_hist = Counter()

    for c in solved:
        q = c.get("quality_report", {})

        status = q.get("quality_status", "unknown")
        status_hist[status] += 1

        if q.get("is_valid", False):
            valid_count += 1

        if q.get("is_usable", False):
            usable_count += 1

        if c.get("solve_report", {}).get("stage2_success", False):
            scipy_success_count += 1

        q_scores.append(float(q.get("quality_score", 999999.0)))
        mean_js.append(float(q.get("mean_junction_px", 999.0)))
        max_js.append(float(q.get("max_junction_px", 999.0)))

        mean_tx_as.append(float(q.get("mean_tx_angle_diff_deg", 999.0)))
        max_tx_as.append(float(q.get("max_tx_angle_diff_deg", 999.0)))

        mean_all_as.append(float(q.get("mean_all_angle_diff_deg", 999.0)))
        max_all_as.append(float(q.get("max_all_angle_diff_deg", 999.0)))

        totals.append(float(c.get("solver_debug", {}).get("total", 999999.0)))

        edge_count_hist[len(c.get("edges", []))] += 1
        node_count_hist[len(c.get("solved_nodes", []))] += 1

    solved_sorted = sorted(solved, key=candidate_quality_sort_key)

    summary = {
        "solved_count": len(solved),
        "scipy_success_count": scipy_success_count,
        "scipy_success_ratio": round(scipy_success_count / max(1, len(solved)), 4),

        "quality_valid_count": valid_count,
        "quality_valid_ratio": round(valid_count / max(1, len(solved)), 4),

        "quality_usable_count": usable_count,
        "quality_usable_ratio": round(usable_count / max(1, len(solved)), 4),

        "quality_status_hist": dict(status_hist),

        "avg_quality_score": round(float(np.mean(q_scores)), 6),
        "avg_total": round(float(np.mean(totals)), 6),

        "avg_mean_junction_px": round(float(np.mean(mean_js)), 6),
        "avg_max_junction_px": round(float(np.mean(max_js)), 6),

        "avg_mean_tx_angle_diff_deg": round(float(np.mean(mean_tx_as)), 6),
        "avg_max_tx_angle_diff_deg": round(float(np.mean(max_tx_as)), 6),

        "avg_mean_all_angle_diff_deg": round(float(np.mean(mean_all_as)), 6),
        "avg_max_all_angle_diff_deg": round(float(np.mean(max_all_as)), 6),

        "node_count_hist": {str(k): int(v) for k, v in node_count_hist.items()},
        "edge_count_hist": {str(k): int(v) for k, v in edge_count_hist.items()},

        "top_best_ids": [
            {
                "generated_glyph_id": c.get("generated_glyph_id", ""),
                "quality_status": c.get("quality_report", {}).get("quality_status", ""),
                "quality_score": c.get("quality_report", {}).get("quality_score", 999999.0),
                "max_junction_px": c.get("quality_report", {}).get("max_junction_px", 999.0),
                "max_tx_angle_diff_deg": c.get("quality_report", {}).get("max_tx_angle_diff_deg", 999.0),
                "max_all_angle_diff_deg": c.get("quality_report", {}).get("max_all_angle_diff_deg", 999.0),
            }
            for c in solved_sorted[:10]
        ],
    }

    print(f"  solved_count: {summary['solved_count']}")
    print(f"  scipy_success_count: {summary['scipy_success_count']}")
    print(f"  scipy_success_ratio: {summary['scipy_success_ratio']}")
    print(f"  quality_valid_count: {summary['quality_valid_count']}")
    print(f"  quality_valid_ratio: {summary['quality_valid_ratio']}")
    print(f"  quality_usable_count: {summary['quality_usable_count']}")
    print(f"  quality_usable_ratio: {summary['quality_usable_ratio']}")

    print("\n[Quality Status Hist]")
    for k, v in status_hist.items():
        print(f"  {k}: {v}")

    print("\n[Average Metrics]")
    print(f"  avg_quality_score: {summary['avg_quality_score']}")
    print(f"  avg_total: {summary['avg_total']}")
    print(f"  avg_mean_junction_px: {summary['avg_mean_junction_px']}")
    print(f"  avg_max_junction_px: {summary['avg_max_junction_px']}")
    print(f"  avg_mean_tx_angle_diff_deg: {summary['avg_mean_tx_angle_diff_deg']}")
    print(f"  avg_max_tx_angle_diff_deg: {summary['avg_max_tx_angle_diff_deg']}")
    print(f"  avg_mean_all_angle_diff_deg: {summary['avg_mean_all_angle_diff_deg']}")
    print(f"  avg_max_all_angle_diff_deg: {summary['avg_max_all_angle_diff_deg']}")

    print("\n[Top-10 Best Glyphs]")
    for item in summary["top_best_ids"]:
        print(
            f"  {item['generated_glyph_id']}: "
            f"status={item['quality_status']} "
            f"score={item['quality_score']} "
            f"maxJ={item['max_junction_px']}px "
            f"maxTXA={item['max_tx_angle_diff_deg']}° "
            f"maxAllA={item['max_all_angle_diff_deg']}°"
        )

    return summary


# =========================================================
# 🖼️ 渲染
# =========================================================
def render_candidate_preview(solved_candidate, out_path):
    nodes = solved_candidate["solved_nodes"]

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))

    colors = [
        "#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA",
        "#00897B", "#6D4C41", "#3949AB", "#D81B60", "#7CB342",
        "#546E7A", "#F4511E"
    ]

    for i, node in enumerate(nodes):
        poly = np.asarray(node["solved_polyline_px"], dtype=float)
        c = colors[i % len(colors)]

        lw = 1.8 + 1.2 * float(node.get("width_token", 0))

        ax.plot(poly[:, 0], poly[:, 1], color=c, lw=lw, alpha=0.95)

        p0 = poly[0]
        p1 = poly[-1]

        ax.scatter([p0[0]], [p0[1]], color="green", s=26, zorder=5)
        ax.scatter([p1[0]], [p1[1]], color="red", s=26, zorder=5)

        center = np.asarray(node["center_px"], dtype=float)

        ax.text(
            center[0] + 4,
            center[1] - 4,
            f"N{node['node_id']}",
            fontsize=9,
            color=c,
            fontweight="bold",
        )

    for e in solved_candidate.get("solver_debug", {}).get("edge_debug", []):
        pu = np.asarray(e["junction_point_u"], dtype=float)
        pv = np.asarray(e["junction_point_v"], dtype=float)
        pm = 0.5 * (pu + pv)

        et = e.get("j_type", "?")
        dist = e.get("junction_distance_px", 0.0)

        ax.plot(
            [pu[0], pv[0]],
            [pu[1], pv[1]],
            "--",
            color="#333333",
            lw=0.8,
            alpha=0.65,
        )

        ax.scatter([pm[0]], [pm[1]], color="#111111", s=12, zorder=7)

        ax.text(
            pm[0] + 2,
            pm[1] + 2,
            f"{et} {dist:.1f}px",
            fontsize=7,
            color="#222222",
        )

    q = solved_candidate.get("quality_report", {})

    title = (
        f"{solved_candidate.get('generated_glyph_id', '')} | {q.get('quality_status', '')}\n"
        f"score={q.get('quality_score', 0)} "
        f"maxJ={q.get('max_junction_px', 0)}px "
        f"maxTXA={q.get('max_tx_angle_diff_deg', 0)}°"
    )

    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, CANVAS_SIZE)
    ax.set_ylim(CANVAS_SIZE, 0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def render_previews(solved):
    print_header("🖼️ Rendering Preview Images")

    ensure_dir(BEST_PREVIEW_DIR)
    ensure_dir(ROUGH_PREVIEW_DIR)
    ensure_dir(BAD_PREVIEW_DIR)

    solved_sorted = sorted(solved, key=candidate_quality_sort_key)

    good = [
        c for c in solved_sorted
        if c.get("quality_report", {}).get("quality_status") == "good"
    ]

    rough = [
        c for c in solved_sorted
        if c.get("quality_report", {}).get("quality_status") == "usable_but_rough"
    ]

    bad = [
        c for c in solved_sorted
        if c.get("quality_report", {}).get("quality_status") in ["bad", "bad_no_edges"]
    ]

    for i, c in enumerate(good[:PREVIEW_COUNT_BEST]):
        out_path = os.path.join(BEST_PREVIEW_DIR, f"best_{i:03d}_{c['generated_glyph_id']}.png")
        render_candidate_preview(c, out_path)
        print(f"  best saved: {out_path}")

    for i, c in enumerate(rough[:PREVIEW_COUNT_ROUGH]):
        out_path = os.path.join(ROUGH_PREVIEW_DIR, f"rough_{i:03d}_{c['generated_glyph_id']}.png")
        render_candidate_preview(c, out_path)
        print(f"  rough saved: {out_path}")

    for i, c in enumerate(bad[:PREVIEW_COUNT_BAD]):
        out_path = os.path.join(BAD_PREVIEW_DIR, f"bad_{i:03d}_{c['generated_glyph_id']}.png")
        render_candidate_preview(c, out_path)
        print(f"  bad saved: {out_path}")

    print("\n[Preview Summary]")
    print(f"  good previews:  {min(len(good), PREVIEW_COUNT_BEST)}")
    print(f"  rough previews: {min(len(rough), PREVIEW_COUNT_ROUGH)}")
    print(f"  bad previews:   {min(len(bad), PREVIEW_COUNT_BAD)}")


# =========================================================
# 🧵 并行主求解
# =========================================================
def solve_all_candidates(candidates):
    tasks = []

    for idx, candidate in enumerate(candidates):
        seed = RANDOM_SEED + idx * 1009
        tasks.append((idx, candidate, seed))

    results = []

    t0 = time.time()

    if PARALLEL_SOLVE and NUM_WORKERS > 1:
        print_header("🚀 Start Parallel Constraint Solving")
        print(f"  workers: {NUM_WORKERS}")
        print(f"  tasks:   {len(tasks)}")

        done_count = 0

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [
                executor.submit(worker_solve_one, task)
                for task in tasks
            ]

            for fut in as_completed(futures):
                item = fut.result()
                results.append(item)
                done_count += 1

                if PRINT_MAIN_PROGRESS and (
                    done_count % PROGRESS_EVERY == 0
                    or done_count == len(tasks)
                ):
                    elapsed = time.time() - t0
                    speed = done_count / max(elapsed, 1e-6)
                    remain = (len(tasks) - done_count) / max(speed, 1e-6)

                    print(
                        f"  progress {done_count}/{len(tasks)} "
                        f"| {speed:.2f} cand/s "
                        f"| ETA {remain:.1f}s "
                        f"| last: {item.get('summary_line', '')}"
                    )

    else:
        print_header("🚀 Start Serial Constraint Solving")

        for i, task in enumerate(tasks):
            item = worker_solve_one(task)
            results.append(item)

            if PRINT_MAIN_PROGRESS:
                print(f"  {i + 1}/{len(tasks)} | {item.get('summary_line', '')}")

    elapsed = time.time() - t0

    print_header("⏱️ Solve Time")
    print(f"  total time: {elapsed:.2f}s")
    print(f"  avg per candidate: {elapsed / max(1, len(tasks)):.3f}s")
    print(f"  parallel: {PARALLEL_SOLVE}")
    print(f"  workers: {NUM_WORKERS}")

    # 排序回原始顺序
    results = sorted(results, key=lambda x: x.get("candidate_index", 10**9))

    solved = []
    errors = []

    for r in results:
        if r.get("ok") and r.get("result") is not None:
            solved.append(r["result"])
        else:
            errors.append(r)

    if errors:
        print_header("⚠️ Solve Errors / Skips")
        for e in errors[:20]:
            print(f"  idx={e.get('candidate_index')} gid={e.get('generated_glyph_id')} error={e.get('error')}")

        if len(errors) > 20:
            print(f"  ... plus {len(errors) - 20} more")

    return solved, errors, elapsed


# =========================================================
# 🎬 主程序
# =========================================================
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ensure_dir(PREVIEW_DIR)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return

    data = load_json(INPUT_FILE)
    candidates = get_candidate_list(data)

    if len(candidates) == 0:
        print("❌ 输入文件中没有 candidate 数据。")
        return

    corpus_diagnostics(candidates)
    print_solver_config()

    if MAX_SOLVE_COUNT is not None:
        candidates = candidates[:MAX_SOLVE_COUNT]
        print(f"\n⚠️ 调试模式：仅求解前 {len(candidates)} 个 candidate")

    solved, errors, elapsed = solve_all_candidates(candidates)

    solved_sorted = sorted(solved, key=candidate_quality_sort_key)

    output_summary = summarize_solved_outputs(solved_sorted)

    valid_solved = [
        c for c in solved_sorted
        if c.get("quality_report", {}).get("is_valid", False)
    ]

    usable_solved = [
        c for c in solved_sorted
        if c.get("quality_report", {}).get("is_usable", False)
    ]

    rejected_solved = [
        c for c in solved_sorted
        if not c.get("quality_report", {}).get("is_usable", False)
    ]

    render_previews(solved_sorted)

    output_obj = {
        "schema_version": "constraint_solver_v3_parallel_e2e_angle_relaxed",

        "description": (
            "Constraint-based geometric solver for topology + stroke primitive glyph candidates. "
            "This version uses multiprocessing, multi-start two-stage optimization, "
            "relaxes E2E angle in objective/quality filtering, and ranks outputs by geometry quality."
        ),

        "source_input_file": INPUT_FILE,
        "output_file": OUTPUT_FILE,

        "preview_dir": PREVIEW_DIR,
        "best_preview_dir": BEST_PREVIEW_DIR,
        "rough_preview_dir": ROUGH_PREVIEW_DIR,
        "bad_preview_dir": BAD_PREVIEW_DIR,

        "random_seed": RANDOM_SEED,
        "canvas_size": CANVAS_SIZE,
        "canvas_margin": CANVAS_MARGIN,

        "parallel_config": {
            "parallel_solve": PARALLEL_SOLVE,
            "num_workers": NUM_WORKERS,
            "chunksize": CHUNKSIZE,
            "elapsed_seconds": round(float(elapsed), 4),
            "avg_seconds_per_candidate": round(float(elapsed / max(1, len(candidates))), 6),
            "env_threads": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            },
        },

        "solver_config": {
            "method": SOLVER_METHOD,
            "num_restarts": NUM_RESTARTS,
            "run_two_stage_solver": RUN_TWO_STAGE_SOLVER,
            "maxiter_stage1": SOLVER_MAXITER_STAGE1,
            "maxiter_stage2": SOLVER_MAXITER_STAGE2,
            "max_solve_count": MAX_SOLVE_COUNT,
            "target_min_span_ratio": TARGET_MIN_SPAN_RATIO,
            "target_max_span_ratio": TARGET_MAX_SPAN_RATIO,
        },

        "angle_type_weight": ANGLE_TYPE_WEIGHT,
        "loss_weights_stage1": LOSS_WEIGHTS_STAGE1,
        "loss_weights_stage2": LOSS_WEIGHTS_STAGE2,
        "quality_thresholds": QUALITY_THRESHOLDS,

        "output_summary": output_summary,

        "solved_count": len(solved_sorted),
        "valid_count": len(valid_solved),
        "usable_count": len(usable_solved),
        "rejected_count": len(rejected_solved),
        "error_count": len(errors),

        "solved_glyph_candidates": solved_sorted,
        "valid_solved_glyph_candidates": valid_solved,
        "usable_solved_glyph_candidates": usable_solved,
        "rejected_solved_glyph_candidates": rejected_solved,
        "solve_errors": errors,
    }

    save_json(output_obj, OUTPUT_FILE)

    print_header("💾 Saved")
    print(f"  solved json: {OUTPUT_FILE}")
    print(f"  previews:    {PREVIEW_DIR}")

    print("\n📌 下一步：")
    print("  1. 优先打开 solver_previews/best 看 good 样本")
    print("  2. 如果 CPU 仍然很低，确认是不是 NUM_WORKERS 太小，或 macOS 活动监视器显示的是单进程占比")
    print("  3. aesthetic_scorer.py 后续优先读取 valid_solved_glyph_candidates")
    print("  4. 如果要进一步加速，可以把 NUM_RESTARTS 从 3 改 2，或先用 graph_grammar_sampler 多生成、solver 只过滤")


if __name__ == "__main__":
    main()