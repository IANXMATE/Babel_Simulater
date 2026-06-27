# -*- coding: utf-8 -*-
"""
dtg_transformer_v4_inference.py

Junction-Anchor DTG-Transformer inference script.

论文方案版：
    用端到端 Neural Geometry Solver / Graph Transformer
    直接替代 constraint_solver.py。

输入：
    glyph_candidates_with_primitives.json
    dtg_transformer_v4_best.pt 或 dtg_transformer_v4.pt

输出：
    solved_glyph_candidates_v5_robust.json
    dtg_transformer_v5_robust_inference_report.json

核心特征：
    - 不调用 constraint_solver.py
    - 不使用 L-BFGS / Adam per candidate
    - 不做 iterative post-processing
    - 只做一次 Graph Transformer forward
    - forward 内含一次 closed-form differentiable Bézier anchor projection

推荐 pipeline：
    layout_aware_graph_sampler.py
    -> stroke_primitive_sampler.py
    -> dtg_transformer_v4_inference.py
    -> score_solved_glyphs.py
"""

import os
import sys
import json
import math
import time
import copy
import random
from collections import Counter, defaultdict

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as e:
    raise RuntimeError("dtg_transformer_v4_inference.py requires PyTorch.") from e


# =========================================================
# 0. Paths
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(SCRIPT_DIR, "glyph_candidates_with_primitives.json")

# 优先使用 best，如果不存在再用普通 v4。
V5_BEST_MODEL_FILE = os.path.join(SCRIPT_DIR, "dtg_transformer_v5_robust_best.pt")
V5_MODEL_FILE = os.path.join(SCRIPT_DIR, "dtg_transformer_v5_robust.pt")
V5_FINAL_MODEL_FILE = os.path.join(SCRIPT_DIR, "dtg_transformer_v5_robust_final.pt")
BEST_MODEL_FILE = os.path.join(SCRIPT_DIR, "dtg_transformer_v4_best.pt")
MODEL_FILE = os.path.join(SCRIPT_DIR, "dtg_transformer_v4.pt")
FINAL_MODEL_FILE = os.path.join(SCRIPT_DIR, "dtg_transformer_v4_final.pt")

OUTPUT_FILE = os.path.join(SCRIPT_DIR, "solved_glyph_candidates_v5_robust.json")
OUTPUT_REPORT_FILE = os.path.join(SCRIPT_DIR, "dtg_transformer_v5_robust_inference_report.json")


# =========================================================
# 1. Config: 必须和 train_dtg_transformer_v4.py 保持一致
# =========================================================
RANDOM_SEED = 42
DEVICE_MODE = "auto"  # auto / cuda / mps / cpu
CANVAS_SIZE = 400.0

MAX_NODES = 8
MAX_EDGES = 24
MAX_SHAPE_CODE = 160
MAX_WIDTH_TOKEN = 8

D_MODEL = 128
D_EDGE = 48
NUM_LAYERS = 4
NUM_HEADS = 4
DROPOUT = 0.10

USE_RESIDUAL_DECODER = True
MAX_CONTROL_DELTA = 0.35

# v4 anchor decoder
ANCHOR_PROJECT_STRENGTH = 1.00
ANCHOR_PRIOR_BLEND = 0.05
ANCHOR_MAX_DELTA = 0.55

BATCH_SIZE_INFER = 128
PRINT_EVERY = 50

GOOD_MAX_JUNCTION_PX = 1.20
USABLE_MAX_JUNCTION_PX = 6.0

NUM_RENDER_SAMPLES = 32

# v2 inference fix:
# 训练时用了 topology canonicalization；推理时也要把 raw candidate edge/t 对齐到同一格式。
USE_INFERENCE_TOPO_CANONICALIZATION = True
CANONICALIZE_SAMPLE_N = 64
CANONICALIZE_INTERNAL_EPS = 0.05

# v3 inference fix:
# 如果 prior topology 已经闭合，就不要让 neural decoder 把它拉坏。
# 这不是 solver，只是选择 topology error 更低的几何输出。
USE_SAFE_GEOMETRY_SELECTION = True
SAFE_PRIOR_ALREADY_GOOD_PX = 1.20
SAFE_REQUIRE_MODEL_IMPROVE_BY_PX = 0.25

JTYPE_TO_ID = {
    "NONE": 0,
    "E2E": 1,
    "T": 2,
    "X": 3,
}
ID_TO_JTYPE = {v: k for k, v in JTYPE_TO_ID.items()}


# =========================================================
# 2. Basic utils
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device():
    if DEVICE_MODE == "cuda":
        return torch.device("cuda")
    if DEVICE_MODE == "mps":
        return torch.device("mps")
    if DEVICE_MODE == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_json(path, required=True):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


def angle_wrap(a):
    a = float(a)
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


def parse_jtype(raw):
    if raw is None:
        return "NONE"
    s = str(raw).upper()

    if "E2E" in s or "ENDPOINT_TO_ENDPOINT" in s or "END_TO_END" in s:
        return "E2E"
    if "T_ATTACH" in s or "T-JUNCTION" in s or "TJUNCTION" in s or s == "T":
        return "T"
    if s == "X" or "CROSS" in s or "INTERSECT" in s:
        return "X"

    # 兼容 j_type_idx
    idx = safe_int(raw, -1)
    if idx == 1:
        return "E2E"
    if idx == 2:
        # 注意：v4 训练时 2 是 T，不是旧 neural_constraint_solver 的 X
        return "T"
    if idx == 3:
        return "X"

    return "NONE"


def stable_sigmoid(x):
    x = np.clip(x, -50, 50)
    return float(1.0 / (1.0 + np.exp(-x)))


# =========================================================
# 3. JSON candidate parsing
# =========================================================
def get_candidate_list(data):
    if isinstance(data, list):
        return data

    keys = [
        "glyph_candidates_with_primitives",
        "glyph_candidates",
        "candidates",
        "sampled_candidates",
        "solved_glyph_candidates",
        "ranked_solved_glyph_candidates_by_combined",
    ]

    if isinstance(data, dict):
        for k in keys:
            if isinstance(data.get(k), list):
                return data[k]

        for k, v in data.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                print(f"[WARN] using unknown candidate key: {k}")
                return v

    return []


def get_candidate_id(candidate, idx=0):
    for k in ["generated_glyph_id", "glyph_candidate_id", "candidate_id", "sample_id", "grammar_sample_id"]:
        if isinstance(candidate, dict) and candidate.get(k, ""):
            return str(candidate[k])
    return f"glyph_candidate_{idx:05d}"


def get_nodes(candidate):
    for k in ["nodes", "primitive_nodes", "layout_nodes", "solved_nodes", "final_nodes", "optimized_nodes"]:
        if isinstance(candidate, dict) and isinstance(candidate.get(k), list):
            return candidate[k]
    return []


def get_edges(candidate):
    if not isinstance(candidate, dict):
        return []

    topo = candidate.get("topology", {})
    if isinstance(topo, dict):
        for k in ["positive_edges_undirected", "edges", "positive_edges_directed"]:
            if isinstance(topo.get(k), list):
                if k == "positive_edges_directed":
                    return [e for e in topo[k] if not isinstance(e, dict) or e.get("direction", "forward") == "forward"]
                return topo[k]

    for k in ["edges", "topology_edges", "relations"]:
        if isinstance(candidate.get(k), list):
            return candidate[k]

    return []


def node_id_of(node, fallback):
    if not isinstance(node, dict):
        return fallback
    for k in ["node_id", "source_node_id", "bezier_id", "stroke_id", "id"]:
        if k in node:
            return safe_int(node[k], fallback)
    return fallback


def normalize_edges(edges):
    clean = []
    for e in edges:
        if not isinstance(e, dict):
            continue

        ee = copy.deepcopy(e)

        if "u" not in ee:
            for k in ["src", "source", "a", "node_u", "stroke_a", "source_id", "bezier_id_a", "stroke_id_a"]:
                if k in ee:
                    ee["u"] = ee[k]
                    break

        if "v" not in ee:
            for k in ["dst", "target", "b", "node_v", "stroke_b", "target_id", "bezier_id_b", "stroke_id_b"]:
                if k in ee:
                    ee["v"] = ee[k]
                    break

        if isinstance(ee.get("u"), dict):
            ee["u"] = ee["u"].get("node_id", ee["u"].get("stroke_id", ee["u"].get("id", -1)))
        if isinstance(ee.get("v"), dict):
            ee["v"] = ee["v"].get("node_id", ee["v"].get("stroke_id", ee["v"].get("id", -1)))

        if "u" not in ee or "v" not in ee:
            continue

        ee["u"] = safe_int(ee["u"], -1)
        ee["v"] = safe_int(ee["v"], -1)
        if ee["u"] < 0 or ee["v"] < 0 or ee["u"] == ee["v"]:
            continue

        jt = None
        for k in ["j_type", "type", "event_type", "relation_type", "topology_type", "action"]:
            if k in ee:
                jt = parse_jtype(ee[k])
                break
        if jt == "NONE" or jt is None:
            if "j_type_idx" in ee:
                jt = parse_jtype(ee["j_type_idx"])
            else:
                jt = "E2E"

        ee["j_type"] = jt
        ee["t_u"] = clamp(safe_float(ee.get("t_u", ee.get("t_a", ee.get("source_t", 0.0))), 0.0), 0.0, 1.0)
        ee["t_v"] = clamp(safe_float(ee.get("t_v", ee.get("t_b", ee.get("target_t", 0.0))), 0.0), 0.0, 1.0)

        clean.append(ee)

    return clean


# =========================================================
# 4. Geometry utils: numpy
# =========================================================
def normalize_points(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    arr = arr[:, :2].copy()
    if not np.all(np.isfinite(arr)):
        return None
    if np.max(np.abs(arr)) > 2.0:
        arr = arr / float(CANVAS_SIZE)
    return np.clip(arr, -0.25, 1.25).astype(np.float32)


def denorm_points(arr):
    return (np.asarray(arr, dtype=np.float32) * float(CANVAS_SIZE)).astype(np.float32)


def read_layout_transform(node):
    lp = node.get("layout_prior", {})
    if not isinstance(lp, dict):
        lp = {}

    center = lp.get("center_norm", node.get("center_norm", [0.5, 0.5]))
    center = np.asarray(center, dtype=np.float32)[:2]
    if np.max(np.abs(center)) > 1.5:
        center = center / CANVAS_SIZE

    theta = safe_float(lp.get("rotation_rad", node.get("rotation_rad", 0.0)), 0.0)
    if "rotation_deg" in lp and "rotation_rad" not in lp:
        theta = math.radians(safe_float(lp["rotation_deg"], 0.0))

    length = safe_float(
        lp.get("length_norm", lp.get("scale_norm", node.get("length_norm", node.get("scale_norm", 0.25)))),
        0.25,
    )
    if abs(length) > 1.5:
        length = length / CANVAS_SIZE
    length = clamp(length, 0.02, 1.5)

    return {"center": center.astype(np.float32), "theta": float(theta), "scale": float(length)}


def normalize_local_polyline(polyline):
    arr = np.asarray(polyline, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) < 2:
        return np.asarray([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32)

    arr = arr[:, :2].copy()
    finite = np.all(np.isfinite(arr), axis=1)
    arr = arr[finite]
    if len(arr) < 2:
        return np.asarray([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32)

    xmin, xmax = float(arr[:, 0].min()), float(arr[:, 0].max())
    ymin, ymax = float(arr[:, 1].min()), float(arr[:, 1].max())
    xr = xmax - xmin

    if xr < 1e-6:
        return np.asarray([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32)

    arr[:, 0] = (arr[:, 0] - 0.5 * (xmin + xmax)) / xr
    arr[:, 1] = (arr[:, 1] - 0.5 * (ymin + ymax)) / xr

    arr[:, 0] = np.clip(arr[:, 0], -1.0, 1.0)
    arr[:, 1] = np.clip(arr[:, 1], -1.0, 1.0)
    return arr.astype(np.float32)


def apply_variant_np(polyline, variant_id):
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


def primitive_entry_to_polyline(node):
    ref = node.get("primitive_ref", {})
    if isinstance(ref, dict):
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
                arr = np.asarray(ref[k], dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) >= 2:
                    return normalize_local_polyline(arr[:, :2])

    return np.asarray([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32)


def extract_local_polyline(node):
    return apply_variant_np(primitive_entry_to_polyline(node), node.get("variant_id", 0))


def transform_polyline_np(local, center, theta, scale):
    center = np.asarray(center, dtype=np.float32)
    local = np.asarray(local, dtype=np.float32)
    c = math.cos(theta)
    s = math.sin(theta)
    R = np.asarray([[c, -s], [s, c]], dtype=np.float32)

    pts = local * (scale * CANVAS_SIZE)
    pts = pts @ R.T
    pts = pts + center[None, :] * CANVAS_SIZE
    return pts.astype(np.float32)


def bezier_from_polyline(polyline):
    pts = np.asarray(polyline, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 2:
        return np.asarray([[0.0, 0.0], [0.33, 0.0], [0.66, 0.0], [1.0, 0.0]], dtype=np.float32)

    pts = pts[:, :2]

    if len(pts) == 2:
        p0, p3 = pts[0], pts[-1]
        return np.stack([p0, p0 * (2 / 3) + p3 * (1 / 3), p0 * (1 / 3) + p3 * (2 / 3), p3], axis=0).astype(np.float32)

    idx1 = max(1, int(round((len(pts) - 1) / 3)))
    idx2 = min(len(pts) - 2, int(round((len(pts) - 1) * 2 / 3)))
    return np.stack([pts[0], pts[idx1], pts[idx2], pts[-1]], axis=0).astype(np.float32)


def read_prior_bezier_from_node(node):
    """
    返回 normalized cubic Bézier [4,2]。
    优先级：
        1. 已有 mother_bezier / bezier / control_points
        2. primitive_ref 里的 Bézier
        3. primitive polyline + layout_prior transform
    """
    for k in ["mother_bezier", "bezier", "control_points"]:
        if k in node:
            arr = normalize_points(node[k])
            if arr is not None and len(arr) >= 4:
                return arr[:4].astype(np.float32)

    ref = node.get("primitive_ref", {})
    if isinstance(ref, dict):
        for k in ["mother_bezier", "prototype_bezier_local_norm", "bezier_local_norm", "control_points_local_norm"]:
            if k in ref:
                arr = np.asarray(ref[k], dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) >= 4:
                    arr = arr[:4, :2]
                    # 如果看起来是 local [-0.5,0.5]，先按 layout 变换
                    if np.max(np.abs(arr)) <= 2.0 and (np.min(arr) < -0.05 or np.max(arr) <= 1.05):
                        tr = read_layout_transform(node)
                        c = math.cos(tr["theta"])
                        s = math.sin(tr["theta"])
                        R = np.asarray([[c, -s], [s, c]], dtype=np.float32)
                        pts = arr * (tr["scale"] * CANVAS_SIZE)
                        pts = pts @ R.T
                        pts = pts + tr["center"][None, :] * CANVAS_SIZE
                        return (pts / CANVAS_SIZE).astype(np.float32)
                    return normalize_points(arr)[:4].astype(np.float32)

    local = extract_local_polyline(node)
    tr = read_layout_transform(node)
    world_poly = transform_polyline_np(local, tr["center"], tr["theta"], tr["scale"])
    P_px = bezier_from_polyline(world_poly)
    return (P_px / CANVAS_SIZE).astype(np.float32)


def bezier_np(P, t):
    P = np.asarray(P, dtype=np.float32).reshape(4, 2)
    t = float(clamp(t, 0.0, 1.0))
    u = 1.0 - t
    return (
        (u ** 3) * P[0]
        + 3 * (u ** 2) * t * P[1]
        + 3 * u * (t ** 2) * P[2]
        + (t ** 3) * P[3]
    )


def bezier_deriv_np(P, t):
    P = np.asarray(P, dtype=np.float32).reshape(4, 2)
    t = float(clamp(t, 0.0, 1.0))
    u = 1.0 - t
    d = (
        3 * (u ** 2) * (P[1] - P[0])
        + 6 * u * t * (P[2] - P[1])
        + 3 * (t ** 2) * (P[3] - P[2])
    )
    n = np.linalg.norm(d)
    if n < 1e-8:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    return d / n


def sample_bezier_np(P, n=NUM_RENDER_SAMPLES):
    ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.stack([bezier_np(P, t) for t in ts], axis=0).astype(np.float32)


def closest_point_on_bezier_grid(P, q, n=CANONICALIZE_SAMPLE_N, t_min=0.0, t_max=1.0):
    ts = np.linspace(float(t_min), float(t_max), int(n), dtype=np.float32)
    pts = np.stack([bezier_np(P, t) for t in ts], axis=0)
    q = np.asarray(q, dtype=np.float32).reshape(1, 2)
    d = np.linalg.norm(pts - q, axis=1)
    k = int(np.argmin(d))
    return float(d[k]), float(ts[k])


def closest_curve_pair_grid(Pu, Pv, n=CANONICALIZE_SAMPLE_N, internal=False):
    if internal:
        lo, hi = CANONICALIZE_INTERNAL_EPS, 1.0 - CANONICALIZE_INTERNAL_EPS
    else:
        lo, hi = 0.0, 1.0

    ts = np.linspace(lo, hi, int(n), dtype=np.float32)
    us = np.stack([bezier_np(Pu, t) for t in ts], axis=0)
    vs = np.stack([bezier_np(Pv, t) for t in ts], axis=0)

    diff = us[:, None, :] - vs[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    flat = int(np.argmin(dist))
    i, j = np.unravel_index(flat, dist.shape)
    return float(dist[i, j]), float(ts[i]), float(ts[j])


def canonicalize_edge_t_from_prior(Pu, Pv, j_type):
    """
    推理阶段的 prior-based topology canonicalization。

    注意：
      这不是 constraint solver，也不是 iterative post-processing。
      它只是在模型 forward 前，把 raw candidate 的 edge/t
      规整成训练时同分布的 canonical topology representation。
    """
    jt = j_type if j_type in JTYPE_TO_ID and j_type != "NONE" else "E2E"

    # E2E：四个端点组合里选最近。
    if jt == "E2E":
        candidates = []
        for tu in [0.0, 1.0]:
            pu = bezier_np(Pu, tu)
            for tv in [0.0, 1.0]:
                pv = bezier_np(Pv, tv)
                d = float(np.linalg.norm(pu - pv))
                candidates.append((d, tu, tv))
        d, tu, tv = min(candidates, key=lambda x: x[0])
        return jt, float(tu), float(tv), {"mode": "E2E_endpoint_closest", "prior_dist_px": d * CANVAS_SIZE}

    # T：一条曲线的端点接另一条曲线中段。
    # 由于 generated topology 通常是无向的，这里同时尝试 u endpoint -> v curve 和 v endpoint -> u curve。
    if jt == "T":
        candidates = []
        for tu in [0.0, 1.0]:
            pu = bezier_np(Pu, tu)
            d, tv = closest_point_on_bezier_grid(Pv, pu, t_min=0.0, t_max=1.0)
            candidates.append((d, tu, tv, "u_endpoint_to_v_curve"))

        for tv in [0.0, 1.0]:
            pv = bezier_np(Pv, tv)
            d, tu = closest_point_on_bezier_grid(Pu, pv, t_min=0.0, t_max=1.0)
            candidates.append((d, tu, tv, "v_endpoint_to_u_curve"))

        d, tu, tv, mode = min(candidates, key=lambda x: x[0])
        return jt, float(tu), float(tv), {"mode": mode, "prior_dist_px": d * CANVAS_SIZE}

    # X：两条曲线内部采样点对的最近位置。
    if jt == "X":
        d, tu, tv = closest_curve_pair_grid(Pu, Pv, internal=True)
        return jt, float(tu), float(tv), {"mode": "X_internal_closest_pair", "prior_dist_px": d * CANVAS_SIZE}

    # fallback
    d, tu, tv = closest_curve_pair_grid(Pu, Pv, internal=False)
    return "E2E", float(tu), float(tv), {"mode": "fallback_curve_pair", "prior_dist_px": d * CANVAS_SIZE}


def curve_center_angle_length(P):
    P = np.asarray(P, dtype=np.float32).reshape(4, 2)
    p0 = P[0]
    p3 = P[3]
    center = P.mean(axis=0)
    d = p3 - p0
    length = float(np.linalg.norm(d))
    angle = math.atan2(float(d[1]), float(d[0])) if length > 1e-8 else 0.0
    return center.astype(np.float32), float(angle), float(length)


def primitive_straightness_from_bezier(P):
    samples = sample_bezier_np(P, n=12)
    chord = float(np.linalg.norm(samples[-1] - samples[0]))
    arc = float(np.sum(np.linalg.norm(samples[1:] - samples[:-1], axis=1)))
    if arc < 1e-8:
        return 1.0
    return float(np.clip(chord / arc, 0.0, 1.0))


def node_features_from_prior(P_prior, degree, counts):
    center, angle, length = curve_center_angle_length(P_prior)
    samples = sample_bezier_np(P_prior, n=12)
    bbox_min = samples.min(axis=0)
    bbox_max = samples.max(axis=0)
    bbox_wh = bbox_max - bbox_min
    straight = primitive_straightness_from_bezier(P_prior)

    return np.asarray([
        center[0],
        center[1],
        math.sin(angle),
        math.cos(angle),
        length,
        straight,
        bbox_wh[0],
        bbox_wh[1],
        degree / 8.0,
        counts["E2E"] / 8.0,
        counts["T"] / 8.0,
        counts["X"] / 8.0,
    ], dtype=np.float32)


# =========================================================
# 5. Torch Bézier + Model
# =========================================================
def bezier_torch(P, t):
    """
    P: [..., 4, 2]
    t: broadcastable tensor [...]
    return: [..., 2]
    """
    t = torch.clamp(t, 0.0, 1.0)
    while t.ndim < P.ndim - 1:
        t = t.unsqueeze(-1)
    u = 1.0 - t
    return (
        (u ** 3) * P[..., 0, :]
        + 3 * (u ** 2) * t * P[..., 1, :]
        + 3 * u * (t ** 2) * P[..., 2, :]
        + (t ** 3) * P[..., 3, :]
    )


class EdgeConditionedSelfAttention(nn.Module):
    def __init__(self, d_model, d_edge, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.d_edge = d_edge
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.edge_bias = nn.Sequential(
            nn.Linear(d_edge, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_heads),
        )
        self.edge_value = nn.Sequential(
            nn.Linear(d_edge, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_emb, node_mask):
        B, N, _ = x.shape

        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        bias = self.edge_bias(edge_emb).permute(0, 3, 1, 2)
        scores = scores + bias

        key_mask = node_mask[:, None, None, :]
        scores = scores.masked_fill(key_mask <= 0, -1e4)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        ev = self.edge_value(edge_emb).view(B, N, N, self.num_heads, self.d_head)
        ev = ev.permute(0, 3, 1, 2, 4)
        edge_context = (attn.unsqueeze(-1) * ev).sum(dim=3)

        out = out + edge_context
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.out(out)

        return out * node_mask.unsqueeze(-1)


class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model, d_edge, num_heads, dropout=0.1):
        super().__init__()
        self.attn = EdgeConditionedSelfAttention(d_model, d_edge, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, edge_emb, node_mask):
        x = self.norm1(x + self.attn(x, edge_emb, node_mask))
        x = x * node_mask.unsqueeze(-1)
        x = self.norm2(x + self.ffn(x))
        x = x * node_mask.unsqueeze(-1)
        return x


class DTGTransformer(nn.Module):
    """
    Junction-Anchor DTG-Transformer v4.
    """
    def __init__(self):
        super().__init__()
        self.shape_emb = nn.Embedding(MAX_SHAPE_CODE, 24)
        self.width_emb = nn.Embedding(MAX_WIDTH_TOKEN, 8)
        self.edge_type_emb = nn.Embedding(4, 16)

        self.node_in = nn.Sequential(
            nn.Linear(12 + 8 + 24 + 8, D_MODEL),
            nn.GELU(),
            nn.LayerNorm(D_MODEL),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, D_MODEL),
        )

        self.edge_in = nn.Sequential(
            nn.Linear(8 + 16, D_EDGE),
            nn.GELU(),
            nn.LayerNorm(D_EDGE),
            nn.Linear(D_EDGE, D_EDGE),
        )

        self.layers = nn.ModuleList([
            GraphTransformerLayer(D_MODEL, D_EDGE, NUM_HEADS, DROPOUT)
            for _ in range(NUM_LAYERS)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, 8),
        )

        self.anchor_head = nn.Sequential(
            nn.Linear(D_MODEL * 2 + D_EDGE + 2, D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.GELU(),
            nn.Linear(D_MODEL // 2, 2),
        )

    def _decode_raw_bezier(self, h, P_prior, node_mask):
        raw = self.decoder(h).view(P_prior.shape[0], P_prior.shape[1], 4, 2)
        if USE_RESIDUAL_DECODER:
            P_raw = P_prior + torch.tanh(raw) * MAX_CONTROL_DELTA
        else:
            P_raw = torch.sigmoid(raw)
        return P_raw * node_mask.view(node_mask.shape[0], node_mask.shape[1], 1, 1)

    def _bezier_weights(self, t):
        t = torch.clamp(t, 0.0, 1.0)
        u = 1.0 - t
        return torch.stack([
            u ** 3,
            3 * (u ** 2) * t,
            3 * u * (t ** 2),
            t ** 3,
        ], dim=-1)

    def _gather_edge_node_hidden(self, h, edge_index):
        B, E = edge_index.shape[:2]
        batch_idx = torch.arange(B, device=h.device).view(B, 1).expand(B, E)
        u = edge_index[..., 0].clamp(0, MAX_NODES - 1)
        v = edge_index[..., 1].clamp(0, MAX_NODES - 1)
        hu = h[batch_idx, u]
        hv = h[batch_idx, v]
        return hu, hv, u, v, batch_idx

    def _predict_edge_anchors(self, h, edge_emb, P_prior, edge_index, edge_t, edge_mask):
        hu, hv, u, v, batch_idx = self._gather_edge_node_hidden(h, edge_index)

        e_emb = edge_emb[batch_idx, u, v]

        Pi_prior = P_prior[batch_idx, u]
        Pj_prior = P_prior[batch_idx, v]
        tu = edge_t[..., 0]
        tv = edge_t[..., 1]

        ai_prior = bezier_torch(Pi_prior, tu)
        aj_prior = bezier_torch(Pj_prior, tv)
        a_prior = 0.5 * (ai_prior + aj_prior)

        inp = torch.cat([hu, hv, e_emb, a_prior], dim=-1)
        da = torch.tanh(self.anchor_head(inp)) * ANCHOR_MAX_DELTA
        anchor = a_prior * ANCHOR_PRIOR_BLEND + (a_prior + da) * (1.0 - ANCHOR_PRIOR_BLEND)
        anchor = torch.clamp(anchor, -0.15, 1.15)
        return anchor * edge_mask.unsqueeze(-1)

    def _anchor_project(self, P_raw, anchors, edge_index, edge_t, edge_mask):
        """
        单次 closed-form differentiable Bézier anchor projection。
        """
        B, N = P_raw.shape[:2]
        E = edge_index.shape[1]
        device = P_raw.device

        delta = torch.zeros_like(P_raw)
        weight_accum = torch.zeros((B, N, 1, 1), device=device, dtype=P_raw.dtype)

        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, E)
        u = edge_index[..., 0].clamp(0, MAX_NODES - 1)
        v = edge_index[..., 1].clamp(0, MAX_NODES - 1)
        tu = edge_t[..., 0]
        tv = edge_t[..., 1]

        for idx, t in [(u, tu), (v, tv)]:
            P_side = P_raw[batch_idx, idx]
            q = bezier_torch(P_side, t)
            d = (anchors - q) * edge_mask.unsqueeze(-1)

            w = self._bezier_weights(t)
            denom = (w ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-6)
            ctrl_delta = (w / denom).unsqueeze(-1) * d.unsqueeze(2)
            ctrl_delta = ctrl_delta * ANCHOR_PROJECT_STRENGTH

            flat_index = (batch_idx * N + idx).reshape(-1)
            delta_flat = delta.reshape(B * N, 4, 2)
            acc_flat = weight_accum.reshape(B * N, 1, 1)

            delta_flat.index_add_(0, flat_index, ctrl_delta.reshape(B * E, 4, 2))
            acc_flat.index_add_(0, flat_index, edge_mask.reshape(B * E, 1, 1))

        P_proj = P_raw + delta / weight_accum.clamp_min(1.0)
        return torch.clamp(P_proj, -0.10, 1.10)

    def forward(self, batch, return_aux=False):
        node_feat = batch["node_feat"]
        P_prior = batch["P_prior"]
        shape_ids = batch["shape_ids"]
        width_ids = batch["width_ids"]
        node_mask = batch["node_mask"]
        edge_type = batch["edge_type"]
        edge_feat = batch["edge_feat"]
        edge_index = batch["edge_index"]
        edge_t = batch["edge_t"]
        edge_mask = batch["edge_mask"]

        prior_flat = P_prior.reshape(P_prior.shape[0], P_prior.shape[1], 8)

        x = torch.cat([
            node_feat,
            prior_flat,
            self.shape_emb(shape_ids),
            self.width_emb(width_ids),
        ], dim=-1)

        h = self.node_in(x) * node_mask.unsqueeze(-1)

        et = self.edge_type_emb(edge_type)
        edge_emb = self.edge_in(torch.cat([edge_feat, et], dim=-1))

        for layer in self.layers:
            h = layer(h, edge_emb, node_mask)

        P_raw = self._decode_raw_bezier(h, P_prior, node_mask)
        anchors = self._predict_edge_anchors(h, edge_emb, P_prior, edge_index, edge_t, edge_mask)
        P_pred = self._anchor_project(P_raw, anchors, edge_index, edge_t, edge_mask)
        P_pred = P_pred * node_mask.view(node_mask.shape[0], node_mask.shape[1], 1, 1)

        if return_aux:
            return {
                "P_pred": P_pred,
                "P_raw": P_raw,
                "edge_anchors": anchors,
            }

        return P_pred


# =========================================================
# 6. Graph tensors
# =========================================================
def graph_to_tensors(candidate):
    nodes = get_nodes(candidate)
    raw_edges = normalize_edges(get_edges(candidate))

    N = len(nodes)
    if N <= 0:
        return None, "no_nodes"
    if N > MAX_NODES:
        return None, f"too_many_nodes:{N}>{MAX_NODES}"

    id_to_idx = {}
    for i, n in enumerate(nodes):
        id_to_idx[node_id_of(n, i)] = i
        id_to_idx[i] = i

    P_prior = np.zeros((MAX_NODES, 4, 2), dtype=np.float32)
    node_feat = np.zeros((MAX_NODES, 12), dtype=np.float32)
    shape_ids = np.zeros((MAX_NODES,), dtype=np.int64)
    width_ids = np.zeros((MAX_NODES,), dtype=np.int64)
    node_mask = np.zeros((MAX_NODES,), dtype=np.float32)

    prior_list = []
    for i, n in enumerate(nodes):
        P = read_prior_bezier_from_node(n)
        prior_list.append(P)
        P_prior[i] = P
        node_mask[i] = 1.0

        shape = safe_int(n.get("shape_code", n.get("shape_token", 0)), 0)
        shape_ids[i] = int(np.clip(shape, 0, MAX_SHAPE_CODE - 1))

        width = safe_int(n.get("width_token", 0), 0)
        width_ids[i] = int(np.clip(width, 0, MAX_WIDTH_TOKEN - 1))

    degree = Counter()
    type_counts = defaultdict(Counter)

    clean_edges = []
    for e in raw_edges:
        if e["u"] not in id_to_idx or e["v"] not in id_to_idx:
            continue
        u = id_to_idx[e["u"]]
        v = id_to_idx[e["v"]]
        if u == v or u < 0 or v < 0 or u >= N or v >= N:
            continue
        jt = e.get("j_type", "E2E")
        if jt not in JTYPE_TO_ID or jt == "NONE":
            jt = "E2E"

        ee = copy.deepcopy(e)
        ee["u_idx"] = u
        ee["v_idx"] = v
        ee["j_type"] = jt
        clean_edges.append(ee)

        degree[u] += 1
        degree[v] += 1
        type_counts[u][jt] += 1
        type_counts[v][jt] += 1

    for i in range(N):
        node_feat[i] = node_features_from_prior(
            P_prior[i],
            degree[i],
            type_counts[i],
        )

    edge_type = np.zeros((MAX_NODES, MAX_NODES), dtype=np.int64)
    edge_feat = np.zeros((MAX_NODES, MAX_NODES, 8), dtype=np.float32)

    edge_index = np.zeros((MAX_EDGES, 2), dtype=np.int64)
    edge_jtype = np.zeros((MAX_EDGES,), dtype=np.int64)
    edge_t = np.zeros((MAX_EDGES, 2), dtype=np.float32)
    edge_mask = np.zeros((MAX_EDGES,), dtype=np.float32)

    clean_edges = clean_edges[:MAX_EDGES]

    for k, e in enumerate(clean_edges):
        u = int(e["u_idx"])
        v = int(e["v_idx"])
        jt = e.get("j_type", "E2E")
        if jt not in JTYPE_TO_ID or jt == "NONE":
            jt = "E2E"

        raw_tu = clamp(safe_float(e.get("t_u", 0.0), 0.0), 0.0, 1.0)
        raw_tv = clamp(safe_float(e.get("t_v", 0.0), 0.0), 0.0, 1.0)

        if USE_INFERENCE_TOPO_CANONICALIZATION:
            jt, tu, tv, canon_info = canonicalize_edge_t_from_prior(P_prior[u], P_prior[v], jt)
            e["raw_t_u"] = float(raw_tu)
            e["raw_t_v"] = float(raw_tv)
            e["t_u"] = float(tu)
            e["t_v"] = float(tv)
            e["canonicalization"] = canon_info
        else:
            tu, tv = raw_tu, raw_tv

        jid = JTYPE_TO_ID.get(jt, 1)

        cu = P_prior[u].mean(axis=0)
        cv = P_prior[v].mean(axis=0)
        duv = cv - cu

        feat_uv = np.asarray([
            tu,
            tv,
            abs(tu - tv),
            tu * tv,
            duv[0],
            duv[1],
            1.0 if jt == "E2E" else 0.0,
            1.0 if jt in ["T", "X"] else 0.0,
        ], dtype=np.float32)

        feat_vu = np.asarray([
            tv,
            tu,
            abs(tu - tv),
            tu * tv,
            -duv[0],
            -duv[1],
            1.0 if jt == "E2E" else 0.0,
            1.0 if jt in ["T", "X"] else 0.0,
        ], dtype=np.float32)

        edge_type[u, v] = jid
        edge_type[v, u] = jid
        edge_feat[u, v] = feat_uv
        edge_feat[v, u] = feat_vu

        edge_index[k] = np.asarray([u, v], dtype=np.int64)
        edge_jtype[k] = jid
        edge_t[k] = np.asarray([tu, tv], dtype=np.float32)
        edge_mask[k] = 1.0

    graph = {
        "P_prior": P_prior,
        "node_feat": node_feat,
        "shape_ids": shape_ids,
        "width_ids": width_ids,
        "node_mask": node_mask,
        "edge_type": edge_type,
        "edge_feat": edge_feat,
        "edge_index": edge_index,
        "edge_jtype": edge_jtype,
        "edge_t": edge_t,
        "edge_mask": edge_mask,
        "nodes": nodes,
        "edges": clean_edges,
        "N": N,
        "E": len(clean_edges),
        "id_to_idx": id_to_idx,
    }

    return graph, "ok"


def batch_graphs(graphs, device):
    keys_float = ["P_prior", "node_feat", "node_mask", "edge_feat", "edge_t", "edge_mask"]
    keys_long = ["shape_ids", "width_ids", "edge_type", "edge_index", "edge_jtype"]

    batch = {}
    for k in keys_float:
        batch[k] = torch.tensor(np.stack([g[k] for g in graphs], axis=0), dtype=torch.float32, device=device)
    for k in keys_long:
        batch[k] = torch.tensor(np.stack([g[k] for g in graphs], axis=0), dtype=torch.long, device=device)

    return batch


# =========================================================
# 7. Metrics / output
# =========================================================
def evaluate_pred_geometry(P_pred, graph):
    edges = graph["edges"]
    if len(edges) == 0:
        return {
            "mean_junction_px": 0.0,
            "max_junction_px": 0.0,
            "min_tx_angle_deg": 180.0,
            "edge_debug": [],
        }

    dists = []
    tx_angles = []
    edge_debug = []

    for k, e in enumerate(edges):
        u = int(e["u_idx"])
        v = int(e["v_idx"])
        tu = clamp(safe_float(e.get("t_u", 0.0), 0.0), 0.0, 1.0)
        tv = clamp(safe_float(e.get("t_v", 0.0), 0.0), 0.0, 1.0)
        jt = e.get("j_type", "E2E")

        pi = bezier_np(P_pred[u], tu)
        pj = bezier_np(P_pred[v], tv)
        dist_px = float(np.linalg.norm(pi - pj) * CANVAS_SIZE)
        dists.append(dist_px)

        angle_deg = None
        if jt in ["T", "X"]:
            du = bezier_deriv_np(P_pred[u], tu)
            dv = bezier_deriv_np(P_pred[v], tv)
            c = float(np.clip(abs(np.dot(du, dv)), 0.0, 1.0))
            angle_deg = float(math.degrees(math.acos(c)))
            tx_angles.append(angle_deg)

        edge_debug.append({
            "edge_id": k,
            "u": int(u),
            "v": int(v),
            "j_type": jt,
            "t_u": float(tu),
            "t_v": float(tv),
            "junction_px": dist_px,
            "tx_angle_deg": angle_deg,
        })

    return {
        "mean_junction_px": float(np.mean(dists)) if dists else 0.0,
        "max_junction_px": float(np.max(dists)) if dists else 0.0,
        "min_tx_angle_deg": float(np.min(tx_angles)) if tx_angles else 180.0,
        "edge_debug": edge_debug,
    }


def classify_quality(max_j):
    if max_j <= GOOD_MAX_JUNCTION_PX:
        return "good"
    if max_j <= USABLE_MAX_JUNCTION_PX:
        return "usable_but_rough"
    return "bad"


def build_solved_candidate(candidate, graph, P_pred, aux_pred, idx=0):
    cid = get_candidate_id(candidate, idx)
    N = graph["N"]

    # prior/model 分别评估。
    # 之前日志出现 prior_maxJ=0 但 model 后 final_maxJ 很大，
    # 说明 generated candidates 的 prior topology 已经闭合，模型反而破坏了它。
    prior_eval_info = evaluate_pred_geometry(graph["P_prior"], graph)
    model_eval_info = evaluate_pred_geometry(P_pred, graph)

    prior_max_j = float(prior_eval_info["max_junction_px"])
    prior_mean_j = float(prior_eval_info["mean_junction_px"])
    model_max_j = float(model_eval_info["max_junction_px"])
    model_mean_j = float(model_eval_info["mean_junction_px"])

    selected_geometry = "model"
    selected_reason = "model_default"
    P_final = P_pred

    if USE_SAFE_GEOMETRY_SELECTION:
        if prior_max_j <= SAFE_PRIOR_ALREADY_GOOD_PX:
            # prior 已经满足 good 拓扑闭合时，直接保留 prior，避免神经网络把拓扑拉坏。
            selected_geometry = "prior"
            selected_reason = "prior_already_good"
            P_final = graph["P_prior"]
        elif model_max_j + SAFE_REQUIRE_MODEL_IMPROVE_BY_PX < prior_max_j:
            selected_geometry = "model"
            selected_reason = "model_improves_topology"
            P_final = P_pred
        else:
            selected_geometry = "prior"
            selected_reason = "model_not_better_than_prior"
            P_final = graph["P_prior"]

    eval_info = evaluate_pred_geometry(P_final, graph)
    max_j = float(eval_info["max_junction_px"])
    mean_j = float(eval_info["mean_junction_px"])
    status = classify_quality(max_j)

    solved_nodes = []
    for i, node in enumerate(graph["nodes"]):
        nnod = copy.deepcopy(node)

        P_norm = P_final[i].astype(np.float32)
        P_px = denorm_points(P_norm)
        poly_px = denorm_points(sample_bezier_np(P_norm, n=NUM_RENDER_SAMPLES))

        center, theta, length = curve_center_angle_length(P_norm)

        lp = nnod.setdefault("layout_prior", {})
        if not isinstance(lp, dict):
            lp = {}
            nnod["layout_prior"] = lp

        lp["center_norm"] = [float(center[0]), float(center[1])]
        lp["rotation_rad"] = float(theta)
        lp["rotation_deg"] = float(math.degrees(theta))
        lp["length_norm"] = float(length)
        lp["scale_norm"] = float(length)

        d = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float32) * float(length) * 0.5
        lp["p0_norm"] = [float((center - d)[0]), float((center - d)[1])]
        lp["p3_norm"] = [float((center + d)[0]), float((center + d)[1])]

        nnod["center_norm"] = lp["center_norm"]
        nnod["rotation_rad"] = lp["rotation_rad"]
        nnod["rotation_deg"] = lp["rotation_deg"]
        nnod["length_norm"] = lp["length_norm"]
        nnod["scale_norm"] = lp["scale_norm"]

        # 为兼容原 pipeline，这里用 px 版 mother_bezier / polyline
        nnod["mother_bezier"] = P_px.astype(float).tolist()
        nnod["solved_polyline_px"] = poly_px.astype(float).tolist()

        # 额外保留 norm 版，方便后续论文实验
        nnod["mother_bezier_norm"] = P_norm.astype(float).tolist()

        solved_nodes.append(nnod)

    quality_score = float(mean_j + 0.5 * max_j)

    out = copy.deepcopy(candidate)
    out["candidate_id"] = cid
    out["generated_glyph_id"] = out.get("generated_glyph_id", cid)
    out["glyph_candidate_id"] = out.get("glyph_candidate_id", cid)

    out["solved_nodes"] = solved_nodes
    out["quality_status"] = status
    out["quality_score"] = quality_score
    out["edge_debug"] = eval_info["edge_debug"]

    out["dtg_solver_debug"] = {
        "candidate_id": cid,
        "solver_type": "dtg_transformer_v5_robust_junction_anchor_safe",
        "num_nodes": int(N),
        "num_edges": int(graph["E"]),

        "selected_geometry": selected_geometry,
        "selected_reason": selected_reason,

        "prior_mean_junction_px": prior_mean_j,
        "prior_max_junction_px": prior_max_j,
        "model_mean_junction_px": model_mean_j,
        "model_max_junction_px": model_max_j,

        "mean_junction_px": float(mean_j),
        "max_junction_px": float(max_j),
        "min_tx_angle_deg": float(eval_info["min_tx_angle_deg"]),
        "no_iterative_postprocessing": True,
        "uses_constraint_solver": False,
        "uses_lbfgs": False,
        "uses_adam_per_candidate": False,
    }

    out["quality_report"] = {
        "quality_status": status,
        "quality_score": quality_score,

        "selected_geometry": selected_geometry,
        "selected_reason": selected_reason,

        "prior_mean_junction_px": prior_mean_j,
        "prior_max_junction_px": prior_max_j,
        "model_mean_junction_px": model_mean_j,
        "model_max_junction_px": model_max_j,

        "mean_junction_px": float(mean_j),
        "max_junction_px": float(max_j),
        "min_tx_angle_deg": float(eval_info["min_tx_angle_deg"]),
        "max_angle_diff_deg": 0.0,
        "solver_type": "dtg_transformer_v5_robust_junction_anchor_safe",
        "no_iterative_postprocessing": True,
    }

    return out, out["dtg_solver_debug"]


def make_invalid_candidate(candidate, reason, idx=0):
    cid = get_candidate_id(candidate, idx)
    out = copy.deepcopy(candidate)
    out["candidate_id"] = cid
    out["quality_status"] = "bad"
    out["quality_score"] = 1e9
    out["solved_nodes"] = get_nodes(candidate)
    out["quality_report"] = {
        "quality_status": "bad",
        "quality_score": 1e9,
        "mean_junction_px": 1e9,
        "max_junction_px": 1e9,
        "solver_type": "dtg_transformer_v5_robust_junction_anchor_safe",
        "failure_reason": reason,
    }
    out["dtg_solver_debug"] = {
        "candidate_id": cid,
        "solver_type": "dtg_transformer_v5_robust_junction_anchor_safe",
        "failure_reason": reason,
    }
    return out, out["dtg_solver_debug"]


def stats(vals):
    vals = np.asarray(vals, dtype=np.float32)
    if len(vals) == 0:
        return {}
    return {
        "mean": round(float(np.mean(vals)), 6),
        "p50": round(float(np.percentile(vals, 50)), 6),
        "p90": round(float(np.percentile(vals, 90)), 6),
        "max": round(float(np.max(vals)), 6),
    }


def summarize(results, debug_items, elapsed):
    status_hist = Counter()
    max_j = []
    mean_j = []
    prior_max_j = []
    prior_mean_j = []
    model_max_j = []
    model_mean_j = []
    selected_hist = Counter()
    node_hist = Counter()
    edge_hist = Counter()
    failures = Counter()

    for r, d in zip(results, debug_items):
        q = r.get("quality_report", {})
        st = q.get("quality_status", r.get("quality_status", "unknown"))
        status_hist[st] += 1

        if q.get("max_junction_px", 1e9) < 1e8:
            max_j.append(q.get("max_junction_px", 0.0))
            mean_j.append(q.get("mean_junction_px", 0.0))
            prior_max_j.append(q.get("prior_max_junction_px", 0.0))
            prior_mean_j.append(q.get("prior_mean_junction_px", 0.0))
            model_max_j.append(q.get("model_max_junction_px", q.get("max_junction_px", 0.0)))
            model_mean_j.append(q.get("model_mean_junction_px", q.get("mean_junction_px", 0.0)))
            selected_hist[q.get("selected_geometry", "unknown")] += 1

        node_hist[d.get("num_nodes", 0)] += 1
        edge_hist[d.get("num_edges", 0)] += 1
        if "failure_reason" in d:
            failures[d["failure_reason"]] += 1

    return {
        "count": len(results),
        "elapsed_sec": round(float(elapsed), 3),
        "cand_per_sec": round(float(len(results) / max(elapsed, 1e-6)), 3),
        "status_hist": dict(status_hist),
        "good_or_usable_count": status_hist.get("good", 0) + status_hist.get("usable_but_rough", 0),
        "selected_geometry_hist": dict(selected_hist),
        "prior_max_junction_px_stats": stats(prior_max_j),
        "prior_mean_junction_px_stats": stats(prior_mean_j),
        "model_max_junction_px_stats": stats(model_max_j),
        "model_mean_junction_px_stats": stats(model_mean_j),
        "max_junction_px_stats": stats(max_j),
        "mean_junction_px_stats": stats(mean_j),
        "node_hist": dict(node_hist),
        "edge_hist": dict(edge_hist),
        "failures": dict(failures),
    }


# =========================================================
# 8. Model loading
# =========================================================

def _apply_checkpoint_config(ckpt):
    """
    读取 checkpoint 里的 config，自动设置 inference 模型尺寸。
    兼容大模型 v4 和 lite-best v4。
    """
    global D_MODEL, D_EDGE, NUM_LAYERS, NUM_HEADS
    global MAX_NODES, MAX_EDGES, MAX_SHAPE_CODE, MAX_WIDTH_TOKEN
    global USE_RESIDUAL_DECODER, MAX_CONTROL_DELTA
    global ANCHOR_PROJECT_STRENGTH, ANCHOR_PRIOR_BLEND, ANCHOR_MAX_DELTA

    if not isinstance(ckpt, dict):
        return {}

    cfg = ckpt.get("config", {})
    if not isinstance(cfg, dict):
        return {}

    D_MODEL = int(cfg.get("D_MODEL", D_MODEL))
    D_EDGE = int(cfg.get("D_EDGE", D_EDGE))
    NUM_LAYERS = int(cfg.get("NUM_LAYERS", NUM_LAYERS))
    NUM_HEADS = int(cfg.get("NUM_HEADS", NUM_HEADS))

    MAX_NODES = int(cfg.get("MAX_NODES", MAX_NODES))
    MAX_EDGES = int(cfg.get("MAX_EDGES", MAX_EDGES))
    MAX_SHAPE_CODE = int(cfg.get("MAX_SHAPE_CODE", MAX_SHAPE_CODE))
    MAX_WIDTH_TOKEN = int(cfg.get("MAX_WIDTH_TOKEN", MAX_WIDTH_TOKEN))

    USE_RESIDUAL_DECODER = bool(cfg.get("USE_RESIDUAL_DECODER", USE_RESIDUAL_DECODER))
    MAX_CONTROL_DELTA = float(cfg.get("MAX_CONTROL_DELTA", MAX_CONTROL_DELTA))

    # v5_robust 训练脚本的旧 checkpoint 可能没有保存 ANCHOR_PRIOR_BLEND / ANCHOR_MAX_DELTA。
    # 但 forward 行为必须和训练时一致，否则模型会严重失真。
    is_robust = bool(cfg.get("ROBUST_CORRUPTION", False))
    ANCHOR_PROJECT_STRENGTH = float(cfg.get("ANCHOR_PROJECT_STRENGTH", 1.00 if is_robust else ANCHOR_PROJECT_STRENGTH))
    ANCHOR_PRIOR_BLEND = float(cfg.get("ANCHOR_PRIOR_BLEND", 0.05 if is_robust else ANCHOR_PRIOR_BLEND))
    ANCHOR_MAX_DELTA = float(cfg.get("ANCHOR_MAX_DELTA", 0.55 if is_robust else ANCHOR_MAX_DELTA))

    # 同理，v5 robust 使用更大的 residual 输出范围。
    MAX_CONTROL_DELTA = float(cfg.get("MAX_CONTROL_DELTA", 0.70 if is_robust else MAX_CONTROL_DELTA))

    return cfg


def load_model(device):
    """
    修复点：
      旧 inference 脚本写死 D_MODEL=192/D_EDGE=64/NUM_LAYERS=6/NUM_HEADS=8，
      但 lite-best checkpoint 是 D_MODEL=128/D_EDGE=48/NUM_LAYERS=4/NUM_HEADS=4。
      这里先加载 checkpoint，再用 ckpt['config'] 设置模型结构，然后实例化模型。
    """
    candidates = [
        V5_BEST_MODEL_FILE,
        V5_MODEL_FILE,
        V5_FINAL_MODEL_FILE,
        BEST_MODEL_FILE,
        MODEL_FILE,
        FINAL_MODEL_FILE,
    ]
    model_path = None
    for p in candidates:
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        raise FileNotFoundError("Cannot find model file. Expected one of: " + " | ".join(candidates))

    ckpt = torch.load(model_path, map_location=device)
    cfg = _apply_checkpoint_config(ckpt)

    model = DTGTransformer().to(device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        epoch = ckpt.get("epoch", None)
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        epoch = ckpt.get("epoch", None)
    else:
        state = ckpt
        epoch = None

    model.load_state_dict(state, strict=True)
    model.eval()

    return model, {
        "model_path": model_path,
        "epoch": epoch,
        "checkpoint_config": cfg,
        "inference_model_config": {
            "D_MODEL": D_MODEL,
            "D_EDGE": D_EDGE,
            "NUM_LAYERS": NUM_LAYERS,
            "NUM_HEADS": NUM_HEADS,
            "MAX_NODES": MAX_NODES,
            "MAX_EDGES": MAX_EDGES,
            "USE_RESIDUAL_DECODER": USE_RESIDUAL_DECODER,
            "MAX_CONTROL_DELTA": MAX_CONTROL_DELTA,
            "ANCHOR_PROJECT_STRENGTH": ANCHOR_PROJECT_STRENGTH,
            "ANCHOR_PRIOR_BLEND": ANCHOR_PRIOR_BLEND,
            "ANCHOR_MAX_DELTA": ANCHOR_MAX_DELTA,
        },
        "missing_keys": [],
        "unexpected_keys": [],
    }


# =========================================================
# 9. Main
# =========================================================
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    set_seed(RANDOM_SEED)
    device = choose_device()

    print("\n" + "=" * 80)
    print("DTG-Transformer v5 Robust Inference")
    print("Junction-Anchor Neural Geometry Solver")
    print("=" * 80)
    print(f"  input_file:   {INPUT_FILE}")
    print(f"  v5_best:      {V5_BEST_MODEL_FILE}")
    print(f"  v5_model:     {V5_MODEL_FILE}")
    print(f"  v5_final:     {V5_FINAL_MODEL_FILE}")
    print(f"  best_model:   {BEST_MODEL_FILE}")
    print(f"  model_file:   {MODEL_FILE}")
    print(f"  final_model:  {FINAL_MODEL_FILE}")
    print(f"  output_file:  {OUTPUT_FILE}")
    print(f"  report_file:  {OUTPUT_REPORT_FILE}")
    print(f"  device:       {device}")
    print("=" * 80)

    print("\n[Method]")
    print("  No constraint_solver.py is used.")
    print("  No L-BFGS / Adam per candidate is used.")
    print("  One Graph Transformer forward pass + closed-form Bézier anchor projection.")
    print(f"  MAX_NODES: {MAX_NODES}")
    print(f"  MAX_EDGES: {MAX_EDGES}")
    print(f"  BATCH_SIZE_INFER: {BATCH_SIZE_INFER}")
    print("  model config will be auto-detected from checkpoint if available.")
    print(f"  USE_INFERENCE_TOPO_CANONICALIZATION: {USE_INFERENCE_TOPO_CANONICALIZATION}")
    print(f"  CANONICALIZE_SAMPLE_N: {CANONICALIZE_SAMPLE_N}")
    print(f"  USE_SAFE_GEOMETRY_SELECTION: {USE_SAFE_GEOMETRY_SELECTION}")
    print(f"  SAFE_PRIOR_ALREADY_GOOD_PX: {SAFE_PRIOR_ALREADY_GOOD_PX}")
    print(f"  GOOD_MAX_JUNCTION_PX: {GOOD_MAX_JUNCTION_PX}")
    print(f"  USABLE_MAX_JUNCTION_PX: {USABLE_MAX_JUNCTION_PX}")

    data = load_json(INPUT_FILE, required=True)
    candidates = get_candidate_list(data)

    print("\n[Input]")
    print(f"  candidate_count: {len(candidates)}")
    if len(candidates) == 0:
        raise RuntimeError("No candidates found in glyph_candidates_with_primitives.json")

    model, model_info = load_model(device)
    print("\n[Model]")
    print(f"  loaded: {model_info['model_path']}")
    print(f"  epoch:  {model_info.get('epoch')}")
    print(f"  detected_config: {model_info.get('inference_model_config')}")
    if model_info.get("checkpoint_config", {}).get("ROBUST_CORRUPTION", False):
        print("  robust_forward_fix: enabled fallback ANCHOR_PRIOR_BLEND=0.05 / ANCHOR_MAX_DELTA=0.55 if missing in checkpoint")
    if model_info["missing_keys"]:
        print(f"  missing_keys: {len(model_info['missing_keys'])}")
    if model_info["unexpected_keys"]:
        print(f"  unexpected_keys: {len(model_info['unexpected_keys'])}")

    graphs = []
    graph_indices = []
    results = [None] * len(candidates)
    debug_items = [None] * len(candidates)

    skipped = Counter()

    for idx, cand in enumerate(candidates):
        graph, reason = graph_to_tensors(cand)
        if graph is None:
            solved, dbg = make_invalid_candidate(cand, reason, idx=idx)
            results[idx] = solved
            debug_items[idx] = dbg
            skipped[reason] += 1
        else:
            graphs.append(graph)
            graph_indices.append(idx)

    print("\n[Graph Build]")
    print(f"  valid_graphs: {len(graphs)}")
    print(f"  skipped: {dict(skipped)}")

    print("\n" + "=" * 80)
    print("Start DTG v4 Inference")
    print("=" * 80)

    t0 = time.time()

    with torch.no_grad():
        for start in range(0, len(graphs), BATCH_SIZE_INFER):
            end = min(start + BATCH_SIZE_INFER, len(graphs))
            batch_graph_list = graphs[start:end]
            batch_indices = graph_indices[start:end]

            batch = batch_graphs(batch_graph_list, device)
            aux = model(batch, return_aux=True)
            P_pred_batch = aux["P_pred"].detach().cpu().numpy()

            # 可选保存 anchors 统计，不写入每个点，避免 JSON 太大
            anchors_np = aux["edge_anchors"].detach().cpu().numpy()

            for bi, original_idx in enumerate(batch_indices):
                graph = batch_graph_list[bi]
                cand = candidates[original_idx]
                P_pred = P_pred_batch[bi]

                solved, dbg = build_solved_candidate(
                    cand,
                    graph,
                    P_pred,
                    {
                        "edge_anchors": anchors_np[bi],
                    },
                    idx=original_idx,
                )

                results[original_idx] = solved
                debug_items[original_idx] = dbg

            done = end
            if done % PRINT_EVERY == 0 or done == len(graphs):
                elapsed = time.time() - t0
                cps = done / max(elapsed, 1e-6)
                last_idx = graph_indices[done - 1] if done > 0 else -1
                last_dbg = debug_items[last_idx] if last_idx >= 0 else {}
                print(
                    f"  progress {done}/{len(graphs)} valid | "
                    f"{cps:.2f} cand/s | "
                    f"last={last_dbg.get('candidate_id', '')} "
                    f"N={last_dbg.get('num_nodes')} "
                    f"E={last_dbg.get('num_edges')} "
                    f"maxJ={last_dbg.get('max_junction_px', 0.0):.3f}px"
                )

    # Safety fill
    for idx in range(len(results)):
        if results[idx] is None:
            solved, dbg = make_invalid_candidate(candidates[idx], "unknown_inference_failure", idx=idx)
            results[idx] = solved
            debug_items[idx] = dbg

    elapsed = time.time() - t0
    summary = summarize(results, debug_items, elapsed)

    print("\n" + "=" * 80)
    print("DTG v4 Inference Summary")
    print("=" * 80)
    print(f"  count: {summary['count']}")
    print(f"  elapsed_sec: {summary['elapsed_sec']}")
    print(f"  cand_per_sec: {summary['cand_per_sec']}")
    print(f"  status_hist: {summary['status_hist']}")
    print(f"  good_or_usable_count: {summary['good_or_usable_count']}")
    print(f"  selected_geometry_hist: {summary['selected_geometry_hist']}")
    print(f"  prior_max_junction_px_stats: {summary['prior_max_junction_px_stats']}")
    print(f"  model_max_junction_px_stats: {summary['model_max_junction_px_stats']}")
    print(f"  final_max_junction_px_stats: {summary['max_junction_px_stats']}")
    print(f"  prior_mean_junction_px_stats:{summary['prior_mean_junction_px_stats']}")
    print(f"  model_mean_junction_px_stats:{summary['model_mean_junction_px_stats']}")
    print(f"  final_mean_junction_px_stats:{summary['mean_junction_px_stats']}")
    print(f"  failures: {summary['failures']}")

    output_obj = {
        "schema_version": "solved_glyph_candidates_dtg_transformer_v5_robust_safe_inference",
        "solver_type": "dtg_transformer_v5_robust_junction_anchor_safe",
        "input_file": INPUT_FILE,
        "model_info": model_info,
        "output_file": OUTPUT_FILE,
        "summary": summary,
        "config": {
            "CANVAS_SIZE": CANVAS_SIZE,
            "MAX_NODES": MAX_NODES,
            "MAX_EDGES": MAX_EDGES,
            "D_MODEL": D_MODEL,
            "D_EDGE": D_EDGE,
            "NUM_LAYERS": NUM_LAYERS,
            "NUM_HEADS": NUM_HEADS,
            "GOOD_MAX_JUNCTION_PX": GOOD_MAX_JUNCTION_PX,
            "USABLE_MAX_JUNCTION_PX": USABLE_MAX_JUNCTION_PX,
            "USE_INFERENCE_TOPO_CANONICALIZATION": USE_INFERENCE_TOPO_CANONICALIZATION,
            "CANONICALIZE_SAMPLE_N": CANONICALIZE_SAMPLE_N,
            "USE_SAFE_GEOMETRY_SELECTION": USE_SAFE_GEOMETRY_SELECTION,
            "SAFE_PRIOR_ALREADY_GOOD_PX": SAFE_PRIOR_ALREADY_GOOD_PX,
            "SAFE_REQUIRE_MODEL_IMPROVE_BY_PX": SAFE_REQUIRE_MODEL_IMPROVE_BY_PX,
            "no_constraint_solver": True,
            "no_iterative_postprocessing": True,
        },
        "solved_glyph_candidates": results,
    }

    report_obj = {
        "schema_version": "dtg_transformer_v5_robust_safe_inference_report",
        "solver_type": "dtg_transformer_v5_robust_junction_anchor_safe",
        "summary": summary,
        "model_info": model_info,
        "debug_items": debug_items,
    }

    save_json(output_obj, OUTPUT_FILE)
    save_json(report_obj, OUTPUT_REPORT_FILE)

    print("\n" + "=" * 80)
    print("Saved")
    print("=" * 80)
    print(f"  solved: {OUTPUT_FILE}")
    print(f"  report: {OUTPUT_REPORT_FILE}")
    print("\nNext:")
    print("  1. Run score_solved_glyphs.py")
    print("  2. Check gnn_scored_solved_previews/top_combined")
    print("  3. Compare status_hist and cand_per_sec against old constraint_solver.py")


if __name__ == "__main__":
    main()
