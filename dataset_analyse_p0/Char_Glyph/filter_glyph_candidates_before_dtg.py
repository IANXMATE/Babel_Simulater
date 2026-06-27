# -*- coding: utf-8 -*-
"""
filter_glyph_candidates_before_dtg.py

在 DTG / Neural Geometry Solver 前，对 glyph candidates 做预筛选。

目的：
    glyph_candidates_with_primitives.json 里的大量 candidate
    prior topology error 太大，会把 DTG solver 淹没。
    这个脚本先计算每个 candidate 的 prior topology quality，
    保留更可能被 neural solver 修好的候选。

输入：
    glyph_candidates_with_primitives.json

输出：
    glyph_candidates_filtered_for_dtg.json
    filter_glyph_candidates_before_dtg_report.json

后续用法：
    方案 A：
        手动把 glyph_candidates_filtered_for_dtg.json 改名 / 复制成
        glyph_candidates_with_primitives.json
        然后运行 dtg_transformer_v5_robust_inference.py

    方案 B：
        修改 inference 脚本里的 INPUT_FILE，
        指向 glyph_candidates_filtered_for_dtg.json

注意：
    这不是 constraint_solver.py。
    不做 L-BFGS / Adam / iterative solve。
    只做 candidate quality prefilter。
"""

import os
import sys
import json
import math
import time
import copy
from collections import Counter, defaultdict

import numpy as np


# =========================================================
# 0. Paths
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(SCRIPT_DIR, "glyph_candidates_with_primitives.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "glyph_candidates_filtered_for_dtg.json")
OUTPUT_REPORT_FILE = os.path.join(SCRIPT_DIR, "filter_glyph_candidates_before_dtg_report.json")

# 是否额外写一份供 inference 直接使用的文件。
# 默认 False，避免覆盖原始 glyph_candidates_with_primitives.json。
WRITE_COMPAT_INPUT_FILE = False
COMPAT_INPUT_FILE = os.path.join(SCRIPT_DIR, "glyph_candidates_with_primitives.json")


# =========================================================
# 1. Filter config
# =========================================================
CANVAS_SIZE = 400.0

MAX_NODES = 8
MAX_EDGES = 24

KEEP_TOP_K = 50
MIN_KEEP = 20

HARD_MAX_JUNCTION_PX = 18.0
HARD_MEAN_JUNCTION_PX = 12.0

ALLOW_BACKFILL = True

# 太离谱的几何直接丢。
MAX_CANVAS_OUTSIDE_RATIO = 0.35
MAX_BBOX_DIAG_PX = 900.0
MIN_BBOX_DIAG_PX = 8.0

# topology 基本约束。
REQUIRE_EDGES = True
ALLOW_ZERO_EDGE_IF_NODES_LE_1 = False

# canonicalization
CANONICALIZE_SAMPLE_N = 64
CANONICALIZE_INTERNAL_EPS = 0.05

# scoring weights
W_MAX_JUNCTION = 1.00
W_MEAN_JUNCTION = 0.50
W_CANVAS_OUTSIDE = 80.0
W_BBOX_PENALTY = 0.15
W_EDGE_COUNT_PENALTY = 2.0
W_NODE_COUNT_PENALTY = 0.5


JTYPE_TO_ID = {
    "NONE": 0,
    "E2E": 1,
    "T": 2,
    "X": 3,
}


# =========================================================
# 2. Basic utils
# =========================================================
def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
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


def stats(vals):
    vals = np.asarray(vals, dtype=np.float32)
    if len(vals) == 0:
        return {}
    return {
        "mean": round(float(np.mean(vals)), 6),
        "p50": round(float(np.percentile(vals, 50)), 6),
        "p90": round(float(np.percentile(vals, 90)), 6),
        "p95": round(float(np.percentile(vals, 95)), 6),
        "max": round(float(np.max(vals)), 6),
    }


# =========================================================
# 3. Candidate parsing
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
                return data[k], k

        for k, v in data.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                print(f"[WARN] using unknown candidate key: {k}")
                return v, k

    return [], None


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

    idx = safe_int(raw, -1)
    if idx == 1:
        return "E2E"
    if idx == 2:
        return "T"
    if idx == 3:
        return "X"

    return "NONE"


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
# 4. Geometry
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
    return np.clip(arr, -0.5, 1.5).astype(np.float32)


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

    return {
        "center": center.astype(np.float32),
        "theta": float(theta),
        "scale": float(length),
    }


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


def sample_bezier_np(P, n=32):
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
    jt = j_type if j_type in JTYPE_TO_ID and j_type != "NONE" else "E2E"

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

    if jt == "X":
        d, tu, tv = closest_curve_pair_grid(Pu, Pv, internal=True)
        return jt, float(tu), float(tv), {"mode": "X_internal_closest_pair", "prior_dist_px": d * CANVAS_SIZE}

    d, tu, tv = closest_curve_pair_grid(Pu, Pv, internal=False)
    return "E2E", float(tu), float(tv), {"mode": "fallback_curve_pair", "prior_dist_px": d * CANVAS_SIZE}


def evaluate_graph_prior(Ps, edges):
    if len(edges) == 0:
        return {
            "mean_junction_px": 0.0,
            "max_junction_px": 0.0,
            "edge_debug": [],
        }

    dists = []
    edge_debug = []

    for k, e in enumerate(edges):
        u = int(e["u_idx"])
        v = int(e["v_idx"])
        jt = e.get("j_type", "E2E")
        jt, tu, tv, canon = canonicalize_edge_t_from_prior(Ps[u], Ps[v], jt)

        pi = bezier_np(Ps[u], tu)
        pj = bezier_np(Ps[v], tv)
        dist_px = float(np.linalg.norm(pi - pj) * CANVAS_SIZE)

        dists.append(dist_px)
        edge_debug.append({
            "edge_id": k,
            "u": u,
            "v": v,
            "j_type": jt,
            "t_u": tu,
            "t_v": tv,
            "junction_px": dist_px,
            "canonicalization": canon,
        })

    return {
        "mean_junction_px": float(np.mean(dists)) if dists else 0.0,
        "max_junction_px": float(np.max(dists)) if dists else 0.0,
        "edge_debug": edge_debug,
    }


def geometry_basic_stats(Ps):
    if len(Ps) == 0:
        return {
            "bbox_diag_px": 0.0,
            "canvas_outside_ratio": 1.0,
            "bbox": [0, 0, 0, 0],
        }

    pts = []
    for P in Ps:
        pts.append(sample_bezier_np(P, n=16))
    pts = np.concatenate(pts, axis=0)

    px = pts * CANVAS_SIZE
    xmin, ymin = px.min(axis=0)
    xmax, ymax = px.max(axis=0)

    diag = float(np.linalg.norm([xmax - xmin, ymax - ymin]))
    outside = np.logical_or.reduce([
        pts[:, 0] < 0.0,
        pts[:, 0] > 1.0,
        pts[:, 1] < 0.0,
        pts[:, 1] > 1.0,
    ])
    outside_ratio = float(np.mean(outside))

    return {
        "bbox_diag_px": diag,
        "canvas_outside_ratio": outside_ratio,
        "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
    }


# =========================================================
# 5. Candidate evaluation
# =========================================================
def build_candidate_graph(candidate, idx):
    nodes = get_nodes(candidate)
    raw_edges = normalize_edges(get_edges(candidate))

    cid = get_candidate_id(candidate, idx)
    N = len(nodes)

    if N <= 0:
        return None, "no_nodes"
    if N > MAX_NODES:
        return None, f"too_many_nodes:{N}>{MAX_NODES}"

    id_to_idx = {}
    for i, n in enumerate(nodes):
        id_to_idx[node_id_of(n, i)] = i
        id_to_idx[i] = i

    Ps = []
    for n in nodes:
        Ps.append(read_prior_bezier_from_node(n))
    Ps = np.stack(Ps, axis=0).astype(np.float32)

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

    if REQUIRE_EDGES and len(clean_edges) <= 0:
        if not (ALLOW_ZERO_EDGE_IF_NODES_LE_1 and N <= 1):
            return None, "no_edges"

    if len(clean_edges) > MAX_EDGES:
        return None, f"too_many_edges:{len(clean_edges)}>{MAX_EDGES}"

    return {
        "candidate_id": cid,
        "nodes": nodes,
        "edges": clean_edges,
        "P_prior": Ps,
        "N": N,
        "E": len(clean_edges),
    }, "ok"


def score_candidate(candidate, idx):
    graph, reason = build_candidate_graph(candidate, idx)
    if graph is None:
        return {
            "idx": idx,
            "candidate_id": get_candidate_id(candidate, idx),
            "ok": False,
            "reason": reason,
            "score": 1e18,
            "hard_pass": False,
        }

    prior_eval = evaluate_graph_prior(graph["P_prior"], graph["edges"])
    geom = geometry_basic_stats(graph["P_prior"])

    max_j = prior_eval["max_junction_px"]
    mean_j = prior_eval["mean_junction_px"]

    N = graph["N"]
    E = graph["E"]

    bbox_diag = geom["bbox_diag_px"]
    outside = geom["canvas_outside_ratio"]

    hard_reasons = []

    if max_j > HARD_MAX_JUNCTION_PX:
        hard_reasons.append(f"maxJ>{HARD_MAX_JUNCTION_PX}")
    if mean_j > HARD_MEAN_JUNCTION_PX:
        hard_reasons.append(f"meanJ>{HARD_MEAN_JUNCTION_PX}")
    if outside > MAX_CANVAS_OUTSIDE_RATIO:
        hard_reasons.append(f"outside>{MAX_CANVAS_OUTSIDE_RATIO}")
    if bbox_diag > MAX_BBOX_DIAG_PX:
        hard_reasons.append(f"bbox_diag>{MAX_BBOX_DIAG_PX}")
    if bbox_diag < MIN_BBOX_DIAG_PX:
        hard_reasons.append(f"bbox_diag<{MIN_BBOX_DIAG_PX}")

    # edge count 的软惩罚：太少或太多都不理想，但不硬杀。
    expected_edges = max(1, N - 1)
    edge_penalty = abs(E - expected_edges)

    bbox_penalty = 0.0
    if bbox_diag > 0:
        if bbox_diag < 40:
            bbox_penalty += 40 - bbox_diag
        elif bbox_diag > 650:
            bbox_penalty += bbox_diag - 650

    score = (
        W_MAX_JUNCTION * max_j
        + W_MEAN_JUNCTION * mean_j
        + W_CANVAS_OUTSIDE * outside
        + W_BBOX_PENALTY * bbox_penalty
        + W_EDGE_COUNT_PENALTY * edge_penalty
        + W_NODE_COUNT_PENALTY * abs(N - 3)
    )

    return {
        "idx": idx,
        "candidate_id": graph["candidate_id"],
        "ok": True,
        "reason": "ok",
        "score": float(score),
        "hard_pass": len(hard_reasons) == 0,
        "hard_reasons": hard_reasons,
        "num_nodes": int(N),
        "num_edges": int(E),
        "prior_mean_junction_px": float(mean_j),
        "prior_max_junction_px": float(max_j),
        "bbox_diag_px": float(bbox_diag),
        "canvas_outside_ratio": float(outside),
        "bbox": geom["bbox"],
        "edge_debug": prior_eval["edge_debug"],
    }


def attach_filter_info(candidate, item, rank, selected_by):
    out = copy.deepcopy(candidate)
    out["dtg_prefilter"] = {
        "rank": int(rank),
        "selected_by": selected_by,
        "score": float(item["score"]),
        "hard_pass": bool(item.get("hard_pass", False)),
        "hard_reasons": item.get("hard_reasons", []),
        "prior_mean_junction_px": float(item.get("prior_mean_junction_px", 1e9)),
        "prior_max_junction_px": float(item.get("prior_max_junction_px", 1e9)),
        "bbox_diag_px": float(item.get("bbox_diag_px", 0.0)),
        "canvas_outside_ratio": float(item.get("canvas_outside_ratio", 1.0)),
        "num_nodes": int(item.get("num_nodes", 0)),
        "num_edges": int(item.get("num_edges", 0)),
    }
    return out


# =========================================================
# 6. Main
# =========================================================
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("\n" + "=" * 80)
    print("Filter Glyph Candidates Before DTG")
    print("=" * 80)
    print(f"  input:   {INPUT_FILE}")
    print(f"  output:  {OUTPUT_FILE}")
    print(f"  report:  {OUTPUT_REPORT_FILE}")
    print("=" * 80)

    print("\n[Filter Config]")
    print(f"  KEEP_TOP_K: {KEEP_TOP_K}")
    print(f"  MIN_KEEP: {MIN_KEEP}")
    print(f"  HARD_MAX_JUNCTION_PX: {HARD_MAX_JUNCTION_PX}")
    print(f"  HARD_MEAN_JUNCTION_PX: {HARD_MEAN_JUNCTION_PX}")
    print(f"  MAX_CANVAS_OUTSIDE_RATIO: {MAX_CANVAS_OUTSIDE_RATIO}")
    print(f"  MAX_NODES: {MAX_NODES}")
    print(f"  MAX_EDGES: {MAX_EDGES}")

    data = load_json(INPUT_FILE)
    candidates, key = get_candidate_list(data)

    if not candidates:
        raise RuntimeError("No candidates found.")

    print("\n[Input]")
    print(f"  candidate_key: {key}")
    print(f"  candidate_count: {len(candidates)}")

    t0 = time.time()

    items = []
    fail_counter = Counter()

    for idx, cand in enumerate(candidates):
        item = score_candidate(cand, idx)
        items.append(item)
        if not item.get("ok", False):
            fail_counter[item.get("reason", "unknown")] += 1

    ok_items = [x for x in items if x.get("ok", False)]
    hard_pass = [x for x in ok_items if x.get("hard_pass", False)]

    ok_sorted = sorted(ok_items, key=lambda x: x["score"])
    hard_sorted = sorted(hard_pass, key=lambda x: x["score"])

    selected = []
    selected_by = {}

    for x in hard_sorted[:KEEP_TOP_K]:
        selected.append(x)
        selected_by[x["idx"]] = "hard_pass_top_score"

    if ALLOW_BACKFILL and len(selected) < MIN_KEEP:
        already = set(x["idx"] for x in selected)
        for x in ok_sorted:
            if x["idx"] in already:
                continue
            selected.append(x)
            selected_by[x["idx"]] = "backfill_top_score"
            already.add(x["idx"])
            if len(selected) >= MIN_KEEP:
                break

    # 如果 hard pass 很多，但超过 KEEP_TOP_K，就只保留 top K。
    if len(selected) > KEEP_TOP_K:
        selected = selected[:KEEP_TOP_K]

    selected = sorted(selected, key=lambda x: x["score"])

    filtered_candidates = []
    for rank, item in enumerate(selected):
        filtered_candidates.append(
            attach_filter_info(candidates[item["idx"]], item, rank, selected_by.get(item["idx"], "selected"))
        )

    elapsed = time.time() - t0

    prior_max_all = [x.get("prior_max_junction_px", np.nan) for x in ok_items]
    prior_mean_all = [x.get("prior_mean_junction_px", np.nan) for x in ok_items]
    prior_max_selected = [x.get("prior_max_junction_px", np.nan) for x in selected]
    prior_mean_selected = [x.get("prior_mean_junction_px", np.nan) for x in selected]

    hard_reason_counter = Counter()
    for x in ok_items:
        for r in x.get("hard_reasons", []):
            hard_reason_counter[r] += 1

    selected_by_counter = Counter(selected_by.get(x["idx"], "selected") for x in selected)

    summary = {
        "input_count": len(candidates),
        "ok_count": len(ok_items),
        "failed_count": len(candidates) - len(ok_items),
        "hard_pass_count": len(hard_pass),
        "selected_count": len(filtered_candidates),
        "selected_by": dict(selected_by_counter),
        "elapsed_sec": round(float(elapsed), 3),
        "cand_per_sec": round(float(len(candidates) / max(elapsed, 1e-6)), 3),
        "failures": dict(fail_counter),
        "hard_reason_hist": dict(hard_reason_counter),
        "all_prior_max_junction_px_stats": stats(prior_max_all),
        "all_prior_mean_junction_px_stats": stats(prior_mean_all),
        "selected_prior_max_junction_px_stats": stats(prior_max_selected),
        "selected_prior_mean_junction_px_stats": stats(prior_mean_selected),
        "top10": [
            {
                "rank": i,
                "idx": int(x["idx"]),
                "candidate_id": x["candidate_id"],
                "score": round(float(x["score"]), 4),
                "prior_max_junction_px": round(float(x.get("prior_max_junction_px", 0)), 4),
                "prior_mean_junction_px": round(float(x.get("prior_mean_junction_px", 0)), 4),
                "num_nodes": int(x.get("num_nodes", 0)),
                "num_edges": int(x.get("num_edges", 0)),
                "hard_pass": bool(x.get("hard_pass", False)),
                "hard_reasons": x.get("hard_reasons", []),
            }
            for i, x in enumerate(selected[:10])
        ],
    }

    output_obj = {
        "schema_version": "glyph_candidates_filtered_for_dtg",
        "source_file": INPUT_FILE,
        "filter_config": {
            "KEEP_TOP_K": KEEP_TOP_K,
            "MIN_KEEP": MIN_KEEP,
            "HARD_MAX_JUNCTION_PX": HARD_MAX_JUNCTION_PX,
            "HARD_MEAN_JUNCTION_PX": HARD_MEAN_JUNCTION_PX,
            "MAX_CANVAS_OUTSIDE_RATIO": MAX_CANVAS_OUTSIDE_RATIO,
            "MAX_NODES": MAX_NODES,
            "MAX_EDGES": MAX_EDGES,
        },
        "summary": summary,
        "glyph_candidates_with_primitives": filtered_candidates,
    }

    report_obj = {
        "schema_version": "filter_glyph_candidates_before_dtg_report",
        "source_file": INPUT_FILE,
        "output_file": OUTPUT_FILE,
        "summary": summary,
        "all_items": [
            {
                k: v
                for k, v in x.items()
                if k not in ["edge_debug"]
            }
            for x in items
        ],
    }

    save_json(output_obj, OUTPUT_FILE)
    save_json(report_obj, OUTPUT_REPORT_FILE)

    if WRITE_COMPAT_INPUT_FILE:
        # 危险选项：会覆盖 inference 默认输入。
        # 默认关闭。
        save_json(output_obj, COMPAT_INPUT_FILE)

    print("\n" + "=" * 80)
    print("Filter Summary")
    print("=" * 80)
    print(f"  input_count: {summary['input_count']}")
    print(f"  ok_count: {summary['ok_count']}")
    print(f"  hard_pass_count: {summary['hard_pass_count']}")
    print(f"  selected_count: {summary['selected_count']}")
    print(f"  selected_by: {summary['selected_by']}")
    print(f"  failures: {summary['failures']}")
    print(f"  hard_reason_hist: {summary['hard_reason_hist']}")
    print(f"  all_prior_max_junction_px_stats: {summary['all_prior_max_junction_px_stats']}")
    print(f"  selected_prior_max_junction_px_stats: {summary['selected_prior_max_junction_px_stats']}")
    print(f"  selected_prior_mean_junction_px_stats:{summary['selected_prior_mean_junction_px_stats']}")

    print("\n[Top 10]")
    for x in summary["top10"]:
        print(
            f"  #{x['rank']:02d} idx={x['idx']} id={x['candidate_id']} "
            f"score={x['score']:.3f} maxJ={x['prior_max_junction_px']:.3f}px "
            f"meanJ={x['prior_mean_junction_px']:.3f}px "
            f"N={x['num_nodes']} E={x['num_edges']} hard={x['hard_pass']}"
        )

    print("\n" + "=" * 80)
    print("Saved")
    print("=" * 80)
    print(f"  filtered: {OUTPUT_FILE}")
    print(f"  report:   {OUTPUT_REPORT_FILE}")

    print("\nNext:")
    print("  1. 修改 dtg_transformer_v5_robust_inference.py 的 INPUT_FILE 指向 glyph_candidates_filtered_for_dtg.json")
    print("  2. 或者手动复制 glyph_candidates_filtered_for_dtg.json 为 glyph_candidates_with_primitives.json 后再跑 inference")
    print("  3. 然后运行 python dtg_transformer_v5_robust_inference.py")


if __name__ == "__main__":
    main()
