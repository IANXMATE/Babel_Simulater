# -*- coding: utf-8 -*-
"""
topostyle_retrieval_selector.py

Topology-first + Codebook + Retrieval style selector.

依赖同目录：
    train_topostyle_transformer.py
    topostyle_style_codebook.json
    topostyle_codebook_assignments.json

默认输入：
    优先 glyph_candidates_filtered_for_dtg.json
    否则 glyph_candidates_with_primitives.json

默认输出：
    topostyle_retrieval_solved_candidates.json
    topostyle_retrieval_selector_report.json
    solved_glyph_candidates.json  # 兼容旧 scorer

不使用 constraint_solver.py，不使用 L-BFGS / Adam per candidate。
"""

import os
import sys
import json
import math
import copy
import time
from collections import Counter

import numpy as np

try:
    import train_topostyle_transformer as ts
except Exception as e:
    raise RuntimeError(
        "请把 topostyle_retrieval_selector.py 放在 train_topostyle_transformer.py 同一个目录下运行。"
    ) from e


# =========================================================
# 0. Paths / Config
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CODEBOOK_FILE = os.path.join(SCRIPT_DIR, "topostyle_style_codebook.json")
ASSIGNMENTS_FILE = os.path.join(SCRIPT_DIR, "topostyle_codebook_assignments.json")

FILTERED_INPUT_FILE = os.path.join(SCRIPT_DIR, "glyph_candidates_filtered_for_dtg.json")
RAW_INPUT_FILE = os.path.join(SCRIPT_DIR, "glyph_candidates_with_primitives.json")

OUTPUT_FILE = os.path.join(SCRIPT_DIR, "topostyle_retrieval_solved_candidates.json")
REPORT_FILE = os.path.join(SCRIPT_DIR, "topostyle_retrieval_selector_report.json")
COMPAT_SOLVED_FILE = os.path.join(SCRIPT_DIR, "solved_glyph_candidates.json")

CANVAS_SIZE = 400.0

MAX_NODES = 8
MAX_EDGES = 32
MAX_INPUT_CANDIDATES = 300
MAX_OUTPUT_CANDIDATES = 240

# 候选图不像人工标注干净，允许更大的连接误差。
CANDIDATE_EDGE_MAXJ_PX = 45.0

TOP_K_TOKENS_PER_SEG = 6
BEAM_SIZE = 8
MEMORY_TOP_N = 50

# retrieval distance weights
W_STYLE = np.asarray([1.0, 1.4, 1.0, 1.4, 0.35], dtype=np.float32)
W_LENGTH = 0.35
W_STRAIGHT = 0.40
W_SHAPE_MISMATCH = 0.35
W_WIDTH_TOKEN_MISMATCH = 0.08
W_DEGREE = 0.06
W_PARENT_INDEX = 0.03

# token score weights
W_TOKEN_RETRIEVAL = 1.0
W_PRIOR_STYLE = 0.30
W_RARE_TOKEN = 0.04

SAVE_TOP_TOKEN_DEBUG = True
WRITE_COMPAT_SOLVED_FILE = True


# =========================================================
# 1. Basic utils
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


def denorm_points(arr):
    return np.asarray(arr, dtype=np.float32) * float(CANVAS_SIZE)


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


def sample_bezier_np(P, n=24):
    return ts.sample_bezier_np(P, n=n)


def style_to_bezier_np(A, B, style):
    return ts.style_to_bezier_np(A, B, style)


def curve_rmse_px(P1, P2, n=24):
    C1 = sample_bezier_np(P1, n=n)
    C2 = sample_bezier_np(P2, n=n)
    return float(np.sqrt(np.mean(np.sum((C1 - C2) ** 2, axis=-1))) * CANVAS_SIZE)


def get_candidate_id(candidate, idx=0):
    for k in ["generated_glyph_id", "glyph_candidate_id", "candidate_id", "sample_id", "grammar_sample_id"]:
        if isinstance(candidate, dict) and candidate.get(k, ""):
            return str(candidate[k])
    if isinstance(candidate, dict) and candidate.get("dtg_prefilter", {}).get("rank") is not None:
        return f"filtered_{candidate['dtg_prefilter']['rank']:05d}"
    return f"glyph_candidate_{idx:05d}"


# =========================================================
# 2. Codebook / memory bank
# =========================================================
def load_codebook_styles():
    obj = load_json(CODEBOOK_FILE)
    entries = obj.get("codebook", [])
    if not entries:
        raise RuntimeError("topostyle_style_codebook.json 没有 codebook 字段。")

    max_token = max(int(e["style_token"]) for e in entries)
    styles = np.zeros((max_token + 1, 5), dtype=np.float32)
    token_count = np.ones((max_token + 1,), dtype=np.float32)

    for e in entries:
        t = int(e["style_token"])
        if "style_vector" in e:
            styles[t] = np.asarray(e["style_vector"], dtype=np.float32)
        else:
            c = e["centroid_style"]
            styles[t] = np.asarray([c["alpha1"], c["beta1"], c["alpha2"], c["beta2"], c["width_norm"]], dtype=np.float32)
        token_count[t] = float(e.get("count", 1))

    return obj, styles, token_count


def make_assignment_map():
    obj = load_json(ASSIGNMENTS_FILE)
    rows = obj.get("assignments", [])
    if not rows:
        raise RuntimeError("topostyle_codebook_assignments.json 没有 assignments 字段。")
    mp = {}
    for r in rows:
        key = (str(r["source_file"]), str(r["hex_key"]), int(r["segment_index_in_glyph"]))
        mp[key] = int(r["style_token"])
    return mp, rows


def segment_length_straightness(seg):
    if "length" in seg:
        length = float(seg["length"])
    elif "length_norm" in seg:
        length = float(seg["length_norm"])
    else:
        A = np.asarray(seg["A"], dtype=np.float32)
        B = np.asarray(seg["B"], dtype=np.float32)
        length = float(np.linalg.norm(B - A))

    if "straight" in seg:
        straight = float(seg["straight"])
    elif "straightness" in seg:
        straight = float(seg["straightness"])
    else:
        P = np.asarray(seg["P_gt"], dtype=np.float32)
        C = sample_bezier_np(P, n=16)
        chord = np.linalg.norm(C[-1] - C[0])
        arc = np.sum(np.linalg.norm(C[1:] - C[:-1], axis=1))
        straight = float(chord / max(arc, 1e-8))
    return float(length), float(straight)


def build_memory_bank(codebook_styles):
    assignment_map, _ = make_assignment_map()
    glyphs = ts.load_annotation_glyphs()

    memory = []
    token_hist = Counter()
    missed = 0

    for g in glyphs:
        deg = Counter()
        for seg in g["segments"]:
            deg[int(seg["anchor_start"])] += 1
            deg[int(seg["anchor_end"])] += 1

        M = len(g["segments"])
        for si, seg in enumerate(g["segments"]):
            key = (str(g["source_file"]), str(g["hex_key"]), int(si))
            if key not in assignment_map:
                missed += 1
                continue
            token = int(assignment_map[key])
            length, straight = segment_length_straightness(seg)
            style = np.asarray(seg["style"], dtype=np.float32)
            a0 = int(seg["anchor_start"])
            a1 = int(seg["anchor_end"])
            row = {
                "memory_index": len(memory),
                "style_token": token,
                "style": style,
                "shape_code": int(seg.get("shape_code", 0)),
                "width_token": int(seg.get("width_token", 0)),
                "length": float(length),
                "straight": float(straight),
                "degree_start": int(deg[a0]),
                "degree_end": int(deg[a1]),
                "parent_index_norm": float(seg.get("parent_stroke", 0)) / max(1.0, ts.MAX_STROKES - 1.0),
                "segment_index_norm": float(si) / max(1.0, M - 1.0),
                "source_file": g["source_file"],
                "hex_key": g["hex_key"],
                "char": g.get("char", ""),
            }
            memory.append(row)
            token_hist[token] += 1

    if len(memory) == 0:
        raise RuntimeError("memory bank 为空。")

    return memory, {
        "memory_size": len(memory),
        "labeled_glyphs": len(glyphs),
        "missed_assignments": missed,
        "token_hist": dict(token_hist),
    }


# =========================================================
# 3. Candidate parsing
# =========================================================
def get_candidate_list(data):
    if isinstance(data, list):
        return data, None
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
                return topo[k]
    for k in ["edges", "topology_edges", "relations"]:
        if isinstance(candidate.get(k), list):
            return candidate[k]
    return []


def node_id_of(node, fallback):
    if not isinstance(node, dict):
        return fallback
    for k in ["node_id", "source_node_id", "bezier_id", "stroke_id", "id", "old_index"]:
        if k in node:
            return safe_int(node[k], fallback)
    return fallback


def parse_jtype(raw):
    if raw is None:
        return "E2E"
    if isinstance(raw, str):
        s = raw.upper()
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
    return "E2E"


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
        for k in ["j_type", "type", "event_type", "relation_type", "topology_type", "action", "j_type_idx"]:
            if k in ee:
                jt = parse_jtype(ee[k])
                break
        ee["j_type"] = jt or "E2E"
        ee["t_u"] = clamp(safe_float(ee.get("t_u", ee.get("t_a", ee.get("source_t", 0.0))), 0.0), 0.0, 1.0)
        ee["t_v"] = clamp(safe_float(ee.get("t_v", ee.get("t_b", ee.get("target_t", 0.0))), 0.0), 0.0, 1.0)
        clean.append(ee)
    return clean


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
    length = safe_float(lp.get("length_norm", lp.get("scale_norm", node.get("length_norm", node.get("scale_norm", 0.25)))), 0.25)
    if abs(length) > 1.5:
        length = length / CANVAS_SIZE
    return {"center": center.astype(np.float32), "theta": float(theta), "scale": float(clamp(length, 0.02, 1.5))}


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
    out = pts.copy()
    out[:, 1] = -pts[::-1, 1]
    return out


def primitive_entry_to_polyline(node):
    ref = node.get("primitive_ref", {})
    if isinstance(ref, dict):
        for k in [
            "primitive_polyline_local_norm", "prototype_polyline_local_norm", "prototype_polyline_norm",
            "polyline_local_norm", "polyline_norm", "polyline", "points",
        ]:
            if k in ref:
                arr = np.asarray(ref[k], dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) >= 2:
                    return normalize_local_polyline(arr[:, :2])
    return np.asarray([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32)


def transform_polyline_np(local, center, theta, scale):
    center = np.asarray(center, dtype=np.float32)
    local = np.asarray(local, dtype=np.float32)
    c, s = math.cos(theta), math.sin(theta)
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
                        c, s = math.cos(tr["theta"]), math.sin(tr["theta"])
                        R = np.asarray([[c, -s], [s, c]], dtype=np.float32)
                        pts = arr * (tr["scale"] * CANVAS_SIZE)
                        pts = pts @ R.T
                        pts = pts + tr["center"][None, :] * CANVAS_SIZE
                        return (pts / CANVAS_SIZE).astype(np.float32)
                    return normalize_points(arr)[:4].astype(np.float32)
    local = apply_variant_np(primitive_entry_to_polyline(node), node.get("variant_id", 0))
    tr = read_layout_transform(node)
    world_poly = transform_polyline_np(local, tr["center"], tr["theta"], tr["scale"])
    P_px = bezier_from_polyline(world_poly)
    return (P_px / CANVAS_SIZE).astype(np.float32)


def node_to_stroke(node, idx):
    P = read_prior_bezier_from_node(node)
    shape_code = safe_int(node.get("shape_code", node.get("shape_token", -1)), -1)
    if shape_code < 0 and isinstance(node.get("primitive_ref", {}), dict):
        shape_code = safe_int(node["primitive_ref"].get("shape_code", node["primitive_ref"].get("shape_token", -1)), -1)
    if shape_code < 0:
        C = sample_bezier_np(P, n=12)
        chord = np.linalg.norm(C[-1] - C[0])
        arc = np.sum(np.linalg.norm(C[1:] - C[:-1], axis=1))
        straight = chord / max(arc, 1e-8)
        shape_code = 20 if straight > 0.96 else (14 if straight > 0.85 else 16)
    width_token = safe_int(node.get("width_token", 0), 0)
    width_norm = safe_float(node.get("width_norm", node.get("width", 0.035)), 0.035)
    if width_norm > 1.0:
        width_norm = width_norm / CANVAS_SIZE
    return {
        "old_index": idx,
        "stroke_id": node_id_of(node, idx),
        "P": P.astype(np.float32),
        "shape_code": int(np.clip(shape_code, 0, ts.MAX_SHAPE_CODE - 1)),
        "width_token": int(np.clip(width_token, 0, ts.MAX_WIDTH_TOKEN - 1)),
        "width_norm": float(clamp(width_norm, 0.002, ts.WIDTH_MAX_NORM)),
        "stroke_type": node.get("stroke_type", "open"),
    }


def candidate_to_topostyle_glyph(candidate, idx):
    nodes = get_nodes(candidate)
    if not nodes:
        return None, "no_nodes"
    if len(nodes) > MAX_NODES:
        return None, f"too_many_nodes:{len(nodes)}"
    strokes = [node_to_stroke(n, i) for i, n in enumerate(nodes)]
    id_to_idx = {}
    for i, s in enumerate(strokes):
        id_to_idx[s["stroke_id"]] = i
        id_to_idx[s["old_index"]] = i
        id_to_idx[i] = i
    edges0 = []
    for e in normalize_edges(get_edges(candidate)):
        if e["u"] not in id_to_idx or e["v"] not in id_to_idx:
            continue
        u = id_to_idx[e["u"]]
        v = id_to_idx[e["v"]]
        if u == v:
            continue
        edges0.append({"u": int(u), "v": int(v), "j_type": e.get("j_type", "E2E"), "t_u": float(e.get("t_u", 0.0)), "t_v": float(e.get("t_v", 0.0)), "source": "candidate"})
    if not edges0:
        return None, "no_edges"
    edges = []
    dropped = []
    seen = set()
    for e in edges0:
        ce = ts.canonicalize_edge(e, strokes)
        if ce is None:
            dropped.append({"reason": "canonicalize_failed", "edge": e})
            continue
        err = safe_float(ce.get("oracle_junction_px", 0.0), 0.0)
        if err > CANDIDATE_EDGE_MAXJ_PX:
            dropped.append({"reason": "candidate_edge_too_large", "oracle_junction_px": err, "edge": ce})
            continue
        key = (ce["u"], ce["v"], ce["j_type"], round(ce["t_u"], 3), round(ce["t_v"], 3))
        if key in seen:
            continue
        seen.add(key)
        edges.append(ce)
    if not edges:
        return None, "no_edges_after_canonicalization"
    raw_glyph = {
        "source_file": "candidate_json",
        "hex_key": get_candidate_id(candidate, idx),
        "char": candidate.get("char", candidate.get("glyph_char", "")) if isinstance(candidate, dict) else "",
        "strokes": strokes,
        "edges": edges,
    }
    tg, reason = ts.build_topostyle_glyph(raw_glyph)
    if tg is None:
        return None, f"segmentize_{reason}"
    tg["source_candidate_index"] = idx
    tg["source_candidate_id"] = get_candidate_id(candidate, idx)
    tg["source_candidate"] = candidate
    tg["candidate_dropped_edges"] = dropped
    return tg, "ok"


# =========================================================
# 4. Retrieval / reconstruction
# =========================================================
def query_features(seg, glyph_segments):
    deg = Counter()
    for s in glyph_segments:
        deg[int(s["anchor_start"])] += 1
        deg[int(s["anchor_end"])] += 1
    M = len(glyph_segments)
    length, straight = segment_length_straightness(seg)
    return {
        "style": np.asarray(seg["style"], dtype=np.float32),
        "shape_code": int(seg.get("shape_code", 0)),
        "width_token": int(seg.get("width_token", 0)),
        "length": float(length),
        "straight": float(straight),
        "degree_start": int(deg[int(seg["anchor_start"])]),
        "degree_end": int(deg[int(seg["anchor_end"])]),
        "parent_index_norm": float(seg.get("parent_stroke", 0)) / max(1.0, ts.MAX_STROKES - 1.0),
        "segment_index_norm": float(seg.get("segment_id", 0)) / max(1.0, M - 1.0),
    }


def retrieval_distance(q, m):
    q_style = np.asarray(q["style"], dtype=np.float32)
    m_style = np.asarray(m["style"], dtype=np.float32)
    d_style = float(np.sqrt(np.sum(((q_style - m_style) * W_STYLE) ** 2)))
    q_len = max(1e-6, float(q["length"]))
    m_len = max(1e-6, float(m["length"]))
    d_len = abs(math.log(q_len / m_len))
    d_straight = abs(float(q["straight"]) - float(m["straight"]))
    d_shape = 0.0 if int(q["shape_code"]) == int(m["shape_code"]) else 1.0
    d_width = 0.0 if int(q["width_token"]) == int(m["width_token"]) else 1.0
    d_deg = abs(int(q["degree_start"]) - int(m["degree_start"])) + abs(int(q["degree_end"]) - int(m["degree_end"]))
    d_parent = abs(float(q["parent_index_norm"]) - float(m["parent_index_norm"]))
    return d_style + W_LENGTH * d_len + W_STRAIGHT * d_straight + W_SHAPE_MISMATCH * d_shape + W_WIDTH_TOKEN_MISMATCH * d_width + W_DEGREE * d_deg + W_PARENT_INDEX * d_parent


def direct_codebook_distance(q_style, token_style):
    return float(np.sqrt(np.sum(((np.asarray(q_style) - np.asarray(token_style)) * W_STYLE) ** 2)))


def retrieve_top_tokens_for_segment(seg, glyph_segments, memory, codebook_styles, token_count):
    q = query_features(seg, glyph_segments)
    rows = []
    for m in memory:
        rows.append((retrieval_distance(q, m), m))
    rows.sort(key=lambda x: x[0])
    rows = rows[:MEMORY_TOP_N]

    token_best = {}
    token_examples = {}
    for tok in range(codebook_styles.shape[0]):
        direct = direct_codebook_distance(q["style"], codebook_styles[tok])
        rarity = W_RARE_TOKEN / math.sqrt(max(1.0, float(token_count[tok])))
        token_best[tok] = W_PRIOR_STYLE * direct + rarity
        token_examples[tok] = {"source": "direct_codebook", "direct_style_dist": direct}

    for rank, (d, m) in enumerate(rows):
        tok = int(m["style_token"])
        direct = direct_codebook_distance(q["style"], codebook_styles[tok])
        rarity = W_RARE_TOKEN / math.sqrt(max(1.0, float(token_count[tok])))
        score = W_TOKEN_RETRIEVAL * d + W_PRIOR_STYLE * direct + rarity + 0.002 * rank
        if score < token_best.get(tok, 1e18):
            token_best[tok] = score
            token_examples[tok] = {
                "source": "memory",
                "memory_index": int(m["memory_index"]),
                "memory_source_file": m["source_file"],
                "memory_hex_key": m["hex_key"],
                "memory_char": m.get("char", ""),
                "retrieval_dist": float(d),
                "direct_style_dist": float(direct),
            }

    ranked = sorted(token_best.items(), key=lambda x: x[1])[:TOP_K_TOKENS_PER_SEG]
    out = []
    for rank, (tok, score) in enumerate(ranked):
        style = codebook_styles[int(tok)]
        P = style_to_bezier_np(seg["A"], seg["B"], style)
        prior_rmse = curve_rmse_px(P, seg["P_gt"])
        out.append({
            "rank": int(rank),
            "style_token": int(tok),
            "score": float(score),
            "prior_reconstruction_rmse_px": float(prior_rmse),
            "style_vector": style.astype(float).tolist(),
            "example": token_examples[int(tok)],
        })
    return out


def build_beams(segment_top_tokens):
    beams = [{"tokens": [], "score": 0.0, "token_details": []}]
    for toks in segment_top_tokens:
        new_beams = []
        for b in beams:
            for t in toks:
                new_beams.append({
                    "tokens": b["tokens"] + [int(t["style_token"])],
                    "score": float(b["score"] + t["score"]),
                    "token_details": b["token_details"] + [t],
                })
        new_beams.sort(key=lambda x: x["score"])
        beams = new_beams[:BEAM_SIZE]
    return beams


def reconstruct_beam_candidate(tg, beam, codebook_styles, output_rank, beam_rank):
    solved_nodes = []
    segment_edges = []
    max_prior_rmse = 0.0
    mean_prior_rmse_vals = []

    for si, seg in enumerate(tg["segments"]):
        tok = int(beam["tokens"][si])
        style = codebook_styles[tok]
        P = style_to_bezier_np(seg["A"], seg["B"], style)
        prior_rmse = curve_rmse_px(P, seg["P_gt"])
        max_prior_rmse = max(max_prior_rmse, prior_rmse)
        mean_prior_rmse_vals.append(prior_rmse)
        P_px = denorm_points(P)
        A_px = denorm_points(seg["A"])
        B_px = denorm_points(seg["B"])
        node = {
            "node_id": int(si),
            "bezier_id": int(si),
            "parent_stroke": int(seg.get("parent_stroke", -1)),
            "shape_code": int(seg.get("shape_code", 0)),
            "width_token": int(seg.get("width_token", 0)),
            "width_norm": float(style[4]),
            "width": float(style[4] * CANVAS_SIZE),
            "style_token": tok,
            "mother_bezier": P_px.astype(float).tolist(),
            "control_points": P_px.astype(float).tolist(),
            "topostyle": {
                "anchor_start": int(seg["anchor_start"]),
                "anchor_end": int(seg["anchor_end"]),
                "A_px": A_px.astype(float).tolist(),
                "B_px": B_px.astype(float).tolist(),
                "style_vector": style.astype(float).tolist(),
                "prior_reconstruction_rmse_px": float(prior_rmse),
                "t0": float(seg.get("t0", 0.0)),
                "t1": float(seg.get("t1", 1.0)),
            },
        }
        solved_nodes.append(node)
        segment_edges.append({"segment_id": int(si), "anchor_start": int(seg["anchor_start"]), "anchor_end": int(seg["anchor_end"]), "style_token": tok})

    anchors_px = []
    for a in tg["anchors"]:
        anchors_px.append({
            "anchor_id": int(a["anchor_id"]),
            "pos_px": denorm_points(a["pos"]).astype(float).tolist(),
            "keys": [[int(k[0]), float(k[1])] for k in a["keys"]],
        })

    source_candidate = tg.get("source_candidate", {})
    source_id = tg.get("source_candidate_id", "")
    out = {
        "generated_glyph_id": f"{source_id}_topostyle_beam_{beam_rank:02d}",
        "source_candidate_id": source_id,
        "source_candidate_index": int(tg.get("source_candidate_index", -1)),
        "topostyle_output_rank": int(output_rank),
        "topostyle_beam_rank": int(beam_rank),
        "char": tg.get("char", ""),
        "nodes": solved_nodes,
        "solved_nodes": solved_nodes,
        "solved_segments": solved_nodes,
        "topostyle_topology": {
            "anchors": anchors_px,
            "segment_edges": segment_edges,
            "topology_junction_px_by_construction": 0.0,
        },
        "quality_report": {
            "method": "topostyle_retrieval_selector",
            "beam_score": float(beam["score"]),
            "mean_prior_reconstruction_rmse_px": float(np.mean(mean_prior_rmse_vals)) if mean_prior_rmse_vals else 0.0,
            "max_prior_reconstruction_rmse_px": float(max_prior_rmse),
            "topology_junction_px_by_construction": 0.0,
            "num_segments": int(len(solved_nodes)),
            "num_anchors": int(len(tg["anchors"])),
        },
        "topostyle_retrieval": {
            "segment_token_details": beam["token_details"] if SAVE_TOP_TOKEN_DEBUG else [],
            "candidate_dropped_edges": tg.get("candidate_dropped_edges", []),
        },
        "source_candidate_meta": {
            "dtg_prefilter": source_candidate.get("dtg_prefilter", None) if isinstance(source_candidate, dict) else None,
            "original_id": source_id,
        },
    }
    return out


# =========================================================
# 5. Main
# =========================================================
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("\n" + "=" * 80)
    print("TopoStyle Retrieval Selector")
    print("=" * 80)
    print(f"  codebook:       {CODEBOOK_FILE}")
    print(f"  assignments:    {ASSIGNMENTS_FILE}")
    print(f"  filtered_input: {FILTERED_INPUT_FILE}")
    print(f"  raw_input:      {RAW_INPUT_FILE}")
    print(f"  output:         {OUTPUT_FILE}")
    print(f"  report:         {REPORT_FILE}")
    print("=" * 80)

    t0 = time.time()
    _, codebook_styles, token_count = load_codebook_styles()
    memory, memory_report = build_memory_bank(codebook_styles)

    input_file = FILTERED_INPUT_FILE if os.path.exists(FILTERED_INPUT_FILE) else RAW_INPUT_FILE
    data = load_json(input_file)
    candidates, candidate_key = get_candidate_list(data)
    if not candidates:
        raise RuntimeError("没有找到 candidates。")

    candidates = list(candidates)
    if isinstance(candidates[0], dict) and candidates[0].get("dtg_prefilter"):
        candidates.sort(key=lambda c: safe_float(c.get("dtg_prefilter", {}).get("score", 1e9), 1e9))
    candidates = candidates[:MAX_INPUT_CANDIDATES]

    print("\n[Loaded]")
    print(f"  input_file: {input_file}")
    print(f"  candidate_key: {candidate_key}")
    print(f"  input_candidates_used: {len(candidates)}")
    print(f"  codebook_size: {codebook_styles.shape[0]}")
    print(f"  memory_size: {memory_report['memory_size']}")
    print(f"  token_hist_top10: {Counter({int(k): int(v) for k, v in memory_report['token_hist'].items()}).most_common(10)}")

    solved = []
    skipped = Counter()
    graph_stats = []
    valid_count = 0

    for idx, cand in enumerate(candidates):
        tg, reason = candidate_to_topostyle_glyph(cand, idx)
        if tg is None:
            skipped[reason] += 1
            continue
        valid_count += 1

        segment_top_tokens = []
        for seg in tg["segments"]:
            segment_top_tokens.append(retrieve_top_tokens_for_segment(seg, tg["segments"], memory, codebook_styles, token_count))
        beams = build_beams(segment_top_tokens)

        for br, beam in enumerate(beams):
            out_rank = len(solved)
            solved.append(reconstruct_beam_candidate(tg, beam, codebook_styles, out_rank, br))
            if len(solved) >= MAX_OUTPUT_CANDIDATES:
                break

        graph_stats.append({
            "candidate_id": tg["source_candidate_id"],
            "num_segments": len(tg["segments"]),
            "num_anchors": len(tg["anchors"]),
            "num_beams": len(beams),
            "best_beam_score": float(beams[0]["score"]) if beams else None,
        })

        if (idx + 1) % 20 == 0 or idx == len(candidates) - 1:
            print(f"  progress {idx+1}/{len(candidates)} | valid={valid_count} | solved_outputs={len(solved)}")
        if len(solved) >= MAX_OUTPUT_CANDIDATES:
            break

    solved.sort(key=lambda x: (x["quality_report"]["max_prior_reconstruction_rmse_px"], x["quality_report"]["beam_score"]))
    for i, x in enumerate(solved):
        x["topostyle_output_rank"] = i

    max_rmses = [x["quality_report"]["max_prior_reconstruction_rmse_px"] for x in solved]
    mean_rmses = [x["quality_report"]["mean_prior_reconstruction_rmse_px"] for x in solved]
    beam_scores = [x["quality_report"]["beam_score"] for x in solved]
    elapsed = time.time() - t0

    output_obj = {
        "schema_version": "topostyle_retrieval_solved_candidates_v1",
        "method": "topology-first codebook retrieval selector",
        "source_input_file": input_file,
        "codebook_file": CODEBOOK_FILE,
        "assignments_file": ASSIGNMENTS_FILE,
        "summary": {
            "input_candidates_used": len(candidates),
            "valid_topostyle_graphs": valid_count,
            "solved_output_count": len(solved),
            "skipped": dict(skipped),
            "topology_junction_px_by_construction": 0.0,
            "elapsed_sec": round(float(elapsed), 3),
        },
        "solved_glyph_candidates": solved,
        "ranked_solved_glyph_candidates_by_combined": solved,
    }

    report_obj = {
        "schema_version": "topostyle_retrieval_selector_report_v1",
        "method": "topology-first codebook retrieval selector",
        "config": {
            "MAX_INPUT_CANDIDATES": MAX_INPUT_CANDIDATES,
            "MAX_OUTPUT_CANDIDATES": MAX_OUTPUT_CANDIDATES,
            "CANDIDATE_EDGE_MAXJ_PX": CANDIDATE_EDGE_MAXJ_PX,
            "TOP_K_TOKENS_PER_SEG": TOP_K_TOKENS_PER_SEG,
            "BEAM_SIZE": BEAM_SIZE,
            "MEMORY_TOP_N": MEMORY_TOP_N,
            "W_STYLE": W_STYLE.astype(float).tolist(),
            "W_LENGTH": W_LENGTH,
            "W_STRAIGHT": W_STRAIGHT,
            "W_SHAPE_MISMATCH": W_SHAPE_MISMATCH,
            "W_WIDTH_TOKEN_MISMATCH": W_WIDTH_TOKEN_MISMATCH,
        },
        "input": {
            "input_file": input_file,
            "candidate_key": candidate_key,
            "input_candidates_used": len(candidates),
        },
        "memory": memory_report,
        "summary": {
            "valid_topostyle_graphs": valid_count,
            "solved_output_count": len(solved),
            "skipped": dict(skipped),
            "elapsed_sec": round(float(elapsed), 3),
            "cand_per_sec": round(float(len(candidates) / max(elapsed, 1e-6)), 3),
            "output_per_sec": round(float(len(solved) / max(elapsed, 1e-6)), 3),
            "max_prior_reconstruction_rmse_px_stats": stats(max_rmses),
            "mean_prior_reconstruction_rmse_px_stats": stats(mean_rmses),
            "beam_score_stats": stats(beam_scores),
            "topology_junction_px_by_construction": 0.0,
        },
        "graph_stats": graph_stats,
        "top10_outputs": [
            {
                "rank": i,
                "generated_glyph_id": x["generated_glyph_id"],
                "source_candidate_id": x["source_candidate_id"],
                "beam_rank": x["topostyle_beam_rank"],
                "beam_score": x["quality_report"]["beam_score"],
                "max_prior_reconstruction_rmse_px": x["quality_report"]["max_prior_reconstruction_rmse_px"],
                "mean_prior_reconstruction_rmse_px": x["quality_report"]["mean_prior_reconstruction_rmse_px"],
                "num_segments": x["quality_report"]["num_segments"],
            }
            for i, x in enumerate(solved[:10])
        ],
    }

    save_json(output_obj, OUTPUT_FILE)
    save_json(report_obj, REPORT_FILE)
    if WRITE_COMPAT_SOLVED_FILE:
        save_json(output_obj, COMPAT_SOLVED_FILE)

    print("\n" + "=" * 80)
    print("Retrieval Summary")
    print("=" * 80)
    print(f"  input_candidates_used: {len(candidates)}")
    print(f"  valid_topostyle_graphs: {valid_count}")
    print(f"  solved_output_count: {len(solved)}")
    print(f"  skipped: {dict(skipped)}")
    print(f"  elapsed_sec: {elapsed:.3f}")
    print(f"  max_prior_reconstruction_rmse_px_stats: {report_obj['summary']['max_prior_reconstruction_rmse_px_stats']}")
    print(f"  mean_prior_reconstruction_rmse_px_stats: {report_obj['summary']['mean_prior_reconstruction_rmse_px_stats']}")
    print(f"  beam_score_stats: {report_obj['summary']['beam_score_stats']}")
    print("  topology_junction_px_by_construction: 0.0")

    print("\n[Top 10 outputs]")
    for row in report_obj["top10_outputs"]:
        print(
            f"  #{row['rank']:02d} {row['generated_glyph_id']} "
            f"beam={row['beam_rank']} maxRMSE={row['max_prior_reconstruction_rmse_px']:.3f}px "
            f"meanRMSE={row['mean_prior_reconstruction_rmse_px']:.3f}px "
            f"score={row['beam_score']:.3f} segs={row['num_segments']}"
        )

    print("\n" + "=" * 80)
    print("Saved")
    print("=" * 80)
    print(f"  solved: {OUTPUT_FILE}")
    print(f"  report: {REPORT_FILE}")
    if WRITE_COMPAT_SOLVED_FILE:
        print(f"  compat solved_glyph_candidates.json: {COMPAT_SOLVED_FILE}")

    print("\nNext:")
    print("  1. 直接运行 python score_solved_glyphs.py")
    print("  2. 查看 gnn_scored_solved_previews/top_combined")
    print("  3. 如果输出太多，可以降低 BEAM_SIZE 或 MAX_OUTPUT_CANDIDATES")


if __name__ == "__main__":
    main()
