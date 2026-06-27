# -*- coding: utf-8 -*-
"""
neural_constraint_solver.py

Neural warm-start + fast differentiable projection solver.

Input:
    glyph_candidates_with_primitives.json
Optional teacher:
    solved_glyph_candidates_teacher.json or existing solved_glyph_candidates.json
Output:
    solved_glyph_candidates.json

This is designed as a faster replacement candidate for constraint_solver.py.
It prints key parameters and before/warm/final junction metrics so you can judge feasibility.
"""

import os
import sys
import json
import math
import time
import random
import copy
from collections import Counter, defaultdict

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    raise RuntimeError("neural_constraint_solver.py requires PyTorch.") from e


# =========================================================
# Paths
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(SCRIPT_DIR, "glyph_candidates_with_primitives.json")
TEACHER_SOLVED_FILE = os.path.join(SCRIPT_DIR, "solved_glyph_candidates_teacher.json")
OLD_SOLVED_FILE = os.path.join(SCRIPT_DIR, "solved_glyph_candidates.json")

OUTPUT_FILE = os.path.join(SCRIPT_DIR, "solved_glyph_candidates.json")
OUTPUT_REPORT_FILE = os.path.join(SCRIPT_DIR, "neural_constraint_solver_report.json")
OUTPUT_MODEL_FILE = os.path.join(SCRIPT_DIR, "solver_warmstart_gnn.pt")

RANDOM_SEED = 42
DEVICE_MODE = "auto"  # auto / cuda / mps / cpu
CANVAS_SIZE = 400.0

# =========================================================
# Config
# =========================================================
MAX_NODES = 8
MAX_SHAPE_CODE = 128
MAX_WIDTH_TOKEN = 8
JTYPE_TO_ID = {"NONE": 0, "E2E": 1, "X": 2, "T": 3, "UNKNOWN": 0}

AUTO_TRAIN_IF_TEACHER_EXISTS = False
FORCE_RETRAIN = False

# v2: 默认关闭随机 GNN warm-start。只有确认 teacher fingerprint 匹配后再打开。
USE_GNN_WARMSTART = False

# v2: projection 中允许优化部分 edge 的 t 参数。
OPTIMIZE_EDGE_T = True
MAX_EDGE_T_DELTA = 0.25
EDGE_T_REG_WEIGHT = 0.01

TRAIN_EPOCHS = 80
TRAIN_LR = 2e-3
TRAIN_WEIGHT_DECAY = 1e-4
TRAIN_BATCH_SIZE = 64
TRAIN_MIN_MATCHES = 32
TRAIN_PRINT_EVERY = 10

D_EDGE = 32
D_HIDDEN = 128
NUM_GNN_LAYERS = 4
DROPOUT = 0.10

MAX_DELTA_CENTER = 0.12
MAX_DELTA_THETA_RAD = math.radians(35.0)
MAX_DELTA_LOG_SCALE = 0.45

USE_PROJECTION = True
PROJECTION_ITERS = 48
PROJECTION_LR = 0.055
PROJECTION_PRIOR_WEIGHT = 0.02
PROJECTION_WARMSTART_WEIGHT = 0.0
PROJECTION_SCALE_WEIGHT = 0.01
PROJECTION_ANGLE_WEIGHT = 0.02
GRAD_CLIP_NORM = 3.0

GOOD_MAX_JUNCTION_PX = 1.20
GOOD_MAX_TX_ANGLE_DEG = 12.0
USABLE_MAX_JUNCTION_PX = 6.0
USABLE_MAX_TX_ANGLE_DEG = 30.0

HARD_EDGE_DENSITY_FILTER = True
MAX_EDGES_BY_N = {2: 1, 3: 4, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10}

STORE_SOLVED_POLYLINE = True
STORE_SOLVED_MOTHER_BEZIER = True
PRINT_EVERY = 50


# =========================================================
# Basic utils
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


def angle_wrap_np(a):
    a = float(a)
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


def angle_diff_np(a, b):
    return angle_wrap_np(float(a) - float(b))


def stable_sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


# =========================================================
# Candidate reading
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
    for k in keys:
        if isinstance(data, dict) and k in data and isinstance(data[k], list):
            return data[k]
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                print(f"[WARN] use unknown candidate key: {k}")
                return v
    return []


def get_candidate_id(candidate, idx=0):
    for k in ["generated_glyph_id", "glyph_candidate_id", "candidate_id", "sample_id", "grammar_sample_id"]:
        if isinstance(candidate, dict) and candidate.get(k, ""):
            return str(candidate[k])
    return f"glyph_candidate_{idx:05d}"


def get_nodes(candidate):
    for k in ["nodes", "primitive_nodes", "layout_nodes", "solved_nodes", "final_nodes", "optimized_nodes"]:
        if isinstance(candidate, dict) and k in candidate and isinstance(candidate[k], list):
            return candidate[k]
    return []


def get_solved_nodes(candidate):
    for k in ["solved_nodes", "final_nodes", "optimized_nodes", "layout_nodes", "nodes"]:
        if isinstance(candidate, dict) and k in candidate and isinstance(candidate[k], list):
            return candidate[k]
    return []


def get_edges(candidate):
    if not isinstance(candidate, dict):
        return []
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


# =========================================================
# Geometry reading
# =========================================================
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
        out = pts.copy(); out[:, 1] = pts[::-1, 1]; return out
    if v == 2:
        out = pts.copy(); out[:, 1] = -out[:, 1]; return out
    if v == 3:
        out = pts.copy(); out[:, 1] = -pts[::-1, 1]; return out
    return pts


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


def extract_local_polyline(node):
    return apply_variant_np(primitive_entry_to_polyline(node), node.get("variant_id", 0))


def transform_polyline_np(local, center, theta, scale):
    center = np.asarray(center, dtype=np.float32)
    local = np.asarray(local, dtype=np.float32)
    c = math.cos(theta); s = math.sin(theta)
    R = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    pts = local * (scale * CANVAS_SIZE)
    pts = pts @ R.T
    pts = pts + center[None, :] * CANVAS_SIZE
    return pts.astype(np.float32)


def extract_world_polyline_from_node(node, fallback_to_layout=True):
    for k in ["solved_polyline_px", "polyline_px", "world_polyline_px", "transformed_polyline_px", "render_polyline_px", "points_px", "polyline", "points"]:
        if k in node:
            arr = np.asarray(node[k], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) >= 2:
                arr = arr[:, :2]
                if np.max(np.abs(arr)) <= 1.5:
                    arr = arr * CANVAS_SIZE
                return arr
    if not fallback_to_layout:
        return None
    local = extract_local_polyline(node)
    tr = read_layout_transform(node)
    return transform_polyline_np(local, tr["center"], tr["theta"], tr["scale"])


def read_transform_from_solved_node(node):
    lp = node.get("layout_prior", {})
    if isinstance(lp, dict) and "center_norm" in lp:
        return read_layout_transform(node)
    if "center_norm" in node:
        return read_layout_transform(node)
    poly = extract_world_polyline_from_node(node, fallback_to_layout=False)
    if poly is not None and len(poly) >= 2:
        p0 = poly[0]; p1 = poly[-1]
        center = 0.5 * (p0 + p1) / CANVAS_SIZE
        d = p1 - p0
        length = float(np.linalg.norm(d)) / CANVAS_SIZE
        theta = float(math.atan2(d[1], d[0])) if np.linalg.norm(d) > 1e-6 else 0.0
        return {"center": center.astype(np.float32), "theta": theta, "scale": clamp(length, 0.02, 1.5)}
    return read_layout_transform(node)


def bezier_from_polyline(polyline):
    pts = np.asarray(polyline, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 2:
        return np.asarray([[180, 180], [190, 190], [210, 210], [220, 220]], dtype=np.float32)
    pts = pts[:, :2]
    p0 = pts[0]; p3 = pts[-1]
    if len(pts) >= 4:
        p1 = pts[max(1, len(pts) // 3)]
        p2 = pts[min(len(pts) - 2, 2 * len(pts) // 3)]
    elif len(pts) == 3:
        p1 = pts[1]; p2 = pts[1]
    else:
        p1 = p0 + (p3 - p0) / 3.0
        p2 = p0 + 2 * (p3 - p0) / 3.0
    return np.stack([p0, p1, p2, p3], axis=0).astype(np.float32)


def interp_polyline_np(poly, t):
    poly = np.asarray(poly, dtype=np.float32)
    if len(poly) <= 1:
        return poly[0]
    t = clamp(float(t), 0.0, 1.0)
    x = t * (len(poly) - 1)
    i = int(math.floor(x)); j = min(i + 1, len(poly) - 1)
    a = x - i
    return (1 - a) * poly[i] + a * poly[j]


def tangent_polyline_np(poly, t):
    poly = np.asarray(poly, dtype=np.float32)
    if len(poly) <= 1:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    t = clamp(float(t), 0.0, 1.0)
    x = t * (len(poly) - 1)
    i = int(round(x)); i0 = max(0, i - 1); i1 = min(len(poly) - 1, i + 1)
    d = poly[i1] - poly[i0]
    if np.linalg.norm(d) < 1e-8:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    return d / (np.linalg.norm(d) + 1e-8)


def angle_between_deg(v1, v2):
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    c = np.clip(float(np.dot(v1, v2)) / (float(n1) * float(n2)), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


# =========================================================
# Torch differentiable geometry
# =========================================================
def interp_polyline_torch(poly, t):
    P = poly.shape[0]
    if P <= 1:
        return poly[0]
    t = torch.clamp(t, 0.0, 1.0)
    x = t * float(P - 1)
    i = torch.floor(x).long().clamp(0, P - 1)
    j = (i + 1).clamp(0, P - 1)
    a = (x - i.float()).view(1)
    return (1 - a) * poly[i] + a * poly[j]


def tangent_polyline_torch(poly, t):
    P = poly.shape[0]
    if P <= 1:
        return torch.tensor([1.0, 0.0], device=poly.device, dtype=poly.dtype)
    t = torch.clamp(t, 0.0, 1.0)
    x = t * float(P - 1)
    i = torch.round(x).long()
    i0 = (i - 1).clamp(0, P - 1); i1 = (i + 1).clamp(0, P - 1)
    d = poly[i1] - poly[i0]
    return d / (torch.norm(d) + 1e-8)


def transform_point_torch(local_pt, center, theta, scale):
    c = torch.cos(theta); s = torch.sin(theta)
    x = local_pt[0] * scale * CANVAS_SIZE
    y = local_pt[1] * scale * CANVAS_SIZE
    wx = c * x - s * y + center[0] * CANVAS_SIZE
    wy = s * x + c * y + center[1] * CANVAS_SIZE
    return torch.stack([wx, wy], dim=0)


def transform_tangent_torch(local_tan, theta):
    c = torch.cos(theta); s = torch.sin(theta)
    x = local_tan[0]; y = local_tan[1]
    d = torch.stack([c * x - s * y, s * x + c * y], dim=0)
    return d / (torch.norm(d) + 1e-8)


# =========================================================
# Feature construction
# =========================================================
def normalize_edges(edges):
    clean = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        ee = copy.deepcopy(e)
        if "u" not in ee:
            for k in ["src", "source", "a", "node_u"]:
                if k in ee:
                    ee["u"] = ee[k]; break
        if "v" not in ee:
            for k in ["dst", "target", "b", "node_v"]:
                if k in ee:
                    ee["v"] = ee[k]; break
        if "u" not in ee or "v" not in ee:
            continue
        ee["u"] = safe_int(ee["u"], -1); ee["v"] = safe_int(ee["v"], -1)
        if ee["u"] < 0 or ee["v"] < 0:
            continue
        if "j_type" not in ee:
            idx = safe_int(ee.get("j_type_idx", 0), 0)
            ee["j_type"] = {1: "E2E", 2: "X", 3: "T"}.get(idx, "UNKNOWN")
        ee["t_u"] = clamp(safe_float(ee.get("t_u", ee.get("t_a", 0.0)), 0.0), 0.0, 1.0)
        ee["t_v"] = clamp(safe_float(ee.get("t_v", ee.get("t_b", 0.0)), 0.0), 0.0, 1.0)
        clean.append(ee)
    return clean


def primitive_straightness(node):
    poly = extract_local_polyline(node)
    if len(poly) < 2:
        return 1.0
    chord = float(np.linalg.norm(poly[-1] - poly[0]))
    seg = poly[1:] - poly[:-1]
    arc = float(np.sum(np.linalg.norm(seg, axis=1)))
    if arc < 1e-8:
        return 1.0
    return float(np.clip(chord / arc, 0.0, 1.0))


def graph_to_tensors(candidate):
    nodes = get_nodes(candidate)
    edges = normalize_edges(get_edges(candidate))
    N = len(nodes)
    if N <= 0 or N > MAX_NODES:
        return None
    id_to_idx = {node_id_of(n, i): i for i, n in enumerate(nodes)}
    node_feats = np.zeros((MAX_NODES, 12), dtype=np.float32)
    shape_ids = np.zeros((MAX_NODES,), dtype=np.int64)
    width_ids = np.zeros((MAX_NODES,), dtype=np.int64)
    mask = np.zeros((MAX_NODES,), dtype=np.float32)
    degree = Counter(); type_count = defaultdict(Counter)
    for e in edges:
        if e["u"] not in id_to_idx or e["v"] not in id_to_idx:
            continue
        iu = id_to_idx[e["u"]]; iv = id_to_idx[e["v"]]
        jt = str(e.get("j_type", "UNKNOWN"))
        degree[iu] += 1; degree[iv] += 1
        type_count[iu][jt] += 1; type_count[iv][jt] += 1
    init_transforms = []
    for i, n in enumerate(nodes):
        tr = read_layout_transform(n); init_transforms.append(tr)
        c = tr["center"]; th = tr["theta"]; sc = tr["scale"]
        node_feats[i] = np.asarray([
            c[0], c[1], math.sin(th), math.cos(th), sc,
            degree[i] / 8.0,
            type_count[i]["E2E"] / 8.0,
            type_count[i]["T"] / 8.0,
            type_count[i]["X"] / 8.0,
            primitive_straightness(n),
            len(extract_local_polyline(n)) / 64.0,
            1.0,
        ], dtype=np.float32)
        shape = safe_int(n.get("shape_code", n.get("shape_token", 0)), 0)
        shape_ids[i] = int(np.clip(shape, 0, MAX_SHAPE_CODE - 1))
        width = safe_int(n.get("width_token", 0), 0)
        width_ids[i] = int(np.clip(width, 0, MAX_WIDTH_TOKEN - 1))
        mask[i] = 1.0
    edge_type = np.zeros((MAX_NODES, MAX_NODES), dtype=np.int64)
    edge_feats = np.zeros((MAX_NODES, MAX_NODES, 8), dtype=np.float32)
    for e in edges:
        if e["u"] not in id_to_idx or e["v"] not in id_to_idx:
            continue
        iu = id_to_idx[e["u"]]; iv = id_to_idx[e["v"]]
        jid = JTYPE_TO_ID.get(str(e.get("j_type", "UNKNOWN")), 0)
        tu = e["t_u"]; tv = e["t_v"]
        ci = init_transforms[iu]["center"]; cj = init_transforms[iv]["center"]
        d = cj - ci
        feat_uv = np.asarray([tu, tv, abs(tu - tv), tu * tv, d[0], d[1], math.sin(init_transforms[iv]["theta"] - init_transforms[iu]["theta"]), math.cos(init_transforms[iv]["theta"] - init_transforms[iu]["theta"])], dtype=np.float32)
        feat_vu = np.asarray([tv, tu, abs(tu - tv), tu * tv, -d[0], -d[1], math.sin(init_transforms[iu]["theta"] - init_transforms[iv]["theta"]), math.cos(init_transforms[iu]["theta"] - init_transforms[iv]["theta"])], dtype=np.float32)
        edge_type[iu, iv] = jid; edge_type[iv, iu] = jid
        edge_feats[iu, iv] = feat_uv; edge_feats[iv, iu] = feat_vu
    return {"node_feats": node_feats, "shape_ids": shape_ids, "width_ids": width_ids, "edge_type": edge_type, "edge_feats": edge_feats, "mask": mask, "nodes": nodes, "edges": edges, "id_to_idx": id_to_idx, "init_transforms": init_transforms}


# =========================================================
# GNN model
# =========================================================
class SolverWarmstartGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape_emb = nn.Embedding(MAX_SHAPE_CODE, 16)
        self.width_emb = nn.Embedding(MAX_WIDTH_TOKEN, 4)
        self.jtype_emb = nn.Embedding(4, D_EDGE)
        self.node_proj = nn.Sequential(nn.Linear(12 + 16 + 4, D_HIDDEN), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(D_HIDDEN, D_HIDDEN), nn.GELU())
        self.edge_proj = nn.Sequential(nn.Linear(8 + D_EDGE, D_HIDDEN), nn.GELU(), nn.Linear(D_HIDDEN, D_HIDDEN), nn.GELU())
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "msg": nn.Sequential(nn.Linear(D_HIDDEN * 2, D_HIDDEN), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(D_HIDDEN, D_HIDDEN)),
                "upd": nn.Sequential(nn.Linear(D_HIDDEN * 2, D_HIDDEN), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(D_HIDDEN, D_HIDDEN)),
                "norm": nn.LayerNorm(D_HIDDEN),
            }) for _ in range(NUM_GNN_LAYERS)
        ])
        self.delta_head = nn.Sequential(nn.Linear(D_HIDDEN, D_HIDDEN), nn.GELU(), nn.Linear(D_HIDDEN, 4))
        self.quality_head = nn.Sequential(nn.Linear(D_HIDDEN * 2, D_HIDDEN), nn.GELU(), nn.Linear(D_HIDDEN, 3))

    def forward(self, node_feats, shape_ids, width_ids, edge_type, edge_feats, mask):
        shape = self.shape_emb(shape_ids)
        width = self.width_emb(width_ids)
        x = torch.cat([node_feats, shape, width], dim=-1)
        h = self.node_proj(x)
        edge_emb = self.jtype_emb(edge_type)
        ef = self.edge_proj(torch.cat([edge_feats, edge_emb], dim=-1))
        B, N, _ = h.shape
        node_mask = mask.unsqueeze(-1)
        pair_mask = (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(-1)
        edge_exist = (edge_type > 0).float().unsqueeze(-1) * pair_mask
        h = h * node_mask
        for layer in self.layers:
            hj = h.unsqueeze(1).expand(B, N, N, D_HIDDEN)
            msg = layer["msg"](torch.cat([hj, ef], dim=-1)) * edge_exist
            denom = edge_exist.sum(dim=2).clamp_min(1.0)
            agg = msg.sum(dim=2) / denom
            upd = layer["upd"](torch.cat([h, agg], dim=-1))
            h = layer["norm"](h + upd) * node_mask
        raw = self.delta_head(h)
        dc = torch.tanh(raw[..., 0:2]) * MAX_DELTA_CENTER
        dtheta = torch.tanh(raw[..., 2:3]) * MAX_DELTA_THETA_RAD
        dlog = torch.tanh(raw[..., 3:4]) * MAX_DELTA_LOG_SCALE
        delta = torch.cat([dc, dtheta, dlog], dim=-1) * node_mask
        mean_pool = (h * node_mask).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        max_pool = h.masked_fill(node_mask == 0, -1e6).max(dim=1).values
        quality = self.quality_head(torch.cat([mean_pool, max_pool], dim=-1))
        return delta, quality


# =========================================================
# Training from teacher
# =========================================================
def load_teacher_solved():
    path = None
    if os.path.exists(TEACHER_SOLVED_FILE):
        path = TEACHER_SOLVED_FILE
    elif os.path.exists(OLD_SOLVED_FILE):
        path = OLD_SOLVED_FILE
    if path is None:
        return None, {}
    data = load_json(path, required=False)
    if data is None:
        return None, {}
    solved_list = get_candidate_list(data)
    return path, {get_candidate_id(c, i): c for i, c in enumerate(solved_list)}


def build_teacher_targets(candidate, solved_candidate):
    nodes = get_nodes(candidate)
    solved_nodes = get_solved_nodes(solved_candidate)
    if not nodes or not solved_nodes:
        return None
    solved_by_id = {node_id_of(n, i): n for i, n in enumerate(solved_nodes)}
    target = np.zeros((MAX_NODES, 4), dtype=np.float32)
    valid = np.zeros((MAX_NODES,), dtype=np.float32)
    for i, n in enumerate(nodes):
        nid = node_id_of(n, i)
        if nid not in solved_by_id:
            continue
        init_tr = read_layout_transform(n)
        sol_tr = read_transform_from_solved_node(solved_by_id[nid])
        dc = sol_tr["center"] - init_tr["center"]
        dtheta = angle_diff_np(sol_tr["theta"], init_tr["theta"])
        dlog = math.log(max(1e-6, sol_tr["scale"]) / max(1e-6, init_tr["scale"]))
        target[i, 0:2] = np.clip(dc, -MAX_DELTA_CENTER, MAX_DELTA_CENTER)
        target[i, 2] = np.clip(dtheta, -MAX_DELTA_THETA_RAD, MAX_DELTA_THETA_RAD)
        target[i, 3] = np.clip(dlog, -MAX_DELTA_LOG_SCALE, MAX_DELTA_LOG_SCALE)
        valid[i] = 1.0
    if valid.sum() == 0:
        return None
    q = solved_candidate.get("quality_report", {})
    if not isinstance(q, dict):
        q = {}
    status = q.get("quality_status", solved_candidate.get("quality_status", "unknown"))
    maxJ = safe_float(q.get("max_junction_px", solved_candidate.get("max_junction_px", 0.0)), 0.0)
    maxA = safe_float(q.get("max_tx_angle_deg", q.get("max_angle_diff_deg", solved_candidate.get("maxTXA", 0.0))), 0.0)
    bad = 1.0 if str(status) == "bad" else 0.0
    quality_target = np.asarray([maxJ / 10.0, maxA / 90.0, bad], dtype=np.float32)
    return {"target_delta": target, "target_valid": valid, "quality_target": quality_target}


def build_training_dataset(candidates, teacher_by_id):
    dataset = []
    skipped = Counter()
    for idx, cand in enumerate(candidates):
        cid = get_candidate_id(cand, idx)
        if cid not in teacher_by_id:
            skipped["no_teacher_match"] += 1; continue
        g = graph_to_tensors(cand)
        if g is None:
            skipped["bad_graph"] += 1; continue
        tgt = build_teacher_targets(cand, teacher_by_id[cid])
        if tgt is None:
            skipped["bad_target"] += 1; continue
        item = {"candidate_id": cid, **{k: g[k] for k in ["node_feats", "shape_ids", "width_ids", "edge_type", "edge_feats", "mask"]}, **tgt}
        dataset.append(item)
    return dataset, skipped


def batch_items(items, device):
    def stack(k, dtype):
        return torch.tensor(np.stack([it[k] for it in items], axis=0), dtype=dtype, device=device)
    return {"node_feats": stack("node_feats", torch.float32), "shape_ids": stack("shape_ids", torch.long), "width_ids": stack("width_ids", torch.long), "edge_type": stack("edge_type", torch.long), "edge_feats": stack("edge_feats", torch.float32), "mask": stack("mask", torch.float32), "target_delta": stack("target_delta", torch.float32), "target_valid": stack("target_valid", torch.float32), "quality_target": stack("quality_target", torch.float32)}


def train_model_if_possible(model, candidates, device):
    teacher_path, teacher_by_id = load_teacher_solved()
    if teacher_path is None:
        print("\n[Warmstart Model]\n  teacher: not found\n  mode: identity warm-start + projection only")
        return model, {"trained": False, "teacher_path": None, "train_size": 0, "reason": "teacher_not_found"}
    dataset, skipped = build_training_dataset(candidates, teacher_by_id)
    print("\n[Warmstart Dataset]")
    print(f"  teacher_path: {teacher_path}")
    print(f"  teacher_count: {len(teacher_by_id)}")
    print(f"  matched_train_samples: {len(dataset)}")
    print(f"  skipped: {dict(skipped)}")
    if len(dataset) < TRAIN_MIN_MATCHES:
        print(f"  train skipped: dataset smaller than TRAIN_MIN_MATCHES={TRAIN_MIN_MATCHES}")
        return model, {"trained": False, "teacher_path": teacher_path, "train_size": len(dataset), "reason": "too_few_matches"}
    if os.path.exists(OUTPUT_MODEL_FILE) and not FORCE_RETRAIN:
        try:
            ckpt = torch.load(OUTPUT_MODEL_FILE, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print("\n[Warmstart Model]")
            print(f"  loaded existing model: {OUTPUT_MODEL_FILE}")
            print(f"  checkpoint_train_size: {ckpt.get('train_size')}")
            return model, {"trained": False, "loaded": True, "teacher_path": teacher_path, "train_size": len(dataset)}
        except Exception as e:
            print(f"  load existing failed, retrain: {e}")
    print("\n[Train SolverWarmstartGNN]")
    print(f"  epochs: {TRAIN_EPOCHS}")
    print(f"  batch_size: {TRAIN_BATCH_SIZE}")
    print(f"  lr: {TRAIN_LR}")
    print(f"  device: {device}")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY)
    rng = random.Random(RANDOM_SEED)
    history = []
    n = len(dataset)
    for epoch in range(1, TRAIN_EPOCHS + 1):
        rng.shuffle(dataset)
        total_loss = total_delta = total_quality = total_count = 0.0
        for bi in range(0, n, TRAIN_BATCH_SIZE):
            batch = dataset[bi:bi + TRAIN_BATCH_SIZE]
            b = batch_items(batch, device)
            pred_delta, pred_quality = model(b["node_feats"], b["shape_ids"], b["width_ids"], b["edge_type"], b["edge_feats"], b["mask"])
            valid = b["target_valid"].unsqueeze(-1) * b["mask"].unsqueeze(-1)
            delta_loss = ((pred_delta - b["target_delta"]) ** 2 * valid).sum() / valid.sum().clamp_min(1.0)
            q = b["quality_target"]
            q_loss = F.smooth_l1_loss(pred_quality[:, 0], q[:, 0]) + F.smooth_l1_loss(pred_quality[:, 1], q[:, 1])
            bce = F.binary_cross_entropy_with_logits(pred_quality[:, 2], q[:, 2])
            quality_loss = q_loss + 0.5 * bce
            loss = delta_loss + 0.25 * quality_loss
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            bs = len(batch)
            total_loss += float(loss.detach().cpu()) * bs
            total_delta += float(delta_loss.detach().cpu()) * bs
            total_quality += float(quality_loss.detach().cpu()) * bs
            total_count += bs
        rec = {"epoch": epoch, "loss": total_loss / total_count, "delta_loss": total_delta / total_count, "quality_loss": total_quality / total_count}
        history.append(rec)
        if epoch == 1 or epoch % TRAIN_PRINT_EVERY == 0 or epoch == TRAIN_EPOCHS:
            print(f"  epoch {epoch:03d} | loss={rec['loss']:.6f} | delta={rec['delta_loss']:.6f} | quality={rec['quality_loss']:.6f}")
    torch.save({"schema_version": "solver_warmstart_gnn_v1", "model_state_dict": model.state_dict(), "train_size": len(dataset), "teacher_path": teacher_path, "history": history}, OUTPUT_MODEL_FILE)
    print(f"  saved model: {OUTPUT_MODEL_FILE}")
    model.eval()
    return model, {"trained": True, "teacher_path": teacher_path, "train_size": len(dataset), "history_last": history[-1]}


def load_or_init_model(device):
    model = SolverWarmstartGNN().to(device)
    model.eval()
    if os.path.exists(OUTPUT_MODEL_FILE) and not FORCE_RETRAIN:
        try:
            ckpt = torch.load(OUTPUT_MODEL_FILE, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print("\n[Warmstart Model]")
            print(f"  loaded: {OUTPUT_MODEL_FILE}")
            print(f"  train_size: {ckpt.get('train_size')}")
            return model, {"loaded": True, "path": OUTPUT_MODEL_FILE}
        except Exception as e:
            print("\n[Warmstart Model]")
            print(f"  load failed, use fresh model: {e}")
    print("\n[Warmstart Model]")
    print("  no trained model loaded yet")
    return model, {"loaded": False}


# =========================================================
# Projection and evaluation
# =========================================================
def initial_arrays_from_graph(graph):
    N = int(graph["mask"].sum())
    centers, thetas, scales, locals_ = [], [], [], []
    for i in range(N):
        tr = graph["init_transforms"][i]
        centers.append(tr["center"]); thetas.append(tr["theta"]); scales.append(tr["scale"])
        locals_.append(extract_local_polyline(graph["nodes"][i]))
    return np.asarray(centers, dtype=np.float32), np.asarray(thetas, dtype=np.float32), np.asarray(scales, dtype=np.float32), locals_


def apply_delta_np(centers, thetas, scales, delta):
    N = len(centers)
    d = delta[:N]
    c = np.clip(centers + d[:, 0:2], 0.02, 0.98)
    th = np.asarray([angle_wrap_np(x) for x in (thetas + d[:, 2])], dtype=np.float32)
    sc = np.clip(scales * np.exp(d[:, 3]), 0.02, 1.5)
    return c, th, sc


def evaluate_geometry(edges, id_to_idx, locals_, centers, thetas, scales):
    junctions, angles, edge_debug = [], [], []
    world_polys = [transform_polyline_np(locals_[i], centers[i], thetas[i], scales[i]) for i in range(len(locals_))]
    for e in edges:
        u, v = e["u"], e["v"]
        if u not in id_to_idx or v not in id_to_idx:
            continue
        iu, iv = id_to_idx[u], id_to_idx[v]
        tu = clamp(safe_float(e.get("t_u", 0.0), 0.0), 0.0, 1.0)
        tv = clamp(safe_float(e.get("t_v", 0.0), 0.0), 0.0, 1.0)
        pu = interp_polyline_np(world_polys[iu], tu)
        pv = interp_polyline_np(world_polys[iv], tv)
        dist = float(np.linalg.norm(pu - pv)); junctions.append(dist)
        jt = str(e.get("j_type", "UNKNOWN")); angle_diff = 0.0
        if jt in ["T", "X"]:
            du = tangent_polyline_np(world_polys[iu], tu); dv = tangent_polyline_np(world_polys[iv], tv)
            ang = angle_between_deg(du, dv)
            angle_diff = max(0.0, 35.0 - min(ang, 180.0 - ang))
            angles.append(angle_diff)
        edge_debug.append({"u": int(u), "v": int(v), "j_type": jt, "t_u": round(float(tu), 4), "t_v": round(float(tv), 4), "junction_px": round(float(dist), 4), "angle_diff_deg": round(float(angle_diff), 4), "position_u": [round(float(pu[0]), 3), round(float(pu[1]), 3)], "position_v": [round(float(pv[0]), 3), round(float(pv[1]), 3)]})
    if not junctions:
        return {"mean_junction_px": 999.0, "max_junction_px": 999.0, "mean_tx_angle_deg": 999.0, "max_tx_angle_deg": 999.0, "edge_debug": edge_debug, "world_polys": world_polys}
    return {"mean_junction_px": float(np.mean(junctions)), "max_junction_px": float(np.max(junctions)), "mean_tx_angle_deg": float(np.mean(angles)) if angles else 0.0, "max_tx_angle_deg": float(np.max(angles)) if angles else 0.0, "edge_debug": edge_debug, "world_polys": world_polys}


def classify_quality(maxJ, maxA, hard_case=False):
    if hard_case:
        if maxJ <= GOOD_MAX_JUNCTION_PX and maxA <= GOOD_MAX_TX_ANGLE_DEG:
            return "usable_but_rough"
        return "bad"
    if maxJ <= GOOD_MAX_JUNCTION_PX and maxA <= GOOD_MAX_TX_ANGLE_DEG:
        return "good"
    if maxJ <= USABLE_MAX_JUNCTION_PX and maxA <= USABLE_MAX_TX_ANGLE_DEG:
        return "usable_but_rough"
    return "bad"


def is_hard_topology(graph):
    N = int(graph["mask"].sum())
    E = len(graph["edges"])
    if not HARD_EDGE_DENSITY_FILTER:
        return False
    return E > MAX_EDGES_BY_N.get(N, N + 2)


def make_optimizable_edges(edges):
    """
    给每条 edge 添加 t_u/t_v 优化掩码。
    规则：
      - E2E：通常端点连接，t 固定。
      - T：如果某一端 t 在内部区间，则允许优化该端；端点不动。
      - X：两端通常都允许优化。
      - 兜底：只要 t 不接近 0/1，就允许小范围优化。
    """
    out = []
    for e in edges:
        ee = copy.deepcopy(e)
        jt = str(ee.get("j_type", "UNKNOWN"))
        tu = clamp(safe_float(ee.get("t_u", 0.0), 0.0), 0.0, 1.0)
        tv = clamp(safe_float(ee.get("t_v", 0.0), 0.0), 0.0, 1.0)
        ee["t_u"] = tu
        ee["t_v"] = tv

        def interior(t):
            return 0.08 < float(t) < 0.92

        if not OPTIMIZE_EDGE_T:
            mu, mv = 0.0, 0.0
        elif jt == "E2E":
            mu, mv = 0.0, 0.0
        elif jt == "X":
            mu, mv = 1.0, 1.0
        elif jt == "T":
            # 对 T：端点通常是 guest，不动；host 的内部 t 可动。
            mu, mv = float(interior(tu)), float(interior(tv))
            # 如果两边都像端点，至少允许其中一边轻微移动，避免完全无自由度。
            if mu == 0.0 and mv == 0.0:
                if abs(tu - 0.5) < abs(tv - 0.5):
                    mu = 0.5
                else:
                    mv = 0.5
        else:
            mu, mv = float(interior(tu)), float(interior(tv))

        ee["_opt_t_u_mask"] = mu
        ee["_opt_t_v_mask"] = mv
        out.append(ee)
    return out


def projection_refine(graph, warm_centers, warm_thetas, warm_scales, device):
    """
    v2 projection:
      - 优化 stroke transform: center / theta / log_scale
      - 同时优化部分 edge t: T / X / interior t
      - 返回 opt_edges，后续 quality 和输出都使用优化后的 t
    """
    N = int(graph["mask"].sum())
    edges0 = make_optimizable_edges(graph["edges"])
    id_to_idx = graph["id_to_idx"]
    nodes = graph["nodes"]

    locals_np = [extract_local_polyline(nodes[i]) for i in range(N)]
    locals_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in locals_np]

    prior_centers, prior_thetas, prior_scales, _ = initial_arrays_from_graph(graph)

    center = torch.tensor(warm_centers, dtype=torch.float32, device=device, requires_grad=True)
    theta = torch.tensor(warm_thetas, dtype=torch.float32, device=device, requires_grad=True)
    log_scale = torch.tensor(np.log(np.maximum(warm_scales, 1e-6)), dtype=torch.float32, device=device, requires_grad=True)

    warm_center_t = torch.tensor(warm_centers, dtype=torch.float32, device=device)
    warm_theta_t = torch.tensor(warm_thetas, dtype=torch.float32, device=device)
    warm_log_scale_t = torch.tensor(np.log(np.maximum(warm_scales, 1e-6)), dtype=torch.float32, device=device)

    prior_center_t = torch.tensor(prior_centers, dtype=torch.float32, device=device)
    prior_theta_t = torch.tensor(prior_thetas, dtype=torch.float32, device=device)
    prior_log_scale_t = torch.tensor(np.log(np.maximum(prior_scales, 1e-6)), dtype=torch.float32, device=device)

    # edge t 参数。raw_t_delta 通过 tanh 限制在 ±MAX_EDGE_T_DELTA。
    E = len(edges0)
    t_init = np.zeros((E, 2), dtype=np.float32)
    t_mask = np.zeros((E, 2), dtype=np.float32)
    for ei, e in enumerate(edges0):
        t_init[ei, 0] = clamp(safe_float(e.get("t_u", 0.0), 0.0), 0.0, 1.0)
        t_init[ei, 1] = clamp(safe_float(e.get("t_v", 0.0), 0.0), 0.0, 1.0)
        t_mask[ei, 0] = safe_float(e.get("_opt_t_u_mask", 0.0), 0.0)
        t_mask[ei, 1] = safe_float(e.get("_opt_t_v_mask", 0.0), 0.0)

    t_init_t = torch.tensor(t_init, dtype=torch.float32, device=device)
    t_mask_t = torch.tensor(t_mask, dtype=torch.float32, device=device)
    raw_t_delta = torch.zeros((E, 2), dtype=torch.float32, device=device, requires_grad=True)

    params = [center, theta, log_scale]
    if OPTIMIZE_EDGE_T and E > 0 and float(t_mask.sum()) > 0:
        params.append(raw_t_delta)

    opt = torch.optim.Adam(params, lr=PROJECTION_LR)
    last_loss = 0.0
    last_junction = 0.0
    last_t_reg = 0.0

    for _ in range(PROJECTION_ITERS):
        opt.zero_grad()
        scale = torch.exp(log_scale).clamp(0.02, 1.5)

        if E > 0:
            t_cur = torch.clamp(t_init_t + torch.tanh(raw_t_delta) * MAX_EDGE_T_DELTA * t_mask_t, 0.0, 1.0)
        else:
            t_cur = t_init_t

        junction_terms, angle_terms = [], []

        for ei, e in enumerate(edges0):
            u, v = e["u"], e["v"]
            if u not in id_to_idx or v not in id_to_idx:
                continue
            iu, iv = id_to_idx[u], id_to_idx[v]
            if iu >= N or iv >= N:
                continue

            tu = t_cur[ei, 0]
            tv = t_cur[ei, 1]

            li = interp_polyline_torch(locals_t[iu], tu)
            lj = interp_polyline_torch(locals_t[iv], tv)
            pi = transform_point_torch(li, center[iu], theta[iu], scale[iu])
            pj = transform_point_torch(lj, center[iv], theta[iv], scale[iv])
            junction_terms.append(((pi - pj) ** 2).sum())

            if str(e.get("j_type", "UNKNOWN")) in ["T", "X"]:
                ti = tangent_polyline_torch(locals_t[iu], tu)
                tj = tangent_polyline_torch(locals_t[iv], tv)
                wi = transform_tangent_torch(ti, theta[iu])
                wj = transform_tangent_torch(tj, theta[iv])
                cosv = torch.clamp(torch.abs((wi * wj).sum()), 0.0, 1.0)
                angle_terms.append(F.relu(cosv - math.cos(math.radians(35.0))) ** 2)

        junction_loss = torch.stack(junction_terms).mean() if junction_terms else torch.tensor(1000.0, device=device)
        angle_loss = torch.stack(angle_terms).mean() * (CANVAS_SIZE ** 2) if angle_terms else torch.tensor(0.0, device=device)

        prior_loss = (
            ((center - prior_center_t) ** 2).mean() * (CANVAS_SIZE ** 2)
            + ((theta - prior_theta_t) ** 2).mean() * 8.0
            + ((log_scale - prior_log_scale_t) ** 2).mean() * 8.0
        )
        warm_loss = (
            ((center - warm_center_t) ** 2).mean() * (CANVAS_SIZE ** 2)
            + ((theta - warm_theta_t) ** 2).mean() * 8.0
            + ((log_scale - warm_log_scale_t) ** 2).mean() * 8.0
        )
        scale_loss = (F.relu(log_scale - math.log(1.5)) ** 2 + F.relu(math.log(0.02) - log_scale) ** 2).mean()

        if E > 0:
            t_delta = (t_cur - t_init_t) * t_mask_t
            t_reg_loss = (t_delta ** 2).mean() * (CANVAS_SIZE ** 2)
        else:
            t_reg_loss = torch.tensor(0.0, device=device)

        loss = (
            junction_loss
            + PROJECTION_ANGLE_WEIGHT * angle_loss
            + PROJECTION_PRIOR_WEIGHT * prior_loss
            + PROJECTION_WARMSTART_WEIGHT * warm_loss
            + PROJECTION_SCALE_WEIGHT * scale_loss
            + EDGE_T_REG_WEIGHT * t_reg_loss
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
        opt.step()

        with torch.no_grad():
            center.clamp_(0.02, 0.98)
            log_scale.clamp_(math.log(0.02), math.log(1.5))

        last_loss = float(loss.detach().cpu())
        last_junction = float(junction_loss.detach().cpu())
        last_t_reg = float(t_reg_loss.detach().cpu())

    out_centers = center.detach().cpu().numpy().astype(np.float32)
    out_thetas = theta.detach().cpu().numpy().astype(np.float32)
    out_thetas = np.asarray([angle_wrap_np(x) for x in out_thetas], dtype=np.float32)
    out_scales = np.clip(torch.exp(log_scale).detach().cpu().numpy().astype(np.float32), 0.02, 1.5)

    if E > 0:
        t_final = torch.clamp(t_init_t + torch.tanh(raw_t_delta) * MAX_EDGE_T_DELTA * t_mask_t, 0.0, 1.0).detach().cpu().numpy()
    else:
        t_final = t_init

    opt_edges = []
    num_t_moved = 0
    max_t_delta = 0.0
    for ei, e in enumerate(edges0):
        ee = copy.deepcopy(e)
        old_tu = safe_float(ee.get("t_u", 0.0), 0.0)
        old_tv = safe_float(ee.get("t_v", 0.0), 0.0)
        ee["t_u"] = float(t_final[ei, 0])
        ee["t_v"] = float(t_final[ei, 1])
        dt = max(abs(ee["t_u"] - old_tu), abs(ee["t_v"] - old_tv))
        if dt > 1e-4:
            num_t_moved += 1
        max_t_delta = max(max_t_delta, float(dt))
        ee.pop("_opt_t_u_mask", None)
        ee.pop("_opt_t_v_mask", None)
        opt_edges.append(ee)

    info = {
        "projection_loss": last_loss,
        "projection_junction_loss": last_junction,
        "projection_t_reg_loss": last_t_reg,
        "num_t_moved_edges": int(num_t_moved),
        "max_t_delta": float(max_t_delta),
        "optimize_edge_t": bool(OPTIMIZE_EDGE_T),
    }
    return out_centers, out_thetas, out_scales, info, opt_edges

def solve_candidate_neural(candidate, model, device, idx=0):
    cid = get_candidate_id(candidate, idx)
    g = graph_to_tensors(candidate)
    if g is None:
        out = copy.deepcopy(candidate)
        out["quality_report"] = {"quality_status": "bad", "quality_score": 999.0, "reason": "bad_graph"}
        return out, {"status": "bad", "reason": "bad_graph"}
    N = int(g["mask"].sum())
    hard_topo = is_hard_topology(g)
    centers0, thetas0, scales0, locals_ = initial_arrays_from_graph(g)
    before_eval = evaluate_geometry(g["edges"], g["id_to_idx"], locals_, centers0, thetas0, scales0)
    if USE_GNN_WARMSTART:
        with torch.no_grad():
            b = {
                "node_feats": torch.tensor(g["node_feats"][None, ...], dtype=torch.float32, device=device),
                "shape_ids": torch.tensor(g["shape_ids"][None, ...], dtype=torch.long, device=device),
                "width_ids": torch.tensor(g["width_ids"][None, ...], dtype=torch.long, device=device),
                "edge_type": torch.tensor(g["edge_type"][None, ...], dtype=torch.long, device=device),
                "edge_feats": torch.tensor(g["edge_feats"][None, ...], dtype=torch.float32, device=device),
                "mask": torch.tensor(g["mask"][None, ...], dtype=torch.float32, device=device),
            }
            delta_t, quality_t = model(b["node_feats"], b["shape_ids"], b["width_ids"], b["edge_type"], b["edge_feats"], b["mask"])
            delta = delta_t[0].detach().cpu().numpy().astype(np.float32)
            quality_pred = quality_t[0].detach().cpu().numpy().astype(np.float32)
    else:
        delta = np.zeros((MAX_NODES, 4), dtype=np.float32)
        quality_pred = np.zeros((3,), dtype=np.float32)
    warm_centers, warm_thetas, warm_scales = apply_delta_np(centers0, thetas0, scales0, delta)
    warm_eval = evaluate_geometry(g["edges"], g["id_to_idx"], locals_, warm_centers, warm_thetas, warm_scales)
    if USE_PROJECTION:
        centers, thetas, scales, proj_info, opt_edges = projection_refine(g, warm_centers, warm_thetas, warm_scales, device)
    else:
        centers, thetas, scales = warm_centers, warm_thetas, warm_scales
        proj_info = {"projection_loss": 0.0, "num_t_moved_edges": 0, "max_t_delta": 0.0}
        opt_edges = g["edges"]
    final_eval = evaluate_geometry(opt_edges, g["id_to_idx"], locals_, centers, thetas, scales)
    maxJ = final_eval["max_junction_px"]; meanJ = final_eval["mean_junction_px"]
    maxA = final_eval["max_tx_angle_deg"]; meanA = final_eval["mean_tx_angle_deg"]
    status = classify_quality(maxJ, maxA, hard_case=hard_topo)
    quality_score = float(meanJ + 0.5 * maxJ + 0.03 * maxA)
    solved_nodes = []
    for i, node in enumerate(g["nodes"]):
        nnod = copy.deepcopy(node)
        c = centers[i]; th = float(thetas[i]); sc = float(scales[i])
        lp = nnod.setdefault("layout_prior", {})
        lp["center_norm"] = [float(c[0]), float(c[1])]
        lp["rotation_rad"] = th; lp["rotation_deg"] = float(th * 180.0 / math.pi)
        lp["length_norm"] = sc; lp["scale_norm"] = sc
        d = np.asarray([math.cos(th), math.sin(th)], dtype=np.float32) * sc * 0.5
        lp["p0_norm"] = [float((c - d)[0]), float((c - d)[1])]
        lp["p3_norm"] = [float((c + d)[0]), float((c + d)[1])]
        nnod["center_norm"] = lp["center_norm"]; nnod["rotation_rad"] = lp["rotation_rad"]; nnod["length_norm"] = lp["length_norm"]; nnod["scale_norm"] = lp["scale_norm"]
        poly = final_eval["world_polys"][i]
        if STORE_SOLVED_POLYLINE:
            nnod["solved_polyline_px"] = poly.astype(float).tolist()
        if STORE_SOLVED_MOTHER_BEZIER:
            nnod["mother_bezier"] = bezier_from_polyline(poly).astype(float).tolist()
        solved_nodes.append(nnod)
    out = copy.deepcopy(candidate)
    out["solved_nodes"] = solved_nodes
    out["quality_status"] = status
    out["quality_score"] = quality_score
    out["edge_debug"] = final_eval["edge_debug"]
    if isinstance(out.get("topology"), dict):
        out["topology"]["positive_edges_undirected"] = opt_edges
        out["topology"]["edges"] = opt_edges
    else:
        out["topology"] = {"positive_edges_undirected": opt_edges, "edges": opt_edges}
    out["neural_solver_debug"] = {
        "candidate_id": cid, "num_nodes": N, "num_edges": len(g["edges"]), "hard_topology": bool(hard_topo),
        "before_mean_junction_px": float(before_eval["mean_junction_px"]), "before_max_junction_px": float(before_eval["max_junction_px"]),
        "warm_mean_junction_px": float(warm_eval["mean_junction_px"]), "warm_max_junction_px": float(warm_eval["max_junction_px"]),
        "final_mean_junction_px": float(meanJ), "final_max_junction_px": float(maxJ), "final_max_tx_angle_deg": float(maxA),
        "pred_maxJ_norm": float(quality_pred[0]), "pred_maxA_norm": float(quality_pred[1]), "pred_bad_logit": float(quality_pred[2]), "pred_bad_prob": float(stable_sigmoid(quality_pred[2])),
        "projection_loss": safe_float(proj_info.get("projection_loss", 0.0), 0.0),
        "num_t_moved_edges": int(proj_info.get("num_t_moved_edges", 0)),
        "max_t_delta": float(proj_info.get("max_t_delta", 0.0)),
        "use_gnn_warmstart": bool(USE_GNN_WARMSTART),
    }
    out["quality_report"] = {"quality_status": status, "quality_score": quality_score, "mean_junction_px": float(meanJ), "max_junction_px": float(maxJ), "mean_tx_angle_deg": float(meanA), "max_tx_angle_deg": float(maxA), "max_angle_diff_deg": float(maxA), "solver_type": "neural_projection_v2", "projection_iters": int(PROJECTION_ITERS), "hard_topology": bool(hard_topo)}
    return out, out["neural_solver_debug"]


def summarize_results(results, debug_items, elapsed):
    status_hist = Counter(); node_hist = Counter(); edge_hist = Counter(); hard_count = 0
    before_max, warm_max, final_max, final_angle, quality_scores = [], [], [], [], []
    for r, d in zip(results, debug_items):
        q = r.get("quality_report", {})
        st = q.get("quality_status", r.get("quality_status", "unknown"))
        status_hist[st] += 1; node_hist[d.get("num_nodes", 0)] += 1; edge_hist[d.get("num_edges", 0)] += 1
        hard_count += int(bool(d.get("hard_topology", False)))
        before_max.append(d.get("before_max_junction_px", 0.0)); warm_max.append(d.get("warm_max_junction_px", 0.0)); final_max.append(d.get("final_max_junction_px", 0.0)); final_angle.append(d.get("final_max_tx_angle_deg", 0.0)); quality_scores.append(r.get("quality_score", 0.0))
    def stats(vals):
        vals = np.asarray(vals, dtype=np.float32)
        if len(vals) == 0: return {}
        return {"mean": round(float(np.mean(vals)), 6), "p50": round(float(np.percentile(vals, 50)), 6), "p90": round(float(np.percentile(vals, 90)), 6), "max": round(float(np.max(vals)), 6)}
    return {"count": len(results), "elapsed_sec": round(float(elapsed), 3), "cand_per_sec": round(float(len(results) / max(elapsed, 1e-6)), 3), "status_hist": dict(status_hist), "node_hist": dict(node_hist), "edge_hist": dict(edge_hist), "hard_topology_count": hard_count, "before_maxJ_stats": stats(before_max), "warm_maxJ_stats": stats(warm_max), "final_maxJ_stats": stats(final_max), "final_maxTXA_stats": stats(final_angle), "quality_score_stats": stats(quality_scores), "good_or_usable_count": status_hist.get("good", 0) + status_hist.get("usable_but_rough", 0)}


# =========================================================
# Main
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
    print("Neural Constraint Solver")
    print("=" * 80)
    print(f"  input_file:   {INPUT_FILE}")
    print(f"  output_file:  {OUTPUT_FILE}")
    print(f"  model_file:   {OUTPUT_MODEL_FILE}")
    print(f"  device:       {device}")
    print("=" * 80)
    print("\n[Key Config]")
    print(f"  MAX_NODES: {MAX_NODES}")
    print(f"  USE_PROJECTION: {USE_PROJECTION}")
    print(f"  USE_GNN_WARMSTART: {USE_GNN_WARMSTART}")
    print(f"  OPTIMIZE_EDGE_T: {OPTIMIZE_EDGE_T}")
    print(f"  MAX_EDGE_T_DELTA: {MAX_EDGE_T_DELTA}")
    print(f"  EDGE_T_REG_WEIGHT: {EDGE_T_REG_WEIGHT}")
    print(f"  PROJECTION_ITERS: {PROJECTION_ITERS}")
    print(f"  PROJECTION_LR: {PROJECTION_LR}")
    print(f"  PROJECTION_PRIOR_WEIGHT: {PROJECTION_PRIOR_WEIGHT}")
    print(f"  PROJECTION_WARMSTART_WEIGHT: {PROJECTION_WARMSTART_WEIGHT}")
    print(f"  GOOD_MAX_JUNCTION_PX: {GOOD_MAX_JUNCTION_PX}")
    print(f"  USABLE_MAX_JUNCTION_PX: {USABLE_MAX_JUNCTION_PX}")
    print(f"  HARD_EDGE_DENSITY_FILTER: {HARD_EDGE_DENSITY_FILTER}")
    print(f"  MAX_EDGES_BY_N: {MAX_EDGES_BY_N}")
    print(f"  AUTO_TRAIN_IF_TEACHER_EXISTS: {AUTO_TRAIN_IF_TEACHER_EXISTS}")
    print(f"  TRAIN_EPOCHS: {TRAIN_EPOCHS}")
    data = load_json(INPUT_FILE, required=True)
    candidates = get_candidate_list(data)
    print("\n[Input]")
    print(f"  candidate_count: {len(candidates)}")
    if not candidates:
        raise RuntimeError("No candidates found in glyph_candidates_with_primitives.json")
    normalized = []
    for i, c in enumerate(candidates):
        cc = copy.deepcopy(c)
        cid = get_candidate_id(cc, i)
        cc["candidate_id"] = cid; cc["generated_glyph_id"] = cc.get("generated_glyph_id", cid); cc["glyph_candidate_id"] = cc.get("glyph_candidate_id", cid)
        normalized.append(cc)
    candidates = normalized
    model, load_info = load_or_init_model(device)
    if AUTO_TRAIN_IF_TEACHER_EXISTS and (FORCE_RETRAIN or not load_info.get("loaded", False)):
        model, train_info = train_model_if_possible(model, candidates, device)
    else:
        train_info = {"trained": False, "reason": "model_already_loaded_or_auto_train_disabled"}
    model.eval()
    print("\n" + "=" * 80)
    print("Start Neural Solving")
    print("=" * 80)
    results, debug_items = [], []
    t0 = time.time()
    for idx, cand in enumerate(candidates):
        solved, dbg = solve_candidate_neural(cand, model, device, idx=idx)
        results.append(solved); debug_items.append(dbg)
        st = solved.get("quality_report", {}).get("quality_status", "unknown")
        if (idx + 1) % PRINT_EVERY == 0 or idx == 0 or idx + 1 == len(candidates):
            elapsed = time.time() - t0
            cps = (idx + 1) / max(elapsed, 1e-6)
            print(f"  progress {idx+1}/{len(candidates)} | {cps:.2f} cand/s | last={get_candidate_id(cand, idx)} N={dbg.get('num_nodes')} E={dbg.get('num_edges')} hard={dbg.get('hard_topology')} beforeMaxJ={dbg.get('before_max_junction_px'):.3f}px warmMaxJ={dbg.get('warm_max_junction_px'):.3f}px finalMaxJ={dbg.get('final_max_junction_px'):.3f}px maxTXA={dbg.get('final_max_tx_angle_deg'):.3f}° tMoved={dbg.get('num_t_moved_edges',0)} maxDt={dbg.get('max_t_delta',0.0):.3f} status={st}")
    elapsed = time.time() - t0
    summary = summarize_results(results, debug_items, elapsed)
    print("\n" + "=" * 80)
    print("Neural Constraint Solver Summary")
    print("=" * 80)
    print(f"  count: {summary['count']}")
    print(f"  elapsed_sec: {summary['elapsed_sec']}")
    print(f"  cand_per_sec: {summary['cand_per_sec']}")
    print(f"  status_hist: {summary['status_hist']}")
    print(f"  good_or_usable_count: {summary['good_or_usable_count']}")
    print(f"  hard_topology_count: {summary['hard_topology_count']}")
    print(f"  before_maxJ_stats: {summary['before_maxJ_stats']}")
    print(f"  warm_maxJ_stats:   {summary['warm_maxJ_stats']}")
    print(f"  final_maxJ_stats:  {summary['final_maxJ_stats']}")
    print(f"  final_maxTXA_stats:{summary['final_maxTXA_stats']}")
    output = {"schema_version": "solved_glyph_candidates_neural_v2", "solver_type": "neural_projection_v2", "input_file": INPUT_FILE, "output_file": OUTPUT_FILE, "model_file": OUTPUT_MODEL_FILE, "config": {"MAX_NODES": MAX_NODES, "USE_PROJECTION": USE_PROJECTION, "USE_GNN_WARMSTART": USE_GNN_WARMSTART, "OPTIMIZE_EDGE_T": OPTIMIZE_EDGE_T, "MAX_EDGE_T_DELTA": MAX_EDGE_T_DELTA, "EDGE_T_REG_WEIGHT": EDGE_T_REG_WEIGHT, "PROJECTION_ITERS": PROJECTION_ITERS, "PROJECTION_LR": PROJECTION_LR, "PROJECTION_PRIOR_WEIGHT": PROJECTION_PRIOR_WEIGHT, "PROJECTION_WARMSTART_WEIGHT": PROJECTION_WARMSTART_WEIGHT, "GOOD_MAX_JUNCTION_PX": GOOD_MAX_JUNCTION_PX, "USABLE_MAX_JUNCTION_PX": USABLE_MAX_JUNCTION_PX, "HARD_EDGE_DENSITY_FILTER": HARD_EDGE_DENSITY_FILTER, "MAX_EDGES_BY_N": MAX_EDGES_BY_N}, "train_info": train_info, "summary": summary, "solved_glyph_candidates": results}
    save_json(output, OUTPUT_FILE)
    report = {"schema_version": "neural_constraint_solver_report_v1", "summary": summary, "train_info": train_info, "debug_items": debug_items}
    save_json(report, OUTPUT_REPORT_FILE)
    print("\n" + "=" * 80)
    print("Saved")
    print("=" * 80)
    print(f"  solved: {OUTPUT_FILE}")
    print(f"  report: {OUTPUT_REPORT_FILE}")
    print("\nNext:")
    print("  1. Run score_solved_glyphs.py")
    print("  2. Check gnn_scored_solved_previews/top_combined")
    print("  3. Compare cand_per_sec and status_hist against old constraint_solver.py")


if __name__ == "__main__":
    main()
