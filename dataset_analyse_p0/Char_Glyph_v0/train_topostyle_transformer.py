# -*- coding: utf-8 -*-
"""
train_topostyle_transformer.py

Topology-first / Style-second stroke generator.

核心思想：
    不再训练一个 solver 去修坏掉的 Bézier 几何；
    而是从人工标注数据中读取干净拓扑，把拓扑 junction 显式变成 anchors，
    然后把每条 stroke split 成 anchor-to-anchor 的 segment。

    对每条 segment：
        endpoint 由 topology anchors 决定；
        模型只预测局部 style 参数：
            alpha1, beta1, alpha2, beta2, width_norm

    Bézier 构造：
        A, B = 两端拓扑 anchor
        d = normalize(B - A)
        n = perpendicular(d)

        P0 = A
        P1 = A + alpha1 * |AB| * d + beta1 * |AB| * n
        P2 = B - alpha2 * |AB| * d + beta2 * |AB| * n
        P3 = B

    因此 junction consistency 是 by construction，不需要 constraint_solver.py，
    不需要 L-BFGS / Adam / anchor projection 后处理。

默认读取：
    ../AI_VECTOR_ROUTER_With_topo/annotations_topo/*_topo.json

默认输出：
    topostyle_transformer_best.pt
    topostyle_transformer_final.pt
    topostyle_transformer_train_report.json
    topostyle_transformer_val_predictions.json
    topostyle_dataset_report.json
"""

import os
import sys
import json
import math
import time
import random
import hashlib
from glob import glob
from collections import Counter, defaultdict

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    raise RuntimeError("需要 PyTorch。请先安装 torch。") from e


# =========================================================
# 0. Paths
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ANNOTATION_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, "../AI_VECTOR_ROUTER_With_topo/annotations_topo")
)

OUTPUT_BEST_MODEL_FILE = os.path.join(SCRIPT_DIR, "topostyle_transformer_best.pt")
OUTPUT_FINAL_MODEL_FILE = os.path.join(SCRIPT_DIR, "topostyle_transformer_final.pt")
OUTPUT_MODEL_FILE = os.path.join(SCRIPT_DIR, "topostyle_transformer.pt")
OUTPUT_REPORT_FILE = os.path.join(SCRIPT_DIR, "topostyle_transformer_train_report.json")
OUTPUT_VAL_PRED_FILE = os.path.join(SCRIPT_DIR, "topostyle_transformer_val_predictions.json")
OUTPUT_DATASET_REPORT_FILE = os.path.join(SCRIPT_DIR, "topostyle_dataset_report.json")


# =========================================================
# 1. Config
# =========================================================
RANDOM_SEED = 42
DEVICE_MODE = "auto"  # auto / cuda / mps / cpu

CANVAS_SIZE = 400.0

MAX_STROKES = 8
MAX_SEGMENTS = 32
MAX_ANCHORS = 32
MAX_EDGES = 32
MAX_SHAPE_CODE = 160
MAX_WIDTH_TOKEN = 8

TRAIN_RATIO = 0.80
EPOCHS = 500
BATCH_SIZE = 64
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 2.0

PRINT_EVERY_EPOCH = 10
SAVE_EVERY_EPOCH = 50

# Model
SEG_FEAT_DIM = 20
D_MODEL = 128
NUM_LAYERS = 4
NUM_HEADS = 4
DROPOUT = 0.10

# Style parameter ranges
ALPHA_MAX = 1.25
BETA_MAX = 1.25
WIDTH_MAX_NORM = 0.20

# Loss weights
LAMBDA_STYLE = 1.0
LAMBDA_CONTROL = 12.0
LAMBDA_CURVE = 32.0
LAMBDA_WIDTH = 0.20
LAMBDA_SMOOTH = 0.04
LAMBDA_CANVAS = 0.10

NUM_CURVE_SAMPLES = 24

# Topology canonicalization
CANONICALIZE_TOPOLOGY = True
INFER_EDGES_IF_MISSING = True
ORACLE_EDGE_MAXJ_PX = 8.0
ORACLE_GLYPH_MAXJ_PX = 8.0
MIN_EDGES_REQUIRED = 1

# Split / anchor rules
T_KEY_ROUND = 4
ENDPOINT_EPS = 1e-3
MIN_SEGMENT_T_GAP = 0.03
MIN_SEGMENT_LEN_NORM = 0.008

# Augment whole glyph anchors and GT consistently.
# 注意：这不是 corruption solver 训练；只是常规等变增强。
AUGMENT_PER_GLYPH = 24
GLOBAL_TRANSLATE_STD = 0.020
GLOBAL_ROTATE_STD_DEG = 5.0
GLOBAL_SCALE_STD = 0.040
ANCHOR_JITTER_STD = 0.000  # 默认 0：拓扑 anchor 精确来自人工标注


# =========================================================
# 2. Basic utils
# =========================================================
def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def stable_hash_str(obj):
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


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
    return np.asarray(arr, dtype=np.float32) * float(CANVAS_SIZE)


def node_id_of(node, fallback):
    if not isinstance(node, dict):
        return fallback
    for k in ["bezier_id", "stroke_id", "node_id", "id", "old_index"]:
        if k in node:
            return safe_int(node[k], fallback)
    return fallback


# =========================================================
# 3. Bézier utils
# =========================================================
def bezier_np(P, t):
    P = np.asarray(P, dtype=np.float32).reshape(4, 2)
    t = float(clamp(t, 0.0, 1.0))
    u = 1.0 - t
    return (
        (u ** 3) * P[0]
        + 3.0 * (u ** 2) * t * P[1]
        + 3.0 * u * (t ** 2) * P[2]
        + (t ** 3) * P[3]
    ).astype(np.float32)


def bezier_deriv_np(P, t):
    P = np.asarray(P, dtype=np.float32).reshape(4, 2)
    t = float(clamp(t, 0.0, 1.0))
    u = 1.0 - t
    return (
        3.0 * (u ** 2) * (P[1] - P[0])
        + 6.0 * u * t * (P[2] - P[1])
        + 3.0 * (t ** 2) * (P[3] - P[2])
    ).astype(np.float32)


def sample_bezier_np(P, n=NUM_CURVE_SAMPLES):
    ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.stack([bezier_np(P, t) for t in ts], axis=0).astype(np.float32)


def split_bezier_np(P, t):
    P = np.asarray(P, dtype=np.float32).reshape(4, 2)
    t = float(clamp(t, 0.0, 1.0))

    P01 = (1 - t) * P[0] + t * P[1]
    P12 = (1 - t) * P[1] + t * P[2]
    P23 = (1 - t) * P[2] + t * P[3]
    P012 = (1 - t) * P01 + t * P12
    P123 = (1 - t) * P12 + t * P23
    P0123 = (1 - t) * P012 + t * P123

    left = np.stack([P[0], P01, P012, P0123], axis=0).astype(np.float32)
    right = np.stack([P0123, P123, P23, P[3]], axis=0).astype(np.float32)
    return left, right


def segment_bezier_np(P, t0, t1):
    t0 = float(clamp(t0, 0.0, 1.0))
    t1 = float(clamp(t1, 0.0, 1.0))
    if t1 < t0:
        t0, t1 = t1, t0
    if abs(t1 - t0) < 1e-8:
        q = bezier_np(P, t0)
        return np.stack([q, q, q, q], axis=0).astype(np.float32)

    _, right = split_bezier_np(P, t0)
    local_t = (t1 - t0) / max(1e-8, 1.0 - t0)
    seg, _ = split_bezier_np(right, local_t)
    return seg.astype(np.float32)


def bezier_torch(P, t):
    # P: [..., 4, 2], t: [S]
    t = t.to(P.device).float()
    while t.ndim < P.ndim - 1:
        t = t.view(*([1] * (P.ndim - 2)), -1)
    # easier use explicit sampling outside
    raise NotImplementedError


def sample_bezier_torch(P, n=NUM_CURVE_SAMPLES):
    # P: [B,M,4,2]
    ts = torch.linspace(0.0, 1.0, n, device=P.device, dtype=P.dtype)
    u = 1.0 - ts
    w0 = (u ** 3).view(1, 1, n, 1)
    w1 = (3 * (u ** 2) * ts).view(1, 1, n, 1)
    w2 = (3 * u * (ts ** 2)).view(1, 1, n, 1)
    w3 = (ts ** 3).view(1, 1, n, 1)
    return (
        w0 * P[:, :, 0:1, :]
        + w1 * P[:, :, 1:2, :]
        + w2 * P[:, :, 2:3, :]
        + w3 * P[:, :, 3:4, :]
    )


def curve_center_angle_length(P):
    P = np.asarray(P, dtype=np.float32).reshape(4, 2)
    p0 = P[0]
    p3 = P[3]
    d = p3 - p0
    length = float(np.linalg.norm(d))
    if length < 1e-8:
        angle = 0.0
    else:
        angle = math.atan2(float(d[1]), float(d[0]))
    center = 0.5 * (p0 + p3)
    return center.astype(np.float32), float(angle), float(length)


def transform_points_np(points, trans, rot_rad, scale, center=None):
    pts = np.asarray(points, dtype=np.float32)
    if center is None:
        center = pts.reshape(-1, 2).mean(axis=0, keepdims=True)
    else:
        center = np.asarray(center, dtype=np.float32).reshape(1, 2)
    trans = np.asarray(trans, dtype=np.float32).reshape(1, 2)
    co, si = math.cos(rot_rad), math.sin(rot_rad)
    R = np.asarray([[co, -si], [si, co]], dtype=np.float32)
    shp = pts.shape
    X = pts.reshape(-1, 2) - center
    Y = (X * scale) @ R.T + center + trans
    return Y.reshape(shp).astype(np.float32)


# =========================================================
# 4. Topology parsing / canonicalization
# =========================================================
def closest_point_on_bezier(P, q, n=80):
    ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
    pts = np.stack([bezier_np(P, t) for t in ts], axis=0)
    d = np.linalg.norm(pts - np.asarray(q, dtype=np.float32).reshape(1, 2), axis=1)
    k = int(np.argmin(d))
    return float(ts[k]), float(d[k])


def closest_pair_on_beziers(Pu, Pv, n=64):
    ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
    Pu_s = np.stack([bezier_np(Pu, t) for t in ts], axis=0)
    Pv_s = np.stack([bezier_np(Pv, t) for t in ts], axis=0)
    dist = np.linalg.norm(Pu_s[:, None, :] - Pv_s[None, :, :], axis=-1)
    idx = int(np.argmin(dist))
    i, j = np.unravel_index(idx, dist.shape)
    return float(ts[i]), float(ts[j]), float(dist[i, j])


def parse_jtype(x):
    if x is None:
        return None
    if isinstance(x, str):
        s = x.upper()
        if "E2E" in s or "ENDPOINT_TO_ENDPOINT" in s or "END_TO_END" in s:
            return "E2E"
        if "T_ATTACH" in s or "T-JUNCTION" in s or "TJUNCTION" in s or s == "T":
            return "T"
        if s == "X" or "CROSS" in s or "INTERSECT" in s:
            return "X"
        return None
    idx = safe_int(x, -1)
    if idx == 1:
        return "E2E"
    if idx == 2:
        return "T"
    if idx == 3:
        return "X"
    return None


def find_first_key(d, keys):
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d:
            return d[k]
    return None


def extract_strokes_from_bundle(bundle):
    strokes = bundle.get("strokes", [])
    out = []
    for i, s in enumerate(strokes):
        if not isinstance(s, dict):
            continue

        P = None
        for k in ["mother_bezier", "bezier", "control_points", "points"]:
            if k in s:
                arr = normalize_points(s[k])
                if arr is not None and arr.shape[0] >= 4:
                    P = arr[:4]
                    break
        if P is None:
            continue

        sid = node_id_of(s, i)
        width_token = safe_int(s.get("width_token", 0), 0)
        width_norm = safe_float(s.get("width_norm", s.get("width", 0.035)), 0.035)
        if width_norm > 1.0:
            width_norm = width_norm / CANVAS_SIZE
        width_norm = clamp(width_norm, 0.002, WIDTH_MAX_NORM)

        shape_code = safe_int(s.get("shape_code", s.get("shape_token", -1)), -1)
        if shape_code < 0:
            samples = sample_bezier_np(P, n=12)
            chord = np.linalg.norm(samples[-1] - samples[0])
            arc = np.sum(np.linalg.norm(samples[1:] - samples[:-1], axis=1))
            straight = chord / max(arc, 1e-8)
            if straight > 0.96:
                shape_code = 20
            elif straight > 0.85:
                shape_code = 14
            else:
                shape_code = 16

        center, angle, length = curve_center_angle_length(P)

        out.append({
            "old_index": i,
            "stroke_id": sid,
            "P": P.astype(np.float32),
            "shape_code": int(np.clip(shape_code, 0, MAX_SHAPE_CODE - 1)),
            "width_token": int(np.clip(width_token, 0, MAX_WIDTH_TOKEN - 1)),
            "width_norm": float(width_norm),
            "center": center.astype(np.float32),
            "angle": float(angle),
            "length": float(length),
            "stroke_type": s.get("stroke_type", "open"),
        })
    return out


def parse_topology_edges(bundle, strokes):
    events = []
    for k in ["topology_events", "topology", "edges", "relations"]:
        v = bundle.get(k, None)
        if isinstance(v, list):
            events.extend(v)
        elif isinstance(v, dict):
            for kk in ["events", "edges", "positive_edges_undirected"]:
                if isinstance(v.get(kk), list):
                    events.extend(v[kk])

    if len(events) == 0:
        return []

    id_to_idx = {}
    for idx, s in enumerate(strokes):
        id_to_idx[s["stroke_id"]] = idx
        id_to_idx[s["old_index"]] = idx

    edges = []
    seen = set()
    for ev in events:
        if not isinstance(ev, dict):
            continue

        jt = None
        for k in ["j_type", "type", "event_type", "relation_type", "topology_type", "action"]:
            jt = parse_jtype(ev.get(k))
            if jt:
                break
        if jt is None:
            continue

        u_raw = find_first_key(ev, [
            "u", "src", "source", "source_id", "stroke_a", "a", "node_u",
            "guest_id", "guest", "guest_stroke", "bezier_id_a", "stroke_id_a"
        ])
        v_raw = find_first_key(ev, [
            "v", "dst", "target", "target_id", "stroke_b", "b", "node_v",
            "host_id", "host", "host_stroke", "bezier_id_b", "stroke_id_b"
        ])

        if isinstance(u_raw, dict):
            u_raw = find_first_key(u_raw, ["bezier_id", "stroke_id", "node_id", "id"])
        if isinstance(v_raw, dict):
            v_raw = find_first_key(v_raw, ["bezier_id", "stroke_id", "node_id", "id"])

        if u_raw is None or v_raw is None:
            continue

        u_id = safe_int(u_raw, -999999)
        v_id = safe_int(v_raw, -999999)
        if u_id not in id_to_idx or v_id not in id_to_idx:
            continue

        u = id_to_idx[u_id]
        v = id_to_idx[v_id]
        if u == v:
            continue

        tu = find_first_key(ev, ["t_u", "t_a", "source_t", "guest_t", "t_guest", "u_t", "t1"])
        tv = find_first_key(ev, ["t_v", "t_b", "target_t", "host_t", "t_host", "v_t", "t2"])
        if tu is None:
            tu = 1.0 if jt in ["E2E", "T"] else 0.5
        if tv is None:
            tv = 0.0 if jt == "E2E" else 0.5

        tu = clamp(safe_float(tu, 0.0), 0.0, 1.0)
        tv = clamp(safe_float(tv, 0.0), 0.0, 1.0)

        key = (min(u, v), max(u, v), jt, round(tu, 3), round(tv, 3))
        if key in seen:
            continue
        seen.add(key)

        edges.append({
            "u": u,
            "v": v,
            "j_type": jt,
            "t_u": tu,
            "t_v": tv,
            "source": "json",
        })

    return edges


def segment_intersection(a, b, c, d):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    d = np.asarray(d, dtype=np.float32)

    r = b - a
    s = d - c
    den = r[0] * s[1] - r[1] * s[0]
    if abs(float(den)) < 1e-8:
        return None
    q = c - a
    t = (q[0] * s[1] - q[1] * s[0]) / den
    u = (q[0] * r[1] - q[1] * r[0]) / den
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return float(t), float(u), a + t * r
    return None


def infer_edges_from_geometry(strokes):
    edges = []
    N = len(strokes)

    # endpoint-to-endpoint
    for i in range(N):
        Pi = strokes[i]["P"]
        for j in range(i + 1, N):
            Pj = strokes[j]["P"]
            candidates = []
            for ti in [0.0, 1.0]:
                ai = bezier_np(Pi, ti)
                for tj in [0.0, 1.0]:
                    bj = bezier_np(Pj, tj)
                    d = float(np.linalg.norm(ai - bj) * CANVAS_SIZE)
                    candidates.append((d, ti, tj))
            d, ti, tj = min(candidates, key=lambda x: x[0])
            if d <= ORACLE_EDGE_MAXJ_PX:
                edges.append({
                    "u": i, "v": j, "j_type": "E2E",
                    "t_u": ti, "t_v": tj, "source": "inferred_e2e",
                })

    # simple X intersections
    for i in range(N):
        Pi_s = sample_bezier_np(strokes[i]["P"], n=40)
        for j in range(i + 1, N):
            Pj_s = sample_bezier_np(strokes[j]["P"], n=40)
            best = None
            for a in range(len(Pi_s) - 1):
                for b in range(len(Pj_s) - 1):
                    inter = segment_intersection(Pi_s[a], Pi_s[a + 1], Pj_s[b], Pj_s[b + 1])
                    if inter is None:
                        continue
                    ti_l, tj_l, p = inter
                    ti = (a + ti_l) / (len(Pi_s) - 1)
                    tj = (b + tj_l) / (len(Pj_s) - 1)
                    if ti < 0.05 or ti > 0.95 or tj < 0.05 or tj > 0.95:
                        continue
                    best = (ti, tj)
                    break
                if best:
                    break
            if best:
                edges.append({
                    "u": i, "v": j, "j_type": "X",
                    "t_u": best[0], "t_v": best[1], "source": "inferred_x",
                })

    return edges[:MAX_EDGES]


def canonicalize_edge(e, strokes):
    u, v = int(e["u"]), int(e["v"])
    if u < 0 or v < 0 or u >= len(strokes) or v >= len(strokes) or u == v:
        return None

    Pu = strokes[u]["P"]
    Pv = strokes[v]["P"]
    jt = e.get("j_type", "E2E")

    if jt == "E2E":
        candidates = []
        for tu in [0.0, 1.0]:
            qu = bezier_np(Pu, tu)
            for tv in [0.0, 1.0]:
                qv = bezier_np(Pv, tv)
                d = float(np.linalg.norm(qu - qv) * CANVAS_SIZE)
                candidates.append((d, tu, tv))
        d, tu, tv = min(candidates, key=lambda x: x[0])
        ce = dict(e)
        ce.update({"t_u": tu, "t_v": tv, "oracle_junction_px": d})
        return ce

    if jt == "T":
        candidates = []
        for tu in [0.0, 1.0]:
            qu = bezier_np(Pu, tu)
            tv, d = closest_point_on_bezier(Pv, qu)
            candidates.append((d * CANVAS_SIZE, tu, tv, "u_endpoint_to_v_curve"))
        for tv in [0.0, 1.0]:
            qv = bezier_np(Pv, tv)
            tu, d = closest_point_on_bezier(Pu, qv)
            candidates.append((d * CANVAS_SIZE, tu, tv, "v_endpoint_to_u_curve"))

        d, tu, tv, mode = min(candidates, key=lambda x: x[0])
        ce = dict(e)
        ce.update({"t_u": tu, "t_v": tv, "oracle_junction_px": d, "canonical_mode": mode})
        return ce

    if jt == "X":
        tu, tv, d = closest_pair_on_beziers(Pu, Pv)
        ce = dict(e)
        ce.update({"t_u": tu, "t_v": tv, "oracle_junction_px": d * CANVAS_SIZE})
        return ce

    return None


def canonicalize_edges(edges, strokes):
    out = []
    dropped = []
    seen = set()

    for e in edges:
        ce = canonicalize_edge(e, strokes) if CANONICALIZE_TOPOLOGY else dict(e)
        if ce is None:
            dropped.append({"reason": "canonicalize_failed", "edge": e})
            continue

        err = safe_float(ce.get("oracle_junction_px", 0.0), 0.0)
        if err > ORACLE_EDGE_MAXJ_PX:
            dropped.append({"reason": "oracle_edge_too_large", "oracle_junction_px": err, "edge": ce})
            continue

        key = (ce["u"], ce["v"], ce["j_type"], round(ce["t_u"], 3), round(ce["t_v"], 3))
        if key in seen:
            continue
        seen.add(key)
        out.append(ce)
        if len(out) >= MAX_EDGES:
            break

    return out, dropped


# =========================================================
# 5. Topology-first segmentization
# =========================================================
class UnionFind:
    def __init__(self):
        self.parent = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x):
        self.add(x)
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def quant_t(t):
    t = float(clamp(t, 0.0, 1.0))
    if t <= ENDPOINT_EPS:
        return 0.0
    if t >= 1.0 - ENDPOINT_EPS:
        return 1.0
    return round(t, T_KEY_ROUND)


def style_from_bezier(P_seg, A, B, width_norm):
    P = np.asarray(P_seg, dtype=np.float32).reshape(4, 2)
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)

    dvec = B - A
    L = float(np.linalg.norm(dvec))
    if L < 1e-8:
        d = np.asarray([1.0, 0.0], dtype=np.float32)
        L = 1e-8
    else:
        d = (dvec / L).astype(np.float32)

    n = np.asarray([-d[1], d[0]], dtype=np.float32)

    # 这里 P_seg 的 P0/P3 来自原曲线，A/B 来自共享 anchor。
    # 人工标注中它们几乎一致；如果有微小差异，以 A/B 作为 topology endpoint。
    v1 = P[1] - A
    v2 = P[2] - B

    alpha1 = float(np.dot(v1, d) / L)
    beta1 = float(np.dot(v1, n) / L)

    # P2 = B - alpha2 * L*d + beta2 * L*n
    alpha2 = float(-np.dot(v2, d) / L)
    beta2 = float(np.dot(v2, n) / L)

    return np.asarray([
        clamp(alpha1, 0.0, ALPHA_MAX),
        clamp(beta1, -BETA_MAX, BETA_MAX),
        clamp(alpha2, 0.0, ALPHA_MAX),
        clamp(beta2, -BETA_MAX, BETA_MAX),
        clamp(float(width_norm), 0.0, WIDTH_MAX_NORM),
    ], dtype=np.float32)


def style_to_bezier_np(A, B, style):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    style = np.asarray(style, dtype=np.float32).reshape(-1)

    alpha1, beta1, alpha2, beta2 = [float(x) for x in style[:4]]

    dvec = B - A
    L = float(np.linalg.norm(dvec))
    if L < 1e-8:
        d = np.asarray([1.0, 0.0], dtype=np.float32)
        L = 1e-8
    else:
        d = (dvec / L).astype(np.float32)

    n = np.asarray([-d[1], d[0]], dtype=np.float32)

    P0 = A
    P1 = A + alpha1 * L * d + beta1 * L * n
    P2 = B - alpha2 * L * d + beta2 * L * n
    P3 = B

    return np.stack([P0, P1, P2, P3], axis=0).astype(np.float32)


def style_to_bezier_torch(A, B, style):
    # A/B: [B,M,2], style: [B,M,5]
    dvec = B - A
    L = torch.linalg.norm(dvec, dim=-1, keepdim=True).clamp_min(1e-8)
    d = dvec / L
    n = torch.stack([-d[..., 1], d[..., 0]], dim=-1)

    alpha1 = style[..., 0:1]
    beta1 = style[..., 1:2]
    alpha2 = style[..., 2:3]
    beta2 = style[..., 3:4]

    P0 = A
    P1 = A + alpha1 * L * d + beta1 * L * n
    P2 = B - alpha2 * L * d + beta2 * L * n
    P3 = B
    return torch.stack([P0, P1, P2, P3], dim=-2)


def build_topostyle_glyph(glyph):
    strokes = glyph["strokes"]
    edges = glyph["edges"]
    N = len(strokes)

    uf = UnionFind()
    attach_keys = set()

    # endpoints always exist as topology anchors
    for s in range(N):
        for t in [0.0, 1.0]:
            key = (s, quant_t(t))
            uf.add(key)
            attach_keys.add(key)

    # topology edges merge their two incident attachment keys
    topo_edges_debug = []
    for e in edges:
        u = int(e["u"])
        v = int(e["v"])
        tu = quant_t(e.get("t_u", 0.0))
        tv = quant_t(e.get("t_v", 0.0))
        ku = (u, tu)
        kv = (v, tv)
        uf.add(ku)
        uf.add(kv)
        uf.union(ku, kv)
        attach_keys.add(ku)
        attach_keys.add(kv)
        topo_edges_debug.append({
            "u": u, "v": v, "j_type": e.get("j_type", "E2E"),
            "t_u": tu, "t_v": tv,
        })

    # group keys into anchors
    root_to_keys = defaultdict(list)
    for key in sorted(attach_keys):
        root_to_keys[uf.find(key)].append(key)

    anchors = []
    key_to_anchor = {}
    for aid, (root, keys) in enumerate(root_to_keys.items()):
        pts = []
        for s, t in keys:
            pts.append(bezier_np(strokes[s]["P"], t))
        pos = np.mean(np.stack(pts, axis=0), axis=0).astype(np.float32)
        anchors.append({
            "anchor_id": aid,
            "pos": pos,
            "keys": keys,
        })
        for k in keys:
            key_to_anchor[k] = aid

    if len(anchors) > MAX_ANCHORS:
        return None, "too_many_anchors"

    # split strokes at all attached t positions
    segments = []
    skipped_short = 0

    for s_idx, s in enumerate(strokes):
        P = s["P"]

        ts = sorted(set([0.0, 1.0] + [t for (ss, t) in attach_keys if ss == s_idx]))
        # remove too-close duplicates, preserving 0 and 1
        clean_ts = []
        for t in ts:
            if not clean_ts or abs(t - clean_ts[-1]) >= MIN_SEGMENT_T_GAP or t in [0.0, 1.0]:
                clean_ts.append(t)
        ts = clean_ts

        for j in range(len(ts) - 1):
            t0 = float(ts[j])
            t1 = float(ts[j + 1])
            if t1 - t0 < MIN_SEGMENT_T_GAP:
                continue

            k0 = (s_idx, quant_t(t0))
            k1 = (s_idx, quant_t(t1))
            if k0 not in key_to_anchor or k1 not in key_to_anchor:
                continue

            a0 = key_to_anchor[k0]
            a1 = key_to_anchor[k1]
            A = anchors[a0]["pos"]
            B = anchors[a1]["pos"]

            if float(np.linalg.norm(B - A)) < MIN_SEGMENT_LEN_NORM:
                skipped_short += 1
                continue

            P_seg = segment_bezier_np(P, t0, t1)
            style = style_from_bezier(P_seg, A, B, s["width_norm"])
            P_recon = style_to_bezier_np(A, B, style)

            center, angle, length = curve_center_angle_length(P_recon)
            samples = sample_bezier_np(P_recon, n=16)
            chord = np.linalg.norm(samples[-1] - samples[0])
            arc = np.sum(np.linalg.norm(samples[1:] - samples[:-1], axis=1))
            straight = float(chord / max(arc, 1e-8))

            segments.append({
                "segment_id": len(segments),
                "parent_stroke": s_idx,
                "shape_code": s["shape_code"],
                "width_token": s["width_token"],
                "width_norm": s["width_norm"],
                "anchor_start": a0,
                "anchor_end": a1,
                "t0": t0,
                "t1": t1,
                "A": A.astype(np.float32),
                "B": B.astype(np.float32),
                "P_gt": P_recon.astype(np.float32),  # topology endpoint aligned GT segment
                "P_orig_seg": P_seg.astype(np.float32),
                "style": style.astype(np.float32),
                "center": center.astype(np.float32),
                "angle": float(angle),
                "length": float(length),
                "straight": straight,
            })

    if len(segments) <= 0:
        return None, "no_segments"
    if len(segments) > MAX_SEGMENTS:
        return None, f"too_many_segments:{len(segments)}>{MAX_SEGMENTS}"

    # anchor degrees by generated segment graph
    anchor_degree = Counter()
    for seg in segments:
        anchor_degree[seg["anchor_start"]] += 1
        anchor_degree[seg["anchor_end"]] += 1

    for seg in segments:
        seg["degree_start"] = anchor_degree[seg["anchor_start"]]
        seg["degree_end"] = anchor_degree[seg["anchor_end"]]

    return {
        "source_file": glyph["source_file"],
        "hex_key": glyph["hex_key"],
        "char": glyph.get("char", ""),
        "strokes": strokes,
        "topology_edges": edges,
        "anchors": anchors,
        "segments": segments,
        "skipped_short_segments": skipped_short,
        "fingerprint": stable_hash_str({
            "source": glyph["source_file"],
            "hex": glyph["hex_key"],
            "n_strokes": len(strokes),
            "n_edges": len(edges),
            "n_anchors": len(anchors),
            "n_segments": len(segments),
        }),
    }, "ok"


def load_annotation_glyphs():
    files = sorted(glob(os.path.join(ANNOTATION_DIR, "*_topo.json")))
    raw_glyphs = []
    glyphs = []
    stats = Counter()
    oracle_edges = []
    dropped_examples = []

    for path in files:
        try:
            data = load_json(path)
        except Exception as e:
            print(f"[WARN] read failed: {path} | {e}")
            continue
        if not isinstance(data, dict):
            continue

        for hex_key, bundle in data.items():
            if not isinstance(bundle, dict):
                continue

            strokes = extract_strokes_from_bundle(bundle)
            if len(strokes) == 0:
                stats["no_strokes"] += 1
                continue
            if len(strokes) > MAX_STROKES:
                stats["too_many_strokes"] += 1
                continue

            raw_edges = parse_topology_edges(bundle, strokes)
            if len(raw_edges) == 0 and INFER_EDGES_IF_MISSING:
                raw_edges = infer_edges_from_geometry(strokes)
                if len(raw_edges) > 0:
                    stats["edges_inferred_glyphs"] += 1

            if len(raw_edges) < MIN_EDGES_REQUIRED:
                stats["no_edges"] += 1
                continue

            edges, dropped = canonicalize_edges(raw_edges, strokes)
            stats["raw_edges"] += len(raw_edges)
            stats["kept_edges"] += len(edges)
            stats["dropped_edges"] += len(dropped)
            if dropped and len(dropped_examples) < 20:
                dropped_examples.extend(dropped[: max(0, 20 - len(dropped_examples))])

            if len(edges) < MIN_EDGES_REQUIRED:
                stats["no_edges_after_canonicalization"] += 1
                continue

            max_oracle = max(safe_float(e.get("oracle_junction_px", 0.0), 0.0) for e in edges)
            mean_oracle = float(np.mean([safe_float(e.get("oracle_junction_px", 0.0), 0.0) for e in edges]))
            oracle_edges.extend([safe_float(e.get("oracle_junction_px", 0.0), 0.0) for e in edges])

            if max_oracle > ORACLE_GLYPH_MAXJ_PX:
                stats["glyph_oracle_too_large"] += 1
                continue

            glyph_info = bundle.get("glyph_info", {})
            raw = {
                "source_file": os.path.basename(path),
                "hex_key": str(hex_key),
                "char": glyph_info.get("char", ""),
                "strokes": strokes,
                "edges": edges,
                "oracle_max_junction_px": max_oracle,
                "oracle_mean_junction_px": mean_oracle,
            }
            raw_glyphs.append(raw)

            tg, reason = build_topostyle_glyph(raw)
            if tg is None:
                stats[f"topostyle_skip_{reason}"] += 1
                continue

            glyphs.append(tg)

    if len(oracle_edges) > 0:
        oracle_stats = {
            "mean": round(float(np.mean(oracle_edges)), 6),
            "p50": round(float(np.percentile(oracle_edges, 50)), 6),
            "p90": round(float(np.percentile(oracle_edges, 90)), 6),
            "max": round(float(np.max(oracle_edges)), 6),
        }
    else:
        oracle_stats = {}

    node_hist = Counter(len(g["strokes"]) for g in glyphs)
    segment_hist = Counter(len(g["segments"]) for g in glyphs)
    anchor_hist = Counter(len(g["anchors"]) for g in glyphs)

    style_vals = []
    for g in glyphs:
        for seg in g["segments"]:
            style_vals.append(seg["style"])
    style_vals = np.asarray(style_vals, dtype=np.float32) if style_vals else np.zeros((0, 5), dtype=np.float32)

    style_stats = {}
    if len(style_vals) > 0:
        names = ["alpha1", "beta1", "alpha2", "beta2", "width_norm"]
        for i, name in enumerate(names):
            style_stats[name] = {
                "mean": round(float(np.mean(style_vals[:, i])), 6),
                "p50": round(float(np.percentile(style_vals[:, i], 50)), 6),
                "p90": round(float(np.percentile(style_vals[:, i], 90)), 6),
                "min": round(float(np.min(style_vals[:, i])), 6),
                "max": round(float(np.max(style_vals[:, i])), 6),
            }

    report = {
        "schema_version": "topostyle_dataset_report",
        "annotation_dir": ANNOTATION_DIR,
        "files": len(files),
        "usable_glyphs": len(glyphs),
        "stats": dict(stats),
        "node_hist": dict(node_hist),
        "segment_hist": dict(segment_hist),
        "anchor_hist": dict(anchor_hist),
        "oracle_edge_junction_px_stats": oracle_stats,
        "style_stats": style_stats,
        "dropped_edge_examples": dropped_examples,
    }
    save_json(report, OUTPUT_DATASET_REPORT_FILE)

    print("\n[Annotation Loading + TopoStyle Segmentization]")
    print(f"  annotation_dir: {ANNOTATION_DIR}")
    print(f"  files: {len(files)}")
    print(f"  usable_glyphs: {len(glyphs)}")
    print(f"  skipped/stats: {dict(stats)}")
    print(f"  node_hist: {dict(node_hist)}")
    print(f"  segment_hist: {dict(segment_hist)}")
    print(f"  anchor_hist: {dict(anchor_hist)}")
    print(f"  oracle_edge_junction_px_stats: {oracle_stats}")
    print(f"  dataset_report: {OUTPUT_DATASET_REPORT_FILE}")

    if len(glyphs) == 0:
        raise RuntimeError("没有读到可训练 glyph。请检查 annotations_topo 数据。")

    return glyphs


# =========================================================
# 6. Dataset
# =========================================================
def augment_glyph_arrays(anchors, segments, split_name, fingerprint):
    anchors = np.asarray(anchors, dtype=np.float32).copy()
    P_gt = np.stack([seg["P_gt"] for seg in segments], axis=0).astype(np.float32)

    if split_name == "train":
        trans = np.random.normal(0.0, GLOBAL_TRANSLATE_STD, size=(2,)).astype(np.float32)
        rot = math.radians(np.random.normal(0.0, GLOBAL_ROTATE_STD_DEG))
        scale = float(np.exp(np.random.normal(0.0, GLOBAL_SCALE_STD)))
    else:
        rng_state = np.random.get_state()
        fixed_seed = int(hashlib.md5(str(fingerprint).encode("utf-8")).hexdigest()[:8], 16)
        np.random.seed(fixed_seed % (2 ** 32 - 1))
        trans = np.random.normal(0.0, GLOBAL_TRANSLATE_STD, size=(2,)).astype(np.float32)
        rot = math.radians(np.random.normal(0.0, GLOBAL_ROTATE_STD_DEG))
        scale = float(np.exp(np.random.normal(0.0, GLOBAL_SCALE_STD)))
        np.random.set_state(rng_state)

    center = anchors.mean(axis=0, keepdims=True) if len(anchors) > 0 else np.asarray([[0.5, 0.5]], dtype=np.float32)

    anchors_aug = transform_points_np(anchors, trans, rot, scale, center=center)
    P_aug = transform_points_np(P_gt, trans, rot, scale, center=center)

    if ANCHOR_JITTER_STD > 0 and split_name == "train":
        anchors_aug = anchors_aug + np.random.normal(0.0, ANCHOR_JITTER_STD, size=anchors_aug.shape).astype(np.float32)

    return anchors_aug.astype(np.float32), P_aug.astype(np.float32)


def segment_feature(seg, A, B, anchor_degree_start, anchor_degree_end, idx, M):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    d = B - A
    L = float(np.linalg.norm(d))
    if L < 1e-8:
        angle = 0.0
        d_unit = np.asarray([1.0, 0.0], dtype=np.float32)
    else:
        angle = math.atan2(float(d[1]), float(d[0]))
        d_unit = d / L

    center = 0.5 * (A + B)
    t0 = float(seg["t0"])
    t1 = float(seg["t1"])

    return np.asarray([
        A[0], A[1],
        B[0], B[1],
        center[0], center[1],
        math.sin(angle), math.cos(angle),
        L,
        d_unit[0], d_unit[1],
        t0, t1, t1 - t0,
        float(seg["parent_stroke"]) / max(1.0, MAX_STROKES - 1.0),
        float(idx) / max(1.0, M - 1.0),
        float(anchor_degree_start) / 8.0,
        float(anchor_degree_end) / 8.0,
        float(seg["straight"]),
        float(seg["width_norm"]) / WIDTH_MAX_NORM,
    ], dtype=np.float32)


class TopoStyleGlyphDataset(torch.utils.data.Dataset):
    def __init__(self, glyphs, augment_per_glyph=1, split_name="train"):
        self.glyphs = glyphs
        self.augment_per_glyph = max(1, int(augment_per_glyph))
        self.split_name = split_name

    def __len__(self):
        return len(self.glyphs) * self.augment_per_glyph

    def __getitem__(self, idx):
        glyph = self.glyphs[idx // self.augment_per_glyph]
        segments = glyph["segments"]
        anchors0 = np.stack([a["pos"] for a in glyph["anchors"]], axis=0).astype(np.float32)
        anchors_aug, P_gt_aug = augment_glyph_arrays(anchors0, segments, self.split_name, glyph["fingerprint"])

        M = len(segments)

        seg_feat = np.zeros((MAX_SEGMENTS, SEG_FEAT_DIM), dtype=np.float32)
        shape_ids = np.zeros((MAX_SEGMENTS,), dtype=np.int64)
        width_ids = np.zeros((MAX_SEGMENTS,), dtype=np.int64)
        seg_mask = np.zeros((MAX_SEGMENTS,), dtype=np.float32)
        A_arr = np.zeros((MAX_SEGMENTS, 2), dtype=np.float32)
        B_arr = np.zeros((MAX_SEGMENTS, 2), dtype=np.float32)
        P_gt = np.zeros((MAX_SEGMENTS, 4, 2), dtype=np.float32)
        style_gt = np.zeros((MAX_SEGMENTS, 5), dtype=np.float32)
        conn = np.zeros((MAX_SEGMENTS, MAX_SEGMENTS), dtype=np.float32)

        # augmented style target recomputed from augmented anchors/curves
        anchor_degree = Counter()
        for seg in segments:
            anchor_degree[seg["anchor_start"]] += 1
            anchor_degree[seg["anchor_end"]] += 1

        for i, seg in enumerate(segments):
            a0 = int(seg["anchor_start"])
            a1 = int(seg["anchor_end"])
            A = anchors_aug[a0]
            B = anchors_aug[a1]
            P = P_gt_aug[i]

            # force topology endpoints by construction in target
            P = P.copy()
            P[0] = A
            P[3] = B

            style = style_from_bezier(P, A, B, seg["width_norm"])
            P_topo = style_to_bezier_np(A, B, style)

            seg_feat[i] = segment_feature(seg, A, B, anchor_degree[a0], anchor_degree[a1], i, M)
            shape_ids[i] = int(np.clip(seg["shape_code"], 0, MAX_SHAPE_CODE - 1))
            width_ids[i] = int(np.clip(seg["width_token"], 0, MAX_WIDTH_TOKEN - 1))
            seg_mask[i] = 1.0
            A_arr[i] = A
            B_arr[i] = B
            P_gt[i] = P_topo
            style_gt[i] = style

        # segment adjacency if share anchor
        for i, si in enumerate(segments):
            for j, sj in enumerate(segments):
                if i == j:
                    continue
                if (
                    si["anchor_start"] == sj["anchor_start"]
                    or si["anchor_start"] == sj["anchor_end"]
                    or si["anchor_end"] == sj["anchor_start"]
                    or si["anchor_end"] == sj["anchor_end"]
                ):
                    conn[i, j] = 1.0

        return {
            "seg_feat": torch.from_numpy(seg_feat),
            "shape_ids": torch.from_numpy(shape_ids),
            "width_ids": torch.from_numpy(width_ids),
            "seg_mask": torch.from_numpy(seg_mask),
            "A": torch.from_numpy(A_arr),
            "B": torch.from_numpy(B_arr),
            "P_gt": torch.from_numpy(P_gt),
            "style_gt": torch.from_numpy(style_gt),
            "conn": torch.from_numpy(conn),
            "meta": {
                "source_file": glyph["source_file"],
                "hex_key": glyph["hex_key"],
                "char": glyph.get("char", ""),
                "num_segments": M,
                "num_anchors": len(glyph["anchors"]),
                "fingerprint": glyph["fingerprint"],
            },
        }


def collate_fn(items):
    out = {}
    keys = ["seg_feat", "shape_ids", "width_ids", "seg_mask", "A", "B", "P_gt", "style_gt", "conn"]
    for k in keys:
        out[k] = torch.stack([it[k] for it in items], dim=0)
    out["meta"] = [it["meta"] for it in items]
    return out


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# =========================================================
# 7. Model
# =========================================================
class TopoStyleTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.shape_emb = nn.Embedding(MAX_SHAPE_CODE, 24)
        self.width_emb = nn.Embedding(MAX_WIDTH_TOKEN, 8)

        self.in_proj = nn.Sequential(
            nn.Linear(SEG_FEAT_DIM + 24 + 8, D_MODEL),
            nn.GELU(),
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, D_MODEL),
        )

        self.neighbor_proj = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, D_MODEL),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            dim_feedforward=D_MODEL * 4,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)

        self.out = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, 5),
        )

    def forward(self, batch):
        feat = batch["seg_feat"].float()
        shape_ids = batch["shape_ids"].long().clamp(0, MAX_SHAPE_CODE - 1)
        width_ids = batch["width_ids"].long().clamp(0, MAX_WIDTH_TOKEN - 1)
        mask = batch["seg_mask"].float()
        conn = batch["conn"].float()

        x = torch.cat([feat, self.shape_emb(shape_ids), self.width_emb(width_ids)], dim=-1)
        h = self.in_proj(x)

        # one topology message-passing step before transformer
        conn_masked = conn * mask[:, None, :] * mask[:, :, None]
        deg = conn_masked.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neigh = torch.bmm(conn_masked, h) / deg
        h = h + self.neighbor_proj(neigh)

        key_padding_mask = mask < 0.5
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)

        raw = self.out(h)

        alpha1 = torch.sigmoid(raw[..., 0:1]) * ALPHA_MAX
        beta1 = torch.tanh(raw[..., 1:2]) * BETA_MAX
        alpha2 = torch.sigmoid(raw[..., 2:3]) * ALPHA_MAX
        beta2 = torch.tanh(raw[..., 3:4]) * BETA_MAX
        width = torch.sigmoid(raw[..., 4:5]) * WIDTH_MAX_NORM

        style = torch.cat([alpha1, beta1, alpha2, beta2, width], dim=-1)
        P_pred = style_to_bezier_torch(batch["A"].float(), batch["B"].float(), style)

        return {
            "style": style,
            "P_pred": P_pred,
        }


# =========================================================
# 8. Loss / metrics
# =========================================================
def compute_losses(model_out, batch):
    mask = batch["seg_mask"].float()
    mask_exp = mask[..., None]
    mask_p = mask[:, :, None, None]

    style_pred = model_out["style"]
    P_pred = model_out["P_pred"]
    style_gt = batch["style_gt"].float()
    P_gt = batch["P_gt"].float()

    denom = mask.sum().clamp_min(1.0)

    style_loss = (F.smooth_l1_loss(style_pred, style_gt, reduction="none").sum(dim=-1) * mask).sum() / denom

    ctrl_loss = (((P_pred - P_gt) ** 2).sum(dim=(-1, -2)) * mask).sum() / denom

    C_pred = sample_bezier_torch(P_pred, n=NUM_CURVE_SAMPLES)
    C_gt = sample_bezier_torch(P_gt, n=NUM_CURVE_SAMPLES)
    curve_loss = (((C_pred - C_gt) ** 2).sum(dim=(-1, -2)) * mask).sum() / (denom * NUM_CURVE_SAMPLES)

    width_loss = (((style_pred[..., 4] - style_gt[..., 4]) ** 2) * mask).sum() / denom

    # smooth style: avoid absurd local oscillation among connected segments
    conn = batch["conn"].float() * mask[:, None, :] * mask[:, :, None]
    if conn.sum() > 0:
        ds = (style_pred[:, :, None, :4] - style_pred[:, None, :, :4]).pow(2).sum(dim=-1)
        smooth_loss = (ds * conn).sum() / conn.sum().clamp_min(1.0)
    else:
        smooth_loss = torch.zeros((), device=P_pred.device)

    # canvas soft penalty
    low = F.relu(-P_pred)
    high = F.relu(P_pred - 1.0)
    canvas_loss = (((low + high) ** 2).sum(dim=(-1, -2)) * mask).sum() / denom

    total = (
        LAMBDA_STYLE * style_loss
        + LAMBDA_CONTROL * ctrl_loss
        + LAMBDA_CURVE * curve_loss
        + LAMBDA_WIDTH * width_loss
        + LAMBDA_SMOOTH * smooth_loss
        + LAMBDA_CANVAS * canvas_loss
    )

    # metrics in pixels
    curve_rmse_px = torch.sqrt(curve_loss.detach().clamp_min(1e-12)) * CANVAS_SIZE
    ctrl_rmse_px = torch.sqrt(ctrl_loss.detach().clamp_min(1e-12)) * CANVAS_SIZE

    per_seg_curve_mse = ((C_pred - C_gt) ** 2).sum(dim=(-1, -2)) / float(NUM_CURVE_SAMPLES)
    per_seg_curve_rmse_px = torch.sqrt(per_seg_curve_mse.clamp_min(1e-12)) * CANVAS_SIZE

    return {
        "total": total,
        "style": style_loss.detach(),
        "control": ctrl_loss.detach(),
        "curve": curve_loss.detach(),
        "width": width_loss.detach(),
        "smooth": smooth_loss.detach(),
        "canvas": canvas_loss.detach(),
        "curve_rmse_px": curve_rmse_px.detach(),
        "control_rmse_px": ctrl_rmse_px.detach(),
        "per_seg_curve_rmse_px": per_seg_curve_rmse_px.detach(),
    }


def evaluate(model, loader, device):
    model.eval()
    totals = defaultdict(float)
    n_batches = 0
    all_seg_rmse = []
    all_glyph_max_rmse = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            out = model(batch)
            losses = compute_losses(out, batch)

            for k, v in losses.items():
                if k in ["per_seg_curve_rmse_px"]:
                    continue
                if torch.is_tensor(v) and v.ndim == 0:
                    totals[k] += float(v.detach().cpu())
            n_batches += 1

            seg_rmse = losses["per_seg_curve_rmse_px"].detach().cpu().numpy()
            mask = batch["seg_mask"].detach().cpu().numpy()
            for i in range(seg_rmse.shape[0]):
                vals = seg_rmse[i][mask[i] > 0.5]
                if len(vals) > 0:
                    all_seg_rmse.extend(vals.tolist())
                    all_glyph_max_rmse.append(float(np.max(vals)))

    out = {k: v / max(1, n_batches) for k, v in totals.items()}

    arr = np.asarray(all_glyph_max_rmse, dtype=np.float32)
    seg_arr = np.asarray(all_seg_rmse, dtype=np.float32)
    if len(arr) > 0:
        out["mean_glyph_max_curve_rmse_px"] = float(np.mean(arr))
        out["p50_glyph_max_curve_rmse_px"] = float(np.percentile(arr, 50))
        out["p90_glyph_max_curve_rmse_px"] = float(np.percentile(arr, 90))
        out["max_glyph_curve_rmse_px"] = float(np.max(arr))
        out["good_rate_curve"] = float(np.mean(arr <= 2.0))
        out["usable_rate_curve"] = float(np.mean((arr > 2.0) & (arr <= 6.0)))
        out["bad_rate_curve"] = float(np.mean(arr > 6.0))
    if len(seg_arr) > 0:
        out["mean_segment_curve_rmse_px"] = float(np.mean(seg_arr))
        out["p90_segment_curve_rmse_px"] = float(np.percentile(seg_arr, 90))

    out["topology_junction_px_by_construction"] = 0.0
    out["count"] = int(len(arr))
    return out


def save_checkpoint(path, model, optimizer, epoch, best_metric, config_extra=None):
    cfg = {
        "MODEL_NAME": "TopoStyleTransformer",
        "CANVAS_SIZE": CANVAS_SIZE,
        "MAX_STROKES": MAX_STROKES,
        "MAX_SEGMENTS": MAX_SEGMENTS,
        "MAX_ANCHORS": MAX_ANCHORS,
        "MAX_SHAPE_CODE": MAX_SHAPE_CODE,
        "MAX_WIDTH_TOKEN": MAX_WIDTH_TOKEN,
        "SEG_FEAT_DIM": SEG_FEAT_DIM,
        "D_MODEL": D_MODEL,
        "NUM_LAYERS": NUM_LAYERS,
        "NUM_HEADS": NUM_HEADS,
        "DROPOUT": DROPOUT,
        "ALPHA_MAX": ALPHA_MAX,
        "BETA_MAX": BETA_MAX,
        "WIDTH_MAX_NORM": WIDTH_MAX_NORM,
        "STYLE_PARAM_ORDER": ["alpha1", "beta1", "alpha2", "beta2", "width_norm"],
        "TOPOLOGY_BY_CONSTRUCTION": True,
    }
    if config_extra:
        cfg.update(config_extra)

    torch.save({
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "best_metric": float(best_metric),
        "config": cfg,
    }, path)


def export_val_predictions(model, loader, device, out_path, max_items=50):
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in loader:
            batch_dev = move_batch_to_device(batch, device)
            pred = model(batch_dev)

            P_pred = pred["P_pred"].detach().cpu().numpy()
            style_pred = pred["style"].detach().cpu().numpy()
            P_gt = batch["P_gt"].numpy()
            style_gt = batch["style_gt"].numpy()
            mask = batch["seg_mask"].numpy()
            A = batch["A"].numpy()
            B = batch["B"].numpy()

            for bi, meta in enumerate(batch["meta"]):
                M = int(meta["num_segments"])
                segs = []
                max_rmse = 0.0
                for si in range(M):
                    Cp = sample_bezier_np(P_pred[bi, si], n=NUM_CURVE_SAMPLES)
                    Cg = sample_bezier_np(P_gt[bi, si], n=NUM_CURVE_SAMPLES)
                    rmse = float(np.sqrt(np.mean(np.sum((Cp - Cg) ** 2, axis=-1))) * CANVAS_SIZE)
                    max_rmse = max(max_rmse, rmse)
                    segs.append({
                        "segment_id": si,
                        "A_px": denorm_points(A[bi, si]).astype(float).tolist(),
                        "B_px": denorm_points(B[bi, si]).astype(float).tolist(),
                        "mother_bezier_pred_px": denorm_points(P_pred[bi, si]).astype(float).tolist(),
                        "mother_bezier_gt_px": denorm_points(P_gt[bi, si]).astype(float).tolist(),
                        "style_pred": style_pred[bi, si].astype(float).tolist(),
                        "style_gt": style_gt[bi, si].astype(float).tolist(),
                        "curve_rmse_px": rmse,
                    })

                rows.append({
                    "source_file": meta["source_file"],
                    "hex_key": meta["hex_key"],
                    "char": meta.get("char", ""),
                    "num_segments": M,
                    "num_anchors": int(meta["num_anchors"]),
                    "max_curve_rmse_px": max_rmse,
                    "segments": segs,
                })

                if len(rows) >= max_items:
                    save_json({
                        "schema_version": "topostyle_transformer_val_predictions",
                        "note": "topology endpoints are fixed by anchors; junction error is 0 by construction",
                        "predictions": rows,
                    }, out_path)
                    return

    save_json({
        "schema_version": "topostyle_transformer_val_predictions",
        "note": "topology endpoints are fixed by anchors; junction error is 0 by construction",
        "predictions": rows,
    }, out_path)


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
    print("TopoStyle Transformer Training")
    print("Topology-first / Style-second stroke generator")
    print("=" * 80)
    print(f"  annotation_dir: {ANNOTATION_DIR}")
    print(f"  output_best:    {OUTPUT_BEST_MODEL_FILE}")
    print(f"  output_final:   {OUTPUT_FINAL_MODEL_FILE}")
    print(f"  output_compat:  {OUTPUT_MODEL_FILE}")
    print(f"  output_report:  {OUTPUT_REPORT_FILE}")
    print(f"  device:         {device}")
    print("=" * 80)

    print("\n[Key Method Config]")
    print("  No constraint_solver.py is used.")
    print("  No L-BFGS / Adam per candidate is used.")
    print("  Topology anchors define segment endpoints by construction.")
    print("  Model predicts only style parameters: alpha1, beta1, alpha2, beta2, width.")
    print(f"  MAX_STROKES: {MAX_STROKES}")
    print(f"  MAX_SEGMENTS: {MAX_SEGMENTS}")
    print(f"  MAX_ANCHORS: {MAX_ANCHORS}")
    print(f"  TRAIN_RATIO: {TRAIN_RATIO}")
    print(f"  AUGMENT_PER_GLYPH: {AUGMENT_PER_GLYPH}")
    print(f"  EPOCHS: {EPOCHS}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  LR: {LR}")
    print(f"  D_MODEL: {D_MODEL}")
    print(f"  NUM_LAYERS: {NUM_LAYERS}")
    print(f"  NUM_HEADS: {NUM_HEADS}")
    print(f"  LAMBDA_STYLE: {LAMBDA_STYLE}")
    print(f"  LAMBDA_CONTROL: {LAMBDA_CONTROL}")
    print(f"  LAMBDA_CURVE: {LAMBDA_CURVE}")

    glyphs = load_annotation_glyphs()

    glyphs = list(glyphs)
    random.Random(RANDOM_SEED).shuffle(glyphs)
    n_train = max(1, int(round(len(glyphs) * TRAIN_RATIO)))
    n_train = min(n_train, len(glyphs) - 1) if len(glyphs) > 1 else len(glyphs)

    train_glyphs = glyphs[:n_train]
    val_glyphs = glyphs[n_train:] if n_train < len(glyphs) else glyphs[:]

    train_ds = TopoStyleGlyphDataset(train_glyphs, augment_per_glyph=AUGMENT_PER_GLYPH, split_name="train")
    val_ds = TopoStyleGlyphDataset(val_glyphs, augment_per_glyph=1, split_name="val")

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )

    print("\n[Dataset]")
    print(f"  train_glyphs: {len(train_glyphs)}")
    print(f"  val_glyphs:   {len(val_glyphs)}")
    print(f"  train_items:  {len(train_ds)}")
    print(f"  val_items:    {len(val_ds)}")
    print(f"  train_steps_per_epoch: {len(train_loader)}")

    model = TopoStyleTransformer().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("\n[Model]")
    print(f"  parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = []
    best_metric = float("inf")
    best_epoch = -1

    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = defaultdict(float)
        n_batches = 0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            out = model(batch)
            losses = compute_losses(out, batch)
            loss = losses["total"]

            loss.backward()
            if GRAD_CLIP and GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            for k, v in losses.items():
                if k in ["per_seg_curve_rmse_px"]:
                    continue
                if torch.is_tensor(v) and v.ndim == 0:
                    running[k] += float(v.detach().cpu())
            n_batches += 1

        train_metrics = {k: v / max(1, n_batches) for k, v in running.items()}
        val_metrics = evaluate(model, val_loader, device)

        metric = val_metrics.get("mean_glyph_max_curve_rmse_px", 1e9)
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            save_checkpoint(
                OUTPUT_BEST_MODEL_FILE,
                model,
                optimizer,
                epoch,
                best_metric,
                config_extra={
                    "n_params": n_params,
                    "best_epoch": best_epoch,
                },
            )
            saved_best = True
        else:
            saved_best = False

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
        }
        history.append(row)

        if epoch == 1 or epoch % PRINT_EVERY_EPOCH == 0 or saved_best:
            elapsed_min = (time.time() - t_start) / 60.0
            print(
                f"  epoch {epoch:04d}/{EPOCHS} | time={elapsed_min:.1f}m | "
                f"train_total={train_metrics.get('total', 0):.6f} | "
                f"train_curveRMSE={train_metrics.get('curve_rmse_px', 0):.3f}px | "
                f"val_glyphMaxCurveRMSE={val_metrics.get('mean_glyph_max_curve_rmse_px', 0):.3f}px | "
                f"val_p50={val_metrics.get('p50_glyph_max_curve_rmse_px', 0):.3f}px | "
                f"val_p90={val_metrics.get('p90_glyph_max_curve_rmse_px', 0):.3f}px | "
                f"val_good={val_metrics.get('good_rate_curve', 0):.3f} | "
                f"val_usable={val_metrics.get('usable_rate_curve', 0):.3f} | "
                f"val_bad={val_metrics.get('bad_rate_curve', 0):.3f}"
            )
            if saved_best:
                print(f"    saved best model: {OUTPUT_BEST_MODEL_FILE} | val_mean_glyphMaxCurveRMSE={best_metric:.4f}px")

        if epoch % SAVE_EVERY_EPOCH == 0:
            save_checkpoint(
                OUTPUT_FINAL_MODEL_FILE,
                model,
                optimizer,
                epoch,
                best_metric,
                config_extra={
                    "n_params": n_params,
                    "best_epoch": best_epoch,
                },
            )

    save_checkpoint(
        OUTPUT_FINAL_MODEL_FILE,
        model,
        optimizer,
        EPOCHS,
        best_metric,
        config_extra={
            "n_params": n_params,
            "best_epoch": best_epoch,
        },
    )

    # compatibility model always points to best.
    if os.path.exists(OUTPUT_BEST_MODEL_FILE):
        import shutil
        shutil.copyfile(OUTPUT_BEST_MODEL_FILE, OUTPUT_MODEL_FILE)

    # load best before exporting validation predictions
    if os.path.exists(OUTPUT_BEST_MODEL_FILE):
        ckpt = torch.load(OUTPUT_BEST_MODEL_FILE, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

    final_val = evaluate(model, val_loader, device)
    export_val_predictions(model, val_loader, device, OUTPUT_VAL_PRED_FILE, max_items=80)

    report = {
        "schema_version": "topostyle_transformer_train_report",
        "method": "Topology-first / Style-second Transformer",
        "claim": "junction consistency is guaranteed by topology-anchor parameterization; the neural net predicts only local stroke style",
        "no_constraint_solver": True,
        "no_lbfgs": True,
        "no_adam_per_candidate": True,
        "topology_by_construction": True,
        "config": {
            "MAX_STROKES": MAX_STROKES,
            "MAX_SEGMENTS": MAX_SEGMENTS,
            "MAX_ANCHORS": MAX_ANCHORS,
            "AUGMENT_PER_GLYPH": AUGMENT_PER_GLYPH,
            "EPOCHS": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "LR": LR,
            "D_MODEL": D_MODEL,
            "NUM_LAYERS": NUM_LAYERS,
            "NUM_HEADS": NUM_HEADS,
            "ALPHA_MAX": ALPHA_MAX,
            "BETA_MAX": BETA_MAX,
            "WIDTH_MAX_NORM": WIDTH_MAX_NORM,
            "LAMBDA_STYLE": LAMBDA_STYLE,
            "LAMBDA_CONTROL": LAMBDA_CONTROL,
            "LAMBDA_CURVE": LAMBDA_CURVE,
        },
        "data": {
            "annotation_dir": ANNOTATION_DIR,
            "train_glyphs": len(train_glyphs),
            "val_glyphs": len(val_glyphs),
            "train_items": len(train_ds),
            "val_items": len(val_ds),
        },
        "model": {
            "parameters": n_params,
            "best_epoch": best_epoch,
            "best_val_mean_glyph_max_curve_rmse_px": best_metric,
            "final_val": final_val,
        },
        "outputs": {
            "best_model": OUTPUT_BEST_MODEL_FILE,
            "final_model": OUTPUT_FINAL_MODEL_FILE,
            "compat_model": OUTPUT_MODEL_FILE,
            "val_predictions": OUTPUT_VAL_PRED_FILE,
            "dataset_report": OUTPUT_DATASET_REPORT_FILE,
        },
        "history": history,
    }

    save_json(report, OUTPUT_REPORT_FILE)

    print("\n" + "=" * 80)
    print("Training Finished")
    print("=" * 80)
    print(f"  best_epoch: {best_epoch}")
    print(f"  best_val_mean_glyph_max_curve_rmse_px: {best_metric:.4f}")
    print(f"  final_val: {final_val}")
    print(f"  saved_best_model: {OUTPUT_BEST_MODEL_FILE}")
    print(f"  saved_final_model: {OUTPUT_FINAL_MODEL_FILE}")
    print(f"  saved_compat_model(best copy): {OUTPUT_MODEL_FILE}")
    print(f"  saved_report: {OUTPUT_REPORT_FILE}")
    print(f"  val_predictions: {OUTPUT_VAL_PRED_FILE}")
    print(f"  dataset_report: {OUTPUT_DATASET_REPORT_FILE}")

    print("\nNext:")
    print("  1. 先看 topostyle_dataset_report.json 的 segment_hist / style_stats")
    print("  2. 再看 topostyle_transformer_val_predictions.json 里的 predicted segments")
    print("  3. 后续 inference 应该从 topology anchors 生成 segments，再用本模型预测 style，不再走 DTG solver")


if __name__ == "__main__":
    main()
