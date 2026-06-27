# -*- coding: utf-8 -*-
"""
train_dtg_transformer_v4.py

DTG-Transformer v4: Differentiable Topology-Geometry Graph Transformer

论文版目标：
    使用端到端 Neural Geometry Solver / Graph Transformer，直接从
    stroke topology graph + primitive/layout prior 解码 cubic Bézier 控制点，
    推理阶段不调用 constraint_solver.py / L-BFGS / Adam 后处理优化器。

相对 v2 的关键修改：
    1. Topology canonicalization：训练前根据人工 GT Bézier 重新校准每条边的 t_u/t_v。
    2. GT oracle filtering：如果某条拓扑边在人工 GT 几何上仍无法闭合，则从 junction loss 中剔除。
    3. 更强 junction supervision：提高 Bézier junction loss 权重，降低 supervised/prior 权重。
    4. 更干净的验证指标：打印 GT oracle maxJ、canonical edge 保留率、val p50/p90/maxJ。
    5. 更温和的数据增强：先学习从轻中度 noisy prior 到 clean human geometry 的映射。

默认读取：
    ../AI_VECTOR_ROUTER_With_topo/annotations_topo/*_topo.json

默认输出：
    dtg_transformer_v4.pt
    dtg_transformer_v4_train_report.json
    dtg_transformer_v4_val_predictions.json
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
# 0. 路径配置
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ANNOTATION_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, "../AI_VECTOR_ROUTER_With_topo/annotations_topo")
)

OUTPUT_BEST_MODEL_FILE = os.path.join(SCRIPT_DIR, "dtg_transformer_v4_best.pt")
OUTPUT_FINAL_MODEL_FILE = os.path.join(SCRIPT_DIR, "dtg_transformer_v4_final.pt")
# 兼容旧 inference：训练结束后会把 best 额外复制到 dtg_transformer_v4.pt。
OUTPUT_MODEL_FILE = os.path.join(SCRIPT_DIR, "dtg_transformer_v4.pt")
OUTPUT_REPORT_FILE = os.path.join(SCRIPT_DIR, "dtg_transformer_v4_train_report.json")
OUTPUT_VAL_PRED_FILE = os.path.join(SCRIPT_DIR, "dtg_transformer_v4_val_predictions.json")
OUTPUT_CANONICAL_DATA_FILE = os.path.join(SCRIPT_DIR, "dtg_transformer_v4_canonical_dataset_report.json")


# =========================================================
# 1. 训练配置
# =========================================================
RANDOM_SEED = 42
DEVICE_MODE = "auto"  # auto / cuda / mps / cpu

CANVAS_SIZE = 400.0

MAX_NODES = 8
MAX_EDGES = 24
MAX_SHAPE_CODE = 160
MAX_WIDTH_TOKEN = 8

MIN_EDGES_REQUIRED = 1
SKIP_GLYPHS_WITH_TOO_MANY_NODES = True

# v2：验证集稍微扩大，避免 13 个 glyph 波动太大
TRAIN_RATIO = 0.80

EPOCHS = 500
BATCH_SIZE = 64
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 2.0

PRINT_EVERY_EPOCH = 10
SAVE_EVERY_EPOCH = 50

# Graph Transformer
D_MODEL = 128
D_EDGE = 48
NUM_LAYERS = 4
NUM_HEADS = 4
DROPOUT = 0.10

# Decoder
USE_RESIDUAL_DECODER = True
MAX_CONTROL_DELTA = 0.45  # normalized canvas，允许网络修正更大几何偏移

# v2：更温和的数据增强。先把拓扑闭合学会，再逐渐增强。
AUGMENT_PER_GLYPH = 16
NOISE_CENTER_STD = 0.018
NOISE_ROTATION_STD_DEG = 6.0
NOISE_SCALE_STD = 0.045
NOISE_CONTROL_STD = 0.012
NOISE_EDGE_T_STD = 0.006
GLOBAL_TRANSLATE_STD = 0.012
GLOBAL_ROTATE_STD_DEG = 3.0
GLOBAL_SCALE_STD = 0.030

# Loss weights：v2 明确强调 topology closure
LAMBDA_SUPERVISED = 2.0
LAMBDA_CURVE = 1.2
LAMBDA_JUNCTION = 180.0
LAMBDA_ANGLE = 1.2
LAMBDA_PRIOR = 0.03
LAMBDA_SMOOTH = 0.05
LAMBDA_CANVAS = 0.20
LAMBDA_OVERLAP = 0.02
LAMBDA_JUNCTION_MAX = 40.0   # v4: 保留弱 max-aware，避免它压过 anchor decoder
# v4: junction-anchor decoder。模型显式预测每条拓扑边的共享 junction anchor，
# 然后在 forward 内做一次可微的 closed-form Bézier anchor projection。
LAMBDA_ANCHOR = 120.0
ANCHOR_PROJECT_STRENGTH = 0.95
ANCHOR_PRIOR_BLEND = 0.25
ANCHOR_MAX_DELTA = 0.35
JUNCTION_SOFTMAX_TAU = 0.012 # normalized，约 4.8px；越小越接近 max


# Junction evaluation thresholds
GOOD_JUNCTION_PX = 1.2
USABLE_JUNCTION_PX = 6.0
MIN_TX_ANGLE_DEG = 35.0

# canonicalization 阈值
CANONICALIZE_TOPOLOGY = True
ORACLE_EDGE_MAXJ_PX = 8.0       # 单条边在 GT 上超过该误差，视为脏边，不用于 junction loss
ORACLE_GLYPH_MAXJ_PX = 20.0     # glyph canonical 后仍太差，可以丢弃
CANONICAL_SEARCH_SAMPLES = 80

# Sampling points per curve for curve loss / anti-collapse
NUM_CURVE_SAMPLES = 16

# 如果人工 topology_events 缺失，是否根据几何自动推断弱拓扑边
INFER_EDGES_IF_MISSING = True
INFER_E2E_THRESHOLD_PX = 5.0
INFER_T_THRESHOLD_PX = 5.0
INFER_X_THRESHOLD_PX = 4.0


# =========================================================
# 2. 基础工具
# =========================================================
JTYPE_TO_ID = {
    "NONE": 0,
    "E2E": 1,
    "T": 2,
    "X": 3,
}
ID_TO_JTYPE = {v: k for k, v in JTYPE_TO_ID.items()}


def set_seed(seed: int):
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


def normalize_points(arr, canvas_size=CANVAS_SIZE):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    arr = arr[:, :2].copy()
    if np.nanmax(np.abs(arr)) > 2.0:
        arr = arr / float(canvas_size)
    arr = np.clip(arr, -0.25, 1.25)
    return arr.astype(np.float32)


def denorm_points(arr, canvas_size=CANVAS_SIZE):
    return np.asarray(arr, dtype=np.float32) * float(canvas_size)


def node_id_of(stroke, fallback):
    for k in ["bezier_id", "stroke_id", "node_id", "id", "source_node_id"]:
        if isinstance(stroke, dict) and k in stroke:
            return safe_int(stroke[k], fallback)
    return fallback


def stable_hash_str(x):
    s = json.dumps(x, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]


# =========================================================
# 3. Bézier 几何 numpy / torch
# =========================================================
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


def sample_bezier_np(P, n=NUM_CURVE_SAMPLES):
    ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.stack([bezier_np(P, t) for t in ts], axis=0)


def bezier_torch(P, t):
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


def bezier_deriv_torch(P, t):
    t = torch.clamp(t, 0.0, 1.0)
    while t.ndim < P.ndim - 1:
        t = t.unsqueeze(-1)
    u = 1.0 - t
    d = (
        3 * (u ** 2) * (P[..., 1, :] - P[..., 0, :])
        + 6 * u * t * (P[..., 2, :] - P[..., 1, :])
        + 3 * (t ** 2) * (P[..., 3, :] - P[..., 2, :])
    )
    return d / (torch.norm(d, dim=-1, keepdim=True) + 1e-8)


def sample_bezier_torch(P, n=NUM_CURVE_SAMPLES):
    device = P.device
    ts = torch.linspace(0.0, 1.0, n, device=device, dtype=P.dtype)
    B, N = P.shape[:2]
    P_exp = P.unsqueeze(2).expand(B, N, n, 4, 2)
    t_exp = ts.view(1, 1, n).expand(B, N, n)
    return bezier_torch(P_exp, t_exp)


def curve_center_angle_length(P):
    p0 = P[0]
    p3 = P[3]
    center = P.mean(axis=0)
    d = p3 - p0
    length = float(np.linalg.norm(d))
    angle = math.atan2(float(d[1]), float(d[0])) if length > 1e-8 else 0.0
    return center, angle, length


def transform_bezier_np(P, center_noise, rot_rad, scale):
    P = np.asarray(P, dtype=np.float32).reshape(4, 2)
    c = P.mean(axis=0, keepdims=True)
    X = P - c
    co = math.cos(rot_rad)
    si = math.sin(rot_rad)
    R = np.asarray([[co, -si], [si, co]], dtype=np.float32)
    Y = X * float(scale)
    Y = Y @ R.T
    Y = Y + c + np.asarray(center_noise, dtype=np.float32).reshape(1, 2)
    return Y.astype(np.float32)


def closest_point_on_bezier(P, point, n=CANONICAL_SEARCH_SAMPLES):
    ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
    pts = np.stack([bezier_np(P, t) for t in ts], axis=0)
    d = np.linalg.norm(pts - point[None, :], axis=1)
    k = int(np.argmin(d))
    return float(d[k]), float(ts[k]), pts[k]


def closest_pair_on_beziers(Pa, Pb, n=CANONICAL_SEARCH_SAMPLES):
    ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
    A = np.stack([bezier_np(Pa, t) for t in ts], axis=0)
    B = np.stack([bezier_np(Pb, t) for t in ts], axis=0)
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    flat = int(np.argmin(D))
    ia, ib = np.unravel_index(flat, D.shape)
    return float(D[ia, ib]), float(ts[ia]), float(ts[ib]), A[ia], B[ib]


# =========================================================
# 4. 从人工 topo JSON 读取 glyph + canonicalize topology
# =========================================================
def parse_jtype(raw):
    if raw is None:
        return None
    s = str(raw).upper()
    if "E2E" in s or "ENDPOINT_TO_ENDPOINT" in s or "END_TO_END" in s:
        return "E2E"
    if "T_ATTACH" in s or "T-JUNCTION" in s or "TJUNCTION" in s or s == "T":
        return "T"
    if s == "X" or "CROSS" in s or "INTERSECT" in s:
        return "X"
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
            "center": center.astype(np.float32),
            "angle": float(angle),
            "length": float(length),
            "stroke_type": s.get("stroke_type", "open"),
        })
    return out


def find_first_key(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


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
        edges.append({"u": u, "v": v, "j_type": jt, "t_u": tu, "t_v": tv, "source": "json"})
    return edges


def segment_intersection(p, p2, q, q2):
    r = p2 - p
    s = q2 - q
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(float(denom)) < 1e-8:
        return None
    qp = q - p
    t = (qp[0] * s[1] - qp[1] * s[0]) / denom
    u = (qp[0] * r[1] - qp[1] * r[0]) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return float(t), float(u)
    return None


def infer_edges_from_geometry(strokes):
    edges = []
    seen = set()
    n = len(strokes)
    P_list = [s["P"] for s in strokes]
    e2e_th = INFER_E2E_THRESHOLD_PX / CANVAS_SIZE
    t_th = INFER_T_THRESHOLD_PX / CANVAS_SIZE

    endpoints = [(0.0, 0), (1.0, 3)]
    for i in range(n):
        for j in range(i + 1, n):
            best = None
            for ti, idx_i in endpoints:
                for tj, idx_j in endpoints:
                    dist = float(np.linalg.norm(P_list[i][idx_i] - P_list[j][idx_j]))
                    if best is None or dist < best[0]:
                        best = (dist, ti, tj)
            if best and best[0] < e2e_th:
                seen.add((i, j, "E2E"))
                edges.append({"u": i, "v": j, "j_type": "E2E", "t_u": best[1], "t_v": best[2], "source": "infer"})

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for ti, idx_i in endpoints:
                p = P_list[i][idx_i]
                dist, tj, _ = closest_point_on_bezier(P_list[j], p, n=50)
                if dist < t_th and 0.05 < tj < 0.95:
                    key = (min(i, j), max(i, j), "T", round(ti, 2), round(tj, 2))
                    if key not in seen:
                        seen.add(key)
                        edges.append({"u": i, "v": j, "j_type": "T", "t_u": ti, "t_v": tj, "source": "infer"})

    for i in range(n):
        si = sample_bezier_np(P_list[i], n=24)
        for j in range(i + 1, n):
            sj = sample_bezier_np(P_list[j], n=24)
            found = None
            for a in range(len(si) - 1):
                for b in range(len(sj) - 1):
                    inter = segment_intersection(si[a], si[a + 1], sj[b], sj[b + 1])
                    if inter is not None:
                        ta = (a + inter[0]) / (len(si) - 1)
                        tb = (b + inter[1]) / (len(sj) - 1)
                        if 0.05 < ta < 0.95 and 0.05 < tb < 0.95:
                            found = (ta, tb)
                            break
                if found:
                    break
            if found:
                key = (i, j, "X")
                if key not in seen:
                    seen.add(key)
                    edges.append({"u": i, "v": j, "j_type": "X", "t_u": found[0], "t_v": found[1], "source": "infer"})
    return edges


def canonicalize_edge(edge, strokes):
    """
    用人工 GT Bézier 几何重新计算 edge 的 t_u/t_v。
    这一步只用于训练数据预处理，不是 inference-time solver。
    """
    u = int(edge["u"])
    v = int(edge["v"])
    jt = edge.get("j_type", "E2E")
    Pa = strokes[u]["P"]
    Pb = strokes[v]["P"]

    endpoints = [(0.0, 0), (1.0, 3)]

    if jt == "E2E":
        best = None
        for tu, iu in endpoints:
            for tv, iv in endpoints:
                pu = Pa[iu]
                pv = Pb[iv]
                d = float(np.linalg.norm(pu - pv))
                if best is None or d < best[0]:
                    best = (d, u, v, tu, tv)
        d, cu, cv, tu, tv = best
        return {
            "u": cu, "v": cv, "j_type": "E2E", "t_u": tu, "t_v": tv,
            "oracle_junction_px": d * CANVAS_SIZE,
            "source": edge.get("source", "json"),
            "canonicalized": True,
        }

    if jt == "T":
        candidates = []
        # u endpoint -> v curve
        for tu, iu in endpoints:
            p = Pa[iu]
            dist, tv, _ = closest_point_on_bezier(Pb, p)
            candidates.append((dist, u, v, tu, tv))
        # v endpoint -> u curve，方向反转成 guest->host
        for tv0, iv in endpoints:
            p = Pb[iv]
            dist, tu0, _ = closest_point_on_bezier(Pa, p)
            candidates.append((dist, v, u, tv0, tu0))
        best = min(candidates, key=lambda x: x[0])
        d, gu, hv, t_guest, t_host = best
        return {
            "u": gu, "v": hv, "j_type": "T", "t_u": t_guest, "t_v": t_host,
            "oracle_junction_px": d * CANVAS_SIZE,
            "source": edge.get("source", "json"),
            "canonicalized": True,
        }

    if jt == "X":
        d, tu, tv, _, _ = closest_pair_on_beziers(Pa, Pb)
        # X 要尽量是内部交叉；如果最优点贴边，也保留但会被 oracle threshold 过滤掉一部分
        return {
            "u": u, "v": v, "j_type": "X", "t_u": tu, "t_v": tv,
            "oracle_junction_px": d * CANVAS_SIZE,
            "source": edge.get("source", "json"),
            "canonicalized": True,
        }

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


def load_annotation_glyphs():
    files = sorted(glob(os.path.join(ANNOTATION_DIR, "*_topo.json")))
    glyphs = []
    stats = Counter()
    oracle_edges = []
    dropped_examples = []

    for path in files:
        try:
            data = load_json(path)
        except Exception as e:
            print(f"[WARN] 读取失败: {path} | {e}")
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
            if SKIP_GLYPHS_WITH_TOO_MANY_NODES and len(strokes) > MAX_NODES:
                stats["too_many_nodes"] += 1
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
            glyphs.append({
                "source_file": os.path.basename(path),
                "hex_key": str(hex_key),
                "char": glyph_info.get("char", ""),
                "strokes": strokes,
                "edges": edges,
                "cycles": bundle.get("cycles", []),
                "oracle_max_junction_px": max_oracle,
                "oracle_mean_junction_px": mean_oracle,
                "fingerprint": stable_hash_str({
                    "file": os.path.basename(path),
                    "hex": str(hex_key),
                    "n": len(strokes),
                    "e": len(edges),
                    "oracle": round(max_oracle, 3),
                }),
            })

    if len(oracle_edges) > 0:
        oracle_stats = {
            "mean": round(float(np.mean(oracle_edges)), 6),
            "p50": round(float(np.percentile(oracle_edges, 50)), 6),
            "p90": round(float(np.percentile(oracle_edges, 90)), 6),
            "max": round(float(np.max(oracle_edges)), 6),
        }
    else:
        oracle_stats = {}

    print("\n[Annotation Loading + Topology Canonicalization]")
    print(f"  annotation_dir: {ANNOTATION_DIR}")
    print(f"  files: {len(files)}")
    print(f"  usable_glyphs: {len(glyphs)}")
    print(f"  skipped/stats: {dict(stats)}")
    print(f"  oracle_edge_junction_px_stats: {oracle_stats}")

    save_json({
        "schema_version": "dtg_transformer_v4_canonical_dataset_report",
        "stats": dict(stats),
        "oracle_edge_junction_px_stats": oracle_stats,
        "dropped_edge_examples": dropped_examples,
        "usable_glyphs": len(glyphs),
    }, OUTPUT_CANONICAL_DATA_FILE)

    if len(glyphs) == 0:
        raise RuntimeError("没有读到可训练 glyph。请检查 annotations_topo 目录、edge/t 质量或放宽 ORACLE_EDGE_MAXJ_PX。")
    return glyphs, oracle_stats, dict(stats)


# =========================================================
# 5. Dataset
# =========================================================
def corrupt_bezier(P):
    P = np.asarray(P, dtype=np.float32).reshape(4, 2)
    center_noise = np.random.normal(0.0, NOISE_CENTER_STD, size=(2,)).astype(np.float32)
    rot = math.radians(np.random.normal(0.0, NOISE_ROTATION_STD_DEG))
    scale = float(np.exp(np.random.normal(0.0, NOISE_SCALE_STD)))
    out = transform_bezier_np(P, center_noise, rot, scale)
    out += np.random.normal(0.0, NOISE_CONTROL_STD, size=out.shape).astype(np.float32)
    out = np.clip(out, -0.15, 1.15)
    return out.astype(np.float32)


def apply_global_corruption(Ps):
    Ps = np.asarray(Ps, dtype=np.float32).copy()
    g_trans = np.random.normal(0.0, GLOBAL_TRANSLATE_STD, size=(2,)).astype(np.float32)
    g_rot = math.radians(np.random.normal(0.0, GLOBAL_ROTATE_STD_DEG))
    g_scale = float(np.exp(np.random.normal(0.0, GLOBAL_SCALE_STD)))
    all_pts = Ps.reshape(-1, 2)
    center = all_pts.mean(axis=0, keepdims=True)
    co = math.cos(g_rot)
    si = math.sin(g_rot)
    R = np.asarray([[co, -si], [si, co]], dtype=np.float32)
    X = Ps.reshape(-1, 2) - center
    Y = (X * g_scale) @ R.T + center + g_trans.reshape(1, 2)
    return np.clip(Y.reshape(Ps.shape), -0.20, 1.20).astype(np.float32)


def node_features_from_prior(P_prior, shape_code, width_token, degree, counts):
    center, angle, length = curve_center_angle_length(P_prior)
    samples = sample_bezier_np(P_prior, n=12)
    chord = np.linalg.norm(samples[-1] - samples[0])
    arc = np.sum(np.linalg.norm(samples[1:] - samples[:-1], axis=1))
    straight = chord / max(arc, 1e-8)
    bbox_min = P_prior.min(axis=0)
    bbox_max = P_prior.max(axis=0)
    bbox_wh = bbox_max - bbox_min
    return np.asarray([
        center[0], center[1],
        math.sin(angle), math.cos(angle),
        length,
        straight,
        bbox_wh[0], bbox_wh[1],
        degree / 8.0,
        counts["E2E"] / 8.0,
        counts["T"] / 8.0,
        counts["X"] / 8.0,
    ], dtype=np.float32)


class DTGGlyphDataset(torch.utils.data.Dataset):
    def __init__(self, glyphs, augment_per_glyph=1, split_name="train"):
        self.glyphs = glyphs
        self.augment_per_glyph = max(1, int(augment_per_glyph))
        self.split_name = split_name

    def __len__(self):
        return len(self.glyphs) * self.augment_per_glyph

    def __getitem__(self, idx):
        glyph = self.glyphs[idx // self.augment_per_glyph]
        strokes = glyph["strokes"]
        edges = glyph["edges"]
        N = len(strokes)
        E = min(len(edges), MAX_EDGES)

        P_gt = np.zeros((MAX_NODES, 4, 2), dtype=np.float32)
        P_prior = np.zeros((MAX_NODES, 4, 2), dtype=np.float32)
        node_feat = np.zeros((MAX_NODES, 12), dtype=np.float32)
        shape_ids = np.zeros((MAX_NODES,), dtype=np.int64)
        width_ids = np.zeros((MAX_NODES,), dtype=np.int64)
        node_mask = np.zeros((MAX_NODES,), dtype=np.float32)

        gt_arr = np.stack([s["P"] for s in strokes], axis=0)
        if self.split_name == "train":
            prior_arr = np.stack([corrupt_bezier(P) for P in gt_arr], axis=0)
            prior_arr = apply_global_corruption(prior_arr)
        else:
            # v4：验证集使用固定随机种子，避免每个 epoch 的 val 指标因随机扰动大幅抖动。
            rng_state = np.random.get_state()
            fixed_seed = int(hashlib.md5(str(glyph.get("fingerprint", idx)).encode("utf-8")).hexdigest()[:8], 16)
            np.random.seed(fixed_seed % (2 ** 32 - 1))
            prior_arr = np.stack([corrupt_bezier(P) for P in gt_arr], axis=0)
            prior_arr = apply_global_corruption(prior_arr)
            np.random.set_state(rng_state)

        degree = Counter()
        type_counts = defaultdict(Counter)
        for e in edges:
            u, v = int(e["u"]), int(e["v"])
            if u < N and v < N:
                degree[u] += 1
                degree[v] += 1
                jt = e["j_type"]
                type_counts[u][jt] += 1
                type_counts[v][jt] += 1

        for i, s in enumerate(strokes):
            P_gt[i] = s["P"]
            P_prior[i] = prior_arr[i]
            shape_ids[i] = int(np.clip(s["shape_code"], 0, MAX_SHAPE_CODE - 1))
            width_ids[i] = int(np.clip(s["width_token"], 0, MAX_WIDTH_TOKEN - 1))
            node_mask[i] = 1.0
            node_feat[i] = node_features_from_prior(P_prior[i], shape_ids[i], width_ids[i], degree[i], type_counts[i])

        edge_type = np.zeros((MAX_NODES, MAX_NODES), dtype=np.int64)
        edge_feat = np.zeros((MAX_NODES, MAX_NODES, 8), dtype=np.float32)
        edge_index = np.zeros((MAX_EDGES, 2), dtype=np.int64)
        edge_jtype = np.zeros((MAX_EDGES,), dtype=np.int64)
        edge_t = np.zeros((MAX_EDGES, 2), dtype=np.float32)
        edge_mask = np.zeros((MAX_EDGES,), dtype=np.float32)
        edge_oracle_px = np.zeros((MAX_EDGES,), dtype=np.float32)

        for k in range(E):
            e = edges[k]
            u, v = int(e["u"]), int(e["v"])
            if u >= N or v >= N or u < 0 or v < 0:
                continue
            jt = e.get("j_type", "E2E")
            jid = JTYPE_TO_ID.get(jt, 0)
            tu = clamp(safe_float(e.get("t_u", 0.0), 0.0), 0.0, 1.0)
            tv = clamp(safe_float(e.get("t_v", 0.0), 0.0), 0.0, 1.0)

            if self.split_name == "train" and NOISE_EDGE_T_STD > 0:
                tu = clamp(tu + np.random.normal(0.0, NOISE_EDGE_T_STD), 0.0, 1.0)
                tv = clamp(tv + np.random.normal(0.0, NOISE_EDGE_T_STD), 0.0, 1.0)

            edge_type[u, v] = jid
            edge_type[v, u] = jid

            cu = P_prior[u].mean(axis=0)
            cv = P_prior[v].mean(axis=0)
            duv = cv - cu
            feat_uv = np.asarray([
                tu, tv, abs(tu - tv), tu * tv,
                duv[0], duv[1],
                1.0 if jt == "E2E" else 0.0,
                1.0 if jt in ["T", "X"] else 0.0,
            ], dtype=np.float32)
            feat_vu = np.asarray([
                tv, tu, abs(tu - tv), tu * tv,
                -duv[0], -duv[1],
                1.0 if jt == "E2E" else 0.0,
                1.0 if jt in ["T", "X"] else 0.0,
            ], dtype=np.float32)
            edge_feat[u, v] = feat_uv
            edge_feat[v, u] = feat_vu

            edge_index[k] = np.asarray([u, v], dtype=np.int64)
            edge_jtype[k] = jid
            edge_t[k] = np.asarray([tu, tv], dtype=np.float32)
            edge_mask[k] = 1.0
            edge_oracle_px[k] = safe_float(e.get("oracle_junction_px", 0.0), 0.0)

        return {
            "P_gt": torch.tensor(P_gt, dtype=torch.float32),
            "P_prior": torch.tensor(P_prior, dtype=torch.float32),
            "node_feat": torch.tensor(node_feat, dtype=torch.float32),
            "shape_ids": torch.tensor(shape_ids, dtype=torch.long),
            "width_ids": torch.tensor(width_ids, dtype=torch.long),
            "node_mask": torch.tensor(node_mask, dtype=torch.float32),
            "edge_type": torch.tensor(edge_type, dtype=torch.long),
            "edge_feat": torch.tensor(edge_feat, dtype=torch.float32),
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "edge_jtype": torch.tensor(edge_jtype, dtype=torch.long),
            "edge_t": torch.tensor(edge_t, dtype=torch.float32),
            "edge_mask": torch.tensor(edge_mask, dtype=torch.float32),
            "edge_oracle_px": torch.tensor(edge_oracle_px, dtype=torch.float32),
            "meta": {
                "hex_key": glyph["hex_key"],
                "char": glyph["char"],
                "source_file": glyph["source_file"],
                "fingerprint": glyph["fingerprint"],
                "N": N,
                "E": E,
                "oracle_max_junction_px": glyph["oracle_max_junction_px"],
                "oracle_mean_junction_px": glyph["oracle_mean_junction_px"],
            },
        }


# =========================================================
# 6. Edge-conditioned Graph Transformer
# =========================================================
class EdgeConditionedSelfAttention(nn.Module):
    def __init__(self, d_model, d_edge, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.edge_bias = nn.Sequential(nn.Linear(d_edge, d_model), nn.GELU(), nn.Linear(d_model, num_heads))
        self.edge_value = nn.Sequential(nn.Linear(d_edge, d_model), nn.GELU(), nn.Linear(d_model, d_model))
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
        return self.out(out) * node_mask.unsqueeze(-1)


class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model, d_edge, num_heads, dropout=0.1):
        super().__init__()
        self.attn = EdgeConditionedSelfAttention(d_model, d_edge, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, edge_emb, node_mask):
        x = self.norm1(x + self.attn(x, edge_emb, node_mask))
        x = x * node_mask.unsqueeze(-1)
        x = self.norm2(x + self.ffn(x))
        return x * node_mask.unsqueeze(-1)


class DTGTransformer(nn.Module):
    """
    v4: Junction-Anchor DTG-Transformer.

    v4 只靠 loss 压 junction，val p50 / p90 仍然过高。
    v4 显式预测每条 topology edge 的共享 junction anchor A_e，
    并在 forward 里用一次 closed-form Bézier anchor projection
    把两条 incident curves 拉向同一个 anchor。

    这不是后处理 optimizer：
        - 无 L-BFGS
        - 无 Adam per-candidate
        - 无 iterative solve
        - 是网络 forward 内的一次可微解析几何层
    """
    def __init__(self):
        super().__init__()
        self.shape_emb = nn.Embedding(MAX_SHAPE_CODE, 24)
        self.width_emb = nn.Embedding(MAX_WIDTH_TOKEN, 8)
        self.edge_type_emb = nn.Embedding(4, 16)

        self.node_in = nn.Sequential(
            nn.Linear(12 + 8 + 24 + 8, D_MODEL),
            nn.GELU(), nn.LayerNorm(D_MODEL), nn.Dropout(DROPOUT), nn.Linear(D_MODEL, D_MODEL),
        )
        self.edge_in = nn.Sequential(
            nn.Linear(8 + 16, D_EDGE), nn.GELU(), nn.LayerNorm(D_EDGE), nn.Linear(D_EDGE, D_EDGE),
        )
        self.layers = nn.ModuleList([
            GraphTransformerLayer(D_MODEL, D_EDGE, NUM_HEADS, DROPOUT)
            for _ in range(NUM_LAYERS)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(D_MODEL, 8)
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
        单次可微 Bézier anchor projection。

        Bézier 点对控制点线性：
            B(t) = Σ_l w_l(t) P_l

        若希望 B(t) 移到 anchor，可加：
            ΔP_l = w_l / Σw_l² * (anchor - B(t))

        多条 topology edge 作用到同一 stroke 时，scatter-add 累加并平均。
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
            P_side = P_raw[batch_idx, idx]      # [B,E,4,2]
            q = bezier_torch(P_side, t)        # [B,E,2]
            d = (anchors - q) * edge_mask.unsqueeze(-1)

            w = self._bezier_weights(t)        # [B,E,4]
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
        x = torch.cat([node_feat, prior_flat, self.shape_emb(shape_ids), self.width_emb(width_ids)], dim=-1)
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
# 7. Loss / Metrics
# =========================================================
def gather_edges(P, edge_index, edge_t):
    B, E = edge_index.shape[:2]
    batch_idx = torch.arange(B, device=P.device).view(B, 1).expand(B, E)
    u = edge_index[..., 0].clamp(0, MAX_NODES - 1)
    v = edge_index[..., 1].clamp(0, MAX_NODES - 1)
    Pi = P[batch_idx, u]
    Pj = P[batch_idx, v]
    tu = edge_t[..., 0]
    tv = edge_t[..., 1]
    return Pi, Pj, tu, tv


def junction_metrics(P_pred, edge_index, edge_t, edge_mask):
    Pi, Pj, tu, tv = gather_edges(P_pred, edge_index, edge_t)
    pi = bezier_torch(Pi, tu)
    pj = bezier_torch(Pj, tv)
    dist = torch.norm(pi - pj, dim=-1) * CANVAS_SIZE
    masked = dist * edge_mask
    denom = edge_mask.sum(dim=1).clamp_min(1.0)
    mean_j = masked.sum(dim=1) / denom
    max_j = masked.masked_fill(edge_mask <= 0, -1.0).max(dim=1).values
    return mean_j, max_j, dist


def compute_losses(model, batch, device):
    P_gt = batch["P_gt"]
    P_prior = batch["P_prior"]
    node_mask = batch["node_mask"]
    edge_index = batch["edge_index"]
    edge_jtype = batch["edge_jtype"]
    edge_t = batch["edge_t"]
    edge_mask = batch["edge_mask"]
    aux = model(batch, return_aux=True)
    P_pred = aux["P_pred"]
    edge_anchors = aux["edge_anchors"]
    node_m = node_mask.view(node_mask.shape[0], node_mask.shape[1], 1, 1)

    supervised = F.smooth_l1_loss(P_pred * node_m, P_gt * node_m, reduction="sum") / node_m.sum().clamp_min(1.0)

    curve_pred = sample_bezier_torch(P_pred, NUM_CURVE_SAMPLES)
    curve_gt = sample_bezier_torch(P_gt, NUM_CURVE_SAMPLES)
    curve_mask = node_mask.view(node_mask.shape[0], node_mask.shape[1], 1, 1)
    curve_loss = F.smooth_l1_loss(curve_pred * curve_mask, curve_gt * curve_mask, reduction="sum") / curve_mask.sum().clamp_min(1.0)

    Pi, Pj, tu, tv = gather_edges(P_pred, edge_index, edge_t)
    pi = bezier_torch(Pi, tu)
    pj = bezier_torch(Pj, tv)
    edge_dist_norm = torch.norm(pi - pj, dim=-1)

    # v4: anchor supervision。canonical GT topology 已保证 B_gt(t_u) 与 B_gt(t_v) 重合，
    # 这里用二者均值作为 oracle anchor，直接监督 edge anchor head。
    Pi_gt, Pj_gt, tu_gt, tv_gt = gather_edges(P_gt, edge_index, edge_t)
    ai_gt = bezier_torch(Pi_gt, tu_gt)
    aj_gt = bezier_torch(Pj_gt, tv_gt)
    anchor_gt = 0.5 * (ai_gt + aj_gt)
    anchor_loss = (((edge_anchors - anchor_gt) ** 2).sum(dim=-1) * edge_mask).sum() / edge_mask.sum().clamp_min(1.0)
    jdist2 = edge_dist_norm ** 2
    junction = (jdist2 * edge_mask).sum() / edge_mask.sum().clamp_min(1.0)

    # v4: max-aware junction loss。普通 mean loss 会让少数断裂很大的边长期存在，
    # 这里用 masked logsumexp 近似每个 glyph 的 max junction error。
    masked_for_lse = edge_dist_norm / JUNCTION_SOFTMAX_TAU
    masked_for_lse = masked_for_lse.masked_fill(edge_mask <= 0, -1e4)
    valid_glyph = (edge_mask.sum(dim=1) > 0).float()
    junction_max = (
        JUNCTION_SOFTMAX_TAU * torch.logsumexp(masked_for_lse, dim=1) * valid_glyph
    ).sum() / valid_glyph.sum().clamp_min(1.0)

    di = bezier_deriv_torch(Pi, tu)
    dj = bezier_deriv_torch(Pj, tv)
    cos_abs = torch.abs((di * dj).sum(dim=-1)).clamp(0.0, 1.0)
    tx_mask = ((edge_jtype == JTYPE_TO_ID["T"]) | (edge_jtype == JTYPE_TO_ID["X"])).float() * edge_mask
    max_allowed_cos = math.cos(math.radians(MIN_TX_ANGLE_DEG))
    angle_loss = (F.relu(cos_abs - max_allowed_cos) ** 2 * tx_mask).sum() / tx_mask.sum().clamp_min(1.0)

    prior_loss = F.smooth_l1_loss(P_pred * node_m, P_prior * node_m, reduction="sum") / node_m.sum().clamp_min(1.0)

    d2a = P_pred[:, :, 0] - 2 * P_pred[:, :, 1] + P_pred[:, :, 2]
    d2b = P_pred[:, :, 1] - 2 * P_pred[:, :, 2] + P_pred[:, :, 3]
    smooth = (((d2a ** 2).sum(dim=-1) + (d2b ** 2).sum(dim=-1)) * node_mask).sum() / node_mask.sum().clamp_min(1.0)

    canvas_pen = (F.relu(-P_pred) ** 2 + F.relu(P_pred - 1.0) ** 2).sum(dim=(-1, -2))
    canvas_pen = (canvas_pen * node_mask).sum() / node_mask.sum().clamp_min(1.0)

    centers = P_pred.mean(dim=2)
    B, N, _ = centers.shape
    dij = torch.norm(centers.unsqueeze(2) - centers.unsqueeze(1), dim=-1)
    pair_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    eye = torch.eye(N, device=device).view(1, N, N)
    pair_mask = pair_mask * (1.0 - eye)
    overlap = (F.relu(0.05 - dij) ** 2 * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)

    total = (
        LAMBDA_SUPERVISED * supervised
        + LAMBDA_CURVE * curve_loss
        + LAMBDA_JUNCTION * junction
        + LAMBDA_JUNCTION_MAX * junction_max
        + LAMBDA_ANCHOR * anchor_loss
        + LAMBDA_ANGLE * angle_loss
        + LAMBDA_PRIOR * prior_loss
        + LAMBDA_SMOOTH * smooth
        + LAMBDA_CANVAS * canvas_pen
        + LAMBDA_OVERLAP * overlap
    )
    mean_j, max_j, _ = junction_metrics(P_pred, edge_index, edge_t, edge_mask)
    return {
        "total": total,
        "supervised": supervised.detach(),
        "curve": curve_loss.detach(),
        "junction": junction.detach(),
        "junction_max": junction_max.detach(),
        "anchor": anchor_loss.detach(),
        "angle": angle_loss.detach(),
        "prior": prior_loss.detach(),
        "smooth": smooth.detach(),
        "canvas": canvas_pen.detach(),
        "overlap": overlap.detach(),
        "mean_j_px": mean_j.detach(),
        "max_j_px": max_j.detach(),
        "P_pred": P_pred,
    }


# =========================================================
# 8. Train / Eval
# =========================================================
def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def collate_fn(items):
    out = {}
    keys = [k for k in items[0].keys() if k != "meta"]
    for k in keys:
        out[k] = torch.stack([it[k] for it in items], dim=0)
    out["meta"] = [it["meta"] for it in items]
    return out


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    sums = defaultdict(float)
    count = 0
    good = usable = bad = 0
    all_max_j = []

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        losses = compute_losses(model, batch, device)
        bs = batch["node_mask"].shape[0]
        count += bs
        for k in ["total", "supervised", "curve", "junction", "junction_max", "anchor", "angle", "prior", "smooth", "canvas", "overlap"]:
            sums[k] += float(losses[k].detach().cpu()) * bs
        max_j = losses["max_j_px"].detach().cpu().numpy()
        all_max_j.extend(max_j.tolist())
        good += int(np.sum(max_j <= GOOD_JUNCTION_PX))
        usable += int(np.sum((max_j > GOOD_JUNCTION_PX) & (max_j <= USABLE_JUNCTION_PX)))
        bad += int(np.sum(max_j > USABLE_JUNCTION_PX))

    if count == 0:
        return {}
    out = {k: v / count for k, v in sums.items()}
    arr = np.asarray(all_max_j, dtype=np.float32)
    out["mean_max_j_px"] = float(np.mean(arr))
    out["p50_max_j_px"] = float(np.percentile(arr, 50))
    out["p90_max_j_px"] = float(np.percentile(arr, 90))
    out["max_j_px"] = float(np.max(arr))
    out["good_rate"] = good / count
    out["usable_rate"] = usable / count
    out["bad_rate"] = bad / count
    out["count"] = count
    return out


def save_checkpoint(model, optimizer, epoch, train_metrics, val_metrics, glyph_stats, output_path):
    ckpt = {
        "schema_version": "dtg_transformer_v4",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "MAX_NODES": MAX_NODES,
            "MAX_EDGES": MAX_EDGES,
            "D_MODEL": D_MODEL,
            "D_EDGE": D_EDGE,
            "NUM_LAYERS": NUM_LAYERS,
            "NUM_HEADS": NUM_HEADS,
            "USE_RESIDUAL_DECODER": USE_RESIDUAL_DECODER,
            "MAX_CONTROL_DELTA": MAX_CONTROL_DELTA,
            "CANONICALIZE_TOPOLOGY": CANONICALIZE_TOPOLOGY,
            "ORACLE_EDGE_MAXJ_PX": ORACLE_EDGE_MAXJ_PX,
            "LAMBDA_SUPERVISED": LAMBDA_SUPERVISED,
            "LAMBDA_CURVE": LAMBDA_CURVE,
            "LAMBDA_JUNCTION": LAMBDA_JUNCTION,
            "LAMBDA_JUNCTION_MAX": LAMBDA_JUNCTION_MAX,
            "LAMBDA_ANCHOR": LAMBDA_ANCHOR,
            "ANCHOR_PROJECT_STRENGTH": ANCHOR_PROJECT_STRENGTH,
            "JUNCTION_SOFTMAX_TAU": JUNCTION_SOFTMAX_TAU,
            "LAMBDA_ANGLE": LAMBDA_ANGLE,
            "LAMBDA_PRIOR": LAMBDA_PRIOR,
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "glyph_stats": glyph_stats,
    }
    torch.save(ckpt, output_path)


@torch.no_grad()
def export_val_predictions(model, val_loader, device, max_items=32):
    model.eval()
    exported = []
    for batch in val_loader:
        batch_dev = move_batch_to_device(batch, device)
        P_pred = model(batch_dev).detach().cpu().numpy()
        P_prior = batch["P_prior"].numpy()
        P_gt = batch["P_gt"].numpy()
        node_mask = batch["node_mask"].numpy()
        edge_index = batch["edge_index"].numpy()
        edge_t = batch["edge_t"].numpy()
        edge_jtype = batch["edge_jtype"].numpy()
        edge_mask = batch["edge_mask"].numpy()

        for b in range(P_pred.shape[0]):
            N = int(node_mask[b].sum())
            E = int(edge_mask[b].sum())
            meta = batch["meta"][b]
            strokes = []
            for i in range(N):
                strokes.append({
                    "stroke_id": i,
                    "prior_bezier": denorm_points(P_prior[b, i]).round(3).tolist(),
                    "pred_bezier": denorm_points(P_pred[b, i]).round(3).tolist(),
                    "gt_bezier": denorm_points(P_gt[b, i]).round(3).tolist(),
                })
            edges = []
            max_j = 0.0
            for k in range(E):
                u, v = edge_index[b, k]
                tu, tv = edge_t[b, k]
                jid = int(edge_jtype[b, k])
                pu = bezier_np(P_pred[b, u], float(tu))
                pv = bezier_np(P_pred[b, v], float(tv))
                d = float(np.linalg.norm(pu - pv) * CANVAS_SIZE)
                max_j = max(max_j, d)
                edges.append({
                    "u": int(u), "v": int(v),
                    "j_type": ID_TO_JTYPE.get(jid, "NONE"),
                    "t_u": round(float(tu), 4),
                    "t_v": round(float(tv), 4),
                    "junction_px": round(d, 4),
                })
            exported.append({"meta": meta, "max_junction_px": round(max_j, 4), "strokes": strokes, "edges": edges})
            if len(exported) >= max_items:
                save_json({"schema_version": "dtg_transformer_v4_val_predictions", "items": exported}, OUTPUT_VAL_PRED_FILE)
                return
    save_json({"schema_version": "dtg_transformer_v4_val_predictions", "items": exported}, OUTPUT_VAL_PRED_FILE)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    set_seed(RANDOM_SEED)
    device = choose_device()

    print("\n" + "=" * 80)
    print("DTG-Transformer v4 Training Lite-Best")
    print("End-to-End Neural Geometry Solver with Topology Canonicalization")
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
    print("  Topology canonicalization is used only as training-data preprocessing.")
    print(f"  MAX_NODES: {MAX_NODES}")
    print(f"  MAX_EDGES: {MAX_EDGES}")
    print(f"  TRAIN_RATIO: {TRAIN_RATIO}")
    print(f"  AUGMENT_PER_GLYPH: {AUGMENT_PER_GLYPH}")
    print(f"  EPOCHS: {EPOCHS}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  LR: {LR}")
    print(f"  D_MODEL: {D_MODEL}  # reduced")
    print(f"  D_EDGE: {D_EDGE}    # reduced")
    print(f"  NUM_LAYERS: {NUM_LAYERS}  # reduced")
    print(f"  NUM_HEADS: {NUM_HEADS}    # reduced")
    print(f"  LAMBDA_JUNCTION: {LAMBDA_JUNCTION}")
    print(f"  LAMBDA_JUNCTION_MAX: {LAMBDA_JUNCTION_MAX}")
    print(f"  LAMBDA_ANCHOR: {LAMBDA_ANCHOR}")
    print(f"  ANCHOR_PROJECT_STRENGTH: {ANCHOR_PROJECT_STRENGTH}")
    print(f"  JUNCTION_SOFTMAX_TAU: {JUNCTION_SOFTMAX_TAU}")
    print(f"  LAMBDA_SUPERVISED: {LAMBDA_SUPERVISED}")
    print(f"  LAMBDA_CURVE: {LAMBDA_CURVE}")
    print(f"  ORACLE_EDGE_MAXJ_PX: {ORACLE_EDGE_MAXJ_PX}")

    glyphs, oracle_stats, canonical_stats = load_annotation_glyphs()
    random.shuffle(glyphs)

    node_hist = Counter(len(g["strokes"]) for g in glyphs)
    edge_hist = Counter(len(g["edges"]) for g in glyphs)
    type_hist = Counter()
    for g in glyphs:
        for e in g["edges"]:
            type_hist[e["j_type"]] += 1

    glyph_stats = {
        "usable_glyphs": len(glyphs),
        "node_hist": dict(node_hist),
        "edge_hist": dict(edge_hist),
        "edge_type_hist": dict(type_hist),
        "oracle_edge_junction_px_stats": oracle_stats,
        "canonical_stats": canonical_stats,
    }

    print("\n[Glyph Stats After Canonicalization]")
    print(f"  node_hist: {dict(node_hist)}")
    print(f"  edge_hist: {dict(edge_hist)}")
    print(f"  edge_type_hist: {dict(type_hist)}")
    print(f"  oracle_edge_junction_px_stats: {oracle_stats}")

    split = max(1, int(len(glyphs) * TRAIN_RATIO))
    train_glyphs = glyphs[:split]
    val_glyphs = glyphs[split:] if split < len(glyphs) else glyphs[-max(1, len(glyphs) // 10):]

    train_ds = DTGGlyphDataset(train_glyphs, augment_per_glyph=AUGMENT_PER_GLYPH, split_name="train")
    val_ds = DTGGlyphDataset(val_glyphs, augment_per_glyph=1, split_name="val")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    print("\n[Dataset]")
    print(f"  train_glyphs: {len(train_glyphs)}")
    print(f"  val_glyphs:   {len(val_glyphs)}")
    print(f"  train_items:  {len(train_ds)}")
    print(f"  val_items:    {len(val_ds)}")
    print(f"  train_steps_per_epoch: {len(train_loader)}")

    model = DTGTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print("\n[Model]")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {param_count:,}")

    history = []
    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        sums = defaultdict(float)
        count = 0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            losses = compute_losses(model, batch, device)
            loss = losses["total"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            bs = batch["node_mask"].shape[0]
            count += bs
            for k in ["total", "supervised", "curve", "junction", "junction_max", "anchor", "angle", "prior", "smooth", "canvas", "overlap"]:
                sums[k] += float(losses[k].detach().cpu()) * bs
            sums["mean_max_j_px"] += float(losses["max_j_px"].detach().mean().cpu()) * bs

        train_metrics = {k: v / max(count, 1) for k, v in sums.items()}

        if epoch == 1 or epoch % PRINT_EVERY_EPOCH == 0 or epoch == EPOCHS:
            val_metrics = evaluate(model, val_loader, device)
            elapsed = time.time() - t0
            print(
                f"  epoch {epoch:04d}/{EPOCHS} | "
                f"time={elapsed/60:.1f}m | "
                f"train_total={train_metrics['total']:.6f} | "
                f"train_junc={train_metrics['junction']:.6f} | "
                f"train_maxJ={train_metrics['mean_max_j_px']:.3f}px | "
                f"val_total={val_metrics.get('total', 0):.6f} | "
                f"val_maxJ={val_metrics.get('mean_max_j_px', 0):.3f}px | "
                f"val_p50={val_metrics.get('p50_max_j_px', 0):.3f}px | "
                f"val_p90={val_metrics.get('p90_max_j_px', 0):.3f}px | "
                f"val_good={val_metrics.get('good_rate', 0):.3f} | "
                f"val_usable={val_metrics.get('usable_rate', 0):.3f} | "
                f"val_bad={val_metrics.get('bad_rate', 0):.3f}"
            )

            rec = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "elapsed_sec": elapsed}
            history.append(rec)
            val_score = val_metrics.get("mean_max_j_px", float("inf"))
            if val_score < best_val:
                best_val = val_score
                save_checkpoint(model, optimizer, epoch, train_metrics, val_metrics, glyph_stats, OUTPUT_BEST_MODEL_FILE)
                print(f"    saved best model: {OUTPUT_BEST_MODEL_FILE} | val_mean_maxJ={best_val:.4f}px")

        if epoch % SAVE_EVERY_EPOCH == 0:
            val_metrics_tmp = evaluate(model, val_loader, device, max_batches=3)
            save_checkpoint(model, optimizer, epoch, train_metrics, val_metrics_tmp, glyph_stats, OUTPUT_FINAL_MODEL_FILE)

    final_val = evaluate(model, val_loader, device)
    save_checkpoint(model, optimizer, EPOCHS, train_metrics, final_val, glyph_stats, OUTPUT_FINAL_MODEL_FILE)
    # 兼容旧 inference 脚本：dtg_transformer_v4.pt 始终指向 best，而不是 final。
    if os.path.exists(OUTPUT_BEST_MODEL_FILE):
        import shutil
        shutil.copyfile(OUTPUT_BEST_MODEL_FILE, OUTPUT_MODEL_FILE)
    export_val_predictions(model, val_loader, device, max_items=32)

    report = {
        "schema_version": "dtg_transformer_v4_train_report",
        "method": "DTG-Transformer v2 / End-to-End Neural Geometry Solver",
        "claim": "replace post-processing constraint solver with topology-conditioned neural geometry decoding",
        "no_post_processing_solver_used": True,
        "topology_canonicalization_used_for_training_data_only": True,
        "config": {
            "ANNOTATION_DIR": ANNOTATION_DIR,
            "MAX_NODES": MAX_NODES,
            "MAX_EDGES": MAX_EDGES,
            "AUGMENT_PER_GLYPH": AUGMENT_PER_GLYPH,
            "EPOCHS": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "LR": LR,
            "D_MODEL": D_MODEL,
            "D_EDGE": D_EDGE,
            "NUM_LAYERS": NUM_LAYERS,
            "NUM_HEADS": NUM_HEADS,
            "USE_RESIDUAL_DECODER": USE_RESIDUAL_DECODER,
            "MAX_CONTROL_DELTA": MAX_CONTROL_DELTA,
            "CANONICALIZE_TOPOLOGY": CANONICALIZE_TOPOLOGY,
            "ORACLE_EDGE_MAXJ_PX": ORACLE_EDGE_MAXJ_PX,
            "ORACLE_GLYPH_MAXJ_PX": ORACLE_GLYPH_MAXJ_PX,
            "LAMBDA_SUPERVISED": LAMBDA_SUPERVISED,
            "LAMBDA_CURVE": LAMBDA_CURVE,
            "LAMBDA_JUNCTION": LAMBDA_JUNCTION,
            "LAMBDA_JUNCTION_MAX": LAMBDA_JUNCTION_MAX,
            "JUNCTION_SOFTMAX_TAU": JUNCTION_SOFTMAX_TAU,
            "LAMBDA_ANGLE": LAMBDA_ANGLE,
            "LAMBDA_PRIOR": LAMBDA_PRIOR,
            "LAMBDA_SMOOTH": LAMBDA_SMOOTH,
            "LAMBDA_CANVAS": LAMBDA_CANVAS,
            "LAMBDA_OVERLAP": LAMBDA_OVERLAP,
        },
        "glyph_stats": glyph_stats,
        "final_val": final_val,
        "best_val_mean_maxJ_px": best_val,
        "history": history,
        "outputs": {
            "best_model": OUTPUT_BEST_MODEL_FILE,
            "final_model": OUTPUT_FINAL_MODEL_FILE,
            "compat_model": OUTPUT_MODEL_FILE,
            "report": OUTPUT_REPORT_FILE,
            "val_predictions": OUTPUT_VAL_PRED_FILE,
            "canonical_dataset_report": OUTPUT_CANONICAL_DATA_FILE,
        },
    }
    save_json(report, OUTPUT_REPORT_FILE)

    print("\n" + "=" * 80)
    print("Training Finished")
    print("=" * 80)
    print(f"  best_val_mean_maxJ_px: {best_val:.4f}")
    print(f"  final_val: {final_val}")
    print(f"  saved_best_model: {OUTPUT_BEST_MODEL_FILE}")
    print(f"  saved_final_model: {OUTPUT_FINAL_MODEL_FILE}")
    print(f"  saved_compat_model(best copy): {OUTPUT_MODEL_FILE}")
    print(f"  saved_report: {OUTPUT_REPORT_FILE}")
    print(f"  val_predictions: {OUTPUT_VAL_PRED_FILE}")
    print(f"  canonical_dataset_report: {OUTPUT_CANONICAL_DATA_FILE}")
    print("\nNext:")
    print("  1. 先看 canonical_dataset_report 里的 dropped_edges / oracle_edge_junction_px_stats")
    print("  2. 如果 usable_glyphs 过少，放宽 ORACLE_EDGE_MAXJ_PX 到 12")
    print("  3. 如果 val_bad 仍高，检查 edge anchor 是否学到；下一步可增加 anchor decoder 层数或扩大数据集")


if __name__ == "__main__":
    main()
