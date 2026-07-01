# -*- coding: utf-8 -*-
r"""
train_flywheel_topohgt_v6_relationcalib_ckptselect_ckptselect.py

建议放置位置：
    dataset_analyse_p0/Char_Glyph_v0/flywheel_model/train_flywheel_topohgt_v6_relationcalib_ckptselect_ckptselect.py

模型保存位置：
    dataset_analyse_p0/Char_Glyph_v0/flywheel_model/model/

V3 核心目标：
    更接近你提出的“点/边拓扑 attention + GNN 递归扩散”的结构。

相比 V2：
    1. 从 stroke-only graph 改为 stroke + endpoint typed graph：
       - stroke node
       - endpoint_start node
       - endpoint_end node
    2. Edge-aware global attention message passing：
       - node 通过 pair edge feature attend 到其它 node
       - 多层递归传播，显式扩散相邻点/边的信息
    3. Relation labels 变成 node-pair labels：
       - E2E: endpoint <-> endpoint
       - T:   endpoint -> stroke
       - X:   stroke <-> stroke
    4. Quality head 增加 raster CNN branch：
       - graph embedding + raster image embedding
       - 更适合判断整体视觉质量
    5. 继续保留：
       - stable_uid 去重
       - cleaned > good > annotations_topo > bad
       - group split
       - early stopping
       - threshold sweep
       - FP/FN/top/bottom error export
       - model/ 文件夹保存

读取数据：
    dataset_analyse_p0/AI_VECTOR_ROUTER_With_topo/annotations_topo
    dataset_analyse_p0/Char_Glyph_v0/annotation_tool/pcg_filebacked_stage2_schema/bad
    dataset_analyse_p0/Char_Glyph_v0/annotation_tool/pcg_filebacked_stage2_schema/good
    dataset_analyse_p0/Char_Glyph_v0/annotation_tool/pcg_filebacked_stage2_schema/cleaned
"""

import os
import json
import glob
import math
import time
import random
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


# =============================================================================
# 0. Config
# =============================================================================

@dataclass
class Cfg:
    # data
    max_strokes: int = 16          # 你的当前数据 100% <= 8，16 更快；如果以后 stroke 更多再调大。
    canvas_norm: float = 400.0
    width_norm: float = 20.0
    dist_norm: float = 32.0
    curve_sample_n: int = 32
    raster_size: int = 48

    # relation candidate-pair training
    # 只在“可能相关”的候选 pair 上训练 relation head，避免全 N×N 的 NONE 淹没 E2E/T/X。
    relation_candidate_only: bool = True
    cand_e2e_endpoint_dist_max: float = 16.0
    cand_t_endpoint_curve_dist_max: float = 16.0
    cand_x_curve_curve_dist_max: float = 8.0
    cand_include_all_positive: bool = True

    # typed binary heads
    # 不再做 NONE/E2E/T/X 四分类，而是三个独立二分类：
    #   E2E head: endpoint-endpoint candidate -> is_E2E
    #   T   head: endpoint-stroke candidate   -> is_T
    #   X   head: stroke-stroke candidate     -> is_X
    #
    # V6 改动：
    #   1. 降低 T/X pos_weight，避免 V5 那种高召回低精度的过度预测。
    #   2. 对 E2E/T/X 分别做 threshold sweep 和校准。
    rel_pos_weight_clip: float = 20.0
    use_manual_rel_pos_weight: bool = True
    e2e_pos_weight_manual: float = 1.0
    t_pos_weight_manual: float = 20.0
    x_pos_weight_manual: float = 8.0
    rel_eval_threshold: float = 0.5
    rel_threshold_min: float = 0.05
    rel_threshold_max: float = 0.95
    rel_threshold_step: float = 0.05
    rel_precision_target: float = 0.80

    # preprocess speed
    # v3 的 graph/raster 构造很重；默认训练前一次性预计算，避免每个 epoch 重复 Python/PIL/Numpy 计算。
    precompute_items: bool = True
    cache_preprocessed_pt: bool = True
    force_rebuild_cache: bool = False

    # split
    val_ratio: float = 0.12
    group_split: bool = True
    seed: int = 42

    # model
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.15

    # training
    batch_size: int = 256
    epochs: int = 35
    lr: float = 1.5e-4
    weight_decay: float = 3e-4
    grad_clip: float = 1.0
    early_stop_patience: int = 8
    early_stop_metric: str = "quality_best_f1"

    # loss
    w_quality: float = 2.0
    w_relation: float = 1.0
    quality_pos_weight_clip: float = 20.0
    rel_weight_clip: float = 8.0

    # threshold sweep
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_step: float = 0.05
    precision_target: float = 0.80

    # debug / save
    num_workers: int = 0
    log_every: int = 5
    save_prefix: str = "flywheel_topohgt_v6_relationcalib_ckptselect"
    export_error_topk: int = 100

    # source labels
    annotations_are_positive: bool = True


REL_MAP = {"NONE": 0, "E2E": 1, "T": 2, "X": 3}
REL_INV = {v: k for k, v in REL_MAP.items()}
IGNORE = -100

NODE_TYPE = {"stroke": 0, "endpoint_start": 1, "endpoint_end": 2}
NODE_TYPE_INV = {v: k for k, v in NODE_TYPE.items()}

SOURCE_PRIORITY = {
    "bad": 10,
    "annotations_topo": 20,
    "good": 30,
    "cleaned": 40,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAR_GLYPH_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATASET_ANALYSE_DIR = os.path.abspath(os.path.join(CHAR_GLYPH_DIR, ".."))

DEFAULT_ANNOTATIONS_TOPO_DIR = os.path.join(DATASET_ANALYSE_DIR, "AI_VECTOR_ROUTER_With_topo", "annotations_topo")
DEFAULT_PCG_POOL_ROOT = os.path.join(CHAR_GLYPH_DIR, "annotation_tool", "pcg_filebacked_stage2_schema")

MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, "model")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

PREPROCESS_CACHE_DIR = os.path.join(MODEL_SAVE_DIR, "preprocess_cache")
os.makedirs(PREPROCESS_CACHE_DIR, exist_ok=True)


# =============================================================================
# 1. Basic helpers
# =============================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def choose_checkpoint_from_index(index_path: str, purpose: str = "quality_balanced") -> Dict[str, Any]:
    """
    后续推理脚本可以 import 这个函数自动选择 checkpoint。

    purpose 可选：
      - quality_balanced:     good/bad 平衡 F1 最优
      - quality_conservative: precision>=target 时 recall 最优，适合高置信 good 自动通过
      - relation_overall:     E2E/T/X macro F1 最优
      - relation_e2e:         E2E F1 最优
      - relation_t:           T F1 最优
      - relation_x:           X F1 最优
      - loss:                 val loss 最低
      - latest:               最新 checkpoint

    返回内容包含：
      path, epoch, score, thresholds, metrics_summary
    """
    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    rec = idx.get("recommendations", {}).get(purpose)
    if rec is None:
        raise KeyError(f"purpose={purpose!r} not found in checkpoint index. available={list(idx.get('recommendations', {}).keys())}")
    return rec


def cache_signature(records, cfg):
    payload = {
        "script": "train_flywheel_topohgt_v6_relationcalib_ckptselect",
        "uids": [r["uid"] for r in records],
        "sources": [r["source"] for r in records],
        "labels": [int(r["quality_label"]) for r in records],
        "cfg": {
            "max_strokes": cfg.max_strokes,
            "canvas_norm": cfg.canvas_norm,
            "width_norm": cfg.width_norm,
            "dist_norm": cfg.dist_norm,
            "curve_sample_n": cfg.curve_sample_n,
            "raster_size": cfg.raster_size,
            "relation_candidate_only": cfg.relation_candidate_only,
            "cand_e2e_endpoint_dist_max": cfg.cand_e2e_endpoint_dist_max,
            "cand_t_endpoint_curve_dist_max": cfg.cand_t_endpoint_curve_dist_max,
            "cand_x_curve_curve_dist_max": cfg.cand_x_curve_curve_dist_max,
            "binary_heads": True,
            "relation_calib_v6": True,
            "rel_pos_weight_clip": cfg.rel_pos_weight_clip,
            "use_manual_rel_pos_weight": cfg.use_manual_rel_pos_weight,
            "e2e_pos_weight_manual": cfg.e2e_pos_weight_manual,
            "t_pos_weight_manual": cfg.t_pos_weight_manual,
            "x_pos_weight_manual": cfg.x_pos_weight_manual,
        },
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def percentile(xs, ps=(0, 50, 90, 95, 99, 100)):
    if not xs:
        return {str(p): None for p in ps}
    arr = np.asarray(xs, dtype=np.float32)
    return {str(p): round(float(np.percentile(arr, p)), 4) for p in ps}


# =============================================================================
# 2. stable uid / metadata
# =============================================================================

def round_obj(x, nd=3):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, int):
        return x
    if isinstance(x, list):
        return [round_obj(v, nd) for v in x]
    if isinstance(x, tuple):
        return [round_obj(v, nd) for v in x]
    if isinstance(x, dict):
        return {str(k): round_obj(v, nd) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    return x


def glyph_info(bundle: Dict[str, Any]) -> Dict[str, Any]:
    gi = bundle.get("glyph_info", {})
    return gi if isinstance(gi, dict) else {}


def candidate_id(bundle: Dict[str, Any]) -> str:
    gi = glyph_info(bundle)
    for k in ["candidate_id", "source_candidate_id", "generated_glyph_id", "glyph_candidate_id", "sample_id", "id"]:
        if gi.get(k):
            return str(gi[k])
    hist = bundle.get("edit_history", [])
    if isinstance(hist, list):
        for ev in hist:
            if isinstance(ev, dict):
                for k in ["candidate_id", "source_candidate_id", "generated_glyph_id"]:
                    if ev.get(k):
                        return str(ev[k])
    return ""


def geometry_sig(bundle: Dict[str, Any]):
    arr = []
    strokes = bundle.get("strokes", [])
    if not isinstance(strokes, list):
        return arr
    for i, s in enumerate(strokes):
        if not isinstance(s, dict):
            continue
        try:
            bid = int(s.get("bezier_id", i + 1))
        except Exception:
            bid = i + 1
        arr.append({
            "id": bid,
            "mother_bezier": round_obj(s.get("mother_bezier"), 3),
            "width_bezier": round_obj(s.get("width_bezier", s.get("width")), 3),
            "stroke_type": s.get("stroke_type", ""),
        })
    arr.sort(key=lambda x: x["id"])
    return arr


def stable_uid(bundle: Dict[str, Any], outer_key: str = "") -> str:
    gi = glyph_info(bundle)
    payload = {
        "schema": "pcg_good_stable_uid_v1",
        "candidate_id": candidate_id(bundle),
        "style_mode": gi.get("style_mode", ""),
        "topology_family": gi.get("topology_family", ""),
        "strokes": geometry_sig(bundle),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "PCGGOOD_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def group_key_for_bundle(bundle: Dict[str, Any], source: str, uid: str) -> str:
    gi = glyph_info(bundle)
    cid = candidate_id(bundle)
    style = str(gi.get("style_mode", ""))
    topo = str(gi.get("topology_family", ""))
    if cid:
        return "cid:" + cid
    if style or topo:
        return f"style_topo:{style}|{topo}|source:{source}"
    return "uid:" + uid


def json_slim_bundle_info(bundle: Dict[str, Any]) -> Dict[str, Any]:
    gi = glyph_info(bundle)
    cm = bundle.get("clean_meta", {}) if isinstance(bundle.get("clean_meta", {}), dict) else {}
    return {
        "glyph_info": {
            "hex_key": gi.get("hex_key", gi.get("unicode_hex", "")),
            "char": gi.get("char", ""),
            "candidate_id": candidate_id(bundle),
            "style_mode": gi.get("style_mode", ""),
            "topology_family": gi.get("topology_family", ""),
        },
        "stroke_count": len(bundle.get("strokes", [])) if isinstance(bundle.get("strokes", []), list) else None,
        "clean_meta": {
            "saved_variant": cm.get("saved_variant"),
            "source_stable_uid": cm.get("source_stable_uid"),
            "stage2_changed": cm.get("stage2_changed"),
            "recursive_changed": cm.get("recursive_changed"),
        },
    }


# =============================================================================
# 3. Geometry
# =============================================================================

def cubic(P: np.ndarray, t: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)
    if t.ndim == 1:
        t = t[:, None]
    mt = 1.0 - t
    return mt**3 * P[0] + 3 * mt**2 * t * P[1] + 3 * mt * t**2 * P[2] + t**3 * P[3]


def deriv(P: np.ndarray, t: float) -> np.ndarray:
    P = np.asarray(P, dtype=np.float32)
    mt = 1.0 - float(t)
    tt = float(t)
    return 3 * mt**2 * (P[1] - P[0]) + 6 * mt * tt * (P[2] - P[1]) + 3 * tt**2 * (P[3] - P[2])


def norm_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def width_ctrl_from_stroke(s: Dict[str, Any]) -> np.ndarray:
    wb = s.get("width_bezier")
    if isinstance(wb, list):
        try:
            a = np.asarray(wb, dtype=np.float32).reshape(-1)
            if len(a) == 4:
                return np.maximum(a, 0.5).astype(np.float32)
            if len(a) > 0:
                m = float(np.mean(a))
                return np.array([m, m, m, m], dtype=np.float32)
        except Exception:
            pass
    for k in ["width", "width_mean", "stroke_width"]:
        if k in s:
            try:
                w = max(float(s[k]), 0.5)
                return np.array([w, w, w, w], dtype=np.float32)
            except Exception:
                pass
    return np.array([8.0, 8.0, 8.0, 8.0], dtype=np.float32)


def valid_bundle(bundle: Dict[str, Any], max_strokes: int) -> Tuple[bool, str]:
    if not isinstance(bundle, dict):
        return False, "not_dict"
    strokes = bundle.get("strokes", [])
    if not isinstance(strokes, list) or len(strokes) == 0:
        return False, "missing_strokes"
    if len(strokes) > max_strokes:
        return False, f"too_many_strokes:{len(strokes)}"
    for s in strokes:
        if not isinstance(s, dict):
            return False, "bad_stroke_dict"
        try:
            P = np.asarray(s.get("mother_bezier"), dtype=np.float32)
            W = width_ctrl_from_stroke(s)
            if P.shape != (4, 2) or W.shape != (4,):
                return False, "bad_shape"
            if not np.isfinite(P).all() or not np.isfinite(W).all():
                return False, "nan_or_inf"
        except Exception:
            return False, "exception_shape"
    return True, "ok"


def bezier_id_map(strokes: List[Dict[str, Any]]) -> Dict[int, int]:
    m = {}
    for i, s in enumerate(strokes):
        try:
            bid = int(s.get("bezier_id", i + 1))
        except Exception:
            bid = i + 1
        m[bid] = i
    return m


def closest_endpoint_pair(Pa, Pb):
    pts_a = [Pa[0], Pa[3]]
    pts_b = [Pb[0], Pb[3]]
    best = (1e9, 0, 0)
    for ia in [0, 1]:
        for ib in [0, 1]:
            d = float(np.linalg.norm(pts_a[ia] - pts_b[ib]))
            if d < best[0]:
                best = (d, ia, ib)
    return best  # dist, endpoint index a, endpoint index b


def endpoint_index_from_name(x):
    if x is None:
        return None
    s = str(x).lower()
    if s in ["0", "p0", "start", "s", "head", "begin", "beg"]:
        return 0
    if s in ["1", "3", "p3", "end", "e", "tail", "finish"]:
        return 1
    return None


def point_to_curve_min_dist_and_t(pt, P, sample_n=64):
    ts = np.linspace(0, 1, sample_n, dtype=np.float32)
    curve = cubic(P, ts)
    d = np.linalg.norm(curve - np.asarray(pt, dtype=np.float32)[None, :], axis=1)
    k = int(np.argmin(d))
    return float(d[k]), float(ts[k])


def choose_guest_endpoint_for_T(ev, guest_stroke, host_stroke):
    for k in ["guest_endpoint", "endpoint", "guest_ep", "guest_point", "endpoint_name"]:
        if k in ev:
            v = endpoint_index_from_name(ev.get(k))
            if v is not None:
                return v
    Pg = np.asarray(guest_stroke["mother_bezier"], dtype=np.float32)
    Ph = np.asarray(host_stroke["mother_bezier"], dtype=np.float32)
    d0, _ = point_to_curve_min_dist_and_t(Pg[0], Ph)
    d1, _ = point_to_curve_min_dist_and_t(Pg[3], Ph)
    return 0 if d0 <= d1 else 1


def stroke_stats(s, cfg: Cfg):
    P = np.asarray(s["mother_bezier"], dtype=np.float32)
    W = width_ctrl_from_stroke(s)
    ts = np.linspace(0, 1, cfg.curve_sample_n, dtype=np.float32)
    curve = cubic(P, ts)
    center = P.mean(axis=0)
    mn, mx = curve.min(axis=0), curve.max(axis=0)
    length = float(np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1))) if len(curve) > 1 else 0.0
    chord = P[3] - P[0]
    chord_len = float(np.linalg.norm(chord))
    t0 = norm_vec(deriv(P, 0.0))
    t1 = norm_vec(deriv(P, 1.0))
    dif = np.diff(curve, axis=0)
    dirs = dif / (np.linalg.norm(dif, axis=1, keepdims=True) + 1e-6)
    if len(dirs) > 1:
        dots = np.clip(np.sum(dirs[:-1] * dirs[1:], axis=1), -1, 1)
        curvature = float(np.mean(np.arccos(dots))) / math.pi
    else:
        curvature = 0.0
    return {
        "P": P, "W": W, "curve": curve, "center": center, "bbox_min": mn, "bbox_max": mx,
        "length": length, "chord_len": chord_len, "t0": t0, "t1": t1,
        "closed": 1.0 if chord_len < 2.0 else 0.0,
        "curvature": curvature,
        "width_mean": float(np.mean(W)),
    }


def make_node_meta(strokes: List[Dict[str, Any]], cfg: Cfg):
    metas = []
    stats = [stroke_stats(s, cfg) for s in strokes]
    # stroke nodes: 0..n-1
    for i, st in enumerate(stats):
        metas.append({
            "node_type": "stroke",
            "stroke_idx": i,
            "endpoint_idx": -1,
            "pos": st["center"],
            "tangent": norm_vec(st["P"][3] - st["P"][0]),
            "stats": st,
        })
    # endpoint nodes: n + 2*i, n + 2*i + 1
    n = len(strokes)
    for i, st in enumerate(stats):
        metas.append({
            "node_type": "endpoint_start",
            "stroke_idx": i,
            "endpoint_idx": 0,
            "pos": st["P"][0],
            "tangent": st["t0"],
            "stats": st,
        })
        metas.append({
            "node_type": "endpoint_end",
            "stroke_idx": i,
            "endpoint_idx": 1,
            "pos": st["P"][3],
            "tangent": st["t1"],
            "stats": st,
        })
    return metas


def node_index_for_endpoint(n_strokes: int, stroke_idx: int, endpoint_idx: int) -> int:
    return n_strokes + 2 * stroke_idx + int(endpoint_idx)


def node_feature(meta: Dict[str, Any], idx: int, cfg: Cfg) -> np.ndarray:
    st = meta["stats"]
    P, W = st["P"], st["W"]
    typ = np.zeros(3, dtype=np.float32)
    typ[NODE_TYPE[meta["node_type"]]] = 1.0
    pos = np.asarray(meta["pos"], dtype=np.float32)
    endpoint_kind = np.zeros(2, dtype=np.float32)
    if meta["endpoint_idx"] == 0:
        endpoint_kind[0] = 1.0
    elif meta["endpoint_idx"] == 1:
        endpoint_kind[1] = 1.0

    feat = np.concatenate([
        typ,                                             # 3
        pos / cfg.canvas_norm,                           # 2
        P.reshape(-1) / cfg.canvas_norm,                 # 8
        W / cfg.width_norm,                              # 4
        st["bbox_min"] / cfg.canvas_norm,                # 2
        st["bbox_max"] / cfg.canvas_norm,                # 2
        np.array([
            st["length"] / cfg.canvas_norm,
            st["chord_len"] / cfg.canvas_norm,
            st["width_mean"] / cfg.width_norm,
            st["closed"],
            st["curvature"],
            meta["stroke_idx"] / max(1, cfg.max_strokes - 1),
        ], dtype=np.float32),                            # 6
        st["t0"], st["t1"],                              # 4
        endpoint_kind,                                   # 2
        np.asarray(meta["tangent"], dtype=np.float32),    # 2
        st["center"] / cfg.canvas_norm,                  # 2
    ]).astype(np.float32)                                # total 37
    return feat


def edge_feature(mi: Dict[str, Any], mj: Dict[str, Any], cfg: Cfg) -> np.ndarray:
    ti = NODE_TYPE[mi["node_type"]]
    tj = NODE_TYPE[mj["node_type"]]
    type_pair = np.zeros(9, dtype=np.float32)
    type_pair[ti * 3 + tj] = 1.0

    pi = np.asarray(mi["pos"], dtype=np.float32)
    pj = np.asarray(mj["pos"], dtype=np.float32)
    delta = pj - pi
    dist = float(np.linalg.norm(delta))

    same_stroke = 1.0 if mi["stroke_idx"] == mj["stroke_idx"] else 0.0
    both_endpoint = 1.0 if mi["node_type"].startswith("endpoint") and mj["node_type"].startswith("endpoint") else 0.0
    endpoint_to_stroke = 1.0 if mi["node_type"].startswith("endpoint") and mj["node_type"] == "stroke" else 0.0
    stroke_to_endpoint = 1.0 if mi["node_type"] == "stroke" and mj["node_type"].startswith("endpoint") else 0.0
    stroke_to_stroke = 1.0 if mi["node_type"] == "stroke" and mj["node_type"] == "stroke" else 0.0

    tangent_cos = float(np.clip(np.dot(norm_vec(mi["tangent"]), norm_vec(mj["tangent"])), -1, 1))
    tangent_sin_abs = float(abs(mi["tangent"][0] * mj["tangent"][1] - mi["tangent"][1] * mj["tangent"][0]))

    endpoint_curve_dist = 0.0
    endpoint_curve_t = 0.0
    curve_curve_dist = 0.0

    if endpoint_to_stroke:
        endpoint_curve_dist, endpoint_curve_t = point_to_curve_min_dist_and_t(pi, mj["stats"]["P"], sample_n=48)
    elif stroke_to_endpoint:
        endpoint_curve_dist, endpoint_curve_t = point_to_curve_min_dist_and_t(pj, mi["stats"]["P"], sample_n=48)
    elif stroke_to_stroke:
        c1 = mi["stats"]["curve"][::2]
        c2 = mj["stats"]["curve"][::2]
        dd = np.linalg.norm(c1[:, None, :] - c2[None, :, :], axis=2)
        curve_curve_dist = float(dd.min())

    feat = np.concatenate([
        type_pair,                                       # 9
        np.array([
            dist / cfg.dist_norm,
            delta[0] / cfg.canvas_norm,
            delta[1] / cfg.canvas_norm,
            same_stroke,
            both_endpoint,
            endpoint_to_stroke,
            stroke_to_endpoint,
            stroke_to_stroke,
            tangent_cos,
            tangent_sin_abs,
            endpoint_curve_dist / cfg.dist_norm,
            endpoint_curve_t,
            curve_curve_dist / cfg.dist_norm,
        ], dtype=np.float32),                            # 13
    ]).astype(np.float32)                                # total 22
    return feat


def render_raster(bundle: Dict[str, Any], cfg: Cfg) -> np.ndarray:
    size = cfg.raster_size
    strokes = bundle.get("strokes", [])
    if not PIL_AVAILABLE or not strokes:
        return np.zeros((1, size, size), dtype=np.float32)

    curves = []
    widths = []
    for s in strokes:
        try:
            P = np.asarray(s["mother_bezier"], dtype=np.float32)
            ts = np.linspace(0, 1, 64, dtype=np.float32)
            c = cubic(P, ts)
            curves.append(c)
            widths.append(float(np.mean(width_ctrl_from_stroke(s))))
        except Exception:
            pass

    if not curves:
        return np.zeros((1, size, size), dtype=np.float32)

    allp = np.concatenate(curves, axis=0)
    mn = allp.min(axis=0)
    mx = allp.max(axis=0)
    center = (mn + mx) / 2.0
    span = float(max(mx[0] - mn[0], mx[1] - mn[1], 1.0))
    scale = (size * 0.78) / span

    img = Image.new("L", (size, size), 0)
    dr = ImageDraw.Draw(img)
    for c, w in zip(curves, widths):
        pts = (c - center[None, :]) * scale + np.array([size / 2, size / 2], dtype=np.float32)
        pts = [(float(x), float(y)) for x, y in pts]
        lw = max(1, int(round(w * scale)))
        if len(pts) >= 2:
            dr.line(pts, fill=255, width=lw, joint="curve")

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[None, :, :]


def relation_labels_for_typed_nodes(bundle: Dict[str, Any], cfg: Cfg) -> Tuple[np.ndarray, Dict[str, int]]:
    strokes = bundle.get("strokes", [])
    n = len(strokes)
    N = n * 3
    rel = np.zeros((N, N), dtype=np.int64)
    idmap = bezier_id_map(strokes)
    counts = {"NONE": 0, "E2E": 0, "T": 0, "X": 0}

    events = bundle.get("topology_events", [])
    if isinstance(events, list):
        for ev in events:
            if not isinstance(ev, dict):
                continue
            typ = ev.get("type")
            try:
                if typ == "E2E":
                    a_id, b_id = int(ev["stroke_a"]), int(ev["stroke_b"])
                    if a_id in idmap and b_id in idmap:
                        ia, ib = idmap[a_id], idmap[b_id]
                        Pa = np.asarray(strokes[ia]["mother_bezier"], dtype=np.float32)
                        Pb = np.asarray(strokes[ib]["mother_bezier"], dtype=np.float32)
                        _, ea, eb = closest_endpoint_pair(Pa, Pb)
                        na = node_index_for_endpoint(n, ia, ea)
                        nb = node_index_for_endpoint(n, ib, eb)
                        rel[na, nb] = REL_MAP["E2E"]
                        rel[nb, na] = REL_MAP["E2E"]
                elif typ == "X":
                    a_id, b_id = int(ev["stroke_a"]), int(ev["stroke_b"])
                    if a_id in idmap and b_id in idmap:
                        ia, ib = idmap[a_id], idmap[b_id]
                        rel[ia, ib] = REL_MAP["X"]
                        rel[ib, ia] = REL_MAP["X"]
                elif typ == "T":
                    g_id, h_id = int(ev["guest"]), int(ev["host"])
                    if g_id in idmap and h_id in idmap:
                        ig, ih = idmap[g_id], idmap[h_id]
                        ep = choose_guest_endpoint_for_T(ev, strokes[ig], strokes[ih])
                        ng = node_index_for_endpoint(n, ig, ep)
                        nh = ih  # host stroke node
                        rel[ng, nh] = REL_MAP["T"]
            except Exception:
                continue

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            counts[REL_INV[int(rel[i, j])]] += 1
    return rel, counts


def relation_binary_targets_for_typed_nodes(metas: List[Dict[str, Any]], rel: np.ndarray, cfg: Cfg) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, int]]:
    """
    为三个 typed binary heads 构造 label/mask：

      E2E head:
        endpoint <-> endpoint candidate
        label=1 当 pair 是 E2E

      T head:
        endpoint -> stroke candidate
        label=1 当 pair 是 T

      X head:
        stroke <-> stroke candidate
        label=1 当 pair 是 X

    只在对应类型的 candidate pair 上计算 BCE loss。
    正例默认总是加入，负例只保留几何 hard negatives。
    """
    N = len(metas)

    labels = {
        "e2e": np.zeros((N, N), dtype=np.float32),
        "t":   np.zeros((N, N), dtype=np.float32),
        "x":   np.zeros((N, N), dtype=np.float32),
    }
    masks = {
        "e2e": np.zeros((N, N), dtype=np.bool_),
        "t":   np.zeros((N, N), dtype=np.bool_),
        "x":   np.zeros((N, N), dtype=np.bool_),
    }
    stats = {
        "e2e_pos": 0, "e2e_neg": 0, "e2e_candidates": 0,
        "t_pos": 0,   "t_neg": 0,   "t_candidates": 0,
        "x_pos": 0,   "x_neg": 0,   "x_candidates": 0,
        "all_candidate_pairs": 0,
    }

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            mi, mj = metas[i], metas[j]
            lab = int(rel[i, j])

            # ---------------- E2E: endpoint <-> endpoint ----------------
            is_e2e_pair_type = (
                mi["node_type"].startswith("endpoint")
                and mj["node_type"].startswith("endpoint")
                and mi["stroke_idx"] != mj["stroke_idx"]
            )
            if is_e2e_pair_type:
                positive = (lab == REL_MAP["E2E"])
                is_candidate = positive
                if not is_candidate:
                    d = float(np.linalg.norm(np.asarray(mi["pos"]) - np.asarray(mj["pos"])))
                    is_candidate = d <= cfg.cand_e2e_endpoint_dist_max
                if is_candidate:
                    masks["e2e"][i, j] = True
                    labels["e2e"][i, j] = 1.0 if positive else 0.0
                    if positive:
                        stats["e2e_pos"] += 1
                    else:
                        stats["e2e_neg"] += 1

            # ---------------- T: endpoint -> stroke ----------------
            is_t_pair_type = (
                mi["node_type"].startswith("endpoint")
                and mj["node_type"] == "stroke"
                and mi["stroke_idx"] != mj["stroke_idx"]
            )
            if is_t_pair_type:
                positive = (lab == REL_MAP["T"])
                is_candidate = positive
                if not is_candidate:
                    d, _ = point_to_curve_min_dist_and_t(mi["pos"], mj["stats"]["P"], sample_n=48)
                    is_candidate = d <= cfg.cand_t_endpoint_curve_dist_max
                if is_candidate:
                    masks["t"][i, j] = True
                    labels["t"][i, j] = 1.0 if positive else 0.0
                    if positive:
                        stats["t_pos"] += 1
                    else:
                        stats["t_neg"] += 1

            # ---------------- X: stroke <-> stroke ----------------
            is_x_pair_type = (
                mi["node_type"] == "stroke"
                and mj["node_type"] == "stroke"
                and mi["stroke_idx"] != mj["stroke_idx"]
            )
            if is_x_pair_type:
                positive = (lab == REL_MAP["X"])
                is_candidate = positive
                if not is_candidate:
                    c1 = mi["stats"]["curve"][::2]
                    c2 = mj["stats"]["curve"][::2]
                    dd = np.linalg.norm(c1[:, None, :] - c2[None, :, :], axis=2)
                    is_candidate = float(dd.min()) <= cfg.cand_x_curve_curve_dist_max
                if is_candidate:
                    masks["x"][i, j] = True
                    labels["x"][i, j] = 1.0 if positive else 0.0
                    if positive:
                        stats["x_pos"] += 1
                    else:
                        stats["x_neg"] += 1

    for k in ["e2e", "t", "x"]:
        stats[f"{k}_candidates"] = int(masks[k].sum())
    stats["all_candidate_pairs"] = int(masks["e2e"].sum() + masks["t"].sum() + masks["x"].sum())
    return labels, masks, stats


# =============================================================================
# 4. Data loading
# =============================================================================

def iter_bundles_from_json_file(fp: str):
    try:
        data = load_json(fp)
    except Exception as e:
        print(f"[LoadError] {fp} | {repr(e)}")
        return

    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict) and isinstance(v.get("strokes"), list):
                yield str(k), v
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, dict) and isinstance(vv.get("strokes"), list):
                        yield f"{k}/{kk}", vv
    elif isinstance(data, list):
        for i, v in enumerate(data):
            if isinstance(v, dict) and isinstance(v.get("strokes"), list):
                yield str(i), v


def json_files_under(d: str) -> List[str]:
    if not os.path.exists(d):
        return []
    return sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True))


def load_source_records(cfg: Cfg, annotations_dir: str, pool_root: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    source_dirs = [
        ("annotations_topo", annotations_dir, 1 if cfg.annotations_are_positive else None),
        ("bad", os.path.join(pool_root, "bad"), 0),
        ("good", os.path.join(pool_root, "good"), 1),
        ("cleaned", os.path.join(pool_root, "cleaned"), 1),
    ]

    raw_records = []
    stats = {"source_dirs": {}, "skip_reasons": {}, "duplicates": 0, "conflicts": []}

    for source, d, label in source_dirs:
        files = json_files_under(d)
        stats["source_dirs"][source] = {"dir": d, "files": len(files), "raw_items": 0, "valid_items": 0}
        print(f"[DataLoad] source={source:<16} files={len(files):5d} dir={d}")

        for fp in files:
            for outer, bundle in iter_bundles_from_json_file(fp) or []:
                stats["source_dirs"][source]["raw_items"] += 1
                if label is None:
                    continue
                ok, reason = valid_bundle(bundle, cfg.max_strokes)
                if not ok:
                    stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
                    continue
                uid = stable_uid(bundle, outer)
                raw_records.append({
                    "uid": uid,
                    "outer_key": outer,
                    "file": fp,
                    "source": source,
                    "priority": SOURCE_PRIORITY.get(source, 0),
                    "quality_label": int(label),
                    "bundle": bundle,
                    "stroke_count": len(bundle.get("strokes", [])),
                    "group_key": group_key_for_bundle(bundle, source, uid),
                })
                stats["source_dirs"][source]["valid_items"] += 1

        s = stats["source_dirs"][source]
        print(f"[DataLoad]   raw_items={s['raw_items']} valid_items={s['valid_items']}")

    chosen = {}
    for r in raw_records:
        uid = r["uid"]
        if uid not in chosen:
            chosen[uid] = r
            continue
        prev = chosen[uid]
        if prev["quality_label"] != r["quality_label"]:
            stats["conflicts"].append({
                "uid": uid,
                "prev_source": prev["source"],
                "prev_label": prev["quality_label"],
                "new_source": r["source"],
                "new_label": r["quality_label"],
            })
        if r["priority"] > prev["priority"]:
            chosen[uid] = r
        stats["duplicates"] += 1

    records = list(chosen.values())
    records.sort(key=lambda x: x["uid"])

    label_counts = {0: 0, 1: 0}
    source_counts = {}
    stroke_counts = []
    group_counts = {}
    for r in records:
        label_counts[int(r["quality_label"])] += 1
        source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1
        stroke_counts.append(r["stroke_count"])
        group_counts[r["group_key"]] = group_counts.get(r["group_key"], 0) + 1

    stats["deduped_count"] = len(records)
    stats["label_counts"] = label_counts
    stats["source_counts_after_dedup"] = source_counts
    stats["stroke_count_percentile"] = percentile(stroke_counts)
    stats["group_count"] = len(group_counts)
    stats["group_size_percentile"] = percentile(list(group_counts.values()))
    stats["conflict_count"] = len(stats["conflicts"])

    print("[DataLoad] ---------------- summary ----------------")
    print(f"[DataLoad] raw_records={len(raw_records)} deduped={len(records)} duplicates_hidden={stats['duplicates']}")
    print(f"[DataLoad] labels: bad=0 -> {label_counts[0]}, good=1 -> {label_counts[1]}")
    print(f"[DataLoad] source after dedup: {source_counts}")
    print(f"[DataLoad] groups={stats['group_count']} group_size={stats['group_size_percentile']}")
    print(f"[DataLoad] stroke_count percentiles: {stats['stroke_count_percentile']}")
    print(f"[DataLoad] skip_reasons: {stats['skip_reasons']}")
    print(f"[DataLoad] conflict_count={stats['conflict_count']}")
    if stats["conflicts"]:
        print("[DataLoad] first conflicts:")
        for c in stats["conflicts"][:8]:
            print("  ", c)

    return records, stats


# =============================================================================
# 5. Dataset
# =============================================================================

class FlywheelTopoDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], cfg: Cfg):
        self.records = records
        self.cfg = cfg
        self.max_nodes = cfg.max_strokes * 3
        self.items = None

        self.cache_key = cache_signature(records, cfg)
        self.cache_path = os.path.join(PREPROCESS_CACHE_DIR, f"{cfg.save_prefix}_precomputed_{self.cache_key}.pt")

        loaded_cache = False
        if cfg.precompute_items and cfg.cache_preprocessed_pt and (not cfg.force_rebuild_cache) and os.path.exists(self.cache_path):
            try:
                t0 = time.perf_counter()
                payload = torch.load(self.cache_path, map_location="cpu")
                if payload.get("cache_key") == self.cache_key:
                    self.items = payload["items"]
                    self.node_dim = int(payload["node_dim"])
                    self.edge_dim = int(payload["edge_dim"])
                    self.rel_counts = np.asarray(payload["rel_counts"], dtype=np.int64)
                    self.raw_rel_counts = np.asarray(payload.get("raw_rel_counts", payload["rel_counts"]), dtype=np.int64)
                    self.candidate_pair_counts = payload.get("candidate_pair_counts", {})
                    self.binary_rel_counts = payload.get("binary_rel_counts", {})
                    self.quality_counts = np.asarray(payload["quality_counts"], dtype=np.int64)
                    self.stroke_counts = list(payload["stroke_counts"])
                    self.valid_node_counts = list(payload["valid_node_counts"])
                    loaded_cache = True
                    print(f"[DatasetCache] loaded precomputed items: {self.cache_path} | n={len(self.items)} | time={time.perf_counter()-t0:.3f}s")
            except Exception as e:
                print(f"[DatasetCache][WARN] failed to load cache, rebuild. path={self.cache_path} err={repr(e)}")
                self.items = None

        if not loaded_cache:
            item0 = self._make_item(0)
            self.node_dim = item0["node_features"].shape[-1]
            self.edge_dim = item0["edge_features"].shape[-1]

            self.rel_counts = np.zeros(4, dtype=np.int64)       # candidate-pair counts for loss
            self.raw_rel_counts = np.zeros(4, dtype=np.int64)   # full valid-pair counts for debug
            self.candidate_pair_counts = {}
            self.binary_rel_counts = {"e2e_pos": 0, "e2e_neg": 0, "t_pos": 0, "t_neg": 0, "x_pos": 0, "x_neg": 0}
            self.quality_counts = np.zeros(2, dtype=np.int64)
            self.stroke_counts = []
            self.valid_node_counts = []

            if cfg.precompute_items:
                print(f"[DatasetCache] precomputing graph+raster tensors once... n={len(self.records)} cache={self.cache_path}")
                t0 = time.perf_counter()
                items = []
                cand_stats_acc = {}
                binary_rel_acc = {"e2e_pos": 0, "e2e_neg": 0, "t_pos": 0, "t_neg": 0, "x_pos": 0, "x_neg": 0}
                for i in range(len(self.records)):
                    if i > 0 and i % 250 == 0:
                        print(f"[DatasetCache]   {i}/{len(self.records)} elapsed={time.perf_counter()-t0:.1f}s")
                    it = self._make_item(i)
                    items.append(it)

                    r = self.records[i]
                    self.quality_counts[int(r["quality_label"])] += 1
                    self.stroke_counts.append(int(r["stroke_count"]))
                    self.valid_node_counts.append(int(r["stroke_count"]) * 3)
                    rel = it["rel_labels"].numpy()
                    raw_mask = rel != IGNORE
                    vals_raw = rel[raw_mask]
                    for k in range(4):
                        self.raw_rel_counts[k] += int((vals_raw == k).sum())

                    # v5 binary-head relation counts
                    e2e_mask = it["e2e_mask"].numpy().astype(bool)
                    t_mask = it["t_mask"].numpy().astype(bool)
                    x_mask = it["x_mask"].numpy().astype(bool)
                    e2e_lab = it["e2e_label"].numpy()
                    t_lab = it["t_label"].numpy()
                    x_lab = it["x_label"].numpy()

                    binary_rel_acc["e2e_pos"] += int(((e2e_lab > 0.5) & e2e_mask).sum())
                    binary_rel_acc["e2e_neg"] += int(((e2e_lab <= 0.5) & e2e_mask).sum())
                    binary_rel_acc["t_pos"] += int(((t_lab > 0.5) & t_mask).sum())
                    binary_rel_acc["t_neg"] += int(((t_lab <= 0.5) & t_mask).sum())
                    binary_rel_acc["x_pos"] += int(((x_lab > 0.5) & x_mask).sum())
                    binary_rel_acc["x_neg"] += int(((x_lab <= 0.5) & x_mask).sum())

                    # for backward-compatible print only: candidate multiclass-ish distribution
                    for kk, vv in it.get("candidate_stats", {}).items():
                        cand_stats_acc[kk] = cand_stats_acc.get(kk, 0) + int(vv)

                self.items = items
                self.candidate_pair_counts = cand_stats_acc
                self.binary_rel_counts = binary_rel_acc

                # fill rel_counts for compatibility with checkpoint/logs:
                # NONE count = all binary negatives; E2E/T/X = each head positives.
                self.rel_counts = np.array([
                    binary_rel_acc["e2e_neg"] + binary_rel_acc["t_neg"] + binary_rel_acc["x_neg"],
                    binary_rel_acc["e2e_pos"],
                    binary_rel_acc["t_pos"],
                    binary_rel_acc["x_pos"],
                ], dtype=np.int64)

                print(f"[DatasetCache] precompute done. time={time.perf_counter()-t0:.2f}s")

                if cfg.cache_preprocessed_pt:
                    try:
                        torch.save({
                            "cache_key": self.cache_key,
                            "items": self.items,
                            "node_dim": self.node_dim,
                            "edge_dim": self.edge_dim,
                            "rel_counts": self.rel_counts.tolist(),
                            "raw_rel_counts": self.raw_rel_counts.tolist(),
                            "candidate_pair_counts": self.candidate_pair_counts,
                            "binary_rel_counts": self.binary_rel_counts,
                            "quality_counts": self.quality_counts.tolist(),
                            "stroke_counts": self.stroke_counts,
                            "valid_node_counts": self.valid_node_counts,
                        }, self.cache_path)
                        print(f"[DatasetCache] saved: {self.cache_path}")
                    except Exception as e:
                        print(f"[DatasetCache][WARN] failed to save cache: {repr(e)}")
            else:
                # old behavior: no item cache, but still compute counts once
                for r in self.records:
                    self.quality_counts[int(r["quality_label"])] += 1
                    self.stroke_counts.append(int(r["stroke_count"]))
                    self.valid_node_counts.append(int(r["stroke_count"]) * 3)
                    it = self._make_item(len(self.stroke_counts)-1)
                    rel = it["rel_labels"].numpy()
                    raw_mask = rel != IGNORE
                    vals_raw = rel[raw_mask]
                    for k in range(4):
                        self.raw_rel_counts[k] += int((vals_raw == k).sum())
                    for kk, vv in it.get("candidate_stats", {}).items():
                        self.candidate_pair_counts[kk] = self.candidate_pair_counts.get(kk, 0) + int(vv)
                    for head in ["e2e", "t", "x"]:
                        lab = it[f"{head}_label"].numpy()
                        msk = it[f"{head}_mask"].numpy().astype(bool)
                        self.binary_rel_counts[f"{head}_pos"] += int(((lab > 0.5) & msk).sum())
                        self.binary_rel_counts[f"{head}_neg"] += int(((lab <= 0.5) & msk).sum())
                    self.rel_counts = np.array([
                        self.binary_rel_counts["e2e_neg"] + self.binary_rel_counts["t_neg"] + self.binary_rel_counts["x_neg"],
                        self.binary_rel_counts["e2e_pos"],
                        self.binary_rel_counts["t_pos"],
                        self.binary_rel_counts["x_pos"],
                    ], dtype=np.int64)

        print(f"[Dataset] samples={len(self.records)} max_nodes={self.max_nodes} node_dim={self.node_dim} edge_dim={self.edge_dim}")
        print(f"[Dataset] precompute_items={self.cfg.precompute_items} cache_preprocessed_pt={self.cfg.cache_preprocessed_pt}")
        print("[Dataset] node_types: stroke + endpoint_start + endpoint_end")
        print("[Dataset] quality_counts:", {str(k): int(v) for k, v in enumerate(self.quality_counts)})
        print("[Dataset] binary_rel_counts_for_loss:", self.binary_rel_counts)
        print("[Dataset] rel_counts_compat_binary_summary:", {REL_INV[i]: int(self.rel_counts[i]) for i in range(4)})
        print("[Dataset] rel_counts_raw_full_pairs:", {REL_INV[i]: int(self.raw_rel_counts[i]) for i in range(4)})
        print("[Dataset] candidate_pair_counts:", self.candidate_pair_counts)
        print("[Dataset] stroke_count:", percentile(self.stroke_counts))
        print("[Dataset] valid_node_count:", percentile(self.valid_node_counts))

    def __len__(self):
        return len(self.records)

    def _make_item(self, idx):
        cfg = self.cfg
        r = self.records[idx]
        bundle = r["bundle"]
        strokes = bundle["strokes"]
        n_strokes = len(strokes)
        max_nodes = self.max_nodes

        metas = make_node_meta(strokes, cfg)
        n_nodes = len(metas)

        node_features_valid = np.asarray([node_feature(m, i, cfg) for i, m in enumerate(metas)], dtype=np.float32)
        node_dim = node_features_valid.shape[-1]
        node_features = np.zeros((max_nodes, node_dim), dtype=np.float32)
        node_features[:n_nodes] = node_features_valid

        edge_dim = edge_feature(metas[0], metas[0], cfg).shape[-1]
        edge_features = np.zeros((max_nodes, max_nodes, edge_dim), dtype=np.float32)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    edge_features[i, j] = 0.0
                else:
                    edge_features[i, j] = edge_feature(metas[i], metas[j], cfg)

        rel_raw, _ = relation_labels_for_typed_nodes(bundle, cfg)
        bin_labels_raw, bin_masks_raw, candidate_stats = relation_binary_targets_for_typed_nodes(metas, rel_raw, cfg)

        rel_labels = np.full((max_nodes, max_nodes), IGNORE, dtype=np.int64)
        rel_labels[:n_nodes, :n_nodes] = rel_raw

        # v5 binary labels/masks, padded to max_nodes
        e2e_label = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        t_label = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        x_label = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        e2e_mask = np.zeros((max_nodes, max_nodes), dtype=np.bool_)
        t_mask = np.zeros((max_nodes, max_nodes), dtype=np.bool_)
        x_mask = np.zeros((max_nodes, max_nodes), dtype=np.bool_)

        e2e_label[:n_nodes, :n_nodes] = bin_labels_raw["e2e"]
        t_label[:n_nodes, :n_nodes] = bin_labels_raw["t"]
        x_label[:n_nodes, :n_nodes] = bin_labels_raw["x"]
        e2e_mask[:n_nodes, :n_nodes] = bin_masks_raw["e2e"]
        t_mask[:n_nodes, :n_nodes] = bin_masks_raw["t"]
        x_mask[:n_nodes, :n_nodes] = bin_masks_raw["x"]

        for i in range(max_nodes):
            rel_labels[i, i] = IGNORE
            e2e_mask[i, i] = False
            t_mask[i, i] = False
            x_mask[i, i] = False
        if n_nodes < max_nodes:
            rel_labels[n_nodes:, :] = IGNORE
            rel_labels[:, n_nodes:] = IGNORE

        node_mask = np.zeros((max_nodes,), dtype=np.float32)
        node_mask[:n_nodes] = 1.0

        raster = render_raster(bundle, cfg)

        return {
            "uid": r["uid"],
            "source": r["source"],
            "group_key": r["group_key"],
            "node_features": torch.tensor(node_features, dtype=torch.float32),
            "edge_features": torch.tensor(edge_features, dtype=torch.float32),
            "node_mask": torch.tensor(node_mask, dtype=torch.float32),
            "rel_labels": torch.tensor(rel_labels, dtype=torch.long),  # debug only
            "e2e_label": torch.tensor(e2e_label, dtype=torch.float32),
            "e2e_mask": torch.tensor(e2e_mask, dtype=torch.bool),
            "t_label": torch.tensor(t_label, dtype=torch.float32),
            "t_mask": torch.tensor(t_mask, dtype=torch.bool),
            "x_label": torch.tensor(x_label, dtype=torch.float32),
            "x_mask": torch.tensor(x_mask, dtype=torch.bool),
            "candidate_stats": candidate_stats,
            "raster": torch.tensor(raster, dtype=torch.float32),
            "quality": torch.tensor(float(r["quality_label"]), dtype=torch.float32),
            "stroke_count": torch.tensor(n_strokes, dtype=torch.long),
        }

    def __getitem__(self, idx):
        if self.items is not None:
            return self.items[idx]
        return self._make_item(idx)


# =============================================================================
# 6. Split
# =============================================================================

def split_dataset(ds: FlywheelTopoDataset, cfg: Cfg):
    n = len(ds)
    rng = random.Random(cfg.seed)

    if n < 10:
        print("[Split] too few samples; train=val=all")
        return Subset(ds, list(range(n))), Subset(ds, list(range(n))), {"note": "too_few_samples", "train": n, "val": n}

    if cfg.group_split:
        group_to_indices = {}
        for i, r in enumerate(ds.records):
            group_to_indices.setdefault(r["group_key"], []).append(i)

        groups = list(group_to_indices.keys())
        rng.shuffle(groups)

        pos_groups, neg_groups, mixed_groups = [], [], []
        for g in groups:
            labs = [int(ds.records[i]["quality_label"]) for i in group_to_indices[g]]
            if all(x == 1 for x in labs):
                pos_groups.append(g)
            elif all(x == 0 for x in labs):
                neg_groups.append(g)
            else:
                mixed_groups.append(g)

        def take_val(gs):
            k = max(1, int(round(len(gs) * cfg.val_ratio))) if len(gs) > 1 else 0
            return set(gs[:k])

        val_groups = take_val(pos_groups) | take_val(neg_groups)
        train_idx, val_idx = [], []
        for g, ids in group_to_indices.items():
            if g in val_groups:
                val_idx.extend(ids)
            else:
                train_idx.extend(ids)

        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        train_groups = set(ds.records[i]["group_key"] for i in train_idx)
        val_groups_real = set(ds.records[i]["group_key"] for i in val_idx)
        overlap = sorted(train_groups.intersection(val_groups_real))

        info = {
            "mode": "group_split",
            "groups_total": len(group_to_indices),
            "groups_train": len(train_groups),
            "groups_val": len(val_groups_real),
            "group_overlap_count": len(overlap),
            "train": len(train_idx),
            "val": len(val_idx),
            "train_pos": sum(int(ds.records[i]["quality_label"]) == 1 for i in train_idx),
            "train_neg": sum(int(ds.records[i]["quality_label"]) == 0 for i in train_idx),
            "val_pos": sum(int(ds.records[i]["quality_label"]) == 1 for i in val_idx),
            "val_neg": sum(int(ds.records[i]["quality_label"]) == 0 for i in val_idx),
            "mixed_group_count": len(mixed_groups),
        }

        if len(val_idx) > 0 and info["val_pos"] > 0 and info["val_neg"] > 0:
            print(f"[Split] {info}")
            if overlap:
                print("[Split][WARN] group leakage overlap:", overlap[:8])
            return Subset(ds, train_idx), Subset(ds, val_idx), info

        print("[Split][WARN] group split produced weak val distribution, fallback to stratified random.")

    idxs = list(range(n))
    rng.shuffle(idxs)
    pos = [i for i in idxs if int(ds.records[i]["quality_label"]) == 1]
    neg = [i for i in idxs if int(ds.records[i]["quality_label"]) == 0]
    if len(pos) > 1 and len(neg) > 1:
        val_pos_n = max(1, int(round(len(pos) * cfg.val_ratio)))
        val_neg_n = max(1, int(round(len(neg) * cfg.val_ratio)))
        val_idx = pos[:val_pos_n] + neg[:val_neg_n]
        train_idx = pos[val_pos_n:] + neg[val_neg_n:]
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        info = {
            "mode": "stratified_random",
            "train": len(train_idx),
            "val": len(val_idx),
            "train_pos": sum(int(ds.records[i]["quality_label"]) == 1 for i in train_idx),
            "train_neg": sum(int(ds.records[i]["quality_label"]) == 0 for i in train_idx),
            "val_pos": sum(int(ds.records[i]["quality_label"]) == 1 for i in val_idx),
            "val_neg": sum(int(ds.records[i]["quality_label"]) == 0 for i in val_idx),
        }
    else:
        val_n = max(1, int(round(n * cfg.val_ratio)))
        val_idx = idxs[:val_n]
        train_idx = idxs[val_n:]
        info = {"mode": "random", "train": len(train_idx), "val": len(val_idx)}
    print(f"[Split] {info}")
    return Subset(ds, train_idx), Subset(ds, val_idx), info


# =============================================================================
# 7. Model
# =============================================================================

class EdgeAwareGraphBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        h = hidden_dim
        self.score_mlp = nn.Sequential(nn.Linear(h * 2 + edge_dim, h), nn.GELU(), nn.Linear(h, 1))
        self.msg_mlp = nn.Sequential(nn.Linear(h + edge_dim, h), nn.GELU(), nn.Linear(h, h))
        self.out = nn.Sequential(nn.Linear(h * 2, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, h))
        self.norm1 = nn.LayerNorm(h)
        self.ffn = nn.Sequential(nn.Linear(h, h * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(h * 4, h))
        self.norm2 = nn.LayerNorm(h)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge, node_mask):
        B, N, H = h.shape
        hi = h.unsqueeze(2).expand(B, N, N, H)
        hj = h.unsqueeze(1).expand(B, N, N, H)
        score = self.score_mlp(torch.cat([hi, hj, edge], dim=-1)).squeeze(-1)

        key_mask = node_mask.unsqueeze(1).expand(B, N, N) > 0
        query_mask = node_mask.unsqueeze(2).expand(B, N, N) > 0
        attn_mask = key_mask & query_mask
        score = score.masked_fill(~attn_mask, -1e9)
        alpha = torch.softmax(score, dim=-1)
        alpha = alpha.masked_fill(~attn_mask, 0.0)

        msg = self.msg_mlp(torch.cat([hj, edge], dim=-1))
        agg = (alpha.unsqueeze(-1) * msg).sum(dim=2)
        h2 = self.norm1(h + self.dropout(self.out(torch.cat([h, agg], dim=-1))))
        h3 = self.norm2(h2 + self.dropout(self.ffn(h2)))
        return h3 * node_mask.unsqueeze(-1)


class RasterCNN(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.GELU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(96, out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class TopoHGTv6(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim
        self.node_embed = nn.Sequential(nn.Linear(node_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, h))
        self.edge_embed = nn.Sequential(nn.Linear(edge_dim, h // 2), nn.LayerNorm(h // 2), nn.GELU(), nn.Linear(h // 2, h // 2))
        self.blocks = nn.ModuleList([EdgeAwareGraphBlock(h, h // 2, cfg.dropout) for _ in range(cfg.num_layers)])
        self.raster_cnn = RasterCNN(h)

        self.quality_head = nn.Sequential(
            nn.Linear(h * 3 + 1, h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, 1),
        )

        pair_in_dim = h * 2 + h // 2
        self.e2e_head = nn.Sequential(
            nn.Linear(pair_in_dim, h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, 1),
        )
        self.t_head = nn.Sequential(
            nn.Linear(pair_in_dim, h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, 1),
        )
        self.x_head = nn.Sequential(
            nn.Linear(pair_in_dim, h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, 1),
        )

    def forward(self, batch):
        x = batch["node_features"]
        e = batch["edge_features"]
        m = batch["node_mask"]
        raster = batch["raster"]

        h = self.node_embed(x) * m.unsqueeze(-1)
        ee = self.edge_embed(e)

        for blk in self.blocks:
            h = blk(h, ee, m)

        B, N, H = h.shape
        hi = h.unsqueeze(2).expand(B, N, N, H)
        hj = h.unsqueeze(1).expand(B, N, N, H)
        pair_repr = torch.cat([hi, hj, ee], dim=-1)
        e2e_logit = self.e2e_head(pair_repr).squeeze(-1)
        t_logit = self.t_head(pair_repr).squeeze(-1)
        x_logit = self.x_head(pair_repr).squeeze(-1)

        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_pool = (h * m.unsqueeze(-1)).sum(dim=1) / denom
        h_masked = h.masked_fill(m.unsqueeze(-1) <= 0, -1e9)
        max_pool = h_masked.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        raster_emb = self.raster_cnn(raster)
        node_count_norm = denom / float(self.cfg.max_strokes * 3)

        q_logit = self.quality_head(torch.cat([mean_pool, max_pool, raster_emb, node_count_norm], dim=-1)).squeeze(-1)
        return {
            "quality_logit": q_logit,
            "e2e_logit": e2e_logit,
            "t_logit": t_logit,
            "x_logit": x_logit,
        }


# =============================================================================
# 8. Loss / metrics
# =============================================================================

def relation_pos_weights(binary_counts: Dict[str, int], cfg: Cfg):
    if cfg.use_manual_rel_pos_weight:
        return {
            "e2e": float(cfg.e2e_pos_weight_manual),
            "t": float(cfg.t_pos_weight_manual),
            "x": float(cfg.x_pos_weight_manual),
        }

    out = {}
    for head in ["e2e", "t", "x"]:
        pos = float(binary_counts.get(f"{head}_pos", 0))
        neg = float(binary_counts.get(f"{head}_neg", 0))
        if pos <= 0:
            out[head] = 1.0
        else:
            out[head] = float(min(cfg.rel_pos_weight_clip, max(1.0, neg / max(1.0, pos))))
    return out


def quality_pos_weight(counts: np.ndarray, cfg: Cfg):
    neg = float(counts[0])
    pos = float(counts[1])
    if pos <= 0:
        return 1.0
    return float(min(cfg.quality_pos_weight_clip, max(1.0, neg / max(1.0, pos))))


def masked_bce_loss(logit, label, mask, loss_fn):
    raw = loss_fn(logit, label)
    mask = mask.bool()
    if bool(mask.any().detach().cpu()):
        return raw[mask].mean()
    return logit.sum() * 0.0


def compute_loss(out, batch, cfg: Cfg, bce_quality, bce_rel):
    q_loss = bce_quality(out["quality_logit"], batch["quality"])

    e2e_loss = masked_bce_loss(out["e2e_logit"], batch["e2e_label"], batch["e2e_mask"], bce_rel["e2e"])
    t_loss   = masked_bce_loss(out["t_logit"],   batch["t_label"],   batch["t_mask"],   bce_rel["t"])
    x_loss   = masked_bce_loss(out["x_logit"],   batch["x_label"],   batch["x_mask"],   bce_rel["x"])

    rel_loss = e2e_loss + t_loss + x_loss
    loss = cfg.w_quality * q_loss + cfg.w_relation * rel_loss
    return loss, {
        "loss": loss.detach(),
        "quality": q_loss.detach(),
        "relation": rel_loss.detach(),
        "e2e_loss": e2e_loss.detach(),
        "t_loss": t_loss.detach(),
        "x_loss": x_loss.detach(),
    }


def binary_metrics_at_threshold(probs, labels, th=0.5):
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    pred = probs >= th
    true = labels > 0
    tp = int(np.logical_and(pred, true).sum())
    fp = int(np.logical_and(pred, ~true).sum())
    fn = int(np.logical_and(~pred, true).sum())
    tn = int(np.logical_and(~pred, ~true).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)
    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    return {"threshold": float(th), "precision": prec, "recall": rec, "f1": f1, "accuracy": acc, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def simple_auc(probs, labels):
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    true = labels > 0
    pos_scores = probs[true]
    neg_scores = probs[~true]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None
    total = 0
    good = 0.0
    for ps in pos_scores:
        cmp = ps - neg_scores
        good += float((cmp > 0).sum()) + 0.5 * float((cmp == 0).sum())
        total += len(neg_scores)
    return good / max(1, total)


def threshold_sweep(probs, labels, cfg: Cfg):
    ths = []
    t = cfg.threshold_min
    while t <= cfg.threshold_max + 1e-9:
        ths.append(round(t, 4))
        t += cfg.threshold_step
    rows = [binary_metrics_at_threshold(probs, labels, th) for th in ths]
    best_f1 = max(rows, key=lambda r: (r["f1"], r["precision"], r["recall"]))
    feasible = [r for r in rows if r["precision"] >= cfg.precision_target]
    best_at_precision = max(feasible, key=lambda r: (r["recall"], r["f1"])) if feasible else None
    return rows, best_f1, best_at_precision


def binary_head_values_from_logits(logit, label, mask):
    prob = torch.sigmoid(logit).detach().cpu().numpy().reshape(-1)
    lab = label.detach().cpu().numpy().reshape(-1) > 0.5
    m = mask.detach().cpu().numpy().reshape(-1).astype(bool)
    return prob[m].astype(np.float32), lab[m].astype(np.int64)


def binary_metrics_from_arrays(probs, labels, th=0.5):
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64) > 0
    if len(probs) == 0:
        return {
            "threshold": float(th), "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "accuracy": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "count": 0,
        }
    pred = probs >= th
    tp = int(np.logical_and(pred, labels).sum())
    fp = int(np.logical_and(pred, ~labels).sum())
    fn = int(np.logical_and(~pred, labels).sum())
    tn = int(np.logical_and(~pred, ~labels).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)
    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    return {
        "threshold": float(th), "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "accuracy": float(acc), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "count": int(len(probs)),
    }


def relation_threshold_sweep(probs, labels, cfg: Cfg):
    ths = []
    t = cfg.rel_threshold_min
    while t <= cfg.rel_threshold_max + 1e-9:
        ths.append(round(t, 4))
        t += cfg.rel_threshold_step
    rows = [binary_metrics_from_arrays(probs, labels, th) for th in ths]
    best_f1 = max(rows, key=lambda r: (r["f1"], r["precision"], r["recall"]))
    feasible = [r for r in rows if r["precision"] >= cfg.rel_precision_target]
    best_at_precision = max(feasible, key=lambda r: (r["recall"], r["f1"])) if feasible else None
    at_default = binary_metrics_from_arrays(probs, labels, cfg.rel_eval_threshold)
    return rows, best_f1, best_at_precision, at_default


@torch.no_grad()
def evaluate(model, loader, device, cfg, bce_quality, bce_rel):
    model.eval()
    sums = {"loss": 0.0, "quality": 0.0, "relation": 0.0}
    batches = 0
    all_probs = []
    all_labels = []
    e2e_probs, e2e_labels = [], []
    t_probs, t_labels = [], []
    x_probs, x_labels = [], []

    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch)
        _, parts = compute_loss(out, batch, cfg, bce_quality, bce_rel)
        for k in sums:
            sums[k] += float(parts[k].item())
        batches += 1
        prob = torch.sigmoid(out["quality_logit"]).detach().cpu().numpy()
        lab = batch["quality"].detach().cpu().numpy().astype(np.int64)
        all_probs.extend(prob.tolist())
        all_labels.extend(lab.tolist())
        p, y = binary_head_values_from_logits(out["e2e_logit"], batch["e2e_label"], batch["e2e_mask"])
        e2e_probs.extend(p.tolist())
        e2e_labels.extend(y.tolist())
        p, y = binary_head_values_from_logits(out["t_logit"], batch["t_label"], batch["t_mask"])
        t_probs.extend(p.tolist())
        t_labels.extend(y.tolist())
        p, y = binary_head_values_from_logits(out["x_logit"], batch["x_label"], batch["x_mask"])
        x_probs.extend(p.tolist())
        x_labels.extend(y.tolist())

    if batches == 0:
        return {"loss": math.inf}

    m = {k: sums[k] / batches for k in sums}
    rows, best_f1, best_at_precision = threshold_sweep(all_probs, all_labels, cfg)
    m05 = binary_metrics_at_threshold(all_probs, all_labels, 0.5)
    auc = simple_auc(all_probs, all_labels)

    m.update({
        "quality_auc": auc,
        "quality_at_0_5": m05,
        "quality_threshold_sweep": rows,
        "quality_best_f1": best_f1,
        "quality_best_at_precision_target": best_at_precision,
    })

    rel_inputs = {
        "E2E": (e2e_probs, e2e_labels),
        "T": (t_probs, t_labels),
        "X": (x_probs, x_labels),
    }
    rel_m = {}
    for name, (pp, yy) in rel_inputs.items():
        rows, best_f1, best_at_precision, at_default = relation_threshold_sweep(pp, yy, cfg)
        rel_m[name] = {
            "at_default": at_default,
            "best_f1": best_f1,
            "best_at_precision_target": best_at_precision,
            "threshold_sweep": rows,
        }
        # backward-compatible short fields use best_f1, not default threshold
        m[f"rel_{name}_precision"] = best_f1["precision"]
        m[f"rel_{name}_recall"] = best_f1["recall"]
        m[f"rel_{name}_f1"] = best_f1["f1"]
        m[f"rel_{name}_threshold"] = best_f1["threshold"]
        m[f"rel_{name}_tp"] = best_f1["tp"]
        m[f"rel_{name}_fp"] = best_f1["fp"]
        m[f"rel_{name}_fn"] = best_f1["fn"]
        m[f"rel_{name}_tn"] = best_f1["tn"]
        m[f"rel_{name}_count"] = best_f1["count"]
    m["rel_binary_metrics"] = rel_m
    model.train()
    return m


@torch.no_grad()
def predict_collect(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        tensor_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(tensor_batch)
        probs = torch.sigmoid(out["quality_logit"]).detach().cpu().numpy().tolist()
        labels = batch["quality"].detach().cpu().numpy().astype(int).tolist()
        for uid, src, grp, p, lab, sc in zip(
            batch["uid"], batch["source"], batch["group_key"], probs, labels,
            batch["stroke_count"].detach().cpu().numpy().astype(int).tolist()
        ):
            rows.append({"uid": uid, "source": src, "group_key": grp, "prob": float(p), "label": int(lab), "stroke_count": int(sc)})
    model.train()
    return rows


def print_metrics(epoch, train_loss, m, lr, cfg):
    auc_str = "None" if m.get("quality_auc") is None else f"{m['quality_auc']:.3f}"
    m05 = m["quality_at_0_5"]
    bf = m["quality_best_f1"]
    bp = m.get("quality_best_at_precision_target")

    print(
        f"Epoch {epoch:03d} | TrainLoss={train_loss:.4f} | ValLoss={m['loss']:.4f} | LR={lr:.2e}\n"
        f"  parts: quality={m['quality']:.4f} relation={m['relation']:.4f}\n"
        f"  quality@0.50 P/R/F1/Acc/AUC={m05['precision']:.3f}/"
        f"{m05['recall']:.3f}/{m05['f1']:.3f}/{m05['accuracy']:.3f}/{auc_str} "
        f"(tp={m05['tp']} fp={m05['fp']} fn={m05['fn']} tn={m05['tn']})\n"
        f"  quality bestF1 th={bf['threshold']:.2f} P/R/F1={bf['precision']:.3f}/"
        f"{bf['recall']:.3f}/{bf['f1']:.3f} "
        f"(tp={bf['tp']} fp={bf['fp']} fn={bf['fn']} tn={bf['tn']})"
    )
    if bp is not None:
        print(
            f"  quality P>={cfg.precision_target:.2f} bestRecall th={bp['threshold']:.2f} "
            f"P/R/F1={bp['precision']:.3f}/{bp['recall']:.3f}/{bp['f1']:.3f} "
            f"(tp={bp['tp']} fp={bp['fp']} fn={bp['fn']} tn={bp['tn']})"
        )
    else:
        print(f"  quality P>={cfg.precision_target:.2f}: no threshold reached target precision")

    print(
        f"  relation bestF1 E2E/T/X={m['rel_E2E_f1']:.3f}/"
        f"{m['rel_T_f1']:.3f}/{m['rel_X_f1']:.3f} "
        f"th={m['rel_E2E_threshold']:.2f}/{m['rel_T_threshold']:.2f}/{m['rel_X_threshold']:.2f}"
    )
    print(
        f"  relation bestF1 PR E2E={m['rel_E2E_precision']:.3f}/{m['rel_E2E_recall']:.3f} "
        f"T={m['rel_T_precision']:.3f}/{m['rel_T_recall']:.3f} "
        f"X={m['rel_X_precision']:.3f}/{m['rel_X_recall']:.3f}"
    )
    for rn in ["E2E", "T", "X"]:
        rbp = m["rel_binary_metrics"][rn]["best_at_precision_target"]
        if rbp is None:
            print(f"  relation {rn} P>={cfg.rel_precision_target:.2f}: no threshold reached target precision")
        else:
            print(
                f"  relation {rn} P>={cfg.rel_precision_target:.2f} bestRecall "
                f"th={rbp['threshold']:.2f} P/R/F1={rbp['precision']:.3f}/{rbp['recall']:.3f}/{rbp['f1']:.3f} "
                f"(tp={rbp['tp']} fp={rbp['fp']} fn={rbp['fn']} tn={rbp['tn']})"
            )
    top5 = sorted(m["quality_threshold_sweep"], key=lambda r: r["f1"], reverse=True)[:5]
    print("  threshold top5:", " | ".join(
        f"th={r['threshold']:.2f}:P={r['precision']:.3f},R={r['recall']:.3f},F1={r['f1']:.3f}"
        for r in top5
    ))



def node_desc_from_index(idx: int, stroke_count: int) -> Dict[str, Any]:
    idx = int(idx)
    stroke_count = int(stroke_count)
    if idx < stroke_count:
        return {"node_index": idx, "node_type": "stroke", "stroke_idx": idx, "endpoint": None}
    k = idx - stroke_count
    stroke_idx = k // 2
    ep = k % 2
    return {
        "node_index": idx,
        "node_type": "endpoint_start" if ep == 0 else "endpoint_end",
        "stroke_idx": int(stroke_idx),
        "endpoint": "start" if ep == 0 else "end",
    }


@torch.no_grad()
def collect_relation_errors(model, loader, device, cfg: Cfg, rel_metrics: Dict[str, Any], topk: int = 100):
    model.eval()
    rows = []

    # use best-f1 threshold for each relation head
    thresholds = {
        "E2E": float(rel_metrics["E2E"]["best_f1"]["threshold"]),
        "T": float(rel_metrics["T"]["best_f1"]["threshold"]),
        "X": float(rel_metrics["X"]["best_f1"]["threshold"]),
    }

    for batch in loader:
        tensor_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(tensor_batch)

        batch_size = tensor_batch["quality"].shape[0]
        for b in range(batch_size):
            uid = batch["uid"][b]
            source = batch["source"][b]
            group_key = batch["group_key"][b]
            sc = int(batch["stroke_count"][b].detach().cpu())

            for head, logit_key, label_key, mask_key in [
                ("E2E", "e2e_logit", "e2e_label", "e2e_mask"),
                ("T", "t_logit", "t_label", "t_mask"),
                ("X", "x_logit", "x_label", "x_mask"),
            ]:
                prob = torch.sigmoid(out[logit_key][b]).detach().cpu().numpy()
                lab = batch[label_key][b].detach().cpu().numpy() > 0.5
                msk = batch[mask_key][b].detach().cpu().numpy().astype(bool)
                th = thresholds[head]
                pred = prob >= th

                fp_idx = np.argwhere(msk & pred & (~lab))
                fn_idx = np.argwhere(msk & (~pred) & lab)

                for i, j in fp_idx[:topk]:
                    rows.append({
                        "head": head,
                        "error_type": "FP",
                        "uid": uid,
                        "source": source,
                        "group_key": group_key,
                        "stroke_count": sc,
                        "i": int(i),
                        "j": int(j),
                        "node_i": node_desc_from_index(int(i), sc),
                        "node_j": node_desc_from_index(int(j), sc),
                        "prob": float(prob[i, j]),
                        "label": 0,
                        "threshold": th,
                    })
                for i, j in fn_idx[:topk]:
                    rows.append({
                        "head": head,
                        "error_type": "FN",
                        "uid": uid,
                        "source": source,
                        "group_key": group_key,
                        "stroke_count": sc,
                        "i": int(i),
                        "j": int(j),
                        "node_i": node_desc_from_index(int(i), sc),
                        "node_j": node_desc_from_index(int(j), sc),
                        "prob": float(prob[i, j]),
                        "label": 1,
                        "threshold": th,
                    })

    # prioritize the most confident false positives and least confident false negatives
    fp = [r for r in rows if r["error_type"] == "FP"]
    fn = [r for r in rows if r["error_type"] == "FN"]
    fp.sort(key=lambda r: r["prob"], reverse=True)
    fn.sort(key=lambda r: r["prob"])
    out = {}
    for head in ["E2E", "T", "X"]:
        out[f"{head}_false_positives"] = [r for r in fp if r["head"] == head][:topk]
        out[f"{head}_false_negatives"] = [r for r in fn if r["head"] == head][:topk]
        out[f"{head}_fp_count_collected"] = sum(1 for r in fp if r["head"] == head)
        out[f"{head}_fn_count_collected"] = sum(1 for r in fn if r["head"] == head)
    return out


# =============================================================================
# 9. Error export
# =============================================================================

def export_error_analysis(model, ds: FlywheelTopoDataset, val_loader, device, cfg: Cfg, m: Dict[str, Any], tag: str):
    pred_rows = predict_collect(model, val_loader, device)
    relation_errors = collect_relation_errors(model, val_loader, device, cfg, m["rel_binary_metrics"], topk=cfg.export_error_topk)
    best_th = float(m["quality_best_f1"]["threshold"])
    by_uid_record = {r["uid"]: r for r in ds.records}

    for row in pred_rows:
        row["pred"] = int(row["prob"] >= best_th)
        row["threshold"] = best_th
        row["error_type"] = None
        if row["pred"] == 1 and row["label"] == 0:
            row["error_type"] = "FP_pred_good_but_bad"
        elif row["pred"] == 0 and row["label"] == 1:
            row["error_type"] = "FN_pred_bad_but_good"
        rec = by_uid_record.get(row["uid"])
        if rec:
            row["file"] = rec.get("file")
            row["outer_key"] = rec.get("outer_key")
            row["bundle_info"] = json_slim_bundle_info(rec.get("bundle", {}))

    fp = [r for r in pred_rows if r["error_type"] == "FP_pred_good_but_bad"]
    fn = [r for r in pred_rows if r["error_type"] == "FN_pred_bad_but_good"]
    fp.sort(key=lambda r: r["prob"], reverse=True)
    fn.sort(key=lambda r: r["prob"])

    top_good = sorted(pred_rows, key=lambda r: r["prob"], reverse=True)[:cfg.export_error_topk]
    bottom_good = sorted(pred_rows, key=lambda r: r["prob"])[:cfg.export_error_topk]

    out = {
        "created_at": now_str(),
        "tag": tag,
        "best_threshold": best_th,
        "metrics": m,
        "false_positive_count": len(fp),
        "false_negative_count": len(fn),
        "false_positives": fp[:cfg.export_error_topk],
        "false_negatives": fn[:cfg.export_error_topk],
        "top_good_scores": top_good,
        "bottom_good_scores": bottom_good,
        "relation_errors": relation_errors,
    }

    out_path = os.path.join(MODEL_SAVE_DIR, f"{cfg.save_prefix}_{tag}_error_analysis.json")
    save_json(out, out_path)
    print(f"  🧪 error-analysis saved: {out_path}")
    return out_path


# =============================================================================
# 10. Train
# =============================================================================

def train():
    cfg = Cfg()
    seed_everything(cfg.seed)

    annotations_dir = os.environ.get("FLYWHEEL_ANNOTATIONS_TOPO_DIR", DEFAULT_ANNOTATIONS_TOPO_DIR)
    pool_root = os.environ.get("FLYWHEEL_PCG_POOL_ROOT", DEFAULT_PCG_POOL_ROOT)

    print("=" * 88)
    print("[FlywheelTopoHGT-v6-relationcalib-ckptselect] start")
    print(f"[Time] {now_str()}")
    print(f"[ScriptDir] {SCRIPT_DIR}")
    print(f"[ModelSaveDir] {MODEL_SAVE_DIR}")
    print(f"[CharGlyphDir] {CHAR_GLYPH_DIR}")
    print(f"[DatasetAnalyseDir] {DATASET_ANALYSE_DIR}")
    print(f"[AnnotationsTopoDir] {annotations_dir}")
    print(f"[PCGPoolRoot] {pool_root}")
    print(f"[PIL_AVAILABLE] {PIL_AVAILABLE}")
    print(f"[Config] {json.dumps(asdict(cfg), ensure_ascii=False, indent=2)}")
    print("=" * 88)

    records, load_stats = load_source_records(cfg, annotations_dir, pool_root)
    if not records:
        print("[ERROR] no usable records.")
        return

    manifest_path = os.path.join(MODEL_SAVE_DIR, f"{cfg.save_prefix}_data_manifest.json")
    manifest = {
        "created_at": now_str(),
        "script": os.path.abspath(__file__),
        "annotations_dir": annotations_dir,
        "pool_root": pool_root,
        "model_save_dir": MODEL_SAVE_DIR,
        "config": asdict(cfg),
        "load_stats": load_stats,
        "records": [
            {
                "uid": r["uid"],
                "source": r["source"],
                "quality_label": int(r["quality_label"]),
                "stroke_count": int(r["stroke_count"]),
                "group_key": r["group_key"],
                "file": r["file"],
                "outer_key": r["outer_key"],
            }
            for r in records
        ],
    }
    save_json(manifest, manifest_path)
    print(f"[Manifest] saved: {manifest_path}")

    ds = FlywheelTopoDataset(records, cfg)
    train_set, val_set, split_info = split_dataset(ds, cfg)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=False, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)

    device = get_device()
    print(f"[Device] {device}")

    model = TopoHGTv6(ds.node_dim, ds.edge_dim, cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] TopoHGTv6 node_dim={ds.node_dim} edge_dim={ds.edge_dim} params={params:,} trainable={trainable:,}")

    q_pos_weight_val = quality_pos_weight(ds.quality_counts, cfg)
    bce_quality = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(q_pos_weight_val, dtype=torch.float32, device=device))

    rel_pos_w = relation_pos_weights(ds.binary_rel_counts, cfg)
    bce_rel = {
        "e2e": nn.BCEWithLogitsLoss(pos_weight=torch.tensor(rel_pos_w["e2e"], dtype=torch.float32, device=device), reduction="none"),
        "t":   nn.BCEWithLogitsLoss(pos_weight=torch.tensor(rel_pos_w["t"], dtype=torch.float32, device=device), reduction="none"),
        "x":   nn.BCEWithLogitsLoss(pos_weight=torch.tensor(rel_pos_w["x"], dtype=torch.float32, device=device), reduction="none"),
    }

    print(f"[Loss] quality_pos_weight={q_pos_weight_val:.4f}")
    print(f"[Loss] use_manual_rel_pos_weight={cfg.use_manual_rel_pos_weight}")
    print("[Loss] relation_binary_pos_weights:", {k: round(float(v), 4) for k, v in rel_pos_w.items()})

    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))

    latest_path = os.path.join(MODEL_SAVE_DIR, f"{cfg.save_prefix}_latest.pth")

    ckpt_paths = {
        "latest": os.path.join(MODEL_SAVE_DIR, f"{cfg.save_prefix}_latest.pth"),
        "loss": os.path.join(MODEL_SAVE_DIR, f"{cfg.save_prefix}_best_loss.pth"),
        "quality_balanced": os.path.join(MODEL_SAVE_DIR, f"{cfg.save_prefix}_best_quality_f1.pth"),
        "quality_conservative": os.path.join(MODEL_SAVE_DIR, f"{cfg.save_prefix}_best_quality_precision{int(cfg.precision_target * 100)}_recall.pth"),
        "relation_overall": os.path.join(MODEL_SAVE_DIR, f"{cfg.save_prefix}_best_relation_macro_f1.pth"),
        "relation_e2e": os.path.join(MODEL_SAVE_DIR, f"{cfg.save_prefix}_best_relation_E2E_f1.pth"),
        "relation_t": os.path.join(MODEL_SAVE_DIR, f"{cfg.save_prefix}_best_relation_T_f1.pth"),
        "relation_x": os.path.join(MODEL_SAVE_DIR, f"{cfg.save_prefix}_best_relation_X_f1.pth"),
    }
    checkpoint_index_path = os.path.join(MODEL_SAVE_DIR, f"{cfg.save_prefix}_checkpoint_index.json")

    best_scores = {
        "loss": math.inf,
        "quality_balanced": -math.inf,
        "quality_conservative": -math.inf,
        "relation_overall": -math.inf,
        "relation_e2e": -math.inf,
        "relation_t": -math.inf,
        "relation_x": -math.inf,
    }

    checkpoint_index = {
        "created_at": now_str(),
        "updated_at": now_str(),
        "script": os.path.abspath(__file__),
        "model_save_dir": MODEL_SAVE_DIR,
        "checkpoint_paths": ckpt_paths,
        "recommendations": {},
        "history": [],
        "selection_rules": {
            "quality_balanced": "max val_metrics.quality_best_f1.f1",
            "quality_conservative": "max val_metrics.quality_best_at_precision_target.recall, when precision>=precision_target",
            "relation_overall": "max mean(E2E_bestF1, T_bestF1, X_bestF1)",
            "relation_e2e": "max E2E bestF1",
            "relation_t": "max T bestF1",
            "relation_x": "max X bestF1",
            "loss": "min val loss",
            "latest": "latest evaluated checkpoint",
        },
        "config": asdict(cfg),
    }

    best_metric_for_stop = -math.inf if cfg.early_stop_metric == "quality_best_f1" else math.inf
    bad_epochs = 0

    def metric_summary(m: Dict[str, Any]) -> Dict[str, Any]:
        qhp = m.get("quality_best_at_precision_target")
        rel = m.get("rel_binary_metrics", {})
        out = {
            "loss": float(m.get("loss", math.inf)),
            "quality_best_f1": {
                "score": float(m["quality_best_f1"]["f1"]),
                "threshold": float(m["quality_best_f1"]["threshold"]),
                "precision": float(m["quality_best_f1"]["precision"]),
                "recall": float(m["quality_best_f1"]["recall"]),
                "tp": int(m["quality_best_f1"]["tp"]),
                "fp": int(m["quality_best_f1"]["fp"]),
                "fn": int(m["quality_best_f1"]["fn"]),
                "tn": int(m["quality_best_f1"]["tn"]),
            },
            "quality_precision_target": None if qhp is None else {
                "score": float(qhp["recall"]),
                "threshold": float(qhp["threshold"]),
                "precision": float(qhp["precision"]),
                "recall": float(qhp["recall"]),
                "f1": float(qhp["f1"]),
                "tp": int(qhp["tp"]),
                "fp": int(qhp["fp"]),
                "fn": int(qhp["fn"]),
                "tn": int(qhp["tn"]),
            },
            "relation": {},
        }
        rel_f1s = []
        for rn in ["E2E", "T", "X"]:
            best = rel.get(rn, {}).get("best_f1", {})
            hp = rel.get(rn, {}).get("best_at_precision_target")
            if best:
                rel_f1s.append(float(best["f1"]))
                out["relation"][rn] = {
                    "best_f1": {
                        "score": float(best["f1"]),
                        "threshold": float(best["threshold"]),
                        "precision": float(best["precision"]),
                        "recall": float(best["recall"]),
                        "tp": int(best["tp"]),
                        "fp": int(best["fp"]),
                        "fn": int(best["fn"]),
                        "tn": int(best["tn"]),
                    },
                    "precision_target": None if hp is None else {
                        "score": float(hp["recall"]),
                        "threshold": float(hp["threshold"]),
                        "precision": float(hp["precision"]),
                        "recall": float(hp["recall"]),
                        "f1": float(hp["f1"]),
                        "tp": int(hp["tp"]),
                        "fp": int(hp["fp"]),
                        "fn": int(hp["fn"]),
                        "tn": int(hp["tn"]),
                    },
                }
        out["relation_macro_f1"] = float(sum(rel_f1s) / max(1, len(rel_f1s)))
        return out

    def make_recommendation_record(tag: str, path: str, epoch: int, train_loss: float, score: float, m: Dict[str, Any]) -> Dict[str, Any]:
        ms = metric_summary(m)
        rec = {
            "purpose": tag,
            "path": path,
            "epoch": int(epoch),
            "score": float(score),
            "train_loss": float(train_loss),
            "saved_at": now_str(),
            "metrics_summary": ms,
            "thresholds": {
                "quality_best_f1": ms["quality_best_f1"]["threshold"],
                "quality_precision_target": None if ms["quality_precision_target"] is None else ms["quality_precision_target"]["threshold"],
                "relation_E2E_best_f1": ms["relation"].get("E2E", {}).get("best_f1", {}).get("threshold"),
                "relation_T_best_f1": ms["relation"].get("T", {}).get("best_f1", {}).get("threshold"),
                "relation_X_best_f1": ms["relation"].get("X", {}).get("best_f1", {}).get("threshold"),
                "relation_E2E_precision_target": None if ms["relation"].get("E2E", {}).get("precision_target") is None else ms["relation"]["E2E"]["precision_target"]["threshold"],
                "relation_T_precision_target": None if ms["relation"].get("T", {}).get("precision_target") is None else ms["relation"]["T"]["precision_target"]["threshold"],
                "relation_X_precision_target": None if ms["relation"].get("X", {}).get("precision_target") is None else ms["relation"]["X"]["precision_target"]["threshold"],
            },
        }
        return rec

    def save_checkpoint_index():
        checkpoint_index["updated_at"] = now_str()
        save_json(checkpoint_index, checkpoint_index_path)

    def save_selected_checkpoint(tag: str, path: str, score: float, epoch: int, train_loss: float, m: Dict[str, Any], ckpt: Dict[str, Any], export_errors: bool = False):
        ckpt = dict(ckpt)
        ckpt["selection_purpose"] = tag
        ckpt["selection_score"] = float(score)
        ckpt["metrics_summary"] = metric_summary(m)
        torch.save(ckpt, path)

        rec = make_recommendation_record(tag, path, epoch, train_loss, score, m)
        checkpoint_index["recommendations"][tag] = rec
        checkpoint_index["history"].append(rec)
        save_checkpoint_index()

        if export_errors:
            export_error_analysis(model, ds, val_loader, device, cfg, m, tag=tag)

        print(f"  ✅ checkpoint saved [{tag}]: {path} | score={score:.6f}")

    def make_ckpt(epoch, train_loss, m):
        return {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "node_dim": ds.node_dim,
            "edge_dim": ds.edge_dim,
            "max_nodes": ds.max_nodes,
            "node_type": NODE_TYPE,
            "rel_map": REL_MAP,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_metrics": m,
            "metrics_summary": metric_summary(m),
            "split_info": split_info,
            "data_manifest_path": manifest_path,
            "checkpoint_index_path": checkpoint_index_path,
            "quality_counts": ds.quality_counts.tolist(),
            "relation_counts": ds.rel_counts.tolist(),
            "binary_relation_counts": ds.binary_rel_counts,
            "quality_pos_weight": q_pos_weight_val,
            "relation_binary_pos_weights": rel_pos_w,
            "created_at": now_str(),
        }

    print("[Train] start...")
    for epoch in range(1, cfg.epochs + 1):
        epoch_t0 = time.perf_counter()
        model.train()
        total_loss = 0.0
        batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            out = model(batch)
            loss, _ = compute_loss(out, batch, cfg, bce_quality, bce_rel)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total_loss += float(loss.item())
            batches += 1

        sched.step()

        do_eval = (epoch == 1) or (epoch % cfg.log_every == 0) or (epoch == cfg.epochs)
        if not do_eval:
            if epoch == 1 or epoch % 2 == 0:
                print(f"[TrainTiming] epoch={epoch:03d} train_only_time={time.perf_counter()-epoch_t0:.3f}s batches={batches}")
            continue

        train_loss = total_loss / max(1, batches)
        m = evaluate(model, val_loader, device, cfg, bce_quality, bce_rel)
        lr = sched.get_last_lr()[0]
        print_metrics(epoch, train_loss, m, lr, cfg)
        print(f"  [TrainTiming] epoch_total_time={time.perf_counter()-epoch_t0:.3f}s batches={batches}")

        ckpt = make_ckpt(epoch, train_loss, m)

        # Always save latest and update index.
        torch.save(ckpt, ckpt_paths["latest"])
        checkpoint_index["recommendations"]["latest"] = make_recommendation_record(
            "latest", ckpt_paths["latest"], epoch, train_loss, float(m["quality_best_f1"]["f1"]), m
        )
        save_checkpoint_index()

        # Compute selection scores.
        cur_loss = float(m["loss"])
        cur_f1 = float(m["quality_best_f1"]["f1"])

        qhp = m.get("quality_best_at_precision_target")
        cur_quality_conservative = float(qhp["recall"]) if qhp is not None else -math.inf

        cur_relation_e2e = float(m["rel_E2E_f1"])
        cur_relation_t = float(m["rel_T_f1"])
        cur_relation_x = float(m["rel_X_f1"])
        cur_relation_macro = float((cur_relation_e2e + cur_relation_t + cur_relation_x) / 3.0)

        # Save best checkpoints by purpose.
        if cur_loss < best_scores["loss"]:
            best_scores["loss"] = cur_loss
            save_selected_checkpoint("loss", ckpt_paths["loss"], cur_loss, epoch, train_loss, m, ckpt, export_errors=True)

        if cur_f1 > best_scores["quality_balanced"]:
            best_scores["quality_balanced"] = cur_f1
            save_selected_checkpoint("quality_balanced", ckpt_paths["quality_balanced"], cur_f1, epoch, train_loss, m, ckpt, export_errors=True)

        if cur_quality_conservative > best_scores["quality_conservative"]:
            best_scores["quality_conservative"] = cur_quality_conservative
            save_selected_checkpoint("quality_conservative", ckpt_paths["quality_conservative"], cur_quality_conservative, epoch, train_loss, m, ckpt, export_errors=False)

        if cur_relation_macro > best_scores["relation_overall"]:
            best_scores["relation_overall"] = cur_relation_macro
            save_selected_checkpoint("relation_overall", ckpt_paths["relation_overall"], cur_relation_macro, epoch, train_loss, m, ckpt, export_errors=True)

        if cur_relation_e2e > best_scores["relation_e2e"]:
            best_scores["relation_e2e"] = cur_relation_e2e
            save_selected_checkpoint("relation_e2e", ckpt_paths["relation_e2e"], cur_relation_e2e, epoch, train_loss, m, ckpt, export_errors=False)

        if cur_relation_t > best_scores["relation_t"]:
            best_scores["relation_t"] = cur_relation_t
            save_selected_checkpoint("relation_t", ckpt_paths["relation_t"], cur_relation_t, epoch, train_loss, m, ckpt, export_errors=True)

        if cur_relation_x > best_scores["relation_x"]:
            best_scores["relation_x"] = cur_relation_x
            save_selected_checkpoint("relation_x", ckpt_paths["relation_x"], cur_relation_x, epoch, train_loss, m, ckpt, export_errors=True)

        if cfg.early_stop_metric == "quality_best_f1":
            cur_metric = cur_f1
            is_better = cur_metric > best_metric_for_stop + 1e-6
            if is_better:
                best_metric_for_stop = cur_metric
        else:
            cur_metric = m["loss"]
            is_better = cur_metric < best_metric_for_stop - 1e-6
            if is_better:
                best_metric_for_stop = cur_metric

        if is_better:
            bad_epochs = 0
        else:
            bad_epochs += cfg.log_every

        print(f"  early_stop metric={cfg.early_stop_metric} best={best_metric_for_stop:.6f} bad_epochs={bad_epochs}/{cfg.early_stop_patience}")
        if bad_epochs >= cfg.early_stop_patience:
            print(f"[EarlyStop] stop at epoch={epoch}, bad_epochs={bad_epochs}")
            break

    print("[Train] done.")
    print("[BestScores]", {k: (round(v, 6) if math.isfinite(v) else None) for k, v in best_scores.items()})
    print(f"latest:           {ckpt_paths['latest']}")
    print(f"best_loss:        {ckpt_paths['loss']}")
    print(f"best_quality:     {ckpt_paths['quality_balanced']}")
    print(f"best_quality_hp:  {ckpt_paths['quality_conservative']}")
    print(f"best_rel_macro:   {ckpt_paths['relation_overall']}")
    print(f"best_rel_E2E:     {ckpt_paths['relation_e2e']}")
    print(f"best_rel_T:       {ckpt_paths['relation_t']}")
    print(f"best_rel_X:       {ckpt_paths['relation_x']}")
    print(f"checkpoint_index: {checkpoint_index_path}")
    print(f"manifest:         {manifest_path}")
    print(f"model_dir:        {MODEL_SAVE_DIR}")

    print("[AutoSelect] import this script and call:")
    print(f"  choose_checkpoint_from_index(r'{checkpoint_index_path}', purpose='quality_balanced')")


if __name__ == "__main__":
    train()
