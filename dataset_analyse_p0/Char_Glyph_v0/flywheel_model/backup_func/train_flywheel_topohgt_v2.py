# -*- coding: utf-8 -*-
r"""
train_flywheel_topohgt_v2.py

建议放置位置：
    dataset_analyse_p0/Char_Glyph_v0/flywheel_model/train_flywheel_topohgt_v2.py

模型保存位置：
    dataset_analyse_p0/Char_Glyph_v0/flywheel_model/model/

V2 相比 V1 的改动：
    1. 模型文件统一保存到脚本目录下的 model/ 文件夹。
    2. 默认使用 no-leak relation 特征：
       - relation head 不再从 topology_events 派生的 degree / in_cycle 特征中“看答案”。
       - relation label 仍来自 topology_events，但输入只使用几何特征。
    3. 使用 group split，按 candidate_id/style_mode/topology_family/source 做分组，降低同源泄漏。
    4. 加 early stopping。
    5. 加 quality threshold sweep，保存 best threshold。
    6. 导出 FP/FN/Top/Bottom 样本，便于人工排查错误。
    7. 保留关键数据读取、训练、保存日志。

读取数据：
    dataset_analyse_p0/AI_VECTOR_ROUTER_With_topo/annotations_topo
    dataset_analyse_p0/Char_Glyph_v0/annotation_tool/pcg_filebacked_stage2_schema/bad
    dataset_analyse_p0/Char_Glyph_v0/annotation_tool/pcg_filebacked_stage2_schema/good
    dataset_analyse_p0/Char_Glyph_v0/annotation_tool/pcg_filebacked_stage2_schema/cleaned

去重：
    使用 annotation_tool 的 stable_uid 思路：
      candidate_id + style_mode + topology_family + rounded strokes geometry
    source priority:
      cleaned > good > annotations_topo > bad
"""

import os
import json
import glob
import math
import time
import copy
import random
import hashlib
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset


# =============================================================================
# 0. Config
# =============================================================================

@dataclass
class Cfg:
    # data
    max_strokes: int = 32
    canvas_norm: float = 400.0
    width_norm: float = 20.0
    dist_norm: float = 32.0
    curve_sample_n: int = 32

    # no-leak relation input
    # False = 不把 topology_events 派生的 e2e/t/x degree、in_cycle 放进 node feature。
    # True  = debug 模式，会让 relation head 明显变容易，但有标签泄漏嫌疑。
    use_topology_leak_features: bool = False

    # split
    val_ratio: float = 0.12
    group_split: bool = True
    seed: int = 42

    # model
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.10

    # training
    batch_size: int = 512
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stop_patience: int = 12
    early_stop_metric: str = "quality_best_f1"  # "quality_best_f1" or "loss"

    # loss
    w_quality: float = 2.0
    w_relation: float = 0.7
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
    save_prefix: str = "flywheel_topohgt_v2"
    export_error_topk: int = 80

    # source labels
    annotations_are_positive: bool = True


REL_MAP = {"NONE": 0, "E2E": 1, "T": 2, "X": 3}
REL_INV = {v: k for k, v in REL_MAP.items()}
IGNORE = -100

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


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def percentile(xs, ps=(0, 50, 90, 95, 99, 100)):
    if not xs:
        return {str(p): None for p in ps}
    arr = np.asarray(xs, dtype=np.float32)
    return {str(p): round(float(np.percentile(arr, p)), 4) for p in ps}


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
# 2. stable_uid：复制 annotation_tool 的几何 sha256 思路
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
        raw_id = s.get("bezier_id", i + 1)
        try:
            bid = int(raw_id)
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

    # fallback: source + geometry-independent-ish short group
    # 不用 uid 直接分组太细；这里用 stroke_count/topology_family 空时只能 fallback uid。
    return "uid:" + uid


# =============================================================================
# 3. Geometry helpers
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


def stroke_node_feature(
    s: Dict[str, Any],
    idx: int,
    n: int,
    cfg: Cfg,
    topo_deg=None,
    in_cycle=False,
) -> np.ndarray:
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
    closed = 1.0 if chord_len < 2.0 else 0.0

    dif = np.diff(curve, axis=0)
    dirs = dif / (np.linalg.norm(dif, axis=1, keepdims=True) + 1e-6)
    if len(dirs) > 1:
        dots = np.clip(np.sum(dirs[:-1] * dirs[1:], axis=1), -1, 1)
        curvature = float(np.mean(np.arccos(dots))) / math.pi
    else:
        curvature = 0.0

    base = [
        P.reshape(-1) / cfg.canvas_norm,                  # 8
        W / cfg.width_norm,                               # 4
        center / cfg.canvas_norm,                         # 2
        mn / cfg.canvas_norm, mx / cfg.canvas_norm,        # 4
        np.array([
            length / cfg.canvas_norm,
            chord_len / cfg.canvas_norm,
            float(np.mean(W)) / cfg.width_norm,
            closed,
            curvature,
            idx / max(1, cfg.max_strokes - 1),
        ], dtype=np.float32),                              # 6
        t0, t1,                                            # 4
    ]

    if cfg.use_topology_leak_features:
        topo_deg = topo_deg or {}
        base.append(np.array([
            topo_deg.get("e2e", 0) / 4.0,
            topo_deg.get("t_guest", 0) / 4.0,
            topo_deg.get("t_host", 0) / 4.0,
            topo_deg.get("x", 0) / 4.0,
            1.0 if in_cycle else 0.0,
        ], dtype=np.float32))                              # +5

    return np.concatenate(base).astype(np.float32)


def pair_edge_feature(sa: Dict[str, Any], sb: Dict[str, Any], cfg: Cfg) -> np.ndarray:
    Pa = np.asarray(sa["mother_bezier"], dtype=np.float32)
    Pb = np.asarray(sb["mother_bezier"], dtype=np.float32)
    Wa = width_ctrl_from_stroke(sa)
    Wb = width_ctrl_from_stroke(sb)

    endpoints_a = [Pa[0], Pa[3]]
    endpoints_b = [Pb[0], Pb[3]]
    dmat = np.zeros((2, 2), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            dmat[i, j] = float(np.linalg.norm(endpoints_a[i] - endpoints_b[j]))
    min_ep = float(dmat.min())
    min_idx = int(dmat.argmin())
    ep_onehot = np.zeros(4, dtype=np.float32)
    ep_onehot[min_idx] = 1.0

    ca = Pa.mean(axis=0)
    cb = Pb.mean(axis=0)
    center_delta = cb - ca
    center_dist = float(np.linalg.norm(center_delta))

    chord_a = norm_vec(Pa[3] - Pa[0])
    chord_b = norm_vec(Pb[3] - Pb[0])
    angle_cos = float(np.clip(np.dot(chord_a, chord_b), -1, 1))
    angle_sin_abs = float(abs(chord_a[0] * chord_b[1] - chord_a[1] * chord_b[0]))

    ts = np.linspace(0, 1, 16, dtype=np.float32)
    ca_curve = cubic(Pa, ts)
    cb_curve = cubic(Pb, ts)
    dd = np.linalg.norm(ca_curve[:, None, :] - cb_curve[None, :, :], axis=2)
    min_curve_dist = float(dd.min())

    feat = np.concatenate([
        np.array([
            min_ep / cfg.dist_norm,
            center_dist / cfg.canvas_norm,
            center_delta[0] / cfg.canvas_norm,
            center_delta[1] / cfg.canvas_norm,
            angle_cos,
            angle_sin_abs,
            min_curve_dist / cfg.dist_norm,
            float(np.mean(Wa)) / cfg.width_norm,
            float(np.mean(Wb)) / cfg.width_norm,
            abs(float(np.mean(Wa)) - float(np.mean(Wb))) / cfg.width_norm,
        ], dtype=np.float32),
        ep_onehot,
    ]).astype(np.float32)
    return feat


def bezier_id_map(strokes: List[Dict[str, Any]]) -> Dict[int, int]:
    m = {}
    for i, s in enumerate(strokes):
        try:
            bid = int(s.get("bezier_id", i + 1))
        except Exception:
            bid = i + 1
        m[bid] = i
    return m


def relation_labels_and_degrees(bundle: Dict[str, Any], max_strokes: int) -> Tuple[np.ndarray, Dict[int, Dict[str, int]], set, Dict[str, int]]:
    strokes = bundle.get("strokes", [])
    n = len(strokes)
    idmap = bezier_id_map(strokes)
    rel = np.zeros((n, n), dtype=np.int64)
    deg = {i: {"e2e": 0, "t_guest": 0, "t_host": 0, "x": 0} for i in range(n)}
    counts = {"NONE": 0, "E2E": 0, "T": 0, "X": 0}
    in_cycle = set()

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
                        a, b = idmap[a_id], idmap[b_id]
                        rel[a, b] = REL_MAP["E2E"]
                        rel[b, a] = REL_MAP["E2E"]
                        deg[a]["e2e"] += 1
                        deg[b]["e2e"] += 1
                elif typ == "X":
                    a_id, b_id = int(ev["stroke_a"]), int(ev["stroke_b"])
                    if a_id in idmap and b_id in idmap:
                        a, b = idmap[a_id], idmap[b_id]
                        rel[a, b] = REL_MAP["X"]
                        rel[b, a] = REL_MAP["X"]
                        deg[a]["x"] += 1
                        deg[b]["x"] += 1
                elif typ == "T":
                    g_id, h_id = int(ev["guest"]), int(ev["host"])
                    if g_id in idmap and h_id in idmap:
                        g, h = idmap[g_id], idmap[h_id]
                        rel[g, h] = REL_MAP["T"]
                        deg[g]["t_guest"] += 1
                        deg[h]["t_host"] += 1
            except Exception:
                continue

    cycles = bundle.get("cycles", [])
    if isinstance(cycles, list):
        for cyc in cycles:
            if not isinstance(cyc, dict):
                continue
            members = cyc.get("members", [])
            if isinstance(members, list):
                for bid in members:
                    try:
                        bid = int(bid)
                        if bid in idmap:
                            in_cycle.add(idmap[bid])
                    except Exception:
                        pass

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            counts[REL_INV[int(rel[i, j])]] += 1

    return rel, deg, in_cycle, counts


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
    stats = {
        "source_dirs": {},
        "skip_reasons": {},
        "duplicates": 0,
        "conflicts": [],
    }

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

    if label_counts[0] == 0:
        print("[WARN] 没有 bad/negative 样本，quality head 只能学 positive，F1 没有实际意义。")
    if label_counts[1] == 0:
        print("[WARN] 没有 good/positive 样本，quality head 无法正常训练。")

    return records, stats


# =============================================================================
# 5. Dataset
# =============================================================================

class FlywheelTopoDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], cfg: Cfg):
        self.records = records
        self.cfg = cfg

        item0 = self._make_item(0)
        self.node_dim = item0["node_features"].shape[-1]
        self.edge_dim = item0["edge_features"].shape[-1]

        self.rel_counts = np.zeros(4, dtype=np.int64)
        self.quality_counts = np.zeros(2, dtype=np.int64)
        self.stroke_counts = []

        for r in self.records:
            self.quality_counts[int(r["quality_label"])] += 1
            self.stroke_counts.append(int(r["stroke_count"]))
            rel, _, _, _ = relation_labels_and_degrees(r["bundle"], self.cfg.max_strokes)
            n = rel.shape[0]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        self.rel_counts[int(rel[i, j])] += 1

        print(f"[Dataset] samples={len(self.records)} node_dim={self.node_dim} edge_dim={self.edge_dim}")
        print(f"[Dataset] use_topology_leak_features={self.cfg.use_topology_leak_features}")
        print("[Dataset] quality_counts:", {str(k): int(v) for k, v in enumerate(self.quality_counts)})
        print("[Dataset] rel_counts:", {REL_INV[i]: int(self.rel_counts[i]) for i in range(4)})
        print("[Dataset] stroke_count:", percentile(self.stroke_counts))

    def __len__(self):
        return len(self.records)

    def _make_item(self, idx):
        cfg = self.cfg
        r = self.records[idx]
        bundle = r["bundle"]
        strokes = bundle["strokes"]
        n = len(strokes)
        S = cfg.max_strokes

        rel_raw, topo_deg, in_cycle, _ = relation_labels_and_degrees(bundle, cfg.max_strokes)

        node_list = []
        for i, s in enumerate(strokes):
            node_list.append(stroke_node_feature(s, i, n, cfg, topo_deg=topo_deg.get(i, {}), in_cycle=(i in in_cycle)))
        node_features_valid = np.asarray(node_list, dtype=np.float32)
        node_dim = node_features_valid.shape[-1]
        node_features = np.zeros((S, node_dim), dtype=np.float32)
        node_features[:n] = node_features_valid

        zero_edge = pair_edge_feature(strokes[0], strokes[0], cfg)
        edge_features = np.zeros((S, S, zero_edge.shape[-1]), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i == j:
                    edge_features[i, j] = 0.0
                else:
                    edge_features[i, j] = pair_edge_feature(strokes[i], strokes[j], cfg)

        rel_labels = np.full((S, S), IGNORE, dtype=np.int64)
        rel_labels[:n, :n] = rel_raw
        for i in range(S):
            rel_labels[i, i] = IGNORE
        if n < S:
            rel_labels[n:, :] = IGNORE
            rel_labels[:, n:] = IGNORE

        node_mask = np.zeros((S,), dtype=np.float32)
        node_mask[:n] = 1.0

        return {
            "uid": r["uid"],
            "source": r["source"],
            "group_key": r["group_key"],
            "node_features": torch.tensor(node_features, dtype=torch.float32),
            "edge_features": torch.tensor(edge_features, dtype=torch.float32),
            "node_mask": torch.tensor(node_mask, dtype=torch.float32),
            "rel_labels": torch.tensor(rel_labels, dtype=torch.long),
            "quality": torch.tensor(float(r["quality_label"]), dtype=torch.float32),
            "stroke_count": torch.tensor(n, dtype=torch.long),
        }

    def __getitem__(self, idx):
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

        pos_groups = []
        neg_groups = []
        mixed_groups = []
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

        val_groups = set()
        val_groups |= take_val(pos_groups)
        val_groups |= take_val(neg_groups)
        # mixed groups are rare/conflicts; keep in train to avoid val ambiguity
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

        if len(val_idx) == 0 or info["val_pos"] == 0 or info["val_neg"] == 0:
            print("[Split][WARN] group split produced weak val distribution, fallback to stratified random.")
        else:
            print(f"[Split] {info}")
            if overlap:
                print("[Split][WARN] group leakage overlap:", overlap[:8])
            return Subset(ds, train_idx), Subset(ds, val_idx), info

    # fallback stratified random
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
        self.score_mlp = nn.Sequential(
            nn.Linear(h * 2 + edge_dim, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(h + edge_dim, h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        self.out = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, h),
        )
        self.norm1 = nn.LayerNorm(h)
        self.ffn = nn.Sequential(
            nn.Linear(h, h * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h * 4, h),
        )
        self.norm2 = nn.LayerNorm(h)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge, node_mask):
        B, S, H = h.shape
        hi = h.unsqueeze(2).expand(B, S, S, H)
        hj = h.unsqueeze(1).expand(B, S, S, H)
        score_in = torch.cat([hi, hj, edge], dim=-1)
        score = self.score_mlp(score_in).squeeze(-1)

        key_mask = node_mask.unsqueeze(1).expand(B, S, S) > 0
        query_mask = node_mask.unsqueeze(2).expand(B, S, S) > 0
        attn_mask = key_mask & query_mask
        score = score.masked_fill(~attn_mask, -1e9)
        alpha = torch.softmax(score, dim=-1)
        alpha = alpha.masked_fill(~attn_mask, 0.0)

        msg_in = torch.cat([hj, edge], dim=-1)
        msg = self.msg_mlp(msg_in)
        agg = (alpha.unsqueeze(-1) * msg).sum(dim=2)

        h2 = self.norm1(h + self.dropout(self.out(torch.cat([h, agg], dim=-1))))
        h3 = self.norm2(h2 + self.dropout(self.ffn(h2)))
        return h3 * node_mask.unsqueeze(-1)


class TopoHGTv2(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim
        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_dim, h // 2),
            nn.LayerNorm(h // 2),
            nn.GELU(),
            nn.Linear(h // 2, h // 2),
        )
        self.blocks = nn.ModuleList([
            EdgeAwareGraphBlock(h, h // 2, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])

        self.quality_head = nn.Sequential(
            nn.Linear(h * 2 + 1, h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, 1),
        )

        self.rel_head = nn.Sequential(
            nn.Linear(h * 2 + h // 2, h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, 4),
        )

    def forward(self, batch):
        x = batch["node_features"]
        e = batch["edge_features"]
        m = batch["node_mask"]

        h = self.node_embed(x) * m.unsqueeze(-1)
        ee = self.edge_embed(e)

        for blk in self.blocks:
            h = blk(h, ee, m)

        B, S, H = h.shape
        hi = h.unsqueeze(2).expand(B, S, S, H)
        hj = h.unsqueeze(1).expand(B, S, S, H)
        rel_logits = self.rel_head(torch.cat([hi, hj, ee], dim=-1))

        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_pool = (h * m.unsqueeze(-1)).sum(dim=1) / denom
        h_masked = h.masked_fill(m.unsqueeze(-1) <= 0, -1e9)
        max_pool = h_masked.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        stroke_count_norm = denom / float(self.cfg.max_strokes)
        q_logit = self.quality_head(torch.cat([mean_pool, max_pool, stroke_count_norm], dim=-1)).squeeze(-1)

        return {"quality_logit": q_logit, "rel_logits": rel_logits}


# =============================================================================
# 8. Loss / metrics
# =============================================================================

def relation_class_weights(counts: np.ndarray, cfg: Cfg):
    c = torch.tensor(counts, dtype=torch.float32).clamp_min(1.0)
    total = c.sum()
    w = torch.sqrt(total / (len(c) * c))
    return torch.clamp(w, 0.10, cfg.rel_weight_clip)


def quality_pos_weight(counts: np.ndarray, cfg: Cfg):
    neg = float(counts[0])
    pos = float(counts[1])
    if pos <= 0:
        return 1.0
    return float(min(cfg.quality_pos_weight_clip, max(1.0, neg / max(1.0, pos))))


def compute_loss(out, batch, cfg: Cfg, bce, rel_ce):
    q_loss = bce(out["quality_logit"], batch["quality"])
    rel_lab = batch["rel_labels"]
    rel_loss = rel_ce(out["rel_logits"].reshape(-1, 4), rel_lab.reshape(-1))
    loss = cfg.w_quality * q_loss + cfg.w_relation * rel_loss
    return loss, {"loss": loss.detach(), "quality": q_loss.detach(), "relation": rel_loss.detach()}


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
    # pairwise exact for small val sets
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


def rel_confusion_from_logits(logits, labels):
    pred = logits.argmax(dim=-1).detach().cpu().numpy().reshape(-1)
    lab = labels.detach().cpu().numpy().reshape(-1)
    mask = lab != IGNORE
    pred = pred[mask]
    lab = lab[mask]
    conf = np.zeros((4, 4), dtype=np.int64)
    for t, p in zip(lab, pred):
        if 0 <= int(t) < 4 and 0 <= int(p) < 4:
            conf[int(t), int(p)] += 1
    return conf


def f1_from_conf(conf, cls):
    tp = int(conf[cls, cls])
    fp = int(conf[:, cls].sum() - tp)
    fn = int(conf[cls, :].sum() - tp)
    pr = tp / max(1, tp + fp)
    rc = tp / max(1, tp + fn)
    f1 = 2 * pr * rc / max(1e-8, pr + rc)
    return pr, rc, f1


@torch.no_grad()
def predict_collect(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        tensor_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(tensor_batch)
        probs = torch.sigmoid(out["quality_logit"]).detach().cpu().numpy().tolist()
        labels = batch["quality"].detach().cpu().numpy().astype(int).tolist()
        uids = batch["uid"]
        sources = batch["source"]
        groups = batch["group_key"]
        scounts = batch["stroke_count"].detach().cpu().numpy().astype(int).tolist()
        for uid, src, grp, p, lab, sc in zip(uids, sources, groups, probs, labels, scounts):
            rows.append({"uid": uid, "source": src, "group_key": grp, "prob": float(p), "label": int(lab), "stroke_count": int(sc)})
    model.train()
    return rows


@torch.no_grad()
def evaluate(model, loader, device, cfg, bce, rel_ce):
    model.eval()
    sums = {"loss": 0.0, "quality": 0.0, "relation": 0.0}
    batches = 0
    all_probs = []
    all_labels = []
    rel_conf = np.zeros((4, 4), dtype=np.int64)

    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch)
        loss, parts = compute_loss(out, batch, cfg, bce, rel_ce)
        for k in sums:
            sums[k] += float(parts[k].item())
        batches += 1
        prob = torch.sigmoid(out["quality_logit"]).detach().cpu().numpy()
        lab = batch["quality"].detach().cpu().numpy().astype(np.int64)
        all_probs.extend(prob.tolist())
        all_labels.extend(lab.tolist())
        rel_conf += rel_confusion_from_logits(out["rel_logits"], batch["rel_labels"])

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

    for name in ["E2E", "T", "X"]:
        cls = REL_MAP[name]
        pr, rc, f1 = f1_from_conf(rel_conf, cls)
        m[f"rel_{name}_precision"] = pr
        m[f"rel_{name}_recall"] = rc
        m[f"rel_{name}_f1"] = f1

    m["rel_confusion"] = rel_conf.tolist()
    model.train()
    return m


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
        f"  relation F1 E2E/T/X={m['rel_E2E_f1']:.3f}/"
        f"{m['rel_T_f1']:.3f}/{m['rel_X_f1']:.3f}"
    )
    top5 = sorted(m["quality_threshold_sweep"], key=lambda r: r["f1"], reverse=True)[:5]
    print("  threshold top5:", " | ".join(
        f"th={r['threshold']:.2f}:P={r['precision']:.3f},R={r['recall']:.3f},F1={r['f1']:.3f}"
        for r in top5
    ))


# =============================================================================
# 9. Error export
# =============================================================================

def export_error_analysis(model, ds: FlywheelTopoDataset, val_set: Subset, val_loader, device, cfg: Cfg, m: Dict[str, Any], tag: str):
    pred_rows = predict_collect(model, val_loader, device)
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
    fn.sort(key=lambda r: r["prob"])  # very low score but should be good

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
    print("[FlywheelTopoHGT-v2] start")
    print(f"[Time] {now_str()}")
    print(f"[ScriptDir] {SCRIPT_DIR}")
    print(f"[ModelSaveDir] {MODEL_SAVE_DIR}")
    print(f"[CharGlyphDir] {CHAR_GLYPH_DIR}")
    print(f"[DatasetAnalyseDir] {DATASET_ANALYSE_DIR}")
    print(f"[AnnotationsTopoDir] {annotations_dir}")
    print(f"[PCGPoolRoot] {pool_root}")
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

    model = TopoHGTv2(ds.node_dim, ds.edge_dim, cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] TopoHGTv2 node_dim={ds.node_dim} edge_dim={ds.edge_dim} params={params:,} trainable={trainable:,}")

    q_pos_weight_val = quality_pos_weight(ds.quality_counts, cfg)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(q_pos_weight_val, dtype=torch.float32, device=device))
    rel_w = relation_class_weights(ds.rel_counts, cfg).to(device)
    rel_ce = nn.CrossEntropyLoss(weight=rel_w, ignore_index=IGNORE)

    print(f"[Loss] quality_pos_weight={q_pos_weight_val:.4f}")
    print("[Loss] relation_weights:", {REL_INV[i]: round(float(rel_w[i].detach().cpu()), 4) for i in range(4)})

    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))

    latest_path = os.path.join(MODEL_SAVE_DIR, f"{cfg.save_prefix}_latest.pth")
    best_loss_path = os.path.join(MODEL_SAVE_DIR, f"{cfg.save_prefix}_best_loss.pth")
    best_f1_path = os.path.join(MODEL_SAVE_DIR, f"{cfg.save_prefix}_best_quality_f1.pth")

    best_loss = math.inf
    best_f1 = -1.0
    best_metric_for_stop = -math.inf if cfg.early_stop_metric == "quality_best_f1" else math.inf
    bad_epochs = 0

    def make_ckpt(epoch, train_loss, m, extra=None):
        return {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "node_dim": ds.node_dim,
            "edge_dim": ds.edge_dim,
            "rel_map": REL_MAP,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_metrics": m,
            "split_info": split_info,
            "data_manifest_path": manifest_path,
            "quality_counts": ds.quality_counts.tolist(),
            "relation_counts": ds.rel_counts.tolist(),
            "quality_pos_weight": q_pos_weight_val,
            "relation_weights": rel_w.detach().cpu().tolist(),
            "created_at": now_str(),
            "extra": extra or {},
        }

    print("[Train] start...")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            out = model(batch)
            loss, parts = compute_loss(out, batch, cfg, bce, rel_ce)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total_loss += float(loss.item())
            batches += 1

        sched.step()

        do_eval = (epoch == 1) or (epoch % cfg.log_every == 0) or (epoch == cfg.epochs)
        if not do_eval:
            continue

        train_loss = total_loss / max(1, batches)
        m = evaluate(model, val_loader, device, cfg, bce, rel_ce)
        lr = sched.get_last_lr()[0]
        print_metrics(epoch, train_loss, m, lr, cfg)

        ckpt = make_ckpt(epoch, train_loss, m)
        torch.save(ckpt, latest_path)

        improved = False

        if m["loss"] < best_loss:
            best_loss = m["loss"]
            torch.save(ckpt, best_loss_path)
            export_error_analysis(model, ds, val_set, val_loader, device, cfg, m, tag="best_loss")
            print(f"  ✅ best-loss saved: {best_loss_path}")
            improved = True

        cur_f1 = float(m["quality_best_f1"]["f1"])
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            torch.save(ckpt, best_f1_path)
            export_error_analysis(model, ds, val_set, val_loader, device, cfg, m, tag="best_quality_f1")
            print(f"  ✅ best-quality-f1 saved: {best_f1_path} | f1={best_f1:.4f} th={m['quality_best_f1']['threshold']:.2f}")
            improved = True

        # early stopping
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
    print(f"best_loss={best_loss:.4f} | best_quality_f1={best_f1:.4f}")
    print(f"latest:      {latest_path}")
    print(f"best_loss:   {best_loss_path}")
    print(f"best_q_f1:   {best_f1_path}")
    print(f"manifest:    {manifest_path}")
    print(f"model_dir:   {MODEL_SAVE_DIR}")


if __name__ == "__main__":
    train()
