# -*- coding: utf-8 -*-
r"""
pcg_good_cleaner_stage1_stage2_refiner.py

放置位置：
    dataset_analyse_p0/Char_Glyph_v0/annotation_tool/

作用：
    读取 annotation_tool/pcg_filebacked_stage2_schema/good 中的 Good pool，
    用一阶段 graph_editor_best.pth 作为“Merge/Delete 建议器”，对 Good 样本做保守清理。
    原 good pool 永远不删除、不覆盖；clean 后结果写入：
        annotation_tool/pcg_filebacked_stage2_schema/cleaned/

关键约束：
    1. 待 clean 与 cleaned 不重叠：用 stable_uid 识别，不依赖 outer hex_key / char。
    2. cleaned 外层 key = source stable_uid。
    3. 删除 cleaned 结果只删除 cleaned，不动 good；删除后样本回到待 clean。
    4. 单 JSON 文件默认限制 80MB。
    5. pth 只提供操作建议；最终 Merge/Delete 必须通过几何/轮廓护栏。
"""

import os, sys, json, math, time, copy, hashlib, traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from PIL import Image, ImageDraw, ImageTk

try:
    import networkx as nx
except Exception:
    nx = None

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

try:
    from scipy.ndimage import distance_transform_edt
except Exception:
    distance_transform_edt = None

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


# =============================================================================
# 0. 路径与参数
# =============================================================================

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
CHAR_GLYPH_DIR = os.path.abspath(os.path.join(TOOL_DIR, "..")) if os.path.basename(TOOL_DIR).lower() == "annotation_tool" else TOOL_DIR
DATASET_ANALYSE_DIR = os.path.abspath(os.path.join(CHAR_GLYPH_DIR, ".."))

POOL_ROOT_DEFAULT = os.path.join(TOOL_DIR, "pcg_filebacked_stage2_schema")
GOOD_DIR = "good"
CLEANED_DIR = "cleaned"
REPORT_DIR = "clean_reports"

CLEANED_PREFIX = "PCG_GoodCleaned_Stage1Refiner"
REPORT_PREFIX = "PCG_GoodCleanReport"

DEFAULT_MAX_JSON_MB = 80
PREVIEW_SIZE = 150
PAGE_ROWS_DEFAULT = 3
PAGE_COLS_DEFAULT = 4
CANVAS_SIZE = 400.0

# clean 保守参数
MAX_REFINE_STEPS = 5
MAX_ACCEPTED_OPS = 4
MIN_STROKES_AFTER_CLEAN = 2

MERGE_ENDPOINT_DIST_MAX = 14.0
MERGE_MAX_BEZIER_RMS_ERROR = 2.8
MERGE_MAX_BEZIER_MAX_ERROR = 6.0
MERGE_MIN_MASK_IOU = 0.86
MERGE_MAX_AREA_DELTA_RATIO = 0.18

DELETE_MIN_COVERAGE_BY_OTHERS = 0.78
DELETE_MAX_GLYPH_AREA_LOSS = 0.075

MASK_SIZE = 256
TOPO_SAMPLE_N = 80
TOPO_THRESH = 2.0
TOPO_CYCLE_POINT_THRESH = 5.0

ACTION_VOCAB = {"Merge": 0, "Delete": 1, "Split": 2, "Add_Dot": 3, "Done": 4}
ACTION_INV = {v: k for k, v in ACTION_VOCAB.items()}

# Phase2 topology candidate model
PHASE2_MODEL_FILENAME = "topo_phase2_candidate_v4_best_candidate_f1.pth"
PHASE2_MODEL_REL_DIR = os.path.join("AI_VECTOR_ROUTER_With_topo", "ml_engine_phase2")
PHASE2_MIN_ACCEPT_THRESHOLD = 0.80
PHASE2_MAX_OPS = 6
PHASE2_MAX_MOVE = 14.0
PHASE2_MIN_OUTLINE_IOU = 0.92
PHASE2_MAX_X_INCREASE = 0
PHASE2_ALLOW_CC_INCREASE = False
PHASE2_ALLOW_CYCLE_DECREASE = False

# Recursive clean:
#   repeatedly apply Stage1 -> Stage2 until one full iteration has no Stage1 change and no Stage2 change.
RECURSIVE_CLEAN_MAX_ITERS = 8
RECURSIVE_STAGE1_STAGE2_LABEL = "Stage1->Stage2 recursive clean"



# =============================================================================
# 1. JSON / pool
# =============================================================================

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_good_file(fn: str) -> bool:
    return fn.endswith("_topo.json") and ("AestheticGood" in fn or "Good" in fn) and "part" in fn

def is_cleaned_file(fn: str) -> bool:
    return fn.endswith("_topo.json") and fn.startswith(CLEANED_PREFIX) and "part" in fn

def load_pool(pool_root: str, kind: str) -> Dict[str, Any]:
    if kind == "good":
        d, matcher = os.path.join(pool_root, GOOD_DIR), is_good_file
    elif kind == "cleaned":
        d, matcher = os.path.join(pool_root, CLEANED_DIR), is_cleaned_file
    else:
        raise ValueError(kind)

    if not os.path.exists(d):
        return {}

    out = {}
    files = 0
    for fn in sorted(os.listdir(d)):
        if not matcher(fn):
            continue
        fp = os.path.join(d, fn)
        try:
            data = load_json(fp)
            if isinstance(data, dict):
                out.update(data)
                files += 1
        except Exception:
            print("[WARN] failed to load", fp)
            traceback.print_exc()
    print(f"[PoolLoad] {kind}: files={files}, items={len(out)}, dir={d}")
    return out

def estimate_entry_bytes(k: str, v: Any) -> int:
    try:
        return len(json.dumps({k: v}, ensure_ascii=False, indent=2).encode("utf-8")) + 8
    except Exception:
        return 1024 * 1024

def write_split_pool(data: Dict[str, Any], out_dir: str, prefix: str, max_json_mb: int = 80) -> List[str]:
    ensure_dir(out_dir)
    for fn in os.listdir(out_dir):
        if fn.startswith(prefix) and fn.endswith("_topo.json"):
            try:
                os.remove(os.path.join(out_dir, fn))
            except Exception:
                pass

    max_bytes = int(max_json_mb * 1024 * 1024)
    files, cur, cur_bytes, part = [], {}, 2, 0

    def flush(d, idx):
        if not d:
            return None
        fp = os.path.join(out_dir, f"{prefix}_part{idx:03d}_topo.json")
        save_json(d, fp)
        return fp

    for k, v in data.items():
        eb = estimate_entry_bytes(k, v)
        if cur and cur_bytes + eb > max_bytes:
            fp = flush(cur, part)
            if fp:
                files.append(fp)
            cur, cur_bytes, part = {}, 2, part + 1
        cur[k] = v
        cur_bytes += eb

    fp = flush(cur, part)
    if fp:
        files.append(fp)
    return files

def write_cleaned(pool_root: str, cleaned: Dict[str, Any], max_json_mb: int) -> List[str]:
    return write_split_pool(cleaned, os.path.join(pool_root, CLEANED_DIR), CLEANED_PREFIX, max_json_mb)


# =============================================================================
# 2. stable_uid：抗 Compact / merge 文件名 / hex_key 变化
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
    for ev in bundle.get("edit_history", []) if isinstance(bundle.get("edit_history", []), list) else []:
        if isinstance(ev, dict):
            for k in ["candidate_id", "source_candidate_id", "generated_glyph_id"]:
                if ev.get(k):
                    return str(ev[k])
    return ""

def geometry_sig(bundle: Dict[str, Any]):
    arr = []
    for i, s in enumerate(bundle.get("strokes", []) if isinstance(bundle.get("strokes", []), list) else []):
        if not isinstance(s, dict):
            continue
        arr.append({
            "id": int(s.get("bezier_id", i + 1)) if str(s.get("bezier_id", i + 1)).lstrip("-").isdigit() else i + 1,
            "mother_bezier": round_obj(s.get("mother_bezier"), 3),
            "width_bezier": round_obj(s.get("width_bezier", s.get("width")), 3),
            "stroke_type": s.get("stroke_type", ""),
        })
    arr.sort(key=lambda x: x["id"])
    return arr

def stable_uid(bundle: Dict[str, Any], outer_key: str = "") -> str:
    """
    注意：不使用 outer_key / hex_key / unicode_hex / char。
    Compact 或分片合并后依然稳定。
    """
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


# =============================================================================
# 3. Bézier / stroke / render
# =============================================================================

def cubic(P: np.ndarray, t: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)
    if t.ndim == 1:
        t = t[:, None]
    mt = 1 - t
    return mt**3 * P[0] + 3 * mt**2 * t * P[1] + 3 * mt * t**2 * P[2] + t**3 * P[3]

def deriv(P: np.ndarray, t: float) -> np.ndarray:
    P = np.asarray(P, dtype=np.float32)
    mt = 1 - float(t)
    t = float(t)
    return 3*mt**2*(P[1]-P[0]) + 6*mt*t*(P[2]-P[1]) + 3*t**2*(P[3]-P[2])

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
    return np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)


def edge_width_ctrl(e: Dict[str, Any]) -> np.ndarray:
    wb = e.get("width_bezier")
    if isinstance(wb, (list, tuple, np.ndarray)):
        try:
            a = np.asarray(wb, dtype=np.float32).reshape(-1)
            if len(a) == 4:
                return np.maximum(a, 0.5).astype(np.float32)
            if len(a) > 0:
                m = float(np.mean(a))
                return np.array([m, m, m, m], dtype=np.float32)
        except Exception:
            pass
    w = max(float(e.get("width", 10.0)), 0.5)
    return np.array([w, w, w, w], dtype=np.float32)


def width_at_ts(width_ctrl, ts) -> np.ndarray:
    w = np.asarray(width_ctrl, dtype=np.float32).reshape(4)
    t = np.asarray(ts, dtype=np.float32).reshape(-1)
    mt = 1.0 - t
    vals = mt**3*w[0] + 3*mt**2*t*w[1] + 3*mt*t**2*w[2] + t**3*w[3]
    return np.maximum(vals.astype(np.float32), 0.5)


def avg_width(s: Dict[str, Any]) -> float:
    return float(np.mean(width_ctrl_from_stroke(s)))


def bundle_to_edges(bundle: Dict[str, Any], sample_n: int = 64) -> List[Dict[str, Any]]:
    edges = []
    ts = np.linspace(0, 1, sample_n)
    for i, s in enumerate(bundle.get("strokes", []) if isinstance(bundle.get("strokes", []), list) else []):
        if not isinstance(s, dict):
            continue
        mb = s.get("mother_bezier")
        if not (isinstance(mb, list) and len(mb) == 4):
            continue
        try:
            P = np.asarray(mb, dtype=np.float32)
            if P.shape != (4, 2):
                continue
        except Exception:
            continue
        try:
            eid = int(s.get("bezier_id", i + 1))
        except Exception:
            eid = i + 1
        wc = width_ctrl_from_stroke(s)
        edges.append({
            "id": eid,
            "path": cubic(P, ts).astype(np.float32),
            "mother_bezier": P,
            "width": float(np.mean(wc)),
            "width_bezier": wc.astype(float).tolist(),
        })
    return edges

def clean_path(path):
    path = np.asarray(path, dtype=np.float32)
    if len(path) <= 1:
        return path
    d = np.linalg.norm(np.diff(path, axis=0), axis=1)
    keep = [0] + list(np.where(d > 0.1)[0] + 1)
    return path[keep]

def chord_t(path):
    path = np.asarray(path, dtype=np.float32)
    if len(path) <= 1:
        return np.array([0.0], dtype=np.float32)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-6:
        return np.linspace(0, 1, len(path), dtype=np.float32)
    return (cum / cum[-1]).astype(np.float32)

def fit_cubic(path) -> Tuple[np.ndarray, float, float]:
    path = clean_path(path)
    if len(path) < 2:
        p = path[0] if len(path) else np.array([0, 0], dtype=np.float32)
        P = np.stack([p, p, p, p]).astype(np.float32)
        return P, 0.0, 0.0
    P0, P3 = path[0], path[-1]
    if len(path) < 4:
        v = P3 - P0
        P = np.stack([P0, P0 + v/3, P0 + 2*v/3, P3]).astype(np.float32)
        err = np.linalg.norm(cubic(P, np.linspace(0, 1, len(path))) - path, axis=1)
        return P, float(np.sqrt(np.mean(err**2))), float(np.max(err))
    t = chord_t(path)
    mt = 1 - t
    b0, b1, b2, b3 = mt**3, 3*mt**2*t, 3*mt*t**2, t**3
    rhs = path - b0[:, None] * P0 - b3[:, None] * P3
    A = np.stack([b1, b2], axis=1)
    try:
        sx, *_ = np.linalg.lstsq(A, rhs[:, 0], rcond=None)
        sy, *_ = np.linalg.lstsq(A, rhs[:, 1], rcond=None)
        P1, P2 = np.array([sx[0], sy[0]], dtype=np.float32), np.array([sx[1], sy[1]], dtype=np.float32)
    except Exception:
        v = P3 - P0
        P1, P2 = P0 + v/3, P0 + 2*v/3
    P = np.stack([P0, P1, P2, P3]).astype(np.float32)
    err = np.linalg.norm(cubic(P, t) - path, axis=1)
    return P, float(np.sqrt(np.mean(err**2))), float(np.max(err))

def phase1_resample_merged_path(path: np.ndarray) -> np.ndarray:
    """
    贴近一阶段 action_merge 的路径预处理：
      - stitch 后如果几乎是直线，直接拉成直线均匀采样；
      - 否则按弧长重采样，避免点密度影响 Bézier 拟合。
    """
    path = clean_path(np.asarray(path, dtype=np.float32))
    if len(path) <= 2:
        return path
    diffs = np.diff(path, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    total_len = float(np.sum(dists))
    chord_len = float(np.linalg.norm(path[-1] - path[0]))
    if total_len < 1e-6:
        return path
    if chord_len > 1.0 and (total_len / chord_len) < 1.05:
        num_pts = len(path)
        t_vals = np.linspace(0, 1, num_pts).reshape(-1, 1)
        return (path[0] * (1 - t_vals) + path[-1] * t_vals).astype(np.float32)
    cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
    t_uniform = np.linspace(0, total_len, max(20, int(total_len)))
    new_x = np.interp(t_uniform, cum_dist, path[:, 0])
    new_y = np.interp(t_uniform, cum_dist, path[:, 1])
    return np.column_stack([new_x, new_y]).astype(np.float32)


def stitch(a, b) -> Tuple[np.ndarray, str, float]:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    modes = {
        "end_to_start": (np.linalg.norm(a[-1]-b[0]), np.vstack([a, b])),
        "end_to_end": (np.linalg.norm(a[-1]-b[-1]), np.vstack([a, b[::-1]])),
        "start_to_start": (np.linalg.norm(a[0]-b[0]), np.vstack([a[::-1], b])),
        "start_to_end": (np.linalg.norm(a[0]-b[-1]), np.vstack([a[::-1], b[::-1]])),
    }
    mode = min(modes, key=lambda k: modes[k][0])
    d, p = modes[mode]
    return clean_path(p), mode, float(d)

def mask_view(edge_groups: List[List[Dict[str, Any]]], size=MASK_SIZE, pad=18):
    pts = []
    for edges in edge_groups:
        for e in edges:
            p = np.asarray(e.get("path", []), dtype=np.float32)
            if len(p):
                pts.append(p)
    if not pts:
        return np.array([0.0, 0.0], dtype=np.float32), 1.0
    allp = np.concatenate(pts, axis=0)
    mn, mx = np.min(allp, axis=0), np.max(allp, axis=0)
    span = np.maximum(mx - mn, 1e-5)
    scale = min((size - 2*pad) / span[0], (size - 2*pad) / span[1])
    return mn.astype(np.float32), float(scale)


def _map_points_to_mask(points: np.ndarray, view, size=MASK_SIZE, pad=18) -> np.ndarray:
    mn, scale = view
    pts = np.asarray(points, dtype=np.float32)
    x = (pts[:, 0] - mn[0]) * scale + pad
    y = (pts[:, 1] - mn[1]) * scale + pad
    y = size - y
    return np.column_stack([x, y]).astype(np.float32)


def render_mask(edges: List[Dict[str, Any]], size=MASK_SIZE, pad=18, view=None) -> np.ndarray:
    """
    按真实 width_bezier 渲染轮廓 mask。
    """
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    if not edges:
        return np.array(img, dtype=np.uint8)
    if view is None:
        view = mask_view([edges], size=size, pad=pad)

    for e in edges:
        path = np.asarray(e.get("path", []), dtype=np.float32)
        if len(path) < 2:
            continue
        if "mother_bezier" in e:
            try:
                P = np.asarray(e["mother_bezier"], dtype=np.float32)
                if P.shape == (4, 2):
                    ts = np.linspace(0, 1, max(80, len(path)))
                    path = cubic(P, ts).astype(np.float32)
                else:
                    ts = np.linspace(0, 1, len(path))
            except Exception:
                ts = np.linspace(0, 1, len(path))
        else:
            ts = np.linspace(0, 1, len(path))

        pix = _map_points_to_mask(path, view, size=size, pad=pad)
        _, scale = view
        w_pix = np.maximum(width_at_ts(edge_width_ctrl(e), ts) * scale, 0.75)

        dp = np.gradient(pix, axis=0)
        n = np.zeros_like(dp)
        n[:, 0], n[:, 1] = -dp[:, 1], dp[:, 0]
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-6)

        upper = pix + n * w_pix[:, None]
        lower = pix - n * w_pix[:, None]
        poly = np.vstack([upper, lower[::-1]])
        draw.polygon([tuple(map(float, q)) for q in poly], fill=255)

        for q, r in [(pix[0], float(w_pix[0])), (pix[-1], float(w_pix[-1]))]:
            x, y = float(q[0]), float(q[1])
            draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

    return np.array(img, dtype=np.uint8)


def fit_width_bezier_from_outline(P: np.ndarray, target_edges: List[Dict[str, Any]], size=MASK_SIZE, pad=18) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    从合并前两条带宽度 stroke 的轮廓中回归合并后 width_bezier。
    逻辑类似 regress_width_dt_fast：目标轮廓 mask -> distance transform -> 中心线采样 -> 三次宽度拟合。
    """
    P = np.asarray(P, dtype=np.float32)
    ts = np.linspace(0, 1, 96)
    curve = cubic(P, ts).astype(np.float32)

    base_w = float(np.mean([np.mean(edge_width_ctrl(e)) for e in target_edges]))
    tmp_edge = {"id": -999, "path": curve, "mother_bezier": P, "width_bezier": [base_w]*4}
    view = mask_view([target_edges, [tmp_edge]], size=size, pad=pad)
    target_mask = render_mask(target_edges, size=size, pad=pad, view=view)

    if distance_transform_edt is None or int((target_mask > 0).sum()) <= 0:
        vals, lens = [], []
        for e in target_edges:
            vals.append(float(np.mean(edge_width_ctrl(e))))
            p = np.asarray(e.get("path", []), dtype=np.float32)
            lens.append(float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if len(p) > 1 else 1.0)
        w = float(np.average(vals, weights=np.maximum(lens, 1e-6)))
        return np.array([w, w, w, w], dtype=np.float32), {"width_fit_mode": "fallback_weighted_mean"}

    dt = distance_transform_edt(target_mask > 0).astype(np.float32)
    pix = _map_points_to_mask(curve, view, size=size, pad=pad)
    xs = np.clip(np.round(pix[:, 0]).astype(int), 0, size - 1)
    ys = np.clip(np.round(pix[:, 1]).astype(int), 0, size - 1)
    _, scale = view
    w_samples = dt[ys, xs] / max(scale, 1e-6)

    positive = w_samples[w_samples > 0.25]
    if len(positive) == 0:
        w0 = base_w
        return np.array([w0, w0, w0, w0], dtype=np.float32), {"width_fit_mode": "fallback_no_positive_dt"}

    med = float(np.median(positive))
    w_samples = np.where(w_samples > 0.25, w_samples, med)

    t = ts.astype(np.float32)
    mt = 1.0 - t
    A = np.stack([mt**3, 3*mt**2*t, 3*mt*t**2, t**3], axis=1)
    try:
        wc, *_ = np.linalg.lstsq(A, w_samples, rcond=None)
    except Exception:
        wc = np.array([med, med, med, med], dtype=np.float32)

    old_ws = np.concatenate([edge_width_ctrl(e) for e in target_edges])
    lo = max(0.5, float(np.percentile(old_ws, 5)) * 0.45)
    hi = max(lo + 0.5, float(np.percentile(old_ws, 95)) * 2.2)
    wc = np.clip(np.asarray(wc, dtype=np.float32), lo, hi)

    pred = A @ wc
    rms = float(np.sqrt(np.mean((pred - w_samples) ** 2)))
    return wc.astype(np.float32), {
        "width_fit_mode": "distance_transform_lstsq",
        "width_rms_error": round(rms, 3),
        "width_ctrl": [round(float(x), 3) for x in wc.tolist()],
        "width_clip_range": [round(float(lo), 3), round(float(hi), 3)],
    }

def mask_iou(a, b) -> float:
    A, B = a > 0, b > 0
    u = np.logical_or(A, B).sum()
    if u <= 0:
        return 1.0
    return float(np.logical_and(A, B).sum()) / float(u)

def render_bundle(bundle: Dict[str, Any], size=PREVIEW_SIZE) -> Image.Image:
    edges = bundle_to_edges(bundle, sample_n=70)
    gap = 8
    img = Image.new("RGB", (size*2 + gap, size), "white")
    draw = ImageDraw.Draw(img)
    if not edges:
        draw.text((10, 10), "NO STROKES", fill=(0, 0, 0))
        return img
    curves = [e["path"] for e in edges]
    widths = [float(e.get("width", 10.0)) for e in edges]
    allp = np.concatenate(curves, axis=0)
    mn, mx = np.min(allp, axis=0), np.max(allp, axis=0)
    span = np.maximum(mx - mn, 1e-5)
    scale = min((size-34) / span[0], (size-34) / span[1])
    med = max(float(np.median(widths)), 1e-3)

    def mp(p, xoff=0):
        x = (p[0]-mn[0])*scale + 17 + xoff
        y = (p[1]-mn[1])*scale + 17
        return float(x), float(size-y)

    palette = [(220,20,60),(30,144,255),(34,139,34),(255,140,0),(138,43,226),(0,170,170),(180,90,20),(210,40,160)]
    draw.text((4,4), "black", fill=(0,0,0))
    draw.text((size+gap+4,4), "color", fill=(0,0,0))
    for e in edges:
        xy = [mp(p, 0) for p in e["path"]]
        w = int(round(max(2, min(12, 5*float(e.get("width", 10.0))/med))))
        draw.line(xy, fill=(0,0,0), width=w, joint="curve")
    for i, e in enumerate(edges):
        xy = [mp(p, size+gap) for p in e["path"]]
        w = int(round(max(2, min(12, 5*float(e.get("width", 10.0))/med))))
        draw.line(xy, fill=(210,210,210), width=w+2, joint="curve")
        draw.line(xy, fill=palette[i % len(palette)], width=w, joint="curve")
    return img


# =============================================================================
# 4. Topology
# =============================================================================

def angle(v1, v2) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-5 or n2 < 1e-5:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(n1*n2), -1, 1))))

def seg_inter(p0, p1, q0, q1, eps=1e-8):
    p0, p1, q0, q1 = map(lambda x: np.asarray(x, dtype=np.float32), [p0,p1,q0,q1])
    r, s = p1-p0, q1-q0
    den = float(r[0]*s[1] - r[1]*s[0])
    if abs(den) < eps:
        return None
    qp = q0-p0
    t = float((qp[0]*s[1] - qp[1]*s[0]) / den)
    u = float((qp[0]*r[1] - qp[1]*r[0]) / den)
    if 0 <= t <= 1 and 0 <= u <= 1:
        return p0 + t*r, t, u
    return None

def poly_x(c1, c2, margin=2):
    n1, n2 = len(c1), len(c2)
    for i in range(margin, max(margin, n1-1-margin)):
        for j in range(margin, max(margin, n2-1-margin)):
            h = seg_inter(c1[i], c1[i+1], c2[j], c2[j+1])
            if h is None:
                continue
            pt, lt, lu = h
            t1, t2 = (i+lt)/(n1-1), (j+lu)/(n2-1)
            if 0.02 < t1 < 0.98 and 0.02 < t2 < 0.98:
                return True, pt, t1, t2
    return False, None, None, None

def compute_topo(bundle: Dict[str, Any]) -> Dict[str, Any]:
    edges = bundle_to_edges(bundle, sample_n=TOPO_SAMPLE_N)
    ts = np.linspace(0, 1, TOPO_SAMPLE_N)
    evs, graph_edges, conn_pts = [], [], defaultdict(list)

    def add_conn(a, b, pos):
        u, v = sorted([int(a), int(b)])
        graph_edges.append((u, v))
        conn_pts[(u, v)].append(np.asarray(pos, dtype=np.float32))

    for i, e1 in enumerate(edges):
        for j, e2 in enumerate(edges):
            if i >= j:
                continue
            id1, id2 = int(e1["id"]), int(e2["id"])
            P1, P2 = e1["mother_bezier"], e2["mother_bezier"]
            c1, c2 = cubic(P1, ts), cubic(P2, ts)
            is_e2e, t_events = False, []

            for a, ta in [(0,0.0),(3,1.0)]:
                for b, tb in [(0,0.0),(3,1.0)]:
                    if np.linalg.norm(P1[a] - P2[b]) < TOPO_THRESH:
                        pos = P1[a]
                        is_e2e = True
                        evs.append({"type":"E2E","stroke_a":id1,"t_a":ta,"stroke_b":id2,"t_b":tb,
                                    "position":[round(float(pos[0]),1),round(float(pos[1]),1)]})
                        add_conn(id1, id2, pos)

            if not is_e2e:
                for a, ta in [(0,0.0),(3,1.0)]:
                    d = np.linalg.norm(c2 - P1[a], axis=1)
                    m = int(np.argmin(d))
                    if d[m] < TOPO_THRESH:
                        tb = m / float(TOPO_SAMPLE_N-1)
                        pos = c2[m]
                        ev = {"type":"T","guest":id1,"guest_t":ta,"host":id2,"host_t":round(float(tb),3),
                              "angle":round(angle(deriv(P1,ta), deriv(P2,tb)),1),
                              "position":[round(float(pos[0]),1),round(float(pos[1]),1)]}
                        evs.append(ev); t_events.append(ev); add_conn(id1, id2, pos)
                for b, tb in [(0,0.0),(3,1.0)]:
                    d = np.linalg.norm(c1 - P2[b], axis=1)
                    m = int(np.argmin(d))
                    if d[m] < TOPO_THRESH:
                        ta = m / float(TOPO_SAMPLE_N-1)
                        pos = c1[m]
                        ev = {"type":"T","guest":id2,"guest_t":tb,"host":id1,"host_t":round(float(ta),3),
                              "angle":round(angle(deriv(P1,ta), deriv(P2,tb)),1),
                              "position":[round(float(pos[0]),1),round(float(pos[1]),1)]}
                        evs.append(ev); t_events.append(ev); add_conn(id1, id2, pos)

            if not is_e2e and not t_events:
                hit, pt, ta, tb = poly_x(c1, c2)
                if hit:
                    evs.append({"type":"X","stroke_a":id1,"t_a":round(float(ta),3),"stroke_b":id2,"t_b":round(float(tb),3),
                                "angle":round(angle(deriv(P1,ta), deriv(P2,tb)),1),
                                "position":[round(float(pt[0]),1),round(float(pt[1]),1)]})
                    add_conn(id1, id2, pt)

    nodes = [int(e["id"]) for e in edges]
    unique = sorted(set(graph_edges))
    if nx is not None:
        G = nx.Graph()
        G.add_nodes_from(nodes); G.add_edges_from(unique)
        cc = nx.number_connected_components(G) if nodes else 0
        basis = nx.cycle_basis(G)
    else:
        cc = 1 if nodes else 0
        basis = [[] for _ in range(max(0, len(unique) - len(nodes) + cc))]

    cycles, seen = [], set()
    for (u, v), pts in conn_pts.items():
        if len(pts) >= 2:
            ok = any(np.linalg.norm(pts[a]-pts[b]) >= TOPO_CYCLE_POINT_THRESH for a in range(len(pts)) for b in range(a+1,len(pts)))
            if ok:
                key = tuple(sorted([u, v])); seen.add(key)
                cycles.append({"cycle_id":len(cycles),"members":[u,v],"orientation":"unknown"})
    for cyc in basis:
        if not cyc or len(cyc) < 3: continue
        key = tuple(sorted(map(int, cyc)))
        if key in seen: continue
        seen.add(key)
        cycles.append({"cycle_id":len(cycles),"members":[int(x) for x in cyc],"orientation":"unknown"})

    e2e = [e for e in evs if e["type"] == "E2E"]
    tj = [e for e in evs if e["type"] == "T"]
    xj = [e for e in evs if e["type"] == "X"]
    parts = []
    if e2e: parts.append("E2E: " + "、".join(f"{e['stroke_a']}-{e['stroke_b']}" for e in e2e[:16]))
    if tj: parts.append("T: " + "、".join(f"{e['guest']}搭{e['host']}" for e in tj[:16]))
    if xj: parts.append("X: " + "、".join(f"{e['stroke_a']}交叉{e['stroke_b']}" for e in xj[:16]))
    if cycles: parts.append("Cycles: " + " | ".join(" ".join(map(str,c["members"])) for c in cycles[:12]))
    if not parts: parts.append("No physical collision")
    stats = {"stroke_count":len(edges),"topology_event_count":len(evs),"connected_components":int(cc),
             "cycle_count":len(cycles),"e2e_count":len(e2e),"t_count":len(tj),"x_count":len(xj)}
    return {"topology_events":evs,"cycles":cycles,"topology_text":" ; ".join(parts),"stats":stats}


# =============================================================================
# 5. 一阶段模型
# =============================================================================

class EnvLocal:
    def calc_overlap(self, target, others, thresh=2.0):
        target = np.asarray(target, dtype=np.float32)
        valid = [np.asarray(p, dtype=np.float32) for p in others if len(p) > 0]
        if len(target) == 0 or not valid:
            return 0.0
        allp = np.vstack(valid)
        if cKDTree is not None:
            d, _ = cKDTree(allp).query(target, k=1, workers=-1)
        else:
            d = np.array([np.min(np.linalg.norm(allp - p[None,:], axis=1)) for p in target])
        return float(np.sum(d < thresh)) / max(1, len(target))

    def extract(self, edges):
        N = len(edges)
        if N == 0:
            return np.zeros((0,9), dtype=np.float32), np.zeros((0,0), dtype=np.float32)
        deg = defaultdict(int); endpoints = []
        def rpt(p): return (round(float(p[0]),1), round(float(p[1]),1))
        for e in edges:
            p = np.asarray(e["path"], dtype=np.float32)
            a, b = rpt(p[0]), rpt(p[-1])
            endpoints.append((a,b)); deg[a]+=1; deg[b]+=1
        feats = []
        for i, e in enumerate(edges):
            p = np.asarray(e["path"], dtype=np.float32)
            length = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if len(p) > 1 else 0.0
            center = np.mean(p, axis=0)
            theta = math.atan2(float(p[-1,1]-p[0,1]), float(p[-1,0]-p[0,0]))
            ds, de = deg[endpoints[i][0]], deg[endpoints[i][1]]
            overlap = self.calc_overlap(p, [x["path"] for j,x in enumerate(edges) if j != i], thresh=2.0)
            feats.append([length/800.0, float(center[0])/400.0, float(center[1])/400.0,
                          math.sin(theta), math.cos(theta), ds/4.0, de/4.0,
                          1.0 if (ds == 1 or de == 1) else 0.0, overlap])
        bias = np.full((N,N), -1.0, dtype=np.float32); np.fill_diagonal(bias, 0.0)
        for i in range(N):
            for j in range(i+1,N):
                if set(endpoints[i]).intersection(set(endpoints[j])):
                    bias[i,j]=bias[j,i]=2.0
        return np.asarray(feats, dtype=np.float32), bias

if torch is not None:
    class GraphEditorTransformer(nn.Module):
        def __init__(self, feature_dim=9, hidden_dim=128, n_heads=4, n_layers=3):
            super().__init__()
            self.edge_embedding = nn.Linear(feature_dim, hidden_dim)
            self.input_norm = nn.LayerNorm(hidden_dim)
            enc = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*2,
                                             batch_first=True, activation="gelu")
            self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
            self.type_head = nn.Sequential(nn.Linear(hidden_dim,64), nn.ReLU(), nn.Linear(64, len(ACTION_VOCAB)))
            self.pointer_query = nn.Linear(hidden_dim, hidden_dim)
            self.pointer_key = nn.Linear(hidden_dim, hidden_dim)
        def forward(self, x_feat, padding_mask):
            tokens = self.input_norm(self.edge_embedding(x_feat))
            enc = self.transformer(tokens, src_key_padding_mask=padding_mask)
            active = enc.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            cnt = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
            ctx = active.sum(dim=1) / cnt
            type_logits = self.type_head(ctx)
            q = self.pointer_query(ctx).unsqueeze(1)
            k = self.pointer_key(enc)
            ptr = torch.bmm(q, k.transpose(1,2)).squeeze(1)
            ptr = ptr.masked_fill(padding_mask, float("-inf"))
            return type_logits, ptr

def find_pth(verbose: bool = False):
    # 脚本位置：
    #   dataset_analyse_p0/Char_Glyph_v0/annotation_tool/pcg_good_cleaner_stage1_refiner_pth_fixed.py
    # 模型位置：
    #   dataset_analyse_p0/AI_VECTOR_ROUTER_With_topo/ml_engine/graph_editor_best.pth
    # 所以从 TOOL_DIR 出发，正确相对路径是 ../../AI_VECTOR_ROUTER_With_topo/ml_engine/graph_editor_best.pth
    expected = os.path.abspath(os.path.join(
        TOOL_DIR, "..", "..", "AI_VECTOR_ROUTER_With_topo", "ml_engine", "graph_editor_best.pth"
    ))

    cands = [
        expected,
        os.path.join(DATASET_ANALYSE_DIR, "AI_VECTOR_ROUTER_With_topo", "ml_engine", "graph_editor_best.pth"),
        os.path.join(CHAR_GLYPH_DIR, "..", "AI_VECTOR_ROUTER_With_topo", "ml_engine", "graph_editor_best.pth"),
        os.path.join(TOOL_DIR, "..", "AI_VECTOR_ROUTER_With_topo", "ml_engine", "graph_editor_best.pth"),
        os.path.join(TOOL_DIR, "ml_engine", "graph_editor_best.pth"),
    ]

    seen = set()
    for p in cands:
        p = os.path.abspath(p)
        if p in seen:
            continue
        seen.add(p)
        if verbose:
            print(f"[ModelPathCheck] exists={os.path.exists(p)} | {p}")
        if os.path.exists(p):
            return p
    return None

class Stage1Refiner:
    def __init__(self):
        self.env = EnvLocal()
        self.model_path = find_pth(verbose=False)

        self.model = None
        self.device = None
        self.status = ""

        if torch is None:
            self.status = "torch unavailable; using geometry heuristic only."
            print("[Stage1Refiner]", self.status)
            return

        if not self.model_path:
            self.status = "graph_editor_best.pth not found; using geometry heuristic only."
            print("[Stage1Refiner]", self.status)
            return
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = GraphEditorTransformer(feature_dim=9).to(self.device)
            state = torch.load(self.model_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict):
                state = {k.replace("module.", ""): v for k,v in state.items()}
            self.model.load_state_dict(state, strict=True)
            self.model.eval()
            self.status = "loaded pth: " + self.model_path
            print("[Stage1Refiner]", self.status)
        except Exception as e:
            self.model = None
            self.status = "pth load failed; using geometry heuristic only: " + repr(e)
            traceback.print_exc()

    def predict(self, edges, locked):
        if self.model is not None and torch is not None:
            try:
                x, _ = self.env.extract(edges)
                if len(x) == 0:
                    return "Done", None, {"mode":"model","reason":"empty"}
                xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
                pm = torch.zeros((1, len(x)), dtype=torch.bool).to(self.device)
                with torch.no_grad():
                    typ, ptr = self.model(xt, pm)
                typ[0, ACTION_VOCAB["Split"]] = -1e9
                typ[0, ACTION_VOCAB["Add_Dot"]] = -1e9
                for i,e in enumerate(edges):
                    if e["id"] in locked:
                        ptr[0,i] = -1e9
                if ptr[0].max().item() < -1e8:
                    return "Done", None, {"mode":"model","reason":"all_locked"}
                act = ACTION_INV.get(int(typ[0].argmax().item()), "Done")
                if act == "Done":
                    return "Done", None, {"mode":"model","reason":"done"}
                idx = int(ptr[0].argmax().item())
                return act, idx, {"mode":"model","target_id":int(edges[idx]["id"])}
            except Exception:
                traceback.print_exc()

        # fallback heuristic
        ovs = []
        for i,e in enumerate(edges):
            if e["id"] in locked: continue
            ovs.append((self.env.calc_overlap(e["path"], [x["path"] for j,x in enumerate(edges) if j != i], thresh=4.0), i))
        ovs.sort(reverse=True)
        if ovs and ovs[0][0] >= DELETE_MIN_COVERAGE_BY_OTHERS:
            return "Delete", ovs[0][1], {"mode":"heuristic","overlap":ovs[0][0]}
        best = (1e9, None)
        for i in range(len(edges)):
            if edges[i]["id"] in locked: continue
            for j in range(i+1, len(edges)):
                if edges[j]["id"] in locked: continue
                _, _, d = stitch(edges[i]["path"], edges[j]["path"])
                if d < best[0]: best = (d, i)
        if best[1] is not None and best[0] <= MERGE_ENDPOINT_DIST_MAX:
            return "Merge", best[1], {"mode":"heuristic","endpoint_dist":best[0]}
        return "Done", None, {"mode":"heuristic"}

    def find_partner(self, idx, edges, taboo, locked):
        e = edges[idx]; best = (1e9, None)
        for j,b in enumerate(edges):
            if j == idx or b["id"] in locked: continue
            pair = tuple(sorted([int(e["id"]), int(b["id"])]))
            if pair in taboo: continue
            _, _, d = stitch(e["path"], b["path"])
            if d < best[0]: best = (d, j)
        return best[1], best[0]

    def try_merge(self, edges, ia, ib):
        e1, e2 = edges[ia], edges[ib]
        path, mode, ed = stitch(e1["path"], e2["path"])
        path = phase1_resample_merged_path(path)

        rep = {"type":"Merge","old_ids":[int(e1["id"]),int(e2["id"])],
               "endpoint_dist":round(ed,3),"mode":mode,"accepted":False}

        if ed > MERGE_ENDPOINT_DIST_MAX:
            rep["reject_reason"] = f"endpoint too far: {ed:.3f} > {MERGE_ENDPOINT_DIST_MAX}"
            return False, edges, rep

        P, rms, mx = fit_cubic(path)
        rep["rms_error"], rep["max_error"] = round(rms,3), round(mx,3)

        if rms > MERGE_MAX_BEZIER_RMS_ERROR or mx > MERGE_MAX_BEZIER_MAX_ERROR:
            reasons = []
            if rms > MERGE_MAX_BEZIER_RMS_ERROR:
                reasons.append(f"rms_error {rms:.3f} > threshold {MERGE_MAX_BEZIER_RMS_ERROR}")
            if mx > MERGE_MAX_BEZIER_MAX_ERROR:
                reasons.append(f"max_error {mx:.3f} > threshold {MERGE_MAX_BEZIER_MAX_ERROR}")
            rep["reject_reason"] = "bezier fit error too high: " + "; ".join(reasons)
            return False, edges, rep

        # 关键修改：合并后 width 不是平均值，而是从合并前轮廓中重新回归 width_bezier。
        wc, wdbg = fit_width_bezier_from_outline(P, [e1, e2], size=MASK_SIZE)
        rep.update(wdbg)

        new = {"id":max(int(e["id"]) for e in edges)+1,
               "path":cubic(P, np.linspace(0,1,max(len(path),96))).astype(np.float32),
               "mother_bezier":P.astype(np.float32),
               "width":float(np.mean(wc)),
               "width_bezier":wc.astype(float).tolist()}

        shared_view = mask_view([[e1, e2], [new]], size=MASK_SIZE)
        before = render_mask([e1,e2], view=shared_view)
        after = render_mask([new], view=shared_view)
        iou = mask_iou(before, after)
        area_delta = abs(int((after>0).sum()) - int((before>0).sum())) / max(int((before>0).sum()), 1)
        rep["mask_iou"], rep["area_delta_ratio"] = round(iou,4), round(area_delta,4)

        if iou < MERGE_MIN_MASK_IOU or area_delta > MERGE_MAX_AREA_DELTA_RATIO:
            reasons = []
            if iou < MERGE_MIN_MASK_IOU:
                reasons.append(f"mask_iou {iou:.4f} < threshold {MERGE_MIN_MASK_IOU}")
            if area_delta > MERGE_MAX_AREA_DELTA_RATIO:
                reasons.append(f"area_delta_ratio {area_delta:.4f} > threshold {MERGE_MAX_AREA_DELTA_RATIO}")
            rep["reject_reason"] = "outline guard rejected: " + "; ".join(reasons)
            return False, edges, rep

        out = [copy.deepcopy(e) for k,e in enumerate(edges) if k not in (ia,ib)] + [new]
        rep["accepted"], rep["new_id"] = True, int(new["id"])
        rep["new_width_bezier"] = [round(float(x),3) for x in wc.tolist()]
        return True, out, rep

    def try_delete(self, edges, idx):
        e = edges[idx]
        rep = {"type":"Delete","old_id":int(e["id"]),"accepted":False}
        if len(edges) <= MIN_STROKES_AFTER_CLEAN:
            rep["reject_reason"] = "too few strokes"; return False, edges, rep
        rem = [copy.deepcopy(x) for k,x in enumerate(edges) if k != idx]
        shared_view = mask_view([[e], rem, edges], size=MASK_SIZE)
        target = render_mask([e], view=shared_view)
        rem_mask = render_mask(rem, view=shared_view)
        full = render_mask(edges, view=shared_view)
        target_area = max(int((target>0).sum()), 1)
        uncovered_target = np.logical_and(target > 0, rem_mask == 0).sum()
        covered = 1.0 - float(uncovered_target) / target_area
        full_area = max(int((full>0).sum()), 1)
        full_loss = float(np.logical_and(full > 0, rem_mask == 0).sum()) / full_area
        rep["coverage_by_others"], rep["glyph_area_loss"] = round(covered,4), round(full_loss,4)
        if covered < DELETE_MIN_COVERAGE_BY_OTHERS:
            rep["reject_reason"] = "not covered enough"; return False, edges, rep
        if full_loss > DELETE_MAX_GLYPH_AREA_LOSS:
            rep["reject_reason"] = "glyph area loss too high"; return False, edges, rep
        rep["accepted"] = True
        return True, rem, rep

    def all_close_merge_candidates(self, edges, taboo, locked=None, max_candidates=16):
        """
        几何 fallback：不完全相信模型的 action type。
        当模型连续建议 Delete 但被覆盖率护栏拒绝时，扫描所有近端点 pair，
        例如 E2E: 1-2, 2-3, 5-6 这种 pair，并按 endpoint_dist 排序。
        """
        locked = locked or set()
        cand = []
        for i in range(len(edges)):
            if edges[i]["id"] in locked:
                continue
            for j in range(i+1, len(edges)):
                if edges[j]["id"] in locked:
                    continue
                pair = tuple(sorted([int(edges[i]["id"]), int(edges[j]["id"])]))
                if pair in taboo:
                    continue
                _, mode, d = stitch(edges[i]["path"], edges[j]["path"])
                if d <= MERGE_ENDPOINT_DIST_MAX:
                    cand.append((float(d), i, j, mode, pair))
        cand.sort(key=lambda x: (x[0], x[4][0], x[4][1]))
        return cand[:max_candidates]

    def try_fallback_merge_scan(self, edges, taboo, locked, step, trigger, max_candidates=12, report_top_k=6):
        """
        Delete 被拒绝后，主动尝试最接近的几何 merge 候选。
        若有可接受 merge，直接执行；若都失败，写一个 FallbackMergeScan 报告，
        里面列出最接近的若干 pair 以及 reject_reason，方便人工判断。
        """
        candidates = self.all_close_merge_candidates(edges, taboo, locked=locked, max_candidates=max_candidates)
        summary = {
            "type": "FallbackMergeScan",
            "trigger": trigger,
            "step": step,
            "accepted": False,
            "candidate_count": len(candidates),
            "top_candidates": [],
        }

        for rank, (dist, i, j, mode, pair) in enumerate(candidates):
            ok, new_edges, rep = self.try_merge(edges, i, j)
            rep["fallback_rank"] = rank
            rep["fallback_trigger"] = trigger
            rep["fallback_pair"] = [int(edges[i]["id"]), int(edges[j]["id"])]
            rep["fallback_endpoint_dist"] = round(float(dist), 3)

            if ok:
                rep["fallback_after_rejected_action"] = True
                rep["step"] = step
                return True, new_edges, rep

            taboo.add(pair)
            if len(summary["top_candidates"]) < report_top_k:
                summary["top_candidates"].append(rep)

        if not candidates:
            summary["note"] = "No close endpoint pair found under MERGE_ENDPOINT_DIST_MAX."
        return False, edges, summary

    def refine(self, bundle, uid):
        before_topo = compute_topo(bundle)
        edges = bundle_to_edges(bundle, sample_n=64)
        locked, taboo, ops, rejected = set(), set(), [], []

        for step in range(MAX_REFINE_STEPS):
            if len(ops) >= MAX_ACCEPTED_OPS or len(edges) <= MIN_STROKES_AFTER_CLEAN:
                break

            act, idx, dbg = self.predict(edges, locked)
            if act == "Done" or idx is None:
                # 即使模型说 Done，也做一次轻量 fallback merge 扫描：
                # 如果存在非常明显的 E2E 可合并 pair，它仍然有机会被执行。
                ok, new_edges, scan = self.try_fallback_merge_scan(
                    edges, taboo, locked, step,
                    trigger="model_done_or_no_pointer",
                    max_candidates=8,
                    report_top_k=4,
                )
                if ok:
                    edges = new_edges
                    ops.append(scan)
                    continue
                if scan.get("candidate_count", 0) > 0:
                    rejected.append(scan)
                break

            if act == "Delete":
                ok, new_edges, rep = self.try_delete(edges, idx)
                rep["step"], rep["model_debug"] = step, dbg

                if ok:
                    edges = new_edges
                    ops.append(rep)
                    continue

                # 先记录 Delete 被拒绝。
                rejected.append(rep)

                # 关键新增：Delete 被拒绝后，不马上进入下一轮 Delete，
                # 而是扫描所有 E2E / 近端点 pair，尝试最合理的 Merge。
                target_id = int(edges[idx]["id"])
                ok_m, merge_edges, merge_rep = self.try_fallback_merge_scan(
                    edges, taboo, locked,
                    step,
                    trigger=f"delete_rejected_target_{target_id}",
                    max_candidates=12,
                    report_top_k=6,
                )

                if ok_m:
                    merge_rep["model_debug"] = {
                        "mode": "fallback_after_delete_reject",
                        "original_model_action": "Delete",
                        "delete_target_id": target_id,
                    }
                    edges = merge_edges
                    ops.append(merge_rep)
                    continue

                if merge_rep.get("candidate_count", 0) > 0:
                    rejected.append(merge_rep)

                # fallback merge 也没成，才 lock 这个 Delete target，避免模型反复删同一条。
                locked.add(target_id)
                continue

            if act == "Merge":
                j, _ = self.find_partner(idx, edges, taboo, locked)
                if j is None:
                    target_id = int(edges[idx]["id"])
                    locked.add(target_id)
                    rejected.append({
                        "step": step,
                        "type": "Merge",
                        "old_id": target_id,
                        "accepted": False,
                        "reject_reason": "no partner",
                        "model_debug": dbg,
                    })
                    continue

                ok, new_edges, rep = self.try_merge(edges, idx, j)
                rep["step"], rep["model_debug"] = step, dbg
                if ok:
                    edges = new_edges
                    ops.append(rep)
                    continue

                rejected.append(rep)
                taboo.add(tuple(sorted([int(edges[idx]["id"]), int(edges[j]["id"])])))

                # 模型指出的 merge pair 失败后，也扫描其它近端点 pair。
                ok_m, merge_edges, scan = self.try_fallback_merge_scan(
                    edges, taboo, locked,
                    step,
                    trigger=f"model_merge_rejected_target_{int(edges[idx]['id'])}",
                    max_candidates=10,
                    report_top_k=5,
                )
                if ok_m:
                    scan["model_debug"] = {
                        "mode": "fallback_after_merge_reject",
                        "original_model_action": "Merge",
                        "original_model_target_id": int(edges[idx]["id"]),
                    }
                    edges = merge_edges
                    ops.append(scan)
                    continue
                if scan.get("candidate_count", 0) > 0:
                    rejected.append(scan)

                locked.add(int(edges[idx]["id"]))
                continue

            break

        cleaned = rebuild_bundle(bundle, edges, uid, ops)
        cleaned["clean_meta"]["rejected_ops"] = rejected
        cleaned["clean_meta"]["topology_text_before"] = before_topo["topology_text"]
        cleaned["clean_meta"]["topology_stats_before"] = before_topo["stats"]
        cleaned["clean_meta"]["model_status"] = self.status
        cleaned["clean_meta"]["changed"] = len(ops) > 0
        return cleaned


def rebuild_bundle(original, edges, uid, ops):
    b = copy.deepcopy(original)
    strokes = []
    ts = np.linspace(0, 1, TOPO_SAMPLE_N)
    for new_id, e in enumerate(edges, start=1):
        P = np.asarray(e.get("mother_bezier"), dtype=np.float32)
        if P.shape != (4,2):
            P, _, _ = fit_cubic(e["path"])
        c = cubic(P, ts)
        length = float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
        mn, mx = np.min(c, axis=0), np.max(c, axis=0)
        wc = edge_width_ctrl(e)
        strokes.append({"bezier_id":new_id,
                        "stroke_type":"closed" if np.linalg.norm(P[0]-P[3]) < TOPO_THRESH else "open",
                        "length":round(length,2),
                        "bbox":[round(float(mn[0]),1),round(float(mn[1]),1),round(float(mx[0]),1),round(float(mx[1]),1)],
                        "mother_bezier":P.astype(float).tolist(),
                        "width_bezier":[round(float(x), 3) for x in wc.tolist()]})
    b["strokes"] = strokes
    topo = compute_topo(b)
    b["topology_events"], b["cycles"] = topo["topology_events"], topo["cycles"]
    gi = b.setdefault("glyph_info", {})
    if isinstance(gi, dict):
        gi["stable_uid"] = uid
        gi["source_stable_uid"] = uid
        gi["clean_status"] = "cleaned"
        gi.setdefault("original_hex_key_before_clean", gi.get("hex_key", ""))
    b["clean_meta"] = {"schema_version":"pcg_good_stage1_refine_v1",
                       "source_stable_uid":uid,
                       "cleaned_pool_key":uid,
                       "cleaned_at":time.strftime("%Y-%m-%d %H:%M:%S"),
                       "source_model":"graph_editor_best.pth",
                       "accepted_ops":ops,
                       "original_stroke_count":len(original.get("strokes", [])),
                       "cleaned_stroke_count":len(strokes),
                       "topology_text_after":topo["topology_text"],
                       "topology_stats_after":topo["stats"],
                       "identity_note":"cleaned outer key = source_stable_uid; original good pool untouched."}
    b.setdefault("edit_history", [])
    if isinstance(b["edit_history"], list):
        b["edit_history"].append({"action":"PCG_GOOD_STAGE1_REFINE","source_stable_uid":uid,"ops":ops,
                                  "timestamp":time.strftime("%Y-%m-%d %H:%M:%S")})
    return b


# =============================================================================
# 5.5 Phase2 Topology Candidate Refiner
# =============================================================================

def find_phase2_pth(verbose=True):
    candidates = [
        os.path.join(DATASET_ANALYSE_DIR, PHASE2_MODEL_REL_DIR, PHASE2_MODEL_FILENAME),
        os.path.join(TOOL_DIR, "..", "..", "AI_VECTOR_ROUTER_With_topo", "ml_engine_phase2", PHASE2_MODEL_FILENAME),
        os.path.join(os.getcwd(), PHASE2_MODEL_FILENAME),
    ]
    seen = set()
    for p in candidates:
        p = os.path.abspath(p)
        if p in seen:
            continue
        seen.add(p)
        if verbose:
            print(f"[Phase2ModelPathCheck] exists={os.path.exists(p)} | {p}")
        if os.path.exists(p):
            return p
    return None

class Phase2Cfg:
    def __init__(self, d=None):
        d = d or {}
        self.max_strokes = int(d.get("max_strokes", 30))
        self.max_candidates = int(d.get("max_candidates", 64))
        self.canvas_norm = float(d.get("canvas_norm", 400.0))
        self.width_norm = float(d.get("width_norm", 20.0))
        self.delta_norm = float(d.get("delta_norm", 24.0))
        self.snap_radius = float(d.get("snap_radius", 12.0))
        self.t_attach_radius = float(d.get("t_attach_radius", 12.0))
        self.hidden_dim = int(d.get("hidden_dim", 256))
        self.num_heads = int(d.get("num_heads", 8))
        self.num_layers = int(d.get("num_layers", 4))
        self.dropout = float(d.get("dropout", 0.10))

PHASE2_REL_MAP = {"NONE": 0, "E2E": 1, "T": 2, "X": 3}
PHASE2_CAND_TYPE = {"SNAP": 0, "T_ATTACH": 1, "CONTROL_MOVE": 2}
PHASE2_CAND_INV = {v:k for k,v in PHASE2_CAND_TYPE.items()}

def phase2_endpoint_t(pidx: int) -> float:
    return 0.0 if int(pidx) == 0 else 1.0

def phase2_closest_t_on_curve(point: np.ndarray, P: np.ndarray, n: int = 96):
    ts = np.linspace(0, 1, n, dtype=np.float32)
    curve = cubic(np.asarray(P, dtype=np.float32), ts)
    d = np.linalg.norm(curve - point[None, :], axis=1)
    idx = int(np.argmin(d))
    return float(ts[idx]), curve[idx].astype(np.float32), float(d[idx])

def phase2_point_token_features(P: np.ndarray, W: np.ndarray, stroke_idx: int, point_idx: int, cfg: Phase2Cfg) -> np.ndarray:
    P = np.asarray(P, dtype=np.float32)
    W = np.asarray(W, dtype=np.float32).reshape(4)
    pt = P[point_idx]
    start, end = P[0], P[3]
    center = P.mean(axis=0)
    chord = end - start
    chord_len = np.linalg.norm(chord) + 1e-6
    t0 = deriv(P, 0.0); t0 = t0 / (np.linalg.norm(t0) + 1e-6)
    t1 = deriv(P, 1.0); t1 = t1 / (np.linalg.norm(t1) + 1e-6)

    role = np.zeros(4, dtype=np.float32)
    role[point_idx] = 1.0
    endpoint_flag = 1.0 if point_idx in (0, 3) else 0.0

    return np.concatenate([
        pt / cfg.canvas_norm,                  # 2
        start / cfg.canvas_norm,               # 2
        end / cfg.canvas_norm,                 # 2
        center / cfg.canvas_norm,              # 2
        np.array([chord_len / cfg.canvas_norm], dtype=np.float32),
        t0.astype(np.float32),
        t1.astype(np.float32),
        W / cfg.width_norm,
        np.array([stroke_idx / max(1, cfg.max_strokes-1)], dtype=np.float32),
        role,
        np.array([endpoint_flag], dtype=np.float32),
    ]).astype(np.float32)

def phase2_strokes_to_point_features(strokes: List[Dict[str, Any]], cfg: Phase2Cfg) -> np.ndarray:
    feats = []
    for si, s in enumerate(strokes):
        P = np.asarray(s.get("mother_bezier"), dtype=np.float32)
        W = np.asarray(s.get("width_bezier", [6,6,6,6]), dtype=np.float32).reshape(4)
        for pi in range(4):
            feats.append(phase2_point_token_features(P, W, si, pi, cfg))
    return np.asarray(feats, dtype=np.float32)

def phase2_candidate_geom(strokes, q_si, q_pi, cand_type, cfg: Phase2Cfg, host_si=-1, host_pi=-1, host_t=0.0):
    qP = np.asarray(strokes[q_si]["mother_bezier"], dtype=np.float32)
    q = qP[q_pi].astype(np.float32)

    host_exists = 0.0
    host_pt = np.zeros(2, dtype=np.float32)
    host_role = np.zeros(4, dtype=np.float32)
    dist = 0.0

    if cand_type == "SNAP" and host_si >= 0 and host_pi in (0, 3):
        hP = np.asarray(strokes[host_si]["mother_bezier"], dtype=np.float32)
        host_pt = hP[host_pi].astype(np.float32)
        host_role[host_pi] = 1.0
        host_exists = 1.0
        host_t = phase2_endpoint_t(host_pi)
        dist = float(np.linalg.norm(host_pt - q))

    elif cand_type == "T_ATTACH" and host_si >= 0:
        hP = np.asarray(strokes[host_si]["mother_bezier"], dtype=np.float32)
        if host_t is None:
            host_t, host_pt, dist = phase2_closest_t_on_curve(q, hP)
        else:
            host_pt = cubic(hP, np.asarray([host_t], dtype=np.float32))[0].astype(np.float32)
            dist = float(np.linalg.norm(host_pt - q))
        host_exists = 1.0

    elif cand_type == "CONTROL_MOVE":
        host_pt = q.copy()
        host_exists = 0.0
        host_t = 0.0
        dist = 0.0

    delta = host_pt - q
    q_role = np.zeros(4, dtype=np.float32)
    q_role[q_pi] = 1.0
    q_endpoint = 1.0 if q_pi in (0, 3) else 0.0

    type_onehot = np.zeros(3, dtype=np.float32)
    type_onehot[PHASE2_CAND_TYPE[cand_type]] = 1.0

    return np.concatenate([
        type_onehot,                                      # 3
        q / cfg.canvas_norm,                              # 2
        host_pt / cfg.canvas_norm,                        # 2
        delta / cfg.delta_norm,                           # 2
        np.array([dist / cfg.delta_norm, float(host_t)], dtype=np.float32), # 2
        q_role,                                           # 4
        host_role,                                        # 4
        np.array([q_endpoint, host_exists, q_si / max(1, cfg.max_strokes-1)], dtype=np.float32), # 3
    ]).astype(np.float32)  # 22 dim

class Phase2CandidateTopoModel(nn.Module if nn is not None else object):
    def __init__(self, point_dim: int, geom_dim: int, cfg: Phase2Cfg):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim

        self.point_embed = nn.Sequential(
            nn.Linear(point_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=cfg.num_heads,
            dim_feedforward=h * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)

        self.stroke_proj = nn.Sequential(nn.Linear(h, h), nn.LayerNorm(h), nn.GELU())

        self.type_emb = nn.Embedding(3, h)
        self.geom_mlp = nn.Sequential(nn.Linear(geom_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, h))

        self.cand_mlp = nn.Sequential(
            nn.Linear(h * 5, h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, h),
            nn.GELU(),
        )
        self.accept_head = nn.Linear(h, 1)
        self.delta_head = nn.Linear(h, 2)
        self.host_t_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1), nn.Sigmoid())

        self.rel_head = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, 4),
        )
        self.rel_t_head = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.GELU(),
            nn.Linear(h, 2),
            nn.Sigmoid(),
        )

    def _gather_tokens(self, tokens, idx):
        b, c = idx.shape
        h = tokens.shape[-1]
        idx_exp = idx.unsqueeze(-1).expand(b, c, h)
        return torch.gather(tokens, 1, idx_exp)

    def forward(self, batch):
        x = self.point_embed(batch["point_features"])
        pad = batch["point_mask"] == 0
        enc = self.encoder(x, src_key_padding_mask=pad)

        b, p, h = enc.shape
        s = self.cfg.max_strokes
        stroke = enc.reshape(b, s, 4, h).mean(dim=2)
        stroke = self.stroke_proj(stroke)

        si = stroke.unsqueeze(2).expand(b, s, s, h)
        sj = stroke.unsqueeze(1).expand(b, s, s, h)
        pair = torch.cat([si, sj], dim=-1)
        rel_logits = self.rel_head(pair)
        rel_t = self.rel_t_head(pair)

        qtok = self._gather_tokens(enc, batch["q_idx"])
        hptok = self._gather_tokens(enc, batch["hp_idx"])
        hstok = self._gather_tokens(stroke, batch["hs_idx"])
        typ = self.type_emb(batch["cand_type"])
        geom = self.geom_mlp(batch["cand_geom"])

        cand = torch.cat([qtok, hptok, hstok, typ, geom], dim=-1)
        ch = self.cand_mlp(cand)
        accept_logit = self.accept_head(ch).squeeze(-1)
        delta = self.delta_head(ch)
        host_t = self.host_t_head(ch).squeeze(-1)

        return {
            "rel_logits": rel_logits,
            "rel_t": rel_t,
            "accept_logit": accept_logit,
            "delta": delta,
            "host_t": host_t,
        }

def phase2_copy_bundle_with_strokes(bundle, strokes):
    b = copy.deepcopy(bundle)
    out = []
    for i, s in enumerate(strokes, start=1):
        ss = copy.deepcopy(s)
        ss["bezier_id"] = i
        P = np.asarray(ss["mother_bezier"], dtype=np.float32)
        ts = np.linspace(0, 1, TOPO_SAMPLE_N)
        c = cubic(P, ts)
        length = float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
        mn, mx = np.min(c, axis=0), np.max(c, axis=0)
        ss["stroke_type"] = "closed" if np.linalg.norm(P[0] - P[3]) < TOPO_THRESH else "open"
        ss["length"] = round(length, 2)
        ss["bbox"] = [round(float(mn[0]),1), round(float(mn[1]),1), round(float(mx[0]),1), round(float(mx[1]),1)]
        ss["mother_bezier"] = [[round(float(x),3), round(float(y),3)] for x,y in P.tolist()]
        if "width_bezier" in ss:
            ss["width_bezier"] = [round(float(x),3) for x in np.asarray(ss["width_bezier"], dtype=np.float32).reshape(4).tolist()]
        out.append(ss)
    b["strokes"] = out
    topo = compute_topo(b)
    b["topology_events"], b["cycles"] = topo["topology_events"], topo["cycles"]
    return b

def phase2_translate_endpoint_and_handle(strokes, q_si, q_pi, new_pt):
    st = copy.deepcopy(strokes)
    P = np.asarray(st[q_si]["mother_bezier"], dtype=np.float32)
    old = P[q_pi].copy()
    new_pt = np.asarray(new_pt, dtype=np.float32)
    delta = new_pt - old
    P[q_pi] = new_pt
    # 轻量保持切线：移动端点时同步平移相邻 handle。
    if q_pi == 0:
        P[1] = P[1] + delta
    elif q_pi == 3:
        P[2] = P[2] + delta
    else:
        P[q_pi] = P[q_pi] + delta
    st[q_si]["mother_bezier"] = P.astype(float).tolist()
    return st, delta

def phase2_translate_control_point(strokes, q_si, q_pi, delta):
    st = copy.deepcopy(strokes)
    P = np.asarray(st[q_si]["mother_bezier"], dtype=np.float32)
    P[q_pi] = P[q_pi] + np.asarray(delta, dtype=np.float32)
    st[q_si]["mother_bezier"] = P.astype(float).tolist()
    return st

class Phase2TopoRefiner:
    def __init__(self):
        self.model_path = find_phase2_pth(verbose=True)
        self.model = None
        self.device = None
        self.cfg = Phase2Cfg()
        self.threshold = PHASE2_MIN_ACCEPT_THRESHOLD
        self.status = ""

        if torch is None or nn is None:
            self.status = "torch unavailable; phase2 disabled."
            print("[Phase2Refiner]", self.status)
            return
        if not self.model_path:
            self.status = f"{PHASE2_MODEL_FILENAME} not found; phase2 disabled."
            print("[Phase2Refiner]", self.status)
            return

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # PyTorch >= 2.6 默认 torch.load(weights_only=True)，
            # 但我们保存的 phase2 checkpoint 里包含 config / threshold_sweep 等 numpy 标量元信息，
            # 需要显式 weights_only=False 才能完整读取。
            # 仅对你自己训练生成的可信 checkpoint 使用。
            try:
                ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
            except TypeError:
                # 兼容旧版 PyTorch：旧版本没有 weights_only 参数。
                ckpt = torch.load(self.model_path, map_location="cpu")
            cfg_dict = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
            self.cfg = Phase2Cfg(cfg_dict)
            point_dim = int(ckpt.get("point_dim", 23)) if isinstance(ckpt, dict) else 23
            geom_dim = int(ckpt.get("geom_dim", 22)) if isinstance(ckpt, dict) else 22
            self.threshold = max(PHASE2_MIN_ACCEPT_THRESHOLD, float(ckpt.get("best_threshold", PHASE2_MIN_ACCEPT_THRESHOLD)) if isinstance(ckpt, dict) else PHASE2_MIN_ACCEPT_THRESHOLD)

            self.model = Phase2CandidateTopoModel(point_dim, geom_dim, self.cfg).to(self.device)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            if isinstance(state, dict):
                state = {k.replace("module.", ""): v for k, v in state.items()}
            self.model.load_state_dict(state, strict=True)
            self.model.eval()

            self.status = f"loaded pth: {self.model_path} | threshold={self.threshold:.2f}"
            print("[Phase2Refiner]", self.status)
        except Exception as e:
            self.model = None
            self.status = "phase2 pth load failed: " + repr(e)
            print("[Phase2Refiner]", self.status)
            traceback.print_exc()

    def _strokes_ok(self, bundle):
        strokes = bundle.get("strokes", [])
        if not isinstance(strokes, list) or not strokes:
            return False
        if len(strokes) > self.cfg.max_strokes:
            return False
        for s in strokes:
            try:
                P = np.asarray(s.get("mother_bezier"), dtype=np.float32)
                W = np.asarray(s.get("width_bezier", [6,6,6,6]), dtype=np.float32).reshape(-1)
                if P.shape != (4,2) or len(W) != 4:
                    return False
            except Exception:
                return False
        return True

    def build_candidates(self, strokes):
        cfg = self.cfg
        n = len(strokes)
        endpoints = [(si, pi) for si in range(n) for pi in (0,3)]
        cands = []

        # SNAP: endpoint -> endpoint
        for qsi, qpi in endpoints:
            q = np.asarray(strokes[qsi]["mother_bezier"], dtype=np.float32)[qpi]
            local = []
            for hsi, hpi in endpoints:
                if hsi == qsi:
                    continue
                hp = np.asarray(strokes[hsi]["mother_bezier"], dtype=np.float32)[hpi]
                d = float(np.linalg.norm(hp - q))
                if 0.01 <= d <= cfg.snap_radius:
                    local.append((d, hsi, hpi))
            local.sort(key=lambda x: x[0])
            for d, hsi, hpi in local[:2]:
                cands.append({
                    "kind":"SNAP", "q_si":qsi, "q_pi":qpi, "host_si":hsi, "host_pi":hpi,
                    "host_t":phase2_endpoint_t(hpi), "dist":d,
                    "q_idx":qsi*4+qpi, "hp_idx":hsi*4+hpi, "hs_idx":hsi,
                    "cand_type":PHASE2_CAND_TYPE["SNAP"],
                    "geom":phase2_candidate_geom(strokes, qsi, qpi, "SNAP", cfg, hsi, hpi)
                })

        # T_ATTACH: endpoint -> host curve
        for qsi, qpi in endpoints:
            q = np.asarray(strokes[qsi]["mother_bezier"], dtype=np.float32)[qpi]
            local = []
            for hsi in range(n):
                if hsi == qsi:
                    continue
                hP = np.asarray(strokes[hsi]["mother_bezier"], dtype=np.float32)
                ht, hp, d = phase2_closest_t_on_curve(q, hP)
                if 0.03 < ht < 0.97 and 0.01 <= d <= cfg.t_attach_radius:
                    local.append((d, hsi, ht))
            local.sort(key=lambda x: x[0])
            for d, hsi, ht in local[:2]:
                cands.append({
                    "kind":"T_ATTACH", "q_si":qsi, "q_pi":qpi, "host_si":hsi, "host_pi":-1,
                    "host_t":ht, "dist":d,
                    "q_idx":qsi*4+qpi, "hp_idx":0, "hs_idx":hsi,
                    "cand_type":PHASE2_CAND_TYPE["T_ATTACH"],
                    "geom":phase2_candidate_geom(strokes, qsi, qpi, "T_ATTACH", cfg, hsi, -1, ht)
                })

        # CONTROL_MOVE：风险较大，只给模型少量机会，自动执行时需要更高阈值。
        for qsi in range(n):
            for qpi in range(4):
                cands.append({
                    "kind":"CONTROL_MOVE", "q_si":qsi, "q_pi":qpi, "host_si":-1, "host_pi":-1,
                    "host_t":0.0, "dist":0.0,
                    "q_idx":qsi*4+qpi, "hp_idx":0, "hs_idx":0,
                    "cand_type":PHASE2_CAND_TYPE["CONTROL_MOVE"],
                    "geom":phase2_candidate_geom(strokes, qsi, qpi, "CONTROL_MOVE", cfg)
                })

        return cands[:self.cfg.max_candidates]

    def predict_candidates(self, strokes, candidates):
        if self.model is None or not candidates:
            return []

        max_s = self.cfg.max_strokes
        max_p = max_s * 4
        max_c = self.cfg.max_candidates
        n = len(strokes)
        cur_p = n * 4

        point_feats = torch.zeros((1, max_p, 23), dtype=torch.float32)
        feats = phase2_strokes_to_point_features(strokes, self.cfg)
        point_feats[0, :len(feats), :] = torch.tensor(feats, dtype=torch.float32)
        point_mask = torch.zeros((1, max_p), dtype=torch.float32); point_mask[0, :cur_p] = 1.0
        stroke_mask = torch.zeros((1, max_s), dtype=torch.float32); stroke_mask[0, :n] = 1.0

        q_idx = torch.zeros((1, max_c), dtype=torch.long)
        hp_idx = torch.zeros((1, max_c), dtype=torch.long)
        hs_idx = torch.zeros((1, max_c), dtype=torch.long)
        cand_type = torch.zeros((1, max_c), dtype=torch.long)
        cand_geom = torch.zeros((1, max_c, 22), dtype=torch.float32)
        cand_mask = torch.zeros((1, max_c), dtype=torch.float32)

        for i, c in enumerate(candidates[:max_c]):
            q_idx[0, i] = int(c["q_idx"])
            hp_idx[0, i] = int(c["hp_idx"])
            hs_idx[0, i] = int(c["hs_idx"])
            cand_type[0, i] = int(c["cand_type"])
            cand_geom[0, i] = torch.tensor(c["geom"], dtype=torch.float32)
            cand_mask[0, i] = 1.0

        batch = {
            "point_features": point_feats.to(self.device),
            "point_mask": point_mask.to(self.device),
            "stroke_mask": stroke_mask.to(self.device),
            "q_idx": q_idx.to(self.device),
            "hp_idx": hp_idx.to(self.device),
            "hs_idx": hs_idx.to(self.device),
            "cand_type": cand_type.to(self.device),
            "cand_geom": cand_geom.to(self.device),
        }

        with torch.no_grad():
            out = self.model(batch)
            probs = torch.sigmoid(out["accept_logit"])[0].detach().cpu().numpy()
            deltas = out["delta"][0].detach().cpu().numpy()
            host_ts = out["host_t"][0].detach().cpu().numpy()

        scored = []
        for i, c in enumerate(candidates[:max_c]):
            cc = copy.deepcopy(c)
            cc["prob"] = float(probs[i])
            cc["pred_delta"] = (deltas[i] * self.cfg.delta_norm).astype(np.float32)
            cc["pred_host_t"] = float(host_ts[i])
            scored.append(cc)
        scored.sort(key=lambda x: x["prob"], reverse=True)
        return scored

    def _outline_iou(self, before_bundle, after_bundle):
        try:
            e1 = bundle_to_edges(before_bundle, sample_n=64)
            e2 = bundle_to_edges(after_bundle, sample_n=64)
            shared_view = mask_view([e1, e2], size=MASK_SIZE)
            m1 = render_mask(e1, view=shared_view)
            m2 = render_mask(e2, view=shared_view)
            return mask_iou(m1, m2)
        except Exception:
            traceback.print_exc()
            return 0.0

    def _guard(self, before_bundle, candidate_bundle, move_norm):
        tb = compute_topo(before_bundle)["stats"]
        ta = compute_topo(candidate_bundle)["stats"]
        rep = {
            "outline_iou": round(float(self._outline_iou(before_bundle, candidate_bundle)), 4),
            "move_norm": round(float(move_norm), 3),
            "topology_before": tb,
            "topology_after": ta,
        }

        if move_norm > PHASE2_MAX_MOVE:
            rep["reject_reason"] = f"move too large: {move_norm:.3f} > {PHASE2_MAX_MOVE}"
            return False, rep
        if rep["outline_iou"] < PHASE2_MIN_OUTLINE_IOU:
            rep["reject_reason"] = f"outline_iou {rep['outline_iou']:.4f} < threshold {PHASE2_MIN_OUTLINE_IOU}"
            return False, rep
        if ta.get("x_count", 0) > tb.get("x_count", 0) + PHASE2_MAX_X_INCREASE:
            rep["reject_reason"] = "x_count increased"
            return False, rep
        if (not PHASE2_ALLOW_CC_INCREASE) and ta.get("connected_components", 0) > tb.get("connected_components", 0):
            rep["reject_reason"] = "connected_components increased"
            return False, rep
        if (not PHASE2_ALLOW_CYCLE_DECREASE) and ta.get("cycle_count", 0) < tb.get("cycle_count", 0):
            rep["reject_reason"] = "cycle_count decreased"
            return False, rep
        return True, rep

    def refine(self, bundle, uid):
        before = copy.deepcopy(bundle)
        before_topo = compute_topo(before)
        if self.model is None:
            b = copy.deepcopy(bundle)
            cm = b.setdefault("clean_meta", {})
            cm["stage2_model_status"] = self.status
            cm["stage2_changed"] = False
            cm["stage2_accepted_ops"] = []
            cm["stage2_rejected_ops"] = [{"type":"Phase2Disabled","reason":self.status}]
            return b
        if not self._strokes_ok(bundle):
            b = copy.deepcopy(bundle)
            cm = b.setdefault("clean_meta", {})
            cm["stage2_model_status"] = self.status
            cm["stage2_changed"] = False
            cm["stage2_accepted_ops"] = []
            cm["stage2_rejected_ops"] = [{"type":"InvalidStrokes","reason":"strokes missing or exceed max_strokes"}]
            return b

        strokes = copy.deepcopy(bundle.get("strokes", []))
        candidates = self.build_candidates(strokes)
        scored = self.predict_candidates(strokes, candidates)

        accepted, rejected = [], []
        used_points = set()
        cur_bundle = phase2_copy_bundle_with_strokes(bundle, strokes)

        for rank, cand in enumerate(scored):
            if len(accepted) >= PHASE2_MAX_OPS:
                break
            kind = cand["kind"]
            qkey = (int(cand["q_si"]), int(cand["q_pi"]))
            prob = float(cand["prob"])

            # CONTROL_MOVE 误伤风险大，阈值更高。
            local_threshold = max(self.threshold, 0.90) if kind == "CONTROL_MOVE" else self.threshold
            if prob < local_threshold:
                continue
            if qkey in used_points:
                continue

            rep = {
                "type": kind,
                "accepted": False,
                "rank": rank,
                "prob": round(prob, 4),
                "threshold": round(local_threshold, 3),
                "q_stroke": int(cand["q_si"] + 1),
                "q_point": "P0" if cand["q_pi"] == 0 else ("P3" if cand["q_pi"] == 3 else f"P{cand['q_pi']}"),
                "distance": round(float(cand.get("dist", 0.0)), 3),
            }

            try:
                before_op_bundle = phase2_copy_bundle_with_strokes(bundle, strokes)

                if kind == "SNAP":
                    hsi, hpi = int(cand["host_si"]), int(cand["host_pi"])
                    target = np.asarray(strokes[hsi]["mother_bezier"], dtype=np.float32)[hpi]
                    new_strokes, delta = phase2_translate_endpoint_and_handle(strokes, int(cand["q_si"]), int(cand["q_pi"]), target)
                    rep.update({
                        "host_stroke": hsi + 1,
                        "host_point": "P0" if hpi == 0 else "P3",
                        "before": np.asarray(strokes[int(cand["q_si"])]["mother_bezier"], dtype=np.float32)[int(cand["q_pi"])].round(3).tolist(),
                        "after": target.round(3).tolist(),
                    })

                elif kind == "T_ATTACH":
                    hsi = int(cand["host_si"])
                    # 使用模型预测 host_t，但限制在曲线内部，避免误吸到端点。
                    ht = float(np.clip(cand.get("pred_host_t", cand.get("host_t", 0.5)), 0.03, 0.97))
                    hP = np.asarray(strokes[hsi]["mother_bezier"], dtype=np.float32)
                    target = cubic(hP, np.asarray([ht], dtype=np.float32))[0]
                    new_strokes, delta = phase2_translate_endpoint_and_handle(strokes, int(cand["q_si"]), int(cand["q_pi"]), target)
                    rep.update({
                        "host_stroke": hsi + 1,
                        "host_t": round(ht, 4),
                        "before": np.asarray(strokes[int(cand["q_si"])]["mother_bezier"], dtype=np.float32)[int(cand["q_pi"])].round(3).tolist(),
                        "after": target.round(3).tolist(),
                    })

                elif kind == "CONTROL_MOVE":
                    delta = np.asarray(cand["pred_delta"], dtype=np.float32)
                    if np.linalg.norm(delta) > PHASE2_MAX_MOVE:
                        delta = delta / (np.linalg.norm(delta) + 1e-6) * PHASE2_MAX_MOVE
                    new_strokes = phase2_translate_control_point(strokes, int(cand["q_si"]), int(cand["q_pi"]), delta)
                    after_pt = np.asarray(new_strokes[int(cand["q_si"])]["mother_bezier"], dtype=np.float32)[int(cand["q_pi"])]
                    rep.update({
                        "before": np.asarray(strokes[int(cand["q_si"])]["mother_bezier"], dtype=np.float32)[int(cand["q_pi"])].round(3).tolist(),
                        "after": after_pt.round(3).tolist(),
                    })
                else:
                    continue

                cand_bundle = phase2_copy_bundle_with_strokes(bundle, new_strokes)
                ok, guard_rep = self._guard(before_op_bundle, cand_bundle, float(np.linalg.norm(delta)))
                rep.update(guard_rep)

                if ok:
                    rep["accepted"] = True
                    strokes = new_strokes
                    cur_bundle = cand_bundle
                    accepted.append(rep)
                    used_points.add(qkey)
                else:
                    rejected.append(rep)

            except Exception as e:
                rep["reject_reason"] = "exception: " + repr(e)
                rejected.append(rep)
                traceback.print_exc()

        out = phase2_copy_bundle_with_strokes(bundle, strokes)
        topo_after = compute_topo(out)
        cm = out.setdefault("clean_meta", {})
        cm["schema_version"] = "pcg_good_stage1_stage2_refine_v1"
        cm["source_stable_uid"] = uid
        cm["cleaned_pool_key"] = uid
        cm["cleaned_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        cm["stage2_model_status"] = self.status
        cm["stage2_model_path"] = self.model_path
        cm["stage2_threshold"] = self.threshold
        cm["stage2_changed"] = len(accepted) > 0
        cm["stage2_candidate_count"] = len(candidates)
        cm["stage2_accepted_ops"] = accepted
        cm["stage2_rejected_ops"] = rejected[:60]
        cm["topology_text_before_stage2"] = before_topo["topology_text"]
        cm["topology_stats_before_stage2"] = before_topo["stats"]
        cm["topology_text_after_stage2"] = topo_after["topology_text"]
        cm["topology_stats_after_stage2"] = topo_after["stats"]

        out.setdefault("edit_history", [])
        if isinstance(out["edit_history"], list):
            out["edit_history"].append({
                "action":"PCG_GOOD_STAGE2_TOPO_REFINE",
                "source_stable_uid":uid,
                "ops":accepted,
                "auto":True,
                "source":"clean_good_stage2_candidate_v4",
                "timestamp":time.strftime("%Y-%m-%d %H:%M:%S")
            })
        return out


# =============================================================================
# 6. GUI
# =============================================================================

class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, height=700):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, borderwidth=0, height=height)
        self.vs = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.win = self.canvas.create_window((0,0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vs.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vs.pack(side="right", fill="y")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.win, width=e.width))
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("PCG Good Cleaner — Stage1 + Stage2 Refiner")
        self.root.geometry("1500x950")
        self.pool_root = tk.StringVar(value=POOL_ROOT_DEFAULT)
        self.max_mb = tk.IntVar(value=DEFAULT_MAX_JSON_MB)
        self.preview_px = tk.IntVar(value=PREVIEW_SIZE)
        self.page_rows = tk.IntVar(value=PAGE_ROWS_DEFAULT)
        self.page_cols = tk.IntVar(value=PAGE_COLS_DEFAULT)
        self.page_idx = 0
        self.status = tk.StringVar(value="Ready")
        self.refiner = Stage1Refiner()
        self.phase2_refiner = Phase2TopoRefiner()
        self.good_raw, self.good_by_uid, self.uid_outer, self.cleaned, self.pending = {}, {}, {}, {}, []
        self.selected_uid = None
        self.photos = []
        self._ui()
        self.status.set("UI ready. Loading pools after first paint...")
        self.root.after(100, self.refresh)

    def _ui(self):
        top = ttk.Frame(self.root); top.pack(side="top", fill="x", padx=8, pady=6)
        ttk.Label(top, text="Pool Root:").pack(side="left")
        ttk.Entry(top, textvariable=self.pool_root, width=90).pack(side="left", padx=4)
        ttk.Button(top, text="Browse", command=self.browse).pack(side="left")
        ttk.Label(top, text="Preview:").pack(side="left", padx=(12,2))
        ttk.Spinbox(top, from_=100, to=260, textvariable=self.preview_px, width=6).pack(side="left")
        ttk.Label(top, text="Rows:").pack(side="left", padx=(12,2))
        ttk.Spinbox(top, from_=1, to=20, textvariable=self.page_rows, width=4).pack(side="left")
        ttk.Label(top, text="Cols:").pack(side="left", padx=(8,2))
        ttk.Spinbox(top, from_=1, to=10, textvariable=self.page_cols, width=4).pack(side="left")
        ttk.Label(top, text="JSON MB:").pack(side="left", padx=(12,2))
        ttk.Spinbox(top, from_=10, to=95, textvariable=self.max_mb, width=6).pack(side="left")
        row = ttk.Frame(self.root); row.pack(side="top", fill="x", padx=8, pady=4)
        ttk.Button(row, text="Refresh Pools", command=self.refresh).pack(side="left", padx=3)
        ttk.Button(row, text="Prev Page", command=self.prev_page).pack(side="left", padx=3)
        ttk.Button(row, text="Next Page", command=self.next_page).pack(side="left", padx=3)
        ttk.Button(row, text="Preview Clean Selected", command=self.preview_selected).pack(side="left", padx=3)
        ttk.Button(row, text="Preview Recursive Selected", command=self.preview_selected).pack(side="left", padx=3)
        ttk.Button(row, text="Open Cleaned Preview Window", command=self.open_cleaned).pack(side="left", padx=3)
        ttk.Button(row, text="Write Cleaned Pool", command=lambda: self.write_cleaned(True)).pack(side="left", padx=3)
        ttk.Label(row, textvariable=self.status, foreground="blue").pack(side="left", padx=12)
        note = ttk.LabelFrame(self.root, text="规则")
        note.pack(side="top", fill="x", padx=8, pady=4)
        ttk.Label(note, text="待 clean = good 中 stable_uid 不在 cleaned 中；保存只写 cleaned/，不删除/覆盖 good；预览为 原图 → 一阶段 clean → 二阶段 topo clean → 递归 clean。").pack(side="left", padx=8, pady=4)
        self.scroll = ScrollableFrame(self.root, height=760)
        self.scroll.pack(fill="both", expand=True, padx=8, pady=6)

    def browse(self):
        d = filedialog.askdirectory(initialdir=self.pool_root.get() or TOOL_DIR)
        if d:
            self.pool_root.set(d); self.refresh()

    def clear(self):
        for w in self.scroll.inner.winfo_children(): w.destroy()
        self.photos.clear()

    def refresh(self):
        pr = self.pool_root.get().strip() or POOL_ROOT_DEFAULT
        ensure_dir(os.path.join(pr, GOOD_DIR)); ensure_dir(os.path.join(pr, CLEANED_DIR)); ensure_dir(os.path.join(pr, REPORT_DIR))
        self.good_raw = load_pool(pr, "good")
        self.cleaned = load_pool(pr, "cleaned")
        self.good_by_uid, self.uid_outer = {}, {}
        dup = 0
        for outer, b in self.good_raw.items():
            if not isinstance(b, dict): continue
            uid = stable_uid(b, outer)
            if uid in self.good_by_uid:
                dup += 1; continue
            bb = copy.deepcopy(b)
            bb.setdefault("pcg_identity", {})["stable_uid"] = uid
            bb["pcg_identity"]["outer_key_at_read"] = str(outer)
            self.good_by_uid[uid] = bb
            self.uid_outer[uid] = str(outer)
        cleaned_uids = set()
        for k,b in self.cleaned.items():
            if isinstance(b, dict):
                cm = b.get("clean_meta", {})
                cleaned_uids.add(str(cm.get("source_stable_uid", k)) if isinstance(cm, dict) else str(k))
            else:
                cleaned_uids.add(str(k))
        self.pending = [u for u in sorted(self.good_by_uid.keys()) if u not in cleaned_uids]
        self.page_idx = 0
        page_size = max(1, int(self.page_rows.get()) * int(self.page_cols.get()))
        total_pages = max(1, math.ceil(len(self.pending) / page_size))
        self.status.set(
            f"Good={len(self.good_by_uid)} | Cleaned={len(cleaned_uids)} | Pending={len(self.pending)} "
            f"| Page=1/{total_pages} size={page_size} | dup_hidden={dup} | Stage1: {self.refiner.status} | Stage2: {self.phase2_refiner.status}"
        )
        self.render_grid()

    def _page_size(self):
        return max(1, int(self.page_rows.get()) * int(self.page_cols.get()))

    def _total_pages(self):
        return max(1, math.ceil(len(self.pending) / self._page_size()))

    def prev_page(self):
        self.page_idx = max(0, self.page_idx - 1)
        self.render_grid()

    def next_page(self):
        self.page_idx = min(self._total_pages() - 1, self.page_idx + 1)
        self.render_grid()

    def render_grid(self):
        self.clear()
        size = int(self.preview_px.get())
        rows = max(1, int(self.page_rows.get()))
        cols = max(1, int(self.page_cols.get()))
        page_size = rows * cols
        total_pages = self._total_pages()
        self.page_idx = max(0, min(self.page_idx, total_pages - 1))

        start_i = self.page_idx * page_size
        end_i = min(len(self.pending), start_i + page_size)
        visible = self.pending[start_i:end_i]

        info = ttk.Label(
            self.scroll.inner,
            text=(
                f"Page {self.page_idx + 1}/{total_pages} | showing pending[{start_i}:{end_i}] "
                f"of {len(self.pending)} | grid={rows}x{cols}"
            ),
            foreground="blue",
        )
        info.grid(row=0, column=0, columnspan=cols, sticky="w", padx=8, pady=6)

        for local_idx, uid in enumerate(visible):
            idx = start_i + local_idx
            b = self.good_by_uid[uid]
            r,c = divmod(local_idx, cols)
            frame = ttk.Frame(self.scroll.inner, relief="groove", borderwidth=2)
            frame.grid(row=r + 1, column=c, padx=8, pady=8, sticky="n")
            img = render_bundle(b, size)
            ph = ImageTk.PhotoImage(img); self.photos.append(ph)
            ttk.Label(frame, image=ph).pack(side="top", padx=4, pady=4)
            topo = compute_topo(b)
            gi = glyph_info(b)
            txt = f"{idx:04d} outer={self.uid_outer.get(uid,'')}\nhex={gi.get('hex_key', gi.get('unicode_hex',''))}\n{uid}\nS={topo['stats']['stroke_count']} CC={topo['stats']['connected_components']} Cy={topo['stats']['cycle_count']} X={topo['stats']['x_count']}\n{topo['topology_text'][:80]}"
            ttk.Label(frame, text=txt, font=("Consolas",8), justify="center", wraplength=size*2+50).pack(side="top")
            ttk.Button(frame, text="Select", command=lambda u=uid:self.select(u)).pack(side="left", padx=4, pady=4)
            ttk.Button(frame, text="Clean Preview", command=lambda u=uid:self.preview_uid(u)).pack(side="left", padx=4, pady=4)
            ttk.Button(frame, text="Recursive", command=lambda u=uid:self.preview_uid(u)).pack(side="left", padx=4, pady=4)

        self.status.set(
            f"Good={len(self.good_by_uid)} | Cleaned={len(self.cleaned)} | Pending={len(self.pending)} "
            f"| Page={self.page_idx + 1}/{total_pages} size={page_size} | Stage1: {self.refiner.status} | Stage2: {self.phase2_refiner.status}"
        )

    def select(self, uid):
        self.selected_uid = uid
        self.status.set("Selected: " + uid)

    def preview_selected(self):
        if not self.selected_uid:
            messagebox.showwarning("No selection", "请先选择一个待 clean 的 Good 样本。")
            return
        self.preview_uid(self.selected_uid)

    def recursive_stage1_stage2_clean(self, uid, original):
        """
        递归清理：
            cur -> Stage1 clean -> Stage2 topo clean -> next iteration

        停止条件：
            某一轮 stage1_changed=False 且 stage2_changed=False
        """
        cur = copy.deepcopy(original)
        iterations = []
        stopped_reason = "max_iters_reached"

        for it in range(1, RECURSIVE_CLEAN_MAX_ITERS + 1):
            before_iter = copy.deepcopy(cur)

            s1 = self.refiner.refine(before_iter, uid)
            s1_meta = s1.get("clean_meta", {}) if isinstance(s1, dict) else {}
            s1_changed = bool(s1_meta.get("changed", False))

            s2 = self.phase2_refiner.refine(s1, uid)
            s2_meta = s2.get("clean_meta", {}) if isinstance(s2, dict) else {}
            s2_changed = bool(s2_meta.get("stage2_changed", False))

            iter_topo_before = compute_topo(before_iter)
            iter_topo_after = compute_topo(s2)

            iterations.append({
                "iter": it,
                "stage1_changed": s1_changed,
                "stage2_changed": s2_changed,
                "stage1_ops": len(s1_meta.get("accepted_ops", [])),
                "stage2_ops": len(s2_meta.get("stage2_accepted_ops", [])),
                "stage1_accepted_ops": copy.deepcopy(s1_meta.get("accepted_ops", [])),
                "stage2_accepted_ops": copy.deepcopy(s2_meta.get("stage2_accepted_ops", [])),
                "stage1_rejected_count": len(s1_meta.get("rejected_ops", [])),
                "stage2_rejected_count": len(s2_meta.get("stage2_rejected_ops", [])),
                "topology_before": iter_topo_before.get("stats", {}),
                "topology_after": iter_topo_after.get("stats", {}),
            })

            cur = s2

            if (not s1_changed) and (not s2_changed):
                stopped_reason = "converged_no_stage1_or_stage2_change"
                break

        out = copy.deepcopy(cur)
        topo_final = compute_topo(out)
        cm = out.setdefault("clean_meta", {})
        cm["schema_version"] = "pcg_good_stage1_stage2_recursive_refine_v1"
        cm["source_stable_uid"] = uid
        cm["cleaned_pool_key"] = uid
        cm["recursive_clean"] = True
        cm["recursive_max_iters"] = RECURSIVE_CLEAN_MAX_ITERS
        cm["recursive_iter_count"] = len(iterations)
        cm["recursive_stopped_reason"] = stopped_reason
        cm["recursive_iterations"] = iterations
        cm["recursive_changed"] = any(x.get("stage1_changed") or x.get("stage2_changed") for x in iterations)
        cm["topology_text_after_recursive"] = topo_final["topology_text"]
        cm["topology_stats_after_recursive"] = topo_final["stats"]
        cm["cleaned_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        out.setdefault("edit_history", [])
        if isinstance(out["edit_history"], list):
            out["edit_history"].append({
                "action": "PCG_GOOD_RECURSIVE_STAGE1_STAGE2_CLEAN",
                "source_stable_uid": uid,
                "iterations": iterations,
                "stopped_reason": stopped_reason,
                "auto": True,
                "source": "clean_good_recursive_stage1_stage2",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
        return out

    def preview_uid(self, uid):
        if uid not in self.good_by_uid:
            messagebox.showerror("Missing", uid); return
        try:
            original = self.good_by_uid[uid]
            stage1 = self.refiner.refine(original, uid)
            stage2 = self.phase2_refiner.refine(stage1, uid)
            recursive = self.recursive_stage1_stage2_clean(uid, original)
            self.before_stage1_stage2_recursive(uid, original, stage1, stage2, recursive)
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Clean Preview Failed", str(e))

    def _panel_text(self, title, bundle):
        topo = compute_topo(bundle)
        cm = bundle.get("clean_meta", {}) if isinstance(bundle, dict) else {}
        txt = f"Stats:\\n{json.dumps(topo['stats'], ensure_ascii=False, indent=2)}\\n\\nTopology:\\n{topo['topology_text']}\\n"
        if title.startswith("一阶段"):
            txt += "\\nStage1 Accepted ops:\\n" + json.dumps(cm.get("accepted_ops", []), ensure_ascii=False, indent=2)
            txt += "\\n\\nStage1 Rejected ops:\\n" + json.dumps(cm.get("rejected_ops", []), ensure_ascii=False, indent=2)
            txt += f"\\n\\nChanged={cm.get('changed')} | Stage1 Model={cm.get('model_status')}"
        elif title.startswith("二阶段"):
            txt += "\\nStage1 Accepted ops:\\n" + json.dumps(cm.get("accepted_ops", []), ensure_ascii=False, indent=2)
            txt += "\\n\\nStage2 Accepted ops:\\n" + json.dumps(cm.get("stage2_accepted_ops", []), ensure_ascii=False, indent=2)
            txt += "\\n\\nStage2 Rejected ops:\\n" + json.dumps(cm.get("stage2_rejected_ops", []), ensure_ascii=False, indent=2)
            txt += f"\\n\\nStage1 Changed={cm.get('changed')} | Stage2 Changed={cm.get('stage2_changed')}"
            txt += f"\\nStage2 Candidate Count={cm.get('stage2_candidate_count')} | Threshold={cm.get('stage2_threshold')}"
            txt += f"\\nStage2 Model={cm.get('stage2_model_status')}"
        elif title.startswith("递归"):
            txt += "\\nRecursive Summary:\\n" + json.dumps({
                "recursive_changed": cm.get("recursive_changed"),
                "recursive_iter_count": cm.get("recursive_iter_count"),
                "recursive_stopped_reason": cm.get("recursive_stopped_reason"),
                "recursive_max_iters": cm.get("recursive_max_iters"),
            }, ensure_ascii=False, indent=2)
            txt += "\\n\\nRecursive Iterations:\\n" + json.dumps(cm.get("recursive_iterations", []), ensure_ascii=False, indent=2)
            txt += f"\\n\\nStage2 Model={cm.get('stage2_model_status')}"
        return txt

    def before_stage1_stage2_recursive(self, uid, original, stage1, stage2, recursive):
        win = tk.Toplevel(self.root)
        win.title("Clean Preview — Original → Stage1 → Stage2 → Recursive — " + uid)
        win.geometry("1960x940")

        top = ttk.Frame(win); top.pack(side="top", fill="x", padx=8, pady=6)
        ttk.Label(top, text=uid, font=("Consolas",10)).pack(side="left")

        ttk.Button(top, text="保存原图为 cleaned", command=lambda:self.accept(uid, original, win, variant="original")).pack(side="right", padx=4)
        ttk.Button(top, text="保存递归 cleaned 为 cleaned", command=lambda:self.accept(uid, recursive, win, variant="recursive")).pack(side="right", padx=4)
        ttk.Button(top, text="保存二阶段 cleaned 为 cleaned", command=lambda:self.accept(uid, stage2, win, variant="stage2")).pack(side="right", padx=4)
        ttk.Button(top, text="保存一阶段 clean 为 cleaned", command=lambda:self.accept(uid, stage1, win, variant="stage1")).pack(side="right", padx=4)
        ttk.Button(top, text="Reject / Close", command=win.destroy).pack(side="right", padx=4)

        body = ttk.Frame(win); body.pack(fill="both", expand=True, padx=8, pady=6)
        size = 190
        p0 = ImageTk.PhotoImage(render_bundle(original, size))
        p1 = ImageTk.PhotoImage(render_bundle(stage1, size))
        p2 = ImageTk.PhotoImage(render_bundle(stage2, size))
        p3 = ImageTk.PhotoImage(render_bundle(recursive, size))
        win._photos = [p0, p1, p2, p3]

        panels = [
            ("原图 Original", original, p0),
            ("一阶段 clean Stage1", stage1, p1),
            ("二阶段 topology clean Stage2", stage2, p2),
            ("递归 clean Recursive", recursive, p3),
        ]
        for title, bundle, photo in panels:
            f = ttk.LabelFrame(body, text=title); f.pack(side="left", fill="both", expand=True, padx=4, pady=6)
            ttk.Label(f, image=photo).pack(side="top", padx=6, pady=6)
            t = tk.Text(f, height=28, width=50, font=("Consolas", 8))
            t.pack(fill="both", expand=True, padx=6, pady=6)
            t.insert("1.0", self._panel_text(title, bundle))
            t.configure(state="disabled")

    def _prepare_cleaned_bundle(self, uid, bundle, variant):
        bb = copy.deepcopy(bundle)
        topo = compute_topo(bb)
        cm = bb.setdefault("clean_meta", {})
        cm["schema_version"] = "pcg_good_stage1_stage2_recursive_refine_v1"
        cm["source_stable_uid"] = uid
        cm["cleaned_pool_key"] = uid
        cm["cleaned_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        cm["saved_variant"] = variant
        cm["topology_text_saved"] = topo["topology_text"]
        cm["topology_stats_saved"] = topo["stats"]

        if variant == "original":
            cm.setdefault("accepted_ops", [])
            cm.setdefault("rejected_ops", [])
            cm["changed"] = False
            cm["stage2_changed"] = False
            cm["recursive_changed"] = False
            cm["saved_note"] = "Original good saved to cleaned without Stage1/Stage2/recursive modification."
        elif variant == "stage1":
            cm.setdefault("stage2_accepted_ops", [])
            cm.setdefault("stage2_rejected_ops", [])
            cm["stage2_changed"] = False
            cm["recursive_changed"] = False
            cm["saved_note"] = "Stage1 clean result saved to cleaned."
        elif variant == "stage2":
            cm["recursive_changed"] = False
            cm["saved_note"] = "Stage1 + Stage2 topology clean result saved to cleaned."
        elif variant == "recursive":
            cm["saved_note"] = "Recursive Stage1->Stage2 clean result saved to cleaned."

        gi = bb.setdefault("glyph_info", {})
        if isinstance(gi, dict):
            gi["stable_uid"] = uid
            gi["source_stable_uid"] = uid
            gi["clean_status"] = "cleaned"
            gi["clean_saved_variant"] = variant
            gi.setdefault("original_hex_key_before_clean", gi.get("hex_key", ""))

        return bb

    def accept(self, uid, bundle, win=None, variant="recursive"):
        bb = self._prepare_cleaned_bundle(uid, bundle, variant)
        self.cleaned[uid] = bb
        self.write_cleaned(False)
        if win: win.destroy()
        self.refresh()
        messagebox.showinfo("Accepted", f"已写入 cleaned：\\n{uid}\\n\\n保存版本：{variant}\\n原 good 未删除/未覆盖。")

    def write_cleaned(self, show=True):
        pr = self.pool_root.get().strip() or POOL_ROOT_DEFAULT
        files = write_cleaned(pr, self.cleaned, int(self.max_mb.get()))
        report = {"schema_version":"pcg_good_cleaned_manifest_v1",
                  "created_at":time.strftime("%Y-%m-%d %H:%M:%S"),
                  "cleaned_count":len(self.cleaned),
                  "files":files,
                  "key_rule":"cleaned outer key = source_stable_uid; original good untouched.",
                  "max_json_mb":int(self.max_mb.get())}
        save_json(report, os.path.join(pr, REPORT_DIR, f"{REPORT_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}.json"))
        self.status.set(f"Wrote cleaned: items={len(self.cleaned)}, files={len(files)}")
        if show:
            messagebox.showinfo("Write Cleaned", f"cleaned={len(self.cleaned)}\nfiles={len(files)}")

    def open_cleaned(self):
        win = tk.Toplevel(self.root); win.title("Cleaned Good Preview"); win.geometry("1450x900")
        top = ttk.Frame(win); top.pack(side="top", fill="x", padx=8, pady=6)
        ttk.Label(top, text=f"Cleaned count: {len(self.cleaned)}").pack(side="left")
        ttk.Button(top, text="Refresh", command=lambda:(win.destroy(), self.refresh(), self.open_cleaned())).pack(side="left", padx=8)
        ttk.Button(top, text="Close", command=win.destroy).pack(side="right")
        scroll = ScrollableFrame(win, height=820); scroll.pack(fill="both", expand=True, padx=8, pady=6)
        photos=[]; win._photos=photos
        size=int(self.preview_px.get()); cols=4
        def remove(uid):
            if uid not in self.cleaned: return
            if not messagebox.askyesno("Delete cleaned result", f"删除 cleaned 结果？\n\n{uid}\n\n原 good 不会被删除，样本会回到待 clean。"):
                return
            self.cleaned.pop(uid, None)
            self.write_cleaned(False)
            win.destroy(); self.refresh(); self.open_cleaned()
        for idx,(uid,b) in enumerate(sorted(self.cleaned.items(), key=lambda kv:kv[0])):
            r,c = divmod(idx, cols)
            frame = ttk.Frame(scroll.inner, relief="groove", borderwidth=2)
            frame.grid(row=r, column=c, padx=8, pady=8, sticky="n")
            ph = ImageTk.PhotoImage(render_bundle(b, size)); photos.append(ph)
            ttk.Label(frame, image=ph).pack(side="top", padx=4, pady=4)
            topo=compute_topo(b); cm=b.get("clean_meta", {})
            txt=f"{idx:04d}\n{uid}\nS={topo['stats']['stroke_count']} CC={topo['stats']['connected_components']} Cy={topo['stats']['cycle_count']} X={topo['stats']['x_count']}\nvariant={cm.get('saved_variant')} s1={cm.get('changed')} s2={cm.get('stage2_changed')} rec={cm.get('recursive_changed')} ops={len(cm.get('accepted_ops', []))}+{len(cm.get('stage2_accepted_ops', []))}\n{topo['topology_text'][:90]}"
            ttk.Label(frame, text=txt, font=("Consolas",8), justify="center", wraplength=size*2+50).pack(side="top")
            ttk.Button(frame, text="Delete Cleaned Result", command=lambda u=uid:remove(u)).pack(side="top", pady=4)

def main():
    ensure_dir(POOL_ROOT_DEFAULT)
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
