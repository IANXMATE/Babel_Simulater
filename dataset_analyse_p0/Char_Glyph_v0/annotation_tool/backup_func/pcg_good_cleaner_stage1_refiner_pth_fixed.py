# -*- coding: utf-8 -*-
r"""
pcg_good_cleaner_stage1_refiner_pth_fixed.py

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
CANVAS_SIZE = 400.0

# clean 保守参数
MAX_REFINE_STEPS = 5
MAX_ACCEPTED_OPS = 4
MIN_STROKES_AFTER_CLEAN = 2

MERGE_ENDPOINT_DIST_MAX = 14.0
MERGE_MAX_BEZIER_RMS_ERROR = 4.5
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

def avg_width(s: Dict[str, Any]) -> float:
    wb = s.get("width_bezier")
    if isinstance(wb, list):
        try:
            a = np.asarray(wb, dtype=np.float32).reshape(-1)
            if len(a):
                return float(np.mean(a))
        except Exception:
            pass
    for k in ["width", "width_mean", "stroke_width"]:
        if k in s:
            try:
                return float(s[k])
            except Exception:
                pass
    return 10.0

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
        edges.append({
            "id": eid,
            "path": cubic(P, ts).astype(np.float32),
            "mother_bezier": P,
            "width": avg_width(s),
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

def render_mask(edges: List[Dict[str, Any]], size=MASK_SIZE, pad=18) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    if not edges:
        return np.array(img, dtype=np.uint8)
    pts = [np.asarray(e["path"], dtype=np.float32) for e in edges if len(e.get("path", []))]
    if not pts:
        return np.array(img, dtype=np.uint8)
    allp = np.concatenate(pts, axis=0)
    mn, mx = np.min(allp, axis=0), np.max(allp, axis=0)
    span = np.maximum(mx - mn, 1e-5)
    scale = min((size - 2*pad) / span[0], (size - 2*pad) / span[1])
    widths = [float(e.get("width", 10.0)) for e in edges]
    med = max(float(np.median(widths)), 1e-3)

    def mp(p):
        x = (p[0]-mn[0])*scale + pad
        y = (p[1]-mn[1])*scale + pad
        return float(x), float(size-y)

    for e in edges:
        p = np.asarray(e["path"], dtype=np.float32)
        if len(p) < 2:
            continue
        xy = [mp(q) for q in p]
        w = max(1, int(round(4.0 * float(e.get("width", 10.0)) / med)))
        draw.line(xy, fill=255, width=w, joint="curve")
        r = w / 2
        for q in [p[0], p[-1]]:
            x, y = mp(q)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
    return np.array(img, dtype=np.uint8)

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
        rep = {"type":"Merge","old_ids":[int(e1["id"]),int(e2["id"])],"endpoint_dist":round(ed,3),"mode":mode,"accepted":False}
        if ed > MERGE_ENDPOINT_DIST_MAX:
            rep["reject_reason"] = "endpoint too far"; return False, edges, rep
        P, rms, mx = fit_cubic(path)
        rep["rms_error"], rep["max_error"] = round(rms,3), round(mx,3)
        if rms > MERGE_MAX_BEZIER_RMS_ERROR or mx > MERGE_MAX_BEZIER_MAX_ERROR:
            rep["reject_reason"] = "bezier fit error too high"; return False, edges, rep
        l1 = float(np.sum(np.linalg.norm(np.diff(e1["path"], axis=0), axis=1)))
        l2 = float(np.sum(np.linalg.norm(np.diff(e2["path"], axis=0), axis=1)))
        w = (float(e1.get("width",10))*l1 + float(e2.get("width",10))*l2) / max(l1+l2, 1e-6)
        new = {"id":max(int(e["id"]) for e in edges)+1, "path":cubic(P, np.linspace(0,1,max(len(path),64))).astype(np.float32),
               "mother_bezier":P, "width":float(w)}
        before, after = render_mask([e1,e2]), render_mask([new])
        iou = mask_iou(before, after)
        area_delta = abs(int((after>0).sum()) - int((before>0).sum())) / max(int((before>0).sum()), 1)
        rep["mask_iou"], rep["area_delta_ratio"] = round(iou,4), round(area_delta,4)
        if iou < MERGE_MIN_MASK_IOU or area_delta > MERGE_MAX_AREA_DELTA_RATIO:
            rep["reject_reason"] = "outline guard rejected"; return False, edges, rep
        out = [copy.deepcopy(e) for k,e in enumerate(edges) if k not in (ia,ib)] + [new]
        rep["accepted"], rep["new_id"], rep["new_width"] = True, int(new["id"]), round(float(w),3)
        return True, out, rep

    def try_delete(self, edges, idx):
        e = edges[idx]
        rep = {"type":"Delete","old_id":int(e["id"]),"accepted":False}
        if len(edges) <= MIN_STROKES_AFTER_CLEAN:
            rep["reject_reason"] = "too few strokes"; return False, edges, rep
        rem = [copy.deepcopy(x) for k,x in enumerate(edges) if k != idx]
        target, rem_mask, full = render_mask([e]), render_mask(rem), render_mask(edges)
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

    def refine(self, bundle, uid):
        before_topo = compute_topo(bundle)
        edges = bundle_to_edges(bundle, sample_n=64)
        locked, taboo, ops, rejected = set(), set(), [], []
        for step in range(MAX_REFINE_STEPS):
            if len(ops) >= MAX_ACCEPTED_OPS or len(edges) <= MIN_STROKES_AFTER_CLEAN:
                break
            act, idx, dbg = self.predict(edges, locked)
            if act == "Done" or idx is None:
                break
            if act == "Delete":
                ok, new_edges, rep = self.try_delete(edges, idx)
                rep["step"], rep["model_debug"] = step, dbg
                if ok: edges = new_edges; ops.append(rep)
                else: rejected.append(rep); locked.add(int(edges[idx]["id"]))
            elif act == "Merge":
                j, _ = self.find_partner(idx, edges, taboo, locked)
                if j is None:
                    locked.add(int(edges[idx]["id"]))
                    rejected.append({"step":step,"type":"Merge","old_id":int(edges[idx]["id"]),"accepted":False,"reject_reason":"no partner","model_debug":dbg})
                    continue
                ok, new_edges, rep = self.try_merge(edges, idx, j)
                rep["step"], rep["model_debug"] = step, dbg
                if ok: edges = new_edges; ops.append(rep)
                else:
                    rejected.append(rep)
                    taboo.add(tuple(sorted([int(edges[idx]["id"]), int(edges[j]["id"])])))
                    locked.add(int(edges[idx]["id"]))
            else:
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
        w = float(e.get("width", 10.0))
        strokes.append({"bezier_id":new_id,
                        "stroke_type":"closed" if np.linalg.norm(P[0]-P[3]) < TOPO_THRESH else "open",
                        "length":round(length,2),
                        "bbox":[round(float(mn[0]),1),round(float(mn[1]),1),round(float(mx[0]),1),round(float(mx[1]),1)],
                        "mother_bezier":P.astype(float).tolist(),
                        "width_bezier":[w,w,w,w]})
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
        self.root.title("PCG Good Cleaner — Stage1 pth Refiner")
        self.root.geometry("1500x950")
        self.pool_root = tk.StringVar(value=POOL_ROOT_DEFAULT)
        self.max_mb = tk.IntVar(value=DEFAULT_MAX_JSON_MB)
        self.preview_px = tk.IntVar(value=PREVIEW_SIZE)
        self.status = tk.StringVar(value="Ready")
        self.refiner = Stage1Refiner()
        self.good_raw, self.good_by_uid, self.uid_outer, self.cleaned, self.pending = {}, {}, {}, {}, []
        self.selected_uid = None
        self.photos = []
        self._ui()
        self.refresh()

    def _ui(self):
        top = ttk.Frame(self.root); top.pack(side="top", fill="x", padx=8, pady=6)
        ttk.Label(top, text="Pool Root:").pack(side="left")
        ttk.Entry(top, textvariable=self.pool_root, width=90).pack(side="left", padx=4)
        ttk.Button(top, text="Browse", command=self.browse).pack(side="left")
        ttk.Label(top, text="Preview:").pack(side="left", padx=(12,2))
        ttk.Spinbox(top, from_=100, to=260, textvariable=self.preview_px, width=6).pack(side="left")
        ttk.Label(top, text="JSON MB:").pack(side="left", padx=(12,2))
        ttk.Spinbox(top, from_=10, to=95, textvariable=self.max_mb, width=6).pack(side="left")
        row = ttk.Frame(self.root); row.pack(side="top", fill="x", padx=8, pady=4)
        ttk.Button(row, text="Refresh Pools", command=self.refresh).pack(side="left", padx=3)
        ttk.Button(row, text="Preview Clean Selected", command=self.preview_selected).pack(side="left", padx=3)
        ttk.Button(row, text="Open Cleaned Preview Window", command=self.open_cleaned).pack(side="left", padx=3)
        ttk.Button(row, text="Write Cleaned Pool", command=lambda: self.write_cleaned(True)).pack(side="left", padx=3)
        ttk.Label(row, textvariable=self.status, foreground="blue").pack(side="left", padx=12)
        note = ttk.LabelFrame(self.root, text="规则")
        note.pack(side="top", fill="x", padx=8, pady=4)
        ttk.Label(note, text="待 clean = good 中 stable_uid 不在 cleaned 中；Accept 只写 cleaned/，不删除/覆盖 good；删除 cleaned 后样本回到待 clean。").pack(side="left", padx=8, pady=4)
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
        self.status.set(f"Good={len(self.good_by_uid)} | Cleaned={len(cleaned_uids)} | Pending={len(self.pending)} | dup_hidden={dup} | Model: {self.refiner.status}")
        self.render_grid()

    def render_grid(self):
        self.clear()
        size = int(self.preview_px.get())
        cols = 4
        for idx, uid in enumerate(self.pending):
            b = self.good_by_uid[uid]
            r,c = divmod(idx, cols)
            frame = ttk.Frame(self.scroll.inner, relief="groove", borderwidth=2)
            frame.grid(row=r, column=c, padx=8, pady=8, sticky="n")
            img = render_bundle(b, size)
            ph = ImageTk.PhotoImage(img); self.photos.append(ph)
            ttk.Label(frame, image=ph).pack(side="top", padx=4, pady=4)
            topo = compute_topo(b)
            gi = glyph_info(b)
            txt = f"{idx:04d} outer={self.uid_outer.get(uid,'')}\nhex={gi.get('hex_key', gi.get('unicode_hex',''))}\n{uid}\nS={topo['stats']['stroke_count']} CC={topo['stats']['connected_components']} Cy={topo['stats']['cycle_count']} X={topo['stats']['x_count']}\n{topo['topology_text'][:80]}"
            ttk.Label(frame, text=txt, font=("Consolas",8), justify="center", wraplength=size*2+50).pack(side="top")
            ttk.Button(frame, text="Select", command=lambda u=uid:self.select(u)).pack(side="left", padx=4, pady=4)
            ttk.Button(frame, text="Clean Preview", command=lambda u=uid:self.preview_uid(u)).pack(side="left", padx=4, pady=4)

    def select(self, uid):
        self.selected_uid = uid
        self.status.set("Selected: " + uid)

    def preview_selected(self):
        if not self.selected_uid:
            messagebox.showwarning("No selection", "请先选择一个待 clean 的 Good 样本。")
            return
        self.preview_uid(self.selected_uid)

    def preview_uid(self, uid):
        if uid not in self.good_by_uid:
            messagebox.showerror("Missing", uid); return
        try:
            before = self.good_by_uid[uid]
            after = self.refiner.refine(before, uid)
            self.before_after(uid, before, after)
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Clean Preview Failed", str(e))

    def before_after(self, uid, before, after):
        win = tk.Toplevel(self.root); win.title("Clean Preview — " + uid); win.geometry("1320x850")
        top = ttk.Frame(win); top.pack(side="top", fill="x", padx=8, pady=6)
        ttk.Label(top, text=uid, font=("Consolas",10)).pack(side="left")
        ttk.Button(top, text="Accept Clean Result", command=lambda:self.accept(uid, after, win)).pack(side="right", padx=4)
        ttk.Button(top, text="Reject / Close", command=win.destroy).pack(side="right", padx=4)
        body = ttk.Frame(win); body.pack(fill="both", expand=True, padx=8, pady=6)
        size = 230
        p1, p2 = ImageTk.PhotoImage(render_bundle(before,size)), ImageTk.PhotoImage(render_bundle(after,size))
        win._photos = [p1,p2]
        for title, bundle, photo in [("Before Clean", before, p1), ("After Clean", after, p2)]:
            f = ttk.LabelFrame(body, text=title); f.pack(side="left", fill="both", expand=True, padx=6, pady=6)
            ttk.Label(f, image=photo).pack(side="top", padx=6, pady=6)
            topo = compute_topo(bundle)
            cm = bundle.get("clean_meta", {})
            txt = f"Stats:\n{json.dumps(topo['stats'], ensure_ascii=False, indent=2)}\n\nTopology:\n{topo['topology_text']}\n"
            if title.startswith("After"):
                txt += "\nAccepted ops:\n" + json.dumps(cm.get("accepted_ops", []), ensure_ascii=False, indent=2)
                txt += "\n\nRejected ops:\n" + json.dumps(cm.get("rejected_ops", []), ensure_ascii=False, indent=2)
                txt += f"\n\nChanged={cm.get('changed')} | Model={cm.get('model_status')}"
            t = tk.Text(f, height=18, width=70); t.pack(fill="both", expand=True, padx=6, pady=6)
            t.insert("1.0", txt); t.configure(state="disabled")

    def accept(self, uid, bundle, win=None):
        bb = copy.deepcopy(bundle)
        bb.setdefault("clean_meta", {})["source_stable_uid"] = uid
        bb["clean_meta"]["cleaned_pool_key"] = uid
        self.cleaned[uid] = bb
        self.write_cleaned(False)
        if win: win.destroy()
        self.refresh()
        messagebox.showinfo("Accepted", f"已写入 cleaned：\n{uid}\n\n原 good 未删除/未覆盖。")

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
            txt=f"{idx:04d}\n{uid}\nS={topo['stats']['stroke_count']} CC={topo['stats']['connected_components']} Cy={topo['stats']['cycle_count']} X={topo['stats']['x_count']}\nchanged={cm.get('changed')} ops={len(cm.get('accepted_ops', []))}\n{topo['topology_text'][:90]}"
            ttk.Label(frame, text=txt, font=("Consolas",8), justify="center", wraplength=size*2+50).pack(side="top")
            ttk.Button(frame, text="Delete Cleaned Result", command=lambda u=uid:remove(u)).pack(side="top", pady=4)

def main():
    ensure_dir(POOL_ROOT_DEFAULT)
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
