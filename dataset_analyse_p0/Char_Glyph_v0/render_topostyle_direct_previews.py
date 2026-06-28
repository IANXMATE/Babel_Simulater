# -*- coding: utf-8 -*-
"""
render_topostyle_direct_previews.py

直接渲染 topostyle_retrieval_selector.py 生成的 Bézier 输出，绕开 GNN scorer。

为什么需要这个脚本：
    score_solved_glyphs.py 的预览很可能只读取 center/length/rotation 之类的 layout_prior，
    从而把 Bézier 曲线退化显示成直线。
    本脚本直接读取 nodes/solved_nodes 里的 mother_bezier/control_points 绘制曲线。

运行：
    cd .../Char_Glyph_v0
    python render_topostyle_direct_previews.py

输入：
    topostyle_retrieval_solved_candidates.json
    或 solved_glyph_candidates.json

输出：
    topostyle_direct_previews/
        top_by_quality/
        by_source_candidate/
        contact_sheet_top.png
        render_report.json
"""

import os
import json
import math
from collections import defaultdict, Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "topostyle_retrieval_solved_candidates.json")
FALLBACK_INPUT_FILE = os.path.join(SCRIPT_DIR, "solved_glyph_candidates.json")
OUT_DIR = os.path.join(SCRIPT_DIR, "topostyle_direct_previews")
TOP_DIR = os.path.join(OUT_DIR, "top_by_quality")
GROUP_DIR = os.path.join(OUT_DIR, "by_source_candidate")
REPORT_FILE = os.path.join(OUT_DIR, "render_report.json")
CONTACT_SHEET_FILE = os.path.join(OUT_DIR, "contact_sheet_top.png")

CANVAS_SIZE = 400
PAD = 34
TILE_W = 880
TILE_H = 460
TOP_N = 80
GROUP_TOP_N = 8
SAMPLE_N = 80
DEFAULT_WIDTH = 14


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


def get_candidate_list(data):
    if isinstance(data, list):
        return data, None
    if not isinstance(data, dict):
        return [], None
    for k in [
        "solved_glyph_candidates",
        "ranked_solved_glyph_candidates_by_combined",
        "glyph_candidates",
        "candidates",
    ]:
        if isinstance(data.get(k), list):
            return data[k], k
    for k, v in data.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            return v, k
    return [], None


def get_nodes(c):
    for k in ["solved_nodes", "nodes", "solved_segments", "segments"]:
        if isinstance(c.get(k), list):
            return c[k]
    return []


def as_bezier_px(x):
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) < 4:
        return None
    arr = arr[:4, :2]
    if not np.all(np.isfinite(arr)):
        return None
    if np.max(np.abs(arr)) <= 2.0:
        arr = arr * CANVAS_SIZE
    return arr.astype(np.float32)


def node_bezier_px(node):
    for k in ["mother_bezier", "control_points", "bezier"]:
        if k in node:
            P = as_bezier_px(node[k])
            if P is not None:
                return P
    topo = node.get("topostyle", {}) if isinstance(node, dict) else {}
    if isinstance(topo, dict):
        for k in ["P_px", "mother_bezier", "control_points"]:
            if k in topo:
                P = as_bezier_px(topo[k])
                if P is not None:
                    return P
    return None


def bezier_points(P, n=SAMPLE_N):
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    mt = 1.0 - t
    C = (mt ** 3) * P[0] + 3 * (mt ** 2) * t * P[1] + 3 * mt * (t ** 2) * P[2] + (t ** 3) * P[3]
    return C.astype(np.float32)


def curve_curvature_proxy(P):
    # 控制点到 chord 的最大垂距，px。越大越弯。
    p0, p1, p2, p3 = P
    d = p3 - p0
    L = float(np.linalg.norm(d))
    if L < 1e-8:
        return 0.0
    n = np.asarray([-d[1], d[0]], dtype=np.float32) / L
    h1 = abs(float(np.dot(p1 - p0, n)))
    h2 = abs(float(np.dot(p2 - p0, n)))
    return max(h1, h2)


def curve_signature(P):
    p0, p1, p2, p3 = P
    d = p3 - p0
    L = float(np.linalg.norm(d))
    if L < 1e-8:
        return (0.0, 0.0, 0.0, 0.0)
    n = np.asarray([-d[1], d[0]], dtype=np.float32) / L
    h1 = float(np.dot(p1 - p0, n) / L)
    h2 = float(np.dot(p2 - p0, n) / L)
    a1 = float(np.dot(p1 - p0, d / L) / L)
    a2 = float(np.dot(p3 - p2, d / L) / L)
    return (round(a1, 3), round(h1, 3), round(a2, 3), round(h2, 3))


def get_width(node):
    w = node.get("width", None)
    if w is not None:
        w = safe_float(w, DEFAULT_WIDTH)
        if w <= 1.0:
            w = w * CANVAS_SIZE
        return int(max(2, min(40, round(w))))
    wn = safe_float(node.get("width_norm", 0.035), 0.035)
    if wn > 1.0:
        return int(max(2, min(40, round(wn))))
    return int(max(2, min(40, round(wn * CANVAS_SIZE))))


def bounds_for_candidate(c):
    pts = []
    for node in get_nodes(c):
        P = node_bezier_px(node)
        if P is not None:
            pts.append(bezier_points(P, n=16))
    if not pts:
        return None
    X = np.concatenate(pts, axis=0)
    xmin, ymin = X.min(axis=0)
    xmax, ymax = X.max(axis=0)
    return float(xmin), float(ymin), float(xmax), float(ymax)


def make_transform(c, side_w=400, side_h=400):
    b = bounds_for_candidate(c)
    if b is None:
        return lambda p: p
    xmin, ymin, xmax, ymax = b
    w = max(1e-6, xmax - xmin)
    h = max(1e-6, ymax - ymin)
    scale = min((side_w - 2 * PAD) / w, (side_h - 2 * PAD) / h)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    ox = side_w / 2.0
    oy = side_h / 2.0

    def tf(P):
        Q = np.asarray(P, dtype=np.float32).copy()
        Q[:, 0] = (Q[:, 0] - cx) * scale + ox
        Q[:, 1] = (Q[:, 1] - cy) * scale + oy
        return Q

    return tf


def draw_curve(draw, pts, color, width):
    xy = [(float(x), float(y)) for x, y in pts]
    if len(xy) >= 2:
        draw.line(xy, fill=color, width=width, joint="curve")
        r = max(2, width // 2)
        for x, y in [xy[0], xy[-1]]:
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def render_candidate(c, rank=0):
    img = Image.new("RGB", (TILE_W, TILE_H), "white")
    draw = ImageDraw.Draw(img)

    left_x = 20
    right_x = 450
    side = 400
    tf = make_transform(c, side_w=side, side_h=side)

    nodes = get_nodes(c)
    palette = [
        (230, 60, 60), (60, 130, 230), (60, 180, 100), (210, 140, 40),
        (150, 80, 210), (40, 190, 190), (220, 80, 160), (90, 90, 90),
    ]

    # panels
    draw.rectangle([left_x, 40, left_x + side, 40 + side], outline=(210, 210, 210), width=1)
    draw.rectangle([right_x, 40, right_x + side, 40 + side], outline=(210, 210, 210), width=1)

    # left: colored control/debug
    for i, node in enumerate(nodes):
        P = node_bezier_px(node)
        if P is None:
            continue
        Q = tf(P)
        color = palette[i % len(palette)]
        pts = bezier_points(Q, n=SAMPLE_N)
        draw_curve(draw, pts + np.asarray([left_x, 40], dtype=np.float32), color=color, width=max(3, get_width(node) // 2))
        # control polygon
        R = Q + np.asarray([left_x, 40], dtype=np.float32)
        draw.line([(float(x), float(y)) for x, y in R], fill=(180, 180, 180), width=1)
        for x, y in R:
            draw.ellipse([x-3, y-3, x+3, y+3], fill=(0, 0, 0))

    # right: black width render
    for i, node in enumerate(nodes):
        P = node_bezier_px(node)
        if P is None:
            continue
        Q = tf(P)
        pts = bezier_points(Q, n=SAMPLE_N)
        draw_curve(draw, pts + np.asarray([right_x, 40], dtype=np.float32), color=(0, 0, 0), width=get_width(node))

    q = c.get("quality_report", {}) if isinstance(c.get("quality_report", {}), dict) else {}
    title = str(c.get("generated_glyph_id", c.get("source_candidate_id", f"cand_{rank}")))
    line1 = f"#{rank:03d} {title}"
    line2 = (
        f"maxRMSE={safe_float(q.get('max_prior_reconstruction_rmse_px', 999)):.3f}px  "
        f"meanRMSE={safe_float(q.get('mean_prior_reconstruction_rmse_px', 999)):.3f}px  "
        f"segs={len(nodes)}"
    )
    draw.text((20, 8), line1[:120], fill=(0, 0, 0))
    draw.text((20, 24), line2, fill=(40, 40, 40))
    draw.text((left_x, TILE_H - 18), "colored Bezier/control", fill=(60, 60, 60))
    draw.text((right_x, TILE_H - 18), "direct black width render", fill=(60, 60, 60))
    return img


def image_hash_signature(c):
    sigs = []
    for node in get_nodes(c):
        P = node_bezier_px(node)
        if P is None:
            continue
        sigs.append(curve_signature(P))
    return tuple(sigs)


def candidate_curvature_stats(c):
    vals = []
    tokens = []
    for node in get_nodes(c):
        P = node_bezier_px(node)
        if P is None:
            continue
        vals.append(curve_curvature_proxy(P))
        tokens.append(safe_int(node.get("style_token", -1), -1))
    return {
        "max_curve_px": float(max(vals)) if vals else 0.0,
        "mean_curve_px": float(np.mean(vals)) if vals else 0.0,
        "style_tokens": tokens,
        "unique_style_token_count": len(set(tokens)),
    }


def main():
    os.makedirs(TOP_DIR, exist_ok=True)
    os.makedirs(GROUP_DIR, exist_ok=True)

    input_file = INPUT_FILE if os.path.exists(INPUT_FILE) else FALLBACK_INPUT_FILE
    data = load_json(input_file)
    candidates, key = get_candidate_list(data)
    if not candidates:
        raise RuntimeError("没有找到 solved candidates。")

    def sort_key(c):
        q = c.get("quality_report", {}) if isinstance(c.get("quality_report", {}), dict) else {}
        return (
            safe_float(q.get("max_prior_reconstruction_rmse_px", 999999), 999999),
            safe_float(q.get("mean_prior_reconstruction_rmse_px", 999999), 999999),
            safe_float(q.get("beam_score", 999999), 999999),
        )

    ranked = sorted(candidates, key=sort_key)
    top = ranked[:TOP_N]

    # render top images
    saved = []
    for i, c in enumerate(top):
        img = render_candidate(c, rank=i)
        name = str(c.get("generated_glyph_id", c.get("source_candidate_id", f"cand_{i}")))
        safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)[:120]
        path = os.path.join(TOP_DIR, f"top_{i:03d}_{safe_name}.png")
        img.save(path)
        saved.append(path)

    # group by source_candidate_id, render beams side by side indirectly as files
    groups = defaultdict(list)
    for c in candidates:
        sid = str(c.get("source_candidate_id", c.get("generated_glyph_id", "unknown")))
        groups[sid].append(c)

    group_reports = []
    for sid, arr in list(groups.items())[:80]:
        arr = sorted(arr, key=sort_key)[:GROUP_TOP_N]
        sigs = [image_hash_signature(c) for c in arr]
        unique_sigs = len(set(sigs))
        token_sets = [tuple(candidate_curvature_stats(c)["style_tokens"]) for c in arr]
        unique_token_sets = len(set(token_sets))
        group_reports.append({
            "source_candidate_id": sid,
            "beam_count": len(arr),
            "unique_geometry_signature_count": unique_sigs,
            "unique_token_sequence_count": unique_token_sets,
            "curvature": [candidate_curvature_stats(c) for c in arr],
        })

        # sheet for this source candidate
        sheet = Image.new("RGB", (TILE_W, TILE_H * len(arr)), "white")
        for j, c in enumerate(arr):
            sheet.paste(render_candidate(c, rank=j), (0, j * TILE_H))
        safe_sid = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in sid)[:80]
        sheet.save(os.path.join(GROUP_DIR, f"{safe_sid}_beams.png"))

    # contact sheet top
    cols = 2
    rows = int(math.ceil(min(len(top), 16) / cols))
    sheet = Image.new("RGB", (TILE_W * cols, TILE_H * rows), "white")
    for i, c in enumerate(top[:16]):
        img = render_candidate(c, rank=i)
        sheet.paste(img, ((i % cols) * TILE_W, (i // cols) * TILE_H))
    sheet.save(CONTACT_SHEET_FILE)

    all_curv = [candidate_curvature_stats(c) for c in candidates]
    report = {
        "input_file": input_file,
        "candidate_key": key,
        "candidate_count": len(candidates),
        "top_rendered_count": len(top),
        "top_dir": TOP_DIR,
        "group_dir": GROUP_DIR,
        "contact_sheet": CONTACT_SHEET_FILE,
        "global": {
            "unique_candidate_geometry_signatures": len(set(image_hash_signature(c) for c in candidates)),
            "unique_token_sequences": len(set(tuple(candidate_curvature_stats(c)["style_tokens"]) for c in candidates)),
            "max_curve_px_stats": summarize([x["max_curve_px"] for x in all_curv]),
            "mean_curve_px_stats": summarize([x["mean_curve_px"] for x in all_curv]),
            "unique_style_token_count_stats": summarize([x["unique_style_token_count"] for x in all_curv]),
        },
        "group_reports_first80": group_reports,
    }
    save_json(report, REPORT_FILE)

    print("\n" + "=" * 80)
    print("Direct TopoStyle Preview Render")
    print("=" * 80)
    print(f"  input_file: {input_file}")
    print(f"  candidate_count: {len(candidates)}")
    print(f"  top_dir: {TOP_DIR}")
    print(f"  group_dir: {GROUP_DIR}")
    print(f"  contact_sheet: {CONTACT_SHEET_FILE}")
    print(f"  report: {REPORT_FILE}")
    print("\n[Diversity]")
    print(f"  unique_candidate_geometry_signatures: {report['global']['unique_candidate_geometry_signatures']}")
    print(f"  unique_token_sequences: {report['global']['unique_token_sequences']}")
    print(f"  max_curve_px_stats: {report['global']['max_curve_px_stats']}")
    print(f"  mean_curve_px_stats: {report['global']['mean_curve_px_stats']}")
    print(f"  unique_style_token_count_stats: {report['global']['unique_style_token_count_stats']}")
    print("\nNext:")
    print("  1. 打开 topostyle_direct_previews/contact_sheet_top.png")
    print("  2. 再看 topostyle_direct_previews/by_source_candidate 里同一 candidate 的 beam 是否真的不同")


def summarize(vals):
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


if __name__ == "__main__":
    main()
