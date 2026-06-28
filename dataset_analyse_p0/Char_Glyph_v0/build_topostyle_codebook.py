# -*- coding: utf-8 -*-
"""
build_topostyle_codebook.py

从人工标注数据里建立 TopoStyle 离散 style codebook。

依赖：
    同目录下需要有 train_topostyle_transformer.py
    本脚本会复用其中的：
        load_annotation_glyphs()
        style_to_bezier_np()
        sample_bezier_np()
        denorm_points()

运行：
    cd .../Char_Glyph_v0
    python build_topostyle_codebook.py

输出：
    topostyle_style_codebook.json
    topostyle_codebook_assignments.json
    topostyle_codebook_report.json
"""

import os
import sys
import json
import math
import time
import random
from collections import Counter

import numpy as np

try:
    import train_topostyle_transformer as ts
except Exception as e:
    raise RuntimeError(
        "请把 build_topostyle_codebook.py 放在 train_topostyle_transformer.py 同一个目录下运行。"
    ) from e


# =========================================================
# 0. Config
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_CODEBOOK_FILE = os.path.join(SCRIPT_DIR, "topostyle_style_codebook.json")
OUTPUT_ASSIGNMENTS_FILE = os.path.join(SCRIPT_DIR, "topostyle_codebook_assignments.json")
OUTPUT_REPORT_FILE = os.path.join(SCRIPT_DIR, "topostyle_codebook_report.json")

RANDOM_SEED = 42

CODEBOOK_SIZE = 64
KMEANS_RESTARTS = 12
KMEANS_MAX_ITER = 250
KMEANS_TOL = 1e-6

# width 参与聚类，但权重低一点，避免“粗细”压过曲率形态。
WIDTH_CLUSTER_WEIGHT = 0.35

# 如果 codebook 量化误差太大，可以把 CODEBOOK_SIZE 改成 64 或 96。
SAVE_ALL_ASSIGNMENTS = True


# =========================================================
# 1. Utils
# =========================================================
def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


def curve_rmse_px(P_pred, P_gt):
    Cp = ts.sample_bezier_np(P_pred, n=ts.NUM_CURVE_SAMPLES)
    Cg = ts.sample_bezier_np(P_gt, n=ts.NUM_CURVE_SAMPLES)
    return float(np.sqrt(np.mean(np.sum((Cp - Cg) ** 2, axis=-1))) * ts.CANVAS_SIZE)


def unit_prototype_from_style(style):
    A = np.asarray([0.0, 0.0], dtype=np.float32)
    B = np.asarray([1.0, 0.0], dtype=np.float32)
    return ts.style_to_bezier_np(A, B, style)


def style_to_cluster_feature(style):
    style = np.asarray(style, dtype=np.float32).reshape(-1)
    return np.asarray([
        style[0],
        style[1],
        style[2],
        style[3],
        (style[4] / max(ts.WIDTH_MAX_NORM, 1e-8)) * WIDTH_CLUSTER_WEIGHT,
    ], dtype=np.float32)


def cluster_feature_to_style(feat):
    feat = np.asarray(feat, dtype=np.float32).reshape(-1)
    return np.asarray([
        float(np.clip(feat[0], 0.0, ts.ALPHA_MAX)),
        float(np.clip(feat[1], -ts.BETA_MAX, ts.BETA_MAX)),
        float(np.clip(feat[2], 0.0, ts.ALPHA_MAX)),
        float(np.clip(feat[3], -ts.BETA_MAX, ts.BETA_MAX)),
        float(np.clip(feat[4] / max(WIDTH_CLUSTER_WEIGHT, 1e-8) * ts.WIDTH_MAX_NORM, 0.0, ts.WIDTH_MAX_NORM)),
    ], dtype=np.float32)


def majority(vals):
    c = Counter(vals)
    if not c:
        return None
    return c.most_common(1)[0][0]


# =========================================================
# 2. K-means
# =========================================================
def kmeans_plus_plus_init(X, K, rng):
    N, D = X.shape
    centers = np.zeros((K, D), dtype=np.float32)

    first = int(rng.integers(0, N))
    centers[0] = X[first]
    dist2 = np.sum((X - centers[0:1]) ** 2, axis=1)

    for k in range(1, K):
        total = float(np.sum(dist2))
        if total <= 1e-12:
            centers[k] = X[int(rng.integers(0, N))]
            continue

        probs = dist2 / total
        idx = int(rng.choice(N, p=probs))
        centers[k] = X[idx]

        new_dist2 = np.sum((X - centers[k:k+1]) ** 2, axis=1)
        dist2 = np.minimum(dist2, new_dist2)

    return centers


def assign_kmeans(X, centers):
    dist2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    labels = np.argmin(dist2, axis=1).astype(np.int64)
    inertia = float(np.sum(dist2[np.arange(X.shape[0]), labels]))
    return labels, inertia


def run_kmeans(X, K):
    N, D = X.shape
    K = int(min(max(1, K), N))

    rng_master = np.random.default_rng(RANDOM_SEED)
    best = None

    for r in range(KMEANS_RESTARTS):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        centers = kmeans_plus_plus_init(X, K, rng)

        last_inertia = None
        used_iter = 0

        for it in range(KMEANS_MAX_ITER):
            labels, inertia = assign_kmeans(X, centers)
            new_centers = centers.copy()

            for k in range(K):
                members = X[labels == k]
                if len(members) > 0:
                    new_centers[k] = members.mean(axis=0)
                else:
                    new_centers[k] = X[int(rng.integers(0, N))]

            shift = float(np.sqrt(np.mean((new_centers - centers) ** 2)))
            centers = new_centers
            used_iter = it + 1

            if last_inertia is not None:
                rel = abs(last_inertia - inertia) / max(1e-8, abs(last_inertia))
                if rel < KMEANS_TOL and shift < 1e-5:
                    break
            last_inertia = inertia

        labels, inertia = assign_kmeans(X, centers)
        if best is None or inertia < best["inertia"]:
            best = {
                "centers": centers.copy(),
                "labels": labels.copy(),
                "inertia": float(inertia),
                "iterations": int(used_iter),
                "restart": int(r),
            }

    return best


# =========================================================
# 3. Segment extraction
# =========================================================
def collect_segments_from_annotations():
    glyphs = ts.load_annotation_glyphs()

    segments = []
    glyph_rows = []

    for gi, g in enumerate(glyphs):
        glyph_rows.append({
            "glyph_index": gi,
            "source_file": g["source_file"],
            "hex_key": g["hex_key"],
            "char": g.get("char", ""),
            "num_anchors": len(g["anchors"]),
            "num_segments": len(g["segments"]),
            "num_strokes": len(g["strokes"]),
            "num_topology_edges": len(g["topology_edges"]),
        })

        for si, seg in enumerate(g["segments"]):
            row = dict(seg)
            row["glyph_index"] = gi
            row["segment_index_in_glyph"] = si
            row["source_file"] = g["source_file"]
            row["hex_key"] = g["hex_key"]
            row["char"] = g.get("char", "")
            segments.append(row)

    return glyphs, segments, glyph_rows


# =========================================================
# 4. Build codebook
# =========================================================
def build_codebook(segments):
    styles = np.stack([np.asarray(s["style"], dtype=np.float32) for s in segments], axis=0)
    X = np.stack([style_to_cluster_feature(s) for s in styles], axis=0)

    K = min(CODEBOOK_SIZE, len(segments))
    km = run_kmeans(X, K)

    labels = km["labels"]
    centers_feat = km["centers"]
    centers_style = np.stack([cluster_feature_to_style(c) for c in centers_feat], axis=0)

    all_rmse = []
    all_style_l2 = []

    for i, seg in enumerate(segments):
        token = int(labels[i])
        style_pred = centers_style[token]
        P_pred = ts.style_to_bezier_np(seg["A"], seg["B"], style_pred)
        rmse = curve_rmse_px(P_pred, seg["P_gt"])
        all_rmse.append(rmse)
        all_style_l2.append(float(np.linalg.norm(np.asarray(seg["style"]) - style_pred)))

    all_rmse = np.asarray(all_rmse, dtype=np.float32)
    all_style_l2 = np.asarray(all_style_l2, dtype=np.float32)

    codebook = []

    for k in range(K):
        idxs = np.where(labels == k)[0]
        if len(idxs) == 0:
            continue

        d2 = np.sum((X[idxs] - centers_feat[k:k+1]) ** 2, axis=1)
        medoid_idx = int(idxs[int(np.argmin(d2))])
        medoid = segments[medoid_idx]

        centroid_style = centers_style[k]
        proto_unit = unit_prototype_from_style(centroid_style)

        shape_vals = [int(segments[i]["shape_code"]) for i in idxs]
        width_token_vals = [int(segments[i]["width_token"]) for i in idxs]
        length_vals = [float(segments[i]["length"]) for i in idxs if "length" in segments[i]]
        if not length_vals:
            length_vals = [float(segments[i].get("length_norm", 0.0)) for i in idxs]
        straight_vals = [float(segments[i].get("straight", segments[i].get("straightness", 0.0))) for i in idxs]

        codebook.append({
            "style_token": int(k),
            "count": int(len(idxs)),
            "centroid_style": {
                "alpha1": float(centroid_style[0]),
                "beta1": float(centroid_style[1]),
                "alpha2": float(centroid_style[2]),
                "beta2": float(centroid_style[3]),
                "width_norm": float(centroid_style[4]),
            },
            "style_vector": centroid_style.astype(float).tolist(),
            "prototype_bezier_unit": proto_unit.astype(float).tolist(),
            "dominant_shape_code": int(majority(shape_vals)),
            "dominant_width_token": int(majority(width_token_vals)),
            "length_norm_stats": stats(length_vals),
            "straightness_stats": stats(straight_vals),
            "reconstruction_curve_rmse_px_stats": stats(all_rmse[idxs]),
            "style_l2_stats": stats(all_style_l2[idxs]),
            "medoid": {
                "segment_global_index": medoid_idx,
                "source_file": medoid["source_file"],
                "hex_key": medoid["hex_key"],
                "char": medoid.get("char", ""),
                "parent_stroke": int(medoid["parent_stroke"]),
                "shape_code": int(medoid["shape_code"]),
                "width_token": int(medoid["width_token"]),
                "style_vector": np.asarray(medoid["style"]).astype(float).tolist(),
                "unit_bezier": unit_prototype_from_style(medoid["style"]).astype(float).tolist(),
            },
        })

    assignments = []

    for i, seg in enumerate(segments):
        token = int(labels[i])
        center_style = centers_style[token]
        P_recon = ts.style_to_bezier_np(seg["A"], seg["B"], center_style)

        assignments.append({
            "segment_global_index": int(i),
            "style_token": token,
            "source_file": seg["source_file"],
            "hex_key": seg["hex_key"],
            "char": seg.get("char", ""),
            "glyph_index": int(seg["glyph_index"]),
            "segment_index_in_glyph": int(seg["segment_index_in_glyph"]),
            "parent_stroke": int(seg["parent_stroke"]),
            "shape_code": int(seg["shape_code"]),
            "width_token": int(seg["width_token"]),
            "anchor_start": int(seg["anchor_start"]),
            "anchor_end": int(seg["anchor_end"]),
            "t0": float(seg["t0"]),
            "t1": float(seg["t1"]),
            "length_norm": float(seg.get("length", seg.get("length_norm", 0.0))),
            "straightness": float(seg.get("straight", seg.get("straightness", 0.0))),
            "style_vector": np.asarray(seg["style"]).astype(float).tolist(),
            "assigned_centroid_style": center_style.astype(float).tolist(),
            "reconstruction_curve_rmse_px": float(all_rmse[i]),
            "style_l2": float(all_style_l2[i]),
            "A_px": ts.denorm_points(seg["A"]).astype(float).tolist(),
            "B_px": ts.denorm_points(seg["B"]).astype(float).tolist(),
            "P_gt_px": ts.denorm_points(seg["P_gt"]).astype(float).tolist(),
            "P_recon_px": ts.denorm_points(P_recon).astype(float).tolist(),
        })

    report = {
        "kmeans": {
            "K": int(K),
            "inertia": float(km["inertia"]),
            "iterations": int(km["iterations"]),
            "best_restart": int(km["restart"]),
            "max_iter": KMEANS_MAX_ITER,
            "restarts": KMEANS_RESTARTS,
        },
        "segment_count": int(len(segments)),
        "codebook_size": int(K),
        "cluster_count_stats": stats([c["count"] for c in codebook]),
        "overall_reconstruction_curve_rmse_px_stats": stats(all_rmse),
        "overall_style_l2_stats": stats(all_style_l2),
        "token_hist": {str(k): int(np.sum(labels == k)) for k in range(K)},
    }

    return codebook, assignments, report


def style_stats(segments):
    if not segments:
        return {}
    styles = np.stack([np.asarray(s["style"], dtype=np.float32) for s in segments], axis=0)
    names = ["alpha1", "beta1", "alpha2", "beta2", "width_norm"]
    return {name: stats(styles[:, i]) for i, name in enumerate(names)}


# =========================================================
# 5. Main
# =========================================================
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\n" + "=" * 80)
    print("Build TopoStyle Codebook")
    print("=" * 80)
    print(f"  annotation_dir: {ts.ANNOTATION_DIR}")
    print(f"  output_codebook:    {OUTPUT_CODEBOOK_FILE}")
    print(f"  output_assignments: {OUTPUT_ASSIGNMENTS_FILE}")
    print(f"  output_report:      {OUTPUT_REPORT_FILE}")
    print("=" * 80)

    print("\n[Config]")
    print(f"  CODEBOOK_SIZE: {CODEBOOK_SIZE}")
    print(f"  KMEANS_RESTARTS: {KMEANS_RESTARTS}")
    print(f"  KMEANS_MAX_ITER: {KMEANS_MAX_ITER}")
    print(f"  WIDTH_CLUSTER_WEIGHT: {WIDTH_CLUSTER_WEIGHT}")

    t0 = time.time()
    glyphs, segments, glyph_rows = collect_segments_from_annotations()

    if len(segments) < 2:
        raise RuntimeError("可用 segments 太少，无法构建 codebook。")

    print("\n[Segments]")
    print(f"  usable_glyphs: {len(glyphs)}")
    print(f"  segment_count: {len(segments)}")
    print(f"  glyph_segment_hist: {dict(Counter(g['num_segments'] for g in glyph_rows))}")
    print(f"  shape_code_hist_top10: {Counter(int(s['shape_code']) for s in segments).most_common(10)}")

    codebook, assignments, kreport = build_codebook(segments)

    elapsed = time.time() - t0

    codebook_obj = {
        "schema_version": "topostyle_style_codebook_v1",
        "description": "Discrete TopoStyle prototypes from human annotated anchor-to-anchor segments.",
        "style_param_order": ["alpha1", "beta1", "alpha2", "beta2", "width_norm"],
        "geometry_formula": {
            "P0": "A",
            "P1": "A + alpha1 * |AB| * d + beta1 * |AB| * n",
            "P2": "B - alpha2 * |AB| * d + beta2 * |AB| * n",
            "P3": "B",
            "d": "normalize(B - A)",
            "n": "perpendicular(d)",
        },
        "config": {
            "CODEBOOK_SIZE": CODEBOOK_SIZE,
            "actual_codebook_size": len(codebook),
            "WIDTH_CLUSTER_WEIGHT": WIDTH_CLUSTER_WEIGHT,
            "ALPHA_MAX": ts.ALPHA_MAX,
            "BETA_MAX": ts.BETA_MAX,
            "WIDTH_MAX_NORM": ts.WIDTH_MAX_NORM,
            "CANVAS_SIZE": ts.CANVAS_SIZE,
        },
        "dataset": {
            "annotation_dir": ts.ANNOTATION_DIR,
            "usable_glyphs": len(glyphs),
            "segment_count": len(segments),
            "style_stats": style_stats(segments),
        },
        "codebook": codebook,
    }

    assignments_obj = {
        "schema_version": "topostyle_codebook_assignments_v1",
        "codebook_file": OUTPUT_CODEBOOK_FILE,
        "segment_count": len(assignments),
        "assignments": assignments if SAVE_ALL_ASSIGNMENTS else [],
    }

    top_tokens = sorted(
        [
            {
                "style_token": c["style_token"],
                "count": c["count"],
                "dominant_shape_code": c["dominant_shape_code"],
                "rmse_mean": c["reconstruction_curve_rmse_px_stats"].get("mean", None),
                "rmse_p90": c["reconstruction_curve_rmse_px_stats"].get("p90", None),
                "style_vector": c["style_vector"],
            }
            for c in codebook
        ],
        key=lambda x: -x["count"],
    )

    report_obj = {
        "schema_version": "topostyle_codebook_report_v1",
        "elapsed_sec": round(float(elapsed), 3),
        "segments": {
            "usable_glyphs": len(glyphs),
            "segment_count": len(segments),
            "glyph_segment_hist": dict(Counter(g["num_segments"] for g in glyph_rows)),
            "shape_code_hist": dict(Counter(int(s["shape_code"]) for s in segments)),
            "width_token_hist": dict(Counter(int(s["width_token"]) for s in segments)),
            "style_stats": style_stats(segments),
        },
        "codebook_report": kreport,
        "top_tokens_by_count": top_tokens,
        "outputs": {
            "codebook": OUTPUT_CODEBOOK_FILE,
            "assignments": OUTPUT_ASSIGNMENTS_FILE,
            "report": OUTPUT_REPORT_FILE,
        },
    }

    save_json(codebook_obj, OUTPUT_CODEBOOK_FILE)
    save_json(assignments_obj, OUTPUT_ASSIGNMENTS_FILE)
    save_json(report_obj, OUTPUT_REPORT_FILE)

    print("\n" + "=" * 80)
    print("Codebook Summary")
    print("=" * 80)
    print(f"  segment_count: {len(segments)}")
    print(f"  codebook_size: {len(codebook)}")
    print(f"  elapsed_sec: {elapsed:.3f}")
    print(f"  kmeans_inertia: {kreport['kmeans']['inertia']:.6f}")
    print(f"  cluster_count_stats: {kreport['cluster_count_stats']}")
    print(f"  reconstruction_curve_rmse_px_stats: {kreport['overall_reconstruction_curve_rmse_px_stats']}")
    print(f"  style_l2_stats: {kreport['overall_style_l2_stats']}")

    print("\n[Top tokens by count]")
    for row in top_tokens[:10]:
        style_short = np.asarray(row["style_vector"]).round(3).tolist()
        print(
            f"  token={row['style_token']:02d} count={row['count']:3d} "
            f"shape={row['dominant_shape_code']} "
            f"rmse_mean={row['rmse_mean']} rmse_p90={row['rmse_p90']} "
            f"style={style_short}"
        )

    print("\n" + "=" * 80)
    print("Saved")
    print("=" * 80)
    print(f"  codebook:    {OUTPUT_CODEBOOK_FILE}")
    print(f"  assignments: {OUTPUT_ASSIGNMENTS_FILE}")
    print(f"  report:      {OUTPUT_REPORT_FILE}")

    print("\nNext:")
    print("  1. 看 topostyle_codebook_report.json 的 reconstruction_curve_rmse_px_stats")
    print("  2. 如果 p90 仍大，把 CODEBOOK_SIZE 改成 64 或 96")
    print("  3. 下一步训练 token predictor：输入 topology segment features，输出 style_token 分类")


if __name__ == "__main__":
    main()
