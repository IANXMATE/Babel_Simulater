# -*- coding: utf-8 -*-
"""
topostyle_retrieval_selector_v2_diverse.py

Diverse TopoStyle retrieval selector.

为什么要有 v2：
    v1 会按 prior_reconstruction_rmse 排序；如果 candidate prior 本身接近直线，
    它会强烈偏向直线 token，导致输出图片全是直线、而且同一 candidate 的 beam 很像。

v2 改法：
    1. 不再按 prior RMSE 排序。
    2. 每个 candidate 输出多个显式 style mode：
        retrieval / straight / left_curve / right_curve / s_curve / high_curve / shape_match
    3. 每个 mode 每条 segment 从 codebook 的对应曲率 bucket 里选 token。
    4. 输出仍然兼容 score_solved_glyphs.py，写 solved_glyph_candidates.json。

依赖同目录：
    train_topostyle_transformer.py
    topostyle_retrieval_selector.py
    topostyle_style_codebook.json
    topostyle_codebook_assignments.json
    glyph_candidates_filtered_for_dtg.json 或 glyph_candidates_with_primitives.json

运行：
    cd .../Char_Glyph_v0
    python topostyle_retrieval_selector_v2_diverse.py
    python patch_topostyle_solved_for_scorer.py
    python score_solved_glyphs.py
"""

import os
import sys
import json
import math
from collections import Counter
import numpy as np

try:
    import topostyle_retrieval_selector as base
except Exception as e:
    raise RuntimeError("请把本脚本放在 topostyle_retrieval_selector.py 同目录下。") from e

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_FILE = os.path.join(SCRIPT_DIR, "topostyle_retrieval_v2_diverse_solved_candidates.json")
REPORT_FILE = os.path.join(SCRIPT_DIR, "topostyle_retrieval_v2_diverse_report.json")
COMPAT_FILE = os.path.join(SCRIPT_DIR, "solved_glyph_candidates.json")

MAX_INPUT_CANDIDATES = 80
MAX_OUTPUT_CANDIDATES = 240

# 每个 candidate 输出这些模式；不要全靠 beam 分数，否则又会坍缩成直线。
STYLE_MODES = [
    "retrieval",
    "straight",
    "left_curve",
    "right_curve",
    "s_curve",
    "high_curve",
    "shape_match",
]

# 形状多样性参数
SHAPE_MISMATCH_PENALTY = 0.35
DIRECT_STYLE_WEIGHT = 0.08       # 降低，因为 candidate prior 可能本来就是错误直线
LENGTH_PRIOR_WEIGHT = 0.05
RARITY_WEIGHT = 0.02


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


def codebook_meta(codebook_obj, K):
    meta = {}
    for e in codebook_obj.get("codebook", []):
        t = int(e["style_token"])
        meta[t] = {
            "count": int(e.get("count", 1)),
            "dominant_shape_code": int(e.get("dominant_shape_code", -1)),
            "dominant_width_token": int(e.get("dominant_width_token", -1)),
        }
    for t in range(K):
        meta.setdefault(t, {"count": 1, "dominant_shape_code": -1, "dominant_width_token": -1})
    return meta


def curvature_features(style):
    a1, b1, a2, b2, w = [float(x) for x in style]
    curv = abs(b1) + abs(b2)
    signed = b1 + b2
    sness = abs(b1 - b2)
    is_s = (b1 * b2) < -0.015
    return {
        "curv": curv,
        "signed": signed,
        "sness": sness,
        "is_s": is_s,
    }


def token_pool_for_mode(mode, codebook_styles, meta, shape_code=None):
    pool = []
    for t, st in enumerate(codebook_styles):
        cf = curvature_features(st)
        curv = cf["curv"]
        signed = cf["signed"]
        is_s = cf["is_s"]

        ok = False
        if mode == "retrieval":
            ok = True
        elif mode == "straight":
            ok = curv <= 0.12
        elif mode == "left_curve":
            ok = signed >= 0.28 and not is_s
        elif mode == "right_curve":
            ok = signed <= -0.28 and not is_s
        elif mode == "s_curve":
            ok = is_s or cf["sness"] >= 0.55
        elif mode == "high_curve":
            ok = curv >= 0.55
        elif mode == "shape_match":
            ok = (shape_code is not None and meta[t].get("dominant_shape_code", -1) == int(shape_code))

        if ok:
            pool.append(t)

    # 防止某个 bucket 为空
    if not pool:
        pool = list(range(len(codebook_styles)))
    return pool


def query_style(seg):
    # 注意：这里的 seg['style'] 是 candidate prior 诱导出来的，只作弱参考。
    return np.asarray(seg.get("style", [0.33, 0.0, 0.33, 0.0, 0.035]), dtype=np.float32)


def direct_style_dist(q_style, token_style):
    w = np.asarray([1.0, 1.2, 1.0, 1.2, 0.25], dtype=np.float32)
    return float(np.sqrt(np.sum(((q_style - token_style) * w) ** 2)))


def choose_token_for_mode(seg, glyph_segments, mode, memory, codebook_styles, token_count, meta):
    # retrieval 模式保留 v1 最佳 token
    if mode == "retrieval":
        tops = base.retrieve_top_tokens_for_segment(seg, glyph_segments, memory, codebook_styles, token_count)
        if tops:
            t = int(tops[0]["style_token"])
            return t, float(tops[0]["score"]), {"mode": mode, "source": "v1_retrieval", "v1": tops[0]}

    q = base.query_features(seg, glyph_segments)
    q_style = query_style(seg)
    shape_code = int(seg.get("shape_code", q.get("shape_code", -1)))
    width_token = int(seg.get("width_token", q.get("width_token", -1)))
    q_len = max(1e-6, float(q.get("length", 0.1)))

    pool = token_pool_for_mode(mode, codebook_styles, meta, shape_code=shape_code)

    best = None
    for t in pool:
        st = codebook_styles[t]
        cf = curvature_features(st)
        m = meta[t]

        score = 0.0
        score += DIRECT_STYLE_WEIGHT * direct_style_dist(q_style, st)
        score += SHAPE_MISMATCH_PENALTY * (0 if m.get("dominant_shape_code", -1) == shape_code else 1)
        score += 0.08 * (0 if m.get("dominant_width_token", -1) == width_token else 1)
        score += RARITY_WEIGHT / math.sqrt(max(1.0, float(token_count[t])))

        # mode-specific preference，让不同 mode 真正不同
        if mode == "straight":
            score += 0.20 * cf["curv"]
        elif mode == "left_curve":
            score += 0.20 * abs(cf["signed"] - 0.55)
        elif mode == "right_curve":
            score += 0.20 * abs(cf["signed"] + 0.55)
        elif mode == "s_curve":
            score += 0.25 * (0 if cf["is_s"] else 1) + 0.08 * abs(cf["sness"] - 0.75)
        elif mode == "high_curve":
            score += 0.12 * abs(cf["curv"] - 0.90)
        elif mode == "shape_match":
            # 在同 shape 下保留一点曲率多样性，别自动坍缩成最直的。
            score += 0.03 * abs(cf["curv"] - 0.35)

        if best is None or score < best[0]:
            best = (score, t, cf)

    if best is None:
        t = 0
        return t, 999.0, {"mode": mode, "source": "fallback"}

    score, t, cf = best
    return int(t), float(score), {
        "mode": mode,
        "source": "diverse_bucket",
        "curvature": cf,
        "shape_code": shape_code,
        "dominant_shape_code": meta[int(t)].get("dominant_shape_code", -1),
    }


def make_mode_beam(tg, mode, memory, codebook_styles, token_count, meta):
    tokens = []
    details = []
    score = 0.0

    for si, seg in enumerate(tg["segments"]):
        tok, sc, info = choose_token_for_mode(seg, tg["segments"], mode, memory, codebook_styles, token_count, meta)
        tokens.append(tok)
        score += sc
        details.append({
            "rank": 0,
            "segment_id": int(si),
            "style_token": int(tok),
            "score": float(sc),
            "style_vector": codebook_styles[tok].astype(float).tolist(),
            "example": info,
        })

    return {
        "tokens": tokens,
        "score": float(score),
        "token_details": details,
        "mode": mode,
    }


def load_candidates():
    input_file = base.FILTERED_INPUT_FILE if os.path.exists(base.FILTERED_INPUT_FILE) else base.RAW_INPUT_FILE
    data = base.load_json(input_file)
    candidates, key = base.get_candidate_list(data)
    if not candidates:
        raise RuntimeError("没有找到 candidates。")
    candidates = list(candidates)
    if isinstance(candidates[0], dict) and candidates[0].get("dtg_prefilter"):
        candidates.sort(key=lambda c: base.safe_float(c.get("dtg_prefilter", {}).get("score", 1e9), 1e9))
    return input_file, key, candidates[:MAX_INPUT_CANDIDATES]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("\n" + "=" * 80)
    print("TopoStyle Retrieval Selector V2 Diverse")
    print("=" * 80)
    print(f"  output: {OUTPUT_FILE}")
    print(f"  compat: {COMPAT_FILE}")
    print(f"  report: {REPORT_FILE}")
    print("=" * 80)

    codebook_obj, codebook_styles, token_count = base.load_codebook_styles()
    meta = codebook_meta(codebook_obj, codebook_styles.shape[0])
    memory, memory_report = base.build_memory_bank(codebook_styles)
    input_file, candidate_key, candidates = load_candidates()

    print("\n[Loaded]")
    print(f"  input_file: {input_file}")
    print(f"  candidate_key: {candidate_key}")
    print(f"  candidates: {len(candidates)}")
    print(f"  codebook_size: {len(codebook_styles)}")
    print(f"  memory_size: {memory_report['memory_size']}")
    print(f"  modes: {STYLE_MODES}")

    solved = []
    skipped = Counter()
    mode_hist = Counter()
    source_hist = Counter()

    for idx, cand in enumerate(candidates):
        tg, reason = base.candidate_to_topostyle_glyph(cand, idx)
        if tg is None:
            skipped[reason] += 1
            continue

        for mode in STYLE_MODES:
            beam = make_mode_beam(tg, mode, memory, codebook_styles, token_count, meta)
            out = base.reconstruct_beam_candidate(tg, beam, codebook_styles, len(solved), 0)
            out["generated_glyph_id"] = f"{tg['source_candidate_id']}_topostyle_{mode}"
            out["topostyle_style_mode"] = mode
            out["quality_report"]["style_mode"] = mode
            out["topostyle_retrieval"]["style_mode"] = mode
            solved.append(out)
            mode_hist[mode] += 1
            source_hist[tg["source_candidate_id"]] += 1

            if len(solved) >= MAX_OUTPUT_CANDIDATES:
                break
        if len(solved) >= MAX_OUTPUT_CANDIDATES:
            break

        if (idx + 1) % 20 == 0 or idx == len(candidates) - 1:
            print(f"  progress {idx+1}/{len(candidates)} | solved={len(solved)}")

    # 关键：不要按 prior RMSE 排序；保留 candidate order + mode diversity。
    for i, x in enumerate(solved):
        x["topostyle_output_rank"] = i

    max_rmses = [x["quality_report"]["max_prior_reconstruction_rmse_px"] for x in solved]
    mean_rmses = [x["quality_report"]["mean_prior_reconstruction_rmse_px"] for x in solved]

    output_obj = {
        "schema_version": "topostyle_retrieval_v2_diverse_solved_candidates_v1",
        "method": "topology-first codebook retrieval selector with explicit style diversity",
        "source_input_file": input_file,
        "style_modes": STYLE_MODES,
        "summary": {
            "input_candidates_used": len(candidates),
            "solved_output_count": len(solved),
            "skipped": dict(skipped),
            "mode_hist": dict(mode_hist),
            "topology_junction_px_by_construction": 0.0,
        },
        "solved_glyph_candidates": solved,
        "ranked_solved_glyph_candidates_by_combined": solved,
    }

    report = {
        "schema_version": "topostyle_retrieval_v2_diverse_report_v1",
        "input_file": input_file,
        "candidate_key": candidate_key,
        "candidate_count": len(candidates),
        "solved_output_count": len(solved),
        "skipped": dict(skipped),
        "mode_hist": dict(mode_hist),
        "source_candidate_output_count_stats": stats(list(source_hist.values())),
        "max_prior_reconstruction_rmse_px_stats": stats(max_rmses),
        "mean_prior_reconstruction_rmse_px_stats": stats(mean_rmses),
        "note": "Do not rank by prior RMSE; prior may be a straight fallback. Use visual scorer/manual preview to select.",
    }

    save_json(output_obj, OUTPUT_FILE)
    save_json(output_obj, COMPAT_FILE)
    save_json(report, REPORT_FILE)

    print("\n" + "=" * 80)
    print("V2 Diverse Summary")
    print("=" * 80)
    print(f"  solved_output_count: {len(solved)}")
    print(f"  skipped: {dict(skipped)}")
    print(f"  mode_hist: {dict(mode_hist)}")
    print(f"  max_prior_reconstruction_rmse_px_stats: {report['max_prior_reconstruction_rmse_px_stats']}")
    print(f"  mean_prior_reconstruction_rmse_px_stats: {report['mean_prior_reconstruction_rmse_px_stats']}")
    print("  topology_junction_px_by_construction: 0.0")

    print("\nSaved:")
    print(f"  {OUTPUT_FILE}")
    print(f"  {COMPAT_FILE}")
    print(f"  {REPORT_FILE}")

    print("\nNext:")
    print("  python patch_topostyle_solved_for_scorer.py")
    print("  python score_solved_glyphs.py")
    print("  然后不要只看 top_gnn；同时看各个 style_mode 的预览差异。")


if __name__ == "__main__":
    main()
