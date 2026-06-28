# -*- coding: utf-8 -*-
"""
patch_topostyle_solved_for_scorer.py

把 topostyle_retrieval_selector.py 的输出补成 score_solved_glyphs.py 更容易识别的格式。

用途：
    score_solved_glyphs.py 报：scoreable_graphs: 0 / bad_graph
    通常是因为 solved_nodes 有了，但旧 scorer 找不到 topology edges。

本脚本会读取：
    topostyle_retrieval_solved_candidates.json

并写出：
    solved_glyph_candidates.json
    topostyle_retrieval_solved_candidates_scorer_compatible.json
    patch_topostyle_solved_for_scorer_report.json

运行：
    cd .../Char_Glyph_v0
    python patch_topostyle_solved_for_scorer.py
    python score_solved_glyphs.py
"""

import os
import json
import math
from collections import Counter, defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "topostyle_retrieval_solved_candidates.json")
FALLBACK_INPUT_FILE = os.path.join(SCRIPT_DIR, "solved_glyph_candidates.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "solved_glyph_candidates.json")
OUTPUT_DEBUG_FILE = os.path.join(SCRIPT_DIR, "topostyle_retrieval_solved_candidates_scorer_compatible.json")
REPORT_FILE = os.path.join(SCRIPT_DIR, "patch_topostyle_solved_for_scorer_report.json")
CANVAS_SIZE = 400.0


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


def get_candidate_list(data):
    if isinstance(data, list):
        return data, None
    if isinstance(data, dict):
        for k in ["solved_glyph_candidates", "ranked_solved_glyph_candidates_by_combined", "glyph_candidates", "candidates"]:
            if isinstance(data.get(k), list):
                return data[k], k
        for k, v in data.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v, k
    return [], None


def as_points_px(P):
    arr = np.asarray(P, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) < 4:
        return None
    arr = arr[:4, :2]
    if np.max(np.abs(arr)) <= 2.0:
        arr = arr * CANVAS_SIZE
    return arr.astype(np.float32)


def get_node_bezier_px(node):
    for k in ["mother_bezier", "control_points", "bezier"]:
        if isinstance(node, dict) and k in node:
            P = as_points_px(node[k])
            if P is not None:
                return P
    return None


def patch_node(node, idx):
    node = dict(node)
    node["node_id"] = safe_int(node.get("node_id", node.get("bezier_id", idx)), idx)
    node["bezier_id"] = safe_int(node.get("bezier_id", node.get("node_id", idx)), idx)

    P = get_node_bezier_px(node)
    if P is None:
        return node, False

    p0, p3 = P[0], P[3]
    center = 0.5 * (p0 + p3)
    d = p3 - p0
    length = float(np.linalg.norm(d))
    theta = math.atan2(float(d[1]), float(d[0])) if length > 1e-8 else 0.0

    node["mother_bezier"] = P.astype(float).tolist()
    node["control_points"] = P.astype(float).tolist()
    node["center_norm"] = (center / CANVAS_SIZE).astype(float).tolist()
    node["length_norm"] = float(length / CANVAS_SIZE)
    node["rotation_rad"] = float(theta)

    width_norm = safe_float(node.get("width_norm", safe_float(node.get("width", 14.0), 14.0) / CANVAS_SIZE), 0.035)
    if width_norm > 1.0:
        width_norm = width_norm / CANVAS_SIZE
    node["width_norm"] = float(width_norm)
    node["width"] = float(width_norm * CANVAS_SIZE)
    node.setdefault("shape_code", safe_int(node.get("shape_token", 20), 20))
    node.setdefault("width_token", 0)

    lp = node.get("layout_prior", {})
    if not isinstance(lp, dict):
        lp = {}
    lp.update({
        "center_norm": node["center_norm"],
        "length_norm": node["length_norm"],
        "rotation_rad": node["rotation_rad"],
        "width_norm": node["width_norm"],
    })
    node["layout_prior"] = lp
    return node, True


def build_edges_from_topostyle(candidate):
    topo = candidate.get("topostyle_topology", {})
    if not isinstance(topo, dict):
        return []
    seg_edges = topo.get("segment_edges", [])
    if not isinstance(seg_edges, list):
        return []

    anchor_to_incidents = defaultdict(list)
    for row in seg_edges:
        if not isinstance(row, dict):
            continue
        sid = safe_int(row.get("segment_id", row.get("node_id", -1)), -1)
        if sid < 0:
            continue
        if row.get("anchor_start", None) is not None:
            anchor_to_incidents[int(row["anchor_start"])].append((sid, 0.0))
        if row.get("anchor_end", None) is not None:
            anchor_to_incidents[int(row["anchor_end"])].append((sid, 1.0))

    edges = []
    seen = set()
    for anchor_id, inc in anchor_to_incidents.items():
        inc2, used = [], set()
        for sid, t in inc:
            key = (int(sid), float(t))
            if key not in used:
                used.add(key)
                inc2.append(key)
        inc = inc2
        if len(inc) < 2:
            continue
        for i in range(len(inc)):
            for j in range(i + 1, len(inc)):
                u, tu = inc[i]
                v, tv = inc[j]
                if u == v:
                    continue
                skey = (min(u, v), max(u, v), round(tu, 3), round(tv, 3), int(anchor_id))
                if skey in seen:
                    continue
                seen.add(skey)
                edges.append({
                    "u": int(u), "v": int(v),
                    "src": int(u), "dst": int(v),
                    "source": int(u), "target": int(v),
                    "node_u": int(u), "node_v": int(v),
                    "stroke_a": int(u), "stroke_b": int(v),
                    "j_type": "E2E", "j_type_idx": 1, "type": "E2E",
                    "t_u": float(tu), "t_v": float(tv),
                    "source_t": float(tu), "target_t": float(tv),
                    "t_a": float(tu), "t_b": float(tv),
                    "anchor_id": int(anchor_id),
                })
    return edges


def existing_edges(candidate):
    out = []
    topo = candidate.get("topology", {})
    if isinstance(topo, dict):
        for k in ["positive_edges_undirected", "edges"]:
            if isinstance(topo.get(k), list):
                out.extend(topo[k])
    for k in ["topology_edges", "edges"]:
        if isinstance(candidate.get(k), list):
            out.extend(candidate[k])
    return out


def patch_candidate(candidate, idx):
    cand = dict(candidate)
    nodes = cand.get("solved_nodes") if isinstance(cand.get("solved_nodes"), list) else cand.get("nodes")
    if not isinstance(nodes, list):
        return cand, {"ok": False, "reason": "no_solved_nodes", "edge_count": 0, "node_count": 0, "good_node_count": 0}

    patched_nodes, good_node_count = [], 0
    for i, node in enumerate(nodes):
        pn, ok = patch_node(node, i)
        patched_nodes.append(pn)
        good_node_count += int(ok)

    cand["solved_nodes"] = patched_nodes
    cand["nodes"] = patched_nodes
    cand["solved_segments"] = patched_nodes

    edges = build_edges_from_topostyle(cand) or existing_edges(cand)

    topology = cand.get("topology", {})
    if not isinstance(topology, dict):
        topology = {}
    topology["positive_edges_undirected"] = edges
    topology["edges"] = edges
    topology["num_nodes"] = len(patched_nodes)
    topology["num_edges"] = len(edges)
    topology["topology_junction_px_by_construction"] = 0.0
    cand["topology"] = topology
    cand["topology_edges"] = edges
    cand["edges"] = edges

    qr = cand.get("quality_report", {})
    if not isinstance(qr, dict):
        qr = {}
    qr.update({
        "scorer_compatible_patch": True,
        "patched_edge_count": len(edges),
        "patched_node_count": len(patched_nodes),
        "good_node_bezier_count": good_node_count,
    })
    cand["quality_report"] = qr

    ok = len(patched_nodes) >= 2 and len(edges) >= 1 and good_node_count >= 2
    return cand, {
        "ok": ok,
        "reason": "ok" if ok else "still_bad_graph",
        "edge_count": len(edges),
        "node_count": len(patched_nodes),
        "good_node_count": good_node_count,
    }


def main():
    print("\n" + "=" * 80)
    print("Patch TopoStyle Solved Candidates for score_solved_glyphs.py")
    print("=" * 80)
    input_file = INPUT_FILE if os.path.exists(INPUT_FILE) else FALLBACK_INPUT_FILE
    print(f"  input:  {input_file}")
    print(f"  output: {OUTPUT_FILE}")

    data = load_json(input_file)
    candidates, key = get_candidate_list(data)
    if not candidates:
        raise RuntimeError("没有找到 candidate list。")

    patched = []
    reason_hist = Counter()
    edge_counts, node_counts, good_node_counts = [], [], []
    for i, c in enumerate(candidates):
        pc, info = patch_candidate(c, i)
        patched.append(pc)
        reason_hist[info["reason"]] += 1
        edge_counts.append(info["edge_count"])
        node_counts.append(info["node_count"])
        good_node_counts.append(info["good_node_count"])

    out = dict(data) if isinstance(data, dict) else {}
    out["schema_version"] = "topostyle_retrieval_solved_candidates_scorer_compatible_v1"
    out["solved_glyph_candidates"] = patched
    out["ranked_solved_glyph_candidates_by_combined"] = patched
    out["patch_for_scorer"] = {
        "source_input_file": input_file,
        "candidate_key": key,
        "candidate_count": len(patched),
        "reason_hist": dict(reason_hist),
        "edge_count_stats": stats(edge_counts),
        "node_count_stats": stats(node_counts),
        "good_node_count_stats": stats(good_node_counts),
    }

    save_json(out, OUTPUT_FILE)
    save_json(out, OUTPUT_DEBUG_FILE)
    save_json(out["patch_for_scorer"], REPORT_FILE)

    print("\n" + "=" * 80)
    print("Patch Summary")
    print("=" * 80)
    print(f"  candidate_count: {len(patched)}")
    print(f"  reason_hist: {dict(reason_hist)}")
    print(f"  edge_count_stats: {out['patch_for_scorer']['edge_count_stats']}")
    print(f"  node_count_stats: {out['patch_for_scorer']['node_count_stats']}")
    print(f"  good_node_count_stats: {out['patch_for_scorer']['good_node_count_stats']}")
    print("\nSaved:")
    print(f"  {OUTPUT_FILE}")
    print(f"  {OUTPUT_DEBUG_FILE}")
    print(f"  {REPORT_FILE}")
    print("\nNext:")
    print("  python score_solved_glyphs.py")


if __name__ == "__main__":
    main()
