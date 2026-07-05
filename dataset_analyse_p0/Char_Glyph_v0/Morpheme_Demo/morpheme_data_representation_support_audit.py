# -*- coding: utf-8 -*-
r"""
morpheme_data_representation_support_audit.py

放置位置：
    Babel_Simulater/dataset_analyse_p0/Char_Glyph_v0/Morpheme_Demo/

运行：
    cd Babel_Simulater/dataset_analyse_p0/Char_Glyph_v0/Morpheme_Demo
    python morpheme_data_representation_support_audit.py

作用：
    一次性检查现在的数据支持哪些表示/生成方式：
      1. stroke geometry
      2. topology event graph
      3. morpheme tree
      4. pseudo radical sequence / 语素序列
      5. formation-like layout tree / 布局语素树
      6. axis-aligned 横竖/正交生成
      7. topology question discovery 自动增加问题
      8. neural quality gate 拓扑过滤模型

输出：
    控制台打印可复制总结；同时保存：
      audit_reports/latest_report.json
      audit_reports/latest_report.txt

依赖：
    标准库 + numpy。networkx 可选。
"""

from __future__ import annotations

import os, sys, json, math, time, traceback
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple, Optional, Iterable

import numpy as np

try:
    import networkx as nx
except Exception:
    nx = None

SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.name.lower() == "morpheme_demo":
    CHAR_GLYPH_DIR = SCRIPT_DIR.parent
elif (SCRIPT_DIR / "Morpheme").exists():
    CHAR_GLYPH_DIR = SCRIPT_DIR
else:
    CHAR_GLYPH_DIR = SCRIPT_DIR.parent

DATASET_ANALYSE_DIR = CHAR_GLYPH_DIR.parent
MORPHEME_DIR = CHAR_GLYPH_DIR / "Morpheme"
MORPHEME_OUTPUT_TREE = MORPHEME_DIR / "output_tree"
ANNOTATIONS_TOPO_DIR = DATASET_ANALYSE_DIR / "AI_VECTOR_ROUTER_With_topo" / "annotations_topo"
PCG_POOL_ROOT = CHAR_GLYPH_DIR / "annotation_tool" / "pcg_filebacked_stage2_schema"
REPORT_DIR = SCRIPT_DIR / "audit_reports"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def percentile(values, qs=(0, 25, 50, 75, 90, 95, 99, 100)):
    if not values:
        return {str(q): 0.0 for q in qs}
    arr = np.asarray(values, dtype=np.float64)
    return {str(q): round(float(np.percentile(arr, q)), 4) for q in qs}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def try_load_json(path: Path):
    try:
        return load_json(path)
    except Exception:
        return None


def save_json(obj, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =============================================================================
# 1. JSON bundle 加载
# =============================================================================

def looks_like_bundle(obj):
    return isinstance(obj, dict) and (
        isinstance(obj.get("strokes"), list)
        or (isinstance(obj.get("topology_events"), list) and ("glyph_info" in obj or "cycles" in obj))
    )


def iter_bundles_from_obj(obj, depth=0, max_depth=7):
    if depth > max_depth:
        return
    if looks_like_bundle(obj):
        yield obj
        return
    if isinstance(obj, list):
        for x in obj:
            yield from iter_bundles_from_obj(x, depth + 1, max_depth)
    elif isinstance(obj, dict):
        for k in ["rows", "items", "data", "bundles", "records", "glyphs", "samples"]:
            v = obj.get(k)
            if isinstance(v, (list, dict)):
                yield from iter_bundles_from_obj(v, depth + 1, max_depth)
        for k, v in obj.items():
            if k in ["rows", "items", "data", "bundles", "records", "glyphs", "samples"]:
                continue
            if isinstance(v, (list, dict)):
                yield from iter_bundles_from_obj(v, depth + 1, max_depth)


def bundle_uid(bundle, source, idx):
    gi = bundle.get("glyph_info") or {}
    for k in ["candidate_id", "source_candidate_id", "generated_glyph_id", "glyph_candidate_id", "sample_id", "id"]:
        if gi.get(k):
            return f"{source}:{gi.get(k)}"
        if bundle.get(k):
            return f"{source}:{bundle.get(k)}"
    hx = gi.get("hex_key") or bundle.get("hex_key")
    ch = gi.get("char") or bundle.get("char")
    if hx:
        return f"{source}:hex:{hx}"
    if ch:
        return f"{source}:char:{ch}"
    return f"{source}:idx:{idx}"


def load_bundles_from_dir(path: Path, source: str):
    rows, seen = [], set()
    if not path.exists():
        return rows
    files = sorted(path.rglob("*.json"))
    for fp in files:
        obj = try_load_json(fp)
        if obj is None:
            continue
        for b in iter_bundles_from_obj(obj):
            uid = bundle_uid(b, source, len(rows))
            if uid in seen:
                continue
            seen.add(uid)
            b = b.copy()
            b["_audit_source"] = source
            b["_audit_file"] = str(fp)
            b["_audit_uid"] = uid
            rows.append(b)
    return rows


def load_all_corpus():
    return {
        "annotations_topo": load_bundles_from_dir(ANNOTATIONS_TOPO_DIR, "annotations_topo"),
        "good": load_bundles_from_dir(PCG_POOL_ROOT / "good", "good"),
        "cleaned": load_bundles_from_dir(PCG_POOL_ROOT / "cleaned", "cleaned"),
        "bad": load_bundles_from_dir(PCG_POOL_ROOT / "bad", "bad"),
    }


# =============================================================================
# 2. stroke / graph / layout features
# =============================================================================

def get_strokes(bundle):
    if isinstance(bundle.get("strokes"), list):
        return bundle["strokes"]
    for k in ["solved_nodes", "nodes", "segments"]:
        if isinstance(bundle.get(k), list):
            return bundle[k]
    return []


def stroke_id(stroke, idx):
    for k in ["bezier_id", "id", "stroke_id", "node_id"]:
        if k in stroke:
            try:
                return int(stroke[k])
            except Exception:
                pass
    return idx + 1


def get_mother_bezier(stroke):
    for k in ["mother_bezier", "bezier", "path", "curve", "control_points", "solved_bezier"]:
        v = stroke.get(k)
        if isinstance(v, list) and len(v) == 4:
            try:
                arr = np.asarray(v, dtype=np.float32)
                if arr.shape == (4, 2):
                    return arr
            except Exception:
                pass
    if "p0" in stroke and "p1" in stroke:
        try:
            p0 = np.asarray(stroke["p0"], dtype=np.float32)
            p3 = np.asarray(stroke["p1"], dtype=np.float32)
            if p0.shape == (2,) and p3.shape == (2,):
                return np.stack([p0, p0 * 2 / 3 + p3 / 3, p0 / 3 + p3 * 2 / 3, p3])
        except Exception:
            pass
    return None


def stroke_basic_features(P):
    p0, p1, p2, p3 = P[0], P[1], P[2], P[3]
    v = p3 - p0
    length = float(np.linalg.norm(v))
    if length < 1e-6:
        angle = max_dev = 0.0
        straightness = 0.0
    else:
        angle = math.degrees(math.atan2(float(v[1]), float(v[0]))) % 180.0
        def point_line_dist(p):
            return abs(float(np.cross(v, p - p0))) / max(1e-6, length)
        max_dev = max(point_line_dist(p1), point_line_dist(p2))
        straightness = max(0.0, 1.0 - max_dev / max(1.0, 0.12 * length))
    dist_h = min(angle, 180.0 - angle)
    dist_v = abs(angle - 90.0)
    axis_dist = min(dist_h, dist_v)
    return {
        "length": length,
        "angle": angle,
        "max_dev": max_dev,
        "straightness": straightness,
        "axis_score": max(0.0, 1.0 - axis_dist / 45.0),
        "vertical_score": max(0.0, 1.0 - dist_v / 45.0),
        "horizontal_score": max(0.0, 1.0 - dist_h / 45.0),
    }


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
    def groups(self):
        d = defaultdict(list)
        for x in list(self.parent.keys()):
            d[self.find(x)].append(x)
        return list(d.values())


def get_event_strokes(ev):
    typ = ev.get("type")
    try:
        if typ in ("E2E", "X"):
            return int(ev.get("stroke_a")), int(ev.get("stroke_b"))
        if typ == "T":
            return int(ev.get("guest")), int(ev.get("host"))
    except Exception:
        return None
    return None


def articulation_bridge_fallback(nodes, edges):
    nodes = list(nodes)
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    def count_cc(remove_node=None, remove_edge=None):
        remain = [n for n in nodes if n != remove_node]
        seen, cnt = set(), 0
        for s in remain:
            if s in seen:
                continue
            cnt += 1
            stack = [s]; seen.add(s)
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v == remove_node:
                        continue
                    if remove_edge and tuple(sorted((u, v))) == tuple(sorted(remove_edge)):
                        continue
                    if v not in seen:
                        seen.add(v); stack.append(v)
        return cnt
    base = count_cc()
    arts = sum(1 for n in nodes if count_cc(remove_node=n) > base)
    bridges = sum(1 for e in set(tuple(sorted(e)) for e in edges) if count_cc(remove_edge=e) > base)
    return arts, bridges


def build_graph_from_events(bundle, stroke_ids, P_by_id):
    events = bundle.get("topology_events") if isinstance(bundle.get("topology_events"), list) else []
    edges, rel_counter = [], Counter()
    for ev in events:
        if not isinstance(ev, dict):
            continue
        pair = get_event_strokes(ev)
        if pair is None:
            continue
        a, b = pair
        if a == b:
            continue
        edges.append((a, b))
        rel_counter[str(ev.get("type"))] += 1

    used_fallback = False
    if not edges:
        used_fallback = True
        ids = list(P_by_id.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                P1, P2 = P_by_id[ids[i]], P_by_id[ids[j]]
                md = min(float(np.linalg.norm(a - b)) for a in [P1[0], P1[3]] for b in [P2[0], P2[3]])
                if md <= 5.0:
                    edges.append((ids[i], ids[j])); rel_counter["E2E_fallback"] += 1

    uf = UnionFind(stroke_ids)
    for a, b in edges:
        uf.union(a, b)
    simple_edges = sorted(set(tuple(sorted(e)) for e in edges))
    deg = Counter()
    for a, b in simple_edges:
        deg[a] += 1; deg[b] += 1
    cc = len(uf.groups()) if stroke_ids else 0
    v, e = len(stroke_ids), len(simple_edges)
    cycle_rank = max(0, e - v + cc)

    articulation_count = bridge_count = biconnected_count = 0
    if nx is not None and v > 0:
        G = nx.Graph(); G.add_nodes_from(stroke_ids); G.add_edges_from(simple_edges)
        try:
            articulation_count = len(list(nx.articulation_points(G)))
            bridge_count = len(list(nx.bridges(G)))
            biconnected_count = len(list(nx.biconnected_components(G)))
        except Exception:
            pass
    else:
        articulation_count, bridge_count = articulation_bridge_fallback(stroke_ids, simple_edges)
    return {
        "events": events,
        "relation_counter": dict(rel_counter),
        "edges": simple_edges,
        "degree": dict(deg),
        "connected_components": cc,
        "cycle_rank": cycle_rank,
        "articulation_count": articulation_count,
        "bridge_count": bridge_count,
        "biconnected_count": biconnected_count,
        "used_fallback": used_fallback,
    }


def bbox_overlap_1d(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    denom = max(1e-6, min(a1 - a0, b1 - b0))
    return inter / denom


def bbox_contains(outer, inner, margin=3.0):
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    return ox0 - margin <= ix0 and oy0 - margin <= iy0 and ox1 + margin >= ix1 and oy1 + margin >= iy1


def component_bboxes(P_by_id, graph_pack):
    ids = list(P_by_id.keys())
    uf = UnionFind(ids)
    for a, b in graph_pack.get("edges", []):
        uf.union(a, b)
    comps = []
    for group in uf.groups():
        pts = [P_by_id[sid].reshape(-1, 2) for sid in group if sid in P_by_id]
        if not pts:
            continue
        X = np.concatenate(pts, axis=0)
        mn, mx = X.min(axis=0), X.max(axis=0)
        comps.append({
            "stroke_ids": list(group),
            "bbox": [float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])],
            "cx": float((mn[0] + mx[0]) / 2), "cy": float((mn[1] + mx[1]) / 2),
            "w": float(mx[0] - mn[0]), "h": float(mx[1] - mn[1]),
        })
    return comps


def infer_layout(P_by_id, graph_pack, stroke_feats):
    comps = component_bboxes(P_by_id, graph_pack)
    left_right = top_bottom = enclosure = 0
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            a, b = comps[i], comps[j]
            ax0, ay0, ax1, ay1 = a["bbox"]
            bx0, by0, bx1, by1 = b["bbox"]
            vo = bbox_overlap_1d(ay0, ay1, by0, by1)
            ho = bbox_overlap_1d(ax0, ax1, bx0, bx1)
            if abs(a["cx"] - b["cx"]) > 0.25 * max(1.0, a["w"] + b["w"]) and vo > 0.35:
                left_right += 1
            if abs(a["cy"] - b["cy"]) > 0.25 * max(1.0, a["h"] + b["h"]) and ho > 0.35:
                top_bottom += 1
            if bbox_contains(a["bbox"], b["bbox"]) or bbox_contains(b["bbox"], a["bbox"]):
                enclosure += 1

    n = max(1, len(stroke_feats))
    vertical_count = sum(1 for f in stroke_feats if f["vertical_score"] >= 0.80 and f["straightness"] >= 0.70)
    horizontal_count = sum(1 for f in stroke_feats if f["horizontal_score"] >= 0.80 and f["straightness"] >= 0.70)
    axis_count = sum(1 for f in stroke_feats if f["axis_score"] >= 0.80 and f["straightness"] >= 0.70)
    straight_count = sum(1 for f in stroke_feats if f["straightness"] >= 0.70)
    lens = [f["length"] for f in stroke_feats]
    med = float(np.median(lens)) if lens else 0.0
    long_vertical = sum(1 for f in stroke_feats if f["vertical_score"] >= 0.80 and f["length"] >= max(1.0, med * 1.2))
    short_horiz = sum(1 for f in stroke_feats if f["horizontal_score"] >= 0.70 and f["length"] <= max(1.0, med * 1.2))
    spine_attach = int(long_vertical >= 1 and short_horiz >= 1 and graph_pack.get("connected_components", 0) <= 2)
    return {
        "component_count": len(comps),
        "has_left_right_layout": int(left_right > 0),
        "has_top_bottom_layout": int(top_bottom > 0),
        "has_enclosure_layout": int(enclosure > 0),
        "has_spine_attach_layout": spine_attach,
        "has_parallel_verticals": int(vertical_count >= 2),
        "has_stacked_horizontals": int(horizontal_count >= 2),
        "vertical_count": vertical_count,
        "horizontal_count": horizontal_count,
        "axis_count": axis_count,
        "straight_count": straight_count,
        "vertical_ratio": vertical_count / n,
        "horizontal_ratio": horizontal_count / n,
        "axis_ratio": axis_count / n,
        "straight_ratio": straight_count / n,
        "diagonal_ratio": 1.0 - axis_count / n,
    }


def analyze_bundle(bundle):
    strokes = get_strokes(bundle)
    P_by_id, stroke_feats = {}, []
    for i, s in enumerate(strokes):
        P = get_mother_bezier(s)
        if P is None:
            continue
        sid = stroke_id(s, i)
        P_by_id[sid] = P
        stroke_feats.append(stroke_basic_features(P))
    ids = list(P_by_id.keys())
    graph = build_graph_from_events(bundle, ids, P_by_id)
    layout = infer_layout(P_by_id, graph, stroke_feats)
    cycles = bundle.get("cycles") if isinstance(bundle.get("cycles"), list) else []
    return {
        "uid": bundle.get("_audit_uid"), "source": bundle.get("_audit_source"),
        "stroke_count": len(P_by_id),
        "has_strokes": int(len(P_by_id) > 0),
        "has_topology_events": int(isinstance(bundle.get("topology_events"), list)),
        "topology_event_count": len(graph["events"]),
        "cycle_count_recorded": len(cycles),
        "graph": {
            "connected_components": graph["connected_components"],
            "cycle_rank": graph["cycle_rank"],
            "articulation_count": graph["articulation_count"],
            "bridge_count": graph["bridge_count"],
            "biconnected_count": graph["biconnected_count"],
            "relation_counter": graph["relation_counter"],
            "used_fallback": graph["used_fallback"],
            "max_degree": max(graph.get("degree", {}).values()) if graph.get("degree") else 0,
        },
        "stroke": {
            "angle_values": [round(f["angle"], 3) for f in stroke_feats],
            "length_values": [round(f["length"], 3) for f in stroke_feats],
            "straightness_mean": float(np.mean([f["straightness"] for f in stroke_feats])) if stroke_feats else 0.0,
            "axis_score_mean": float(np.mean([f["axis_score"] for f in stroke_feats])) if stroke_feats else 0.0,
            "vertical_score_mean": float(np.mean([f["vertical_score"] for f in stroke_feats])) if stroke_feats else 0.0,
            "horizontal_score_mean": float(np.mean([f["horizontal_score"] for f in stroke_feats])) if stroke_feats else 0.0,
        },
        "layout": layout,
    }


def summarize(rows):
    n = len(rows)
    if n == 0:
        return {"count": 0}
    def mean(fn):
        vals = [safe_float(fn(r), 0.0) for r in rows]
        return round(float(np.mean(vals)), 6) if vals else 0.0
    relation = Counter()
    all_angles, all_lengths = [], []
    for r in rows:
        relation.update(r["graph"]["relation_counter"])
        all_angles.extend(r["stroke"]["angle_values"])
        all_lengths.extend(r["stroke"]["length_values"])
    h = v = diag = 0
    for a in all_angles:
        a = float(a); dh = min(a, 180 - a); dv = abs(a - 90)
        if dh <= 12: h += 1
        elif dv <= 12: v += 1
        else: diag += 1
    total_strokes = max(1, len(all_angles))
    layout_keys = [
        "has_left_right_layout", "has_top_bottom_layout", "has_enclosure_layout",
        "has_spine_attach_layout", "has_parallel_verticals", "has_stacked_horizontals",
    ]
    layout_counts = {k: sum(int(r["layout"].get(k, 0)) for r in rows) for k in layout_keys}
    return {
        "count": n,
        "stroke_count_percentiles": percentile([r["stroke_count"] for r in rows]),
        "event_count_percentiles": percentile([r["topology_event_count"] for r in rows]),
        "cycle_rank_percentiles": percentile([r["graph"]["cycle_rank"] for r in rows]),
        "max_degree_percentiles": percentile([r["graph"]["max_degree"] for r in rows]),
        "has_strokes_ratio": mean(lambda r: r["has_strokes"]),
        "has_topology_events_ratio": mean(lambda r: r["has_topology_events"]),
        "relation_counter": dict(relation),
        "recorded_cycle_glyph_ratio": mean(lambda r: int(r["cycle_count_recorded"] > 0)),
        "graph_cycle_glyph_ratio": mean(lambda r: int(r["graph"]["cycle_rank"] > 0)),
        "articulation_glyph_ratio": mean(lambda r: int(r["graph"]["articulation_count"] > 0)),
        "bridge_glyph_ratio": mean(lambda r: int(r["graph"]["bridge_count"] > 0)),
        "biconnected_glyph_ratio": mean(lambda r: int(r["graph"]["biconnected_count"] > 0)),
        "straight_ratio_mean": mean(lambda r: r["layout"]["straight_ratio"]),
        "axis_ratio_mean": mean(lambda r: r["layout"]["axis_ratio"]),
        "vertical_ratio_mean": mean(lambda r: r["layout"]["vertical_ratio"]),
        "horizontal_ratio_mean": mean(lambda r: r["layout"]["horizontal_ratio"]),
        "diagonal_ratio_mean": mean(lambda r: r["layout"]["diagonal_ratio"]),
        "angle_bucket_stroke_count": {"horizontal": h, "vertical": v, "diagonal": diag, "total": total_strokes},
        "angle_bucket_stroke_ratio": {"horizontal": round(h / total_strokes, 6), "vertical": round(v / total_strokes, 6), "diagonal": round(diag / total_strokes, 6)},
        "layout_counts": layout_counts,
        "layout_ratios": {k: round(c / n, 6) for k, c in layout_counts.items()},
        "fallback_topology_ratio": mean(lambda r: int(r["graph"]["used_fallback"])),
        "length_percentiles": percentile(all_lengths),
        "angle_percentiles": percentile(all_angles),
    }


# =============================================================================
# 3. Morpheme/output_tree
# =============================================================================

def load_sharded_rows(tree_dir: Path, prefix: str):
    rows = []
    manifest = tree_dir / f"{prefix}_manifest.json"
    if manifest.exists():
        obj = try_load_json(manifest)
        if isinstance(obj, dict):
            for sh in obj.get("shards", []):
                fp = tree_dir / sh.get("file", "")
                data = try_load_json(fp)
                if isinstance(data, dict) and isinstance(data.get("rows"), list):
                    rows.extend(data["rows"])
                elif isinstance(data, list):
                    rows.extend(data)
            return rows
    for fp in sorted(tree_dir.glob(f"{prefix}_shard_*.json")):
        data = try_load_json(fp)
        if isinstance(data, dict) and isinstance(data.get("rows"), list):
            rows.extend(data["rows"])
        elif isinstance(data, list):
            rows.extend(data)
    return rows


def analyze_morpheme_tree():
    tree = MORPHEME_OUTPUT_TREE
    nodes = load_sharded_rows(tree, "morpheme_nodes")
    comps = load_sharded_rows(tree, "composition_alternatives")
    greedy = load_sharded_rows(tree, "greedy_decomposition")
    glyph_roots = try_load_json(tree / "glyph_roots.json")
    summary = try_load_json(tree / "summary.json")
    symmetry = try_load_json(tree / "symmetry_report.json")
    mirror = try_load_json(tree / "existing_mirror_candidates.json")
    stroke_counts = [safe_int(n.get("stroke_count"), 0) for n in nodes]
    stable_count = sum(1 for n in nodes if n.get("is_stable"))
    source_glyph_counts = [safe_int(n.get("source_glyph_count"), 0) for n in nodes]
    support_counts = [safe_int(n.get("support_count"), 0) for n in nodes]
    relation, angle, topo_keys = Counter(), Counter(), Counter()
    parent_count, child_count = Counter(), Counter()
    explicit_layout_count, layout_key_counter = 0, Counter()
    for c in comps:
        p = c.get("parent_morpheme_id")
        if p: parent_count[p] += 1
        for ch in c.get("child_morpheme_ids") or []: child_count[ch] += 1
        tw = c.get("topology_way") or {}
        if isinstance(tw, dict):
            relation.update(tw.get("relation_hist") or {})
            angle.update(tw.get("angle_class_hist") or {})
            topo_keys.update(tw.keys())
        for k in ["layout_way", "composition_way", "spatial_relation", "formation_relation"]:
            if c.get(k):
                explicit_layout_count += 1; layout_key_counter[str(c.get(k))] += 1
    glyph_root_count = 0
    if isinstance(glyph_roots, dict): glyph_root_count = len(glyph_roots.get("rows", glyph_roots))
    elif isinstance(glyph_roots, list): glyph_root_count = len(glyph_roots)
    return {
        "tree_dir": str(tree), "exists": tree.exists(),
        "node_count": len(nodes), "composition_count": len(comps),
        "greedy_decomposition_count": len(greedy), "glyph_root_count": glyph_root_count,
        "stable_node_count": stable_count,
        "stable_node_ratio": round(stable_count / max(1, len(nodes)), 6),
        "has_M_LINE": any(n.get("morpheme_id") == "M_LINE" for n in nodes),
        "stroke_count_percentiles": percentile(stroke_counts),
        "source_glyph_count_percentiles": percentile(source_glyph_counts),
        "support_count_percentiles": percentile(support_counts),
        "composition_parent_unique_count": len(parent_count),
        "composition_child_unique_count": len(child_count),
        "composition_relation_counter": dict(relation),
        "composition_angle_counter": dict(angle),
        "topology_way_keys": dict(topo_keys),
        "explicit_layout_relation_count": explicit_layout_count,
        "explicit_layout_relation_counter": dict(layout_key_counter),
        "summary": summary,
        "symmetry_report": symmetry,
        "existing_mirror_candidate_count": len(mirror) if isinstance(mirror, list) else (len(mirror or {}) if mirror else 0),
    }


# =============================================================================
# 4. 支持矩阵
# =============================================================================

def build_support_matrix(pos, bad, tree):
    pos_n = pos.get("count", 0)
    has_strokes = pos.get("has_strokes_ratio", 0) > 0.9
    has_topo = pos.get("has_topology_events_ratio", 0) > 0.75
    tree_ok = tree.get("node_count", 0) > 100 and tree.get("composition_count", 0) > 100
    greedy_ok = tree.get("greedy_decomposition_count", 0) > 0 or tree.get("glyph_root_count", 0) > 0
    explicit_layout = tree.get("explicit_layout_relation_count", 0) > 0
    axis_ratio = pos.get("axis_ratio_mean", 0)
    layout_ratios = pos.get("layout_ratios", {})
    has_layout_signal = any(layout_ratios.get(k, 0) > 0.05 for k in ["has_left_right_layout", "has_top_bottom_layout", "has_enclosure_layout", "has_spine_attach_layout", "has_parallel_verticals", "has_stacked_horizontals"])
    graph_questions_ok = pos.get("articulation_glyph_ratio", 0) > 0.02 or pos.get("bridge_glyph_ratio", 0) > 0.02 or pos.get("graph_cycle_glyph_ratio", 0) > 0.02
    rows = []
    def add(scheme, cn, score, status, evidence, meaning):
        rows.append({"scheme": scheme, "cn": cn, "support_score_1_to_5": score, "status": status, "evidence": evidence, "what_it_means": meaning})
    add("stroke_geometry", "笔画几何记录", 5 if has_strokes else 1, "强支持" if has_strokes else "弱支持", f"positive_has_strokes_ratio={pos.get('has_strokes_ratio')}", "可以直接用 mother_bezier / width_bezier 做生成、渲染、直线/竖线/横线分析。")
    add("topology_event_graph", "E2E/T/X/cycle 拓扑事件图", 5 if has_topo else 3 if pos.get("has_topology_events_ratio", 0) > 0.3 else 1, "强支持" if has_topo else "部分支持", f"positive_has_topology_events_ratio={pos.get('has_topology_events_ratio')}, relation_counter={pos.get('relation_counter')}", "可以构建图，计算 degree、cycle、bridge、articulation 等问题。")
    add("morpheme_tree", "你的广义语素树", 5 if tree_ok else 1, "强支持" if tree_ok else "弱支持", f"nodes={tree.get('node_count')}, comps={tree.get('composition_count')}, stable={tree.get('stable_node_count')}", "支持 parent <- child_a + child_b + topology_way 的语素组合扩散。")
    add("pseudo_radical_sequence", "伪 radical sequence / 语素序列", 5 if (greedy_ok or tree_ok) else 2, "可做伪序列，不是传统偏旁序列", f"greedy_decomposition_count={tree.get('greedy_decomposition_count')}, glyph_root_count={tree.get('glyph_root_count')}", "可以把 glyph 表示成 Mxxxx 语素序列；但没有传统汉字 radical 语义名。")
    add("formation_like_layout_tree", "formation-like layout tree / 布局语素树", 5 if explicit_layout else 4 if has_layout_signal and has_strokes else 2, "显式支持" if explicit_layout else "可从 bbox/组件隐式推断" if has_layout_signal else "需要新增 layout_way", f"explicit_layout_relations={tree.get('explicit_layout_relation_count')}, layout_ratios={layout_ratios}", "可加入 LEFT_RIGHT / TOP_BOTTOM / ENCLOSURE / SPINE_ATTACH；若没有显式 layout_way，需要从 bbox 推断。")
    add("axis_aligned_generation", "横竖/正交/类汉字直线生成", 5 if axis_ratio > 0.55 else 4 if axis_ratio > 0.35 else 3, "已有横竖信号" if axis_ratio > 0.35 else "数据可分析，但生成器需要强约束", f"axis_ratio_mean={axis_ratio}, vertical_ratio_mean={pos.get('vertical_ratio_mean')}, horizontal_ratio_mean={pos.get('horizontal_ratio_mean')}, angle_bucket={pos.get('angle_bucket_stroke_ratio')}", "可以加入 axis_snap/grid_snap/vertical_column/horizontal_bar_stack/orthogonal_cross，解决候选太斜。")
    add("topology_question_discovery", "自动新增问题 / 自动派生拓扑 rule", 5 if tree.get("node_count", 0) > 3000 and graph_questions_ok else 4 if tree.get("node_count", 0) > 1000 else 2, "数据量足够做第一版" if tree.get("node_count", 0) > 1000 else "数据偏少", f"morpheme_nodes={tree.get('node_count')}, stable={tree.get('stable_node_count')}, graph_signal={{cycle:{pos.get('graph_cycle_glyph_ratio')}, bridge:{pos.get('bridge_glyph_ratio')}, articulation:{pos.get('articulation_glyph_ratio')}}}", "可以尝试 bridge/articulation/biconnected/path/orientation/layout questions，并用 split gain / cluster gain 筛选。")
    add("neural_quality_gate", "拓扑过滤模型 / recall gate", 4 if pos_n > 500 and bad.get("count", 0) > 1000 else 2, "可训练 recall gate" if pos_n > 500 else "正样本仍偏少", f"positive={pos_n}, bad={bad.get('count', 0)}", "可以继续训练 Stage1 topology filter；审美 precision 依赖 hard negatives 和更多 positive。")
    return rows


def recommendations(pos, tree):
    rec = []
    if pos.get("axis_ratio_mean", 0) < 0.60 or pos.get("diagonal_ratio_mean", 0) > 0.35:
        rec.append("优先加 axis-aligned generation：axis_snap_prob、grid_snap_prob、vertical_column、horizontal_bar_stack、orthogonal_cross。")
    if tree.get("explicit_layout_relation_count", 0) <= 0:
        rec.append("在 composition_alternatives 或新生成器里新增 layout_way：LEFT_RIGHT / TOP_BOTTOM / ENCLOSURE / SPINE_ATTACH / PARALLEL_COLUMNS / STACKED_BARS。")
    if tree.get("node_count", 0) > 1000:
        rec.append("可以做 topology question discovery 第一版：bridge_count、articulation_count、biconnected_count、longest_path_ratio、orientation_group_count。")
    rec.append("不要把 traditional radical sequence 当主目标；你的系统更适合做 pseudo radical sequence，即 Mxxxx morpheme 序列。")
    rec.append("formation tree 可以理解成 layout-aware morpheme tree；当前 morpheme tree 应扩展 layout_way，而不是重写成汉字偏旁系统。")
    return rec


def format_report(report):
    lines = []
    lines.append("=" * 96)
    lines.append("[Morpheme Data Representation Support Audit]")
    lines.append(f"time: {report['created_at']}")
    for k, v in report["paths"].items(): lines.append(f"{k}: {v}")
    lines.append("=" * 96)
    lines.append("\n--- Source counts ---")
    for k, v in report["source_counts"].items(): lines.append(f"{k}: {v}")
    ps, bs, mt = report["positive_summary"], report["bad_summary"], report["morpheme_tree_summary"]
    lines.append("\n--- Positive corpus core ---")
    for k in ["count", "has_strokes_ratio", "has_topology_events_ratio", "straight_ratio_mean", "axis_ratio_mean", "vertical_ratio_mean", "horizontal_ratio_mean", "diagonal_ratio_mean", "graph_cycle_glyph_ratio", "articulation_glyph_ratio", "bridge_glyph_ratio", "recorded_cycle_glyph_ratio", "fallback_topology_ratio"]:
        lines.append(f"{k}: {ps.get(k)}")
    lines.append(f"angle_bucket_stroke_ratio: {ps.get('angle_bucket_stroke_ratio')}")
    lines.append(f"layout_ratios: {ps.get('layout_ratios')}")
    lines.append(f"relation_counter: {ps.get('relation_counter')}")
    lines.append(f"stroke_count_percentiles: {ps.get('stroke_count_percentiles')}")
    lines.append(f"cycle_rank_percentiles: {ps.get('cycle_rank_percentiles')}")
    lines.append(f"max_degree_percentiles: {ps.get('max_degree_percentiles')}")
    lines.append("\n--- Bad corpus core ---")
    for k in ["count", "axis_ratio_mean", "vertical_ratio_mean", "horizontal_ratio_mean", "diagonal_ratio_mean", "has_topology_events_ratio"]:
        lines.append(f"{k}: {bs.get(k)}")
    lines.append(f"angle_bucket_stroke_ratio: {bs.get('angle_bucket_stroke_ratio')}")
    lines.append(f"layout_ratios: {bs.get('layout_ratios')}")
    lines.append("\n--- Morpheme tree core ---")
    for k in ["exists", "node_count", "stable_node_count", "stable_node_ratio", "composition_count", "greedy_decomposition_count", "glyph_root_count", "has_M_LINE", "composition_parent_unique_count", "composition_child_unique_count", "explicit_layout_relation_count", "existing_mirror_candidate_count"]:
        lines.append(f"{k}: {mt.get(k)}")
    lines.append(f"stroke_count_percentiles: {mt.get('stroke_count_percentiles')}")
    lines.append(f"composition_relation_counter: {mt.get('composition_relation_counter')}")
    lines.append(f"topology_way_keys: {mt.get('topology_way_keys')}")
    lines.append("\n=== Representation / Generation Scheme Support Matrix ===")
    for r in report["support_matrix"]:
        lines.append(f"[{r['support_score_1_to_5']}/5] {r['cn']} ({r['scheme']}) | {r['status']}")
        lines.append(f"  evidence: {r['evidence']}")
        lines.append(f"  meaning : {r['what_it_means']}")
    lines.append("\n--- Recommendations ---")
    for i, r in enumerate(report["recommendations"], 1): lines.append(f"{i}. {r}")
    lines.append("\n--- Copy this block to ChatGPT ---")
    copy_obj = {
        "source_counts": report["source_counts"],
        "positive_core": {k: ps.get(k) for k in ["count", "axis_ratio_mean", "vertical_ratio_mean", "horizontal_ratio_mean", "diagonal_ratio_mean", "layout_ratios", "graph_cycle_glyph_ratio", "articulation_glyph_ratio", "bridge_glyph_ratio", "angle_bucket_stroke_ratio"]},
        "morpheme_tree_core": {k: mt.get(k) for k in ["node_count", "stable_node_count", "composition_count", "greedy_decomposition_count", "glyph_root_count", "explicit_layout_relation_count", "composition_relation_counter"]},
        "support_matrix": [{"scheme": r["scheme"], "score": r["support_score_1_to_5"], "status": r["status"]} for r in report["support_matrix"]],
    }
    lines.append(json.dumps(copy_obj, ensure_ascii=False, indent=2))
    return "\n".join(lines)


def main():
    t0 = time.time(); ensure_dir(REPORT_DIR)
    print("=" * 96)
    print("[Audit] start")
    print(f"SCRIPT_DIR={SCRIPT_DIR}")
    print(f"CHAR_GLYPH_DIR={CHAR_GLYPH_DIR}")
    print(f"MORPHEME_OUTPUT_TREE={MORPHEME_OUTPUT_TREE}")
    print(f"ANNOTATIONS_TOPO_DIR={ANNOTATIONS_TOPO_DIR}")
    print(f"PCG_POOL_ROOT={PCG_POOL_ROOT}")
    print("=" * 96)

    print("[Audit] loading corpus ...")
    corpus = load_all_corpus()
    source_counts = {k: len(v) for k, v in corpus.items()}
    print("[Audit] source_counts:", source_counts)

    positive = corpus.get("annotations_topo", []) + corpus.get("good", []) + corpus.get("cleaned", [])
    bad = corpus.get("bad", [])

    print(f"[Audit] analyzing positive bundles ... n={len(positive)}")
    pos_rows = []
    for i, b in enumerate(positive):
        if i and i % 500 == 0: print(f"  positive {i}/{len(positive)}")
        try: pos_rows.append(analyze_bundle(b))
        except Exception as e: print("  positive error:", repr(e))

    print(f"[Audit] analyzing bad bundles ... n={len(bad)}")
    bad_rows = []
    for i, b in enumerate(bad):
        if i and i % 1000 == 0: print(f"  bad {i}/{len(bad)}")
        try: bad_rows.append(analyze_bundle(b))
        except Exception: pass

    print("[Audit] summarizing ...")
    pos_sum = summarize(pos_rows)
    bad_sum = summarize(bad_rows)
    tree_sum = analyze_morpheme_tree()
    matrix = build_support_matrix(pos_sum, bad_sum, tree_sum)
    rec = recommendations(pos_sum, tree_sum)

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - t0, 3),
        "paths": {
            "script_dir": str(SCRIPT_DIR),
            "char_glyph_dir": str(CHAR_GLYPH_DIR),
            "dataset_analyse_dir": str(DATASET_ANALYSE_DIR),
            "morpheme_output_tree": str(MORPHEME_OUTPUT_TREE),
            "annotations_topo_dir": str(ANNOTATIONS_TOPO_DIR),
            "pcg_pool_root": str(PCG_POOL_ROOT),
            "report_dir": str(REPORT_DIR),
        },
        "source_counts": source_counts,
        "positive_summary": pos_sum,
        "bad_summary": bad_sum,
        "morpheme_tree_summary": tree_sum,
        "support_matrix": matrix,
        "recommendations": rec,
        "positive_examples_head": pos_rows[:20],
        "bad_examples_head": bad_rows[:20],
    }
    save_json(report, REPORT_DIR / "latest_report.json")
    txt = format_report(report)
    with open(REPORT_DIR / "latest_report.txt", "w", encoding="utf-8") as f: f.write(txt)
    print("\n" + txt)
    print("\n[Audit] saved:")
    print(" ", REPORT_DIR / "latest_report.json")
    print(" ", REPORT_DIR / "latest_report.txt")
    print(f"[Audit] elapsed={report['elapsed_seconds']}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[FATAL]")
        print(traceback.format_exc())
        raise
