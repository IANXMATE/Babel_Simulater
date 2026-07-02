# -*- coding: utf-8 -*-
r"""
topology_pattern_diffusion_api.py

放置：
    dataset_analyse_p0/Char_Glyph_v0/Morpheme/topology_pattern_diffusion_api.py

定位：
    这是给“飞轮脚本”调用的函数库，不主动接管 GUI，不主动生成图形。
    飞轮脚本只需要 import 这里的函数，把 GUI 勾选的 topology pattern 传进来，
    就能得到：
      1. 每个 morpheme 对各种 topology pattern 的 soft score
      2. 由勾选 pattern 在语素二叉/图式构成树上的递归扩散权重
      3. 自动发现的新 pattern rule 候选
      4. 人工 pattern rule 的派生/组合规则

核心思想：
    - 不把 star / ladder / fork 写成固定 generator。
    - 把它们写成“可存储、可组合、可派生”的 rule。
    - rule 是 feature terms 的集合，例如：
        star ≈ intersection_hub_ratio 高 + orientation_entropy 高 + stroke_count 偏多
        ladder ≈ parallel_pair_ratio 高 + perpendicular_pair_ratio 高 + right_ratio 高
        cycle_with_tail ≈ cycle_rank 高 + leaf_ratio 高 + max_degree 高
    - 扩散时，总概率质量守恒，最终 weights 总和为 1。

典型调用：
    from topology_pattern_diffusion_api import (
        load_morpheme_tree,
        build_default_rule_registry,
        derive_builtin_rules,
        score_morphemes_by_rules,
        diffuse_by_flywheel_selection,
        discover_new_pattern_rules,
    )

    nodes, comps = load_morpheme_tree("output_tree")
    registry = derive_builtin_rules(build_default_rule_registry())

    result = diffuse_by_flywheel_selection(
        nodes, comps,
        selected_patterns={"star": 1.0, "fork": 0.8, "ladder": 0.6},
        registry=registry,
        stable_only=True,
        max_depth=4,
    )

    weights = result["weights"]  # {morpheme_id: probability}, sum = 1.0
"""

from __future__ import annotations

import os
import json
import math
import time
import csv
from copy import deepcopy
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple, Optional, Iterable

import numpy as np


# =============================================================================
# 0. IO
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TREE_DIR = os.path.join(SCRIPT_DIR, "output_tree")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, sort_keys=True)
    os.replace(tmp, path)


def load_sharded_rows(tree_dir: str, prefix: str) -> List[Dict[str, Any]]:
    rows = []
    manifest_path = os.path.join(tree_dir, f"{prefix}_manifest.json")
    if os.path.exists(manifest_path):
        mani = load_json(manifest_path)
        for sh in mani.get("shards", []):
            p = os.path.join(tree_dir, sh.get("file", ""))
            if os.path.exists(p):
                rows.extend(load_json(p).get("rows", []))
        return rows

    if not os.path.isdir(tree_dir):
        return rows

    for fn in sorted(os.listdir(tree_dir)):
        if fn.startswith(prefix + "_shard_") and fn.endswith(".json"):
            rows.extend(load_json(os.path.join(tree_dir, fn)).get("rows", []))
    return rows


def load_morpheme_tree(tree_dir: str = DEFAULT_TREE_DIR) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    飞轮脚本调用入口：
        nodes, compositions = load_morpheme_tree(".../Morpheme/output_tree")
    """
    nodes = load_sharded_rows(tree_dir, "morpheme_nodes")
    comps = load_sharded_rows(tree_dir, "composition_alternatives")
    return nodes, comps


def save_rule_registry(registry: Dict[str, Any], path: str) -> None:
    save_json(registry, path, indent=2)


def load_rule_registry(path: str) -> Dict[str, Any]:
    return load_json(path)


# =============================================================================
# 1. 数值工具
# =============================================================================

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def logistic01(x: float, pivot: float, scale: float) -> float:
    scale = max(1e-6, abs(float(scale)))
    z = (float(x) - float(pivot)) / scale
    z = max(-40.0, min(40.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def gaussian01(x: float, target: float, sigma: float) -> float:
    sigma = max(1e-6, abs(float(sigma)))
    z = (float(x) - float(target)) / sigma
    return math.exp(-0.5 * z * z)


def range01(x: float, low: float, high: float, soft: float = 0.05) -> float:
    """
    soft range membership.
    """
    if low > high:
        low, high = high, low
    left = logistic01(x, low, soft)
    right = 1.0 - logistic01(x, high, soft)
    return clamp01(left * right)


def normalize_dict(d: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(max(0.0, safe_float(v)) for v in d.values()))
    if s <= 0:
        return {}
    return {str(k): max(0.0, safe_float(v)) / s for k, v in d.items()}


def softmax_dict(logits: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    if not logits:
        return {}
    temp = max(1e-6, float(temperature))
    keys = list(logits.keys())
    arr = np.asarray([safe_float(logits[k]) / temp for k in keys], dtype=np.float64)
    arr = arr - np.max(arr)
    ex = np.exp(arr)
    s = float(np.sum(ex))
    if s <= 0:
        return {k: 1.0 / len(keys) for k in keys}
    return {k: float(ex[i] / s) for i, k in enumerate(keys)}


# =============================================================================
# 2. 从 morpheme node 提取拓扑/几何 feature
# =============================================================================

FEATURE_KEYS = [
    # basic size / graph
    "stroke_count",
    "stroke_count_norm",
    "edge_count",
    "edge_density",
    "cycle_rank",
    "cycle_rank_norm",
    "max_degree",
    "max_degree_norm",
    "leaf_ratio",
    "degree2_ratio",
    "branch_ratio",
    # relation hist
    "T_ratio",
    "X_ratio",
    "E2E_ratio",
    # angle hist
    "acute_ratio",
    "right_ratio",
    "obtuse_ratio",
    "collinear_ratio",
    # geometry
    "endpoint_hub_ratio",
    "intersection_hub_ratio",
    "intersection_density",
    "parallel_pair_ratio",
    "perpendicular_pair_ratio",
    "orientation_entropy",
    "dominant_orientation_ratio",
    "length_cv",
    "aspect_log_abs",
    # derived soft indicators
    "tree_like",
    "cycle_like",
    "hub_like",
    "parallel_like",
    "grid_like",
    "dense_like",
    "sparse_like",
]


def _seg_points(seg: Dict[str, Any]):
    p0 = seg.get("p0", [0, 0])
    p1 = seg.get("p1", [1, 0])
    return (
        np.asarray([safe_float(p0[0]), safe_float(p0[1])], dtype=np.float32),
        np.asarray([safe_float(p1[0]), safe_float(p1[1])], dtype=np.float32),
    )


def _normalize_points(points: np.ndarray) -> np.ndarray:
    X = np.asarray(points, dtype=np.float32)
    if len(X) == 0:
        return X.reshape(0, 2)
    X = X - X.mean(axis=0, keepdims=True)
    scale = float(np.sqrt((X ** 2).sum(axis=1).max()))
    if scale < 1e-6:
        scale = 1.0
    return X / scale


def _normalized_segments(segments: List[Dict[str, Any]]) -> np.ndarray:
    pairs = []
    for seg in segments or []:
        p0, p1 = _seg_points(seg)
        pairs.append([p0, p1])
    if not pairs:
        return np.zeros((0, 2, 2), dtype=np.float32)
    arr = np.asarray(pairs, dtype=np.float32)
    pts = _normalize_points(arr.reshape(-1, 2))
    return pts.reshape(arr.shape)


def _angle_of_vec_mod_pi(v: np.ndarray) -> float:
    return float(math.atan2(float(v[1]), float(v[0])) % math.pi)


def _angle_diff_mod_pi(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % math.pi
    return min(d, math.pi - d)


def _cluster_points(points: List[np.ndarray], tol: float = 0.08) -> List[List[np.ndarray]]:
    clusters = []
    for p in points:
        hit = False
        for c in clusters:
            center = np.mean(np.stack(c, axis=0), axis=0)
            if float(np.linalg.norm(p - center)) <= tol:
                c.append(p)
                hit = True
                break
        if not hit:
            clusters.append([p])
    return clusters


def _segment_intersection(p0, p1, q0, q1):
    p = np.asarray(p0, dtype=np.float64)
    r = np.asarray(p1, dtype=np.float64) - p
    q = np.asarray(q0, dtype=np.float64)
    s = np.asarray(q1, dtype=np.float64) - q

    def cross2(a, b):
        return float(a[0] * b[1] - a[1] * b[0])

    denom = cross2(r, s)
    if abs(denom) < 1e-8:
        return None
    qp = q - p
    t = cross2(qp, s) / denom
    u = cross2(qp, r) / denom
    if -1e-6 <= t <= 1.0 + 1e-6 and -1e-6 <= u <= 1.0 + 1e-6:
        return (p + t * r).astype(np.float32)
    return None


def extract_morpheme_features(node: Dict[str, Any]) -> Dict[str, float]:
    """
    对单个 morpheme node 提取 rule 所需 feature。
    注意：只使用 topology / motherline，不使用 width/style。
    """
    segments = node.get("prototype_segments", []) or []
    S = _normalized_segments(segments)
    n = len(S)

    relation_hist = node.get("relation_hist") or {}
    angle_class_hist = node.get("angle_class_hist") or {}
    cycle_rank = safe_float(node.get("cycle_rank_mode"), 0.0)
    degree_hist = [int(x) for x in (node.get("degree_hist_mode") or []) if safe_float(x, None) is not None]

    edge_count = sum(int(v) for v in relation_hist.values()) if relation_hist else max(0, n - 1)
    edge_density = float(edge_count / max(1.0, n))

    max_degree = max(degree_hist, default=0)
    leaf_ratio = sum(1 for d in degree_hist if d <= 1) / max(1.0, len(degree_hist))
    degree2_ratio = sum(1 for d in degree_hist if d == 2) / max(1.0, len(degree_hist))
    branch_ratio = sum(1 for d in degree_hist if d >= 3) / max(1.0, len(degree_hist))

    t_rel = safe_float(relation_hist.get("T"), 0.0)
    x_rel = safe_float(relation_hist.get("X"), 0.0)
    e2e_rel = sum(safe_float(v) for k, v in relation_hist.items() if str(k).startswith("E2E"))

    acute = safe_float(angle_class_hist.get("acute"), 0.0)
    right = safe_float(angle_class_hist.get("right"), 0.0)
    obtuse = safe_float(angle_class_hist.get("obtuse"), 0.0)
    collinear = safe_float(angle_class_hist.get("collinear"), 0.0)

    if n == 0:
        base = {k: 0.0 for k in FEATURE_KEYS}
        return base

    lengths, angles, endpoints = [], [], []
    for i in range(n):
        p0, p1 = S[i, 0], S[i, 1]
        v = p1 - p0
        lengths.append(float(np.linalg.norm(v)))
        angles.append(_angle_of_vec_mod_pi(v))
        endpoints.extend([p0, p1])

    pair_count = n * (n - 1) / 2 if n >= 2 else 1.0
    parallel_pairs, perpendicular_pairs = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            d = _angle_diff_mod_pi(angles[i], angles[j])
            if d <= math.radians(12):
                parallel_pairs += 1
            if abs(d - math.pi / 2) <= math.radians(15):
                perpendicular_pairs += 1

    bins = np.zeros(8, dtype=np.float32)
    for a in angles:
        bi = int((a / math.pi) * len(bins)) % len(bins)
        bins[bi] += 1
    if bins.sum() > 0:
        p = bins / bins.sum()
        orientation_entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum() / math.log(len(bins)))
        dominant_orientation_ratio = float(np.max(p))
    else:
        orientation_entropy = 0.0
        dominant_orientation_ratio = 0.0

    endpoint_clusters = _cluster_points(endpoints, tol=0.08)
    max_endpoint_cluster = max([len(c) for c in endpoint_clusters], default=0)
    endpoint_hub_ratio = float(max_endpoint_cluster / max(1.0, n))

    intersections = []
    for i in range(n):
        for j in range(i + 1, n):
            ip = _segment_intersection(S[i, 0], S[i, 1], S[j, 0], S[j, 1])
            if ip is not None:
                intersections.append(ip)
    inter_clusters = _cluster_points(intersections, tol=0.08)
    max_inter_cluster = max([len(c) for c in inter_clusters], default=0)
    intersection_hub_ratio = float(max_inter_cluster / max(1.0, len(intersections))) if intersections else 0.0
    intersection_density = float(len(intersections) / max(1.0, pair_count))

    pts = S.reshape(-1, 2)
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    aspect = float(xmax - xmin) / max(1e-6, float(ymax - ymin))
    aspect_log_abs = min(3.0, abs(math.log(max(1e-6, aspect)))) / 3.0
    length_cv = min(2.0, float(np.std(lengths) / max(1e-6, np.mean(lengths)))) / 2.0

    f = {
        "stroke_count": float(n),
        "stroke_count_norm": min(1.0, float(n) / 8.0),
        "edge_count": float(edge_count),
        "edge_density": float(edge_density),
        "cycle_rank": float(cycle_rank),
        "cycle_rank_norm": min(1.0, float(cycle_rank) / 3.0),
        "max_degree": float(max_degree),
        "max_degree_norm": min(1.0, float(max_degree) / 5.0),
        "leaf_ratio": float(leaf_ratio),
        "degree2_ratio": float(degree2_ratio),
        "branch_ratio": float(branch_ratio),

        "T_ratio": float(t_rel / max(1.0, edge_count)),
        "X_ratio": float(x_rel / max(1.0, edge_count)),
        "E2E_ratio": float(e2e_rel / max(1.0, edge_count)),

        "acute_ratio": float(acute / max(1.0, edge_count)),
        "right_ratio": float(right / max(1.0, edge_count)),
        "obtuse_ratio": float(obtuse / max(1.0, edge_count)),
        "collinear_ratio": float(collinear / max(1.0, edge_count)),

        "endpoint_hub_ratio": endpoint_hub_ratio,
        "intersection_hub_ratio": intersection_hub_ratio,
        "intersection_density": intersection_density,
        "parallel_pair_ratio": float(parallel_pairs / max(1.0, pair_count)),
        "perpendicular_pair_ratio": float(perpendicular_pairs / max(1.0, pair_count)),
        "orientation_entropy": float(orientation_entropy),
        "dominant_orientation_ratio": float(dominant_orientation_ratio),
        "length_cv": float(length_cv),
        "aspect_log_abs": float(aspect_log_abs),
    }

    f["tree_like"] = clamp01((1.0 - f["cycle_rank_norm"]) * (0.55 + 0.45 * f["leaf_ratio"]))
    f["cycle_like"] = clamp01(0.75 * f["cycle_rank_norm"] + 0.25 * f["degree2_ratio"])
    f["hub_like"] = clamp01(0.45 * f["endpoint_hub_ratio"] + 0.35 * f["intersection_hub_ratio"] + 0.20 * f["max_degree_norm"])
    f["parallel_like"] = clamp01(0.75 * f["parallel_pair_ratio"] + 0.25 * f["dominant_orientation_ratio"])
    f["grid_like"] = clamp01(0.35 * f["parallel_pair_ratio"] + 0.30 * f["perpendicular_pair_ratio"] + 0.25 * f["intersection_density"] + 0.10 * f["right_ratio"])
    f["dense_like"] = clamp01(0.35 * min(1.0, f["edge_density"] / 2.0) + 0.35 * f["intersection_density"] + 0.30 * f["stroke_count_norm"])
    f["sparse_like"] = clamp01(1.0 - f["dense_like"])

    return f


# =============================================================================
# 3. 规则存储格式与内置规则库
# =============================================================================

def term(feature: str, op: str, weight: float = 1.0, **kwargs) -> Dict[str, Any]:
    d = {"feature": feature, "op": op, "weight": float(weight)}
    d.update(kwargs)
    return d


def make_rule(
    rule_id: str,
    terms: List[Dict[str, Any]],
    *,
    aliases: Optional[List[str]] = None,
    description: str = "",
    parent_rules: Optional[List[str]] = None,
    source: str = "builtin",
    version: str = "1.0",
) -> Dict[str, Any]:
    return {
        "rule_id": rule_id,
        "aliases": aliases or [],
        "description": description,
        "terms": terms,
        "parent_rules": parent_rules or [],
        "source": source,
        "version": version,
    }


def build_default_rule_registry() -> Dict[str, Any]:
    """
    返回所有人工拓扑结构规则。规则都是 JSON-serializable，可保存/派生/编辑。
    """
    rules = {}

    def add(r):
        rules[r["rule_id"]] = r

    # primitives / basic lines
    add(make_rule("line", [
        term("stroke_count", "near", target=1, sigma=0.65, weight=1.0),
        term("edge_count", "near", target=0, sigma=0.75, weight=0.5),
    ], aliases=["single_stroke"], description="单条直线终结符。"))

    add(make_rule("parallel_pair", [
        term("stroke_count", "near", target=2, sigma=0.8, weight=0.4),
        term("parallel_pair_ratio", "high", pivot=0.65, scale=0.12, weight=1.2),
        term("intersection_density", "low", pivot=0.15, scale=0.08, weight=0.5),
    ], aliases=["double_parallel"], description="两条或少量近似平行线。"))

    # chain family
    add(make_rule("chain", [
        term("tree_like", "high", pivot=0.55, scale=0.12, weight=1.0),
        term("E2E_ratio", "high", pivot=0.45, scale=0.15, weight=1.0),
        term("max_degree_norm", "low", pivot=0.55, scale=0.12, weight=0.7),
        term("cycle_rank_norm", "low", pivot=0.2, scale=0.10, weight=1.0),
    ], aliases=["path", "polyline"], description="端点到端点串联的链式结构。"))

    add(make_rule("long_chain", [
        term("tree_like", "high", pivot=0.55, scale=0.12, weight=0.8),
        term("E2E_ratio", "high", pivot=0.45, scale=0.15, weight=0.8),
        term("stroke_count_norm", "high", pivot=0.45, scale=0.12, weight=0.9),
        term("max_degree_norm", "low", pivot=0.55, scale=0.12, weight=0.5),
    ], parent_rules=["chain"], description="更长的链式结构。"))

    add(make_rule("corner", [
        term("stroke_count", "near", target=2, sigma=0.9, weight=0.8),
        term("right_ratio", "high", pivot=0.5, scale=0.15, weight=1.2),
        term("E2E_ratio", "high", pivot=0.3, scale=0.15, weight=0.6),
    ], aliases=["right_angle", "L_shape"], description="两笔直角折角。"))

    add(make_rule("v_shape", [
        term("stroke_count", "near", target=2, sigma=1.0, weight=0.5),
        term("endpoint_hub_ratio", "high", pivot=0.75, scale=0.12, weight=1.0),
        term("acute_ratio", "high", pivot=0.25, scale=0.16, weight=0.5),
        term("obtuse_ratio", "high", pivot=0.25, scale=0.16, weight=0.5),
    ], aliases=["angle_join"], description="两线共享端点形成 V/Λ 型。"))

    add(make_rule("zigzag", [
        term("chain", "pattern_high", pivot=0.35, scale=0.12, weight=0.55),
        term("acute_ratio", "high", pivot=0.2, scale=0.15, weight=0.45),
        term("obtuse_ratio", "high", pivot=0.2, scale=0.15, weight=0.45),
        term("length_cv", "high", pivot=0.15, scale=0.12, weight=0.25),
    ], parent_rules=["chain"], description="连续折线 / 锯齿结构。"))

    # branch family
    add(make_rule("fork", [
        term("tree_like", "high", pivot=0.55, scale=0.12, weight=0.8),
        term("endpoint_hub_ratio", "high", pivot=0.75, scale=0.12, weight=0.9),
        term("max_degree_norm", "high", pivot=0.45, scale=0.10, weight=0.8),
        term("leaf_ratio", "high", pivot=0.45, scale=0.12, weight=0.5),
    ], aliases=["Y_shape", "branch"], description="多条线在端点附近分叉。"))

    add(make_rule("tri_fork", [
        term("fork", "pattern_high", pivot=0.45, scale=0.12, weight=1.0),
        term("stroke_count", "near", target=3, sigma=1.0, weight=0.5),
    ], parent_rules=["fork"], description="三笔左右的经典 Y / 三叉。"))

    add(make_rule("multi_fork", [
        term("fork", "pattern_high", pivot=0.45, scale=0.12, weight=0.8),
        term("stroke_count_norm", "high", pivot=0.45, scale=0.15, weight=0.5),
        term("max_degree_norm", "high", pivot=0.55, scale=0.12, weight=0.5),
    ], parent_rules=["fork"], description="多叉分支。"))

    add(make_rule("tree", [
        term("tree_like", "high", pivot=0.6, scale=0.12, weight=1.0),
        term("branch_ratio", "high", pivot=0.15, scale=0.10, weight=0.5),
        term("leaf_ratio", "high", pivot=0.35, scale=0.12, weight=0.5),
    ], aliases=["branch_tree"], description="无环树状结构。"))

    add(make_rule("balanced_tree", [
        term("tree", "pattern_high", pivot=0.4, scale=0.15, weight=0.8),
        term("orientation_entropy", "high", pivot=0.4, scale=0.15, weight=0.4),
        term("leaf_ratio", "range", low=0.35, high=0.8, soft=0.08, weight=0.4),
    ], parent_rules=["tree"], description="更均衡展开的树状结构。"))

    # star/radial/fan
    add(make_rule("star", [
        term("intersection_hub_ratio", "high", pivot=0.65, scale=0.12, weight=1.0),
        term("intersection_density", "high", pivot=0.25, scale=0.12, weight=0.6),
        term("orientation_entropy", "high", pivot=0.45, scale=0.12, weight=0.7),
        term("stroke_count_norm", "high", pivot=0.35, scale=0.12, weight=0.4),
    ], aliases=["asterisk"], description="多组线段相交/汇聚于一点，方向分散。"))

    add(make_rule("radial", [
        term("hub_like", "high", pivot=0.55, scale=0.12, weight=0.8),
        term("orientation_entropy", "high", pivot=0.45, scale=0.12, weight=0.8),
        term("stroke_count_norm", "high", pivot=0.35, scale=0.12, weight=0.3),
    ], aliases=["spoke"], description="放射状结构。"))

    add(make_rule("fan", [
        term("endpoint_hub_ratio", "high", pivot=0.70, scale=0.12, weight=1.0),
        term("orientation_entropy", "high", pivot=0.40, scale=0.14, weight=0.7),
        term("intersection_density", "low", pivot=0.35, scale=0.12, weight=0.4),
    ], aliases=["hand_fan"], description="端点汇聚的扇形结构。"))

    # parallel / ladder / grid
    add(make_rule("parallel_bundle", [
        term("parallel_like", "high", pivot=0.55, scale=0.12, weight=1.0),
        term("dominant_orientation_ratio", "high", pivot=0.45, scale=0.12, weight=0.6),
        term("intersection_density", "low", pivot=0.25, scale=0.12, weight=0.5),
    ], aliases=["parallel"], description="多条近似平行线。"))

    add(make_rule("ladder", [
        term("parallel_pair_ratio", "high", pivot=0.35, scale=0.12, weight=0.9),
        term("perpendicular_pair_ratio", "high", pivot=0.20, scale=0.12, weight=0.6),
        term("right_ratio", "high", pivot=0.20, scale=0.12, weight=0.55),
        term("stroke_count_norm", "high", pivot=0.35, scale=0.15, weight=0.35),
    ], aliases=["rail", "rungs"], description="两组或多组平行近线 + 横向连接，梯子结构。"))

    add(make_rule("comb", [
        term("parallel_pair_ratio", "high", pivot=0.35, scale=0.12, weight=0.7),
        term("T_ratio", "high", pivot=0.15, scale=0.10, weight=0.45),
        term("leaf_ratio", "high", pivot=0.35, scale=0.12, weight=0.45),
        term("tree_like", "high", pivot=0.45, scale=0.14, weight=0.45),
    ], aliases=["rake"], description="梳子/耙子，主干上接多条近似平行齿。"))

    add(make_rule("grid", [
        term("grid_like", "high", pivot=0.35, scale=0.12, weight=1.0),
        term("stroke_count_norm", "high", pivot=0.45, scale=0.12, weight=0.4),
        term("right_ratio", "high", pivot=0.15, scale=0.12, weight=0.35),
    ], aliases=["mesh"], description="网格/井字型趋势。"))

    add(make_rule("h_shape", [
        term("stroke_count", "near", target=3, sigma=1.1, weight=0.5),
        term("parallel_pair_ratio", "high", pivot=0.25, scale=0.12, weight=0.7),
        term("right_ratio", "high", pivot=0.25, scale=0.12, weight=0.5),
        term("perpendicular_pair_ratio", "high", pivot=0.25, scale=0.12, weight=0.5),
    ], aliases=["H"], description="H 型，两竖一横或等价拓扑。"))

    # intersections
    add(make_rule("t_junction", [
        term("T_ratio", "high", pivot=0.25, scale=0.12, weight=1.0),
        term("right_ratio", "high", pivot=0.20, scale=0.14, weight=0.35),
    ], aliases=["T_shape"], description="T 型依附/交汇。"))

    add(make_rule("x_intersection", [
        term("X_ratio", "high", pivot=0.20, scale=0.12, weight=0.9),
        term("intersection_density", "high", pivot=0.20, scale=0.12, weight=0.5),
    ], aliases=["X_shape"], description="X 型交叉。"))

    add(make_rule("cross", [
        term("intersection_density", "high", pivot=0.25, scale=0.12, weight=0.65),
        term("perpendicular_pair_ratio", "high", pivot=0.25, scale=0.12, weight=0.55),
        term("stroke_count", "range", low=2, high=5, soft=1.0, weight=0.25),
    ], aliases=["plus", "crossing"], description="十字/交叉结构。"))

    add(make_rule("dense_crossing", [
        term("intersection_density", "high", pivot=0.35, scale=0.12, weight=0.8),
        term("dense_like", "high", pivot=0.45, scale=0.12, weight=0.6),
        term("orientation_entropy", "high", pivot=0.45, scale=0.12, weight=0.35),
    ], parent_rules=["cross"], description="较复杂的多交叉结构。"))

    # cycles
    add(make_rule("cycle", [
        term("cycle_like", "high", pivot=0.45, scale=0.12, weight=1.0),
        term("cycle_rank_norm", "high", pivot=0.25, scale=0.10, weight=0.9),
        term("leaf_ratio", "low", pivot=0.25, scale=0.12, weight=0.4),
    ], aliases=["loop"], description="闭环结构。"))

    add(make_rule("triangle_cycle", [
        term("cycle", "pattern_high", pivot=0.45, scale=0.12, weight=0.8),
        term("stroke_count", "near", target=3, sigma=1.0, weight=0.45),
        term("right_ratio", "low", pivot=0.45, scale=0.14, weight=0.2),
    ], parent_rules=["cycle"], description="三角环。"))

    add(make_rule("box_cycle", [
        term("cycle", "pattern_high", pivot=0.45, scale=0.12, weight=0.8),
        term("stroke_count", "near", target=4, sigma=1.2, weight=0.45),
        term("right_ratio", "high", pivot=0.20, scale=0.12, weight=0.35),
    ], aliases=["rect_cycle", "square_cycle"], parent_rules=["cycle"], description="四边形/盒状环。"))

    add(make_rule("cycle_with_tail", [
        term("cycle_like", "high", pivot=0.35, scale=0.12, weight=0.75),
        term("cycle_rank_norm", "high", pivot=0.20, scale=0.10, weight=0.65),
        term("leaf_ratio", "high", pivot=0.15, scale=0.10, weight=0.55),
        term("max_degree_norm", "high", pivot=0.40, scale=0.12, weight=0.45),
    ], aliases=["lollipop", "loop_tail"], parent_rules=["cycle", "chain"], description="环 + 尾巴，类似 lollipop。"))

    add(make_rule("cycle_with_chord", [
        term("cycle_like", "high", pivot=0.35, scale=0.12, weight=0.7),
        term("edge_density", "high", pivot=1.2, scale=0.25, weight=0.5),
        term("intersection_density", "high", pivot=0.15, scale=0.10, weight=0.25),
        term("leaf_ratio", "low", pivot=0.25, scale=0.12, weight=0.25),
    ], aliases=["loop_with_bridge"], parent_rules=["cycle"], description="闭环中带桥/弦。"))

    add(make_rule("theta_graph", [
        term("cycle_rank_norm", "high", pivot=0.45, scale=0.12, weight=0.8),
        term("edge_density", "high", pivot=1.2, scale=0.25, weight=0.45),
        term("branch_ratio", "high", pivot=0.10, scale=0.08, weight=0.35),
    ], parent_rules=["cycle_with_chord"], description="theta / 双路径环结构。"))

    add(make_rule("double_cycle", [
        term("cycle_rank_norm", "high", pivot=0.50, scale=0.12, weight=0.85),
        term("stroke_count_norm", "high", pivot=0.45, scale=0.12, weight=0.45),
        term("dense_like", "high", pivot=0.35, scale=0.12, weight=0.35),
    ], aliases=["double_loop"], parent_rules=["cycle"], description="双环/多环结构。"))

    add(make_rule("nested_or_touching_cycle", [
        term("double_cycle", "pattern_high", pivot=0.35, scale=0.12, weight=0.55),
        term("intersection_density", "range", low=0.05, high=0.55, soft=0.10, weight=0.35),
        term("orientation_entropy", "high", pivot=0.35, scale=0.15, weight=0.25),
    ], parent_rules=["double_cycle"], description="嵌套/相切倾向的复杂环。"))

    # dense/sparse and miscellaneous
    add(make_rule("dense_cluster", [
        term("dense_like", "high", pivot=0.45, scale=0.12, weight=1.0),
        term("stroke_count_norm", "high", pivot=0.45, scale=0.12, weight=0.45),
    ], aliases=["dense"], description="高密度拓扑块。"))

    add(make_rule("sparse", [
        term("sparse_like", "high", pivot=0.55, scale=0.12, weight=1.0),
        term("edge_density", "low", pivot=1.0, scale=0.25, weight=0.35),
    ], description="稀疏结构。"))

    add(make_rule("barb", [
        term("chain", "pattern_high", pivot=0.35, scale=0.12, weight=0.55),
        term("fork", "pattern_high", pivot=0.25, scale=0.12, weight=0.45),
        term("acute_ratio", "high", pivot=0.2, scale=0.12, weight=0.35),
    ], aliases=["hooked_branch"], description="主链末端带短分叉/倒刺。"))

    add(make_rule("arrow_like", [
        term("chain", "pattern_high", pivot=0.30, scale=0.12, weight=0.45),
        term("fork", "pattern_high", pivot=0.25, scale=0.12, weight=0.45),
        term("endpoint_hub_ratio", "high", pivot=0.55, scale=0.12, weight=0.35),
        term("acute_ratio", "high", pivot=0.20, scale=0.12, weight=0.35),
    ], aliases=["arrow"], description="箭头/尖端分叉趋势。"))

    return {
        "schema": "topology_pattern_rule_registry.v1",
        "created_by": "topology_pattern_diffusion_api.build_default_rule_registry",
        "rules": rules,
        "feature_keys": FEATURE_KEYS,
    }


# =============================================================================
# 4. rule scoring
# =============================================================================

def _term_score(t: Dict[str, Any], features: Dict[str, float], prior_scores: Optional[Dict[str, float]] = None) -> float:
    op = t.get("op")
    feat = t.get("feature")

    if op == "pattern_high":
        v = safe_float((prior_scores or {}).get(feat), 0.0)
    else:
        v = safe_float(features.get(feat), 0.0)

    if op == "high" or op == "pattern_high":
        return logistic01(v, safe_float(t.get("pivot"), 0.5), safe_float(t.get("scale"), 0.1))
    if op == "low":
        return 1.0 - logistic01(v, safe_float(t.get("pivot"), 0.5), safe_float(t.get("scale"), 0.1))
    if op == "near":
        return gaussian01(v, safe_float(t.get("target"), 0.0), safe_float(t.get("sigma"), 1.0))
    if op == "range":
        return range01(v, safe_float(t.get("low"), 0.0), safe_float(t.get("high"), 1.0), safe_float(t.get("soft"), 0.05))
    if op == "linear":
        return clamp01(v)
    if op == "inverse_linear":
        return clamp01(1.0 - v)

    # unknown term should not dominate.
    return 0.0


def score_rule(
    features: Dict[str, float],
    rule: Dict[str, Any],
    prior_scores: Optional[Dict[str, float]] = None,
) -> float:
    """
    weighted average of term scores.
    """
    terms = rule.get("terms") or []
    if not terms:
        return 0.0
    total_w, total = 0.0, 0.0
    for t in terms:
        w = max(0.0, safe_float(t.get("weight"), 1.0))
        total += w * _term_score(t, features, prior_scores=prior_scores)
        total_w += w
    if total_w <= 0:
        return 0.0
    return clamp01(total / total_w)


def score_features_by_registry(features: Dict[str, float], registry: Dict[str, Any]) -> Dict[str, float]:
    """
    由于某些规则引用 pattern_high，例如 cycle_with_tail 可引用 cycle，
    这里做两轮/三轮传播。
    """
    rules = registry.get("rules", {})
    scores = {rid: 0.0 for rid in rules}

    for _ in range(3):
        new_scores = {}
        for rid, rule in rules.items():
            new_scores[rid] = score_rule(features, rule, prior_scores=scores)
        scores = new_scores

    return scores


def score_morphemes_by_rules(
    nodes: List[Dict[str, Any]],
    registry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    返回：
      {
        morpheme_id: {
          "features": {...},
          "scores": {rule_id: score}
        }
      }
    """
    registry = registry or derive_builtin_rules(build_default_rule_registry())
    out = {}
    for node in nodes:
        mid = node.get("morpheme_id")
        if not mid:
            continue
        f = extract_morpheme_features(node)
        s = score_features_by_registry(f, registry)
        out[mid] = {"features": f, "scores": s}
    return out


# =============================================================================
# 5. rule 派生能力：人工规则组合 + 自动规则发现
# =============================================================================

def compose_rules(
    new_rule_id: str,
    registry: Dict[str, Any],
    parent_rule_ids: List[str],
    *,
    weights: Optional[List[float]] = None,
    extra_terms: Optional[List[Dict[str, Any]]] = None,
    description: str = "",
    source: str = "derived_manual",
) -> Dict[str, Any]:
    """
    将已有规则组合成一个派生规则。
    注意：这不是生成形状，只是生成一个新 rule。
    """
    rules = registry.get("rules", {})
    weights = weights or [1.0] * len(parent_rule_ids)
    terms = []
    for rid, w in zip(parent_rule_ids, weights):
        terms.append(term(rid, "pattern_high", pivot=0.35, scale=0.12, weight=float(w)))
    terms.extend(extra_terms or [])
    return make_rule(
        new_rule_id,
        terms,
        aliases=[],
        description=description or ("derived from " + ",".join(parent_rule_ids)),
        parent_rules=parent_rule_ids,
        source=source,
    )


def derive_builtin_rules(registry: Dict[str, Any]) -> Dict[str, Any]:
    """
    添加一批可解释的派生结构。
    如果你不想要这些派生结构，可以直接用 build_default_rule_registry()。
    """
    reg = deepcopy(registry)
    rules = reg.setdefault("rules", {})

    derived = [
        compose_rules(
            "branch_chain",
            reg,
            ["chain", "fork"],
            weights=[0.55, 0.45],
            extra_terms=[term("tree_like", "high", pivot=0.55, scale=0.12, weight=0.4)],
            description="链条上带分叉。",
        ),
        compose_rules(
            "fork_with_tail",
            reg,
            ["fork", "chain"],
            weights=[0.6, 0.4],
            extra_terms=[term("leaf_ratio", "high", pivot=0.35, scale=0.12, weight=0.3)],
            description="fork 结构加尾部延伸。",
        ),
        compose_rules(
            "ladder_with_tail",
            reg,
            ["ladder", "chain"],
            weights=[0.65, 0.35],
            extra_terms=[term("parallel_pair_ratio", "high", pivot=0.30, scale=0.12, weight=0.4)],
            description="梯子结构附带链式延伸。",
        ),
        compose_rules(
            "star_with_tail",
            reg,
            ["star", "chain"],
            weights=[0.75, 0.25],
            extra_terms=[term("intersection_hub_ratio", "high", pivot=0.55, scale=0.12, weight=0.4)],
            description="星型中心结构附带尾巴。",
        ),
        compose_rules(
            "radial_fork",
            reg,
            ["radial", "fork"],
            weights=[0.55, 0.45],
            description="放射状分叉。",
        ),
        compose_rules(
            "grid_ladder",
            reg,
            ["grid", "ladder"],
            weights=[0.55, 0.45],
            description="更接近网格的梯子。",
        ),
        compose_rules(
            "cross_with_tail",
            reg,
            ["cross", "chain"],
            weights=[0.70, 0.30],
            description="交叉结构带尾巴。",
        ),
        compose_rules(
            "cycle_ladder",
            reg,
            ["cycle", "ladder"],
            weights=[0.50, 0.50],
            extra_terms=[term("cycle_rank_norm", "high", pivot=0.2, scale=0.10, weight=0.25)],
            description="环和梯子倾向混合的结构。",
        ),
    ]

    for r in derived:
        rules[r["rule_id"]] = r
    reg["derived_rule_count"] = len(derived)
    return reg


def feature_vector_for_discovery(features: Dict[str, float], scores: Dict[str, float], registry: Dict[str, Any]) -> np.ndarray:
    rule_ids = sorted(registry.get("rules", {}).keys())
    vals = [safe_float(features.get(k), 0.0) for k in FEATURE_KEYS]
    vals += [safe_float(scores.get(rid), 0.0) for rid in rule_ids]
    return np.asarray(vals, dtype=np.float32)


def _kmeans_np(X: np.ndarray, k: int, iterations: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(X)
    if n == 0:
        return np.zeros(0, dtype=np.int32), np.zeros((0, X.shape[1] if X.ndim == 2 else 0), dtype=np.float32)
    k = max(1, min(int(k), n))
    centers = [X[int(rng.integers(0, n))]]
    for _ in range(1, k):
        d2 = np.min(np.stack([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0), axis=0)
        if float(np.sum(d2)) <= 1e-12:
            centers.append(X[int(rng.integers(0, n))])
        else:
            centers.append(X[int(rng.choice(n, p=d2 / np.sum(d2)))])
    C = np.stack(centers, axis=0)

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iterations):
        D = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
        labels = np.argmin(D, axis=1).astype(np.int32)
        newC = []
        for j in range(k):
            idx = np.where(labels == j)[0]
            if len(idx) == 0:
                newC.append(X[int(rng.integers(0, n))])
            else:
                newC.append(np.mean(X[idx], axis=0))
        newC = np.stack(newC, axis=0)
        if np.allclose(newC, C, atol=1e-5):
            C = newC
            break
        C = newC
    return labels, C


def discover_new_pattern_rules(
    nodes: List[Dict[str, Any]],
    registry: Optional[Dict[str, Any]] = None,
    *,
    k: int = 16,
    stable_only: bool = True,
    min_support: int = 1,
    min_source_glyphs: int = 1,
    iterations: int = 40,
    seed: int = 20260702,
    sample_top: int = 12,
) -> Dict[str, Any]:
    """
    探索新模式：
      - 对每个 morpheme 抽 feature + 已有 rule score
      - 聚类
      - 每个 cluster 生成一个 auto rule
        auto rule 不是固定模板，而是一组 near-centroid feature terms。
    """
    registry = registry or derive_builtin_rules(build_default_rule_registry())
    rule_ids = sorted(registry.get("rules", {}).keys())

    scored = score_morphemes_by_rules(nodes, registry)
    mids, X = [], []
    node_by_id = {n.get("morpheme_id"): n for n in nodes if n.get("morpheme_id")}

    for mid, pack in scored.items():
        node = node_by_id.get(mid, {})
        if mid == "M_LINE":
            continue
        if stable_only and not node.get("is_stable"):
            continue
        if int(node.get("support_count") or 0) < min_support:
            continue
        if int(node.get("source_glyph_count") or 0) < min_source_glyphs:
            continue
        mids.append(mid)
        X.append(feature_vector_for_discovery(pack["features"], pack["scores"], registry))

    if not X:
        return {"schema": "auto_pattern_discovery.v1", "rules": {}, "clusters": []}

    X = np.stack(X, axis=0)
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0) + 1e-6
    Xz = (X - mu) / sd
    labels, centers_z = _kmeans_np(Xz, k=k, iterations=iterations, seed=seed)
    centers = centers_z * sd + mu

    f_len = len(FEATURE_KEYS)
    clusters, auto_rules = [], {}

    for ci in range(len(centers)):
        idx = np.where(labels == ci)[0]
        if len(idx) == 0:
            continue

        center = centers[ci]
        center_features = {FEATURE_KEYS[i]: float(center[i]) for i in range(f_len)}
        center_scores = {rule_ids[i]: float(center[f_len + i]) for i in range(len(rule_ids))}

        top_rule_scores = sorted(center_scores.items(), key=lambda kv: kv[1], reverse=True)[:6]
        top_features = sorted(center_features.items(), key=lambda kv: kv[1], reverse=True)[:8]

        # 用最有信息量的 features 生成 near-centroid rule。
        selected_terms = []
        for fk, val in top_features:
            if fk in ("stroke_count", "edge_count", "max_degree", "cycle_rank"):
                continue
            if val > 0.18:
                selected_terms.append(term(fk, "near", target=round(float(val), 5), sigma=0.18, weight=0.6))

        for rid, val in top_rule_scores[:3]:
            if val > 0.25:
                selected_terms.append(term(rid, "pattern_high", pivot=max(0.15, val * 0.75), scale=0.14, weight=0.45))

        if not selected_terms:
            selected_terms = [term(top_features[0][0], "near", target=round(float(top_features[0][1]), 5), sigma=0.20, weight=1.0)]

        auto_id = f"auto_pattern_{ci:03d}"
        auto_label = "+".join([rid for rid, val in top_rule_scores[:3] if val > 0.25]) or "auto"
        auto_rules[auto_id] = make_rule(
            auto_id,
            selected_terms,
            aliases=[auto_label],
            description=f"Auto-discovered topology mode. Top existing rules: {top_rule_scores[:5]}",
            parent_rules=[rid for rid, val in top_rule_scores[:3] if val > 0.25],
            source="auto_discovered",
        )

        d = np.sum((Xz[idx] - centers_z[ci]) ** 2, axis=1)
        order = np.argsort(d)[:sample_top]
        samples = []
        for oi in order:
            mid = mids[int(idx[int(oi)])]
            node = node_by_id.get(mid, {})
            samples.append({
                "morpheme_id": mid,
                "stroke_count": node.get("stroke_count"),
                "support_count": node.get("support_count"),
                "source_glyph_count": node.get("source_glyph_count"),
                "is_stable": node.get("is_stable"),
                "top_rules": sorted(scored[mid]["scores"].items(), key=lambda kv: kv[1], reverse=True)[:6],
            })

        clusters.append({
            "auto_pattern_id": auto_id,
            "auto_label": auto_label,
            "size": int(len(idx)),
            "center_features": {k: round(v, 6) for k, v in center_features.items()},
            "center_rule_scores": {k: round(v, 6) for k, v in center_scores.items()},
            "samples": samples,
        })

    clusters.sort(key=lambda c: c["size"], reverse=True)
    return {
        "schema": "auto_pattern_discovery.v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "k": len(clusters),
        "rules": auto_rules,
        "clusters": clusters,
    }


def merge_auto_rules(registry: Dict[str, Any], discovered: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 discover_new_pattern_rules 的结果合并回 registry。
    """
    reg = deepcopy(registry)
    reg.setdefault("rules", {}).update(discovered.get("rules", {}))
    reg["auto_rule_count"] = len(discovered.get("rules", {}))
    return reg


# =============================================================================
# 6. 飞轮勾选 -> 扩散
# =============================================================================

def pattern_weights_from_flywheel_selection(selection: Any, registry: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    支持 GUI 传入多种格式：

    list:
        ["star", "fork", "ladder"]

    dict:
        {"star": True, "fork": 0.8, "ladder": False}

    string:
        "star,fork,ladder"
        "star:1.0,fork:0.8,ladder:0.5"
    """
    registry = registry or derive_builtin_rules(build_default_rule_registry())
    valid = set(registry.get("rules", {}).keys())
    weights = {}

    if selection is None:
        return {}

    if isinstance(selection, str):
        for part in selection.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                k, v = part.split(":", 1)
                k = k.strip()
                if k in valid:
                    weights[k] = max(0.0, safe_float(v, 1.0))
            else:
                if part in valid:
                    weights[part] = 1.0

    elif isinstance(selection, (list, tuple, set)):
        for k in selection:
            if str(k) in valid:
                weights[str(k)] = 1.0

    elif isinstance(selection, dict):
        for k, v in selection.items():
            k = str(k)
            if k not in valid:
                continue
            if isinstance(v, bool):
                if v:
                    weights[k] = 1.0
            else:
                fv = safe_float(v, 0.0)
                if fv > 0:
                    weights[k] = fv

    return normalize_dict(weights)


def _node_valid_for_diffusion(node: Dict[str, Any], stable_only: bool, min_support: int, min_source_glyphs: int, min_stroke_count: int, max_stroke_count: int) -> bool:
    if stable_only and not node.get("is_stable"):
        return False
    if int(node.get("support_count") or 0) < int(min_support):
        return False
    if int(node.get("source_glyph_count") or 0) < int(min_source_glyphs):
        return False
    sc = int(node.get("stroke_count") or 0)
    if sc < min_stroke_count or sc > max_stroke_count:
        return False
    return True


def _rule_score_for_topology_way(comp: Dict[str, Any], selected_rule_scores: Dict[str, float]) -> float:
    """
    composition rule 自身的 topology_way 也参与扩散。
    这里用 relation_hist / angle_class_hist 做轻量评分。
    """
    topo = comp.get("topology_way") or {}
    rel = topo.get("relation_hist") or {}
    angle = topo.get("angle_class_hist") or {}
    edge_count = max(1.0, safe_float(topo.get("edge_count"), 1.0))

    pseudo_features = {
        "T_ratio": safe_float(rel.get("T"), 0.0) / edge_count,
        "X_ratio": safe_float(rel.get("X"), 0.0) / edge_count,
        "E2E_ratio": sum(safe_float(v) for k, v in rel.items() if str(k).startswith("E2E")) / edge_count,
        "acute_ratio": safe_float(angle.get("acute"), 0.0) / edge_count,
        "right_ratio": safe_float(angle.get("right"), 0.0) / edge_count,
        "obtuse_ratio": safe_float(angle.get("obtuse"), 0.0) / edge_count,
        "collinear_ratio": safe_float(angle.get("collinear"), 0.0) / edge_count,
    }

    # topology_way 对几类基础结构有直接偏置。
    pseudo_scores = {
        "t_junction": pseudo_features["T_ratio"],
        "x_intersection": pseudo_features["X_ratio"],
        "cross": 0.55 * pseudo_features["X_ratio"] + 0.45 * pseudo_features["right_ratio"],
        "chain": pseudo_features["E2E_ratio"],
        "fork": pseudo_features["T_ratio"] + 0.25 * (pseudo_features["acute_ratio"] + pseudo_features["obtuse_ratio"]),
        "ladder": 0.50 * pseudo_features["right_ratio"] + 0.50 * pseudo_features["collinear_ratio"],
        "parallel_bundle": pseudo_features["collinear_ratio"],
        "zigzag": pseudo_features["acute_ratio"] + pseudo_features["obtuse_ratio"],
    }
    total = 0.0
    for rid, w in selected_rule_scores.items():
        total += w * clamp01(pseudo_scores.get(rid, 0.0))
    return clamp01(total)


def diffuse_by_flywheel_selection(
    nodes: List[Dict[str, Any]],
    compositions: List[Dict[str, Any]],
    selected_patterns: Any,
    *,
    registry: Optional[Dict[str, Any]] = None,
    precomputed_scores: Optional[Dict[str, Dict[str, Any]]] = None,
    stable_only: bool = True,
    min_support: int = 1,
    min_source_glyphs: int = 1,
    min_stroke_count: int = 1,
    max_stroke_count: int = 8,
    temperature: float = 0.75,
    max_depth: int = 4,
    keep_ratio: float = 0.25,
    top_roots: int = 512,
    top_rules_per_parent: int = 8,
    support_alpha: float = 0.12,
    rule_support_alpha: float = 0.18,
    topology_way_alpha: float = 0.35,
    child_pattern_alpha: float = 1.0,
) -> Dict[str, Any]:
    """
    飞轮调用的主函数。

    返回：
      {
        "weights": {morpheme_id: probability},  # sum = 1
        "trace": [...],
        "root_weights": {...},
        "pattern_weights": {...},
        "scored": {morpheme_id: {"features":..., "scores":...}},
      }
    """
    registry = registry or derive_builtin_rules(build_default_rule_registry())
    pattern_weights = pattern_weights_from_flywheel_selection(selected_patterns, registry=registry)
    if not pattern_weights:
        return {
            "weights": {},
            "trace": [],
            "root_weights": {},
            "pattern_weights": {},
            "scored": {},
            "weight_sum": 0.0,
            "warning": "No valid selected pattern.",
        }

    node_by_id = {n.get("morpheme_id"): n for n in nodes if n.get("morpheme_id")}
    comps_by_parent = defaultdict(list)
    for c in compositions:
        p = c.get("parent_morpheme_id")
        if p:
            comps_by_parent[p].append(c)

    scored = precomputed_scores or score_morphemes_by_rules(nodes, registry)

    def combined_score(mid: str) -> float:
        s = scored.get(mid, {}).get("scores", {})
        return sum(pattern_weights.get(rid, 0.0) * safe_float(s.get(rid), 0.0) for rid in pattern_weights)

    def is_valid_mid(mid: str) -> bool:
        node = node_by_id.get(mid)
        if node is None:
            return False
        return _node_valid_for_diffusion(node, stable_only, min_support, min_source_glyphs, min_stroke_count, max_stroke_count)

    # Initial root mass.
    root_logits = {}
    for mid, node in node_by_id.items():
        if mid == "M_LINE":
            continue
        if not is_valid_mid(mid):
            continue
        pat = combined_score(mid)
        if pat <= 1e-9:
            continue
        root_logits[mid] = pat + support_alpha * math.log1p(max(0, int(node.get("source_glyph_count") or node.get("support_count") or 0)))

    root_logits = dict(sorted(root_logits.items(), key=lambda kv: kv[1], reverse=True)[:max(1, int(top_roots))])
    active = softmax_dict(root_logits, temperature=temperature)
    kept = defaultdict(float)
    trace = []

    for depth in range(int(max_depth) + 1):
        if not active:
            break

        next_active = defaultdict(float)
        input_mass = sum(active.values())
        kept_mass = 0.0
        pushed_mass = 0.0

        for mid, mass in active.items():
            rules = []
            for comp in comps_by_parent.get(mid, []):
                child_ids = list(comp.get("child_morpheme_ids") or [])
                if len(child_ids) != 2:
                    continue
                if not all(is_valid_mid(c) for c in child_ids):
                    continue

                child_pat = sum(combined_score(c) for c in child_ids) / max(1.0, len(child_ids))
                topo_pat = _rule_score_for_topology_way(comp, pattern_weights)
                rule_sup = math.log1p(max(0, int(comp.get("source_glyph_count") or comp.get("support_count") or 0)))
                logit = child_pattern_alpha * child_pat + topology_way_alpha * topo_pat + rule_support_alpha * rule_sup
                rules.append((comp, logit))

            rules.sort(key=lambda x: x[1], reverse=True)
            rules = rules[:max(1, int(top_rules_per_parent))]

            if depth >= int(max_depth) or not rules:
                kept[mid] += mass
                kept_mass += mass
                continue

            self_mass = mass * clamp01(keep_ratio)
            out_mass = mass - self_mass
            kept[mid] += self_mass
            kept_mass += self_mass

            rule_probs = softmax_dict({r[0].get("composition_id", str(i)): r[1] for i, r in enumerate(rules)}, temperature=temperature)
            comp_by_key = {r[0].get("composition_id", str(i)): r[0] for i, r in enumerate(rules)}

            for rid, rp in rule_probs.items():
                comp = comp_by_key[rid]
                child_ids = list(comp.get("child_morpheme_ids") or [])
                child_logits = {}
                for c in child_ids:
                    child_node = node_by_id.get(c, {})
                    child_logits[c] = combined_score(c) + support_alpha * math.log1p(max(0, int(child_node.get("source_glyph_count") or child_node.get("support_count") or 0)))
                child_probs = softmax_dict(child_logits, temperature=temperature)
                for c, cp in child_probs.items():
                    delta = out_mass * rp * cp
                    next_active[c] += delta
                    pushed_mass += delta

        trace.append({
            "depth": depth,
            "input_mass": round(float(input_mass), 12),
            "kept_mass": round(float(kept_mass), 12),
            "pushed_mass": round(float(pushed_mass), 12),
            "active_nodes": len(active),
            "next_active_nodes": len(next_active),
        })
        active = dict(next_active)

    # Should usually be empty after max_depth loop, but keep if any.
    for mid, mass in active.items():
        kept[mid] += mass

    weights = normalize_dict(dict(kept))
    return {
        "weights": weights,
        "weight_sum": float(sum(weights.values())),
        "trace": trace,
        "root_weights": active,
        "root_logits": root_logits,
        "pattern_weights": pattern_weights,
        "scored": scored,
        "registry_schema": registry.get("schema"),
    }


# =============================================================================
# 7. 辅助：候选输出
# =============================================================================

def top_weighted_morphemes(
    diffusion_result: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    *,
    top_k: int = 100,
) -> List[Dict[str, Any]]:
    node_by_id = {n.get("morpheme_id"): n for n in nodes if n.get("morpheme_id")}
    scored = diffusion_result.get("scored", {})
    rows = []
    for mid, w in sorted(diffusion_result.get("weights", {}).items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
        node = node_by_id.get(mid, {})
        scores = scored.get(mid, {}).get("scores", {})
        rows.append({
            "morpheme_id": mid,
            "weight": w,
            "stroke_count": node.get("stroke_count"),
            "is_stable": node.get("is_stable"),
            "support_count": node.get("support_count"),
            "source_glyph_count": node.get("source_glyph_count"),
            "top_pattern_scores": sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:10],
            "prototype_segments": node.get("prototype_segments", []),
        })
    return rows


def export_diffusion_result(
    diffusion_result: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    out_dir: str,
    *,
    top_k: int = 500,
) -> None:
    """
    不是必须给飞轮用；调试时可导出。
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = top_weighted_morphemes(diffusion_result, nodes, top_k=top_k)
    save_json({
        "weight_sum": diffusion_result.get("weight_sum"),
        "pattern_weights": diffusion_result.get("pattern_weights"),
        "trace": diffusion_result.get("trace"),
        "rows": rows,
    }, os.path.join(out_dir, "diffusion_result.json"), indent=2)

    with open(os.path.join(out_dir, "diffusion_result.csv"), "w", encoding="utf-8", newline="") as f:
        fields = ["morpheme_id", "weight", "stroke_count", "is_stable", "support_count", "source_glyph_count", "top_pattern_scores"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})



# =============================================================================
# 8. V2: 派生闭包 + child-softmax/product-AND parent keep 扩散
# =============================================================================

def activate_rule_closure(
    selected_patterns: Any,
    registry: Optional[Dict[str, Any]] = None,
    *,
    mode: str = "all_parents",
    full_weight: float = 1.0,
    partial_weight: float = 0.45,
    max_rounds: int = 4,
    include_auto_rules: bool = True,
) -> Dict[str, float]:
    """
    将飞轮勾选的规则做 activation closure。

    目的：
      如果用户勾选 chain + fork，那么 branch_chain / fork_with_tail 这类
      直接依赖它们的派生规则也应该被激活。

    参数：
      mode:
        "all_parents":
            派生 rule 的 parent_rules 全部被激活时，派生 rule 置为 full_weight。
            这是默认、最干净的逻辑。
        "any_parent":
            只要命中一个 parent_rule，就给 partial activation。
            全部命中时给 full_weight。
        "off":
            不做派生闭包。
      include_auto_rules:
            是否允许 auto_discovered 规则参与闭包。
    """
    registry = registry or derive_builtin_rules(build_default_rule_registry())
    rules = registry.get("rules", {})
    active = pattern_weights_from_flywheel_selection(selected_patterns, registry=registry)

    if mode == "off":
        return normalize_dict(active)

    for _ in range(max(1, int(max_rounds))):
        changed = False
        for rid, rule in rules.items():
            if rid in active and active[rid] >= full_weight:
                continue
            if (not include_auto_rules) and str(rule.get("source")) == "auto_discovered":
                continue

            parents = [p for p in (rule.get("parent_rules") or []) if p in rules]
            if not parents:
                continue

            parent_vals = [safe_float(active.get(p), 0.0) for p in parents]
            hit_count = sum(1 for v in parent_vals if v > 0)

            if mode == "all_parents":
                if hit_count == len(parents):
                    val = full_weight
                else:
                    continue
            elif mode == "any_parent":
                if hit_count == 0:
                    continue
                if hit_count == len(parents):
                    val = full_weight
                else:
                    val = partial_weight * (hit_count / max(1, len(parents)))
            else:
                continue

            if val > safe_float(active.get(rid), 0.0):
                active[rid] = float(val)
                changed = True

        if not changed:
            break

    return normalize_dict(active)


def _child_softmax_and_score(
    child_ids: List[str],
    node_by_id: Dict[str, Dict[str, Any]],
    scored: Dict[str, Dict[str, Any]],
    pattern_weights: Dict[str, float],
    *,
    support_alpha: float,
    temperature: float,
    and_power: float,
    and_normalize_for_binary: bool,
) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    """
    对 child morphemes 做 softmax，并返回 product-AND score。

    child_probs:
        子语素之间的 softmax 分配关系。

    and_score:
        对二元组合，默认就是 p_a * p_b。
        如果 and_normalize_for_binary=True，则变成 4 * p_a * p_b，使平衡二元组合最大为 1。
        但默认 False，因为用户原始设想就是“两个子语素相乘”。
    """
    logits = {}
    raw_scores = {}
    for c in child_ids:
        node = node_by_id.get(c, {})
        scores = scored.get(c, {}).get("scores", {})
        pat = sum(pattern_weights.get(rid, 0.0) * safe_float(scores.get(rid), 0.0) for rid in pattern_weights)
        sup = math.log1p(max(0, int(node.get("source_glyph_count") or node.get("support_count") or 0)))
        logits[c] = pat + support_alpha * sup
        raw_scores[c] = pat

    child_probs = softmax_dict(logits, temperature=temperature)

    prod = 1.0
    for c in child_ids:
        prod *= max(1e-12, safe_float(child_probs.get(c), 0.0))

    if and_normalize_for_binary and len(child_ids) == 2:
        prod *= 4.0

    and_score = clamp01(prod ** max(1e-6, float(and_power)))
    return child_probs, and_score, raw_scores


def diffuse_by_flywheel_selection_v2(
    nodes: List[Dict[str, Any]],
    compositions: List[Dict[str, Any]],
    selected_patterns: Any,
    *,
    registry: Optional[Dict[str, Any]] = None,
    precomputed_scores: Optional[Dict[str, Dict[str, Any]]] = None,

    # filters
    stable_only: bool = True,
    min_support: int = 1,
    min_source_glyphs: int = 1,
    min_stroke_count: int = 1,
    max_stroke_count: int = 8,

    # rule activation
    activate_derived: bool = True,
    derived_activation_mode: str = "all_parents",
    derived_full_weight: float = 1.0,
    derived_partial_weight: float = 0.45,
    include_auto_rules_in_closure: bool = True,

    # diffusion
    temperature: float = 0.75,
    max_depth: int = 4,
    top_roots: int = 512,
    top_rules_per_parent: int = 8,
    support_alpha: float = 0.12,
    rule_support_alpha: float = 0.18,
    topology_way_alpha: float = 0.35,
    child_pattern_alpha: float = 1.0,

    # parent keep from child product
    keep_floor: float = 0.08,
    keep_strength: float = 0.72,
    and_power: float = 1.0,
    and_normalize_for_binary: bool = False,
    and_rule_alpha: float = 0.65,
) -> Dict[str, Any]:
    """
    V2 主扩散函数。推荐飞轮脚本调用这个。

    和旧版的区别：
      1. 支持 rule activation closure：
         chain + fork 被选中时，branch_chain / fork_with_tail 等派生规则会自动激活。
      2. 对 parent <- child_a + child_b 的每条 composition rule：
         child_a / child_b 先做 softmax。
      3. parent 的保留质量由 child product-AND 决定：
         and_score = p(child_a) * p(child_b)
         parent_keep = keep_floor + keep_strength * and_score
      4. 每轮质量守恒，最后归一化，weights 总和为 1。
    """
    registry = registry or derive_builtin_rules(build_default_rule_registry())

    if activate_derived:
        pattern_weights = activate_rule_closure(
            selected_patterns,
            registry,
            mode=derived_activation_mode,
            full_weight=derived_full_weight,
            partial_weight=derived_partial_weight,
            include_auto_rules=include_auto_rules_in_closure,
        )
    else:
        pattern_weights = pattern_weights_from_flywheel_selection(selected_patterns, registry=registry)

    if not pattern_weights:
        return {
            "weights": {},
            "weight_sum": 0.0,
            "trace": [],
            "initial_root_weights": {},
            "pattern_weights": {},
            "scored": {},
            "warning": "No valid selected pattern after activation closure.",
        }

    node_by_id = {n.get("morpheme_id"): n for n in nodes if n.get("morpheme_id")}
    comps_by_parent = defaultdict(list)
    for c in compositions:
        p = c.get("parent_morpheme_id")
        if p:
            comps_by_parent[p].append(c)

    scored = precomputed_scores or score_morphemes_by_rules(nodes, registry)

    def combined_score(mid: str) -> float:
        scores = scored.get(mid, {}).get("scores", {})
        return sum(pattern_weights.get(rid, 0.0) * safe_float(scores.get(rid), 0.0) for rid in pattern_weights)

    def is_valid_mid(mid: str) -> bool:
        node = node_by_id.get(mid)
        if node is None:
            return False
        return _node_valid_for_diffusion(node, stable_only, min_support, min_source_glyphs, min_stroke_count, max_stroke_count)

    # root distribution
    root_logits = {}
    for mid, node in node_by_id.items():
        if mid == "M_LINE":
            continue
        if not is_valid_mid(mid):
            continue
        pat = combined_score(mid)
        if pat <= 1e-12:
            continue
        sup = math.log1p(max(0, int(node.get("source_glyph_count") or node.get("support_count") or 0)))
        root_logits[mid] = pat + support_alpha * sup

    root_logits = dict(sorted(root_logits.items(), key=lambda kv: kv[1], reverse=True)[:max(1, int(top_roots))])
    active = softmax_dict(root_logits, temperature=temperature)
    initial_root_weights = dict(active)

    kept = defaultdict(float)
    trace = []

    for depth in range(int(max_depth) + 1):
        if not active:
            break

        next_active = defaultdict(float)
        input_mass = float(sum(active.values()))
        kept_mass = 0.0
        pushed_mass = 0.0
        rule_count = 0

        for parent_id, parent_mass in active.items():
            candidate_rules = []
            for comp in comps_by_parent.get(parent_id, []):
                child_ids = list(comp.get("child_morpheme_ids") or [])
                if len(child_ids) != 2:
                    continue
                if not all(is_valid_mid(c) for c in child_ids):
                    continue

                child_probs, and_score, child_raw_scores = _child_softmax_and_score(
                    child_ids,
                    node_by_id,
                    scored,
                    pattern_weights,
                    support_alpha=support_alpha,
                    temperature=temperature,
                    and_power=and_power,
                    and_normalize_for_binary=and_normalize_for_binary,
                )

                child_pair_score = sum(child_raw_scores.values()) / max(1.0, len(child_raw_scores))
                topo_score = _rule_score_for_topology_way(comp, pattern_weights)
                rule_sup = math.log1p(max(0, int(comp.get("source_glyph_count") or comp.get("support_count") or 0)))

                rule_logit = (
                    child_pattern_alpha * child_pair_score
                    + and_rule_alpha * and_score
                    + topology_way_alpha * topo_score
                    + rule_support_alpha * rule_sup
                )

                candidate_rules.append({
                    "comp": comp,
                    "logit": rule_logit,
                    "child_probs": child_probs,
                    "and_score": and_score,
                    "child_raw_scores": child_raw_scores,
                    "topology_way_score": topo_score,
                })

            candidate_rules.sort(key=lambda x: x["logit"], reverse=True)
            candidate_rules = candidate_rules[:max(1, int(top_rules_per_parent))]

            if depth >= int(max_depth) or not candidate_rules:
                kept[parent_id] += parent_mass
                kept_mass += parent_mass
                continue

            rule_logits = {
                r["comp"].get("composition_id", str(i)): r["logit"]
                for i, r in enumerate(candidate_rules)
            }
            rule_probs = softmax_dict(rule_logits, temperature=temperature)
            by_key = {
                r["comp"].get("composition_id", str(i)): r
                for i, r in enumerate(candidate_rules)
            }

            for rule_key, rule_prob in rule_probs.items():
                r = by_key[rule_key]
                comp = r["comp"]
                child_ids = list(comp.get("child_morpheme_ids") or [])
                and_score = r["and_score"]
                parent_keep = clamp01(keep_floor + keep_strength * and_score)

                local_mass = parent_mass * rule_prob
                parent_delta = local_mass * parent_keep
                child_mass = local_mass - parent_delta

                kept[parent_id] += parent_delta
                kept_mass += parent_delta

                for c, cp in r["child_probs"].items():
                    delta = child_mass * cp
                    next_active[c] += delta
                    pushed_mass += delta

                rule_count += 1

        trace.append({
            "depth": depth,
            "input_mass": round(input_mass, 12),
            "kept_mass": round(float(kept_mass), 12),
            "pushed_mass": round(float(pushed_mass), 12),
            "active_nodes": len(active),
            "next_active_nodes": len(next_active),
            "expanded_rule_count": rule_count,
        })
        active = dict(next_active)

    for mid, mass in active.items():
        kept[mid] += mass

    weights = normalize_dict(dict(kept))

    return {
        "weights": weights,
        "weight_sum": float(sum(weights.values())),
        "trace": trace,
        "initial_root_weights": initial_root_weights,
        "root_logits": root_logits,
        "pattern_weights": pattern_weights,
        "scored": scored,
        "registry_schema": registry.get("schema"),
        "diffusion_version": "v2_child_softmax_product_and_parent_keep",
        "params": {
            "activate_derived": activate_derived,
            "derived_activation_mode": derived_activation_mode,
            "keep_floor": keep_floor,
            "keep_strength": keep_strength,
            "and_power": and_power,
            "and_normalize_for_binary": and_normalize_for_binary,
            "temperature": temperature,
            "max_depth": max_depth,
        },
    }


# =============================================================================
# 9. 飞轮按钮 1：已标记库偏移按钮
# =============================================================================

def compute_marked_library_rule_distribution(
    nodes: List[Dict[str, Any]],
    registry: Optional[Dict[str, Any]] = None,
    *,
    marked_morpheme_ids: Optional[Iterable[str]] = None,
    marked_weights: Optional[Dict[str, float]] = None,
    stable_only: bool = False,
    min_support: int = 1,
    min_source_glyphs: int = 1,
    score_threshold: float = 0.25,
    precomputed_scores: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    统计“已标记库”里的 topology rule 分布。

    marked_morpheme_ids:
      - 如果传入，只统计这些 morpheme。
      - 如果不传入，则默认统计 nodes 中符合 filter 的全部 morpheme。
      - 飞轮里可以把 Good/Cleaned/人工确认过的 morpheme id 传进来。

    marked_weights:
      - 可选。某些已标记样本更重要时，可以给 morpheme_id 额外权重。

    输出：
      rule_mass: 连续质量，按 rule score 累加。
      rule_prob: rule_mass 归一化。
      rule_hit_count: score 超过 threshold 的命中次数。
      unseen_rules: 在已标记库中几乎没出现的规则。
    """
    registry = registry or derive_builtin_rules(build_default_rule_registry())
    rules = registry.get("rules", {})
    scored = precomputed_scores or score_morphemes_by_rules(nodes, registry)
    node_by_id = {n.get("morpheme_id"): n for n in nodes if n.get("morpheme_id")}

    if marked_morpheme_ids is None:
        mids = []
        for mid, node in node_by_id.items():
            if mid == "M_LINE":
                continue
            if stable_only and not node.get("is_stable"):
                continue
            if int(node.get("support_count") or 0) < int(min_support):
                continue
            if int(node.get("source_glyph_count") or 0) < int(min_source_glyphs):
                continue
            mids.append(mid)
    else:
        mids = [str(x) for x in marked_morpheme_ids if str(x) in node_by_id]

    marked_weights = marked_weights or {}
    rule_mass = {rid: 0.0 for rid in rules}
    rule_hit_count = {rid: 0 for rid in rules}
    total_item_weight = 0.0

    for mid in mids:
        node = node_by_id.get(mid, {})
        base_w = safe_float(marked_weights.get(mid), 0.0)
        if base_w <= 0:
            # 默认用 source_glyph_count 的轻微加权，避免大支持语素和一次性语素完全等价。
            base_w = 1.0 + 0.15 * math.log1p(max(0, int(node.get("source_glyph_count") or node.get("support_count") or 0)))

        scores = scored.get(mid, {}).get("scores", {})
        total_item_weight += base_w
        for rid in rules:
            s = clamp01(safe_float(scores.get(rid), 0.0))
            rule_mass[rid] += base_w * s
            if s >= score_threshold:
                rule_hit_count[rid] += 1

    rule_prob = normalize_dict(rule_mass)
    unseen = [rid for rid, v in rule_mass.items() if v <= 1e-9]

    return {
        "schema": "marked_library_rule_distribution.v1",
        "marked_count": len(mids),
        "total_item_weight": total_item_weight,
        "rule_mass": rule_mass,
        "rule_prob": rule_prob,
        "rule_hit_count": rule_hit_count,
        "unseen_rules": unseen,
        "score_threshold": score_threshold,
    }


def build_marked_library_shift(
    distribution: Dict[str, Any],
    registry: Optional[Dict[str, Any]] = None,
    *,
    mode: str = "both",
    frequent_strength: float = 0.75,
    rare_strength: float = 0.35,
    floor: float = 1e-6,
) -> Dict[str, Any]:
    """
    根据已标记库分布构造全局 rule multiplier。

    mode:
      "reinforce_frequent":
          出现多的 topology rule 权重增加。
      "boost_rare":
          出现少/没出现的 topology rule 权重增加，用来鼓励探索。
      "both":
          两者都做：常见结构保留风格，稀有结构补充多样性。
          这正适合“已标记库偏移按钮”的飞轮语义。

    返回：
      multipliers: {rule_id: multiplier}
    """
    registry = registry or derive_builtin_rules(build_default_rule_registry())
    rules = registry.get("rules", {})
    prob = distribution.get("rule_prob") or {}
    if not prob:
        prob = {rid: 0.0 for rid in rules}

    max_p = max([safe_float(prob.get(rid), 0.0) for rid in rules] + [floor])
    multipliers = {}

    for rid in rules:
        p = safe_float(prob.get(rid), 0.0)
        freq_norm = clamp01(p / max_p)
        rare_norm = clamp01(1.0 - freq_norm)

        m = 1.0
        if mode in ("reinforce_frequent", "both"):
            m += frequent_strength * freq_norm
        if mode in ("boost_rare", "both"):
            m += rare_strength * rare_norm

        multipliers[rid] = float(max(floor, m))

    return {
        "schema": "marked_library_shift.v1",
        "mode": mode,
        "frequent_strength": frequent_strength,
        "rare_strength": rare_strength,
        "multipliers": multipliers,
    }


def apply_marked_library_shift_to_selection(
    selected_patterns: Any,
    shift: Dict[str, Any],
    registry: Optional[Dict[str, Any]] = None,
    *,
    include_unselected: bool = False,
    unselected_base_weight: float = 0.05,
) -> Dict[str, float]:
    """
    将“已标记库偏移”应用到飞轮勾选规则上。

    include_unselected=False:
      只调整已勾选规则。
    include_unselected=True:
      未勾选规则也给一个很小 base weight，用于探索。
    """
    registry = registry or derive_builtin_rules(build_default_rule_registry())
    rules = registry.get("rules", {})
    base = pattern_weights_from_flywheel_selection(selected_patterns, registry=registry)
    mult = shift.get("multipliers") or {}

    out = {}
    for rid in rules:
        b = safe_float(base.get(rid), 0.0)
        if b <= 0 and include_unselected:
            b = unselected_base_weight
        if b <= 0:
            continue
        out[rid] = b * safe_float(mult.get(rid), 1.0)

    return normalize_dict(out)


def button_marked_library_shift(
    nodes: List[Dict[str, Any]],
    selected_patterns: Any,
    *,
    registry: Optional[Dict[str, Any]] = None,
    marked_morpheme_ids: Optional[Iterable[str]] = None,
    marked_weights: Optional[Dict[str, float]] = None,
    mode: str = "both",
    include_unselected: bool = False,
    stable_only: bool = False,
    frequent_strength: float = 0.75,
    rare_strength: float = 0.35,
) -> Dict[str, Any]:
    """
    飞轮按钮 1：已标记库偏移按钮。

    GUI 按下后建议调用：
        pack = button_marked_library_shift(nodes, selected_patterns, registry=registry)
        selected_patterns = pack["adjusted_selection"]

    然后把 adjusted_selection 传给 diffuse_by_flywheel_selection_v2。
    """
    registry = registry or derive_builtin_rules(build_default_rule_registry())

    dist = compute_marked_library_rule_distribution(
        nodes,
        registry,
        marked_morpheme_ids=marked_morpheme_ids,
        marked_weights=marked_weights,
        stable_only=stable_only,
    )
    shift = build_marked_library_shift(
        dist,
        registry,
        mode=mode,
        frequent_strength=frequent_strength,
        rare_strength=rare_strength,
    )
    adjusted = apply_marked_library_shift_to_selection(
        selected_patterns,
        shift,
        registry,
        include_unselected=include_unselected,
    )

    return {
        "adjusted_selection": adjusted,
        "distribution": dist,
        "shift": shift,
        "button": "marked_library_shift",
    }


# =============================================================================
# 10. 飞轮按钮 2：派生规则增加按钮 / 新模式规则影响
# =============================================================================

def discovered_rule_selection_weights(
    discovered: Dict[str, Any],
    *,
    base_weight: float = 0.35,
    use_cluster_size: bool = True,
) -> Dict[str, float]:
    """
    将 discover_new_pattern_rules 的 auto rules 转成飞轮 selection 权重。
    """
    clusters = discovered.get("clusters") or []
    if not clusters:
        return {}

    max_size = max([int(c.get("size") or 0) for c in clusters] + [1])
    out = {}
    for c in clusters:
        rid = c.get("auto_pattern_id")
        if not rid:
            continue
        if use_cluster_size:
            w = base_weight * (int(c.get("size") or 0) / max(1, max_size))
        else:
            w = base_weight
        if w > 0:
            out[str(rid)] = float(w)

    return out


def button_enable_discovered_rule_influence(
    nodes: List[Dict[str, Any]],
    selected_patterns: Any,
    *,
    registry: Optional[Dict[str, Any]] = None,
    discover_k: int = 16,
    auto_rule_weight: float = 0.35,
    stable_only: bool = True,
    min_support: int = 1,
    min_source_glyphs: int = 1,
    include_auto_in_selection: bool = True,
    merge_into_registry: bool = True,
    seed: int = 20260702,
) -> Dict[str, Any]:
    """
    飞轮按钮 2：派生规则增加按钮。

    按下后：
      1. 从当前语素库聚类探索新模式。
      2. 将 auto_pattern_xxx 转成普通 rule。
      3. 合并到 registry。
      4. 可选：把 auto rules 加入 selected_patterns，使后续新拓扑结构受这些新 rule 影响。

    返回：
      {
        "registry": new_registry,
        "adjusted_selection": new_selected_patterns,
        "discovered": discovered,
      }
    """
    registry = registry or derive_builtin_rules(build_default_rule_registry())

    discovered = discover_new_pattern_rules(
        nodes,
        registry,
        k=discover_k,
        stable_only=stable_only,
        min_support=min_support,
        min_source_glyphs=min_source_glyphs,
        seed=seed,
    )

    if merge_into_registry:
        new_registry = merge_auto_rules(registry, discovered)
    else:
        new_registry = registry

    base_selection = pattern_weights_from_flywheel_selection(selected_patterns, registry=new_registry)

    if include_auto_in_selection:
        auto_weights = discovered_rule_selection_weights(
            discovered,
            base_weight=auto_rule_weight,
            use_cluster_size=True,
        )
        mixed = dict(base_selection)
        mixed.update(auto_weights)
        adjusted = normalize_dict(mixed)
    else:
        adjusted = normalize_dict(base_selection)

    return {
        "registry": new_registry,
        "adjusted_selection": adjusted,
        "discovered": discovered,
        "auto_selection_weights": discovered_rule_selection_weights(discovered, base_weight=auto_rule_weight),
        "button": "enable_discovered_rule_influence",
    }


def prepare_flywheel_pattern_context_v2(
    nodes: List[Dict[str, Any]],
    selected_patterns: Any,
    *,
    registry: Optional[Dict[str, Any]] = None,
    enable_marked_library_shift: bool = False,
    enable_discovered_rule_influence: bool = False,
    marked_morpheme_ids: Optional[Iterable[str]] = None,
    marked_weights: Optional[Dict[str, float]] = None,
    marked_shift_mode: str = "both",
    discover_k: int = 16,
    auto_rule_weight: float = 0.35,
) -> Dict[str, Any]:
    """
    飞轮集成用的上下文准备函数。
    如果 GUI 有两个按钮，可以在调用 diffusion 前统一调用这里。

    返回：
      {
        "registry": registry,
        "selected_patterns": selected_patterns_after_buttons,
        "marked_shift_pack": ... or None,
        "discovered_pack": ... or None
      }
    """
    registry = registry or derive_builtin_rules(build_default_rule_registry())
    selected = selected_patterns
    marked_pack = None
    discovered_pack = None

    if enable_discovered_rule_influence:
        discovered_pack = button_enable_discovered_rule_influence(
            nodes,
            selected,
            registry=registry,
            discover_k=discover_k,
            auto_rule_weight=auto_rule_weight,
            include_auto_in_selection=True,
            merge_into_registry=True,
        )
        registry = discovered_pack["registry"]
        selected = discovered_pack["adjusted_selection"]

    if enable_marked_library_shift:
        marked_pack = button_marked_library_shift(
            nodes,
            selected,
            registry=registry,
            marked_morpheme_ids=marked_morpheme_ids,
            marked_weights=marked_weights,
            mode=marked_shift_mode,
            include_unselected=False,
        )
        selected = marked_pack["adjusted_selection"]

    return {
        "registry": registry,
        "selected_patterns": selected,
        "marked_shift_pack": marked_pack,
        "discovered_pack": discovered_pack,
    }


# 为了避免旧飞轮误用旧算法，这里提供一个显式别名。
diffuse_by_flywheel_selection_latest = diffuse_by_flywheel_selection_v2


if __name__ == "__main__":
    nodes, comps = load_morpheme_tree(DEFAULT_TREE_DIR)
    registry = derive_builtin_rules(build_default_rule_registry())
    print("[SmokeTest V2] nodes=", len(nodes), "compositions=", len(comps), "rules=", len(registry.get("rules", {})))
    if nodes and comps:
        ctx = prepare_flywheel_pattern_context_v2(
            nodes,
            {"star": 1.0, "fork": 0.8, "ladder": 0.6, "cycle_with_tail": 0.5},
            registry=registry,
            enable_marked_library_shift=False,
            enable_discovered_rule_influence=False,
        )
        res = diffuse_by_flywheel_selection_v2(
            nodes,
            comps,
            ctx["selected_patterns"],
            registry=ctx["registry"],
            stable_only=True,
            max_depth=3,
        )
        print("[SmokeTest V2] weight_sum=", res["weight_sum"], "nonzero=", len(res["weights"]))
        print("[SmokeTest V2] trace=", res["trace"])
        print("[SmokeTest V2] top=", top_weighted_morphemes(res, nodes, top_k=5))
