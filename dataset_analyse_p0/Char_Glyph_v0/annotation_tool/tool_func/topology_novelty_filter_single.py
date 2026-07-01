# -*- coding: utf-8 -*-
r"""
topology_novelty_filter_single.py

单文件版：PCG / FontGPT 飞轮候选拓扑去重 + 当前字体库缓存。

推荐放置位置：
    Babel_Simulater\dataset_analyse_p0\Char_Glyph_v0\annotation_tool\tool_func\topology_novelty_filter_single.py

默认读取当前正向字体库：
    1. Char_Glyph_v0\annotation_tool\pcg_filebacked_stage2_schema\good
    2. Char_Glyph_v0\annotation_tool\pcg_filebacked_stage2_schema\cleaned
    3. dataset_analyse_p0\AI_VECTOR_ROUTER_With_topo\annotations_topo

默认不读取 bad，因为负样本越多越好，bad 不应该用于“避免重复正样本拓扑”的历史库。

核心功能：
    - 为 candidate / bundle 计算:
        strict_topo_hash
        family_hash
        geometry_hash
    - 构建/读取缓存，单个 JSON shard <= 80MB
    - 如果源文件 path/size/mtime_ns 没变化，直接读取缓存
    - 如果 good -> cleaned 发生移动，path 变化，manifest hash 会变化，因此自动重建缓存
    - 支持多进程榨干 CPU 做缓存构建
    - 在线生成时可直接调用 TopologyNoveltyFilter.check(cand)

CLI:
    cd Babel_Simulater\dataset_analyse_p0\Char_Glyph_v0\annotation_tool
    python tool_func\topology_novelty_filter_single.py

强制重建:
    python tool_func\topology_novelty_filter_single.py --force --workers auto

在 GUI 中调用示例:
    from tool_func.topology_novelty_filter_single import (
        build_or_load_default_positive_cache,
        TopologyNoveltyFilter,
    )

    cache = build_or_load_default_positive_cache(max_json_mb=80, workers="auto")
    novelty = TopologyNoveltyFilter(cache, max_family_per_batch=2)
"""

from __future__ import annotations

import os
import re
import json
import math
import time
import copy
import argparse
import hashlib
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set, DefaultDict
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


# =============================================================================
# 0. 默认路径
# =============================================================================

CACHE_VERSION = "topology_novelty_positive_cache_v2"
DEFAULT_MAX_JSON_MB = 80
DEFAULT_CANVAS_NORM = 400.0


def _infer_dirs() -> Dict[str, str]:
    """
    假设本文件位于:
        Char_Glyph_v0/annotation_tool/tool_func/topology_novelty_filter_single.py

    同时兼容你临时从 annotation_tool 或其他目录运行。
    """
    this_file = os.path.abspath(__file__)
    tool_func_dir = os.path.dirname(this_file)

    # 标准位置: .../Char_Glyph_v0/annotation_tool/tool_func
    if os.path.basename(tool_func_dir).lower() == "tool_func":
        annotation_tool_dir = os.path.abspath(os.path.join(tool_func_dir, ".."))
    else:
        annotation_tool_dir = tool_func_dir

    if os.path.basename(annotation_tool_dir).lower() == "annotation_tool":
        char_glyph_dir = os.path.abspath(os.path.join(annotation_tool_dir, ".."))
    else:
        # 兼容：如果直接放在 Char_Glyph_v0 下
        char_glyph_dir = annotation_tool_dir

    dataset_analyse_dir = os.path.abspath(os.path.join(char_glyph_dir, ".."))

    return {
        "this_file": this_file,
        "tool_func_dir": tool_func_dir,
        "annotation_tool_dir": annotation_tool_dir,
        "char_glyph_dir": char_glyph_dir,
        "dataset_analyse_dir": dataset_analyse_dir,
        "pcg_pool_root": os.path.join(annotation_tool_dir, "pcg_filebacked_stage2_schema"),
        "good_dir": os.path.join(annotation_tool_dir, "pcg_filebacked_stage2_schema", "good"),
        "cleaned_dir": os.path.join(annotation_tool_dir, "pcg_filebacked_stage2_schema", "cleaned"),
        "annotations_topo_dir": os.path.join(dataset_analyse_dir, "AI_VECTOR_ROUTER_With_topo", "annotations_topo"),
        "cache_dir": os.path.join(tool_func_dir, "topology_positive_cache"),
    }


def default_positive_source_roots() -> List[str]:
    d = _infer_dirs()
    return [d["good_dir"], d["cleaned_dir"], d["annotations_topo_dir"]]


def default_cache_dir() -> str:
    return _infer_dirs()["cache_dir"]


# =============================================================================
# 1. 通用工具
# =============================================================================

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha1_text(s: str, n: int = 24) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def sha1_obj(obj: Any, n: int = 24) -> str:
    return sha1_text(stable_json_dumps(obj), n=n)


def json_size_bytes(obj: Any) -> int:
    return len(stable_json_dumps(obj).encode("utf-8"))


def atomic_save_json(obj: Any, path: str, indent: Optional[int] = None) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, sort_keys=True)
    os.replace(tmp, path)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _norm_path(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")


def iter_json_files(root: str) -> Iterable[str]:
    if not root or not os.path.exists(root):
        return
    if os.path.isfile(root) and root.lower().endswith(".json"):
        yield os.path.abspath(root)
        return
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            low = fn.lower()
            if low.endswith(".json") and not low.endswith(".tmp"):
                yield os.path.abspath(os.path.join(dirpath, fn))


def quick_file_token(path: str, root_label: str = "") -> Dict[str, Any]:
    """
    快速检测源文件是否变化。

    注意：这里保留 path 和 root_label。
    如果某个 JSON 从 good 移动到 cleaned，即使 size/mtime 差不多，path/root_label 也会变化，
    source manifest hash 会变化，从而触发缓存重建。
    """
    st = os.stat(path)
    return {
        "source": str(root_label),
        "path": _norm_path(path),
        "basename": os.path.basename(path),
        "size": int(st.st_size),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
    }


def build_source_manifest(source_roots: Iterable[str]) -> Dict[str, Any]:
    roots = [os.path.abspath(r) for r in source_roots if r and os.path.exists(r)]
    files = []
    for root in roots:
        root_norm = _norm_path(root)
        label = os.path.basename(root_norm.rstrip("/")) or root_norm
        for p in iter_json_files(root) or []:
            try:
                files.append(quick_file_token(p, root_label=label))
            except FileNotFoundError:
                pass
    files.sort(key=lambda x: (x["source"], x["path"], x["size"], x["mtime_ns"]))
    return {
        "cache_version": CACHE_VERSION,
        "created_from_roots": [_norm_path(r) for r in roots],
        "file_count": len(files),
        "files": files,
        "manifest_hash": sha1_obj(files, n=32),
    }


def resolve_workers(workers: Any) -> int:
    if workers is None or workers == "auto":
        cpu = os.cpu_count() or 1
        return max(1, cpu - 1)
    try:
        w = int(workers)
        return max(1, w)
    except Exception:
        return 1


# =============================================================================
# 2. JSON 记录抽取
# =============================================================================

def flatten_json_records(obj: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """
    尽量兼容已有 good/cleaned/annotations_topo 结构：
      - {hex_key: bundle}
      - {"items": [...]}
      - [{"glyph_info":..., "strokes":...}, ...]
      - {"good": [...], "cleaned": [...]}
      - 单个 bundle
    """
    if obj is None:
        return

    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                yield str(i), item
        return

    if not isinstance(obj, dict):
        return

    # 单个 bundle / candidate
    if isinstance(obj.get("strokes"), list) or isinstance(obj.get("solved_nodes"), list) or isinstance(obj.get("nodes"), list):
        cid = (
            obj.get("candidate_id")
            or obj.get("generated_glyph_id")
            or (obj.get("glyph_info") or {}).get("candidate_id")
            or (obj.get("glyph_info") or {}).get("hex_key")
            or "direct"
        )
        yield str(cid), obj
        return

    # 常见容器
    for key in ["items", "records", "candidates", "bundles", "good", "cleaned", "data"]:
        v = obj.get(key)
        if isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    yield f"{key}:{i}", item

    # hex_key -> bundle mapping
    for k, v in obj.items():
        if isinstance(v, dict) and (
            isinstance(v.get("strokes"), list)
            or isinstance(v.get("solved_nodes"), list)
            or isinstance(v.get("nodes"), list)
        ):
            yield str(k), v


# =============================================================================
# 3. Candidate / Bundle 解析
# =============================================================================

def _as_np_bezier(v: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(v, dtype=np.float32)
        if arr.shape == (4, 2):
            return arr
    except Exception:
        return None
    return None


def _get_candidate_id(candidate: Dict[str, Any], fallback: str = "") -> str:
    gi = candidate.get("glyph_info", {}) if isinstance(candidate.get("glyph_info"), dict) else {}
    for k in [
        "generated_glyph_id",
        "source_candidate_id",
        "candidate_id",
        "glyph_candidate_id",
        "sample_id",
        "grammar_sample_id",
        "id",
    ]:
        if candidate.get(k):
            return str(candidate[k])
    for k in ["candidate_id", "hex_key", "char"]:
        if gi.get(k):
            return str(gi[k])
    return str(fallback or "unknown")


def _node_list(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    for k in ["strokes", "solved_nodes", "nodes", "solved_segments"]:
        v = candidate.get(k)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    return []


def _get_bezier(node: Dict[str, Any]) -> Optional[np.ndarray]:
    for k in ["mother_bezier", "bezier", "curve", "solved_bezier", "control_points", "path"]:
        arr = _as_np_bezier(node.get(k))
        if arr is not None:
            return arr
    return None


def _get_width(node: Dict[str, Any], default: float = 1.0) -> float:
    for k in ["width", "width_mean", "stroke_width"]:
        try:
            w = float(node.get(k))
            if w > 0:
                return w
        except Exception:
            pass
    wb = node.get("width_bezier")
    if isinstance(wb, list) and wb:
        try:
            vals = [float(x) for x in wb]
            vals = [x for x in vals if x > 0]
            if vals:
                return float(sum(vals) / len(vals))
        except Exception:
            pass
    return float(default)


def extract_strokes(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    标准化 stroke:
        {"id": int, "P": np.ndarray(4,2), "width": float}
    """
    strokes = []
    for i, nd in enumerate(_node_list(candidate)):
        P = _get_bezier(nd)
        if P is None:
            continue
        bid = nd.get("bezier_id", nd.get("id", nd.get("node_id", i + 1)))
        try:
            bid = int(bid)
        except Exception:
            bid = i + 1
        strokes.append({
            "id": bid,
            "P": np.asarray(P, dtype=np.float32),
            "width": float(_get_width(nd, default=1.0)),
        })
    return strokes


# =============================================================================
# 4. 拓扑 / 几何签名
# =============================================================================

def _cluster_points(points: List[np.ndarray], tol: float) -> Tuple[List[int], List[np.ndarray]]:
    """
    端点聚类。n 很小，O(n^2) 足够快。
    """
    clusters: List[List[np.ndarray]] = []
    labels: List[int] = []
    for p in points:
        assigned = -1
        for ci, ps in enumerate(clusters):
            center = np.mean(np.stack(ps, axis=0), axis=0)
            if float(np.linalg.norm(p - center)) <= tol:
                assigned = ci
                break
        if assigned < 0:
            clusters.append([p])
            labels.append(len(clusters) - 1)
        else:
            clusters[assigned].append(p)
            labels.append(assigned)
    centers = [np.mean(np.stack(ps, axis=0), axis=0).astype(np.float32) for ps in clusters]
    return labels, centers


def _relation_edges_from_events(candidate: Dict[str, Any]) -> List[Tuple[int, int, str]]:
    events = candidate.get("topology_events", [])
    if not isinstance(events, list):
        return []
    out = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        typ = str(ev.get("type", "REL")).upper()
        if typ == "E2E":
            a, b = ev.get("stroke_a"), ev.get("stroke_b")
        elif typ == "T":
            a, b = ev.get("guest"), ev.get("host")
        elif typ == "X":
            a, b = ev.get("stroke_a"), ev.get("stroke_b")
        else:
            continue
        try:
            a, b = int(a), int(b)
        except Exception:
            continue
        if a == b:
            continue
        out.append((min(a, b), max(a, b), typ))
    return sorted(set(out))


def _skeleton_graph_from_endpoints(strokes: List[Dict[str, Any]], endpoint_tol: float) -> Tuple[List[int], List[Tuple[int, int]]]:
    raw_points = []
    for st in strokes:
        P = st["P"]
        raw_points.append(np.asarray(P[0], dtype=np.float32))
        raw_points.append(np.asarray(P[3], dtype=np.float32))

    if not raw_points:
        return [], []

    labels, centers = _cluster_points(raw_points, endpoint_tol)
    edges = []
    for si, _st in enumerate(strokes):
        a = labels[2 * si]
        b = labels[2 * si + 1]
        if a == b:
            continue
        edges.append((min(a, b), max(a, b)))
    nodes = list(range(len(centers)))
    return nodes, sorted(set(edges))


def _connected_components(nodes: List[int], edges: List[Tuple[int, int]]) -> int:
    if not nodes:
        return 0
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        if a not in parent:
            parent[a] = a
        if b not in parent:
            parent[b] = b
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)
    return len(set(find(n) for n in parent))


def _cycle_rank(nodes: List[int], edges: List[Tuple[int, int]]) -> int:
    return max(0, len(set(tuple(sorted(e)) for e in edges)) - len(nodes) + _connected_components(nodes, edges))


def _degree_hist(nodes: List[int], edges: List[Tuple[int, int]]) -> Tuple[int, ...]:
    deg = Counter()
    for n in nodes:
        deg[n] = 0
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    return tuple(sorted(deg.values()))


def _wl_hash(nodes: List[int], edges: List[Tuple[int, int]], rounds: int = 3) -> str:
    """
    Weisfeiler-Lehman color refinement hash。
    不是严格图同构证明，但对 PCG 小图非常高效实用。
    """
    adj: DefaultDict[int, List[int]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    colors = {n: f"N:deg{len(adj[n])}" for n in nodes}
    history = []
    for _ in range(rounds):
        new_colors = {}
        for n in nodes:
            neigh = sorted(colors.get(nb, "UNK") for nb in adj[n])
            new_colors[n] = sha1_obj([colors[n], neigh], n=16)
        colors = new_colors
        history.append(sorted(colors.values()))
    return sha1_obj(history, n=24)


def _normalized_endpoint_cloud(strokes: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    pts = []
    edges = []
    for st in strokes:
        P = st["P"]
        a = len(pts)
        pts.append(np.asarray(P[0], dtype=np.float32))
        b = len(pts)
        pts.append(np.asarray(P[3], dtype=np.float32))
        edges.append((a, b))

    if not pts:
        return np.zeros((0, 2), dtype=np.float32), []

    X = np.stack(pts, axis=0).astype(np.float32)
    X = X - X.mean(axis=0, keepdims=True)

    scale = float(np.sqrt((X ** 2).sum(axis=1).max()))
    if scale < 1e-6:
        scale = 1.0
    X = X / scale

    # PCA 对齐，减少旋转差异。
    try:
        cov = X.T @ X
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, int(np.argmax(vals))]
        theta = math.atan2(float(axis[1]), float(axis[0]))
        c, s = math.cos(-theta), math.sin(-theta)
        R = np.asarray([[c, -s], [s, c]], dtype=np.float32)
        X = X @ R.T

        # 规范符号方向。
        if np.sum(X[:, 0]) < -1e-6:
            X[:, 0] *= -1.0
        if np.sum(X[:, 1]) < -1e-6:
            X[:, 1] *= -1.0
    except Exception:
        pass

    return X.astype(np.float32), edges


def _geometry_hash(strokes: List[Dict[str, Any]], bins: int = 24) -> str:
    """
    粗几何 hash:
      - 平移不敏感
      - 尺度不敏感
      - 大体旋转不敏感
      - 对轻微 jitter 不太敏感
    """
    X, edges = _normalized_endpoint_cloud(strokes)
    if X.shape[0] == 0:
        return sha1_text("empty", n=24)

    pair_d = []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            pair_d.append(float(np.linalg.norm(X[i] - X[j])))
    pair_bins = tuple(sorted(int(round(d * bins)) for d in pair_d))

    lens = []
    angles = []
    for a, b in edges:
        v = X[b] - X[a]
        lens.append(float(np.linalg.norm(v)))
        ang = math.atan2(float(v[1]), float(v[0])) % math.pi
        angles.append(ang)

    len_bins = tuple(sorted(int(round(x * bins)) for x in lens))
    ang_bins = tuple(sorted(int(round(a / math.pi * bins)) for a in angles))
    cloud = tuple(sorted((int(round(x * bins)), int(round(y * bins))) for x, y in X.tolist()))

    return sha1_obj({
        "pair": pair_bins,
        "len": len_bins,
        "ang": ang_bins,
        "cloud": cloud,
    }, n=24)


@dataclass
class CandidateSignatures:
    candidate_id: str
    stroke_count: int
    connected_components: int
    cycle_rank: int
    degree_hist: Tuple[int, ...]
    relation_hist: Dict[str, int]
    strict_topo_hash: str
    family_hash: str
    geometry_hash: str

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["degree_hist"] = list(self.degree_hist)
        return d

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "CandidateSignatures":
        return CandidateSignatures(
            candidate_id=str(d.get("candidate_id", "")),
            stroke_count=int(d.get("stroke_count", 0)),
            connected_components=int(d.get("connected_components", 0)),
            cycle_rank=int(d.get("cycle_rank", 0)),
            degree_hist=tuple(int(x) for x in d.get("degree_hist", [])),
            relation_hist={str(k): int(v) for k, v in dict(d.get("relation_hist", {})).items()},
            strict_topo_hash=str(d.get("strict_topo_hash", "")),
            family_hash=str(d.get("family_hash", "")),
            geometry_hash=str(d.get("geometry_hash", "")),
        )


def compute_candidate_signatures(
    candidate: Dict[str, Any],
    *,
    candidate_id: Optional[str] = None,
    endpoint_tol: float = 4.0,
    geometry_bins: int = 24,
) -> CandidateSignatures:
    strokes = extract_strokes(candidate)
    cid = candidate_id or _get_candidate_id(candidate)

    nodes, sk_edges = _skeleton_graph_from_endpoints(strokes, endpoint_tol=endpoint_tol)
    cc = _connected_components(nodes, sk_edges)
    cyc = _cycle_rank(nodes, sk_edges)
    deg_hist = _degree_hist(nodes, sk_edges)

    rel_edges = _relation_edges_from_events(candidate)
    rel_hist = Counter([r for _, _, r in rel_edges])

    skeleton_wl4 = _wl_hash(nodes, sk_edges, rounds=4)
    skeleton_wl3 = _wl_hash(nodes, sk_edges, rounds=3)

    strict_obj = {
        "stroke_count": len(strokes),
        "node_count": len(nodes),
        "edge_count": len(sk_edges),
        "cc": cc,
        "cycle_rank": cyc,
        "degree_hist": deg_hist,
        "skeleton_wl": skeleton_wl4,
        "relation_edges": rel_edges,
        "relation_hist": dict(sorted(rel_hist.items())),
    }
    family_obj = {
        "stroke_count": len(strokes),
        "cc": cc,
        "cycle_rank": cyc,
        "degree_hist": deg_hist,
        "skeleton_wl": skeleton_wl3,
        "relation_hist": dict(sorted(rel_hist.items())),
    }

    return CandidateSignatures(
        candidate_id=str(cid),
        stroke_count=len(strokes),
        connected_components=cc,
        cycle_rank=cyc,
        degree_hist=deg_hist,
        relation_hist=dict(sorted(rel_hist.items())),
        strict_topo_hash=sha1_obj(strict_obj, n=24),
        family_hash=sha1_obj(family_obj, n=20),
        geometry_hash=_geometry_hash(strokes, bins=geometry_bins),
    )


# =============================================================================
# 5. 缓存构建
# =============================================================================

def _process_json_file_worker(args: Tuple[str, str, float, int]) -> Dict[str, Any]:
    """
    多进程 worker：解析一个 JSON 文件并返回签名。
    """
    path, source_label, endpoint_tol, geometry_bins = args
    rows = []
    errors = []

    try:
        obj = load_json(path)
        for local_id, record in flatten_json_records(obj) or []:
            if not isinstance(record, dict):
                continue
            fallback = f"{_norm_path(path)}:{local_id}"
            cid = _get_candidate_id(record, fallback=fallback)
            try:
                sig = compute_candidate_signatures(
                    record,
                    candidate_id=cid,
                    endpoint_tol=endpoint_tol,
                    geometry_bins=geometry_bins,
                )
                if sig.stroke_count > 0:
                    row = sig.to_json()
                    row["_source_file"] = _norm_path(path)
                    row["_source_label"] = str(source_label)
                    row["_local_id"] = str(local_id)
                    rows.append(row)
            except Exception as e:
                errors.append({
                    "path": _norm_path(path),
                    "local_id": str(local_id),
                    "error": repr(e),
                })
    except Exception as e:
        errors.append({
            "path": _norm_path(path),
            "local_id": "__file__",
            "error": repr(e),
            "traceback": traceback.format_exc()[:2000],
        })

    return {
        "path": _norm_path(path),
        "source_label": source_label,
        "items": rows,
        "errors": errors,
    }


@dataclass
class TopologyCache:
    manifest: Dict[str, Any]
    items: List[CandidateSignatures]

    def strict_hashes(self) -> Set[str]:
        return {x.strict_topo_hash for x in self.items if x.strict_topo_hash}

    def geometry_hashes(self) -> Set[str]:
        return {x.geometry_hash for x in self.items if x.geometry_hash}

    def family_counts(self) -> Counter:
        c = Counter()
        for x in self.items:
            if x.family_hash:
                c[x.family_hash] += 1
        return c

    def summary(self) -> Dict[str, Any]:
        return {
            "item_count": len(self.items),
            "strict_hash_count": len(self.strict_hashes()),
            "geometry_hash_count": len(self.geometry_hashes()),
            "family_hash_count": len(self.family_counts()),
            "source_manifest_hash": self.manifest.get("source_manifest", {}).get("manifest_hash"),
        }

    def save(self, cache_dir: str, *, max_json_mb: int = DEFAULT_MAX_JSON_MB, prefix: str = "topology_positive_cache") -> None:
        ensure_dir(cache_dir)
        max_bytes = int(max_json_mb * 1024 * 1024)

        # 删除旧 shard，避免过期数据残留。
        for fn in os.listdir(cache_dir):
            if fn.startswith(prefix + "_shard_") and fn.endswith(".json"):
                try:
                    os.remove(os.path.join(cache_dir, fn))
                except Exception:
                    pass

        shards = []
        cur = []
        cur_size = 0

        for item in self.items:
            row = item.to_json()
            row_size = json_size_bytes(row) + 4
            if cur and cur_size + row_size > max_bytes:
                shards.append(cur)
                cur = []
                cur_size = 0
            cur.append(row)
            cur_size += row_size

        if cur:
            shards.append(cur)

        shard_files = []
        for i, rows in enumerate(shards):
            fn = f"{prefix}_shard_{i:04d}.json"
            path = os.path.join(cache_dir, fn)
            atomic_save_json({
                "cache_version": CACHE_VERSION,
                "shard_index": i,
                "item_count": len(rows),
                "items": rows,
            }, path, indent=None)
            shard_files.append({
                "file": fn,
                "item_count": len(rows),
                "size": os.path.getsize(path),
            })

        meta = copy.deepcopy(self.manifest)
        meta.update({
            "cache_version": CACHE_VERSION,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "max_json_mb": int(max_json_mb),
            "item_count": len(self.items),
            "shards": shard_files,
            "summary": self.summary(),
        })
        atomic_save_json(meta, os.path.join(cache_dir, f"{prefix}_manifest.json"), indent=2)

    @staticmethod
    def load(cache_dir: str, *, prefix: str = "topology_positive_cache") -> "TopologyCache":
        manifest_path = os.path.join(cache_dir, f"{prefix}_manifest.json")
        manifest = load_json(manifest_path)
        items = []
        for sh in manifest.get("shards", []):
            p = os.path.join(cache_dir, sh["file"])
            obj = load_json(p)
            for row in obj.get("items", []):
                if isinstance(row, dict):
                    items.append(CandidateSignatures.from_json(row))
        return TopologyCache(manifest=manifest, items=items)


def _cache_matches_sources(cache_dir: str, source_roots: Iterable[str], *, prefix: str = "topology_positive_cache") -> bool:
    manifest_path = os.path.join(cache_dir, f"{prefix}_manifest.json")
    if not os.path.exists(manifest_path):
        return False
    try:
        old = load_json(manifest_path)
        old_hash = old.get("source_manifest", {}).get("manifest_hash")
        new_hash = build_source_manifest(source_roots).get("manifest_hash")
        return bool(old_hash and old_hash == new_hash)
    except Exception:
        return False


def build_positive_topology_cache(
    source_roots: Iterable[str],
    *,
    endpoint_tol: float = 4.0,
    geometry_bins: int = 24,
    workers: Any = "auto",
    verbose: bool = True,
) -> TopologyCache:
    """
    从 good / cleaned / annotations_topo 构建正向拓扑历史缓存。
    """
    source_roots = [os.path.abspath(r) for r in source_roots if r and os.path.exists(r)]
    src_manifest = build_source_manifest(source_roots)

    jobs = []
    for root in source_roots:
        label = os.path.basename(os.path.abspath(root).rstrip(os.sep)) or root
        for path in iter_json_files(root) or []:
            jobs.append((path, label, float(endpoint_tol), int(geometry_bins)))

    n_workers = resolve_workers(workers)
    if verbose:
        print(f"[TopologyCache] source_roots={len(source_roots)} json_files={len(jobs)} workers={n_workers}")

    t0 = time.time()
    rows = []
    errors = []

    if n_workers <= 1 or len(jobs) <= 1:
        for i, job in enumerate(jobs, 1):
            res = _process_json_file_worker(job)
            rows.extend(res["items"])
            errors.extend(res["errors"])
            if verbose and (i % 20 == 0 or i == len(jobs)):
                print(f"[TopologyCache] processed {i}/{len(jobs)} files, items={len(rows)}, elapsed={time.time()-t0:.1f}s")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_process_json_file_worker, job) for job in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                rows.extend(res["items"])
                errors.extend(res["errors"])
                if verbose and (i % 20 == 0 or i == len(futs)):
                    print(f"[TopologyCache] processed {i}/{len(futs)} files, items={len(rows)}, elapsed={time.time()-t0:.1f}s")

    # 去重：同一个 hash 可以多次出现，但 cache items 保留所有候选有利于 family count。
    items = [CandidateSignatures.from_json(r) for r in rows]

    manifest = {
        "cache_version": CACHE_VERSION,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_manifest": src_manifest,
        "endpoint_tol": float(endpoint_tol),
        "geometry_bins": int(geometry_bins),
        "workers": int(n_workers),
        "source_policy": "positive_only: good + cleaned + annotations_topo; bad is intentionally ignored",
        "errors": errors[:200],
        "error_count": len(errors),
        "build_seconds": round(time.time() - t0, 3),
    }

    cache = TopologyCache(manifest=manifest, items=items)
    if verbose:
        print(f"[TopologyCache] build done: {cache.summary()} seconds={time.time()-t0:.2f}")
    return cache


def build_or_load_positive_topology_cache(
    source_roots: Iterable[str],
    *,
    cache_dir: str,
    max_json_mb: int = DEFAULT_MAX_JSON_MB,
    endpoint_tol: float = 4.0,
    geometry_bins: int = 24,
    workers: Any = "auto",
    prefix: str = "topology_positive_cache",
    force_rebuild: bool = False,
    verbose: bool = True,
) -> TopologyCache:
    """
    如果源文件未变化，直接加载缓存；
    如果 good/cleaned/annotations_topo 任意源文件 path/size/mtime 变化，则重建缓存。

    good -> cleaned 会导致 path/source_label 变化，因此会重建缓存。
    """
    source_roots = [os.path.abspath(r) for r in source_roots if r and os.path.exists(r)]
    ensure_dir(cache_dir)

    if (not force_rebuild) and _cache_matches_sources(cache_dir, source_roots, prefix=prefix):
        if verbose:
            print("[TopologyCache] source unchanged; loading cache:", os.path.abspath(cache_dir))
        return TopologyCache.load(cache_dir, prefix=prefix)

    if verbose:
        print("[TopologyCache] source changed or cache missing; rebuilding...")
    cache = build_positive_topology_cache(
        source_roots,
        endpoint_tol=endpoint_tol,
        geometry_bins=geometry_bins,
        workers=workers,
        verbose=verbose,
    )
    cache.save(cache_dir, max_json_mb=max_json_mb, prefix=prefix)
    if verbose:
        print("[TopologyCache] saved:", os.path.abspath(cache_dir))
    return cache


def build_or_load_default_positive_cache(
    *,
    cache_dir: Optional[str] = None,
    max_json_mb: int = DEFAULT_MAX_JSON_MB,
    endpoint_tol: float = 4.0,
    geometry_bins: int = 24,
    workers: Any = "auto",
    force_rebuild: bool = False,
    verbose: bool = True,
) -> TopologyCache:
    return build_or_load_positive_topology_cache(
        default_positive_source_roots(),
        cache_dir=cache_dir or default_cache_dir(),
        max_json_mb=max_json_mb,
        endpoint_tol=endpoint_tol,
        geometry_bins=geometry_bins,
        workers=workers,
        force_rebuild=force_rebuild,
        verbose=verbose,
    )


# =============================================================================
# 6. 在线去重过滤器
# =============================================================================

@dataclass
class NoveltyDecision:
    accept: bool
    reason: str
    signatures: CandidateSignatures
    detail: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        return {
            "accept": bool(self.accept),
            "reason": self.reason,
            "signatures": self.signatures.to_json(),
            "detail": self.detail,
        }


class TopologyNoveltyFilter:
    """
    在线去重器。

    推荐：
        novelty = TopologyNoveltyFilter(
            cache,
            reject_seen_topology=True,
            reject_seen_geometry=True,
            max_family_per_batch=2,
        )

    在 generate_batch() 开头:
        novelty.reset_batch()

    对每个 candidate:
        decision = novelty.check(cand)
        if not decision.accept:
            reject[decision.reason] += 1
            continue
        novelty.add(decision)
    """

    def __init__(
        self,
        cache: Optional[TopologyCache] = None,
        *,
        endpoint_tol: float = 4.0,
        geometry_bins: int = 24,
        reject_seen_topology: bool = True,
        reject_seen_geometry: bool = True,
        max_family_per_batch: int = 2,
        max_family_in_history: Optional[int] = None,
    ):
        self.cache = cache or TopologyCache(manifest={}, items=[])
        self.endpoint_tol = float(endpoint_tol)
        self.geometry_bins = int(geometry_bins)
        self.reject_seen_topology = bool(reject_seen_topology)
        self.reject_seen_geometry = bool(reject_seen_geometry)
        self.max_family_per_batch = int(max_family_per_batch)
        self.max_family_in_history = max_family_in_history if max_family_in_history is None else int(max_family_in_history)

        self.history_topo = self.cache.strict_hashes()
        self.history_geo = self.cache.geometry_hashes()
        self.history_family = self.cache.family_counts()

        self.batch_topo: Set[str] = set()
        self.batch_geo: Set[str] = set()
        self.batch_family: Counter = Counter()

    def reset_batch(self) -> None:
        self.batch_topo.clear()
        self.batch_geo.clear()
        self.batch_family.clear()

    def check(self, candidate: Dict[str, Any], *, candidate_id: Optional[str] = None) -> NoveltyDecision:
        sig = compute_candidate_signatures(
            candidate,
            candidate_id=candidate_id,
            endpoint_tol=self.endpoint_tol,
            geometry_bins=self.geometry_bins,
        )

        detail = {
            "history_family_count": int(self.history_family.get(sig.family_hash, 0)),
            "batch_family_count": int(self.batch_family.get(sig.family_hash, 0)),
        }

        if self.reject_seen_topology:
            if sig.strict_topo_hash in self.batch_topo:
                return NoveltyDecision(False, "duplicate_topology_in_batch", sig, detail)
            if sig.strict_topo_hash in self.history_topo:
                return NoveltyDecision(False, "duplicate_topology_in_positive_cache", sig, detail)

        if self.reject_seen_geometry:
            if sig.geometry_hash in self.batch_geo:
                return NoveltyDecision(False, "near_duplicate_geometry_in_batch", sig, detail)
            if sig.geometry_hash in self.history_geo:
                return NoveltyDecision(False, "near_duplicate_geometry_in_positive_cache", sig, detail)

        if self.max_family_per_batch is not None and self.max_family_per_batch >= 0:
            if self.batch_family.get(sig.family_hash, 0) >= self.max_family_per_batch:
                return NoveltyDecision(False, "topology_family_overflow_in_batch", sig, detail)

        if self.max_family_in_history is not None and self.max_family_in_history >= 0:
            if self.history_family.get(sig.family_hash, 0) >= self.max_family_in_history:
                return NoveltyDecision(False, "topology_family_overflow_in_positive_cache", sig, detail)

        return NoveltyDecision(True, "novel", sig, detail)

    def add(self, decision_or_candidate: Any) -> CandidateSignatures:
        if isinstance(decision_or_candidate, NoveltyDecision):
            sig = decision_or_candidate.signatures
        elif isinstance(decision_or_candidate, CandidateSignatures):
            sig = decision_or_candidate
        elif isinstance(decision_or_candidate, dict):
            sig = compute_candidate_signatures(
                decision_or_candidate,
                endpoint_tol=self.endpoint_tol,
                geometry_bins=self.geometry_bins,
            )
        else:
            raise TypeError("add() expects NoveltyDecision, CandidateSignatures, or candidate dict")

        self.batch_topo.add(sig.strict_topo_hash)
        self.batch_geo.add(sig.geometry_hash)
        self.batch_family[sig.family_hash] += 1
        return sig

    def batch_summary(self) -> Dict[str, Any]:
        return {
            "batch_topology_count": len(self.batch_topo),
            "batch_geometry_count": len(self.batch_geo),
            "batch_family_count": len(self.batch_family),
            "history_topology_count": len(self.history_topo),
            "history_geometry_count": len(self.history_geo),
            "history_family_count": len(self.history_family),
        }


# =============================================================================
# 7. 速度评估
# =============================================================================

def benchmark_signature_speed(
    source_roots: Optional[List[str]] = None,
    *,
    sample_limit: int = 1000,
    endpoint_tol: float = 4.0,
    geometry_bins: int = 24,
) -> Dict[str, Any]:
    """
    单进程 signature 速度评估。
    实际缓存构建多进程时，吞吐一般近似按 CPU 核数提升，但 JSON 读取会受硬盘影响。
    """
    roots = source_roots or default_positive_source_roots()
    records = []
    for root in roots:
        for path in iter_json_files(root) or []:
            try:
                obj = load_json(path)
                for local_id, rec in flatten_json_records(obj) or []:
                    if isinstance(rec, dict):
                        records.append((path, local_id, rec))
                        if len(records) >= sample_limit:
                            break
            except Exception:
                pass
            if len(records) >= sample_limit:
                break
        if len(records) >= sample_limit:
            break

    t0 = time.time()
    ok = 0
    for path, local_id, rec in records:
        try:
            sig = compute_candidate_signatures(
                rec,
                candidate_id=f"{path}:{local_id}",
                endpoint_tol=endpoint_tol,
                geometry_bins=geometry_bins,
            )
            if sig.stroke_count > 0:
                ok += 1
        except Exception:
            pass

    dt = max(1e-9, time.time() - t0)
    return {
        "sample_count": len(records),
        "ok_count": ok,
        "seconds": round(dt, 4),
        "signatures_per_second_single_process": round(ok / dt, 2) if dt > 0 else None,
        "note": "GPU is not recommended: JSON parsing + tiny irregular graph hashing is CPU-bound; multiprocessing is the right acceleration.",
    }


# =============================================================================
# 8. CLI
# =============================================================================

def main():
    d = _infer_dirs()

    parser = argparse.ArgumentParser(description="Build positive topology novelty cache.")
    parser.add_argument("--good-dir", default=d["good_dir"])
    parser.add_argument("--cleaned-dir", default=d["cleaned_dir"])
    parser.add_argument("--annotations-dir", default=d["annotations_topo_dir"])
    parser.add_argument("--cache-dir", default=d["cache_dir"])
    parser.add_argument("--max-mb", type=int, default=DEFAULT_MAX_JSON_MB)
    parser.add_argument("--endpoint-tol", type=float, default=4.0)
    parser.add_argument("--geometry-bins", type=int, default=24)
    parser.add_argument("--workers", default="auto", help="'auto' or integer")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    roots = [args.good_dir, args.cleaned_dir, args.annotations_dir]

    print("=" * 88)
    print("[TopologyPositiveCache] start")
    print("[Dirs]")
    for k, v in d.items():
        print(f"  {k}: {v}")
    print("[Sources]")
    for r in roots:
        print(f"  {r} exists={os.path.exists(r)}")
    print(f"[CacheDir] {args.cache_dir}")
    print(f"[Workers] {args.workers}")
    print("=" * 88)

    if args.benchmark:
        bench = benchmark_signature_speed(
            roots,
            sample_limit=1000,
            endpoint_tol=args.endpoint_tol,
            geometry_bins=args.geometry_bins,
        )
        print("[Benchmark]", json.dumps(bench, ensure_ascii=False, indent=2))

    cache = build_or_load_positive_topology_cache(
        roots,
        cache_dir=args.cache_dir,
        max_json_mb=args.max_mb,
        endpoint_tol=args.endpoint_tol,
        geometry_bins=args.geometry_bins,
        workers=args.workers,
        force_rebuild=args.force,
        verbose=True,
    )

    print("[Summary]", json.dumps(cache.summary(), ensure_ascii=False, indent=2))
    print("[Done]")


if __name__ == "__main__":
    main()
