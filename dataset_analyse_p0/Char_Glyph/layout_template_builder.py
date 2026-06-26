import os
import json
import math
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# ⚙️ 全局配置
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CORPUS_FILE = os.path.join(SCRIPT_DIR, "alien_glyph_pcg_corpus.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "glyph_layout_templates.json")

PREVIEW_DIR = os.path.join(SCRIPT_DIR, "layout_template_previews")

# 如果 corpus 中 px 坐标没有显式 canvas size，就用这个兜底
DEFAULT_CANVAS_SIZE = 256.0

# 是否保留 derived samples
# True: 包含 rotate / mirror / order 派生样本，模板更多
# False: 只保留原始样本，模板更干净但数量少
INCLUDE_DERIVED_TEMPLATES = True

# 调试时可以设成 200，正式 None
MAX_TEMPLATES = None

# 是否要求 template 至少有一条拓扑边
REQUIRE_EDGES = True

# 是否要求至少两个节点
MIN_NODE_COUNT = 2

# 预览图数量
PREVIEW_COUNT = 36

# 是否保存 source_mother_bezier_norm，后续可以用于预览 / oracle validation
KEEP_SOURCE_BEZIER = True

# 是否打印每个 source 的统计
PRINT_SOURCE_DETAIL = True


# =========================================================
# 🧮 基础工具
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def round_list(x, ndigits=6):
    return np.round(np.asarray(x, dtype=float), ndigits).tolist()


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


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def radians_to_degrees(rad):
    return float(rad * 180.0 / math.pi)


def normalize_angle_pi(rad):
    a = float(rad)
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


def make_template_id(idx):
    return f"layout_template_{idx:05d}"


def make_glyph_uid(source_file, hex_key):
    return f"{source_file}::{hex_key}"


def edge_type_name(edge):
    if "j_type" in edge:
        return str(edge["j_type"])
    idx = safe_int(edge.get("j_type_idx", 0), 0)
    return {
        1: "E2E",
        2: "X",
        3: "T",
    }.get(idx, "UNKNOWN")


def get_candidate_edges(sample):
    """
    标准位置：
        sample["topology"]["positive_edges_undirected"]
    """
    topo = sample.get("topology", {})

    if isinstance(topo, dict):
        if "positive_edges_undirected" in topo and isinstance(topo["positive_edges_undirected"], list):
            return topo["positive_edges_undirected"]

        if "edges" in topo and isinstance(topo["edges"], list):
            return topo["edges"]

        if "positive_edges_directed" in topo and isinstance(topo["positive_edges_directed"], list):
            directed = topo["positive_edges_directed"]
            return [
                e for e in directed
                if e.get("direction", "forward") == "forward"
            ]

    if "edges" in sample and isinstance(sample["edges"], list):
        return sample["edges"]

    return []


def cubic_bezier_points(ctrl, n=48):
    ctrl = np.asarray(ctrl, dtype=float)

    if ctrl.shape != (4, 2):
        return np.zeros((0, 2), dtype=float)

    t = np.linspace(0.0, 1.0, n)[:, None]

    p0, p1, p2, p3 = ctrl

    pts = (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t ** 2 * p2
        + t ** 3 * p3
    )

    return pts


def bbox_from_points(points):
    pts = np.asarray(points, dtype=float)

    if len(pts) == 0:
        return [0.0, 0.0, 0.0, 0.0]

    return [
        float(np.min(pts[:, 0])),
        float(np.min(pts[:, 1])),
        float(np.max(pts[:, 0])),
        float(np.max(pts[:, 1])),
    ]


def bbox_union(boxes):
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]

    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def infer_canvas_size_from_corpus(corpus):
    """
    优先从 config 读；没有就兜底 DEFAULT_CANVAS_SIZE。
    """
    cfg = corpus.get("config", {})

    for key in ["canvas_size", "CANVAS_SIZE"]:
        if key in cfg:
            try:
                return float(cfg[key])
            except Exception:
                pass

    # 有些 config 可能嵌套
    for k, v in cfg.items():
        if isinstance(v, dict):
            for key in ["canvas_size", "CANVAS_SIZE"]:
                if key in v:
                    try:
                        return float(v[key])
                    except Exception:
                        pass

    return float(DEFAULT_CANVAS_SIZE)


def normalize_points_maybe_px(points, canvas_size):
    """
    如果点已经是 0~1 归一化坐标，就原样返回；
    如果像素坐标明显大于 1，则除以 canvas_size。
    """
    arr = np.asarray(points, dtype=float)

    if arr.size == 0:
        return arr

    max_abs = float(np.max(np.abs(arr)))

    if max_abs <= 1.5:
        return arr

    return arr / float(canvas_size)


def get_norm_point_from_node(node, key_norm, key_px, canvas_size):
    """
    优先读取 center_norm / p0_norm / p3_norm。
    如果没有，则从 px 字段除以 canvas_size。
    """
    if key_norm in node and node[key_norm] is not None:
        arr = np.asarray(node[key_norm], dtype=float)
        if arr.shape[0] >= 2:
            return arr[:2]

    if key_px in node and node[key_px] is not None:
        arr = np.asarray(node[key_px], dtype=float)
        if arr.shape[0] >= 2:
            return arr[:2] / float(canvas_size)

    return None


def get_mother_bezier_norm(node, canvas_size):
    """
    返回 4x2 normalized Bézier。
    """
    for key in ["mother_bezier_norm", "source_mother_bezier_norm"]:
        if key in node and node[key] is not None:
            arr = np.asarray(node[key], dtype=float)
            if arr.shape == (4, 2):
                return normalize_points_maybe_px(arr, canvas_size)

    for key in ["mother_bezier_px", "mother_bezier", "source_mother_bezier_px"]:
        if key in node and node[key] is not None:
            arr = np.asarray(node[key], dtype=float)
            if arr.shape == (4, 2):
                return normalize_points_maybe_px(arr, canvas_size)

    return None


def estimate_rotation_from_points(p0, p3):
    if p0 is None or p3 is None:
        return 0.0

    v = np.asarray(p3, dtype=float) - np.asarray(p0, dtype=float)

    if np.linalg.norm(v) < 1e-8:
        return 0.0

    return float(math.atan2(v[1], v[0]))


def estimate_chord_length_norm(p0, p3):
    if p0 is None or p3 is None:
        return 0.0

    return float(np.linalg.norm(np.asarray(p3, dtype=float) - np.asarray(p0, dtype=float)))


def estimate_arc_length_norm_from_bezier(bezier_norm):
    if bezier_norm is None:
        return 0.0

    pts = cubic_bezier_points(bezier_norm, n=64)

    if len(pts) < 2:
        return 0.0

    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def is_derived_sample(sample):
    """
    尽量兼容 derivation 格式。
    """
    d = sample.get("derivation", {})

    if not d:
        return False

    if isinstance(d, str):
        return d not in ["", "original", "orig", "none"]

    if isinstance(d, dict):
        rule = str(d.get("rule_name", d.get("rule", d.get("name", "")))).lower()
        idx = d.get("derived_idx", d.get("idx", None))

        if rule in ["", "original", "orig", "none"]:
            return False

        if idx is not None:
            try:
                return int(idx) != 0 or rule not in ["", "original", "orig", "none"]
            except Exception:
                return True

        return True

    return False


# =========================================================
# 🕸️ 拓扑摘要
# =========================================================
def connected_components(num_nodes, edges):
    parent = list(range(num_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for e in edges:
        u = safe_int(e.get("u", -1), -1)
        v = safe_int(e.get("v", -1), -1)
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            union(u, v)

    groups = defaultdict(list)
    for i in range(num_nodes):
        groups[find(i)].append(i)

    return list(groups.values())


def summarize_topology(num_nodes, edges):
    degree = [0 for _ in range(num_nodes)]
    edge_type_counts = Counter()

    for e in edges:
        u = safe_int(e.get("u", -1), -1)
        v = safe_int(e.get("v", -1), -1)
        jt = edge_type_name(e)
        edge_type_counts[jt] += 1

        if 0 <= u < num_nodes:
            degree[u] += 1
        if 0 <= v < num_nodes:
            degree[v] += 1

    comps = connected_components(num_nodes, edges)
    num_components = len(comps)

    # cycle_rank = E - V + C
    cycle_rank = max(0, len(edges) - num_nodes + num_components)

    return {
        "num_nodes": int(num_nodes),
        "num_edges": int(len(edges)),
        "edge_type_counts": {str(k): int(v) for k, v in edge_type_counts.items()},
        "degrees": [int(x) for x in degree],
        "max_degree": int(max(degree) if degree else 0),
        "num_components": int(num_components),
        "component_sizes": [int(len(c)) for c in comps],
        "cycle_rank": int(cycle_rank),
        "has_cycle": bool(cycle_rank > 0),
    }


def topology_signature(num_nodes, edges):
    summary = summarize_topology(num_nodes, edges)

    deg_sig = ",".join(map(str, sorted(summary["degrees"])))

    edge_types = summary["edge_type_counts"]
    type_sig = ",".join(
        f"{k}:{edge_types[k]}"
        for k in sorted(edge_types.keys())
    )

    return (
        f"N={num_nodes}|"
        f"E={len(edges)}|"
        f"deg={deg_sig}|"
        f"type={type_sig}|"
        f"cycle={summary['cycle_rank']}"
    )


def motif_flags(summary):
    edge_counts = summary.get("edge_type_counts", {})

    return {
        "connected": bool(summary.get("num_components", 999) == 1),
        "has_E2E": bool(edge_counts.get("E2E", 0) > 0),
        "has_T": bool(edge_counts.get("T", 0) > 0),
        "has_X": bool(edge_counts.get("X", 0) > 0),
        "cycle": bool(summary.get("cycle_rank", 0) > 0),
        "branch_or_hub": bool(summary.get("max_degree", 0) >= 3),
    }


# =========================================================
# 🧱 构建单个 layout template
# =========================================================
def extract_layout_node(node, canvas_size):
    node_id = safe_int(node.get("node_id", node.get("bezier_id", -1)), -1)

    p0_norm = get_norm_point_from_node(node, "p0_norm", "p0_px", canvas_size)
    p3_norm = get_norm_point_from_node(node, "p3_norm", "p3_px", canvas_size)
    center_norm = get_norm_point_from_node(node, "center_norm", "center_px", canvas_size)

    mother_bezier_norm = get_mother_bezier_norm(node, canvas_size)

    if center_norm is None:
        if p0_norm is not None and p3_norm is not None:
            center_norm = 0.5 * (p0_norm + p3_norm)
        elif mother_bezier_norm is not None:
            pts = cubic_bezier_points(mother_bezier_norm, n=32)
            center_norm = np.mean(pts, axis=0)
        else:
            center_norm = None

    if p0_norm is None and mother_bezier_norm is not None:
        p0_norm = mother_bezier_norm[0]

    if p3_norm is None and mother_bezier_norm is not None:
        p3_norm = mother_bezier_norm[-1]

    solver_init = node.get("solver_init", {})

    rotation_rad = None

    if isinstance(solver_init, dict):
        if "rotation_rad" in solver_init:
            rotation_rad = safe_float(solver_init.get("rotation_rad"), None)
        elif "angle_rad" in solver_init:
            rotation_rad = safe_float(solver_init.get("angle_rad"), None)

    if rotation_rad is None:
        rotation_rad = estimate_rotation_from_points(p0_norm, p3_norm)

    rotation_rad = normalize_angle_pi(rotation_rad)

    chord_length_norm = None
    arc_length_norm = None

    if isinstance(solver_init, dict):
        if "chord_length_norm" in solver_init:
            chord_length_norm = safe_float(solver_init.get("chord_length_norm"), None)
        if "arc_length_norm" in solver_init:
            arc_length_norm = safe_float(solver_init.get("arc_length_norm"), None)

    if chord_length_norm is None:
        chord_length_norm = estimate_chord_length_norm(p0_norm, p3_norm)

    if arc_length_norm is None:
        arc_length_norm = estimate_arc_length_norm_from_bezier(mother_bezier_norm)

    # scale_norm 用 arc length 更贴近 solver 中 length_prior_norm
    if arc_length_norm is not None and arc_length_norm > 1e-6:
        scale_norm = float(arc_length_norm)
    else:
        scale_norm = float(chord_length_norm)

    if center_norm is None:
        return None, "missing_center"

    layout_prior = {
        "center_norm": round_list(center_norm, 6),
        "rotation_rad": round(float(rotation_rad), 8),
        "rotation_deg": round(radians_to_degrees(rotation_rad), 6),
        "scale_norm": round(float(scale_norm), 6),
        "length_norm": round(float(scale_norm), 6),
        "chord_length_norm": round(float(chord_length_norm), 6),
        "arc_length_norm": round(float(arc_length_norm), 6),
    }

    if p0_norm is not None:
        layout_prior["p0_norm"] = round_list(p0_norm, 6)

    if p3_norm is not None:
        layout_prior["p3_norm"] = round_list(p3_norm, 6)

    out = {
        "node_id": int(node_id),
        "source_node_id": int(node_id),

        "bezier_id": node.get("bezier_id", None),
        "shape_code": safe_int(node.get("shape_code", node.get("shape_token", -1)), -1),
        "width_token": safe_int(node.get("width_token", 0), 0),

        "source_file": node.get("source_file", ""),
        "hex_key": node.get("hex_key", ""),
        "glyph_uid": node.get("glyph_uid", ""),
        "source_stroke_uid": node.get("source_stroke_uid", ""),

        "grammar_role": node.get("graph_role", node.get("grammar_role", {})),

        "layout_prior": layout_prior,
    }

    if KEEP_SOURCE_BEZIER and mother_bezier_norm is not None:
        out["source_mother_bezier_norm"] = round_list(mother_bezier_norm, 6)

    return out, None


def clean_edge(edge):
    """
    保留 solver / graph grammar 需要的字段，去掉太重的无关字段。
    """
    jt = edge_type_name(edge)

    out = {
        "edge_id": edge.get("edge_id", None),
        "j_type": jt,
        "j_type_idx": safe_int(edge.get("j_type_idx", {"E2E": 1, "X": 2, "T": 3}.get(jt, 0)), 0),

        "u": safe_int(edge.get("u", -1), -1),
        "v": safe_int(edge.get("v", -1), -1),

        "role_u": edge.get("role_u", ""),
        "role_v": edge.get("role_v", ""),

        "t_u": safe_float(edge.get("t_u", 0.0), 0.0),
        "t_v": safe_float(edge.get("t_v", 0.0), 0.0),

        "t_u_bin": edge.get("t_u_bin", None),
        "t_v_bin": edge.get("t_v_bin", None),

        "angle_deg": edge.get("angle_deg", None),
        "angle_bin": edge.get("angle_bin", None),
        "angle_sin": edge.get("angle_sin", None),
        "angle_cos": edge.get("angle_cos", None),
    }

    # source-aware provenance
    for k in [
        "edge_uid",
        "source_event_idx",
        "source_file",
        "hex_key",
        "glyph_uid",
        "bid_u",
        "bid_v",
        "stroke_uid_u",
        "stroke_uid_v",
    ]:
        if k in edge:
            out[k] = edge[k]

    return out


def build_layout_bbox_from_nodes(nodes):
    boxes = []

    for n in nodes:
        if "source_mother_bezier_norm" in n:
            pts = cubic_bezier_points(n["source_mother_bezier_norm"], n=48)
            boxes.append(bbox_from_points(pts))
        else:
            lp = n.get("layout_prior", {})
            c = lp.get("center_norm", None)
            if c is not None:
                c = np.asarray(c, dtype=float)
                eps = 0.005
                boxes.append([c[0] - eps, c[1] - eps, c[0] + eps, c[1] + eps])

    return bbox_union(boxes)


def build_template_from_sample(sample, template_idx, canvas_size):
    nodes_raw = sample.get("nodes", [])
    edges_raw = get_candidate_edges(sample)

    if len(nodes_raw) < MIN_NODE_COUNT:
        return None, "too_few_nodes"

    if REQUIRE_EDGES and len(edges_raw) == 0:
        return None, "no_edges"

    layout_nodes = []
    missing_reasons = Counter()

    for node in nodes_raw:
        ln, reason = extract_layout_node(node, canvas_size)

        if ln is None:
            missing_reasons[reason] += 1
            continue

        layout_nodes.append(ln)

    if len(layout_nodes) < MIN_NODE_COUNT:
        return None, "too_few_valid_layout_nodes"

    # 保证 node_id 连续性：如果原始 node_id 不是 0..N-1，建立重映射
    old_ids = [safe_int(n["node_id"], -1) for n in layout_nodes]
    old_to_new = {old_id: i for i, old_id in enumerate(old_ids)}

    for i, n in enumerate(layout_nodes):
        n["source_node_id"] = int(n["node_id"])
        n["node_id"] = int(i)

    clean_edges = []

    for e in edges_raw:
        ce = clean_edge(e)

        old_u = safe_int(ce.get("u", -1), -1)
        old_v = safe_int(ce.get("v", -1), -1)

        if old_u not in old_to_new or old_v not in old_to_new:
            continue

        ce["source_u"] = int(old_u)
        ce["source_v"] = int(old_v)
        ce["u"] = int(old_to_new[old_u])
        ce["v"] = int(old_to_new[old_v])

        clean_edges.append(ce)

    if REQUIRE_EDGES and len(clean_edges) == 0:
        return None, "no_valid_edges_after_remap"

    n = len(layout_nodes)
    summary = summarize_topology(n, clean_edges)
    sig = topology_signature(n, clean_edges)
    motifs = motif_flags(summary)

    layout_bbox = build_layout_bbox_from_nodes(layout_nodes)
    x0, y0, x1, y1 = layout_bbox
    bw = max(0.0, x1 - x0)
    bh = max(0.0, y1 - y0)

    source_file = sample.get("source_file", "")
    hex_key = sample.get("hex_key", "")
    glyph_uid = sample.get("glyph_uid", "")

    if not glyph_uid and source_file and hex_key:
        glyph_uid = make_glyph_uid(source_file, hex_key)

    template_id = make_template_id(template_idx)

    template = {
        "layout_template_id": template_id,

        "source_sample_id": sample.get("sample_id", ""),
        "source_file": source_file,
        "hex_key": hex_key,
        "glyph_uid": glyph_uid,
        "char": sample.get("char", ""),

        "is_derived": bool(is_derived_sample(sample)),
        "derivation": sample.get("derivation", {}),

        "topology_signature": sig,
        "motifs": motifs,

        "num_nodes": int(n),
        "num_edges": int(len(clean_edges)),

        "layout_bbox_norm": round_list(layout_bbox, 6),

        "layout_features": {
            "bbox_width_norm": round(float(bw), 6),
            "bbox_height_norm": round(float(bh), 6),
            "bbox_area_norm": round(float(bw * bh), 6),
            "aspect_ratio": round(float(bw / max(bh, 1e-8)), 6),
            "center_norm": round_list([0.5 * (x0 + x1), 0.5 * (y0 + y1)], 6),
        },

        "nodes": layout_nodes,

        "topology": {
            **summary,
            "positive_edges_undirected": clean_edges,
        },

        # 原始 sample 的部分 summary，方便追溯
        "source_geometry": sample.get("geometry", {}),
        "source_aesthetic_features": sample.get("aesthetic_features", {}),
    }

    return template, None


# =========================================================
# 📊 统计与诊断
# =========================================================
def build_template_stats(templates, skip_counter):
    source_hist = Counter()
    glyph_uid_hist = Counter()
    node_hist = Counter()
    edge_hist = Counter()
    sig_hist = Counter()
    motif_hist = Counter()
    edge_type_hist = Counter()
    shape_hist = Counter()
    width_hist = Counter()
    derived_hist = Counter()

    for t in templates:
        source_hist[t.get("source_file", "UNKNOWN")] += 1
        glyph_uid_hist[t.get("glyph_uid", "UNKNOWN")] += 1
        node_hist[str(t["num_nodes"])] += 1
        edge_hist[str(t["num_edges"])] += 1
        sig_hist[t["topology_signature"]] += 1
        derived_hist[str(t.get("is_derived", False))] += 1

        for k, v in t.get("motifs", {}).items():
            if v:
                motif_hist[k] += 1

        for k, v in t.get("topology", {}).get("edge_type_counts", {}).items():
            edge_type_hist[k] += int(v)

        for n in t.get("nodes", []):
            shape_hist[str(n.get("shape_code", -1))] += 1
            width_hist[str(n.get("width_token", 0))] += 1

    stats = {
        "template_count": len(templates),
        "unique_source_file_count": len(source_hist),
        "unique_glyph_uid_count": len(glyph_uid_hist),
        "unique_topology_signature_count": len(sig_hist),

        "skip_counter": {str(k): int(v) for k, v in skip_counter.items()},

        "source_file_hist": {str(k): int(v) for k, v in source_hist.items()},
        "node_count_hist": {str(k): int(v) for k, v in node_hist.items()},
        "edge_count_hist": {str(k): int(v) for k, v in edge_hist.items()},
        "topology_signature_hist_top20": {
            str(k): int(v)
            for k, v in sig_hist.most_common(20)
        },

        "motif_hist": {str(k): int(v) for k, v in motif_hist.items()},
        "edge_type_hist": {str(k): int(v) for k, v in edge_type_hist.items()},
        "shape_code_hist": {str(k): int(v) for k, v in shape_hist.items()},
        "width_token_hist": {str(k): int(v) for k, v in width_hist.items()},
        "is_derived_hist": {str(k): int(v) for k, v in derived_hist.items()},
    }

    return stats


def print_stats(stats):
    print("\n" + "=" * 80)
    print("📦 Layout Template Builder Summary")
    print("=" * 80)

    print(f"  template_count: {stats['template_count']}")
    print(f"  unique_source_file_count: {stats['unique_source_file_count']}")
    print(f"  unique_glyph_uid_count: {stats['unique_glyph_uid_count']}")
    print(f"  unique_topology_signature_count: {stats['unique_topology_signature_count']}")

    print("\n[Skip Counter]")
    for k, v in stats["skip_counter"].items():
        print(f"  {k}: {v}")

    print("\n[Node Count Hist]")
    for k, v in sorted(stats["node_count_hist"].items(), key=lambda x: int(x[0])):
        print(f"  {k}: {v}")

    print("\n[Edge Count Hist]")
    for k, v in sorted(stats["edge_count_hist"].items(), key=lambda x: int(x[0])):
        print(f"  {k}: {v}")

    print("\n[Edge Type Hist]")
    for k, v in stats["edge_type_hist"].items():
        print(f"  {k}: {v}")

    print("\n[Motif Hist]")
    for k, v in stats["motif_hist"].items():
        print(f"  {k}: {v}")

    print("\n[Derived Hist]")
    for k, v in stats["is_derived_hist"].items():
        print(f"  {k}: {v}")

    if PRINT_SOURCE_DETAIL:
        print("\n[Source File Hist]")
        for k, v in stats["source_file_hist"].items():
            print(f"  {k}: {v}")

    print("\n[Topology Signature Top20]")
    for k, v in stats["topology_signature_hist_top20"].items():
        print(f"  {v:>5d} | {k}")

    print("=" * 80 + "\n")


# =========================================================
# 🎨 预览图
# =========================================================
def render_template_preview(template, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax_src, ax_layout, ax_topo = axes

    colors = [
        "#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA",
        "#00897B", "#6D4C41", "#3949AB", "#D81B60", "#7CB342",
        "#546E7A", "#F4511E"
    ]

    node_pos = {}

    # =====================================================
    # 1. Source Bézier View
    # =====================================================
    for i, node in enumerate(template["nodes"]):
        c = colors[i % len(colors)]

        lp = node["layout_prior"]
        center = np.asarray(lp["center_norm"], dtype=float)
        node_pos[node["node_id"]] = center

        if "source_mother_bezier_norm" in node:
            pts = cubic_bezier_points(node["source_mother_bezier_norm"], n=64)

            ax_src.plot(
                pts[:, 0],
                pts[:, 1],
                color=c,
                lw=3.0,
                alpha=0.95,
            )

            ax_src.scatter([pts[0, 0]], [pts[0, 1]], color="green", s=35, zorder=5)
            ax_src.scatter([pts[-1, 0]], [pts[-1, 1]], color="red", s=35, zorder=5)

        else:
            p0 = np.asarray(lp.get("p0_norm", center), dtype=float)
            p3 = np.asarray(lp.get("p3_norm", center), dtype=float)

            ax_src.plot(
                [p0[0], p3[0]],
                [p0[1], p3[1]],
                color=c,
                lw=3.0,
                alpha=0.95,
            )

        ax_src.scatter([center[0]], [center[1]], color="black", s=18, zorder=8)
        ax_src.text(
            center[0] + 0.008,
            center[1] - 0.008,
            f"N{node['node_id']}",
            fontsize=9,
            color="black",
            fontweight="bold",
        )

    ax_src.set_title("Source Bézier / Real Glyph", fontsize=10)

    # =====================================================
    # 2. Layout Prior View
    # =====================================================
    for i, node in enumerate(template["nodes"]):
        c = colors[i % len(colors)]
        lp = node["layout_prior"]

        center = np.asarray(lp["center_norm"], dtype=float)
        theta = float(lp.get("rotation_rad", 0.0))
        length = float(lp.get("length_norm", lp.get("scale_norm", 0.2)))

        # 画 layout 主轴，不画真实曲线
        dx = math.cos(theta) * length * 0.5
        dy = math.sin(theta) * length * 0.5

        p0 = center - np.array([dx, dy])
        p1 = center + np.array([dx, dy])

        ax_layout.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=c,
            lw=4.0,
            alpha=0.95,
        )

        ax_layout.scatter([center[0]], [center[1]], color=c, s=45, zorder=7)
        ax_layout.text(
            center[0] + 0.008,
            center[1] - 0.008,
            f"N{node['node_id']}",
            fontsize=9,
            color=c,
            fontweight="bold",
        )

        ax_layout.text(
            center[0] + 0.008,
            center[1] + 0.018,
            f"s={length:.2f}\nθ={math.degrees(theta):.0f}°",
            fontsize=7,
            color=c,
        )

    ax_layout.set_title("Layout Prior: center / rotation / scale", fontsize=10)

    # =====================================================
    # 3. Topology Graph View
    # =====================================================
    for i, node in enumerate(template["nodes"]):
        c = colors[i % len(colors)]
        lp = node["layout_prior"]
        center = np.asarray(lp["center_norm"], dtype=float)

        ax_topo.scatter(
            [center[0]],
            [center[1]],
            color=c,
            s=90,
            zorder=8,
        )

        ax_topo.text(
            center[0] + 0.01,
            center[1] - 0.01,
            f"N{node['node_id']}",
            fontsize=10,
            color=c,
            fontweight="bold",
        )

    edge_style = {
        "E2E": {"color": "#111111", "lw": 2.0, "ls": "-"},
        "T": {"color": "#D81B60", "lw": 2.4, "ls": "--"},
        "X": {"color": "#1E88E5", "lw": 2.4, "ls": ":"},
    }

    for e in template["topology"]["positive_edges_undirected"]:
        u = e["u"]
        v = e["v"]

        if u not in node_pos or v not in node_pos:
            continue

        pu = node_pos[u]
        pv = node_pos[v]
        pm = 0.5 * (pu + pv)

        jt = e.get("j_type", "?")
        st = edge_style.get(jt, {"color": "#555555", "lw": 1.5, "ls": "--"})

        ax_topo.plot(
            [pu[0], pv[0]],
            [pu[1], pv[1]],
            color=st["color"],
            lw=st["lw"],
            ls=st["ls"],
            alpha=0.85,
        )

        ax_topo.text(
            pm[0],
            pm[1],
            jt,
            fontsize=9,
            color=st["color"],
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )

    ax_topo.set_title("Topology Graph: E2E / T / X", fontsize=10)

    # =====================================================
    # Shared formatting
    # =====================================================
    for ax in axes:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1.0, 0.0)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)

    title = (
        f"{template['layout_template_id']} | "
        f"N={template['num_nodes']} E={template['num_edges']} | "
        f"{template.get('source_file', '')} | {template.get('hex_key', '')}"
    )

    fig.suptitle(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)



def render_previews(templates):
    ensure_dir(PREVIEW_DIR)

    print("\n" + "=" * 80)
    print("🖼️ Rendering Layout Template Previews")
    print("=" * 80)

    # 优先渲染不同 topology signature 的模板
    seen_sig = set()
    selected = []

    for t in templates:
        sig = t["topology_signature"]
        if sig not in seen_sig:
            selected.append(t)
            seen_sig.add(sig)

        if len(selected) >= PREVIEW_COUNT:
            break

    # 如果不同 signature 不够，再补充
    if len(selected) < PREVIEW_COUNT:
        for t in templates:
            if t not in selected:
                selected.append(t)
            if len(selected) >= PREVIEW_COUNT:
                break

    for i, t in enumerate(selected):
        out_path = os.path.join(PREVIEW_DIR, f"layout_template_{i:03d}_{t['layout_template_id']}.png")
        render_template_preview(t, out_path)
        print(f"  saved: {out_path}")

    print(f"\n  preview_count: {len(selected)}")
    print("=" * 80 + "\n")


# =========================================================
# 🚀 主程序
# =========================================================
def main():
    print("\n" + "=" * 80)
    print("🚀 Layout Template Builder")
    print("=" * 80)
    print(f"  corpus_file: {CORPUS_FILE}")
    print(f"  output_file: {OUTPUT_FILE}")
    print(f"  include_derived_templates: {INCLUDE_DERIVED_TEMPLATES}")
    print(f"  max_templates: {MAX_TEMPLATES}")
    print(f"  require_edges: {REQUIRE_EDGES}")
    print("=" * 80)

    corpus = load_json(CORPUS_FILE)

    canvas_size = infer_canvas_size_from_corpus(corpus)

    print(f"\n[Config]")
    print(f"  inferred canvas_size: {canvas_size}")

    glyph_samples = corpus.get("glyph_samples", [])

    print(f"\n[Input]")
    print(f"  glyph_samples: {len(glyph_samples)}")

    templates = []
    skip_counter = Counter()

    for sidx, sample in enumerate(glyph_samples):
        if not INCLUDE_DERIVED_TEMPLATES and is_derived_sample(sample):
            skip_counter["skip_derived"] += 1
            continue

        template, reason = build_template_from_sample(
            sample=sample,
            template_idx=len(templates),
            canvas_size=canvas_size,
        )

        if template is None:
            skip_counter[reason] += 1
            continue

        templates.append(template)

        if MAX_TEMPLATES is not None and len(templates) >= MAX_TEMPLATES:
            break

        if len(templates) % 500 == 0:
            print(f"  built templates: {len(templates)} / scanned samples: {sidx + 1}")

    stats = build_template_stats(templates, skip_counter)

    # signature index：后续 layout-aware sampler 可以按拓扑签名查模板
    topology_signature_index = defaultdict(list)
    glyph_uid_index = defaultdict(list)
    source_file_index = defaultdict(list)

    for t in templates:
        topology_signature_index[t["topology_signature"]].append(t["layout_template_id"])
        glyph_uid_index[t.get("glyph_uid", "UNKNOWN")].append(t["layout_template_id"])
        source_file_index[t.get("source_file", "UNKNOWN")].append(t["layout_template_id"])

    output = {
        "schema_version": "glyph_layout_templates_v1_source_aware",

        "description": (
            "Source-aware glyph layout templates extracted from alien_glyph_pcg_corpus.json. "
            "Each template preserves real glyph topology plus node-level layout priors "
            "including center_norm, rotation_rad, scale_norm, endpoints, and optional source Bézier geometry."
        ),

        "source_corpus_file": CORPUS_FILE,
        "output_file": OUTPUT_FILE,
        "preview_dir": PREVIEW_DIR,

        "config": {
            "canvas_size": canvas_size,
            "include_derived_templates": INCLUDE_DERIVED_TEMPLATES,
            "max_templates": MAX_TEMPLATES,
            "require_edges": REQUIRE_EDGES,
            "min_node_count": MIN_NODE_COUNT,
            "keep_source_bezier": KEEP_SOURCE_BEZIER,
            "id_policy": {
                "glyph_uid": "source_file::hex_key",
                "layout_template_id": "layout_template_%05d",
            },
        },

        "stats": stats,

        "topology_signature_index": {
            str(k): list(v)
            for k, v in topology_signature_index.items()
        },

        "glyph_uid_index": {
            str(k): list(v)
            for k, v in glyph_uid_index.items()
        },

        "source_file_index": {
            str(k): list(v)
            for k, v in source_file_index.items()
        },

        "layout_templates": templates,
    }

    save_json(output, OUTPUT_FILE)

    print_stats(stats)

    render_previews(templates)

    print("\n" + "=" * 80)
    print("💾 Saved")
    print("=" * 80)
    print(f"  output:  {OUTPUT_FILE}")
    print(f"  preview: {PREVIEW_DIR}")

    print("\n📌 下一步：")
    print("  1. 打开 layout_template_previews，确认这些模板确实像真实字符")
    print("  2. 如果模板 preview 正常，就写 layout_aware_graph_sampler.py")
    print("  3. layout_aware_graph_sampler.py 应该从这些模板采样 center/rotation/scale prior")
    print("  4. constraint_solver.py 后续应优先使用 node.layout_prior，而不是 spring_layout")


if __name__ == "__main__":
    main()