import os
import json
import glob
import math
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

# 导入派生引擎
from stroke_derivation import (
    generate_derived_sequences,
    normalize_and_sample_function,
    CANVAS_SIZE
)

# ==========================================
# ⚙️ 全局配置
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TOPO_DATA_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, "../AI_VECTOR_ROUTER_With_topo/annotations_topo")
)

CLUSTER_FILE = os.path.join(SCRIPT_DIR, "clustered_results.json")

OUTPUT_FILE = os.path.join(SCRIPT_DIR, "alien_glyph_pcg_corpus.json")

# 派生策略：保留你的旧逻辑
MAX_ORDER_SAMPLES = 5
ROTATION_MODE = 3
MIRROR_MODE = True

# 网格与离散化
GRID_BINS = 32
T_BINS = 32
ANGLE_BINS = 24

# Bézier 采样
BEZIER_SAMPLE_N = 48

# Edge 类型
J_TYPE_TO_IDX = {
    "NONE": 0,
    "E2E": 1,
    "X": 2,
    "T": 3,
}

IDX_TO_J_TYPE = {
    0: "NONE",
    1: "E2E",
    2: "X",
    3: "T",
}


# ==========================================
# 🆔 UID 工具：核心防冲突逻辑
# ==========================================
def make_glyph_uid(source_file, hex_key):
    """
    字符唯一 ID。
    必须包含字体/拓扑文件名，否则不同字体里的同一个 U+XXXX 会冲突。
    """
    return f"{source_file}::{hex_key}"


def make_stroke_uid(source_file, hex_key, bezier_id):
    """
    源 stroke 唯一 ID。
    """
    return f"{source_file}::{hex_key}::{bezier_id}"


def make_sample_id(glyph_uid, rule_name, derived_idx):
    """
    派生样本唯一 ID。
    """
    return f"{glyph_uid}::{rule_name}::{derived_idx}"


def bid_key(bezier_id):
    """
    防止 int / str 类型不一致导致查表失败。
    """
    return str(bezier_id)


# ==========================================
# 🧮 基础工具
# ==========================================
def to_float_list(x, ndigits=6):
    arr = np.asarray(x, dtype=float)
    return np.round(arr, ndigits).tolist()


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def get_cell_and_offset(val, max_val=CANVAS_SIZE, bins=GRID_BINS):
    """
    新版 offset 使用 [0,1) cell 内偏移。
    不再使用旧 token 方案里的 offset = grid_float - (cell + 0.5)。
    """
    norm_val = np.clip(float(val) / max_val, 0.0, 0.999999)
    grid_float = norm_val * bins
    cell = int(grid_float)
    offset = grid_float - cell
    return cell, round(float(offset), 6)


def quantize_t(t_val, bins=T_BINS):
    t_val = np.clip(float(t_val), 0.0, 1.0)
    return int(np.round(t_val * bins))


def quantize_angle_deg(angle_deg, bins=ANGLE_BINS):
    angle = float(angle_deg) % 180.0
    return int(np.floor(angle / 180.0 * bins))


def angle_to_sincos(angle_deg):
    rad = math.radians(float(angle_deg))
    return abs(math.sin(rad)), math.cos(rad)


def quantize_width(mean_w, stroke_length, num_bins=4):
    if stroke_length < 1e-5:
        return 0

    relative_w = float(mean_w) / float(stroke_length)
    norm_w = np.clip(relative_w / 0.5, 0.0, 0.999999)
    return int(norm_w * num_bins)


def cubic_bezier_point(ctrl, t):
    ctrl = np.asarray(ctrl, dtype=float)
    p0, p1, p2, p3 = ctrl

    t = float(t)
    mt = 1.0 - t

    return (
        (mt ** 3) * p0
        + 3 * (mt ** 2) * t * p1
        + 3 * mt * (t ** 2) * p2
        + (t ** 3) * p3
    )


def sample_cubic_bezier(ctrl, n=BEZIER_SAMPLE_N):
    ts = np.linspace(0.0, 1.0, n)
    return np.stack([cubic_bezier_point(ctrl, t) for t in ts], axis=0)


def calculate_stroke_length(bezier_pts, n=BEZIER_SAMPLE_N):
    pts = sample_cubic_bezier(bezier_pts, n=n)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def calculate_chord_length(bezier_pts):
    pts = np.asarray(bezier_pts, dtype=float)
    return float(np.linalg.norm(pts[3] - pts[0]))


def calculate_angle_rad(bezier_pts):
    pts = np.asarray(bezier_pts, dtype=float)
    v = pts[3] - pts[0]

    if np.linalg.norm(v) < 1e-8:
        return 0.0

    return float(math.atan2(v[1], v[0]))


def calculate_bbox(points):
    pts = np.asarray(points, dtype=float)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return mn, mx


def connected_components(num_nodes, undirected_edges):
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

    for e in undirected_edges:
        union(e["u"], e["v"])

    comps = defaultdict(list)

    for i in range(num_nodes):
        comps[find(i)].append(i)

    return list(comps.values())


def calc_cycle_rank(num_nodes, undirected_edges):
    comps = connected_components(num_nodes, undirected_edges)
    num_components = len(comps)
    num_edges = len(undirected_edges)
    return max(0, num_edges - num_nodes + num_components)


def default_angle_for_type(j_type):
    if j_type == "E2E":
        return 180.0
    if j_type == "T":
        return 90.0
    if j_type == "X":
        return 90.0
    return 0.0


def extract_event_basic(ev):
    """
    把 E2E / X / T 统一成：

    {
        j_type,
        bid_a,
        bid_b,
        t_a,
        t_b,
        role_a,
        role_b,
        angle_deg
    }

    对 T:
        a = guest
        b = host
    """
    ev_type = ev.get("type", "")

    if ev_type in ["E2E", "X"]:
        bid_a = ev.get("stroke_a")
        bid_b = ev.get("stroke_b")
        t_a = ev.get("t_a", 0.0)
        t_b = ev.get("t_b", 0.0)
        role_a = "stroke"
        role_b = "stroke"

    elif ev_type == "T":
        bid_a = ev.get("guest")
        bid_b = ev.get("host")
        t_a = ev.get("guest_t", 0.0)
        t_b = ev.get("host_t", 0.0)
        role_a = "guest"
        role_b = "host"

    else:
        return None

    if bid_a is None or bid_b is None:
        return None

    angle_deg = ev.get("angle", None)

    if angle_deg is None:
        angle_deg = default_angle_for_type(ev_type)

    return {
        "j_type": ev_type,
        "bid_a": bid_a,
        "bid_b": bid_b,
        "t_a": float(np.clip(t_a, 0.0, 1.0)),
        "t_b": float(np.clip(t_b, 0.0, 1.0)),
        "role_a": role_a,
        "role_b": role_b,
        "angle_deg": float(angle_deg),
    }


# ==========================================
# 📚 加载 Shape Codebook
# ==========================================
def load_shape_codebook(cluster_file):
    """
    读取新版 clustered_results.json。

    要求每条记录至少包含：
        source_file
        glyph_uid
        stroke_uid
        hex_key
        bezier_id
        mother_bezier
        cluster_id

    输出：
        shape_dict[glyph_uid][bezier_id] = cluster_id
    """
    if not os.path.exists(cluster_file):
        raise FileNotFoundError(f"未找到 Shape Token 词表: {cluster_file}")

    with open(cluster_file, "r", encoding="utf-8") as f:
        cluster_data = json.load(f)

    shape_dict = {}
    cluster_refs = {}
    primitive_library = {}

    missing_source = 0
    missing_glyph_uid = 0
    missing_stroke_uid = 0

    for item in cluster_data:
        cid = int(item.get("cluster_id", -1))

        if cid == -1:
            continue

        source_file = item.get("source_file", None)
        hex_key = item.get("hex_key")
        bezier_id = item.get("bezier_id")

        if source_file is None:
            source_file = "UNKNOWN_SOURCE"
            missing_source += 1

        glyph_uid = item.get("glyph_uid", None)
        if glyph_uid is None:
            glyph_uid = make_glyph_uid(source_file, hex_key)
            missing_glyph_uid += 1

        stroke_uid = item.get("stroke_uid", None)
        if stroke_uid is None:
            stroke_uid = make_stroke_uid(source_file, hex_key, bezier_id)
            missing_stroke_uid += 1

        if glyph_uid not in shape_dict:
            shape_dict[glyph_uid] = {}

        # 同时放 str 和原始 key，最大程度兼容 generate_derived_sequences
        shape_dict[glyph_uid][bid_key(bezier_id)] = cid
        shape_dict[glyph_uid][bezier_id] = cid

        if cid not in cluster_refs:
            y = normalize_and_sample_function(item["mother_bezier"], N=50)

            if y is not None:
                cluster_refs[cid] = y

                primitive_library[str(cid)] = {
                    "shape_code": cid,
                    "prototype_y_function_norm": to_float_list(y),
                    "source_refs": [
                        {
                            "source_file": source_file,
                            "glyph_uid": glyph_uid,
                            "stroke_uid": stroke_uid,
                            "hex_key": hex_key,
                            "bezier_id": bezier_id,
                        }
                    ],
                    "count_instances": 0,
                    "width_token_hist": {},
                    "degree_hist": {},
                    "role_hist": {},
                    "length_norm_values": [],
                }

        else:
            primitive_library[str(cid)]["source_refs"].append(
                {
                    "source_file": source_file,
                    "glyph_uid": glyph_uid,
                    "stroke_uid": stroke_uid,
                    "hex_key": hex_key,
                    "bezier_id": bezier_id,
                }
            )

    if missing_source > 0 or missing_glyph_uid > 0 or missing_stroke_uid > 0:
        print("⚠️ clustered_results.json 存在缺失字段：")
        print(f"   missing source_file: {missing_source}")
        print(f"   missing glyph_uid:   {missing_glyph_uid}")
        print(f"   missing stroke_uid:  {missing_stroke_uid}")
        print("   建议使用新版聚类脚本重新生成 clustered_results.json。")

    return shape_dict, cluster_refs, primitive_library


# ==========================================
# 🧩 节点构建
# ==========================================
def build_nodes_from_strokes(
    stroke_seq,
    cluster_refs,
    source_file,
    hex_key,
    glyph_uid,
    sample_id,
):
    nodes = []
    all_endpoint_norm = []
    all_sampled_pts_px = []

    bid_to_node = {}

    for node_id, ds in enumerate(stroke_seq):
        bezier_id = ds["bezier_id"]
        stroke_uid = make_stroke_uid(source_file, hex_key, bezier_id)
        node_instance_uid = f"{sample_id}::node_{node_id}"

        pts = np.asarray(ds["mother_bezier"], dtype=float)

        shape_code = int(ds.get("shape_token", ds.get("shape_code", -1)))
        variant_id = int(ds.get("variant_id", ds.get("v_orig", 0)))
        v_orig = int(ds.get("v_orig", variant_id))

        arc_len_px = calculate_stroke_length(pts)
        chord_len_px = calculate_chord_length(pts)
        angle_rad = calculate_angle_rad(pts)

        width_mean = safe_float(ds.get("width_mean", 1.0), 1.0)
        width_token = quantize_width(width_mean, arc_len_px)

        p0 = pts[0]
        p3 = pts[3]
        center_px = (p0 + p3) / 2.0

        p0_norm = p0 / CANVAS_SIZE
        p3_norm = p3 / CANVAS_SIZE
        center_norm = center_px / CANVAS_SIZE

        sampled_pts = sample_cubic_bezier(pts, n=BEZIER_SAMPLE_N)
        all_sampled_pts_px.append(sampled_pts)
        all_endpoint_norm.append(p0_norm)
        all_endpoint_norm.append(p3_norm)

        p0_cx, p0_ox = get_cell_and_offset(p0[0])
        p0_cy, p0_oy = get_cell_and_offset(p0[1])
        p3_cx, p3_ox = get_cell_and_offset(p3[0])
        p3_cy, p3_oy = get_cell_and_offset(p3[1])

        primitive_available = shape_code in cluster_refs

        node = {
            "node_id": int(node_id),

            # 防冲突 ID
            "source_file": source_file,
            "hex_key": hex_key,
            "glyph_uid": glyph_uid,
            "source_stroke_uid": stroke_uid,
            "node_instance_uid": node_instance_uid,

            # 原始 stroke id
            "bezier_id": bezier_id,

            # primitive / codebook 信息
            "shape_code": shape_code,
            "variant_id": variant_id,
            "v_orig": v_orig,
            "width_token": int(width_token),
            "width_mean_px": round(width_mean, 6),

            "primitive_ref": {
                "shape_code": shape_code,
                "prototype_available": bool(primitive_available),
            },

            # 原始几何
            "mother_bezier_px": to_float_list(pts),
            "p0_px": to_float_list(p0),
            "p3_px": to_float_list(p3),
            "center_px": to_float_list(center_px),

            # 归一化几何
            "p0_norm": to_float_list(p0_norm),
            "p3_norm": to_float_list(p3_norm),
            "center_norm": to_float_list(center_norm),

            # token 兼容字段
            "p0_cell": [p0_cx, p0_cy],
            "p0_offset": [p0_ox, p0_oy],
            "p3_cell": [p3_cx, p3_cy],
            "p3_offset": [p3_ox, p3_oy],

            # solver 初始变量
            "solver_init": {
                "center_norm": to_float_list(center_norm),
                "angle_rad": round(float(angle_rad), 6),
                "angle_deg": round(float(math.degrees(angle_rad)), 6),
                "arc_length_norm": round(arc_len_px / CANVAS_SIZE, 6),
                "chord_length_norm": round(chord_len_px / CANVAS_SIZE, 6),
                "scale_px": round(chord_len_px, 6),
                "rotation_rad": round(float(angle_rad), 6),
            },

            # 后面根据 topology 再填
            "graph_role": {
                "degree": 0,
                "in_positive_edges": 0,
                "roles": [],
            },
        }

        nodes.append(node)

        # 双 key，兼容 int / str
        bid_to_node[bezier_id] = node_id
        bid_to_node[bid_key(bezier_id)] = node_id

    if all_endpoint_norm:
        glyph_center_norm = np.mean(np.stack(all_endpoint_norm, axis=0), axis=0)
    else:
        glyph_center_norm = np.array([0.5, 0.5], dtype=float)

    if all_sampled_pts_px:
        all_pts = np.concatenate(all_sampled_pts_px, axis=0)
        bbox_min_px, bbox_max_px = calculate_bbox(all_pts)
    else:
        bbox_min_px = np.array([0.0, 0.0])
        bbox_max_px = np.array([0.0, 0.0])

    bbox_size_px = bbox_max_px - bbox_min_px

    bbox_area_norm = float(
        (bbox_size_px[0] / CANVAS_SIZE)
        * (bbox_size_px[1] / CANVAS_SIZE)
    )

    for node in nodes:
        p0_norm = np.asarray(node["p0_norm"], dtype=float)
        p3_norm = np.asarray(node["p3_norm"], dtype=float)
        center_norm = np.asarray(node["center_norm"], dtype=float)

        node["p0_local_norm"] = to_float_list(p0_norm - glyph_center_norm)
        node["p3_local_norm"] = to_float_list(p3_norm - glyph_center_norm)
        node["center_local_norm"] = to_float_list(center_norm - glyph_center_norm)

    glyph_geom_summary = {
        "glyph_center_norm": to_float_list(glyph_center_norm),
        "bbox_min_norm": to_float_list(bbox_min_px / CANVAS_SIZE),
        "bbox_max_norm": to_float_list(bbox_max_px / CANVAS_SIZE),
        "bbox_size_norm": to_float_list(bbox_size_px / CANVAS_SIZE),
        "bbox_area_norm": round(bbox_area_norm, 6),
        "aspect_ratio": round(
            float(bbox_size_px[0] / max(bbox_size_px[1], 1e-6)),
            6,
        ),
    }

    return nodes, bid_to_node, glyph_geom_summary


# ==========================================
# 🕸️ Edge / Topology 构建
# ==========================================
def build_topology_edges(derived_events, bid_to_node, source_file, hex_key, glyph_uid, sample_id):
    undirected_edges = []
    directed_edges = []

    for ev_idx, ev in enumerate(derived_events):
        basic = extract_event_basic(ev)

        if basic is None:
            continue

        bid_a = basic["bid_a"]
        bid_b = basic["bid_b"]

        key_a = bid_a if bid_a in bid_to_node else bid_key(bid_a)
        key_b = bid_b if bid_b in bid_to_node else bid_key(bid_b)

        if key_a not in bid_to_node or key_b not in bid_to_node:
            continue

        u = bid_to_node[key_a]
        v = bid_to_node[key_b]

        if u == v:
            continue

        j_type = basic["j_type"]
        j_type_idx = J_TYPE_TO_IDX.get(j_type, 0)

        t_u = float(basic["t_a"])
        t_v = float(basic["t_b"])

        angle_deg = float(basic["angle_deg"])
        angle_sin, angle_cos = angle_to_sincos(angle_deg)

        edge_id = len(undirected_edges)
        edge_uid = f"{sample_id}::edge_{edge_id}"

        stroke_uid_a = make_stroke_uid(source_file, hex_key, bid_a)
        stroke_uid_b = make_stroke_uid(source_file, hex_key, bid_b)

        edge_base = {
            "edge_id": int(edge_id),
            "edge_uid": edge_uid,
            "source_event_idx": int(ev_idx),

            "source_file": source_file,
            "hex_key": hex_key,
            "glyph_uid": glyph_uid,

            "j_type": j_type,
            "j_type_idx": int(j_type_idx),

            "u": int(u),
            "v": int(v),

            "bid_u": bid_a,
            "bid_v": bid_b,

            "stroke_uid_u": stroke_uid_a,
            "stroke_uid_v": stroke_uid_b,

            "role_u": basic["role_a"],
            "role_v": basic["role_b"],

            "t_u": round(t_u, 6),
            "t_v": round(t_v, 6),
            "t_u_bin": quantize_t(t_u),
            "t_v_bin": quantize_t(t_v),

            "t_diff": round(abs(t_u - t_v), 6),
            "t_prod": round(t_u * t_v, 6),

            "angle_deg": round(angle_deg, 6),
            "angle_bin": quantize_angle_deg(angle_deg),
            "angle_sin": round(float(angle_sin), 6),
            "angle_cos": round(float(angle_cos), 6),
        }

        undirected_edges.append(edge_base)

        # 正向
        directed_edges.append({
            **edge_base,
            "directed": True,
            "direction": "forward",
        })

        # 反向：u/v、bid、role、t_u/t_v 必须互换
        directed_edges.append({
            **edge_base,
            "directed": True,
            "direction": "reverse",

            "u": int(v),
            "v": int(u),

            "bid_u": bid_b,
            "bid_v": bid_a,

            "stroke_uid_u": stroke_uid_b,
            "stroke_uid_v": stroke_uid_a,

            "role_u": basic["role_b"],
            "role_v": basic["role_a"],

            "t_u": round(t_v, 6),
            "t_v": round(t_u, 6),
            "t_u_bin": quantize_t(t_v),
            "t_v_bin": quantize_t(t_u),
        })

    return undirected_edges, directed_edges


def attach_graph_roles(nodes, undirected_edges, directed_edges):
    degree = Counter()
    role_hist = defaultdict(list)

    for e in undirected_edges:
        u, v = e["u"], e["v"]

        degree[u] += 1
        degree[v] += 1

        role_hist[u].append(e["role_u"])
        role_hist[v].append(e["role_v"])

    in_edge_count = Counter()

    for e in directed_edges:
        in_edge_count[e["v"]] += 1

    for node in nodes:
        i = node["node_id"]
        roles = role_hist.get(i, [])

        node["graph_role"] = {
            "degree": int(degree[i]),
            "in_positive_edges": int(in_edge_count[i]),
            "roles": sorted(list(set(roles))),
        }

    return nodes


def build_topology_summary(num_nodes, undirected_edges, directed_edges):
    edge_type_counts = Counter(e["j_type"] for e in undirected_edges)

    degree = Counter()

    for e in undirected_edges:
        degree[e["u"]] += 1
        degree[e["v"]] += 1

    degrees = [degree[i] for i in range(num_nodes)]
    comps = connected_components(num_nodes, undirected_edges)
    cycle_rank = calc_cycle_rank(num_nodes, undirected_edges)

    return {
        "num_nodes": int(num_nodes),
        "num_undirected_edges": int(len(undirected_edges)),
        "num_directed_edges": int(len(directed_edges)),
        "edge_type_counts": dict(edge_type_counts),
        "degrees": [int(x) for x in degrees],
        "max_degree": int(max(degrees) if degrees else 0),
        "num_components": int(len(comps)),
        "component_sizes": [int(len(c)) for c in comps],
        "cycle_rank": int(cycle_rank),
        "has_cycle": bool(cycle_rank > 0),
    }


# ==========================================
# 🧬 保留旧派生轨迹
# ==========================================
def build_sequential_trace(stroke_seq, derived_events, bid_to_node):
    drawn_bids = []
    emitted_events = set()
    trace = []

    for ds in stroke_seq:
        bid = ds["bezier_id"]
        node_id = bid_to_node[bid if bid in bid_to_node else bid_key(bid)]

        events_for_this_stroke = []

        for ev_idx, ev in enumerate(derived_events):
            if ev_idx in emitted_events:
                continue

            basic = extract_event_basic(ev)

            if basic is None:
                continue

            b_a = basic["bid_a"]
            b_b = basic["bid_b"]
            t_a = basic["t_a"]
            t_b = basic["t_b"]

            if (b_a == bid and b_b in drawn_bids) or (b_b == bid and b_a in drawn_bids):
                if b_a == bid:
                    t_self = t_a
                    t_target = t_b
                    target_bid = b_b
                else:
                    t_self = t_b
                    t_target = t_a
                    target_bid = b_a

                target_node = bid_to_node[
                    target_bid if target_bid in bid_to_node else bid_key(target_bid)
                ]

                target_dist = len(drawn_bids) - drawn_bids.index(target_bid)

                events_for_this_stroke.append({
                    "j_type": basic["j_type"],
                    "target_dist": int(target_dist),
                    "target_bid": target_bid,
                    "target_node": int(target_node),
                    "t_self": round(float(t_self), 6),
                    "t_target": round(float(t_target), 6),
                    "t_self_bin": quantize_t(t_self),
                    "t_target_bin": quantize_t(t_target),
                    "ev_idx": int(ev_idx),
                })

        if not events_for_this_stroke:
            trace.append({
                "token_type": "NEW_ROOT",
                "node_id": int(node_id),
                "bezier_id": bid,
            })
        else:
            for j_info in events_for_this_stroke:
                trace.append({
                    "token_type": "JUNCTION",
                    "node_id": int(node_id),
                    "bezier_id": bid,
                    "j_type": j_info["j_type"],
                    "target_dist": j_info["target_dist"],
                    "target_node": j_info["target_node"],
                    "target_bid": j_info["target_bid"],
                    "t_self": j_info["t_self"],
                    "t_target": j_info["t_target"],
                    "t_self_bin": j_info["t_self_bin"],
                    "t_target_bin": j_info["t_target_bin"],
                })
                emitted_events.add(j_info["ev_idx"])

        trace.append({
            "token_type": "STROKE",
            "node_id": int(node_id),
            "bezier_id": bid,
            "shape_code": int(ds.get("shape_token", ds.get("shape_code", -1))),
            "variant_id": int(ds.get("variant_id", ds.get("v_orig", 0))),
        })

        drawn_bids.append(bid)

    return trace


# ==========================================
# 🎨 Aesthetic / PCG 特征
# ==========================================
def build_aesthetic_features(nodes, topology_summary, glyph_geom_summary):
    lengths = [
        float(n["solver_init"]["arc_length_norm"])
        for n in nodes
    ]

    if lengths:
        length_mean = float(np.mean(lengths))
        length_std = float(np.std(lengths))
        length_min = float(np.min(lengths))
        length_max = float(np.max(lengths))
    else:
        length_mean = length_std = length_min = length_max = 0.0

    bbox_size = np.asarray(glyph_geom_summary["bbox_size_norm"], dtype=float)
    bbox_area = float(glyph_geom_summary["bbox_area_norm"])
    aspect_ratio = float(glyph_geom_summary["aspect_ratio"])

    num_nodes = int(topology_summary["num_nodes"])
    num_edges = int(topology_summary["num_undirected_edges"])
    max_possible_edges = max(1, num_nodes * (num_nodes - 1) / 2)

    graph_density = float(num_edges / max_possible_edges)

    center = np.asarray(glyph_geom_summary["glyph_center_norm"], dtype=float)
    center_bias = float(np.linalg.norm(center - np.array([0.5, 0.5])))

    return {
        "stroke_count": int(num_nodes),
        "edge_count": int(num_edges),
        "graph_density": round(graph_density, 6),

        "bbox_area_norm": round(bbox_area, 6),
        "bbox_width_norm": round(float(bbox_size[0]), 6),
        "bbox_height_norm": round(float(bbox_size[1]), 6),
        "aspect_ratio": round(aspect_ratio, 6),
        "center_bias": round(center_bias, 6),
        "fill_ratio": round(bbox_area, 6),

        "length_mean_norm": round(length_mean, 6),
        "length_std_norm": round(length_std, 6),
        "length_min_norm": round(length_min, 6),
        "length_max_norm": round(length_max, 6),

        "cycle_rank": int(topology_summary["cycle_rank"]),
        "has_cycle": bool(topology_summary["has_cycle"]),

        "num_E2E": int(topology_summary["edge_type_counts"].get("E2E", 0)),
        "num_T": int(topology_summary["edge_type_counts"].get("T", 0)),
        "num_X": int(topology_summary["edge_type_counts"].get("X", 0)),
    }


# ==========================================
# 📊 Grammar 统计
# ==========================================
class GrammarStats:
    def __init__(self):
        self.stroke_count_hist = Counter()
        self.edge_count_hist = Counter()
        self.edge_type_hist = Counter()
        self.degree_hist = Counter()

        self.shape_code_hist = Counter()
        self.width_token_hist = Counter()

        self.shape_by_degree = defaultdict(Counter)
        self.width_by_shape = defaultdict(Counter)
        self.shape_by_role = defaultdict(Counter)

        self.t_bin_by_edge_type = defaultdict(Counter)
        self.angle_bin_by_edge_type = defaultdict(Counter)

        self.motif_hist = Counter()

        self.source_file_hist = Counter()
        self.glyph_uid_hist = Counter()

    def update(self, sample):
        topo = sample["topology"]
        nodes = sample["nodes"]
        edges = sample["topology"]["positive_edges_undirected"]

        n = topo["num_nodes"]
        m = topo["num_undirected_edges"]

        self.stroke_count_hist[n] += 1
        self.edge_count_hist[m] += 1

        self.source_file_hist[sample["source_file"]] += 1
        self.glyph_uid_hist[sample["glyph_uid"]] += 1

        if topo["has_cycle"]:
            self.motif_hist["cycle"] += 1

        if topo["edge_type_counts"].get("T", 0) > 0:
            self.motif_hist["has_T"] += 1

        if topo["edge_type_counts"].get("X", 0) > 0:
            self.motif_hist["has_X"] += 1

        if topo["edge_type_counts"].get("E2E", 0) > 0:
            self.motif_hist["has_E2E"] += 1

        if topo["max_degree"] >= 3:
            self.motif_hist["branch_or_hub"] += 1

        for k, v in topo["edge_type_counts"].items():
            self.edge_type_hist[k] += int(v)

        for d in topo["degrees"]:
            self.degree_hist[int(d)] += 1

        for node in nodes:
            shape = int(node["shape_code"])
            width = int(node["width_token"])
            degree = int(node["graph_role"]["degree"])
            roles = node["graph_role"]["roles"]

            self.shape_code_hist[shape] += 1
            self.width_token_hist[width] += 1

            self.shape_by_degree[str(degree)][str(shape)] += 1
            self.width_by_shape[str(shape)][str(width)] += 1

            if not roles:
                self.shape_by_role["none"][str(shape)] += 1
            else:
                for r in roles:
                    self.shape_by_role[r][str(shape)] += 1

        for e in edges:
            jt = e["j_type"]

            self.t_bin_by_edge_type[jt][str(e["t_u_bin"])] += 1
            self.t_bin_by_edge_type[jt][str(e["t_v_bin"])] += 1

            self.angle_bin_by_edge_type[jt][str(e["angle_bin"])] += 1

    def to_json(self):
        def counter_to_dict(c):
            return {str(k): int(v) for k, v in c.items()}

        def nested_counter_to_dict(d):
            return {
                str(k): counter_to_dict(v)
                for k, v in d.items()
            }

        return {
            "stroke_count_hist": counter_to_dict(self.stroke_count_hist),
            "edge_count_hist": counter_to_dict(self.edge_count_hist),
            "edge_type_hist": counter_to_dict(self.edge_type_hist),
            "degree_hist": counter_to_dict(self.degree_hist),

            "shape_code_hist": counter_to_dict(self.shape_code_hist),
            "width_token_hist": counter_to_dict(self.width_token_hist),

            "shape_by_degree": nested_counter_to_dict(self.shape_by_degree),
            "width_by_shape": nested_counter_to_dict(self.width_by_shape),
            "shape_by_role": nested_counter_to_dict(self.shape_by_role),

            "t_bin_by_edge_type": nested_counter_to_dict(self.t_bin_by_edge_type),
            "angle_bin_by_edge_type": nested_counter_to_dict(self.angle_bin_by_edge_type),

            "motif_hist": counter_to_dict(self.motif_hist),

            "source_file_hist": counter_to_dict(self.source_file_hist),
            "glyph_uid_hist": counter_to_dict(self.glyph_uid_hist),
        }


def update_primitive_library_stats(primitive_library, sample):
    for node in sample["nodes"]:
        cid = str(node["shape_code"])

        if cid not in primitive_library:
            continue

        degree = str(node["graph_role"]["degree"])
        width = str(node["width_token"])
        roles = node["graph_role"]["roles"]
        length_norm = float(node["solver_init"]["arc_length_norm"])

        primitive_library[cid]["count_instances"] += 1
        primitive_library[cid]["length_norm_values"].append(length_norm)

        primitive_library[cid]["width_token_hist"][width] = (
            primitive_library[cid]["width_token_hist"].get(width, 0) + 1
        )

        primitive_library[cid]["degree_hist"][degree] = (
            primitive_library[cid]["degree_hist"].get(degree, 0) + 1
        )

        if not roles:
            primitive_library[cid]["role_hist"]["none"] = (
                primitive_library[cid]["role_hist"].get("none", 0) + 1
            )
        else:
            for r in roles:
                primitive_library[cid]["role_hist"][r] = (
                    primitive_library[cid]["role_hist"].get(r, 0) + 1
                )


def finalize_primitive_library(primitive_library):
    for cid, item in primitive_library.items():
        vals = item.get("length_norm_values", [])

        if vals:
            item["length_norm_mean"] = round(float(np.mean(vals)), 6)
            item["length_norm_std"] = round(float(np.std(vals)), 6)
            item["length_norm_min"] = round(float(np.min(vals)), 6)
            item["length_norm_max"] = round(float(np.max(vals)), 6)
        else:
            item["length_norm_mean"] = 0.0
            item["length_norm_std"] = 0.0
            item["length_norm_min"] = 0.0
            item["length_norm_max"] = 0.0

        item.pop("length_norm_values", None)

        if len(item["source_refs"]) > 20:
            item["source_refs_preview"] = item["source_refs"][:20]
            item["source_refs_count"] = len(item["source_refs"])
            item.pop("source_refs", None)

    return primitive_library


# ==========================================
# 🚀 主程序
# ==========================================
def main():
    print("📖 正在加载 Shape Codebook / Stroke Primitive Library...")
    shape_dict, cluster_refs, primitive_library = load_shape_codebook(CLUSTER_FILE)

    topo_files = sorted(glob.glob(os.path.join(TOPO_DATA_DIR, "*_topo.json")))

    glyph_samples = []
    grammar_stats = GrammarStats()

    orig_sample_count = 0
    skipped_no_shape = 0
    skipped_empty = 0
    skipped_no_uid_match = 0

    print(f"📁 TOPO_DATA_DIR: {TOPO_DATA_DIR}")
    print(f"📄 topo 文件数: {len(topo_files)}")
    print("🧩 正在构建 PCG Glyph Corpus...")

    for fp in tqdm(topo_files):
        source_file = os.path.basename(fp)

        with open(fp, "r", encoding="utf-8") as f:
            char_data_map = json.load(f)

        for hex_key, char_data in char_data_map.items():
            glyph_uid = make_glyph_uid(source_file, hex_key)

            # 核心：必须用 source_file::hex_key 查 shape_dict
            if glyph_uid not in shape_dict:
                skipped_no_uid_match += 1
                continue

            orig_sample_count += 1

            orig_strokes = []

            for s in char_data.get("strokes", []):
                bezier_id = s["bezier_id"]

                key = bezier_id if bezier_id in shape_dict[glyph_uid] else bid_key(bezier_id)

                if key not in shape_dict[glyph_uid]:
                    skipped_no_shape += 1
                    continue

                cid = int(shape_dict[glyph_uid][key])

                stroke_payload = s.copy()

                stroke_payload["source_file"] = source_file
                stroke_payload["hex_key"] = hex_key
                stroke_payload["glyph_uid"] = glyph_uid
                stroke_payload["source_stroke_uid"] = make_stroke_uid(source_file, hex_key, bezier_id)

                stroke_payload["shape_token"] = cid
                stroke_payload["shape_code"] = cid

                v_orig = 0

                if cid in cluster_refs:
                    y_raw = normalize_and_sample_function(
                        s["mother_bezier"],
                        N=50,
                    )

                    if y_raw is not None:
                        y0 = y_raw.copy()
                        y1 = -y_raw[::-1]
                        y2 = -y_raw
                        y3 = y_raw[::-1]

                        candidates = [y0, y1, y2, y3]

                        dists = [
                            np.mean(np.abs(cluster_refs[cid] - v))
                            for v in candidates
                        ]

                        v_orig = int(np.argmin(dists))

                stroke_payload["v_orig"] = int(v_orig)

                orig_strokes.append(stroke_payload)

            if not orig_strokes:
                skipped_empty += 1
                continue

            topo_events = char_data.get("topology_events", [])

            # 注意：这里传入 glyph_uid，而不是 hex_key
            # 因为 generate_derived_sequences 内部可能使用 shape_dict[hex_key]
            derived_samples = generate_derived_sequences(
                strokes=orig_strokes,
                topo_events=topo_events,
                shape_dict=shape_dict,
                hex_key=glyph_uid,
                cluster_refs=cluster_refs,
                max_order_samples=MAX_ORDER_SAMPLES,
                rot_mode=ROTATION_MODE,
                mirror_mode=MIRROR_MODE,
            )

            for derived_idx, (rule_name, stroke_seq, derived_events) in enumerate(derived_samples):
                if not stroke_seq:
                    skipped_empty += 1
                    continue

                sample_id = make_sample_id(glyph_uid, rule_name, derived_idx)

                nodes, bid_to_node, glyph_geom_summary = build_nodes_from_strokes(
                    stroke_seq=stroke_seq,
                    cluster_refs=cluster_refs,
                    source_file=source_file,
                    hex_key=hex_key,
                    glyph_uid=glyph_uid,
                    sample_id=sample_id,
                )

                undirected_edges, directed_edges = build_topology_edges(
                    derived_events=derived_events,
                    bid_to_node=bid_to_node,
                    source_file=source_file,
                    hex_key=hex_key,
                    glyph_uid=glyph_uid,
                    sample_id=sample_id,
                )

                nodes = attach_graph_roles(
                    nodes=nodes,
                    undirected_edges=undirected_edges,
                    directed_edges=directed_edges,
                )

                topology_summary = build_topology_summary(
                    num_nodes=len(nodes),
                    undirected_edges=undirected_edges,
                    directed_edges=directed_edges,
                )

                sequential_trace = build_sequential_trace(
                    stroke_seq=stroke_seq,
                    derived_events=derived_events,
                    bid_to_node=bid_to_node,
                )

                aesthetic_features = build_aesthetic_features(
                    nodes=nodes,
                    topology_summary=topology_summary,
                    glyph_geom_summary=glyph_geom_summary,
                )

                sample = {
                    "sample_id": sample_id,

                    # 防冲突字符 key
                    "glyph_uid": glyph_uid,
                    "char_key": glyph_uid,

                    "source_file": source_file,
                    "hex_key": hex_key,
                    "char": char_data.get("glyph_info", {}).get("char", ""),

                    "derivation": {
                        "rule_name": rule_name,
                        "derived_index": int(derived_idx),
                        "max_order_samples": MAX_ORDER_SAMPLES,
                        "rotation_mode": ROTATION_MODE,
                        "mirror_mode": MIRROR_MODE,
                    },

                    "nodes": nodes,

                    "topology": {
                        **topology_summary,
                        "positive_edges_undirected": undirected_edges,
                        "positive_edges_directed": directed_edges,
                    },

                    "geometry": glyph_geom_summary,

                    "aesthetic_features": aesthetic_features,

                    # 保留旧 token 生成轨迹，方便 baseline / ablation
                    "sequential_trace": sequential_trace,
                }

                glyph_samples.append(sample)
                grammar_stats.update(sample)
                update_primitive_library_stats(primitive_library, sample)

    primitive_library = finalize_primitive_library(primitive_library)

    output = {
        "schema_version": "alien_glyph_pcg_corpus_v2_source_aware",

        "description": (
            "PCG-oriented alien glyph corpus built from expert-annotated "
            "Bezier strokes and topology. The corpus is source-aware: glyph keys "
            "are stored as source_file::hex_key to avoid collisions across fonts."
        ),

        "config": {
            "canvas_size": CANVAS_SIZE,
            "topo_data_dir": TOPO_DATA_DIR,
            "cluster_file": CLUSTER_FILE,

            "grid_bins": GRID_BINS,
            "t_bins": T_BINS,
            "angle_bins": ANGLE_BINS,
            "bezier_sample_n": BEZIER_SAMPLE_N,

            "derivation": {
                "max_order_samples": MAX_ORDER_SAMPLES,
                "rotation_mode": ROTATION_MODE,
                "mirror_mode": MIRROR_MODE,
            },

            "edge_types": J_TYPE_TO_IDX,

            "id_policy": {
                "glyph_uid": "source_file::hex_key",
                "stroke_uid": "source_file::hex_key::bezier_id",
                "sample_id": "source_file::hex_key::rule_name::derived_idx",
            },
        },

        "stroke_primitive_library": primitive_library,

        "grammar_stats": grammar_stats.to_json(),

        "glyph_samples": glyph_samples,

        "build_report": {
            "topo_file_count": len(topo_files),
            "original_valid_char_count": int(orig_sample_count),
            "derived_sample_count": int(len(glyph_samples)),

            "skipped_no_uid_match_count": int(skipped_no_uid_match),
            "skipped_no_shape_count": int(skipped_no_shape),
            "skipped_empty_count": int(skipped_empty),

            "avg_derivation_per_original": (
                round(len(glyph_samples) / orig_sample_count, 4)
                if orig_sample_count > 0
                else 0.0
            ),

            "unique_source_file_count": len(
                set(s["source_file"] for s in glyph_samples)
            ),

            "unique_glyph_uid_count": len(
                set(s["glyph_uid"] for s in glyph_samples)
            ),
        },
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n🎉 PCG Glyph Corpus 构建完成！")
    print(f"📊 原始有效字符数: {orig_sample_count}")
    print(f"📈 派生后样本数: {len(glyph_samples)}")
    print(f"🧱 Stroke primitive 数量: {len(primitive_library)}")
    print(f"📁 unique source files: {output['build_report']['unique_source_file_count']}")
    print(f"🔑 unique glyph_uid: {output['build_report']['unique_glyph_uid_count']}")
    print(f"⚠️ skipped_no_uid_match: {skipped_no_uid_match}")
    print(f"⚠️ skipped_no_shape: {skipped_no_shape}")
    print(f"⚠️ skipped_empty: {skipped_empty}")
    print(f"💾 数据已保存至: {OUTPUT_FILE}")

    print("\n📌 下一步建议：")
    print("  1. 用 grammar_stats 写 graph_grammar_sampler.py")
    print("  2. 用 stroke_primitive_library 写 stroke_primitive_sampler.py")
    print("  3. 用 topology.positive_edges_undirected 写 constraint_solver.py")
    print("  4. 用 aesthetic_features 写 aesthetic_scorer.py")
    print("  5. 再接 LLM，把自然语言 prompt 转为 grammar / scorer 参数")


if __name__ == "__main__":
    main()