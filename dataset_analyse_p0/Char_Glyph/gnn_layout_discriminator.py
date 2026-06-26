import os
import json
import math
import random
from copy import deepcopy
from collections import Counter, defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


# =========================================================
# ⚙️ 全局配置
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LAYOUT_TEMPLATE_FILE = os.path.join(SCRIPT_DIR, "glyph_layout_templates.json")
SOLVED_FILE = os.path.join(SCRIPT_DIR, "solved_glyph_candidates.json")

OUTPUT_MODEL_FILE = os.path.join(SCRIPT_DIR, "gnn_layout_discriminator_v2.pt")
OUTPUT_SCORE_FILE = os.path.join(SCRIPT_DIR, "gnn_layout_scores_v2.json")
OUTPUT_CURVE_FILE = os.path.join(SCRIPT_DIR, "gnn_layout_training_curves_v2.png")

RANDOM_SEED = 42

# auto / cuda / mps / cpu
DEVICE_MODE = "auto"

# 调试时可改小
MAX_REAL_GRAPHS = None

# 第一版先不要用 solved fake，避免标签污染
USE_SOLVED_AS_FAKE = False

# 如果之后想加入 solved fake，可改 True
MAX_SOLVED_FAKE_GRAPHS = None
SOLVED_FAKE_MODE = "rejected_only"
# 可选：
#   all
#   rejected_only
#   rejected_and_rough

# 每个 real template 生成几个 corrupted fake
CORRUPTED_FAKE_PER_REAL = 2

# 只保留强 fake，先验证模型能不能学到显著 layout 差异
CORRUPTED_FAKE_MODES = [
    "random_centers",
    "collapse",
    "rotation_scramble",
]

# fake 最多是真实样本的多少倍
MAX_FAKE_RATIO = 2.5

# 按 glyph_uid 分组切分，防止 derived 样本泄漏
TEST_GROUP_RATIO = 0.20

# 训练参数
EPOCHS = 120
BATCH_SIZE = 64
LR = 2e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 128
NUM_GNN_LAYERS = 4
DROPOUT = 0.15

# 是否使用 balanced batch
USE_BALANCED_BATCH = True

# 特征维度
SHAPE_CODE_DIM = 64
WIDTH_TOKEN_DIM = 8
JTYPE_DIM = 4

CANVAS_SIZE = 400.0

# fake 构造强度
CENTER_JITTER_STD = 0.16
ROT_JITTER_STD_DEG = 65.0
SCALE_JITTER_STD = 0.50

SAVE_ALL_SCORES = True


# =========================================================
# 🧮 基础工具
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device():
    if DEVICE_MODE == "cuda":
        return torch.device("cuda")
    if DEVICE_MODE == "mps":
        return torch.device("mps")
    if DEVICE_MODE == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


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


def one_hot(idx, dim):
    arr = np.zeros(dim, dtype=np.float32)
    idx = int(idx)

    if 0 <= idx < dim:
        arr[idx] = 1.0
    else:
        arr[-1] = 1.0

    return arr


def deg_to_rad(deg):
    return float(deg) * math.pi / 180.0


def normalize_angle_pi(rad):
    a = float(rad)

    while a <= -math.pi:
        a += 2.0 * math.pi
    while a > math.pi:
        a -= 2.0 * math.pi

    return a


def angle_to_sincos(rad):
    return math.sin(rad), math.cos(rad)


def edge_type_name(edge):
    if "j_type" in edge:
        return str(edge["j_type"])

    idx = safe_int(edge.get("j_type_idx", 0), 0)

    return {
        1: "E2E",
        2: "X",
        3: "T",
    }.get(idx, "UNKNOWN")


def jtype_to_idx(jtype):
    return {
        "E2E": 0,
        "T": 1,
        "X": 2,
        "UNKNOWN": 3,
    }.get(str(jtype), 3)


def get_edges(obj):
    if not isinstance(obj, dict):
        return []

    topo = obj.get("topology", {})

    if isinstance(topo, dict):
        if "positive_edges_undirected" in topo and isinstance(topo["positive_edges_undirected"], list):
            return topo["positive_edges_undirected"]

        if "edges" in topo and isinstance(topo["edges"], list):
            return topo["edges"]

        if "positive_edges_directed" in topo and isinstance(topo["positive_edges_directed"], list):
            return [
                e for e in topo["positive_edges_directed"]
                if e.get("direction", "forward") == "forward"
            ]

    if "edges" in obj and isinstance(obj["edges"], list):
        return obj["edges"]

    return []


def get_nodes(obj):
    if "nodes" in obj and isinstance(obj["nodes"], list):
        return obj["nodes"]

    if "solved_nodes" in obj and isinstance(obj["solved_nodes"], list):
        return obj["solved_nodes"]

    return []


def get_graph_id(obj, fallback):
    for k in [
        "layout_template_id",
        "generated_glyph_id",
        "source_sample_id",
        "sample_id",
        "glyph_uid",
    ]:
        if obj.get(k, ""):
            return str(obj[k])

    return fallback


def get_group_id(obj, fallback):
    for k in ["glyph_uid", "source_file"]:
        if obj.get(k, ""):
            return str(obj[k])

    return fallback


# =========================================================
# 📖 读取 node 属性
# =========================================================
def read_center_from_node(node):
    if "layout_prior" in node:
        lp = node.get("layout_prior", {})
        if "center_norm" in lp:
            arr = np.asarray(lp["center_norm"], dtype=np.float32)
            return arr[:2]

    if "center_norm" in node:
        arr = np.asarray(node["center_norm"], dtype=np.float32)
        return arr[:2]

    if "center_px" in node:
        arr = np.asarray(node["center_px"], dtype=np.float32)
        return arr[:2] / float(CANVAS_SIZE)

    return np.array([0.5, 0.5], dtype=np.float32)


def read_angle_from_node(node):
    if "layout_prior" in node:
        lp = node.get("layout_prior", {})

        if "rotation_rad" in lp:
            return safe_float(lp.get("rotation_rad", 0.0), 0.0)

        if "rotation_deg" in lp:
            return deg_to_rad(safe_float(lp.get("rotation_deg", 0.0), 0.0))

    if "rotation_rad" in node:
        return safe_float(node.get("rotation_rad", 0.0), 0.0)

    if "rotation_deg" in node:
        return deg_to_rad(safe_float(node.get("rotation_deg", 0.0), 0.0))

    return 0.0


def read_length_from_node(node):
    if "layout_prior" in node:
        lp = node.get("layout_prior", {})

        for k in ["length_norm", "scale_norm", "arc_length_norm", "chord_length_norm"]:
            if k in lp:
                return safe_float(lp[k], 0.25)

    if "actual_length_px" in node:
        return safe_float(node["actual_length_px"], 100.0) / float(CANVAS_SIZE)

    if "target_length_px" in node:
        return safe_float(node["target_length_px"], 100.0) / float(CANVAS_SIZE)

    if "scale_px" in node:
        return safe_float(node["scale_px"], 100.0) / float(CANVAS_SIZE)

    solver_priors = node.get("solver_priors", {})
    if isinstance(solver_priors, dict):
        if "length_prior_norm" in solver_priors:
            return safe_float(solver_priors["length_prior_norm"], 0.25)

    return 0.25


def read_scale_from_node(node):
    if "layout_prior" in node:
        lp = node.get("layout_prior", {})
        if "scale_norm" in lp:
            return safe_float(lp["scale_norm"], 0.25)

    if "scale_px" in node:
        return safe_float(node["scale_px"], 100.0) / float(CANVAS_SIZE)

    return read_length_from_node(node)


def read_role_flags(node):
    role_obj = node.get("grammar_role", node.get("graph_role", {}))

    roles = []

    if isinstance(role_obj, dict):
        raw = role_obj.get("roles", [])
        if isinstance(raw, list):
            roles = [str(x).lower() for x in raw]
        elif isinstance(raw, str):
            roles = [raw.lower()]
    elif isinstance(role_obj, list):
        roles = [str(x).lower() for x in role_obj]
    elif isinstance(role_obj, str):
        roles = [role_obj.lower()]

    guest = any("guest" in r for r in roles)
    host = any("host" in r for r in roles)
    stroke = any("stroke" in r for r in roles)

    none = not (guest or host or stroke)

    return np.array(
        [
            1.0 if guest else 0.0,
            1.0 if host else 0.0,
            1.0 if stroke else 0.0,
            1.0 if none else 0.0,
        ],
        dtype=np.float32,
    )


# =========================================================
# 🌍 显式 graph-level geometry features
# =========================================================
def extract_global_graph_features(obj):
    nodes = get_nodes(obj)
    edges = get_edges(obj)

    if len(nodes) == 0:
        return np.zeros(48, dtype=np.float32)

    centers = []
    rotations = []
    lengths = []
    scales = []
    shape_codes = []
    width_tokens = []

    for n in nodes:
        centers.append(read_center_from_node(n))
        rotations.append(read_angle_from_node(n))
        lengths.append(read_length_from_node(n))
        scales.append(read_scale_from_node(n))
        shape_codes.append(safe_int(n.get("shape_code", n.get("shape_token", 0)), 0))
        width_tokens.append(safe_int(n.get("width_token", 0), 0))

    centers = np.asarray(centers, dtype=np.float32)
    rotations = np.asarray(rotations, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)

    x = centers[:, 0]
    y = centers[:, 1]

    bbox_w = float(np.max(x) - np.min(x))
    bbox_h = float(np.max(y) - np.min(y))
    bbox_area = bbox_w * bbox_h
    aspect = bbox_w / max(bbox_h, 1e-6)

    center_mean = np.mean(centers, axis=0)
    center_std = np.std(centers, axis=0)

    pair_dists = []

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            pair_dists.append(float(np.linalg.norm(centers[i] - centers[j])))

    if len(pair_dists) == 0:
        pair_dists = [0.0]

    id_to_idx = {}

    for i, n in enumerate(nodes):
        id_to_idx[safe_int(n.get("node_id", i), i)] = i

    degrees = np.zeros(len(nodes), dtype=np.float32)

    edge_dists = []
    edge_dx = []
    edge_dy = []
    edge_dir_sin = []
    edge_dir_cos = []
    rot_delta_sin = []
    rot_delta_cos = []
    edge_types = Counter()

    valid_edge_count = 0

    for e in edges:
        u_raw = safe_int(e.get("u", -1), -1)
        v_raw = safe_int(e.get("v", -1), -1)

        if u_raw not in id_to_idx or v_raw not in id_to_idx:
            continue

        u = id_to_idx[u_raw]
        v = id_to_idx[v_raw]

        if u == v:
            continue

        valid_edge_count += 1
        degrees[u] += 1.0
        degrees[v] += 1.0

        dxy = centers[v] - centers[u]
        d = float(np.linalg.norm(dxy))

        edge_dists.append(d)
        edge_dx.append(float(dxy[0]))
        edge_dy.append(float(dxy[1]))

        if d > 1e-8:
            edge_dir_cos.append(float(dxy[0] / d))
            edge_dir_sin.append(float(dxy[1] / d))
        else:
            edge_dir_cos.append(1.0)
            edge_dir_sin.append(0.0)

        rd = normalize_angle_pi(rotations[v] - rotations[u])
        rot_delta_sin.append(math.sin(rd))
        rot_delta_cos.append(math.cos(rd))

        edge_types[edge_type_name(e)] += 1

    if len(edge_dists) == 0:
        edge_dists = [0.0]
        edge_dx = [0.0]
        edge_dy = [0.0]
        edge_dir_sin = [0.0]
        edge_dir_cos = [1.0]
        rot_delta_sin = [0.0]
        rot_delta_cos = [1.0]

    rot_sin = np.sin(rotations)
    rot_cos = np.cos(rotations)

    n_nodes = len(nodes)
    n_edges = max(1, valid_edge_count)

    # graph density, undirected
    max_edges = max(1.0, n_nodes * (n_nodes - 1) / 2.0)
    density = valid_edge_count / max_edges

    unique_shape_ratio = len(set(shape_codes)) / max(1, n_nodes)
    structural_ratio = sum(1 for s in shape_codes if s == 20) / max(1, n_nodes)

    # 大部分特征都在 0~1 或小范围，aspect 做缩放防爆
    feat = np.array([
        n_nodes / 10.0,
        valid_edge_count / 20.0,
        density,

        bbox_w,
        bbox_h,
        bbox_area,
        clamp(aspect / 5.0, 0.0, 5.0),

        center_mean[0],
        center_mean[1],
        center_std[0],
        center_std[1],

        np.mean(pair_dists),
        np.std(pair_dists),
        np.min(pair_dists),
        np.max(pair_dists),

        np.mean(edge_dists),
        np.std(edge_dists),
        np.min(edge_dists),
        np.max(edge_dists),

        np.mean(edge_dx),
        np.std(edge_dx),
        np.mean(edge_dy),
        np.std(edge_dy),

        np.mean(edge_dir_sin),
        np.mean(edge_dir_cos),
        np.std(edge_dir_sin),
        np.std(edge_dir_cos),

        np.mean(lengths),
        np.std(lengths),
        np.min(lengths),
        np.max(lengths),

        np.mean(scales),
        np.std(scales),

        np.mean(rot_sin),
        np.mean(rot_cos),
        np.std(rot_sin),
        np.std(rot_cos),

        np.mean(rot_delta_sin),
        np.mean(rot_delta_cos),
        np.std(rot_delta_sin),
        np.std(rot_delta_cos),

        np.mean(degrees) / 5.0,
        np.std(degrees) / 5.0,
        np.max(degrees) / 8.0,

        edge_types.get("E2E", 0) / n_edges,
        edge_types.get("T", 0) / n_edges,
        edge_types.get("X", 0) / n_edges,

        unique_shape_ratio,
        structural_ratio,
    ], dtype=np.float32)

    feat = np.nan_to_num(feat, nan=0.0, posinf=10.0, neginf=-10.0)
    feat = np.clip(feat, -10.0, 10.0)

    return feat.astype(np.float32)


# =========================================================
# 🧱 node / edge features
# =========================================================
def extract_node_feature(node, degree):
    shape_code = safe_int(node.get("shape_code", node.get("shape_token", 0)), 0)
    width_token = safe_int(node.get("width_token", 0), 0)

    shape_code_clamped = clamp(shape_code, 0, SHAPE_CODE_DIM - 1)
    width_token_clamped = clamp(width_token, 0, WIDTH_TOKEN_DIM - 1)

    center = read_center_from_node(node)
    center = np.clip(center.astype(np.float32), 0.0, 1.0)

    rotation = read_angle_from_node(node)
    rsin, rcos = math.sin(rotation), math.cos(rotation)

    length_norm = clamp(read_length_from_node(node), 0.0, 1.5)
    scale_norm = clamp(read_scale_from_node(node), 0.0, 1.5)

    role_flags = read_role_flags(node)

    is_structural = 1.0 if shape_code == 20 else 0.0

    feat = np.concatenate(
        [
            one_hot(shape_code_clamped, SHAPE_CODE_DIM),
            one_hot(width_token_clamped, WIDTH_TOKEN_DIM),

            np.array(
                [
                    degree / 8.0,
                    center[0],
                    center[1],
                    rsin,
                    rcos,
                    length_norm,
                    scale_norm,
                    is_structural,
                ],
                dtype=np.float32,
            ),

            role_flags,
        ],
        axis=0,
    )

    return feat.astype(np.float32)


def extract_edge_feature(edge, node_i, node_j, reverse=False):
    jtype = edge_type_name(edge)
    jidx = jtype_to_idx(jtype)

    if not reverse:
        t_src = safe_float(edge.get("t_u", 0.0), 0.0)
        t_dst = safe_float(edge.get("t_v", 0.0), 0.0)
    else:
        t_src = safe_float(edge.get("t_v", 0.0), 0.0)
        t_dst = safe_float(edge.get("t_u", 0.0), 0.0)

    if "angle_sin" in edge and "angle_cos" in edge:
        angle_sin = safe_float(edge.get("angle_sin", 0.0), 0.0)
        angle_cos = safe_float(edge.get("angle_cos", 1.0), 1.0)
    elif "angle_deg" in edge:
        ar = deg_to_rad(safe_float(edge.get("angle_deg", 0.0), 0.0))
        angle_sin, angle_cos = math.sin(ar), math.cos(ar)
    else:
        angle_sin, angle_cos = 0.0, 1.0

    ci = read_center_from_node(node_i)
    cj = read_center_from_node(node_j)

    if reverse:
        src = cj
        dst = ci
    else:
        src = ci
        dst = cj

    dxy = dst - src
    dist = float(np.linalg.norm(dxy))

    if dist < 1e-8:
        dir_cos = 1.0
        dir_sin = 0.0
    else:
        dir_cos = float(dxy[0] / dist)
        dir_sin = float(dxy[1] / dist)

    ri = read_angle_from_node(node_i)
    rj = read_angle_from_node(node_j)

    if reverse:
        rot_delta = normalize_angle_pi(ri - rj)
    else:
        rot_delta = normalize_angle_pi(rj - ri)

    rot_delta_sin = math.sin(rot_delta)
    rot_delta_cos = math.cos(rot_delta)

    same_shape = (
        1.0
        if safe_int(node_i.get("shape_code", -1), -1) == safe_int(node_j.get("shape_code", -2), -2)
        else 0.0
    )

    feat = np.concatenate(
        [
            one_hot(jidx, JTYPE_DIM),

            np.array(
                [
                    t_src,
                    t_dst,
                    angle_sin,
                    angle_cos,

                    float(dxy[0]),
                    float(dxy[1]),
                    dist,

                    dir_sin,
                    dir_cos,

                    rot_delta_sin,
                    rot_delta_cos,

                    same_shape,
                ],
                dtype=np.float32,
            ),
        ],
        axis=0,
    )

    return feat.astype(np.float32)


def graph_from_object(obj, label, kind, graph_id=None, group_id=None):
    nodes = get_nodes(obj)
    edges = get_edges(obj)

    if graph_id is None:
        graph_id = get_graph_id(obj, f"{kind}_unknown")

    if group_id is None:
        group_id = get_group_id(obj, graph_id)

    if len(nodes) < 2 or len(edges) < 1:
        return None

    raw_node_ids = []

    for i, n in enumerate(nodes):
        raw_node_ids.append(safe_int(n.get("node_id", i), i))

    id_to_idx = {nid: i for i, nid in enumerate(raw_node_ids)}

    n_nodes = len(nodes)
    degrees = np.zeros(n_nodes, dtype=np.float32)

    valid_edges = []

    for e in edges:
        u_raw = safe_int(e.get("u", -1), -1)
        v_raw = safe_int(e.get("v", -1), -1)

        if u_raw not in id_to_idx or v_raw not in id_to_idx:
            continue

        u = id_to_idx[u_raw]
        v = id_to_idx[v_raw]

        if u == v:
            continue

        valid_edges.append((u, v, e))
        degrees[u] += 1.0
        degrees[v] += 1.0

    if len(valid_edges) < 1:
        return None

    node_feats = []

    for i, node in enumerate(nodes):
        node_feats.append(extract_node_feature(node, degrees[i]))

    node_feats = np.asarray(node_feats, dtype=np.float32)

    edge_feat_dim = JTYPE_DIM + 12
    edge_feats = np.zeros((n_nodes, n_nodes, edge_feat_dim), dtype=np.float32)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    for u, v, e in valid_edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0

        edge_feats[u, v] = extract_edge_feature(e, nodes[u], nodes[v], reverse=False)
        edge_feats[v, u] = extract_edge_feature(e, nodes[u], nodes[v], reverse=True)

    global_feats = extract_global_graph_features(obj)

    return {
        "graph_id": str(graph_id),
        "group_id": str(group_id),
        "kind": str(kind),
        "label": float(label),

        "node_feats": node_feats,
        "edge_feats": edge_feats,
        "global_feats": global_feats,
        "adj": adj,

        "num_nodes": int(n_nodes),
        "num_edges": int(len(valid_edges)),

        "source_file": obj.get("source_file", ""),
        "glyph_uid": obj.get("glyph_uid", ""),
    }


# =========================================================
# 🧪 fake 构造
# =========================================================
def corrupt_template(template, mode, rng, fake_idx):
    t = deepcopy(template)

    t["layout_template_id"] = f"{template.get('layout_template_id', 'template')}_fake_{mode}_{fake_idx}"
    t["fake_mode"] = mode

    nodes = t.get("nodes", [])

    if len(nodes) == 0:
        return t

    centers = []

    for n in nodes:
        centers.append(read_center_from_node(n))

    centers = np.asarray(centers, dtype=np.float32)

    if mode == "random_centers":
        for n in nodes:
            lp = n.setdefault("layout_prior", {})
            lp["center_norm"] = rng.uniform(0.08, 0.92, size=2).astype(float).tolist()

    elif mode == "collapse":
        base = rng.uniform(0.35, 0.65, size=2)

        for n in nodes:
            lp = n.setdefault("layout_prior", {})
            c = base + rng.normal(0.0, 0.025, size=2)
            c = np.clip(c, 0.05, 0.95)
            lp["center_norm"] = c.astype(float).tolist()

    elif mode == "rotation_scramble":
        for n in nodes:
            lp = n.setdefault("layout_prior", {})
            rot = rng.uniform(-math.pi, math.pi)
            lp["rotation_rad"] = float(rot)
            lp["rotation_deg"] = float(rot * 180.0 / math.pi)

    elif mode == "shuffle_centers":
        perm = rng.permutation(len(nodes))
        new_centers = centers[perm]

        for i, n in enumerate(nodes):
            lp = n.setdefault("layout_prior", {})
            lp["center_norm"] = new_centers[i].astype(float).tolist()

    elif mode == "jitter_layout":
        for n in nodes:
            lp = n.setdefault("layout_prior", {})

            c = read_center_from_node(n)
            c = c + rng.normal(0.0, CENTER_JITTER_STD, size=2)
            c = np.clip(c, 0.05, 0.95)

            rot = read_angle_from_node(n)
            rot = normalize_angle_pi(rot + rng.normal(0.0, deg_to_rad(ROT_JITTER_STD_DEG)))

            length = read_length_from_node(n)
            length = float(length * np.exp(rng.normal(0.0, SCALE_JITTER_STD)))
            length = clamp(length, 0.04, 1.2)

            lp["center_norm"] = c.astype(float).tolist()
            lp["rotation_rad"] = float(rot)
            lp["rotation_deg"] = float(rot * 180.0 / math.pi)
            lp["length_norm"] = float(length)
            lp["scale_norm"] = float(length)

    else:
        for n in nodes:
            lp = n.setdefault("layout_prior", {})
            c = read_center_from_node(n)
            c = c + rng.normal(0.0, CENTER_JITTER_STD, size=2)
            c = np.clip(c, 0.05, 0.95)
            lp["center_norm"] = c.astype(float).tolist()

    return t


def load_real_graphs(layout_data):
    templates = layout_data.get("layout_templates", [])

    if MAX_REAL_GRAPHS is not None:
        templates = templates[:MAX_REAL_GRAPHS]

    graphs = []
    skip = Counter()

    for i, t in enumerate(templates):
        gid = t.get("layout_template_id", f"real_{i:05d}")
        group = t.get("glyph_uid", gid)

        g = graph_from_object(
            obj=t,
            label=1.0,
            kind="real_layout_template",
            graph_id=gid,
            group_id=group,
        )

        if g is None:
            skip["bad_real_graph"] += 1
            continue

        graphs.append(g)

    return graphs, skip


def build_corrupted_fake_graphs(layout_data, rng, max_count=None):
    templates = layout_data.get("layout_templates", [])

    if MAX_REAL_GRAPHS is not None:
        templates = templates[:MAX_REAL_GRAPHS]

    graphs = []
    skip = Counter()
    fake_idx = 0

    for i, t in enumerate(templates):
        for k in range(CORRUPTED_FAKE_PER_REAL):
            mode = CORRUPTED_FAKE_MODES[(i + k) % len(CORRUPTED_FAKE_MODES)]

            fake_t = corrupt_template(
                template=t,
                mode=mode,
                rng=rng,
                fake_idx=fake_idx,
            )

            gid = fake_t.get("layout_template_id", f"corrupted_fake_{fake_idx:05d}")
            group = t.get("glyph_uid", gid)

            g = graph_from_object(
                obj=fake_t,
                label=0.0,
                kind=f"corrupted_fake::{mode}",
                graph_id=gid,
                group_id=group,
            )

            fake_idx += 1

            if g is None:
                skip[f"bad_corrupted_fake::{mode}"] += 1
                continue

            graphs.append(g)

            if max_count is not None and len(graphs) >= max_count:
                return graphs, skip

    return graphs, skip


def load_solved_fake_graphs(solved_data):
    if not USE_SOLVED_AS_FAKE:
        return [], Counter()

    if SOLVED_FAKE_MODE == "rejected_only":
        candidates = solved_data.get("rejected_solved_glyph_candidates", [])

    elif SOLVED_FAKE_MODE == "rejected_and_rough":
        candidates = []
        all_solved = solved_data.get("solved_glyph_candidates", [])
        for c in all_solved:
            q = c.get("quality_report", {})
            if q.get("quality_status", "") in ["bad", "usable_but_rough"]:
                candidates.append(c)

    else:
        candidates = solved_data.get("solved_glyph_candidates", [])

    if MAX_SOLVED_FAKE_GRAPHS is not None:
        candidates = candidates[:MAX_SOLVED_FAKE_GRAPHS]

    graphs = []
    skip = Counter()

    for i, c in enumerate(candidates):
        gid = c.get("generated_glyph_id", f"solved_fake_{i:05d}")
        group = f"solved::{gid}"

        g = graph_from_object(
            obj=c,
            label=0.0,
            kind="solved_generated_fake",
            graph_id=gid,
            group_id=group,
        )

        if g is None:
            skip["bad_solved_fake_graph"] += 1
            continue

        graphs.append(g)

    return graphs, skip


# =========================================================
# 📦 Dataset / batching
# =========================================================
def split_by_group(graphs, test_ratio, rng):
    groups = defaultdict(list)

    for idx, g in enumerate(graphs):
        groups[g["group_id"]].append(idx)

    group_ids = list(groups.keys())
    rng.shuffle(group_ids)

    n_test = max(1, int(len(group_ids) * test_ratio))
    test_groups = set(group_ids[:n_test])

    train_idx = []
    test_idx = []

    for gid, idxs in groups.items():
        if gid in test_groups:
            test_idx.extend(idxs)
        else:
            train_idx.extend(idxs)

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return train_idx, test_idx


def batch_graphs(graphs, indices, device):
    batch = [graphs[i] for i in indices]

    max_n = max(g["num_nodes"] for g in batch)
    node_dim = batch[0]["node_feats"].shape[1]
    edge_dim = batch[0]["edge_feats"].shape[2]
    global_dim = batch[0]["global_feats"].shape[0]

    B = len(batch)

    node_feats = np.zeros((B, max_n, node_dim), dtype=np.float32)
    edge_feats = np.zeros((B, max_n, max_n, edge_dim), dtype=np.float32)
    global_feats = np.zeros((B, global_dim), dtype=np.float32)
    adj = np.zeros((B, max_n, max_n), dtype=np.float32)
    mask = np.zeros((B, max_n), dtype=np.float32)
    labels = np.zeros((B,), dtype=np.float32)

    for b, g in enumerate(batch):
        n = g["num_nodes"]

        node_feats[b, :n] = g["node_feats"]
        edge_feats[b, :n, :n] = g["edge_feats"]
        global_feats[b] = g["global_feats"]
        adj[b, :n, :n] = g["adj"]
        mask[b, :n] = 1.0
        labels[b] = g["label"]

    return {
        "node_feats": torch.tensor(node_feats, dtype=torch.float32, device=device),
        "edge_feats": torch.tensor(edge_feats, dtype=torch.float32, device=device),
        "global_feats": torch.tensor(global_feats, dtype=torch.float32, device=device),
        "adj": torch.tensor(adj, dtype=torch.float32, device=device),
        "mask": torch.tensor(mask, dtype=torch.float32, device=device),
        "labels": torch.tensor(labels, dtype=torch.float32, device=device),
    }


def iterate_batches(indices, batch_size, shuffle=True):
    idxs = list(indices)

    if shuffle:
        random.shuffle(idxs)

    for start in range(0, len(idxs), batch_size):
        yield idxs[start:start + batch_size]


def iterate_balanced_batches(graphs, indices, batch_size):
    pos = [i for i in indices if graphs[i]["label"] == 1.0]
    neg = [i for i in indices if graphs[i]["label"] == 0.0]

    if len(pos) == 0 or len(neg) == 0:
        yield from iterate_batches(indices, batch_size, shuffle=True)
        return

    random.shuffle(pos)
    random.shuffle(neg)

    half = max(1, batch_size // 2)

    steps = max(len(pos), len(neg)) // half + 1

    for s in range(steps):
        p_batch = [pos[(s * half + k) % len(pos)] for k in range(half)]
        n_batch = [neg[(s * half + k) % len(neg)] for k in range(batch_size - half)]

        batch = p_batch + n_batch
        random.shuffle(batch)

        yield batch


# =========================================================
# 🧠 Model
# =========================================================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DenseGNNDiscriminator(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        global_dim,
        hidden_dim=128,
        num_layers=4,
        dropout=0.15,
    ):
        super().__init__()

        self.node_proj = MLP(node_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.edge_proj = MLP(edge_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.global_proj = MLP(global_dim, hidden_dim, hidden_dim, dropout=dropout)

        self.msg_mlps = nn.ModuleList()
        self.upd_mlps = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.msg_mlps.append(
                MLP(hidden_dim * 3, hidden_dim, hidden_dim, dropout=dropout)
            )

            self.upd_mlps.append(
                MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout=dropout)
            )

            self.norms.append(nn.LayerNorm(hidden_dim))

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, node_feats, edge_feats, adj, mask, global_feats):
        B, N, _ = node_feats.shape

        h = self.node_proj(node_feats)
        e = self.edge_proj(edge_feats)
        g = self.global_proj(global_feats)

        h = h * mask.unsqueeze(-1)

        for msg_mlp, upd_mlp, norm in zip(self.msg_mlps, self.upd_mlps, self.norms):
            h_i = h.unsqueeze(2).expand(B, N, N, h.shape[-1])
            h_j = h.unsqueeze(1).expand(B, N, N, h.shape[-1])

            msg_input = torch.cat([h_i, h_j, e], dim=-1)
            msg = msg_mlp(msg_input)

            msg = msg * adj.unsqueeze(-1)

            deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
            agg = msg.sum(dim=2) / deg

            upd_input = torch.cat([h, agg], dim=-1)
            dh = upd_mlp(upd_input)

            h = norm(h + dh)
            h = h * mask.unsqueeze(-1)

        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_pool = (h * mask.unsqueeze(-1)).sum(dim=1) / denom

        h_masked = h.masked_fill(mask.unsqueeze(-1) < 0.5, -1e6)
        max_pool = h_masked.max(dim=1).values

        max_pool = torch.where(
            torch.isfinite(max_pool),
            max_pool,
            torch.zeros_like(max_pool),
        )

        num_nodes = mask.sum(dim=1, keepdim=True) / 10.0
        num_edges = adj.sum(dim=(1, 2), keepdim=False).unsqueeze(-1) / 20.0
        density = num_edges / (num_nodes * num_nodes * 10.0 + 1e-6)

        trivial_global = torch.cat(
            [
                num_nodes,
                num_edges,
                density,
                torch.ones_like(num_nodes),
            ],
            dim=-1,
        )

        graph_emb = torch.cat(
            [
                mean_pool,
                max_pool,
                g,
                trivial_global,
            ],
            dim=-1,
        )

        logits = self.readout(graph_emb).squeeze(-1)

        return logits


# =========================================================
# 📈 Metrics
# =========================================================
def binary_auc_score(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.float32)
    y_score = np.asarray(y_score).astype(np.float32)

    pos = y_true == 1
    neg = y_true == 0

    n_pos = int(pos.sum())
    n_neg = int(neg.sum())

    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float32)

    sum_ranks_pos = ranks[pos].sum()

    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    return float(auc)


@torch.no_grad()
def evaluate_model(model, graphs, indices, device, batch_size):
    model.eval()

    all_logits = []
    all_labels = []

    total_loss = 0.0
    total_count = 0

    criterion = nn.BCEWithLogitsLoss()

    for batch_idx in iterate_batches(indices, batch_size, shuffle=False):
        batch = batch_graphs(graphs, batch_idx, device)

        logits = model(
            batch["node_feats"],
            batch["edge_feats"],
            batch["adj"],
            batch["mask"],
            batch["global_feats"],
        )

        labels = batch["labels"]

        loss = criterion(logits, labels)

        total_loss += float(loss.item()) * len(batch_idx)
        total_count += len(batch_idx)

        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)

    logits_clip = np.clip(logits_np, -50.0, 50.0)
    probs = 1.0 / (1.0 + np.exp(-logits_clip))
    preds = (probs >= 0.5).astype(np.float32)

    acc = float((preds == labels_np).mean())
    auc = binary_auc_score(labels_np, probs)

    return {
        "loss": total_loss / max(1, total_count),
        "acc": acc,
        "auc": auc,
        "probs": probs,
        "labels": labels_np,
        "logits": logits_np,
    }


def score_all_graphs(model, graphs, device, batch_size, train_idx_set, test_idx_set):
    model.eval()

    result = []

    with torch.no_grad():
        all_indices = list(range(len(graphs)))

        for batch_idx in iterate_batches(all_indices, batch_size, shuffle=False):
            batch = batch_graphs(graphs, batch_idx, device)

            logits = model(
                batch["node_feats"],
                batch["edge_feats"],
                batch["adj"],
                batch["mask"],
                batch["global_feats"],
            )

            probs = torch.sigmoid(logits).detach().cpu().numpy()

            for local_i, gidx in enumerate(batch_idx):
                g = graphs[gidx]

                if gidx in train_idx_set:
                    split = "train"
                elif gidx in test_idx_set:
                    split = "test"
                else:
                    split = "unknown"

                result.append({
                    "graph_index": int(gidx),
                    "graph_id": g["graph_id"],
                    "group_id": g["group_id"],
                    "kind": g["kind"],
                    "label": int(g["label"]),
                    "split": split,
                    "realness_score": float(probs[local_i]),
                    "num_nodes": int(g["num_nodes"]),
                    "num_edges": int(g["num_edges"]),
                    "source_file": g.get("source_file", ""),
                    "glyph_uid": g.get("glyph_uid", ""),
                })

    return result


# =========================================================
# 🧪 数据构建
# =========================================================
def build_dataset():
    rng = np.random.default_rng(RANDOM_SEED)

    print("\n" + "=" * 80)
    print("📦 Building GNN Layout Discriminator Dataset V2")
    print("=" * 80)

    layout_data = load_json(LAYOUT_TEMPLATE_FILE)

    real_graphs, real_skip = load_real_graphs(layout_data)

    print("[Real]")
    print(f"  real_graphs: {len(real_graphs)}")
    print(f"  real_skip: {dict(real_skip)}")

    max_corrupted = int(len(real_graphs) * CORRUPTED_FAKE_PER_REAL)

    corrupted_graphs, corrupted_skip = build_corrupted_fake_graphs(
        layout_data=layout_data,
        rng=rng,
        max_count=max_corrupted,
    )

    print("\n[Corrupted Fake]")
    print(f"  corrupted_graphs: {len(corrupted_graphs)}")
    print(f"  corrupted_fake_modes: {CORRUPTED_FAKE_MODES}")
    print(f"  corrupted_skip: {dict(corrupted_skip)}")

    solved_graphs = []
    solved_skip = Counter()

    if USE_SOLVED_AS_FAKE and os.path.exists(SOLVED_FILE):
        solved_data = load_json(SOLVED_FILE)
        solved_graphs, solved_skip = load_solved_fake_graphs(solved_data)

    print("\n[Solved Generated Fake]")
    print(f"  use_solved_as_fake: {USE_SOLVED_AS_FAKE}")
    print(f"  solved_file_exists: {os.path.exists(SOLVED_FILE)}")
    print(f"  solved_fake_graphs: {len(solved_graphs)}")
    print(f"  solved_skip: {dict(solved_skip)}")

    fake_graphs = corrupted_graphs + solved_graphs

    max_fake = int(len(real_graphs) * MAX_FAKE_RATIO)

    if len(fake_graphs) > max_fake:
        rng.shuffle(fake_graphs)
        fake_graphs = fake_graphs[:max_fake]

    graphs = real_graphs + fake_graphs
    rng.shuffle(graphs)

    kind_hist = Counter(g["kind"] for g in graphs)
    label_hist = Counter(int(g["label"]) for g in graphs)
    node_hist = Counter(g["num_nodes"] for g in graphs)
    edge_hist = Counter(g["num_edges"] for g in graphs)

    print("\n[Dataset Summary]")
    print(f"  total_graphs: {len(graphs)}")
    print(f"  label_hist: {dict(label_hist)}")

    print("\n[Kind Hist]")
    for k, v in kind_hist.items():
        print(f"  {k}: {v}")

    print("\n[Node Count Hist]")
    for k, v in sorted(node_hist.items()):
        print(f"  {k}: {v}")

    print("\n[Edge Count Hist]")
    for k, v in sorted(edge_hist.items()):
        print(f"  {k}: {v}")

    if len(graphs) == 0:
        raise RuntimeError("没有可训练图。")

    node_dim = graphs[0]["node_feats"].shape[1]
    edge_dim = graphs[0]["edge_feats"].shape[2]
    global_dim = graphs[0]["global_feats"].shape[0]

    print("\n[Feature Dims]")
    print(f"  node_dim: {node_dim}")
    print(f"  edge_dim: {edge_dim}")
    print(f"  global_dim: {global_dim}")

    return graphs


# =========================================================
# 📊 Score summary / plotting
# =========================================================
def summarize_scores(score_items):
    by_kind = defaultdict(list)
    by_label = defaultdict(list)
    by_split_label = defaultdict(list)

    for item in score_items:
        by_kind[item["kind"]].append(float(item["realness_score"]))
        by_label[str(item["label"])].append(float(item["realness_score"]))
        by_split_label[f"{item['split']}::label={item['label']}"].append(float(item["realness_score"]))

    def stats(vals):
        vals = np.asarray(vals, dtype=np.float32)

        if len(vals) == 0:
            return {}

        return {
            "count": int(len(vals)),
            "mean": round(float(np.mean(vals)), 6),
            "std": round(float(np.std(vals)), 6),
            "min": round(float(np.min(vals)), 6),
            "p10": round(float(np.percentile(vals, 10)), 6),
            "p50": round(float(np.percentile(vals, 50)), 6),
            "p90": round(float(np.percentile(vals, 90)), 6),
            "max": round(float(np.max(vals)), 6),
        }

    return {
        "by_kind": {k: stats(v) for k, v in by_kind.items()},
        "by_label": {k: stats(v) for k, v in by_label.items()},
        "by_split_label": {k: stats(v) for k, v in by_split_label.items()},
    }


def print_score_summary(summary):
    print("\n" + "=" * 80)
    print("📊 Score Summary")
    print("=" * 80)

    print("[By Label]")
    for k, v in summary.get("by_label", {}).items():
        print(f"  label={k}: {v}")

    print("\n[By Split Label]")
    for k, v in summary.get("by_split_label", {}).items():
        print(f"  {k}: {v}")

    print("\n[By Kind]")
    for k, v in summary.get("by_kind", {}).items():
        print(f"  {k}: {v}")


def plot_training_curves(history, out_path):
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["test_loss"], label="test")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["test_acc"], label="test")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["train_auc"], label="train")
    axes[2].plot(epochs, history["test_auc"], label="test")
    axes[2].set_title("AUC")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 🚂 Train
# =========================================================
def train():
    set_seed(RANDOM_SEED)

    device = choose_device()

    print("\n" + "=" * 80)
    print("🚀 GNN Layout Discriminator V2")
    print("=" * 80)
    print(f"  layout_template_file: {LAYOUT_TEMPLATE_FILE}")
    print(f"  solved_file:          {SOLVED_FILE}")
    print(f"  device_mode:          {DEVICE_MODE}")
    print(f"  selected device:      {device}")
    print(f"  epochs:               {EPOCHS}")
    print(f"  batch_size:           {BATCH_SIZE}")
    print(f"  lr:                   {LR}")
    print(f"  use_balanced_batch:   {USE_BALANCED_BATCH}")
    print(f"  use_solved_as_fake:   {USE_SOLVED_AS_FAKE}")
    print("=" * 80)

    graphs = build_dataset()

    rng = random.Random(RANDOM_SEED)
    train_idx, test_idx = split_by_group(graphs, TEST_GROUP_RATIO, rng)

    print("\n" + "=" * 80)
    print("✂️ Group Split")
    print("=" * 80)
    print(f"  train_graphs: {len(train_idx)}")
    print(f"  test_graphs:  {len(test_idx)}")

    train_label_hist = Counter(int(graphs[i]["label"]) for i in train_idx)
    test_label_hist = Counter(int(graphs[i]["label"]) for i in test_idx)

    print(f"  train_label_hist: {dict(train_label_hist)}")
    print(f"  test_label_hist:  {dict(test_label_hist)}")

    node_dim = graphs[0]["node_feats"].shape[1]
    edge_dim = graphs[0]["edge_feats"].shape[2]
    global_dim = graphs[0]["global_feats"].shape[0]

    model = DenseGNNDiscriminator(
        node_dim=node_dim,
        edge_dim=edge_dim,
        global_dim=global_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_GNN_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    # balanced batch 下不需要 pos_weight，避免常数分类器震荡
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_auc": [],
        "test_loss": [],
        "test_acc": [],
        "test_auc": [],
    }

    best_test_auc = -1.0
    best_state = None

    print("\n" + "=" * 80)
    print("🚂 Training")
    print("=" * 80)

    for epoch in range(1, EPOCHS + 1):
        model.train()

        if USE_BALANCED_BATCH:
            batch_iter = iterate_balanced_batches(graphs, train_idx, BATCH_SIZE)
        else:
            batch_iter = iterate_batches(train_idx, BATCH_SIZE, shuffle=True)

        for batch_idx in batch_iter:
            batch = batch_graphs(graphs, batch_idx, device)

            logits = model(
                batch["node_feats"],
                batch["edge_feats"],
                batch["adj"],
                batch["mask"],
                batch["global_feats"],
            )

            labels = batch["labels"]

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

        train_eval = evaluate_model(model, graphs, train_idx, device, BATCH_SIZE)
        test_eval = evaluate_model(model, graphs, test_idx, device, BATCH_SIZE)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_eval["loss"])
        history["train_acc"].append(train_eval["acc"])
        history["train_auc"].append(train_eval["auc"])
        history["test_loss"].append(test_eval["loss"])
        history["test_acc"].append(test_eval["acc"])
        history["test_auc"].append(test_eval["auc"])

        if test_eval["auc"] > best_test_auc:
            best_test_auc = test_eval["auc"]
            best_state = {
                "model_state_dict": deepcopy(model.state_dict()),
                "epoch": epoch,
                "test_auc": test_eval["auc"],
                "test_acc": test_eval["acc"],
            }

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_eval['loss']:.4f} "
                f"train_acc={train_eval['acc']:.4f} "
                f"train_auc={train_eval['auc']:.4f} | "
                f"test_loss={test_eval['loss']:.4f} "
                f"test_acc={test_eval['acc']:.4f} "
                f"test_auc={test_eval['auc']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    final_train = evaluate_model(model, graphs, train_idx, device, BATCH_SIZE)
    final_test = evaluate_model(model, graphs, test_idx, device, BATCH_SIZE)

    print("\n" + "=" * 80)
    print("📊 Final Evaluation")
    print("=" * 80)
    print(f"  best_test_auc:   {best_test_auc:.4f}")
    print(f"  final_train_auc: {final_train['auc']:.4f}")
    print(f"  final_train_acc: {final_train['acc']:.4f}")
    print(f"  final_test_auc:  {final_test['auc']:.4f}")
    print(f"  final_test_acc:  {final_test['acc']:.4f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "global_dim": global_dim,
        "hidden_dim": HIDDEN_DIM,
        "num_gnn_layers": NUM_GNN_LAYERS,
        "dropout": DROPOUT,
        "best_state": {
            "epoch": None if best_state is None else best_state["epoch"],
            "test_auc": None if best_state is None else best_state["test_auc"],
            "test_acc": None if best_state is None else best_state["test_acc"],
        },
        "config": {
            "shape_code_dim": SHAPE_CODE_DIM,
            "width_token_dim": WIDTH_TOKEN_DIM,
            "jtype_dim": JTYPE_DIM,
            "canvas_size": CANVAS_SIZE,
            "corrupted_fake_per_real": CORRUPTED_FAKE_PER_REAL,
            "corrupted_fake_modes": CORRUPTED_FAKE_MODES,
            "use_solved_as_fake": USE_SOLVED_AS_FAKE,
            "solved_fake_mode": SOLVED_FAKE_MODE,
            "use_balanced_batch": USE_BALANCED_BATCH,
        },
    }

    torch.save(checkpoint, OUTPUT_MODEL_FILE)

    print("\n" + "=" * 80)
    print("💾 Saved Model")
    print("=" * 80)
    print(f"  {OUTPUT_MODEL_FILE}")

    if SAVE_ALL_SCORES:
        train_set = set(train_idx)
        test_set = set(test_idx)

        score_items = score_all_graphs(
            model=model,
            graphs=graphs,
            device=device,
            batch_size=BATCH_SIZE,
            train_idx_set=train_set,
            test_idx_set=test_set,
        )

        score_summary = summarize_scores(score_items)

        save_json(
            {
                "schema_version": "gnn_layout_discriminator_scores_v2",
                "model_file": OUTPUT_MODEL_FILE,
                "score_summary": score_summary,
                "scores": score_items,
            },
            OUTPUT_SCORE_FILE,
        )

        print("\n" + "=" * 80)
        print("💾 Saved Scores")
        print("=" * 80)
        print(f"  {OUTPUT_SCORE_FILE}")

        print_score_summary(score_summary)

    plot_training_curves(history, OUTPUT_CURVE_FILE)

    print("\n" + "=" * 80)
    print("📈 Saved Training Curves")
    print("=" * 80)
    print(f"  {OUTPUT_CURVE_FILE}")

    print("\n📌 如何判断这次实验：")
    print("  1. 如果 test_auc > 0.80，说明 global geometry + GNN 可以区分真实 layout 和强 fake")
    print("  2. 如果 real score 明显高于 fake score，说明 discriminator 不再塌缩")
    print("  3. 如果 test_auc 接近 1.0，说明强 fake 太简单，下一步要加入 solved hard fake")
    print("  4. 如果还是 0.5 左右，说明 layout feature 读取或 real/fake 构造仍有问题")
    print("  5. 这版成功后，再把 USE_SOLVED_AS_FAKE 改成 True，并使用 rejected_only")


# =========================================================
# 🎬 main
# =========================================================
if __name__ == "__main__":
    train()