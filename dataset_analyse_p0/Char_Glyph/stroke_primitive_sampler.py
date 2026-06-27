import os
import json
import math
import random
import shutil
from copy import deepcopy
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# ⚙️ 路径配置
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GRAPH_INPUT_FILE = os.path.join(SCRIPT_DIR, "layout_aware_graph_samples.json")
CORPUS_FILE = os.path.join(SCRIPT_DIR, "alien_glyph_pcg_corpus.json")

OUTPUT_FILE = os.path.join(SCRIPT_DIR, "glyph_candidates_with_primitives.json")
PREVIEW_DIR = os.path.join(SCRIPT_DIR, "glyph_candidates_with_primitives_previews")

RANDOM_SEED = 239351275

NUM_CANDIDATES = None
CLEAR_PREVIEW_DIR = True
PREVIEW_COUNT = 80

CANVAS_SIZE = 400.0


# =========================================================
# 🎛️ primitive sampler 配置
# =========================================================
SAMPLER_CONFIG = {
    # 是否优先使用 layout graph 里保留的 template_shape_code
    # 当前建议 False，让 primitive 重新采样
    "use_template_shape_hint": False,
    "template_shape_keep_prob": 0.10,

    # 是否根据 node role 调整 shape 权重
    "use_role_aware_sampling": True,

    # 是否根据 edge type 调整 shape 权重
    "use_edge_type_aware_sampling": True,

    # shape 采样温度
    # 越大越随机，越小越偏高频
    "shape_temperature": 0.85,

    # variant 采样
    "variant_count": 4,

    # width 暂时不是核心，因此用保守策略
    "width_sampling_mode": "mostly_thin",
    "width_token_probs": {
        0: 0.86,
        1: 0.13,
        2: 0.01,
    },

    # 结构直线 primitive，比如 shape_code=20
    "structural_line_shape_codes": [20],

    # 提高直线 primitive 的采样概率
    "prefer_straight_primitives": True,
    "straightness_threshold": 0.94,
    "straightness_bonus_strength": 2.5,
    "structural_line_shape_boost": 4.0,

    # 避免一个 glyph 里全是同一种 primitive
    "avoid_single_shape_collapse": True,
    "max_same_shape_ratio_per_glyph": 0.75,

    # 如果候选里某个 shape 超过比例，则重采样几次
    "resample_attempts_per_node": 8,

    # 是否把 primitive local polyline 写进 primitive_ref
    "store_polyline_in_node": True,

    # 如果 primitive library 里找不到 polyline，用直线兜底
    "fallback_to_line_primitive": True,

    # 是否强制每个 node 都写 solver_priors
    "write_solver_priors": True,

    # 是否复制 layout_prior 到 solver_priors
    "copy_layout_prior_to_solver_priors": True,
}


# =========================================================
# 🧮 基础工具
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def reset_dir(path):
    if CLEAR_PREVIEW_DIR and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def load_json(path, required=True):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"找不到文件: {path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=json_default)


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


def normalize_angle_pi(rad):
    a = float(rad)

    while a <= -math.pi:
        a += 2.0 * math.pi
    while a > math.pi:
        a -= 2.0 * math.pi

    return a


def deg_to_rad(deg):
    return float(deg) * math.pi / 180.0


def rad_to_deg(rad):
    return float(rad) * 180.0 / math.pi


def safe_filename(s):
    s = str(s)
    for ch in ["/", "\\", ":", "*", "?", "\"", "<", ">", "|", " ", "\n", "\t"]:
        s = s.replace(ch, "_")
    return s[:160]


def weighted_choice(items, weights):
    if len(items) == 0:
        raise ValueError("weighted_choice got empty items")

    total = float(sum(weights))

    if total <= 0:
        return random.choice(items)

    r = random.random() * total
    acc = 0.0

    for item, w in zip(items, weights):
        acc += float(w)
        if acc >= r:
            return item

    return items[-1]


def sample_from_prob_dict(prob_dict):
    items = list(prob_dict.keys())
    probs = [float(prob_dict[k]) for k in items]
    return weighted_choice(items, probs)


def to_np2(x, default=None):
    if x is None:
        return default

    arr = np.asarray(x, dtype=np.float32)

    if arr.ndim == 1 and arr.shape[0] >= 2:
        return arr[:2]

    return default


# =========================================================
# 📦 输入 graph samples
# =========================================================
def get_graph_samples(data):
    if isinstance(data, list):
        return data

    keys = [
        "sampled_topologies",
        "layout_aware_graph_samples",
        "graph_grammar_samples",
        "samples",
        "generated_samples",
        "candidates",
    ]

    for k in keys:
        if k in data and isinstance(data[k], list):
            return data[k]

    for k, v in data.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            print(f"[WARN] 未识别标准 graph sample key，使用字段: {k}")
            return v

    return []


def get_sample_id(sample, idx):
    for k in [
        "grammar_sample_id",
        "sample_id",
        "generated_glyph_id",
        "candidate_id",
        "layout_template_id",
    ]:
        if isinstance(sample, dict) and sample.get(k, ""):
            return str(sample[k])

    return f"graph_sample_{idx:05d}"


def get_nodes(sample):
    if isinstance(sample, dict) and "nodes" in sample and isinstance(sample["nodes"], list):
        return sample["nodes"]
    return []


def get_edges(sample):
    if not isinstance(sample, dict):
        return []

    topo = sample.get("topology", {})

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

    if "edges" in sample and isinstance(sample["edges"], list):
        return sample["edges"]

    return []


def node_id_of(node, fallback):
    return safe_int(node.get("node_id", node.get("source_node_id", fallback)), fallback)


# =========================================================
# 📚 primitive library
# =========================================================
def load_primitive_library(corpus_data):
    if not isinstance(corpus_data, dict):
        return {}

    lib = corpus_data.get("stroke_primitive_library", None)

    if lib is None:
        for k in [
            "primitive_library",
            "stroke_primitives",
            "primitive_clusters",
            "clustered_stroke_primitives",
        ]:
            if k in corpus_data:
                lib = corpus_data[k]
                break

    if lib is None:
        return {}

    out = {}

    if isinstance(lib, dict):
        for k, v in lib.items():
            if not isinstance(v, dict):
                continue

            vv = deepcopy(v)

            if "shape_code" not in vv:
                vv["shape_code"] = safe_int(k, k)

            out[str(vv["shape_code"])] = vv

        return out

    if isinstance(lib, list):
        for i, item in enumerate(lib):
            if not isinstance(item, dict):
                continue

            vv = deepcopy(item)

            shape_code = vv.get(
                "shape_code",
                vv.get("shape_token", vv.get("cluster_id", vv.get("id", i)))
            )

            vv["shape_code"] = safe_int(shape_code, shape_code)

            out[str(vv["shape_code"])] = vv

        return out

    return {}


def primitive_entry_to_polyline(entry):
    if not isinstance(entry, dict):
        return None

    poly_keys = [
        "primitive_polyline_local_norm",
        "prototype_polyline_local_norm",
        "prototype_polyline_norm",
        "prototype_norm_polyline",
        "polyline_local_norm",
        "polyline_norm",
        "prototype_polyline",
        "polyline",
        "points",
    ]

    for k in poly_keys:
        if k in entry:
            arr = np.asarray(entry[k], dtype=np.float32)

            if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) >= 2:
                return arr[:, :2]

    y_keys = [
        "prototype_y_function_norm",
        "prototype_norm_y",
        "norm_y",
        "y_function",
        "y",
    ]

    for k in y_keys:
        if k in entry:
            y = np.asarray(entry[k], dtype=np.float32).reshape(-1)

            if len(y) >= 2:
                x = np.linspace(-0.5, 0.5, len(y), dtype=np.float32)
                return np.stack([x, y], axis=1)

    for parent in ["prototype", "primitive", "data"]:
        if parent in entry and isinstance(entry[parent], dict):
            p = primitive_entry_to_polyline(entry[parent])
            if p is not None:
                return p

    return None


def normalize_local_polyline(polyline):
    arr = np.asarray(polyline, dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) < 2:
        return np.asarray([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32)

    arr = arr[:, :2].copy()

    finite = np.all(np.isfinite(arr), axis=1)
    arr = arr[finite]

    if len(arr) < 2:
        return np.asarray([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32)

    x = arr[:, 0]
    y = arr[:, 1]

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    xr = xmax - xmin

    if xr < 1e-6:
        return np.asarray([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32)

    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)

    arr[:, 0] = (arr[:, 0] - xmid) / xr
    arr[:, 1] = (arr[:, 1] - ymid) / xr

    arr[:, 0] = np.clip(arr[:, 0], -1.0, 1.0)
    arr[:, 1] = np.clip(arr[:, 1], -1.0, 1.0)

    return arr.astype(np.float32)


def build_primitive_entries(primitive_library):
    entries = []

    for k, entry in primitive_library.items():
        if not isinstance(entry, dict):
            continue

        e = deepcopy(entry)

        shape_code = e.get("shape_code", e.get("shape_token", k))
        shape_code = safe_int(shape_code, safe_int(k, 0))

        e["shape_code"] = shape_code
        e["shape_token"] = shape_code

        poly = primitive_entry_to_polyline(e)

        if poly is None:
            if SAMPLER_CONFIG["fallback_to_line_primitive"]:
                poly = np.asarray([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32)
            else:
                continue

        poly = normalize_local_polyline(poly)

        e["primitive_polyline_local_norm"] = poly.astype(float).tolist()

        # count / frequency 兜底
        count = e.get(
            "count",
            e.get("sample_count", e.get("cluster_size", e.get("freq", 1)))
        )

        e["_sample_weight_base"] = max(1.0, safe_float(count, 1.0))

        entries.append(e)

    entries = sorted(entries, key=lambda x: safe_int(x.get("shape_code", 0), 0))

    return entries


# =========================================================
# 🧠 role-aware 权重
# =========================================================
def get_node_degree_and_edge_types(node, edges):
    nid = node_id_of(node, 0)

    degree = 0
    edge_types = []

    for e in edges:
        u = safe_int(e.get("u", -1), -1)
        v = safe_int(e.get("v", -1), -1)

        if u == nid or v == nid:
            degree += 1
            edge_types.append(str(e.get("j_type", "UNKNOWN")))

    return degree, edge_types


def shape_family_score(entry):
    """
    只基于 primitive polyline 的粗略形状特征。
    返回:
      straightness 越高越直
      curvature 越高越弯
    """
    poly = np.asarray(entry.get("primitive_polyline_local_norm", []), dtype=np.float32)

    if poly.ndim != 2 or poly.shape[1] < 2 or len(poly) < 2:
        return {
            "straightness": 1.0,
            "curvature": 0.0,
        }

    start = poly[0]
    end = poly[-1]
    chord = float(np.linalg.norm(end - start))

    seg = poly[1:] - poly[:-1]
    arc = float(np.sum(np.linalg.norm(seg, axis=1)))

    if arc < 1e-8:
        return {
            "straightness": 1.0,
            "curvature": 0.0,
        }

    straightness = clamp(chord / arc, 0.0, 1.0)
    curvature = clamp(1.0 - straightness, 0.0, 1.0)

    return {
        "straightness": float(straightness),
        "curvature": float(curvature),
    }


def compute_entry_weight_for_node(entry, node, edges, current_shape_counter):
    base = safe_float(entry.get("_sample_weight_base", 1.0), 1.0)

    # 温度缩放
    temp = max(0.05, float(SAMPLER_CONFIG["shape_temperature"]))
    w = base ** (1.0 / temp)

    shape = safe_int(entry.get("shape_code", 0), 0)
    structural_lines = set(int(x) for x in SAMPLER_CONFIG["structural_line_shape_codes"])

    degree, edge_types = get_node_degree_and_edge_types(node, edges)
    role = node.get("grammar_role", {})

    if not isinstance(role, dict):
        role = {}

    incident_types = set(edge_types)
    role_types = set(role.get("incident_edge_types", []))

    incident_types |= role_types

    fam = shape_family_score(entry)
    straightness = fam["straightness"]
    curvature = fam["curvature"]

    # =========================================================
    # boost straight / structural primitives
    # =========================================================
    if SAMPLER_CONFIG.get("prefer_straight_primitives", False):
        straight_threshold = SAMPLER_CONFIG.get("straightness_threshold", 0.94)
        bonus_strength = SAMPLER_CONFIG.get("straightness_bonus_strength", 2.5)

        # 所有近似直线 primitive 都加权
        if straightness >= straight_threshold:
            w *= 1.0 + bonus_strength

        # 越直，权重越高
        w *= 1.0 + bonus_strength * (straightness ** 2)

        # 指定 structural_line_shape_codes 额外加强
        structural_lines = set(
            int(x) for x in SAMPLER_CONFIG.get("structural_line_shape_codes", [])
        )

        if shape in structural_lines:
            w *= SAMPLER_CONFIG.get("structural_line_shape_boost", 4.0)

    if SAMPLER_CONFIG.get("use_role_aware_sampling", True):
        # 高 degree / hub 更偏直线或稳定 primitive
        if degree >= 3:
            w *= 1.0 + 0.50 * straightness

        # 普通端点 stroke 可以更允许曲线
        if degree <= 1:
            w *= 1.0 + 0.35 * curvature

    if SAMPLER_CONFIG.get("use_edge_type_aware_sampling", True):
        if "T" in incident_types:
            # T junction host/guest 用太弯的 primitive 容易接不上，略偏直
            w *= 1.0 + 0.35 * straightness

        if "X" in incident_types:
            # X 交叉也略偏直
            w *= 1.0 + 0.45 * straightness

        if "E2E" in incident_types and degree <= 2:
            # 链式可以有一点弯曲
            w *= 1.0 + 0.20 * curvature

    # 避免单 glyph 内一种 shape 过多
    if SAMPLER_CONFIG.get("avoid_single_shape_collapse", True):
        already = current_shape_counter.get(shape, 0)
        if already >= 2:
            w *= 0.55 ** (already - 1)

    # 如果 template_shape_code 给了 hint，可轻微加权
    if SAMPLER_CONFIG.get("use_template_shape_hint", False):
        t_shape = node.get("template_shape_code", None)
        if t_shape is not None and safe_int(t_shape, -999) == shape:
            w *= 1.8

    return max(1e-8, float(w))


def sample_shape_entry_for_node(node, edges, primitive_entries, current_shape_counter):
    if len(primitive_entries) == 0:
        raise RuntimeError("primitive_entries 为空，无法采样 primitive。")

    # 小概率直接保留 template shape
    if SAMPLER_CONFIG.get("use_template_shape_hint", False):
        t_shape = node.get("template_shape_code", None)
        if t_shape is not None and random.random() < SAMPLER_CONFIG["template_shape_keep_prob"]:
            t_shape = safe_int(t_shape, -1)
            for e in primitive_entries:
                if safe_int(e.get("shape_code", -2), -2) == t_shape:
                    return e

    weights = [
        compute_entry_weight_for_node(e, node, edges, current_shape_counter)
        for e in primitive_entries
    ]

    return weighted_choice(primitive_entries, weights)


def sample_variant_id():
    vc = int(SAMPLER_CONFIG["variant_count"])
    if vc <= 1:
        return 0
    return random.randint(0, vc - 1)


def sample_width_token():
    mode = SAMPLER_CONFIG.get("width_sampling_mode", "mostly_thin")

    if mode == "fixed_zero":
        return 0

    prob_dict = SAMPLER_CONFIG.get("width_token_probs", {0: 1.0})
    return int(sample_from_prob_dict(prob_dict))


# =========================================================
# 🧱 layout_prior / solver_priors
# =========================================================
def read_or_build_layout_prior(src_node):
    lp = src_node.get("layout_prior", {})

    if isinstance(lp, dict) and "center_norm" in lp:
        out = deepcopy(lp)
    else:
        out = {}

        # fallback
        center = None

        for k in ["center_norm", "center"]:
            if k in src_node:
                center = to_np2(src_node[k])
                break

        if center is None:
            center = np.asarray([0.5, 0.5], dtype=np.float32)

        rotation = safe_float(src_node.get("rotation_rad", 0.0), 0.0)

        if "rotation_deg" in src_node:
            rotation = deg_to_rad(safe_float(src_node["rotation_deg"], 0.0))

        length = safe_float(src_node.get("length_norm", src_node.get("scale_norm", 0.25)), 0.25)

        out["center_norm"] = center.astype(float).tolist()
        out["rotation_rad"] = float(normalize_angle_pi(rotation))
        out["rotation_deg"] = float(rad_to_deg(rotation))
        out["length_norm"] = float(clamp(length, 0.03, 1.5))
        out["scale_norm"] = float(clamp(length, 0.03, 1.5))

    center = to_np2(out.get("center_norm", [0.5, 0.5]), np.asarray([0.5, 0.5], dtype=np.float32))
    rotation = normalize_angle_pi(safe_float(out.get("rotation_rad", 0.0), 0.0))

    if "rotation_deg" in out and "rotation_rad" not in out:
        rotation = normalize_angle_pi(deg_to_rad(safe_float(out["rotation_deg"], 0.0)))

    length = safe_float(out.get("length_norm", out.get("scale_norm", 0.25)), 0.25)
    length = clamp(length, 0.03, 1.5)

    d = np.asarray([math.cos(rotation), math.sin(rotation)], dtype=np.float32) * length * 0.5

    out["center_norm"] = center.astype(float).tolist()
    out["rotation_rad"] = float(rotation)
    out["rotation_deg"] = float(rad_to_deg(rotation))
    out["length_norm"] = float(length)
    out["scale_norm"] = float(safe_float(out.get("scale_norm", length), length))
    out["p0_norm"] = (center - d).astype(float).tolist()
    out["p3_norm"] = (center + d).astype(float).tolist()

    return out


def build_solver_priors(src_node, layout_prior):
    old = src_node.get("solver_priors", {})

    if not isinstance(old, dict):
        old = {}

    sp = deepcopy(old)

    if SAMPLER_CONFIG.get("copy_layout_prior_to_solver_priors", True):
        sp["center_norm"] = layout_prior.get("center_norm", None)
        sp["rotation_rad"] = layout_prior.get("rotation_rad", None)
        sp["length_prior_norm"] = layout_prior.get(
            "length_norm",
            layout_prior.get("scale_norm", sp.get("length_prior_norm", None))
        )
        sp["scale_init_norm"] = layout_prior.get(
            "scale_norm",
            layout_prior.get("length_norm", sp.get("scale_init_norm", None))
        )
        sp["rotation_init_policy"] = "layout_prior"

    return sp


# =========================================================
# 🧬 build candidate
# =========================================================
def build_primitive_ref(entry, variant_id):
    shape_code = safe_int(entry.get("shape_code", 0), 0)

    ref = {
        "shape_code": shape_code,
        "shape_token": shape_code,
        "variant_id": int(variant_id),
        "source": "stroke_primitive_library",
    }

    for k in [
        "cluster_id",
        "count",
        "sample_count",
        "cluster_size",
        "prototype_id",
        "primitive_id",
    ]:
        if k in entry:
            ref[k] = entry[k]

    if SAMPLER_CONFIG.get("store_polyline_in_node", True):
        ref["primitive_polyline_local_norm"] = deepcopy(entry["primitive_polyline_local_norm"])

    return ref


def build_node_with_primitive(src_node, edges, primitive_entries, current_shape_counter, fallback_idx):
    new_node = deepcopy(src_node)

    node_id = node_id_of(src_node, fallback_idx)
    new_node["node_id"] = int(node_id)

    # layout prior 必须保留
    layout_prior = read_or_build_layout_prior(src_node)
    new_node["layout_prior"] = layout_prior

    # solver priors
    if SAMPLER_CONFIG.get("write_solver_priors", True):
        new_node["solver_priors"] = build_solver_priors(src_node, layout_prior)

    # 采样 primitive
    chosen = None

    for _ in range(SAMPLER_CONFIG.get("resample_attempts_per_node", 8)):
        cand = sample_shape_entry_for_node(
            node=src_node,
            edges=edges,
            primitive_entries=primitive_entries,
            current_shape_counter=current_shape_counter,
        )

        shape = safe_int(cand.get("shape_code", 0), 0)

        # glyph 内同一 shape 不要太极端
        total_so_far = max(1, sum(current_shape_counter.values()) + 1)
        ratio_after = (current_shape_counter.get(shape, 0) + 1) / total_so_far

        if ratio_after <= SAMPLER_CONFIG["max_same_shape_ratio_per_glyph"]:
            chosen = cand
            break

        chosen = cand

    shape_code = safe_int(chosen.get("shape_code", 0), 0)
    variant_id = sample_variant_id()
    width_token = sample_width_token()

    current_shape_counter[shape_code] += 1

    new_node["shape_code"] = int(shape_code)
    new_node["shape_token"] = int(shape_code)
    new_node["variant_id"] = int(variant_id)
    new_node["width_token"] = int(width_token)

    new_node["primitive_ref"] = build_primitive_ref(chosen, variant_id)

    new_node["primitive_assignment"] = {
        "assignment_source": "layout_aware_primitive_sampler",
        "shape_sampling_mode": "role_edge_aware",
        "shape_temperature": SAMPLER_CONFIG["shape_temperature"],
        "variant_id": int(variant_id),
        "width_token": int(width_token),
    }

    return new_node


def normalize_edges(edges):
    clean = []

    for e in edges:
        if not isinstance(e, dict):
            continue

        ee = deepcopy(e)

        if "u" not in ee:
            for k in ["src", "source", "a", "node_u"]:
                if k in ee:
                    ee["u"] = ee[k]
                    break

        if "v" not in ee:
            for k in ["dst", "target", "b", "node_v"]:
                if k in ee:
                    ee["v"] = ee[k]
                    break

        if "u" not in ee or "v" not in ee:
            continue

        ee["u"] = safe_int(ee["u"], -1)
        ee["v"] = safe_int(ee["v"], -1)

        if ee["u"] < 0 or ee["v"] < 0:
            continue

        if "j_type" not in ee:
            idx = safe_int(ee.get("j_type_idx", 0), 0)
            ee["j_type"] = {
                1: "E2E",
                2: "X",
                3: "T",
            }.get(idx, "UNKNOWN")

        if "t_u" not in ee:
            ee["t_u"] = 0.0

        if "t_v" not in ee:
            ee["t_v"] = 0.0

        clean.append(ee)

    return clean


def build_topology_summary(nodes, edges):
    degree = Counter()
    edge_type_counts = Counter()

    for e in edges:
        u = safe_int(e.get("u", -1), -1)
        v = safe_int(e.get("v", -1), -1)

        if u >= 0 and v >= 0:
            degree[u] += 1
            degree[v] += 1

        edge_type_counts[str(e.get("j_type", "UNKNOWN"))] += 1

    n = len(nodes)
    m = len(edges)

    ids = [node_id_of(node, i) for i, node in enumerate(nodes)]
    id_set = set(ids)

    adj = defaultdict(list)

    for e in edges:
        u = safe_int(e.get("u", -1), -1)
        v = safe_int(e.get("v", -1), -1)

        if u in id_set and v in id_set:
            adj[u].append(v)
            adj[v].append(u)

    seen = set()
    comp_sizes = []

    for nid in ids:
        if nid in seen:
            continue

        stack = [nid]
        seen.add(nid)
        cnt = 0

        while stack:
            u = stack.pop()
            cnt += 1

            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)

        comp_sizes.append(cnt)

    num_components = len(comp_sizes)
    cycle_rank = max(0, m - n + num_components)

    motifs = []

    if num_components == 1:
        motifs.append("connected")
    if cycle_rank > 0:
        motifs.append("cycle")
    if len(degree) > 0 and max(degree.values()) >= 3:
        motifs.append("branch_or_hub")

    for jt in edge_type_counts:
        if jt == "E2E":
            motifs.append("has_E2E")
        elif jt == "T":
            motifs.append("has_T")
        elif jt == "X":
            motifs.append("has_X")

    return {
        "num_nodes": n,
        "num_edges": m,
        "edge_type_counts": dict(edge_type_counts),
        "degrees": {str(k): int(v) for k, v in degree.items()},
        "max_degree": max(degree.values()) if len(degree) > 0 else 0,
        "num_components": num_components,
        "component_sizes": comp_sizes,
        "cycle_rank": cycle_rank,
        "has_cycle": cycle_rank > 0,
        "motifs": sorted(list(set(motifs))),
    }


def build_candidate_from_graph_sample(sample, sample_idx, primitive_entries):
    src_nodes = get_nodes(sample)
    edges = normalize_edges(get_edges(sample))

    if len(src_nodes) == 0:
        return None

    current_shape_counter = Counter()

    nodes = []

    for i, src_node in enumerate(src_nodes):
        node = build_node_with_primitive(
            src_node=src_node,
            edges=edges,
            primitive_entries=primitive_entries,
            current_shape_counter=current_shape_counter,
            fallback_idx=i,
        )
        nodes.append(node)

    nodes = sorted(nodes, key=lambda x: x["node_id"])

    topo_summary = build_topology_summary(nodes, edges)

    sample_id = get_sample_id(sample, sample_idx)
    cand_id = f"glyph_candidate_{sample_idx:05d}"

    template_ref = sample.get("template_ref", {})

    candidate = {
        "generated_glyph_id": cand_id,
        "glyph_candidate_id": cand_id,
        "candidate_id": cand_id,

        "source_graph_sample_id": sample_id,
        "generation_type": "layout_aware_graph_with_primitives",

        "template_ref": deepcopy(template_ref),

        "glyph_uid": template_ref.get("glyph_uid", sample.get("glyph_uid", "")),
        "source_file": template_ref.get("source_file", sample.get("source_file", "")),
        "hex_key": template_ref.get("hex_key", sample.get("hex_key", "")),
        "char": template_ref.get("char", sample.get("char", "")),

        "nodes": nodes,
        "topology": {
            "positive_edges_undirected": edges,
            "topology_summary": topo_summary,
        },
        "topology_summary": topo_summary,

        "primitive_sampling_summary": {
            "shape_counter": dict(current_shape_counter),
            "unique_shape_count": len(current_shape_counter),
            "num_nodes": len(nodes),
            "num_edges": len(edges),
        },

        "sampler_config": deepcopy(SAMPLER_CONFIG),
    }

    return candidate


# =========================================================
# 🎨 Preview
# =========================================================
def apply_variant(polyline, variant_id):
    pts = np.asarray(polyline, dtype=np.float32).copy()
    v = int(variant_id) % 4

    if v == 0:
        return pts

    if v == 1:
        out = pts.copy()
        out[:, 1] = pts[::-1, 1]
        return out

    if v == 2:
        out = pts.copy()
        out[:, 1] = -out[:, 1]
        return out

    if v == 3:
        out = pts.copy()
        out[:, 1] = -pts[::-1, 1]
        return out

    return pts


def transform_polyline(local_poly, layout_prior):
    poly = np.asarray(local_poly, dtype=np.float32)

    center = np.asarray(layout_prior["center_norm"], dtype=np.float32) * CANVAS_SIZE
    theta = safe_float(layout_prior["rotation_rad"], 0.0)
    length = safe_float(layout_prior["length_norm"], 0.25) * CANVAS_SIZE

    c = math.cos(theta)
    s = math.sin(theta)

    R = np.asarray([[c, -s], [s, c]], dtype=np.float32)

    pts = poly * length
    pts = pts @ R.T
    pts = pts + center[None, :]

    return pts


def extract_node_polyline_world(node):
    ref = node.get("primitive_ref", {})

    poly = ref.get("primitive_polyline_local_norm", None)

    if poly is None:
        poly = [[-0.5, 0.0], [0.5, 0.0]]

    poly = normalize_local_polyline(poly)
    poly = apply_variant(poly, node.get("variant_id", 0))

    lp = node.get("layout_prior", {})
    return transform_polyline(poly, lp)


def draw_candidate_preview(candidate, out_path):
    nodes = candidate["nodes"]
    edges = candidate["topology"]["positive_edges_undirected"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))

    ax_axis, ax_color, ax_black = axes

    colors = [
        "#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA",
        "#00897B", "#6D4C41", "#3949AB", "#D81B60", "#7CB342",
        "#546E7A", "#F4511E",
    ]

    node_pos = {}

    # 1. layout axis
    for i, node in enumerate(nodes):
        lp = node["layout_prior"]
        c = np.asarray(lp["center_norm"], dtype=np.float32) * CANVAS_SIZE
        theta = safe_float(lp["rotation_rad"], 0.0)
        length = safe_float(lp["length_norm"], 0.25) * CANVAS_SIZE

        d = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float32) * length * 0.5

        p0 = c - d
        p1 = c + d

        color = colors[i % len(colors)]
        node_pos[node["node_id"]] = c

        ax_axis.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=3.0)
        ax_axis.scatter([c[0]], [c[1]], color=color, s=35)
        ax_axis.text(c[0] + 4, c[1] - 4, f"N{node['node_id']}", fontsize=8, color=color)

    for e in edges:
        u = safe_int(e.get("u", -1), -1)
        v = safe_int(e.get("v", -1), -1)

        if u not in node_pos or v not in node_pos:
            continue

        pu = node_pos[u]
        pv = node_pos[v]
        pm = 0.5 * (pu + pv)
        jt = str(e.get("j_type", "?"))

        if jt == "T":
            color = "#D81B60"
            ls = "--"
        elif jt == "X":
            color = "#1E88E5"
            ls = ":"
        else:
            color = "#555555"
            ls = "-"

        ax_axis.plot([pu[0], pv[0]], [pu[1], pv[1]], color=color, lw=1.0, ls=ls, alpha=0.6)
        ax_axis.text(pm[0], pm[1], jt, fontsize=7, color=color)

    ax_axis.set_title("layout axis + topology", fontsize=9)

    # 2. colored primitive
    for i, node in enumerate(nodes):
        pts = extract_node_polyline_world(node)
        color = colors[i % len(colors)]

        ax_color.plot(pts[:, 0], pts[:, 1], color=color, lw=2.8)
        ax_color.scatter([pts[0, 0]], [pts[0, 1]], color="green", s=20, zorder=6)
        ax_color.scatter([pts[-1, 0]], [pts[-1, 1]], color="red", s=20, zorder=6)

        c = np.asarray(node["layout_prior"]["center_norm"], dtype=np.float32) * CANVAS_SIZE
        label = f"S{node['shape_code']} V{node['variant_id']} W{node['width_token']}"
        ax_color.text(c[0] + 4, c[1] + 8, label, fontsize=7, color=color)

    ax_color.set_title("colored primitive", fontsize=9)

    # 3. black render
    for node in nodes:
        pts = extract_node_polyline_world(node)
        ax_black.plot(
            pts[:, 0],
            pts[:, 1],
            color="black",
            lw=5.0,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    ax_black.set_title("black primitive skeleton", fontsize=9)

    title = (
        f"{candidate['generated_glyph_id']} | "
        f"N={len(nodes)} E={len(edges)} | "
        f"unique_shape={candidate['primitive_sampling_summary']['unique_shape_count']}"
    )

    fig.suptitle(title, fontsize=10)

    for ax in axes:
        ax.set_xlim(0, CANVAS_SIZE)
        ax.set_ylim(CANVAS_SIZE, 0)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.22)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_previews(candidates):
    reset_dir(PREVIEW_DIR)

    n = min(PREVIEW_COUNT, len(candidates))

    print("\n" + "=" * 80)
    print("🖼️ Rendering Primitive Sampler Previews")
    print("=" * 80)

    for i in range(n):
        cand = candidates[i]
        out_path = os.path.join(
            PREVIEW_DIR,
            f"{i:03d}_{safe_filename(cand['generated_glyph_id'])}.png",
        )
        draw_candidate_preview(cand, out_path)

    print(f"  saved preview count: {n}")
    print(f"  preview_dir: {PREVIEW_DIR}")
    print("=" * 80 + "\n")


# =========================================================
# 📊 Summary
# =========================================================
def summarize_candidates(candidates):
    shape_hist = Counter()
    width_hist = Counter()
    variant_hist = Counter()
    source_hist = Counter()
    stroke_count_hist = Counter()
    edge_count_hist = Counter()
    motif_hist = Counter()

    missing_layout = 0
    missing_primitive_ref = 0
    total_nodes = 0

    for cand in candidates:
        nodes = cand.get("nodes", [])
        edges = cand.get("topology", {}).get("positive_edges_undirected", [])

        stroke_count_hist[len(nodes)] += 1
        edge_count_hist[len(edges)] += 1

        source_hist[cand.get("source_file", "")] += 1

        topo_summary = cand.get("topology_summary", {})

        for m in topo_summary.get("motifs", []):
            motif_hist[m] += 1

        for node in nodes:
            total_nodes += 1

            shape_hist[safe_int(node.get("shape_code", -1), -1)] += 1
            width_hist[safe_int(node.get("width_token", -1), -1)] += 1
            variant_hist[safe_int(node.get("variant_id", -1), -1)] += 1

            if "layout_prior" not in node or not isinstance(node["layout_prior"], dict):
                missing_layout += 1
            elif "center_norm" not in node["layout_prior"]:
                missing_layout += 1

            if "primitive_ref" not in node or not isinstance(node["primitive_ref"], dict):
                missing_primitive_ref += 1

    return {
        "candidate_count": len(candidates),
        "total_nodes": total_nodes,
        "missing_layout_nodes": missing_layout,
        "missing_layout_rate": round(missing_layout / max(1, total_nodes), 6),
        "missing_primitive_ref_nodes": missing_primitive_ref,
        "missing_primitive_ref_rate": round(missing_primitive_ref / max(1, total_nodes), 6),
        "shape_code_hist": dict(shape_hist),
        "width_token_hist": dict(width_hist),
        "variant_id_hist": dict(variant_hist),
        "stroke_count_hist": dict(stroke_count_hist),
        "edge_count_hist": dict(edge_count_hist),
        "motif_hist": dict(motif_hist),
        "source_file_hist": dict(source_hist),
    }


# =========================================================
# 🚀 Main
# =========================================================
def main():
    set_seed(RANDOM_SEED)

    print("\n" + "=" * 80)
    print("🚀 Stroke Primitive Sampler - Layout Aware")
    print("=" * 80)
    print(f"  graph_input_file: {GRAPH_INPUT_FILE}")
    print(f"  corpus_file:      {CORPUS_FILE}")
    print(f"  output_file:      {OUTPUT_FILE}")
    print(f"  preview_dir:      {PREVIEW_DIR}")
    print("=" * 80)

    graph_data = load_json(GRAPH_INPUT_FILE, required=True)
    corpus_data = load_json(CORPUS_FILE, required=True)

    graph_samples = get_graph_samples(graph_data)

    if NUM_CANDIDATES is not None:
        graph_samples = graph_samples[:int(NUM_CANDIDATES)]

    primitive_library = load_primitive_library(corpus_data)
    primitive_entries = build_primitive_entries(primitive_library)

    print("\n[Input]")
    print(f"  graph_sample_count: {len(graph_samples)}")
    print(f"  primitive_library_count: {len(primitive_library)}")
    print(f"  primitive_entries_count: {len(primitive_entries)}")

    if len(graph_samples) == 0:
        raise RuntimeError("没有读取到 graph samples。请检查 layout_aware_graph_samples.json。")

    if len(primitive_entries) == 0:
        raise RuntimeError("没有读取到 primitive entries。请检查 alien_glyph_pcg_corpus.json 的 stroke_primitive_library。")

    print("\n[Primitive Entries]")
    shape_codes = [safe_int(e.get("shape_code", -1), -1) for e in primitive_entries]
    print(f"  shape_codes: {shape_codes}")

    candidates = []
    skipped = Counter()

    for idx, sample in enumerate(graph_samples):
        try:
            cand = build_candidate_from_graph_sample(
                sample=sample,
                sample_idx=idx,
                primitive_entries=primitive_entries,
            )
        except Exception as e:
            skipped[f"exception::{type(e).__name__}"] += 1
            continue

        if cand is None:
            skipped["none_candidate"] += 1
            continue

        if len(cand["nodes"]) == 0:
            skipped["empty_nodes"] += 1
            continue

        candidates.append(cand)

    if len(candidates) == 0:
        raise RuntimeError("没有成功生成 glyph candidates with primitives。")

    summary = summarize_candidates(candidates)

    print("\n" + "=" * 80)
    print("📊 Primitive Sampling Summary")
    print("=" * 80)
    print(f"  candidate_count: {summary['candidate_count']}")
    print(f"  total_nodes: {summary['total_nodes']}")
    print(f"  skipped: {dict(skipped)}")
    print(f"  missing_layout_rate: {summary['missing_layout_rate']}")
    print(f"  missing_primitive_ref_rate: {summary['missing_primitive_ref_rate']}")

    print("\n[Stroke Count Hist]")
    for k, v in sorted(summary["stroke_count_hist"].items()):
        print(f"  {k}: {v}")

    print("\n[Edge Count Hist]")
    for k, v in sorted(summary["edge_count_hist"].items()):
        print(f"  {k}: {v}")

    print("\n[Shape Code Hist]")
    for k, v in Counter(summary["shape_code_hist"]).most_common(20):
        print(f"  shape_code={k}: {v}")

    print("\n[Width Token Hist]")
    for k, v in sorted(summary["width_token_hist"].items(), key=lambda x: str(x[0])):
        print(f"  width_token={k}: {v}")

    print("\n[Variant Hist]")
    for k, v in sorted(summary["variant_id_hist"].items(), key=lambda x: str(x[0])):
        print(f"  variant_id={k}: {v}")

    print("\n[Motif Hist]")
    for k, v in summary["motif_hist"].items():
        print(f"  {k}: {v}")

    output = {
        "schema_version": "glyph_candidates_with_primitives_v2_layout_aware",
        "graph_input_file": GRAPH_INPUT_FILE,
        "corpus_file": CORPUS_FILE,
        "output_file": OUTPUT_FILE,
        "preview_dir": PREVIEW_DIR,

        "sampler_config": deepcopy(SAMPLER_CONFIG),
        "summary": summary,

        # 标准 key
        "glyph_candidates_with_primitives": candidates,

        # 兼容 key
        "glyph_candidates": candidates,
        "candidates": candidates,
    }

    save_json(output, OUTPUT_FILE)

    print("\n" + "=" * 80)
    print("💾 Saved")
    print("=" * 80)
    print(f"  {OUTPUT_FILE}")

    render_previews(candidates)

    print("\n📌 下一步：")
    print("  1. 先打开 glyph_candidates_with_primitives_previews 看 primitive 是否跟 layout 对齐")
    print("  2. 再运行 primitive_sampler_audit.py")
    print("  3. audit 中 missing_layout_rate 应该变成 0.0")
    print("  4. 如果 missing_layout_rate 仍然是 1.0，说明后续脚本读错了旧文件")
    print("  5. 然后运行 constraint_solver.py")
    print("  6. 最后运行 score_solved_glyphs.py")


if __name__ == "__main__":
    main()