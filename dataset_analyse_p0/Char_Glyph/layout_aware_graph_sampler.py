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

LAYOUT_TEMPLATE_FILE = os.path.join(SCRIPT_DIR, "glyph_layout_templates.json")

OUTPUT_FILE = os.path.join(SCRIPT_DIR, "layout_aware_graph_samples.json")
PREVIEW_DIR = os.path.join(SCRIPT_DIR, "layout_aware_graph_previews")

RANDOM_SEED = 239351274

NUM_SAMPLES = 1000

CLEAR_PREVIEW_DIR = True
PREVIEW_COUNT = 60

CANVAS_SIZE = 400.0


# =========================================================
# 🎛️ Layout-aware sampling config
# =========================================================
STYLE_CONFIG = {
    # 先不要太复杂，3~6 条 stroke 比较稳
    "stroke_count_range": [3, 6],

    # 是否偏好 connected template
    "prefer_connected": True,

    # motif 权重
    "motif_weights": {
        "connected": 1.25,
        "has_E2E": 1.00,
        "has_T": 1.05,
        "has_X": 0.70,
        "cycle": 0.45,
        "branch_or_hub": 0.85,
    },

    # source 权重，可留空
    # 例如 {"Megrim_topo.json": 1.5}
    "source_file_weights": {},

    # 是否重排 node id，避免直接复制模板身份
    "relabel_nodes": True,

    # 是否清空 shape_code，让 primitive sampler 重新采样
    "clear_shape_code": True,

    # 如果 clear_shape_code=False，可以有概率保留原 shape_code
    "keep_template_shape_code_prob": 0.0,

    # 全局 layout 变换
    "global_rotation_jitter_deg": 18.0,
    "global_scale_jitter": 0.10,
    "global_translate_jitter": 0.045,

    # node 局部 jitter
    "jitter_center_std": 0.020,
    "jitter_rotation_std_deg": 7.0,
    "jitter_length_std": 0.08,

    # 边属性 jitter
    "jitter_t_std": 0.025,
    "jitter_angle_std_deg": 5.0,

    # 变换后保持在画布中心区域
    "normalize_to_canvas": True,
    "canvas_margin_norm": 0.08,

    # 是否轻微 dropout edge，不建议第一版开
    "mutate_drop_edge_prob": 0.0,

    # 是否保存 source bezier，仅调试用
    "keep_source_bezier": False,
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


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")

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


def to_np2(x, default=None):
    if x is None:
        return default

    arr = np.asarray(x, dtype=np.float32)

    if arr.ndim == 1 and arr.shape[0] >= 2:
        return arr[:2]

    return default


def rotate_point_around(p, center, theta):
    p = np.asarray(p, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)

    d = p - center
    c = math.cos(theta)
    s = math.sin(theta)

    R = np.asarray([[c, -s], [s, c]], dtype=np.float32)

    return center + d @ R.T


# =========================================================
# 📦 Template 读取
# =========================================================
def get_template_list(data):
    if isinstance(data, list):
        return data

    for k in [
        "layout_templates",
        "templates",
        "glyph_layout_templates",
        "samples",
        "glyph_samples",
    ]:
        if k in data and isinstance(data[k], list):
            return data[k]

    for k, v in data.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            print(f"[WARN] 未识别标准 template key，使用字段: {k}")
            return v

    return []


def get_template_id(template, idx):
    for k in [
        "layout_template_id",
        "template_id",
        "sample_id",
        "source_sample_id",
        "glyph_uid",
    ]:
        if template.get(k, ""):
            return str(template[k])

    return f"layout_template_{idx:05d}"


def get_nodes(template):
    if "nodes" in template and isinstance(template["nodes"], list):
        return template["nodes"]
    return []


def get_edges(template):
    topo = template.get("topology", {})

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

    if "edges" in template and isinstance(template["edges"], list):
        return template["edges"]

    return []


def node_id_of(node, fallback):
    return safe_int(node.get("node_id", fallback), fallback)


def read_layout_prior(node):
    lp = node.get("layout_prior", {})

    if not isinstance(lp, dict):
        lp = {}

    center = None

    if "center_norm" in lp:
        center = to_np2(lp["center_norm"])
    elif "center_norm" in node:
        center = to_np2(node["center_norm"])
    elif "center_px" in node:
        center = to_np2(node["center_px"]) / CANVAS_SIZE

    if center is None:
        p0 = None
        p3 = None

        if "p0_norm" in lp:
            p0 = to_np2(lp["p0_norm"])
        elif "p0_norm" in node:
            p0 = to_np2(node["p0_norm"])

        if "p3_norm" in lp:
            p3 = to_np2(lp["p3_norm"])
        elif "p3_norm" in node:
            p3 = to_np2(node["p3_norm"])

        if p0 is not None and p3 is not None:
            center = 0.5 * (p0 + p3)

    if center is None:
        center = np.asarray([0.5, 0.5], dtype=np.float32)

    rotation = 0.0

    if "rotation_rad" in lp:
        rotation = safe_float(lp["rotation_rad"], 0.0)
    elif "rotation_deg" in lp:
        rotation = deg_to_rad(safe_float(lp["rotation_deg"], 0.0))
    elif "rotation_rad" in node:
        rotation = safe_float(node["rotation_rad"], 0.0)
    elif "rotation_deg" in node:
        rotation = deg_to_rad(safe_float(node["rotation_deg"], 0.0))
    else:
        p0 = None
        p3 = None

        if "p0_norm" in lp:
            p0 = to_np2(lp["p0_norm"])
        elif "p0_norm" in node:
            p0 = to_np2(node["p0_norm"])

        if "p3_norm" in lp:
            p3 = to_np2(lp["p3_norm"])
        elif "p3_norm" in node:
            p3 = to_np2(node["p3_norm"])

        if p0 is not None and p3 is not None:
            d = p3 - p0
            if np.linalg.norm(d) > 1e-8:
                rotation = math.atan2(d[1], d[0])

    length = None

    for k in ["length_norm", "scale_norm", "arc_length_norm", "chord_length_norm"]:
        if k in lp:
            length = safe_float(lp[k], 0.25)
            break

    if length is None:
        for k in ["length_norm", "scale_norm", "arc_length_norm", "chord_length_norm"]:
            if k in node:
                length = safe_float(node[k], 0.25)
                break

    if length is None:
        p0 = None
        p3 = None

        if "p0_norm" in lp:
            p0 = to_np2(lp["p0_norm"])
        elif "p0_norm" in node:
            p0 = to_np2(node["p0_norm"])

        if "p3_norm" in lp:
            p3 = to_np2(lp["p3_norm"])
        elif "p3_norm" in node:
            p3 = to_np2(node["p3_norm"])

        if p0 is not None and p3 is not None:
            length = float(np.linalg.norm(p3 - p0))

    if length is None:
        length = 0.25

    length = clamp(length, 0.03, 1.5)
    rotation = normalize_angle_pi(rotation)

    p0 = center - 0.5 * length * np.asarray([math.cos(rotation), math.sin(rotation)], dtype=np.float32)
    p3 = center + 0.5 * length * np.asarray([math.cos(rotation), math.sin(rotation)], dtype=np.float32)

    return {
        "center_norm": center.astype(np.float32),
        "rotation_rad": float(rotation),
        "length_norm": float(length),
        "scale_norm": float(length),
        "p0_norm": p0.astype(np.float32),
        "p3_norm": p3.astype(np.float32),
    }


def template_motifs(template):
    motifs = set()

    topo_summary = template.get("topology_summary", template.get("summary", {}))

    if isinstance(topo_summary, dict):
        if topo_summary.get("num_components", 1) == 1:
            motifs.add("connected")
        if topo_summary.get("has_cycle", False):
            motifs.add("cycle")
        if topo_summary.get("max_degree", 0) >= 3:
            motifs.add("branch_or_hub")

    edges = get_edges(template)

    if len(edges) > 0:
        motifs.add("connected")

    for e in edges:
        jt = str(e.get("j_type", ""))

        if jt == "E2E":
            motifs.add("has_E2E")
        elif jt == "T":
            motifs.add("has_T")
        elif jt == "X":
            motifs.add("has_X")

    # 从 node degree 补 branch
    degree = Counter()

    for e in edges:
        u = safe_int(e.get("u", -1), -1)
        v = safe_int(e.get("v", -1), -1)
        if u >= 0 and v >= 0:
            degree[u] += 1
            degree[v] += 1

    if len(degree) > 0 and max(degree.values()) >= 3:
        motifs.add("branch_or_hub")

    return motifs


def is_connected_template(template):
    nodes = get_nodes(template)
    edges = get_edges(template)

    if len(nodes) <= 1:
        return True

    ids = [node_id_of(n, i) for i, n in enumerate(nodes)]
    id_set = set(ids)

    adj = defaultdict(list)

    for e in edges:
        u = safe_int(e.get("u", -1), -1)
        v = safe_int(e.get("v", -1), -1)

        if u in id_set and v in id_set:
            adj[u].append(v)
            adj[v].append(u)

    if len(adj) == 0:
        return False

    start = ids[0]
    seen = set([start])
    stack = [start]

    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)

    return len(seen) == len(id_set)


# =========================================================
# 🎲 Template weighting / sampling
# =========================================================
def template_weight(template):
    nodes = get_nodes(template)
    n = len(nodes)

    lo, hi = STYLE_CONFIG["stroke_count_range"]

    if n < lo or n > hi:
        return 0.0

    edges = get_edges(template)

    if len(edges) == 0:
        return 0.0

    w = 1.0

    motifs = template_motifs(template)
    motif_weights = STYLE_CONFIG.get("motif_weights", {})

    for m, mw in motif_weights.items():
        if m in motifs:
            w *= float(mw)

    if STYLE_CONFIG.get("prefer_connected", True):
        if is_connected_template(template):
            w *= 1.25
        else:
            w *= 0.35

    source_file = template.get("source_file", "")
    source_weights = STYLE_CONFIG.get("source_file_weights", {})

    if source_file in source_weights:
        w *= float(source_weights[source_file])

    # 过复杂的先略微压低
    if n >= 6:
        w *= 0.80

    return max(0.0, float(w))


def weighted_choice(items, weights):
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


# =========================================================
# 🔄 Layout mutation
# =========================================================
def collect_layout_points(nodes):
    pts = []

    for n in nodes:
        lp = read_layout_prior(n)
        pts.append(lp["center_norm"])

    if len(pts) == 0:
        return np.asarray([[0.5, 0.5]], dtype=np.float32)

    return np.asarray(pts, dtype=np.float32)


def normalize_centers_after_transform(layout_items):
    if not STYLE_CONFIG.get("normalize_to_canvas", True):
        return layout_items

    margin = float(STYLE_CONFIG.get("canvas_margin_norm", 0.08))
    lo = margin
    hi = 1.0 - margin

    centers = np.asarray([x["center_norm"] for x in layout_items], dtype=np.float32)

    xmin, ymin = np.min(centers, axis=0)
    xmax, ymax = np.max(centers, axis=0)

    w = xmax - xmin
    h = ymax - ymin

    if w < 1e-6 and h < 1e-6:
        shift = np.asarray([0.5, 0.5], dtype=np.float32) - centers[0]
        for item in layout_items:
            item["center_norm"] = item["center_norm"] + shift
        return layout_items

    scale = min((hi - lo) / max(w, 1e-6), (hi - lo) / max(h, 1e-6), 1.0)

    old_center = np.asarray([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5], dtype=np.float32)
    new_center = np.asarray([0.5, 0.5], dtype=np.float32)

    for item in layout_items:
        c = item["center_norm"]
        c = new_center + (c - old_center) * scale
        c = np.clip(c, lo, hi)

        item["center_norm"] = c.astype(np.float32)
        item["length_norm"] = clamp(item["length_norm"] * scale, 0.03, 1.2)
        item["scale_norm"] = item["length_norm"]

    return layout_items


def mutate_layout_items(template):
    nodes = get_nodes(template)

    if len(nodes) == 0:
        return []

    centers = collect_layout_points(nodes)
    template_center = np.mean(centers, axis=0)

    global_rot = deg_to_rad(random.gauss(0.0, STYLE_CONFIG["global_rotation_jitter_deg"]))
    global_scale = math.exp(random.gauss(0.0, STYLE_CONFIG["global_scale_jitter"]))
    global_translate = np.asarray(
        [
            random.gauss(0.0, STYLE_CONFIG["global_translate_jitter"]),
            random.gauss(0.0, STYLE_CONFIG["global_translate_jitter"]),
        ],
        dtype=np.float32,
    )

    layout_items = []

    for i, node in enumerate(nodes):
        old_id = node_id_of(node, i)
        lp = read_layout_prior(node)

        c = lp["center_norm"].copy()

        # global rotation around template center
        c = rotate_point_around(c, template_center, global_rot)

        # global scale around template center
        c = template_center + (c - template_center) * global_scale

        # translate
        c = c + global_translate

        # local jitter
        c = c + np.random.normal(
            0.0,
            STYLE_CONFIG["jitter_center_std"],
            size=2,
        ).astype(np.float32)

        rot = normalize_angle_pi(
            lp["rotation_rad"]
            + global_rot
            + deg_to_rad(random.gauss(0.0, STYLE_CONFIG["jitter_rotation_std_deg"]))
        )

        length = lp["length_norm"] * global_scale
        length = length * math.exp(random.gauss(0.0, STYLE_CONFIG["jitter_length_std"]))
        length = clamp(length, 0.04, 1.2)

        layout_items.append({
            "old_node_id": old_id,
            "source_node": node,
            "center_norm": c.astype(np.float32),
            "rotation_rad": float(rot),
            "length_norm": float(length),
            "scale_norm": float(length),
        })

    layout_items = normalize_centers_after_transform(layout_items)

    for item in layout_items:
        c = item["center_norm"]
        rot = item["rotation_rad"]
        length = item["length_norm"]

        d = np.asarray([math.cos(rot), math.sin(rot)], dtype=np.float32) * length * 0.5

        item["p0_norm"] = (c - d).astype(np.float32)
        item["p3_norm"] = (c + d).astype(np.float32)

    return layout_items


def build_relabel_mapping(layout_items):
    old_ids = [x["old_node_id"] for x in layout_items]

    if not STYLE_CONFIG.get("relabel_nodes", True):
        return {old_id: i for i, old_id in enumerate(old_ids)}

    new_ids = list(range(len(old_ids)))
    random.shuffle(new_ids)

    return {
        old_id: new_id
        for old_id, new_id in zip(old_ids, new_ids)
    }


def infer_node_role_from_edges(new_id, edges):
    roles = set()
    incident_edge_types = []
    degree = 0

    for e in edges:
        u = safe_int(e.get("u", -1), -1)
        v = safe_int(e.get("v", -1), -1)

        if u == new_id or v == new_id:
            degree += 1
            jt = str(e.get("j_type", "UNKNOWN"))
            incident_edge_types.append(jt)

            if jt == "T":
                # 通常 T 里 host/guest 字段可能存在
                role_u = str(e.get("role_u", "")).lower()
                role_v = str(e.get("role_v", "")).lower()

                if u == new_id:
                    if "host" in role_u:
                        roles.add("host")
                    elif "guest" in role_u:
                        roles.add("guest")
                if v == new_id:
                    if "host" in role_v:
                        roles.add("host")
                    elif "guest" in role_v:
                        roles.add("guest")

    if len(roles) == 0:
        roles.add("stroke")

    return {
        "degree": degree,
        "incident_edge_types": sorted(list(set(incident_edge_types))),
        "roles": sorted(list(roles)),
    }


def mutate_edge(edge, relabel):
    ee = deepcopy(edge)

    if "u" not in ee or "v" not in ee:
        return None

    old_u = safe_int(ee["u"], -1)
    old_v = safe_int(ee["v"], -1)

    if old_u not in relabel or old_v not in relabel:
        return None

    ee["u"] = int(relabel[old_u])
    ee["v"] = int(relabel[old_v])

    ee["source_u"] = int(old_u)
    ee["source_v"] = int(old_v)

    if "j_type" not in ee:
        idx = safe_int(ee.get("j_type_idx", 0), 0)
        ee["j_type"] = {
            1: "E2E",
            2: "X",
            3: "T",
        }.get(idx, "UNKNOWN")

    if "t_u" in ee:
        ee["t_u"] = clamp(
            safe_float(ee["t_u"], 0.0) + random.gauss(0.0, STYLE_CONFIG["jitter_t_std"]),
            0.0,
            1.0,
        )

    if "t_v" in ee:
        ee["t_v"] = clamp(
            safe_float(ee["t_v"], 0.0) + random.gauss(0.0, STYLE_CONFIG["jitter_t_std"]),
            0.0,
            1.0,
        )

    if "angle_deg" in ee:
        ee["angle_deg"] = safe_float(ee["angle_deg"], 0.0) + random.gauss(
            0.0,
            STYLE_CONFIG["jitter_angle_std_deg"],
        )

    if "angle_sin" in ee or "angle_cos" in ee:
        if "angle_deg" in ee:
            ar = deg_to_rad(safe_float(ee["angle_deg"], 0.0))
        else:
            ar = math.atan2(
                safe_float(ee.get("angle_sin", 0.0), 0.0),
                safe_float(ee.get("angle_cos", 1.0), 1.0),
            )
            ar += deg_to_rad(random.gauss(0.0, STYLE_CONFIG["jitter_angle_std_deg"]))

        ee["angle_sin"] = math.sin(ar)
        ee["angle_cos"] = math.cos(ar)

    return ee


def build_topology_summary(nodes, edges):
    n = len(nodes)
    m = len(edges)

    degree = Counter()
    edge_type_counts = Counter()

    for e in edges:
        u = safe_int(e.get("u", -1), -1)
        v = safe_int(e.get("v", -1), -1)

        if u >= 0 and v >= 0:
            degree[u] += 1
            degree[v] += 1

        edge_type_counts[str(e.get("j_type", "UNKNOWN"))] += 1

    ids = [safe_int(nod["node_id"], i) for i, nod in enumerate(nodes)]
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
    if any(v >= 3 for v in degree.values()):
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


# =========================================================
# 🧬 Build sample
# =========================================================
def build_sample_from_template(template, template_idx, sample_idx):
    template_id = get_template_id(template, template_idx)

    layout_items = mutate_layout_items(template)

    if len(layout_items) == 0:
        return None

    relabel = build_relabel_mapping(layout_items)

    # edges
    source_edges = get_edges(template)
    edges = []

    for e in source_edges:
        if random.random() < STYLE_CONFIG.get("mutate_drop_edge_prob", 0.0):
            continue

        ee = mutate_edge(e, relabel)
        if ee is not None:
            edges.append(ee)

    # node map sorted by new id
    nodes = []

    for item in layout_items:
        old_id = item["old_node_id"]
        new_id = int(relabel[old_id])
        src_node = item["source_node"]

        template_shape_code = src_node.get("shape_code", src_node.get("shape_token", None))
        template_width_token = src_node.get("width_token", None)

        keep_shape = False

        if not STYLE_CONFIG.get("clear_shape_code", True):
            keep_shape = random.random() < STYLE_CONFIG.get("keep_template_shape_code_prob", 0.0)

        shape_code = template_shape_code if keep_shape else None
        width_token = template_width_token if keep_shape else None

        lp = {
            "center_norm": item["center_norm"].astype(float).tolist(),
            "rotation_rad": float(item["rotation_rad"]),
            "rotation_deg": float(rad_to_deg(item["rotation_rad"])),
            "length_norm": float(item["length_norm"]),
            "scale_norm": float(item["scale_norm"]),
            "p0_norm": item["p0_norm"].astype(float).tolist(),
            "p3_norm": item["p3_norm"].astype(float).tolist(),
        }

        node = {
            "node_id": new_id,
            "source_template_node_id": old_id,
            "source_layout_template_id": template_id,

            # 这些让 primitive sampler 后续重采样
            "shape_code": shape_code,
            "shape_token": shape_code,
            "width_token": width_token,

            # 仅用于分析，不作为正式 primitive
            "template_shape_code": template_shape_code,
            "template_width_token": template_width_token,

            "layout_prior": lp,

            "solver_variable_slot": {
                "center_norm": lp["center_norm"],
                "rotation_rad": lp["rotation_rad"],
                "scale": lp["scale_norm"],
            },

            "grammar_role": {
                "degree": 0,
                "incident_edge_types": [],
                "roles": ["stroke"],
            },
        }

        if STYLE_CONFIG.get("keep_source_bezier", False):
            if "source_mother_bezier_norm" in src_node:
                node["source_mother_bezier_norm"] = src_node["source_mother_bezier_norm"]

        nodes.append(node)

    nodes = sorted(nodes, key=lambda x: x["node_id"])

    # edge role after relabel
    for node in nodes:
        node["grammar_role"] = infer_node_role_from_edges(node["node_id"], edges)

    topo_summary = build_topology_summary(nodes, edges)

    sample_id = f"layout_graph_sample_{sample_idx:05d}"

    sample = {
        "grammar_sample_id": sample_id,
        "generation_type": "layout_aware_graph_grammar",
        "template_ref": {
            "layout_template_id": template_id,
            "template_index": template_idx,
            "source_file": template.get("source_file", ""),
            "glyph_uid": template.get("glyph_uid", ""),
            "hex_key": template.get("hex_key", ""),
            "char": template.get("char", ""),
        },
        "relabel_mapping": {
            str(k): int(v)
            for k, v in relabel.items()
        },
        "nodes": nodes,
        "topology": {
            "positive_edges_undirected": edges,
            "topology_summary": topo_summary,
        },
        "topology_summary": topo_summary,
        "style_config": deepcopy(STYLE_CONFIG),
    }

    return sample


# =========================================================
# 📊 Summary
# =========================================================
def summarize_samples(samples):
    stroke_count_hist = Counter()
    edge_count_hist = Counter()
    edge_type_hist = Counter()
    motif_hist = Counter()
    source_hist = Counter()
    template_hist = Counter()

    for s in samples:
        nodes = s.get("nodes", [])
        edges = s.get("topology", {}).get("positive_edges_undirected", [])

        stroke_count_hist[len(nodes)] += 1
        edge_count_hist[len(edges)] += 1

        for e in edges:
            edge_type_hist[str(e.get("j_type", "UNKNOWN"))] += 1

        summary = s.get("topology_summary", {})

        for m in summary.get("motifs", []):
            motif_hist[m] += 1

        ref = s.get("template_ref", {})
        source_hist[ref.get("source_file", "")] += 1
        template_hist[ref.get("layout_template_id", "")] += 1

    return {
        "sample_count": len(samples),
        "stroke_count_hist": dict(stroke_count_hist),
        "edge_count_hist": dict(edge_count_hist),
        "edge_type_hist": dict(edge_type_hist),
        "motif_hist": dict(motif_hist),
        "source_file_hist": dict(source_hist),
        "template_ref_hist_top20": dict(template_hist.most_common(20)),
    }


# =========================================================
# 🎨 Preview
# =========================================================
def draw_sample_preview(sample, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    ax_axis, ax_topo, ax_black = axes

    nodes = sample["nodes"]
    edges = sample["topology"]["positive_edges_undirected"]

    colors = [
        "#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA",
        "#00897B", "#6D4C41", "#3949AB", "#D81B60", "#7CB342",
        "#546E7A", "#F4511E",
    ]

    node_pos = {}

    # 1. Layout axis
    for i, n in enumerate(nodes):
        c = colors[i % len(colors)]
        lp = n["layout_prior"]

        center = np.asarray(lp["center_norm"], dtype=np.float32) * CANVAS_SIZE
        theta = safe_float(lp["rotation_rad"], 0.0)
        length = safe_float(lp["length_norm"], 0.2) * CANVAS_SIZE

        d = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float32) * length * 0.5

        p0 = center - d
        p1 = center + d

        node_pos[n["node_id"]] = center

        ax_axis.plot([p0[0], p1[0]], [p0[1], p1[1]], color=c, lw=3.0, alpha=0.95)
        ax_axis.scatter([center[0]], [center[1]], color=c, s=36, zorder=5)
        ax_axis.text(center[0] + 4, center[1] - 4, f"N{n['node_id']}", fontsize=8, color=c)

    ax_axis.set_title("layout axis", fontsize=10)

    # 2. Topology graph
    for i, n in enumerate(nodes):
        c = colors[i % len(colors)]
        p = node_pos[n["node_id"]]

        ax_topo.scatter([p[0]], [p[1]], color=c, s=70, zorder=5)
        ax_topo.text(p[0] + 4, p[1] - 4, f"N{n['node_id']}", fontsize=9, color=c)

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

        ax_topo.plot([pu[0], pv[0]], [pu[1], pv[1]], color=color, lw=1.6, ls=ls, alpha=0.8)
        ax_topo.text(pm[0], pm[1], jt, fontsize=8, color=color)

    ax_topo.set_title("topology graph", fontsize=10)

    # 3. black skeleton preview
    for i, n in enumerate(nodes):
        lp = n["layout_prior"]

        p0 = np.asarray(lp["p0_norm"], dtype=np.float32) * CANVAS_SIZE
        p3 = np.asarray(lp["p3_norm"], dtype=np.float32) * CANVAS_SIZE

        ax_black.plot(
            [p0[0], p3[0]],
            [p0[1], p3[1]],
            color="black",
            lw=5.0,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    ax_black.set_title("black skeleton preview", fontsize=10)

    title = (
        f"{sample['grammar_sample_id']} | "
        f"N={len(nodes)} E={len(edges)} | "
        f"{sample['template_ref'].get('source_file', '')} | "
        f"{sample['template_ref'].get('hex_key', '')}"
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


def render_previews(samples):
    reset_dir(PREVIEW_DIR)

    n = min(PREVIEW_COUNT, len(samples))

    print("\n" + "=" * 80)
    print("🖼️ Rendering Layout-aware Graph Previews")
    print("=" * 80)

    for i in range(n):
        s = samples[i]
        out_path = os.path.join(
            PREVIEW_DIR,
            f"{i:03d}_{safe_filename(s['grammar_sample_id'])}.png",
        )
        draw_sample_preview(s, out_path)

    print(f"  saved preview count: {n}")
    print(f"  preview_dir: {PREVIEW_DIR}")
    print("=" * 80 + "\n")


# =========================================================
# 🚀 Main
# =========================================================
def main():
    set_seed(RANDOM_SEED)

    print("\n" + "=" * 80)
    print("🚀 Layout-aware Graph Sampler")
    print("=" * 80)
    print(f"  layout_template_file: {LAYOUT_TEMPLATE_FILE}")
    print(f"  output_file:          {OUTPUT_FILE}")
    print(f"  preview_dir:          {PREVIEW_DIR}")
    print(f"  num_samples:          {NUM_SAMPLES}")
    print("=" * 80)

    data = load_json(LAYOUT_TEMPLATE_FILE)
    templates = get_template_list(data)

    if len(templates) == 0:
        raise RuntimeError("没有读取到 layout templates。请检查 glyph_layout_templates.json。")

    weights = [template_weight(t) for t in templates]
    valid_templates = [t for t, w in zip(templates, weights) if w > 0]
    valid_weights = [w for w in weights if w > 0]

    print("\n[Template Input]")
    print(f"  total_templates: {len(templates)}")
    print(f"  valid_templates: {len(valid_templates)}")
    print(f"  stroke_count_range: {STYLE_CONFIG['stroke_count_range']}")

    if len(valid_templates) == 0:
        raise RuntimeError("没有符合条件的 layout template。请放宽 stroke_count_range 或检查边。")

    template_node_hist = Counter(len(get_nodes(t)) for t in valid_templates)
    template_edge_hist = Counter(len(get_edges(t)) for t in valid_templates)
    template_source_hist = Counter(t.get("source_file", "") for t in valid_templates)

    print("\n[Valid Template Node Count]")
    for k, v in sorted(template_node_hist.items()):
        print(f"  {k}: {v}")

    print("\n[Valid Template Edge Count]")
    for k, v in sorted(template_edge_hist.items()):
        print(f"  {k}: {v}")

    print("\n[Valid Template Source]")
    for k, v in template_source_hist.most_common():
        print(f"  {k}: {v}")

    samples = []
    skipped = Counter()

    template_to_index = {
        id(t): i
        for i, t in enumerate(templates)
    }

    attempts = 0
    max_attempts = NUM_SAMPLES * 20

    while len(samples) < NUM_SAMPLES and attempts < max_attempts:
        attempts += 1

        t = weighted_choice(valid_templates, valid_weights)
        tidx = template_to_index.get(id(t), -1)

        try:
            sample = build_sample_from_template(t, tidx, len(samples))
        except Exception as e:
            skipped[f"exception::{type(e).__name__}"] += 1
            continue

        if sample is None:
            skipped["none_sample"] += 1
            continue

        if len(sample["nodes"]) < 2:
            skipped["too_few_nodes"] += 1
            continue

        if len(sample["topology"]["positive_edges_undirected"]) < 1:
            skipped["no_edges"] += 1
            continue

        samples.append(sample)

    if len(samples) == 0:
        raise RuntimeError("没有成功生成 layout-aware graph samples。")

    summary = summarize_samples(samples)

    print("\n" + "=" * 80)
    print("📊 Layout-aware Sampling Summary")
    print("=" * 80)
    print(f"  sample_count: {summary['sample_count']}")
    print(f"  skipped: {dict(skipped)}")

    print("\n[Stroke Count Hist]")
    for k, v in sorted(summary["stroke_count_hist"].items()):
        print(f"  {k}: {v}")

    print("\n[Edge Count Hist]")
    for k, v in sorted(summary["edge_count_hist"].items()):
        print(f"  {k}: {v}")

    print("\n[Edge Type Hist]")
    for k, v in summary["edge_type_hist"].items():
        print(f"  {k}: {v}")

    print("\n[Motif Hist]")
    for k, v in summary["motif_hist"].items():
        print(f"  {k}: {v}")

    print("\n[Source Hist]")
    for k, v in summary["source_file_hist"].items():
        print(f"  {k}: {v}")

    output = {
        "schema_version": "layout_aware_graph_samples_v1",
        "source_layout_template_file": LAYOUT_TEMPLATE_FILE,
        "output_file": OUTPUT_FILE,
        "preview_dir": PREVIEW_DIR,
        "num_samples": len(samples),
        "style_config": deepcopy(STYLE_CONFIG),
        "summary": summary,

        # 为了兼容 stroke_primitive_sampler.py，仍然叫 sampled_topologies
        "sampled_topologies": samples,

        # 额外别名，方便后续脚本读取
        "layout_aware_graph_samples": samples,
    }

    save_json(output, OUTPUT_FILE)

    print("\n" + "=" * 80)
    print("💾 Saved")
    print("=" * 80)
    print(f"  {OUTPUT_FILE}")

    render_previews(samples)

    print("\n📌 下一步：")
    print("  1. 打开 layout_aware_graph_previews 看 layout skeleton 是否像字符")
    print("  2. 修改 stroke_primitive_sampler.py 读取 layout_aware_graph_samples.json")
    print("  3. 确保 primitive sampler 把 node.layout_prior 原样复制到输出 node")
    print("  4. 重新跑 primitive_sampler_audit.py，missing_layout_rate 应该接近 0")
    print("  5. 再跑 constraint_solver.py 和 score_solved_glyphs.py")


if __name__ == "__main__":
    main()