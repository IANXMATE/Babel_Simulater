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

CANDIDATE_FILE = os.path.join(SCRIPT_DIR, "glyph_candidates_with_primitives.json")
CORPUS_FILE = os.path.join(SCRIPT_DIR, "alien_glyph_pcg_corpus.json")

# 可选：如果存在，会把 GNN / solver 后验分数关联进 audit
SCORED_SOLVED_FILE = os.path.join(SCRIPT_DIR, "gnn_scored_solved_glyphs.json")

OUTPUT_FILE = os.path.join(SCRIPT_DIR, "primitive_sampler_audit_report.json")
PREVIEW_DIR = os.path.join(SCRIPT_DIR, "primitive_sampler_audit_previews")

RANDOM_SEED = 42

CANVAS_SIZE = 400.0

# 预览数量
PREVIEW_RANDOM_COUNT = 24
PREVIEW_LOW_DIVERSITY_COUNT = 16
PREVIEW_HIGH_SHAPE20_COUNT = 16
PREVIEW_MISSING_PRIMITIVE_COUNT = 16
PREVIEW_MISSING_LAYOUT_COUNT = 16

CLEAR_PREVIEW_DIR = True

# 宽度渲染
DEFAULT_RENDER_WIDTH_PX = 8.0
MIN_RENDER_WIDTH_PX = 3.0
MAX_RENDER_WIDTH_PX = 26.0

WIDTH_TOKEN_TO_PX = {
    0: 6.0,
    1: 10.0,
    2: 14.0,
    3: 18.0,
}

# shape_code=20 暂时按“结构直线 primitive”看待
STRUCTURAL_SHAPE_CODES = {20, "20"}

# 如果 candidate 没有 layout_prior，使用 fallback layout 预览
USE_FALLBACK_LAYOUT_IF_MISSING = True


# =========================================================
# 🧮 基础工具
# =========================================================
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


def safe_filename(s):
    s = str(s)
    bad = ["/", "\\", ":", "*", "?", "\"", "<", ">", "|", " ", "\n", "\t"]
    for b in bad:
        s = s.replace(b, "_")
    return s[:160]


def entropy_from_counter(counter):
    total = sum(counter.values())
    if total <= 0:
        return 0.0

    ent = 0.0
    for v in counter.values():
        p = v / total
        if p > 0:
            ent -= p * math.log(p + 1e-12)

    return float(ent)


def normalize_angle_pi(rad):
    a = float(rad)

    while a <= -math.pi:
        a += 2.0 * math.pi
    while a > math.pi:
        a -= 2.0 * math.pi

    return a


def deg_to_rad(deg):
    return float(deg) * math.pi / 180.0


def to_norm_xy(xy):
    arr = np.asarray(xy, dtype=np.float32)

    if arr.ndim != 1 or arr.shape[0] < 2:
        return None

    arr = arr[:2]

    if np.max(np.abs(arr)) > 1.5:
        arr = arr / float(CANVAS_SIZE)

    return arr.astype(np.float32)


def to_px_xy(xy):
    arr = np.asarray(xy, dtype=np.float32)

    if arr.ndim != 1 or arr.shape[0] < 2:
        return None

    arr = arr[:2]

    if np.max(np.abs(arr)) <= 1.5:
        arr = arr * float(CANVAS_SIZE)

    return arr.astype(np.float32)


# =========================================================
# 📦 输入读取
# =========================================================
def get_candidate_list(data):
    if isinstance(data, list):
        return data

    candidate_keys = [
        "glyph_candidates",
        "glyph_candidates_with_primitives",
        "candidates",
        "sampled_candidates",
        "generated_candidates",
        "primitive_candidates",
        "sampled_topologies",
        "solved_glyph_candidates",
    ]

    for k in candidate_keys:
        if k in data and isinstance(data[k], list):
            return data[k]

    # 兜底：找第一个 list[dict]
    for k, v in data.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            print(f"[WARN] 未识别标准 candidate key，使用字段: {k}")
            return v

    return []


def get_candidate_id(candidate, idx):
    for k in [
        "generated_glyph_id",
        "glyph_candidate_id",
        "candidate_id",
        "grammar_sample_id",
        "sample_id",
        "layout_template_id",
        "source_sample_id",
        "glyph_uid",
    ]:
        if isinstance(candidate, dict) and candidate.get(k, ""):
            return str(candidate[k])

    return f"candidate_{idx:05d}"


def get_nodes(candidate):
    for k in [
        "nodes",
        "primitive_nodes",
        "layout_nodes",
        "solved_nodes",
        "final_nodes",
        "optimized_nodes",
    ]:
        if k in candidate and isinstance(candidate[k], list):
            return candidate[k]

    return []


def get_edges(candidate):
    if not isinstance(candidate, dict):
        return []

    topo = candidate.get("topology", {})

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

    if "edges" in candidate and isinstance(candidate["edges"], list):
        return candidate["edges"]

    return []


def node_id_of(node, fallback):
    return safe_int(node.get("node_id", node.get("source_node_id", fallback)), fallback)


def get_shape_code(node):
    return node.get("shape_code", node.get("shape_token", None))


def get_shape_code_int(node, default=-1):
    return safe_int(node.get("shape_code", node.get("shape_token", default)), default)


def get_width_token(node):
    return safe_int(node.get("width_token", 0), 0)


def get_variant_id(node):
    return safe_int(node.get("variant_id", 0), 0)


# =========================================================
# 🧱 layout 属性读取
# =========================================================
def read_center_norm(node):
    if "layout_prior" in node and isinstance(node["layout_prior"], dict):
        lp = node["layout_prior"]

        if "center_norm" in lp:
            p = to_norm_xy(lp["center_norm"])
            if p is not None:
                return p

        if "center_px" in lp:
            p = to_norm_xy(lp["center_px"])
            if p is not None:
                return p

    for k in ["center_norm", "center"]:
        if k in node:
            p = to_norm_xy(node[k])
            if p is not None:
                return p

    for k in ["center_px", "center_xy_px"]:
        if k in node:
            p = to_norm_xy(node[k])
            if p is not None:
                return p

    for pair in [
        ("cx_norm", "cy_norm"),
        ("center_x_norm", "center_y_norm"),
        ("cx", "cy"),
        ("center_x", "center_y"),
        ("x", "y"),
    ]:
        kx, ky = pair
        if kx in node and ky in node:
            p = np.array([safe_float(node[kx]), safe_float(node[ky])], dtype=np.float32)
            return to_norm_xy(p)

    # p0 / p3 兜底
    p0 = None
    p3 = None

    for k in ["p0_norm", "start_norm"]:
        if k in node:
            p0 = to_norm_xy(node[k])
            break

    for k in ["p3_norm", "end_norm"]:
        if k in node:
            p3 = to_norm_xy(node[k])
            break

    if p0 is not None and p3 is not None:
        return ((p0 + p3) * 0.5).astype(np.float32)

    for parent in ["solver_priors", "solver_params", "params", "final_params", "optimized_params"]:
        if parent in node and isinstance(node[parent], dict):
            p = read_center_norm(node[parent])
            if p is not None:
                return p

    return None


def read_rotation_rad(node):
    if "layout_prior" in node and isinstance(node["layout_prior"], dict):
        lp = node["layout_prior"]

        if "rotation_rad" in lp:
            return normalize_angle_pi(safe_float(lp["rotation_rad"], 0.0))

        if "rotation_deg" in lp:
            return normalize_angle_pi(deg_to_rad(safe_float(lp["rotation_deg"], 0.0)))

    for k in ["rotation_rad", "theta", "theta_rad", "angle_rad"]:
        if k in node:
            return normalize_angle_pi(safe_float(node[k], 0.0))

    for k in ["rotation_deg", "theta_deg", "angle_deg"]:
        if k in node:
            return normalize_angle_pi(deg_to_rad(safe_float(node[k], 0.0)))

    # p0 / p3 兜底
    p0 = None
    p3 = None

    for k in ["p0_norm", "start_norm", "p0_px", "start_px"]:
        if k in node:
            p0 = to_norm_xy(node[k])
            break

    for k in ["p3_norm", "end_norm", "p3_px", "end_px"]:
        if k in node:
            p3 = to_norm_xy(node[k])
            break

    if p0 is not None and p3 is not None:
        d = p3 - p0
        if np.linalg.norm(d) > 1e-8:
            return float(math.atan2(d[1], d[0]))

    for parent in ["solver_priors", "solver_params", "params", "final_params", "optimized_params"]:
        if parent in node and isinstance(node[parent], dict):
            return read_rotation_rad(node[parent])

    return 0.0


def read_length_norm(node):
    if "layout_prior" in node and isinstance(node["layout_prior"], dict):
        lp = node["layout_prior"]

        for k in ["length_norm", "scale_norm", "arc_length_norm", "chord_length_norm"]:
            if k in lp:
                v = safe_float(lp[k], 0.25)
                if abs(v) > 1.5:
                    v = v / CANVAS_SIZE
                return clamp(v, 0.02, 1.5)

    if "solver_priors" in node and isinstance(node["solver_priors"], dict):
        sp = node["solver_priors"]

        for k in ["length_prior_norm", "scale_init_norm", "scale_norm"]:
            if k in sp:
                v = safe_float(sp[k], 0.25)
                if abs(v) > 1.5:
                    v = v / CANVAS_SIZE
                return clamp(v, 0.02, 1.5)

    for k in ["length_norm", "scale_norm", "arc_length_norm", "chord_length_norm"]:
        if k in node:
            v = safe_float(node[k], 0.25)
            if abs(v) > 1.5:
                v = v / CANVAS_SIZE
            return clamp(v, 0.02, 1.5)

    for k in ["actual_length_px", "target_length_px", "length_px", "scale_px", "scale"]:
        if k in node:
            v = safe_float(node[k], 100.0)
            if abs(v) > 1.5:
                v = v / CANVAS_SIZE
            return clamp(v, 0.02, 1.5)

    # p0 / p3 兜底
    p0 = None
    p3 = None

    for k in ["p0_norm", "start_norm", "p0_px", "start_px"]:
        if k in node:
            p0 = to_norm_xy(node[k])
            break

    for k in ["p3_norm", "end_norm", "p3_px", "end_px"]:
        if k in node:
            p3 = to_norm_xy(node[k])
            break

    if p0 is not None and p3 is not None:
        d = float(np.linalg.norm(p3 - p0))
        return clamp(d, 0.02, 1.5)

    return 0.25


def has_real_layout_info(node):
    if "layout_prior" in node and isinstance(node["layout_prior"], dict):
        lp = node["layout_prior"]
        if "center_norm" in lp or "center_px" in lp:
            return True

    for k in [
        "center_norm",
        "center_px",
        "center",
        "cx_norm",
        "cy_norm",
        "p0_norm",
        "p3_norm",
        "p0_px",
        "p3_px",
    ]:
        if k in node:
            return True

    if "solver_priors" in node and isinstance(node["solver_priors"], dict):
        sp = node["solver_priors"]
        if sp.get("center_norm", None) is not None:
            return True

    return False


# =========================================================
# 📚 primitive library 读取
# =========================================================
def load_primitive_library(corpus_data):
    if not isinstance(corpus_data, dict):
        return {}

    lib = corpus_data.get("stroke_primitive_library", {})

    if isinstance(lib, list):
        out = {}
        for item in lib:
            if not isinstance(item, dict):
                continue

            sid = item.get("shape_code", item.get("cluster_id", item.get("id", None)))
            if sid is None:
                continue

            out[str(sid)] = item

        return out

    if isinstance(lib, dict):
        return lib

    return {}


def entry_to_polyline(entry):
    """
    从 primitive_ref 或 primitive_library entry 中抽取 local polyline。
    返回 None 表示找不到。
    """
    if entry is None:
        return None

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

    # 递归 prototype
    for parent in ["prototype", "primitive", "data"]:
        if parent in entry and isinstance(entry[parent], dict):
            p = entry_to_polyline(entry[parent])
            if p is not None:
                return p

    return None


def get_primitive_polyline_for_node(node, primitive_library):
    """
    返回:
      polyline or None,
      source string
    """
    if "primitive_ref" in node and isinstance(node["primitive_ref"], dict):
        p = entry_to_polyline(node["primitive_ref"])
        if p is not None:
            return p, "node.primitive_ref"

    shape_code = get_shape_code(node)

    if shape_code is not None:
        for key in [str(shape_code), shape_code]:
            if key in primitive_library:
                p = entry_to_polyline(primitive_library[key])
                if p is not None:
                    return p, "corpus.stroke_primitive_library"

    return None, "missing"


def normalize_local_polyline(polyline):
    """
    把 primitive local polyline 规范到大致 x ∈ [-0.5, 0.5]。
    y 按 x_range 同比例缩放。
    """
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

    x_range = xmax - xmin

    if x_range < 1e-6:
        return np.asarray([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32)

    x_mid = 0.5 * (xmin + xmax)
    y_mid = 0.5 * (ymin + ymax)

    arr[:, 0] = (arr[:, 0] - x_mid) / x_range
    arr[:, 1] = (arr[:, 1] - y_mid) / x_range

    # 防止极端 primitive 把 preview 撑爆
    arr[:, 0] = np.clip(arr[:, 0], -1.0, 1.0)
    arr[:, 1] = np.clip(arr[:, 1], -1.0, 1.0)

    return arr.astype(np.float32)


def apply_variant(polyline, variant_id):
    pts = np.asarray(polyline, dtype=np.float32).copy()
    v = int(variant_id) % 4

    # 这里是用于预览的近似变体，不影响原始数据
    if v == 0:
        return pts

    if v == 1:
        # x 方向反转：y(x) -> y(-x)
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


def transform_local_to_world_px(local_poly, center_norm, rotation_rad, length_norm):
    local = np.asarray(local_poly, dtype=np.float32)
    center = np.asarray(center_norm, dtype=np.float32) * CANVAS_SIZE

    scale = float(length_norm) * CANVAS_SIZE

    c = math.cos(rotation_rad)
    s = math.sin(rotation_rad)

    R = np.asarray([[c, -s], [s, c]], dtype=np.float32)

    pts = local * scale
    pts = pts @ R.T
    pts = pts + center[None, :]

    return pts.astype(np.float32)


# =========================================================
# 🔁 fallback layout
# =========================================================
def build_fallback_layout(candidate, nodes):
    """
    当 candidate 没有 layout_prior 时，为 preview 生成一个简单布局。
    注意：这只是为了看 primitive 分配，不代表真实 layout generator。
    """
    n = len(nodes)
    edges = get_edges(candidate)

    centers = {}

    if n == 0:
        return centers

    # 优先使用 networkx spring layout
    try:
        import networkx as nx

        G = nx.Graph()
        for i, node in enumerate(nodes):
            G.add_node(node_id_of(node, i))

        for e in edges:
            u = safe_int(e.get("u", -1), -1)
            v = safe_int(e.get("v", -1), -1)
            if u >= 0 and v >= 0:
                G.add_edge(u, v)

        pos = nx.spring_layout(G, seed=RANDOM_SEED)

        xs = np.asarray([p[0] for p in pos.values()], dtype=np.float32)
        ys = np.asarray([p[1] for p in pos.values()], dtype=np.float32)

        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())

        for nid, p in pos.items():
            x = (p[0] - xmin) / max(xmax - xmin, 1e-6)
            y = (p[1] - ymin) / max(ymax - ymin, 1e-6)

            x = 0.18 + 0.64 * x
            y = 0.18 + 0.64 * y

            centers[nid] = np.asarray([x, y], dtype=np.float32)

        return centers

    except Exception:
        pass

    # circle fallback
    for i, node in enumerate(nodes):
        nid = node_id_of(node, i)
        ang = 2.0 * math.pi * i / max(1, n)
        centers[nid] = np.asarray(
            [
                0.5 + 0.28 * math.cos(ang),
                0.5 + 0.28 * math.sin(ang),
            ],
            dtype=np.float32,
        )

    return centers


def infer_fallback_rotation(node_id, fallback_centers, edges):
    c0 = fallback_centers.get(node_id, None)
    if c0 is None:
        return 0.0

    neighbor_vecs = []

    for e in edges:
        u = safe_int(e.get("u", -1), -1)
        v = safe_int(e.get("v", -1), -1)

        if u == node_id and v in fallback_centers:
            neighbor_vecs.append(fallback_centers[v] - c0)

        elif v == node_id and u in fallback_centers:
            neighbor_vecs.append(fallback_centers[u] - c0)

    if len(neighbor_vecs) > 0:
        d = np.mean(np.asarray(neighbor_vecs, dtype=np.float32), axis=0)
        if np.linalg.norm(d) > 1e-8:
            return float(math.atan2(d[1], d[0]))

    return 0.0


# =========================================================
# 🎨 渲染信息构造
# =========================================================
def extract_render_width_px(node):
    for key in [
        "width_px",
        "stroke_width_px",
        "render_width_px",
        "width_mean_px",
        "actual_width_px",
        "target_width_px",
    ]:
        if key in node:
            return clamp(safe_float(node[key], DEFAULT_RENDER_WIDTH_PX), MIN_RENDER_WIDTH_PX, MAX_RENDER_WIDTH_PX)

    for parent in ["layout_prior", "solver_priors", "primitive_ref", "primitive_assignment"]:
        if parent in node and isinstance(node[parent], dict):
            for key in [
                "width_px",
                "stroke_width_px",
                "render_width_px",
                "width_mean_px",
                "actual_width_px",
                "target_width_px",
            ]:
                if key in node[parent]:
                    return clamp(
                        safe_float(node[parent][key], DEFAULT_RENDER_WIDTH_PX),
                        MIN_RENDER_WIDTH_PX,
                        MAX_RENDER_WIDTH_PX,
                    )

    width_token = get_width_token(node)

    return clamp(WIDTH_TOKEN_TO_PX.get(width_token, DEFAULT_RENDER_WIDTH_PX), MIN_RENDER_WIDTH_PX, MAX_RENDER_WIDTH_PX)


def build_render_nodes(candidate, primitive_library):
    nodes = get_nodes(candidate)
    edges = get_edges(candidate)

    fallback_centers = build_fallback_layout(candidate, nodes)

    render_nodes = []
    missing_layout_count = 0
    missing_primitive_count = 0

    for i, node in enumerate(nodes):
        nid = node_id_of(node, i)

        center = read_center_norm(node)
        layout_missing = False

        if center is None:
            layout_missing = True
            missing_layout_count += 1

            if USE_FALLBACK_LAYOUT_IF_MISSING:
                center = fallback_centers.get(nid, np.asarray([0.5, 0.5], dtype=np.float32))
            else:
                center = np.asarray([0.5, 0.5], dtype=np.float32)

        rotation = read_rotation_rad(node)

        if layout_missing:
            rotation = infer_fallback_rotation(nid, fallback_centers, edges)

        length = read_length_norm(node)

        poly, poly_source = get_primitive_polyline_for_node(node, primitive_library)

        primitive_missing = False

        if poly is None:
            primitive_missing = True
            missing_primitive_count += 1
            poly = np.asarray([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32)

        poly = normalize_local_polyline(poly)
        poly = apply_variant(poly, get_variant_id(node))
        world_poly = transform_local_to_world_px(poly, center, rotation, length)

        render_nodes.append({
            "node_index": i,
            "node_id": nid,
            "shape_code": get_shape_code(node),
            "shape_code_int": get_shape_code_int(node, -1),
            "width_token": get_width_token(node),
            "variant_id": get_variant_id(node),
            "center_norm": center,
            "rotation_rad": rotation,
            "length_norm": length,
            "width_px": extract_render_width_px(node),
            "world_polyline_px": world_poly,
            "layout_missing": layout_missing,
            "primitive_missing": primitive_missing,
            "primitive_source": poly_source,
            "assignment_source": (
                node.get("primitive_assignment", {}).get("assignment_source", "unknown")
                if isinstance(node.get("primitive_assignment", {}), dict)
                else "unknown"
            ),
        })

    return render_nodes, {
        "missing_layout_count": missing_layout_count,
        "missing_primitive_count": missing_primitive_count,
    }


def estimate_black_fill_ratio(render_nodes):
    """
    近似面积：sum(polyline_length * width) / bbox_area
    """
    if len(render_nodes) == 0:
        return 0.0

    all_pts = []
    area = 0.0

    for rn in render_nodes:
        pts = rn["world_polyline_px"]
        if pts is None or len(pts) < 2:
            continue

        all_pts.append(pts)

        seg = pts[1:] - pts[:-1]
        length = float(np.sum(np.linalg.norm(seg, axis=1)))
        area += length * float(rn["width_px"])

    if len(all_pts) == 0:
        return 0.0

    pts_all = np.concatenate(all_pts, axis=0)

    xmin, ymin = np.min(pts_all, axis=0)
    xmax, ymax = np.max(pts_all, axis=0)

    bbox_area = max(1.0, float((xmax - xmin) * (ymax - ymin)))

    return float(clamp(area / bbox_area, 0.0, 5.0))


# =========================================================
# 📊 下游分数读取，可选
# =========================================================
def load_downstream_scores(path):
    data = load_json(path, required=False)

    if data is None:
        return {}

    score_items = []

    for k in [
        "scored_items_by_combined",
        "scored_items_by_gnn",
        "scored_items_sorted",
        "solver_allowed_items_by_gnn",
    ]:
        if k in data and isinstance(data[k], list):
            score_items.extend(data[k])

    out = {}

    for item in score_items:
        if not isinstance(item, dict):
            continue

        cid = item.get("candidate_id", None)
        if cid is None:
            continue

        if cid not in out:
            out[cid] = item

    return out


# =========================================================
# 📈 Audit 统计
# =========================================================
def audit_candidates(candidates, primitive_library, downstream_scores):
    global_shape_counter = Counter()
    global_width_counter = Counter()
    global_variant_counter = Counter()
    assignment_source_counter = Counter()
    primitive_source_counter = Counter()

    per_shape_downstream = defaultdict(list)

    candidate_metrics = []

    total_nodes = 0
    total_edges = 0
    total_missing_primitive = 0
    total_missing_layout = 0
    total_shape20 = 0

    for idx, cand in enumerate(candidates):
        cid = get_candidate_id(cand, idx)
        nodes = get_nodes(cand)
        edges = get_edges(cand)

        render_nodes, rn_stat = build_render_nodes(cand, primitive_library)

        n_nodes = len(nodes)
        n_edges = len(edges)

        total_nodes += n_nodes
        total_edges += n_edges

        shape_counter = Counter()
        width_counter = Counter()
        variant_counter = Counter()

        for rn in render_nodes:
            shape = rn["shape_code_int"]
            width = rn["width_token"]
            variant = rn["variant_id"]

            shape_counter[shape] += 1
            width_counter[width] += 1
            variant_counter[variant] += 1

            global_shape_counter[shape] += 1
            global_width_counter[width] += 1
            global_variant_counter[variant] += 1

            assignment_source_counter[rn["assignment_source"]] += 1
            primitive_source_counter[rn["primitive_source"]] += 1

            if shape in STRUCTURAL_SHAPE_CODES:
                total_shape20 += 1

        missing_primitive = rn_stat["missing_primitive_count"]
        missing_layout = rn_stat["missing_layout_count"]

        total_missing_primitive += missing_primitive
        total_missing_layout += missing_layout

        unique_shape_count = len(shape_counter)
        shape20_count = shape_counter.get(20, 0)
        shape20_ratio = shape20_count / max(1, n_nodes)

        width0_ratio = width_counter.get(0, 0) / max(1, n_nodes)
        unique_width_count = len(width_counter)

        fill_ratio = estimate_black_fill_ratio(render_nodes)

        downstream = downstream_scores.get(cid, {})
        gnn_score = downstream.get("gnn_layout_realness_score", None)
        combined_score = downstream.get("combined_score", None)
        solver_allowed = downstream.get("solver_allowed", None)
        quality_status = downstream.get("quality_status", None)

        if gnn_score is not None:
            for shape in shape_counter:
                per_shape_downstream[shape].append({
                    "candidate_id": cid,
                    "gnn_layout_realness_score": safe_float(gnn_score, 0.0),
                    "combined_score": safe_float(combined_score, 0.0),
                    "solver_allowed": solver_allowed,
                    "quality_status": quality_status,
                })

        candidate_metrics.append({
            "candidate_index": idx,
            "candidate_id": cid,
            "num_nodes": n_nodes,
            "num_edges": n_edges,

            "unique_shape_count": unique_shape_count,
            "unique_width_count": unique_width_count,

            "shape_entropy": entropy_from_counter(shape_counter),
            "width_entropy": entropy_from_counter(width_counter),

            "shape20_count": shape20_count,
            "shape20_ratio": round(float(shape20_ratio), 6),
            "width0_ratio": round(float(width0_ratio), 6),

            "missing_primitive_count": missing_primitive,
            "missing_layout_count": missing_layout,

            "black_render_fill_ratio": round(float(fill_ratio), 6),

            "shape_counter": dict(shape_counter),
            "width_counter": dict(width_counter),
            "variant_counter": dict(variant_counter),

            "gnn_layout_realness_score": None if gnn_score is None else safe_float(gnn_score),
            "combined_score": None if combined_score is None else safe_float(combined_score),
            "solver_allowed": solver_allowed,
            "quality_status": quality_status,
        })

    avg_unique_shape = float(np.mean([m["unique_shape_count"] for m in candidate_metrics])) if candidate_metrics else 0.0
    avg_shape_entropy = float(np.mean([m["shape_entropy"] for m in candidate_metrics])) if candidate_metrics else 0.0
    avg_fill_ratio = float(np.mean([m["black_render_fill_ratio"] for m in candidate_metrics])) if candidate_metrics else 0.0

    total_candidates = len(candidate_metrics)

    summary = {
        "candidate_count": total_candidates,
        "total_nodes": total_nodes,
        "total_edges": total_edges,

        "missing_primitive_nodes": total_missing_primitive,
        "missing_primitive_rate": round(total_missing_primitive / max(1, total_nodes), 6),

        "missing_layout_nodes": total_missing_layout,
        "missing_layout_rate": round(total_missing_layout / max(1, total_nodes), 6),

        "shape20_nodes": total_shape20,
        "shape20_ratio": round(total_shape20 / max(1, total_nodes), 6),

        "avg_unique_shape_per_candidate": round(avg_unique_shape, 6),
        "avg_shape_entropy_per_candidate": round(avg_shape_entropy, 6),
        "avg_black_render_fill_ratio": round(avg_fill_ratio, 6),

        "shape_code_hist": dict(global_shape_counter),
        "width_token_hist": dict(global_width_counter),
        "variant_id_hist": dict(global_variant_counter),
        "assignment_source_hist": dict(assignment_source_counter),
        "primitive_source_hist": dict(primitive_source_counter),
    }

    # 下游关联分析
    by_shape_downstream_summary = {}

    for shape, items in per_shape_downstream.items():
        if len(items) == 0:
            continue

        gnn_vals = np.asarray([x["gnn_layout_realness_score"] for x in items], dtype=np.float32)
        comb_vals = np.asarray([x["combined_score"] for x in items], dtype=np.float32)

        allowed_vals = [
            x["solver_allowed"]
            for x in items
            if x["solver_allowed"] is not None
        ]

        if len(allowed_vals) > 0:
            allowed_rate = sum(1 for x in allowed_vals if x) / len(allowed_vals)
        else:
            allowed_rate = None

        by_shape_downstream_summary[str(shape)] = {
            "candidate_appear_count": len(items),
            "mean_gnn_score": round(float(np.mean(gnn_vals)), 6),
            "median_gnn_score": round(float(np.median(gnn_vals)), 6),
            "mean_combined_score": round(float(np.mean(comb_vals)), 6),
            "solver_allowed_rate": None if allowed_rate is None else round(float(allowed_rate), 6),
        }

    return summary, candidate_metrics, by_shape_downstream_summary


def build_warnings(summary):
    warnings = []

    if summary["missing_primitive_rate"] > 0.0:
        warnings.append(
            f"存在 primitive 缺失: missing_primitive_rate={summary['missing_primitive_rate']}. "
            "这会导致部分 stroke 用 fallback line 渲染。"
        )

    if summary["missing_layout_rate"] > 0.5:
        warnings.append(
            f"layout_prior 覆盖率低: missing_layout_rate={summary['missing_layout_rate']}. "
            "primitive preview 使用了 fallback layout，不能严格判断 layout generator。"
        )
    elif summary["missing_layout_rate"] > 0.0:
        warnings.append(
            f"部分 node 缺少 layout_prior: missing_layout_rate={summary['missing_layout_rate']}."
        )

    if summary["shape20_ratio"] > 0.65:
        warnings.append(
            f"shape_code=20 占比偏高: shape20_ratio={summary['shape20_ratio']}. "
            "如果它是直线结构 primitive，这不一定是错误，但需要看黑色渲染是否过于单调。"
        )

    width_hist = summary.get("width_token_hist", {})
    total_width = sum(width_hist.values())

    if total_width > 0:
        width0_ratio = width_hist.get(0, width_hist.get("0", 0)) / total_width
        if width0_ratio > 0.90:
            warnings.append(
                f"width_token=0 占比过高: {round(width0_ratio, 6)}. "
                "可能导致黑色宽度渲染风格过于单一。"
            )

    if summary["avg_unique_shape_per_candidate"] < 2.0:
        warnings.append(
            f"每个 glyph 平均 unique shape 偏低: {summary['avg_unique_shape_per_candidate']}. "
            "primitive sampler 可能过度塌缩。"
        )

    return warnings


# =========================================================
# 🎨 Preview
# =========================================================
def draw_layout_axis(ax, candidate, render_nodes):
    edges = get_edges(candidate)

    centers = {
        rn["node_id"]: rn["center_norm"] * CANVAS_SIZE
        for rn in render_nodes
    }

    # 先画 topology center-to-center
    for e in edges:
        u = safe_int(e.get("u", -1), -1)
        v = safe_int(e.get("v", -1), -1)

        if u not in centers or v not in centers:
            continue

        pu = centers[u]
        pv = centers[v]
        pm = 0.5 * (pu + pv)

        jt = str(e.get("j_type", "?"))

        if jt == "T":
            color = "#D81B60"
            ls = "--"
        elif jt == "X":
            color = "#1E88E5"
            ls = ":"
        else:
            color = "#666666"
            ls = "--"

        ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color=color, lw=1.0, ls=ls, alpha=0.6)
        ax.text(pm[0], pm[1], jt, fontsize=7, color=color)

    for rn in render_nodes:
        c = rn["center_norm"] * CANVAS_SIZE
        theta = rn["rotation_rad"]
        length = rn["length_norm"] * CANVAS_SIZE

        dx = math.cos(theta) * length * 0.5
        dy = math.sin(theta) * length * 0.5

        p0 = c - np.asarray([dx, dy], dtype=np.float32)
        p1 = c + np.asarray([dx, dy], dtype=np.float32)

        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="black", lw=2.2, alpha=0.85)
        ax.scatter([c[0]], [c[1]], color="black", s=18, zorder=5)
        ax.text(c[0] + 4, c[1] - 4, f"N{rn['node_id']}", fontsize=8, color="black")

    ax.set_title("layout axis + topology", fontsize=9)


def draw_colored_primitive(ax, render_nodes):
    colors = [
        "#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA",
        "#00897B", "#6D4C41", "#3949AB", "#D81B60", "#7CB342",
        "#546E7A", "#F4511E",
    ]

    for i, rn in enumerate(render_nodes):
        pts = rn["world_polyline_px"]
        color = colors[i % len(colors)]

        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2.8, alpha=0.95)
        ax.scatter([pts[0, 0]], [pts[0, 1]], color="green", s=20, zorder=6)
        ax.scatter([pts[-1, 0]], [pts[-1, 1]], color="red", s=20, zorder=6)

        c = rn["center_norm"] * CANVAS_SIZE

        label = f"S{rn['shape_code']} V{rn['variant_id']} W{rn['width_token']}"
        if rn["primitive_missing"]:
            label += " !P"
        if rn["layout_missing"]:
            label += " !L"

        ax.text(c[0] + 4, c[1] + 8, label, fontsize=7, color=color)

    ax.set_title("colored primitive assignment", fontsize=9)


def draw_black_width(ax, render_nodes):
    for rn in render_nodes:
        pts = rn["world_polyline_px"]
        w = rn["width_px"]

        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color="black",
            lw=w,
            alpha=1.0,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    ax.set_title("black width render", fontsize=9)


def render_candidate_preview(candidate, primitive_library, metric, out_path):
    render_nodes, rn_stat = build_render_nodes(candidate, primitive_library)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    ax_axis, ax_color, ax_black = axes

    draw_layout_axis(ax_axis, candidate, render_nodes)
    draw_colored_primitive(ax_color, render_nodes)
    draw_black_width(ax_black, render_nodes)

    title = (
        f"{metric['candidate_id']} | "
        f"N={metric['num_nodes']} E={metric['num_edges']} | "
        f"unique_shape={metric['unique_shape_count']} | "
        f"shape20={metric['shape20_ratio']:.2f} | "
        f"missingP={metric['missing_primitive_count']} | "
        f"missingL={metric['missing_layout_count']} | "
        f"fill={metric['black_render_fill_ratio']:.2f}"
    )

    fig.suptitle(title, fontsize=10)

    for ax in axes:
        ax.set_xlim(0, CANVAS_SIZE)
        ax.set_ylim(CANVAS_SIZE, 0)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_preview_group(name, items, candidates, primitive_library, max_count):
    out_dir = os.path.join(PREVIEW_DIR, name)
    ensure_dir(out_dir)

    for rank, metric in enumerate(items[:max_count]):
        idx = metric["candidate_index"]
        cand = candidates[idx]

        out_path = os.path.join(
            out_dir,
            f"{name}_{rank:03d}_{safe_filename(metric['candidate_id'])}.png",
        )

        render_candidate_preview(cand, primitive_library, metric, out_path)


def render_previews(candidates, primitive_library, candidate_metrics):
    reset_dir(PREVIEW_DIR)

    rng = random.Random(RANDOM_SEED)

    random_items = list(candidate_metrics)
    rng.shuffle(random_items)

    low_diversity = sorted(
        candidate_metrics,
        key=lambda x: (x["unique_shape_count"], x["shape_entropy"], -x["shape20_ratio"]),
    )

    high_shape20 = sorted(
        candidate_metrics,
        key=lambda x: (x["shape20_ratio"], x["shape20_count"]),
        reverse=True,
    )

    missing_primitive = [
        x for x in candidate_metrics
        if x["missing_primitive_count"] > 0
    ]

    missing_primitive = sorted(
        missing_primitive,
        key=lambda x: x["missing_primitive_count"],
        reverse=True,
    )

    missing_layout = [
        x for x in candidate_metrics
        if x["missing_layout_count"] > 0
    ]

    missing_layout = sorted(
        missing_layout,
        key=lambda x: x["missing_layout_count"],
        reverse=True,
    )

    # 如果存在下游 GNN 分数，额外输出 top/bottom
    scored = [
        x for x in candidate_metrics
        if x.get("gnn_layout_realness_score", None) is not None
    ]

    scored_top = sorted(
        scored,
        key=lambda x: x["gnn_layout_realness_score"],
        reverse=True,
    )

    scored_bottom = sorted(
        scored,
        key=lambda x: x["gnn_layout_realness_score"],
    )

    print("\n" + "=" * 80)
    print("🖼️ Rendering Primitive Sampler Audit Previews")
    print("=" * 80)

    render_preview_group("random", random_items, candidates, primitive_library, PREVIEW_RANDOM_COUNT)
    print(f"  random: {min(len(random_items), PREVIEW_RANDOM_COUNT)}")

    render_preview_group("low_diversity", low_diversity, candidates, primitive_library, PREVIEW_LOW_DIVERSITY_COUNT)
    print(f"  low_diversity: {min(len(low_diversity), PREVIEW_LOW_DIVERSITY_COUNT)}")

    render_preview_group("high_shape20", high_shape20, candidates, primitive_library, PREVIEW_HIGH_SHAPE20_COUNT)
    print(f"  high_shape20: {min(len(high_shape20), PREVIEW_HIGH_SHAPE20_COUNT)}")

    render_preview_group("missing_primitive", missing_primitive, candidates, primitive_library, PREVIEW_MISSING_PRIMITIVE_COUNT)
    print(f"  missing_primitive: {min(len(missing_primitive), PREVIEW_MISSING_PRIMITIVE_COUNT)}")

    render_preview_group("missing_layout", missing_layout, candidates, primitive_library, PREVIEW_MISSING_LAYOUT_COUNT)
    print(f"  missing_layout: {min(len(missing_layout), PREVIEW_MISSING_LAYOUT_COUNT)}")

    if len(scored_top) > 0:
        render_preview_group("downstream_gnn_top", scored_top, candidates, primitive_library, 16)
        render_preview_group("downstream_gnn_bottom", scored_bottom, candidates, primitive_library, 16)
        print(f"  downstream_gnn_top/bottom: {min(len(scored_top), 16)}")

    print(f"  preview_dir: {PREVIEW_DIR}")
    print("=" * 80 + "\n")


# =========================================================
# 🚀 main
# =========================================================
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\n" + "=" * 80)
    print("🚀 Primitive Sampler Audit")
    print("=" * 80)
    print(f"  candidate_file: {CANDIDATE_FILE}")
    print(f"  corpus_file:    {CORPUS_FILE}")
    print(f"  scored_file:    {SCORED_SOLVED_FILE}")
    print(f"  output_file:    {OUTPUT_FILE}")
    print(f"  preview_dir:    {PREVIEW_DIR}")
    print("=" * 80)

    candidate_data = load_json(CANDIDATE_FILE, required=True)
    corpus_data = load_json(CORPUS_FILE, required=False)

    candidates = get_candidate_list(candidate_data)

    if len(candidates) == 0:
        raise RuntimeError(
            "没有读取到 candidate。请检查 glyph_candidates_with_primitives.json 的 key。"
        )

    primitive_library = load_primitive_library(corpus_data)

    downstream_scores = load_downstream_scores(SCORED_SOLVED_FILE)

    print("\n[Input]")
    print(f"  candidate_count: {len(candidates)}")
    print(f"  primitive_library_count: {len(primitive_library)}")
    print(f"  downstream_score_count: {len(downstream_scores)}")

    summary, candidate_metrics, by_shape_downstream = audit_candidates(
        candidates=candidates,
        primitive_library=primitive_library,
        downstream_scores=downstream_scores,
    )

    warnings = build_warnings(summary)

    print("\n" + "=" * 80)
    print("📊 Primitive Sampler Audit Summary")
    print("=" * 80)

    print(f"  candidate_count: {summary['candidate_count']}")
    print(f"  total_nodes: {summary['total_nodes']}")
    print(f"  total_edges: {summary['total_edges']}")
    print(f"  missing_primitive_rate: {summary['missing_primitive_rate']}")
    print(f"  missing_layout_rate: {summary['missing_layout_rate']}")
    print(f"  shape20_ratio: {summary['shape20_ratio']}")
    print(f"  avg_unique_shape_per_candidate: {summary['avg_unique_shape_per_candidate']}")
    print(f"  avg_shape_entropy_per_candidate: {summary['avg_shape_entropy_per_candidate']}")
    print(f"  avg_black_render_fill_ratio: {summary['avg_black_render_fill_ratio']}")

    print("\n[Top Shape Codes]")
    for k, v in Counter(summary["shape_code_hist"]).most_common(15):
        print(f"  shape_code={k}: {v}")

    print("\n[Width Token Hist]")
    for k, v in sorted(summary["width_token_hist"].items(), key=lambda x: str(x[0])):
        print(f"  width_token={k}: {v}")

    print("\n[Variant Hist]")
    for k, v in sorted(summary["variant_id_hist"].items(), key=lambda x: str(x[0])):
        print(f"  variant_id={k}: {v}")

    print("\n[Primitive Source Hist]")
    for k, v in summary["primitive_source_hist"].items():
        print(f"  {k}: {v}")

    if len(warnings) > 0:
        print("\n[Warnings]")
        for w in warnings:
            print(f"  ⚠️ {w}")
    else:
        print("\n[Warnings]")
        print("  ✅ 没有发现明显 collapse / missing 问题。")

    # 选择最容易出问题的 candidate
    low_diversity = sorted(
        candidate_metrics,
        key=lambda x: (x["unique_shape_count"], x["shape_entropy"], -x["shape20_ratio"]),
    )[:10]

    high_shape20 = sorted(
        candidate_metrics,
        key=lambda x: x["shape20_ratio"],
        reverse=True,
    )[:10]

    print("\n[Potential Low-Diversity Candidates]")
    for m in low_diversity:
        print(
            f"  {m['candidate_id']} | "
            f"unique_shape={m['unique_shape_count']} | "
            f"shape20={m['shape20_ratio']:.2f} | "
            f"missingP={m['missing_primitive_count']} | "
            f"missingL={m['missing_layout_count']}"
        )

    print("\n[Potential High-Shape20 Candidates]")
    for m in high_shape20:
        print(
            f"  {m['candidate_id']} | "
            f"shape20={m['shape20_ratio']:.2f} | "
            f"unique_shape={m['unique_shape_count']} | "
            f"fill={m['black_render_fill_ratio']:.2f}"
        )

    output = {
        "schema_version": "primitive_sampler_audit_v1",
        "candidate_file": CANDIDATE_FILE,
        "corpus_file": CORPUS_FILE,
        "scored_solved_file": SCORED_SOLVED_FILE,
        "preview_dir": PREVIEW_DIR,

        "config": {
            "canvas_size": CANVAS_SIZE,
            "width_token_to_px": WIDTH_TOKEN_TO_PX,
            "default_render_width_px": DEFAULT_RENDER_WIDTH_PX,
            "structural_shape_codes": list(STRUCTURAL_SHAPE_CODES),
            "use_fallback_layout_if_missing": USE_FALLBACK_LAYOUT_IF_MISSING,
        },

        "summary": summary,
        "warnings": warnings,
        "candidate_metrics": candidate_metrics,
        "by_shape_downstream_summary": by_shape_downstream,
    }

    save_json(output, OUTPUT_FILE)

    print("\n" + "=" * 80)
    print("💾 Saved Audit Report")
    print("=" * 80)
    print(f"  {OUTPUT_FILE}")

    render_previews(candidates, primitive_library, candidate_metrics)

    print("\n📌 如何看结果：")
    print("  1. 先看 primitive_sampler_audit_previews/random")
    print("  2. 再看 low_diversity：检查 primitive 是否塌缩")
    print("  3. 再看 high_shape20：检查直线 primitive 是否过多")
    print("  4. 如果 missing_primitive 非空，说明 primitive_ref / primitive_library 有缺失")
    print("  5. 如果 missing_layout 很多，说明当前 primitive sampler 输出没有 layout_prior，preview 使用 fallback layout")
    print("  6. 每张图左=layout axis，中=colored primitive，右=black width render")
    print("  7. 如果左边像字、中间/右边不像，主要查 primitive sampler")
    print("  8. 如果左边就不像字，主要查 layout generator")


if __name__ == "__main__":
    main()