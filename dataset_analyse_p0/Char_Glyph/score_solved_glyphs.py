import os
import sys
import json
import math
import copy
from collections import Counter, defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt


# =========================================================
# ⚙️ 路径配置
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 需要和 gnn_layout_discriminator.py 放在同一个目录
GNN_SCRIPT_DIR = SCRIPT_DIR
sys.path.insert(0, GNN_SCRIPT_DIR)

import gnn_layout_discriminator as gld


MODEL_FILE = os.path.join(SCRIPT_DIR, "gnn_layout_discriminator_v2.pt")
SOLVED_FILE = os.path.join(SCRIPT_DIR, "solved_glyph_candidates.json")

OUTPUT_FILE = os.path.join(SCRIPT_DIR, "gnn_scored_solved_glyphs.json")

PREVIEW_DIR = os.path.join(SCRIPT_DIR, "gnn_scored_solved_previews")

TOP_GNN_DIR = os.path.join(PREVIEW_DIR, "top_gnn")
BOTTOM_GNN_DIR = os.path.join(PREVIEW_DIR, "bottom_gnn")
MIDDLE_GNN_DIR = os.path.join(PREVIEW_DIR, "middle_gnn")

TOP_GNN_GOOD_ONLY_DIR = os.path.join(PREVIEW_DIR, "top_gnn_good_only")
TOP_COMBINED_DIR = os.path.join(PREVIEW_DIR, "top_combined")
BAD_BUT_GNN_HIGH_DIR = os.path.join(PREVIEW_DIR, "top_bad_but_gnn_high")

# auto / cuda / mps / cpu
DEVICE_MODE = "auto"

CANVAS_SIZE = 400.0

BATCH_SIZE = 64

TOP_K_PREVIEW = 24
BOTTOM_K_PREVIEW = 16
MIDDLE_K_PREVIEW = 12
GOOD_ONLY_PREVIEW = 24
COMBINED_PREVIEW = 24
BAD_BUT_HIGH_PREVIEW = 16

SAVE_FULL_CANDIDATES = True
SAVE_PREVIEWS = True

# solver gate
GOOD_STATUS_SET = {"good"}
USABLE_STATUS_SET = {"usable_but_rough", "usable"}

USABLE_MAX_JUNCTION_PX = 6.0
USABLE_MAX_ANGLE_DEG = 80.0

# combined score 超参
COMBINED_JUNCTION_WEIGHT = 0.20
COMBINED_ANGLE_WEIGHT = 0.015

# 纯黑宽度渲染默认宽度
DEFAULT_RENDER_WIDTH_PX = 8.0
MIN_RENDER_WIDTH_PX = 3.0
MAX_RENDER_WIDTH_PX = 24.0

# width_token 兜底映射
WIDTH_TOKEN_TO_PX = {
    0: 6.0,
    1: 10.0,
    2: 14.0,
    3: 18.0,
}


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


def stable_sigmoid_np(logits):
    logits = np.asarray(logits, dtype=np.float32)
    logits = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


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


def to_norm_xy(xy, canvas_size=CANVAS_SIZE):
    arr = np.asarray(xy, dtype=np.float32)

    if arr.shape[0] < 2:
        return None

    arr = arr[:2]

    if np.max(np.abs(arr)) > 1.5:
        arr = arr / float(canvas_size)

    return arr.astype(np.float32)


def to_px_xy(xy, canvas_size=CANVAS_SIZE):
    arr = np.asarray(xy, dtype=np.float32)

    if arr.shape[0] < 2:
        return None

    arr = arr[:2]

    if np.max(np.abs(arr)) <= 1.5:
        arr = arr * float(canvas_size)

    return arr.astype(np.float32)


def normalize_angle_pi(rad):
    a = float(rad)

    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi

    return a


def deg_to_rad(deg):
    return float(deg) * math.pi / 180.0


def read_metric_from_quality(q, keys, default=0.0):
    """
    兼容各种 quality_report 字段名。
    """
    if not isinstance(q, dict):
        return default

    for k in keys:
        if k in q:
            return safe_float(q[k], default)

    for parent in [
        "metrics",
        "quality_metrics",
        "final_metrics",
        "best_final_metrics",
        "quality_report",
    ]:
        if parent in q and isinstance(q[parent], dict):
            for k in keys:
                if k in q[parent]:
                    return safe_float(q[parent][k], default)

    return default


# =========================================================
# 📦 solved candidate 读取
# =========================================================
def get_candidate_list(solved_data):
    if isinstance(solved_data, list):
        return solved_data

    if "solved_glyph_candidates" in solved_data:
        return solved_data["solved_glyph_candidates"]

    candidates = []

    for k in [
        "valid_solved_glyph_candidates",
        "usable_solved_glyph_candidates",
        "rejected_solved_glyph_candidates",
    ]:
        if k in solved_data and isinstance(solved_data[k], list):
            candidates.extend(solved_data[k])

    return candidates


def get_candidate_id(candidate, idx):
    for k in [
        "generated_glyph_id",
        "glyph_candidate_id",
        "candidate_id",
        "sample_id",
        "layout_template_id",
    ]:
        if candidate.get(k, ""):
            return str(candidate[k])

    return f"solved_candidate_{idx:05d}"


def get_candidate_edges(candidate):
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


def choose_solved_nodes(candidate):
    """
    打分和渲染时，优先用 solver 后节点。
    """
    for k in [
        "solved_nodes",
        "final_nodes",
        "optimized_nodes",
        "layout_nodes",
    ]:
        if k in candidate and isinstance(candidate[k], list):
            return candidate[k]

    if "nodes" in candidate and isinstance(candidate["nodes"], list):
        return candidate["nodes"]

    return []


def choose_original_nodes(candidate):
    if "nodes" in candidate and isinstance(candidate["nodes"], list):
        return candidate["nodes"]

    return []


# =========================================================
# 🧱 node 标准化
# =========================================================
def extract_center_norm(node):
    if "layout_prior" in node and isinstance(node["layout_prior"], dict):
        lp = node["layout_prior"]
        if "center_norm" in lp:
            return to_norm_xy(lp["center_norm"])
        if "center_px" in lp:
            return to_norm_xy(lp["center_px"])

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

    for key in ["solver_params", "params", "final_params", "optimized_params"]:
        if key in node and isinstance(node[key], dict):
            p = extract_center_norm(node[key])
            if p is not None:
                return p

    return None


def extract_rotation_rad(node):
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

    for key in ["solver_params", "params", "final_params", "optimized_params"]:
        if key in node and isinstance(node[key], dict):
            val = extract_rotation_rad(node[key])
            if val is not None:
                return val

    return 0.0


def extract_length_norm(node):
    if "layout_prior" in node and isinstance(node["layout_prior"], dict):
        lp = node["layout_prior"]

        for k in ["length_norm", "scale_norm", "arc_length_norm", "chord_length_norm"]:
            if k in lp:
                v = safe_float(lp[k], 0.25)
                if abs(v) > 1.5:
                    v = v / CANVAS_SIZE
                return clamp(v, 0.0, 1.5)

    for k in [
        "length_norm",
        "scale_norm",
        "arc_length_norm",
        "chord_length_norm",
    ]:
        if k in node:
            v = safe_float(node[k], 0.25)
            if abs(v) > 1.5:
                v = v / CANVAS_SIZE
            return clamp(v, 0.0, 1.5)

    for k in [
        "actual_length_px",
        "target_length_px",
        "scale_px",
        "length_px",
        "scale",
    ]:
        if k in node:
            v = safe_float(node[k], 100.0)

            if abs(v) > 1.5:
                v = v / CANVAS_SIZE

            return clamp(v, 0.0, 1.5)

    for key in ["solver_priors", "solver_params", "params", "final_params", "optimized_params"]:
        if key in node and isinstance(node[key], dict):
            v = extract_length_norm(node[key])
            if v is not None:
                return v

    return 0.25


def node_id_of(node, fallback):
    return safe_int(node.get("node_id", node.get("source_node_id", fallback)), fallback)


def normalize_node_for_gnn(solved_node, original_node=None, fallback_idx=0):
    out = {}

    if original_node is not None:
        out.update(copy.deepcopy(original_node))

    out.update(copy.deepcopy(solved_node))

    node_id = node_id_of(out, fallback_idx)
    out["node_id"] = int(node_id)

    if "shape_code" not in out and original_node is not None:
        out["shape_code"] = original_node.get("shape_code", original_node.get("shape_token", 0))

    if "shape_token" not in out and "shape_code" in out:
        out["shape_token"] = out["shape_code"]

    if "width_token" not in out and original_node is not None:
        out["width_token"] = original_node.get("width_token", 0)

    center_norm = extract_center_norm(out)
    rotation_rad = extract_rotation_rad(out)
    length_norm = extract_length_norm(out)

    if center_norm is None:
        center_norm = np.array([0.5, 0.5], dtype=np.float32)

    center_norm = np.clip(center_norm, 0.0, 1.0)

    lp = out.setdefault("layout_prior", {})

    lp["center_norm"] = center_norm.astype(float).tolist()
    lp["rotation_rad"] = float(rotation_rad)
    lp["rotation_deg"] = float(rotation_rad * 180.0 / math.pi)
    lp["length_norm"] = float(length_norm)
    lp["scale_norm"] = float(length_norm)

    out["center_norm"] = lp["center_norm"]
    out["rotation_rad"] = lp["rotation_rad"]
    out["length_norm"] = lp["length_norm"]
    out["scale_norm"] = lp["scale_norm"]

    return out


def normalize_edges_for_gnn(edges):
    clean = []

    for i, e in enumerate(edges):
        if not isinstance(e, dict):
            continue

        ee = copy.deepcopy(e)

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


def build_score_object_from_candidate(candidate, idx):
    cand_id = get_candidate_id(candidate, idx)

    solved_nodes = choose_solved_nodes(candidate)
    original_nodes = choose_original_nodes(candidate)

    orig_by_id = {}

    for i, n in enumerate(original_nodes):
        oid = node_id_of(n, i)
        orig_by_id[oid] = n

    norm_nodes = []

    for i, sn in enumerate(solved_nodes):
        nid = node_id_of(sn, i)
        on = orig_by_id.get(nid, None)

        norm_nodes.append(
            normalize_node_for_gnn(
                solved_node=sn,
                original_node=on,
                fallback_idx=i,
            )
        )

    edges = normalize_edges_for_gnn(get_candidate_edges(candidate))

    obj = {
        "generated_glyph_id": cand_id,
        "glyph_uid": candidate.get("glyph_uid", cand_id),
        "source_file": candidate.get("source_file", ""),
        "nodes": norm_nodes,
        "topology": {
            "positive_edges_undirected": edges,
        },
    }

    return obj


# =========================================================
# 🧠 加载模型
# =========================================================
def load_model(device):
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"找不到模型文件: {MODEL_FILE}")

    checkpoint = torch.load(MODEL_FILE, map_location="cpu")

    node_dim = checkpoint["node_dim"]
    edge_dim = checkpoint["edge_dim"]
    global_dim = checkpoint["global_dim"]

    hidden_dim = checkpoint.get("hidden_dim", 128)
    num_layers = checkpoint.get("num_gnn_layers", 4)
    dropout = checkpoint.get("dropout", 0.15)

    model = gld.DenseGNNDiscriminator(
        node_dim=node_dim,
        edge_dim=edge_dim,
        global_dim=global_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


# =========================================================
# 📊 打分
# =========================================================
def score_graphs(model, graphs, device):
    scores = []

    all_indices = list(range(len(graphs)))

    with torch.no_grad():
        for start in range(0, len(all_indices), BATCH_SIZE):
            batch_idx = all_indices[start:start + BATCH_SIZE]

            batch = gld.batch_graphs(graphs, batch_idx, device)

            logits = model(
                batch["node_feats"],
                batch["edge_feats"],
                batch["adj"],
                batch["mask"],
                batch["global_feats"],
            )

            logits_np = logits.detach().cpu().numpy()
            probs = stable_sigmoid_np(logits_np)

            for local_i, gi in enumerate(batch_idx):
                scores.append(float(probs[local_i]))

    return scores


def get_quality_report(candidate):
    q = candidate.get("quality_report", {})

    if not isinstance(q, dict):
        q = {}

    return q


def get_quality_status(candidate):
    q = get_quality_report(candidate)

    status = q.get("quality_status", q.get("QualityStatus", candidate.get("quality_status", "unknown")))

    return str(status)


def get_quality_score(candidate):
    q = get_quality_report(candidate)

    for k in ["quality_score", "score"]:
        if k in q:
            return safe_float(q[k], 999999.0)

    return safe_float(candidate.get("quality_score", 999999.0), 999999.0)


def compute_solver_feasibility_score(max_junction_px, max_angle_diff_deg):
    return float(
        math.exp(
            -COMBINED_JUNCTION_WEIGHT * max(0.0, max_junction_px)
            -COMBINED_ANGLE_WEIGHT * max(0.0, max_angle_diff_deg)
        )
    )


def is_solver_allowed(item):
    status = item["quality_status"]

    if status in GOOD_STATUS_SET:
        return True

    if status in USABLE_STATUS_SET:
        if (
            item["max_junction_px"] <= USABLE_MAX_JUNCTION_PX
            and item["max_angle_diff_deg"] <= USABLE_MAX_ANGLE_DEG
        ):
            return True

    return False


def summarize_scored_items(items):
    by_quality = defaultdict(list)
    by_node_count = defaultdict(list)
    by_edge_count = defaultdict(list)
    by_allowed = defaultdict(list)

    for it in items:
        by_quality[it["quality_status"]].append(it["gnn_layout_realness_score"])
        by_node_count[str(it["num_nodes"])].append(it["gnn_layout_realness_score"])
        by_edge_count[str(it["num_edges"])].append(it["gnn_layout_realness_score"])
        by_allowed[str(it["solver_allowed"])].append(it["gnn_layout_realness_score"])

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

    all_scores = [it["gnn_layout_realness_score"] for it in items]
    all_combined = [it["combined_score"] for it in items]

    return {
        "count": len(items),
        "overall_gnn": stats(all_scores),
        "overall_combined": stats(all_combined),
        "by_quality_status": {
            k: stats(v)
            for k, v in by_quality.items()
        },
        "by_solver_allowed": {
            k: stats(v)
            for k, v in by_allowed.items()
        },
        "by_node_count": {
            k: stats(v)
            for k, v in by_node_count.items()
        },
        "by_edge_count": {
            k: stats(v)
            for k, v in by_edge_count.items()
        },
    }


# =========================================================
# 🎨 polyline / width 渲染
# =========================================================
def extract_polyline_px(node):
    keys = [
        "solved_polyline_px",
        "polyline_px",
        "world_polyline_px",
        "transformed_polyline_px",
        "render_polyline_px",
        "points_px",
        "polyline",
        "points",
    ]

    for k in keys:
        if k in node:
            arr = np.asarray(node[k], dtype=np.float32)

            if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) >= 2:
                arr = arr[:, :2]

                if np.max(np.abs(arr)) <= 1.5:
                    arr = arr * CANVAS_SIZE

                return arr

    return None


def line_from_layout_prior_px(node):
    lp = node.get("layout_prior", {})

    center = None

    if "center_norm" in lp:
        center = to_px_xy(lp["center_norm"])
    else:
        center = extract_center_norm(node)
        if center is not None:
            center = center * CANVAS_SIZE

    if center is None:
        return None

    theta = extract_rotation_rad(node)
    length_norm = extract_length_norm(node)

    length_px = length_norm * CANVAS_SIZE

    dx = math.cos(theta) * length_px * 0.5
    dy = math.sin(theta) * length_px * 0.5

    p0 = center - np.array([dx, dy], dtype=np.float32)
    p1 = center + np.array([dx, dy], dtype=np.float32)

    return np.stack([p0, p1], axis=0)


def extract_render_width_px(node):
    """
    优先读取真实宽度字段；否则用 width_token 兜底。
    """
    for key in [
        "width_px",
        "stroke_width_px",
        "render_width_px",
        "width_mean_px",
        "actual_width_px",
        "target_width_px",
    ]:
        if key in node:
            w = safe_float(node[key], DEFAULT_RENDER_WIDTH_PX)
            return clamp(w, MIN_RENDER_WIDTH_PX, MAX_RENDER_WIDTH_PX)

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
                    w = safe_float(node[parent][key], DEFAULT_RENDER_WIDTH_PX)
                    return clamp(w, MIN_RENDER_WIDTH_PX, MAX_RENDER_WIDTH_PX)

    width_token = safe_int(node.get("width_token", 0), 0)
    w = WIDTH_TOKEN_TO_PX.get(width_token, DEFAULT_RENDER_WIDTH_PX)

    return clamp(w, MIN_RENDER_WIDTH_PX, MAX_RENDER_WIDTH_PX)


def draw_debug_view(ax, candidate, score_item):
    score_obj = build_score_object_from_candidate(candidate, score_item["candidate_index"])
    nodes = score_obj["nodes"]
    edges = score_obj["topology"]["positive_edges_undirected"]

    colors = [
        "#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA",
        "#00897B", "#6D4C41", "#3949AB", "#D81B60", "#7CB342",
        "#546E7A", "#F4511E",
    ]

    node_center_px = {}

    for i, node in enumerate(nodes):
        c = colors[i % len(colors)]

        poly = extract_polyline_px(node)

        if poly is None:
            poly = line_from_layout_prior_px(node)

        if poly is not None:
            ax.plot(poly[:, 0], poly[:, 1], color=c, lw=2.6, alpha=0.95)
            ax.scatter([poly[0, 0]], [poly[0, 1]], color="green", s=25, zorder=6)
            ax.scatter([poly[-1, 0]], [poly[-1, 1]], color="red", s=25, zorder=6)

        center_norm = extract_center_norm(node)

        if center_norm is not None:
            center_px = center_norm * CANVAS_SIZE
            node_center_px[node["node_id"]] = center_px

            ax.scatter([center_px[0]], [center_px[1]], color="black", s=14, zorder=8)
            ax.text(
                center_px[0] + 4,
                center_px[1] - 4,
                f"N{node['node_id']}",
                fontsize=8,
                color="black",
            )

    for e in edges:
        u = safe_int(e.get("u", -1), -1)
        v = safe_int(e.get("v", -1), -1)

        if u not in node_center_px or v not in node_center_px:
            continue

        pu = node_center_px[u]
        pv = node_center_px[v]
        pm = 0.5 * (pu + pv)

        jt = e.get("j_type", "?")

        if jt == "T":
            color = "#D81B60"
            ls = "--"
        elif jt == "X":
            color = "#1E88E5"
            ls = ":"
        else:
            color = "#555555"
            ls = "--"

        ax.plot(
            [pu[0], pv[0]],
            [pu[1], pv[1]],
            color=color,
            lw=0.9,
            ls=ls,
            alpha=0.55,
        )

        ax.text(
            pm[0],
            pm[1],
            jt,
            fontsize=7,
            color=color,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )

    ax.set_title("debug: colored strokes + topology", fontsize=9)


def draw_black_width_view(ax, candidate, score_item):
    """
    纯黑带宽度渲染。
    """
    score_obj = build_score_object_from_candidate(candidate, score_item["candidate_index"])
    nodes = score_obj["nodes"]

    for node in nodes:
        poly = extract_polyline_px(node)

        if poly is None:
            poly = line_from_layout_prior_px(node)

        if poly is None:
            continue

        w = extract_render_width_px(node)

        ax.plot(
            poly[:, 0],
            poly[:, 1],
            color="black",
            lw=w,
            alpha=1.0,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=5,
        )

    ax.set_title("black width render", fontsize=9)


def render_candidate_preview(candidate, score_item, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))

    ax_debug, ax_black = axes

    draw_debug_view(ax_debug, candidate, score_item)
    draw_black_width_view(ax_black, candidate, score_item)

    title = (
        f"{score_item['candidate_id']} | "
        f"GNN={score_item['gnn_layout_realness_score']:.4f} | "
        f"solver={score_item['quality_status']} | "
        f"feas={score_item['solver_feasibility_score']:.4f} | "
        f"combined={score_item['combined_score']:.4f}\n"
        f"Q={score_item['solver_quality_score']:.3f} | "
        f"maxJ={score_item['max_junction_px']:.2f}px | "
        f"maxA={score_item['max_angle_diff_deg']:.1f}°"
    )

    fig.suptitle(title, fontsize=10)

    for ax in axes:
        ax.set_xlim(0, CANVAS_SIZE)
        ax.set_ylim(CANVAS_SIZE, 0)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.20)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_previews(candidates, scored_items):
    ensure_dir(PREVIEW_DIR)
    ensure_dir(TOP_GNN_DIR)
    ensure_dir(BOTTOM_GNN_DIR)
    ensure_dir(MIDDLE_GNN_DIR)
    ensure_dir(TOP_GNN_GOOD_ONLY_DIR)
    ensure_dir(TOP_COMBINED_DIR)
    ensure_dir(BAD_BUT_GNN_HIGH_DIR)

    sorted_by_gnn = sorted(
        scored_items,
        key=lambda x: x["gnn_layout_realness_score"],
        reverse=True,
    )

    sorted_by_combined = sorted(
        scored_items,
        key=lambda x: x["combined_score"],
        reverse=True,
    )

    good_only = [
        x for x in sorted_by_gnn
        if x["solver_allowed"]
    ]

    bad_but_high = [
        x for x in sorted_by_gnn
        if not x["solver_allowed"]
    ]

    top_gnn_items = sorted_by_gnn[:TOP_K_PREVIEW]
    bottom_items = sorted_by_gnn[-BOTTOM_K_PREVIEW:]

    mid_start = max(0, len(sorted_by_gnn) // 2 - MIDDLE_K_PREVIEW // 2)
    middle_items = sorted_by_gnn[mid_start:mid_start + MIDDLE_K_PREVIEW]

    good_only_items = good_only[:GOOD_ONLY_PREVIEW]
    combined_items = sorted_by_combined[:COMBINED_PREVIEW]
    bad_high_items = bad_but_high[:BAD_BUT_HIGH_PREVIEW]

    print("\n" + "=" * 80)
    print("🖼️ Rendering GNN Scored Previews")
    print("=" * 80)

    def render_group(items, out_dir, prefix):
        for rank, item in enumerate(items):
            idx = item["candidate_index"]
            cand = candidates[idx]

            out_path = os.path.join(
                out_dir,
                (
                    f"{prefix}_{rank:03d}_"
                    f"{item['candidate_id']}_"
                    f"gnn_{item['gnn_layout_realness_score']:.4f}_"
                    f"comb_{item['combined_score']:.4f}.png"
                ),
            )

            render_candidate_preview(cand, item, out_path)
            print(f"  saved: {out_path}")

    render_group(top_gnn_items, TOP_GNN_DIR, "top_gnn")
    render_group(bottom_items, BOTTOM_GNN_DIR, "bottom_gnn")
    render_group(middle_items, MIDDLE_GNN_DIR, "middle_gnn")
    render_group(good_only_items, TOP_GNN_GOOD_ONLY_DIR, "good_only")
    render_group(combined_items, TOP_COMBINED_DIR, "combined")
    render_group(bad_high_items, BAD_BUT_GNN_HIGH_DIR, "bad_high")

    print("=" * 80 + "\n")


# =========================================================
# 🚀 主流程
# =========================================================
def main():
    print("\n" + "=" * 80)
    print("🚀 Score Solved Glyphs with GNN Layout Critic V2")
    print("=" * 80)
    print(f"  model_file:  {MODEL_FILE}")
    print(f"  solved_file: {SOLVED_FILE}")
    print(f"  output_file: {OUTPUT_FILE}")
    print(f"  device_mode: {DEVICE_MODE}")
    print("=" * 80)

    device = choose_device()
    print(f"\n[Device]")
    print(f"  selected device: {device}")

    model, checkpoint = load_model(device)

    print("\n[Model]")
    print(f"  best_state: {checkpoint.get('best_state', {})}")
    print(f"  node_dim:   {checkpoint.get('node_dim')}")
    print(f"  edge_dim:   {checkpoint.get('edge_dim')}")
    print(f"  global_dim: {checkpoint.get('global_dim')}")

    solved_data = load_json(SOLVED_FILE)
    candidates = get_candidate_list(solved_data)

    print("\n[Input Candidates]")
    print(f"  candidate_count: {len(candidates)}")

    graphs = []
    graph_to_candidate_idx = []
    skipped = Counter()

    for idx, cand in enumerate(candidates):
        cand_id = get_candidate_id(cand, idx)

        score_obj = build_score_object_from_candidate(cand, idx)

        g = gld.graph_from_object(
            obj=score_obj,
            label=0.0,
            kind="solved_generated_candidate",
            graph_id=cand_id,
            group_id=cand_id,
        )

        if g is None:
            skipped["bad_graph"] += 1
            continue

        graphs.append(g)
        graph_to_candidate_idx.append(idx)

    print("\n[Graph Build]")
    print(f"  scoreable_graphs: {len(graphs)}")
    print(f"  skipped: {dict(skipped)}")

    if len(graphs) == 0:
        raise RuntimeError("没有可以打分的 solved graph。请检查 solved_nodes / topology edges 是否存在。")

    scores = score_graphs(model, graphs, device)

    scored_items = []

    for gi, score in enumerate(scores):
        cand_idx = graph_to_candidate_idx[gi]
        cand = candidates[cand_idx]
        cand_id = get_candidate_id(cand, cand_idx)

        edges = get_candidate_edges(cand)
        nodes = choose_solved_nodes(cand)

        q = get_quality_report(cand)

        mean_junction_px = read_metric_from_quality(
            q,
            ["mean_junction_px", "mean_j_px", "mean_junction", "mean_j"],
            0.0,
        )

        max_junction_px = read_metric_from_quality(
            q,
            ["max_junction_px", "max_j_px", "max_junction", "max_j"],
            0.0,
        )

        mean_angle_diff_deg = read_metric_from_quality(
            q,
            ["mean_angle_diff_deg", "mean_ang_deg", "mean_angle_deg", "mean_angle"],
            0.0,
        )

        max_angle_diff_deg = read_metric_from_quality(
            q,
            ["max_angle_diff_deg", "max_ang_deg", "max_angle_deg", "max_angle"],
            0.0,
        )

        solver_feasibility_score = compute_solver_feasibility_score(
            max_junction_px=max_junction_px,
            max_angle_diff_deg=max_angle_diff_deg,
        )

        combined_score = float(score * solver_feasibility_score)

        item = {
            "rank_by_gnn": None,
            "rank_by_combined": None,

            "candidate_index": int(cand_idx),
            "candidate_id": cand_id,

            "gnn_layout_realness_score": float(score),
            "solver_feasibility_score": float(solver_feasibility_score),
            "combined_score": float(combined_score),

            "quality_status": get_quality_status(cand),
            "solver_quality_score": get_quality_score(cand),

            "solver_allowed": None,

            "num_nodes": int(len(nodes)),
            "num_edges": int(len(edges)),

            "mean_junction_px": float(mean_junction_px),
            "max_junction_px": float(max_junction_px),
            "mean_angle_diff_deg": float(mean_angle_diff_deg),
            "max_angle_diff_deg": float(max_angle_diff_deg),
        }

        item["solver_allowed"] = bool(is_solver_allowed(item))

        scored_items.append(item)

    scored_by_gnn = sorted(
        scored_items,
        key=lambda x: x["gnn_layout_realness_score"],
        reverse=True,
    )

    scored_by_combined = sorted(
        scored_items,
        key=lambda x: x["combined_score"],
        reverse=True,
    )

    for rank, item in enumerate(scored_by_gnn):
        item["rank_by_gnn"] = int(rank)

    for rank, item in enumerate(scored_by_combined):
        item["rank_by_combined"] = int(rank)

    summary = summarize_scored_items(scored_items)

    allowed_items = [x for x in scored_by_gnn if x["solver_allowed"]]
    bad_but_high_items = [x for x in scored_by_gnn if not x["solver_allowed"]]

    print("\n" + "=" * 80)
    print("📊 GNN + Solver Score Summary")
    print("=" * 80)

    print(f"  scored_count: {summary['count']}")
    print(f"  overall_gnn:      {summary['overall_gnn']}")
    print(f"  overall_combined: {summary['overall_combined']}")
    print(f"  solver_allowed_count: {len(allowed_items)}")
    print(f"  solver_rejected_count: {len(bad_but_high_items)}")

    print("\n[By Solver Quality Status]")
    for k, v in summary["by_quality_status"].items():
        print(f"  {k}: {v}")

    print("\n[By Solver Allowed]")
    for k, v in summary["by_solver_allowed"].items():
        print(f"  {k}: {v}")

    print("\n[Top-20 by GNN Realness]")
    for item in scored_by_gnn[:20]:
        print(
            f"  GNN#{item['rank_by_gnn']:03d} | "
            f"{item['candidate_id']} | "
            f"gnn={item['gnn_layout_realness_score']:.4f} | "
            f"feas={item['solver_feasibility_score']:.4f} | "
            f"comb={item['combined_score']:.4f} | "
            f"allowed={item['solver_allowed']} | "
            f"solver={item['quality_status']} | "
            f"Q={item['solver_quality_score']:.3f} | "
            f"maxJ={item['max_junction_px']:.2f}px | "
            f"maxA={item['max_angle_diff_deg']:.1f}°"
        )

    print("\n[Top-20 by Combined Score]")
    for item in scored_by_combined[:20]:
        print(
            f"  COMB#{item['rank_by_combined']:03d} | "
            f"{item['candidate_id']} | "
            f"comb={item['combined_score']:.4f} | "
            f"gnn={item['gnn_layout_realness_score']:.4f} | "
            f"feas={item['solver_feasibility_score']:.4f} | "
            f"allowed={item['solver_allowed']} | "
            f"solver={item['quality_status']} | "
            f"Q={item['solver_quality_score']:.3f} | "
            f"maxJ={item['max_junction_px']:.2f}px | "
            f"maxA={item['max_angle_diff_deg']:.1f}°"
        )

    print("\n[Top-20 by GNN among Solver-Allowed]")
    for rank, item in enumerate(allowed_items[:20]):
        print(
            f"  GOOD#{rank:03d} | "
            f"{item['candidate_id']} | "
            f"gnn={item['gnn_layout_realness_score']:.4f} | "
            f"comb={item['combined_score']:.4f} | "
            f"solver={item['quality_status']} | "
            f"Q={item['solver_quality_score']:.3f} | "
            f"maxJ={item['max_junction_px']:.2f}px | "
            f"maxA={item['max_angle_diff_deg']:.1f}°"
        )

    print("\n[Top Bad-but-GNN-High]")
    for rank, item in enumerate(bad_but_high_items[:16]):
        print(
            f"  BADHIGH#{rank:03d} | "
            f"{item['candidate_id']} | "
            f"gnn={item['gnn_layout_realness_score']:.4f} | "
            f"comb={item['combined_score']:.4f} | "
            f"solver={item['quality_status']} | "
            f"Q={item['solver_quality_score']:.3f} | "
            f"maxJ={item['max_junction_px']:.2f}px | "
            f"maxA={item['max_angle_diff_deg']:.1f}°"
        )

    print("\n[Bottom-10 by GNN Realness]")
    for item in scored_by_gnn[-10:]:
        print(
            f"  GNN#{item['rank_by_gnn']:03d} | "
            f"{item['candidate_id']} | "
            f"gnn={item['gnn_layout_realness_score']:.4f} | "
            f"comb={item['combined_score']:.4f} | "
            f"solver={item['quality_status']} | "
            f"Q={item['solver_quality_score']:.3f}"
        )

    if SAVE_FULL_CANDIDATES:
        ranked_candidates_by_combined = []

        for item in scored_by_combined:
            cand = copy.deepcopy(candidates[item["candidate_index"]])
            cand["gnn_layout_realness_score"] = item["gnn_layout_realness_score"]
            cand["solver_feasibility_score"] = item["solver_feasibility_score"]
            cand["combined_score"] = item["combined_score"]
            cand["rank_by_gnn"] = item["rank_by_gnn"]
            cand["rank_by_combined"] = item["rank_by_combined"]
            cand["solver_allowed"] = item["solver_allowed"]
            cand["gnn_score_item"] = item
            ranked_candidates_by_combined.append(cand)
    else:
        ranked_candidates_by_combined = []

    output = {
        "schema_version": "gnn_scored_solved_glyphs_v2_combined",
        "model_file": MODEL_FILE,
        "solved_file": SOLVED_FILE,
        "output_file": OUTPUT_FILE,

        "model_best_state": checkpoint.get("best_state", {}),
        "config": {
            "combined_junction_weight": COMBINED_JUNCTION_WEIGHT,
            "combined_angle_weight": COMBINED_ANGLE_WEIGHT,
            "usable_max_junction_px": USABLE_MAX_JUNCTION_PX,
            "usable_max_angle_deg": USABLE_MAX_ANGLE_DEG,
            "width_token_to_px": WIDTH_TOKEN_TO_PX,
            "default_render_width_px": DEFAULT_RENDER_WIDTH_PX,
        },

        "summary": summary,

        "scored_items_by_gnn": scored_by_gnn,
        "scored_items_by_combined": scored_by_combined,
        "solver_allowed_items_by_gnn": allowed_items,
        "bad_but_gnn_high_items": bad_but_high_items,

        "top_gnn_candidate_ids": [
            item["candidate_id"]
            for item in scored_by_gnn[:TOP_K_PREVIEW]
        ],

        "top_combined_candidate_ids": [
            item["candidate_id"]
            for item in scored_by_combined[:COMBINED_PREVIEW]
        ],

        "top_gnn_good_only_candidate_ids": [
            item["candidate_id"]
            for item in allowed_items[:GOOD_ONLY_PREVIEW]
        ],

        "bad_but_gnn_high_candidate_ids": [
            item["candidate_id"]
            for item in bad_but_high_items[:BAD_BUT_HIGH_PREVIEW]
        ],

        "ranked_solved_glyph_candidates_by_combined": ranked_candidates_by_combined,
    }

    save_json(output, OUTPUT_FILE)

    print("\n" + "=" * 80)
    print("💾 Saved")
    print("=" * 80)
    print(f"  scored json: {OUTPUT_FILE}")

    if SAVE_PREVIEWS:
        render_previews(candidates, scored_items)
        print(f"  previews:    {PREVIEW_DIR}")

    print("\n📌 下一步看法：")
    print("  1. 优先打开 gnn_scored_solved_previews/top_gnn_good_only")
    print("  2. 再打开 gnn_scored_solved_previews/top_combined")
    print("  3. 每张图左侧是彩色 debug/topology，右侧是纯黑带宽度渲染")
    print("  4. 如果右侧黑色宽度图更像字符，说明宽度渲染方向有价值")
    print("  5. 如果 top_bad_but_gnn_high 看起来像字符但连接差，说明 solver 还需要二次修复")
    print("  6. 如果 good_only 仍然不像字符，就需要加入人工 hard negative 继续训练 GNN critic")


if __name__ == "__main__":
    main()