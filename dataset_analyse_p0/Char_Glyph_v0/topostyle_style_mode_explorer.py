# -*- coding: utf-8 -*-
"""
TopoStyle Style Mode Explorer
--------------------------------
作用：
1. 读取 topostyle retrieval / solved candidates
2. 对每个 candidate 显式生成多种 style mode
3. 保持 topology endpoints 不变，只修改 cubic Bézier 控制点
4. 输出多样化候选 JSON + preview 图

适用场景：
- 你已经发现 retrieval 的 token 虽然不同，但肉眼看起来差异很小
- 你想强制生成 “直线 / 左弯 / 右弯 / S弯 / hook” 等明显不同的曲线族

输出：
- topostyle_style_mode_explorer_outputs.json
- topostyle_style_mode_explorer_report.json
- topostyle_style_mode_previews/

作者注释：
- 这个脚本不训练模型
- 它是一个“显式 style mode 扩展器”
- 非常适合拿来验证：你的 topology-first 方案，在固定拓扑下，是否能撑起多种美观曲线风格
"""
import re
import os
import json
import math
import copy
import random
from collections import Counter, defaultdict

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except Exception:
    PIL_OK = False


# =============================================================================
# 配置
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "topostyle_retrieval_solved_candidates.json"),
    os.path.join(SCRIPT_DIR, "solved_glyph_candidates.json"),
]

OUTPUT_FILE = os.path.join(SCRIPT_DIR, "topostyle_style_mode_explorer_outputs.json")
REPORT_FILE = os.path.join(SCRIPT_DIR, "topostyle_style_mode_explorer_report.json")
PREVIEW_DIR = os.path.join(SCRIPT_DIR, "topostyle_style_mode_previews")

# 是否限制输入候选数（None 表示全量）
MAX_INPUT_CANDIDATES = None  # 例如可改成 24 先小试

# 每个 source candidate 生成哪些 style modes
STYLE_MODES = [
    "straight",
    "mild_left",
    "mild_right",
    "strong_left",
    "strong_right",
    "s_curve_left",
    "s_curve_right",
    "hook_left",
    "hook_right",
]

# preview 参数
PREVIEW_SOURCE_LIMIT = 50          # 最多展示多少个 source candidate
PREVIEW_TILE_W = 220
PREVIEW_TILE_H = 220
PREVIEW_MARGIN = 18
PREVIEW_STROKE_BASE = 6
PREVIEW_SAMPLES_PER_CURVE = 60

# 如果原曲线过短，避免极端扭曲
MIN_CURVE_LENGTH = 8.0

# 是否混入一点原始 style，避免所有 mode 太模板化
BLEND_WITH_ORIGINAL = 0.15

# 结构性 shape 可以少动一点（例如直线 shape=20）
STRUCTURAL_SHAPES = {20}

# 对结构性 shape 的 style 强度衰减
STRUCTURAL_SHAPE_SCALE = 0.65

# 可视化时的宽度默认值
DEFAULT_WIDTH = 10.0


# =============================================================================
# 工具函数
# =============================================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def choose_input_file():
    for fp in INPUT_FILE_CANDIDATES:
        if os.path.exists(fp):
            return fp
    raise FileNotFoundError(
        "找不到输入文件。请确保存在以下任一文件：\n" +
        "\n".join(INPUT_FILE_CANDIDATES)
    )


def stats_dict(vals):
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=np.float32)
    return {
        "mean": round(float(np.mean(arr)), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
        "max": round(float(np.max(arr)), 6),
    }


def cubic_bezier_point(pts, t):
    pts = np.asarray(pts, dtype=np.float32)
    mt = 1.0 - t
    return (
        (mt ** 3) * pts[0]
        + 3 * (mt ** 2) * t * pts[1]
        + 3 * mt * (t ** 2) * pts[2]
        + (t ** 3) * pts[3]
    )


def sample_bezier(pts, n=60):
    ts = np.linspace(0.0, 1.0, n)
    return np.stack([cubic_bezier_point(pts, t) for t in ts], axis=0)


def curve_length(pts, n=60):
    s = sample_bezier(pts, n=n)
    d = np.linalg.norm(np.diff(s, axis=0), axis=1)
    return float(np.sum(d))


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def get_candidate_list(data):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for k in [
            "topostyle_retrieval_solved_candidates",
            "solved_glyph_candidates",
            "glyph_candidates_with_primitives",
            "candidates",
            "items",
            "data",
        ]:
            if k in data and isinstance(data[k], list):
                return data[k]

    raise RuntimeError("无法从 JSON 中识别 candidate list 结构。")

def get_candidate_id(cand, idx):
    for k in [
        "generated_glyph_id",
        "source_candidate_id",
        "candidate_id",
        "glyph_id",
        "glyph_candidate_id",
        "id",
        "name",
    ]:
        if k in cand:
            return str(cand[k])
    return f"candidate_{idx:05d}"


def get_base_candidate_id(cand, idx):
    """
    把 glyph_candidate_00005_topostyle_beam_03
    归并成 glyph_candidate_00005
    """
    cid = get_candidate_id(cand, idx)
    cid = re.sub(r"_topostyle_beam_\d+$", "", cid)
    return cid


def get_nodes(cand):
    for k in ["solved_nodes", "nodes", "strokes"]:
        if k in cand and isinstance(cand[k], list):
            return cand[k]
    return []


def set_nodes(cand, nodes):
    out = copy.deepcopy(cand)
    if "solved_nodes" in out:
        out["solved_nodes"] = nodes
    elif "nodes" in out:
        out["nodes"] = nodes
    elif "strokes" in out:
        out["strokes"] = nodes
    else:
        out["solved_nodes"] = nodes
    return out


def get_bezier_from_node(node):
    """
    尽量鲁棒地取 cubic bezier。
    """
    for k in ["mother_bezier", "bezier", "curve", "solved_bezier"]:
        if k in node and isinstance(node[k], list) and len(node[k]) == 4:
            return np.asarray(node[k], dtype=np.float32)

    # 有些结构可能在 primitive_ref 里
    pref = node.get("primitive_ref", {})
    for k in ["mother_bezier", "bezier", "curve"]:
        if k in pref and isinstance(pref[k], list) and len(pref[k]) == 4:
            return np.asarray(pref[k], dtype=np.float32)

    return None


def set_bezier_to_node(node, pts):
    pts_list = np.asarray(pts, dtype=np.float32).tolist()
    out = copy.deepcopy(node)
    if "mother_bezier" in out:
        out["mother_bezier"] = pts_list
    elif "bezier" in out:
        out["bezier"] = pts_list
    elif "curve" in out:
        out["curve"] = pts_list
    elif "solved_bezier" in out:
        out["solved_bezier"] = pts_list
    else:
        out["mother_bezier"] = pts_list
    return out


def get_width_from_node(node):
    for k in ["width_mean", "width", "stroke_width"]:
        if k in node:
            return safe_float(node[k], DEFAULT_WIDTH)
    if "primitive_ref" in node and isinstance(node["primitive_ref"], dict):
        pref = node["primitive_ref"]
        for k in ["width_mean", "width", "stroke_width"]:
            if k in pref:
                return safe_float(pref[k], DEFAULT_WIDTH)
    return DEFAULT_WIDTH


def get_shape_code_from_node(node):
    if "shape_code" in node:
        return node["shape_code"]
    if "primitive_ref" in node and isinstance(node["primitive_ref"], dict):
        return node["primitive_ref"].get("shape_code", None)
    return None


# =============================================================================
# style mode 核心：在局部坐标系中改控制点
# =============================================================================
def build_local_frame(p0, p3):
    v = p3 - p0
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        return None, None, 0.0
    ex = v / L
    ey = np.array([-ex[1], ex[0]], dtype=np.float32)
    return ex, ey, L


def to_local(pts):
    """
    把 cubic Bézier 变到以 p0->p3 为 x 轴的局部坐标系。
    返回 normalized local points (按 chord 长度归一化)。
    """
    pts = np.asarray(pts, dtype=np.float32)
    p0, p1, p2, p3 = pts
    ex, ey, L = build_local_frame(p0, p3)
    if ex is None:
        return None

    def proj(p):
        d = p - p0
        x = np.dot(d, ex) / L
        y = np.dot(d, ey) / L
        return np.array([x, y], dtype=np.float32)

    return np.stack([proj(p) for p in pts], axis=0)


def from_local(local_pts, p0, p3):
    """
    从 local normalized 坐标还原回全局。
    """
    local_pts = np.asarray(local_pts, dtype=np.float32)
    ex, ey, L = build_local_frame(np.asarray(p0), np.asarray(p3))
    if ex is None:
        return None

    out = []
    for x, y in local_pts:
        g = p0 + (x * L) * ex + (y * L) * ey
        out.append(g)
    return np.stack(out, axis=0)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def mode_target_local(mode_name, base_local, shape_code=None):
    """
    输入原始 local cubic，输出目标 local cubic（端点固定）。
    控制点采用显式 style 模式。
    """
    pts = np.asarray(base_local, dtype=np.float32)

    # 原始控制点 x 可以保留一部分
    x1 = clamp(float(pts[1, 0]), 0.12, 0.45)
    x2 = clamp(float(pts[2, 0]), 0.55, 0.88)

    # 如果原始很怪，也给默认
    if x1 >= x2:
        x1, x2 = 0.33, 0.67

    # style 强度
    structural_scale = STRUCTURAL_SHAPE_SCALE if shape_code in STRUCTURAL_SHAPES else 1.0

    # base target
    p0 = np.array([0.0, 0.0], dtype=np.float32)
    p3 = np.array([1.0, 0.0], dtype=np.float32)

    if mode_name == "straight":
        p1 = np.array([x1, 0.0], dtype=np.float32)
        p2 = np.array([x2, 0.0], dtype=np.float32)

    elif mode_name == "mild_left":
        a = 0.16 * structural_scale
        p1 = np.array([x1, +a], dtype=np.float32)
        p2 = np.array([x2, +a], dtype=np.float32)

    elif mode_name == "mild_right":
        a = 0.16 * structural_scale
        p1 = np.array([x1, -a], dtype=np.float32)
        p2 = np.array([x2, -a], dtype=np.float32)

    elif mode_name == "strong_left":
        a = 0.30 * structural_scale
        p1 = np.array([x1, +a], dtype=np.float32)
        p2 = np.array([x2, +a], dtype=np.float32)

    elif mode_name == "strong_right":
        a = 0.30 * structural_scale
        p1 = np.array([x1, -a], dtype=np.float32)
        p2 = np.array([x2, -a], dtype=np.float32)

    elif mode_name == "s_curve_left":
        a = 0.26 * structural_scale
        p1 = np.array([x1, +a], dtype=np.float32)
        p2 = np.array([x2, -a], dtype=np.float32)

    elif mode_name == "s_curve_right":
        a = 0.26 * structural_scale
        p1 = np.array([x1, -a], dtype=np.float32)
        p2 = np.array([x2, +a], dtype=np.float32)

    elif mode_name == "hook_left":
        a1 = 0.36 * structural_scale
        a2 = 0.06 * structural_scale
        p1 = np.array([0.20, +a1], dtype=np.float32)
        p2 = np.array([0.70, +a2], dtype=np.float32)

    elif mode_name == "hook_right":
        a1 = 0.36 * structural_scale
        a2 = 0.06 * structural_scale
        p1 = np.array([0.20, -a1], dtype=np.float32)
        p2 = np.array([0.70, -a2], dtype=np.float32)

    else:
        # fallback
        p1 = np.array([x1, 0.0], dtype=np.float32)
        p2 = np.array([x2, 0.0], dtype=np.float32)

    target = np.stack([p0, p1, p2, p3], axis=0)

    # 混一点原始形状，避免太模板化
    blended = (1.0 - BLEND_WITH_ORIGINAL) * target + BLEND_WITH_ORIGINAL * pts

    # 强行固定端点
    blended[0] = np.array([0.0, 0.0], dtype=np.float32)
    blended[3] = np.array([1.0, 0.0], dtype=np.float32)
    return blended


def apply_style_mode_to_bezier(global_pts, mode_name, shape_code=None):
    """
    保持 p0/p3 不变，只修改控制点。
    """
    global_pts = np.asarray(global_pts, dtype=np.float32)
    p0, _, _, p3 = global_pts

    # 过短曲线直接少动
    L = np.linalg.norm(p3 - p0)
    if L < MIN_CURVE_LENGTH:
        base_local = to_local(global_pts)
        if base_local is None:
            return global_pts.copy()
        tgt = mode_target_local("straight", base_local, shape_code=shape_code)
        out = from_local(tgt, p0, p3)
        return out if out is not None else global_pts.copy()

    base_local = to_local(global_pts)
    if base_local is None:
        return global_pts.copy()

    target_local = mode_target_local(mode_name, base_local, shape_code=shape_code)
    out = from_local(target_local, p0, p3)
    if out is None:
        return global_pts.copy()

    # 再次固定端点，防止数值漂移
    out[0] = p0
    out[3] = p3
    return out


# =============================================================================
# 渲染
# =============================================================================
def fit_points_to_canvas(all_points, canvas_w, canvas_h, margin=18):
    pts = np.asarray(all_points, dtype=np.float32)
    min_xy = np.min(pts, axis=0)
    max_xy = np.max(pts, axis=0)

    span = max_xy - min_xy
    sx = max(span[0], 1e-5)
    sy = max(span[1], 1e-5)

    scale = min(
        (canvas_w - 2 * margin) / sx,
        (canvas_h - 2 * margin) / sy
    )

    def map_fn(p):
        x = (p[0] - min_xy[0]) * scale + margin
        y = (p[1] - min_xy[1]) * scale + margin
        # 图像坐标 y 朝下
        y = canvas_h - y
        return np.array([x, y], dtype=np.float32)

    return map_fn


def render_candidate_black(candidate, canvas_w=220, canvas_h=220):
    if not PIL_OK:
        return None

    nodes = get_nodes(candidate)
    curves = []
    widths = []
    for nd in nodes:
        bez = get_bezier_from_node(nd)
        if bez is None:
            continue
        curves.append(sample_bezier(bez, n=PREVIEW_SAMPLES_PER_CURVE))
        widths.append(get_width_from_node(nd))

    if not curves:
        img = Image.new("RGB", (canvas_w, canvas_h), "white")
        return img

    all_points = np.concatenate(curves, axis=0)
    mapper = fit_points_to_canvas(all_points, canvas_w, canvas_h, margin=PREVIEW_MARGIN)

    img = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(img)

    for curve_pts, w in zip(curves, widths):
        mapped = [tuple(mapper(p)) for p in curve_pts]
        width_px = max(2, int(round(PREVIEW_STROKE_BASE * (w / max(DEFAULT_WIDTH, 1e-5)))))
        try:
            draw.line(mapped, fill="black", width=width_px, joint="curve")
        except Exception:
            draw.line(mapped, fill="black", width=width_px)

    return img


def draw_text(img, text, xy=(8, 8)):
    if not PIL_OK:
        return img
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.size[0], 24], fill=(255, 255, 255))
    draw.text(xy, text, fill=(30, 30, 30))
    return img


def stack_tiles_h(tiles, bg="white", gap=8):
    if not PIL_OK or not tiles:
        return None
    w = sum(t.size[0] for t in tiles) + gap * (len(tiles) - 1)
    h = max(t.size[1] for t in tiles)
    out = Image.new("RGB", (w, h), bg)
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.size[0] + gap
    return out


def stack_tiles_v(tiles, bg="white", gap=8):
    if not PIL_OK or not tiles:
        return None
    w = max(t.size[0] for t in tiles)
    h = sum(t.size[1] for t in tiles) + gap * (len(tiles) - 1)
    out = Image.new("RGB", (w, h), bg)
    y = 0
    for t in tiles:
        out.paste(t, (0, y))
        y += t.size[1] + gap
    return out


def make_contact_sheet(group_images, cols=3, bg="white", gap=10):
    if not PIL_OK or not group_images:
        return None
    rows = int(math.ceil(len(group_images) / cols))
    cell_w = max(im.size[0] for im in group_images)
    cell_h = max(im.size[1] for im in group_images)
    out_w = cols * cell_w + gap * (cols - 1)
    out_h = rows * cell_h + gap * (rows - 1)
    out = Image.new("RGB", (out_w, out_h), bg)

    for idx, im in enumerate(group_images):
        r = idx // cols
        c = idx % cols
        x = c * (cell_w + gap)
        y = r * (cell_h + gap)
        out.paste(im, (x, y))
    return out


# =============================================================================
# 主逻辑
# =============================================================================
def candidate_quality_score(cand):
    """
    用于 preview 排序。优先使用已有字段。
    越小越好 或 越大越好 都可能存在，所以做尽量鲁棒的组合。
    """
    # beam score 一般越小越好
    if "beam_score" in cand:
        return float(cand["beam_score"])
    # reconstruction rmse 越小越好
    if "max_prior_reconstruction_rmse_px" in cand:
        return float(cand["max_prior_reconstruction_rmse_px"])
    # 否则按 id
    return 1e9


def summarize_candidate_diversity(variants):
    """
    统计每个 style mode 相比原始几何的控制点位移。
    """
    disp_by_mode = defaultdict(list)
    node_hist = []
    for item in variants:
        mode = item.get("style_mode", "unknown")
        meta = item.get("style_mode_meta", {})
        d = meta.get("mean_control_displacement_px", None)
        if d is not None:
            disp_by_mode[mode].append(d)

        nodes = get_nodes(item)
        node_hist.append(len(nodes))

    mode_disp_stats = {k: stats_dict(v) for k, v in disp_by_mode.items()}

    return {
        "node_count_stats": stats_dict(node_hist),
        "mode_control_displacement_px_stats": mode_disp_stats
    }


def build_mode_variant(candidate, source_idx, mode_name):
    cand_id = get_candidate_id(candidate, source_idx)
    nodes = get_nodes(candidate)

    new_nodes = []
    control_disps = []
    valid_curve_count = 0

    for nd in nodes:
        bez = get_bezier_from_node(nd)
        if bez is None:
            new_nodes.append(copy.deepcopy(nd))
            continue

        shape_code = get_shape_code_from_node(nd)
        old = np.asarray(bez, dtype=np.float32)
        new = apply_style_mode_to_bezier(old, mode_name, shape_code=shape_code)

        # 只看控制点变化（P1/P2）
        ctrl_disp = float(
            0.5 * np.linalg.norm(new[1] - old[1]) +
            0.5 * np.linalg.norm(new[2] - old[2])
        )
        control_disps.append(ctrl_disp)
        valid_curve_count += 1

        nd2 = set_bezier_to_node(nd, new)
        new_nodes.append(nd2)

    out = set_nodes(candidate, new_nodes)
    out["source_candidate_id"] = cand_id
    out["style_mode"] = mode_name
    out["style_mode_rank"] = STYLE_MODES.index(mode_name)
    out["candidate_id"] = f"{cand_id}__{mode_name}"
    out["glyph_candidate_id"] = out["candidate_id"]
    out["style_mode_meta"] = {
        "valid_curve_count": valid_curve_count,
        "mean_control_displacement_px": round(float(np.mean(control_disps)) if control_disps else 0.0, 6),
        "max_control_displacement_px": round(float(np.max(control_disps)) if control_disps else 0.0, 6),
    }
    return out


def main():
    ensure_dir(PREVIEW_DIR)

    input_file = choose_input_file()
    raw = load_json(input_file)
    candidates = get_candidate_list(raw)

    if MAX_INPUT_CANDIDATES is not None:
        candidates = candidates[:MAX_INPUT_CANDIDATES]

    print("\n" + "=" * 80)
    print("TopoStyle Style Mode Explorer")
    print("=" * 80)
    print(f"  input_file:   {input_file}")
    print(f"  output_file:  {OUTPUT_FILE}")
    print(f"  report_file:  {REPORT_FILE}")
    print(f"  preview_dir:  {PREVIEW_DIR}")
    print("=" * 80)

    print("\n[Config]")
    print(f"  style_modes: {STYLE_MODES}")
    print(f"  blend_with_original: {BLEND_WITH_ORIGINAL}")
    print(f"  structural_shapes: {sorted(list(STRUCTURAL_SHAPES))}")
    print(f"  structural_shape_scale: {STRUCTURAL_SHAPE_SCALE}")
    print(f"  preview_source_limit: {PREVIEW_SOURCE_LIMIT}")

    print("\n[Input]")
    print(f"  candidate_count: {len(candidates)}")

    valid_inputs = []
    skipped = Counter()
    node_hist = Counter()

    for i, cand in enumerate(candidates):
        nodes = get_nodes(cand)
        if not nodes:
            skipped["no_nodes"] += 1
            continue

        valid_curve_num = 0
        for nd in nodes:
            if get_bezier_from_node(nd) is not None:
                valid_curve_num += 1

        if valid_curve_num == 0:
            skipped["no_bezier"] += 1
            continue

        valid_inputs.append(cand)
        node_hist[len(nodes)] += 1

    print("\n[Valid Input Candidates]")
    print(f"  valid_count: {len(valid_inputs)}")
    print(f"  skipped: {dict(skipped)}")
    print(f"  node_hist: {dict(node_hist)}")

        # -------------------------------------------------------------------------
    # Deduplicate retrieval beams:
    # 192 = 24 source topology candidates * 8 retrieval beams
    # style explorer 应该只对 unique topology 做扩展，否则会大量重复
    # -------------------------------------------------------------------------
    dedup = {}
    for i, cand in enumerate(valid_inputs):
        base_id = get_base_candidate_id(cand, i)

        # 如果有质量分数，用质量最好的那个 beam；没有就保留第一个
        q = candidate_quality_score(cand)

        if base_id not in dedup:
            dedup[base_id] = (q, cand)
        else:
            old_q, _ = dedup[base_id]
            if q < old_q:
                dedup[base_id] = (q, cand)

    before_dedup = len(valid_inputs)
    valid_inputs = [x[1] for x in dedup.values()]

    print("\n[Dedup Retrieval Beams]")
    print(f"  before_dedup: {before_dedup}")
    print(f"  after_dedup:  {len(valid_inputs)}")
    print(f"  expected_output_count: {len(valid_inputs) * len(STYLE_MODES)}")
    
    # -------------------------------------------------------------------------
    # 生成 style mode 变体
    # -------------------------------------------------------------------------
    outputs = []
    progress_total = len(valid_inputs)

    for i, cand in enumerate(valid_inputs):
        for mode in STYLE_MODES:
            out_item = build_mode_variant(cand, i, mode)
            outputs.append(out_item)

        if (i + 1) % 20 == 0 or (i + 1) == progress_total:
            print(f"  progress {i+1}/{progress_total} | outputs={len(outputs)}")

    # -------------------------------------------------------------------------
    # 统计
    # -------------------------------------------------------------------------
    mode_hist = Counter([x.get("style_mode", "unknown") for x in outputs])
    diversity = summarize_candidate_diversity(outputs)

    report = {
        "input_file": input_file,
        "input_candidate_count": len(candidates),
        "valid_input_count": len(valid_inputs),
        "skipped": dict(skipped),
        "style_modes": STYLE_MODES,
        "generated_output_count": len(outputs),
        "mode_hist": dict(mode_hist),
        "node_hist": dict(node_hist),
        "diversity": diversity,
        "notes": [
            "该脚本保持每条 Bézier 的 P0/P3 不变，因此原有 topology endpoint 关系不被破坏。",
            "它不是 retrieval，也不是神经网络推理，而是显式 style family 扩展器。",
            "如果你觉得差异还不够大，可以进一步增大 mild/strong/s_curve/hook 的振幅。"
        ]
    }

    # -------------------------------------------------------------------------
    # 保存 JSON
    # -------------------------------------------------------------------------
    save_json(outputs, OUTPUT_FILE)
    save_json(report, REPORT_FILE)

    print("\n" + "=" * 80)
    print("Style Mode Summary")
    print("=" * 80)
    print(f"  generated_output_count: {len(outputs)}")
    print(f"  mode_hist: {dict(mode_hist)}")
    print(f"  node_count_stats: {diversity.get('node_count_stats', {})}")

    print("\n[Mean control displacement by mode]")
    mode_disp = diversity.get("mode_control_displacement_px_stats", {})
    for k in STYLE_MODES:
        print(f"  {k}: {mode_disp.get(k, {})}")

    # -------------------------------------------------------------------------
    # 渲染 preview
    # -------------------------------------------------------------------------
    if not PIL_OK:
        print("\n⚠️ PIL 不可用，跳过预览图渲染。请安装 pillow。")
        print(f"JSON 已输出到: {OUTPUT_FILE}")
        print(f"Report 已输出到: {REPORT_FILE}")
        return

    preview_groups = []
    per_source = defaultdict(list)
    source_score = {}

    for item in outputs:
        src = item.get("source_candidate_id", "unknown_source")
        per_source[src].append(item)

    # 尝试从原始候选里找一个“排序分数”
    original_score_map = {}
    for i, cand in enumerate(valid_inputs):
        cid = get_candidate_id(cand, i)
        original_score_map[cid] = candidate_quality_score(cand)

    sorted_sources = sorted(
        per_source.keys(),
        key=lambda x: original_score_map.get(x, 1e9)
    )[:PREVIEW_SOURCE_LIMIT]

    print("\n" + "=" * 80)
    print("Rendering Previews")
    print("=" * 80)
    print(f"  preview_source_count: {len(sorted_sources)}")

    for src in sorted_sources:
        items = per_source[src]
        items = sorted(items, key=lambda z: z.get("style_mode_rank", 999))

        row_tiles = []
        for it in items:
            img = render_candidate_black(it, canvas_w=PREVIEW_TILE_W, canvas_h=PREVIEW_TILE_H)
            label = it.get("style_mode", "unknown")
            img = draw_text(img, label, xy=(8, 4))
            row_tiles.append(img)

        row = stack_tiles_h(row_tiles, gap=8)
        if row is None:
            continue

        title_bar = Image.new("RGB", (row.size[0], 28), "white")
        draw = ImageDraw.Draw(title_bar)
        draw.text((8, 6), f"{src}", fill=(10, 10, 10))

        group_img = stack_tiles_v([title_bar, row], gap=4)
        preview_groups.append(group_img)

        out_fp = os.path.join(PREVIEW_DIR, f"{src}.png")
        group_img.save(out_fp)

    if preview_groups:
        contact = make_contact_sheet(preview_groups, cols=1, gap=12)
        if contact is not None:
            contact_fp = os.path.join(PREVIEW_DIR, "contact_sheet.png")
            contact.save(contact_fp)
        else:
            contact_fp = None
    else:
        contact_fp = None

    print("\n" + "=" * 80)
    print("Saved")
    print("=" * 80)
    print(f"  outputs:     {OUTPUT_FILE}")
    print(f"  report:      {REPORT_FILE}")
    print(f"  preview_dir: {PREVIEW_DIR}")
    if contact_fp:
        print(f"  contact:     {contact_fp}")

    print("\nNext:")
    print("  1. 打开 topostyle_style_mode_previews/contact_sheet.png")
    print("  2. 观察同一个 source candidate 在不同 mode 下是否肉眼明显不同")
    print("  3. 如果 still 不够明显，直接增大 mode_target_local() 里的振幅")
    print("  4. 如果你满意，可以把这个脚本接到 annotation_flywheel_app 之前，作为批量生成器")


if __name__ == "__main__":
    main()