# -*- coding: utf-8 -*-
r"""
annotation_flywheel_pcg_app_no_json_v5_stage2_schema.py

完整 GUI 脚本：不读取任何现有候选 JSON，直接在内存中随机生成 N*N 个拓扑字体候选，
再由人工勾选 Good；未勾选为 Bad。

推荐放置位置：
    Char_Glyph_v0/
        annotation_flywheel_app_fixed.py          # 可选；若存在则优先复用其转换函数
        pcg_flywheel_app/
            annotation_flywheel_pcg_app_no_json.py

运行：
    cd C:\Users\Administrator\Documents\BST\Babel_Simulater\dataset_analyse_p0\Char_Glyph_v0\pcg_flywheel_app
    python annotation_flywheel_pcg_app_no_json.py

核心差异：
    这个版本不读取 topostyle_retrieval_solved_candidates.json
    不读取 solved_glyph_candidates.json
    不读取任何现有候选 JSON
    每次点击 Generate 都是直接从程序函数随机生成 topology + style

功能：
1. 直接随机生成 N*N 个字体候选。
2. 可自定义复杂度：笔画数、联通数、环数、拓扑风格、曲线风格。
3. 每个缩略图旁有 Good 勾选框；勾选=好样本，不勾=坏样本。
4. Good / Bad 样本池管理：可预览并剔除。
5. 预览为纯黑带宽度字体。
6. 保存 Good/Bad 分文件夹，80MB 智能分割 JSON。
7. 输出 JSON 尽量复用 annotation_flywheel_app_fixed.py 的格式；若不可用则生成 annotations_topo-compatible fallback。
8. 预留未来 aesthetic / topology / style 模型接口。

注意：
    生成阶段不产生中间 JSON / PNG 文件。
    只有点击 Save Pools JSON 时才写最终 good/bad JSON + manifest。
"""

import os
import re
import sys
import json
import math
import time
import copy
import random
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except Exception as e:
    raise RuntimeError("需要 tkinter。Windows Python 默认通常自带 tkinter。") from e

try:
    from PIL import Image, ImageDraw, ImageTk
except Exception as e:
    raise RuntimeError("需要 pillow：pip install pillow") from e


# =============================================================================
# 0. 路径配置：脚本通常在 Char_Glyph_v0/pcg_flywheel_app 下，用 ../ 找旧人工标注脚本
# =============================================================================

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))

# 本脚本按你的新放置方式设计：
#   Char_Glyph_v0/
#     annotation_flywheel_app_fixed.py
#     annotation_tool/
#       annotation_flywheel_pcg_app_no_json_v5_stage2_schema.py
#
# 因此默认用 ../ 找 Char_Glyph_v0 和第二阶段人工标注脚本。
if os.path.basename(TOOL_DIR).lower() == "annotation_tool":
    CHAR_GLYPH_DIR = os.path.abspath(os.path.join(TOOL_DIR, ".."))
else:
    # 兼容：如果你直接放在 Char_Glyph_v0 下，也可以运行。
    CHAR_GLYPH_DIR = TOOL_DIR

SCRIPT_DIR = TOOL_DIR
PROJECT_DIR = CHAR_GLYPH_DIR

# 把 Char_Glyph_v0 与 annotation_tool 都加入 import 路径。
for _p in [CHAR_GLYPH_DIR, TOOL_DIR, os.path.join(CHAR_GLYPH_DIR, "annotation_tool")]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# 默认输出到第二阶段人工标注同级目录下，避免混乱：
#   Babel_Simulater/dataset_analyse_p0/AI_VECTOR_ROUTER_With_topo/annotations_topo/pcg_filebacked_stage2_schema
ANNOTATIONS_TOPO_DIR = os.path.abspath(
    os.path.join(CHAR_GLYPH_DIR, "..", "AI_VECTOR_ROUTER_With_topo", "annotations_topo")
)
DEFAULT_OUTPUT_ROOT = os.path.join(ANNOTATIONS_TOPO_DIR, "pcg_filebacked_stage2_schema")

DEFAULT_MAX_JSON_MB = 80
DEFAULT_FONT_NAME = "PCG_Procedural_TopoStyle_AestheticFlywheel.ttf"

# File-backed pool prefix：Commit/Manage 都直接读写这些 active pool 文件。
POOL_PREFIX = "PCG_Direct_Stage2Schema"

# 强约束：为了尽可能等同第二阶段人工标注格式，不再静默 fallback。
# 如果无法导入 annotation_flywheel_app_fixed.py 或无法调用 candidate_to_char_bundle()，
# 程序会直接报错，而不是写出“不完全一致”的 JSON。
REQUIRE_STAGE2_CONVERTER = True

# True 表示 commit 后不额外改写 bundle 结构，只使用第二阶段函数返回的 bundle。
# label/style/topology_family 等信息主要保存在文件夹、manifest 和 candidate 输入字段中。
PRESERVE_STAGE2_BUNDLE = True

# Private Use Area 起点，用于模拟 unicode
PUA_BASE = 0xE000

CANVAS_SIZE = 400.0


# =============================================================================
# 1. 未来模型接口：当前为空实现，后续可接模型
# =============================================================================

class AestheticAndGenerationHooks:
    """
    未来可替换接口。

    后续你可以接：
    1. topology model:
        generate_topology(condition, rng) -> candidate
    2. aesthetic prefilter:
        prefilter_topology(candidate, condition) -> bool
    3. style policy:
        propose_style(candidate, condition, rng) -> style_mode
    4. aesthetic scorer:
        score_aesthetic(candidate, condition) -> float
    5. model acceptance:
        accept_by_model(candidate, score, condition) -> True/False/None
    """

    def generate_topology(
        self,
        condition: Dict[str, Any],
        rng: random.Random,
    ) -> Optional[Dict[str, Any]]:
        return None

    def prefilter_topology(
        self,
        candidate: Dict[str, Any],
        condition: Dict[str, Any],
    ) -> bool:
        return True

    def propose_style(
        self,
        candidate: Dict[str, Any],
        condition: Dict[str, Any],
        rng: random.Random,
    ) -> Optional[str]:
        return None

    def score_aesthetic(
        self,
        candidate: Dict[str, Any],
        condition: Dict[str, Any],
    ) -> Optional[float]:
        return None

    def accept_by_model(
        self,
        candidate: Dict[str, Any],
        score: Optional[float],
        condition: Dict[str, Any],
    ) -> Optional[bool]:
        # None 表示交给人工
        return None


MODEL_HOOKS = AestheticAndGenerationHooks()


# =============================================================================
# 2. 复用 annotation_flywheel_app_fixed.py 的转换函数
# =============================================================================

_ANNOTATION_APP = None


def _try_import_annotation_app():
    """
    严格导入第二阶段人工标注脚本。
    脚本位于 Char_Glyph_v0/annotation_tool 时，会从 ../Char_Glyph_v0 搜索。
    """
    global _ANNOTATION_APP
    if _ANNOTATION_APP is not None:
        return _ANNOTATION_APP if _ANNOTATION_APP is not False else None

    search_dirs = [
        CHAR_GLYPH_DIR,
        TOOL_DIR,
        os.path.join(CHAR_GLYPH_DIR, "annotation_tool"),
    ]
    for d in search_dirs:
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)

    module_names = [
        "annotation_flywheel_app_fixed",
        "annotation_flywheel_app",
    ]

    last_err = None
    for mod_name in module_names:
        try:
            mod = __import__(mod_name)
            if not hasattr(mod, "candidate_to_char_bundle"):
                last_err = RuntimeError(f"{mod_name} lacks candidate_to_char_bundle()")
                continue
            _ANNOTATION_APP = mod
            print(f"[Stage2Schema] imported {mod_name} from {getattr(mod, '__file__', '')}")
            return mod
        except Exception as e:
            last_err = e

    _ANNOTATION_APP = False
    if REQUIRE_STAGE2_CONVERTER:
        raise ImportError(
            "无法导入第二阶段人工标注转换函数。请确认文件位置：\n"
            f"  当前脚本目录: {TOOL_DIR}\n"
            f"  期望 Char_Glyph_v0: {CHAR_GLYPH_DIR}\n"
            "并确认存在：\n"
            "  ../annotation_flywheel_app_fixed.py\n"
            "且其中包含 candidate_to_char_bundle()。\n"
            f"最后错误: {repr(last_err)}"
        )
    return None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def json_size_bytes(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))


def _get_candidate_id(candidate: Dict[str, Any], idx: int = 0) -> str:
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
    return f"pcg_candidate_{idx:05d}"


def _get_nodes(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    for k in ["solved_nodes", "nodes", "strokes", "solved_segments"]:
        if isinstance(candidate.get(k), list):
            return candidate[k]
    return []


def _get_bezier_from_node(node: Dict[str, Any]) -> Optional[np.ndarray]:
    for k in ["mother_bezier", "bezier", "curve", "solved_bezier", "control_points"]:
        v = node.get(k)
        if isinstance(v, list) and len(v) == 4:
            try:
                arr = np.asarray(v, dtype=np.float32)
                if arr.shape == (4, 2):
                    return arr
            except Exception:
                pass
    return None


def _get_width_from_node(node: Dict[str, Any]) -> float:
    for k in ["width", "width_mean", "stroke_width"]:
        if k in node:
            try:
                w = float(node[k])
                if w > 0:
                    return w
            except Exception:
                pass
    if "width_norm" in node:
        try:
            w = float(node["width_norm"]) * CANVAS_SIZE
            if w > 0:
                return w
        except Exception:
            pass
    return 10.0


def candidate_to_strokes(candidate: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    优先复用第二阶段人工标注脚本的 candidate_to_strokes。
    如果旧脚本没有该函数，则仅用于 GUI 复杂度/预览时走本地解析；
    保存 JSON 时必须通过 candidate_to_char_bundle。
    """
    app = _try_import_annotation_app()
    if app is not None and hasattr(app, "candidate_to_strokes"):
        try:
            return app.candidate_to_strokes(candidate)
        except Exception as e:
            print(f"[WARN] stage2 candidate_to_strokes failed, GUI will use local parser only: {repr(e)}")

    strokes = []
    for i, nd in enumerate(_get_nodes(candidate)):
        bez = _get_bezier_from_node(nd)
        if bez is None:
            continue
        width = _get_width_from_node(nd)
        bid = nd.get("bezier_id", nd.get("node_id", i))
        try:
            bid = int(bid)
        except Exception:
            bid = i

        strokes.append({
            "bezier_id": bid,
            "mother_bezier": np.asarray(bez, dtype=np.float32).tolist(),
            "width": float(width),
            "width_norm": float(width / CANVAS_SIZE),
            "alpha": float(nd.get("alpha", 1.0)),
            "exist": bool(nd.get("exist", True)),
            "role": nd.get("role", "stroke"),
            "shape_code": nd.get("shape_code", None),
            "style_token": nd.get("style_token", None),
            "width_token": nd.get("width_token", None),
            "start_anchor": nd.get("anchor_start"),
            "end_anchor": nd.get("anchor_end"),
        })

    meta = {
        "stroke_count": len(strokes),
        "candidate_id": _get_candidate_id(candidate),
        "style_mode": candidate.get("style_mode"),
        "topology_family": candidate.get("topology_family"),
    }
    return strokes, meta


def candidate_to_char_bundle(candidate: Dict[str, Any], label: str, hex_key: str) -> Tuple[str, Dict[str, Any]]:
    """
    严格复用第二阶段人工标注脚本的 candidate_to_char_bundle。
    不再静默 fallback 写近似结构，避免 Good/Bad 与第二阶段格式不一致。

    调用前会把 PCG candidate 补成旧标注脚本更可能识别的字段：
      - char / unicode_hex / font_path / font_name / label
      - strokes / solved_nodes / nodes
      - topology / cycles / edit_history
    """
    app = _try_import_annotation_app()
    if app is None or not hasattr(app, "candidate_to_char_bundle"):
        raise RuntimeError("第二阶段 candidate_to_char_bundle() 不存在，拒绝保存非第二阶段格式 JSON。")

    c = copy.deepcopy(candidate)
    c["unicode_hex"] = hex_key
    c["char"] = chr(int(hex_key, 16))
    c["font_path"] = c.get("font_path", DEFAULT_FONT_NAME)
    c["font_name"] = c.get("font_name", DEFAULT_FONT_NAME)
    c["label"] = label

    try:
        strokes, _ = candidate_to_strokes(c)
        c.setdefault("strokes", strokes)
    except Exception:
        pass

    fn = app.candidate_to_char_bundle
    errors = []

    for call in [
        lambda: fn(c, label, hex_key),
        lambda: fn(c, label),
        lambda: fn(c),
    ]:
        try:
            result = call()
            if isinstance(result, tuple) and len(result) == 2:
                out_hex, bundle = result
                return str(out_hex), bundle
            if isinstance(result, dict):
                return hex_key, result
        except TypeError as e:
            errors.append(repr(e))
        except Exception as e:
            errors.append(repr(e))

    raise RuntimeError(
        "第二阶段 candidate_to_char_bundle() 调用失败，拒绝使用 fallback 保存。\n"
        "请检查 annotation_flywheel_app_fixed.py 的函数签名，或把函数名/签名告诉我做适配。\n"
        "尝试过签名: (candidate,label,hex_key), (candidate,label), (candidate)\n"
        f"errors={errors}"
    )


# =============================================================================
# 3. Procedural topology + style generator
# =============================================================================

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
    "mixed",
]

TOPOLOGY_FAMILIES = [
    "chain",
    "fork",
    "star",
    "zigzag",
    "triangle",
    "box",
    "rune_cross",
    "ladder",
    "parallel_slash",
    "arc_spine",
    "random_tree",
    "cycle_with_tail",
]


@dataclass
class GeneratorConfig:
    stroke_min: int = 3
    stroke_max: int = 8
    cc_min: int = 1
    cc_max: int = 1
    cycle_min: int = 0
    cycle_max: int = 2
    width_min: float = 7.0
    width_max: float = 14.0
    jitter: float = 16.0
    style_modes: List[str] = field(default_factory=lambda: ["straight", "mild_left", "mild_right", "s_curve_left", "s_curve_right", "hook_left", "hook_right", "mixed"])
    topology_families: List[str] = field(default_factory=lambda: TOPOLOGY_FAMILIES.copy())
    aesthetic_condition: Dict[str, Any] = field(default_factory=dict)
    topology_condition: Dict[str, Any] = field(default_factory=dict)
    style_condition: Dict[str, Any] = field(default_factory=dict)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def rand_pt(rng: random.Random, xlo=70, xhi=330, ylo=70, yhi=330) -> np.ndarray:
    return np.array([rng.uniform(xlo, xhi), rng.uniform(ylo, yhi)], dtype=np.float32)


def rotate(v: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float32)


def normalize(v: np.ndarray, eps=1e-6) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.array([1.0, 0.0], dtype=np.float32)
    return v / n


def add_anchor(anchors: List[np.ndarray], p: np.ndarray) -> int:
    p = np.asarray(p, dtype=np.float32)
    p[0] = clamp(float(p[0]), 35.0, CANVAS_SIZE - 35.0)
    p[1] = clamp(float(p[1]), 35.0, CANVAS_SIZE - 35.0)
    anchors.append(p)
    return len(anchors) - 1


def edge_dict(u: int, v: int, anchor_u: int, anchor_v: int) -> Dict[str, Any]:
    return {
        "u": int(u),
        "v": int(v),
        "src": int(u),
        "dst": int(v),
        "source": int(u),
        "target": int(v),
        "j_type": "E2E",
        "type": "E2E",
        "anchor_u": int(anchor_u),
        "anchor_v": int(anchor_v),
    }


def local_style_params(mode: str, rng: random.Random) -> Tuple[float, float, float, float]:
    """
    返回 normalized local 控制点：
        P1=(x1,y1), P2=(x2,y2)
    """
    if mode == "straight":
        return 0.33, rng.uniform(-0.02, 0.02), 0.67, rng.uniform(-0.02, 0.02)
    if mode == "mild_left":
        a = rng.uniform(0.10, 0.20)
        return rng.uniform(0.25, 0.38), +a, rng.uniform(0.58, 0.75), +a * rng.uniform(0.75, 1.15)
    if mode == "mild_right":
        a = rng.uniform(0.10, 0.20)
        return rng.uniform(0.25, 0.38), -a, rng.uniform(0.58, 0.75), -a * rng.uniform(0.75, 1.15)
    if mode == "strong_left":
        a = rng.uniform(0.22, 0.38)
        return rng.uniform(0.20, 0.35), +a, rng.uniform(0.60, 0.82), +a * rng.uniform(0.70, 1.20)
    if mode == "strong_right":
        a = rng.uniform(0.22, 0.38)
        return rng.uniform(0.20, 0.35), -a, rng.uniform(0.60, 0.82), -a * rng.uniform(0.70, 1.20)
    if mode == "s_curve_left":
        a = rng.uniform(0.18, 0.34)
        return rng.uniform(0.22, 0.38), +a, rng.uniform(0.58, 0.78), -a * rng.uniform(0.70, 1.15)
    if mode == "s_curve_right":
        a = rng.uniform(0.18, 0.34)
        return rng.uniform(0.22, 0.38), -a, rng.uniform(0.58, 0.78), +a * rng.uniform(0.70, 1.15)
    if mode == "hook_left":
        return rng.uniform(0.14, 0.26), rng.uniform(0.28, 0.48), rng.uniform(0.55, 0.80), rng.uniform(0.02, 0.14)
    if mode == "hook_right":
        return rng.uniform(0.14, 0.26), -rng.uniform(0.28, 0.48), rng.uniform(0.55, 0.80), -rng.uniform(0.02, 0.14)

    # mixed
    return local_style_params(rng.choice(STYLE_MODES[:-1]), rng)


def bezier_from_anchors(p0: np.ndarray, p3: np.ndarray, mode: str, rng: random.Random, jitter=0.0) -> np.ndarray:
    p0 = np.asarray(p0, dtype=np.float32)
    p3 = np.asarray(p3, dtype=np.float32)
    v = p3 - p0
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        v = np.array([1.0, 0.0], dtype=np.float32)
        L = 1.0
    ex = v / L
    ey = np.array([-ex[1], ex[0]], dtype=np.float32)
    x1, y1, x2, y2 = local_style_params(mode, rng)

    p1 = p0 + x1 * L * ex + y1 * L * ey
    p2 = p0 + x2 * L * ex + y2 * L * ey

    if jitter > 0:
        # 小扰动，不移动端点
        p1 = p1 + np.array([rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)], dtype=np.float32)
        p2 = p2 + np.array([rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)], dtype=np.float32)

    return np.stack([p0, p1, p2, p3], axis=0).astype(np.float32)


def make_nodes_from_anchor_edges(
    anchors: List[np.ndarray],
    anchor_edges: List[Tuple[int, int]],
    style_mode: str,
    rng: random.Random,
    cfg: GeneratorConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    nodes = []
    topo_edges = []
    incident = defaultdict(list)

    for i, (a, b) in enumerate(anchor_edges):
        p0, p3 = anchors[a], anchors[b]
        mode = style_mode
        if style_mode == "mixed":
            mode = rng.choice([m for m in cfg.style_modes if m != "mixed"] or STYLE_MODES[:-1])

        P = bezier_from_anchors(p0, p3, mode, rng, jitter=cfg.jitter * 0.18)
        width = rng.uniform(cfg.width_min, cfg.width_max)

        nd = {
            "node_id": i,
            "bezier_id": i,
            "anchor_start": int(a),
            "anchor_end": int(b),
            "mother_bezier": P.tolist(),
            "control_points": P.tolist(),
            "width": float(width),
            "width_norm": float(width / CANVAS_SIZE),
            "alpha": 1.0,
            "exist": True,
            "shape_code": 20 if mode == "straight" else 30,
            "style_mode": mode,
            "style_token": mode,
            "width_token": int(round(width)),
        }
        nodes.append(nd)
        incident[a].append(i)
        incident[b].append(i)

    seen = set()
    for anchor_id, segs in incident.items():
        segs = sorted(set(segs))
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                u, v = segs[i], segs[j]
                key = (min(u, v), max(u, v), int(anchor_id))
                if key in seen:
                    continue
                seen.add(key)
                topo_edges.append(edge_dict(u, v, int(anchor_id), int(anchor_id)))

    anchor_json = [{"anchor_id": i, "xy": np.asarray(p, dtype=np.float32).tolist()} for i, p in enumerate(anchors)]
    return nodes, topo_edges, anchor_json


def graph_complexity_from_anchor_edges(stroke_count: int, anchor_edges: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    返回 connected_components, cycle_count。这里按 stroke graph 的共享 anchor 关系计算。
    """
    if stroke_count <= 0:
        return 0, 0

    incident = defaultdict(list)
    for sid, (a, b) in enumerate(anchor_edges):
        incident[a].append(sid)
        incident[b].append(sid)

    parent = list(range(stroke_count))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edge_count_seg_graph = 0
    seen = set()
    for anchor_id, segs in incident.items():
        segs = sorted(set(segs))
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                u, v = segs[i], segs[j]
                union(u, v)
                key = (min(u, v), max(u, v), anchor_id)
                if key not in seen:
                    seen.add(key)
                    edge_count_seg_graph += 1

    cc = len(set(find(i) for i in range(stroke_count)))
    cycles = max(0, edge_count_seg_graph - stroke_count + cc)
    return cc, cycles


def generate_anchor_edges_for_family(
    family: str,
    stroke_count: int,
    cycles_target: int,
    rng: random.Random,
    cfg: GeneratorConfig,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    直接生成 anchor graph。一个 stroke = 一条 anchor edge。
    """
    anchors: List[np.ndarray] = []
    edges: List[Tuple[int, int]] = []

    cx, cy = rng.uniform(145, 255), rng.uniform(145, 255)
    base = np.array([cx, cy], dtype=np.float32)
    scale = rng.uniform(75, 135)
    theta = rng.uniform(0, 2 * math.pi)

    def P(x, y):
        v = np.array([x, y], dtype=np.float32) * scale
        v = rotate(v, theta)
        v = base + v + np.array([rng.uniform(-cfg.jitter, cfg.jitter), rng.uniform(-cfg.jitter, cfg.jitter)], dtype=np.float32)
        v[0] = clamp(float(v[0]), 35, CANVAS_SIZE - 35)
        v[1] = clamp(float(v[1]), 35, CANVAS_SIZE - 35)
        return v

    if family == "chain":
        # random walk chain
        pts = [P(-0.7, 0.0)]
        direction = normalize(np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)], dtype=np.float32))
        for i in range(stroke_count):
            direction = normalize(rotate(direction, rng.uniform(-0.95, 0.95)))
            step = rng.uniform(0.45, 0.85) * scale
            newp = pts[-1] + direction * step
            newp[0] = clamp(float(newp[0]), 40, CANVAS_SIZE - 40)
            newp[1] = clamp(float(newp[1]), 40, CANVAS_SIZE - 40)
            pts.append(newp.astype(np.float32))
        ids = [add_anchor(anchors, p) for p in pts]
        edges = [(ids[i], ids[i + 1]) for i in range(stroke_count)]

    elif family == "fork":
        root = add_anchor(anchors, P(0, 0))
        branch_n = min(stroke_count, rng.choice([3, 4, 5]))
        tips = []
        for i in range(branch_n):
            ang = theta + (2 * math.pi * i / branch_n) + rng.uniform(-0.35, 0.35)
            tip = base + np.array([math.cos(ang), math.sin(ang)], dtype=np.float32) * rng.uniform(75, 145)
            tips.append(add_anchor(anchors, tip))
            edges.append((root, tips[-1]))
        # 多出来的 stroke 接在随机 tip 上
        while len(edges) < stroke_count:
            a = rng.choice(tips + [root])
            p = anchors[a] + normalize(anchors[a] - anchors[root] + np.array([rng.uniform(-30, 30), rng.uniform(-30, 30)], dtype=np.float32)) * rng.uniform(50, 100)
            b = add_anchor(anchors, p)
            edges.append((a, b))
            tips.append(b)

    elif family == "star":
        root = add_anchor(anchors, P(0, 0))
        for i in range(stroke_count):
            ang = theta + 2 * math.pi * i / max(stroke_count, 1) + rng.uniform(-0.22, 0.22)
            p = base + np.array([math.cos(ang), math.sin(ang)], dtype=np.float32) * rng.uniform(80, 145)
            b = add_anchor(anchors, p)
            edges.append((root, b))

    elif family == "zigzag":
        pts = []
        for i in range(stroke_count + 1):
            x = -0.8 + 1.6 * i / max(stroke_count, 1)
            y = (0.38 if i % 2 == 0 else -0.38) * rng.uniform(0.75, 1.15)
            pts.append(P(x, y))
        ids = [add_anchor(anchors, p) for p in pts]
        edges = [(ids[i], ids[i + 1]) for i in range(stroke_count)]

    elif family == "triangle":
        ids = [
            add_anchor(anchors, P(-0.65, -0.45)),
            add_anchor(anchors, P(0.65, -0.35)),
            add_anchor(anchors, P(0.05, 0.65)),
        ]
        edges = [(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])]
        while len(edges) < stroke_count:
            a = rng.choice(ids)
            b = add_anchor(anchors, anchors[a] + np.array([rng.uniform(-80, 80), rng.uniform(-80, 80)], dtype=np.float32))
            edges.append((a, b))

    elif family == "box":
        ids = [
            add_anchor(anchors, P(-0.6, -0.55)),
            add_anchor(anchors, P(0.6, -0.55)),
            add_anchor(anchors, P(0.6, 0.55)),
            add_anchor(anchors, P(-0.6, 0.55)),
        ]
        edges = [(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[3]), (ids[3], ids[0])]
        while len(edges) < stroke_count:
            a = rng.choice(ids)
            # chord / tail
            if rng.random() < 0.45:
                b = rng.choice([x for x in ids if x != a])
            else:
                b = add_anchor(anchors, anchors[a] + np.array([rng.uniform(-75, 75), rng.uniform(-75, 75)], dtype=np.float32))
            edges.append((a, b))

    elif family == "rune_cross":
        center = add_anchor(anchors, P(0, 0))
        top = add_anchor(anchors, P(0, 0.75))
        bottom = add_anchor(anchors, P(0, -0.75))
        left = add_anchor(anchors, P(-0.75, 0.0))
        right = add_anchor(anchors, P(0.75, 0.0))
        diag1 = add_anchor(anchors, P(-0.55, 0.55))
        diag2 = add_anchor(anchors, P(0.55, -0.55))
        pool = [(bottom, center), (center, top), (left, center), (center, right), (diag1, center), (center, diag2), (left, top), (right, bottom)]
        rng.shuffle(pool)
        edges = pool[:stroke_count]
        while len(edges) < stroke_count:
            a = rng.choice([center, top, bottom, left, right])
            b = add_anchor(anchors, anchors[a] + np.array([rng.uniform(-90, 90), rng.uniform(-90, 90)], dtype=np.float32))
            edges.append((a, b))

    elif family == "ladder":
        left1 = add_anchor(anchors, P(-0.45, -0.75))
        left2 = add_anchor(anchors, P(-0.45, 0.75))
        right1 = add_anchor(anchors, P(0.45, -0.75))
        right2 = add_anchor(anchors, P(0.45, 0.75))
        edges = [(left1, left2), (right1, right2)]
        rung_count = max(1, stroke_count - 2)
        for i in range(rung_count):
            t = (i + 1) / (rung_count + 1)
            a = add_anchor(anchors, anchors[left1] * (1 - t) + anchors[left2] * t)
            b = add_anchor(anchors, anchors[right1] * (1 - t) + anchors[right2] * t)
            edges.append((a, b))
        edges = edges[:stroke_count]

    elif family == "parallel_slash":
        k = stroke_count
        for i in range(k):
            x = -0.55 + 1.1 * i / max(k - 1, 1)
            a = add_anchor(anchors, P(x - 0.15, -0.7))
            b = add_anchor(anchors, P(x + 0.15, 0.7))
            edges.append((a, b))
        # 让部分平行线有共享连接，避免全 disconnected
        if k >= 3 and rng.random() < 0.65:
            # 加一条横连，替换最后一条
            edges[-1] = (edges[0][0], edges[-2][1])

    elif family == "arc_spine":
        # 一条主链 + 分支
        pts = []
        for i in range(max(3, min(stroke_count + 1, 6))):
            t = i / max(1, min(stroke_count, 5))
            x = -0.75 + 1.5 * t
            y = 0.35 * math.sin(t * math.pi * rng.uniform(0.8, 1.4))
            pts.append(P(x, y))
        ids = [add_anchor(anchors, p) for p in pts]
        for i in range(len(ids) - 1):
            edges.append((ids[i], ids[i + 1]))
        while len(edges) < stroke_count:
            a = rng.choice(ids)
            dirv = normalize(anchors[a] - base)
            if np.linalg.norm(dirv) < 1e-6:
                dirv = normalize(np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)], dtype=np.float32))
            b = add_anchor(anchors, anchors[a] + rotate(dirv, rng.uniform(-0.8, 0.8)) * rng.uniform(55, 115))
            edges.append((a, b))

    elif family == "cycle_with_tail":
        ncycle = rng.choice([3, 4, 5])
        ids = []
        for i in range(ncycle):
            ang = 2 * math.pi * i / ncycle + theta
            ids.append(add_anchor(anchors, base + np.array([math.cos(ang), math.sin(ang)], dtype=np.float32) * rng.uniform(65, 115)))
        edges = [(ids[i], ids[(i + 1) % ncycle]) for i in range(ncycle)]
        while len(edges) < stroke_count:
            a = rng.choice(ids)
            b = add_anchor(anchors, anchors[a] + np.array([rng.uniform(-110, 110), rng.uniform(-110, 110)], dtype=np.float32))
            edges.append((a, b))
            ids.append(b)

    else:
        # random_tree
        root = add_anchor(anchors, P(0, 0))
        active = [root]
        while len(edges) < stroke_count:
            a = rng.choice(active)
            ang = rng.uniform(0, 2 * math.pi)
            step = rng.uniform(50, 115)
            p = anchors[a] + np.array([math.cos(ang), math.sin(ang)], dtype=np.float32) * step
            b = add_anchor(anchors, p)
            edges.append((a, b))
            active.append(b)
            if len(active) > 8:
                active.pop(0)

    # 调整数量
    if len(edges) > stroke_count:
        # 保留前 stroke_count 条
        edges = edges[:stroke_count]

    # 如果要求环，但当前没有，尝试添加/替换一条 chord
    cc, cyc = graph_complexity_from_anchor_edges(len(edges), edges)
    if cycles_target > 0 and cyc < cycles_target and len(anchors) >= 3 and len(edges) >= 3:
        attempts = 0
        while len(edges) < stroke_count and cyc < cycles_target and attempts < 20:
            a, b = rng.sample(range(len(anchors)), 2)
            if (a, b) not in edges and (b, a) not in edges:
                edges.append((a, b))
                cc, cyc = graph_complexity_from_anchor_edges(len(edges), edges)
            attempts += 1
        # 如果已经满了，替换最后一条成 chord
        attempts = 0
        while cyc < cycles_target and attempts < 20:
            a, b = rng.sample(range(len(anchors)), 2)
            if (a, b) not in edges and (b, a) not in edges:
                edges[-1] = (a, b)
                cc, cyc = graph_complexity_from_anchor_edges(len(edges), edges)
            attempts += 1

    return anchors, edges


def split_into_components(
    stroke_count: int,
    cc_target: int,
    rng: random.Random,
) -> List[int]:
    cc_target = max(1, min(cc_target, stroke_count))
    sizes = [1] * cc_target
    remain = stroke_count - cc_target
    for _ in range(remain):
        sizes[rng.randrange(cc_target)] += 1
    rng.shuffle(sizes)
    return sizes


def translate_component(anchors: List[np.ndarray], dx: float, dy: float):
    for p in anchors:
        p[0] = clamp(float(p[0] + dx), 35, CANVAS_SIZE - 35)
        p[1] = clamp(float(p[1] + dy), 35, CANVAS_SIZE - 35)


def generate_procedural_candidate(cfg: GeneratorConfig, rng: random.Random, seq_idx: int) -> Dict[str, Any]:
    """
    直接从函数随机生成 topology + Bezier，不读取任何候选 JSON。
    """
    # 未来模型优先
    generated = MODEL_HOOKS.generate_topology(cfg.topology_condition, rng)
    if generated is not None:
        return generated

    stroke_count = rng.randint(int(cfg.stroke_min), int(cfg.stroke_max))
    cc_target = rng.randint(int(cfg.cc_min), max(int(cfg.cc_min), int(cfg.cc_max)))
    cc_target = max(1, min(cc_target, stroke_count))

    cycle_target = rng.randint(int(cfg.cycle_min), max(int(cfg.cycle_min), int(cfg.cycle_max)))
    style_mode = MODEL_HOOKS.propose_style(
        {},
        cfg.style_condition,
        rng,
    )
    if not style_mode:
        style_mode = rng.choice(cfg.style_modes)

    # 根据 stroke_count / cc_target 分组件
    comp_sizes = split_into_components(stroke_count, cc_target, rng)

    all_anchors: List[np.ndarray] = []
    all_edges: List[Tuple[int, int]] = []
    families = []
    anchor_offset = 0

    # 多组件时分散布局
    comp_centers = []
    if cc_target == 1:
        comp_centers = [(0, 0)]
    else:
        for i in range(cc_target):
            ang = 2 * math.pi * i / cc_target + rng.uniform(-0.4, 0.4)
            comp_centers.append((math.cos(ang) * rng.uniform(45, 85), math.sin(ang) * rng.uniform(45, 85)))

    remaining_cycles = cycle_target

    for comp_i, size in enumerate(comp_sizes):
        family = rng.choice(cfg.topology_families)
        families.append(family)

        comp_cycle = 0
        if remaining_cycles > 0 and size >= 3 and rng.random() < 0.65:
            comp_cycle = min(remaining_cycles, rng.choice([1, 1, 2]))
            remaining_cycles -= comp_cycle

        anchors, edges = generate_anchor_edges_for_family(family, size, comp_cycle, rng, cfg)

        dx, dy = comp_centers[comp_i]
        translate_component(anchors, dx, dy)

        all_anchors.extend(anchors)
        all_edges.extend([(a + anchor_offset, b + anchor_offset) for a, b in edges])
        anchor_offset += len(anchors)

    # 如果由于函数产生数量差异，修正到 stroke_count
    if len(all_edges) > stroke_count:
        all_edges = all_edges[:stroke_count]
    while len(all_edges) < stroke_count:
        # 加随机尾巴到已有 anchor
        if not all_anchors:
            add_anchor(all_anchors, rand_pt(rng))
        a = rng.randrange(len(all_anchors))
        p = all_anchors[a] + np.array([rng.uniform(-95, 95), rng.uniform(-95, 95)], dtype=np.float32)
        b = add_anchor(all_anchors, p)
        all_edges.append((a, b))

    nodes, topo_edges, anchor_json = make_nodes_from_anchor_edges(
        all_anchors,
        all_edges,
        style_mode,
        rng,
        cfg,
    )

    cc, cycles = graph_complexity_from_anchor_edges(len(all_edges), all_edges)
    cid = f"pcg_direct_{int(time.time()*1000)%100000000}_{seq_idx:05d}_{rng.randint(0,999999):06d}"

    candidate = {
        "schema_version": "pcg_direct_candidate_v1",
        "candidate_id": cid,
        "glyph_candidate_id": cid,
        "generated_glyph_id": cid,
        "source_candidate_id": cid,
        "topology_family": "+".join(families),
        "style_mode": style_mode,
        "font_path": DEFAULT_FONT_NAME,
        "font_name": DEFAULT_FONT_NAME,
        "anchors": anchor_json,
        "anchor_edges": [{"anchor_start": int(a), "anchor_end": int(b)} for a, b in all_edges],
        "nodes": nodes,
        "solved_nodes": nodes,
        "solved_segments": nodes,
        "topology": {
            "positive_edges_undirected": topo_edges,
            "edges": topo_edges,
            "connections": topo_edges,
            "t_junctions": [],
            "cycles": [],
            "anchors": anchor_json,
        },
        "topostyle_topology": {
            "segment_edges": [
                {
                    "segment_id": int(i),
                    "node_id": int(i),
                    "anchor_start": int(a),
                    "anchor_end": int(b),
                }
                for i, (a, b) in enumerate(all_edges)
            ],
            "anchors": anchor_json,
        },
        "cycles": [],
        "pcg_meta": {
            "generator": "procedural_topology_direct",
            "stroke_count_target": stroke_count,
            "connected_components_target": cc_target,
            "cycle_target": cycle_target,
            "connected_components_actual": cc,
            "cycle_count_actual": cycles,
            "families": families,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    return candidate


# =============================================================================
# 4. 渲染
# =============================================================================

def cubic_point(P: np.ndarray, t: float) -> np.ndarray:
    mt = 1.0 - t
    return (
        (mt ** 3) * P[0]
        + 3 * (mt ** 2) * t * P[1]
        + 3 * mt * (t ** 2) * P[2]
        + (t ** 3) * P[3]
    )


def sample_bezier(P: np.ndarray, n: int = 72) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, n)
    return np.stack([cubic_point(P, float(t)) for t in ts], axis=0)


def candidate_curve_samples(candidate: Dict[str, Any]) -> Tuple[List[np.ndarray], List[float]]:
    curves = []
    widths = []
    for nd in _get_nodes(candidate):
        P = _get_bezier_from_node(nd)
        if P is None:
            continue
        curves.append(sample_bezier(P, n=72))
        widths.append(_get_width_from_node(nd))
    return curves, widths


def render_candidate_black_width(candidate: Dict[str, Any], size: int = 170, margin: int = 18) -> Image.Image:
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    curves, widths = candidate_curve_samples(candidate)
    if not curves:
        draw.text((10, 10), "NO CURVE", fill=(0, 0, 0))
        return img

    all_pts = np.concatenate(curves, axis=0)
    mn = np.min(all_pts, axis=0)
    mx = np.max(all_pts, axis=0)
    span = np.maximum(mx - mn, 1e-5)
    scale = min((size - 2 * margin) / span[0], (size - 2 * margin) / span[1])

    def mp(p):
        x = (p[0] - mn[0]) * scale + margin
        y = (p[1] - mn[1]) * scale + margin
        y = size - y
        return (float(x), float(y))

    med_w = max(float(np.median(widths)) if widths else 10.0, 1e-3)

    for pts, w in zip(curves, widths):
        xy = [mp(p) for p in pts]
        line_w = int(round(max(2.0, min(12.0, 5.0 * float(w) / med_w))))
        try:
            draw.line(xy, fill=(0, 0, 0), width=line_w, joint="curve")
        except Exception:
            draw.line(xy, fill=(0, 0, 0), width=line_w)

    return img


def render_candidate_color_width(candidate: Dict[str, Any], size: int = 170, margin: int = 18) -> Image.Image:
    """
    彩色带宽度预览：每条笔画不同颜色，便于判断笔画宽度、交叉、拓扑连接。
    """
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    curves, widths = candidate_curve_samples(candidate)
    if not curves:
        draw.text((10, 10), "NO CURVE", fill=(0, 0, 0))
        return img

    all_pts = np.concatenate(curves, axis=0)
    mn = np.min(all_pts, axis=0)
    mx = np.max(all_pts, axis=0)
    span = np.maximum(mx - mn, 1e-5)
    scale = min((size - 2 * margin) / span[0], (size - 2 * margin) / span[1])

    def mp(p):
        x = (p[0] - mn[0]) * scale + margin
        y = (p[1] - mn[1]) * scale + margin
        y = size - y
        return (float(x), float(y))

    palette = [
        (220, 20, 60),
        (30, 144, 255),
        (34, 139, 34),
        (255, 140, 0),
        (138, 43, 226),
        (0, 170, 170),
        (180, 90, 20),
        (210, 40, 160),
        (80, 80, 80),
        (20, 120, 220),
    ]

    med_w = max(float(np.median(widths)) if widths else 10.0, 1e-3)

    # 先画浅灰底影，增强整体字形感
    for pts, w in zip(curves, widths):
        xy = [mp(p) for p in pts]
        line_w = int(round(max(2.0, min(13.0, 5.0 * float(w) / med_w))))
        try:
            draw.line(xy, fill=(210, 210, 210), width=line_w + 2, joint="curve")
        except Exception:
            draw.line(xy, fill=(210, 210, 210), width=line_w + 2)

    # 再画彩色笔画
    for i, (pts, w) in enumerate(zip(curves, widths)):
        xy = [mp(p) for p in pts]
        line_w = int(round(max(2.0, min(12.0, 5.0 * float(w) / med_w))))
        color = palette[i % len(palette)]
        try:
            draw.line(xy, fill=color, width=line_w, joint="curve")
        except Exception:
            draw.line(xy, fill=color, width=line_w)

    return img


def render_candidate_dual_preview(candidate: Dict[str, Any], size: int = 150) -> Image.Image:
    """
    左：纯黑带宽度；右：彩色带宽度。
    """
    black = render_candidate_black_width(candidate, size=size)
    color = render_candidate_color_width(candidate, size=size)
    gap = 8
    out = Image.new("RGB", (size * 2 + gap, size), "white")
    out.paste(black, (0, 0))
    out.paste(color, (size + gap, 0))
    draw = ImageDraw.Draw(out)
    draw.text((4, 4), "black", fill=(0, 0, 0))
    draw.text((size + gap + 4, 4), "color", fill=(0, 0, 0))
    return out


def bundle_to_candidate(bundle: Dict[str, Any], hex_key: str = "") -> Dict[str, Any]:
    """
    把磁盘中的 annotations_topo bundle 反转成临时 candidate，用于 Manage 直接从文件预览。
    """
    strokes = bundle.get("strokes", [])
    if not isinstance(strokes, list):
        strokes = []

    nodes = []
    for i, st in enumerate(strokes):
        if not isinstance(st, dict):
            continue
        bez = st.get("mother_bezier", st.get("bezier", st.get("control_points")))
        if not (isinstance(bez, list) and len(bez) == 4):
            continue
        width = st.get("width", st.get("width_mean", st.get("stroke_width", 10.0)))
        try:
            width = float(width)
        except Exception:
            width = 10.0

        nodes.append({
            "node_id": i,
            "bezier_id": st.get("bezier_id", i),
            "mother_bezier": bez,
            "control_points": bez,
            "width": width,
            "width_norm": st.get("width_norm", width / CANVAS_SIZE),
            "alpha": st.get("alpha", 1.0),
            "exist": st.get("exist", True),
            "style_token": st.get("style_token"),
            "shape_code": st.get("shape_code"),
        })

    gi = bundle.get("glyph_info", {}) if isinstance(bundle.get("glyph_info"), dict) else {}
    cid = gi.get("candidate_id", f"stored_{hex_key}")

    return {
        "candidate_id": cid,
        "glyph_candidate_id": cid,
        "generated_glyph_id": cid,
        "style_mode": gi.get("style_mode", ""),
        "topology_family": gi.get("topology_family", ""),
        "nodes": nodes,
        "solved_nodes": nodes,
        "pcg_meta": gi.get("pcg_meta", {}),
    }


# =============================================================================
# 5. 复杂度计算
# =============================================================================

def candidate_edges(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    topo = candidate.get("topology", {})
    if isinstance(topo, dict):
        for k in ["positive_edges_undirected", "edges", "connections"]:
            if isinstance(topo.get(k), list):
                return [e for e in topo[k] if isinstance(e, dict)]
    return []


def _edge_uv(edge: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    def si(x, default=-1):
        try:
            return int(x)
        except Exception:
            return default

    u = si(edge.get("u", edge.get("src", edge.get("source", edge.get("node_u", -1)))))
    v = si(edge.get("v", edge.get("dst", edge.get("target", edge.get("node_v", -1)))))
    if u < 0 or v < 0 or u == v:
        return None
    return u, v


def connected_components_from_edges(stroke_count: int, edges: List[Dict[str, Any]]) -> int:
    if stroke_count <= 0:
        return 0
    parent = list(range(stroke_count))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        if 0 <= a < stroke_count and 0 <= b < stroke_count:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

    for e in edges:
        uv = _edge_uv(e)
        if uv is None:
            continue
        u, v = uv
        union(u, v)

    return len(set(find(i) for i in range(stroke_count)))


def candidate_complexity(candidate: Dict[str, Any]) -> Dict[str, Any]:
    stroke_count = len(_get_nodes(candidate))
    edges = candidate_edges(candidate)
    edge_count = len(edges)
    cc = connected_components_from_edges(stroke_count, edges)
    cycles = max(0, edge_count - stroke_count + cc)
    return {
        "stroke_count": int(stroke_count),
        "edge_count": int(edge_count),
        "connected_components": int(cc),
        "cycle_count": int(cycles),
        "style_mode": candidate.get("style_mode"),
        "topology_family": candidate.get("topology_family"),
    }


def in_range(x: int, lo: Optional[int], hi: Optional[int]) -> bool:
    if lo is not None and x < int(lo):
        return False
    if hi is not None and x > int(hi):
        return False
    return True


def passes_complexity(candidate: Dict[str, Any], cfg: GeneratorConfig) -> Tuple[bool, Dict[str, Any], str]:
    c = candidate_complexity(candidate)
    rules = [
        ("stroke_count", cfg.stroke_min, cfg.stroke_max),
        ("connected_components", cfg.cc_min, cfg.cc_max),
        ("cycle_count", cfg.cycle_min, cfg.cycle_max),
    ]
    for key, lo, hi in rules:
        if not in_range(c[key], lo, hi):
            return False, c, f"{key}_out_of_range"
    return True, c, "ok"


# =============================================================================
# 6. 80MB 分文件
# =============================================================================

def write_split_json_pool(
    data_dict: Dict[str, Any],
    out_dir: str,
    prefix: str,
    max_json_mb: int = 80,
    clear_old: bool = False,
) -> List[str]:
    ensure_dir(out_dir)

    if clear_old:
        for fn in os.listdir(out_dir):
            if fn.startswith(prefix) and fn.endswith("_topo.json"):
                try:
                    os.remove(os.path.join(out_dir, fn))
                except Exception:
                    pass

    max_bytes = int(max_json_mb * 1024 * 1024)
    files = []
    current = {}
    part = 0

    def flush(d, idx):
        if not d:
            return None
        fp = os.path.join(out_dir, f"{prefix}_part{idx:03d}_topo.json")
        save_json(d, fp)
        return fp

    for k, v in data_dict.items():
        test = dict(current)
        test[k] = v
        if current and json_size_bytes(test) > max_bytes:
            fp = flush(current, part)
            if fp:
                files.append(fp)
            part += 1
            current = {k: v}
        else:
            current = test

    fp = flush(current, part)
    if fp:
        files.append(fp)
    return files


# =============================================================================
# 6.5 文件直读直写 Good/Bad Pool
# =============================================================================

def pool_label_dir(out_root: str, label: str) -> str:
    return os.path.join(out_root, "good" if label == "good" else "bad")


def pool_file_prefix(label: str) -> str:
    return f"{POOL_PREFIX}_{'AestheticGood' if label == 'good' else 'AestheticBad'}"


def load_pool_dict_from_disk(out_root: str, label: str) -> Dict[str, Any]:
    """
    Manage 直接调用这个函数读取磁盘文件。
    只读取本脚本维护的 active pool 文件：
        good/PCG_Direct_FileBacked_AestheticGood_part000_topo.json
        bad/PCG_Direct_FileBacked_AestheticBad_part000_topo.json
    """
    folder = pool_label_dir(out_root, label)
    prefix = pool_file_prefix(label)
    if not os.path.exists(folder):
        return {}

    merged: Dict[str, Any] = {}
    for fn in sorted(os.listdir(folder)):
        if not (fn.startswith(prefix + "_part") and fn.endswith("_topo.json")):
            continue
        fp = os.path.join(folder, fn)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                merged.update(data)
        except Exception:
            print(f"[WARN] failed to load pool file: {fp}")
    return merged


def write_pool_dict_to_disk(out_root: str, label: str, pool_dict: Dict[str, Any], max_json_mb: int = 80) -> List[str]:
    """
    Commit / Manage 删除后直接调用这个函数重写磁盘 pool。
    """
    folder = pool_label_dir(out_root, label)
    prefix = pool_file_prefix(label)
    return write_split_json_pool(
        pool_dict,
        folder,
        prefix,
        max_json_mb=max_json_mb,
        clear_old=True,
    )


def load_all_pools_from_disk(out_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return (
        load_pool_dict_from_disk(out_root, "good"),
        load_pool_dict_from_disk(out_root, "bad"),
    )


def next_available_hex_key(good_dict: Dict[str, Any], bad_dict: Dict[str, Any]) -> str:
    max_code = PUA_BASE - 1
    for d in [good_dict, bad_dict]:
        for k in d.keys():
            try:
                code = int(str(k), 16)
                max_code = max(max_code, code)
            except Exception:
                pass
    return f"{max(PUA_BASE, max_code + 1):04X}"


def increment_hex_key(hex_key: str, used: set) -> str:
    try:
        code = int(hex_key, 16)
    except Exception:
        code = PUA_BASE
    while f"{code:04X}" in used:
        code += 1
    return f"{code:04X}"


# =============================================================================
# 7. GUI 滚动容器
# =============================================================================

class ScrollableFrame(ttk.Frame):
    def __init__(self, master, height=650, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.canvas = tk.Canvas(self, height=height, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.inner = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")

        self._mousewheel_bound = False

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # 不再永久 bind_all。只在鼠标进入当前滚动区域时绑定，离开/销毁时解绑。
        # 这样关闭 Manage Good/Bad Pool 窗口后，不会留下指向已销毁窗口的回调。
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)
        self.bind("<Destroy>", self._on_destroy, add="+")

    def _on_inner_configure(self, event=None):
        try:
            if self.winfo_exists():
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        except tk.TclError:
            pass

    def _on_canvas_configure(self, event):
        try:
            if self.winfo_exists():
                self.canvas.itemconfig(self.window_id, width=event.width)
        except tk.TclError:
            pass

    def _bind_mousewheel(self, event=None):
        try:
            if self.winfo_exists() and not self._mousewheel_bound:
                self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
                self._mousewheel_bound = True
        except tk.TclError:
            pass

    def _unbind_mousewheel(self, event=None):
        try:
            if self._mousewheel_bound:
                self.canvas.unbind_all("<MouseWheel>")
                self._mousewheel_bound = False
        except tk.TclError:
            self._mousewheel_bound = False

    def _on_destroy(self, event=None):
        # 只处理当前 ScrollableFrame 自己的销毁事件，避免子控件 Destroy 反复触发。
        if event is not None and event.widget is not self:
            return
        self._unbind_mousewheel()

    def _on_mousewheel(self, event):
        try:
            if not self.winfo_exists():
                return
            if not self.canvas.winfo_exists():
                return
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except tk.TclError:
            self._mousewheel_bound = False


# =============================================================================
# 8. 主 GUI
# =============================================================================

class AnnotationFlywheelPCGNoJSONApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Annotation Flywheel PCG App - No JSON Source - File Backed + Stage2 Schema Strict")
        self.root.geometry("1520x910")

        self.rng = random.Random(int(time.time()))

        self.output_root = tk.StringVar(value=DEFAULT_OUTPUT_ROOT)
        self.n_var = tk.IntVar(value=5)
        self.preview_size_var = tk.IntVar(value=170)
        self.seed_var = tk.StringVar(value="")

        self.stroke_min_var = tk.StringVar(value="3")
        self.stroke_max_var = tk.StringVar(value="8")
        self.cc_min_var = tk.StringVar(value="1")
        self.cc_max_var = tk.StringVar(value="1")
        self.cycles_min_var = tk.StringVar(value="0")
        self.cycles_max_var = tk.StringVar(value="2")
        self.width_min_var = tk.StringVar(value="7")
        self.width_max_var = tk.StringVar(value="14")
        self.jitter_var = tk.StringVar(value="16")
        self.max_json_mb_var = tk.IntVar(value=80)

        self.status_var = tk.StringVar(value="Ready. No source candidate JSON. File-backed. Stage2 schema strict.")

        self.style_mode_vars: Dict[str, tk.BooleanVar] = {}
        self.family_vars: Dict[str, tk.BooleanVar] = {}

        self.current_batch: List[Dict[str, Any]] = []
        self.current_vars: List[tk.BooleanVar] = []
        self.current_photos: List[ImageTk.PhotoImage] = []

        self.good_pool: List[Dict[str, Any]] = []
        self.bad_pool: List[Dict[str, Any]] = []

        self._build_ui()
        self.refresh_disk_counts()

    def _build_ui(self):
        top = ttk.Frame(self.root)
        top.pack(side="top", fill="x", padx=8, pady=6)

        row1 = ttk.Frame(top)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="Output Root:").pack(side="left")
        ttk.Entry(row1, textvariable=self.output_root, width=95).pack(side="left", padx=4)
        ttk.Button(row1, text="Browse", command=self.browse_output).pack(side="left", padx=2)

        ttk.Label(row1, text="N:").pack(side="left", padx=(12, 2))
        ttk.Spinbox(row1, from_=1, to=20, textvariable=self.n_var, width=5).pack(side="left")
        ttk.Label(row1, text="Preview px:").pack(side="left", padx=(12, 2))
        ttk.Spinbox(row1, from_=100, to=260, textvariable=self.preview_size_var, width=6).pack(side="left")
        ttk.Label(row1, text="Seed:").pack(side="left", padx=(12, 2))
        ttk.Entry(row1, textvariable=self.seed_var, width=12).pack(side="left")

        row2 = ttk.LabelFrame(top, text="Procedural Complexity")
        row2.pack(fill="x", pady=4)
        self._add_entry(row2, "strokes min", self.stroke_min_var)
        self._add_entry(row2, "max", self.stroke_max_var)
        self._add_entry(row2, "CC min", self.cc_min_var)
        self._add_entry(row2, "max", self.cc_max_var)
        self._add_entry(row2, "cycles min", self.cycles_min_var)
        self._add_entry(row2, "max", self.cycles_max_var)
        self._add_entry(row2, "width min", self.width_min_var)
        self._add_entry(row2, "max", self.width_max_var)
        self._add_entry(row2, "jitter", self.jitter_var)
        ttk.Label(row2, text="JSON MB:").pack(side="left", padx=(8, 2))
        ttk.Spinbox(row2, from_=10, to=95, textvariable=self.max_json_mb_var, width=5).pack(side="left")

        row3 = ttk.LabelFrame(top, text="Style Modes")
        row3.pack(fill="x", pady=4)
        default_styles = {"straight", "mild_left", "mild_right", "s_curve_left", "s_curve_right", "hook_left", "hook_right", "mixed"}
        for mode in STYLE_MODES:
            v = tk.BooleanVar(value=mode in default_styles)
            self.style_mode_vars[mode] = v
            ttk.Checkbutton(row3, text=mode, variable=v).pack(side="left", padx=4)

        row4 = ttk.LabelFrame(top, text="Topology Families")
        row4.pack(fill="x", pady=4)
        for fam in TOPOLOGY_FAMILIES:
            v = tk.BooleanVar(value=True)
            self.family_vars[fam] = v
            ttk.Checkbutton(row4, text=fam, variable=v).pack(side="left", padx=4)

        row5 = ttk.Frame(top)
        row5.pack(fill="x", pady=4)
        ttk.Button(row5, text="Generate N×N Direct Random Batch", command=self.generate_batch).pack(side="left", padx=4)
        ttk.Button(row5, text="Select All Good", command=lambda: self.set_all_current(True)).pack(side="left", padx=4)
        ttk.Button(row5, text="Select All Bad", command=lambda: self.set_all_current(False)).pack(side="left", padx=4)
        ttk.Button(row5, text="Invert", command=self.invert_current).pack(side="left", padx=4)
        ttk.Button(row5, text="Commit Current Labels", command=self.commit_current_labels).pack(side="left", padx=10)
        ttk.Button(row5, text="Manage Good Pool", command=lambda: self.open_pool_manager("good")).pack(side="left", padx=4)
        ttk.Button(row5, text="Manage Bad Pool", command=lambda: self.open_pool_manager("bad")).pack(side="left", padx=4)
        ttk.Button(row5, text="Refresh Disk Counts", command=self.refresh_disk_counts).pack(side="left", padx=4)
        ttk.Button(row5, text="Write Manifest", command=self.save_pools).pack(side="left", padx=10)

        ttk.Label(top, textvariable=self.status_var, foreground="blue").pack(fill="x", pady=2)

        self.scroll = ScrollableFrame(self.root, height=700)
        self.scroll.pack(fill="both", expand=True, padx=8, pady=4)

    def _add_entry(self, parent, label, var):
        ttk.Label(parent, text=label).pack(side="left", padx=(8, 2))
        ttk.Entry(parent, textvariable=var, width=5).pack(side="left", padx=2)

    def browse_output(self):
        d = filedialog.askdirectory(title="Select output root", initialdir=SCRIPT_DIR)
        if d:
            self.output_root.set(d)

    def parse_int(self, var, default):
        s = str(var.get()).strip()
        try:
            return int(s)
        except Exception:
            return default

    def parse_float(self, var, default):
        s = str(var.get()).strip()
        try:
            return float(s)
        except Exception:
            return default

    def get_generator_config(self) -> GeneratorConfig:
        styles = [m for m, v in self.style_mode_vars.items() if v.get()]
        if not styles:
            styles = ["straight"]

        families = [f for f, v in self.family_vars.items() if v.get()]
        if not families:
            families = ["random_tree"]

        stroke_min = self.parse_int(self.stroke_min_var, 3)
        stroke_max = self.parse_int(self.stroke_max_var, 8)
        if stroke_max < stroke_min:
            stroke_max = stroke_min

        cc_min = self.parse_int(self.cc_min_var, 1)
        cc_max = self.parse_int(self.cc_max_var, 1)
        if cc_max < cc_min:
            cc_max = cc_min

        cycle_min = self.parse_int(self.cycles_min_var, 0)
        cycle_max = self.parse_int(self.cycles_max_var, 2)
        if cycle_max < cycle_min:
            cycle_max = cycle_min

        wmin = self.parse_float(self.width_min_var, 7)
        wmax = self.parse_float(self.width_max_var, 14)
        if wmax < wmin:
            wmax = wmin

        return GeneratorConfig(
            stroke_min=stroke_min,
            stroke_max=stroke_max,
            cc_min=cc_min,
            cc_max=cc_max,
            cycle_min=cycle_min,
            cycle_max=cycle_max,
            width_min=wmin,
            width_max=wmax,
            jitter=self.parse_float(self.jitter_var, 16),
            style_modes=styles,
            topology_families=families,
            aesthetic_condition={
                "target_family": "alien_rune",
                "human_labeling": True,
                "black_width_preview": True,
            },
            topology_condition={
                "families": families,
                "stroke_min": stroke_min,
                "stroke_max": stroke_max,
                "cc_min": cc_min,
                "cc_max": cc_max,
                "cycle_min": cycle_min,
                "cycle_max": cycle_max,
            },
            style_condition={"styles": styles},
        )

    def refresh_disk_counts(self):
        out_root = self.output_root.get().strip() or DEFAULT_OUTPUT_ROOT
        ensure_dir(pool_label_dir(out_root, "good"))
        ensure_dir(pool_label_dir(out_root, "bad"))
        good, bad = load_all_pools_from_disk(out_root)
        self.status_var.set(
            f"Disk pool: good={len(good)}, bad={len(bad)} | out_root={out_root}"
        )

    def generate_batch(self):
        seed_text = self.seed_var.get().strip()
        if seed_text:
            try:
                self.rng = random.Random(int(seed_text))
            except Exception:
                self.rng = random.Random(seed_text)
        else:
            self.rng = random.Random(int(time.time() * 1000) % (2**32 - 1))

        cfg = self.get_generator_config()
        n = int(self.n_var.get())
        max_items = n * n

        selected = []
        reject = Counter()
        attempts = 0
        max_attempts = max(200, max_items * 80)

        while len(selected) < max_items and attempts < max_attempts:
            attempts += 1
            cand = generate_procedural_candidate(cfg, self.rng, attempts)

            if not MODEL_HOOKS.prefilter_topology(cand, cfg.topology_condition):
                reject["model_prefilter_reject"] += 1
                continue

            ok, comp, reason = passes_complexity(cand, cfg)
            if not ok:
                reject[reason] += 1
                continue

            score = MODEL_HOOKS.score_aesthetic(cand, cfg.aesthetic_condition)
            model_accept = MODEL_HOOKS.accept_by_model(cand, score, cfg.aesthetic_condition)
            if model_accept is False:
                reject["model_rejected"] += 1
                continue

            cand.setdefault("pcg_meta", {})
            cand["pcg_meta"].update({
                "complexity": comp,
                "aesthetic_score": score,
                "model_accept": model_accept,
                "attempt": attempts,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            selected.append(cand)

        self.current_batch = selected
        self.current_vars = [tk.BooleanVar(value=False) for _ in selected]
        self.status_var.set(
            f"Generated {len(selected)}/{max_items} direct procedural candidates. attempts={attempts}, reject={dict(reject)}"
        )
        self.render_current_grid()

    def clear_grid(self):
        for w in self.scroll.inner.winfo_children():
            w.destroy()
        self.current_photos = []

    def render_current_grid(self):
        self.clear_grid()
        size = int(self.preview_size_var.get())
        n = int(self.n_var.get())
        cols = max(1, n)

        for idx, cand in enumerate(self.current_batch):
            r, c = divmod(idx, cols)
            frame = ttk.Frame(self.scroll.inner, relief="groove", borderwidth=1)
            frame.grid(row=r, column=c, padx=6, pady=6, sticky="n")

            img = render_candidate_dual_preview(cand, size=size)
            photo = ImageTk.PhotoImage(img)
            self.current_photos.append(photo)

            ttk.Label(frame, image=photo).pack(side="top", padx=2, pady=2)

            comp = cand.get("pcg_meta", {}).get("complexity", candidate_complexity(cand))
            txt = (
                f"{idx:03d} {cand.get('style_mode','')}\n"
                f"{cand.get('topology_family','')[:22]}\n"
                f"S={comp.get('stroke_count')} E={comp.get('edge_count')} "
                f"CC={comp.get('connected_components')} Cy={comp.get('cycle_count')}"
            )
            ttk.Label(frame, text=txt, font=("Consolas", 8), justify="center").pack(side="top")
            ttk.Checkbutton(frame, text="Good ✓", variable=self.current_vars[idx]).pack(side="top", pady=(2, 4))

        for c in range(cols):
            self.scroll.inner.grid_columnconfigure(c, weight=1)

    def set_all_current(self, value: bool):
        for v in self.current_vars:
            v.set(bool(value))

    def invert_current(self):
        for v in self.current_vars:
            v.set(not v.get())

    def commit_current_labels(self):
        """
        文件直写版：
        每次点击 Commit，立即读取磁盘 Good/Bad pool，追加当前 batch，并重写 80MB split 文件。
        Manage 不依赖内存，也会直接读这些文件。
        """
        if not self.current_batch:
            messagebox.showwarning("No batch", "当前没有 batch。")
            return

        out_root = self.output_root.get().strip() or DEFAULT_OUTPUT_ROOT
        ensure_dir(pool_label_dir(out_root, "good"))
        ensure_dir(pool_label_dir(out_root, "bad"))

        max_mb = int(self.max_json_mb_var.get())

        good_dict, bad_dict = load_all_pools_from_disk(out_root)
        used = set(good_dict.keys()) | set(bad_dict.keys())

        good_added, bad_added = 0, 0

        for cand, var in zip(self.current_batch, self.current_vars):
            label = "good" if var.get() else "bad"

            hex_key = increment_hex_key(next_available_hex_key(good_dict, bad_dict), used)
            used.add(hex_key)

            c = copy.deepcopy(cand)
            c["human_label"] = label
            c["font_path"] = c.get("font_path", DEFAULT_FONT_NAME)
            c["font_name"] = c.get("font_name", DEFAULT_FONT_NAME)
            c.setdefault("pcg_meta", {})
            c["pcg_meta"]["human_label"] = label
            c["pcg_meta"]["committed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            c["pcg_meta"]["storage_mode"] = "file_backed_commit"

            out_hex, bundle = candidate_to_char_bundle(c, label, hex_key)

            # 为避免 key 冲突，磁盘 pool 的外层 key 使用本次分配的 hex_key。
            # 当 PRESERVE_STAGE2_BUNDLE=True 时，不再额外改写 bundle 内部字段，
            # 以最大程度保持第二阶段人工标注输出结构。
            out_hex = hex_key

            if (not PRESERVE_STAGE2_BUNDLE) and isinstance(bundle, dict):
                bundle.setdefault("glyph_info", {})
                if isinstance(bundle["glyph_info"], dict):
                    bundle["glyph_info"]["char"] = chr(int(hex_key, 16))
                    bundle["glyph_info"]["unicode_hex"] = hex_key
                    bundle["glyph_info"]["label"] = label
                    bundle["glyph_info"]["font_path"] = bundle["glyph_info"].get("font_path", DEFAULT_FONT_NAME)
                    bundle["glyph_info"]["font_name"] = bundle["glyph_info"].get("font_name", DEFAULT_FONT_NAME)
                    bundle["glyph_info"]["candidate_id"] = _get_candidate_id(c)
                    bundle["glyph_info"]["style_mode"] = c.get("style_mode")
                    bundle["glyph_info"]["topology_family"] = c.get("topology_family")
                    bundle["glyph_info"]["pcg_meta"] = c.get("pcg_meta", {})

                bundle.setdefault("edit_history", [])
                if isinstance(bundle["edit_history"], list):
                    bundle["edit_history"].append({
                        "action": "PCG_DIRECT_FILE_BACKED_COMMIT",
                        "label": label,
                        "candidate_id": _get_candidate_id(c),
                        "style_mode": c.get("style_mode"),
                        "topology_family": c.get("topology_family"),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "source": "annotation_flywheel_pcg_app_no_json_v5_stage2_schema.py",
                    })

            if label == "good":
                good_dict[out_hex] = bundle
                good_added += 1
            else:
                bad_dict[out_hex] = bundle
                bad_added += 1

        good_files = write_pool_dict_to_disk(out_root, "good", good_dict, max_json_mb=max_mb)
        bad_files = write_pool_dict_to_disk(out_root, "bad", bad_dict, max_json_mb=max_mb)

        self.status_var.set(
            f"Committed to disk: +good={good_added}, +bad={bad_added}. "
            f"Disk pool good={len(good_dict)}, bad={len(bad_dict)} | files good={len(good_files)}, bad={len(bad_files)}"
        )
        messagebox.showinfo(
            "Committed To Disk",
            f"已直接写入存储文件。\\n\\n"
            f"+Good={good_added}\\n+Bad={bad_added}\\n\\n"
            f"Disk Good={len(good_dict)}\\nDisk Bad={len(bad_dict)}"
        )

    def open_pool_manager(self, pool_name: str):
        """
        文件直读版 Manage：
        每次打开都从磁盘 pool 文件读取；删除也直接重写磁盘 pool 文件。
        """
        out_root = self.output_root.get().strip() or DEFAULT_OUTPUT_ROOT
        pool_dict = load_pool_dict_from_disk(out_root, pool_name)

        win = tk.Toplevel(self.root)
        win.title(f"Manage {pool_name.upper()} Disk Pool - {len(pool_dict)} items")
        win.geometry("1450x820")

        top = ttk.Frame(win)
        top.pack(side="top", fill="x", padx=8, pady=6)

        ttk.Label(top, text=f"{pool_name.upper()} disk pool count: {len(pool_dict)}").pack(side="left")
        remove_vars: List[tk.BooleanVar] = []

        items = list(pool_dict.items())

        def remove_selected():
            remove_keys = []
            for (hex_key, bundle), var in zip(items, remove_vars):
                if var.get():
                    remove_keys.append(hex_key)

            if not remove_keys:
                messagebox.showinfo("No Remove", "没有勾选要剔除的样本。")
                return

            latest = load_pool_dict_from_disk(out_root, pool_name)
            for k in remove_keys:
                latest.pop(k, None)

            max_mb = int(self.max_json_mb_var.get())
            files = write_pool_dict_to_disk(out_root, pool_name, latest, max_json_mb=max_mb)

            win.destroy()
            self.status_var.set(
                f"Removed {len(remove_keys)} from disk {pool_name}. remain={len(latest)}, files={len(files)}"
            )
            self.open_pool_manager(pool_name)

        ttk.Button(top, text="Remove Selected From Disk Pool", command=remove_selected).pack(side="left", padx=12)
        ttk.Button(top, text="Refresh", command=lambda: (win.destroy(), self.open_pool_manager(pool_name))).pack(side="left", padx=4)
        ttk.Button(top, text="Close", command=win.destroy).pack(side="right")

        sf = ScrollableFrame(win, height=740)
        sf.pack(fill="both", expand=True, padx=8, pady=4)

        photos = []
        cols = 4
        size = 150

        for idx, (hex_key, bundle) in enumerate(items):
            var = tk.BooleanVar(value=False)
            remove_vars.append(var)

            cand = bundle_to_candidate(bundle, hex_key=hex_key)
            r, c = divmod(idx, cols)
            frame = ttk.Frame(sf.inner, relief="groove", borderwidth=1)
            frame.grid(row=r, column=c, padx=6, pady=6, sticky="n")

            img = render_candidate_dual_preview(cand, size=size)
            photo = ImageTk.PhotoImage(img)
            photos.append(photo)

            ttk.Label(frame, image=photo).pack(side="top", padx=2, pady=2)
            comp = candidate_complexity(cand)
            txt = (
                f"{idx:03d} U+{hex_key} {cand.get('style_mode','')}\\n"
                f"{cand.get('topology_family','')[:30]}\\n"
                f"S={comp.get('stroke_count')} E={comp.get('edge_count')} "
                f"CC={comp.get('connected_components')} Cy={comp.get('cycle_count')}"
            )
            ttk.Label(frame, text=txt, font=("Consolas", 8), justify="center").pack(side="top")
            ttk.Checkbutton(frame, text="Remove ✕", variable=var).pack(side="top", pady=4)

        win._photos = photos

    def save_pools(self):
        """
        文件直写版中，Commit 已经保存 Good/Bad pool。
        这个按钮只刷新 active split 文件并写 manifest。
        """
        out_root = self.output_root.get().strip() or DEFAULT_OUTPUT_ROOT
        ensure_dir(out_root)

        max_mb = int(self.max_json_mb_var.get())
        good_dict, bad_dict = load_all_pools_from_disk(out_root)

        good_files = write_pool_dict_to_disk(out_root, "good", good_dict, max_json_mb=max_mb)
        bad_files = write_pool_dict_to_disk(out_root, "bad", bad_dict, max_json_mb=max_mb)

        batch_name = POOL_PREFIX + "_Manifest_" + time.strftime("%Y%m%d_%H%M%S")
        cfg = self.get_generator_config()

        manifest = {
            "schema_version": "annotation_flywheel_pcg_no_json_stage2_schema_manifest_v1",
            "batch_name": batch_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "direct_procedural_generation_no_existing_json",
            "storage_mode": "file_backed_pool_commit_stage2_schema_strict",
            "output_root": os.path.abspath(out_root),
            "good_dir": os.path.abspath(pool_label_dir(out_root, "good")),
            "bad_dir": os.path.abspath(pool_label_dir(out_root, "bad")),
            "good_count": len(good_dict),
            "bad_count": len(bad_dict),
            "max_json_mb": max_mb,
            "good_files": good_files,
            "bad_files": bad_files,
            "pool_prefix": POOL_PREFIX,
            "require_stage2_converter": REQUIRE_STAGE2_CONVERTER,
            "preserve_stage2_bundle": PRESERVE_STAGE2_BUNDLE,
            "generator_config": {
                "stroke_min": cfg.stroke_min,
                "stroke_max": cfg.stroke_max,
                "cc_min": cfg.cc_min,
                "cc_max": cfg.cc_max,
                "cycle_min": cfg.cycle_min,
                "cycle_max": cfg.cycle_max,
                "width_min": cfg.width_min,
                "width_max": cfg.width_max,
                "jitter": cfg.jitter,
                "style_modes": cfg.style_modes,
                "topology_families": cfg.topology_families,
            },
            "notes": [
                "No existing candidate JSON is read for generation.",
                "Commit Current Labels directly modifies disk-backed Good/Bad pool files using the stage-2 converter.",
                "Manage Good/Bad Pool directly reads and modifies disk files.",
                "Preview uses dual rendering: black-width preview and colored-width preview.",
                "Each JSON is split before exceeding max_json_mb.",
                "Future aesthetic/topology/style models can be connected through AestheticAndGenerationHooks.",
            ],
        }

        manifest_path = os.path.join(out_root, f"{batch_name}.json")
        save_json(manifest, manifest_path)

        self.status_var.set(
            f"Manifest written. Disk good={len(good_dict)}, bad={len(bad_dict)}, "
            f"good_files={len(good_files)}, bad_files={len(bad_files)}"
        )
        messagebox.showinfo(
            "Manifest Written",
            f"Commit 已经直接保存样本。\\n\\n"
            f"当前磁盘池：\\nGood={len(good_dict)}\\nBad={len(bad_dict)}\\n\\n"
            f"Manifest:\\n{manifest_path}",
        )


def main():
    ensure_dir(DEFAULT_OUTPUT_ROOT)
    root = tk.Tk()
    app = AnnotationFlywheelPCGNoJSONApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
