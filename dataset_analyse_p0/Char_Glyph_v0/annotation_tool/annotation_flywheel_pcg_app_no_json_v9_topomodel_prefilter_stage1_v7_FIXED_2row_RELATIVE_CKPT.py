# -*- coding: utf-8 -*-
r"""
annotation_flywheel_pcg_app_no_json_v8_fast_x_stable_local_pool_incremental_commit_compact_fast.py

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
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

import numpy as np

try:
    import networkx as nx
except Exception:
    nx = None

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
#       annotation_flywheel_pcg_app_no_json_v8_fast_x_stable_local_pool_incremental_commit_compact_fast.py
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

# 默认输出到当前脚本所在目录下，方便直接找到 Good / Bad 文件池：
#   Babel_Simulater/dataset_analyse_p0/Char_Glyph_v0/annotation_tool/pcg_filebacked_stage2_schema/
#
# 目录结构：
#   annotation_tool/
#     pcg_filebacked_stage2_schema/
#       good/
#       bad/
#       PCG_Direct_Stage2TopoEngineXStableFastLocal_Manifest_*.json
DEFAULT_OUTPUT_ROOT = os.path.join(TOOL_DIR, "pcg_filebacked_stage2_schema")

DEFAULT_MAX_JSON_MB = 80
DEFAULT_FONT_NAME = "PCG_Procedural_TopoStyle_AestheticFlywheel.ttf"

# Stage1 topology gate runtime defaults.
# 默认读取刚训练出的 V7 stage1 recall 模型。
# 这个模型只作用于三阶段生成的第一阶段：拓扑骨架粗筛。
DEFAULT_TOPO_MODEL_INDEX = os.path.join(
    CHAR_GLYPH_DIR,
    "flywheel_model",
    "model",
    "flywheel_topohgt_v7_stage1_recall_ckptselect_checkpoint_index.json",
)
DEFAULT_TOPO_MODEL_PURPOSE = "quality_high_recall"
TOPO_MODEL_PURPOSES = [
    "quality_high_recall",
    "quality_balanced",
    "quality_conservative",
    "relation_overall",
    "relation_t",
    "relation_x",
    "relation_e2e",
    "loss",
    "latest",
]

TOPO_MODEL_SCORE_MODES = [
    "quality_recall_target",
    "quality_best_f1",
    "quality_precision_target",
    "manual",
    "off",
]
DEFAULT_TOPO_MODEL_SCORE_MODE = "quality_recall_target"
DEFAULT_TOPO_MODEL_MANUAL_SCORE_THRESHOLD = 0.40

# File-backed pool prefix：Commit/Manage 都直接读写这些 active pool 文件。
POOL_PREFIX = "PCG_Direct_Stage2TopoEngineXStableFastLocal"

# 强约束：为了尽可能等同第二阶段人工标注格式，不再静默 fallback。
# 如果无法导入 annotation_flywheel_app_fixed.py 或无法调用 candidate_to_char_bundle()，
# 程序会直接报错，而不是写出“不完全一致”的 JSON。
REQUIRE_STAGE2_CONVERTER = True

# True 表示 commit 后不额外改写 bundle 结构，只使用第二阶段函数返回的 bundle。
# label/style/topology_family 等信息主要保存在文件夹、manifest 和 candidate 输入字段中。
PRESERVE_STAGE2_BUNDLE = True

# v7: 直接复用你第二阶段 topo_editor_workspace.py 的拓扑事件/闭环输出结构。
# 这样 Good/Bad 的 bundle 结构与 action_complete_topo() 保存的结构一致：
# glyph_info / strokes / topology_events / cycles / edit_history
USE_STAGE2_DIRECT_BUNDLE = True
STAGE2_TOPO_SAMPLE_N = 80
STAGE2_COLLISION_THRESHOLD = 2.0
STAGE2_CYCLE_POINT_THRESHOLD = 5.0

# Private Use Area 起点，用于模拟 unicode
#
# 旧版本用 0xE000，即 BMP Private Use Area，只有 6400 个码位；
# PCG 样本多后容易冲突，也容易和人工标注/旧实验混在一起。
#
# 新版本默认从 Supplementary Private Use Area-A: U+F0000 开始。
# 可用到 U+10FFFD，容量约 13 万；同时 Compact Pool 会统一重编号。
PUA_BASE = 0xF0000
PUA_MAX = 0x10FFFD

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



# =============================================================================
# 2.5 第二阶段拓扑引擎：从 topo_editor_workspace.py 抽取的纯函数版
# =============================================================================

def cubic_bezier_np_stage2(pts: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """
    兼容 geometry_vision.cubic_bezier_np 的纯 numpy 版本。
    pts: (4,2), ts: (N,1) or (N,)
    """
    pts = np.asarray(pts, dtype=np.float32)
    ts = np.asarray(ts, dtype=np.float32)
    if ts.ndim == 1:
        ts = ts[:, None]
    mt = 1.0 - ts
    return (
        (mt ** 3) * pts[0]
        + 3 * (mt ** 2) * ts * pts[1]
        + 3 * mt * (ts ** 2) * pts[2]
        + (ts ** 3) * pts[3]
    )


def get_bezier_derivative_stage2(pts: np.ndarray, t: float) -> np.ndarray:
    """
    对齐第二阶段 topo_editor_workspace.py 中 get_bezier_derivative() 的计算。
    """
    pts = np.asarray(pts, dtype=np.float32)
    mt = 1 - float(t)
    d = (
        3 * mt ** 2 * (pts[1] - pts[0])
        + 6 * mt * float(t) * (pts[2] - pts[1])
        + 3 * float(t) ** 2 * (pts[3] - pts[2])
    )
    return d


def get_angle_stage2(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    对齐第二阶段 get_angle()：返回两向量夹角，单位 degree。
    """
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-5 or n2 < 1e-5:
        return 0.0
    cos_th = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_th)))


def get_polygon_orientation_stage2(pts: List[np.ndarray]) -> str:
    """
    对齐第二阶段 get_polygon_orientation()：基于符号面积判断 cw / ccw。
    """
    area = 0.0
    n = len(pts)
    if n <= 2:
        return "unknown"
    for i in range(n):
        j = (i + 1) % n
        area += (pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1])
    return "ccw" if area > 0 else "cw"


# -----------------------------------------------------------------------------
# v8: 稳定 X 交叉检测
# -----------------------------------------------------------------------------
# v7 的 X 检测是“采样点最近距离 < 阈值”，交点落在采样间隙时会漏判。
# v8_fast 改为：Bézier -> polyline 后，使用 NumPy 向量化线段相交检测。
STAGE2_X_ENDPOINT_MARGIN = 2
STAGE2_SEG_EPS = 1e-8


def _cross2d_stage2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def segment_intersection_stage2(p0, p1, q0, q1, eps: float = STAGE2_SEG_EPS):
    """
    线段 p0-p1 与 q0-q1 相交检测。
    返回:
      None
      或 (point, local_t_on_p, local_t_on_q)
    """
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    q0 = np.asarray(q0, dtype=np.float32)
    q1 = np.asarray(q1, dtype=np.float32)

    r = p1 - p0
    s = q1 - q0
    denom = _cross2d_stage2(r, s)

    # 平行/近似平行不当作 X；端点接触由 E2E/T 处理。
    if abs(denom) < eps:
        return None

    qp = q0 - p0
    t = _cross2d_stage2(qp, s) / denom
    u = _cross2d_stage2(qp, r) / denom

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        pt = p0 + t * r
        return pt, float(t), float(u)

    return None


def find_polyline_x_intersection_stage2(
    c1: np.ndarray,
    c2: np.ndarray,
    endpoint_margin: int = STAGE2_X_ENDPOINT_MARGIN,
) -> Tuple[bool, Optional[np.ndarray], Optional[float], Optional[float]]:
    """
    v8_fast: 向量化 polyline 线段相交检测。

    v8 慢的原因是 Python 双重 for 循环：
        (sample_n-1) * (sample_n-1)
    在 N=120 时，每对 stroke 约 14161 次 Python 循环；一屏几十个候选会非常卡。

    这里改成 NumPy 广播一次性计算所有线段对的交点参数 t/u，
    仍然是线段相交算法，但主要计算在 NumPy/C 层完成。
    """
    c1 = np.asarray(c1, dtype=np.float32)
    c2 = np.asarray(c2, dtype=np.float32)
    n1 = len(c1)
    n2 = len(c2)
    if n1 < 2 or n2 < 2:
        return False, None, None, None

    m = int(endpoint_margin)
    i0 = max(0, m)
    i1 = max(i0, n1 - 1 - m)
    j0 = max(0, m)
    j1 = max(j0, n2 - 1 - m)

    if i1 <= i0 or j1 <= j0:
        return False, None, None, None

    p0 = c1[i0:i1]          # (A, 2)
    p1 = c1[i0 + 1:i1 + 1]
    q0 = c2[j0:j1]          # (B, 2)
    q1 = c2[j0 + 1:j1 + 1]

    r = p1 - p0             # (A, 2)
    s = q1 - q0             # (B, 2)

    # denom[a,b] = cross(r[a], s[b])
    denom = r[:, None, 0] * s[None, :, 1] - r[:, None, 1] * s[None, :, 0]
    valid = np.abs(denom) >= STAGE2_SEG_EPS

    if not np.any(valid):
        return False, None, None, None

    qp = q0[None, :, :] - p0[:, None, :]  # (A, B, 2)

    # local_t on segment p and local_u on segment q
    t = (qp[:, :, 0] * s[None, :, 1] - qp[:, :, 1] * s[None, :, 0]) / np.where(valid, denom, 1.0)
    u = (qp[:, :, 0] * r[:, None, 1] - qp[:, :, 1] * r[:, None, 0]) / np.where(valid, denom, 1.0)

    hit_mask = valid & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)

    if not np.any(hit_mask):
        return False, None, None, None

    # 转换为 Bézier 参数，过滤端点附近，避免 E2E/T 被误记成 X
    hit_i, hit_j = np.where(hit_mask)
    lt = t[hit_i, hit_j]
    lu = u[hit_i, hit_j]

    bez_t1 = (i0 + hit_i + lt) / float(n1 - 1)
    bez_t2 = (j0 + hit_j + lu) / float(n2 - 1)

    interior = (bez_t1 > 0.02) & (bez_t1 < 0.98) & (bez_t2 > 0.02) & (bez_t2 < 0.98)

    if not np.any(interior):
        return False, None, None, None

    # 取第一个内部交点。通常两条简单 stroke 只有一个 X。
    k = int(np.where(interior)[0][0])
    ii = int(hit_i[k])
    jj = int(hit_j[k])
    lt_k = float(lt[k])

    pt = p0[ii] + lt_k * r[ii]
    return True, np.asarray(pt, dtype=np.float32), float(bez_t1[k]), float(bez_t2[k])


def candidate_to_stage2_edges(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    把 PCG candidate 转成第二阶段 workspace 使用的 edge 格式：
      edge = {"id": int, "path": [[x0,y0],[x1,y1],[x2,y2],[x3,y3]], ...}
    """
    edges = []
    for i, nd in enumerate(_get_nodes(candidate)):
        P = _get_bezier_from_node(nd)
        if P is None:
            continue
        try:
            eid = int(nd.get("id", nd.get("node_id", nd.get("bezier_id", i))))
        except Exception:
            eid = i
        edges.append({
            "id": eid,
            "path": np.asarray(P, dtype=np.float32).tolist(),
            "width": float(_get_width_from_node(nd)),
            "source_node": nd,
        })
    return edges


def stage2_display_map(edges: List[Dict[str, Any]]) -> Dict[int, str]:
    """
    对齐第二阶段 get_display_map()：真实 edge id -> 人眼显示 id，从 1 开始。
    """
    return {edge["id"]: str(i + 1) for i, edge in enumerate(edges)}


def _simple_cycle_basis_fallback(nodes: List[int], undirected_edges: List[Tuple[int, int]]) -> List[List[int]]:
    """
    没有 networkx 时的弱 fallback。正式环境建议安装 networkx。
    """
    # 只用于避免崩溃，不保证列出全部 basis。
    parent = {}
    graph = defaultdict(list)
    cycles = []
    seen_cycles = set()

    for u, v in undirected_edges:
        graph[u].append(v)
        graph[v].append(u)

    def dfs(start, cur, path, visited):
        if len(path) > len(nodes):
            return
        for nb in graph[cur]:
            if nb == start and len(path) >= 3:
                cyc = tuple(sorted(path))
                if cyc not in seen_cycles:
                    seen_cycles.add(cyc)
                    cycles.append(path[:])
            elif nb not in visited and nb > -1:
                visited.add(nb)
                dfs(start, nb, path + [nb], visited)
                visited.remove(nb)

    for n in nodes:
        dfs(n, n, [n], {n})
    return cycles[:16]


def compute_stage2_topology_from_edges(
    edges: List[Dict[str, Any]],
    sample_n: int = STAGE2_TOPO_SAMPLE_N,
    collision_threshold: float = STAGE2_COLLISION_THRESHOLD,
    cycle_point_threshold: float = STAGE2_CYCLE_POINT_THRESHOLD,
) -> Dict[str, Any]:
    """
    纯函数版第二阶段拓扑反馈/落盘引擎。

    逻辑来源：
      - update_topology_text(): E2E / T / X 检测 + 2-stroke cycle + >=3-stroke cycle
      - action_complete_topo(): topology_events / cycles 的落盘 schema

    输出：
      topology_events: 与第二阶段 action_complete_topo() 同构
      cycles: 与第二阶段 action_complete_topo() 同构
      topology_text: 用于 GUI 预览
      stats: stroke_count / topology_event_count / connected_components / cycle_count
    """
    id_map = stage2_display_map(edges)
    ts = np.linspace(0, 1, sample_n)[:, None]

    topology_events: List[Dict[str, Any]] = []
    end_to_end, t_junctions, x_junctions = [], [], []
    connection_points: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
    graph_edges: List[Tuple[int, int]] = []

    display_ids = []
    for e in edges:
        try:
            display_ids.append(int(id_map[e["id"]]))
        except Exception:
            pass

    def add_graph_connection(a_display: int, b_display: int, pos):
        u, v = min(int(a_display), int(b_display)), max(int(a_display), int(b_display))
        graph_edges.append((u, v))
        connection_points[(u, v)].append(np.asarray(pos, dtype=np.float32))

    # pairwise topology events
    for i, e1 in enumerate(edges):
        for j, e2 in enumerate(edges):
            if i >= j:
                continue

            p1 = np.asarray(e1["path"], dtype=np.float32)
            p2 = np.asarray(e2["path"], dtype=np.float32)
            c1 = cubic_bezier_np_stage2(p1, ts)
            c2 = cubic_bezier_np_stage2(p2, ts)

            id1 = int(id_map[e1["id"]])
            id2 = int(id_map[e2["id"]])

            pair_has_connection = False
            is_e2e = False

            # --- E2E ---
            for pt1_idx, t1 in [(0, 0.0), (3, 1.0)]:
                for pt2_idx, t2 in [(0, 0.0), (3, 1.0)]:
                    if np.linalg.norm(p1[pt1_idx] - p2[pt2_idx]) < collision_threshold:
                        is_e2e = True
                        pair_has_connection = True
                        pos = p1[pt1_idx]
                        topology_events.append({
                            "type": "E2E",
                            "stroke_a": id1, "t_a": t1,
                            "stroke_b": id2, "t_b": t2,
                            "position": [round(float(pos[0]), 1), round(float(pos[1]), 1)]
                        })
                        add_graph_connection(id1, id2, pos)

            # --- T attach ---
            local_t_events = []
            if not is_e2e:
                for pt1_idx, t1 in [(0, 0.0), (3, 1.0)]:
                    dists = np.linalg.norm(c2 - p1[pt1_idx], axis=1)
                    m_idx = int(np.argmin(dists))
                    if dists[m_idx] < collision_threshold:
                        t2 = m_idx / float(sample_n - 1)
                        ang = get_angle_stage2(get_bezier_derivative_stage2(p1, t1), get_bezier_derivative_stage2(p2, t2))
                        pos = c2[m_idx]
                        ev = {
                            "type": "T",
                            "guest": id1, "guest_t": t1,
                            "host": id2, "host_t": round(float(t2), 3),
                            "angle": round(float(ang), 1),
                            "position": [round(float(pos[0]), 1), round(float(pos[1]), 1)]
                        }
                        topology_events.append(ev)
                        local_t_events.append(ev)
                        pair_has_connection = True
                        add_graph_connection(id1, id2, pos)

                for pt2_idx, t2 in [(0, 0.0), (3, 1.0)]:
                    dists = np.linalg.norm(c1 - p2[pt2_idx], axis=1)
                    m_idx = int(np.argmin(dists))
                    if dists[m_idx] < collision_threshold:
                        t1 = m_idx / float(sample_n - 1)
                        ang = get_angle_stage2(get_bezier_derivative_stage2(p1, t1), get_bezier_derivative_stage2(p2, t2))
                        pos = c1[m_idx]
                        ev = {
                            "type": "T",
                            "guest": id2, "guest_t": t2,
                            "host": id1, "host_t": round(float(t1), 3),
                            "angle": round(float(ang), 1),
                            "position": [round(float(pos[0]), 1), round(float(pos[1]), 1)]
                        }
                        topology_events.append(ev)
                        local_t_events.append(ev)
                        pair_has_connection = True
                        add_graph_connection(id1, id2, pos)

            # --- X cross ---
            # v8: 非 E2E 且无 T 时，用 polyline 线段相交检测 X。
            # 这样不会因为真实交点落在采样点间隙里而漏判。
            if not is_e2e and not local_t_events:
                hit_x, x_pt, t1, t2 = find_polyline_x_intersection_stage2(c1, c2)
                if hit_x:
                    ang = get_angle_stage2(
                        get_bezier_derivative_stage2(p1, t1),
                        get_bezier_derivative_stage2(p2, t2),
                    )
                    topology_events.append({
                        "type": "X",
                        "stroke_a": id1, "t_a": round(float(t1), 3),
                        "stroke_b": id2, "t_b": round(float(t2), 3),
                        "angle": round(float(ang), 1),
                        "position": [round(float(x_pt[0]), 1), round(float(x_pt[1]), 1)]
                    })
                    pair_has_connection = True
                    add_graph_connection(id1, id2, x_pt)

    # 文本摘要
    for ev in topology_events:
        if ev["type"] == "E2E":
            end_to_end.append(f"{ev['stroke_a']}-{ev['stroke_b']}")
        elif ev["type"] == "T":
            t_junctions.append(f"{ev['guest']}搭{ev['host']}")
        elif ev["type"] == "X":
            x_junctions.append(f"{ev['stroke_a']}交叉{ev['stroke_b']}")

    # graph connected components
    unique_graph_edges = sorted(set(graph_edges))
    active_nodes = sorted(set(display_ids))

    if nx is not None:
        G = nx.Graph()
        G.add_nodes_from(active_nodes)
        G.add_edges_from(unique_graph_edges)
        cc_count = nx.number_connected_components(G) if len(active_nodes) > 0 else 0
        basis = nx.cycle_basis(G)
    else:
        # union-find cc fallback
        parent = {n: n for n in active_nodes}
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
        for u, v in unique_graph_edges:
            union(u, v)
        cc_count = len(set(find(n) for n in active_nodes)) if active_nodes else 0
        basis = _simple_cycle_basis_fallback(active_nodes, unique_graph_edges)

    # cycles: 对齐第二阶段 action_complete_topo
    valid_cycle_lists = []

    # 1. 2-stroke cycles: 两根线有两个不同物理碰撞点
    for (u, v), pts_list in connection_points.items():
        if len(pts_list) >= 2:
            for idx1 in range(len(pts_list)):
                for idx2 in range(idx1 + 1, len(pts_list)):
                    if np.linalg.norm(pts_list[idx1] - pts_list[idx2]) >= cycle_point_threshold:
                        valid_cycle_lists.append([u, v])
                        break
                else:
                    continue
                break

    # 2. >=3-stroke cycles
    for cycle_nodes in basis:
        members = [int(n) for n in cycle_nodes]
        k = len(members)
        if k < 3:
            continue

        is_valid = True
        for idx in range(k):
            u, v, w = members[idx - 1], members[idx], members[(idx + 1) % k]
            key_in = (min(u, v), max(u, v))
            key_out = (min(v, w), max(v, w))
            if not connection_points.get(key_in) or not connection_points.get(key_out):
                is_valid = False
                break
            p_in = connection_points[key_in][0]
            p_out = connection_points[key_out][0]
            if np.linalg.norm(p_in - p_out) < cycle_point_threshold:
                is_valid = False
                break

        if is_valid:
            valid_cycle_lists.append(members)

    # 去重 cycle
    dedup_cycles = []
    seen = set()
    for cyc in valid_cycle_lists:
        key = tuple(sorted([int(x) for x in cyc]))
        if key not in seen:
            seen.add(key)
            dedup_cycles.append([int(x) for x in cyc])

    cycles_tokens = []
    # id -> edge path by displayed id
    display_to_edge = {int(stage2_display_map(edges)[e["id"]]): e for e in edges}

    for members in dedup_cycles:
        pts_for_orient = []
        for n in members:
            edge = display_to_edge.get(int(n))
            if edge is not None:
                pts_for_orient.append(np.mean(np.asarray(edge["path"], dtype=np.float32), axis=0))
        orient = get_polygon_orientation_stage2(pts_for_orient)
        cycles_tokens.append({
            "cycle_id": len(cycles_tokens),
            "members": [int(x) for x in members],
            "orientation": orient
        })

    text_parts = []
    if end_to_end:
        text_parts.append("E2E: " + "、".join(end_to_end))
    if t_junctions:
        text_parts.append("T: " + "、".join(sorted(set(t_junctions))))
    if x_junctions:
        text_parts.append("X: " + "、".join(x_junctions))
    if cycles_tokens:
        text_parts.append("Cycles: " + " | ".join(" ".join(map(str, c["members"])) for c in cycles_tokens))
    if not text_parts:
        text_parts.append("No physical collision")

    return {
        "topology_events": topology_events,
        "cycles": cycles_tokens,
        "topology_text": " ; ".join(text_parts),
        "stats": {
            "stroke_count": int(len(edges)),
            "topology_event_count": int(len(topology_events)),
            "connected_components": int(cc_count),
            "cycle_count": int(len(cycles_tokens)),
            "e2e_count": int(len([e for e in topology_events if e["type"] == "E2E"])),
            "t_count": int(len([e for e in topology_events if e["type"] == "T"])),
            "x_count": int(len([e for e in topology_events if e["type"] == "X"])),
        }
    }


def compute_stage2_topology_from_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return compute_stage2_topology_from_edges(candidate_to_stage2_edges(candidate))


def attach_stage2_topology(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    把第二阶段拓扑引擎结果写回 candidate。
    后续筛选、显示、保存都使用这套结果。
    """
    c = candidate
    result = compute_stage2_topology_from_candidate(c)
    c["topology_events"] = result["topology_events"]
    c["cycles"] = result["cycles"]
    c.setdefault("pcg_meta", {})
    c["pcg_meta"]["stage2_topology_text"] = result["topology_text"]
    c["pcg_meta"]["stage2_topology_stats"] = result["stats"]
    return c


def candidate_has_model_topology(candidate: Dict[str, Any]) -> bool:
    """
    True 表示这个 candidate 的 topology_events/cycles 已经由“拓扑生成模型”生成。
    V7 stage1 recall gate 不会设置这个标记，因为它只做第一阶段粗筛；
    最终 topology_events/cycles 仍由第二阶段规则引擎生成。
    """
    meta = candidate.get("pcg_meta", {})
    if not isinstance(meta, dict):
        return False
    src = str(meta.get("topology_source", ""))
    return src.startswith("topo_model") and isinstance(candidate.get("topology_events"), list)


def ensure_topology_for_bundle(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    保存 bundle 前的拓扑保障：
      - 如果 GUI 勾选了 topo model，candidate 已经带有模型拓扑，则原样保存；
      - 否则继续使用原来的第二阶段规则拓扑引擎。
    """
    if candidate_has_model_topology(candidate):
        return candidate
    return attach_stage2_topology(candidate)


def candidate_to_stage2_bundle_direct(candidate: Dict[str, Any], label: str, hex_key: str) -> Tuple[str, Dict[str, Any]]:
    """
    直接输出与你附件中 TopoAnnotationWorkspace.action_complete_topo() 相同的 5 层结构：
      glyph_info / strokes / topology_events / cycles / edit_history

    对于 PCG 字符：
      - width_bezier 用当前 stroke width 扩展为 [w,w,w,w]
      - topology_events / cycles 使用上面的第二阶段拓扑引擎生成
    """
    c = ensure_topology_for_bundle(copy.deepcopy(candidate))
    edges = candidate_to_stage2_edges(c)
    id_map = stage2_display_map(edges)

    strokes_tokens = []
    ts = np.linspace(0, 1, STAGE2_TOPO_SAMPLE_N)[:, None]

    for edge in edges:
        eid = edge["id"]
        p_opt = np.asarray(edge["path"], dtype=np.float32)
        w = float(edge.get("width", 10.0))
        w_opt = np.asarray([w, w, w, w], dtype=np.float32)

        c_pts = cubic_bezier_np_stage2(p_opt, ts)
        length = float(np.sum(np.linalg.norm(np.diff(c_pts, axis=0), axis=1)))
        xmin, ymin = np.min(c_pts, axis=0)
        xmax, ymax = np.max(c_pts, axis=0)
        s_type = "closed" if np.linalg.norm(p_opt[0] - p_opt[3]) < STAGE2_COLLISION_THRESHOLD else "open"

        strokes_tokens.append({
            "bezier_id": int(id_map[eid]),
            "stroke_type": s_type,
            "length": round(length, 2),
            "bbox": [round(float(xmin), 1), round(float(ymin), 1), round(float(xmax), 1), round(float(ymax), 1)],
            "mother_bezier": p_opt.tolist(),
            "width_bezier": w_opt.tolist()
        })

    bundle = {
        "glyph_info": {
            "hex_key": hex_key,
            "char": chr(int(hex_key, 16)),
            "label": label,
            "font_path": candidate.get("font_path", DEFAULT_FONT_NAME),
            "font_name": candidate.get("font_name", DEFAULT_FONT_NAME),
            "source": "PCG_stage2_topology_engine",
            "candidate_id": _get_candidate_id(candidate),
            "style_mode": candidate.get("style_mode"),
            "topology_family": candidate.get("topology_family"),
        },
        "strokes": strokes_tokens,
        "topology_events": c.get("topology_events", []),
        "cycles": c.get("cycles", []),
        "edit_history": candidate.get("edit_history", []) + [{
            "action": "PCG_GENERATE_AND_LABEL",
            "label": label,
            "candidate_id": _get_candidate_id(candidate),
            "style_mode": candidate.get("style_mode"),
            "topology_family": candidate.get("topology_family"),
            "topology_text": c.get("pcg_meta", {}).get("stage2_topology_text", ""),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }]
    }
    return hex_key, bundle


# =============================================================================
# 2.6 Topo 模型运行时：可选读取 v6 ckptselect 模型生成 topology_events/cycles
# =============================================================================

def _model_sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def _point_to_curve_min_dist_t_pos(point: np.ndarray, P: np.ndarray, sample_n: int = 96) -> Tuple[float, float, np.ndarray]:
    ts = np.linspace(0.0, 1.0, sample_n, dtype=np.float32)[:, None]
    curve = cubic_bezier_np_stage2(np.asarray(P, dtype=np.float32), ts)
    d = np.linalg.norm(curve - np.asarray(point, dtype=np.float32)[None, :], axis=1)
    idx = int(np.argmin(d))
    t = idx / float(max(1, sample_n - 1))
    return float(d[idx]), float(t), curve[idx]


def _stroke_bundle_for_topo_model(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage1-only 模型输入 bundle。

    三阶段约束：
      - Stage1 gate 只能看到拓扑骨架/位置；
      - 不允许使用二阶段 Bézier 形状细节；
      - 不允许使用 width / raster 视觉信息；
      - 不允许使用 topology_events 标签泄漏。

    因此这里把每条 stroke 强制降级成 P0->P3 直线骨架：
      P1/P2 = 线性插值占位
      width_bezier = [1,1,1,1]
    """
    edges = candidate_to_stage2_edges(candidate)
    id_map = stage2_display_map(edges)
    strokes_tokens = []
    ts = np.linspace(0, 1, STAGE2_TOPO_SAMPLE_N)[:, None]

    for edge in edges:
        eid = edge["id"]
        p_raw = np.asarray(edge["path"], dtype=np.float32)
        if p_raw.shape != (4, 2):
            continue

        p_opt = np.zeros((4, 2), dtype=np.float32)
        p_opt[0] = p_raw[0]
        p_opt[3] = p_raw[3]
        p_opt[1] = p_opt[0] * (2.0 / 3.0) + p_opt[3] * (1.0 / 3.0)
        p_opt[2] = p_opt[0] * (1.0 / 3.0) + p_opt[3] * (2.0 / 3.0)
        w_opt = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        c_pts = cubic_bezier_np_stage2(p_opt, ts)
        length = float(np.sum(np.linalg.norm(np.diff(c_pts, axis=0), axis=1)))
        xmin, ymin = np.min(c_pts, axis=0)
        xmax, ymax = np.max(c_pts, axis=0)
        s_type = "closed" if np.linalg.norm(p_opt[0] - p_opt[3]) < STAGE2_COLLISION_THRESHOLD else "open"

        strokes_tokens.append({
            "bezier_id": int(id_map[eid]),
            "stroke_type": s_type,
            "length": round(length, 2),
            "bbox": [round(float(xmin), 1), round(float(ymin), 1), round(float(xmax), 1), round(float(ymax), 1)],
            "mother_bezier": p_opt.tolist(),
            "width_bezier": w_opt.tolist(),
        })

    return {
        "glyph_info": {
            "hex_key": "F0000",
            "char": chr(PUA_BASE),
            "label": "stage1_model_runtime",
            "font_path": candidate.get("font_path", DEFAULT_FONT_NAME),
            "font_name": candidate.get("font_name", DEFAULT_FONT_NAME),
            "source": "PCG_stage1_topology_gate_runtime",
            "candidate_id": _get_candidate_id(candidate),
            "style_mode": candidate.get("style_mode"),
            "topology_family": candidate.get("topology_family"),
        },
        "strokes": strokes_tokens,
        "topology_events": [],
        "cycles": [],
        "edit_history": [],
    }


def _node_desc_for_model_index(node_idx: int, stroke_count: int) -> Dict[str, Any]:
    node_idx = int(node_idx)
    stroke_count = int(stroke_count)
    if node_idx < stroke_count:
        return {"node_type": "stroke", "stroke_idx": node_idx, "endpoint": None}
    k = node_idx - stroke_count
    stroke_idx = k // 2
    endpoint = k % 2
    return {
        "node_type": "endpoint_start" if endpoint == 0 else "endpoint_end",
        "stroke_idx": int(stroke_idx),
        "endpoint": int(endpoint),
    }


def _position_from_stroke_endpoint(stroke: Dict[str, Any], endpoint: int) -> np.ndarray:
    P = np.asarray(stroke.get("mother_bezier", []), dtype=np.float32)
    if P.shape != (4, 2):
        return np.zeros((2,), dtype=np.float32)
    return P[0] if int(endpoint) == 0 else P[3]


def _topology_summary_from_model_events(strokes: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    根据模型预测出来的 E2E/T/X 事件构造 cycles/stats/text。
    不再调用规则拓扑检测，只根据模型事件建图。
    """
    display_ids = []
    for i, st in enumerate(strokes):
        try:
            display_ids.append(int(st.get("bezier_id", i + 1)))
        except Exception:
            display_ids.append(i + 1)

    graph_edges: List[Tuple[int, int]] = []
    connection_points: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)

    def add_connection(a, b, pos):
        try:
            a, b = int(a), int(b)
        except Exception:
            return
        if a == b:
            return
        u, v = min(a, b), max(a, b)
        graph_edges.append((u, v))
        if pos is not None:
            connection_points[(u, v)].append(np.asarray(pos, dtype=np.float32))

    for ev in events:
        if not isinstance(ev, dict):
            continue
        typ = ev.get("type")
        pos = ev.get("position")
        if isinstance(pos, list) and len(pos) == 2:
            pos_arr = np.asarray(pos, dtype=np.float32)
        else:
            pos_arr = None
        if typ == "E2E":
            add_connection(ev.get("stroke_a"), ev.get("stroke_b"), pos_arr)
        elif typ == "T":
            add_connection(ev.get("guest"), ev.get("host"), pos_arr)
        elif typ == "X":
            add_connection(ev.get("stroke_a"), ev.get("stroke_b"), pos_arr)

    unique_graph_edges = sorted(set(graph_edges))
    active_nodes = sorted(set(display_ids))

    if nx is not None:
        G = nx.Graph()
        G.add_nodes_from(active_nodes)
        G.add_edges_from(unique_graph_edges)
        cc_count = nx.number_connected_components(G) if active_nodes else 0
        basis = nx.cycle_basis(G)
    else:
        parent = {n: n for n in active_nodes}
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
        for u, v in unique_graph_edges:
            union(u, v)
        cc_count = len(set(find(n) for n in active_nodes)) if active_nodes else 0
        basis = _simple_cycle_basis_fallback(active_nodes, unique_graph_edges)

    valid_cycle_lists = []

    # 2-stroke cycle: same two strokes connected at two distinct physical points
    for (u, v), pts_list in connection_points.items():
        if len(pts_list) >= 2:
            for a in range(len(pts_list)):
                for b in range(a + 1, len(pts_list)):
                    if np.linalg.norm(pts_list[a] - pts_list[b]) >= STAGE2_CYCLE_POINT_THRESHOLD:
                        valid_cycle_lists.append([u, v])
                        break
                else:
                    continue
                break

    for cyc in basis:
        members = [int(x) for x in cyc]
        if len(members) >= 3:
            valid_cycle_lists.append(members)

    dedup_cycles = []
    seen = set()
    for cyc in valid_cycle_lists:
        key = tuple(sorted([int(x) for x in cyc]))
        if key not in seen:
            seen.add(key)
            dedup_cycles.append([int(x) for x in cyc])

    # orientation from stroke centers
    id_to_stroke = {}
    for i, st in enumerate(strokes):
        try:
            bid = int(st.get("bezier_id", i + 1))
        except Exception:
            bid = i + 1
        id_to_stroke[bid] = st

    cycles_tokens = []
    for members in dedup_cycles:
        pts_for_orient = []
        for n in members:
            st = id_to_stroke.get(int(n))
            if st is not None:
                P = np.asarray(st.get("mother_bezier", []), dtype=np.float32)
                if P.shape == (4, 2):
                    pts_for_orient.append(np.mean(P, axis=0))
        orient = get_polygon_orientation_stage2(pts_for_orient)
        cycles_tokens.append({
            "cycle_id": len(cycles_tokens),
            "members": [int(x) for x in members],
            "orientation": orient,
        })

    e2e = [f"{e.get('stroke_a')}-{e.get('stroke_b')}" for e in events if isinstance(e, dict) and e.get("type") == "E2E"]
    ts = [f"{e.get('guest')}搭{e.get('host')}" for e in events if isinstance(e, dict) and e.get("type") == "T"]
    xs = [f"{e.get('stroke_a')}交叉{e.get('stroke_b')}" for e in events if isinstance(e, dict) and e.get("type") == "X"]

    text_parts = []
    if e2e:
        text_parts.append("E2E(model): " + "、".join(e2e))
    if ts:
        text_parts.append("T(model): " + "、".join(sorted(set(ts))))
    if xs:
        text_parts.append("X(model): " + "、".join(xs))
    if cycles_tokens:
        text_parts.append("Cycles(model): " + " | ".join(" ".join(map(str, c["members"])) for c in cycles_tokens))
    if not text_parts:
        text_parts.append("No model topology above threshold")

    return {
        "topology_events": events,
        "cycles": cycles_tokens,
        "topology_text": " ; ".join(text_parts),
        "stats": {
            "stroke_count": int(len(strokes)),
            "topology_event_count": int(len(events)),
            "connected_components": int(cc_count),
            "cycle_count": int(len(cycles_tokens)),
            "e2e_count": int(len(e2e)),
            "t_count": int(len(ts)),
            "x_count": int(len(xs)),
        }
    }


class TopoModelRuntime:
    """
    Stage1 topology gate runtime.

    三阶段语义：
      Stage1: PCG 生成 anchor/topology skeleton，V7 recall 模型只做高召回粗筛；
      Stage2: 通过 Stage1 后，保留 PCG 的 Bézier 笔画细节，并由 stage2_rule_engine 生成 topology_events/cycles；
      Stage3: aesthetic hooks + 人工 Good/Bad。

    注意：
      这个 V7 模型不再负责生成最终 topology_events/cycles。
      它只输出 quality_score，用于“第一阶段拓扑骨架是否值得进入二阶段”的 gate。
    """
    def __init__(self, index_path: str, purpose: str = DEFAULT_TOPO_MODEL_PURPOSE):
        self.index_path = os.path.abspath(index_path)
        self.purpose = str(purpose or DEFAULT_TOPO_MODEL_PURPOSE)
        self.module = None
        self.torch = None
        self.cfg = None
        self.model = None
        self.device = None
        self.rec = None
        self.ckpt_path = None
        self.thresholds = {}
        self.loaded_at = None

    @staticmethod
    def _candidate_train_scripts() -> List[str]:
        fm = os.path.join(CHAR_GLYPH_DIR, "flywheel_model")
        return [
            os.path.join(fm, "train_flywheel_topohgt_v7_stage1_recall_ckptselect.py"),
            os.path.join(fm, "train_flywheel_topohgt_v7_stage1_topologyonly_ckptselect.py"),
            os.path.join(fm, "train_flywheel_topohgt_v6_relationcalib_ckptselect.py"),
            os.path.join(fm, "train_flywheel_topohgt_v6_relationcalib.py"),
            os.path.join(fm, "train_flywheel_topohgt_v5_binaryheads.py"),
        ]

    def _import_train_module(self):
        last_err = None
        for path in self._candidate_train_scripts():
            if not os.path.exists(path):
                continue
            try:
                module_name = "topo_model_runtime_" + re.sub(r"\\W+", "_", os.path.basename(path))
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if not hasattr(mod, "TopoHGTv6"):
                    last_err = RuntimeError(f"{path} does not define TopoHGTv6")
                    continue
                return mod
            except Exception as e:
                last_err = e
        raise RuntimeError(
            "无法导入 topo 训练脚本。请确认以下文件至少存在一个：\\n"
            + "\\n".join(self._candidate_train_scripts())
            + f"\\nlast_err={repr(last_err)}"
        )

    def load(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                "Topo checkpoint index 不存在：\\n"
                f"{self.index_path}\\n\\n"
                "请先运行 train_flywheel_topohgt_v7_stage1_recall_ckptselect.py，"
                "或在 GUI 里 Browse 到正确的 *_checkpoint_index.json。"
            )

        self.module = self._import_train_module()

        import torch
        self.torch = torch

        if hasattr(self.module, "choose_checkpoint_from_index"):
            rec = self.module.choose_checkpoint_from_index(self.index_path, purpose=self.purpose)
        else:
            with open(self.index_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            rec = idx.get("recommendations", {}).get(self.purpose)
            if rec is None:
                raise KeyError(f"purpose={self.purpose!r} not found in {self.index_path}")

        ckpt_path = rec.get("path")
        if not ckpt_path:
            raise RuntimeError(f"checkpoint record lacks path: {rec}")

        ckpt_path = self._resolve_checkpoint_path(ckpt_path)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg_dict = dict(ckpt.get("config", {}))
        self.cfg = self.module.Cfg(**cfg_dict)
        # Runtime 不需要 dataset cache。
        if hasattr(self.cfg, "precompute_items"):
            self.cfg.precompute_items = False
        if hasattr(self.cfg, "cache_preprocessed_pt"):
            self.cfg.cache_preprocessed_pt = False

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = self.module.TopoHGTv6(
            int(ckpt.get("node_dim")),
            int(ckpt.get("edge_dim")),
            self.cfg,
        )
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.rec = rec
        self.ckpt_path = ckpt_path
        self.thresholds = dict(rec.get("thresholds", {}))
        ms = rec.get("metrics_summary", {})
        self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "purpose": self.purpose,
            "ckpt_path": self.ckpt_path,
            "device": str(self.device),
            "score": rec.get("score"),
            "epoch": rec.get("epoch"),
            "thresholds": self.thresholds,
            "metrics_summary": ms,
        }

    def is_loaded(self) -> bool:
        return self.model is not None and self.module is not None and self.torch is not None

    def _resolve_checkpoint_path(self, ckpt_path: str) -> str:
        """
        checkpoint_index.json 里可能保存的是另一台机器上的绝对路径，例如：
          /Users/.../model/xxx.pth
          C:\\Users\\...\\model\\xxx.pth

        为了在 Mac / Windows 两台机器之间共用 index，这里会优先按
        “当前 index 所在目录 + checkpoint 文件名”解析。
        也就是只要求 checkpoint_index.json 和 .pth 在同一个 model 目录下。
        """
        raw = str(ckpt_path)
        index_dir = os.path.abspath(os.path.dirname(self.index_path))

        candidates = []

        # 1) 原路径本身可用时直接用。
        candidates.append(raw)

        # 2) 如果是相对路径，按 index 所在目录解析。
        if not os.path.isabs(raw):
            candidates.append(os.path.abspath(os.path.join(index_dir, raw)))

        # 3) 无论原路径是 Mac 绝对路径还是 Windows 绝对路径，都取 basename 到当前 index_dir 找。
        # Windows 路径在 Mac/Python 上 basename 可能无法正确拆，所以同时处理 / 和 \\。
        base = raw.replace("\\", "/").split("/")[-1]
        if base:
            candidates.append(os.path.join(index_dir, base))
            candidates.append(os.path.join(CHAR_GLYPH_DIR, "flywheel_model", "model", base))

        # 去重并检查。
        tried = []
        for c in candidates:
            if not c:
                continue
            c = os.path.abspath(c)
            if c in tried:
                continue
            tried.append(c)
            if os.path.exists(c):
                return c

        raise FileNotFoundError(
            "checkpoint 不存在。通常是 checkpoint_index.json 里记录了另一台机器的绝对路径。\\n"
            "请确认 .pth 和 checkpoint_index.json 在当前机器的同一个 model 目录下。\\n\\n"
            f"index_path = {self.index_path}\\n"
            f"raw checkpoint path = {raw}\\n"
            "tried:\\n  " + "\\n  ".join(tried)
        )


    def _threshold(self, key: str, default: float = 0.5) -> float:
        v = self.thresholds.get(key)
        try:
            if v is not None:
                return float(v)
        except Exception:
            pass
        return float(default)

    def make_batch_from_bundle(self, bundle: Dict[str, Any]):
        if not self.is_loaded():
            raise RuntimeError("Topo model is not loaded.")

        cfg = self.cfg
        mod = self.module
        torch = self.torch

        strokes = list(bundle.get("strokes", []))
        if not strokes:
            raise RuntimeError("candidate has no strokes for topo model.")
        max_strokes = int(getattr(cfg, "max_strokes", 16))
        if len(strokes) > max_strokes:
            strokes = strokes[:max_strokes]
            bundle = copy.deepcopy(bundle)
            bundle["strokes"] = strokes

        max_nodes = max_strokes * 3
        metas = mod.make_node_meta(strokes, cfg)
        n_nodes = len(metas)

        node_features_valid = np.asarray([mod.node_feature(m, i, cfg) for i, m in enumerate(metas)], dtype=np.float32)
        node_dim = int(node_features_valid.shape[-1])
        node_features = np.zeros((max_nodes, node_dim), dtype=np.float32)
        node_features[:n_nodes] = node_features_valid

        edge_dim = int(mod.edge_feature(metas[0], metas[0], cfg).shape[-1])
        edge_features = np.zeros((max_nodes, max_nodes, edge_dim), dtype=np.float32)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    edge_features[i, j] = 0.0
                else:
                    edge_features[i, j] = mod.edge_feature(metas[i], metas[j], cfg)

        node_mask = np.zeros((max_nodes,), dtype=np.float32)
        node_mask[:n_nodes] = 1.0
        raster = mod.render_raster(bundle, cfg)

        batch = {
            "node_features": torch.tensor(node_features, dtype=torch.float32, device=self.device).unsqueeze(0),
            "edge_features": torch.tensor(edge_features, dtype=torch.float32, device=self.device).unsqueeze(0),
            "node_mask": torch.tensor(node_mask, dtype=torch.float32, device=self.device).unsqueeze(0),
            "raster": torch.tensor(raster, dtype=torch.float32, device=self.device).unsqueeze(0),
        }
        return batch, strokes

    def predict_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_loaded():
            raise RuntimeError("Topo model is not loaded.")
        torch = self.torch
        batch, strokes = self.make_batch_from_bundle(bundle)
        with torch.no_grad():
            out = self.model(batch)

        def cpu_np(name):
            return out[name][0].detach().cpu().float().numpy()

        q = float(torch.sigmoid(out["quality_logit"])[0].detach().cpu().item())
        return {
            "quality_score": q,
            "e2e_prob": _model_sigmoid_np(cpu_np("e2e_logit")),
            "t_prob": _model_sigmoid_np(cpu_np("t_logit")),
            "x_prob": _model_sigmoid_np(cpu_np("x_logit")),
            "strokes": strokes,
        }

    def decode_topology(self, pred: Dict[str, Any]) -> Dict[str, Any]:
        strokes = pred["strokes"]
        n = len(strokes)
        e2e = pred["e2e_prob"]
        tprob = pred["t_prob"]
        xprob = pred["x_prob"]

        th_e2e = self._threshold("relation_E2E_best_f1", 0.60)
        th_t = self._threshold("relation_T_best_f1", 0.55)
        th_x = self._threshold("relation_X_best_f1", 0.70)

        events: List[Dict[str, Any]] = []

        def sid(sidx):
            try:
                return int(strokes[sidx].get("bezier_id", sidx + 1))
            except Exception:
                return int(sidx + 1)

        # E2E: endpoint endpoint, undirected; use max of both directions
        endpoint_nodes = []
        for s in range(n):
            endpoint_nodes.append((n + 2 * s, s, 0))
            endpoint_nodes.append((n + 2 * s + 1, s, 1))

        seen_e2e = set()
        for a in range(len(endpoint_nodes)):
            ia, sa, ea = endpoint_nodes[a]
            for b in range(a + 1, len(endpoint_nodes)):
                ib, sb, eb = endpoint_nodes[b]
                if sa == sb:
                    continue
                prob = float(max(e2e[ia, ib], e2e[ib, ia]))
                if prob < th_e2e:
                    continue
                key = tuple(sorted([(sid(sa), ea), (sid(sb), eb)]))
                if key in seen_e2e:
                    continue
                seen_e2e.add(key)

                pa = _position_from_stroke_endpoint(strokes[sa], ea)
                pb = _position_from_stroke_endpoint(strokes[sb], eb)
                pos = (pa + pb) / 2.0
                events.append({
                    "type": "E2E",
                    "stroke_a": sid(sa),
                    "t_a": 0.0 if ea == 0 else 1.0,
                    "stroke_b": sid(sb),
                    "t_b": 0.0 if eb == 0 else 1.0,
                    "position": [round(float(pos[0]), 1), round(float(pos[1]), 1)],
                    "model_score": round(prob, 6),
                    "model_threshold": round(th_e2e, 6),
                    "model_head": "E2E",
                })

        # T: endpoint -> stroke, directed
        seen_t = set()
        for i_node, guest_idx, guest_ep in endpoint_nodes:
            guest_sid = sid(guest_idx)
            guest_pos = _position_from_stroke_endpoint(strokes[guest_idx], guest_ep)
            guest_P = np.asarray(strokes[guest_idx].get("mother_bezier", []), dtype=np.float32)
            for host_idx in range(n):
                if host_idx == guest_idx:
                    continue
                prob = float(tprob[i_node, host_idx])
                if prob < th_t:
                    continue
                host_sid = sid(host_idx)
                key = (guest_sid, guest_ep, host_sid)
                if key in seen_t:
                    continue
                seen_t.add(key)

                host_P = np.asarray(strokes[host_idx].get("mother_bezier", []), dtype=np.float32)
                if host_P.shape != (4, 2):
                    continue
                _, host_t, pos = _point_to_curve_min_dist_t_pos(guest_pos, host_P, sample_n=96)

                angle = 0.0
                try:
                    gv = get_bezier_derivative_stage2(guest_P, 0.0 if guest_ep == 0 else 1.0)
                    hv = get_bezier_derivative_stage2(host_P, host_t)
                    angle = get_angle_stage2(gv, hv)
                except Exception:
                    angle = 0.0

                events.append({
                    "type": "T",
                    "guest": guest_sid,
                    "guest_t": 0.0 if guest_ep == 0 else 1.0,
                    "host": host_sid,
                    "host_t": round(float(host_t), 3),
                    "angle": round(float(angle), 1),
                    "position": [round(float(pos[0]), 1), round(float(pos[1]), 1)],
                    "model_score": round(prob, 6),
                    "model_threshold": round(th_t, 6),
                    "model_head": "T",
                })

        # X: stroke <-> stroke, undirected; use max of both directions
        seen_x = set()
        ts = np.linspace(0, 1, STAGE2_TOPO_SAMPLE_N)[:, None]
        for a in range(n):
            for b in range(a + 1, n):
                prob = float(max(xprob[a, b], xprob[b, a]))
                if prob < th_x:
                    continue
                key = tuple(sorted([sid(a), sid(b)]))
                if key in seen_x:
                    continue
                seen_x.add(key)

                P1 = np.asarray(strokes[a].get("mother_bezier", []), dtype=np.float32)
                P2 = np.asarray(strokes[b].get("mother_bezier", []), dtype=np.float32)
                if P1.shape != (4, 2) or P2.shape != (4, 2):
                    continue
                c1 = cubic_bezier_np_stage2(P1, ts)
                c2 = cubic_bezier_np_stage2(P2, ts)
                hit_x, x_pt, t1, t2 = find_polyline_x_intersection_stage2(c1, c2)
                if not hit_x:
                    dd = np.linalg.norm(c1[:, None, :] - c2[None, :, :], axis=2)
                    k = int(np.argmin(dd))
                    i1, i2 = np.unravel_index(k, dd.shape)
                    t1 = i1 / float(STAGE2_TOPO_SAMPLE_N - 1)
                    t2 = i2 / float(STAGE2_TOPO_SAMPLE_N - 1)
                    x_pt = (c1[i1] + c2[i2]) / 2.0

                angle = 0.0
                try:
                    angle = get_angle_stage2(get_bezier_derivative_stage2(P1, t1), get_bezier_derivative_stage2(P2, t2))
                except Exception:
                    angle = 0.0

                events.append({
                    "type": "X",
                    "stroke_a": sid(a),
                    "t_a": round(float(t1), 3),
                    "stroke_b": sid(b),
                    "t_b": round(float(t2), 3),
                    "angle": round(float(angle), 1),
                    "position": [round(float(x_pt[0]), 1), round(float(x_pt[1]), 1)],
                    "model_score": round(prob, 6),
                    "model_threshold": round(th_x, 6),
                    "model_head": "X",
                })

        return _topology_summary_from_model_events(strokes, events)

    def attach_model_topology(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        兼容旧函数名，但现在只做 Stage1 quality scoring。

        不写入 topology_events / cycles。
        最终 topology_events / cycles 由第二阶段规则引擎生成。
        """
        c = copy.deepcopy(candidate)
        bundle = _stroke_bundle_for_topo_model(c)
        pred = self.predict_bundle(bundle)

        c.setdefault("pcg_meta", {})
        c["pcg_meta"]["topology_source"] = "stage1_gate_v7_then_stage2_rule_engine"
        c["pcg_meta"]["stage1_model_gate"] = {
            "passed": None,
            "input_view": "P0/P3 skeleton only; P1/P2 linearized; width stripped; no raster/topology label",
            "model_family": "v7_stage1_recall",
        }
        c["pcg_meta"]["topo_model"] = {
            "index_path": self.index_path,
            "checkpoint_path": self.ckpt_path,
            "purpose": self.purpose,
            "loaded_at": self.loaded_at,
            "device": str(self.device),
            "quality_score": round(float(pred["quality_score"]), 6),
            "thresholds": self.thresholds,
            "candidate_id": _get_candidate_id(c),
            "stage": "stage1_topology_gate_only",
            "does_generate_final_topology": False,
        }
        return c



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
    v7 默认直接输出与第二阶段 action_complete_topo() 相同的 5 层结构。
    如需强行调用旧 annotation_flywheel_app_fixed.py，把 USE_STAGE2_DIRECT_BUNDLE 改为 False。
    """
    if USE_STAGE2_DIRECT_BUNDLE:
        return candidate_to_stage2_bundle_direct(candidate, label, hex_key)

    # 以下是可选兼容路径：严格复用第二阶段人工标注脚本的 candidate_to_char_bundle。
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

    topo_text_parts = []
    evs = bundle.get("topology_events", [])
    if isinstance(evs, list):
        e2e = [f"{e.get('stroke_a')}-{e.get('stroke_b')}" for e in evs if isinstance(e, dict) and e.get("type") == "E2E"]
        ts = [f"{e.get('guest')}搭{e.get('host')}" for e in evs if isinstance(e, dict) and e.get("type") == "T"]
        xs = [f"{e.get('stroke_a')}交叉{e.get('stroke_b')}" for e in evs if isinstance(e, dict) and e.get("type") == "X"]
        if e2e: topo_text_parts.append("E2E: " + "、".join(e2e))
        if ts: topo_text_parts.append("T: " + "、".join(ts))
        if xs: topo_text_parts.append("X: " + "、".join(xs))
    cys = bundle.get("cycles", [])
    if isinstance(cys, list) and cys:
        topo_text_parts.append("Cycles: " + " | ".join(" ".join(map(str, c.get("members", []))) for c in cys if isinstance(c, dict)))
    meta = gi.get("pcg_meta", {}) if isinstance(gi.get("pcg_meta", {}), dict) else {}
    if topo_text_parts:
        meta["stage2_topology_text"] = " ; ".join(topo_text_parts)

    return {
        "candidate_id": cid,
        "glyph_candidate_id": cid,
        "generated_glyph_id": cid,
        "style_mode": gi.get("style_mode", ""),
        "topology_family": gi.get("topology_family", ""),
        "nodes": nodes,
        "solved_nodes": nodes,
        "topology_events": bundle.get("topology_events", []),
        "cycles": bundle.get("cycles", []),
        "pcg_meta": meta,
    }


# =============================================================================
# 5. 复杂度计算
# =============================================================================

def candidate_complexity_from_existing_topology(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    当 topology_events 已由模型生成时，复杂度直接从已有事件计算，
    避免再次调用第二阶段规则引擎覆盖/改写模型拓扑语义。
    """
    events = candidate.get("topology_events", [])
    if not isinstance(events, list):
        events = []
    cycles = candidate.get("cycles", [])
    if not isinstance(cycles, list):
        cycles = []

    stroke_count = len(_get_nodes(candidate))
    graph_edges = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("type") == "E2E":
            a, b = ev.get("stroke_a"), ev.get("stroke_b")
        elif ev.get("type") == "T":
            a, b = ev.get("guest"), ev.get("host")
        elif ev.get("type") == "X":
            a, b = ev.get("stroke_a"), ev.get("stroke_b")
        else:
            continue
        try:
            a, b = int(a), int(b)
            if a != b:
                graph_edges.append((min(a, b), max(a, b)))
        except Exception:
            pass

    active = list(range(1, stroke_count + 1))
    if nx is not None:
        G = nx.Graph()
        G.add_nodes_from(active)
        G.add_edges_from(sorted(set(graph_edges)))
        cc = nx.number_connected_components(G) if active else 0
    else:
        parent = {n: n for n in active}
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
        for a, b in graph_edges:
            union(a, b)
        cc = len(set(find(n) for n in active)) if active else 0

    e2e = [e for e in events if isinstance(e, dict) and e.get("type") == "E2E"]
    ts = [e for e in events if isinstance(e, dict) and e.get("type") == "T"]
    xs = [e for e in events if isinstance(e, dict) and e.get("type") == "X"]

    topo_text = candidate.get("pcg_meta", {}).get("stage2_topology_text", "")
    if not topo_text:
        parts = []
        if e2e:
            parts.append("E2E(model): " + "、".join(f"{e.get('stroke_a')}-{e.get('stroke_b')}" for e in e2e))
        if ts:
            parts.append("T(model): " + "、".join(f"{e.get('guest')}搭{e.get('host')}" for e in ts))
        if xs:
            parts.append("X(model): " + "、".join(f"{e.get('stroke_a')}交叉{e.get('stroke_b')}" for e in xs))
        if cycles:
            parts.append("Cycles(model): " + " | ".join(" ".join(map(str, c.get("members", []))) for c in cycles if isinstance(c, dict)))
        topo_text = " ; ".join(parts) if parts else "No model topology above threshold"

    return {
        "stroke_count": int(stroke_count),
        "edge_count": int(len(events)),
        "connected_components": int(cc),
        "cycle_count": int(len(cycles)),
        "e2e_count": int(len(e2e)),
        "t_count": int(len(ts)),
        "x_count": int(len(xs)),
        "topology_text": topo_text,
        "style_mode": candidate.get("style_mode"),
        "topology_family": candidate.get("topology_family"),
    }



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
    """
    v7：复杂度直接使用第二阶段拓扑反馈引擎。
    v9-topomodel：如果 candidate 已有模型生成的 topology_events，则直接使用模型拓扑计算复杂度。
    """
    if candidate_has_model_topology(candidate):
        return candidate_complexity_from_existing_topology(candidate)

    try:
        result = compute_stage2_topology_from_candidate(candidate)
        stats = result["stats"]
        return {
            "stroke_count": int(stats["stroke_count"]),
            "edge_count": int(stats["topology_event_count"]),
            "connected_components": int(stats["connected_components"]),
            "cycle_count": int(stats["cycle_count"]),
            "e2e_count": int(stats["e2e_count"]),
            "t_count": int(stats["t_count"]),
            "x_count": int(stats["x_count"]),
            "topology_text": result["topology_text"],
            "style_mode": candidate.get("style_mode"),
            "topology_family": candidate.get("topology_family"),
        }
    except Exception as e:
        stroke_count = len(_get_nodes(candidate))
        return {
            "stroke_count": int(stroke_count),
            "edge_count": 0,
            "connected_components": int(stroke_count),
            "cycle_count": 0,
            "topology_text": f"stage2_topology_failed: {repr(e)}",
            "style_mode": candidate.get("style_mode"),
            "topology_family": candidate.get("topology_family"),
        }


def in_range(x: int, lo: Optional[int], hi: Optional[int]) -> bool:
    if lo is not None and x < int(lo):
        return False
    if hi is not None and x > int(hi):
        return False
    return True


def candidate_generation_filter_complexity(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate 阶段用于 stroke_count / CC / cycle sliders 的复杂度。

    重要：
      - 未启用 topo model 时，沿用第二阶段规则拓扑复杂度；
      - 启用 topo model 时，不能用模型预测出来的 cycles 做硬过滤。
        否则模型一旦预测出很多 E2E/T/X，cycle_count 会远超 UI 的 cycle_max，
        导致整屏候选全部被 reject。
      - 所以模型模式下，Generate 的复杂度过滤回到“程序生成的 anchor graph”
        只控制笔画结构；保存/显示的 topology_events/cycles 仍然来自模型。
    """
    if not candidate_has_model_topology(candidate):
        return candidate_complexity(candidate)

    stroke_count = len(_get_nodes(candidate))

    anchor_edges = []
    raw_edges = candidate.get("anchor_edges", [])
    if isinstance(raw_edges, list):
        for e in raw_edges:
            if isinstance(e, dict):
                a = e.get("anchor_start", e.get("start", e.get("u")))
                b = e.get("anchor_end", e.get("end", e.get("v")))
                try:
                    anchor_edges.append((int(a), int(b)))
                except Exception:
                    pass
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                try:
                    anchor_edges.append((int(e[0]), int(e[1])))
                except Exception:
                    pass

    # fallback: 从每个 node 的 anchor_start / anchor_end 恢复
    if not anchor_edges:
        for nd in _get_nodes(candidate):
            try:
                anchor_edges.append((int(nd.get("anchor_start")), int(nd.get("anchor_end"))))
            except Exception:
                pass

    cc, cyc = graph_complexity_from_anchor_edges(stroke_count, anchor_edges)
    return {
        "stroke_count": int(stroke_count),
        "edge_count": int(len(anchor_edges)),
        "connected_components": int(cc),
        "cycle_count": int(cyc),
        "e2e_count": 0,
        "t_count": 0,
        "x_count": 0,
        "topology_text": "PCG anchor graph complexity filter; topology_events are generated by topo model",
        "style_mode": candidate.get("style_mode"),
        "topology_family": candidate.get("topology_family"),
    }


def passes_complexity(candidate: Dict[str, Any], cfg: GeneratorConfig) -> Tuple[bool, Dict[str, Any], str]:
    # display/save complexity: model 模式下来自模型 topology_events/cycles
    c_display = candidate_complexity(candidate)

    # generation filter complexity: model 模式下来自 PCG anchor graph，避免模型 cycles 造成全 reject
    c_filter = candidate_generation_filter_complexity(candidate)

    rules = [
        ("stroke_count", cfg.stroke_min, cfg.stroke_max),
        ("connected_components", cfg.cc_min, cfg.cc_max),
        ("cycle_count", cfg.cycle_min, cfg.cycle_max),
    ]
    for key, lo, hi in rules:
        if not in_range(c_filter[key], lo, hi):
            c_display = dict(c_display)
            c_display["generation_filter_complexity"] = c_filter
            c_display["complexity_filter_source"] = "pcg_anchor_graph" if candidate_has_model_topology(candidate) else "stage2_rule_engine"
            return False, c_display, f"{key}_out_of_range"

    c_display = dict(c_display)
    c_display["generation_filter_complexity"] = c_filter
    c_display["complexity_filter_source"] = "pcg_anchor_graph" if candidate_has_model_topology(candidate) else "stage2_rule_engine"
    return True, c_display, "ok"


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
    """
    快速 split writer。

    旧版本每加入一个样本都会 json.dumps(整个 current chunk) 来判断大小，
    对几千/几万样本会退化成 O(n^2)，Compact 时 CPU 会爆、风扇狂转。

    新版本只估算“单条样本”的 JSON 大小并累加，复杂度约 O(n)。
    分片大小是近似值，但对 80MB 分片足够安全。
    """
    ensure_dir(out_dir)

    if clear_old:
        for fn in os.listdir(out_dir):
            if fn.startswith(prefix) and fn.endswith("_topo.json"):
                try:
                    os.remove(os.path.join(out_dir, fn))
                except Exception:
                    pass

    max_bytes = int(max_json_mb * 1024 * 1024)
    files: List[str] = []
    current: Dict[str, Any] = {}
    current_bytes = 2  # {}

    part = 0

    def flush(d, idx):
        if not d:
            return None
        fp = os.path.join(out_dir, f"{prefix}_part{idx:03d}_topo.json")
        save_json(d, fp)
        return fp

    for k, v in data_dict.items():
        # 只计算单条 entry 大小，避免每次 dump 当前大 chunk
        try:
            entry_bytes = len(json.dumps({k: v}, ensure_ascii=False, indent=2).encode("utf-8")) + 4
        except Exception:
            entry_bytes = 1024 * 1024  # 极端异常时保守估计 1MB

        if current and (current_bytes + entry_bytes > max_bytes):
            fp = flush(current, part)
            if fp:
                files.append(fp)
            part += 1
            current = {}
            current_bytes = 2

        current[k] = v
        current_bytes += entry_bytes

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


def pool_file_matches_label(fn: str, label: str) -> bool:
    """
    兼容读取不同版本脚本写出的 Good/Bad pool 文件。

    旧版本问题：
      v5/v7/v8/local_pool 的 POOL_PREFIX 不一样。
      原 load_pool_dict_from_disk() 只读取当前 POOL_PREFIX，
      所以把旧 pcg_filebacked_stage2_schema 剪切过来后，Manage 看不到。

    现在改成：
      good 文件夹里读取所有包含 AestheticGood 且 *_topo.json 的分片文件；
      bad  文件夹里读取所有包含 AestheticBad  且 *_topo.json 的分片文件。
    """
    if not fn.endswith("_topo.json"):
        return False

    tag = "AestheticGood" if label == "good" else "AestheticBad"

    # 当前版本 active pool
    if fn.startswith(pool_file_prefix(label) + "_part"):
        return True

    # 兼容历史版本 active pool：
    # PCG_Direct_FileBacked_AestheticGood_part000_topo.json
    # PCG_Direct_Stage2Schema_AestheticGood_part000_topo.json
    # PCG_Direct_Stage2TopoEngine_AestheticGood_part000_topo.json
    # PCG_Direct_Stage2TopoEngineXStableFast_AestheticGood_part000_topo.json
    if tag in fn and "_part" in fn:
        return True

    return False


def load_pool_dict_from_disk(out_root: str, label: str) -> Dict[str, Any]:
    """
    Manage 直接调用这个函数读取磁盘文件。

    v8_local_pool_legacy_loader:
      不再只读取当前 POOL_PREFIX；
      会兼容读取当前 good/bad 文件夹中历史版本写出的 AestheticGood/AestheticBad 分片。
    """
    folder = pool_label_dir(out_root, label)
    if not os.path.exists(folder):
        return {}

    merged: Dict[str, Any] = {}
    loaded_files = 0

    for fn in sorted(os.listdir(folder)):
        if not pool_file_matches_label(fn, label):
            continue

        fp = os.path.join(folder, fn)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                merged.update(data)
                loaded_files += 1
        except Exception:
            print(f"[WARN] failed to load pool file: {fp}")

    if loaded_files > 0:
        print(f"[PoolLoad] label={label}, files={loaded_files}, items={len(merged)}, folder={folder}")

    return merged


def clear_all_pool_files_for_label(folder: str, label: str) -> None:
    """
    重写 pool 时，清理当前版本和历史版本的同类分片文件。
    这样一旦 Commit / Manage Remove 触发重写，旧 prefix 文件会被迁移成当前 prefix。
    """
    if not os.path.exists(folder):
        return

    for fn in os.listdir(folder):
        if not pool_file_matches_label(fn, label):
            continue
        try:
            os.remove(os.path.join(folder, fn))
        except Exception:
            pass


def write_pool_dict_to_disk(out_root: str, label: str, pool_dict: Dict[str, Any], max_json_mb: int = 80) -> List[str]:
    """
    Commit / Manage 删除后直接调用这个函数重写磁盘 pool。

    注意：
      这里会先清理同 label 的历史 prefix 分片，再写成当前 prefix。
      等价于自动迁移旧 pool 文件名。
    """
    folder = pool_label_dir(out_root, label)
    prefix = pool_file_prefix(label)

    ensure_dir(folder)
    clear_all_pool_files_for_label(folder, label)

    return write_split_json_pool(
        pool_dict,
        folder,
        prefix,
        max_json_mb=max_json_mb,
        clear_old=False,
    )


def load_all_pools_from_disk(out_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return (
        load_pool_dict_from_disk(out_root, "good"),
        load_pool_dict_from_disk(out_root, "bad"),
    )


def normalize_hex_key_value(value: Any) -> Optional[str]:
    """
    把各种 hex key 写法统一成大写无前缀字符串：
      "E000" / "U+E000" / "0xE000" / 57344 -> "E000"
      "F0000" -> "F0000"
    """
    if value is None:
        return None
    try:
        if isinstance(value, int):
            code = int(value)
        else:
            s = str(value).strip().upper()
            if not s:
                return None
            if s.startswith("U+"):
                s = s[2:]
            if s.startswith("0X"):
                s = s[2:]
            code = int(s, 16)
        if code < 0:
            return None
        return f"{code:X}"
    except Exception:
        return None


def code_to_hex_key(code: int) -> str:
    return f"{int(code):X}"


def code_to_char_safe(code: int) -> str:
    try:
        return chr(int(code))
    except Exception:
        return ""


def iter_bundle_hex_candidates(key: Any, bundle: Any):
    """
    从 outer key 与 glyph_info 中提取可能的编码。
    用于 build index / compact 冲突检查。
    """
    hk = normalize_hex_key_value(key)
    if hk:
        yield hk

    if isinstance(bundle, dict):
        gi = bundle.get("glyph_info", {})
        if isinstance(gi, dict):
            for field in ["hex_key", "unicode_hex", "unicode", "codepoint"]:
                hk = normalize_hex_key_value(gi.get(field))
                if hk:
                    yield hk


def set_bundle_codepoint(bundle: Any, new_hex: str, label: str, old_key: Any = None, compact_event: bool = False) -> Any:
    """
    统一更新 bundle 内部的 glyph_info 编码字段。
    不改变 strokes/topology_events/cycles 的 schema。
    """
    if not isinstance(bundle, dict):
        return bundle

    code = int(str(new_hex), 16)
    ch = code_to_char_safe(code)

    gi = bundle.setdefault("glyph_info", {})
    if isinstance(gi, dict):
        old_inner = {
            "outer_key": str(old_key) if old_key is not None else None,
            "hex_key": gi.get("hex_key"),
            "unicode_hex": gi.get("unicode_hex"),
            "char": gi.get("char"),
        }
        # 首次 compact 时保留旧编码线索，后续不覆盖
        gi.setdefault("original_codepoint_before_compact", old_inner)

        # 兼容第二阶段字段：它用 hex_key；某些旧 converter 用 unicode_hex
        gi["hex_key"] = str(new_hex)
        gi["unicode_hex"] = str(new_hex)
        gi["char"] = ch
        gi["label"] = label

    if compact_event:
        bundle.setdefault("edit_history", [])
        if isinstance(bundle["edit_history"], list):
            bundle["edit_history"].append({
                "action": "COMPACT_POOL_REASSIGN_CODEPOINT",
                "old_outer_key": str(old_key) if old_key is not None else None,
                "new_hex_key": str(new_hex),
                "label": label,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "annotation_flywheel_pcg_app_no_json_v8_fast_x_stable_local_pool_incremental_commit_compact_fast.py",
            })

    return bundle


def next_available_hex_key(good_dict: Dict[str, Any], bad_dict: Dict[str, Any]) -> str:
    max_code = PUA_BASE - 1
    for d in [good_dict, bad_dict]:
        for k, bundle in d.items():
            for hk in iter_bundle_hex_candidates(k, bundle):
                try:
                    max_code = max(max_code, int(hk, 16))
                except Exception:
                    pass
    code = max(PUA_BASE, max_code + 1)
    if code > PUA_MAX:
        raise RuntimeError(f"PUA code exhausted: next=U+{code:X}, max=U+{PUA_MAX:X}")
    return code_to_hex_key(code)


def increment_hex_key(hex_key: str, used: set) -> str:
    try:
        code = int(str(hex_key), 16)
    except Exception:
        code = PUA_BASE
    code = max(code, PUA_BASE)
    while code_to_hex_key(code) in used:
        code += 1
        if code > PUA_MAX:
            raise RuntimeError(f"PUA code exhausted: next=U+{code:X}, max=U+{PUA_MAX:X}")
    return code_to_hex_key(code)


# =============================================================================
# 6.6 增量 Commit 加速 + 编码索引 + Compact Pool
# =============================================================================

def pool_commit_index_path(out_root: str) -> str:
    return os.path.join(out_root, f"{POOL_PREFIX}_pool_index.json")


def build_pool_commit_index(out_root: str) -> Dict[str, Any]:
    """
    扫描磁盘 pool，建立 commit 索引。

    这个函数会读取全部 good/bad pool，因此可能慢；
    但只在第一次运行、Refresh Disk Count、Compact Pool、或者索引丢失时触发。
    """
    good, bad = load_all_pools_from_disk(out_root)

    max_code = PUA_BASE - 1
    seen_hex = set()

    for d in [good, bad]:
        for k, bundle in d.items():
            for hk in iter_bundle_hex_candidates(k, bundle):
                seen_hex.add(hk)
                try:
                    code = int(hk, 16)
                    # 旧 E000 段不推动新分配；新分配统一从 F0000 起
                    if code >= PUA_BASE:
                        max_code = max(max_code, code)
                except Exception:
                    pass

    next_code = max(PUA_BASE, max_code + 1)
    if next_code > PUA_MAX:
        raise RuntimeError(f"PUA code exhausted: next=U+{next_code:X}, max=U+{PUA_MAX:X}")

    index = {
        "version": "pool_commit_index_v3_pua15_compact",
        "pool_prefix": POOL_PREFIX,
        "pua_base": int(PUA_BASE),
        "pua_max": int(PUA_MAX),
        "next_code": int(next_code),
        "good_count": int(len(good)),
        "bad_count": int(len(bad)),
        "seen_hex_count": int(len(seen_hex)),
        "rebuilt_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(index, pool_commit_index_path(out_root))
    return index


def load_or_build_pool_commit_index(out_root: str) -> Dict[str, Any]:
    """
    读取增量 commit 索引；不存在、旧版本、或编码区间不对则扫描磁盘建立一次。
    """
    fp = pool_commit_index_path(out_root)
    if os.path.exists(fp):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                idx = json.load(f)

            if isinstance(idx, dict) and "next_code" in idx:
                next_code = int(idx.get("next_code", PUA_BASE))
                version = str(idx.get("version", ""))

                # 旧索引可能从 E000 开始；这里强制重建，避免继续用 BMP PUA。
                if next_code >= PUA_BASE and version == "pool_commit_index_v3_pua15_compact":
                    idx.setdefault("good_count", 0)
                    idx.setdefault("bad_count", 0)
                    return idx
        except Exception:
            pass

    return build_pool_commit_index(out_root)


def save_pool_commit_index(out_root: str, index: Dict[str, Any]) -> None:
    index["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(index, pool_commit_index_path(out_root))


def write_incremental_pool_fragment(out_root: str, label: str, add_dict: Dict[str, Any]) -> Optional[str]:
    """
    只写本次新增样本，不再重写整个 pool。
    Manage/load 会自动合并所有 part 文件。
    """
    if not add_dict:
        return None

    folder = pool_label_dir(out_root, label)
    ensure_dir(folder)
    prefix = pool_file_prefix(label)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"{int(time.time() * 1000) % 1000000:06d}_{random.randint(0, 999999):06d}"
    fp = os.path.join(folder, f"{prefix}_partinc_{stamp}_{suffix}_topo.json")
    save_json(add_dict, fp)
    return fp


def compact_merge_pool_fast(out_root: str, max_json_mb: int = 80) -> Dict[str, Any]:
    """
    快速 Compact：只合并增量分片，不重新分配字符编码。

    用途：
      - 日常清理 partinc 小文件
      - 减少 Manage/Refresh 要读取的文件数量
      - 避免大量样本重新 mapping 导致 CPU/风扇暴涨

    它不会修改 bundle 内的 glyph_info.hex_key / unicode_hex / char。
    如果需要统一从 U+F0000 重编号，再用 Reassign 模式。
    """
    ensure_dir(out_root)
    good_old, bad_old = load_all_pools_from_disk(out_root)

    good_files = write_pool_dict_to_disk(out_root, "good", good_old, max_json_mb=max_json_mb)
    bad_files = write_pool_dict_to_disk(out_root, "bad", bad_old, max_json_mb=max_json_mb)

    idx = build_pool_commit_index(out_root)

    summary = {
        "schema_version": "pcg_pool_fast_compact_summary_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "merge_only_keep_existing_codepoints",
        "pool_prefix": POOL_PREFIX,
        "output_root": os.path.abspath(out_root),
        "good_count_after": len(good_old),
        "bad_count_after": len(bad_old),
        "next_code": f"U+{int(idx.get('next_code', PUA_BASE)):X}",
        "good_files": good_files,
        "bad_files": bad_files,
    }

    summary_path = os.path.join(
        out_root,
        f"{POOL_PREFIX}_FastCompactSummary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    save_json(summary, summary_path)
    summary["summary_path"] = summary_path
    return summary


def compact_reassign_pool(out_root: str, max_json_mb: int = 80, record_per_sample_event: bool = False) -> Dict[str, Any]:
    """
    Compact Pool:
      1. 读取 good/bad 全部历史分片
      2. 从 U+F0000 开始统一重新分配编码，避免旧 E000 / 多版本 prefix 冲突
      3. 更新 glyph_info.hex_key / unicode_hex / char
      4. 重写成少量 split 大文件
      5. 清理旧分片并重建 commit index
      6. 写出 mapping 文件，保留 old_key -> new_key 对照

    注意：不修改 strokes/topology_events/cycles 结构。
    """
    ensure_dir(out_root)
    good_old, bad_old = load_all_pools_from_disk(out_root)

    mapping = {"good": {}, "bad": {}}
    summary = {
        "schema_version": "pcg_pool_compact_mapping_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pool_prefix": POOL_PREFIX,
        "pua_base": f"U+{PUA_BASE:X}",
        "pua_max": f"U+{PUA_MAX:X}",
        "output_root": os.path.abspath(out_root),
        "good_count_before": len(good_old),
        "bad_count_before": len(bad_old),
        "mapping": mapping,
    }

    next_code = PUA_BASE
    good_new: Dict[str, Any] = {}
    bad_new: Dict[str, Any] = {}

    # 稳定顺序：先 Good 后 Bad；各自按旧 key 字符串排序
    for label, src_dict, dst_dict in [
        ("good", good_old, good_new),
        ("bad", bad_old, bad_new),
    ]:
        for old_key, bundle in sorted(src_dict.items(), key=lambda kv: str(kv[0])):
            if next_code > PUA_MAX:
                raise RuntimeError(f"PUA code exhausted during compact: next=U+{next_code:X}, max=U+{PUA_MAX:X}")

            new_hex = code_to_hex_key(next_code)
            next_code += 1

            # bundle 是刚从磁盘 load 出来的对象，可以就地修改，避免 deepcopy 大对象。
            b2 = bundle
            b2 = set_bundle_codepoint(b2, new_hex, label=label, old_key=old_key, compact_event=record_per_sample_event)
            dst_dict[new_hex] = b2
            mapping[label][str(old_key)] = new_hex

    good_files = write_pool_dict_to_disk(out_root, "good", good_new, max_json_mb=max_json_mb)
    bad_files = write_pool_dict_to_disk(out_root, "bad", bad_new, max_json_mb=max_json_mb)

    index = {
        "version": "pool_commit_index_v3_pua15_compact",
        "pool_prefix": POOL_PREFIX,
        "pua_base": int(PUA_BASE),
        "pua_max": int(PUA_MAX),
        "next_code": int(next_code),
        "good_count": int(len(good_new)),
        "bad_count": int(len(bad_new)),
        "compacted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_pool_commit_index(out_root, index)

    summary.update({
        "good_count_after": len(good_new),
        "bad_count_after": len(bad_new),
        "next_code": f"U+{next_code:X}",
        "good_files": good_files,
        "bad_files": bad_files,
    })

    mapping_path = os.path.join(
        out_root,
        f"{POOL_PREFIX}_CompactMapping_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    save_json(summary, mapping_path)
    summary["mapping_path"] = mapping_path
    return summary


# =============================================================================
# 7. GUI 滚动容器
# =============================================================================
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
        self.root.title("Annotation Flywheel PCG App - No JSON Source - File Backed + Stage2 Topology Engine")
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

        # Optional topo model control.
        self.use_topo_model_var = tk.BooleanVar(value=False)
        self.topo_model_index_var = tk.StringVar(value=DEFAULT_TOPO_MODEL_INDEX)
        self.topo_model_purpose_var = tk.StringVar(value=DEFAULT_TOPO_MODEL_PURPOSE)
        self.topo_model_score_mode_var = tk.StringVar(value=DEFAULT_TOPO_MODEL_SCORE_MODE)
        self.topo_model_manual_score_threshold_var = tk.DoubleVar(value=DEFAULT_TOPO_MODEL_MANUAL_SCORE_THRESHOLD)
        self.topo_model_status_var = tk.StringVar(value="Stage1 model: OFF")
        self.topo_model_runtime: Optional[TopoModelRuntime] = None

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

        # 显眼的 Stage1 模型开关：只做第一阶段拓扑骨架粗筛；
        # 最终 topology_events / cycles 仍由第二阶段规则引擎生成。
        row_model = ttk.LabelFrame(top, text="🔥 Stage1 Topology Gate - V7 召回粗筛，只作用第一阶段")
        row_model.pack(fill="x", pady=6)

        row_model_path = ttk.Frame(row_model)
        row_model_path.pack(fill="x", pady=(4, 2))

        ttk.Checkbutton(
            row_model_path,
            text="启用 V7 stage1 模型粗筛；最终 topology 仍由 stage2 规则生成",
            variable=self.use_topo_model_var,
            command=self.on_toggle_topo_model,
        ).pack(side="left", padx=8)
        ttk.Label(row_model_path, text="Index:").pack(side="left", padx=(10, 2))
        ttk.Entry(row_model_path, textvariable=self.topo_model_index_var, width=88).pack(side="left", padx=2)
        ttk.Button(row_model_path, text="Browse", command=self.browse_topo_model_index).pack(side="left", padx=2)

        # 第二行：模型选择 / 阈值 / 加载状态，避免第一行太宽。
        row_model_opts = ttk.Frame(row_model)
        row_model_opts.pack(fill="x", pady=(2, 4))

        ttk.Label(row_model_opts, text="Purpose:").pack(side="left", padx=(8, 2))
        ttk.Combobox(
            row_model_opts,
            textvariable=self.topo_model_purpose_var,
            values=TOPO_MODEL_PURPOSES,
            state="readonly",
            width=20,
        ).pack(side="left", padx=2)

        ttk.Label(row_model_opts, text="Score:").pack(side="left", padx=(12, 2))
        ttk.Combobox(
            row_model_opts,
            textvariable=self.topo_model_score_mode_var,
            values=TOPO_MODEL_SCORE_MODES,
            state="readonly",
            width=22,
        ).pack(side="left", padx=2)

        ttk.Label(row_model_opts, text="Manual th:").pack(side="left", padx=(12, 2))
        ttk.Spinbox(
            row_model_opts,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.topo_model_manual_score_threshold_var,
            width=6,
        ).pack(side="left", padx=2)

        ttk.Button(row_model_opts, text="Load Stage1 Model", command=self.load_topo_model_now).pack(side="left", padx=12)
        ttk.Label(row_model_opts, textvariable=self.topo_model_status_var, foreground="purple").pack(side="left", padx=8)

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
        ttk.Button(row5, text="Compact Pool", command=self.compact_pool).pack(side="left", padx=4)
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

    def browse_topo_model_index(self):
        fp = filedialog.askopenfilename(
            title="Select topo checkpoint_index.json",
            initialdir=os.path.dirname(self.topo_model_index_var.get()) if self.topo_model_index_var.get() else os.path.join(CHAR_GLYPH_DIR, "flywheel_model", "model"),
            filetypes=[("Checkpoint index JSON", "*checkpoint_index*.json"), ("JSON", "*.json"), ("All files", "*.*")]
        )
        if fp:
            self.topo_model_index_var.set(fp)
            self.topo_model_runtime = None
            self.topo_model_status_var.set("Topo model: index selected, not loaded")

    def on_toggle_topo_model(self):
        if self.use_topo_model_var.get():
            self.topo_model_status_var.set("Topo model: ON, will load before Generate")
        else:
            self.topo_model_status_var.set("Stage1 model: OFF")
            self.topo_model_runtime = None

    def load_topo_model_now(self):
        try:
            info = self.ensure_topo_model_loaded(force=True)
            messagebox.showinfo(
                "Topo Model Loaded",
                "Topo 模型已加载。\n\n"
                f"purpose={info.get('purpose')}\n"
                f"epoch={info.get('epoch')} score={info.get('score')}\n"
                f"device={info.get('device')}\n"
                f"ckpt={info.get('ckpt_path')}\n\n"
                f"thresholds={info.get('thresholds')}"
            )
        except Exception as e:
            self.topo_model_runtime = None
            self.topo_model_status_var.set("Stage1 model: LOAD FAILED")
            messagebox.showerror("Stage1 Model Load Failed", traceback.format_exc())

    def ensure_topo_model_loaded(self, force: bool = False) -> Dict[str, Any]:
        if (not force) and self.topo_model_runtime is not None and self.topo_model_runtime.is_loaded():
            return {
                "purpose": self.topo_model_runtime.purpose,
                "ckpt_path": self.topo_model_runtime.ckpt_path,
                "device": str(self.topo_model_runtime.device),
                "score": None if self.topo_model_runtime.rec is None else self.topo_model_runtime.rec.get("score"),
                "epoch": None if self.topo_model_runtime.rec is None else self.topo_model_runtime.rec.get("epoch"),
                "thresholds": self.topo_model_runtime.thresholds,
            }

        index_path = self.topo_model_index_var.get().strip() or DEFAULT_TOPO_MODEL_INDEX
        purpose = self.topo_model_purpose_var.get().strip() or DEFAULT_TOPO_MODEL_PURPOSE
        rt = TopoModelRuntime(index_path=index_path, purpose=purpose)
        info = rt.load()
        self.topo_model_runtime = rt
        self.topo_model_status_var.set(
            f"Topo model: ON | {purpose} | epoch={info.get('epoch')} | score={info.get('score')} | {info.get('device')}"
        )
        return info

    def get_topo_model_score_threshold(self) -> Tuple[Optional[float], str]:
        """
        返回候选进入候选池所需的 quality_score 阈值。
        注意：这个阈值用于“模型打分筛选候选”，不是 E2E/T/X 生成 topology_events 的阈值。
        """
        mode = self.topo_model_score_mode_var.get().strip() or DEFAULT_TOPO_MODEL_SCORE_MODE
        if mode == "off":
            return None, mode
        if mode == "manual":
            try:
                return float(self.topo_model_manual_score_threshold_var.get()), mode
            except Exception:
                return DEFAULT_TOPO_MODEL_MANUAL_SCORE_THRESHOLD, mode

        rt = self.topo_model_runtime
        thresholds = {}
        if rt is not None and rt.rec is not None:
            thresholds = dict(rt.rec.get("thresholds", {}))

        if mode == "quality_recall_target":
            v = thresholds.get("quality_recall_target")
            if v is None:
                return DEFAULT_TOPO_MODEL_MANUAL_SCORE_THRESHOLD, mode
            return float(v), mode

        if mode == "quality_precision_target":
            v = thresholds.get("quality_precision_target")
            if v is None:
                # conservative fallback
                return 0.95, mode
            return float(v), mode

        # default: quality_best_f1
        v = thresholds.get("quality_best_f1")
        if v is None:
            return DEFAULT_TOPO_MODEL_MANUAL_SCORE_THRESHOLD, mode
        return float(v), mode

    def passes_topo_model_score(self, candidate: Dict[str, Any]) -> Tuple[bool, str]:
        th, mode = self.get_topo_model_score_threshold()
        meta = candidate.get("pcg_meta", {}).get("topo_model", {}) if isinstance(candidate.get("pcg_meta", {}), dict) else {}
        try:
            score = float(meta.get("quality_score"))
        except Exception:
            score = None

        candidate.setdefault("pcg_meta", {})
        candidate["pcg_meta"].setdefault("topo_model", {})
        candidate["pcg_meta"]["topo_model"]["score_filter_mode"] = mode
        candidate["pcg_meta"]["topo_model"]["score_filter_threshold"] = th

        if th is None:
            candidate["pcg_meta"]["topo_model"]["score_filter_pass"] = True
            return True, "ok"
        if score is None:
            candidate["pcg_meta"]["topo_model"]["score_filter_pass"] = False
            return False, "topo_model_score_missing"
        ok = score >= float(th)
        candidate["pcg_meta"]["topo_model"]["score_filter_pass"] = bool(ok)
        candidate["pcg_meta"].setdefault("stage1_model_gate", {})
        candidate["pcg_meta"]["stage1_model_gate"]["passed"] = bool(ok)
        candidate["pcg_meta"]["stage1_model_gate"]["threshold"] = th
        candidate["pcg_meta"]["stage1_model_gate"]["score_mode"] = mode
        if not ok:
            return False, "stage1_model_score_below_threshold"
        return True, "ok"

    def attach_topology_for_generation(self, candidate: Dict[str, Any], cfg: GeneratorConfig) -> Dict[str, Any]:
        if not self.use_topo_model_var.get():
            return attach_stage2_topology(candidate)

        self.ensure_topo_model_loaded(force=False)
        if self.topo_model_runtime is None or not self.topo_model_runtime.is_loaded():
            raise RuntimeError("Stage1 topology gate model is enabled but not loaded.")

        # Stage1: V7 recall 模型只给拓扑骨架打分，不生成最终拓扑。
        c = self.topo_model_runtime.attach_model_topology(candidate)

        # Stage2: 最终 topology_events/cycles 仍由规则引擎生成。
        c = attach_stage2_topology(c)
        return c

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

        # Refresh 时做一次完整扫描，并重建增量 commit 索引。
        # 之后 Commit 就不需要每次读取/重写全部 pool。
        idx = build_pool_commit_index(out_root)
        self.status_var.set(
            f"Disk pool: good={idx.get('good_count', 0)}, bad={idx.get('bad_count', 0)} | "
            f"next=U+{int(idx.get('next_code', PUA_BASE)):X} | out_root={out_root}"
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

        if self.use_topo_model_var.get():
            try:
                self.ensure_topo_model_loaded(force=False)
            except Exception:
                self.topo_model_status_var.set("Stage1 model: LOAD FAILED")
                messagebox.showerror("Stage1 Model Load Failed", traceback.format_exc())
                return

        n = int(self.n_var.get())
        max_items = n * n

        selected = []
        reject = Counter()
        attempts = 0
        max_attempts = max(200, max_items * 80)

        while len(selected) < max_items and attempts < max_attempts:
            attempts += 1
            cand = generate_procedural_candidate(cfg, self.rng, attempts)

            # 1) 先按人为配置过滤随机拓扑结构。
            #    这里必须发生在 topo 模型生成 topology_events 之前；
            #    否则模型预测出的 cycles 会反过来改变用户的 cycle_max 语义。
            gen_comp = candidate_generation_filter_complexity(cand)
            for _key, _lo, _hi in [
                ("stroke_count", cfg.stroke_min, cfg.stroke_max),
                ("connected_components", cfg.cc_min, cfg.cc_max),
                ("cycle_count", cfg.cycle_min, cfg.cycle_max),
            ]:
                if not in_range(gen_comp[_key], _lo, _hi):
                    reject[f"{_key}_out_of_range"] += 1
                    break
            else:
                # 2) 通过人为拓扑配置后：
                #    - 如果启用模型：先做 Stage1 V7 recall 粗筛打分，再由 Stage2 规则生成最终 topology；
                #    - 如果未启用模型：直接由 Stage2 规则生成最终 topology。
                try:
                    cand = self.attach_topology_for_generation(cand, cfg)
                except Exception:
                    self.status_var.set("Stage1 model / stage2 topology generation failed.")
                    messagebox.showerror("Stage1/Stage2 Generation Failed", traceback.format_exc())
                    return

                # 3) Stage1 模型模式下，用 quality_score 过滤候选。
                #    只有第一阶段高召回 gate 通过的样本才进入候选池。
                if self.use_topo_model_var.get():
                    ok_score, score_reason = self.passes_topo_model_score(cand)
                    if not ok_score:
                        reject[score_reason] += 1
                        continue

                if not MODEL_HOOKS.prefilter_topology(cand, cfg.topology_condition):
                    reject["model_prefilter_reject"] += 1
                    continue

                # 4) 最终显示/保存复杂度：
                #    - model 模式：来自模型 topology_events/cycles；
                #    - rule 模式：来自第二阶段规则引擎。
                comp = candidate_complexity(cand)
                comp = dict(comp)
                comp["generation_filter_complexity"] = gen_comp
                comp["complexity_filter_source"] = "pcg_anchor_graph"

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
        topo_src = "stage1_model_gate + stage2_rule_engine" if self.use_topo_model_var.get() else "stage2_rule_engine"
        if self.use_topo_model_var.get():
            th, mode = self.get_topo_model_score_threshold()
            filter_src = f"human_topology_config_then_stage1_recall_gate({mode}, th={th})"
        else:
            filter_src = "human_topology_config_then_stage2_rule_engine"
        self.status_var.set(
            f"Generated {len(selected)}/{max_items} direct procedural candidates. "
            f"pipeline=stage1->stage2->stage3. topology={topo_src}. "
            f"filter={filter_src}. attempts={attempts}, reject={dict(reject)}"
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
            topo_line = comp.get("topology_text", cand.get("pcg_meta", {}).get("stage2_topology_text", ""))
            topo_meta = cand.get("pcg_meta", {}).get("topo_model", {}) if isinstance(cand.get("pcg_meta", {}), dict) else {}
            has_stage1_gate = isinstance(topo_meta, dict) and "quality_score" in topo_meta
            topo_badge = "S1MODEL+RULE" if has_stage1_gate else ("MODEL" if candidate_has_model_topology(cand) else "RULE")
            qtxt = ""
            if isinstance(topo_meta, dict) and "quality_score" in topo_meta:
                qtxt = f" Q={topo_meta.get('quality_score')}"
                if topo_meta.get("score_filter_threshold") is not None:
                    qtxt += f">={topo_meta.get('score_filter_threshold')}"
            filter_comp = comp.get("generation_filter_complexity") if isinstance(comp, dict) else None
            filter_txt = ""
            if isinstance(filter_comp, dict):
                filter_txt = f" | filterCC={filter_comp.get('connected_components')} filterCy={filter_comp.get('cycle_count')}"
            txt = (
                f"{idx:03d} {cand.get('style_mode','')} [{topo_badge}]{qtxt}\n"
                f"{cand.get('topology_family','')[:22]}\n"
                f"S={comp.get('stroke_count')} E={comp.get('edge_count')} "
                f"CC={comp.get('connected_components')} Cy={comp.get('cycle_count')}{filter_txt}\n"
                f"{str(topo_line)[:60]}"
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
        增量文件直写版：
        - 不再每次读取整个 Good/Bad pool
        - 不再每次重写整个 Good/Bad pool
        - 只把当前 batch 新增样本写成一个新的 partinc JSON 分片

        注意：
        - 第一次 commit 如果没有索引，会扫描磁盘建立索引，可能慢一次。
        - 点击 Refresh Disk Count 会重建索引。
        - Manage 仍然会读取全部分片用于预览/删除，所以 Manage 可能随数据量变慢，但 Commit 会快很多。
        """
        if not self.current_batch:
            messagebox.showwarning("No batch", "当前没有 batch。")
            return

        out_root = self.output_root.get().strip() or DEFAULT_OUTPUT_ROOT
        ensure_dir(pool_label_dir(out_root, "good"))
        ensure_dir(pool_label_dir(out_root, "bad"))

        idx = load_or_build_pool_commit_index(out_root)
        next_code = int(idx.get("next_code", PUA_BASE))

        good_new: Dict[str, Any] = {}
        bad_new: Dict[str, Any] = {}

        good_added, bad_added = 0, 0

        for cand, var in zip(self.current_batch, self.current_vars):
            label = "good" if var.get() else "bad"

            # 直接从索引分配 hex，不再扫描全部 dict。
            # 新样本统一分配到 Supplementary Private Use Area-A: U+F0000 起。
            if next_code < PUA_BASE:
                next_code = PUA_BASE
            if next_code > PUA_MAX:
                raise RuntimeError(
                    f"PUA code exhausted: next=U+{next_code:X}, max=U+{PUA_MAX:X}. "
                    "请先 Compact Pool 或改用 sample_id 作为主键。"
                )
            hex_key = code_to_hex_key(next_code)
            next_code += 1

            c = copy.deepcopy(cand)
            c["human_label"] = label
            c["font_path"] = c.get("font_path", DEFAULT_FONT_NAME)
            c["font_name"] = c.get("font_name", DEFAULT_FONT_NAME)
            c.setdefault("pcg_meta", {})
            c["pcg_meta"]["human_label"] = label
            c["pcg_meta"]["committed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            c["pcg_meta"]["storage_mode"] = "file_backed_incremental_commit"

            out_hex, bundle = candidate_to_char_bundle(c, label, hex_key)

            # 为避免 key 冲突，磁盘 pool 的外层 key 使用本次分配的 hex_key。
            # 同时更新 bundle 内部 glyph_info.hex_key / unicode_hex / char，
            # 避免 outer key 与内部编码不一致。
            out_hex = hex_key
            bundle = set_bundle_codepoint(bundle, hex_key, label=label, old_key=None, compact_event=False)

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
                        "action": "PCG_DIRECT_FILE_BACKED_INCREMENTAL_COMMIT",
                        "label": label,
                        "candidate_id": _get_candidate_id(c),
                        "style_mode": c.get("style_mode"),
                        "topology_family": c.get("topology_family"),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "source": "annotation_flywheel_pcg_app_no_json_v8_fast_x_stable_local_pool_incremental_commit_compact_fast.py",
                    })

            if label == "good":
                good_new[out_hex] = bundle
                good_added += 1
            else:
                bad_new[out_hex] = bundle
                bad_added += 1

        good_fp = write_incremental_pool_fragment(out_root, "good", good_new)
        bad_fp = write_incremental_pool_fragment(out_root, "bad", bad_new)

        idx["next_code"] = int(next_code)
        idx["good_count"] = int(idx.get("good_count", 0)) + int(good_added)
        idx["bad_count"] = int(idx.get("bad_count", 0)) + int(bad_added)
        save_pool_commit_index(out_root, idx)

        self.status_var.set(
            f"Incremental commit: +good={good_added}, +bad={bad_added}. "
            f"Disk count approx good={idx['good_count']}, bad={idx['bad_count']} | "
            f"files: good={'yes' if good_fp else 'no'}, bad={'yes' if bad_fp else 'no'}"
        )
        messagebox.showinfo(
            "Incremental Commit",
            f"已增量写入本次 batch，不再重写整个 pool。\n\n"
            f"+Good={good_added}\n+Bad={bad_added}\n\n"
            f"Disk Good≈{idx['good_count']}\nDisk Bad≈{idx['bad_count']}"
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

    def compact_pool(self):
        """
        GUI 按钮：Compact Pool

        Yes  = 重新分配编码 + 合并。重操作，只建议旧数据迁移时偶尔用。
        No   = 仅合并分片，不重编号。日常推荐，速度明显更快。
        Cancel = 取消。
        """
        out_root = self.output_root.get().strip() or DEFAULT_OUTPUT_ROOT
        max_mb = int(self.max_json_mb_var.get())

        choice = messagebox.askyesnocancel(
            "Compact Pool",
            "选择 Compact 模式：\n\n"
            "Yes：重新分配字符编码 + 合并分片。\n"
            "     会从 U+F0000 重新 mapping，较慢，样本多时风扇会转。\n"
            "     只建议迁移旧 E000 数据、彻底统一编码时使用。\n\n"
            "No：只合并增量分片，不重新分配编码。\n"
            "    日常推荐，速度更快，也不会改 glyph_info.hex_key/char。\n\n"
            "Cancel：取消。"
        )
        if choice is None:
            return

        do_reassign = bool(choice)

        if do_reassign:
            ok = messagebox.askyesno(
                "Reassign Codepoints",
                "你选择了【重新分配编码】。\n\n"
                "它会读取全部 Good/Bad 分片，从 U+F0000 开始统一重编号，"
                "并重写 good/bad pool 文件。\n\n"
                "样本多时 CPU 和磁盘占用会很高，风扇转是正常的。\n"
                "建议先备份 pcg_filebacked_stage2_schema 文件夹。\n\n"
                "是否继续？"
            )
            if not ok:
                return

        try:
            if do_reassign:
                self.status_var.set("Compacting pool with codepoint reassignment... please wait.")
                self.root.update_idletasks()
                summary = compact_reassign_pool(out_root, max_json_mb=max_mb, record_per_sample_event=False)
                self.status_var.set(
                    f"Reassign compact done: good={summary['good_count_after']}, "
                    f"bad={summary['bad_count_after']}, next={summary['next_code']} | "
                    f"mapping={summary['mapping_path']}"
                )
                messagebox.showinfo(
                    "Reassign Compact Done",
                    f"重新编号 Compact 完成。\n\n"
                    f"Good={summary['good_count_after']}\n"
                    f"Bad={summary['bad_count_after']}\n"
                    f"Next={summary['next_code']}\n\n"
                    f"Mapping 文件：\n{summary['mapping_path']}"
                )
            else:
                self.status_var.set("Fast compacting pool without codepoint reassignment... please wait.")
                self.root.update_idletasks()
                summary = compact_merge_pool_fast(out_root, max_json_mb=max_mb)
                self.status_var.set(
                    f"Fast compact done: good={summary['good_count_after']}, "
                    f"bad={summary['bad_count_after']}, next={summary['next_code']} | "
                    f"summary={summary['summary_path']}"
                )
                messagebox.showinfo(
                    "Fast Compact Done",
                    f"快速 Compact 完成。\n\n"
                    f"Good={summary['good_count_after']}\n"
                    f"Bad={summary['bad_count_after']}\n"
                    f"Next={summary['next_code']}\n\n"
                    f"Summary 文件：\n{summary['summary_path']}"
                )

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Compact Pool Failed", str(e))
            self.status_var.set(f"Compact failed: {e}")

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
            "use_stage2_direct_bundle": USE_STAGE2_DIRECT_BUNDLE,
            "stage2_collision_threshold": STAGE2_COLLISION_THRESHOLD,
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
