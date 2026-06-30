# -*- coding: utf-8 -*-
"""
phase2_edit_history_cleaner.py

Phase 2 人工标注历史清理工具。

用途：
    不修改原始 annotations_topo/*.json，只在读取 edit_history 时做语义清理。

核心规则：
    1. 第一次 SNAP / T_ATTACH 保留，表示“建立拓扑连接”。
    2. 后续同一连接对象上的重复 SNAP / T_ATTACH 视为“已连接结构的位置调整”，
       转换为 CONTROL_MOVE。
    3. 连续作用在同一 stroke/control 上的 CONTROL_MOVE 可以合并为一次移动，
       before 使用第一步 before，after 使用最后一步 after。

该模块同时供：
    - action_stage2_preview_tool_cleaned.py
    - train_topo_phase2_candidate_v4_groupsplit_cleaned.py
调用。
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Tuple


def _deepcopy_jsonable(x: Any) -> Any:
    """尽量用 JSON 方式深拷贝，失败时退化到 copy.deepcopy。"""
    try:
        return json.loads(json.dumps(x, ensure_ascii=False))
    except Exception:
        return copy.deepcopy(x)


def _round_coord(coord: Any) -> Optional[List[float]]:
    """将坐标统一为 [float, float]，非法则返回 None。"""
    if not isinstance(coord, (list, tuple)) or len(coord) < 2:
        return None
    try:
        return [round(float(coord[0]), 2), round(float(coord[1]), 2)]
    except Exception:
        return None


def _same_coord(a: Any, b: Any, eps: float = 1e-6) -> bool:
    aa, bb = _round_coord(a), _round_coord(b)
    if aa is None or bb is None:
        return False
    return abs(aa[0] - bb[0]) <= eps and abs(aa[1] - bb[1]) <= eps


def _control_move_key(op: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """CONTROL_MOVE 的合并 key：同一 stroke + control。"""
    if op.get("action") != "CONTROL_MOVE":
        return None
    stroke = op.get("stroke")
    control = op.get("control")
    if stroke is None or control is None:
        return None
    return str(stroke), str(control)


def _repeat_relation_key(op: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    """重复拓扑连接识别 key。"""
    action = op.get("action")
    if action == "T_ATTACH":
        guest = op.get("guest")
        guest_endpoint = op.get("guest_endpoint")
        host = op.get("host")
        if guest is None or guest_endpoint is None or host is None:
            return None
        # 同一 guest 端点搭在同一 host 曲线上，多次出现视为后续位置调整。
        return ("T_ATTACH", str(guest), str(guest_endpoint), str(host))

    if action == "SNAP":
        stroke = op.get("stroke")
        endpoint = op.get("endpoint")
        host_stroke = op.get("host_stroke")
        host_endpoint = op.get("host_endpoint")
        if stroke is None or endpoint is None or host_stroke is None or host_endpoint is None:
            return None
        # 同一端点与同一 host 端点多次 SNAP，后续视为连接后的整体移动。
        return ("SNAP", str(stroke), str(endpoint), str(host_stroke), str(host_endpoint))

    return None


def _relation_op_to_control_move(op: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """将重复 SNAP / T_ATTACH 转成 CONTROL_MOVE。"""
    action = op.get("action")
    before = _round_coord(op.get("before"))
    after = _round_coord(op.get("after"))
    if before is None or after is None:
        return None

    if action == "T_ATTACH":
        guest = op.get("guest")
        guest_endpoint = op.get("guest_endpoint")
        if guest is None or guest_endpoint is None:
            return None
        new_op = {
            "action": "CONTROL_MOVE",
            "stroke": int(guest) if str(guest).isdigit() else guest,
            "control": str(guest_endpoint),
            "before": before,
            "after": after,
            "source_action": "T_ATTACH",
            "source_host": op.get("host"),
            "source_host_t": op.get("host_t"),
        }
        return new_op

    if action == "SNAP":
        stroke = op.get("stroke")
        endpoint = op.get("endpoint")
        if stroke is None or endpoint is None:
            return None
        new_op = {
            "action": "CONTROL_MOVE",
            "stroke": int(stroke) if str(stroke).isdigit() else stroke,
            "control": str(endpoint),
            "before": before,
            "after": after,
            "source_action": "SNAP",
            "source_host_stroke": op.get("host_stroke"),
            "source_host_endpoint": op.get("host_endpoint"),
        }
        return new_op

    return None


def _normalize_original_op(op: Dict[str, Any]) -> Dict[str, Any]:
    """保留原 action，但规范 before/after 坐标，便于后续 replay/training。"""
    new_op = _deepcopy_jsonable(op)
    before = _round_coord(new_op.get("before"))
    after = _round_coord(new_op.get("after"))
    if before is not None:
        new_op["before"] = before
    if after is not None:
        new_op["after"] = after
    return new_op


def _append_or_merge_move(cleaned: List[Dict[str, Any]], move_op: Dict[str, Any], *, merge_moves: bool) -> None:
    """追加 CONTROL_MOVE；若连续同点移动则合并。"""
    if not merge_moves or not cleaned:
        cleaned.append(move_op)
        return

    last = cleaned[-1]
    if last.get("action") != "CONTROL_MOVE":
        cleaned.append(move_op)
        return

    if _control_move_key(last) != _control_move_key(move_op):
        cleaned.append(move_op)
        return

    # 如果上一条 after 与当前 before 对不上，说明不是连续位置调整，不能安全合并。
    if not _same_coord(last.get("after"), move_op.get("before")):
        cleaned.append(move_op)
        return

    last["after"] = move_op.get("after")

    # 记录来源，方便排查，但不影响 preview/training 对 CONTROL_MOVE 的解析。
    sources = last.setdefault("merged_sources", [])
    src_action = move_op.get("source_action")
    if src_action:
        sources.append({
            "source_action": src_action,
            "source_host": move_op.get("source_host"),
            "source_host_t": move_op.get("source_host_t"),
            "source_host_stroke": move_op.get("source_host_stroke"),
            "source_host_endpoint": move_op.get("source_host_endpoint"),
            "before": move_op.get("before"),
            "after": move_op.get("after"),
        })


def clean_edit_history(edit_history: List[Dict[str, Any]], *, merge_moves: bool = True) -> List[Dict[str, Any]]:
    """
    清理 Phase 2 edit_history。

    参数：
        edit_history: 原始 edit_history 列表。
        merge_moves: 是否合并连续同一控制点的 CONTROL_MOVE。

    返回：
        cleaned edit_history。输入对象不会被修改。
    """
    if not isinstance(edit_history, list):
        return []

    cleaned: List[Dict[str, Any]] = []
    seen_relations = set()

    for raw_op in edit_history:
        if not isinstance(raw_op, dict):
            continue

        op = _normalize_original_op(raw_op)
        action = op.get("action")

        if action in ("T_ATTACH", "SNAP"):
            rel_key = _repeat_relation_key(op)
            if rel_key is not None and rel_key in seen_relations:
                move_op = _relation_op_to_control_move(op)
                if move_op is not None:
                    _append_or_merge_move(cleaned, move_op, merge_moves=merge_moves)
                continue

            if rel_key is not None:
                seen_relations.add(rel_key)
            cleaned.append(op)
            continue

        if action == "CONTROL_MOVE":
            move_op = _normalize_original_op(op)
            _append_or_merge_move(cleaned, move_op, merge_moves=merge_moves)
            continue

        # 未知 action 不擅自删除，原样保留。
        cleaned.append(op)

    return cleaned


def clean_bundle_edit_history(bundle: Dict[str, Any], *, merge_moves: bool = True) -> Dict[str, Any]:
    """
    返回 bundle 深拷贝，并将其中 edit_history 替换为 cleaned edit_history。
    不修改输入 bundle。
    """
    if not isinstance(bundle, dict):
        return {}
    new_bundle = _deepcopy_jsonable(bundle)
    new_bundle["edit_history"] = clean_edit_history(new_bundle.get("edit_history", []), merge_moves=merge_moves)
    return new_bundle


if __name__ == "__main__":
    demo = [
        {"action": "T_ATTACH", "guest": 2, "guest_endpoint": "P0", "host": 1, "host_t": 0.4, "before": [1, 1], "after": [2, 2]},
        {"action": "T_ATTACH", "guest": 2, "guest_endpoint": "P0", "host": 1, "host_t": 0.5, "before": [2, 2], "after": [3, 3]},
        {"action": "CONTROL_MOVE", "stroke": 2, "control": "P0", "before": [3, 3], "after": [4, 4]},
    ]
    print(json.dumps(clean_edit_history(demo), ensure_ascii=False, indent=2))
