# -*- coding: utf-8 -*-
"""
phase2_edit_history_cleaner.py

Phase 2 人工标注历史清理工具。

用途：
    不修改原始 annotations_topo/*.json，只在读取 edit_history 时做语义清理。

核心规则：
    1. 第一次 SNAP / T_ATTACH 保留，表示“建立拓扑连接”。
    2. 后续同一连接对象上的重复 SNAP / T_ATTACH 视为“已连接结构的位置调整”，
       先转换为 CONTROL_MOVE。
    3. 在完成重复连接转换之后，全局合并同一 stroke/control 的多步 CONTROL_MOVE，
       即使这些移动中间穿插了其他控制点移动。

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


def _control_move_key(op: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """CONTROL_MOVE 的全局合并 key：同一 stroke + control。"""
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
        return ("T_ATTACH", str(guest), str(guest_endpoint), str(host))

    if action == "SNAP":
        stroke = op.get("stroke")
        endpoint = op.get("endpoint")
        host_stroke = op.get("host_stroke")
        host_endpoint = op.get("host_endpoint")
        if stroke is None or endpoint is None or host_stroke is None or host_endpoint is None:
            return None
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
        return {
            "action": "CONTROL_MOVE",
            "stroke": int(guest) if str(guest).isdigit() else guest,
            "control": str(guest_endpoint),
            "before": before,
            "after": after,
            "source_action": "T_ATTACH",
            "source_host": op.get("host"),
            "source_host_t": op.get("host_t"),
        }

    if action == "SNAP":
        stroke = op.get("stroke")
        endpoint = op.get("endpoint")
        if stroke is None or endpoint is None:
            return None
        return {
            "action": "CONTROL_MOVE",
            "stroke": int(stroke) if str(stroke).isdigit() else stroke,
            "control": str(endpoint),
            "before": before,
            "after": after,
            "source_action": "SNAP",
            "source_host_stroke": op.get("host_stroke"),
            "source_host_endpoint": op.get("host_endpoint"),
        }

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


def _source_record(op: Dict[str, Any]) -> Dict[str, Any]:
    """为 merged_sources 生成一条可追踪来源记录。"""
    return {
        "action": op.get("action"),
        "source_action": op.get("source_action"),
        "stroke": op.get("stroke"),
        "control": op.get("control"),
        "before": op.get("before"),
        "after": op.get("after"),
        "source_host": op.get("source_host"),
        "source_host_t": op.get("source_host_t"),
        "source_host_stroke": op.get("source_host_stroke"),
        "source_host_endpoint": op.get("source_host_endpoint"),
    }


def _pass1_convert_repeated_relations(edit_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """第一阶段：保留首次 SNAP/T_ATTACH，重复连接转 CONTROL_MOVE。"""
    converted: List[Dict[str, Any]] = []
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
                    converted.append(move_op)
                continue

            if rel_key is not None:
                seen_relations.add(rel_key)
            converted.append(op)
            continue

        converted.append(op)

    return converted


def _pass2_merge_control_moves(ops: List[Dict[str, Any]], *, merge_moves: bool) -> List[Dict[str, Any]]:
    """
    第二阶段：全局合并同一 stroke/control 的 CONTROL_MOVE。

    注意：这里不要求同点移动相邻。输出顺序以该点第一次出现的位置为准。
    """
    if not merge_moves:
        return [_deepcopy_jsonable(op) for op in ops]

    output: List[Dict[str, Any]] = []
    move_index: Dict[Tuple[str, str], int] = {}

    for op in ops:
        if not isinstance(op, dict):
            continue

        if op.get("action") != "CONTROL_MOVE":
            output.append(_deepcopy_jsonable(op))
            continue

        move_op = _normalize_original_op(op)
        key = _control_move_key(move_op)
        if key is None:
            output.append(move_op)
            continue

        if key not in move_index:
            move_index[key] = len(output)
            move_op.setdefault("merged_sources", [])
            output.append(move_op)
            continue

        base = output[move_index[key]]
        base_sources = base.setdefault("merged_sources", [])
        base_sources.append(_source_record(move_op))
        # before 保留第一次，after 更新为最后一次。
        if move_op.get("after") is not None:
            base["after"] = move_op.get("after")

        # 如果后续移动来自重复 T/SNAP，保留最近的来源字段，便于排查。
        for field in (
            "source_action", "source_host", "source_host_t",
            "source_host_stroke", "source_host_endpoint",
        ):
            if field in move_op:
                base[field] = move_op.get(field)

    return output


def clean_edit_history(edit_history: List[Dict[str, Any]], *, merge_moves: bool = True) -> List[Dict[str, Any]]:
    """
    清理 Phase 2 edit_history。

    清理顺序：
        1. 重复 SNAP/T_ATTACH -> CONTROL_MOVE。
        2. 全局合并同一 stroke/control 的多步 CONTROL_MOVE。

    参数：
        edit_history: 原始 edit_history 列表。
        merge_moves: 是否合并同一控制点的 CONTROL_MOVE。

    返回：
        cleaned edit_history。输入对象不会被修改。
    """
    if not isinstance(edit_history, list):
        return []

    converted = _pass1_convert_repeated_relations(edit_history)
    return _pass2_merge_control_moves(converted, merge_moves=merge_moves)


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
        {"action": "CONTROL_MOVE", "stroke": 3, "control": "P1", "before": [8, 8], "after": [9, 9]},
        {"action": "T_ATTACH", "guest": 2, "guest_endpoint": "P0", "host": 1, "host_t": 0.5, "before": [2, 2], "after": [3, 3]},
        {"action": "CONTROL_MOVE", "stroke": 2, "control": "P0", "before": [3, 3], "after": [4, 4]},
    ]
    print(json.dumps(clean_edit_history(demo), ensure_ascii=False, indent=2))
