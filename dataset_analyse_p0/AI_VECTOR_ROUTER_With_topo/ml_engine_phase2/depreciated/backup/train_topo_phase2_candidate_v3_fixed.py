# -*- coding: utf-8 -*-
"""
train_topo_phase2_candidate_v3.py

位置建议：
    Babel_Simulater/dataset_analyse_p0/AI_VECTOR_ROUTER_With_topo/ml_engine_phase2/train_topo_phase2_candidate_v3.py

用途：
    Phase 2 Topology Micro-Refiner 的候选式训练脚本。

核心变化：
    不再让模型在所有控制点上硬猜 action。
    先由几何规则生成 near-miss candidates：
        - SNAP candidate: endpoint -> endpoint
        - T_ATTACH candidate: endpoint -> host stroke
        - CONTROL_MOVE candidate: control point move
    模型只判断这些候选是否应该执行，并预测 delta / host_t。
    这样比 point-level dense action classification 更适合 SNAP/T_ATTACH 稀有数据。

训练目标：
    1. relation head:
        stroke-pair desired topology: NONE / E2E / T / X
    2. candidate accept head:
        当前 candidate 是否应执行
    3. candidate delta head:
        before -> after 的移动量
    4. candidate host_t head:
        T_ATTACH 的 host_t

输入数据：
    ../annotations_topo/*_topo.json
    使用 strokes + topology_events + edit_history。
    edit_history 中的 before/after 会被反向还原成“操作前状态”。

输出：
    topo_phase2_candidate_v3_best.pth
    topo_phase2_candidate_v3_latest.pth
"""

import os
import json
import glob
import math
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset


# =============================================================================
# 0. Config
# =============================================================================

@dataclass
class Cfg:
    max_strokes: int = 30
    max_candidates: int = 64

    canvas_norm: float = 400.0
    width_norm: float = 20.0
    delta_norm: float = 24.0

    snap_radius: float = 12.0
    t_attach_radius: float = 12.0
    no_action_radius_min: float = 0.0

    neg_per_positive: int = 12
    final_noop_per_char: int = 1
    snap_repeat: int = 8
    t_attach_repeat: int = 16
    control_move_repeat: int = 2

    hidden_dim: int = 128
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.05

    batch_size: int = 256
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_ratio: float = 0.10
    seed: int = 42

    w_rel: float = 1.0
    w_rel_t: float = 1.5
    w_accept: float = 2.0
    w_delta: float = 2.0
    w_host_t: float = 2.0

    candidate_pos_weight_clip: float = 20.0

    best_name: str = "topo_phase2_candidate_v3_best.pth"
    latest_name: str = "topo_phase2_candidate_v3_latest.pth"


REL_MAP = {"NONE": 0, "E2E": 1, "T": 2, "X": 3}
REL_INV = {v: k for k, v in REL_MAP.items()}

CAND_TYPE = {"SNAP": 0, "T_ATTACH": 1, "CONTROL_MOVE": 2}
CAND_INV = {v: k for k, v in CAND_TYPE.items()}

IGNORE = -100

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOPO_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "annotations_topo"))
MODEL_SAVE_DIR = SCRIPT_DIR
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


# =============================================================================
# 1. helpers
# =============================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cubic(P: np.ndarray, t: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)
    if t.ndim == 1:
        t = t[:, None]
    mt = 1 - t
    return mt**3 * P[0] + 3 * mt**2 * t * P[1] + 3 * mt * t**2 * P[2] + t**3 * P[3]


def deriv(P: np.ndarray, t: float) -> np.ndarray:
    P = np.asarray(P, dtype=np.float32)
    mt = 1.0 - float(t)
    tt = float(t)
    return 3*mt**2*(P[1]-P[0]) + 6*mt*tt*(P[2]-P[1]) + 3*tt**2*(P[3]-P[2])


def ctrl_name_to_idx(name: Any) -> Optional[int]:
    return {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(str(name).upper())


def endpoint_name_to_idx(name: Any) -> Optional[int]:
    return {"P0": 0, "P3": 3}.get(str(name).upper())


def endpoint_t(pidx: int) -> float:
    return 0.0 if pidx == 0 else 1.0


def bezier_id_map(strokes: List[Dict[str, Any]]) -> Dict[int, int]:
    m = {}
    for i, s in enumerate(strokes):
        try:
            bid = int(s.get("bezier_id", i + 1))
        except Exception:
            bid = i + 1
        m[bid] = i
    return m


def valid_strokes(strokes: List[Dict[str, Any]]) -> bool:
    try:
        for s in strokes:
            P = np.asarray(s["mother_bezier"], dtype=np.float32)
            W = np.asarray(s["width_bezier"], dtype=np.float32).reshape(-1)
            if P.shape != (4, 2) or len(W) != 4:
                return False
        return True
    except Exception:
        return False


def deep_json_copy(x):
    return json.loads(json.dumps(x))


def closest_t_on_curve(point: np.ndarray, P: np.ndarray, n: int = 96) -> Tuple[float, np.ndarray, float]:
    ts = np.linspace(0, 1, n, dtype=np.float32)
    curve = cubic(P, ts)
    d = np.linalg.norm(curve - point[None, :], axis=1)
    idx = int(np.argmin(d))
    return float(ts[idx]), curve[idx].astype(np.float32), float(d[idx])


def point_token_features(P: np.ndarray, W: np.ndarray, stroke_idx: int, point_idx: int, cfg: Cfg) -> np.ndarray:
    P = np.asarray(P, dtype=np.float32)
    W = np.asarray(W, dtype=np.float32).reshape(4)
    pt = P[point_idx]
    start, end = P[0], P[3]
    center = P.mean(axis=0)
    chord = end - start
    chord_len = np.linalg.norm(chord) + 1e-6
    t0 = deriv(P, 0.0); t0 = t0 / (np.linalg.norm(t0) + 1e-6)
    t1 = deriv(P, 1.0); t1 = t1 / (np.linalg.norm(t1) + 1e-6)

    role = np.zeros(4, dtype=np.float32)
    role[point_idx] = 1.0
    endpoint_flag = 1.0 if point_idx in (0, 3) else 0.0

    feat = np.concatenate([
        pt / cfg.canvas_norm,                  # 2
        start / cfg.canvas_norm,               # 2
        end / cfg.canvas_norm,                 # 2
        center / cfg.canvas_norm,              # 2
        np.array([chord_len / cfg.canvas_norm], dtype=np.float32), # 1
        t0.astype(np.float32),                 # 2
        t1.astype(np.float32),                 # 2
        W / cfg.width_norm,                    # 4
        np.array([stroke_idx / max(1, cfg.max_strokes-1)], dtype=np.float32), # 1
        role,                                  # 4
        np.array([endpoint_flag], dtype=np.float32), # 1
    ]).astype(np.float32)
    return feat


def strokes_to_point_features(strokes: List[Dict[str, Any]], cfg: Cfg) -> np.ndarray:
    feats = []
    for si, s in enumerate(strokes):
        P = np.asarray(s["mother_bezier"], dtype=np.float32)
        W = np.asarray(s["width_bezier"], dtype=np.float32)
        for pi in range(4):
            feats.append(point_token_features(P, W, si, pi, cfg))
    return np.asarray(feats, dtype=np.float32)


# candidate geom dim = 22
def candidate_geom(strokes, q_si, q_pi, cand_type, cfg: Cfg, host_si=-1, host_pi=-1, host_t=0.0):
    qP = np.asarray(strokes[q_si]["mother_bezier"], dtype=np.float32)
    q = qP[q_pi].astype(np.float32)

    host_exists = 0.0
    host_pt = np.zeros(2, dtype=np.float32)
    host_role = np.zeros(4, dtype=np.float32)
    dist = 0.0

    if cand_type == "SNAP" and host_si >= 0 and host_pi in (0, 3):
        hP = np.asarray(strokes[host_si]["mother_bezier"], dtype=np.float32)
        host_pt = hP[host_pi].astype(np.float32)
        host_role[host_pi] = 1.0
        host_exists = 1.0
        host_t = endpoint_t(host_pi)
        dist = float(np.linalg.norm(host_pt - q))

    elif cand_type == "T_ATTACH" and host_si >= 0:
        hP = np.asarray(strokes[host_si]["mother_bezier"], dtype=np.float32)
        if host_t is None:
            host_t, host_pt, dist = closest_t_on_curve(q, hP)
        else:
            host_pt = cubic(hP, np.asarray([host_t], dtype=np.float32))[0].astype(np.float32)
            dist = float(np.linalg.norm(host_pt - q))
        host_exists = 1.0

    elif cand_type == "CONTROL_MOVE":
        host_pt = q.copy()
        host_exists = 0.0
        host_t = 0.0
        dist = 0.0

    delta = host_pt - q
    q_role = np.zeros(4, dtype=np.float32)
    q_role[q_pi] = 1.0
    q_endpoint = 1.0 if q_pi in (0, 3) else 0.0

    type_onehot = np.zeros(3, dtype=np.float32)
    type_onehot[CAND_TYPE[cand_type]] = 1.0

    feat = np.concatenate([
        type_onehot,                                      # 3
        q / cfg.canvas_norm,                              # 2
        host_pt / cfg.canvas_norm,                        # 2
        delta / cfg.delta_norm,                           # 2
        np.array([dist / cfg.delta_norm, float(host_t)], dtype=np.float32), # 2
        q_role,                                           # 4
        host_role,                                        # 4
        np.array([q_endpoint, host_exists, q_si / max(1, cfg.max_strokes-1)], dtype=np.float32), # 3
    ]).astype(np.float32)
    return feat


def reconstruct_pre_state(final_strokes: List[Dict[str, Any]], op: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    strokes = deep_json_copy(final_strokes)
    idmap = bezier_id_map(strokes)
    action = op.get("action")

    try:
        if action == "SNAP":
            sid = int(op["stroke"])
            hid = int(op["host_stroke"])
            qpi = endpoint_name_to_idx(op["endpoint"])
            hpi = endpoint_name_to_idx(op["host_endpoint"])
            if sid not in idmap or hid not in idmap or qpi is None or hpi is None:
                return strokes, None
            qsi, hsi = idmap[sid], idmap[hid]
            before = np.asarray(op["before"], dtype=np.float32)
            after = np.asarray(op["after"], dtype=np.float32)
            strokes[qsi]["mother_bezier"][qpi] = before.tolist()
            return strokes, {
                "action": "SNAP",
                "q_si": qsi, "q_pi": qpi,
                "host_si": hsi, "host_pi": hpi,
                "host_t": endpoint_t(hpi),
                "before": before, "after": after,
                "delta": after - before,
            }

        if action == "T_ATTACH":
            gid = int(op["guest"])
            hid = int(op["host"])
            qpi = endpoint_name_to_idx(op["guest_endpoint"])
            if gid not in idmap or hid not in idmap or qpi is None:
                return strokes, None
            qsi, hsi = idmap[gid], idmap[hid]
            before = np.asarray(op["before"], dtype=np.float32)
            after = np.asarray(op["after"], dtype=np.float32)
            ht = float(op.get("host_t", 0.5))
            strokes[qsi]["mother_bezier"][qpi] = before.tolist()
            return strokes, {
                "action": "T_ATTACH",
                "q_si": qsi, "q_pi": qpi,
                "host_si": hsi, "host_pi": -1,
                "host_t": ht,
                "before": before, "after": after,
                "delta": after - before,
            }

        if action == "CONTROL_MOVE":
            sid = int(op["stroke"])
            qpi = ctrl_name_to_idx(op["control"])
            if sid not in idmap or qpi is None:
                return strokes, None
            qsi = idmap[sid]
            before = np.asarray(op["before"], dtype=np.float32)
            after = np.asarray(op["after"], dtype=np.float32)
            strokes[qsi]["mother_bezier"][qpi] = before.tolist()
            return strokes, {
                "action": "CONTROL_MOVE",
                "q_si": qsi, "q_pi": qpi,
                "host_si": -1, "host_pi": -1,
                "host_t": 0.0,
                "before": before, "after": after,
                "delta": after - before,
            }
    except Exception:
        return strokes, None

    return strokes, None


# =============================================================================
# 2. Dataset
# =============================================================================

class CandidateDataset(Dataset):
    def __init__(self, data_dir: str, cfg: Cfg):
        self.cfg = cfg
        self.samples = []
        fps = sorted(glob.glob(os.path.join(data_dir, "*_topo.json")))
        if not fps:
            print(f"⚠️ no topo files: {data_dir}")

        for fp in fps:
            self._load_file(fp)

        print(f"✅ candidate phase2 samples={len(self.samples)} | dir={data_dir}")

    def _load_file(self, fp):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"load error {fp}: {e}")
            return

        for hex_key, bundle in data.items():
            if not isinstance(bundle, dict):
                continue
            strokes = bundle.get("strokes", [])
            if not isinstance(strokes, list) or not strokes or len(strokes) > self.cfg.max_strokes:
                continue
            if not valid_strokes(strokes):
                continue

            events = bundle.get("topology_events", [])

            # final no-op samples
            for _ in range(self.cfg.final_noop_per_char):
                self.samples.append(self._make_sample(strokes, events, None))

            hist = bundle.get("edit_history", [])
            if isinstance(hist, list):
                for op in hist:
                    if not isinstance(op, dict):
                        continue
                    pre, label = reconstruct_pre_state(strokes, op)
                    if label is None or not valid_strokes(pre):
                        continue
                    repeat = 1
                    if label["action"] == "SNAP":
                        repeat = self.cfg.snap_repeat
                    elif label["action"] == "T_ATTACH":
                        repeat = self.cfg.t_attach_repeat
                    elif label["action"] == "CONTROL_MOVE":
                        repeat = self.cfg.control_move_repeat

                    for _ in range(repeat):
                        self.samples.append(self._make_sample(pre, events, label))

    def _relation_labels(self, strokes, events):
        n = len(strokes)
        idmap = bezier_id_map(strokes)
        rel = torch.zeros((n, n), dtype=torch.long)
        rt = torch.zeros((n, n, 2), dtype=torch.float32)

        if not isinstance(events, list):
            return rel, rt

        for ev in events:
            if not isinstance(ev, dict):
                continue
            typ = ev.get("type")
            if typ not in REL_MAP:
                continue
            try:
                if typ in ("E2E", "X"):
                    aid, bid = int(ev["stroke_a"]), int(ev["stroke_b"])
                    if aid not in idmap or bid not in idmap:
                        continue
                    a, b = idmap[aid], idmap[bid]
                    ta, tb = float(ev.get("t_a", 0.0)), float(ev.get("t_b", 0.0))
                    rel[a, b] = rel[b, a] = REL_MAP[typ]
                    rt[a, b] = torch.tensor([ta, tb])
                    rt[b, a] = torch.tensor([tb, ta])
                elif typ == "T":
                    gid, hid = int(ev["guest"]), int(ev["host"])
                    if gid not in idmap or hid not in idmap:
                        continue
                    g, h = idmap[gid], idmap[hid]
                    gt, ht = float(ev.get("guest_t", 0.0)), float(ev.get("host_t", 0.0))
                    rel[g, h] = REL_MAP[typ]
                    rt[g, h] = torch.tensor([gt, ht])
            except Exception:
                continue
        return rel, rt

    def _positive_candidate(self, strokes, label):
        if label is None:
            return None
        qsi, qpi = label["q_si"], label["q_pi"]
        action = label["action"]
        if action == "SNAP":
            hsi, hpi = label["host_si"], label["host_pi"]
            hp_idx = hsi * 4 + hpi
            hs_idx = hsi
            geom = candidate_geom(strokes, qsi, qpi, "SNAP", self.cfg, hsi, hpi, label["host_t"])
        elif action == "T_ATTACH":
            hsi = label["host_si"]
            hp_idx = 0
            hs_idx = hsi
            geom = candidate_geom(strokes, qsi, qpi, "T_ATTACH", self.cfg, hsi, -1, label["host_t"])
        else:
            hp_idx = 0
            hs_idx = 0
            geom = candidate_geom(strokes, qsi, qpi, "CONTROL_MOVE", self.cfg)

        return {
            "q_idx": qsi * 4 + qpi,
            "hp_idx": hp_idx,
            "hs_idx": hs_idx,
            "cand_type": CAND_TYPE[action],
            "geom": geom,
            "accept": 1.0,
            "delta": label["delta"] / self.cfg.delta_norm,
            "delta_mask": 1.0,
            "host_t": float(label.get("host_t", 0.0)),
            "host_t_mask": 1.0 if action == "T_ATTACH" else 0.0,
        }

    def _negative_candidates(self, strokes, label):
        cfg = self.cfg
        n = len(strokes)
        cands = []
        pos_sig = None
        if label is not None:
            pos_sig = (label["action"], label["q_si"], label["q_pi"], label["host_si"], label["host_pi"])

        # SNAP candidates endpoint -> endpoint
        endpoints = [(si, pi) for si in range(n) for pi in (0, 3)]
        for qsi, qpi in endpoints:
            q = np.asarray(strokes[qsi]["mother_bezier"], dtype=np.float32)[qpi]
            local = []
            for hsi, hpi in endpoints:
                if hsi == qsi:
                    continue
                sig = ("SNAP", qsi, qpi, hsi, hpi)
                if sig == pos_sig:
                    continue
                hp = np.asarray(strokes[hsi]["mother_bezier"], dtype=np.float32)[hpi]
                d = float(np.linalg.norm(hp - q))
                if cfg.no_action_radius_min <= d <= cfg.snap_radius:
                    local.append((d, hsi, hpi))
            local.sort(key=lambda x: x[0])
            for _, hsi, hpi in local[:2]:
                cands.append({
                    "q_idx": qsi*4 + qpi,
                    "hp_idx": hsi*4 + hpi,
                    "hs_idx": hsi,
                    "cand_type": CAND_TYPE["SNAP"],
                    "geom": candidate_geom(strokes, qsi, qpi, "SNAP", cfg, hsi, hpi),
                    "accept": 0.0,
                    "delta": np.zeros(2, dtype=np.float32),
                    "delta_mask": 0.0,
                    "host_t": endpoint_t(hpi),
                    "host_t_mask": 0.0,
                })

        # T_ATTACH candidates endpoint -> stroke
        for qsi, qpi in endpoints:
            q = np.asarray(strokes[qsi]["mother_bezier"], dtype=np.float32)[qpi]
            local = []
            for hsi in range(n):
                if hsi == qsi:
                    continue
                sig = ("T_ATTACH", qsi, qpi, hsi, -1)
                if sig == pos_sig:
                    continue
                hP = np.asarray(strokes[hsi]["mother_bezier"], dtype=np.float32)
                ht, hp, d = closest_t_on_curve(q, hP)
                # endpoint-to-curve near miss; 排除太靠近 host 端点的点，端点问题让 SNAP 处理
                if 0.03 < ht < 0.97 and cfg.no_action_radius_min <= d <= cfg.t_attach_radius:
                    local.append((d, hsi, ht))
            local.sort(key=lambda x: x[0])
            for _, hsi, ht in local[:2]:
                cands.append({
                    "q_idx": qsi*4 + qpi,
                    "hp_idx": 0,
                    "hs_idx": hsi,
                    "cand_type": CAND_TYPE["T_ATTACH"],
                    "geom": candidate_geom(strokes, qsi, qpi, "T_ATTACH", cfg, hsi, -1, ht),
                    "accept": 0.0,
                    "delta": np.zeros(2, dtype=np.float32),
                    "delta_mask": 0.0,
                    "host_t": float(ht),
                    "host_t_mask": 0.0,
                })

        # CONTROL_MOVE negative candidates: random controls
        ctrl_points = [(si, pi) for si in range(n) for pi in range(4)]
        random.shuffle(ctrl_points)
        for qsi, qpi in ctrl_points[:min(12, len(ctrl_points))]:
            if pos_sig == ("CONTROL_MOVE", qsi, qpi, -1, -1):
                continue
            cands.append({
                "q_idx": qsi*4 + qpi,
                "hp_idx": 0,
                "hs_idx": 0,
                "cand_type": CAND_TYPE["CONTROL_MOVE"],
                "geom": candidate_geom(strokes, qsi, qpi, "CONTROL_MOVE", cfg),
                "accept": 0.0,
                "delta": np.zeros(2, dtype=np.float32),
                "delta_mask": 0.0,
                "host_t": 0.0,
                "host_t_mask": 0.0,
            })

        return cands

    def _make_sample(self, strokes, events, label):
        n = len(strokes)
        point_feats = torch.tensor(strokes_to_point_features(strokes, self.cfg), dtype=torch.float32)
        rel, rt = self._relation_labels(strokes, events)

        cands = []
        pos = self._positive_candidate(strokes, label)
        if pos is not None:
            cands.append(pos)
        negs = self._negative_candidates(strokes, label)
        random.shuffle(negs)
        if pos is not None:
            negs = negs[:self.cfg.neg_per_positive]
        else:
            negs = negs[:self.cfg.max_candidates]
        cands.extend(negs)

        # 如果没有任何 candidate，补一个无效 no-op candidate，后面用 mask 忽略
        if not cands:
            cands.append({
                "q_idx": 0, "hp_idx": 0, "hs_idx": 0,
                "cand_type": CAND_TYPE["CONTROL_MOVE"],
                "geom": np.zeros(22, dtype=np.float32),
                "accept": 0.0,
                "delta": np.zeros(2, dtype=np.float32),
                "delta_mask": 0.0,
                "host_t": 0.0,
                "host_t_mask": 0.0,
                "force_mask0": True,
            })

        cands = cands[:self.cfg.max_candidates]

        return {
            "point_features": point_feats,
            "stroke_count": n,
            "rel_labels": rel,
            "rel_t_labels": rt,
            "candidates": cands,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        cfg = self.cfg
        n = int(s["stroke_count"])
        max_s = cfg.max_strokes
        max_p = max_s * 4
        cur_p = n * 4
        pad_p = max_p - cur_p
        pad_s = max_s - n

        point_features = torch.nn.functional.pad(s["point_features"], (0, 0, 0, pad_p))
        point_mask = torch.zeros(max_p, dtype=torch.float32); point_mask[:cur_p] = 1.0
        stroke_mask = torch.zeros(max_s, dtype=torch.float32); stroke_mask[:n] = 1.0

        rel = torch.nn.functional.pad(s["rel_labels"], (0, pad_s, 0, pad_s))
        rt = torch.nn.functional.pad(s["rel_t_labels"], (0, 0, 0, pad_s, 0, pad_s))

        C = cfg.max_candidates
        q_idx = torch.zeros(C, dtype=torch.long)
        hp_idx = torch.zeros(C, dtype=torch.long)
        hs_idx = torch.zeros(C, dtype=torch.long)
        cand_type = torch.zeros(C, dtype=torch.long)
        cand_geom = torch.zeros(C, 22, dtype=torch.float32)
        cand_mask = torch.zeros(C, dtype=torch.float32)
        accept = torch.zeros(C, dtype=torch.float32)
        delta = torch.zeros(C, 2, dtype=torch.float32)
        delta_mask = torch.zeros(C, dtype=torch.float32)
        host_t = torch.zeros(C, dtype=torch.float32)
        host_t_mask = torch.zeros(C, dtype=torch.float32)

        for i, c in enumerate(s["candidates"][:C]):
            q_idx[i] = int(c["q_idx"])
            hp_idx[i] = int(c["hp_idx"])
            hs_idx[i] = int(c["hs_idx"])
            cand_type[i] = int(c["cand_type"])
            cand_geom[i] = torch.tensor(c["geom"], dtype=torch.float32)
            cand_mask[i] = 0.0 if c.get("force_mask0", False) else 1.0
            accept[i] = float(c["accept"])
            delta[i] = torch.tensor(c["delta"], dtype=torch.float32)
            delta_mask[i] = float(c["delta_mask"])
            host_t[i] = float(c["host_t"])
            host_t_mask[i] = float(c["host_t_mask"])

        return {
            "point_features": point_features,
            "point_mask": point_mask,
            "stroke_mask": stroke_mask,
            "rel_labels": rel,
            "rel_t_labels": rt,
            "q_idx": q_idx,
            "hp_idx": hp_idx,
            "hs_idx": hs_idx,
            "cand_type": cand_type,
            "cand_geom": cand_geom,
            "cand_mask": cand_mask,
            "accept": accept,
            "delta": delta,
            "delta_mask": delta_mask,
            "host_t": host_t,
            "host_t_mask": host_t_mask,
        }

    def counts(self):
        pos = neg = 0
        rel_counts = torch.zeros(4, dtype=torch.long)
        type_counts = torch.zeros(3, dtype=torch.long)
        for i in range(len(self)):
            item = self[i]
            m = item["cand_mask"] > 0
            pos += int(((item["accept"] > 0.5) & m).sum())
            neg += int(((item["accept"] <= 0.5) & m).sum())
            for k in range(3):
                type_counts[k] += int(((item["cand_type"] == k) & m).sum())
            rel = item["rel_labels"]
            sm = item["stroke_mask"] > 0
            pair = (sm.unsqueeze(1) & sm.unsqueeze(0))
            pair.fill_diagonal_(False)
            vals = rel[pair]
            for k in range(4):
                rel_counts[k] += int((vals == k).sum())
        return pos, neg, type_counts, rel_counts


# =============================================================================
# 3. Model
# =============================================================================

class CandidateTopoModel(nn.Module):
    def __init__(self, point_dim: int, geom_dim: int, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim

        self.point_embed = nn.Sequential(
            nn.Linear(point_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=cfg.num_heads,
            dim_feedforward=h * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)

        self.stroke_proj = nn.Sequential(nn.Linear(h, h), nn.LayerNorm(h), nn.GELU())

        self.type_emb = nn.Embedding(3, h)
        self.geom_mlp = nn.Sequential(nn.Linear(geom_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, h))

        self.cand_mlp = nn.Sequential(
            nn.Linear(h * 5, h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, h),
            nn.GELU(),
        )
        self.accept_head = nn.Linear(h, 1)
        self.delta_head = nn.Linear(h, 2)
        self.host_t_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1), nn.Sigmoid())

        self.rel_head = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, 4),
        )
        self.rel_t_head = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.GELU(),
            nn.Linear(h, 2),
            nn.Sigmoid(),
        )

    def _gather_tokens(self, tokens, idx):
        # tokens: [B, L, H], idx: [B, C]
        b, c = idx.shape
        h = tokens.shape[-1]
        idx_exp = idx.unsqueeze(-1).expand(b, c, h)
        return torch.gather(tokens, 1, idx_exp)

    def forward(self, batch):
        x = self.point_embed(batch["point_features"])
        pad = batch["point_mask"] == 0
        enc = self.encoder(x, src_key_padding_mask=pad)

        b, p, h = enc.shape
        s = self.cfg.max_strokes
        stroke = enc.reshape(b, s, 4, h).mean(dim=2)
        stroke = self.stroke_proj(stroke)

        # relation
        si = stroke.unsqueeze(2).expand(b, s, s, h)
        sj = stroke.unsqueeze(1).expand(b, s, s, h)
        pair = torch.cat([si, sj], dim=-1)
        rel_logits = self.rel_head(pair)
        rel_t = self.rel_t_head(pair)

        # candidates
        qtok = self._gather_tokens(enc, batch["q_idx"])
        hptok = self._gather_tokens(enc, batch["hp_idx"])
        hstok = self._gather_tokens(stroke, batch["hs_idx"])
        typ = self.type_emb(batch["cand_type"])
        geom = self.geom_mlp(batch["cand_geom"])

        cand = torch.cat([qtok, hptok, hstok, typ, geom], dim=-1)
        ch = self.cand_mlp(cand)
        accept_logit = self.accept_head(ch).squeeze(-1)
        delta = self.delta_head(ch)
        host_t = self.host_t_head(ch).squeeze(-1)

        return {
            "rel_logits": rel_logits,
            "rel_t": rel_t,
            "accept_logit": accept_logit,
            "delta": delta,
            "host_t": host_t,
        }


# =============================================================================
# 4. Loss / eval
# =============================================================================

def pair_mask(stroke_mask):
    b, s = stroke_mask.shape
    pm = (stroke_mask.unsqueeze(2) * stroke_mask.unsqueeze(1)) > 0
    eye = torch.eye(s, dtype=torch.bool, device=stroke_mask.device).unsqueeze(0)
    return pm & (~eye)


def rel_class_weights(counts):
    c = counts.float().clamp_min(1.0)
    total = c.sum()
    w = torch.sqrt(total / (4.0 * c))
    return torch.clamp(w, 0.10, 8.0)


def compute_loss(out, batch, cfg: Cfg, rel_ce, bce, huber):
    pm = pair_mask(batch["stroke_mask"])
    rel_lab = batch["rel_labels"].clone()
    rel_lab[~pm] = IGNORE
    loss_rel = rel_ce(out["rel_logits"].reshape(-1, 4), rel_lab.reshape(-1))

    rel_event = (batch["rel_labels"] > 0) & pm
    if rel_event.any():
        loss_rel_t = huber(out["rel_t"][rel_event], batch["rel_t_labels"][rel_event])
    else:
        loss_rel_t = torch.zeros((), device=out["rel_logits"].device)

    cm = batch["cand_mask"] > 0
    if cm.any():
        raw = bce(out["accept_logit"], batch["accept"])
        loss_accept = raw[cm].mean()
    else:
        loss_accept = torch.zeros((), device=out["rel_logits"].device)

    dm = (batch["delta_mask"] > 0) & cm
    if dm.any():
        loss_delta = huber(out["delta"][dm], batch["delta"][dm])
    else:
        loss_delta = torch.zeros((), device=out["rel_logits"].device)

    hm = (batch["host_t_mask"] > 0) & cm
    if hm.any():
        loss_host_t = huber(out["host_t"][hm], batch["host_t"][hm])
    else:
        loss_host_t = torch.zeros((), device=out["rel_logits"].device)

    loss = (
        cfg.w_rel * loss_rel
        + cfg.w_rel_t * loss_rel_t
        + cfg.w_accept * loss_accept
        + cfg.w_delta * loss_delta
        + cfg.w_host_t * loss_host_t
    )
    parts = {
        "loss": loss.detach(),
        "rel": loss_rel.detach(),
        "rel_t": loss_rel_t.detach(),
        "accept": loss_accept.detach(),
        "delta": loss_delta.detach(),
        "host_t": loss_host_t.detach(),
    }
    return loss, parts


@torch.no_grad()
def evaluate(model, loader, device, cfg, rel_ce, bce, huber):
    model.eval()
    sums = {k: 0.0 for k in ["loss", "rel", "rel_t", "accept", "delta", "host_t"]}
    batches = 0

    rel_conf = torch.zeros((4, 4), dtype=torch.long)
    tp = fp = fn = tn = 0
    delta_abs_sum = 0.0
    delta_count = 0
    ht_abs_sum = 0.0
    ht_count = 0

    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch)
        loss, parts = compute_loss(out, batch, cfg, rel_ce, bce, huber)
        for k in sums:
            sums[k] += float(parts[k].item())
        batches += 1

        # relation
        pm = pair_mask(batch["stroke_mask"])
        pred_rel = out["rel_logits"].argmax(dim=-1)
        for t, p in zip(batch["rel_labels"][pm].detach().cpu().tolist(), pred_rel[pm].detach().cpu().tolist()):
            rel_conf[int(t), int(p)] += 1

        # candidate accept
        cm = batch["cand_mask"] > 0
        prob = torch.sigmoid(out["accept_logit"])
        pred = (prob > 0.5) & cm
        true = (batch["accept"] > 0.5) & cm
        tp += int((pred & true).sum().item())
        fp += int((pred & (~true) & cm).sum().item())
        fn += int(((~pred) & true).sum().item())
        tn += int(((~pred) & (~true) & cm).sum().item())

        dm = (batch["delta_mask"] > 0) & cm
        if dm.any():
            delta_abs_sum += float(torch.abs(out["delta"][dm] - batch["delta"][dm]).mean(dim=-1).sum().item())
            delta_count += int(dm.sum().item())

        hm = (batch["host_t_mask"] > 0) & cm
        if hm.any():
            ht_abs_sum += float(torch.abs(out["host_t"][hm] - batch["host_t"][hm]).sum().item())
            ht_count += int(hm.sum().item())

    if batches == 0:
        return {"loss": math.inf}

    m = {k: sums[k] / batches for k in sums}

    def f1_cls(conf, cls):
        _tp = conf[cls, cls].item()
        _fp = conf[:, cls].sum().item() - _tp
        _fn = conf[cls, :].sum().item() - _tp
        pr = _tp / max(1, _tp + _fp)
        rc = _tp / max(1, _tp + _fn)
        return 2 * pr * rc / max(1e-8, pr + rc)

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    cand_f1 = 2 * prec * rec / max(1e-8, prec + rec)

    m.update({
        "rel_f1_e2e": f1_cls(rel_conf, REL_MAP["E2E"]),
        "rel_f1_t": f1_cls(rel_conf, REL_MAP["T"]),
        "rel_f1_x": f1_cls(rel_conf, REL_MAP["X"]),
        "cand_precision": prec,
        "cand_recall": rec,
        "cand_f1": cand_f1,
        "cand_tp": tp, "cand_fp": fp, "cand_fn": fn, "cand_tn": tn,
        "delta_abs": delta_abs_sum / max(1, delta_count),
        "host_t_abs": ht_abs_sum / max(1, ht_count),
    })
    model.train()
    return m


def print_metrics(epoch, train_loss, m, lr):
    print(
        f"Epoch {epoch:03d} | TrainLoss={train_loss:.4f} | ValLoss={m['loss']:.4f} | LR={lr:.2e}\n"
        f"  parts: rel={m['rel']:.3f} rel_t={m['rel_t']:.3f} accept={m['accept']:.3f} "
        f"delta={m['delta']:.3f} host_t={m['host_t']:.3f}\n"
        f"  relation F1 E2E/T/X={m['rel_f1_e2e']:.3f}/{m['rel_f1_t']:.3f}/{m['rel_f1_x']:.3f} | "
        f"candidate P/R/F1={m['cand_precision']:.3f}/{m['cand_recall']:.3f}/{m['cand_f1']:.3f} "
        f"(tp={m['cand_tp']} fp={m['cand_fp']} fn={m['cand_fn']}) | "
        f"delta_abs={m['delta_abs']:.4f} host_t_abs={m['host_t_abs']:.4f}"
    )


# =============================================================================
# 5. Train
# =============================================================================

def split_dataset(ds, cfg):
    n = len(ds)
    idxs = list(range(n))
    rng = random.Random(cfg.seed)
    rng.shuffle(idxs)
    if n < 10:
        return Subset(ds, idxs), Subset(ds, idxs)
    val_n = max(1, int(round(n * cfg.val_ratio)))
    return Subset(ds, idxs[val_n:]), Subset(ds, idxs[:val_n])


def train():
    cfg = Cfg()
    seed_everything(cfg.seed)
    device = get_device()
    print(f"🔥 device={device}")
    print(f"📁 topo data dir={TOPO_DATA_DIR}")

    ds = CandidateDataset(TOPO_DATA_DIR, cfg)
    if len(ds) == 0:
        print("❌ dataset empty.")
        return

    pos, neg, type_counts, rel_counts = ds.counts()
    print(f"📊 candidate accept: pos={pos}, neg={neg}, pos_rate={pos/max(1,pos+neg):.4f}")
    print("📊 candidate types:", {CAND_INV[i]: int(type_counts[i]) for i in range(3)})
    print("📊 relation labels:", {REL_INV[i]: int(rel_counts[i]) for i in range(4)})

    train_set, val_set = split_dataset(ds, cfg)
    print(f"📦 split train={len(train_set)} val={len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    sample0 = ds[0]
    point_dim = sample0["point_features"].shape[-1]
    geom_dim = sample0["cand_geom"].shape[-1]

    model = CandidateTopoModel(point_dim, geom_dim, cfg).to(device)

    rw = rel_class_weights(rel_counts).to(device)
    rel_ce = nn.CrossEntropyLoss(weight=rw, ignore_index=IGNORE)

    pos_weight_val = min(cfg.candidate_pos_weight_clip, max(1.0, neg / max(1, pos)))
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, dtype=torch.float32, device=device), reduction="none")
    huber = nn.SmoothL1Loss(reduction="mean")

    print("⚖️ relation weights:", [round(float(x), 3) for x in rw.detach().cpu()])
    print(f"⚖️ candidate BCE pos_weight={pos_weight_val:.3f}")

    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))

    best = math.inf
    best_path = os.path.join(MODEL_SAVE_DIR, cfg.best_name)
    latest_path = os.path.join(MODEL_SAVE_DIR, cfg.latest_name)

    print("🚀 start candidate Phase2 training...")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            out = model(batch)
            loss, parts = compute_loss(out, batch, cfg, rel_ce, bce, huber)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += float(loss.item())
            batches += 1

        sched.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            train_loss = total / max(1, batches)
            m = evaluate(model, val_loader, device, cfg, rel_ce, bce, huber)
            lr = sched.get_last_lr()[0]
            print_metrics(epoch, train_loss, m, lr)

            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "point_dim": point_dim,
                "geom_dim": geom_dim,
                "rel_map": REL_MAP,
                "cand_type": CAND_TYPE,
                "epoch": epoch,
                "val_loss": m["loss"],
                "rel_counts": rel_counts.tolist(),
                "candidate_pos": int(pos),
                "candidate_neg": int(neg),
                "candidate_pos_weight": float(pos_weight_val),
            }
            torch.save(ckpt, latest_path)
            if m["loss"] < best:
                best = m["loss"]
                torch.save(ckpt, best_path)
                print(f"  ✅ best saved: {best_path}")

    print(f"🎉 done. best={best:.4f}")
    print(f"best:   {best_path}")
    print(f"latest: {latest_path}")


if __name__ == "__main__":
    train()
