# -*- coding: utf-8 -*-
"""
train_topo_phase2_advanced.py

位置建议：
    Babel_Simulater/dataset_analyse_p0/AI_VECTOR_ROUTER_With_topo/ml_engine_phase2/train_topo_phase2_advanced.py

目标：
    Phase 2 Topo 微调模型，面向 Clean Good 的第二阶段：
      - 预测最终拓扑关系 E2E / T / X
      - 预测端点级动作 NONE / SNAP / T_ATTACH / CONTROL_MOVE
      - 预测 SNAP/T_ATTACH 的 host 指针
      - 预测 T_ATTACH 的 host_t
      - 预测控制点或端点的 delta 微调量

核心思想：
    不只训练 pair topology classifier。
    利用 annotations_topo/*_topo.json 里的 edit_history，把人工操作反向还原成“操作前状态”，
    让模型学习：
        pre-adjust strokes -> 人工会做什么吸附 / 微调动作

    例如最终状态已经吸附好，edit_history 记录:
        before=[45, 123], after=[46,124]
    则训练时把该控制点临时改回 before，作为输入；标签是把它移动到 after。

输出：
    topo_phase2_advanced_best.pth
    topo_phase2_advanced_latest.pth
"""

import os
import json
import glob
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional

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
    canvas_norm: float = 400.0
    width_norm: float = 20.0
    delta_norm: float = 24.0

    hidden_dim: int = 128
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.05

    batch_size: int = 128
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_ratio: float = 0.10
    seed: int = 42

    # loss 权重
    w_rel: float = 1.0
    w_action: float = 1.5
    w_ep_ptr: float = 1.0
    w_host_ptr: float = 1.0
    w_t: float = 2.0
    w_delta: float = 2.0

    # 从 edit_history 生成训练样本时，保留多少 no-op final 状态样本
    noop_per_char: int = 1

    best_name: str = "topo_phase2_advanced_best.pth"
    latest_name: str = "topo_phase2_advanced_latest.pth"


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOPO_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "annotations_topo"))
MODEL_SAVE_DIR = SCRIPT_DIR
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

REL_MAP = {"NONE": 0, "E2E": 1, "T": 2, "X": 3}
REL_INV = {v: k for k, v in REL_MAP.items()}

ACT_MAP = {"NONE": 0, "SNAP": 1, "T_ATTACH": 2, "CONTROL_MOVE": 3}
ACT_INV = {v: k for k, v in ACT_MAP.items()}

IGNORE = -100


# =============================================================================
# 1. Geometry helpers
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


def point_features(P: np.ndarray, W: np.ndarray, stroke_idx: int, point_idx: int, cfg: Cfg) -> np.ndarray:
    """
    每个控制点一个 token。
    token feature 包含：
      - 当前控制点 xy
      - 四控制点全局几何摘要：start/end/center/tangent/length
      - 当前点角色 one-hot P0/P1/P2/P3
      - endpoint flag
      - stroke index normalized
      - width summary
    """
    P = np.asarray(P, dtype=np.float32)
    W = np.asarray(W, dtype=np.float32).reshape(4)
    pt = P[point_idx]

    start, end = P[0], P[3]
    center = P.mean(axis=0)
    chord = end - start
    chord_len = np.linalg.norm(chord) + 1e-6
    tangent0 = deriv(P, 0.0)
    tangent1 = deriv(P, 1.0)
    t0 = tangent0 / (np.linalg.norm(tangent0) + 1e-6)
    t1 = tangent1 / (np.linalg.norm(tangent1) + 1e-6)

    role = np.zeros(4, dtype=np.float32)
    role[point_idx] = 1.0
    endpoint = 1.0 if point_idx in (0, 3) else 0.0

    feat = np.concatenate([
        pt / cfg.canvas_norm,                 # 2
        start / cfg.canvas_norm,              # 2
        end / cfg.canvas_norm,                # 2
        center / cfg.canvas_norm,             # 2
        np.array([chord_len / cfg.canvas_norm], dtype=np.float32),  # 1
        t0.astype(np.float32),                # 2
        t1.astype(np.float32),                # 2
        W / cfg.width_norm,                   # 4
        np.array([float(stroke_idx) / max(1, cfg.max_strokes - 1)], dtype=np.float32), # 1
        role,                                 # 4
        np.array([endpoint], dtype=np.float32) # 1
    ]).astype(np.float32)
    return feat


def strokes_to_point_tokens(strokes: List[Dict[str, Any]], cfg: Cfg) -> np.ndarray:
    feats = []
    for si, s in enumerate(strokes):
        P = np.asarray(s["mother_bezier"], dtype=np.float32)
        W = np.asarray(s["width_bezier"], dtype=np.float32).reshape(4)
        for pi in range(4):
            feats.append(point_features(P, W, si, pi, cfg))
    return np.asarray(feats, dtype=np.float32)


def stroke_features_from_point_tokens(point_tokens: torch.Tensor, max_strokes: int):
    """
    point_tokens: [B, S*4, F]
    reshape 后每条 stroke 的 4 个控制点 token mean pool。
    """
    b, _, f = point_tokens.shape
    return point_tokens.reshape(b, max_strokes, 4, f).mean(dim=2)


def bezier_id_map(strokes: List[Dict[str, Any]]) -> Dict[int, int]:
    m = {}
    for idx, s in enumerate(strokes):
        try:
            bid = int(s.get("bezier_id", idx + 1))
        except Exception:
            bid = idx + 1
        m[bid] = idx
    return m


def ctrl_name_to_idx(name: str) -> Optional[int]:
    table = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    return table.get(str(name).upper())


def endpoint_name_to_idx(name: str) -> Optional[int]:
    table = {"P0": 0, "P3": 3}
    return table.get(str(name).upper())


def make_pre_state_from_op(final_strokes: List[Dict[str, Any]], op: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    根据 edit_history 反向构造操作前状态。
    返回:
        pre_strokes, action_label
    action_label 内部全部使用 local index。
    """
    strokes = json.loads(json.dumps(final_strokes))  # cheap deep copy for json-compatible data
    idmap = bezier_id_map(strokes)

    action = op.get("action")
    if action == "SNAP":
        stroke_id = int(op.get("stroke"))
        host_id = int(op.get("host_stroke"))
        ep_idx = endpoint_name_to_idx(op.get("endpoint"))
        host_ep_idx = endpoint_name_to_idx(op.get("host_endpoint"))
        if stroke_id not in idmap or host_id not in idmap or ep_idx is None or host_ep_idx is None:
            return strokes, None

        si = idmap[stroke_id]
        hi = idmap[host_id]
        before = np.asarray(op.get("before"), dtype=np.float32)
        after = np.asarray(op.get("after"), dtype=np.float32)

        strokes[si]["mother_bezier"][ep_idx] = before.tolist()
        label = {
            "action": "SNAP",
            "stroke_idx": si,
            "point_idx": ep_idx,
            "host_stroke_idx": hi,
            "host_point_idx": host_ep_idx,
            "before": before,
            "after": after,
            "delta": after - before,
            "host_t": 0.0 if host_ep_idx == 0 else 1.0,
        }
        return strokes, label

    if action == "T_ATTACH":
        guest_id = int(op.get("guest"))
        host_id = int(op.get("host"))
        ep_idx = endpoint_name_to_idx(op.get("guest_endpoint"))
        if guest_id not in idmap or host_id not in idmap or ep_idx is None:
            return strokes, None

        gi = idmap[guest_id]
        hi = idmap[host_id]
        before = np.asarray(op.get("before"), dtype=np.float32)
        after = np.asarray(op.get("after"), dtype=np.float32)
        host_t = float(op.get("host_t", 0.5))

        strokes[gi]["mother_bezier"][ep_idx] = before.tolist()
        label = {
            "action": "T_ATTACH",
            "stroke_idx": gi,
            "point_idx": ep_idx,
            "host_stroke_idx": hi,
            "host_point_idx": -1,
            "before": before,
            "after": after,
            "delta": after - before,
            "host_t": host_t,
        }
        return strokes, label

    if action == "CONTROL_MOVE":
        stroke_id = int(op.get("stroke"))
        pidx = ctrl_name_to_idx(op.get("control"))
        if stroke_id not in idmap or pidx is None:
            return strokes, None

        si = idmap[stroke_id]
        before = np.asarray(op.get("before"), dtype=np.float32)
        after = np.asarray(op.get("after"), dtype=np.float32)

        strokes[si]["mother_bezier"][pidx] = before.tolist()
        label = {
            "action": "CONTROL_MOVE",
            "stroke_idx": si,
            "point_idx": pidx,
            "host_stroke_idx": -1,
            "host_point_idx": -1,
            "before": before,
            "after": after,
            "delta": after - before,
            "host_t": 0.0,
        }
        return strokes, label

    return strokes, None


# =============================================================================
# 2. Dataset
# =============================================================================

class Phase2AdvancedDataset(Dataset):
    def __init__(self, data_dir: str, cfg: Cfg):
        self.cfg = cfg
        self.samples = []

        fps = sorted(glob.glob(os.path.join(data_dir, "*_topo.json")))
        if not fps:
            print(f"⚠️ no topo files found: {data_dir}")

        for fp in fps:
            self._load_file(fp)

        print(f"✅ advanced phase2 samples={len(self.samples)} | dir={data_dir}")

    def _load_file(self, fp: str):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"load error {fp}: {e}")
            return

        font = os.path.basename(fp).replace("_topo.json", "")
        for hex_key, bundle in data.items():
            if not isinstance(bundle, dict):
                continue

            strokes = bundle.get("strokes", [])
            if not isinstance(strokes, list) or not strokes:
                continue
            if len(strokes) > self.cfg.max_strokes:
                continue
            if not self._valid_strokes(strokes):
                continue

            # 1) final no-op state：训练模型识别最终状态没有需要微调的动作，同时学习拓扑关系。
            for _ in range(self.cfg.noop_per_char):
                self.samples.append(self._make_sample(
                    font, hex_key, strokes, bundle.get("topology_events", []), None
                ))

            # 2) 从 edit_history 反向构造 pre-adjust state。
            edit_history = bundle.get("edit_history", [])
            if isinstance(edit_history, list):
                for op in edit_history:
                    if not isinstance(op, dict):
                        continue
                    pre_strokes, label = make_pre_state_from_op(strokes, op)
                    if label is None:
                        continue
                    if not self._valid_strokes(pre_strokes):
                        continue
                    self.samples.append(self._make_sample(
                        font, hex_key, pre_strokes, bundle.get("topology_events", []), label
                    ))

    def _valid_strokes(self, strokes):
        try:
            for s in strokes:
                P = np.asarray(s["mother_bezier"], dtype=np.float32)
                W = np.asarray(s["width_bezier"], dtype=np.float32).reshape(-1)
                if P.shape != (4, 2) or len(W) != 4:
                    return False
            return True
        except Exception:
            return False

    def _extract_relation_labels(self, strokes, events):
        n = len(strokes)
        idmap = bezier_id_map(strokes)
        rel = torch.zeros((n, n), dtype=torch.long)
        t = torch.zeros((n, n, 2), dtype=torch.float32)

        if not isinstance(events, list):
            return rel, t

        for ev in events:
            if not isinstance(ev, dict):
                continue
            typ = ev.get("type")
            if typ not in REL_MAP:
                continue
            k = REL_MAP[typ]
            try:
                if typ in ("E2E", "X"):
                    a_id, b_id = int(ev["stroke_a"]), int(ev["stroke_b"])
                    if a_id not in idmap or b_id not in idmap:
                        continue
                    a, b = idmap[a_id], idmap[b_id]
                    ta, tb = float(ev.get("t_a", 0.0)), float(ev.get("t_b", 0.0))
                    rel[a, b] = rel[b, a] = k
                    t[a, b] = torch.tensor([ta, tb], dtype=torch.float32)
                    t[b, a] = torch.tensor([tb, ta], dtype=torch.float32)
                elif typ == "T":
                    gid, hid = int(ev["guest"]), int(ev["host"])
                    if gid not in idmap or hid not in idmap:
                        continue
                    g, h = idmap[gid], idmap[hid]
                    gt, ht = float(ev.get("guest_t", 0.0)), float(ev.get("host_t", 0.0))
                    rel[g, h] = k
                    t[g, h] = torch.tensor([gt, ht], dtype=torch.float32)
            except Exception:
                continue

        return rel, t

    def _make_sample(self, font, hex_key, strokes, events, action_label):
        n = len(strokes)
        pt = strokes_to_point_tokens(strokes, self.cfg)  # [n*4, F]
        rel, rel_t = self._extract_relation_labels(strokes, events)

        # point action labels: [max_strokes*4]
        point_count = self.cfg.max_strokes * 4
        action_labels = torch.full((n * 4,), ACT_MAP["NONE"], dtype=torch.long)
        point_delta = torch.zeros((n * 4, 2), dtype=torch.float32)
        point_delta_mask = torch.zeros((n * 4,), dtype=torch.float32)

        ep_ptr_labels = torch.full((n * 4,), IGNORE, dtype=torch.long)       # SNAP: target point token idx
        host_ptr_labels = torch.full((n * 4,), IGNORE, dtype=torch.long)     # T_ATTACH: target stroke idx
        host_t_labels = torch.zeros((n * 4,), dtype=torch.float32)
        host_t_mask = torch.zeros((n * 4,), dtype=torch.float32)

        if action_label is not None:
            si = action_label["stroke_idx"]
            pi = action_label["point_idx"]
            tok = si * 4 + pi
            act = action_label["action"]
            action_labels[tok] = ACT_MAP[act]
            point_delta[tok] = torch.tensor(action_label["delta"] / self.cfg.delta_norm, dtype=torch.float32)
            point_delta_mask[tok] = 1.0

            if act == "SNAP":
                ht = action_label["host_stroke_idx"] * 4 + action_label["host_point_idx"]
                ep_ptr_labels[tok] = int(ht)
            elif act == "T_ATTACH":
                host_ptr_labels[tok] = int(action_label["host_stroke_idx"])
                host_t_labels[tok] = float(action_label["host_t"])
                host_t_mask[tok] = 1.0

        return {
            "font": font,
            "hex_key": hex_key,
            "point_features": torch.tensor(pt, dtype=torch.float32),
            "stroke_count": n,
            "rel_labels": rel,
            "rel_t_labels": rel_t,
            "action_labels": action_labels,
            "point_delta": point_delta,
            "point_delta_mask": point_delta_mask,
            "ep_ptr_labels": ep_ptr_labels,
            "host_ptr_labels": host_ptr_labels,
            "host_t_labels": host_t_labels,
            "host_t_mask": host_t_mask,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        n = int(s["stroke_count"])
        max_s = self.cfg.max_strokes
        max_p = max_s * 4
        cur_p = n * 4
        pad_p = max_p - cur_p
        pad_s = max_s - n

        point_features = torch.nn.functional.pad(s["point_features"], (0, 0, 0, pad_p))
        point_mask = torch.zeros(max_p, dtype=torch.float32)
        point_mask[:cur_p] = 1.0

        stroke_mask = torch.zeros(max_s, dtype=torch.float32)
        stroke_mask[:n] = 1.0

        rel = torch.nn.functional.pad(s["rel_labels"], (0, pad_s, 0, pad_s))
        rel_t = torch.nn.functional.pad(s["rel_t_labels"], (0, 0, 0, pad_s, 0, pad_s))

        action_labels = torch.nn.functional.pad(s["action_labels"], (0, pad_p), value=IGNORE)
        # padding point action 不参与 action loss
        action_labels[cur_p:] = IGNORE

        point_delta = torch.nn.functional.pad(s["point_delta"], (0, 0, 0, pad_p))
        point_delta_mask = torch.nn.functional.pad(s["point_delta_mask"], (0, pad_p))

        ep_ptr_labels = torch.nn.functional.pad(s["ep_ptr_labels"], (0, pad_p), value=IGNORE)
        host_ptr_labels = torch.nn.functional.pad(s["host_ptr_labels"], (0, pad_p), value=IGNORE)
        host_t_labels = torch.nn.functional.pad(s["host_t_labels"], (0, pad_p))
        host_t_mask = torch.nn.functional.pad(s["host_t_mask"], (0, pad_p))

        return {
            "point_features": point_features,
            "point_mask": point_mask,
            "stroke_mask": stroke_mask,
            "rel_labels": rel,
            "rel_t_labels": rel_t,
            "action_labels": action_labels,
            "point_delta": point_delta,
            "point_delta_mask": point_delta_mask,
            "ep_ptr_labels": ep_ptr_labels,
            "host_ptr_labels": host_ptr_labels,
            "host_t_labels": host_t_labels,
            "host_t_mask": host_t_mask,
        }

    def action_counts(self):
        c = torch.zeros(4, dtype=torch.long)
        for s in self.samples:
            labels = s["action_labels"]
            for k in range(4):
                c[k] += int((labels == k).sum())
        return c

    def rel_counts(self):
        c = torch.zeros(4, dtype=torch.long)
        for s in self.samples:
            rel = s["rel_labels"]
            n = rel.shape[0]
            mask = torch.ones((n, n), dtype=torch.bool)
            mask.fill_diagonal_(False)
            vals = rel[mask]
            for k in range(4):
                c[k] += int((vals == k).sum())
        return c


# =============================================================================
# 3. Model: control-point Transformer + stroke-pair graph heads
# =============================================================================

class TopoPhase2AdvancedModel(nn.Module):
    def __init__(self, point_input_dim: int, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim

        self.point_embed = nn.Sequential(
            nn.Linear(point_input_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=cfg.num_heads,
            dim_feedforward=h * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)

        # point-level action
        self.action_head = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, 4),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, 2),
        )

        # pointer heads
        self.ep_q = nn.Linear(h, h)
        self.ep_k = nn.Linear(h, h)
        self.host_q = nn.Linear(h, h)
        self.host_k = nn.Linear(h, h)

        self.host_t_head = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.GELU(),
            nn.Linear(h, 1),
            nn.Sigmoid(),
        )

        # relation head over stroke tokens
        self.stroke_proj = nn.Sequential(
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
        )
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

    def forward(self, point_features, point_mask, stroke_mask):
        """
        point_features: [B, max_strokes*4, F]
        point_mask:     [B, max_strokes*4]
        stroke_mask:    [B, max_strokes]
        """
        x = self.point_embed(point_features)
        pad_mask = (point_mask == 0)
        enc = self.encoder(x, src_key_padding_mask=pad_mask)

        b, p, h = enc.shape
        s = self.cfg.max_strokes

        action_logits = self.action_head(enc)       # [B, P, 4]
        delta_pred = self.delta_head(enc)           # [B, P, 2]

        # endpoint pointer: point -> point
        q = self.ep_q(enc)
        k = self.ep_k(enc)
        ep_ptr_logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(h)  # [B, P, P]
        ep_ptr_logits = ep_ptr_logits.masked_fill(point_mask.unsqueeze(1) == 0, -1e9)

        # stroke tokens from 4 point tokens
        stroke_tokens = enc.reshape(b, s, 4, h).mean(dim=2)
        stroke_tokens = self.stroke_proj(stroke_tokens)

        # host pointer: point -> stroke
        hq = self.host_q(enc)
        hk = self.host_k(stroke_tokens)
        host_ptr_logits = torch.matmul(hq, hk.transpose(1, 2)) / math.sqrt(h)  # [B, P, S]
        host_ptr_logits = host_ptr_logits.masked_fill(stroke_mask.unsqueeze(1) == 0, -1e9)

        # host_t for every point-stroke pair
        p_tok = enc.unsqueeze(2).expand(b, p, s, h)
        s_tok = stroke_tokens.unsqueeze(1).expand(b, p, s, h)
        point_stroke_pair = torch.cat([p_tok, s_tok], dim=-1)
        host_t_pred = self.host_t_head(point_stroke_pair).squeeze(-1)  # [B, P, S]

        # relation over stroke pairs
        si = stroke_tokens.unsqueeze(2).expand(b, s, s, h)
        sj = stroke_tokens.unsqueeze(1).expand(b, s, s, h)
        pair = torch.cat([si, sj], dim=-1)
        rel_logits = self.rel_head(pair)       # [B, S, S, 4]
        rel_t_pred = self.rel_t_head(pair)     # [B, S, S, 2]

        return {
            "action_logits": action_logits,
            "delta_pred": delta_pred,
            "ep_ptr_logits": ep_ptr_logits,
            "host_ptr_logits": host_ptr_logits,
            "host_t_pred": host_t_pred,
            "rel_logits": rel_logits,
            "rel_t_pred": rel_t_pred,
        }


# =============================================================================
# 4. Loss / Metrics
# =============================================================================

def pair_mask_from_stroke_mask(stroke_mask):
    b, s = stroke_mask.shape
    pair = (stroke_mask.unsqueeze(2) * stroke_mask.unsqueeze(1)) > 0
    eye = torch.eye(s, dtype=torch.bool, device=stroke_mask.device).unsqueeze(0)
    return pair & (~eye)


def class_weights(counts, clip_min=0.10, clip_max=8.0):
    c = counts.float().clamp_min(1.0)
    total = c.sum()
    w = torch.sqrt(total / (len(c) * c))
    return torch.clamp(w, clip_min, clip_max)


def compute_loss(out, batch, cfg: Cfg, criteria):
    ce_action, ce_rel, ce_plain, huber = criteria

    action_logits = out["action_logits"]
    rel_logits = out["rel_logits"]

    action_labels = batch["action_labels"]
    rel_labels = batch["rel_labels"]

    # action CE
    loss_action = ce_action(action_logits.reshape(-1, 4), action_labels.reshape(-1))

    # relation CE
    active_pair = pair_mask_from_stroke_mask(batch["stroke_mask"])
    rel_lab = rel_labels.clone()
    rel_lab[~active_pair] = IGNORE
    loss_rel = ce_rel(rel_logits.reshape(-1, 4), rel_lab.reshape(-1))

    # endpoint pointer CE: only SNAP labels are non-ignore
    ep_labels = batch["ep_ptr_labels"]
    valid_ep = ep_labels != IGNORE
    if valid_ep.any():
        loss_ep_ptr = ce_plain(out["ep_ptr_logits"][valid_ep], ep_labels[valid_ep])
    else:
        loss_ep_ptr = torch.zeros((), device=action_logits.device)

    # host pointer CE: only T_ATTACH labels are non-ignore
    host_labels = batch["host_ptr_labels"]
    valid_host = host_labels != IGNORE
    if valid_host.any():
        loss_host_ptr = ce_plain(out["host_ptr_logits"][valid_host], host_labels[valid_host])
    else:
        loss_host_ptr = torch.zeros((), device=action_logits.device)

    # host_t regression: only T_ATTACH
    ht_mask = batch["host_t_mask"] > 0
    if ht_mask.any():
        # gather predicted t at target host index
        hidx = host_labels.clamp_min(0)
        pred_t = out["host_t_pred"].gather(2, hidx.unsqueeze(-1)).squeeze(-1)
        loss_host_t = huber(pred_t[ht_mask], batch["host_t_labels"][ht_mask])
    else:
        loss_host_t = torch.zeros((), device=action_logits.device)

    # delta regression: SNAP / T_ATTACH / CONTROL_MOVE
    dm = batch["point_delta_mask"] > 0
    if dm.any():
        loss_delta = huber(out["delta_pred"][dm], batch["point_delta"][dm])
    else:
        loss_delta = torch.zeros((), device=action_logits.device)

    # relation t regression: only rel > 0
    rel_event = (batch["rel_labels"] > 0) & active_pair
    if rel_event.any():
        loss_rel_t = huber(out["rel_t_pred"][rel_event], batch["rel_t_labels"][rel_event])
    else:
        loss_rel_t = torch.zeros((), device=action_logits.device)

    loss = (
        cfg.w_action * loss_action
        + cfg.w_rel * loss_rel
        + cfg.w_ep_ptr * loss_ep_ptr
        + cfg.w_host_ptr * loss_host_ptr
        + cfg.w_t * (loss_host_t + loss_rel_t)
        + cfg.w_delta * loss_delta
    )

    parts = {
        "loss": loss.detach(),
        "action": loss_action.detach(),
        "rel": loss_rel.detach(),
        "ep_ptr": loss_ep_ptr.detach(),
        "host_ptr": loss_host_ptr.detach(),
        "host_t": loss_host_t.detach(),
        "rel_t": loss_rel_t.detach(),
        "delta": loss_delta.detach(),
    }
    return loss, parts


@torch.no_grad()
def evaluate(model, loader, device, cfg, criteria):
    model.eval()
    sums = {k: 0.0 for k in ["loss", "action", "rel", "ep_ptr", "host_ptr", "host_t", "rel_t", "delta"]}
    batches = 0

    action_conf = torch.zeros((4, 4), dtype=torch.long)
    rel_conf = torch.zeros((4, 4), dtype=torch.long)

    delta_abs_sum = 0.0
    delta_count = 0
    ht_abs_sum = 0.0
    ht_count = 0

    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch["point_features"], batch["point_mask"], batch["stroke_mask"])
        loss, parts = compute_loss(out, batch, cfg, criteria)

        for k in sums:
            sums[k] += float(parts[k].item())
        batches += 1

        # action metrics
        a_true = batch["action_labels"]
        a_pred = out["action_logits"].argmax(dim=-1)
        mask = a_true != IGNORE
        for t, p in zip(a_true[mask].detach().cpu().tolist(), a_pred[mask].detach().cpu().tolist()):
            action_conf[int(t), int(p)] += 1

        # relation metrics
        active_pair = pair_mask_from_stroke_mask(batch["stroke_mask"])
        r_true = batch["rel_labels"]
        r_pred = out["rel_logits"].argmax(dim=-1)
        for t, p in zip(r_true[active_pair].detach().cpu().tolist(), r_pred[active_pair].detach().cpu().tolist()):
            rel_conf[int(t), int(p)] += 1

        dm = batch["point_delta_mask"] > 0
        if dm.any():
            delta_abs_sum += float(torch.abs(out["delta_pred"][dm] - batch["point_delta"][dm]).mean(dim=-1).sum().item())
            delta_count += int(dm.sum().item())

        hm = batch["host_t_mask"] > 0
        if hm.any():
            host_labels = batch["host_ptr_labels"].clamp_min(0)
            pred_t = out["host_t_pred"].gather(2, host_labels.unsqueeze(-1)).squeeze(-1)
            ht_abs_sum += float(torch.abs(pred_t[hm] - batch["host_t_labels"][hm]).sum().item())
            ht_count += int(hm.sum().item())

    if batches == 0:
        return {"loss": math.inf}

    metrics = {k: sums[k] / batches for k in sums}
    metrics["action_conf"] = action_conf
    metrics["rel_conf"] = rel_conf
    metrics["delta_abs"] = delta_abs_sum / max(1, delta_count)
    metrics["host_t_abs"] = ht_abs_sum / max(1, ht_count)

    def f1(conf, cls):
        tp = conf[cls, cls].item()
        fp = conf[:, cls].sum().item() - tp
        fn = conf[cls, :].sum().item() - tp
        pr = tp / max(1, tp + fp)
        rc = tp / max(1, tp + fn)
        return 2 * pr * rc / max(1e-8, pr + rc)

    metrics["f1_snap"] = f1(action_conf, ACT_MAP["SNAP"])
    metrics["f1_tattach"] = f1(action_conf, ACT_MAP["T_ATTACH"])
    metrics["f1_cmove"] = f1(action_conf, ACT_MAP["CONTROL_MOVE"])
    metrics["f1_e2e"] = f1(rel_conf, REL_MAP["E2E"])
    metrics["f1_t"] = f1(rel_conf, REL_MAP["T"])
    metrics["f1_x"] = f1(rel_conf, REL_MAP["X"])

    model.train()
    return metrics


def print_eval(epoch, train_loss, m, lr):
    print(
        f"Epoch {epoch:03d} | TrainLoss={train_loss:.4f} | ValLoss={m['loss']:.4f} | LR={lr:.2e}\n"
        f"  parts: action={m['action']:.3f} rel={m['rel']:.3f} ep_ptr={m['ep_ptr']:.3f} "
        f"host_ptr={m['host_ptr']:.3f} host_t={m['host_t']:.3f} rel_t={m['rel_t']:.3f} delta={m['delta']:.3f}\n"
        f"  action F1 SNAP/T_ATTACH/CONTROL_MOVE = {m['f1_snap']:.3f}/{m['f1_tattach']:.3f}/{m['f1_cmove']:.3f} | "
        f"rel F1 E2E/T/X = {m['f1_e2e']:.3f}/{m['f1_t']:.3f}/{m['f1_x']:.3f} | "
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

    ds = Phase2AdvancedDataset(TOPO_DATA_DIR, cfg)
    if len(ds) == 0:
        print("❌ dataset empty.")
        return

    ac = ds.action_counts()
    rc = ds.rel_counts()
    print("📊 action counts:", {ACT_INV[i]: int(ac[i]) for i in range(4)})
    print("📊 relation counts:", {REL_INV[i]: int(rc[i]) for i in range(4)})

    action_w = class_weights(ac).to(device)
    rel_w = class_weights(rc).to(device)
    print("⚖️ action weights:", [round(float(x), 3) for x in action_w.detach().cpu()])
    print("⚖️ relation weights:", [round(float(x), 3) for x in rel_w.detach().cpu()])

    train_set, val_set = split_dataset(ds, cfg)
    print(f"📦 split train={len(train_set)} val={len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # infer point feature dim
    feat_dim = ds[0]["point_features"].shape[-1]
    model = TopoPhase2AdvancedModel(feat_dim, cfg).to(device)

    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))

    ce_action = nn.CrossEntropyLoss(weight=action_w, ignore_index=IGNORE)
    ce_rel = nn.CrossEntropyLoss(weight=rel_w, ignore_index=IGNORE)
    ce_plain = nn.CrossEntropyLoss(ignore_index=IGNORE)
    huber = nn.SmoothL1Loss(reduction="mean")
    criteria = (ce_action, ce_rel, ce_plain, huber)

    best = math.inf
    best_path = os.path.join(MODEL_SAVE_DIR, cfg.best_name)
    latest_path = os.path.join(MODEL_SAVE_DIR, cfg.latest_name)

    print("🚀 start advanced Phase2 training...")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            out = model(batch["point_features"], batch["point_mask"], batch["stroke_mask"])
            loss, parts = compute_loss(out, batch, cfg, criteria)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            total += float(loss.item())
            batches += 1

        sched.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            val_m = evaluate(model, val_loader, device, cfg, criteria)
            lr = sched.get_last_lr()[0]
            train_loss = total / max(1, batches)
            print_eval(epoch, train_loss, val_m, lr)

            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "point_feature_dim": feat_dim,
                "rel_map": REL_MAP,
                "act_map": ACT_MAP,
                "epoch": epoch,
                "val_loss": val_m["loss"],
                "action_counts": ac.tolist(),
                "relation_counts": rc.tolist(),
            }
            torch.save(ckpt, latest_path)
            if val_m["loss"] < best:
                best = val_m["loss"]
                torch.save(ckpt, best_path)
                print(f"  ✅ best saved: {best_path}")

    print(f"🎉 done. best={best:.4f}")
    print(f"best:   {best_path}")
    print(f"latest: {latest_path}")


if __name__ == "__main__":
    train()
