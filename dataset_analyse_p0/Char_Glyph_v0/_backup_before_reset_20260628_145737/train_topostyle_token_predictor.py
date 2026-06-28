# -*- coding: utf-8 -*-
"""
train_topostyle_token_predictor.py

TopoStyle Token Predictor

前置文件：
    1. train_topostyle_transformer.py
    2. topostyle_style_codebook.json
    3. topostyle_codebook_assignments.json

训练目标：
    不再连续回归 alpha/beta。
    输入 topology-first segment features，输出 style_token 分类。

结构：
    topology anchors / segment graph
        ↓
    Transformer over segments
        ↓
    style_token logits
        ↓
    codebook[token] -> deterministic Bézier

运行：
    cd .../Char_Glyph_v0
    python train_topostyle_token_predictor.py

输出：
    topostyle_token_predictor_best.pt
    topostyle_token_predictor_final.pt
    topostyle_token_predictor.pt
    topostyle_token_predictor_train_report.json
    topostyle_token_predictor_val_predictions.json
"""

import os
import sys
import json
import math
import time
import random
from collections import Counter, defaultdict

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    raise RuntimeError("需要 PyTorch。请先安装 torch。") from e

try:
    import train_topostyle_transformer as ts
except Exception as e:
    raise RuntimeError(
        "请把 train_topostyle_token_predictor.py 放在 train_topostyle_transformer.py 同一个目录下运行。"
    ) from e


# =========================================================
# 0. Paths
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CODEBOOK_FILE = os.path.join(SCRIPT_DIR, "topostyle_style_codebook.json")
ASSIGNMENTS_FILE = os.path.join(SCRIPT_DIR, "topostyle_codebook_assignments.json")

OUTPUT_BEST_MODEL_FILE = os.path.join(SCRIPT_DIR, "topostyle_token_predictor_best.pt")
OUTPUT_FINAL_MODEL_FILE = os.path.join(SCRIPT_DIR, "topostyle_token_predictor_final.pt")
OUTPUT_MODEL_FILE = os.path.join(SCRIPT_DIR, "topostyle_token_predictor.pt")
OUTPUT_REPORT_FILE = os.path.join(SCRIPT_DIR, "topostyle_token_predictor_train_report.json")
OUTPUT_VAL_PRED_FILE = os.path.join(SCRIPT_DIR, "topostyle_token_predictor_val_predictions.json")


# =========================================================
# 1. Config
# =========================================================
RANDOM_SEED = 42
DEVICE_MODE = "auto"  # auto / cuda / mps / cpu

TRAIN_RATIO = 0.80
EPOCHS = 500
BATCH_SIZE = 64
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 2.0

PRINT_EVERY_EPOCH = 10
SAVE_EVERY_EPOCH = 50

# data augment：对整体 glyph 做刚体/尺度增强，style_token 不变
AUGMENT_PER_GLYPH = 32
GLOBAL_TRANSLATE_STD = 0.020
GLOBAL_ROTATE_STD_DEG = 7.0
GLOBAL_SCALE_STD = 0.050

# model
SEG_FEAT_DIM = 22
D_MODEL = 128
NUM_LAYERS = 4
NUM_HEADS = 4
DROPOUT = 0.10
SHAPE_EMB_DIM = 24
WIDTH_EMB_DIM = 8

# loss
LABEL_SMOOTHING = 0.03
USE_CLASS_WEIGHTS = True
CLASS_WEIGHT_POWER = 0.50
CLASS_WEIGHT_MAX = 3.00
CLASS_WEIGHT_MIN = 0.25

# metrics
TOPK_LIST = [1, 3, 5, 10]

# 使用 codebook token 的 deterministic Bézier 重建误差作为最终视觉指标
GOOD_GLYPH_MAX_RMSE_PX = 3.0
USABLE_GLYPH_MAX_RMSE_PX = 8.0


# =========================================================
# 2. Utils
# =========================================================
def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def stats(vals):
    vals = np.asarray(vals, dtype=np.float32)
    if len(vals) == 0:
        return {}
    return {
        "mean": round(float(np.mean(vals)), 6),
        "p50": round(float(np.percentile(vals, 50)), 6),
        "p90": round(float(np.percentile(vals, 90)), 6),
        "p95": round(float(np.percentile(vals, 95)), 6),
        "max": round(float(np.max(vals)), 6),
    }


def curve_rmse_px(P_pred, P_gt):
    Cp = ts.sample_bezier_np(P_pred, n=ts.NUM_CURVE_SAMPLES)
    Cg = ts.sample_bezier_np(P_gt, n=ts.NUM_CURVE_SAMPLES)
    return float(np.sqrt(np.mean(np.sum((Cp - Cg) ** 2, axis=-1))) * ts.CANVAS_SIZE)


def style_to_bezier_np(A, B, style):
    return ts.style_to_bezier_np(A, B, style)


def transform_points_np(points, trans, rot_rad, scale, center):
    pts = np.asarray(points, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32).reshape(1, 2)
    trans = np.asarray(trans, dtype=np.float32).reshape(1, 2)

    co, si = math.cos(rot_rad), math.sin(rot_rad)
    R = np.asarray([[co, -si], [si, co]], dtype=np.float32)

    shp = pts.shape
    X = pts.reshape(-1, 2) - center
    Y = (X * scale) @ R.T + center + trans
    return Y.reshape(shp).astype(np.float32)


def codebook_style_array(codebook_obj):
    entries = codebook_obj.get("codebook", [])
    if not entries:
        raise RuntimeError("codebook json 中没有 codebook 字段。")

    max_token = max(int(e["style_token"]) for e in entries)
    styles = np.zeros((max_token + 1, 5), dtype=np.float32)
    valid = np.zeros((max_token + 1,), dtype=np.float32)

    for e in entries:
        token = int(e["style_token"])
        if "style_vector" in e:
            styles[token] = np.asarray(e["style_vector"], dtype=np.float32)
        else:
            c = e["centroid_style"]
            styles[token] = np.asarray(
                [c["alpha1"], c["beta1"], c["alpha2"], c["beta2"], c["width_norm"]],
                dtype=np.float32,
            )
        valid[token] = 1.0

    if np.any(valid < 0.5):
        missing = np.where(valid < 0.5)[0].tolist()
        raise RuntimeError(f"codebook token 不连续，missing={missing}")

    return styles


def make_assignment_map(assignments_obj):
    rows = assignments_obj.get("assignments", [])
    if not rows:
        raise RuntimeError("assignments json 中没有 assignments。")

    mp = {}
    for r in rows:
        key = (str(r["source_file"]), str(r["hex_key"]), int(r["segment_index_in_glyph"]))
        mp[key] = int(r["style_token"])
    return mp, rows


# =========================================================
# 3. Data preparation
# =========================================================
def collect_labeled_glyphs():
    codebook_obj = load_json(CODEBOOK_FILE)
    assignments_obj = load_json(ASSIGNMENTS_FILE)

    codebook_styles = codebook_style_array(codebook_obj)
    assignment_map, assignment_rows = make_assignment_map(assignments_obj)

    glyphs = ts.load_annotation_glyphs()

    labeled = []
    missed = 0
    token_hist = Counter()

    for g in glyphs:
        labels = []
        ok = True
        for si, seg in enumerate(g["segments"]):
            key = (str(g["source_file"]), str(g["hex_key"]), int(si))
            if key not in assignment_map:
                ok = False
                missed += 1
                break
            token = int(assignment_map[key])
            labels.append(token)
            token_hist[token] += 1

        if ok:
            gg = dict(g)
            gg["style_tokens"] = labels
            labeled.append(gg)

    if len(labeled) == 0:
        raise RuntimeError("没有匹配到 labeled glyph。请确认 codebook/assignment 与当前 annotations 对应。")

    return labeled, codebook_obj, codebook_styles, {
        "usable_labeled_glyphs": len(labeled),
        "missed_segments": missed,
        "token_hist": dict(token_hist),
        "codebook_size": int(codebook_styles.shape[0]),
        "assignment_count": len(assignment_rows),
    }


def glyph_segment_feature(seg, A, B, anchor_degree_start, anchor_degree_end, idx, M, glyph_center, glyph_bbox):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    glyph_center = np.asarray(glyph_center, dtype=np.float32)

    d = B - A
    L = float(np.linalg.norm(d))
    if L < 1e-8:
        angle = 0.0
        d_unit = np.asarray([1.0, 0.0], dtype=np.float32)
    else:
        angle = math.atan2(float(d[1]), float(d[0]))
        d_unit = d / L

    center = 0.5 * (A + B)
    rel_center = center - glyph_center

    xmin, ymin, xmax, ymax = glyph_bbox
    bbox_w = max(1e-6, xmax - xmin)
    bbox_h = max(1e-6, ymax - ymin)
    bbox_diag = math.sqrt(bbox_w * bbox_w + bbox_h * bbox_h)

    t0 = float(seg["t0"])
    t1 = float(seg["t1"])
    straight = float(seg.get("straight", seg.get("straightness", 1.0)))

    return np.asarray([
        A[0], A[1],
        B[0], B[1],
        center[0], center[1],
        rel_center[0], rel_center[1],
        math.sin(angle), math.cos(angle),
        L,
        L / max(1e-6, bbox_diag),
        d_unit[0], d_unit[1],
        t0, t1, t1 - t0,
        float(seg["parent_stroke"]) / max(1.0, ts.MAX_STROKES - 1.0),
        float(idx) / max(1.0, M - 1.0),
        float(anchor_degree_start) / 8.0,
        float(anchor_degree_end) / 8.0,
        straight,
    ], dtype=np.float32)


def augment_anchors_and_gt(glyph, split_name):
    anchors = np.stack([a["pos"] for a in glyph["anchors"]], axis=0).astype(np.float32)
    P_gt = np.stack([seg["P_gt"] for seg in glyph["segments"]], axis=0).astype(np.float32)

    center = anchors.mean(axis=0, keepdims=True) if len(anchors) else np.asarray([[0.5, 0.5]], dtype=np.float32)

    if split_name == "train":
        trans = np.random.normal(0.0, GLOBAL_TRANSLATE_STD, size=(2,)).astype(np.float32)
        rot = math.radians(np.random.normal(0.0, GLOBAL_ROTATE_STD_DEG))
        scale = float(np.exp(np.random.normal(0.0, GLOBAL_SCALE_STD)))
    else:
        trans = np.zeros((2,), dtype=np.float32)
        rot = 0.0
        scale = 1.0

    anchors_aug = transform_points_np(anchors, trans, rot, scale, center=center)
    P_gt_aug = transform_points_np(P_gt, trans, rot, scale, center=center)
    return anchors_aug, P_gt_aug


class TopoStyleTokenDataset(torch.utils.data.Dataset):
    def __init__(self, glyphs, codebook_styles, augment_per_glyph=1, split_name="train"):
        self.glyphs = glyphs
        self.codebook_styles = np.asarray(codebook_styles, dtype=np.float32)
        self.augment_per_glyph = max(1, int(augment_per_glyph))
        self.split_name = split_name

    def __len__(self):
        return len(self.glyphs) * self.augment_per_glyph

    def __getitem__(self, idx):
        glyph = self.glyphs[idx // self.augment_per_glyph]
        segments = glyph["segments"]
        tokens = glyph["style_tokens"]
        M = len(segments)

        anchors_aug, P_gt_aug = augment_anchors_and_gt(glyph, self.split_name)

        seg_feat = np.zeros((ts.MAX_SEGMENTS, SEG_FEAT_DIM), dtype=np.float32)
        shape_ids = np.zeros((ts.MAX_SEGMENTS,), dtype=np.int64)
        width_ids = np.zeros((ts.MAX_SEGMENTS,), dtype=np.int64)
        labels = np.zeros((ts.MAX_SEGMENTS,), dtype=np.int64)
        seg_mask = np.zeros((ts.MAX_SEGMENTS,), dtype=np.float32)
        conn = np.zeros((ts.MAX_SEGMENTS, ts.MAX_SEGMENTS), dtype=np.float32)

        A_arr = np.zeros((ts.MAX_SEGMENTS, 2), dtype=np.float32)
        B_arr = np.zeros((ts.MAX_SEGMENTS, 2), dtype=np.float32)
        P_gt = np.zeros((ts.MAX_SEGMENTS, 4, 2), dtype=np.float32)
        P_oracle_code = np.zeros((ts.MAX_SEGMENTS, 4, 2), dtype=np.float32)

        anchor_degree = Counter()
        for seg in segments:
            anchor_degree[int(seg["anchor_start"])] += 1
            anchor_degree[int(seg["anchor_end"])] += 1

        used_anchor_ids = sorted(set([int(s["anchor_start"]) for s in segments] + [int(s["anchor_end"]) for s in segments]))
        used_anchor_pts = anchors_aug[used_anchor_ids]
        xmin, ymin = used_anchor_pts.min(axis=0)
        xmax, ymax = used_anchor_pts.max(axis=0)
        glyph_bbox = (float(xmin), float(ymin), float(xmax), float(ymax))
        glyph_center = used_anchor_pts.mean(axis=0)

        for i, seg in enumerate(segments):
            a0 = int(seg["anchor_start"])
            a1 = int(seg["anchor_end"])
            A = anchors_aug[a0]
            B = anchors_aug[a1]

            P = P_gt_aug[i].copy()
            P[0] = A
            P[3] = B

            token = int(tokens[i])
            oracle_style = self.codebook_styles[token]
            P_code = style_to_bezier_np(A, B, oracle_style)

            seg_feat[i] = glyph_segment_feature(
                seg,
                A,
                B,
                anchor_degree[a0],
                anchor_degree[a1],
                i,
                M,
                glyph_center,
                glyph_bbox,
            )

            shape_ids[i] = int(np.clip(seg["shape_code"], 0, ts.MAX_SHAPE_CODE - 1))
            width_ids[i] = int(np.clip(seg["width_token"], 0, ts.MAX_WIDTH_TOKEN - 1))
            labels[i] = token
            seg_mask[i] = 1.0
            A_arr[i] = A
            B_arr[i] = B
            P_gt[i] = P
            P_oracle_code[i] = P_code

        for i, si in enumerate(segments):
            for j, sj in enumerate(segments):
                if i == j:
                    continue
                if (
                    si["anchor_start"] == sj["anchor_start"]
                    or si["anchor_start"] == sj["anchor_end"]
                    or si["anchor_end"] == sj["anchor_start"]
                    or si["anchor_end"] == sj["anchor_end"]
                ):
                    conn[i, j] = 1.0

        return {
            "seg_feat": torch.from_numpy(seg_feat),
            "shape_ids": torch.from_numpy(shape_ids),
            "width_ids": torch.from_numpy(width_ids),
            "labels": torch.from_numpy(labels),
            "seg_mask": torch.from_numpy(seg_mask),
            "conn": torch.from_numpy(conn),
            "A": torch.from_numpy(A_arr),
            "B": torch.from_numpy(B_arr),
            "P_gt": torch.from_numpy(P_gt),
            "P_oracle_code": torch.from_numpy(P_oracle_code),
            "meta": {
                "source_file": glyph["source_file"],
                "hex_key": glyph["hex_key"],
                "char": glyph.get("char", ""),
                "num_segments": M,
                "num_anchors": len(glyph["anchors"]),
            },
        }


def collate_fn(items):
    keys = [
        "seg_feat", "shape_ids", "width_ids", "labels", "seg_mask", "conn",
        "A", "B", "P_gt", "P_oracle_code",
    ]
    out = {k: torch.stack([it[k] for it in items], dim=0) for k in keys}
    out["meta"] = [it["meta"] for it in items]
    return out


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# =========================================================
# 4. Model
# =========================================================
class TopoStyleTokenPredictor(nn.Module):
    def __init__(self, num_tokens):
        super().__init__()
        self.num_tokens = int(num_tokens)

        self.shape_emb = nn.Embedding(ts.MAX_SHAPE_CODE, SHAPE_EMB_DIM)
        self.width_emb = nn.Embedding(ts.MAX_WIDTH_TOKEN, WIDTH_EMB_DIM)

        self.in_proj = nn.Sequential(
            nn.Linear(SEG_FEAT_DIM + SHAPE_EMB_DIM + WIDTH_EMB_DIM, D_MODEL),
            nn.GELU(),
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, D_MODEL),
        )

        self.neighbor_proj = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, D_MODEL),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            dim_feedforward=D_MODEL * 4,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=NUM_LAYERS)

        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, self.num_tokens),
        )

    def forward(self, batch):
        feat = batch["seg_feat"].float()
        shape_ids = batch["shape_ids"].long().clamp(0, ts.MAX_SHAPE_CODE - 1)
        width_ids = batch["width_ids"].long().clamp(0, ts.MAX_WIDTH_TOKEN - 1)
        mask = batch["seg_mask"].float()
        conn = batch["conn"].float()

        x = torch.cat([feat, self.shape_emb(shape_ids), self.width_emb(width_ids)], dim=-1)
        h = self.in_proj(x)

        conn_masked = conn * mask[:, None, :] * mask[:, :, None]
        deg = conn_masked.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neigh = torch.bmm(conn_masked, h) / deg
        h = h + self.neighbor_proj(neigh)

        key_padding_mask = mask < 0.5
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        logits = self.head(h)

        return {"logits": logits}


# =========================================================
# 5. Loss / metrics
# =========================================================
def make_class_weights(token_hist, num_tokens, device):
    counts = np.zeros((num_tokens,), dtype=np.float32)
    for k, v in token_hist.items():
        counts[int(k)] = float(v)

    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        return None

    median = float(np.median(nonzero))
    weights = np.ones((num_tokens,), dtype=np.float32)
    for k in range(num_tokens):
        if counts[k] > 0:
            weights[k] = (median / counts[k]) ** CLASS_WEIGHT_POWER
        else:
            weights[k] = 0.0

    weights = np.clip(weights, CLASS_WEIGHT_MIN, CLASS_WEIGHT_MAX)
    return torch.from_numpy(weights).float().to(device)


def compute_ce_loss(logits, labels, mask, class_weights=None):
    B, M, K = logits.shape
    logits_f = logits.reshape(B * M, K)
    labels_f = labels.reshape(B * M).long()
    mask_f = mask.reshape(B * M).float()

    loss = F.cross_entropy(
        logits_f,
        labels_f,
        reduction="none",
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )
    return (loss * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def reconstruct_from_tokens(A, B, token_ids, codebook_styles):
    Bn, M = token_ids.shape
    out = np.zeros((Bn, M, 4, 2), dtype=np.float32)
    for bi in range(Bn):
        for mi in range(M):
            tok = int(token_ids[bi, mi])
            style = codebook_styles[tok]
            out[bi, mi] = style_to_bezier_np(A[bi, mi], B[bi, mi], style)
    return out


def evaluate(model, loader, device, codebook_styles):
    model.eval()

    total_loss = 0.0
    n_batches = 0

    token_correct = {k: 0 for k in TOPK_LIST}
    token_total = 0

    pred_seg_rmse = []
    pred_glyph_max_rmse = []
    oracle_seg_rmse = []
    oracle_glyph_max_rmse = []
    topk5_best_seg_rmse = []
    topk5_best_glyph_max_rmse = []

    with torch.no_grad():
        for batch in loader:
            batch_dev = move_batch_to_device(batch, device)
            out = model(batch_dev)
            logits = out["logits"]
            labels = batch_dev["labels"]
            mask = batch_dev["seg_mask"]

            loss = compute_ce_loss(logits, labels, mask, class_weights=None)
            total_loss += float(loss.detach().cpu())
            n_batches += 1

            probs = torch.softmax(logits, dim=-1)
            max_k = min(max(TOPK_LIST), logits.shape[-1])
            topk = torch.topk(probs, k=max_k, dim=-1).indices.detach().cpu().numpy()

            labels_np = batch["labels"].numpy()
            A_np = batch["A"].numpy()
            B_np = batch["B"].numpy()
            P_gt_np = batch["P_gt"].numpy()
            P_oracle_np = batch["P_oracle_code"].numpy()

            pred_tokens = topk[:, :, 0]
            P_pred_np = reconstruct_from_tokens(A_np, B_np, pred_tokens, codebook_styles)

            for bi in range(labels_np.shape[0]):
                vals_pred = []
                vals_oracle = []
                vals_top5 = []

                M = int(batch["meta"][bi]["num_segments"])

                for si in range(M):
                    token_total += 1
                    gt_tok = int(labels_np[bi, si])

                    for k in TOPK_LIST:
                        kk = min(k, max_k)
                        if gt_tok in topk[bi, si, :kk].tolist():
                            token_correct[k] += 1

                    rmse_pred = curve_rmse_px(P_pred_np[bi, si], P_gt_np[bi, si])
                    rmse_oracle = curve_rmse_px(P_oracle_np[bi, si], P_gt_np[bi, si])

                    best_top5 = 1e9
                    for cand_tok in topk[bi, si, :min(5, max_k)]:
                        Pc = style_to_bezier_np(A_np[bi, si], B_np[bi, si], codebook_styles[int(cand_tok)])
                        best_top5 = min(best_top5, curve_rmse_px(Pc, P_gt_np[bi, si]))

                    vals_pred.append(rmse_pred)
                    vals_oracle.append(rmse_oracle)
                    vals_top5.append(best_top5)

                    pred_seg_rmse.append(rmse_pred)
                    oracle_seg_rmse.append(rmse_oracle)
                    topk5_best_seg_rmse.append(best_top5)

                if vals_pred:
                    pred_glyph_max_rmse.append(float(np.max(vals_pred)))
                    oracle_glyph_max_rmse.append(float(np.max(vals_oracle)))
                    topk5_best_glyph_max_rmse.append(float(np.max(vals_top5)))

    metrics = {
        "ce_loss": total_loss / max(1, n_batches),
        "token_total": int(token_total),
    }

    for k in TOPK_LIST:
        metrics[f"top{k}_acc"] = float(token_correct[k] / max(1, token_total))

    metrics["pred_segment_curve_rmse_px_stats"] = stats(pred_seg_rmse)
    metrics["pred_glyph_max_curve_rmse_px_stats"] = stats(pred_glyph_max_rmse)
    metrics["oracle_codebook_segment_curve_rmse_px_stats"] = stats(oracle_seg_rmse)
    metrics["oracle_codebook_glyph_max_curve_rmse_px_stats"] = stats(oracle_glyph_max_rmse)
    metrics["top5_best_segment_curve_rmse_px_stats"] = stats(topk5_best_seg_rmse)
    metrics["top5_best_glyph_max_curve_rmse_px_stats"] = stats(topk5_best_glyph_max_rmse)

    arr = np.asarray(pred_glyph_max_rmse, dtype=np.float32)
    if len(arr) > 0:
        metrics["good_rate_pred"] = float(np.mean(arr <= GOOD_GLYPH_MAX_RMSE_PX))
        metrics["usable_rate_pred"] = float(np.mean((arr > GOOD_GLYPH_MAX_RMSE_PX) & (arr <= USABLE_GLYPH_MAX_RMSE_PX)))
        metrics["bad_rate_pred"] = float(np.mean(arr > USABLE_GLYPH_MAX_RMSE_PX))
    else:
        metrics["good_rate_pred"] = 0.0
        metrics["usable_rate_pred"] = 0.0
        metrics["bad_rate_pred"] = 1.0

    metrics["topology_junction_px_by_construction"] = 0.0
    metrics["glyph_count"] = int(len(pred_glyph_max_rmse))

    return metrics


def train_one_epoch(model, loader, optimizer, device, class_weights):
    model.train()

    running = 0.0
    n_batches = 0
    correct = 0
    total = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        logits = out["logits"]

        loss = compute_ce_loss(
            logits,
            batch["labels"],
            batch["seg_mask"],
            class_weights=class_weights,
        )
        loss.backward()

        if GRAD_CLIP and GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        running += float(loss.detach().cpu())
        n_batches += 1

        with torch.no_grad():
            pred = torch.argmax(logits, dim=-1)
            mask = batch["seg_mask"] > 0.5
            correct += int(((pred == batch["labels"]) & mask).sum().detach().cpu())
            total += int(mask.sum().detach().cpu())

    return {
        "ce_loss": running / max(1, n_batches),
        "top1_acc": correct / max(1, total),
    }


def save_checkpoint(path, model, optimizer, epoch, best_metric, codebook_styles, config_extra=None):
    cfg = {
        "MODEL_NAME": "TopoStyleTokenPredictor",
        "CODEBOOK_FILE": CODEBOOK_FILE,
        "ASSIGNMENTS_FILE": ASSIGNMENTS_FILE,
        "NUM_STYLE_TOKENS": int(codebook_styles.shape[0]),
        "STYLE_PARAM_ORDER": ["alpha1", "beta1", "alpha2", "beta2", "width_norm"],
        "TOPOLOGY_BY_CONSTRUCTION": True,
        "SEG_FEAT_DIM": SEG_FEAT_DIM,
        "D_MODEL": D_MODEL,
        "NUM_LAYERS": NUM_LAYERS,
        "NUM_HEADS": NUM_HEADS,
        "DROPOUT": DROPOUT,
        "SHAPE_EMB_DIM": SHAPE_EMB_DIM,
        "WIDTH_EMB_DIM": WIDTH_EMB_DIM,
        "MAX_SEGMENTS": ts.MAX_SEGMENTS,
        "MAX_SHAPE_CODE": ts.MAX_SHAPE_CODE,
        "MAX_WIDTH_TOKEN": ts.MAX_WIDTH_TOKEN,
        "CANVAS_SIZE": ts.CANVAS_SIZE,
        "GOOD_GLYPH_MAX_RMSE_PX": GOOD_GLYPH_MAX_RMSE_PX,
        "USABLE_GLYPH_MAX_RMSE_PX": USABLE_GLYPH_MAX_RMSE_PX,
    }
    if config_extra:
        cfg.update(config_extra)

    torch.save({
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "best_metric": float(best_metric),
        "config": cfg,
        "codebook_styles": codebook_styles.astype(np.float32),
    }, path)


def export_val_predictions(model, loader, device, codebook_styles, out_path, max_items=80):
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in loader:
            batch_dev = move_batch_to_device(batch, device)
            logits = model(batch_dev)["logits"]
            probs = torch.softmax(logits, dim=-1)
            max_k = min(10, logits.shape[-1])
            topk = torch.topk(probs, k=max_k, dim=-1)

            topk_ids = topk.indices.detach().cpu().numpy()
            topk_probs = topk.values.detach().cpu().numpy()

            labels_np = batch["labels"].numpy()
            A_np = batch["A"].numpy()
            B_np = batch["B"].numpy()
            P_gt_np = batch["P_gt"].numpy()
            P_oracle_np = batch["P_oracle_code"].numpy()

            for bi, meta in enumerate(batch["meta"]):
                M = int(meta["num_segments"])
                segs = []
                max_pred_rmse = 0.0
                max_oracle_rmse = 0.0

                for si in range(M):
                    pred_tok = int(topk_ids[bi, si, 0])
                    gt_tok = int(labels_np[bi, si])

                    P_pred = style_to_bezier_np(A_np[bi, si], B_np[bi, si], codebook_styles[pred_tok])
                    pred_rmse = curve_rmse_px(P_pred, P_gt_np[bi, si])
                    oracle_rmse = curve_rmse_px(P_oracle_np[bi, si], P_gt_np[bi, si])

                    top_tokens = []
                    for kk in range(max_k):
                        tok = int(topk_ids[bi, si, kk])
                        Pc = style_to_bezier_np(A_np[bi, si], B_np[bi, si], codebook_styles[tok])
                        top_tokens.append({
                            "rank": kk,
                            "style_token": tok,
                            "prob": float(topk_probs[bi, si, kk]),
                            "curve_rmse_px": curve_rmse_px(Pc, P_gt_np[bi, si]),
                            "style_vector": codebook_styles[tok].astype(float).tolist(),
                        })

                    max_pred_rmse = max(max_pred_rmse, pred_rmse)
                    max_oracle_rmse = max(max_oracle_rmse, oracle_rmse)

                    segs.append({
                        "segment_id": si,
                        "gt_style_token": gt_tok,
                        "pred_style_token": pred_tok,
                        "correct_top1": bool(pred_tok == gt_tok),
                        "gt_in_top5": bool(gt_tok in topk_ids[bi, si, :min(5, max_k)].tolist()),
                        "pred_curve_rmse_px": pred_rmse,
                        "oracle_codebook_curve_rmse_px": oracle_rmse,
                        "A_px": ts.denorm_points(A_np[bi, si]).astype(float).tolist(),
                        "B_px": ts.denorm_points(B_np[bi, si]).astype(float).tolist(),
                        "P_gt_px": ts.denorm_points(P_gt_np[bi, si]).astype(float).tolist(),
                        "P_pred_px": ts.denorm_points(P_pred).astype(float).tolist(),
                        "P_oracle_code_px": ts.denorm_points(P_oracle_np[bi, si]).astype(float).tolist(),
                        "top_tokens": top_tokens,
                    })

                rows.append({
                    "source_file": meta["source_file"],
                    "hex_key": meta["hex_key"],
                    "char": meta.get("char", ""),
                    "num_segments": M,
                    "num_anchors": int(meta["num_anchors"]),
                    "max_pred_curve_rmse_px": max_pred_rmse,
                    "max_oracle_codebook_curve_rmse_px": max_oracle_rmse,
                    "segments": segs,
                })

                if len(rows) >= max_items:
                    save_json({
                        "schema_version": "topostyle_token_predictor_val_predictions",
                        "note": "topology endpoints are fixed by anchors; junction error is 0 by construction",
                        "predictions": rows,
                    }, out_path)
                    return

    save_json({
        "schema_version": "topostyle_token_predictor_val_predictions",
        "note": "topology endpoints are fixed by anchors; junction error is 0 by construction",
        "predictions": rows,
    }, out_path)


# =========================================================
# 6. Main
# =========================================================
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    set_seed(RANDOM_SEED)
    device = choose_device()

    print("\n" + "=" * 80)
    print("TopoStyle Token Predictor Training")
    print("=" * 80)
    print(f"  annotation_dir: {ts.ANNOTATION_DIR}")
    print(f"  codebook:       {CODEBOOK_FILE}")
    print(f"  assignments:    {ASSIGNMENTS_FILE}")
    print(f"  output_best:    {OUTPUT_BEST_MODEL_FILE}")
    print(f"  output_final:   {OUTPUT_FINAL_MODEL_FILE}")
    print(f"  output_compat:  {OUTPUT_MODEL_FILE}")
    print(f"  output_report:  {OUTPUT_REPORT_FILE}")
    print(f"  device:         {device}")
    print("=" * 80)

    print("\n[Method]")
    print("  No constraint_solver.py.")
    print("  No continuous alpha/beta regression.")
    print("  Topology anchors define segment endpoints by construction.")
    print("  Model predicts discrete style_token, then codebook[token] builds Bézier.")
    print(f"  EPOCHS: {EPOCHS}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  AUGMENT_PER_GLYPH: {AUGMENT_PER_GLYPH}")
    print(f"  LR: {LR}")
    print(f"  D_MODEL: {D_MODEL}")
    print(f"  NUM_LAYERS: {NUM_LAYERS}")
    print(f"  NUM_HEADS: {NUM_HEADS}")
    print(f"  USE_CLASS_WEIGHTS: {USE_CLASS_WEIGHTS}")
    print(f"  LABEL_SMOOTHING: {LABEL_SMOOTHING}")

    glyphs, codebook_obj, codebook_styles, data_info = collect_labeled_glyphs()

    print("\n[Data]")
    print(f"  labeled_glyphs: {len(glyphs)}")
    print(f"  codebook_size: {codebook_styles.shape[0]}")
    print(f"  assignment_count: {data_info['assignment_count']}")
    print(f"  token_hist_top10: {Counter({int(k): int(v) for k, v in data_info['token_hist'].items()}).most_common(10)}")

    random.Random(RANDOM_SEED).shuffle(glyphs)
    n_train = max(1, int(round(len(glyphs) * TRAIN_RATIO)))
    n_train = min(n_train, len(glyphs) - 1) if len(glyphs) > 1 else len(glyphs)

    train_glyphs = glyphs[:n_train]
    val_glyphs = glyphs[n_train:] if n_train < len(glyphs) else glyphs[:]

    train_token_hist = Counter()
    for g in train_glyphs:
        for t in g["style_tokens"]:
            train_token_hist[int(t)] += 1

    train_ds = TopoStyleTokenDataset(
        train_glyphs,
        codebook_styles=codebook_styles,
        augment_per_glyph=AUGMENT_PER_GLYPH,
        split_name="train",
    )
    val_ds = TopoStyleTokenDataset(
        val_glyphs,
        codebook_styles=codebook_styles,
        augment_per_glyph=1,
        split_name="val",
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )

    print("\n[Split]")
    print(f"  train_glyphs: {len(train_glyphs)}")
    print(f"  val_glyphs:   {len(val_glyphs)}")
    print(f"  train_items:  {len(train_ds)}")
    print(f"  val_items:    {len(val_ds)}")
    print(f"  train_steps_per_epoch: {len(train_loader)}")
    print(f"  train_token_hist_top10: {train_token_hist.most_common(10)}")

    model = TopoStyleTokenPredictor(num_tokens=codebook_styles.shape[0]).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    print("\n[Model]")
    print(f"  parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    class_weights = None
    if USE_CLASS_WEIGHTS:
        class_weights = make_class_weights(train_token_hist, codebook_styles.shape[0], device)
        print(f"  class_weights: enabled min={float(class_weights.min()):.3f} max={float(class_weights.max()):.3f}")

    history = []
    best_metric = float("inf")
    best_epoch = -1

    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, class_weights)
        val_metrics = evaluate(model, val_loader, device, codebook_styles)

        metric = val_metrics["pred_glyph_max_curve_rmse_px_stats"].get("mean", 1e9)

        saved_best = False
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            save_checkpoint(
                OUTPUT_BEST_MODEL_FILE,
                model,
                optimizer,
                epoch,
                best_metric,
                codebook_styles,
                config_extra={"n_params": n_params, "best_epoch": best_epoch},
            )
            saved_best = True

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
        }
        history.append(row)

        if epoch == 1 or epoch % PRINT_EVERY_EPOCH == 0 or saved_best:
            elapsed_min = (time.time() - t_start) / 60.0
            pred_stats = val_metrics["pred_glyph_max_curve_rmse_px_stats"]
            oracle_stats = val_metrics["oracle_codebook_glyph_max_curve_rmse_px_stats"]
            top5_stats = val_metrics["top5_best_glyph_max_curve_rmse_px_stats"]

            print(
                f"  epoch {epoch:04d}/{EPOCHS} | time={elapsed_min:.1f}m | "
                f"train_ce={train_metrics['ce_loss']:.4f} | "
                f"train_acc={train_metrics['top1_acc']:.3f} | "
                f"val_top1={val_metrics['top1_acc']:.3f} | "
                f"val_top3={val_metrics['top3_acc']:.3f} | "
                f"val_top5={val_metrics['top5_acc']:.3f} | "
                f"pred_glyphMaxRMSE_mean={pred_stats.get('mean', 0):.3f}px | "
                f"p50={pred_stats.get('p50', 0):.3f}px | "
                f"p90={pred_stats.get('p90', 0):.3f}px | "
                f"top5Best_mean={top5_stats.get('mean', 0):.3f}px | "
                f"oracleCode_mean={oracle_stats.get('mean', 0):.3f}px | "
                f"good={val_metrics.get('good_rate_pred', 0):.3f} | "
                f"usable={val_metrics.get('usable_rate_pred', 0):.3f} | "
                f"bad={val_metrics.get('bad_rate_pred', 0):.3f}"
            )
            if saved_best:
                print(f"    saved best: {OUTPUT_BEST_MODEL_FILE} | metric={best_metric:.4f}px")

        if epoch % SAVE_EVERY_EPOCH == 0:
            save_checkpoint(
                OUTPUT_FINAL_MODEL_FILE,
                model,
                optimizer,
                epoch,
                best_metric,
                codebook_styles,
                config_extra={"n_params": n_params, "best_epoch": best_epoch},
            )

    save_checkpoint(
        OUTPUT_FINAL_MODEL_FILE,
        model,
        optimizer,
        EPOCHS,
        best_metric,
        codebook_styles,
        config_extra={"n_params": n_params, "best_epoch": best_epoch},
    )

    if os.path.exists(OUTPUT_BEST_MODEL_FILE):
        import shutil
        shutil.copyfile(OUTPUT_BEST_MODEL_FILE, OUTPUT_MODEL_FILE)

    if os.path.exists(OUTPUT_BEST_MODEL_FILE):
        ckpt = torch.load(OUTPUT_BEST_MODEL_FILE, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

    final_val = evaluate(model, val_loader, device, codebook_styles)
    export_val_predictions(model, val_loader, device, codebook_styles, OUTPUT_VAL_PRED_FILE, max_items=80)

    report = {
        "schema_version": "topostyle_token_predictor_train_report",
        "method": "Topology-first anchor segments + discrete TopoStyle token prediction",
        "no_constraint_solver": True,
        "no_continuous_regression": True,
        "topology_by_construction": True,
        "config": {
            "CODEBOOK_FILE": CODEBOOK_FILE,
            "ASSIGNMENTS_FILE": ASSIGNMENTS_FILE,
            "NUM_STYLE_TOKENS": int(codebook_styles.shape[0]),
            "TRAIN_RATIO": TRAIN_RATIO,
            "AUGMENT_PER_GLYPH": AUGMENT_PER_GLYPH,
            "EPOCHS": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "LR": LR,
            "D_MODEL": D_MODEL,
            "NUM_LAYERS": NUM_LAYERS,
            "NUM_HEADS": NUM_HEADS,
            "USE_CLASS_WEIGHTS": USE_CLASS_WEIGHTS,
            "LABEL_SMOOTHING": LABEL_SMOOTHING,
            "GOOD_GLYPH_MAX_RMSE_PX": GOOD_GLYPH_MAX_RMSE_PX,
            "USABLE_GLYPH_MAX_RMSE_PX": USABLE_GLYPH_MAX_RMSE_PX,
        },
        "data": {
            "labeled_glyphs": len(glyphs),
            "train_glyphs": len(train_glyphs),
            "val_glyphs": len(val_glyphs),
            "train_items": len(train_ds),
            "val_items": len(val_ds),
            "global_token_hist": data_info["token_hist"],
            "train_token_hist": dict(train_token_hist),
        },
        "model": {
            "parameters": n_params,
            "best_epoch": best_epoch,
            "best_val_pred_glyph_max_rmse_px": best_metric,
            "final_val": final_val,
        },
        "outputs": {
            "best_model": OUTPUT_BEST_MODEL_FILE,
            "final_model": OUTPUT_FINAL_MODEL_FILE,
            "compat_model": OUTPUT_MODEL_FILE,
            "val_predictions": OUTPUT_VAL_PRED_FILE,
        },
        "history": history,
    }

    save_json(report, OUTPUT_REPORT_FILE)

    print("\n" + "=" * 80)
    print("Training Finished")
    print("=" * 80)
    print(f"  best_epoch: {best_epoch}")
    print(f"  best_val_pred_glyph_max_rmse_px: {best_metric:.4f}")
    print(f"  final_val: {final_val}")
    print(f"  saved_best_model: {OUTPUT_BEST_MODEL_FILE}")
    print(f"  saved_final_model: {OUTPUT_FINAL_MODEL_FILE}")
    print(f"  saved_compat_model(best copy): {OUTPUT_MODEL_FILE}")
    print(f"  saved_report: {OUTPUT_REPORT_FILE}")
    print(f"  val_predictions: {OUTPUT_VAL_PRED_FILE}")

    print("\nHow to judge:")
    print("  1. oracleCode_mean 是 codebook 本身上限；pred_mean 越接近 oracle 越好。")
    print("  2. top5Best_mean 如果明显好于 top1，后续 inference 可让 scorer 在 top-k token 中选择。")
    print("  3. topology_junction_px_by_construction 永远是 0。")


if __name__ == "__main__":
    main()
