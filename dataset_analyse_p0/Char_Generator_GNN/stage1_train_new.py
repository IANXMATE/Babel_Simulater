import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==========================================
# ⚙️ 全局配置与超参数
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(SCRIPT_DIR, "fontgpt_dataset_graph.json")
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "fontgpt_canonical_graph_latest.pth")

# 图模型特征维度
D_MODEL = 256
D_EDGE = 64
N_HEADS = 8
N_LAYERS = 6
DROPOUT = 0.05

BATCH_SIZE = 256
EPOCHS = 400
LR = 1e-3

NUM_SHAPES = 1000
NUM_WIDTHS = 4
NUM_EDGE_TYPES = 4  # 0:NONE, 1:E2E, 2:X, 3:T
MAX_NODES = 50

CANVAS_SIZE = 400.0

# 固定项
MIN_NODE_DIST_SQ = 0.04  # 最小节点间距² = 0.2²


# ==========================================
# 🎚️ Loss Schedule
# ==========================================
def _lerp(a, b, t):
    """线性插值"""
    return a + (b - a) * t


def get_loss_weights(epoch, total_epochs):
    """
    epoch: 0-based epoch index

    设计思路：
    Phase 1: 先强保形，避免模型学成“拓扑合法但不像原字”
    Phase 2: 慢慢增强 junction / coord_angle / edge-t
    Phase 3: 保持物理约束，但 coord 不能降太低
    """
    p = epoch / max(total_epochs - 1, 1)

    # -------------------------------
    # Phase 1: 0% ~ 25%
    # 强保形阶段
    # -------------------------------
    if p < 0.25:
        return {
            "edge": 0.8,
            "t": 3.0,
            "angle": 0.5,
            "length": 5.0,

            "coord": 50.0,
            "junction": 1.0,
            "coord_angle": 1.0,
            "repulse": 0.5,
        }

    # -------------------------------
    # Phase 2: 25% ~ 70%
    # 从保形逐渐过渡到物理约束
    # -------------------------------
    elif p < 0.70:
        q = (p - 0.25) / (0.70 - 0.25)

        return {
            "edge": _lerp(0.8, 1.0, q),
            "t": _lerp(3.0, 5.0, q),
            "angle": _lerp(0.5, 1.0, q),
            "length": 5.0,

            "coord": _lerp(50.0, 50.0, q),
            "junction": _lerp(1.0, 5.0, q),
            "coord_angle": _lerp(1.0, 3.0, q),
            "repulse": _lerp(0.5, 1.0, q),
        }

    # -------------------------------
    # Phase 3: 70% ~ 100%
    # 物理约束精修阶段
    # -------------------------------
    else:
        return {
        "edge": 0.3,
        "t": 2.0,
        "angle": 1.0,
        "length": 5.0,

        "coord": 50.0,
        "junction": 3.0,
        "coord_angle": 2.0,
        "repulse": 0.5,
    }


# ==========================================
# 📦 规范化完备图 Dataset
# ==========================================
class FontCompleteGraphDataset(Dataset):
    def __init__(self, json_file):
        print(f"📖 加载 Canonical Graph 数据: {json_file}")
        with open(json_file, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        item = self.raw_data[idx]
        num_nodes = item["num_nodes"]

        shapes = torch.zeros(num_nodes, dtype=torch.long)
        widths = torch.zeros(num_nodes, dtype=torch.long)
        coords = torch.zeros(num_nodes, 4, dtype=torch.float)

        for n in item["nodes"]:
            i = n["node_id"]
            shapes[i] = n["shape_code"]
            widths[i] = n["width_token"]

            p0_x = (n["p0_cell"][0] + n["p0_offset"][0]) / 32.0
            p0_y = (n["p0_cell"][1] + n["p0_offset"][1]) / 32.0
            p3_x = (n["p3_cell"][0] + n["p3_offset"][0]) / 32.0
            p3_y = (n["p3_cell"][1] + n["p3_offset"][1]) / 32.0

            coords[i] = torch.tensor([p0_x, p0_y, p3_x, p3_y])

        # 字形中心化：去除绝对位置
        all_pts = coords[:num_nodes].view(-1, 2)
        center = all_pts.mean(dim=0)

        coords[:num_nodes, 0:2] -= center
        coords[:num_nodes, 2:4] -= center

        edge_types = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
        edge_ts = torch.zeros((num_nodes, num_nodes, 6), dtype=torch.float)

        for e in item["edges"]:
            u, v = e["u"], e["v"]
            edge_types[u, v] = e["j_type_idx"]
            edge_ts[u, v] = torch.tensor([
                e["t_u"],
                e["t_v"],
                e["t_diff"],
                e["t_prod"],
                e.get("angle_sin", 0.0),
                e.get("angle_cos", 1.0),
            ])

        return num_nodes, shapes, widths, coords, edge_types, edge_ts


def collate_complete_graphs(batch):
    max_n = max(item[0] for item in batch)
    B = len(batch)

    b_shapes = torch.zeros(B, max_n, dtype=torch.long)
    b_widths = torch.zeros(B, max_n, dtype=torch.long)
    b_coords = torch.zeros(B, max_n, 4, dtype=torch.float)
    b_edge_types = torch.zeros(B, max_n, max_n, dtype=torch.long)
    b_edge_ts = torch.zeros(B, max_n, max_n, 6, dtype=torch.float)
    b_mask = torch.zeros(B, max_n, dtype=torch.bool)

    for i, (n, shapes, widths, coords, etypes, ets) in enumerate(batch):
        b_shapes[i, :n] = shapes
        b_widths[i, :n] = widths
        b_coords[i, :n] = coords
        b_edge_types[i, :n, :n] = etypes
        b_edge_ts[i, :n, :n] = ets
        b_mask[i, n:] = True

    return b_shapes, b_widths, b_coords, b_edge_types, b_edge_ts, b_mask


# ==========================================
# 🧩 TopoNodeEncoder
# ==========================================
class TopoNodeEncoder(nn.Module):
    def __init__(self, d_edge, d_model, num_edge_types):
        super().__init__()

        self.edge_type_emb = nn.Embedding(num_edge_types, d_edge)
        self.edge_t_enc = nn.Linear(6, d_edge)

        self.agg_proj = nn.Sequential(
            nn.Linear(d_edge * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, edge_types, edge_ts):
        type_e = self.edge_type_emb(edge_types)
        t_e = self.edge_t_enc(edge_ts)

        edge_feat = self.agg_proj(
            torch.cat([type_e, t_e], dim=-1)
        )

        valid_mask = (edge_types > 0).float().unsqueeze(-1)

        edge_sum = (edge_feat * valid_mask).sum(dim=2)
        valid_count = valid_mask.sum(dim=2).clamp(min=1.0)

        node_emb = edge_sum / valid_count

        return node_emb


# ==========================================
# 🧠 Edge-Aware Graph Transformer
# ==========================================
class EdgeAwareGraphAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_edge):
        super().__init__()

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.edge_proj = nn.Linear(d_edge, num_heads)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, edge_attr, padding_mask):
        B, N, _ = x.size()

        Q = self.W_q(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        edge_bias = self.edge_proj(edge_attr).permute(0, 3, 1, 2)
        edge_bias = edge_bias.clamp(min=-5.0, max=5.0)

        scores = scores + edge_bias

        if padding_mask is not None:
            mask_exp = padding_mask.unsqueeze(1).unsqueeze(2).expand(
                B, self.num_heads, N, N
            )
            scores = scores.masked_fill(mask_exp, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)

        return self.out_proj(out)


class GraphTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_edge, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = EdgeAwareGraphAttention(d_model, num_heads, d_edge)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, edge_attr, padding_mask):
        x = x + self.attn(self.ln1(x), edge_attr, padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class FontGraphGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.node_id_emb = nn.Embedding(MAX_NODES, D_MODEL)
        self.topo_node_enc = TopoNodeEncoder(D_EDGE, D_MODEL, NUM_EDGE_TYPES)

        self.edge_type_emb = nn.Embedding(NUM_EDGE_TYPES, D_EDGE)
        self.edge_t_mlp = nn.Sequential(
            nn.Linear(6, D_EDGE),
            nn.GELU(),
            nn.Linear(D_EDGE, D_EDGE),
        )

        self.layers = nn.ModuleList([
            GraphTransformerBlock(D_MODEL, N_HEADS, D_EDGE, DROPOUT)
            for _ in range(N_LAYERS)
        ])

        self.ln_out = nn.LayerNorm(D_MODEL)

        self.decode_shape = nn.Linear(D_MODEL, NUM_SHAPES)
        self.decode_width = nn.Linear(D_MODEL, NUM_WIDTHS)

        self.decode_coords = nn.Sequential(
            nn.Linear(D_MODEL, 128),
            nn.GELU(),
            nn.Linear(128, 4),
            nn.Tanh(),
        )

        self.decode_edge = nn.Sequential(
            nn.Linear(D_MODEL * 2, 128),
            nn.GELU(),
            nn.Linear(128, NUM_EDGE_TYPES + 4),
        )

    def forward(self, edge_types, edge_ts, padding_mask):
        B, N, _ = edge_types.shape
        device = edge_types.device

        node_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, N)

        x = self.node_id_emb(node_ids)
        x = x + self.topo_node_enc(edge_types, edge_ts)

        type_emb = self.edge_type_emb(edge_types)
        t_emb = self.edge_t_mlp(edge_ts)
        edge_attr = type_emb + t_emb

        for layer in self.layers:
            x = layer(x, edge_attr, padding_mask)

        x = self.ln_out(x)

        shape_logits = self.decode_shape(x)
        width_logits = self.decode_width(x)
        coords_pred = self.decode_coords(x)

        x_i = x.unsqueeze(2).expand(B, N, N, -1)
        x_j = x.unsqueeze(1).expand(B, N, N, -1)

        edge_raw = self.decode_edge(torch.cat([x_i, x_j], dim=-1))

        edge_type_logits = edge_raw[..., :NUM_EDGE_TYPES]
        edge_geom_pred = torch.sigmoid(edge_raw[..., NUM_EDGE_TYPES:])

        edge_t_pred = edge_geom_pred[..., :2]
        angle_pred_raw = edge_geom_pred[..., 2:]
        angle_pred = angle_pred_raw * 2.0 - 1.0

        edge_preds = torch.cat(
            [edge_type_logits, edge_t_pred, angle_pred],
            dim=-1,
        )

        return shape_logits, width_logits, coords_pred, edge_preds


# ==========================================
# 🎯 Loss
# ==========================================
def compute_graph_loss(preds, targets, mask, weights):
    shape_logits, width_logits, coords_pred, edge_preds = preds
    gt_shapes, gt_widths, gt_coords, gt_edge_types, gt_edge_ts = targets

    valid_nodes = ~mask

    # ------------------------------
    # 1. Node Loss
    # ------------------------------
    loss_shape = F.cross_entropy(
        shape_logits[valid_nodes],
        gt_shapes[valid_nodes],
    )

    loss_width = F.cross_entropy(
        width_logits[valid_nodes],
        gt_widths[valid_nodes],
    )

    p0_pred = coords_pred[:, :, :2]
    p3_pred = coords_pred[:, :, 2:]

    gt_p0 = gt_coords[:, :, :2]
    gt_p3 = gt_coords[:, :, 2:]

    pred_len = (p3_pred - p0_pred).norm(dim=-1)
    gt_len = (gt_p3 - gt_p0).norm(dim=-1)

    loss_length = F.smooth_l1_loss(
        pred_len[valid_nodes],
        gt_len[valid_nodes],
    )

    loss_node = loss_shape + loss_width + weights["length"] * loss_length

    # ------------------------------
    # 2. Edge Loss
    # ------------------------------
    valid_edges = (
        (gt_edge_types > 0)
        & (~mask.unsqueeze(2))
        & (~mask.unsqueeze(1))
    )

    valid_pairs = (
        (~mask.unsqueeze(2))
        & (~mask.unsqueeze(1))
    )

    edge_type_logits = edge_preds[..., :NUM_EDGE_TYPES]
    edge_t_pred = edge_preds[..., NUM_EDGE_TYPES:NUM_EDGE_TYPES + 2]
    angle_pred = edge_preds[..., NUM_EDGE_TYPES + 2:NUM_EDGE_TYPES + 4]

    loss_edge_type = F.cross_entropy(
        edge_type_logits[valid_pairs],
        gt_edge_types[valid_pairs],
    )

    tu_pred = edge_t_pred[..., 0]
    tv_pred = edge_t_pred[..., 1]

    if valid_edges.sum() > 0:
        gt_tu = gt_edge_ts[..., 0][valid_edges]
        gt_tv = gt_edge_ts[..., 1][valid_edges]

        tu_p = tu_pred[valid_edges]
        tv_p = tv_pred[valid_edges]

        edge_type_pos = gt_edge_types[valid_edges]

        e2e_t_mask = edge_type_pos == 1
        x_t_mask = edge_type_pos == 2
        t_t_mask = edge_type_pos == 3

        loss_t_parts = []

        if e2e_t_mask.sum() > 0:
            loss_t_parts.append(
                F.binary_cross_entropy(tu_p[e2e_t_mask], gt_tu[e2e_t_mask])
                + F.binary_cross_entropy(tv_p[e2e_t_mask], gt_tv[e2e_t_mask])
            )

        # 暂时保持你当前逻辑：
        # T 边也用 BCE。后续可细分端点侧 BCE / host 内部侧 SmoothL1。
        if t_t_mask.sum() > 0:
            loss_t_parts.append(
                F.binary_cross_entropy(tu_p[t_t_mask], gt_tu[t_t_mask])
                + F.binary_cross_entropy(tv_p[t_t_mask], gt_tv[t_t_mask])
            )

        if x_t_mask.sum() > 0:
            loss_t_parts.append(
                F.smooth_l1_loss(tu_p[x_t_mask], gt_tu[x_t_mask])
                + F.smooth_l1_loss(tv_p[x_t_mask], gt_tv[x_t_mask])
            )

        loss_t = sum(loss_t_parts) / max(len(loss_t_parts), 1)
    else:
        loss_t = torch.tensor(0.0, device=coords_pred.device)

    loss_edge = loss_edge_type + weights["t"] * loss_t

    # ------------------------------
    # 2b. Edge angle auxiliary loss
    # ------------------------------
    gt_angle_sin = gt_edge_ts[..., 4]
    gt_angle_cos = gt_edge_ts[..., 5]

    angle_sin_pred = angle_pred[..., 0]
    angle_cos_pred = angle_pred[..., 1]

    if valid_edges.sum() > 0:
        loss_angle = (
            F.smooth_l1_loss(
                angle_sin_pred[valid_edges],
                gt_angle_sin[valid_edges],
            )
            + F.smooth_l1_loss(
                angle_cos_pred[valid_edges],
                gt_angle_cos[valid_edges],
            )
        )
    else:
        loss_angle = torch.tensor(0.0, device=coords_pred.device)

    # ------------------------------
    # 3. Junction Loss
    # ------------------------------
    B, N, _ = gt_edge_types.shape

    p0_i = p0_pred.unsqueeze(2).expand(B, N, N, 2)
    p3_i = p3_pred.unsqueeze(2).expand(B, N, N, 2)

    p0_j = p0_pred.unsqueeze(1).expand(B, N, N, 2)
    p3_j = p3_pred.unsqueeze(1).expand(B, N, N, 2)

    gt_tu_full = gt_edge_ts[..., 0]
    gt_tv_full = gt_edge_ts[..., 1]

    pt_i = p0_i * (1.0 - gt_tu_full.unsqueeze(-1)) + p3_i * gt_tu_full.unsqueeze(-1)
    pt_j = p0_j * (1.0 - gt_tv_full.unsqueeze(-1)) + p3_j * gt_tv_full.unsqueeze(-1)

    dist_sq = (pt_i - pt_j).pow(2).sum(dim=-1)
    dist_l1 = (pt_i - pt_j).abs().sum(dim=-1)

    if valid_edges.sum() > 0:
        loss_junction = (dist_sq + 0.1 * dist_l1).masked_select(valid_edges).mean()
    else:
        loss_junction = torch.tensor(0.0, device=coords_pred.device)

    # ------------------------------
    # 3b. Coordinate angle loss
    # ------------------------------
    vec = p3_pred - p0_pred
    vec_norm = vec / (vec.norm(dim=-1, keepdim=True) + 1e-6)

    vec_i = vec_norm.unsqueeze(2).expand(B, N, N, 2)
    vec_j = vec_norm.unsqueeze(1).expand(B, N, N, 2)

    cos_actual = (vec_i * vec_j).sum(dim=-1)

    sin_actual = (
        vec_i[..., 0] * vec_j[..., 1]
        - vec_i[..., 1] * vec_j[..., 0]
    ).abs()

    gt_cos = gt_edge_ts[..., 5]
    gt_sin = gt_edge_ts[..., 4]

    if valid_edges.sum() > 0:
        loss_coord_angle = (
            F.smooth_l1_loss(cos_actual[valid_edges], gt_cos[valid_edges])
            + F.smooth_l1_loss(sin_actual[valid_edges], gt_sin[valid_edges])
        )
    else:
        loss_coord_angle = torch.tensor(0.0, device=coords_pred.device)

    # ------------------------------
    # 4. Repulse Loss
    # ------------------------------
    mid_pts = (coords_pred[:, :, :2] + coords_pred[:, :, 2:]) / 2.0

    mid_i = mid_pts.unsqueeze(2).expand(B, N, N, 2)
    mid_j = mid_pts.unsqueeze(1).expand(B, N, N, 2)

    dist_sq_nodes = (mid_i - mid_j).pow(2).sum(dim=-1)

    upper_tri = torch.triu(
        torch.ones(N, N, dtype=torch.bool, device=coords_pred.device),
        diagonal=1,
    )

    upper_tri = upper_tri.unsqueeze(0).expand(B, N, N)

    valid_pair = (
        upper_tri
        & (~mask.unsqueeze(2))
        & (~mask.unsqueeze(1))
    )

    if valid_pair.sum() > 0:
        dists_sq = dist_sq_nodes[valid_pair]
        loss_repulse = F.relu(MIN_NODE_DIST_SQ - dists_sq).mean()
    else:
        loss_repulse = torch.tensor(0.0, device=coords_pred.device)

    # ------------------------------
    # 5. Coord Loss：核心保形项
    # ------------------------------
    loss_coord = F.smooth_l1_loss(
        coords_pred[valid_nodes],
        gt_coords[valid_nodes],
    )

    coord_mae_norm = (
        coords_pred[valid_nodes]
        - gt_coords[valid_nodes]
    ).abs().mean()

    coord_mae_px = coord_mae_norm * CANVAS_SIZE

    length_mae_norm = (
        pred_len[valid_nodes]
        - gt_len[valid_nodes]
    ).abs().mean()

    length_mae_px = length_mae_norm * CANVAS_SIZE

    # ------------------------------
    # 6. Total
    # ------------------------------
    total = (
        loss_node
        + weights["edge"] * loss_edge
        + weights["angle"] * loss_angle
        + weights["junction"] * loss_junction
        + weights["coord_angle"] * loss_coord_angle
        + weights["repulse"] * loss_repulse
        + weights["coord"] * loss_coord
    )

    return (
        total,
        loss_node,
        loss_edge,
        loss_angle,
        loss_junction,
        loss_repulse,
        loss_coord,
        coord_mae_px,
        length_mae_px,
    )


# ==========================================
# 🚂 训练循环
# ==========================================
def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"🚀 拓扑驱动几何求解引擎 v3 Schedule版 (Device: {device})")
    print("   节点初始化: NodeIdEmb + TopoNodeEncoder")
    print("   输入: edge_types + edge_ts")
    print("   输出: shape / width / coords / edge reconstruction")
    print("   Loss Schedule: 前期强保形，后期增强 junction/angle")

    dataset = FontCompleteGraphDataset(DATASET_FILE)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_complete_graphs,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = FontGraphGenerator().to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-5,
    )

    for epoch in range(EPOCHS):
        model.train()

        weights = get_loss_weights(epoch, EPOCHS)

        print(
            f"\n🎚️ Loss Weights | "
            f"coord={weights['coord']:.2f}, "
            f"junction={weights['junction']:.2f}, "
            f"coord_angle={weights['coord_angle']:.2f}, "
            f"repulse={weights['repulse']:.2f}, "
            f"edge={weights['edge']:.2f}, "
            f"t={weights['t']:.2f}, "
            f"angle={weights['angle']:.2f}"
        )

        total_loss = 0.0
        total_n = 0.0
        total_e = 0.0
        total_a = 0.0
        total_j = 0.0
        total_s = 0.0
        total_c = 0.0

        total_coord_px = 0.0
        total_len_px = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for b_shapes, b_widths, b_coords, b_edge_types, b_edge_ts, b_mask in pbar:
            b_shapes = b_shapes.to(device, non_blocking=True)
            b_widths = b_widths.to(device, non_blocking=True)
            b_coords = b_coords.to(device, non_blocking=True)
            b_edge_types = b_edge_types.to(device, non_blocking=True)
            b_edge_ts = b_edge_ts.to(device, non_blocking=True)
            b_mask = b_mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            preds = model(
                b_edge_types,
                b_edge_ts,
                b_mask,
            )

            targets = (
                b_shapes,
                b_widths,
                b_coords,
                b_edge_types,
                b_edge_ts,
            )

            (
                loss,
                n_loss,
                e_loss,
                a_loss,
                j_loss,
                r_loss,
                c_loss,
                coord_mae_px,
                length_mae_px,
            ) = compute_graph_loss(
                preds,
                targets,
                b_mask,
                weights,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += loss.item()
            total_n += n_loss.item()
            total_e += e_loss.item()
            total_a += a_loss.item()
            total_j += j_loss.item()
            total_s += r_loss.item()
            total_c += c_loss.item()
            total_coord_px += coord_mae_px.item()
            total_len_px += length_mae_px.item()

            pbar.set_postfix({
                "Total": f"{loss.item():.3f}",
                "Node": f"{n_loss.item():.3f}",
                "Edge": f"{e_loss.item():.3f}",
                "Junc": f"{j_loss.item():.4f}",
                "Coord": f"{c_loss.item():.4f}",
                "Cpx": f"{coord_mae_px.item():.1f}",
                "Lpx": f"{length_mae_px.item():.1f}",
            })

        steps = max(len(dataloader), 1)

        print(
            f"📈 Epoch {epoch+1:3d} | "
            f"Total: {total_loss/steps:.4f} | "
            f"Node: {total_n/steps:.4f} | "
            f"Edge: {total_e/steps:.4f} | "
            f"Angle: {total_a/steps:.5f} | "
            f"Junc: {total_j/steps:.5f} | "
            f"Repul: {total_s/steps:.5f} | "
            f"Coord: {total_c/steps:.5f} | "
            f"CoordPx: {total_coord_px/steps:.2f} | "
            f"LenPx: {total_len_px/steps:.2f}"
        )

    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\n💾 已保存至: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()