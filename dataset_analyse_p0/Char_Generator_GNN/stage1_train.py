import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
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
DROPOUT = 0.1

BATCH_SIZE = 64
EPOCHS = 100
LR = 3e-4

NUM_SHAPES = 1000
NUM_WIDTHS = 4
NUM_EDGE_TYPES = 4  # 0:NONE, 1:E2E, 2:X, 3:T
MAX_NODES = 50

# Loss 权重
LAMBDA_EDGE = 1.0      # L_edge 权重
LAMBDA_JUNCTION = 10.0 # L_junction 权重（核心物理约束）

# ==========================================
# 📦 规范化完备图 (Canonical Complete Graph) 数据集
# ==========================================
class FontCompleteGraphDataset(Dataset):
    def __init__(self, json_file):
        print(f"📖 加载 Canonical Graph 数据: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
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

            # 无偏置的纯净 [0, 1) 坐标还原
            p0_x = (n["p0_cell"][0] + n["p0_offset"][0]) / 32.0
            p0_y = (n["p0_cell"][1] + n["p0_offset"][1]) / 32.0
            p3_x = (n["p3_cell"][0] + n["p3_offset"][0]) / 32.0
            p3_y = (n["p3_cell"][1] + n["p3_offset"][1]) / 32.0
            coords[i] = torch.tensor([p0_x, p0_y, p3_x, p3_y])

        # 完备图 N x N 特征
        edge_types = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
        edge_ts = torch.zeros((num_nodes, num_nodes, 4), dtype=torch.float)  # [t_u, t_v, t_diff, t_prod]

        for e in item["edges"]:
            u, v = e["u"], e["v"]
            edge_types[u, v] = e["j_type_idx"]
            edge_ts[u, v] = torch.tensor([e["t_u"], e["t_v"], e["t_diff"], e["t_prod"]])

        return num_nodes, shapes, widths, coords, edge_types, edge_ts


def collate_complete_graphs(batch):
    max_n = max([item[0] for item in batch])
    B = len(batch)

    b_shapes = torch.zeros(B, max_n, dtype=torch.long)
    b_widths = torch.zeros(B, max_n, dtype=torch.long)
    b_coords = torch.zeros(B, max_n, 4, dtype=torch.float)
    b_edge_types = torch.zeros(B, max_n, max_n, dtype=torch.long)
    b_edge_ts = torch.zeros(B, max_n, max_n, 4, dtype=torch.float)
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
# 🧠 核心架构：Edge-Aware Graph Transformer
# ==========================================
class EdgeAwareGraphAttention(nn.Module):
    """
    Graph Transformer Attention:
        A_ij = Q_i K_j / sqrt(d) + W_e(e_ij)
    edge feature 是 attention bias source，不是标签。
    """
    def __init__(self, d_model, num_heads, d_edge):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # d_edge → num_heads，作为每个 head 的注意力偏置
        self.edge_proj = nn.Linear(d_edge, num_heads)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, edge_attr, padding_mask):
        B, N, _ = x.size()

        Q = self.W_q(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)

        # 基础 QK 相似度
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 🚀 注入拓扑偏置：edge_attr [B,N,N,d_edge] → bias [B,heads,N,N]
        edge_bias = self.edge_proj(edge_attr).permute(0, 3, 1, 2)
        scores = scores + edge_bias

        # Padding mask：屏蔽填充节点
        if padding_mask is not None:
            mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2).expand(B, self.num_heads, N, N)
            scores = scores.masked_fill(mask_expanded, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, N, -1)
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
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, edge_attr, padding_mask):
        x = x + self.attn(self.ln1(x), edge_attr, padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class FontGraphGenerator(nn.Module):
    """
    Graph Transformer Encoder + Node/Edge Geometry Decoder

    输入层（Graph Construction）：
        节点: h_i = Emb(shape_code) + Emb(width_token) + Linear(p0, p3)
        边:   e_ij = Emb(edge_type) + MLP(t_u, t_v, t_diff, t_prod)

    Encoder：
        6层 EdgeAwareGraphAttention（edge 作为 attention bias）

    Decoder：
        方案 A - Node Geometry Decoder：每个节点 → shape/width/coords
        方案 B - Edge Geometry Decoder：每对节点 → edge_type/t_u/t_v
    """
    def __init__(self):
        super().__init__()

        # ──────────────────────────────────────────────
        # 1. Node 输入编码：语义内容嵌入（对齐方案）
        #    h_i = Emb(shape_code) + Emb(width_token) + Linear(p0, p3)
        # ──────────────────────────────────────────────
        self.shape_emb = nn.Embedding(NUM_SHAPES, D_MODEL)
        self.width_emb = nn.Embedding(NUM_WIDTHS, D_MODEL)
        self.coord_proj = nn.Linear(4, D_MODEL)  # [p0_x, p0_y, p3_x, p3_y] → D_MODEL

        # ──────────────────────────────────────────────
        # 2. Edge 输入编码：关系 + 几何约束
        #    e_ij = Emb(edge_type) + MLP(t_u, t_v, t_diff, t_prod)
        # ──────────────────────────────────────────────
        self.edge_type_emb = nn.Embedding(NUM_EDGE_TYPES, D_EDGE)
        self.edge_t_mlp = nn.Sequential(
            nn.Linear(4, D_EDGE),
            nn.GELU(),
            nn.Linear(D_EDGE, D_EDGE)
        )

        # ──────────────────────────────────────────────
        # 3. Graph Transformer Encoder（6层）
        # ──────────────────────────────────────────────
        self.layers = nn.ModuleList([
            GraphTransformerBlock(D_MODEL, N_HEADS, D_EDGE, DROPOUT) for _ in range(N_LAYERS)
        ])
        self.ln_out = nn.LayerNorm(D_MODEL)

        # ──────────────────────────────────────────────
        # 4. 方案 A：Node Geometry Decoder
        #    h_i → shape_logits, width_logits, coords
        # ──────────────────────────────────────────────
        self.decode_shape = nn.Linear(D_MODEL, NUM_SHAPES)
        self.decode_width = nn.Linear(D_MODEL, NUM_WIDTHS)
        self.decode_coords = nn.Sequential(
            nn.Linear(D_MODEL, 128),
            nn.GELU(),
            nn.Linear(128, 4),
            nn.Sigmoid()  # 物理约束：画布 [0, 1)
        )

        # ──────────────────────────────────────────────
        # 5. 方案 B：Edge Geometry Decoder
        #    (h_i, h_j) → edge_type_logits (4) + t_pred (2)
        # ──────────────────────────────────────────────
        self.decode_edge = nn.Sequential(
            nn.Linear(D_MODEL * 2, 128),
            nn.GELU(),
            nn.Linear(128, NUM_EDGE_TYPES + 2)  # 4维类别 logits + 2维 [t_u, t_v]
        )

    def forward(self, shapes, widths, coords, edge_types, edge_ts, padding_mask):
        """
        Args:
            shapes:       [B, N]    节点形状 token
            widths:       [B, N]    节点宽度 token
            coords:       [B, N, 4] 节点坐标 [p0_x, p0_y, p3_x, p3_y]
            edge_types:   [B, N, N] 边类型 (Topology Prior，固定输入)
            edge_ts:      [B, N, N, 4] 边几何特征 [t_u, t_v, t_diff, t_prod]
            padding_mask: [B, N]   True = pad 节点
        Returns:
            shape_logits: [B, N, NUM_SHAPES]
            width_logits: [B, N, NUM_WIDTHS]
            coords_pred:  [B, N, 4]
            edge_preds:   [B, N, N, NUM_EDGE_TYPES+2]  前4维为类型logits，后2维为t_u/t_v
        """
        B, N = shapes.size()

        # ── 节点语义嵌入（方案核心：从 shape/width/coords 构建初始特征）
        x = self.shape_emb(shapes) + self.width_emb(widths) + self.coord_proj(coords)

        # ── 边特征嵌入（Topology Prior 编码为 attention bias）
        type_emb = self.edge_type_emb(edge_types)
        t_emb = self.edge_t_mlp(edge_ts)
        edge_attr = type_emb + t_emb  # [B, N, N, D_EDGE]

        # ── Graph Transformer 深度传播
        for layer in self.layers:
            x = layer(x, edge_attr, padding_mask)
        x = self.ln_out(x)

        # ── 方案 A：Node 解码
        shape_logits = self.decode_shape(x)
        width_logits = self.decode_width(x)
        coords_pred = self.decode_coords(x)

        # ── 方案 B：Edge 解码
        x_i = x.unsqueeze(2).expand(B, N, N, -1)
        x_j = x.unsqueeze(1).expand(B, N, N, -1)
        edge_preds_raw = self.decode_edge(torch.cat([x_i, x_j], dim=-1))  # [B, N, N, 6]

        # 将 t_u, t_v 用 sigmoid 约束到 [0, 1]
        edge_type_logits = edge_preds_raw[..., :NUM_EDGE_TYPES]  # [B, N, N, 4]
        edge_t_pred = torch.sigmoid(edge_preds_raw[..., NUM_EDGE_TYPES:])  # [B, N, N, 2]
        edge_preds = torch.cat([edge_type_logits, edge_t_pred], dim=-1)

        return shape_logits, width_logits, coords_pred, edge_preds


# ==========================================
# 🎯 Loss 设计（对齐方案）
# ==========================================
def compute_graph_loss(preds, targets, mask):
    """
    L = L_node + λ1*L_edge + λ2*L_junction

    L_node = CE(shape) + CE(width) + MSE(p0) + MSE(p3)
    L_edge = CE(edge_type) + MSE(t_u) + MSE(t_v)
    L_junction = || B_i(t_u) - B_j(t_v) ||^2   ← 系统灵魂：预测 t 必须真的在曲线上相交
    """
    shape_logits, width_logits, coords_pred, edge_preds = preds
    gt_shapes, gt_widths, gt_coords, gt_edge_types, gt_edge_ts = targets

    valid_nodes = ~mask  # [B, N]

    # ──────────────────────────────
    # 1. Node Loss
    # ──────────────────────────────
    loss_shape = F.cross_entropy(shape_logits[valid_nodes], gt_shapes[valid_nodes])
    loss_width = F.cross_entropy(width_logits[valid_nodes], gt_widths[valid_nodes])

    p0_pred = coords_pred[:, :, :2]
    p3_pred = coords_pred[:, :, 2:]
    p0_gt = gt_coords[:, :, :2]
    p3_gt = gt_coords[:, :, 2:]

    loss_p0 = F.mse_loss(p0_pred[valid_nodes], p0_gt[valid_nodes])
    loss_p3 = F.mse_loss(p3_pred[valid_nodes], p3_gt[valid_nodes])

    loss_node = loss_shape + loss_width + loss_p0 + loss_p3

    # ──────────────────────────────
    # 2. Edge Loss
    # ──────────────────────────────
    # 只对真实连接的边（非 NONE）计算 loss
    valid_edges = (gt_edge_types > 0) & (~mask.unsqueeze(2)) & (~mask.unsqueeze(1))

    edge_type_logits = edge_preds[..., :NUM_EDGE_TYPES]    # [B, N, N, 4]
    edge_t_pred = edge_preds[..., NUM_EDGE_TYPES:]         # [B, N, N, 2]

    # CE(edge_type)：在所有有效节点对的边上计算（含 NONE 作为负样本）
    valid_node_pairs = (~mask.unsqueeze(2)) & (~mask.unsqueeze(1))  # [B, N, N]
    loss_edge_type = F.cross_entropy(
        edge_type_logits[valid_node_pairs],
        gt_edge_types[valid_node_pairs]
    )

    # MSE(t_u, t_v)：仅在真实连接边上计算
    tu_pred = edge_t_pred[..., 0]
    tv_pred = edge_t_pred[..., 1]
    tu_gt = gt_edge_ts[..., 0]
    tv_gt = gt_edge_ts[..., 1]

    if valid_edges.sum() > 0:
        loss_tu = F.mse_loss(tu_pred[valid_edges], tu_gt[valid_edges])
        loss_tv = F.mse_loss(tv_pred[valid_edges], tv_gt[valid_edges])
        loss_t = loss_tu + loss_tv
    else:
        loss_t = torch.tensor(0.0, device=coords_pred.device)

    loss_edge = loss_edge_type + loss_t

    # ──────────────────────────────
    # 3. Geometry Consistency Loss（系统灵魂）
    #    L_junction = || B_i(t_u) - B_j(t_v) ||^2
    #    强制：预测的 t 值对应的曲线点必须物理相交
    # ──────────────────────────────
    B, N, _ = gt_edge_types.shape

    p0_i = p0_pred.unsqueeze(2).expand(B, N, N, 2)
    p3_i = p3_pred.unsqueeze(2).expand(B, N, N, 2)
    p0_j = p0_pred.unsqueeze(1).expand(B, N, N, 2)
    p3_j = p3_pred.unsqueeze(1).expand(B, N, N, 2)

    # 使用预测的 t 值插值出交点位置（线性近似，因为只有端点）
    tu_exp = tu_pred.unsqueeze(-1)
    tv_exp = tv_pred.unsqueeze(-1)
    pt_i = p0_i * (1 - tu_exp) + p3_i * tu_exp  # 笔画 i 上 t_u 处的点
    pt_j = p0_j * (1 - tv_exp) + p3_j * tv_exp  # 笔画 j 上 t_v 处的点

    dist_sq = (pt_i - pt_j).pow(2).sum(dim=-1)
    dist_l1 = (pt_i - pt_j).abs().sum(dim=-1)

    if valid_edges.sum() > 0:
        loss_junction = (dist_sq + 0.1 * dist_l1).masked_select(valid_edges).mean()
    else:
        loss_junction = torch.tensor(0.0, device=coords_pred.device)

    # ──────────────────────────────
    # 总 Loss
    # ──────────────────────────────
    total_loss = loss_node + LAMBDA_EDGE * loss_edge + LAMBDA_JUNCTION * loss_junction
    return total_loss, loss_node, loss_edge, loss_junction


# ==========================================
# 🚂 完备图模型训练循环
# ==========================================
def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"🚀 启动 Graph Transformer (Canonical 拓扑→几何 求解引擎) (Device: {device})...")
    print(f"   架构: Node Emb(shape+width+coords) → EdgeAttnBias → NodeDecoder + EdgeDecoder")
    print(f"   Loss: L_node + {LAMBDA_EDGE}*L_edge(CE+MSE) + {LAMBDA_JUNCTION}*L_junction")

    dataset = FontCompleteGraphDataset(DATASET_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_complete_graphs)

    model = FontGraphGenerator().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = total_n_loss = total_e_loss = total_j_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for b_shapes, b_widths, b_coords, b_edge_types, b_edge_ts, b_mask in pbar:
            b_shapes = b_shapes.to(device)
            b_widths = b_widths.to(device)
            b_coords = b_coords.to(device)
            b_edge_types = b_edge_types.to(device)
            b_edge_ts = b_edge_ts.to(device)
            b_mask = b_mask.to(device)

            optimizer.zero_grad()

            # 🚀 前向传播：节点语义 + 拓扑先验 → 一次性解算所有几何属性
            preds = model(b_shapes, b_widths, b_coords, b_edge_types, b_edge_ts, b_mask)

            targets = (b_shapes, b_widths, b_coords, b_edge_types, b_edge_ts)
            loss, n_loss, e_loss, j_loss = compute_graph_loss(preds, targets, b_mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_n_loss += n_loss.item()
            total_e_loss += e_loss.item()
            total_j_loss += j_loss.item()

            pbar.set_postfix({
                "Total": f"{loss.item():.3f}",
                "Node": f"{n_loss.item():.3f}",
                "Edge": f"{e_loss.item():.3f}",
                "Junc": f"{j_loss.item():.4f}",  # 核心：交点物理吸合程度
            })

        steps = len(dataloader)
        print(
            f"📈 Epoch {epoch+1:3d} | "
            f"Total: {total_loss/steps:.4f} | "
            f"Node: {total_n_loss/steps:.4f} | "
            f"Edge: {total_e_loss/steps:.4f} | "
            f"Junc: {total_j_loss/steps:.5f}"
        )

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 Graph Transformer 模型已保存至 {MODEL_SAVE_PATH}！")


if __name__ == "__main__":
    main()
