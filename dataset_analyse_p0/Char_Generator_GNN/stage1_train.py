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
LAMBDA_EDGE = 1.0        # L_edge 权重
LAMBDA_T = 5.0           # L_t 权重：t 值只有 0/1，加大权重强化端点对齐监督
LAMBDA_JUNCTION = 5.0    # L_junction 权重（降低防止梯度爆炸）
LAMBDA_COORD_ANGLE = 3.0 # L_coord_angle 权重：坐标切线角度约束（直接作用于坐标）
LAMBDA_REPULSE = 2.0     # 节点互斥力：持续驱动坐标分散
MIN_NODE_DIST_SQ = 0.04  # 最小节点间距^2（=0.2^2，用平方距避免 sqrt 导数爆炸）
LAMBDA_ANGLE = 1.0       # L_angle 权重：监督 edge_preds 里的 angle sin/cos（辅助监督）
LAMBDA_LENGTH = 5.0      # L_length 权重：监督笔画长度（平移不变，不引起均值坍塌）

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
            p0_x = (n["p0_cell"][0] + n["p0_offset"][0]) / 32.0
            p0_y = (n["p0_cell"][1] + n["p0_offset"][1]) / 32.0
            p3_x = (n["p3_cell"][0] + n["p3_offset"][0]) / 32.0
            p3_y = (n["p3_cell"][1] + n["p3_offset"][1]) / 32.0
            coords[i] = torch.tensor([p0_x, p0_y, p3_x, p3_y])

        # 计算字形中心并中心化坐标：去除绝对位置信息
        # 字体数据对称分布导致均值恒为 0.5，中心化后坐标方差更大、模型更好学习
        all_pts = coords[:num_nodes].view(-1, 2)   # [2N, 2]
        center = all_pts.mean(dim=0)               # [2]
        # 中心化：每个笔画端点减字形中心
        coords[:num_nodes, 0:2] -= center
        coords[:num_nodes, 2:4] -= center

        edge_types = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
        edge_ts = torch.zeros((num_nodes, num_nodes, 6), dtype=torch.float)  # [t_u,t_v,t_diff,t_prod,angle_sin,angle_cos]

        for e in item["edges"]:
            u, v = e["u"], e["v"]
            edge_types[u, v] = e["j_type_idx"]
            edge_ts[u, v] = torch.tensor([
                e["t_u"], e["t_v"], e["t_diff"], e["t_prod"],
                e.get("angle_sin", 0.0), e.get("angle_cos", 1.0)
            ])

        # shapes/widths/coords 只作为 GT 监督目标，不再传入模型
        return num_nodes, shapes, widths, coords, edge_types, edge_ts


def collate_complete_graphs(batch):
    max_n = max([item[0] for item in batch])
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
# 🧩 TopoNodeEncoder：从拓扑边聚合出节点嵌入
# ==========================================
class TopoNodeEncoder(nn.Module):
    """
    核心设计：节点嵌入完全由拓扑结构决定，不依赖任何 GT 节点属性。

    对节点 i，聚合所有出边 (i, j) 的特征（edge_type + t 值）：
        node_emb_i = MeanPool_{j: type(i,j)>0} [ MLP(Emb(type_ij) + Linear(t_ij)) ]

    推理时只需要 edge_types 和 edge_ts，与训练完全一致，无 train-test gap。
    """
    def __init__(self, d_edge, d_model, num_edge_types):
        super().__init__()
        self.edge_type_emb = nn.Embedding(num_edge_types, d_edge)
        self.edge_t_enc = nn.Linear(6, d_edge)
        self.agg_proj = nn.Sequential(
            nn.Linear(d_edge * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, edge_types, edge_ts):
        """
        edge_types: [B, N, N]
        edge_ts:    [B, N, N, 4]
        return:     [B, N, D_MODEL]
        """
        # 编码每条边
        type_e = self.edge_type_emb(edge_types)    # [B, N, N, d_edge]
        t_e = self.edge_t_enc(edge_ts)             # [B, N, N, d_edge]
        edge_feat = self.agg_proj(
            torch.cat([type_e, t_e], dim=-1)
        )  # [B, N, N, D_MODEL]

        # 仅聚合有效边（type > 0），沿 j 维度 mean pooling
        valid_mask = (edge_types > 0).float().unsqueeze(-1)   # [B, N, N, 1]
        edge_sum = (edge_feat * valid_mask).sum(dim=2)         # [B, N, D_MODEL]
        valid_count = valid_mask.sum(dim=2).clamp(min=1.0)    # [B, N, 1]
        node_emb = edge_sum / valid_count                      # [B, N, D_MODEL]

        return node_emb


# ==========================================
# 🧠 核心架构：Edge-Aware Graph Transformer
# ==========================================
class EdgeAwareGraphAttention(nn.Module):
    """
    A_ij = Q_i K_j / sqrt(d) + W_e(e_ij)
    edge feature 是 attention bias source。
    """
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
        scores = scores + edge_bias
        if padding_mask is not None:
            mask_exp = padding_mask.unsqueeze(1).unsqueeze(2).expand(B, self.num_heads, N, N)
            scores = scores.masked_fill(mask_exp, float('-inf'))
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
    拓扑驱动几何生成器（Train-Inference 对齐版）

    节点初始化（完全无 GT 特征输入）：
        x_i = NodeIdEmb(i) + TopoNodeEncoder(edge_types, edge_ts)_i

    - NodeIdEmb(i): 提供节点索引区分能力（i=0,1,...,N-1），训练推理均可用
    - TopoNodeEncoder: 从边的拓扑约束聚合节点的几何约束信号，训练推理均可用

    模型不再接受 shapes/widths/coords 作为输入，彻底消除训练-推理鸿沟。
    shapes/widths/coords 仅作为 Loss 的监督目标。
    """
    def __init__(self):
        super().__init__()

        # ── 1. 节点索引编码（提供基础的节点区分能力）
        self.node_id_emb = nn.Embedding(MAX_NODES, D_MODEL)

        # ── 2. 拓扑感知节点编码（从边聚合，推理时完全可用）
        self.topo_node_enc = TopoNodeEncoder(D_EDGE, D_MODEL, NUM_EDGE_TYPES)

        # ── 3. Edge 输入编码（Attention Bias Source）
        self.edge_type_emb = nn.Embedding(NUM_EDGE_TYPES, D_EDGE)
        self.edge_t_mlp = nn.Sequential(
            nn.Linear(6, D_EDGE),
            nn.GELU(),
            nn.Linear(D_EDGE, D_EDGE)
        )

        # ── 4. Graph Transformer Encoder
        self.layers = nn.ModuleList([
            GraphTransformerBlock(D_MODEL, N_HEADS, D_EDGE, DROPOUT)
            for _ in range(N_LAYERS)
        ])
        self.ln_out = nn.LayerNorm(D_MODEL)

        # ── 5. 坐标解码器（无约束输出 + 训练时用 tanh 配合重背景巭）
        # 同时保留 shape/width decoder
        self.decode_shape = nn.Linear(D_MODEL, NUM_SHAPES)
        self.decode_width = nn.Linear(D_MODEL, NUM_WIDTHS)
        self.decode_coords = nn.Sequential(
            nn.Linear(D_MODEL, 128),
            nn.GELU(),
            nn.Linear(128, 4),
            nn.Tanh()    # 输出范围 [-1, 1]，非对称坐标空间，梯度流在中逆局更健康
        )

        # ── 6. Edge Geometry Decoder：预测 edge_type + t_u/t_v + angle_sin/cos
        self.decode_edge = nn.Sequential(
            nn.Linear(D_MODEL * 2, 128),
            nn.GELU(),
            nn.Linear(128, NUM_EDGE_TYPES + 4)  # 4 类 logits + [t_u, t_v, angle_sin, angle_cos]
        )

    def forward(self, edge_types, edge_ts, padding_mask):
        """
        完全拓扑驱动的 forward，与推理完全对齐。

        Args:
            edge_types:   [B, N, N]    拓扑先验（已知）
            edge_ts:      [B, N, N, 4] 边几何特征（已知）
            padding_mask: [B, N]       True = pad 节点

        Returns:
            shape_logits: [B, N, NUM_SHAPES]
            width_logits: [B, N, NUM_WIDTHS]
            coords_pred:  [B, N, 4]
            edge_preds:   [B, N, N, NUM_EDGE_TYPES+4]  # type_logits + t_u + t_v + angle_sin + angle_cos
        """
        B, N, _ = edge_types.shape
        device = edge_types.device

        # ── 节点初始化（纯拓扑驱动，无 GT 依赖）
        node_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        x = self.node_id_emb(node_ids) + self.topo_node_enc(edge_types, edge_ts)

        # ── Edge 特征编码（作为 Attention Bias）
        type_emb = self.edge_type_emb(edge_types)
        t_emb = self.edge_t_mlp(edge_ts)
        edge_attr = type_emb + t_emb  # [B, N, N, D_EDGE]

        # ── Graph Transformer 深度传播
        for layer in self.layers:
            x = layer(x, edge_attr, padding_mask)
        x = self.ln_out(x)

        # ── Node 解码
        shape_logits = self.decode_shape(x)
        width_logits = self.decode_width(x)
        coords_pred = self.decode_coords(x)

        # ── Edge 解码
        x_i = x.unsqueeze(2).expand(B, N, N, -1)
        x_j = x.unsqueeze(1).expand(B, N, N, -1)
        edge_raw = self.decode_edge(torch.cat([x_i, x_j], dim=-1))

        edge_type_logits = edge_raw[..., :NUM_EDGE_TYPES]
        edge_geom_pred = torch.sigmoid(edge_raw[..., NUM_EDGE_TYPES:])  # [t_u, t_v, angle_sin_raw, angle_cos_raw]
        # angle_sin/cos 用 sigmoid 映射到 [0,1]，再线性映射到 [-1,1] 匹配真实值域
        edge_t_pred = edge_geom_pred[..., :2]              # [B,N,N,2]
        angle_pred_raw = edge_geom_pred[..., 2:]           # [B,N,N,2] in [0,1]
        angle_pred = angle_pred_raw * 2.0 - 1.0            # 映射到 [-1,1]
        edge_preds = torch.cat([edge_type_logits, edge_t_pred, angle_pred], dim=-1)  # [B,N,N, 4+2+2]

        return shape_logits, width_logits, coords_pred, edge_preds


# ==========================================
# 🎯 Loss 设计
# ==========================================
def compute_graph_loss(preds, targets, mask):
    """
    L = L_node + λ1*L_edge + λ2*L_angle + λ3*L_junction + λ4*L_repulse

    L_node    = CE(shape) + CE(width) + λ_len*SmoothL1(||P3-P0||)
    L_edge    = CE(edge_type) + SmoothL1(t_u) + SmoothL1(t_v)
    L_angle   = SmoothL1(sin差) + SmoothL1(cos差)  ← 监督交互切线夹角
    L_junction = || B_i(t_u) - B_j(t_v) ||^2       ← 物理交点监督
    L_repulse = 节点互斥力防坐标坍塌
    """
    shape_logits, width_logits, coords_pred, edge_preds = preds
    gt_shapes, gt_widths, gt_coords, gt_edge_types, gt_edge_ts = targets

    valid_nodes = ~mask

    # ── 1. Node Loss：含笔画长度回归（平移不变，不引起均值坍塌）
    # GT 坐标已在 Dataset 中心化 ∈ [-0.5, 0.5]
    # 坐标本身不监督，只监督 ||P3-P0|| 长度（相对坐标差，不依赖绝对位置）
    loss_shape = F.cross_entropy(shape_logits[valid_nodes], gt_shapes[valid_nodes])
    loss_width = F.cross_entropy(width_logits[valid_nodes], gt_widths[valid_nodes])

    p0_pred = coords_pred[:, :, :2]   # Tanh 空间 [-1,1]
    p3_pred = coords_pred[:, :, 2:]

    # 笔画长度 = ||P3-P0||（平移不变：P3-P0 不受中心化影响）
    gt_p0 = gt_coords[:, :, :2]
    gt_p3 = gt_coords[:, :, 2:]
    pred_len = (p3_pred - p0_pred).norm(dim=-1)           # [B, N]
    gt_len   = (gt_p3 - gt_p0).norm(dim=-1)               # [B, N]
    loss_length = F.smooth_l1_loss(pred_len[valid_nodes], gt_len[valid_nodes])

    loss_node = loss_shape + loss_width + LAMBDA_LENGTH * loss_length

    # ── 2. Edge Loss
    valid_edges = (gt_edge_types > 0) & (~mask.unsqueeze(2)) & (~mask.unsqueeze(1))
    valid_pairs = (~mask.unsqueeze(2)) & (~mask.unsqueeze(1))

    edge_type_logits = edge_preds[..., :NUM_EDGE_TYPES]
    edge_t_pred = edge_preds[..., NUM_EDGE_TYPES:NUM_EDGE_TYPES+2]       # [t_u, t_v]
    angle_pred = edge_preds[..., NUM_EDGE_TYPES+2:NUM_EDGE_TYPES+4]      # [angle_sin, angle_cos]

    loss_edge_type = F.cross_entropy(
        edge_type_logits[valid_pairs], gt_edge_types[valid_pairs]
    )

    tu_pred = edge_t_pred[..., 0]   # sigmoid 输出 ∈ [0,1]
    tv_pred = edge_t_pred[..., 1]

    if valid_edges.sum() > 0:
        gt_tu = gt_edge_ts[..., 0][valid_edges]
        gt_tv = gt_edge_ts[..., 1][valid_edges]
        tu_p  = tu_pred[valid_edges]
        tv_p  = tv_pred[valid_edges]
        # 对于 E2E/T 类型 GT t 只有 0 和 1，用 BCE 更适合（输出层已经是 sigmoid）
        # X 类型 GT t 可以是任意小数，用 SmoothL1
        e2e_t_mask = (gt_edge_types[valid_edges] == 1)
        x_t_mask   = (gt_edge_types[valid_edges] == 2)
        t_t_mask   = (gt_edge_types[valid_edges] == 3)

        loss_t_parts = []
        if e2e_t_mask.sum() > 0:
            loss_t_parts.append(
                F.binary_cross_entropy(tu_p[e2e_t_mask], gt_tu[e2e_t_mask]) +
                F.binary_cross_entropy(tv_p[e2e_t_mask], gt_tv[e2e_t_mask])
            )
        if t_t_mask.sum() > 0:
            loss_t_parts.append(
                F.binary_cross_entropy(tu_p[t_t_mask], gt_tu[t_t_mask]) +
                F.binary_cross_entropy(tv_p[t_t_mask], gt_tv[t_t_mask])
            )
        if x_t_mask.sum() > 0:
            loss_t_parts.append(
                F.smooth_l1_loss(tu_p[x_t_mask], gt_tu[x_t_mask]) +
                F.smooth_l1_loss(tv_p[x_t_mask], gt_tv[x_t_mask])
            )
        loss_t = sum(loss_t_parts) / max(len(loss_t_parts), 1)
    else:
        loss_t = torch.tensor(0.0, device=coords_pred.device)

    loss_edge = loss_edge_type + LAMBDA_T * loss_t

    # ── 2b. L_angle: 交互切线夹角监紹（sin/cos 双通道 SmoothL1）
    # gt_edge_ts[:,:,:,4] = angle_sin, gt_edge_ts[:,:,:,5] = angle_cos
    gt_angle_sin = gt_edge_ts[..., 4]   # [B, N, N]
    gt_angle_cos = gt_edge_ts[..., 5]
    angle_sin_pred = angle_pred[..., 0]
    angle_cos_pred = angle_pred[..., 1]

    if valid_edges.sum() > 0:
        loss_angle = (
            F.smooth_l1_loss(angle_sin_pred[valid_edges], gt_angle_sin[valid_edges]) +
            F.smooth_l1_loss(angle_cos_pred[valid_edges], gt_angle_cos[valid_edges])
        )
    else:
        loss_angle = torch.tensor(0.0, device=coords_pred.device)

    # ── 3. L_junction: 物理交点吸合（在 Tanh 空间计算）
    # ⚠️ 必须用 GT t 值（gt_edge_ts[:,0]/[:,1]），而非预测 t——
    # 预测 t 在训练初期不准，用它反而会给坐标错误的梯度方向
    B, N, _ = gt_edge_types.shape
    p0_i = p0_pred.unsqueeze(2).expand(B, N, N, 2)
    p3_i = p3_pred.unsqueeze(2).expand(B, N, N, 2)
    p0_j = p0_pred.unsqueeze(1).expand(B, N, N, 2)
    p3_j = p3_pred.unsqueeze(1).expand(B, N, N, 2)

    gt_tu = gt_edge_ts[..., 0]   # [B, N, N]  GT t_u（交点在 Node_i 上的位置）
    gt_tv = gt_edge_ts[..., 1]   # [B, N, N]  GT t_v（交点在 Node_j 上的位置）

    pt_i = p0_i * (1 - gt_tu.unsqueeze(-1)) + p3_i * gt_tu.unsqueeze(-1)
    pt_j = p0_j * (1 - gt_tv.unsqueeze(-1)) + p3_j * gt_tv.unsqueeze(-1)

    dist_sq = (pt_i - pt_j).pow(2).sum(dim=-1)
    dist_l1 = (pt_i - pt_j).abs().sum(dim=-1)

    if valid_edges.sum() > 0:
        loss_junction = (dist_sq + 0.1 * dist_l1).masked_select(valid_edges).mean()
    else:
        loss_junction = torch.tensor(0.0, device=coords_pred.device)

    # ── 3b. L_coord_angle: 坐标角度约束
    # 约束两笔画在交点处切线方向满足 GT 角度（而非只约束 edge_preds 里的 angle_sin/cos）
    # 笔画切线方向 = (P3-P0) / ||P3-P0||（简化直线模型）
    vec_i = p3_pred - p0_pred                                    # [B, N, 2]
    vec_i_norm = vec_i / (vec_i.norm(dim=-1, keepdim=True) + 1e-6)

    vec_i_exp = vec_i_norm.unsqueeze(2).expand(B, N, N, 2)      # [B, N, N, 2]
    vec_j_exp = vec_i_norm.unsqueeze(1).expand(B, N, N, 2)      # [B, N, N, 2]

    # cos+sin 双通道：精确区分所有角度，解决 cos 偶函数的方向歧义
    # cos = dot(vi, vj)                 -- 区分 0°/180°
    # |sin| = |cross(vi, vj)|           -- 区分 0°/90°，取绝对値避免笔画方向符号歧义
    # GT angle_sin 始终是 sin(角度) >= 0（只存角度大小），所以必须用绝对値匹配
    cos_actual = (vec_i_exp * vec_j_exp).sum(dim=-1)              # [B, N, N]
    sin_actual = (vec_i_exp[..., 0] * vec_j_exp[..., 1]
                - vec_i_exp[..., 1] * vec_j_exp[..., 0]).abs()    # [B, N, N] |cross|

    gt_cos = gt_edge_ts[..., 5]   # angle_cos
    gt_sin = gt_edge_ts[..., 4]   # angle_sin (>= 0)

    if valid_edges.sum() > 0:
        loss_coord_angle = (
            F.smooth_l1_loss(cos_actual[valid_edges], gt_cos[valid_edges]) +
            F.smooth_l1_loss(sin_actual[valid_edges], gt_sin[valid_edges])
        )
    else:
        loss_coord_angle = torch.tensor(0.0, device=coords_pred.device)

    # ── 4. L_repulse: 节点互斥力（用平方距避免 sqrt 导数在 0 处爆炸）
    B, N, _ = gt_edge_types.shape
    mid_pts = (coords_pred[:, :, :2] + coords_pred[:, :, 2:]) / 2.0  # [B, N, 2]
    mid_i = mid_pts.unsqueeze(2).expand(B, N, N, 2)
    mid_j = mid_pts.unsqueeze(1).expand(B, N, N, 2)
    dist_sq_nodes = (mid_i - mid_j).pow(2).sum(-1)   # [B, N, N] 平方距离，导数永远有界
    upper_tri = torch.triu(torch.ones(N, N, dtype=torch.bool, device=coords_pred.device), diagonal=1)
    upper_tri = upper_tri.unsqueeze(0).expand(B, N, N)
    valid_pair = upper_tri & (~mask.unsqueeze(2)) & (~mask.unsqueeze(1))
    if valid_pair.sum() > 0:
        dists_sq = dist_sq_nodes[valid_pair]
        loss_repulse = F.relu(MIN_NODE_DIST_SQ - dists_sq).mean()
    else:
        loss_repulse = torch.tensor(0.0, device=coords_pred.device)

    total = (loss_node
             + LAMBDA_EDGE * loss_edge
             + LAMBDA_ANGLE * loss_angle
             + LAMBDA_JUNCTION * loss_junction
             + LAMBDA_COORD_ANGLE * loss_coord_angle
             + LAMBDA_REPULSE * loss_repulse)
    return total, loss_node, loss_edge, loss_angle, loss_junction, loss_repulse


# ==========================================
# 🚂 训练循环
# ==========================================
def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"🚀 拓扑驱动几何求解引擎 v2 (Device: {device})")
    print(f"   节点初始化: NodeIdEmb + TopoNodeEncoder（纯拓扑，无 GT 特征输入）")
    print(f"   Loss: L_node(shape+width+{LAMBDA_LENGTH}×length) + {LAMBDA_EDGE}×L_edge({LAMBDA_T}×L_t_bce+L_type) + {LAMBDA_ANGLE}×L_angle + {LAMBDA_JUNCTION}×L_junction + {LAMBDA_REPULSE}×L_repulse")

    dataset = FontCompleteGraphDataset(DATASET_FILE)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_complete_graphs
    )

    model = FontGraphGenerator().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = total_n = total_e = total_a = total_j = total_s = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for b_shapes, b_widths, b_coords, b_edge_types, b_edge_ts, b_mask in pbar:
            # shapes/widths/coords 只作为 GT 标签，不传入模型
            b_shapes = b_shapes.to(device)
            b_widths = b_widths.to(device)
            b_coords = b_coords.to(device)
            b_edge_types = b_edge_types.to(device)
            b_edge_ts = b_edge_ts.to(device)
            b_mask = b_mask.to(device)

            optimizer.zero_grad()

            # 🚀 模型只接收拓扑（edge_types + edge_ts），不接收任何节点 GT 特征
            preds = model(b_edge_types, b_edge_ts, b_mask)

            targets = (b_shapes, b_widths, b_coords, b_edge_types, b_edge_ts)
            loss, n_loss, e_loss, a_loss, j_loss, r_loss = compute_graph_loss(preds, targets, b_mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_n += n_loss.item()
            total_e += e_loss.item()
            total_a += a_loss.item()
            total_j += j_loss.item()
            total_s += r_loss.item()

            pbar.set_postfix({
                "Total": f"{loss.item():.3f}",
                "Node": f"{n_loss.item():.3f}",
                "Edge": f"{e_loss.item():.3f}",
                "Angle": f"{a_loss.item():.4f}",
                "Junc": f"{j_loss.item():.4f}",
                "Repul": f"{r_loss.item():.4f}",
            })

        steps = len(dataloader)
        print(
            f"📈 Epoch {epoch+1:3d} | "
            f"Total: {total_loss/steps:.4f} | "
            f"Node: {total_n/steps:.4f} | "
            f"Edge: {total_e/steps:.4f} | "
            f"Angle: {total_a/steps:.5f} | "
            f"Junc: {total_j/steps:.5f} | "
            f"Repul: {total_s/steps:.5f}"
        )

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 已保存至 {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
