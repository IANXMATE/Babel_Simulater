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
NUM_EDGE_TYPES = 4 # 0:NONE(负样本), 1:E2E, 2:X, 3:T
MAX_NODES = 50     

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
        edge_ts = torch.zeros((num_nodes, num_nodes, 4), dtype=torch.float) # [t_u, t_v, t_diff, t_prod]
        
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
# 🧠 核心架构：Edge-Symmetric Graph Transformer
# ==========================================
class EdgeAwareGraphAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_edge):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 将融合后的 d_edge 投射到 num_heads，作为强力的注意力偏置
        self.edge_proj = nn.Linear(d_edge, num_heads)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, edge_attr, padding_mask):
        B, N, _ = x.size()
        
        Q = self.W_q(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)

        # 基础 Query-Key 相似度矩阵
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) 

        # 🚀 注入对称破缺的拓扑偏置 (Edge Bias)
        edge_bias = self.edge_proj(edge_attr).permute(0, 3, 1, 2) 
        scores = scores + edge_bias

        # 掩码屏蔽 Pad 空节点 (消除越界注意力的梯度毒药)
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
    def __init__(self):
        super().__init__()
        
        # 1. Node 输入编码 (由于按 bezier_id 严格排序，node_id 具有全局稳定语义)
        self.node_id_emb = nn.Embedding(MAX_NODES, D_MODEL) 
        
        # 2. Edge 输入编码 (离散类别 + 连续对称特征融合)
        self.edge_type_emb = nn.Embedding(NUM_EDGE_TYPES, D_EDGE)
        self.edge_t_mlp = nn.Sequential(
            nn.Linear(4, D_EDGE), # 传入 [t_u, t_v, t_diff, t_prod]
            nn.GELU(),
            nn.Linear(D_EDGE, D_EDGE)
        )
        
        # 3. 消息传递层
        self.layers = nn.ModuleList([
            GraphTransformerBlock(D_MODEL, N_HEADS, D_EDGE, DROPOUT) for _ in range(N_LAYERS)
        ])
        
        self.ln_out = nn.LayerNorm(D_MODEL)
        
        # 4. Node Geometry Decoder (基于拓扑求解出实体几何属性)
        self.decode_shape = nn.Linear(D_MODEL, NUM_SHAPES)
        self.decode_width = nn.Linear(D_MODEL, NUM_WIDTHS)
        self.decode_coords = nn.Sequential(
            nn.Linear(D_MODEL, 128),
            nn.GELU(),
            nn.Linear(128, 4),
            nn.Sigmoid() # 物理约束：强制输出在画布 [0, 1) 内
        )
        
        # ⚠️ 注意：此时不再需要 decode_edge_t，因为 T 值本身就是图结构的先决条件（Input Prior）！

    def forward(self, edge_types, edge_ts, padding_mask):
        B, N, _ = edge_types.size()
        device = edge_types.device
        
        # Node 初始化
        node_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        x = self.node_id_emb(node_ids) 
        
        # 🚀 融合 Edge Features: 结合类别拓扑与浮点几何约束
        type_emb = self.edge_type_emb(edge_types)
        t_emb = self.edge_t_mlp(edge_ts)
        edge_attr = type_emb + t_emb # [B, N, N, D_EDGE]
        
        # 深度图传播
        for layer in self.layers:
            x = layer(x, edge_attr, padding_mask)
            
        x = self.ln_out(x)
        
        # 纯 Node 解码 (方程求解结果)
        shape_logits = self.decode_shape(x)
        width_logits = self.decode_width(x)
        coords_pred = self.decode_coords(x)
        
        return shape_logits, width_logits, coords_pred

def compute_graph_loss(preds, targets, mask):
    shape_logits, width_logits, coords_pred = preds
    gt_shapes, gt_widths, gt_coords, gt_edge_types, gt_edge_ts = targets
    
    valid_nodes = ~mask 
    
    # --- 1. Node Loss ---
    loss_shape = F.cross_entropy(shape_logits[valid_nodes], gt_shapes[valid_nodes])
    loss_coords = F.mse_loss(coords_pred[valid_nodes], gt_coords[valid_nodes])
    
    # --- 2. 改进版 Junction Loss (加入绝对距离约束) ---
    B, N, _ = gt_edge_types.shape
    p0 = coords_pred[:, :, :2]; p3 = coords_pred[:, :, 2:]
    
    # 构建坐标广播矩阵
    p0_i = p0.unsqueeze(2).expand(B, N, N, 2); p3_i = p3.unsqueeze(2).expand(B, N, N, 2)
    p0_j = p0.unsqueeze(1).expand(B, N, N, 2); p3_j = p3.unsqueeze(1).expand(B, N, N, 2)
    
    tu = gt_edge_ts[:, :, :, 0:1]; tv = gt_edge_ts[:, :, :, 1:2]
    pt_i = p0_i * (1 - tu) + p3_i * tu
    pt_j = p0_j * (1 - tv) + p3_j * tv
    
    # 【重点】计算距离：不仅仅是平方和，加入 L1 距离增加初期约束力
    dist_sq = (pt_i - pt_j).pow(2).sum(dim=-1)
    dist_l1 = (pt_i - pt_j).abs().sum(dim=-1)
    
    valid_edges = (gt_edge_types > 0) & (~mask.unsqueeze(2)) & (~mask.unsqueeze(1))
    
    if valid_edges.sum() > 0:
        # 同时惩罚平方距离和绝对距离，解决初期梯度消失问题
        loss_junction = (dist_sq + 0.1 * dist_l1).masked_select(valid_edges).mean()
    else:
        loss_junction = torch.tensor(0.0, device=coords_pred.device)

    # 【重要】调整权重，确保 junction loss 在量级上不小于 coordinate loss
    return loss_shape * 1.0 + loss_coords * 10.0 + loss_junction * 100.0, loss_junction


# ==========================================
# 🚂 完备图模型训练循环
# ==========================================
def main():
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    
    print(f"🚀 启动 Graph Transformer (Canonical 拓扑-几何) 求解引擎 (Device: {device})...")
    
    dataset = FontCompleteGraphDataset(DATASET_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_complete_graphs)

    model = FontGraphGenerator().to(device)
    # Graph 模型使用相对较低的 weight_decay 以保护结构化特征
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_j_loss = 0, 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for b_shapes, b_widths, b_coords, b_edge_types, b_edge_ts, b_mask in pbar:
            b_shapes, b_widths, b_coords = b_shapes.to(device), b_widths.to(device), b_coords.to(device)
            b_edge_types, b_edge_ts = b_edge_types.to(device), b_edge_ts.to(device)
            b_mask = b_mask.to(device)
            
            optimizer.zero_grad()
            
            # 🚀 推理流变迁：输入包含了 Topology Prior（网络无需去猜类型和 T，只负责算几何！）
            preds = model(b_edge_types, b_edge_ts, b_mask)
            
            targets = (b_shapes, b_widths, b_coords, b_edge_types, b_edge_ts)
            
            loss, j_loss = compute_graph_loss(preds, targets, b_mask)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_j_loss += j_loss.item()
            
            pbar.set_postfix({
                "Loss": f"{loss.item():.3f}", 
                "J_Loss": f"{j_loss.item():.4f}" # 重点观测这个值，它代表物理吸合程度
            })
            
        print(f"📈 Epoch {epoch+1} | Avg Total: {total_loss/len(dataloader):.4f} | Avg Junction: {total_j_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 纯粹图模型已保存至 {MODEL_SAVE_PATH}！")

if __name__ == "__main__":
    main()