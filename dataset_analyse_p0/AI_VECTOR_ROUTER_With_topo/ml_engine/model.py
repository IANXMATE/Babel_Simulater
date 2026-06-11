import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def extract_pairwise_features(path_a, path_b, dt_map):
    """提取两个笔画片段之间的几何特征，输入给模型预测是否应该合并"""
    ends_a, ends_b = [path_a[0], path_a[-1]], [path_b[0], path_b[-1]]
    min_dist, best_a_idx, best_b_idx = float('inf'), 0, 0
    for i, ea in enumerate(ends_a):
        for j, eb in enumerate(ends_b):
            dist = np.linalg.norm(ea - eb)
            if dist < min_dist: 
                min_dist, best_a_idx, best_b_idx = dist, i, j
                
    dist_feat = np.clip(min_dist / 10.0, 0, 1)
    step = min(4, len(path_a)-1, len(path_b)-1)
    if step < 1: step = 1
    
    vec_a = path_a[step] - path_a[0] if best_a_idx == 0 else path_a[-1-step] - path_a[-1]
    vec_b = path_b[step] - path_b[0] if best_b_idx == 0 else path_b[-1-step] - path_b[-1]
    norm_a, norm_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
    
    cos_theta = 0 if norm_a < 1e-5 or norm_b < 1e-5 else np.dot(vec_a, vec_b) / (norm_a * norm_b)
    len_ratio = min(len(path_a), len(path_b)) / max(len(path_a), len(path_b))
    
    w_a = dt_map[int(ends_a[best_a_idx][0]), int(ends_a[best_a_idx][1])]
    w_b = dt_map[int(ends_b[best_b_idx][0]), int(ends_b[best_b_idx][1])]
    w_ratio = min(w_a, w_b) / (max(w_a, w_b) + 1e-5)
    
    return [dist_feat, cos_theta, len_ratio, w_ratio]


class GraphEditorTransformer(nn.Module):
    def __init__(self, feature_dim=8, hidden_dim=128, n_heads=4, n_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. 节点嵌入 (将 N*8 的几何特征映射到高维空间)
        self.edge_embedding = nn.Linear(feature_dim, hidden_dim)
        
        # 2. Graph Transformer 编码器层
        # 注意：实际工程中，为了注入 Attention Bias (+2, -1)，
        # 我们通常会自定义 Attention 层，这里用标准层做演示
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # ==========================================
        # 🌟 华丽的 Actor Heads (分类与指针输出)
        # ==========================================
        
        # 头 1: Action Type Head (5分类: Merge, Delete, Split, AddDot, Done)
        # 读入全局特征，决定这一步干什么
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        
        # 头 2: Target Pointer Head (指针网络)
        # 读入全局意图 Query 和所有边的 Key，计算 Attention Score 选出要操作的边
        self.pointer_query = nn.Linear(hidden_dim, hidden_dim)
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x_feat, padding_mask=None, attn_bias=None):
        """
        x_feat: (Batch, Max_N, Feature_Dim) - 当前所有的边特征
        padding_mask: (Batch, Max_N) - 屏蔽掉不存在的边 (因为不同字边数不同)
        attn_bias: (Batch, Max_N, Max_N) - 我们在 data_builder 算好的拓扑连接矩阵
        """
        B, N, D = x_feat.shape
        
        # 1. 提取高维 Token
        tokens = self.edge_embedding(x_feat) # (B, N, hidden_dim)
        
        # 2. 全局信息交互 (让左偏旁看见右偏旁)
        # 将 attn_bias 融入 transformer 内部 (伪代码示意，PyTorch 中可传入 mask)
        encoded_tokens = self.transformer(tokens, src_key_padding_mask=padding_mask)
        
        # 3. 提取全局图特征 (Global Graph Context)，常用 Global Average Pooling
        # 屏蔽掉 padding 的部分求平均
        if padding_mask is not None:
            active_tokens = encoded_tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            valid_counts = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
            global_context = active_tokens.sum(dim=1) / valid_counts
        else:
            global_context = encoded_tokens.mean(dim=1) # (B, hidden_dim)
            
        # --- Head 1: 预测动作类型 ---
        type_logits = self.type_head(global_context) # (B, 5)
        
        # --- Head 2: 预测目标边 (Pointer) ---
        # 用全局特征作为 Query，去寻找图中最应该被操作的那条边 Key
        query = self.pointer_query(global_context).unsqueeze(1) # (B, 1, hidden_dim)
        keys = self.pointer_key(encoded_tokens)                 # (B, N, hidden_dim)
        
        # 内积计算打分
        pointer_logits = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) # (B, N)
        
        # 把 padding 的边打入冷宫 (-inf)
        if padding_mask is not None:
            pointer_logits = pointer_logits.masked_fill(padding_mask, float('-inf'))
            
        return type_logits, pointer_logits