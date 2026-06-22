import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# ⚙️ FontGPT 模型配置
# ==========================================
class FontGPTConfig:
    def __init__(self):
        self.d_model = 256
        self.n_heads = 8
        self.n_layers = 6
        self.dropout = 0.01
        
        # 词表大小配置 (根据你 step1 和 step3 的输出而定)
        self.shape_vocab_size = 32  # Shape Token 的最大数量
        self.width_vocab_size = 4    # 宽度量化级数
        self.grid_bins = 32           # Cell 网格大小
        self.topo_relations = 6       # 拓扑关系种类 (0:无, 1:E2E, 2:T客, 3:T主, 4:X等)

# ==========================================
# 🧠 1. 拓扑感知注意力层 (Graphormer 核心机制)
# ==========================================
class TopologyAwareAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_k = config.d_model // config.n_heads
        
        self.q_linear = nn.Linear(config.d_model, config.d_model)
        self.k_linear = nn.Linear(config.d_model, config.d_model)
        self.v_linear = nn.Linear(config.d_model, config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        
        # 🌟 杀手锏：拓扑关系偏置表征
        # 它将 0~5 的离散拓扑关系，映射为一个标量偏置，直接加到 Attention Logits 上！
        self.topo_bias_embedding = nn.Embedding(config.topo_relations, config.n_heads)

    def forward(self, q, k, v, topo_matrix, mask=None):
        bs = q.size(0)
        seq_len = q.size(1)
        
        # 线性变换并划分多头 -> [bs, n_heads, seq_len, d_k]
        q = self.q_linear(q).view(bs, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 标准 QK^T 注意力分数 -> [bs, n_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 🌟 注入拓扑先验 (Topology Bias Injection)
        # topo_matrix: [bs, seq_len, seq_len]
        # topo_bias: [bs, seq_len, seq_len, n_heads] -> permute -> [bs, n_heads, seq_len, seq_len]
        if topo_matrix is not None:
            topo_bias = self.topo_bias_embedding(topo_matrix).permute(0, 3, 1, 2)
            scores = scores + topo_bias # 强迫模型看向那些有拓扑约束的笔画！
            
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        # 拼接多头并输出
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)
        return self.out(output)

# ==========================================
# 🧱 2. Transformer Block
# ==========================================
class FontGPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = TopologyAwareAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
            nn.Dropout(config.dropout)
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, topo_matrix, mask=None):
        # 带有拓扑偏置的 Self-Attention
        attn_out = self.attention(x, x, x, topo_matrix, mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

# ==========================================
# 🚀 3. 终极架构：FontGPT Backbone
# ==========================================
class FontGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # --- 输入层：解耦表征融合 (Decoupled Embedding Fusion) ---
        self.shape_emb = nn.Embedding(config.shape_vocab_size, config.d_model)
        self.width_emb = nn.Embedding(config.width_vocab_size, config.d_model)
        self.cell_emb = nn.Embedding(config.grid_bins, config.d_model)
        # 连续坐标偏移采用 Linear 投影
        self.offset_proj = nn.Linear(2, config.d_model) 
        
        # 融合降维
        self.fusion = nn.Linear(config.d_model * 6, config.d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, config.d_model)) # 简单绝对位置编码
        
        # --- 骨干层：拓扑感知 Transformer ---
        self.layers = nn.ModuleList([FontGPTBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # --- 输出层：多头预测 (Multi-Head Predictors) ---
        # 1. 离散类别预测 (Cross Entropy)
        self.head_shape = nn.Linear(config.d_model, config.shape_vocab_size)
        self.head_width = nn.Linear(config.d_model, config.width_vocab_size)
        self.head_p0_cx = nn.Linear(config.d_model, config.grid_bins)
        self.head_p0_cy = nn.Linear(config.d_model, config.grid_bins)
        self.head_p3_cx = nn.Linear(config.d_model, config.grid_bins)
        self.head_p3_cy = nn.Linear(config.d_model, config.grid_bins)
        
        # 2. 连续坐标预测 (MSE)
        self.head_p0_offset = nn.Sequential(nn.Linear(config.d_model, 64), nn.ReLU(), nn.Linear(64, 2))
        self.head_p3_offset = nn.Sequential(nn.Linear(config.d_model, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, 
                shape_tokens, width_tokens, 
                p0_cells_x, p0_cells_y, p0_offsets,
                p3_cells_x, p3_cells_y, p3_offsets,
                topo_matrix, mask=None):
        """
        前向传播计算图
        """
        bs, seq_len = shape_tokens.size()
        
        # 1. 提取所有特征向量
        s_emb = self.shape_emb(shape_tokens)
        w_emb = self.width_emb(width_tokens)
        p0_cx_emb = self.cell_emb(p0_cells_x)
        p0_cy_emb = self.cell_emb(p0_cells_y)
        p3_cx_emb = self.cell_emb(p3_cells_x)
        p3_cy_emb = self.cell_emb(p3_cells_y)
        
        # 合并 Cell
        p0_cell_fused = p0_cx_emb + p0_cy_emb
        p3_cell_fused = p3_cx_emb + p3_cy_emb
        
        p0_off_emb = self.offset_proj(p0_offsets)
        p3_off_emb = self.offset_proj(p3_offsets)
        
        # 2. 特征大融合 (Concat -> Linear)
        x = torch.cat([s_emb, w_emb, p0_cell_fused, p0_off_emb, p3_cell_fused, p3_off_emb], dim=-1)
        x = self.fusion(x)
        
        # 加上序列位置编码
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # 3. 经过带有图拓扑偏置的 Transformer 层
        for layer in self.layers:
            x = layer(x, topo_matrix, mask)
        x = self.ln_f(x)
        
        # 4. 多头输出拆解
        logits_shape = self.head_shape(x)
        logits_width = self.head_width(x)
        
        logits_p0_cx = self.head_p0_cx(x)
        logits_p0_cy = self.head_p0_cy(x)
        pred_p0_offset = torch.tanh(self.head_p0_offset(x)) * 0.5 # 约束 offset 在 [-0.5, 0.5]
        
        logits_p3_cx = self.head_p3_cx(x)
        logits_p3_cy = self.head_p3_cy(x)
        pred_p3_offset = torch.tanh(self.head_p3_offset(x)) * 0.5
        
        return {
            "logits_shape": logits_shape,
            "logits_width": logits_width,
            "logits_p0_cx": logits_p0_cx, "logits_p0_cy": logits_p0_cy,
            "pred_p0_offset": pred_p0_offset,
            "logits_p3_cx": logits_p3_cx, "logits_p3_cy": logits_p3_cy,
            "pred_p3_offset": pred_p3_offset
        }

# ==========================================
# 🧪 测试模型连通性
# ==========================================
if __name__ == "__main__":
    print("🤖 正在实例化 FontGPT 模型...")
    config = FontGPTConfig()
    model = FontGPT(config)
    
    # 模拟 Batch=2, Seq_len=5 的一个 Mini-batch 数据
    bs, seq_len = 2, 5
    dummy_shape = torch.randint(0, config.shape_vocab_size, (bs, seq_len))
    dummy_width = torch.randint(0, config.width_vocab_size, (bs, seq_len))
    
    dummy_p0_cx = torch.randint(0, config.grid_bins, (bs, seq_len))
    dummy_p0_cy = torch.randint(0, config.grid_bins, (bs, seq_len))
    dummy_p0_off = torch.rand((bs, seq_len, 2)) - 0.5
    
    dummy_p3_cx = torch.randint(0, config.grid_bins, (bs, seq_len))
    dummy_p3_cy = torch.randint(0, config.grid_bins, (bs, seq_len))
    dummy_p3_off = torch.rand((bs, seq_len, 2)) - 0.5
    
    # 模拟 Graphormer 拓扑矩阵 (0~5的离散关系)
    dummy_topo = torch.randint(0, config.topo_relations, (bs, seq_len, seq_len))
    
    print("⚡ 执行 Forward Pass...")
    outputs = model(
        dummy_shape, dummy_width, 
        dummy_p0_cx, dummy_p0_cy, dummy_p0_off,
        dummy_p3_cx, dummy_p3_cy, dummy_p3_off,
        dummy_topo
    )
    
    print("✅ Forward 成功！输出维度检查：")
    print(f"Shape Logits: {outputs['logits_shape'].shape} (期待: [2, 5, 2000])")
    print(f"P0 CX Logits: {outputs['logits_p0_cx'].shape} (期待: [2, 5, 32])")
    print(f"P0 Offset Pred: {outputs['pred_p0_offset'].shape} (期待: [2, 5, 2])")