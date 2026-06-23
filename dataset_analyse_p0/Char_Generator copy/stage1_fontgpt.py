import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FontGPTConfig:
    def __init__(self):
        self.d_model = 256
        self.n_heads = 8
        self.n_layers = 6
        self.dropout = 0.01
        self.shape_vocab_size = 2000  
        self.morph_vocab_size = 4     # 🌟 形态 ID 词表 (0, 1, 2, 3)
        self.width_vocab_size = 4    
        self.grid_bins = 32           
        self.topo_relations = 6       


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
        
        # 1. 用于影响 Attention 概率分布的偏置
        self.topo_bias_embedding = nn.Embedding(config.topo_relations, config.n_heads)
        # 🌟 2. 新增：用于突破 Softmax 洗牌，直接注入特征的 Value Embedding
        self.topo_value_embedding = nn.Embedding(config.topo_relations, self.d_k)

    def forward(self, q, k, v, topo_matrix, mask=None):
        bs = q.size(0); seq_len = q.size(1)
        
        q = self.q_linear(q).view(bs, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if topo_matrix is not None:
            topo_bias = self.topo_bias_embedding(topo_matrix).permute(0, 3, 1, 2)
            scores = scores + topo_bias 
            
        if mask is not None: 
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        
        # 🌟 核心魔法：拓扑值注入 (Topology Value Injection)
        if topo_matrix is not None:
            # 获取拓扑图的物理特征向量: [Batch, Seq_Q, Seq_K, d_k]
            topo_val = self.topo_value_embedding(topo_matrix)
            
            # 第一部分：正常的视觉空间特征聚合
            out_v = torch.matmul(attn, v)
            
            # 第二部分：利用 Attention 概率，聚合拓扑连接特征！
            # b: batch, h: heads, i: seq_q, j: seq_k, d: d_k
            out_topo = torch.einsum('bhij,bijd->bhid', attn, topo_val)
            
            # 空间与拓扑在隐空间完美融合
            output = out_v + out_topo
        else:
            output = torch.matmul(attn, v)
            
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)
        return self.out(output)


class FontGPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = TopologyAwareAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(nn.Linear(config.d_model, config.d_model * 4), nn.GELU(), nn.Linear(config.d_model * 4, config.d_model), nn.Dropout(config.dropout))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, topo_matrix, mask=None):
        x = self.norm1(x + self.dropout(self.attention(x, x, x, topo_matrix, mask)))
        x = self.norm2(x + self.ffn(x))
        return x

class FontGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # --- 输入解耦表征 ---
        self.shape_emb = nn.Embedding(config.shape_vocab_size, config.d_model)
        self.morph_emb = nn.Embedding(config.morph_vocab_size, config.d_model) # 🌟 独立的形态 Embedding
        self.width_emb = nn.Embedding(config.width_vocab_size, config.d_model)
        self.cell_emb = nn.Embedding(config.grid_bins, config.d_model)
        self.offset_proj = nn.Linear(2, config.d_model) 

        self.fusion = nn.Linear(config.d_model * 7, config.d_model) # 🌟 6 变 7，拼接了 morph
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, config.d_model)) 
        
        self.layers = nn.ModuleList([FontGPTBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # --- 多头输出预测 ---
        self.head_shape = nn.Linear(config.d_model, config.shape_vocab_size)
        self.head_morph = nn.Linear(config.d_model, config.morph_vocab_size) # 🌟 独立的形态预测头
        self.head_width = nn.Linear(config.d_model, config.width_vocab_size)
        self.head_p0_cx = nn.Linear(config.d_model, config.grid_bins)
        self.head_p0_cy = nn.Linear(config.d_model, config.grid_bins)
        self.head_p3_cx = nn.Linear(config.d_model, config.grid_bins)
        self.head_p3_cy = nn.Linear(config.d_model, config.grid_bins)
        
        self.head_p0_offset = nn.Sequential(nn.Linear(config.d_model, 64), nn.ReLU(), nn.Linear(64, 2))
        self.head_p3_offset = nn.Sequential(nn.Linear(config.d_model, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, shape_tokens, morph_tokens, width_tokens, p0_cells_x, p0_cells_y, p0_offsets, p3_cells_x, p3_cells_y, p3_offsets, topo_matrix, mask=None):
        s_emb = self.shape_emb(shape_tokens)
        m_emb = self.morph_emb(morph_tokens) # 🌟 提取形态特征
        w_emb = self.width_emb(width_tokens)
        p0_cx_emb = self.cell_emb(p0_cells_x)
        p0_cy_emb = self.cell_emb(p0_cells_y)
        p3_cx_emb = self.cell_emb(p3_cells_x)
        p3_cy_emb = self.cell_emb(p3_cells_y)
        
        p0_cell_fused = p0_cx_emb + p0_cy_emb
        p3_cell_fused = p3_cx_emb + p3_cy_emb
        p0_off_emb = self.offset_proj(p0_offsets)
        p3_off_emb = self.offset_proj(p3_offsets)
        
        # 将 morph_emb 拼接进隐空间
        x = torch.cat([s_emb, m_emb, w_emb, p0_cell_fused, p0_off_emb, p3_cell_fused, p3_off_emb], dim=-1)
        x = self.fusion(x)
        
        # 🌟 删除了 imm_topo 的相关代码，直接加上位置编码
        x = x + self.pos_encoder[:, :shape_tokens.size(1), :]
        
        for layer in self.layers: x = layer(x, topo_matrix, mask)
        x = self.ln_f(x)
        
        return {
            "logits_shape": self.head_shape(x),
            "logits_morph": self.head_morph(x), # 🌟 输出形态预测
            "logits_width": self.head_width(x),
            "logits_p0_cx": self.head_p0_cx(x), "logits_p0_cy": self.head_p0_cy(x),
            "pred_p0_offset": torch.tanh(self.head_p0_offset(x)) * 0.5,
            "logits_p3_cx": self.head_p3_cx(x), "logits_p3_cy": self.head_p3_cy(x),
            "pred_p3_offset": torch.tanh(self.head_p3_offset(x)) * 0.5
        }