import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ==========================================
# 1. 极坐标数据集与特征工程
# ==========================================
class PolarStrokeDataset(Dataset):
    def __init__(self, data_path="multilingual_polar_trajectories.pt"):
        # 加载预处理好的轨迹数据
        self.data = torch.load(data_path)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # seq shape: (L, 4) -> [stroke_id, d, delta_theta, delta_w]
        seq = self.data[idx]["sequence"]
        
        # 拆分输入和目标 (自回归偏移 1 个步长)
        inputs = seq[:-1].clone()
        targets_raw = seq[1:].clone()
        
        # --- 核心特征工程：提取连续值目标 ---
        targets_cont = targets_raw[:, 1:4] # [d, delta_theta, delta_w]
        
        # --- 核心特征工程：构建分类状态目标 ---
        curr_id = inputs[:, 0]
        next_id = targets_raw[:, 0]
        
        # 状态机：0=继续画(同ID), 1=换新笔画(ID增加), 2=结束符(EOS, ID=-1)
        targets_state = torch.zeros_like(curr_id, dtype=torch.long)
        targets_state[(next_id > curr_id) & (next_id != -1)] = 1
        targets_state[next_id == -1] = 2
        
        return inputs, targets_cont, targets_state

def collate_fn(batch):
    """
    处理变长序列的批次拼接与掩码生成
    """
    inputs, targets_cont, targets_state = zip(*batch)
    
    # 记录每个序列的真实长度
    seq_lens = torch.tensor([len(seq) for seq in inputs])
    
    # 补齐到 Batch 内最大长度 (默认补 0.0)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    targets_cont_padded = pad_sequence(targets_cont, batch_first=True, padding_value=0.0)
    # 分类标签通常用 -100 补齐，告诉 CrossEntropy 忽略这些位置
    targets_state_padded = pad_sequence(targets_state, batch_first=True, padding_value=-100)
    
    # 生成 Transformer 需要的 padding_mask (True 表示补齐的无用位置，需屏蔽)
    max_len = inputs_padded.size(1)
    padding_mask = torch.arange(max_len).expand(len(seq_lens), max_len) >= seq_lens.unsqueeze(1)
    
    return inputs_padded, targets_cont_padded, targets_state_padded, padding_mask

# ==========================================
# 2. 混合特征编码器与位置编码
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class PolarStrokeTransformer(nn.Module):
    def __init__(self, max_strokes=51, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 特征拆分处理：离散特征用 Embedding，连续特征用 Linear
        # 强制将 d_model 平分，比如 256 拆成 128 给编号，128 给物理坐标
        self.embed_dim = d_model // 2
        self.cont_dim = d_model - self.embed_dim
        
        self.stroke_embedding = nn.Embedding(max_strokes, self.embed_dim)
        self.continuous_linear = nn.Linear(3, self.cont_dim)
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # 输出头
        self.continuous_head = nn.Linear(d_model, 3) # 预测下一次的 [d, theta, w]
        self.state_head = nn.Linear(d_model, 3)      # 预测状态机 [继续, 换笔, 结束]

    def generate_causal_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, padding_mask=None):
        # src: (Batch, SeqLen, 4)
        seq_len = src.size(1)
        
        # 1. 拆分特征
        stroke_ids = src[:, :, 0].long()
        # 将 EOS (-1) 映射为 Embedding 词表里的最后一个 ID (例如 50)
        stroke_ids[stroke_ids == -1] = 50 
        stroke_ids = torch.clamp(stroke_ids, 0, 50)
        
        cont_features = src[:, :, 1:4]
        
        # 2. 分别编码并拼接
        embed_out = self.stroke_embedding(stroke_ids)          # (B, L, 128)
        cont_out = self.continuous_linear(cont_features)       # (B, L, 128)
        x = torch.cat([embed_out, cont_out], dim=-1)           # (B, L, 256)
        
        # 3. 乘以放缩因子并加入位置编码
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # 4. 因果掩码 (防止偷看未来)
        causal_mask = self.generate_causal_mask(seq_len).to(src.device)
        
        # 5. Transformer 处理 (同时接受因果掩码和补齐掩码)
        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        
        return self.continuous_head(out), self.state_head(out)

# ==========================================
# 3. 训练主循环与精确掩码 Loss
# ==========================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 超参设定
    batch_size = 128
    epochs = 50
    lr = 3e-4
    
    # 加载数据集
    dataset = PolarStrokeDataset("multilingual_polar_trajectories.pt")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    model = PolarStrokeTransformer().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 损失函数配置
    # 分类头：忽略 -100 的 padding 标签
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    # 连续头：不自动求平均，以便后续手动剔除 padding 位置
    mse_loss_fn = nn.MSELoss(reduction='none') 
    
    # 损失权重 (角度与距离非常难学，权重拉高)
    w_cont = 10.0
    w_cat = 1.0

    model.train()
    print("🚀 开始极坐标残差架构训练...")

    for epoch in range(epochs):
        epoch_loss, total_cont, total_cat = 0.0, 0.0, 0.0
        
        for batch_idx, (inputs, targets_cont, targets_state, padding_mask) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets_cont = targets_cont.to(device)
            targets_state = targets_state.to(device)
            padding_mask = padding_mask.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播 (传入 padding_mask 阻断注意力分配给空白区)
            pred_cont, pred_state_logits = model(inputs, padding_mask=padding_mask)
            
            # --- 分类 Loss ---
            loss_cat = ce_loss_fn(pred_state_logits.view(-1, 3), targets_state.view(-1))
            
            # --- 连续值 Loss (需手动剔除 padding) ---
            loss_cont_matrix = mse_loss_fn(pred_cont, targets_cont) # Shape: (B, L, 3)
            # padding_mask 为 True 的地方，Loss 强制清零
            loss_cont_matrix[padding_mask] = 0.0
            # 计算有效元素的平均值
            valid_elements = (~padding_mask).sum() * 3 
            loss_cont = loss_cont_matrix.sum() / (valid_elements + 1e-8)
            
            # 融合总 Loss
            loss = (w_cont * loss_cont) + (w_cat * loss_cat)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            total_cont += loss_cont.item()
            total_cat += loss_cat.item()
            
        # 每个 Epoch 打印状态
        num_batches = len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Total Loss: {epoch_loss/num_batches:.4f} | "
              f"Cont(MSE): {total_cont/num_batches:.4f} | "
              f"State(CE): {total_cat/num_batches:.4f}")

    # 保存模型
    save_path = "polar_transformer.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n🎉 极坐标残差模型训练完成！权重已保存至: {save_path}")

if __name__ == "__main__":
    train_model()