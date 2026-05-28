import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass

# ==========================================
# 1. 高阶参数化配置中心
# ==========================================
@dataclass
class ModelConfig:
    # --- 7维数据配置 ---
    # 输入: [dx, dy, L, θ, κ, W, P]
    input_dim: int = 7         
    # 连续预测输出: [dx, dy, L, θ, κ, W] (去掉了状态 P)
    cont_out_dim: int = 6      
    # 状态分类: 0(画线), 1(结束 EOS)
    num_states: int = 2        

    # --- Transformer 架构 (极其轻量) ---
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.01

    # --- 训练超参数 ---
    batch_size: int = 256      # 序列大幅缩短，Batch 可以尽情开大！
    epochs: int = 100
    lr: float = 5e-3           # 参数化模型比较稳，可以稍微调高学习率
    weight_decay: float = 1e-4

cfg = ModelConfig()

# ==========================================
# 2. 参数化数据集解析与防护
# ==========================================
class ParametricStrokeDataset(Dataset):
    def __init__(self, data_path="omniglot_parametric_7d.pt"):
        self.data = torch.load(data_path)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # seq shape: (L, 7)
        seq = self.data[idx]["sequence"]
        
        inputs = seq[:-1].clone()
        targets_raw = seq[1:].clone()
        
        # 目标一：预测 6 个连续物理量 [dx, dy, L, θ, κ, W]
        targets_cont = targets_raw[:, :6]
        
        # 目标二：预测离散状态 P (0 或 1)
        targets_state = targets_raw[:, 6].long()
        
        return inputs, targets_cont, targets_state

def collate_fn(batch):
    inputs, targets_cont, targets_state = zip(*batch)
    
    # 🚨 防爆截断：参数化后，超过 50 笔的字几乎不存在，这里设为 100 绝对安全
    MAX_LEN = 100 
    inputs = [seq[:MAX_LEN] for seq in inputs]
    targets_cont = [seq[:MAX_LEN] for seq in targets_cont]
    targets_state = [seq[:MAX_LEN] for seq in targets_state]

    seq_lens = torch.tensor([len(seq) for seq in inputs])
    
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    targets_cont_padded = pad_sequence(targets_cont, batch_first=True, padding_value=0.0)
    targets_state_padded = pad_sequence(targets_state, batch_first=True, padding_value=-100)
    
    max_len = inputs_padded.size(1)
    padding_mask = torch.arange(max_len).expand(len(seq_lens), max_len) >= seq_lens.unsqueeze(1)
    
    return inputs_padded, targets_cont_padded, targets_state_padded, padding_mask

# ==========================================
# 3. 极简网络架构 (Direct Linear Encoding)
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
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class ParametricStrokeTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.cfg = config
        self.input_linear = nn.Linear(config.input_dim, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.nhead, 
            dim_feedforward=config.dim_feedforward, 
            dropout=config.dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=config.num_layers)
        
        # 6维物理头 + 2维分类头
        self.continuous_head = nn.Linear(config.d_model, config.cont_out_dim)
        self.state_head = nn.Linear(config.d_model, config.num_states)        

    def generate_causal_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, padding_mask=None):
        seq_len = src.size(1)
        x = self.input_linear(src)
        x = x * math.sqrt(self.cfg.d_model)
        x = self.pos_encoder(x)
        
        causal_mask = self.generate_causal_mask(seq_len).to(src.device)
        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        
        return self.continuous_head(out), self.state_head(out)

# ==========================================
# 4. 🚀 训练主循环 (含几何感知 Loss)
# ==========================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device} (启用 AMP 混合精度)")
    
    dataset = ParametricStrokeDataset("omniglot_parametric_7d.pt")
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, 
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    model = ParametricStrokeTransformer(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    mse_loss_fn = nn.MSELoss(reduction='none') 
    
    scaler = torch.cuda.amp.GradScaler() # 显存救星

    model.train()
    print(f"🚀 开始训练 Parametric 架构！(输入维度: {cfg.input_dim})")

    try:
        for epoch in range(cfg.epochs):
            epoch_loss, total_mse, total_geo, total_cat = 0.0, 0.0, 0.0, 0.0
            
            for inputs, targets_cont, targets_state, padding_mask in dataloader:
                inputs, targets_cont = inputs.to(device), targets_cont.to(device)
                targets_state, padding_mask = targets_state.to(device), padding_mask.to(device)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    pred_cont, pred_state_logits = model(inputs, padding_mask=padding_mask)
                    
                    # 1. 状态 Loss (结束符判断)
                    loss_cat = ce_loss_fn(pred_state_logits.view(-1, cfg.num_states), targets_state.view(-1))
                    
                    # 2. 基础数值 MSE Loss
                    loss_mse_matrix = mse_loss_fn(pred_cont, targets_cont)
                    
                    # ==========================================
                    # 🎯 3. 核心黑魔法：几何感知补偿 (Geometry-Aware Loss)
                    # ==========================================
                    # 提取参数: [dx, dy, L, θ, κ, W]
                    pred_dx, pred_dy = pred_cont[:,:,0], pred_cont[:,:,1]
                    pred_L, pred_theta = pred_cont[:,:,2], pred_cont[:,:,3]
                    
                    targ_dx, targ_dy = targets_cont[:,:,0], targets_cont[:,:,1]
                    targ_L, targ_theta = targets_cont[:,:,2], targets_cont[:,:,3]
                    
                    # 近似计算这笔画的“真实终点坐标”
                    # End_X = 起点偏移量(dx) + 长度(L) * cos(射出角θ)
                    pred_end_x = pred_dx + pred_L * torch.cos(pred_theta)
                    pred_end_y = pred_dy + pred_L * torch.sin(pred_theta)
                    
                    targ_end_x = targ_dx + targ_L * torch.cos(targ_theta)
                    targ_end_y = targ_dy + targ_L * torch.sin(targ_theta)
                    
                    # 计算端点漂移误差
                    loss_geo_matrix = (pred_end_x - targ_end_x)**2 + (pred_end_y - targ_end_y)**2
                    
                    # --- Mask 与归一化 ---
                    loss_mse_matrix[padding_mask] = 0.0
                    loss_geo_matrix[padding_mask] = 0.0
                    
                    valid_elements = (~padding_mask).sum()
                    loss_mse = loss_mse_matrix.sum() / (valid_elements * cfg.cont_out_dim + 1e-8)
                    loss_geo = loss_geo_matrix.sum() / (valid_elements + 1e-8)
                    
                    # 损失加权组合：几何补偿占据 50% 的重要性
                    loss = loss_mse + 0.5 * loss_geo + loss_cat
                
                # 混合精度反向传播
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                total_mse += loss_mse.item()
                total_geo += loss_geo.item()
                total_cat += loss_cat.item()
                
            num_batches = len(dataloader)
            print(f"Epoch [{epoch+1}/{cfg.epochs}] | "
                  f"Total: {epoch_loss/num_batches:.4f} | "
                  f"MSE: {total_mse/num_batches:.4f} | "
                  f"Geo: {total_geo/num_batches:.4f} | "
                  f"State: {total_cat/num_batches:.4f}")

    except KeyboardInterrupt:
        print("\n⚠️ 捕获到 Ctrl+C！正在紧急停止...")
        
    finally:
        save_path = "parametric_transformer.pth"
        torch.save(model.state_dict(), save_path)
        print(f"\n🎉 模型权重已安全保存至: {save_path}")

if __name__ == "__main__":
    train_model()