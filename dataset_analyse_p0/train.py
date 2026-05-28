import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass

# ==========================================
# 1. 架构极简化的配置中心
# ==========================================
@dataclass
class ModelConfig:
    # --- 全新 5 维数据配置 ---
    input_dim: int = 5         # 输入: [dx, dy, p1, p2, p3]
    cont_out_dim: int = 2      # 连续预测输出: [dx, dy]
    num_states: int = 3        # 状态分类输出: 0(画线), 1(悬空), 2(结束)

    # --- Transformer 架构 ---
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 256
    dropout: float = 0.02

    # --- 训练超参数 ---
    batch_size: int = 128      # 5维数据运算更快，Batch可以开大点
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-4

cfg = ModelConfig()

# ==========================================
# 2. 5 维直角坐标数据集解析
# ==========================================
class CartesianStrokeDataset(Dataset):
    def __init__(self, data_path="omniglot_cartesian_5d.pt"):
        self.data = torch.load(data_path)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # seq shape: (L, 5) -> [dx, dy, p1, p2, p3]
        seq = self.data[idx]["sequence"]
        
        # 错位预测：输入前 L-1 步，预测后 L-1 步
        inputs = seq[:-1].clone()
        targets_raw = seq[1:].clone()
        
        # 目标一：预测下一个相对位移 [dx, dy]
        targets_cont = targets_raw[:, 0:2]
        
        # 目标二：将 [p1, p2, p3] 独热编码还原为类别索引 (0, 1, 2)
        # argmax: [1, 0, 0]->0 (画线), [0, 1, 0]->1 (悬空), [0, 0, 1]->2 (结束)
        targets_state = torch.argmax(targets_raw[:, 2:5], dim=1)
        
        return inputs, targets_cont, targets_state

def collate_fn(batch):
    inputs, targets_cont, targets_state = zip(*batch)
    seq_lens = torch.tensor([len(seq) for seq in inputs])
    
    # 用 0.0 填充输入，不影响 dx, dy 且 [0,0,0] 的状态不起作用
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    targets_cont_padded = pad_sequence(targets_cont, batch_first=True, padding_value=0.0)
    
    # 状态分类的 padding_value 必须是 -100，为了在 CrossEntropyLoss 中被忽略
    targets_state_padded = pad_sequence(targets_state, batch_first=True, padding_value=-100)
    
    max_len = inputs_padded.size(1)
    padding_mask = torch.arange(max_len).expand(len(seq_lens), max_len) >= seq_lens.unsqueeze(1)
    
    return inputs_padded, targets_cont_padded, targets_state_padded, padding_mask

# ==========================================
# 3. 直角坐标 Transformer (告别 Embedding)
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

class CartesianStrokeTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.cfg = config
        
        # 架构大简化：5 维全数字直接用 Linear 投射升维，不需要复杂的特征拆分了
        self.input_linear = nn.Linear(config.input_dim, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, 
            nhead=config.nhead, 
            dim_feedforward=config.dim_feedforward, 
            dropout=config.dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=config.num_layers)
        
        # --- 双路输出头 ---
        self.continuous_head = nn.Linear(config.d_model, config.cont_out_dim) # 预测 dx, dy
        self.state_head = nn.Linear(config.d_model, config.num_states)        # 预测 p1, p2, p3 的 logits

    def generate_causal_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, padding_mask=None):
        seq_len = src.size(1)
        
        # 直接过全连接层，极其高效
        x = self.input_linear(src)
        x = x * math.sqrt(self.cfg.d_model)
        x = self.pos_encoder(x)
        
        causal_mask = self.generate_causal_mask(seq_len).to(src.device)
        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        
        return self.continuous_head(out), self.state_head(out)

# ==========================================
# 4. 训练主循环 (包含中断保护)
# ==========================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = CartesianStrokeDataset("omniglot_cartesian_5d.pt")
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    
    model = CartesianStrokeTransformer(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    mse_loss_fn = nn.MSELoss(reduction='none') 
    
    # 损失权重调整：在直角坐标系下，坐标误差对整体形状的破坏不如分类错误致命
    # 所以把状态分类的权重稍微调高，确保它该抬笔的时候果断抬笔
    w_cont = 1.0
    w_cat = 2.0

    model.train()
    print(f"🚀 开始训练 Cartesian 架构！(输入维度: {cfg.input_dim})")

    try:
        for epoch in range(cfg.epochs):
            epoch_loss, total_cont, total_cat = 0.0, 0.0, 0.0
            
            for batch_idx, (inputs, targets_cont, targets_state, padding_mask) in enumerate(dataloader):
                inputs = inputs.to(device)
                targets_cont = targets_cont.to(device)
                targets_state = targets_state.to(device)
                padding_mask = padding_mask.to(device)
                
                optimizer.zero_grad()
                
                pred_cont, pred_state_logits = model(inputs, padding_mask=padding_mask)
                
                # 状态 Loss (CrossEntropy)
                loss_cat = ce_loss_fn(pred_state_logits.view(-1, cfg.num_states), targets_state.view(-1))
                
                # 坐标 Loss (MSE)
                loss_cont_matrix = mse_loss_fn(pred_cont, targets_cont)
                loss_cont_matrix[padding_mask] = 0.0
                
                valid_elements = (~padding_mask).sum() * cfg.cont_out_dim 
                loss_cont = loss_cont_matrix.sum() / (valid_elements + 1e-8)
                
                loss = (w_cont * loss_cont) + (w_cat * loss_cat)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                total_cont += loss_cont.item()
                total_cat += loss_cat.item()
                
            num_batches = len(dataloader)
            print(f"Epoch [{epoch+1}/{cfg.epochs}] | "
                  f"Total Loss: {epoch_loss/num_batches:.4f} | "
                  f"Coord(MSE): {total_cont/num_batches:.4f} | "
                  f"State(CE): {total_cat/num_batches:.4f}")

    except KeyboardInterrupt:
        print("\n⚠️ 捕获到 Ctrl+C！正在紧急停止训练...")
        
    finally:
        save_path = "cartesian_transformer.pth"
        torch.save(model.state_dict(), save_path)
        print(f"\n🎉 模型权重已安全保存至: {save_path}")

if __name__ == "__main__":
    train_model()