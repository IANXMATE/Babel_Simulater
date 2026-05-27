import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ==========================================
# 1. 配置与模型架构 (必须与 train_polar.py 完全一致)
# ==========================================
@dataclass
class ModelConfig:
    cont_feature_dim: int = 3  
    max_stroke_id: int = 50   
    num_states: int = 3        
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1

cfg = ModelConfig()

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

class PolarStrokeTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.cfg = config
        self.embed_dim = config.d_model // 2
        self.cont_dim = config.d_model - self.embed_dim
        
        self.stroke_embedding = nn.Embedding(config.max_stroke_id + 1, self.embed_dim)
        self.continuous_linear = nn.Linear(config.cont_feature_dim, self.cont_dim)
        self.pos_encoder = PositionalEncoding(config.d_model)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.nhead, dim_feedforward=config.dim_feedforward, 
            dropout=config.dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=config.num_layers)
        self.continuous_head = nn.Linear(config.d_model, config.cont_feature_dim) 
        self.state_head = nn.Linear(config.d_model, config.num_states)      

    def generate_causal_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, padding_mask=None):
        seq_len = src.size(1)
        stroke_ids = src[:, :, 0].long()
        stroke_ids[stroke_ids == -1] = self.cfg.max_stroke_id 
        stroke_ids = torch.clamp(stroke_ids, 0, self.cfg.max_stroke_id)
        cont_features = src[:, :, 1 : 1 + self.cfg.cont_feature_dim]
        
        embed_out = self.stroke_embedding(stroke_ids)
        cont_out = self.continuous_linear(cont_features)
        x = torch.cat([embed_out, cont_out], dim=-1)
        x = x * math.sqrt(self.cfg.d_model)
        x = self.pos_encoder(x)
        
        causal_mask = self.generate_causal_mask(seq_len).to(src.device)
        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.continuous_head(out), self.state_head(out)

# ==========================================
# 2. 自回归生成引擎 (带状态机)
# ==========================================
def generate_polar_sequence(model, device, max_len=150, temp_state=0.8, noise_cont=0.05):
    """
    自回归生成极坐标序列
    :param temp_state: 控制状态(换笔/结束)的随机性，越低越严谨
    :param noise_cont: 注入到连续物理量(角度/距离)的随机扰动
    """
    model.eval()
    
    # 初始状态 BOS: [stroke_id=0, d=0, delta_theta=0, delta_w=0]
    current_id = 0.0
    start_token = [[current_id, 0.0, 0.0, 0.0]]
    current_seq = torch.tensor([start_token], dtype=torch.float32).to(device)
    
    generated_tokens = [start_token[0]]
    
    with torch.no_grad():
        for i in range(max_len):
            pred_cont, pred_state_logits = model(current_seq)
            
            # 只取最后一步的预测结果
            last_cont = pred_cont[:, -1, :]              # (1, 3) -> [d, delta_theta, w]
            last_state_logits = pred_state_logits[:, -1, :] # (1, 3) -> [继续, 换笔, 结束]
            
            # --- 1. 离散状态预测 (马尔可夫转移) ---
            # 增加强制安全策略：前 15 步禁止结束，防止早停
            if len(generated_tokens) < 15:
                last_state_logits[0, 2] = -1e4 
                
            probs = F.softmax(last_state_logits / temp_state, dim=-1)
            state_idx = torch.multinomial(probs, num_samples=1).item()
            
            if state_idx == 2: # 结束符
                print(f"   [状态机] 收到结束指令，生成完毕 (共 {len(generated_tokens)} 步)")
                break
            elif state_idx == 1: # 换笔 (ID 增加)
                current_id += 1.0
            # 如果 state_idx == 0，保持 current_id 不变 (继续画当前笔画)

            # --- 2. 连续物理量预测 (带扰动) ---
            d, delta_theta, delta_w = last_cont.squeeze().tolist()
            
            # 注入高斯噪声，模拟手腕微小的抖动
            d += torch.randn(1).item() * noise_cont
            delta_theta += torch.randn(1).item() * noise_cont
            
            # 记录 Token
            next_token = [current_id, d, delta_theta, delta_w]
            generated_tokens.append(next_token)
            
            # 拼接进入下一轮循环
            next_tensor = torch.tensor([[next_token]], dtype=torch.float32).to(device)
            current_seq = torch.cat([current_seq, next_tensor], dim=1)
            
    return generated_tokens

# ==========================================
# 3. 极坐标积分渲染器 (物理引擎)
# ==========================================
def normalize_angle(angle):
    """保持角度在 [-pi, pi] 之间，防止旋转溢出"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def render_polar_to_cartesian(tokens, char_index):
    """
    将极坐标变化量通过积分还原为绝对坐标，并渲染图像
    """
    # 初始绝对物理状态
    x, y = 0.0, 0.0
    current_angle = 0.0 
    
    strokes = [] 
    current_stroke_x, current_stroke_y = [], []
    prev_id = 0.0
    
    # 丢弃第一个 BOS token
    valid_tokens = tokens[1:]
    
    for token in valid_tokens:
        stroke_id, d, delta_theta, delta_w = token
        
        # --- 核心物理积分 (Euler Integration) ---
        current_angle += delta_theta
        current_angle = normalize_angle(current_angle)
        
        next_x = x + d * math.cos(current_angle)
        next_y = y + d * math.sin(current_angle)
        
        # --- 判断是画线还是悬空跳跃 ---
        if stroke_id == prev_id:
            # 同一笔画内：留下墨迹
            if not current_stroke_x: # 如果是该笔画的第一个点，先把当前起点加进去
                current_stroke_x.append(x)
                current_stroke_y.append(y)
            current_stroke_x.append(next_x)
            current_stroke_y.append(next_y)
        else:
            # 换笔了：当前笔画结束，记录下来。
            # 此时的 d 和 delta_theta 是在空中“悬空位移”寻找新起点，不画线！
            if current_stroke_x:
                strokes.append((current_stroke_x, current_stroke_y))
            current_stroke_x, current_stroke_y = [], []
            
        # 更新状态机位置
        x, y = next_x, next_y
        prev_id = stroke_id
            
    # 收尾最后一笔
    if current_stroke_x:
        strokes.append((current_stroke_x, current_stroke_y))

    # --- 开始绘图 ---
    plt.figure(figsize=(5, 5))
    
    if not strokes:
        plt.text(0.5, 0.5, "Empty", fontsize=24, ha='center', color='red')
    else:
        for sx, sy in strokes:
            # 恢复训练时可能产生的缩放，这里使用 Matplotlib 的 autoscale 自动适应视野
            plt.plot(sx, sy, color='black', linewidth=3, solid_capstyle='round', solid_joinstyle='round')

    # Y 轴反转以匹配 SVG 标准视觉习惯
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis('off')
    plt.title(f"Polar Generated Character #{char_index}")
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. 主函数
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = PolarStrokeTransformer(cfg).to(device)
    model_path = "polar_transformer.pth"
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 模型 ({model_path}) 加载成功！准备生成...")
    except FileNotFoundError:
        print(f"❌ 错误: 未找到 {model_path}。请先运行 train_polar.py")
        return

    # 连续生成 3 个字符
    for i in range(3):
        print(f"\n--- 正在生成第 {i+1} 个外星极坐标文字 ---")
        
        # 核心超参调试建议：
        # temp_state 调大 -> 频繁乱换笔；调小 -> 长线一笔连。
        # noise_cont 调大 -> 肌肉痉挛般曲折；调小 -> 机器般的圆润完美。
        tokens = generate_polar_sequence(model, device, max_len=120, temp_state=0.6, noise_cont=0.08)
        
        render_polar_to_cartesian(tokens, i+1)

if __name__ == "__main__":
    main()