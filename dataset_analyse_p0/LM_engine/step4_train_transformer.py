import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import math
import numpy as np

# ==========================================
# 📂 路径与配置
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(SCRIPT_DIR, "model_config.json")

# ==========================================
# 📊 步骤一：序列化数据集 (处理 SOS, EOS, PAD)
# ==========================================
class TokenizedFontDataset(Dataset):
    def __init__(self, pkl_path, max_seq_len=20, num_tokens=1024):
        with open(pkl_path, 'rb') as f:
            self.data_dict = pickle.load(f)
            
        self.max_seq_len = max_seq_len
        self.num_tokens = num_tokens
        
        # 定义特殊 Token
        self.PAD_IDX = num_tokens       # 1024
        self.SOS_IDX = num_tokens + 1   # 1025
        self.EOS_IDX = num_tokens + 2   # 1026
        
        self.samples = []
        self._prepare_data()
        
    def _prepare_data(self):
        # 遍历所有字，将每个字的笔画序列转化为固定长度的 Tensor
        for hex_key, strokes in self.data_dict.items():
            # 截断过长的字
            if len(strokes) > self.max_seq_len - 2:
                strokes = strokes[:self.max_seq_len - 2]
                
            seq_len = len(strokes)
            
            # 初始化数组
            token_seq = np.full(self.max_seq_len, self.PAD_IDX, dtype=np.int64)
            # 🌟 修复：将维度扩充到 8，以容纳 4个坐标 + 4个宽度的特征
            spatial_seq = np.zeros((self.max_seq_len, 8), dtype=np.float32)
            
            # 填入 <SOS>
            token_seq[0] = self.SOS_IDX
            
            # 填入真实笔画
            for i, s in enumerate(strokes):
                token_seq[i + 1] = s['token_id']
                # 🌟 核心：空间坐标极度需要归一化到 [-1, 1] 区间，否则 MSE Loss 会爆炸
                # 假设画布 400x400，长度最大 400，角度 -pi 到 pi
                norm_x = (s['start_x'] / 200.0) - 1.0
                norm_y = (s['start_y'] / 200.0) - 1.0
                norm_len = s['length'] / 400.0
                norm_ang = s['angle'] / math.pi
                # 🌟 核心：加上 4 个宽度的归一化 (假设最大线宽为 30)
                norm_w0 = s['width_0'] / 30.0
                norm_w1 = s['width_1'] / 30.0
                norm_w2 = s['width_2'] / 30.0
                norm_w3 = s['width_3'] / 30.0
                spatial_seq[i + 1] = [norm_x, norm_y, norm_len, norm_ang, norm_w0, norm_w1, norm_w2, norm_w3]
                
            # 填入 <EOS>
            token_seq[seq_len + 1] = self.EOS_IDX
            
            # 构建自回归的 Input 和 Target (错开一位)
            input_tokens = token_seq[:-1]
            input_spatial = spatial_seq[:-1]
            
            target_tokens = token_seq[1:]
            target_spatial = spatial_seq[1:]
            
            # padding mask (True 表示是 PAD，不需要计算 Loss)
            padding_mask = (target_tokens == self.PAD_IDX)
            
            self.samples.append({
                "input_tokens": torch.tensor(input_tokens),
                "input_spatial": torch.tensor(input_spatial),
                "target_tokens": torch.tensor(target_tokens),
                "target_spatial": torch.tensor(target_spatial),
                "padding_mask": torch.tensor(padding_mask)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ==========================================
# 🧠 步骤二：FontGPT 架构 (自回归混合输出)
# ==========================================
class FontGPT(nn.Module):
    def __init__(self, vocab_size=1027, hidden_dim=256, n_heads=8, n_layers=4, max_seq_len=20):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. 嵌入层 (Embedding)
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.spatial_emb = nn.Linear(8, hidden_dim) # 连续坐标投影为特征向量
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim) # 位置编码
        
        # 2. GPT 核心 (Causal Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 3. 混合输出头 (Hybrid Heads)
        # 头 A：猜下一个笔画的形状 (分类)
        self.token_head = nn.Linear(hidden_dim, vocab_size)
        
        # 头 B：猜下一个笔画的位置 (回归)
        self.spatial_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 8) # 🌟 预测 8 维：x, y, len, ang, w0, w1, w2, w3
        )

    def forward(self, tokens, spatial):
        B, SeqLen = tokens.shape
        
        # 将 Token 特征和空间特征融合，加上位置编码
        positions = torch.arange(0, SeqLen, device=tokens.device).unsqueeze(0).expand(B, SeqLen)
        
        # 输入 = Token嵌入 + 坐标嵌入 + 位置嵌入
        x = self.token_emb(tokens) + self.spatial_emb(spatial) + self.pos_emb(positions)
        
        # 生成因果掩码 (Causal Mask: 只能看当前和过去的笔画，不能偷看未来)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(SeqLen).to(tokens.device)
        
        # 通过 Transformer
        hidden_states = self.transformer(x, mask=causal_mask)
        
        # 预测下一步
        token_logits = self.token_head(hidden_states)
        spatial_preds = self.spatial_head(hidden_states)
        
        return token_logits, spatial_preds

# ==========================================
# 🚀 步骤三：训练主循环
# ==========================================
def train():
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"⚙️ 训练设备: {device}")

    # 1. 加载数据
    pkl_path = os.path.join(SCRIPT_DIR, "tokenized_fonts_dataset.pkl")
    num_tokens = config['model_config']['num_tokens']
    dataset = TokenizedFontDataset(pkl_path, max_seq_len=20, num_tokens=num_tokens)
    
    # 你现在的字数比较少(166)，我们用小的 batch_size 防止过拟合
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 2. 初始化 FontGPT
    VOCAB_SIZE = num_tokens + 3 # 1024 + SOS, EOS, PAD
    model = FontGPT(vocab_size=VOCAB_SIZE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 分类损失 (忽略 PAD_IDX 的损失)
    criterion_token = nn.CrossEntropyLoss(ignore_index=dataset.PAD_IDX)
    # 空间回归损失
    criterion_spatial = nn.MSELoss(reduction='none')

    epochs = 300
    print(f"\n🚀 开始训练 FontGPT (字形语法大模型)...")
    
    for epoch in range(epochs):
        model.train()
        total_tok_loss = 0
        total_spa_loss = 0
        correct_tokens = 0
        total_valid_tokens = 0
        
        for batch in dataloader:
            inp_tok = batch['input_tokens'].to(device)
            inp_spa = batch['input_spatial'].to(device)
            tgt_tok = batch['target_tokens'].to(device)
            tgt_spa = batch['target_spatial'].to(device)
            pad_mask = batch['padding_mask'].to(device)
            
            optimizer.zero_grad()
            
            # 前向预测
            tok_logits, spa_preds = model(inp_tok, inp_spa)
            
            # --- 计算 Token 分类 Loss ---
            # 展平以便 CrossEntropy 计算: (B*SeqLen, VocabSize) 和 (B*SeqLen)
            loss_tok = criterion_token(tok_logits.view(-1, VOCAB_SIZE), tgt_tok.view(-1))
            
            # --- 计算 Spatial 坐标 Loss ---
            # MSE 计算出每个元素的误差，但我们要把 Padding 位置的误差抹零
            mse = criterion_spatial(spa_preds, tgt_spa) # (B, SeqLen, 4)
            # 把 pad_mask 扩展到 4 维 (B, SeqLen, 1) 然后抹掉废弃位置
            mse = mse.masked_fill(pad_mask.unsqueeze(-1), 0.0) 
            # 只对有效的笔画计算平均坐标误差
            valid_count = (~pad_mask).sum().clamp(min=1)
            loss_spa = mse.sum() / (valid_count * 4)
            
            # 🌟 联合 Loss：坐标误差在 [-1,1] 之间数值很小，所以需要放大权重
            loss = loss_tok + 5.0 * loss_spa
            loss.backward()
            optimizer.step()
            
            # 统计指标
            total_tok_loss += loss_tok.item()
            total_spa_loss += loss_spa.item()
            
            preds = tok_logits.argmax(dim=-1)
            valid_mask = ~pad_mask
            correct_tokens += ((preds == tgt_tok) & valid_mask).sum().item()
            total_valid_tokens += valid_mask.sum().item()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_tok = total_tok_loss / len(dataloader)
            avg_spa = total_spa_loss / len(dataloader)
            acc = (correct_tokens / total_valid_tokens) * 100 if total_valid_tokens > 0 else 0
            print(f"Epoch [{epoch+1:03d}/{epochs}] | Tok Loss: {avg_tok:.4f} (Acc: {acc:.1f}%) | Spa MSE: {avg_spa:.4f}")

    # 保存模型
    save_dir = os.path.join(SCRIPT_DIR, config['train_config']['save_dir'])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "font_gpt_best.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n🎉 训练完成！FontGPT 权重已保存至: {save_path}")
    
    # ==========================================
    # 🎯 终极魔法演示：让 AI 瞎编（生成）一个字！
    # ==========================================
    model.eval()
    print("\n🪄 --- 魔法时刻：让 AI 闭眼写一个字 ---")
    with torch.no_grad():
        # 起手式：只给一个 <SOS> Token，空间坐标全 0
        gen_toks = [dataset.SOS_IDX]
        gen_spas = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        
        for step in range(15): # 最多画 15 笔
            inp_tok = torch.tensor([gen_toks]).to(device)
            inp_spa = torch.tensor([gen_spas], dtype=torch.float32).to(device)
            
            # 模型预测 (拿最后一步的输出)
            tok_logits, spa_preds = model(inp_tok, inp_spa)
            next_tok_logits = tok_logits[0, -1, :] # 最后一个时间步的概率分布
            next_spa = spa_preds[0, -1, :].cpu().tolist()
            
            # 采用贪心策略，直接选概率最大的 Token
            next_tok_id = next_tok_logits.argmax().item()
            
            if next_tok_id == dataset.EOS_IDX:
                print(f"  👉 步骤 {step+1}: 模型输出 <EOS>，写完了！")
                break
                
            print(f"  👉 步骤 {step+1}: 决定画 Token [{next_tok_id:04d}] | 预测归一化坐标: x={next_spa[0]:.2f}, y={next_spa[1]:.2f}")
            
            # 把预测结果喂回输入，准备画下一笔
            gen_toks.append(next_tok_id)
            gen_spas.append(next_spa)

if __name__ == "__main__":
    train()