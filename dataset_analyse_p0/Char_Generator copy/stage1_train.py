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
DATASET_FILE = os.path.join(SCRIPT_DIR, "fontgpt_dataset.json")
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "fontgpt_vector_lm_latest.pth")

VOCAB_SIZE = 10000 
MAX_SEQ_LEN = 256 

# Transformer 超参数
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 6
DROPOUT = 0.1 # 训练时适当加大防止过拟合
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4

# ==========================================
# 🔠 核心基建：矢量分词器 (Font Tokenizer)
# ==========================================
class FontTokenizer:
    def __init__(self):
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        
        self.CMD_STROKE = 3     
        self.CMD_JUNCTION = 4   
        
        self.OFFSET_SHAPE = 100    
        self.OFFSET_VAR = 1000      
        self.OFFSET_CELL = 2000     
        self.OFFSET_OFFSET = 3000   
        self.OFFSET_WIDTH = 4000    
        
        self.OFFSET_JTYPE = 5000    
        self.OFFSET_DIST = 6000     
        self.OFFSET_TBIN = 7000     

    def encode(self, sequence_dicts):
        tokens = [self.BOS]
        for item in sequence_dicts:
            if item["token_type"] == "STROKE":
                tokens.append(self.CMD_STROKE)
                tokens.append(self.OFFSET_SHAPE + item["shape_code"])
                tokens.append(self.OFFSET_VAR + item["variant_id"])
                tokens.append(self.OFFSET_CELL + item["p0_cell"][0])
                tokens.append(self.OFFSET_CELL + item["p0_cell"][1])
                tokens.append(self.OFFSET_OFFSET + int(max(0, min(0.999, item["p0_offset"][0])) * 32))
                tokens.append(self.OFFSET_OFFSET + int(max(0, min(0.999, item["p0_offset"][1])) * 32))
                tokens.append(self.OFFSET_CELL + item["p3_cell"][0])
                tokens.append(self.OFFSET_CELL + item["p3_cell"][1])
                tokens.append(self.OFFSET_OFFSET + int(max(0, min(0.999, item["p3_offset"][0])) * 32))
                tokens.append(self.OFFSET_OFFSET + int(max(0, min(0.999, item["p3_offset"][1])) * 32))
                tokens.append(self.OFFSET_WIDTH + item["width_token"])
                
            elif item["token_type"] == "JUNCTION":
                tokens.append(self.CMD_JUNCTION)
                jmap = {"E2E": 0, "X": 1, "T": 2}
                tokens.append(self.OFFSET_JTYPE + jmap.get(item["j_type"], 0))
                tokens.append(self.OFFSET_DIST + min(item["ref_a_dist"], 99)) 
                tokens.append(self.OFFSET_DIST + min(item["ref_b_dist"], 99))
                tokens.append(self.OFFSET_TBIN + item["ta_bin"])
                tokens.append(self.OFFSET_TBIN + item["tb_bin"])
                
        tokens.append(self.EOS)
        return tokens

# ==========================================
# 📦 数据集与加载器
# ==========================================
class FontVectorLanguageDataset(Dataset):
    def __init__(self, json_file):
        print(f"📖 加载数据: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        self.tokenizer = FontTokenizer()
        self.samples = []
        
        for d in raw_data:
            tokens = self.tokenizer.encode(d["sequence"])
            if len(tokens) <= MAX_SEQ_LEN:
                pad_len = MAX_SEQ_LEN - len(tokens)
                tokens.extend([self.tokenizer.PAD] * pad_len)
                self.samples.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        return seq[:-1], seq[1:]

# ==========================================
# 🧠 纯粹的 GPT 架构：手写 Decoder Block
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class GPTBlock(nn.Module):
    """🌟 修复1：最纯正的自回归 GPT 块，彻底抛弃 Encoder 歧义"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, causal_mask, padding_mask=None):
        # 严格遵守 Pre-LN 架构，对梯度的流动极其友好
        x_norm = self.ln_1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, 
                                attn_mask=causal_mask, 
                                key_padding_mask=padding_mask,
                                need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x

class FontVectorLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = FontTokenizer()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=0)
        self.pos_encoder = PositionalEncoding(D_MODEL)
        
        # 🌟 修复1：使用 ModuleList 堆叠手写的 GPTBlock
        self.blocks = nn.ModuleList([GPTBlock(D_MODEL, N_HEADS, DROPOUT) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL) # GPT 必须有一个 final layernorm
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def generate_causal_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, padding_mask=None):
        seq_len = x.size(1)
        x = self.embedding(x) * math.sqrt(D_MODEL)
        x = self.pos_encoder(x)
        
        causal_mask = self.generate_causal_mask(seq_len).to(x.device)
        
        for block in self.blocks:
            x = block(x, causal_mask, padding_mask)
            
        x = self.ln_f(x)
        return self.lm_head(x)

    # ==========================================
    # 🛡️ 修复2：带状态机约束的强力生成器
    # ==========================================
    @torch.no_grad()
    def generate_safe(self, start_tokens, max_new_tokens, temperature=1.0):
        """
        基于状态机的安全解码器。确保模型生成的 Token 绝对符合拓扑语法。
        """
        self.eval()
        device = next(self.parameters()).device
        seq = torch.tensor([start_tokens], dtype=torch.long).to(device) # [1, seq_len]
        tk = self.tokenizer

        for _ in range(max_new_tokens):
            logits = self(seq)[:, -1, :] / temperature # 只取最后一个位置的预测
            
            # --- 构建 Logit 掩码 (全置为负无穷) ---
            mask = torch.full((VOCAB_SIZE,), float('-inf'), device=device)
            
            # --- 状态机解析：根据历史寻找当前上下文 ---
            curr_list = seq[0].tolist()
            last_cmd = None
            dist = 0
            
            # 倒序寻找最近的一个指令 (STROKE 或 JUNCTION)
            for i in range(len(curr_list) - 1, -1, -1):
                if curr_list[i] in [tk.CMD_STROKE, tk.CMD_JUNCTION]:
                    last_cmd = curr_list[i]
                    dist = len(curr_list) - 1 - i
                    break
            
            # --- 动态赋权：开放允许的 Token 区间 ---
            if last_cmd is None or (last_cmd == tk.CMD_STROKE and dist == 11) or (last_cmd == tk.CMD_JUNCTION and dist == 5):
                # 状态 0：期待新指令
                mask[tk.CMD_STROKE] = 0
                mask[tk.CMD_JUNCTION] = 0
                mask[tk.EOS] = 0
            
            elif last_cmd == tk.CMD_STROKE:
                # 笔画属性解析流 (严格的顺序依赖)
                if dist == 0: mask[tk.OFFSET_SHAPE:tk.OFFSET_VAR] = 0       # Shape
                elif dist == 1: mask[tk.OFFSET_VAR:tk.OFFSET_CELL] = 0      # Variant
                elif dist == 2: mask[tk.OFFSET_CELL:tk.OFFSET_OFFSET] = 0   # p0_cx
                elif dist == 3: mask[tk.OFFSET_CELL:tk.OFFSET_OFFSET] = 0   # p0_cy
                elif dist == 4: mask[tk.OFFSET_OFFSET:tk.OFFSET_WIDTH] = 0  # p0_ox
                elif dist == 5: mask[tk.OFFSET_OFFSET:tk.OFFSET_WIDTH] = 0  # p0_oy
                elif dist == 6: mask[tk.OFFSET_CELL:tk.OFFSET_OFFSET] = 0   # p3_cx
                elif dist == 7: mask[tk.OFFSET_CELL:tk.OFFSET_OFFSET] = 0   # p3_cy
                elif dist == 8: mask[tk.OFFSET_OFFSET:tk.OFFSET_WIDTH] = 0  # p3_ox
                elif dist == 9: mask[tk.OFFSET_OFFSET:tk.OFFSET_WIDTH] = 0  # p3_oy
                elif dist == 10: mask[tk.OFFSET_WIDTH:tk.OFFSET_JTYPE] = 0  # Width
                
            elif last_cmd == tk.CMD_JUNCTION:
                # 🌟 动态向回追溯，获取当前刚刚生成的交点类型 (j_type)
                # 当 dist=3 时，我们在预测 ta_bin，此时 j_type 刚好在倒数第 3 个位置
                # 当 dist=4 时，我们在预测 tb_bin，此时 j_type 刚好在倒数第 4 个位置
                j_type_val = -1
                if dist in [3, 4]:
                    j_type_token = curr_list[-dist]
                    j_type_val = j_type_token - tk.OFFSET_JTYPE

                # 交点属性解析流
                if dist == 0: mask[tk.OFFSET_JTYPE:tk.OFFSET_DIST] = 0      # JType
                elif dist == 1: mask[tk.OFFSET_DIST:tk.OFFSET_TBIN] = 0     # ref_a
                elif dist == 2: mask[tk.OFFSET_DIST:tk.OFFSET_TBIN] = 0     # ref_b
                elif dist == 3 or dist == 4:                                # ta_bin & tb_bin
                    if j_type_val == 0:
                        # 🎯 物理法则强干预：如果是 E2E (0)，强制 t 值只能是 0 或 32！
                        mask[tk.OFFSET_TBIN + 0] = 0
                        mask[tk.OFFSET_TBIN + 32] = 0
                    else:
                        # 如果是 T 型或 X 型，允许 0 到 32 的所有比例 (共 33 个 Token)
                        mask[tk.OFFSET_TBIN : tk.OFFSET_TBIN + 33] = 0

            # 将掩码加到 logits 上，非法的选项会被拉到 -inf，softmax 后概率直接为 0
            safe_logits = logits + mask
            probs = F.softmax(safe_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            seq = torch.cat([seq, next_token], dim=1)
            if next_token.item() == tk.EOS: break
                
        return seq[0].tolist()

# ==========================================
# 🚂 极简训练循环
# ==========================================
def main():
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    
    print(f"🚀 初始化纯正 GPT 矢量语言模型 (Device: {device})...")
    
    dataset = FontVectorLanguageDataset(DATASET_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # 调试时设为 0
    print(f"✅ 数据集加载完成，样本数: {len(dataset)}")

    model = FontVectorLM().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            padding_mask = (batch_x == 0)
            
            optimizer.zero_grad()
            logits = model(batch_x, padding_mask=padding_mask)
            
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), batch_y.reshape(-1))
            loss.backward()
            
            # GPT 必备的梯度裁剪，防止早期梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        print(f"📈 Epoch {epoch+1} | 平均 Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 模型已保存至 {MODEL_SAVE_PATH}！")

if __name__ == "__main__":
    main()