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
# 🌟 接入全新架构的拓扑先行数据集
DATASET_FILE = os.path.join(SCRIPT_DIR, "fontgpt_dataset_topology_first.json")
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "fontgpt_vector_lm_latest.pth")

VOCAB_SIZE = 10000 
MAX_SEQ_LEN = 256 

# Transformer 超参数
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 6
DROPOUT = 0.1 
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4

# ==========================================
# 🎯 核心进化：Focal Loss (治理长尾与模式崩溃)
# ==========================================
class FocalLoss(nn.Module):
    """
    通过引入调制系数 (1 - pt)^gamma，让模型降低对 Shape 7 / E2E 等高频简单样本的关注，
    将损失算力集中在长尾的撇、捺曲线以及复杂的 T/X 拓扑交叉上。
    """
    def __init__(self, gamma=2.0, ignore_index=0):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits 尺寸: [Batch * Seq_len, Vocab_size]
        # targets 尺寸: [Batch * Seq_len]
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # 只对非 PAD 区域求均值
        non_pad_mask = targets != self.ignore_index
        if non_pad_mask.sum() > 0:
            return focal_loss[non_pad_mask].mean()
        return focal_loss.mean()

# ==========================================
# 🔠 核心基建：拓扑先行分词器 (Font Tokenizer)
# ==========================================
class FontTokenizer:
    def __init__(self):
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        
        self.CMD_STROKE = 3     
        self.CMD_JUNCTION = 4   
        self.CMD_NEW_ROOT = 5   # 🌟 新增：无约束起笔指令
        
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
            if item["token_type"] == "NEW_ROOT":
                tokens.append(self.CMD_NEW_ROOT)
                
            elif item["token_type"] == "JUNCTION":
                tokens.append(self.CMD_JUNCTION)
                jmap = {"E2E": 0, "X": 1, "T": 2}
                tokens.append(self.OFFSET_JTYPE + jmap.get(item["j_type"], 0))
                # 🌟 拓扑先行架构：单向指向历史目标笔画
                tokens.append(self.OFFSET_DIST + min(item["target_dist"], 99)) 
                tokens.append(self.OFFSET_TBIN + item["t_self_bin"])
                tokens.append(self.OFFSET_TBIN + item["t_target_bin"])
                
            elif item["token_type"] == "STROKE":
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
        
        self.blocks = nn.ModuleList([GPTBlock(D_MODEL, N_HEADS, DROPOUT) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
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
    # 🛡️ 条件约束强力生成器 (Topology-First)
    # ==========================================
    @torch.no_grad()
    def generate_safe(self, start_tokens, max_new_tokens, temperature=1.0):
        self.eval()
        device = next(self.parameters()).device
        seq = torch.tensor([start_tokens], dtype=torch.long).to(device)
        tk = self.tokenizer

        for _ in range(max_new_tokens):
            logits = self(seq)[:, -1, :] / temperature 
            mask = torch.full((VOCAB_SIZE,), float('-inf'), device=device)
            
            curr_list = seq[0].tolist()
            last_cmd = None
            dist = 0
            
            # 倒序寻找最近的一个指令 (STROKE, JUNCTION 或 NEW_ROOT)
            for i in range(len(curr_list) - 1, -1, -1):
                if curr_list[i] in [tk.CMD_STROKE, tk.CMD_JUNCTION, tk.CMD_NEW_ROOT]:
                    last_cmd = curr_list[i]
                    dist = len(curr_list) - 1 - i
                    break
            
            # 状态机：条件路由解析
            if last_cmd is None or (last_cmd == tk.CMD_STROKE and dist == 11):
                # 刚开始，或者画完了一笔：准备宣告下一步是凭空起笔(NEW_ROOT)还是挂载连接(JUNCTION)
                mask[tk.CMD_NEW_ROOT] = 0
                mask[tk.CMD_JUNCTION] = 0
                mask[tk.EOS] = 0
            
            elif last_cmd == tk.CMD_NEW_ROOT:
                # 宣告了起笔，下一步必须直接画实体
                if dist == 0: mask[tk.CMD_STROKE] = 0
                
            elif last_cmd == tk.CMD_JUNCTION:
                # 动态获取已生成的 JTYPE
                j_type_val = -1
                if dist in [2, 3]:
                    j_type_token = curr_list[-dist]
                    j_type_val = j_type_token - tk.OFFSET_JTYPE

                # 条件挂载属性解析
                if dist == 0: mask[tk.OFFSET_JTYPE:tk.OFFSET_DIST] = 0      # JType
                elif dist == 1: mask[tk.OFFSET_DIST:tk.OFFSET_TBIN] = 0     # target_dist
                elif dist == 2:                                             # t_self
                    if j_type_val == 0 or j_type_val == 2:  # E2E 或 T 型(Self是Guest) 必须在端点
                        mask[tk.OFFSET_TBIN + 0] = 0
                        mask[tk.OFFSET_TBIN + 32] = 0
                    elif j_type_val == 1:                   # X 型必须在中间
                        mask[tk.OFFSET_TBIN + 1 : tk.OFFSET_TBIN + 32] = 0
                elif dist == 3:                                             # t_target
                    if j_type_val == 0:                     # E2E 必须在端点
                        mask[tk.OFFSET_TBIN + 0] = 0
                        mask[tk.OFFSET_TBIN + 32] = 0
                    elif j_type_val == 1 or j_type_val == 2:# X 型或 T型(Target是Host) 必须在中间
                        mask[tk.OFFSET_TBIN + 1 : tk.OFFSET_TBIN + 32] = 0
                elif dist == 4:
                    # 挂载条件配置完毕，立刻画实体
                    mask[tk.CMD_STROKE] = 0
                    
            elif last_cmd == tk.CMD_STROKE:
                # 在确定好的拓扑约束下，生成笔画物理属性
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

            safe_logits = logits + mask
            probs = F.softmax(safe_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            seq = torch.cat([seq, next_token], dim=1)
            if next_token.item() == tk.EOS: break
                
        return seq[0].tolist()

# ==========================================
# 🚂 强力训练循环
# ==========================================
def main():
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    
    print(f"🚀 初始化拓扑先行条件模型 (Device: {device})...")
    
    dataset = FontVectorLanguageDataset(DATASET_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"✅ 数据集加载完成，样本数: {len(dataset)}")

    model = FontVectorLM().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # 🌟 使用 FocalLoss 强行打压高频直线(Shape 7)与 E2E 的权重，逼迫模型学习长尾弧线
    criterion = FocalLoss(gamma=2.0, ignore_index=0)

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
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Focal_Loss": f"{loss.item():.4f}"})
            
        print(f"📈 Epoch {epoch+1} | 平均 Focal Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 核心模型已保存至 {MODEL_SAVE_PATH}！")

if __name__ == "__main__":
    main()