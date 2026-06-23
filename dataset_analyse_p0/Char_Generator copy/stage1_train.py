import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm

# ==========================================
# ⚙️ 全局配置与超参数
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = "fontgpt_dataset.json"
DATASET_FILE = os.path.join(SCRIPT_DIR, DATASET_FILE)

MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "fontgpt_vector_lm_latest.pth")

# 统一词表空间配置 (Token Space)
VOCAB_SIZE = 10000 
MAX_SEQ_LEN = 256  # 根据你的序列长度可以适当调大

# Transformer 超参数
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 6
DROPOUT = 0.01
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4

# ==========================================
# 🔠 核心基建：矢量分词器 (Font Tokenizer)
# ==========================================
class FontTokenizer:
    """
    将 JSON 中的异构字典 (STROKE / JUNCTION)
    展平并映射为 1D 的全局统一整数 Token 序列。
    """
    def __init__(self):
        # 特殊控制 Token
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        
        self.CMD_STROKE = 3     # 宣告：接下来是一笔
        self.CMD_JUNCTION = 4   # 宣告：接下来是一个交点
        
        # 物理意义空间的偏移量 (避免不同属性的 ID 冲突)
        self.OFFSET_SHAPE = 100     # 形状 ID (如 0-500)
        self.OFFSET_VAR = 1000      # 变体 ID (0-3)
        self.OFFSET_CELL = 2000     # 网格 ID (0-31)
        self.OFFSET_OFFSET = 3000   # 偏移量离散化 (0-31)
        self.OFFSET_WIDTH = 4000    # 宽度 ID (0-4)
        
        self.OFFSET_JTYPE = 5000    # 交点类型 (0:E2E, 1:X, 2:T)
        self.OFFSET_DIST = 6000     # 相对距离引用 (0-99)
        self.OFFSET_TBIN = 7000     # T 值比例桶 (0-31)

    def encode(self, sequence_dicts):
        """将 [Dict, Dict...] 编码为 [Int, Int...]"""
        tokens = [self.BOS]
        for item in sequence_dicts:
            if item["token_type"] == "STROKE":
                tokens.append(self.CMD_STROKE)
                tokens.append(self.OFFSET_SHAPE + item["shape_code"])
                tokens.append(self.OFFSET_VAR + item["variant_id"])
                tokens.append(self.OFFSET_CELL + item["p0_cell"][0])
                tokens.append(self.OFFSET_CELL + item["p0_cell"][1])
                # 将 float 偏移量 (0~1) 临时量化为 32 个 bin 以适应纯 Token 架构
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
                # 限制最大向前回溯距离为 99 笔
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
                # 补齐 PAD
                pad_len = MAX_SEQ_LEN - len(tokens)
                tokens.extend([self.tokenizer.PAD] * pad_len)
                self.samples.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # GPT 训练范式：输入是 [0...N-1], 标签是 [1...N]
        seq = self.samples[idx]
        return seq[:-1], seq[1:]

# ==========================================
# 🧠 极简大模型：纯 Causal Transformer
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

class FontVectorLM(nn.Module):
    def __init__(self):
        super().__init__()
        # 全局统一 Embedding
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=0)
        self.pos_encoder = PositionalEncoding(D_MODEL)
        
        # 核心大脑：标准的自回归编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, 
            nhead=N_HEADS, 
            dim_feedforward=D_MODEL * 4, 
            dropout=DROPOUT,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        
        # 语言模型头 (LM Head)：预测下一个 Token 的概率分布
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def generate_square_subsequent_mask(self, sz):
        """生成因果掩码 (Causal Mask)，确保模型看不见未来"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, padding_mask=None):
        # x shape: [Batch, Seq_len]
        seq_len = x.size(1)
        
        # 1. 词嵌入 + 位置编码
        x_emb = self.embedding(x) * math.sqrt(D_MODEL)
        x_emb = self.pos_encoder(x_emb)
        
        # 2. 生成因果掩码 (Causal Mask)
        causal_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # 3. Transformer 推理
        # 注意：在 PyTorch 中，src_key_padding_mask 控制 PAD 不参与计算
        out = self.transformer(x_emb, mask=causal_mask, src_key_padding_mask=padding_mask)
        
        # 4. 投影回全局词表空间
        logits = self.lm_head(out)
        return logits

# ==========================================
# 🚂 极简训练循环
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 初始化极简矢量语言模型 (Device: {device})...")
    
    dataset = FontVectorLanguageDataset(DATASET_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"✅ 数据集加载完成，样本数: {len(dataset)}")

    model = FontVectorLM().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # 使用交叉熵计算 Loss，极其关键的一点：忽略 PAD Token (0) 的 Loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # 构建 Padding Mask (为 True 的地方在 Attention 中会被忽略)
            padding_mask = (batch_x == 0)
            
            optimizer.zero_grad()
            logits = model(batch_x, padding_mask=padding_mask)
            
            # 计算 Next-Token Prediction Loss
            # logits 展平为 [Batch * Seq_len, Vocab_size], batch_y 展平为 [Batch * Seq_len]
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), batch_y.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        print(f"📈 Epoch {epoch+1} | 平均 Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 模型已保存至 {MODEL_SAVE_PATH}！大一统时代降临！")

if __name__ == "__main__":
    main()