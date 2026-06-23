import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

# 导入你的模型 (假设你把上一版的模型保存为 stage1_fontgpt.py)
from stage1_fontgpt import FontGPT, FontGPTConfig
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 📦 1. 极其硬核的 Dataset 与 二维 Padding
# ==========================================
class FontGPTDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        seq = item["sequence"]
        
        # 提取序列特征
        shape_tokens = [s["shape_code"] for s in seq]
        width_tokens = [s["width_token"] for s in seq]
        
        p0_cx = [s["p0_cell"][0] for s in seq]; p0_cy = [s["p0_cell"][1] for s in seq]
        p3_cx = [s["p3_cell"][0] for s in seq]; p3_cy = [s["p3_cell"][1] for s in seq]
        
        p0_off = [s["p0_offset"] for s in seq]; p3_off = [s["p3_offset"] for s in seq]
        
        topo_matrix = item["topology_bias_matrix"]
        
        return {
            "shape": torch.tensor(shape_tokens, dtype=torch.long),
            "width": torch.tensor(width_tokens, dtype=torch.long),
            "p0_cx": torch.tensor(p0_cx, dtype=torch.long),
            "p0_cy": torch.tensor(p0_cy, dtype=torch.long),
            "p3_cx": torch.tensor(p3_cx, dtype=torch.long),
            "p3_cy": torch.tensor(p3_cy, dtype=torch.long),
            "p0_off": torch.tensor(p0_off, dtype=torch.float32),
            "p3_off": torch.tensor(p3_off, dtype=torch.float32),
            "topo": torch.tensor(topo_matrix, dtype=torch.long)
        }

def fontgpt_collate_fn(batch):
    """
    处理变长序列的 Padding。
    注意：topo_matrix 是 N x N 的二维矩阵，需要沿着两个维度同时 Pad！
    """
    # 找出当前 batch 中的最大序列长度
    lengths = [b["shape"].size(0) for b in batch]
    max_len = max(lengths)
    
    padded_batch = {}
    keys_1d = ["shape", "width", "p0_cx", "p0_cy", "p3_cx", "p3_cy"]
    keys_2d = ["p0_off", "p3_off"] # 序列长度 x 2维偏移
    
    # 1. 填充 1D 和 2D 的序列特征
    for k in keys_1d:
        # 使用 0 作为 padding index (建议预留 0 作为特殊 token)
        padded_batch[k] = torch.nn.utils.rnn.pad_sequence([b[k] for b in batch], batch_first=True, padding_value=0)
        
    for k in keys_2d:
        padded_batch[k] = torch.nn.utils.rnn.pad_sequence([b[k] for b in batch], batch_first=True, padding_value=0.0)
        
    # 🌟 2. 填充 3D 拓扑矩阵 (Batch, Seq, Seq)
    # 因为拓扑矩阵是方阵，我们需要在右侧和下方同时补 0
    padded_topos = []
    for b in batch:
        t = b["topo"]
        seq_len = t.size(0)
        pad_size = max_len - seq_len
        # F.pad 格式: (左右，上下)。注意是从最后一个维度开始往前倒推。
        padded_t = F.pad(t, (0, pad_size, 0, pad_size), mode='constant', value=0)
        padded_topos.append(padded_t)
        
    padded_batch["topo"] = torch.stack(padded_topos)
    
    # 3. 生成 Attention 用的 Padding Mask
    # True 表示有效，False 表示是被 Pad 出来的垃圾数据
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
    padded_batch["mask"] = mask
    
    return padded_batch

# ==========================================
# ⚖️ 2. 六头融合损失函数 (The Multi-Task Loss)
# ==========================================
def compute_fontgpt_loss(outputs, targets, mask):
    """
    将 6 个物理意义完全不同的 Loss 统一成一个标量。
    使用 mask 屏蔽掉 padding 部分的 loss 计算。
    """
    # 将 mask 展平，提取有效位置的索引
    active_loss = mask.view(-1) == True
    
    # --- 1. 离散类别特征 (Cross-Entropy) ---
    ce_loss = nn.CrossEntropyLoss()
    
    # 将预测值展平: [batch * seq_len, num_classes]
    l_shape = ce_loss(outputs["logits_shape"].view(-1, outputs["logits_shape"].size(-1))[active_loss], 
                      targets["shape"].view(-1)[active_loss])
                      
    l_width = ce_loss(outputs["logits_width"].view(-1, outputs["logits_width"].size(-1))[active_loss], 
                      targets["width"].view(-1)[active_loss])
                      
    l_p0_cx = ce_loss(outputs["logits_p0_cx"].view(-1, outputs["logits_p0_cx"].size(-1))[active_loss], targets["p0_cx"].view(-1)[active_loss])
    l_p0_cy = ce_loss(outputs["logits_p0_cy"].view(-1, outputs["logits_p0_cy"].size(-1))[active_loss], targets["p0_cy"].view(-1)[active_loss])
    l_p3_cx = ce_loss(outputs["logits_p3_cx"].view(-1, outputs["logits_p3_cx"].size(-1))[active_loss], targets["p3_cx"].view(-1)[active_loss])
    l_p3_cy = ce_loss(outputs["logits_p3_cy"].view(-1, outputs["logits_p3_cy"].size(-1))[active_loss], targets["p3_cy"].view(-1)[active_loss])
    
    # --- 2. 连续偏移回归 (MSE) ---
    mse_loss = nn.MSELoss()
    
    # Offset 的形状是 [batch, seq_len, 2]
    l_p0_off = mse_loss(outputs["pred_p0_offset"].view(-1, 2)[active_loss], targets["p0_off"].view(-1, 2)[active_loss])
    l_p3_off = mse_loss(outputs["pred_p3_offset"].view(-1, 2)[active_loss], targets["p3_off"].view(-1, 2)[active_loss])
    
    # --- 3. 动态权重融合 (Loss Balancing) ---
    # Shape 是字体的灵魂，给最高权重；Offset 只是微调格子内偏移，权重可以稍低
    total_loss = (
        2.0 * l_shape + 
        1.0 * l_width + 
        1.0 * (l_p0_cx + l_p0_cy + l_p3_cx + l_p3_cy) + 
        5.0 * (l_p0_off + l_p3_off) # Offset 本身值在 [-0.5, 0.5]，MSE 算出来会很小（比如0.01），所以要适当放大系数
    )
    
    return total_loss, {
        "shape": l_shape.item(), "cell": l_p0_cx.item(), "offset": l_p0_off.item()
    }

# ==========================================
# 🚀 3. 标准自回归训练循环 (Teacher Forcing)
# ==========================================
def generate_causal_mask(seq_len):
    """
    生成对角线以下的下三角矩阵，防止 Transformer 看到未来的笔画！
    (1 代表可见，0 代表遮蔽)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    return mask

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 训练设备: {device}")
    
    DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "fontgpt_dataset.json"))
    # 1. 加载数据
    dataset = FontGPTDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=fontgpt_collate_fn)
    
    # 2. 初始化模型与优化器
    config = FontGPTConfig()
    model = FontGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 3. 训练循环
    epochs = 50
    model.train()
    
    for epoch in range(epochs):
        total_epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            # --- 构建 Teacher Forcing 的输入与目标 ---
            inputs = {}
            targets = {}
            for k, v in batch.items():
                if k == "mask": 
                    continue
                if k == "topo":
                    # 🌟 核心修复：拓扑错位！
                    # Row 是 1: (未来的 Target), Col 是 :-1 (当前的 Context)
                    # 这样在 t 时刻，Query 查到的偏置正好是 Target 节点与已知节点的拓扑关系！
                    inputs[k] = v[:, 1:, :-1].to(device)
                    targets[k] = v[:, 1:, 1:].to(device)
                else:
                    inputs[k] = v[:, :-1].to(device)
                    targets[k] = v[:, 1:].to(device)
            
            # Mask 需要特殊处理：结合 Padding Mask 和 逻辑 Causal Mask
            pad_mask = batch["mask"][:, :-1].to(device) 
            seq_len = inputs["shape"].size(1)
            causal_mask = generate_causal_mask(seq_len).to(device)
            
            # 将二维的 pad_mask 广播为四维，和 causal_mask 取交集
            final_mask = (pad_mask.unsqueeze(1).unsqueeze(2) == True) & (causal_mask == 1)
            
            # --- Forward ---
            outputs = model(
                inputs["shape"], inputs["width"], 
                inputs["p0_cx"], inputs["p0_cy"], inputs["p0_off"],
                inputs["p3_cx"], inputs["p3_cy"], inputs["p3_off"],
                inputs["topo"], mask=final_mask
            )
            
            # --- Compute Loss ---
            loss, metrics = compute_fontgpt_loss(outputs, targets, pad_mask)
            
            # --- Backward ---
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            
            # 🌟 核心修复 2: 修正优化器步进 API
            optimizer.step() 
            
            total_epoch_loss += loss.item()
            pbar.set_postfix(Loss=f"{loss.item():.4f}", Shp=f"{metrics['shape']:.4f}", Off=f"{metrics['offset']:.4f}")
        if epoch % 5 == 0 :    
            print(f"✅ Epoch {epoch+1} 结束 | 平均 Loss: {total_epoch_loss/len(dataloader):.4f}")

    # 4. 保存模型权重
    print("🎉 训练全部完成！正在保存 FontGPT 权重...")
    save_path = "fontgpt_stage1_latest.pth"
    save_path = os.path.abspath(os.path.join(SCRIPT_DIR, save_path))
    # 推荐保存 state_dict，这是 PyTorch 的最佳实践
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_epoch_loss/len(dataloader),
        'config': config.__dict__ # 顺便把超参数配置也存下来，方便以后推理
    }, save_path)
    print(f"💾 模型已成功保存至: {save_path}")

if __name__ == "__main__":
    train()