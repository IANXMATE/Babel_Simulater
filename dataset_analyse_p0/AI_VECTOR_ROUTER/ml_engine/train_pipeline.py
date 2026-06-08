import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle

# 🌟 1. 无缝导入咱们刚刚写的 data_builder
try :
    from ml_engine.data_builder import build_dataset_from_logs, ACTION_VOCAB
except :
    from data_builder import build_dataset_from_logs, ACTION_VOCAB
# ==========================================
# 📊 步骤一：PyTorch Dataset 与动态 Padding
# ==========================================
class GraphEditingDataset(Dataset):
    def __init__(self, data_list, max_edges=64):
        """
        max_edges: 图中允许的最大边数 N。
        Transformer 需要固定的 Tensor shape，因此我们必须进行 Padding。
        """
        self.data = data_list
        self.max_edges = max_edges

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        x_feat_np = sample["x_feat"] # (N, D)
        x_bias_np = sample["x_bias"] # (N, N)
        
        N, D = x_feat_np.shape
        # 防止越界截断
        actual_n = min(N, self.max_edges)
        
        # 初始化固定大小的空 Tensor
        x_feat = torch.zeros((self.max_edges, D), dtype=torch.float32)
        x_bias = torch.full((self.max_edges, self.max_edges), -1.0, dtype=torch.float32)
        padding_mask = torch.ones(self.max_edges, dtype=torch.bool) # True 表示是填充的废弃位置
        
        if actual_n > 0:
            x_feat[:actual_n, :] = torch.tensor(x_feat_np[:actual_n, :])
            x_bias[:actual_n, :actual_n] = torch.tensor(x_bias_np[:actual_n, :actual_n])
            padding_mask[:actual_n] = False # False 表示是真实的边
            
        y_type = torch.tensor(sample["y_type"], dtype=torch.long)
        
        # 取第一条目标边作为指针预测的目标 (如果没目标则置为 -1)
        y_targets = sample["y_targets"]
        target_idx = y_targets[0] if len(y_targets) > 0 and y_targets[0] < self.max_edges else -1
        y_target1 = torch.tensor(target_idx, dtype=torch.long)
        
        return {
            "x_feat": x_feat,
            "x_bias": x_bias,
            "padding_mask": padding_mask,
            "y_type": y_type,
            "y_target1": y_target1
        }

# ==========================================
# 🧠 步骤二：Graph Editor Foundation Model
# ==========================================
class GraphEditorTransformer(nn.Module):
    def __init__(self, feature_dim=8, hidden_dim=512, n_heads=4, n_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.edge_embedding = nn.Linear(feature_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # --- Actor Heads ---
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, len(ACTION_VOCAB)) # 5 分类
        )
        
        # 指针网络
        self.pointer_query = nn.Linear(hidden_dim, hidden_dim)
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x_feat, padding_mask):
        B, N, D = x_feat.shape
        
        # 1. 特征升维
        tokens = self.edge_embedding(x_feat) 
        
        # 2. 全局交互 (此处利用 src_key_padding_mask 屏蔽掉 padding 的边)
        # TODO 后期升级：这里可以手写 Attention 加上传入的 x_bias 拓扑偏置
        encoded_tokens = self.transformer(tokens, src_key_padding_mask=padding_mask)
        
        # 3. 提取全局图状态 (屏蔽掉 padding 求平均)
        active_tokens = encoded_tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        valid_counts = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
        global_context = active_tokens.sum(dim=1) / valid_counts
            
        # 4. 预测 Action Type
        type_logits = self.type_head(global_context) # (B, 5)
        
        # 5. 预测 Target Edge
        query = self.pointer_query(global_context).unsqueeze(1) # (B, 1, hidden_dim)
        keys = self.pointer_key(encoded_tokens)                 # (B, N, hidden_dim)
        pointer_logits = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) # (B, N)
        
        # 强制把 padding 的位置打分设为极小值，防止被 softmax 选中
        pointer_logits = pointer_logits.masked_fill(padding_mask, float('-inf'))
            
        return type_logits, pointer_logits

# ==========================================
# 🚀 步骤三：一体化训练主循环
# ==========================================
def train():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    ACTION_LOGS_DIR = os.path.join(PROJECT_ROOT, "action_logs")
    DATASET_CACHE = os.path.join(CURRENT_DIR, "expert_bc_dataset.pkl")
    
    # 🌟 智能数据管道：如果 PKL 缓存不存在，立刻自动调用 data_builder 生成！
    if not os.path.exists(DATASET_CACHE) or True: # 改为 False 可以避免每次重构
        print("🔄 Building dataset from action logs...")
        build_dataset_from_logs(ACTION_LOGS_DIR, DATASET_CACHE)
        
    # 加载数据
    print(f"📦 Loading dataset from {DATASET_CACHE}")
    with open(DATASET_CACHE, 'rb') as f:
        raw_data = pickle.load(f)
        
    dataset = GraphEditingDataset(raw_data, max_edges=64)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 初始化模型与设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Using device: {device}")
    
    model = GraphEditorTransformer(feature_dim=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    epochs = 100
    best_loss = float('inf') # 🌟 新增：用来记录历史最低 Loss
    MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, "graph_editor_best.pth")
    print("\n🚀 Starting Training...")
    
    for epoch in range(epochs):
        model.train()
        total_type_loss, total_ptr_loss = 0, 0
        correct_type, correct_ptr, total_ptr_targets = 0, 0, 0
        
        for batch in dataloader:
            x_feat = batch["x_feat"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            y_type = batch["y_type"].to(device)
            y_target1 = batch["y_target1"].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            type_logits, pointer_logits = model(x_feat, padding_mask)
            
            # --- 损失计算 1：动作类型 ---
            loss_type = F.cross_entropy(type_logits, y_type)
            
            # --- 损失计算 2：指针目标 ---
            # 过滤掉不需要预测目标的动作 (y_target1 == -1，比如 Done)
            valid_ptr_mask = y_target1 != -1
            if valid_ptr_mask.sum() > 0:
                loss_ptr = F.cross_entropy(
                    pointer_logits[valid_ptr_mask], 
                    y_target1[valid_ptr_mask]
                )
            else:
                loss_ptr = torch.tensor(0.0, device=device)
                
            # 联合 Loss
            loss = loss_type + loss_ptr
            loss.backward()
            optimizer.step()
            
            # --- 统计准确率 ---
            total_type_loss += loss_type.item()
            total_ptr_loss += loss_ptr.item()
            
            pred_type = type_logits.argmax(dim=-1)
            correct_type += (pred_type == y_type).sum().item()
            
            if valid_ptr_mask.sum() > 0:
                pred_ptr = pointer_logits[valid_ptr_mask].argmax(dim=-1)
                correct_ptr += (pred_ptr == y_target1[valid_ptr_mask]).sum().item()
                total_ptr_targets += valid_ptr_mask.sum().item()
                
        # 打印日志
        avg_type_acc = correct_type / len(dataset) * 100
        avg_ptr_acc = (correct_ptr / total_ptr_targets * 100) if total_ptr_targets > 0 else 0.0
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:02d}/{epochs}] | "
                  f"Type Loss: {total_type_loss/len(dataloader):.4f} (Acc: {avg_type_acc:.1f}%) | "
                  f"Ptr Loss: {total_ptr_loss/len(dataloader):.4f} (Acc: {avg_ptr_acc:.1f}%)")
        
        epoch_avg_loss = (total_type_loss + total_ptr_loss) / len(dataloader)
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            # 保存模型的 State Dict (纯权重字典，最安全的保存方式)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  👉 [Checkpoint] Best model saved to disk! (Loss: {best_loss:.4f})")
    
    # 循环彻底结束后
    print(f"\n🎉 Training Complete! The best weights are safely stored at:\n{MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()