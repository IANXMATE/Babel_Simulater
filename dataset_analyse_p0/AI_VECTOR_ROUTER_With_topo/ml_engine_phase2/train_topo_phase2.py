import os
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ==========================================
# ⚙️ 配置参数
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOPO_DATA_DIR = "annotations_topo"
TOPO_DATA_DIR = os.path.join(SCRIPT_DIR, "../" + TOPO_DATA_DIR)
MODEL_SAVE_DIR = SCRIPT_DIR
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

BATCH_SIZE = 64
HIDDEN_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 3
EPOCHS = 200
LR = 5e-4

# 拓扑事件类型映射 (0: 无连接, 1: 端点对接, 2: T型搭接, 3: X型交叉)
EVENT_TYPE_MAP = {"NONE": 0, "E2E": 1, "T": 2, "X": 3}

# ==========================================
# 📊 1. 数据集加载器 (解析 5 层架构)
# ==========================================
class TopoDataset(Dataset):
    def __init__(self, data_dir, max_strokes=30):
        self.max_strokes = max_strokes
        self.samples = []
        
        file_paths = glob.glob(os.path.join(data_dir, "*_topo.json"))
        for fp in file_paths:
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for hex_key, char_data in data.items():
                        strokes = char_data.get("strokes", [])
                        if not strokes or len(strokes) > max_strokes: continue
                        
                        # 提取特征 (Layer 1)
                        stroke_features = self._extract_stroke_features(strokes)
                        
                        # 提取标签 (Layer 3: Topology Events)
                        edge_labels, t_labels = self._extract_topology_labels(
                            len(strokes), char_data.get("topology_events", [])
                        )
                        
                        self.samples.append({
                            "hex_key": hex_key,
                            "features": stroke_features,      # [num_strokes, 12]
                            "edge_labels": edge_labels,       # [num_strokes, num_strokes]
                            "t_labels": t_labels,             # [num_strokes, num_strokes, 2]
                            "mask": torch.ones(len(strokes))  # 标记有效笔画
                        })
            except Exception as e:
                print(f"Error loading {fp}: {e}")
                
        print(f"✅ 成功加载 {len(self.samples)} 个高质量拓扑字符样本。")

    def _extract_stroke_features(self, strokes):
        features = []
        for s in strokes:
            # 展平 mother_bezier (4x2=8) 和 width_bezier (4) => 12 维特征
            m_bez = np.array(s["mother_bezier"]).flatten() / 400.0 # 归一化到 0~1
            w_bez = np.array(s["width_bezier"]) / 20.0             # 宽度归一化
            features.append(np.concatenate([m_bez, w_bez]))
        return torch.tensor(np.array(features), dtype=torch.float32)

    def _extract_topology_labels(self, num_strokes, events):
        edge_labels = torch.zeros((num_strokes, num_strokes), dtype=torch.long)
        # 存储 [t_a, t_b] 或 [guest_t, host_t]
        t_labels = torch.zeros((num_strokes, num_strokes, 2), dtype=torch.float32) 
        
        for ev in events:
            ev_type = ev["type"]
            type_idx = EVENT_TYPE_MAP.get(ev_type, 0)
            
            if ev_type == "E2E" or ev_type == "X":
                a, b = ev["stroke_a"], ev["stroke_b"]
                ta, tb = ev["t_a"], ev["t_b"]
                if a < num_strokes and b < num_strokes:
                    edge_labels[a, b] = edge_labels[b, a] = type_idx
                    t_labels[a, b] = torch.tensor([ta, tb])
                    t_labels[b, a] = torch.tensor([tb, ta])
                    
            elif ev_type == "T":
                guest, host = ev["guest"], ev["host"]
                gt, ht = ev["guest_t"], ev["host_t"]
                if guest < num_strokes and host < num_strokes:
                    edge_labels[guest, host] = type_idx
                    # T 型搭接不对称，特意设定 [guest_t, host_t]
                    t_labels[guest, host] = torch.tensor([gt, ht]) 

        return edge_labels, t_labels

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        N = len(s["features"])
        pad_len = self.max_strokes - N
        
        # 补齐到 max_strokes，方便 Batch 计算
        feat_pad = torch.nn.functional.pad(s["features"], (0, 0, 0, pad_len))
        edge_pad = torch.nn.functional.pad(s["edge_labels"], (0, pad_len, 0, pad_len))
        t_pad = torch.nn.functional.pad(s["t_labels"], (0, 0, 0, pad_len, 0, pad_len))
        mask_pad = torch.nn.functional.pad(s["mask"], (0, pad_len))
        
        return feat_pad, edge_pad, t_pad, mask_pad


# ==========================================
# 🧠 2. 模型架构 (FontTopologyGPT 基座)
# ==========================================
class FontTopologyGPT(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        
        # 1. 特征升维
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. 笔画级上下文交互 (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 多任务 Head (Multi-Task Learning)
        # 预测搭接类型 (4 分类: 0, 1, 2, 3)
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        
        # 回归参数 t (2 维: t_a, t_b)
        self.t_regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid() # 强行约束在 0~1 之间！
        )

    def forward(self, x, mask):
        # x: [B, N, 12], mask: [B, N]
        emb = self.embedding(x) # [B, N, H]
        
        # Transformer 需要的 padding_mask (True 表示忽略)
        src_key_padding_mask = (mask == 0) 
        encoded = self.transformer(emb, src_key_padding_mask=src_key_padding_mask) # [B, N, H]
        
        B, N, H = encoded.shape
        
        # 构建所有笔画对的组合特征 [B, N, N, H*2]
        x_i = encoded.unsqueeze(2).expand(B, N, N, H) # 行
        x_j = encoded.unsqueeze(1).expand(B, N, N, H) # 列
        pair_features = torch.cat([x_i, x_j], dim=-1)
        
        # 输出预测
        edge_logits = self.edge_classifier(pair_features) # [B, N, N, 4]
        t_preds = self.t_regressor(pair_features)         # [B, N, N, 2]
        
        return edge_logits, t_preds


# ==========================================
# 🚀 3. 训练主循环
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔥 使用设备: {device}")
    
    dataset = TopoDataset(TOPO_DATA_DIR)
    if len(dataset) == 0:
        print("❌ 数据集为空，请检查 annotations_topo 文件夹下是否存在合法的 _topo.json 数据！")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = FontTopologyGPT(hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # 损失函数隔离与定制
    criterion_cls = nn.CrossEntropyLoss(ignore_index=-100) # 忽略 padding 的 loss
    criterion_reg = nn.MSELoss(reduction='none')           # 自定义回归 Loss
    
    model.train()
    print("🚀 开始训练...")
    
    for epoch in range(1, EPOCHS + 1):
        total_loss, cls_loss_sum, reg_loss_sum = 0, 0, 0
        
        for batch_idx, (feat, edge_labels, t_labels, mask) in enumerate(dataloader):
            feat, edge_labels, t_labels, mask = feat.to(device), edge_labels.to(device), t_labels.to(device), mask.to(device)
            
            optimizer.zero_grad()
            
            edge_logits, t_preds = model(feat, mask)
            
            # 生成 Pair Mask 忽略 padding 区域产生的配对
            B, N = mask.shape
            pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1) # [B, N, N]
            
            # 1. 分类 Loss
            active_logits = edge_logits.reshape(-1, 4)
            active_labels = edge_labels.reshape(-1)
            # 把被 mask 掉的部分的 label 设为 -100 (被 CrossEntropy 忽略)
            active_labels[pair_mask.reshape(-1) == 0] = -100 
            loss_cls = criterion_cls(active_logits, active_labels)
            
            # 2. 回归 Loss (仅对真实发生拓扑连接的边计算 t 的误差)
            valid_edge_mask = (edge_labels > 0) & (pair_mask == 1) # 只有发生 E2E, T, X 的地方才算 MSE
            if valid_edge_mask.sum() > 0:
                loss_reg = criterion_reg(t_preds, t_labels)
                # 仅累加有效位置的误差
                loss_reg = (loss_reg[valid_edge_mask]).mean()
            else:
                loss_reg = torch.tensor(0.0).to(device)
                
            # 联合 Loss
            loss = loss_cls + 2.0 * loss_reg 
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            cls_loss_sum += loss_cls.item()
            reg_loss_sum += loss_reg.item()
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{EPOCHS} | Total Loss: {total_loss/len(dataloader):.4f} "
                  f"(Cls: {cls_loss_sum/len(dataloader):.4f}, Reg: {reg_loss_sum/len(dataloader):.4f})")
            
    # 保存模型权重
    save_path = os.path.join(MODEL_SAVE_DIR, "topo_gpt_baseline.pth")
    torch.save(model.state_dict(), save_path)
    print(f"🎉 训练完成！模型已保存至: {save_path}")

if __name__ == "__main__":
    train()