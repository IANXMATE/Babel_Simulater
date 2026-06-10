import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

# 从我们刚刚写的 step1 导入数据集
try:
    from step1_data_pipeline import AlienStrokeDataset
except ImportError:
    print("❌ 找不到 step1_data_pipeline.py，请确保它们在同一目录下。")
    exit(1)

# ==========================================
# 🧠 核心模块 1：直通估计器 VQ 层 (Vector Quantizer)
# ==========================================
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # 这就是我们未来的“拼音字典”，随机初始化
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs):
        # 计算输入与字典中每个 Token 的 L2 距离
        # (x - y)^2 = x^2 + y^2 - 2xy
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self.embedding.weight.t()))
            
        # 找到最近的字典 Token 索引
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # 提取出对应的离散向量
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding.weight)

        # 🌟 VQ Loss
        # e_latent_loss: 字典不要跑太偏，向 Encoder 靠近 (Commitment)
        # q_latent_loss: Encoder 输出要向字典靠近
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # 🌟 直通估计器 (Straight-Through Estimator)
        # 前向传播用量化后的离散值，反向传播时梯度直接跳过量化步骤，传给 inputs
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices.squeeze(1)


# ==========================================
# 🧠 核心模块 2：Stroke VQ-VAE 架构
# ==========================================
class BezierVQVAE(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=256, latent_dim=64, num_tokens=1024, commitment_cost=0.25):
        super(BezierVQVAE, self).__init__()
        
        # Encoder: 提取高维几何特征
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # VQ 层
        self.vq_layer = VectorQuantizer(num_tokens, latent_dim, commitment_cost)
        
        # Decoder: 重构贝塞尔参数
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim) # 输出 9 维重构结果
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, token_ids = self.vq_layer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, token_ids


# ==========================================
# 🚀 训练主循环
# ==========================================
def train():
    # 1. 读取配置
    config_path = "model_config.json"
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    config_path = config_path=os.path.join(SCRIPT_DIR, config_path)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    mc = config['model_config']
    tc = config['train_config']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): # Mac 芯片加速支持
        device = torch.device("mps")
    print(f"⚙️ 训练设备: {device}")

    # 2. 准备数据
    dataset = AlienStrokeDataset(config_path)
    
    # 你的测试集很小（500多条），我们动态调一下 batch_size 保证能跑起来
    actual_batch_size = min(tc['batch_size'], len(dataset) // 2)
    if actual_batch_size < 16: actual_batch_size = 16
    
    dataloader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=True)

    # 3. 初始化模型与优化器
    model = BezierVQVAE(
        input_dim=mc['input_dim'],
        hidden_dim=mc['hidden_dim'],
        latent_dim=mc['latent_dim'],
        num_tokens=mc['num_tokens'],
        commitment_cost=mc['commitment_cost']
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=tc['learning_rate'])
    
    # 确保存档目录存在
    save_dir = tc['save_dir']
    save_dir = os.path.join(SCRIPT_DIR, save_dir)
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n🚀 开始训练 Stroke-VQ 字典...")
    epochs = tc['epochs']
    
    for epoch in range(epochs):
        model.train()
        total_recon_loss = 0
        total_vq_loss = 0
        
        for batch in dataloader:
            x_input = batch['vq_input'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            x_recon, vq_loss, _ = model(x_input)
            
            # 🌟 我们的终极目标：让重构出来的曲线，尽量贴合原来的曲线
            recon_loss = F.mse_loss(x_recon, x_input)
            
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            
        # 打印日志
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_recon = total_recon_loss / len(dataloader)
            avg_vq = total_vq_loss / len(dataloader)
            print(f"Epoch [{epoch+1:03d}/{epochs}] | Recon Loss (MSE): {avg_recon:.6f} | VQ Loss: {avg_vq:.6f}")

    # 4. 保存最终模型权重
    save_path = os.path.join(save_dir, "stroke_vqvae_best.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n🎉 训练完成！模型已保存至: {save_path}")
    
    # ==========================================
    # 🎯 终极验证：取一条数据跑一下，看看它学到了什么
    # ==========================================
    model.eval()
    print("\n🧐 --- 验证字典的威力 ---")
    with torch.no_grad():
        sample = dataset[0] # 取第一条笔画
        x_test = sample['vq_input'].unsqueeze(0).to(device) # Shape: (1, 9)
        
        # 只取 Token ID
        z_e = model.encoder(x_test)
        _, _, token_id = model.vq_layer(z_e)
        
        print(f"1️⃣ 原始笔画的纯几何特征 (9维):\n{sample['vq_input'].numpy().round(3)}")
        print(f"2️⃣ 经过 Encoder，它被压缩成了字典里的第 {token_id[0].item()} 号 Token！")
        
        # 强行拿这个 Token ID 去解码
        test_encoding = torch.zeros(1, mc['num_tokens'], device=device)
        test_encoding[0, token_id[0]] = 1.0
        z_q_test = torch.matmul(test_encoding, model.vq_layer.embedding.weight)
        
        x_recon_test = model.decoder(z_q_test)
        print(f"3️⃣ Decoder 仅根据 Token ID 重构出的几何特征:\n{x_recon_test[0].cpu().numpy().round(3)}")

if __name__ == "__main__":
    train()