import torch
import matplotlib.pyplot as plt
# 仅仅导入模型结构类，不导入变量
from train import CharacterSetGenerator

# 🌟 直接在这里写死我们刚刚训练时使用的维度参数
LATENT_DIM = 16
HIDDEN_DIM = 32  # 因为你代码里写的是 2 * LATENT_DIM

def visualize_alien_character():
    print("正在唤醒模型...")
    # 1. 实例化模型
    model = CharacterSetGenerator(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM)
    
    # 2. 加载刚刚跑出来的热乎权重
    model.load_state_dict(torch.load("babel_generator_epoch_20.pth", map_location="cpu"))
    model.eval()

    # 3. 注入“灵魂”：生成一个纯随机的高斯噪声特征 (相当于让模型在大脑中随机构思一个字)
    z = torch.randn(1, LATENT_DIM)

    # 4. 造字！
    with torch.no_grad():
        preds = model(z)  # 输出形状: [1, seq_len, 5] (x, y, dx, dy, w)

    # 5. 提取坐标点
    # 取出第一个 batch 的所有生成点
    points = preds[0].numpy()  
    
    # 你的模型输出有5个值，前两个是中心坐标 x, y
    x = points[:, 0]
    y = points[:, 1]

    # 6. 用 Matplotlib 渲染出来
    plt.figure(figsize=(6, 6))
    
    # 我们先把所有的中心点画出来，看看它们的位置分布
    plt.scatter(x, y, c='black', alpha=0.7, s=30) 
    
    plt.title("Babel Alien Character (Epoch 1)", fontsize=14)
    plt.axis('equal')           # 保持 XY 比例一致
    plt.gca().invert_yaxis()    # 图像 Y 轴通常向下，翻转一下符合直觉
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 保存并展示
    plt.savefig("alien_char_test.png")
    print("✨ 外星文字已渲染完毕，请查看项目文件夹下的 alien_char_test.png！")
    
    # 如果你在有界面的环境，这会弹出一个窗口；如果没有，直接看图片文件就行
    plt.show()

if __name__ == "__main__":
    visualize_alien_character()