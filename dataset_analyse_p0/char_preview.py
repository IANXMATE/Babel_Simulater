import torch
import random
import math
import matplotlib.pyplot as plt
import numpy as np

def preview_random_sample(data_path="omniglot_parametric_7d.pt"):
    # 1. 尝试加载数据集
    try:
        dataset = torch.load(data_path)
        print(f"✅ 成功加载数据集，共包含 {len(dataset)} 个样本。")
    except FileNotFoundError:
        print(f"❌ 找不到文件 {data_path}，请确认你已经运行了数据预处理脚本。")
        return

    # 2. 随机抽签
    sample = random.choice(dataset)
    seq = sample['sequence']
    label = sample.get('label', 'Unknown')
    
    print(f"🎲 随机抽中字符标签: {label}")
    print(f"📏 该字符由 {len(seq)-1} 笔画构成 (已自动忽略结束符)")

    # 3. 渲染引擎初始化
    plt.figure(figsize=(6, 6))
    colors = plt.cm.get_cmap('tab10', max(10, len(seq)))
    
    # 记录上一笔的物理终点坐标
    prev_x, prev_y = 0.0, 0.0
    
    # 画笔粗细参数 (模拟提按)
    MAX_WIDTH = 4.5
    MIN_WIDTH = 1.0

    # 4. 开始解析 7 维方程
    for i, token in enumerate(seq):
        # 解析 7 维 Token: [dx, dy, L, θ, κ, W, P]
        dx, dy, L, theta, kappa, W, P = token.tolist()
        
        # 遇到结束符，直接停止渲染
        if P == 1.0: 
            break
            
        # 确定这一笔的绝对起点位置
        start_x = prev_x + dx
        start_y = prev_y + dy
        
        # 积分采样步数：线越长，采样的点越多，保证圆弧绝对顺滑
        steps = max(20, int(L * 100)) 
        t_vals = np.linspace(0, 1.0, steps)
        pts_x, pts_y = [], []
        
        for t in t_vals:
            # 核心黑魔法：根据曲率还原几何轨迹
            if abs(kappa) < 1e-4:
                # 是一条直线
                x = start_x + L * t * math.cos(theta)
                y = start_y + L * t * math.sin(theta)
            else:
                # 是一条平滑圆弧
                x = start_x + (L / kappa) * (math.sin(theta + kappa * t) - math.sin(theta))
                y = start_y - (L / kappa) * (math.cos(theta + kappa * t) - math.cos(theta))
            pts_x.append(x)
            pts_y.append(y)
            
        color = colors(i % 10)
        
        # 绘制起笔的“顿笔”圆点
        plt.plot(pts_x[0], pts_y[0], marker='o', markersize=5, color=color)
        
        # 绘制线条，注入渐变笔锋
        for j in range(1, len(pts_x)):
            progress = j / steps
            # 随着笔画延伸，线条逐渐变细
            dynamic_width = MAX_WIDTH - progress * (MAX_WIDTH - MIN_WIDTH)
            
            plt.plot([pts_x[j-1], pts_x[j]], [pts_y[j-1], pts_y[j]], 
                     color=color, linewidth=dynamic_width, solid_capstyle='round')
            
        # 更新记录状态
        prev_x, prev_y = pts_x[-1], pts_y[-1]
        
    # 5. 画布设置与展示
    plt.gca().invert_yaxis()  # Y轴反转以匹配真实屏幕习惯
    plt.axis('equal')
    plt.axis('off')
    plt.title(f"Random Sample: {label}\n(Parametric Equation Render)", fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    preview_random_sample()