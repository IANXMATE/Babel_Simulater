import torch
import random
import math
import matplotlib.pyplot as plt
import numpy as np

def preview_parametric_character(data_path="omniglot_parametric_7d.pt"):
    try:
        data = torch.load(data_path)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {data_path}，请确认路径。")
        return

    if not data:
        print("❌ 数据集是空的！")
        return

    # ==========================================
    # 1. 随机抽签
    # ==========================================
    sample = random.choice(data)
    label = sample["label"]
    filename = sample["filename"]
    seq = sample["sequence"]  # Tensor shape: (L, 7)

    print(f"🎲 随机抽取成功！")
    print(f"🏷️ 语言/字符集: {label}")
    print(f"📄 原始文件名: {filename}")
    print(f"📏 压缩后的方程组数量: 仅仅 {len(seq)-1} 笔！")

    # ==========================================
    # 2. 纯数学方程渲染 (Math Engine)
    # ==========================================
    plt.figure(figsize=(6, 6))
    colors = plt.cm.get_cmap('tab10', max(10, len(seq)))
    
    # 初始绝对坐标原点 (我们将以第一笔为基准进行相对偏移)
    prev_end_x, prev_end_y = 0.0, 0.0 
    
    # 画笔粗细设置
    MAX_WIDTH = 5.0  # 顿笔起步宽度
    MIN_WIDTH = 1.0  # 笔锋收尾宽度

    for i, token in enumerate(seq):
        dx, dy, L, theta, kappa, W, P = token.tolist()
        
        # 遇到结束符，停止渲染
        if P == 1.0:
            break
            
        # 1. 计算这一笔的绝对起点 (上一个终点 + 悬空偏移量)
        start_x = prev_end_x + dx
        start_y = prev_end_y + dy
        
        # 2. 对方程进行离散化采样 (将 t 从 0 积分到 1)
        # 因为我们画的是数学曲线，只要把 steps 设得足够高，曲线就绝对平滑！
        steps = 50 
        t_vals = np.linspace(0, 1.0, steps)
        
        pts_x, pts_y = [], []
        
        for t in t_vals:
            # 核心黑魔法：根据曲率 (Kappa) 还原数学方程
            if abs(kappa) < 1e-4:
                # 极限情况：曲率趋近于 0，退化为一次直线方程
                x = start_x + L * t * math.cos(theta)
                y = start_y + L * t * math.sin(theta)
            else:
                # 正常情况：还原为圆弧参数方程
                # 积分公式: x(t) = x0 + (L/k) * [sin(θ + kt) - sin(θ)]
                x = start_x + (L / kappa) * (math.sin(theta + kappa * t) - math.sin(theta))
                y = start_y - (L / kappa) * (math.cos(theta + kappa * t) - math.cos(theta))
                
            pts_x.append(x)
            pts_y.append(y)

        # 3. 开始绘图 (加入毛笔渐变效果)
        color = colors(i % 10)
        
        # 画个小圆点代表起笔处的“顿笔”
        plt.plot(pts_x[0], pts_y[0], marker='o', markersize=4, color=color)
        
        # 分段绘制以实现动态宽度
        for j in range(1, len(pts_x)):
            x1, y1 = pts_x[j-1], pts_y[j-1]
            x2, y2 = pts_x[j], pts_y[j]
            
            # 当前进度比例 (0 到 1)
            progress = j / steps
            
            # 模拟真实的毛笔笔锋：起笔粗，收笔细 (线性衰减)
            dynamic_width = MAX_WIDTH - progress * (MAX_WIDTH - MIN_WIDTH)
            
            plt.plot([x1, x2], [y1, y2], color=color, 
                     linewidth=dynamic_width, 
                     solid_capstyle='round')

        # 4. 状态机流转：把这一笔的真正终点，记录为下一笔的起点基准
        prev_end_x, prev_end_y = pts_x[-1], pts_y[-1]

    # ==========================================
    # 3. 画布展示
    # ==========================================
    plt.gca().invert_yaxis()  # Y轴向下为正
    plt.axis('equal')
    plt.axis('off')
    plt.title(f"Label: {label}\n(7D Parametric Rendering Engine)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    preview_parametric_character()