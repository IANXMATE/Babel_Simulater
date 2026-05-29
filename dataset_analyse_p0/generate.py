import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

from train import ParametricStrokeTransformer, ModelConfig 

def generate_unconditional_parametric(model_path="parametric_transformer.pth", max_steps=50, temperature=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = ModelConfig()
    model = ParametricStrokeTransformer(cfg).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ 成功加载模型权重: {model_path}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    current_seq = [torch.zeros(cfg.input_dim, device=device)]
    generated_data = []
    
    print(f"✍️ 模型开始自由挥毫 (Temperature: {temperature})...")

    with torch.no_grad():
        for step in range(max_steps):
            inputs = torch.stack(current_seq).unsqueeze(0)
            pred_cont, pred_state_logits = model(inputs)
            
            # ==========================================
            # 🎯 核心修复：打破 MSE 均值坍缩！
            # ==========================================
            next_cont = pred_cont[0, -1, :] 
            
            # 注入高斯噪声以模拟真实手写的方差，temperature 此时也用来控制物理形变的大小
            noise_scale = 0.5 * temperature
            next_cont_noisy = next_cont + torch.randn_like(next_cont) * noise_scale
            
            dx, dy, L, theta, kappa, W = next_cont_noisy.tolist()
            
            # 物理常识收敛：长度和线宽绝对不能是负数！
            L = abs(L)
            W = abs(W)
            
            probs = F.softmax(pred_state_logits[0, -1, :] / temperature, dim=-1)
            p = torch.multinomial(probs, 1).item()
            
            # 强制前 3 笔不能交白卷
            if step < 3:
                p = 0.0
            
            new_token = torch.tensor([dx, dy, L, theta, kappa, W, float(p)], device=device)
            generated_data.append(new_token.cpu().numpy())
            current_seq.append(new_token)
            
            if p == 1.0:
                print(f"🏁 模型在第 {step+1} 步主动收笔结束！")
                break

    # ==========================================
    # 纯数学方程渲染 (原封不动复刻你的 Math Engine)
    # ==========================================
    plt.figure(figsize=(6, 6))
    colors = plt.cm.get_cmap('tab10', max(10, len(generated_data)))
    
    prev_end_x, prev_end_y = 0.0, 0.0 
    MAX_WIDTH = 5.0  
    MIN_WIDTH = 1.0  

    for i, token in enumerate(generated_data):
        dx, dy, L, theta, kappa, W, P = token
        
        if P == 1.0:
            break
            
        start_x = prev_end_x + dx
        start_y = prev_end_y + dy
        
        steps = 50 
        t_vals = np.linspace(0, 1.0, steps)
        pts_x, pts_y = [], []
        
        for t in t_vals:
            if abs(kappa) < 1e-4:
                x = start_x + L * t * math.cos(theta)
                y = start_y + L * t * math.sin(theta)
            else:
                x = start_x + (L / kappa) * (math.sin(theta + kappa * t) - math.sin(theta))
                y = start_y - (L / kappa) * (math.cos(theta + kappa * t) - math.cos(theta))
            pts_x.append(x)
            pts_y.append(y)

        color = colors(i % 10)
        plt.plot(pts_x[0], pts_y[0], marker='o', markersize=4, color=color)
        
        for j in range(1, len(pts_x)):
            x1, y1 = pts_x[j-1], pts_y[j-1]
            x2, y2 = pts_x[j], pts_y[j]
            progress = j / steps
            dynamic_width = MAX_WIDTH - progress * (MAX_WIDTH - MIN_WIDTH)
            plt.plot([x1, x2], [y1, y2], color=color, linewidth=dynamic_width, solid_capstyle='round')

        prev_end_x, prev_end_y = pts_x[-1], pts_y[-1]

    plt.gca().invert_yaxis() 
    plt.axis('equal')
    plt.axis('off')
    plt.title(f"AI Generated Character\n(Noisy Decoding | Temp: {temperature})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 尝试把 temperature 调到 1.0 甚至 1.5，让它疯狂发挥
    generate_unconditional_parametric(temperature=1.0)