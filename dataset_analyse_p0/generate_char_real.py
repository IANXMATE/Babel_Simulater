import torch
import torch.nn.functional as F
import math
import random
import matplotlib.pyplot as plt
import numpy as np

# 确保从你的 train.py 导入正确的类
from train import ParametricStrokeTransformer, ModelConfig 

def generate_stable_character(
    model_path="parametric_transformer.pth", 
    max_steps=50, 
    initial_temperature=1.0, # 初始温度
    min_steps=10,       
    force_steps=None,   
    min_length=0.15     
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig()
    model = ParametricStrokeTransformer(cfg).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ 成功加载大脑: {model_path}")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return

    current_seq = [torch.zeros(cfg.input_dim, device=device)]
    generated_data = []
    
    mode_text = f"FORCE:{force_steps}" if force_steps else f"MIN:{min_steps}"
    print(f"✍️ 启动防坍缩引擎 | 协议: [{mode_text}] | 启用温度退火...")

    with torch.no_grad():
        for step in range(max_steps):
            inputs = torch.stack(current_seq).unsqueeze(0)
            pred_cont, pred_state_logits = model(inputs)
            
            # ==========================================
            # 🛡️ 核心修复 1：动态温度退火 (Temperature Decay)
            # 每多画一笔，温度下降 10%。越往后越稳，防止长序列崩溃！
            # ==========================================
            current_temp = initial_temperature * (0.9 ** step)
            current_temp = max(current_temp, 0.1) # 设定保底温度
            
            next_cont = pred_cont[0, -1, :] 
            dx, dy, L, theta, kappa, W = next_cont.tolist()
            
            # 注入随着温度衰减的温和噪声
            dx += random.gauss(0, 0.15 * current_temp)
            dy += random.gauss(0, 0.15 * current_temp)
            theta += random.gauss(0, 0.3 * current_temp)
            
            # 适度放大曲率并加入噪声，但不再乘以夸张的倍数
            kappa = (kappa * 1.5) + random.gauss(0, 0.5 * current_temp)
            
            # ==========================================
            # 🛡️ 核心修复 2：数值防爆伞 (Value Clipping)
            # 绝对不把超出正常范围的离谱数据喂回给 Transformer
            # ==========================================
            L = max(abs(L + random.gauss(0, 0.2 * current_temp)), min_length)
            L = min(L, 10.0) # 防止单笔长到飞出宇宙
            
            kappa = max(min(kappa, 2.5), -2.5) # 把曲率锁死在安全范围内
            W = abs(W)
            
            # --- LLM 原生状态控制 (Logit Masking) ---
            logits = pred_state_logits[0, -1, :].clone()
            
            if force_steps is not None:
                if step < force_steps - 1:
                    logits[1] = -float('inf') 
                elif step == force_steps - 1:
                    logits[0] = -float('inf') 
            else:
                if step < min_steps - 1:
                    logits[1] = -float('inf') 
            
            # 使用衰减后的当前温度进行 Softmax
            probs = F.softmax(logits / current_temp, dim=-1)
            p_final = float(torch.multinomial(probs, 1).item())
                    
            new_token = torch.tensor([dx, dy, L, theta, kappa, W, p_final], device=device)
            generated_data.append(new_token.cpu().numpy())
            current_seq.append(new_token)
            
            if p_final == 1.0:
                print(f"🏁 任务完成，模型在第 {step+1} 步稳稳收笔。")
                break

    # ==========================================
    # 纯数学方程渲染 (100% 完美解耦)
    # ==========================================
    plt.figure(figsize=(6, 6))
    colors = plt.cm.get_cmap('tab10', max(10, len(generated_data)))
    
    prev_end_x, prev_end_y = 0.0, 0.0 
    MAX_WIDTH = 5.0  
    MIN_WIDTH = 1.0  

    for i, token in enumerate(generated_data):
        dx, dy, L, theta, kappa, W, P = token
        if P == 1.0: break
            
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
    
    title_mode = f"Force {force_steps} Strokes" if force_steps else f"Min {min_steps} Strokes"
    plt.title(f"Label: AI Generated ({title_mode})\n(Stable Autoregressive + Math Engine)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 就算你强制它画 20 笔，它也会在中后期逐渐趋于冷静，画出一个复杂的几何阵列而不会崩溃！
    generate_stable_character(force_steps=6, min_length=0.15, initial_temperature=1.0)