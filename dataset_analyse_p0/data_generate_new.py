import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

# 确保从你的 train.py 导入正确的类
from train import ParametricStrokeTransformer, ModelConfig 

def generate_cyberpunk_character(
    model_path="parametric_transformer.pth", 
    max_steps=50, 
    temperature=1.0,
    min_steps=10,       # 🛡️ 至少生成的笔画数 (保底打工)
    force_steps=None,   # 🎯 强制精准笔画数 (绝对指令)
    min_length=5.0      # 📏 强制最小物理长度 (防止 AI 画小点摸鱼)
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
    print(f"🛰️ 启动赛博生成引擎 | 温度: {temperature} | 协议: [{mode_text}] | 最小长度: {min_length}")

    # ==========================================
    # 1. 强制指令自回归生成 (Override Engine)
    # ==========================================
    with torch.no_grad():
        for step in range(max_steps):
            inputs = torch.stack(current_seq).unsqueeze(0)
            pred_cont, pred_state_logits = model(inputs)
            
            # 物理量预测与创造力注入 (防均值坍缩)
            next_cont = pred_cont[0, -1, :] 
            noise_scale = 0.5 * temperature
            next_cont_noisy = next_cont + torch.randn_like(next_cont) * noise_scale
            dx, dy, L, theta, kappa, W = next_cont_noisy.tolist()
            
            # 🛡️ 终极物理约束：长度不仅要是正数，还必须大于设定的最小阈值！
            L = max(abs(L), min_length)
            W = abs(W) 
            
            # 原始 AI 意图采样
            probs = F.softmax(pred_state_logits[0, -1, :] / temperature, dim=-1)
            p_ai = torch.multinomial(probs, 1).item()
            
            # 🎯 核心逻辑：上帝视角的协议覆盖 (Override)
            p_final = float(p_ai)
            
            if force_steps is not None:
                # 绝对指令模式
                if step < force_steps - 1:
                    p_final = 0.0  
                elif step == force_steps - 1:
                    p_final = 1.0  
            else:
                # 保底限制模式
                if step < min_steps - 1:
                    p_final = 0.0
                    
            new_token = torch.tensor([dx, dy, L, theta, kappa, W, p_final], device=device)
            generated_data.append(new_token.cpu().numpy())
            current_seq.append(new_token)
            
            if p_final == 1.0:
                print(f"🏁 指令完成，模型在第 {step+1} 步收笔。")
                break

    # ==========================================
    # 2. 赛博全息渲染 (Cyber-Holographic Engine)
    # ==========================================
    fig, ax = plt.subplots(figsize=(7, 7))
    bg_color = '#030613' 
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.grid(True, color='#0a192f', linestyle='-', linewidth=0.5, alpha=0.8)
    
    prev_end_x, prev_end_y = 0.0, 0.0 
    CORE_COLOR = '#FFFFFF'
    GLOW_COLOR = '#00F0FF'
    MAX_ENERGY = 3.0
    MIN_ENERGY = 0.5

    for i, token in enumerate(generated_data):
        dx, dy, L, theta, kappa, W, P = token
        if P == 1.0: break
            
        start_x = prev_end_x + dx
        start_y = prev_end_y + dy
        
        steps = 100 
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

        for j in range(1, len(pts_x)):
            x1, y1 = pts_x[j-1], pts_y[j-1]
            x2, y2 = pts_x[j], pts_y[j]
            
            progress = j / steps
            energy_width = MAX_ENERGY - progress * (MAX_ENERGY - MIN_ENERGY)
            
            # 辉光叠加效应
            ax.plot([x1, x2], [y1, y2], color=GLOW_COLOR, linewidth=energy_width * 4, alpha=0.10, solid_capstyle='round')
            ax.plot([x1, x2], [y1, y2], color=GLOW_COLOR, linewidth=energy_width * 1.5, alpha=0.40, solid_capstyle='round')
            ax.plot([x1, x2], [y1, y2], color=CORE_COLOR, linewidth=energy_width * 0.4, alpha=1.0, solid_capstyle='round')

        prev_end_x, prev_end_y = pts_x[-1], pts_y[-1]

    # ==========================================
    # 3. HUD UI 动态更新 (加入了 MIN_LEN 显示)
    # ==========================================
    ax.text(0.03, 0.97, 'SYS.OVERRIDE // ACTIVE', 
            transform=ax.transAxes, color=GLOW_COLOR, fontsize=8, fontfamily='monospace', va='top', alpha=0.7)
    
    # 动态显示强制指令的配置，增加了 L 参数监控
    ui_mode = f"EXACT_{force_steps}" if force_steps else f"MIN_{min_steps}"
    ui_text = f'PROTOCOL: [{ui_mode}]\nSTROKES:  [{len(generated_data)}]\nMIN_LEN:  [{min_length:.1f}]'
    
    ax.text(0.03, 0.88, ui_text, 
            transform=ax.transAxes, color='#FFFFFF', fontsize=10, fontfamily='monospace', fontweight='bold', va='top')
            
    ax.text(0.97, 0.03, 'SCALE: 7D-TENSOR', 
            transform=ax.transAxes, color=GLOW_COLOR, fontsize=8, fontfamily='monospace', ha='right', va='bottom', alpha=0.7)

    for spine in ax.spines.values():
        spine.set_edgecolor(GLOW_COLOR)
        spine.set_linewidth(1.5)
        spine.set_alpha(0.5)
        
    ax.tick_params(axis='both', colors=GLOW_COLOR, labelsize=6)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.invert_yaxis() 
    ax.axis('equal')
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    plt.show()

if __name__ == "__main__":
    # 测试绝对指令模式：必须画 15 笔，且每一笔不能短于 3.0 的物理单位
    generate_cyberpunk_character(force_steps=5, min_length=10.0, temperature=1.0)