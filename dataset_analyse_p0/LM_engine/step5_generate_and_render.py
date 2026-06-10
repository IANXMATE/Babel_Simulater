import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import copy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(SCRIPT_DIR, "model_config.json")

try:
    from step2_train_vqvae import BezierVQVAE
    from step4_train_transformer import FontGPT
except ImportError:
    print("❌ 无法导入模型，请确保 step2 和 step4 在同一目录下。")
    exit(1)

# ==========================================
# 📐 动态笔锋几何引擎
# ==========================================
def evaluate_cubic_bezier_with_width(p0, p1, p2, p3, widths, num_points=60):
    """
    同时对坐标和线宽进行三次贝塞尔插值，生成极度平滑的笔锋过渡
    """
    t = np.linspace(0, 1, num_points)
    
    # 坐标插值
    curve_x = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * p2[0] + t**3 * p3[0]
    curve_y = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * p2[1] + t**3 * p3[1]
    
    # 线宽插值 (让粗细变化也像贝塞尔曲线一样顺滑)
    w0, w1, w2, w3 = widths
    curve_w = (1-t)**3 * w0 + 3*(1-t)**2 * t * w1 + 3*(1-t) * t**2 * w2 + t**3 * w3
    
    return np.column_stack([curve_x, curve_y]), curve_w

def generate_stroke_polygon(curve_pts, curve_w):
    """
    完美复刻主程序的法线挤出算法，但支持每个像素点独立的动态线宽！
    """
    dp = np.gradient(curve_pts, axis=0)
    n = np.zeros_like(dp)
    n[:, 0], n[:, 1] = -dp[:, 1], dp[:, 0]
    n_norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-5
    n = n / n_norm
    
    w_vals = curve_w.reshape(-1, 1) # (N, 1) 的动态线宽
    upper = curve_pts + n * w_vals
    lower = curve_pts - n * w_vals
    
    poly = np.vstack([upper, lower[::-1]])
    return poly

def top_k_filtering(logits, top_k=5, filter_value=-float('Inf')):
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    return logits

def apply_topological_snapping(stroke_data_list, snap_threshold=15.0):
    repaired = copy.deepcopy(stroke_data_list)
    n = len(repaired)
    dense_curves = []
    for s in repaired:
        mb = s['mother_bezier']
        # 吸附只看骨架，所以线宽随意给一个 0
        pts, _ = evaluate_cubic_bezier_with_width(mb[0], mb[1], mb[2], mb[3], [0,0,0,0], 100)
        dense_curves.append(pts)

    for i in range(n):
        for idx_i in [0, 3]: 
            pt = np.array(repaired[i]['mother_bezier'][idx_i])
            min_dist, best_match = float('inf'), None
            for j in range(n):
                if i == j: continue
                dists = np.linalg.norm(dense_curves[j] - pt, axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] < min_dist:
                    min_dist, best_match = dists[min_idx], dense_curves[j][min_idx]

            if min_dist < snap_threshold and best_match is not None:
                repaired[i]['mother_bezier'][idx_i] = best_match.tolist()
                mb = repaired[i]['mother_bezier']
                dense_curves[i], _ = evaluate_cubic_bezier_with_width(mb[0], mb[1], mb[2], mb[3], [0,0,0,0], 100)
    return repaired

def generate_and_render():
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
        
    mc = config['model_config']
    save_dir = os.path.join(SCRIPT_DIR, config['train_config']['save_dir'])
    
    VOCAB_SIZE = mc['num_tokens'] + 3
    SOS_IDX = mc['num_tokens'] + 1
    EOS_IDX = mc['num_tokens'] + 2
    
    print("🧠 正在唤醒 FontGPT 大脑...")
    font_gpt = FontGPT(vocab_size=VOCAB_SIZE).to(device)
    font_gpt.load_state_dict(torch.load(os.path.join(save_dir, "font_gpt_best.pth"), map_location=device))
    font_gpt.eval()
    
    print("✍️ 正在装载 VQ-VAE 机械手...")
    vq_vae = BezierVQVAE(
        input_dim=mc['input_dim'], hidden_dim=mc['hidden_dim'],
        latent_dim=mc['latent_dim'], num_tokens=mc['num_tokens']
    ).to(device)
    vq_vae.load_state_dict(torch.load(os.path.join(save_dir, "stroke_vqvae_best.pth"), map_location=device))
    vq_vae.eval()

    pkl_path = os.path.join(SCRIPT_DIR, "tokenized_fonts_dataset.pkl")
    with open(pkl_path, 'rb') as f: dataset = pickle.load(f)
        
    sample_key = list(dataset.keys())[0] 
    real_strokes = dataset[sample_key]
    first_stroke = real_strokes[0]
    
    # ==========================================
    # 🌟 修复：起手式扩充至 8 维空间序列
    # ==========================================
    gen_toks = [SOS_IDX, first_stroke['token_id']]
    
    nx = (first_stroke['start_x'] / 200.0) - 1.0
    ny = (first_stroke['start_y'] / 200.0) - 1.0
    nl = first_stroke['length'] / 400.0
    na = first_stroke['angle'] / math.pi
    nw0 = first_stroke['width_0'] / 30.0
    nw1 = first_stroke['width_1'] / 30.0
    nw2 = first_stroke['width_2'] / 30.0
    nw3 = first_stroke['width_3'] / 30.0
    
    gen_spas = [[0.0]*8, [nx, ny, nl, na, nw0, nw1, nw2, nw3]]

    temperature = 0.6     
    top_k = 5             
    repetition_penalty = 1.3 

    print(f"\n🪄 AI 续写创作中 (Prompt: 字符 '{sample_key}' 第一笔)...")
    with torch.no_grad():
        for step in range(2, 10):
            inp_tok = torch.tensor([gen_toks]).to(device)
            inp_spa = torch.tensor([gen_spas], dtype=torch.float32).to(device)
            
            tok_logits, spa_preds = font_gpt(inp_tok, inp_spa)
            next_tok_logits = tok_logits[0, -1, :mc['num_tokens']+3]
            
            for past_tok in set(gen_toks):
                if past_tok < mc['num_tokens']: 
                    next_tok_logits[past_tok] /= repetition_penalty
            
            next_tok_logits = next_tok_logits / temperature
            filtered_logits = top_k_filtering(next_tok_logits, top_k=top_k)
            
            probs = F.softmax(filtered_logits, dim=-1)
            next_tok_id = torch.multinomial(probs, num_samples=1).item()
            
            # 拿到预测的 8 维坐标与宽度
            next_spa = spa_preds[0, -1, :].cpu().numpy()
            
            if next_tok_id == EOS_IDX:
                print(f"  👉 步骤 {step+1}: 构思完毕 <EOS>！")
                break
                
            w_disp = next_spa[4:] * 30.0
            print(f"  👉 步骤 {step+1}: Token [{next_tok_id:04d}] | 坐标 [{next_spa[0]:.2f}, {next_spa[1]:.2f}] | 笔锋: {w_disp.round(1)}")
            
            gen_toks.append(next_tok_id)
            gen_spas.append(next_spa.tolist())

    print("\n🎨 正在将构思转换为动态笔锋贝塞尔...")
    raw_strokes = []
    with torch.no_grad():
        for i in range(1, len(gen_toks)):
            token_id = gen_toks[i]
            spa = gen_spas[i]
            
            # 1. 字典只恢复骨架形态 (8维)
            test_encoding = torch.zeros(1, mc['num_tokens'], device=device)
            test_encoding[0, token_id] = 1.0
            z_q_test = torch.matmul(test_encoding, vq_vae.vq_layer.embedding.weight)
            recon_output = vq_vae.decoder(z_q_test)[0].cpu().numpy()
            norm_mb = recon_output.reshape(4, 2)
            
            # 2. 从 GPT 输出中反归一化物理尺寸和 4 维笔锋
            start_x = (spa[0] + 1.0) * 200.0
            start_y = (spa[1] + 1.0) * 200.0
            length = max(1.0, spa[2] * 400.0)
            angle = spa[3] * math.pi
            widths = [max(1.0, spa[w_idx] * 30.0) for w_idx in [4, 5, 6, 7]]
            
            # 3. 物理挂载
            scaled_mb = norm_mb * length
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a,  cos_a]])
            rotated_mb = np.dot(scaled_mb, rot_matrix)
            final_mb = rotated_mb + np.array([start_x, start_y])
            
            raw_strokes.append({"mother_bezier": final_mb.tolist(), "widths": widths})

    print("🧲 执行强力拓扑吸附 (整合零散笔画)...")
    final_strokes = apply_topological_snapping(raw_strokes, snap_threshold=20.0)

    # ==========================================
    # 🖼️ 高清多边形渲染 (见证奇迹)
    # ==========================================
    plt.figure(figsize=(6, 6))
    plt.title(f"Dynamic Width Stroke Render (Prompt: {sample_key})", fontsize=12)
    plt.xlim(0, 400); plt.ylim(400, 0)
    plt.gca().set_facecolor('#F5F5F5')
    
    colors = ['#2C3E50', '#E74C3C', '#2980B9', '#27AE60', '#F39C12', '#8E44AD']
    
    for idx, stroke in enumerate(final_strokes):
        mb = np.array(stroke['mother_bezier'])
        widths = stroke['widths']
        c = colors[0] if idx == 0 else colors[(idx % (len(colors)-1)) + 1] 
        alpha_val = 1.0 if idx == 0 else 0.85
        
        # 使用我们的动态笔锋几何引擎生成多边形
        curve_pts, curve_w = evaluate_cubic_bezier_with_width(mb[0], mb[1], mb[2], mb[3], widths, 60)
        poly = generate_stroke_polygon(curve_pts, curve_w)
        
        # 填充多边形 (这就是真实的 TTF 效果！)
        plt.fill(poly[:, 0], poly[:, 1], color=c, alpha=alpha_val)
        
        # 画出内部细细的骨架导线
        plt.plot(curve_pts[:, 0], curve_pts[:, 1], color='white' if idx==0 else 'black', linewidth=1.0, alpha=0.5, linestyle='--')

    plt.axis('equal'); plt.axis('off')
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    generate_and_render()