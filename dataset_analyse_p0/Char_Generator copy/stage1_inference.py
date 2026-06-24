import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import json

# ==========================================
# ⚙️ 导入与训练一致的基建参数
# ==========================================
from stage1_train import FontTokenizer, FontVectorLM, VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(SCRIPT_DIR, "fontgpt_vector_lm_latest.pth")
CLUSTER_FILE = os.path.join(SCRIPT_DIR, "clustered_results.json") # 🌟 强制绝对路径

CANVAS_SIZE = 400.0
GRID_BINS = 32
T_BINS = 32
TEMPERATURE = 0.9  # 🌟 强行降温，约束语法

# ==========================================
# 📚 核心补丁：强力加载 VQ-VAE 形状密码本
# ==========================================
shape_codebook = {}
try:
    with open(CLUSTER_FILE, 'r', encoding='utf-8') as f:
        cluster_data = json.load(f)
        for item in cluster_data:
            cid = int(item["cluster_id"]) # 🌟 强制转为 int，防止类型匹配失败
            if cid != -1 and cid not in shape_codebook:
                shape_codebook[cid] = item["mother_bezier"]
    print(f"📚 成功加载 VQ-VAE 形状密码本！共包含 {len(shape_codebook)} 种标准曲线原型。")
except Exception as e:
    print(f"⚠️ 致命警告：无法加载密码本 {CLUSTER_FILE}")
    print(f"⚠️ 报错详情: {e}")
    print("⚠️ 渲染器将强制降级为【画直线模式】。")

# ==========================================
# 🔍 矢量解码器：Tokens -> 几何对象 (自带语法容错)
# ==========================================
# ==========================================
# 🔍 矢量解码器：Tokens -> 几何对象 (自带语法容错)
# ==========================================
def decode_tokens_to_geometry(tokens, tokenizer):
    sequence = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == tokenizer.EOS: break
        if t == tokenizer.CMD_STROKE:
            chunk = tokens[i+1 : i+12]
            if len(chunk) == 11 and not any(c <= tokenizer.CMD_JUNCTION for c in chunk):
                try:
                    shape_code = max(0, chunk[0] - tokenizer.OFFSET_SHAPE)
                    var_id     = np.clip(chunk[1] - tokenizer.OFFSET_VAR, 0, 3)
                    p0_cx      = np.clip(chunk[2] - tokenizer.OFFSET_CELL, 0, 31)
                    p0_cy      = np.clip(chunk[3] - tokenizer.OFFSET_CELL, 0, 31)
                    p0_ox      = np.clip(chunk[4] - tokenizer.OFFSET_OFFSET, 0, 31) / 32.0
                    p0_oy      = np.clip(chunk[5] - tokenizer.OFFSET_OFFSET, 0, 31) / 32.0
                    p3_cx      = np.clip(chunk[6] - tokenizer.OFFSET_CELL, 0, 31)
                    p3_cy      = np.clip(chunk[7] - tokenizer.OFFSET_CELL, 0, 31)
                    p3_ox      = np.clip(chunk[8] - tokenizer.OFFSET_OFFSET, 0, 31) / 32.0
                    p3_oy      = np.clip(chunk[9] - tokenizer.OFFSET_OFFSET, 0, 31) / 32.0
                    w_token    = np.clip(chunk[10]- tokenizer.OFFSET_WIDTH, 0, 3)
                    
                    sequence.append({
                        "type": "STROKE",
                        "shape_code": int(shape_code),
                        "variant_id": int(var_id),    # 🌟 修复：保存形态 ID
                        "width_token": int(w_token),  # 🌟 修复：保存线宽 ID
                        "p0": [(p0_cx + 0.5 + p0_ox) * (CANVAS_SIZE / GRID_BINS), 
                               (p0_cy + 0.5 + p0_oy) * (CANVAS_SIZE / GRID_BINS)],
                        "p3": [(p3_cx + 0.5 + p3_ox) * (CANVAS_SIZE / GRID_BINS), 
                               (p3_cy + 0.5 + p3_oy) * (CANVAS_SIZE / GRID_BINS)]
                    })
                    i += 12 
                except Exception: i += 1
            else: i += 1
                
        elif t == tokenizer.CMD_JUNCTION:
            chunk = tokens[i+1 : i+6]
            if len(chunk) == 5 and not any(c <= tokenizer.CMD_JUNCTION for c in chunk):
                try:
                    j_types = ["E2E", "X", "T"]
                    j_idx   = chunk[0] - tokenizer.OFFSET_JTYPE
                    j_type  = j_types[j_idx] if 0 <= j_idx < len(j_types) else "Unknown"
                    dist_a  = max(0, chunk[1] - tokenizer.OFFSET_DIST)
                    dist_b  = max(0, chunk[2] - tokenizer.OFFSET_DIST)
                    ta_val  = np.clip(chunk[3] - tokenizer.OFFSET_TBIN, 0, 32) / float(T_BINS)
                    tb_val  = np.clip(chunk[4] - tokenizer.OFFSET_TBIN, 0, 32) / float(T_BINS)
                    sequence.append({
                        "type": f"JUNCTION_{j_type}",
                        "dist_a": int(dist_a),
                        "dist_b": int(dist_b),
                        "ta": float(ta_val),
                        "tb": float(tb_val)
                    })
                    i += 6
                except Exception: i += 1
            else: i += 1
        else: i += 1 
    return sequence

# ==========================================
# 🚀 核心自回归生成流水线 (带强约束状态机)
# ==========================================
def generate_font_sequence(model, tokenizer, device, max_len=200):
    model.eval()
    generated_tokens = [tokenizer.BOS]
    expected_queue = []
    
    current_jtype = -1 # 🌟 新增：用来记住当前正在处理什么类型的交点

    print(f"\n🔮 开始自回归推理 (启用强约束状态机, Temperature: {TEMPERATURE})...")
    with torch.no_grad():
        for step in range(max_len):
            x = torch.tensor([generated_tokens], dtype=torch.long).to(device)
            logits = model(x)
            next_token_logits = logits[0, -1, :] 
            
            mask = torch.full_like(next_token_logits, float('-inf'))
            if not expected_queue:
                mask[tokenizer.EOS] = 0
                mask[tokenizer.CMD_STROKE] = 0
                mask[tokenizer.CMD_JUNCTION] = 0
            else:
                expected_type = expected_queue[0]
                if expected_type == "SHAPE":   mask[tokenizer.OFFSET_SHAPE : tokenizer.OFFSET_VAR] = 0
                elif expected_type == "VAR":   mask[tokenizer.OFFSET_VAR : tokenizer.OFFSET_CELL] = 0
                elif expected_type == "CELL":  mask[tokenizer.OFFSET_CELL : tokenizer.OFFSET_OFFSET] = 0
                elif expected_type == "OFFSET":mask[tokenizer.OFFSET_OFFSET : tokenizer.OFFSET_WIDTH] = 0
                elif expected_type == "WIDTH": mask[tokenizer.OFFSET_WIDTH : tokenizer.OFFSET_JTYPE] = 0
                elif expected_type == "JTYPE": mask[tokenizer.OFFSET_JTYPE : tokenizer.OFFSET_DIST] = 0
                elif expected_type == "DIST":  mask[tokenizer.OFFSET_DIST : tokenizer.OFFSET_TBIN] = 0
                elif expected_type == "TBIN":
                    # 🌟 核心拦截逻辑：物理法则降临！
                    if current_jtype == 0: # 0 代表 E2E
                        mask[tokenizer.OFFSET_TBIN + 0] = 0
                        mask[tokenizer.OFFSET_TBIN + 32] = 0
                    else:
                        mask[tokenizer.OFFSET_TBIN : tokenizer.OFFSET_TBIN + 33] = 0
            
            next_token_logits = next_token_logits + mask
            next_token_logits = next_token_logits / TEMPERATURE
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if not expected_queue:
                if next_token == tokenizer.CMD_STROKE:
                    expected_queue = ["SHAPE", "VAR", "CELL", "CELL", "OFFSET", "OFFSET", "CELL", "CELL", "OFFSET", "OFFSET", "WIDTH"]
                elif next_token == tokenizer.CMD_JUNCTION:
                    expected_queue = ["JTYPE", "DIST", "DIST", "TBIN", "TBIN"]
            else:
                completed_type = expected_queue.pop(0)
                # 🌟 如果刚刚生成的是 JTYPE，立刻把它记下来，留给后面的 TBIN 用！
                if completed_type == "JTYPE":
                    current_jtype = next_token - tokenizer.OFFSET_JTYPE
                    
            generated_tokens.append(next_token)
            if next_token == tokenizer.EOS:
                print("🏁 接收到 [EOS] 结束符，生成完毕！")
                break
    return generated_tokens

# ==========================================
# 🎨 极简渲染引擎：完美还原贝塞尔曲线
# ==========================================
def render_sequence(sequence):
    plt.figure(figsize=(8, 8))
    plt.title("FontGPT Vector Language - Zero-Shot Inference")
    plt.xlim(0, CANVAS_SIZE)
    plt.ylim(CANVAS_SIZE, 0) # Y 轴向下
    plt.grid(True, linestyle='--', alpha=0.5)
    
    stroke_idx = 0
    drawn_strokes = []
    
    for item in sequence:
        if item["type"] == "STROKE":
            p0 = np.array(item["p0"])
            p3 = np.array(item["p3"])
            drawn_strokes.append((p0, p3))
            
            shape_code = int(item["shape_code"])
            var_id = int(item.get("variant_id", 0))   # 🌟 获取变体 ID
            w_token = int(item.get("width_token", 0)) # 🌟 获取线宽 ID
            
            # 将粗细 ID 映射为实际渲染的 linewidth (例如 0->2, 1->4, 2->6, 3->8)
            dynamic_lw = 2 + w_token * 2 
            
            if shape_code in shape_codebook:
                canon_pts = np.array(shape_codebook[shape_code]).copy()
                
                # ==========================================
                # 🌟 核心魔法回归：纯几何反射 (解决凹凸错乱)
                # ==========================================
                if var_id in [2, 3]:
                    u = canon_pts[3] - canon_pts[0]
                    u_dot_u = np.dot(u, u)
                    if u_dot_u > 1e-5:
                        for i in (1, 2):
                            v = canon_pts[i] - canon_pts[0]
                            proj = (np.dot(v, u) / u_dot_u) * u
                            perp = v - proj
                            canon_pts[i] = canon_pts[0] + proj - perp
                
                if var_id in [1, 3]:
                    canon_pts = canon_pts[::-1]
                # ==========================================
                
                c0, c3 = canon_pts[0], canon_pts[3]
                v_canon = c3 - c0
                v_pred = p3 - p0
                len_canon = np.linalg.norm(v_canon)
                len_pred = np.linalg.norm(v_pred)
                
                if len_canon > 1e-5 and len_pred > 1e-5:
                    scale = len_pred / len_canon
                    angle_canon = np.arctan2(v_canon[1], v_canon[0])
                    angle_pred = np.arctan2(v_pred[1], v_pred[0])
                    theta = angle_pred - angle_canon
                    
                    cos_t, sin_t = np.cos(theta), np.sin(theta)
                    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                    
                    mapped_pts = (canon_pts - c0) @ R.T * scale + p0
                    
                    ts = np.linspace(0, 1, 50)[:, None]
                    mt = 1 - ts
                    curve = (mt**3)*mapped_pts[0] + 3*(mt**2)*ts*mapped_pts[1] + 3*mt*(ts**2)*mapped_pts[2] + (ts**3)*mapped_pts[3]
                    
                    # 🌟 修复：应用大模型预测出来的动态粗细 (dynamic_lw)
                    plt.plot(curve[:, 0], curve[:, 1], color='blue', linewidth=dynamic_lw, zorder=1)
                else:
                    plt.plot([p0[0], p3[0]], [p0[1], p3[1]], color='blue', linewidth=dynamic_lw, zorder=1)
            else:
                print(f"⚠️ 未找到 Shape Code: {shape_code}，被迫降级画直线。")
                plt.plot([p0[0], p3[0]], [p0[1], p3[1]], color='blue', linewidth=dynamic_lw, zorder=1)
                
            plt.scatter(p0[0], p0[1], color='green', s=30, zorder=2)
            plt.scatter(p3[0], p3[1], color='red', s=30, zorder=2)
            plt.text(p0[0]+5, p0[1]-5, f"S{stroke_idx}\n(ID:{shape_code})", fontsize=10, color='blue')
            stroke_idx += 1
            
        elif item["type"].startswith("JUNCTION"):
            j_type = item["type"].split("_")[1]
            dist_a = item["dist_a"]
            dist_b = item["dist_b"]
            
            idx_a = len(drawn_strokes) - 1 - dist_a
            idx_b = len(drawn_strokes) - 1 - dist_b
            
            if 0 <= idx_a < len(drawn_strokes) and 0 <= idx_b < len(drawn_strokes):
                p0_a, p3_a = drawn_strokes[idx_a]
                p0_b, p3_b = drawn_strokes[idx_b]
                
                center_a = [(p0_a[0]+p3_a[0])/2, (p0_a[1]+p3_a[1])/2]
                center_b = [(p0_b[0]+p3_b[0])/2, (p0_b[1]+p3_b[1])/2]
                
                plt.plot([center_a[0], center_b[0]], [center_a[1], center_b[1]], 
                         color='orange', linestyle='--', linewidth=2, alpha=0.7)
                
                mid_x = (center_a[0] + center_b[0]) / 2
                mid_y = (center_a[1] + center_b[1]) / 2
                plt.text(mid_x, mid_y, f"{j_type}\n({item['ta']:.1f}, {item['tb']:.1f})", 
                         color='purple', fontsize=9, ha='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.show()

# ==========================================
# 🎬 主入口
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = FontTokenizer()
    model = FontVectorLM().to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"✅ 成功加载预训练权重: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"❌ 找不到模型权重 {MODEL_PATH}")
        return

    raw_tokens = generate_font_sequence(model, tokenizer, device)
    sequence = decode_tokens_to_geometry(raw_tokens, tokenizer)
    
    print("\n📜 翻译为几何结构:")
    for item in sequence: print(item)
        
    print("\n🎨 正在启动高精度渲染引擎...")
    render_sequence(sequence)

if __name__ == "__main__":
    main()