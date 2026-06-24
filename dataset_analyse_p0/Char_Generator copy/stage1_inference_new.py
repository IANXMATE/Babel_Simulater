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
            cid = int(item["cluster_id"]) 
            if cid != -1 and cid not in shape_codebook:
                shape_codebook[cid] = item["mother_bezier"]
    print(f"📚 成功加载 VQ-VAE 形状密码本！共包含 {len(shape_codebook)} 种标准曲线原型。")
except Exception as e:
    print(f"⚠️ 致命警告：无法加载密码本 {CLUSTER_FILE}")
    print(f"⚠️ 报错详情: {e}")
    print("⚠️ 渲染器将强制降级为【画直线模式】。")

# ==========================================
# 📐 贝塞尔曲线单点求值工具
# ==========================================
def get_bezier_point_single(pts, t):
    """根据 t 值 (0.0~1.0)，在 4 个控制点构成的贝塞尔曲线上求出精确坐标"""
    mt = 1.0 - t
    return (mt**3)*pts[0] + 3*(mt**2)*t*pts[1] + 3*mt*(t**2)*pts[2] + (t**3)*pts[3]

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
                        "variant_id": int(var_id),
                        "width_token": int(w_token),
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
    
    current_jtype = -1

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
                # elif expected_type == "TBIN":
                #     if current_jtype == 0: # 0 代表 E2E
                #         mask[tokenizer.OFFSET_TBIN + 0] = 0
                #         mask[tokenizer.OFFSET_TBIN + 32] = 0
                #     else:
                #         mask[tokenizer.OFFSET_TBIN : tokenizer.OFFSET_TBIN + 33] = 0
                elif expected_type == "TBIN":
                    # 🌟 绝妙技巧：通过判断 expected_queue 里还剩几个 TBIN，
                    # 来知道现在是在预测第一根线 (ta) 还是第二根线 (tb)
                    is_ta = (len(expected_queue) == 2) 
                    
                    if current_jtype == 0: # 🎯 E2E: ta 和 tb 必须都在两端
                        mask[tokenizer.OFFSET_TBIN + 0] = 0
                        mask[tokenizer.OFFSET_TBIN + 32] = 0
                        
                    elif current_jtype == 1: # 🎯 X: 必须在路中相交，绝对不能是两端
                        mask[tokenizer.OFFSET_TBIN + 1 : tokenizer.OFFSET_TBIN + 32] = 0
                        
                    elif current_jtype == 2: # 🎯 T: Guest (ta) 在端点，Host (tb) 在路中
                        if is_ta:
                            mask[tokenizer.OFFSET_TBIN + 0] = 0
                            mask[tokenizer.OFFSET_TBIN + 32] = 0
                        else:
                            mask[tokenizer.OFFSET_TBIN + 1 : tokenizer.OFFSET_TBIN + 32] = 0
            
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
                if completed_type == "JTYPE":
                    current_jtype = next_token - tokenizer.OFFSET_JTYPE
                    
            generated_tokens.append(next_token)
            if next_token == tokenizer.EOS:
                print("🏁 接收到 [EOS] 结束符，生成完毕！")
                break
    return generated_tokens

# ==========================================
# 🎨 极简渲染引擎：完美还原贝塞尔曲线与拓扑连接
# ==========================================
def render_sequence(sequence):
    plt.figure(figsize=(8, 8))
    plt.title("FontGPT Vector Language - Zero-Shot Inference")
    plt.xlim(0, CANVAS_SIZE)
    plt.ylim(CANVAS_SIZE, 0) # Y 轴向下
    plt.grid(True, linestyle='--', alpha=0.5)
    
    stroke_idx = 0
    drawn_strokes = [] # 🌟 现在保存的是完整的 4 个贝塞尔控制点矩阵，而不仅仅是端点
    
    for item in sequence:
        if item["type"] == "STROKE":
            p0 = np.array(item["p0"])
            p3 = np.array(item["p3"])
            
            shape_code = int(item["shape_code"])
            var_id = int(item.get("variant_id", 0))
            w_token = int(item.get("width_token", 0))
            
            dynamic_lw = 2 + w_token * 2 
            
            mapped_pts = None # 用来保存还原出的曲线控制点
            
            if shape_code in shape_codebook:
                canon_pts = np.array(shape_codebook[shape_code]).copy()
                
                # 纯几何反射
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
                    
                    plt.plot(curve[:, 0], curve[:, 1], color='blue', linewidth=dynamic_lw, zorder=1)
                
            # 如果无法还原贝塞尔（长度为0，或密码本无该shape），回退为绝对直线控制点
            if mapped_pts is None:
                mapped_pts = np.array([p0, p0 + (p3 - p0) * 0.333, p0 + (p3 - p0) * 0.666, p3])
                plt.plot([p0[0], p3[0]], [p0[1], p3[1]], color='blue', linewidth=dynamic_lw, zorder=1)
                
            drawn_strokes.append(mapped_pts) # 将完整的 4 个控制点存入历史，供交点查询
                
            plt.scatter(p0[0], p0[1], color='green', s=30, zorder=2)
            plt.scatter(p3[0], p3[1], color='red', s=30, zorder=2)
            plt.text(p0[0]+5, p0[1]-5, f"S{stroke_idx}\n(ID:{shape_code})", fontsize=10, color='blue')
            stroke_idx += 1
            
        elif item["type"].startswith("JUNCTION"):
            j_type = item["type"].split("_")[1]
            dist_a = item["dist_a"]
            dist_b = item["dist_b"]
            ta = item["ta"]
            tb = item["tb"]
            
            idx_a = len(drawn_strokes) - 1 - dist_a
            idx_b = len(drawn_strokes) - 1 - dist_b
            
            if 0 <= idx_a < len(drawn_strokes) and 0 <= idx_b < len(drawn_strokes):
                pts_a = drawn_strokes[idx_a]
                pts_b = drawn_strokes[idx_b]
                
                # 🌟 核心：利用大模型预测出来的 t 值，计算在曲线上到底落在哪个物理坐标点
                pt_a = get_bezier_point_single(pts_a, ta)
                pt_b = get_bezier_point_single(pts_b, tb)
                
                # 🌟 根据不同连接类型赋予专属颜色
                if j_type == "E2E":
                    j_color = "gold"   # 黄色虚线
                elif j_type == "T":
                    j_color = "green"  # 绿色虚线
                elif j_type == "X":
                    j_color = "purple" # 紫色虚线
                else:
                    j_color = "gray"
                
                # 画出精准连接施力点的虚线
                plt.plot([pt_a[0], pt_b[0]], [pt_a[1], pt_b[1]], 
                         color=j_color, linestyle='--', linewidth=2, alpha=0.9, zorder=3)
                
                # 在施力连线的中点位置显示文字标注
                mid_x = (pt_a[0] + pt_b[0]) / 2
                mid_y = (pt_a[1] + pt_b[1]) / 2
                
                plt.text(mid_x, mid_y, f"{j_type}\n({ta:.2f}, {tb:.2f})", 
                         color=j_color if j_color != "gold" else "darkgoldenrod", # 避免纯黄字看不清
                         fontsize=9, ha='center', 
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), zorder=4)

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