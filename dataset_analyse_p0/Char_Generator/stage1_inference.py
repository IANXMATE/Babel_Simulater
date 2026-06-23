import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from stage1_fontgpt import FontGPT, FontGPTConfig

# 配置路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(SCRIPT_DIR, "fontgpt_dataset.json")
CODEBOOK_FILE = os.path.join(SCRIPT_DIR, "clustered_results.json")
MODEL_WEIGHTS = os.path.join(SCRIPT_DIR, "fontgpt_stage1_latest.pth")

CANVAS_SIZE = 1000.0
GRID_BINS = 32

# ==========================================
# 📐 1. 可微几何前置引擎 (简易版 Stage 2)
# ==========================================
def get_bezier_curve(pts, num_points=50):
    """根据 4 个控制点生成用于绘制的平滑曲线点集"""
    t = np.linspace(0, 1, num_points)[:, None]
    mt = 1 - t
    curve = (mt**3)*pts[0] + 3*(mt**2)*t*pts[1] + 3*mt*(t**2)*pts[2] + (t**3)*pts[3]
    return curve

def map_bezier_to_endpoints(ref_bezier, target_p0, target_p3):
    """
    ✨ 核心几何魔法：仿射变换映射
    将 Codebook 里的基准形状，通过旋转、缩放、平移，完美镶嵌到模型预测的 P0 和 P3 之间！
    """
    pts = np.array(ref_bezier)
    ref_vec = pts[3] - pts[0]
    ref_len = np.linalg.norm(ref_vec)
    
    tgt_vec = np.array(target_p3) - np.array(target_p0)
    tgt_len = np.linalg.norm(tgt_vec)
    
    if ref_len < 1e-5 or tgt_len < 1e-5: 
        return None
        
    # 1. 移动到原点并缩放
    pts_t = (pts - pts[0]) * (tgt_len / ref_len)
    ref_vec_scaled = pts_t[3]
    
    # 2. 计算旋转角差值
    theta_ref = np.arctan2(ref_vec_scaled[1], ref_vec_scaled[0])
    theta_tgt = np.arctan2(tgt_vec[1], tgt_vec[0])
    theta_diff = theta_tgt - theta_ref
    
    cos_t, sin_t = np.cos(theta_diff), np.sin(theta_diff)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    
    # 3. 旋转并平移到预测起点
    pts_final = (pts_t @ R.T) + target_p0
    return pts_final

def decode_coordinate(cell, offset):
    """反量化：将 Cell 和 Offset 还原为真实物理坐标"""
    return (cell + 0.5 + offset) * (CANVAS_SIZE / GRID_BINS)

# ==========================================
# 🚀 2. 加载与自回归生成 (Autoregressive Inference)
# ==========================================
def load_codebook():
    print("📖 加载 Shape Codebook...")
    with open(CODEBOOK_FILE, 'r') as f:
        data = json.load(f)
    codebook = {}
    for item in data:
        cid = item["cluster_id"]
        if cid != -1 and cid not in codebook:
            codebook[cid] = item["mother_bezier"]
    return codebook

def generate_strokes(model, device, prompt_stroke, topo_matrix, target_length):
    """
    自回归生成核心逻辑：给定第一笔和拓扑蓝图，一笔一笔地往下猜！
    """
    model.eval()
    
    # 初始化上下文序列 (Context)
    seq = {
        "shape": [prompt_stroke["shape_code"]],
        "width": [prompt_stroke["width_token"]],
        "p0_cx": [prompt_stroke["p0_cell"][0]], "p0_cy": [prompt_stroke["p0_cell"][1]],
        "p3_cx": [prompt_stroke["p3_cell"][0]], "p3_cy": [prompt_stroke["p3_cell"][1]],
        "p0_off": [prompt_stroke["p0_offset"]], "p3_off": [prompt_stroke["p3_offset"]]
    }
    
    # 将 Topo 矩阵准备好 (Batch=1)
    topo_t = torch.tensor([topo_matrix], dtype=torch.long).to(device)
    
    print(f"🤖 开始自回归生成，目标笔画数: {target_length}...")
    with torch.no_grad():
        for step in range(1, target_length):
            # 将当前序列转换为 Tensor
            inputs = {
                "shape_tokens": torch.tensor([seq["shape"]], dtype=torch.long).to(device),
                "width_tokens": torch.tensor([seq["width"]], dtype=torch.long).to(device),
                "p0_cells_x": torch.tensor([seq["p0_cx"]], dtype=torch.long).to(device),
                "p0_cells_y": torch.tensor([seq["p0_cy"]], dtype=torch.long).to(device),
                "p3_cells_x": torch.tensor([seq["p3_cx"]], dtype=torch.long).to(device),
                "p3_cells_y": torch.tensor([seq["p3_cy"]], dtype=torch.long).to(device),
                "p0_offsets": torch.tensor([seq["p0_off"]], dtype=torch.float32).to(device),
                "p3_offsets": torch.tensor([seq["p3_off"]], dtype=torch.float32).to(device),
            }
            
            # 当前长度的因果掩码
            cur_len = len(seq["shape"])
            mask = torch.tril(torch.ones(cur_len, cur_len)).unsqueeze(0).unsqueeze(0).to(device)
            
            # 🌟 推理同步修复：强制读取下一笔的拓扑蓝图！
            # 行切片为 1 : cur_len+1，列切片为 : cur_len
            cur_topo = topo_t[:, 1:cur_len+1, :cur_len]
            
            # Forward 推理
            outputs = model(**inputs, topo_matrix=cur_topo, mask=mask)
            
            # 提取最后一步的预测结果 (Greedy Search: Argmax)
            next_shape = torch.argmax(outputs["logits_shape"][0, -1, :]).item()
            next_width = torch.argmax(outputs["logits_width"][0, -1, :]).item()
            
            next_p0_cx = torch.argmax(outputs["logits_p0_cx"][0, -1, :]).item()
            next_p0_cy = torch.argmax(outputs["logits_p0_cy"][0, -1, :]).item()
            next_p3_cx = torch.argmax(outputs["logits_p3_cx"][0, -1, :]).item()
            next_p3_cy = torch.argmax(outputs["logits_p3_cy"][0, -1, :]).item()
            
            next_p0_off = outputs["pred_p0_offset"][0, -1, :].cpu().tolist()
            next_p3_off = outputs["pred_p3_offset"][0, -1, :].cpu().tolist()
            
            # 追加到序列中
            seq["shape"].append(next_shape)
            seq["width"].append(next_width)
            seq["p0_cx"].append(next_p0_cx); seq["p0_cy"].append(next_p0_cy)
            seq["p3_cx"].append(next_p3_cx); seq["p3_cy"].append(next_p3_cy)
            seq["p0_off"].append(next_p0_off); seq["p3_off"].append(next_p3_off)
            
    return seq

# ==========================================
# 🎨 3. 主程序：绘图对决 (Ground Truth vs FontGPT)
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载配置与权重
    checkpoint = torch.load(MODEL_WEIGHTS, map_location=device)
    config = FontGPTConfig() # 必须保证配置和训练时一致
    model = FontGPT(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ FontGPT 大脑加载完毕！")
    
    codebook = load_codebook()
    
    # 2. 从数据集中挑一个顺眼的测试用例
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    # 随便挑一个长一点的字形序列 (比如笔画数 >= 4 的)
    test_sample = next(s for s in dataset if s["sequence_length"] >= 4)
    char_name = test_sample.get("char", "Unknown")
    gt_seq = test_sample["sequence"]
    topo_matrix = test_sample["topology_bias_matrix"]
    seq_len = test_sample["sequence_length"]
    
    print(f"🎯 测试目标字符: '{char_name}' (笔画数: {seq_len})")
    
    # 3. 让模型根据第一笔和拓扑蓝图生成剩余笔画
    pred_seq_dict = generate_strokes(model, device, gt_seq[0], topo_matrix, seq_len)
    
    # 4. 可视化渲染
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"FontGPT Zero-Shot Inference | Char: {char_name}", fontsize=16)
    
    # 解析并绘制的内部函数
    def render_sequence(ax, seq_dict_or_list, title, is_gt=False):
        ax.set_title(title)
        ax.set_xlim(0, CANVAS_SIZE); ax.set_ylim(CANVAS_SIZE, 0) # 字体坐标系通常Y轴向下
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        for i in range(seq_len):
            # 兼容两种数据结构 (GT 是 List, Pred 是 Dict)
            if is_gt:
                s = seq_dict_or_list[i]
                shape_id = s["shape_code"]
                p0x = decode_coordinate(s["p0_cell"][0], s["p0_offset"][0])
                p0y = decode_coordinate(s["p0_cell"][1], s["p0_offset"][1])
                p3x = decode_coordinate(s["p3_cell"][0], s["p3_offset"][0])
                p3y = decode_coordinate(s["p3_cell"][1], s["p3_offset"][1])
            else:
                shape_id = seq_dict_or_list["shape"][i]
                p0x = decode_coordinate(seq_dict_or_list["p0_cx"][i], seq_dict_or_list["p0_off"][i][0])
                p0y = decode_coordinate(seq_dict_or_list["p0_cy"][i], seq_dict_or_list["p0_off"][i][1])
                p3x = decode_coordinate(seq_dict_or_list["p3_cx"][i], seq_dict_or_list["p3_off"][i][0])
                p3y = decode_coordinate(seq_dict_or_list["p3_cy"][i], seq_dict_or_list["p3_off"][i][1])
                
            # 从字典提取参考曲线
            ref_bezier = codebook.get(shape_id, None)
            if ref_bezier is None: continue
            
            # 仿射变换映射到 P0, P3
            final_bezier = map_bezier_to_endpoints(ref_bezier, [p0x, p0y], [p3x, p3y])
            if final_bezier is None: continue
            
            # 画线
            curve_pts = get_bezier_curve(final_bezier)
            color = 'blue' if i == 0 else ('green' if is_gt else 'red')
            linewidth = 3 if i == 0 else 2
            ax.plot(curve_pts[:, 0], curve_pts[:, 1], color=color, linewidth=linewidth, alpha=0.8)
            
            # 画端点和文本
            ax.plot([p0x, p3x], [p0y, p3y], 'o', color=color, markersize=4)
            ax.text((p0x+p3x)/2, (p0y+p3y)/2, str(i), color='black', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    # 左图：真实标注 (Ground Truth)
    render_sequence(axs[0], gt_seq, "Ground Truth", is_gt=True)
    
    # 右图：FontGPT 预测
    render_sequence(axs[1], pred_seq_dict, "FontGPT Prediction", is_gt=False)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()