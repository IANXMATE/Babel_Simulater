import os
import json
import glob
import numpy as np
from tqdm import tqdm

# ==========================================
# ⚙️ 配置参数
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOPO_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../AI_VECTOR_ROUTER_With_topo/annotations_topo"))
CLUSTER_FILE = os.path.join(SCRIPT_DIR, "clustered_results.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "fontgpt_dataset.json")

# 坐标系与网格配置
CANVAS_SIZE = 1000.0  # 假设原始坐标域在 0~1000 左右
GRID_BINS = 32        # 将画布切分为 nxn 的 Cell 网格

# ==========================================
# 🧮 转换函数：位置、宽度与拓扑
# ==========================================
def get_cell_and_offset(val, max_val=CANVAS_SIZE, bins=GRID_BINS):
    """将连续坐标转换为 Cell (离散类别) 和 Offset ([-0.5, 0.5] 连续值)"""
    norm_val = np.clip(val / max_val, 0.0, 0.9999) # 归一化到 [0, 1)
    grid_float = norm_val * bins
    cell = int(grid_float)
    # Offset 定义为相对于 Cell 中心的偏移，范围 [-0.5, 0.5]
    offset = grid_float - (cell + 0.5) 
    return cell, round(offset, 4)

def quantize_width(width_bezier, stroke_length, num_bins=64):
    """
    形宽解耦：将宽度除以母线长度，反映'相对粗细'，并量化为 64 种 Token。
    这里使用简化的平均宽度，你也可以将其扩展为 [w0, w1, w2, w3] 的 4 个 Token。
    """
    mean_w = np.mean(width_bezier)
    if stroke_length < 1e-5: return 0
    relative_w = mean_w / stroke_length
    
    # 假设相对宽度通常在 0 ~ 0.5 之间
    norm_w = np.clip(relative_w / 0.5, 0.0, 0.9999)
    width_token = int(norm_w * num_bins)
    return width_token

def calculate_stroke_length(bezier_pts):
    """估算贝塞尔曲线母线的物理弧长"""
    pts = np.array(bezier_pts)
    # 简单的控制点多边形长度估算
    return np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))

# ==========================================
# 🚀 主程序：组装 FontGPT 训练集
# ==========================================
def main():
    if not os.path.exists(CLUSTER_FILE):
        print(f"❌ 未找到 Shape Token 词表: {CLUSTER_FILE}\n请先运行 step1 分桶脚本！")
        return

    # 1. 加载 Shape Token 字典 (Bezier ID -> Shape Code)
    print("📖 正在加载 Shape VQ-Codebook...")
    with open(CLUSTER_FILE, 'r') as f:
        cluster_data = json.load(f)
        
    shape_dict = {}
    for item in cluster_data:
        hex_key = item["hex_key"]
        bid = item["bezier_id"]
        cid = item["cluster_id"]
        if cid == -1: continue # 抛弃噪声笔画
        if hex_key not in shape_dict: shape_dict[hex_key] = {}
        shape_dict[hex_key][bid] = cid

    # 2. 读取所有五层架构原始拓扑文件
    topo_files = glob.glob(os.path.join(TOPO_DATA_DIR, "*_topo.json"))
    dataset = []
    
    print("🧩 正在组装 FontGPT 混合表征数据集...")
    for fp in tqdm(topo_files):
        with open(fp, 'r', encoding='utf-8') as f:
            char_data_map = json.load(f)
            
        for hex_key, char_data in char_data_map.items():
            if hex_key not in shape_dict: continue
            
            strokes = char_data.get("strokes", [])
            topo_events = char_data.get("topology_events", [])
            
            # --- 构建 Stroke Sequence ---
            sequence = []
            valid_bids = [] # 记录在这个字中真正存活的笔画 ID
            
            for s in strokes:
                bid = s["bezier_id"]
                if bid not in shape_dict[hex_key]: continue # 是噪声，跳过
                
                shape_token = shape_dict[hex_key][bid]
                pts = s["mother_bezier"]
                widths = s["width_bezier"]
                length = calculate_stroke_length(pts)
                
                # 计算位置 Token (P0 和 P3)
                p0_cx, p0_ox = get_cell_and_offset(pts[0][0])
                p0_cy, p0_oy = get_cell_and_offset(pts[0][1])
                p3_cx, p3_ox = get_cell_and_offset(pts[3][0])
                p3_cy, p3_oy = get_cell_and_offset(pts[3][1])
                
                # 计算宽度 Token
                w_token = quantize_width(widths, length)
                
                sequence.append({
                    "bezier_id": bid,
                    "shape_code": shape_token,
                    "p0_cell": [p0_cx, p0_cy],
                    "p0_offset": [p0_ox, p0_oy],
                    "p3_cell": [p3_cx, p3_cy],
                    "p3_offset": [p3_ox, p3_oy],
                    "width_token": w_token
                })
                valid_bids.append(bid)
                
            if len(sequence) == 0: continue
            
            # --- 构建 Attention Topology Bias Matrix (N x N) ---
            N = len(sequence)
            topo_matrix = np.zeros((N, N), dtype=int)
            
            # 建立 bid 到 sequence index 的映射
            bid_to_idx = {bid: idx for idx, bid in enumerate(valid_bids)}
            
            # 定义拓扑关系的离散 Embedding 类别
            # 0: 无连接, 1: 端点对接(E2E), 2: T型客体, 3: T型主体, 4: X型交叉
            for ev in topo_events:
                ev_type = ev.get("type")
                if ev_type == "E2E":
                    a, b = ev.get("stroke_a"), ev.get("stroke_b")
                    if a in bid_to_idx and b in bid_to_idx:
                        idx_a, idx_b = bid_to_idx[a], bid_to_idx[b]
                        topo_matrix[idx_a, idx_b] = 1
                        topo_matrix[idx_b, idx_a] = 1
                        
                elif ev_type == "T":
                    guest, host = ev.get("guest"), ev.get("host")
                    if guest in bid_to_idx and host in bid_to_idx:
                        idx_g, idx_h = bid_to_idx[guest], bid_to_idx[host]
                        topo_matrix[idx_g, idx_h] = 2 # Guest 看 Host 是类型 2
                        topo_matrix[idx_h, idx_g] = 3 # Host 看 Guest 是类型 3
                        
                elif ev_type == "X":
                    a, b = ev.get("stroke_a"), ev.get("stroke_b")
                    if a in bid_to_idx and b in bid_to_idx:
                        idx_a, idx_b = bid_to_idx[a], bid_to_idx[b]
                        topo_matrix[idx_a, idx_b] = 4
                        topo_matrix[idx_b, idx_a] = 4

            dataset.append({
                "hex_key": hex_key,
                "char": char_data.get("glyph_info", {}).get("char", ""),
                "sequence_length": N,
                "sequence": sequence,
                "topology_bias_matrix": topo_matrix.tolist()
            })

    # 3. 落盘存储
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print(f"\n🎉 成功组装 {len(dataset)} 个高质量 FontGPT 训练样本！")
    print(f"💾 数据已保存至: {OUTPUT_FILE}")
    print("✅ 格式完全对齐: [Shape VQ] + [Cell+Offset Reg] + [Width Quant] + [Graphormer TopoBias]")

if __name__ == "__main__":
    main()