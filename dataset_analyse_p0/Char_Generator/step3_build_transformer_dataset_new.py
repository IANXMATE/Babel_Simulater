import os
import json
import glob
import numpy as np
from tqdm import tqdm

# 导入派生引擎
from stroke_derivation import (
    generate_derived_sequences, 
    normalize_and_sample_function,
    CANVAS_SIZE
)

# ==========================================
# ⚙️ 全局超参数配置
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOPO_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../AI_VECTOR_ROUTER_With_topo/annotations_topo"))
CLUSTER_FILE = os.path.join(SCRIPT_DIR, "clustered_results.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "fontgpt_dataset.json")

# 坐标系与网格配置
GRID_BINS = 32  # 将画布切分为 32x32 的 Cell

# 派生策略参数
MAX_ORDER_SAMPLES = 5   # 最多派生 5 种笔画顺序
ROTATION_MODE = 3       # 1(不转), 2(180度), 3(90度递进)
MIRROR_MODE = True      # 开启翻转

# ==========================================
# 🧮 转换函数：位置与宽度
# ==========================================
def get_cell_and_offset(val, max_val=CANVAS_SIZE, bins=GRID_BINS):
    norm_val = np.clip(val / max_val, 0.0, 0.9999)
    grid_float = norm_val * bins
    cell = int(grid_float)
    offset = grid_float - (cell + 0.5) 
    return cell, round(offset, 4)

def quantize_width(mean_w, stroke_length, num_bins=64):
    if stroke_length < 1e-5: return 0
    relative_w = mean_w / stroke_length
    norm_w = np.clip(relative_w / 0.5, 0.0, 0.9999)
    return int(norm_w * num_bins)

def calculate_stroke_length(bezier_pts):
    pts = np.array(bezier_pts)
    return np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))

# ==========================================
# 🚀 主程序：组装混合表征并落盘
# ==========================================
def main():
    if not os.path.exists(CLUSTER_FILE):
        print("❌ 未找到 Shape Token 词表，请先运行分桶脚本！")
        return

    # 1. 加载 Shape Codebook 与基准原型 (用于算 0123)
    print("📖 正在加载 Shape VQ-Codebook 并计算绝对基准靶心...")
    with open(CLUSTER_FILE, 'r') as f: cluster_data = json.load(f)
        
    shape_dict = {}
    cluster_refs = {}
    for item in cluster_data:
        hex_key, bid, cid = item["hex_key"], item["bezier_id"], item["cluster_id"]
        if cid == -1: continue 
        if hex_key not in shape_dict: shape_dict[hex_key] = {}
        shape_dict[hex_key][bid] = cid
        
        # 记录每个桶的第一个样本作为绝对参照 Y 数组
        if cid not in cluster_refs:
            Y = normalize_and_sample_function(item["mother_bezier"])
            if Y is not None: cluster_refs[cid] = Y

    # 2. 读取原始拓扑文件并进行派生展开
    topo_files = glob.glob(os.path.join(TOPO_DATA_DIR, "*_topo.json"))
    final_dataset = []
    
    # 🌟 新增：原始有效样本计数器
    orig_sample_count = 0

    print(f"🧩 正在执行流水线组装 (允许最大顺序派生: {MAX_ORDER_SAMPLES}, 包含空间翻转)...")
    for fp in tqdm(topo_files):
        with open(fp, 'r', encoding='utf-8') as f:
            char_data_map = json.load(f)
            
        for hex_key, char_data in char_data_map.items():
            if hex_key not in shape_dict: continue
            
            # 🌟 新增：确认为有效原始样本后，计数器 +1
            orig_sample_count += 1

            orig_strokes = [s for s in char_data.get("strokes", []) if s["bezier_id"] in shape_dict[hex_key]]
            topo_events = char_data.get("topology_events", [])
            
            # --- 调用独立引擎进行样本派生 ---
            derived_samples = generate_derived_sequences(
                strokes=orig_strokes, 
                topo_events=topo_events, 
                shape_dict=shape_dict, 
                hex_key=hex_key, 
                cluster_refs=cluster_refs,
                max_order_samples=MAX_ORDER_SAMPLES,
                rot_mode=ROTATION_MODE,
                mirror_mode=MIRROR_MODE
            )
            
            # --- 将每一个派生样本组装为 FontGPT 的 Token 序列 ---
            for rule_name, stroke_seq in derived_samples:
                sequence = []
                valid_bids = []
                
                for ds in stroke_seq:
                    bid = ds["bezier_id"]
                    pts = ds["mother_bezier"]
                    length = calculate_stroke_length(pts)
                    
                    p0_cx, p0_ox = get_cell_and_offset(pts[0][0])
                    p0_cy, p0_oy = get_cell_and_offset(pts[0][1])
                    p3_cx, p3_ox = get_cell_and_offset(pts[3][0])
                    p3_cy, p3_oy = get_cell_and_offset(pts[3][1])
                    
                    w_token = quantize_width(ds["width_mean"], length)
                    
                    sequence.append({
                        "bezier_id": bid,
                        "shape_code": ds["shape_token"],
                        "variant_id": ds["variant_id"], # 0,1,2,3 的形态修饰符
                        "p0_cell": [p0_cx, p0_cy],
                        "p0_offset": [p0_ox, p0_oy],
                        "p3_cell": [p3_cx, p3_cy],
                        "p3_offset": [p3_ox, p3_oy],
                        "width_token": w_token
                    })
                    valid_bids.append(bid)
                    
                if not sequence: continue
                
                # --- 构建 Graphormer Topology Bias Matrix ---
                # 注意：矩阵的索引必须按照当前派生的笔画顺序进行对齐！
                N = len(sequence)
                topo_matrix = np.zeros((N, N), dtype=int)
                bid_to_idx = {bid: idx for idx, bid in enumerate(valid_bids)}
                
                for ev in topo_events:
                    ev_type = ev.get("type")
                    if ev_type in ["E2E", "X"]:
                        a, b = ev.get("stroke_a"), ev.get("stroke_b")
                        if a in bid_to_idx and b in bid_to_idx:
                            i_a, i_b = bid_to_idx[a], bid_to_idx[b]
                            topo_matrix[i_a, i_b] = topo_matrix[i_b, i_a] = 1
                    elif ev_type == "T":
                        g, h = ev.get("guest"), ev.get("host")
                        if g in bid_to_idx and h in bid_to_idx:
                            i_g, i_h = bid_to_idx[g], bid_to_idx[h]
                            topo_matrix[i_g, i_h] = 2 # 客看主
                            topo_matrix[i_h, i_g] = 3 # 主看客
                            
                final_dataset.append({
                    "hex_key": hex_key,
                    "char": char_data.get("glyph_info", {}).get("char", ""),
                    "derivation": rule_name,
                    "sequence_length": N,
                    "sequence": sequence,
                    "topology_bias_matrix": topo_matrix.tolist()
                })

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)
        
    # 🌟 新增：计算平均派生率
    avg_derivation = len(final_dataset) / orig_sample_count if orig_sample_count > 0 else 0
        
    print(f"\n🎉 成功落盘！共组装 {len(final_dataset)} 个包含变体与拓扑的高质量 Token 序列。")
    print(f"📊 派生前有效字符样本数: {orig_sample_count} 个")
    print(f"📈 平均派生膨胀率: 每个原始字符派生出 {avg_derivation:.1f} 个序列样本")
    print(f"💾 数据已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()