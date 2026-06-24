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
T_BINS = 32     # 将连续的 t 值 (0.0~1.0) 切分为 32 个离散分类桶

# 派生策略参数
MAX_ORDER_SAMPLES = 5   # 最多派生 5 种笔画顺序
ROTATION_MODE = 3       # 1(不转), 2(180度), 3(90度递进)
MIRROR_MODE = True      # 开启翻转

# ==========================================
# 🧮 转换函数：量化与离散化
# ==========================================
def get_cell_and_offset(val, max_val=CANVAS_SIZE, bins=GRID_BINS):
    norm_val = np.clip(val / max_val, 0.0, 0.9999)
    grid_float = norm_val * bins
    cell = int(grid_float)
    offset = grid_float - (cell + 0.5) 
    return cell, round(offset, 4)

def quantize_width(mean_w, stroke_length, num_bins=4):
    if stroke_length < 1e-5: return 0
    relative_w = mean_w / stroke_length
    norm_w = np.clip(relative_w / 0.5, 0.0, 0.9999)
    return int(norm_w * num_bins)

def quantize_t(t_val, bins=T_BINS):
    """🌟 核心新增：将连续的相交比例 t 离散化为 Token ID"""
    t_val = np.clip(float(t_val), 0.0, 0.9999)
    return int(t_val * bins)

def calculate_stroke_length(bezier_pts):
    pts = np.array(bezier_pts)
    return np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))

# ==========================================
# 🚀 主程序：组装大一统图序列 (Vector Language)
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
    orig_sample_count = 0

    print(f"🧩 正在执行流水线组装 (纯 Token 序列化模式)...")
    for fp in tqdm(topo_files):
        # 🌟 新增：提取源文件名称作为标识 (例如 "font1_topo.json")
        source_filename = os.path.basename(fp)

        with open(fp, 'r', encoding='utf-8') as f:
            char_data_map = json.load(f)
            
        for hex_key, char_data in char_data_map.items():
            if hex_key not in shape_dict: continue
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
            
            # --- 🌟 将每一个派生样本组装为纯 1D Token 序列 ---
            for rule_name, stroke_seq, derived_events in derived_samples:
                flat_token_sequence = []
                drawn_bids = []      # 记录在当前序列中已经画出的笔画顺序
                emitted_events = set() # 记录已经转换为 Junction Token 的事件
                
                for ds in stroke_seq:
                    bid = ds["bezier_id"]
                    pts = ds["mother_bezier"]
                    length = calculate_stroke_length(pts)
                    
                    p0_cx, p0_ox = get_cell_and_offset(pts[0][0])
                    p0_cy, p0_oy = get_cell_and_offset(pts[0][1])
                    p3_cx, p3_ox = get_cell_and_offset(pts[3][0])
                    p3_cy, p3_oy = get_cell_and_offset(pts[3][1])
                    w_token = quantize_width(ds["width_mean"], length)
                    
                    # 1️⃣ 压入 STROKE Token 实体
                    flat_token_sequence.append({
                        "token_type": "STROKE",
                        "bezier_id": bid,
                        "shape_code": ds["shape_token"],
                        "variant_id": ds["variant_id"], 
                        "p0_cell": [p0_cx, p0_cy],
                        "p0_offset": [p0_ox, p0_oy],
                        "p3_cell": [p3_cx, p3_cy],
                        "p3_offset": [p3_ox, p3_oy],
                        "width_token": w_token
                    })
                    drawn_bids.append(bid)
                    current_step_idx = len(drawn_bids) - 1
                    
                    # 2️⃣ 实时检索：是否触发了新的 JUNCTION Token？
                    for ev_idx, ev in enumerate(derived_events):
                        if ev_idx in emitted_events: continue
                            
                        ev_type = ev.get("type")
                        
                        # 统一提取参与者 A 和 B，以及各自的 t 值
                        b_a, b_b = -1, -1
                        ta_val, tb_val = 0.0, 0.0
                        
                        if ev_type in ["E2E", "X"]:
                            b_a, b_b = ev.get("stroke_a"), ev.get("stroke_b")
                            ta_val, tb_val = ev.get("t_a", 0.0), ev.get("t_b", 0.0)
                        elif ev_type == "T":
                            # 将 Guest 视为 A，Host 视为 B
                            b_a, b_b = ev.get("guest"), ev.get("host")
                            ta_val, tb_val = ev.get("guest_t", 0.0), ev.get("host_t", 0.0)
                            
                        # 如果构成该交互事件的两笔都已经存在于画布序列中，立刻触发发牌！
                        if b_a in drawn_bids and b_b in drawn_bids:
                            # 相对距离引用：0 代表当前最新的一笔，1 代表上一笔，以此类推
                            dist_a = current_step_idx - drawn_bids.index(b_a)
                            dist_b = current_step_idx - drawn_bids.index(b_b)
                            
                            flat_token_sequence.append({
                                "token_type": "JUNCTION",
                                "j_type": ev_type,          # "E2E", "X", "T"
                                "ref_a_dist": dist_a,       # 指向局部上下文中的实体
                                "ref_b_dist": dist_b,
                                "ta_bin": quantize_t(ta_val), # 纯离散化表征
                                "tb_bin": quantize_t(tb_val)
                            })
                            emitted_events.add(ev_idx)
                            
                if not flat_token_sequence: continue
                            
                final_dataset.append({
                    "source_file": source_filename,  # 🌟 新增：记录所属文件标识
                    "hex_key": hex_key,
                    "char": char_data.get("glyph_info", {}).get("char", ""),
                    "derivation": rule_name,
                    "sequence_length": len(flat_token_sequence),
                    "sequence": flat_token_sequence
                    # 🌟 拓扑矩阵和绝对坐标被彻底移除，Transformer 现已升维为纯粹的序列生成器
                })

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)
        
    avg_derivation = len(final_dataset) / orig_sample_count if orig_sample_count > 0 else 0
        
    print(f"\n🎉 成功落盘！共组装 {len(final_dataset)} 个「矢量语言」混合序列。")
    print(f"📊 派生前有效字符样本数: {orig_sample_count} 个")
    print(f"📈 平均派生膨胀率: 每个原始字符派生出 {avg_derivation:.1f} 个序列样本")
    print(f"💾 数据已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()