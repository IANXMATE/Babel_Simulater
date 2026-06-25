import os
import json
import glob
import math
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
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "fontgpt_dataset_graph.json")

# 坐标系与网格配置
GRID_BINS = 32  

# 图派生策略参数：彻底抛弃顺序派生！
MAX_ORDER_SAMPLES = 1   # Graph 具有排列不变性，顺序派生失去意义，设为 1
ROTATION_MODE = 0       # 🚫 关闭旋转增强：旋转后 edge_ts 不变但坐标改变，造成"同输入多GT"歧义
MIRROR_MODE = False     # 🚫 关闭镜像增强：同上

# ==========================================
# 🧮 转换函数：量化与离散化
# ==========================================
def get_cell_and_offset(val, max_val=CANVAS_SIZE, bins=GRID_BINS):
    norm_val = np.clip(val / max_val, 0.0, 0.9999)
    grid_float = norm_val * bins
    cell = int(grid_float)
    # 🌟 修复 4: 消除 Offset 偏置。去掉 -0.5 的 zero-mean 强加假设，恢复严格的 [0, 1) 区间
    offset = grid_float - cell 
    return cell, round(offset, 4)

def quantize_width(mean_w, stroke_length, num_bins=4):
    if stroke_length < 1e-5: return 0
    relative_w = mean_w / stroke_length
    norm_w = np.clip(relative_w / 0.5, 0.0, 0.9999)
    return int(norm_w * num_bins)

def calculate_stroke_length(bezier_pts):
    pts = np.array(bezier_pts)
    return np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))

# ==========================================
# 🚀 主程序：组装完备图序列 (Canonical Complete Graph)
# ==========================================
def main():
    if not os.path.exists(CLUSTER_FILE):
        print("❌ 未找到 Shape Token 词表！")
        return

    print("📖 正在加载 Shape VQ-Codebook 并计算基准...")
    with open(CLUSTER_FILE, 'r') as f: cluster_data = json.load(f)
        
    shape_dict = {}
    cluster_refs = {}
    
    # 🌟 修复 5: 提前 Normalize Cache，大幅提升后续比对速度
    for item in cluster_data:
        hex_key, bid, cid = item["hex_key"], item["bezier_id"], item["cluster_id"]
        if cid == -1: continue 
        if hex_key not in shape_dict: shape_dict[hex_key] = {}
        shape_dict[hex_key][bid] = cid
        
        if cid not in cluster_refs:
            Y = normalize_and_sample_function(item["mother_bezier"])
            if Y is not None: cluster_refs[cid] = Y

    topo_files = glob.glob(os.path.join(TOPO_DATA_DIR, "*_topo.json"))
    final_dataset = []
    orig_sample_count = 0

    print(f"🧩 正在执行流水线组装 (Complete Graph Canonical 模式)...")
    for fp in tqdm(topo_files):
        source_filename = os.path.basename(fp)

        with open(fp, 'r', encoding='utf-8') as f:
            char_data_map = json.load(f)
            
        for hex_key, char_data in char_data_map.items():
            if hex_key not in shape_dict: continue
            orig_sample_count += 1

            orig_strokes = []
            for s in char_data.get("strokes", []):
                bid = s["bezier_id"]
                if bid in shape_dict[hex_key]:
                    cid = shape_dict[hex_key][bid]
                    stroke_payload = s.copy()
                    
                    v_orig = 0
                    if cid in cluster_refs:
                        Y_raw = normalize_and_sample_function(s["mother_bezier"], N=50)
                        if Y_raw is not None:
                            Y0 = Y_raw.copy()
                            Y1 = -Y_raw[::-1]
                            Y2 = -Y_raw
                            Y3 = Y_raw[::-1]
                            # O(1) 预存比对
                            v_orig = int(np.argmin([np.mean(np.abs(cluster_refs[cid] - v)) for v in [Y0, Y1, Y2, Y3]]))
                    
                    stroke_payload["v_orig"] = v_orig 
                    orig_strokes.append(stroke_payload)

            topo_events = char_data.get("topology_events", [])
            
            # 🌟 修复 6: 先提取无序的空间变换实体集合 (Augmentation 前置，不带顺序污染)
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
            
            for rule_name, stroke_seq, derived_events in derived_samples:
                
                # ==========================================
                # 🌟 修复 1: 消除顺序泄漏 (Canonical Nodes)
                # 强制将空间增强后的笔画按原生 bezier_id 重新排序！
                # ==========================================
                canonical_strokes = sorted(stroke_seq, key=lambda x: x["bezier_id"])
                
                nodes = []
                bid_to_node_idx = {}
                
                for idx, ds in enumerate(canonical_strokes):
                    bid = ds["bezier_id"]
                    bid_to_node_idx[bid] = idx # 锁定映射
                    
                    pts = ds["mother_bezier"]
                    length = calculate_stroke_length(pts)
                    
                    p0_cx, p0_ox = get_cell_and_offset(pts[0][0])
                    p0_cy, p0_oy = get_cell_and_offset(pts[0][1])
                    p3_cx, p3_ox = get_cell_and_offset(pts[3][0])
                    p3_cy, p3_oy = get_cell_and_offset(pts[3][1])
                    w_token = quantize_width(ds["width_mean"], length)
                    
                    nodes.append({
                        "node_id": idx,  # 现在的 Node ID 具有了绝对的全局稳定性
                        "bezier_id": bid,
                        "shape_code": ds["shape_token"],
                        "variant_id": ds["variant_id"], 
                        "p0_cell": [p0_cx, p0_cy],
                        "p0_offset": [p0_ox, p0_oy],
                        "p3_cell": [p3_cx, p3_cy],
                        "p3_offset": [p3_ox, p3_oy],
                        "width_token": w_token
                    })
                    
                # ==========================================
                # 🌟 修复 2 & 3: 完备图 (N x N Complete Graph) 与 破缺边特征
                # ==========================================
                N_nodes = len(nodes)
                if N_nodes == 0: continue
                
                j_map = {"NONE": 0, "E2E": 1, "X": 2, "T": 3}
                
                # (A) 将事件提取为高效的 Hash 表
                # event_dict value: (ev_type, ta_val, tb_val, angle_deg)
                event_dict = {}
                for ev in derived_events:
                    ev_type = ev.get("type")
                    if ev_type in ["E2E", "X"]:
                        b_a, b_b = ev.get("stroke_a"), ev.get("stroke_b")
                        ta_val, tb_val = ev.get("t_a", 0.0), ev.get("t_b", 0.0)
                    elif ev_type == "T":
                        b_a, b_b = ev.get("guest"), ev.get("host")
                        ta_val, tb_val = ev.get("guest_t", 0.0), ev.get("host_t", 0.0)
                    else:
                        continue

                    # 🌟 新增：读取夹角信息
                    # E2E 端对端，切线方向相反，定义夹角 180°
                    # X/T 已有 angle 字段（切线夹角，度数）
                    if ev_type == "E2E":
                        angle_deg = 180.0
                    else:
                        angle_deg = float(ev.get("angle") or 0.0)
                        
                    if b_a in bid_to_node_idx and b_b in bid_to_node_idx:
                        u_idx = bid_to_node_idx[b_a]
                        v_idx = bid_to_node_idx[b_b]
                        event_dict[(u_idx, v_idx)] = (ev_type, ta_val, tb_val, angle_deg)
                        # 反向有向边：角度不变（夹角是无向量）
                        event_dict[(v_idx, u_idx)] = (ev_type, tb_val, ta_val, angle_deg)
                
                edges = []
                # (B) N² 遍历，暴力生成完备边 (包含强负样本)
                for u in range(N_nodes):
                    for v in range(N_nodes):
                        # Self-loop 设为 0 (NONE)
                        if u == v:
                            edges.append({
                                "u": u, "v": v, "j_type": "NONE", "j_type_idx": 0,
                                "t_u": 0.0, "t_v": 0.0, "t_diff": 0.0, "t_prod": 0.0,
                                "angle_sin": 0.0, "angle_cos": 1.0
                            })
                            continue
                            
                        if (u, v) in event_dict:
                            ev_type, ta_val, tb_val, angle_deg = event_dict[(u, v)]
                            angle_rad = math.radians(angle_deg)
                            # 🌟 修复 3: 注入 t_diff 和 t_prod 进行对称性破缺 (Symmetry Breaking)
                            edges.append({
                                "u": u, "v": v,
                                "j_type": ev_type,
                                "j_type_idx": j_map.get(ev_type, 0),
                                "t_u": float(np.round(ta_val, 4)),
                                "t_v": float(np.round(tb_val, 4)),
                                "t_diff": float(np.round(abs(ta_val - tb_val), 4)),
                                "t_prod": float(np.round(ta_val * tb_val, 4)),
                                "angle_sin": float(np.round(math.sin(angle_rad), 6)),
                                "angle_cos": float(np.round(math.cos(angle_rad), 6))
                            })
                        else:
                            # 🌟 注入显式负样本（angle=0，sin=0, cos=1）
                            edges.append({
                                "u": u, "v": v, "j_type": "NONE", "j_type_idx": 0,
                                "t_u": 0.0, "t_v": 0.0, "t_diff": 0.0, "t_prod": 0.0,
                                "angle_sin": 0.0, "angle_cos": 1.0
                            })
                
                final_dataset.append({
                    "source_file": source_filename,  
                    "hex_key": hex_key,
                    "char": char_data.get("glyph_info", {}).get("char", ""),
                    "derivation": rule_name,
                    "num_nodes": N_nodes,
                    "num_edges": len(edges), # 必然是 N²
                    "nodes": nodes,
                    "edges": edges
                })

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)
        
    avg_derivation = len(final_dataset) / orig_sample_count if orig_sample_count > 0 else 0
        
    print(f"\n🎉 成功落盘！共组装 {len(final_dataset)} 个规范化图结构 (Canonical Complete Graph)。")
    print(f"📊 派生前有效字符样本数: {orig_sample_count} 个")
    print(f"📈 空间增强样本倍率: {avg_derivation:.1f}x")
    print(f"💾 数据已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()