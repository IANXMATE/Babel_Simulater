import os
import json
import glob
import math
import numpy as np
from collections import defaultdict
from collections import Counter
from scipy.spatial import cKDTree

# 如果你已经在这个环境里装了 PyTorch，取消下面这行的注释
# import torch 

# 动作类别词典 (分类任务的 Label)
ACTION_VOCAB = {
    "Merge": 0,
    "Delete": 1,
    "Split": 2,
    "Add_Dot": 3,
    "Done": 4
}
def calculate_overlap_ratio(target_path, other_paths, distance_thresh=2.0):
    """
    极速重叠度计算：测算 target_path 中有多少比例的点，被其他线条覆盖。
    """
    if not other_paths or len(target_path) == 0:
        return 0.0
        
    valid_others = [p for p in other_paths if len(p) > 0]
    if not valid_others:
        return 0.0
        
    all_other_pts = np.vstack(valid_others)
    tree = cKDTree(all_other_pts)
    
    # 查询 target_path 中每个点到其他所有点的最近距离
    dists, _ = tree.query(target_path, k=1, workers=-1)
    
    # 统计距离小于阈值的点数，除以总长度即为重叠面积比例
    overlap_count = np.sum(dists < distance_thresh)
    return float(overlap_count) / len(target_path)

class GraphReplayEnvironment:
    """
    图编辑环境回放器：
    负责将离散的 JSON 录像带，转化为 Graph Transformer 需要的 (S_t, A_t) 张量对。
    """
    def __init__(self, tolerance=2.0):
        # 容差用于判断两个端点是否在物理上相连（应对浮点数误差）
        self.tolerance = tolerance 

    def _round_pt(self, pt):
        """将坐标圆整，用于哈希和图节点拓扑判断"""
        return (round(pt[0], 1), round(pt[1], 1))

    def _euclidean(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def _extract_single_edge_geometry(self, path, dt_map):
        """
        🌟 继承了你旧代码的灵魂，但转为了 Graph Transformer 需要的 Token 特征！
        提取单条边的绝对几何与物理特征。
        """
        path = np.array(path)
        
        # 1. 基础特征：绝对长度
        length = 0
        if len(path) > 1:
            diffs = np.diff(path, axis=0)
            length = np.sum(np.linalg.norm(diffs, axis=1))
            
        # 2. 物理特征：从 dt_map 获取起止点的绝对线宽 (继承你的 w_a, w_b)
        w_start = dt_map[int(path[0][0]), int(path[0][1])]
        w_end = dt_map[int(path[-1][0]), int(path[-1][1])]
        
        # 3. 几何切线特征：起止点的绝对发射角 (继承你的 step 向量计算)
        step = min(4, len(path) - 1)
        if step < 1: step = 1
        
        # 起点发出的向量
        vec_start = path[step] - path[0]
        norm_start = np.linalg.norm(vec_start)
        if norm_start > 1e-5:
            cos_start, sin_start = vec_start[0]/norm_start, vec_start[1]/norm_start
        else:
            cos_start, sin_start = 0.0, 0.0
            
        # 终点发出的向量 (注意方向是从倒数第 step 个点指向终点，表示流入终点的趋势)
        vec_end = path[-1] - path[-1-step]
        norm_end = np.linalg.norm(vec_end)
        if norm_end > 1e-5:
            cos_end, sin_end = vec_end[0]/norm_end, vec_end[1]/norm_end
        else:
            cos_end, sin_end = 0.0, 0.0

        # 返回 7 维独立特征
        return [length, w_start, w_end, cos_start, sin_start, cos_end, sin_end]
    
    def extract_state_features(self, edges):
        """
        核心引擎 1：计算单帧状态 S_t 的几何特征与 Attention Bias
        """
        N = len(edges)
        if N == 0:
            # 🌟 修复点 1：特征维度变为 9 维空数组
            return np.zeros((0, 9)), np.zeros((0, 0)) 

        # 1. 扫描拓扑，计算 Node Degree
        node_degrees = defaultdict(int)
        edge_endpoints = []
        for e in edges:
            path = e['path']
            p_start, p_end = self._round_pt(path[0]), self._round_pt(path[-1])
            edge_endpoints.append((p_start, p_end))
            node_degrees[p_start] += 1
            node_degrees[p_end] += 1

        # 2. 构建 Edge Token Features (几何 + 拓扑 + 重叠度)
        features = []
        for i, e in enumerate(edges):
            path = e['path']
            p_start, p_end = edge_endpoints[i]
            
            # --- 几何特征 ---
            length = sum(self._euclidean(path[k], path[k+1]) for k in range(len(path)-1)) if len(path)>1 else 0
            center_x = sum(p[0] for p in path) / len(path)
            center_y = sum(p[1] for p in path) / len(path)
            dx = path[-1][0] - path[0][0]
            dy = path[-1][1] - path[0][1]
            theta = math.atan2(dy, dx)
            
            # --- 拓扑特征 ---
            deg_start = node_degrees[p_start]
            deg_end = node_degrees[p_end]
            is_spur = 1.0 if (deg_start == 1 or deg_end == 1) else 0.0
            
            # --- 🌟 新增：重叠度特征 ---
            target_path = path
            other_paths = [e2['path'] for j, e2 in enumerate(edges) if j != i]
            overlap_ratio = calculate_overlap_ratio(target_path, other_paths, distance_thresh=2.0)
            
            # D = 9 维特征向量 (增加了 overlap_ratio)
            feat = [
                length / 800.0,            
                center_x / 400.0,          
                center_y / 400.0,          
                math.sin(theta),           
                math.cos(theta),           
                deg_start / 4.0,           
                deg_end / 4.0,             
                is_spur,                   
                overlap_ratio              
            ]
            
            features.append(feat)

        # 3. 构建 Graphormer Attention Bias 矩阵
        attn_bias = np.full((N, N), -1.0) 
        np.fill_diagonal(attn_bias, 0.0)  
        
        for i in range(N):
            pts_i = set(edge_endpoints[i])
            for j in range(i + 1, N):
                pts_j = set(edge_endpoints[j])
                if len(pts_i.intersection(pts_j)) > 0:
                    attn_bias[i, j] = 2.0 
                    attn_bias[j, i] = 2.0
                    
        return np.array(features, dtype=np.float32), np.array(attn_bias, dtype=np.float32)
    
    def extract_action_labels(self, edges_t, edges_next, action_str):
        """
        核心引擎 2：通过对比前后帧 (State Diffing)，反推专家操作的目标 Indices
        """
        # 解析动作名字，例如 "Merge (M)" -> "Merge"
        base_action = action_str.split(" ")[0].replace("Add", "Add_Dot")
        if base_action not in ACTION_VOCAB:
            base_action = "Done" # 兜底

        # 提取前后两帧的全局 ID 集合
        ids_t = {e['id'] for e in edges_t}
        ids_next = {e['id'] for e in edges_next}
        
        # 🌟 魔法所在：被专家操作的边，其 ID 一定会在下一帧消失！
        target_ids = list(ids_t - ids_next)
        
        # 将全局 ID 映射为当前 Token 序列的相对 Index [0, N-1]
        id_to_index = {e['id']: idx for idx, e in enumerate(edges_t)}
        target_indices = [id_to_index[tid] for tid in target_ids if tid in id_to_index]

        # 如果是 Split，需要进一步解算 1D 离散化打断点位置 (后续可实现)
        position_bin = -1 
        
        return ACTION_VOCAB[base_action], target_indices, position_bin

    def process_trajectory(self, char_hex, log_sequence):
        """将单个字符的整条录像带，切片为独立的 (S, A) 样本"""
        trajectory_samples = []
        
        # 过滤出有效步骤
        steps = [s for s in log_sequence if s["action"] not in ("Init (Raw)", "Re-edit Init")]
        # 获取第 0 步 (初始状态)
        initial_step = [s for s in log_sequence if s["action"] in ("Init (Raw)", "Re-edit Init")]
        if not initial_step or not steps:
            return []
            
        current_edges = initial_step[0]["edges"]
        
        for next_step in steps:
            next_action_str = next_step["action"]
            next_edges = next_step["edges"]
            
            # 1. 计算当前状态 S_t
            x_feat, x_bias = self.extract_state_features(current_edges)
            
            # 2. 对比获取当前动作 A_t (人类为什么把 current_edges 变成了 next_edges)
            y_type, y_targets, y_pos = self.extract_action_labels(current_edges, next_edges, next_action_str)
            
            # 3. 保存样本
            sample = {
                "char_hex": char_hex,
                "x_feat": x_feat,       # Shape: (N, D)
                "x_bias": x_bias,       # Shape: (N, N)
                "y_type": y_type,       # Int: 0-4
                "y_targets": y_targets, # List[Int]: [idx1, idx2] (Pointer 目标的索引)
                "y_pos_bin": y_pos      # Int (1D 位置分类，目前为 -1 预留)
            }
            trajectory_samples.append(sample)
            
            # 步进
            current_edges = next_edges
            
        # 🌟 轨迹终点：追加一个 Done 动作，告诉模型何时停手
        x_feat_final, x_bias_final = self.extract_state_features(current_edges)
        trajectory_samples.append({
            "char_hex": char_hex,
            "x_feat": x_feat_final,
            "x_bias": x_bias_final,
            "y_type": ACTION_VOCAB["Done"],
            "y_targets": [],
            "y_pos_bin": -1
        })
            
        return trajectory_samples

# ==========================================
# 批处理执行入口
# ==========================================
def build_dataset_from_logs(action_logs_dir, output_path):
    print(f"🚀 Starting Dataset Builder...")
    env = GraphReplayEnvironment()
    all_samples = []
    log_files = glob.glob(os.path.join(action_logs_dir, "*_actions*.json"))
    for file_path in log_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for hex_key, log_seq in data.items():
                samples = env.process_trajectory(hex_key, log_seq)
                all_samples.extend(samples)
                
    print(f"✅ Extracted {len(all_samples)} total (State, Action) pairs.")
    
    # 分析动作分布
    type_counts = defaultdict(int)
    for s in all_samples:
        type_counts[s["y_type"]] += 1
    
    inv_vocab = {v: k for k, v in ACTION_VOCAB.items()}
    print("📊 Label Distribution:")
    for k, v in type_counts.items():
        print(f"  - {inv_vocab[k]}: {v} samples")
        
    # 保存为 Numpy/Pickle 格式 (方便 DataLoader 读取)
    import pickle
    with open(output_path, 'wb') as f:
        pickle.dump(all_samples, f)
    print(f"💾 Dataset saved to {output_path}")

def analyze_action_logs(action_logs_dir):
    search_pattern = os.path.join(action_logs_dir, "*.json")
    all_files = glob.glob(search_pattern)

    if not all_files:
        print("❌ No action logs found!")
        return

    unique_fonts = set() # 🌟 新增：用集合来统计去重后的独立字体数
    total_chars = 0
    action_counts = Counter()
    trajectory_lengths = []
    action_bigrams = Counter() 

    print(f"🔍 Found {len(all_files)} action log file(s), starting analysis...")

    for file_path in all_files:
        filename = os.path.basename(file_path)
        # 🌟 核心修复：从 "微软雅黑_actions_1.json" 中精准提取出 "微软雅黑"
        font_name = filename.split("_actions")[0]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
            
            # 如果这个文件里确实有标注数据，才把该字体计入已开工名单
            if data:
                unique_fonts.add(font_name)

            for hex_key, log_sequence in data.items():
                total_chars += 1
                # 过滤掉 Init 状态，只看真实动作
                actual_actions = [step["action"] for step in log_sequence if step["action"] not in ("Init (Raw)", "Re-edit Init")]
                
                trajectory_lengths.append(len(actual_actions))
                action_counts.update(actual_actions)

                # 提取动作连招 (Bi-grams)
                for i in range(len(actual_actions) - 1):
                    a1 = actual_actions[i].split(" ")[0]
                    a2 = actual_actions[i+1].split(" ")[0]
                    action_bigrams[f"{a1} -> {a2}"] += 1

    if total_chars == 0:
        print("⚠️ Found files, but no valid character trajectories inside.")
        return

    # --- 打印统计报告 ---
    print("\n" + "="*45)
    print("📊 GRAPH EDITING DATASET STATISTICS")
    print("="*45)
    
    # 🌟 新增的字体级统计输出
    print(f"📂 Total Unique Fonts Annotated : {len(unique_fonts)}")
    if len(unique_fonts) > 0:
        print(f"   (List: {', '.join(list(unique_fonts)[:5])}{'...' if len(unique_fonts)>5 else ''})")
    
    print(f"🔠 Total Characters Annotated   : {total_chars}")
    if len(unique_fonts) > 0:
        print(f"📈 Avg Characters per Font      : {total_chars / len(unique_fonts):.1f}")
        
    print("-" * 45)
    print(f"Average Graph Edits per Char: {sum(trajectory_lengths)/total_chars:.2f} steps")
    print(f"Max Edits on a single Char  : {max(trajectory_lengths)} steps")
    
    print("\n🛠️ Action Type Distribution:")
    total_actions = sum(action_counts.values())
    for act, count in action_counts.most_common():
        print(f"  - {act.ljust(15)}: {count} ({count/total_actions*100:.1f}%)")

    print("\n🔗 Top 5 Action Sequences (The 'Expert Combos'):")
    for combo, count in action_bigrams.most_common(5):
        print(f"  - {combo.ljust(20)}: {count} times")
    print("="*45)

if __name__ == "__main__":
    # 配置路径
    # 1. 获取当前文件所在目录 (即 ml_engine 文件夹)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 获取上一级目录 (即整个工程的根目录)
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    
    # 3. 直接在根目录下拼接 action_logs (不要加 ..)
    ACTION_LOGS_DIR = os.path.join(PROJECT_ROOT, "action_logs")
    
    # 4. 输出文件保存在 ml_engine 目录下
    OUTPUT_FILE = os.path.join(CURRENT_DIR, "expert_bc_dataset.pkl")
    
    # ==========================================
    # 🔍 诊断打印：帮你肉眼确认路径对不对
    # ==========================================
    print("-" * 40)
    print(f"📁 Project Root:  {PROJECT_ROOT}")
    print(f"📂 Looking for Action Logs in: {ACTION_LOGS_DIR}")
    print(f"   -> Exists? {os.path.exists(ACTION_LOGS_DIR)}")
    print("-" * 40)
    
    if not os.path.exists(ACTION_LOGS_DIR):
        print("❌ Fatal Error: Cannot find 'action_logs' directory. Please check the folder structure.")
    else:
        build_dataset_from_logs(ACTION_LOGS_DIR, OUTPUT_FILE)
    
    build_dataset_from_logs(ACTION_LOGS_DIR, OUTPUT_FILE)
    analyze_action_logs(ACTION_LOGS_DIR)