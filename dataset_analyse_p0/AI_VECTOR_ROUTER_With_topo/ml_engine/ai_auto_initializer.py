import os
import json
import torch
import numpy as np
import copy
import sys

from ml_engine.train_pipeline import GraphEditorTransformer
from ml_engine.data_builder import GraphReplayEnvironment, ACTION_VOCAB

from scipy.spatial import cKDTree

# ==========================================
# 🌟 新增：极速重叠度计算函数
# ==========================================
def calculate_overlap_ratio(target_path, other_paths, distance_thresh=2.0):
    if not other_paths or len(target_path) == 0:
        return 0.0
    valid_others = [p for p in other_paths if len(p) > 0]
    if not valid_others:
        return 0.0
        
    all_other_pts = np.vstack(valid_others)
    tree = cKDTree(all_other_pts)
    dists, _ = tree.query(target_path, k=1, workers=-1)
    overlap_count = np.sum(dists < distance_thresh)
    return float(overlap_count) / len(target_path)


# 🌟 核心新增：引入主程序的贝塞尔拟合器作为 AI 的“物理法则裁判”
try:
    from geometry_vision import fit_bezier_basic_with_error
except ImportError:
    print("⚠️ 警告：无法导入 geometry_vision，AI 误差探测器可能失效。")
    # 提供一个兜底的假函数防止崩溃
    def fit_bezier_basic_with_error(path): return None, 0.0 

class UICompatibleAIExecutor:
    def __init__(self, model_filename="graph_editor_best.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🤖 Initializing UI-Compatible AI on {self.device}...")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_file = os.path.join(current_dir, model_filename)
        
        self.model = GraphEditorTransformer(feature_dim=9).to(self.device)
        self.model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        self.model.to(self.device)
        self.model.eval()
        
        self.vocab_inv = {v: k for k, v in ACTION_VOCAB.items()}
        self.env = GraphReplayEnvironment()
        
        # 🌟 定义 AI 的物理极限 (与 main.py 保持一致)
        self.MAX_BEZIER_ERROR = 3.0

    def _clean_path(self, path):
        if len(path) < 2: return path
        diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
        valid_indices = [0] + list(np.where(diffs > 0.1)[0] + 1)
        return path[valid_indices]

    def _physical_stitch(self, path_a, path_b):
        a_start, a_end = path_a[0], path_a[-1]
        b_start, b_end = path_b[0], path_b[-1]
        
        dists = {
            "end_to_start": np.linalg.norm(a_end - b_start),
            "end_to_end": np.linalg.norm(a_end - b_end),
            "start_to_start": np.linalg.norm(a_start - b_start),
            "start_to_end": np.linalg.norm(a_start - b_end)
        }
        best_mode = min(dists, key=dists.get)
        
        if best_mode == "end_to_start": merged = np.vstack([path_a, path_b])
        elif best_mode == "end_to_end": merged = np.vstack([path_a, path_b[::-1]])
        elif best_mode == "start_to_start": merged = np.vstack([path_a[::-1], path_b])
        elif best_mode == "start_to_end": merged = np.vstack([path_a[::-1], path_b[::-1]])
            
        return self._clean_path(merged)

    def generate_ai_init_graph(self, raw_edges, max_steps=40):
        print(f"\n[AI黑匣子] 收到初始请求，线条总数: {len(raw_edges)}")
        if not raw_edges: return []
            
        current_edges = copy.deepcopy(raw_edges)
        
        # 🌟 核心新增：AI 的“失败记忆库”
        taboo_pairs = set()   # 记录被物理引擎否决的合并对 (id_a, id_b)
        locked_nodes = set()  # 记录所有伙伴都尝试失败的“孤儿节点”，强制模型忽略它们
        
        for step in range(max_steps):
            if len(current_edges) <= 1: break
                
            x_feat, x_bias = self.env.extract_state_features(current_edges)
            if len(x_feat) == 0: break
                
            x_feat_t = torch.tensor(x_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
            padding_mask = torch.zeros((1, len(x_feat)), dtype=torch.bool).to(self.device)
            
            with torch.no_grad():
                type_logits, pointer_logits = self.model(x_feat_t, padding_mask)
            
            # ==========================================
            # 🌟 新增补丁：强力护栏 - 读心术与防偷懒机制
            # ==========================================
            probs = torch.softmax(type_logits[0], dim=0)
            prob_str = ", ".join([f"{self.vocab_inv[i]}:{probs[i]:.2f}" for i in range(len(probs))])
            print(f"[AI 脑电波] Step {step} 决策概率 -> {prob_str}")

            # 绝对物理拦截：如果还没走几步，或者画板上线条还很多，强行剥夺 Done 的权利！
            if step < 3 or len(current_edges) > 4:
                done_idx = ACTION_VOCAB["Done"]
                type_logits[0, done_idx] = -1e9  # 强行扣至负无穷，逼它去干活
            # ==========================================

            # ==========================================
            # 🌟 动作掩码 (Action Masking): 物理剥夺 AI 对死胡同节点的点击权
            # ==========================================
            for i, edge in enumerate(current_edges):
                if edge['id'] in locked_nodes:
                    pointer_logits[0, i] = -1e9 # 强行把概率降为极小值，迫使 argmax 转向其他健康线条
            
            # 如果所有的指针都被 mask 掉了，说明全图都处理不动了
            if pointer_logits[0].max().item() < -1e8:
                print(f"[AI黑匣子] 所有剩余节点均触达物理上限，提前终止！")
                break
            # ==========================================
                
            action_name = self.vocab_inv[type_logits[0].argmax().item()]
            if action_name == "Done": 
                print(f"[AI黑匣子] 第 {step} 步，模型判定清理完毕，输出 Done。")
                break
                
            target_idx = pointer_logits[0].argmax().item()
            if target_idx >= len(current_edges): break
            
            # if action_name == "Delete":
            #     if len(current_edges) > 3:
            #         target_edge = current_edges[target_idx]
            #         other_paths = [e['path'] for i, e in enumerate(current_edges) if i != target_idx]
                    
            #         # 🌟 修复 1：把物理探伤的视野从 2.0 像素放大到 6.0 像素，兼容视觉上的笔画厚度
            #         overlap_ratio = calculate_overlap_ratio(target_edge['path'], other_paths, distance_thresh=6.0)
                    
            #         # 🌟 修复 2：把要求 75% 重合的严苛条件降到 40%
            #         SAFE_DELETE_THRESHOLD = 0.40 
                    
            #         # 🌟 修复 3：新增“微小噪点特权”
            #         # 如果这根线长度不到 15 个像素（典型的交叉口碎线），无需重叠，直接批准删除！
            #         edge_length = len(target_edge['path'])
            #         is_tiny_noise = (edge_length < 15)
                    
            #         if overlap_ratio >= SAFE_DELETE_THRESHOLD or is_tiny_noise:
            #             # 物理裁判绿灯
            #             deleted = current_edges.pop(target_idx)
            #             reason = f"重叠度达标 {overlap_ratio*100:.1f}%" if not is_tiny_noise else f"清理交叉口微小碎线 (长度:{edge_length})"
            #             print(f"[AI黑匣子] -> ✅ 允许 Delete: 删除了 ID {deleted['id']} ({reason})")
            #         else:
            #             # 物理裁判红灯
            #             print(f"[AI黑匣子] -> ⛔ 拦截 Delete: ID {target_edge['id']} 是承重骨架，禁止删除！(重叠度仅: {overlap_ratio*100:.1f}%, 长度: {edge_length})")
            #             locked_nodes.add(target_edge['id'])
            #     else:
            #         break

            if action_name == "Delete":
                if len(current_edges) > 3:
                    target_edge = current_edges[target_idx]
                    other_paths = [e['path'] for i, e in enumerate(current_edges) if i != target_idx]
                    
                    # 🌟 贯彻你的覆盖率判定法：
                    # distance_thresh=5.0 模拟了笔画的平均覆盖半径。
                    # 如果原字体极粗，可以调大；如果极细，可以调小。
                    overlap_ratio = calculate_overlap_ratio(target_edge['path'], other_paths, distance_thresh=5.0)
                    
                    # 🌟 核心阈值：如果删掉这根线，它至少要有 60% 的面积被其他线“代为覆盖”。
                    # 否则，一旦删除就会导致原图覆盖率严重下降（漏风/缺口）！
                    MIN_COVERAGE_RETAINED = 0.60 
                    
                    if overlap_ratio >= MIN_COVERAGE_RETAINED:
                        # 覆盖率不会下降，放心删除
                        deleted = current_edges.pop(target_idx)
                        print(f"[AI黑匣子] -> ✅ 允许 Delete: 删掉不影响覆盖率 (冗余度: {overlap_ratio*100:.1f}%)")
                    else:
                        # 覆盖率将严重下降，强行锁定
                        print(f"[AI黑匣子] -> ⛔ 拦截 Delete: ID {target_edge['id']} 若删除会导致覆盖率严重下降！(冗余度仅: {overlap_ratio*100:.1f}%)")
                        locked_nodes.add(target_edge['id'])
                else:
                    break
                    
            elif action_name == "Merge":
                edge_a = current_edges.pop(target_idx)
                # 传入 taboo_pairs 黑名单
                best_b_idx = self._find_partner_for_ui(edge_a, current_edges, taboo_pairs)
                
                if best_b_idx is not None:
                    edge_b = current_edges.pop(best_b_idx)
                    merged_path = self._physical_stitch(edge_a['path'], edge_b['path'])
                    
                    try: _, error = fit_bezier_basic_with_error(merged_path)
                    except: error = float('inf')
                        
                    if error <= self.MAX_BEZIER_ERROR:
                        new_edge = {'id': 9000 + step, 'path': merged_path, 'control_points': [], 'type': 'bezier'}
                        current_edges.append(new_edge)
                        print(f"[AI黑匣子] -> Merge: 缝合 {edge_a['id']} 和 {edge_b['id']} (误差: {error:.2f} 🟢)")
                    else:
                        print(f"[AI黑匣子] -> ⛔ 拦截 Merge: {edge_a['id']}+{edge_b['id']} 变形 (误差: {error:.2f} 🔴)")
                        # 🌟 失败惩罚：拉黑这个组合，把线条放回原处，继续跑循环！
                        pair_key = tuple(sorted([edge_a['id'], edge_b['id']]))
                        taboo_pairs.add(pair_key)
                        
                        current_edges.append(edge_a)
                        current_edges.append(edge_b)
                else:
                    # 🌟 绝望判定：如果 A 找不到任何合法的 B，说明它是个死节点
                    # 把它塞回画板，并挂上“请勿打扰”的牌子
                    print(f"[AI黑匣子] -> 🔒 锁定节点 {edge_a['id']}: 无合法物理伙伴可合并。")
                    current_edges.append(edge_a)
                    locked_nodes.add(edge_a['id'])
                    
        print(f"[AI黑匣子] 推理结束，返回给前端线条数: {len(current_edges)}")
        if len(current_edges) == 0: return copy.deepcopy(raw_edges)
        return current_edges

    def _find_partner_for_ui(self, edge_a, remaining_edges, taboo_pairs):
        if not remaining_edges: return None
        pts_a = [np.array(edge_a['path'][0]), np.array(edge_a['path'][-1])]
        min_dist = float('inf')
        best_idx = None
        for i, edge_b in enumerate(remaining_edges):
            # 🌟 查阅黑名单：如果这俩之前被物理引擎否决过，直接跳过！
            pair_key = tuple(sorted([edge_a['id'], edge_b['id']]))
            if pair_key in taboo_pairs:
                continue 
                
            pts_b = [np.array(edge_b['path'][0]), np.array(edge_b['path'][-1])]
            for p_a in pts_a:
                for p_b in pts_b:
                    dist = np.linalg.norm(p_a - p_b)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i
        return best_idx