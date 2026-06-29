# AI Vector Router — 字体初始化模型全量技术文档

> 生成时间：2026-06-29  
> 模型路径：`ml_engine/graph_editor_best.pth`  
> 所属系统：`AI_VECTOR_ROUTER_With_topo`

---

## 目录

1. [系统概述与完整流程](#1-系统概述与完整流程)
2. [数据来源：主程序录制逻辑（main_with_record_action_and_model.py）](#2-数据来源主程序录制逻辑)
3. [数据构建脚本（data_builder.py）](#3-数据构建脚本-data_builderpy)
4. [模型训练脚本（train_pipeline.py）](#4-模型训练脚本-train_pipelinepy)
5. [AI 推理执行器（ai_auto_initializer.py）](#5-ai-推理执行器-ai_auto_initializerpy)
6. [设计决策与关键机制说明](#6-设计决策与关键机制说明)

---

## 1. 系统概述与完整流程

本系统是一个**行为克隆（Behavioral Cloning）**驱动的字形矢量骨架自动初始化模型。

**核心任务**：
给定一个字形的原始骨架线条集合（由形态学细化 + 图剪枝得到），AI 自动学习人类专家的手工清理策略，将杂乱的骨架边图整理为干净的笔画骨架图，供后续 Topo 标注使用。

**整体流程：**

```
字体文件 (.ttf/.otf)
    │
    ▼ render_unicode_glyph() → 二值 mask
    ▼ medial_axis() → 骨架图
    ▼ prune_spurs() + collapse_degree2_nodes() → 粗骨架
    ▼ split_pixel_path_adaptively() → edges（初始边集合）
    │
    ├─── [人工模式] 人类标注专家手动编辑（Merge/Delete/Split/Add_Dot）
    │                      ↓
    │              action_logs/{字体名}_actions.json（操作录像）
    │
    │─── [AI模式]  ai_executor.generate_ai_init_graph(edges)
    │                      ↓
    │              多步自回归清理（最多 40 步）
    │
    ▼
清理后的 edges
    │
    ▼ action_proceed_to_phase2() → 进入拓扑标注阶段
```

**训练数据生成流程：**

```
action_logs/*.json
    │
    ▼ GraphReplayEnvironment.process_trajectory()
    │    State Diffing: ids_t - ids_next = 被操作的边
    ▼
expert_bc_dataset.pkl
    [(x_feat [N,9], x_bias [N,N], y_type int, y_targets [idx])]
    │
    ▼ GraphEditorTransformer 训练 (200 epochs)
    ▼
graph_editor_best.pth
```

---

## 2. 数据来源：主程序录制逻辑

以下是 `main_with_record_action_and_model.py` 中与 AI 训练数据生成直接相关的代码片段。

### 2.1 全局配置与 AI 模块检测

```python
# ==========================================
# ⚙️ 全局配置与 AI 探测
# ==========================================
CANVAS_SIZE = 400
MAX_BEZIER_ERROR = 4.5
MAX_SPUR_LENGTH = 20.0
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 动作日志与拓扑输出目录
ACTION_LOG_DIR = os.path.join(SCRIPT_DIR, "action_logs")
os.makedirs(ACTION_LOG_DIR, exist_ok=True)

# AI 模块探测
_model_path = os.path.join(SCRIPT_DIR, "ml_engine", "graph_editor_best.pth")
HAS_AI_MODULE = False
try:
    from ml_engine.ai_auto_initializer import UICompatibleAIExecutor
    HAS_AI_MODULE = True
except Exception as e:
    print(f"❌ AI 代码模块导入失败: {e}")
```

### 2.2 初始骨架提取（load_char_topology）

将字形二值图转化为初始边集合，这是训练数据的"原始状态 S_0"：

```python
def load_char_topology(self):
    skel, distance = medial_axis(self.binary, return_distance=True)
    skel_obj = Skeleton(skel)
    branch_data = summarize(skel_obj, separator='-')
    G = nx.MultiGraph()
    for index, row in branch_data.iterrows():
        coords = skel_obj.path_coordinates(index)
        if len(coords) > 2:
            path = np.column_stack([coords[:, 1], coords[:, 0]])
            src, dst = int(row['node-id-src']), int(row['node-id-dst'])
            if np.linalg.norm(path[0] - skel_obj.coordinates[src][::-1]) > 1.0: path = path[::-1]
            G.add_edge(src, dst, key=index, path=path)

    G = prune_spurs(G, max_length=MAX_SPUR_LENGTH)   # 剪除毛刺
    G = collapse_degree2_nodes(G)                      # 合并度2节点

    self.edges = []
    global_id = 0
    for u, v, k, d in G.edges(keys=True, data=True):
        sub_paths = split_pixel_path_adaptively(d['path'], MAX_BEZIER_ERROR)
        for sp in sub_paths:
            self.edges.append({'id': global_id, 'path': sp.copy()})
            global_id += 1

    self.pure_raw_edges = copy.deepcopy(self.edges)
    self.init_mode = "raw"
    # 录像起点：记录初始状态
    self.action_log = [{"action": "Init (Raw)", "edges": copy.deepcopy(self.edges)}]
```

### 2.3 操作录像记录（record_step）

每次人类专家执行编辑操作后，将**操作后的完整图状态**追加到 action_log：

```python
def record_step(self, action_name):
    if hasattr(self, 'action_log'):
        self.action_log.append({
            "action": action_name,
            "edges": copy.deepcopy(self.edges)
        })
```

**触发时机（键盘快捷键绑定）：**

| 按键 | 操作 | record_step 记录名 |
|------|------|------------------|
| `M`  | 合并选中的两条边 | `"Merge (M)"` |
| `D`  | 删除选中的边 | `"Delete (D)"` |
| `C`  | 删除选中两条中较短的平行边 | `"Prune (C)"` |
| `B`  | 在点击处断开边 | `"Split (B)"` |
| `A`  | 在点击处添加圆点 | `"Add Dot (A)"` |

### 2.4 Merge 操作的物理约束

合并时验证贝塞尔拟合误差，超限则自动拆分：

```python
def action_merge(self):
    if len(self.selected_edge_ids) == 2:
        self.save_state()
        id_a, id_b = self.selected_edge_ids
        paths_to_stitch = [e['path'] for e in self.edges if e['id'] in (id_a, id_b)]
        new_unified_path = stitch_paths(paths_to_stitch)

        _, error = fit_bezier_basic_with_error(new_unified_path)

        sub_paths = []
        if error > MAX_BEZIER_ERROR:  # MAX_BEZIER_ERROR = 4.5
            sub_paths = split_pixel_path_adaptively(new_unified_path, MAX_BEZIER_ERROR)
            QMessageBox.warning(self, "Fitting Alert",
                f"Merge resulted in a high deviation (Error: {error:.2f}).\nSplit into {len(sub_paths)} parts.")
        else:
            sub_paths = [new_unified_path]

        self.edges = [e for e in self.edges if e['id'] not in (id_a, id_b)]
        new_id = max([e['id'] for e in self.edges] + [0]) + 1
        for sp in sub_paths:
            self.edges.append({'id': new_id, 'path': sp})
            new_id += 1

        self.record_step("Merge (M)")
```

### 2.5 动作日志保存（action_save_phase1_only / action_proceed_to_phase2）

保存时将内存中的 action_log 序列化为 JSON，写入 `action_logs/{字体名}_actions.json`：

```python
if hasattr(self, 'action_log'):
    log_file = os.path.join(ACTION_LOG_DIR, f"{self.font_filename}_actions.json")
    all_logs = {}
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f: all_logs = json.load(f)
        except: pass

    serialized_log = []
    for step in self.action_log:
        step_edges = [
            {
                "id": e['id'],
                "path": e['path'].tolist() if isinstance(e['path'], np.ndarray) else e['path']
            }
            for e in step["edges"]
        ]
        serialized_log.append({"action": step["action"], "edges": step_edges})

    # 仅在内容变化时写入（防止空转覆写）
    if all_logs.get(hex_key) != serialized_log:
        all_logs[hex_key] = serialized_log
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(all_logs, f, ensure_ascii=False, indent=2)
```

**最终保存的 JSON 结构：**

```json
{
  "U+4E00": [
    { "action": "Init (Raw)", "edges": [{"id": 0, "path": [[x,y], ...]}, ...] },
    { "action": "Merge (M)",  "edges": [{"id": 2, "path": [[x,y], ...]}, ...] },
    { "action": "Delete (D)", "edges": [{"id": 2, "path": [[x,y], ...]}, ...] }
  ]
}
```

### 2.6 AI 模式切换（action_toggle_init）

推理时切换到 AI 初始化模式，调用训练好的模型处理原始边集合：

```python
def action_toggle_init(self):
    if not self.has_ai or self.ai_executor is None:
        QMessageBox.warning(self, "AI Offline", "AI 模型未就绪")
        return

    if self.init_mode == "raw":
        ai_output = self.ai_executor.generate_ai_init_graph(
            copy.deepcopy(self.pure_raw_edges)
        )
        if not ai_output: raise ValueError("AI 返回空路径集。")
        self.edges = ai_output
        self.init_mode = "ai"
        self.record_step("AI Process Applied")
    else:
        # 一键还原到原始状态
        self.edges = copy.deepcopy(self.pure_raw_edges)
        self.init_mode = "raw"
        self.record_step("Revert to Raw")
```

---

## 3. 数据构建脚本（data_builder.py）

> 职责：读取 `action_logs/*.json` 录像带，通过 State Diffing 反推专家操作，生成 `(状态S, 动作A)` 训练样本对，保存为 `expert_bc_dataset.pkl`。

---

```python
import os
import json
import glob
import math
import numpy as np
from collections import defaultdict
from collections import Counter
from scipy.spatial import cKDTree

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
    dists, _ = tree.query(target_path, k=1, workers=-1)
    overlap_count = np.sum(dists < distance_thresh)
    return float(overlap_count) / len(target_path)


class GraphReplayEnvironment:
    """
    图编辑环境回放器：
    负责将离散的 JSON 录像带，转化为 Graph Transformer 需要的 (S_t, A_t) 张量对。
    """
    def __init__(self, tolerance=2.0):
        self.tolerance = tolerance

    def _round_pt(self, pt):
        return (round(pt[0], 1), round(pt[1], 1))

    def _euclidean(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def extract_state_features(self, edges):
        """
        核心引擎 1：计算单帧状态 S_t 的几何特征与 Attention Bias
        每条边被编码为 9 维特征向量：
          [0] length/800        路径归一化长度
          [1] center_x/400      路径中心 X
          [2] center_y/400      路径中心 Y
          [3] sin(θ)            路径方向角正弦
          [4] cos(θ)            路径方向角余弦
          [5] deg_start/4       起点连接度（拓扑特征）
          [6] deg_end/4         终点连接度（拓扑特征）
          [7] is_spur           是否为悬挂端（毛刺特征）
          [8] overlap_ratio     与其他路径的像素重叠度（用 cKDTree 计算）
        """
        N = len(edges)
        if N == 0:
            return np.zeros((0, 9)), np.zeros((0, 0))

        node_degrees = defaultdict(int)
        edge_endpoints = []
        for e in edges:
            path = e['path']
            p_start, p_end = self._round_pt(path[0]), self._round_pt(path[-1])
            edge_endpoints.append((p_start, p_end))
            node_degrees[p_start] += 1
            node_degrees[p_end] += 1

        features = []
        for i, e in enumerate(edges):
            path = e['path']
            p_start, p_end = edge_endpoints[i]

            length = sum(self._euclidean(path[k], path[k+1]) for k in range(len(path)-1)) if len(path)>1 else 0
            center_x = sum(p[0] for p in path) / len(path)
            center_y = sum(p[1] for p in path) / len(path)
            dx = path[-1][0] - path[0][0]
            dy = path[-1][1] - path[0][1]
            theta = math.atan2(dy, dx)

            deg_start = node_degrees[p_start]
            deg_end = node_degrees[p_end]
            is_spur = 1.0 if (deg_start == 1 or deg_end == 1) else 0.0

            target_path = path
            other_paths = [e2['path'] for j, e2 in enumerate(edges) if j != i]
            overlap_ratio = calculate_overlap_ratio(target_path, other_paths, distance_thresh=2.0)

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

        # Graphormer Attention Bias 矩阵
        # 自身=0.0, 共享端点（物理相连）=+2.0, 无连接=-1.0
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

        关键逻辑：被专家操作的边，其 ID 一定会在下一帧消失！
          target_ids = ids_t - ids_next
        """
        base_action = action_str.split(" ")[0].replace("Add", "Add_Dot")
        if base_action not in ACTION_VOCAB:
            base_action = "Done"

        ids_t = {e['id'] for e in edges_t}
        ids_next = {e['id'] for e in edges_next}
        target_ids = list(ids_t - ids_next)

        id_to_index = {e['id']: idx for idx, e in enumerate(edges_t)}
        target_indices = [id_to_index[tid] for tid in target_ids if tid in id_to_index]

        position_bin = -1
        return ACTION_VOCAB[base_action], target_indices, position_bin

    def process_trajectory(self, char_hex, log_sequence):
        """将单个字符的整条录像带，切片为独立的 (S, A) 样本"""
        trajectory_samples = []

        steps = [s for s in log_sequence if s["action"] not in ("Init (Raw)", "Re-edit Init")]
        initial_step = [s for s in log_sequence if s["action"] in ("Init (Raw)", "Re-edit Init")]
        if not initial_step or not steps:
            return []

        current_edges = initial_step[0]["edges"]

        for next_step in steps:
            next_action_str = next_step["action"]
            next_edges = next_step["edges"]

            x_feat, x_bias = self.extract_state_features(current_edges)
            y_type, y_targets, y_pos = self.extract_action_labels(
                current_edges, next_edges, next_action_str
            )

            sample = {
                "char_hex": char_hex,
                "x_feat": x_feat,       # Shape: (N, 9)
                "x_bias": x_bias,       # Shape: (N, N)
                "y_type": y_type,       # Int: 0-4
                "y_targets": y_targets, # List[Int]: 被操作边的相对索引
                "y_pos_bin": y_pos      # Int (预留位，目前为 -1)
            }
            trajectory_samples.append(sample)
            current_edges = next_edges

        # 轨迹终点：追加 Done 样本，告诉模型何时停手
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

    type_counts = defaultdict(int)
    for s in all_samples:
        type_counts[s["y_type"]] += 1
    inv_vocab = {v: k for k, v in ACTION_VOCAB.items()}
    print("📊 Label Distribution:")
    for k, v in type_counts.items():
        print(f"  - {inv_vocab[k]}: {v} samples")

    import pickle
    with open(output_path, 'wb') as f:
        pickle.dump(all_samples, f)
    print(f"💾 Dataset saved to {output_path}")
```

---

## 4. 模型训练脚本（train_pipeline.py）

> 职责：定义 `GraphEditorTransformer` 模型架构，加载 `expert_bc_dataset.pkl`，使用 Focal Loss + 指针网络联合损失训练，保存最佳权重到 `graph_editor_best.pth`。

---

```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle

try:
    from ml_engine.data_builder import build_dataset_from_logs, ACTION_VOCAB
except:
    from data_builder import build_dataset_from_logs, ACTION_VOCAB


class MultiClassFocalLoss(nn.Module):
    """
    针对多分类图编辑动作的 Focal Loss
    解决 Delete 动作过度泛滥（数据不平衡）问题：
      loss = alpha_t * (1-pt)^gamma * CE_loss
    gamma=2.0 会大幅压制模型对简单、高置信样本的学习，让它更专注于难样本。
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_factor = (1 - pt) ** self.gamma
        loss = focal_factor * ce_loss
        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class GraphEditingDataset(Dataset):
    """
    PyTorch Dataset，将 pkl 里的样本 list 封装为固定 shape 的 Tensor。
    max_edges=64：最大边数，超过的截断，不足的 padding。
    """
    def __init__(self, data_list, max_edges=64):
        self.data = data_list
        self.max_edges = max_edges

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        x_feat_np = sample["x_feat"]  # (N, 9)
        x_bias_np = sample["x_bias"]  # (N, N)

        N, D = x_feat_np.shape
        actual_n = min(N, self.max_edges)

        x_feat = torch.zeros((self.max_edges, D), dtype=torch.float32)
        x_bias = torch.full((self.max_edges, self.max_edges), -1.0, dtype=torch.float32)
        padding_mask = torch.ones(self.max_edges, dtype=torch.bool)  # True=padding位置

        if actual_n > 0:
            x_feat[:actual_n, :] = torch.tensor(x_feat_np[:actual_n, :])
            x_bias[:actual_n, :actual_n] = torch.tensor(x_bias_np[:actual_n, :actual_n])
            padding_mask[:actual_n] = False  # False=真实边

        y_type = torch.tensor(sample["y_type"], dtype=torch.long)

        y_targets = sample["y_targets"]
        target_idx = y_targets[0] if len(y_targets) > 0 and y_targets[0] < self.max_edges else -1
        y_target1 = torch.tensor(target_idx, dtype=torch.long)

        return {
            "x_feat": x_feat,
            "x_bias": x_bias,
            "padding_mask": padding_mask,
            "y_type": y_type,
            "y_target1": y_target1
        }


class GraphEditorTransformer(nn.Module):
    """
    Graph Transformer 模型架构：

    输入：x_feat [B, N, 9]（边特征）+ padding_mask [B, N]
    ↓
    Linear(9 → 128) + LayerNorm（特征升维 + 稳定化）
    ↓
    Transformer Encoder × 3 层（d_model=128, n_heads=4, FFN=256, GELU）
    ↓
    Global Average Pooling → global_context [B, 128]
    ↓
    ┌─────────────────────┬────────────────────────────────┐
    │ [Type Head]         │ [Pointer Head]                 │
    │ Linear(128→64)→ReLU │ Query = Linear(global_context) │
    │ Linear(64→5)        │ Key   = Linear(encoded_tokens) │
    │ → type_logits [B,5] │ Score = Q·Kᵀ → [B, N]         │
    └─────────────────────┴────────────────────────────────┘
    """
    def __init__(self, feature_dim=8, hidden_dim=128, n_heads=4, n_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.edge_embedding = nn.Linear(feature_dim, hidden_dim)
        # LayerNorm：防止特征数值爆炸，让 Transformer 输入极度平滑
        self.input_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Type Head：预测动作类型（5 分类）
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, len(ACTION_VOCAB))
        )

        # Pointer Head：指针网络，预测要操作的边的索引
        self.pointer_query = nn.Linear(hidden_dim, hidden_dim)
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_feat, padding_mask):
        B, N, D = x_feat.shape

        tokens = self.edge_embedding(x_feat)
        tokens = self.input_norm(tokens)  # 升维后立刻 LayerNorm

        encoded_tokens = self.transformer(tokens, src_key_padding_mask=padding_mask)

        # Global Average Pooling（屏蔽 padding 位置）
        active_tokens = encoded_tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        valid_counts = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
        global_context = active_tokens.sum(dim=1) / valid_counts

        # Type Head
        type_logits = self.type_head(global_context)  # (B, 5)

        # Pointer Head（内积打分 + 屏蔽 padding）
        query = self.pointer_query(global_context).unsqueeze(1)   # (B, 1, hidden_dim)
        keys = self.pointer_key(encoded_tokens)                    # (B, N, hidden_dim)
        pointer_logits = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)  # (B, N)
        pointer_logits = pointer_logits.masked_fill(padding_mask, float('-inf'))

        return type_logits, pointer_logits


def train():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    ACTION_LOGS_DIR = os.path.join(PROJECT_ROOT, "action_logs")
    DATASET_CACHE = os.path.join(CURRENT_DIR, "expert_bc_dataset.pkl")

    # 每次都重构数据集（改为 False 可跳过）
    if not os.path.exists(DATASET_CACHE) or True:
        print("🔄 Building dataset from action logs...")
        build_dataset_from_logs(ACTION_LOGS_DIR, DATASET_CACHE)

    print(f"📦 Loading dataset from {DATASET_CACHE}")
    with open(DATASET_CACHE, 'rb') as f:
        raw_data = pickle.load(f)

    dataset = GraphEditingDataset(raw_data, max_edges=64)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Using device: {device}")

    model = GraphEditorTransformer(feature_dim=9).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 类别权重：强压 Delete 泛滥，拔高 Merge 比重
    # ACTION_VOCAB 顺序: Merge:0, Delete:1, Split:2, Add_Dot:3, Done:4
    class_weights = torch.ones(len(ACTION_VOCAB), device=device)
    class_weights[ACTION_VOCAB["Delete"]] = 1.0   # 抑制泛滥
    class_weights[ACTION_VOCAB["Merge"]] = 5.0    # 拔高权重
    class_weights[ACTION_VOCAB["Done"]] = 0.1     # 保证模型知道何时停手

    criterion_type = MultiClassFocalLoss(alpha=class_weights, gamma=2.0)

    epochs = 200
    best_loss = float('inf')
    MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, "graph_editor_best.pth")
    print("\n🚀 Starting Training...")

    for epoch in range(epochs):
        model.train()
        total_type_loss, total_ptr_loss = 0, 0
        correct_type, correct_ptr, total_ptr_targets = 0, 0, 0

        for batch in dataloader:
            x_feat = batch["x_feat"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            y_type = batch["y_type"].to(device)
            y_target1 = batch["y_target1"].to(device)

            optimizer.zero_grad()
            type_logits, pointer_logits = model(x_feat, padding_mask)

            # Loss 1：动作类型（Focal Loss + 类别权重）
            loss_type = criterion_type(type_logits, y_type)

            # Loss 2：指针目标（CE Loss，仅对有目标的样本）
            valid_ptr_mask = y_target1 != -1
            if valid_ptr_mask.sum() > 0:
                loss_ptr = F.cross_entropy(
                    pointer_logits[valid_ptr_mask],
                    y_target1[valid_ptr_mask]
                )
            else:
                loss_ptr = torch.tensor(0.0, device=device)

            loss = loss_type + loss_ptr
            loss.backward()
            optimizer.step()

            total_type_loss += loss_type.item()
            total_ptr_loss += loss_ptr.item()
            pred_type = type_logits.argmax(dim=-1)
            correct_type += (pred_type == y_type).sum().item()
            if valid_ptr_mask.sum() > 0:
                pred_ptr = pointer_logits[valid_ptr_mask].argmax(dim=-1)
                correct_ptr += (pred_ptr == y_target1[valid_ptr_mask]).sum().item()
                total_ptr_targets += valid_ptr_mask.sum().item()

        avg_type_acc = correct_type / len(dataset) * 100
        avg_ptr_acc = (correct_ptr / total_ptr_targets * 100) if total_ptr_targets > 0 else 0.0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:02d}/{epochs}] | "
                  f"Type Loss: {total_type_loss/len(dataloader):.4f} (Acc: {avg_type_acc:.1f}%) | "
                  f"Ptr Loss: {total_ptr_loss/len(dataloader):.4f} (Acc: {avg_ptr_acc:.1f}%)")

        epoch_avg_loss = (total_type_loss + total_ptr_loss) / len(dataloader)
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  👉 [Checkpoint] Best model saved! (Loss: {best_loss:.4f})")

    print(f"\n🎉 Training Complete! Best weights at:\n{MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
```

---

## 5. AI 推理执行器（ai_auto_initializer.py）

> 职责：加载训练好的 `graph_editor_best.pth`，在推理时以多步自回归方式自动清理初始骨架边集合，并用物理护栏（贝塞尔误差校验 + 覆盖率校验）拦截非物理操作。

---

```python
import os
import json
import torch
import numpy as np
import copy
import sys

from ml_engine.train_pipeline import GraphEditorTransformer
from ml_engine.data_builder import GraphReplayEnvironment, ACTION_VOCAB
from scipy.spatial import cKDTree

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

try:
    from geometry_vision import fit_bezier_basic_with_error
except ImportError:
    print("⚠️ 警告：无法导入 geometry_vision，AI 误差探测器可能失效。")
    def fit_bezier_basic_with_error(path): return None, 0.0


class UICompatibleAIExecutor:
    """
    推理执行器。核心方法：generate_ai_init_graph(raw_edges)
    采用多步自回归循环（最多 max_steps=40 步）：
      每步：抽取当前图状态特征 → 模型预测动作类型+目标边 → 物理校验 → 执行动作
    配合三重物理护栏：
      1. 强制步骤护栏：前3步和边数>4时，禁止 Done（防止偷懒）
      2. taboo_pairs：被贝塞尔否决的合并对，不再重试
      3. locked_nodes：无合法伙伴或被覆盖率否决的节点，屏蔽 pointer
    """
    def __init__(self, model_filename="graph_editor_best.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_file = os.path.join(current_dir, model_filename)

        self.model = GraphEditorTransformer(feature_dim=9).to(self.device)
        self.model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        self.model.to(self.device)
        self.model.eval()

        self.vocab_inv = {v: k for k, v in ACTION_VOCAB.items()}
        self.env = GraphReplayEnvironment()
        self.MAX_BEZIER_ERROR = 3.0  # Merge 后贝塞尔拟合误差阈值（像素）

    def _clean_path(self, path):
        if len(path) < 2: return path
        diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
        valid_indices = [0] + list(np.where(diffs > 0.1)[0] + 1)
        return path[valid_indices]

    def _physical_stitch(self, path_a, path_b):
        """物理拼接：找最近端点对，自动翻转方向拼接"""
        a_start, a_end = path_a[0], path_a[-1]
        b_start, b_end = path_b[0], path_b[-1]
        dists = {
            "end_to_start":   np.linalg.norm(a_end - b_start),
            "end_to_end":     np.linalg.norm(a_end - b_end),
            "start_to_start": np.linalg.norm(a_start - b_start),
            "start_to_end":   np.linalg.norm(a_start - b_end)
        }
        best_mode = min(dists, key=dists.get)
        if best_mode == "end_to_start":   merged = np.vstack([path_a, path_b])
        elif best_mode == "end_to_end":   merged = np.vstack([path_a, path_b[::-1]])
        elif best_mode == "start_to_start": merged = np.vstack([path_a[::-1], path_b])
        elif best_mode == "start_to_end": merged = np.vstack([path_a[::-1], path_b[::-1]])
        return self._clean_path(merged)

    def _find_partner_for_ui(self, edge_a, remaining_edges, taboo_pairs):
        """为 edge_a 寻找端点距离最近的合并伙伴（跳过黑名单）"""
        if not remaining_edges: return None
        pts_a = [np.array(edge_a['path'][0]), np.array(edge_a['path'][-1])]
        min_dist = float('inf')
        best_idx = None
        for i, edge_b in enumerate(remaining_edges):
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

    def generate_ai_init_graph(self, raw_edges, max_steps=40):
        """
        主推理循环。输入原始边集合，输出清理后的边集合。
        """
        print(f"\n[AI黑匣子] 收到初始请求，线条总数: {len(raw_edges)}")
        if not raw_edges: return []

        current_edges = copy.deepcopy(raw_edges)
        taboo_pairs = set()    # 被物理引擎否决的合并对
        locked_nodes = set()   # 无合法伙伴的"孤儿节点"

        for step in range(max_steps):
            if len(current_edges) <= 1: break

            x_feat, x_bias = self.env.extract_state_features(current_edges)
            if len(x_feat) == 0: break

            x_feat_t = torch.tensor(x_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
            padding_mask = torch.zeros((1, len(x_feat)), dtype=torch.bool).to(self.device)

            with torch.no_grad():
                type_logits, pointer_logits = self.model(x_feat_t, padding_mask)

            # 护栏 1：步骤前 3 步或边数 > 4，强制屏蔽 Done
            if step < 3 or len(current_edges) > 4:
                type_logits[0, ACTION_VOCAB["Done"]] = -1e9

            # 护栏 2：locked_nodes 中的边，强制屏蔽 pointer
            for i, edge in enumerate(current_edges):
                if edge['id'] in locked_nodes:
                    pointer_logits[0, i] = -1e9

            if pointer_logits[0].max().item() < -1e8:
                print(f"[AI黑匣子] 所有节点均触达物理上限，提前终止！")
                break

            action_name = self.vocab_inv[type_logits[0].argmax().item()]
            if action_name == "Done":
                print(f"[AI黑匣子] 第 {step} 步，Done。")
                break

            target_idx = pointer_logits[0].argmax().item()
            if target_idx >= len(current_edges): break

            if action_name == "Delete":
                if len(current_edges) > 3:
                    target_edge = current_edges[target_idx]
                    other_paths = [e['path'] for i, e in enumerate(current_edges) if i != target_idx]
                    overlap_ratio = calculate_overlap_ratio(
                        target_edge['path'], other_paths, distance_thresh=5.0
                    )
                    MIN_COVERAGE_RETAINED = 0.60  # 至少 60% 被其他线覆盖才允许删除
                    if overlap_ratio >= MIN_COVERAGE_RETAINED:
                        deleted = current_edges.pop(target_idx)
                        print(f"[AI黑匣子] -> ✅ Delete: 冗余度 {overlap_ratio*100:.1f}%")
                    else:
                        print(f"[AI黑匣子] -> ⛔ 拦截 Delete: 冗余度仅 {overlap_ratio*100:.1f}%")
                        locked_nodes.add(target_edge['id'])
                else:
                    break

            elif action_name == "Merge":
                edge_a = current_edges.pop(target_idx)
                best_b_idx = self._find_partner_for_ui(edge_a, current_edges, taboo_pairs)

                if best_b_idx is not None:
                    edge_b = current_edges.pop(best_b_idx)
                    merged_path = self._physical_stitch(edge_a['path'], edge_b['path'])

                    try: _, error = fit_bezier_basic_with_error(merged_path)
                    except: error = float('inf')

                    if error <= self.MAX_BEZIER_ERROR:
                        new_edge = {
                            'id': 9000 + step, 'path': merged_path,
                            'control_points': [], 'type': 'bezier'
                        }
                        current_edges.append(new_edge)
                        print(f"[AI黑匣子] -> Merge (误差: {error:.2f} 🟢)")
                    else:
                        # 失败惩罚：拉黑此对，放回原处
                        print(f"[AI黑匣子] -> ⛔ 拦截 Merge (误差: {error:.2f} 🔴)")
                        pair_key = tuple(sorted([edge_a['id'], edge_b['id']]))
                        taboo_pairs.add(pair_key)
                        current_edges.append(edge_a)
                        current_edges.append(edge_b)
                else:
                    # 无合法伙伴，挂上"孤儿"标记
                    print(f"[AI黑匣子] -> 🔒 锁定节点 {edge_a['id']}: 无合法物理伙伴。")
                    current_edges.append(edge_a)
                    locked_nodes.add(edge_a['id'])

        print(f"[AI黑匣子] 推理结束，返回线条数: {len(current_edges)}")
        if len(current_edges) == 0: return copy.deepcopy(raw_edges)
        return current_edges
```

---

## 6. 设计决策与关键机制说明

### 6.1 为什么用 Behavioral Cloning 而非强化学习

字形骨架清理是一个**稀疏奖励、高精度**的任务。RL 需要定义奖励函数（如"最终 Topo 标注质量"），但 Topo 质量的评估本身就需要人工参与，无法自动化。BC 直接学习专家轨迹，收敛快，且专家的操作具有很强的一致性规律（Merge 后贝塞尔误差必须 < 4.5px，Delete 前覆盖率必须 > 60%），这些规律可以被模型可靠地捕捉。

### 6.2 State Diffing 的关键假设

`ids_t - ids_next = 被操作的边 ID`，这依赖于**每次操作完成后边的 ID 不复用**（新边分配新 ID）。主程序 `action_merge()` 中确实遵守了这一原则：`new_id = max([e['id'] for e in self.edges] + [0]) + 1`，保证全局单调递增。

### 6.3 Focal Loss + 类别权重的组合策略

| 动作 | 类别权重 | 说明 |
|------|---------|------|
| Merge | ×5.0 | 最重要的操作，人工数据中频率高但被 Delete 淹没 |
| Delete | ×1.0 | 数据中最多，但错误删除会破坏骨架完整性 |
| Split | ×1.0 | 较少见，权重保持默认 |
| Add_Dot | ×1.0 | 较少见，权重保持默认 |
| Done | ×0.1 | 极度压制，防止模型过早停手 |

Focal Loss（gamma=2.0）在此基础上进一步让模型关注"难样本"——即那些模型不确定该操作什么的复杂图状态。

### 6.4 指针网络（Pointer Network）的作用

模型不是从固定大小的 token 里分类出目标，而是用**全局上下文向量作为 Query，与所有边的 Key 做内积打分**，argmax 直接给出被操作边的索引。这样不论图中有多少条边（1~64），都能在同一个模型中完成指向，无需按边数设计不同的分类头。

### 6.5 推理时的物理护栏机制

| 护栏 | 触发条件 | 行为 |
|------|---------|------|
| 强制步骤护栏 | step < 3 或 len(edges) > 4 | 屏蔽 Done（-1e9），强制继续工作 |
| taboo_pairs | Merge 后贝塞尔误差 > 3.0px | 记录此对，后续不再尝试合并 |
| locked_nodes | Delete 被拒（冗余度<60%）或 Merge 找不到伙伴 | 屏蔽该节点的 pointer，不再操作 |
| 全图终止 | 所有 pointer_logits < -1e8 | 提前退出循环，返回当前结果 |

这三层护栏将模型的"策略输出"和物理约束解耦：模型只负责预测意图，物理引擎负责校验可行性，失败时惩罚并继续尝试而非直接崩溃。

### 6.6 训练-推理一致性

训练时的 `x_feat` 特征提取（`data_builder.extract_state_features`）和推理时的特征提取（`ai_auto_initializer.env.extract_state_features`）**共用同一个 `GraphReplayEnvironment` 实例**，确保特征计算逻辑完全一致，不存在 train-test gap。
