# Phase 2 拓扑标注系统 — 完整技术文档

> 生成时间：2026-06-29  
> 源文件：`topo_editor_workspace.py` / `main_with_record_action_and_model.py`  
> 阶段定位：Phase 1（骨架清理）完成后，对贝塞尔曲线笔画进行拓扑关系标注与数据落盘

---

## 目录

1. [整体流程概述](#1-整体流程概述)
2. [数据入口：Phase 1 → Phase 2 的交接](#2-数据入口-phase-1--phase-2-的交接)
3. [拓扑检测数学工具函数](#3-拓扑检测数学工具函数)
4. [TopoAnnotationWorkspace 核心类](#4-topoannotationworkspace-核心类)
5. [拓扑事件实时检测引擎](#5-拓扑事件实时检测引擎)
6. [操作录像：edit_history 格式](#6-操作录像-edit_history-格式)
7. [落盘数据结构：5 层架构](#7-落盘数据结构-5-层架构)
8. [保存入口：save_phase2_topo_data（main 侧）](#8-保存入口-save_phase2_topo_datamain-侧)
9. [最终 JSON 输出完整样例](#9-最终-json-输出完整样例)
10. [关键设计决策说明](#10-关键设计决策说明)

---

## 1. 整体流程概述

```
Phase 1 完工
  edges = [{'id': int, 'path': np.ndarray [4,2]}]  ← 贝塞尔控制点
     │
     ▼ action_proceed_to_phase2()
  贝塞尔拟合（fit_bezier_basic_with_error）
  phase2_edges = [{'id': int, 'path': [[x,y],[x,y],[x,y],[x,y]]}]
     │
     ▼ TopoAnnotationWorkspace(main_ws, hex_key, char, binary, dt_map, phase2_edges)
     │
  ┌──────────────────────────────────────────────────────────────┐
  │  Phase 2 标注工作区                                           │
  │                                                              │
  │  实时拓扑检测（三类型）：                                      │
  │    E2E：端点距离 < 2.0px                                     │
  │    T：端点到对方贝塞尔曲线距离 < 2.0px                        │
  │    X：polyline 线段相交（非端点区域）                          │
  │                                                              │
  │  人工操作（拖拽控制点）→ 自动吸附 + 记录 edit_history          │
  │                                                              │
  │  ✅ FINISH TOPO → action_complete_topo()                     │
  └──────────────────────────────────────────────────────────────┘
     │
     ▼ save_phase2_topo_data(hex_key, char_bundle)
  annotations_topo/{字体名}_topo.json
  （以 U+XXXX 为 key，追加写入全局聚合文件）
```

---

## 2. 数据入口：Phase 1 → Phase 2 的交接

来源：`main_with_record_action_and_model.py` → `action_proceed_to_phase2()`

```python
def action_proceed_to_phase2(self):
    if not self.edges: return

    # ── Step 1: 保存 Phase 1 records（raw_edges + action_log）──
    hex_key = f"U+{ord(self.char):04X}"

    raw_edges_serializable = [
        {"id": e['id'], "path": e['path'].tolist() if isinstance(e['path'], np.ndarray) else e['path']}
        for e in self.edges
    ]
    current_saved_raw = self.db.meta_data.get("raw_edges", {}).get(hex_key)
    if current_saved_raw != raw_edges_serializable:
        self.db.meta_data.setdefault("raw_edges", {})[hex_key] = raw_edges_serializable
        self.db.save_data()

    if hasattr(self, 'action_log'):
        log_file = os.path.join(ACTION_LOG_DIR, f"{self.font_filename}_actions.json")
        all_logs = {}
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f: all_logs = json.load(f)
        serialized_log = [
            {"action": step["action"], "edges": [
                {"id": e['id'], "path": e['path'].tolist() if isinstance(e['path'], np.ndarray) else e['path']}
                for e in step["edges"]
            ]}
            for step in self.action_log
        ]
        if all_logs.get(hex_key) != serialized_log:
            all_logs[hex_key] = serialized_log
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(all_logs, f, ensure_ascii=False, indent=2)

    # ── Step 2: 对每条边进行贝塞尔拟合，生成 phase2_edges ──
    # phase2_edges 中每条边的 'path' 为 4×2 贝塞尔控制点列表（非像素路径）
    phase2_edges = []
    for edge in self.edges:
        eid = edge['id']
        if eid not in self.bezier_cache:
            path_arr = np.array(edge['path'])
            if len(path_arr) == 4:
                p_opt = path_arr                         # 已是 4 控制点，直接用
            else:
                p_opt, _ = fit_bezier_basic_with_error(path_arr)  # 从像素路径拟合
            self.bezier_cache[eid] = (p_opt, None)
        p_opt, _ = self.bezier_cache[eid]
        phase2_edges.append({'id': eid, 'path': p_opt.tolist()})

    # ── Step 3: 实例化 Phase 2 工作区 ──
    self.topo_widget = TopoAnnotationWorkspace(
        self, hex_key, self.char, self.binary, self.dt_map, phase2_edges
    )
    self.inner_stack.addWidget(self.topo_widget)
    self.inner_stack.setCurrentIndex(1)
```

**Phase 2 接收的 `phase1_edges` 格式：**
```python
[
    {
        'id': 0,
        'path': [[x0,y0], [cx1,cy1], [cx2,cy2], [x3,y3]]  # 4 个贝塞尔控制点
    },
    ...
]
```

---

## 3. 拓扑检测数学工具函数

来源：`topo_editor_workspace.py` 顶层函数

```python
import numpy as np

TOPO_SAMPLE_N = 120    # 贝塞尔曲线离散采样点数（越多精度越高，E2E/T 检测）
X_ENDPOINT_MARGIN = 2  # X 型检测时两端忽略的采样点数（避免端点误检为 X）
SEG_EPS = 1e-8


# ────────────────────────────────────────────────
# 贝塞尔曲线切线导数（用于计算交叉角度）
# ────────────────────────────────────────────────
def get_bezier_derivative(pts, t):
    """
    计算三次贝塞尔曲线在参数 t 处的切线向量。
    pts: [4, 2]  四个控制点
    t:   float   参数 ∈ [0, 1]
    返回: [2,] 切线向量（未归一化）
    """
    mt = 1 - t
    d = 3*mt**2*(pts[1]-pts[0]) + 6*mt*t*(pts[2]-pts[1]) + 3*t**2*(pts[3]-pts[2])
    return d


def get_angle(v1, v2):
    """
    计算两向量之间的夹角（度数，范围 [0°, 180°]）。
    用于 T 型和 X 型拓扑事件的 angle 字段。
    """
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-5 or n2 < 1e-5: return 0.0
    cos_th = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_th)))


def get_polygon_orientation(pts):
    """
    计算多边形的绕向（基于带符号面积）。
    返回: "ccw"（逆时针）或 "cw"（顺时针）
    用于 Layer 4 闭环方向标注。
    """
    area = 0.0
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        area += (pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1])
    return "ccw" if area > 0 else "cw"


# ────────────────────────────────────────────────
# X 型交叉检测：polyline 线段相交（比采样点距离更稳定）
# ────────────────────────────────────────────────
def _cross2d(a, b):
    """2D 叉积（标量）"""
    return float(a[0] * b[1] - a[1] * b[0])


def segment_intersection(p0, p1, q0, q1, eps=SEG_EPS):
    """
    判断线段 p0-p1 与 q0-q1 是否相交。
    返回:
        None                              不相交
        (point, t_on_p, t_on_q)          相交点坐标 + 两段的局部参数
    
    使用参数化方程：
        P(t) = p0 + t*(p1-p0)
        Q(u) = q0 + u*(q1-q0)
        解: t = cross(q0-p0, s) / cross(r, s)
            u = cross(q0-p0, r) / cross(r, s)
    """
    r = p1 - p0
    s = q1 - q0
    denom = _cross2d(r, s)
    if abs(denom) < eps:   # 平行，不算 X
        return None
    qp = q0 - p0
    t = _cross2d(qp, s) / denom
    u = _cross2d(qp, r) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return p0 + t * r, float(t), float(u)
    return None


def find_polyline_x_intersection(c1, c2, endpoint_margin=X_ENDPOINT_MARGIN):
    """
    在两条离散 polyline 之间寻找真正的线段交点（X 型拓扑）。
    
    参数：
        c1, c2:          np.ndarray [N, 2]   两条曲线的离散采样点
        endpoint_margin: int                  两端忽略的点数（避免端点误检）
    
    返回：
        (hit:bool, point:[x,y]|None, t1:float|None, t2:float|None)
        t1/t2 是相交位置在各自曲线上的近似贝塞尔参数 ∈ [0, 1]
    
    设计目的：
        旧方法用"两组采样点之间最近点距离 < 2.0px"，当交点恰好落在两个采样点
        之间的间隙时会漏检，拖拽时产生闪烁。新方法用"相邻采样点构成的线段做
        几何相交测试"，结果稳定，与采样点分布无关。
    """
    n1, n2 = len(c1), len(c2)
    if n1 < 2 or n2 < 2:
        return False, None, None, None

    i_start = max(0, endpoint_margin)
    i_end   = max(i_start, n1 - 1 - endpoint_margin)
    j_start = max(0, endpoint_margin)
    j_end   = max(j_start, n2 - 1 - endpoint_margin)

    best = None
    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            hit = segment_intersection(c1[i], c1[i+1], c2[j], c2[j+1])
            if hit is None: continue
            pt, lt1, lt2 = hit
            t1 = (i + lt1) / (n1 - 1)
            t2 = (j + lt2) / (n2 - 1)
            # 只接受内部交叉（t ∈ [0.02, 0.98]），端点碰触由 E2E/T 处理
            if t1 <= 0.02 or t1 >= 0.98 or t2 <= 0.02 or t2 >= 0.98:
                continue
            best = (pt, t1, t2)
            break
        if best is not None: break

    if best is None:
        return False, None, None, None
    pt, t1, t2 = best
    return True, pt, float(t1), float(t2)
```

---

## 4. TopoAnnotationWorkspace 核心类

来源：`topo_editor_workspace.py` — `class TopoAnnotationWorkspace`

### 4.1 初始化（`__init__`）

```python
class TopoAnnotationWorkspace(QWidget):
    def __init__(self, main_workspace, hex_key, char, binary, dt_map, phase1_edges):
        """
        参数：
            main_workspace: AnnotationWorkspace   Phase 1 主工作区（用于调用 save_phase2_topo_data）
            hex_key:        str                   如 "U+4E00"
            char:           str                   Unicode 字符，如 "一"
            binary:         np.ndarray [H, W]     字形二值掩码（来自字体渲染）
            dt_map:         np.ndarray [H, W]     距离变换图（用于宽度回归）
            phase1_edges:   list                  贝塞尔控制点边列表（4控制点格式）
        """
        self.main_ws = main_workspace
        self.hex_key = hex_key
        self.char = char
        self.binary = binary
        self.dt_map = dt_map

        self.initial_edges = copy.deepcopy(phase1_edges)  # 重置基准
        self.edges = copy.deepcopy(phase1_edges)          # 当前工作副本

        self.history_stack = []        # Undo 栈（最多 30 帧）
        self.selected_edge_ids = []    # 当前选中的边 ID 列表

        # 拖拽状态
        self.dragging_point = None     # (edge_idx, point_idx) 当前被拖拽的控制点
        self.co_dragged_points = []    # 所有与被拖拽点共位的控制点（端点对接时一起移动）
        self.t_constraints = []        # T 型搭接约束：[(o_idx, ep_idx, h_idx, min_t)]
        self.width_cache = {}          # 宽度缓存 {edge_id: w_bezier[4,]}

        # Phase 2 操作录像
        self.edit_history = []         # 落盘用，记录每次拖拽的语义操作
        self.drag_start_pos = None     # 拖拽起始控制点坐标（用于判断是否实质移动）
```

### 4.2 宽度回归（`force_recompute_all_widths`）

```python
def force_recompute_all_widths(self):
    """
    对所有边重新计算宽度贝塞尔参数 w_opt [4,]。
    依赖 regress_width_dt_fast(path, dt_map)：
      - 在路径上均匀采样点，从距离变换图读取各点的局部宽度
      - 拟合成 3 次贝塞尔宽度函数的 4 个控制值
    结果存入 self.width_cache[edge_id]，用于 Layer 1 数据落盘和画布渲染。
    """
    for edge in self.edges:
        self.width_cache[edge['id']] = regress_width_dt_fast(
            np.array(edge['path']), self.dt_map
        )
```

---

## 5. 拓扑事件实时检测引擎

### 5.1 实时反馈（`update_topology_text`）

在每次拖拽释放、撤销、重置后触发，实时扫描所有边对，检测三种拓扑关系并渲染到文本框。

**检测算法流程（双重循环 `i < j`）：**

```
for each pair (e1, e2), i < j:
    c1 = cubicBezier(e1.path, 120采样点)
    c2 = cubicBezier(e2.path, 120采样点)

    ── 检测 1: E2E（端点对接）──
    for pt1 in [P0, P3]:
        for pt2 in [P0, P3]:
            if dist(pt1, pt2) < 2.0px → E2E

    ── 检测 2: T 型搭接（主客体判定）──
    if NOT E2E:
        for pt1 in [P0, P3]:  # e1 的端点搭在 e2 上
            if min_dist(pt1, c2_samples) < 2.0px → T (e1 is guest, e2 is host)
        for pt2 in [P0, P3]:  # e2 的端点搭在 e1 上
            if min_dist(pt2, c1_samples) < 2.0px → T (e2 is guest, e1 is host)

    ── 检测 3: X 型交叉 ──
    if NOT E2E and NOT T:
        find_polyline_x_intersection(c1, c2) → 线段相交法
        if hit → X

    ── 闭环检测 ──
    构建图 G，用 networkx:
    1. 2-Stroke 闭环：同一对 (u,v) 有 ≥2 个相距 > 5px 的碰撞点（如字母 'o'）
    2. ≥3-Stroke 闭环：cycle_basis(G) 并验证相邻交点距离 > 5px
```

### 5.2 落盘时的精确检测（`action_complete_topo`）

落盘时复用同样的检测逻辑，但额外计算 **精确 t 值** 和 **夹角**：

```python
# ── E2E: 记录精确 t 值（端点 → 0.0 or 1.0）──
for pt1_idx, t1 in [(0, 0.0), (3, 1.0)]:
    for pt2_idx, t2 in [(0, 0.0), (3, 1.0)]:
        if dist(p1[pt1_idx], p2[pt2_idx]) < 2.0:
            → {"type":"E2E", "stroke_a":id1, "t_a":t1, "stroke_b":id2, "t_b":t2, ...}

# ── T: 记录 host 上的精确 t 值 + 夹角 ──
for pt1_idx, t1 in [(0, 0.0), (3, 1.0)]:
    m_idx = argmin_dist(p1[pt1_idx], c2_samples)   # 120 采样点中最近的
    t2 = m_idx / 119.0                             # 换算为 [0,1] 参数
    ang = get_angle(bezier_deriv(p1, t1), bezier_deriv(p2, t2))
    → {"type":"T", "guest":id1, "guest_t":t1, "host":id2, "host_t":t2, "angle":ang, ...}

# ── X: polyline 相交法 → 精确 t1, t2 + 夹角 ──
hit, x_pt, t1, t2 = find_polyline_x_intersection(c1, c2)
ang = get_angle(bezier_deriv(p1, t1), bezier_deriv(p2, t2))
→ {"type":"X", "stroke_a":id1, "t_a":t1, "stroke_b":id2, "t_b":t2, "angle":ang, ...}
```

---

## 6. 操作录像：edit_history 格式

来源：`topo_editor_workspace.py` → `on_mouse_release()`

每次拖拽控制点完成后，根据是否发生吸附，记录三种语义操作：

### 6.1 端点对端点吸附（SNAP）

触发条件：鼠标释放时最近他端点距离 < 12px 且用户确认

```python
{
    "action": "SNAP",
    "stroke": 2,              # guest 笔画的显示编号（1-indexed）
    "endpoint": "P0",         # guest 端点：P0 或 P3
    "host_stroke": 1,         # host 笔画的显示编号
    "host_endpoint": "P3",    # host 端点：P0 或 P3
    "before": [45.2, 123.8],  # 拖拽前坐标 [x, y]
    "after": [46.0, 124.5]    # 拖拽后坐标（吸附点坐标）
}
```

### 6.2 T 型端点吸附到曲线（T_ATTACH）

触发条件：鼠标释放时最近曲线距离 < 12px 且用户确认（且远于端点吸附距离）

```python
{
    "action": "T_ATTACH",
    "guest": 2,               # 端点所在笔画的显示编号
    "guest_endpoint": "P0",   # 该端点：P0 或 P3
    "host": 1,                # 被搭接的主干笔画编号
    "host_t": 0.483,          # 搭接点在 host 曲线上的参数 t ∈ [0,1]
    "before": [45.2, 123.8],
    "after": [46.0, 124.5]
}
```

### 6.3 自由控制点移动（CONTROL_MOVE）

触发条件：未发生吸附，但控制点实质性移动（位移 > 0.5px）

```python
{
    "action": "CONTROL_MOVE",
    "stroke": 2,              # 所在笔画的显示编号
    "control": "P1",          # 控制点：P0/P1/P2/P3
    "before": [45.2, 123.8],
    "after": [52.1, 119.3]
}
```

### 6.4 吸附判定的优先级逻辑

```
鼠标释放 on_mouse_release():
    如果操作的是端点（P0 或 P3）：
        ├── 计算 min_ep_dist  = 与所有他端点的最近距离
        └── 计算 min_curve_dist = 与所有曲线的最近距离
    
    判断顺序（优先级）：
        1. min_ep_dist < 12px AND 用户确认 → SNAP（端点吸附端点）
        2. elif min_curve_dist < 12px AND 用户确认 → T_ATTACH（T型搭接）
        3. else → CONTROL_MOVE（自由移动）
    
    如果操作的是内部控制点（P1 或 P2）：
        → 始终记录 CONTROL_MOVE
```

### 6.5 共位点联动机制（co_dragged_points）

拖拽端点时，自动检测并联动所有与该端点**坐标完全重合（距离 < 1px）**的其他边端点，实现"已对接的端点同步移动"：

```python
if p_idx in [0, 3]:  # 拖拽的是端点
    target_pos = self.edges[e_idx]['path'][p_idx]
    for j, edge in enumerate(self.edges):
        for ep_idx in [0, 3]:
            if dist(edge['path'][ep_idx], target_pos) < 1.0:
                self.co_dragged_points.append((j, ep_idx))
```

### 6.6 T 型约束传播（t_constraints）

拖拽端点时，如果该端点的宿主曲线上有其他曲线的端点"搭接"其上，则这些搭接点在拖拽过程中自动跟随宿主曲线移动：

```python
# 扫描：哪些边的端点正搭接在 co_dragged_points 所在边的曲线上
for h_idx in host_curves:
    c_host = cubicBezier(self.edges[h_idx]['path'], 120点)
    for o_idx, other_edge in enumerate(self.edges):
        for ep_idx in [0, 3]:
            ep = other_edge['path'][ep_idx]
            dists = norm(c_host - ep)
            if min(dists) < 1.5px:
                t_constraints.append((o_idx, ep_idx, h_idx, min_t_idx))

# 在 on_mouse_move 中持续更新 T 约束点位置
for o_idx, ep_idx, h_idx, min_t in t_constraints:
    c_host_updated = cubicBezier(self.edges[h_idx]['path'], 120点)
    self.edges[o_idx]['path'][ep_idx] = c_host_updated[min_t]
```

---

## 7. 落盘数据结构：5 层架构

来源：`topo_editor_workspace.py` → `action_complete_topo()`

按下 **✅ FINISH TOPO (Enter)** 后，组装并落盘一个 `char_bundle` 字典：

```python
char_bundle = {
    "glyph_info":       {...},   # Layer 0: 字符元信息
    "strokes":          [...],   # Layer 1: 笔画几何与派生特征
    "topology_events":  [...],   # Layer 2+3: 拓扑事件表
    "cycles":           [...],   # Layer 4: 闭环结构
    "edit_history":     [...]    # Layer 5: 操作录像
}
```

### Layer 0：字符元信息（glyph_info）

```python
{
    "hex_key": "U+4E00",
    "char": "一"
}
```

### Layer 1：笔画几何与派生特征（strokes）

```python
[
    {
        "bezier_id": 1,                    # 显示编号（1-indexed，与 edit_history 对应）
        "stroke_type": "open",             # "open" 或 "closed"（P0 与 P3 距离 < 2px 为 closed）
        "length": 183.45,                  # 贝塞尔曲线弧长（50 采样点梯形积分）
        "bbox": [10.5, 20.3, 180.2, 45.1], # [xmin, ymin, xmax, ymax]
        "mother_bezier": [                 # 形状贝塞尔：4 个控制点坐标
            [x0, y0],
            [cx1, cy1],
            [cx2, cy2],
            [x3, y3]
        ],
        "width_bezier": [w0, w1, w2, w3]  # 宽度贝塞尔：4 个宽度控制值（像素）
    },
    ...
]
```

**width_bezier 的来源：**  
`regress_width_dt_fast(path, dt_map)` — 在曲线上均匀采样，从距离变换图（`distance_transform_edt(binary)`）读取各点的局部半宽度，拟合为 3 次贝塞尔宽度函数的 4 个控制值。

### Layer 2+3：拓扑事件表（topology_events）

三种事件类型混存于同一数组：

#### E2E（端点对接）
```python
{
    "type": "E2E",
    "stroke_a": 1,       # 笔画 A 的 bezier_id
    "t_a": 1.0,          # A 上的参数 t（0.0=P0端, 1.0=P3端）
    "stroke_b": 2,       # 笔画 B 的 bezier_id
    "t_b": 0.0,          # B 上的参数 t
    "position": [46.0, 124.5]  # 接合点物理坐标 [x, y]
}
```

#### T（T 型搭接）
```python
{
    "type": "T",
    "guest": 2,          # 搭接者（端点在 host 上）的 bezier_id
    "guest_t": 0.0,      # guest 搭接的是哪个端点（0.0=P0端, 1.0=P3端）
    "host": 1,           # 被搭接的主干 bezier_id
    "host_t": 0.483,     # 搭接点在 host 曲线上的参数 t ∈ [0,1]
    "angle": 89.5,       # 两曲线在交叉点处的夹角（度数）
    "position": [46.0, 124.5]
}
```

#### X（十字交叉）
```python
{
    "type": "X",
    "stroke_a": 1,       # 笔画 A 的 bezier_id
    "t_a": 0.523,        # 交叉点在 A 上的参数 t ∈ [0,1]
    "stroke_b": 3,       # 笔画 B 的 bezier_id
    "t_b": 0.481,        # 交叉点在 B 上的参数 t ∈ [0,1]
    "angle": 92.3,       # 两曲线在交叉点处的夹角（度数）
    "position": [200.1, 150.3]
}
```

### Layer 4：闭环结构（cycles）

```python
[
    {
        "cycle_id": 0,
        "members": [1, 2],        # 参与构成闭环的笔画 bezier_id 列表
        "orientation": "ccw"      # 闭环绕向："ccw"（逆时针）或 "cw"（顺时针）
    },
    {
        "cycle_id": 1,
        "members": [3, 4, 5],
        "orientation": "cw"
    }
]
```

**闭环检测逻辑：**
- **2-Stroke 闭环**：两条笔画之间存在 ≥2 个物理碰撞点，且两点距离 ≥ 5px（如字母 "o"，两半圆各自的起/终端点对接两次）
- **≥3-Stroke 闭环**：`networkx.cycle_basis(G)` 在拓扑图上提取环，并验证环上相邻交点物理距离 ≥ 5px（排除多重连接的伪环）

### Layer 5：操作录像（edit_history）

见第 6 节，格式为 `SNAP / T_ATTACH / CONTROL_MOVE` 的操作列表。

---

## 8. 保存入口：save_phase2_topo_data（main 侧）

来源：`main_with_record_action_and_model.py` → `AnnotationWorkspace.save_phase2_topo_data()`

```python
def save_phase2_topo_data(self, hex_key, char_bundle):
    """
    将 Phase 2 完工的字形数据追加写入聚合文件。
    
    文件路径: annotations_topo/{字体名}_topo.json
    写入方式: 以 U+XXXX 为 key 追加到全局字典（不覆盖其他字符数据）
    同步更新: self.db.annotated_outlines 局部索引数据库
    """
    topo_file = os.path.join(TOPO_OUT_DIR, f"{self.font_filename}_topo.json")
    all_topo_data = {}

    # 读取已有数据（append 模式）
    if os.path.exists(topo_file):
        try:
            with open(topo_file, 'r', encoding='utf-8') as f:
                all_topo_data = json.load(f)
        except: pass

    # 写入当前字符
    all_topo_data[hex_key] = char_bundle

    with open(topo_file, 'w', encoding='utf-8') as f:
        json.dump(all_topo_data, f, ensure_ascii=False, indent=2)

    # 同步更新局部快速索引（用于进度统计）
    self.db.annotated_outlines[hex_key] = char_bundle["strokes"]
    print(f"✅ Topo & Geometry Annotation for {hex_key} aggregated successfully!")

    # 刷新 UI 进度统计 + 自动跳下一个字符
    self.update_stats_display()
    self.inner_stack.setCurrentIndex(0)
    self.action_next_char()
```

**文件目录结构：**
```
AI_VECTOR_ROUTER_With_topo/
├── action_logs/                       ← Phase 1 操作录像
│   └── {字体名}_actions.json
└── annotations_topo/                  ← Phase 2 拓扑标注产出
    └── {字体名}_topo.json             ← 按 U+XXXX 键组织的聚合文件
```

---

## 9. 最终 JSON 输出完整样例

`annotations_topo/微软雅黑_topo.json`:

```json
{
  "U+4E00": {
    "glyph_info": {
      "hex_key": "U+4E00",
      "char": "一"
    },
    "strokes": [
      {
        "bezier_id": 1,
        "stroke_type": "open",
        "length": 183.45,
        "bbox": [10.5, 85.2, 350.8, 110.3],
        "mother_bezier": [
          [12.0, 95.0], [80.0, 88.0], [280.0, 88.0], [350.0, 95.0]
        ],
        "width_bezier": [6.2, 5.8, 5.9, 6.3]
      }
    ],
    "topology_events": [],
    "cycles": [],
    "edit_history": [
      {
        "action": "CONTROL_MOVE",
        "stroke": 1,
        "control": "P1",
        "before": [78.5, 90.3],
        "after": [80.0, 88.0]
      }
    ]
  },
  "U+4E8C": {
    "glyph_info": {"hex_key": "U+4E8C", "char": "二"},
    "strokes": [
      {
        "bezier_id": 1,
        "stroke_type": "open",
        "length": 120.3,
        "bbox": [40.0, 80.0, 320.0, 100.0],
        "mother_bezier": [[40.0, 90.0], [130.0, 85.0], [230.0, 85.0], [320.0, 90.0]],
        "width_bezier": [4.5, 4.2, 4.3, 4.5]
      },
      {
        "bezier_id": 2,
        "stroke_type": "open",
        "length": 200.1,
        "bbox": [10.0, 160.0, 360.0, 185.0],
        "mother_bezier": [[10.0, 170.0], [110.0, 162.0], [260.0, 162.0], [360.0, 170.0]],
        "width_bezier": [7.0, 6.5, 6.6, 7.0]
      }
    ],
    "topology_events": [],
    "cycles": [],
    "edit_history": []
  },
  "U+4E09": {
    "glyph_info": {"hex_key": "U+4E09", "char": "三"},
    "strokes": ["...（3条笔画）"],
    "topology_events": [],
    "cycles": [],
    "edit_history": []
  }
}
```

**含 T 型关系的样例（如"丁"字）：**
```json
{
  "U+4E01": {
    "glyph_info": {"hex_key": "U+4E01", "char": "丁"},
    "strokes": [
      {"bezier_id": 1, "stroke_type": "open", "length": 200.0, "...": "横画"},
      {"bezier_id": 2, "stroke_type": "open", "length": 150.0, "...": "竖画"}
    ],
    "topology_events": [
      {
        "type": "T",
        "guest": 2,
        "guest_t": 0.0,
        "host": 1,
        "host_t": 0.512,
        "angle": 89.8,
        "position": [200.3, 95.1]
      }
    ],
    "cycles": [],
    "edit_history": [
      {
        "action": "T_ATTACH",
        "guest": 2,
        "guest_endpoint": "P0",
        "host": 1,
        "host_t": 0.512,
        "before": [198.5, 92.0],
        "after": [200.3, 95.1]
      }
    ]
  }
}
```

---

## 10. 关键设计决策说明

### 10.1 为什么 X 型检测改用线段相交而非采样点距离

旧方法（50 采样点最近距离 < 2px）的缺陷：当两条曲线的真实交叉点恰好落在相邻两个采样点之间的"间隙"时，最近点距离可能 > 2px，导致漏检。用户拖拽控制点时，检测结果会随采样点错开而时好时坏（闪烁）。

新方法：将 120 个采样点连成折线段，对每对线段做几何相交测试，只要两曲线确实相交，必然有某对线段相交，与采样点分布无关。`endpoint_margin=2` 排除端点附近的接触，防止端点碰触被误检为 X。

### 10.2 为什么宽度用贝塞尔参数化而非逐点存储

笔画宽度沿曲线方向通常是平滑渐变的（起笔细、行笔匀、收笔细或反之）。用 4 个贝塞尔控制值 `[w0, w1, w2, w3]` 就能精确描述这种渐变规律，存储极为紧凑，且与 `mother_bezier` 的参数化坐标系完全对齐（同参数 t 对应同位置的宽度）。

### 10.3 t 值的语义

| 字段 | 含义 | 范围 |
|------|------|------|
| `t_a = 0.0` | 笔画 A 的 P0 端（起点） | 精确值 |
| `t_a = 1.0` | 笔画 A 的 P3 端（终点） | 精确值 |
| `host_t = 0.483` | T 型时 guest 端点吸附在 host 的 t=0.483 处 | 120点归一化近似值 |
| `t_a = 0.523` | X 型时交叉点在 A 的参数位置 | polyline相交法近似值 |

### 10.4 edit_history 与 topology_events 的关系

`edit_history` 记录人类**操作意图**（"我把端点 P0 吸附到了笔画 1 的中间"），是行为克隆训练的原始数据。`topology_events` 记录**当前状态的拓扑事实**（"笔画 1 和 2 在 t=0.512 处发生 T 型搭接"），是几何/图神经网络的训练标签。两者互补：前者描述如何到达这个状态，后者描述这个状态是什么。

### 10.5 bezier_id（1-indexed）vs edge['id']（全局 ID）

在整个标注系统中，`edge['id']` 是全生命周期唯一的整数 ID（Phase 1 的 Merge/Delete 不重用 ID）。`bezier_id` 是当前 Phase 2 状态下的局部显示序号（1, 2, 3...），在 `get_display_map()` 中按当前 `self.edges` 的顺序重新分配。`edit_history` 使用 `bezier_id`（显示序号），`topology_events` 也使用 `bezier_id`，两者在同一字符的 `char_bundle` 内完全对应。
