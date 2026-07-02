# PCG 标注飞轮：Style Modes 与 Topology Families 说明

## 0. Good / Bad 保存格式是否与第二阶段人工标注格式一致？

当前 `annotation_flywheel_pcg_app_no_json_v4_filebacked.py` 的保存逻辑是：

```python
out_hex, bundle = candidate_to_char_bundle(c, label, hex_key)
```

其中 `candidate_to_char_bundle()` 会优先尝试复用上一层目录里的旧人工标注脚本：

```python
annotation_flywheel_app_fixed.py
```

并优先调用其中的：

```python
candidate_to_char_bundle()
candidate_to_strokes()
```

所以结论是：

### 情况 A：成功导入 `annotation_flywheel_app_fixed.py`

Good / Bad 保存出来的 JSON **主体格式应当与第二阶段保存格式一致**。

它仍然是类似：

```json
{
  "E000": {
    "glyph_info": {},
    "strokes": [],
    "topology": {},
    "cycles": [],
    "edit_history": []
  }
}
```

区别是 PCG 飞轮会额外补充一些字段，例如：

```json
{
  "label": "good / bad",
  "style_mode": "s_curve_left",
  "topology_family": "rune_cross",
  "pcg_meta": {}
}
```

这些字段通常会写入：

```text
glyph_info
edit_history
```

因此更准确地说是：

```text
第二阶段格式 + PCG 生成与人工筛选元信息
```

这对后续训练有益，因为可以知道样本来自哪个生成器、哪个拓扑族、哪个 style mode，以及人工标注结果。

### 情况 B：没有成功导入第二阶段人工标注脚本

脚本会使用 fallback converter。fallback 会尽量模拟 `annotations_topo` 的结构，但不能保证和第二阶段代码 **100% 字段级一致**。

fallback 主要保证这些核心字段：

```json
{
  "glyph_info": {
    "char": "...",
    "unicode_hex": "E000",
    "font_path": "...",
    "font_name": "...",
    "label": "good",
    "candidate_id": "...",
    "style_mode": "...",
    "topology_family": "...",
    "complexity": {}
  },
  "strokes": [
    {
      "bezier_id": 0,
      "mother_bezier": [[x0,y0], [x1,y1], [x2,y2], [x3,y3]],
      "width": 10.0,
      "width_norm": 0.025,
      "alpha": 1.0,
      "exist": true
    }
  ],
  "topology": {
    "connections": [],
    "positive_edges_undirected": [],
    "t_junctions": [],
    "cycles": [],
    "anchors": []
  },
  "cycles": [],
  "edit_history": [],
  "metadata": {}
}
```

正式实验前建议检查控制台是否出现：

```text
[Reuse] imported annotation_flywheel_app_fixed
```

如果出现，说明当前保存路径基本复用了第二阶段人工标注格式。

---

# 1. Style Modes

Style mode 控制的是**单条 stroke 的 Bézier 控制点形状**。

一条 stroke 的母线是 cubic Bézier：

```text
P0 —— P1 —— P2 —— P3
```

其中：

```text
P0 / P3 = 两端 anchor，决定拓扑连接，不轻易移动
P1 / P2 = 控制点，决定弯曲形态
width   = 笔画宽度
```

生成器的基本思想是：

```python
p0 = anchor_start
p3 = anchor_end

ex = normalize(p3 - p0)
ey = perpendicular(ex)

p1 = p0 + x1 * L * ex + y1 * L * ey
p2 = p0 + x2 * L * ex + y2 * L * ey
```

其中：

```text
L  = |p3 - p0|
ex = 端点方向
ey = 垂直方向
x1/x2 = 沿 stroke 前进方向的位置
y1/y2 = 相对 stroke 主方向的偏移
```

也就是说，style mode 本质上是在控制：

```text
P1 / P2 相对于 P0→P3 主轴的横向偏移
```

---

## 1.1 `straight`

### 物理意义

近似直线笔画。

适合：

```text
直线符文
横竖结构
几何字体
工业风 UI 图标
```

### 生成方式

```python
x1 = 0.33
y1 = near 0

x2 = 0.67
y2 = near 0
```

### 简单代码

```python
if mode == "straight":
    return 0.33, rng.uniform(-0.02, 0.02), 0.67, rng.uniform(-0.02, 0.02)
```

---

## 1.2 `mild_left`

### 物理意义

轻微向左侧弯曲。

这里的 left 不是屏幕绝对左，而是相对于 stroke 方向 `P0 → P3` 的左法线方向。

适合：

```text
柔和符文
生物感线条
自然弯曲笔画
```

### 生成方式

```python
y1 > 0
y2 > 0
```

### 简单代码

```python
if mode == "mild_left":
    a = rng.uniform(0.10, 0.20)
    return rng.uniform(0.25, 0.38), +a, rng.uniform(0.58, 0.75), +a * rng.uniform(0.75, 1.15)
```

---

## 1.3 `mild_right`

### 物理意义

轻微向右侧弯曲。

适合：

```text
与 mild_left 成对生成
制造左右方向变化
让同一拓扑产生不同视觉性格
```

### 生成方式

```python
y1 < 0
y2 < 0
```

### 简单代码

```python
if mode == "mild_right":
    a = rng.uniform(0.10, 0.20)
    return rng.uniform(0.25, 0.38), -a, rng.uniform(0.58, 0.75), -a * rng.uniform(0.75, 1.15)
```

---

## 1.4 `strong_left`

### 物理意义

强烈向左弯曲。

适合：

```text
夸张符文
魔法阵感线条
装饰性强的曲线 glyph
```

注意：过强时可能导致结构变软、交叉增加或可读性下降。

### 生成方式

```python
y1, y2 比 mild_left 更大
```

### 简单代码

```python
if mode == "strong_left":
    a = rng.uniform(0.22, 0.38)
    return rng.uniform(0.20, 0.35), +a, rng.uniform(0.60, 0.82), +a * rng.uniform(0.70, 1.20)
```

---

## 1.5 `strong_right`

### 物理意义

强烈向右弯曲。

适合：

```text
夸张符号
异形符文
高装饰度 UI 符号
```

### 简单代码

```python
if mode == "strong_right":
    a = rng.uniform(0.22, 0.38)
    return rng.uniform(0.20, 0.35), -a, rng.uniform(0.60, 0.82), -a * rng.uniform(0.70, 1.20)
```

---

## 1.6 `s_curve_left`

### 物理意义

S 型曲线：前半段向左，后半段向右。

适合：

```text
蛇形符文
能量流动感
魔法文字
生物/藤蔓风格字符
```

### 生成方式

```python
y1 > 0
y2 < 0
```

### 简单代码

```python
if mode == "s_curve_left":
    a = rng.uniform(0.18, 0.34)
    return rng.uniform(0.22, 0.38), +a, rng.uniform(0.58, 0.78), -a * rng.uniform(0.70, 1.15)
```

---

## 1.7 `s_curve_right`

### 物理意义

反向 S 型曲线：前半段向右，后半段向左。

### 生成方式

```python
y1 < 0
y2 > 0
```

### 简单代码

```python
if mode == "s_curve_right":
    a = rng.uniform(0.18, 0.34)
    return rng.uniform(0.22, 0.38), -a, rng.uniform(0.58, 0.78), +a * rng.uniform(0.70, 1.15)
```

---

## 1.8 `hook_left`

### 物理意义

钩状曲线，起笔附近有较大偏移，末端逐渐回到主轴。

适合：

```text
爪形符文
哥特/魔法符号
装饰性端部
```

### 生成方式

```python
P1 横向偏移大
P2 横向偏移小
```

### 简单代码

```python
if mode == "hook_left":
    return rng.uniform(0.14, 0.26), rng.uniform(0.28, 0.48), rng.uniform(0.55, 0.80), rng.uniform(0.02, 0.14)
```

---

## 1.9 `hook_right`

### 物理意义

反向钩状曲线。

### 简单代码

```python
if mode == "hook_right":
    return rng.uniform(0.14, 0.26), -rng.uniform(0.28, 0.48), rng.uniform(0.55, 0.80), -rng.uniform(0.02, 0.14)
```

---

## 1.10 `mixed`

### 物理意义

每条 stroke 可以随机选择不同 style mode。

适合：

```text
高多样性符号
异星文字
非规则 glyph
早期探索
```

### 简单代码

```python
if mode == "mixed":
    return local_style_params(rng.choice(STYLE_MODES[:-1]), rng)
```

---

# 2. Topology Families

Topology family 控制的是**一个字符由哪些 anchor 和 stroke 构成**。

基本元素是：

```text
anchor = 笔画端点 / 连接点
stroke = 两个 anchor 之间的一条 Bézier 曲线
```

生成器先构造 anchor graph：

```python
anchors = []
edges = [(anchor_a, anchor_b), ...]
```

然后把每条 edge 转换为一条 Bézier stroke：

```python
for anchor_a, anchor_b in edges:
    p0 = anchors[anchor_a]
    p3 = anchors[anchor_b]
    bezier = bezier_from_anchors(p0, p3, style_mode)
```

---

## 2.1 `chain`

### 物理意义

链式结构。像一条连续的折线或藤蔓。

适合：

```text
笔画连续的文字
蛇形符号
路径型 glyph
```

### 生成方式

```text
A0 - A1 - A2 - A3 - ...
```

### 简单代码

```python
pts = [start_point]
for i in range(stroke_count):
    direction = rotate(direction, rng.uniform(-0.95, 0.95))
    pts.append(pts[-1] + direction * step)

edges = [(i, i + 1) for i in range(stroke_count)]
```

---

## 2.2 `fork`

### 物理意义

分叉结构。像树枝、鹿角、符文分叉。

适合：

```text
自然系符文
魔法文字
技能图标
```

### 生成方式

```text
      A2
      |
A1 -- O -- A3
      |
      A4
```

### 简单代码

```python
root = center_anchor
for i in range(branch_n):
    tip = point_on_circle(i)
    edges.append((root, tip))

while len(edges) < stroke_count:
    a = random_existing_tip
    b = new_tip_near_a
    edges.append((a, b))
```

---

## 2.3 `star`

### 物理意义

星状放射结构。所有笔画从中心向外发散。

适合：

```text
星芒符号
技能图标
魔法印记
```

### 生成方式

```text
       A1
        |
A2 ---- O ---- A3
        |
       A4
```

### 简单代码

```python
root = center_anchor
for i in range(stroke_count):
    angle = 2 * pi * i / stroke_count
    tip = center + radius * [cos(angle), sin(angle)]
    edges.append((root, tip))
```

---

## 2.4 `zigzag`

### 物理意义

锯齿结构。

适合：

```text
闪电符号
攻击图标
古代刻痕感文字
```

### 生成方式

```text
A0 / A1 \ A2 / A3 \ A4
```

### 简单代码

```python
for i in range(stroke_count + 1):
    x = -0.8 + 1.6 * i / stroke_count
    y = 0.38 if i % 2 == 0 else -0.38
    pts.append(P(x, y))

edges = [(i, i + 1) for i in range(stroke_count)]
```

---

## 2.5 `triangle`

### 物理意义

三角环结构。

适合：

```text
魔法阵符号
警示符号
神秘几何符号
```

### 生成方式

```text
A0 -- A1
 \    /
  A2
```

### 简单代码

```python
ids = [A0, A1, A2]
edges = [
    (A0, A1),
    (A1, A2),
    (A2, A0),
]
```

如果笔画数更多，则从三角顶点继续长出尾巴：

```python
while len(edges) < stroke_count:
    a = random_triangle_vertex
    b = new_anchor_near_a
    edges.append((a, b))
```

---

## 2.6 `box`

### 物理意义

方框 / 四边形结构。

适合：

```text
封印符号
UI 方形 glyph
古文明铭文
```

### 生成方式

```text
A3 ---- A2
|       |
A0 ---- A1
```

### 简单代码

```python
ids = [bottom_left, bottom_right, top_right, top_left]
edges = [
    (A0, A1),
    (A1, A2),
    (A2, A3),
    (A3, A0),
]
```

可以额外添加 chord 或 tail：

```python
if rng.random() < 0.45:
    edges.append((random_corner, another_corner))
else:
    edges.append((random_corner, new_anchor))
```

---

## 2.7 `rune_cross`

### 物理意义

符文十字结构。由竖线、横线、斜线组合而成。

适合：

```text
北欧符文感
宗教/神秘符号
技能标记
```

### 生成方式

```text
       top
        |
left -- center -- right
        |
      bottom
```

再加斜线：

```text
diag1 -- center -- diag2
```

### 简单代码

```python
center = A0
top    = A1
bottom = A2
left   = A3
right  = A4
diag1  = A5
diag2  = A6

pool = [
    (bottom, center),
    (center, top),
    (left, center),
    (center, right),
    (diag1, center),
    (center, diag2),
    (left, top),
    (right, bottom),
]

edges = random_subset(pool, stroke_count)
```

---

## 2.8 `ladder`

### 物理意义

梯形 / 栅栏结构。

适合：

```text
机械符号
门禁符号
古代铭刻
科技 UI glyph
```

### 生成方式

```text
|---|
|---|
|---|
```

### 简单代码

```python
edges = [
    (left_bottom, left_top),
    (right_bottom, right_top),
]

for each rung:
    a = interpolate(left_bottom, left_top, t)
    b = interpolate(right_bottom, right_top, t)
    edges.append((a, b))
```

---

## 2.9 `parallel_slash`

### 物理意义

多条平行斜线。

适合：

```text
刻痕符号
计数符号
攻击/速度图标
```

### 生成方式

```text
/ / / /
```

### 简单代码

```python
for i in range(stroke_count):
    x = -0.55 + 1.1 * i / (stroke_count - 1)
    a = P(x - 0.15, -0.7)
    b = P(x + 0.15, 0.7)
    edges.append((a, b))
```

为了避免完全分离，可以把最后一条替换为连接线：

```python
if k >= 3 and rng.random() < 0.65:
    edges[-1] = (edges[0][0], edges[-2][1])
```

---

## 2.10 `arc_spine`

### 物理意义

弧形主干 + 分支。

适合：

```text
生物感符文
藤蔓文字
自然魔法符号
```

### 生成方式

先生成弧形主链：

```python
for i in range(main_count):
    x = -0.75 + 1.5 * t
    y = 0.35 * sin(t * pi)
    pts.append(P(x, y))
```

然后加分支：

```python
while len(edges) < stroke_count:
    a = random_anchor_on_spine
    b = new_anchor_outward
    edges.append((a, b))
```

---

## 2.11 `random_tree`

### 物理意义

随机树结构。

适合：

```text
探索阶段
生成大量候选
自然分叉符号
```

### 生成方式

```python
root = center
active = [root]

while len(edges) < stroke_count:
    a = random.choice(active)
    b = new_anchor_near_a
    edges.append((a, b))
    active.append(b)
```

特征：

```text
无闭环
通常 CC = 1
分支不规则
```

---

## 2.12 `cycle_with_tail`

### 物理意义

闭环 + 尾巴。

适合：

```text
P 形符号
钥匙形符号
魔法印记
图腾文字
```

### 生成方式

先生成一个 n 边形环：

```python
for i in range(ncycle):
    anchor_i = point_on_circle(i)

edges = [
    (A0, A1),
    (A1, A2),
    ...
    (An, A0),
]
```

再从环上随机点长出尾巴：

```python
while len(edges) < stroke_count:
    a = random_cycle_anchor
    b = new_anchor_near_a
    edges.append((a, b))
```

---

# 3. 复杂度指标

## 3.1 S = Stroke Count

笔画数。

```text
S = len(strokes)
```

建议：

```text
简单符号：S = 2~4
中等符文：S = 4~8
复杂魔法符号：S = 8~14
```

## 3.2 E = Edge Count

笔画之间的连接关系数量。

这里的 edge 不是 Bézier stroke 本身，而是：

```text
两个 stroke 因为共享 anchor 而产生的拓扑邻接关系
```

## 3.3 CC = Connected Components

联通分量数。

```text
CC = 1
```

表示整个字符是一个连通整体。

```text
CC = 2
```

表示字符有两个互不相连的部分。

## 3.4 Cy = Cycle Count

环数量。近似计算：

```python
Cy = max(0, edge_count - stroke_count + connected_components)
```

含义：

```text
Cy = 0：开放线条
Cy = 1：一个闭环，例如三角形、方框
Cy = 2：两个闭环，例如双环、8 字形结构
```

---

# 4. 论文描述建议

这个生成器可以写成：

```text
A topology-driven procedural glyph generator with human-in-the-loop aesthetic curation.
```

中文：

```text
一种拓扑驱动、带人工审美筛选闭环的程序化字符资产生成器。
```

重点不是完全自动替代美术，而是：

```text
低成本生成大量可编辑候选
由人快速筛选可用资产
把 Good/Bad 样本沉淀为后续审美模型训练数据
```

---

# 5. 推荐默认配置

## 5.1 游戏符文 / 外星文字

```text
S: 3~8
CC: 1~2
Cy: 0~2
style: straight, mild_left, mild_right, s_curve_left, s_curve_right, hook_left, hook_right, mixed
family: rune_cross, fork, zigzag, arc_spine, random_tree, cycle_with_tail
```

## 5.2 科技 UI 符号

```text
S: 3~7
CC: 1
Cy: 0~1
style: straight, mild_left, mild_right
family: ladder, box, parallel_slash, rune_cross
```

## 5.3 魔法阵 / 封印符号

```text
S: 5~12
CC: 1
Cy: 1~3
style: mild_left, mild_right, s_curve_left, s_curve_right, hook_left, hook_right
family: triangle, box, cycle_with_tail, star
```

## 5.4 原始刻痕 / 古文明铭文

```text
S: 2~6
CC: 1~3
Cy: 0~1
style: straight, mild_left, mild_right
family: chain, zigzag, parallel_slash, random_tree
```
