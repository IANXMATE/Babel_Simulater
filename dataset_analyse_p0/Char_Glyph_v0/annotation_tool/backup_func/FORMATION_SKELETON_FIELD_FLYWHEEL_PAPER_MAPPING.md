# Formation / Skeleton Field Flywheel：论文架构映射说明

对应脚本：`annotation_flywheel_pcg_app_morpheme_diffusion_v2_FORMATION_SKELETON_FIELD_FLYWHEEL.py`

> 这份脚本不是直接复现某篇 diffusion/font 论文，而是把几篇论文中适合当前数据飞轮阶段的架构思想，落成一个**非神经随机 proposal generator**。目标是先改善候选集，让后续训练能吃到更自然的“类汉字构造骨架 + 原拓扑符文风格”融合样本。

---

## 1. 整体生成流程

新版脚本里新增的核心流程是：

```text
style / statistics field
    ↓
formation-like layout tree
    ↓
component bounding boxes
    ↓
component internal skeleton primitives
    ↓
structure-aware rune residual
    ↓
axis / grid projector
    ↓
physical split-fragment projector
    ↓
Stage1 topology recall gate / Stage2 rule topology engine / human Good-Bad flywheel
```

和旧版区别：

```text
旧版：
    outline_box / vertical_column / horizontal_stack 模板
    + 随机斜线 residual

新版：
    先采样整体 layout tree 和 skeleton field
    再在 component 内部生成 stroke graph
    rune residual 从已有 scaffold 的安全比例位置长出来
```

所以它不是“机械横竖模板 + 几根斜线”，而是更接近：

```text
整体构造意图
→ 部件布局
→ 局部骨架
→ 风格残差
```

---

## 2. 涉及的论文与对应脚本模块

| 论文 / 方法 | 原论文核心架构 | 脚本中采用的工程化对应 | 脚本位置 / 名称 |
|---|---|---|---|
| FT-CLIP / Formation Tree | 用 formation tree 表示汉字部件层级和空间组合，优于简单 radical sequence | 不使用真实汉字偏旁名，而是采样 `LEFT_RIGHT / TOP_BOTTOM / ENCLOSURE / SPINE_ATTACH / GRID_BLOCK` 的 formation-like layout tree | `_sample_formation_layout_tree()` |
| Skeleton-Guided Diffusion for Font Generation | 用 skeleton 作为显式结构先验，使字体生成保持结构稳定 | 在每个 component bbox 内生成 `spine / bar / corner / enclosure / divider / chord` 等 skeleton primitives | `_primitive_schedule_for_component()`、`_add_component_primitive()` |
| VecFusion: Vector Font Generation with Diffusion | 级联生成：先 raster/global shape，再 vector/control points | 不跑 diffusion，但采用 global-to-local 思路：先整体结构场，再落成 stroke graph / Bézier | `generate_skeleton_field_anchor_edges()` |
| NGG / Neural Graph Generator | 用图统计条件向量控制生成分布 | 用连续风格向量控制横竖性、符文残差、布局凝聚度、斜向动势 | `build_global_style_vector()` |
| ConStruct / hard-constrained graph generation | graph diffusion 中每一步用 projector 保证硬约束 | 不采用它的 diffusion 架构，只借 hard projector 思想，保证最短线段、截断片段、近轴微斜合法 | `physical_split_fragment_check()` 相关逻辑、`enforce_human_readable_axis_geometry()` |

---

## 3. 为什么不直接用 ConStruct

ConStruct 的主场景是 graph diffusion。它适合在扩散采样过程中保证图满足平面性、无环性等硬约束。

你的当前阶段不是 graph diffusion，而是：

```text
Bézier stroke 生成
+ 端点 / T / X 物理接触
+ 横竖排版构造
+ 人工标注飞轮
```

所以脚本只采用 ConStruct 的局部思想：

```text
proposal 生成后必须经过 hard projector
```

不采用：

```text
edge-absorbing graph diffusion
```

当前 projector 主要约束：

```text
1. 最短线条长度
2. 近横竖角度禁止 10° 内假斜
3. 任意物理切分片段不得短于总长度 1/6
4. 端点附近 corner / E2E 不误判为内部截断
```

---

## 4. 新增 topology families

新版默认增加并勾选：

```text
formation_skeleton_field
layout_skeleton_field
organic_seal_field
```

含义：

### `formation_skeleton_field`

偏向通用 formation-like 结构：

```text
LEFT_RIGHT
TOP_BOTTOM
ENCLOSURE
SPINE_ATTACH
GRID_BLOCK
```

适合产生：

```text
类汉字部件组合 + 符文内生残差
```

### `layout_skeleton_field`

偏向布局骨架：

```text
左右排布
上下排布
中心主干
网格块
```

适合产生更可读的部件布局。

### `organic_seal_field`

偏向外框、半包围、印章感结构：

```text
outer enclosure
inner divider
corner
flow chord
```

适合产生“古代印记 / 外星符文 / 方块封印”的融合形态。

---

## 5. 脚本顶部默认配置

脚本开头仍保留可改配置：

```python
DEFAULT_GUI_PROFILE = {
    "style_modes_checked": [...],
    "base_topology_families_checked": [...],
    "human_topology_families_checked": [...],
    "checkbox_defaults": {...},
    "scalar_defaults": {...},
}
```

新增默认项包括：

```python
"use_skeleton_field_generator": True,
"skeleton_field_weight_mult": 5.2,
"skeleton_component_complexity": 0.68,
"skeleton_enclosure_bias": 0.54,
"skeleton_rune_residual": 0.50,
"skeleton_layout_mutation": 0.34,
```

参数解释：

| 参数 | 含义 |
|---|---|
| `skeleton_field_weight_mult` | skeleton field families 在采样中的权重倍率 |
| `skeleton_component_complexity` | component 内部 primitive 数量和复杂度 |
| `skeleton_enclosure_bias` | 外框 / 半包围 / 印章结构倾向 |
| `skeleton_rune_residual` | 斜向符文 residual 的出现概率 |
| `skeleton_layout_mutation` | component bbox 和 primitive 的非机械扰动强度 |

---

## 6. 数据保存与后续模型训练价值

每个候选会在 `pcg_meta["axis_layout_prior"]` 中记录：

```json
{
  "use_skeleton_field_generator": true,
  "skeleton_field_weight_mult": 5.2,
  "skeleton_component_complexity": 0.68,
  "skeleton_enclosure_bias": 0.54,
  "skeleton_rune_residual": 0.50,
  "skeleton_layout_mutation": 0.34
}
```

这对后续训练很重要，因为模型可以学习：

```text
哪些 layout tree / skeleton field / rune residual 组合更容易被人工标 Good
```

后续可以升级为：

```text
good pool → 学习 style/statistics vector 分布
morpheme tree → 学习 component skeleton prior
topology model → 预测候选可读性 / 美感 / 结构完整度
```

---

## 7. 参考论文链接

- FT-CLIP / Formation Tree: <https://arxiv.org/abs/2404.12693>
- Skeleton-Guided Diffusion for Font Generation: <https://www.mdpi.com/2079-9292/14/19/3932>
- VecFusion: Vector Font Generation with Diffusion: <https://arxiv.org/abs/2312.10540>
- NGG: Neural Graph Generator: <https://arxiv.org/abs/2403.01535>
- ConStruct: Generative Modelling of Structurally Constrained Graphs: <https://arxiv.org/abs/2406.17341>

---

## 8. 当前脚本的边界

当前脚本仍然是随机 proposal generator，不是训练好的 diffusion / transformer model。

它解决的是：

```text
先让数据飞轮产生更好的融合候选集
```

而不是：

```text
直接完成最终神经生成模型
```

下一阶段更合理的神经路线是：

```text
formation-like layout tree encoder
+ skeleton graph encoder
+ morpheme tree prior
+ topology-conditioned vector generator
+ physical projector
```

也就是把当前脚本生成并人工标注的数据，作为下一阶段模型训练的数据源。
