# Creative Formation-Flow Flywheel：方案与论文架构对应说明

## 1. 目标

这个脚本不是继续枚举“汉字规则模板”，而是把随机生成阶段改成一个非神经版的论文式 proposal generator：

```text
global style/statistics vector
→ recursive formation program
→ skeleton field + ports
→ readable axis scaffold
→ structure-aware rune flow
→ physical hard projector
→ novelty / stage1 recall gate / stage2 topology engine / human Good-Bad
```

它的目标是让“人类可读构造”和“曲线符文动势”从同一个结构场中生长出来，而不是简单地：

```text
汉字横竖模板 + 随机符文斜线
```

---

## 2. 这版脚本新增了什么

脚本文件：

```text
annotation_flywheel_pcg_app_morpheme_diffusion_v2_CREATIVE_FORMATION_FLOW.py
```

新增 families：

```text
creative_formation_flow
recursive_portal_field
interlock_rune_flow
orbit_weave_field
```

这些 family 不对应固定汉字模板，而是触发一个新的生成分支：

```text
generate_creative_formation_flow_anchor_edges()
```

新增 GUI 默认配置开关位于脚本开头：

```python
DEFAULT_GUI_PROFILE = {
    "style_modes_checked": [...],
    "base_topology_families_checked": [...],
    "human_topology_families_checked": [...],
    "checkbox_defaults": {...},
    "scalar_defaults": {...},
}
```

你可以直接在这个字典里改默认勾选项和默认参数，不需要每次打开工具手动勾选。

---

## 3. 核心生成流程

### 3.1 global style/statistics vector

对应脚本函数：

```python
_creative_style_vector()
build_global_style_vector()
```

它采样连续风格条件，而不是硬标签：

```text
axis_readability
curve_flow_strength
non_hanzi_operator_prob
port_coupling
nested_enclosure_prob
interlock_prob
orbit_weave_prob
rune_residual
field_strength
```

这一步呼应 NGG / feature-conditioned graph generation 的思想：用 graph/statistics-like 条件向量控制生成属性，而不是简单选择一个离散 family。

---

### 3.2 recursive formation program

对应脚本函数：

```python
_sample_creative_program()
_execute_creative_op()
```

生成器会采样 operator 序列，例如：

```text
NESTED_PORTAL
SPINE_GROW
INTERLOCK
FLOW_BRANCH
```

或者：

```text
INTERLOCK
WEAVE
HINGE
ECHO
```

这些 operator 不是传统汉字模板，而是更一般的构成动作：

```text
ANCHOR_FRAME      锚定框架
NESTED_PORTAL     多层/错位/破口门户
PORTAL            外壳 + 内核 + 结构流
SPINE_GROW        主干生长
BALANCED_STACK    平衡堆叠
INTERLOCK         互锁
ORBIT             环绕
WEAVE             交织
HINGE             铰接
ECHO              回声式重复
FLOW_BRANCH       沿流场生长的符文分支
```

这一步呼应 formation tree 的抽象：字符由层级结构组合而来。但这里不使用传统汉字 radical 名称，而是用更通用的 formation operators。

---

### 3.3 skeleton field + ports

对应脚本函数：

```python
_creative_anchor_id()
_creative_line()
_creative_shell()
_execute_creative_op()
```

生成器会维护 anchors / ports，让结构真的连接在同一个图中，而不是多个模板贴在一起。

可读线条和符文线都从同一组 anchors / ports 中生成：

```text
外壳角点
内部核心
主干切点
安全比例点
互锁接触点
orbit 中心点
```

这一步呼应 skeleton-guided font generation：用 skeleton / skeletal attributes 作为结构先验。

---

### 3.4 structure-aware rune flow

对应脚本函数：

```python
FLOW_BRANCH
HINGE
WEAVE
ORBIT
apply_style_field_fusion()
make_nodes_from_anchor_edges()
```

脚本不会再把符文线随便插入，而是从已有结构端口生长：

```text
host edge 的安全比例点
outer shell 的角点
inner core
orbit center
interlock contact
spine attach point
```

同时，横竖/近轴结构仍会强制 straight；非横竖 rune-flow edge 才允许 mild / s-curve 样式。

这一步呼应 VecFusion 的 global-to-vector 思想：先生成整体意图和结构场，再落成 vector stroke / control points。

---

### 3.5 physical hard projector

对应脚本已有函数：

```python
enforce_human_readable_axis_geometry()
candidate_physical_split_fragment_diagnostics()
passes_min_split_fragment_filter()
```

它继续保证：

```text
线段不能太短
近横竖不能 10° 内假斜
真实物理切分后的片段不能短于总长度的 1/6
端点附近 corner 接触不误判为内部截断
```

这里借的是 hard constraint projector 的思想。它不是完整 ConStruct graph diffusion 架构，只是把“约束投影”思想用于当前随机飞轮生成阶段。

---

## 4. 与论文架构的对应关系

| 论文/方向 | 论文里的关键思想 | 脚本里的对应实现 | 是否完整复现 |
|---|---|---|---|
| FT-CLIP / formation tree | 用 formation tree 表达字符部件层级，而不是 radical sequence | `_sample_creative_program()` 采样 recursive formation program | 否，只借层级构成思想 |
| Skeleton-guided font generation / SGCE | skeleton 提供局部和全局结构指导 | anchors / ports / spine / shell / core 形成 skeleton field | 否，非神经实现 |
| VecFusion | 先全局 raster/shape，再 vector/control points | global style vector → skeleton field → Bézier strokes | 否，非 diffusion 近似 |
| NGG | 用 graph statistics vector 条件控制生成 | `_creative_style_vector()` 连续控制 axis / flow / interlock / nesting | 否，只借条件向量思想 |
| ConStruct | diffusion 中用 projector 保持硬图约束 | physical split-fragment projector / axis projector | 否，只借 hard projector 思想 |

---

## 5. 为什么这版比规则模板更适合你的目标

旧方案更像：

```text
outline_box + random diagonal
vertical_column + rune_chord
hanzi_block + residual line
```

这会显得机械。

新方案更像：

```text
采样整体风格向量
→ 采样构成程序
→ 建立同一个 skeleton / port field
→ 横竖骨架和符文曲线都从这个 field 生长
```

因此它更符合你的目标：

```text
人类能理解的构造
+
非汉字的想象力结构
+
曲线符文样式
```

三者不是相加，而是共同生长。

---

## 6. 推荐参数

默认值在脚本开头：

```python
"creative_operator_depth": 3,
"creative_non_hanzi_operator_prob": 0.58,
"creative_curve_flow_strength": 0.62,
"creative_port_coupling": 0.76,
"creative_nested_enclosure_prob": 0.46,
"creative_interlock_prob": 0.44,
"creative_orbit_weave_prob": 0.34,
```

调参建议：

```text
更像人类可读结构：
    creative_non_hanzi_operator_prob ↓
    creative_nested_enclosure_prob ↑
    axis_snap_prob ↑

更像外星/符文：
    creative_non_hanzi_operator_prob ↑
    creative_curve_flow_strength ↑
    creative_interlock_prob ↑
    creative_orbit_weave_prob ↑

更少机械：
    creative_operator_depth ↑ 到 4
    creative_port_coupling ↑
    organic_layout_mutation_prob ↑

更稳定、更少 reject：
    creative_curve_flow_strength ↓
    creative_orbit_weave_prob ↓
    min_human_line_length ↓ 一点
```

---

## 7. 参考论文/方向

- FT-CLIP: Improving Chinese Character Representation with Formation Tree-CLIP, 2024.  
  https://arxiv.org/abs/2404.12693

- VecFusion: Vector Font Generation with Diffusion, 2023 / CVPR 2024.  
  https://arxiv.org/abs/2312.10540

- Neural Graph Generator: Feature-Conditioned Graph Generation using Latent Diffusion Models, 2024.  
  https://arxiv.org/abs/2403.01535

- SGCE-Font: Skeleton Guided Channel Expansion for Chinese Font Generation, 2022.  
  https://arxiv.org/abs/2211.14475

- Skeleton-Guided Diffusion for Font Generation, 2025.  
  https://www.mdpi.com/2079-9292/14/19/3932

- ConStruct: hard constraints in graph diffusion, 2024.  
  https://arxiv.org/abs/2406.17341
