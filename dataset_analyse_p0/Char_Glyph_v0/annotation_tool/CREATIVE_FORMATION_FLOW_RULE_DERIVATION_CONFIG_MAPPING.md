# Creative Formation-Flow + Independent Rule Derivation + Config Profiles

## 1. 这个版本新增什么

本版本基于当前 Creative Formation-Flow 飞轮继续扩展，目标有三点：

1. 保留原来的随机生成 / novelty / stage1 recall gate / stage2 topology engine / Good-Bad commit 流程。
2. 将 GUI 中主要勾选栏按真实处理顺序编号。
3. 增加 `annotation_tool/config` 配置目录，支持快速 Load 和 Save Current As。

配置文件使用 `.cfg` 后缀，内部为 JSON，便于手工编辑、复制、版本管理。

默认配置目录：

```text
Babel_Simulater/dataset_analyse_p0/Char_Glyph_v0/annotation_tool/config
```

脚本第一次运行会自动写入内置配置。

---

## 2. GUI 序号与真实处理顺序

本版本把主要勾选栏按流水线编号：

```text
00 Config Profile
01 Stage 0M Morpheme Grammar Diffusion
02 Stage 1A Style Modes
03 Stage 1B Base Topology Families
04 Stage 1C Axis / Human-readable Construction
05 Stage 1D Creative Formation-Flow
06 Stage 1E Human / Skeleton / Creative Families
07 Stage 1.5 Positive Topology Novelty
08 Stage 2 V7 Stage1 Topology Gate
```

注意：`strokes / CC / cycles` 属于最终硬筛，虽然在 GUI 顶部显示，但其语义不是“先生成”，而是“生成后容量限制”。

实际生成流程：

```text
读取配置
→ 语素/规则 prior
→ Style / Family / Axis / Creative 参数组成 GeneratorConfig
→ PCG proposal
→ complexity hard gate
→ novelty filter
→ stage1 model recall gate
→ stage2 topology rule engine
→ physical split-fragment projector
→ aesthetic hook / manual Good-Bad
```

---

## 3. 配置文件说明

### runic_default.cfg

目标：尽量复刻 `annotation_flywheel_pcg_app_morpheme_diffusion_v2_FAST_CACHE_Last_valueable.py` 的默认生成逻辑。

特征：

```text
只启用旧 PCG topology families
使用旧默认 style modes
关闭 Axis / Human-readable Construction
关闭 Style Field Fusion
关闭 Creative Formation-Flow
关闭物理短碎片过滤
保留原始 complexity: stroke 3~8, CC 1, cycles 0~2
```

适合用途：回到旧版本作为对照组。

### creative_fusion_balanced.cfg

主工作区：人类可读骨架 + 曲线符文流 + 中等互锁。

### ancient_seal_portal.cfg

偏封印、门洞、多层包围、壳层结构。

### alien_totem_explore.cfg

更外星、更互锁、更环绕/编织；通过率较低，但容易挖出新结构。

### clean_readable_axis.cfg

偏干净、可读、低噪声，适合积累稳定好样本。

---

## 4. 关于“只凭 Good/Bad 样本派生独立拓扑规则”

可以实现。关键是不要让新规则依赖现有 rule_id，而是只从样本图本身抽取结构描述。

推荐 pipeline：

```text
Good/Bad JSON pool
→ stroke graph / topology event graph
→ canonical graph encoding
→ candidate substructure mining
→ positive-vs-negative discriminative scoring
→ rule compression / dedup
→ 导出 independent_rule_xxx
→ 作为新 prior 或 filter 接入飞轮
```

独立规则的形式可以是：

```json
{
  "rule_id": "independent_topo_0007",
  "source": "good_bad_discriminative_mining",
  "conditions": [
    {"feature": "cycle_count", "op": ">=", "value": 1},
    {"feature": "T_ratio", "op": "between", "value": [0.05, 0.25]},
    {"feature": "articulation_count", "op": ">=", "value": 2},
    {"subgraph_signature": "dfs_code:...", "support_good": 0.31, "support_bad": 0.04}
  ],
  "score": {
    "precision": 0.84,
    "recall": 0.19,
    "lift": 5.7
  }
}
```

这种规则不需要叫 `box`, `cross`, `fork`, `grid_ladder`。它只需要在 Good 中显著多、在 Bad 中显著少。

---

## 5. 可借鉴论文 / 架构

### 5.1 gSpan / closed subgraph mining

用途：从 Good/Bad 图中挖频繁子图，用 canonical DFS code 去重。

对应本项目：

```text
stroke graph / topology event graph
→ frequent subgraph
→ discriminative subgraph
→ independent topology rule
```

### 5.2 Discriminative subgraph mining

用途：不是找“常见结构”，而是找“能区分 Good 和 Bad 的结构”。

对应本项目：

```text
support_good 高
support_bad 低
lift / information gain / chi-square 高
```

### 5.3 SUBDUE / MDL graph discovery

用途：用 Minimum Description Length 找能压缩图数据库的结构概念。

对应本项目：

```text
不是只找局部高频结构
而是找能解释一批 Good 字形的结构 motif
```

### 5.4 Graph grammar induction

用途：从样本图中诱导可复用的 graph grammar production。

对应本项目：

```text
independent rule
→ production rule
→ 可递归构成 operator
→ 进入 Creative Formation-Flow
```

### 5.5 Inductive Logic Programming / FOIL-style rule learning

用途：从正例和负例中学习可解释逻辑规则。

对应本项目：

```text
Good = positive examples
Bad = negative examples
features/subgraphs = predicates
rule = explain Good but avoid Bad
```

### 5.6 Formation Tree / Skeleton-guided / VecFusion / NGG

这些论文对应当前生成器的结构方式：

```text
Formation Tree:
    hierarchical component / operator program

Skeleton-guided font generation:
    skeleton / port / spine / shell as structure prior

VecFusion:
    global field first, vector strokes later

NGG-like conditioning:
    continuous graph/statistics vector controls generation
```

---

## 6. 当前脚本里的定位

本脚本已经实现：

```text
配置文件读写
序号化 GUI
Creative Formation-Flow 生成器
runic_default 对照配置
多风格配置文件
```

本脚本已内置一个轻量版 `01B Independent Topology Rule Miner`：

```text
Good/Bad pool -> topology atom features -> positive/negative discriminative score -> independent_topo_atom_xxxx
```

它会把初版独立规则输出到：

```text
Char_Glyph_v0/Morpheme/new_rule_cache/
```

这些规则目前是轻量 atom 级别，适合做第一轮可解释审计和 rule cache 种子；后续可以把 atom 升级为 gSpan/graph grammar 级别的真正子图规则，再由 Morpheme Diffusion / Creative Formation-Flow 读取。

