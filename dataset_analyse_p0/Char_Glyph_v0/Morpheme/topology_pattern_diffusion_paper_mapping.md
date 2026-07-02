# `topology_pattern_diffusion_api.py` 的论文来源与创新点映射说明

本文档解释 `topology_pattern_diffusion_api.py` 中与近年图学习 / 图语法 / motif mining 论文相关的创新来源，并说明这些思想如何被改造成字体拓扑语素飞轮中的工程模块。

> 重要说明：  
> 当前脚本不是对某篇论文的复现，也没有声称使用某个 2024+ “最优算法”。  
> 它是一个面向字体拓扑语素的领域定制方法，核心是把近年论文中的三个方向转译为：
>
> 1. 层级图 tokenization；
> 2. 频繁子图 / motif 发现；
> 3. 图语法 production rule 与递归展开；
>
> 然后进一步结合你的字体拓扑语素系统，形成：
>
> **rule-stored topology predicates + morpheme grammar diffusion + data-driven pattern rule discovery**

---

## 0. 脚本核心结构概览

脚本文件：

```text
dataset_analyse_p0/Char_Glyph_v0/Morpheme/topology_pattern_diffusion_api.py
```

飞轮调用主入口：

```python
from topology_pattern_diffusion_api import (
    load_morpheme_tree,
    build_default_rule_registry,
    derive_builtin_rules,
    score_morphemes_by_rules,
    diffuse_by_flywheel_selection,
    discover_new_pattern_rules,
    merge_auto_rules,
)
```

核心流程：

```text
morpheme_nodes + composition_alternatives
        ↓
extract_morpheme_features()
        ↓
rule registry: star / fork / ladder / chain / cycle_with_tail / ...
        ↓
score_morphemes_by_rules()
        ↓
飞轮 GUI 勾选 selected_patterns
        ↓
diffuse_by_flywheel_selection()
        ↓
weights: {morpheme_id: probability}, sum = 1
```

这个过程不是 “生成 star / ladder / fork 模板”，而是：

```text
用户勾选结构意图
→ 结构意图变成 rule prior
→ 在语素文法图上扩散
→ 输出可复用语素的概率分布
```

---

# 1. 论文一：HIGHT: Hierarchical Graph Tokenization for Graph-Language Alignment

## 1.1 论文信息

**论文名：**

> HIGHT: Hierarchical Graph Tokenization for Graph-Language Alignment

**时间：** 2024  
**方向：** Graph Tokenization / Graph-Language Alignment / Hierarchical Graph Representation  
**链接：** https://arxiv.org/abs/2406.14021

## 1.2 论文的核心创新点

HIGHT 关注的问题是：

```text
LLM 主要处理一维文本 token，
但是 graph 本身有层级结构。
如果只把 graph 拆成 node tokens，
会丢失 motif / graph-level 的高阶结构信息。
```

它的关键创新是：

```text
node-level token
motif-level token
graph-level token
```

也就是说，graph 不应该只看节点和边，而应该显式把高阶子结构作为 token。

论文中特别强调 molecular graph 中的 functional groups / motifs 对语义很重要。换到你的字体任务里，对应关系是：

```text
atom / node              → stroke endpoint / stroke primitive
molecular motif          → 字体拓扑语素 morpheme
whole molecular graph    → 完整 glyph topology
```

## 1.3 对脚本的启发

你的字体拓扑系统天然也是层级结构：

```text
line
→ small morpheme
→ composed morpheme
→ full glyph topology
```

对应到脚本：

```python
load_morpheme_tree()
score_morphemes_by_rules()
diffuse_by_flywheel_selection()
```

脚本没有把一个 glyph 当成扁平 stroke 集合，而是读入：

```text
morpheme_nodes
composition_alternatives
```

其中：

```text
morpheme_nodes              = motif-level topology token
composition_alternatives    = token 之间的构成关系
glyph                       = graph-level structure
```

## 1.4 脚本中的对应设计

### 设计 A：把语素当成 graph token

脚本中：

```python
def load_morpheme_tree(tree_dir):
    nodes = load_sharded_rows(tree_dir, "morpheme_nodes")
    comps = load_sharded_rows(tree_dir, "composition_alternatives")
    return nodes, comps
```

这里的 `nodes` 不是普通 graph node，而是你前面构建出来的 topology morpheme node。

也就是说，脚本操作的基本单位不是：

```text
单个 stroke
```

而是：

```text
可复用 topology morpheme
```

这就是 HIGHT 思想在字体拓扑上的转译：

```text
node token / motif token / graph token
↓
line primitive / morpheme token / glyph topology
```

### 设计 B：保留层级扩散，而不是一次性扁平选择

飞轮勾选某个 pattern 后，脚本不是只选 top-k 语素，而是沿着 `composition_alternatives` 做递归扩散：

```python
diffuse_by_flywheel_selection(...)
```

其逻辑是：

```text
root morpheme softmax
→ parent 保留一部分质量
→ 余量沿 production rule 分给 child morphemes
→ child 继续递归
```

这对应 HIGHT 的层级 tokenization 思想：结构语义不只存在于最底层，也存在于中间 motif 和整体 graph 层。

## 1.5 可写进论文 / 报告的表述

可以这样写：

> Inspired by hierarchical graph tokenization, we treat glyph topology not as a flat set of strokes but as a hierarchy of topology morphemes. The terminal token is a straight stroke primitive, while non-terminal tokens correspond to reusable topology motifs. User-selected structural intents are propagated over this induced morpheme hierarchy rather than directly applied to raw strokes.

中文版本：

> 受层级图 tokenization 思想启发，我们不再将字体拓扑视为扁平笔画集合，而是将其表示为由直线终结符、拓扑语素和完整字形组成的层级 token 系统。飞轮中的结构意图不是直接作用于原始笔画，而是在诱导得到的语素层级上递归扩散。

---

# 2. 论文二：Representation Learning for Frequent Subgraph Mining / SPMiner

## 2.1 论文信息

**论文名：**

> Representation Learning for Frequent Subgraph Mining

**方法名：**

> SPMiner, Subgraph Pattern Miner

**时间：** 2024  
**方向：** Frequent Subgraph Mining / Network Motif Discovery / Representation Learning  
**链接：** https://arxiv.org/abs/2402.14367

## 2.2 论文的核心创新点

这篇论文关注的是 frequent subgraph mining，也就是在大图里发现高频子图 / motif。

传统 frequent subgraph mining 的问题是：

```text
子图计数包含 NP-hard 子问题
候选子图数量指数爆炸
大 motif 很难精确枚举
```

SPMiner 的创新点是：

```text
把子图映射到表示空间
在 embedding / order embedding space 中搜索频繁 motif
用近似神经搜索替代暴力枚举
```

它不是简单地手写 motif 模板，而是从图数据中发现 frequent patterns。

## 2.3 对脚本的启发

你的任务中，人工定义的：

```text
star
fork
ladder
chain
cycle_with_tail
```

本质上都是 topology motifs。

但是你明确指出：

```text
模式不应该局限于人为定义的几种形状。
```

所以脚本加入了：

```python
discover_new_pattern_rules(...)
```

它不是完整复现 SPMiner，而是一个轻量的 topology motif discovery：

```text
morpheme topology features
+ existing rule scores
→ feature embedding
→ clustering
→ auto_pattern_xxx
→ 自动生成新的 pattern rule
```

这对应 SPMiner 的精神：

```text
不要只依赖人工 motif 模板，
而要允许数据中的高频结构自己浮现。
```

## 2.4 脚本中的对应设计

### 设计 A：从语素中抽取 topology embedding

脚本中：

```python
def extract_morpheme_features(node):
    ...
```

提取了大量可解释 topology features：

```text
stroke_count
edge_density
cycle_rank
max_degree
leaf_ratio
branch_ratio
T_ratio
X_ratio
E2E_ratio
acute_ratio
right_ratio
collinear_ratio
endpoint_hub_ratio
intersection_hub_ratio
intersection_density
parallel_pair_ratio
orientation_entropy
...
```

这些就是当前版本的手工 topology embedding。

### 设计 B：规则分数也进入 embedding

```python
def feature_vector_for_discovery(features, scores, registry):
    vals = [features[k] for k in FEATURE_KEYS]
    vals += [scores[rule_id] for rule_id in rule_ids]
    return np.asarray(vals)
```

这一步很重要，因为它不是只用几何统计，而是把现有人工规则也作为语义坐标：

```text
raw topology features
+
known pattern scores
=
pattern discovery embedding
```

这使得自动发现的新模式可以表达为：

```text
接近 star 但更稀疏
接近 ladder 但带 tail
接近 cycle 但有 branch
```

而不是完全黑箱聚类。

### 设计 C：自动发现新 rule

```python
def discover_new_pattern_rules(...):
    ...
    auto_rules[auto_id] = make_rule(
        auto_id,
        selected_terms,
        aliases=[auto_label],
        description=...,
        parent_rules=...,
        source="auto_discovered",
    )
```

输出的新模式不是临时 cluster id，而是被转换成和人工规则同格式的 rule：

```json
{
  "rule_id": "auto_pattern_003",
  "source": "auto_discovered",
  "terms": [...],
  "parent_rules": [...],
  "description": "Auto-discovered topology mode..."
}
```

因此它可以继续被：

```python
merge_auto_rules(registry, discovered)
```

合并回规则库，然后继续被飞轮调用。

## 2.5 这部分和 SPMiner 的关系

不能说脚本实现了 SPMiner，因为当前脚本没有使用 GNN、order embedding，也没有做 neural motif search。

准确说法应该是：

```text
受 frequent subgraph mining / SPMiner 的思想启发，
脚本提供了轻量的 topology motif discovery 入口。
```

你的系统现在是：

```text
人工规则
+
自动聚类发现的新规则
+
后续可以替换为 GNN / SPMiner-like embedding
```

## 2.6 可写进论文 / 报告的表述

英文：

> Inspired by representation-learning-based frequent subgraph mining, we provide a rule discovery module that maps each topology morpheme into an interpretable feature-and-rule-score embedding. Clustering in this space yields new candidate topology modes, which are converted back into editable pattern rules and merged into the flywheel registry.

中文：

> 受表示学习式频繁子图挖掘启发，我们将每个拓扑语素映射到由拓扑统计特征和已有规则分数组成的可解释嵌入空间。系统在该空间中聚类发现新的候选结构模式，并将其反向转化为可编辑、可扩展的拓扑规则，从而避免结构模式被限制在人为枚举的 star、fork、ladder 等类别中。

---

# 3. 论文三：Directed Graph Grammars for Sequence-based Learning

## 3.1 论文信息

**论文名：**

> Directed Graph Grammars for Sequence-based Learning

**时间：** 2025  
**方向：** Graph Grammar / DAG Representation / Sequence-based Learning / Production Rules  
**链接：** https://arxiv.org/abs/2505.22949

## 3.2 论文的核心创新点

这篇论文关注的问题是：

```text
DAG 是常见图结构，
但是 DAG 可以有很多拓扑排序。
如果要把图喂给 sequence model，
需要一个原则性的 graph → sequence 表示。
```

它的核心思想是：

```text
把 graph 看作 grammar derivation
用 production rules 表示图的生成/压缩过程
得到紧凑、原则性、等价的序列表达
```

论文强调的是：

```text
graph ↔ production-rule sequence
```

这种思路对你的字体拓扑系统非常直接，因为你已经有：

```text
parent_morpheme
← child_a + child_b + topology_way
```

这本质就是 production rule。

## 3.3 对脚本的启发

你的语素树构建器已经输出：

```text
composition_alternatives
```

每一条 composition alternative 表示：

```text
一个 parent morpheme 可以由两个 child morphemes
通过某种 topology_way 构成
```

脚本在此基础上做了扩散：

```python
diffuse_by_flywheel_selection(...)
```

也就是说：

```text
飞轮结构意图
→ 选择 parent morpheme
→ 沿 production rules 向 child 扩散
→ 得到递归构成权重
```

这和图语法中的 derivation / production rule 思想是一致的。

## 3.4 脚本中的对应设计

### 设计 A：composition alternative 作为 production rule

脚本读入：

```python
compositions = load_sharded_rows(tree_dir, "composition_alternatives")
```

每条 composition 类似：

```json
{
  "parent_morpheme_id": "M001",
  "child_morpheme_ids": ["M023", "M087"],
  "topology_way": {
    "relation_hist": {...},
    "angle_class_hist": {...}
  }
}
```

这可以解释为：

```text
M001 <- M023 + M087 + topology_way
```

也就是字体拓扑语素文法的 production rule。

### 设计 B：规则自身也有 topology_way score

脚本中：

```python
def _rule_score_for_topology_way(comp, selected_rule_scores):
    ...
```

它会读取 production rule 的：

```text
relation_hist
angle_class_hist
```

并估计它是否符合某些结构意图：

```text
T 型
X 型
chain
fork
ladder
cross
zigzag
```

这意味着扩散不是只看 parent / child 的形状，也看：

```text
parent 是如何由 child 连接起来的
```

这点非常适合讲你的语素定义：

```text
子语素集合 + 构成拓扑方式 = 父语素语义
```

### 设计 C：质量守恒式递归扩散

核心函数：

```python
diffuse_by_flywheel_selection(...)
```

逻辑：

```text
root morpheme softmax
→ parent 保留 keep_ratio
→ 剩余质量根据 production rule softmax 分给 child
→ child 按 pattern importance softmax 分配
→ 递归 max_depth 层
→ 归一化，最终总和为 1
```

它不是随机游走，也不是纯采样，而是：

```text
给定结构意图，在语素文法图上的 soft derivation prior
```

这可以被解释为一种面向生成的 prior distribution：

```text
P(morpheme | selected topology intents)
```

## 3.5 可写进论文 / 报告的表述

英文：

> Inspired by graph grammar representations, each morpheme composition alternative is treated as a production rule of the form parent ← child_a + child_b + topology_way. Given a set of user-selected topology intents, we compute a mass-conserving recursive diffusion over this production-rule graph, producing a normalized prior distribution over reusable morphemes.

中文：

> 受图语法 production rule 表达启发，我们将每条语素构成关系表示为 `parent ← child_a + child_b + topology_way`。给定飞轮中勾选的拓扑结构意图，系统在该 production-rule 图上进行质量守恒的递归扩散，得到一个总和为 1 的可复用语素先验分布。

---

# 4. 三篇论文与脚本模块的对应表

| 论文 | 论文创新点 | 脚本中的对应模块 | 在字体拓扑中的改造 |
|---|---|---|---|
| HIGHT: Hierarchical Graph Tokenization for Graph-Language Alignment | node / motif / graph 层级 tokenization | `load_morpheme_tree`, `score_morphemes_by_rules`, `diffuse_by_flywheel_selection` | 将 `line → morpheme → composed morpheme → glyph` 作为字体拓扑层级 token |
| Representation Learning for Frequent Subgraph Mining / SPMiner | 用表示学习近似发现 frequent motif，避免纯枚举爆炸 | `extract_morpheme_features`, `feature_vector_for_discovery`, `discover_new_pattern_rules` | 用 topology feature + known rule score 聚类，发现 `auto_pattern_xxx` |
| Directed Graph Grammars for Sequence-based Learning | 用 graph grammar / production rules 表示图的等价序列或推导 | `composition_alternatives`, `_rule_score_for_topology_way`, `diffuse_by_flywheel_selection` | 将 `parent ← child_a + child_b + topology_way` 作为字体拓扑语素 production rule |

---

# 5. 脚本中的主要创新点

## 5.1 Rule-stored topology predicates

传统做法可能是：

```python
generate_star()
generate_ladder()
generate_fork()
```

这会导致模式被写死。

脚本改成：

```python
build_default_rule_registry()
```

每个结构模式都是可存储规则：

```json
{
  "rule_id": "ladder",
  "terms": [
    {"feature": "parallel_pair_ratio", "op": "high", ...},
    {"feature": "perpendicular_pair_ratio", "op": "high", ...},
    {"feature": "right_ratio", "op": "high", ...}
  ]
}
```

因此：

```text
star 不再是固定图形模板
ladder 不再是固定图形模板
fork 不再是固定图形模板
```

而是：

```text
可解释 topology predicate
```

## 5.2 可派生的拓扑规则

脚本提供：

```python
compose_rules(...)
derive_builtin_rules(...)
```

比如：

```text
branch_chain = chain + fork
cycle_ladder = cycle + ladder
star_with_tail = star + chain
```

这使结构空间可以逐步生长：

```text
人工基础结构
→ 组合派生结构
→ 数据发现结构
→ 再参与扩散
```

这比手工维护固定 shape 列表更适合飞轮系统。

## 5.3 Data-driven auto pattern rule

脚本提供：

```python
discover_new_pattern_rules(...)
merge_auto_rules(...)
```

它会把聚类结果变成真正的 rule：

```text
auto_pattern_000
auto_pattern_001
...
```

而不是只输出临时 cluster。

这样新模式可以回到飞轮里：

```python
registry = merge_auto_rules(registry, discovered)
```

之后飞轮 GUI 也可以显示这些新 rule。

## 5.4 Mass-conserving morpheme grammar diffusion

脚本主函数：

```python
diffuse_by_flywheel_selection(...)
```

保证输出：

```python
sum(result["weights"].values()) == 1.0
```

这点很重要，因为你的需求是：

```text
递归分配，但是要求总和为 1
```

质量守恒的意义是：

```text
每个勾选意图最终转化为一个规范化 morpheme prior
```

后续生成器可以直接用它采样或排序候选语素。

---

# 6. 可以用在论文 / 项目文档里的总叙述

## 6.1 中文版

我们提出一种面向字体拓扑生成飞轮的 **Topology Morpheme Grammar Diffusion** 方法。该方法首先将清洗后的正样本字形拓扑诱导为层级语素文法，其中直线是终结符，复杂语素是非终结符，每条构成关系表示为 `parent ← child_a + child_b + topology_way`。随后，我们将人工结构选项，如 `star`、`fork`、`ladder`、`chain`、`cycle_with_tail` 等，不再实现为固定形状生成器，而是存储为可解释的 topology predicate rules。飞轮中勾选的结构意图被转化为 rule prior，并在语素 production-rule 图上进行质量守恒的递归扩散，得到总和为 1 的语素出现分布。进一步地，系统支持基于 topology feature 与 rule-score embedding 的无监督 pattern discovery，将新发现的结构簇转化为可编辑的 auto rules，从而避免拓扑模式被限制在人为枚举的少数类别中。

## 6.2 英文版

We propose **Topology Morpheme Grammar Diffusion** for a glyph topology generation flywheel. Given a canonical corpus of cleaned positive glyph topologies, we induce a hierarchical morpheme grammar where straight strokes serve as terminals and reusable topology morphemes serve as non-terminals. Each composition alternative is represented as a production rule of the form `parent ← child_a + child_b + topology_way`. Instead of implementing user-facing structural options such as `star`, `fork`, `ladder`, `chain`, and `cycle_with_tail` as hard-coded shape generators, we store them as interpretable topology predicate rules. User-selected structural intents are transformed into rule priors and recursively diffused over the morpheme production-rule graph in a mass-conserving manner, yielding a normalized prior distribution over reusable morphemes. Furthermore, the system supports unsupervised pattern discovery based on topology-feature and rule-score embeddings, converting newly discovered clusters into editable auto rules that can be merged back into the flywheel registry.

---

# 7. 参考文献

1. **HIGHT: Hierarchical Graph Tokenization for Graph-Language Alignment**  
   arXiv: https://arxiv.org/abs/2406.14021

2. **Representation Learning for Frequent Subgraph Mining**  
   arXiv: https://arxiv.org/abs/2402.14367

3. **Directed Graph Grammars for Sequence-based Learning**  
   arXiv: https://arxiv.org/abs/2505.22949
