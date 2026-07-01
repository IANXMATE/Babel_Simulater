# topology_novelty_filter_single.py 去重逻辑说明

## 结论先说

是的，当前去重实现**本质上就包含了你说的三层机制**，只是名字上稍微不同：

```text
随机生成 topology
→ 计算 strict_topo_hash
→ 如果 strict_topo_hash 已出现，直接丢弃

→ 计算 family_hash（内部包含 WL / Weisfeiler-Lehman 风格的拓扑族哈希）
→ 如果 family_hash 在当前 batch 或正样本库中过多，则限流 / 丢弃

→ 计算 geometry_hash（coarse_geo_hash 的实现版本）
→ 如果 geometry_hash 太像已有样本，再丢弃

→ 通过者才进入后续笔画与美感阶段
```

也就是说：

- 你的 **strict_topo_hash**：有
- 你的 **wl_hash / 拓扑族限流**：有，但在实现里叫 **family_hash**
- 你的 **coarse_geo_hash**：有，在实现里叫 **geometry_hash**

---

# 1. 整体流程

在飞轮脚本里，当前生成与过滤的大致流程是：

```text
PCG 随机生成 candidate
→ 人工规则复杂度过滤（stroke_count / connected_components / cycle_count 等）
→ novelty 去重模块
    → strict_topo_hash 检查
    → family_hash 检查
    → geometry_hash 检查
→ Stage1 topo 模型粗筛
→ Stage2 拓扑生成 / 拓扑补全
→ 后续笔画与美感阶段
```

注意：

- novelty 去重模块发生在 **Stage1 模型前**
- 这样做的目的是：
  1. 先把明显重复的 candidate 提前丢掉
  2. 避免模型反复处理大同小异的拓扑
  3. 提高样本多样性

---

# 2. 三层 hash 各自负责什么

---

## 2.1 strict_topo_hash：严格拓扑去重

### 目标
用于过滤“**几乎同一个拓扑**”的候选。

### 它依赖的核心信息
严格拓扑 hash 不只看“边数”，而是看更完整的拓扑结构摘要，例如：

- 节点度分布
- 骨架边连接关系
- 组件数
- 环数
- 一些拓扑事件统计
- skeleton 的 WL-style 拓扑摘要（更严格版本）

实现里会把这些信息组织成一个 `strict_obj`，然后再做 hash：

```text
strict_obj
→ sha1
→ strict_topo_hash
```

### 过滤逻辑
如果：

```text
strict_topo_hash ∈ 当前 batch 已见集合
```

则拒绝，原因类似：

```text
duplicate_strict_topology_in_batch
```

如果：

```text
strict_topo_hash ∈ 正样本历史缓存
```

则拒绝，原因类似：

```text
duplicate_strict_topology_in_positive_cache
```

### 作用
这一层最狠，专门干掉“拓扑上已经见过”的重复结构。

---

## 2.2 family_hash：拓扑家族 / WL 风格限流

### 目标
防止出现下面这种情况：

```text
虽然不是完全相同的 strict topology，
但本质属于同一类拓扑家族，
只是换了局部连接方式、轻微变体、旋转平移或一些小扰动。
```

### 它和 WL hash 的关系
你之前说的 `wl_hash`，在当前实现里并不是单独作为最终对外字段保存，而是**被吸收进 family_hash 的构造里**。

也就是说：

```text
WL / Weisfeiler-Lehman color refinement hash
→ 作为 family_hash 的核心组成部分之一
```

在代码里有：

```text
_wl_hash(nodes, edges, rounds=3 or 4)
```

然后构造：

- `strict_topo_hash`：更严格
- `family_hash`：更粗粒度，代表“拓扑家族”

### 过滤逻辑
family_hash 不一定直接“一刀切判死刑”，它主要承担**限流**作用。

当前逻辑支持两类限流：

#### A. 当前 batch 限流
如果同一个 `family_hash` 在当前生成 batch 中出现太多次，例如：

```text
batch_family_count >= max_family_per_batch
```

则拒绝，原因类似：

```text
topology_family_overflow_in_batch
```

#### B. 正样本历史库限流
如果某个 `family_hash` 在正样本缓存里已经很多，也可以限流：

```text
history_family_count >= max_family_in_history
```

则拒绝，原因类似：

```text
topology_family_overflow_in_positive_cache
```

### 作用
这层不是查“完全重复”，而是查“**同类拓扑是否已经太多**”。

它解决的是你非常在意的这个问题：

```text
很多生成结果不是一模一样，
但“似曾相识”，只是换个曲线样式、轻微扰动、旋转平移。
```

---

## 2.3 geometry_hash：粗几何重复过滤

### 目标
专门处理这种情况：

```text
拓扑不完全一样，
但几何轮廓非常像，
肉眼看起来还是高度雷同。
```

### 它依赖的核心信息
`geometry_hash` 不是精确几何匹配，而是**粗几何签名**，例如：

- 笔画长度统计
- 方向分布
- 宽高比
- 边界盒
- 端点分布
- 若干分桶后的几何摘要

实现里通过 `_geometry_hash(strokes, bins=24)` 得到。

你可以把它理解成：

```text
coarse_geo_hash 的工程版实现
```

### 过滤逻辑
如果：

```text
geometry_hash ∈ 当前 batch 已见集合
```

则拒绝，原因类似：

```text
near_duplicate_geometry_in_batch
```

如果：

```text
geometry_hash ∈ 正样本历史缓存
```

则拒绝，原因类似：

```text
near_duplicate_geometry_in_positive_cache
```

### 作用
这层是为了补 strict_topo_hash 的盲区。

因为现实中可能出现：

```text
拓扑略有不同
但整体几何观感仍然几乎一样
```

这时 strict topology 可能放过，但 geometry_hash 会拦住。

---

# 3. 为什么要三层，而不是一层

因为三层分别解决的是三种不同的“重复”：

---

## 第一层：strict_topo_hash
解决：

```text
完全或近乎完全相同的拓扑
```

---

## 第二层：family_hash
解决：

```text
不是完全一样，但属于同一拓扑家族的过量重复
```

---

## 第三层：geometry_hash
解决：

```text
拓扑可能不同，但视觉骨架太像
```

---

## 总结成一句话

```text
strict_topo_hash = 防“完全重复”
family_hash      = 防“同一家族刷屏”
geometry_hash    = 防“视觉骨架太像”
```

---

# 4. 当前实现不是“概率降低”，而是“规则拒绝 / 限流”

你之前写的是：

```text
如果 wl_hash 在当前池中过多，降低通过概率 / 限流
```

当前实现里，`family_hash` 的行为偏向于：

```text
达到阈值后直接 reject
```

而不是“软概率衰减”。

也就是说它更像：

```text
hard cap / hard throttle
```

不是：

```text
soft sampling probability decay
```

当然，未来可以改成软策略，比如：

```text
family_count 越多，通过概率越低
```

例如：

```python
pass_prob = exp(-alpha * overflow)
```

但**当前版本不是这么做的**，当前版本是更可控、更稳定的**硬限流**。

---

# 5. 正样本缓存读取范围

当前正样本缓存只读取：

```text
pcg_filebacked_stage2_schema/good
pcg_filebacked_stage2_schema/cleaned
AI_VECTOR_ROUTER_With_topo/annotations_topo
```

不会读取：

```text
pcg_filebacked_stage2_schema/bad
```

原因是：

- `good / cleaned / annotations_topo` 是“正向目标分布”
- `bad` 不适合作为“我要避免重复的优质库”
- bad 越多越好，但它不是你想维持新颖性的目标池

---

# 6. good 和 cleaned 冲突时怎么处理

在缓存层面，会读取 good / cleaned / annotations_topo 作为“正样本历史库”。

如果某个样本从 `good` 移到 `cleaned`，哪怕内容近似，**由于源文件集合发生变化**，缓存 manifest 会变化，因此会触发缓存重建。

也就是说，这种变化不会被忽略。

---

# 7. 缓存为什么能自动更新

缓存系统会记录 source manifest，大意相当于：

```text
所有源文件的 path + size + mtime_ns
→ 组成 source manifest
→ 再 hash 成 manifest_hash
```

只要以下任何事情发生：

- 新增 good
- 删除 good
- good → cleaned 移动
- cleaned → good 移动
- annotations_topo 有改动

都会让：

```text
manifest_hash 变化
```

从而自动触发缓存重建。

如果 source manifest 没变，则直接复用缓存 shard。

---

# 8. 当前 batch 内也会去重

除了和“历史正样本缓存”比较之外，当前实现还会维护 batch 内的集合：

- `batch_topo`
- `batch_geo`
- `batch_family`

所以它不只是“避免和历史重复”，还会避免“本轮生成自己内部互相重复”。

这点很重要，因为你当前最担心的问题之一就是：

```text
同一轮 Generate 里冒出很多似曾相识的结构
```

---

# 9. 现在的 novelty 模块输出什么信息

每个 candidate 在通过 novelty 检查后，会附带一个 `novelty_decision`，里面大致会有：

- accept / reject
- reason
- strict_topo_hash
- family_hash
- geometry_hash
- history_family_count
- batch_family_count
- 其他调试信息

这也是为什么飞轮界面后面能做调试展示。

---

# 10. 一个实际例子

假设随机生成了 candidate A。

---

## Step 1：计算 strict_topo_hash
如果 A 的严格拓扑在历史正样本库里已经存在：

```text
reject = duplicate_strict_topology_in_positive_cache
```

直接丢弃。

---

## Step 2：计算 family_hash
如果 A 的 family_hash 没有完全重复，但这类家族在当前 batch 已经出现太多次：

```text
reject = topology_family_overflow_in_batch
```

丢弃。

---

## Step 3：计算 geometry_hash
如果 A 虽然 strict topology 不完全相同，family 也还没超限，但整体粗几何和历史样本几乎一样：

```text
reject = near_duplicate_geometry_in_positive_cache
```

丢弃。

---

## Step 4：通过
只有全部通过，才会进入：

```text
Stage1 topo model
→ Stage2 topology completion
→ 后续笔画 / 美感
```

---

# 11. 对你目标的意义

你的目标不是“无穷多样但毫无控制”，而是：

```text
尽量保留好看的、合理的、可用的字符，
同时压制大同小异的拓扑反复出现。
```

这套三层去重机制正好对应这个目标：

- strict topology：保证“别老重复同一个骨架”
- family 限流：保证“别整个批次都是一个拓扑家族”
- geometry 去重：保证“别虽然拓扑不同但看起来还是差不多”

---

# 12. 当前实现与您设想的对应关系表

| 你设想的名字 | 当前实现里的名字 | 是否已实现 | 作用 |
|---|---|---:|---|
| strict_topo_hash | `strict_topo_hash` | 是 | 严格拓扑重复过滤 |
| wl_hash | `_wl_hash()`，并被吸收到 `family_hash` / `strict_topo_hash` 的构造中 | 是 | 拓扑结构族摘要 |
| current pool 过多限流 | `max_family_per_batch` | 是 | 当前生成批次的拓扑家族限流 |
| 历史库过多限流 | `max_family_in_history` | 是 | 历史正样本库的拓扑家族限流 |
| coarse_geo_hash | `geometry_hash` | 是 | 粗几何重复过滤 |

---

# 13. 当前版本与未来可扩展方向

当前版本是一个**高效、稳健、解释性强**的版本，适合飞轮脚本在线调用。

未来可以继续升级为：

---

## 13.1 soft limit / 概率衰减
把现在的：

```text
超过 family 数量阈值 → 直接拒绝
```

改成：

```text
family 越多 → 通过概率越低
```

---

## 13.2 learned novelty scorer
除了手工 hash 规则，还可以训练一个轻量模型做：

```text
candidate 与正样本库 embedding 距离
→ novelty 分数
```

---

## 13.3 shape-aware geometry signature
目前是粗几何 hash，以后可以加入更强的：

- stroke direction histogram
- normalized Laplacian spectrum
- graph edit distance approximation
- learned topology embedding

---

# 14. 最终一句话总结

当前去重实现可以概括为：

```text
strict_topo_hash 负责“完全重复”
family_hash(WL风格) 负责“同一家族过多”
geometry_hash 负责“视觉骨架太像”
```

它们一起工作，构成你飞轮生成前端的**高效 novelty prefilter**。

---

# 15. 相关脚本位置

核心库脚本：

```text
Babel_Simulater/dataset_analyse_p0/Char_Glyph_v0/annotation_tool/tool_func/topology_novelty_filter_single.py
```

飞轮调用脚本（带 novelty UI 和 Generate 报告）：

```text
Babel_Simulater/dataset_analyse_p0/Char_Glyph_v0/annotation_tool/annotation_flywheel_pcg_app_no_json_v9_topomodel_prefilter_stage1_v7_FIXED_2row_RELATIVE_CKPT_NOVELTY_REPORT.py
```
