# 均值坍塌（Mean Collapse）Debug 记录

> 时间：2026-06-25  
> 文件：`stage1_train.py` / `stage1_inference.py`  
> 症状：推理输出所有笔画堆叠在画布中心，长度 0.5~2.6px（GT 均值应为 180px）

---

## 一、症状表现

```
Node 0: P0=[200.5 202.1], P3=[199.7 199.6], len=2.6px
Node 1: P0=[200.3 200.5], P3=[199.9 199.9], len=0.7px
Node 2: P0=[199.8 200.2], P3=[200.3 200.3], len=0.5px
```

所有节点坐标几乎相同，全部挤在画布中心 `(200, 200)` 附近。

---

## 二、根因分析（三层剥洋葱）

### 第一层根因：三节点嵌入完全相同

**旧代码（有问题）**：
```python
def forward(self, shapes, widths, coords, edge_types, edge_ts, padding_mask):
    x = self.shape_emb(shapes) + self.width_emb(widths) + self.coord_proj(coords)
```

推理时输入：
```python
b_shapes = torch.zeros(B, N, dtype=torch.long)    # 全 0
b_widths = torch.zeros(B, N, dtype=torch.long)    # 全 0
b_coords = torch.full((B, N, 4), 0.5, ...)        # 全 0.5
```

三个节点送入 `shape_emb(0) + width_emb(0) + coord_proj([0.5,0.5,0.5,0.5])` 得到**完全相同的初始嵌入** → Graph Transformer 无法区分节点 → 输出退化为同一坐标。

**这是训练-推理鸿沟**：模型训练时接收真实的 `shapes/widths/coords` 作为输入，但这些恰好是要预测的目标，推理时根本不知道这些值。

---

### 第二层根因：绝对坐标的均值坍塌

**数据分布**：
```
GT 坐标: mean=0.500, std=0.245, range=[0.104, 0.896]
```

字体数据关于画布中心对称，所有坐标均值恰好等于 `0.5`。

**MSE Loss 的数学特性**：
$$\min_{\hat{y}} \mathbb{E}[(y - \hat{y})^2] \Rightarrow \hat{y} = \mathbb{E}[y] = 0.5$$

即：**MSE 在数据均值处取得全局最优**。模型学会了预测常数 `0.5`，这是数学上的最优解，不是 Bug。

**训练过程验证**：
```
Epoch 1: coords std=0.027, 笔画长度≈0.02（归一化）
Epoch 2: coords std=0.009
...
Epoch 8: coords std=0.006  ← 方差持续收缩到 0
```

---

### 第三层根因（本质）：坐标没有唯一性

相同拓扑的字符可以出现在画布任意位置（"工"字可以在左上角也可以在右下角，拓扑完全相同）。**模型无法从拓扑中推断出绝对坐标**，L_node 的坐标 MSE 在驱使模型预测均值，L_junction 在坐标全为 0 时距离也为 0（loss 消失），两者合谋加速坍塌。

```
L_junction 的陷阱：
  当 coords → 0 时，pt_i = p0_i*(1-tu) + p3_i*tu → 0
  所有 pt_i ≈ pt_j ≈ 0，距离 dist → 0
  L_junction = 0，不产生任何梯度驱动坐标扩散
```

---

## 三、修复过程（演进记录）

### 尝试 1：分散节点初始化（治标）

```python
# 推理时将 N 个节点分散到画布不同位置
for i in range(N):
    x = (i + 1) / (N + 1)
    coords_list.append([x, 0.25, x, 0.75])
```

**效果**：节点坐标不同了，但模型从未在这种初始化下训练过，输出仍然混乱。

---

### 尝试 2：去掉节点 GT 输入 + TopoNodeEncoder（治本但不够）

**改动**：`forward` 签名不再接收 `shapes/widths/coords`，节点初始嵌入完全来自拓扑：

```python
def forward(self, edge_types, edge_ts, padding_mask):
    node_ids = torch.arange(N).unsqueeze(0).expand(B, N)
    x = self.node_id_emb(node_ids) + self.topo_node_enc(edge_types, edge_ts)
```

`TopoNodeEncoder` 从出边聚合信号：
```python
edge_feat = MLP(Emb(edge_type) || Linear(t_u, t_v, t_diff, t_prod))
node_emb_i = MeanPool_{j: type(i,j)>0} [ edge_feat_ij ]
```

**效果**：消除了训练-推理鸿沟，但均值坍塌依然存在（MSE + 数据均值=0.5 的问题）。

---

### 尝试 3：MSE → SmoothL1 + 提高 L_junction 权重

```python
# LAMBDA_JUNCTION: 10 → 50
loss_p0 = F.smooth_l1_loss(p0_pred[valid_nodes], gt_coords[:, :, :2][valid_nodes])
```

**效果**：坐标在 Epoch 1 有所扩散，但之后仍然持续收缩。SmoothL1 在偏差小时斜率=1，不根本改变均值最优解的性质。

---

### 尝试 4：Anti-Collapse Loss（L_spread）

```python
TARGET_VAR = 0.04
coord_var = valid_coords.var(dim=0)
loss_spread = F.relu(TARGET_VAR - coord_var).mean()
```

**效果**：`spread` 始终等于 `TARGET_VAR`（梯度被 Node Loss 的均值引力抵消），方差仍然收缩。

---

### 尝试 5：坐标解码器 Sigmoid → Tanh + GT 坐标重映射（治标）

```python
# decode_coords 最后一层
nn.Tanh()  # 输出 [-1,1]

# Loss 里 GT 映射
gt_coords_n = gt_coords * 2.0 - 1.0
```

**效果**：Tanh 在零附近梯度更健康，但本质问题（MSE 的均值最优解）没有解决。

---

### 最终方案：GT 坐标中心化 + 删除坐标 L_node ✅

**改动 1：Dataset 中心化坐标**

```python
# FontCompleteGraphDataset.__getitem__
all_pts = coords[:num_nodes].view(-1, 2)   # [2N, 2]
center = all_pts.mean(dim=0)               # 字形中心
coords[:num_nodes, 0:2] -= center          # p0 减去中心
coords[:num_nodes, 2:4] -= center          # p3 减去中心
```

中心化后：`mean=0.000, std=0.228, range=[-0.533, 0.580]`  
均值变为 0，消除了"全局最优=预测常数"的问题。

**改动 2：删除 L_node 中的坐标 Loss**

```python
# 正确做法：坐标完全由 L_junction 决定，不加 MSE/SmoothL1
loss_node = loss_shape + loss_width   # 不包含坐标项！
```

**理论依据**：坐标是由拓扑交点物理约束（L_junction）决定的，而不是需要"回归"的离散目标。加入坐标 MSE 会与 L_junction 产生竞争，而且在坐标坍塌时 MSE 的梯度会把坐标拉向均值，恰好让 L_junction 也消失。

**改动 3：Tanh 坐标解码器**

```python
self.decode_coords = nn.Sequential(
    nn.Linear(D_MODEL, 128),
    nn.GELU(),
    nn.Linear(128, 4),
    nn.Tanh()   # 输出 [-1,1]，与中心化坐标空间对齐
)
```

**验证结果**：
```
E1: pred std=0.199  笔画长度(px@400): [188, 178, 185, 234, 154]  ✅
E4: pred std=0.087  笔画长度(px@400): [83, 59, 72, 97, 66]       ✅
E8: pred std=0.072  笔画长度(px@400): [84, 44, 61, 97, 69]       ✅（稳定）
```

坍塌完全解决，笔画长度稳定在 40~100px 量级（合理范围）。

---

## 四、最终 Loss 设计

```
L = L_node + λ1*L_edge + λ2*L_junction + λ3*L_spread

L_node    = CE(shape) + CE(width)          ← 不含坐标！
L_edge    = CE(edge_type) + SmoothL1(t_u) + SmoothL1(t_v)
L_junction = ||B_i(t_u) - B_j(t_v)||^2   ← 坐标的唯一驱动力，λ=50
L_spread   = ReLU(TARGET_STD^2 - var(coords))  ← 兜底防坍塌，λ=5

超参：LAMBDA_JUNCTION=50, LAMBDA_SPREAD=5, TARGET_STD=0.20
```

---

## 五、推理时坐标还原

模型输出 Tanh 空间的中心化坐标 `∈ [-1, 1]`，对应字形中心偏移。  
可视化时还原：

```python
# coords_pred: Tanh 输出 [-1,1]，表示相对字形中心的归一化偏移
# 还原到 [0, 1] 再转像素
coords_px = (coords_pred + 1.0) / 2.0 * CANVAS_SIZE
# 注：画布中心 ≈ CANVAS_SIZE/2，坐标分布在中心附近
```

---

## 六、经验总结

| 经验 | 说明 |
|------|------|
| **MSE 的均值陷阱** | 当目标变量均值为常数时，MSE 的全局最优解就是预测该常数。字体坐标均值=0.5，MSE 必然坍塌 |
| **L_junction 的自我消除** | 当坐标坍塌到均值时，所有交点距离→0，L_junction 消失，失去驱动坐标扩散的能力 |
| **坐标中心化是关键** | 将均值强制变为 0，打破"预测常数"的全局最优解，是解决均值坍塌最有效的预处理 |
| **坐标不应该在 L_node 里** | 坐标是物理约束（拓扑交点）的结果，不是需要 MSE 回归的离散标签，放在 L_node 里是错的 |
| **SmoothL1/MSE 无法解决均值坍塌** | 换 Loss 函数只是治标，真正的问题是数据分布的对称性和 Loss 设计的逻辑错误 |
