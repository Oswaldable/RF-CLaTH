## 1. Agentic Memory 的核心思想

这里的 agentic 不是 LLM agent，也不是强化学习 agent，而是一个检索反馈闭环：

```text
Observation -> Action -> Feedback -> Memory -> Adaptation
```

在 RF-CLaTH 中对应为：

```text
Observation:
  T-SAS / PER-SAS 选择语义锚点，模型先决定重点看哪些帧。

Action:
  HashHead 生成 soft hash code / binary hash code。
  哈希码会直接决定 Hamming retrieval ranking。

Feedback:
  raw kNN、planned neighbors、actual retrieval trace、missed / false retrievals
  共同提供检索反馈。

Memory:
  online EMA memory bank 保存每个训练样本的历史 hash 状态。

Adaptation:
  L_semantic 建立 batch 内干净结构；
  L_memory 根据 self-calibrated retrieval feedback 修正全局 hash 几何；
  L_hash 保证二值码质量。
```

一句话：

```text
RF-CLaTH 不是只学习一个静态 hash 映射，
而是在 memory-bank 检索环境中生成 hash action，
再用检索反馈和历史记忆持续校正 hash 空间。
```

## 2. 最新三损失结构

### 2.1 总体数据流

```text
pre-extracted frame features
  -> FeatureProjector
  -> T-SAS semantic anchor selection
  -> slow semantic branch
  -> fast temporal branch
  -> content-time lateral fusion
  -> HashHead
  -> u_a, u_b
  -> L_semantic + L_memory + L_hash
```

`u_a`、`u_b` 是同一样本两个增强视图的 soft hash code：

```text
u_a, u_b in R^{B x K}
```

### 2.2 L_semantic

`L_semantic` 把原来的 `L_view` 和 `L_batch_neighbor` 合并为一个
weighted multi-positive InfoNCE。

核心想法：

```text
同一样本的另一增强视图是正样本；
raw-feature kNN 邻居也是正样本；
两类正样本共享同一个分母候选池。
```

记：

```text
U = [u_a; u_b] in R^{2B x K}
s_ij = cosine(U_i, U_j) / tau
```

正样本权重：

```text
alpha_ij = alpha_view      j 是 i 的 paired view
alpha_ij = alpha_neighbor  j 是 i 的 raw-kNN neighbor
alpha_ij = 0               其它
```

损失：

```text
L_semantic =
  mean_i [
      logsumexp_{k != i} s_ik
    - logsumexp_{j: alpha_ij > 0} (log alpha_ij + s_ij)
  ]
```

这样做的关键收益是避免旧结构里的 false-negative 冲突：

```text
旧结构:
  L_view 不知道 raw-neighbor 的存在，可能把 raw-neighbor 当负样本推开；
  L_batch_neighbor 又把它拉近。

新结构:
  paired view 和 raw-neighbor 都进入同一个正样本集合 P(i)，
  raw-neighbor 不再同时扮演正负两种角色。
```

当前最新脚本中：

```text
lambda_semantic = 0.8
view_positive_weight = 1.0
neighbor_positive_weight = 1.0
max_positive_weight = 2.0
```

### 2.3 L_memory

`L_memory` 是 agentic memory 的核心。它不是普通的 raw-only memory neighbor，
而是 self-calibrated memory-bank InfoNCE。

它有三层信号：

```text
1. raw positives:
   来自离线 raw-feature kNN 表，是稳定的静态邻居锚点。

2. planned / missed positives:
   来自 planner 与 actual retrieval trace 的差异。
   planned 是 planner 认为应该近的邻居；
   missed 是应该近但当前 hash 没检索到的邻居。

3. false hard negatives:
   actual retrieval 检索到了，但 planner 不认为应该近。
```

`L_memory` 是当前三损失里唯一使用 agentic feedback 的项：

```text
L_semantic:
  不吃 retrieval trace，保持 batch 内结构约束干净。

L_memory:
  吃 planned / actual / missed / false / trust / persistence。

L_hash:
  不吃 retrieval trace，只管 hash code 质量。
```

### 2.4 L_hash

`L_hash` 合并两个哈希码正则：

```text
L_quant:
  让 soft hash code 靠近 -1 / +1，方便二值化。

L_balance:
  让每个 bit 的 batch 均值接近 0，避免 bit 坍塌或长期偏一侧。
```

可以写成：

```text
L_hash =
  rho_q * L_quant
+ rho_b * L_balance
```

实现里仍然分别记录 `component_quant` 和 `component_bit_balance`，但方法解释中
它们归入同一个 `L_hash`。

## 3. Memory Bank 是什么

Memory bank 是一张全训练集级别的 hash 状态表：

```text
M:     [N, K]  每个训练样本的历史 soft hash 表示
valid: [N]     对应样本是否已经写入过 memory
```

其中：

```text
N:
  训练集样本数。

K:
  hash_bits，实验中通常跑 16 / 32 / 64。

M[j]:
  样本 j 的历史 hash 表示，使用 EMA 更新。
```

Memory bank 的作用是把 mini-batch 之外的历史样本也纳入检索环境。否则一个
batch 只能看到当前 256 个样本，无法形成全局近邻结构。

## 4. Query 与 Update

当前实现里，memory query 和 memory update 不是同一个口径。

### 4.1 Query

用于对 memory bank 做检索的 query 是两个视图分别参与：

```text
Q = normalize([u_a; u_b])
Q: [2B, K]
```

也就是说，每个样本有两条 query：

```text
u_a^i -> memory bank
u_b^i -> memory bank
```

### 4.2 Update

写回 memory bank 的当前观测使用两视图平均：

```text
current_i = normalize(0.5 * (u_a^i + u_b^i))
```

如果样本 i 之前没有写入过：

```text
M[i] = current_i
valid[i] = true
```

如果已经写入过：

```text
M[i] = normalize(momentum * M[i] + (1 - momentum) * current_i)
```

当前默认 `momentum=0.9`，所以 memory 是慢变的历史状态，而不是被单步噪声
直接覆盖。

## 5. Raw Memory Positives

raw positives 来自离线 raw-feature kNN 表：

```text
R(i) = raw-feature nearest neighbors of sample i
```

最新脚本中：

```text
memory positives per anchor = 3
raw trust top-k = 5
```

含义：

```text
用于 memory 正样本的 raw neighbors:
  取 R(i) 的前若干个有效 memory entries。

用于 trust observation 的 raw neighbors:
  用 top-5 raw neighbors 检查 actual retrieval 是否命中已知邻居。
```

raw positives 是 `L_memory` 的稳定下界：

```text
当 trace 不可信或尚未启用时，
L_memory 退回 raw-only memory InfoNCE。
```

## 6. Planner 与 Actual Trace

最新 self-calibrated memory 使用 planner 和 actual retrieval trace。

### 6.1 Planned

planner 给每个 anchor i 生成 planned neighbors：

```text
planned(i):
  planner 认为应该检索到的邻居。
```

planner 评分由三类信息组合：

```text
s_final(i,j) =
  omega_s * s_sem(i,j)
+ omega_t * s_temp(i,j)
+ omega_z * s_hash(i,j)
```

当前 latest 脚本中，trace 启用后：

```text
omega_s = 0.45
omega_t = 0.25
omega_z = 0.30
```

warmup 阶段：

```text
omega_s = 0.65
omega_t = 0.35
omega_z = 0.0
```

这表示早期先不让 hash 状态参与 planner；trace 启用后再把当前 hash 空间状态
纳入规划。

### 6.2 Actual

actual trace 是当前 hash code 在 memory bank 上实际检索到的邻居：

```text
actual(i):
  query u_i 在 memory bank 上按 hash 相似度取 top-r 的结果。
```

当前 latest 脚本中：

```text
top_r = 20
actual_trace_start_epoch = cut = 35
hard_mining_start_epoch = cut = 35
```

也就是说，第 35 个 epoch 开始，actual retrieval trace 和 hard mining 同时进入
memory self-calibration。

## 7. Missed 与 False

有了 `planned(i)` 和 `actual(i)`，就能得到两类 agentic feedback：

```text
missed(i) = planned(i) - actual(i)
false(i)  = actual(i)  - planned(i)
```

含义：

```text
missed:
  planner 认为应该近，但当前 hash 没检索到。
  这是 hard positive，应该拉近。

false:
  当前 hash 检索到了，但 planner 不认为应该近。
  这是 hard negative，应该推开。
```

直觉：

```text
missed 补课；
false 纠错。
```

`missed / false` 不是单独形成第四个或第五个损失，它们只进入 `L_memory`，
改变 memory InfoNCE 中正样本和分母候选的权重。

## 8. Trust：为什么叫 Self-Calibrated

不是所有 actual trace 都可信。训练早期或者某些难样本上，当前 hash 检索可能
很差，直接相信 missed / false 会把噪声写进监督。

因此每个 anchor 有一个信任度：

```text
g_i = EMA( |actual(i) intersect R(i)| / |R_top(i)| )
```

其中：

```text
R_top(i):
  用于校准的 raw-neighbor top-k，当前 latest 脚本默认 top-5。
```

直觉：

```text
如果当前 hash 连 raw-feature 邻居都检索不到，
说明 actual trace 不可信；
planned / missed / false 的权重要降低。

如果当前 hash 能命中一部分 raw-feature 邻居，
说明 actual trace 有参考价值；
agentic feedback 可以逐步变强。
```

这就是 self-calibrated 的含义：不是按固定 epoch 调度盲目相信反馈，而是用
每个样本自己的检索表现来决定信多少。

## 9. Edge Persistence：边持续性

单步 missed / false 可能只是噪声。为了避免某一次偶然检索结果直接变成强监督，
代码维护了边级持续性：

```text
positive edge bank:
  planned / missed 边的持续性。

negative edge bank:
  false 边的持续性。
```

持续多次出现的边权重更高；只出现一次的边会被 EMA 衰减。

这对应公式中的：

```text
m_ij:
  正反馈边的持续性。

f_ij:
  hard negative 边的持续性。
```

## 10. L_memory 的完整形式

记：

```text
V:
  valid memory entries。

s(i,j):
  query i 与 memory entry j 的相似度 logit。

R(i):
  raw-neighbor positives。

P(i):
  memory 正样本集合。
```

正样本权重：

```text
a_ij = alpha_raw                       j in R(i)
a_ij = alpha_plan * g_i * m_ij         j in planned(i)
a_ij = alpha_miss * g_i * m_ij         j in missed(i)
```

分母 hard-negative 权重：

```text
d_ij = 1                                      默认
d_ij = 1 + (gamma_false - 1) * g_i * f_ij     j in false(i)
```

memory loss：

```text
L_memory =
  mean_i [
      logsumexp_{j in V}    (log d_ij + s(i,j))
    - logsumexp_{j in P(i)} (log a_ij + s(i,j))
  ]
```

当前 latest 脚本默认权重：

```text
alpha_raw = 1.0
alpha_plan = 0.5
alpha_miss = 1.25
gamma_false = 1.10
trust_momentum = 0.9
edge_momentum = 0.9
planned_positive_topk = 5
missed_positive_topk = 5
```

关键退化性质：

```text
trace 未启用 或 g_i 接近 0 时：
  planned / missed / false 的有效权重接近 0；
  L_memory 退回 raw-only memory InfoNCE。
```

这保证 agentic feedback 只在可相信时进入。

## 11. 三损失之间的边界

最新实验的分工是：

```text
L_semantic:
  管 batch 内方向/夹角结构。
  候选池是当前 batch 的两视图。
  不吃 actual trace。

L_memory:
  管全局 memory 几何。
  候选池是 valid memory bank。
  唯一吃 planned / actual / missed / false / trust / persistence。

L_hash:
  管二值码质量。
  不处理邻居图，也不吃 retrieval trace。
```

三项来源关系是：

```text
view consistency + batch raw-neighbor supervision -> L_semantic
memory-bank retrieval feedback                         -> L_memory
quantization + bit balance                             -> L_hash
```