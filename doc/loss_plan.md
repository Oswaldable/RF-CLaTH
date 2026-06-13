下面把前几轮的结论收敛成一份完整、自洽的损失设计方案。前向结构（T-SAS/PER-SAS + SelectedClassAttention + BidirectionalMamba + ContentTimeLateralFusion + HashHead）一行不动，损失从 5 项降到 3 项。

## 总损失（先给结论）

```text
L_total = λ_sem · L_semantic + λ_mem · L_memory + λ_hash · L_hash

         ≈ 0.80 · L_semantic + 0.04 · L_memory + 0.05 · L_hash
```

三项的划分不是按"信号种类"切的，而是按**优化几何 + 候选池/时标 + 是否吃 agentic 反馈**三条正交的轴切的：

```text
L_semantic : 管"方向/夹角"，batch 候选池，干净结构约束（不碰反馈）
L_memory   : 管"全局几何"，EMA memory 候选池，慢时标，唯一吃 agentic 反馈
L_hash     : 管"码向量幅度"，逐元素 + 逐 bit 正则
```

统一记号：`B`=batch，`K`=hash_bits，`u_a,u_b ∈ R^{B×K}` 为两视图 tanh 软码，`ũ=u/‖u‖`，`τ=0.2`。

## 1. L_semantic —— 合并 L_view + L_batch_neighbor

利用 §15 里 view ⊂ batch_neighbor 的天然包含关系，把两套正样本并进同一个加权 multi-positive NT-Xent。

```text
U = [u_a; u_b] ∈ R^{2B × K}          # 两视图堆叠
s_ij = (ũ_i · ũ_j) / τ
A(i) = {1..2B} \ {i}                  # 分母候选

正样本权重：
  α_ij = α_v    j 是 i 的 paired view（同视频另一增强）
  α_ij = α_n    j（任一视图）是 i 的 raw-kNN 邻居（含 symmetric）
  α_ij = 0      其它
P(i) = {j : α_ij > 0}

L_semantic =
  (1/2B) Σ_i [
      logsumexp_{k∈A(i)} ( s_ik )
    − logsumexp_{j∈P(i)} ( log α_ij + s_ij )
  ]
```

权重映射：原 `0.30·L_view + 0.50·L_batch_neighbor` → 外层一个 `λ_sem`，比例搬进内部 `α_v : α_n = 0.6 : 1.0`。合并的额外收益是消掉了原设计的 false-negative 冲突——以前 `L_view` 不知道邻居存在、会把邻居当负样本推开，而 `L_batch_neighbor` 同时在拉近；共享分母后，邻居进了 `P(i)` 就自动退出负样本角色。

## 2. L_memory —— agentic 自校准 InfoNCE（方案核心）

这是唯一被改造的项。设计目标：让"信任反馈"从 v2 那种**按 epoch 排程的 β**，变成**按 anchor 测出来的门控**。

**缓存来源（不变）**：边来自离线静态表 `cache/repartition_*_train_rawmean_top20.pt`（视频级 mean-pool 后的 raw 余弦 kNN，无标签）；坐标来自在线 EMA memory bank `M ∈ R^{N×K}`。改造不动这两套，只改"反馈新边相对静态边的权重"。

```text
q_i = normalize(0.5(u_a^i + u_b^i))    # 与 memory 更新口径一致
s_ij = (q_i · M_j) / τ
V = valid memory entries
R(i) = raw-kNN top-k                    # 唯一可验证的"标注集"

# 信任标量：用模型在已知邻居上的召回率，跨步 EMA 平滑
g_i ← μ_g · g_i + (1-μ_g) · |Actual(i) ∩ R(i)| / |R(i)|

# 边持续性：单步 missed 多为噪声，连续 missed 才采纳
m_ij ← μ_m · m_ij + (1-μ_m) · 1[j ∈ Missed(i) 本步]

正样本权重 a_ij：
  a_ij = α_raw                    j ∈ R(i)         # 静态锚点，恒信
  a_ij = α_plan · g_i · m_ij      j ∈ Planned(i)   # trust + 持续门控
  a_ij = α_miss · g_i · m_ij      j ∈ Missed(i)    # hard positive
P(i) = R(i) ∪ Planned(i) ∪ Missed(i)

负样本权重 d_ij：
  d_ij = 1                                          # 默认
  d_ij = 1 + (γ_false − 1) · g_i · m_ij   j ∈ False(i)   # hard negative

L_memory =
  (1/B) Σ_i [
      logsumexp_{j∈V}    ( log d_ij + s_ij )
    − logsumexp_{j∈P(i)} ( log a_ij + s_ij )
  ]
```

**关键性质——优雅退化**：当 `g_i → 0`（trace 不可信），`α_plan·g`、`α_miss·g` 全归零、`d_false → 1`，`L_memory` 精确退回当前稳定的 raw-only `L_memory_neighbor`。也就是说它是现有损失的**严格推广，下界就是 baseline**，从机制上堵死了 v2 "可能比 0.0994 还低"的风险。同时它不再需要 `start_epoch / β ramp`，注入率是 per-anchor 自调的。

## 3. L_hash —— 合并 L_quant + L_balance

两者作用对象不同（逐元素幅度 vs 逐 bit 一阶矩），但能写进同一行矩阵范数。

```text
L_hash =
  (1/2) Σ_{v∈{a,b}} [
      ρ_q · ‖ |U_v| − 1 ‖²_F / (B·K)        # quantization：靠近 ±1
    + ρ_e · ‖ (1/B) · U_vᵀ · 1_B ‖²₂ / K     # bit balance：列均值去偏
  ]
```

权重映射：`ρ_q : ρ_e = 2 : 3`（对应原 0.02 : 0.03），外层 `λ_hash ≈ 0.05`。

**为什么 L_semantic 和 L_hash 不能再合**：前者作用在 `ũ`（scale-invariant，管角度），后者作用在 `|u|`（管半径），数学上正交，求和才有意义、写成一条只是凑加号。这就是 3 项的下界。

## 4. 冷启动等价性（最重要的安全保证）

把新旧权重对齐看，在 `g_i=0, m_ij=0` 的冷状态下，新总损失**精确等于**当前 Stage1 original：

```text
旧 5 项                          新 3 项（冷状态 t=0）
─────────────────────────────────────────────────────
0.30 L_view          ┐
                     ├─ 0.80 L_semantic   (α_v:α_n=0.6:1.0)
0.50 L_batch_neighbor┘
0.04 L_memory_neighbor → 0.04 L_memory     (g=0 ⟹ raw-only，逐项相等)
0.02 L_quant         ┐
                     ├─ 0.05 L_hash        (ρ_q:ρ_e=2:3)
0.03 L_balance       ┘
─────────────────────────────────────────────────────
t=0 时 L_total ≡ Stage1 original，下界 = 0.0994
```

这正是 §22 反复确认的最可靠基础损失。新设计只是给了它一个 `g>0` 时才打开的 agentic 上行通道。

## 5. 调度：只剩一个诚实的旋钮

不再对任何 `λ` 做 epoch 排程，agentic 动态全部活在 `g_i / m_ij` 里。唯一保留的调度是 **trace 何时开始计算**：

```text
Phase A (epoch 1 ~ T_w):  不算 retrieval trace，g_i 恒 0，L_memory = raw-only
                          ≡ Stage1 original，精确复刻已验证的 warmup
Phase B (epoch T_w+):     开始算 trace，g_i / m_ij 开始累积，反馈 per-anchor 自门控
```

这和 v2 的 `β(epoch)` 有本质区别，必须讲清楚：`β` 控制的是"信多少"——这件事应当数据自适应，写死就是 v2 失败的根因；而 `T_w` 控制的是"trace 此刻有没有意义算"——在哈希空间还是随机的时候 retrieval trace 本就没有信号可校准 `g`，这是**信号可用性门控**，不是信任门控。`T_w` 之后的信任校准仍然完全自驱。文档证据（warmup60 hard-switch = 0.1036 是 agentic 方向最好）支持 `T_w ≈ 60` 作为先验。

## 6. 超参合并清单

```text
全局：    τ = 0.2

L_semantic：
  λ_sem = 0.80,  α_v:α_n = 0.6:1.0

L_memory（含 agentic）：
  λ_mem = 0.04
  α_raw = 1.0,  α_plan = 0.5,  α_miss = 1.25,  γ_false = 1.10
  μ_g = 0.9,  μ_m = 0.9
  R(i) = raw top-k（建议 top10），Planned/Missed 各取 top5
  T_w ≈ 60（trace 起始）

L_hash：
  λ_hash = 0.05,  ρ_q:ρ_e = 2:3
```

相比原设计，砍掉了 `memory_neighbor.start_epoch / positives_per_anchor / include_self` 以及 v2 的 `β ramp 区间`，新增的只有两个 EMA 率 `μ_g, μ_m` 和 trust 相关的 `g/m` 机制——净复杂度下降，且新增项都是 per-anchor 自调的鲁棒量。

## 7. 边界（明确不纳入）

```text
agentic 反馈只进 L_memory 一个通道；L_semantic / L_hash 全程不碰 trace
  —— 延续 §21 source-factored 教训，强结构约束不被噪声污染

前向结构、memory bank、raw kNN 表全部不变（§11 边界）

不引入 hash center / prototype / reconstruction / trainable selector（§12）
```

---

一句话总结这套方案：**5 项 → 3 项不是做减法，而是按几何/候选池/反馈三轴重新归位**——`L_semantic` 把同源对比的三层嵌套压成一个带原理的加权 NT-Xent，`L_hash` 把两个微权正则并成一条矩阵范数，`L_memory` 从"微权辅助项"升级成"冷启动严格等价于 baseline、`g>0` 才上行的自校准 retrieval-feedback 接口"。整体下界被钉死在已验证的 0.0994，上行空间交给可测的信任而非排程。