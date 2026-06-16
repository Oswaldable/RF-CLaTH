把 agentic memory 从最底层的存储讲到最终的损失，每个符号在哪一步出现、是什么形状、谁更新它、为什么需要它,我按数据流的顺序一层层往上搭。

## 第 0 层：记号约定（先把字母固定下来）

```text
N         : 训练集样本总数（HMDB16 上就是训练视频条数）
B         : 一个 mini-batch 的样本数（日志里 batch_size=256）
K         : hash 位数 hash_bits（本实验 = 16）
i         : anchor 样本的下标（"当前这条样本"）
j         : 候选样本的下标（"可能是邻居的那条"）
u_a, u_b  : 同一样本两个增强视图的 soft hash code，形状 [B, K]，值域 (−1, 1)
            （来自 HashHead 的 tanh 输出，a/b 是两个 augmentation view）
τ         : 温度 temperature（≈0.2），控制 InfoNCE 里相似度的尖锐程度
```

这几个先记住，后面所有公式都由它们拼出来。

## 第 1 层：Memory Bank —— 整套机制的地基

**它是什么**：一张表，记录**每个训练样本"此刻在 Hamming 空间的位置"**。

```text
符号        形状        含义
─────────────────────────────────────────────────────────────
memory     [N, K]      每一行 memory[i] 是样本 i 当前的 hash 表示向量（已 L2 归一化）
valid      [N]         布尔/0-1 向量，valid[i]=1 表示样本 i 这一行已经被填过、可用
```

**为什么需要它**：一个 mini-batch 只有 B=256 个样本，但你想让样本 i 和**全数据集 N 个**样本里的语义邻居对齐。不可能每步把 N 个样本都前向一遍（太贵）。所以用一张表把"每个样本最近一次的 hash 码"缓存下来，让 batch 内的样本能去和**batch 外的历史坐标**做对比。memory bank = 用一张缓存表近似"全局检索池"。

**谁更新它、怎么更新**（这就是 §16 那条 EMA）：

```text
每个 step，对 batch 里出现的样本 i：
  current_i = normalize( 0.5 * (u_a^i + u_b^i) )      # 两视图平均后归一化，作为"本步观测"
  memory[i] = normalize( m * memory[i] + (1−m) * current_i )   # 与历史值做指数滑动平均
              其中 m = momentum ≈ 0.9
  valid[i]  = 1
```

逐项解释这条更新：
- `0.5*(u_a+u_b)`：把同一样本两个视图的码取平均，得到一个更稳的单一表示。
- `momentum m=0.9`：新观测只占 10%，旧值占 90%。这让 memory bank **慢变**——它不会因为某一步的抖动剧烈跳动，而是平滑地跟随模型。这就是为什么把它叫"慢时标"通道。
- `valid`：训练刚开始时大多数行还没被填（日志里 ep1 的 `valid=0.81`），随着 batch 轮换逐渐填满（ep3 → `valid=1.0`）。算损失时只在 `valid=1` 的行上做。

一句话：**memory bank 是模型 hash 码的一份滚动快照，提供"batch 外的全局检索坐标"。**

## 第 2 层：两类"邻居来源"——坐标 vs 边

要做对比学习，光有坐标（谁在哪）不够，还得知道**谁该和谁靠近**（边）。agentic memory 用到三样东西，务必分清：

```text
名称              类型        提供什么          来源
──────────────────────────────────────────────────────────────────────
memory bank      坐标        每个样本在哪      模型 hash 码的 EMA（第1层）
raw kNN 表 R(i)   边（静态）   谁该是邻居        离线算一次的 rawmean 余弦近邻
planner 图        边（动态）   谁该是邻居        三通道打分（第4层）
```

```text
符号        形状/含义
────────────────────────────────────────────────────────────
R(i)        样本 i 的 raw-feature 最近邻集合（top-k）
            来自缓存 cache/repartition_*_train_rawmean_top20.pt
            "rawmean" = 把每个视频的预提取帧特征 mean-pool 成一个视频级向量，
                        再做余弦 kNN 取 top20。不使用类别标签，是纯特征代理图。
```

`R(i)` 是**唯一可被验证的"参考答案"**——后面 `g_i` 信任度就是拿它当标尺。普通 `L_memory_neighbor` 只用 memory 坐标 + `R(i)` 这条静态边。agentic 版本多引入了一个会变的 planner 图。

## 第 3 层：把对比的"相似度"定义出来

无论 raw 还是 agentic，核心都是在 memory bank 上算相似度：

```text
q_i = 样本 i 的查询向量（用它的当前 hash 码）
s(i, j) = ( q_i · memory[j] ) / τ        # 点积 / 温度 = logit

直觉：q_i 和 memory[j] 越像，s 越大，InfoNCE 越倾向把它们当"靠近"。
```

> 注意一个上几轮认账过的细节：当前实现里 query 用的是 `u_a` 和 `u_b` 两个视图**分别**去对 memory（共 2B 条 query），只有 memory 的 **update** 用平均码。我之前误把 query 也写成平均，会导致 query 数从 2B 变 B。正确口径是 `queries=[u_a; u_b]`、`update=0.5(u_a+u_b)`。

## 第 4 层：Planner —— "理想邻居图"是怎么规划出来的

**它是什么**：一个打分器，对每个 anchor i 给所有候选 j 打分，取 top-M 作为"计划邻居 `planned(i)`"。打分是**三通道线性组合**，对应日志里的 `omega_*` 和 `p_*_topm`：

```text
s_final(i,j) = ω_s · s_sem(i,j) + ω_t · s_temp(i,j) + ω_z · s_hash(i,j)

符号        含义                              日志值      作用
──────────────────────────────────────────────────────────────────────────
s_sem       raw/rawmean 内容特征相似度         —          "内容像不像"
ω_s         语义通道权重                       0.650      主导通道
s_temp      时间结构亲和度                     —          "时间上配不配"
ω_t         时间通道权重                       0.350
s_hash      在【当前学到的 hash 空间】的相似度  —          ★让 planner"看见"模型当下状态
ω_z         hash 通道权重                      0.000      ★命门：当前为 0，眼睛被蒙上
top_m       每个 anchor 取多少计划邻居          20

planned(i) = top_M of candidates by s_final(i, ·)
```

**`s_hash` 与 `ω_z` 是理解 agentic 与否的关键**：`s_hash` 是唯一把"模型现在学成什么样"反馈进规划的通道。`ω_z=0` 等于让 planner 完全无视 hash 空间，于是 `planned` 退化成 raw 内容+时间的固定图。

日志里几个诊断字段就是 planner 质量的体检：

```text
p_s_topm ≈ 0.76   语义通道把 planned 排前的质量
p_t_topm ≈ 0.83   时间通道的
p_z_topm ≈ 0.71   hash 通道的（算出来了，但 ω_z=0 没用上）
p_final_topm≈0.78  最终组合的
p_random ≈ 0.55   随机 anchor 基线（说明通道都在随机之上）
overlap_final_s≈0.92  planned 与"纯语义 top-M"的重合度 → planned≈语义图
label_prec ≈ 0.34  planned 里真同类的比例（只有 1/3）
valid/z_valid     planner 用的 bank 里有效条目占比
```

## 第 5 层：actual —— 模型"实际"检索到谁

```text
actual(i) = 用样本 i 的当前 hash 码 q_i，在 memory bank 上按相似度取 top-k
含义：如果现在拿这个码去真做检索，实际会捞回哪些样本。
```

planner 给的是**应该**捞回谁（planned），actual 是**实际**捞回谁。agentic 的全部增量信息都来自这两者的差。

## 第 6 层：四个集合 —— agentic 的灵魂（§18）

这是 agentic 区别于普通 memory loss 的唯一地方，全部由 planned 和 actual 做集合运算得到：

```text
符号                     定义                  角色            直觉
──────────────────────────────────────────────────────────────────────────────
planned(i)               planner 计划的邻居     —              "该近的"
actual(i)                实际检索到的邻居       —              "实际近的"
missed(i)=planned−actual 该近却没检索到         hard positive   "模型的盲点"，用力拉
false(i) =actual−planned 检索到却不该近         hard negative   "模型的错误"，用力推
```

记忆口诀：**missed 补课，false 纠错**。理想下，用 missed 把漏掉的正确邻居拉回来、用 false 把学错的虚假邻居推开，hash 检索就逐步逼近 planner 的理想图。

## 第 7 层：损失装配 —— 这些集合怎么变成一个数（§21）

agentic 信号**只进 memory 一个通道**（view/batch 保持独立 InfoNCE 不动）。memory 通道劈成 raw 和 feedback 两支，用 β 混合：

```text
L_memory_agentic = (1−β) · L_memory_raw + β · L_memory_feedback

符号    含义
──────────────────────────────────────────────────────────
β       feedback 混入比例。调度：
        ep1–60 β=0（纯 raw，等于 base）；ep61–80 β:0→0.25；ep80+ β=0.25 并开 hard mining
```

`L_memory_feedback` 本身是一条 memory-bank 上的 multi-positive InfoNCE：

```text
正样本 P(i) = R(i) 的 raw 正样本  ∪  planned(i) top5  ∪  missed(i) top5
hard neg   = false(i)
每类带权重 a_ij / d_ij：
  a_raw=1.0,  a_planned=0.5,  a_missed=1.25       （正样本，进分子）
  d_false=1.10                                    （hard neg，进分母加重）

L_feedback_i =
    logsumexp_{j∈valid}  ( log d_ij + s(i,j) )      # 分母：所有有效候选竞争
  − logsumexp_{j∈P(i)}   ( log a_ij + s(i,j) )      # 分子：各类正样本按权聚合
```

权重表透露设计意图：missed=1.25 比 raw 正样本=1.0 更用力（盲点最该补）；planned=0.5 压低（和 raw 大量重叠，避免重复计数）；false=1.10 温和推开。

## 第 8 层（我上几轮加的）：自校准符号

把"信多少"从固定 β 改成 per-anchor 测量出来的门控：

```text
符号    形状/定义                                   含义
──────────────────────────────────────────────────────────────────────────
g_i     |actual(i) ∩ R(i)| / |R(i)|，再做跨步 EMA    信任标量：模型在【可验证的
                                                     raw 邻居】上的召回率。
                                                     g 高 = 连已知邻居都查得准
                                                          → 它对未知候选的判断可信
p_ij    planned/missed 边的持续性 EMA                单步 missed 多是噪声，连续
                                                     missed 才采纳
f_ij    false 边的持续性 EMA                          false 自己的持续性信号
门控    planned/missed 权重 ×g_i×p_ij，
        false 权重 ×g_i×f_ij
退化    g_i→0 时反馈项全归零 → 整条退回纯 raw memory
```

## 最后：把符号串成一句话的数据流

```text
模型出码 u_a,u_b
  → EMA 写进 memory bank（坐标，[N,K]，慢变）
  → planner 用 ω_s·s_sem + ω_t·s_temp + ω_z·s_hash 打分，取 top_M = planned（理想边）
  → 当前码 q_i 在 bank 上检索 top_k = actual（实际边）
  → missed=planned−actual（补课正样本），false=actual−planned（纠错负样本）
  → 连同 raw 正样本 R(i) 一起，按权重 a/d 装进 memory InfoNCE = L_memory_feedback
  → 与 L_memory_raw 按 β（或 g_i 门控）混合 = L_memory_agentic
```

而这套机制本应是个闭环——**模型的检索行为(actual) 反过来塑造下一步的监督(planned)**——但 `ω_z=0` 把 `s_hash` 这条反馈支路剪断，planned 退化成 raw 图的固定复述，missed/false 也就不再携带 raw memory 之外的信息。这就是为什么日志里它空转：所有符号都在转，但反馈回路没接通。要让这些符号真正"活"起来，第一步是把 `ω_z` 设成非零，让 `s_hash` 通道把模型状态喂回 planner。