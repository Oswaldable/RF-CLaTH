# Agentic Fusion 详细讲解

本文把 agentic fusion 当作一个完整机制来讲：它解决的不是“再加一个损失项”
这么简单，而是让检索反馈进入表示融合层，决定 slow 语义分支和 fast 时序分支
在每个样本上应该如何分工、如何被训练、如何共同形成最终哈希码。

核心思想可以概括为：

```text
检索反馈不只告诉模型“应该学什么邻居结构”，
还告诉模型“这条样本应该更依赖内容语义，还是更依赖时序动态”。
```

也就是说，agentic fusion 把 RF-CLaTH 的反馈闭环从 loss 层推进到 fusion 层：

```text
hash action -> retrieval feedback -> branch attribution
            -> adaptive fusion / branch routing -> next hash action
```

## 1. 从普通融合到 Agentic Fusion

视频哈希里的两个分支天然承担不同角色：

```text
slow branch:
  处理 T-SAS 选出的语义锚点。
  更偏内容、场景、对象、动作关键状态。

fast branch:
  处理全帧或剩余帧的时序动态。
  更偏运动轨迹、动作过程、局部变化。
```

普通融合通常只做一件事：

```text
z = fuse(h_s, h_f)
u = HashHead(z)
```

这里 `h_s` 是 slow 分支输出，`h_f` 是 fast 分支输出，`u` 是最终 soft hash code。
这种融合是静态的：同一个融合规则作用于所有样本，不关心某个样本的检索错误
到底来自语义分支不足，还是时序分支不足。

Agentic fusion 的变化是：

```text
先让模型生成 hash code 并产生 retrieval trace，
再根据检索反馈判断每个样本更需要哪条分支负责，
最后用这个判断反过来调节 slow / fast 的融合和训练强度。
```

所以 agentic fusion 的重点不是“把两个向量拼起来”，而是建立一个可反馈的
分支路由机制。

## 2. Agentic Fusion 的闭环

一个样本 `i` 在训练中经历下面的闭环：

```text
1. Observation:
   T-SAS 从视频中选择语义锚点，slow branch 观察关键内容；
   fast branch 观察完整或剩余时序动态。

2. Action:
   slow / fast 共同生成 hash code。
   这个 hash code 会决定 Hamming retrieval ranking。

3. Feedback:
   raw-neighbor、memory-neighbor、planner trace 或 actual retrieval trace
   告诉模型哪些样本应该近、哪些样本被漏检、哪些样本被误检。

4. Attribution:
   把检索错误归因到 slow 或 fast：
   是内容语义没抓住，还是时序动态没建好？

5. Routing:
   根据归因结果调节融合门控，让更可靠或更需要补强的分支获得更大训练压力。

6. Update:
   通过融合门控损失、分支互补损失和哈希损失共同更新模型。
```

这就是 agentic 的地方：模型不是被动接收一个固定融合策略，而是根据自己的检索
行为和反馈，动态调整下一轮表示构造方式。

## 3. 双段哈希码

为了让 slow 和 fast 都有清晰职责，agentic fusion 使用双段哈希码：

```text
u_s_i = hash_head_s(h_s_i)  in R^{K_s}
u_f_i = hash_head_f(h_f_i)  in R^{K_f}
u_i   = [u_s_i ; u_f_i]     in R^K
K     = K_s + K_f
```

含义：

```text
u_s:
  slow semantic code，主要承载内容、对象、场景、关键动作状态。

u_f:
  fast temporal code，主要承载运动、顺序、局部变化和动态模式。

u:
  最终检索码。评估时仍然直接使用完整 K 位二值码做标准 Hamming 检索。
```

这种设计的好处是：slow 和 fast 不再只是融合成一个不可解释的向量，而是各自
拥有一段可单独分析、可单独监督、可单独归因的哈希子码。

## 4. 逐样本融合门控

双段码本身只是结构基础。真正的 agentic fusion 在于给每个样本一个门控：

```text
a_i in [0, 1]
```

`a_i` 表示训练时 slow 段和 fast 段在相似度计算中的权重：

```text
sim_train(i, j) =
    a_i       * <u_s_i, u_s_j> / K_s
  + (1 - a_i) * <u_f_i, u_f_j> / K_f
```

直觉：

```text
a_i 接近 1:
  样本 i 的检索更依赖 slow semantic code。
  训练会更强调内容语义段的相似度。

a_i 接近 0:
  样本 i 的检索更依赖 fast temporal code。
  训练会更强调时序动态段的相似度。

a_i 在中间:
  内容和时序都重要，两段共同塑形。
```

这个门控只用于训练期的反馈建模和对比目标。最终评估仍然使用完整拼接码 `u_i`
做标准 Hamming 检索，避免让评估协议依赖样本级动态权重。

## 5. 门控输入：不仅看特征，也看反馈

门控 `a_i` 可以由一个小 MLP 生成：

```text
a_i = sigmoid(MLP([h_s_i ; h_f_i ; c_i]))
```

其中 `c_i` 是检索反馈上下文：

```text
c_i = [q_sem_i, q_temp_i, g_i]
```

各项含义：

```text
q_sem_i:
  语义通道在样本 i 上的邻居质量。
  可以理解为：内容语义线索对这个样本是否可靠。

q_temp_i:
  时序通道在样本 i 上的邻居质量。
  可以理解为：运动/时间线索对这个样本是否可靠。

g_i:
  当前样本的检索信任度。
  如果当前 hash trace 本身还不可信，反馈就应该弱一些。
```

这样一来，门控不是只看 `h_s` 和 `h_f` 的特征值，而是同时看“检索环境反馈回来
的证据”。

可以把它理解成：

```text
这个样本的内容线索是否可靠？
这个样本的时序线索是否可靠？
当前 hash 检索轨迹是否值得相信？
根据这些信息决定 slow / fast 的训练权重。
```

## 6. 检索错误归因

Agentic fusion 需要知道一个检索错误应该归因到哪条分支。

先定义两类集合：

```text
planned(i):
  规划或代理图认为样本 i 应该检索到的邻居。

actual(i):
  当前 hash code 在 memory bank 或检索库中实际检索到的邻居。
```

由此得到：

```text
missed(i) = planned(i) - actual(i)
false(i)  = actual(i)  - planned(i)
```

含义：

```text
missed:
  应该检索到但没有检索到。
  这是漏检，通常作为 hard positive。

false:
  实际检索到了但不应该这么近。
  这是误检，通常作为 hard negative。
```

归因的目标是判断：

```text
这条 missed / false 边主要暴露了 slow 的问题，
还是 fast 的问题？
```

可以用三种相似度辅助判断：

```text
s_sem(i, j):
  语义通道相似度。高说明内容语义认为 i 和 j 应该近。

s_temp(i, j):
  时序通道相似度。高说明运动/时间结构认为 i 和 j 应该近。

s_hash(i, j):
  当前 hash 空间相似度。高说明模型当前真的把 i 和 j 放近了。
```

对 missed 边：

```text
如果 s_sem 高、s_hash 低：
  内容语义其实能看出这条边，但最终 hash 没检到。
  这说明 slow 线索没有被充分传到最终检索码里。

如果 s_temp 高、s_hash 低：
  时序动态其实能看出这条边，但最终 hash 没检到。
  这说明 fast 线索没有被充分利用。
```

对 false 边：

```text
如果某条分支把 false 样本看得过近，
说明这条分支在该样本上可能提供了误导性线索；
门控和分支损失需要抑制这类错误相似度。
```

把这些边级判断聚合到样本级，可以得到归因向量：

```text
r_i = (r_sem_i, r_temp_i)
```

其中：

```text
r_sem_i:
  当前样本应该更多由 slow semantic branch 负责的证据强度。

r_temp_i:
  当前样本应该更多由 fast temporal branch 负责的证据强度。
```

为了减少单步噪声，归因可以使用边持续性 EMA：

```text
持续出现的 missed / false 边更可信；
偶然出现一次的错误边不直接强监督门控。
```

## 7. BAAL：分支归因合并目标

这里不建议再把 agentic fusion 写成三个额外损失。RF-CLaTH 已经有
view、batch-neighbor、memory-neighbor、quant、balance 等目标，如果再并列加入
门控、互补和多样性三项，会引入过多权重，训练稳定性和论文表述都会变复杂。

更合适的写法是把 BAAL 理解为 Branch Attribution Agentic Learning：
它把检索反馈合并进一个 source-aware multi-positive InfoNCE，所有语义反馈
只是在同一个正样本权重矩阵里贡献不同来源的权重。

### 7.1 L_AUCL：统一 Agentic 对比目标

对样本 `i`，正样本来源统一写成：

```text
P_i =
  paired_view(i)
  union raw_batch_neighbor(i)
  union memory_neighbor(i)
  union planned(i)
  union missed(i)
```

这些来源不再对应多个独立 loss，而是累加到同一个边权重：

```text
w_ij =
  w_view   * 1[j in paired_view(i)]
+ w_raw    * 1[j in raw_batch_neighbor(i)]
+ w_mem    * g_i * 1[j in memory_neighbor(i)]
+ w_plan   * g_i * 1[j in planned(i)]
+ w_missed * g_i * 1[j in missed(i)]
```

`false(i)` 不作为另一个损失，而是作为 hard negative 改变同一个分母：

```text
eta_ij =
  1 + (w_false - 1) * g_i * 1[j in false(i)]
```

然后使用一个合并后的 AUCL：

```text
L_AUCL =
  - mean_i log
    sum_{j in C_i} w_ij * exp(s_agentic(i, j) / tau)
    ---------------------------------------------------
    sum_{j in C_i} eta_ij * exp(s_agentic(i, j) / tau)
```

这里 `C_i` 是 batch 候选和 memory 候选的并集。这样一来，paired-view、
raw neighbor、memory neighbor、planned neighbor、missed neighbor 和 false
retrieval 都进入同一个对比学习目标，论文里也只需要讲一个主监督项。

### 7.2 分支归因进入相似度，而不是另开互补损失

分支互补不需要再写一个单独的 `MultiPositiveInfoNCE(u_f, slow_missed)`。
互补关系应该进入 AUCL 的相似度计算：

```text
s_agentic(i, j) =
    alpha_ij       * <u_s_i, u_s_j> / K_s
  + (1 - alpha_ij) * <u_f_i, u_f_j> / K_f
```

其中 `alpha_ij` 由分支归因得到：

```text
alpha_ij = sigmoid(kappa * (r_sem_ij - r_temp_ij))
```

实现时不一定要新增一个边级门控网络。可以继续使用样本级门控 `a_i`，把
`alpha_ij` 近似为 `a_i`、`0.5 * (a_i + a_j)`，或把边级归因只作为
`a_i` 的监督目标。边级写法主要用于说明梯度应该按关系类型路由到哪段子码。

直觉是：

```text
如果某条正边主要由内容语义支持，
这条边在 AUCL 里更多通过 slow 子码拉近。

如果某条正边主要由时序动态支持，
这条边在 AUCL 里更多通过 fast 子码拉近。

如果某条 false 边是 slow 误导造成的，
hard negative 的梯度会更多作用到 slow 子码。
```

因此 slow / fast 的互补不是靠第二个 loss 额外施压，而是通过同一个
AUCL 中的边级路由自然发生。

### 7.3 可选 L_route：只做轻量门控校准

如果实现里需要一个显式门控 `a_i`，可以保留一个很轻的校准项：

```text
target_a_i = sigmoid(kappa * (r_sem_i - r_temp_i))

L_route = mean_i [ g_i * BCE(a_i, target_a_i) ]
```

`L_route` 的作用只是让门控读懂可信归因，不承担主要检索监督。默认权重应该很小，
也可以先设为 0，只靠 AUCL 的 routed similarity 训练分支。

如果观察到门控塌缩，不建议再新增一个独立 `L_div`。更稳的做法是把一个弱先验
并入 `L_route`，例如只在 `gate_std` 过低时加入很小的 batch-level penalty。

## 8. 总体目标

合并后的总体目标可以写成：

```text
L_total =
  L_AUCL
+ lambda_route * L_route
+ lambda_quant * L_quant
+ lambda_balance * L_balance
```

其中：

```text
L_AUCL:
  唯一的语义/检索反馈主目标。
  view、raw neighbor、memory、planned、missed 和 false 都只是它的不同来源。

L_route:
  可选的轻量门控校准项。
  不负责单独优化检索结构，也不默认承担互补/多样性监督。

L_quant / L_balance:
  哈希码性质约束，和语义反馈正交，继续单独保留。
```

这样损失栈从“基础五项 + agentic 三项”收敛为：

```text
1. 一个统一语义反馈目标 L_AUCL。
2. 一个可选轻量路由校准 L_route。
3. 两个必要哈希正则 L_quant / L_balance。
```

## 9. 完整数据流

可以把一次训练 step 写成下面的数据流：

```text
输入视频特征 x
  -> T-SAS 选择语义锚点
  -> slow branch 得到 h_s
  -> fast branch 得到 h_f
  -> hash_head_s 得到 u_s
  -> hash_head_f 得到 u_f
  -> 拼接得到 u = [u_s ; u_f]

u 与 memory bank / neighbor graph 交互
  -> 得到 planned / actual / missed / false 或其它邻居反馈
  -> 计算 q_sem, q_temp, g_i
  -> 计算归因 r_sem, r_temp
  -> 门控 MLP 输出 a_i

训练目标
  -> 用 a_i 加权 slow / fast 两段相似度
  -> 把 view / raw / memory / planned / missed 合并为 AUCL 正样本权重
  -> 把 false retrieval 合并为 AUCL hard negative 权重
  -> 可选用 L_route 轻量校准门控
  -> 用 L_quant / L_balance 约束哈希码
  -> 更新编码器、哈希头、门控和融合相关参数
```

这个流程里，检索反馈同时影响两件事：

```text
监督层:
  哪些样本应该被拉近或推远。

融合层:
  哪条分支应该对这个样本承担更多责任。
```

这就是 agentic fusion 相比普通融合最核心的地方。

## 10. 直观例子

假设有两个视频：

```text
video A:
  人在打网球。

video B:
  人在挥拍训练。
```

如果内容帧都有人、球场、球拍，slow 分支可能认为它们很近；如果时序动作也相似，
fast 分支也会支持它们靠近。这种样本可以得到较均衡的门控。

再看另一个样本：

```text
video C:
  人站在球场上没有击球。
```

它和 video A 在静态内容上很像，但动作过程不同。若当前 hash 把 A 和 C 排得过近，
这可能是一个 false retrieval。归因时会发现：

```text
s_sem(A, C) 高:
  内容确实像。

s_temp(A, C) 低:
  时序动作不支持它们靠太近。

s_hash(A, C) 高:
  当前 hash 被内容误导了。
```

于是 agentic fusion 会给出反馈：

```text
这个样本不能只相信 slow 内容分支；
应该提高 fast 时序分支在训练中的作用，
让最终 hash code 能区分“球场上站着”和“正在击球”。
```

这就是“检索错误归因到分支，再反过来调节融合”的实际含义。

## 11. 设计要点

### 11.1 门控要按样本变化

不同视频的判别依据不同：

```text
有些类别更依赖场景和对象。
有些类别更依赖动作过程和时序变化。
```

因此 `a_i` 应该是 per-sample gate，而不是全局常数。

### 11.2 反馈要有信任门控

训练早期的 hash trace 可能很噪。如果直接把 noisy retrieval 当成强监督，
会让门控学到错误路由。因此所有基于 actual trace 的反馈最好乘上 `g_i`：

```text
g_i 高:
  当前样本的检索轨迹可信，可以使用更强反馈。

g_i 低:
  当前样本的检索轨迹不可信，门控和归因损失降权。
```

### 11.3 归因要看分支与 hash 的差

只看 `s_sem` 或 `s_temp` 不够。关键是比较：

```text
分支认为谁该近
当前 hash 实际把谁放近
```

两者的偏差才是“这个分支有没有被最终 hash 充分利用”的证据。

### 11.4 互补比单纯融合更重要

如果 slow 和 fast 两段都学同样的静态内容，双段码没有意义。
这里的互补不通过额外 `L_comp` 完成，而是通过 AUCL 里的边级路由完成：

```text
slow 负责内容证据强的正边和误检修正。
fast 负责时序证据强的正边和误检修正。
同一条 AUCL 监督边根据归因把梯度分配给不同子码。
```

这样最终拼接码仍然包含多源信息，但不需要再引入第二个互补对比损失。

## 12. 可观测诊断

为了判断 agentic fusion 是否真的工作，可以看下面几类指标：

```text
gate_mean:
  a_i 的平均值。长期接近 0 或 1 说明可能塌缩。

gate_std:
  a_i 的样本间差异。过小说明所有样本走同一路由。

slow_only_mAP:
  只用 u_s 检索的性能。

fast_only_mAP:
  只用 u_f 检索的性能。

concat_mAP:
  用完整 u=[u_s;u_f] 检索的性能。

missed_overlap:
  slow 和 fast 的漏检集合重合度。下降说明两段更互补。

false_overlap:
  slow 和 fast 的误检集合重合度。下降说明错误模式更分散。

gate_vs_attribution_corr:
  a_i 是否真的跟 r_sem - r_temp 同向变化。
```

一个健康的现象是：

```text
concat_mAP > slow_only_mAP 和 fast_only_mAP；
gate_std 不为 0；
slow / fast 的 missed overlap 下降；
门控变化和归因信号有正相关。
```

## 13. 写进论文时的表述

中文表述：

```text
我们进一步将检索反馈从监督目标扩展到表示融合层，提出 agentic fusion 机制。
该机制将慢分支语义码与快分支时序码划分为两个互补的哈希子空间，
并将 paired-view、raw-neighbor、memory-neighbor 以及检索反馈产生的
planned、missed 和 false 关系合并为单一的 agentic unified contrastive
目标。通过将漏检和误检归因到具体分支，模型能够在同一对比目标中自适应地
增强内容或时序线索，从而形成“检索动作-错误归因-分支路由-哈希更新”的闭环。
```

英文表述：

```text
We extend retrieval feedback from the objective level to the representation
fusion level through an agentic fusion mechanism. The slow semantic branch and
the fast temporal branch produce two complementary hash subcodes. Paired-view
consistency, raw-neighbor supervision, memory-neighbor feedback, and retrieval
feedback from planned, missed, and false relations are merged into a single
agentic unified contrastive objective. By attributing missed and false
retrievals to specific branches, the model routes training pressure toward
semantic or temporal cues within the same objective, forming a closed loop of
retrieval action, error attribution, branch routing, and hash-space update.
```

## 14. 最核心的一句话

```text
Agentic fusion 的本质是：
让检索反馈不仅塑造哈希空间中的邻居关系，
还塑造 slow / fast 两条分支在每个样本上的融合责任。
```
