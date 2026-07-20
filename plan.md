# RF-CLaTH UCF101 AUCL 实验结论与后续计划

更新时间：2026-07-20（Asia/Shanghai）

## 一、当前结论

截至目前，UCF101 32-bit 的 C0-H1 实验已全部完成。

```text
当前最优 AUCL：E1，best.pth epoch 25，mAP@100=0.2705
旧目标对照 C0：epoch 10，mAP@100=0.3015
当前绝对差值：-0.0310
当前相对差距：约 -10.3%
```

已经确认：

1. 数据、标签、neighbor cache、评估和服务器训练链路正常。
2. routed similarity 本身不是问题；`routed similarity × 过早全量 memory` 会触发
   hash collapse。
3. epoch 1-6 先完成 batch-only warmup，再在 epoch 7-10 线性加入 memory，可以
   保持 hash 健康。
4. memory 对 AUCL 有效：E1 相比无 memory 的 E0，在 best checkpoint 上提升
   `0.0226`，相对提升约 `9.1%`。
5. hard top-K、supervised-only 和 uniform-1024 均未超过全量 memory；当前不再继续
   memory candidate 裁剪路线。
6. H0/H1 确认 actual trace 在关闭 hard mining、feedback graph 和反馈权重时对训练目标
   基本中性，epoch 15 的 H1-H0 `mAP@100=-0.0008`。
7. H1 的 actual-only 样本仍有很高的真实同类比例，epoch 15 为 `0.357`；集合意义上的
   `false = actual - planned` 不能直接作为语义 hard negative。
8. 当前仍不启用 missed/false loss、hard mining 或 feedback graph；下一阶段先做
   top-r 和 planner-score 分层的只读语义诊断。

当前主线配置固定为 E1：

```text
dataset: UCF101 / s5vh_ucf
hash bits: 32
seed: 3346
objective: agentic_unified_contrastive
routed similarity: on
epoch 1-6: no memory candidates
epoch 7-10: candidate/source scale = 0.25 / 0.50 / 0.75 / 1.00
epoch 11+: full memory candidates
actual trace: off
hard mining: off
feedback graph write: off
```

## 二、统一评估口径

主要指标：

```text
mAP@100 / P@5 / R@100
hash entropy / bit-use
```

注意：训练代码使用全库 `mAP` 选择 `best.pth`，报告主要比较 `mAP@100`。因此必须
同时记录 best checkpoint epoch 和 mAP@100 曲线，不能直接把最后一个 epoch 当成
最终结果。

所有实验均使用新划分 UCF101：

```text
train/database: 9990
query: 3330
feature: 25 x 4096
batch size: 256
binary code: {-1, +1}
```

## 三、C0-D2：失败原因定位

### 3.1 第一轮 C0-C3

统一为 seed 3346、32-bit、10 epoch，关闭 actual trace、hard mining、feedback graph。

| 实验 | 核心设置 | Epoch 10 mAP@100 | P@5 | R@100 |
|---|---|---:|---:|---:|
| C0 Legacy control | 旧 `merged_semantic_self_calibrated` | **0.3015** | **0.7395** | **0.3812** |
| C1 | AUCL，route off，memory from epoch 1 | 0.1665 | 0.5223 | 0.2610 |
| C2 | AUCL，route on，10 epoch 内无 memory | **0.2172** | **0.6447** | **0.3003** |
| C3 | AUCL，route off，epoch 6 加 memory | 0.1733 | 0.5375 | 0.2634 |

结论：

- C0 复现健康，排除数据和训练链路异常。
- C2 是最强 AUCL 分支，说明 route 在 batch-only 阶段有益。
- 关闭 route 不能解决 AUCL 的质量上限。
- 第一轮所有组 hash 健康，但健康 hash 不等于健康检索。

### 3.2 第二轮 D0-D2

| 实验 | 核心设置 | 最佳 mAP@100 | Hash 状态 |
|---|---|---:|---|
| D0 | route on，epoch 1 全量 memory | 0.0138（epoch 5） | 严重坍缩，entropy=0.186，bit-use=0.188 |
| D1 | route on，epoch 6 硬切换全量 memory | 0.2268（epoch 10） | 健康 |
| D2 | route on，epoch 7-10 线性 ramp | **0.2287（epoch 10）** | 健康 |

D2 的实际 ramp：

| Epoch | Candidate scale | Memory candidates | Source scale |
|---:|---:|---:|---:|
| 6 | 0.00 | 0 | 0.00 |
| 7 | 0.25 | 2498 | 0.25 |
| 8 | 0.50 | 4995 | 0.50 |
| 9 | 0.75 | 7492 | 0.75 |
| 10 | 1.00 | 9990 | 1.00 |

因果结论：

```text
route + epoch 1 全量 memory -> hash collapse
batch-only warmup -> memory 可以安全加入
linear ramp 比硬切换略好，但不是决定性因素
```

## 四、E0-E1：30-epoch 主线验证

### 4.1 成对曲线

| Epoch | E0 无 memory | E1 full-memory ramp | E1 - E0 |
|---:|---:|---:|---:|
| 5  | 0.1690 | 0.1694 | +0.0004 |
| 10 | 0.2173 | 0.2299 | +0.0126 |
| 15 | 0.2390 | 0.2594 | +0.0204 |
| 20 | 0.2457 | 0.2690 | +0.0233 |
| 25 | 0.2479 | **0.2705** | +0.0226 |
| 30 | 0.2484 | 0.2697 | +0.0213 |

Checkpoint 口径：

| 实验 | best.pth epoch | best checkpoint mAP@100 | 曲线峰值 |
|---|---:|---:|---:|
| E0 | 25 | 0.2479 | 0.2484（epoch 30） |
| E1 | 25 | **0.2705** | **0.2705（epoch 25）** |

E1 在 epoch 20-30 的优势稳定在 `0.0213-0.0233`，epoch 30 的
`hash_entropy=0.993`、`bit_use=1.000`，不存在 D0 式坍缩。

E1 的增益主要来自召回：

```text
E0 epoch 25: P@5=0.7071, R@100=0.3240
E1 epoch 25: P@5=0.6728, R@100=0.3487
```

即 full memory 提高覆盖率，但牺牲了一部分近邻精度。

### 4.2 Logit 诊断

E1 epoch 30：

```text
batch logits:  mean=0.039, std=1.359, min=-4.298, max=4.994
memory logits: mean=0.012, std=1.303, min=-4.707, max=4.995
memory denominator mass ratio: 约 16.5
memory-vs-batch log-mass gap: 约 2.78
positive memory logit gap: 约 4.03
```

batch 与 memory 的数值尺度基本一致，positive gap 也健康。因此主要矛盾不是
normalization 或 temperature，而是候选数量和负例分布。

## 五、F0-G1：Memory candidate 消融

### 5.1 完整曲线

| Epoch | F0 hard-64 | F1 hard-128 | G0 supervised-only | G1 uniform-1024 | E1 full-memory |
|---:|---:|---:|---:|---:|---:|
| 5  | 0.1694 | 0.1691 | 0.1694 | 0.1693 | 0.1694 |
| 10 | 0.1993 | 0.2078 | 0.2153 | 0.2223 | 0.2299 |
| 15 | 0.2146 | 0.2227 | 0.2330 | 0.2484 | 0.2594 |
| 20 | **0.2153** | **0.2434** | 0.2358 | **0.2567** | 0.2690 |
| 25 | 0.2116 | 0.2407 | **0.2373** | 0.2551 | **0.2705** |
| 30 | 0.2111 | 0.2416 | 0.2371 | 0.2557 | 0.2697 |

### 5.2 Best checkpoint

| 实验 | best.pth epoch | mAP@100 | P@5 | R@100 |
|---|---:|---:|---:|---:|
| F0 hard-64 | 15 | 0.2146 | 0.6691 | 0.2962 |
| F1 hard-128 | 20 | 0.2434 | 0.6952 | 0.3182 |
| G0 supervised-only | 25 | 0.2373 | 0.6949 | 0.3151 |
| G1 uniform-1024 | 20 | **0.2567** | 0.6834 | **0.3355** |
| E1 full-memory | 25 | **0.2705** | 0.6728 | **0.3487** |

### 5.3 机制解释

| 实验 | Epoch 30 candidates | Denominator ratio | Positive gap | 结论 |
|---|---:|---:|---:|---|
| F0 hard-64 | 68 | 1.81 | 0.14 | hardest negatives 过强 |
| F1 hard-128 | 130 | 2.72 | 0.44 | 比 F0 好，但仍低于 E0/E1 |
| G0 supervised-only | 14 | 0.57 | 0.00 | 只有 positives 不足以学习区分性 |
| G1 uniform-1024 | 1036 | 2.22 | 3.86 | 保留部分收益，但召回低于 E1 |
| E1 full-memory | 9990 | 16.5 | 4.03 | 当前最优 AUCL |

最终判断：

1. 降低 denominator mass 本身不会自动提高检索性能。
2. hard top-K 会把候选集中到高相似难负例，压缩 positive gap，导致优化过难。
3. supervised-only 低于 E0，说明 memory positives 不是 E1 增益的唯一来源。
4. uniform-1024 高于 E0、低于 E1，说明大范围、分布温和的 memory negatives 对召回
   有实际贡献。
5. 后续主线恢复 `memory_candidate_strategy=all`，停止 candidate 裁剪实验。

## 六、诊断口径修正

### 6.1 Neighbor label precision 假零值

训练日志中的：

```text
neighbor_label_precision@20=0.0000
```

不是 neighbor cache 质量为零，而是 `estimate_neighbor_label_precision(...)` 只读取
`dataset.records`，S5VH 数据集实际使用 `dataset.labels`。

使用真实 UCF train labels 和 neighbor cache 复算。旧记录的 `0.64445` 是等距抽取
2,000 个 anchor 的估计值；修正后配置中的 `label_precision_probe=0` 表示全量统计：

```text
全量 9,990 个 anchor: precision@20 = 0.642007
平均每个 anchor 的 20 个邻居中有 12.840 个同类样本
等距抽取 2,000 个 anchor: precision@20 = 0.64445
```

该问题只影响诊断日志，不影响训练。该函数已在 H0/H1 前修正。

### 6.2 Memory mass 诊断

G0 部分 query 没有 memory candidate 时，旧实现会让 `denom_gap` 被掩码值拉成极大
负数。代码已经改为仅在具有 memory candidate 的 query 上计算 mass ratio/log-gap。

G0/G1 进程在修复前已经启动，因此本轮 G0 应忽略异常的 `denom_gap`，使用有效的
`denom_ratio`；训练 loss 不受影响。

## 七、H0/H1 Actual Trace 纯观察

### 7.1 目标

确认 planner top-M、当前 hash Hamming retrieval 和真实语义标签之间是否一致。

这一阶段只做观察，不让 actual trace 进入损失或 memory 写回。代码核对表明，在以下
设置下 actual trace 不会改变训练目标：

```text
hard mining: off
feedback graph: off
eta_missed: 0
eta_false: 0
```

### 7.2 先补充只读诊断

1. 修正 S5VH `neighbor_label_precision@20`，支持 `dataset.labels`。
2. 在 `arf_trace_targets(...)` 中增加：
   - planned label precision
   - actual retrieval label precision
   - missed-only label precision
   - actual-only false label precision
3. 上述指标全部 detach，只写日志，不参与 loss、policy 或 memory 更新。

### 7.3 H0/H1 设置

| 实验 | 设置 | GPU 建议 |
|---|---|---|
| H0 Trace-off control | E1 全量 memory 主线，actual trace off | cuda2 |
| H1 Trace-only | 与 H0 相同，epoch 7 开启 actual trace | cuda3 |

共同设置：

```text
UCF101 / 32-bit / seed 3346 / 15 epoch
epoch 1-6 no memory
epoch 7-10 memory candidate/source ramp
epoch 11-15 full memory
hard_mining_start_epoch=999
update_feedback_graph=false
eta_missed_start/final=0
eta_false_start/final=0
eval every 5 epoch
```

### 7.4 进入反馈训练的门槛

```text
H1 与 H0 的 mAP@100 差异 <= 0.003
actual_overlap >= 0.20
false_ratio <= 0.80
planned label precision >= 0.55
actual label precision >= 0.55
actual-only false label precision <= 0.20
hash_entropy >= 0.90
bit_use >= 0.90
```

最后一项语义门槛尤其重要：当前 `false` 只表示“不在 planner 集合中”，不代表真实
语义负例。如果 actual-only 样本仍具有较高 label precision，就不能直接作为 hard
negative。

### 7.5 条件式后续

若 H0/H1 满足全部门槛：

```text
先启用 missed/false loss，feedback graph 继续关闭
eta 从 0 缓慢 ramp 到 0.1，不直接使用 1.0
完成稳定性与回退对照后，再考虑 graph write
```

若 actual overlap 低，但 planned/actual label precision 都高：

```text
说明两者检索到的是不同的同类样本
不能按集合差异定义 false negative
需要把 feedback 改为语义或 planner-score 感知
```

若 actual label precision 低：

```text
暂不启用 hard mining
优先改进 hash retrieval 与 planner representation 的一致性
```

### 7.6 执行状态

2026-07-20 20:57（Asia/Shanghai）已完成只读诊断实现和远端验证：

```text
neighbor_label_precision@20 支持 S5VH dataset.labels
label_precision_probe=0 按全量样本统计
actual trace 新增 label_p/a/m/f：
  planned / actual / missed-only / actual-only-false label precision
远端验证：git diff --check / bash -n / py_compile / diagnostic smoke 均通过
```

第一次 H1 预跑暴露了一个 schedule 门控问题：`actual_trace_start_epoch=7` 会被
`planner.warmup.epochs=10` 再次关闭。直接缩短 planner warmup 会改变
`omega_s/omega_t/omega_z`，破坏 H0/H1 控制，因此修正为 actual trace 起点与 planner
score warmup 独立。远端 schedule smoke 确认：

```text
epoch 6: actual trace off
epoch 7: actual trace on
epoch 7 planner weights: omega_s/t/z = 0.65 / 0.35 / 0.0
eta_missed/eta_false = 0 / 0
```

无效 H1 预跑在 epoch 7 step 20 停止，日志保留但不纳入结论。正式 H0/H1 均已完成：

```text
H0: cuda2, trace off, 2026-07-20 20:57-21:48（Asia/Shanghai）
H1: cuda3, trace from epoch 7, 2026-07-20 21:18-21:59（Asia/Shanghai）
output root: /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_ucf_h0_h1_actual_trace
H0 log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_ucf_h0_actual_trace_cuda2_20260720_125729.queue.log
H1 log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_ucf_h1_actual_trace_cuda3_20260720_131803.queue.log
无效 H1 预跑: rf_clath_ucf_h1_actual_trace_cuda3_20260720_125729.queue.log
```

### 7.7 检索指标

| Epoch | H0 mAP@100 | H1 mAP@100 | H1-H0 | H0 P@5 / R@100 | H1 P@5 / R@100 |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.1697 | 0.1688 | -0.0009 | 0.5697 / 0.2592 | 0.5696 / 0.2583 |
| 10 | 0.2277 | 0.2307 | +0.0030 | 0.6115 / 0.3135 | 0.6165 / 0.3140 |
| 15 | **0.2538** | **0.2530** | **-0.0008** | 0.6439 / 0.3353 | 0.6435 / 0.3351 |

两组的训练 loss 和 hash 轨迹一致；epoch 15 的差异远小于 `0.003`。因此 actual trace
在当前只读设置下没有系统性改变训练目标，epoch 10 的 `+0.0030` 不应解释为增益。

### 7.8 H1 trace 语义曲线

| Epoch | Overlap | False ratio | Planned label | Actual label | Missed-only label | Actual-only false label | Entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7  | 0.293 | 0.695 | 0.661 | 0.440 | 0.592 | 0.270 | 0.994 |
| 8  | 0.301 | 0.688 | 0.661 | 0.453 | 0.590 | 0.283 | 0.995 |
| 9  | 0.305 | 0.683 | 0.661 | 0.464 | 0.586 | 0.294 | 0.991 |
| 10 | 0.314 | 0.675 | 0.661 | 0.478 | 0.581 | 0.305 | 0.992 |
| 11 | 0.363 | 0.624 | 0.629 | 0.491 | 0.541 | 0.315 | 0.994 |
| 12 | 0.369 | 0.618 | 0.630 | 0.505 | 0.539 | 0.332 | 0.993 |
| 13 | 0.373 | 0.615 | 0.631 | 0.514 | 0.540 | 0.345 | 0.992 |
| 14 | 0.377 | 0.611 | 0.630 | 0.518 | 0.539 | 0.351 | 0.990 |
| 15 | **0.381** | **0.607** | **0.635** | **0.525** | **0.544** | **0.357** | **0.992** |

全程 `bit_use=1.000`。actual overlap 和 actual label precision 随训练提高，但
actual-only false label precision 也从 `0.270` 升到 `0.357`。这说明 hash retrieval
逐渐找到更多同类样本，而 planner top-M 没覆盖其中相当一部分；这些样本只是集合上的
`false`，并非语义负例。

### 7.9 门槛判定

| 门槛 | Epoch 15 | 结果 |
|---|---:|---|
| `abs(H1-H0 mAP@100) <= 0.003` | 0.0008 | 通过 |
| `actual_overlap >= 0.20` | 0.381 | 通过 |
| `false_ratio <= 0.80` | 0.607 | 通过 |
| `planned label precision >= 0.55` | 0.635 | 通过 |
| `actual label precision >= 0.55` | 0.525 | **未通过** |
| `actual-only false label precision <= 0.20` | 0.357 | **未通过** |
| `hash_entropy >= 0.90` | 0.992 | 通过 |
| `bit_use >= 0.90` | 1.000 | 通过 |

结论：H0/H1 未满足进入反馈训练的全部门槛。当前不能启用 false hard negative，
也不能写 feedback graph；missed-only precision 只有 `0.544`，也不应直接加权训练。

### 7.10 下一阶段 I0：多深度与 planner-score 只读诊断

下一步仍保持 loss、hard mining 和 graph write 全部关闭，只补充一次 trace 诊断：

1. 同一 Hamming trace 同时统计 `top-r = 5 / 10 / 20` 的 overlap、actual label precision
   和 actual-only label precision，判断问题是否主要来自检索深度。
2. 对 actual-only 样本按 planner score 分层，记录高/低 score 子集的覆盖率和 label
   precision，判断 planner score 能否作为 label-free 的 false 过滤代理。
3. 若低 top-r 能让 actual precision 达标，但 actual-only label precision仍高，则只允许
   actual 作为候选正例/忽略项，不构造 false negatives。
4. 只有找到不使用真实标签、且能把 actual-only 语义正例污染压到 `<=0.20` 的代理规则后，
   才进入低权重反馈训练。

I0 诊断实现已通过远端 `git diff --check`、`bash -n`、`py_compile` 和 toy trace smoke，
并于 2026-07-20 22:05（Asia/Shanghai）在 cuda3 启动：

```text
PID: 2324214
fixed trace budget: 20
output root: /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_ucf_i0_trace_depth_score
log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_ucf_i0_trace_depth_score_cuda3_20260720_140548.queue.log
script: tools/run_rf_clath_ucf_i0_trace_depth_score.sh
```

## 八、明确暂不执行

在 I0 诊断完成前，不执行：

1. hard mining。
2. feedback graph 写回。
3. missed/false feedback loss。
4. 继续缩小 memory candidate pool。
5. 单独调整 memory temperature 或 L2 normalization。
6. 16/64-bit 扩展和多 seed 正式实验。
7. 128-bit 实验。

## 九、实验产物索引

| 阶段 | Output root | Queue log / script |
|---|---|---|
| C0-C3 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_ucf_round1/` | `rf_clath_ucf_round1_cuda2_c0-c3_20260715_131852.queue.log` / `rf_clath_ucf_round1_cuda3_c1-c2_20260715_131852.queue.log` / `tools/run_rf_clath_ucf_round1_cuda23.sh` |
| D0-D2 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_ucf_round2/` | `rf_clath_ucf_round2_cuda2_d0_20260715_140726.queue.log` / `rf_clath_ucf_round2_cuda3_d1-d2_20260715_140726.queue.log` / `tools/run_rf_clath_ucf_round2_d0_d2_cuda23.sh` |
| E0-E1 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_ucf_e0_e1_30ep/` | `rf_clath_ucf_e0_30ep_cuda2_20260715_160539.queue.log` / `rf_clath_ucf_e1_30ep_cuda3_20260715_160539.queue.log` / `tools/run_rf_clath_ucf_e0_e1_30ep_cuda23.sh` |
| F0-F1 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_ucf_f0_f1_topk/` | `rf_clath_ucf_f0_topk_cuda2_20260715_233840.queue.log` / `rf_clath_ucf_f1_topk_cuda3_20260715_233840.queue.log` / `tools/run_rf_clath_ucf_f0_f1_topk_cuda23.sh` |
| G0-G1 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_ucf_g0_g1_memory_pool/` | `rf_clath_ucf_g0_memory_pool_cuda2_20260716_011756.queue.log` / `rf_clath_ucf_g1_memory_pool_cuda3_20260716_011756.queue.log` / `tools/run_rf_clath_ucf_g0_g1_memory_pool_cuda23.sh` |
| H0-H1 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_ucf_h0_h1_actual_trace/` | `rf_clath_ucf_h0_actual_trace_cuda2_20260720_125729.queue.log` / `rf_clath_ucf_h1_actual_trace_cuda3_20260720_131803.queue.log` / `tools/run_rf_clath_ucf_h0_h1_actual_trace_cuda23.sh` |
| I0 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_ucf_i0_trace_depth_score/` | `rf_clath_ucf_i0_trace_depth_score_cuda3_20260720_140548.queue.log` / `tools/run_rf_clath_ucf_i0_trace_depth_score.sh` |

远端统一目录：

```text
project: /mnt/disk2/yql/RF-CLaTH
outputs: /mnt/disk2/yql/RF-CLaTH_outputs
logs: /mnt/disk2/yql/RF-CLaTH_run_logs
```
