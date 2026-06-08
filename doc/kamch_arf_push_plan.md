# KAMCH + PER-SAS + ARF 推进计划

> 版本：v1.0  
> 目标：在当前强 KAMCH 基线之上，按阶段实现 `plan.md` 中的 **training-free 语义锚点选择器** 与 **Agentic Retrieval Feedback 训练范式**，并形成可投稿级别的完整实验与论文材料。

---

## 1. 当前状态判断

当前 KAMCH 原始实现已经不是弱 baseline，而是一个强基线。实验记录显示：

- RePartition 协议已固定，后续实验应继续使用该协议，避免因数据划分或评估口径变化导致结果不可比。
- 当前固定 bits 为 `16 / 32 / 64`，后续不再进行 128-bit 实验。
- 当前训练优化参数为：`epochs=150`，`lr=3e-5`，`warmup_epochs=10`，`weight_decay=1e-4`。
- HMDB 上 E23-2 在既有 bit 设置的 `mAP@5-100` 全部超过 S5VH。
- UCF 上 E23-2 在 head/mid top-K 表现强，但 `mAP@100` 在部分 bit 上近似持平或略低，说明中后段检索结构仍有优化空间。
- E25-2 w/o center/prototype 在 UCF 16-bit 上没有塌缩，并且 `mAP@100=0.3399` 明显高于 S5VH 16-bit 的 `0.3020` 和 E23-2 16-bit 的 `0.3015`，但 `mAP@5` 低于 E23-2，说明去除强 center/prototype 约束后，中后段邻域检索有更大收益。

因此，接下来的推进策略不是“推翻原 KAMCH”，而是：

```text
保留当前 KAMCH 强结构优势，
用 PER-SAS 提升慢分支输入质量，
用 ARF 进一步优化实际哈希检索行为。
```

最终目标是让模型从：

```text
strong representation hashing
```

升级为：

```text
retrieval-behavior-aware hashing
```

---

## 2. 总体推进原则

### 2.1 不一次性全替换

不要直接把 selector、loss、memory、planner、feedback 全部一次性换掉。这样一旦性能下降，很难定位问题。

推荐采用分阶段推进：

```text
KAMCH-Base
    ↓
+ T-SAS / PER-SAS selector
    ↓
+ Static ARF
    ↓
+ Actual Retrieval Trace
    ↓
+ Feedback Weights
    ↓
+ Memory P_z
    ↓
Full KAMCH-PER-SAS-ARF
```

---

### 2.2 先小数据集短码验证，再扩展全量

优先调试：

```text
UCF 16-bit
UCF 32-bit
```

原因：

- UCF 当前 head/mid 指标强，但 `mAP@100` 仍有优化空间。
- ARF 的核心作用是修正 `missed neighbors` 和 `false retrievals`，更可能体现在中后段检索指标上。
- 16/32-bit 码长更短，哈希空间压缩更强，更容易体现 ARF 对检索结构的改善。

然后验证：

```text
HMDB 64-bit
```

原因：

- HMDB 当前已经全线超过 S5VH，适合验证 ARF 是否能在强势 setting 上继续提升。

最后扩展：

```text
ActivityNet 16/32/64
FCVID 16/32/64
```

---

### 2.3 主线指标不要只盯 mAP@5

ARF 优化的是检索轨迹，而不是单纯优化最前排样本。因此主分析应同时关注：

```text
mAP@5 / 20 / 40 / 60 / 80 / 100
P@5 / 10 / 20 / 40 / 60 / 80 / 100
R@5 / 10 / 20 / 40 / 60 / 80 / 100
planned/actual overlap
false retrieval ratio
retrieved samples mean P_ij
Hamming distance distribution
bit balance
```

预期最明显的提升位置：

```text
mAP@20-100
P@20-100
R@20-100
16-bit / 32-bit short codes
```

---

## 3. 最终目标版本

最终希望形成以下版本：

```text
KAMCH-PER-SAS-ARF
```

整体结构：

```text
video frame features
    ↓
PER-SAS / T-SAS training-free keyframe selector
    ↓
selected semantic anchors ─────→ slow semantic branch: selected_class_attention
all frames ───────────────────→ fast temporal branch: bidirectional_mamba
    ↓                              ↓
semantic representation s_i        temporal representation t_i
    ↓                              ↓
        content_time_lateral fusion
                    ↓
              video representation z_i
                    ↓
                 hash head
                    ↓
              soft hash code u_i
                    ↓
          h_i = tanh(u_i), b_i = sign(u_i)
```

训练目标：

$$
\mathcal L
=
\mathcal L_{\text{ARF}}
+
\lambda_q \mathcal L_{\text{quant}}
+
\lambda_b \mathcal L_{\text{balance}}
$$

默认：

```text
lambda_q = 0.10
lambda_b = 0.05
```

后期二值化强化：

```text
lambda_q = 0.20
lambda_b = 0.05
gamma = 10
```

主贡献叙事：

```text
KAMCH 不再以 view-level contrastive learning 为主监督，
而是将 hash code 视为 retrieval action，
通过 memory retrieval environment 得到 actual retrieval trace，
再用 planned neighbors 与 actual retrieval trace 的偏差生成 feedback weights，
直接优化哈希空间中的实际检索行为。
```

---

## 4. 阶段 0：锁定强基线与协议

### 4.1 目标

建立不可随意变动的 KAMCH-Base，用作所有后续新范式实验的参照。

### 4.2 固定内容

```yaml
protocol:
  data_root: /mnt/disk2/yql/dataset_rePartition
  bits: [16, 32, 64]
  train_batch:
    activitynet: 256
    hmdb: 256
    ucf: 256
    fcv: 512
  eval_batch: 256
  precision_recall_topk: [5, 10, 20, 40, 60, 80, 100]
  map_topk: [5, 20, 40, 60, 80, 100]

training:
  epochs: 150
  lr: 3e-5
  warmup_epochs: 10
  weight_decay: 1e-4
```

### 4.3 需要整理的基线表

必须形成一份干净表格：

| Dataset | Bits | KAMCH-Base mAP@5 | mAP@20 | mAP@40 | mAP@60 | mAP@80 | mAP@100 | Selected Epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ActivityNet | 16 | | | | | | | |
| ActivityNet | 32 | | | | | | | |
| ActivityNet | 64 | | | | | | | |
| FCVID | 16 | | | | | | | |
| FCVID | 32 | | | | | | | |
| FCVID | 64 | | | | | | | |
| HMDB | 16 | | | | | | | |
| HMDB | 32 | | | | | | | |
| HMDB | 64 | | | | | | | |
| UCF | 16 | | | | | | | |
| UCF | 32 | | | | | | | |
| UCF | 64 | | | | | | | |

### 4.4 验收标准

```text
[ ] KAMCH-Base 所有结果有 official recompute。
[ ] 每个结果有 run dir、checkpoint、selected epoch。
[ ] 指标包含 mAP/P/R 全 top-K。
[ ] 与 S5VH / AutoSSVH 的差值表完整。
[ ] 后续所有实验严格复用相同协议。
```

---

## 5. 阶段 1：只替换关键帧选择器

### 5.1 实验名

```text
E26-1: KAMCH + T-SAS/PER-SAS + Original Loss
```

### 5.2 目标

确认 training-free selector 不破坏当前强 KAMCH。

此阶段只替换：

```text
segment_rerank_gumbel_topk
```

为：

```text
T-SAS / PER-SAS
```

训练损失仍然保持原 KAMCH 设置。

---

### 5.3 实现任务

```text
[ ] 新增 selector/per_sas.py
[ ] 实现 batch 级 per_sas_selector_batch
[ ] 支持输入 [B, 30, 2048]
[ ] 每个视频返回 6 个 keyframe indices
[ ] 保证每个 5-frame segment 选择 1 帧
[ ] selector 全程 no_grad / detach
[ ] selector 不使用 hash output / loss / reconstruction error
[ ] 两个增强视图共享同一个 key_idx
[ ] slow branch 使用 key_idx 选帧
[ ] fast branch 继续使用 all frames
```

---

### 5.4 推荐配置

```yaml
model:
  keyframe_selector:
    type: t_sas
    trainable: false
    num_keyframes: 6
    segment_size: 5
    share_key_idx_across_views: true
    score_weights:
      global_repr: 0.4
      local_repr: 0.5
      local_stability: 0.1
    set_objective_weights:
      coverage: 0.6
      quality: 0.3
      redundancy: 0.1

  fast_encoder:
    input_frames: all
```

---

### 5.5 首批实验

优先：

```text
UCF 16-bit
UCF 32-bit
```

其次：

```text
HMDB 64-bit
```

---

### 5.6 监控指标

```text
[ ] key_idx 是否每段 1 帧
[ ] key_idx 是否过度集中在 segment 边界
[ ] selector 平均耗时
[ ] slow branch 输入 shape 是否正确：[B, 6, 2048]
[ ] fast branch 输入 shape 是否仍为：[B, 30, 2048]
[ ] mAP/P/R 是否接近或超过 KAMCH-Base
```

---

### 5.7 验收标准

理想：

```text
T-SAS/PER-SAS + original loss >= KAMCH-Base
```

可接受：

```text
mAP@100 下降不超过 0.5 point，且部分 top-K 有提升。
```

不可接受：

```text
mAP@100 下降超过 1.0 point；
或者 slow branch 输入变化导致训练明显不稳定。
```

若不可接受，优先排查：

```text
1. 两个 view 是否共享 key_idx。
2. key_idx 是否在增强后发生错位。
3. T-SAS 是否错误使用增强后的 noisy features。
4. fast branch 是否仍然使用 all frames。
5. selector 是否返回重复帧或越界 index。
```

---

## 6. 阶段 2：实现 Planner Graph，不启用 ARF

### 6.1 实验名

```text
E26-2: Planner Graph Sanity
```

### 6.2 目标

先实现 ARF 所需的非参数伪检索图，但不改变训练目标。

这一阶段只做 graph 构建与日志记录，用于检查：

```text
P_s / P_t / P_z 是否合理；
top-M planned neighbors 是否比随机样本更相似；
semantic graph 与 temporal graph 是否有互补性。
```

---

### 6.3 实现任务

```text
[ ] 新增 memory/memory_bank.py
[ ] 新增 planner/retrieval_graph_planner.py
[ ] 实现 semantic prototype sem_proto
[ ] 实现 dynamic prototype dyn_proto
[ ] 实现 P_s = max(0, cos(sem_proto_i, sem_proto_j))
[ ] 实现 P_t = max(0, cos(dyn_proto_i, dyn_proto_j))
[ ] 实现 P_z = max(0, cos(memory_z_i, memory_z_j))
[ ] 实现 P = omega_s P_s + omega_t P_t + omega_z P_z
[ ] 实现 TopM planned neighbors N_i
[ ] 支持 warm-up 阶段 omega_z = 0
```

---

### 6.4 原型定义

语义原型：

$$
a_i^s
=
\operatorname{Norm}
\left(
\frac{1}{K}\sum_{k\in\mathcal K_i}x_{i,k}
\right)
$$

动态原型：

$$
a_i^t
=
\operatorname{Norm}
\left(
\frac{1}{T-1}\sum_{r=1}^{T-1}|x_{i,r+1}-x_{i,r}|
\right)
$$

融合图：

warm-up：

$$
P_{ij}=0.65P_{ij}^s+0.35P_{ij}^t
$$

完整阶段：

$$
P_{ij}=0.45P_{ij}^s+0.25P_{ij}^t+0.30P_{ij}^z
$$

---

### 6.5 需要打印的 sanity log

每个 epoch 或每若干 step 记录：

```text
mean(P_s topM)
mean(P_t topM)
mean(P_z topM)
mean(P final topM)
mean(P random)
std(P final topM)
overlap(N_s, N_t)
overlap(N_final, N_s)
overlap(N_final, N_t)
```

建议额外记录：

```text
TopM planned neighbors 的类别一致率
```

仅用于分析，不参与训练。

---

### 6.6 验收标准

```text
[ ] mean(P final topM) 明显高于 mean(P random)。
[ ] P_s 与 P_t 的 top-M overlap 不应过高，否则二者信息冗余。
[ ] P_s 与 P_t 的 top-M overlap 也不应接近 0，否则融合可能噪声大。
[ ] warm-up 阶段不依赖 P_z 也能形成合理邻域。
[ ] P_ij 数值稳定在 [0, 1]。
```

---

## 7. 阶段 3：Static ARF

### 7.1 实验名

```text
E26-3: KAMCH + T-SAS + Static ARF
```

### 7.2 目标

验证非对比式 graph fitting 是否可以稳定训练。

此阶段不使用 actual retrieval trace，也不使用 feedback weight。

即：

```text
A_i 不进入训练集合
eta_m = 0
eta_f = 0
P_z = 0
```

训练集合：

$$
\mathcal S_i=\mathcal N_i\cup\mathcal R_i
$$

伪图：

$$
P=0.65P^s+0.35P^t
$$

损失：

$$
\mathcal L
=
\mathcal L_{\text{ARF-static}}
+0.10\mathcal L_{\text{quant}}
+0.05\mathcal L_{\text{balance}}
$$

---

### 7.3 实现任务

```text
[ ] 新增 losses/arf_loss.py
[ ] 实现 soft BCE with P_ij target
[ ] 实现 random anchors R_i
[ ] 实现 hash similarity prediction
[ ] 实现 quantization loss
[ ] 实现 balance loss
[ ] 禁用 view contrast 主监督
[ ] 禁用 batch/memory neighbor 原损失
[ ] 支持 stage-wise warm-up schedule
```

预测相似度：

$$
\hat P_{ij}^{v}
=
\sigma
\left(
\gamma\frac{(h_i^v)^\top\operatorname{sg}(\bar h_j)}{L}
\right)
$$

Static ARF：

$$
\mathcal L_{\text{ARF-static}}
=
\frac{1}{2B}
\sum_{v\in\{a,b\}}
\sum_i
\frac{1}{|\mathcal S_i|}
\sum_{j\in\mathcal S_i}
\operatorname{BCE}(\hat P_{ij}^{v},P_{ij})
$$

---

### 7.4 推荐配置

```yaml
training:
  objective: static_arf
  use_view_contrast: false
  use_old_neighbor_loss: false

planner:
  top_m: 20
  omega_s: 0.65
  omega_t: 0.35
  omega_z: 0.00

retrieval_environment:
  use_actual_trace: false
  random_anchors: 40

arf_loss:
  gamma: 8

loss_weights:
  lambda_quant: 0.10
  lambda_balance: 0.05
```

---

### 7.5 首批实验

```text
UCF 16-bit
```

如果稳定，再跑：

```text
UCF 32-bit
HMDB 64-bit
```

---

### 7.6 验收标准

```text
[ ] L_ARF-static 稳定下降。
[ ] quant loss 不爆炸。
[ ] balance loss 不持续升高。
[ ] bit mean 接近 0。
[ ] h 的分布逐渐接近 -1/+1。
[ ] mAP@100 不明显低于 KAMCH-Base。
```

可接受情况：

```text
Static ARF 略低于原 KAMCH，但训练稳定。
```

因为真正的创新点在下一阶段的 actual retrieval trace feedback。

不可接受情况：

```text
hash bit 塌缩；
所有样本 h 接近同一方向；
P/R/mAP 全面崩溃。
```

应对：

```text
1. 增加 random_anchors: 40 -> 80。
2. 提高 lambda_balance: 0.05 -> 0.10。
3. 降低 top_m: 20 -> 10。
4. 降低 gamma: 8 -> 5。
5. warm-up 时临时保留 0.05 old loss 作为稳定项。
```

---

## 8. 阶段 4：Full ARF without P_z

### 8.1 实验名

```text
E26-4: KAMCH + T-SAS + ARF Trace Feedback w/o P_z
```

### 8.2 目标

验证 ARF 的核心创新：

```text
planned neighbors vs actual retrieval trace
```

是否能带来进一步提升。

此阶段启用：

```text
A_i actual retrieval trace
missed-neighbor feedback
false-retrieval feedback
```

但仍然不启用：

```text
P_z memory fused graph
```

伪图保持：

$$
P=0.65P^s+0.35P^t
$$

---

### 8.3 实现任务

```text
[ ] 实现 retrieve_topR(h_i, memory_h)
[ ] 对两个 view 分别得到 A_i^a 和 A_i^b
[ ] 构造 S_i^v = N_i ∪ A_i^v ∪ R_i
[ ] 实现 missed neighbor 判断：N_i \ A_i^v
[ ] 实现 false retrieval 判断：A_i^v \ N_i
[ ] 实现 feedback weight w_ij^v
[ ] 支持 eta_m / eta_f ramp
[ ] 支持 w_max clip
```

反馈权重：

$$
w_{ij}^{v}
=
1
+
\eta_m\mathbf 1[j\in\mathcal N_i\setminus\mathcal A_i^v]P_{ij}
+
\eta_f\mathbf 1[j\in\mathcal A_i^v\setminus\mathcal N_i](1-P_{ij})
$$

完整 ARF：

$$
\mathcal L_{\text{ARF}}
=
\frac{1}{2B}
\sum_{v\in\{a,b\}}
\sum_i
\frac{1}{|\mathcal S_i^v|}
\sum_{j\in\mathcal S_i^v}
w_{ij}^{v}\operatorname{BCE}(\hat P_{ij}^{v},P_{ij})
$$

---

### 8.4 推荐配置

```yaml
planner:
  top_m: 20
  omega_s: 0.65
  omega_t: 0.35
  omega_z: 0.00

retrieval_environment:
  top_r: 20
  random_anchors: 40

feedback:
  eta_missed_start: 0.0
  eta_false_start: 0.0
  eta_missed_final: 1.0
  eta_false_final: 1.0
  ramp_epochs: 10
  weight_clip: 3.0
```

建议更保守的第一版：

```text
eta_missed_final = 1.0
eta_false_final = 0.5
```

原因：误检抑制过强可能导致训练早期过度排斥噪声邻居。

---

### 8.5 首批实验

```text
UCF 16-bit
UCF 32-bit
```

然后：

```text
HMDB 64-bit
```

---

### 8.6 验收标准

核心递进应当是：

```text
Static ARF
    <
ARF + actual trace, w=1
    <
ARF + missed feedback
    <
ARF + missed + false feedback
```

至少需要证明：

```text
Full ARF without P_z > Static ARF
```

同时监控：

```text
planned/actual overlap 上升
false retrieval ratio 下降
retrieved samples mean P_ij 上升
mAP@20-100 提升
```

---

## 9. 阶段 5：Full ARF with P_z

### 9.1 实验名

```text
E26-5: Full KAMCH + PER-SAS + ARF
```

### 9.2 目标

实现 `plan.md` 的完整主方法。

加入 memory fused representation graph：

$$
P_{ij}^z=\max(0,\cos(\bar z_i,\bar z_j))
$$

完整伪图：

$$
P_{ij}=0.45P_{ij}^s+0.25P_{ij}^t+0.30P_{ij}^z
$$

---

### 9.3 训练 schedule

#### Stage 1：Graph warm-up

```yaml
warmup_epochs: 5 或 10
omega_s: 0.65
omega_t: 0.35
omega_z: 0.00
eta_missed: 0.00
eta_false: 0.00
use_actual_trace: false
```

#### Stage 2：ARF main training

```yaml
omega_s: 0.45
omega_t: 0.25
omega_z: 0.30
eta_missed: 1.00
eta_false: 1.00
use_actual_trace: true
```

#### Stage 3：Late binarization sharpening

最后 30% epoch：

```yaml
lambda_quant: 0.20
lambda_balance: 0.05
gamma: 10
```

---

### 9.4 推荐完整配置

#### UCF / HMDB

```yaml
planner:
  top_m: 20
  omega_s: 0.45
  omega_t: 0.25
  omega_z: 0.30

retrieval_environment:
  top_r: 20
  random_anchors: 40

feedback:
  eta_missed: 1.0
  eta_false: 1.0
  weight_clip: 3.0

arf_loss:
  gamma: 8

loss_weights:
  lambda_quant: 0.10
  lambda_balance: 0.05
```

#### ActivityNet / FCVID

```yaml
planner:
  top_m: 50
  omega_s: 0.45
  omega_t: 0.25
  omega_z: 0.30

retrieval_environment:
  top_r: 50
  random_anchors: 50

feedback:
  eta_missed: 1.0
  eta_false: 1.0
  weight_clip: 3.0

arf_loss:
  gamma: 8

loss_weights:
  lambda_quant: 0.10
  lambda_balance: 0.05
```

---

### 9.5 首批完整验证

优先顺序：

```text
1. UCF 16-bit
2. UCF 32-bit
3. HMDB 64-bit
4. ActivityNet 16-bit
5. FCVID 16-bit
```

如果以上稳定，再扩展到四数据集三 bits：

```text
ActivityNet: 16 / 32 / 64
FCVID:      16 / 32 / 64
HMDB:       16 / 32 / 64
UCF:        16 / 32 / 64
```

---

### 9.6 验收标准

最低验收：

```text
Full ARF 在多数 dataset/bit 上超过 KAMCH-Base。
```

强验收：

```text
Full ARF 在 16/32-bit 短码上提升明显；
UCF mAP@20-100 得到修复；
HMDB 在强基线上继续提升；
ActivityNet/FCVID 稳定提升。
```

论文级验收：

```text
Static Graph < +Actual Trace < +Feedback Weight < Full ARF
```

这条消融链必须成立，否则 ARF 的“agentic retrieval feedback”叙事会变弱。

---

## 10. 阶段 6：完整消融实验

### 10.1 ARF 主消融

| ID | 设置 | 目的 |
|---|---|---|
| A0 | KAMCH-Base original loss | 当前强基线 |
| A1 | Static Graph，只有 $P_{ij}$，不用 $A_i$ | 普通伪图学习 |
| A2 | + Actual Trace，$S_i=N_i\cup A_i\cup R_i$，但 $w=1$ | 验证实际检索轨迹 |
| A3 | + Missed Feedback only | 验证漏检邻居修正 |
| A4 | + False Feedback only | 验证误检样本抑制 |
| A5 | Full ARF without $P_z$ | 验证非参数语义-时序图 + feedback |
| A6 | Full ARF with $P_z$ | 完整方法 |

最重要的递进：

```text
A1 < A2 < A5 < A6
```

---

### 10.2 关键帧选择消融

| ID | Selector | Training | 目的 |
|---|---|---|---|
| K0 | Uniform | 原 loss / ARF | 基础时间覆盖 |
| K1 | Random | 原 loss / ARF | 随机选择 |
| K2 | K-Medoids | 原 loss / ARF | 传统代表性 |
| K3 | Local Medoid | 原 loss / ARF | 局部代表性 |
| K4 | T-SAS | 原 loss / ARF | 时间分层集合评价 |
| K5 | PER-SAS | 原 loss / ARF | plan-evaluate-refine 表述 |

主文建议只放：

```text
Uniform
K-Medoids
T-SAS/PER-SAS
```

其他放附录。

---

### 10.3 Planner 组成消融

| ID | 伪图组成 | 目的 |
|---|---|---|
| P1 | $P^s$ only | 语义锚点图贡献 |
| P2 | $P^t$ only | 时序动态图贡献 |
| P3 | $P^s+P^t$ | 非参数双源图 |
| P4 | $P^s+P^t+P^z$ | 完整 planner |

推荐公式：

$$
P=0.45P^s+0.25P^t+0.30P^z
$$

---

### 10.4 分支结构消融

| ID | 设置 | 目的 |
|---|---|---|
| B1 | slow branch only | 验证语义分支 |
| B2 | fast branch only | 验证时序分支 |
| B3 | slow + fast concat | 简单融合 baseline |
| B4 | slow + fast content_time_lateral | 完整融合 |
| B5 | fast branch uses remaining frames only | 验证是否需要全帧时序 |
| B6 | fast branch uses all frames | 当前主设置 |

主文重点：

```text
fast branch uses all frames
```

因为慢分支已经抽取语义关键帧，快分支继续使用全部帧能保留完整动态信息。

---

### 10.5 哈希正则消融

| ID | 设置 | 目的 |
|---|---|---|
| H1 | w/o quant | 验证量化 |
| H2 | w/o balance | 验证 bit balance |
| H3 | quant + balance | 完整设置 |
| H4 | quant + balance + independence | 可选附录 |

默认不把 bit independence 放入主方法，避免 loss 变多。

---

## 11. 必须记录的机制指标

为了证明 ARF 不是普通 pseudo graph fitting，需要记录以下机制指标。

### 11.1 Planned/Actual Overlap

$$
O_i=\frac{|\mathcal N_i\cap\mathcal A_i|}{|\mathcal N_i|}
$$

预期：训练过程中逐渐上升。

含义：当前 hash retrieval trace 越来越接近 planner 的语义-时序邻域。

---

### 11.2 False Retrieval Ratio

$$
F_i=\frac{|\mathcal A_i\setminus\mathcal N_i|}{|\mathcal A_i|}
$$

预期：训练过程中下降。

含义：ARF 抑制了误检索样本。

---

### 11.3 Retrieved Samples Mean $P_{ij}$

$$
\frac{1}{|\mathcal A_i|}\sum_{j\in\mathcal A_i}P_{ij}
$$

预期：训练过程中上升。

含义：模型实际检索到的样本，在 planner graph 里也越来越可靠。

---

### 11.4 Bit Balance

$$
\left\|\frac{1}{B}\sum_i h_i\right\|_2
$$

预期：保持较低，不持续升高。

---

### 11.5 Hamming Distance Distribution

建议绘制：

```text
planned neighbors 的 Hamming distance 分布
actual retrieved samples 的 Hamming distance 分布
random anchors 的 Hamming distance 分布
```

理想现象：

```text
planned neighbors 和 actual retrieved samples 更集中在小 Hamming distance；
random anchors 更分散。
```

---

## 12. 实验优先级总表

| 优先级 | 实验 | Dataset | Bits | 目的 |
|---:|---|---|---|---|
| P0 | KAMCH-Base official recompute | all | all | 锁定强基线 |
| P1 | T-SAS + original loss | UCF | 16, 32 | 验证 selector 安全性 |
| P2 | T-SAS + original loss | HMDB | 64 | 验证 selector 在强 setting 上稳定 |
| P3 | Static ARF | UCF | 16 | 验证非对比图训练不塌缩 |
| P4 | ARF without P_z | UCF | 16, 32 | 验证 actual trace feedback |
| P5 | Full ARF with P_z | UCF | 16, 32 | 验证完整 ARF |
| P6 | Full ARF with P_z | HMDB | 64 | 验证强基线继续提升 |
| P7 | Full ARF with P_z | ActivityNet | 16, 32, 64 | 大数据集验证 |
| P8 | Full ARF with P_z | FCVID | 16, 32, 64 | 大数据集验证 |
| P9 | 完整消融 | UCF/HMDB | selected bits | 支撑论文创新 |

---

## 13. 结果判定规则

### 13.1 最理想结果

```text
Full ARF 在四数据集三 bits 上大部分超过 KAMCH-Base；
16/32-bit 提升更明显；
UCF mAP@20-100 明显修复；
机制指标 planned/actual overlap 上升；
false retrieval ratio 下降；
消融链条递进清晰。
```

这种结果足以作为强投稿版本。

---

### 13.2 可接受结果

```text
Full ARF 平均提升 0.5 到 1.5 mAP points；
多数 dataset/bit 提升；
少数 head metric 小幅下降；
中后段 mAP@40-100 和 Recall@K 提升明显；
ARF 消融成立。
```

这种结果也值得写，因为 KAMCH-Base 已经是强基线。

---

### 13.3 需要调整的结果

```text
Full ARF 不如 Static ARF；
feedback weight 不如 w=1；
planned/actual overlap 不上升；
false retrieval ratio 不下降；
hash bit 出现塌缩。
```

优先调整：

```text
1. 降低 eta_false。
2. 减小 top_m / top_r。
3. 增加 random anchors。
4. 提高 lambda_balance。
5. 延长 warm-up。
6. P_z 延后加入或减小 omega_z。
7. 使用 tiny old loss 作为过渡稳定项。
```

---

## 14. 备选路线

### 14.1 如果纯 ARF 不稳定

引入极小的旧监督作为稳定项：

$$
\mathcal L
=
\mathcal L_{\text{ARF}}
+
\epsilon\mathcal L_{\text{old}}
+
\lambda_q\mathcal L_{\text{quant}}
+
\lambda_b\mathcal L_{\text{balance}}
$$

推荐：

```text
epsilon = 0.05 or 0.10
```

论文中可以解释为 warm-up stabilization，而不是主监督。

---

### 14.2 如果 false feedback 过强

改为只使用 missed-neighbor feedback：

$$
w_{ij}^{v}=1+\eta_m\mathbf 1[j\in\mathcal N_i\setminus\mathcal A_i^v]P_{ij}
$$

此时叙事变成：

```text
ARF 主要修正应该检索但遗漏的语义-时序邻居。
```

---

### 14.3 如果 P_z 噪声大

改为：

$$
P=0.55P^s+0.35P^t+0.10P^z
$$

或者后期再加入：

```text
前 50% epoch: omega_z = 0
后 50% epoch: omega_z = 0.30
```

---

### 14.4 如果 T-SAS 不如原 selector

保留原 selector 做 baseline，不把 T-SAS 作为强贡献。

最终可以写成：

```text
ARF training is compatible with different keyframe selectors.
```

但优先仍然保留 T-SAS/PER-SAS，因为它能避免 selector 与 hash feedback 绑定。

---

## 15. 代码模块拆分建议

建议新增或改造以下模块：

```text
kamch/
  selectors/
    per_sas.py
    uniform.py
    kmedoids.py

  memory/
    memory_bank.py

  planner/
    retrieval_graph_planner.py

  losses/
    arf_loss.py
    hash_regularization.py

  metrics/
    arf_diagnostics.py

  configs/
    kamch_t_sas_original_loss.yaml
    kamch_t_sas_static_arf.yaml
    kamch_t_sas_full_arf.yaml
    kamch_per_sas_full_arf.yaml

  tools/
    run_e26_1_selector_original.sh
    run_e26_3_static_arf.sh
    run_e26_4_trace_arf_wo_pz.sh
    run_e26_5_full_arf.sh
    recompute_e26_results.sh
```

---

## 16. 配置模板

### 16.1 E26-1：T-SAS + 原训练

```yaml
experiment: e26_1_t_sas_original_loss

model:
  keyframe_selector:
    type: t_sas
    trainable: false
    share_key_idx_across_views: true
  fast_encoder:
    input_frames: all

training:
  objective: original_kamch
  epochs: 150
  lr: 3e-5
  warmup_epochs: 10
```

---

### 16.2 E26-3：Static ARF

```yaml
experiment: e26_3_static_arf

model:
  keyframe_selector:
    type: t_sas
    trainable: false
    share_key_idx_across_views: true
  fast_encoder:
    input_frames: all

training:
  objective: static_arf
  use_view_contrast: false
  use_old_neighbor_loss: false

planner:
  top_m: 20
  omega_s: 0.65
  omega_t: 0.35
  omega_z: 0.00

retrieval_environment:
  use_actual_trace: false
  random_anchors: 40

loss:
  gamma: 8
  lambda_quant: 0.10
  lambda_balance: 0.05
```

---

### 16.3 E26-4：Trace ARF without P_z

```yaml
experiment: e26_4_trace_arf_wo_pz

training:
  objective: arf
  use_view_contrast: false
  use_old_neighbor_loss: false

planner:
  top_m: 20
  omega_s: 0.65
  omega_t: 0.35
  omega_z: 0.00

retrieval_environment:
  use_actual_trace: true
  top_r: 20
  random_anchors: 40

feedback:
  eta_missed: 1.0
  eta_false: 1.0
  weight_clip: 3.0

loss:
  gamma: 8
  lambda_quant: 0.10
  lambda_balance: 0.05
```

---

### 16.4 E26-5：Full ARF

```yaml
experiment: e26_5_full_arf

training:
  objective: arf
  use_view_contrast: false
  use_old_neighbor_loss: false
  epochs: 150
  lr: 3e-5
  warmup_epochs: 10

planner:
  top_m: 20
  omega_s: 0.45
  omega_t: 0.25
  omega_z: 0.30
  warmup:
    omega_s: 0.65
    omega_t: 0.35
    omega_z: 0.00

retrieval_environment:
  use_actual_trace: true
  top_r: 20
  random_anchors: 40

feedback:
  eta_missed: 1.0
  eta_false: 1.0
  weight_clip: 3.0
  ramp_epochs: 10

loss:
  gamma: 8
  lambda_quant: 0.10
  lambda_balance: 0.05
  late_sharpen:
    start_ratio: 0.70
    lambda_quant: 0.20
    gamma: 10
```

---

## 17. 每日推进清单

### Day Block A：实现与单元测试

```text
[ ] per_sas_selector.py 完成
[ ] per_sas_selector_batch 完成
[ ] selector no_grad 测试
[ ] key_idx 分布可视化
[ ] slow/fast branch shape 测试
[ ] memory_bank.py 完成
[ ] planner graph 完成
[ ] arf_loss.py 完成
[ ] diagnostics 完成
```

### Day Block B：UCF 16-bit 快速验证

```text
[ ] E26-1 UCF 16-bit
[ ] E26-3 UCF 16-bit
[ ] E26-4 UCF 16-bit
[ ] E26-5 UCF 16-bit
[ ] official recompute
[ ] 写入实验日志
```

### Day Block C：UCF 32-bit + HMDB 验证

```text
[ ] E26-1 UCF 32-bit
[ ] E26-4 UCF 32-bit
[ ] E26-5 UCF 32-bit
[ ] E26-5 HMDB 64-bit
[ ] official recompute
```

### Day Block D：扩展全数据集

```text
[ ] ActivityNet 16/32/64
[ ] FCVID 16/32/64
[ ] HMDB 16/32/64
[ ] UCF 16/32/64
[ ] 完整 SOTA 对比表
```

### Day Block E：消融与论文材料

```text
[ ] ARF 消融表
[ ] selector 消融表
[ ] planner 消融表
[ ] branch 消融表
[ ] planned/actual overlap 曲线
[ ] false retrieval ratio 曲线
[ ] Hamming distance 分布图
[ ] 方法图
[ ] 论文方法部分初稿
```

---

## 18. 实验记录模板

每个实验必须按如下模板写入日志：

```markdown
### YYYY-MM-DD HH:MM E26-X <dataset> <bits>

method:
  - 

config:
  - selector:
  - training objective:
  - planner:
  - top_m/top_r:
  - eta_m/eta_f:
  - lambda_quant/lambda_balance:

script:
  - 

device:
  - 

output dir:
  - 

log:
  - 

checkpoint:
  - 

metrics:
  mAP@5/20/40/60/80/100:
  P@5/10/20/40/60/80/100:
  R@5/10/20/40/60/80/100:

arf diagnostics:
  planned_actual_overlap:
  false_retrieval_ratio:
  retrieved_mean_Pij:
  bit_balance_norm:
  quant_loss:

comparison:
  vs KAMCH-Base:
  vs S5VH:
  vs AutoSSVH:

conclusion:
  - 

next:
  - 
```

---

## 19. 论文叙事对齐

### 19.1 不要这样写

```text
We propose an agent for video hashing.
```

容易被质疑：没有 LLM、没有 RL、没有策略网络，不是真正 autonomous agent。

---

### 19.2 推荐这样写

```text
Inspired by agentic workflows, we formulate hash learning as a plan-act-observe-evaluate feedback process.
```

或者：

```text
We propose an agentic retrieval feedback mechanism that treats hash codes as retrieval actions and optimizes them according to the discrepancy between planned neighborhoods and actual retrieval traces.
```

---

### 19.3 核心贡献写法

```text
1. We introduce a branch-aware training-free semantic anchor selection strategy, which allocates representative and temporally balanced frames to the slow semantic branch while preserving all frames for the fast temporal branch.

2. We design a slow-fast content-time hashing architecture, where selected semantic anchors are processed by class attention and all-frame temporal dynamics are modeled by bidirectional Mamba, followed by lateral content-time fusion.

3. We propose Agentic Retrieval Feedback, a non-contrastive training paradigm that uses a memory retrieval environment to compare planned semantic-temporal neighbors with actual hash retrieval traces, converting missed and false retrievals into adaptive feedback weights for hash learning.
```

---

## 20. 最终投稿级检查清单

### 方法完整性

```text
[ ] PER-SAS/T-SAS 完整实现
[ ] slow branch selected_class_attention 正常
[ ] fast branch bidirectional_mamba 使用 all frames
[ ] content_time_lateral 正常
[ ] ARF loss 完整实现
[ ] memory bank detach 正确
[ ] P_s/P_t/P_z 正确
[ ] N_i/A_i/R_i 正确
[ ] feedback weights 正确且 clip
[ ] quant/balance 正常
```

### 实验完整性

```text
[ ] 四数据集三 bits 主结果
[ ] 与 S5VH/AutoSSVH/ConMH 等方法对比
[ ] 与 KAMCH-Base 对比
[ ] ARF 主消融
[ ] selector 消融
[ ] planner 消融
[ ] branch 消融
[ ] hash regularization 消融
[ ] 官方 recompute 结果
```

### 机制证据

```text
[ ] planned/actual overlap 曲线
[ ] false retrieval ratio 曲线
[ ] retrieved samples mean P_ij 曲线
[ ] Hamming distance 分布图
[ ] bit balance 监控
[ ] keyframe 可视化或 index 分布
```

### 写作材料

```text
[ ] 方法总图
[ ] ARF 流程图
[ ] PER-SAS 示意图
[ ] 训练伪代码
[ ] 主结果表
[ ] 消融表
[ ] 可视化图
[ ] 失败/风险分析
[ ] 与 AutoSSVH/S5VH 的差异说明
```

---

## 21. 一句话推进策略

```text
先用 T-SAS 安全替换 selector，确保不破坏强 KAMCH；
再用 Static ARF 验证非对比伪图训练稳定性；
随后启用 actual retrieval trace 和 missed/false feedback，证明 ARF 的核心价值；
最后加入 P_z memory graph，扩展到四数据集三 bits，并用递进消融证明这不是普通 pseudo graph learning，而是实际检索行为反馈驱动的哈希学习。
```
