# RF-CLaTH 当前模型架构与训练策略完整解析

本文按当前主配置 `configs/rf_clath_ucf.yaml` 和实际训练入口整理 RF-CLaTH 的完整执行链路：
数据从原始视频离线特征到训练 batch，再到 soft hash code、二值哈希码、检索反馈、
memory graph 更新、评估和训练停止，每一步经历什么策略。

当前主线名称：

```text
RF-CLaTH: Retrieval-Feedback Content-Lateral Temporal Hashing
```

当前主配置：

```text
config: configs/rf_clath_ucf.yaml
dataset: s5vh_ucf / ucf
objective: agentic_unified_contrastive
train bits: 16 / 32 / 64 由 tools/run_rf_clath_ucf_disk2.sh 循环覆盖
default hash_bits: 64
binary format: {-1, +1}
```

明确不属于当前主方法的内容：

```text
hash center
prototype alignment
prototype cache
reconstruction head
online raw-video decoder training
```

注意：当前训练代码不直接从视频文件在线抽帧训练，而是从新划分数据集中的 H5
预提取帧级特征开始。原始视频到 H5 特征这一步属于离线预处理阶段；训练框架读取的
`video` tensor 实际是一个定长帧级特征序列。

## 1. 总体流程

端到端流程可以概括为：

```text
原始视频
  -> 离线帧采样与特征抽取
  -> repartition H5 特征与标签缓存
  -> DataLoader 读取 [T, D] 视频特征序列
  -> raw-feature top-k 邻居缓存
  -> neighbor-aware mini-batch 采样
  -> FeatureProjector 映射到 hidden_dim
  -> T-SAS / PER-SAS 选择语义关键帧
  -> slow semantic branch 编码关键帧语义
  -> fast temporal branch 编码剩余帧或全帧时序
  -> content-time lateral fusion 注入快分支上下文
  -> semantic / temporal hash heads 生成 soft subcodes
  -> concat 得到完整 soft hash code
  -> Agentic controller 观察 planner/memory 状态
  -> 生成 alpha / omega / budget 动作
  -> AUCL 统一建模 view/raw/memory/planned/missed/false 信号
  -> 反向传播更新模型和 policy
  -> 写回 planner memory 与 feedback edges
  -> 周期性抽取 query/database binary hash codes
  -> Hamming 检索评估 mAP / P@K / R@K
  -> 保存 best/last checkpoint 或触发 agent stop
```

其中内层优化器仍然是 AdamW；agent 不替代梯度下降。agent 的作用是让训练过程显式
拥有 observation、action、feedback、memory、adaptation 五个环节。

## 2. 数据阶段

### 2.1 离线数据来源

当前 UCF101 主线使用新划分数据：

```text
dataset root: /mnt/disk2/yql/dataset_rePartition
train feature: ucf/ucf_train_feats.h5
query feature: ucf/ucf_test_feats.h5
database feature: ucf/ucf_train_feats.h5
train label: cache/repartition_s5vh_ucf_train_label.pt
query label: cache/repartition_s5vh_ucf_q_label.pt
database label: cache/repartition_s5vh_ucf_re_label.pt
```

UCF101 样本规模按当前约定：

```text
train/database: 9990
query: 3330
feature shape: 25 x 4096
num_classes: 101
```

训练代码中的 `S5VHFeatureDataset` 读取 array-style H5：

```text
H5 key: feats
single item:
  video: [25, 4096]
  label: [101]
  video_id: split_index 字符串
  index: 训练集内整数 id
```

如果特征长度不是 25，`sample_or_pad_sequence` 会采样或补齐到 `num_frames=25`。

### 2.2 为什么这里仍称作 video

DataLoader 返回字段叫 `video`，但在当前 `model.input_type=features` 下它表示：

```text
video = pre-extracted frame feature sequence
shape = [B, 25, 4096]
```

如果以后切到 `input_type=frames`，模型会先走 frame backbone；但当前主线直接从 H5
特征进入 `FeatureProjector`，这是为了和 S5VH/AVHash 既有协议保持一致。

### 2.3 raw-feature 邻居缓存

训练开始前，如果 `neighbor.enabled=true`，代码会读取训练集所有样本，构造 raw feature
bank：

```text
video feature: [25, 4096]
raw row: mean over temporal dimension -> [4096]
bank: normalize(raw rows)
neighbor table: cosine top-20
cache: cache/repartition_s5vh_ucf_train_rawmean_top20.pt
```

这个邻居表有两个用途：

```text
1. NeighborBatchSampler:
   让 anchor 和若干 raw-feature neighbor 进入同一个 batch，
   提高 batch 内正样本出现概率。

2. AUCL 中的 raw / batch neighbor source:
   判断当前 batch 内哪些样本对属于 raw-neighbor positives。
```

当前 batch 策略：

```text
batch_size: 256
neighbors_per_anchor: 1
topk raw neighbors: 20
drop_last: true
```

## 3. 模型总体结构

模型类：

```text
models/model.py
RetrievalFeedbackContentLateralTemporalHashing
```

当前主配置：

```text
input_type: features
num_frames: 25
num_keyframes: 5
feature_dim: 4096
hidden_dim: 512
hash_bits: 64 或运行脚本覆盖为 16 / 32 / 64
selector: t_sas
slow_encoder: selected_class_attention
fast_encoder: bidirectional_mamba
fusion: content_time_lateral
subcode concat: true
semantic_hash_ratio: 0.5
```

模型 forward 的主要输出：

```text
h_s:
  slow semantic representation

h_s_a, h_s_b:
  两个训练视图对应的 slow representation

h_f_a, h_f_b:
  两个训练视图对应的 fast temporal representation

z_a, z_b:
  融合后 representation；当前 content-time lateral 主线下基本等于 slow readout

u_s_a, u_s_b:
  semantic soft hash subcode

u_f_a, u_f_b:
  temporal soft hash subcode

u_a, u_b:
  concat 后的完整 soft hash code

selected_indices:
  每个视频选中的关键帧下标

fast_mask_a, fast_mask_b:
  两个 fast view 的 mask/dropout 位置
```

## 4. 输入映射：从 H5 特征到 hidden tokens

当前输入：

```text
x_raw: [B, 25, 4096]
```

模型先通过 `FeatureProjector` 将每帧特征映射到隐藏维：

```text
x = FeatureProjector(x_raw)
x: [B, 25, 512]
```

关键帧选择的打分源由配置控制：

```text
keyframe_selector.feature_source: input
```

因此 T-SAS / PER-SAS 使用原始 4096 维输入特征打分，但最终被选出来送入网络的是
projected hidden token。

## 5. 关键帧选择策略：T-SAS / PER-SAS

当前关键帧选择器：

```text
strategy: t_sas
trainable: false
num_frames: 25
num_keyframes: 5
segment_size: 5
```

它是 training-free 选择器，不通过梯度学习关键帧分数。每个视频被划分为 5 个时间段，
每段 5 帧，每段选 1 个 semantic anchor。

### 5.1 单帧质量分数

对每个视频，先归一化帧级特征，计算帧间相似矩阵：

```text
sim[i, j] = cosine(frame_i, frame_j), then clamp to >= 0
```

每一帧有三类质量信号：

```text
global_repr:
  该帧与所有帧的平均相似度，表示全局代表性。

local_repr:
  该帧与所在时间段内其他帧的平均相似度，表示局部代表性。

local_stability:
  该帧与相邻帧的相似度，表示局部稳定性。
```

当前权重：

```text
frame_quality =
  0.4 * global_repr
  + 0.5 * local_repr
  + 0.1 * local_stability
```

这些分数都做 min-max 归一化后再组合。

### 5.2 关键帧集合打分

选择器枚举“每段选一帧”的所有组合，然后为每个组合打分：

```text
set_score =
  0.6 * coverage
  + 0.3 * anchor_quality
  - 0.1 * redundancy
```

含义：

```text
coverage:
  被选关键帧集合对全视频帧的覆盖能力。

anchor_quality:
  被选帧自身质量的均值。

redundancy:
  被选帧之间的相互相似度，越高说明越重复，因此要扣分。
```

最终输出：

```text
selected_indices: [B, 5]
x_s: [B, 5, 512]
slow_mask: [B, 25]
```

## 6. 慢分支：Selected Class Attention

慢分支负责内容语义：

```text
input: selected tokens [B, 5, 512]
encoder: SelectedClassAttentionEncoder
token_layers: 2
class_layers: 2
num_queries: 2
num_heads: 8
pooling: attention
```

处理过程：

```text
1. 给 5 个 selected tokens 加可学习 position embedding。
2. 经过 2 层 TransformerEncoder，让关键帧之间先交互。
3. 初始化 2 个 semantic query tokens。
4. semantic queries 通过 class attention 反复读取 selected tokens。
5. 多个 semantic query 通过 attention pooling 汇聚。
6. 输出 slow representation h_s。
```

输出：

```text
h_s: [B, 512]
```

在 content-time lateral fusion 主线中，slow branch 会对两个 fast-augmented 视图分别生成：

```text
h_s_a: [B, 512]
h_s_b: [B, 512]
h_s = 0.5 * (h_s_a + h_s_b)
```

## 7. 快分支：Masked Bidirectional Mamba

快分支负责时序动态。当前配置：

```text
fast_encoder.type: bidirectional_mamba
input_frames: remaining
depth: 2
d_state: 16
d_conv: 4
expand: 2
pooling: mean
use_official_mamba: true
strict_official_mamba: true
```

### 7.1 快分支输入帧

当前配置使用：

```text
input_frames: remaining
```

也就是说：

```text
selected keyframes: 5 帧，交给 slow branch
remaining frames: 20 帧，交给 fast branch
```

代码也支持 `input_frames: all`，那时 fast branch 会读取全部 25 帧；但当前 UCF 主配置是
`remaining`。

### 7.2 两视图时序增强

训练时，fast source 会构造两个视图：

```text
x_f_a, mask_a = mask_aug(x_fast_source)
x_f_b, mask_b = mask_aug(x_fast_source)
```

当前增强：

```text
mask_ratio: 0.2
use_temporal_jitter: true
use_frame_dropout: true
frame_dropout_ratio: 0.05
use_motion_aware_mask: false
```

具体策略：

```text
temporal jitter:
  以一定概率交换相邻帧顺序。

random mask:
  随机选约 20% 时间位置，用可学习 mask token 替换。

frame dropout:
  以 0.05 比例额外替换时间位置。
```

这两个视图共享同一个 keyframe selection，但 fast augment 独立，因此生成 `u_a` 和
`u_b` 两个训练视图。

### 7.3 Bidirectional Mamba 编码

快分支核心是双向序列 mixer：

```text
forward blocks:
  从前向后建模时序。

backward blocks:
  翻转序列后从后向前建模。

merge:
  concat forward/backward token states -> linear projection -> LayerNorm。

pool:
  mean over temporal tokens。
```

当 content-time lateral fusion 需要 token-level 信息时，fast encoder 返回完整 token 序列：

```text
fast_tokens_a: [B, 20, 512]
fast_tokens_b: [B, 20, 512]
h_f_a = mean(fast_tokens_a)
h_f_b = mean(fast_tokens_b)
```

## 8. Content-Time Lateral Fusion

当前 fusion：

```text
type: content_time_lateral
lateral_temperature: 2.0
content_temperature: 0.5
num_time_buckets: 30
lateral_gamma_init: 0.1
```

这个 fusion 不是简单把 slow/fast 向量拼接，也不是最终层门控，而是在 token 层把 fast
temporal context 注入 selected slow tokens。

### 8.1 fast-to-slow token 注入

对每个 selected slow token 和每个 fast token 计算两种 logit：

```text
content_logits:
  normalized query(selected_token) dot key(fast_token) / content_temperature

temporal_logits:
  -abs(selected_frame_index - fast_frame_index) / lateral_temperature
  + learnable time_bucket_bias

final_logits:
  content_logits + temporal_logits
```

然后：

```text
weights = softmax(final_logits over fast tokens)
lateral = weighted sum of fast_tokens
delta = lateral - selected_token
fused_input = concat[selected_token, lateral, delta, selected_token * lateral]
update = MLP(fused_input)
gate = sigmoid(MLP(fused_input))
selected_token_new = LayerNorm(selected_token + gamma * gate * update)
```

这样 slow branch 仍然以关键帧为主体，但每个关键帧都能读取与自己内容相似且时间接近的
fast temporal context。

### 8.2 当前 z 的含义

在当前主配置下：

```text
use_lateral_fusion: true
final_residual: false
```

因此 `_fuse_or_bypass` 对最终 `z_a/z_b` 返回 slow representation：

```text
z_a = h_s_a
z_b = h_s_b
```

真正的 slow/fast 融合发生在 slow encoder 之前的 selected token 注入阶段，而不是最终
`z = gate(h_s, h_f)` 这种向量级融合。

## 9. 哈希码生成

当前 agentic policy 使用：

```text
use_subcode_concat: true
semantic_hash_ratio: 0.5
```

因此完整 hash code 被拆为两个子码：

```text
semantic bits = hash_bits * 0.5
temporal bits = hash_bits - semantic_bits
```

以 64-bit 为例：

```text
u_s: [B, 32]
u_f: [B, 32]
u: concat[u_s, u_f] -> [B, 64]
```

两个 hash head 都是：

```text
soft_code = tanh(linear(representation))
```

具体为：

```text
u_s_a = semantic_hash_head(h_s_a)
u_s_b = semantic_hash_head(h_s_b)
u_f_a = temporal_hash_head(h_f_a)
u_f_b = temporal_hash_head(h_f_b)

u_a = concat[u_s_a, u_f_a]
u_b = concat[u_s_b, u_f_b]
```

训练时使用 soft code 参与相似度和损失；评估/检索时通过 sign 二值化：

```text
binary_code = sign(soft_code)
0 is mapped to +1
binary format = {-1, +1}
```

## 10. 训练初始化

训练入口：

```text
train.py -> engine/train.py::train_rf_clath
```

初始化顺序：

```text
1. set_seed(project.seed)
2. 构造 output_dir 和 train.log
3. build_dataloaders(train, val, retrieval)
4. load_or_build_neighbors 构造 raw-feature top-20 邻居表
5. 用 NeighborBatchSampler 重建 train_loader
6. 构造 RF-CLaTH 模型
7. 构造 AgenticUnifiedContrastiveLoss
8. 构造 PlannerMemoryBank
9. 构造 RetrievalGraphPlanner
10. 构造 AgenticTrainingController
11. 构造 AdamW optimizer，参数包含 model + criterion + agent_controller
12. 构造 warmup + cosine scheduler
13. 如果 resume，加载 checkpoint
```

当前 optimizer：

```text
optimizer: AdamW
lr: 3e-5
weight_decay: 1e-4
grad_clip: 5.0
amp: true
warmup_epochs: 10
epochs: 150
eval_interval: 5
save_interval: 5
```

## 11. Planner Memory Bank

`PlannerMemoryBank` 是训练中的外部记忆。它不是一个可训练模块，而是 runtime state。

保存的主要内容：

```text
sem_proto_bank:
  每个样本的语义原型，来自 selected raw features 的均值。

dyn_proto_bank:
  每个样本的动态原型，来自 raw feature 的相邻帧差分均值。

z_bank:
  每个样本的融合 representation EMA。

u_bank:
  每个样本的完整 soft hash code EMA。

u_s_bank / u_f_bank:
  每个样本的 semantic / temporal subcode EMA。

route_alpha / route_omega:
  agent 最近一次为该样本产生的路由和样本权重。

update_count / last_epoch:
  样本被写入 memory 的次数和最近 epoch。

edge_indices:
  每个样本的反馈边缓存。

edge_weight / edge_posterior / edge_reliability / edge_decay / edge_flags:
  边强度、后验、可靠度、时间衰减和 planned/actual/missed/false 标志。
```

每个 batch 反向传播后，memory 更新：

```text
1. update_feedback_edges:
   使用本轮 AUCL 产生的 planned / actual / missed / false 反馈更新边。

2. update_batch:
   写入 sem_proto、dyn_proto、z、u、u_s、u_f 和 update_count。
```

当前 checkpoint 已保存模型、optimizer、scheduler、criterion 和 agent_controller；
planner memory graph 本体仍是运行时状态，resume 后会重新初始化。

## 12. Retrieval Graph Planner

`RetrievalGraphPlanner` 用 memory bank 构造检索规划信号。

候选集合：

```text
candidates = memory 中已 valid 的样本
```

三类相似度：

```text
p_s:
  semantic prototype similarity
  来自 sem_proto_bank

p_t:
  temporal dynamics prototype similarity
  来自 dyn_proto_bank

p_z:
  fused representation similarity
  来自 z_bank，可选
```

warmup 阶段和主阶段权重不同：

```text
planner warmup epochs: 10
warmup p_final = 0.65 * p_s + 0.35 * p_t + 0.00 * p_z

main p_final = 0.45 * p_s + 0.25 * p_t + 0.30 * p_z
```

planner 主要输出：

```text
planned neighbors:
  p_final top-M，当前 top_m=20。

random anchors:
  用于估计随机基准和构造更稳定的候选池，当前 random_anchors=40。

action context:
  p_s_topm、p_t_topm、p_final_topm、p_random、valid 等 per-sample 观测。
```

## 13. Agentic Controller

`AgenticTrainingController` 是外层 agent 的实现。它不使用 policy gradient，而是在普通
backprop 中通过 AUCL 和 route loss 训练一个轻量 MLP policy。

### 13.1 Observation

每个 batch 前向后，controller 从 outputs、planner 和 memory 组装状态。当前 MLP 输入
是 14 维 compact numeric state：

```text
1. p_s_topm
2. p_t_topm
3. p_final_topm
4. p_random
5. p_s_topm - p_t_topm
6. p_final_topm - p_random
7. trust
8. cold_gate
9. norm(h_s)
10. norm(h_f)
11. mean_abs(u_s)
12. mean_abs(u_f)
13. epoch_fraction
14. planner_valid
```

其中：

```text
cold_gate = clamp(update_count / cold_start_updates, 0, 1)

margin = p_final_topm - p_random

trust =
  sigmoid(trust_kappa * (margin - trust_center))
  * planner_valid
  * cold_gate
```

当前参数：

```text
cold_start_updates: 2
trust_kappa: 8.0
trust_center: 0.02
```

### 13.2 Action

controller 输出三类主要动作：

```text
alpha_i:
  semantic / temporal routed similarity 的样本级路由。

omega_i:
  该样本在 AUCL row loss 中的贡献权重。

budget_i:
  该样本 actual retrieval trace 的 top-r 预算。
```

启发式初始化：

```text
heuristic_alpha = sigmoid(alpha_kappa * (p_s_topm - p_t_topm) + alpha_bias)

heuristic_omega =
  clamp(1 + omega_scale * (trust - 0.5), omega_min, omega_max)

budget =
  round(budget_min + trust * (budget_max - budget_min))
```

当前参数：

```text
alpha_kappa: 6.0
alpha_default: 0.5
omega_min: 0.5
omega_max: 1.5
omega_scale: 0.5
budget_min: 8
budget_max: 20
```

learnable policy 是一个小 MLP：

```text
LayerNorm(14)
Linear -> SiLU
Linear -> SiLU
alpha_head
omega_head
```

两个 head 零初始化，因此训练刚开始时行为等同于启发式 controller，随后学习残差：

```text
learned_alpha =
  sigmoid(logit(heuristic_alpha) + alpha_delta)

learned_omega =
  heuristic_omega + learned_omega_delta * tanh(omega_delta)
```

### 13.3 Action 如何进入训练

`alpha` 进入 routed similarity：

```text
alpha_ij = 0.5 * (alpha_i + alpha_j)

s_agentic(i, j) =
  alpha_ij * sim_semantic(i, j)
  + (1 - alpha_ij) * sim_temporal(i, j)
```

`omega` 进入 AUCL row weight：

```text
loss_i weighted by omega_i
```

`budget` 控制 actual trace 的 per-sample top-r：

```text
top_r_i in [8, 20]
```

source weights 也会按全 batch trust 和 omega 做轻量调制：

```text
view: fixed
batch_neighbor: fixed
memory_neighbor: base * mean(omega)
arf_planned: base * mean(trust)
arf_missed_bonus: base * mean(trust)
hard_negative_weight: 1 + (base - 1) * mean(trust)
```

## 14. AUCL：统一检索反馈目标

当前主损失是 `AgenticUnifiedContrastiveLoss`，核心思想是：

```text
不要把 view、batch neighbor、memory neighbor、planned、missed、false
拆成多个互相拉扯的独立语义损失。

它们都进入同一个 source-aware multi-positive InfoNCE 目标。
```

总体目标：

```text
L_total =
  L_AUCL
  + lambda_quant * L_quant
  + lambda_balance * L_balance
  + lambda_route * L_route
```

当前权重：

```text
lambda_quant: 0.02
lambda_balance: 0.03
lambda_route: 0.005
```

### 14.1 候选池

每个 batch 有两个增强视图，因此 query 数是：

```text
query_count = 2 * batch_size
```

候选池由两部分拼接：

```text
batch candidates:
  当前 batch 的 u_a 和 u_b，共 2B 个候选。

memory candidates:
  memory.u_bank 中 valid 的全局训练样本。
```

自匹配会被 mask 掉。

### 14.2 正样本来源

AUCL 中的 positive weight matrix 由多个 source 叠加：

```text
view positive:
  同一个样本的两个增强视图互为正样本。

batch neighbor positive:
  当前 batch 内 raw-feature neighbor 表命中的样本。

memory neighbor positive:
  raw-feature neighbor 表在 memory candidates 中命中的样本。

arf planned positive:
  planner 基于 p_final 规划出的 top-M 邻居。

arf missed hard positive:
  planned 中应该检索到但 actual trace 没检索到的样本。
```

当前 source base weights：

```text
view: 1.0
batch_neighbor: 0.75
memory_neighbor: 0.25
arf_planned: 0.25
arf_missed_bonus: 0.25
max_positive_weight: 2.0
normalize_sources: true
```

### 14.3 false retrieval 如何处理

false retrieval 定义：

```text
planned = planner 认为应该近的邻居
actual = 当前 binary-like hash trace 实际检索到的邻居
missed = planned - actual
false = actual - planned
```

在 AUCL 中：

```text
missed:
  作为 hard positive 加入 numerator。

false:
  不作为单独 loss。
  它作为 hard-negative scale 加入 denominator。
```

也就是说 false 样本会让 denominator 更强，迫使当前 hash 表示把错误靠近的样本推开。

### 14.4 routed similarity

当 `agentic.policy.use_routed_similarity=true` 且存在 semantic/temporal subcodes 时，
相似度不再直接用完整 `u` 点积，而是按 agent 的 `alpha` 路由：

```text
semantic sim:
  dot(u_s_i, u_s_j) / semantic_bits

temporal sim:
  dot(u_f_i, u_f_j) / temporal_bits

routed sim:
  alpha_ij * semantic_sim
  + (1 - alpha_ij) * temporal_sim
```

对 memory candidates，如果 memory 中保存了 `u_s_bank/u_f_bank`，也使用相同 routed
similarity；memory 中保存的历史 `route_alpha` 会参与 memory pair 的 alpha 平均。

### 14.5 AUCL 计算形式

每一行 query 的 loss 是：

```text
denominator:
  所有合法候选的 exp(logit)
  false / hard-negative 候选会乘上额外 scale

numerator:
  所有 positive 候选的 exp(logit) * positive_weight

loss_i:
  logsumexp(denominator_logits) - logsumexp(positive_logits)
```

如果 sample weighting 开启：

```text
L_AUCL = weighted_mean(loss_i, weight=omega_i)
```

否则就是普通 mean。

## 15. Route Calibration Loss

`L_route` 是轻量路由校准，不是新的主语义损失。

目标：

```text
target_alpha =
  sigmoid(route_target_kappa * (p_s_topm - p_t_topm))
```

损失：

```text
L_route =
  mean over valid/trusted samples:
    trust_i * BCE(alpha_i, target_alpha_i)
```

当前配置：

```text
lambda_route: 0.005
min_trust: 0.05
collapse_only: false
entropy_low: 0.05
entropy_high: 0.98
```

含义：

```text
trust 太低的样本不强行校准；
当前设置不是只在 gate collapse 时启用，而是只要 trust 有效就轻量约束 alpha。
```

## 16. 哈希约束

哈希约束包含两个部分：

```text
L_quant:
  让 soft code 靠近 -1 / +1。

L_balance:
  让每个 bit 在 batch 内不要塌缩到同一符号。
```

实现：

```text
L_quant =
  mean((abs(u_a) - 1)^2 and (abs(u_b) - 1)^2)

L_balance =
  mean((mean_batch(u_a))^2 and (mean_batch(u_b))^2)
```

它们只负责哈希码健康，不承载主要语义关系。

## 17. 一个 mini-batch 内发生了什么

每个训练 step 的顺序：

```text
1. DataLoader 给出 batch:
   video: [B, 25, 4096]
   index: [B]

2. 模型 forward:
   生成 selected_indices、h_s_a/h_s_b、h_f_a/h_f_b、
   u_s_a/u_s_b、u_f_a/u_f_b、u_a/u_b。

3. 附加训练上下文:
   sample_indices
   epoch
   raw neighbor_indices
   planner_memory
   graph_planner

4. agent controller act:
   读取 planner.action_context 和 memory.update_count。
   计算 trust/cold_gate。
   生成 alpha/omega/budget。
   写入 outputs["agent_action"]。

5. criterion forward:
   planner 构造 planned / actual / missed / false targets。
   AUCL 构造 source-aware positive matrix。
   false retrieval 加入 denominator hard-negative scale。
   route loss 和 hash loss 加入 total。

6. backward:
   AMP scaler 反传 total loss。
   梯度更新 model、criterion 中可训练参数、agent_controller policy。

7. memory feedback update:
   使用 outputs["agent_feedback_targets"] 写 edge graph。
   写入本 batch 的 sem/dyn/z/u/u_s/u_f memory。

8. logging:
   记录 observe/action/feedback/memory/adapt 指标。
```

关键点：

```text
feedback update 发生在 optimizer step 后；
memory 中写入的表示全部 detach，不把 memory 当可微参数。
```

## 18. Feedback Edge 更新

`update_feedback_edges` 接收 AUCL 产生的 per-view targets：

```text
planned_indices / planned_scores
actual_indices / actual_scores
planned_mask / actual_mask
```

对每个 anchor，合并 planned 和 actual 后得到观察到的边：

```text
success = in_planned and in_actual
missed = in_planned and not in_actual
false = in_actual and not in_planned
```

边缓存会保留：

```text
1. 当前观察到的高优先级边。
2. 一部分旧边，避免 memory 每轮完全重写。
3. 一部分 false 边，形成 persistent false memory。
```

当前保留比例：

```text
edge_slots: 40
old_edge_reserve_ratio: 0.25
false_edge_reserve_ratio: 0.25
edge_decay_gamma: 0.98
```

posterior 和 reliability 目标大致为：

```text
success:
  posterior high, reliability high

missed:
  posterior medium, reliability medium

false:
  posterior low, reliability very low

planned only:
  positive but not fully trusted

actual only:
  suspicious / false candidate
```

这些边之后会影响 AUCL：

```text
edge_factor:
  对可靠正边提高 positive weight。

false_edge_factor:
  对持久 false 边提高 denominator hard-negative scale。
```

## 19. 训练时间线

当前主配置的时间线：

```text
epoch 1-10:
  planner warmup。
  p_final 使用 semantic/dynamic prototype，不使用 z。
  actual trace 关闭。
  missed/false feedback 权重为 0。
  memory 逐步冷启动。

epoch 11-29:
  planner 进入主权重。
  p_final = 0.45 p_s + 0.25 p_t + 0.30 p_z。
  但 AgenticUnifiedContrastiveLoss 仍因 actual_trace_start_epoch=30 关闭 actual trace。
  AUCL 主要依赖 view、batch neighbor、memory neighbor、planned 等较稳信号。

epoch >= 30:
  actual Hamming trace 开启。
  hard mining 开启。
  missed 作为 hard positive。
  false 作为 hard negative。
  feedback edges 开始显著影响 memory graph。

epoch >= 50:
  agent stop policy 满足最小 epoch 条件，可以开始判断是否提前停止。

epoch <= 150:
  如果没有 agent stop 或其他 stop，训练到最大 epoch。
```

## 20. 评估和最终哈希码

每 `eval_interval=5` 个 epoch，训练会运行：

```text
evaluate_retrieval(model, val_loader, retrieval_loader)
```

评估过程：

```text
1. model.eval()
2. 对 query split 调用 model.encode()
3. 对 retrieval/database split 调用 model.encode()
4. encode 内部 deterministic=True 且 return_one_view=True
5. 不启用 fast augmentation
6. 得到 soft_code
7. sign 得到 binary_code in {-1, +1}
8. 计算 query 到 database 的 Hamming distance
9. 按距离排序，计算 mAP、mAP@K、Precision@K、Recall@K
```

当前评估指标：

```text
P/R@K: 5, 10, 20, 40, 60, 80, 100
mAP@K: 5, 20, 40, 60, 80, 100
binary format: pm1, 即 {-1, +1}
```

`best.pth` 保存条件：

```text
metrics["mAP"] > best_map + early_stop_min_delta
```

当前：

```text
early_stop_min_delta: 0.0
```

## 21. 训练结束条件

训练可能由三类条件结束：

```text
1. 达到最大 epoch:
   train.epochs = 150。

2. agent stop:
   stop_policy.enabled = true。
   epoch >= 50 后，如果收益低、memory 稳定、gate entropy 正常、
   utility 低，并连续 3 个 eval window 成立，则停止。

3. 传统 early stop:
   early_stop_patience > 0 时启用。
   当前配置 early_stop_patience = 0，因此关闭。
```

agent stop 判断：

```text
gain = current_mAP - last_mAP
gain_ema = beta * old_gain_ema + (1 - beta) * gain
utility = gain_ema * horizon_windows - cost_weight * train_time * horizon_windows

stop_ready when:
  epoch >= min_epoch
  gain_ema < min_gain
  memory_stability >= min_memory_stability
  gate_entropy in [gate_entropy_low, gate_entropy_high]
  utility < min_utility

should_stop when:
  stop_ready 连续 patience_windows 次成立
```

当前参数：

```text
min_epoch: 50
patience_windows: 3
gain_beta: 0.7
min_gain: 0.0001
min_memory_stability: 0.95
gate_entropy_low: 0.05
gate_entropy_high: 0.98
horizon_windows: 3
cost_weight: 0.000001
min_utility: 0.0
```

训练结束后一定保存：

```text
last.pth
```

周期性保存：

```text
epoch_0005.pth, epoch_0010.pth, ...
```

最优保存：

```text
best.pth
```

## 22. 当前 checkpoint 保存内容

当前 `save_checkpoint` 保存：

```text
model.state_dict()
optimizer.state_dict()
scheduler.state_dict()
criterion.state_dict()
agent_controller.state_dict()
epoch
best_metric
cfg
```

这意味着 learnable route/sample-weight policy 会随 checkpoint 保存。

当前未完整持久化：

```text
PlannerMemoryBank runtime graph:
  sem/dyn/z/u banks
  route_alpha/route_omega
  update_count
  edge posterior/reliability/decay/flags
```

所以 resume 可以恢复模型与 optimizer，但 memory graph 会重新冷启动。这一点不影响从头
训练的完整闭环，但会影响中断恢复后的外部记忆连续性。

## 23. 当前实现的核心策略总结

从模型角度：

```text
1. 用 T-SAS/PER-SAS 选择稳定且覆盖好的语义关键帧。
2. 慢分支专注关键帧语义。
3. 快分支专注剩余帧时序动态，并用 mask/jitter/dropout 构造两个训练视图。
4. content-time lateral fusion 在 token 层把 fast context 注入 selected slow tokens。
5. semantic hash subcode 和 temporal hash subcode 分开生成，再 concat。
```

从训练角度：

```text
1. raw-feature neighbor cache 提供初始邻居监督。
2. planner memory 在训练中逐步建立 semantic/dynamic/z/hash 记忆。
3. agent 根据 planner 信号输出 alpha/omega/budget。
4. AUCL 用一个统一目标承载 view、batch、memory、planned、missed 和 false feedback。
5. false 不单独成损失，而是作为 hard-negative denominator scale。
6. hash quant/balance 保证最终二值码可用。
7. evaluation 使用 sign 后的 {-1, +1} binary code 做 Hamming retrieval。
8. agent stop 用 mAP 增益、memory 稳定性、gate entropy 和计算成本判断训练是否继续。
```

## 24. 文件对应关系

主要代码路径：

```text
configs/rf_clath_ucf.yaml
  当前 UCF 主配置。

train.py
  CLI 入口，读取配置、覆盖 dataset/hash_bits 等参数。

engine/train.py
  训练主循环、optimizer/scheduler、eval/save/stop。

datasets/video_dataset.py
  S5VH repartition H5 特征读取。

utils/neighbor.py
  raw feature neighbor cache 和 NeighborBatchSampler。

models/model.py
  RF-CLaTH 主模型 forward / encode。

models/keyframe_selector.py
  T-SAS / PER-SAS keyframe selection。

models/slow_transformer.py
  SelectedClassAttentionEncoder。

models/bidirectional_mamba.py
  MaskedTemporalAugmentation 和 BidirectionalMambaEncoder。

models/fusion.py
  ContentTimeLateralFusion。

models/hash_head.py
  tanh soft hash 和 sign binarization。

agentic/controller.py
  outer agent controller、learnable policy、stop policy。

planner/retrieval_graph_planner.py
  planned / actual / missed / false target 构造和 planner context。

memory/memory_bank.py
  runtime memory graph、edge posterior/reliability/decay。

losses/arf_loss.py
  AgenticUnifiedContrastiveLoss 和 AUCL 逻辑。

losses/hash_losses.py
  quantization 和 bit balance。

engine/extract_hash.py
  评估时抽取 soft/binary hash codes。

engine/evaluate.py
  Hamming retrieval evaluation。

utils/metrics.py
  mAP、mAP@K、Precision@K、Recall@K。

tools/run_rf_clath_ucf_disk2.sh
  远端 UCF 16/32/64-bit 训练脚本。
```

## 25. 一句话闭环理解

RF-CLaTH 当前主线不是“先学表示，再事后评估检索”，而是把检索行为放回训练过程：

```text
关键帧选择决定看哪些语义锚点；
慢/快分支分别建模内容与时序；
content-time lateral fusion 让时序上下文修正语义锚点；
subcode hash head 生成可解释的语义/时序二值子码；
planner memory 根据历史表示规划应该靠近的邻居；
actual hash trace 暴露当前二值空间真正检索到了谁；
missed / false 反馈进入 AUCL；
agent 根据反馈调整路由、样本权重和检索预算；
memory graph 写回新的边置信度；
下一轮训练再基于更新后的世界模型继续学习。
```

