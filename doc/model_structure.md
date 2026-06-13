# RF-CLaTH 模型结构说明

本文档按当前代码实现整理 RF-CLaTH 的模型结构。核心模型类为
`models/model.py::RetrievalFeedbackContentLateralTemporalHashing`。

当前主线模型名称：

```text
RF-CLaTH: Retrieval-Feedback Content-Lateral Temporal Hashing
```

## 1. 总体结构

当前模型可以概括为：

```text
pre-extracted frame features
  -> FeatureProjector
  -> T-SAS / PER-SAS keyframe selector
  -> slow semantic branch
  -> fast temporal branch
  -> Content-Time Lateral Fusion
  -> HashHead
  -> soft hash code / binary hash code
```

以 UCF/HMDB 主配置为例，输入不是 RGB 帧，而是预提取的帧级特征：

```text
input_type: features
num_frames: 25
num_keyframes: 5
feature_dim: 4096
hidden_dim: 512
```

输入输出形状：

```text
input features:  [B, 25, 4096]
projected x:     [B, 25, 512]
selected x_s:    [B, 5, 512]
fast tokens:     [B, 25, 512] or [B, 20, 512]
z_a / z_b:       [B, 512]
u_a / u_b:       [B, hash_bits]
binary code:     [B, hash_bits], values in {-1, +1}
```

## 2. 输入与特征投影

当前实验使用 `input_type=features`：

```yaml
model:
  input_type: features
  feature_dim: 4096
  hidden_dim: 512
```

输入张量：

```text
x_raw: [B, T, D_raw] = [B, 25, 4096]
```

首先经过 `FeatureProjector`：

```text
FeatureProjector = Linear(4096, 512)
x = FeatureProjector(x_raw)
x: [B, 25, 512]
```

代码位置：

```text
models/backbone.py::FeatureProjector
models/model.py::_encode_input
```

如果未来使用 `input_type=frames`，代码中也保留了 `ResNet50` frame backbone 路径，但当前主实验不使用该路径。

## 3. Keyframe Selector: T-SAS / PER-SAS

当前主配置：

```yaml
keyframe_selector:
  strategy: t_sas
  trainable: false
  feature_source: input
  segment_size: 5
  score_weights:
    global_repr: 0.4
    local_repr: 0.5
    local_stability: 0.1
  set_objective_weights:
    coverage: 0.6
    quality: 0.3
    redundancy: 0.1
```

### 3.1 当前实现中的 T-SAS / PER-SAS 关系

当前代码中，`t_sas` 和 `per_sas` 都会走同一个 training-free selector 实现：

```text
models/keyframe_selector.py::per_sas_selector_batch
```

因此在当前实现里：

```text
T-SAS == PER-SAS
```

区别主要是命名与论文叙事：

```text
T-SAS:
  Temporal-stratified Semantic Anchor Selection
  强调时间分层和语义锚点选择。

PER-SAS:
  Plan-Evaluate-Refine Semantic Anchor Selection
  强调 plan/evaluate/refine 的结构化解释。
```

当前没有额外的 retrieval feedback refine、planner tree 或反思迭代。因此如果输入和参数相同，`strategy=t_sas` 与
`strategy=per_sas` 的输出关键帧索引相同。

### 3.2 选择流程

对于 25 帧、5 个 keyframes，使用 `segment_size=5` 时，时间段为：

```text
[0, 1, 2, 3, 4]
[5, 6, 7, 8, 9]
[10, 11, 12, 13, 14]
[15, 16, 17, 18, 19]
[20, 21, 22, 23, 24]
```

每段选 1 帧，最终得到：

```text
selected_indices: [B, 5]
```

选择器使用 raw input features 做打分：

```text
selector_feature_source = input
```

但真正送入 slow branch 的是投影后的 token：

```text
score input: raw features       [B, 25, 4096]
select_from: projected tokens   [B, 25, 512]
selected tokens:                [B, 5, 512]
```

### 3.3 单帧质量分数

先归一化帧特征并计算帧间相似度：

```text
sim = cosine(frame_i, frame_j)
sim = clamp(sim, min=0)
```

每个 frame 的质量由三部分组成：

```text
quality =
  0.4 * global_repr
+ 0.5 * local_repr
+ 0.1 * local_stability
```

其中：

```text
global_repr:
  该帧与全视频所有帧的平均相似度。

local_repr:
  该帧与所在 segment 内其它帧的平均相似度。

local_stability:
  该帧与相邻帧的相似度，表示局部稳定性。
```

### 3.4 集合评价

选择器不是简单地逐段取最高分帧，而是枚举“每段选一帧”的所有组合，然后计算集合分数：

```text
set_score =
  0.6 * coverage
+ 0.3 * quality
- 0.1 * redundancy
```

含义：

```text
coverage:
  选出的关键帧对全视频帧的覆盖能力。

quality:
  选中关键帧自身的平均质量。

redundancy:
  选中关键帧之间的相似度，越高说明重复越多，需要惩罚。
```

最终选出集合分数最高的关键帧组合，并按时间顺序排序：

```text
selected_indices = sort(best_combination)
```

## 4. Slow Branch: Selected Class-Attention Encoder

当前 slow branch：

```yaml
slow_encoder:
  type: selected_class_attention
  token_layers: 2
  class_layers: 2
  num_queries: 2
  num_heads: 8
  mlp_ratio: 2.0
  dropout: 0.1
  pooling: attention
  use_residual_gate: true
```

输入：

```text
x_s: [B, 5, 512]
```

结构：

```text
selected keyframe tokens
  + learnable positional embedding
  -> 2-layer TransformerEncoder over selected tokens
  -> 2 learnable semantic query tokens
  -> class-attention residual blocks
  -> attention pooling over semantic queries
  -> h_s: [B, 512]
```

### 4.1 Selected token encoder

关键帧 token 首先加上 learnable positional embedding：

```text
tokens = selected + pos_embed
```

然后经过 `token_layers=2` 的 Transformer encoder。

### 4.2 Semantic query class attention

模型维护 `num_queries=2` 个 learnable semantic queries：

```text
semantic_queries: [1, 2, 512]
```

每个 class-attention block 中：

```text
Q = semantic queries
K,V = selected keyframe tokens
```

并使用 gated residual 更新 query：

```text
ctx = MultiHeadAttention(Q, K, V)
gate = sigmoid(MLP([query, ctx]))
query = query + gate * ctx
query = query + FFN(query)
```

最后对 semantic queries 做 attention pooling，得到慢分支视频级语义表示：

```text
h_s: [B, 512]
```

代码位置：

```text
models/slow_transformer.py::SelectedClassAttentionEncoder
models/slow_transformer.py::ClassAttentionResidualBlock
```

## 5. Fast Branch: Bidirectional Mamba Encoder

当前 fast branch：

```yaml
fast_encoder:
  type: bidirectional_mamba
  input_frames: all
  depth: 2
  d_state: 16
  d_conv: 4
  expand: 2
  pooling: mean
  use_official_mamba: true
  strict_official_mamba: true
```

### 5.1 all 与 remaining 两种输入模式

代码支持两种 fast 输入模式：

```text
input_frames = all:
  fast branch 使用全部帧。

input_frames = remaining:
  fast branch 使用去掉 selected keyframes 后的剩余帧。
```

代码默认值是 `remaining`，但当前主配置显式设置为 `all`。

当前主配置实际运行：

```text
slow branch: selected keyframes [B, 5, 512]
fast branch: all frames         [B, 25, 512]
```

如果改成 remaining：

```text
slow branch: selected keyframes       [B, 5, 512]
fast branch: non-keyframe remaining   [B, 20, 512]
```

对应代码：

```text
models/model.py::gather_remaining_frames_with_indices
models/model.py::forward
```

### 5.2 Fast view augmentation

训练时 fast branch 会生成两个增强视图：

```text
x_f_a, mask_a
x_f_b, mask_b
```

增强模块：

```text
models/bidirectional_mamba.py::MaskedTemporalAugmentation
```

当前配置：

```yaml
mask_ratio: 0.2
use_temporal_jitter: true
use_frame_dropout: true
frame_dropout_ratio: 0.05
use_motion_aware_mask: false
```

增强包括：

```text
temporal jitter:
  随机交换相邻帧 token。

random mask:
  按 mask_ratio 随机选择 token，用 learnable mask token 替换。

frame dropout:
  额外随机丢弃部分 token，也用 mask token 替换。
```

评估时：

```text
deterministic=True
enabled=False
```

不做 fast augmentation。

### 5.3 Bidirectional Mamba 编码

`BidirectionalMambaEncoder` 的结构：

```text
input fast tokens: [B, T_fast, 512]

forward path:
  MambaBlock x 2

backward path:
  flip temporal order
  MambaBlock x 2
  flip back

concat:
  [forward, backward] -> [B, T_fast, 1024]

projection:
  Linear(1024, 512)
  LayerNorm

pooling:
  mean pool
```

输出：

```text
fast_tokens_a: [B, T_fast, 512]
fast_tokens_b: [B, T_fast, 512]
h_f_a:         [B, 512]
h_f_b:         [B, 512]
```

在 lateral fusion 主路径里，真正用于注入 slow branch 的是 token 级输出：

```text
fast_tokens_a / fast_tokens_b
```

而不是只用池化后的 `h_f_a / h_f_b`。

代码位置：

```text
models/bidirectional_mamba.py::BidirectionalMambaEncoder
models/bidirectional_mamba.py::MambaBlock
```

## 6. Content-Time Lateral Fusion

当前 fusion：

```yaml
fusion:
  type: content_time_lateral
  lateral_temperature: 2.0
  content_temperature: 0.5
  num_time_buckets: 30
  lateral_gamma_init: 0.1
  dropout: 0.1
```

该模块是当前模型结构的关键点。它不是先分别生成完整的 slow video feature 和 fast video feature 再做 late fusion，而是：

```text
先运行 fast branch 得到 fast tokens，
再用 fast tokens 注入 selected keyframe tokens，
最后 slow encoder 处理注入后的 selected tokens。
```

实际流程：

```text
x_s:           [B, 5, 512]
fast_tokens_a: [B, T_fast, 512]
fast_indices:  [B, T_fast]
selected_idx:  [B, 5]

x_s_a = ContentTimeLateralFusion(x_s, fast_tokens_a, fast_indices, selected_idx)
h_s_a = SlowEncoder(x_s_a)
z_a   = h_s_a
```

对 view b 同理：

```text
x_s_b = ContentTimeLateralFusion(x_s, fast_tokens_b, fast_indices, selected_idx)
h_s_b = SlowEncoder(x_s_b)
z_b   = h_s_b
```

### 6.1 Content logits

先对 selected tokens 和 fast tokens 分别做投影并归一化：

```text
query = normalize(query_proj(selected_tokens))
key   = normalize(key_proj(fast_tokens))
```

计算内容相似度：

```text
content_logits = query @ key^T / content_temperature
```

形状：

```text
content_logits: [B, K, T_fast]
```

### 6.2 Temporal logits

根据 selected frame index 与 fast frame index 的时间距离：

```text
dist = abs(selected_indices - fast_indices)
```

计算时间偏置：

```text
temporal_logits = -dist / lateral_temperature + time_bias[dist_bucket]
```

其中 `time_bias` 是 learnable embedding：

```text
time_bias: Embedding(num_time_buckets, 1)
```

### 6.3 Content-time attention

总 logits：

```text
logits = content_logits + temporal_logits
```

权重：

```text
weights = softmax(logits, dim=fast_time)
```

聚合 fast tokens：

```text
lateral = weights @ fast_tokens
lateral: [B, K, 512]
```

### 6.4 Gated residual update

构造融合输入：

```text
fused_input = [
  selected_tokens,
  lateral,
  lateral - selected_tokens,
  selected_tokens * lateral
]
```

然后：

```text
update = MLP(fused_input)
gate   = sigmoid(MLP(fused_input))
```

更新 selected tokens：

```text
selected' = LayerNorm(selected + gamma * gate * update)
```

其中 `gamma` 是可训练标量，初始化为 `0.1`。

代码位置：

```text
models/fusion.py::ContentTimeLateralFusion
```

### 6.5 当前不是 late fusion

当前主配置没有开启 `fusion.final_residual`，因此最终不是：

```text
z = fuse(h_s, h_f)
```

而是：

```text
fast tokens -> update selected tokens -> slow encoder -> z
```

因此 fast branch 的作用是为 slow branch 的 semantic anchors 提供内容-时间上下文，而不是在最后一层和 slow feature 简单拼接。

## 7. Hash Head

Hash head 很简单：

```text
Linear(512, hash_bits)
tanh
```

输出 soft hash code：

```text
u_a = tanh(W z_a + b)
u_b = tanh(W z_b + b)
u_a/u_b: [B, hash_bits], values in (-1, 1)
```

评估时二值化：

```text
binary_code = sign(u)
0 -> +1
```

当前评估使用：

```text
binary format: {-1, +1}
```

代码位置：

```text
models/hash_head.py::HashHead
```

## 8. 训练 forward 流程

训练时模型执行：

```text
1. x_raw [B,25,4096]
2. x = FeatureProjector(x_raw) -> [B,25,512]
3. T-SAS/PER-SAS 选 keyframes:
     selected_indices [B,5]
     x_s [B,5,512]
4. fast branch 输入:
     all 模式:       x_fast = x [B,25,512]
     remaining 模式: x_fast = x without selected frames [B,20,512]
5. 对 x_fast 做两次随机增强:
     x_f_a, x_f_b
6. BidirectionalMambaEncoder:
     fast_tokens_a, fast_tokens_b
7. ContentTimeLateralFusion:
     x_s_a = lateral(x_s, fast_tokens_a)
     x_s_b = lateral(x_s, fast_tokens_b)
8. Slow encoder:
     h_s_a = slow_encoder(x_s_a)
     h_s_b = slow_encoder(x_s_b)
9. z_a = h_s_a, z_b = h_s_b
10. HashHead:
     u_a, u_b
```

模型 forward 输出：

```text
h_s
h_f_a, h_f_b
z_a, z_b
u_a, u_b
selected_indices
fast_indices
slow_mask
fast_mask_a, fast_mask_b
```

## 9. 评估 encode 流程

评估时：

```text
model.eval()
deterministic=True
return_one_view=True
```

因此：

```text
不做 temporal jitter
不做 random mask
不做 frame dropout
只生成 u_a
```

然后：

```text
soft_code = u_a
binary_code = sign(u_a)
```

检索时使用 binary code 计算 Hamming distance / similarity。

## 10. all-frame fast 与 remaining-fast 的结构差异

### 10.1 all-frame fast

当前主配置：

```yaml
fast_encoder:
  input_frames: all
```

结构：

```text
slow branch: selected keyframes [B,5,512]
fast branch: all frames         [B,25,512]
```

优点：

```text
fast branch 可以建模完整时序场。
Content-Time Lateral Fusion 可以从全帧中抽取上下文。
```

潜在问题：

```text
fast tokens 包含 selected keyframes 本身。
selected token 可能 attend 到同一时间位置的 fast token。
lateral fusion 可能部分退化为重复注入关键帧自身信息。
```

### 10.2 remaining-fast

早期结构和当前可选 ablation：

```yaml
fast_encoder:
  input_frames: remaining
```

结构：

```text
slow branch: selected keyframes     [B,5,512]
fast branch: remaining non-keyframes [B,20,512]
```

优点：

```text
slow/fast 信息源更互补。
slow branch 负责语义锚点。
fast branch 负责非锚点时序动态。
lateral fusion 被迫从关键帧之外的上下文帧补充信息。
```

潜在问题：

```text
fast branch 缺少关键帧位置本身。
如果 selected keyframes 也是动作关键转折点，remaining 模式可能丢掉部分强动态 token。
```

当前已经启动 HMDB16 remaining-fast 对照实验，用来判断该结构是否优于 all-frame fast baseline。

## 11. 模型结构与训练目标的边界

以下模块属于训练目标或训练环境，不改变模型 forward 结构：

```text
neighbor cache
MemoryNeighborContrastiveLoss
PlannerMemoryBank
RetrievalGraphPlanner
ARF / AUCL losses
```

也就是说，Stage1 original、warmup hard switch、AUCL v1/v2 的区别主要在 loss，不在模型结构。

模型结构固定为：

```text
T-SAS/PER-SAS selector
+ SelectedClassAttention slow branch
+ BidirectionalMamba fast branch
+ Content-Time Lateral Fusion
+ HashHead
```

## 12. 当前明确不使用的结构

当前主方法不使用：

```text
hash center
prototype alignment
prototype cache
reconstruction head
trainable keyframe selector
RGB frame backbone training path
```

这些模块不属于当前 RF-CLaTH 主线模型结构。

## 13. 当前使用的训练损失

当前 Stage1 主线使用 `RFClathLoss`，代码位置：

```text
losses/total_loss.py::RFClathLoss
```

总损失：

```text
L_total =
  0.30 * L_view
+ 0.50 * L_batch_neighbor
+ 0.04 * L_memory_neighbor
+ 0.02 * L_quant
+ 0.03 * L_balance
```

对应配置：

```yaml
loss:
  view:
    lambda: 0.3
  semantic:
    lambda_batch_neighbor: 0.5
    lambda_memory_neighbor: 0.04
  hash:
    lambda_quant: 0.02
    lambda_bit_balance: 0.03
  temperature: 0.2
  neighbor_temperature: 0.2
```

该损失可以分为三组：

```text
view:
  两视图 hash 一致性。

semantic:
  batch raw-neighbor contrastive
  memory raw-neighbor contrastive

hash:
  quantization regularization
  bit balance regularization
```

## 14. L_view: 两视图对比损失

`L_view` 使用普通 NT-Xent，对当前 batch 的两次增强视图做实例级一致性约束。

输入：

```text
u_a: [B, hash_bits]
u_b: [B, hash_bits]
```

拼接：

```text
u = concat(u_a, u_b)  # [2B, hash_bits]
```

正样本：

```text
positive(i) = 同一视频的另一增强视图
```

负样本：

```text
negative(i) = batch 内其它视频的视图
```

公式：

```text
L_view = CrossEntropy(sim(u_i, u_j) / tau, paired_view_index)
```

其中：

```text
tau = 0.2
sim = cosine similarity after L2 normalization
```

作用：

```text
保证同一个视频的两个增强视图得到一致的 hash 表示。
这是整个 hash 空间稳定性的基础约束。
```

代码位置：

```text
losses/contrastive.py::NTXentLoss
losses/contrastive.py::HashContrastiveLoss
```

## 15. L_batch_neighbor: batch 内邻居对比损失

`L_batch_neighbor` 是当前 Stage1 最重要的语义监督项，权重为 `0.50`。

它使用 raw/pre-extracted video feature 构建的静态邻居表：

```text
neighbor cache:
  cache/repartition_*_train_rawmean_top20.pt
```

其中每个训练样本都有 raw-feature nearest neighbors。训练时通过 neighbor sampler 提高同 batch 内出现邻居的概率。

输入：

```text
u_a, u_b:          [B, hash_bits]
sample_indices:   [B]
neighbor_indices: [B, topk]
```

正样本包括：

```text
1. 同一个视频的 paired view
2. 当前 batch 中出现的 raw-feature neighbors
3. symmetric_neighbors=true 时，反向邻居关系也视作 positive
```

负样本：

```text
当前 batch 中不是 positive 的其它样本视图
```

该损失是 multi-positive NT-Xent：

```text
L_i =
  logsumexp(sim(i, all_batch_candidates) / tau)
- logsumexp(sim(i, positive_batch_candidates) / tau)
```

作用：

```text
把 raw feature 空间中相近的视频拉近到 hash 空间中。
这是当前 Stage1 里最强、最稳定的语义结构来源。
```

代码位置：

```text
losses/contrastive.py::MultiPositiveNTXentLoss
losses/contrastive.py::NeighborHashContrastiveLoss
```

## 16. L_memory_neighbor: memory bank 邻居对比损失

`L_memory_neighbor` 是 batch neighbor 的 memory-bank 扩展，权重为 `0.04`。

batch neighbor 只能看到当前 mini-batch 内的样本，而 memory neighbor 可以看到更大的历史候选池。

memory bank 内容：

```text
memory: [num_train_samples, hash_bits]
valid:  [num_train_samples]
```

每个样本的 memory entry 由当前 hash embedding 的 EMA 更新：

```text
current_value = normalize(0.5 * (u_a + u_b))
memory[i] = normalize(momentum * memory[i] + (1 - momentum) * current_value)
```

当前配置：

```yaml
memory_neighbor:
  start_epoch: 2
  momentum: 0.9
  positives_per_anchor: 15
  include_self: false
```

正样本：

```text
raw-feature neighbor top15 的 valid memory entries
```

负样本：

```text
其它所有 valid memory entries
```

公式：

```text
L_memory_neighbor_i =
  logsumexp(sim(query_i, all_valid_memory) / tau)
- logsumexp(sim(query_i, raw_neighbor_memory) / tau)
```

作用：

```text
在 batch 外维持全局语义结构。
让 hash 空间不只依赖当前 mini-batch 的邻居采样。
```

近期实验表明，Stage1 warmup 中去掉 `L_memory_neighbor` 会明显削弱后续 agentic 切换效果：

```text
HMDB16 Stage1 baseline with memory:
  best mAP@100 = 0.0994

warmup60 -> AUCL v1 hard switch with Stage1 memory:
  best mAP@100 = 0.1036

warmup60 -> AUCL v1 hard switch without Stage1 memory:
  best mAP@100 = 0.0958
```

因此当前判断是：

```text
L_memory_neighbor 不是可有可无的弱辅助项。
它对构建可切换到 agentic refinement 的初始检索空间很关键。
```

代码位置：

```text
losses/contrastive.py::MemoryNeighborContrastiveLoss
```

## 17. L_quant 与 L_balance

### 17.1 L_quant

`L_quant` 让 soft hash code 靠近二值端点：

```text
L_quant = mean((|u| - 1)^2)
```

对两个视图取平均：

```text
L_quant = 0.5 * (L_quant(u_a) + L_quant(u_b))
```

作用：

```text
减少训练时 soft code 与评估时 binary code 之间的落差。
```

### 17.2 L_balance

`L_balance` 让每个 bit 在 batch 内尽量均衡，避免某些 bit 退化成常量：

```text
L_balance = mean(mean_batch(u_bit)^2)
```

对两个视图取平均：

```text
L_balance = 0.5 * (L_balance(u_a) + L_balance(u_b))
```

代码位置：

```text
losses/hash_losses.py::QuantizationLoss
losses/hash_losses.py::BalanceLoss
```

## 18. Agentic / AUCL 损失尝试

后续实验围绕一个目标展开：

```text
把 view consistency、batch neighbor、memory neighbor 和 retrieval feedback
融合成更统一的 agentic contrastive objective。
```

动机：

```text
当前 Stage1 的三个对比项是手工相加的。
Agentic 方向希望利用 planner / actual retrieval trace 产生更有信息量的 positives 和 hard negatives。
```

核心反馈集合：

```text
planned:
  planner 认为应该检索到的邻居。

actual:
  当前 hash code 实际检索到的邻居。

missed = planned - actual:
  应该近但没检索到，作为 hard positive。

false = actual - planned:
  检索到了但 planner 认为不该近，作为 hard negative。
```

## 19. AUCL v1: 单池 Agentic Unified Contrastive Loss

`AgenticUnifiedContrastiveLoss` 是第一版真正单一 InfoNCE。

代码位置：

```text
losses/arf_loss.py::AgenticUnifiedContrastiveLoss
```

候选池：

```text
C_i = current batch two-view candidates + valid planner memory bank entries
```

正样本来源：

```text
paired view same video
batch raw-feature neighbor
memory raw-feature neighbor
planner planned positives
missed hard positives
```

默认 source weights：

```text
view:              1.00
batch_neighbor:    0.75
memory_neighbor:   0.25
arf_planned:       0.25
arf_missed_bonus:  0.25
```

hard negative：

```text
false retrieval = actual - planned
denominator weight = 1.25
```

训练公式：

```text
L_i =
  logsumexp(logit_all + log denom_weight)
- logsumexp(logit_positive + log positive_weight)
```

直观含义：

```text
分子：
  所有来源给出的 positives 按 source weight 聚合。

分母：
  所有候选都参与竞争，false retrieval 在分母中加权更重。
```

实验结论：

```text
直接从 epoch 1 使用 AUCL v1 会明显变差。
原因是早期 hash retrieval trace 噪声很大，agentic feedback 会把噪声注入训练。
```

但如果先用 Stage1 建立检索空间，再硬切 AUCL v1，则当前 HMDB16 agentic 方向最好：

```text
epoch 1-60:
  Stage1 original loss

epoch 61-150:
  L_agentic_unified_contrastive_v1
+ 0.02 L_quant
+ 0.03 L_balance

HMDB16 best:
  epoch 105
  mAP@100 = 0.1036
```

该结果说明：

```text
AUCL v1 不能作为冷启动主损失。
但在 Stage1 warmup 后，它可以作为 retrieval-feedback refinement。
```

## 20. True Two-Phase AUCL

后来实现过 `PhasedAgenticUnifiedContrastiveLoss`，目标是把 warmup 与 agentic refinement
写成同一个 AUCL 内部的 source schedule，而不是外层调用旧 `RFClathLoss`。

设计口径：

```text
L_total(t) = L_AUCL(t) + L_hash
```

Phase I:

```text
source = {view, batch_neighbor, memory_neighbor}
arf_planned = 0
arf_missed_bonus = 0
hard_negative_weight = 1.0
actual_trace = false
hard_mining = false
```

Phase II:

```text
source = {view, batch_neighbor, memory_neighbor, arf_planned, arf_missed_bonus}
hard_negative_weight = 1.25
actual_trace = true
hard_mining = true
```

调度：

```text
hard switch
ramp
retain part of bootstrap sources
```

实验观察：

```text
比直接 AUCL v1 更合理，但 true unified 单池结构仍然容易破坏 Stage1 已形成的结构。
尤其在 ramp20 设置下出现明显下滑。
```

因此该方向不是当前最佳。

## 21. AUCL v2: Source-Factored Agentic Unified Contrastive Loss

为避免 v1 单池候选池过于激进，又实现了 source-factored AUCL v2。

代码位置：

```text
losses/arf_loss.py::AgenticUnifiedContrastiveLossV2
losses/contrastive.py::AgenticMemoryNeighborContrastiveLoss
```

设计：

```text
L_total =
  L_AUCL_v2
+ 0.02 L_quant
+ 0.03 L_balance

L_AUCL_v2 =
  0.30 L_view_pair
+ 0.50 L_batch_neighbor
+ 0.04 L_memory_agentic
```

其中 view 和 batch neighbor 保持 Stage1 的独立 InfoNCE，不进入单一候选池。

memory 通道改成：

```text
L_memory_agentic =
  (1 - beta) L_memory_raw
+ beta       L_memory_feedback
```

调度：

```text
epoch 1-60:
  beta = 0
  L_memory_agentic = L_memory_raw

epoch 61-80:
  beta 从 0 ramp 到 0.25

epoch 80+:
  beta = 0.25
  hard mining enabled
```

因此 epoch 80 后实际是：

```text
0.04 * L_memory_agentic
= 0.04 * (0.75 L_memory_raw + 0.25 L_memory_feedback)
= 0.03 L_memory_raw + 0.01 L_memory_feedback
```

`L_memory_feedback` 使用 memory-bank InfoNCE：

```text
positives =
  raw memory positives
+ planner planned positives top5
+ missed positives top5

hard negatives =
  actual-not-planned
```

默认权重：

```text
raw positive:      1.0
planned positive:  0.5
missed positive:   1.25
hard negative:     1.10
```

实验结果：

```text
HMDB16 AUCL v2 warm60 ramp20:
  best epoch = 115
  best mAP@100 = 0.0983
```

结论：

```text
AUCL v2 明显避免了 AUCL v1 从头训练时的崩塌。
但它没有超过 Stage1 baseline 0.0994，也低于 warmup60 hard switch 0.1036。
说明当前 feedback 信号能做到不严重伤害结构，但还没有带来稳定净收益。
```

## 22. 当前损失选择结论

目前最可靠的基础损失仍然是 Stage1 original：

```text
0.30 L_view
+ 0.50 L_batch_neighbor
+ 0.04 L_memory_neighbor
+ 0.02 L_quant
+ 0.03 L_balance
```

当前 HMDB16 agentic 方向最好结果来自：

```text
epoch 1-60:
  Stage1 original loss

epoch 61-150:
  AUCL v1 hard switch

best mAP@100 = 0.1036
```

当前不建议：

```text
1. 直接从 epoch 1 使用 AUCL v1。
2. 在 Stage1 warmup 中去掉 L_memory_neighbor。
3. 用 AUCL v2 替代 Stage1 主监督。
```

当前更合理的后续方向：

```text
1. 保留 Stage1 original loss 作为 warmup。
2. 继续围绕切换时机、切换窗口、early stop 调参。
3. 保留 L_batch_neighbor 和 L_memory_neighbor 的主结构约束。
4. 让 agentic feedback 作为后期 refinement，而不是冷启动主监督。
```
