# KAMCH 训练与关键帧选择整体改造计划

## 1. 目标与设计边界

本文档给出 KAMCH 的完整改造方案，目标是在保持当前模型结构基本稳定的前提下，完成两项核心调整：

1. 将原来的 `segment_rerank_gumbel_topk` 关键帧选择器替换为 **training-free 的语义锚点选择策略**，避免关键帧选择过程参与训练，也避免与基于哈希反馈的采样方法产生过强重叠。
2. 将训练监督从“以视图对比损失为主”调整为 **agentic retrieval feedback 驱动的非对比式哈希学习模式**，使模型直接围绕检索轨迹、伪检索图和哈希空间反馈进行优化。

整体设计坚持以下原则：

- 关键帧选择器不引入可学习参数。
- 关键帧选择器不使用哈希损失、重构损失或训练反馈。
- 慢分支负责高层内容语义建模。
- 快分支负责全帧时序动态建模。
- 主监督不再依赖 view-level contrastive learning。
- 训练目标保持简洁，不堆叠过多损失。
- Agent 思想用于组织训练流程，而不是引入 LLM、RL 或真正的可学习 agent。

---

## 2. 当前模型结构

KAMCH 是一个自监督视频哈希模型。输入为视频的帧级特征。训练时构造两个增强视图，分别经过共享的视频编码器，得到连续哈希表示；推理时通过符号函数得到二值码。

整体流程为：

```text
video frame features
    -> keyframe selector
    -> slow semantic branch
    -> fast temporal branch
    -> content-time lateral fusion
    -> hash head
    -> soft hash code u
    -> binary hash code b = sign(u)
```

设输入帧级特征为：

$$
F_i=\{f_{i,1}, f_{i,2}, \dots, f_{i,T}\}, \quad f_{i,t}\in\mathbb R^{2048}
$$

当前默认：

$$
T=30
$$

语义关键帧数量为：

$$
K=\frac{T}{5}=6
$$

哈希码长度为：

$$
L\in\{16,32,64,128\}
$$

模型输出 soft hash code：

$$
u_i\in\mathbb R^L
$$

训练时使用连续近似码：

$$
h_i=\tanh(u_i)
$$

推理时使用二值码：

$$
b_i=\operatorname{sign}(u_i)
$$

---

## 3. 改造后的整体框架

改造后的 KAMCH 包含五个部分：

```text
Input frame features
    ↓
Training-free PER-SAS keyframe selector
    ↓
Selected keyframes ───────────────→ Slow semantic branch
All frames ───────────────────────→ Fast temporal branch
    ↓                                  ↓
Slow semantic representation s_i       Fast temporal representation t_i
    ↓                                  ↓
        Content-time lateral fusion
                    ↓
             Video representation z_i
                    ↓
                 Hash head
                    ↓
              Soft hash code u_i
                    ↓
          Continuous code h_i = tanh(u_i)
                    ↓
          Binary code b_i = sign(u_i)
```

训练监督改为：

```text
KAMCH generates hash code
    -> memory retrieval environment
    -> planner constructs pseudo retrieval graph
    -> current hash code performs retrieval action
    -> evaluator compares planned neighbors and actual retrieval trace
    -> feedback-weighted ARF loss optimizes hash space
```

最终总损失为：

$$
\boxed{
\mathcal L
=
\mathcal L_{\text{ARF}}
+
\lambda_q\mathcal L_{\text{quant}}
+
\lambda_b\mathcal L_{\text{balance}}
}
$$

其中：

- $\mathcal L_{\text{ARF}}$：Agentic Retrieval Feedback Loss，主监督。
- $\mathcal L_{\text{quant}}$：量化损失，使连续码靠近二值码。
- $\mathcal L_{\text{balance}}$：bit balance 损失，防止 hash bit 塌缩。

默认不再使用：

- $\mathcal L_{\text{view contrast}}$ 作为主监督。
- 单独的 branch role loss。
- 单独的 reflective code target loss。
- 单独的 view stability loss。

这些思想统一合并进 $\mathcal L_{\text{ARF}}$ 和伪检索图构建中。

---

## 4. 模块一：Training-free 关键帧选择器 PER-SAS

### 4.1 模块名称

推荐名称：

```text
PER-SAS: Plan-Evaluate-Refine Semantic Anchor Selection
```

中文名称：

```text
计划—评估—反思式语义锚点选择
```

注意：这里是 agent-inspired 的确定性选择流程，不是真正的可学习 agent。

---

### 4.2 替换目标

将当前关键帧选择器：

```text
segment_rerank_gumbel_topk
```

替换为：

```text
PER-SAS / T-SAS
```

推荐先实现核心版本 T-SAS，再在论文叙事中扩展为 PER-SAS：

```text
T-SAS: Temporal-stratified Semantic Anchor Selection
PER-SAS: Plan-Evaluate-Refine Semantic Anchor Selection
```

二者底层公式可以一致，PER-SAS 只是加入 planner、candidate plan tree、evaluator、optional reflector 的结构表述。

---

### 4.3 设计动机

慢分支只处理 1/5 的帧，因此输入给慢分支的帧应该是：

- 局部片段内有代表性的帧；
- 对全局视频语义有覆盖作用的帧；
- 与其他已选帧低冗余的帧；
- 时间上均匀分布的帧；
- 局部语义相对稳定的帧。

不建议慢分支过度选择剧烈变化帧，因为快分支已经使用全部帧并通过 bidirectional Mamba 建模动态信息。

因此，慢分支关键帧选择应强调：

$$
\text{semantic representativeness}
+
\text{semantic coverage}
+
\text{low redundancy}
+
\text{temporal balance}
$$

而不是：

$$
\text{hard-frame sampling}
\quad \text{or} \quad
\text{hash-feedback sampling}
$$

---

### 4.4 时间分层

对 30 帧划分为 6 个连续时间片段：

$$
\mathcal T_b=\{5(b-1)+1,\dots,5b\}, \quad b=1,2,\dots,6
$$

即：

```text
T_1 = {1, 2, 3, 4, 5}
T_2 = {6, 7, 8, 9, 10}
T_3 = {11, 12, 13, 14, 15}
T_4 = {16, 17, 18, 19, 20}
T_5 = {21, 22, 23, 24, 25}
T_6 = {26, 27, 28, 29, 30}
```

约束为：

$$
|\mathcal K_i\cap\mathcal T_b|=1, \quad b=1,2,\dots,6
$$

即每个时间段选择 1 帧送入慢分支。

---

### 4.5 特征归一化与相似度矩阵

对帧特征进行 L2 normalization：

$$
x_{i,t}=\frac{f_{i,t}}{\|f_{i,t}\|_2}
$$

计算帧间相似度矩阵：

$$
A_{pq}=\max(0, x_{i,p}^{\top}x_{i,q})
$$

其中：

$$
p,q\in\{1,2,\dots,T\}
$$

使用非负相似度可以避免负相似度干扰集合评价。

---

### 4.6 单帧语义质量分数

#### 4.6.1 全局代表性

$$
R_p^g=\frac{1}{T}\sum_{q=1}^{T}A_{pq}
$$

表示第 $p$ 帧是否接近整个视频的全局语义中心。

#### 4.6.2 局部代表性

若第 $p$ 帧属于时间片段 $\mathcal T_{b(p)}$，则：

$$
R_p^l=\frac{1}{|\mathcal T_{b(p)}|}\sum_{q\in\mathcal T_{b(p)}}A_{pq}
$$

表示第 $p$ 帧是否能代表所在局部片段。

#### 4.6.3 局部语义稳定性

$$
U_p=
\begin{cases}
A_{1,2}, & p=1 \\
\frac{A_{p-1,p}+A_{p,p+1}}{2}, & 1<p<T \\
A_{T-1,T}, & p=T
\end{cases}
$$

$U_p$ 越大，表示该帧在局部邻域中越稳定，更适合作为语义锚点。

#### 4.6.4 归一化与综合质量

对 $R_p^g$、$R_p^l$、$U_p$ 在单个视频内部做 min-max normalization：

$$
\hat R_p^g, \quad \hat R_p^l, \quad \hat U_p
$$

定义单帧语义质量分数：

$$
q_p=0.4\hat R_p^g+0.5\hat R_p^l+0.1\hat U_p
$$

默认权重：

```text
global representativeness: 0.4
local representativeness:  0.5
local stability:           0.1
```

---

### 4.7 集合级评价函数

候选关键帧集合为：

$$
\mathcal K=\{k_1,k_2,\dots,k_6\}
$$

集合评价函数：

$$
J(\mathcal K)
=
0.6\operatorname{Cov}(\mathcal K)
+
0.3\operatorname{Qua}(\mathcal K)
-
0.1\operatorname{Red}(\mathcal K)
$$

其中语义覆盖性为：

$$
\operatorname{Cov}(\mathcal K)
=
\frac{1}{T}\sum_{p=1}^{T}\max_{k\in\mathcal K}A_{pk}
$$

锚点质量为：

$$
\operatorname{Qua}(\mathcal K)
=
\frac{1}{K}\sum_{k\in\mathcal K}q_k
$$

关键帧冗余为：

$$
\operatorname{Red}(\mathcal K)
=
\frac{2}{K(K-1)}\sum_{p<q,\ p,q\in\mathcal K}A_{pq}
$$

最终选择：

$$
\mathcal K_i^*
=
\arg\max_{|\mathcal K\cap\mathcal T_b|=1}J(\mathcal K)
$$

---

### 4.8 搜索方式

因为每段 5 帧，共 6 段，所以总组合数为：

$$
5^6=15625
$$

可以直接全枚举：

$$
\mathcal K_i^*
=
\arg\max_{k_1\in\mathcal T_1,\dots,k_6\in\mathcal T_6}
J(\{k_1,\dots,k_6\})
$$

该搜索不需要训练，不需要反向传播，不需要 Gumbel-Softmax。

---

### 4.9 训练时两个视图的 keyframe index 策略

推荐策略：

```text
先在原始帧特征上运行 PER-SAS，得到 key_idx。
然后两个增强视图共享同一个 key_idx。
```

即：

$$
\mathcal K_i^{a}=\mathcal K_i^{b}=\mathcal K_i^*
$$

这样可以避免两个增强视图因为关键帧选择抖动导致慢分支输入不一致。

如果增强包含 temporal crop 或 frame dropping，需要做 index remapping；如果增强仅包含 feature dropout、mask、Gaussian noise，则直接共享 key_idx 即可。

---

### 4.10 PER-SAS 伪代码

```python
def per_sas_selector(frame_feats, T=30, K=6):
    """
    frame_feats: Tensor [T, D], D=2048
    return: selected indices, length K
    """
    x = l2_normalize(frame_feats, dim=-1)
    A = torch.clamp(x @ x.T, min=0.0)

    segments = [range(5*b, 5*(b+1)) for b in range(K)]

    Rg = A.mean(dim=1)

    Rl = torch.zeros(T)
    for b, seg in enumerate(segments):
        seg = list(seg)
        Rl[seg] = A[seg][:, seg].mean(dim=1)

    U = torch.zeros(T)
    U[0] = A[0, 1]
    U[-1] = A[-2, -1]
    for p in range(1, T - 1):
        U[p] = 0.5 * (A[p - 1, p] + A[p, p + 1])

    Rg = minmax_norm(Rg)
    Rl = minmax_norm(Rl)
    U = minmax_norm(U)

    q = 0.4 * Rg + 0.5 * Rl + 0.1 * U

    best_score = -1e9
    best_set = None

    for cand in itertools.product(*segments):
        K_set = list(cand)

        cov = A[:, K_set].max(dim=1).values.mean()
        qua = q[K_set].mean()

        pair_sims = []
        for a in range(len(K_set)):
            for b in range(a + 1, len(K_set)):
                pair_sims.append(A[K_set[a], K_set[b]])
        red = torch.stack(pair_sims).mean()

        score = 0.6 * cov + 0.3 * qua - 0.1 * red

        if score > best_score:
            best_score = score
            best_set = K_set

    return best_set
```

---

## 5. 慢分支语义编码器

### 5.1 输入

慢分支输入为 PER-SAS 选择的关键帧：

$$
F_i^s=\{f_{i,k}\mid k\in\mathcal K_i^*\}
$$

形状为：

```text
[B, K, 2048]
```

其中：

```text
K = 6
```

---

### 5.2 selected_class_attention

慢分支使用 `selected_class_attention`。核心思想是使用 learnable semantic queries 对关键帧 token 做注意力聚合。

将关键帧投影到模型维度：

$$
e_{i,k}=W_s f_{i,k}+p_k
$$

其中：

- $W_s$：frame feature projection；
- $p_k$：可选位置编码；
- $e_{i,k}\in\mathbb R^d$。

引入 $N_q$ 个 learnable semantic queries：

$$
Q_s=\{q_1^s,q_2^s,\dots,q_{N_q}^s\}
$$

执行 cross-attention：

$$
O_i^s=\operatorname{Attention}(Q_s, E_i^s, E_i^s)
$$

其中：

$$
E_i^s=[e_{i,k_1},e_{i,k_2},\dots,e_{i,k_K}]
$$

最后聚合得到慢分支语义表示：

$$
s_i=\operatorname{Pool}(O_i^s)
$$

可选实现：

```text
Pool = mean pooling over semantic queries
Pool = first semantic query token
Pool = attention pooling over semantic queries
```

默认推荐：

```text
mean pooling over semantic queries
```

---

### 5.3 慢分支定位

慢分支的目标不是建模完整时间变化，而是从语义锚点中抽取高层内容表示：

$$
s_i=\operatorname{SlowEncoder}(F_i^s)
$$

它主要服务于：

```text
object / scene / content-level semantic representation
```

---

## 6. 快分支时序编码器

### 6.1 输入

快分支使用全部帧：

```text
model.fast_encoder.input_frames = all
```

输入为：

$$
F_i^t=F_i=\{f_{i,1},f_{i,2},\dots,f_{i,30}\}
$$

形状为：

```text
[B, T, 2048]
```

---

### 6.2 bidirectional_mamba

将所有帧投影到模型维度：

$$
e_{i,t}=W_t f_{i,t}+p_t
$$

输入 bidirectional Mamba：

$$
H_i^{\rightarrow}=\operatorname{Mamba}_{\rightarrow}(E_i^t)
$$

$$
H_i^{\leftarrow}=\operatorname{Mamba}_{\leftarrow}(E_i^t)
$$

融合双向输出：

$$
H_i^t=\operatorname{Fuse}(H_i^{\rightarrow},H_i^{\leftarrow})
$$

默认可以使用：

```text
Fuse = concat + linear
```

或者：

```text
Fuse = sum
```

再通过 mean pooling 得到时序动态表示：

$$
t_i=\frac{1}{T}\sum_{r=1}^{T}H_{i,r}^t
$$

---

### 6.3 快分支定位

快分支保留全部帧，因此负责：

```text
motion / order / temporal transition / dynamic pattern
```

与慢分支形成互补：

```text
slow branch: selected semantic anchors
fast branch: full-frame temporal dynamics
```

---

## 7. Content-Time Lateral Fusion

### 7.1 输入

慢分支输出：

$$
s_i\in\mathbb R^d
$$

快分支输出：

$$
t_i\in\mathbb R^d
$$

---

### 7.2 推荐融合形式

保留当前 `content_time_lateral` 模块，但建议在文档中解释为：

```text
slow semantic representation provides content guidance;
fast temporal representation provides dynamic guidance;
lateral fusion adaptively combines content and time information.
```

一个可写入论文的通用公式为：

$$
g_s=\sigma(W_s^g[s_i;t_i])
$$

$$
g_t=\sigma(W_t^g[s_i;t_i])
$$

$$
\tilde s_i=s_i+g_s\odot\phi_t(t_i)
$$

$$
\tilde t_i=t_i+g_t\odot\phi_s(s_i)
$$

$$
z_i=\operatorname{LN}(W_z[\tilde s_i;\tilde t_i])
$$

其中：

- $g_s$：时序信息注入语义分支的门控；
- $g_t$：语义信息注入时序分支的门控；
- $\phi_s,\phi_t$：线性投影或 MLP；
- $z_i$：最终视频级表示。

如果已有 `content_time_lateral` 实现不同，可保持实现不变，只需在论文中用上述形式解释其功能。

---

## 8. Hash Head

### 8.1 Soft hash code

视频级表示 $z_i$ 经过 hash head：

$$
u_i=\operatorname{HashHead}(z_i)
$$

其中：

$$
u_i\in\mathbb R^L
$$

训练时使用：

$$
h_i=\tanh(u_i)
$$

推理时使用：

$$
b_i=\operatorname{sign}(u_i)
$$

---

### 8.2 两个增强视图

训练时构造两个增强视图：

$$
F_i^a,\quad F_i^b
$$

通过共享编码器得到：

$$
u_i^a,\quad u_i^b
$$

连续码为：

$$
h_i^a=\tanh(u_i^a)
$$

$$
h_i^b=\tanh(u_i^b)
$$

注意：两个视图不再通过 InfoNCE 或 view contrast 作为主监督，而是共同被同一个 agentic retrieval feedback 目标约束。

---

## 9. Agentic Retrieval Feedback 训练模式

### 9.1 总体思想

将哈希学习建模为一个 agentic retrieval loop：

```text
Planner:    构造软伪检索图和计划邻居
Actor:      KAMCH 生成当前 hash code
Environment: memory bank 执行近邻检索
Observation: 当前实际检索结果
Evaluator:  比较计划邻居和实际检索结果
Reflection: 用反馈权重强化错误轨迹修正
Update:     优化 ARF loss
```

这里的 agentic 训练不是强化学习，也不是 LLM agent，而是借鉴 agent 工作流中的：

```text
plan -> act -> observe -> evaluate -> reflect -> update
```

---

### 9.2 Memory Bank

建立 memory bank，存储每个训练视频的历史表示：

```python
memory = {
    "z": fused_video_repr,        # [N, d]
    "h": continuous_hash_repr,    # [N, L]
    "b": binary_hash_code,        # [N, L]
    "sem_proto": semantic_proto,  # [N, 2048]
    "dyn_proto": dynamic_proto,   # [N, 2048]
}
```

其中：

- `z`：融合后视频表示；
- `h`：历史连续哈希码；
- `b`：历史二值哈希码；
- `sem_proto`：由语义关键帧得到的非参数语义原型；
- `dyn_proto`：由全帧差分得到的非参数动态原型。

---

### 9.3 Memory 更新

每次 batch 训练后，使用 EMA 更新 memory：

$$
\bar z_i\leftarrow m\bar z_i+(1-m)\operatorname{sg}(z_i)
$$

$$
\bar h_i\leftarrow m\bar h_i+(1-m)\operatorname{sg}(h_i)
$$

其中：

$$
h_i=\frac{h_i^a+h_i^b}{2}
$$

然后：

$$
\bar b_i=\operatorname{sign}(\bar h_i)
$$

推荐：

```text
memory momentum m = 0.95
```

训练前期可使用更小动量：

```text
m = 0.90
```

训练后期可增大：

```text
m = 0.99
```

---

## 10. Planner：构建软伪检索图

Planner 为每个视频构建一个软伪检索图：

$$
P_{ij}\in[0,1]
$$

该图不是标签，而是由三类信息共同产生：

1. 语义锚点相似度；
2. 时序动态相似度；
3. 历史融合表示相似度。

---

### 10.1 语义锚点原型

对视频 $i$，使用 PER-SAS 选出的关键帧构造非参数语义原型：

$$
a_i^s=
\operatorname{Norm}
\left(
\frac{1}{K}\sum_{k\in\mathcal K_i^*}x_{i,k}
\right)
$$

其中：

$$
x_{i,k}=\frac{f_{i,k}}{\|f_{i,k}\|_2}
$$

语义相似度为：

$$
P_{ij}^s=\max(0,\cos(a_i^s,a_j^s))
$$

---

### 10.2 时序动态原型

使用全帧差分构造动态原型：

$$
a_i^t=
\operatorname{Norm}
\left(
\frac{1}{T-1}\sum_{r=1}^{T-1}|x_{i,r+1}-x_{i,r}|
\right)
$$

时序相似度为：

$$
P_{ij}^t=\max(0,\cos(a_i^t,a_j^t))
$$

---

### 10.3 历史融合表示相似度

使用 memory 中的历史 fused representation：

$$
P_{ij}^z=\max(0,\cos(\bar z_i,\bar z_j))
$$

---

### 10.4 软伪检索图

最终：

$$
P_{ij}=\omega_sP_{ij}^s+\omega_tP_{ij}^t+\omega_zP_{ij}^z
$$

推荐权重：

```text
omega_s = 0.45
omega_t = 0.25
omega_z = 0.30
```

训练前期 memory 不稳定，可以使用 warm-up 权重：

```text
omega_s = 0.65
omega_t = 0.35
omega_z = 0.00
```

然后逐步过渡到默认权重。

---

### 10.5 计划邻居

Planner 为每个样本选择 top-M 计划邻居：

$$
\mathcal N_i=\operatorname{TopM}_{j\neq i}(P_{ij})
$$

推荐：

```text
M = 20 or 50
```

数据集较小时：

```text
M = 20
```

数据集较大时：

```text
M = 50
```

---

## 11. Actor：KAMCH 生成当前哈希动作

KAMCH 对两个增强视图输出：

$$
u_i^a,\quad u_i^b
$$

连续码：

$$
h_i^a=\tanh(u_i^a)
$$

$$
h_i^b=\tanh(u_i^b)
$$

在 agentic 视角下，当前 hash code 是模型执行的检索动作：

$$
\text{action}=h_i^v,\quad v\in\{a,b\}
$$

---

## 12. Retrieval Environment：返回实际检索轨迹

使用当前 hash code 与 memory bank 中的历史 hash code 进行检索。

当前 hash 相似度为：

$$
S_{ij}^{h,v}=\frac{(h_i^v)^\top \operatorname{sg}(\bar h_j)}{L}
$$

其中：

$$
v\in\{a,b\}
$$

实际检索结果为：

$$
\mathcal A_i^v=\operatorname{TopR}_{j\neq i}(S_{ij}^{h,v})
$$

推荐：

```text
R = 20 or 50
```

通常可以设置：

```text
R = M
```

即实际检索集合大小与计划邻居集合大小一致。

---

## 13. Evaluator 与 Reflection：反馈权重

Evaluator 比较：

```text
planned neighbors: N_i
actual retrieval trace: A_i^v
```

如果某个样本是 Planner 认为应该检索到、但当前 hash code 没有检索到，则认为是 missed neighbor。

$$
j\in\mathcal N_i\setminus\mathcal A_i^v
$$

如果某个样本是当前 hash code 检索到、但 Planner 不认为它是可靠邻居，则认为是 false retrieval。

$$
j\in\mathcal A_i^v\setminus\mathcal N_i
$$

定义反馈权重：

$$
w_{ij}^{v}
=
1
+
\eta_m\mathbf 1[j\in \mathcal N_i\setminus \mathcal A_i^v]P_{ij}
+
\eta_f\mathbf 1[j\in \mathcal A_i^v\setminus \mathcal N_i](1-P_{ij})
$$

推荐：

```text
eta_m = 1.0
eta_f = 1.0
```

为了稳定训练，可以进行截断：

$$
w_{ij}^{v}=\operatorname{clip}(w_{ij}^{v},1,w_{\max})
$$

推荐：

```text
w_max = 3.0
```

该权重就是 reflection 的数值形式：

- 漏掉应该检索到的邻居，则增强修正；
- 错误检索到不应靠近的样本，则增强抑制；
- 正常样本保持基础权重。

---

## 14. Agentic Retrieval Feedback Loss

### 14.1 训练集合

对每个样本 $i$ 和视图 $v$，构造训练集合：

$$
\mathcal S_i^v=\mathcal N_i\cup\mathcal A_i^v\cup\mathcal R_i
$$

其中：

- $\mathcal N_i$：计划邻居；
- $\mathcal A_i^v$：实际检索结果；
- $\mathcal R_i$：随机 memory anchors。

推荐随机 anchors 数量：

```text
|R_i| = 20
```

随机 anchors 的作用是防止只在局部邻域中训练，降低 hash space 塌缩风险。

---

### 14.2 预测相似度

用当前 hash code 预测 $i$ 与 memory 样本 $j$ 的相似度：

$$
\hat P_{ij}^{v}
=
\sigma
\left(
\gamma\frac{(h_i^v)^\top\operatorname{sg}(\bar h_j)}{L}
\right)
$$

其中：

- $\sigma(\cdot)$：sigmoid；
- $\gamma$：温度缩放系数；
- $L$：哈希码长度。

推荐：

```text
gamma = 8
```

可选范围：

```text
gamma = 5 ~ 10
```

---

### 14.3 Soft BCE

目标相似度为 Planner 给出的：

$$
P_{ij}\in[0,1]
$$

定义 soft binary cross entropy：

$$
\operatorname{BCE}(\hat P_{ij}^{v},P_{ij})
=
-
P_{ij}\log(\hat P_{ij}^{v})
-
(1-P_{ij})\log(1-\hat P_{ij}^{v})
$$

主损失为：

$$
\boxed{
\mathcal L_{\text{ARF}}
=
\frac{1}{2B}
\sum_{v\in\{a,b\}}
\sum_{i=1}^{B}
\frac{1}{|\mathcal S_i^v|}
\sum_{j\in\mathcal S_i^v}
 w_{ij}^{v}
\cdot
\operatorname{BCE}
(\hat P_{ij}^{v},P_{ij})
}
$$

该损失同时完成：

```text
retrieval graph fitting
missed-neighbor correction
false-retrieval suppression
two-view implicit alignment
```

因此不需要再单独使用 view contrastive loss。

---

## 15. 哈希正则

### 15.1 Quantization Loss

连续码：

$$
h_i^v=\tanh(u_i^v)
$$

量化损失：

$$
\mathcal L_{\text{quant}}
=
\frac{1}{2B}
\sum_{v\in\{a,b\}}
\sum_{i=1}^{B}
\left\||h_i^v|-\mathbf 1\right\|_1
$$

该项使连续码靠近 $-1/+1$。

---

### 15.2 Bit Balance Loss

$$
\mathcal L_{\text{balance}}
=
\frac{1}{2}
\sum_{v\in\{a,b\}}
\left\|
\frac{1}{B}\sum_{i=1}^{B}h_i^v
\right\|_2^2
$$

该项避免某些 hash bit 长期塌缩为常数。

---

### 15.3 最终损失

$$
\boxed{
\mathcal L
=
\mathcal L_{\text{ARF}}
+
\lambda_q\mathcal L_{\text{quant}}
+
\lambda_b\mathcal L_{\text{balance}}
}
$$

推荐初始权重：

```text
lambda_q = 0.10
lambda_b = 0.05
```

后期可以加强量化：

```text
lambda_q = 0.20
lambda_b = 0.05
```

---

## 16. 训练阶段设计

### 16.1 Stage 0：Memory 初始化

在训练开始前，可以先用当前随机初始化模型跑一遍训练集，初始化 memory；但随机模型的 `z` 和 `h` 不可靠。因此更推荐：

```text
sem_proto 和 dyn_proto 直接由原始帧特征初始化；
z 和 h 初始为零向量或首轮前向输出；
前期 pseudo graph 不使用 P_z。
```

初始化内容：

```text
memory.sem_proto = semantic anchors prototype
memory.dyn_proto = full-frame dynamic prototype
memory.z = zeros or first forward z
memory.h = zeros or first forward h
memory.b = sign(memory.h)
```

---

### 16.2 Stage 1：Graph Warm-up

前若干 epoch 不启用完整 retrieval feedback，仅使用 Planner 的软伪检索图进行训练。

推荐：

```text
warmup epochs = 5 ~ 10
```

此时：

```text
omega_s = 0.65
omega_t = 0.35
omega_z = 0.00
eta_m = 0.00
eta_f = 0.00
```

训练集合：

$$
\mathcal S_i^v=\mathcal N_i\cup\mathcal R_i
$$

损失仍然使用：

$$
\mathcal L_{\text{ARF}}
+
\lambda_q\mathcal L_{\text{quant}}
+
\lambda_b\mathcal L_{\text{balance}}
$$

但此时没有 actual retrieval trace feedback，主要用于稳定初始化 hash space 和 memory bank。

---

### 16.3 Stage 2：Agentic Retrieval Feedback Training

从 warm-up 后启用完整流程：

```text
omega_s = 0.45
omega_t = 0.25
omega_z = 0.30
eta_m = 1.00
eta_f = 1.00
```

此时训练集合变为：

$$
\mathcal S_i^v=\mathcal N_i\cup\mathcal A_i^v\cup\mathcal R_i
$$

模型开始利用 actual retrieval trace：

```text
planned neighbors vs actual retrieved neighbors
```

产生反馈权重 $w_{ij}^v$。

---

### 16.4 Stage 3：Binarization Sharpening

训练后期强化量化，使 soft code 更接近最终二值码。

推荐最后 30% epoch：

```text
lambda_q = 0.20
lambda_b = 0.05
gamma = 10
```

其他保持不变。

---

## 17. 每个 Batch 的训练伪代码

```python
for batch in dataloader:
    video_ids, frame_feats = batch
    # frame_feats: [B, T, 2048]

    # --------------------------------------------------
    # 1. training-free keyframe selection
    # --------------------------------------------------
    # compute on original frame features before stochastic augmentation
    key_idx = per_sas_selector_batch(frame_feats)

    # --------------------------------------------------
    # 2. two augmented views
    # --------------------------------------------------
    view_a = augment(frame_feats)
    view_b = augment(frame_feats)

    # both views share the same key_idx

    # --------------------------------------------------
    # 3. KAMCH forward
    # --------------------------------------------------
    out_a = model(view_a, key_idx=key_idx, fast_input="all")
    out_b = model(view_b, key_idx=key_idx, fast_input="all")

    u_a, u_b = out_a["u"], out_b["u"]
    z_a, z_b = out_a["z"], out_b["z"]

    h_a = torch.tanh(u_a)
    h_b = torch.tanh(u_b)
    h = 0.5 * (h_a + h_b)
    z = 0.5 * (z_a + z_b)

    # --------------------------------------------------
    # 4. Planner: build pseudo retrieval graph
    # --------------------------------------------------
    P, N = planner(
        video_ids=video_ids,
        sem_proto=current_sem_proto,
        dyn_proto=current_dyn_proto,
        memory=memory,
        top_m=M,
        omega_s=omega_s,
        omega_t=omega_t,
        omega_z=omega_z,
    )

    # --------------------------------------------------
    # 5. Environment: retrieve actual hash neighbors
    # --------------------------------------------------
    A_a = retrieve_topR(h_a, memory["h"], top_r=R)
    A_b = retrieve_topR(h_b, memory["h"], top_r=R)

    # --------------------------------------------------
    # 6. Evaluator / Reflection: feedback weights
    # --------------------------------------------------
    S_a = build_training_set(N, A_a, random_anchors)
    S_b = build_training_set(N, A_b, random_anchors)

    w_a = compute_feedback_weight(N, A_a, P, eta_m, eta_f, w_max=3.0)
    w_b = compute_feedback_weight(N, A_b, P, eta_m, eta_f, w_max=3.0)

    # --------------------------------------------------
    # 7. Loss
    # --------------------------------------------------
    L_arf_a = arf_loss(h_a, memory["h"], P, S_a, w_a, gamma=gamma)
    L_arf_b = arf_loss(h_b, memory["h"], P, S_b, w_b, gamma=gamma)
    L_arf = 0.5 * (L_arf_a + L_arf_b)

    L_quant = quantization_loss(h_a, h_b)
    L_balance = balance_loss(h_a, h_b)

    loss = L_arf + lambda_q * L_quant + lambda_b * L_balance

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # --------------------------------------------------
    # 8. Update memory
    # --------------------------------------------------
    memory.update(
        ids=video_ids,
        z=z.detach(),
        h=h.detach(),
        momentum=m,
    )
```

---

## 18. 推荐配置

### 18.1 模型配置

```yaml
model:
  name: KAMCH
  frame_dim: 2048
  num_frames: 30
  hash_bits: 64

  keyframe_selector:
    type: per_sas
    trainable: false
    num_keyframes: 6
    segment_size: 5
    use_hash_feedback: false
    score_weights:
      global_repr: 0.4
      local_repr: 0.5
      local_stability: 0.1
    set_objective_weights:
      coverage: 0.6
      quality: 0.3
      redundancy: 0.1

  slow_encoder:
    type: selected_class_attention
    input_frames: selected
    num_semantic_queries: 4
    pooling: mean

  fast_encoder:
    type: bidirectional_mamba
    input_frames: all
    pooling: mean

  fusion:
    type: content_time_lateral

  hash_head:
    activation_train: tanh
    activation_eval: sign
```

---

### 18.2 训练配置

```yaml
training:
  objective: agentic_retrieval_feedback
  use_view_contrast: false
  use_hash_feedback_for_selector: false

  two_views:
    share_keyframe_indices: true

  memory_bank:
    enabled: true
    momentum: 0.95
    store_z: true
    store_h: true
    store_b: true
    store_sem_proto: true
    store_dyn_proto: true

  planner:
    top_m: 50
    omega_s: 0.45
    omega_t: 0.25
    omega_z: 0.30

  retrieval_environment:
    top_r: 50
    random_anchors: 20

  evaluator:
    eta_missed: 1.0
    eta_false: 1.0
    weight_clip: 3.0

  arf_loss:
    gamma: 8

  loss_weights:
    lambda_quant: 0.10
    lambda_balance: 0.05

  schedule:
    warmup_epochs: 5
    warmup_omega_s: 0.65
    warmup_omega_t: 0.35
    warmup_omega_z: 0.00
    warmup_eta_missed: 0.00
    warmup_eta_false: 0.00
    late_sharpen_ratio: 0.30
    late_lambda_quant: 0.20
    late_gamma: 10
```

---

## 19. 与原始 KAMCH 的差异

### 19.1 关键帧选择器

原始：

```text
segment_rerank_gumbel_topk
```

改造后：

```text
PER-SAS / T-SAS
```

核心差异：

| 项目 | 原始 selector | 改造后 selector |
|---|---|---|
| 是否训练 | 可能涉及可导选择 | 否 |
| 是否使用 Gumbel | 是 | 否 |
| 是否使用 hash feedback | 可能可扩展 | 否 |
| 是否保证时间覆盖 | 是/部分 | 强保证 |
| 是否强调集合覆盖 | 不一定 | 是 |
| 是否适合慢分支 | 较适合 | 更明确 |

---

### 19.2 训练监督

原始：

$$
\mathcal L
=
\lambda_{view}\mathcal L_{view\_contrast}
+
\lambda_{batch}\mathcal L_{batch\_neighbor}
+
\lambda_{memory}\mathcal L_{memory\_neighbor}
+
\lambda_{quant}\mathcal L_{quant}
+
\lambda_{balance}\mathcal L_{balance}
$$

改造后：

$$
\mathcal L
=
\mathcal L_{ARF}
+
\lambda_q\mathcal L_{quant}
+
\lambda_b\mathcal L_{balance}
$$

核心差异：

| 项目 | 原始训练 | 改造后训练 |
|---|---|---|
| 主监督 | view contrast + pseudo neighbor | agentic retrieval feedback |
| 是否依赖负样本对比 | 是 | 否 |
| 是否显式使用实际检索轨迹 | 弱 | 强 |
| memory 角色 | 近邻监督 | retrieval environment |
| agent 思想 | 无 | planner-actor-environment-evaluator-reflection |
| loss 数量 | 多 | 少 |

---

## 20. 消融实验计划

### 20.1 关键帧选择消融

| 编号 | 方法 | 目的 |
|---|---|---|
| K1 | Uniform Sampling | 基础时间覆盖 baseline |
| K2 | Random Sampling | 随机选择 baseline |
| K3 | K-Medoids | 传统代表性 baseline |
| K4 | Local Medoid | 只用局部代表性 |
| K5 | T-SAS | 时间分层 + 集合评价 |
| K6 | PER-SAS | 计划—评估—反思式选择 |

建议主文只保留 K1、K3、K5/K6，其他放附录。

---

### 20.2 训练目标消融

| 编号 | 方法 | 目的 |
|---|---|---|
| T1 | 原始 KAMCH contrastive loss | 原始训练 baseline |
| T2 | 只用 planner graph，无 actual retrieval trace | 验证伪检索图的作用 |
| T3 | ARF without feedback weights | 验证检索轨迹集合的作用 |
| T4 | ARF with missed-neighbor feedback only | 验证漏检修正 |
| T5 | ARF with false-retrieval feedback only | 验证误检抑制 |
| T6 | Full ARF | 完整 agentic retrieval feedback |

最关键对比：

```text
T2 vs T6
```

用于证明方法不是普通伪邻居学习，而是利用 actual retrieval trace 的 agentic feedback。

---

### 20.3 分支结构消融

| 编号 | 方法 | 目的 |
|---|---|---|
| B1 | slow branch only | 验证语义分支贡献 |
| B2 | fast branch only | 验证时序分支贡献 |
| B3 | slow + fast concat fusion | 验证简单融合 |
| B4 | slow + fast content_time_lateral | 验证 lateral fusion |
| B5 | fast branch uses remaining frames only | 对比是否需要全帧时序 |
| B6 | fast branch uses all frames | 当前推荐设置 |

当前推荐：

```text
model.fast_encoder.input_frames = all
```

因为快分支使用全部帧更能保留完整动态信息，慢分支只抽语义锚点。

---

### 20.4 Planner 组成消融

| 编号 | 伪图组成 | 目的 |
|---|---|---|
| P1 | semantic only $P^s$ | 验证语义锚点图 |
| P2 | temporal only $P^t$ | 验证动态结构图 |
| P3 | semantic + temporal | 验证双源非参数图 |
| P4 | semantic + temporal + memory z | 完整 planner |

推荐主方法使用：

$$
P_{ij}=0.45P_{ij}^s+0.25P_{ij}^t+0.30P_{ij}^z
$$

---

### 20.5 哈希正则消融

| 编号 | 设置 | 目的 |
|---|---|---|
| H1 | no quant | 验证量化损失 |
| H2 | no balance | 验证 bit balance |
| H3 | quant + balance | 完整设置 |
| H4 | quant + balance + bit independence | 可选增强 |

默认不使用 bit independence，避免 loss 变多。

---

## 21. 预期创新点

### 21.1 Branch-aware training-free semantic anchor selection

PER-SAS 不是为了选择最难帧，而是为了给慢分支分配高质量语义锚点，同时让快分支保留完整帧序列。

可以总结为：

```text
semantic anchors for slow branch;
all-frame dynamics for fast branch.
```

---

### 21.2 Agentic retrieval feedback hashing

KAMCH 不再以 view contrastive learning 为主要监督，而是将 hash code 看作检索动作：

```text
hash code as action
memory bank as retrieval environment
retrieved top-R as observation
planned neighbors as goal
evaluator feedback as reflection
```

核心是：

$$
\text{planned retrieval structure}
\quad vs. \quad
\text{actual hash retrieval trace}
$$

---

### 21.3 简洁的非对比式训练目标

最终损失只有：

$$
\mathcal L
=
\mathcal L_{ARF}
+
\lambda_q\mathcal L_{quant}
+
\lambda_b\mathcal L_{balance}
$$

相比多损失堆叠，该目标更集中、更容易解释。

---

## 22. 可能风险与应对

### 22.1 Early memory 不稳定

风险：训练前期 $\bar z$、$\bar h$ 不可靠。

应对：

```text
warm-up 阶段不使用 P_z；
warm-up 阶段不使用 actual retrieval trace feedback；
只用 P_s + P_t 构建 planner graph。
```

---

### 22.2 Hash code 塌缩

风险：非对比训练如果目标图过平滑，hash bit 可能塌缩。

应对：

```text
保留 quantization loss；
保留 balance loss；
加入 random anchors；
控制 top-M/top-R 不要过大；
后期提升 lambda_quant。
```

---

### 22.3 Pseudo graph 噪声

风险：伪图 $P_{ij}$ 不是真实标签，可能包含错误邻居。

应对：

```text
使用 semantic + temporal + memory 三源融合；
训练前期不用 memory z；
只保留 top-M 可靠邻居；
ARF 使用 soft target，不使用硬 0/1 标签。
```

---

### 22.4 Feedback weight 过强

风险：$w_{ij}$ 太大会导致训练不稳定。

应对：

```text
w_max = 3.0；
eta_m 和 eta_f 从 0 ramp 到 1；
先启用 missed feedback，再启用 false feedback。
```

---

### 22.5 Agent 包装过度

风险：审稿人认为方法并不是真正 agent。

应对：

正文中使用：

```text
agent-inspired retrieval feedback
agentic workflow
```

避免直接声称：

```text
we train an autonomous agent
we use multi-agent reasoning
we use LLM agent
```

推荐表述：

```text
The proposed training scheme is inspired by agentic workflows, but it is implemented as a deterministic and non-parametric retrieval feedback mechanism.
```

---

## 23. 实现路线

### 23.1 第一阶段：替换关键帧选择器

目标：先不改训练损失，只替换 selector，确认模型能正常训练。

任务：

```text
1. 实现 per_sas_selector.py
2. 在 dataloader 或 model forward 前计算 key_idx
3. 两个增强视图共享 key_idx
4. 慢分支输入 selected keyframes
5. 快分支继续输入 all frames
6. 复现实验 baseline
```

验收：

```text
PER-SAS 不降低或小幅提升原始 KAMCH 性能；
训练速度可接受；
keyframe indices 分布满足每段 1 帧。
```

---

### 23.2 第二阶段：实现 Planner Graph

任务：

```text
1. 计算 semantic proto
2. 计算 dynamic proto
3. 建立 memory bank
4. 构造 P_s, P_t, P_z
5. 得到 P_ij 和 N_i
```

先实现：

```text
P = 0.65 * P_s + 0.35 * P_t
```

再加入：

```text
P_z
```

验收：

```text
top-M neighbors 的平均 P_ij 明显高于随机样本；
不同 epoch 的 N_i 不剧烈抖动；
P_s 与 P_t 有互补性。
```

---

### 23.3 第三阶段：实现 ARF Loss

任务：

```text
1. 实现 memory hash retrieval
2. 得到 A_i^a 和 A_i^b
3. 构造 S_i^v = N_i ∪ A_i^v ∪ R_i
4. 计算 feedback weight w_ij^v
5. 实现 soft BCE ARF loss
6. 加入 quant 和 balance
```

先跑：

```text
eta_m = 0
eta_f = 0
```

确认稳定后启用：

```text
eta_m = 1
eta_f = 1
```

验收：

```text
L_ARF 稳定下降；
hash bit balance 不塌缩；
retrieval overlap |N_i ∩ A_i| 逐渐上升。
```

---

### 23.4 第四阶段：完整训练与消融

任务：

```text
1. 完整训练 KAMCH + PER-SAS + ARF
2. 与原始 contrastive KAMCH 对比
3. 做关键帧选择消融
4. 做 ARF 消融
5. 做分支结构消融
6. 做 hash bits 对比
```

验收：

```text
mAP / Precision@K / Recall@K 等检索指标提升；
短码长度下提升尤其明显；
消融能证明 actual retrieval feedback 有贡献。
```

---

## 24. 论文方法部分可用描述

### 24.1 英文描述

```text
We propose KAMCH, a self-supervised video hashing framework that separates content abstraction and temporal dynamics through a slow-fast architecture. A training-free semantic anchor selector first allocates representative and temporally balanced keyframes to the slow semantic branch, while the fast temporal branch processes the full frame sequence with a bidirectional Mamba encoder. The two representations are then integrated by a content-time lateral fusion module and projected into compact hash codes.

Instead of relying on view-level contrastive learning as the dominant self-supervision, we formulate hash learning as an agentic retrieval feedback process. The model first acts by producing hash codes, which interact with a non-parametric memory-bank retrieval environment. A planner constructs a soft semantic-temporal retrieval graph from semantic anchors, temporal dynamics, and historical representations. The environment returns the actual top-ranked retrieval trace induced by current hash codes, and an evaluator compares it with the planned neighborhood. Missed neighbors and false retrievals are assigned larger feedback weights, leading to an Agentic Retrieval Feedback loss that directly optimizes the retrieval structure in the hash space. The final objective consists of this single retrieval feedback loss and two standard hash regularizers for quantization and bit balance.
```

---

### 24.2 中文描述

```text
本文提出 KAMCH，一种自监督视频哈希框架，通过慢快双分支结构解耦内容语义和时序动态。首先，训练无关的语义锚点选择器从输入帧序列中选择具有局部代表性、全局覆盖性和时间均衡性的关键帧，并将其送入慢分支语义编码器；快分支则使用全部帧，通过双向 Mamba 编码器建模完整时序动态。随后，内容—时间横向融合模块将两种表示融合，并由哈希头映射为紧凑哈希码。

不同于以增强视图对比学习为主的训练方式，本文将哈希学习建模为一个智能体启发的检索反馈过程。模型首先生成哈希码，并利用该哈希码与由 memory bank 构成的非参数检索环境交互。Planner 根据语义锚点、时序动态和历史表示构建软语义—时序检索图；检索环境返回当前哈希码产生的实际 top-ranked retrieval trace；Evaluator 比较计划邻域和实际检索结果，并对漏检邻居和误检样本赋予更高反馈权重。最终，模型通过 Agentic Retrieval Feedback loss 直接优化哈希空间中的检索结构，同时使用量化损失和 bit balance 损失保证二值码质量。
```

---

## 25. 最终推荐版本

最终建议采用以下组合：

```text
Keyframe selector:
    PER-SAS / T-SAS
    training-free
    no hash feedback
    one keyframe per 5-frame segment

Slow branch:
    selected_class_attention
    input = selected keyframes

Fast branch:
    bidirectional_mamba
    input = all frames

Fusion:
    content_time_lateral

Training:
    no main view contrastive loss
    use Agentic Retrieval Feedback Loss
    keep quantization and balance only
```

最终损失：

$$
\boxed{
\mathcal L
=
\mathcal L_{\text{ARF}}
+0.10\mathcal L_{\text{quant}}
+0.05\mathcal L_{\text{balance}}
}
$$

后期：

$$
\boxed{
\mathcal L
=
\mathcal L_{\text{ARF}}
+0.20\mathcal L_{\text{quant}}
+0.05\mathcal L_{\text{balance}}
}
$$

最核心的论文贡献可以概括为三点：

1. **Branch-aware semantic anchor selection**：为慢分支设计训练无关的语义锚点选择，使慢分支获得紧凑、低冗余、时间均衡的内容输入。
2. **Slow-fast content-time hashing architecture**：慢分支建模语义内容，快分支建模全帧动态，并通过 lateral fusion 生成视频级哈希表示。
3. **Agentic retrieval feedback training**：将哈希码视为检索动作，通过 memory retrieval environment 返回实际检索轨迹，并根据 planned neighbors 与 actual retrieval trace 的偏差生成反馈权重，直接优化哈希空间检索结构。

---

## 26. 最小可行实验版本

如果需要先快速验证，建议实现最小版本：

```text
Selector:
    T-SAS full enumeration

Pseudo graph:
    P = 0.65 * P_s + 0.35 * P_t
    no P_z in first version

Training set:
    S_i = N_i ∪ random anchors
    no actual A_i in first 5 epochs

Then enable:
    A_i actual retrieval trace
    feedback weight w_ij

Loss:
    L = L_ARF + 0.1 L_quant + 0.05 L_balance
```

这样可以分两步验证：

```text
Step 1: T-SAS + 原训练是否稳定
Step 2: T-SAS + ARF 是否优于原 contrastive training
```

如果 Step 2 成功，再加入：

```text
P_z memory representation
full PER-SAS naming
late quantization sharpening
完整消融实验
```

---

## 27. 最终检查清单

实现前检查：

```text
[ ] PER-SAS 是否完全不反传梯度
[ ] 两个增强视图是否共享 key_idx
[ ] fast branch 是否使用全部帧
[ ] selector 是否不使用 hash loss / reconstruction loss
[ ] memory bank 是否正确 detach
[ ] ARF 中 memory h 是否 stop-gradient
[ ] P_ij 是否在 [0, 1]
[ ] feedback weights 是否 clip
[ ] random anchors 是否加入训练集合
[ ] quant / balance 是否正常计算
[ ] warm-up 阶段是否关闭 P_z 和 feedback
[ ] 后期是否提高 lambda_quant
```

训练监控指标：

```text
[ ] L_ARF
[ ] L_quant
[ ] L_balance
[ ] mean bit value
[ ] bit variance
[ ] planned/actual overlap: |N_i ∩ A_i| / |N_i|
[ ] false retrieval ratio: |A_i \ N_i| / |A_i|
[ ] average P_ij of retrieved samples
[ ] average Hamming distance distribution
[ ] validation mAP / Precision@K
```

---

## 28. 推荐命名

模型整体可命名为：

```text
KAMCH: Keyframe-aware Agentic Memory Content-time Hashing
```

或保持原名：

```text
KAMCH: Keyframe-guided Agentic Memory Content-time Hashing
```

关键帧选择模块：

```text
PER-SAS: Plan-Evaluate-Refine Semantic Anchor Selection
```

训练目标：

```text
ARF: Agentic Retrieval Feedback Loss
```

完整方法名可以写作：

```text
KAMCH with PER-SAS and ARF training
```

论文中建议避免过度强调“agent 模型”，而强调：

```text
agent-inspired retrieval feedback mechanism
```

---

## 29. 一句话总结

最终方案可以概括为：

```text
KAMCH 使用 training-free 的时间分层语义锚点选择器为慢分支提供高质量内容帧，使用 bidirectional Mamba 快分支建模全帧时序动态，并通过 content-time lateral fusion 生成视频级表示；训练时不再以视图对比为主，而是将哈希码视为检索动作，通过 memory bank 构成的检索环境返回实际检索轨迹，再由 planner-evaluator 反馈机制构造 ARF loss，直接优化哈希空间的检索结构。
```
