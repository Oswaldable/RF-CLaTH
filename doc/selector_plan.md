# 自监督视频哈希中的语义关键帧选择方案设计

## 0. 设计前提

你的整体方法是**无监督/自监督视频哈希**，关键帧选择只是进入语义分支前的轻量预处理模块，因此这里不建议把关键帧选择做成可学习模块，也不建议引入哈希损失反馈。原因是 AutoSSVH 已经明确提出了自动 hard-frame sampling，并将 adversarial frame sampling 与 hash-based contrastive learning 结合起来，用于选择信息更丰富、重建更困难的帧。你的方法如果再用哈希反馈指导采样，容易和它的贡献叙事重叠。([arXiv][1])

本文档建议保留一个核心原则：

> **语义分支选语义稳定、代表性强、覆盖均衡的帧；时序分支保留剩余帧，用于建模动作、顺序和动态变化。**

设输入为：

$$
F=\{f_1,f_2,\dots,f_T\}, \quad f_i\in \mathbb R^{2048}
$$

其中：

$$
T=30,\quad K=\frac{T}{5}=6
$$

选出的语义关键帧为：

$$
\mathcal K=\{k_1,k_2,\dots,k_6\}
$$

送入语义分支，剩余帧：

$$
\mathcal R=\{1,\dots,T\}\setminus\mathcal K
$$

送入时序分支。

---

# 1. 方案一：Temporal-stratified Semantic Anchor Selection

## 1.1 方法名称

**Temporal-stratified Semantic Anchor Selection**

中文可写为：

**时间分层语义锚点选择**

这是最稳妥、最干净的版本，适合作为你论文里的轻量关键帧选择模块。

---

## 1.2 核心思想

将 30 帧划分成 6 个连续时间片段，每个片段选 1 个语义锚点。这样既保留了 uniform sampling 的时间覆盖优势，又避免 uniform sampling 完全忽略帧内容的问题。

划分为：

$$
\mathcal T_1=\{1,2,3,4,5\}
$$

$$
\mathcal T_2=\{6,7,8,9,10\}
$$

$$
\dots
$$

$$
\mathcal T_6=\{26,27,28,29,30\}
$$

约束为：

$$
|\mathcal K\cap \mathcal T_b|=1,\quad b=1,2,\dots,6
$$

也就是说，每 5 帧中选 1 帧进入语义分支，其余 4 帧仍保留给时序分支。

---

## 1.3 特征归一化与相似度矩阵

先对帧特征做 L2 normalization：

$$
x_i=\frac{f_i}{|f_i|_2}
$$

计算帧间相似度：

$$
A_{ij}=x_i^\top x_j
$$

为了避免负相似度干扰，也可以使用非负相似度：

$$
A_{ij}=\max(0,x_i^\top x_j)
$$

推荐使用：

$$
A_{ij}=\max(0,x_i^\top x_j)
$$

---

## 1.4 单帧语义质量分数

### 1.4.1 全局代表性

$$
R_i^g=
\frac{1}{T}
\sum_{j=1}^{T}A_{ij}
$$

表示第 (i) 帧是否接近整个视频的全局语义中心。

---

### 1.4.2 局部代表性

如果第 (i) 帧属于时间片段 (\mathcal T_{b(i)})，则：

$$
R_i^l=
\frac{1}{|\mathcal T_{b(i)}|}
\sum_{j\in\mathcal T_{b(i)}}A_{ij}
$$

表示第 (i) 帧是否能代表自己所在的局部片段。

---

### 1.4.3 局部语义稳定性

语义分支不应过度偏向剧烈变化帧，因为变化帧更适合留给时序分支。因此这里使用局部稳定性，而不是局部变化性：

$$
U_i=
\begin{cases}
A_{1,2}, & i=1\\
\frac{A_{i-1,i}+A_{i,i+1}}{2}, & 1<i<T\\
A_{T-1,T}, & i=T
\end{cases}
$$

(U_i) 越大，说明该帧在局部邻域中越稳定，更适合作为语义锚点。

---

### 1.4.4 综合质量分数

对 (R_i^g)、(R_i^l)、(U_i) 做视频内部 min-max normalization：

$$
\hat R_i^g,\quad \hat R_i^l,\quad \hat U_i
$$

然后定义：

$$
q_i=
\alpha \hat R_i^g
+
\beta \hat R_i^l
+
\gamma \hat U_i
$$

推荐初始权重：

$$
\alpha=0.4,\quad \beta=0.5,\quad \gamma=0.1
$$

即：

$$
q_i=
0.4\hat R_i^g
+
0.5\hat R_i^l
+
0.1\hat U_i
$$

这里局部代表性权重略高，因为你的语义分支希望从每个时间片段中取一个稳定语义锚点。

---

## 1.5 集合级目标函数

只看单帧质量会导致关键帧之间可能冗余，因此最终用集合级目标选择关键帧集合。

目标函数为：

$$
J(\mathcal K)
=
\lambda_1\operatorname{Cov}(\mathcal K)
+
\lambda_2\operatorname{Qua}(\mathcal K)
-
\lambda_3\operatorname{Red}(\mathcal K)
$$

其中：

### 语义覆盖性

$$
\operatorname{Cov}(\mathcal K)
=
\frac{1}{T}
\sum_{j=1}^{T}
\max_{k\in\mathcal K}A_{jk}
$$

表示所选语义关键帧对原始视频所有帧的覆盖程度。

---

### 锚点质量

$$
\operatorname{Qua}(\mathcal K)
=
\frac{1}{K}
\sum_{k\in\mathcal K}q_k
$$

表示所选帧自身的语义质量。

---

### 关键帧冗余

$$
\operatorname{Red}(\mathcal K)
=
\frac{2}{K(K-1)}
\sum_{p<q,p,q\in\mathcal K}A_{pq}
$$

表示所选关键帧之间的平均相似度。该值越高，说明关键帧越重复。

---

推荐权重：

$$
\lambda_1=0.6,\quad
\lambda_2=0.3,\quad
\lambda_3=0.1
$$

即：

$$
J(\mathcal K)
=
0.6\operatorname{Cov}(\mathcal K)
+
0.3\operatorname{Qua}(\mathcal K)
-
0.1\operatorname{Red}(\mathcal K)
$$

---

## 1.6 搜索方式

由于每个时间片段有 5 帧，共 6 个时间片段，所以总组合数为：

$$
5^6=15625
$$

可以直接枚举所有组合：

$$
\mathcal K^*
=
\arg\max_{\substack{
i_1\in\mathcal T_1,\dots,i_6\in\mathcal T_6
}}
J(\{i_1,i_2,\dots,i_6\})
$$

不需要贪心，不需要训练，不需要强化学习，也不需要 Gumbel-Softmax。

---

## 1.7 算法伪代码

```python
Input:
    F: frame features with shape [T, 2048]
    T = 30
    K = 6

Output:
    K_set: selected semantic keyframe indices
    R_set: remaining temporal frame indices

Step 1:
    L2-normalize frame features:
        x_i = f_i / ||f_i||

Step 2:
    Compute similarity matrix:
        A_ij = max(0, x_i^T x_j)

Step 3:
    Divide video into K temporal segments:
        T_b = {5(b-1)+1, ..., 5b}

Step 4:
    Compute per-frame scores:
        global representativeness R_g
        local representativeness R_l
        local semantic stability U

Step 5:
    Normalize R_g, R_l, U within the video

Step 6:
    Compute semantic anchor quality:
        q_i = 0.4 * R_g_i + 0.5 * R_l_i + 0.1 * U_i

Step 7:
    Enumerate all combinations:
        one frame from each temporal segment

Step 8:
    For each candidate set K:
        compute Cov(K)
        compute Qua(K)
        compute Red(K)
        compute J(K)

Step 9:
    Select:
        K_set = argmax J(K)

Step 10:
    R_set = all_indices - K_set
```

---

## 1.8 优点

* 完全 training-free。
* 不引入哈希反馈，避免和 AutoSSVH 撞核心叙事。
* 保证每 5 帧选 1 帧，时间覆盖稳定。
* 比 uniform sampling 更有语义自适应性。
* 比普通 Top-K 更不容易集中在某一段。
* 复杂度极低，30 帧输入下可以直接精确搜索。
* 很适合作为双分支结构中的轻量语义分支前处理模块。

---

## 1.9 缺点

* 创新性中等。
* 对 (2048) 维特征质量有依赖。
* 不建模更复杂的长期事件结构。
* 如果视频中某个 5 帧片段全是无意义帧，也仍然必须从中选 1 帧。

---

## 1.10 推荐定位

这个方案适合作为：

* 论文主方法中的轻量关键帧选择模块；
* semantic branch 的输入分配策略；
* uniform sampling 的语义增强版本；
* 消融实验中的稳定版本。

---

# 2. 方案二：Agent-inspired Plan-Evaluate-Refine Semantic Anchor Selection

## 2.1 方法名称

**Agent-inspired Plan-Evaluate-Refine Semantic Anchor Selection**

可缩写为：

**PER-SAS**

中文可写为：

**智能体启发的计划—评估—反思式语义锚点选择**

这是我最推荐你在论文中采用的版本。

它本质上仍然是 training-free 的关键帧选择，但借鉴了近期 agent 架构中的：

$$
\text{Planning}
\rightarrow
\text{Acting}
\rightarrow
\text{Evaluation}
\rightarrow
\text{Reflection}
$$

思想。

ReAct 强调推理与行动交替进行，Reflexion 强调通过反馈和记忆改进后续决策而非更新模型权重，Tree-of-Thoughts 强调探索多个候选路径并自评选择，Anthropic 的 evaluator-optimizer workflow 也强调生成—评价—反馈式工作流。这里可以借鉴这些结构思想，但不引入 LLM、不引入 RL、不引入训练参数。([arXiv][2])

---

## 2.2 核心思想

相比方案一直接枚举所有组合，方案二把关键帧选择组织成一个轻量的 agent-inspired workflow：

$$
F
\rightarrow
\text{Planner}
\rightarrow
\text{Candidate Plan Tree}
\rightarrow
\text{Evaluator}
\rightarrow
\text{Reflector}
\rightarrow
\mathcal K
$$

其中：

* **Planner**：把 30 帧分成 6 个时间片段，并为每段生成候选锚点。
* **Candidate Plan Tree**：组合不同时间片段的候选锚点，形成多个候选关键帧计划。
* **Evaluator**：从覆盖性、质量、冗余三个角度评价完整关键帧集合。
* **Reflector**：如果发现冗余过高或覆盖不足，在同一时间段内做局部替换。

注意，这里所有模块都是确定性规则，不是可学习 agent。

---

## 2.3 Planner：时间分层候选规划

划分时间段：

$$
\mathcal T_b=\{5(b-1)+1,\dots,5b\},\quad b=1,\dots,6
$$

每段生成候选集：

$$
\mathcal C_b=\operatorname{TopM}_{i\in\mathcal T_b}(q_i)
$$

其中 (q_i) 仍然使用方案一中的语义锚点质量分数：

$$
q_i=
0.4\hat R_i^g
+
0.5\hat R_i^l
+
0.1\hat U_i
$$

如果每段只有 5 帧，建议直接取：

$$
M=5
$$

即：

$$
\mathcal C_b=\mathcal T_b
$$

如果以后输入帧数变多，比如 (T=60)，每段 10 帧，则可以取：

$$
M=4 \text{ or } 5
$$

降低搜索空间。

---

## 2.4 Candidate Plan Tree：候选计划树

每个完整候选计划为：

$$
\mathcal K=\{i_1,i_2,\dots,i_6\}
$$

其中：

$$
i_b\in\mathcal C_b
$$

这个结构类似 Tree-of-Thoughts 中的多路径探索，只不过这里的 “thought path” 被替换成了 “keyframe candidate path”。Tree-of-Thoughts 的核心是探索多个中间路径并自评选择，你这里对应的是探索多个候选关键帧集合并用集合级指标评价。([arXiv][3])

如果 (M=5)，搜索空间为：

$$
M^K=5^6=15625
$$

仍然非常小，可以直接遍历。

---

## 2.5 Evaluator：集合级评价器

Evaluator 使用与方案一相同的集合级目标：

$$
J(\mathcal K)
=
0.6\operatorname{Cov}(\mathcal K)
+
0.3\operatorname{Qua}(\mathcal K)
-
0.1\operatorname{Red}(\mathcal K)
$$

其中：

$$
\operatorname{Cov}(\mathcal K)
=
\frac{1}{T}
\sum_{j=1}^{T}
\max_{k\in\mathcal K}A_{jk}
$$

$$
\operatorname{Qua}(\mathcal K)
=
\frac{1}{K}
\sum_{k\in\mathcal K}q_k
$$

$$
\operatorname{Red}(\mathcal K)
=
\frac{2}{K(K-1)}
\sum_{p<q,p,q\in\mathcal K}A_{pq}
$$

选择初始计划：

$$
\mathcal K^0
=
\arg\max_{\mathcal K}
J(\mathcal K)
$$

---

## 2.6 Reflector：反思式局部替换

如果方案二已经枚举了所有候选组合，那么 Reflector 理论上不是必须的，因为 (\mathcal K^0) 已经是候选空间内的最优解。

但为了引入 agent-inspired 的 “reflection” 思想，可以将 Reflector 设计为一个**轻量可选模块**，只在以下情况下触发：

### 触发条件 1：冗余过高

$$
\operatorname{Red}(\mathcal K^0)>\tau_r
$$

例如：

$$
\tau_r=0.85
$$

---

### 触发条件 2：覆盖不足

$$
\operatorname{Cov}(\mathcal K^0)<\tau_c
$$

例如：

$$
\tau_c=0.65
$$

---

### 触发条件 3：某个片段锚点质量过低

若某一段选择的关键帧 (k_b) 满足：

$$
q_{k_b}<\tau_q
$$

则考虑替换。

---

## 2.7 局部替换策略

对第 (b) 个时间段，当前选中帧为 (k_b)，尝试用同段内其他候选帧替换：

$$
\mathcal K'
=
\mathcal K^r\setminus\{k_b\}\cup\{i\}
$$

其中：

$$
i\in\mathcal T_b,\quad i\neq k_b
$$

计算增益：

$$
\Delta J_{b,i}
=
J(\mathcal K')-J(\mathcal K^r)
$$

选择最大正增益：

$$
(b^*,i^*)
=
\arg\max_{b,i}\Delta J_{b,i}
$$

如果：

$$
\Delta J_{b^*,i^*}>0
$$

则更新：

$$
\mathcal K^{r+1}
=
\mathcal K^r\setminus\{k_{b^*}\}\cup\{i^*\}
$$

否则停止。

最多执行：

$$
R=1 \text{ or } 2
$$

轮。

---

## 2.8 算法伪代码

```python
Input:
    F: frame features with shape [30, 2048]
    K = 6
    M = 5
    max_refine_round = 1 or 2

Output:
    K_set: selected semantic keyframes
    R_set: remaining temporal frames

Step 1:
    Normalize F and compute similarity matrix A

Step 2:
    Divide frames into 6 temporal segments:
        T_1, ..., T_6

Step 3:
    Compute semantic anchor quality q_i:
        q_i = 0.4 * R_g_i + 0.5 * R_l_i + 0.1 * U_i

Step 4:
    Planner:
        For each temporal segment T_b:
            C_b = TopM frames according to q_i
        If each segment contains 5 frames:
            C_b = T_b

Step 5:
    Candidate Plan Tree:
        Enumerate all candidate sets:
            K = {i_1, ..., i_6}, where i_b in C_b

Step 6:
    Evaluator:
        For each candidate set K:
            compute Cov(K)
            compute Qua(K)
            compute Red(K)
            compute J(K)

Step 7:
    Select initial best set:
        K_0 = argmax J(K)

Step 8:
    Reflector:
        For r in range(max_refine_round):
            Try replacing one selected frame within its own temporal segment
            Compute delta J for all possible replacements
            If max delta J > 0:
                accept the replacement
            Else:
                break

Step 9:
    Return:
        K_set = final K
        R_set = all_indices - K_set
```

---

## 2.9 论文中的推荐表述

可以写成：

> Inspired by recent agentic workflows, we formulate semantic anchor selection as a lightweight plan-evaluate-refine process rather than a trainable sampling policy. A temporal planner first decomposes the input sequence into several non-overlapping segments and constructs a small candidate-plan tree. A set-level evaluator then assesses each candidate plan according to semantic coverage, anchor quality, and inter-anchor redundancy. Finally, a deterministic reflection step performs local replacement when the selected anchors exhibit excessive redundancy or insufficient coverage. The entire procedure is conducted before the semantic branch and introduces no trainable parameters or hash-loss feedback.

中文对应为：

> 受近期智能体工作流中计划、评估和反思机制的启发，我们将语义锚点选择建模为一个轻量的计划—评估—反思过程，而不是可学习采样策略。时间规划器首先将输入帧序列划分为多个不重叠片段，并构建小规模候选计划树；集合级评估器随后从语义覆盖性、锚点质量和锚点间冗余三个角度评价每个候选计划；最后，确定性的反思步骤在锚点冗余过高或覆盖不足时执行局部替换。整个过程在语义分支之前完成，不引入可训练参数，也不使用哈希损失反馈。

---

## 2.10 优点

* 比方案一更有方法结构感。
* 能自然融入 agent 架构中的 planning、evaluation、reflection 思想。
* 不需要 LLM，不需要 RL，不需要训练 selector。
* 和 AutoSSVH 的 hard-frame sampling / hash-feedback sampling 区分明显。
* 仍然保持轻量、可解释、可复现。
* 适合作为论文中的关键帧选择模块命名方法。

---

## 2.11 缺点

* Reflector 如果在全枚举之后再做，实际提升可能有限。
* agent-inspired 包装要克制，不能写成真正的 autonomous agent。
* 需要在论文中明确说明：这是 agent-inspired deterministic workflow，而不是可学习 agent。

---

## 2.12 推荐定位

这是我最推荐的最终版本。

建议你在论文里采用：

$$
\boxed{
\textbf{PER-SAS: Plan-Evaluate-Refine Semantic Anchor Selection}
}
$$

它比方案一更有新意，又不会把关键帧选择模块做得过重。

---

# 3. 方案三：Coverage-Memory Sequential Semantic Anchor Selection

## 3.1 方法名称

**Coverage-Memory Sequential Semantic Anchor Selection**

可缩写为：

**CM-SAS**

中文可写为：

**覆盖记忆引导的序列式语义锚点选择**

这个方案适合你想保留 “memory-guided agent” 味道，但又不想做完整候选计划树的时候使用。

---

## 3.2 核心思想

方案一和方案二都是集合级搜索。方案三则采用序列式选择：

$$
\mathcal K_0=\varnothing
$$

每一步选一个能带来最大新增语义覆盖的帧，同时考虑单帧语义质量和时间均衡。

它借鉴 agent 中的 memory 思想，但不使用均值 memory，而是使用 coverage memory。

Reflexion 类方法强调通过反馈和记忆改进后续决策而不更新模型权重；这里的 coverage memory 也是一种非参数记忆，用来记录当前已选关键帧集合对原视频的覆盖状态。([arXiv][4])

---

## 3.3 Coverage Memory

定义 coverage memory：

$$
c_j^{(t)}
=
\max_{k\in\mathcal K_t}A_{jk}
$$

其中 (c_j^{(t)}) 表示原始视频中第 (j) 帧已经被当前关键帧集合覆盖到什么程度。

初始化：

$$
c_j^{(0)}=0
$$

---

## 3.4 候选帧的新增覆盖收益

当考虑加入候选帧 (i) 时，它带来的新增覆盖为：

$$
\Delta_{\text{cov}}(i|\mathcal K_t)
=
\frac{1}{T}
\sum_{j=1}^{T}
\max(0,A_{ji}-c_j^{(t)})
$$

如果候选帧 (i) 和已有关键帧很相似，那么它对未覆盖区域的新增贡献会很小。

这比简单的：

$$
1-\cos(f_i,m_t)
$$

更稳，因为均值 memory 容易把多个语义模式平均掉，而 coverage memory 能保留“哪些帧已经被覆盖、哪些帧还没被覆盖”的细粒度信息。

---

## 3.5 时间均衡项

为了避免关键帧集中在某一段，引入时间均衡项。

设 (b(i)) 表示第 (i) 帧所属时间段，(n_{b(i)}^{(t)}) 表示当前已经在该时间段选择了多少帧：

$$
B_i^{(t)}
=
\frac{1}{1+n_{b(i)}^{(t)}}
$$

如果某个时间段还没选过帧，则：

$$
B_i^{(t)}=1
$$

如果已经选过 1 帧，则：

$$
B_i^{(t)}=\frac{1}{2}
$$

这可以作为软时间覆盖约束。

如果你希望严格每段选 1 帧，则可以在选择时限制：

$$
i_t\in \mathcal T_t
$$

或者限制每个时间段最多选 1 帧：

$$
n_b^{(t)}\leq 1
$$

对于你的 30 帧选 6 帧场景，推荐使用硬约束：

$$
|\mathcal K\cap \mathcal T_b|=1
$$

---

## 3.6 序列选择分数

候选帧 (i) 在第 (t) 步的分数为：

$$
S_i^{(t)}
=
\eta_1
\widehat{\Delta_{\text{cov}}(i|\mathcal K_{t-1})}
+
\eta_2 q_i
+
\eta_3 B_i^{(t)}
$$

推荐权重：

$$
\eta_1=0.6,\quad
\eta_2=0.3,\quad
\eta_3=0.1
$$

即：

$$
S_i^{(t)}
=
0.6
\widehat{\Delta_{\text{cov}}(i|\mathcal K_{t-1})}
+
0.3q_i
+
0.1B_i^{(t)}
$$

其中 (q_i) 仍然使用：

$$
q_i=
0.4\hat R_i^g
+
0.5\hat R_i^l
+
0.1\hat U_i
$$

---

## 3.7 更新规则

第 (t) 步选择：

$$
i_t^*
=
\arg\max_{i\notin\mathcal K_{t-1}}S_i^{(t)}
$$

更新关键帧集合：

$$
\mathcal K_t
=
\mathcal K_{t-1}\cup\{i_t^*\}
$$

更新 coverage memory：

$$
c_j^{(t)}
=
\max(c_j^{(t-1)},A_{j,i_t^*})
$$

直到：

$$
|\mathcal K_t|=K
$$

---

## 3.8 算法伪代码

```python
Input:
    F: frame features with shape [30, 2048]
    K = 6

Output:
    K_set: selected semantic keyframes
    R_set: remaining temporal frames

Step 1:
    Normalize frame features and compute A

Step 2:
    Divide frames into 6 temporal segments

Step 3:
    Compute semantic anchor quality q_i

Step 4:
    Initialize:
        K_set = empty set
        c_j = 0 for all frames
        n_b = 0 for all temporal segments

Step 5:
    For t in range(1, K + 1):

        For each candidate frame i not in K_set:

            If hard temporal constraint is used:
                skip i if its segment has already selected one frame

            Compute coverage gain:
                delta_cov_i = mean_j max(0, A[j, i] - c_j)

            Compute temporal balance:
                B_i = 1 / (1 + n_segment(i))

            Compute selection score:
                S_i = 0.6 * delta_cov_i + 0.3 * q_i + 0.1 * B_i

        Select:
            i_star = argmax S_i

        Update:
            K_set = K_set union {i_star}
            c_j = max(c_j, A[j, i_star])
            n_segment(i_star) += 1

Step 6:
    R_set = all_indices - K_set
```

---

## 3.9 优点

* 比方案一更像 memory-guided sequential decision。
* 不使用均值 memory，避免语义平均化问题。
* 复杂度低。
* 可以自然扩展到任意 (T)，不局限于 30 帧。
* 可以作为 agent-inspired 方案的简化版。
* 对长视频更友好，因为不需要枚举所有组合。

---

## 3.10 缺点

* 贪心选择可能不如方案一或方案二的全局枚举稳定。
* 对选择顺序敏感。
* 如果使用软时间均衡，可能仍然出现时间段覆盖不均的问题。
* 如果使用硬时间约束，则与方案一较接近。

---

## 3.11 推荐定位

这个方案适合作为：

* 长视频或可变帧数输入下的泛化版本；
* 方案二的高效近似版本；
* 消融实验中的 memory-guided variant；
* 论文附录或扩展实验中的补充方案。

如果你的输入始终是 30 帧，优先推荐方案二；如果未来要扩展到 60/90/120 帧，方案三更方便。

---

# 4. 三个方案对比

| 维度                 |   方案一：T-SAS |     方案二：PER-SAS |  方案三：CM-SAS |
| ------------------ | ----------: | --------------: | ----------: |
| 是否训练 selector      |           否 |               否 |           否 |
| 是否使用 hash feedback |           否 |               否 |           否 |
| 是否使用 agent 思想      |           弱 |               强 |           中 |
| 核心机制               | 时间分层 + 集合评价 |    计划 + 评估 + 反思 | 覆盖记忆 + 序列选择 |
| 搜索方式               |         全枚举 | 候选计划树 + 评价 + 反思 |      贪心序列选择 |
| 是否适合 30 帧输入        |         很适合 |             最适合 |          适合 |
| 是否适合长视频            |          一般 |              中等 |          较好 |
| 方法复杂度              |           低 |              中低 |          中低 |
| 论文方法感              |           中 |               高 |          中高 |
| 实验稳定性              |           高 |               高 |          中高 |
| 推荐程度               |           高 |              最高 |          中高 |

---

# 5. 最终推荐

如果你只想选一个最终方案，建议采用：

$$
\boxed{
\textbf{方案二：PER-SAS}
}
$$

即：

$$
\boxed{
\textbf{Agent-inspired Plan-Evaluate-Refine Semantic Anchor Selection}
}
$$

它的实际实现可以以方案一为核心，即时间分层、语义质量评分、集合级评价和精确搜索；在论文叙事上引入 agent-inspired 的 planner、candidate plan tree、evaluator、reflector 结构。

推荐最终流程为：

```text
Input 30 frame features
        ↓
L2 normalization
        ↓
Similarity matrix construction
        ↓
Temporal planner: split into 6 segments
        ↓
Semantic anchor scorer: compute q_i
        ↓
Candidate plan tree: one candidate from each segment
        ↓
Set-level evaluator: Cov + Qua - Red
        ↓
Optional reflector: local swap refinement
        ↓
Selected 6 semantic anchors → semantic branch
Remaining 24 frames → temporal branch
```

---

# 6. 建议的论文模块命名

可以选下面几个名字之一：

## 推荐名称 1

**PER-SAS: Plan-Evaluate-Refine Semantic Anchor Selection**

优点：突出 agent-inspired workflow。

---

## 推荐名称 2

**Agent-inspired Semantic Anchor Selection**

优点：简洁，容易理解。

---

## 推荐名称 3

**Temporal-stratified Plan-Evaluate Semantic Anchor Selection**

优点：更学术、更克制，不会显得强行蹭 agent。

---

我最建议使用：

$$
\boxed{
\textbf{PER-SAS: Plan-Evaluate-Refine Semantic Anchor Selection}
}
$$

但在正文里一定要强调：

> The proposed selection strategy is agent-inspired but not agent-based. It introduces no LLM, no reinforcement learning, no additional trainable parameters, and no hash-loss feedback.

中文就是：

> 本文的关键帧选择策略受到智能体工作流启发，但并不是一个真正的可学习智能体。它不引入 LLM、不使用强化学习、不增加可训练参数，也不使用哈希损失反馈。

---

# 7. 推荐消融实验

建议不要做太多关键帧选择对比，否则会喧宾夺主。可以做下面这些：

| 实验编号 | 方法               | 目的                               |
| ---- | ---------------- | -------------------------------- |
| A1   | Uniform Sampling | 基础时间覆盖 baseline                  |
| A2   | Random Sampling  | 随机 baseline                      |
| A3   | K-Medoids        | 传统语义代表性 baseline                 |
| A4   | Local Medoid     | 只用局部代表性                          |
| A5   | T-SAS            | 时间分层 + 集合评价                      |
| A6   | PER-SAS          | 加入 plan-evaluate-refine workflow |
| A7   | CM-SAS           | coverage memory 序列选择             |

核心消融可以写成：

$$
\text{Uniform}
\rightarrow
\text{Local Quality}
\rightarrow
\text{Local + Global Quality}
\rightarrow
\text{Coverage + Redundancy}
\rightarrow
\text{Plan-Evaluate-Refine}
$$

---

# 8. 推荐最终公式

最终建议你在论文主文中使用下面这个目标函数：

$$
\mathcal K^*
=
\arg\max_{\substack{
|\mathcal K\cap\mathcal T_b|=1
}}
\left[
0.6\operatorname{Cov}(\mathcal K)
+
0.3\operatorname{Qua}(\mathcal K)
-
0.1\operatorname{Red}(\mathcal K)
\right]
$$

其中：

$$
\operatorname{Cov}(\mathcal K)
=
\frac{1}{T}
\sum_{j=1}^{T}
\max_{k\in\mathcal K}A_{jk}
$$

$$
\operatorname{Qua}(\mathcal K)
=
\frac{1}{K}
\sum_{k\in\mathcal K}
\left(
0.4\hat R_k^g
+
0.5\hat R_k^l
+
0.1\hat U_k
\right)
$$

$$
\operatorname{Red}(\mathcal K)
=
\frac{2}{K(K-1)}
\sum_{p<q,p,q\in\mathcal K}A_{pq}
$$

这个公式足够轻量，也足够完整。它能说明你的 6 个语义关键帧不是随机选的，也不是简单 uniform，而是通过局部代表性、全局覆盖性、低冗余性和时间均衡共同确定的。

---

# 9. 最终写作建议

你的关键帧选择模块不要写成论文最大贡献，而应写成主框架中的一个合理设计：

> 为了让语义分支获得紧凑且有代表性的内容锚点，同时让时序分支保留充足的动态帧，我们设计了一个训练无关的语义锚点选择模块。该模块受到近期 agentic workflow 中计划、评估和反思机制的启发，但不引入额外可学习参数，也不使用哈希损失反馈。具体而言，我们将视频划分为多个时间片段，从每个片段中选择一个语义锚点，并通过集合级目标函数综合优化语义覆盖性、锚点质量和锚点间冗余。最终，选出的 1/5 帧输入语义分支，剩余 4/5 帧输入时序分支。

这段话基本可以直接改写进论文方法部分。

[1]: https://arxiv.org/abs/2504.03587?utm_source=chatgpt.com "AutoSSVH: Exploring Automated Frame Sampling for ..."
[2]: https://arxiv.org/abs/2210.03629?utm_source=chatgpt.com "ReAct: Synergizing Reasoning and Acting in Language Models"
[3]: https://arxiv.org/abs/2305.10601?utm_source=chatgpt.com "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
[4]: https://arxiv.org/abs/2303.11366?utm_source=chatgpt.com "Reflexion: Language Agents with Verbal Reinforcement Learning"
