# RF-CLaTH Agent Handoff Notes

更新时间：2026-06-08，时区 Asia/Shanghai。

本目录是 RF-CLaTH 的干净主方法目录：

```text
RF-CLaTH: Retrieval-Feedback Content-Lateral Temporal Hashing
```

后续不要把 KAMCH 旧实验日志、旧配置、prototype/center 分支、
reconstruction 分支、旧划分 cache 迁移进来。

## 来源上下文

RF-CLaTH 从原 KAMCH 项目清理迁移而来。原项目名和论文草稿口径：

```text
KAMCH: Keyframe-Anchored Masked Contrastive Hashing
中文名：关键帧锚定的掩码对比自监督视频哈希方法
local source: /Users/oswald/Desktop/learning/KAMCH
old remote project: /mnt/disk2/yql/KAMCH
old outputs: /mnt/disk2/yql/KAMCH_outputs
```

最终保留的实验原名：

```text
E25-2 w/o center/prototype UCF
旧 project.name: KAMCH-Center-Free-Neighbor-UCF-RePartition
新 project.name: RF-CLaTH-UCF-RePartition
```

重命名映射：

```text
configs/kamch_center_free_neighbor_ucf.yaml -> configs/rf_clath_ucf.yaml
tools/run_center_free_neighbor_ucf_disk2.sh -> tools/run_rf_clath_ucf_disk2.sh
KeyframeAnchoredMaskedContrastiveHashing -> RetrievalFeedbackContentLateralTemporalHashing
KAMCHLoss -> RFClathLoss
```

改名动机：最终方法的主要创新点不再强调原始 KAMCH 的
masked contrastive 叙事，而是强调 retrieval-feedback neighbor supervision、
content-lateral fusion 和 temporal hashing。方法仍保留关键帧选择、慢/快分支
和 content-time lateral fusion，但去掉了 center/prototype/reconstruction 相关
训练信号。

## 服务器情况

远端实验服务器：

```text
ssh alias: exp-server
remote project: /mnt/disk2/yql/RF-CLaTH
remote outputs: /mnt/disk2/yql/RF-CLaTH_outputs
remote run logs: /mnt/disk2/yql/RF-CLaTH_run_logs
dataset root: /mnt/disk2/yql/dataset_rePartition
```

常用 Python 环境：

```text
train env: /mnt/disk2/yql/miniconda3/envs/mamba_env/bin/python
label env: /mnt/disk2/yql/miniconda3/envs/s5vh/bin/python
```

同步本地代码到服务器：

```bash
cd /Users/oswald/Desktop/learning
rsync -av \
  --exclude "__pycache__/" --exclude "*.pyc" \
  --exclude "outputs/" --exclude "cache/" --exclude "checkpoints/" \
  --exclude "*.pth" --exclude "*.pt" \
  RF-CLaTH/ exp-server:/mnt/disk2/yql/RF-CLaTH/
```

监控远端训练：

```bash
ssh exp-server 'ps -Ao pid,ppid,stat,etime,args --cols 240 | grep -E "RF-CLaTH|train.py|compute_s5vh" | grep -v grep'
ssh exp-server 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'
ssh exp-server 'tail -n 80 /mnt/disk2/yql/RF-CLaTH_run_logs/<log_file>'
```

## 新划分数据集

后续只使用新划分数据：

```text
/mnt/disk2/yql/dataset_rePartition
```

样本规模按该目录 README：

```text
ActivityNet: train 8836, database 3399, query 1359, feature 30x2048
HMDB51:      train/database 3825, query 1275, feature 25x4096
UCF101:      train/database 9990, query 3330, feature 25x4096
FCVID:       train 45593, database 22796, query 22796, feature 25x4096
```

FCVID 使用独立 query/database：

```text
query:    fcv_query_feats.h5 / repartition_s5vh_fcv_q_label.pt
database: fcv_test_feats.h5  / repartition_s5vh_fcv_re_label.pt
```

RF-CLaTH 当前主方法只固定在 UCF101：

```text
dataset: s5vh_ucf
config: configs/rf_clath_ucf.yaml
train feature: /mnt/disk2/yql/dataset_rePartition/ucf/ucf_train_feats.h5
query feature: /mnt/disk2/yql/dataset_rePartition/ucf/ucf_test_feats.h5
database feature: /mnt/disk2/yql/dataset_rePartition/ucf/ucf_train_feats.h5
train label: cache/repartition_s5vh_ucf_train_label.pt
query label: cache/repartition_s5vh_ucf_q_label.pt
database label: cache/repartition_s5vh_ucf_re_label.pt
neighbor cache: cache/repartition_s5vh_ucf_train_rawmean_top20.pt
```

标签转换命令：

```bash
cd /mnt/disk2/yql/RF-CLaTH
/mnt/disk2/yql/miniconda3/envs/s5vh/bin/python \
  tools/prepare_repartition_labels.py \
  --source-root /mnt/disk2/yql/dataset_rePartition \
  --output-dir cache
```

## 当前主方法

```text
selector: segment_rerank_gumbel_topk
slow branch: selected_class_attention
fast branch: bidirectional_mamba mean-pooling
fusion: content_time_lateral
loss:
  0.3 * L_view
  0.5 * L_batch_neighbor
  0.04 * L_memory_neighbor
  0.02 * L_quant
  0.03 * L_balance
```

明确不使用：

```text
hash center
prototype alignment
prototype cache
reconstruction head
```

## 训练与评估

运行 UCF 主方法：

```bash
ssh exp-server 'cd /mnt/disk2/yql/RF-CLaTH && BITS="16 32 64 128" tools/run_rf_clath_ucf_disk2.sh <gpu>'
```

复算指标：

```bash
ssh exp-server 'cd /mnt/disk2/yql/RF-CLaTH && /mnt/disk2/yql/miniconda3/envs/mamba_env/bin/python tools/compute_s5vh_official_map.py --config configs/rf_clath_ucf.yaml --dataset s5vh_ucf --checkpoint <checkpoint>'
```

评估口径：

```text
hash bits: 16, 32, 64, 128
train batch: UCF = 256
eval batch: 256
P/R@K: 5, 10, 20, 40, 60, 80, 100
mAP@K: 5, 20, 40, 60, 80, 100
binary format: {-1, +1}
AP@K = sum(precision_at_relevant_rank for ranks <= K) / K
```
