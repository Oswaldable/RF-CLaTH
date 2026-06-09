# RF-CLaTH Experiment Log

更新时间：2026-06-09 18:13，时区 Asia/Shanghai。

记录规则：

```text
所有训练、评估、smoke 和结果复算都在 exp-server 执行。
本地不执行测试或验证脚本。
后续实验只记录 16 / 32 / 64 bit，不再进行 128-bit 实验。
```

## 2026-06-08 Stage1 T-SAS + Original Loss Launch

### UCF101 16/32/64-bit

```text
dataset: s5vh_ucf
config: configs/rf_clath_ucf.yaml
method: RF-CLaTH + T-SAS selector + original RF-CLaTH loss
gpu: cuda1
bits: 16, 32, 64
execution: serial within one queue
remote output root: /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_ucf_disk2
launcher pid: 1135399
active train pid: 715494
remote launcher log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_t_sas_ucf_cuda1_launcher_20260608_141126.log
remote queue log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_t_sas_ucf_disk2_20260608_141126.queue.log
status: 16/32-bit completed; 64-bit running
latest eval: 64-bit epoch 25, mAP@5=0.8008, mAP@20=0.6330, mAP@100=0.3987
```

结果记录：

| Bits | Run Dir | Best Checkpoint | Selected Epoch | mAP@5 | mAP@20 | mAP@40 | mAP@60 | mAP@80 | mAP@100 | P@5 | P@20 | P@100 | R@5 | R@20 | R@100 | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_ucf_disk2/s5vh_ucf_16b_20260608_141136 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_ucf_disk2/s5vh_ucf_16b_20260608_141136/best.pth | 70 | 0.6782 | 0.5820 | 0.5159 | 0.4550 | 0.3980 | 0.3434 | 0.7214 | 0.6462 | 0.4143 | 0.0361 | 0.1288 | 0.4109 | completed; final epoch 150 mAP@100=0.3410 |
| 32 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_ucf_disk2/s5vh_ucf_32b_20260608_223349 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_ucf_disk2/s5vh_ucf_32b_20260608_223349/best.pth | 60 | 0.7579 | 0.6324 | 0.5593 | 0.5027 | 0.4458 | 0.3888 | 0.7869 | 0.6823 | 0.4573 | 0.0395 | 0.1363 | 0.4523 | completed; final epoch 150 mAP@100=0.3831 |
| 64 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_ucf_disk2/s5vh_ucf_64b_20260609_084849 | TBD | 25 | 0.8008 | 0.6330 | 0.5563 | 0.5027 | 0.4508 | 0.3987 | 0.8177 | 0.6757 | 0.4703 | 0.0410 | 0.1350 | 0.4637 | running; latest epoch 25/150, not a completed result |

### HMDB51 16/32/64-bit

```text
dataset: hmdb
config: configs/rf_clath_hmdb.yaml
method: RF-CLaTH + T-SAS selector + original RF-CLaTH loss
gpu: cuda3
bits: 16, 32, 64
execution: serial within one queue
remote output root: /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_hmdb_disk2
launcher pid: 1201491
active train pid: none
remote launcher log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_t_sas_hmdb_cuda3_launcher_20260608_142724.log
remote queue log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_t_sas_hmdb_disk2_20260608_142724.queue.log
status: completed 16/32/64-bit
latest eval: 64-bit epoch 150, mAP@5=0.4075, mAP@20=0.2918, mAP@100=0.1257
note: previous mistaken s5vh_hmdb queue was stopped at epoch 2 and is not used for results.
```

结果记录：

| Bits | Run Dir | Best Checkpoint | Selected Epoch | mAP@5 | mAP@20 | mAP@40 | mAP@60 | mAP@80 | mAP@100 | P@5 | P@20 | P@100 | R@5 | R@20 | R@100 | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_hmdb_disk2/hmdb_16b_20260608_142727 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_hmdb_disk2/hmdb_16b_20260608_142727/best.pth | 65 | 0.3125 | 0.2342 | 0.1826 | 0.1457 | 0.1183 | 0.0994 | 0.3702 | 0.3117 | 0.1719 | 0.0247 | 0.0831 | 0.2292 | completed; final epoch 150 mAP@100=0.0952 |
| 32 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_hmdb_disk2/hmdb_32b_20260608_182721 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_hmdb_disk2/hmdb_32b_20260608_182721/best.pth | 100 | 0.3678 | 0.2693 | 0.2084 | 0.1656 | 0.1352 | 0.1134 | 0.4238 | 0.3460 | 0.1893 | 0.0283 | 0.0923 | 0.2523 | completed; final epoch 150 mAP@100=0.1122 |
| 64 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_hmdb_disk2/hmdb_64b_20260608_212823 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_hmdb_disk2/hmdb_64b_20260608_212823/best.pth | 100 | 0.4118 | 0.2923 | 0.2289 | 0.1839 | 0.1512 | 0.1273 | 0.4569 | 0.3622 | 0.2063 | 0.0305 | 0.0966 | 0.2751 | completed; final epoch 150 mAP@100=0.1257 |

### HMDB51 Trainable Selector 16/32/64-bit

```text
dataset: hmdb
config: configs/rf_clath_hmdb_trainable.yaml
method: RF-CLaTH + trainable segment_rerank_gumbel_topk selector + original RF-CLaTH loss
gpu: cuda2
bits: 16, 32, 64
execution: serial within one queue
remote output root: /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_trainable_hmdb_disk2
launcher pid: stopped
active train pid: none
remote launcher log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_trainable_hmdb_cuda2_launcher_20260609_003938.log
remote queue log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_trainable_hmdb_disk2_20260609_003938.queue.log
status: 16/32-bit completed; 64-bit stopped at epoch 29 before completion
note: selector ablation. Current evidence does not justify using trainable selector as the main method.
```

结果记录：

| Bits | Run Dir | Best Checkpoint | Selected Epoch | mAP@5 | mAP@20 | mAP@40 | mAP@60 | mAP@80 | mAP@100 | P@5 | P@20 | P@100 | R@5 | R@20 | R@100 | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_trainable_hmdb_disk2/hmdb_16b_20260609_003941 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_trainable_hmdb_disk2/hmdb_16b_20260609_003941/best.pth | 145 | 0.3050 | 0.2357 | 0.1825 | 0.1437 | 0.1156 | 0.0971 | 0.3595 | 0.3129 | 0.1681 | 0.0240 | 0.0834 | 0.2241 | completed; final epoch 150 mAP@100=0.0969 |
| 32 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_trainable_hmdb_disk2/hmdb_32b_20260609_044742 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_trainable_hmdb_disk2/hmdb_32b_20260609_044742/best.pth | 145 | 0.3705 | 0.2774 | 0.2137 | 0.1685 | 0.1364 | 0.1141 | 0.4191 | 0.3507 | 0.1880 | 0.0279 | 0.0935 | 0.2507 | completed; final epoch 150 mAP@100=0.1139 |
| 64 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_trainable_hmdb_disk2/hmdb_64b_20260609_092340 | TBD | 25 | 0.3716 | 0.2570 | 0.1985 | 0.1599 | 0.1321 | 0.1118 | 0.4174 | 0.3329 | 0.1921 | 0.0278 | 0.0888 | 0.2561 | stopped before completion; latest/best eval at epoch 25, not a completed result |

### Completed Result Snapshot

Only fully completed runs are included in this snapshot.

| Dataset | Method | Bits | Best Epoch | Best mAP@100 | Final mAP@100 | Main Note |
|---|---|---:|---:|---:|---:|---|
| UCF101 | T-SAS + original loss | 16 | 70 | 0.3434 | 0.3410 | completed |
| UCF101 | T-SAS + original loss | 32 | 60 | 0.3888 | 0.3831 | completed |
| HMDB51 | T-SAS + original loss | 16 | 65 | 0.0994 | 0.0952 | completed |
| HMDB51 | T-SAS + original loss | 32 | 100 | 0.1134 | 0.1122 | completed |
| HMDB51 | T-SAS + original loss | 64 | 100 | 0.1273 | 0.1257 | completed |
| HMDB51 | trainable selector + original loss | 16 | 145 | 0.0971 | 0.0969 | selector ablation |
| HMDB51 | trainable selector + original loss | 32 | 145 | 0.1141 | 0.1139 | selector ablation |

## 2026-06-09 Stage2 Planner Graph Sanity

实现范围：

```text
scope: UCF101, HMDB51, ActivityNet, and FCVID
doc: doc/kamch_arf_push_plan.md, section 6
goal: build Planner Graph and log sanity metrics without enabling ARF
loss: unchanged original RFClathLoss
new modules:
  memory/memory_bank.py
  planner/retrieval_graph_planner.py
config enabled:
  configs/rf_clath_ucf.yaml
  configs/rf_clath_hmdb.yaml
  configs/rf_clath_activitynet.yaml
  configs/rf_clath_fcv.yaml
```

Planner 配置：

```yaml
planner_small_datasets:
  enabled: true
  top_m: 20
  omega_s: 0.65
  omega_t: 0.35
  omega_z: 0.0
  random_anchors: 40
  z_momentum: 0.9
  log_interval: 20
  include_z_metrics: true
  label_precision: true

planner_large_datasets:
  enabled: true
  top_m: 50
  omega_s: 0.65
  omega_t: 0.35
  omega_z: 0.0
  random_anchors: 50
  z_momentum: 0.9
  log_interval: 20
  include_z_metrics: true
```

远端验证：

```text
py_compile:
  engine/train.py
  memory/memory_bank.py
  planner/retrieval_graph_planner.py

short sanity command:
  train.max_steps_per_epoch=2
  train.eval_interval=999
  train.save_interval=999
  planner.log_interval=1
  neighbor.enabled=false for ActivityNet/FCVID short sanity only
```

Sanity 结果：

| Dataset | Run Dir | Steps | valid_final_avg | z_valid_avg | P_s topM | P_t topM | P_z topM | P final topM | P random | P final std | overlap(N_s,N_t) | overlap(N_final,N_s) | overlap(N_final,N_t) | label precision | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| UCF101 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_stage2_planner_sanity/s5vh_ucf_16b_20260609_102114 | 2 | 0.037 | 0.037 | 0.7154 | 0.8269 | 0.6982 | 0.7531 | 0.5680 | 0.0614 | 0.746 | 0.944 | 0.793 | 0.161 | passed; P final topM > P random |
| HMDB51 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_stage2_planner_sanity/hmdb_16b_20260609_102234 | 2 | 0.096 | 0.096 | 0.6803 | 0.7964 | 0.6549 | 0.7200 | 0.5540 | 0.0523 | 0.781 | 0.947 | 0.826 | 0.136 | passed; P final topM > P random |
| ActivityNet | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_stage2_planner_sanity/s5vh_activitynet_16b_20260609_102949 | 2 | 0.043 | 0.043 | 0.8026 | 0.8899 | 0.7388 | 0.8315 | 0.7618 | 0.0346 | 0.654 | 0.928 | 0.718 | 0.000 | passed; P final topM > P random; train labels unavailable so label precision disabled |
| FCVID | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_stage2_planner_sanity/s5vh_fcv_16b_20260609_103121 | 2 | 0.008 | 0.008 | 0.5318 | 0.6816 | 0.5552 | 0.5827 | 0.4321 | 0.0748 | 0.801 | 0.942 | 0.855 | 0.028 | passed; P final topM > P random |

解释：

```text
valid_final/z_valid 偏低是因为 sanity run 只跑了 2 个 step，memory bank 只覆盖了少量训练样本。
P final topM 明显高于 P random，说明 Planner Graph 在四个数据集上有基本筛选能力。
omega_z=0，所以 P_z 只作为日志诊断，不参与 P final。
ActivityNet train label cache 不存在，label precision 不作为该数据集的阶段二验收指标。
```
