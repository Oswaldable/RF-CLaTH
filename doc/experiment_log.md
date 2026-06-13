# RF-CLaTH Experiment Log

更新时间：2026-06-10，时区 Asia/Shanghai。

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
status: completed 16/32/64-bit
latest eval: 64-bit epoch 150, mAP@5=0.7902, mAP@20=0.6470, mAP@100=0.4072
```

结果记录：

| Bits | Run Dir | Best Checkpoint | Selected Epoch | mAP@5 | mAP@20 | mAP@40 | mAP@60 | mAP@80 | mAP@100 | P@5 | P@20 | P@100 | R@5 | R@20 | R@100 | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_ucf_disk2/s5vh_ucf_16b_20260608_141136 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_ucf_disk2/s5vh_ucf_16b_20260608_141136/best.pth | 70 | 0.6782 | 0.5820 | 0.5159 | 0.4550 | 0.3980 | 0.3434 | 0.7214 | 0.6462 | 0.4143 | 0.0361 | 0.1288 | 0.4109 | completed; final epoch 150 mAP@100=0.3410 |
| 32 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_ucf_disk2/s5vh_ucf_32b_20260608_223349 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_ucf_disk2/s5vh_ucf_32b_20260608_223349/best.pth | 60 | 0.7579 | 0.6324 | 0.5593 | 0.5027 | 0.4458 | 0.3888 | 0.7869 | 0.6823 | 0.4573 | 0.0395 | 0.1363 | 0.4523 | completed; final epoch 150 mAP@100=0.3831 |
| 64 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_ucf_disk2/s5vh_ucf_64b_20260609_084849 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_ucf_disk2/s5vh_ucf_64b_20260609_084849/best.pth | 85 | 0.7960 | 0.6510 | 0.5713 | 0.5158 | 0.4632 | 0.4075 | 0.8154 | 0.6945 | 0.4788 | 0.0409 | 0.1389 | 0.4737 | completed; final epoch 150 mAP@100=0.4072 |

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
| UCF101 | T-SAS + original loss | 64 | 85 | 0.4075 | 0.4072 | completed |
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

## 2026-06-09 Stage3 Static ARF Launch

实现范围：

```text
doc: doc/kamch_arf_push_plan.md, section 7
objective: static_arf
datasets: UCF101 and HMDB51
loss:
  L = L_ARF-static + 0.10 L_quant + 0.05 L_balance
disabled:
  view contrast
  batch neighbor contrast
  memory neighbor contrast
not enabled:
  actual retrieval trace A_i
  missed/false feedback
  P_z in final graph
new modules:
  losses/arf_loss.py
updated:
  memory/memory_bank.py now tracks u_bank/u_valid
  planner/retrieval_graph_planner.py now builds Static ARF targets S_i=N_i union R_i
```

配置：

```text
UCF:
  config: configs/rf_clath_ucf_static_arf.yaml
  dataset: s5vh_ucf
  gpu: cuda0
  bits: 16, 32, 64
  output root: /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_static_arf_ucf_disk2
  launcher pid: 1456162
  queue log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_static_arf_ucf_disk2_20260609_104702.queue.log
  active first run: /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_static_arf_ucf_disk2/s5vh_ucf_16b_20260609_104704

HMDB:
  config: configs/rf_clath_hmdb_static_arf.yaml
  dataset: hmdb
  gpu: cuda2
  bits: 16, 32, 64
  output root: /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_static_arf_hmdb_disk2
  launcher pid: 1456319
  queue log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_static_arf_hmdb_disk2_20260609_104703.queue.log
  active first run: /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_static_arf_hmdb_disk2/hmdb_16b_20260609_104705
```

短程远端验证：

| Dataset | Steps | loss | L_ARF raw | L_hash | target count | target mean | P final topM | P random | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| UCF101 | 2 | 0.7057 | 0.6560 | 0.0498 | 59.9 | 0.630 | 0.7376 | 0.5762 | passed; old view/neighbor losses are 0 |
| HMDB51 | 2 | 0.7256 | 0.6759 | 0.0496 | 59.9 | 0.591 | 0.7018 | 0.5353 | passed; old view/neighbor losses are 0 |

正式队列启动后首个日志确认：

| Dataset | Bit | Epoch/Step | loss | L_ARF raw | L_hash | target mean | P final topM | P random | label precision | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| UCF101 | 16 | epoch 1 step 20/39 | 0.6628 | 0.6213 | 0.0415 | 0.658 | 0.8228 | 0.5737 | 0.519 | running |
| HMDB51 | 16 | epoch 1 step 14/14 | 0.6952 | 0.6510 | 0.0442 | 0.617 | 0.7688 | 0.5406 | 0.317 | running |

## 2026-06-09 Stage3 Static ARF Tuned Restart

触发原因：

```text
初版 Static ARF early metrics 明显偏低：
  UCF16 best observed mAP@100=0.1048 at epoch 15
  HMDB16 best observed mAP@100=0.0322 at epoch 30

主要问题不是 hash collapse：
  bit entropy/bit_use 正常
  planner P_final_topM > P_random

更可能的问题：
  Static ARF target 过软，random anchors 的 P_ij 也偏高；
  random anchors 没有形成足够排斥力；
  UCF16 soft saturation 下滑，quant 不够强。
```

停止旧队列：

```text
stopped:
  UCF Static ARF launcher pid 1456162
  UCF Static ARF trainer pid 1456173
  HMDB Static ARF launcher pid 1456319
  HMDB Static ARF trainer pid 1456384
```

参数调整：

| Parameter | Old | New | Reason |
|---|---:|---:|---|
| arf_loss.gamma | 8 | 6 | 降低早期 logits 压力，减少软目标拟合震荡 |
| loss_weights.lambda_quant | 0.10 | 0.20 | 抑制 UCF soft saturation 下滑 |
| loss_weights.lambda_balance | 0.05 | 0.10 | 更强 bit balance，防止新损失下 bit 偏移 |
| planner.top_m | 20 | 10 | 使用更高置信 planned neighbors |
| planner.random_anchors | 40 | 20 | 降低中等相似 random anchors 对 BCE 的牵引 |
| planner.u_momentum | 0.9 | 0.5 | 降低早期坏 soft code 在 u_bank 中的滞留 |

新队列：

```text
UCF:
  gpu: cuda0
  launcher pid: 1660464
  trainer pid: 1660474
  queue log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_static_arf_ucf_disk2_20260609_112131.queue.log
  active first run: /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_static_arf_ucf_disk2/s5vh_ucf_16b_20260609_112135

HMDB:
  gpu: cuda2
  launcher pid: 1660480
  trainer pid: 1660491
  queue log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_static_arf_hmdb_disk2_20260609_112131.queue.log
  active first run: /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_static_arf_hmdb_disk2/hmdb_16b_20260609_112135
```

已确认 tuned 参数生效：

| Dataset | Bit | Epoch | top_m | random_anchors | target count | target mean | P final topM | P random | label precision | sat | entropy | bit_use | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| UCF101 | 16 | 1 | 10 | 20 | 30.0 | 0.661 | 0.8568 | 0.5738 | 0.683 | 0.521 | 0.795 | 0.969 | running; first eval pending |
| HMDB51 | 16 | 4 | 10 | 20 | 30.0 | 0.625 | 0.7959 | 0.5424 | 0.429 | 0.612 | 0.930 | 1.000 | running; first eval pending |

备注：

```text
2026-06-09 11:25 左右远端 SSH/rsync 多次在握手阶段 reset。
训练进程已确认启动，epoch5 eval 暂未成功拉取。
后续需要继续同步 train.log，重点看 tuned 后 mAP@100 是否超过初版 epoch5：
  UCF16 old epoch5 mAP@100=0.0821
  HMDB16 old epoch5 mAP@100=0.0271
```

后续远端检查：

| Dataset | Bit | Tuned Epoch | mAP@5 | mAP@20 | mAP@40 | mAP@60 | mAP@80 | mAP@100 | P@5 | P@20 | P@100 | Old Static ARF Reference | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| UCF101 | 16 | 5 | 0.2475 | 0.1532 | 0.1128 | 0.0920 | 0.0795 | 0.0709 | 0.3210 | 0.2467 | 0.1585 | old epoch5 mAP@100=0.0821 | worse |
| UCF101 | 16 | 10 | 0.2586 | 0.1595 | 0.1184 | 0.0990 | 0.0864 | 0.0767 | 0.3299 | 0.2525 | 0.1671 | old epoch10 mAP@100=0.0884 | worse |
| UCF101 | 16 | 15 | 0.2335 | 0.1442 | 0.1078 | 0.0890 | 0.0772 | 0.0677 | 0.3069 | 0.2401 | 0.1566 | old epoch15 mAP@100=0.1048 | worse |
| HMDB51 | 16 | 5 | 0.1079 | 0.0584 | 0.0403 | 0.0326 | 0.0279 | 0.0246 | 0.1562 | 0.1299 | 0.0924 | old epoch5 mAP@100=0.0271 | worse |
| HMDB51 | 16 | 10 | 0.1240 | 0.0665 | 0.0458 | 0.0365 | 0.0311 | 0.0274 | 0.1805 | 0.1431 | 0.0985 | old epoch10 mAP@100=0.0285 | slightly worse |
| HMDB51 | 16 | 15 | 0.1224 | 0.0676 | 0.0467 | 0.0378 | 0.0320 | 0.0279 | 0.1738 | 0.1415 | 0.0966 | old epoch15 mAP@100=0.0284 | slightly worse |
| HMDB51 | 16 | 20 | 0.1151 | 0.0591 | 0.0400 | 0.0318 | 0.0270 | 0.0235 | 0.1663 | 0.1308 | 0.0918 | old epoch20 mAP@100=0.0320 | worse |

最新训练健康指标：

| Dataset | Epoch | target count | target mean | P final topM | P random | label precision | quant | bit balance | sat | entropy | bit_use | Conclusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| UCF101 | 16 | 30.0 | 0.672 | 0.8719 | 0.5742 | 0.735 | 0.1195 | 0.0512 | 0.783 | 0.918 | 0.938 | hash healthier but retrieval worse |
| HMDB51 | 21 | 30.0 | 0.624 | 0.7948 | 0.5400 | 0.409 | 0.1662 | 0.0262 | 0.717 | 0.984 | 1.000 | hash healthy but retrieval worse |

结论：

```text
Tuned Static ARF 参数失败。
更强 quant/balance 和更窄 top_m 没有修复检索，反而让 UCF/HMDB 的 early mAP 更低。
这说明当前主要问题不是二值化强度，而是 Static ARF 的 target calibration / target semantics。
planner graph 自身仍有区分度，但 BCE 拟合 P_ij 不能有效转化为 Hamming ranking。
继续跑 tuned 三 bits 的价值较低。
```

## 2026-06-09 Stage3 Static ARF LR1e-4 16-bit Restart

目的：

```text
恢复旧 Static ARF 参数，只提高学习率，看 16-bit 最终是否能明显好于低学习率版本。
不进入阶段4，不加回 view/contrastive/neighbor。
```

恢复/调整参数：

| Parameter | Value |
|---|---:|
| arf_loss.gamma | 8 |
| loss_weights.lambda_quant | 0.10 |
| loss_weights.lambda_balance | 0.05 |
| planner.top_m | 20 |
| planner.random_anchors | 40 |
| planner.u_momentum | 0.9 |
| train.lr | 1e-4 |
| bits | 16 only |

停止 tuned 队列：

```text
stopped:
  UCF tuned launcher pid 1660464
  UCF tuned trainer pid 1660474
  HMDB tuned launcher pid 1660480
  HMDB tuned trainer pid 1660491
```

新队列：

```text
UCF:
  gpu: cuda0
  launcher pid: 1810249
  trainer pid: 1810274
  queue log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_static_arf_ucf_disk2_20260609_114307.queue.log
  run: /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_static_arf_ucf_disk2/s5vh_ucf_16b_20260609_114312

HMDB:
  gpu: cuda2
  launcher pid: 1810248
  trainer pid: 1810273
  queue log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_static_arf_hmdb_disk2_20260609_114307.queue.log
  run: /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_static_arf_hmdb_disk2/hmdb_16b_20260609_114312
```

验证：

```text
saved config confirmed:
  hash_bits=16
  lr=1e-4
  top_m=20
  random_anchors=40
  gamma=8
  lambda_quant=0.10
  lambda_balance=0.05

queue log confirmed:
  BITS=16 only
```

## 2026-06-09 Stage5 Full ARF Implementation

用户指令：

```text
停止当前两个 Static ARF LR1e-4 进程，直接实现阶段5。
```

已停止：

```text
UCF Static ARF LR1e-4:
  launcher pid 1810249
  trainer pid 1810274

HMDB Static ARF LR1e-4:
  launcher pid 1810248
  trainer pid 1810273
```

实现范围：

```text
objective: arf / full_arf / trace_arf
loss:
  L = L_ARF + lambda_quant L_quant + lambda_balance L_balance

enabled:
  actual retrieval trace A_i
  missed-neighbor feedback
  false-retrieval feedback
  P_z fused memory graph
  graph warmup schedule
  feedback ramp schedule
  late binarization sharpen

still disabled:
  view contrast
  batch neighbor contrast
  memory neighbor contrast
```

代码变更：

```text
planner/retrieval_graph_planner.py
  added arf_trace_targets(...)
  supports P = omega_s P_s + omega_t P_t + omega_z P_z with per-call weights
  builds S_i = N_i union A_i union R_i
  computes missed/false feedback weights and diagnostics

losses/arf_loss.py
  added ARFLoss
  supports warmup, actual trace, feedback ramp, late sharpen

engine/train.py
  objective=arf/full_arf/trace_arf now uses ARFLoss
  logs arf_overlap, arf_false, arf_missed, arf_retrieved, arf_weight, eta_m, eta_f, omega_z, arf_gamma

configs:
  configs/rf_clath_ucf_full_arf.yaml
  configs/rf_clath_hmdb_full_arf.yaml

scripts:
  tools/run_rf_clath_ucf_full_arf_disk2.sh
  tools/run_rf_clath_hmdb_full_arf_disk2.sh
```

阶段5默认配置：

| Component | Value |
|---|---|
| planner top_m | 20 |
| main omega_s / omega_t / omega_z | 0.45 / 0.25 / 0.30 |
| warmup epochs | 10 |
| warmup omega_s / omega_t / omega_z | 0.65 / 0.35 / 0.00 |
| retrieval top_r | 20 |
| random anchors | 40 |
| eta_missed_final | 1.0 |
| eta_false_final | 1.0 |
| feedback ramp epochs | 10 |
| weight clip | 3.0 |
| gamma | 8 |
| late sharpen start_ratio | 0.70 |
| late gamma | 10 |
| late lambda_quant | 0.20 |
| lambda_balance | 0.05 |

远端验证：

```text
py_compile passed:
  planner/retrieval_graph_planner.py
  losses/arf_loss.py
  losses/__init__.py
  engine/train.py
  train.py

bash -n passed:
  tools/run_rf_clath_ucf_full_arf_disk2.sh
  tools/run_rf_clath_hmdb_full_arf_disk2.sh
```

Full ARF sanity：

```text
command:
  HMDB16, cuda2, max_steps_per_epoch=2, run_until_epoch=1,
  planner.warmup.epochs=0, feedback.ramp_epochs=1

run:
  /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_full_arf_sanity/hmdb_16b_20260609_131205
```

Sanity 结果：

| Dataset | Bit | Steps | loss | L_ARF raw | L_hash | target count | target mean | actual overlap | false ratio | missed ratio | retrieved target | feedback weight | eta_m | eta_f | omega_z | gamma | P final topM | P random | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HMDB51 | 16 | 2 | 0.9948 | 0.8979 | 0.0969 | 79.9 | 0.567 | 0.165 | 0.835 | 0.729 | 0.564 | 1.265 | 1.0 | 1.0 | 0.30 | 10.0 | 0.6930 | 0.5121 | passed; actual trace and P_z active |

## 2026-06-09 Stage5 Full ARF 16-bit Official Launch

目的：按阶段5 Full ARF 方案正式验证 16-bit，不启用 32/64b，先看 UCF/HMDB 是否能走出 Static ARF 的低 mAP 问题。

当前学习率：

```text
configs/rf_clath_ucf_full_arf.yaml:  train.lr = 3e-5
configs/rf_clath_hmdb_full_arf.yaml: train.lr = 3e-5
```

说明：`1e-4` 只用于前一轮 Static ARF 诊断实验；Full ARF 正式启动回到主方法一致的 `3e-5`。

启动任务：

| Dataset | Bit | GPU | Status | Launcher PID | Train Dir | Launcher Log | Queue Log |
|---|---:|---:|---|---:|---|---|---|
| UCF101 | 16 | cuda0 | running | 2297060 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_full_arf_ucf_disk2/s5vh_ucf_16b_20260609_131518` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_full_arf_ucf16_cuda0_launcher_20260609_131514.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_full_arf_ucf_disk2_20260609_131514.queue.log` |
| HMDB51 | 16 | cuda2 | running | 2298870 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_full_arf_hmdb_disk2/hmdb_16b_20260609_131532` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_full_arf_hmdb16_cuda2_launcher_20260609_131528.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_full_arf_hmdb_disk2_20260609_131529.queue.log` |

待观察指标：

```text
hash / quant / balance:
  判断哈希层是否仍然正常学习

arf_overlap / arf_false / arf_missed / arf_retrieved / arf_weight:
  判断 actual retrieval trace 是否真的进入训练反馈

mAP@100:
  UCF16 先对比 Stage1 T-SAS best 0.3434
  HMDB16 先对比 Stage1 T-SAS best 0.0994
```

## 2026-06-09 Stage5 Hybrid ARF Replacing Memory Neighbor

目的：如果纯 Full ARF 作为主监督不稳定，验证更保守的 hybrid 方案：
保留 Stage1 的 view contrastive 和 batch neighbor contrastive，把旧的 memory neighbor contrastive 替换为 Full ARF trace-fitting。

当前损失：

```text
L_total =
  0.30 * L_view
+ 0.50 * L_batch_neighbor
+ 0.04 * L_ARF
+ 0.02 * L_quant
+ 0.03 * L_balance

L_memory_neighbor = 0
late_sharpen disabled for isolation
```

代码/配置：

```text
losses/arf_loss.py
  HybridARFLoss

engine/train.py
  objective=hybrid_arf / arf_hybrid / contrastive_arf

configs/rf_clath_hmdb_hybrid_arf.yaml
tools/run_rf_clath_hmdb_hybrid_arf_disk2.sh
```

启动任务：

| Dataset | Bit | GPU | Status | Launcher PID | Train PID | Train Dir | Launcher Log | Queue Log |
|---|---:|---:|---|---:|---:|---|---|---|
| HMDB51 | 16 | cuda3 | running | 2418847 | 2418868 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_hybrid_arf_hmdb_disk2/hmdb_16b_20260609_133200` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_hybrid_arf_hmdb16_cuda3_launcher_20260609_133157.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_hybrid_arf_hmdb_disk2_20260609_133157.queue.log` |

启动验证：

```text
remote py_compile passed
bash -n passed
saved config objective=hybrid_arf, hash_bits=16
cuda3 was already shared with AutoSSVH FCV64; memory after launch about 15.3GB / 32GB
```

对比目标：

```text
HMDB16 Stage1 T-SAS best mAP@100: 0.0994
HMDB16 Full ARF official launch: running separately on cuda2
```

## 2026-06-09 Stage5 Full ARF Stop

停止原因：纯 Full ARF 当前明显低于 Stage1，并且 HMDB Hybrid ARF 已经超过纯 Full ARF，同步释放 cuda0/cuda2。

停止口径：

```text
TERM matched:
  tools/run_rf_clath_ucf_full_arf_disk2.sh
  configs/rf_clath_ucf_full_arf.yaml
  tools/run_rf_clath_hmdb_full_arf_disk2.sh
  configs/rf_clath_hmdb_full_arf.yaml
```

停止前最好指标：

| Dataset | Bit | Experiment | Best Epoch | Best mAP@100 | Status |
|---|---:|---|---:|---:|---|
| UCF101 | 16 | Full ARF | 20 | 0.1390 | stopped |
| HMDB51 | 16 | Full ARF | 45 | 0.0394 | stopped |

保留运行：

| Dataset | Bit | Experiment | GPU | Train Dir | Status |
|---|---:|---|---:|---|---|
| HMDB51 | 16 | Hybrid ARF replacing memory neighbor | cuda3 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_hybrid_arf_hmdb_disk2/hmdb_16b_20260609_133200` | completed |

## 2026-06-09 Stage5 Hybrid ARF v2 and ARF-only v3 Launch

目的：继续拆分 ARF 的有效性：

```text
v2:
  降低 batch neighbor，增强 ARF

v3:
  去掉 view / batch neighbor / memory neighbor / quant / balance
  只保留 L_ARF
```

代码/脚本：

```text
tools/run_rf_clath_hmdb_hybrid_arf_v2_disk2.sh
  base config: configs/rf_clath_hmdb_hybrid_arf.yaml
  overrides:
    loss.semantic.lambda_batch_neighbor=0.3
    loss_weights.lambda_arf=0.08
    loss.arf.lambda=0.08

tools/run_rf_clath_hmdb_arf_only_v3_disk2.sh
  base config: configs/rf_clath_hmdb_full_arf.yaml
  overrides:
    loss_weights.lambda_quant=0.0
    loss_weights.lambda_balance=0.0
    arf_loss.late_sharpen.start_ratio=1.10
    arf_loss.late_sharpen.lambda_quant=0.0
    arf_loss.late_sharpen.lambda_balance=0.0
```

实际损失：

```text
v2:
  L_total =
    0.30 * L_view
  + 0.30 * L_batch_neighbor
  + 0.08 * L_ARF
  + 0.02 * L_quant
  + 0.03 * L_balance

v3:
  L_total = 1.00 * L_ARF
```

启动任务：

| Dataset | Bit | Experiment | GPU | Status | Launcher PID | Train PID | Train Dir | Launcher Log | Queue Log |
|---|---:|---|---:|---|---:|---:|---|---|---|
| HMDB51 | 16 | Hybrid ARF v2 | cuda0 | completed | 2592850 | 2592858 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_hybrid_arf_v2_hmdb_disk2/hmdb_16b_20260609_140921` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_hybrid_arf_v2_hmdb16_cuda0_launcher_20260609_140910.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_hybrid_arf_v2_hmdb_disk2_20260609_140910.queue.log` |
| HMDB51 | 16 | ARF-only v3 | cuda2 | completed | 2594100 | 2594108 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_arf_only_v3_hmdb_disk2/hmdb_16b_20260609_140921` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_arf_only_v3_hmdb16_cuda2_launcher_20260609_140919.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_arf_only_v3_hmdb_disk2_20260609_140920.queue.log` |

启动验证：

```text
bash -n passed for both scripts
saved config verified for both runs
```

完成结果：

| Dataset | Bit | Experiment | Best Epoch | mAP@5 | mAP@20 | mAP@40 | mAP@60 | mAP@80 | mAP@100 | P@100 | R@100 | Final mAP@100 | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HMDB51 | 16 | Hybrid ARF v1, replace memory neighbor | 85 | 0.3120 | 0.2324 | 0.1737 | 0.1364 | 0.1102 | 0.0924 | 0.1633 | 0.2178 | 0.0908 | close to Stage1 HMDB16 but still lower |
| HMDB51 | 16 | Hybrid ARF v2, lower batch / higher ARF | 135 | 0.3004 | 0.2114 | 0.1537 | 0.1196 | 0.0956 | 0.0802 | 0.1480 | 0.1973 | 0.0792 | worse than v1; ARF weight too high or batch too low |
| HMDB51 | 16 | ARF-only v3 | 120 | 0.1796 | 0.1016 | 0.0693 | 0.0540 | 0.0449 | 0.0392 | 0.1116 | 0.1489 | 0.0379 | confirms ARF-only is insufficient |

## 2026-06-10 Stage5 Hybrid ARF Lambda Sweep

目的：围绕 v1 的有效区域继续细调，只改变 `L_ARF` 权重；固定 view/batch contrastive，避免再次削弱主监督。

固定损失：

```text
L_view = 0.30
L_batch_neighbor = 0.50
L_memory_neighbor = 0
L_quant = 0.02
L_balance = 0.03
```

变量：

```text
L_ARF = 0.02 / 0.04 / 0.06
```

代码/脚本：

```text
tools/run_rf_clath_hmdb_hybrid_arf_lambda_disk2.sh
base config: configs/rf_clath_hmdb_hybrid_arf.yaml
```

启动任务：

| Dataset | Bit | L_ARF | GPU | Status | Launcher PID | Train PID | Train Dir | Launcher Log | Queue Log |
|---|---:|---:|---:|---|---:|---:|---|---|---|
| HMDB51 | 16 | 0.02 | cuda0 | completed | 499693 | 499702 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_hybrid_arf_lambda0p02_hmdb_disk2/hmdb_16b_20260610_004604` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_hybrid_arf_lambda0p02_hmdb16_cuda0_launcher_20260610_004558.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_hybrid_arf_lambda0p02_hmdb_disk2_20260610_004558.queue.log` |
| HMDB51 | 16 | 0.04 | cuda1 | completed | 500451 | 500461 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_hybrid_arf_lambda0p04_hmdb_disk2/hmdb_16b_20260610_004611` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_hybrid_arf_lambda0p04_hmdb16_cuda1_launcher_20260610_004608.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_hybrid_arf_lambda0p04_hmdb_disk2_20260610_004608.queue.log` |
| HMDB51 | 16 | 0.06 | cuda2 | completed | 502138 | 502156 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_hybrid_arf_lambda0p06_hmdb_disk2/hmdb_16b_20260610_004621` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_hybrid_arf_lambda0p06_hmdb16_cuda2_launcher_20260610_004619.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_hybrid_arf_lambda0p06_hmdb_disk2_20260610_004619.queue.log` |

启动验证：

```text
bash -n passed
saved config verified:
  lambda0p02: view=0.30, batch=0.50, arf=0.02
  lambda0p04: view=0.30, batch=0.50, arf=0.04
  lambda0p06: view=0.30, batch=0.50, arf=0.06
all first epoch logs started normally
```

完成结果：

| Dataset | Bit | L_ARF | Best Epoch | mAP@5 | mAP@20 | mAP@40 | mAP@60 | mAP@80 | mAP@100 | P@100 | R@100 | Final mAP@100 | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HMDB51 | 16 | 0.02 | 65 | 0.2996 | 0.2264 | 0.1727 | 0.1372 | 0.1114 | 0.0937 | 0.1677 | 0.2237 | 0.0909 | best in this sweep, but below Stage1 |
| HMDB51 | 16 | 0.04 | 65 | 0.2990 | 0.2268 | 0.1734 | 0.1382 | 0.1119 | 0.0938 | 0.1670 | 0.2227 | 0.0905 | numerically tied with 0.02; old v1 best was 0.0924 |
| HMDB51 | 16 | 0.06 | 85 | 0.3101 | 0.2266 | 0.1688 | 0.1344 | 0.1084 | 0.0909 | 0.1621 | 0.2161 | 0.0892 | higher ARF hurts mAP@100 |

结论：

```text
The lambda sweep did not beat Stage1 HMDB16:
  Stage1 HMDB16 best mAP@100 = 0.0994
  best sweep mAP@100 = 0.0938

L_ARF=0.02 and 0.04 are effectively tied.
L_ARF=0.06 is worse, consistent with v2 showing that too much ARF weakens retrieval.
```

## 2026-06-10 ARF Memory Contrastive Implementation

目的：按当前分析把 ARF 从 BCE soft-target fitting 改成 contrastive 形式。ARF 不再直接拟合 `P_ij`，只负责提供 memory contrastive 的邻居来源。

新增 objective：

```text
arf_memory_contrastive
aliases:
  contrastive_arf
  hybrid_contrastive_arf
```

损失形式：

```text
L_total =
  0.30 * L_view
+ 0.50 * L_batch_neighbor
+ lambda_arf * L_ARF_memory_InfoNCE
+ 0.02 * L_quant
+ 0.03 * L_balance
```

其中 `L_ARF_memory_InfoNCE`：

```text
positives:
  Planner top-M 中前 positive_topk 个样本
  可选加入 planned ∩ actual overlap

denominator:
  PlannerMemoryBank.u_bank 中全部有效样本，排除 self

hard negatives:
  actual retrieval 中存在、但 planner 中不存在的样本
  在 denominator 中乘 hard_negative_weight
```

默认脚本参数：

```text
ARF_LAMBDA=0.04
ARF_POSITIVE_TOPK=10
ARF_POSITIVE_THRESHOLD=0.0
ARF_HARD_NEGATIVE_WEIGHT=1.5
```

新增脚本：

```text
tools/run_rf_clath_hmdb_arf_memory_contrastive_disk2.sh
tools/run_rf_clath_ucf_arf_memory_contrastive_disk2.sh
```

远端验证：

```text
py_compile: passed
bash -n: passed
fake memory-bank forward/backward sanity: passed
example sanity metrics:
  loss=1.2858
  arf_raw=2.3615
  arf_positive_count=4.19
  arf_hard_negative_count=4.69
```

训练日志查看重点：

```text
arf_raw      = L_ARF_memory_InfoNCE raw value
arf_targets  = positives per anchor
arf_hpos     = planned-not-actual hard positives per anchor
arf_hard     = actual-not-planned hard negatives per anchor
arf_overlap  = planned/actual overlap
arf_false    = actual retrieval false ratio
arf_missed   = planned missed by actual ratio
```

## 2026-06-10 ARF Memory Contrastive A/C Launch

目的：并行验证两条低风险改进：

```text
A: missed hard positive weighting
   hard positive = N_i \ A_i
   numerator weight = 1.5

C: delayed actual trace / hard mining
   actual_trace_start_epoch = 40
   hard_mining_start_epoch = 40
```

实现修正：

```text
hard mining is enabled only when actual trace is enabled.
This prevents warmup epochs from treating all planned neighbors as missed positives.
```

启动任务：

| Dataset | Bit | Experiment | GPU | Launcher PID | Train PID | Train Dir | Launcher Log | Queue Log | Notes |
|---|---:|---|---:|---:|---:|---|---|---|---|
| HMDB51 | 16 | A: missed hard positive | cuda2 | 3394129 | 3394137 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_arf_mem_contrastive_A_hmdb16_hmdb_disk2/hmdb_16b_20260610_102658` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_arf_mem_contrastive_A_hmdb16_cuda2_launcher_20260610_102657.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_arf_mem_contrastive_A_hmdb16_hmdb_disk2_20260610_102657.queue.log` | restarted after warmup hard-positive guard |
| HMDB51 | 16 | C: delayed hard mining | cuda3 | 3353673 | 3353682 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_arf_mem_contrastive_C_hmdb16_hmdb_disk2/hmdb_16b_20260610_102258` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_arf_mem_contrastive_C_hmdb16_cuda3_launcher_20260610_102256.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_arf_mem_contrastive_C_hmdb16_hmdb_disk2_20260610_102256.queue.log` | actual/hard mining start at epoch 40 |

启动验证：

```text
remote py_compile: passed
bash -n: passed
fake sanity:
  A: hpos > 0, hard > 0
  C pre-start: hpos = 0, hard = 0

A first epoch after restart:
  arf_targets=10.0
  arf_hpos=0.0
  arf_hard=0.0

C first epochs:
  arf_targets=10.0
  arf_hpos=0.0
  arf_hard=0.0
```

## 2026-06-10 Agentic Unified Contrastive v1

目的：把 Stage1 的 `L_view / L_batch_neighbor / L_memory_neighbor` 和 ARF 的 planned/actual feedback 融合成一个 source-aware InfoNCE，而不是多个 contrastive loss 外层相加。

实现口径：

```text
objective: agentic_unified_contrastive

L_total =
  L_agentic_unified_contrastive
+ 0.02 * L_quant
+ 0.03 * L_balance

candidate pool:
  current batch two-view candidates
  valid PlannerMemoryBank.u_bank entries

positive sources:
  paired view same video:              1.00
  batch raw-feature neighbor:          0.75
  memory raw-feature neighbor top10:   0.25
  planner planned/overlap positive:    0.25
  missed hard positive bonus:          0.25

hard negative:
  actual-not-planned denominator weight = 1.25

curriculum:
  actual_trace_start_epoch = 30
  hard_mining_start_epoch = 30
```

新增文件：

```text
configs/rf_clath_hmdb_agentic_unified.yaml
tools/run_rf_clath_hmdb_agentic_unified_disk2.sh
losses/arf_loss.py: AgenticUnifiedContrastiveLoss
```

远端验证：

```text
py_compile: passed
bash -n: passed
criterion build: AgenticUnifiedContrastiveLoss requires_planner_memory=True

smoke, HMDB16, batch_size=256, max_steps=2:
  agentic_raw > 0
  agentic_pos_view > 0
  agentic_pos_batch > 0
  agentic_pos_memory > 0
  agentic_pos_arf > 0
  pre-start agentic_hpos = 0
  pre-start agentic_hneg = 0

hard-mining sanity, HMDB16, batch_size=256, max_steps=2,
temporary actual_trace_start_epoch=1 / hard_mining_start_epoch=1:
  agentic_hpos > 0
  agentic_hneg > 0
```

正式实验：

| Dataset | Bit | Experiment | GPU | Launcher PID | Train PID | Train Dir | Launcher Log | Queue Log | Status |
|---|---:|---|---:|---:|---:|---|---|---|---|
| HMDB51 | 16 | Agentic Unified Contrastive v1 | cuda0 | 518022 | 518032 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_agentic_unified_v1_hmdb_disk2/hmdb_16b_20260610_151305` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_agentic_unified_hmdb16_cuda0_launcher_20260610_151304.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_agentic_unified_v1_hmdb_disk2_20260610_151304.queue.log` | running |

对比目标：

```text
Stage1 HMDB16 best mAP@100 = 0.0994
A missed-positive best mAP@100 = 0.0993
```

Update:

```text
cuda0 run stopped before checkpoint due GPU load.
Restarted direct Agentic Unified v1 on cuda2:
  launcher PID: 564028
  train PID: 564035
  run: /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_agentic_unified_v1_cuda2_hmdb_disk2/hmdb_16b_20260610_152357
  queue log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_agentic_unified_v1_cuda2_hmdb_disk2_20260610_152356.queue.log
```

## 2026-06-10 Stage1 Warmup 50 -> Agentic Unified v1

目的：先用已验证稳定的 Stage1 original loss 建立检索空间，再切换到 agentic unified contrastive 进行 retrieval-feedback refinement。

训练策略：

```text
epoch 1-50:
  0.30 * L_view
+ 0.50 * L_batch_neighbor
+ 0.04 * L_memory_neighbor
+ 0.02 * L_quant
+ 0.03 * L_balance

epoch 51-150:
  L_agentic_unified_contrastive
+ 0.02 * L_quant
+ 0.03 * L_balance

actual_trace_start_epoch = 51
hard_mining_start_epoch = 51
```

实现：

```text
objective: stage1_warmup_agentic_unified
loss wrapper: Stage1WarmupAgenticUnifiedLoss
script: tools/run_rf_clath_hmdb_stage1_warmup_agentic_unified_disk2.sh
```

启动记录：

| Dataset | Bit | Experiment | GPU | Launcher PID | Train PID | Train Dir | Launcher Log | Queue Log | Status |
|---|---:|---|---:|---:|---:|---|---|---|---|
| HMDB51 | 16 | Stage1 warmup 50 -> Agentic Unified v1 | cuda3 | 645235 | 645242 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_stage1warm50_agentic_unified_v1_hmdb_disk2/hmdb_16b_20260610_154220` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_stage1warm50_agentic_unified_hmdb16_cuda3_launcher_20260610_154218.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_stage1warm50_agentic_unified_v1_hmdb_disk2_20260610_154218.queue.log` | running |

停止记录（2026-06-11）：

```text
Stopped direct Agentic Unified v1 on cuda2:
  run: /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_agentic_unified_v1_cuda2_hmdb_disk2/hmdb_16b_20260610_152357
  stopped at train epoch 61
  best eval epoch 45:
    mAP@5   = 0.1916
    mAP@20  = 0.1317
    mAP@40  = 0.1022
    mAP@60  = 0.0840
    mAP@80  = 0.0708
    mAP@100 = 0.0618
    P@100   = 0.1418
    R@100   = 0.1891
  last eval epoch 60:
    mAP@100 = 0.0615
  conclusion:
    Directly replacing Stage1 with unified agentic contrastive is clearly worse than Stage1 HMDB16 mAP@100=0.0994.

Stopped Stage1 warmup 50 -> Agentic Unified v1 on cuda3:
  run: /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_stage1warm50_agentic_unified_v1_hmdb_disk2/hmdb_16b_20260610_154220
  stopped at train epoch 116
  best eval epoch 65:
    mAP@5   = 0.3221
    mAP@20  = 0.2465
    mAP@40  = 0.1915
    mAP@60  = 0.1524
    mAP@80  = 0.1219
    mAP@100 = 0.1018
    P@100   = 0.1724
    R@100   = 0.2298
  last eval epoch 115:
    mAP@100 = 0.1003
  last agentic diagnostics, train epoch 116:
    agentic_raw        = 3.8456
    agentic_pos_view   = 1.1
    agentic_pos_batch  = 7.5
    agentic_pos_memory = 10.0
    agentic_pos_arf    = 14.7
    agentic_hpos       = 2.5
    agentic_hneg       = 7.7
    agentic_pos_weight = 0.092
  conclusion:
    Warmup switch improves over Stage1 HMDB16 mAP@100=0.0994 by +0.0024, but the peak appears soon after switching and then mildly decays.
    Next tuning should focus on switch timing, switch ramp, and whether to keep part of Stage1 contrastive after the switch.
```

Correction / resume:

```text
The Stage1 warmup 50 -> Agentic Unified v1 run should not have been stopped.
It was resumed from:
  /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_stage1warm50_agentic_unified_v1_hmdb_disk2/hmdb_16b_20260610_154220/epoch_0115.pth

Resume command status:
  GPU: cuda3
  PID: 1008191
  start_epoch: 116
  resume log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_stage1warm50_agentic_unified_resume_hmdb16_cuda3_20260610_171639.log

Keep this run as the active switch-strategy baseline and continue it to epoch 150 unless the next tuning plan replaces it.
```

## 2026-06-11 Switch Timing Sweep: Warmup 40 / 50 / 60

目的：验证 agentic unified contrastive 的最佳切换时机。当前证据显示直接使用 agentic unified 会崩，warmup50 硬切能小幅超过 Stage1，但切后长训会回落。因此先固定损失形式，只 sweep 切换 epoch。

共同设置：

```text
dataset: HMDB51
bits: 16
batch_size: 256
epoch 1-warmup:
  0.30 * L_view
+ 0.50 * L_batch_neighbor
+ 0.04 * L_memory_neighbor
+ 0.02 * L_quant
+ 0.03 * L_balance

epoch warmup+1 - 150:
  L_agentic_unified_contrastive
+ 0.02 * L_quant
+ 0.03 * L_balance

actual_trace_start_epoch = warmup + 1
hard_mining_start_epoch  = warmup + 1
```

启动记录：

| Dataset | Bit | Experiment | GPU | Launcher PID | Train PID | Train Dir | Launcher Log | Queue Log | Status |
|---|---:|---|---:|---:|---:|---|---|---|---|
| HMDB51 | 16 | Stage1 warmup 40 -> Agentic Unified v1 | cuda0 | 1018108 | 1018121 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_stage1warm40_agentic_unified_v1_hmdb_disk2` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_stage1warm40_agentic_unified_hmdb16_cuda0_launcher_20260610_171903.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_stage1warm40_agentic_unified_v1_hmdb_disk2_20260610_171903.queue.log` | running |
| HMDB51 | 16 | Stage1 warmup 50 -> Agentic Unified v1 | cuda3 | resumed | 1008191 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_stage1warm50_agentic_unified_v1_hmdb_disk2/hmdb_16b_20260610_154220` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_stage1warm50_agentic_unified_resume_hmdb16_cuda3_20260610_171639.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_stage1warm50_agentic_unified_v1_hmdb_disk2_20260610_154218.queue.log` | running from epoch 116 |
| HMDB51 | 16 | Stage1 warmup 60 -> Agentic Unified v1 | cuda2 | 1018110 | 1018124 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_stage1warm60_agentic_unified_v1_hmdb_disk2` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_stage1warm60_agentic_unified_hmdb16_cuda2_launcher_20260610_171903.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_stage1warm60_agentic_unified_v1_hmdb_disk2_20260610_171903.queue.log` | running |

判定口径：

```text
Primary: best mAP@100
Secondary: mAP@20 / P@100 / R@100
Watch:
  peak epoch after switch
  whether mAP@100 decays after peak
  agentic_hpos / agentic_hneg activation after switch
```

阶段结果（2026-06-11）：

| Experiment | Status | Best Epoch | Best mAP@20 | Best mAP@100 | P@100 | R@100 | Last Eval | Last mAP@100 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| warmup40 -> epoch41 switch | running, train epoch 98 | 80 | 0.2480 | 0.1022 | 0.1684 | 0.2245 | 95 | 0.0976 |
| warmup50 -> epoch51 switch | finished epoch 150 | 65 | 0.2465 | 0.1018 | 0.1724 | 0.2298 | 150 | 0.0983 |
| warmup60 -> epoch61 switch | finished epoch 150 | 105 | 0.2479 | 0.1036 | 0.1714 | 0.2285 | 150 | 0.1006 |

对比：

```text
Stage1 HMDB16 baseline:
  best mAP@100 = 0.0994

warmup40:
  best mAP@100 = 0.1022, +0.0028 over Stage1
  peak appears around epoch 80, then decays to 0.0976 by epoch 95.

warmup50:
  best mAP@100 = 0.1018, +0.0024 over Stage1
  peak appears around epoch 65, then decays to 0.0983 by epoch 150.

warmup60:
  best mAP@100 = 0.1036, +0.0042 over Stage1
  peak appears around epoch 105, then decays to 0.1006 by epoch 150.
```

当前判断：

```text
1. warmup60 is the best current switch timing.
2. All hard-switch variants show post-peak decay.
3. Agentic hard mining is active and stable after switch:
   warm40 latest agentic_hpos=2.6, agentic_hneg=7.8
   warm50 latest agentic_hpos=2.4, agentic_hneg=7.7
   warm60 latest agentic_hpos=2.5, agentic_hneg=7.9
4. Next experiment should not be another earlier hard switch.
   More promising:
     a) warmup60 with early stop / short post-switch window
     b) warmup60 with Stage1-to-agentic ramp
     c) warmup60 with partial Stage1 contrastive retained after switch
```

## 2026-06-11 Warmup60 Switch-Strategy Follow-up

目的：基于 warmup60 hard switch 的当前最佳结果，继续验证三条切换策略：

```text
baseline in this branch:
  warmup60 hard switch
  best epoch 105
  best mAP@100 = 0.1036

new variants:
  1. short window:
     warmup60 hard switch, run_until_epoch=110

  2. ramp:
     epoch 1-60 Stage1
     epoch 61-80 linear ramp from Stage1 to AgenticUnified
     epoch 81-150 AgenticUnified

  3. retain Stage1:
     epoch 1-60 Stage1
     epoch 61-150 0.5 * Stage1 + 0.5 * AgenticUnified
```

实现：

```text
loss wrapper: Stage1ScheduledAgenticUnifiedLoss
objective: stage1_scheduled_agentic_unified
script: tools/run_rf_clath_hmdb_stage1_scheduled_agentic_unified_disk2.sh

new schedule diagnostics:
  mix_alpha
  stage1_keep
```

远端验证：

```text
py_compile: passed
bash -n scheduled script: passed

schedule check:
  hard:
    epoch60 -> (stage1=1.0, agentic=0.0)
    epoch61 -> (stage1=0.0, agentic=1.0)

  ramp20:
    epoch60 -> (stage1=1.0, agentic=0.0)
    epoch61 -> (stage1=0.95, agentic=0.05)
    epoch80 -> (stage1=0.0, agentic=1.0)

  retain50:
    epoch60 -> (stage1=1.0, agentic=0.0)
    epoch61+ -> (stage1=0.5, agentic=0.5)
```

启动记录：

| Dataset | Bit | Experiment | GPU | Launcher PID | Train PID | Train Dir | Launcher Log | Queue Log | Status |
|---|---:|---|---:|---:|---:|---|---|---|---|
| HMDB51 | 16 | warmup60 short window, hard switch, stop at epoch110 | cuda2 | 2499853 | 2499860 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_stage1warm60_short110_agentic_unified_v1_hmdb_disk2` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_stage1warm60_short110_agentic_unified_hmdb16_cuda2_launcher_20260611_003353.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_stage1warm60_short110_agentic_unified_v1_hmdb_disk2_20260611_003353.queue.log` | running |
| HMDB51 | 16 | warmup60 ramp20 -> AgenticUnified | cuda3 | 2501110 | 2501119 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_stage1warm60_ramp20_agentic_unified_v1_hmdb_disk2` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_stage1warm60_ramp20_agentic_unified_hmdb16_cuda3_launcher_20260611_003400.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_stage1warm60_ramp20_agentic_unified_v1_hmdb_disk2_20260611_003400.queue.log` | running |
| HMDB51 | 16 | warmup60 retain50, 0.5 Stage1 + 0.5 AgenticUnified after switch | cuda0 | 2502301 | 2502308 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_stage1warm60_retain50_agentic_unified_v1_hmdb_disk2` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_stage1warm60_retain50_agentic_unified_hmdb16_cuda0_launcher_20260611_003408.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_stage1warm60_retain50_agentic_unified_v1_hmdb_disk2_20260611_003408.queue.log` | running |

Current status check:

```text
2026-06-11:

Still running:
  warmup40 hard switch:
    train epoch: 124
    latest eval epoch: 120
    best epoch: 80
    best mAP@100: 0.1022
    latest mAP@100: 0.0977
    interpretation: already peaked and decayed; no longer promising as a final setting.

  warmup60 short110:
    train epoch: 2
    latest eval: none yet
    mix_alpha=0.0, stage1_keep=1.0
    interpretation: still in Stage1 warmup; no retrieval result yet.

  warmup60 ramp20:
    train epoch: 2
    latest eval: none yet
    mix_alpha=0.0, stage1_keep=1.0
    interpretation: still in Stage1 warmup; ramp starts at epoch 61.

  warmup60 retain50:
    train epoch: 1
    latest eval: none yet
    mix_alpha=0.0, stage1_keep=1.0
    interpretation: still in Stage1 warmup; 0.5/0.5 hybrid starts at epoch 61.

Reference:
  warmup60 hard switch completed:
    best epoch: 105
    best mAP@100: 0.1036
    latest epoch 150 mAP@100: 0.1006
```

## 2026-06-11 True Two-Phase AUCL Implementation

目的：把 `Stage1ScheduledAgenticUnifiedLoss` 从“旧 Stage1 loss + Agentic loss”改成真正的一个 AUCL loss 的两阶段 source schedule。

实现后训练口径：

```text
L_total(t) =
  L_AUCL(t)
+ 0.02 * L_quant
+ 0.03 * L_balance

Phase I bootstrap:
  source = {view, batch_neighbor, memory_neighbor}
  arf_planned = 0
  arf_missed_bonus = 0
  hard_negative_weight = 1.0
  actual_trace = false
  hard_mining = false

Phase II agentic refinement:
  source = {view, batch_neighbor, memory_neighbor, arf_planned, arf_missed_bonus}
  hard_negative_weight = 1.25
  actual_trace = true
  hard_mining = true
```

代码变更：

```text
losses/arf_loss.py:
  Added PhasedAgenticUnifiedContrastiveLoss.
  Stage1ScheduledAgenticUnifiedLoss now subclasses phased AUCL.
  Stage1WarmupAgenticUnifiedLoss remains a compatibility alias.
  Removed RFClathLoss dependency from scheduled/warmup AUCL objectives.

engine/train.py:
  Added phased_agentic_unified / phased_agentic_unified_contrastive objective aliases.

tools:
  Updated scheduled/warmup HMDB launch scripts to pass AUCL source weights instead of old Stage1 loss weights.
```

远端轻量验证：

```text
py_compile: passed
bash -n scheduled script: passed
bash -n warmup script: passed

hard:
  epoch 1:  mix_alpha=0.0, stage1_keep=1.0, arf=0.0, hard_negative=1.0
  epoch 60: mix_alpha=0.0, stage1_keep=1.0, arf=0.0, hard_negative=1.0
  epoch 61: mix_alpha=1.0, stage1_keep=0.0, arf=0.25, hard_negative=1.25

ramp20:
  epoch 61: mix_alpha=0.05, stage1_keep=0.95, arf=0.0125, hard_negative=1.0125
  epoch 80: mix_alpha=1.0, stage1_keep=0.0, arf=0.25, hard_negative=1.25

retain50:
  epoch 61: mix_alpha=0.5, stage1_keep=0.5, arf=0.125, hard_negative=1.125

aliases:
  stage1_warmup_agentic_unified -> Phased AUCL
  stage1_scheduled_agentic_unified -> Phased AUCL
  phased_agentic_unified_contrastive -> Phased AUCL
```

备注：

```text
Current running experiments were not stopped. They keep the Python code loaded at process start.
Future experiments launched from the updated scripts will use the true two-phase AUCL implementation.
```

Running experiment status check:

```text
2026-06-11:

Important:
  The currently running warmup60 short/ramp/retain jobs were launched before the true two-phase AUCL code change.
  They are old-code jobs and still use the legacy Stage1 warmup behavior inside the running Python processes.
  Future launches from the updated scripts will use true two-phase AUCL.

Still running:
  warmup40 hard switch:
    train epoch: 132
    latest eval epoch: 130
    best epoch: 80
    best mAP@100: 0.1022
    latest mAP@100: 0.0978
    interpretation: peaked early and has clearly decayed; not a promising final setting.

  warmup60 short110:
    train epoch: 13
    latest eval epoch: 10
    best mAP@100: 0.0608
    mix_alpha=0.0, stage1_keep=1.0
    interpretation: still in Phase I warmup; no switch-strategy signal yet.

  warmup60 ramp20:
    train epoch: 11
    latest eval epoch: 10
    best mAP@100: 0.0608
    mix_alpha=0.0, stage1_keep=1.0
    interpretation: still in Phase I warmup; ramp starts at epoch 61.

  warmup60 retain50:
    train epoch: 9
    latest eval epoch: 5
    best mAP@100: 0.0458
    mix_alpha=0.0, stage1_keep=1.0
    interpretation: still in Phase I warmup; retain50 starts at epoch 61.

Reference completed run:
  warmup60 hard switch:
    best epoch: 105
    best mAP@100: 0.1036
    latest epoch 150 mAP@100: 0.1006
```

Running / completed old-code switch-strategy results:

```text
2026-06-11:

Important:
  These jobs were launched before the true two-phase AUCL code change.
  They are useful as old-code switch-strategy ablations, but not as final true-AUCL evidence.

warmup40 hard switch:
  status: finished epoch 150
  best epoch: 80
  best mAP@100: 0.1022
  last epoch 150 mAP@100: 0.0975
  conclusion: earlier switch peaks but decays badly.

warmup60 short110:
  status: finished epoch 110
  best epoch: 105
  best mAP@5:   0.3177
  best mAP@20:  0.2486
  best mAP@100: 0.1033
  last epoch 110 mAP@100: 0.1012
  schedule sanity:
    epoch 60: mix_alpha=0.0, stage1_keep=1.0
    epoch 61: mix_alpha=1.0, stage1_keep=0.0
  conclusion: close to warmup60 hard switch, but slightly lower than 0.1036.

warmup60 ramp20:
  status: still running, latest train epoch 110
  best epoch so far: 80
  best mAP@5:   0.3276
  best mAP@20:  0.2499
  best mAP@100: 0.1031
  latest epoch 110 mAP@100: 0.1003
  schedule sanity:
    epoch 60: mix_alpha=0.0, stage1_keep=1.0
    epoch 61: mix_alpha=0.05, stage1_keep=0.95
    epoch 80: mix_alpha=1.0, stage1_keep=0.0
  conclusion: ramp improves early peak timing and mAP@5/mAP@20, but not mAP@100.

warmup60 retain50:
  status: finished epoch 150
  best epoch: 45
  best mAP@100: 0.0983
  last epoch 150 mAP@100: 0.0958
  schedule sanity:
    epoch 61+: mix_alpha=0.5, stage1_keep=0.5
  conclusion: retaining old Stage1 objective at 0.5 hurts mAP@100 in this old-code wrapper.

Reference:
  warmup60 hard switch:
    best epoch: 105
    best mAP@100: 0.1036
    epoch 150 mAP@100: 0.1006

Current best old-code setting:
  warmup60 hard switch remains best by mAP@100.

Next useful run:
  Relaunch true two-phase AUCL warmup60 hard / ramp20 / retain50 from the updated codebase when server resources are available.
```

## 2026-06-11 True Two-Phase AUCL Launch

目的：使用已更新的 true two-phase AUCL 实现，重新运行 warmup60 hard/short、ramp20、retain50 三组。区别于上一批 old-code jobs：Phase I 不再调用旧 `RFClathLoss`，而是同一个 AUCL 的 bootstrap source schedule。

共同设置：

```text
dataset: HMDB51
bits: 16
batch_size: 256
objective: stage1_scheduled_agentic_unified -> PhasedAgenticUnifiedContrastiveLoss

Phase I, epoch 1-60:
  L_AUCL(source={view,batch_neighbor,memory_neighbor})
  arf_planned = 0
  arf_missed_bonus = 0
  hard_negative_weight = 1.0
  actual_trace = false
  hard_mining = false

Phase II:
  hard/short110:
    epoch 61-110, mix_alpha=1.0

  ramp20:
    epoch 61-80, mix_alpha linearly 0.05 -> 1.0
    epoch 81-150, mix_alpha=1.0

  retain50:
    epoch 61-150, mix_alpha=0.5, stage1_keep=0.5
```

启动记录：

| Dataset | Bit | Experiment | GPU | Launcher PID | Train PID | Train Dir | Launcher Log | Queue Log | Status |
|---|---:|---|---:|---:|---:|---|---|---|---|
| HMDB51 | 16 | true-AUCL warmup60 hard, short window to epoch110 | cuda0 | 3534564 | 3534571 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_trueaucl_stage1warm60_short110_v1_hmdb_disk2` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_trueaucl_warm60_short110_hmdb16_cuda0_launcher_20260611_045434.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_trueaucl_stage1warm60_short110_v1_hmdb_disk2_20260611_045434.queue.log` | running |
| HMDB51 | 16 | true-AUCL warmup60 ramp20 | cuda2 | 3535820 | 3535828 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_trueaucl_stage1warm60_ramp20_v1_hmdb_disk2` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_trueaucl_warm60_ramp20_hmdb16_cuda2_launcher_20260611_045442.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_trueaucl_stage1warm60_ramp20_v1_hmdb_disk2_20260611_045442.queue.log` | running |
| HMDB51 | 16 | true-AUCL warmup60 retain50 | cuda3 | 3537313 | 3537322 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_trueaucl_stage1warm60_retain50_v1_hmdb_disk2` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_trueaucl_warm60_retain50_hmdb16_cuda3_launcher_20260611_045450.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_trueaucl_stage1warm60_retain50_v1_hmdb_disk2_20260611_045450.queue.log` | running |

备注：

```text
cuda3 still has the old-code warm60 ramp20 process running.
true-AUCL retain50 was placed on cuda3 because memory remained sufficient.
```

## 2026-06-11 AUCL v2 HMDB16 Launch

目的：停止 cuda2 上旧的 true-AUCL ramp20 方向，改跑重新设计后的 source-factored AUCL v2。

停止记录：

```text
stopped:
  launcher PID: 3535820
  train PID: 3535828
  experiment: true-AUCL warmup60 ramp20
  queue log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_trueaucl_stage1warm60_ramp20_v1_hmdb_disk2_20260611_045442.queue.log
```

新实验设置：

```text
dataset: HMDB51
bits: 16
gpu: cuda2
objective: agentic_unified_contrastive_v2

L_AUCL_v2:
  0.30 L_view_pair
  0.50 L_batch_neighbor
  0.04 L_memory_agentic
  + 0.02 L_quant
  + 0.03 L_balance

schedule:
  epoch 1-60: beta=0, Stage1-equivalent memory raw positives
  epoch 61-80: beta ramps to 0.25
  epoch 80+: hard mining enabled
```

启动记录：

| Dataset | Bit | Experiment | GPU | Launcher PID | Train PID | Train Dir | Launcher Log | Queue Log | Status |
|---|---:|---|---:|---:|---:|---|---|---|---|
| HMDB51 | 16 | AUCL v2 warm60 ramp20 | cuda2 | 2192714 | 2192724 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_agentic_unified_v2_warm60_ramp20_hmdb16_cuda2_hmdb_disk2/hmdb_16b_20260611_104019` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_auclv2_hmdb16_cuda2_launcher_20260611_104018.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_agentic_unified_v2_warm60_ramp20_hmdb16_cuda2_hmdb_disk2_20260611_104018.queue.log` | completed |

完成结果：

| Dataset | Bit | Experiment | Best Epoch | mAP@5 | mAP@20 | mAP@40 | mAP@60 | mAP@80 | mAP@100 | P@100 | R@100 | Last Epoch | Last mAP@100 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HMDB51 | 16 | AUCL v2 warm60 ramp20 | 115 | 0.3207 | 0.2408 | 0.1857 | 0.1453 | 0.1173 | 0.0983 | 0.1689 | 0.2252 | 150 | 0.0963 |

对比：

```text
Stage1 HMDB16 baseline:             best mAP@100 = 0.0994
old warmup60 hard switch:           best mAP@100 = 0.1036
failed true-AUCL v1 ramp20:          best mAP@100 = 0.0653
AUCL v2 warm60 ramp20:               best mAP@100 = 0.0983
```

诊断：

```text
AUCL v2 明显避免了 true-AUCL v1 的崩塌，但未超过 Stage1 baseline。
epoch 80 后 hard mining 正常开启，late epoch 典型值：
  aucl_beta ~= 0.250
  aucl_pos_raw ~= 10.0
  aucl_pos_planned ~= 5.0
  aucl_pos_missed ~= 4.7-4.8
  aucl_hneg ~= 9.0-9.3
  aucl_memory_raw ~= 2.71
  aucl_memory_feedback ~= 2.16
```

## 2026-06-11 Legacy Stage1-No-Memory Warmup60 -> Agentic Unified v1 HMDB16

目的：验证当前最好方向 `warmup60 -> epoch61 hard switch` 中，Phase I 去掉 `L_memory_neighbor`
是否改善后续 v1 AUCL 切换效果。

实现口径：

```text
objective: legacy_stage1_warmup_agentic_unified
loss wrapper: LegacyStage1WarmupAgenticUnifiedLoss

epoch 1-60:
  0.30 * L_view
+ 0.50 * L_batch_neighbor
+ 0.00 * L_memory_neighbor
+ 0.02 * L_quant
+ 0.03 * L_balance

epoch 61-150:
  L_agentic_unified_contrastive_v1
+ 0.02 * L_quant
+ 0.03 * L_balance

actual_trace_start_epoch = 61
hard_mining_start_epoch = 61
```

远端轻量验证：

```text
py_compile: passed
bash -n: passed
criterion smoke:
  class: LegacyStage1WarmupAgenticUnifiedLoss
  stage1_memory_lambda: 0.0
  agentic_memory_source: 0.25
  weights60: (1.0, 0.0)
  weights61: (0.0, 1.0)
```

启动记录：

| Dataset | Bit | Experiment | GPU | Launcher PID | Train PID | Train Dir | Launcher Log | Queue Log | Status |
|---|---:|---|---:|---:|---:|---|---|---|---|
| HMDB51 | 16 | legacy Stage1-no-memory warmup60 -> AUCL v1 hard switch | cuda2 | 1005259 | 1005270 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_legacy_stage1nomem_warm60_agentic_unified_v1_hmdb_disk2/hmdb_16b_20260611_152256` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_legacy_stage1nomem_warm60_agentic_unified_hmdb16_cuda2_launcher_20260611_152255.log` | `/mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_legacy_stage1nomem_warm60_agentic_unified_v1_hmdb_disk2_20260611_152255.queue.log` | completed |

首个训练 sanity：

```text
epoch=1 step=20/29
loss=4.3943
view_raw=3.2502
batch_neigh=2.2693
mem_neigh=0.0000
agentic_raw=0.0000
mix_alpha=0.000
stage1_keep=1.000
```

完成结果：

| Dataset | Bit | Experiment | Best Epoch | mAP@5 | mAP@20 | mAP@40 | mAP@60 | mAP@80 | mAP@100 | P@100 | R@100 | Last Epoch | Last mAP@100 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HMDB51 | 16 | legacy Stage1-no-memory warmup60 -> AUCL v1 hard switch | 110 | 0.3131 | 0.2364 | 0.1827 | 0.1442 | 0.1149 | 0.0958 | 0.1610 | 0.2147 | 150 | 0.0954 |

关键 eval：

```text
epoch 60  mAP@100 = 0.0551
epoch 65  mAP@100 = 0.0680
epoch 80  mAP@100 = 0.0851
epoch 100 mAP@100 = 0.0948
epoch 110 mAP@100 = 0.0958  best
epoch 150 mAP@100 = 0.0954
```

对比：

```text
Stage1 HMDB16 baseline:                         best mAP@100 = 0.0994
old warmup60 hard switch with Stage1 memory:    best mAP@100 = 0.1036
legacy Stage1-no-memory warmup60 hard switch:   best mAP@100 = 0.0958
AUCL v2 warm60 ramp20:                          best mAP@100 = 0.0983
```

结论：

```text
去掉 Phase I 的 L_memory_neighbor 明显削弱 warmup 表征。
epoch 60 只有 0.0551，切到 AUCL v1 后能恢复到 0.0958，但仍低于 Stage1 baseline 和最好版本。
当前证据支持：Stage1 warmup 中 L_memory_neighbor 不能去掉，它对构建可切换到 agentic refinement 的检索空间是关键项。
```

## 2026-06-13 HMDB16 Remaining-Fast Stage1 Launch

目的：验证早期 slow/fast 分工设定，即 fast branch 只使用去掉关键帧后的 remaining frames，是否优于当前
`input_frames=all` 的 HMDB16 Stage1 baseline。

实验设置：

```text
dataset: HMDB51
bits: 16
gpu: cuda3
objective: RFClathLoss / Stage1 original loss

model.fast_encoder.input_frames = remaining

loss:
  0.30 L_view
+ 0.50 L_batch_neighbor
+ 0.04 L_memory_neighbor
+ 0.02 L_quant
+ 0.03 L_balance
```

对比目标：

```text
HMDB16 Stage1 all-frame fast baseline:
  best mAP@100 = 0.0994
  log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_t_sas_hmdb_disk2_20260608_142724.queue.log
```

启动记录：

| Dataset | Bit | Experiment | GPU | PID | Train Dir | Log | Status |
|---|---:|---|---:|---:|---|---|---|
| HMDB51 | 16 | T-SAS Stage1, fast input remaining frames | cuda3 | 3698325 | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_hmdb_remaining_fast_disk2/hmdb_16b_20260613_072347` | `/mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_hmdb_remaining_fast_disk2/hmdb_16b_20260613_072347/train.log` | running |

确认：

```text
saved config:
  model.fast_encoder.input_frames: remaining

launcher/nohup log:
  /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_hmdb16_remaining_fast_cuda3_20260613_072341.log

note:
  cuda3 already had substantial non-RF-CLaTH load at launch, so early progress may be slower than the previous HMDB16 baseline.
```
