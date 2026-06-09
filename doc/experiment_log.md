# RF-CLaTH Experiment Log

更新时间：2026-06-09 18:48，时区 Asia/Shanghai。

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
