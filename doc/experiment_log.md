# RF-CLaTH Experiment Log

更新时间：2026-06-09 08:27，时区 Asia/Shanghai。

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
active train pid: 2629931
remote launcher log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_t_sas_ucf_cuda1_launcher_20260608_141126.log
remote queue log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_t_sas_ucf_disk2_20260608_141126.queue.log
status: 16-bit completed; 32-bit running, current epoch 40/150; 64-bit queued
latest eval: 32-bit epoch 40, mAP@5=0.7586, mAP@20=0.6255, mAP@100=0.3809
```

结果记录：

| Bits | Run Dir | Best Checkpoint | Selected Epoch | mAP@5 | mAP@20 | mAP@40 | mAP@60 | mAP@80 | mAP@100 | P@5 | P@20 | P@100 | R@5 | R@20 | R@100 | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_ucf_disk2/s5vh_ucf_16b_20260608_141136 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_ucf_disk2/s5vh_ucf_16b_20260608_141136/best.pth | 70 | 0.6782 | 0.5820 | 0.5159 | 0.4550 | 0.3980 | 0.3434 | 0.7214 | 0.6462 | 0.4143 | 0.0361 | 0.1288 | 0.4109 | completed; final epoch 150 mAP@100=0.3410 |
| 32 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_ucf_disk2/s5vh_ucf_32b_20260608_223349 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_ucf_disk2/s5vh_ucf_32b_20260608_223349/best.pth | 35 | 0.7660 | 0.6258 | 0.5509 | 0.4933 | 0.4366 | 0.3814 | 0.7936 | 0.6751 | 0.4515 | 0.0398 | 0.1349 | 0.4458 | running; current epoch 40/150 |
| 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | queued |

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
launcher pid: 2985206
active train pid: 2985228
remote launcher log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_trainable_hmdb_cuda2_launcher_20260609_003938.log
remote queue log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_trainable_hmdb_disk2_20260609_003938.queue.log
status: running 16-bit
note: this is a selector ablation against the completed HMDB T-SAS runs above.
```

结果记录：

| Bits | Run Dir | Best Checkpoint | Selected Epoch | mAP@5 | mAP@20 | mAP@40 | mAP@60 | mAP@80 | mAP@100 | P@5 | P@20 | P@100 | R@5 | R@20 | R@100 | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_trainable_hmdb_disk2/hmdb_16b_20260609_003941 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | running |
| 32 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | queued |
| 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | queued |
