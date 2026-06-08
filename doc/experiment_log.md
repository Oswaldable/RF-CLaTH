# RF-CLaTH Experiment Log

更新时间：2026-06-08，时区 Asia/Shanghai。

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
active train pid: 1135424
remote launcher log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_t_sas_ucf_cuda1_launcher_20260608_141126.log
remote queue log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_t_sas_ucf_disk2_20260608_141126.queue.log
status: running 16-bit, building neighbor table
```

结果记录：

| Bits | Run Dir | Best Checkpoint | Selected Epoch | mAP@5 | mAP@20 | mAP@40 | mAP@60 | mAP@80 | mAP@100 | P@5 | P@20 | P@100 | R@5 | R@20 | R@100 | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_t_sas_ucf_disk2/s5vh_ucf_16b_20260608_141136 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | running |
| 32 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | queued |
| 64 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | queued |
