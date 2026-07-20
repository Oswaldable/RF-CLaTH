# RF-CLaTH UCF101 实验结果总表

更新时间：2026-07-20（Asia/Shanghai）

本页是 `plan.md` 的结果索引，只收录已经核验的数值。详细实验动机、机制分析和
后续计划见 [`plan.md`](../plan.md)，可分析的逐轮数据见：

- [`ucf101_experiment_summary.csv`](ucf101_experiment_summary.csv)：每个实验一行；
- [`ucf101_map100_curves.csv`](ucf101_map100_curves.csv)：已有的 mAP@100 评估点；
- [`ucf101_h1_trace_diagnostics.csv`](ucf101_h1_trace_diagnostics.csv)：H1 epoch 7-15
  的 actual-trace 语义诊断。

## 1. 当前结论

统一实验口径为 UCF101 新划分、32-bit、seed 3346、batch size 256。

```text
当前最优 AUCL：E1，best.pth epoch 25，mAP@100=0.2705
旧目标对照：C0，epoch 10，mAP@100=0.3015
绝对差值：-0.0310
相对差距：约 -10.3%
```

E1 相比同训练长度的无 memory 对照 E0 提升 `0.0226`，说明 warmup 后加入全量
memory 有效。F0/F1/G0/G1 均未超过 E1，因此不再继续 memory candidate 裁剪路线。

H0/H1 表明只读 actual trace 不改变训练目标，但 H1 epoch 15 的 actual-only 样本
label precision 为 `0.357`，不能把集合差异直接当作语义 hard negative。因此当前
不启用 hard mining、feedback graph 或 missed/false loss。

## 2. 阶段结果

### C0-C3：第一轮定位，10 epoch

| 实验 | 设置 | mAP@100 | P@5 | R@100 |
|---|---|---:|---:|---:|
| C0 | 旧目标 `merged_semantic_self_calibrated` | **0.3015** | **0.7395** | **0.3812** |
| C1 | AUCL，route off，memory from epoch 1 | 0.1665 | 0.5223 | 0.2610 |
| C2 | AUCL，route on，10 epoch 内无 memory | **0.2172** | **0.6447** | **0.3003** |
| C3 | AUCL，route off，epoch 6 加 memory | 0.1733 | 0.5375 | 0.2634 |

### D0-D2：memory 加入时机，10 epoch

| 实验 | 设置 | 最佳点 mAP@100 | Hash 状态 |
|---|---|---:|---|
| D0 | route on，epoch 1 全量 memory | 0.0138（epoch 5） | 坍缩：entropy=0.186，bit-use=0.188 |
| D1 | route on，epoch 6 硬切换全量 memory | 0.2268（epoch 10） | 健康 |
| D2 | route on，epoch 7-10 线性 ramp | **0.2287（epoch 10）** | 健康 |

### E0-E1：30-epoch 主线

| 实验 | best.pth epoch | best checkpoint mAP@100 | 曲线峰值 | P@5 | R@100 |
|---|---:|---:|---:|---:|---:|
| E0 无 memory | 25 | 0.2479 | 0.2484（epoch 30） | 0.7071 | 0.3240 |
| E1 full-memory ramp | 25 | **0.2705** | **0.2705（epoch 25）** | 0.6728 | **0.3487** |

E1 的主要收益来自召回：相对 E0，P@5 下降 `0.0343`，R@100 提升 `0.0247`。

### F0-G1：memory candidate 消融，30 epoch

| 实验 | best.pth epoch | mAP@100 | P@5 | R@100 |
|---|---:|---:|---:|---:|
| F0 hard-64 | 15 | 0.2146 | 0.6691 | 0.2962 |
| F1 hard-128 | 20 | 0.2434 | **0.6952** | 0.3182 |
| G0 supervised-only | 25 | 0.2373 | 0.6949 | 0.3151 |
| G1 uniform-1024 | 20 | **0.2567** | 0.6834 | **0.3355** |
| E1 full-memory | 25 | **0.2705** | 0.6728 | **0.3487** |

### H0-H1：只读 actual trace，15 epoch

| Epoch | H0 mAP@100 | H1 mAP@100 | H1-H0 | H0 P@5 / R@100 | H1 P@5 / R@100 |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.1697 | 0.1688 | -0.0009 | 0.5697 / 0.2592 | 0.5696 / 0.2583 |
| 10 | 0.2277 | 0.2307 | +0.0030 | 0.6115 / 0.3135 | 0.6165 / 0.3140 |
| 15 | **0.2538** | **0.2530** | **-0.0008** | 0.6439 / 0.3353 | 0.6435 / 0.3351 |

H1 epoch 15 的关键诊断为：overlap `0.381`、planned label precision `0.635`、
actual label precision `0.525`、actual-only label precision `0.357`、entropy `0.992`、
bit-use `1.000`。语义门槛中后两项未通过。

## 3. 数据完整度与使用规则

| 阶段 | 当前保存的结果粒度 | 推荐用途 |
|---|---|---|
| C0-C3 | epoch 10 的 mAP@100、P@5、R@100 | 第一轮横向定位 |
| D0-D2 | 最佳 mAP@100；D0 有坍缩指标 | memory schedule 因果判断，不做精细曲线比较 |
| E0-E1 | 每 5 epoch 的完整 mAP@100 曲线和选中点 P/R | 当前主线结论 |
| F0-G1 | 每 5 epoch 的完整 mAP@100 曲线和选中点 P/R | candidate 消融结论 |
| H0-H1 | epoch 5/10/15 检索指标；H1 epoch 7-15 trace | trace 中性及语义门槛判断 |
| I0 | 运行中，尚未进入结果表 | 完成后再归档，不能引用预热期日志作为结论 |

`best.pth` 由训练代码按全库 mAP 选择，而论文主报指标是 mAP@100。因此 E0 和 F0
等实验可能出现 `best.pth` 对应点不是 mAP@100 曲线峰值的情况；汇报时必须同时保留
“checkpoint 选择口径”和“曲线峰值”。

无效 H1 预跑目录 `h1_trace_only_from_epoch7_15ep/s5vh_ucf_32b_20260720_125731`
由于 trace 被 planner warmup 误门控，在 epoch 7 step 20 停止，只用于问题追踪，不得
纳入实验比较。有效 H1 目录的时间戳为 `20260720_131804`。

## 4. 正在运行的 I0

I0 是只读诊断实验，固定 trace budget 20，同时统计 top-5/10/20 actual precision 和
actual-only precision，并按 planner score 高低分层。截至本页整理时已完成 epoch 2，
trace 从 epoch 7 才开始，因此当前没有可报告的 I0 诊断结论。

```text
output: /mnt/disk2/yql/RF-CLaTH_outputs/rf_clath_ucf_i0_trace_depth_score/i0_trace_depth_score_15ep/s5vh_ucf_32b_20260720_140550
log: /mnt/disk2/yql/RF-CLaTH_run_logs/rf_clath_ucf_i0_trace_depth_score_cuda3_20260720_140548.queue.log
```
