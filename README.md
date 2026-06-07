# RF-CLaTH: Retrieval-Feedback Content-Lateral Temporal Hashing

RF-CLaTH is a clean implementation of the center-free, prototype-free UCF
mainline for self-supervised video hashing on the repartitioned S5VH protocol.

## Method

The model uses pre-extracted frame features as input:

- keyframe selector: `segment_rerank_gumbel_topk`
- slow branch: `SelectedClassAttentionEncoder` over selected semantic frames
- fast branch: `BidirectionalMambaEncoder` over temporal context
- fusion: `content_time_lateral`
- hash head: soft hash code projection with binarization for retrieval

The training objective is:

```text
L = 0.3  * L_view
  + 0.5  * L_batch_neighbor
  + 0.04 * L_memory_neighbor
  + 0.02 * L_quant
  + 0.03 * L_balance
```

`L_batch_neighbor` uses static raw-feature nearest neighbors as retrieval
feedback. `L_memory_neighbor` uses the online hash memory bank. The final method
does not use hash centers, prototype alignment, or reconstruction.

## Data

The active config targets UCF101 under:

```text
/mnt/disk2/yql/dataset_rePartition/ucf
```

Expected files:

```text
ucf_train_feats.h5
ucf_test_feats.h5
repartition_s5vh_ucf_train_label.pt
repartition_s5vh_ucf_q_label.pt
repartition_s5vh_ucf_re_label.pt
```

Convert labels if needed:

```bash
python tools/prepare_repartition_labels.py \
  --source-root /mnt/disk2/yql/dataset_rePartition \
  --output-dir cache
```

## Train

```bash
python train.py \
  --config configs/rf_clath_ucf.yaml \
  --dataset s5vh_ucf \
  --hash-bits 16 \
  --device cuda:0
```

Remote disk2 entry:

```bash
ssh exp-server 'cd /mnt/disk2/yql/RF-CLaTH && BITS="16 32 64 128" tools/run_rf_clath_ucf_disk2.sh <gpu>'
```

## Evaluate

```bash
python tools/compute_s5vh_official_map.py \
  --config configs/rf_clath_ucf.yaml \
  --dataset s5vh_ucf \
  --checkpoint outputs/rf_clath_ucf/best.pth
```

Evaluation uses `{-1, +1}` binary codes, `mAP@K` for K in
`5, 20, 40, 60, 80, 100`, and `P/R@K` for K in
`5, 10, 20, 40, 60, 80, 100`.

## Smoke Test

```bash
python train.py --demo
```

## Project Layout

- `configs/rf_clath_ucf.yaml`: final UCF method config
- `models/`: RF-CLaTH model, selector, encoders, fusion, hash head
- `losses/`: view, neighbor-feedback, quantization, and balance objectives
- `datasets/`: S5VH repartition feature dataset loaders
- `engine/`: training, evaluation, and hash extraction loops
- `tools/`: label conversion, official-style metrics, remote run script
