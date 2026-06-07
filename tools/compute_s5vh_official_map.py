import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.video_dataset import build_dataloader
from engine.extract_hash import extract_hash_codes
from models import RetrievalFeedbackContentLateralTemporalHashing
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.cuda import configure_cuda_attention
from utils.metrics import compute_retrieval_metrics, hamming_distance


def s5vh_official_map(
    query_codes: torch.Tensor,
    retrieval_codes: torch.Tensor,
    query_labels: torch.Tensor,
    retrieval_labels: torch.Tensor,
    topk: int,
) -> float:
    """S5VH official top-K mAP convention.

    S5VH's utils.tools.mAP accumulates precision at relevant ranks in the
    top-K list and divides by K, not by the number of relevant items.
    """
    dist = hamming_distance(query_codes, retrieval_codes, binary_format="pm1")
    order = dist.argsort(dim=1)[:, :topk]
    rel = (query_labels.float() @ retrieval_labels.float().t()) > 0
    rel_sorted = torch.gather(rel.to(query_codes.device), 1, order).float()
    ranks = torch.arange(1, rel_sorted.shape[1] + 1, device=rel_sorted.device).float()
    precision_at_rank = rel_sorted.cumsum(dim=1) / ranks
    ap = (precision_at_rank * rel_sorted).sum(dim=1) / float(topk)
    return float(ap.mean().cpu())


def clath_legacy_topk_metrics(
    query_codes: torch.Tensor,
    retrieval_codes: torch.Tensor,
    query_labels: torch.Tensor,
    retrieval_labels: torch.Tensor,
    topk: list[int],
) -> dict[str, float]:
    return compute_retrieval_metrics(
        query_codes,
        retrieval_codes,
        query_labels,
        retrieval_labels,
        topk=topk,
        binary_format="pm1",
        map_topk=[k for k in topk if k != 10],
        gmap_topk=[],
    )


def main():
    parser = argparse.ArgumentParser(description="Compute CLaTH/S5VH legacy top-K retrieval metrics for an RF-CLaTH checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="s5vh_ucf")
    parser.add_argument("--device", default=None)
    parser.add_argument("--topk", nargs="+", type=int, default=[5, 10, 20, 40, 60, 80, 100])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    configure_cuda_attention(device)
    model = RetrievalFeedbackContentLateralTemporalHashing(cfg).to(device)
    # Retrieval evaluation only needs model weights. Loading the loss module can
    # break across old checkpoints when auxiliary loss heads are added later.
    load_checkpoint(args.checkpoint, model, map_location=str(device))

    query_loader = build_dataloader(cfg, "test", dataset_name=args.dataset, shuffle=False, project_root=PROJECT_ROOT)
    retrieval_loader = build_dataloader(
        cfg,
        "retrieval",
        dataset_name=args.dataset,
        shuffle=False,
        project_root=PROJECT_ROOT,
    )
    query = extract_hash_codes(model, query_loader, device=device, binary_format="pm1")
    retrieval = extract_hash_codes(model, retrieval_loader, device=device, binary_format="pm1")

    metrics = clath_legacy_topk_metrics(
        query["binary_code"],
        retrieval["binary_code"],
        query["label"],
        retrieval["label"],
        topk=args.topk,
    )
    for k in args.topk:
        map_key = f"mAP@{k}"
        if map_key in metrics:
            print(f"{map_key}: {metrics[map_key]:.6f}")
        print(f"P@{k}: {metrics[f'Precision@{k}']:.6f}")
        print(f"R@{k}: {metrics[f'Recall@{k}']:.6f}")


if __name__ == "__main__":
    main()
