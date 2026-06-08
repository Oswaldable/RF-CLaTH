import argparse
from pathlib import Path

import torch

from datasets.video_dataset import build_dataloader
from engine.evaluate import evaluate_retrieval
from losses import RFClathLoss
from models import RetrievalFeedbackContentLateralTemporalHashing
from utils.checkpoint import load_checkpoint
from utils.config import apply_overrides, load_config
from utils.cuda import configure_cuda_attention


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Evaluate RF-CLaTH retrieval metrics.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset",
        choices=["activitynet", "fcvid", "hmdb", "s5vh_activitynet", "s5vh_fcv", "s5vh_hmdb", "s5vh_ucf"],
        default=None,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--binary-format", choices=["pm1", "01"], default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    dataset_name = args.dataset or cfg["data"].get("name")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    configure_cuda_attention(device)
    model = RetrievalFeedbackContentLateralTemporalHashing(cfg).to(device)
    criterion = RFClathLoss(cfg).to(device)
    load_checkpoint(args.checkpoint, model, criterion=criterion, map_location=str(device))

    query_loader = build_dataloader(cfg, "test", dataset_name=dataset_name, shuffle=False, project_root=PROJECT_ROOT)
    retrieval_loader = build_dataloader(
        cfg, "retrieval", dataset_name=dataset_name, shuffle=False, project_root=PROJECT_ROOT
    )
    binary_format = args.binary_format or cfg.get("eval", {}).get("binary_format", "pm1")
    eval_cfg = cfg.get("eval", {})
    metrics = evaluate_retrieval(
        model,
        query_loader,
        retrieval_loader,
        device=device,
        topk=eval_cfg.get("topk", [100, 200, 500, 1000]),
        binary_format=binary_format,
        map_topk=eval_cfg.get("map_topk", []),
        gmap_topk=eval_cfg.get("gmap_topk", []),
    )
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
