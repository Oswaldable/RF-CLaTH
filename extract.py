import argparse
from pathlib import Path

import torch

from datasets.video_dataset import build_dataloader
from engine.extract_hash import extract_hash_codes
from losses import RFClathLoss
from models import RetrievalFeedbackContentLateralTemporalHashing
from utils.checkpoint import load_checkpoint
from utils.config import apply_overrides, load_config
from utils.cuda import configure_cuda_attention


PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_output(path: str) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Extract RF-CLaTH hash codes.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset",
        choices=["activitynet", "fcvid", "hmdb", "s5vh_activitynet", "s5vh_fcv", "s5vh_hmdb", "s5vh_ucf"],
        default=None,
    )
    parser.add_argument("--split", choices=["train", "val", "test", "retrieval"], default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--binary-format", choices=["pm1", "01"], default=None)
    parser.add_argument("--output", default="outputs/hash_codes.pt")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    dataset_name = args.dataset or cfg["data"].get("name")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    configure_cuda_attention(device)
    model = RetrievalFeedbackContentLateralTemporalHashing(cfg).to(device)
    criterion = RFClathLoss(cfg).to(device)
    load_checkpoint(args.checkpoint, model, criterion=criterion, map_location=str(device))
    loader = build_dataloader(cfg, args.split, dataset_name=dataset_name, shuffle=False, project_root=PROJECT_ROOT)
    binary_format = args.binary_format or cfg.get("eval", {}).get("binary_format", "pm1")
    codes = extract_hash_codes(model, loader, device=device, binary_format=binary_format)
    out = _resolve_output(args.output)
    torch.save(codes, out)
    print(f"saved: {out}")
    print(f"num_videos: {len(codes['video_id'])}, hash_bits: {codes['binary_code'].shape[1]}")


if __name__ == "__main__":
    main()
