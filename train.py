import argparse
from copy import deepcopy
from pathlib import Path

import torch

from engine.train import train_rf_clath
from losses import RFClathLoss
from models import RetrievalFeedbackContentLateralTemporalHashing
from utils.config import apply_overrides, load_config
from utils.cuda import configure_cuda_attention
from utils.seed import set_seed


PROJECT_ROOT = Path(__file__).resolve().parent


def build_argparser():
    parser = argparse.ArgumentParser(description="Train RF-CLaTH.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    parser.add_argument(
        "--dataset",
        choices=["activitynet", "fcvid", "s5vh_activitynet", "s5vh_fcv", "s5vh_hmdb", "s5vh_ucf"],
        default=None,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hash-bits", type=int, choices=[16, 32, 64, 128], default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--demo", action="store_true", help="Run the random feature demo instead of training.")
    parser.add_argument("--override", action="append", default=[], help="Dotted config override, e.g. model.hash_bits=32")
    return parser


def prepare_cfg(args):
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)
    if args.dataset:
        cfg["data"]["name"] = args.dataset
    if args.output_dir:
        cfg["project"]["output_dir"] = args.output_dir
    if args.resume:
        cfg["train"]["resume"] = args.resume
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.hash_bits is not None:
        cfg["model"]["hash_bits"] = args.hash_bits
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    return cfg


def run_demo(cfg):
    """Minimal runnable demo with random X in R^{4 x 25 x 768}."""
    demo_cfg = deepcopy(cfg)
    demo_cfg["model"]["input_type"] = "features"
    demo_cfg["model"]["feature_dim"] = 768
    demo_cfg["model"]["hidden_dim"] = 768
    demo_cfg["model"]["num_frames"] = 25
    demo_cfg["model"]["num_keyframes"] = 5
    demo_cfg["model"]["slow_encoder"]["depth"] = 1
    demo_cfg["model"]["fast_encoder"]["depth"] = 1
    demo_cfg["train"]["amp"] = False
    set_seed(int(demo_cfg.get("project", {}).get("seed", 3346)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_cuda_attention(device)
    model = RetrievalFeedbackContentLateralTemporalHashing(demo_cfg).to(device)
    criterion = RFClathLoss(demo_cfg).to(device)
    model.train()
    x = torch.randn(4, 25, 768, device=device)
    outputs = model(x)
    losses = criterion(outputs)

    print("Random demo input:", tuple(x.shape))
    for key in [
        "h_s",
        "h_f_a",
        "h_f_b",
        "z_a",
        "z_b",
        "u_a",
        "u_b",
        "selected_indices",
        "fast_mask_a",
        "fast_mask_b",
    ]:
        value = outputs[key]
        print(f"{key}: {tuple(value.shape)}")
    for key, value in losses.items():
        print(f"{key}: {float(value.detach().cpu()):.6f}")


def main():
    args = build_argparser().parse_args()
    cfg = prepare_cfg(args)
    if args.demo:
        run_demo(cfg)
        return
    train_rf_clath(cfg, project_root=PROJECT_ROOT, dataset_name=cfg["data"].get("name"), device=args.device)


if __name__ == "__main__":
    main()
