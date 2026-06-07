import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.video_dataset import build_dataloaders
from engine.evaluate import evaluate_retrieval
from losses import RFClathLoss
from models import RetrievalFeedbackContentLateralTemporalHashing
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import save_config
from utils.cuda import configure_cuda_attention
from utils.logger import setup_logger
from utils.metrics import compute_self_supervised_hash_metrics
from utils.neighbor import NeighborBatchSampler, estimate_neighbor_label_precision, load_or_build_neighbors
from utils.seed import set_seed


def build_optimizer(model: nn.Module, criterion: nn.Module, cfg: Dict):
    train_cfg = cfg.get("train", {})
    params = list(model.parameters()) + list(criterion.parameters())
    if train_cfg.get("optimizer", "adamw").lower() != "adamw":
        raise ValueError("Only AdamW is implemented.")
    return torch.optim.AdamW(
        params,
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )


def build_scheduler(optimizer, cfg: Dict):
    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 200))
    warmup = int(train_cfg.get("warmup_epochs", 5))

    def lr_lambda(epoch: int):
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        denom = max(1, epochs - warmup)
        progress = min(1.0, max(0.0, (epoch - warmup) / denom))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def format_retrieval_metrics(metrics: Dict[str, float], topk, map_topk=None) -> str:
    parts = [f"mAP={metrics.get('mAP', 0.0):.4f}"]
    for k in map_topk or []:
        key = f"mAP@{k}"
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    if "GmAP" in metrics:
        parts.append(f"GmAP={metrics['GmAP']:.4f}")
    for k in topk:
        p_key = f"Precision@{k}"
        r_key = f"Recall@{k}"
        if p_key in metrics:
            parts.append(f"P@{k}={metrics[p_key]:.4f}")
        if r_key in metrics:
            parts.append(f"R@{k}={metrics[r_key]:.4f}")
    return " ".join(parts)


def train_one_epoch(
    model,
    criterion,
    dataloader,
    optimizer,
    device: torch.device,
    epoch: int,
    cfg: Dict,
    logger,
    neighbor_indices: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    model.train()
    criterion.train()
    if hasattr(getattr(dataloader, "batch_sampler", None), "set_epoch"):
        dataloader.batch_sampler.set_epoch(epoch)
    train_cfg = cfg.get("train", {})
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    log_interval = int(train_cfg.get("log_interval", 20))
    max_steps = int(train_cfg.get("max_steps_per_epoch", 0))

    totals = {}
    metric_totals = {}
    metric_count = 0
    actual_steps = 0
    start = time.time()
    for step, batch in enumerate(dataloader, start=1):
        actual_steps = step
        batch_indices_cpu = batch["index"].long()
        video = batch["video"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(video)
            outputs["sample_indices"] = batch_indices_cpu.to(device, non_blocking=True)
            outputs["epoch"] = torch.tensor(epoch, device=device)
            if neighbor_indices is not None:
                outputs["neighbor_indices"] = neighbor_indices[batch_indices_cpu].to(device, non_blocking=True)
            losses = criterion(outputs)
            loss = losses["loss"]
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.parameters()), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())

        should_log = step % log_interval == 0 or step == len(dataloader) or (max_steps > 0 and step == max_steps)
        if should_log:
            hash_metrics = compute_self_supervised_hash_metrics(outputs)
            for key, value in hash_metrics.items():
                metric_totals[key] = metric_totals.get(key, 0.0) + value
            metric_count += 1
            logger.info(
                "epoch=%d step=%d/%d loss=%.4f view=%.4f semantic=%.4f hash=%.4f "
                "view_raw=%.4f batch_neigh=%.4f mem_neigh=%.4f quant=%.4f bit_bal=%.4f "
                "agree=%.3f hamm=%.3f entropy=%.3f bit_use=%.3f sat=%.3f mask=%.3f fast_cos=%.3f",
                epoch,
                step,
                len(dataloader),
                float(loss.detach().cpu()),
                float(losses["loss_view"].detach().cpu()),
                float(losses["loss_semantic"].detach().cpu()),
                float(losses["loss_hash"].detach().cpu()),
                float(losses["component_view_contrast"].detach().cpu()),
                float(losses["component_batch_neighbor"].detach().cpu()),
                float(losses["component_memory_neighbor"].detach().cpu()),
                float(losses["component_quant"].detach().cpu()),
                float(losses["component_bit_balance"].detach().cpu()),
                hash_metrics["metric_hash_agree"],
                hash_metrics["metric_hamm_norm"],
                hash_metrics["metric_bit_entropy"],
                hash_metrics["metric_bit_usage"],
                hash_metrics["metric_soft_saturation"],
                hash_metrics.get("metric_mask_ratio", 0.0),
                hash_metrics.get("metric_fast_cos", 0.0),
            )

        if max_steps > 0 and step >= max_steps:
            break

    count = max(1, actual_steps)
    averaged = {key: value / count for key, value in totals.items()}
    metric_divisor = max(1, metric_count)
    averaged.update({key: value / metric_divisor for key, value in metric_totals.items()})
    logger.info(
        "epoch=%d train_time=%.1fs loss=%.4f view=%.4f semantic=%.4f hash=%.4f "
        "view_raw=%.4f batch_neigh=%.4f mem_neigh=%.4f quant=%.4f bit_bal=%.4f "
        "agree=%.3f hamm=%.3f entropy=%.3f bit_use=%.3f balance_abs=%.3f sat=%.3f std=%.3f pos=%.3f mask=%.3f fast_cos=%.3f fused_cos=%.3f",
        epoch,
        time.time() - start,
        averaged.get("loss", 0.0),
        averaged.get("loss_view", 0.0),
        averaged.get("loss_semantic", 0.0),
        averaged.get("loss_hash", 0.0),
        averaged.get("component_view_contrast", 0.0),
        averaged.get("component_batch_neighbor", 0.0),
        averaged.get("component_memory_neighbor", 0.0),
        averaged.get("component_quant", 0.0),
        averaged.get("component_bit_balance", 0.0),
        averaged.get("metric_hash_agree", 0.0),
        averaged.get("metric_hamm_norm", 0.0),
        averaged.get("metric_bit_entropy", 0.0),
        averaged.get("metric_bit_usage", 0.0),
        averaged.get("metric_bit_balance_abs", 0.0),
        averaged.get("metric_soft_saturation", 0.0),
        averaged.get("metric_soft_std", 0.0),
        averaged.get("metric_positive_rate", 0.0),
        averaged.get("metric_mask_ratio", 0.0),
        averaged.get("metric_fast_cos", 0.0),
        averaged.get("metric_fused_cos", 0.0),
    )
    return averaged


def _resolve_output_dir(project_root: Path, cfg: Dict) -> Path:
    out = Path(cfg.get("project", {}).get("output_dir", "./outputs"))
    if not out.is_absolute():
        out = project_root / out
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_run_dir(output_root: Path, dataset_name: str, hash_bits: int, resume: str = "") -> Path:
    if resume:
        resume_parent = Path(resume).expanduser().resolve().parent
        resume_parent.mkdir(parents=True, exist_ok=True)
        return resume_parent
    run_name = f"{dataset_name}_{hash_bits}b_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train_rf_clath(
    cfg: Dict,
    project_root: Path,
    dataset_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict:
    seed = int(cfg.get("project", {}).get("seed", 3346))
    set_seed(seed)
    output_root = _resolve_output_dir(project_root, cfg)
    run_dataset = dataset_name or cfg["data"].get("name", "dataset")
    hash_bits = int(cfg.get("model", {}).get("hash_bits", 64))
    output_dir = _build_run_dir(output_root, run_dataset, hash_bits, cfg.get("train", {}).get("resume", ""))
    logger = setup_logger(log_file=str(output_dir / "train.log"))
    save_config(cfg, str(output_dir / "config.yaml"))

    use_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    configure_cuda_attention(use_device, logger=logger)
    logger.info(
        "device=%s dataset=%s hash_bits=%d output_dir=%s",
        use_device,
        run_dataset,
        hash_bits,
        output_dir,
    )

    train_loader, val_loader, retrieval_loader = build_dataloaders(
        cfg,
        dataset_name=dataset_name,
        project_root=project_root,
    )
    neighbor_indices = None
    neighbor_cfg = cfg.get("neighbor", {})
    if bool(neighbor_cfg.get("enabled", False)):
        topk_neighbors = int(neighbor_cfg.get("topk", 20))
        cache_path = str(neighbor_cfg.get("cache_path", "") or "")
        if bool(neighbor_cfg.get("cache", False)) and not cache_path:
            cache_path = str(output_dir / f"neighbors_train_top{topk_neighbors}.pt")
        elif cache_path:
            cache = Path(cache_path).expanduser()
            if not cache.is_absolute():
                cache_path = str((project_root / cache).resolve())
        build_on_cuda = bool(neighbor_cfg.get("build_on_cuda", True)) and use_device.type == "cuda"
        neighbor_device = use_device if build_on_cuda else torch.device("cpu")
        neighbor_indices = load_or_build_neighbors(
            train_loader.dataset,
            topk=topk_neighbors,
            feature_batch_size=int(neighbor_cfg.get("feature_batch_size", 256)),
            sim_chunk_size=int(neighbor_cfg.get("sim_chunk_size", 512)),
            device=neighbor_device,
            cache_path=cache_path,
            logger=logger,
        )
        precision = estimate_neighbor_label_precision(
            train_loader.dataset,
            neighbor_indices,
            max_items=int(neighbor_cfg.get("label_precision_probe", 2000)),
        )
        logger.info("neighbor_label_precision@%d=%.4f", topk_neighbors, precision)
        train_dataset = train_loader.dataset
        train_batch_sampler = NeighborBatchSampler(
            dataset_size=len(train_dataset),
            neighbor_indices=neighbor_indices,
            batch_size=int(cfg.get("train", {}).get("batch_size", 64)),
            neighbors_per_anchor=int(neighbor_cfg.get("neighbors_per_anchor", 1)),
            drop_last=True,
            seed=seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=int(cfg.get("data", {}).get("num_workers", 0)),
            pin_memory=bool(cfg.get("data", {}).get("pin_memory", True)),
        )
        logger.info(
            "neighbor_sampler batches=%d batch_size=%d neighbors_per_anchor=%d",
            len(train_loader),
            int(cfg.get("train", {}).get("batch_size", 64)),
            int(neighbor_cfg.get("neighbors_per_anchor", 1)),
        )
    model = RetrievalFeedbackContentLateralTemporalHashing(cfg).to(use_device)
    if "loss" in cfg and "memory_neighbor" in cfg["loss"]:
        cfg["loss"]["memory_neighbor"]["num_items"] = len(train_loader.dataset)
    criterion = RFClathLoss(cfg).to(use_device)
    save_config(cfg, str(output_dir / "config.yaml"))
    optimizer = build_optimizer(model, criterion, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    start_epoch = 1
    best_map = 0.0
    resume = cfg.get("train", {}).get("resume", "")
    if resume:
        state = load_checkpoint(resume, model, optimizer, scheduler, criterion, map_location=str(use_device))
        start_epoch = int(state.get("epoch", 0)) + 1
        best_map = float(state.get("best_metric", 0.0))
        logger.info("resumed checkpoint=%s start_epoch=%d best_mAP=%.4f", resume, start_epoch, best_map)

    epochs = int(cfg.get("train", {}).get("epochs", 200))
    eval_interval = int(cfg.get("train", {}).get("eval_interval", 5))
    save_interval = int(cfg.get("train", {}).get("save_interval", 5))
    early_stop_patience = int(cfg.get("train", {}).get("early_stop_patience", 0))
    early_stop_min_delta = float(cfg.get("train", {}).get("early_stop_min_delta", 0.0))
    topk = cfg.get("eval", {}).get("topk", [100, 200, 500, 1000])
    map_topk = cfg.get("eval", {}).get("map_topk", [])
    gmap_topk = cfg.get("eval", {}).get("gmap_topk", [])
    binary_format = cfg.get("eval", {}).get("binary_format", "pm1")
    no_improve_evals = 0
    last_epoch = start_epoch - 1
    run_until_epoch = min(epochs, int(cfg.get("train", {}).get("run_until_epoch", epochs)))
    if run_until_epoch < epochs:
        logger.info("run_until_epoch=%d total_epochs=%d", run_until_epoch, epochs)

    for epoch in range(start_epoch, run_until_epoch + 1):
        last_epoch = epoch
        train_stats = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            use_device,
            epoch,
            cfg,
            logger,
            neighbor_indices=neighbor_indices,
        )
        scheduler.step()

        if epoch % save_interval == 0:
            save_checkpoint(
                str(output_dir / f"epoch_{epoch:04d}.pth"),
                model,
                optimizer,
                scheduler,
                criterion,
                epoch=epoch,
                best_metric=best_map,
                cfg=cfg,
            )

        if epoch % eval_interval == 0:
            metrics = evaluate_retrieval(
                model,
                val_loader,
                retrieval_loader,
                device=use_device,
                topk=topk,
                binary_format=binary_format,
                map_topk=map_topk,
                gmap_topk=gmap_topk,
            )
            logger.info("epoch=%d eval %s", epoch, format_retrieval_metrics(metrics, topk, map_topk=map_topk))
            if metrics["mAP"] > best_map + early_stop_min_delta:
                best_map = metrics["mAP"]
                no_improve_evals = 0
                save_checkpoint(
                    str(output_dir / "best.pth"),
                    model,
                    optimizer,
                    scheduler,
                    criterion,
                    epoch=epoch,
                    best_metric=best_map,
                    cfg=cfg,
                )
            else:
                no_improve_evals += 1
                if early_stop_patience > 0 and no_improve_evals >= early_stop_patience:
                    logger.info(
                        "early_stop epoch=%d patience=%d best_mAP=%.4f",
                        epoch,
                        early_stop_patience,
                        best_map,
                    )
                    break

    save_checkpoint(
        str(output_dir / "last.pth"),
        model,
        optimizer,
        scheduler,
        criterion,
        epoch=last_epoch,
        best_metric=best_map,
        cfg=cfg,
    )
    return {"best_mAP": best_map}
