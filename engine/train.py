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
from losses import (
    ARFLoss,
    AgenticUnifiedContrastiveLoss,
    ContrastiveARFLoss,
    HybridARFLoss,
    RFClathLoss,
    Stage1WarmupAgenticUnifiedLoss,
    StaticARFLoss,
)
from memory import PlannerMemoryBank, build_label_bank
from models import RetrievalFeedbackContentLateralTemporalHashing
from planner import RetrievalGraphPlanner
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


def build_criterion(cfg: Dict) -> nn.Module:
    objective = str(
        cfg.get("training", {}).get(
            "objective",
            cfg.get("train", {}).get("objective", cfg.get("loss", {}).get("type", "rf_clath")),
        )
    ).lower()
    if objective in {"static_arf", "arf_static"}:
        return StaticARFLoss(cfg)
    if objective in {"arf", "full_arf", "trace_arf"}:
        return ARFLoss(cfg)
    if objective in {"hybrid_arf", "arf_hybrid"}:
        return HybridARFLoss(cfg)
    if objective in {"arf_memory_contrastive", "contrastive_arf", "hybrid_contrastive_arf"}:
        return ContrastiveARFLoss(cfg)
    if objective in {"agentic_unified_contrastive", "agentic_contrastive", "unified_agentic_contrastive"}:
        return AgenticUnifiedContrastiveLoss(cfg)
    if objective in {"stage1_warmup_agentic_unified", "stage1_then_agentic_unified"}:
        return Stage1WarmupAgenticUnifiedLoss(cfg)
    return RFClathLoss(cfg)


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
    planner_memory: Optional[PlannerMemoryBank] = None,
    graph_planner: Optional[RetrievalGraphPlanner] = None,
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
    planner_log_interval = int(cfg.get("planner", {}).get("log_interval", log_interval))
    max_steps = int(train_cfg.get("max_steps_per_epoch", 0))

    totals = {}
    metric_totals = {}
    metric_count = 0
    planner_totals = {}
    planner_count = 0
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
            if planner_memory is not None and graph_planner is not None:
                with torch.no_grad():
                    planner_memory.update_batch(
                        batch_indices_cpu,
                        video.detach(),
                        outputs["selected_indices"].detach(),
                        outputs["z_a"].detach(),
                        outputs["z_b"].detach(),
                        epoch=epoch,
                        u_a=outputs["u_a"].detach(),
                        u_b=outputs["u_b"].detach(),
                    )
                outputs["planner_memory"] = planner_memory
                outputs["graph_planner"] = graph_planner
            losses = criterion(outputs)
            loss = losses["loss"]
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.parameters()), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        planner_metrics = None
        if planner_memory is not None and graph_planner is not None:
            with torch.no_grad():
                planner_should_log = (
                    step % planner_log_interval == 0
                    or step == len(dataloader)
                    or (max_steps > 0 and step == max_steps)
                )
                if planner_should_log:
                    planner_metrics = graph_planner.compute_sanity(planner_memory, batch_indices_cpu)
                    for key, value in planner_metrics.items():
                        planner_totals[key] = planner_totals.get(key, 0.0) + float(value)
                    planner_count += 1
                    logger.info(
                        "epoch=%d step=%d/%d planner_sanity valid=%.3f z_valid=%.3f "
                        "p_s_topm=%.4f p_t_topm=%.4f p_z_topm=%.4f p_final_topm=%.4f "
                        "p_random=%.4f p_final_std=%.4f overlap_s_t=%.3f "
                        "overlap_final_s=%.3f overlap_final_t=%.3f label_prec=%.3f",
                        epoch,
                        step,
                        len(dataloader),
                        planner_metrics.get("planner_valid_final", 0.0),
                        planner_metrics.get("planner_valid_z", 0.0),
                        planner_metrics.get("planner_p_s_topm", 0.0),
                        planner_metrics.get("planner_p_t_topm", 0.0),
                        planner_metrics.get("planner_p_z_topm", 0.0),
                        planner_metrics.get("planner_p_final_topm", 0.0),
                        planner_metrics.get("planner_p_random", 0.0),
                        planner_metrics.get("planner_p_final_std", 0.0),
                        planner_metrics.get("planner_overlap_s_t", 0.0),
                        planner_metrics.get("planner_overlap_final_s", 0.0),
                        planner_metrics.get("planner_overlap_final_t", 0.0),
                        planner_metrics.get("planner_label_precision_topm", 0.0),
                    )

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
                "view_raw=%.4f batch_neigh=%.4f mem_neigh=%.4f arf_raw=%.4f "
                "quant=%.4f bit_bal=%.4f arf_targets=%.1f arf_target=%.3f arf_hpos=%.1f arf_hard=%.1f "
                "arf_overlap=%.3f arf_false=%.3f arf_missed=%.3f arf_retrieved=%.3f "
                "arf_weight=%.3f eta_m=%.3f eta_f=%.3f omega_z=%.3f arf_gamma=%.1f "
                "agentic_raw=%.4f agentic_pos_view=%.1f agentic_pos_batch=%.1f "
                "agentic_pos_memory=%.1f agentic_pos_arf=%.1f agentic_hpos=%.1f "
                "agentic_hneg=%.1f agentic_pos_weight=%.3f "
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
                float(losses.get("component_arf_static", torch.zeros((), device=device)).detach().cpu()),
                float(losses["component_quant"].detach().cpu()),
                float(losses["component_bit_balance"].detach().cpu()),
                float(losses.get("metric_arf_target_count", torch.zeros((), device=device)).detach().cpu()),
                float(losses.get("metric_arf_target_mean", torch.zeros((), device=device)).detach().cpu()),
                float(losses.get("metric_arf_hard_positive_count", torch.zeros((), device=device)).detach().cpu()),
                float(losses.get("metric_arf_hard_negative_count", torch.zeros((), device=device)).detach().cpu()),
                float(losses.get("metric_arf_actual_overlap", torch.zeros((), device=device)).detach().cpu()),
                float(losses.get("metric_arf_false_ratio", torch.zeros((), device=device)).detach().cpu()),
                float(losses.get("metric_arf_missed_ratio", torch.zeros((), device=device)).detach().cpu()),
                float(losses.get("metric_arf_retrieved_target_mean", torch.zeros((), device=device)).detach().cpu()),
                float(losses.get("metric_arf_feedback_weight_mean", torch.zeros((), device=device)).detach().cpu()),
                float(losses.get("metric_arf_eta_missed", torch.zeros((), device=device)).detach().cpu()),
                float(losses.get("metric_arf_eta_false", torch.zeros((), device=device)).detach().cpu()),
                float(losses.get("metric_arf_omega_z", torch.zeros((), device=device)).detach().cpu()),
                float(losses.get("metric_arf_gamma", torch.zeros((), device=device)).detach().cpu()),
                float(losses.get("metric_agentic_raw", torch.zeros((), device=device)).detach().cpu()),
                float(losses.get("metric_agentic_pos_view", torch.zeros((), device=device)).detach().cpu()),
                float(losses.get("metric_agentic_pos_batch", torch.zeros((), device=device)).detach().cpu()),
                float(losses.get("metric_agentic_pos_memory", torch.zeros((), device=device)).detach().cpu()),
                float(losses.get("metric_agentic_pos_arf", torch.zeros((), device=device)).detach().cpu()),
                float(losses.get("metric_agentic_hard_positive_count", torch.zeros((), device=device)).detach().cpu()),
                float(losses.get("metric_agentic_hard_negative_count", torch.zeros((), device=device)).detach().cpu()),
                float(losses.get("metric_agentic_positive_weight_mean", torch.zeros((), device=device)).detach().cpu()),
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
    if planner_count > 0:
        averaged.update({key: value / planner_count for key, value in planner_totals.items()})
        logger.info(
            "epoch=%d planner_sanity_avg valid=%.3f z_valid=%.3f "
            "p_s_topm=%.4f p_t_topm=%.4f p_z_topm=%.4f p_final_topm=%.4f "
            "p_random=%.4f p_final_std=%.4f overlap_s_t=%.3f "
            "overlap_final_s=%.3f overlap_final_t=%.3f label_prec=%.3f",
            epoch,
            averaged.get("planner_valid_final", 0.0),
            averaged.get("planner_valid_z", 0.0),
            averaged.get("planner_p_s_topm", 0.0),
            averaged.get("planner_p_t_topm", 0.0),
            averaged.get("planner_p_z_topm", 0.0),
            averaged.get("planner_p_final_topm", 0.0),
            averaged.get("planner_p_random", 0.0),
            averaged.get("planner_p_final_std", 0.0),
            averaged.get("planner_overlap_s_t", 0.0),
            averaged.get("planner_overlap_final_s", 0.0),
            averaged.get("planner_overlap_final_t", 0.0),
            averaged.get("planner_label_precision_topm", 0.0),
        )
    logger.info(
        "epoch=%d train_time=%.1fs loss=%.4f view=%.4f semantic=%.4f hash=%.4f "
        "view_raw=%.4f batch_neigh=%.4f mem_neigh=%.4f arf_raw=%.4f "
        "quant=%.4f bit_bal=%.4f arf_targets=%.1f arf_target=%.3f arf_hpos=%.1f arf_hard=%.1f "
        "arf_overlap=%.3f arf_false=%.3f arf_missed=%.3f arf_retrieved=%.3f "
        "arf_weight=%.3f eta_m=%.3f eta_f=%.3f omega_z=%.3f arf_gamma=%.1f "
        "agentic_raw=%.4f agentic_pos_view=%.1f agentic_pos_batch=%.1f "
        "agentic_pos_memory=%.1f agentic_pos_arf=%.1f agentic_hpos=%.1f "
        "agentic_hneg=%.1f agentic_pos_weight=%.3f "
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
        averaged.get("component_arf_static", 0.0),
        averaged.get("component_quant", 0.0),
        averaged.get("component_bit_balance", 0.0),
        averaged.get("metric_arf_target_count", 0.0),
        averaged.get("metric_arf_target_mean", 0.0),
        averaged.get("metric_arf_hard_positive_count", 0.0),
        averaged.get("metric_arf_hard_negative_count", 0.0),
        averaged.get("metric_arf_actual_overlap", 0.0),
        averaged.get("metric_arf_false_ratio", 0.0),
        averaged.get("metric_arf_missed_ratio", 0.0),
        averaged.get("metric_arf_retrieved_target_mean", 0.0),
        averaged.get("metric_arf_feedback_weight_mean", 0.0),
        averaged.get("metric_arf_eta_missed", 0.0),
        averaged.get("metric_arf_eta_false", 0.0),
        averaged.get("metric_arf_omega_z", 0.0),
        averaged.get("metric_arf_gamma", 0.0),
        averaged.get("metric_agentic_raw", 0.0),
        averaged.get("metric_agentic_pos_view", 0.0),
        averaged.get("metric_agentic_pos_batch", 0.0),
        averaged.get("metric_agentic_pos_memory", 0.0),
        averaged.get("metric_agentic_pos_arf", 0.0),
        averaged.get("metric_agentic_hard_positive_count", 0.0),
        averaged.get("metric_agentic_hard_negative_count", 0.0),
        averaged.get("metric_agentic_positive_weight_mean", 0.0),
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
    criterion = build_criterion(cfg).to(use_device)
    planner_memory = None
    graph_planner = None
    planner_cfg = cfg.get("planner", {})
    if bool(planner_cfg.get("enabled", False)):
        planner_device_cfg = str(planner_cfg.get("device", "auto")).lower()
        if planner_device_cfg == "cpu":
            planner_device = torch.device("cpu")
        else:
            planner_device = use_device
        labels = None
        if bool(planner_cfg.get("label_precision", True)):
            labels = build_label_bank(train_loader.dataset, planner_device)
        planner_memory = PlannerMemoryBank(
            num_items=len(train_loader.dataset),
            device=planner_device,
            raw_dim=int(cfg.get("model", {}).get("feature_dim", 0))
            if cfg.get("model", {}).get("input_type", "features") == "features"
            else 0,
            z_dim=int(cfg.get("model", {}).get("hidden_dim", 0)),
            hash_dim=int(cfg.get("model", {}).get("hash_bits", 0)),
            z_momentum=float(planner_cfg.get("z_momentum", 0.9)),
            u_momentum=float(planner_cfg.get("u_momentum", planner_cfg.get("z_momentum", 0.9))),
            labels=labels,
        )
        graph_planner = RetrievalGraphPlanner.from_config(planner_cfg)
        logger.info(
            "planner_graph enabled top_m=%d omega_s=%.3f omega_t=%.3f omega_z=%.3f "
            "random_anchors=%d bank_device=%s label_precision=%s",
            graph_planner.top_m,
            graph_planner.omega_s,
            graph_planner.omega_t,
            graph_planner.omega_z,
            graph_planner.random_anchors,
            planner_device,
            labels is not None,
        )
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
            planner_memory=planner_memory,
            graph_planner=graph_planner,
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
