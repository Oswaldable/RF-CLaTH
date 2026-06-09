from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .contrastive import HashContrastiveLoss, NeighborHashContrastiveLoss
from .hash_losses import BalanceLoss, QuantizationLoss


def _get_nested(cfg: Dict, path: Tuple[str, ...], default=None):
    cur = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _get_float(cfg: Dict, paths: Iterable[Tuple[str, ...]], default: float) -> float:
    for path in paths:
        value = _get_nested(cfg, path)
        if value is not None:
            return float(value)
    return float(default)


class StaticARFLoss(nn.Module):
    """Static ARF graph-fitting objective.

    This loss replaces the old view/neighbor contrastive terms. It uses the
    Planner Graph soft target P_ij and fits current soft hash similarities to
    that target, while keeping quantization and bit-balance regularization.
    """

    requires_planner_memory = True

    def __init__(self, cfg: Dict):
        super().__init__()
        model_cfg = cfg.get("model", {})
        loss_cfg = cfg.get("loss", {})
        arf_cfg = cfg.get("arf_loss", {})
        weights_cfg = cfg.get("loss_weights", {})
        self.hash_bits = int(model_cfg.get("hash_bits", 64))
        self.gamma = float(arf_cfg.get("gamma", loss_cfg.get("gamma", 8.0)))
        self.lambda_arf = _get_float(
            cfg,
            (("loss_weights", "lambda_arf"), ("loss", "static_arf", "lambda"), ("loss", "lambda_arf")),
            1.0,
        )
        self.lambda_quant = float(
            weights_cfg.get(
                "lambda_quant",
                _get_float(loss_cfg, (("hash", "lambda_quant"), ("lambda_quant",)), 0.10),
            )
        )
        self.lambda_bit_balance = float(
            weights_cfg.get(
                "lambda_balance",
                weights_cfg.get(
                    "lambda_bit_balance",
                    _get_float(loss_cfg, (("hash", "lambda_bit_balance"), ("hash", "lambda_balance")), 0.05),
                ),
            )
        )
        self.quantization = QuantizationLoss()
        self.balance = BalanceLoss()

    def _logits(self, u: torch.Tensor, memory_u: torch.Tensor, gamma: float | None = None) -> torch.Tensor:
        bits = max(1, int(u.shape[-1]))
        use_gamma = self.gamma if gamma is None else float(gamma)
        return use_gamma * torch.einsum("bd,bsd->bs", u.float(), memory_u.float()) / float(bits)

    def _view_loss(
        self,
        u: torch.Tensor,
        memory_u: torch.Tensor,
        target_indices: torch.Tensor,
        target_scores: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        if target_indices.numel() == 0:
            return u.sum() * 0.0
        neighbor_u = memory_u[target_indices].detach()
        logits = self._logits(u, neighbor_u)
        bce = F.binary_cross_entropy_with_logits(logits, target_scores.float(), reduction="none")
        mask = target_mask.float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        per_anchor = (bce * mask).sum(dim=1) / denom
        valid_anchor = target_mask.any(dim=1).float()
        return (per_anchor * valid_anchor).sum() / valid_anchor.sum().clamp_min(1.0)

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = outputs["u_a"].device
        zero = torch.zeros((), device=device)
        memory = outputs.get("planner_memory", None)
        planner = outputs.get("graph_planner", None)
        sample_indices = outputs.get("sample_indices", None)
        if memory is None or planner is None or sample_indices is None:
            raise ValueError("StaticARFLoss requires planner_memory, graph_planner, and sample_indices in outputs.")

        targets = planner.static_arf_targets(memory, sample_indices)
        target_indices = targets["target_indices"].to(device)
        target_scores = targets["target_scores"].to(device)
        target_mask = targets["target_mask"].to(device)
        memory_u = memory.u_bank.to(device)

        loss_a = self._view_loss(outputs["u_a"], memory_u, target_indices, target_scores, target_mask)
        loss_b = self._view_loss(outputs["u_b"], memory_u, target_indices, target_scores, target_mask)
        component_arf_static = 0.5 * (loss_a + loss_b)

        component_quant = self.quantization(outputs["u_a"], outputs["u_b"])
        component_bit_balance = self.balance(outputs["u_a"], outputs["u_b"])

        loss_arf = self.lambda_arf * component_arf_static
        loss_hash = self.lambda_quant * component_quant + self.lambda_bit_balance * component_bit_balance
        total = loss_arf + loss_hash

        target_count = target_mask.float().sum(dim=1).mean() if target_mask.numel() > 0 else zero
        target_mean = target_scores[target_mask].mean() if target_mask.any() else zero

        return {
            "component_view_contrast": zero,
            "component_batch_neighbor": zero,
            "component_memory_neighbor": zero,
            "component_arf_static": component_arf_static,
            "component_quant": component_quant,
            "component_bit_balance": component_bit_balance,
            "loss_view": zero,
            "loss_semantic": loss_arf,
            "loss_arf": loss_arf,
            "loss_hash": loss_hash,
            "loss": total,
            "metric_arf_target_count": target_count,
            "metric_arf_target_mean": target_mean,
        }


class ARFLoss(StaticARFLoss):
    """Full ARF objective with actual retrieval trace feedback and P_z schedule."""

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.cfg = cfg
        train_cfg = cfg.get("train", {})
        planner_cfg = cfg.get("planner", {})
        warmup_cfg = planner_cfg.get("warmup", {})
        retrieval_cfg = cfg.get("retrieval_environment", {})
        feedback_cfg = cfg.get("feedback", {})
        arf_cfg = cfg.get("arf_loss", {})
        late_cfg = arf_cfg.get("late_sharpen", cfg.get("loss", {}).get("late_sharpen", {}))

        self.total_epochs = int(train_cfg.get("epochs", 150))
        self.graph_warmup_epochs = int(
            warmup_cfg.get("epochs", warmup_cfg.get("warmup_epochs", train_cfg.get("warmup_epochs", 0)))
        )

        self.main_omega_s = float(planner_cfg.get("omega_s", 0.45))
        self.main_omega_t = float(planner_cfg.get("omega_t", 0.25))
        self.main_omega_z = float(planner_cfg.get("omega_z", 0.30))
        self.warmup_omega_s = float(warmup_cfg.get("omega_s", 0.65))
        self.warmup_omega_t = float(warmup_cfg.get("omega_t", 0.35))
        self.warmup_omega_z = float(warmup_cfg.get("omega_z", 0.0))

        self.use_actual_trace = bool(retrieval_cfg.get("use_actual_trace", True))
        self.top_r = int(retrieval_cfg.get("top_r", planner_cfg.get("top_m", 20)))
        self.random_anchors = int(retrieval_cfg.get("random_anchors", planner_cfg.get("random_anchors", 40)))

        self.eta_missed_start = float(feedback_cfg.get("eta_missed_start", 0.0))
        self.eta_false_start = float(feedback_cfg.get("eta_false_start", 0.0))
        self.eta_missed_final = float(
            feedback_cfg.get("eta_missed_final", feedback_cfg.get("eta_missed", 1.0))
        )
        self.eta_false_final = float(feedback_cfg.get("eta_false_final", feedback_cfg.get("eta_false", 1.0)))
        self.feedback_ramp_epochs = int(feedback_cfg.get("ramp_epochs", 10))
        self.weight_clip = float(feedback_cfg.get("weight_clip", 3.0))

        self.late_start_ratio = float(late_cfg.get("start_ratio", 1.1))
        self.late_lambda_quant = float(late_cfg.get("lambda_quant", self.lambda_quant))
        self.late_lambda_balance = float(late_cfg.get("lambda_balance", self.lambda_bit_balance))
        self.late_gamma = float(late_cfg.get("gamma", self.gamma))

    def _schedule(self, epoch: int) -> Dict[str, float | bool]:
        if self.graph_warmup_epochs > 0 and epoch <= self.graph_warmup_epochs:
            omega_s = self.warmup_omega_s
            omega_t = self.warmup_omega_t
            omega_z = self.warmup_omega_z
            eta_missed = 0.0
            eta_false = 0.0
            use_actual_trace = False
        else:
            omega_s = self.main_omega_s
            omega_t = self.main_omega_t
            omega_z = self.main_omega_z
            if self.feedback_ramp_epochs > 0:
                ramp_epoch = max(0, epoch - self.graph_warmup_epochs)
                ramp = min(1.0, float(ramp_epoch) / float(self.feedback_ramp_epochs))
            else:
                ramp = 1.0
            eta_missed = self.eta_missed_start + (self.eta_missed_final - self.eta_missed_start) * ramp
            eta_false = self.eta_false_start + (self.eta_false_final - self.eta_false_start) * ramp
            use_actual_trace = self.use_actual_trace

        lambda_quant = self.lambda_quant
        lambda_balance = self.lambda_bit_balance
        gamma = self.gamma
        if self.total_epochs > 0 and float(epoch) / float(self.total_epochs) >= self.late_start_ratio:
            lambda_quant = self.late_lambda_quant
            lambda_balance = self.late_lambda_balance
            gamma = self.late_gamma

        return {
            "omega_s": omega_s,
            "omega_t": omega_t,
            "omega_z": omega_z,
            "eta_missed": eta_missed,
            "eta_false": eta_false,
            "use_actual_trace": use_actual_trace,
            "lambda_quant": lambda_quant,
            "lambda_balance": lambda_balance,
            "gamma": gamma,
        }

    def _weighted_view_loss(
        self,
        u: torch.Tensor,
        memory_u: torch.Tensor,
        target_indices: torch.Tensor,
        target_scores: torch.Tensor,
        target_mask: torch.Tensor,
        target_weights: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        if target_indices.numel() == 0:
            return u.sum() * 0.0
        neighbor_u = memory_u[target_indices].detach()
        logits = self._logits(u, neighbor_u, gamma=gamma)
        bce = F.binary_cross_entropy_with_logits(logits, target_scores.float(), reduction="none")
        mask = target_mask.float()
        weights = target_weights.float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        per_anchor = (bce * weights * mask).sum(dim=1) / denom
        valid_anchor = target_mask.any(dim=1).float()
        return (per_anchor * valid_anchor).sum() / valid_anchor.sum().clamp_min(1.0)

    def _trace_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        view_key: str,
        memory,
        planner,
        sample_indices: torch.Tensor,
        schedule: Dict[str, float | bool],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = outputs[view_key].device
        targets = planner.arf_trace_targets(
            memory,
            sample_indices,
            outputs[view_key],
            top_r=self.top_r,
            random_anchors=self.random_anchors,
            use_actual_trace=bool(schedule["use_actual_trace"]),
            omega_s=float(schedule["omega_s"]),
            omega_t=float(schedule["omega_t"]),
            omega_z=float(schedule["omega_z"]),
            eta_missed=float(schedule["eta_missed"]),
            eta_false=float(schedule["eta_false"]),
            weight_clip=self.weight_clip,
        )
        loss = self._weighted_view_loss(
            outputs[view_key],
            memory.u_bank.to(device),
            targets["target_indices"].to(device),
            targets["target_scores"].to(device),
            targets["target_mask"].to(device),
            targets["target_weights"].to(device),
            gamma=float(schedule["gamma"]),
        )
        return loss, targets

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = outputs["u_a"].device
        zero = torch.zeros((), device=device)
        memory = outputs.get("planner_memory", None)
        planner = outputs.get("graph_planner", None)
        sample_indices = outputs.get("sample_indices", None)
        if memory is None or planner is None or sample_indices is None:
            raise ValueError("ARFLoss requires planner_memory, graph_planner, and sample_indices in outputs.")

        epoch_tensor = outputs.get("epoch", torch.ones((), device=device))
        epoch = int(epoch_tensor.detach().cpu().item()) if torch.is_tensor(epoch_tensor) else int(epoch_tensor)
        schedule = self._schedule(epoch)

        loss_a, targets_a = self._trace_loss(outputs, "u_a", memory, planner, sample_indices, schedule)
        loss_b, targets_b = self._trace_loss(outputs, "u_b", memory, planner, sample_indices, schedule)
        component_arf = 0.5 * (loss_a + loss_b)

        component_quant = self.quantization(outputs["u_a"], outputs["u_b"])
        component_bit_balance = self.balance(outputs["u_a"], outputs["u_b"])

        loss_arf = self.lambda_arf * component_arf
        loss_hash = (
            float(schedule["lambda_quant"]) * component_quant
            + float(schedule["lambda_balance"]) * component_bit_balance
        )
        total = loss_arf + loss_hash

        masks = [targets_a["target_mask"].to(device), targets_b["target_mask"].to(device)]
        scores = [targets_a["target_scores"].to(device), targets_b["target_scores"].to(device)]
        target_count = 0.5 * sum(mask.float().sum(dim=1).mean() if mask.numel() > 0 else zero for mask in masks)
        target_means = [score[mask].mean() for score, mask in zip(scores, masks) if mask.any()]
        target_mean = sum(target_means) / len(target_means) if target_means else zero

        def avg_metric(key: str) -> torch.Tensor:
            a = targets_a[key].to(device) if torch.is_tensor(targets_a[key]) else torch.tensor(targets_a[key], device=device)
            b = targets_b[key].to(device) if torch.is_tensor(targets_b[key]) else torch.tensor(targets_b[key], device=device)
            return 0.5 * (a.float() + b.float())

        return {
            "component_view_contrast": zero,
            "component_batch_neighbor": zero,
            "component_memory_neighbor": zero,
            "component_arf_static": component_arf,
            "component_quant": component_quant,
            "component_bit_balance": component_bit_balance,
            "loss_view": zero,
            "loss_semantic": loss_arf,
            "loss_arf": loss_arf,
            "loss_hash": loss_hash,
            "loss": total,
            "metric_arf_target_count": target_count,
            "metric_arf_target_mean": target_mean,
            "metric_arf_actual_overlap": avg_metric("metric_actual_overlap"),
            "metric_arf_false_ratio": avg_metric("metric_false_ratio"),
            "metric_arf_missed_ratio": avg_metric("metric_missed_ratio"),
            "metric_arf_retrieved_target_mean": avg_metric("metric_retrieved_target_mean"),
            "metric_arf_feedback_weight_mean": avg_metric("metric_feedback_weight_mean"),
            "metric_arf_eta_missed": torch.tensor(float(schedule["eta_missed"]), device=device),
            "metric_arf_eta_false": torch.tensor(float(schedule["eta_false"]), device=device),
            "metric_arf_omega_z": torch.tensor(float(schedule["omega_z"]), device=device),
            "metric_arf_gamma": torch.tensor(float(schedule["gamma"]), device=device),
            "metric_arf_lambda_quant": torch.tensor(float(schedule["lambda_quant"]), device=device),
        }


class HybridARFLoss(ARFLoss):
    """Stage1 contrastive objective with ARF replacing memory-neighbor supervision."""

    requires_planner_memory = True

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        loss_cfg = cfg.get("loss", {})
        temperature = float(loss_cfg.get("temperature", 0.2))
        self.lambda_view = _get_float(
            loss_cfg,
            (("view", "lambda"), ("consistency", "lambda"), ("lambda_consistency",), ("lambda_hash_con",)),
            0.3,
        )
        self.lambda_batch_neighbor = _get_float(
            loss_cfg,
            (("semantic", "lambda_batch_neighbor"), ("lambda_neighbor_con",)),
            0.5,
        )
        self.hash_contrast = HashContrastiveLoss(temperature=temperature)
        self.neighbor_hash_contrast = NeighborHashContrastiveLoss(
            temperature=float(loss_cfg.get("neighbor_temperature", temperature)),
            symmetric_neighbors=bool(loss_cfg.get("symmetric_neighbors", True)),
        )

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = super().forward(outputs)
        device = outputs["u_a"].device
        zero = torch.zeros((), device=device)

        component_view = zero
        if self.lambda_view > 0:
            component_view = self.hash_contrast(outputs["u_a"], outputs["u_b"])

        component_batch_neighbor = zero
        has_neighbors = "sample_indices" in outputs and "neighbor_indices" in outputs
        if self.lambda_batch_neighbor > 0 and has_neighbors:
            component_batch_neighbor = self.neighbor_hash_contrast(
                outputs["u_a"],
                outputs["u_b"],
                outputs["sample_indices"],
                outputs["neighbor_indices"],
            )

        loss_view = self.lambda_view * component_view
        loss_batch_neighbor = self.lambda_batch_neighbor * component_batch_neighbor
        loss_arf = losses["loss_arf"]
        loss_semantic = loss_batch_neighbor + loss_arf
        total = loss_view + loss_semantic + losses["loss_hash"]

        return {
            **losses,
            "component_view_contrast": component_view,
            "component_batch_neighbor": component_batch_neighbor,
            "component_memory_neighbor": zero,
            "loss_view": loss_view,
            "loss_semantic": loss_semantic,
            "loss_arf": loss_arf,
            "loss": total,
        }
