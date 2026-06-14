from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from copy import deepcopy

from .contrastive import (
    AgenticMemoryNeighborContrastiveLoss,
    HashContrastiveLoss,
    NeighborHashContrastiveLoss,
    SelfCalibratedMemoryNeighborContrastiveLoss,
    WeightedSemanticContrastiveLoss,
)
from .hash_losses import BalanceLoss, QuantizationLoss
from .total_loss import RFClathLoss


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


class ContrastiveARFLoss(HybridARFLoss):
    """Stage1 contrastive objective with ARF-sourced memory positives.

    ARF is used only as the neighbor source: planner top neighbors become
    memory-bank positives, actual retrieval misses remain in the denominator as
    hard negatives, and the loss keeps the InfoNCE competition pressure.
    """

    requires_planner_memory = True

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        loss_cfg = cfg.get("loss", {})
        contrast_cfg = cfg.get("arf_contrastive", loss_cfg.get("arf_contrastive", {}))
        memory_cfg = loss_cfg.get("memory_neighbor", {})
        default_temperature = float(loss_cfg.get("neighbor_temperature", loss_cfg.get("temperature", 0.2)))
        self.arf_temperature = float(contrast_cfg.get("temperature", default_temperature))
        self.arf_positive_topk = int(
            contrast_cfg.get(
                "positive_topk",
                memory_cfg.get("positives_per_anchor", min(10, int(cfg.get("planner", {}).get("top_m", 20)))),
            )
        )
        self.arf_positive_threshold = float(contrast_cfg.get("positive_threshold", 0.0))
        self.include_planned_actual_overlap = bool(contrast_cfg.get("include_planned_actual_overlap", True))
        self.include_missed_as_positive = bool(contrast_cfg.get("include_missed_as_positive", False))
        self.fallback_positive_topk = max(1, int(contrast_cfg.get("fallback_positive_topk", 1)))
        self.hard_positive_weight = max(1.0, float(contrast_cfg.get("hard_positive_weight", 1.0)))
        self.hard_negative_weight = max(1.0, float(contrast_cfg.get("hard_negative_weight", 1.0)))
        self.actual_trace_start_epoch = int(contrast_cfg.get("actual_trace_start_epoch", 0))
        self.hard_mining_start_epoch = int(contrast_cfg.get("hard_mining_start_epoch", 0))

    def _membership(
        self,
        target_indices: torch.Tensor,
        member_indices: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        if target_indices.numel() == 0 or member_indices.numel() == 0:
            return torch.zeros_like(target_mask)
        return (target_indices.unsqueeze(-1) == member_indices.unsqueeze(1)).any(dim=-1) & target_mask

    def _positive_negative_masks(
        self,
        targets: Dict[str, torch.Tensor],
        device: torch.device,
        hard_mining_enabled: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_indices = targets["target_indices"].to(device)
        target_scores = targets["target_scores"].to(device)
        target_mask = targets["target_mask"].to(device)
        planned_indices = targets["planned_indices"].to(device)
        actual_indices = targets["actual_indices"].to(device)

        positive_mask = torch.zeros_like(target_mask)
        planned_cols = min(target_indices.shape[1], planned_indices.shape[1])
        if planned_cols > 0:
            positive_cols = planned_cols if self.arf_positive_topk <= 0 else min(planned_cols, self.arf_positive_topk)
            planned_positive = target_mask[:, :positive_cols]
            if self.arf_positive_threshold > 0:
                planned_positive = planned_positive & (target_scores[:, :positive_cols] >= self.arf_positive_threshold)
            positive_mask[:, :positive_cols] = planned_positive

        in_planned = self._membership(target_indices, planned_indices, target_mask)
        in_actual = self._membership(target_indices, actual_indices, target_mask)
        if self.include_planned_actual_overlap:
            positive_mask = positive_mask | (in_planned & in_actual)

        missed_positive_mask = torch.zeros_like(target_mask)
        if hard_mining_enabled:
            missed_positive_mask = in_planned & (~in_actual) & target_mask
            if self.include_missed_as_positive:
                positive_mask = positive_mask | missed_positive_mask

        if planned_cols > 0:
            no_positive = ~positive_mask.any(dim=1)
            fallback_cols = min(planned_cols, self.fallback_positive_topk)
            fallback = torch.zeros_like(target_mask)
            fallback[:, :fallback_cols] = target_mask[:, :fallback_cols]
            positive_mask = positive_mask | (no_positive[:, None] & fallback)

        hard_negative_mask = torch.zeros_like(target_mask)
        if hard_mining_enabled:
            hard_negative_mask = in_actual & (~in_planned) & (~positive_mask)
        hard_positive_mask = missed_positive_mask & positive_mask
        return positive_mask & target_mask, hard_positive_mask & target_mask, hard_negative_mask & target_mask

    def _memory_info_nce(
        self,
        u: torch.Tensor,
        memory,
        sample_indices: torch.Tensor,
        target_indices: torch.Tensor,
        positive_mask: torch.Tensor,
        hard_positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
    ) -> torch.Tensor:
        if memory.u_bank is None or u.shape[0] == 0:
            return u.new_zeros(())

        device = u.device
        valid_indices = torch.nonzero(memory.u_valid, as_tuple=False).flatten().to(device)
        if valid_indices.numel() <= 1:
            return u.new_zeros(())

        sample_indices = sample_indices.to(device=device, dtype=torch.long)
        target_indices = target_indices.to(device=device, dtype=torch.long)
        positive_mask = positive_mask.to(device=device, dtype=torch.bool)
        hard_positive_mask = hard_positive_mask.to(device=device, dtype=torch.bool)
        hard_negative_mask = hard_negative_mask.to(device=device, dtype=torch.bool)

        memory_u = memory.u_bank.index_select(0, valid_indices.to(memory.u_bank.device))
        memory_u = F.normalize(memory_u.to(device=device, dtype=torch.float32), dim=-1)
        query = F.normalize(u.float(), dim=-1)
        logits = query @ memory_u.t()
        logits = logits / max(self.arf_temperature, 1e-6)

        self_mask = valid_indices.unsqueeze(0) == sample_indices.unsqueeze(1)
        positive_indices = target_indices.masked_fill(~positive_mask, -1)
        positive_cols = (positive_indices.unsqueeze(-1) == valid_indices.view(1, 1, -1)).any(dim=1)
        positive_cols = positive_cols & (~self_mask)
        hard_positive_indices = target_indices.masked_fill(~hard_positive_mask, -1)
        hard_positive_cols = (hard_positive_indices.unsqueeze(-1) == valid_indices.view(1, 1, -1)).any(dim=1)
        hard_positive_cols = hard_positive_cols & positive_cols

        hard_negative_indices = target_indices.masked_fill(~hard_negative_mask, -1)
        hard_negative_cols = (hard_negative_indices.unsqueeze(-1) == valid_indices.view(1, 1, -1)).any(dim=1)
        hard_negative_cols = hard_negative_cols & (~self_mask) & (~positive_cols)

        candidate_mask = ~self_mask
        valid_rows = positive_cols.any(dim=1) & candidate_mask.any(dim=1)
        if not bool(valid_rows.any().item()):
            return u.new_zeros(())

        mask_value = -1e4 if logits.dtype in {torch.float16, torch.bfloat16} else -1e9
        denom_logits = logits.masked_fill(~candidate_mask, mask_value)
        if self.hard_negative_weight > 1.0:
            denom_logits = denom_logits + hard_negative_cols.float() * math.log(self.hard_negative_weight)
        positive_logits = logits.masked_fill(~positive_cols, mask_value)
        if self.hard_positive_weight > 1.0:
            positive_logits = positive_logits + hard_positive_cols.float() * math.log(self.hard_positive_weight)
        loss = torch.logsumexp(denom_logits, dim=1) - torch.logsumexp(positive_logits, dim=1)
        return loss[valid_rows].mean().to(dtype=u.dtype)

    def _trace_contrastive_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        view_key: str,
        memory,
        planner,
        sample_indices: torch.Tensor,
        schedule: Dict[str, float | bool],
        hard_mining_enabled: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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
        positive_mask, hard_positive_mask, hard_negative_mask = self._positive_negative_masks(
            targets,
            device,
            hard_mining_enabled=hard_mining_enabled,
        )
        loss = self._memory_info_nce(
            outputs[view_key],
            memory,
            sample_indices,
            targets["target_indices"].to(device),
            positive_mask,
            hard_positive_mask,
            hard_negative_mask,
        )

        zero = torch.zeros((), device=device)
        scores = targets["target_scores"].to(device)
        positive_count = positive_mask.float().sum(dim=1).mean() if positive_mask.numel() > 0 else zero
        positive_mean = scores[positive_mask].mean() if positive_mask.any() else zero
        hard_positive_count = hard_positive_mask.float().sum(dim=1).mean() if hard_positive_mask.numel() > 0 else zero
        hard_count = hard_negative_mask.float().sum(dim=1).mean() if hard_negative_mask.numel() > 0 else zero
        metrics = {
            "positive_count": positive_count,
            "positive_mean": positive_mean,
            "hard_positive_count": hard_positive_count,
            "hard_negative_count": hard_count,
        }
        return loss, targets, metrics

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = outputs["u_a"].device
        zero = torch.zeros((), device=device)
        memory = outputs.get("planner_memory", None)
        planner = outputs.get("graph_planner", None)
        sample_indices = outputs.get("sample_indices", None)
        if memory is None or planner is None or sample_indices is None:
            raise ValueError("ContrastiveARFLoss requires planner_memory, graph_planner, and sample_indices in outputs.")

        epoch_tensor = outputs.get("epoch", torch.ones((), device=device))
        epoch = int(epoch_tensor.detach().cpu().item()) if torch.is_tensor(epoch_tensor) else int(epoch_tensor)
        schedule = self._schedule(epoch)
        if self.actual_trace_start_epoch > 0 and epoch < self.actual_trace_start_epoch:
            schedule = dict(schedule)
            schedule["use_actual_trace"] = False
            schedule["eta_missed"] = 0.0
            schedule["eta_false"] = 0.0
        hard_mining_enabled = bool(schedule["use_actual_trace"]) and (
            self.hard_mining_start_epoch <= 0 or epoch >= self.hard_mining_start_epoch
        )

        loss_a, targets_a, trace_metrics_a = self._trace_contrastive_loss(
            outputs,
            "u_a",
            memory,
            planner,
            sample_indices,
            schedule,
            hard_mining_enabled,
        )
        loss_b, targets_b, trace_metrics_b = self._trace_contrastive_loss(
            outputs,
            "u_b",
            memory,
            planner,
            sample_indices,
            schedule,
            hard_mining_enabled,
        )
        component_arf = 0.5 * (loss_a + loss_b)

        component_view = zero
        if self.lambda_view > 0:
            component_view = self.hash_contrast(outputs["u_a"], outputs["u_b"])

        component_batch_neighbor = zero
        has_neighbors = "neighbor_indices" in outputs
        if self.lambda_batch_neighbor > 0 and has_neighbors:
            component_batch_neighbor = self.neighbor_hash_contrast(
                outputs["u_a"],
                outputs["u_b"],
                outputs["sample_indices"],
                outputs["neighbor_indices"],
            )

        component_quant = self.quantization(outputs["u_a"], outputs["u_b"])
        component_bit_balance = self.balance(outputs["u_a"], outputs["u_b"])

        loss_view = self.lambda_view * component_view
        loss_arf = self.lambda_arf * component_arf
        loss_semantic = self.lambda_batch_neighbor * component_batch_neighbor + loss_arf
        loss_hash = (
            float(schedule["lambda_quant"]) * component_quant
            + float(schedule["lambda_balance"]) * component_bit_balance
        )
        total = loss_view + loss_semantic + loss_hash

        def avg_target_metric(key: str) -> torch.Tensor:
            a = targets_a[key].to(device) if torch.is_tensor(targets_a[key]) else torch.tensor(targets_a[key], device=device)
            b = targets_b[key].to(device) if torch.is_tensor(targets_b[key]) else torch.tensor(targets_b[key], device=device)
            return 0.5 * (a.float() + b.float())

        return {
            "component_view_contrast": component_view,
            "component_batch_neighbor": component_batch_neighbor,
            "component_memory_neighbor": zero,
            "component_arf_static": component_arf,
            "component_arf_contrastive": component_arf,
            "component_quant": component_quant,
            "component_bit_balance": component_bit_balance,
            "loss_view": loss_view,
            "loss_semantic": loss_semantic,
            "loss_arf": loss_arf,
            "loss_hash": loss_hash,
            "loss": total,
            "metric_arf_target_count": 0.5
            * (trace_metrics_a["positive_count"].float() + trace_metrics_b["positive_count"].float()),
            "metric_arf_target_mean": 0.5
            * (trace_metrics_a["positive_mean"].float() + trace_metrics_b["positive_mean"].float()),
            "metric_arf_hard_positive_count": 0.5
            * (trace_metrics_a["hard_positive_count"].float() + trace_metrics_b["hard_positive_count"].float()),
            "metric_arf_hard_negative_count": 0.5
            * (trace_metrics_a["hard_negative_count"].float() + trace_metrics_b["hard_negative_count"].float()),
            "metric_arf_actual_overlap": avg_target_metric("metric_actual_overlap"),
            "metric_arf_false_ratio": avg_target_metric("metric_false_ratio"),
            "metric_arf_missed_ratio": avg_target_metric("metric_missed_ratio"),
            "metric_arf_retrieved_target_mean": avg_target_metric("metric_retrieved_target_mean"),
            "metric_arf_feedback_weight_mean": avg_target_metric("metric_feedback_weight_mean"),
            "metric_arf_eta_missed": torch.tensor(float(schedule["eta_missed"]), device=device),
            "metric_arf_eta_false": torch.tensor(float(schedule["eta_false"]), device=device),
            "metric_arf_omega_z": torch.tensor(float(schedule["omega_z"]), device=device),
            "metric_arf_gamma": torch.tensor(float(schedule["gamma"]), device=device),
            "metric_arf_lambda_quant": torch.tensor(float(schedule["lambda_quant"]), device=device),
        }


class AgenticUnifiedContrastiveLossV2(ContrastiveARFLoss):
    """Source-factored AUCL.

    The public objective is one AUCL term, but view, batch-neighbor, and
    memory-agentic supervision keep the separate candidate spaces that made
    Stage1 stable.
    """

    requires_planner_memory = True

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        loss_cfg = cfg.get("loss", {})
        model_cfg = cfg.get("model", {})
        memory_cfg = loss_cfg.get("memory_neighbor", {})
        aucl_cfg = cfg.get(
            "agentic_contrastive_v2",
            loss_cfg.get("agentic_contrastive_v2", cfg.get("agentic_contrastive", {})),
        )
        temperature = float(loss_cfg.get("neighbor_temperature", loss_cfg.get("temperature", 0.2)))

        self.lambda_view = _get_float(
            loss_cfg,
            (("view", "lambda"), ("consistency", "lambda"), ("lambda_consistency",), ("lambda_hash_con",)),
            0.30,
        )
        self.lambda_batch_neighbor = _get_float(
            loss_cfg,
            (("semantic", "lambda_batch_neighbor"), ("lambda_neighbor_con",)),
            0.50,
        )
        self.lambda_memory_agentic = _get_float(
            loss_cfg,
            (("semantic", "lambda_memory_neighbor"), ("lambda_memory_neighbor",)),
            0.04,
        )

        self.stage1_warmup_epochs = int(aucl_cfg.get("stage1_warmup_epochs", 60))
        self.agentic_ramp_epochs = max(1, int(aucl_cfg.get("agentic_ramp_epochs", aucl_cfg.get("ramp_epochs", 20))))
        self.max_agentic_memory_mix = min(1.0, max(0.0, float(aucl_cfg.get("max_agentic_memory_mix", 0.25))))
        self.actual_trace_start_epoch = int(aucl_cfg.get("actual_trace_start_epoch", self.stage1_warmup_epochs + 1))
        self.hard_mining_start_epoch = int(aucl_cfg.get("hard_mining_start_epoch", self.stage1_warmup_epochs + self.agentic_ramp_epochs))
        self.memory_agentic_start_epoch = int(memory_cfg.get("start_epoch", 2))

        self.agentic_memory_contrast = AgenticMemoryNeighborContrastiveLoss(
            num_items=int(memory_cfg.get("num_items", 0)),
            hash_bits=int(model_cfg.get("hash_bits", 64)),
            temperature=float(memory_cfg.get("temperature", temperature)),
            momentum=float(memory_cfg.get("momentum", 0.9)),
            positives_per_anchor=int(memory_cfg.get("positives_per_anchor", 10)),
            include_self=bool(memory_cfg.get("include_self", False)),
            planned_topk=int(aucl_cfg.get("planned_positive_topk", aucl_cfg.get("arf_positive_topk", 5))),
            missed_topk=int(aucl_cfg.get("missed_positive_topk", 5)),
            raw_positive_weight=float(aucl_cfg.get("raw_positive_weight", 1.0)),
            planned_positive_weight=float(aucl_cfg.get("planned_positive_weight", 0.5)),
            missed_positive_weight=float(aucl_cfg.get("missed_positive_weight", 1.25)),
            hard_negative_weight=float(aucl_cfg.get("hard_negative_weight", 1.10)),
            max_positive_weight=float(aucl_cfg.get("max_positive_weight", 2.0)),
            normalize_sources=bool(aucl_cfg.get("normalize_sources", False)),
        )

    def _aucl_beta(self, epoch: int) -> float:
        if epoch <= self.stage1_warmup_epochs:
            return 0.0
        progress = min(1.0, max(0.0, float(epoch - self.stage1_warmup_epochs) / float(self.agentic_ramp_epochs)))
        return self.max_agentic_memory_mix * progress

    def _trace_targets_for_view(
        self,
        outputs: Dict[str, torch.Tensor],
        view_key: str,
        memory,
        planner,
        sample_indices: torch.Tensor,
        schedule: Dict[str, float | bool],
    ) -> Dict[str, torch.Tensor]:
        return planner.arf_trace_targets(
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

    def _empty_trace_targets(self, sample_indices: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
        batch_size = sample_indices.shape[0]
        empty_idx = torch.empty(batch_size, 0, dtype=torch.long, device=device)
        empty_val = torch.empty(batch_size, 0, dtype=torch.float32, device=device)
        empty_mask = torch.empty(batch_size, 0, dtype=torch.bool, device=device)
        zero = torch.zeros((), device=device)
        return {
            "target_indices": empty_idx,
            "target_scores": empty_val,
            "target_mask": empty_mask,
            "target_weights": empty_val,
            "planned_indices": empty_idx,
            "actual_indices": empty_idx,
            "metric_actual_overlap": zero,
            "metric_false_ratio": zero,
            "metric_missed_ratio": zero,
            "metric_retrieved_target_mean": zero,
            "metric_feedback_weight_mean": zero,
        }

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = outputs["u_a"].device
        zero = torch.zeros((), device=device)
        memory = outputs.get("planner_memory", None)
        planner = outputs.get("graph_planner", None)
        sample_indices = outputs.get("sample_indices", None)
        neighbor_indices = outputs.get("neighbor_indices", None)
        if memory is None or planner is None or sample_indices is None or neighbor_indices is None:
            raise ValueError(
                "AgenticUnifiedContrastiveLossV2 requires planner_memory, graph_planner, sample_indices, and neighbor_indices."
            )

        epoch_tensor = outputs.get("epoch", torch.ones((), device=device))
        epoch = int(epoch_tensor.detach().cpu().item()) if torch.is_tensor(epoch_tensor) else int(epoch_tensor)
        beta = self._aucl_beta(epoch)

        schedule = dict(self._schedule(epoch))
        if epoch < self.actual_trace_start_epoch:
            schedule["use_actual_trace"] = False
            schedule["eta_missed"] = 0.0
            schedule["eta_false"] = 0.0
        else:
            schedule["use_actual_trace"] = True
        hard_mining_enabled = epoch >= self.hard_mining_start_epoch and bool(schedule["use_actual_trace"])

        if trace_enabled:
            targets_a = self._trace_targets_for_view(outputs, "u_a", memory, planner, sample_indices, schedule)
            targets_b = self._trace_targets_for_view(outputs, "u_b", memory, planner, sample_indices, schedule)
        else:
            targets_a = self._empty_trace_targets(sample_indices, device)
            targets_b = self._empty_trace_targets(sample_indices, device)

        component_view = zero
        if self.lambda_view > 0:
            component_view = self.hash_contrast(outputs["u_a"], outputs["u_b"])

        component_batch_neighbor = zero
        if self.lambda_batch_neighbor > 0:
            component_batch_neighbor = self.neighbor_hash_contrast(
                outputs["u_a"],
                outputs["u_b"],
                sample_indices,
                neighbor_indices,
            )

        if self.lambda_memory_agentic > 0 and epoch >= self.memory_agentic_start_epoch:
            memory_metrics = self.agentic_memory_contrast.forward_agentic(
                outputs["u_a"],
                outputs["u_b"],
                sample_indices,
                neighbor_indices,
                targets_a,
                targets_b,
                beta=beta,
                hard_mining_enabled=hard_mining_enabled,
            )
        else:
            memory_metrics = self.agentic_memory_contrast._zero_result(outputs["u_a"])
        component_memory_agentic = memory_metrics["memory_agentic"]

        component_quant = self.quantization(outputs["u_a"], outputs["u_b"])
        component_bit_balance = self.balance(outputs["u_a"], outputs["u_b"])

        loss_view = self.lambda_view * component_view
        loss_batch = self.lambda_batch_neighbor * component_batch_neighbor
        loss_memory = self.lambda_memory_agentic * component_memory_agentic
        loss_aucl = loss_view + loss_batch + loss_memory
        loss_hash = (
            float(schedule["lambda_quant"]) * component_quant
            + float(schedule["lambda_balance"]) * component_bit_balance
        )
        total = loss_aucl + loss_hash

        def avg_target_metric(key: str) -> torch.Tensor:
            a = targets_a[key].to(device) if torch.is_tensor(targets_a[key]) else torch.tensor(targets_a[key], device=device)
            b = targets_b[key].to(device) if torch.is_tensor(targets_b[key]) else torch.tensor(targets_b[key], device=device)
            return 0.5 * (a.float() + b.float())

        return {
            "component_view_contrast": component_view,
            "component_batch_neighbor": component_batch_neighbor,
            "component_memory_neighbor": component_memory_agentic,
            "component_arf_static": zero,
            "component_arf_contrastive": zero,
            "component_agentic_contrastive": loss_aucl,
            "component_quant": component_quant,
            "component_bit_balance": component_bit_balance,
            "loss_view": loss_view,
            "loss_semantic": loss_aucl,
            "loss_arf": zero,
            "loss_hash": loss_hash,
            "loss": total,
            "metric_agentic_raw": loss_aucl.detach(),
            "metric_agentic_pos_view": torch.ones((), device=device),
            "metric_agentic_pos_batch": memory_metrics["pos_raw_count"] * 0.0,
            "metric_agentic_pos_memory": memory_metrics["pos_raw_count"],
            "metric_agentic_pos_arf": memory_metrics["pos_planned_count"],
            "metric_agentic_hard_positive_count": memory_metrics["pos_missed_count"],
            "metric_agentic_hard_negative_count": memory_metrics["hard_negative_count"],
            "metric_agentic_positive_weight_mean": memory_metrics["positive_weight_mean"],
            "metric_agentic_mix_alpha": torch.tensor(float(beta), device=device),
            "metric_stage1_keep_weight": torch.tensor(float(1.0 - beta), device=device),
            "metric_aucl_raw": loss_aucl.detach(),
            "metric_aucl_memory_raw": memory_metrics["memory_raw"].detach(),
            "metric_aucl_memory_feedback": memory_metrics["memory_feedback"].detach(),
            "metric_aucl_beta": torch.tensor(float(beta), device=device),
            "metric_aucl_pos_raw": memory_metrics["pos_raw_count"],
            "metric_aucl_pos_planned": memory_metrics["pos_planned_count"],
            "metric_aucl_pos_missed": memory_metrics["pos_missed_count"],
            "metric_aucl_hard_negative": memory_metrics["hard_negative_count"],
            "metric_arf_target_count": memory_metrics["pos_planned_count"],
            "metric_arf_target_mean": avg_target_metric("metric_retrieved_target_mean"),
            "metric_arf_hard_positive_count": memory_metrics["pos_missed_count"],
            "metric_arf_hard_negative_count": memory_metrics["hard_negative_count"],
            "metric_arf_actual_overlap": avg_target_metric("metric_actual_overlap"),
            "metric_arf_false_ratio": avg_target_metric("metric_false_ratio"),
            "metric_arf_missed_ratio": avg_target_metric("metric_missed_ratio"),
            "metric_arf_retrieved_target_mean": avg_target_metric("metric_retrieved_target_mean"),
            "metric_arf_feedback_weight_mean": memory_metrics["positive_weight_mean"],
            "metric_arf_eta_missed": torch.tensor(float(schedule["eta_missed"]), device=device),
            "metric_arf_eta_false": torch.tensor(float(schedule["eta_false"]), device=device),
            "metric_arf_omega_z": torch.tensor(float(schedule["omega_z"]), device=device),
            "metric_arf_gamma": torch.tensor(float(schedule["gamma"]), device=device),
            "metric_arf_lambda_quant": torch.tensor(float(schedule["lambda_quant"]), device=device),
        }


class MemorySelfCalibratedRFClathLoss(ContrastiveARFLoss):
    """Stage1 objective with only the memory channel replaced by self-calibration."""

    requires_planner_memory = True

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        loss_cfg = cfg.get("loss", {})
        model_cfg = cfg.get("model", {})
        memory_cfg = loss_cfg.get("memory_neighbor", {})
        sc_cfg = cfg.get("memory_self_calibrated", loss_cfg.get("memory_self_calibrated", {}))
        temperature = float(loss_cfg.get("neighbor_temperature", loss_cfg.get("temperature", 0.2)))

        self.lambda_view = _get_float(
            loss_cfg,
            (("view", "lambda"), ("consistency", "lambda"), ("lambda_consistency",), ("lambda_hash_con",)),
            0.30,
        )
        self.lambda_batch_neighbor = _get_float(
            loss_cfg,
            (("semantic", "lambda_batch_neighbor"), ("lambda_neighbor_con",)),
            0.50,
        )
        self.lambda_memory_self_calibrated = _get_float(
            loss_cfg,
            (("semantic", "lambda_memory_neighbor"), ("lambda_memory_neighbor",)),
            0.04,
        )
        self.memory_self_calibrated_start_epoch = int(memory_cfg.get("start_epoch", 2))
        self.actual_trace_start_epoch = int(sc_cfg.get("actual_trace_start_epoch", 61))
        self.hard_mining_start_epoch = int(sc_cfg.get("hard_mining_start_epoch", 80))
        self.self_calibrated_memory = SelfCalibratedMemoryNeighborContrastiveLoss(
            num_items=int(memory_cfg.get("num_items", 0)),
            hash_bits=int(model_cfg.get("hash_bits", 64)),
            temperature=float(memory_cfg.get("temperature", temperature)),
            momentum=float(memory_cfg.get("momentum", 0.9)),
            positives_per_anchor=int(memory_cfg.get("positives_per_anchor", 10)),
            include_self=bool(memory_cfg.get("include_self", False)),
            raw_positive_weight=float(sc_cfg.get("raw_positive_weight", 1.0)),
            planned_positive_weight=float(sc_cfg.get("planned_positive_weight", 0.5)),
            missed_positive_weight=float(sc_cfg.get("missed_positive_weight", 1.25)),
            hard_negative_weight=float(sc_cfg.get("hard_negative_weight", 1.10)),
            max_positive_weight=float(sc_cfg.get("max_positive_weight", 2.0)),
            trust_momentum=float(sc_cfg.get("trust_momentum", 0.9)),
            edge_momentum=float(sc_cfg.get("edge_momentum", 0.9)),
            edge_slots=int(sc_cfg.get("edge_slots", 64)),
            planned_topk=int(sc_cfg.get("planned_positive_topk", 5)),
            missed_topk=int(sc_cfg.get("missed_positive_topk", 5)),
            raw_trust_topk=int(sc_cfg.get("raw_trust_topk", cfg.get("neighbor", {}).get("topk", 20))),
            edge_min_value=float(sc_cfg.get("edge_min_value", 1e-4)),
        )

    def _trace_targets_for_view(
        self,
        outputs: Dict[str, torch.Tensor],
        view_key: str,
        memory,
        planner,
        sample_indices: torch.Tensor,
        schedule: Dict[str, float | bool],
    ) -> Dict[str, torch.Tensor]:
        return planner.arf_trace_targets(
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

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = outputs["u_a"].device
        zero = torch.zeros((), device=device)
        memory = outputs.get("planner_memory", None)
        planner = outputs.get("graph_planner", None)
        sample_indices = outputs.get("sample_indices", None)
        neighbor_indices = outputs.get("neighbor_indices", None)
        if memory is None or planner is None or sample_indices is None or neighbor_indices is None:
            raise ValueError(
                "MemorySelfCalibratedRFClathLoss requires planner_memory, graph_planner, sample_indices, and neighbor_indices."
            )

        epoch_tensor = outputs.get("epoch", torch.ones((), device=device))
        epoch = int(epoch_tensor.detach().cpu().item()) if torch.is_tensor(epoch_tensor) else int(epoch_tensor)
        schedule = dict(self._schedule(epoch))
        trace_enabled = epoch >= self.actual_trace_start_epoch
        if not trace_enabled:
            schedule["use_actual_trace"] = False
            schedule["eta_missed"] = 0.0
            schedule["eta_false"] = 0.0
        else:
            schedule["use_actual_trace"] = True
        hard_mining_enabled = trace_enabled and epoch >= self.hard_mining_start_epoch

        targets_a = self._trace_targets_for_view(outputs, "u_a", memory, planner, sample_indices, schedule)
        targets_b = self._trace_targets_for_view(outputs, "u_b", memory, planner, sample_indices, schedule)

        component_view = zero
        if self.lambda_view > 0:
            component_view = self.hash_contrast(outputs["u_a"], outputs["u_b"])

        component_batch_neighbor = zero
        if self.lambda_batch_neighbor > 0:
            component_batch_neighbor = self.neighbor_hash_contrast(
                outputs["u_a"],
                outputs["u_b"],
                sample_indices,
                neighbor_indices,
            )

        if self.lambda_memory_self_calibrated > 0 and epoch >= self.memory_self_calibrated_start_epoch:
            memory_metrics = self.self_calibrated_memory.forward_self_calibrated(
                outputs["u_a"],
                outputs["u_b"],
                sample_indices,
                neighbor_indices,
                targets_a,
                targets_b,
                trace_enabled=trace_enabled,
                hard_mining_enabled=hard_mining_enabled,
            )
        else:
            memory_metrics = self.self_calibrated_memory._zero_result(outputs["u_a"])

        component_memory = memory_metrics["memory_self_calibrated"]
        component_quant = self.quantization(outputs["u_a"], outputs["u_b"])
        component_bit_balance = self.balance(outputs["u_a"], outputs["u_b"])

        loss_view = self.lambda_view * component_view
        loss_semantic = (
            self.lambda_batch_neighbor * component_batch_neighbor
            + self.lambda_memory_self_calibrated * component_memory
        )
        loss_hash = (
            float(schedule["lambda_quant"]) * component_quant
            + float(schedule["lambda_balance"]) * component_bit_balance
        )
        total = loss_view + loss_semantic + loss_hash

        def avg_target_metric(key: str) -> torch.Tensor:
            a = targets_a[key].to(device) if torch.is_tensor(targets_a[key]) else torch.tensor(targets_a[key], device=device)
            b = targets_b[key].to(device) if torch.is_tensor(targets_b[key]) else torch.tensor(targets_b[key], device=device)
            return 0.5 * (a.float() + b.float())

        return {
            "component_view_contrast": component_view,
            "component_batch_neighbor": component_batch_neighbor,
            "component_memory_neighbor": component_memory,
            "component_arf_static": zero,
            "component_arf_contrastive": zero,
            "component_quant": component_quant,
            "component_bit_balance": component_bit_balance,
            "loss_view": loss_view,
            "loss_semantic": loss_semantic,
            "loss_arf": zero,
            "loss_hash": loss_hash,
            "loss": total,
            "metric_agentic_raw": component_memory.detach(),
            "metric_agentic_pos_memory": memory_metrics["pos_raw_count"],
            "metric_agentic_pos_arf": memory_metrics["pos_planned_count"],
            "metric_agentic_hard_positive_count": memory_metrics["pos_missed_count"],
            "metric_agentic_hard_negative_count": memory_metrics["hard_negative_count"],
            "metric_agentic_positive_weight_mean": memory_metrics["positive_weight_mean"],
            "metric_agentic_mix_alpha": torch.tensor(float(1.0 if trace_enabled else 0.0), device=device),
            "metric_stage1_keep_weight": torch.tensor(float(0.0 if trace_enabled else 1.0), device=device),
            "metric_aucl_raw": component_memory.detach(),
            "metric_aucl_memory_raw": memory_metrics["memory_raw"].detach(),
            "metric_aucl_memory_feedback": component_memory.detach(),
            "metric_aucl_beta": memory_metrics["trust_mean"],
            "metric_aucl_pos_raw": memory_metrics["pos_raw_count"],
            "metric_aucl_pos_planned": memory_metrics["pos_planned_count"],
            "metric_aucl_pos_missed": memory_metrics["pos_missed_count"],
            "metric_aucl_hard_negative": memory_metrics["hard_negative_count"],
            "metric_arf_target_count": memory_metrics["pos_planned_count"],
            "metric_arf_target_mean": avg_target_metric("metric_retrieved_target_mean"),
            "metric_arf_hard_positive_count": memory_metrics["pos_missed_count"],
            "metric_arf_hard_negative_count": memory_metrics["hard_negative_count"],
            "metric_arf_actual_overlap": avg_target_metric("metric_actual_overlap"),
            "metric_arf_false_ratio": avg_target_metric("metric_false_ratio"),
            "metric_arf_missed_ratio": avg_target_metric("metric_missed_ratio"),
            "metric_arf_retrieved_target_mean": avg_target_metric("metric_retrieved_target_mean"),
            "metric_arf_feedback_weight_mean": memory_metrics["positive_weight_mean"],
            "metric_arf_eta_missed": torch.tensor(float(schedule["eta_missed"]), device=device),
            "metric_arf_eta_false": torch.tensor(float(schedule["eta_false"]), device=device),
            "metric_arf_omega_z": torch.tensor(float(schedule["omega_z"]), device=device),
            "metric_arf_gamma": torch.tensor(float(schedule["gamma"]), device=device),
            "metric_selfcal_trust": memory_metrics["trust_mean"],
            "metric_selfcal_trust_active": memory_metrics["trust_active"],
            "metric_selfcal_trust_observation": memory_metrics["trust_observation"],
            "metric_selfcal_pos_persistence": memory_metrics["positive_persistence_mean"],
            "metric_selfcal_neg_persistence": memory_metrics["negative_persistence_mean"],
        }


class MergedSemanticSelfCalibratedRFClathLoss(MemorySelfCalibratedRFClathLoss):
    """Merged semantic objective with agentic feedback limited to memory loss."""

    requires_planner_memory = True

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        loss_cfg = cfg.get("loss", {})
        semantic_cfg = loss_cfg.get("semantic", {})
        temperature = float(loss_cfg.get("neighbor_temperature", loss_cfg.get("temperature", 0.2)))
        self.lambda_merged_semantic = float(
            semantic_cfg.get(
                "lambda_merged",
                semantic_cfg.get("lambda_semantic", self.lambda_view + self.lambda_batch_neighbor),
            )
        )
        self.merged_semantic_contrast = WeightedSemanticContrastiveLoss(
            temperature=temperature,
            symmetric_neighbors=bool(loss_cfg.get("symmetric_neighbors", True)),
            view_positive_weight=float(semantic_cfg.get("view_positive_weight", 1.2)),
            neighbor_positive_weight=float(semantic_cfg.get("neighbor_positive_weight", 1.0)),
            max_positive_weight=float(semantic_cfg.get("max_positive_weight", 2.0)),
        )

    def _empty_trace_targets(self, sample_indices: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
        batch_size = sample_indices.shape[0]
        empty_idx = torch.empty(batch_size, 0, dtype=torch.long, device=device)
        empty_val = torch.empty(batch_size, 0, dtype=torch.float32, device=device)
        empty_mask = torch.empty(batch_size, 0, dtype=torch.bool, device=device)
        zero = torch.zeros((), device=device)
        return {
            "target_indices": empty_idx,
            "target_scores": empty_val,
            "target_mask": empty_mask,
            "target_weights": empty_val,
            "planned_indices": empty_idx,
            "actual_indices": empty_idx,
            "metric_actual_overlap": zero,
            "metric_false_ratio": zero,
            "metric_missed_ratio": zero,
            "metric_retrieved_target_mean": zero,
            "metric_feedback_weight_mean": zero,
        }

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = outputs["u_a"].device
        zero = torch.zeros((), device=device)
        memory = outputs.get("planner_memory", None)
        planner = outputs.get("graph_planner", None)
        sample_indices = outputs.get("sample_indices", None)
        neighbor_indices = outputs.get("neighbor_indices", None)
        if memory is None or planner is None or sample_indices is None or neighbor_indices is None:
            raise ValueError(
                "MergedSemanticSelfCalibratedRFClathLoss requires planner_memory, "
                "graph_planner, sample_indices, and neighbor_indices."
            )

        epoch_tensor = outputs.get("epoch", torch.ones((), device=device))
        epoch = int(epoch_tensor.detach().cpu().item()) if torch.is_tensor(epoch_tensor) else int(epoch_tensor)
        schedule = dict(self._schedule(epoch))
        trace_enabled = epoch >= self.actual_trace_start_epoch
        if not trace_enabled:
            schedule["use_actual_trace"] = False
            schedule["eta_missed"] = 0.0
            schedule["eta_false"] = 0.0
            targets_a = self._empty_trace_targets(sample_indices, device)
            targets_b = self._empty_trace_targets(sample_indices, device)
        else:
            schedule["use_actual_trace"] = True
            targets_a = self._trace_targets_for_view(outputs, "u_a", memory, planner, sample_indices, schedule)
            targets_b = self._trace_targets_for_view(outputs, "u_b", memory, planner, sample_indices, schedule)
        hard_mining_enabled = trace_enabled and epoch >= self.hard_mining_start_epoch

        component_merged_semantic = zero
        component_view = zero
        if self.lambda_merged_semantic > 0:
            component_merged_semantic = self.merged_semantic_contrast(
                outputs["u_a"],
                outputs["u_b"],
                sample_indices,
                neighbor_indices,
            )
            if self.lambda_view > 0:
                component_view = self.hash_contrast(outputs["u_a"], outputs["u_b"])

        if self.lambda_memory_self_calibrated > 0 and epoch >= self.memory_self_calibrated_start_epoch:
            memory_metrics = self.self_calibrated_memory.forward_self_calibrated(
                outputs["u_a"],
                outputs["u_b"],
                sample_indices,
                neighbor_indices,
                targets_a,
                targets_b,
                trace_enabled=trace_enabled,
                hard_mining_enabled=hard_mining_enabled,
            )
        else:
            memory_metrics = self.self_calibrated_memory._zero_result(outputs["u_a"])
        component_memory = memory_metrics["memory_self_calibrated"]

        component_quant = self.quantization(outputs["u_a"], outputs["u_b"])
        component_bit_balance = self.balance(outputs["u_a"], outputs["u_b"])

        loss_semantic = self.lambda_merged_semantic * component_merged_semantic
        loss_semantic = loss_semantic + self.lambda_memory_self_calibrated * component_memory
        loss_hash = (
            float(schedule["lambda_quant"]) * component_quant
            + float(schedule["lambda_balance"]) * component_bit_balance
        )
        total = loss_semantic + loss_hash

        def avg_target_metric(key: str) -> torch.Tensor:
            a = targets_a[key].to(device) if torch.is_tensor(targets_a[key]) else torch.tensor(targets_a[key], device=device)
            b = targets_b[key].to(device) if torch.is_tensor(targets_b[key]) else torch.tensor(targets_b[key], device=device)
            return 0.5 * (a.float() + b.float())

        return {
            "component_view_contrast": component_view,
            "component_batch_neighbor": component_merged_semantic,
            "component_memory_neighbor": component_memory,
            "component_merged_semantic": component_merged_semantic,
            "component_arf_static": zero,
            "component_arf_contrastive": zero,
            "component_quant": component_quant,
            "component_bit_balance": component_bit_balance,
            "loss_view": zero,
            "loss_semantic": loss_semantic,
            "loss_arf": zero,
            "loss_hash": loss_hash,
            "loss": total,
            "metric_agentic_raw": component_memory.detach(),
            "metric_agentic_pos_view": torch.ones((), device=device),
            "metric_agentic_pos_batch": torch.ones((), device=device),
            "metric_agentic_pos_memory": memory_metrics["pos_raw_count"],
            "metric_agentic_pos_arf": memory_metrics["pos_planned_count"],
            "metric_agentic_hard_positive_count": memory_metrics["pos_missed_count"],
            "metric_agentic_hard_negative_count": memory_metrics["hard_negative_count"],
            "metric_agentic_positive_weight_mean": memory_metrics["positive_weight_mean"],
            "metric_agentic_mix_alpha": torch.tensor(float(1.0 if trace_enabled else 0.0), device=device),
            "metric_stage1_keep_weight": torch.tensor(float(0.0 if trace_enabled else 1.0), device=device),
            "metric_aucl_raw": component_memory.detach(),
            "metric_aucl_memory_raw": memory_metrics["memory_raw"].detach(),
            "metric_aucl_memory_feedback": component_memory.detach(),
            "metric_aucl_beta": memory_metrics["trust_mean"],
            "metric_aucl_pos_raw": memory_metrics["pos_raw_count"],
            "metric_aucl_pos_planned": memory_metrics["pos_planned_count"],
            "metric_aucl_pos_missed": memory_metrics["pos_missed_count"],
            "metric_aucl_hard_negative": memory_metrics["hard_negative_count"],
            "metric_arf_target_count": memory_metrics["pos_planned_count"],
            "metric_arf_target_mean": avg_target_metric("metric_retrieved_target_mean"),
            "metric_arf_hard_positive_count": memory_metrics["pos_missed_count"],
            "metric_arf_hard_negative_count": memory_metrics["hard_negative_count"],
            "metric_arf_actual_overlap": avg_target_metric("metric_actual_overlap"),
            "metric_arf_false_ratio": avg_target_metric("metric_false_ratio"),
            "metric_arf_missed_ratio": avg_target_metric("metric_missed_ratio"),
            "metric_arf_retrieved_target_mean": avg_target_metric("metric_retrieved_target_mean"),
            "metric_arf_feedback_weight_mean": memory_metrics["positive_weight_mean"],
            "metric_arf_eta_missed": torch.tensor(float(schedule["eta_missed"]), device=device),
            "metric_arf_eta_false": torch.tensor(float(schedule["eta_false"]), device=device),
            "metric_arf_omega_z": torch.tensor(float(schedule["omega_z"]), device=device),
            "metric_arf_gamma": torch.tensor(float(schedule["gamma"]), device=device),
            "metric_selfcal_trust": memory_metrics["trust_mean"],
            "metric_selfcal_trust_active": memory_metrics["trust_active"],
            "metric_selfcal_trust_observation": memory_metrics["trust_observation"],
            "metric_selfcal_pos_persistence": memory_metrics["positive_persistence_mean"],
            "metric_selfcal_neg_persistence": memory_metrics["negative_persistence_mean"],
        }


class AgenticUnifiedContrastiveLoss(ContrastiveARFLoss):
    """Single source-aware InfoNCE objective for Stage1 + ARF feedback.

    The loss uses one candidate pool for both current-batch embeddings and the
    planner memory bank. Different supervision signals only contribute source
    weights to the same positive matrix; ARF false retrievals add denominator
    weight as hard negatives.
    """

    requires_planner_memory = True

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        loss_cfg = cfg.get("loss", {})
        agentic_cfg = cfg.get("agentic_contrastive", loss_cfg.get("agentic_contrastive", {}))
        source_cfg = agentic_cfg.get("source_weights", {})
        memory_cfg = loss_cfg.get("memory_neighbor", {})
        default_temperature = float(loss_cfg.get("neighbor_temperature", loss_cfg.get("temperature", 0.2)))

        self.agentic_temperature = float(agentic_cfg.get("temperature", default_temperature))
        self.memory_positive_topk = max(
            1,
            int(agentic_cfg.get("memory_positive_topk", memory_cfg.get("positives_per_anchor", 10))),
        )
        self.arf_positive_topk = int(agentic_cfg.get("arf_positive_topk", self.arf_positive_topk))
        self.actual_trace_start_epoch = int(agentic_cfg.get("actual_trace_start_epoch", 30))
        self.hard_mining_start_epoch = int(agentic_cfg.get("hard_mining_start_epoch", 30))
        self.normalize_sources = bool(agentic_cfg.get("normalize_sources", True))
        self.max_positive_weight = float(agentic_cfg.get("max_positive_weight", 2.0))
        self.source_weight_view = float(source_cfg.get("view", 1.0))
        self.source_weight_batch = float(source_cfg.get("batch_neighbor", 0.75))
        self.source_weight_memory = float(source_cfg.get("memory_neighbor", 0.25))
        self.source_weight_arf = float(source_cfg.get("arf_planned", 0.25))
        self.source_weight_missed_bonus = float(source_cfg.get("arf_missed_bonus", 0.25))
        self.hard_negative_weight = max(1.0, float(agentic_cfg.get("hard_negative_weight", 1.25)))

    def _source_addition(self, mask: torch.Tensor, weight: float) -> torch.Tensor:
        if weight <= 0 or mask.numel() == 0:
            return mask.float() * 0.0
        mask_f = mask.float()
        if self.normalize_sources:
            denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            return mask_f * (float(weight) / denom)
        return mask_f * float(weight)

    def _batch_neighbor_mask(
        self,
        sample_indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
        query_base: torch.Tensor,
        candidate_base: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = sample_indices.shape[0]
        if neighbor_indices.numel() == 0:
            return torch.zeros(
                query_base.shape[0],
                candidate_base.shape[0],
                dtype=torch.bool,
                device=sample_indices.device,
            )
        base = (neighbor_indices[:, :, None] == sample_indices[None, None, :]).any(dim=1)
        if bool(getattr(self.neighbor_hash_contrast.nt_xent, "symmetric_neighbors", True)):
            base = base | base.t()
        base = base & (~torch.eye(batch_size, dtype=torch.bool, device=sample_indices.device))
        return base[query_base[:, None], candidate_base[None, :]]

    def _memory_neighbor_mask(
        self,
        sample_indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
        query_base: torch.Tensor,
        valid_indices: torch.Tensor,
    ) -> torch.Tensor:
        if valid_indices.numel() == 0 or neighbor_indices.numel() == 0:
            return torch.zeros(query_base.shape[0], valid_indices.shape[0], dtype=torch.bool, device=sample_indices.device)
        topk = min(self.memory_positive_topk, neighbor_indices.shape[1])
        neighbors = neighbor_indices[:, :topk].long()
        base = (neighbors[:, :, None] == valid_indices[None, None, :]).any(dim=1)
        base = base & (valid_indices[None, :] != sample_indices[:, None])
        return base[query_base]

    def _target_columns(
        self,
        target_indices: torch.Tensor,
        target_mask: torch.Tensor,
        valid_indices: torch.Tensor,
    ) -> torch.Tensor:
        if valid_indices.numel() == 0 or target_indices.numel() == 0:
            return torch.zeros(target_indices.shape[0], valid_indices.shape[0], dtype=torch.bool, device=target_indices.device)
        masked = target_indices.long().masked_fill(~target_mask, -1)
        return (masked.unsqueeze(-1) == valid_indices.view(1, 1, -1)).any(dim=1)

    def _trace_masks_for_view(
        self,
        outputs: Dict[str, torch.Tensor],
        view_key: str,
        memory,
        planner,
        sample_indices: torch.Tensor,
        valid_indices: torch.Tensor,
        schedule: Dict[str, float | bool],
        hard_mining_enabled: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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
        positive_mask, hard_positive_mask, hard_negative_mask = self._positive_negative_masks(
            targets,
            device,
            hard_mining_enabled=hard_mining_enabled,
        )
        target_indices = targets["target_indices"].to(device)
        arf_cols = self._target_columns(target_indices, positive_mask.to(device), valid_indices)
        hard_pos_cols = self._target_columns(target_indices, hard_positive_mask.to(device), valid_indices)
        hard_neg_cols = self._target_columns(target_indices, hard_negative_mask.to(device), valid_indices)
        zero = torch.zeros((), device=device)
        target_scores = targets["target_scores"].to(device)
        metrics = {
            "positive_count": positive_mask.float().sum(dim=1).mean() if positive_mask.numel() > 0 else zero,
            "positive_mean": target_scores[positive_mask.to(device)].mean()
            if positive_mask.numel() > 0 and positive_mask.to(device).any()
            else zero,
            "hard_positive_count": hard_positive_mask.float().sum(dim=1).mean() if hard_positive_mask.numel() > 0 else zero,
            "hard_negative_count": hard_negative_mask.float().sum(dim=1).mean() if hard_negative_mask.numel() > 0 else zero,
        }
        return arf_cols, hard_pos_cols, hard_neg_cols, targets, metrics

    def _unified_info_nce(
        self,
        outputs: Dict[str, torch.Tensor],
        memory,
        planner,
        sample_indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
        schedule: Dict[str, float | bool],
        hard_mining_enabled: bool,
        source_weights: Dict[str, float] | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = outputs["u_a"].device
        batch_size = outputs["u_a"].shape[0]
        query = F.normalize(torch.cat([outputs["u_a"], outputs["u_b"]], dim=0).float(), dim=-1)
        query_count = query.shape[0]

        valid_indices_mem = torch.nonzero(memory.u_valid, as_tuple=False).flatten()
        if valid_indices_mem.numel() > 0:
            memory_u = memory.u_bank.index_select(0, valid_indices_mem)
            memory_u = F.normalize(memory_u.to(device=device, dtype=torch.float32), dim=-1)
            valid_indices = valid_indices_mem.to(device=device, dtype=torch.long)
        else:
            memory_u = query.new_empty(0, query.shape[-1])
            valid_indices = torch.empty(0, dtype=torch.long, device=device)

        batch_logits = query @ query.t()
        memory_logits = query @ memory_u.t() if valid_indices.numel() > 0 else query.new_empty(query_count, 0)
        logits = torch.cat([batch_logits, memory_logits], dim=1) / max(self.agentic_temperature, 1e-6)

        sample_indices = sample_indices.to(device=device, dtype=torch.long)
        neighbor_indices = neighbor_indices.to(device=device, dtype=torch.long)
        query_base = torch.arange(query_count, device=device) % batch_size
        candidate_base = torch.arange(query_count, device=device) % batch_size
        query_sample_ids = sample_indices[query_base]

        batch_self_mask = torch.eye(query_count, dtype=torch.bool, device=device)
        batch_candidate_mask = ~batch_self_mask
        if valid_indices.numel() > 0:
            memory_self_mask = valid_indices.unsqueeze(0) == query_sample_ids.unsqueeze(1)
            memory_candidate_mask = ~memory_self_mask
        else:
            memory_candidate_mask = torch.zeros(query_count, 0, dtype=torch.bool, device=device)
        candidate_mask = torch.cat([batch_candidate_mask, memory_candidate_mask], dim=1)

        view_mask = sample_indices[query_base].unsqueeze(1) == sample_indices[candidate_base].unsqueeze(0)
        view_mask = view_mask & batch_candidate_mask
        batch_neighbor_mask = self._batch_neighbor_mask(sample_indices, neighbor_indices, query_base, candidate_base)
        batch_neighbor_mask = batch_neighbor_mask & batch_candidate_mask
        memory_neighbor_mask = self._memory_neighbor_mask(sample_indices, neighbor_indices, query_base, valid_indices)
        memory_neighbor_mask = memory_neighbor_mask & memory_candidate_mask

        arf_a, hpos_a, hneg_a, targets_a, metrics_a = self._trace_masks_for_view(
            outputs,
            "u_a",
            memory,
            planner,
            sample_indices,
            valid_indices,
            schedule,
            hard_mining_enabled,
        )
        arf_b, hpos_b, hneg_b, targets_b, metrics_b = self._trace_masks_for_view(
            outputs,
            "u_b",
            memory,
            planner,
            sample_indices,
            valid_indices,
            schedule,
            hard_mining_enabled,
        )
        arf_mask = torch.cat([arf_a, arf_b], dim=0) & memory_candidate_mask
        hard_positive_mask = torch.cat([hpos_a, hpos_b], dim=0) & memory_candidate_mask
        hard_negative_mask = torch.cat([hneg_a, hneg_b], dim=0) & memory_candidate_mask

        source_weights = source_weights or {}
        source_weight_view = float(source_weights.get("view", self.source_weight_view))
        source_weight_batch = float(source_weights.get("batch_neighbor", self.source_weight_batch))
        source_weight_memory = float(source_weights.get("memory_neighbor", self.source_weight_memory))
        source_weight_arf = float(source_weights.get("arf_planned", self.source_weight_arf))
        source_weight_missed_bonus = float(
            source_weights.get("arf_missed_bonus", self.source_weight_missed_bonus)
        )
        hard_negative_weight = max(1.0, float(source_weights.get("hard_negative_weight", self.hard_negative_weight)))

        positive_weights = torch.zeros_like(logits, dtype=torch.float32)
        positive_weights[:, :query_count] += self._source_addition(view_mask, source_weight_view)
        positive_weights[:, :query_count] += self._source_addition(batch_neighbor_mask, source_weight_batch)
        if valid_indices.numel() > 0:
            positive_weights[:, query_count:] += self._source_addition(memory_neighbor_mask, source_weight_memory)
            positive_weights[:, query_count:] += self._source_addition(arf_mask, source_weight_arf)
            positive_weights[:, query_count:] += self._source_addition(hard_positive_mask, source_weight_missed_bonus)
        positive_weights = torch.clamp(positive_weights, min=0.0, max=self.max_positive_weight)
        positive_weights = positive_weights * candidate_mask.float()

        valid_rows = (positive_weights > 0).any(dim=1) & candidate_mask.any(dim=1)
        if not bool(valid_rows.any().item()):
            return query.new_zeros(()), {
                "targets_a": targets_a,
                "targets_b": targets_b,
                "metrics_a": metrics_a,
                "metrics_b": metrics_b,
                "view_count": query.new_zeros(()),
                "batch_count": query.new_zeros(()),
                "memory_count": query.new_zeros(()),
                "arf_count": query.new_zeros(()),
                "hard_positive_count": query.new_zeros(()),
                "hard_negative_count": query.new_zeros(()),
                "positive_weight": query.new_zeros(()),
            }

        mask_value = -1e4 if logits.dtype in {torch.float16, torch.bfloat16} else -1e9
        denom_logits = logits.masked_fill(~candidate_mask, mask_value)
        if hard_negative_weight > 1.0 and valid_indices.numel() > 0:
            denom_bonus = torch.zeros_like(logits)
            denom_bonus[:, query_count:] = hard_negative_mask.float() * math.log(hard_negative_weight)
            denom_logits = denom_logits + denom_bonus
        positive_logits = logits + torch.log(positive_weights.clamp_min(1e-12))
        positive_logits = positive_logits.masked_fill(positive_weights <= 0, mask_value)
        loss = torch.logsumexp(denom_logits, dim=1) - torch.logsumexp(positive_logits, dim=1)

        pos_active = positive_weights > 0
        metrics = {
            "targets_a": targets_a,
            "targets_b": targets_b,
            "metrics_a": metrics_a,
            "metrics_b": metrics_b,
            "view_count": view_mask.float().sum(dim=1).mean(),
            "batch_count": batch_neighbor_mask.float().sum(dim=1).mean(),
            "memory_count": memory_neighbor_mask.float().sum(dim=1).mean()
            if valid_indices.numel() > 0 and source_weight_memory > 0
            else query.new_zeros(()),
            "arf_count": arf_mask.float().sum(dim=1).mean()
            if valid_indices.numel() > 0 and source_weight_arf > 0
            else query.new_zeros(()),
            "hard_positive_count": hard_positive_mask.float().sum(dim=1).mean()
            if valid_indices.numel() > 0 and source_weight_missed_bonus > 0
            else query.new_zeros(()),
            "hard_negative_count": hard_negative_mask.float().sum(dim=1).mean()
            if valid_indices.numel() > 0 and hard_negative_weight > 1.0
            else query.new_zeros(()),
            "positive_weight": positive_weights[pos_active].mean() if pos_active.any() else query.new_zeros(()),
        }
        return loss[valid_rows].mean().to(dtype=outputs["u_a"].dtype), metrics

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = outputs["u_a"].device
        zero = torch.zeros((), device=device)
        memory = outputs.get("planner_memory", None)
        planner = outputs.get("graph_planner", None)
        sample_indices = outputs.get("sample_indices", None)
        neighbor_indices = outputs.get("neighbor_indices", None)
        if memory is None or planner is None or sample_indices is None or neighbor_indices is None:
            raise ValueError(
                "AgenticUnifiedContrastiveLoss requires planner_memory, graph_planner, sample_indices, and neighbor_indices."
            )

        epoch_tensor = outputs.get("epoch", torch.ones((), device=device))
        epoch = int(epoch_tensor.detach().cpu().item()) if torch.is_tensor(epoch_tensor) else int(epoch_tensor)
        schedule = self._schedule(epoch)
        if self.actual_trace_start_epoch > 0 and epoch < self.actual_trace_start_epoch:
            schedule = dict(schedule)
            schedule["use_actual_trace"] = False
            schedule["eta_missed"] = 0.0
            schedule["eta_false"] = 0.0
        hard_mining_enabled = bool(schedule["use_actual_trace"]) and (
            self.hard_mining_start_epoch <= 0 or epoch >= self.hard_mining_start_epoch
        )

        component_agentic, metrics = self._unified_info_nce(
            outputs,
            memory,
            planner,
            sample_indices,
            neighbor_indices,
            schedule,
            hard_mining_enabled,
        )
        component_quant = self.quantization(outputs["u_a"], outputs["u_b"])
        component_bit_balance = self.balance(outputs["u_a"], outputs["u_b"])
        loss_hash = (
            float(schedule["lambda_quant"]) * component_quant
            + float(schedule["lambda_balance"]) * component_bit_balance
        )
        total = component_agentic + loss_hash

        targets_a = metrics["targets_a"]
        targets_b = metrics["targets_b"]

        def avg_target_metric(key: str) -> torch.Tensor:
            a = targets_a[key].to(device) if torch.is_tensor(targets_a[key]) else torch.tensor(targets_a[key], device=device)
            b = targets_b[key].to(device) if torch.is_tensor(targets_b[key]) else torch.tensor(targets_b[key], device=device)
            return 0.5 * (a.float() + b.float())

        return {
            "component_view_contrast": zero,
            "component_batch_neighbor": zero,
            "component_memory_neighbor": zero,
            "component_arf_static": zero,
            "component_arf_contrastive": zero,
            "component_agentic_contrastive": component_agentic,
            "component_quant": component_quant,
            "component_bit_balance": component_bit_balance,
            "loss_view": zero,
            "loss_semantic": component_agentic,
            "loss_arf": zero,
            "loss_hash": loss_hash,
            "loss": total,
            "metric_agentic_raw": component_agentic.detach(),
            "metric_agentic_pos_view": metrics["view_count"],
            "metric_agentic_pos_batch": metrics["batch_count"],
            "metric_agentic_pos_memory": metrics["memory_count"],
            "metric_agentic_pos_arf": metrics["arf_count"],
            "metric_agentic_hard_positive_count": metrics["hard_positive_count"],
            "metric_agentic_hard_negative_count": metrics["hard_negative_count"],
            "metric_agentic_positive_weight_mean": metrics["positive_weight"],
            "metric_arf_target_count": 0.5
            * (metrics["metrics_a"]["positive_count"].float() + metrics["metrics_b"]["positive_count"].float()),
            "metric_arf_target_mean": 0.5
            * (metrics["metrics_a"]["positive_mean"].float() + metrics["metrics_b"]["positive_mean"].float()),
            "metric_arf_hard_positive_count": metrics["hard_positive_count"],
            "metric_arf_hard_negative_count": metrics["hard_negative_count"],
            "metric_arf_actual_overlap": avg_target_metric("metric_actual_overlap"),
            "metric_arf_false_ratio": avg_target_metric("metric_false_ratio"),
            "metric_arf_missed_ratio": avg_target_metric("metric_missed_ratio"),
            "metric_arf_retrieved_target_mean": avg_target_metric("metric_retrieved_target_mean"),
            "metric_arf_feedback_weight_mean": avg_target_metric("metric_feedback_weight_mean"),
            "metric_arf_eta_missed": torch.tensor(float(schedule["eta_missed"]), device=device),
            "metric_arf_eta_false": torch.tensor(float(schedule["eta_false"]), device=device),
            "metric_arf_omega_z": torch.tensor(float(schedule["omega_z"]), device=device),
            "metric_arf_gamma": torch.tensor(float(schedule["gamma"]), device=device),
            "metric_arf_lambda_quant": torch.tensor(float(schedule["lambda_quant"]), device=device),
        }


class PhasedAgenticUnifiedContrastiveLoss(AgenticUnifiedContrastiveLoss):
    """Two-phase AUCL source schedule without falling back to the legacy Stage1 loss."""

    requires_planner_memory = True

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        agentic_cfg = cfg.get("agentic_contrastive", cfg.get("loss", {}).get("agentic_contrastive", {}))
        self.stage1_warmup_epochs = int(agentic_cfg.get("stage1_warmup_epochs", 30))
        self.schedule_mode = str(agentic_cfg.get("schedule_mode", agentic_cfg.get("switch_mode", "hard"))).lower()
        self.ramp_epochs = max(0, int(agentic_cfg.get("ramp_epochs", 0)))
        self.post_stage1_weight = float(
            agentic_cfg.get("post_stage1_weight", agentic_cfg.get("stage1_keep_weight", 0.0))
        )
        self.post_stage1_weight = min(1.0, max(0.0, self.post_stage1_weight))

    def _current_epoch(self, outputs: Dict[str, torch.Tensor]) -> int:
        epoch = outputs.get("epoch", None)
        if epoch is None:
            return 1
        if torch.is_tensor(epoch):
            return int(epoch.detach().cpu().item())
        return int(epoch)

    def _phase_weights(self, epoch: int) -> Tuple[float, float]:
        if epoch <= self.stage1_warmup_epochs:
            return 1.0, 0.0
        if self.schedule_mode == "ramp":
            if self.ramp_epochs <= 0:
                progress = 1.0
            else:
                progress = min(1.0, max(0.0, float(epoch - self.stage1_warmup_epochs) / float(self.ramp_epochs)))
            stage1_weight = 1.0 - progress * (1.0 - self.post_stage1_weight)
            return stage1_weight, 1.0 - stage1_weight
        if self.schedule_mode in {"hybrid", "retain_stage1", "retained_stage1"}:
            return self.post_stage1_weight, 1.0 - self.post_stage1_weight
        return self.post_stage1_weight, 1.0 - self.post_stage1_weight

    def _effective_source_weights(self, agentic_alpha: float) -> Dict[str, float]:
        alpha = min(1.0, max(0.0, float(agentic_alpha)))
        return {
            "view": self.source_weight_view,
            "batch_neighbor": self.source_weight_batch,
            "memory_neighbor": self.source_weight_memory,
            "arf_planned": alpha * self.source_weight_arf,
            "arf_missed_bonus": alpha * self.source_weight_missed_bonus,
            "hard_negative_weight": 1.0 + alpha * (self.hard_negative_weight - 1.0),
        }

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        epoch = self._current_epoch(outputs)
        device = outputs["u_a"].device
        zero = torch.zeros((), device=device)
        memory = outputs.get("planner_memory", None)
        planner = outputs.get("graph_planner", None)
        sample_indices = outputs.get("sample_indices", None)
        neighbor_indices = outputs.get("neighbor_indices", None)
        if memory is None or planner is None or sample_indices is None or neighbor_indices is None:
            raise ValueError(
                "PhasedAgenticUnifiedContrastiveLoss requires planner_memory, graph_planner, sample_indices, and neighbor_indices."
            )

        stage1_weight, agentic_alpha = self._phase_weights(epoch)
        schedule = self._schedule(epoch)
        if agentic_alpha <= 0.0 or (self.actual_trace_start_epoch > 0 and epoch < self.actual_trace_start_epoch):
            schedule = dict(schedule)
            schedule["use_actual_trace"] = False
            schedule["eta_missed"] = 0.0
            schedule["eta_false"] = 0.0
        hard_mining_enabled = agentic_alpha > 0.0 and bool(schedule["use_actual_trace"]) and (
            self.hard_mining_start_epoch <= 0 or epoch >= self.hard_mining_start_epoch
        )

        component_agentic, metrics = self._unified_info_nce(
            outputs,
            memory,
            planner,
            sample_indices,
            neighbor_indices,
            schedule,
            hard_mining_enabled,
            source_weights=self._effective_source_weights(agentic_alpha),
        )
        component_quant = self.quantization(outputs["u_a"], outputs["u_b"])
        component_bit_balance = self.balance(outputs["u_a"], outputs["u_b"])
        loss_hash = (
            float(schedule["lambda_quant"]) * component_quant
            + float(schedule["lambda_balance"]) * component_bit_balance
        )
        total = component_agentic + loss_hash

        targets_a = metrics["targets_a"]
        targets_b = metrics["targets_b"]

        def avg_target_metric(key: str) -> torch.Tensor:
            a = targets_a[key].to(device) if torch.is_tensor(targets_a[key]) else torch.tensor(targets_a[key], device=device)
            b = targets_b[key].to(device) if torch.is_tensor(targets_b[key]) else torch.tensor(targets_b[key], device=device)
            return 0.5 * (a.float() + b.float())

        return {
            "component_view_contrast": zero,
            "component_batch_neighbor": zero,
            "component_memory_neighbor": zero,
            "component_arf_static": zero,
            "component_arf_contrastive": zero,
            "component_agentic_contrastive": component_agentic,
            "component_quant": component_quant,
            "component_bit_balance": component_bit_balance,
            "loss_view": zero,
            "loss_semantic": component_agentic,
            "loss_arf": zero,
            "loss_hash": loss_hash,
            "loss": total,
            "metric_agentic_raw": component_agentic.detach(),
            "metric_agentic_pos_view": metrics["view_count"],
            "metric_agentic_pos_batch": metrics["batch_count"],
            "metric_agentic_pos_memory": metrics["memory_count"],
            "metric_agentic_pos_arf": metrics["arf_count"],
            "metric_agentic_hard_positive_count": metrics["hard_positive_count"],
            "metric_agentic_hard_negative_count": metrics["hard_negative_count"],
            "metric_agentic_positive_weight_mean": metrics["positive_weight"],
            "metric_agentic_mix_alpha": torch.tensor(float(agentic_alpha), device=device),
            "metric_stage1_keep_weight": torch.tensor(float(stage1_weight), device=device),
            "metric_arf_target_count": 0.5
            * (metrics["metrics_a"]["positive_count"].float() + metrics["metrics_b"]["positive_count"].float()),
            "metric_arf_target_mean": 0.5
            * (metrics["metrics_a"]["positive_mean"].float() + metrics["metrics_b"]["positive_mean"].float()),
            "metric_arf_hard_positive_count": metrics["hard_positive_count"],
            "metric_arf_hard_negative_count": metrics["hard_negative_count"],
            "metric_arf_actual_overlap": avg_target_metric("metric_actual_overlap"),
            "metric_arf_false_ratio": avg_target_metric("metric_false_ratio"),
            "metric_arf_missed_ratio": avg_target_metric("metric_missed_ratio"),
            "metric_arf_retrieved_target_mean": avg_target_metric("metric_retrieved_target_mean"),
            "metric_arf_feedback_weight_mean": avg_target_metric("metric_feedback_weight_mean"),
            "metric_arf_eta_missed": torch.tensor(float(schedule["eta_missed"]), device=device),
            "metric_arf_eta_false": torch.tensor(float(schedule["eta_false"]), device=device),
            "metric_arf_omega_z": torch.tensor(float(schedule["omega_z"]), device=device),
            "metric_arf_gamma": torch.tensor(float(schedule["gamma"]), device=device),
            "metric_arf_lambda_quant": torch.tensor(float(schedule["lambda_quant"]), device=device),
        }

    _weights = _phase_weights


class Stage1ScheduledAgenticUnifiedLoss(PhasedAgenticUnifiedContrastiveLoss):
    """Backward-compatible name for phased AUCL."""


class Stage1WarmupAgenticUnifiedLoss(Stage1ScheduledAgenticUnifiedLoss):
    """Backward-compatible hard-switch name for phased AUCL."""


class LegacyStage1ScheduledAgenticUnifiedLoss(nn.Module):
    """Historical Stage1 loss followed by v1 single-pool AUCL.

    This preserves the old switch-strategy experiment semantics after
    ``stage1_warmup_agentic_unified`` was repointed to the true phased AUCL.
    """

    requires_planner_memory = True

    def __init__(self, cfg: Dict):
        super().__init__()
        agentic_cfg = cfg.get("agentic_contrastive", cfg.get("loss", {}).get("agentic_contrastive", {}))
        self.stage1_warmup_epochs = int(agentic_cfg.get("stage1_warmup_epochs", 30))
        self.schedule_mode = str(agentic_cfg.get("schedule_mode", agentic_cfg.get("switch_mode", "hard"))).lower()
        self.ramp_epochs = max(0, int(agentic_cfg.get("ramp_epochs", 0)))
        self.post_stage1_weight = float(
            agentic_cfg.get("post_stage1_weight", agentic_cfg.get("stage1_keep_weight", 0.0))
        )
        self.post_stage1_weight = min(1.0, max(0.0, self.post_stage1_weight))

        stage1_cfg = deepcopy(cfg)
        stage1_override = agentic_cfg.get("stage1_lambda_memory_neighbor", None)
        if stage1_override is not None:
            stage1_loss_cfg = stage1_cfg.setdefault("loss", {})
            stage1_semantic_cfg = stage1_loss_cfg.setdefault("semantic", {})
            stage1_semantic_cfg["lambda_memory_neighbor"] = float(stage1_override)

        self.stage1_loss = RFClathLoss(stage1_cfg)
        self.agentic_loss = AgenticUnifiedContrastiveLoss(cfg)

    def _current_epoch(self, outputs: Dict[str, torch.Tensor]) -> int:
        epoch = outputs.get("epoch", None)
        if epoch is None:
            return 1
        if torch.is_tensor(epoch):
            return int(epoch.detach().cpu().item())
        return int(epoch)

    def _with_agentic_defaults(self, losses: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        zero = torch.zeros((), device=device)
        return {
            **losses,
            "component_agentic_contrastive": zero,
            "loss_arf": zero,
            "metric_agentic_raw": zero,
            "metric_agentic_pos_view": zero,
            "metric_agentic_pos_batch": zero,
            "metric_agentic_pos_memory": zero,
            "metric_agentic_pos_arf": zero,
            "metric_agentic_hard_positive_count": zero,
            "metric_agentic_hard_negative_count": zero,
            "metric_agentic_positive_weight_mean": zero,
            "metric_agentic_mix_alpha": zero,
            "metric_stage1_keep_weight": torch.ones((), device=device),
            "metric_arf_target_count": zero,
            "metric_arf_target_mean": zero,
            "metric_arf_hard_positive_count": zero,
            "metric_arf_hard_negative_count": zero,
            "metric_arf_actual_overlap": zero,
            "metric_arf_false_ratio": zero,
            "metric_arf_missed_ratio": zero,
            "metric_arf_retrieved_target_mean": zero,
            "metric_arf_feedback_weight_mean": zero,
            "metric_arf_eta_missed": zero,
            "metric_arf_eta_false": zero,
            "metric_arf_omega_z": zero,
            "metric_arf_gamma": zero,
            "metric_arf_lambda_quant": zero,
        }

    def _weights(self, epoch: int) -> Tuple[float, float]:
        if epoch <= self.stage1_warmup_epochs:
            return 1.0, 0.0
        if self.schedule_mode == "ramp":
            if self.ramp_epochs <= 0:
                progress = 1.0
            else:
                progress = min(1.0, max(0.0, float(epoch - self.stage1_warmup_epochs) / float(self.ramp_epochs)))
            stage1_weight = 1.0 - progress * (1.0 - self.post_stage1_weight)
            return stage1_weight, 1.0 - stage1_weight
        if self.schedule_mode in {"hybrid", "retain_stage1", "retained_stage1"}:
            return self.post_stage1_weight, 1.0 - self.post_stage1_weight
        return self.post_stage1_weight, 1.0 - self.post_stage1_weight

    def _with_schedule_metrics(
        self,
        losses: Dict[str, torch.Tensor],
        device: torch.device,
        stage1_weight: float,
        agentic_weight: float,
    ) -> Dict[str, torch.Tensor]:
        return {
            **losses,
            "metric_agentic_mix_alpha": torch.tensor(agentic_weight, device=device),
            "metric_stage1_keep_weight": torch.tensor(stage1_weight, device=device),
        }

    def _combine_losses(
        self,
        stage1_losses: Dict[str, torch.Tensor],
        agentic_losses: Dict[str, torch.Tensor],
        device: torch.device,
        stage1_weight: float,
        agentic_weight: float,
    ) -> Dict[str, torch.Tensor]:
        zero = torch.zeros((), device=device)
        combined: Dict[str, torch.Tensor] = {}
        keys = set(stage1_losses) | set(agentic_losses)
        for key in keys:
            stage1_value = stage1_losses.get(key, zero)
            agentic_value = agentic_losses.get(key, zero)
            if key.startswith("metric_agentic_") or key.startswith("metric_arf_"):
                combined[key] = agentic_value if agentic_weight > 0.0 else zero
            else:
                combined[key] = stage1_value * stage1_weight + agentic_value * agentic_weight
        combined["loss"] = stage1_losses["loss"] * stage1_weight + agentic_losses["loss"] * agentic_weight
        combined["metric_agentic_mix_alpha"] = torch.tensor(agentic_weight, device=device)
        combined["metric_stage1_keep_weight"] = torch.tensor(stage1_weight, device=device)
        return combined

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        epoch = self._current_epoch(outputs)
        device = outputs["u_a"].device
        stage1_weight, agentic_weight = self._weights(epoch)
        if agentic_weight <= 0.0:
            losses = self._with_agentic_defaults(self.stage1_loss(outputs), device)
            return self._with_schedule_metrics(losses, device, stage1_weight, agentic_weight)
        if stage1_weight <= 0.0:
            losses = self.agentic_loss(outputs)
            return self._with_schedule_metrics(losses, device, stage1_weight, agentic_weight)
        stage1_losses = self._with_agentic_defaults(self.stage1_loss(outputs), device)
        agentic_losses = self.agentic_loss(outputs)
        return self._combine_losses(stage1_losses, agentic_losses, device, stage1_weight, agentic_weight)


class LegacyStage1WarmupAgenticUnifiedLoss(LegacyStage1ScheduledAgenticUnifiedLoss):
    """Historical hard switch: Stage1 RFClathLoss, then v1 AUCL."""
