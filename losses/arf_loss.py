from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn

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

    def _logits(self, u: torch.Tensor, memory_u: torch.Tensor) -> torch.Tensor:
        bits = max(1, int(u.shape[-1]))
        return self.gamma * torch.einsum("bd,bsd->bs", u.float(), memory_u.float()) / float(bits)

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
