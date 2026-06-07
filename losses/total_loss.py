from typing import Dict, Iterable, Tuple

import torch
from torch import nn

from .contrastive import HashContrastiveLoss, MemoryNeighborContrastiveLoss, NeighborHashContrastiveLoss
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


class RFClathLoss(nn.Module):
    """Merged training objective for RF-CLaTH.

    The implementation exposes the final objective as three paper-facing
    groups:
        1. view: hash-code agreement between two augmented views.
        2. semantic: batch/memory pseudo-neighbor structure.
        3. hash: quantization and bit balance regularization.

    Returned ``component_*`` values are raw diagnostics; returned ``loss_*``
    values are weighted groups that sum to ``loss``.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        model_cfg = cfg["model"]
        loss_cfg = cfg.get("loss", {})
        ablation_cfg = cfg.get("ablation", {})
        hash_bits = int(model_cfg.get("hash_bits", 64))
        temperature = float(loss_cfg.get("temperature", 0.2))

        self.lambda_view = _get_float(
            loss_cfg,
            (("view", "lambda"), ("consistency", "lambda"), ("lambda_consistency",), ("lambda_hash_con",)),
            1.0,
        )
        self.lambda_batch_neighbor = _get_float(
            loss_cfg,
            (
                ("semantic", "lambda_batch_neighbor"),
                ("lambda_neighbor_con",),
            ),
            0.0,
        )
        self.lambda_memory_neighbor = _get_float(
            loss_cfg,
            (
                ("semantic", "lambda_memory_neighbor"),
                ("lambda_memory_neighbor",),
            ),
            0.0,
        )
        self.lambda_quant = _get_float(
            loss_cfg,
            (("hash", "lambda_quant"), ("hash_regularization", "lambda_quant"), ("lambda_quant",)),
            0.1,
        )
        self.lambda_bit_balance = _get_float(
            loss_cfg,
            (
                ("hash", "lambda_bit_balance"),
                ("hash", "lambda_balance"),
                ("hash_regularization", "lambda_balance"),
                ("lambda_balance",),
            ),
            0.01,
        )
        if not bool(ablation_cfg.get("use_hash_con", True)):
            self.lambda_view = 0.0

        self.memory_neighbor_start_epoch = int(loss_cfg.get("memory_neighbor", {}).get("start_epoch", 1))
        self.hash_contrast = HashContrastiveLoss(temperature=temperature)
        self.neighbor_hash_contrast = NeighborHashContrastiveLoss(
            temperature=float(loss_cfg.get("neighbor_temperature", temperature)),
            symmetric_neighbors=bool(loss_cfg.get("symmetric_neighbors", True)),
        )
        memory_cfg = loss_cfg.get("memory_neighbor", {})
        self.memory_neighbor_contrast = MemoryNeighborContrastiveLoss(
            num_items=int(memory_cfg.get("num_items", 0)),
            hash_bits=hash_bits,
            temperature=float(memory_cfg.get("temperature", loss_cfg.get("neighbor_temperature", temperature))),
            momentum=float(memory_cfg.get("momentum", 0.9)),
            positives_per_anchor=int(memory_cfg.get("positives_per_anchor", 10)),
            include_self=bool(memory_cfg.get("include_self", False)),
        )
        self.quantization = QuantizationLoss()
        self.balance = BalanceLoss()

    def _current_epoch(self, outputs: Dict[str, torch.Tensor]) -> int:
        epoch = outputs.get("epoch", None)
        if epoch is None:
            return 1
        if torch.is_tensor(epoch):
            return int(epoch.detach().cpu().item())
        return int(epoch)

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = outputs["u_a"].device
        zero = torch.zeros((), device=device)
        components = {
            "component_view_contrast": zero,
            "component_batch_neighbor": zero,
            "component_memory_neighbor": zero,
            "component_quant": zero,
            "component_bit_balance": zero,
        }

        if self.lambda_view > 0:
            components["component_view_contrast"] = self.hash_contrast(outputs["u_a"], outputs["u_b"])
        if self.lambda_quant > 0:
            components["component_quant"] = self.quantization(outputs["u_a"], outputs["u_b"])
        if self.lambda_bit_balance > 0:
            components["component_bit_balance"] = self.balance(outputs["u_a"], outputs["u_b"])

        has_neighbors = "sample_indices" in outputs and "neighbor_indices" in outputs
        if self.lambda_batch_neighbor > 0 and has_neighbors:
            components["component_batch_neighbor"] = self.neighbor_hash_contrast(
                outputs["u_a"],
                outputs["u_b"],
                outputs["sample_indices"],
                outputs["neighbor_indices"],
            )

        current_epoch = self._current_epoch(outputs)
        if self.lambda_memory_neighbor > 0 and current_epoch >= self.memory_neighbor_start_epoch and has_neighbors:
            components["component_memory_neighbor"] = self.memory_neighbor_contrast(
                outputs["u_a"],
                outputs["u_b"],
                outputs["sample_indices"],
                outputs["neighbor_indices"],
            )

        loss_view = self.lambda_view * components["component_view_contrast"]
        loss_semantic = (
            self.lambda_batch_neighbor * components["component_batch_neighbor"]
            + self.lambda_memory_neighbor * components["component_memory_neighbor"]
        )
        loss_hash = (
            self.lambda_quant * components["component_quant"]
            + self.lambda_bit_balance * components["component_bit_balance"]
        )
        total = loss_view + loss_semantic + loss_hash

        return {
            **components,
            "loss_view": loss_view,
            "loss_semantic": loss_semantic,
            "loss_hash": loss_hash,
            "loss": total,
        }
