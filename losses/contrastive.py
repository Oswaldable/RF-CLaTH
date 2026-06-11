import math

import torch
from torch import nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """Single-machine NT-Xent loss.

    Inputs:
        z_a, z_b: [B, D]
    Positive pairs are aligned rows across the two views.
    """

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        if z_a.shape[0] != z_b.shape[0]:
            raise ValueError("z_a and z_b must have the same batch size.")
        batch_size = z_a.shape[0]
        if batch_size < 2:
            return z_a.new_tensor(0.0)
        z = torch.cat([z_a, z_b], dim=0)
        z = F.normalize(z, dim=-1)
        logits = z @ z.t()
        logits = logits / self.temperature
        mask_value = -1e4 if logits.dtype in {torch.float16, torch.bfloat16} else -1e9
        logits = logits.masked_fill(torch.eye(2 * batch_size, device=z.device, dtype=torch.bool), mask_value)
        targets = torch.arange(2 * batch_size, device=z.device)
        targets = (targets + batch_size) % (2 * batch_size)
        return F.cross_entropy(logits, targets)


class HashContrastiveLoss(nn.Module):
    """NT-Xent over soft hash codes."""

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.nt_xent = NTXentLoss(temperature=temperature)

    def forward(self, u_a: torch.Tensor, u_b: torch.Tensor) -> torch.Tensor:
        return self.nt_xent(u_a, u_b)


class MultiPositiveNTXentLoss(nn.Module):
    """Multi-positive NT-Xent loss for pseudo-neighbor hash supervision.

    Inputs:
        z_a, z_b: [B, D], two augmented views of the current batch.
        sample_indices: [B], global dataset indices for the current batch.
        neighbor_indices: [B, K], global top-K neighbor indices per sample.

    Positives include the paired view of the same sample and any configured
    nearest-neighbor samples that are present in the same mini-batch.
    """

    def __init__(self, temperature: float = 0.2, symmetric_neighbors: bool = True):
        super().__init__()
        self.temperature = temperature
        self.symmetric_neighbors = symmetric_neighbors

    def _positive_mask(
        self,
        sample_indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        device = sample_indices.device
        base = torch.eye(batch_size, device=device, dtype=torch.bool)
        if neighbor_indices.numel() > 0:
            row_neighbor = (neighbor_indices[:, :, None] == sample_indices[None, None, :]).any(dim=1)
            if self.symmetric_neighbors:
                row_neighbor = row_neighbor | row_neighbor.t()
            base = base | row_neighbor

        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, device=device, dtype=torch.bool)
        pos_mask[:batch_size, :batch_size] = base
        pos_mask[:batch_size, batch_size:] = base
        pos_mask[batch_size:, :batch_size] = base
        pos_mask[batch_size:, batch_size:] = base
        pos_mask.fill_diagonal_(False)
        return pos_mask

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        sample_indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        if z_a.shape[0] != z_b.shape[0]:
            raise ValueError("z_a and z_b must have the same batch size.")
        batch_size = z_a.shape[0]
        if batch_size < 2:
            return z_a.new_tensor(0.0)

        sample_indices = sample_indices.to(device=z_a.device, dtype=torch.long)
        neighbor_indices = neighbor_indices.to(device=z_a.device, dtype=torch.long)
        z = torch.cat([z_a, z_b], dim=0)
        z = F.normalize(z, dim=-1)
        logits = z @ z.t()
        logits = logits / self.temperature

        self_mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        pos_mask = self._positive_mask(sample_indices, neighbor_indices, batch_size)
        valid = pos_mask.any(dim=1)
        if not valid.any():
            return z_a.new_tensor(0.0)

        mask_value = -1e4 if logits.dtype in {torch.float16, torch.bfloat16} else -1e9
        denom_logits = logits.masked_fill(self_mask, mask_value)
        pos_logits = logits.masked_fill(~pos_mask, mask_value)
        loss = torch.logsumexp(denom_logits, dim=1) - torch.logsumexp(pos_logits, dim=1)
        return loss[valid].mean()


class NeighborHashContrastiveLoss(nn.Module):
    """Multi-positive contrastive loss over soft hash codes."""

    def __init__(self, temperature: float = 0.2, symmetric_neighbors: bool = True):
        super().__init__()
        self.nt_xent = MultiPositiveNTXentLoss(
            temperature=temperature,
            symmetric_neighbors=symmetric_neighbors,
        )

    def forward(
        self,
        u_a: torch.Tensor,
        u_b: torch.Tensor,
        sample_indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.nt_xent(u_a, u_b, sample_indices, neighbor_indices)


class MemoryNeighborContrastiveLoss(nn.Module):
    """Memory-bank multi-positive contrastive loss for pseudo-neighbor hashes.

    The static neighbor table comes from raw/pre-extracted video features, while
    this module stores the latest EMA hash embedding for each training sample.
    For every query in the current mini-batch, positives are the memory entries
    of its raw-feature nearest neighbors; all valid memory entries are negatives.

    Shapes:
        u_a, u_b: [B, K]
        sample_indices: [B]
        neighbor_indices: [B, M]
    """

    def __init__(
        self,
        num_items: int,
        hash_bits: int,
        temperature: float = 0.2,
        momentum: float = 0.9,
        positives_per_anchor: int = 10,
        include_self: bool = False,
    ):
        super().__init__()
        self.num_items = int(num_items)
        self.hash_bits = int(hash_bits)
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        self.positives_per_anchor = max(1, int(positives_per_anchor))
        self.include_self = bool(include_self)
        if self.num_items > 0:
            memory = F.normalize(torch.randn(self.num_items, self.hash_bits), dim=-1)
            valid = torch.zeros(self.num_items, dtype=torch.bool)
        else:
            memory = torch.empty(0, self.hash_bits)
            valid = torch.zeros(0, dtype=torch.bool)
        self.register_buffer("memory", memory)
        self.register_buffer("valid", valid)

    @torch.no_grad()
    def _update_memory(self, sample_indices: torch.Tensor, values: torch.Tensor):
        if self.num_items <= 0:
            return
        sample_indices = sample_indices.detach().long()
        values = F.normalize(values.detach().float(), dim=-1)
        old = self.memory.index_select(0, sample_indices)
        was_valid = self.valid.index_select(0, sample_indices)
        blended = F.normalize(old * self.momentum + values * (1.0 - self.momentum), dim=-1)
        updated = torch.where(was_valid[:, None], blended, values)
        self.memory.index_copy_(0, sample_indices, updated.to(self.memory.dtype))
        self.valid.index_fill_(0, sample_indices, True)

    def _positive_mask(
        self,
        sample_indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
        valid_indices: torch.Tensor,
    ) -> torch.Tensor:
        neighbors = neighbor_indices[:, : self.positives_per_anchor].long()
        if self.include_self:
            neighbors = torch.cat([sample_indices[:, None].long(), neighbors], dim=1)
        pos = (neighbors[:, :, None] == valid_indices[None, None, :]).any(dim=1)
        return torch.cat([pos, pos], dim=0)

    def forward(
        self,
        u_a: torch.Tensor,
        u_b: torch.Tensor,
        sample_indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_items <= 0 or u_a.shape[0] < 2:
            return u_a.new_zeros(())

        sample_indices = sample_indices.to(device=u_a.device, dtype=torch.long)
        neighbor_indices = neighbor_indices.to(device=u_a.device, dtype=torch.long)
        valid_indices = torch.nonzero(self.valid, as_tuple=False).flatten().to(u_a.device)
        valid_indices = valid_indices[(valid_indices >= 0) & (valid_indices < self.num_items)]
        current_value = 0.5 * (u_a.detach() + u_b.detach())

        if valid_indices.numel() == 0:
            self._update_memory(sample_indices, current_value)
            return u_a.new_zeros(())

        memory = F.normalize(self.memory.index_select(0, valid_indices).to(device=u_a.device, dtype=torch.float32), dim=-1)
        queries = F.normalize(torch.cat([u_a, u_b], dim=0).float(), dim=-1)
        logits = queries @ memory.t()
        logits = logits / max(self.temperature, 1e-6)
        pos_mask = self._positive_mask(sample_indices, neighbor_indices, valid_indices)
        valid_rows = pos_mask.any(dim=1)
        if not bool(valid_rows.any().item()):
            self._update_memory(sample_indices, current_value)
            return u_a.new_zeros(())

        mask_value = -1e4 if logits.dtype in {torch.float16, torch.bfloat16} else -1e9
        pos_logits = logits.masked_fill(~pos_mask, mask_value)
        loss = torch.logsumexp(logits, dim=1) - torch.logsumexp(pos_logits, dim=1)
        loss = loss[valid_rows].mean()
        self._update_memory(sample_indices, current_value)
        return loss.to(dtype=u_a.dtype)


class AgenticMemoryNeighborContrastiveLoss(MemoryNeighborContrastiveLoss):
    """Memory-neighbor InfoNCE with planner feedback inside the memory channel.

    This keeps the Stage1 memory-bank behavior and update timing intact while
    exposing a feedback variant that adds planner positives, missed positives,
    and actual-not-planned hard negatives.
    """

    def __init__(
        self,
        num_items: int,
        hash_bits: int,
        temperature: float = 0.2,
        momentum: float = 0.9,
        positives_per_anchor: int = 10,
        include_self: bool = False,
        planned_topk: int = 5,
        missed_topk: int = 5,
        raw_positive_weight: float = 1.0,
        planned_positive_weight: float = 0.5,
        missed_positive_weight: float = 1.25,
        hard_negative_weight: float = 1.10,
        max_positive_weight: float = 2.0,
        normalize_sources: bool = False,
    ):
        super().__init__(
            num_items=num_items,
            hash_bits=hash_bits,
            temperature=temperature,
            momentum=momentum,
            positives_per_anchor=positives_per_anchor,
            include_self=include_self,
        )
        self.planned_topk = max(0, int(planned_topk))
        self.missed_topk = max(0, int(missed_topk))
        self.raw_positive_weight = float(raw_positive_weight)
        self.planned_positive_weight = float(planned_positive_weight)
        self.missed_positive_weight = float(missed_positive_weight)
        self.hard_negative_weight = max(1.0, float(hard_negative_weight))
        self.max_positive_weight = float(max_positive_weight)
        self.normalize_sources = bool(normalize_sources)

    def _zero_result(self, u_a: torch.Tensor):
        zero = u_a.new_zeros(())
        return {
            "memory_raw": zero,
            "memory_feedback": zero,
            "memory_agentic": zero,
            "pos_raw_count": zero,
            "pos_planned_count": zero,
            "pos_missed_count": zero,
            "hard_negative_count": zero,
            "positive_weight_mean": zero,
        }

    def _source_weight(self, mask: torch.Tensor, weight: float) -> torch.Tensor:
        if weight <= 0 or mask.numel() == 0:
            return mask.float() * 0.0
        mask_f = mask.float()
        if self.normalize_sources:
            denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            return mask_f * (float(weight) / denom)
        return mask_f * float(weight)

    def _rows_to_columns(
        self,
        row_indices: torch.Tensor,
        row_mask: torch.Tensor,
        valid_indices: torch.Tensor,
    ) -> torch.Tensor:
        if row_indices.numel() == 0 or valid_indices.numel() == 0:
            return torch.zeros(
                row_indices.shape[0],
                valid_indices.shape[0],
                dtype=torch.bool,
                device=valid_indices.device,
            )
        row_indices = row_indices.to(device=valid_indices.device, dtype=torch.long)
        row_mask = row_mask.to(device=valid_indices.device, dtype=torch.bool)
        masked = row_indices.masked_fill(~row_mask, -1)
        return (masked.unsqueeze(-1) == valid_indices.view(1, 1, -1)).any(dim=1)

    def _feedback_masks_for_targets(
        self,
        targets,
        valid_indices: torch.Tensor,
        hard_mining_enabled: bool,
    ):
        planned_indices = targets["planned_indices"].to(device=valid_indices.device, dtype=torch.long)
        actual_indices = targets["actual_indices"].to(device=valid_indices.device, dtype=torch.long)
        target_mask = targets["target_mask"].to(device=valid_indices.device, dtype=torch.bool)

        batch_size = planned_indices.shape[0]
        if planned_indices.numel() == 0:
            empty = torch.zeros(batch_size, valid_indices.shape[0], dtype=torch.bool, device=valid_indices.device)
            return empty, empty, empty

        planned_cols = planned_indices.shape[1]
        planned_mask = target_mask[:, :planned_cols] if target_mask.shape[1] >= planned_cols else torch.ones_like(planned_indices, dtype=torch.bool)

        if self.planned_topk > 0:
            planned_positive_mask = planned_mask.clone()
            planned_positive_mask[:, self.planned_topk :] = False
        else:
            planned_positive_mask = torch.zeros_like(planned_mask)

        if actual_indices.numel() > 0:
            actual_cols = actual_indices.shape[1]
            actual_start = planned_cols
            actual_end = actual_start + actual_cols
            if target_mask.shape[1] >= actual_end:
                actual_mask = target_mask[:, actual_start:actual_end]
            else:
                actual_mask = torch.ones_like(actual_indices, dtype=torch.bool)
            actual_valid_indices = actual_indices.masked_fill(~actual_mask, -1)
            in_actual = (planned_indices.unsqueeze(-1) == actual_valid_indices.unsqueeze(1)).any(dim=-1)
            planned_valid_indices = planned_indices.masked_fill(~planned_mask, -1)
            in_planned = (actual_indices.unsqueeze(-1) == planned_valid_indices.unsqueeze(1)).any(dim=-1)
        else:
            actual_mask = torch.zeros_like(actual_indices, dtype=torch.bool)
            in_actual = torch.zeros_like(planned_mask)
            in_planned = torch.zeros_like(actual_mask)

        if hard_mining_enabled and self.missed_topk > 0:
            missed_mask = planned_mask & (~in_actual)
            missed_rank = missed_mask.long().cumsum(dim=1)
            missed_mask = missed_mask & (missed_rank <= self.missed_topk)
        else:
            missed_mask = torch.zeros_like(planned_mask)

        if hard_mining_enabled and actual_indices.numel() > 0:
            false_mask = actual_mask & (~in_planned)
        else:
            false_mask = torch.zeros_like(actual_mask)

        planned_cols_mask = self._rows_to_columns(planned_indices, planned_positive_mask, valid_indices)
        missed_cols_mask = self._rows_to_columns(planned_indices, missed_mask, valid_indices)
        false_cols_mask = self._rows_to_columns(actual_indices, false_mask, valid_indices)
        return planned_cols_mask, missed_cols_mask, false_cols_mask

    def forward_agentic(
        self,
        u_a: torch.Tensor,
        u_b: torch.Tensor,
        sample_indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
        targets_a,
        targets_b,
        beta: float,
        hard_mining_enabled: bool,
    ):
        if self.num_items <= 0 or u_a.shape[0] < 2:
            return self._zero_result(u_a)

        sample_indices = sample_indices.to(device=u_a.device, dtype=torch.long)
        neighbor_indices = neighbor_indices.to(device=u_a.device, dtype=torch.long)
        valid_indices = torch.nonzero(self.valid, as_tuple=False).flatten().to(u_a.device)
        valid_indices = valid_indices[(valid_indices >= 0) & (valid_indices < self.num_items)]
        current_value = 0.5 * (u_a.detach() + u_b.detach())

        if valid_indices.numel() == 0:
            self._update_memory(sample_indices, current_value)
            return self._zero_result(u_a)

        memory = F.normalize(self.memory.index_select(0, valid_indices).to(device=u_a.device, dtype=torch.float32), dim=-1)
        queries = F.normalize(torch.cat([u_a, u_b], dim=0).float(), dim=-1)
        logits = queries @ memory.t()
        logits = logits / max(self.temperature, 1e-6)

        raw_pos = self._positive_mask(sample_indices, neighbor_indices, valid_indices)
        valid_rows_raw = raw_pos.any(dim=1)
        mask_value = -1e4 if logits.dtype in {torch.float16, torch.bfloat16} else -1e9

        if bool(valid_rows_raw.any().item()):
            raw_pos_logits = logits.masked_fill(~raw_pos, mask_value)
            raw_loss = torch.logsumexp(logits, dim=1) - torch.logsumexp(raw_pos_logits, dim=1)
            raw_loss = raw_loss[valid_rows_raw].mean().to(dtype=u_a.dtype)
        else:
            raw_loss = u_a.new_zeros(())

        planned_a, missed_a, false_a = self._feedback_masks_for_targets(
            targets_a,
            valid_indices,
            hard_mining_enabled=hard_mining_enabled,
        )
        planned_b, missed_b, false_b = self._feedback_masks_for_targets(
            targets_b,
            valid_indices,
            hard_mining_enabled=hard_mining_enabled,
        )
        planned_pos = torch.cat([planned_a, planned_b], dim=0)
        missed_pos = torch.cat([missed_a, missed_b], dim=0)
        hard_neg = torch.cat([false_a, false_b], dim=0)

        positive_weights = torch.zeros_like(logits, dtype=torch.float32)
        positive_weights += self._source_weight(raw_pos, self.raw_positive_weight)
        positive_weights += self._source_weight(planned_pos, self.planned_positive_weight)
        positive_weights += self._source_weight(missed_pos, self.missed_positive_weight)
        positive_weights = torch.clamp(positive_weights, min=0.0, max=self.max_positive_weight)

        valid_rows_feedback = (positive_weights > 0).any(dim=1)
        if bool(valid_rows_feedback.any().item()):
            denom_logits = logits
            if self.hard_negative_weight > 1.0:
                denom_logits = denom_logits + hard_neg.float() * math.log(self.hard_negative_weight)
            feedback_pos_logits = logits + torch.log(positive_weights.clamp_min(1e-12))
            feedback_pos_logits = feedback_pos_logits.masked_fill(positive_weights <= 0, mask_value)
            feedback_loss = torch.logsumexp(denom_logits, dim=1) - torch.logsumexp(feedback_pos_logits, dim=1)
            feedback_loss = feedback_loss[valid_rows_feedback].mean().to(dtype=u_a.dtype)
        else:
            feedback_loss = u_a.new_zeros(())

        use_beta = min(1.0, max(0.0, float(beta)))
        memory_agentic = (1.0 - use_beta) * raw_loss + use_beta * feedback_loss
        pos_active = positive_weights > 0
        positive_weight_mean = positive_weights[pos_active].mean() if pos_active.any() else u_a.new_zeros(())

        result = {
            "memory_raw": raw_loss,
            "memory_feedback": feedback_loss,
            "memory_agentic": memory_agentic.to(dtype=u_a.dtype),
            "pos_raw_count": raw_pos.float().sum(dim=1).mean(),
            "pos_planned_count": planned_pos.float().sum(dim=1).mean(),
            "pos_missed_count": missed_pos.float().sum(dim=1).mean(),
            "hard_negative_count": hard_neg.float().sum(dim=1).mean(),
            "positive_weight_mean": positive_weight_mean,
        }
        self._update_memory(sample_indices, current_value)
        return result
