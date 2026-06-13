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


class WeightedSemanticContrastiveLoss(nn.Module):
    """One weighted multi-positive InfoNCE for paired-view and batch neighbors."""

    def __init__(
        self,
        temperature: float = 0.2,
        symmetric_neighbors: bool = True,
        view_positive_weight: float = 0.6,
        neighbor_positive_weight: float = 1.0,
        max_positive_weight: float = 2.0,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.symmetric_neighbors = bool(symmetric_neighbors)
        self.view_positive_weight = float(view_positive_weight)
        self.neighbor_positive_weight = float(neighbor_positive_weight)
        self.max_positive_weight = float(max_positive_weight)

    def _masks(
        self,
        sample_indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = sample_indices.device
        query_base = torch.arange(2 * batch_size, device=device) % batch_size
        candidate_base = torch.arange(2 * batch_size, device=device) % batch_size
        same_sample = sample_indices[query_base].unsqueeze(1) == sample_indices[candidate_base].unsqueeze(0)
        self_mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        view_mask = same_sample & (~self_mask)

        if neighbor_indices.numel() == 0:
            neighbor_base = torch.zeros(batch_size, batch_size, device=device, dtype=torch.bool)
        else:
            neighbor_base = (neighbor_indices[:, :, None] == sample_indices[None, None, :]).any(dim=1)
            if self.symmetric_neighbors:
                neighbor_base = neighbor_base | neighbor_base.t()
            neighbor_base = neighbor_base & (~torch.eye(batch_size, device=device, dtype=torch.bool))
        neighbor_mask = neighbor_base[query_base[:, None], candidate_base[None, :]]
        neighbor_mask = neighbor_mask & (~self_mask)
        return view_mask, neighbor_mask

    def forward(
        self,
        u_a: torch.Tensor,
        u_b: torch.Tensor,
        sample_indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        if u_a.shape[0] != u_b.shape[0]:
            raise ValueError("u_a and u_b must have the same batch size.")
        batch_size = u_a.shape[0]
        if batch_size < 2:
            return u_a.new_zeros(())

        sample_indices = sample_indices.to(device=u_a.device, dtype=torch.long)
        neighbor_indices = neighbor_indices.to(device=u_a.device, dtype=torch.long)
        z = F.normalize(torch.cat([u_a, u_b], dim=0).float(), dim=-1)
        logits = (z @ z.t()) / max(self.temperature, 1e-6)
        self_mask = torch.eye(2 * batch_size, device=u_a.device, dtype=torch.bool)

        view_mask, neighbor_mask = self._masks(sample_indices, neighbor_indices, batch_size)
        positive_weights = torch.zeros_like(logits, dtype=torch.float32)
        if self.view_positive_weight > 0:
            positive_weights = positive_weights + view_mask.float() * self.view_positive_weight
        if self.neighbor_positive_weight > 0:
            positive_weights = positive_weights + neighbor_mask.float() * self.neighbor_positive_weight
        positive_weights = positive_weights.clamp(min=0.0, max=self.max_positive_weight)

        valid_rows = (positive_weights > 0).any(dim=1)
        if not bool(valid_rows.any().item()):
            return u_a.new_zeros(())

        mask_value = -1e4 if logits.dtype in {torch.float16, torch.bfloat16} else -1e9
        denom_logits = logits.masked_fill(self_mask, mask_value)
        pos_logits = logits + torch.log(positive_weights.clamp_min(1e-12))
        pos_logits = pos_logits.masked_fill(positive_weights <= 0, mask_value)
        loss = torch.logsumexp(denom_logits, dim=1) - torch.logsumexp(pos_logits, dim=1)
        return loss[valid_rows].mean().to(dtype=u_a.dtype)


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


class SelfCalibratedMemoryNeighborContrastiveLoss(MemoryNeighborContrastiveLoss):
    """Memory InfoNCE with trust-gated planner feedback in the memory channel."""

    def __init__(
        self,
        num_items: int,
        hash_bits: int,
        temperature: float = 0.2,
        momentum: float = 0.9,
        positives_per_anchor: int = 10,
        include_self: bool = False,
        raw_positive_weight: float = 1.0,
        planned_positive_weight: float = 0.5,
        missed_positive_weight: float = 1.25,
        hard_negative_weight: float = 1.10,
        max_positive_weight: float = 2.0,
        trust_momentum: float = 0.9,
        edge_momentum: float = 0.9,
        edge_slots: int = 64,
        planned_topk: int = 5,
        missed_topk: int = 5,
        raw_trust_topk: int = 20,
        edge_min_value: float = 1e-4,
    ):
        super().__init__(
            num_items=num_items,
            hash_bits=hash_bits,
            temperature=temperature,
            momentum=momentum,
            positives_per_anchor=positives_per_anchor,
            include_self=include_self,
        )
        self.raw_positive_weight = float(raw_positive_weight)
        self.planned_positive_weight = float(planned_positive_weight)
        self.missed_positive_weight = float(missed_positive_weight)
        self.hard_negative_weight = max(1.0, float(hard_negative_weight))
        self.max_positive_weight = float(max_positive_weight)
        self.trust_momentum = min(1.0, max(0.0, float(trust_momentum)))
        self.edge_momentum = min(1.0, max(0.0, float(edge_momentum)))
        self.edge_slots = max(1, int(edge_slots))
        self.planned_topk = max(0, int(planned_topk))
        self.missed_topk = max(0, int(missed_topk))
        self.raw_trust_topk = max(1, int(raw_trust_topk))
        self.edge_min_value = max(0.0, float(edge_min_value))

        if self.num_items > 0:
            trust = torch.zeros(self.num_items, dtype=torch.float32)
            pos_edge_indices = torch.full((self.num_items, self.edge_slots), -1, dtype=torch.long)
            pos_edge_values = torch.zeros(self.num_items, self.edge_slots, dtype=torch.float32)
            neg_edge_indices = torch.full((self.num_items, self.edge_slots), -1, dtype=torch.long)
            neg_edge_values = torch.zeros(self.num_items, self.edge_slots, dtype=torch.float32)
        else:
            trust = torch.empty(0, dtype=torch.float32)
            pos_edge_indices = torch.empty(0, self.edge_slots, dtype=torch.long)
            pos_edge_values = torch.empty(0, self.edge_slots, dtype=torch.float32)
            neg_edge_indices = torch.empty(0, self.edge_slots, dtype=torch.long)
            neg_edge_values = torch.empty(0, self.edge_slots, dtype=torch.float32)
        self.register_buffer("trust", trust)
        self.register_buffer("pos_edge_indices", pos_edge_indices)
        self.register_buffer("pos_edge_values", pos_edge_values)
        self.register_buffer("neg_edge_indices", neg_edge_indices)
        self.register_buffer("neg_edge_values", neg_edge_values)

    def _zero_result(self, u_a: torch.Tensor):
        zero = u_a.new_zeros(())
        return {
            "memory_raw": zero,
            "memory_self_calibrated": zero,
            "trust_mean": zero,
            "trust_active": zero,
            "pos_raw_count": zero,
            "pos_planned_count": zero,
            "pos_missed_count": zero,
            "hard_negative_count": zero,
            "positive_weight_mean": zero,
            "positive_persistence_mean": zero,
            "negative_persistence_mean": zero,
            "trust_observation": zero,
        }

    def _target_parts(
        self,
        targets,
        neighbor_indices: torch.Tensor,
        hard_mining_enabled: bool,
    ):
        planned_indices = targets["planned_indices"].to(device=neighbor_indices.device, dtype=torch.long)
        actual_indices = targets["actual_indices"].to(device=neighbor_indices.device, dtype=torch.long)
        target_mask = targets["target_mask"].to(device=neighbor_indices.device, dtype=torch.bool)
        batch_size = neighbor_indices.shape[0]
        if planned_indices.numel() == 0:
            empty_idx = torch.empty(batch_size, 0, dtype=torch.long, device=neighbor_indices.device)
            empty_mask = torch.empty(batch_size, 0, dtype=torch.bool, device=neighbor_indices.device)
            empty_score = torch.zeros(batch_size, device=neighbor_indices.device)
            return empty_idx, empty_mask, empty_mask, actual_indices, empty_mask, empty_score

        planned_cols = planned_indices.shape[1]
        planned_mask = (
            target_mask[:, :planned_cols]
            if target_mask.shape[1] >= planned_cols
            else torch.ones_like(planned_indices, dtype=torch.bool)
        )
        planned_positive_mask = planned_mask.clone()
        if self.planned_topk > 0:
            planned_positive_mask[:, self.planned_topk :] = False
        else:
            planned_positive_mask.zero_()

        if actual_indices.numel() > 0:
            actual_cols = actual_indices.shape[1]
            actual_start = planned_cols
            actual_end = actual_start + actual_cols
            actual_mask = (
                target_mask[:, actual_start:actual_end]
                if target_mask.shape[1] >= actual_end
                else torch.ones_like(actual_indices, dtype=torch.bool)
            )
            actual_valid = actual_indices.masked_fill(~actual_mask, -1)
            planned_valid = planned_indices.masked_fill(~planned_mask, -1)
            in_actual = (planned_indices.unsqueeze(-1) == actual_valid.unsqueeze(1)).any(dim=-1)
            in_planned = (actual_indices.unsqueeze(-1) == planned_valid.unsqueeze(1)).any(dim=-1)
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

        false_mask = actual_mask & (~in_planned) if hard_mining_enabled else torch.zeros_like(actual_mask)

        raw_topk = min(self.raw_trust_topk, neighbor_indices.shape[1])
        raw_neighbors = neighbor_indices[:, :raw_topk].long()
        if actual_indices.numel() > 0 and raw_topk > 0:
            actual_valid = actual_indices.masked_fill(~actual_mask, -1)
            raw_hits = (actual_valid.unsqueeze(-1) == raw_neighbors.unsqueeze(1)).any(dim=-1) & actual_mask
            trust_observation = raw_hits.float().sum(dim=1) / float(raw_topk)
        else:
            trust_observation = torch.zeros(batch_size, device=neighbor_indices.device)
        return planned_indices, planned_positive_mask, missed_mask, actual_indices, false_mask, trust_observation

    @torch.no_grad()
    def _update_trust(self, sample_indices: torch.Tensor, trust_observation: torch.Tensor):
        if self.num_items <= 0 or sample_indices.numel() == 0:
            return
        sample_indices = sample_indices.detach().to(device=self.trust.device, dtype=torch.long)
        trust_observation = trust_observation.detach().to(device=self.trust.device, dtype=torch.float32)
        old = self.trust.index_select(0, sample_indices)
        updated = old * self.trust_momentum + trust_observation * (1.0 - self.trust_momentum)
        self.trust.index_copy_(0, sample_indices, updated)

    @torch.no_grad()
    def _update_edge_bank(
        self,
        edge_indices_bank: torch.Tensor,
        edge_values_bank: torch.Tensor,
        sample_indices: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_mask: torch.Tensor,
    ):
        if self.num_items <= 0 or edge_indices.numel() == 0:
            return
        sample_indices = sample_indices.detach().to(device=edge_indices_bank.device, dtype=torch.long)
        edge_indices = edge_indices.detach().to(device=edge_indices_bank.device, dtype=torch.long)
        edge_mask = edge_mask.detach().to(device=edge_indices_bank.device, dtype=torch.bool)

        for row_pos in range(sample_indices.shape[0]):
            row_id = int(sample_indices[row_pos].detach().cpu().item())
            if row_id < 0 or row_id >= self.num_items:
                continue
            bank_idx = edge_indices_bank[row_id]
            bank_val = edge_values_bank[row_id]
            bank_val.mul_(self.edge_momentum)
            if self.edge_min_value > 0:
                stale = bank_val <= self.edge_min_value
                bank_idx[stale] = -1
                bank_val[stale] = 0.0

            ids = edge_indices[row_pos][edge_mask[row_pos]]
            if ids.numel() == 0:
                continue
            for edge_id in ids.unique().tolist():
                edge_id = int(edge_id)
                if edge_id < 0 or edge_id == row_id:
                    continue
                match = torch.nonzero(bank_idx == edge_id, as_tuple=False).flatten()
                if match.numel() > 0:
                    slot = int(match[0].detach().cpu().item())
                else:
                    empty = torch.nonzero(bank_idx < 0, as_tuple=False).flatten()
                    if empty.numel() > 0:
                        slot = int(empty[0].detach().cpu().item())
                    else:
                        slot = int(torch.argmin(bank_val).detach().cpu().item())
                    bank_idx[slot] = edge_id
                    bank_val[slot] = 0.0
                bank_val[slot] = bank_val[slot] + (1.0 - self.edge_momentum)

    def _edge_values_to_columns(
        self,
        sample_indices: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_mask: torch.Tensor,
        valid_indices: torch.Tensor,
        edge_indices_bank: torch.Tensor,
        edge_values_bank: torch.Tensor,
    ) -> torch.Tensor:
        rows = edge_indices.shape[0]
        cols = valid_indices.shape[0]
        out = torch.zeros(rows, cols, dtype=torch.float32, device=valid_indices.device)
        if rows == 0 or cols == 0 or edge_indices.numel() == 0:
            return out

        sample_indices = sample_indices.to(device=edge_indices_bank.device, dtype=torch.long)
        edge_indices = edge_indices.to(device=valid_indices.device, dtype=torch.long)
        edge_mask = edge_mask.to(device=valid_indices.device, dtype=torch.bool)
        valid_indices = valid_indices.to(device=valid_indices.device, dtype=torch.long)
        for row_pos in range(rows):
            row_id = int(sample_indices[row_pos].detach().cpu().item())
            if row_id < 0 or row_id >= self.num_items or not bool(edge_mask[row_pos].any().item()):
                continue
            ids = edge_indices[row_pos]
            bank_idx = edge_indices_bank[row_id].to(device=valid_indices.device)
            bank_val = edge_values_bank[row_id].to(device=valid_indices.device)
            slot_match = ids.unsqueeze(1) == bank_idx.unsqueeze(0)
            values = (slot_match.float() * bank_val.unsqueeze(0)).max(dim=1).values
            values = values * edge_mask[row_pos].float()
            if not bool((values > 0).any().item()):
                continue
            col_match = ids.unsqueeze(1) == valid_indices.unsqueeze(0)
            row_values = (col_match.float() * values.unsqueeze(1)).max(dim=0).values
            out[row_pos] = row_values
        return out

    def forward_self_calibrated(
        self,
        u_a: torch.Tensor,
        u_b: torch.Tensor,
        sample_indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
        targets_a,
        targets_b,
        trace_enabled: bool,
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
        logits = (queries @ memory.t()) / max(self.temperature, 1e-6)
        raw_pos = self._positive_mask(sample_indices, neighbor_indices, valid_indices)

        mask_value = -1e4 if logits.dtype in {torch.float16, torch.bfloat16} else -1e9
        valid_rows_raw = raw_pos.any(dim=1)
        if bool(valid_rows_raw.any().item()):
            raw_pos_logits = logits.masked_fill(~raw_pos, mask_value)
            raw_loss = torch.logsumexp(logits, dim=1) - torch.logsumexp(raw_pos_logits, dim=1)
            raw_loss = raw_loss[valid_rows_raw].mean().to(dtype=u_a.dtype)
        else:
            raw_loss = u_a.new_zeros(())

        planned_persist = torch.zeros_like(logits, dtype=torch.float32)
        missed_persist = torch.zeros_like(logits, dtype=torch.float32)
        false_persist = torch.zeros_like(logits, dtype=torch.float32)
        trust_current = self.trust.index_select(0, sample_indices.to(self.trust.device)).to(device=u_a.device)
        trust_observation = torch.zeros_like(trust_current)

        if trace_enabled:
            planned_a, planned_mask_a, missed_mask_a, actual_a, false_mask_a, trust_a = self._target_parts(
                targets_a,
                neighbor_indices,
                hard_mining_enabled=hard_mining_enabled,
            )
            planned_b, planned_mask_b, missed_mask_b, actual_b, false_mask_b, trust_b = self._target_parts(
                targets_b,
                neighbor_indices,
                hard_mining_enabled=hard_mining_enabled,
            )
            trust_observation = 0.5 * (trust_a + trust_b)
            self._update_trust(sample_indices, trust_observation)
            trust_current = self.trust.index_select(0, sample_indices.to(self.trust.device)).to(device=u_a.device)

            pos_update_indices = torch.cat([planned_a, planned_b], dim=1)
            pos_update_mask = torch.cat([planned_mask_a | missed_mask_a, planned_mask_b | missed_mask_b], dim=1)
            neg_update_indices = torch.cat([actual_a, actual_b], dim=1) if actual_a.numel() or actual_b.numel() else actual_a
            neg_update_mask = torch.cat([false_mask_a, false_mask_b], dim=1) if false_mask_a.numel() or false_mask_b.numel() else false_mask_a
            self._update_edge_bank(self.pos_edge_indices, self.pos_edge_values, sample_indices, pos_update_indices, pos_update_mask)
            self._update_edge_bank(self.neg_edge_indices, self.neg_edge_values, sample_indices, neg_update_indices, neg_update_mask)

            planned_persist_a = self._edge_values_to_columns(
                sample_indices,
                planned_a,
                planned_mask_a,
                valid_indices,
                self.pos_edge_indices,
                self.pos_edge_values,
            )
            planned_persist_b = self._edge_values_to_columns(
                sample_indices,
                planned_b,
                planned_mask_b,
                valid_indices,
                self.pos_edge_indices,
                self.pos_edge_values,
            )
            missed_persist_a = self._edge_values_to_columns(
                sample_indices,
                planned_a,
                missed_mask_a,
                valid_indices,
                self.pos_edge_indices,
                self.pos_edge_values,
            )
            missed_persist_b = self._edge_values_to_columns(
                sample_indices,
                planned_b,
                missed_mask_b,
                valid_indices,
                self.pos_edge_indices,
                self.pos_edge_values,
            )
            false_persist_a = self._edge_values_to_columns(
                sample_indices,
                actual_a,
                false_mask_a,
                valid_indices,
                self.neg_edge_indices,
                self.neg_edge_values,
            )
            false_persist_b = self._edge_values_to_columns(
                sample_indices,
                actual_b,
                false_mask_b,
                valid_indices,
                self.neg_edge_indices,
                self.neg_edge_values,
            )
            planned_persist = torch.cat([planned_persist_a, planned_persist_b], dim=0)
            missed_persist = torch.cat([missed_persist_a, missed_persist_b], dim=0)
            false_persist = torch.cat([false_persist_a, false_persist_b], dim=0)

        trust_rows = torch.cat([trust_current, trust_current], dim=0).float().unsqueeze(1)
        positive_weights = raw_pos.float() * self.raw_positive_weight
        if trace_enabled:
            positive_weights = positive_weights + planned_persist * trust_rows * self.planned_positive_weight
            positive_weights = positive_weights + missed_persist * trust_rows * self.missed_positive_weight
        positive_weights = positive_weights.clamp(min=0.0, max=self.max_positive_weight)

        denom_logits = logits
        if trace_enabled and self.hard_negative_weight > 1.0:
            denom_weight = 1.0 + (self.hard_negative_weight - 1.0) * trust_rows * false_persist
            denom_logits = denom_logits + torch.log(denom_weight.clamp_min(1e-6))

        valid_rows = (positive_weights > 0).any(dim=1)
        if bool(valid_rows.any().item()):
            pos_logits = logits + torch.log(positive_weights.clamp_min(1e-12))
            pos_logits = pos_logits.masked_fill(positive_weights <= 0, mask_value)
            calibrated_loss = torch.logsumexp(denom_logits, dim=1) - torch.logsumexp(pos_logits, dim=1)
            calibrated_loss = calibrated_loss[valid_rows].mean().to(dtype=u_a.dtype)
        else:
            calibrated_loss = u_a.new_zeros(())

        pos_active = positive_weights > 0
        pos_persist_active = torch.cat([planned_persist, missed_persist], dim=1)
        positive_persistence_mean = (
            pos_persist_active[pos_persist_active > 0].mean() if (pos_persist_active > 0).any() else u_a.new_zeros(())
        )
        negative_persistence_mean = false_persist[false_persist > 0].mean() if (false_persist > 0).any() else u_a.new_zeros(())
        result = {
            "memory_raw": raw_loss,
            "memory_self_calibrated": calibrated_loss,
            "trust_mean": trust_current.mean() if trust_current.numel() > 0 else u_a.new_zeros(()),
            "trust_active": (trust_current > 0).float().mean() if trust_current.numel() > 0 else u_a.new_zeros(()),
            "pos_raw_count": raw_pos.float().sum(dim=1).mean(),
            "pos_planned_count": (planned_persist > 0).float().sum(dim=1).mean(),
            "pos_missed_count": (missed_persist > 0).float().sum(dim=1).mean(),
            "hard_negative_count": (false_persist > 0).float().sum(dim=1).mean(),
            "positive_weight_mean": positive_weights[pos_active].mean() if pos_active.any() else u_a.new_zeros(()),
            "positive_persistence_mean": positive_persistence_mean,
            "negative_persistence_mean": negative_persistence_mean,
            "trust_observation": trust_observation.mean() if trust_observation.numel() > 0 else u_a.new_zeros(()),
        }
        self._update_memory(sample_indices, current_value)
        return result
