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
