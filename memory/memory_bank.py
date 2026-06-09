from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _flatten_temporal_features(video: torch.Tensor) -> torch.Tensor:
    if video.ndim < 3:
        raise ValueError(f"Expected video tensor [B, T, ...], got shape={tuple(video.shape)}")
    return video.float().flatten(start_dim=2)


@torch.no_grad()
def build_label_bank(dataset, device: torch.device) -> Optional[torch.Tensor]:
    """Build a label bank for diagnostics only.

    The returned labels are never used for training. They support the optional
    planned-neighbor label precision sanity metric.
    """

    labels = getattr(dataset, "labels", None)
    if labels is not None:
        return labels.float().to(device)

    records = getattr(dataset, "records", None)
    num_classes = int(getattr(dataset, "num_classes", 0) or 0)
    if not records or num_classes <= 0:
        return None

    rows = []
    for record in records:
        target = record.get("target", None)
        if torch.is_tensor(target):
            rows.append(target.float())
            continue
        row = torch.zeros(num_classes, dtype=torch.float32)
        for label in record.get("labels", []):
            if 0 <= int(label) < num_classes:
                row[int(label)] = 1.0
        rows.append(row)
    if not rows:
        return None
    return torch.stack(rows, dim=0).to(device)


class PlannerMemoryBank:
    """Memory bank used by Planner Graph sanity logging.

    Stage 2 does not use this bank for loss computation. It stores per-sample
    non-parametric prototypes and the current fused representation so the graph
    planner can compute top-M neighbor diagnostics on demand.
    """

    def __init__(
        self,
        num_items: int,
        device: torch.device,
        raw_dim: int = 0,
        z_dim: int = 0,
        hash_dim: int = 0,
        z_momentum: float = 0.9,
        u_momentum: float = 0.9,
        labels: Optional[torch.Tensor] = None,
    ):
        self.num_items = int(num_items)
        self.device = device
        self.z_momentum = float(z_momentum)
        self.u_momentum = float(u_momentum)
        self.sem_proto_bank = self._empty_bank(raw_dim)
        self.dyn_proto_bank = self._empty_bank(raw_dim)
        self.z_bank = self._empty_bank(z_dim)
        self.u_bank = self._empty_bank(hash_dim)
        self.sem_valid = torch.zeros(self.num_items, dtype=torch.bool, device=device)
        self.dyn_valid = torch.zeros(self.num_items, dtype=torch.bool, device=device)
        self.z_valid = torch.zeros(self.num_items, dtype=torch.bool, device=device)
        self.u_valid = torch.zeros(self.num_items, dtype=torch.bool, device=device)
        self.update_count = torch.zeros(self.num_items, dtype=torch.long, device=device)
        self.last_epoch = torch.zeros(self.num_items, dtype=torch.long, device=device)
        self.labels = labels.to(device) if labels is not None else None

    def _empty_bank(self, dim: int) -> Optional[torch.Tensor]:
        dim = int(dim)
        if dim <= 0:
            return None
        return torch.zeros(self.num_items, dim, dtype=torch.float32, device=self.device)

    def _ensure_raw_dim(self, dim: int):
        if self.sem_proto_bank is not None and self.sem_proto_bank.shape[1] == int(dim):
            return
        self.sem_proto_bank = torch.zeros(self.num_items, int(dim), dtype=torch.float32, device=self.device)
        self.dyn_proto_bank = torch.zeros(self.num_items, int(dim), dtype=torch.float32, device=self.device)
        self.sem_valid.zero_()
        self.dyn_valid.zero_()

    def _ensure_z_dim(self, dim: int):
        if self.z_bank is not None and self.z_bank.shape[1] == int(dim):
            return
        self.z_bank = torch.zeros(self.num_items, int(dim), dtype=torch.float32, device=self.device)
        self.z_valid.zero_()

    def _ensure_u_dim(self, dim: int):
        if self.u_bank is not None and self.u_bank.shape[1] == int(dim):
            return
        self.u_bank = torch.zeros(self.num_items, int(dim), dtype=torch.float32, device=self.device)
        self.u_valid.zero_()

    @torch.no_grad()
    def update_batch(
        self,
        sample_indices: torch.Tensor,
        video: torch.Tensor,
        selected_indices: torch.Tensor,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        epoch: int,
        u_a: Optional[torch.Tensor] = None,
        u_b: Optional[torch.Tensor] = None,
    ):
        indices = sample_indices.detach().long().to(self.device)
        raw = _flatten_temporal_features(video.detach()).to(self.device)
        selected = selected_indices.detach().long().to(self.device)
        if selected.ndim != 2:
            raise ValueError(f"Expected selected_indices [B, K], got shape={tuple(selected.shape)}")

        self._ensure_raw_dim(raw.shape[-1])
        self._ensure_z_dim(z_a.shape[-1])

        gather_idx = selected.unsqueeze(-1).expand(-1, -1, raw.shape[-1])
        sem_proto = torch.gather(raw, dim=1, index=gather_idx).mean(dim=1)
        sem_proto = F.normalize(sem_proto.float(), dim=-1)

        if raw.shape[1] > 1:
            dyn_proto = raw[:, 1:] - raw[:, :-1]
            dyn_proto = dyn_proto.abs().mean(dim=1)
        else:
            dyn_proto = torch.zeros_like(sem_proto)
        dyn_proto = F.normalize(dyn_proto.float(), dim=-1)

        z_proto = 0.5 * (z_a.detach().float().to(self.device) + z_b.detach().float().to(self.device))
        z_proto = F.normalize(z_proto, dim=-1)

        self.sem_proto_bank[indices] = sem_proto
        self.dyn_proto_bank[indices] = dyn_proto
        self.sem_valid[indices] = True
        self.dyn_valid[indices] = True

        old_valid = self.z_valid[indices]
        if old_valid.any():
            old_indices = indices[old_valid]
            mixed = self.z_momentum * self.z_bank[old_indices] + (1.0 - self.z_momentum) * z_proto[old_valid]
            self.z_bank[old_indices] = F.normalize(mixed, dim=-1)
        if (~old_valid).any():
            new_indices = indices[~old_valid]
            self.z_bank[new_indices] = z_proto[~old_valid]
        self.z_valid[indices] = True

        if u_a is not None and u_b is not None:
            self._ensure_u_dim(u_a.shape[-1])
            u_proto = 0.5 * (u_a.detach().float().to(self.device) + u_b.detach().float().to(self.device))
            old_u_valid = self.u_valid[indices]
            if old_u_valid.any():
                old_u_indices = indices[old_u_valid]
                mixed_u = self.u_momentum * self.u_bank[old_u_indices] + (1.0 - self.u_momentum) * u_proto[old_u_valid]
                self.u_bank[old_u_indices] = torch.clamp(mixed_u, min=-1.0, max=1.0)
            if (~old_u_valid).any():
                new_u_indices = indices[~old_u_valid]
                self.u_bank[new_u_indices] = u_proto[~old_u_valid]
            self.u_valid[indices] = True

        self.update_count[indices] += 1
        self.last_epoch[indices] = int(epoch)

    @property
    def sem_dyn_valid(self) -> torch.Tensor:
        return self.sem_valid & self.dyn_valid

    def valid_ratio(self, mask: torch.Tensor) -> float:
        if self.num_items <= 0:
            return 0.0
        return float(mask.float().mean().detach().cpu().item())
