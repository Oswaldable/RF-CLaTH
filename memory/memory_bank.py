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
    """Memory bank for planner state, routed hash codes, and feedback edges."""

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
        self.u_s_bank = self._empty_bank(0)
        self.u_f_bank = self._empty_bank(0)
        self.sem_valid = torch.zeros(self.num_items, dtype=torch.bool, device=device)
        self.dyn_valid = torch.zeros(self.num_items, dtype=torch.bool, device=device)
        self.z_valid = torch.zeros(self.num_items, dtype=torch.bool, device=device)
        self.u_valid = torch.zeros(self.num_items, dtype=torch.bool, device=device)
        self.u_s_valid = torch.zeros(self.num_items, dtype=torch.bool, device=device)
        self.u_f_valid = torch.zeros(self.num_items, dtype=torch.bool, device=device)
        self.route_alpha = torch.full((self.num_items,), 0.5, dtype=torch.float32, device=device)
        self.route_omega = torch.ones(self.num_items, dtype=torch.float32, device=device)
        self.route_valid = torch.zeros(self.num_items, dtype=torch.bool, device=device)
        self.update_count = torch.zeros(self.num_items, dtype=torch.long, device=device)
        self.last_epoch = torch.zeros(self.num_items, dtype=torch.long, device=device)
        self.labels = labels.to(device) if labels is not None else None
        self.edge_indices: Optional[torch.Tensor] = None
        self.prev_edge_indices: Optional[torch.Tensor] = None
        self.edge_weight: Optional[torch.Tensor] = None
        self.edge_posterior: Optional[torch.Tensor] = None
        self.edge_reliability: Optional[torch.Tensor] = None
        self.edge_decay: Optional[torch.Tensor] = None
        self.edge_last_epoch: Optional[torch.Tensor] = None
        self.edge_flags: Optional[torch.Tensor] = None

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

    def _ensure_u_s_dim(self, dim: int):
        if self.u_s_bank is not None and self.u_s_bank.shape[1] == int(dim):
            return
        self.u_s_bank = torch.zeros(self.num_items, int(dim), dtype=torch.float32, device=self.device)
        self.u_s_valid.zero_()

    def _ensure_u_f_dim(self, dim: int):
        if self.u_f_bank is not None and self.u_f_bank.shape[1] == int(dim):
            return
        self.u_f_bank = torch.zeros(self.num_items, int(dim), dtype=torch.float32, device=self.device)
        self.u_f_valid.zero_()

    def _update_code_bank(
        self,
        bank: torch.Tensor,
        valid: torch.Tensor,
        indices: torch.Tensor,
        values: torch.Tensor,
        momentum: float,
        normalize: bool = False,
    ):
        values = values.detach().float().to(self.device)
        if normalize:
            values = F.normalize(values, dim=-1)
        old_valid = valid[indices]
        if old_valid.any():
            old_indices = indices[old_valid]
            mixed = momentum * bank[old_indices] + (1.0 - momentum) * values[old_valid]
            bank[old_indices] = F.normalize(mixed, dim=-1) if normalize else torch.clamp(mixed, min=-1.0, max=1.0)
        if (~old_valid).any():
            new_indices = indices[~old_valid]
            bank[new_indices] = values[~old_valid]
        valid[indices] = True

    def _ensure_edge_slots(self, slots: int):
        slots = max(1, int(slots))
        if self.edge_indices is not None and self.edge_indices.shape[1] >= slots:
            return

        old_slots = 0 if self.edge_indices is None else int(self.edge_indices.shape[1])
        new_edge_indices = torch.full((self.num_items, slots), -1, dtype=torch.long, device=self.device)
        new_prev_edge_indices = torch.full((self.num_items, slots), -1, dtype=torch.long, device=self.device)
        new_edge_weight = torch.zeros(self.num_items, slots, dtype=torch.float32, device=self.device)
        new_edge_posterior = torch.zeros(self.num_items, slots, dtype=torch.float32, device=self.device)
        new_edge_reliability = torch.zeros(self.num_items, slots, dtype=torch.float32, device=self.device)
        new_edge_decay = torch.zeros(self.num_items, slots, dtype=torch.float32, device=self.device)
        new_edge_last_epoch = torch.zeros(self.num_items, slots, dtype=torch.long, device=self.device)
        new_edge_flags = torch.zeros(self.num_items, slots, dtype=torch.int16, device=self.device)

        if old_slots > 0:
            new_edge_indices[:, :old_slots] = self.edge_indices
            new_prev_edge_indices[:, :old_slots] = self.prev_edge_indices
            new_edge_weight[:, :old_slots] = self.edge_weight
            new_edge_posterior[:, :old_slots] = self.edge_posterior
            new_edge_reliability[:, :old_slots] = self.edge_reliability
            new_edge_decay[:, :old_slots] = self.edge_decay
            new_edge_last_epoch[:, :old_slots] = self.edge_last_epoch
            new_edge_flags[:, :old_slots] = self.edge_flags

        self.edge_indices = new_edge_indices
        self.prev_edge_indices = new_prev_edge_indices
        self.edge_weight = new_edge_weight
        self.edge_posterior = new_edge_posterior
        self.edge_reliability = new_edge_reliability
        self.edge_decay = new_edge_decay
        self.edge_last_epoch = new_edge_last_epoch
        self.edge_flags = new_edge_flags

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
        u_s_a: Optional[torch.Tensor] = None,
        u_s_b: Optional[torch.Tensor] = None,
        u_f_a: Optional[torch.Tensor] = None,
        u_f_b: Optional[torch.Tensor] = None,
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
            self._update_code_bank(self.u_bank, self.u_valid, indices, u_proto, self.u_momentum)
            if u_s_a is not None and u_s_b is not None:
                self._ensure_u_s_dim(u_s_a.shape[-1])
                u_s_proto = 0.5 * (u_s_a.detach().float().to(self.device) + u_s_b.detach().float().to(self.device))
                self._update_code_bank(self.u_s_bank, self.u_s_valid, indices, u_s_proto, self.u_momentum)
            if u_f_a is not None and u_f_b is not None:
                self._ensure_u_f_dim(u_f_a.shape[-1])
                u_f_proto = 0.5 * (u_f_a.detach().float().to(self.device) + u_f_b.detach().float().to(self.device))
                self._update_code_bank(self.u_f_bank, self.u_f_valid, indices, u_f_proto, self.u_momentum)

        self.update_count[indices] += 1
        self.last_epoch[indices] = int(epoch)

    @torch.no_grad()
    def update_agent_actions(
        self,
        sample_indices: torch.Tensor,
        alpha: torch.Tensor,
        omega: torch.Tensor,
        epoch: int,
    ):
        indices = sample_indices.detach().long().to(self.device)
        self.route_alpha[indices] = alpha.detach().float().to(self.device).clamp(0.0, 1.0)
        self.route_omega[indices] = omega.detach().float().to(self.device).clamp_min(0.0)
        self.route_valid[indices] = True
        self.last_epoch[indices] = int(epoch)

    @torch.no_grad()
    def update_feedback_edges(
        self,
        sample_indices: torch.Tensor,
        targets: dict,
        epoch: int,
        targets_b: Optional[dict] = None,
        max_edges: int = 40,
        posterior_momentum: float = 0.80,
        reliability_momentum: float = 0.80,
        edge_decay_gamma: float = 0.98,
        old_edge_reserve_ratio: float = 0.25,
        false_edge_reserve_ratio: float = 0.25,
    ):
        planned_indices = targets.get("planned_indices", None)
        actual_indices = targets.get("actual_indices", None)
        if planned_indices is None or actual_indices is None:
            return
        planned_indices = planned_indices.detach().long().to(self.device)
        actual_indices = actual_indices.detach().long().to(self.device)
        planned_mask = targets.get("planned_mask", torch.ones_like(planned_indices, dtype=torch.bool))
        actual_mask = targets.get("actual_mask", torch.ones_like(actual_indices, dtype=torch.bool))
        planned_mask = planned_mask.detach().bool().to(self.device)
        actual_mask = actual_mask.detach().bool().to(self.device)
        planned_scores = targets.get("planned_scores", torch.ones_like(planned_indices, dtype=torch.float32))
        actual_scores = targets.get("actual_scores", torch.ones_like(actual_indices, dtype=torch.float32))
        planned_scores = planned_scores.detach().float().to(self.device)
        actual_scores = actual_scores.detach().float().to(self.device)
        if targets_b is not None:
            planned_indices_b = targets_b.get("planned_indices", None)
            actual_indices_b = targets_b.get("actual_indices", None)
            if planned_indices_b is not None and actual_indices_b is not None:
                planned_indices_b = planned_indices_b.detach().long().to(self.device)
                actual_indices_b = actual_indices_b.detach().long().to(self.device)
                planned_mask_b = targets_b.get("planned_mask", torch.ones_like(planned_indices_b, dtype=torch.bool))
                actual_mask_b = targets_b.get("actual_mask", torch.ones_like(actual_indices_b, dtype=torch.bool))
                planned_scores_b = targets_b.get("planned_scores", torch.ones_like(planned_indices_b, dtype=torch.float32))
                actual_scores_b = targets_b.get("actual_scores", torch.ones_like(actual_indices_b, dtype=torch.float32))
                planned_indices = torch.cat([planned_indices, planned_indices_b], dim=1)
                actual_indices = torch.cat([actual_indices, actual_indices_b], dim=1)
                planned_mask = torch.cat([planned_mask, planned_mask_b.detach().bool().to(self.device)], dim=1)
                actual_mask = torch.cat([actual_mask, actual_mask_b.detach().bool().to(self.device)], dim=1)
                planned_scores = torch.cat([planned_scores, planned_scores_b.detach().float().to(self.device)], dim=1)
                actual_scores = torch.cat([actual_scores, actual_scores_b.detach().float().to(self.device)], dim=1)

        slots = max(int(max_edges), 1)
        self._ensure_edge_slots(slots)
        slots = int(self.edge_indices.shape[1])
        indices = sample_indices.detach().long().to(self.device)
        posterior_momentum = float(posterior_momentum)
        reliability_momentum = float(reliability_momentum)
        edge_decay_gamma = min(1.0, max(0.0, float(edge_decay_gamma)))
        old_edge_reserve_ratio = min(0.9, max(0.0, float(old_edge_reserve_ratio)))
        false_edge_reserve_ratio = min(0.9, max(0.0, float(false_edge_reserve_ratio)))

        for row, anchor in enumerate(indices):
            planned = planned_indices[row][planned_mask[row]]
            actual = actual_indices[row][actual_mask[row]]
            planned_score = planned_scores[row][planned_mask[row]]
            actual_score = actual_scores[row][actual_mask[row]]
            if planned.numel() == 0 and actual.numel() == 0:
                continue
            merged = torch.cat([planned, actual], dim=0)
            merged = merged[(merged >= 0) & (merged != anchor)]
            if merged.numel() == 0:
                continue
            observed_edges = torch.unique(merged, sorted=False)
            prev_edges = self.edge_indices[anchor].clone()
            prev_weight = self.edge_weight[anchor].clone()
            prev_posterior = self.edge_posterior[anchor].clone()
            prev_reliability = self.edge_reliability[anchor].clone()
            prev_decay = self.edge_decay[anchor].clone()
            prev_last_epoch = self.edge_last_epoch[anchor].clone()
            prev_flags = self.edge_flags[anchor].clone()

            prev_active = prev_edges >= 0
            old_edges = prev_edges[prev_active]
            old_weight_active = prev_weight[prev_active]
            old_posterior_active = prev_posterior[prev_active]
            old_reliability_active = prev_reliability[prev_active]
            old_decay_active = prev_decay[prev_active]
            old_last_active = prev_last_epoch[prev_active]
            old_flags_active = prev_flags[prev_active]
            if old_edges.numel() > 0:
                age = (int(epoch) - old_last_active).clamp_min(0).float()
                old_decay_active = old_decay_active * torch.pow(
                    torch.full_like(age, edge_decay_gamma),
                    age,
                )
                keep_old = old_decay_active > 1e-4
                if observed_edges.numel() > 0:
                    keep_old = keep_old & ~((old_edges.unsqueeze(1) == observed_edges.unsqueeze(0)).any(dim=1))
                old_edges = old_edges[keep_old]
                old_weight_active = old_weight_active[keep_old]
                old_posterior_active = old_posterior_active[keep_old]
                old_reliability_active = old_reliability_active[keep_old]
                old_decay_active = old_decay_active[keep_old]
                old_last_active = old_last_active[keep_old]
                old_flags_active = old_flags_active[keep_old]

            if observed_edges.numel() > 0:
                obs_in_planned = (
                    (observed_edges.unsqueeze(1) == planned.unsqueeze(0)).any(dim=1)
                    if planned.numel() > 0
                    else torch.zeros(observed_edges.numel(), dtype=torch.bool, device=self.device)
                )
                obs_in_actual = (
                    (observed_edges.unsqueeze(1) == actual.unsqueeze(0)).any(dim=1)
                    if actual.numel() > 0
                    else torch.zeros(observed_edges.numel(), dtype=torch.bool, device=self.device)
                )
                obs_success = obs_in_planned & obs_in_actual
                obs_false = obs_in_actual & (~obs_in_planned)
                obs_score = torch.zeros(observed_edges.numel(), dtype=torch.float32, device=self.device)
                if planned.numel() > 0:
                    obs_planned_match = observed_edges.unsqueeze(1) == planned.unsqueeze(0)
                    obs_score = torch.maximum(
                        obs_score,
                        (obs_planned_match.float() * planned_score.unsqueeze(0)).max(dim=1).values,
                    )
                if actual.numel() > 0:
                    obs_actual_match = observed_edges.unsqueeze(1) == actual.unsqueeze(0)
                    obs_score = torch.maximum(
                        obs_score,
                        (obs_actual_match.float() * actual_score.unsqueeze(0)).max(dim=1).values,
                    )
                observed_priority = (
                    obs_score
                    + obs_in_planned.float()
                    + 0.5 * obs_in_actual.float()
                    + obs_success.float()
                    + 1.5 * obs_false.float()
                )
                observed_order = torch.argsort(observed_priority, descending=True)
            else:
                obs_false = torch.empty(0, dtype=torch.bool, device=self.device)
                observed_priority = torch.empty(0, dtype=torch.float32, device=self.device)
                observed_order = torch.empty(0, dtype=torch.long, device=self.device)

            if old_edges.numel() > 0:
                old_false = torch.bitwise_and(old_flags_active, torch.ones_like(old_flags_active) * 8) != 0
                old_positive_priority = old_weight_active * old_posterior_active * old_reliability_active * old_decay_active
                old_false_priority = (
                    (1.0 - old_posterior_active.clamp(0.0, 1.0))
                    * (1.0 - old_reliability_active.clamp(0.0, 1.0))
                    * old_decay_active.clamp(0.0, 1.0)
                    + 1.0
                )
                old_priority = torch.where(old_false, old_false_priority, old_positive_priority)
                old_order = torch.argsort(old_priority, descending=True)
            else:
                old_false = torch.empty(0, dtype=torch.bool, device=self.device)
                old_priority = torch.empty(0, dtype=torch.float32, device=self.device)
                old_order = torch.empty(0, dtype=torch.long, device=self.device)

            selected_observed_idx = torch.empty(0, dtype=torch.long, device=self.device)
            selected_old_idx = torch.empty(0, dtype=torch.long, device=self.device)
            false_reserve = min(int(round(float(slots) * false_edge_reserve_ratio)), slots - 1)
            false_reserve = max(0, false_reserve)
            if false_reserve > 0 and (bool(obs_false.any().item()) or bool(old_false.any().item())):
                false_sources = []
                false_positions = []
                false_priorities = []
                if observed_edges.numel() > 0 and bool(obs_false.any().item()):
                    obs_pos = torch.nonzero(obs_false, as_tuple=False).flatten()
                    false_sources.append(torch.zeros_like(obs_pos))
                    false_positions.append(obs_pos)
                    false_priorities.append(observed_priority.index_select(0, obs_pos))
                if old_edges.numel() > 0 and bool(old_false.any().item()):
                    old_pos = torch.nonzero(old_false, as_tuple=False).flatten()
                    false_sources.append(torch.ones_like(old_pos))
                    false_positions.append(old_pos)
                    false_priorities.append(old_priority.index_select(0, old_pos))
                source_cat = torch.cat(false_sources, dim=0)
                position_cat = torch.cat(false_positions, dim=0)
                priority_cat = torch.cat(false_priorities, dim=0)
                false_order = torch.argsort(priority_cat, descending=True)[:false_reserve]
                chosen_source = source_cat.index_select(0, false_order)
                chosen_position = position_cat.index_select(0, false_order)
                selected_observed_idx = chosen_position[chosen_source == 0]
                selected_old_idx = chosen_position[chosen_source == 1]

            observed_selected_mask = torch.zeros(observed_edges.numel(), dtype=torch.bool, device=self.device)
            old_selected_mask = torch.zeros(old_edges.numel(), dtype=torch.bool, device=self.device)
            if selected_observed_idx.numel() > 0:
                observed_selected_mask[selected_observed_idx] = True
            if selected_old_idx.numel() > 0:
                old_selected_mask[selected_old_idx] = True

            remaining_slots = max(0, slots - int(selected_observed_idx.numel()) - int(selected_old_idx.numel()))
            observed_available_order = observed_order[~observed_selected_mask[observed_order]]
            old_available_order = old_order[~old_selected_mask[old_order]]
            old_reserve = min(
                int(old_available_order.numel()),
                int(round(float(slots) * old_edge_reserve_ratio)),
                remaining_slots,
            )
            if observed_available_order.numel() > 0 and old_reserve >= remaining_slots and remaining_slots > 0:
                old_reserve = remaining_slots - 1
            observed_keep = min(int(observed_available_order.numel()), max(0, remaining_slots - old_reserve))
            if observed_keep > 0:
                extra_observed = observed_available_order[:observed_keep]
                selected_observed_idx = torch.cat([selected_observed_idx, extra_observed], dim=0)
                observed_selected_mask[extra_observed] = True
                remaining_slots -= observed_keep

            old_keep = min(int(old_available_order.numel()), remaining_slots)
            if old_keep > 0:
                extra_old = old_available_order[:old_keep]
                selected_old_idx = torch.cat([selected_old_idx, extra_old], dim=0)
                old_selected_mask[extra_old] = True
                remaining_slots -= old_keep

            if remaining_slots > 0 and observed_edges.numel() > int(observed_selected_mask.sum().item()):
                observed_available_order = observed_order[~observed_selected_mask[observed_order]]
                fill_observed = observed_available_order[:remaining_slots]
                selected_observed_idx = torch.cat([selected_observed_idx, fill_observed], dim=0)
                observed_selected_mask[fill_observed] = True
                remaining_slots -= int(fill_observed.numel())
            if remaining_slots > 0 and old_edges.numel() > int(old_selected_mask.sum().item()):
                old_available_order = old_order[~old_selected_mask[old_order]]
                fill_old = old_available_order[:remaining_slots]
                selected_old_idx = torch.cat([selected_old_idx, fill_old], dim=0)
                old_selected_mask[fill_old] = True

            selected_observed = (
                observed_edges.index_select(0, selected_observed_idx)
                if selected_observed_idx.numel() > 0
                else observed_edges[:0]
            )
            selected_old = old_edges.index_select(0, selected_old_idx) if selected_old_idx.numel() > 0 else old_edges[:0]
            if selected_old.numel() > 0:
                current = torch.cat([selected_observed, selected_old], dim=0)
            else:
                current = selected_observed
            count = int(current.numel())

            self.prev_edge_indices[anchor] = prev_edges
            self.edge_indices[anchor].fill_(-1)
            self.edge_weight[anchor].zero_()
            self.edge_posterior[anchor].zero_()
            self.edge_reliability[anchor].zero_()
            self.edge_decay[anchor].zero_()
            self.edge_last_epoch[anchor].zero_()
            self.edge_flags[anchor].zero_()

            current_is_observed = (
                (current.unsqueeze(1) == observed_edges.unsqueeze(0)).any(dim=1)
                if observed_edges.numel() > 0
                else torch.zeros(count, dtype=torch.bool, device=self.device)
            )
            in_planned = (current.unsqueeze(1) == planned.unsqueeze(0)).any(dim=1) if planned.numel() > 0 else torch.zeros(count, dtype=torch.bool, device=self.device)
            in_actual = (current.unsqueeze(1) == actual.unsqueeze(0)).any(dim=1) if actual.numel() > 0 else torch.zeros(count, dtype=torch.bool, device=self.device)
            missed = in_planned & (~in_actual)
            false = in_actual & (~in_planned)
            success = in_planned & in_actual

            p_target = torch.full((count,), 0.35, dtype=torch.float32, device=self.device)
            p_target = torch.where(in_planned, torch.full_like(p_target, 0.65), p_target)
            p_target = torch.where(success, torch.full_like(p_target, 0.90), p_target)
            p_target = torch.where(missed, torch.full_like(p_target, 0.55), p_target)
            p_target = torch.where(false, torch.full_like(p_target, 0.10), p_target)
            r_target = torch.full((count,), 0.30, dtype=torch.float32, device=self.device)
            r_target = torch.where(in_planned, torch.full_like(r_target, 0.60), r_target)
            r_target = torch.where(success, torch.full_like(r_target, 1.00), r_target)
            r_target = torch.where(missed, torch.full_like(r_target, 0.50), r_target)
            r_target = torch.where(false, torch.full_like(r_target, 0.05), r_target)

            score_target = torch.zeros(count, dtype=torch.float32, device=self.device)
            if planned.numel() > 0:
                planned_match = current.unsqueeze(1) == planned.unsqueeze(0)
                planned_value = (planned_match.float() * planned_score.unsqueeze(0)).max(dim=1).values
                score_target = torch.maximum(score_target, planned_value)
            if actual.numel() > 0:
                actual_match = current.unsqueeze(1) == actual.unsqueeze(0)
                actual_value = (actual_match.float() * actual_score.unsqueeze(0)).max(dim=1).values
                score_target = torch.maximum(score_target, actual_value)
            score_target = score_target.clamp(0.0, 1.0)

            old_match = current.unsqueeze(1) == prev_edges.unsqueeze(0)
            has_old = old_match.any(dim=1)
            old_weight = (old_match.float() * prev_weight.unsqueeze(0)).max(dim=1).values
            old_posterior = (old_match.float() * prev_posterior.unsqueeze(0)).max(dim=1).values
            old_reliability = (old_match.float() * prev_reliability.unsqueeze(0)).max(dim=1).values
            old_decay = (old_match.float() * prev_decay.unsqueeze(0)).max(dim=1).values
            old_last_epoch = (old_match.long() * prev_last_epoch.unsqueeze(0)).max(dim=1).values
            old_flags = (old_match.to(prev_flags.dtype) * prev_flags.unsqueeze(0)).max(dim=1).values
            if has_old.any():
                old_age = (int(epoch) - old_last_epoch).clamp_min(0).float()
                old_decay = torch.where(
                    has_old,
                    old_decay * torch.pow(torch.full_like(old_age, edge_decay_gamma), old_age),
                    old_decay,
                )
            new_weight = torch.where(
                current_is_observed,
                torch.where(has_old, 0.5 * old_weight + 0.5 * score_target, score_target),
                old_weight,
            )
            new_posterior = torch.where(
                current_is_observed,
                torch.where(
                    has_old,
                    posterior_momentum * old_posterior + (1.0 - posterior_momentum) * p_target,
                    p_target,
                ),
                old_posterior,
            )
            new_reliability = torch.where(
                current_is_observed,
                torch.where(
                    has_old,
                    reliability_momentum * old_reliability + (1.0 - reliability_momentum) * r_target,
                    r_target,
                ),
                old_reliability,
            )

            flags = torch.zeros(count, dtype=torch.int16, device=self.device)
            flags = flags + in_planned.to(torch.int16)
            flags = flags + in_actual.to(torch.int16) * 2
            flags = flags + missed.to(torch.int16) * 4
            flags = flags + false.to(torch.int16) * 8
            flags = torch.where(current_is_observed, flags, old_flags)
            self.edge_indices[anchor, :count] = current
            self.edge_weight[anchor, :count] = new_weight
            self.edge_posterior[anchor, :count] = new_posterior
            self.edge_reliability[anchor, :count] = new_reliability
            self.edge_decay[anchor, :count] = torch.where(current_is_observed, torch.ones_like(new_weight), old_decay)
            self.edge_last_epoch[anchor, :count] = torch.full_like(old_last_epoch, int(epoch))
            self.edge_flags[anchor, :count] = flags

    def edge_summary(self, sample_indices: torch.Tensor) -> dict:
        if self.edge_indices is None:
            return {
                "memory_edge_weight": 0.0,
                "memory_edge_posterior": 0.0,
                "memory_edge_reliability": 0.0,
                "memory_edge_decay": 0.0,
                "memory_edge_false_ratio": 0.0,
                "memory_edge_stability": 0.0,
            }
        indices = sample_indices.detach().long().to(self.device)
        edge_indices = self.edge_indices.index_select(0, indices)
        edge_mask = edge_indices >= 0
        if not bool(edge_mask.any().item()):
            return {
                "memory_edge_weight": 0.0,
                "memory_edge_posterior": 0.0,
                "memory_edge_reliability": 0.0,
                "memory_edge_decay": 0.0,
                "memory_edge_false_ratio": 0.0,
                "memory_edge_stability": 0.0,
            }
        weights = self.edge_weight.index_select(0, indices)[edge_mask]
        posterior = self.edge_posterior.index_select(0, indices)[edge_mask]
        reliability = self.edge_reliability.index_select(0, indices)[edge_mask]
        decay = self.edge_decay.index_select(0, indices)[edge_mask]
        flags = self.edge_flags.index_select(0, indices)[edge_mask]
        false_ratio = (torch.bitwise_and(flags, torch.ones_like(flags) * 8) != 0).float().mean()

        prev = self.prev_edge_indices.index_select(0, indices)
        prev_mask = prev >= 0
        intersection = ((edge_indices.unsqueeze(-1) == prev.unsqueeze(1)) & edge_mask.unsqueeze(-1) & prev_mask.unsqueeze(1)).any(dim=-1)
        inter_count = intersection.float().sum(dim=1)
        union_count = edge_mask.float().sum(dim=1) + prev_mask.float().sum(dim=1) - inter_count
        stability = (inter_count / union_count.clamp_min(1.0)).mean()
        return {
            "memory_edge_weight": float(weights.mean().detach().cpu().item()),
            "memory_edge_posterior": float(posterior.mean().detach().cpu().item()),
            "memory_edge_reliability": float(reliability.mean().detach().cpu().item()),
            "memory_edge_decay": float(decay.mean().detach().cpu().item()),
            "memory_edge_false_ratio": float(false_ratio.detach().cpu().item()),
            "memory_edge_stability": float(stability.detach().cpu().item()),
        }

    def global_edge_stability(self) -> float:
        if self.edge_indices is None or self.prev_edge_indices is None:
            return 0.0
        edge_mask = self.edge_indices >= 0
        prev_mask = self.prev_edge_indices >= 0
        active_rows = edge_mask.any(dim=1) | prev_mask.any(dim=1)
        if not bool(active_rows.any().item()):
            return 0.0
        edge_indices = self.edge_indices[active_rows]
        prev = self.prev_edge_indices[active_rows]
        edge_mask = edge_mask[active_rows]
        prev_mask = prev_mask[active_rows]
        intersection = ((edge_indices.unsqueeze(-1) == prev.unsqueeze(1)) & edge_mask.unsqueeze(-1) & prev_mask.unsqueeze(1)).any(dim=-1)
        inter_count = intersection.float().sum(dim=1)
        union_count = edge_mask.float().sum(dim=1) + prev_mask.float().sum(dim=1) - inter_count
        return float((inter_count / union_count.clamp_min(1.0)).mean().detach().cpu().item())

    def edge_factor(
        self,
        anchor_indices: torch.Tensor,
        candidate_indices: torch.Tensor,
        default: float = 1.0,
        boost_scale: float = 1.0,
        min_factor: float = 1.0,
        max_factor: Optional[float] = None,
        current_epoch: Optional[int] = None,
        edge_decay_gamma: float = 1.0,
    ) -> torch.Tensor:
        anchors = anchor_indices.detach().long().to(self.device)
        candidates = candidate_indices.detach().long().to(self.device)
        if candidates.numel() == 0:
            return torch.empty(anchors.numel(), 0, dtype=torch.float32, device=self.device)
        factors = torch.full(
            (anchors.numel(), candidates.numel()),
            float(default),
            dtype=torch.float32,
            device=self.device,
        )
        if self.edge_indices is None:
            return factors
        edge_indices = self.edge_indices.index_select(0, anchors)
        edge_mask = edge_indices >= 0
        if not bool(edge_mask.any().item()):
            return factors
        weight = self.edge_weight.index_select(0, anchors).clamp_min(0.0)
        posterior = self.edge_posterior.index_select(0, anchors).clamp(0.0, 1.0)
        reliability = self.edge_reliability.index_select(0, anchors).clamp(0.0, 1.0)
        decay = self.edge_decay.index_select(0, anchors).clamp(0.0, 1.0)
        if current_epoch is not None:
            last_epoch = self.edge_last_epoch.index_select(0, anchors)
            age = (int(current_epoch) - last_epoch).clamp_min(0).float()
            gamma = min(1.0, max(0.0, float(edge_decay_gamma)))
            decay = decay * torch.pow(torch.full_like(age, gamma), age)
        edge_value = (weight * posterior * reliability * decay).masked_fill(~edge_mask, 0.0)
        match = edge_indices.unsqueeze(1) == candidates.view(1, -1, 1)
        matched = match.any(dim=-1)
        matched_value = (match.float() * edge_value.unsqueeze(1)).max(dim=-1).values
        matched_factor = float(default) + float(boost_scale) * matched_value
        matched_factor = torch.clamp(matched_factor, min=float(min_factor))
        if max_factor is not None and float(max_factor) > 0:
            matched_factor = torch.clamp(matched_factor, max=float(max_factor))
        return torch.where(matched, matched_factor, factors)

    def false_edge_factor(
        self,
        anchor_indices: torch.Tensor,
        candidate_indices: torch.Tensor,
        default: float = 1.0,
        boost_scale: float = 1.0,
        min_factor: float = 1.0,
        max_factor: Optional[float] = None,
        current_epoch: Optional[int] = None,
        edge_decay_gamma: float = 1.0,
    ) -> torch.Tensor:
        anchors = anchor_indices.detach().long().to(self.device)
        candidates = candidate_indices.detach().long().to(self.device)
        if candidates.numel() == 0:
            return torch.empty(anchors.numel(), 0, dtype=torch.float32, device=self.device)
        factors = torch.full(
            (anchors.numel(), candidates.numel()),
            float(default),
            dtype=torch.float32,
            device=self.device,
        )
        if self.edge_indices is None:
            return factors
        edge_indices = self.edge_indices.index_select(0, anchors)
        edge_mask = edge_indices >= 0
        flags = self.edge_flags.index_select(0, anchors)
        false_mask = torch.bitwise_and(flags, torch.ones_like(flags) * 8) != 0
        edge_mask = edge_mask & false_mask
        if not bool(edge_mask.any().item()):
            return factors
        posterior = self.edge_posterior.index_select(0, anchors).clamp(0.0, 1.0)
        reliability = self.edge_reliability.index_select(0, anchors).clamp(0.0, 1.0)
        decay = self.edge_decay.index_select(0, anchors).clamp(0.0, 1.0)
        if current_epoch is not None:
            last_epoch = self.edge_last_epoch.index_select(0, anchors)
            age = (int(current_epoch) - last_epoch).clamp_min(0).float()
            gamma = min(1.0, max(0.0, float(edge_decay_gamma)))
            decay = decay * torch.pow(torch.full_like(age, gamma), age)
        false_value = ((1.0 - posterior) * (1.0 - reliability) * decay).masked_fill(~edge_mask, 0.0)
        match = edge_indices.unsqueeze(1) == candidates.view(1, -1, 1)
        matched = match.any(dim=-1)
        matched_value = (match.float() * false_value.unsqueeze(1)).max(dim=-1).values
        matched_factor = float(default) + float(boost_scale) * matched_value
        matched_factor = torch.clamp(matched_factor, min=float(min_factor))
        if max_factor is not None and float(max_factor) > 0:
            matched_factor = torch.clamp(matched_factor, max=float(max_factor))
        return torch.where(matched, matched_factor, factors)

    @property
    def sem_dyn_valid(self) -> torch.Tensor:
        return self.sem_valid & self.dyn_valid

    def valid_ratio(self, mask: torch.Tensor) -> float:
        if self.num_items <= 0:
            return 0.0
        return float(mask.float().mean().detach().cpu().item())
