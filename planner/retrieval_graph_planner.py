from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from memory import PlannerMemoryBank


def _safe_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    finite = torch.isfinite(values)
    if not finite.any():
        return 0.0
    return float(values[finite].mean().detach().cpu().item())


def _safe_std(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    finite = torch.isfinite(values)
    if finite.sum() <= 1:
        return 0.0
    return float(values[finite].std(unbiased=False).detach().cpu().item())


def _topk(sim: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if sim.shape[1] == 0 or k <= 0:
        empty_values = sim.new_empty(sim.shape[0], 0)
        empty_indices = torch.empty(sim.shape[0], 0, dtype=torch.long, device=sim.device)
        return empty_values, empty_indices
    k = min(int(k), sim.shape[1])
    return torch.topk(sim, k=k, dim=1)


def _row_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 or right.numel() == 0:
        return 0.0
    hits = (left.unsqueeze(-1) == right.unsqueeze(1)).any(dim=-1).float()
    return float(hits.mean().detach().cpu().item())


class RetrievalGraphPlanner:
    """Planner Graph builder used for Stage 2 sanity diagnostics."""

    def __init__(
        self,
        top_m: int = 20,
        omega_s: float = 0.65,
        omega_t: float = 0.35,
        omega_z: float = 0.0,
        random_anchors: int = 40,
        include_z_metrics: bool = True,
    ):
        self.top_m = int(top_m)
        self.omega_s = float(omega_s)
        self.omega_t = float(omega_t)
        self.omega_z = float(omega_z)
        self.random_anchors = int(random_anchors)
        self.include_z_metrics = bool(include_z_metrics)

    @classmethod
    def from_config(cls, cfg: Dict) -> "RetrievalGraphPlanner":
        return cls(
            top_m=int(cfg.get("top_m", 20)),
            omega_s=float(cfg.get("omega_s", 0.65)),
            omega_t=float(cfg.get("omega_t", 0.35)),
            omega_z=float(cfg.get("omega_z", 0.0)),
            random_anchors=int(cfg.get("random_anchors", cfg.get("random_samples", 40))),
            include_z_metrics=bool(cfg.get("include_z_metrics", True)),
        )

    def _candidate_indices(self, valid_mask: torch.Tensor) -> torch.Tensor:
        return torch.nonzero(valid_mask, as_tuple=False).flatten()

    def _mask_self(self, sim: torch.Tensor, anchor_indices: torch.Tensor, candidates: torch.Tensor):
        if sim.numel() == 0:
            return
        self_mask = candidates.unsqueeze(0) == anchor_indices.unsqueeze(1)
        sim[self_mask] = -float("inf")

    def _component_sim(
        self,
        anchor_indices: torch.Tensor,
        anchor_bank: torch.Tensor,
        candidate_bank: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        anchors = anchor_bank[anchor_indices]
        candidates = candidate_bank[candidate_indices]
        return torch.clamp(anchors @ candidates.t(), min=0.0)

    def _final_scores(
        self,
        memory: PlannerMemoryBank,
        anchors: torch.Tensor,
        candidates: torch.Tensor,
        omega_s: Optional[float] = None,
        omega_t: Optional[float] = None,
        omega_z: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        p_s = self._component_sim(anchors, memory.sem_proto_bank, memory.sem_proto_bank, candidates)
        p_t = self._component_sim(anchors, memory.dyn_proto_bank, memory.dyn_proto_bank, candidates)
        self._mask_self(p_s, anchors, candidates)
        self._mask_self(p_t, anchors, candidates)
        use_omega_s = self.omega_s if omega_s is None else float(omega_s)
        use_omega_t = self.omega_t if omega_t is None else float(omega_t)
        use_omega_z = self.omega_z if omega_z is None else float(omega_z)
        p_final = use_omega_s * p_s + use_omega_t * p_t
        p_z: Optional[torch.Tensor] = None
        if use_omega_z > 0:
            p_z = self._component_sim(anchors, memory.z_bank, memory.z_bank, candidates)
            self._mask_self(p_z, anchors, candidates)
            p_final = p_final + use_omega_z * p_z
        return p_final, p_s, p_t, p_z

    @torch.no_grad()
    def compute_sanity(
        self,
        memory: PlannerMemoryBank,
        anchor_indices: torch.Tensor,
    ) -> Dict[str, float]:
        anchors = anchor_indices.detach().long().to(memory.device)
        sem_dyn_valid = memory.sem_dyn_valid
        z_valid = memory.z_valid
        final_valid = sem_dyn_valid & z_valid if self.omega_z > 0 else sem_dyn_valid
        final_candidates = self._candidate_indices(final_valid)

        metrics = {
            "planner_valid_sem_dyn": memory.valid_ratio(sem_dyn_valid),
            "planner_valid_z": memory.valid_ratio(z_valid),
            "planner_valid_final": memory.valid_ratio(final_valid),
            "planner_omega_s": self.omega_s,
            "planner_omega_t": self.omega_t,
            "planner_omega_z": self.omega_z,
        }
        if final_candidates.numel() <= 1:
            return metrics

        p_final, p_s, p_t, _ = self._final_scores(memory, anchors, final_candidates)

        if self.include_z_metrics and memory.z_bank is not None:
            z_candidates = self._candidate_indices(z_valid)
            if z_candidates.numel() > 1:
                p_z_metric = self._component_sim(anchors, memory.z_bank, memory.z_bank, z_candidates)
                self._mask_self(p_z_metric, anchors, z_candidates)
                z_top_values, _ = _topk(p_z_metric, self.top_m)
                metrics["planner_p_z_topm"] = _safe_mean(z_top_values)

        s_top_values, s_top_local = _topk(p_s, self.top_m)
        t_top_values, t_top_local = _topk(p_t, self.top_m)
        f_top_values, f_top_local = _topk(p_final, self.top_m)
        n_s = final_candidates[s_top_local]
        n_t = final_candidates[t_top_local]
        n_final = final_candidates[f_top_local]

        metrics.update(
            {
                "planner_p_s_topm": _safe_mean(s_top_values),
                "planner_p_t_topm": _safe_mean(t_top_values),
                "planner_p_final_topm": _safe_mean(f_top_values),
                "planner_p_final_std": _safe_std(f_top_values),
                "planner_overlap_s_t": _row_overlap(n_s, n_t),
                "planner_overlap_final_s": _row_overlap(n_final, n_s),
                "planner_overlap_final_t": _row_overlap(n_final, n_t),
            }
        )
        metrics["planner_p_random"] = self._random_final_mean(anchors, final_candidates, p_final)
        label_precision = self._label_precision(memory.labels, anchors, n_final)
        if label_precision is not None:
            metrics["planner_label_precision_topm"] = label_precision
        return metrics

    @torch.no_grad()
    def static_arf_targets(
        self,
        memory: PlannerMemoryBank,
        anchor_indices: torch.Tensor,
        omega_s: Optional[float] = None,
        omega_t: Optional[float] = None,
        omega_z: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build fixed-size Static ARF targets for a batch.

        Returns indices and soft graph targets for S_i = N_i union R_i. The
        union is represented as concatenation with a mask; duplicate samples do
        not change the semantics materially and keep the implementation dense.
        """

        anchors = anchor_indices.detach().long().to(memory.device)
        if memory.u_bank is None:
            raise ValueError("Static ARF requires memory.u_bank; pass u_a/u_b to PlannerMemoryBank.update_batch.")
        use_omega_z = self.omega_z if omega_z is None else float(omega_z)
        final_valid = memory.sem_dyn_valid & memory.u_valid
        if use_omega_z > 0:
            final_valid = final_valid & memory.z_valid
        candidates = self._candidate_indices(final_valid)
        if candidates.numel() <= 1:
            empty_idx = torch.empty(anchors.shape[0], 0, dtype=torch.long, device=memory.device)
            empty_val = torch.empty(anchors.shape[0], 0, dtype=torch.float32, device=memory.device)
            empty_mask = torch.empty(anchors.shape[0], 0, dtype=torch.bool, device=memory.device)
            return {"target_indices": empty_idx, "target_scores": empty_val, "target_mask": empty_mask}

        p_final, _, _, _ = self._final_scores(
            memory,
            anchors,
            candidates,
            omega_s=omega_s,
            omega_t=omega_t,
            omega_z=omega_z,
        )
        top_values, top_local = _topk(p_final, self.top_m)
        top_indices = candidates[top_local]
        top_mask = torch.isfinite(top_values)

        random_count = max(0, int(self.random_anchors))
        if random_count > 0:
            random_local = torch.randint(
                low=0,
                high=candidates.numel(),
                size=(anchors.shape[0], random_count),
                device=memory.device,
            )
            random_indices = candidates[random_local]
            random_values = p_final.gather(dim=1, index=random_local)
            random_mask = random_indices != anchors.unsqueeze(1)
            random_mask = random_mask & torch.isfinite(random_values)
            target_indices = torch.cat([top_indices, random_indices], dim=1)
            target_scores = torch.cat([top_values, random_values], dim=1)
            target_mask = torch.cat([top_mask, random_mask], dim=1)
        else:
            target_indices = top_indices
            target_scores = top_values
            target_mask = top_mask

        target_scores = torch.clamp(torch.nan_to_num(target_scores, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
        return {
            "target_indices": target_indices,
            "target_scores": target_scores,
            "target_mask": target_mask,
        }

    @torch.no_grad()
    def arf_trace_targets(
        self,
        memory: PlannerMemoryBank,
        anchor_indices: torch.Tensor,
        query_u: torch.Tensor,
        top_r: int = 20,
        random_anchors: Optional[int] = None,
        use_actual_trace: bool = True,
        omega_s: Optional[float] = None,
        omega_t: Optional[float] = None,
        omega_z: Optional[float] = None,
        eta_missed: float = 1.0,
        eta_false: float = 1.0,
        weight_clip: float = 3.0,
    ) -> Dict[str, torch.Tensor]:
        """Build Full ARF targets for S_i = N_i union A_i union R_i.

        N_i are planner top-M neighbors, A_i are actual Hamming retrieval
        traces from current view hash codes against memory.u_bank, and R_i are
        random anchors. The returned dense representation may contain
        duplicates, but membership masks are computed against full N_i/A_i
        sets so missed/false feedback still follows the ARF definition.
        """

        anchors = anchor_indices.detach().long().to(memory.device)
        if memory.u_bank is None:
            raise ValueError("Full ARF requires memory.u_bank; pass u_a/u_b to PlannerMemoryBank.update_batch.")

        use_omega_z = self.omega_z if omega_z is None else float(omega_z)
        final_valid = memory.sem_dyn_valid & memory.u_valid
        if use_omega_z > 0:
            final_valid = final_valid & memory.z_valid
        candidates = self._candidate_indices(final_valid)
        if candidates.numel() <= 1:
            empty_idx = torch.empty(anchors.shape[0], 0, dtype=torch.long, device=memory.device)
            empty_val = torch.empty(anchors.shape[0], 0, dtype=torch.float32, device=memory.device)
            empty_mask = torch.empty(anchors.shape[0], 0, dtype=torch.bool, device=memory.device)
            return {
                "target_indices": empty_idx,
                "target_scores": empty_val,
                "target_mask": empty_mask,
                "target_weights": empty_val,
                "planned_indices": empty_idx,
                "actual_indices": empty_idx,
                "metric_actual_overlap": torch.zeros((), device=memory.device),
                "metric_false_ratio": torch.zeros((), device=memory.device),
                "metric_missed_ratio": torch.zeros((), device=memory.device),
                "metric_retrieved_target_mean": torch.zeros((), device=memory.device),
                "metric_feedback_weight_mean": torch.zeros((), device=memory.device),
            }

        p_final, _, _, _ = self._final_scores(
            memory,
            anchors,
            candidates,
            omega_s=omega_s,
            omega_t=omega_t,
            omega_z=omega_z,
        )
        planned_values, planned_local = _topk(p_final, self.top_m)
        planned_indices = candidates[planned_local]
        planned_mask = torch.isfinite(planned_values)

        actual_count = max(0, int(top_r)) if use_actual_trace else 0
        if actual_count > 0:
            query = query_u.detach().float().to(memory.device)
            query_bits = torch.sign(query)
            query_bits[query_bits == 0] = 1
            memory_bits = torch.sign(memory.u_bank[candidates].detach().float())
            memory_bits[memory_bits == 0] = 1
            trace_sim = query_bits @ memory_bits.t()
            self._mask_self(trace_sim, anchors, candidates)
            actual_values, actual_local = _topk(trace_sim, actual_count)
            actual_indices = candidates[actual_local]
            actual_mask = torch.isfinite(actual_values)
            actual_scores = p_final.gather(dim=1, index=actual_local)
        else:
            actual_indices = torch.empty(anchors.shape[0], 0, dtype=torch.long, device=memory.device)
            actual_scores = torch.empty(anchors.shape[0], 0, dtype=torch.float32, device=memory.device)
            actual_mask = torch.empty(anchors.shape[0], 0, dtype=torch.bool, device=memory.device)

        random_count = max(0, int(self.random_anchors if random_anchors is None else random_anchors))
        if random_count > 0:
            random_local = torch.randint(
                low=0,
                high=candidates.numel(),
                size=(anchors.shape[0], random_count),
                device=memory.device,
            )
            random_indices = candidates[random_local]
            random_scores = p_final.gather(dim=1, index=random_local)
            random_mask = random_indices != anchors.unsqueeze(1)
            random_mask = random_mask & torch.isfinite(random_scores)
        else:
            random_indices = torch.empty(anchors.shape[0], 0, dtype=torch.long, device=memory.device)
            random_scores = torch.empty(anchors.shape[0], 0, dtype=torch.float32, device=memory.device)
            random_mask = torch.empty(anchors.shape[0], 0, dtype=torch.bool, device=memory.device)

        target_indices = torch.cat([planned_indices, actual_indices, random_indices], dim=1)
        target_scores = torch.cat([planned_values, actual_scores, random_scores], dim=1)
        target_mask = torch.cat([planned_mask, actual_mask, random_mask], dim=1)

        in_planned = (target_indices.unsqueeze(-1) == planned_indices.unsqueeze(1)).any(dim=-1)
        in_actual = (
            (target_indices.unsqueeze(-1) == actual_indices.unsqueeze(1)).any(dim=-1)
            if actual_indices.numel() > 0
            else torch.zeros_like(target_mask)
        )
        missed = in_planned & (~in_actual) & target_mask
        false = in_actual & (~in_planned) & target_mask

        target_scores = torch.clamp(torch.nan_to_num(target_scores, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
        weights = torch.ones_like(target_scores)
        weights = weights + float(eta_missed) * missed.float() * target_scores
        weights = weights + float(eta_false) * false.float() * (1.0 - target_scores)
        if weight_clip > 0:
            weights = torch.clamp(weights, max=float(weight_clip))
        weights = weights * target_mask.float()

        if actual_indices.numel() > 0:
            planned_actual_hits = (planned_indices.unsqueeze(-1) == actual_indices.unsqueeze(1)).any(dim=-1)
            actual_planned_hits = (actual_indices.unsqueeze(-1) == planned_indices.unsqueeze(1)).any(dim=-1)
            overlap = planned_actual_hits[planned_mask].float().mean() if planned_mask.any() else torch.zeros((), device=memory.device)
            false_ratio = (~actual_planned_hits & actual_mask).float().sum() / actual_mask.float().sum().clamp_min(1.0)
            retrieved_mean = actual_scores[actual_mask].mean() if actual_mask.any() else torch.zeros((), device=memory.device)
        else:
            overlap = torch.zeros((), device=memory.device)
            false_ratio = torch.zeros((), device=memory.device)
            retrieved_mean = torch.zeros((), device=memory.device)
        missed_ratio = missed.float().sum() / (in_planned & target_mask).float().sum().clamp_min(1.0)
        feedback_mean = weights[target_mask].mean() if target_mask.any() else torch.zeros((), device=memory.device)

        return {
            "target_indices": target_indices,
            "target_scores": target_scores,
            "target_mask": target_mask,
            "target_weights": weights,
            "planned_indices": planned_indices,
            "actual_indices": actual_indices,
            "metric_actual_overlap": overlap,
            "metric_false_ratio": false_ratio,
            "metric_missed_ratio": missed_ratio,
            "metric_retrieved_target_mean": retrieved_mean,
            "metric_feedback_weight_mean": feedback_mean,
        }

    def _random_final_mean(
        self,
        anchors: torch.Tensor,
        candidates: torch.Tensor,
        p_final: torch.Tensor,
    ) -> float:
        if self.random_anchors <= 0 or candidates.numel() <= 1:
            return 0.0
        draws = torch.randint(
            low=0,
            high=candidates.numel(),
            size=(anchors.shape[0], self.random_anchors),
            device=anchors.device,
        )
        random_candidates = candidates[draws]
        values = p_final.gather(dim=1, index=draws)
        values[random_candidates == anchors.unsqueeze(1)] = float("nan")
        return _safe_mean(values)

    def _label_precision(
        self,
        labels: Optional[torch.Tensor],
        anchors: torch.Tensor,
        neighbors: torch.Tensor,
    ) -> Optional[float]:
        if labels is None or neighbors.numel() == 0:
            return None
        anchor_labels = labels[anchors].float()
        neighbor_labels = labels[neighbors].float()
        valid_anchor = anchor_labels.sum(dim=-1) > 0
        if not valid_anchor.any():
            return None
        hits = (neighbor_labels * anchor_labels.unsqueeze(1)).sum(dim=-1) > 0
        hits = hits[valid_anchor]
        if hits.numel() == 0:
            return None
        return float(hits.float().mean().detach().cpu().item())
