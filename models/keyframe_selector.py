import itertools
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


TRAINING_FREE_STRATEGIES = {
    "t_sas",
    "tsas",
    "per_sas",
    "persas",
    "temporal_stratified_semantic_anchor_selection",
    "plan_evaluate_refine_semantic_anchor_selection",
}


def _canonical_strategy(strategy: str) -> str:
    strategy = str(strategy or "segment_rerank_gumbel_topk").lower()
    if strategy in {"type", "default"}:
        return "segment_rerank_gumbel_topk"
    if strategy in {"tsas", "temporal_stratified_semantic_anchor_selection"}:
        return "t_sas"
    if strategy in {"persas", "plan_evaluate_refine_semantic_anchor_selection"}:
        return "per_sas"
    return strategy


def _get_weight(cfg: Dict | None, keys: Sequence[str], default: float) -> float:
    cfg = cfg or {}
    for key in keys:
        if key in cfg:
            return float(cfg[key])
    return float(default)


def _build_temporal_segments(total_frames: int, num_keyframes: int, segment_size: int = 0) -> List[List[int]]:
    if num_keyframes > total_frames:
        raise ValueError(f"num_keyframes={num_keyframes} cannot exceed T={total_frames}")
    if segment_size > 0 and segment_size * num_keyframes == total_frames:
        return [
            list(range(i * segment_size, (i + 1) * segment_size))
            for i in range(num_keyframes)
        ]

    boundaries = torch.linspace(0, total_frames, num_keyframes + 1).round().long().tolist()
    segments = []
    for i in range(num_keyframes):
        start = int(boundaries[i])
        end = int(boundaries[i + 1])
        if end <= start:
            end = min(start + 1, total_frames)
        segments.append(list(range(start, end)))
    return segments


def _minmax_norm(values: torch.Tensor, dim: int = 1) -> torch.Tensor:
    min_v = values.min(dim=dim, keepdim=True).values
    max_v = values.max(dim=dim, keepdim=True).values
    return (values - min_v) / (max_v - min_v).clamp_min(1e-6)


def _semantic_anchor_quality(
    sim: torch.Tensor,
    segments: Sequence[Sequence[int]],
    score_weights: Dict[str, float],
) -> torch.Tensor:
    b, t, _ = sim.shape
    rg = sim.mean(dim=2)
    rl = torch.zeros(b, t, device=sim.device, dtype=sim.dtype)
    for segment in segments:
        seg = torch.tensor(segment, device=sim.device, dtype=torch.long)
        local = sim.index_select(1, seg).index_select(2, seg).mean(dim=2)
        rl.index_copy_(1, seg, local)

    stability = torch.zeros(b, t, device=sim.device, dtype=sim.dtype)
    if t == 1:
        stability.zero_()
    elif t == 2:
        stability[:, 0] = sim[:, 0, 1]
        stability[:, 1] = sim[:, 0, 1]
    else:
        stability[:, 0] = sim[:, 0, 1]
        stability[:, -1] = sim[:, -2, -1]
        left = sim[:, torch.arange(0, t - 2, device=sim.device), torch.arange(1, t - 1, device=sim.device)]
        right = sim[:, torch.arange(1, t - 1, device=sim.device), torch.arange(2, t, device=sim.device)]
        stability[:, 1:-1] = 0.5 * (left + right)

    rg = _minmax_norm(rg, dim=1)
    rl = _minmax_norm(rl, dim=1)
    stability = _minmax_norm(stability, dim=1)
    return (
        score_weights["global_repr"] * rg
        + score_weights["local_repr"] * rl
        + score_weights["local_stability"] * stability
    )


def _all_segment_combinations(segments: Sequence[Sequence[int]], device: torch.device) -> torch.Tensor:
    combos = list(itertools.product(*segments))
    return torch.tensor(combos, device=device, dtype=torch.long)


def _score_combinations_batched(
    sim: torch.Tensor,
    quality: torch.Tensor,
    combinations: torch.Tensor,
    objective_weights: Dict[str, float],
    chunk_size: int = 2048,
) -> torch.Tensor:
    b, t, _ = sim.shape
    k = combinations.shape[1]
    best_scores = torch.full((b,), -torch.inf, device=sim.device, dtype=sim.dtype)
    best_combo_ids = torch.zeros(b, device=sim.device, dtype=torch.long)
    pair_positions = list(itertools.combinations(range(k), 2))
    chunk_size = max(1, int(chunk_size))

    for start in range(0, combinations.shape[0], chunk_size):
        combo = combinations[start : start + chunk_size]
        c = combo.shape[0]
        flat = combo.reshape(-1)

        selected_sim = sim.index_select(2, flat).reshape(b, t, c, k)
        coverage = selected_sim.max(dim=-1).values.mean(dim=1)
        anchor_quality = quality.index_select(1, flat).reshape(b, c, k).mean(dim=2)

        if pair_positions:
            redundancy = sim.new_zeros((b, c))
            for left, right in pair_positions:
                redundancy = redundancy + sim[:, combo[:, left], combo[:, right]]
            redundancy = redundancy / float(len(pair_positions))
        else:
            redundancy = sim.new_zeros((b, c))

        score = (
            objective_weights["coverage"] * coverage
            + objective_weights["quality"] * anchor_quality
            - objective_weights["redundancy"] * redundancy
        )
        chunk_scores, chunk_ids = score.max(dim=1)
        update = chunk_scores > best_scores
        best_scores = torch.where(update, chunk_scores, best_scores)
        best_combo_ids = torch.where(update, chunk_ids + start, best_combo_ids)

    return torch.sort(combinations.index_select(0, best_combo_ids), dim=1).values


@torch.no_grad()
def per_sas_selector_batch(
    frame_feats: torch.Tensor,
    num_keyframes: int,
    segment_size: int = 0,
    score_weights: Dict[str, float] | None = None,
    set_objective_weights: Dict[str, float] | None = None,
    search_chunk_size: int = 512,
) -> torch.Tensor:
    """Training-free T-SAS/PER-SAS selector.

    Args:
        frame_feats: [B, T, D] frame-level features used only for deterministic
            semantic anchor scoring. The tensor is detached inside this function.
    Returns:
        selected_indices: [B, K], sorted by temporal order, with exactly one
            frame selected from each temporal segment.
    """

    if frame_feats.ndim != 3:
        raise ValueError(f"Expected frame_feats [B, T, D], got {tuple(frame_feats.shape)}")
    _, total_frames, _ = frame_feats.shape
    segments = _build_temporal_segments(total_frames, int(num_keyframes), int(segment_size))
    score_weights = {
        "global_repr": _get_weight(score_weights, ("global_repr", "global", "global_representativeness"), 0.4),
        "local_repr": _get_weight(score_weights, ("local_repr", "local", "local_representativeness"), 0.5),
        "local_stability": _get_weight(score_weights, ("local_stability", "stability"), 0.1),
    }
    objective_weights = {
        "coverage": _get_weight(set_objective_weights, ("coverage", "cov"), 0.6),
        "quality": _get_weight(set_objective_weights, ("quality", "qua"), 0.3),
        "redundancy": _get_weight(set_objective_weights, ("redundancy", "red"), 0.1),
    }

    x = F.normalize(frame_feats.detach().float(), dim=-1)
    sim = torch.bmm(x, x.transpose(1, 2)).clamp_min(0.0)
    quality = _semantic_anchor_quality(sim, segments, score_weights)
    combinations = _all_segment_combinations(segments, device=sim.device)
    return _score_combinations_batched(
        sim,
        quality,
        combinations,
        objective_weights,
        chunk_size=search_chunk_size,
    )


class KeyFrameSelector(nn.Module):
    """Key-frame selector.

    Input:
        x: [B, T, D] scoring features
    Output:
        selected_x: [B, K, D]
        selected_indices: [B, K] in ascending temporal order
        selected_mask: [B, T] with 1 for selected key frames

    Supported strategies:
        segment_rerank_gumbel_topk: trainable legacy selector.
        t_sas / per_sas: training-free temporal-stratified semantic anchor
        selection with exact set-level evaluation.
    """

    def __init__(
        self,
        d_model: int,
        num_keyframes: int = 5,
        num_frames: int = 0,
        strategy: str = "segment_rerank_gumbel_topk",
        temperature: float = 1.0,
        use_straight_through: bool = True,
        candidate_topm: int = 2,
        alpha_motion: float = 0.10,
        beta_redundancy: float = 0.05,
        gamma_coverage: float = 0.05,
        segment_size: int = 0,
        trainable: bool = True,
        score_weights: Dict | None = None,
        set_objective_weights: Dict | None = None,
        search_chunk_size: int = 512,
    ):
        super().__init__()
        strategy = _canonical_strategy(strategy)
        if strategy != "segment_rerank_gumbel_topk" and strategy not in TRAINING_FREE_STRATEGIES:
            raise ValueError(
                "keyframe_selector.strategy must be one of "
                "'segment_rerank_gumbel_topk', 't_sas', or 'per_sas'."
            )
        self.num_keyframes = num_keyframes
        self.num_frames = int(num_frames)
        self.strategy = strategy
        self.is_training_free = strategy in TRAINING_FREE_STRATEGIES
        self.trainable = bool(trainable) and not self.is_training_free
        self.temperature = temperature
        self.use_straight_through = use_straight_through
        self.candidate_topm = max(1, int(candidate_topm))
        self.alpha_motion = float(alpha_motion)
        self.beta_redundancy = float(beta_redundancy)
        self.gamma_coverage = float(gamma_coverage)
        self.segment_size = int(segment_size)
        self.score_weights = score_weights or {}
        self.set_objective_weights = set_objective_weights or {}
        self.search_chunk_size = int(search_chunk_size)
        if self.is_training_free:
            self.grade_net = None
        else:
            hidden = max(d_model // 2, 1)
            self.grade_net = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )

    def _add_gumbel_noise(self, scores: torch.Tensor, training: bool) -> torch.Tensor:
        if not training:
            return scores
        eps = 1e-6
        noise = -torch.log(-torch.log(torch.rand_like(scores).clamp_min(eps)).clamp_min(eps))
        return scores + noise * self.temperature

    @staticmethod
    def _minmax_norm(values: torch.Tensor, dim: int = 1) -> torch.Tensor:
        min_v = values.min(dim=dim, keepdim=True).values
        max_v = values.max(dim=dim, keepdim=True).values
        return (values - min_v) / (max_v - min_v).clamp_min(1e-6)

    def _motion_scores(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        motion = torch.zeros(b, t, device=x.device, dtype=x.dtype)
        if t > 1:
            motion[:, 1:] = (x[:, 1:] - x[:, :-1]).norm(dim=-1)
        return self._minmax_norm(motion, dim=1)

    def _segment_candidates(self, scores: torch.Tensor, training: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return candidate indices and candidate segment ids.

        Args:
            scores: [B, T]
        Returns:
            candidate_indices: [B, N]
            candidate_segments: [N]
        """
        rerank_scores = self._add_gumbel_noise(scores, training)
        _, t = scores.shape
        boundaries = torch.linspace(0, t, self.num_keyframes + 1, device=scores.device).round().long()
        candidate_indices = []
        candidate_segments = []
        for i in range(self.num_keyframes):
            start = int(boundaries[i].item())
            end = int(boundaries[i + 1].item())
            if end <= start:
                end = min(start + 1, t)
            topm = min(self.candidate_topm, end - start)
            local = rerank_scores[:, start:end].topk(k=topm, dim=1).indices + start
            candidate_indices.append(local)
            candidate_segments.append(torch.full((topm,), i, device=scores.device, dtype=torch.long))
        return torch.cat(candidate_indices, dim=1), torch.cat(candidate_segments, dim=0)

    def _select_with_segment_reranking(self, x: torch.Tensor, scores: torch.Tensor, training: bool) -> torch.Tensor:
        """Coverage-constrained candidate reranking selector for E13-K2."""
        b, _, d = x.shape
        candidate_indices, candidate_segments = self._segment_candidates(scores, training)
        num_candidates = candidate_indices.shape[1]
        gather_index = candidate_indices.unsqueeze(-1).expand(-1, -1, d)
        candidate_x = torch.gather(x, dim=1, index=gather_index)

        semantic = torch.gather(scores, dim=1, index=candidate_indices)
        semantic = self._minmax_norm(semantic, dim=1)
        motion = torch.gather(self._motion_scores(x), dim=1, index=candidate_indices)
        base_score = semantic + self.alpha_motion * motion

        candidate_feat = F.normalize(candidate_x, dim=-1)
        selected_mask = torch.zeros(b, num_candidates, device=x.device, dtype=torch.bool)
        covered_segments = torch.zeros(b, self.num_keyframes, device=x.device, dtype=torch.bool)
        selected_positions = []

        for _ in range(self.num_keyframes):
            if selected_positions:
                selected_pos = torch.stack(selected_positions, dim=1)
                selected_feat = torch.gather(
                    candidate_feat,
                    dim=1,
                    index=selected_pos.unsqueeze(-1).expand(-1, -1, d),
                )
                redundancy = torch.einsum("bnd,bsd->bns", candidate_feat, selected_feat).max(dim=-1).values
            else:
                redundancy = torch.zeros_like(base_score)

            covered_for_candidate = covered_segments[:, candidate_segments]
            coverage_bonus = (~covered_for_candidate).to(base_score.dtype) * self.gamma_coverage
            rerank_score = base_score - self.beta_redundancy * redundancy + coverage_bonus
            rerank_score = rerank_score.masked_fill(selected_mask, torch.finfo(rerank_score.dtype).min)

            chosen = rerank_score.argmax(dim=1)
            selected_positions.append(chosen)
            selected_mask.scatter_(1, chosen.unsqueeze(1), True)
            chosen_segments = candidate_segments[chosen]
            covered_segments.scatter_(1, chosen_segments.unsqueeze(1), True)

        selected_pos = torch.stack(selected_positions, dim=1)
        selected_indices = torch.gather(candidate_indices, dim=1, index=selected_pos)
        return torch.sort(selected_indices, dim=1).values

    def forward(
        self,
        x: torch.Tensor,
        select_from: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x [B, T, D], got {tuple(x.shape)}")
        if select_from is None:
            select_from = x
        if select_from.ndim != 3:
            raise ValueError(f"Expected select_from [B, T, D], got {tuple(select_from.shape)}")
        if select_from.shape[:2] != x.shape[:2]:
            raise ValueError("x and select_from must have the same [B, T] dimensions.")
        b, t, d = select_from.shape
        if self.num_keyframes > t:
            raise ValueError(f"num_keyframes={self.num_keyframes} cannot exceed T={t}")

        if self.is_training_free:
            selected_indices = per_sas_selector_batch(
                x,
                num_keyframes=self.num_keyframes,
                segment_size=self.segment_size,
                score_weights=self.score_weights,
                set_objective_weights=self.set_objective_weights,
                search_chunk_size=self.search_chunk_size,
            )
        else:
            scores = self.grade_net(x).squeeze(-1)
            selected_indices = self._select_with_segment_reranking(x, scores, self.training)

        gather_index = selected_indices.unsqueeze(-1).expand(-1, -1, d)
        selected_x = torch.gather(select_from, dim=1, index=gather_index)

        selected_mask = torch.zeros(b, t, device=select_from.device, dtype=select_from.dtype)
        selected_mask.scatter_(1, selected_indices, 1.0)

        if not self.is_training_free and self.training and self.use_straight_through:
            probs = F.softmax(scores / max(self.temperature, 1e-6), dim=1)
            soft_context = torch.einsum("bt,btd->bd", probs, x).unsqueeze(1)
            selected_x = selected_x + (soft_context - soft_context.detach())

        return selected_x, selected_indices, selected_mask
