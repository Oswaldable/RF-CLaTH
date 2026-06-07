from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class KeyFrameSelector(nn.Module):
    """E13-K2 key-frame selector with segment candidate reranking.

    Input:
        x: [B, T, D]
    Output:
        selected_x: [B, K, D]
        selected_indices: [B, K] in ascending temporal order
        selected_mask: [B, T] with 1 for selected key frames

    Strategy:
        segment_rerank_gumbel_topk first collects top-m candidates per segment,
        then greedily reranks them with semantic, motion, redundancy, and
        coverage terms. This is the E13-K2 selector and the only active
        key-frame selector in the current mainline.
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
    ):
        super().__init__()
        if strategy != "segment_rerank_gumbel_topk":
            raise ValueError(
                "Only keyframe_selector.strategy='segment_rerank_gumbel_topk' is "
                "supported in the current mainline. Other selector variants were "
                "removed after fixing E13-K2 as the key-frame selector."
            )
        self.num_keyframes = num_keyframes
        self.num_frames = int(num_frames)
        self.strategy = strategy
        self.temperature = temperature
        self.use_straight_through = use_straight_through
        self.candidate_topm = max(1, int(candidate_topm))
        self.alpha_motion = float(alpha_motion)
        self.beta_redundancy = float(beta_redundancy)
        self.gamma_coverage = float(gamma_coverage)
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x [B, T, D], got {tuple(x.shape)}")
        b, t, d = x.shape
        if self.num_keyframes > t:
            raise ValueError(f"num_keyframes={self.num_keyframes} cannot exceed T={t}")

        scores = self.grade_net(x).squeeze(-1)
        selected_indices = self._select_with_segment_reranking(x, scores, self.training)
        gather_index = selected_indices.unsqueeze(-1).expand(-1, -1, d)
        selected_x = torch.gather(x, dim=1, index=gather_index)

        selected_mask = torch.zeros(b, t, device=x.device, dtype=x.dtype)
        selected_mask.scatter_(1, selected_indices, 1.0)

        if self.training and self.use_straight_through:
            probs = F.softmax(scores / max(self.temperature, 1e-6), dim=1)
            soft_context = torch.einsum("bt,btd->bd", probs, x).unsqueeze(1)
            selected_x = selected_x + (soft_context - soft_context.detach())

        return selected_x, selected_indices, selected_mask
