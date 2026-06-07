import torch
from torch import nn
import torch.nn.functional as F


class SemanticMotionFusion(nn.Module):
    """Fuse slow semantic anchor and fast motion feature.

    Inputs:
        h_s: [B, D]
        h_f: [B, D]
    Output:
        z: [B, D]
    """

    def __init__(
        self,
        d_model: int = 768,
        fusion_type: str = "gated",
        dropout: float = 0.1,
        gamma_init: float = 0.1,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        if fusion_type == "concat_mlp":
            self.fusion = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
        elif fusion_type in {"gated", "gated_fusion"}:
            self.gate = nn.Linear(2 * d_model, d_model)
            self.norm = nn.LayerNorm(d_model)
        elif fusion_type in {"residual_slow", "slow_residual", "residual_gated"}:
            self.fast_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.Dropout(dropout),
            )
            self.gate = nn.Sequential(
                nn.LayerNorm(2 * d_model),
                nn.Linear(2 * d_model, d_model),
                nn.Sigmoid(),
            )
            self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
            self.norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")

    def forward(self, h_s: torch.Tensor, h_f: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "concat_mlp":
            return self.fusion(torch.cat([h_s, h_f], dim=-1))
        if self.fusion_type in {"residual_slow", "slow_residual", "residual_gated"}:
            gate = self.gate(torch.cat([h_s, h_f], dim=-1))
            return self.norm(h_s + self.gamma * gate * self.fast_proj(h_f))
        g = torch.sigmoid(self.gate(torch.cat([h_s, h_f], dim=-1)))
        z = g * h_s + (1.0 - g) * h_f
        return self.norm(z)


class FastToSlowLateralFusion(nn.Module):
    """Compress high-resolution fast tokens and inject them into selected tokens."""

    def __init__(
        self,
        d_model: int = 768,
        temperature: float = 4.0,
        gamma_init: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.temperature = max(float(temperature), 1e-4)
        self.update = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        selected_tokens: torch.Tensor,
        fast_tokens: torch.Tensor,
        fast_indices: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> torch.Tensor:
        if selected_tokens.ndim != 3 or fast_tokens.ndim != 3:
            raise ValueError("selected_tokens and fast_tokens must be [B, T, D].")
        if selected_tokens.shape[0] != fast_tokens.shape[0]:
            raise ValueError("selected_tokens and fast_tokens must have the same batch size.")
        if fast_tokens.shape[1] == 0 or selected_tokens.shape[1] == 0:
            return selected_tokens

        dist = (selected_indices.unsqueeze(-1) - fast_indices.unsqueeze(1)).abs().to(dtype=fast_tokens.dtype)
        weights = torch.softmax(-dist / self.temperature, dim=-1)
        lateral = torch.bmm(weights, fast_tokens)
        delta = lateral - selected_tokens
        fused_input = torch.cat([selected_tokens, lateral, delta], dim=-1)
        update = self.update(fused_input)
        gate = self.gate(fused_input)
        return self.norm(selected_tokens + self.gamma * gate * update)


class ContentTimeLateralFusion(nn.Module):
    """Fast-to-slow token injection with semantic content and temporal logits."""

    def __init__(
        self,
        d_model: int = 768,
        temperature: float = 2.0,
        content_temperature: float = 0.5,
        gamma_init: float = 0.1,
        dropout: float = 0.1,
        num_time_buckets: int = 32,
        exclude_self_lateral: bool = False,
    ):
        super().__init__()
        self.temperature = max(float(temperature), 1e-4)
        self.content_temperature = max(float(content_temperature), 1e-4)
        self.num_time_buckets = max(int(num_time_buckets), 1)
        self.exclude_self_lateral = bool(exclude_self_lateral)
        self.query_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model, bias=False))
        self.key_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model, bias=False))
        self.time_bias = nn.Embedding(self.num_time_buckets, 1)
        nn.init.zeros_(self.time_bias.weight)
        self.update = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, d_model),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        selected_tokens: torch.Tensor,
        fast_tokens: torch.Tensor,
        fast_indices: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> torch.Tensor:
        if selected_tokens.ndim != 3 or fast_tokens.ndim != 3:
            raise ValueError("selected_tokens and fast_tokens must be [B, T, D].")
        if selected_tokens.shape[0] != fast_tokens.shape[0]:
            raise ValueError("selected_tokens and fast_tokens must have the same batch size.")
        if fast_tokens.shape[1] == 0 or selected_tokens.shape[1] == 0:
            return selected_tokens

        dist = (selected_indices.unsqueeze(-1) - fast_indices.unsqueeze(1)).abs()
        temporal_logits = -dist.to(dtype=fast_tokens.dtype) / self.temperature
        time_bucket = dist.clamp(max=self.num_time_buckets - 1).long()
        temporal_logits = temporal_logits + self.time_bias(time_bucket).squeeze(-1).to(dtype=fast_tokens.dtype)

        query = F.normalize(self.query_proj(selected_tokens), dim=-1)
        key = F.normalize(self.key_proj(fast_tokens), dim=-1)
        content_logits = torch.bmm(query, key.transpose(1, 2)) / self.content_temperature

        logits = content_logits + temporal_logits
        if self.exclude_self_lateral:
            same_time = selected_indices.unsqueeze(-1) == fast_indices.unsqueeze(1)
            logits = logits.masked_fill(same_time, torch.finfo(logits.dtype).min)

        weights = torch.softmax(logits, dim=-1)
        lateral = torch.bmm(weights, fast_tokens)
        delta = lateral - selected_tokens
        fused_input = torch.cat([selected_tokens, lateral, delta, selected_tokens * lateral], dim=-1)
        update = self.update(fused_input)
        gate = self.gate(fused_input)
        return self.norm(selected_tokens + self.gamma * gate * update)


class SlowResidualOutputFusion(nn.Module):
    """Optional small residual fast readout after token-level lateral fusion."""

    def __init__(
        self,
        d_model: int = 768,
        gamma_init: float = 0.05,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fast_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_s: torch.Tensor, h_f: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([h_s, h_f], dim=-1))
        return self.norm(h_s + self.gamma * gate * self.fast_proj(h_f))
