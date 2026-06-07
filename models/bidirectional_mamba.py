from typing import Tuple
import warnings

import torch
from torch import nn
import torch.nn.functional as F


class MaskedTemporalAugmentation(nn.Module):
    """Random mask, temporal jitter, frame dropout augmentation.

    Input:
        x: [B, T, D]
    Output:
        augmented_x: [B, T, D]
        mask: [B, T] where True marks replaced positions
    """

    def __init__(
        self,
        d_model: int,
        mask_ratio: float = 0.3,
        use_temporal_jitter: bool = True,
        use_frame_dropout: bool = True,
        frame_dropout_ratio: float = 0.1,
        use_motion_aware_mask: bool = False,
        mask_motion: str = "low",
    ):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mask_ratio = mask_ratio
        self.use_temporal_jitter = use_temporal_jitter
        self.use_frame_dropout = use_frame_dropout
        self.frame_dropout_ratio = frame_dropout_ratio
        self.use_motion_aware_mask = use_motion_aware_mask
        self.mask_motion = mask_motion
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def _temporal_jitter(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < 2:
            return x
        x = x.clone()
        b, t, _ = x.shape
        swap_prob = 0.15
        swaps = torch.rand(b, t - 1, device=x.device) < swap_prob
        for batch_idx in range(b):
            positions = torch.nonzero(swaps[batch_idx], as_tuple=False).flatten()
            for pos in positions.tolist():
                tmp = x[batch_idx, pos].clone()
                x[batch_idx, pos] = x[batch_idx, pos + 1]
                x[batch_idx, pos + 1] = tmp
        return x

    def _random_mask(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        if self.mask_ratio <= 0:
            return torch.zeros(b, t, device=x.device, dtype=torch.bool)
        num_mask = max(1, int(round(t * self.mask_ratio)))
        if self.use_motion_aware_mask and t > 1:
            diff = torch.zeros(b, t, device=x.device)
            diff[:, 1:] = torch.norm(x[:, 1:] - x[:, :-1], dim=-1)
            largest = self.mask_motion == "high"
            return torch.zeros_like(diff, dtype=torch.bool).scatter(
                1, torch.topk(diff, k=num_mask, dim=1, largest=largest).indices, True
            )
        noise = torch.rand(b, t, device=x.device)
        idx = torch.topk(noise, k=num_mask, dim=1).indices
        mask = torch.zeros(b, t, device=x.device, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        return mask

    def _dropout_mask(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        if not self.use_frame_dropout or self.frame_dropout_ratio <= 0:
            return torch.zeros(b, t, device=x.device, dtype=torch.bool)
        return torch.rand(b, t, device=x.device) < self.frame_dropout_ratio

    def forward(self, x: torch.Tensor, deterministic: bool = False, enabled: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        if deterministic or not enabled:
            return x, torch.zeros(x.shape[:2], device=x.device, dtype=torch.bool)
        out = self._temporal_jitter(x) if self.use_temporal_jitter else x.clone()
        mask = self._random_mask(out) | self._dropout_mask(out)
        token = self.mask_token.expand(out.shape[0], out.shape[1], -1)
        out = torch.where(mask.unsqueeze(-1), token, out)
        return out, mask


class MambaLikeBlock(nn.Module):
    """Mamba-like fallback block with the same [B, T, D] -> [B, T, D] interface."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, inner)
        self.gate_proj = nn.Linear(d_model, inner)
        self.conv = nn.Conv1d(
            inner,
            inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=inner,
        )
        self.state_proj = nn.Linear(inner, inner)
        self.out_proj = nn.Linear(inner, d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        u = self.in_proj(h)
        gate = torch.sigmoid(self.gate_proj(h))
        conv = self.conv(u.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        h = F.silu(conv) * gate
        h = self.state_proj(h)
        x = residual + self.dropout(self.out_proj(h))
        x = x + self.dropout(self.ffn(x))
        return x


class OfficialMambaResidualBlock(nn.Module):
    """Residual wrapper around official mamba_ssm.Mamba.

    The official Mamba module is a sequence mixer. This wrapper adds the
    LayerNorm, residual path, dropout, and FFN used by the fallback block so
    swapping between official Mamba and MambaLikeBlock keeps similar training
    behavior.
    """

    def __init__(
        self,
        mamba_cls,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = mamba_cls(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.mixer(self.norm(x)))
        x = x + self.dropout(self.ffn(x))
        return x


def _load_official_mamba():
    errors = []
    try:
        from mamba_ssm import Mamba

        return Mamba, None
    except Exception as exc:  # mamba install/ABI errors often surface as ImportError.
        errors.append(exc)
    try:
        from mamba_ssm.modules.mamba_simple import Mamba

        return Mamba, None
    except Exception as exc:
        errors.append(exc)
    return None, errors[-1] if errors else None


class MambaBlock(nn.Module):
    """Wrapper that can use official mamba_ssm.Mamba when available."""

    _warned_fallback = False

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_official_mamba: bool = False,
        strict_official_mamba: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.uses_official_mamba = False
        if use_official_mamba:
            mamba_cls, import_error = _load_official_mamba()
            if mamba_cls is not None:
                self.block = OfficialMambaResidualBlock(
                    mamba_cls,
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                self.uses_official_mamba = True
            else:
                if strict_official_mamba:
                    raise ImportError(f"Official mamba_ssm.Mamba is not usable: {import_error}") from import_error
                if not MambaBlock._warned_fallback:
                    warnings.warn(
                        "Official mamba_ssm.Mamba was requested but is not usable; "
                        f"falling back to MambaLikeBlock. Original error: {import_error}",
                        RuntimeWarning,
                    )
                    MambaBlock._warned_fallback = True
                self.block = MambaLikeBlock(
                    d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
        else:
            self.block = MambaLikeBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TemporalAttentionPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.einsum("bt,btd->bd", weights, x)


class AnchorAwareTemporalPool(nn.Module):
    """Attention pooling that can score fast tokens against their nearest anchors."""

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, nearest_anchor: torch.Tensor | None = None) -> torch.Tensor:
        if nearest_anchor is None:
            nearest_anchor = torch.zeros_like(x)
        delta = x - nearest_anchor
        weights = torch.softmax(self.score(torch.cat([x, nearest_anchor, delta], dim=-1)).squeeze(-1), dim=1)
        return torch.einsum("bt,btd->bd", weights, x)


class AnchorConditioning(nn.Module):
    """Condition remaining-frame tokens on selected keyframe anchors."""

    def __init__(
        self,
        d_model: int,
        num_frames: int = 30,
        num_segments: int = 6,
        max_relative_distance: int | None = None,
        anchor_scale: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_frames = num_frames
        self.num_segments = max(1, num_segments)
        self.max_relative_distance = max_relative_distance or num_frames
        self.anchor_scale = float(anchor_scale)
        self.rel_embed = nn.Embedding(self.max_relative_distance * 2 + 1, d_model)
        self.segment_embed = nn.Embedding(self.num_segments, d_model)
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def _fallback_indices(self, x: torch.Tensor) -> torch.Tensor:
        return torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)

    def _nearest_anchor(
        self,
        x: torch.Tensor,
        fast_indices: torch.Tensor | None,
        anchors: torch.Tensor | None,
        anchor_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        if fast_indices is None:
            fast_indices = self._fallback_indices(x)
        if anchors is None or anchor_indices is None or anchors.shape[1] == 0:
            nearest_anchor = torch.zeros_like(x)
            nearest_anchor_idx = torch.zeros(b, t, device=x.device, dtype=torch.long)
            return nearest_anchor, nearest_anchor_idx

        temporal_delta = fast_indices.unsqueeze(-1) - anchor_indices.unsqueeze(1)
        nearest_pos = temporal_delta.abs().argmin(dim=-1)
        gather_idx = nearest_pos.unsqueeze(-1).expand(-1, -1, anchors.shape[-1])
        nearest_anchor = anchors.gather(dim=1, index=gather_idx)
        nearest_anchor_idx = anchor_indices.gather(dim=1, index=nearest_pos)
        return nearest_anchor, nearest_anchor_idx

    def forward(
        self,
        x: torch.Tensor,
        fast_indices: torch.Tensor | None = None,
        anchors: torch.Tensor | None = None,
        anchor_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if fast_indices is None:
            fast_indices = self._fallback_indices(x)
        nearest_anchor, nearest_anchor_idx = self._nearest_anchor(x, fast_indices, anchors, anchor_indices)
        rel = (fast_indices - nearest_anchor_idx).clamp(
            min=-self.max_relative_distance,
            max=self.max_relative_distance,
        )
        rel_bucket = (rel + self.max_relative_distance).long()
        seg = torch.div(
            fast_indices.clamp(min=0, max=max(0, self.num_frames - 1)) * self.num_segments,
            max(1, self.num_frames),
            rounding_mode="floor",
        ).clamp(max=self.num_segments - 1)
        rel_h = self.rel_embed(rel_bucket)
        seg_h = self.segment_embed(seg.long())
        delta = x - nearest_anchor
        fused = self.fuse(torch.cat([x, delta, rel_h, seg_h], dim=-1))
        gate = self.gate(torch.cat([x, nearest_anchor, delta], dim=-1))
        return self.out_norm(x + self.anchor_scale * gate * fused), nearest_anchor


class SSDStyleBlock(nn.Module):
    """Lightweight SSD/Mamba-2-style selective recurrent mixer fallback."""

    def __init__(self, d_model: int, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.value_proj = nn.Linear(d_model, inner)
        self.decay_proj = nn.Linear(d_model, inner)
        self.gate_proj = nn.Linear(d_model, inner)
        self.out_proj = nn.Linear(inner, d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def _scan(self, value: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
        state = torch.zeros(value.shape[0], value.shape[2], device=value.device, dtype=value.dtype)
        outputs = []
        for step in range(value.shape[1]):
            state = decay[:, step] * state + (1.0 - decay[:, step]) * value[:, step]
            outputs.append(state)
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        value = torch.tanh(self.value_proj(h))
        decay = torch.sigmoid(self.decay_proj(h))
        gate = torch.sigmoid(self.gate_proj(h))
        mixed = self._scan(value, decay) * gate
        x = residual + self.dropout(self.out_proj(mixed))
        x = x + self.dropout(self.ffn(x))
        return x


class OfficialMamba2ResidualBlock(nn.Module):
    """Residual wrapper around official mamba_ssm.Mamba2 when available."""

    def __init__(self, mamba2_cls, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = mamba2_cls(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.mixer(self.norm(x)))
        x = x + self.dropout(self.ffn(x))
        return x


def _load_official_mamba2():
    errors = []
    for module_name in ("mamba_ssm", "mamba_ssm.modules.mamba2", "mamba_ssm.modules.mamba2_simple"):
        try:
            module = __import__(module_name, fromlist=["Mamba2"])
            return getattr(module, "Mamba2"), None
        except Exception as exc:
            errors.append(exc)
    return None, errors[-1] if errors else None


class SSDMixerBlock(nn.Module):
    """Mamba-2/SSD mixer with official Mamba2 fallback support."""

    _warned_fallback = False

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_official_mamba2: bool = False,
        strict_official_mamba2: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if use_official_mamba2:
            mamba2_cls, import_error = _load_official_mamba2()
            if mamba2_cls is not None:
                try:
                    self.block = OfficialMamba2ResidualBlock(
                        mamba2_cls,
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        dropout=dropout,
                    )
                    return
                except Exception as exc:
                    import_error = exc
            if strict_official_mamba2:
                raise ImportError(f"Official mamba_ssm.Mamba2 is not usable: {import_error}") from import_error
            if not SSDMixerBlock._warned_fallback:
                warnings.warn(
                    "Official mamba_ssm.Mamba2 was requested but is not usable; "
                    f"falling back to SSDStyleBlock. Original error: {import_error}",
                    RuntimeWarning,
                )
                SSDMixerBlock._warned_fallback = True
        self.block = SSDStyleBlock(d_model=d_model, expand=expand, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GatedDeltaBlock(nn.Module):
    """Diagonal gated-delta recurrent mixer inspired by Gated DeltaNet."""

    def __init__(self, d_model: int, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.value_proj = nn.Linear(d_model, inner)
        self.key_proj = nn.Linear(d_model, inner)
        self.beta_proj = nn.Linear(d_model, inner)
        self.gate_proj = nn.Linear(d_model, inner)
        self.out_proj = nn.Linear(inner, d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def _scan(self, value: torch.Tensor, key: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        state = torch.zeros(value.shape[0], value.shape[2], device=value.device, dtype=value.dtype)
        outputs = []
        for step in range(value.shape[1]):
            update = beta[:, step] * key[:, step] * (value[:, step] - state)
            state = state + update
            outputs.append(state)
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        value = torch.tanh(self.value_proj(h))
        key = torch.sigmoid(self.key_proj(h))
        beta = torch.sigmoid(self.beta_proj(h))
        gate = torch.sigmoid(self.gate_proj(h))
        mixed = self._scan(value, key, beta) * gate
        x = residual + self.dropout(self.out_proj(mixed))
        x = x + self.dropout(self.ffn(x))
        return x


class BidirectionalDecayConvBlock(nn.Module):
    """Bidirectional decayed-scan + local-conv mixer (NOT the official Hydra).

    Lightweight flip-concat approximation: independent forward/backward gated decay
    scans plus a depthwise local conv, concatenated (3x inner) and projected back to
    d_model. This is NOT the official Hydra quasiseparable single mixer
    (goombalab/hydra, arXiv 2407.09941); the honest name avoids misrepresenting it
    as Hydra in the paper/code.
    """

    def __init__(self, d_model: int, expand: int = 2, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("BidirectionalDecayConvBlock kernel_size must be odd.")
        inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.value_proj = nn.Linear(d_model, inner)
        self.fwd_decay_proj = nn.Linear(d_model, inner)
        self.bwd_decay_proj = nn.Linear(d_model, inner)
        self.gate_proj = nn.Linear(d_model, inner)
        self.local_conv = nn.Conv1d(
            inner,
            inner,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=inner,
        )
        self.out_proj = nn.Linear(inner * 3, d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def _scan(self, value: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
        state = torch.zeros(value.shape[0], value.shape[2], device=value.device, dtype=value.dtype)
        outputs = []
        for step in range(value.shape[1]):
            state = decay[:, step] * state + (1.0 - decay[:, step]) * value[:, step]
            outputs.append(state)
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        value = torch.tanh(self.value_proj(h))
        gate = torch.sigmoid(self.gate_proj(h))
        h_fwd = self._scan(value, torch.sigmoid(self.fwd_decay_proj(h)))
        rev_value = torch.flip(value, dims=[1])
        rev_decay = torch.flip(torch.sigmoid(self.bwd_decay_proj(h)), dims=[1])
        h_bwd = torch.flip(self._scan(rev_value, rev_decay), dims=[1])
        h_local = F.silu(self.local_conv(value.transpose(1, 2)).transpose(1, 2))
        mixed = torch.cat([h_fwd * gate, h_bwd * gate, h_local * gate], dim=-1)
        x = residual + self.dropout(self.out_proj(mixed))
        x = x + self.dropout(self.ffn(x))
        return x


def _make_pool(pooling: str, d_model: int):
    if pooling == "attention":
        return TemporalAttentionPool(d_model)
    if pooling == "anchor_attention":
        return AnchorAwareTemporalPool(d_model)
    if pooling == "mean":
        return None
    raise ValueError(f"Unsupported pooling: {pooling}")


class AnchorConditionedSSDFastEncoder(nn.Module):
    """Selected-anchor conditioned fast encoder with Mamba-2/SSD-style mixer."""

    def __init__(
        self,
        d_model: int = 768,
        num_frames: int = 30,
        num_segments: int = 6,
        depth: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = True,
        pooling: str = "anchor_attention",
        use_official_mamba2: bool = False,
        strict_official_mamba2: bool = False,
        anchor_scale: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.condition = AnchorConditioning(
            d_model=d_model,
            num_frames=num_frames,
            num_segments=num_segments,
            anchor_scale=anchor_scale,
            dropout=dropout,
        )
        self.forward_blocks = nn.ModuleList(
            [
                SSDMixerBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    use_official_mamba2=use_official_mamba2,
                    strict_official_mamba2=strict_official_mamba2,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        if bidirectional:
            self.backward_blocks = nn.ModuleList(
                [
                    SSDMixerBlock(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        use_official_mamba2=use_official_mamba2,
                        strict_official_mamba2=strict_official_mamba2,
                        dropout=dropout,
                    )
                    for _ in range(depth)
                ]
            )
            self.proj = nn.Linear(d_model * 2, d_model)
        else:
            self.backward_blocks = None
            self.proj = nn.Identity()
        self.norm = nn.LayerNorm(d_model)
        self.pool = _make_pool(pooling, d_model)

    def _pool(self, h: torch.Tensor, nearest_anchor: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return h.mean(dim=1)
        if isinstance(self.pool, AnchorAwareTemporalPool):
            return self.pool(h, nearest_anchor)
        return self.pool(h)

    def forward(
        self,
        x: torch.Tensor,
        fast_indices: torch.Tensor | None = None,
        anchors: torch.Tensor | None = None,
        anchor_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h, nearest_anchor = self.condition(x, fast_indices=fast_indices, anchors=anchors, anchor_indices=anchor_indices)
        h_fw = h
        for block in self.forward_blocks:
            h_fw = block(h_fw)
        if self.bidirectional:
            h_bw = torch.flip(h, dims=[1])
            for block in self.backward_blocks:
                h_bw = block(h_bw)
            h_bw = torch.flip(h_bw, dims=[1])
            h = self.proj(torch.cat([h_fw, h_bw], dim=-1))
        else:
            h = h_fw
        h = self.norm(h)
        return self._pool(h, nearest_anchor)


class AnchorConditionedGatedDeltaFastEncoder(nn.Module):
    """Selected-anchor conditioned fast encoder with gated-delta recurrent mixer."""

    def __init__(
        self,
        d_model: int = 768,
        num_frames: int = 30,
        num_segments: int = 6,
        depth: int = 2,
        expand: int = 2,
        bidirectional: bool = True,
        pooling: str = "anchor_attention",
        anchor_scale: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.condition = AnchorConditioning(
            d_model=d_model,
            num_frames=num_frames,
            num_segments=num_segments,
            anchor_scale=anchor_scale,
            dropout=dropout,
        )
        self.forward_blocks = nn.ModuleList(
            [GatedDeltaBlock(d_model=d_model, expand=expand, dropout=dropout) for _ in range(depth)]
        )
        if bidirectional:
            self.backward_blocks = nn.ModuleList(
                [GatedDeltaBlock(d_model=d_model, expand=expand, dropout=dropout) for _ in range(depth)]
            )
            self.proj = nn.Linear(d_model * 2, d_model)
        else:
            self.backward_blocks = None
            self.proj = nn.Identity()
        self.norm = nn.LayerNorm(d_model)
        self.pool = _make_pool(pooling, d_model)

    def _pool(self, h: torch.Tensor, nearest_anchor: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return h.mean(dim=1)
        if isinstance(self.pool, AnchorAwareTemporalPool):
            return self.pool(h, nearest_anchor)
        return self.pool(h)

    def forward(
        self,
        x: torch.Tensor,
        fast_indices: torch.Tensor | None = None,
        anchors: torch.Tensor | None = None,
        anchor_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h, nearest_anchor = self.condition(x, fast_indices=fast_indices, anchors=anchors, anchor_indices=anchor_indices)
        h_fw = h
        for block in self.forward_blocks:
            h_fw = block(h_fw)
        if self.bidirectional:
            h_bw = torch.flip(h, dims=[1])
            for block in self.backward_blocks:
                h_bw = block(h_bw)
            h_bw = torch.flip(h_bw, dims=[1])
            h = self.proj(torch.cat([h_fw, h_bw], dim=-1))
        else:
            h = h_fw
        h = self.norm(h)
        return self._pool(h, nearest_anchor)


class AnchorConditionedSegmentSSMFastEncoder(nn.Module):
    """Hierarchical fast encoder with segment-local SSM and global segment SSM."""

    def __init__(
        self,
        d_model: int = 768,
        num_frames: int = 30,
        num_segments: int = 6,
        local_depth: int = 1,
        global_depth: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = True,
        pooling: str = "mean",
        use_official_mamba2: bool = False,
        strict_official_mamba2: bool = False,
        anchor_scale: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_segments = max(1, num_segments)
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.condition = AnchorConditioning(
            d_model=d_model,
            num_frames=num_frames,
            num_segments=num_segments,
            anchor_scale=anchor_scale,
            dropout=dropout,
        )
        self.local_blocks = nn.ModuleList(
            [
                SSDMixerBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    use_official_mamba2=use_official_mamba2,
                    strict_official_mamba2=strict_official_mamba2,
                    dropout=dropout,
                )
                for _ in range(local_depth)
            ]
        )
        self.segment_fuse = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.forward_blocks = nn.ModuleList(
            [
                SSDMixerBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    use_official_mamba2=use_official_mamba2,
                    strict_official_mamba2=strict_official_mamba2,
                    dropout=dropout,
                )
                for _ in range(global_depth)
            ]
        )
        if bidirectional:
            self.backward_blocks = nn.ModuleList(
                [
                    SSDMixerBlock(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        use_official_mamba2=use_official_mamba2,
                        strict_official_mamba2=strict_official_mamba2,
                        dropout=dropout,
                    )
                    for _ in range(global_depth)
                ]
            )
            self.proj = nn.Linear(d_model * 2, d_model)
        else:
            self.backward_blocks = None
            self.proj = nn.Identity()
        self.norm = nn.LayerNorm(d_model)
        self.pool = _make_pool(pooling, d_model)

    def _fallback_indices(self, x: torch.Tensor) -> torch.Tensor:
        return torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)

    def _segment_ids(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.div(
            indices.clamp(min=0, max=max(0, self.num_frames - 1)) * self.num_segments,
            max(1, self.num_frames),
            rounding_mode="floor",
        ).clamp(max=self.num_segments - 1)

    def _pack_segment(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, d = h.shape
        counts = mask.sum(dim=1)
        max_len = max(1, int(counts.max().item()))
        rows = []
        valids = []
        for batch_idx in range(b):
            tokens = h[batch_idx, mask[batch_idx]]
            valid = torch.zeros(max_len, device=h.device, dtype=torch.bool)
            if tokens.shape[0] == 0:
                tokens = h.new_zeros(max_len, d)
            else:
                valid[: tokens.shape[0]] = True
                if tokens.shape[0] < max_len:
                    tokens = F.pad(tokens, (0, 0, 0, max_len - tokens.shape[0]))
            rows.append(tokens)
            valids.append(valid)
        return torch.stack(rows, dim=0), torch.stack(valids, dim=0)

    def _masked_mean(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=h.dtype).unsqueeze(-1)
        return (h * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _segment_anchor_summary(
        self,
        anchors: torch.Tensor | None,
        anchor_indices: torch.Tensor | None,
        batch_size: int,
        d_model: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if anchors is None or anchor_indices is None or anchors.shape[1] == 0:
            return torch.zeros(batch_size, self.num_segments, d_model, device=device, dtype=dtype)
        anchor_seg = self._segment_ids(anchor_indices)
        summaries = []
        for seg_id in range(self.num_segments):
            mask = anchor_seg == seg_id
            weights = mask.to(dtype=anchors.dtype).unsqueeze(-1)
            denom = weights.sum(dim=1).clamp_min(1.0)
            summaries.append((anchors * weights).sum(dim=1) / denom)
        return torch.stack(summaries, dim=1)

    def _pool(self, h: torch.Tensor, anchor_seq: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return h.mean(dim=1)
        if isinstance(self.pool, AnchorAwareTemporalPool):
            return self.pool(h, anchor_seq)
        return self.pool(h)

    def forward(
        self,
        x: torch.Tensor,
        fast_indices: torch.Tensor | None = None,
        anchors: torch.Tensor | None = None,
        anchor_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if fast_indices is None:
            fast_indices = self._fallback_indices(x)
        h, _ = self.condition(x, fast_indices=fast_indices, anchors=anchors, anchor_indices=anchor_indices)
        seg_ids = self._segment_ids(fast_indices)
        segment_tokens = []
        for seg_id in range(self.num_segments):
            seg_h, seg_mask = self._pack_segment(h, seg_ids == seg_id)
            for block in self.local_blocks:
                seg_h = block(seg_h)
            segment_tokens.append(self._masked_mean(seg_h, seg_mask))
        h = torch.stack(segment_tokens, dim=1)
        anchor_seq = self._segment_anchor_summary(
            anchors,
            anchor_indices,
            batch_size=x.shape[0],
            d_model=x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        h = self.norm(h + self.segment_fuse(torch.cat([h, anchor_seq, h - anchor_seq], dim=-1)))
        h_fw = h
        for block in self.forward_blocks:
            h_fw = block(h_fw)
        if self.bidirectional:
            h_bw = torch.flip(h, dims=[1])
            for block in self.backward_blocks:
                h_bw = block(h_bw)
            h_bw = torch.flip(h_bw, dims=[1])
            h = self.proj(torch.cat([h_fw, h_bw], dim=-1))
        else:
            h = h_fw
        h = self.norm(h)
        return self._pool(h, anchor_seq)


class AnchorConditionedBiDecayConvFastEncoder(nn.Module):
    """Selected-anchor conditioned fast encoder with a bidirectional decayed-scan + conv mixer.

    Uses BidirectionalDecayConvBlock (a flip-concat approximation), NOT the official
    Hydra quasiseparable mixer. Named honestly to avoid the Hydra misrepresentation.
    """

    def __init__(
        self,
        d_model: int = 768,
        num_frames: int = 30,
        num_segments: int = 6,
        depth: int = 2,
        expand: int = 2,
        kernel_size: int = 3,
        pooling: str = "mean",
        anchor_scale: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pooling = pooling
        self.condition = AnchorConditioning(
            d_model=d_model,
            num_frames=num_frames,
            num_segments=num_segments,
            anchor_scale=anchor_scale,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [
                BidirectionalDecayConvBlock(
                    d_model=d_model,
                    expand=expand,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.pool = _make_pool(pooling, d_model)

    def _pool(self, h: torch.Tensor, nearest_anchor: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return h.mean(dim=1)
        if isinstance(self.pool, AnchorAwareTemporalPool):
            return self.pool(h, nearest_anchor)
        return self.pool(h)

    def forward(
        self,
        x: torch.Tensor,
        fast_indices: torch.Tensor | None = None,
        anchors: torch.Tensor | None = None,
        anchor_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h, nearest_anchor = self.condition(x, fast_indices=fast_indices, anchors=anchors, anchor_indices=anchor_indices)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self._pool(h, nearest_anchor)


class BidirectionalMambaEncoder(nn.Module):
    """Bidirectional Mamba-style encoder for fast motion modeling.

    Input:
        x: [B, T, D]
    Output:
        h_f: [B, D]
    """

    def __init__(
        self,
        d_model: int = 768,
        depth: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        pooling: str = "mean",
        use_official_mamba: bool = False,
        strict_official_mamba: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.forward_blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model,
                    d_state,
                    d_conv,
                    expand,
                    use_official_mamba,
                    strict_official_mamba,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.backward_blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model,
                    d_state,
                    d_conv,
                    expand,
                    use_official_mamba,
                    strict_official_mamba,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.proj = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        if pooling == "attention":
            self.pool = TemporalAttentionPool(d_model)
        elif pooling == "mean":
            self.pool = lambda h: h.mean(dim=1)
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")

    def forward(self, x: torch.Tensor, return_tokens: bool = False, **kwargs) -> torch.Tensor:
        x_fw = x
        for block in self.forward_blocks:
            x_fw = block(x_fw)

        x_bw = torch.flip(x, dims=[1])
        for block in self.backward_blocks:
            x_bw = block(x_bw)
        x_bw = torch.flip(x_bw, dims=[1])

        h = torch.cat([x_fw, x_bw], dim=-1)
        h = self.norm(self.proj(h))
        if return_tokens:
            return h
        return self.pool(h)


class NativeHydraQuasiSeparableBlock(nn.Module):
    """Native bidirectional quasiseparable-style single mixer for short video tokens.

    The remote environment does not provide goombalab/hydra. This block keeps
    the E21 intent that matters architecturally: a single bidirectional mixer
    with content terms and relative-time decay, instead of S5VH-style
    forward/backward independent blocks followed by flip-concat.
    """

    def __init__(
        self,
        d_model: int,
        expand: int = 2,
        num_heads: int = 8,
        local_kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        if local_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be odd.")
        inner = d_model * expand
        if inner % num_heads != 0:
            raise ValueError("d_model * expand must be divisible by num_heads.")
        self.num_heads = int(num_heads)
        self.head_dim = inner // self.num_heads
        self.norm = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, inner * 3)
        self.gate_proj = nn.Linear(d_model, inner)
        self.local_conv = nn.Conv1d(
            inner,
            inner,
            kernel_size=local_kernel_size,
            padding=local_kernel_size // 2,
            groups=inner,
        )
        self.log_decay = nn.Parameter(torch.zeros(self.num_heads))
        self.out_proj = nn.Linear(inner, d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        q, k, v = self.qkv_proj(h).chunk(3, dim=-1)
        gate = torch.sigmoid(self.gate_proj(h))
        b, t, _ = q.shape
        q = F.normalize(q.view(b, t, self.num_heads, self.head_dim).float(), dim=-1)
        k = F.normalize(k.view(b, t, self.num_heads, self.head_dim).float(), dim=-1)
        v_heads = v.view(b, t, self.num_heads, self.head_dim).float()

        logits = torch.einsum("bthc,bshc->bhts", q, k) / (self.head_dim ** 0.5)
        pos = torch.arange(t, device=x.device)
        dist = (pos[:, None] - pos[None, :]).abs().to(dtype=logits.dtype)
        decay = F.softplus(self.log_decay).view(1, self.num_heads, 1, 1)
        weights = torch.softmax(logits - decay * dist.view(1, 1, t, t), dim=-1)
        mixed = torch.einsum("bhts,bshc->bthc", weights, v_heads).reshape(b, t, -1).to(dtype=x.dtype)

        local = F.silu(self.local_conv(v.transpose(1, 2)).transpose(1, 2))
        mixed = (mixed + local) * gate
        x = residual + self.dropout(self.out_proj(mixed))
        x = x + self.dropout(self.ffn(x))
        return x


class NativeHydraFastEncoder(nn.Module):
    """Fast encoder using native Hydra-style bidirectional single-mixer blocks."""

    def __init__(
        self,
        d_model: int = 768,
        depth: int = 2,
        expand: int = 2,
        num_heads: int = 8,
        local_kernel_size: int = 3,
        pooling: str = "mean",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                NativeHydraQuasiSeparableBlock(
                    d_model=d_model,
                    expand=expand,
                    num_heads=num_heads,
                    local_kernel_size=local_kernel_size,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.pool = _make_pool(pooling, d_model)

    def forward(self, x: torch.Tensor, return_tokens: bool = False, **kwargs) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        h = self.norm(x)
        if return_tokens:
            return h
        return self.pool(h)


class DecomposedBidirectionalMambaEncoder(nn.Module):
    """DBM-style fast encoder with shared Mamba blocks and separated directional inputs.

    This differs from the S5VH encoder by sharing the sequential mixer
    parameters across directions and by fusing directional states with a gate
    rather than independent forward/backward stacks plus direct flip-concat.
    """

    def __init__(
        self,
        d_model: int = 768,
        depth: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        pooling: str = "mean",
        use_official_mamba: bool = False,
        strict_official_mamba: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.forward_in = nn.Linear(d_model, d_model)
        self.backward_in = nn.Linear(d_model, d_model)
        self.direction_embed = nn.Parameter(torch.zeros(2, 1, d_model))
        self.shared_blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model,
                    d_state,
                    d_conv,
                    expand,
                    use_official_mamba,
                    strict_official_mamba,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.merge_gate = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.Sigmoid(),
        )
        self.merge_proj = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pool = _make_pool(pooling, d_model)
        nn.init.trunc_normal_(self.direction_embed, std=0.02)

    def forward(self, x: torch.Tensor, return_tokens: bool = False, **kwargs) -> torch.Tensor:
        h = self.input_norm(x)
        h_fw = self.forward_in(h) + self.direction_embed[0].unsqueeze(0)
        h_bw = self.backward_in(torch.flip(h, dims=[1])) + self.direction_embed[1].unsqueeze(0)
        for block in self.shared_blocks:
            h_fw = block(h_fw)
            h_bw = block(h_bw)
        h_bw = torch.flip(h_bw, dims=[1])
        gate = self.merge_gate(torch.cat([h_fw, h_bw, h_fw - h_bw], dim=-1))
        merged = self.merge_proj(torch.cat([gate * h_fw, (1.0 - gate) * h_bw], dim=-1))
        h = self.norm(x + self.dropout(merged))
        if return_tokens:
            return h
        return self.pool(h)


class LocalTemporalBlock(nn.Module):
    """Residual local temporal mixer for short-range fast dynamics."""

    def __init__(self, d_model: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("local temporal kernel_size must be odd to preserve sequence length.")
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=d_model,
        )
        self.pointwise = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.depthwise(h.transpose(1, 2)).transpose(1, 2)
        return x + self.pointwise(h)


class LocalTemporalMixer(nn.Module):
    """Stack residual local temporal blocks before sequential Mamba modeling."""

    def __init__(
        self,
        d_model: int,
        local_layers: int = 2,
        kernel_size: int = 3,
        dilations: list[int] | tuple[int, ...] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if local_layers < 1:
            raise ValueError("local_layers must be >= 1.")
        if not dilations:
            dilations = [1]
        self.blocks = nn.ModuleList(
            [
                LocalTemporalBlock(
                    d_model=d_model,
                    kernel_size=kernel_size,
                    dilation=int(dilations[i % len(dilations)]),
                    dropout=dropout,
                )
                for i in range(local_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class LocalMambaFastEncoder(nn.Module):
    """Local temporal mixer followed by Bidirectional Mamba."""

    def __init__(
        self,
        d_model: int = 768,
        local_layers: int = 2,
        local_kernel_size: int = 3,
        local_dilations: list[int] | tuple[int, ...] | None = None,
        depth: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        pooling: str = "mean",
        use_official_mamba: bool = False,
        strict_official_mamba: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.local_mixer = LocalTemporalMixer(
            d_model=d_model,
            local_layers=local_layers,
            kernel_size=local_kernel_size,
            dilations=local_dilations,
            dropout=dropout,
        )
        self.mamba = BidirectionalMambaEncoder(
            d_model=d_model,
            depth=depth,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            pooling=pooling,
            use_official_mamba=use_official_mamba,
            strict_official_mamba=strict_official_mamba,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.mamba(self.local_mixer(x))


class DeltaTwoStreamFastEncoder(nn.Module):
    """Fuse frame tokens with explicit temporal deltas before fast encoding."""

    def __init__(
        self,
        d_model: int = 768,
        depth: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        pooling: str = "mean",
        base_type: str = "bidirectional_mamba",
        use_official_mamba: bool = False,
        strict_official_mamba: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.delta_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.delta_gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.Sigmoid())
        self.fuse_norm = nn.LayerNorm(d_model)
        if base_type == "bidirectional_mamba":
            self.encoder = BidirectionalMambaEncoder(
                d_model=d_model,
                depth=depth,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                pooling=pooling,
                use_official_mamba=use_official_mamba,
                strict_official_mamba=strict_official_mamba,
                dropout=dropout,
            )
        elif base_type == "local_mamba":
            self.encoder = LocalMambaFastEncoder(
                d_model=d_model,
                depth=depth,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                pooling=pooling,
                use_official_mamba=use_official_mamba,
                strict_official_mamba=strict_official_mamba,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported delta two-stream base_type: {base_type}")

    def forward(self, x: torch.Tensor, return_tokens: bool = False, **kwargs) -> torch.Tensor:
        delta = torch.zeros_like(x)
        if x.shape[1] > 1:
            delta[:, 1:] = x[:, 1:] - x[:, :-1]
        token_h = self.token_proj(x)
        delta_h = self.delta_proj(delta)
        fused = self.fuse_norm(token_h + self.delta_gate(delta) * delta_h)
        return self.encoder(fused, return_tokens=return_tokens)


class TemporalResidualFastEncoder(nn.Module):
    """Anchor-conditioned lightweight fast encoder with explicit local residuals."""

    def __init__(
        self,
        d_model: int = 768,
        num_frames: int = 30,
        num_segments: int = 6,
        depth: int = 2,
        expand: int = 1,
        bidirectional: bool = False,
        pooling: str = "mean",
        anchor_scale: float = 0.5,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.condition = AnchorConditioning(
            d_model=d_model,
            num_frames=num_frames,
            num_segments=num_segments,
            anchor_scale=anchor_scale,
            dropout=dropout,
        )
        self.delta_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.delta_gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.delta_norm = nn.LayerNorm(d_model)
        self.forward_blocks = nn.ModuleList(
            [GatedDeltaBlock(d_model=d_model, expand=expand, dropout=dropout) for _ in range(depth)]
        )
        if bidirectional:
            self.backward_blocks = nn.ModuleList(
                [GatedDeltaBlock(d_model=d_model, expand=expand, dropout=dropout) for _ in range(depth)]
            )
            self.proj = nn.Linear(d_model * 2, d_model)
        else:
            self.backward_blocks = None
            self.proj = nn.Identity()
        self.norm = nn.LayerNorm(d_model)
        self.pool = _make_pool(pooling, d_model)

    def _pool(self, h: torch.Tensor, nearest_anchor: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return h.mean(dim=1)
        if isinstance(self.pool, AnchorAwareTemporalPool):
            return self.pool(h, nearest_anchor)
        return self.pool(h)

    def forward(
        self,
        x: torch.Tensor,
        fast_indices: torch.Tensor | None = None,
        anchors: torch.Tensor | None = None,
        anchor_indices: torch.Tensor | None = None,
        return_tokens: bool = False,
    ) -> torch.Tensor:
        h, nearest_anchor = self.condition(x, fast_indices=fast_indices, anchors=anchors, anchor_indices=anchor_indices)
        local_delta = torch.zeros_like(x)
        if x.shape[1] > 1:
            local_delta[:, 1:] = x[:, 1:] - x[:, :-1]
        delta_h = self.delta_proj(local_delta)
        delta_gate = self.delta_gate(torch.cat([h, local_delta], dim=-1))
        h = self.delta_norm(h + delta_gate * delta_h)

        h_fw = h
        for block in self.forward_blocks:
            h_fw = block(h_fw)
        if self.bidirectional:
            h_bw = torch.flip(h, dims=[1])
            for block in self.backward_blocks:
                h_bw = block(h_bw)
            h_bw = torch.flip(h_bw, dims=[1])
            h = self.proj(torch.cat([h_fw, h_bw], dim=-1))
        else:
            h = h_fw
        h = self.norm(h)
        if return_tokens:
            return h
        return self._pool(h, nearest_anchor)


class GRUFastEncoder(nn.Module):
    def __init__(self, d_model: int, depth: int = 2, pooling: str = "mean"):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=depth,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.pool = TemporalAttentionPool(d_model) if pooling == "attention" else lambda h: h.mean(dim=1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h, _ = self.gru(x)
        h = self.norm(h)
        return self.pool(h)


class TemporalConvFastEncoder(nn.Module):
    def __init__(self, d_model: int, depth: int = 4, kernel_size: int = 3, pooling: str = "mean"):
        super().__init__()
        blocks = []
        for _ in range(depth):
            blocks.extend(
                [
                    nn.LayerNorm(d_model),
                    nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=1),
                    nn.GELU(),
                ]
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d_model)
        self.pool = TemporalAttentionPool(d_model) if pooling == "attention" else lambda h: h.mean(dim=1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x
        for layer in self.blocks:
            if isinstance(layer, nn.Conv1d):
                h = layer(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = layer(h)
        return self.pool(self.norm(h))


class TransformerFastEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        depth: int = 2,
        num_heads: int = 8,
        pooling: str = "mean",
        dropout: float = 0.1,
        max_len: int = 64,
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            activation="gelu",
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.pool = TemporalAttentionPool(d_model) if pooling == "attention" else lambda h: h.mean(dim=1)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.shape[1] > self.pos_embed.shape[1]:
            raise ValueError(f"TransformerFastEncoder max_len={self.pos_embed.shape[1]} < sequence length {x.shape[1]}.")
        h = x + self.pos_embed[:, : x.shape[1]]
        return self.pool(self.norm(self.encoder(h)))


class MambaTransformerFastEncoder(nn.Module):
    """Bidirectional Mamba sequence mixer followed by temporal self-attention."""

    def __init__(
        self,
        d_model: int = 768,
        mamba_depth: int = 2,
        transformer_depth: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_heads: int = 8,
        pooling: str = "mean",
        use_official_mamba: bool = False,
        strict_official_mamba: bool = False,
        dropout: float = 0.1,
        max_len: int = 64,
    ):
        super().__init__()
        self.forward_blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model,
                    d_state,
                    d_conv,
                    expand,
                    use_official_mamba,
                    strict_official_mamba,
                    dropout=dropout,
                )
                for _ in range(mamba_depth)
            ]
        )
        self.backward_blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model,
                    d_state,
                    d_conv,
                    expand,
                    use_official_mamba,
                    strict_official_mamba,
                    dropout=dropout,
                )
                for _ in range(mamba_depth)
            ]
        )
        self.bi_proj = nn.Linear(2 * d_model, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            activation="gelu",
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_depth)
        self.norm = nn.LayerNorm(d_model)
        self.pool = TemporalAttentionPool(d_model) if pooling == "attention" else lambda h: h.mean(dim=1)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x_fw = x
        for block in self.forward_blocks:
            x_fw = block(x_fw)
        x_bw = torch.flip(x, dims=[1])
        for block in self.backward_blocks:
            x_bw = block(x_bw)
        x_bw = torch.flip(x_bw, dims=[1])
        h = self.bi_proj(torch.cat([x_fw, x_bw], dim=-1))
        if h.shape[1] > self.pos_embed.shape[1]:
            raise ValueError(f"MambaTransformerFastEncoder max_len={self.pos_embed.shape[1]} < sequence length {h.shape[1]}.")
        h = h + self.pos_embed[:, : h.shape[1]]
        return self.pool(self.norm(self.transformer(h)))


def build_fast_encoder(model_cfg: dict, ablation_cfg: dict = None) -> nn.Module:
    ablation_cfg = ablation_cfg or {}
    cfg = dict(model_cfg.get("fast_encoder", {}))
    encoder_type = ablation_cfg.get("fast_encoder_type") or cfg.get("type", "bidirectional_mamba")
    d_model = int(model_cfg.get("hidden_dim", 768))
    depth = int(cfg.get("depth", 4))
    pooling = cfg.get("pooling", "mean")
    if encoder_type == "bidirectional_mamba":
        return BidirectionalMambaEncoder(
            d_model=d_model,
            depth=depth,
            d_state=int(cfg.get("d_state", 16)),
            d_conv=int(cfg.get("d_conv", 4)),
            expand=int(cfg.get("expand", 2)),
            pooling=pooling,
            use_official_mamba=bool(cfg.get("use_official_mamba", False)),
            strict_official_mamba=bool(cfg.get("strict_official_mamba", False)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    if encoder_type in {"hydra_qs", "native_hydra", "hydra"}:
        return NativeHydraFastEncoder(
            d_model=d_model,
            depth=depth,
            expand=int(cfg.get("expand", 2)),
            num_heads=int(cfg.get("num_heads", 8)),
            local_kernel_size=int(cfg.get("local_kernel_size", cfg.get("kernel_size", 3))),
            pooling=pooling,
            dropout=float(cfg.get("dropout", 0.0)),
        )
    if encoder_type in {"dbm", "decomposed_bidirectional_mamba"}:
        return DecomposedBidirectionalMambaEncoder(
            d_model=d_model,
            depth=depth,
            d_state=int(cfg.get("d_state", 16)),
            d_conv=int(cfg.get("d_conv", 4)),
            expand=int(cfg.get("expand", 2)),
            pooling=pooling,
            use_official_mamba=bool(cfg.get("use_official_mamba", False)),
            strict_official_mamba=bool(cfg.get("strict_official_mamba", False)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    if encoder_type == "local_mamba":
        return LocalMambaFastEncoder(
            d_model=d_model,
            local_layers=int(cfg.get("local_layers", 2)),
            local_kernel_size=int(cfg.get("local_kernel_size", 3)),
            local_dilations=cfg.get("local_dilations", [1, 2]),
            depth=depth,
            d_state=int(cfg.get("d_state", 16)),
            d_conv=int(cfg.get("d_conv", 4)),
            expand=int(cfg.get("expand", 2)),
            pooling=pooling,
            use_official_mamba=bool(cfg.get("use_official_mamba", False)),
            strict_official_mamba=bool(cfg.get("strict_official_mamba", False)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    if encoder_type == "gru":
        return GRUFastEncoder(d_model=d_model, depth=depth, pooling=pooling)
    if encoder_type == "temporal_conv":
        return TemporalConvFastEncoder(d_model=d_model, depth=depth, pooling=pooling)
    if encoder_type == "transformer":
        return TransformerFastEncoder(
            d_model=d_model,
            depth=depth,
            num_heads=int(cfg.get("num_heads", 8)),
            pooling=pooling,
            dropout=float(cfg.get("dropout", 0.1)),
            max_len=int(cfg.get("max_len", 64)),
        )
    if encoder_type == "delta_two_stream":
        return DeltaTwoStreamFastEncoder(
            d_model=d_model,
            depth=depth,
            d_state=int(cfg.get("d_state", 16)),
            d_conv=int(cfg.get("d_conv", 4)),
            expand=int(cfg.get("expand", 2)),
            pooling=pooling,
            base_type=cfg.get("base_type", "bidirectional_mamba"),
            use_official_mamba=bool(cfg.get("use_official_mamba", False)),
            strict_official_mamba=bool(cfg.get("strict_official_mamba", False)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    if encoder_type == "temporal_residual":
        return TemporalResidualFastEncoder(
            d_model=d_model,
            num_frames=int(model_cfg.get("num_frames", 30)),
            num_segments=int(cfg.get("num_segments", model_cfg.get("num_keyframes", 6))),
            depth=depth,
            expand=int(cfg.get("expand", 1)),
            bidirectional=bool(cfg.get("bidirectional", False)),
            pooling=pooling,
            anchor_scale=float(cfg.get("anchor_scale", 0.5)),
            dropout=float(cfg.get("dropout", 0.05)),
        )
    if encoder_type == "anchor_ssd":
        return AnchorConditionedSSDFastEncoder(
            d_model=d_model,
            num_frames=int(model_cfg.get("num_frames", 30)),
            num_segments=int(cfg.get("num_segments", model_cfg.get("num_keyframes", 6))),
            depth=depth,
            d_state=int(cfg.get("d_state", 16)),
            d_conv=int(cfg.get("d_conv", 4)),
            expand=int(cfg.get("expand", 2)),
            bidirectional=bool(cfg.get("bidirectional", True)),
            pooling=pooling,
            use_official_mamba2=bool(cfg.get("use_official_mamba2", False)),
            strict_official_mamba2=bool(cfg.get("strict_official_mamba2", False)),
            anchor_scale=float(cfg.get("anchor_scale", 1.0)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    if encoder_type == "anchor_gated_delta":
        return AnchorConditionedGatedDeltaFastEncoder(
            d_model=d_model,
            num_frames=int(model_cfg.get("num_frames", 30)),
            num_segments=int(cfg.get("num_segments", model_cfg.get("num_keyframes", 6))),
            depth=depth,
            expand=int(cfg.get("expand", 2)),
            bidirectional=bool(cfg.get("bidirectional", True)),
            pooling=pooling,
            anchor_scale=float(cfg.get("anchor_scale", 1.0)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    if encoder_type == "anchor_segment_ssm":
        return AnchorConditionedSegmentSSMFastEncoder(
            d_model=d_model,
            num_frames=int(model_cfg.get("num_frames", 30)),
            num_segments=int(cfg.get("num_segments", model_cfg.get("num_keyframes", 6))),
            local_depth=int(cfg.get("local_depth", 1)),
            global_depth=int(cfg.get("global_depth", depth)),
            d_state=int(cfg.get("d_state", 16)),
            d_conv=int(cfg.get("d_conv", 4)),
            expand=int(cfg.get("expand", 2)),
            bidirectional=bool(cfg.get("bidirectional", True)),
            pooling=pooling,
            use_official_mamba2=bool(cfg.get("use_official_mamba2", False)),
            strict_official_mamba2=bool(cfg.get("strict_official_mamba2", False)),
            anchor_scale=float(cfg.get("anchor_scale", 1.0)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    if encoder_type in ("anchor_bi_decay_conv", "anchor_hydra"):
        # "anchor_hydra" is a legacy alias (NOT official Hydra); kept so older configs
        # and checkpoints that still say "anchor_hydra" continue to load unchanged.
        return AnchorConditionedBiDecayConvFastEncoder(
            d_model=d_model,
            num_frames=int(model_cfg.get("num_frames", 30)),
            num_segments=int(cfg.get("num_segments", model_cfg.get("num_keyframes", 6))),
            depth=depth,
            expand=int(cfg.get("expand", 2)),
            kernel_size=int(cfg.get("kernel_size", 3)),
            pooling=pooling,
            anchor_scale=float(cfg.get("anchor_scale", 1.0)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    if encoder_type == "mamba_transformer":
        return MambaTransformerFastEncoder(
            d_model=d_model,
            mamba_depth=int(cfg.get("mamba_depth", depth)),
            transformer_depth=int(cfg.get("transformer_depth", 1)),
            d_state=int(cfg.get("d_state", 16)),
            d_conv=int(cfg.get("d_conv", 4)),
            expand=int(cfg.get("expand", 2)),
            num_heads=int(cfg.get("num_heads", 8)),
            pooling=pooling,
            use_official_mamba=bool(cfg.get("use_official_mamba", False)),
            strict_official_mamba=bool(cfg.get("strict_official_mamba", False)),
            dropout=float(cfg.get("dropout", 0.1)),
            max_len=int(cfg.get("max_len", 64)),
        )
    raise ValueError(f"Unsupported fast encoder type: {encoder_type}")
