from typing import Dict, Tuple

import torch
from torch import nn

from .backbone import FeatureProjector, build_frame_backbone
from .bidirectional_mamba import MaskedTemporalAugmentation, build_fast_encoder
from .fusion import ContentTimeLateralFusion, FastToSlowLateralFusion, SemanticMotionFusion, SlowResidualOutputFusion
from .hash_head import HashHead
from .keyframe_selector import KeyFrameSelector
from .slow_transformer import build_slow_encoder


def gather_remaining_frames_with_indices(x: torch.Tensor, selected_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remove selected key frames and also return original temporal indices."""
    b, t, _ = x.shape
    mask = torch.ones(b, t, device=x.device, dtype=torch.bool)
    if selected_indices.numel() > 0:
        mask.scatter_(1, selected_indices, False)
    all_indices = torch.arange(t, device=x.device).unsqueeze(0).expand(b, -1)
    remaining = [x[i, mask[i]] for i in range(b)]
    remaining_indices = [all_indices[i, mask[i]] for i in range(b)]
    return torch.stack(remaining, dim=0), torch.stack(remaining_indices, dim=0)


class RetrievalFeedbackContentLateralTemporalHashing(nn.Module):
    """RF-CLaTH: Retrieval-Feedback Content-Lateral Temporal Hashing.

    Forward input:
        features: [B, 25, D] when input_type='features'
        frames: [B, 25, 3, H, W] when input_type='frames'
    Forward output:
        dict containing h_s, h_f_a, h_f_b, z_a, z_b, u_a, u_b,
        selected_indices, fast_mask_a, fast_mask_b.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg["model"]
        ablation_cfg = cfg.get("ablation", {})
        self.input_type = model_cfg.get("input_type", "features")
        self.num_frames = int(model_cfg.get("num_frames", 25))
        self.num_keyframes = int(model_cfg.get("num_keyframes", 5))
        self.feature_dim = int(model_cfg.get("feature_dim", 768))
        self.hidden_dim = int(model_cfg.get("hidden_dim", 768))

        self.use_slow = bool(ablation_cfg.get("use_slow_branch", True))
        self.use_fast = bool(ablation_cfg.get("use_fast_branch", True))
        self.use_keyframe_selector = bool(ablation_cfg.get("use_keyframe_selector", True))
        self.use_mask_aug = bool(ablation_cfg.get("use_mask_aug", True))

        self.frame_backbone = build_frame_backbone(model_cfg)
        self.feature_projector = FeatureProjector(self.feature_dim, self.hidden_dim)

        selector_cfg = model_cfg.get("keyframe_selector", {})
        selector_strategy = selector_cfg.get("strategy", "segment_rerank_gumbel_topk")
        if not self.use_keyframe_selector:
            raise ValueError(
                "ablation.use_keyframe_selector=false is no longer supported in the current mainline. "
                "Use keyframe_selector.strategy='segment_rerank_gumbel_topk' for the E13-K2 selector."
            )
        self.keyframe_selector = KeyFrameSelector(
            d_model=self.hidden_dim,
            num_keyframes=self.num_keyframes,
            num_frames=self.num_frames,
            strategy=selector_strategy,
            temperature=float(selector_cfg.get("temperature", 1.0)),
            use_straight_through=bool(selector_cfg.get("use_straight_through", True)),
            candidate_topm=int(selector_cfg.get("candidate_topm", 2)),
            alpha_motion=float(selector_cfg.get("alpha_motion", 0.10)),
            beta_redundancy=float(selector_cfg.get("beta_redundancy", 0.05)),
            gamma_coverage=float(selector_cfg.get("gamma_coverage", 0.05)),
        )

        self.slow_encoder = build_slow_encoder(model_cfg, ablation_cfg)

        mask_cfg = model_cfg.get("mask_aug", {})
        self.mask_aug = MaskedTemporalAugmentation(
            d_model=self.hidden_dim,
            mask_ratio=float(mask_cfg.get("mask_ratio", 0.3)),
            use_temporal_jitter=bool(mask_cfg.get("use_temporal_jitter", True)),
            use_frame_dropout=bool(mask_cfg.get("use_frame_dropout", True)),
            frame_dropout_ratio=float(mask_cfg.get("frame_dropout_ratio", 0.1)),
            use_motion_aware_mask=bool(mask_cfg.get("use_motion_aware_mask", False)),
            mask_motion=mask_cfg.get("mask_motion", "low"),
        )
        self.fast_encoder = build_fast_encoder(model_cfg, ablation_cfg)
        fast_cfg = model_cfg.get("fast_encoder", {})
        self.fast_input_frames = str(fast_cfg.get("input_frames", "remaining")).lower()
        if self.fast_input_frames not in {"remaining", "all"}:
            raise ValueError("model.fast_encoder.input_frames must be 'remaining' or 'all'.")

        fusion_cfg = model_cfg.get("fusion", {})
        self.fusion_type = fusion_cfg.get("type", "gated")
        self.use_lateral_fusion = self.fusion_type in {
            "fast_to_slow_lateral",
            "fast_to_slow_lateral_fusion",
            "content_time_lateral",
            "content_time_lateral_fusion",
            "lateral",
        }
        self.lateral_final_residual = bool(fusion_cfg.get("final_residual", False))
        if self.use_lateral_fusion:
            lateral_cls = (
                ContentTimeLateralFusion
                if self.fusion_type in {"content_time_lateral", "content_time_lateral_fusion"}
                else FastToSlowLateralFusion
            )
            lateral_kwargs = {}
            if lateral_cls is ContentTimeLateralFusion:
                lateral_kwargs.update(
                    {
                        "content_temperature": float(fusion_cfg.get("content_temperature", 0.5)),
                        "num_time_buckets": int(fusion_cfg.get("num_time_buckets", self.num_frames)),
                        "exclude_self_lateral": bool(fusion_cfg.get("exclude_self_lateral", False)),
                    }
                )
            self.lateral_fusion = lateral_cls(
                d_model=self.hidden_dim,
                temperature=float(fusion_cfg.get("lateral_temperature", 4.0)),
                gamma_init=float(fusion_cfg.get("lateral_gamma_init", 0.1)),
                dropout=float(fusion_cfg.get("dropout", 0.1)),
                **lateral_kwargs,
            )
            self.fusion = (
                SlowResidualOutputFusion(
                    d_model=self.hidden_dim,
                    gamma_init=float(fusion_cfg.get("final_residual_gamma_init", 0.05)),
                    dropout=float(fusion_cfg.get("dropout", 0.1)),
                )
                if self.lateral_final_residual
                else None
            )
        else:
            self.lateral_fusion = None
            self.fusion = SemanticMotionFusion(
                d_model=self.hidden_dim,
                fusion_type=self.fusion_type,
                dropout=float(fusion_cfg.get("dropout", 0.1)),
                gamma_init=float(fusion_cfg.get("gamma_init", 0.1)),
            )
        self.hash_head = HashHead(self.hidden_dim, int(model_cfg.get("hash_bits", 64)))

    def _encode_input(self, video_or_features: torch.Tensor) -> torch.Tensor:
        if self.input_type == "frames":
            x = self.frame_backbone(video_or_features)
        elif self.input_type == "features":
            x = self.feature_projector(video_or_features)
        else:
            raise ValueError(f"Unsupported input_type: {self.input_type}")
        if x.shape[1] != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames/features, got {x.shape[1]}")
        return x

    def _select_keyframes(self, x: torch.Tensor):
        if self.use_slow or self.use_keyframe_selector:
            return self.keyframe_selector(x)
        b, t, _ = x.shape
        empty_idx = torch.empty(b, 0, device=x.device, dtype=torch.long)
        empty_mask = torch.zeros(b, t, device=x.device, dtype=x.dtype)
        return x[:, :0], empty_idx, empty_mask

    def _fuse_or_bypass(self, h_s: torch.Tensor, h_f: torch.Tensor) -> torch.Tensor:
        if self.use_lateral_fusion and self.use_slow:
            if self.lateral_final_residual and self.use_fast:
                return self.fusion(h_s, h_f)
            return h_s
        if self.use_slow and self.use_fast:
            return self.fusion(h_s, h_f)
        if self.use_slow:
            return h_s
        return h_f

    def _encode_fast(
        self,
        x_fast: torch.Tensor,
        fast_indices: torch.Tensor,
        anchors: torch.Tensor,
        anchor_indices: torch.Tensor,
        return_tokens: bool = False,
    ) -> torch.Tensor:
        if return_tokens:
            return self.fast_encoder(
                x_fast,
                fast_indices=fast_indices,
                anchors=anchors,
                anchor_indices=anchor_indices,
                return_tokens=True,
            )
        return self.fast_encoder(
            x_fast,
            fast_indices=fast_indices,
            anchors=anchors,
            anchor_indices=anchor_indices,
        )

    def forward(
        self,
        video_or_features: torch.Tensor,
        deterministic: bool = False,
        return_one_view: bool = False,
    ) -> Dict[str, torch.Tensor]:
        x = self._encode_input(video_or_features)
        x_s, selected_indices, slow_mask = self._select_keyframes(x)

        if self.use_slow:
            h_s = None if self.use_lateral_fusion else self.slow_encoder(x_s, full=x, selected_indices=selected_indices)
            if self.fast_input_frames == "all":
                x_fast_source = x
                fast_indices = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
            else:
                x_fast_source, fast_indices = gather_remaining_frames_with_indices(x, selected_indices)
        else:
            h_s = torch.zeros(x.shape[0], self.hidden_dim, device=x.device, dtype=x.dtype)
            x_fast_source = x
            fast_indices = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
            x_s = x[:, :0]
            selected_indices = torch.empty(x.shape[0], 0, device=x.device, dtype=torch.long)

        if self.use_fast:
            if return_one_view:
                x_f_a, mask_a = self.mask_aug(x_fast_source, deterministic=True, enabled=False)
                x_f_b, mask_b = x_f_a, mask_a
            else:
                enable_aug = self.training and (not deterministic) and self.use_mask_aug
                x_f_a, mask_a = self.mask_aug(x_fast_source, deterministic=deterministic, enabled=enable_aug)
                x_f_b, mask_b = self.mask_aug(x_fast_source, deterministic=deterministic, enabled=enable_aug)
            if self.use_lateral_fusion and self.use_slow:
                fast_tokens_a = self._encode_fast(x_f_a, fast_indices, x_s, selected_indices, return_tokens=True)
                fast_tokens_b = self._encode_fast(x_f_b, fast_indices, x_s, selected_indices, return_tokens=True)
                h_f_a = fast_tokens_a.mean(dim=1)
                h_f_b = fast_tokens_b.mean(dim=1)
                x_s_a = self.lateral_fusion(x_s, fast_tokens_a, fast_indices, selected_indices)
                x_s_b = self.lateral_fusion(x_s, fast_tokens_b, fast_indices, selected_indices)
                h_s_a = self.slow_encoder(x_s_a, full=x, selected_indices=selected_indices)
                h_s_b = self.slow_encoder(x_s_b, full=x, selected_indices=selected_indices)
                h_s = 0.5 * (h_s_a + h_s_b)
            else:
                h_f_a = self._encode_fast(x_f_a, fast_indices, x_s, selected_indices)
                h_f_b = self._encode_fast(x_f_b, fast_indices, x_s, selected_indices)
        else:
            mask_shape = (x.shape[0], x_fast_source.shape[1])
            mask_a = torch.zeros(mask_shape, device=x.device, dtype=torch.bool)
            mask_b = torch.zeros(mask_shape, device=x.device, dtype=torch.bool)
            h_f_a = torch.zeros_like(h_s)
            h_f_b = torch.zeros_like(h_s)
            if self.use_lateral_fusion and self.use_slow:
                h_s_a = self.slow_encoder(x_s, full=x, selected_indices=selected_indices)
                h_s_b = h_s_a
                h_s = h_s_a

        if self.use_lateral_fusion and self.use_slow:
            z_a = self._fuse_or_bypass(h_s_a, h_f_a)
            z_b = self._fuse_or_bypass(h_s_b, h_f_b)
        else:
            z_a = self._fuse_or_bypass(h_s, h_f_a)
            z_b = self._fuse_or_bypass(h_s, h_f_b)
        u_a = self.hash_head(z_a)
        u_b = self.hash_head(z_b)

        outputs = {
            "h_s": h_s,
            "h_f_a": h_f_a,
            "h_f_b": h_f_b,
            "z_a": z_a,
            "z_b": z_b,
            "u_a": u_a,
            "u_b": u_b,
            "selected_indices": selected_indices,
            "fast_indices": fast_indices,
            "slow_mask": slow_mask,
            "fast_mask_a": mask_a,
            "fast_mask_b": mask_b,
            "use_slow": torch.tensor(self.use_slow, device=x.device),
            "use_fast": torch.tensor(self.use_fast, device=x.device),
        }
        return outputs

    @torch.no_grad()
    def encode(self, video_or_features: torch.Tensor, binary_format: str = "pm1") -> Dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()
        out = self.forward(video_or_features, deterministic=True, return_one_view=True)
        u = out["u_a"]
        b = self.hash_head.binarize(u, binary_format=binary_format)
        out["soft_code"] = u
        out["binary_code"] = b
        if was_training:
            self.train()
        return out

