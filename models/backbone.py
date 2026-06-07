from typing import Optional

import torch
from torch import nn


class VideoFrameBackbone(nn.Module):
    """Frame encoder for input video frames.

    Input:
        frames: [B, T, 3, H, W]
    Output:
        features: [B, T, output_dim]
    """

    def __init__(
        self,
        backbone_type: str = "resnet50",
        output_dim: int = 768,
        pretrained: bool = False,
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.output_dim = output_dim
        self.pretrained = pretrained

        if backbone_type != "resnet50":
            raise ValueError("Only resnet50 is implemented for frame input in this skeleton.")
        try:
            from torchvision.models import ResNet50_Weights, resnet50
        except ImportError as exc:  # pragma: no cover
            raise ImportError("torchvision is required for input_type='frames'.") from exc

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        in_dim = model.fc.in_features
        model.fc = nn.Identity()
        self.encoder = model
        self.proj = nn.Identity() if in_dim == output_dim else nn.Linear(in_dim, output_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError(f"Expected frames [B, T, 3, H, W], got {tuple(frames.shape)}")
        b, t, c, h, w = frames.shape
        x = frames.reshape(b * t, c, h, w)
        feat = self.encoder(x)
        feat = self.proj(feat)
        return feat.reshape(b, t, -1)


class FeatureProjector(nn.Module):
    """Projection used when input_type='features'."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected features [B, T, D], got {tuple(x.shape)}")
        return self.proj(x.float())


def build_frame_backbone(model_cfg: dict) -> Optional[VideoFrameBackbone]:
    if model_cfg.get("input_type", "features") != "frames":
        return None
    cfg = model_cfg.get("frame_backbone", {})
    return VideoFrameBackbone(
        backbone_type=cfg.get("type", "resnet50"),
        output_dim=int(model_cfg.get("hidden_dim", 768)),
        pretrained=bool(cfg.get("pretrained", False)),
    )
