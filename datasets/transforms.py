from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch


def uniform_sample_indices(length: int, num_frames: int) -> List[int]:
    """Return uniformly spaced indices for a sequence."""
    if length <= 0:
        raise ValueError("Cannot sample frames from an empty sequence.")
    if length == num_frames:
        return list(range(length))
    return np.linspace(0, length - 1, num_frames).round().astype(np.int64).tolist()


def sample_or_pad_sequence(x: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Convert [T, ...] to exactly num_frames by uniform sampling or last padding."""
    if x.ndim < 2:
        raise ValueError(f"Expected sequence tensor [T, ...], got shape {tuple(x.shape)}")
    t = x.shape[0]
    if t == num_frames:
        return x
    if t > num_frames:
        return x[uniform_sample_indices(t, num_frames)]
    pad = x[-1:].expand(num_frames - t, *x.shape[1:])
    return torch.cat([x, pad], dim=0)


def _pil_to_tensor(image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def build_frame_transform(image_size: int = 224):
    """Build a minimal RGB frame transform without requiring torchvision."""

    def transform(image):
        return _pil_to_tensor(image, image_size=image_size)

    return transform


def list_frame_files(video_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in video_dir.iterdir() if p.suffix.lower() in exts])


def load_frames_from_directory(
    video_dir: Path,
    num_frames: int,
    transform=None,
    image_size: int = 224,
) -> torch.Tensor:
    """Load a frame folder as [T, 3, H, W]."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for input_type='frames'.") from exc

    files = list_frame_files(Path(video_dir))
    if not files:
        raise FileNotFoundError(f"No image frames found under {video_dir}")
    transform = transform or build_frame_transform(image_size=image_size)
    sampled = [files[i] for i in uniform_sample_indices(len(files), num_frames)]
    frames = [transform(Image.open(path)) for path in sampled]
    return torch.stack(frames, dim=0)


def labels_to_multihot(labels: Iterable[int], num_classes: int) -> torch.Tensor:
    target = torch.zeros(num_classes, dtype=torch.float32)
    for label in labels:
        if 0 <= label < num_classes:
            target[label] = 1.0
    return target
