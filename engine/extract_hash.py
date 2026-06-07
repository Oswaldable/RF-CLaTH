from typing import Dict

import torch


@torch.no_grad()
def extract_hash_codes(
    model,
    dataloader,
    device: torch.device,
    binary_format: str = "pm1",
) -> Dict:
    """Extract soft and binary hash codes from a dataloader."""
    model.eval()
    video_ids = []
    soft_codes = []
    binary_codes = []
    labels = []
    selected_indices = []

    for batch in dataloader:
        video = batch["video"].to(device, non_blocking=True)
        out = model.encode(video, binary_format=binary_format)
        soft_codes.append(out["soft_code"].detach().cpu())
        binary_codes.append(out["binary_code"].detach().cpu())
        labels.append(batch["label"].detach().cpu())
        selected_indices.append(out["selected_indices"].detach().cpu())
        video_ids.extend(list(batch["video_id"]))

    return {
        "video_id": video_ids,
        "soft_code": torch.cat(soft_codes, dim=0),
        "binary_code": torch.cat(binary_codes, dim=0),
        "label": torch.cat(labels, dim=0),
        "selected_indices": torch.cat(selected_indices, dim=0),
    }
