import math
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler


def _close_dataset_reader(dataset):
    reader = getattr(dataset, "_h5_reader", None)
    if reader is not None:
        reader.close()
        dataset._h5_reader = None


@torch.no_grad()
def build_raw_feature_bank(
    dataset,
    batch_size: int = 256,
    logger=None,
) -> torch.Tensor:
    """Build a static normalized feature bank from raw pre-extracted frames.

    Each video is represented by mean pooling over its sampled frame features:
        video: [T, D] -> bank row: [D]
    """

    rows = []
    total = len(dataset)
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch_rows = []
        for index in range(start, end):
            item = dataset[index]
            video = item["video"].float()
            if video.ndim == 3:
                video = video.flatten(start_dim=1)
            batch_rows.append(video.mean(dim=0))
        rows.append(torch.stack(batch_rows, dim=0))
        if logger is not None and (end == total or end % (batch_size * 20) == 0):
            logger.info("neighbor_bank progress=%d/%d", end, total)

    _close_dataset_reader(dataset)
    bank = torch.cat(rows, dim=0)
    return F.normalize(bank, dim=-1)


@torch.no_grad()
def compute_topk_neighbors(
    bank: torch.Tensor,
    topk: int,
    chunk_size: int = 512,
    device: Optional[torch.device] = None,
    logger=None,
) -> torch.Tensor:
    """Compute cosine nearest neighbors for a normalized feature bank."""

    total = bank.shape[0]
    topk = min(int(topk), max(1, total - 1))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bank_device = bank.to(device)
    neighbors = []

    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        sim = bank_device[start:end] @ bank_device.t()
        diag = torch.arange(start, end, device=device)
        sim[torch.arange(end - start, device=device), diag] = -float("inf")
        nn_idx = torch.topk(sim, k=topk, dim=1).indices.cpu()
        neighbors.append(nn_idx)
        if logger is not None and (end == total or end % (chunk_size * 20) == 0):
            logger.info("neighbor_topk progress=%d/%d", end, total)

    return torch.cat(neighbors, dim=0).long()


def load_or_build_neighbors(
    dataset,
    topk: int,
    feature_batch_size: int = 256,
    sim_chunk_size: int = 512,
    device: Optional[torch.device] = None,
    cache_path: str = "",
    logger=None,
) -> torch.Tensor:
    """Load cached nearest neighbors or build them from the train dataset."""

    cache = Path(cache_path).expanduser() if cache_path else None
    if cache is not None and cache.exists():
        if logger is not None:
            logger.info("loading neighbor cache=%s", cache)
        table = torch.load(str(cache), map_location="cpu")
        return table.long()

    if logger is not None:
        logger.info("building neighbor table samples=%d topk=%d", len(dataset), topk)
    bank = build_raw_feature_bank(dataset, batch_size=feature_batch_size, logger=logger)
    table = compute_topk_neighbors(
        bank,
        topk=topk,
        chunk_size=sim_chunk_size,
        device=device,
        logger=logger,
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(table, str(cache))
        if logger is not None:
            logger.info("saved neighbor cache=%s", cache)
    return table


def estimate_neighbor_label_precision(dataset, neighbor_indices: torch.Tensor, max_items: int = 2000) -> float:
    """Diagnostic only: estimate how often raw nearest neighbors share labels."""

    if not hasattr(dataset, "records") or len(dataset.records) == 0:
        return 0.0
    total = len(dataset)
    count = min(int(max_items), total)
    if count <= 0:
        return 0.0
    if count < total:
        probe = torch.linspace(0, total - 1, steps=count).long()
    else:
        probe = torch.arange(total)

    hits = []
    for index in probe.tolist():
        labels = set(dataset.records[index].get("labels", []))
        if not labels:
            continue
        local_hits = []
        for neighbor in neighbor_indices[index].tolist():
            other = set(dataset.records[neighbor].get("labels", []))
            if other:
                local_hits.append(1.0 if labels.intersection(other) else 0.0)
        if local_hits:
            hits.append(sum(local_hits) / len(local_hits))
    if not hits:
        return 0.0
    return float(sum(hits) / len(hits))


class NeighborBatchSampler(Sampler):
    """Mini-batch sampler that places raw-feature neighbors in the same batch."""

    def __init__(
        self,
        dataset_size: int,
        neighbor_indices: torch.Tensor,
        batch_size: int,
        neighbors_per_anchor: int = 1,
        drop_last: bool = True,
        seed: int = 0,
    ):
        if batch_size < 2:
            raise ValueError("NeighborBatchSampler requires batch_size >= 2.")
        self.dataset_size = int(dataset_size)
        self.neighbor_indices = neighbor_indices.cpu().long()
        self.batch_size = int(batch_size)
        self.neighbors_per_anchor = max(1, int(neighbors_per_anchor))
        self.drop_last = drop_last
        self.seed = int(seed)
        self.epoch = 0
        self.group_size = 1 + self.neighbors_per_anchor
        self.anchors_per_batch = max(1, math.ceil(self.batch_size / float(self.group_size)))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        anchors = torch.randperm(self.dataset_size, generator=generator).tolist()
        rng = random.Random(self.seed + self.epoch)

        for start in range(0, len(anchors), self.anchors_per_batch):
            anchor_group = anchors[start : start + self.anchors_per_batch]
            batch = []
            for anchor in anchor_group:
                batch.append(int(anchor))
                candidates = self.neighbor_indices[anchor].tolist()
                if not candidates:
                    continue
                for _ in range(self.neighbors_per_anchor):
                    batch.append(int(rng.choice(candidates)))
            if len(batch) < self.batch_size and self.drop_last:
                continue
            if len(batch) < self.batch_size:
                while len(batch) < self.batch_size:
                    batch.append(rng.randrange(self.dataset_size))
            yield batch[: self.batch_size]

    def __len__(self) -> int:
        batches = self.dataset_size / float(self.anchors_per_batch)
        if self.drop_last:
            return int(batches)
        return int(math.ceil(batches))
