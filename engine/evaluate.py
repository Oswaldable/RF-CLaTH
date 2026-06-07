from typing import Dict, Iterable, Optional

import torch

from .extract_hash import extract_hash_codes
from utils.metrics import compute_retrieval_metrics


@torch.no_grad()
def evaluate_codes(
    query: Dict,
    retrieval: Dict,
    topk: Iterable[int],
    binary_format: str = "pm1",
    map_topk: Optional[Iterable[int]] = None,
    gmap_topk: Optional[Iterable[int]] = None,
) -> Dict[str, float]:
    return compute_retrieval_metrics(
        query["binary_code"],
        retrieval["binary_code"],
        query["label"],
        retrieval["label"],
        topk=topk,
        binary_format=binary_format,
        map_topk=map_topk,
        gmap_topk=gmap_topk,
    )


@torch.no_grad()
def evaluate_retrieval(
    model,
    query_loader,
    retrieval_loader,
    device: torch.device,
    topk: Iterable[int],
    binary_format: str = "pm1",
    map_topk: Optional[Iterable[int]] = None,
    gmap_topk: Optional[Iterable[int]] = None,
) -> Dict[str, float]:
    query = extract_hash_codes(model, query_loader, device=device, binary_format=binary_format)
    retrieval = extract_hash_codes(model, retrieval_loader, device=device, binary_format=binary_format)
    return evaluate_codes(
        query,
        retrieval,
        topk=topk,
        binary_format=binary_format,
        map_topk=map_topk,
        gmap_topk=gmap_topk,
    )
