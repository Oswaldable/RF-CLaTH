from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F


def hamming_distance(query_codes: torch.Tensor, retrieval_codes: torch.Tensor, binary_format: str = "pm1") -> torch.Tensor:
    """Pairwise Hamming distance.

    pm1 codes use {-1, +1}: dist = 0.5 * (K - dot).
    01 codes use XOR.
    """
    if binary_format == "pm1":
        q = query_codes.float()
        r = retrieval_codes.float()
        bits = q.shape[1]
        return 0.5 * (bits - q @ r.t())
    if binary_format == "01":
        q = query_codes.bool()
        r = retrieval_codes.bool()
        return (q[:, None, :] ^ r[None, :, :]).sum(dim=-1).float()
    raise ValueError(f"Unsupported binary_format: {binary_format}")


def relevance_matrix(query_labels: torch.Tensor, retrieval_labels: torch.Tensor) -> torch.Tensor:
    if query_labels.ndim == 1:
        return query_labels[:, None].eq(retrieval_labels[None, :])
    return (query_labels.float() @ retrieval_labels.float().t()) > 0


def _average_precision_from_sorted_relevance(
    rel_sorted: torch.Tensor,
    topk: Optional[int] = None,
    average_empty_as_zero: bool = True,
) -> float:
    """Compute CLaTH baseline legacy mAP from sorted relevance.

    The baseline code computes AP@K as:
        sum(precision_at_relevant_rank for ranks <= K) / K
    and full AP as the same expression divided by the retrieval database size.
    This is intentionally different from the common AP denominator of
    ``number_of_relevant_items``.
    """
    denominator = int(topk) if topk is not None else int(rel_sorted.shape[1])
    window = min(denominator, int(rel_sorted.shape[1]))
    rel_sorted = rel_sorted[:, :window]
    rel_sorted = rel_sorted.float()
    ranks = torch.arange(1, rel_sorted.shape[1] + 1, device=rel_sorted.device).float()
    precision_at_rank = rel_sorted.cumsum(dim=1) / ranks
    ap = (precision_at_rank * rel_sorted).sum(dim=1) / float(max(1, denominator))
    if average_empty_as_zero:
        return ap.mean().item()
    valid = rel_sorted.sum(dim=1) > 0
    if valid.any():
        return ap[valid].mean().item()
    return 0.0


def mean_average_precision(
    query_codes: torch.Tensor,
    retrieval_codes: torch.Tensor,
    query_labels: torch.Tensor,
    retrieval_labels: torch.Tensor,
    binary_format: str = "pm1",
    topk: Optional[int] = None,
    average_empty_as_zero: bool = True,
) -> float:
    dist = hamming_distance(query_codes, retrieval_codes, binary_format=binary_format)
    order = dist.argsort(dim=1)
    if topk is not None:
        order = order[:, :topk]
    rel = relevance_matrix(query_labels, retrieval_labels).to(query_codes.device)
    rel_sorted = torch.gather(rel, 1, order).float()
    return _average_precision_from_sorted_relevance(
        rel_sorted,
        topk=None,
        average_empty_as_zero=average_empty_as_zero,
    )


def precision_at_k(
    query_codes: torch.Tensor,
    retrieval_codes: torch.Tensor,
    query_labels: torch.Tensor,
    retrieval_labels: torch.Tensor,
    k: int,
    binary_format: str = "pm1",
) -> float:
    k = min(k, retrieval_codes.shape[0])
    dist = hamming_distance(query_codes, retrieval_codes, binary_format=binary_format)
    order = dist.argsort(dim=1)[:, :k]
    rel = relevance_matrix(query_labels, retrieval_labels).to(query_codes.device)
    rel_sorted = torch.gather(rel, 1, order).float()
    return rel_sorted.mean().item()


def recall_at_k(
    query_codes: torch.Tensor,
    retrieval_codes: torch.Tensor,
    query_labels: torch.Tensor,
    retrieval_labels: torch.Tensor,
    k: int,
    binary_format: str = "pm1",
) -> float:
    k = min(k, retrieval_codes.shape[0])
    dist = hamming_distance(query_codes, retrieval_codes, binary_format=binary_format)
    order = dist.argsort(dim=1)[:, :k]
    rel = relevance_matrix(query_labels, retrieval_labels).to(query_codes.device)
    rel_sorted = torch.gather(rel, 1, order).float()
    total_rel = rel.sum(dim=1).float().clamp_min(1.0)
    return (rel_sorted.sum(dim=1) / total_rel).mean().item()


def compute_retrieval_metrics(
    query_codes: torch.Tensor,
    retrieval_codes: torch.Tensor,
    query_labels: torch.Tensor,
    retrieval_labels: torch.Tensor,
    topk: Iterable[int],
    binary_format: str = "pm1",
    map_topk: Optional[Iterable[int]] = None,
    gmap_topk: Optional[Iterable[int]] = None,
) -> Dict[str, float]:
    """Compute Hamming retrieval metrics using the CLaTH baseline protocol."""
    topk_values = [int(k) for k in topk]
    map_topk_values = [int(k) for k in (map_topk if map_topk is not None else topk_values)]
    gmap_topk_values = [int(k) for k in (gmap_topk if gmap_topk is not None else [])]

    dist = hamming_distance(query_codes, retrieval_codes, binary_format=binary_format)
    order = dist.argsort(dim=1)
    rel = relevance_matrix(query_labels, retrieval_labels).to(query_codes.device)
    rel_sorted = torch.gather(rel, 1, order).float()
    total_rel = rel.sum(dim=1).float().clamp_min(1.0)

    metrics = {}
    metrics["mAP"] = _average_precision_from_sorted_relevance(rel_sorted, topk=None, average_empty_as_zero=True)
    metrics["mAP@All"] = metrics["mAP"]

    for k in map_topk_values:
        metrics[f"mAP@{k}"] = _average_precision_from_sorted_relevance(
            rel_sorted,
            topk=k,
            average_empty_as_zero=True,
        )

    if gmap_topk_values:
        values = [metrics[f"mAP@{k}"] for k in gmap_topk_values if f"mAP@{k}" in metrics]
        if values:
            metrics["GmAP"] = float(sum(value * value for value in values) ** 0.5)

    for k in topk_values:
        k_eff = min(k, retrieval_codes.shape[0])
        rel_at_k = rel_sorted[:, :k_eff]
        metrics[f"Precision@{k}"] = rel_at_k.mean().item()
        metrics[f"Recall@{k}"] = (rel_at_k.sum(dim=1) / total_rel).mean().item()
    return metrics


@torch.no_grad()
def compute_self_supervised_hash_metrics(outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Batch-level diagnostics for self-supervised video hashing.

    These metrics do not use labels. They are intended for training logs:
    agreement/hamming track two-view stability, entropy/usage track bit
    collapse, and saturation tracks how close soft codes are to binary values.
    """
    u_a = outputs["u_a"].detach().float()
    u_b = outputs["u_b"].detach().float()
    bits = max(1, u_a.shape[1])

    b_a = torch.sign(u_a)
    b_b = torch.sign(u_b)
    b_a[b_a == 0] = 1
    b_b[b_b == 0] = 1

    bit_agreement = (b_a == b_b).float().mean()
    hamming = 0.5 * (bits - (b_a * b_b).sum(dim=1))
    hamming_norm = hamming.mean() / bits
    hash_cos = F.cosine_similarity(u_a, u_b, dim=-1).mean()

    u_all = torch.cat([u_a, u_b], dim=0)
    b_all = torch.cat([b_a, b_b], dim=0)
    pos_prob = (b_all > 0).float().mean(dim=0)
    eps = 1e-6
    bit_entropy = -(pos_prob * torch.log2(pos_prob.clamp_min(eps)) + (1.0 - pos_prob) * torch.log2((1.0 - pos_prob).clamp_min(eps)))
    bit_usage = ((pos_prob > 0.05) & (pos_prob < 0.95)).float().mean()
    bit_balance_abs = u_all.mean(dim=0).abs().mean()
    soft_saturation = u_all.abs().mean()
    soft_std = u_all.std(dim=0, unbiased=False).mean()
    positive_rate = pos_prob.mean()

    metrics = {
        "metric_hash_agree": bit_agreement.item(),
        "metric_hamm_norm": hamming_norm.item(),
        "metric_hash_cos": hash_cos.item(),
        "metric_bit_entropy": bit_entropy.mean().item(),
        "metric_bit_usage": bit_usage.item(),
        "metric_bit_balance_abs": bit_balance_abs.item(),
        "metric_soft_saturation": soft_saturation.item(),
        "metric_soft_std": soft_std.item(),
        "metric_positive_rate": positive_rate.item(),
    }

    if "h_f_a" in outputs and "h_f_b" in outputs:
        metrics["metric_fast_cos"] = F.cosine_similarity(
            outputs["h_f_a"].detach().float(),
            outputs["h_f_b"].detach().float(),
            dim=-1,
        ).mean().item()
    if "z_a" in outputs and "z_b" in outputs:
        metrics["metric_fused_cos"] = F.cosine_similarity(
            outputs["z_a"].detach().float(),
            outputs["z_b"].detach().float(),
            dim=-1,
        ).mean().item()
    if "fast_mask_a" in outputs and "fast_mask_b" in outputs:
        metrics["metric_mask_ratio"] = 0.5 * (
            outputs["fast_mask_a"].detach().float().mean().item()
            + outputs["fast_mask_b"].detach().float().mean().item()
        )
    return metrics
