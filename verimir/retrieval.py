from __future__ import annotations
from typing import Dict, Optional, Tuple, Sequence, Iterable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def validate_beta(beta: float) -> None:
    if not 0.0 <= float(beta) <= 0.5:
        raise ValueError("beta must be in [0, 0.5]")


def validate_appearance_weight(appearance_weight: float) -> None:
    if not 0.0 <= float(appearance_weight) <= 1.0:
        raise ValueError("appearance_weight must be in [0, 1]")


def calibrate_scores(scores: torch.Tensor, mean: float, std: float, eps: float = 1e-8) -> torch.Tensor:
    return (scores - float(mean)) / (float(std) + eps)


def fused_score(
    ell_v: torch.Tensor,
    ell_s: torch.Tensor,
    beta: float,
    appearance_weight: float = 0.5,
) -> torch.Tensor:
    validate_beta(beta)
    validate_appearance_weight(appearance_weight)
    alpha = float(appearance_weight)
    return alpha * ell_v + (1.0 - alpha) * ell_s - float(beta) * torch.abs(ell_v - ell_s)


def sample_branch_pair_stats(
    appearance: torch.Tensor, semantic: torch.Tensor, num_pairs: int = 100000, seed: int = 0
) -> dict:
    if len(appearance) != len(semantic):
        raise ValueError("descriptor branches have different sizes")
    n = len(appearance)
    if n < 2:
        raise ValueError("need at least two samples")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randint(0, n, (num_pairs,), generator=generator)
    b = torch.randint(0, n - 1, (num_pairs,), generator=generator)
    b = b + (b >= a).long()
    zv = (appearance[a].float() * appearance[b].float()).sum(-1)
    zs = (semantic[a].float() * semantic[b].float()).sum(-1)
    return {
        "appearance_mean": float(zv.mean()),
        "appearance_std": float(zv.std(unbiased=False).clamp_min(1e-8)),
        "semantic_mean": float(zs.mean()),
        "semantic_std": float(zs.std(unbiased=False).clamp_min(1e-8)),
        "branch_abs_disagreement_mean": float((zv - zs).abs().mean()),
        "num_pairs": int(num_pairs),
        "seed": int(seed),
    }


def fit_train_zscore(appearance: torch.Tensor, semantic: torch.Tensor, num_pairs: int = 100000, seed: int = 0) -> dict:
    return sample_branch_pair_stats(appearance, semantic, num_pairs, seed)


@torch.no_grad()
def strict_v2_metrics_from_scores(
    scores: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: torch.Tensor,
    kappas: tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """Apply the project's strict-self-excluded-v2 rules to model scores.

    VeriMIR defines retrieval similarity by its calibrated dual-branch fused
    score, so this score-matrix form preserves the method while matching the
    canonical v2 evaluator's stable ranking, physical query exclusion,
    trapezoidal AP, binary R@K, and fixed-cutoff P@K definitions.
    """
    if scores.shape != (len(labels), len(labels)) or len(sample_ids) != len(labels):
        raise ValueError("strict-v2 requires one square query/gallery score matrix")
    if len(torch.unique(sample_ids)) != len(sample_ids):
        raise ValueError("strict-v2 requires unique query/gallery sample IDs")
    if not torch.isfinite(scores).all():
        raise ValueError("strict-v2 scores contain NaN or infinity")
    kappas = tuple(int(k) for k in kappas)
    if not kappas or any(k <= 0 for k in kappas):
        raise ValueError("strict-v2 kappas must be positive")

    ranked_scores = scores.float().clone()
    ranked_scores.fill_diagonal_(-float("inf"))
    rankings = torch.argsort(ranked_scores, dim=1, descending=True, stable=True)
    query_indices = torch.arange(len(labels), device=rankings.device)
    rankings = rankings[rankings.ne(query_indices[:, None])].reshape(len(labels), -1)
    if rankings.numel() and rankings.eq(query_indices[:, None]).any():
        raise AssertionError("strict-v2 ranking contains its query index")

    ap_sum = 0.0
    class_ap_sum: dict[int, float] = {}
    class_valid_queries: dict[int, int] = {}
    precision_sum = {k: 0.0 for k in kappas}
    recall_sum = {k: 0.0 for k in kappas}
    valid_queries = 0
    for query_index in range(len(labels)):
        relevant = labels.eq(labels[query_index]).clone()
        relevant[query_index] = False
        num_relevant = int(relevant.sum())
        if num_relevant == 0:
            continue
        hits = relevant[rankings[query_index]]
        positions = torch.nonzero(hits, as_tuple=False).flatten().tolist()
        recall_step = 1.0 / num_relevant
        ap = 0.0
        for hit_index, rank in enumerate(positions):
            precision_before = 1.0 if rank == 0 else float(hit_index) / rank
            precision_after = float(hit_index + 1) / (rank + 1)
            ap += (precision_before + precision_after) * recall_step / 2.0
        ap_sum += ap
        query_class = int(labels[query_index].item())
        class_ap_sum[query_class] = class_ap_sum.get(query_class, 0.0) + ap
        class_valid_queries[query_class] = class_valid_queries.get(query_class, 0) + 1
        for kappa in kappas:
            cutoff = min(kappa, len(hits))
            topk = hits[:cutoff]
            precision_sum[kappa] += float(topk.sum()) / max(cutoff, 1)
            recall_sum[kappa] += float(topk.any())
        valid_queries += 1

    denominator = max(valid_queries, 1)
    per_class_map = {
        str(class_index): 100.0 * class_ap_sum[class_index] / class_valid_queries[class_index]
        for class_index in sorted(class_ap_sum)
    }
    metrics = {
        "mAP": 100.0 * ap_sum / denominator,
        "macro_mAP": sum(per_class_map.values()) / max(1, len(per_class_map)),
        "per_class_mAP": per_class_map,
        "num_queries": int(len(labels)),
        "num_valid_queries": int(valid_queries),
    }
    for kappa in kappas:
        metrics[f"R@{kappa}"] = 100.0 * recall_sum[kappa] / denominator
        metrics[f"P@{kappa}"] = 100.0 * precision_sum[kappa] / denominator
    return metrics
