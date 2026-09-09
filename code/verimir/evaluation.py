"""Image-feature scoring is intentionally separate from label-based metrics."""
import torch
from .retrieval import fused_score, calibrate_scores, strict_v2_metrics_from_scores


def image_scores(appearance, semantic, calibration, appearance_weight=0.5, beta=0.0):
    a = appearance.float() @ appearance.float().T
    s = semantic.float() @ semantic.float().T
    a = calibrate_scores(a, calibration["appearance_mean"], calibration["appearance_std"])
    s = calibrate_scores(s, calibration["semantic_mean"], calibration["semantic_std"])
    return fused_score(a, s, beta, appearance_weight)


def rank_without_self(scores):
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or not torch.isfinite(scores).all():
        raise ValueError("Expected a finite square score matrix")
    scores = scores.clone()
    scores.fill_diagonal_(-float("inf"))
    order = torch.argsort(scores, descending=True, stable=True)
    own = torch.arange(len(scores), device=scores.device)[:, None]
    return order[order.ne(own)].reshape(len(scores), -1)
