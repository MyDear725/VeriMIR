from __future__ import annotations
from typing import Dict, Optional, Tuple, Sequence, Iterable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .retrieval import fused_score


def caption_loss_class_scale(num_classes: int, reference_classes: int = 3) -> float:
    classes = int(num_classes)
    reference = int(reference_classes)
    if classes < 2 or reference < 2:
        raise ValueError("caption loss class normalization requires at least two classes")
    return float(torch.log(torch.tensor(float(reference))) / torch.log(torch.tensor(float(classes))))


def weighted_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(eps)


def verified_augmentation_gate(
    clean_appearance: torch.Tensor,
    clean_semantic: torch.Tensor,
    augmented_appearance: torch.Tensor,
    augmented_semantic: torch.Tensor,
    caption_weights: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return the detached VeriMIR-v19 augmented-view trust budget.

    Both image branches must agree that the stochastic view preserves the
    clean descriptor.  Existing train-only caption reliability enters once as
    the third verifier.  The returned FP32 gate is deliberately detached so a
    model cannot increase its update budget by learning to manipulate the
    verifier itself.
    """
    tensors = (
        clean_appearance,
        clean_semantic,
        augmented_appearance,
        augmented_semantic,
    )
    if any(value.ndim != 2 for value in tensors):
        raise ValueError("verified augmentation descriptors must be matrices")
    if clean_appearance.shape != augmented_appearance.shape:
        raise ValueError("clean and augmented appearance descriptors must match")
    if clean_semantic.shape != augmented_semantic.shape:
        raise ValueError("clean and augmented semantic descriptors must match")
    if len(clean_appearance) != len(clean_semantic):
        raise ValueError("appearance and semantic batches must have equal length")
    if caption_weights.shape != (len(clean_appearance),):
        raise ValueError("caption weights have incompatible shape")
    if not all(torch.isfinite(value).all() for value in (*tensors, caption_weights)):
        raise ValueError("verified augmentation inputs must be finite")

    appearance_agreement = (
        0.5
        * (
            1.0
            + (clean_appearance.float() * augmented_appearance.float()).sum(-1)
        )
    ).clamp(0.0, 1.0)
    semantic_agreement = (
        0.5
        * (
            1.0
            + (clean_semantic.float() * augmented_semantic.float()).sum(-1)
        )
    ).clamp(0.0, 1.0)
    conservative_agreement = torch.minimum(
        appearance_agreement,
        semantic_agreement,
    )
    gate = (
        caption_weights.detach().float().clamp(0.0, 1.0)
        * conservative_agreement
    ).clamp(0.0, 1.0).detach()
    return gate, {
        "appearance_agreement": appearance_agreement.detach(),
        "semantic_agreement": semantic_agreement.detach(),
        "conservative_agreement": conservative_agreement.detach(),
    }


def verified_two_view_objective(
    clean_loss: torch.Tensor,
    augmented_loss: torch.Tensor,
    gate: torch.Tensor,
    mode: str = "absolute",
) -> torch.Tensor:
    """Aggregate verified views with absolute attenuation or relative allocation.

    ``absolute`` preserves the VeriMIR-v19 mean-g attenuation. ``relative``
    and ``residual`` keep the scalar coefficient of each view loss at one
    half; their detached weights reallocate augmented samples inside the
    criterion.
    """
    if clean_loss.ndim != 0 or augmented_loss.ndim != 0:
        raise ValueError("verified two-view losses must be scalars")
    if gate.ndim != 1 or not len(gate):
        raise ValueError("verified two-view gate must be a nonempty vector")
    if not torch.isfinite(gate).all():
        raise ValueError("verified two-view gate must be finite")
    normalized_mode = str(mode).lower()
    if normalized_mode == "absolute":
        augmented_multiplier = gate.detach().float().mean().to(clean_loss.dtype)
    elif normalized_mode in {"relative", "residual"}:
        augmented_multiplier = clean_loss.new_tensor(1.0)
    else:
        raise ValueError(
            "verified augmentation mode must be absolute, relative, or residual"
        )
    return 0.5 * (clean_loss + augmented_multiplier * augmented_loss)


def verified_augmentation_allocation(
    gate: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Return detached augmented-sample weights for a verification mode.

    ``absolute`` and ``relative`` retain the historical multiplicative gate.
    ``residual`` adds a uniform baseline while preserving gate order and an
    exact unit mean: ``1 - mean(gate) + gate``.
    """
    if gate.ndim != 1 or not len(gate):
        raise ValueError("verified augmentation gate must be a nonempty vector")
    if not torch.isfinite(gate).all():
        raise ValueError("verified augmentation gate must be finite")
    detached_gate = gate.detach().float()
    if torch.any((detached_gate < 0.0) | (detached_gate > 1.0)):
        raise ValueError("verified augmentation gate must be within [0, 1]")
    normalized_mode = str(mode).lower()
    if normalized_mode in {"absolute", "relative"}:
        return detached_gate
    if normalized_mode == "residual":
        return 1.0 - detached_gate.mean() + detached_gate
    raise ValueError(
        "verified augmentation mode must be absolute, relative, or residual"
    )


def image_to_text_contrastive(
    semantic: torch.Tensor,
    text: torch.Tensor,
    weights: torch.Tensor,
    tau: float = 0.07,
    labels: Optional[torch.Tensor] = None,
    multi_positive: bool = False,
) -> torch.Tensor:
    logits = semantic @ text.detach().T / tau
    if multi_positive:
        if labels is None:
            raise ValueError("labels are required for multi-positive image-text contrastive loss")
        positive = labels[:, None].eq(labels[None, :])
        if not positive.any(dim=1).all():
            raise ValueError("every anchor must have at least one positive caption")
        numerator = torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=-1)
        denominator = torch.logsumexp(logits, dim=-1)
        per_sample = denominator - numerator
    else:
        targets = torch.arange(len(semantic), device=semantic.device)
        per_sample = F.cross_entropy(logits, targets, reduction="none")
    return weighted_mean(per_sample, weights)


def supervised_class_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    tau: float = 0.07,
) -> torch.Tensor:
    """Pull every same-class sample together without treating the anchor as a positive."""
    if embeddings.ndim != 2 or labels.ndim != 1 or len(embeddings) != len(labels):
        raise ValueError("supervised contrastive inputs have incompatible shapes")
    if weights.ndim != 1 or len(weights) != len(embeddings):
        raise ValueError("supervised contrastive weights have incompatible shape")
    if tau <= 0:
        raise ValueError("supervised contrastive temperature must be positive")
    logits = embeddings.float() @ embeddings.float().T / float(tau)
    nonself = ~torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
    positives = labels[:, None].eq(labels[None, :]) & nonself
    valid = positives.any(dim=-1)
    if not valid.any():
        return embeddings.sum() * 0.0
    denominator = torch.logsumexp(logits.masked_fill(~nonself, float("-inf")), dim=-1)
    log_probability = logits - denominator[:, None]
    positive_count = positives.sum(dim=-1).clamp_min(1)
    per_sample = -log_probability.masked_fill(~positives, 0.0).sum(dim=-1) / positive_count
    per_sample = per_sample[valid]
    return weighted_mean(per_sample.to(embeddings.dtype), weights[valid])


def angular_margin_classification_loss(
    cosine_logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    scale: float = 16.0,
    margin: float = 0.1,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Training-only cosine classifier that separates retrieval classes angularly."""
    if cosine_logits.ndim != 2 or labels.ndim != 1 or len(cosine_logits) != len(labels):
        raise ValueError("angular classification inputs have incompatible shapes")
    if weights.ndim != 1 or len(weights) != len(labels):
        raise ValueError("angular classification weights have incompatible shape")
    if int(labels.min()) < 0 or int(labels.max()) >= cosine_logits.shape[1]:
        raise ValueError("angular classification label is outside the classifier")
    if scale <= 0 or margin < 0:
        raise ValueError("angular classification scale/margin is invalid")
    effective_weights = weights
    if class_weights is not None:
        class_weights = torch.as_tensor(
            class_weights, device=weights.device, dtype=weights.dtype,
        )
        if class_weights.ndim != 1 or len(class_weights) != cosine_logits.shape[1]:
            raise ValueError("angular classification class weights have incompatible shape")
        if not torch.isfinite(class_weights).all() or torch.any(class_weights <= 0.0):
            raise ValueError("angular classification class weights must be finite and positive")
        effective_weights = weights * class_weights[labels]
    target = F.one_hot(labels, num_classes=cosine_logits.shape[1]).to(cosine_logits.dtype)
    adjusted = float(scale) * (cosine_logits - float(margin) * target)
    per_sample = F.cross_entropy(adjusted.float(), labels, reduction="none")
    return weighted_mean(per_sample.to(cosine_logits.dtype), effective_weights)


def evidential_fit_loss(
    alpha: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Dirichlet mean/variance fit used by the evidential retrieval head."""
    if alpha.ndim != 2 or labels.ndim != 1 or len(alpha) != len(labels):
        raise ValueError("evidential inputs have incompatible shapes")
    if weights.ndim != 1 or len(weights) != len(labels):
        raise ValueError("evidential weights have incompatible shape")
    if int(labels.min()) < 0 or int(labels.max()) >= alpha.shape[1]:
        raise ValueError("evidential label is outside the Dirichlet table")
    if not torch.isfinite(alpha).all() or torch.any(alpha <= 1.0):
        raise ValueError("evidential alpha must be finite and greater than one")
    strength = alpha.float().sum(dim=-1, keepdim=True)
    probability = alpha.float() / strength
    target = F.one_hot(labels, num_classes=alpha.shape[1]).float()
    variance = alpha.float() * (strength - alpha.float()) / (
        strength.square() * (strength + 1.0)
    )
    per_sample = ((target - probability).square() + variance).sum(dim=-1)
    return weighted_mean(per_sample.to(alpha.dtype), weights)


def koleo_uniformity_loss(
    embeddings: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Nearest-neighbor hypersphere spreading regularizer from KoLeo."""
    if embeddings.ndim != 2 or weights.ndim != 1 or len(embeddings) != len(weights):
        raise ValueError("KoLeo inputs have incompatible shapes")
    if len(embeddings) < 2 or eps <= 0.0:
        raise ValueError("KoLeo requires at least two embeddings and positive eps")
    normalized = F.normalize(embeddings.float(), dim=-1)
    distance = torch.cdist(normalized, normalized)
    nonself = torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
    distance = distance.masked_fill(nonself, float("inf"))
    nearest = distance.min(dim=-1).values.clamp_min(float(eps))
    return weighted_mean((-torch.log(nearest)).to(embeddings.dtype), weights)


def image_to_class_prototype_contrastive(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    tau: float = 0.07,
) -> torch.Tensor:
    if embeddings.ndim != 2 or prototypes.ndim != 2 or embeddings.shape[1] != prototypes.shape[1]:
        raise ValueError("image embeddings and caption prototypes have incompatible shapes")
    if labels.ndim != 1 or len(labels) != len(embeddings):
        raise ValueError("prototype contrastive labels have incompatible shape")
    if int(labels.min()) < 0 or int(labels.max()) >= len(prototypes):
        raise ValueError("prototype contrastive label is outside the prototype table")
    logits = embeddings @ prototypes.detach().T / tau
    per_sample = F.cross_entropy(logits, labels, reduction="none")
    return weighted_mean(per_sample, weights)


def global_class_proxy_compactness_loss(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    tau: float = 0.1,
    compactness: float = 0.5,
) -> torch.Tensor:
    """Train-image proxy classification plus within-class angular compactness."""
    if embeddings.ndim != 2 or prototypes.ndim != 2 or embeddings.shape[1] != prototypes.shape[1]:
        raise ValueError("global proxy embeddings and prototypes have incompatible shapes")
    if labels.ndim != 1 or len(labels) != len(embeddings):
        raise ValueError("global proxy labels have incompatible shape")
    if weights.ndim != 1 or len(weights) != len(labels):
        raise ValueError("global proxy weights have incompatible shape")
    if int(labels.min()) < 0 or int(labels.max()) >= len(prototypes):
        raise ValueError("global proxy label is outside the prototype table")
    if tau <= 0.0 or compactness < 0.0:
        raise ValueError("global proxy temperature/compactness is invalid")
    normalized_prototypes = F.normalize(prototypes.detach().float(), dim=-1).to(embeddings.dtype)
    logits = embeddings @ normalized_prototypes.T / float(tau)
    classification = F.cross_entropy(logits.float(), labels, reduction="none")
    target_similarity = (embeddings * normalized_prototypes[labels]).sum(-1).float()
    per_sample = classification + float(compactness) * (1.0 - target_similarity)
    return weighted_mean(per_sample.to(embeddings.dtype), weights)


def caption_class_prototypes(
    text: torch.Tensor,
    labels: torch.Tensor,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    if text.ndim != 2 or labels.ndim != 1 or len(text) != len(labels):
        raise ValueError("caption prototype inputs have incompatible shapes")
    classes = int(labels.max()) + 1 if num_classes is None else int(num_classes)
    if classes <= int(labels.max()):
        raise ValueError("num_classes does not cover all caption labels")
    prototypes = []
    for class_index in range(classes):
        members = text[labels.eq(class_index)]
        if not len(members):
            raise ValueError(f"caption prototype class {class_index} is empty")
        prototypes.append(F.normalize(members.float().mean(dim=0), dim=0))
    return torch.stack(prototypes).to(text.dtype)


def shrink_caption_embeddings(
    text: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    residual_scale: float,
) -> torch.Tensor:
    rho = float(residual_scale)
    if not 0.0 <= rho <= 1.0:
        raise ValueError("caption residual_scale must be in [0, 1]")
    if text.ndim != 2 or prototypes.ndim != 2 or text.shape[1] != prototypes.shape[1]:
        raise ValueError("caption embeddings and prototypes have incompatible shapes")
    if labels.ndim != 1 or len(labels) != len(text):
        raise ValueError("caption labels have incompatible shape")
    if int(labels.min()) < 0 or int(labels.max()) >= len(prototypes):
        raise ValueError("caption label is outside the prototype table")
    centers = prototypes[labels]
    return F.normalize(centers + rho * (text - centers), dim=-1)


def pointwise_semantic_loss(semantic: torch.Tensor, text: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return weighted_mean(1.0 - (semantic * text.detach()).sum(-1), weights)


def neighborhood_alignment_loss(
    semantic: torch.Tensor,
    text: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    tau_text: float = 0.07,
    tau_semantic: float = 0.07,
    gamma: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    n = len(semantic)
    if n < 2:
        return semantic.sum() * 0.0
    self_mask = torch.eye(n, dtype=torch.bool, device=semantic.device)
    teacher_logits = text.detach() @ text.detach().T
    teacher_logits = teacher_logits + gamma * labels[:, None].eq(labels[None, :]).float()
    student_logits = semantic @ semantic.T
    teacher_logits = teacher_logits.masked_fill(self_mask, float("-inf")) / tau_text
    student_logits = student_logits.masked_fill(self_mask, float("-inf")) / tau_semantic
    teacher_prob = F.softmax(teacher_logits, dim=-1).detach()
    student_log_prob = F.log_softmax(student_logits, dim=-1)
    teacher_log = teacher_prob.clamp_min(eps).log()
    per_sample = (teacher_prob * (teacher_log - student_log_prob)).masked_fill(self_mask, 0.0).sum(-1)
    return weighted_mean(per_sample, weights)


def _candidate_from_bank(
    anchor: torch.Tensor,
    label: int,
    sample_id: int,
    bank: Optional[Dict[str, torch.Tensor]],
    positive: bool,
    threshold: float,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if bank is None:
        return None, None
    labels = bank["labels"].to(anchor.device)
    weights = bank["weights"].to(anchor.device)
    ids = bank["sample_ids"].to(anchor.device)
    valid = weights >= threshold
    valid &= labels.eq(label) if positive else labels.ne(label)
    valid &= ids.ne(sample_id)
    if not valid.any():
        return None, None
    embeddings = bank["embeddings"].to(anchor.device)[valid]
    distances = torch.norm(anchor[None, :] - embeddings, dim=-1)
    index = torch.argmax(distances) if positive else torch.argmin(distances)
    return embeddings[index], weights[valid][index]


def self_verified_roadmap_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    anchor_weights: Optional[torch.Tensor] = None,
    tau: float = 0.01,
    rho: float = 100.0,
    delta: float = 0.05,
    positive_threshold: float = 0.9,
    negative_threshold: float = 0.6,
    calibration_mix: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """ROADMAP loss with detached query-level self-verification weights.

    Candidate ranks are the standard, unweighted ROADMAP ranks.  Reliability
    only aggregates the per-query losses, so the verifier cannot alter which
    positive or negative is ranked for an anchor.  This distinction is the
    central V32 protocol constraint.
    """
    if embeddings.ndim != 2:
        raise ValueError("ROADMAP embeddings must be a matrix")
    if labels.ndim != 1 or len(labels) != len(embeddings):
        raise ValueError("ROADMAP labels have incompatible shape")
    if tau <= 0.0 or rho <= 0.0 or delta <= 0.0:
        raise ValueError("ROADMAP smoothing parameters must be positive")
    if not 0.0 <= calibration_mix <= 1.0:
        raise ValueError("ROADMAP calibration_mix must be in [0, 1]")
    if not torch.isfinite(embeddings).all():
        raise ValueError("ROADMAP embeddings must be finite")

    n = len(embeddings)
    if anchor_weights is None:
        anchor_weights = torch.ones(n, device=embeddings.device)
    if anchor_weights.shape != (n,):
        raise ValueError("ROADMAP anchor weights have incompatible shape")
    weights = anchor_weights.detach().to(device=embeddings.device, dtype=torch.float32)
    if not torch.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("ROADMAP anchor weights must be finite and non-negative")

    descriptors = F.normalize(embeddings.float(), dim=-1)
    scores = descriptors @ descriptors.T
    per_query_roadmap = []
    per_query_supap = []
    per_query_calibration = []
    per_query_active_positive = []
    per_query_active_negative = []
    valid_indices = []
    identity = torch.eye(n, dtype=torch.bool, device=labels.device)

    for query in range(n):
        positive_mask = labels.eq(labels[query]) & ~identity[query]
        negative_mask = labels.ne(labels[query])
        if not positive_mask.any() or not negative_mask.any():
            continue

        positive_scores = scores[query, positive_mask]
        negative_scores = scores[query, negative_mask]

        # For every positive k, count positives p that outrank it exactly.
        positive_differences = positive_scores[None, :] - positive_scores[:, None]
        not_same_positive = ~torch.eye(
            len(positive_scores), dtype=torch.bool, device=scores.device,
        )
        positive_rank = 1.0 + (
            (positive_differences.detach() >= 0.0) & not_same_positive
        ).float().sum(dim=1)

        # Smooth the number of negatives whose score is above positive k.
        negative_differences = negative_scores[None, :] - positive_scores[:, None]
        scaled = (negative_differences / tau).clamp(-50.0, 50.0)
        sigmoid = torch.sigmoid(scaled)
        smooth_negative_rank = torch.where(
            negative_differences <= 0.0,
            sigmoid,
            torch.where(
                negative_differences <= delta,
                sigmoid + 0.5,
                rho * (negative_differences - delta)
                + torch.sigmoid(torch.as_tensor(delta / tau, device=scores.device))
                + 0.5,
            ),
        ).sum(dim=1)
        precision = positive_rank / (positive_rank + smooth_negative_rank)
        supap = 1.0 - precision.mean()

        positive_calibration = F.relu(positive_threshold - positive_scores)
        negative_calibration = F.relu(negative_scores - negative_threshold)
        calibration = positive_calibration.mean() + negative_calibration.mean()
        roadmap = (1.0 - calibration_mix) * supap + calibration_mix * calibration

        per_query_roadmap.append(roadmap)
        per_query_supap.append(supap)
        per_query_calibration.append(calibration)
        per_query_active_positive.append((positive_calibration > 0.0).float().mean())
        per_query_active_negative.append((negative_calibration > 0.0).float().mean())
        valid_indices.append(query)

    if not per_query_roadmap:
        zero = embeddings.sum() * 0.0
        empty = torch.empty(0, device=embeddings.device, dtype=torch.float32)
        return zero, {
            "supap": zero.detach(),
            "calibration": zero.detach(),
            "active_positive_fraction": zero.detach(),
            "active_negative_fraction": zero.detach(),
            "query_std": zero.detach(),
            "anchor_ess_ratio": zero.detach(),
            "valid_query_fraction": zero.detach(),
            "per_query_roadmap": empty,
            "valid_indices": torch.empty(0, device=labels.device, dtype=torch.long),
        }

    per_query = torch.stack(per_query_roadmap)
    supap_values = torch.stack(per_query_supap)
    calibration_values = torch.stack(per_query_calibration)
    active_positive = torch.stack(per_query_active_positive)
    active_negative = torch.stack(per_query_active_negative)
    valid_indices_tensor = torch.as_tensor(valid_indices, device=labels.device, dtype=torch.long)
    valid_weights = weights[valid_indices_tensor]
    if valid_weights.sum() <= 0.0:
        raise ValueError("ROADMAP valid query weights must have positive sum")
    weight_square_sum = valid_weights.square().sum().clamp_min(1e-8)
    ess_ratio = valid_weights.sum().square() / (len(valid_weights) * weight_square_sum)

    return weighted_mean(per_query, valid_weights), {
        "supap": weighted_mean(supap_values, valid_weights).detach(),
        "calibration": weighted_mean(calibration_values, valid_weights).detach(),
        "active_positive_fraction": weighted_mean(active_positive, valid_weights).detach(),
        "active_negative_fraction": weighted_mean(active_negative, valid_weights).detach(),
        "query_std": per_query.detach().std(unbiased=False),
        "anchor_ess_ratio": ess_ratio.detach(),
        "valid_query_fraction": torch.as_tensor(
            len(valid_indices) / max(1, n), device=embeddings.device,
        ),
        "per_query_roadmap": per_query.detach(),
        "valid_indices": valid_indices_tensor,
    }


@torch.no_grad()
def build_bidirectional_top5_teacher(
    bank: Dict[str, torch.Tensor | dict | int],
    *,
    beta: float,
    appearance_weight: float,
    top_k: int = 5,
    chunk_size: int = 512,
) -> Dict[str, torch.Tensor | dict | int]:
    """Materialize the frozen sparse teacher used by V140 SV-BT5SD.

    Only train-snapshot global image descriptors are consumed.  The full score
    matrix is never retained: each query chunk contributes its branch/fused
    fifth cutoffs and fused top-k indices.
    """
    if int(bank.get("snapshot_version", -1)) != int(
        bank.get("calibration_version", -2)
    ):
        raise ValueError("SV-BT5SD bank/calibration version mismatch")
    if int(top_k) < 1 or int(chunk_size) < 1:
        raise ValueError("SV-BT5SD top_k and chunk_size must be positive")
    appearance = bank.get("appearance")
    semantic = bank.get("semantic")
    labels = bank.get("labels")
    sample_ids = bank.get("sample_ids")
    calibration = bank.get("calibration")
    if not all(isinstance(value, torch.Tensor) for value in (
        appearance, semantic, labels, sample_ids,
    )):
        raise TypeError("SV-BT5SD bank must contain tensor descriptors and metadata")
    if not isinstance(calibration, dict):
        raise TypeError("SV-BT5SD bank calibration must be a dictionary")
    if appearance.ndim != 2 or semantic.ndim != 2 or len(appearance) != len(semantic):
        raise ValueError("SV-BT5SD teacher descriptors must be aligned matrices")
    size = len(appearance)
    if size <= 2 * int(top_k):
        raise ValueError("SV-BT5SD teacher requires more than 2*top_k samples")
    if labels.shape != (size,) or sample_ids.shape != (size,):
        raise ValueError("SV-BT5SD teacher metadata has incompatible shape")
    if len(torch.unique(sample_ids)) != size:
        raise ValueError("SV-BT5SD teacher sample IDs must be unique")

    appearance = F.normalize(appearance.float(), dim=-1)
    semantic = F.normalize(semantic.float(), dim=-1)
    appearance_std = float(calibration["appearance_std"])
    semantic_std = float(calibration["semantic_std"])
    if appearance_std <= 0.0 or semantic_std <= 0.0:
        raise ValueError("SV-BT5SD calibration std must be positive")
    device = appearance.device
    top_indices = torch.empty(size, int(top_k), dtype=torch.long, device=device)
    boundary_indices = torch.empty(
        size, int(top_k), dtype=torch.long, device=device,
    )
    appearance_cutoff = torch.empty(size, dtype=torch.float32, device=device)
    semantic_cutoff = torch.empty(size, dtype=torch.float32, device=device)
    fused_cutoff = torch.empty(size, dtype=torch.float32, device=device)
    for start in range(0, size, int(chunk_size)):
        end = min(start + int(chunk_size), size)
        raw_appearance = appearance[start:end] @ appearance.T
        raw_semantic = semantic[start:end] @ semantic.T
        calibrated_appearance = (
            raw_appearance - float(calibration["appearance_mean"])
        ) / appearance_std
        calibrated_semantic = (
            raw_semantic - float(calibration["semantic_mean"])
        ) / semantic_std
        fused = fused_score(
            calibrated_appearance,
            calibrated_semantic,
            float(beta),
            float(appearance_weight),
        )
        rows = torch.arange(end - start, device=device)
        columns = torch.arange(start, end, device=device)
        calibrated_appearance[rows, columns] = -float("inf")
        calibrated_semantic[rows, columns] = -float("inf")
        fused[rows, columns] = -float("inf")
        appearance_cutoff[start:end] = torch.topk(
            calibrated_appearance, int(top_k), dim=1, sorted=True,
        ).values[:, -1]
        semantic_cutoff[start:end] = torch.topk(
            calibrated_semantic, int(top_k), dim=1, sorted=True,
        ).values[:, -1]
        fused_top = torch.topk(fused, 2 * int(top_k), dim=1, sorted=True)
        fused_cutoff[start:end] = fused_top.values[:, int(top_k) - 1]
        top_indices[start:end] = fused_top.indices[:, :int(top_k)]
        boundary_indices[start:end] = fused_top.indices[
            :, int(top_k):2 * int(top_k)
        ]
    return {
        "top_k": int(top_k),
        "snapshot_version": int(bank["snapshot_version"]),
        "top_indices": top_indices,
        "boundary_indices": boundary_indices,
        "appearance_cutoff": appearance_cutoff,
        "semantic_cutoff": semantic_cutoff,
        "fused_cutoff": fused_cutoff,
        "labels": labels.detach().long(),
        "sample_ids": sample_ids.detach().long(),
        "calibration": dict(calibration),
    }


def self_verified_bidirectional_top5_support_loss(
    clean_appearance: torch.Tensor,
    clean_semantic: torch.Tensor,
    augmented_appearance: torch.Tensor,
    augmented_semantic: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: torch.Tensor,
    bank_indices: torch.Tensor,
    bank: Dict[str, torch.Tensor | dict | int],
    teacher: Dict[str, torch.Tensor | dict | int],
    *,
    beta: float,
    appearance_weight: float,
    margin: float = 0.02,
    huber_delta: float = 0.05,
    gate_floor: float = 0.25,
    gate_temperature: float = 0.10,
    disagreement_temperature: float = 0.50,
    include_incoming: bool = True,
    incoming_weight: float = 1.0,
    adaptive_incoming: bool = False,
    incoming_weight_min: float = 0.025,
    incoming_weight_max: float = 0.10,
    incoming_boundary_pivot: float = 0.25,
    incoming_boundary_temperature: float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """V140 self-verified bidirectional top-5 support distillation.

    The outgoing half repairs the exact first-positive/top-5 boundary on the
    worse train view.  The incoming half preserves sparse teacher-positive
    support and excludes new cross-class intrusions while the current sample
    acts as a gallery candidate.  The teacher and all gates are detached.
    """
    tensors = (
        clean_appearance, clean_semantic,
        augmented_appearance, augmented_semantic,
    )
    if any(value.ndim != 2 for value in tensors):
        raise ValueError("SV-BT5SD descriptors must be matrices")
    if clean_appearance.shape != augmented_appearance.shape:
        raise ValueError("SV-BT5SD appearance views must match")
    if clean_semantic.shape != augmented_semantic.shape:
        raise ValueError("SV-BT5SD semantic views must match")
    batch_size = len(clean_appearance)
    if clean_semantic.shape[0] != batch_size:
        raise ValueError("SV-BT5SD branches must have equal batch size")
    if any(value.shape != (batch_size,) for value in (
        labels, sample_ids, bank_indices,
    )):
        raise ValueError("SV-BT5SD query metadata has incompatible shape")
    if not math.isfinite(float(margin)) or float(margin) < 0.0:
        raise ValueError("SV-BT5SD margin must be finite and non-negative")
    if not math.isfinite(float(huber_delta)) or float(huber_delta) <= 0.0:
        raise ValueError("SV-BT5SD huber_delta must be finite and positive")
    if not 0.0 <= float(gate_floor) <= 1.0:
        raise ValueError("SV-BT5SD gate_floor must be in [0, 1]")
    if float(gate_temperature) <= 0.0 or float(disagreement_temperature) <= 0.0:
        raise ValueError("SV-BT5SD gate temperatures must be positive")
    if not math.isfinite(float(incoming_weight)) or float(incoming_weight) < 0.0:
        raise ValueError("SV-BT5SD incoming_weight must be finite and non-negative")
    if not all(math.isfinite(float(value)) for value in (
        incoming_weight_min,
        incoming_weight_max,
        incoming_boundary_pivot,
        incoming_boundary_temperature,
    )):
        raise ValueError("SV-BT5SD adaptive incoming parameters must be finite")
    if float(incoming_weight_min) < 0.0:
        raise ValueError("SV-BT5SD incoming_weight_min must be non-negative")
    if float(incoming_weight_max) < float(incoming_weight_min):
        raise ValueError(
            "SV-BT5SD incoming_weight_max must be >= incoming_weight_min"
        )
    if float(incoming_boundary_temperature) <= 0.0:
        raise ValueError(
            "SV-BT5SD incoming_boundary_temperature must be positive"
        )
    top_k = int(teacher.get("top_k", -1))
    if top_k != 5:
        raise ValueError("SV-BT5SD frozen contract requires top_k=5")
    if int(teacher.get("snapshot_version", -1)) != int(bank.get("snapshot_version", -2)):
        raise ValueError("SV-BT5SD teacher and loss bank versions differ")

    bank_appearance = bank.get("appearance")
    bank_semantic = bank.get("semantic")
    bank_labels = bank.get("labels")
    bank_ids = bank.get("sample_ids")
    calibration = bank.get("calibration")
    top_indices = teacher.get("top_indices")
    boundary_indices = teacher.get("boundary_indices")
    appearance_cutoff = teacher.get("appearance_cutoff")
    semantic_cutoff = teacher.get("semantic_cutoff")
    fused_cutoff = teacher.get("fused_cutoff")
    if not all(isinstance(value, torch.Tensor) for value in (
        bank_appearance, bank_semantic, bank_labels, bank_ids,
        top_indices, boundary_indices,
        appearance_cutoff, semantic_cutoff, fused_cutoff,
    )):
        raise TypeError("SV-BT5SD bank/teacher tensors are incomplete")
    if not isinstance(calibration, dict):
        raise TypeError("SV-BT5SD calibration must be a dictionary")
    bank_size = len(bank_appearance)
    if top_indices.shape != (bank_size, top_k):
        raise ValueError("SV-BT5SD teacher top-index shape is invalid")
    if boundary_indices.shape != (bank_size, top_k):
        raise ValueError("SV-BT5SD teacher boundary-index shape is invalid")
    if any(value.shape != (bank_size,) for value in (
        bank_labels, bank_ids, appearance_cutoff, semantic_cutoff, fused_cutoff,
    )):
        raise ValueError("SV-BT5SD bank/teacher metadata shape is invalid")
    if int(bank_indices.min()) < 0 or int(bank_indices.max()) >= bank_size:
        raise ValueError("SV-BT5SD bank indices are out of range")

    device_type = clean_appearance.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        bank_appearance = F.normalize(
            bank_appearance.detach().to(clean_appearance.device).float(), dim=-1,
        )
        bank_semantic = F.normalize(
            bank_semantic.detach().to(clean_semantic.device).float(), dim=-1,
        )
        bank_labels = bank_labels.detach().to(labels.device).long()
        bank_ids = bank_ids.detach().to(sample_ids.device).long()
        top_indices = top_indices.detach().to(labels.device).long()
        boundary_indices = boundary_indices.detach().to(labels.device).long()
        appearance_cutoff = appearance_cutoff.detach().to(clean_appearance.device).float()
        semantic_cutoff = semantic_cutoff.detach().to(clean_appearance.device).float()
        fused_cutoff = fused_cutoff.detach().to(clean_appearance.device).float()
        calibration = {key: float(value) for key, value in calibration.items()}

        def view_scores(appearance: torch.Tensor, semantic: torch.Tensor):
            raw_a = F.normalize(appearance.float(), dim=-1) @ bank_appearance.T
            raw_s = F.normalize(semantic.float(), dim=-1) @ bank_semantic.T
            cal_a = (raw_a - calibration["appearance_mean"]) / calibration["appearance_std"]
            cal_s = (raw_s - calibration["semantic_mean"]) / calibration["semantic_std"]
            fused = fused_score(cal_a, cal_s, float(beta), float(appearance_weight))
            return cal_a, cal_s, fused

        clean_a, clean_s, clean_f = view_scores(clean_appearance, clean_semantic)
        aug_a, aug_s, aug_f = view_scores(augmented_appearance, augmented_semantic)
        same_id = sample_ids[:, None].eq(bank_ids[None, :])
        if not torch.all(same_id.sum(dim=1).eq(1)):
            raise ValueError("SV-BT5SD snapshot must contain every query exactly once")
        positive = labels[:, None].eq(bank_labels[None, :]) & ~same_id
        negative = labels[:, None].ne(bank_labels[None, :]) & ~same_id
        if not torch.all(positive.sum(dim=1).ge(1) & negative.sum(dim=1).ge(top_k)):
            raise ValueError("SV-BT5SD requires one positive and five negatives per query")

        def stable_kth(scores: torch.Tensor, mask: torch.Tensor, rank: int):
            return torch.sort(
                scores.masked_fill(~mask, -float("inf")),
                dim=1, descending=True, stable=True,
            ).values[:, int(rank) - 1]

        def coverage_boundaries(a: torch.Tensor, s: torch.Tensor, f: torch.Tensor):
            return (
                stable_kth(a, positive, 1) - stable_kth(a, negative, top_k),
                stable_kth(s, positive, 1) - stable_kth(s, negative, top_k),
                stable_kth(f, positive, 1) - stable_kth(f, negative, top_k),
            )

        clean_boundary = coverage_boundaries(clean_a, clean_s, clean_f)
        aug_boundary = coverage_boundaries(aug_a, aug_s, aug_f)
        use_aug = aug_boundary[2].detach().lt(clean_boundary[2].detach())
        outgoing_a = torch.where(use_aug, aug_boundary[0], clean_boundary[0])
        outgoing_s = torch.where(use_aug, aug_boundary[1], clean_boundary[1])
        outgoing_f = torch.where(use_aug, aug_boundary[2], clean_boundary[2])
        outgoing_disagreement = torch.abs(
            torch.tanh(outgoing_a.detach() / float(gate_temperature))
            - torch.tanh(outgoing_s.detach() / float(gate_temperature))
        )
        outgoing_gate = (
            float(gate_floor)
            + (1.0 - float(gate_floor))
            * torch.exp(-outgoing_disagreement / float(disagreement_temperature))
        ).detach()

        def positive_huber(violation: torch.Tensor) -> torch.Tensor:
            violation = torch.relu(violation)
            delta = float(huber_delta)
            return torch.where(
                violation.lt(delta),
                0.5 * violation.square() / delta,
                violation - 0.5 * delta,
            )

        outgoing_per_query = outgoing_gate * positive_huber(
            float(margin) - outgoing_f
        )
        outgoing_active = outgoing_f.lt(float(margin))
        outgoing_class_losses = []
        for class_index in torch.unique(labels, sorted=True):
            class_mask = labels.eq(class_index)
            outgoing_class_losses.append(outgoing_per_query[class_mask].mean())
        outgoing_loss = torch.stack(outgoing_class_losses).mean()
        boundary_reliability = torch.sigmoid(
            (
                outgoing_f.detach().mean()
                - float(incoming_boundary_pivot)
            ) / float(incoming_boundary_temperature)
        )
        effective_incoming_weight = torch.as_tensor(
            float(incoming_weight),
            device=outgoing_loss.device,
            dtype=outgoing_loss.dtype,
        )
        if bool(adaptive_incoming):
            effective_incoming_weight = (
                float(incoming_weight_min)
                + (
                    float(incoming_weight_max)
                    - float(incoming_weight_min)
                ) * boundary_reliability
            ).detach()

        incoming_loss = outgoing_loss * 0.0
        incoming_positive_active = torch.zeros((), device=outgoing_loss.device)
        incoming_intrusion_active = torch.zeros((), device=outgoing_loss.device)
        incoming_gate_values = torch.ones(1, device=outgoing_loss.device)
        if bool(include_incoming):
            batch_bank_indices = bank_indices.detach().to(labels.device).long()
            current_fused = clean_f.T
            teacher_a = (
                bank_appearance @ bank_appearance[batch_bank_indices].T
                - calibration["appearance_mean"]
            ) / calibration["appearance_std"]
            teacher_s = (
                bank_semantic @ bank_semantic[batch_bank_indices].T
                - calibration["semantic_mean"]
            ) / calibration["semantic_std"]
            in_top5 = top_indices[:, :, None].eq(
                batch_bank_indices[None, None, :]
            ).any(dim=1)
            in_boundary_band = boundary_indices[:, :, None].eq(
                batch_bank_indices[None, None, :]
            ).any(dim=1)
            same_class = bank_labels[:, None].eq(labels[None, :])
            positive_support = same_class & in_top5
            intrusion_eligible = ~same_class & in_boundary_band
            incoming_disagreement = torch.abs(
                torch.tanh(
                    (teacher_a - appearance_cutoff[:, None])
                    / float(gate_temperature)
                )
                - torch.tanh(
                    (teacher_s - semantic_cutoff[:, None])
                    / float(gate_temperature)
                )
            )
            incoming_gates = (
                float(gate_floor)
                + (1.0 - float(gate_floor))
                * torch.exp(
                    -incoming_disagreement / float(disagreement_temperature)
                )
            ).detach()
            # A protected positive only needs to remain inside the teacher
            # support boundary.  The explicit margin is reserved for the
            # rank-6..10 cross-class intrusion guard.
            positive_violation = fused_cutoff[:, None] - current_fused
            intrusion_violation = float(margin) + current_fused - fused_cutoff[:, None]
            positive_values = incoming_gates * positive_huber(positive_violation)
            intrusion_values = incoming_gates * positive_huber(intrusion_violation)
            positive_active_mask = positive_support & positive_violation.gt(0.0)
            intrusion_active_mask = intrusion_eligible & intrusion_violation.gt(0.0)
            incoming_positive_active = positive_active_mask.float().mean()
            incoming_intrusion_active = intrusion_active_mask.float().mean()
            incoming_gate_values = incoming_gates[
                positive_support | intrusion_eligible
            ]
            if incoming_gate_values.numel() == 0:
                incoming_gate_values = incoming_gates.reshape(-1)[:1]
            sample_losses = []
            for column in range(batch_size):
                # Normalize over every frozen valid edge, including already
                # satisfied zero-loss edges.  Conditioning the denominator on
                # violations would inflate the incoming branch whenever only
                # a few edges cross the boundary and would violate the frozen
                # per-candidate valid-edge reduction contract.
                pos_mask = positive_support[:, column]
                neg_mask = intrusion_eligible[:, column]
                pos_loss = (
                    positive_values[pos_mask, column].mean()
                    if pos_mask.any() else outgoing_loss * 0.0
                )
                neg_loss = (
                    intrusion_values[neg_mask, column].mean()
                    if neg_mask.any() else outgoing_loss * 0.0
                )
                sample_losses.append(0.5 * (pos_loss + neg_loss))
            sample_losses = torch.stack(sample_losses)
            incoming_class_losses = []
            for class_index in torch.unique(labels, sorted=True):
                incoming_class_losses.append(
                    sample_losses[labels.eq(class_index)].mean()
                )
            incoming_loss = torch.stack(incoming_class_losses).mean()

        total = (
            (
                outgoing_loss
                + effective_incoming_weight * incoming_loss
            ) / (1.0 + effective_incoming_weight)
            if bool(include_incoming) else outgoing_loss
        )
        gate_weights = outgoing_gate / outgoing_gate.sum().clamp_min(1e-8)
        outgoing_ess_ratio = 1.0 / (
            gate_weights.square().sum().clamp_min(1e-8) * float(batch_size)
        )
        details = {
            "outgoing_loss": outgoing_loss.detach(),
            "incoming_loss": incoming_loss.detach(),
            "outgoing_active_fraction": outgoing_active.float().mean().detach(),
            "outgoing_worst_aug_fraction": use_aug.float().mean().detach(),
            "incoming_positive_active_fraction": incoming_positive_active.detach(),
            "incoming_intrusion_active_fraction": incoming_intrusion_active.detach(),
            "outgoing_gate_mean": outgoing_gate.mean().detach(),
            "outgoing_gate_std": outgoing_gate.std(unbiased=False).detach(),
            "incoming_gate_mean": incoming_gate_values.mean().detach(),
            "incoming_gate_std": incoming_gate_values.std(unbiased=False).detach(),
            "gate_ess_ratio": outgoing_ess_ratio.detach(),
            "outgoing_boundary_mean": outgoing_f.mean().detach(),
            "outgoing_boundary_std": outgoing_f.std(unbiased=False).detach(),
            "include_incoming": torch.as_tensor(
                float(bool(include_incoming)), device=total.device,
            ),
            "incoming_weight": effective_incoming_weight.detach(),
            "configured_incoming_weight": torch.as_tensor(
                float(incoming_weight), device=total.device,
            ),
            "adaptive_incoming": torch.as_tensor(
                float(bool(adaptive_incoming)), device=total.device,
            ),
            "incoming_boundary_reliability": boundary_reliability.detach(),
        }
    return total, details


def weighted_batch_hard_triplet(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    sample_ids: torch.Tensor,
    margin: float = 0.2,
    threshold: float = 0.7,
    bank: Optional[Dict[str, torch.Tensor]] = None,
    bank_competes: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    distances = torch.cdist(embeddings.float(), embeddings.float())
    losses, omegas = [], []
    skipped = 0
    bank_positive_selected = 0
    bank_negative_selected = 0
    for i in range(len(embeddings)):
        reliable = weights >= threshold
        positive_mask = labels.eq(labels[i]) & reliable
        positive_mask[i] = False
        negative_mask = labels.ne(labels[i]) & reliable
        if positive_mask.any():
            p_idx = torch.argmax(distances[i].masked_fill(~positive_mask, float("-inf")))
            positive, wp = embeddings[p_idx], weights[p_idx]
            if bank_competes:
                bank_positive, bank_wp = _candidate_from_bank(
                    embeddings[i], int(labels[i]), int(sample_ids[i]), bank, True, threshold,
                )
                if bank_positive is not None and (
                    torch.norm(embeddings[i] - bank_positive).detach()
                    > torch.norm(embeddings[i] - positive).detach()
                ):
                    positive, wp = bank_positive, bank_wp
                    bank_positive_selected += 1
        else:
            positive, wp = _candidate_from_bank(embeddings[i], int(labels[i]), int(sample_ids[i]), bank, True, threshold)
            bank_positive_selected += int(positive is not None)
        if negative_mask.any():
            n_idx = torch.argmin(distances[i].masked_fill(~negative_mask, float("inf")))
            negative, wn = embeddings[n_idx], weights[n_idx]
            if bank_competes:
                bank_negative, bank_wn = _candidate_from_bank(
                    embeddings[i], int(labels[i]), int(sample_ids[i]), bank, False, threshold,
                )
                if bank_negative is not None and (
                    torch.norm(embeddings[i] - bank_negative).detach()
                    < torch.norm(embeddings[i] - negative).detach()
                ):
                    negative, wn = bank_negative, bank_wn
                    bank_negative_selected += 1
        else:
            negative, wn = _candidate_from_bank(embeddings[i], int(labels[i]), int(sample_ids[i]), bank, False, threshold)
            bank_negative_selected += int(negative is not None)
        if positive is None or negative is None:
            skipped += 1
            continue
        omega = weights[i] * wp * wn
        value = F.relu(margin + torch.norm(embeddings[i] - positive) - torch.norm(embeddings[i] - negative))
        losses.append(value)
        omegas.append(omega)
    if not losses:
        return embeddings.sum() * 0.0, {
            "skipped_anchor_ratio": 1.0,
            "eligible_ratio": float((weights >= threshold).float().mean()),
            "bank_positive_selected_ratio": bank_positive_selected / max(1, len(embeddings)),
            "bank_negative_selected_ratio": bank_negative_selected / max(1, len(embeddings)),
        }
    return weighted_mean(torch.stack(losses), torch.stack(omegas)), {
        "skipped_anchor_ratio": skipped / len(embeddings),
        "eligible_ratio": float((weights >= threshold).float().mean()),
        "bank_positive_selected_ratio": bank_positive_selected / len(embeddings),
        "bank_negative_selected_ratio": bank_negative_selected / len(embeddings),
    }


def disagreement_hard_mining_loss(
    appearance: torch.Tensor,
    semantic: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    margin: float = 0.2,
    reliable_threshold: float = 0.7,
    high_quantile: float = 0.8,
    low_quantile: float = 0.2,
) -> torch.Tensor:
    n = len(labels)
    if n < 3:
        return appearance.sum() * 0.0
    zv, zs = appearance @ appearance.T, semantic @ semantic.T
    negative = labels[:, None].ne(labels[None, :])
    reliable_pair = (weights[:, None] >= reliable_threshold) & (weights[None, :] >= reliable_threshold)
    candidates = negative & reliable_pair
    if not candidates.any():
        return appearance.sum() * 0.0
    quantiles = torch.tensor([high_quantile, low_quantile], device=zv.device, dtype=torch.float32)
    qv_hi, qv_lo = torch.quantile(zv[candidates].detach().float(), quantiles)
    qs_hi, qs_lo = torch.quantile(zs[candidates].detach().float(), quantiles)
    terms, term_weights = [], []
    for i in range(n):
        positives = labels.eq(labels[i]) & (weights >= reliable_threshold)
        positives[i] = False
        if not positives.any():
            continue
        pos_v = zv[i][positives].min()
        pos_s = zs[i][positives].min()
        v_only = candidates[i] & (zv[i] >= qv_hi) & (zs[i] <= qs_lo)
        s_only = candidates[i] & (zs[i] >= qs_hi) & (zv[i] <= qv_lo)
        if v_only.any():
            vals = F.relu(margin + zv[i][v_only] - pos_v)
            terms.extend(vals.unbind())
            term_weights.extend((weights[i] * weights[v_only]).unbind())
        if s_only.any():
            vals = F.relu(margin + zs[i][s_only] - pos_s)
            terms.extend(vals.unbind())
            term_weights.extend((weights[i] * weights[s_only]).unbind())
    if not terms:
        return appearance.sum() * 0.0
    return weighted_mean(torch.stack(terms), torch.stack(term_weights))


def fused_ranking_loss(
    appearance: torch.Tensor,
    semantic: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    beta: float,
    margin: float = 0.2,
    threshold: float = 0.7,
    appearance_weight: float = 0.5,
) -> torch.Tensor:
    score = fused_score(
        appearance @ appearance.T,
        semantic @ semantic.T,
        beta,
        appearance_weight,
    )
    losses, anchor_weights = [], []
    for i in range(len(labels)):
        reliable = weights >= threshold
        positives = labels.eq(labels[i]) & reliable
        positives[i] = False
        negatives = labels.ne(labels[i]) & reliable
        if not positives.any() or not negatives.any():
            continue
        pos = score[i][positives].min()
        neg = score[i][negatives].max()
        losses.append(F.relu(margin - pos + neg))
        anchor_weights.append(weights[i])
    if not losses:
        return appearance.sum() * 0.0
    return weighted_mean(torch.stack(losses), torch.stack(anchor_weights))


class CompositeLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        appearance: torch.Tensor,
        semantic: torch.Tensor,
        text: torch.Tensor,
        labels: torch.Tensor,
        sample_ids: torch.Tensor,
        weight_y: torch.Tensor,
        weight_c: torch.Tensor,
        bank: Optional[Dict[str, torch.Tensor]] = None,
        caption_prototypes: Optional[torch.Tensor] = None,
        appearance_logits: Optional[torch.Tensor] = None,
        appearance_prototypes: Optional[torch.Tensor] = None,
        evidential_alpha: Optional[torch.Tensor] = None,
    ):
        c = self.cfg
        zero = appearance.sum() * 0.0
        mining_threshold = c.get("mining_threshold", 0.7) if c.get("reliable_mining", True) else 0.0
        tri, stats = weighted_batch_hard_triplet(
            appearance, labels, weight_y, sample_ids,
            margin=c["triplet_margin"], threshold=mining_threshold, bank=bank,
            bank_competes=bool(c.get("global_bank_mining", False)),
        )
        if c.get("lambda_semantic_triplet", 0.0):
            semantic_tri, _ = weighted_batch_hard_triplet(
                semantic,
                labels,
                weight_y,
                sample_ids,
                margin=c["triplet_margin"],
                threshold=mining_threshold,
                bank=None,
            )
        else:
            semantic_tri = zero
        appearance_supcon = supervised_class_contrastive_loss(
            appearance,
            labels,
            weight_y,
            c.get("tau_supcon", 0.07),
        ) if c.get("lambda_appearance_supcon", 0.0) else zero
        appearance_batch_all, _ = batch_all_triplet_loss(
            appearance,
            labels,
            margin=c["triplet_margin"],
        ) if c.get("lambda_appearance_batch_all", 0.0) else (zero, zero.detach())
        semantic_supcon = supervised_class_contrastive_loss(
            semantic,
            labels,
            weight_y,
            c.get("tau_supcon", 0.07),
        ) if c.get("lambda_semantic_supcon", 0.0) else zero
        if c.get("lambda_appearance_classification", 0.0):
            if appearance_logits is None:
                raise ValueError("appearance logits are required for angular classification")
            appearance_classification = angular_margin_classification_loss(
                appearance_logits,
                labels,
                weight_y,
                c.get("classification_scale", 16.0),
                c.get("classification_margin", 0.1),
                c.get("classification_class_weights"),
            )
        else:
            appearance_classification = zero
        if c.get("lambda_evidential", 0.0):
            if evidential_alpha is None:
                raise ValueError("evidential alpha is required for evidential loss")
            evidential = evidential_fit_loss(
                evidential_alpha, labels, weight_y,
            )
        else:
            evidential = zero
        koleo = koleo_uniformity_loss(
            appearance, weight_y, c.get("koleo_eps", 1e-4),
        ) if c.get("lambda_koleo", 0.0) else zero
        if c.get("lambda_global_proxy", 0.0) and appearance_prototypes is not None:
            global_proxy = global_class_proxy_compactness_loss(
                appearance,
                appearance_prototypes,
                labels,
                weight_y,
                c.get("global_proxy_tau", 0.1),
                c.get("global_proxy_compactness", 0.5),
            )
        else:
            global_proxy = zero
        if c.get("lambda_roadmap", 0.0):
            roadmap, roadmap_stats = self_verified_roadmap_loss(
                appearance,
                labels,
                anchor_weights=weight_y,
                tau=c.get("roadmap_tau", 0.01),
                rho=c.get("roadmap_rho", 100.0),
                delta=c.get("roadmap_delta", 0.05),
                positive_threshold=c.get("roadmap_positive_threshold", 0.9),
                negative_threshold=c.get("roadmap_negative_threshold", 0.6),
                calibration_mix=c.get("roadmap_calibration_mix", 0.5),
            )
        else:
            roadmap = zero
            roadmap_stats = {
                name: zero.detach() for name in (
                    "supap", "calibration", "active_positive_fraction",
                    "active_negative_fraction", "query_std", "anchor_ess_ratio",
                    "valid_query_fraction",
                )
            }
        itc_mode = c.get("itc_mode", "instance")
        if itc_mode not in {"instance", "class_multi_positive", "class_prototype"}:
            raise ValueError(f"unknown itc_mode: {itc_mode}")
        if c["lambda_itc"] and itc_mode == "class_prototype":
            if caption_prototypes is None:
                raise ValueError("caption prototypes are required for class_prototype ITC")
            itc = image_to_class_prototype_contrastive(
                semantic, caption_prototypes, labels, weight_c, c["tau_itc"],
            )
        elif c["lambda_itc"]:
            itc = image_to_text_contrastive(
                semantic,
                text,
                weight_c,
                c["tau_itc"],
                labels=labels,
                multi_positive=itc_mode == "class_multi_positive",
            )
        else:
            itc = zero
        sem = pointwise_semantic_loss(semantic, text, weight_c) if c["lambda_semantic"] else zero
        nad = neighborhood_alignment_loss(
            semantic, text, labels, weight_c,
            c["tau_text_neighbor"], c["tau_semantic_neighbor"], c.get("gamma", 0.0),
        ) if c["lambda_nad"] else zero
        dhm = disagreement_hard_mining_loss(
            appearance, semantic, labels, weight_y, c["dhm_margin"],
            mining_threshold, c.get("dhm_high_quantile", 0.8), c.get("dhm_low_quantile", 0.2),
        ) if c["lambda_dhm"] else zero
        rank = fused_ranking_loss(
            appearance, semantic, labels, weight_y, c["beta"], c["rank_margin"], mining_threshold,
            c.get("appearance_weight", 0.5),
        ) if c["lambda_rank"] else zero
        values = {
            "triplet": tri,
            "semantic_triplet": semantic_tri,
            "appearance_supcon": appearance_supcon,
            "appearance_batch_all": appearance_batch_all,
            "semantic_supcon": semantic_supcon,
            "appearance_classification": appearance_classification,
            "evidential": evidential,
            "koleo": koleo,
            "global_proxy": global_proxy,
            "roadmap": roadmap,
            "roadmap_supap": roadmap_stats["supap"],
            "roadmap_calibration": roadmap_stats["calibration"],
            "roadmap_active_positive_fraction": roadmap_stats["active_positive_fraction"],
            "roadmap_active_negative_fraction": roadmap_stats["active_negative_fraction"],
            "roadmap_query_std": roadmap_stats["query_std"],
            "roadmap_anchor_ess_ratio": roadmap_stats["anchor_ess_ratio"],
            "roadmap_valid_query_fraction": roadmap_stats["valid_query_fraction"],
            "itc": itc,
            "semantic": sem,
            "nad": nad,
            "dhm": dhm,
            "rank": rank,
        }
        total = sum(float(c.get(f"lambda_{name}", 0.0)) * value for name, value in values.items())
        return total, values, stats


def batch_all_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
    p: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Public SD-MIR batch-all mining, averaged over active valid triplets."""
    distances = torch.cdist(embeddings.float(), embeddings.float(), p=p)
    raw = distances.unsqueeze(2) - distances.unsqueeze(1) + margin
    n = len(labels)
    unequal = ~torch.eye(n, dtype=torch.bool, device=labels.device)
    distinct = unequal.unsqueeze(2) & unequal.unsqueeze(1) & unequal.unsqueeze(0)
    same = labels[:, None].eq(labels[None, :])
    valid = distinct & same.unsqueeze(2) & (~same).unsqueeze(1)
    active = F.relu(raw) * valid.float()
    positive = active > 1e-16
    num_positive = positive.sum()
    fraction = num_positive.float() / valid.sum().clamp_min(1).float()
    loss = active.sum() / num_positive.clamp_min(1).float()
    return loss.to(embeddings.dtype), fraction
