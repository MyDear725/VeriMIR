"""Core loss assembly for the frozen B/32 method, not a full experiment runner.

The caller owns optimizer/AMP, epoch EMA, train-bank refresh, checkpoint
selection and data split provenance. See docs/TRAINING.md.
"""
import copy
import torch
from .losses import (
    CompositeLoss, caption_loss_class_scale, shrink_caption_embeddings,
    verified_augmentation_gate, verified_augmentation_allocation,
    self_verified_bidirectional_top5_support_loss,
)
from .gradient_agreement import retrieval_interface_parameters, retrieval_interface_conflict_objective
from .schedule import _sv_bt5sd_effective_weight


def build_criterion(config, epoch):
    if epoch < 1:
        raise ValueError("epoch is one-based")
    c = copy.deepcopy(config["loss"])
    if c.get("normalize_caption_loss_by_classes", False):
        scale = caption_loss_class_scale(config["model"]["num_classes"], c.get("caption_loss_reference_classes", 3))
        for key in ("lambda_itc", "lambda_semantic", "lambda_nad"):
            c[key] *= scale
    active = config["training"].get("classification_active_epochs", c.get("classification_active_epochs", 0))
    if active and epoch > active:
        c["lambda_appearance_classification"] = 0.0
    return CompositeLoss(c)


def training_objective(model, batch, config, epoch, caption_prototypes,
                       rank_bank=None, teacher=None, triplet_bank=None):
    """Assemble the full method for one training batch. No optimizer step.

    Both banks and caption_prototypes MUST be constructed only from train.
    Caption reliability is disabled; all base reliability weights are one.
    This assembly intentionally supports the full ISV+BDB reference settings.
    """
    if batch.get("split") != "train":
        raise ValueError("Only explicitly train-marked batches are accepted")
    if config["reliability"].get("enabled", False) or config["loss"].get("use_caption_reliability", False):
        raise ValueError("Caption self-verification is not part of the released method")
    if config["training"].get("verified_augmentation_mode") != "residual":
        raise ValueError("This integration supports the frozen residual ISV mode")
    labels, ids = batch["label"], batch["sample_id"]
    clean = model.forward_train(batch["image"])
    augmented = model.forward_train(batch["augmented_image"])
    ones = torch.ones(len(labels), device=labels.device)
    text = shrink_caption_embeddings(batch["caption_embedding"].float(), labels,
                                     caption_prototypes, config["loss"]["caption_residual_scale"])
    gate, gate_details = verified_augmentation_gate(
        clean["appearance"], clean["semantic"], augmented["appearance"], augmented["semantic"], ones,
    )
    allocation = verified_augmentation_allocation(gate, "residual")
    criterion = build_criterion(config, epoch)
    losses = []
    for view, weights in ((clean, ones), (augmented, allocation)):
        loss, _, _ = criterion(view["appearance"], view["semantic"], text,
                               labels, ids, weights, weights, triplet_bank,
                               caption_prototypes, view["appearance_logits"])
        losses.append(loss)
    loss, gradient_details = retrieval_interface_conflict_objective(
        losses[0], losses[1], retrieval_interface_parameters(model),
        policy=config["training"]["gradient_agreement"]["policy"],
        eps=config["training"]["gradient_agreement"]["eps"],
    )
    c = config["loss"]
    coefficient = _sv_bt5sd_effective_weight(c["lambda_sv_bt5sd"], epoch)
    bdb_details = {}
    if coefficient > 0:
        if rank_bank is None or teacher is None:
            raise ValueError("Active BDB pulse requires the frozen train-only snapshot and teacher")
        options = {key:c['sv_bt5sd_'+key] for key in (
            'margin','huber_delta','gate_floor','gate_temperature','disagreement_temperature',
            'include_incoming','incoming_weight','adaptive_incoming','incoming_weight_min',
            'incoming_weight_max','incoming_boundary_pivot','incoming_boundary_temperature',
        )}
        boundary, bdb_details = self_verified_bidirectional_top5_support_loss(
            clean["appearance"], clean["semantic"], augmented["appearance"], augmented["semantic"],
            labels, ids, batch["bank_index"], rank_bank, teacher,
            beta=config["fusion"]["beta"], appearance_weight=config["fusion"]["appearance_weight"], **options,
        )
        loss = loss + coefficient * boundary
    return loss, {"feature_check":gate_details, "gradient_check":gradient_details, "bdb":bdb_details}
