from __future__ import annotations
from typing import Dict, Optional, Tuple, Sequence, Iterable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def retrieval_interface_parameters(model: torch.nn.Module) -> tuple[torch.nn.Parameter, ...]:
    """Return the existing deployable projection-interface parameters.

    RICF deliberately excludes the large vision-transformer body and the
    training-only appearance classifier.  The CLIP visual projection and the
    two image retrieval heads are all exported into the image-only artifact.
    """

    modules: list[torch.nn.Module] = []
    visual_projection = getattr(getattr(model, "backbone", None), "visual_projection", None)
    if visual_projection is None:
        raise ValueError("RICF retrieval_interface requires backbone.visual_projection")
    modules.append(visual_projection)
    for name in ("appearance_head", "semantic_head"):
        module = getattr(model, name, None)
        if module is None:
            raise ValueError(f"RICF retrieval_interface requires model.{name}")
        modules.append(module)

    parameters: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
    if not parameters:
        raise ValueError("RICF retrieval interface has no trainable parameters")
    return tuple(parameters)


def _gradient_cosine(
    clean_loss: torch.Tensor,
    augmented_loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if eps <= 0.0:
        raise ValueError("RICF eps must be positive")
    if clean_loss.ndim != 0 or augmented_loss.ndim != 0:
        raise ValueError("RICF losses must be scalar")
    if not parameters:
        raise ValueError("RICF requires at least one probe parameter")

    clean_gradients = torch.autograd.grad(
        clean_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    augmented_gradients = torch.autograd.grad(
        augmented_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    device = clean_loss.device
    dot = torch.zeros((), dtype=torch.float32, device=device)
    clean_sq = torch.zeros_like(dot)
    augmented_sq = torch.zeros_like(dot)
    for clean_gradient, augmented_gradient in zip(clean_gradients, augmented_gradients):
        if clean_gradient is not None:
            clean = clean_gradient.detach().float()
            clean_sq = clean_sq + clean.square().sum()
        else:
            clean = None
        if augmented_gradient is not None:
            augmented = augmented_gradient.detach().float()
            augmented_sq = augmented_sq + augmented.square().sum()
        else:
            augmented = None
        if clean is not None and augmented is not None:
            dot = dot + (clean * augmented).sum()

    finite = torch.isfinite(dot) & torch.isfinite(clean_sq) & torch.isfinite(augmented_sq)
    if not bool(finite):
        raise FloatingPointError("non-finite RICF retrieval-interface gradient statistics")

    clean_norm = clean_sq.sqrt()
    augmented_norm = augmented_sq.sqrt()
    zero_norm = (clean_norm <= eps) | (augmented_norm <= eps)
    if bool(zero_norm):
        cosine = torch.zeros_like(dot)
    else:
        cosine = (dot / (clean_norm * augmented_norm + eps)).clamp(-1.0, 1.0)
    return cosine.detach(), zero_norm.detach(), clean_norm.detach(), augmented_norm.detach()


def retrieval_interface_conflict_objective(
    clean_loss: torch.Tensor,
    augmented_loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    *,
    policy: str = "adaptive",
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
    """Filter only negative conflict from an already-verified augmented loss.

    ``observe_only`` executes the identical gradient probe but forces alpha to
    one.  It is therefore the causally matched V21-compatible control.
    """

    policy = str(policy).lower()
    if policy not in {"adaptive", "observe_only"}:
        raise ValueError("RICF policy must be adaptive or observe_only")
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    cosine, zero_norm, clean_norm, augmented_norm = _gradient_cosine(
        clean_loss,
        augmented_loss,
        parameters,
        float(eps),
    )
    if policy == "observe_only" or bool(zero_norm):
        alpha = torch.ones_like(cosine)
    else:
        alpha = 1.0 + torch.minimum(cosine, torch.zeros_like(cosine))
    alpha = alpha.detach().clamp(0.0, 1.0)
    objective = 0.5 * clean_loss + 0.5 * alpha.to(augmented_loss.dtype) * augmented_loss
    details: dict[str, torch.Tensor | str] = {
        "policy": policy,
        "cosine": cosine,
        "alpha": alpha,
        "negative_conflict": (cosine < 0.0).detach(),
        "zero_norm": zero_norm,
        "clean_gradient_norm": clean_norm,
        "augmented_gradient_norm": augmented_norm,
    }
    return objective, details
