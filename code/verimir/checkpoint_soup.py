from __future__ import annotations
from typing import Dict, Optional, Tuple, Sequence, Iterable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from pathlib import Path


def _validate_ingredients(states: Sequence[dict]) -> None:
    if len(states) < 2:
        raise ValueError("checkpoint soup requires at least two ingredients")
    first = states[0]
    if first.get("checkpoint_type") != "image_only":
        raise ValueError("checkpoint soup requires image-only checkpoints")
    if first.get("selection_split") != "validation":
        raise ValueError("checkpoint soup ingredients must be validation-selected")
    if int(first.get("test_evaluation_count", 0)) != 0:
        raise ValueError("checkpoint soup ingredient has already accessed test")
    first_keys = list(first["model"])
    first_cfg = first["config"]
    for state in states[1:]:
        if state.get("checkpoint_type") != "image_only":
            raise ValueError("checkpoint soup requires image-only checkpoints")
        if state.get("selection_split") != "validation":
            raise ValueError("checkpoint soup ingredients must be validation-selected")
        if int(state.get("test_evaluation_count", 0)) != 0:
            raise ValueError("checkpoint soup ingredient has already accessed test")
        if list(state["model"]) != first_keys:
            raise ValueError("checkpoint soup model keys do not match")
        cfg = state["config"]
        if cfg.get("method") != first_cfg.get("method"):
            raise ValueError("checkpoint soup methods do not match")
        if cfg.get("model") != first_cfg.get("model"):
            raise ValueError("checkpoint soup model configs do not match")
        if cfg.get("data", {}).get("manifest") != first_cfg.get("data", {}).get("manifest"):
            raise ValueError("checkpoint soup manifests do not match")


def average_image_checkpoints(
    states: Sequence[dict],
    weights: Sequence[float] | None = None,
    ingredient_paths: Sequence[str | Path] | None = None,
) -> dict:
    """Average compatible validation-selected image checkpoints into one state."""
    _validate_ingredients(states)
    if weights is None:
        weights = [1.0 / len(states)] * len(states)
    if len(weights) != len(states):
        raise ValueError("checkpoint soup weights and ingredients differ in length")
    weights = [float(value) for value in weights]
    if any(not torch.isfinite(torch.tensor(value)) or value < 0.0 for value in weights):
        raise ValueError("checkpoint soup weights must be finite and nonnegative")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("checkpoint soup weights must have positive sum")
    weights = [value / total for value in weights]

    output = copy.deepcopy(states[0])
    averaged = {}
    for key in states[0]["model"]:
        tensors = [state["model"][key] for state in states]
        first = tensors[0]
        if any(tensor.shape != first.shape or tensor.dtype != first.dtype for tensor in tensors[1:]):
            raise ValueError(f"checkpoint soup tensor mismatch: {key}")
        if first.is_floating_point() or first.is_complex():
            value = torch.zeros_like(first, dtype=torch.float64)
            for weight, tensor in zip(weights, tensors):
                value.add_(tensor.to(torch.float64), alpha=weight)
            averaged[key] = value.to(first.dtype)
        else:
            if any(not torch.equal(first, tensor) for tensor in tensors[1:]):
                raise ValueError(f"checkpoint soup non-floating buffer mismatch: {key}")
            averaged[key] = first.detach().clone()
    output["model"] = averaged
    output["calibration"] = None
    output["epoch"] = None
    output["best_epoch"] = None
    output["best_map"] = None
    output["selection"] = {
        "split": "validation",
        "protocol": "strict_v2",
        "metric": "mAP",
        "value": None,
        "source": "uniform_seed_weight_soup",
    }
    output["selection_split"] = "validation"
    output["test_evaluation_count"] = 0
    output["checkpoint_soup"] = {
        "type": "uniform_seed_weight_soup",
        "weights": weights,
        "ingredients": [str(path) for path in (ingredient_paths or [])],
        "calibration": "pending_train_image_only_recalibration",
    }
    return output
