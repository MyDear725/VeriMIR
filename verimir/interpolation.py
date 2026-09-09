from __future__ import annotations
from pathlib import Path
from typing import Sequence
from .checkpoint_soup import average_image_checkpoints


def build_boundary_anchored_soup(
    control_state: dict,
    self_verified_state: dict,
    candidate_weight: float,
    ingredient_paths: Sequence[str | Path],
) -> dict:
    """Interpolate a self-verified checkpoint toward its matched control anchor.

    Both ingredients must already be image-only, validation-selected and test-clean.
    The returned object is still a single deployable image encoder checkpoint.
    """
    candidate_weight = float(candidate_weight)
    if not 0.0 < candidate_weight < 1.0:
        raise ValueError("candidate_weight must be strictly between zero and one")
    if len(ingredient_paths) != 2:
        raise ValueError("V143 requires exactly one control and one self-verified ingredient")
    soup = average_image_checkpoints(
        [control_state, self_verified_state],
        [1.0 - candidate_weight, candidate_weight],
        ingredient_paths,
    )
    soup["checkpoint_soup"].update({
        "type": "self_verified_boundary_anchored_weight_interpolation",
        "ingredient_roles": ["matched_control", "self_verified_boundary_candidate"],
        "candidate_weight": candidate_weight,
        "image_only_inference": True,
        "single_checkpoint_inference": True,
    })
    return soup
