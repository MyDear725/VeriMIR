from __future__ import annotations
from typing import Dict, Optional, Tuple, Sequence, Iterable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import json
import numpy as np


class MemoryBank:
    def __init__(
        self,
        size: int,
        dim: int,
        momentum: float = 0.9,
        semantic_dim: Optional[int] = None,
    ):
        semantic_dim = int(dim if semantic_dim is None else semantic_dim)
        self.appearance = torch.zeros(size, dim)
        self.semantic = torch.zeros(size, semantic_dim)
        self.text = torch.zeros(size, semantic_dim)
        self.labels = torch.full((size,), -1, dtype=torch.long)
        self.sample_ids = torch.full((size,), -1, dtype=torch.long)
        self.initialized = torch.zeros(size, dtype=torch.bool)
        self.momentum = float(momentum)

    def update(self, indices, appearance, semantic, text, labels, sample_ids):
        indices = indices.cpu().long()
        first = ~self.initialized[indices]
        old = ~first
        if first.any():
            idx = indices[first]
            self.appearance[idx] = appearance[first].detach().cpu()
            self.semantic[idx] = semantic[first].detach().cpu()
        if old.any():
            idx = indices[old]
            self.appearance[idx] = F.normalize(self.momentum * self.appearance[idx] + (1 - self.momentum) * appearance[old].detach().cpu(), dim=-1)
            self.semantic[idx] = F.normalize(self.momentum * self.semantic[idx] + (1 - self.momentum) * semantic[old].detach().cpu(), dim=-1)
        self.text[indices] = F.normalize(text.detach().cpu(), dim=-1)
        self.labels[indices] = labels.detach().cpu()
        self.sample_ids[indices] = sample_ids.detach().cpu()
        self.initialized[indices] = True

    def as_triplet_bank(self, weights: torch.Tensor) -> dict:
        valid = self.initialized
        return {
            "embeddings": self.appearance[valid],
            "labels": self.labels[valid],
            "sample_ids": self.sample_ids[valid],
            "weights": weights[valid],
        }

    def class_prototypes(self, branch: str = "appearance") -> torch.Tensor:
        """Normalized class means from a fully refreshed train-only bank."""
        if branch not in {"appearance", "semantic"}:
            raise ValueError("memory-bank prototype branch must be appearance or semantic")
        if not bool(self.initialized.all()):
            raise ValueError("memory-bank class prototypes require a full train refresh")
        labels = self.labels
        classes = torch.unique(labels, sorted=True)
        expected = torch.arange(int(labels.max()) + 1, dtype=torch.long)
        if not torch.equal(classes, expected):
            raise ValueError("memory-bank labels must be contiguous and cover every class")
        features = getattr(self, branch)
        return torch.stack([
            F.normalize(features[labels.eq(class_index)].float().mean(dim=0), dim=0)
            for class_index in expected
        ])


def _sha256_tensor(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class RankSnapshotBank:
    """Versioned, zero-momentum train snapshot for fused early-recall loss.

    Unlike :class:`MemoryBank`, this object never mixes descriptor versions.
    Descriptors and their train-only calibration are replaced atomically and
    must carry the same version before a loss view can be requested.
    """

    def __init__(self, size: int, appearance_dim: int, semantic_dim: int):
        if int(size) < 1:
            raise ValueError("rank snapshot bank size must be positive")
        self.size = int(size)
        self.appearance = torch.zeros(self.size, int(appearance_dim))
        self.semantic = torch.zeros(self.size, int(semantic_dim))
        self.labels = torch.full((self.size,), -1, dtype=torch.long)
        self.sample_ids = torch.full((self.size,), -1, dtype=torch.long)
        self.initialized = torch.zeros(self.size, dtype=torch.bool)
        self.version = -1
        self.calibration_version = -1
        self.calibration = {
            "appearance_mean": 0.0,
            "appearance_std": 1.0,
            "semantic_mean": 0.0,
            "semantic_std": 1.0,
        }
        self.metadata: dict = {}

    def replace(
        self,
        indices: torch.Tensor,
        appearance: torch.Tensor,
        semantic: torch.Tensor,
        labels: torch.Tensor,
        sample_ids: torch.Tensor,
        calibration: Optional[dict],
        version: int,
    ) -> dict:
        indices = indices.detach().cpu().long()
        appearance = F.normalize(appearance.detach().cpu().float(), dim=-1)
        semantic = F.normalize(semantic.detach().cpu().float(), dim=-1)
        labels = labels.detach().cpu().long()
        sample_ids = sample_ids.detach().cpu().long()
        version = int(version)
        if version < 0:
            raise ValueError("rank snapshot version must be non-negative")
        if len(indices) != self.size:
            raise ValueError("rank snapshot must contain every training sample exactly once")
        expected = torch.arange(self.size, dtype=torch.long)
        if not torch.equal(torch.sort(indices).values, expected):
            raise ValueError("rank snapshot bank indices must be a permutation of the train set")
        if len(torch.unique(sample_ids)) != self.size:
            raise ValueError("rank snapshot sample IDs must be unique")
        if appearance.shape != self.appearance.shape:
            raise ValueError("rank snapshot appearance descriptors have incompatible shape")
        if semantic.shape != self.semantic.shape:
            raise ValueError("rank snapshot semantic descriptors have incompatible shape")
        if labels.shape != (self.size,) or sample_ids.shape != (self.size,):
            raise ValueError("rank snapshot metadata has incompatible shape")
        if not torch.isfinite(appearance).all() or not torch.isfinite(semantic).all():
            raise ValueError("rank snapshot descriptors must be finite")

        ordered_appearance = torch.empty_like(self.appearance)
        ordered_semantic = torch.empty_like(self.semantic)
        ordered_labels = torch.empty_like(self.labels)
        ordered_sample_ids = torch.empty_like(self.sample_ids)
        ordered_appearance[indices] = appearance
        ordered_semantic[indices] = semantic
        ordered_labels[indices] = labels
        ordered_sample_ids[indices] = sample_ids

        if calibration is None:
            normalized_calibration = {
                "appearance_mean": 0.0,
                "appearance_std": 1.0,
                "semantic_mean": 0.0,
                "semantic_std": 1.0,
            }
        else:
            normalized_calibration = {
                key: float(calibration[key]) for key in (
                    "appearance_mean", "appearance_std",
                    "semantic_mean", "semantic_std",
                )
            }
        if (
            not all(np.isfinite(value) for value in normalized_calibration.values())
            or normalized_calibration["appearance_std"] <= 0.0
            or normalized_calibration["semantic_std"] <= 0.0
        ):
            raise ValueError("rank snapshot calibration must be finite with positive std")

        # Commit only after every validation succeeds, keeping replacement atomic.
        self.appearance.copy_(ordered_appearance)
        self.semantic.copy_(ordered_semantic)
        self.labels.copy_(ordered_labels)
        self.sample_ids.copy_(ordered_sample_ids)
        self.initialized.fill_(True)
        self.calibration = normalized_calibration
        self.version = version
        self.calibration_version = version
        calibration_sha256 = hashlib.sha256(
            json.dumps(normalized_calibration, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.metadata = {
            "snapshot_version": version,
            "calibration_version": version,
            "num_samples": self.size,
            "ordered_sample_ids_sha256": _sha256_tensor(self.sample_ids),
            "appearance_sha256": _sha256_tensor(self.appearance),
            "semantic_sha256": _sha256_tensor(self.semantic),
            "calibration_sha256": calibration_sha256,
            "calibration": dict(self.calibration),
        }
        return dict(self.metadata)

    def as_loss_bank(self, expected_version: Optional[int] = None) -> dict:
        if not bool(self.initialized.all()):
            raise ValueError("rank snapshot bank must be fully initialized")
        if self.version != self.calibration_version:
            raise ValueError("rank snapshot bank/calibration version mismatch")
        if expected_version is not None and self.version != int(expected_version):
            raise ValueError("rank snapshot version differs from the expected epoch version")
        return {
            "appearance": self.appearance,
            "semantic": self.semantic,
            "labels": self.labels,
            "sample_ids": self.sample_ids,
            "calibration": dict(self.calibration),
            "snapshot_version": self.version,
            "calibration_version": self.calibration_version,
        }
