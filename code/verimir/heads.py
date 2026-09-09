from __future__ import annotations
from typing import Dict, Optional, Tuple, Sequence, Iterable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1, residual_init: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        if residual_init and input_dim == output_dim:
            self.residual = nn.Identity()
        elif residual_init:
            self.residual = nn.Linear(input_dim, output_dim, bias=False)
            nn.init.orthogonal_(self.residual.weight)
        else:
            self.residual = None
        if self.residual is not None:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.net(x)
        if self.residual is not None:
            projected = self.residual(x) + projected
        return F.normalize(projected, dim=-1)


class ImageOnlyVeriMIR(nn.Module):
    def __init__(self, backbone: nn.Module, appearance_head: nn.Module, semantic_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.appearance_head = appearance_head
        self.semantic_head = semantic_head

    def forward(self, images: torch.Tensor):
        h = self.backbone(images)
        return self.appearance_head(h), self.semantic_head(h)

    def export_image_only(self):
        assert not hasattr(self, "text_encoder")
        return self.eval()
