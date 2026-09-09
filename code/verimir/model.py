"""Compact B/32 adapter for the extracted projection heads.

The image-only parameter names match the reference B/32 exported architecture.
This adapter omits inactive experimental backbone options and the text tower.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .heads import ImageOnlyVeriMIR, ProjectionHead

B32_MODEL_ID = "openai/clip-vit-base-patch32"


class B32VisionBackbone(nn.Module):
    def __init__(self, model_path=B32_MODEL_ID, local_files_only=True):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection
        clip = CLIPVisionModelWithProjection.from_pretrained(
            model_path, local_files_only=local_files_only,
        )
        cfg = clip.config
        if cfg.patch_size != 32 or cfg.hidden_size != 768 or cfg.num_hidden_layers != 12:
            raise ValueError("This release supports CLIP ViT-B/32 only")
        self.vision_model = clip.vision_model
        self.visual_projection = clip.visual_projection
        self.output_dim = clip.config.projection_dim

    def forward(self, images):
        pooled = self.vision_model(pixel_values=images).pooler_output
        return F.normalize(self.visual_projection(pooled), dim=-1)


class VeriMIR(nn.Module):
    def __init__(self, backbone=None, num_classes=7, dropout=0.1,
                 model_path=B32_MODEL_ID, local_files_only=True):
        super().__init__()
        self.backbone = backbone if backbone is not None else B32VisionBackbone(model_path, local_files_only)
        self.appearance_head = ProjectionHead(self.backbone.output_dim, 512, dropout, True)
        self.semantic_head = ProjectionHead(self.backbone.output_dim, 512, dropout, True)
        self.appearance_classifier = nn.Linear(512, num_classes, bias=False)
        nn.init.normal_(self.appearance_classifier.weight, std=0.01)

    def forward_train(self, images):
        h = self.backbone(images)
        appearance = self.appearance_head(h)
        semantic = self.semantic_head(h)
        logits = F.linear(appearance, F.normalize(self.appearance_classifier.weight, dim=-1))
        return {"appearance":appearance, "semantic":semantic, "appearance_logits":logits}

    def forward(self, images):
        h = self.backbone(images)
        return self.appearance_head(h), self.semantic_head(h)

    def export_image_only(self):
        """Return independent image modules without changing the source's mode.

        Deep-copy before eval(): neither parameters/buffers nor train/eval
        changes may propagate between the training model and its export.
        """
        return copy.deepcopy(ImageOnlyVeriMIR(
            self.backbone, self.appearance_head, self.semantic_head,
        )).eval()
