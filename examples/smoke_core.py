"""Exercise ISV+BDB forward/backward on CPU without downloading any data."""
from pathlib import Path
import json
import torch
from torch import nn
from torch.nn import functional as F
import yaml
from verimir.model import VeriMIR
from verimir.losses import caption_class_prototypes, build_bidirectional_top5_teacher
from verimir.retrieval import fit_train_zscore
from verimir.training import training_objective


class SyntheticBackbone(nn.Module):
    output_dim = 512
    def __init__(self):
        super().__init__()
        self.visual_projection = nn.Linear(32, 512, bias=False)
    def forward(self, x):
        return F.normalize(self.visual_projection(x), dim=-1)


def run():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/isic2018_b32.yaml').read_text())
    model = VeriMIR(backbone=SyntheticBackbone(), num_classes=7)
    images = torch.randn(14,32)
    labels, ids = torch.arange(7).repeat_interleave(2), torch.arange(14)
    captions = F.normalize(torch.randn(14,512), dim=-1)
    prototypes = caption_class_prototypes(captions, labels)
    model.eval()
    with torch.no_grad():
        a,s = model(images)
    calibration = fit_train_zscore(a,s,num_pairs=200,seed=42)
    bank = {'appearance':a,'semantic':s,'labels':labels,'sample_ids':ids,
            'calibration':calibration,'snapshot_version':0,'calibration_version':0}
    teacher = build_bidirectional_top5_teacher(bank,beta=0.0,appearance_weight=0.5)
    batch = {'split':'train','image':images,'augmented_image':images+0.4*torch.randn_like(images),
             'label':labels,'sample_id':ids,'bank_index':ids,'caption_embedding':captions}
    model.train()
    loss, details = training_objective(model,batch,cfg,1,prototypes,bank,teacher)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.backbone.visual_projection.weight.grad is not None
    assert torch.isfinite(model.backbone.visual_projection.weight.grad).all()
    return {'synthetic_only':True,'finite_loss':True,'backward_pass':True,
            'bdb_executed':bool(details['bdb']),'no_dataset_or_checkpoint_loaded':True}


if __name__ == '__main__':
    print(json.dumps(run(),indent=2))
