from pathlib import Path
import copy
import torch
import yaml
from verimir.evaluation import image_scores, rank_without_self
from verimir.retrieval import strict_v2_metrics_from_scores
from verimir.training import build_criterion


def test_scoring_interface_has_no_labels_and_excludes_self():
    g = torch.Generator().manual_seed(12)
    a = torch.nn.functional.normalize(torch.randn(9,16,generator=g),dim=-1)
    calibration = {'appearance_mean':0.,'appearance_std':1.,'semantic_mean':0.,'semantic_std':1.}
    scores = image_scores(a,a,calibration)
    ranked = rank_without_self(scores)
    assert ranked.shape == (9,8)
    assert not ranked.eq(torch.arange(9)[:,None]).any()
    assert torch.equal(ranked,rank_without_self(image_scores(a,a,calibration)))


def test_strict_metrics_and_self_exclusion():
    scores = torch.tensor([[99.,1.,0.,0.],[1.,99.,0.,0.],[0.,0.,99.,1.],[0.,0.,1.,99.]])
    metrics = strict_v2_metrics_from_scores(scores,torch.tensor([0,0,1,1]),torch.arange(4))
    assert metrics['mAP'] == 100.
    assert metrics['R@1'] == metrics['P@1'] == 100.
    assert abs(metrics['P@5']-100./3)<1e-5


def test_configs_are_same_method_and_b32_only():
    root=Path(__file__).resolve().parents[1]/'configs'
    a=yaml.safe_load((root/'isic2018_b32.yaml').read_text())
    b=yaml.safe_load((root/'kvasir_b32.yaml').read_text())
    for cfg in (a,b):
        assert cfg['model']['name']=='openai/clip-vit-base-patch32'
        assert cfg['training']['epochs']==20
        assert cfg['loss']['sv_bt5sd_include_incoming'] is True
        assert cfg['loss']['use_caption_reliability'] is False
    for cfg in (a,b):
        for key in ('seed','run_dir','data'):
            del cfg[key]
        del cfg['model']['num_classes']
        del cfg['training']['classes_per_batch']
    assert a==b


def test_epoch_classification_schedule_and_caption_scaling():
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/isic2018_b32.yaml').read_text())
    before=copy.deepcopy(cfg)
    assert build_criterion(cfg,1).cfg['lambda_appearance_classification']==0.2
    assert build_criterion(cfg,5).cfg['lambda_appearance_classification']==0.
    assert build_criterion(cfg,1).cfg['lambda_itc'] < cfg['loss']['lambda_itc']
    assert cfg==before


def test_combined_synthetic_training_step():
    from examples.smoke_core import run
    assert run()['backward_pass']


def test_wrong_split_is_rejected():
    import pytest
    from verimir.training import training_objective
    with pytest.raises(ValueError,match='train'):
        training_objective(None,{'split':'test'},{},1,None)


def test_b32_adapter_with_mocked_pretrained_tower(monkeypatch):
    from types import SimpleNamespace
    import transformers
    from verimir.model import B32VisionBackbone
    class FakeVision(torch.nn.Module):
        def forward(self,pixel_values):
            return SimpleNamespace(pooler_output=pixel_values)
    fake=SimpleNamespace(config=SimpleNamespace(patch_size=32,hidden_size=768,num_hidden_layers=12,projection_dim=512),
                         vision_model=FakeVision(),visual_projection=torch.nn.Linear(768,512,bias=False))
    monkeypatch.setattr(transformers.CLIPVisionModelWithProjection,'from_pretrained',lambda *a,**k:fake)
    backbone=B32VisionBackbone()
    assert backbone(torch.randn(2,768)).shape==(2,512)


def test_export_image_only_is_independent_and_preserves_source_modes():
    from examples.smoke_core import SyntheticBackbone
    from verimir.model import VeriMIR
    backbone=SyntheticBackbone()
    backbone.register_buffer('export_probe',torch.tensor([1.]))
    model=VeriMIR(backbone=backbone,num_classes=7)
    model.train()
    model.semantic_head.eval()  # Preserve mixed module modes too.
    modes={name:module.training for name,module in model.named_modules()}
    exported=model.export_image_only()
    assert modes=={name:module.training for name,module in model.named_modules()}
    assert not any(module.training for module in exported.modules())
    assert not hasattr(exported,'appearance_classifier')
    original=model.state_dict()
    for name,value in exported.state_dict().items():
        assert torch.equal(value,original[name])
        assert value.data_ptr()!=original[name].data_ptr()
    snapshot={name:value.clone() for name,value in exported.state_dict().items()}
    with torch.no_grad():
        model.backbone.visual_projection.weight.add_(1.)
        model.backbone.export_probe.add_(1.)
    model.train()
    assert not any(module.training for module in exported.modules())
    assert all(torch.equal(v,snapshot[k]) for k,v in exported.state_dict().items())
    with torch.no_grad(): exported.backbone.export_probe.add_(8.)
    assert model.backbone.export_probe.item()==2.
    exported.train()
    model.eval()
    assert exported.training and exported.backbone.training
