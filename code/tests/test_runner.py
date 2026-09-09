"""Tiny CPU lifecycle tests; no medical images, CLIP downloads or paper metrics."""
import json
import copy
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import Dataset, DataLoader

from examples.smoke_core import SyntheticBackbone
from verimir.data import PKBatchSampler
from verimir.model import VeriMIR
from verimir.runner import run_training, validate_manifest
from scripts.evaluate import evaluate_checkpoint
from verimir.finalize import finalize_checkpoints


class SyntheticDataset(Dataset):
    def __init__(self,rows,mode,text=None):
        self.records=rows
        self.mode=mode
        self.labels=[r['label'] for r in rows]
        self.images=torch.randn(len(rows),32,generator=torch.Generator().manual_seed(11))
        self.text=text
    def __len__(self): return len(self.records)
    def __getitem__(self,index):
        row=self.records[index]
        item={'image':self.images[index], 'label':torch.tensor(row['label']),
              'sample_id':torch.tensor(row['sample_id'])}
        if self.mode in ('train','clean'):
            item.update(bank_index=torch.tensor(index),caption_embedding=self.text[index])
        if self.mode=='train': item['augmented_image']=item['image']+.1*torch.randn(32)
        return item


@pytest.fixture
def completed_run(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(3)
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/isic2018_b32.yaml').read_text())
    manifest={'root':str(tmp_path),'splits':{}}
    for n,split in enumerate(('train','validation','test')):
        manifest['splits'][split]=[{'sample_id':n*100+i,'source_id':f'{split}/{i}',
                                    'relative_path':f'{split}/{i}.jpg','label':i//2} for i in range(14)]
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest))
    text=torch.nn.functional.normalize(torch.randn(14,512),dim=-1)
    cache=tmp_path/'train.pt'
    torch.save({'sample_ids':torch.arange(14),'embeddings':text},cache)
    cfg['data'].update(manifest=str(path),caption_cache=str(cache))
    cfg['run_dir']=str(tmp_path/'run')
    cfg['training'].update(device='cpu',epochs=3,workers=0,checkpoint_last_interval=1)
    cfg['fusion']['calibration_pairs']=32
    ds=SyntheticDataset(manifest['splits']['train'],'train',text)
    clean=SyntheticDataset(manifest['splits']['train'],'clean',text)
    val=SyntheticDataset(manifest['splits']['validation'],'validation')
    loaders=(DataLoader(ds,batch_sampler=PKBatchSampler(ds.labels,7,2,42)),
             DataLoader(clean,batch_size=14),DataLoader(val,batch_size=14))
    model=VeriMIR(backbone=SyntheticBackbone(),num_classes=7)
    before=model.backbone.visual_projection.weight.detach().clone()
    status=run_training(cfg,model=model,loaders=loaders)
    assert not torch.equal(before,model.backbone.visual_projection.weight)
    return cfg,manifest,model,status,loaders


def test_three_epoch_lifecycle(completed_run):
    cfg,manifest,model,status,loaders=completed_run
    out=Path(cfg['run_dir'])
    assert status['status']=='complete' and status['completed_epochs']==3
    assert status['test_evaluation_count']==0 and not status['test_accessed']
    assert not (out/'test_attempt.json').exists()
    history=json.loads((out/'validation_history.json').read_text())
    assert len(history)==3 and all(h['steps']==1 and h['ema_strict_v2'] for h in history)
    state=torch.load(out/'checkpoint_best_image_only.pt',weights_only=False)
    assert state['selection_split']=='validation' and state['checkpoint_type']=='image_only'
    assert not any('classifier' in k or 'text' in k for k in state['model'])
    with pytest.raises(FileExistsError):
        run_training(cfg,model=model,loaders=loaders)
    result=evaluate_checkpoint(out/'checkpoint_best_image_only.pt',model=model.export_image_only(),loader=loaders[2])
    assert result['strict_v2']['num_queries']==14
    assert result['strict_v2']['P@1']==result['strict_v2']['R@1']


def test_synthetic_test_requires_confirmation_and_only_one_attempt(completed_run):
    cfg,manifest,model,status,_=completed_run
    path=Path(cfg['run_dir'])/'checkpoint_best_image_only.pt'
    with pytest.raises(ValueError,match='confirm-test'):
        evaluate_checkpoint(path,split='test')
    loader=DataLoader(SyntheticDataset(manifest['splits']['test'],'test'),batch_size=14)
    result=evaluate_checkpoint(path,split='test',confirm_test=True,model=model.export_image_only(),loader=loader)
    assert result['test_evaluation_count']==1
    assert result['synthetic_injection'] is True
    with pytest.raises(FileExistsError):
        evaluate_checkpoint(path,split='test',confirm_test=True,model=model.export_image_only(),loader=loader)


def test_manifest_overlap_rejected(tmp_path):
    row={'sample_id':1,'source_id':'same','relative_path':'same.jpg','label':0}
    path=tmp_path/'overlap.json'
    path.write_text(json.dumps({'splits':{k:[row] for k in ('train','validation','test')}}))
    with pytest.raises(ValueError,match='overlap'): validate_manifest(path)


def test_real_loaders_with_generated_images_never_open_test(tmp_path):
    from PIL import Image
    import numpy as np
    from torch import nn
    class TinyImageBackbone(nn.Module):
        output_dim=512
        def __init__(self):
            super().__init__()
            self.visual_projection=nn.Linear(3,512,bias=False)
        def forward(self,x):
            return torch.nn.functional.normalize(self.visual_projection(x.mean(dim=(-1,-2))),dim=-1)
    torch.set_num_threads(2)
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/isic2018_b32.yaml').read_text())
    manifest={'root':str(tmp_path),'splits':{}}
    random=np.random.default_rng(4)
    for n,split in enumerate(('train','validation','test')):
        rows=[]
        for i in range(14):
            name=f'{split}_{i}.png'
            rows.append({'sample_id':n*100+i,'source_id':name,'relative_path':name,'label':i//2})
            if split!='test':
                Image.fromarray(random.integers(0,256,(32,32,3),dtype=np.uint8)).save(tmp_path/name)
        manifest['splits'][split]=rows
    path=tmp_path/'manifest.json'; path.write_text(json.dumps(manifest))
    cache=tmp_path/'captions.pt'
    torch.save({'sample_ids':torch.arange(14),'embeddings':torch.nn.functional.normalize(torch.randn(14,512),dim=-1)},cache)
    cfg['data'].update(manifest=str(path),caption_cache=str(cache),image_size=32)
    cfg['training'].update(device='cpu',workers=0,epochs=1)
    cfg['fusion']['calibration_pairs']=32
    cfg['run_dir']=str(tmp_path/'real_loaders')
    status=run_training(cfg,model=VeriMIR(backbone=TinyImageBackbone(),num_classes=7))
    assert status['status']=='complete' and status['test_accessed'] is False
    assert not any(tmp_path.glob('test_*.png'))
    control=copy.deepcopy(cfg)
    control['loss']['lambda_sv_bt5sd']=0.0
    control['run_dir']=str(tmp_path/'real_control')
    run_training(control,model=VeriMIR(backbone=TinyImageBackbone(),num_classes=7))
    cache.rename(tmp_path/'unavailable_caption_cache.pt')
    final=finalize_checkpoints(Path(control['run_dir'])/'checkpoint_best_image_only.pt',
                    Path(cfg['run_dir'])/'checkpoint_best_image_only.pt',tmp_path/'real_final',
                    model=VeriMIR(backbone=TinyImageBackbone(),num_classes=7).export_image_only())
    assert final['status']=='complete' and final['test_evaluation_count']==0
    assert not any(tmp_path.glob('test_*.png'))


@pytest.fixture
def completed_pair(completed_run):
    cfg,manifest,model,_,loaders=completed_run
    control=copy.deepcopy(cfg)
    control['run_dir']=str(Path(cfg['run_dir']).parent/'control')
    control['loss']['lambda_sv_bt5sd']=0.0
    run_training(control,model=VeriMIR(backbone=SyntheticBackbone(),num_classes=7),loaders=loaders)
    paths=(Path(control['run_dir'])/'checkpoint_best_image_only.pt',
           Path(cfg['run_dir'])/'checkpoint_best_image_only.pt')
    image_loaders=(DataLoader(SyntheticDataset(manifest['splits']['train'],'calibration'),batch_size=14),loaders[2])
    return cfg,manifest,model,paths,image_loaders


def test_final_interpolation_recalibration_and_locked_test(completed_pair):
    cfg,manifest,model,paths,loaders=completed_pair
    # Finalization must not load even the training caption cache.
    Path(cfg['data']['caption_cache']).rename(Path(cfg['data']['caption_cache']).with_suffix('.unused'))
    out=Path(cfg['run_dir']).parent/'final'
    result=finalize_checkpoints(*paths,out,model=model.export_image_only(),loaders=loaders)
    assert result['status']=='complete' and result['test_evaluation_count']==0
    checkpoint=out/'checkpoint_final_image_only.pt'
    state=torch.load(checkpoint,weights_only=False)
    control,method=[torch.load(p,weights_only=False) for p in paths]
    assert state['config']['loss']['lambda_sv_bt5sd']==0.025
    assert state['checkpoint_soup']['weights']==[0.25,0.75]
    assert state['checkpoint_soup']['calibration']=='refitted_on_train_images'
    assert state['calibration'] is not None and state['synthetic_injection']
    for key,value in state['model'].items():
        expected=(control['model'][key].double()*.25+method['model'][key].double()*.75).to(value.dtype)
        torch.testing.assert_close(value,expected,rtol=0,atol=0)
    from verimir.runner import extract_features,calibration_for
    inference=model.export_image_only(); inference.load_state_dict(state['model'])
    assert state['calibration']==calibration_for(extract_features(inference,loaders[0],'cpu'),cfg)
    assert state['selection']['source']=='fixed_0.25_control_0.75_method'
    assert not (out/'test_attempt.json').exists()
    with pytest.raises(FileExistsError):
        finalize_checkpoints(*paths,out,model=model.export_image_only(),loaders=loaders)
    test_loader=DataLoader(SyntheticDataset(manifest['splits']['test'],'test'),batch_size=14)
    tested=evaluate_checkpoint(checkpoint,split='test',confirm_test=True,model=model.export_image_only(),loader=test_loader)
    assert tested['test_evaluation_count']==1 and tested['synthetic_injection']
    with pytest.raises(FileExistsError):
        evaluate_checkpoint(checkpoint,split='test',confirm_test=True,model=model.export_image_only(),loader=test_loader)


@pytest.mark.parametrize('problem',['used_test','unfinished','seed','cache_hash','roles'])
def test_finalizer_rejects_invalid_ingredients(completed_pair,problem):
    cfg,_,model,paths,loaders=completed_pair
    control_path,method_path=paths
    if problem in ('used_test','unfinished'):
        status_path=control_path.parent/'run_status.json'
        status=json.loads(status_path.read_text())
        if problem=='used_test': status['test_evaluation_count']=1
        else: status['completed_epochs']=0
        status_path.write_text(json.dumps(status))
    elif problem in ('seed','cache_hash'):
        state=torch.load(method_path,weights_only=False)
        if problem=='seed': state['config']['seed']+=1
        else: state['caption_cache_sha256']='different'
        torch.save(state,method_path)
    else:
        paths=paths[::-1]
    out=Path(cfg['run_dir']).parent/'invalid_final'
    with pytest.raises(ValueError):
        finalize_checkpoints(*paths,out,model=model.export_image_only(),loaders=loaders)
    assert not out.exists()


def test_finalizer_rejects_caption_fields_and_changed_lock(completed_pair):
    cfg,_,model,paths,loaders=completed_pair
    out=Path(cfg['run_dir']).parent/'caption_rejected'
    bad=DataLoader(SyntheticDataset(loaders[0].dataset.records,'clean',torch.randn(14,512)),batch_size=14)
    with pytest.raises(ValueError,match='Text fields'):
        finalize_checkpoints(*paths,out,model=model.export_image_only(),loaders=(bad,loaders[1]))
    assert json.loads((out/'run_status.json').read_text())['status']=='failed'
    good=Path(cfg['run_dir']).parent/'locked_final'
    finalize_checkpoints(*paths,good,model=model.export_image_only(),loaders=loaders)
    lock_path=good/'finalization.json'
    lock=json.loads(lock_path.read_text()); lock['checkpoint_sha256']='changed'
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError,match='validation-locked'):
        evaluate_checkpoint(good/'checkpoint_final_image_only.pt',split='test',confirm_test=True,
                            model=model.export_image_only(),loader=loaders[1])
    assert not (good/'test_attempt.json').exists()


def test_full_pipeline_orchestration_on_synthetic_data(completed_run,monkeypatch):
    from scripts import train_final as pipeline
    cfg,manifest,_,_,loaders=completed_run
    roles=[]
    def synthetic_train(config):
        roles.append(config['loss']['lambda_sv_bt5sd'])
        return run_training(config,model=VeriMIR(backbone=SyntheticBackbone(),num_classes=7),loaders=loaders)
    def synthetic_finalize(control,method,output_dir,**kwargs):
        image_loaders=(DataLoader(SyntheticDataset(manifest['splits']['train'],'calibration'),batch_size=14),loaders[2])
        return finalize_checkpoints(control,method,output_dir,**kwargs,
                    model=VeriMIR(backbone=SyntheticBackbone(),num_classes=7).export_image_only(),loaders=image_loaders)
    monkeypatch.setattr(pipeline,'run_training',synthetic_train)
    monkeypatch.setattr(pipeline,'finalize_checkpoints',synthetic_finalize)
    root=Path(cfg['run_dir']).parent/'pipeline'
    status=pipeline.train_final(cfg,root)
    assert roles==[0.0,0.025]
    assert status['status']=='complete' and status['stage']=='validation_locked'
    assert Path(status['final_checkpoint']).is_file() and status['test_evaluation_count']==0
    assert not list(root.rglob('test_attempt.json'))
    with pytest.raises(FileExistsError): pipeline.train_final(cfg,root)
