"""Basic B/32 train/validation lifecycle; never evaluates the test split.

This is a newly packaged runner, not a claim of byte-identical reproduction
of a historical experiment. The supplied method/loss settings are preserved.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .bank import MemoryBank, RankSnapshotBank
from .data import TrainCaptionDataset, DeterministicTrainDataset, ImageOnlyEvalDataset, PKBatchSampler
from .evaluation import image_scores
from .losses import caption_class_prototypes, build_bidirectional_top5_teacher
from .model import VeriMIR
from .retrieval import fit_train_zscore, strict_v2_metrics_from_scores
from .training import training_objective


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def validate_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    sets = {}
    for split in ('train','validation','test'):
        rows = manifest['splits'][split]
        if not rows:
            raise ValueError(f'Empty {split} split')
        sets[split] = {}
        for field in ('sample_id','source_id','relative_path'):
            values = [r[field] for r in rows]
            if len(set(values)) != len(values):
                raise ValueError(f'Duplicate {field} in {split}')
            sets[split][field] = set(values)
    for a,b in (('train','validation'),('train','test'),('validation','test')):
        for field in sets[a]:
            if sets[a][field] & sets[b][field]:
                raise ValueError(f'{field} overlap between {a} and {b}')
    return manifest


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def cpu_state(module):
    return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}


def update_ema(previous, model, decay):
    result = {}
    for key,value in model.state_dict().items():
        value = value.detach().cpu()
        result[key] = previous[key]*decay + value*(1-decay) if value.is_floating_point() else value.clone()
    return result


def build_optimizer(model, training):
    heads = list(model.appearance_head.parameters()) + list(model.semantic_head.parameters()) + list(model.appearance_classifier.parameters())
    heads = [p for p in heads if p.requires_grad]
    head_ids = {id(p) for p in heads}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = []
    if backbone:
        groups.append({'params':backbone,'lr':training['lr'],'group_name':'backbone'})
    if heads:
        groups.append({'params':heads,'lr':training['lr']*training['head_lr_multiplier'],'group_name':'heads'})
    return torch.optim.AdamW(groups, weight_decay=training['weight_decay'])


def set_epoch_lr(optimizer, training, epoch):
    milestones = [int(x) for x in training.get('lr_milestones',[])]
    if milestones != sorted(set(milestones)) or any(x<1 for x in milestones):
        raise ValueError('LR milestones must be unique positive ascending epochs')
    gamma = float(training.get('lr_gamma',0.1))
    if not 0 < gamma <= 1:
        raise ValueError('LR gamma must be in (0,1]')
    factor = gamma**sum(epoch>=x for x in milestones)
    for group in optimizer.param_groups:
        multiplier = training['head_lr_multiplier'] if group['group_name']=='heads' else 1.0
        group['lr'] = float(training['lr'])*multiplier*factor


def make_model(config):
    m = config['model']
    if m['name'] != 'openai/clip-vit-base-patch32':
        raise ValueError('Only CLIP ViT-B/32 is supported')
    if m.get('embedding_dim',512)!=512 or not m.get('residual_init',True):
        raise ValueError('This runner requires the reference 512-D residual heads')
    model = VeriMIR(num_classes=m['num_classes'], dropout=m.get('dropout',0.1),
                    model_path=m['name'], local_files_only=m.get('local_files_only',True))
    if m.get('freeze_backbone',False):
        model.backbone.requires_grad_(False)
    return model


def make_loaders(config):
    d,t = config['data'],config['training']
    train = TrainCaptionDataset(d['manifest'],d['caption_cache'],d.get('image_size',224),d.get('augmentation_profile','standard'))
    clean = DeterministicTrainDataset(d['manifest'],d['caption_cache'],d.get('image_size',224))
    validation = ImageOnlyEvalDataset(d['manifest'],'validation',d.get('image_size',224))
    sampler = PKBatchSampler(train.labels,t['classes_per_batch'],t['samples_per_class'],config['seed'])
    common = {'num_workers':t.get('workers',4),'pin_memory':str(t['device']).startswith('cuda')}
    return (DataLoader(train,batch_sampler=sampler,**common),
            DataLoader(clean,batch_size=t.get('eval_batch_size',128),shuffle=False,**common),
            DataLoader(validation,batch_size=t.get('eval_batch_size',128),shuffle=False,**common))


@torch.no_grad()
def extract_features(model, loader, device, include_train_text=False):
    model.eval()
    output = {key:[] for key in ('appearance','semantic','labels','sample_ids','bank_indices','text')}
    forbidden = {'caption','caption_id','caption_embedding','text_input_ids','text_encoder'}
    if include_train_text:
        forbidden.remove('caption_embedding')
    for batch in loader:
        if forbidden & batch.keys():
            raise ValueError('Text fields are forbidden in image-only evaluation')
        a,s = model(batch['image'].to(device))
        output['appearance'].append(a.float().cpu()); output['semantic'].append(s.float().cpu())
        output['labels'].append(batch['label'].cpu()); output['sample_ids'].append(batch['sample_id'].cpu())
        if 'bank_index' in batch:
            output['bank_indices'].append(batch['bank_index'].cpu())
        if include_train_text:
            output['text'].append(batch['caption_embedding'].float().cpu())
    if not output['appearance']:
        raise ValueError('Empty image loader')
    return {k:torch.cat(v) for k,v in output.items() if v}


def calibration_for(features, config):
    if config['fusion']['calibration'] != 'train_zscore':
        raise ValueError('This runner requires train_zscore calibration')
    return fit_train_zscore(features['appearance'],features['semantic'],
                            config['fusion'].get('calibration_pairs',100000), config['seed'])


def metrics_for(features, calibration, config):
    scores = image_scores(features['appearance'],features['semantic'],calibration,
                          config['fusion']['appearance_weight'],config['fusion']['beta'])
    return strict_v2_metrics_from_scores(scores,features['labels'],features['sample_ids'])


def check_loader_ids(loader, expected_rows, name):
    actual = {(int(r['sample_id']),r['source_id']) for r in loader.dataset.records}
    expected = {(int(r['sample_id']),r['source_id']) for r in expected_rows}
    if len(loader.dataset.records)!=len(expected_rows) or actual!=expected:
        raise ValueError(f'{name} loader does not match the frozen split')


def run_training(config, *, model=None, loaders=None):
    """Train one arm; model/loader injection is for synthetic testing only.

    No resume support: existing output directories are rejected. Test images
    are never loaded. A separate explicit test command is required.
    """
    synthetic_injection = model is not None or loaders is not None
    config = copy.deepcopy(config)
    config['data']['manifest'] = str(Path(config['data']['manifest']).resolve())
    config['data']['caption_cache'] = str(Path(config['data']['caption_cache']).resolve())
    manifest = validate_manifest(config['data']['manifest'])
    labels = {int(r['label']) for r in manifest['splits']['train']}
    if labels != set(range(config['model']['num_classes'])):
        raise ValueError('Train classes do not match model.num_classes')
    t=config['training']
    if config['evaluation'] != {'primary_protocol':'strict_v2','selection_metric':'mAP'}:
        raise ValueError('This runner selects by validation strict_v2 mAP only')
    if t.get('resume') or t.get('init_image_only'):
        raise ValueError('Resume/initialization overrides are not supported by this basic runner')
    if not t.get('use_augmented') or not t.get('verified_augmentation') or not t['gradient_agreement'].get('enabled'):
        raise ValueError('This runner supports the full ISV method (including the matched BDB=0 control)')
    if config['fusion'].get('rerank') or config['fusion'].get('adaptive'):
        raise ValueError('Extra inference refinements are not supported')
    if int(t['epochs'])<1 or not 0<=float(t.get('ema_decay',0.9))<1:
        raise ValueError('Invalid epoch count or EMA decay')
    device=torch.device(t['device'])
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; explicitly set --device cpu for CPU execution')
    out=Path(config['run_dir']).resolve()
    if out.exists():
        raise FileExistsError('Use a new run directory; existing runs are never overwritten')
    set_seed(int(config['seed']))
    model=(model if model is not None else make_model(config)).to(device)
    train_loader,clean_loader,val_loader=loaders if loaders is not None else make_loaders(config)
    for loader,split in ((train_loader,'train'),(clean_loader,'train'),(val_loader,'validation')):
        check_loader_ids(loader,manifest['splits'][split],split)
    # Audit the exact cache consumed by the training loader, even when loaders are injected.
    cache=torch.load(config['data']['caption_cache'],map_location='cpu',weights_only=False)
    cache_ids=[int(x) for x in cache['sample_ids']]
    expected_ids={int(r['sample_id']) for r in manifest['splits']['train']}
    if len(cache_ids)!=len(set(cache_ids)) or set(cache_ids)!=expected_ids or len(cache['embeddings'])!=len(cache_ids):
        raise ValueError('Caption cache must cover exactly the train split')
    caption_by_id={sid:cache['embeddings'][i].float() for i,sid in enumerate(cache_ids)}
    prototype_text=torch.stack([caption_by_id[int(r['sample_id'])] for r in train_loader.dataset.records])
    prototype_labels=torch.tensor([r['label'] for r in train_loader.dataset.records])
    prototypes=caption_class_prototypes(prototype_text,prototype_labels).to(device)
    del cache,caption_by_id,prototype_text
    optimizer=build_optimizer(model,t)
    amp=bool(t.get('amp',True) and device.type=='cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    ema=cpu_state(model)
    memory=MemoryBank(len(train_loader.dataset),512,config['reliability'].get('bank_momentum',0.9))
    out.mkdir(parents=True,exist_ok=False)
    manifest_hash=file_hash(config['data']['manifest'])
    provenance={'manifest_sha256':manifest_hash,'caption_cache_sha256':file_hash(config['data']['caption_cache']),
                'synthetic_injection':synthetic_injection}
    write_json(out/'config.json',config)
    write_json(out/'input_hashes.json',provenance)
    status={'status':'running','completed_epochs':0,'test_accessed':False,'test_evaluation_count':0,
            'runner':'public-basic-b32-v1','selection_split':'validation','synthetic_injection':synthetic_injection}
    write_json(out/'run_status.json',status)
    history=[]; best=-float('inf')
    try:
        # Snapshot extraction must not consume the RNG stream of the first training batch.
        rng=capture_rng()
        try:
            initial=extract_features(model,clean_loader,device,include_train_text=True)
        finally:
            restore_rng(rng)
        rank=RankSnapshotBank(len(train_loader.dataset),512,512)
        rank.replace(initial['bank_indices'],initial['appearance'],initial['semantic'],initial['labels'],
                     initial['sample_ids'],calibration_for(initial,config),version=0)
        rank_bank={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in rank.as_loss_bank(0).items()}
        teacher=None
        if config['loss']['lambda_sv_bt5sd']>0:
            teacher=build_bidirectional_top5_teacher(rank_bank,beta=config['fusion']['beta'],
                appearance_weight=config['fusion']['appearance_weight'],top_k=config['loss']['sv_bt5sd_top_k'],
                chunk_size=config['loss']['sv_bt5sd_teacher_chunk_size'])
        write_json(out/'rank_snapshot.json',rank.metadata)
        del initial
        for epoch in range(1,int(t['epochs'])+1):
            started=time.time()
            set_epoch_lr(optimizer,t,epoch)
            train_loader.batch_sampler.set_epoch(epoch)
            model.train(); total=0.; steps=0
            triplet=memory.as_triplet_bank(torch.ones(len(memory.labels))) if memory.initialized.any() else None
            if triplet is not None:
                triplet={k:v.to(device) for k,v in triplet.items()}
            for batch in train_loader:
                batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
                batch['split']='train'
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                    loss,_=training_objective(model,batch,config,epoch,prototypes,rank_bank,teacher,triplet)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float(t.get('grad_clip',5.0)))
                if not torch.isfinite(norm):
                    raise FloatingPointError('Non-finite gradient norm')
                scaler.step(optimizer); scaler.update()
                total+=float(loss.detach()); steps+=1
            if not steps:
                raise ValueError('No training batches')
            ema_active=epoch>=int(t.get('ema_start_epoch',1))
            if ema_active:
                ema=update_ema(ema,model,float(t.get('ema_decay',0.9)))
            clean=extract_features(model,clean_loader,device,include_train_text=True)
            memory.update(clean['bank_indices'],clean['appearance'],clean['semantic'],clean['text'],clean['labels'],clean['sample_ids'])
            online_calibration=calibration_for(clean,config)
            online_metrics=metrics_for(extract_features(model,val_loader,device),online_calibration,config)
            online_state=cpu_state(model)
            candidates=[('online',online_metrics,online_calibration,online_state)]
            if ema_active:
                model.load_state_dict(ema)
                ema_calibration=calibration_for(extract_features(model,clean_loader,device,include_train_text=True),config)
                ema_metrics=metrics_for(extract_features(model,val_loader,device),ema_calibration,config)
                candidates.append(('ema',ema_metrics,ema_calibration,ema))
                model.load_state_dict(online_state)
            source,metrics,calibration,selected_state=max(candidates,key=lambda x:x[1]['mAP'])
            if not math.isfinite(metrics['mAP']) or metrics['num_valid_queries']==0:
                raise ValueError('Validation needs eligible queries and finite mAP')
            selection={'split':'validation','protocol':'strict_v2','metric':'mAP','value':metrics['mAP'],'epoch':epoch,'source':source}
            if metrics['mAP']>best:
                best=metrics['mAP']
                # Keep only the deployable image model keys; exclude the training classifier.
                # Do not deep-copy a GPU model just to discover its image keys.
                image_keys={k for k in selected_state if k.startswith(
                    ('backbone.','appearance_head.','semantic_head.'))}
                state={'checkpoint_type':'image_only','model':{k:selected_state[k] for k in selected_state if k in image_keys},
                       'config':config,'calibration':calibration,'epoch':epoch,'selection':selection,
                       'selection_split':'validation','test_evaluation_count':0,**provenance,
                       'runner':'public-basic-b32-v1'}
                torch.save(state,out/'checkpoint_best_image_only.pt')
                write_json(out/'checkpoint_selection.json',selection)
                write_json(out/'best_validation_metrics.json',{'strict_v2':metrics})
            history.append({'epoch':epoch,'mean_train_loss':total/steps,'steps':steps,'seconds':time.time()-started,
                            'selected_source':source,'online_strict_v2':online_metrics,
                            'ema_strict_v2':ema_metrics if ema_active else None,
                            'learning_rates':[g['lr'] for g in optimizer.param_groups]})
            write_json(out/'validation_history.json',history)
            status['completed_epochs']=epoch
            write_json(out/'run_status.json',status)
            interval=int(t.get('checkpoint_last_interval',t['epochs']))
            if epoch==int(t['epochs']) or (interval>0 and epoch%interval==0):
                torch.save({'model':online_state,'optimizer':optimizer.state_dict(),'epoch':epoch,'ema_state':ema,
                            'config':config,'checkpoint_type':'training_last','test_evaluation_count':0},out/'checkpoint_last.pt')
            print(f"epoch {epoch}/{t['epochs']} loss={total/steps:.5f} validation_mAP={metrics['mAP']:.4f} source={source}",flush=True)
        status.update(status='complete',best_validation_mAP=best)
        write_json(out/'run_status.json',status)
        return status
    except Exception as error:
        status.update(status='failed',error=str(error))
        write_json(out/'run_status.json',status)
        raise
