"""Image-only evaluation of a validation-selected basic-runner checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from verimir.data import ImageOnlyEvalDataset
from verimir.runner import (check_loader_ids, extract_features, file_hash,
                            make_model, metrics_for, validate_manifest, write_json)


def evaluate_checkpoint(checkpoint, *, split='validation', confirm_test=False,
                        device='cpu', workers=0, allow_download=False,
                        model=None, loader=None):
    """Injected models/loaders are for synthetic tests, never benchmark results.

    Test attempts are locked beside the checkpoint before reading test images.
    A lock is not a cross-machine or cross-version test-selection safeguard.
    Only load checkpoints you trust: PyTorch deserialization can execute code.
    """
    synthetic_injection = model is not None or loader is not None
    if split not in ('validation', 'test'):
        raise ValueError('Choose validation or test')
    if split=='test' and not confirm_test:
        raise ValueError('Test requires an explicit --confirm-test')
    checkpoint=Path(checkpoint).resolve()
    parent=checkpoint.parent
    status=json.loads((parent/'run_status.json').read_text(encoding='utf-8'))
    if status['status']!='complete':
        raise ValueError('Training must be complete before separate evaluation')
    if split=='test' and status.get('test_evaluation_count',0)!=0:
        raise FileExistsError('This run already consumed its test attempt')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state.get('checkpoint_type')!='image_only' or state.get('selection_split')!='validation':
        raise ValueError('Expected a validation-selected image-only checkpoint')
    if state.get('checkpoint_soup') or status.get('stage')=='fixed_interpolation':
        lock=json.loads((parent/'finalization.json').read_text(encoding='utf-8'))
        if (lock.get('status')!='validation_locked' or lock.get('checkpoint')!=checkpoint.name
                or lock.get('checkpoint_sha256')!=file_hash(checkpoint)
                or lock.get('manifest_sha256')!=state.get('manifest_sha256')
                or lock.get('weights')!={'control':0.25,'method':0.75}):
            raise ValueError('Final interpolated checkpoint is not the validation-locked artifact')
    config=state['config']
    if status['completed_epochs']!=config['training']['epochs']:
        raise ValueError('Not all configured epochs completed')
    manifest_path=config['data']['manifest']
    if file_hash(manifest_path)!=state['manifest_sha256']:
        raise ValueError('Manifest hash changed since training')
    manifest=validate_manifest(manifest_path)
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable')
    if model is None:
        config['model']['local_files_only']=not allow_download
        model=make_model(config).export_image_only()
    model.load_state_dict(state['model'],strict=True)
    model.to(device).eval()
    if loader is None:
        dataset=ImageOnlyEvalDataset(manifest_path,split,config['data'].get('image_size',224))
        loader=DataLoader(dataset,batch_size=config['training'].get('eval_batch_size',128),
                          shuffle=False,num_workers=workers)
    check_loader_ids(loader,manifest['splits'][split],split)
    result_path=parent/f'{split}_result.json'
    if result_path.exists():
        raise FileExistsError('Result already exists; do not repeat evaluation')
    record={'split':split,'checkpoint_sha256':file_hash(checkpoint),
            'manifest_sha256':state['manifest_sha256'],'protocol':'strict_v2',
            'image_only':True,'test_evaluation_count':1 if split=='test' else 0,
            'synthetic_injection':synthetic_injection or state.get('synthetic_injection',False)}
    marker=parent/'test_attempt.json'
    if split=='test':
        # Exclusive creation: changing an output filename cannot bypass this guard.
        with marker.open('x',encoding='utf-8') as handle:
            json.dump({**record,'status':'started'},handle,indent=2)
        status.update(test_accessed=True,test_evaluation_count=1)
        write_json(parent/'run_status.json',status)
    try:
        features=extract_features(model,loader,device)
        result={**record,'strict_v2':metrics_for(features,state['calibration'],config)}
        write_json(result_path,result)
        if split=='test': write_json(marker,{**record,'status':'complete'})
        return result
    except Exception as error:
        if split=='test':
            write_json(marker,{**record,'status':'failed','error':str(error)})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--split',choices=['validation','test'],default='validation')
    parser.add_argument('--confirm-test',action='store_true')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--allow-download',action='store_true')
    args=parser.parse_args()
    if args.workers<0: parser.error('--workers must be non-negative')
    result=evaluate_checkpoint(**vars(args))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
