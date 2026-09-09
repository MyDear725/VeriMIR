"""Fixed 0.25 control / 0.75 method interpolation and image-only validation.

No coefficient search, caption-cache access or test-image access occurs here.
The entry point accepts the selected checkpoints of completed basic-runner arms.
"""
import copy
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import TrainImageOnlyCalibrationDataset, ImageOnlyEvalDataset
from .interpolation import build_boundary_anchored_soup
from .runner import (calibration_for, check_loader_ids, extract_features, file_hash,
                     make_model, metrics_for, validate_manifest, write_json)

METHOD_WEIGHT = 0.75


def load_ingredient(path):
    path=Path(path).resolve()
    if path.name!='checkpoint_best_image_only.pt':
        raise ValueError('Use the validation-selected checkpoint_best_image_only.pt of each arm')
    state=torch.load(path,map_location='cpu',weights_only=False)
    status=json.loads((path.parent/'run_status.json').read_text(encoding='utf-8'))
    if status.get('status')!='complete' or status.get('completed_epochs')!=state['config']['training']['epochs']:
        raise ValueError('Every ingredient must have completed all configured epochs')
    if (state.get('test_evaluation_count',0)!=0 or status.get('test_evaluation_count',0)!=0
            or status.get('test_accessed',False)
            or (path.parent/'test_attempt.json').exists()
            or (path.parent/'test_result.json').exists()):
        raise ValueError('An ingredient has already accessed test')
    if state.get('checkpoint_type')!='image_only' or state.get('selection_split')!='validation':
        raise ValueError('Ingredients must be validation-selected image-only checkpoints')
    selection=state['selection']
    if selection.get('protocol')!='strict_v2' or selection.get('metric')!='mAP' or selection.get('split')!='validation':
        raise ValueError('Ingredients must be selected by validation strict_v2 mAP')
    if not math.isfinite(float(selection['value'])):
        raise ValueError('Ingredient validation score must be finite')
    saved_selection=json.loads((path.parent/'checkpoint_selection.json').read_text(encoding='utf-8'))
    if saved_selection!=selection:
        raise ValueError('Checkpoint selection disagrees with the arm record')
    if file_hash(state['config']['data']['manifest'])!=state['manifest_sha256']:
        raise ValueError('Ingredient manifest hash changed')
    return path,state


def matched_configuration(control,method):
    if control['loss']['lambda_sv_bt5sd']!=0 or method['loss']['lambda_sv_bt5sd']!=0.025:
        raise ValueError('Expected BDB=0 control and BDB=0.025 method (in that order)')
    configs=[]
    for cfg in (control,method):
        cfg=copy.deepcopy(cfg)
        cfg.pop('run_dir',None)
        cfg['loss'].pop('lambda_sv_bt5sd')
        # Only execution placement/loader parallelism may differ across arms.
        for key in ('device','workers'):
            cfg['training'].pop(key,None)
        configs.append(cfg)
    if configs[0]!=configs[1]:
        raise ValueError('Control/method configs must match except BDB coefficient and execution paths/devices')


def finalize_checkpoints(control,method,output_dir,*,device='cpu',workers=0,
                         allow_download=False,model=None,loaders=None):
    """Prepare a final checkpoint; injected models/loaders are synthetic-only."""
    injected=model is not None or loaders is not None
    control_path,control_state=load_ingredient(control)
    method_path,method_state=load_ingredient(method)
    if control_path==method_path or control_path.parent==method_path.parent:
        raise ValueError('Control and method must be separate completed runs')
    matched_configuration(control_state['config'],method_state['config'])
    for key in ('manifest_sha256','caption_cache_sha256'):
        if not control_state.get(key) or control_state.get(key)!=method_state.get(key):
            raise ValueError(f'Ingredient {key} differs or is missing')
    out=Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError('Use a new final output directory; existing outputs are never overwritten')
    config=copy.deepcopy(method_state['config'])
    config['run_dir']=str(out)
    manifest=validate_manifest(config['data']['manifest'])
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable')
    if workers<0:
        raise ValueError('workers must be non-negative')
    synthetic=injected or control_state.get('synthetic_injection',False) or method_state.get('synthetic_injection',False)
    soup=build_boundary_anchored_soup(control_state,method_state,METHOD_WEIGHT,[control_path,method_path])
    soup['config']=config  # Do not inherit the control's disabled BDB setting.
    soup['synthetic_injection']=synthetic
    soup['runner']='public-b32-final-interpolation-v1'
    if model is None:
        model_config=copy.deepcopy(config)
        model_config['model']['local_files_only']=not allow_download
        model=make_model(model_config).export_image_only()
    model.load_state_dict(soup['model'],strict=True)
    model.to(device).eval()
    if loaders is None:
        size=config['data'].get('image_size',224)
        manifest_path=config['data']['manifest']
        common={'batch_size':config['training'].get('eval_batch_size',128),
                'shuffle':False,'num_workers':workers}
        loaders=(DataLoader(TrainImageOnlyCalibrationDataset(manifest_path,size),**common),
                 DataLoader(ImageOnlyEvalDataset(manifest_path,'validation',size),**common))
    train_loader,validation_loader=loaders
    check_loader_ids(train_loader,manifest['splits']['train'],'train calibration')
    check_loader_ids(validation_loader,manifest['splits']['validation'],'validation')
    ingredients=[{'role':role,'path':str(path),'sha256':file_hash(path),
                  'weight':weight,'selected_epoch':state['epoch']} for role,path,state,weight in (
        ('control',control_path,control_state,0.25),('method',method_path,method_state,0.75))]
    status={'status':'finalizing','stage':'fixed_interpolation',
            'completed_epochs':config['training']['epochs'],
            'completed_epochs_scope':'each ingredient; no additional training in finalization',
            'test_accessed':False,'test_evaluation_count':0,'synthetic_injection':synthetic}
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'run_status.json',status)
    write_json(out/'config.json',config)
    write_json(out/'ingredients.json',ingredients)
    try:
        # No caption cache or validation labels are used to fit score calibration.
        train=extract_features(model,train_loader,device)
        soup['calibration']=calibration_for(train,config)
        metrics=metrics_for(extract_features(model,validation_loader,device),soup['calibration'],config)
        if not math.isfinite(metrics['mAP']) or metrics['num_valid_queries']==0:
            raise ValueError('Final validation requires finite mAP and eligible queries')
        soup['selection'].update(value=metrics['mAP'],source='fixed_0.25_control_0.75_method',
                                 coefficient_selection='fixed a priori; no coefficient search')
        soup['checkpoint_soup'].update(calibration='refitted_on_train_images',ingredients_verified=ingredients)
        soup['finalization']={'status':'validation_locked','weights':{'control':0.25,'method':0.75},
                               'calibration_split':'train','validation_split':'validation'}
        checkpoint=out/'checkpoint_final_image_only.pt'
        torch.save(soup,checkpoint)
        lock={**soup['finalization'],'checkpoint':checkpoint.name,'checkpoint_sha256':file_hash(checkpoint),
              'manifest_sha256':soup['manifest_sha256'],'ingredients':ingredients}
        write_json(out/'finalization.json',lock)
        write_json(out/'checkpoint_selection.json',soup['selection'])
        write_json(out/'best_validation_metrics.json',{'strict_v2':metrics})
        write_json(out/'validation_result.json',{'split':'validation','protocol':'strict_v2','image_only':True,
                   'strict_v2':metrics,'test_evaluation_count':0,'synthetic_injection':synthetic,
                   'checkpoint_sha256':lock['checkpoint_sha256'],'manifest_sha256':soup['manifest_sha256']})
        status.update(status='complete',final_checkpoint=checkpoint.name,best_validation_mAP=metrics['mAP'])
        write_json(out/'run_status.json',status)
        return status
    except Exception as error:
        status.update(status='failed',error=str(error))
        write_json(out/'run_status.json',status)
        raise
