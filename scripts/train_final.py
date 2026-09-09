"""Sequentially train both matched arms and finalize; never access test images."""
import argparse
import copy
import json
from pathlib import Path
import yaml
from verimir.finalize import finalize_checkpoints
from verimir.runner import run_training, write_json


def train_final(config,output_root):
    root=Path(output_root).resolve()
    if root.exists():
        raise FileExistsError('Pipeline requires a new root; use the stage CLIs to recover without overwriting')
    if config['loss']['lambda_sv_bt5sd']!=0.025:
        raise ValueError('Full pipeline requires the reference method BDB coefficient 0.025')
    root.mkdir(parents=True,exist_ok=False)
    status={'status':'running','stage':'control','test_evaluation_count':0}
    write_json(root/'pipeline_status.json',status)
    try:
        for arm in ('control','method'):
            cfg=copy.deepcopy(config)
            cfg['run_dir']=str(root/arm)
            if arm=='control': cfg['loss']['lambda_sv_bt5sd']=0.0
            status['stage']=arm
            write_json(root/'pipeline_status.json',status)
            run_training(cfg)
        status['stage']='finalize'
        write_json(root/'pipeline_status.json',status)
        final=finalize_checkpoints(root/'control/checkpoint_best_image_only.pt',
                                  root/'method/checkpoint_best_image_only.pt',root/'final',
                                  device=config['training']['device'],workers=config['training'].get('workers',0),
                                  allow_download=not config['model'].get('local_files_only',True))
        status.update(status='complete',stage='validation_locked',final_checkpoint=str(root/'final'/final['final_checkpoint']))
        write_json(root/'pipeline_status.json',status)
        return status
    except Exception as error:
        status.update(status='failed',error=str(error))
        write_json(root/'pipeline_status.json',status)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--device')
    parser.add_argument('--workers',type=int)
    parser.add_argument('--allow-download',action='store_true')
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if args.device: cfg['training']['device']=args.device
    if args.workers is not None:
        if args.workers<0: parser.error('--workers must be non-negative')
        cfg['training']['workers']=args.workers
    if args.allow_download: cfg['model']['local_files_only']=False
    print(json.dumps(train_final(cfg,args.output_root),indent=2))


if __name__=='__main__':
    main()
