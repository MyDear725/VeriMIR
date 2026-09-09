"""Train one B/32 arm with validation-only checkpoint selection."""
import argparse
from pathlib import Path
import yaml
from verimir.runner import run_training


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--run-dir')
    parser.add_argument('--device')
    parser.add_argument('--workers',type=int)
    parser.add_argument('--arm',choices=['method','control'],default='method')
    parser.add_argument('--allow-download',action='store_true')
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if args.arm=='control':
        config['loss']['lambda_sv_bt5sd']=0.0
        if not args.run_dir:
            parser.error('--arm control requires its own --run-dir')
    if args.run_dir: config['run_dir']=args.run_dir
    if args.device: config['training']['device']=args.device
    if args.workers is not None:
        if args.workers<0: parser.error('--workers must be non-negative')
        config['training']['workers']=args.workers
    if args.allow_download: config['model']['local_files_only']=False
    run_training(config)


if __name__=='__main__':
    main()
