"""Interpolate completed arms (0.25 control + 0.75 method), recalibrate and validate."""
import argparse
import json
from verimir.finalize import finalize_checkpoints


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',required=True)
    parser.add_argument('--method',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--allow-download',action='store_true')
    print(json.dumps(finalize_checkpoints(**vars(parser.parse_args())),indent=2))


if __name__=='__main__':
    main()
