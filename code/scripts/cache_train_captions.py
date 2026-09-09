"""Encode audited training captions using the reference B/32 text recipe."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from verimir.captions import audit_training_captions


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captions', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--allow-download', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing cache')
    audit = audit_training_captions(args.captions, args.manifest)
    print(json.dumps(audit, indent=2))
    if not audit['recorded_manifest_hash_matches_current']:
        print('NOTICE: recorded manifest hash differs; exact current train membership verified.')
    rows = json.loads(Path(args.captions).read_text(encoding='utf-8'))['captions']
    from transformers import CLIPModel, CLIPTokenizerFast
    model_name = 'openai/clip-vit-base-patch32'
    tokenizer = CLIPTokenizerFast.from_pretrained(model_name, local_files_only=not args.allow_download)
    clip = CLIPModel.from_pretrained(model_name, local_files_only=not args.allow_download).to(args.device).eval()
    clip.requires_grad_(False)
    embeddings = []
    for start in range(0, len(rows), args.batch_size):
        batch = [row['caption'] for row in rows[start:start + args.batch_size]]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=77, return_tensors='pt').to(args.device)
        features = clip.get_text_features(**tokens)
        embeddings.append(F.normalize(features.float(), dim=-1).half().cpu())
        print(f'encoded {start + len(batch)}/{len(rows)}', flush=True)
    cache = {'split':'train-only', 'sample_ids':[int(r['sample_id']) for r in rows],
             'source_ids':[r['source_id'] for r in rows], 'embeddings':torch.cat(embeddings),
             'model':model_name, 'caption_file_sha256':audit['caption_json_sha256']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        torch.save(cache, handle)


if __name__ == '__main__':
    main()
