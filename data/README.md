# Data preparation

Download ISIC 2018 or Kvasir v2 separately. Caption JSON files are included in `captions/`.

## Image manifest

Create a JSON manifest with the following structure, using an absolute image root and dataset-relative image paths:

```json
{
  "dataset": "isic2018",
  "root": "/path/to/images",
  "splits": {
    "train": [{"sample_id": 1, "source_id": "train.jpg", "relative_path": "train.jpg", "label": 0}],
    "validation": [{"sample_id": 2, "source_id": "val.jpg", "relative_path": "val.jpg", "label": 0}],
    "test": [{"sample_id": 3, "source_id": "test.jpg", "relative_path": "test.jpg", "label": 0}]
  }
}
```

This illustrates the format; replace the rows with your dataset samples. Use unique integer sample IDs and labels 0–6 for ISIC 2018 or 0–7 for Kvasir v2. Training must contain all classes and more than 10 images. The training `sample_id` and `source_id` pairs must match the selected caption JSON exactly. If using a different split, prepare a matching training caption JSON and update its `num_captions`; do not include validation/test samples.

The default manifest paths are `data/splits/isic2018_official_fixed_v1.json` and `data/splits/kvasir_strict_seed42_dedup_v1.json`. For Kvasir, set `"dataset": "kvasir"`. You can change manifest and cache paths in `code/configs/`.

## Training caption cache

Run from the repository's `code/` directory:

```bash
python -m scripts.cache_train_captions --captions ../data/captions/isic2018_train_only.json --manifest ../data/splits/isic2018_official_fixed_v1.json --output ../data/captions/isic2018_train_clip_b32.pt --allow-download
python -m scripts.cache_train_captions --captions ../data/captions/kvasir_train_only.json --manifest ../data/splits/kvasir_strict_seed42_dedup_v1.json --output ../data/captions/kvasir_train_clip_b32.pt --allow-download
```

Run the command for your dataset. `--allow-download` permits downloading the CLIP ViT-B/32 weights; omit it if they are cached locally. Captions and their class prototypes supervise training only; evaluation uses images.
