# Bring your own authorized data

This directory now includes two audited **training-only caption JSON files** and their public dataset image identifiers. It contains no medical images, model weights or embedding caches. No patient-identifying fields were found by the documented pattern scan, which is not a comprehensive privacy certificate. Dataset/caption redistribution rights must be confirmed separately from code rights.

Each reference config points to a locally supplied split manifest and caption cache. Run integrations from the repository root so relative paths resolve consistently.

## Manifest shape (illustrative, not an experimental split)

```json
{
  "dataset": "isic2018",
  "root": "/path/to/authorized/images",
  "splits": {
    "train": [{"sample_id": 1, "source_id": "train_example.jpg", "relative_path": "train_example.jpg", "label": 0}],
    "validation": [{"sample_id": 2, "source_id": "val_example.jpg", "relative_path": "val_example.jpg", "label": 0}],
    "test": [{"sample_id": 3, "source_id": "test_example.jpg", "relative_path": "test_example.jpg", "label": 0}]
  }
}
```

The miniature illustration is not sufficient for training or metric computation. Actual train labels must cover contiguous class indices; BDB requires more than 10 training images. Sample IDs must be unique, stable integers. Keep validation and test independent of train.

## Cached training text

```python
# Format only: raw train captions are supplied, embeddings are not.
cache = {
    "sample_ids": train_sample_ids,  # integer IDs matching the entire train manifest
    "embeddings": train_text_embeddings,  # float [N_train, 512], CLIP B/32 text space
}
```

Use the matching fixed CLIP B/32 text encoder to prepare training targets offline. Preserve the original preprocessing/normalization and caption-generation protocol if reproducing a prior experiment. Exact benchmark manifests and the original raw-caption generation pipeline are not distributed in this core package. The provided captions contain training class-name prefixes. No validation/test captions may be passed to the model.

After supplying the frozen split manifest, create the local cache with:

```bash
python -m scripts.cache_train_captions --captions data/captions/isic2018_train_only.json --manifest data/splits/isic2018_official_fixed_v1.json --output data/captions/isic2018_train_clip_b32.pt
python -m scripts.cache_train_captions --captions data/captions/kvasir_train_only.json --manifest data/splits/kvasir_strict_seed42_dedup_v1.json --output data/captions/kvasir_train_clip_b32.pt
```

These commands require local B/32 weights by default; use `--allow-download` to permit downloading them. They default to CPU; `--device cuda:0` is optional. They validate train membership before loading CLIP and refuse to overwrite an existing output cache.
