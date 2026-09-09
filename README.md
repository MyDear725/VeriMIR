# VeriMIR: Self-Verified Bidirectional Boundary Learning for Medical Image Retrieval

Official implementation for the paper "VERIMIR: SELF-VERIFIED BIDIRECTIONAL BOUNDARY LEARNING FOR MEDICAL IMAGE RETRIEVAL".

## Abstract

Medical image retrieval (MIR) supports clinical assessment by returning similar reference cases beyond categorical predictions. Reliable retrieval therefore requires discriminative representations and stable rankings of relevant cases. However, augmentations that alter local evidence can distort retrieval neighborhoods and introduce unreliable supervision, while conventional pairwise objectives do not explicitly protect the Top-5 cutoff. We propose VeriMIR, a framework integrating image self-verification (ISV) with bidirectional Top-5 boundary learning (BDB). ISV reweights augmented views according to their appearance and semantic agreement with original images and attenuates conflicting gradient contributions. BDB uses a frozen teacher constructed from training-image descriptors to encourage query-side relevant coverage and suppress gallery-side cross-class intrusions. Training-only LLM-generated captions provide semantic supervision. The final model performs retrieval with a single calibrated image encoder, without captions, a teacher, or verification operations at inference. VeriMIR achieves mAP scores of 79.59% and 92.07% on ISIC 2018 and Kvasir v2, with absolute gains of 2.37% and 2.64% over the strongest baseline on each dataset, respectively.

## Quick start

### Installation

Use Python 3.11+ and a PyTorch/torchvision installation suitable for your device. Run the following from the repository root; all subsequent commands run inside `code/`.

```bash
cd code
pip install -r requirements.txt
```

### Data preparation

Download ISIC 2018 or Kvasir v2 and prepare the image paths and train/validation/test manifest following [data/README.md](data/README.md). Set `data.manifest` and `data.caption_cache` in `code/configs/isic2018_b32.yaml` or `code/configs/kvasir_b32.yaml`. Caption JSON files are provided in `data/captions/`.

For ISIC 2018, create the training caption cache:

```bash
python -m scripts.cache_train_captions --captions ../data/captions/isic2018_train_only.json --manifest ../data/splits/isic2018_official_fixed_v1.json --output ../data/captions/isic2018_train_clip_b32.pt --allow-download
```

### Training

```bash
python -m scripts.train_final --config configs/isic2018_b32.yaml --output-root outputs/isic2018 --allow-download
```

This trains the control and full method, combines their model parameters as **0.25 control + 0.75 method**, and recalibrates the final image model on training images. For Kvasir v2, prepare its caption cache using the paths in [data/README.md](data/README.md), then use `configs/kvasir_b32.yaml` and a separate output directory.

### Evaluation

```bash
python -m scripts.evaluate --checkpoint outputs/isic2018/final/checkpoint_final_image_only.pt --split test --confirm-test
```

Evaluation reports mAP, R@1, R@5, P@1, and P@5, excluding each query image from its own ranking.
