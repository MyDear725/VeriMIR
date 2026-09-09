# VeriMIR: Self-Verified Bidirectional Boundary Learning for Medical Image Retrieval

Official implementation for the paper "VERIMIR: SELF-VERIFIED BIDIRECTIONAL BOUNDARY LEARNING FOR MEDICAL IMAGE RETRIEVAL".

## Abstract

Medical image retrieval requires reliable representations that capture clinically relevant similarities. We present VeriMIR, a framework with Image Self-Verification (ISV) and Bidirectional Top-5 Boundary Learning (BDB). ISV checks feature consistency and gradient agreement between original and augmented images to reduce conflicting updates from unreliable augmentations. BDB uses a frozen teacher built from training image descriptors to promote relevant samples into the query-side Top-5 while suppressing cross-class intrusions from the gallery side. Captions provide semantic supervision only during training. At inference, retrieval uses a single image encoder without captions, the teacher, or verification modules. The framework is evaluated on ISIC 2018 and Kvasir v2.

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
