# VeriMIR — ViT-B/32 Core Implementation

Core components of **VeriMIR: self-verified bidirectional boundary learning for medical image retrieval**.

This repository contains **one method**, with dataset-specific configurations for ISIC-2018 and Kvasir. Both use `openai/clip-vit-base-patch32`. The models are trained independently on each dataset; they do not share a cross-dataset checkpoint.

## Release scope

This is a **core implementation with training and final-interpolation runners**, not a certified benchmark reproduction. It includes the loss, verification, memory-bank, projection-head, interpolation and strict_v2 metric implementations extracted from the reference experiments. A compact B/32 adapter, full-method batch objective, epoch/EMA/validation/checkpoint loop, fixed 0.75 method / 0.25 control finalization and separate image-only evaluation entry are provided alongside CPU tests.

Included training text: 10,015 ISIC-2018 captions and 5,600 Kvasir captions, identical to the raw JSON files used for the reference B/32 embedding caches. See [caption provenance](data/captions/README.md).

Not included: dataset images, exact benchmark split manifests, validation/test captions, pretrained weights, trained checkpoints, embedding caches, experiment logs, hyperparameter-search history, or claimed reproduced scores. See [training details](docs/TRAINING.md) for commands and limitations. The synthetic tests are not benchmark experiments. `scripts.train` trains one arm; `scripts.train_final` runs both arms sequentially and automatically finalizes their fixed interpolation without accessing test images.

## Method

- **Dual branch:** a shared CLIP ViT-B/32 image encoder feeds appearance and semantic projection heads. Training-only text embeddings supervise the semantic image branch; the semantic branch is still computed from images at inference.
- **ISV:** detached feature-consistency weights reallocate augmented-sample losses. A retrieval-interface gradient check attenuates conflicting augmented updates. Caption self-verification is disabled in the supplied reference configurations.
- **BDB:** a frozen, training-only snapshot provides outgoing/query-side and incoming/gallery-side Top-5 boundary supervision. The teacher is not used at inference.
- **Image-only retrieval:** train-calibrated branch similarities are combined; query images are excluded from their own rankings. Labels are used for scoring retrieval correctness, not to form the model's similarity scores.
- **Supporting interpolation:** compatible validation-selected image-only checkpoints can be interpolated into a single deployable model. This supporting step is retained because it was part of the reference method; it is not presented as a third core novelty.

The complete base objective also contains the inherited semantic supervision, triplet/ranking and auxiliary training losses in the reference configuration. ISV and BDB are not substitutes for those losses.

## Code map

| File | Purpose |
|---|---|
| `verimir/model.py`, `heads.py` | B/32 adapter, appearance/semantic heads, image-only export |
| `verimir/losses.py` | Reference base losses, feature-level ISV, outgoing/incoming BDB |
| `verimir/gradient_agreement.py` | Gradient-level ISV |
| `verimir/bank.py`, `schedule.py` | Training memory/snapshot bank and two-epoch BDB pulse |
| `verimir/training.py` | Full-method loss assembly for one training batch |
| `verimir/runner.py`, `scripts/train.py` | Optimizer, epoch loop, frozen teacher, memory refresh, EMA, validation-only selection |
| `verimir/finalize.py`, `scripts/finalize.py` | Fixed 0.25 control + 0.75 method interpolation, train-image recalibration, validation lock |
| `scripts/train_final.py` | Sequential control/method training followed by automatic finalization |
| `scripts/evaluate.py` | Separate image-only evaluation; explicit, locally guarded one-shot test |
| `verimir/data.py` | Train-caption and image-only evaluation datasets; train sampler |
| `verimir/retrieval.py`, `evaluation.py` | Calibration, image similarity, strict_v2 metrics |
| `verimir/checkpoint_soup.py`, `interpolation.py` | Validation-selected checkpoint interpolation |
| `configs/` | Two sanitized, resolved B/32 reference configurations |
| `examples/`, `tests/` | Synthetic integration example and CPU tests |

## Installation and smoke test

Python 3.11+ is recommended. Install a matching PyTorch/torchvision pair for your CPU/CUDA platform, then:

```bash
pip install -r requirements.txt
python -m pytest -q
python -m examples.smoke_core
```

The example uses random feature tensors and a tiny mock backbone. It exercises ISV+BDB and backpropagation **without downloading weights or accessing any medical images**. To construct the real B/32 model, obtain the official CLIP weights separately:

```python
from verimir.model import VeriMIR

# local_files_only=False permits an explicit download of the public B/32 weights.
model = VeriMIR(num_classes=7, local_files_only=False)
inference_model = model.export_image_only()
# inference_model(images) -> appearance, semantic
```

For Kvasir, use `num_classes=8`. Inputs use the preprocessing in `verimir/data.py`. The compact adapter preserves the reference exported image-only parameter layout; full training reproducibility, including RNG consumption during initialization, has not been certified for this repackaged adapter.

## Training and final model

First supply an authorized image dataset, its fixed split manifest and a train-caption embedding cache following [data preparation](data/README.md). From the repository root:

```bash
python -m scripts.train_final --config configs/isic2018_b32.yaml --output-root outputs/isic2018_full
python -m scripts.train_final --config configs/kvasir_b32.yaml --output-root outputs/kvasir_full
```

Run these separately or on explicitly assigned free devices, not simultaneously on the same GPU by default. Each command trains a matched control and method (20 epochs each in the supplied configs), then computes **0.25 control + 0.75 method in parameter space**, recalibrates on training images, and evaluates/locks the final checkpoint on validation. There is no coefficient search and no caption access during finalization. Add `--allow-download` if public B/32 pretrained weights are not cached. Existing run directories are rejected. Test is **never** run automatically.

For ISIC-2018, the final artifact is `outputs/isic2018_full/final/checkpoint_final_image_only.pt`. After completing all validation decisions, explicitly test **only that final model**:

```bash
python -m scripts.evaluate --checkpoint outputs/isic2018_full/final/checkpoint_final_image_only.pt --split test --confirm-test
```

The finalization file records ingredient hashes, fixed weights and the locked checkpoint hash. The evaluator verifies this lock. Independent image-only exports deep-copy all image modules: later training, buffer changes and mode switches cannot modify an exported model.

For single-arm commands and finalizing already completed arms without retraining, see [TRAINING.md](docs/TRAINING.md). No full benchmark has been rerun to certify these repackaged runners.

## Dataset configurations

| Setting | ISIC-2018 | Kvasir |
|---|---:|---:|
| Classes | 7 | 8 |
| Seed | 144018 | 144042 |
| Samples per class per batch | 2 | 2 |
| Training batch size | 14 | 16 |
| Epochs | 20 | 20 |
| Backbone | CLIP ViT-B/32 | CLIP ViT-B/32 |

The architecture, loss weights, learning-rate schedule, ISV/BDB settings, fusion and evaluation definitions are identical. The configuration-equivalence test explicitly checks this after excluding the listed dataset-dependent fields.

## Data and evaluation boundaries

See [data format](data/README.md) and [evaluation protocol](docs/PROTOCOL.md). Only the two audited train-caption JSON files are allowlisted for Git. Do not add dataset images, validation/test captions, private metadata or weights. Caption/data redistribution terms are separate from the code license and must be confirmed before public release.

For full reference runs, train/validation/test splits are fixed before training, checkpoints are selected using validation strict_v2 mAP, and the frozen final model is evaluated once on test. Repeated cross-version selection based on test scores is not justified by a per-run one-shot flag.

## Provenance, acknowledgments and licensing

- Reference: [SD-MIR](https://github.com/CaryXiang/SD-MIR) and its acknowledged [X-MIR](https://gitlab.kitware.com/brianhhu/x-mir) foundation.
- Backbone: [OpenAI CLIP](https://github.com/openai/CLIP), via Hugging Face Transformers.
- `SOURCE_PROVENANCE.json` records extracted symbols, source hashes and line ranges. Historical internal function names are preserved where necessary to avoid changing numerical code; they are not separate public method versions.
- `docs/RELEASE_NOTES.md` distinguishes extracted implementation from newly written packaging adapters.

**License selection and provenance review are pending.** This staging package does not grant an assumed MIT/Apache license to third-party-derived code. Before publication, the authors must verify redistribution rights, retain upstream notices, select an appropriate license for code they own, and add the finalized paper citation. See [third-party notices](THIRD_PARTY_NOTICES.md).
