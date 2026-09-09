# Packaging validation

Validation performed on 2026-09-09 using a CPU-only invocation in a separate staging directory. No formal training or dataset evaluation was run.

| Check | Outcome |
|---|---|
| Python syntax compilation | Passed |
| Unit tests | 53 passed in 5.54s (including train-caption, lifecycle and finalization tests) |
| Synthetic ISV+BDB objective | Finite |
| Synthetic backward pass | Passed |
| Three-epoch synthetic lifecycle | Passed; optimizer updates, train memory, EMA, validation selection and checkpoint export |
| Real dataset adapters on generated PNGs | Passed; no test images exist or are opened during training |
| Explicit one-shot test guard | Passed on synthetic tensors only; omitted confirmation and repeated attempts rejected |
| Training/evaluation CLI help | Both entry points passed |
| Independent image-only export | Equal initial weights; independent parameter/buffer storage and mode changes |
| Fixed final interpolation | All floating-point weights equal 0.25 control + 0.75 method, checked against FP64 accumulation |
| Final score calibration | Recomputed from train images; matches direct recomputation |
| Actual finalization loaders | Generated PNGs only; succeeds with caption cache unavailable and test images nonexistent |
| Invalid ingredients / lock | Used-test, unfinished, mismatched seed/cache, swapped roles, caption input and modified lock rejected |
| Full pipeline orchestration | Both arms and finalization execute on synthetic data; no automatic test |
| Finalization/pipeline CLI help | Both new entry points passed |
| B/32 adapter | Mocked pretrained-tower interface test passed |
| Two dataset configs | Same method after excluding data/seed/class-dependent fields |
| Source privacy pattern scan | No private host/user/path/token pattern matches |
| Training-caption JSON/cache provenance | Both original caption file hashes match the reference B/32 cache metadata |
| Training-caption membership | 10,015 ISIC-2018 / 5,600 Kvasir; zero validation/test ID or source-ID overlap |

Caption audit scans are automated checks, not a comprehensive privacy or copyright clearance. Kvasir's original caption header contains a historical manifest hash; exact record correspondence to the current training split was separately verified and recorded in `CAPTION_AUDIT.json`.

Commands:

```bash
python -m pytest -q -p no:cacheprovider
python -m examples.smoke_core
python -m scripts.train --help
python -m scripts.evaluate --help
python -m scripts.finalize --help
python -m scripts.train_final --help
```

Test environment: Python 3.13.2, PyTorch 2.10.0+cu128 (CUDA disabled for these checks), Transformers 4.39.3. The GitHub Actions workflow is included but has not yet run on GitHub. Broad dependency ranges in requirements.txt do not imply that every possible dependency combination has been tested.

Limitations: no new full CLIP training, no end-to-end benchmark reproduction, no accuracy certification for a repackaged trainer, and no completed third-party copyright/license clearance. The B/32 backbone adapter test uses a mock to avoid downloading weights; it is not a full pretrained-model test.
