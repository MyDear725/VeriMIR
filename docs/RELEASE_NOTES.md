# Core release boundaries

## Extracted reference code

Projection heads and image-only container, feature/gradient verification, BDB loss and teacher construction, supporting base losses, banks, preprocessing/datasets/sampler, two-epoch schedule, checkpoint interpolation and strict_v2 metrics are extracted from the shared B/32 reference code. Function/class bodies and decorators are preserved. Only the module/import structure is reorganized. Symbol-level hashes are available in `SOURCE_PROVENANCE.json`.

## New packaging code

`model.py` is a compact B/32-only adapter; `training.py` assembles the active full-method batch objective; `evaluation.py` exposes label-free scoring; `examples/smoke_core.py` and `tests/test_public_core.py` provide synthetic integration checks. These are newly packaged interfaces, not verbatim copies of a complete training runner. No new formal training/test results are claimed.

Reference numerical tests for feature ISV, gradient ISV and BDB are retained. Historical config-sweep tests are excluded because their experimental configs are not distributed. No medical images or model weights are used by the tests. Caption-file integrity tests inspect the released training JSON files without running a model.

## Training caption addition

The package now includes the exact original ISIC-2018 and Kvasir training-caption JSON files (10,015 and 5,600 entries). The JSON-to-cache B/32 encoding helper and strict train-membership audit are included. Embeddings, evaluation captions and images are still excluded. See `data/captions/README.md` and `docs/CAPTION_AUDIT.json` for provenance, Kvasir's historical manifest-hash difference, label-prefix disclosure and privacy/licensing boundaries.

## Basic runner addition

`verimir/runner.py` and `scripts/train.py` now connect the core to a complete single-arm epoch loop: fixed P×K sampling, AdamW groups, LR schedule, ISV+BDB optimization, frozen train teacher, ordinary memory refresh, epoch EMA, validation mAP selection and image-only checkpoint export. `scripts/evaluate.py` is a separate validation/test entry with explicit test confirmation and per-run attempt bookkeeping. Tests include multi-epoch synthetic training and generated-PNG dataset loading with deliberately nonexistent test images. Neither test uses real medical images or produces paper benchmark results.

The basic single-arm runner has no resume support. Initialization RNG equivalence to the historical full-CLIP trainer and end-to-end numerical reproduction remain unverified. Exact benchmark split manifests must still be supplied separately.

## Independent export and final interpolation revision

- `VeriMIR.export_image_only()` now deep-copies the image-only module container **before** switching it to eval mode. It neither changes the source module modes nor shares parameter/buffer storage. The training saver filters the known image-module state keys directly, avoiding a needless deep copy of an entire GPU model just to enumerate keys.
- `verimir/finalize.py` and `scripts/finalize.py` accept completed, validation-selected control/method checkpoints. They enforce matched configurations and seed, test-clean run records and shared data provenance; interpolate **0.25 control + 0.75 method**; discard old calibration; refit on training images only; evaluate once on validation and write a hash-locked final checkpoint. No caption cache or test images are read. Fixed coefficients are not selected by validation or test results.
- `scripts/train_final.py` automates sequential control/method training and finalization from a new output root. Existing successful arms can instead be used with the standalone finalizer. Neither command automatically runs test.
- The separate evaluator verifies the interpolated checkpoint's finalization lock before allowing explicit test access. The basic per-run one-shot guard is preserved.
- New tests cover export isolation, exact interpolation arithmetic, train-image recalibration, real dataset adapters on generated PNGs without captions at finalization, synthetic full-pipeline orchestration and rejection of incompatible, unfinished, already-tested or unlocked artifacts.

These fixes complete the packaged finalization workflow, not a certification of the paper's numerical results. All new execution checks used synthetic data on CPU; no formal training or held-out dataset evaluation was run.

## Before publishing

- Confirm authorship and rights for inherited source; choose a license only for code you have permission to redistribute.
- Add the final paper title, author list and citation; do not claim conference acceptance before it occurs.
- Keep the repository labeled **core implementation with training/finalization runners**, not a certified benchmark reproduction, until full-data numerical equivalence is independently verified.
- If later adding split files, review dataset terms and avoid patient-identifying metadata.
- If later adding weights, review upstream model and dataset terms and prefer a separately documented release asset.
- Never commit API keys, SSH config, hostnames, private network addresses, absolute experiment paths, unreviewed captions or experiment histories. Only the two audited training-caption JSON files are allowlisted; evaluation captions remain excluded.
