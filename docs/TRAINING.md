# Training integration and reference lifecycle

This release includes train/validation and fixed final-interpolation runners. `training_objective` assembles one full-method batch loss; `run_training` implements its epoch lifecycle; `finalize_checkpoints` prepares the final interpolated artifact. They are a repackaged integration, not the original experiment driver. Do not describe synthetic tests as reproducing paper scores.

## Runnable entry points

Prepare data as described in `data/README.md`, install dependencies and run from the repository root:

```bash
# One full-method arm. Default: 20 epochs, GPU cuda:0, local B/32 weights.
python -m scripts.train --config configs/isic2018_b32.yaml
python -m scripts.train --config configs/kvasir_b32.yaml

# Matched control required for the reference final interpolation.
python -m scripts.train --config configs/isic2018_b32.yaml --arm control --run-dir outputs/isic2018/control

# Re-evaluate the selected arm on validation, without captions or refitting calibration.
python -m scripts.evaluate --checkpoint outputs/isic2018/method/checkpoint_best_image_only.pt --split validation
```

Use a free device with `--device cuda:1`, `--workers 0` for single-process loading, or `--allow-download` to explicitly allow public pretrained-weight downloads. The two training commands above are alternatives, not a concurrent launch script. Input paths in YAML are resolved from the working directory. The model identity must remain B/32.

The runner saves `config.json`, input hashes, `rank_snapshot.json`, `validation_history.json`, `checkpoint_selection.json`, `best_validation_metrics.json`, `checkpoint_best_image_only.pt`, `checkpoint_last.pt`, and `run_status.json`. `checkpoint_last.pt` is a training snapshot, not a supported resume interface. Use a new run directory; existing runs are never overwritten. There is no automatic restart, hyperparameter search or test access.

## Fixed final interpolation

To run both arms and finalization automatically from a fresh output root:

```bash
python -m scripts.train_final --config configs/isic2018_b32.yaml --output-root outputs/isic2018_full
# Alternatively, for the other dataset:
python -m scripts.train_final --config configs/kvasir_b32.yaml --output-root outputs/kvasir_full
```

The pipeline uses one selected device sequentially, trains the control (BDB coefficient 0) and method (0.025) for the configured full epoch count, and passes their validation-selected image-only checkpoints to finalization. It does not alter seeds, the other loss coefficients, or ISV/BDB settings. It refuses an existing root and stops on failure, preserving stage logs. There is no automatic retry or resume. If one stage fails, diagnose first; use a new directory for that failed stage and the standalone finalizer to reuse successful arms, never retrain successful arms blindly.

If the two arms already completed, no retraining is needed. Run:

```bash
python -m scripts.finalize --control outputs/isic2018/control/checkpoint_best_image_only.pt --method outputs/isic2018/method/checkpoint_best_image_only.pt --output-dir outputs/isic2018/final
```

Weights are fixed at **0.25 control + 0.75 method**; the command has no coefficient-search option. It verifies completion, validation selection, matching configs/seeds, manifest/cache provenance and test-clean run ledgers. It checks the status/attempt files as well as the checkpoint's counter, because an older checkpoint can retain a zero counter after its run was tested. The two actual parameter states are interpolated; old calibration statistics are discarded. New score calibration is fitted using **training images only**, then the single final model is measured on image-only validation. Neither caption caches nor test images are loaded in this stage.

The final directory contains `checkpoint_final_image_only.pt`, `ingredients.json`, `finalization.json`, `checkpoint_selection.json`, `validation_result.json`, `best_validation_metrics.json`, config and run status. `finalization.json` locks the final artifact hash and records fixed weights and ingredient hashes. `completed_epochs` in this final directory describes each completed ingredient, not a new round of training. Validation performance is recorded even when it is worse than an ingredient; the implementation does not silently fall back to another model or search new weights.

After the final validation decision, test only the final model with an explicit command (example for the one-command pipeline):

```bash
python -m scripts.evaluate --checkpoint outputs/isic2018_full/final/checkpoint_final_image_only.pt --split test --confirm-test
```

For the standalone finalization example, replace the path with `outputs/isic2018/final/checkpoint_final_image_only.pt`. Do not test ingredient arms first or test several candidate mixtures. The final validation result is already saved, so a separate validation-evaluation command on that directory is unnecessary and refused to avoid overwriting it.

The evaluator checks completion, validation selection, manifest hash and (for interpolated artifacts) the finalization lock, loads no caption cache, and creates an exclusive `test_attempt.json` before test image access. Failed attempts remain consumed for diagnosis rather than automatic reruns. `run_status.json` is updated to test count 1. These filesystem guards discourage accidental repeats, but cannot prevent users from copying runs or selecting across versions using test scores. Only load trusted `.pt` files; PyTorch deserialization is not safe for arbitrary downloaded files.

## One model, two datasets

Use the same `VeriMIR` B/32 model and functions for both configurations. Class count, random seed, data sources and the all-class P×K batch size differ. The two experiments train separate weights.

## Required lifecycle

1. Prepare fixed split manifests and a **train-only** caption embedding cache from the provided training JSON files using `python -m scripts.cache_train_captions` (see `data/README.md`). Validate disjoint sample/source IDs and check patient/lesion overlap where metadata exists. `TrainCaptionDataset` requires cached caption IDs to equal train IDs exactly. Its samples have no `split` string by default; the caller must mark verified training batches with `batch['split']='train'` before using the integration helper.
2. Load the B/32 encoder, two 512-dimensional image projection heads and training-only appearance classifier. The semantic projection is applied to image features, not to caption embeddings. Text embeddings are fixed privileged targets.
3. Compute caption class prototypes from the entire training cache using `caption_class_prototypes`. The reference target is `shrink_caption_embeddings` with residual scale 0.25. This is target preparation, **not caption self-verification**. The integration helper applies the class-count normalization of the ITC/semantic/neighborhood loss coefficients.
4. Use the P×K training sampler: all classes per batch, two images per class, frozen dataset seed. Standard training augmentation and deterministic evaluation preprocessing are provided in `data.py`.
5. Before optimization, construct the BDB `RankSnapshotBank` from deterministic, clean **training** images and train labels. Compute train-only score calibration, assign snapshot/calibration version 0 and build the sparse teacher. Keep this BDB snapshot frozen during the pulse; never substitute validation/test data.
6. Maintain the ordinary `MemoryBank` separately as required by the base triplet objective. The reference trainer refreshes it from deterministic training features each epoch. It is not the frozen BDB bank.
7. In a batch, compute clean/augmented image branches. ISV uses conservative branch agreement, detached residual allocation `1 - mean(g) + g`, then gradient agreement at the visual projection and two heads. Because caption verification is disabled, the base reliability weights are ones. Base augmented losses are recomputed with the allocated weights before the gradient check.
8. Add BDB with the supplied schedule: epoch 1 coefficient 0.025, epoch 2 coefficient 0.0125, epochs 3–20 coefficient zero. Top-k=5, candidate boundary ranks 6–10, margin=0.02. Adaptive incoming weighting uses the exact config values. Teacher/gates are detached.
9. Keep the rest of the reference training recipe: 20 epochs, base learning rate 1e-5, head multiplier 4, weight decay 1e-4, LR milestone 4 with gamma 0.2, gradient clip 5, AMP, appearance classification active for the first 4 epochs. Preserve optimizer parameter groups and RNG/augmentation streams for reproducibility.
10. The runner uses epoch EMA (decay 0.9, starting at epoch 1), evaluates online/EMA candidates on image-only validation and selects using strict_v2 mAP. Train-zscore calibration is recomputed from training images for each candidate. Exact ties select online, and an equal best score retains the earlier epoch. These operations are in `runner.py`, not in the one-batch helper.
11. For the supporting final interpolation, independently train the matched control using the same recipe but BDB coefficient zero. Interpolate 0.25 control + 0.75 method via `build_boundary_anchored_soup`. Both ingredients must be validation-selected, image-only and test-clean. Recompute calibration on train and evaluate the resulting single checkpoint on validation before locking test access. Interpolating weights is not averaging test predictions.
12. Finally run one image-only test of the locked checkpoint. Do not use ground-truth labels to filter the primary gallery or choose test-time fusion settings.

## Integration API

```python
loss, diagnostics = training_objective(
    model, batch, config, epoch, caption_prototypes,
    rank_bank=frozen_train_snapshot,
    teacher=frozen_train_teacher,
    triplet_bank=ordinary_train_memory,
)
# The caller handles autocast, scaler, loss.backward(), clipping and optimizer.step().
```

`rank_bank` and `teacher` can be omitted only outside active BDB pulse epochs. Train-bank provenance must be validated by the caller; merely setting a dictionary field is not proof of split correctness.

Some optional loss implementations remain because they are dependencies of the verbatim `CompositeLoss` class. Their corresponding options are inactive in both supplied reference configs. Their presence does not add them to the released method; do not enable them while claiming the reference configuration.
