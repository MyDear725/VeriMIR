# strict_v2 evaluation

The selected evaluation split is used as both query and gallery, with each query physically removed from its own candidate list. Full-gallery retrieval is class-unfiltered. Stable descending similarity ranking is used.

- **mAP:** trapezoidal average precision over all same-class relevant images, then macro-average over eligible queries (query average, not equal class weighting).
- **R@K:** binary hit rate: at least one same-class item in the first K results. This is not the fraction of all relevant items retrieved.
- **P@K:** same-class hits divided by the available cutoff `min(K, gallery_size_without_self)`.
- Queries without any other same-class gallery image are excluded from the metric average; `num_queries` and `num_valid_queries` are reported separately.
- P@1 equals R@1 under this definition. P@5 need not be smaller than P@1 after averaging different ranked positions.

The image similarity function has no label argument. Dataset labels are retained separately for metric evaluation. Train-zscore calibration is obtained from deterministic training descriptors, never fit to test labels or test performance.

## Fair reporting

Use validation for checkpoint/fusion selection and freeze the final configuration before test. Per-run test-once discipline does not eliminate selection bias from looking at many versions' test results. Disclose the actual development protocol.

Label-constructed hard-negative galleries (H0–H100) are a different controlled protocol; they are deliberately not included here. Those scores must not be presented as full-gallery strict_v2 results.

Medical dataset splits must also be audited for duplicate images, patient/lesion overlap and near-duplicate acquisition sequences. Disjoint filenames alone are insufficient.
