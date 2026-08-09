# Stage 4 Report — Source Embeddings and Disease Direction

## Scope

Stage 4 adds a backward-compatible feature interface and exports source-train
embeddings from the two reviewed Stage 3 best checkpoints. It estimates
non-AF and AF source prototypes and the normalized non-AF-to-AF direction.
No target data, target labels, GMM, SupCon, prototype loss, or model update is
used in this stage.

The Stage 3 checkpoints contain a trained 128-dimensional backbone and
classification head but no trained projection head. Formal Stage 4 artifacts
therefore use L2-normalized 128-dimensional mean-pooled backbone embeddings.
A configurable 64-dimensional linear/two-layer MLP projection head is
implemented and tested for later representation training, but random untrained
projection weights are deliberately excluded from all formal results.

## Implementation

- `MedTS_TTT.py` exposes `forward_features(x)` and optional
  `forward(x, return_features=True)` without adding or renaming parameters.
- `src/models/medts_ttt_wrapper.py` validates the feature and logits contracts.
- `src/models/projection_head.py` implements independently trainable linear or
  two-layer MLP projection with L2-normalized output.
- `src/adaptation/disease_direction.py` computes source-only class prototypes
  with CPU float64 accumulation and rejects hidden/invalid labels.
- `src/representation/source_export.py` strictly loads the reviewed checkpoint,
  verifies its dataset and index hash, reads only `source_split=train`, and
  atomically exports embeddings, labels, and window metadata.
- `scripts/export_source_direction.py` is the CLI entry point. Diagnostic
  truncation requires a separate output directory and cannot overwrite a
  formal export.
- `configs/experiments/source_{cpsc2021,ltafdb}_direction.json` freeze the two
  formal export protocols.

Each feature archive preserves `dataset`, `subject_id`, `record_id`, and
`window_start` for every row. Source prototypes are the arithmetic class means
over every deterministic capped source-training window, each used exactly
once without replacement. The fixed source threshold is the midpoint between
the two prototype projections.

## Backward compatibility and automated checks

- Both reviewed Stage 3 `best.pt` files load strictly with all keys matched.
- For CPSC2021 epoch 3 and LTAFDB epoch 6, the refactored logits and features
  are elementwise identical to the original forward computation.
- Input `[2, 2, 2000]` produces logits `[2, 2]` and features `[2, 128]`.
- `python -m pytest -q` -> **38 passed**.
- `python -m compileall -q MedTS_TTT.py src scripts tests` -> passed.

An initial MPS smoke run exposed a direct MPS-to-CPU-float64 conversion error.
The accumulator now moves features to CPU before float64 conversion, and a
dedicated regression test covers that ordering.

## Formal source exports

Both exports ran outside the sandbox on Apple MPS from clean commit `196b80e`.

| Source | Best checkpoint | Windows (non-AF / AF) | Runtime | Max L2-norm error |
| --- | ---: | ---: | ---: | ---: |
| CPSC2021 | epoch 3 | 35,542 (23,883 / 11,659) | 171.82 s | 1.79e-7 |
| LTAFDB | epoch 6 | 40,332 (15,650 / 24,682) | 171.19 s | 1.79e-7 |

All archive rows match the frozen capped source-training index row-for-row,
all metadata keys are unique, and all feature values are finite.

## Source-direction diagnostics

| Source | non-AF mean | AF mean | Mean gap | Pooled separation | Fixed threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPSC2021 | -0.9314 | 0.9261 | 1.8575 | 20.9622 | -0.002645 |
| LTAFDB | -0.8732 | 0.9187 | 1.7919 | 10.9741 | 0.022731 |

Applying the frozen prototype-midpoint threshold back to the same source
training embeddings gives:

| Source | Accuracy | Balanced accuracy | Macro-F1 | AUROC | AUPRC | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPSC2021 | 0.9982 | 0.9977 | 0.9980 | 0.99994 | 0.99992 | 0.9960 |
| LTAFDB | 0.9886 | 0.9866 | 0.9880 | 0.99965 | 0.99979 | 0.9761 |

CPSC2021 confusion matrix: `[[23865, 18], [45, 11614]]`.
LTAFDB confusion matrix: `[[15297, 353], [105, 24577]]`.

These are in-sample source-training mechanism checks because the same source
training features define the prototypes and direction. They establish
self-consistency only and must not be reported as validation, test, or
cross-dataset performance.

## Artifacts

Authoritative outputs are under:

- `outputs/stage4/source_cpsc2021_direction/`
- `outputs/stage4/source_ltafdb_direction/`

Each directory contains `source_train_features.npz`,
`disease_direction.json`, `run_manifest.json`, and `result.json`.

| Source | Feature archive SHA-256 | Direction SHA-256 |
| --- | --- | --- |
| CPSC2021 | `11dd7b7c085872d65d49580ac399f2bca9b8c5e688b8901627938c946c474715` | `f038358003b274df615072d39f2ec8485ee60b7c001eff24c4dd2999f6a69a33` |
| LTAFDB | `5f7b97f4bd57b6639463d3119f0fe4d5e02d5f54baf0b127265f7f8f66a29340` | `2007d22bc6e8388b4f336ecb793ae90dbd5e85937eb9ad2bc996617f7b199084` |

Stage 4 is complete. Stage 5 should next implement an unlabeled target feature
API and one-/two-component GMM diagnostics, beginning with CPSC2021 to SHDB-AF
and LTAFDB to AFDB. Target labels must remain inaccessible until the frozen
adaptation rule is evaluated.
