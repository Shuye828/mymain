# Stage 5C Report: Strong Source Threshold Baselines

## Purpose and updated protocol

Stage 5C follows `EXPERIMENT_PLAN_UPDATE_AFTER_STAGE5B.md`. It establishes a
fair source-only operating-point baseline before any further representation or
target-boundary method is developed.

Only the two previously disclosed development transfers are evaluated:

- CPSC2021 -> SHDB-AF;
- LTAFDB -> AFDB.

No other target labels are accessed. The remaining directed transfers stay
frozen for the final benchmark.

## Frozen baseline definitions

For the binary-head score and source prototype score:

```text
q_head  = logit_AF - logit_nonAF
q_proto = normalize(h) dot d_source
```

| ID | Score | Threshold |
| --- | --- | --- |
| H0 | `q_head` | 0, exactly equivalent to `P(AF) >= 0.5` |
| H1 | `q_head` | source-validation argmax Balanced Accuracy |
| P0 | `q_proto` | frozen Stage 4 source-train prototype midpoint |
| P1 | `q_proto` | source-validation argmax Balanced Accuracy |

The deterministic tie rule was frozen before formal selection: maximize source
validation Macro-F1 among equal-BACC thresholds, then choose the threshold
closest to the corresponding fixed baseline, then the numerically lower one.
Neither formal source had a top-BACC tie.

## Label-access separation

The `select` command verifies and reads the frozen Stage 5A score archive, then
joins labels from the source-validation split only. It does not open the target
index. The threshold artifact, source-validation curve, input hashes, and clean
Git identity are frozen before `evaluate` is allowed to run.

Only the subsequent `evaluate` command reads source-test and development-target
evaluation labels. Target adaptation labels are never read. Both selection
manifests record:

```text
target_index_opened = false
target_labels_accessed = false
adaptation_time_target_label_access = prohibited
```

## Selected thresholds

| Source | Baseline | Threshold | Source-val BACC | Source-val Macro-F1 |
| --- | --- | ---: | ---: | ---: |
| CPSC2021 | H0 | 0.000000 | — | — |
| CPSC2021 | H1 | -0.337628 | 0.991623 | 0.989040 |
| CPSC2021 | P0 | -0.002645 | — | — |
| CPSC2021 | P1 | -0.165604 | 0.991496 | 0.988897 |
| LTAFDB | H0 | 0.000000 | — | — |
| LTAFDB | H1 | 0.331191 | 0.962951 | 0.963523 |
| LTAFDB | P0 | 0.022731 | — | — |
| LTAFDB | P1 | 0.218215 | 0.965473 | 0.965220 |

The optimized thresholds move downward for CPSC2021 and upward for LTAFDB.
This source dependence argues against substituting one global empirical offset.

## Source-test verification

Source-validation threshold optimization generalizes consistently to the held
out source test splits:

| Source | Comparison | Delta BACC | Delta Macro-F1 | Delta MCC | Delta accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| CPSC2021 | H1 - H0 | +0.006539 | +0.009958 | +0.015910 | +0.009619 |
| CPSC2021 | P1 - P0 | +0.012457 | +0.018641 | +0.029683 | +0.018078 |
| LTAFDB | H1 - H0 | +0.004227 | +0.010187 | +0.014958 | +0.010162 |
| LTAFDB | P1 - P0 | +0.006711 | +0.015811 | +0.023596 | +0.015640 |

This confirms that H1/P1 are meaningful strong source baselines rather than
source-validation-only numerical artifacts.

## Development-target results

### CPSC2021 -> SHDB-AF

| Baseline | BACC | Macro-F1 | MCC | Sensitivity | Specificity | Accuracy | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 | 0.856763 | 0.832571 | 0.668559 | 0.780545 | 0.932982 | 0.910787 | 0.957772 | 0.786512 |
| H1 | **0.862919** | **0.833047** | **0.671141** | **0.796897** | 0.928940 | 0.909715 | 0.957772 | 0.786512 |
| P0 | 0.850679 | 0.831522 | 0.665256 | 0.764980 | **0.936379** | **0.911423** | 0.956959 | 0.801630 |
| P1 | 0.861268 | 0.831861 | 0.668645 | 0.793700 | 0.928835 | 0.909159 | 0.956959 | 0.801630 |

Threshold optimization shifts the operating point toward greater sensitivity.
H1 improves BACC by 0.006155 and MCC by 0.002581 over H0. P1 improves BACC by
0.010588 and MCC by 0.003390 over P0. The accompanying specificity and accuracy
reductions are reported rather than hidden.

### LTAFDB -> AFDB

| Baseline | BACC | Macro-F1 | MCC | Sensitivity | Specificity | Accuracy | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 | 0.787071 | 0.774070 | 0.568805 | **0.868732** | 0.705409 | 0.774434 | **0.899813** | **0.879709** |
| H1 | 0.787263 | 0.776125 | 0.568221 | 0.855002 | 0.719523 | 0.776780 | **0.899813** | **0.879709** |
| P0 | **0.788860** | 0.776369 | **0.572024** | 0.866762 | 0.710959 | 0.776805 | 0.898232 | 0.876159 |
| P1 | 0.785831 | **0.777764** | 0.564776 | 0.829155 | **0.742506** | **0.779126** | 0.898232 | 0.876159 |

H1 changes AFDB BACC by only +0.000192 and MCC by -0.000584 relative to H0.
P1 raises Macro-F1 and accuracy but lowers BACC by 0.003030 and MCC by
0.007248 relative to P0. Source-validation thresholding is therefore useful
and fair, but it does not uniformly repair the target operating point.

## Ranking invariance

AUROC and AUPRC are exactly identical between H0/H1 and between P0/P1 on every
source-test and target-evaluation split. This is a required invariant because
each pair uses the same continuous score and differs only in threshold.

Consequently, H1 is adopted as the main strong source-head baseline, while H1
and P1 must both be challenged by later boundary-reconstruction methods.

## Interpretation

Stage 5C supports three conclusions:

1. selecting thresholds on source validation is worthwhile and improves all
   tested source-test operating-point metrics;
2. these improvements transfer only partially and inconsistently to the two
   development targets;
3. a strong source threshold is necessary for fair comparison, but is not a
   sufficient solution to cross-domain boundary shift.

The result is consistent with the updated project hypothesis: strong ranking
can coexist with a target-specific change in the appropriate disease-axis
decision boundary.

## Validation and artifacts

- updated protocol commit: `2a994ae`;
- implementation and both formal selections: clean commit `08748a0`;
- full regression suite: **63 passed**;
- fixture isolation test succeeds with a deliberately nonexistent target index;
- real frozen-score select/evaluate smoke passed;
- both formal score, checkpoint, direction, and index hashes passed;
- no final-transfer target label was accessed.

Authoritative directory:

```text
outputs/stage5c_strong_source_baseline/
```

Root artifacts include `thresholds.csv`, `source_validation_curves.csv`,
`target_results.csv`, `analysis_result.json`, and `run_manifest.json`. Each
source subdirectory includes the frozen threshold artifact, selection manifest,
source-validation curve/results, source-test results, target results, analysis
result, and evaluation manifest.

## Next stage under the updated plan

Stage 5C is complete. The next mandatory task is **Stage 5B+ — Shared Disease
Axis vs Source Linear Head**, followed by Stage 5D. Stage 6 must wait for the
post-Stage-5D decision gate.
