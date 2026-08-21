# Main M3 Report — AFDB to Three Targets Formal Test

## Frozen evaluation contract

Main M3 evaluated the single Main M2 seed-42 checkpoint on CPSC2021,
LTAFDB-clean1h-v1, and SHDB-AF. Every target used the same raw score
`logit_AF - logit_nonAF` and the same AFDB OOF threshold `0.2902590632`.
There was no target-specific lambda, threshold, calibration, checkpoint, or
adaptation.

All three full target score archives were generated with hidden labels and
frozen before any target label was read. Only after their hashes, schemas,
finite values, unique identities, and exact coverage all passed did the
evaluation stage join labels and calculate metrics.

| Target | Windows | Score SHA-256 (prefix) | Labels used while scoring |
|---|---:|---|:---:|
| CPSC2021 | 82,535 | `982dc50c…9adcb` | No |
| LTAFDB-clean1h-v1 | 304,873 | `5e8725b3…4c373` | No |
| SHDB-AF | 410,355 | `2e6c32c4…be4bc` | No |
| **Total** | **797,763** | — | **No** |

The formal completion audit status is `PASS`.

## Formal target results

The comparison below uses the already frozen M1 alpha-0 normalized-head
endpoint as the available full-cohort reference. Delta is M2 Ours-Axis minus
that reference.

| Target | AUROC | Δ | AUPRC | Δ | BACC | Δ | Macro-F1 | Δ | MCC | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CPSC2021 | 0.966999 | +0.002298 | 0.895140 | +0.010463 | 0.908073 | -0.013870 | 0.889610 | +0.002293 | 0.784077 | -0.006466 |
| LTAFDB-clean1h-v1 | 0.944869 | -0.038396 | 0.857398 | -0.125854 | 0.871865 | -0.061359 | 0.872867 | -0.059967 | 0.754178 | -0.111851 |
| SHDB-AF | 0.915908 | -0.026371 | 0.599256 | -0.118806 | 0.824429 | -0.071657 | 0.745675 | -0.049522 | 0.528419 | -0.109166 |
| **Three-target mean** | **0.942592** | **-0.020823** | **0.783931** | **-0.078066** | **0.868123** | **-0.048962** | **0.836051** | **-0.035732** | **0.688891** | **-0.075828** |

Secondary operating metrics are retained in `target_metrics.csv`. In brief,
CPSC2021 sensitivity/specificity were 0.914058/0.902089;
LTAFDB-clean1h-v1 were 0.799124/0.944606; and SHDB-AF were
0.804529/0.844330.

## Post-freeze mechanism statistics

These statistics used labels only after score freeze. Histogram overlap used
200 equal-width bins over each target's label-free observed score range.

| Target | Class gap | d-prime | Histogram overlap |
|---|---:|---:|---:|
| CPSC2021 | 13.298656 | 3.123995 | 0.163498 |
| LTAFDB-clean1h-v1 | 9.453378 | 2.571001 | 0.132715 |
| SHDB-AF | 10.510154 | 2.085651 | 0.281348 |

The model retains clear class separation on all three targets. Nevertheless,
large separation by itself did not preserve the frozen reference ranking or
operating metrics on LTAFDB-clean1h-v1 and SHDB-AF. This is consistent with a
representation/distribution-transfer problem rather than a failure to align
the source head geometrically.

## Decision and interpretation

The predeclared success rule is not met. CPSC2021 improved in both ranking
metrics, but only one of three targets showed consistent ranking improvement.
Across the equal-target mean, AUROC, AUPRC, BACC, Macro-F1, and MCC all
declined. The formal Main M3 status is therefore `failure`.

The important scientific result is that successful source geometry alignment
did not generalize uniformly: the final source angle fell to 3.756°, yet
LTAFDB-clean1h-v1 and SHDB-AF transfer worsened materially. Stronger source
head/prototype collinearity is therefore not sufficient for robust AF transfer
under this implementation.

The full-cohort frozen reference is M1's normalized-head endpoint, whereas M2
uses the protocol-required raw-logit difference. The repository does not yet
contain a full-cohort R2 raw-logit three-target archive; R3 raw logits cover
only its mechanism cohort. A later B1 run should add that exact same-pipeline
CE comparator, but it must be reported as a late baseline and cannot alter,
retune, or erase this frozen M2/M3 result.

## Artifacts and next action

Authoritative outputs are under `outputs/main_m3_afdb_three_target/`:

- one `scores.npz`, `score_artifact.json`, and extraction/evaluation manifest
  per target;
- `analysis_result.json`;
- `target_metrics.csv`;
- `mechanism_metrics.csv`;
- `completion_audit.json`.

The completion audit records analysis SHA-256
`7c65b524f153fb4479122e1bd5c96ed9b524299a8739a9ad21f81591490c79ae`.
Under the governing plan's failure gate, the next experiment should not be an
M4 expansion of this axis-alignment model. First add the late, same-pipeline R2
raw-logit comparator (B1) for reporting completeness; then prioritize a newly
pre-registered representation alternative rather than target-guided repair of
M2.
