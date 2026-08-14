# Main M1 Report — Direct Disease-Axis Utilization

## Scope and frozen protocol

Main M1 tested whether rotating the frozen AFDB classifier direction toward the
AFDB prototype disease direction improves transfer without retraining the
backbone. The governing files are `EXPERIMENT_PLAN_MAIN_FIRST_AXIS_ALIGNMENT.md`
and the pre-implementation `MAIN_M1_PROTOCOL.md`. M1 did not implement M2,
change any Stage 1–5D/R1/R2/R3 result, or use target labels for alpha,
threshold, direction, or model selection.

The five candidates were fixed at `alpha = [0.00, 0.25, 0.50, 0.75, 1.00]`,
where alpha 0 is the normalized linear-head direction and alpha 1 is the
source prototype direction. Every score used `z = normalize(h)` and
`score = z dot d_alpha`; alpha 0 is therefore not the historical raw-logit
H0/H1 baseline.

## AFDB OOF reconstruction and source-only selection

The five frozen R2 fold checkpoints re-extracted all 83,150 held-out AFDB
windows. Each fold's capped training cohort independently reproduced its
frozen prototype direction before its validation features were accepted. The
maximum absolute direction error across folds was `9.26e-10`.

| Fold | OOF windows |
|---:|---:|
| 0 | 17,897 |
| 1 | 18,288 |
| 2 | 17,970 |
| 3 | 14,366 |
| 4 | 14,629 |
| **Total** | **83,150** |

The resulting feature array has shape `[83150, 128]`; every identity occurs
exactly once, every value is finite, and feature norms range from
0.99999988 to 1.00000012. Extraction used MPS for 659.7 seconds at clean
commit `8f6f535`. The OOF archive SHA-256 is
`db37f892446b4c80da5f8921a7841caebafd481152926c73e567a2cac10e7d05`.

| Alpha | OOF AUROC | OOF AUPRC | Optimized BACC | Frozen threshold |
|---:|---:|---:|---:|---:|
| **0.00** | **0.979502** | **0.965239** | **0.943166** | **0.004870** |
| 0.25 | 0.979277 | 0.963913 | 0.942728 | -0.022999 |
| 0.50 | 0.978958 | 0.962057 | 0.942286 | -0.043019 |
| 0.75 | 0.978617 | 0.958847 | 0.941787 | -0.048569 |
| 1.00 | 0.978292 | 0.957177 | 0.941370 | -0.034078 |

All five candidates were within the pre-frozen 0.005 AUROC band. The next
tie-break criterion, maximum source OOF BACC, therefore selected
`alpha = 0.00` and threshold `0.0048703626`. The selection is an allowed
endpoint result and was retained without target-driven repair. Its artifact
SHA-256 is
`250ec67cc6ecd60e6743bb5665dcbf8f18929fdbf282e31096366d7a97a6bb2a`.

The full-source seed-42 checkpoint then used the deterministic 18,319-window
capped AFDB cohort to build the final directions. Final head-to-prototype
cosine is 0.887887. The final-axis artifact SHA-256 is
`28d9dd9e0b047d38cbb1f3f20d7b31f8bf46817c0ae73308302de8b16c8ac862`.

## Label-free target scoring

The same checkpoint and five directions were applied once to the complete
evaluation split of each target. Score archives contained identity plus the
five score fields only. Full coverage, uniqueness, finiteness, schema, and
hashes were verified before labels were joined.

| Target | Evaluation windows | Score SHA-256 (prefix) | Scoring labels accessed |
|---|---:|---|---|
| CPSC2021 | 82,535 | `aa07672c…c615` | No |
| LTAFDB-clean1h-v1 | 304,873 | `02575cf8…8d5a` | No |
| SHDB-AF | 410,355 | `5eb9b363…b584` | No |
| **Total** | **797,763** | — | **No** |

All three extraction manifests record clean commit `749e994`. Labels were
loaded only after the three archives were frozen and the pre-label audit
passed.

## Primary frozen result

Because source-only selection retained alpha 0, the M1 primary method is
identical to the normalized-head comparison endpoint. These are the formal
primary metrics under the source OOF threshold:

| Target | AUROC | AUPRC | BACC | Macro-F1 | MCC | Sens. | Spec. | Precision | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CPSC2021 | 0.964702 | 0.884677 | 0.921944 | 0.887317 | 0.790543 | 0.972617 | 0.871271 | 0.753099 | 0.900418 |
| LTAFDB-clean1h-v1 | 0.983264 | 0.983252 | 0.933224 | 0.932834 | 0.866029 | 0.945054 | 0.921393 | 0.918659 | 0.932854 |
| SHDB-AF | 0.942279 | 0.718062 | 0.896087 | 0.795197 | 0.637585 | 0.936082 | 0.856092 | 0.525728 | 0.867739 |
| **Three-target mean** | **0.963415** | **0.861997** | **0.917085** | **0.871783** | **0.764719** | — | — | — | — |

The formal M1 status is `endpoint_no_axis_utilization`. It is not strong or
partial success: the selected primary method applies no prototype rotation,
so it cannot establish a disease-axis utilization gain. It is also more
specific than the predeclared degradation-only failure category because the
primary and alpha-0 comparator are exactly equal.

## Predeclared dose response

| Alpha | Mean AUROC | Delta | Mean AUPRC | Delta | Mean BACC | Delta | Mean Macro-F1 | Delta | Mean MCC | Delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.00** | **0.963415** | 0 | **0.861997** | 0 | **0.917085** | 0 | **0.871783** | 0 | **0.764719** | 0 |
| 0.25 | 0.963298 | -0.000117 | 0.860761 | -0.001236 | 0.917105 | +0.000020 | 0.870953 | -0.000830 | 0.763823 | -0.000896 |
| 0.50 | 0.963153 | -0.000262 | 0.859264 | -0.002733 | 0.917035 | -0.000050 | 0.870451 | -0.001332 | 0.763220 | -0.001499 |
| 0.75 | 0.962986 | -0.000429 | 0.857523 | -0.004474 | 0.917055 | -0.000030 | 0.870598 | -0.001185 | 0.763403 | -0.001316 |
| 1.00 | 0.962804 | -0.000611 | 0.855579 | -0.006418 | 0.916865 | -0.000220 | 0.871325 | -0.000458 | 0.764057 | -0.000662 |

The aggregate trend hides target heterogeneity. From alpha 0 to 1:

- LTAFDB-clean1h-v1 improves slightly in AUROC (`+0.000601`) and AUPRC
  (`+0.000671`), with essentially unchanged operating metrics.
- CPSC2021 decreases by `0.000245` AUROC and `0.010153` AUPRC.
- SHDB-AF decreases by `0.002189` AUROC and `0.009771` AUPRC, with small
  operating-metric declines.

Thus the source prototype rotation helps one target but does not transfer
consistently, and its three-target mean ranking becomes progressively worse.
The R3 observation that disease directions are geometrically aligned is not,
by itself, sufficient to make direct post-hoc rotation a robust classifier.
This does not contradict R3: M1 evaluates a frozen source-only decision rule on
the full target evaluation splits, whereas R3 was a capped post-hoc mechanism
analysis.

## Verification, artifacts, and next action

Formal outputs are under `outputs/main_m1_axis_interpolation/`. Important
derived files are `analysis_result.json`, `target_metrics.csv`,
`dose_response_means.csv`, `completion_audit.json`, `summary_manifest.json`,
and `m1_dose_response.png`. The completion audit passed every gate and has
SHA-256
`8d965e849f5db8c5b5b874612aad628e25101d6a2165cd546638e777e0881913`.
The frozen analysis result SHA-256 is
`4302a5b51c9090f8a43395a2698cd548a5946435d1890ee1330115a0426822cf`.
The regression suite passed 97 tests, and the dose-response figure was visually
inspected.

The governing plan explicitly permits one M2 learned-alignment experiment even
when M1 has no clear improvement, because post-hoc direction interpolation is
not equivalent to representation/head alignment during source training. The
next task is therefore a narrow M2 implementation audit and protocol freeze:
define source-only axis loss, fold-specific prototype updates, the AFDB OOF
lambda grid and tie-break rule, stopping conditions, artifacts, and tests.
No M2 code or Stage M3 target run should begin before that protocol is reviewed.
