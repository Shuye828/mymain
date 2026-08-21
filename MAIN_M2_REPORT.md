# Main M2 Report — Learned Disease-Axis Alignment

## Scope and frozen protocol

Main M2 tested source-only training with
`L_total = L_CE + lambda_axis * (1 - cosine(d_head, stopgrad(d_source_axis)))`.
The source axis was recomputed at the start of every epoch from the current
model and that fold's capped AFDB training cohort, then held fixed for the
epoch. Only AFDB labels were used. CPSC2021, LTAFDB-clean1h-v1, and SHDB-AF
signals and labels were inaccessible throughout M2 selection and training.

`MAIN_M2_PROTOCOL.md` was frozen at commit `542704a` before implementation.
The implementation was committed as `8311b0a`; all formal folds and the final
model record that clean commit. The governing Stage 1–5 and R1–R3/M1 artifacts
were not modified or overwritten.

## AFDB five-fold OOF selection

Each lambda used the same frozen subject-disjoint five-fold partition. The OOF
archive for every candidate contains every one of the 83,150 AFDB windows
exactly once and uses the raw score `logit_AF - logit_nonAF`. All archives are
finite and all manifests record `target_data_accessed=false`.

The eligibility floors derived from frozen R2 CE were AUROC `0.974851` and
BACC `0.933170`. Every candidate passed. The protocol then selected the
smallest unweighted mean five-fold head-to-source-axis angle, with smaller
lambda as the final tie break.

| Lambda | OOF AUROC | OOF AUPRC | OOF BACC | OOF Macro-F1 | Mean angle | Frozen threshold | Eligible |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.01 | 0.983847 | 0.964964 | 0.956385 | 0.953666 | 14.421° | 1.275782 | Yes |
| **0.05** | **0.983431** | **0.970783** | **0.962356** | **0.958993** | **6.050°** | **0.290259** | **Yes** |
| 0.10 | 0.984789 | 0.975058 | 0.957484 | 0.954989 | 13.333° | 1.057254 | Yes |
| 0.20 | 0.986362 | 0.977335 | 0.964123 | 0.960272 | 6.622° | -0.840044 | Yes |

Lambda 0.20 had the strongest OOF ranking/operating metrics, but it was not
eligible to replace lambda 0.05 under the frozen selection order because its
mean angle was larger. The selected result was retained without target-driven
repair.

For lambda 0.05, the five best epochs were `[11, 17, 12, 14, 18]`; the fold
angles were `[6.571°, 4.221°, 4.959°, 6.323°, 8.178°]`. The exact integer
median rule therefore fixed final training at 14 epochs.

## Full-source seed-42 model

The final model was freshly initialized and trained for exactly 14 epochs on
the deterministic capped cohort containing all 23 AFDB subjects and 18,319
windows. It used lambda 0.05, with no validation, early stopping, target
access, or post-hoc checkpoint choice.

| Item | Frozen value |
|---|---:|
| Final CE loss | 0.016954 |
| Final axis loss | 0.000471 |
| Final training accuracy | 0.994760 |
| Final source head-axis cosine | 0.997852 |
| Final source head-axis angle | 3.756° |
| AFDB OOF threshold | 0.2902590632 |
| Checkpoint SHA-256 | `e6f74ee2…cdff` |

The optimization objective therefore achieved its intended geometric effect:
the final classifier head is almost collinear with the recomputed AFDB disease
axis. This source-domain result alone does not establish cross-domain benefit;
that question belongs to the separately frozen Main M3 evaluation.

## Verification and artifacts

The M2 implementation gates covered axis construction, gradient scope,
source-only selection, tie rules, final-epoch aggregation, atomic resume,
provenance, and label/schema boundaries. A real MPS resume smoke passed, the
M2 repository regression suite passed 106 tests, and the final repository
suite after adding M3 passed 112 tests.

Authoritative outputs are under `outputs/main_m2_axis_alignment/`:

- `folds/lambda_0p01..0p20/fold_0..4/`
- `oof/lambda_0p01..0p20/`
- `oof/selection_artifact.json`
- `oof/final_epoch_rule.json`
- `final_model/seed_42/final.pt`
- `final_model/seed_42/result.json`

M2 completed successfully as a source-only experiment. It selected lambda
0.05, froze a single raw-logit threshold, and produced one final checkpoint
for all subsequent targets. No M2 target metric exists.
