# Revision R3 Report — `M_AFDB` Mechanism Revalidation

## Scope and execution

Revision R3 completed the required A–D mechanism analyses in the frozen
seed-42 `M_AFDB` feature space. It did not retrain or select a model, tune an
adaptation method, or modify any Stage 1–5D/R1/R2 artifact. The primary
checkpoint SHA-256 is
`0bd3bae8240ddf07fb6d0b3c194e1ca782d37c9cb92a698b13abff570409916d`.

The deterministic shared cohort contains 199,104 windows:

| Dataset | Subjects | non-AF | AF | Total |
|---|---:|---:|---:|---:|
| AFDB | 23 | 10,127 | 8,192 | 18,319 |
| CPSC2021 | 105 | 31,953 | 17,984 | 49,937 |
| LTAFDB-clean1h-v1 | 84 | 24,206 | 34,125 | 58,331 |
| SHDB-AF | 93 | 45,328 | 27,189 | 72,517 |

The MPS extractor completed in 879.6 seconds at clean commit `9a9a82e`. It
loaded the manifest without parsing label/rhythm fields, used
`expose_label=False`, and froze only identity, normalized feature, and score
arrays. Labels were joined only after the archive hash was verified. The
selection and archive hashes are respectively
`0f50d48e83ffb88329edc339b07e7fb8bd4cf5072d75f2b3959e9865b7675818`
and `7fc0130df38e8a7983971a694c852822240b14397c3fc57b1f4fa64adbd7c89f`.

## R3-A — AFDB prototype versus linear head

The final-model AFDB prototype direction and binary-head direction have cosine
0.8879 and angle 27.39 degrees. Their score relationship is strong but does not
pass the frozen equivalence rule (cosine >0.95 and every dataset Spearman
>0.98).

| Dataset/cohort | Pearson | Spearman | Prototype AUROC/AUPRC | Head AUROC/AUPRC |
|---|---:|---:|---:|---:|
| AFDB final-model cohort | 0.9907 | 0.8794 | 0.9992 / 0.9989 | 0.9996 / 0.9995 |
| AFDB R2 OOF | 0.9733 | 0.9732 | 0.9777 / 0.9538 | 0.9799 / 0.9675 |
| CPSC2021 | 0.9847 | 0.9630 | 0.9680 / 0.9199 | 0.9698 / 0.9303 |
| LTAFDB-clean1h-v1 | 0.9826 | 0.9700 | 0.9785 / 0.9854 | 0.9710 / 0.9781 |
| SHDB-AF | 0.9872 | 0.9678 | 0.9678 / 0.9215 | 0.9658 / 0.9131 |

The almost-perfect final-model AFDB ranking is an in-sample mechanism result,
not a generalization estimate. The fold-specific R2 OOF row is the unbiased
AFDB source estimate. Together the results reproduce the historical finding:
prototype and head scores are closely related but geometrically non-equivalent.

## R3-B — four-dataset disease geometry

The disease directions remain exceptionally aligned even though absolute
dataset centroids differ.

| Estimand | Direction cosine range (six pairs) | Centroid distance range |
|---|---:|---:|
| Window weighted | 0.9906–0.9964 | 0.0983–0.3737 |
| Subject equal | 0.9921–0.9973 | 0.1109–0.4526 |

AFDB-to-target window-weighted direction cosines are 0.9964 (CPSC2021),
0.9953 (LTAFDB-clean1h), and 0.9909 (SHDB-AF). Subject-equal results preserve
the conclusion. Thus the relative AF-minus-non-AF direction transfers much
more consistently than absolute domain position in `M_AFDB` space.

## R3-C — AFDB head versus shared disease axis

For the window-weighted analysis, the mean prototype–prototype cosine is
0.9934, whereas mean head-to-prototype cosine is 0.8821 (0.8801 when only the
three cross-dataset directions are included). The AFDB head has cosine 0.8843
and angle 27.84 degrees to the four-direction shared axis. Subject-equal values
are almost identical: 0.9945 prototype mean, 0.8835 head mean, and 27.71 degrees
to the shared axis.

This strongly reproduces the Stage 5B+ mechanism: cross-dataset disease
directions form a compact shared geometry while the supervised classification
head is rotated away from that shared axis.

## R3-D — distribution and boundary shift

The fixed axis is the window-weighted AFDB prototype direction. P1 is the
source-only R2 OOF prototype threshold `-0.0443484`; every target oracle is
post-hoc mechanism analysis only and is prohibited from adaptation or model
selection.

| Dataset | Gap ratio | d-prime | Overlap | AUROC | AUPRC | P1 BACC | Oracle BACC | Headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AFDB | 1.0000 | 11.852 | 0.0109 | 0.9992 | 0.9989 | 0.9932 | 0.9941 | 0.0008 |
| CPSC2021 | 0.7937 | 3.591 | 0.1463 | 0.9680 | 0.9199 | 0.9223 | 0.9265 | 0.0042 |
| LTAFDB-clean1h-v1 | 0.8270 | 3.671 | 0.1513 | 0.9785 | 0.9854 | 0.9225 | 0.9241 | 0.0015 |
| SHDB-AF | 0.8248 | 3.625 | 0.1481 | 0.9678 | 0.9215 | 0.9249 | 0.9257 | 0.0008 |

All targets show wider distributions and moderate gap contraction relative to
the very compact in-sample AFDB cohort, but ranking remains strong. Oracle
thresholds shift positively relative to P1 by 0.147–0.277, yet the maximum
possible BACC gain is only 0.42 percentage points. Therefore threshold drift is
visible geometrically but is not a material operating-point bottleneck under
the frozen gate.

## New Decision Gate

None of the three targets triggers the pre-frozen representation criteria:
AUROC is above 0.90, AUPRC does not drop by 0.10 relative to source, gap ratio
is above 0.50, and overlap does not increase by 0.15. None triggers the boundary
criterion either, because oracle-minus-P1 BACC headroom is below 0.03.

The formal result is therefore `NO_PREDEFINED_CASE_TRIGGERED`, not Case A, B,
or C. The gate is inconclusive about Stage 6 order, and the old Stage 6B-first
decision is not inherited. As an explicitly post-result protocol clarification,
the recommended next action is to run the frozen AFDB-to-three-target benchmark
before spending resources on optional Stage 6A/6B development. This
recommendation must not be presented as a pre-frozen fourth case.

## Verification and artifacts

Formal artifacts are under `outputs/revision_r3_afdb_mechanism/`, including:

- selection manifest/artifact and the label-free extraction archive/manifests;
- `r3a_prototype_vs_head.json`;
- `r3b_four_dataset_geometry.json`, four matrix CSVs, and `r3b_geometry.png`;
- `r3c_head_vs_shared_axis.json`;
- `r3d_axis_distribution_shift.json`, statistics CSV, and distribution figure;
- `decision_gate.json`, `analysis_result.json`, and `run_manifest.json`.

Final validation proved exact unique coverage of all 199,104 selected
identities, a `[199104,128]` finite archive without label fields, feature norms
from 0.99999982 to 1.00000012, the clean1h cutoff, all 15 output hashes, clean
extraction/analysis Git identities, and visual integrity of both figures. The
full regression suite passed **89 tests**.
