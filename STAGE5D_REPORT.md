# Stage 5D Report: Disease-Axis Distribution and Boundary Shift

## Purpose and protocol

Stage 5D follows `EXPERIMENT_PLAN_UPDATE_AFTER_STAGE5B.md`. It tests whether
the highly aligned cross-dataset AF-minus-non-AF directions found in Stage 5B
also imply stable class-conditional score distributions and stable decision
boundaries.

The analysis uses the frozen Stage 4 source disease directions and the exact
Stage 5B selected-window cohort. For each reference model and window,

```text
s = normalize(h) dot d_source
```

was extracted before any label was parsed. The resulting score archives were
then frozen and hashed. Labels were joined only in a separate post-hoc analysis
step. Target oracle thresholds in this report are mechanism diagnostics only;
they are prohibited from adaptation, model selection, or deployment.

## Cohort and safeguards

Both reference models used the same 199,923-window manifest:

| Dataset | Windows | non-AF | AF | AF prevalence |
| --- | ---: | ---: | ---: | ---: |
| CPSC2021 | 49,937 | 31,953 | 17,984 | 0.3601 |
| LTAFDB | 59,150 | 24,448 | 34,702 | 0.5867 |
| AFDB | 18,319 | 10,127 | 8,192 | 0.4472 |
| SHDB-AF | 72,517 | 45,328 | 27,189 | 0.3749 |

- selected-window manifest SHA-256:
  `bfe2b8da0c5d7a2d26ca90e7b408a44a7d722af4c06ccd7b885b69eaf3557bd6`;
- formal extraction commit: `6543c13` with a clean working tree;
- both full extractions ran on MPS;
- score archives contain only dataset/subject/record/window identity and
  `axis_score`; they contain no label field;
- both extraction manifests record `labels_accessed=false`;
- target labels were accessed only after score-archive hash freezing.

## Class-conditional distribution results

### CPSC2021 reference space (`M_CPSC`)

| Dataset | Gap | d-prime | overlap | AUROC | AUPRC | gap/source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPSC2021 (source) | 1.7630 | 7.2682 | 0.0294 | 0.9988 | 0.9981 | 1.0000 |
| LTAFDB | 0.7769 | 1.2246 | 0.5433 | 0.8211 | 0.8637 | 0.4406 |
| AFDB | 0.8627 | 1.2754 | 0.4709 | 0.8439 | 0.8199 | 0.4893 |
| SHDB-AF | 1.4242 | 3.0481 | 0.1831 | 0.9667 | 0.9401 | 0.8078 |

The AF mean remains above the non-AF mean in every dataset, so the disease-axis
orientation transfers. However, LTAFDB and AFDB retain only 44% and 49% of the
source class gap and show much larger class overlap. Thus direction stability
does not eliminate a representation/separation limitation for every transfer.

### LTAFDB reference space (`M_LTAF`)

| Dataset | Gap | d-prime | overlap | AUROC | AUPRC | gap/source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LTAFDB (source) | 1.7213 | 6.7181 | 0.0411 | 0.9972 | 0.9981 | 1.0000 |
| CPSC2021 | 1.1463 | 2.2867 | 0.2485 | 0.9403 | 0.8895 | 0.6660 |
| AFDB | 1.1125 | 2.0266 | 0.2999 | 0.9327 | 0.9285 | 0.6463 |
| SHDB-AF | 1.3667 | 3.0198 | 0.1796 | 0.9611 | 0.9177 | 0.7940 |

All three cross-domain AUROCs exceed 0.93, while their class gaps still shrink
to 65%--79% of the source gap. This is the clearest evidence that useful
ranking can coexist with shifted and broadened target distributions.

## Boundary-shift results

P1 is the source-validation prototype threshold frozen in Stage 5C. The oracle
threshold maximizes target Balanced Accuracy after score freezing and is shown
only to quantify the available boundary-reconstruction headroom.

| Reference | Target | P1 | target oracle | oracle - P1 | P1 BACC | oracle BACC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `M_CPSC` | LTAFDB | -0.1656 | -0.0383 | +0.1273 | 0.7257 | 0.7265 |
| `M_CPSC` | AFDB | -0.1656 | -0.2030 | -0.0374 | 0.7441 | 0.7447 |
| `M_CPSC` | SHDB-AF | -0.1656 | -0.6241 | -0.4585 | 0.8986 | 0.9083 |
| `M_LTAF` | CPSC2021 | +0.2182 | +0.7578 | +0.5396 | 0.8502 | 0.8757 |
| `M_LTAF` | AFDB | +0.2182 | +0.7411 | +0.5229 | 0.8176 | 0.8487 |
| `M_LTAF` | SHDB-AF | +0.2182 | +0.5372 | +0.3190 | 0.9032 | 0.9099 |

The strongest boundary evidence occurs in `M_LTAF`: all three target-optimal
thresholds move in the same positive direction by 0.319--0.540, with post-hoc
Balanced Accuracy headroom of 0.0067--0.0311. `M_CPSC` is heterogeneous:
AFDB and LTAFDB receive almost no oracle operating-point gain, consistent with
their larger ranking/separation limitation, whereas SHDB-AF has a large
threshold displacement and 0.0097 BACC headroom.

The density figures visually confirm that target class means, variances,
overlap, and threshold locations change even though the AF/non-AF orientation
is preserved.

## Decision Gate

Stage 5D supports the central mechanism statement:

> Stable disease direction does not imply a stable decision boundary.

The Decision Gate outcome is **Stage 6B first, with Stage 6A retained as a
secondary enhancement rather than a prerequisite**.

This is a qualified Case B decision. `M_LTAF` provides strong cross-domain
ranking together with systematic threshold drift, so the next main experiment
should be Stage 6B Target Boundary Reconstruction v2. Stage 6A representation
learning remains justified later as an auxiliary ablation because
`M_CPSC -> LTAFDB/AFDB` still has AUROC 0.821--0.844 and substantial overlap.
The present evidence does not support treating boundary reconstruction as the
only remaining problem in every reference/target pair.

The Stage 6B method must remain label-free and must challenge both H1 and P1.
The target oracle values above may not be copied into its configuration or used
to choose a model variant.

## Validation and artifacts

- full regression suite before formal extraction: **72 passed**;
- real MPS smoke: both models, four datasets, 16 windows per dataset;
- formal MPS extraction runtimes: 930.3 s (`M_CPSC`) and 949.1 s (`M_LTAF`);
- both score archives contain exactly 199,923 finite scores in `[-1, 1]`;
- both formal archives passed field, count, device, commit, and hash checks;
- both three-panel density figures were visually inspected.

Primary artifacts:

```text
outputs/stage5d_axis_distribution_shift/analysis_result.json
outputs/stage5d_axis_distribution_shift/distribution_statistics.csv
outputs/stage5d_axis_distribution_shift/run_manifest.json
outputs/stage5d_axis_distribution_shift/cpsc2021/axis_scores.npz
outputs/stage5d_axis_distribution_shift/cpsc2021/analysis_result.json
outputs/stage5d_axis_distribution_shift/cpsc2021/axis_distribution_shift.png
outputs/stage5d_axis_distribution_shift/ltafdb/axis_scores.npz
outputs/stage5d_axis_distribution_shift/ltafdb/analysis_result.json
outputs/stage5d_axis_distribution_shift/ltafdb/axis_distribution_shift.png
```

Patient-level direction bootstrap and label-permutation null remain optional
non-blocking stability supplements under the updated plan. They do not block
the Stage 6B implementation plan.
