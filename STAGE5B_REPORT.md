# Stage 5B Report: Four-Dataset Direction Geometry

## Question and analysis role

Stage 5B tests the supplement's mechanism hypothesis in two independent frozen
feature spaces:

> ECG datasets may occupy different absolute locations while preserving a
> similar relative AF-minus-non-AF direction.

No model is trained, adapted, selected, or modified. The true labels from all
four datasets are used only for `post-hoc mechanism analysis` to compute class
means. These results are not deployment metrics and are not used to choose an
adaptation rule or threshold.

Direct comparison between directions from separately trained coordinate
systems is prohibited. Consequently, all four directions are first computed
inside `M_CPSC`, and then independently recomputed inside `M_LTAF`.

For every L2-normalized backbone feature `z` in dataset `D`:

```text
c_D,nonAF = mean(z | y=0)
c_D,AF    = mean(z | y=1)
d_D       = normalize(c_D,AF - c_D,nonAF)
m_D       = mean(z)
```

The reported direction matrix contains `cos(d_A, d_B)`. The absolute-domain
matrix contains `L2(m_A, m_B)`.

## Frozen cohort and weighting

The same deterministic selected-window manifest is used in both reference
spaces. To prevent long records from dominating without bound, selection is
capped at 500 windows per subject and class, matching the established source
training cap.

| Dataset | Subjects | non-AF windows | AF windows | Total windows |
| --- | ---: | ---: | ---: | ---: |
| CPSC2021 | 105 | 31,953 | 17,984 | 49,937 |
| LTAFDB | 84 | 24,448 | 34,702 | 59,150 |
| AFDB | 23 | 10,127 | 8,192 | 18,319 |
| SHDB-AF | 93 | 45,328 | 27,189 | 72,517 |
| **Total** | — | **111,856** | **88,067** | **199,923** |

The primary estimand is window-weighted over this selected cohort. A
subject-equal sensitivity analysis averages each subject before the dataset
mean and averages each subject-class mean before forming the disease direction.

Selected-window manifest SHA-256:

```text
bfe2b8da0c5d7a2d26ca90e7b408a44a7d722af4c06ccd7b885b69eaf3557bd6
```

## Primary direction cosine matrices

### Frozen `M_CPSC` feature space

| Dataset | CPSC2021 | LTAFDB | AFDB | SHDB-AF |
| --- | ---: | ---: | ---: | ---: |
| CPSC2021 | 1.0000 | 0.9443 | 0.9945 | 0.9939 |
| LTAFDB | 0.9443 | 1.0000 | 0.9522 | 0.9701 |
| AFDB | 0.9945 | 0.9522 | 1.0000 | 0.9933 |
| SHDB-AF | 0.9939 | 0.9701 | 0.9933 | 1.0000 |

The six off-diagonal cosines range from **0.9443 to 0.9945**, with mean
**0.9747**. Their corresponding angles range from **6.03 to 19.21 degrees**.

### Frozen `M_LTAF` feature space

| Dataset | CPSC2021 | LTAFDB | AFDB | SHDB-AF |
| --- | ---: | ---: | ---: | ---: |
| CPSC2021 | 1.0000 | 0.9841 | 0.9906 | 0.9929 |
| LTAFDB | 0.9841 | 1.0000 | 0.9926 | 0.9867 |
| AFDB | 0.9906 | 0.9926 | 1.0000 | 0.9883 |
| SHDB-AF | 0.9929 | 0.9867 | 0.9883 | 1.0000 |

The six off-diagonal cosines range from **0.9841 to 0.9929**, with mean
**0.9892**. Their corresponding angles range from **6.82 to 10.22 degrees**.

## Primary absolute domain-centroid distances

### Frozen `M_CPSC` feature space

| Dataset | CPSC2021 | LTAFDB | AFDB | SHDB-AF |
| --- | ---: | ---: | ---: | ---: |
| CPSC2021 | 0.0000 | 0.4541 | 0.2162 | 0.1264 |
| LTAFDB | 0.4541 | 0.0000 | 0.3625 | 0.3597 |
| AFDB | 0.2162 | 0.3625 | 0.0000 | 0.2125 |
| SHDB-AF | 0.1264 | 0.3597 | 0.2125 | 0.0000 |

Off-diagonal distances range from **0.1264 to 0.4541**, with mean **0.2886**.
The largest shift is CPSC2021 versus LTAFDB.

### Frozen `M_LTAF` feature space

| Dataset | CPSC2021 | LTAFDB | AFDB | SHDB-AF |
| --- | ---: | ---: | ---: | ---: |
| CPSC2021 | 0.0000 | 0.3203 | 0.2116 | 0.1631 |
| LTAFDB | 0.3203 | 0.0000 | 0.2287 | 0.3078 |
| AFDB | 0.2116 | 0.2287 | 0.0000 | 0.2702 |
| SHDB-AF | 0.1631 | 0.3078 | 0.2702 | 0.0000 |

Off-diagonal distances range from **0.1631 to 0.3203**, with mean **0.2503**.
The four dataset centers are therefore not collapsed into one common point,
even though their disease directions are closely aligned.

## Subject-equal sensitivity analysis

The direction result is robust to removing residual window-count weighting:

| Reference space | Off-diagonal cosine range | Mean cosine | Angle range | Maximum change from primary |
| --- | ---: | ---: | ---: | ---: |
| `M_CPSC` | 0.9426–0.9965 | 0.9724 | 4.76–19.50 degrees | 0.0075 |
| `M_LTAF` | 0.9843–0.9952 | 0.9902 | 5.64–10.16 degrees | 0.0022 |

Subject-equal centroid distances remain nonzero and become larger on average:
0.3366 in `M_CPSC` and 0.3016 in `M_LTAF`. This strengthens the conclusion
that high direction alignment is not an artifact of a few subjects with many
windows.

## Interpretation and limitations

Stage 5B provides strong descriptive support for the proposed geometry:

1. every cross-dataset disease-direction cosine is positive and high in both
   frozen feature spaces;
2. `M_LTAF` produces especially consistent directions, with all six pairwise
   cosines above 0.984;
3. dataset centroids remain separated, and the most separated dataset pair
   depends on the frozen reference space;
4. subject-equal analysis preserves the direction result.

The supported statement is therefore:

> In these two frozen MedTS-TTT representation spaces and this post-hoc cohort,
> absolute dataset location varies while the relative AF-minus-non-AF direction
> is substantially more consistent.

This is a mechanism result, not yet evidence that a new learned direction
improves transfer performance. It has no patient-level confidence intervals,
does not establish a universal biological axis, and should not be used as a
replacement for Stage 5C's fair source-only performance baseline. The current
prototype directions also use true target labels; deployment-time methods may
not do so.

## Validation, runtime, and artifacts

- implementation commit: `65b4d81` (clean for both formal extractions);
- full regression suite: **59 passed**;
- real-data MPS smoke: both reference models x all four adapters passed;
- both formal runs cover exactly the same 199,923 selected windows;
- both checkpoint hashes and all four index hashes were verified;
- both heatmap figures were visually inspected and are legible.

| Reference model | Checkpoint epoch | Device | Formal runtime |
| --- | ---: | --- | ---: |
| CPSC2021 | 3 | MPS | 1,154.8 s |
| LTAFDB | 6 | MPS | 943.2 s |

Authoritative output directory:

```text
outputs/stage5b_direction_geometry/
```

It contains the frozen selected-window manifest and artifact, unified
`analysis_result.json`, per-reference feature summaries and manifests, four CSV
matrices per reference, and one four-panel heatmap per reference.

## Next stage

Stage 5B is complete. The next required task in
`EXPERIMENT_PLAN_SUPPLEMENT.md` is **Stage 5C — Strong Source Baseline**:

1. retain the existing fixed-0.5 source baseline as B0;
2. select `t_source*` using source validation labels only;
3. freeze that threshold and apply it unchanged to every target dataset as B1;
4. record the threshold source and prohibit any target-label access during
   threshold selection.

Stage 6A (CE + SupCon) must not start until Stage 5C is complete.
