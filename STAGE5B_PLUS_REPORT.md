# Stage 5B+ Report: Source Head vs Shared Cross-Dataset Disease Axis

## Purpose

Stage 5B+ follows `EXPERIMENT_PLAN_UPDATE_AFTER_STAGE5B.md` and asks whether
the source-trained linear head is as closely aligned with the four-dataset
disease geometry as the four labelled prototype directions are with each
other.

This is a read-only post-hoc mechanism analysis. It reuses the frozen Stage 5A
head directions and Stage 5B direction summaries. No ECG is reread, no model is
trained or adapted, and no new target label is accessed.

## Definitions

Within each frozen reference feature space:

```text
d_head   = normalize(w_AF - w_nonAF)
d_D      = normalize(c_D,AF - c_D,nonAF)
d_shared = normalize(sum_D d_D)
```

All dataset directions retain the fixed non-AF-to-AF orientation; negative
cosines are not sign-flipped or converted to absolute values.

The primary analysis uses the Stage 5B window-weighted directions. The
subject-equal directions form a sensitivity analysis.

## Head-to-dataset direction results

### Window-weighted primary analysis

| Reference space | Dataset direction | Head cosine | Angle |
| --- | --- | ---: | ---: |
| `M_CPSC` | CPSC2021 | 0.858407 | 30.86° |
| `M_CPSC` | LTAFDB | 0.822791 | 34.63° |
| `M_CPSC` | AFDB | 0.860669 | 30.61° |
| `M_CPSC` | SHDB-AF | 0.859668 | 30.72° |
| `M_LTAF` | CPSC2021 | 0.874940 | 28.96° |
| `M_LTAF` | LTAFDB | 0.901255 | 25.68° |
| `M_LTAF` | AFDB | 0.886728 | 27.54° |
| `M_LTAF` | SHDB-AF | 0.872691 | 29.23° |

The source head is not unusually aligned with all disease directions. In the
CPSC feature space its lowest cosine is with LTAFDB. In the LTAF feature space,
the source LTAFDB direction is the closest of the four, but the head remains
about 25.7 degrees away.

## Shared-geometry summary

| Reference space | Mean prototype-prototype cosine | Mean head-to-four cosine | Mean head-to-cross-dataset cosine | Prototype mean minus head mean | Head-to-shared-axis cosine | Shared-axis angle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `M_CPSC` | 0.974736 | 0.850383 | 0.847709 | 0.124352 | 0.858556 | 30.85° |
| `M_LTAF` | 0.989227 | 0.883903 | 0.878119 | 0.105324 | 0.887496 | 27.44° |

The prototype directions cluster much more tightly with each other than either
source head clusters with the four directions. Excluding the source dataset
from the head average slightly increases the gap rather than removing it.

## Subject-equal sensitivity

| Reference space | Mean prototype-prototype cosine | Mean head-to-four cosine | Gap | Head-to-shared-axis cosine | Shared-axis angle |
| --- | ---: | ---: | ---: | ---: | ---: |
| `M_CPSC` | 0.972408 | 0.849328 | 0.123080 | 0.858255 | 30.88° |
| `M_LTAF` | 0.990245 | 0.883125 | 0.107120 | 0.886374 | 27.58° |

The subject-equal analysis changes the primary gaps by less than 0.002. The
finding is therefore not explained by subjects contributing different numbers
of selected windows.

## Interpretation

Stage 5B+ supports the updated geometric narrative:

> Four dataset-specific prototype directions form a compact shared disease-axis
> cluster, while the source-trained linear classification heads are rotated
> away from that cluster by approximately 27–31 degrees.

The result is consistent in two independently trained feature spaces and under
two weighting schemes. It also extends Stage 5A: the discrepancy is not limited
to a head-versus-source-train prototype definition, because the same head is
similarly displaced from prototype directions computed across four datasets.

This does not prove that a prototype score always ranks or classifies better.
Stage 5A already showed dataset-dependent AUROC/AUPRC trade-offs. Stage 5B+
instead establishes a mechanism distinction: cross-dataset prototype geometry
is more internally consistent than the source head is aligned with it.

## Validation and artifacts

- implementation and formal analysis commit: `42efab1` (clean);
- full regression suite: **67 passed**;
- Stage 5A comparison hashes verified for both source heads;
- Stage 5B summary and common-cohort hashes verified for both feature spaces;
- no new target label access;
- the four-panel comparison figure was visually inspected.

New artifacts, added without overwriting Stage 5B outputs:

```text
outputs/stage5b_direction_geometry/head_to_dataset_directions.csv
outputs/stage5b_direction_geometry/head_to_shared_axis_summary.json
outputs/stage5b_direction_geometry/head_vs_shared_axis.png
outputs/stage5b_direction_geometry/stage5b_plus_run_manifest.json
```

## Next stage

Stage 5B+ is complete. The next mandatory stage is **Stage 5D — Disease-Axis
Distribution & Boundary Shift Analysis**. It must reuse the exact Stage 5B
selected-window manifest and quantify class means, variances, gap, overlap,
prevalence, source thresholds, target-oracle thresholds, and boundary drift.
