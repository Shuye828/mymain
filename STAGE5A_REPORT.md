# Stage 5A Report: Prototype Direction vs Binary Linear Head

## Question and frozen definitions

Stage 5A tests whether the Stage 4 prototype disease direction is merely a
re-expression of the frozen binary classification head. It does not train,
update, select, or tune a model and does not modify any Stage 1–5 artifact.

For the frozen mean-pooled backbone feature `h`:

```text
d_proto = normalize(c_AF - c_nonAF)
d_head  = normalize(w_AF - w_nonAF)
prototype_score = normalize(h) dot d_proto
classifier_score = logit_AF - logit_nonAF
```

The classifier score is computed directly from the two logits and includes the
classification-head bias difference. It is not reconstructed from a rounded
softmax probability.

The conclusion rule was frozen before the formal runs:

- direction cosine greater than 0.95; and
- the minimum Spearman correlation across source validation, source test, and
  target evaluation greater than 0.98.

Passing both means `highly_equivalent`; passing only one means
`partially_equivalent`; passing neither means `clearly_different`.

## Leakage separation and validation

Score extraction and label-based evaluation are separate commands. Extraction
uses the target `evaluation` inputs through the hidden-label loader and writes
an archive with no label or rhythm field. The archive and all input index,
checkpoint, and direction hashes are frozen before `finalize` is allowed to
join labels.

Target-label use is explicitly `post-hoc analysis only`. No adaptation-time
target label, threshold choice, direction choice, or model choice occurs.

Validation record:

- implementation commit: `2c48ce0` (clean for both formal extractions);
- `python -m pytest -q` -> **53 passed**;
- both CPSC2021 and LTAFDB real-data MPS extract/finalize smoke paths passed;
- formal score archives contain only identity, scope, and the two scores;
- both formal archives cover every required source validation/test and target
  evaluation window exactly once;
- scatter and histogram figures were visually inspected.

## Formal direction comparison

| Source checkpoint | Direction cosine | Angle | Head direction norm before normalization | Head bias difference | Minimum split Spearman | Frozen conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CPSC2021 | 0.855302 | 31.2069 degrees | 0.697910 | -0.146360 | 0.787293 | clearly different |
| LTAFDB | 0.902464 | 25.5161 degrees | 0.502398 | -0.146021 | 0.945945 | clearly different |

The axes are strongly aligned in a broad geometric sense, but neither is close
enough to the pre-registered 0.95 cosine threshold. They are therefore not the
same linear axis up to scale.

## Score correlations

| Source | Evaluation scope | Windows | Pearson | Spearman |
| --- | --- | ---: | ---: | ---: |
| CPSC2021 | source validation | 23,679 | 0.979966 | 0.864024 |
| CPSC2021 | source test | 29,318 | 0.967821 | 0.989435 |
| CPSC2021 | SHDB-AF evaluation | 410,355 | 0.950682 | 0.787293 |
| LTAFDB | source validation | 97,399 | 0.952405 | 0.945945 |
| LTAFDB | source test | 104,601 | 0.962522 | 0.963172 |
| LTAFDB | AFDB evaluation | 39,638 | 0.942317 | 0.988496 |

Pearson correlations are consistently high, showing that the scores share a
large common signal. Rank agreement is split dependent, however. In particular,
CPSC2021-to-SHDB-AF Spearman falls to 0.7873. The scatter plots show a strongly
nonlinear, compressed relationship near the bounded prototype-score extremes;
the head retains distinctions that the normalized dot product compresses.

## Unified ranking comparison

AUROC and AUPRC use the continuous disease score. No target threshold or GMM
posterior is involved.

| Source | Evaluation scope | Score | AUROC | AUPRC |
| --- | --- | --- | ---: | ---: |
| CPSC2021 | source validation | prototype | 0.999209 | 0.998267 |
| CPSC2021 | source validation | linear head | 0.999303 | 0.998608 |
| CPSC2021 | source test | prototype | 0.997749 | 0.999093 |
| CPSC2021 | source test | linear head | 0.998118 | 0.999228 |
| CPSC2021 | SHDB-AF evaluation | prototype | 0.956959 | **0.801630** |
| CPSC2021 | SHDB-AF evaluation | linear head | **0.957772** | 0.786512 |
| LTAFDB | source validation | prototype | **0.995991** | **0.997894** |
| LTAFDB | source validation | linear head | 0.995774 | 0.997776 |
| LTAFDB | source test | prototype | **0.990553** | **0.983493** |
| LTAFDB | source test | linear head | 0.989775 | 0.980200 |
| LTAFDB | AFDB evaluation | prototype | 0.898232 | 0.876159 |
| LTAFDB | AFDB evaluation | linear head | **0.899813** | **0.879709** |

Target-domain deltas, prototype minus head:

| Transfer | Delta AUROC | Delta AUPRC |
| --- | ---: | ---: |
| CPSC2021 -> SHDB-AF | -0.000813 | +0.015118 |
| LTAFDB -> AFDB | -0.001580 | -0.003550 |

The prototype direction is therefore not uniformly better than the head. It
substantially improves SHDB-AF AUPRC while keeping AUROC nearly unchanged, but
is slightly worse on both AFDB ranking metrics. Two development transfers are
not enough to claim general cross-domain superiority.

## Interpretation

The formal answer to Stage 5A Q1 is:

> The prototype disease direction and the binary linear head are strongly
> related, but they are not mathematically or empirically equivalent under the
> pre-registered rule.

Three differences coexist:

1. the axes differ by about 25–31 degrees;
2. the head operates on the raw feature and includes a bias, while the prototype
   score first L2-normalizes the feature;
3. the resulting rank relationship changes across datasets.

Stage 4's prototype subtraction should not be presented as sufficient novelty
by itself. The Stage 5A result instead supports the supplement's revised goal:
explicitly test whether a learned direction can become more stable across
domains than either the current prototype axis or the source classification
head.

## Runtime and artifacts

| Source | Total scored windows | MPS extraction runtime |
| --- | ---: | ---: |
| CPSC2021 | 463,352 | 1,974.1 s |
| LTAFDB | 241,638 | 1,012.7 s |

Authoritative directories:

- `outputs/stage5a_head_vs_direction/cpsc2021/`
- `outputs/stage5a_head_vs_direction/ltafdb/`

Each contains `direction_comparison.json`, `score_correlation.json`,
`split_metrics.csv`, `scores.npz`, `score_scatter.png`,
`score_histograms.png`, `score_artifact.json`, `run_manifest.json`,
`analysis_result.json`, and `evaluation_manifest.json`.

Verified score hashes:

- CPSC2021 analysis:
  `22759397b8ed1f3f36e4896da4256c5f2d4ce37d56e61df7e124ea1fb0a651db`
- LTAFDB analysis:
  `5856cb12a78ecdd721c1d26857736cb8255f3ee8ac15c4917735fcb5d9ca3332`

## Next stage

Stage 5A is complete. The next task under `EXPERIMENT_PLAN_SUPPLEMENT.md` is
Stage 5B: extract four labelled dataset directions in each of the two frozen
reference-model feature spaces, then compare the 4 x 4 direction-cosine matrix
with the 4 x 4 absolute domain-centroid-distance matrix. All target-label use
remains post-hoc mechanism analysis only, and no model is modified.
