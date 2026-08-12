# Revision R3 Implementation Plan — `M_AFDB` Mechanism Revalidation

## Scope and frozen inputs

Revision R3 implements only R3-A–D from
`EXPERIMENT_PLAN_REVISION_AFDB_SOURCE.md`. It does not retrain a model, tune an
adaptation method, enter Stage 6, or modify any historical Stage 1–5D/R1/R2
artifact.

Frozen inputs:

- primary `M_AFDB`: `outputs/revision_r2_afdb_source/full_source/seed_42/final.pt`,
  SHA-256 `0bd3bae8240ddf07fb6d0b3c194e1ca782d37c9cb92a698b13abff570409916d`;
- AFDB OOF scores and thresholds from R2, respectively SHA-256
  `1b3b68543417ae255fddbbb4419c6872c615d0edba25e903073a728924334c2a`
  and `638d75082e0cccc267aa0980ab65920c25c449fbd239e25ad5851c5ee30fb482`;
- AFDB, CPSC2021, `LTAFDB-clean1h-v1`, and SHDB-AF indices with hashes frozen
  in the R3 config. The historical LTAFDB index is prohibited.

## Shared cohort and one-pass extraction

One deterministic labelled post-hoc cohort will be selected with seed 42 and
at most 500 windows per subject/class for each of the four datasets. Selection
uses labels only for the declared mechanism cohort and is frozen before model
extraction.

The formal extractor reads the frozen manifest through a label-hiding parser,
loads every selected ECG with `expose_label=False`, and saves only identities,
L2-normalized 128-dimensional `M_AFDB` backbone features, prototype-axis
scores, and head-logit differences. No label or rhythm field may occur in the
score archive. Labels are joined only after archive and manifest hashes are
verified. This one-pass design makes A–D use exactly the same cohort and avoids
four costly, potentially inconsistent feature extractions.

## R3-A — prototype versus head

- Define `d_proto_AFDB` as the normalized difference between AF and non-AF
  prototypes in the selected AFDB cohort in frozen `M_AFDB` space.
- Define `d_head_AFDB = normalize(w_AF - w_nonAF)` from the seed-42 checkpoint.
- Report cosine, angle, and Pearson/Spearman score correlations on each of the
  four shared-cohort datasets.
- Report AUROC/AUPRC for both scores on all four datasets.
- Separately report the already-frozen R2 fold-specific OOF head/prototype
  correlations and ranking metrics as the unbiased AFDB source estimate. It
  must not be confused with the full-source-model AFDB cohort result.
- Reuse the historical equivalence rule (cosine > 0.95 and minimum Spearman >
  0.98) only as a descriptive mechanism classification, never model selection.

## R3-B — four-dataset geometry

Using the same features and labels, compute window-weighted and subject-equal
AF/non-AF prototypes, disease directions, and dataset centroids. Export 4×4
direction-cosine and centroid-distance matrices for both weightings plus a
four-panel heatmap. All non-AFDB labels are explicitly post-hoc only.

## R3-C — head versus shared axis

For both weightings, construct the normalized sum of the four oriented disease
directions and compare the AFDB head with each dataset direction and the shared
axis. Report mean prototype–prototype cosine versus mean head–prototype cosine,
including the cross-dataset-only head mean.

## R3-D — distribution and boundary shift

Fix the window-weighted AFDB prototype direction. For AFDB and all three
targets report class means/SDs, gap, gap/source ratio, d-prime, histogram
overlap, prevalence, AUROC/AUPRC, P0/P1 performance, post-hoc oracle threshold
and BACC, and boundary headroom. P0=0 and P1 is the R2 OOF prototype threshold.
Target oracle values are analysis-only and are prohibited from Stage 6
configuration, adaptation, seed choice, or model selection.

## Files and artifacts

Planned additions:

- `configs/analysis/revision_r3_afdb_mechanism.json` — frozen inputs, cohort,
  label policy, thresholds, and output contract;
- `src/analysis/revision_r3_afdb_mechanism.py` — shared preparation,
  label-free extraction, verified label join, A–D analysis, plots, and hashes;
- `scripts/run_revision_r3.py` — `prepare`, `extract`, `analyze`, and `all`
  commands;
- `tests/test_revision_r3_afdb_mechanism.py` — protocol, label isolation,
  geometry, OOF identity, threshold, and Decision Gate tests;
- `REVISION_R3_REPORT.md` after the formal run; `TODO.md` only after passing
  final review.

Formal outputs will be new files under
`outputs/revision_r3_afdb_mechanism/`: selection manifest/artifact, label-free
feature-score archive and extraction manifest, R3-A/B/C/D JSON/CSV artifacts,
figures, Decision Gate, and a top-level run manifest.

## Test and execution gates

1. Unit tests for index/hash rejection, clean1h enforcement, deterministic
   selection, hidden-label archive schema, identity join, geometry math,
   score metrics, threshold provenance, and Decision Gate boundaries.
2. Full regression suite.
3. Real-data smoke in a separate diagnostic directory with a small balanced
   subset; diagnostic artifacts cannot be finalized as formal.
4. Commit code/config/tests at a clean Git identity.
5. Formal cohort preparation, one-pass MPS extraction, then CPU finalization.
6. Verify exact cohort coverage, no label fields in the frozen archive,
   finiteness/unit norms, all input/output hashes, and clean provenance.
7. Write report/TODO and commit separately.

## Decision Gate and risks

The new gate is evaluated only after R3-D. Representation-first evidence
includes materially degraded target ranking/separation (especially AUROC below
0.90); boundary-first evidence requires preserved ranking with material P1-to-
oracle operating-point headroom. Mixed targets produce a combined/qualified
case rather than forcing one conclusion.

Primary risks are small-source subject heterogeneity, cohort-label imbalance,
the distinction between fold-specific OOF prototype scores and the final-model
prototype direction, target prevalence effects on AUPRC, and accidental reuse
of the historical LTAFDB index. Each is recorded explicitly rather than
repaired after viewing target results.
