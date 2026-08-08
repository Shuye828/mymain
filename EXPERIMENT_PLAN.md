# Experiment Plan

## Non-negotiable protocol

The task is window-level strict AF versus non-AF classification. All source
splits are subject independent. Target labels are used only after predictions
are frozen, for final metrics and post-hoc analysis.

Every run records code/version identity, complete configuration, dataset
versions, split manifest, random seed, data counts, exclusions, runtime, and
software environment. Because this directory is not currently a Git work tree,
stage 0 records that limitation rather than inventing a commit identifier.

## Staged implementation

### Stage 0 — environment and repository audit

- Inventory code, dependencies, and data locations.
- Document the original model and immutable data/experiment contracts.
- Implement a read-only WFDB header audit with minimal tests.
- Acceptance: tests pass without real data; real-data status is reported
  honestly; no model training code is added.

### Stage 1 — WFDB input and rhythm annotations

- Implement one adapter per dataset.
- Parse raw rhythm intervals while retaining original tokens.
- Establish dataset-specific subject identity and channel conventions.
- Test normal records, short records, missing annotations, unknown rhythms, and
  transitions using tiny fixtures.
- Acceptance: reviewed per-dataset inventory and rhythm mapping; no windows yet.

### Stage 2 — window index and on-demand dataset

- Build reproducible 10-second non-overlapping indices.
- Add filtering, resampling, channel checks, and lazy segment reads.
- Add subject-leakage assertions and capped balanced sampling.
- Acceptance: `[2, 2000]` finite tensors, deterministic indices, and count
  reports for all available datasets.

Completion record: subject-independent manifests, 1,725,350 strict windows,
on-demand filtering/resampling, capped balanced source sampling, target-label
hiding, leakage checks, deterministic rebuild hashes, and real-data smoke tests
are complete. See `STAGE2_REPORT.md`.

### Stage 3 — source-only baselines

- First reproduce CE-only source training.
- Compare B0 (GCB without CLSA-TTT) and B1/B2 (MedTS-TTT with fixed source
  classifier semantics made explicit).
- Early-stop on source validation macro-F1 only.
- Acceptance: overfit a tiny subset, resume checkpoints, reproduce a fixed-seed
  small run, and never inspect target labels during selection.

### Stage 4 — source embeddings and disease direction

- Add `forward_features`, projection head, and L2 normalization.
- Estimate source train-only class prototypes and normalized disease direction.
- Freeze and export provenance-rich embeddings/prototypes.
- Acceptance: unit-norm projection/direction, no validation/test samples in
  prototype estimates.

### Stage 5 — target boundary reconstruction

- Project unlabelled target embeddings onto the frozen source direction.
- Fit one- and two-component 1-D GMMs with fixed random state,
  `n_init=20`, and `reg_covar=1e-4`.
- Map the higher-mean component to AF without labels.
- Report BIC difference, mean gap, pooled separation, posterior entropy, and
  initialization stability. Record unreliable mixtures without label-assisted
  repair.
- Acceptance: synthetic-mixture tests and explicit transductive versus
  inductive-holdout APIs.

### Stage 6 — representation losses

- Add supervised contrastive loss first, then prototype/center loss as separate
  experiments.
- Acceptance: each loss can be independently disabled and matches a small
  reference calculation.

### Stage 7 — full experiments and statistics

- Complete directed transfers, ablations, seed repeats, confidence intervals,
  and mechanism analysis.
- Freeze the protocol before final target-label evaluation.

## Initial experiment order

1. `cpsc2021 -> shdb-af`
2. `ltafdb -> afdb`
3. Remaining available directed dataset pairs

AFDB is not prioritized as a from-scratch source because of its small number of
usable signal records.

## Baselines and ablations

| ID | Backbone/adaptation |
| --- | --- |
| B0 | Source-only GCB, CLSA-TTT disabled |
| B1 | Source-only MedTS-TTT |
| B2 | MedTS-TTT with fixed source classifier |
| B3 | MedTS-TTT + source disease direction + source fixed threshold |
| B4 | MedTS-TTT + source disease direction + target GMM |
| B5 | GCB + source disease direction + target GMM |
| B6 | MedTS-TTT + SupCon/Prototype + direction + target GMM |

T3A, TENT, and SHOT are deferred until the core pipeline is validated.
Single-channel versus two-channel input is a later data/model ablation.

## Metrics

Primary window metrics: AUROC, AUPRC, accuracy, balanced accuracy, macro-F1,
sensitivity, specificity, precision, MCC, and confusion matrix.

Mechanism metrics: true source/target disease-direction cosine (post-hoc only),
class-center drift, source/GMM threshold difference, GMM BIC difference,
separation, posterior entropy, and pre/post-TTT embedding stability.

## Reproducibility checkpoints

At each stage:

1. Run the smallest meaningful unit/smoke tests.
2. Save commands and observed results in `TODO.md` or a stage report.
3. Update contracts before changing semantics.
4. Stop at the stage boundary for review before expanding scope.
