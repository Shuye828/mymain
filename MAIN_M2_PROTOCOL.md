# Main M2 Protocol — Learned Disease-Axis Alignment

## Status and scope

This document is the pre-implementation protocol for Main M2 under
`EXPERIMENT_PLAN_MAIN_FIRST_AXIS_ALIGNMENT.md`, SHA-256
`a1c238c7b21b6637efd14084097579de677ae6252cda43adc39ce3177a76477a`.
It becomes frozen when committed before M2 code changes.

M2 trains and selects an AFDB source-only axis-aligned model. It does not read
CPSC2021, LTAFDB, or SHDB signals or labels, perform the formal three-target
test, implement M3/M4, add late baselines, or modify any frozen Stage
1–5D/R1/R2/R3/M1 artifact. Formal target scoring begins only in a separately
authorized M3 after the M2 final checkpoint and all source-only selection
artifacts are frozen.

## Frozen inputs and reference baseline

M2 reuses without modification:

- AFDB index `data/index/afdb_windows.csv`, SHA-256
  `f6528265320183418b001e14e9ece11e6128263d8b55114e1e37412f984a5d35`;
- five-fold subject manifest
  `configs/experiments/afdb_source_oof_folds_v1.csv`, SHA-256
  `78b9615ab28bb0dbbbfedc41943119c2e53cfb3fed35853707250ea2a8bb07ba`;
- R2 config `configs/experiments/source_afdb_r2.json`, SHA-256
  `f63c6712b3d1764976cedbebffd6ec0cbf6d456a2601e7453a415d3e233e1fd3`;
- R2 CE OOF archive, SHA-256
  `1b3b68543417ae255fddbbb4419c6872c615d0edba25e903073a728924334c2a`;
- R2 CE optimized OOF reference: AUROC `0.9798513055`, AUPRC
  `0.9675388908`, BACC `0.9431701555`, and raw-logit threshold
  `-0.1451964676`.

The R2 CE reference is not retrained as lambda zero. Every M2 candidate uses
the same architecture, preprocessing, subject folds, 500-window
per-subject/class cap, subject-class-balanced sampler, optimizer, learning
rate, batch sizes, maximum 100 epochs, patience 10, and seed 42 as R2.

## Model, source axis, and loss

For each checkpoint, define the normalized binary-head direction

```text
d_head = normalize(w_AF - w_nonAF).
```

For the current fold's capped training cohort, extract the unprojected pooled
embedding `h`, normalize each window as `z = normalize(h)`, and define

```text
c_nonAF = mean(z | y=0)
c_AF    = mean(z | y=1)
d_axis  = normalize(c_AF - c_nonAF).
```

Only AFDB source labels may enter these prototypes. `d_axis` must be finite,
nonzero, detached, and treated as a constant during the epoch. The binary head
loss is

```text
L_axis  = 1 - cosine(d_head, stopgrad(d_axis))
L_total = L_CE + lambda_axis * L_axis.
```

Because the axis is detached, `L_axis` directly updates the classification
head; the backbone continues to update through CE, and the source axis can
change at the next epoch through the changed representation. Bias parameters
do not enter `d_head` but remain trainable through CE.

## Epoch-level axis refresh and optimization

Every candidate/fold starts from scratch with seed 42; no R2 or other lambda
checkpoint is used as a warm start. At the beginning of epoch 1 and every
later epoch:

1. switch the current model to evaluation mode;
2. make one deterministic, unshuffled inference pass over every capped
   training window for that fold;
3. calculate and detach `d_axis` using visible AFDB source labels;
4. record class counts, direction norm, head-axis cosine, and angle;
5. restore training mode and train the epoch with the fixed axis.

Training records CE loss, axis loss, total loss, accuracy, gradient norm, and
start/end head-axis geometry. Validation is CE-only and uses every held-out
window. As in R2, validation Macro-F1 at probability threshold 0.5 controls
best-checkpoint selection and patience-10 early stopping. This checkpoint rule
is not changed by target or post-hoc OOF results.

Formal checkpoints must include the lambda, fold, exact subjects, index/fold
manifest/protocol/config hashes, source-only declaration, epoch axis,
optimizer state for recovery, sampler generator state, history, Git identity,
and environment. A resume is accepted only from the same output directory,
protocol/config, lambda/fold, and code commit; it restores the sampler state
before the next epoch. A fresh formal run refuses a nonempty output directory.

## Candidate grid and OOF construction

The fixed candidates are exactly

```text
lambda_axis = [0.01, 0.05, 0.10, 0.20].
```

All four candidates train all five subject-exclusive folds independently,
for 20 formal fold runs. For each lambda, load each fold's best checkpoint and:

1. recompute its normalized source axis from that fold's capped training
   cohort and verify the stored best-checkpoint geometry;
2. extract raw `logit_AF - logit_nonAF` for every held-out validation window;
3. save identity, source label, fold, and score;
4. prove exact unique coverage of all 83,150 AFDB windows;
5. select one merged OOF threshold by maximum BACC, with ties resolved by
   maximum Macro-F1, closest to zero, then lower threshold, using absolute
   tolerance `1e-12`.

For mechanism selection, calculate the angle between the best checkpoint's
head and recomputed source axis for each fold. The candidate summary uses the
unweighted mean of the five fold angles, so folds rather than window counts
have equal influence. Per-fold angles and their mean/standard deviation are
reported.

## Source-only lambda selection and failure path

A candidate is eligible only when both conditions hold within absolute
tolerance `1e-12`:

```text
candidate OOF AUROC >= R2 CE AUROC - 0.005
candidate optimized OOF BACC >= R2 CE optimized OOF BACC - 0.01.
```

Among eligible candidates, select the smallest unweighted mean five-fold
head-axis angle. Angles tied within `1e-12` degrees are resolved by smaller
lambda. AUPRC and all operating metrics are reported but do not add a hidden
selection criterion.

If no lambda is eligible, freeze `NO_ELIGIBLE_LAMBDA`, retain every failed
candidate, and stop M2 before final training. Target data cannot repair the
grid or rule. Any later variant requires a new pre-registered protocol.

The selection artifact must include every metric, threshold, eligibility
margin, fold angle, archive/checkpoint hash, explicit
`target_data_accessed=false`, and the code/environment provenance. It must be
frozen before final training.

## Final epoch rule and seed-42 model

For the selected lambda, take the integer median of its five best epochs. This
is the same source-only aggregation rule used by R2. Train a fresh model from
scratch on the deterministic capped cohort of all 23 AFDB subjects for exactly
that many epochs, seed 42, using the same epoch-level axis refresh and loss.
There is no validation, early stopping, target access, or checkpoint choice in
the full-source run.

After the last epoch, recompute the final source axis on all 18,319 capped AFDB
windows and freeze the final head-axis cosine/angle, checkpoint hash, training
history, selected lambda, OOF threshold, epoch rule, source hashes, and
`target_data_accessed=false`. M2 ends at this artifact. M3 must reuse this
single checkpoint and source-only raw-logit threshold unchanged for all three
targets.

## Diagnostics, tests, and artifacts

Diagnostics require an explicit temporary output override, a balanced
two-class source subset, and `formal=false`; they cannot be finalized or used
for selection. Before formal training, required gates are:

- unit tests for axis construction, loss gradient scope, selection guardrails,
  tie rules, final epoch aggregation, resume validation, and label/schema
  boundaries;
- full repository regression tests;
- real AFDB MPS smoke covering axis refresh, one optimization step,
  validation, atomic checkpointing, and resume;
- clean committed code/config and verified input hashes.

Formal outputs are new files under `outputs/main_m2_axis_alignment/`:

```text
folds/lambda_0p01..0p20/fold_0..4/
oof/lambda_0p01..0p20/
oof/lambda_metrics.csv
oof/selection_artifact.json
oof/final_epoch_rule.json
final_model/seed_42/
```

No pre-existing M2 output may be overwritten. Fold `last.pt` checkpoints are
recovery artifacts; best checkpoints, histories, result manifests, OOF
archives, selection artifacts, and the final checkpoint are retained. Before
M2 completion, verify all hashes, exact OOF coverage, finite scores/directions,
source-only provenance, optimization stability, tests, report, TODO, commit,
and push. No M2 target metric may exist.
