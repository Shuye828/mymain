# Revision R2 Protocol — AFDB Source Development

## Authority and scope

This implementation protocol instantiates Revision R2 from
`EXPERIMENT_PLAN_REVISION_AFDB_SOURCE.md`. It does not alter frozen Stage 1–5D
results, does not use `LTAFDB-original` as a new source, and does not begin
Stage 6. The new primary labeled source is AFDB; no target dataset or target
label may affect model fitting, epoch selection, threshold selection, seed
selection, or fallback decisions.

## Source cohort

- Source index: `data/index/afdb_windows.csv`.
- Eligible source subjects: all 23 signal-bearing, strictly annotated AFDB
  subjects already present in that index.
- The two official annotation-only AFDB records remain ineligible.
- The historical 16/3/4 source split is preserved for old results but is not
  used by R2 development or final training.
- Training windows retain the deterministic cap of 500 windows per
  subject/class and the subject/class-balanced sampler.
- Validation/OOF prediction uses every accepted window from the held-out
  subjects, with no window cap.

## Five-fold subject-level OOF development

- Number of folds: 5.
- Assignment seed: 42.
- Assignment version: `afdb_source_oof_5fold_v1`.
- Grouping unit: `subject_id`; no subject may cross folds.
- Assignment algorithm: the existing label-independent deterministic subject
  shuffle (`assign_subjects`) with five equal requested ratios and protocol
  namespace `source_oof_5fold_v1`.
- Fold assignment must not inspect rhythm labels, class counts, model outputs,
  target data, or target labels.
- The resulting fold sizes are 5, 5, 5, 4, and 4 subjects. Before training,
  an audit must independently verify that every held-out fold and its
  complement contain both binary classes.

For fold `k`, only subjects outside fold `k` may train the model. Subjects in
fold `k` form the complete validation set and generate that fold's OOF
predictions. Early stopping uses held-out macro-F1 with patience 10. The model,
optimizer, batch sizes, preprocessing, CE loss, and maximum 100 epochs inherit
the frozen Stage 3 CE baseline unless the R2 config explicitly records an
otherwise reviewed value.

Each fold must save its best and last checkpoints, full history, best epoch,
validation metrics, source index hash, fold-manifest hash, config/protocol
hashes, seed, environment, Git identity, subject lists, and exact data counts.
Diagnostic caps require a separate output directory and cannot be accepted as
formal artifacts.

## OOF continuous scores and source thresholds

After the five best checkpoints exist, every AFDB source window must receive
exactly one prediction from a model that did not train on its subject.

Two continuous scores are frozen:

1. `head_logit_difference = logit_AF - logit_nonAF`;
2. `prototype_margin = dot(normalize(feature), d_fold) - midpoint_fold`, where
   `d_fold` and `midpoint_fold` are estimated only from that fold model's
   deterministically capped training subjects.

Fold centering makes the prototype score definition comparable across OOF
models and gives the fixed prototype threshold P0 a value of zero. Head H0 is
also fixed at zero. The combined OOF labels may select:

- `t_AFDB_head*` (H1) from all OOF head scores;
- `t_AFDB_proto*` (P1) from all OOF prototype margins.

Both use the frozen Stage 5C rule: maximize balanced accuracy, then maximize
macro-F1, then choose the threshold closest to the corresponding fixed
threshold, then the lower threshold. OOF labels are source labels and are
allowed for source performance estimation and threshold selection. No target
index may be opened during this process.

The OOF archive must contain labels and full window identity because it is an
explicit source-only labeled artifact. It must also contain fold ID, both
scores, and checkpoint identity. Validation must prove unique full coverage,
no training-subject overlap, finite scores, both classes, and archive/hash
alignment.

## Final full-source models

The final epoch is frozen before full-source fitting as the integer median of
the five best OOF epochs. It cannot be selected from target results.

Using all 23 eligible AFDB subjects and the same cap/balancing protocol, train
one fixed-epoch full-source model for each seed:

- 42 (main model `M_AFDB`);
- 2024 (stability);
- 3407 (stability).

No source validation or target evaluation is performed inside final training,
and no early stopping is used. Each run saves its final checkpoint, training
history, exact epoch count, source/fold/config/protocol hashes, subject list,
seed, environment, and Git identity. Seed 42 is the primary model regardless
of downstream target performance; the other seeds are sensitivity runs and
must not be used to choose a winner.

For later R3 use, each final model may export its all-source prototype
direction and centered prototype definition, but R3 mechanism conclusions are
not part of R2.

## Formal artifact layout

```text
outputs/revision_r2_afdb_source/
  fold_manifest.json
  fold_audit.json
  folds/fold_{0..4}/
    best.pt
    last.pt
    history.json
    result.json
    run_manifest.json
  oof/
    afdb_oof_scores.npz
    fold_score_summaries.json
    thresholds.json
    threshold_curves.csv
    oof_metrics.csv
    run_manifest.json
  final_epoch_rule.json
  full_source/seed_{42,2024,3407}/
    final.pt
    history.json
    result.json
    run_manifest.json
```

## Required gates

Before formal training:

- protocol and fold assignments committed at a clean Git identity;
- unit tests for subject exclusivity, deterministic fold rebuilding, explicit
  subject loading, OOF identity coverage, prototype centering, threshold tie
  rules, and fixed-epoch full-source mode;
- full regression suite;
- real AFDB loader/preprocessing smoke;
- tiny-overfit and bounded real MPS smoke for the R2 fold path;
- fixed-epoch full-source diagnostic smoke in a separate directory.

Before declaring R2 complete:

- five valid formal fold runs and five best checkpoints;
- one unique full-coverage OOF archive;
- frozen H1/P1 thresholds selected only from source OOF labels;
- final epoch artifact written before final training;
- three complete full-source seed runs;
- test, artifact, hash, and Git provenance review;
- `REVISION_R2_REPORT.md` and `TODO.md` update in a separate clean commit.
