# Main M1 Protocol — Direct Disease-Axis Utilization

## Status and scope

This protocol freezes Main M1 before implementation. The governing plan is
`EXPERIMENT_PLAN_MAIN_FIRST_AXIS_ALIGNMENT.md`, SHA-256
`a1c238c7b21b6637efd14084097579de677ae6252cda43adc39ce3177a76477a`.
M1 does not retrain a backbone, implement M2, add late baselines, or modify any
historical Stage/R1/R2/R3 artifact.

## Source-only OOF construction

The five frozen R2 fold checkpoints and subject assignments are used. For fold
`k`:

1. load only the capped AFDB training subjects to verify the frozen prototype
   direction and load every held-out validation window;
2. extract `z = normalize(h)` for each held-out window with fold checkpoint
   `k`;
3. define `d_head,k = normalize(w_AF - w_nonAF)` and use the R2 frozen
   `d_proto,k` from that fold's capped training cohort;
4. for each candidate alpha construct
   `d_alpha,k = normalize((1-alpha)d_head,k + alpha d_proto,k)`;
5. save `score_alpha = z dot d_alpha,k` with identity, fold, and AFDB source
   label.

The fixed alpha candidates are exactly `[0.00, 0.25, 0.50, 0.75, 1.00]`.
The resulting OOF archive must cover all 83,150 AFDB index windows exactly
once, with subject-exclusive fold identity. The existing R2 scalar head and
prototype scores are provenance references only: they cannot replace the new
normalized-feature scores because the trained head consumes unnormalized
features and includes bias.

## Alpha and threshold selection

For every alpha, select its threshold from AFDB OOF labels by maximum balanced
accuracy. Threshold ties are resolved by maximum macro-F1, then closest to
fixed threshold zero, then lower threshold. Floating comparisons use absolute
tolerance `1e-12`.

Select alpha as follows:

1. find the maximum AFDB OOF AUROC;
2. retain candidates no more than `0.005` below that maximum, including the
   boundary within tolerance `1e-12`;
3. among retained candidates choose the largest threshold-optimized OOF BACC;
4. if BACC remains tied within `1e-12`, choose the larger alpha.

The selection artifact, all score/archive hashes, and the explicit statement
`target_data_accessed=false` must be frozen before target extraction. A failed
or endpoint selection is retained; target results cannot change the rule.

## Final source direction

Use the frozen R2 full-source `M_AFDB` seed-42 checkpoint. Independently load
the deterministic AFDB source cohort capped at 500 windows per subject/class,
extract normalized features, and compute `d_proto,final` from source labels.
Extract `d_head,final` from the same checkpoint and construct the selected
`d_alpha,final`. No R3 target-derived direction may enter this artifact.

## Target extraction and evaluation

The three formal targets and indices are:

- CPSC2021 `target_split=evaluation` — 82,535 windows;
- `LTAFDB-clean1h-v1` `target_split=evaluation` — 304,873 windows;
- SHDB-AF `target_split=evaluation` — 410,355 windows.

One final checkpoint, selected alpha, final direction, and AFDB OOF threshold
are applied unchanged to all targets. Target extraction uses
`load_unlabeled_target_rows` and `ECGWindowDataset(expose_label=False)`. The
frozen target archive may contain identity and alpha scores but no label or
rhythm fields. Formal labels are joined only after score/artifact hashes and
full evaluation-split coverage are verified.

Report every alpha as a predeclared dose-response analysis, with the selected
alpha as M1 primary and alpha 0/1 as endpoint sanity checks. Alpha 0 is a
normalized head-direction score and is not the historical raw-logit H0/H1
baseline.

## Metrics and M1 decision

For each target and alpha report AUROC, AUPRC, balanced accuracy, macro-F1,
MCC, sensitivity, specificity, precision, and accuracy. The operating metrics
use that alpha's frozen AFDB OOF threshold.

Compare selected alpha with alpha 0. Strong success requires increased mean
target AUROC and AUPRC, improvement in at least two of mean BACC/macro-F1/MCC,
and consistent improvement on at least two targets. Partial success means
ranking improves but operating changes are small. Failure means mean AUROC and
AUPRC decrease and operating metrics decline on most targets. Results are
reported without target-driven repair.

## Required gates and artifacts

Before formal extraction: unit tests, full regression tests, real AFDB fold and
target MPS smoke, clean committed code/config, checkpoint/index/protocol hash
review. Before completion: exact OOF and target coverage, score finiteness,
archive label-schema audit, all artifact hashes, clean Git provenance, report,
TODO update, and a separate documentation commit.

Formal outputs are new files under `outputs/main_m1_axis_interpolation/` with
separate `oof/`, `final_axis/`, and `targets/` directories. No pre-existing
output may be overwritten.
