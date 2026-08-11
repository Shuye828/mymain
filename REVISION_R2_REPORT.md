# Revision R2 Report — AFDB Source Protocol

## Status

Revision R2 is complete. Five subject-exclusive AFDB OOF folds, the frozen
source-only OOF thresholds/final-epoch rule, and the three fixed-epoch
full-source models all completed on MPS. No target dataset or target label was
opened by this workflow.

The governing protocol is `REVISION_R2_PROTOCOL.md` (SHA-256
`3e3c663176912d3284a2e390476da47dc2d758a60d744324486232d4d9c3c3af`).
The AFDB index and fold-manifest hashes are respectively
`f6528265320183418b001e14e9ece11e6128263d8b55114e1e37412f984a5d35` and
`78b9615ab28bb0dbbbfedc41943119c2e53cfb3fed35853707250ea2a8bb07ba`.

## Five-fold OOF training

All 23 eligible AFDB subjects occur in exactly one validation fold. Training
used the frozen 500-window subject/class cap; validation used every indexed
window for the held-out subjects. Early stopping selected validation macro-F1
with patience 10.

| Fold | Subjects | OOF windows | Best / completed epoch | Macro-F1 | AUROC | AUPRC |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5 | 17,897 | 18 / 28 | 0.9795 | 0.9906 | 0.9907 |
| 1 | 5 | 18,288 | 4 / 14 | 0.9748 | 0.9871 | 0.9884 |
| 2 | 5 | 17,970 | 12 / 22 | 0.9506 | 0.9889 | 0.9873 |
| 3 | 4 | 14,366 | 9 / 19 | 0.9383 | 0.9954 | 0.9607 |
| 4 | 4 | 14,629 | 9 / 19 | 0.7883 | 0.8649 | 0.8291 |

The unweighted fold mean ± sample SD is 0.9263 ± 0.0790 macro-F1 and 0.9654
± 0.0562 AUROC. Fold 4 is a genuine held-out-subject weakness rather than an
incomplete run: its validation set is almost exactly class-balanced (7,315
non-AF and 7,314 AF windows), all 19 epochs and required artifacts are present,
and its best operating point at the fixed classifier threshold has sensitivity
0.9646 but specificity 0.6243. This dispersion must be retained when reporting
AFDB source generalization.

## OOF archive and frozen decisions

The archive contains exactly 83,150 unique AFDB windows: 49,805 non-AF and
33,345 AF. Identity, label, subject-to-fold assignment, and full index coverage
were rechecked row by row; there are no missing, extra, or duplicate windows.

| Score | Threshold | BACC | Macro-F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| Head H0 (fixed) | 0.0000 | 0.9427 | 0.9373 | 0.9799 | 0.9675 |
| Head H1 (OOF) | -0.145196 | 0.9432 | 0.9373 | 0.9799 | 0.9675 |
| Prototype P0 (fixed) | 0.0000 | 0.9409 | 0.9358 | 0.9777 | 0.9538 |
| Prototype P1 (OOF) | -0.044348 | 0.9412 | 0.9354 | 0.9777 | 0.9538 |

H1 and P1 were selected only from source OOF labels by maximum balanced
accuracy with the pre-frozen tie rule. As expected, threshold changes do not
change AUROC or AUPRC. The five best epochs `[18, 4, 12, 9, 9]` give the frozen
integer median `final_epoch = 9`; this artifact existed before any full-source
training was launched.

## Full-source models

Each seed trained on the same 23 subjects and 18,319 deterministically capped
windows for exactly nine epochs. These are training-set diagnostics, not
held-out performance estimates.

| Seed | Final loss | Final train accuracy | Checkpoint SHA-256 |
|---:|---:|---:|---|
| 42 (primary) | 0.02129 | 0.99410 | `0bd3bae8240ddf07fb6d0b3c194e1ca782d37c9cb92a698b13abff570409916d` |
| 2024 | 0.02567 | 0.99269 | `087c7aa9bd4378102487636f1ead2fb362d55fd3410633691ef006b2388a0481` |
| 3407 | 0.02454 | 0.99296 | `44ed6070e8cb217a92284bf9dd28f73fa126df44ec338b87227e23bbfda1938b` |

Final training accuracy is 0.99325 ± 0.00075 across seeds (unweighted mean ±
sample SD). Seed 42 remains the frozen primary `M_AFDB`; the other two seeds
measure optimization stability and must not be selected using target results.

## Artifact and provenance review

Primary artifacts are under `outputs/revision_r2_afdb_source/`:

- `folds/fold_{0..4}/{best.pt,last.pt,history.json,result.json,run_manifest.json}`;
- `oof/afdb_oof_scores.npz`, `thresholds.json`, `threshold_curves.csv`,
  `oof_metrics.json`, `fold_score_summaries.json`, and `run_manifest.json`;
- `final_epoch_rule.json`;
- `full_source/seed_{42,2024,3407}/{final.pt,history.json,result.json,run_manifest.json}`.

The final audit verified archive coverage and finiteness, index labels, fold
assignment, recomputed threshold selection, best/final checkpoint epochs,
checkpoint hashes, nested provenance, and the final-epoch artifact hash. The
full regression command `python -m pytest -q` passed **86 tests** (one benign
joblib physical-core detection warning).

Fold 0–2 manifests record clean commit `fa83474`. Fold 3–4, OOF finalization,
and full-source manifests record commit `536a5cd` with `dirty: true`. The
tracked tree had no changes: the sole status entry was the user's untracked
root-level `data.zip`, which was neither read nor modified by the R2 pipeline.
This limitation is preserved rather than rewriting manifests. The protocol and
fold assignments had already been frozen at clean commits before formal
training, all code/config identities remain attributable, and `/data.zip` is
now explicitly ignored at commit `9efc002` so later runs are not spuriously
marked dirty.

## Interpretation and next step

R2 provides a complete AFDB-source model family and source-only operating
points, while also exposing meaningful between-subject variability. It does
not establish cross-domain performance. The next governed task is Revision R3:
freeze an implementation plan, then re-run prototype-versus-head, four-dataset
geometry, shared-axis, and three-target distribution-shift analyses in
`M_AFDB`, using `LTAFDB-clean1h-v1` and leaving all earlier frozen artifacts
unchanged.
