# TODO

## Stage 0

- [x] Read `PROJECT_MASTER_PROMPT.md`, `README.md`, `MedTS_TTT.py`, and
  `benchmark/README.md`.
- [x] Audit repository layout and installed dependencies.
- [x] Verify the model's 250/256/257-token behavior with a local smoke test.
- [x] Document project, dataset, and experiment contracts.
- [x] Add a read-only WFDB header audit and minimal tests.
- [x] Add the proposed base directory structure and ignore rules.
- [x] Install persistent `wfdb==4.1.2` in the active environment.
- [x] Locate and organize all four supplied datasets under `data/raw`.
- [x] Run structural/header audits against the supplied datasets.
- [x] Verify bundled checksums for LTAFDB and SHDB-AF.
- [ ] Resolve/record upstream provenance for the converted CPSC2021 MATLAB set.
- [ ] Decide whether to restore AFDB's official 25-entry `RECORDS` manifest;
  never change it silently.
- [x] Put the local project under Git and record clean commits in new training
  manifests. This establishes local experiment identity; it does not prove an
  upstream commit for the originally supplied files.

### Validation record (2026-07-27)

- `python -m pytest -q tests/test_audit_wfdb_headers.py` -> **4 passed**.
- `python -m py_compile scripts/audit_wfdb_headers.py
  tests/test_audit_wfdb_headers.py` -> **passed**.
- Real-parser smoke test used a temporary `wfdb==4.3.1` installation and four
  generated two-channel, 200 Hz, 10-second WFDB records -> **4/4 complete** in
  `--strict` mode; CSV and JSON output succeeded.
- Repository-path audit against `data/raw` -> **0 headers**; all four canonical
  dataset directories are missing/incomplete.
- Model smoke test: 250 and 256 tokens succeeded; 257 tokens raised the expected
  positional-embedding size mismatch.

## Stage 1 (do not start without review)

- [x] Extend the data audit/adapter contract for CPSC2021 MATLAB v7.3 files;
  do not read `signal_*_processed`.
- [x] Inventory raw rhythm tokens for each available dataset.
- [x] Define and review dataset-specific AF/non-AF/exclusion mappings.
- [x] Define subject-ID extraction for each dataset.
- [x] Implement WFDB signal/annotation adapters with fixtures.
- [x] Validate channel identities, signal lengths, and annotation coverage.

### Stage 1 validation record

- `python -m pytest -q` -> **12 passed**.
- Real-data adapter validation -> **1673/1673 records valid**.
- Raw signal-slice validation -> **1671/1671 signal records valid**; AFDB's
  two remaining records are official annotation-only records.
- Interval coverage validation -> all annotated signal records are bounded,
  ordered, non-overlapping, gap-free, and complete.
- Generated `record_inventory.csv`, `rhythm_inventory.csv`,
  `inventory_summary.json`, and `adapter_validation.json` under
  `outputs/data_audit`.
- No windowing, preprocessing, splitting, caching, or model training was added.

## Stage 2

- [x] Freeze dataset, split, window, preprocessing, and sampling configs.
- [x] Build deterministic patient-level source and target split manifests.
- [x] Assert zero patient leakage and label-independent target assignment.
- [x] Build complete strict 10-second non-overlapping window indices.
- [x] Exclude transitions, unannotated ranges, confusing rhythms, short
  records, and records without usable signal/annotation.
- [x] Implement on-demand 0.5-40 Hz filtering and 200 Hz resampling.
- [x] Return finite `float32 [2, 2000]` tensors without extra z-score.
- [x] Implement deterministic 500-window subject/class source cap.
- [x] Implement subject/class-balanced source sampling.
- [x] Hide target adaptation labels and prohibit class-aware target caps.
- [x] Validate all indices, real samples, class coverage, and deterministic
  rebuild hashes.

### Stage 2 validation record

- Subject manifests: LTAFDB 84, CPSC2021 105, AFDB 23, SHDB-AF 93 eligible
  subjects.
- Complete indices: **1,725,350 windows**.
- Index consistency and patient leakage: **all valid, zero conflicts**.
- Real preprocessing: **16/16 sampled windows valid**.
- Deterministic rebuild: **four window CSV hashes unchanged**.
- Source capped training set: **139,430 windows**, maximum 500 per
  subject/class.
- Unit tests: **23 passed**.
- `MedTS_TTT.py` remains unchanged; no training was started.

## Later stages

- [x] Reproduce CE-only source baseline before representation losses.
- [x] Add backward-compatible feature export and source disease direction
  from both reviewed formal checkpoints.
- [x] Add target GMM with reliability diagnostics.
- [ ] Add SupCon and prototype losses independently.
- [ ] Run ablations and directed cross-dataset experiments.

## Resolved implementation decisions

- Canonical data root: `data/raw/{dataset}`.
- Local CPSC2021 requires an HDF5 adapter; it does not expose `.atr` files.
- SHDB-AF grouping uses `AdditionalData.csv:Subject_ID`.
- Rebuildable audit outputs live under `outputs/data_audit/`.
- Strict mapping is versioned in `configs/datasets/rhythm_mapping.json`.

## Decisions fixed in Stage 2

- CPSC boundary rule: `cpsc_rpeak_half_open_v1`.
- Source split: 70/15/15 by subject, seed 42.
- Target inductive split: 50/50 adaptation/evaluation by subject, seed 42.
- First SHDB protocol: 98 annotated records with hidden adaptation labels;
  30 unannotated records deferred to a later ablation.
- Window endpoints equal to a half-open rhythm boundary are valid; windows that
  cross the boundary are excluded.

## Stage 3 status

- [x] Install and record the formal environment dependencies.
- [x] Establish clean Git identities for reproducible runs.
- [x] Implement source-only CE training, validation macro-F1 early stopping,
  full metrics, provenance manifests, and atomic best/last checkpoints.
- [x] Restore model, optimizer, history, and early-stopping state and continue
  a real-data run from a checkpoint.
- [x] Pass 16-window tiny-overfit gates for CPSC2021 and LTAFDB.
- [x] Complete fixed-seed, class-balanced small runs for both source datasets.
- [x] Run full CPSC2021 training with complete validation and test evaluation
  (best epoch 3; stopped at epoch 13; test macro-F1 0.8775, AUROC 0.9981).
- [x] Run full LTAFDB training with complete validation and test evaluation
  on MPS (best epoch 6; stopped at epoch 16; test macro-F1 0.8778, AUROC
  0.9898).
- [x] Review both formal checkpoints before Stage 4.

See `STAGE3_REPORT.md`. No target data or labels were accepted by the Stage 3
trainer, and `MedTS_TTT.py` remained unchanged during every formal Stage 3
run. Stage 4 subsequently added only a backward-compatible feature interface.

## Stage 4 status

- [x] Add `forward_features` and optional feature return without changing
  checkpoint keys, logits, or CLSA-TTT mathematics.
- [x] Implement and test a 64-dimensional normalized projection head without
  using random untrained projection weights in formal artifacts.
- [x] Export all deterministic capped source-training embeddings with labels
  and per-window metadata for CPSC2021 and LTAFDB.
- [x] Estimate source-only class prototypes, unit disease directions, and
  prototype-midpoint fixed thresholds.
- [x] Verify strict checkpoint compatibility, exact legacy-forward equality,
  index/metadata alignment, finite unit-norm features, and clean provenance.

See `STAGE4_REPORT.md`. At the Stage 4 boundary, target data had not yet been
projected and no target GMM or target-label evaluation had been performed.

## Stage 5 status

- [x] Implement strict target loading that never parses target label fields.
- [x] Add one-/two-component one-dimensional GMM fitting with fixed
  initialization, regularization, and reliability gates.
- [x] Separate label-free fitting from post-freeze label evaluation with
  archive and index hash verification.
- [x] Run both inductive-holdout and transductive protocols for
  LTAFDB -> AFDB and CPSC2021 -> SHDB-AF on MPS.
- [x] Compare frozen B2 source classifier, B3 source direction, and B4 target
  GMM predictions without target-assisted threshold repair.
- [x] Record reliability diagnostics, complete metrics, provenance, hashes,
  and metric-specific limitations.

See `STAGE5_REPORT.md`. Both formal GMM artifacts passed all frozen reliability
criteria. B4 produced operating-point trade-offs rather than an across-metric
win: it improved AFDB inductive accuracy/macro-F1 and SHDB-AF inductive
sensitivity/balanced accuracy, while reducing other metrics. The former plan to
start Stage 6 immediately is superseded by `EXPERIMENT_PLAN_SUPPLEMENT.md`.

## Stage 5A status

- [x] Freeze the post-Stage-5 supplement and its equivalence thresholds before
  formal analysis.
- [x] Extract `d_head = normalize(w_AF - w_nonAF)` from both reviewed source
  checkpoints and compare it with the frozen Stage 4 prototype direction.
- [x] Freeze exact prototype scores and classifier logit differences for source
  validation, source test, and the available target evaluation split.
- [x] Compute direction cosine/angle and per-split Pearson/Spearman without
  target labels.
- [x] Join labels only after score/hash freezing and compute continuous-score
  AUROC/AUPRC as post-hoc analysis.
- [x] Save unified metrics, scatter plots, class histograms, manifests, hashes,
  and complete per-window identity.
- [x] Run 53 unit/regression tests and both real-data MPS smoke paths.

See `STAGE5A_REPORT.md`. CPSC2021 and LTAFDB yielded direction cosines 0.8553
and 0.9025, angles 31.21 and 25.52 degrees, and minimum split Spearman values
0.7873 and 0.9459. Both are formally `clearly_different` under the pre-frozen
rule: the prototype and head are strongly related but not equivalent.

## Stage 5B status

- [x] Freeze one deterministic four-dataset cohort with at most 500 windows per
  subject and class, shared by both reference models.
- [x] Extract L2-normalized embeddings for CPSC2021, LTAFDB, AFDB, and SHDB-AF
  in the frozen `M_CPSC` and `M_LTAF` feature spaces.
- [x] Use all four datasets' labels only for post-hoc mechanism analysis and
  record that use explicitly in artifacts and manifests.
- [x] Compute 4 x 4 disease-direction cosine and absolute centroid-distance
  matrices with window-weighted primary and subject-equal sensitivity results.
- [x] Save CSV matrices, feature summaries, hashes, manifests, and visually
  inspected four-panel heatmaps.
- [x] Run 59 unit/regression tests and both real-data MPS smoke/formal paths.

See `STAGE5B_REPORT.md`. Across the six dataset pairs, window-weighted disease
direction cosines were 0.9443–0.9945 in `M_CPSC` and 0.9841–0.9929 in `M_LTAF`,
while dataset-centroid distances remained 0.1264–0.4541 and 0.1631–0.3203.
Subject-equal results preserved the high direction alignment. This supports the
post-hoc mechanism hypothesis that absolute domain location shifts while the
relative AF-minus-non-AF direction remains substantially more stable.

## Stage 5C status

- [x] Preserve H0 as the binary-head logit-difference threshold 0, exactly
  equivalent to AF probability 0.5.
- [x] Select H1 on source validation by maximum Balanced Accuracy and freeze it
  before development-target evaluation.
- [x] Preserve P0 as the Stage 4 source prototype-midpoint threshold.
- [x] Select P1 on source validation using the same frozen rule.
- [x] Prove that threshold selection does not open the target index or access
  target labels; read development-target evaluation labels only after freeze.
- [x] Verify exact H0/H1 and P0/P1 AUROC/AUPRC invariance.
- [x] Save complete curves, thresholds, source-test/target metrics, hashes, and
  selection/evaluation manifests at clean commit `08748a0`.
- [x] Run 63 unit/regression tests and real frozen-score select/evaluate smoke.

See `STAGE5C_REPORT.md`. Source-optimized thresholds improved all reported
source-test operating-point metrics. On CPSC2021 -> SHDB-AF, H1 improved BACC
by 0.0062 and MCC by 0.0026 over H0. On LTAFDB -> AFDB, H1 changed BACC by only
+0.0002 and MCC by -0.0006; P1 lowered BACC and MCC versus P0 despite small
Macro-F1/accuracy gains. H1 is now the main fair source baseline, but source
threshold tuning alone does not consistently solve target boundary shift.

## Stage 5B+ status

- [x] Reuse the frozen Stage 5A head directions and Stage 5B four-dataset
  direction summaries without rereading ECG or accessing new target labels.
- [x] Compare each source head with all four same-space disease directions.
- [x] Compare mean prototype-prototype cosine with mean head-to-prototype cosine,
  both including and excluding the source dataset direction.
- [x] Construct the normalized four-direction shared axis and measure each
  source head's cosine and angle to it.
- [x] Repeat the analysis with subject-equal directions.
- [x] Save CSV, JSON, visually inspected figure, input hashes, and a clean-run
  manifest at commit `42efab1`.
- [x] Run 67 unit/regression tests.

See `STAGE5B_PLUS_REPORT.md`. In `M_CPSC`, mean prototype-prototype cosine was
0.9747 versus mean head-to-prototype cosine 0.8504; in `M_LTAF`, the values were
0.9892 versus 0.8839. The heads remain 30.85 and 27.44 degrees from their
four-dataset shared axes. Subject-equal results are nearly identical. This
supports source-specific linear-head rotation relative to the compact shared
disease geometry, without claiming universal prototype-score superiority.

## Stage 5D status

- [x] Reuse the exact frozen Stage 5B cohort for both reference models.
- [x] Extract and hash label-free disease-axis score archives before parsing
  any label.
- [x] Measure class-conditional means, variances, gap, pooled separation,
  overlap, prevalence, AUROC, and AUPRC for all eight model/dataset pairs.
- [x] Compare frozen P0/P1 thresholds with post-hoc target oracle thresholds
  and quantify boundary drift.
- [x] Save complete CSV/JSON statistics, manifests, hashes, and visually
  inspected density figures.
- [x] Run 72 unit/regression tests, two real MPS smoke paths, and two formal
  199,923-window MPS extractions at clean commit `6543c13`.
- [ ] Patient-level direction bootstrap (optional, non-blocking supplement).
- [ ] Label-permutation null (optional, non-blocking supplement).

See `STAGE5D_REPORT.md`. Disease-axis orientation transferred to every dataset,
but target gaps contracted to 44%--81% of the corresponding source gap. In
`M_LTAF`, all three cross-domain AUROCs remained above 0.93 while target-oracle
thresholds shifted by +0.319 to +0.540 relative to P1. The Decision Gate is a
qualified Case B: prioritize label-free Stage 6B boundary reconstruction, with
Stage 6A retained later as an auxiliary representation ablation because
`M_CPSC -> LTAFDB/AFDB` still shows material ranking/separation limitations.

## Historical post-Stage-5B execution order (superseded for new experiments)

The task priority below supersedes the earlier post-Stage-5 supplement order
and follows `EXPERIMENT_PLAN_UPDATE_AFTER_STAGE5B.md`. It records the frozen
Stage 5D decision but no longer controls new experiment execution after the
AFDB-source revision.

- [x] Stage 5A — Direction vs Linear Head.
- [x] Stage 5B — Four-Dataset Direction Geometry.
- [x] Stage 5C — Strong Source Baseline (H0/H1/P0/P1).
- [x] Stage 5B+ — Source Head vs Shared Cross-Dataset Disease Axis.
- [x] Stage 5D — Disease-Axis Distribution & Boundary Shift.
  - [x] Class-conditional means, variances, gap, separation, and prevalence.
  - [x] Source fixed/source-val/target-oracle thresholds and boundary drift.
  - [x] Density figures using the unchanged Stage 5B cohort.
  - [ ] Patient-level direction bootstrap (non-blocking supplement).
  - [ ] Label-permutation null (non-blocking supplement).
- [x] Stage 6 Decision Gate — qualified Case B; Stage 6B is the main priority.
- [ ] Stage 6A — Auxiliary representation enhancement after the Stage 6B main
  path; retained because some transfers remain ranking-limited.
- [ ] Stage 6B — Target Boundary Reconstruction v2.
  - [ ] Subject-balanced target fitting.
  - [ ] Source-anchored GMM.
  - [ ] Constrained positive shift/scale model.
  - [ ] Reliability fallback to H1/P1.
- [ ] Frozen full cross-domain benchmark.
- [ ] Patient-level bootstrap and final tables/mechanism figures.

## AFDB-source revision execution order (current priority)

This section follows `EXPERIMENT_PLAN_REVISION_AFDB_SOURCE.md` and supersedes
conflicting LTAFDB, source-domain, and post-Stage-5D ordering above. Frozen
Stage 1–5D artifacts remain unchanged.

- [x] Freeze the latest AFDB-source revision document before implementation.
- [x] Revision R1 — `LTAFDB-clean1h-v1`.
  - [x] Preserve `LTAFDB-original / pre-revision` and its index hash.
  - [x] Add the independent `ltaf_skip_first_hour_v1` dataset configuration.
  - [x] Exclude every grid window with `window_start_seconds < 3600` without
    labels, partial cropping, shifting, or padding.
  - [x] Build and hash the independent clean index and version manifest.
  - [x] Compare old and clean counts by class, record, and subject.
  - [x] Verify row identity, cutoff, split consistency, and zero leakage.
  - [x] Run the label-free 0–1 h versus `>=1 h` quality audit and preserve its
    mixed/null findings without overclaiming.
  - [x] Verify hidden-label target loading and finite `float32 [2, 2000]`
    preprocessing output.
  - [x] Run 78 unit/regression tests and formal real-data audits at clean commit
    `f274b31`.
  - [x] Document results in `REVISION_R1_REPORT.md`.
- [x] Revision R2 — AFDB Source Protocol.
  - [x] Audit current AFDB source split/trainer assumptions before coding.
  - [x] Freeze deterministic subject-level five-fold assignments (seed 42).
  - [x] Implement five-fold OOF development without target-label access.
  - [x] Freeze OOF head/prototype thresholds and final-epoch aggregation.
  - [x] Train full-source `M_AFDB` and run seed 42/2024/3407 stability.
  - [x] Verify exact OOF coverage, thresholds, checkpoint hashes, provenance,
    and 86 unit/regression tests; see `REVISION_R2_REPORT.md`.
- [x] Revision R3 — `M_AFDB` mechanism revalidation.
  - [x] AFDB prototype versus linear head.
  - [x] Four-dataset geometry using `LTAFDB-clean1h-v1`.
  - [x] AFDB head versus shared disease axis.
  - [x] Three-target disease-axis distribution shift.
  - [x] Verify 199,104-window identity coverage, label-free extraction, all
    output hashes, figures, and 89 unit/regression tests; see
    `REVISION_R3_REPORT.md`.
- [x] Run the new Decision Gate without inheriting historical Stage 6B-first:
  no predefined Case A/B/C was triggered, so the gate is formally
  inconclusive. Run the frozen three-target benchmark next; treat this order as
  a documented post-result protocol clarification, not a pre-frozen Case D.
- [ ] Run the frozen AFDB-to-three-target final benchmark.
- [ ] After the benchmark, reassess whether optional Stage 6A/6B development is
  justified; the R3 gate did not select an A/B order.
- [ ] Run patient-level statistics and assemble final tables/figures.
