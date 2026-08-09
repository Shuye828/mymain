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
sensitivity/balanced accuracy, while reducing other metrics. The next task is
Stage 6 supervised contrastive loss, followed by prototype/center loss as a
separate experiment; target results must not be used for hyperparameter tuning.
