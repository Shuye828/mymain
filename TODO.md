# TODO

## Stage 0

- [x] Read `PROJECT_MASTER_PROMPT.md`, `README.md`, `MedTS_TTT.py`, and
  `benchmark/README.md`.
- [x] Audit repository layout and installed dependencies.
- [x] Verify the model's 250/256/257-token behavior with a local smoke test.
- [x] Document project, dataset, and experiment contracts.
- [x] Add a read-only WFDB header audit and minimal tests.
- [x] Add the proposed base directory structure and ignore rules.
- [ ] Install `wfdb` in the active environment.
- [x] Locate and organize all four supplied datasets under `data/raw`.
- [x] Run structural/header audits against the supplied datasets.
- [x] Verify bundled checksums for LTAFDB and SHDB-AF.
- [ ] Resolve/record upstream provenance for the converted CPSC2021 MATLAB set.
- [ ] Decide whether to restore AFDB's official 25-entry `RECORDS` manifest;
  never change it silently.
- [ ] Put the project under Git or provide the real Git checkout so the pinned
  upstream commit can be verified.

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

- [ ] Reproduce CE-only source baseline before representation losses.
- [ ] Add feature export and source disease direction.
- [ ] Add target GMM with reliability diagnostics.
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

## Open items before Stage 3

- Install `wfdb` persistently in the formal experiment environment and record
  the complete package versions.
- Establish a Git commit identity before producing training checkpoints.
- Implement CE-only source training and checkpointing before changing the
  representation losses or disease-direction model.
