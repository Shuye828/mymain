# Stage 2 Report: Subject Splits, Window Indices, and On-Demand Loading

Stage 2 creates deterministic subject manifests and strict 10-second window
indices, then reads and preprocesses source segments lazily. It does not save
individual window arrays or train/modify a model.

## Frozen configuration

- Split version: `subject_random_v1`
- Window version: `strict_10s_nonoverlap_v1`
- Rhythm mapping: `strict_af_v1`
- CPSC boundary version: `cpsc_rpeak_half_open_v1`
- Seed: 42
- Source split: 70% train / 15% validation / 15% test
- Target inductive split: 50% adaptation / 50% evaluation
- Window duration/stride: 10 seconds / 10 seconds
- Grid origin: source sample zero
- Target sampling frequency: 200 Hz
- Target tensor shape: `[2, 2000]`
- Band-pass: 0.5-40 Hz, fourth-order Butterworth, zero phase
- Extra z-score: disabled
- Source training cap: 500 windows per subject per class

Subject assignment uses identifiers and eligibility only. It does not inspect AF
or non-AF class proportions. The generated source splits were audited afterward
to confirm both classes exist, but assignments were not changed in response.

## Subject manifests

| Dataset | Eligible subjects | Source train/validation/test | Target adaptation/evaluation |
| --- | ---: | --- | --- |
| LTAFDB | 84 | 59 / 12 / 13 | 42 / 42 |
| CPSC2021 | 105 | 73 / 16 / 16 | 53 / 52 |
| AFDB | 23 | 16 / 3 / 4 | 12 / 11 |
| SHDB-AF | 93 | 65 / 14 / 14 | 47 / 46 |

SHDB eligibility initially uses the 98 annotated records, corresponding to 93
subjects. Its 30 unannotated records are not mixed into the first protocol.

## Window construction

Every candidate is anchored to the record-wide 10-second grid. A window is
accepted only when all of its samples lie in one `af` or `nonaf` interval.

- Crossing a rhythm boundary -> `transition`, excluded.
- Lying in an excluded rhythm -> excluded.
- Lying before the first WFDB rhythm marker -> unannotated, excluded.
- Missing annotation/signal -> excluded.
- Shorter than 10 seconds -> excluded.
- A half-open window endpoint exactly equal to a rhythm boundary is valid
  because it contains no sample from the neighboring interval.

No signal is read while building the index.

### Accepted windows

| Dataset | Total | non-AF | AF | Records | Subjects |
| --- | ---: | ---: | ---: | ---: | ---: |
| LTAFDB | 647,492 | 281,801 | 365,691 | 84 | 84 |
| CPSC2021 | 173,316 | 114,669 | 58,647 | 1,435 | 105 |
| AFDB | 83,150 | 49,805 | 33,345 | 23 | 23 |
| SHDB-AF | 821,392 | 658,007 | 163,385 | 98 | 93 |
| **Total** | **1,725,350** | **1,104,282** | **621,068** |  |  |

All source train, validation, and test splits contain both classes.

Notable exclusions:

- LTAFDB: 34,967 transition windows, 14,158 excluded-rhythm windows, and
  9,199 unannotated-prefix windows.
- CPSC2021: 1,090 transition windows and one record shorter than 10 seconds.
- AFDB: two official annotation-only records, 584 transitions, and 600
  excluded-rhythm windows.
- SHDB-AF: 1,717 transitions and 15,946 excluded-rhythm windows. Of the 30
  unannotated records, 29 belong to subjects with no eligible annotated record;
  one shares a subject with an annotated record but itself yields no intervals.

## Index contract

Each accepted row retains:

- dataset, record, subject, and source path;
- original sampling frequency and channel order;
- original-rate start/end samples;
- raw rhythm token and binary label;
- source and target protocol assignments;
- annotation provenance;
- mapping, split, window, and CPSC boundary versions.

The `split` column remains an alias of `source_split` for compatibility with the
original project contract.

## On-demand preprocessing

`ECGWindowDataset` performs:

1. source segment read in original channel order;
2. rank/channel/length and finite-value validation;
3. 0.5-40 Hz zero-phase filtering at the original sampling rate;
4. polyphase resampling to 200 Hz;
5. conversion to `float32 [2, 2000]`.

It deliberately performs no z-score because MedTS-TTT normalizes each channel
internally.

The local CPSC raw arrays use a much larger count-like amplitude scale than the
WFDB physical-unit signals. This is preserved rather than silently calibrated;
the model's internal per-channel z-score handles scale for the initial
experiment. A later units/calibration sensitivity analysis may be added without
changing the stage 2 raw-data contract.

## Sampling and label-leakage safeguards

- The complete window CSVs are retained.
- Source training can deterministically retain at most 500 windows per
  subject/class via stable hashing.
- The real capped source-train collection contains 139,430 windows across 326
  subject/class groups; no group exceeds 500.
- A weighted sampler gives every observed subject/class group equal total mass.
- `load_unlabeled_target_rows` replaces target binary labels with `-1` and
  rhythm labels with `__HIDDEN_TARGET_LABEL__`.
- Class-aware caps are forbidden whenever `target_split` is requested.
- `ECGWindowDataset(expose_label=False)` does not expose target labels in its
  tensor or metadata API.

## Validation

- Unit tests: 23 passed.
- Window consistency: 1,725,350/1,725,350 rows valid.
- Patient leakage/conflicting split assignments: zero.
- Real pipeline smoke test: 16 windows (two per class per dataset) all produced
  finite `float32 [2, 2000]` tensors.
- Deterministic rebuild: all four window-index SHA-256 values were unchanged.

Deterministic window hashes:

```text
ltafdb   214fb681412e06ca9c4b9f152176e803a2c572b9813e4d7b6641f7747368420a
cpsc2021 1e005a25dd87f405c634337e73775df384460bbbadfbc41b1c27ce88be2f0834
afdb     f6528265320183418b001e14e9ece11e6128263d8b55114e1e37412f984a5d35
shdb-af  b6d018738681e9dcd80582d20a276a3b31dda8c8b8d1e2db228dc9bcbd311231
```

## Commands

```bash
python scripts/build_subject_splits.py
python scripts/build_window_indices.py
python scripts/audit_window_indices.py
python scripts/validate_window_pipeline.py
python -m pytest -q
```

The active environment must first install `requirements.txt`.

## Stage boundary

Stage 2 is complete. `MedTS_TTT.py` remains unchanged and no checkpoint,
embedding, prototype, disease direction, GMM, or training result was created.
