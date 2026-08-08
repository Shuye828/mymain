# AF Dataset Audit Report

Audit date: 2026-07-28

This report describes the data copied into `AF_DATA` and subsequently moved,
without conversion or preprocessing, into `data/raw`.

## Final layout

```text
data/raw/
├── afdb/
├── cpsc2021/
├── ltafdb/
└── shdb-af/
```

The source-to-target directory mapping was:

| Source | Target |
| --- | --- |
| `AF_DATA/CPSC2021` | `data/raw/cpsc2021` |
| `AF_DATA/afdb` | `data/raw/afdb` |
| `AF_DATA/ltaf` | `data/raw/ltafdb` |
| `AF_DATA/shdb` | `data/raw/shdb-af` |

The operation was a same-filesystem move. No ECG sample, annotation, MATLAB
field, filename, or bundled metadata file was transformed.

## Completeness summary

| Dataset | Local size | Main records | Result |
| --- | ---: | ---: | --- |
| LTAFDB | 3.4 GB | 84 | Complete against bundled official checksums |
| AFDB | 2.1 GB | 25 headers, 23 signal records | Signal/annotation content complete; local `RECORDS` manifest is modified |
| SHDB-AF v1.0.1 | 8.3 GB | 128 signals, 98 rhythm-annotated | Complete against bundled official checksums |
| CPSC2021 MATLAB collection | 6.5 GB | 1436 | Internally complete converted collection; not the official raw WFDB layout |

### LTAFDB

- 84 `.hea`, 84 `.dat`, 84 `.atr`, and 84 `.qrs` files.
- All 84 headers are readable with WFDB.
- All records have two channels at 128 Hz.
- All 339 entries in `SHA256SUMS.txt` pass.
- No header/signal/rhythm-annotation/QRS sidecar gaps were found.

Conclusion: complete for the planned work.

### MIT-BIH AFDB

- 25 `.hea`, 25 `.atr`, and 25 `.qrs` files.
- 23 records contain `.dat` signals; `00735` and `03665` do not.
- The two missing signals are an intentional property of the official database:
  both records contain only annotations and zero-signal headers.
- The 23 signal-bearing records are two-channel, 250 Hz recordings.
- All 25 headers are readable.
- 154 bundled checksum entries pass. The only mismatch is `RECORDS`.
- Local `RECORDS` is 138 bytes and lists only the 23 signal-bearing records.
  The checksum manifest expects the 150-byte official list, which also contains
  `00735` and `03665`.
- The directory additionally contains 23 MATLAB v5 files dated 2025. Each has
  one variable, `ecg`, and is not covered by the official checksum manifest.
  These are local derivatives and must not be treated as authoritative raw data.

Conclusion: all officially available signal and annotation content needed for
training is present. Preserve the manifest discrepancy as provenance; use the
23 WFDB signal records, not the local MATLAB derivatives.

### SHDB-AF v1.0.1

- 128 `.hea`, 128 `.dat`, and 128 `.qrs` files.
- 98 records have reference `.atr` rhythm annotations.
- 30 records intentionally have no `.atr`:
  `053`, `057`, `058`, `059`, `060`, `061`, `063`, `066`, `067`, `068`,
  `069`, `079`, `080`, `081`, `083`, `085`, `087`, `088`, `090`, `092`,
  `093`, `094`, `095`, `096`, `097`, `098`, `099`, `100`, `101`, `104`.
- The included version 1.0.1 README states that 30 unannotated recordings were
  added. Their missing `.atr` files are therefore not copy failures.
- All records have two channels (`ECG1`, `ECG2`) at 200 Hz.
- `AdditionalData.csv` has one header plus 128 record rows and includes
  `Data_ID`, `Subject_ID`, the `Annotated` flag, clinical metadata, and recording
  timing. The local table contains 122 distinct `Subject_ID` values, so splitting
  must use `Subject_ID` rather than treating all 128 recordings as independent
  patients.
- All 488 entries in `SHA256SUMS.txt` pass.

Conclusion: complete. Only the 98 annotated recordings are initially eligible
for strict supervised window-label construction; the other 30 may be useful as
unlabelled data under a separately declared protocol.

### CPSC2021 converted MATLAB collection

- 1436 MATLAB v7.3/HDF5 files named `data_{subject}_{record}.mat`.
- Subject identifiers span 0 through 104: 105 subjects total.
- Per-subject record numbering starts at 1 and has no internal gaps.
- All 1436 files open successfully and contain the required `record` group.
- Every record reports 200 Hz and contains equal-length lead-I/lead-II raw and
  processed signal arrays.
- Observed record labels:
  - `Normal`: 732
  - `AFp`: 229
  - `AEf`: 475
- Sample lengths range from 1675 to 4,933,325.
- Required signal, R-wave, cycle-type, AF-boundary, and quality fields are
  present in every file.

This directory is not the official PhysioNet CPSC2021 release layout. The
official release provides 1436 training records in WFDB form split between
Training Set I (730) and II (706). The local collection is a larger MATLAB
conversion that duplicates raw and processed signals and does not include the
official `RECORDS` or checksum manifest. Its original byte-level equivalence to
the official release cannot therefore be independently verified.

Conclusion: structurally complete and sufficient to build a dataset adapter,
provided only the raw signal fields and official annotation-derived fields are
used. For strict upstream reproducibility, retain a future task to acquire or
verify against the official WFDB release.

## WFDB file meanings

- `.hea`: text header describing record name, signal count, sampling frequency,
  length, encoding, gain, units, channel names, and optional timing metadata.
- `.dat`: binary interleaved ECG samples interpreted using the corresponding
  header. It has no standalone schema.
- `.atr`: reference annotation stream. It includes beat/rhythm-change events and
  auxiliary rhythm tokens such as `(AFIB`; this is the primary rhythm source.
- `.qrs`: automatically detected QRS/beat locations. It is not the reference
  rhythm label source.
- `.qrsc`: corrected beat locations available for only some AFDB records.
- Files ending in `.hea-`, `.atr-`, or `.qrs-`: historical/backup versions
  retained by the official AFDB distribution. Use the unsuffixed current files.
- `RECORDS`/`RECORDS.txt`: record-name inventory.
- `ANNOTATORS`: description of available annotation suffixes.
- `SHA256SUMS.txt`: expected file hashes for integrity verification.

## CPSC2021 MATLAB field meanings

Each v7.3 file contains a `record` group. Important fields are:

- `signal_lead1`, `signal_lead2`: original two ECG signal arrays in the local
  conversion. These are the only waveform fields eligible for the initial
  no-preprocessing loader.
- `signal_lead1_processed`, `signal_lead2_processed`: previously processed
  copies. Do not use them in the raw-data pipeline.
- `fs`: sampling frequency, consistently 200 Hz.
- `label`: global record class (`Normal`, `AFp`, or `AEf`).
- `offical_RwavePos`: challenge-provided reference R-wave sample positions
  (the spelling `offical` is part of the stored schema).
- `offical_CycleType`: challenge-provided beat/cycle types.
- `AF_startPoints_byOfficalRwave`,
  `AF_endPoints_byOfficalRwave`: AF episode boundaries indexed relative to the
  official R-wave sequence.
- `RwavePos1`, `RwavePos2`: lead-specific R-wave locations.
- `cycles_start1/2`, `cycles_end1/2`: lead-specific cardiac-cycle bounds.
- `signalQaulity_flag1/2`: lead-specific signal-quality arrays; the stored
  spelling `Qaulity` is part of the schema.
- `filename`, `dataset`: source identity fields.

The bundled MATLAB scripts are local experimental utilities with hard-coded
Windows paths. They are preserved for provenance but are not part of the new
data-loading contract.

## Restrictions for subsequent stages

1. Do not use CPSC fields with `_processed` in the initial pipeline.
2. Do not use AFDB's 2025 `.mat` derivatives as raw inputs.
3. Use `.atr`, not `.qrs`, for rhythm labels.
4. Exclude the 30 unannotated SHDB-AF records from labelled window generation.
5. Exclude AFDB `00735` and `03665` from signal-based model inputs.
6. Preserve subject grouping: CPSC filename subject component, SHDB-AF
   `Subject_ID`, and dataset-specific reviewed mappings for the other databases.
7. Do not repair the AFDB `RECORDS` file silently; record its discrepancy.
