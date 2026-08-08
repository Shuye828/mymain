# Stage 1 Report: Source Records and Rhythm Intervals

Stage 1 implements read-only source adapters and rhythm interval parsing. It
does not filter, resample, normalize, cut windows, assign splits, or train a
model.

## Implemented interfaces

Each adapter provides:

- `list_records()`
- `read_metadata(record_id)`
- `read_signal(record_id, start_sample, end_sample)`
- `read_rhythm_intervals(record_id)`

`read_signal` returns decoded physical samples shaped `[channels, time]` while
preserving source channel order. No signal transformation is applied.

Implementations:

- `src/data/wfdb_adapter.py`: LTAFDB, AFDB, and SHDB-AF.
- `src/data/cpsc2021_adapter.py`: local MATLAB v7.3/HDF5 CPSC2021 collection.
- `src/data/rhythm_intervals.py`: complete half-open interval construction.
- `src/data/rhythm_mapping.py`: strict, auditable mapping loader.
- `src/data/registry.py`: canonical adapter construction.

## Subject identity

| Dataset | Stage 1 `subject_id` |
| --- | --- |
| LTAFDB | record ID; one supplied record per subject |
| AFDB | record ID; one supplied record per subject |
| CPSC2021 | first numeric filename component in `data_{subject}_{record}` |
| SHDB-AF | `Subject_ID` from `AdditionalData.csv` |

SHDB-AF has 128 records but 122 distinct subjects. All later splitting must use
the subject field rather than the record ID.

## Strict rhythm mapping

The versioned mapping is
`configs/datasets/rhythm_mapping.json` (`strict_af_v1`).

Rules:

- Explicit AFIB and CPSC AF intervals -> `af`.
- Explicit normal rhythm and CPSC non-AF intervals -> `nonaf`.
- AFL, AT, PAT, NOD, J, and all other non-normal rhythm tokens -> `exclude`.
- Unknown tokens -> `exclude`, even when absent from the mapping.
- Samples before the first WFDB rhythm marker -> `__UNANNOTATED__` and
  `exclude`.
- Records without usable annotations yield no labelled intervals.

This is intentionally more conservative than “AF versus every other rhythm.”

### Observed WFDB tokens

- LTAFDB: `(N`, `(AFIB`, `(VT`, `(AB`, `(SVTA`, `(T`, `(B`, `(SBR`, `(IVR`.
- AFDB: `(N`, `(AFIB`, `(AFL`, `(J`.
- SHDB-AF: `(N`, `(AFIB`, `(AFL`, `(AT`, `(PAT`, `(NOD`, `(AB`.

### CPSC2021 interpretation

- `Normal`: complete record is explicit non-AF.
- `AEf`: complete record is AF.
- `AFp`: AF boundaries are taken from
  `AF_startPoints_byOfficalRwave`/`AF_endPoints_byOfficalRwave`; the complement
  is explicit non-AF.
- A start index of zero is a valid sentinel meaning AF begins at the record
  start. It occurred in 27 AFp records and is not treated as corruption.
- MATLAB beat positions are converted to zero-based, half-open sample
  intervals. Endpoints at the last official beat extend to record end.
- `offical_CycleType` contains beat types, not the rhythm interval contract.
- The adapter reads `signal_lead1/2` only. It never reads
  `signal_lead1/2_processed`.

## Real-data validation

All 1673 supplied records were inspected through their adapters. For every
signal-bearing record, the first 32 raw samples per channel were decoded and
checked for shape and finite values. Every constructed annotated interval list
was required to:

1. remain within signal bounds;
2. be sorted and non-overlapping;
3. start at sample zero;
4. have no gaps;
5. end at the record length.

| Dataset | Records | Subjects | Signal records | Annotated records | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| LTAFDB | 84 | 84 | 84 | 84 | Valid |
| CPSC2021 | 1436 | 105 | 1436 | 1436 | Valid |
| AFDB | 25 | 25 | 23 | 25 | Valid; 2 official annotation-only records |
| SHDB-AF | 128 | 122 | 128 | 98 | Valid; 30 official unannotated records |

No adapter, signal-slice, subject-map, or interval error was observed.

## Generated inventories

Local, rebuildable outputs:

- `outputs/data_audit/record_inventory.csv`
- `outputs/data_audit/rhythm_inventory.csv`
- `outputs/data_audit/inventory_summary.json`
- `outputs/data_audit/adapter_validation.json`

The inventories are ignored by Git because they are derived from local data.

Commands:

```bash
python scripts/validate_dataset_adapters.py \
  --data-root data/raw \
  --output outputs/data_audit/adapter_validation.json \
  --signal-samples 32

python scripts/build_rhythm_inventory.py \
  --data-root data/raw \
  --output-dir outputs/data_audit
```

The active environment must have `wfdb` and `h5py`; both are declared in
`requirements.txt`.

## Tests

The test suite covers:

- unannotated prefixes and rhythm transitions;
- unknown-token exclusion;
- normal, short, and AFp CPSC records;
- proof that CPSC reads unprocessed rather than processed arrays;
- missing annotations;
- AFDB-style annotation-only records;
- SHDB record-to-subject mapping;
- raw signal slice shapes.

Result: `12 passed`.

## Stage boundary

Stage 1 satisfies the repository TODO. Stage 2 has not started. In particular,
there are no window CSVs, filtered signals, resampled signals, cached samples,
or train/validation/test splits.

