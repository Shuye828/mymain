# Revision R1 Report — LTAFDB-clean1h-v1

## Scope and protocol status

Revision R1 implements the frozen rule in
`EXPERIMENT_PLAN_REVISION_AFDB_SOURCE.md`: remove every LTAFDB grid window whose
start time is before 3600 seconds, without consulting AF/non-AF labels and
without changing the raw files. This revision does not enter Stage 6 and does
not modify or reinterpret any frozen Stage 1–5D result.

The implementation ran from clean Git commit
`f274b31b718502b29e03199defdc956aed5a258b`. The governing revision document was
frozen separately at commit `1c5405b` and has SHA-256
`6747ca843386eeb3ef9351af409337ea897c046e019490aa9584d2ce9eaaafb6`.

## Dataset versions

| Role | Name | Index | SHA-256 |
| --- | --- | --- | --- |
| Historical, frozen | `LTAFDB-original / pre-revision` | `data/index/ltafdb_windows.csv` | `214fb681412e06ca9c4b9f152176e803a2c572b9813e4d7b6641f7747368420a` |
| R1 active version | `LTAFDB-clean1h-v1` / `ltaf_skip_first_hour_v1` | `data/index/ltaf_skip_first_hour_v1/ltafdb_windows.csv` | `f5aa7ad7c29fa138445e73a40b76a7140e85c365f0717db2fe1f536ebc98befe` |

The old index hash was checked before and after construction and did not
change. The new version keeps the original sample coordinates; it is not a
physically cropped copy of the raw dataset. Its logical version fingerprint is
`d38ec7f224f3f5b7f643a629d334433a0a7ff62b15a56ec65a5bb613f2bff767`.

## Index construction and comparison

The unchanged 10-second, 10-second-stride global grid was used. The additional
acceptance constraint is `window_start_seconds >= 3600`. Therefore no window is
cut, shifted, padded, or relabelled at the one-hour boundary.

| Measure | Original | Removed at 0–1 h | Clean1h retained |
| --- | ---: | ---: | ---: |
| Accepted windows | 647,492 | 25,255 | 622,237 |
| non-AF windows | 281,801 | 11,963 | 269,838 |
| AF windows | 365,691 | 13,292 | 352,399 |
| Subjects / records | 84 / 84 | — | 84 / 84 |

The clean version retains 96.10% of the historical accepted windows. Its class
composition is 43.37% non-AF and 56.63% AF, close to the historical 43.52% and
56.48%. Across all raw records, the duration remaining after the fixed cutoff
is 6,755,760 seconds (1,876.6 hours).

There are 30,240 global-grid windows before the cutoff (84 records × 360
windows). Of these, 25,255 were accepted in the historical index; the remaining
4,985 were already excluded by strict annotation rules. After the cutoff,
622,237 windows were accepted, while 33,365 transition windows, 13,540 excluded
rhythm windows, and 6,434 unannotated windows were rejected.

The formal audit confirmed:

- all 622,237 clean rows equal the historical rows filtered at 3600 seconds,
  except for the deliberately changed `window_version`;
- the minimum retained start is exactly 3600.0 seconds;
- all 84 subjects retain one consistent source split and one consistent target
  split, with no detected patient leakage;
- the historical index and raw checksum manifests remain unchanged.

## Label-free 0–1 h versus >=1 h quality audit

For each of 84 records, the audit used all 360 non-overlapping windows in the
first hour and a deterministic, label-independent sample of 360 windows from
the remaining recording. Each channel was summarized separately and the final
comparison gave each record equal weight. No annotation file or AF/non-AF label
was opened. The audit cannot dynamically remove individual patients or
windows.

| Metric (record-equal median) | 0–1 h | >=1 h | Median paired difference (early − late) | Records where early is worse |
| --- | ---: | ---: | ---: | ---: |
| Finite-value ratio | 1.000000 | 1.000000 | 0.000000 | 0.0% |
| Flatline ratio | 0.093628 | 0.116106 | -0.019253 | 13.1% |
| Extreme-amplitude ratio (`>5 mV`) | 0.000000 | 0.000000 | 0.000000 | 0.0% |
| Extreme first-difference ratio (`>1 mV/sample`) | 0.000000 | 0.000000 | 0.000000 | 9.5% |
| High-frequency power ratio (20–40 Hz / 0.5–40 Hz) | 0.080942 | 0.080818 | +0.003872 | 66.7% |

### Interpretation

The simple signal-quality evidence is mixed. The high-frequency ratio supports
some excess early high-frequency content: it is higher in the first hour in
two-thirds of records. However, the effect in the record-equal median is small,
the flatline metric is usually higher after one hour, and the remaining three
metrics show no meaningful median separation at the frozen thresholds.

Consequently, this audit does **not** establish that every measured aspect of
signal quality is generally worse during the first hour. It provides limited
support for an early high-frequency-noise tendency while also documenting
contrary or null indicators. The fixed deletion remains a protocol-defined,
uniform sensitivity rule from the latest experiment revision; it must not be
presented as a patient-specific filter or as a conclusion selected after seeing
labels. Any stronger quality claim would require a separately pre-specified SQI
analysis rather than changing R1 retrospectively.

## Loader and preprocessing validation

Both target partitions were loaded through `load_unlabeled_target_rows` and
`ECGWindowDataset(expose_label=False)`:

| Target split | Rows | Visible/returned labels | Real sample result |
| --- | ---: | --- | --- |
| adaptation | 317,364 | `-1` only | finite `torch.float32 [2, 2000]` |
| evaluation | 304,873 | `-1` only | finite `torch.float32 [2, 2000]` |

This verifies that the revised index remains compatible with the frozen
0.5–40 Hz filtering and 200 Hz resampling path and does not expose target
labels.

## Validation record

- Targeted index, quality, preprocessing, and loader tests: **16 passed**.
- Full regression suite: **78 passed**, with one non-failing joblib CPU-count
  warning.
- Real-data diagnostic build: **84 records, 622,237 accepted windows**.
- Real-data diagnostic quality run: **2 records × 4 windows per period**.
- Formal index identity/leakage audit: **valid, zero errors**.
- Formal quality run: **84 records × 360 windows per period**, 27.7 seconds.
- Formal build and quality manifests both record `git.dirty = false` at
  `f274b31`.

## Artifacts

All formal R1 artifacts are under `outputs/revision_r1_ltaf_clean1h/`:

- `dataset_version_manifest.json`
- `index_build_summary.json`
- `old_vs_clean_summary.json`
- `index_validation.json`
- `record_window_counts.csv`
- `subject_window_counts.csv`
- `ltaf_quality_before_after.csv`
- `ltaf_quality_summary.json`
- `ltaf_quality_comparison.png`
- `quality_run_manifest.json`

The 134 MB clean index and the formal output directory are intentionally
ignored by Git, consistent with the repository's existing large-data/artifact
policy. Their paths and hashes are recorded above and in the manifests.

## Completion decision and next step

Revision R1 meets its implementation completion criteria: old and new versions
coexist, the new index has a unique hash, all retained windows start at or after
3600 seconds, no leakage is detected, the target loader works, and
preprocessing returns finite `float32 [2, 2000]` tensors without altering old
artifacts.

Per the latest revision plan, the next task is **Revision R2 — AFDB Source
Protocol**, beginning with a small implementation audit and a frozen design for
subject-level five-fold OOF development plus full-source final training. Stage
6 must not begin before R2, R3, and the new Decision Gate.
