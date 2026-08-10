# Dataset Contract

## Canonical dataset identifiers and paths

The canonical identifiers are:

| Identifier | Dataset | Default local path |
| --- | --- | --- |
| `ltafdb` | Long-Term AF Database | `data/raw/ltafdb` |
| `cpsc2021` | CPSC 2021 | `data/raw/cpsc2021` |
| `afdb` | MIT-BIH Atrial Fibrillation Database | `data/raw/afdb` |
| `shdb-af` | SHDB-AF v1.0.1 | `data/raw/shdb-af` |

Paths are configurable and must never be embedded as machine-specific absolute
paths. For a WFDB dataset, a directory is header-audit complete only when it
exists, contains the expected `.hea` files, all discovered headers are readable,
and every signal-bearing record has positive sampling rate/signal length/channel
count and the expected sidecars. Dataset-defined exceptions (AFDB's two
annotation-only records and SHDB-AF's 30 unannotated records) must be represented
explicitly rather than reported as copy failures. Header completeness does not
imply that signal or annotation contents are semantically valid.

The stage 0 audit searches recursively because some distributions contain
nested record directories. A WFDB record is identified by its `.hea` file;
signal files are not loaded.

The locally supplied CPSC2021 dataset is an exception: it consists of 1436
MATLAB v7.3/HDF5 files rather than the official WFDB directory layout. Its raw
waveform fields are `record/signal_lead1` and `record/signal_lead2`; fields
ending in `_processed` must not be used by the initial pipeline. See
`DATA_AUDIT_REPORT.md` for the verified schema and provenance limitation.

## Unified model sample

| Property | Contract |
| --- | --- |
| Tensor shape | `[2, 2000]` |
| Dtype | `float32` |
| Channel order | Original dataset order, recorded in metadata |
| Duration | 10 seconds |
| Target sampling rate | 200 Hz |
| Initial stride | 10 seconds (non-overlapping) |
| Filtering | 0.5-40 Hz band-pass |
| Missing values | Reject or explicitly repair; never silently propagate |
| Scaling | No preprocessing z-score; model performs per-channel z-score |

Filtering must occur at the original sampling rate before resampling unless a
dataset-specific validated procedure documents otherwise. Any resampling,
filtering, or read failure must be logged with record and sample bounds.

The active implementation uses a fourth-order zero-phase Butterworth filter,
then polyphase resampling. A 10-second segment must return finite
`float32 [2, 2000]`; unexpected shapes, lengths, channels, NaN, or Inf raise an
explicit error. CPSC count-like amplitude scale is preserved, and no additional
z-score is applied.

## Window-index schema

The canonical index lives at `data/index/{dataset}_windows.csv`. Required
columns are:

| Column | Meaning |
| --- | --- |
| `dataset` | Canonical dataset identifier |
| `record_id` | Dataset-relative WFDB record identifier |
| `subject_id` | Stable person/recording-subject identifier |
| `source_path` | Rebuildable path relative to configured dataset root |
| `fs_original` | Original sampling rate |
| `channel_names` | Ordered channel names, JSON encoded |
| `start_sample` | Inclusive original-rate start |
| `end_sample` | Exclusive original-rate end |
| `rhythm_label` | Normalized source rhythm label |
| `binary_label` | `1` AF, `0` non-AF |
| `is_transition` | Whether the interval crosses a rhythm boundary |
| `split` | `train`, `validation`, `test`, `adaptation`, or `evaluation` |
| `annotation_source` | File/stream and parser provenance |

Recommended additional columns are `duration_seconds`, `dataset_version`,
`label_rule_version`, `channel_count`, `quality_status`, and `exclusion_reason`.
All prediction/embedding outputs retain at least `dataset`, `subject_id`,
`record_id`, and `start_sample`.

Stage 2 additionally records `source_split`, `target_split`,
`target_transductive_split`, `mapping_version`, `split_version`,
`window_version`, and `cpsc_boundary_version`. `split` aliases
`source_split`.

Windows are aligned to a record-wide grid beginning at sample zero. Only full
containment within one allowed half-open rhythm interval is accepted. Exact
endpoint equality is allowed; crossing a boundary is not.

### LTAFDB version contract after Revision R1

The historical `data/index/ltafdb_windows.csv` remains immutable and is named
`LTAFDB-original / pre-revision` in later reports. Revision R1 adds the logical
dataset version `ltaf_skip_first_hour_v1`, displayed as
`LTAFDB-clean1h-v1`, with its independent index at
`data/index/ltaf_skip_first_hour_v1/ltafdb_windows.csv`.

The clean version preserves the original WFDB files and their absolute
source-sample coordinates. It applies one label-independent rule to every
record before accepting windows: `window_start_seconds >= 3600`. A grid window
whose start precedes 3600 seconds is excluded in full; no partial window is
cropped, shifted, or padded. All other duration, stride, interval-containment,
label-mapping, subject-split, filtering, resampling, and tensor-shape contracts
remain unchanged. The clean index must therefore equal the historical index
filtered at the cutoff, except for its new `window_version` value.

The frozen version manifest, index comparison, and label-free 0–1 h versus
`>=1 h` raw-signal quality audit live under
`outputs/revision_r1_ltaf_clean1h/`. Quality metrics provide dataset-level
evidence only and must never cause patient-specific filtering or access rhythm
labels. No R1 operation may modify raw LTAFDB files, the historical index, or
pre-revision model artifacts.

## Strict label policy

- Positive: the entire window lies within reliable AFIB rhythm.
- Negative: the entire window lies within reliable non-AF rhythm.
- Exclude any window crossing an annotation boundary.
- Exclude unknown/unreliable annotation intervals.
- Initially exclude explicit AFL, AT, PAT, NOD, and J rhythms.
- Do not infer current-window AF from patient history or record-level diagnosis.
- Do not silently remap an unrecognized rhythm. Add it to a reviewed mapping or
  exclude it with a recorded reason.

Dataset adapters must preserve the raw annotation token alongside the normalized
label. Rhythm mappings will be documented and unit-tested in stage 1.

The active strict mapping is
`configs/datasets/rhythm_mapping.json` (`strict_af_v1`). Only explicit AFIB is
positive and only explicit normal rhythm is negative. Every other observed
WFDB rhythm is excluded in the first version. Samples before a record's first
WFDB rhythm marker are represented as `__UNANNOTATED__` and excluded.

CPSC2021 uses the following source-specific interval contract:

- `Normal`: whole-record non-AF;
- `AEf`: whole-record AF;
- `AFp_AF`: intervals derived from official R-wave-indexed AF boundaries;
- `AFp_nonAF`: the complement of those intervals;
- zero AF start index: record-start sentinel.

All adapter intervals are zero-based, half-open, sorted, non-overlapping, and
cover the complete signal when an annotation contract is available.

## Leakage and sampling rules

- Split by `subject_id`; use record grouping only when subject linkage is truly
  unavailable and document that limitation.
- Assert that group intersections between splits are empty.
- Fix all random seeds and record the split manifest.
- Cap windows with `max_windows_per_subject_per_class`.
- Use subject/record-aware balancing so long recordings cannot dominate.
- Report counts by dataset, split, subject, record, class, and exclusion reason.

Target-domain labels are inaccessible to adaptation. For transductive
evaluation, unlabelled evaluation inputs may define the GMM. For inductive
holdout, only the unlabelled adaptation subjects may define it; evaluation
subjects remain separate.

Target split assignment never uses rhythm labels. Adaptation loaders must use
`load_unlabeled_target_rows` and `ECGWindowDataset(expose_label=False)`.
Class-aware target subsampling is prohibited.

Source training may apply a deterministic cap of 500 windows per subject/class.
The complete index remains unchanged, and patient/class-balanced sampling is
performed only in the training loader.

## Stage 1 subject identity

| Dataset | Subject grouping |
| --- | --- |
| `ltafdb` | record ID |
| `afdb` | record ID |
| `cpsc2021` | first numeric component of `data_{subject}_{record}` |
| `shdb-af` | `Subject_ID` in `AdditionalData.csv` |

SHDB-AF `Data_ID` identifies a recording and must not be substituted for
`Subject_ID` during splitting.

## Stage 0 header-audit output

The audit CSV contains one row per header with:

`dataset`, `record_id`, `header_path`, `read_ok`, `error`, `fs`,
`n_sig`, `channel_names`, `sig_len`, `duration_seconds`,
`annotation_exists`, and `annotation_files`.

The summary JSON records requested/resolved paths, missing directories, record
counts, readable/unreadable headers, distinct sampling rates/channel counts,
annotation coverage, and a conservative `complete` flag. It is an inventory,
not a substitute for stage 1 annotation parsing.
