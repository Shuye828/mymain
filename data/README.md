# Data Layout

Raw or source-format datasets live under `data/raw` and are ignored by Git.
They were moved here without preprocessing on 2026-07-28:

```text
data/raw/
├── afdb/       # MIT-BIH Atrial Fibrillation Database
├── cpsc2021/   # local MATLAB v7.3 conversion of CPSC2021
├── ltafdb/     # Long-Term AF Database
└── shdb-af/    # SHDB-AF v1.0.1
```

Do not edit files under `data/raw`. Rebuildable window inventories belong in
`data/index`, and optional caches belong in `data/cache`.

The source-format audit, exceptions, and file descriptions are recorded in
[`../DATA_AUDIT_REPORT.md`](../DATA_AUDIT_REPORT.md).

