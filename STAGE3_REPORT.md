# Stage 3 Report — CE-only Source Baseline

## Scope

This stage implements the first source-only MedTS-TTT binary classifier. It
uses cross-entropy only. `MedTS_TTT.py` remains unchanged, and this stage does
not add embeddings, projection heads, SupCon, prototype losses, disease
directions, target GMMs, or any target-domain input.

The trainer API rejects configurations whose role is not `source`. Training,
early stopping, and checkpoint selection use source train/validation data only.

## Implementation

- `src/models/medts_ttt_wrapper.py`: validates `[B, 2, T]` input and `[B, 2]`
  logits while preserving the original model forward path.
- `src/training/engine.py`: CE train/evaluation loops with finite-loss and
  finite, non-zero gradient checks.
- `src/training/checkpointing.py`: atomic best/last checkpoints.
- `src/training/early_stopping.py`: source-validation macro-F1 early stopping.
- `src/training/reproducibility.py`: seeds, device selection, Git identity,
  package versions, and index SHA-256 provenance.
- `src/training/train_source.py` and `scripts/train_source.py`: experiment
  orchestration, tiny-overfit mode, complete checkpoint resume, and an explicit
  deterministic class-balanced diagnostic evaluation subset.
- `src/evaluation/metrics.py`: the protocol's binary metrics at a fixed 0.5
  threshold. Single-class diagnostic slices are not reported as perfect
  balanced metrics.
- `configs/experiments/source_{cpsc2021,ltafdb}_ce.json`: frozen CE-only source
  configurations, seed 42, batch size 8, Adam at `1e-4`, and the original
  `dim=128`, six-layer MedTS-TTT.

The diagnostic evaluation cap is opt-in via `--eval-windows-per-class`.
Formal runs omit it and evaluate the complete source validation/test splits.

## Validation results

### Automated checks

- `python -m pytest -q` -> **29 passed**.
- `python -m compileall -q src scripts tests` -> passed.
- Checkpoint unit test restores identical logits and continues optimization.
- A real CPSC2021 checkpoint resumed from epoch 1 through epoch 20.

### Tiny-overfit gates

Both runs used 8 AF and 8 non-AF source-training windows.

| Source | Best epoch | Accuracy | Macro-F1 | Best loss | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| CPSC2021 | 13 | 1.000 | 1.000 | 0.2086 | pass |
| LTAFDB | 7 | 1.000 | 1.000 | 0.2675 | pass |

Outputs are under `outputs/stage3/tiny_cpsc2021_ce/` and
`outputs/stage3/tiny_ltafdb_ce/`. Their manifests record clean Git commit
`a381aa4`, complete environment versions, data counts, and index hashes.

### Fixed-seed small runs

These are pipeline diagnostics, not the final baseline. Each epoch used the
first 100 batches from the seeded subject/class-balanced source sampler.
Validation and test each used a deterministic 256-window-per-class source
subset. The clean recorded commit is `28fa3a3`.

| Source | Train loss (epochs 1 -> 3) | Best val macro-F1 | Val AUROC | Diagnostic test macro-F1 | Test AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| CPSC2021 | 0.6767 -> 0.4224 | 0.7797 | 0.8951 | 0.5504 | 0.7116 |
| LTAFDB | 0.5923 -> 0.3704 | 0.8470 | 0.8860 | 0.8613 | 0.9276 |

The results show decreasing loss and non-trivial source validation behavior on
both input formats. They must not be cited as full-split baseline estimates.

## Full-run sizing and remaining work

The formal loaders contain:

| Source | Train windows/batches | Validation windows/batches | Test windows/batches |
| --- | ---: | ---: | ---: |
| CPSC2021 | 35,542 / 4,443 | 23,679 / 1,480 | 29,318 / 1,833 |
| LTAFDB | 40,332 / 5,042 | 97,399 / 6,088 | 104,601 / 6,538 |

Only CPU is available in both inspected Conda environments. A real full-loader
benchmark measured about 0.47 seconds per batch at the frozen batch sizes;
batch 32 did not improve end-to-end throughput. The rough cost is about 46
minutes per CPSC2021 epoch and 80 minutes per LTAFDB epoch, before the final
test pass. Therefore full early-stopped source runs remain pending and should
prefer a CUDA-capable host. They can be resumed safely from `last.pt`.

Formal commands (no diagnostic caps) are:

```bash
python scripts/train_source.py \
  --config configs/experiments/source_cpsc2021_ce.json --device cuda

python scripts/train_source.py \
  --config configs/experiments/source_ltafdb_ce.json --device cuda
```

Do not begin Stage 4 until both formal source checkpoints and complete-split
metrics have been reviewed.
