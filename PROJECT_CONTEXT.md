# Project Context

## Research objective

This repository adapts MedTS-TTT for window-level, cross-dataset atrial
fibrillation (AF) recognition. A model learns a source-domain disease direction
from non-AF to AF. Unlabelled target-domain samples are projected onto that
direction, and a target-specific boundary is estimated from the target score
distribution without using target labels.

The initial task is strict binary rhythm classification:

- positive: the complete 10-second window is annotated AFIB;
- negative: the complete window is reliably annotated non-AF;
- excluded: transition windows, unreliable annotations, and explicit
  AFL/AT/PAT/NOD/J rhythms.

Target labels are reserved for final evaluation and post-hoc mechanism analysis.
They must not affect GMM fitting, threshold selection, updates, early stopping,
or hyperparameter selection. Every split is grouped by subject (or by record
when subject identity is genuinely unavailable).

## Stage 0 repository audit

Audited on 2026-07-27:

- Read `PROJECT_MASTER_PROMPT.md`, `README.md`, `MedTS_TTT.py`, and
  `benchmark/README.md`.
- The checkout contains the lightweight model, benchmark adapter, demo, and
  documentation, but not the upstream preprocessing/training framework.
- The current directory is not a Git work tree, so an upstream commit cannot
  yet be verified locally.
- Python is 3.12.4. PyTorch 2.6.0, NumPy 1.26.4, pandas 2.2.2, tqdm 4.66.4,
  and pytest 7.4.4 are installed. `wfdb` was not installed at audit time and
  has been added to `requirements.txt`.
- This statement was superseded on 2026-07-28: all four supplied datasets now
  live under `data/raw`. Their integrity and format findings are documented in
  `DATA_AUDIT_REPORT.md`.

## Original model contract

`MedTSTTT.forward` accepts a floating-point tensor shaped `[B, C, T]` and
returns classification logits shaped `[B, num_classes]`. The benchmark adapter
accepts the Medformer convention `[B, T, C]` and transposes it.

The forward path is:

1. Per-sample, per-channel z-score over time.
2. Temporal 2-D convolution.
3. Spatial convolution using the first `C` rows of a kernel declared with
   `max_channel`.
4. Channel-wise normalization, GELU, and non-overlapping temporal patch
   projection.
5. Learned absolute positional embedding.
6. A stack of gated convolution/TTT layers.
7. Mean pooling over tokens and a linear classification head.

The planned AF input is `[B, 2, 2000]`; with `patch_size=8`, this produces 250
tokens and logits `[B, 2]`.

Important constraints:

- `C` must not exceed `max_channel`. The AF configuration should set
  `max_channel=2`.
- The spatial layer's declared bias is not used by the functional convolution.
  This should be preserved during the first reproduction, then reviewed as a
  possible upstream implementation issue.
- Standard deviation uses PyTorch's default correction. Constant channels stay
  numerically finite because of the `1e-6` denominator term.
- The current public API exposes logits only.

## CLSA-TTT implementation

Each `TTTLinear` layer creates per-sample fast copies of slow parameters `W`
and `b`. From the current sample:

- projected keys `k` generate `Z = kW + b`;
- the self-supervised alignment target is `v - k`;
- a manually fused gradient is computed through layer normalization for
  `0.5 * ||LN(Z) - (v-k)||^2`;
- a learned per-head gate produces a sample-specific learning rate;
- exactly one gradient step updates the local fast `W` and `b`;
- queries `q` are evaluated with those updated fast weights.

The local fast weights are not written back to module parameters and therefore
do not persist across samples or batches. The computation occurs in both
training and evaluation forwards; there is currently no switch to disable it.
Because the update is represented as ordinary tensor operations, outer-loop
gradients can flow through the inner update to slow parameters.

## Positional encoding limit

`pos_embed` has fixed shape `[1, 256, dim]`. The number of patch tokens is
`ceil(T / patch_size)`. Local smoke tests established:

- 2000 samples at patch size 8 -> 250 tokens -> succeeds;
- 2048 samples -> 256 tokens -> succeeds;
- 2050 samples -> 257 tokens -> fails with a token-dimension mismatch.

Thus the hard limit is `ceil(T / patch_size) <= 256`. The initial 10-second,
200 Hz configuration is valid but leaves only six token positions of headroom.

## Required model changes after data stages

No model code is changed in stage 0. Later, in small independently tested steps:

1. Add `forward_features(x)` returning the mean-pooled backbone embedding.
2. Preserve `forward(x) -> logits` and optionally support
   `forward(x, return_features=True)`.
3. Add an explicit, tested CLSA-TTT enable/disable path for ablations without
   changing the enabled mathematics.
4. Add a separate projection head with L2-normalized output.
5. Make positional-length validation explicit; only introduce interpolation or
   a longer table if a later input configuration requires it.
6. Validate channel count and input rank with clear errors.
7. Keep source prototypes, disease direction, target GMM, and evaluation logic
   outside the backbone.

## Current limitations

- The active environment has persistent `wfdb==4.1.2`. MPS is unavailable
  inside the sandbox but was verified and used successfully outside the
  sandbox with Python 3.12.4 and PyTorch 2.6.0 for the formal LTAFDB run.
- The local experiment repository now has Git identity. This records future
  code state but cannot retrospectively prove the upstream commit of the
  originally supplied files.
- The locally supplied CPSC2021 collection is a MATLAB conversion and cannot be
  byte-verified against the official WFDB release.
- Dataset-specific rhythm parsing and subject-ID extraction are implemented and
  documented in `STAGE1_REPORT.md`.
- Stage 2 preprocessing and Stage 3 CE-only source training are complete.
  Tiny-overfit, fixed-seed small runs, and formal complete-split baselines
  passed for both CPSC2021 and LTAFDB. The LTAFDB MPS run selected epoch 6,
  stopped at epoch 16, and achieved test macro-F1 0.8778 and AUROC 0.9898.
  `MedTS_TTT.py` is still unchanged. See `STAGE2_REPORT.md` and
  `STAGE3_REPORT.md`.
