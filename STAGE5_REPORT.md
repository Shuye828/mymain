# Stage 5 Report: Label-Free Target GMM Reconstruction

## Scope and protocol

Stage 5 projected target windows onto each frozen source disease direction and
reconstructed a target boundary with a one-dimensional two-component Gaussian
mixture. It did not update the MedTS-TTT model.

The fitting command never parses `binary_label` or `rhythm_label`. It writes a
frozen score archive containing only window identity, target split, source
classifier probability, direction score, and GMM outputs. A separate evaluator
verifies the archive and artifact hashes before it reads the target index labels.

Both required protocols were run:

- **Inductive holdout (primary):** fit on target `adaptation` subjects and
  evaluate on disjoint `evaluation` subjects.
- **Transductive (secondary):** fit and evaluate on the complete target set.

The higher-mean mixture component was fixed as AF without label-assisted
component swapping. Formal GMM settings were seed 42, `n_init=20`,
`reg_covar=1e-4`, and five independent stability fits. A mixture was accepted
only when delta BIC was at least 10, pooled separation at least 2, normalized
posterior entropy at most 0.5, minimum component weight at least 0.05, and
minimum initialization agreement at least 0.98.

## Implementation and validation

The implementation is fixed at Git commit `8594e11`:

- `src/adaptation/target_gmm.py`: ordered one-/two-component GMM fitting,
  density intersection, posterior inference, and reliability diagnostics.
- `src/adaptation/target_workflow.py`: MPS target scoring, split isolation,
  label-free score archives, frozen artifacts, and provenance manifests.
- `src/evaluation/target_gmm_evaluation.py`: post-freeze label join and B2/B3/B4
  evaluation.
- `scripts/fit_target_gmm.py` and `scripts/evaluate_target_gmm.py`: deliberately
  separate fitting and evaluation commands.

Validation results:

- `python -m pytest -q` -> **44 passed**.
- Synthetic separated and unimodal GMM cases passed.
- Archive label-leakage rejection and post-freeze label joining passed.
- A test proves the target loader succeeds even if the two label fields contain
  unparsable text, demonstrating that fitting does not parse them.
- Both source-target paths passed real-data MPS smoke tests before formal runs.
- Formal run manifests record clean commit `8594e11` and no diagnostic cap.
- Frozen score archives were independently checked and contain no label field.

## Formal target runs

| Transfer | Protocol | Fit / evaluation windows | GMM threshold | Delta BIC | Separation | Norm. entropy | Reliable |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| LTAFDB -> AFDB | Inductive | 43,512 / 39,638 | 0.546055 | 60,839.64 | 5.5893 | 0.0135 | yes |
| LTAFDB -> AFDB | Transductive | 83,150 / 83,150 | -0.022623 | 86,009.21 | 6.2786 | 0.0240 | yes |
| CPSC2021 -> SHDB-AF | Inductive | 411,037 / 410,355 | -0.778745 | 917,534.40 | 2.7890 | 0.0813 | yes |
| CPSC2021 -> SHDB-AF | Transductive | 821,392 / 821,392 | -0.805690 | 2,110,859.69 | 2.3992 | 0.0899 | yes |

All four fitted mixtures passed every frozen reliability criterion. The formal
unlabelled runtimes were 352.8 seconds for AFDB and 3,484.2 seconds for SHDB-AF
on MPS.

## Primary inductive results

| Transfer / method | Accuracy | Balanced accuracy | Macro-F1 | AUROC | AUPRC | Sensitivity | Specificity | Precision | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LTAFDB -> AFDB B2 source classifier | 0.7744 | 0.7871 | 0.7741 | 0.8998 | 0.8797 | 0.8687 | 0.7054 | 0.6834 | 0.5688 |
| LTAFDB -> AFDB B3 source direction | 0.7768 | 0.7889 | 0.7764 | 0.8982 | 0.8762 | 0.8668 | 0.7110 | 0.6870 | 0.5720 |
| LTAFDB -> AFDB B4 target GMM | 0.7890 | 0.7817 | 0.7828 | 0.8828 | 0.8625 | 0.7345 | 0.8289 | 0.7586 | 0.5660 |
| CPSC2021 -> SHDB-AF B2 source classifier | 0.9108 | 0.8568 | 0.8326 | 0.9578 | 0.7866 | 0.7805 | 0.9330 | 0.6650 | 0.6686 |
| CPSC2021 -> SHDB-AF B3 source direction | 0.9114 | 0.8507 | 0.8315 | 0.9570 | 0.8016 | 0.7650 | 0.9364 | 0.6720 | 0.6653 |
| CPSC2021 -> SHDB-AF B4 target GMM | 0.8565 | 0.8941 | 0.7835 | 0.9285 | 0.5689 | 0.9471 | 0.8411 | 0.5038 | 0.6235 |

Interpretation must remain metric-specific:

- On AFDB, B4 improved accuracy, macro-F1, specificity, and precision over B3,
  but reduced balanced accuracy, AUROC, AUPRC, sensitivity, and MCC.
- On SHDB-AF, B4 strongly increased sensitivity and balanced accuracy, but
  reduced accuracy, macro-F1, ranking metrics, specificity, precision, and MCC.
- A reliable score mixture is therefore not equivalent to universal predictive
  superiority. Stage 5 demonstrates unsupervised boundary movement and a
  clinically relevant operating-point trade-off, not an across-metric win.
- GMM posterior AUROC/AUPRC may differ from direction-score AUROC/AUPRC because
  unequal component variances make the posterior a non-linear, not necessarily
  globally monotonic transform of the direction score.

## Secondary transductive results

| Transfer / method | Accuracy | Balanced accuracy | Macro-F1 | AUROC | AUPRC | Sensitivity | Specificity | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LTAFDB -> AFDB B2 | 0.8363 | 0.8514 | 0.8349 | 0.9463 | 0.9260 | 0.9272 | 0.7755 | 0.6888 |
| LTAFDB -> AFDB B3 | 0.8416 | 0.8554 | 0.8401 | 0.9466 | 0.9264 | 0.9248 | 0.7860 | 0.6967 |
| LTAFDB -> AFDB B4 | 0.8391 | 0.8540 | 0.8376 | 0.9169 | 0.8231 | 0.9291 | 0.7788 | 0.6939 |
| CPSC2021 -> SHDB-AF B2 | 0.9105 | 0.8879 | 0.8669 | 0.9652 | 0.8734 | 0.8505 | 0.9254 | 0.7372 |
| CPSC2021 -> SHDB-AF B3 | 0.9122 | 0.8848 | 0.8680 | 0.9647 | 0.8822 | 0.8393 | 0.9302 | 0.7382 |
| CPSC2021 -> SHDB-AF B4 | 0.8356 | 0.8871 | 0.7942 | 0.9190 | 0.6200 | 0.9726 | 0.8016 | 0.6470 |

## Artifacts

- `outputs/stage5/ltafdb_to_afdb_gmm/`
- `outputs/stage5/cpsc2021_to_shdb-af_gmm/`

Each directory contains `target_scores.npz`, `gmm_artifact.json`,
`run_manifest.json`, `fit_result.json`, and `evaluation_result.json`.

Verified score hashes:

- AFDB: `6cca3f683fa67d60a1146b89202012726b9a32d0d96869aa0f6864195a6b44c8`
- SHDB-AF: `4323f02ad5d35369f210319b8bf5d3973a13b779e4fa9c0c08f510276eb24c93`

## Next stage

Stage 5 is complete. The next implementation step is Stage 6: add supervised
contrastive loss as an independently switchable source-only representation
experiment, validate it against a small reference calculation and tiny overfit,
then add prototype/center loss separately. Existing B2/B3/B4 results remain
frozen and must not be used to tune Stage 6 hyperparameters on target labels.
