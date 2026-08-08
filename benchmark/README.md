# Benchmark Compatibility

This project intentionally does not copy the full benchmark code from
Medformer or MedTS_Evaluation. Instead, it provides a small adapter so MedTS-TTT
can be evaluated inside the existing medical time-series classification
ecosystem.

## Recommended Benchmark

We recommend using the processed data format and subject-independent setting
from:

- Medformer: https://github.com/DL4mHealth/Medformer
- MedTS_Evaluation: https://github.com/DL4mHealth/MedTS_Evaluation

These repositories organize processed data as:

```text
dataset/DATA_NAME/
  Feature/
    feature_ID.npy
  Label/
    label.npy
```

The feature file stores samples from one subject. In the Medformer dataloader,
the model input is usually provided as `[B, T, C]`.

## Add MedTS-TTT To Medformer

1. Copy the model implementation to the Medformer repository root:

```bash
cp MedTS_TTT.py /path/to/Medformer/MedTS_TTT.py
```

2. Copy the adapter to the Medformer `models/` directory:

```bash
cp benchmark/Medformer/MedTSTTT.py /path/to/Medformer/models/MedTSTTT.py
```

3. Register the model in `exp/exp_basic.py`.

Add `MedTSTTT` to the import list:

```python
from models import (
    Autoformer,
    Crossformer,
    FEDformer,
    Informer,
    iTransformer,
    MTST,
    Nonstationary_Transformer,
    PatchTST,
    Reformer,
    Transformer,
    TCN,
    Medformer,
    MedTSTTT,
)
```

Add it to `self.model_dict`:

```python
self.model_dict = {
    ...
    "Medformer": Medformer,
    "MedTSTTT": MedTSTTT,
}
```

4. Run Medformer with `--model MedTSTTT`.

Example:

```bash
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/ADFTD/ \
  --model_id ADFTD-Indep \
  --model MedTSTTT \
  --data ADFTD \
  --seq_len 256 \
  --enc_in 19 \
  --num_class 3 \
  --d_model 128 \
  --n_heads 8 \
  --e_layers 6 \
  --patch_len 8 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  --des Exp
```

Please follow the dataset-specific `seq_len`, `enc_in`, and `num_class` values
used by your benchmark scripts.

## Paper Setting

The paper reports subject-independent evaluation on four public clinical
datasets:

| Modality | Dataset | Task |
| --- | --- | --- |
| EEG | APAVA | Alzheimer's disease detection |
| EEG | ADFTD | HC / FTD / AD classification |
| ECG | PTB | myocardial infarction detection |
| ECG | PTB-XL | cardiac condition classification |

The main implementation setting is:

| Parameter | Value |
| --- | --- |
| hidden dimension | 128 |
| layers | 6 |
| patch size | 8 |
| optimizer | Adam |
| learning rate | 1e-4 |
| early stopping | validation macro-F1 patience 10 |
