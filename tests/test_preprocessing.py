import numpy as np
import pytest

from src.data.preprocessing import PreprocessingConfig, preprocess_ecg


@pytest.mark.parametrize("fs", [128.0, 200.0, 250.0])
def test_preprocess_produces_two_by_two_thousand(fs: float) -> None:
    time = np.arange(int(fs * 10)) / fs
    signal = np.stack(
        [
            np.sin(2 * np.pi * 5 * time),
            0.5 * np.cos(2 * np.pi * 7 * time),
        ]
    )

    output = preprocess_ecg(signal, fs=fs)

    assert output.shape == (2, 2000)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()


def test_preprocess_rejects_nonfinite_input() -> None:
    signal = np.zeros((2, 2000))
    signal[0, 10] = np.nan

    with pytest.raises(ValueError, match="NaN or Inf"):
        preprocess_ecg(signal, fs=200)
