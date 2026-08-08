"""On-demand ECG preprocessing applied after source segment reads."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


@dataclass(frozen=True)
class PreprocessingConfig:
    target_fs: float = 200.0
    low_hz: float = 0.5
    high_hz: float = 40.0
    filter_order: int = 4
    expected_channels: int = 2
    expected_duration_seconds: float = 10.0


def validate_raw_segment(signal: np.ndarray, expected_channels: int = 2) -> None:
    if signal.ndim != 2:
        raise ValueError(f"ECG segment must be [C,T], got {signal.shape}")
    if signal.shape[0] != expected_channels:
        raise ValueError(
            f"expected {expected_channels} channels, got {signal.shape[0]}"
        )
    if signal.shape[1] == 0:
        raise ValueError("ECG segment is empty")
    if not np.isfinite(signal).all():
        raise ValueError("ECG segment contains NaN or Inf")


def bandpass_filter(
    signal: np.ndarray,
    *,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> np.ndarray:
    """Zero-phase Butterworth filtering in physical units."""

    if fs <= 0:
        raise ValueError("sampling frequency must be positive")
    nyquist = fs / 2.0
    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError(
            f"invalid bandpass {low_hz}-{high_hz} Hz for fs={fs} Hz"
        )
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    try:
        return sosfiltfilt(sos, signal, axis=-1)
    except ValueError as exc:
        raise ValueError(
            f"segment too short for zero-phase bandpass filtering: {signal.shape}"
        ) from exc


def resample_to_fs(signal: np.ndarray, *, fs: float, target_fs: float) -> np.ndarray:
    if fs <= 0 or target_fs <= 0:
        raise ValueError("sampling frequencies must be positive")
    if np.isclose(fs, target_fs):
        return np.array(signal, copy=True)
    ratio = Fraction(str(target_fs)) / Fraction(str(fs))
    return resample_poly(
        signal,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=-1,
    )


def preprocess_ecg(
    signal: np.ndarray,
    *,
    fs: float,
    config: PreprocessingConfig = PreprocessingConfig(),
) -> np.ndarray:
    """Filter then resample, without any z-score normalization."""

    raw = np.asarray(signal)
    validate_raw_segment(raw, config.expected_channels)
    expected_source_length = int(round(config.expected_duration_seconds * fs))
    if raw.shape[1] != expected_source_length:
        raise ValueError(
            f"expected {expected_source_length} source samples for "
            f"{config.expected_duration_seconds}s at {fs}Hz, got {raw.shape[1]}"
        )
    filtered = bandpass_filter(
        raw.astype(np.float64, copy=False),
        fs=fs,
        low_hz=config.low_hz,
        high_hz=config.high_hz,
        order=config.filter_order,
    )
    resampled = resample_to_fs(filtered, fs=fs, target_fs=config.target_fs)
    expected_target_length = int(
        round(config.expected_duration_seconds * config.target_fs)
    )
    if resampled.shape != (config.expected_channels, expected_target_length):
        raise ValueError(
            f"unexpected resampled shape {resampled.shape}; expected "
            f"({config.expected_channels}, {expected_target_length})"
        )
    output = resampled.astype(np.float32, copy=False)
    validate_raw_segment(output, config.expected_channels)
    return output
