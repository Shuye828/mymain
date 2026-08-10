import numpy as np

from src.analysis.ltaf_signal_quality import (
    deterministic_window_starts,
    quality_metrics_by_channel,
)


def _metrics(signal: np.ndarray, fs: float = 128.0) -> dict[str, float]:
    return quality_metrics_by_channel(
        signal[None, :],
        fs=fs,
        extreme_amplitude_mv=5.0,
        extreme_first_difference_mv=1.0,
        high_frequency_band_hz=(20.0, 40.0),
        reference_power_band_hz=(0.5, 40.0),
    )[0]


def test_quality_metrics_detect_flatline_extreme_and_nonfinite() -> None:
    fs = 128.0
    time = np.arange(int(10 * fs)) / fs
    clean = np.sin(2 * np.pi * 5 * time)
    flat = np.zeros_like(clean)
    extreme = clean.copy()
    extreme[100:110] = 10.0
    nonfinite = clean.copy()
    nonfinite[0] = np.nan

    clean_metrics = _metrics(clean)
    flat_metrics = _metrics(flat)
    extreme_metrics = _metrics(extreme)
    nonfinite_metrics = _metrics(nonfinite)

    assert flat_metrics["flatline_ratio"] > clean_metrics["flatline_ratio"]
    assert extreme_metrics["extreme_amplitude_ratio"] > 0
    assert extreme_metrics["extreme_first_difference_ratio"] > 0
    assert nonfinite_metrics["finite_value_ratio"] < 1


def test_high_frequency_ratio_increases_for_30_hz_signal() -> None:
    fs = 128.0
    time = np.arange(int(10 * fs)) / fs
    low = np.sin(2 * np.pi * 5 * time)
    high = np.sin(2 * np.pi * 30 * time)

    assert (
        _metrics(high)["high_frequency_power_ratio"]
        > _metrics(low)["high_frequency_power_ratio"]
    )


def test_deterministic_window_selection_is_order_independent() -> None:
    candidates = list(range(3600, 10000, 10))
    first = deterministic_window_starts(candidates, count=20, seed=42, record_id="00")
    second = deterministic_window_starts(
        reversed(candidates), count=20, seed=42, record_id="00"
    )

    assert first == second
    assert len(first) == 20
    assert first == sorted(first)
