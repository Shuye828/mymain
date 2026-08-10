"""Label-free signal-quality metrics for the LTAFDB clean-1h revision."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
from scipy.signal import welch


QUALITY_METRICS = (
    "finite_value_ratio",
    "flatline_ratio",
    "extreme_amplitude_ratio",
    "extreme_first_difference_ratio",
    "high_frequency_power_ratio",
)


def deterministic_window_starts(
    candidates: Iterable[int], *, count: int, seed: int, record_id: str
) -> list[int]:
    """Select starts by a stable label- and signal-independent hash."""

    if count <= 0:
        raise ValueError("window selection count must be positive")
    ranked = []
    for start in candidates:
        start = int(start)
        payload = f"{seed}:{record_id}:{start}".encode("utf-8")
        priority = int.from_bytes(
            hashlib.blake2b(payload, digest_size=8).digest(), "big"
        )
        ranked.append((priority, start))
    ranked.sort()
    return sorted(start for _, start in ranked[:count])


def quality_metrics_by_channel(
    signal: np.ndarray,
    *,
    fs: float,
    extreme_amplitude_mv: float,
    extreme_first_difference_mv: float,
    high_frequency_band_hz: tuple[float, float],
    reference_power_band_hz: tuple[float, float],
) -> list[dict[str, float]]:
    """Compute five raw-signal metrics without rhythm or class information."""

    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("quality signal must have shape [C,T] with T >= 2")
    if fs <= 0:
        raise ValueError("quality sampling frequency must be positive")
    if extreme_amplitude_mv <= 0 or extreme_first_difference_mv <= 0:
        raise ValueError("quality amplitude thresholds must be positive")
    hf_low, hf_high = map(float, high_frequency_band_hz)
    ref_low, ref_high = map(float, reference_power_band_hz)
    if not 0 <= ref_low < hf_low < hf_high <= ref_high < fs / 2:
        raise ValueError("quality frequency bands are invalid for the sample rate")

    results = []
    for channel in values:
        finite = np.isfinite(channel)
        finite_ratio = float(finite.mean())
        finite_pairs = finite[:-1] & finite[1:]
        differences = np.diff(channel)
        if finite_pairs.any():
            flatline_ratio = float(np.mean(differences[finite_pairs] == 0.0))
            extreme_difference_ratio = float(
                np.mean(np.abs(differences[finite_pairs]) > extreme_first_difference_mv)
            )
        else:
            flatline_ratio = float("nan")
            extreme_difference_ratio = float("nan")
        if finite.any():
            extreme_amplitude_ratio = float(
                np.mean(np.abs(channel[finite]) > extreme_amplitude_mv)
            )
        else:
            extreme_amplitude_ratio = float("nan")

        if finite.all():
            centered = channel - channel.mean()
            frequencies, power = welch(
                centered,
                fs=fs,
                nperseg=min(channel.size, max(256, int(round(4 * fs)))),
            )
            reference_mask = (frequencies >= ref_low) & (frequencies <= ref_high)
            high_mask = (frequencies >= hf_low) & (frequencies <= hf_high)
            reference_power = float(power[reference_mask].sum())
            high_power = float(power[high_mask].sum())
            high_frequency_ratio = (
                high_power / reference_power if reference_power > 0 else 0.0
            )
        else:
            high_frequency_ratio = float("nan")

        results.append(
            {
                "finite_value_ratio": finite_ratio,
                "flatline_ratio": flatline_ratio,
                "extreme_amplitude_ratio": extreme_amplitude_ratio,
                "extreme_first_difference_ratio": extreme_difference_ratio,
                "high_frequency_power_ratio": float(high_frequency_ratio),
            }
        )
    return results
