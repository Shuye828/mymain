"""Label-free one-dimensional GMM boundary reconstruction."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from sklearn.mixture import GaussianMixture


def _as_scores(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or scores.size < 4:
        raise ValueError("GMM scores must be a 1-D array with at least 4 samples")
    if not np.isfinite(scores).all():
        raise ValueError("GMM scores contain NaN or Inf")
    if float(scores.std()) <= 1e-12:
        raise ValueError("GMM scores have zero variance")
    return scores


def _ordered_parameters(model: GaussianMixture) -> dict[str, list[float]]:
    means = model.means_.reshape(-1)
    variances = model.covariances_.reshape(-1)
    weights = model.weights_.reshape(-1)
    order = np.argsort(means)
    return {
        "means": [float(means[index]) for index in order],
        "variances": [float(variances[index]) for index in order],
        "weights": [float(weights[index]) for index in order],
    }


def _log_density_difference(x: float, parameters: dict) -> float:
    means = parameters["means"]
    variances = parameters["variances"]
    weights = parameters["weights"]
    low = (
        math.log(weights[0])
        - 0.5 * math.log(2.0 * math.pi * variances[0])
        - ((x - means[0]) ** 2) / (2.0 * variances[0])
    )
    high = (
        math.log(weights[1])
        - 0.5 * math.log(2.0 * math.pi * variances[1])
        - ((x - means[1]) ** 2) / (2.0 * variances[1])
    )
    return low - high


def _density_intersection(parameters: dict) -> float:
    low_mean, high_mean = parameters["means"]
    low_value = _log_density_difference(low_mean, parameters)
    high_value = _log_density_difference(high_mean, parameters)
    if low_value == 0:
        return float(low_mean)
    if high_value == 0:
        return float(high_mean)
    if low_value * high_value < 0:
        return float(
            brentq(
                lambda value: _log_density_difference(value, parameters),
                low_mean,
                high_mean,
            )
        )
    result = minimize_scalar(
        lambda value: abs(_log_density_difference(value, parameters)),
        bounds=(low_mean, high_mean),
        method="bounded",
    )
    if not result.success:
        raise RuntimeError("could not resolve the GMM density intersection")
    return float(result.x)


def apply_gmm_boundary(scores: np.ndarray, artifact: dict) -> np.ndarray:
    """Return posterior probability of the ordered higher-mean component."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite 1-D array")
    parameters = artifact["two_component"]
    means = np.asarray(parameters["means"], dtype=np.float64)
    variances = np.asarray(parameters["variances"], dtype=np.float64)
    weights = np.asarray(parameters["weights"], dtype=np.float64)
    if means.shape != variances.shape or means.shape != weights.shape or means.shape != (2,):
        raise ValueError("GMM artifact must contain two ordered components")
    log_joint = np.stack(
        [
            np.log(weights[index])
            - 0.5 * np.log(2.0 * np.pi * variances[index])
            - ((values - means[index]) ** 2) / (2.0 * variances[index])
            for index in range(2)
        ],
        axis=1,
    )
    maximum = log_joint.max(axis=1, keepdims=True)
    posterior = np.exp(log_joint - maximum)
    posterior /= posterior.sum(axis=1, keepdims=True)
    probabilities = posterior[:, 1]
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("GMM posterior contains NaN or Inf")
    return probabilities


def fit_target_gmm(
    scores: np.ndarray,
    *,
    random_state: int = 42,
    n_init: int = 20,
    reg_covar: float = 1e-4,
    stability_runs: int = 5,
    reliability: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Fit label-free one-/two-component GMMs and reliability diagnostics."""

    values = _as_scores(scores)
    if n_init <= 0 or stability_runs <= 0:
        raise ValueError("n_init and stability_runs must be positive")
    if reg_covar <= 0:
        raise ValueError("reg_covar must be positive")
    matrix = values[:, None]
    one = GaussianMixture(
        n_components=1,
        covariance_type="full",
        n_init=n_init,
        reg_covar=reg_covar,
        random_state=random_state,
    ).fit(matrix)
    two = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=n_init,
        reg_covar=reg_covar,
        random_state=random_state,
    ).fit(matrix)
    parameters = _ordered_parameters(two)
    threshold = _density_intersection(parameters)
    base_artifact = {"two_component": parameters}
    probabilities = apply_gmm_boundary(values, base_artifact)
    predictions = probabilities >= 0.5
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    entropy = -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))
    mean_entropy = float(entropy.mean())
    normalized_entropy = mean_entropy / math.log(2.0)
    mean_gap = parameters["means"][1] - parameters["means"][0]
    pooled_std = math.sqrt(
        (parameters["variances"][0] + parameters["variances"][1]) / 2.0
    )
    pooled_separation = mean_gap / pooled_std

    stability_details: list[dict[str, Any]] = []
    agreements: list[float] = []
    thresholds: list[float] = []
    ordered_means: list[list[float]] = []
    for offset in range(stability_runs):
        seed = random_state + offset
        candidate = GaussianMixture(
            n_components=2,
            covariance_type="full",
            n_init=1,
            reg_covar=reg_covar,
            random_state=seed,
        ).fit(matrix)
        candidate_parameters = _ordered_parameters(candidate)
        candidate_artifact = {"two_component": candidate_parameters}
        candidate_probabilities = apply_gmm_boundary(values, candidate_artifact)
        agreement = float(np.mean((candidate_probabilities >= 0.5) == predictions))
        candidate_threshold = _density_intersection(candidate_parameters)
        agreements.append(agreement)
        thresholds.append(candidate_threshold)
        ordered_means.append(candidate_parameters["means"])
        stability_details.append(
            {
                "random_state": seed,
                "converged": bool(candidate.converged_),
                "means": candidate_parameters["means"],
                "threshold": candidate_threshold,
                "prediction_agreement": agreement,
            }
        )

    bic_one = float(one.bic(matrix))
    bic_two = float(two.bic(matrix))
    delta_bic = bic_one - bic_two
    criteria = {
        "min_delta_bic": 10.0,
        "min_pooled_separation": 2.0,
        "max_normalized_entropy": 0.5,
        "min_component_weight": 0.05,
        "min_initialization_agreement": 0.98,
    }
    if reliability is not None:
        criteria.update({key: float(value) for key, value in reliability.items()})
    failures: list[str] = []
    if not bool(one.converged_) or not bool(two.converged_):
        failures.append("gmm_not_converged")
    if delta_bic < criteria["min_delta_bic"]:
        failures.append("delta_bic_below_minimum")
    if pooled_separation < criteria["min_pooled_separation"]:
        failures.append("pooled_separation_below_minimum")
    if normalized_entropy > criteria["max_normalized_entropy"]:
        failures.append("posterior_entropy_above_maximum")
    if min(parameters["weights"]) < criteria["min_component_weight"]:
        failures.append("component_weight_below_minimum")
    if min(agreements) < criteria["min_initialization_agreement"]:
        failures.append("initialization_agreement_below_minimum")

    return {
        "frozen": True,
        "labels_accessed": False,
        "sample_count": int(values.size),
        "random_state": int(random_state),
        "n_init": int(n_init),
        "reg_covar": float(reg_covar),
        "one_component": {
            "mean": float(one.means_.reshape(-1)[0]),
            "variance": float(one.covariances_.reshape(-1)[0]),
            "bic": bic_one,
            "converged": bool(one.converged_),
            "iterations": int(one.n_iter_),
        },
        "two_component": {
            **parameters,
            "bic": bic_two,
            "converged": bool(two.converged_),
            "iterations": int(two.n_iter_),
            "higher_mean_component_semantics": "AF",
        },
        "delta_bic": delta_bic,
        "mean_gap": mean_gap,
        "pooled_separation": pooled_separation,
        "posterior_entropy": mean_entropy,
        "normalized_posterior_entropy": normalized_entropy,
        "density_intersection_threshold": threshold,
        "posterior_threshold": 0.5,
        "predicted_af_fraction": float(predictions.mean()),
        "initialization_stability": {
            "runs": int(stability_runs),
            "minimum_prediction_agreement": min(agreements),
            "mean_prediction_agreement": float(np.mean(agreements)),
            "threshold_std": float(np.std(thresholds)),
            "ordered_mean_std": np.std(
                np.asarray(ordered_means, dtype=np.float64), axis=0
            ).tolist(),
            "details": stability_details,
        },
        "reliability_criteria": criteria,
        "reliable": not failures,
        "reliability_failures": failures,
    }
