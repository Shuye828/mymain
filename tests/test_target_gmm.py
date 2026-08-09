import numpy as np

from src.adaptation.target_gmm import apply_gmm_boundary, fit_target_gmm


def test_target_gmm_recovers_ordered_synthetic_mixture() -> None:
    rng = np.random.default_rng(11)
    scores = np.concatenate(
        [rng.normal(-1.2, 0.18, 600), rng.normal(0.9, 0.22, 400)]
    )

    artifact = fit_target_gmm(
        scores,
        random_state=42,
        n_init=5,
        reg_covar=1e-4,
        stability_runs=3,
    )
    probabilities = apply_gmm_boundary(scores, artifact)

    assert artifact["labels_accessed"] is False
    assert artifact["frozen"] is True
    assert artifact["two_component"]["means"][0] < 0
    assert artifact["two_component"]["means"][1] > 0
    assert artifact["delta_bic"] > 10
    assert artifact["pooled_separation"] > 2
    assert artifact["reliable"] is True
    assert probabilities[:600].mean() < 0.01
    assert probabilities[600:].mean() > 0.99


def test_target_gmm_records_unreliable_unimodal_structure() -> None:
    rng = np.random.default_rng(9)
    scores = rng.normal(0.0, 1.0, 1000)

    artifact = fit_target_gmm(
        scores,
        random_state=42,
        n_init=3,
        stability_runs=2,
        reliability={"min_pooled_separation": 10.0},
    )

    assert artifact["reliable"] is False
    assert "pooled_separation_below_minimum" in artifact[
        "reliability_failures"
    ]
    assert (
        artifact["two_component"]["higher_mean_component_semantics"] == "AF"
    )


def test_gmm_probability_increases_toward_high_mean_component() -> None:
    artifact = {
        "two_component": {
            "means": [-1.0, 1.0],
            "variances": [0.25, 0.25],
            "weights": [0.5, 0.5],
        }
    }

    probabilities = apply_gmm_boundary(np.array([-2.0, 0.0, 2.0]), artifact)

    assert np.all(np.diff(probabilities) > 0)
    assert np.isclose(probabilities[1], 0.5)
