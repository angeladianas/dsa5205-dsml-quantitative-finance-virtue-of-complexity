"""Unit tests verifying mathematical correctness of src/metrics.py."""

import numpy as np

from src.metrics import (
    compute_optimal_shrinkage_theory,
    compute_param_norm,
    compute_r2_paper,
    compute_sharpe_uncentered,
    compute_timing_returns,
)


def test_r2_paper_perfect_prediction():
    """Verify R^2_paper is exactly 1.0 when predictions match realized returns."""
    np.random.seed(42)
    y = np.random.randn(500)
    r2 = compute_r2_paper(y, y)
    assert np.isclose(r2, 1.0, atol=1e-12)


def test_r2_paper_zero_prediction():
    """Verify R^2_paper is 0.0 when predictions are all zero."""
    np.random.seed(42)
    y = np.random.randn(500)
    y_pred = np.zeros_like(y)
    r2 = compute_r2_paper(y, y_pred)
    assert np.isclose(r2, 0.0, atol=1e-12)


def test_r2_paper_negative_variance_explosion():
    """Verify R^2_paper can be deeply negative when prediction variance explodes."""
    np.random.seed(42)
    y = np.random.randn(500)
    # Scaled noisy prediction
    y_pred = y + 10.0 * np.random.randn(500)
    r2 = compute_r2_paper(y, y_pred)
    assert r2 < -10.0


def test_r2_paper_formula_equivalence():
    """Verify KMZ Prop 3 numerator identity: 1 - E[(y - yhat)^2]/E[y^2] == (2E[y*yhat] - E[yhat^2])/E[y^2]."""
    np.random.seed(42)
    y = np.random.randn(1000)
    y_pred = 0.5 * y + 0.8 * np.random.randn(1000)

    standard_mse_r2 = 1.0 - np.mean((y - y_pred) ** 2) / np.mean(y**2)
    kmz_r2 = compute_r2_paper(y, y_pred)

    assert np.isclose(standard_mse_r2, kmz_r2, atol=1e-12)


def test_timing_returns():
    r"""Verify timing portfolio returns R^\pi_{t+1} = \hat{R}_{t+1} * R_{t+1}."""
    y = np.array([1.0, -2.0, 3.0])
    y_pred = np.array([0.5, -0.5, 2.0])
    expected = np.array([0.5, 1.0, 6.0])
    assert np.allclose(compute_timing_returns(y, y_pred), expected)


def test_sharpe_uncentered_scale_invariance():
    """Verify uncentered Sharpe ratio is scale invariant under positive scalar multiplication."""
    np.random.seed(42)
    r_pi = np.random.randn(1000) + 0.2
    sr1 = compute_sharpe_uncentered(r_pi)
    sr2 = compute_sharpe_uncentered(5.0 * r_pi)
    assert np.isclose(sr1, sr2, atol=1e-12)


def test_param_norm():
    """Verify Euclidean parameter norm squared."""
    beta = np.array([3.0, 4.0])
    assert np.isclose(compute_param_norm(beta), 25.0)


def test_optimal_shrinkage_theory():
    """Verify theoretical optimal shrinkage matches KMZ Prop 3 and Prop 6(ii)."""
    c = 10.0
    b_star = 0.25

    # Correctly specified case (q = 1.0) -> z* = c / b*
    z_q1 = compute_optimal_shrinkage_theory(c_dgp=c, b_star=b_star, q=1.0)
    assert np.isclose(z_q1, c / b_star)
    assert np.isclose(z_q1, 40.0)

    # Misspecified case q = 0.5 -> z* = c * (1 + b*(1-q)) / b*
    # z* = 10 * (1 + 0.25 * 0.5) / 0.25 = 10 * (1.125) * 4 = 45.0
    z_q05 = compute_optimal_shrinkage_theory(c_dgp=c, b_star=b_star, q=0.5)
    assert np.isclose(z_q05, 45.0)
