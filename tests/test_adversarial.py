"""Adversarial stress-testing suite for numerical stability, edge cases, and theoretical invariants.

Created by CodeReviewerAdversaryAgent to act as Devil's Advocate for DSA5205 Project 1.
"""

import numpy as np
import pytest

from src.metrics import (
    compute_optimal_shrinkage_theory,
    compute_param_norm,
    compute_r2_paper,
    compute_sharpe_uncentered,
    compute_sharpe_var,
    compute_timing_returns,
)
from src.solvers import solve_ridge_svd
from src.task4_dynamics import TwoLayerNetwork


class TestAdversarialSolvers:
    """Devil's advocate stress tests for SVD ridge and Moore-Penrose solvers."""

    def test_rank_deficient_collinear_features(self):
        """Stress test: design matrix with duplicate/collinear columns and zero columns."""
        np.random.seed(101)
        t_tr = 50
        p = 30
        X_base = np.random.randn(t_tr, 10)
        # Duplicate columns, zero column, and linear combinations
        X = np.zeros((t_tr, p))
        X[:, :10] = X_base
        X[:, 10:20] = X_base  # Exact duplicates
        X[:, 20] = 0.0  # Zero column
        X[:, 21:30] = X_base[:, :9] * 3.5  # Scaled duplicates

        y = np.random.randn(t_tr)

        # 1. Ridgeless (z=0) pseudoinverse solver
        beta_0 = solve_ridge_svd(X, y, z=0.0, t_tr=t_tr)
        assert np.all(np.isfinite(beta_0)), (
            "z=0 solver produced non-finite values on rank-deficient X"
        )
        # Zero column must have exactly zero weight
        assert np.isclose(beta_0[20], 0.0, atol=1e-14)

        # 2. Regularized (z > 0)
        for z in [0.01, 1.0, 50.0]:
            beta_z = solve_ridge_svd(X, y, z=z, t_tr=t_tr)
            assert np.all(np.isfinite(beta_z)), f"z={z} solver produced non-finite values"
            assert np.isclose(beta_z[20], 0.0, atol=1e-14)

    def test_extreme_ill_conditioning_at_boundary(self):
        """Stress test: c_q = 1.0 with condition number kappa(X) >= 10^12."""
        np.random.seed(202)
        n = 40
        # Construct synthetic singular value spectrum decaying from 1 down to 1e-12
        U, _ = np.linalg.qr(np.random.randn(n, n))
        V, _ = np.linalg.qr(np.random.randn(n, n))
        s = np.logspace(0, -12, n)
        X = U @ np.diag(s) @ V.T
        y = np.random.randn(n)

        # z=0 must not produce overflow or NaN
        beta_0 = solve_ridge_svd(X, y, z=0.0, t_tr=n)
        assert np.all(np.isfinite(beta_0))

        # Shrinkage dampening test: positive z must strictly compress parameter norm
        norm_0 = compute_param_norm(beta_0)
        for z in [0.1, 1.0, 10.0]:
            beta_z = solve_ridge_svd(X, y, z=z, t_tr=n)
            norm_z = compute_param_norm(beta_z)
            assert norm_z < norm_0, f"Shrinkage failed to compress norm at z={z}"

    def test_asymptotic_regularization_limits(self):
        """Stress test: z -> inf must drive beta -> 0; z = 0 matches minimum norm."""
        np.random.seed(303)
        t_tr, p = 30, 50  # Overparameterized
        X = np.random.randn(t_tr, p)
        y = np.random.randn(t_tr)

        # z -> infinity
        beta_huge_z = solve_ridge_svd(X, y, z=1e8, t_tr=t_tr)
        assert np.all(np.abs(beta_huge_z) < 1e-6), "Huge z did not shrink coefficients toward zero"

        # z = 0 matches pinv
        beta_0 = solve_ridge_svd(X, y, z=0.0, t_tr=t_tr)
        beta_pinv = np.linalg.pinv(X) @ y
        assert np.allclose(beta_0, beta_pinv, atol=1e-8)


class TestAdversarialMetrics:
    """Devil's advocate stress tests for financial evaluation metrics."""

    def test_degenerate_zero_and_constant_returns(self):
        """Stress test: handle all-zero and constant arrays without ZeroDivisionError or NaN."""
        zeros = np.zeros(200)
        ones = np.ones(200)

        # R^2 with all-zero actual returns
        r2_zero = compute_r2_paper(zeros, ones)
        assert np.isclose(r2_zero, 0.0), "Zero denominator should safely return 0.0"

        # Sharpe ratio with all-zero timing returns
        sr_uncentered_zero = compute_sharpe_uncentered(zeros)
        assert np.isclose(sr_uncentered_zero, 0.0), "Zero timing returns must produce 0.0 Sharpe"

        sr_var_zero = compute_sharpe_var(zeros)
        assert np.isclose(sr_var_zero, 0.0), "Zero variance timing returns must produce 0.0 Sharpe"

        # Constant timing returns (variance = 0)
        sr_var_const = compute_sharpe_var(ones)
        assert np.isclose(sr_var_const, 0.0), (
            "Constant timing returns must produce 0.0 variance Sharpe"
        )

    def test_adversarial_counter_trend_predictions(self):
        """Stress test: perfectly inverted predictions y_pred = -y_true."""
        np.random.seed(404)
        y = np.random.randn(500)
        y_inverted = -y

        # By formula: 1 - E[(y - (-y))^2] / E[y^2] = 1 - 4 = -3.0
        r2 = compute_r2_paper(y, y_inverted)
        assert np.isclose(r2, -3.0, atol=1e-12)

        # Timing returns are strictly non-positive (-y^2 <= 0)
        r_pi = compute_timing_returns(y, y_inverted)
        assert np.all(r_pi <= 0.0)
        assert compute_sharpe_uncentered(r_pi) < 0.0

    def test_optimal_shrinkage_boundary_conditions(self):
        """Stress test: illegal or boundary parameters in theoretical formula."""
        with pytest.raises(ValueError):
            compute_optimal_shrinkage_theory(c_dgp=10.0, b_star=-0.5, q=1.0)

        with pytest.raises(ValueError):
            compute_optimal_shrinkage_theory(c_dgp=10.0, b_star=0.0, q=1.0)


class TestAdversarialTask4Dynamics:
    """Devil's advocate stress tests for neural network dynamics under pathological inputs."""

    def test_neural_network_zero_input_stability(self):
        """Stress test: forward pass and backward step on all-zero design matrix."""
        net = TwoLayerNetwork(d=20, m=50, lr=0.01)
        X_zero = np.zeros((30, 20))
        y_rand = np.random.randn(30)

        y_hat, h = net.forward(X_zero)
        assert np.all(np.isfinite(y_hat))
        assert np.all(np.isfinite(h))

        loss = net.train_step(X_zero, y_rand)
        assert np.isfinite(loss)
        assert np.all(np.isfinite(net.W))
        assert np.all(np.isfinite(net.a))

    def test_first_layer_sphere_normalization_invariant(self):
        """Verify that every row of W remains strictly normalized to unit norm after training steps."""
        net = TwoLayerNetwork(d=15, m=40, lr=0.1)
        X = np.random.randn(50, 15)
        y = np.random.randn(50)

        for _ in range(10):
            net.train_step(X, y)
            row_norms = np.linalg.norm(net.W, axis=1)
            assert np.allclose(row_norms, 1.0, atol=1e-12), (
                "W row normalization violated during training"
            )
