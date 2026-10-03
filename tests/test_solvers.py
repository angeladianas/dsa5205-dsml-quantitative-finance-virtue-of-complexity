"""Unit tests verifying numerical stability and equivalence of src/solvers.py."""

import numpy as np
from sklearn.linear_model import Ridge

from src.solvers import batch_solve_ridge_svd, solve_ridge_svd


def test_svd_solver_ols_equivalence_underparameterized():
    """Verify solve_ridge_svd matches standard OLS when P_1 < T_tr and z = 0."""
    np.random.seed(42)
    t_tr, p1 = 100, 25
    X = np.random.randn(t_tr, p1)
    beta_true = np.random.randn(p1)
    y = X @ beta_true + 0.1 * np.random.randn(t_tr)

    # Solve with SVD
    beta_svd = solve_ridge_svd(X, y, z=0.0, t_tr=t_tr)

    # Solve with standard numpy least squares
    beta_ols, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    assert np.allclose(beta_svd, beta_ols, atol=1e-10)


def test_svd_solver_ridge_equivalence_underparameterized():
    """Verify solve_ridge_svd matches sklearn Ridge when z > 0."""
    np.random.seed(42)
    t_tr, p1 = 120, 40
    X = np.random.randn(t_tr, p1)
    y = np.random.randn(t_tr)
    z = 2.5

    # Our SVD solver scales penalty as z * T_tr
    beta_svd = solve_ridge_svd(X, y, z=z, t_tr=t_tr)

    # Scikit-learn Ridge minimizes ||y - X*beta||^2 + alpha * ||beta||^2
    # Setting alpha = z * t_tr makes objectives identical
    clf = Ridge(alpha=z * float(t_tr), fit_intercept=False, solver="svd")
    clf.fit(X, y)
    beta_sklearn = clf.coef_

    assert np.allclose(beta_svd, beta_sklearn, atol=1e-8)


def test_svd_solver_minimum_norm_overparameterized():
    """Verify solve_ridge_svd interpolates and achieves minimum L2 norm when P_1 > T_tr and z = 0."""
    np.random.seed(42)
    t_tr, p1 = 40, 100  # Overparameterized c_q = 2.5
    X = np.random.randn(t_tr, p1)
    y = np.random.randn(t_tr)

    # Solve with SVD (z = 0)
    beta_svd = solve_ridge_svd(X, y, z=0.0, t_tr=t_tr)

    # 1. Must perfectly interpolate: X @ beta == y
    residuals = y - X @ beta_svd
    assert np.linalg.norm(residuals) < 1e-10

    # 2. Must match Moore-Penrose pseudoinverse minimum-norm solution
    beta_pinv = np.linalg.pinv(X) @ y
    assert np.allclose(beta_svd, beta_pinv, atol=1e-10)


def test_svd_solver_ridge_overparameterized():
    """Verify solve_ridge_svd matches sklearn Ridge in overparameterized regime."""
    np.random.seed(42)
    t_tr, p1 = 50, 150
    X = np.random.randn(t_tr, p1)
    y = np.random.randn(t_tr)
    z = 1.0

    beta_svd = solve_ridge_svd(X, y, z=z, t_tr=t_tr)

    clf = Ridge(alpha=z * float(t_tr), fit_intercept=False, solver="svd")
    clf.fit(X, y)
    beta_sklearn = clf.coef_

    assert np.allclose(beta_svd, beta_sklearn, atol=1e-8)


def test_batch_solve_ridge_svd():
    """Verify batch_solve_ridge_svd gives identical results to individual calls."""
    np.random.seed(42)
    t_tr, p1 = 60, 80
    X = np.random.randn(t_tr, p1)
    y = np.random.randn(t_tr)
    z_grid = [0.0, 0.5, 2.0, 10.0]

    batch_res = batch_solve_ridge_svd(X, y, z_grid=z_grid, t_tr=t_tr)

    for z in z_grid:
        single_res = solve_ridge_svd(X, y, z=z, t_tr=t_tr)
        assert np.allclose(batch_res[z], single_res, atol=1e-12)


def test_interpolation_threshold_stability():
    """Verify solver runs cleanly right at the interpolation boundary c_q = 1.0."""
    np.random.seed(42)
    t_tr, p1 = 60, 60  # c_q = 1.0
    X = np.random.randn(t_tr, p1)
    y = np.random.randn(t_tr)

    # Even with z = 0 right at boundary, SVD must produce valid finite numbers
    beta_0 = solve_ridge_svd(X, y, z=0.0, t_tr=t_tr)
    assert np.all(np.isfinite(beta_0))

    # And with z > 0
    beta_ridge = solve_ridge_svd(X, y, z=0.1, t_tr=t_tr)
    assert np.all(np.isfinite(beta_ridge))
