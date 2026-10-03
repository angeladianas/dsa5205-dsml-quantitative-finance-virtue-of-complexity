"""Numerically stable Ridge and Moore-Penrose pseudoinverse solvers via compact SVD.

Strictly adheres to the "Never Invert X^T X" rule to ensure complete numerical
stability near the interpolation threshold (c_q \approx 1) and under ridgeless OLS (z = 0).
"""

from collections.abc import Sequence

import numpy as np


def compute_compact_svd(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute compact Singular Value Decomposition X = U * diag(s) * Vt.

    Args:
        X: Real matrix of shape (T_tr, P_1).

    Returns:
        Tuple of (U, s, Vt) where:
            U is shape (T_tr, k) with orthonormal columns,
            s is 1D array of singular values of length k = min(T_tr, P_1),
            Vt is shape (k, P_1) with orthonormal rows.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return U, s, Vt


def solve_ridge_svd(
    X: np.ndarray,
    y: np.ndarray,
    z: float,
    t_tr: int,
    svd_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    rcond: float = 1e-12,
) -> np.ndarray:
    r"""Solve ridge regression or ridgeless Moore-Penrose OLS using compact SVD.

    Estimator formula:
        \hat{\beta}(z) = (X^\top X + z * T_{tr} * I)^{-1} X^\top y
                       = V * (\Sigma^2 + z * T_{tr} * I)^{-1} \Sigma U^\top y

    For z = 0 (ridgeless OLS):
        Evaluates the exact Moore-Penrose pseudoinverse X^+ y:
        \hat{\beta}(0) = V * \Sigma^+ U^\top y
        where singular values smaller than rcond are zeroed out.

    Args:
        X: Design matrix of shape (T_tr, P_1).
        y: Target return vector of shape (T_tr,).
        z: Regularization parameter (z >= 0.0).
        t_tr: Effective number of training samples for scaling z * T_tr * I.
        svd_cache: Precomputed (U, s, Vt) to bypass recomputing SVD.
        rcond: Cutoff threshold for small singular values when z == 0.

    Returns:
        1D array of estimated coefficients \hat{\beta}(z) of shape (P_1,).
    """
    if svd_cache is None:
        U, s, Vt = compute_compact_svd(X)
    else:
        U, s, Vt = svd_cache

    # Compute projection U^\top y of length k
    Uy = U.T @ y

    # Compute scalar shrinkage weights d_i for each singular value s_i
    if z == 0.0:
        # Exact Moore-Penrose minimum-norm pseudoinverse
        filter_weights = np.where(s > rcond, 1.0 / s, 0.0)
    else:
        # Ridge shrinkage weights: s_i / (s_i^2 + z * T_tr)
        filter_weights = s / (s**2 + z * float(t_tr))

    # \hat{\beta} = V * (filter_weights * Uy)
    # Vt has shape (k, P_1), so Vt.T has shape (P_1, k)
    beta = Vt.T @ (filter_weights * Uy)
    return beta


def batch_solve_ridge_svd(
    X: np.ndarray,
    y: np.ndarray,
    z_grid: Sequence[float],
    t_tr: int,
    rcond: float = 1e-12,
) -> dict[float, np.ndarray]:
    r"""Compute ridge solutions for an entire grid of regularization values z.

    Computes the compact SVD only once, then evaluates each z in z_grid
    in O(k * P_1) time via vector scaling.

    Args:
        X: Design matrix of shape (T_tr, P_1).
        y: Target return vector of shape (T_tr,).
        z_grid: Sequence of regularization parameters z.
        t_tr: Effective number of training samples.
        rcond: Cutoff threshold for zero singular values.

    Returns:
        Dictionary mapping each z to its estimated coefficient vector \hat{\beta}(z).
    """
    svd_cache = compute_compact_svd(X)
    solutions: dict[float, np.ndarray] = {}
    for z in z_grid:
        solutions[float(z)] = solve_ridge_svd(
            X=X, y=y, z=float(z), t_tr=t_tr, svd_cache=svd_cache, rcond=rcond
        )
    return solutions
