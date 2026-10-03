"""Mathematical metric kernels for return prediction and trading performance (KMZ 2024)."""

import numpy as np


def compute_r2_paper(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r"""Compute out-of-sample R^2 according to KMZ (2024) Proposition 3 identity.

    Formula:
        R^2_{paper} = 1 - E[(R_{t+1} - \hat{R}_{t+1})^2] / E[R_{t+1}^2]
                    = (2 * E[\hat{R}_{t+1} * R_{t+1}] - E[\hat{R}_{t+1}^2]) / E[R_{t+1}^2]

    Args:
        y_true: 1D array of realized returns R_{t+1}.
        y_pred: 1D array of predicted returns \hat{R}_{t+1}.

    Returns:
        Scalar out-of-sample R^2_{paper}. Returns 0.0 if y_true has zero second moment.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    if y_t.size == 0 or y_p.size == 0:
        return 0.0

    denom = np.mean(y_t**2)
    if denom < 1e-14:
        return 0.0

    numer = 2.0 * np.mean(y_p * y_t) - np.mean(y_p**2)
    return float(numer / denom)


def compute_timing_returns(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    r"""Compute the realized return series of a linear timing portfolio.

    Following KMZ Eq. (6), portfolio timing weight is \pi_t = \hat{R}_{t+1},
    so the realized timing return is R^\pi_{t+1} = \pi_t * R_{t+1} = \hat{R}_{t+1} * R_{t+1}.

    Args:
        y_true: 1D array of realized returns R_{t+1}.
        y_pred: 1D array of forecasts \hat{R}_{t+1}.

    Returns:
        1D array of timing returns R^\pi_{t+1}.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    return y_p * y_t


def compute_sharpe_uncentered(r_pi: np.ndarray) -> float:
    r"""Compute uncentered Sharpe ratio following KMZ (2024) Eq. (5).

    Formula:
        SR = E[R^\pi_{t+1}] / sqrt(E[(R^\pi_{t+1})^2])

    Args:
        r_pi: 1D array of timing returns R^\pi_{t+1}.

    Returns:
        Scalar uncentered Sharpe ratio. Returns 0.0 if second moment is near zero.
    """
    ret = np.asarray(r_pi, dtype=np.float64).ravel()
    if ret.size == 0:
        return 0.0

    mean_ret = np.mean(ret)
    second_moment = np.mean(ret**2)

    if second_moment < 1e-14:
        return 0.0

    return float(mean_ret / np.sqrt(second_moment))


def compute_sharpe_var(r_pi: np.ndarray) -> float:
    r"""Compute secondary variance-based (centered) Sharpe ratio.

    Formula:
        SR_{var} = E[R^\pi_{t+1}] / sqrt(Var(R^\pi_{t+1}))

    Args:
        r_pi: 1D array of timing returns R^\pi_{t+1}.

    Returns:
        Scalar centered Sharpe ratio. Returns 0.0 if variance is near zero.
    """
    ret = np.asarray(r_pi, dtype=np.float64).ravel()
    if ret.size < 2:
        return 0.0

    mean_ret = np.mean(ret)
    std_ret = np.std(ret, ddof=1)

    if np.isnan(std_ret) or std_ret < 1e-14:
        return 0.0

    return float(mean_ret / std_ret)


def compute_param_norm(beta: np.ndarray) -> float:
    """Compute Euclidean parameter norm squared ||\\hat{\beta}||_2^2.

    Formula:
        ||\\hat{\beta}||_2^2 = \\sum_{j=1}^{P_1} \\hat{\beta}_j^2

    Args:
        beta: 1D or 2D array of estimated coefficients.

    Returns:
        Scalar squared L2 norm.
    """
    b = np.asarray(beta, dtype=np.float64).ravel()
    return float(np.sum(b**2))


def compute_optimal_shrinkage_theory(c_dgp: float, b_star: float, q: float) -> float:
    r"""Compute KMZ (2024) Proposition 6(ii) optimal shrinkage under Psi = I.

    Formula:
        z^*(q) = c * (1 + b^* * (1 - q)) / b^*

    When q = 1 (correctly specified), this simplifies to z^* = c / b^* (Proposition 3).

    Args:
        c_dgp: True data generating process complexity c = P / T_{tr}.
        b_star: True signal strength ||beta*||_2^2.
        q: Revealed feature fraction P_1 / P \in [0, 1].

    Returns:
        Scalar theoretical optimal regularization parameter z^*.
    """
    if b_star <= 0:
        raise ValueError("b_star must be strictly positive.")
    return float(c_dgp * (1.0 + b_star * (1.0 - q)) / b_star)
