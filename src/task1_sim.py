"""Task 1: Monte Carlo simulation of misspecified Ridge regression (KMZ 2024)."""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.config import GLOBAL_SEED, RESULTS_DIR, TASK1_PARAMS
from src.metrics import (
    compute_optimal_shrinkage_theory,
    compute_param_norm,
    compute_r2_paper,
    compute_sharpe_uncentered,
    compute_sharpe_var,
    compute_timing_returns,
)
from src.solvers import compute_compact_svd, solve_ridge_svd


def generate_dgp_run_data(
    t_tr: int,
    t_te: int,
    p_total: int,
    b_star: float,
    sigma_eps: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Generate synthetic market data under the true Data Generating Process (DGP).

    Model:
        R_{t+1} = S_t^\top \beta^* + \varepsilon_{t+1}
        S_t \sim \mathcal{N}(0, I_P), \quad \varepsilon_{t+1} \sim \mathcal{N}(0, \sigma_\varepsilon^2)
        ||\beta^*||_2^2 = b^*

    Args:
        t_tr: Training sample size.
        t_te: Out-of-sample test sample size.
        p_total: True total feature dimension P = c * T_{tr}.
        b_star: True signal strength.
        sigma_eps: Innovation noise standard deviation.
        rng: NumPy random generator instance.

    Returns:
        Tuple of (S, R, beta_star):
            S: Signal matrix of shape (t_tr + t_te, p_total).
            R: Realized return vector of shape (t_tr + t_te,).
            beta_star: True coefficient vector of shape (p_total,).
    """
    t_total = t_tr + t_te

    # Features S_t \sim N(0, I_P)
    S = rng.standard_normal(size=(t_total, p_total))

    # Dense, isotropic beta* normalized to ||beta*||_2 = sqrt(b*)
    beta_raw = rng.standard_normal(size=p_total)
    beta_norm = np.linalg.norm(beta_raw)
    beta_star = beta_raw * (np.sqrt(b_star) / beta_norm)

    # Innovations eps_{t+1} \sim N(0, sigma_eps^2)
    eps = sigma_eps * rng.standard_normal(size=t_total)

    # Realized returns R_{t+1} = S_t^\top \beta^* + \varepsilon_{t+1}
    R = S @ beta_star + eps

    return S, R, beta_star


def run_single_simulation(
    run_id: int,
    t_tr: int,
    t_te: int,
    c_dgp: float,
    b_star: float,
    sigma_eps: float,
    cq_grid: list[float],
    z_grid: list[float],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Execute a single Monte Carlo run across all cq and z grid points.

    Implements misspecification via partial observability: columns of S are
    randomly permuted once, and for each cq, only the first P_1 columns are observed.

    Args:
        run_id: Unique index for this simulation trial.
        t_tr: Training sample size.
        t_te: Test sample size.
        c_dgp: True DGP complexity c = P / T_{tr}.
        b_star: True signal strength.
        sigma_eps: Innovation noise std.
        cq_grid: Observed complexity values cq = P_1 / T_{tr}.
        z_grid: Ridge shrinkage values z.
        rng: Random generator instance.

    Returns:
        List of result record dictionaries for each (cq, z) pair and optimal z*.
    """
    p_total = round(c_dgp * t_tr)
    S, R, _ = generate_dgp_run_data(
        t_tr=t_tr,
        t_te=t_te,
        p_total=p_total,
        b_star=b_star,
        sigma_eps=sigma_eps,
        rng=rng,
    )

    # Partial observability: apply random column permutation once per run
    perm = rng.permutation(p_total)
    S_permuted = S[:, perm]

    # Split into train and out-of-sample test
    S_tr = S_permuted[:t_tr, :]
    y_tr = R[:t_tr]
    S_te = S_permuted[t_tr : t_tr + t_te, :]
    y_te = R[t_tr : t_tr + t_te]

    records: list[dict[str, Any]] = []

    for cq in cq_grid:
        p1 = max(1, round(cq * t_tr))
        q = float(p1 / p_total)

        # Observe only the first P_1 permuted columns
        X_tr = S_tr[:, :p1]
        X_te = S_te[:, :p1]

        # Compute compact SVD of X_tr once per cq block
        svd_cache = compute_compact_svd(X_tr)

        # 1. Evaluate user-specified shrinkage grid z
        for z in z_grid:
            beta_hat = solve_ridge_svd(X=X_tr, y=y_tr, z=float(z), t_tr=t_tr, svd_cache=svd_cache)
            y_pred = X_te @ beta_hat
            r_pi = compute_timing_returns(y_te, y_pred)

            records.append(
                {
                    "run_id": run_id,
                    "cq": float(cq),
                    "q": q,
                    "p1": p1,
                    "z": float(z),
                    "is_optimal_z": False,
                    "r2_paper": compute_r2_paper(y_te, y_pred),
                    "expected_return": float(np.mean(r_pi)),
                    "sharpe": compute_sharpe_uncentered(r_pi),
                    "sharpe_var": compute_sharpe_var(r_pi),
                    "param_norm": compute_param_norm(beta_hat),
                }
            )

        # 2. Evaluate theoretical optimal shrinkage z*(q) (KMZ Prop. 6(ii))
        z_opt = compute_optimal_shrinkage_theory(c_dgp=c_dgp, b_star=b_star, q=q)
        beta_opt = solve_ridge_svd(X=X_tr, y=y_tr, z=z_opt, t_tr=t_tr, svd_cache=svd_cache)
        y_pred_opt = X_te @ beta_opt
        r_pi_opt = compute_timing_returns(y_te, y_pred_opt)

        records.append(
            {
                "run_id": run_id,
                "cq": float(cq),
                "q": q,
                "p1": p1,
                "z": z_opt,
                "is_optimal_z": True,
                "r2_paper": compute_r2_paper(y_te, y_pred_opt),
                "expected_return": float(np.mean(r_pi_opt)),
                "sharpe": compute_sharpe_uncentered(r_pi_opt),
                "sharpe_var": compute_sharpe_var(r_pi_opt),
                "param_norm": compute_param_norm(beta_opt),
            }
        )

    return records


def run_task1_simulation(
    n_runs: int = TASK1_PARAMS["n_runs"],
    seed: int = GLOBAL_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the full Monte Carlo simulation for Task 1 and save CSV caches.

    Args:
        n_runs: Number of independent simulation trials.
        seed: Master random seed.

    Returns:
        Tuple of (df_runs, df_summary):
            df_runs: DataFrame of all raw individual simulation records.
            df_summary: DataFrame of aggregated means and standard errors across runs.
    """
    print("================================================================")
    print("Starting Task 1 Simulation (Misspecified Ridge)")
    print(
        f"Runs: {n_runs} | Seed: {seed} | T_tr: {TASK1_PARAMS['T_tr']} | T_te: {TASK1_PARAMS['T_te']}"
    )
    print(
        f"DGP Complexity c: {TASK1_PARAMS['c_dgp']} | Signal Strength b*: {TASK1_PARAMS['b_star']}"
    )
    print(
        f"Observed cq points: {len(TASK1_PARAMS['cq_grid'])} | Shrinkage z points: {len(TASK1_PARAMS['z_grid'])}"
    )
    print("================================================================")

    start_time = time.time()
    rng = np.random.default_rng(seed)

    all_records: list[dict[str, Any]] = []

    for r in range(1, n_runs + 1):
        r_records = run_single_simulation(
            run_id=r,
            t_tr=TASK1_PARAMS["T_tr"],
            t_te=TASK1_PARAMS["T_te"],
            c_dgp=TASK1_PARAMS["c_dgp"],
            b_star=TASK1_PARAMS["b_star"],
            sigma_eps=TASK1_PARAMS["sigma_eps"],
            cq_grid=TASK1_PARAMS["cq_grid"],
            z_grid=TASK1_PARAMS["z_grid"],
            rng=rng,
        )
        all_records.extend(r_records)
        if r % 10 == 0 or r == n_runs:
            elapsed = time.time() - start_time
            print(f"  Completed {r}/{n_runs} runs ({r / n_runs * 100:.1f}%) in {elapsed:.1f}s")

    df_runs = pd.DataFrame(all_records)

    # Compute aggregated statistics (mean and standard error) across runs
    metrics = ["r2_paper", "expected_return", "sharpe", "sharpe_var", "param_norm"]
    group_cols = ["cq", "q", "p1", "z", "is_optimal_z"]

    agg_dict = {}
    for m in metrics:
        agg_dict[f"{m}_mean"] = (m, "mean")
        agg_dict[f"{m}_sem"] = (
            m,
            lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0,
        )

    df_summary = (
        df_runs.groupby(group_cols, as_index=False)
        .agg(**agg_dict)
        .sort_values(by=["is_optimal_z", "z", "cq"])
    )

    # Ensure results directory exists and save CSV files
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    runs_csv_path = RESULTS_DIR / "task1_simulation_runs.csv"
    summary_csv_path = RESULTS_DIR / "task1_summary.csv"

    df_runs.to_csv(runs_csv_path, index=False)
    df_summary.to_csv(summary_csv_path, index=False)

    total_time = time.time() - start_time
    print(f"Simulation completed in {total_time:.2f} seconds!")
    print(f"Saved raw runs to: {runs_csv_path} ({len(df_runs):,} rows)")
    print(f"Saved summary to:  {summary_csv_path} ({len(df_summary):,} rows)")
    print("================================================================")

    return df_runs, df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 1 Simulation")
    parser.add_argument(
        "--runs",
        type=int,
        default=TASK1_PARAMS["n_runs"],
        help=f"Number of Monte Carlo runs (default: {TASK1_PARAMS['n_runs']})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=GLOBAL_SEED,
        help=f"Master random seed (default: {GLOBAL_SEED})",
    )
    args = parser.parse_args()

    run_task1_simulation(n_runs=args.runs, seed=args.seed)
