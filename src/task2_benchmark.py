"""Task 2: Alternative Model Benchmark (Lasso / L1 Sparsity vs. Ridge).

Investigates how changing the inductive bias from L2 rotational shrinkage (Ridge)
to L1 coordinate sparsity (Lasso) impacts out-of-sample predictability and trading performance
on the exact same simulated DGP and splits as Task 1.

Theoretical Focus:
- The true DGP has dense, isotropic coefficients beta* with no exact zeros.
- Lasso enforces coordinate sparsity, setting small signals to zero.
- Demonstrates how truncation bias degrades out-of-sample R^2 and caps Sharpe ratio
  across the complexity spectrum cq.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import lasso_path

# Filter expected coordinate descent convergence warnings for small alphas
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from src.config import GLOBAL_SEED, RESULTS_DIR, TASK1_PARAMS
from src.metrics import (
    compute_param_norm,
    compute_r2_paper,
    compute_sharpe_uncentered,
    compute_sharpe_var,
    compute_timing_returns,
)
from src.task1_sim import generate_dgp_run_data

# Selected descending penalty grid for efficient coordinate descent path
LASSO_ALPHAS = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]


def run_single_lasso_simulation(
    run_id: int,
    t_tr: int,
    t_te: int,
    c_dgp: float,
    b_star: float,
    sigma_eps: float,
    cq_grid: list[float],
    alphas: list[float],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Run a single Monte Carlo trial evaluating Lasso across all cq and alpha values.

    Uses identical DGP data and column permutation as Task 1.

    Args:
        run_id: Unique trial identifier.
        t_tr: Training sample size.
        t_te: Test sample size.
        c_dgp: True DGP complexity c = P / T_{tr}.
        b_star: True signal strength.
        sigma_eps: Innovation noise std.
        cq_grid: Observed complexity grid points.
        alphas: Descending sequence of L1 penalty parameters.
        rng: Random number generator.

    Returns:
        List of result dictionaries for each (cq, alpha) combination.
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

    # Identical partial observability permutation as Task 1
    perm = rng.permutation(p_total)
    S_permuted = S[:, perm]

    S_tr = S_permuted[:t_tr, :]
    y_tr = R[:t_tr]
    S_te = S_permuted[t_tr : t_tr + t_te, :]
    y_te = R[t_tr : t_tr + t_te]

    records: list[dict[str, Any]] = []

    for cq in cq_grid:
        p1 = max(1, round(cq * t_tr))
        q = float(p1 / p_total)

        X_tr = S_tr[:, :p1]
        X_te = S_te[:, :p1]

        # Compute full Lasso solution path in one fast C-level coordinate descent pass
        alphas_out, coefs, _ = lasso_path(X_tr, y_tr, alphas=alphas, max_iter=2000, tol=1e-3)
        # coefs has shape (p1, len(alphas))

        for idx, alpha_val in enumerate(alphas_out):
            beta_hat = coefs[:, idx]
            y_pred = X_te @ beta_hat
            r_pi = compute_timing_returns(y_te, y_pred)

            n_nonzero = int(np.sum(beta_hat != 0))
            sparsity_ratio = float(n_nonzero / p1)

            records.append(
                {
                    "run_id": run_id,
                    "cq": float(cq),
                    "q": q,
                    "p1": p1,
                    "alpha": float(alpha_val),
                    "r2_paper": compute_r2_paper(y_te, y_pred),
                    "expected_return": float(np.mean(r_pi)),
                    "sharpe": compute_sharpe_uncentered(r_pi),
                    "sharpe_var": compute_sharpe_var(r_pi),
                    "param_norm": compute_param_norm(beta_hat),
                    "n_nonzero": n_nonzero,
                    "sparsity_ratio": sparsity_ratio,
                }
            )

    return records


def run_task2_simulation(
    n_runs: int = TASK1_PARAMS["n_runs"],
    seed: int = GLOBAL_SEED,
    alphas: list[float] = LASSO_ALPHAS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the full Monte Carlo simulation for Task 2 and export CSV caches.

    Args:
        n_runs: Number of independent simulation trials.
        seed: Master random seed (matches Task 1).
        alphas: Sequence of L1 penalty hyperparameters.

    Returns:
        Tuple of (df_runs, df_summary).
    """
    print("================================================================")
    print("Starting Task 2 Simulation (Lasso Benchmark on Same Data)")
    print(f"Runs: {n_runs} | Seed: {seed} | Alphas: {len(alphas)} points")
    print(f"Alpha range: [{min(alphas):.4f}, {max(alphas):.4f}]")
    print("================================================================")

    start_time = time.time()
    rng = np.random.default_rng(seed)

    all_records: list[dict[str, Any]] = []

    for r in range(1, n_runs + 1):
        r_records = run_single_lasso_simulation(
            run_id=r,
            t_tr=TASK1_PARAMS["T_tr"],
            t_te=TASK1_PARAMS["T_te"],
            c_dgp=TASK1_PARAMS["c_dgp"],
            b_star=TASK1_PARAMS["b_star"],
            sigma_eps=TASK1_PARAMS["sigma_eps"],
            cq_grid=TASK1_PARAMS["cq_grid"],
            alphas=alphas,
            rng=rng,
        )
        all_records.extend(r_records)
        if r % 10 == 0 or r == n_runs:
            elapsed = time.time() - start_time
            print(f"  Completed {r}/{n_runs} runs ({r / n_runs * 100:.1f}%) in {elapsed:.1f}s")

    df_runs = pd.DataFrame(all_records)

    # Compute aggregated statistics (mean and standard error) across runs
    metrics = [
        "r2_paper",
        "expected_return",
        "sharpe",
        "sharpe_var",
        "param_norm",
        "n_nonzero",
        "sparsity_ratio",
    ]
    group_cols = ["cq", "q", "p1", "alpha"]

    agg_dict = {}
    for m in metrics:
        agg_dict[f"{m}_mean"] = (m, "mean")
        agg_dict[f"{m}_sem"] = (
            m,
            lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0,
        )

    df_summary = (
        df_runs.groupby(group_cols, as_index=False).agg(**agg_dict).sort_values(by=["alpha", "cq"])
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs_csv_path = RESULTS_DIR / "task2_simulation_runs.csv"
    summary_csv_path = RESULTS_DIR / "task2_summary.csv"

    df_runs.to_csv(runs_csv_path, index=False)
    df_summary.to_csv(summary_csv_path, index=False)

    total_time = time.time() - start_time
    print(f"Task 2 simulation completed in {total_time:.2f} seconds!")
    print(f"Saved raw runs to: {runs_csv_path} ({len(df_runs):,} rows)")
    print(f"Saved summary to:  {summary_csv_path} ({len(df_summary):,} rows)")
    print("================================================================")

    return df_runs, df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 2 Simulation")
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

    run_task2_simulation(n_runs=args.runs, seed=args.seed)
