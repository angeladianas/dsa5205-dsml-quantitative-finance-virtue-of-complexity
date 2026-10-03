"""Global project configuration and hyperparameters.

Single source of truth for all paths, random seeds, and experimental grids.
Strictly follows PEP 8 styling.
"""

from pathlib import Path
from typing import Any

# Repository root and key directory paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"

OUTPUT_DIR: Path = PROJECT_ROOT / "output"
RESULTS_DIR: Path = OUTPUT_DIR / "results"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
PREDICTIONS_DIR: Path = OUTPUT_DIR / "predictions"

# Ensure output directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Student identification metadata
STUDENT_ID: str = "A0327258X"

# Global master seed for deterministic reproducibility
GLOBAL_SEED: int = 42

# Task 1: Misspecified Ridge Simulation Parameters
# KMZ (2024) Theorem 1 & Proposition 6 settings
TASK1_PARAMS: dict[str, Any] = {
    "T_tr": 300,  # Training sample size
    "T_te": 2000,  # Test sample size
    "c_dgp": 10.0,  # True DGP complexity c = P / T_tr
    "b_star": 0.25,  # True signal strength ||beta*||^2 = b*
    "sigma_eps": 1.0,  # Innovation variance
    "n_runs": 50,  # Monte Carlo runs for smooth finite-sample expectations
    "cq_grid": [
        0.5,
        0.75,
        0.9,
        0.95,
        0.98,
        1.02,
        1.05,
        1.1,
        1.25,
        1.5,
        2.0,
        3.0,
        5.0,
        7.0,
        10.0,
    ],
    "z_grid": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0],
}
