"""Task 4: Plotting script for dynamical decoupling in two-layer neural networks.

Loads simulation history from `output/results/task4_dynamics_history.csv`
and renders publication-grade figures:
- `output/figures/fig8_dynamical_decoupling.png`
- `output/figures/fig8_dynamical_decoupling.pdf`
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, RESULTS_DIR


def plot_task4_dynamics(df: pd.DataFrame | None = None) -> None:
    """Render publication figure showing the three dynamical learning regimes."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if df is None:
        csv_path = RESULTS_DIR / "task4_dynamics_history.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing history file: {csv_path}. Run `python -m src.task4_dynamics` first."
            )
        df = pd.read_csv(csv_path)

    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "figure.dpi": 300,
        }
    )

    fig, ax1 = plt.subplots(figsize=(9, 6))

    color_sharpe = "#1D3557"
    color_loss = "#E63946"
    color_l1 = "#2A9D8F"

    # Plot Out-of-Sample Sharpe Ratio on left axis
    ax1.plot(
        df["epoch"],
        df["test_sharpe"],
        color=color_sharpe,
        linewidth=2.5,
        label="Out-of-Sample Sharpe Ratio",
    )
    ax1.set_xlabel(r"Training Epochs (Gradient Descent Time $t$)")
    ax1.set_ylabel(
        r"Out-of-Sample Sharpe Ratio $\mathrm{SR}$",
        color=color_sharpe,
        fontweight="bold",
    )
    ax1.tick_params(axis="y", labelcolor=color_sharpe)

    # Find peak epoch
    peak_row = df.loc[df["test_sharpe"].idxmax()]
    t_star = int(peak_row["epoch"])
    sr_peak = float(peak_row["test_sharpe"])

    ax1.axvline(
        x=t_star,
        color="#E76F51",
        linestyle="--",
        linewidth=2.0,
        label=rf"Optimal Early Stopping ($t^* = {t_star}$)",
    )
    ax1.scatter([t_star], [sr_peak], color="#E76F51", s=80, zorder=5)

    # Create twin axis for Train Loss and L1 complexity
    ax2 = ax1.twinx()
    ax2.plot(
        df["epoch"],
        df["train_loss"],
        color=color_loss,
        linestyle=":",
        linewidth=2.0,
        label="Training Loss (Empirical Risk)",
    )
    ax2.plot(
        df["epoch"],
        df["second_layer_l1"],
        color=color_l1,
        linestyle="-.",
        linewidth=2.0,
        label=r"Weight Norm $\|W^{(2)}\|_1$ (Complexity)",
    )
    ax2.set_ylabel("Training Loss & Parameter Norm", color="#333333", fontweight="bold")
    ax2.grid(False)

    # Shaded regime annotations
    ax1.axvspan(
        0,
        t_star,
        alpha=0.08,
        color="#2A9D8F",
        label=r"Regime 1: Fast Feature Learning ($t \sim \mathcal{O}(1)$)",
    )
    ax1.axvspan(
        t_star,
        int(df["epoch"].max()),
        alpha=0.08,
        color="#E63946",
        label=r"Regime 2: Noise Overfitting / Feature Unlearning ($t \sim \mathcal{O}(m)$)",
    )

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, fontsize=9)

    plt.title(
        r"\textbf{Dynamical Decoupling in Low-SNR Financial Return Prediction}"
        + "\n"
        + r"Feature Learning vs. Overfitting (Montanari \& Urbani 2025)",
        fontsize=13,
        pad=12,
    )

    plt.tight_layout()
    fig_path_png = FIGURES_DIR / "fig8_dynamical_decoupling.png"
    fig_path_pdf = FIGURES_DIR / "fig8_dynamical_decoupling.pdf"
    fig.savefig(fig_path_png, dpi=300)
    fig.savefig(fig_path_pdf)
    plt.close(fig)

    print("Generated Task 4 figures:")
    print(f"  - {fig_path_png}")
    print(f"  - {fig_path_pdf}")
    print(f"  Peak Sharpe Ratio: {sr_peak:.4f} achieved at epoch t* = {t_star}")
    print("================================================================")


if __name__ == "__main__":
    plot_task4_dynamics()
