"""Task 1: Diagnostic figures for the misspecified Ridge simulation (KMZ 2024)."""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, RESULTS_DIR


def set_publication_style() -> None:
    """Set matplotlib styling defaults."""
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "figure.dpi": 300,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "grid.alpha": 0.4,
            "grid.linestyle": "--",
        }
    )


def plot_task1_figures(summary_csv_path: Path | None = None) -> None:
    """Generate all Task 1 publication figures from cached summary CSV data.

    Args:
        summary_csv_path: Path to task1_summary.csv. If None, uses default in RESULTS_DIR.
    """
    if summary_csv_path is None:
        summary_csv_path = RESULTS_DIR / "task1_summary.csv"

    if not summary_csv_path.exists():
        raise FileNotFoundError(
            f"Summary file not found at {summary_csv_path}. Run src/task1_sim.py first."
        )

    df = pd.read_csv(summary_csv_path)
    set_publication_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Separate regular z grid and optimal z*
    df_regular = df[~df["is_optimal_z"]].copy()
    df_opt = df[df["is_optimal_z"]].sort_values("cq").copy()

    # Get sorted unique z values
    z_values = sorted(df_regular["z"].unique())

    # Distinct colormap for regular z values
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(z_values)))

    # -------------------------------------------------------------
    # Helper to plot on a given axis
    # -------------------------------------------------------------
    def plot_metric_on_ax(
        ax: plt.Axes,
        metric_col: str,
        ylabel: str,
        title: str,
        ylim: tuple[float, float] | None = None,
        use_log_y: bool = False,
    ) -> None:
        # Plot each z curve
        for idx, z_val in enumerate(z_values):
            sub = df_regular[df_regular["z"] == z_val].sort_values("cq")
            if z_val == 0.0:
                ax.plot(
                    sub["cq"],
                    sub[metric_col],
                    label="Ridgeless (z = 0)",
                    color="black",
                    linestyle="--",
                    marker="o",
                    linewidth=2.0,
                    zorder=4,
                )
            else:
                ax.plot(
                    sub["cq"],
                    sub[metric_col],
                    label=f"z = {z_val}",
                    color=colors[idx],
                    marker="s",
                    alpha=0.85,
                )

        # Plot theoretical optimal z*
        if not df_opt.empty:
            ax.plot(
                df_opt["cq"],
                df_opt[metric_col],
                label=r"Optimal $z^*(q)$",
                color="#D90429",  # Vivid crimson
                linewidth=2.5,
                marker="D",
                markersize=6,
                zorder=5,
            )

        # Mark interpolation boundary
        ax.axvline(
            x=1.0,
            color="#6C757D",
            linestyle=":",
            linewidth=1.5,
            label="Interpolation ($cq = 1$)",
        )

        ax.set_xlabel(r"Observed Complexity $cq = P_1 / T_{tr}$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        if use_log_y:
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(ylim)

    # -------------------------------------------------------------
    # Figure 1: Out-of-Sample R^2_{paper} vs. cq
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plot_metric_on_ax(
        ax=ax,
        metric_col="r2_paper_mean",
        ylabel=r"Out-of-Sample $R^2_{\mathrm{paper}}$",
        title=r"Out-of-Sample $R^2_{\mathrm{paper}}$ vs. Observed Complexity $cq$",
        ylim=(
            -0.5,
            0.25,
        ),  # Clipped view to clearly reveal all curves without z=0 explosion distorting scale
    )
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_r2_vs_cq.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig1_r2_vs_cq.pdf")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 2: Expected Timing Return vs. cq
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plot_metric_on_ax(
        ax=ax,
        metric_col="expected_return_mean",
        ylabel=r"Expected Timing Return $\mathbb{E}[R^\pi_{t+1}]$",
        title=r"Expected Timing Return vs. Observed Complexity $cq$",
    )
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_expected_return_vs_cq.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig2_expected_return_vs_cq.pdf")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 3: Sharpe Ratio vs. cq (The Virtue of Complexity)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plot_metric_on_ax(
        ax=ax,
        metric_col="sharpe_mean",
        ylabel=r"Timing Sharpe Ratio $\mathrm{SR}$ (KMZ Eq. 5)",
        title=r"Timing Sharpe Ratio vs. Observed Complexity $cq$",
        ylim=(0.0, 0.08),
    )
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_sharpe_vs_cq.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig3_sharpe_vs_cq.pdf")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 4: Parameter Euclidean Norm ||beta||^2 vs. cq (Log Scale)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plot_metric_on_ax(
        ax=ax,
        metric_col="param_norm_mean",
        ylabel=r"Parameter Size $\|\hat{\beta}(z)\|_2^2$ (Log Scale)",
        title=r"Parameter Euclidean Norm vs. Observed Complexity $cq$",
        use_log_y=True,
    )
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_param_norm_vs_cq.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig4_param_norm_vs_cq.pdf")
    plt.close(fig)

    # -------------------------------------------------------------
    # Master 4-Panel Figure for Report Inclusion
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_metric_on_ax(
        ax=axes[0, 0],
        metric_col="r2_paper_mean",
        ylabel=r"Out-of-Sample $R^2_{\mathrm{paper}}$",
        title=r"(a) Out-of-Sample $R^2_{\mathrm{paper}}$",
        ylim=(-0.5, 0.25),
    )
    plot_metric_on_ax(
        ax=axes[0, 1],
        metric_col="expected_return_mean",
        ylabel=r"Expected Return $\mathbb{E}[R^\pi_{t+1}]$",
        title=r"(b) Expected Timing Return",
    )
    plot_metric_on_ax(
        ax=axes[1, 0],
        metric_col="sharpe_mean",
        ylabel=r"Sharpe Ratio $\mathrm{SR}$",
        title=r"(c) Timing Sharpe Ratio (Virtue of Complexity)",
        ylim=(0.0, 0.08),
    )
    plot_metric_on_ax(
        ax=axes[1, 1],
        metric_col="param_norm_mean",
        ylabel=r"Parameter Norm $\|\hat{\beta}(z)\|_2^2$",
        title=r"(d) Parameter Norm (Log Scale)",
        use_log_y=True,
    )

    # Add shared legend at bottom
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(FIGURES_DIR / "task1_master_4panel.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "task1_master_4panel.pdf", bbox_inches="tight")
    plt.close(fig)

    print("================================================================")
    print("Task 1 publication figures successfully generated in:")
    print(f"  {FIGURES_DIR}")
    print("  - fig1_r2_vs_cq.png / .pdf")
    print("  - fig2_expected_return_vs_cq.png / .pdf")
    print("  - fig3_sharpe_vs_cq.png / .pdf")
    print("  - fig4_param_norm_vs_cq.png / .pdf")
    print("  - task1_master_4panel.png / .pdf")
    print("================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Task 1 Figures")
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Path to task1_summary.csv",
    )
    args = parser.parse_args()
    csv_path = Path(args.summary_csv) if args.summary_csv else None
    plot_task1_figures(summary_csv_path=csv_path)
