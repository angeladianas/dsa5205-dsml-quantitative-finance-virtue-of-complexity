"""Task 2: Comparative Publication-Grade Visualizations (Lasso vs. Ridge).

Generates comparison figures contrasting the L1 coordinate sparsity of Lasso
against the L2 rotational shrinkage of Ridge:
1. Figure 5: Out-of-Sample Sharpe Ratio vs. c_q (Ridge vs. Lasso)
2. Figure 6: Out-of-Sample R^2_{paper} vs. c_q (Ridge vs. Lasso)
3. Figure 7: Active Feature Sparsity Fraction vs. c_q
4. Master 3-Panel Comparison Figure for the report.
"""

import argparse
from pathlib import Path
import sys
from typing import Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, RESULTS_DIR


def set_publication_style() -> None:
    """Set publication-quality plotting aesthetic."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
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
    })


def plot_task2_figures() -> None:
    """Load Task 1 and Task 2 summaries and render comparison figures."""
    t1_path = RESULTS_DIR / "task1_summary.csv"
    t2_path = RESULTS_DIR / "task2_summary.csv"

    if not t1_path.exists() or not t2_path.exists():
        raise FileNotFoundError("Summary files missing in output/results/.")

    df1 = pd.read_csv(t1_path)
    df2 = pd.read_csv(t2_path)

    set_publication_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Ridge benchmarks from Task 1
    ridge_opt = df1[df1["is_optimal_z"]].sort_values("cq")
    ridge_zero = df1[df1["z"] == 0.0].sort_values("cq")

    # Select representative alphas for clear visual comparison
    target_alphas = [0.005, 0.01, 0.02, 0.05, 0.1]
    df2_filtered = df2[df2["alpha"].isin(target_alphas)].copy()

    lasso_colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(target_alphas)))

    # -------------------------------------------------------------
    # Figure 5: Sharpe Ratio vs. c_q (Ridge vs. Lasso)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Plot Ridge benchmarks
    ax.plot(
        ridge_opt["cq"],
        ridge_opt["sharpe_mean"],
        label=r"Optimal Ridge $z^*(q)$",
        color="#D90429",
        linewidth=2.6,
        marker="D",
        markersize=6,
        zorder=5,
    )
    ax.plot(
        ridge_zero["cq"],
        ridge_zero["sharpe_mean"],
        label="Ridgeless (z = 0)",
        color="black",
        linestyle="--",
        marker="o",
        linewidth=1.8,
    )

    # Plot Lasso curves
    for idx, a in enumerate(target_alphas):
        sub = df2_filtered[df2_filtered["alpha"] == a].sort_values("cq")
        ax.plot(
            sub["cq"],
            sub["sharpe_mean"],
            label=rf"Lasso ($\alpha = {a}$)",
            color=lasso_colors[idx],
            linestyle="-.",
            marker="^",
            alpha=0.9,
        )

    ax.axvline(x=1.0, color="#6C757D", linestyle=":", linewidth=1.5, label="Interpolation ($c_q = 1$)")
    ax.set_xlabel(r"Observed Complexity $c_q = P_1 / T_{tr}$")
    ax.set_ylabel(r"Timing Sharpe Ratio $\mathrm{SR}$ (KMZ Eq. 5)")
    ax.set_title(r"Sharpe Ratio: Optimal Ridge vs. Lasso Inductive Bias", fontweight="bold")
    ax.set_ylim(0.0, 0.075)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_ridge_vs_lasso_sharpe.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig5_ridge_vs_lasso_sharpe.pdf")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 6: Out-of-Sample R^2 vs. c_q (Ridge vs. Lasso)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(
        ridge_opt["cq"],
        ridge_opt["r2_paper_mean"],
        label=r"Optimal Ridge $z^*(q)$",
        color="#D90429",
        linewidth=2.6,
        marker="D",
        markersize=6,
        zorder=5,
    )
    for idx, a in enumerate(target_alphas):
        sub = df2_filtered[df2_filtered["alpha"] == a].sort_values("cq")
        ax.plot(
            sub["cq"],
            sub["r2_paper_mean"],
            label=rf"Lasso ($\alpha = {a}$)",
            color=lasso_colors[idx],
            linestyle="-.",
            marker="^",
            alpha=0.9,
        )

    ax.axvline(x=1.0, color="#6C757D", linestyle=":", linewidth=1.5, label="Interpolation ($c_q = 1$)")
    ax.set_xlabel(r"Observed Complexity $c_q = P_1 / T_{tr}$")
    ax.set_ylabel(r"Out-of-Sample $R^2_{\mathrm{paper}}$")
    ax.set_title(r"Out-of-Sample $R^2_{\mathrm{paper}}$: Ridge vs. Lasso", fontweight="bold")
    ax.set_ylim(-0.4, 0.05)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_ridge_vs_lasso_r2.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig6_ridge_vs_lasso_r2.pdf")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 7: Sparsity Ratio vs. c_q
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for idx, a in enumerate(target_alphas):
        sub = df2_filtered[df2_filtered["alpha"] == a].sort_values("cq")
        ax.plot(
            sub["cq"],
            sub["sparsity_ratio_mean"] * 100.0,
            label=rf"$\alpha = {a}$",
            color=lasso_colors[idx],
            marker="s",
        )

    ax.axvline(x=1.0, color="#6C757D", linestyle=":", linewidth=1.5, label="Interpolation ($c_q = 1$)")
    ax.set_xlabel(r"Observed Complexity $c_q = P_1 / T_{tr}$")
    ax.set_ylabel("Non-Zero Features (%)")
    ax.set_title("Lasso Sparsity: Percentage of Active Features vs. Complexity", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, title=r"Penalty $\alpha$")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_lasso_sparsity_vs_cq.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig7_lasso_sparsity_vs_cq.pdf")
    plt.close(fig)

    # -------------------------------------------------------------
    # Master 3-Panel Figure
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: Sharpe
    axes[0].plot(ridge_opt["cq"], ridge_opt["sharpe_mean"], label=r"Optimal Ridge $z^*(q)$", color="#D90429", linewidth=2.5, marker="D")
    axes[0].plot(ridge_zero["cq"], ridge_zero["sharpe_mean"], label="Ridgeless (z=0)", color="black", linestyle="--")
    for idx, a in enumerate(target_alphas):
        sub = df2_filtered[df2_filtered["alpha"] == a].sort_values("cq")
        axes[0].plot(sub["cq"], sub["sharpe_mean"], label=rf"Lasso $\alpha={a}$", color=lasso_colors[idx], linestyle="-.")
    axes[0].axvline(1.0, color="#6C757D", linestyle=":")
    axes[0].set_xlabel(r"Observed Complexity $c_q$")
    axes[0].set_ylabel("Sharpe Ratio")
    axes[0].set_title("(a) Sharpe Ratio: Ridge vs. Lasso", fontweight="bold")
    axes[0].set_ylim(0.0, 0.075)

    # Panel B: R2
    axes[1].plot(ridge_opt["cq"], ridge_opt["r2_paper_mean"], label=r"Optimal Ridge $z^*$", color="#D90429", linewidth=2.5, marker="D")
    for idx, a in enumerate(target_alphas):
        sub = df2_filtered[df2_filtered["alpha"] == a].sort_values("cq")
        axes[1].plot(sub["cq"], sub["r2_paper_mean"], label=rf"Lasso $\alpha={a}$", color=lasso_colors[idx], linestyle="-.")
    axes[1].axvline(1.0, color="#6C757D", linestyle=":")
    axes[1].set_xlabel(r"Observed Complexity $c_q$")
    axes[1].set_ylabel(r"Out-of-Sample $R^2_{\mathrm{paper}}$")
    axes[1].set_title(r"(b) Out-of-Sample $R^2_{\mathrm{paper}}$", fontweight="bold")
    axes[1].set_ylim(-0.4, 0.05)

    # Panel C: Sparsity
    for idx, a in enumerate(target_alphas):
        sub = df2_filtered[df2_filtered["alpha"] == a].sort_values("cq")
        axes[2].plot(sub["cq"], sub["sparsity_ratio_mean"] * 100.0, label=rf"$\alpha={a}$", color=lasso_colors[idx], marker="s")
    axes[2].axvline(1.0, color="#6C757D", linestyle=":")
    axes[2].set_xlabel(r"Observed Complexity $c_q$")
    axes[2].set_ylabel("Active Features (%)")
    axes[2].set_title("(c) Lasso Active Features (%)", fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.06), frameon=True, fontsize=10)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "task2_master_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "task2_master_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    print("================================================================")
    print("Task 2 figures successfully generated in output/figures/:")
    print("  - fig5_ridge_vs_lasso_sharpe.png / .pdf")
    print("  - fig6_ridge_vs_lasso_r2.png / .pdf")
    print("  - fig7_lasso_sparsity_vs_cq.png / .pdf")
    print("  - task2_master_comparison.png / .pdf")
    print("================================================================")


if __name__ == "__main__":
    plot_task2_figures()
