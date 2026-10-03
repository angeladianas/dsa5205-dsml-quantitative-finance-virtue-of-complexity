"""Task 4: Empirical Demonstration of Dynamical Decoupling in Neural Networks.

Based on Montanari & Urbani (NeurIPS 2025):
"Dynamical Decoupling of Generalization and Overfitting in Large Two-Layer Networks"

Demonstrates the timescale separation in low-SNR financial return prediction:
1. Fast Feature Learning (t ~ O(1)): Generalization improves, test Sharpe climbs to peak.
2. Optimal Generalization (t*): Maximum out-of-sample trading performance.
3. Slow Noise Memorization & Feature Unlearning (t ~ O(m)): Weight norm expands,
   training loss approaches zero, but test Sharpe decays due to noise interpolation.
"""

import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, GLOBAL_SEED, RESULTS_DIR
from src.metrics import (
    compute_r2_paper,
    compute_sharpe_uncentered,
    compute_timing_returns,
)


class TwoLayerNetwork:
    """Two-layer fully connected neural network f(x) = (1/m) * sum_i a_i * tanh(w_i^T x)."""

    def __init__(self, d: int, m: int, lr: float = 0.05, seed: int = GLOBAL_SEED):
        self.d = d
        self.m = m
        self.lr = lr
        rng = np.random.default_rng(seed)

        # First layer weights W in R^{m x d}, normalized to sphere ||w_i|| = 1
        W_raw = rng.standard_normal(size=(m, d))
        self.W = W_raw / np.linalg.norm(W_raw, axis=1, keepdims=True)

        # Second layer weights a in R^m, initialized to small scale a0
        self.a = 0.1 * rng.standard_normal(size=m)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass: returns predictions y_hat and hidden activations h."""
        # X: (N, d), W: (m, d) -> Z: (N, m)
        Z = X @ self.W.T
        h = np.tanh(Z)
        # f(x) = (1/m) * h @ a
        y_hat = (h @ self.a) / self.m
        return y_hat, h

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """Single gradient descent step on empirical squared loss (1/2N) * ||y - y_hat||^2."""
        N = X.shape[0]
        y_hat, h = self.forward(X)
        errors = y_hat - y  # (N,)

        # Compute MSE loss
        loss = float(0.5 * np.mean(errors**2))

        # Gradient w.r.t second layer weights a:
        # dL/da = (1 / (N * m)) * h^T @ errors
        grad_a = (h.T @ errors) / (N * self.m)

        # Gradient w.r.t first layer weights W:
        # dL/dZ = (errors[:, None] * a[None, :]) * (1 - h^2) / (N * m)
        dZ = (errors[:, None] * self.a[None, :]) * (1.0 - h**2) / (N * self.m)
        grad_W = dZ.T @ X  # (m, d)

        # Parameter updates
        self.a -= self.lr * grad_a
        self.W -= self.lr * grad_W

        # Project first-layer rows back to unit sphere (as in paper Section 1)
        self.W = self.W / np.linalg.norm(self.W, axis=1, keepdims=True)

        return loss

    @property
    def second_layer_l1_norm(self) -> float:
        """Proxy for model complexity: ||a||_1."""
        return float(np.sum(np.abs(self.a)))


def generate_task4_data(
    t_tr: int = 200,
    t_te: int = 1000,
    d: int = 50,
    snr: float = 0.05,
    seed: int = GLOBAL_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate financial return data with low signal-to-noise ratio."""
    rng = np.random.default_rng(seed)
    N = t_tr + t_te

    # Features X ~ N(0, I_d)
    X = rng.standard_normal(size=(N, d))

    # Low-dimensional rank-1 signal vector u
    u = rng.standard_normal(size=d)
    u /= np.linalg.norm(u)

    # Nonlinear signal: g(u^T x)
    signal = np.sqrt(snr) * np.sin(X @ u)
    noise = rng.standard_normal(size=N)

    # Realized returns
    y = signal + noise

    return X[:t_tr], y[:t_tr], X[t_tr:], y[t_tr:]


def run_task4_experiment(epochs: int = 1000, lr: float = 0.08, m_hidden: int = 150) -> pd.DataFrame:
    """Train two-layer neural network across epochs and track dynamical decoupling."""
    print("================================================================")
    print("TASK 4: DYNAMICAL DECOUPLING EXPERIMENT (Montanari & Urbani 2025)")
    print(f"Epochs: {epochs} | Hidden Units: {m_hidden} | Learning Rate: {lr}")
    print("================================================================")

    start_time = time.time()
    X_tr, y_tr, X_te, y_te = generate_task4_data()

    net = TwoLayerNetwork(d=X_tr.shape[1], m=m_hidden, lr=lr)

    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        tr_loss = net.train_step(X_tr, y_tr)

        if epoch % 5 == 0 or epoch == 1:
            y_pred_te, _ = net.forward(X_te)
            r_pi_te = compute_timing_returns(y_te, y_pred_te)
            te_mse = float(0.5 * np.mean((y_te - y_pred_te) ** 2))
            te_sharpe = compute_sharpe_uncentered(r_pi_te)
            te_r2 = compute_r2_paper(y_te, y_pred_te)
            l1_norm = net.second_layer_l1_norm

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "test_mse": te_mse,
                    "test_sharpe": te_sharpe,
                    "test_r2": te_r2,
                    "second_layer_l1": l1_norm,
                }
            )

    df_hist = pd.DataFrame(history)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    history_csv = RESULTS_DIR / "task4_dynamics_history.csv"
    df_hist.to_csv(history_csv, index=False)

    print(f"Experiment completed in {time.time() - start_time:.2f}s!")
    print(f"Saved training history to: {history_csv}")

    # Plot results
    plot_task4_dynamics(df_hist)
    return df_hist


def plot_task4_dynamics(df: pd.DataFrame) -> None:
    """Render publication figure showing the three dynamical learning regimes."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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
    ax1.set_xlabel("Training Epochs (Gradient Descent Time $t$)")
    ax1.set_ylabel(
        r"Out-of-Sample Sharpe Ratio $\mathrm{SR}$",
        color=color_sharpe,
        fontweight="bold",
    )
    ax1.tick_params(axis="y", labelcolor=color_sharpe)

    # Find peak epoch
    peak_row = df.loc[df["test_sharpe"].idxmax()]
    t_star = int(peak_row["epoch"])
    sr_peak = peak_row["test_sharpe"]

    ax1.axvline(
        x=t_star,
        color="#E76F51",
        linestyle="--",
        linewidth=2.0,
        label=f"Optimal Early Stopping ($t^* = {t_star}$)",
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
        df["epoch"].max(),
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
    run_task4_experiment()
