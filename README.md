# The Virtue of Complexity in Return Prediction
## DSA5205 Quantitative Finance & Machine Learning — Project 1

**Student ID:** `A0327258X`  
**Repository:** `dsa5205-dsml-quantitative-finance-virtue-of-complexity`

This repository contains the complete empirical reproduction, algorithmic implementations, and research codebase for DSA5205 Project 1. The study investigates how model complexity, regularization inductive biases (Ridge vs. Lasso), and neural network dynamical decoupling govern out-of-sample predictability and trading strategy returns in financial markets.

---

## 1. Quickstart & Installation

This project uses [`uv`](https://github.com/astral-sh/uv) inside a virtual environment (`.venv`) for deterministic, fast package management. A standard `requirements.txt` is also provided.

### Option A: Using `uv` (Recommended)
```bash
# Clone repository
git clone https://github.com/angeladianas/dsa5205-dsml-quantitative-finance-virtue-of-complexity.git
cd dsa5205-dsml-quantitative-finance-virtue-of-complexity

# Create virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Option B: Standard `python3 -m venv`
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. One-Click Reproduction Pipeline

All tasks and unit tests can be reproduced with a single command via `run_pipeline.py`:

```bash
# Run the complete end-to-end pipeline (Tasks 1-4 + test suite)
python run_pipeline.py --all

# Run automated unit test verification suite
python run_pipeline.py --test

# Run individual tasks
python run_pipeline.py --task 1   # Task 1: Ridge simulation & Double Descent
python run_pipeline.py --task 2   # Task 2: Ridge vs Lasso inductive bias benchmark
python run_pipeline.py --task 3   # Task 3: Empirical predictions for Datasets A, B, and C
python run_pipeline.py --task 4   # Task 4: Dynamical decoupling in two-layer neural net
```

---

## 3. Project Structure

```
dsa5205-dsml-quantitative-finance-virtue-of-complexity/
├── README.md                      # Complete reproduction guide
├── requirements.txt               # Locked dependencies
├── pytest.ini                     # Pytest configuration
├── run_pipeline.py                # Master execution pipeline
├── data/                          # Train & test CSVs (A, B, C)
├── src/
│   ├── __init__.py
│   ├── config.py                  # Global paths, constants, seed
│   ├── metrics.py                 # Pure functional metrics (R2_paper, Sharpe, etc.)
│   ├── solvers.py                 # Numerical SVD ridge/OLS solvers (never inverts X^T X)
│   ├── task1_sim.py               # Task 1 Monte Carlo simulation engine
│   ├── task1_plot.py              # Task 1 figure generator
│   ├── task2_benchmark.py         # Task 2 Lasso benchmark sweep
│   ├── task2_plot.py              # Task 2 comparison figure generator
│   ├── task3_predict.py           # Task 3 CV tournament & prediction generator
│   └── task4_dynamics.py          # Task 4 Montanari & Urbani dynamical decoupling
├── output/
│   ├── figures/                   # High-res publication PDFs and PNGs (Figures 1-8)
│   ├── predictions/               # Submission files (A0327258X_predictions_{A,B,C}.csv)
│   └── results/                   # Cached CSV simulation runs and summary tables
└── tests/
    ├── test_metrics.py            # Unit tests for pure financial metrics
    ├── test_solvers.py            # Numerical stability & equivalence tests for SVD solvers
    └── test_predictions.py        # Submission format, header, and integrity tests
```

---

## 4. Deliverables & Submission Artifacts

### 4.1 Prediction CSV Files (`output/predictions/`)
Verified by `tests/test_predictions.py` (100% pass):
- `A0327258X_predictions_A.csv`: 1,000 predictions ($t = 601 \dots 1600$), Tuned SVD Ridge ($z = 10.0$).
- `A0327258X_predictions_B.csv`: 4,000 predictions ($t = 241 \dots 4240$), Tuned Lasso ($\alpha = 0.08$).
- `A0327258X_predictions_C.csv`: 2,000 predictions ($t = 361 \dots 2360$), PCA-Ridge ($k = 10, \alpha = 0.1$).

### 4.2 Research Report
- LaTeX source: `Scratch/05_LATEX_REPORT_STRUCTURE.tex`
- Rendered PDF: `A0327258X_report.pdf` (compliant with IEEE/ACM formatting, 5–15 pages).

---

## 5. Automated Verification Suite
Run the 17-test verification suite at any time:
```bash
pytest -v tests/
```
Tests assert:
1. Exact mathematical equivalence between SVD solvers and standard matrix inversions.
2. Numerical stability of the SVD solver at the interpolation boundary $c_q \approx 1.0$.
3. Exact metric definitions ($R^2_{paper}$, uncentered Sharpe ratio, parameter norm).
4. Strict prediction file compliance: correct header `t,yhat`, strictly increasing $t$, correct row counts, and zero missing/infinite values.