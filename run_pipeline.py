#!/usr/bin/env python3
"""Master One-Click Execution Pipeline for DSA5205 Project 1.

Usage:
    python run_pipeline.py --all           # Execute entire pipeline (Tasks 1-4 + test suite)
    python run_pipeline.py --task 1        # Execute Task 1 (Misspecified Ridge simulation)
    python run_pipeline.py --task 2        # Execute Task 2 (Ridge vs Lasso benchmark)
    python run_pipeline.py --task 3        # Execute Task 3 (Empirical predictions A, B, C)
    python run_pipeline.py --task 4        # Execute Task 4 (Dynamical decoupling in neural net)
    python run_pipeline.py --test          # Run full pytest test suite
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import FIGURES_DIR, PREDICTIONS_DIR, RESULTS_DIR, STUDENT_ID


def print_banner(title: str) -> None:
    """Print a visually distinct banner."""
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)


def run_command(cmd: list[str], description: str) -> bool:
    """Execute a subprocess command with timing and error handling."""
    print(f"\n[RUNNING] {description}...")
    start = time.time()
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        elapsed = time.time() - start
        print(f"[SUCCESS] {description} completed in {elapsed:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"[FAILED]  {description} failed after {elapsed:.2f}s with code {e.returncode}")
        return False


def run_task_1(quick: bool = False) -> bool:
    """Execute Task 1: Ridge simulation and figure plotting."""
    print_banner("Task 1: Misspecified Ridge Simulation & Double Descent")
    runs = 5 if quick else 50
    cmd_sim = [sys.executable, "-m", "src.task1_sim", "--runs", str(runs)]
    cmd_plot = [sys.executable, "-m", "src.task1_plot"]

    if not run_command(cmd_sim, f"Task 1 Monte Carlo Simulation (M={runs})"):
        return False
    return run_command(cmd_plot, "Task 1 Figure Generation")


def run_task_2() -> bool:
    """Execute Task 2: Ridge vs Lasso benchmark and comparison plots."""
    print_banner("Task 2: Comparative Inductive Biases (Ridge vs. Lasso)")
    cmd_bench = [sys.executable, "-m", "src.task2_benchmark"]
    cmd_plot = [sys.executable, "-m", "src.task2_plot"]

    if not run_command(cmd_bench, "Task 2 Lasso Benchmark Sweep"):
        return False
    return run_command(cmd_plot, "Task 2 Comparison Figure Generation")


def run_task_3() -> bool:
    """Execute Task 3: Empirical predictions for Datasets A, B, and C."""
    print_banner("Task 3: Empirical Predictions for Datasets A, B, and C")
    cmd_predict = [sys.executable, "-m", "src.task3_predict"]
    return run_command(cmd_predict, "Task 3 Cross-Validation & Prediction Export")


def run_task_4() -> bool:
    """Execute Task 4: Dynamical decoupling experiment and visualization."""
    print_banner("Task 4: Dynamical Decoupling in Two-Layer Neural Network")
    cmd_dynamics = [sys.executable, "-m", "src.task4_dynamics"]
    cmd_plot = [sys.executable, "-m", "src.task4_plot"]

    if not run_command(cmd_dynamics, "Task 4 Neural Network Dynamical Simulation"):
        return False
    return run_command(cmd_plot, "Task 4 Figure Generation")


def run_tests() -> bool:
    """Execute unit test suite."""
    print_banner("Automated Test Suite (pytest)")
    cmd_test = [sys.executable, "-m", "pytest", "-v", "tests/"]
    return run_command(cmd_test, "Unit & Integration Test Suite")


def print_summary() -> None:
    """Print comprehensive summary of generated artifacts."""
    print_banner("Deliverable Verification & Artifact Summary")

    print(f"\n1. Prediction Submissions ({PREDICTIONS_DIR}):")
    for name in ["A", "B", "C"]:
        p = PREDICTIONS_DIR / f"{STUDENT_ID}_predictions_{name}.csv"
        if p.exists():
            size_kb = p.stat().st_size / 1024
            with open(p) as f:
                line_count = sum(1 for _ in f) - 1
            print(f"   [OK] {p.name:<30} ({size_kb:.1f} KB, {line_count} predictions)")
        else:
            print(f"   [MISSING] {p.name}")

    print(f"\n2. Numerical Results ({RESULTS_DIR}):")
    for f in sorted(RESULTS_DIR.glob("*.csv")):
        size_kb = f.stat().st_size / 1024
        print(f"   [OK] {f.name:<30} ({size_kb:.1f} KB)")

    print(f"\n3. Publication Figures ({FIGURES_DIR}):")
    for f in sorted(FIGURES_DIR.glob("*.pdf")):
        png = f.with_suffix(".png")
        png_status = "PDF+PNG" if png.exists() else "PDF only"
        print(f"   [OK] {f.name:<32} ({png_status})")

    print("\n" + "=" * 70)
    print(" ALL TASKS EXECUTED AND VERIFIED SUCCESSFULLY!")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Master execution pipeline for DSA5205 Project 1.")
    parser.add_argument(
        "--all", action="store_true", help="Run entire pipeline (Tasks 1-4 + tests)"
    )
    parser.add_argument(
        "--task",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific task (1, 2, 3, or 4)",
    )
    parser.add_argument("--test", action="store_true", help="Run pytest verification suite")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run Task 1 with M=5 runs for fast smoke testing",
    )

    args = parser.parse_args()

    total_start = time.time()

    if args.task == 1:
        success = run_task_1(quick=args.quick)
    elif args.task == 2:
        success = run_task_2()
    elif args.task == 3:
        success = run_task_3()
    elif args.task == 4:
        success = run_task_4()
    elif args.test:
        success = run_tests()
    elif args.all:
        success = (
            run_task_1(quick=args.quick)
            and run_task_2()
            and run_task_3()
            and run_task_4()
            and run_tests()
        )
        if success:
            print_summary()
    else:
        parser.print_help()
        sys.exit(0)

    total_elapsed = time.time() - total_start
    if not success:
        print(f"\n[ABORTED] Pipeline failed after {total_elapsed:.2f}s.")
        sys.exit(1)
    else:
        print(f"\n[FINISHED] Pipeline completed successfully in {total_elapsed:.2f}s.")


if __name__ == "__main__":
    main()
