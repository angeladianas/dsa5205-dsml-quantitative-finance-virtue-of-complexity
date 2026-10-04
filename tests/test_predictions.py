"""Verification suite for Task 3 submission prediction CSVs."""

import numpy as np
import pandas as pd
import pytest

from src.config import PREDICTIONS_DIR, STUDENT_ID
from src.task3_predict import load_data_csv


@pytest.mark.parametrize(
    "dataset_name,expected_rows,t_start,t_end",
    [
        ("A", 1000, 601, 1600),
        ("B", 4000, 241, 4240),
        ("C", 2000, 361, 2360),
    ],
)
def test_prediction_file_integrity(dataset_name: str, expected_rows: int, t_start: int, t_end: int):
    """Verify formatting, ordering, row counts, and values of prediction CSVs."""
    filename = f"{STUDENT_ID}_predictions_{dataset_name}.csv"
    pred_path = PREDICTIONS_DIR / filename

    # 1. File existence
    assert pred_path.exists(), f"Missing prediction file: {pred_path}"

    # 2. Read file and inspect columns
    df_pred = pd.read_csv(pred_path)
    assert list(df_pred.columns) == ["t", "yhat"], (
        f"Incorrect header in {filename}: found {list(df_pred.columns)}, expected ['t', 'yhat']"
    )

    # 3. Exact row count
    assert len(df_pred) == expected_rows, (
        f"Row count mismatch in {filename}: found {len(df_pred)}, expected {expected_rows}"
    )

    # 4. Strict match with test features 't' column
    df_test = load_data_csv(f"pair{dataset_name}_test_features.csv")
    assert np.array_equal(df_pred["t"].values, df_test["t"].values), (
        f"Column 't' in {filename} does not match public test feature file ordering"
    )
    assert df_pred["t"].iloc[0] == t_start
    assert df_pred["t"].iloc[-1] == t_end
    assert df_pred["t"].is_monotonic_increasing

    # 5. Numerical validity
    yhat = df_pred["yhat"].values
    assert np.all(np.isfinite(yhat)), f"Found NaN or Inf in {filename}"
    assert np.std(yhat) > 0.0, f"Predictions in {filename} are degenerate (constant)"


def test_load_data_csv_compressed_formats(tmp_path, monkeypatch):
    """Verify that load_data_csv transparently reads .csv, .csv.gz, and .zip files."""
    import gzip
    import zipfile

    import src.task3_predict as t3

    # Point DATA_DIR to tmp_path for isolated unit test
    monkeypatch.setattr(t3, "DATA_DIR", tmp_path)

    sample_df = pd.DataFrame({"t": [1, 2, 3], "val": [0.1, 0.2, 0.3]})

    # 1. Test plain CSV
    csv_file = tmp_path / "test_sample.csv"
    sample_df.to_csv(csv_file, index=False)
    loaded = t3.load_data_csv("test_sample.csv")
    assert loaded.shape == (3, 2)
    csv_file.unlink()

    # 2. Test .csv.gz
    gz_file = tmp_path / "test_sample.csv.gz"
    with gzip.open(gz_file, "wt") as f:
        sample_df.to_csv(f, index=False)
    loaded = t3.load_data_csv("test_sample.csv")
    assert loaded.shape == (3, 2)
    gz_file.unlink()

    # 3. Test .zip
    zip_file = tmp_path / "test_sample.zip"
    with zipfile.ZipFile(zip_file, "w") as zf:
        zf.writestr("test_sample.csv", sample_df.to_csv(index=False))
    loaded = t3.load_data_csv("test_sample.csv")
    assert loaded.shape == (3, 2)
    zip_file.unlink()

    # 4. Test missing file error
    with pytest.raises(FileNotFoundError):
        t3.load_data_csv("nonexistent_dataset.csv")
