from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analyst_panel import ANALYST_COLUMNS
from .consensus import ALL_COLUMNS, REQUIRED_COLUMNS


def export_csv(
    weekly_parquet_path: str | Path,
    out_csv_path: str | Path,
    required_only: bool = False,
) -> pd.DataFrame:
    frame = pd.read_parquet(weekly_parquet_path)
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)

    expected = REQUIRED_COLUMNS if required_only else ALL_COLUMNS
    for col in expected:
        if col not in frame.columns:
            frame[col] = pd.NA

    output = frame[expected]
    out_path = Path(out_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    return output


def export_analyst_csv(
    analyst_parquet_path: str | Path,
    out_csv_path: str | Path,
) -> pd.DataFrame:
    frame = pd.read_parquet(analyst_parquet_path)
    frame = frame.sort_values(["ticker", "date", "analyst_key"]).reset_index(drop=True)

    for col in ANALYST_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA

    output = frame[ANALYST_COLUMNS]
    out_path = Path(out_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    return output
