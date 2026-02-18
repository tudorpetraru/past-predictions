from __future__ import annotations

from pathlib import Path

import pandas as pd


def _to_markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        values = ["" if pd.isna(v) else str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def generate_spot_check_report(
    weekly_frame: pd.DataFrame,
    tickers: list[str],
    weeks: int,
    out_path: str | Path,
) -> None:
    rows = []
    for ticker in tickers:
        subset = weekly_frame[weekly_frame["ticker"] == ticker].sort_values("date").tail(weeks)
        rows.append(f"## {ticker}")
        if subset.empty:
            rows.append("No rows available.\n")
            continue
        display = subset[
            [
                "date",
                "predicted_min",
                "predicted_avg",
                "predicted_max",
                "actual",
                "n_analysts",
                "data_quality_flags",
            ]
        ]
        rows.append(_to_markdown_table(display))
        rows.append("")

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(rows), encoding="utf-8")
