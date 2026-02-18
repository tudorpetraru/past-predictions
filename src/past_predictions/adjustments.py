from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass
class SplitAdjuster:
    split_dates: list[date]
    split_ratios: list[float]
    end_date: date

    @classmethod
    def from_frame(cls, splits: pd.DataFrame, end_date: date) -> "SplitAdjuster":
        if splits is None or splits.empty:
            return cls([], [], end_date)

        local = splits.copy()
        local["date"] = pd.to_datetime(local["date"]).dt.date
        local["split_ratio"] = pd.to_numeric(local["split_ratio"], errors="coerce")
        local = local.dropna(subset=["date", "split_ratio"])
        local = local[(local["split_ratio"] > 0) & (local["date"] <= end_date)]
        local = local.sort_values("date")

        return cls(
            split_dates=local["date"].tolist(),
            split_ratios=local["split_ratio"].astype(float).tolist(),
            end_date=end_date,
        )

    def factor_to_end(self, value_date: date) -> float:
        if not self.split_dates:
            return 1.0
        ratios = [
            r for d, r in zip(self.split_dates, self.split_ratios) if d > value_date and d <= self.end_date
        ]
        if not ratios:
            return 1.0
        return float(np.prod(ratios))

    def adjust_value(self, value: float, value_date: date) -> float:
        return float(value) / self.factor_to_end(value_date)

    def has_split_adjustment(self, value_date: date) -> bool:
        return self.factor_to_end(value_date) != 1.0
