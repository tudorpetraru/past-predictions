from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .adjustments import SplitAdjuster
from .consensus import select_active_targets
from .prices import build_horizon_map, trading_days, weekly_asof_dates

ANALYST_COLUMNS = [
    "ticker",
    "date",
    "analyst_key",
    "predicted",
    "actual",
    "actual_12m",
    "error_12m",
    "abs_error_12m",
    "hit_direction",
    "data_quality_flags",
]


def _safe_read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    return pd.read_parquet(path)


def _provider_flag(provider: str) -> str:
    if provider == "yahoo":
        return "SOURCE_YAHOO"
    if provider == "fmp":
        return "SOURCE_FMP_FALLBACK"
    return ""


def _compute_hit_direction(predicted: float, actual: float, actual_12m: float) -> bool:
    implied_move = predicted - actual
    realized_move = actual_12m - actual
    return bool(np.sign(implied_move) == np.sign(realized_move))


def compute_analyst_weekly_dataset(
    universe: pd.DataFrame,
    events: pd.DataFrame,
    provider_selection: pd.DataFrame,
    price_dir: str | Path,
    split_dir: str | Path,
    pred_start: date,
    pred_end: date,
    actual_start: date,
    actual_end: date,
    ttl_days: int,
    calendar: str,
    horizon_days: int,
) -> pd.DataFrame:
    weekly_dates = weekly_asof_dates(start=pred_start, end=pred_end, calendar=calendar)
    trading_end = max(actual_end, pred_end + timedelta(days=500))
    all_trading_days = trading_days(start=pred_start, end=trading_end, calendar=calendar)
    horizon_map = build_horizon_map(weekly_dates, all_trading_days, horizon_days=horizon_days)

    provider_lookup = {}
    if not provider_selection.empty and "ticker" in provider_selection.columns and "provider" in provider_selection.columns:
        provider_lookup = dict(zip(provider_selection["ticker"], provider_selection["provider"]))

    events = events.copy()
    if not events.empty:
        events["event_ts"] = pd.to_datetime(events["event_ts"], errors="coerce")
        events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce").dt.date
        events["target_price"] = pd.to_numeric(events["target_price"], errors="coerce")
        events = events.dropna(subset=["event_ts", "event_date", "analyst_key", "target_price"])

    rows: list[dict[str, object]] = []

    for ticker in sorted(universe["ticker_norm"].astype(str).unique()):
        provider = provider_lookup.get(ticker, "none")
        source_flag = _provider_flag(provider)
        ticker_events = events[events["ticker"] == ticker].copy() if not events.empty else pd.DataFrame()

        if ticker_events.empty:
            continue

        prices = _safe_read_parquet(Path(price_dir) / f"{ticker}.parquet", ["date", "close"])
        splits = _safe_read_parquet(Path(split_dir) / f"{ticker}.parquet", ["date", "split_ratio"])

        if not prices.empty:
            prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.date
            prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
            prices = prices.dropna(subset=["date", "close"])
        price_lookup = dict(zip(prices["date"], prices["close"])) if not prices.empty else {}

        # Keep all values on a common split basis at actual_end.
        adjuster = SplitAdjuster.from_frame(splits=splits, end_date=actual_end)

        ticker_events["target_price_adj"] = ticker_events.apply(
            lambda r: adjuster.adjust_value(float(r["target_price"]), r["event_date"]), axis=1
        )
        ticker_events = ticker_events.sort_values("event_ts").reset_index(drop=True)

        for asof in weekly_dates:
            active = select_active_targets(ticker_events, asof, ttl_days=ttl_days)
            if active.empty:
                continue

            horizon_date = horizon_map.get(asof)
            if horizon_date is None:
                continue
            if not (actual_start <= horizon_date <= actual_end):
                continue

            close = price_lookup.get(asof)
            actual = np.nan if close is None or pd.isna(close) else adjuster.adjust_value(float(close), asof)

            horizon_close = price_lookup.get(horizon_date)
            actual_12m = (
                np.nan
                if horizon_close is None or pd.isna(horizon_close)
                else adjuster.adjust_value(float(horizon_close), horizon_date)
            )

            for _, event_row in active.iterrows():
                flags: set[str] = set()
                if source_flag:
                    flags.add(source_flag)

                if np.isnan(actual):
                    flags.add("NO_PRICE")
                if np.isnan(actual_12m):
                    flags.add("NO_12M_PRICE")

                if adjuster.has_split_adjustment(asof) or adjuster.has_split_adjustment(horizon_date):
                    flags.add("SPLIT_ADJUSTED")
                if adjuster.has_split_adjustment(event_row["event_date"]):
                    flags.add("SPLIT_ADJUSTED")

                predicted = float(event_row["target_price_adj"])

                error_12m = np.nan
                abs_error_12m = np.nan
                hit_direction = np.nan
                if not np.isnan(actual_12m):
                    error_12m = float(actual_12m - predicted)
                    abs_error_12m = float(abs(error_12m))
                if not np.isnan(actual) and not np.isnan(actual_12m):
                    hit_direction = _compute_hit_direction(predicted=predicted, actual=float(actual), actual_12m=float(actual_12m))

                rows.append(
                    {
                        "ticker": ticker,
                        "date": asof.isoformat(),
                        "analyst_key": event_row["analyst_key"],
                        "predicted": predicted,
                        "actual": actual,
                        "actual_12m": actual_12m,
                        "error_12m": error_12m,
                        "abs_error_12m": abs_error_12m,
                        "hit_direction": hit_direction,
                        "data_quality_flags": ";".join(sorted(flags)),
                    }
                )

    frame = pd.DataFrame(rows, columns=ANALYST_COLUMNS)
    frame = frame.sort_values(["ticker", "date", "analyst_key"]).reset_index(drop=True)
    return frame
