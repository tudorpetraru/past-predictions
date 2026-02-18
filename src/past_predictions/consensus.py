from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .adjustments import SplitAdjuster
from .prices import build_horizon_map, trading_days, weekly_asof_dates

REQUIRED_COLUMNS = ["ticker", "date", "predicted_min", "predicted_avg", "predicted_max", "actual"]
RECOMMENDED_COLUMNS = [
    "actual_12m",
    "error_avg_12m",
    "hit_within_range_12m",
    "n_analysts",
    "data_quality_flags",
]
ALL_COLUMNS = REQUIRED_COLUMNS + RECOMMENDED_COLUMNS


def select_active_targets(events: pd.DataFrame, asof_date: date, ttl_days: int) -> pd.DataFrame:
    if events.empty:
        return events
    ttl_start = asof_date - timedelta(days=ttl_days)
    filtered = events[(events["event_date"] <= asof_date) & (events["event_date"] >= ttl_start)].copy()
    if filtered.empty:
        return filtered
    latest = filtered.sort_values("event_ts").groupby("analyst_key", as_index=False).tail(1)
    return latest.reset_index(drop=True)


def aggregate_targets(active_events: pd.DataFrame) -> tuple[float, float, float, int]:
    if active_events.empty:
        return (np.nan, np.nan, np.nan, 0)
    values = pd.to_numeric(active_events["target_price_adj"], errors="coerce").dropna()
    if values.empty:
        return (np.nan, np.nan, np.nan, 0)
    return (float(values.min()), float(values.mean()), float(values.max()), int(values.shape[0]))


def _provider_flag(provider: str) -> str:
    if provider == "yahoo":
        return "SOURCE_YAHOO"
    if provider == "fmp":
        return "SOURCE_FMP_FALLBACK"
    return ""


def _safe_read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    return pd.read_parquet(path)


def compute_weekly_dataset(
    universe: pd.DataFrame,
    events: pd.DataFrame,
    provider_selection: pd.DataFrame,
    price_dir: str | Path,
    split_dir: str | Path,
    start: date,
    end: date,
    ttl_days: int,
    min_analysts: int,
    calendar: str,
    horizon_days: int,
) -> pd.DataFrame:
    weekly_dates = weekly_asof_dates(start=start, end=end, calendar=calendar)
    all_trading_days = trading_days(start=start, end=end + timedelta(days=500), calendar=calendar)
    horizon_map = build_horizon_map(weekly_dates, all_trading_days, horizon_days=horizon_days)

    provider_lookup = {}
    if not provider_selection.empty and "ticker" in provider_selection.columns and "provider" in provider_selection.columns:
        provider_lookup = dict(zip(provider_selection["ticker"], provider_selection["provider"]))

    events = events.copy()
    if not events.empty:
        events["event_ts"] = pd.to_datetime(events["event_ts"], errors="coerce")
        events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce").dt.date
        events = events.dropna(subset=["event_ts", "event_date", "analyst_key", "target_price"])

    rows: list[dict[str, object]] = []

    for ticker in sorted(universe["ticker_norm"].astype(str).unique()):
        provider = provider_lookup.get(ticker, "none")
        ticker_events = events[events["ticker"] == ticker].copy() if not events.empty else pd.DataFrame()

        prices = _safe_read_parquet(Path(price_dir) / f"{ticker}.parquet", ["date", "close"])
        splits = _safe_read_parquet(Path(split_dir) / f"{ticker}.parquet", ["date", "split_ratio"])

        if not prices.empty:
            prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.date
            prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
            prices = prices.dropna(subset=["date", "close"])
        price_lookup = dict(zip(prices["date"], prices["close"])) if not prices.empty else {}

        adjuster = SplitAdjuster.from_frame(splits=splits, end_date=end)

        if not ticker_events.empty:
            ticker_events["target_price"] = pd.to_numeric(ticker_events["target_price"], errors="coerce")
            ticker_events = ticker_events.dropna(subset=["target_price", "event_ts", "event_date"])
            ticker_events["target_price_adj"] = ticker_events.apply(
                lambda r: adjuster.adjust_value(float(r["target_price"]), r["event_date"]), axis=1
            )
            ticker_events = ticker_events.sort_values("event_ts").reset_index(drop=True)

        for asof in weekly_dates:
            flags: set[str] = set()
            source_flag = _provider_flag(provider)
            if source_flag:
                flags.add(source_flag)

            active = select_active_targets(ticker_events, asof, ttl_days=ttl_days) if not ticker_events.empty else pd.DataFrame()
            predicted_min, predicted_avg, predicted_max, n_analysts = aggregate_targets(active)

            if n_analysts == 0:
                flags.add("NO_TARGETS")
            if 0 < n_analysts < min_analysts:
                flags.add("LOW_COVERAGE")

            actual = np.nan
            close = price_lookup.get(asof)
            if close is None or pd.isna(close):
                flags.add("NO_PRICE")
            else:
                actual = adjuster.adjust_value(float(close), asof)
                if adjuster.has_split_adjustment(asof):
                    flags.add("SPLIT_ADJUSTED")

            actual_12m = np.nan
            horizon_date = horizon_map.get(asof)
            if horizon_date is None:
                flags.add("NO_12M_PRICE")
            else:
                horizon_close = price_lookup.get(horizon_date)
                if horizon_close is None or pd.isna(horizon_close):
                    flags.add("NO_12M_PRICE")
                else:
                    actual_12m = adjuster.adjust_value(float(horizon_close), horizon_date)
                    if adjuster.has_split_adjustment(horizon_date):
                        flags.add("SPLIT_ADJUSTED")

            if not active.empty and active["event_date"].map(adjuster.has_split_adjustment).any():
                flags.add("SPLIT_ADJUSTED")

            error_avg_12m = np.nan
            hit_within_range_12m = np.nan
            if not np.isnan(actual_12m) and not np.isnan(predicted_avg):
                error_avg_12m = float(actual_12m - predicted_avg)
            if not np.isnan(actual_12m) and not np.isnan(predicted_min) and not np.isnan(predicted_max):
                hit_within_range_12m = bool(predicted_min <= actual_12m <= predicted_max)

            rows.append(
                {
                    "ticker": ticker,
                    "date": asof.isoformat(),
                    "predicted_min": predicted_min,
                    "predicted_avg": predicted_avg,
                    "predicted_max": predicted_max,
                    "actual": actual,
                    "actual_12m": actual_12m,
                    "error_avg_12m": error_avg_12m,
                    "hit_within_range_12m": hit_within_range_12m,
                    "n_analysts": int(n_analysts),
                    "data_quality_flags": ";".join(sorted(flags)),
                }
            )

    frame = pd.DataFrame(rows, columns=ALL_COLUMNS)
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    return frame
