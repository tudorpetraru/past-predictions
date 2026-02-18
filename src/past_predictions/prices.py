from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class PriceFetchResult:
    ticker: str
    status: str
    rows: int
    error: str = ""


def _clamp_sessions_range(cal, start: date, end: date) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    first = pd.Timestamp(cal.first_session)
    last = pd.Timestamp(cal.last_session)
    start_ts = max(start_ts, first)
    end_ts = min(end_ts, last)
    if start_ts > end_ts:
        return None
    return start_ts, end_ts


def weekly_asof_dates(start: date, end: date, calendar: str = "XNYS") -> list[date]:
    cal = xcals.get_calendar(calendar)
    bounds = _clamp_sessions_range(cal, start, end)
    if bounds is None:
        return []
    sessions = cal.sessions_in_range(bounds[0], bounds[1])
    if len(sessions) == 0:
        return []
    days = pd.DatetimeIndex(sessions).tz_localize(None).normalize()
    weekly = (
        pd.Series(days, index=days)
        .groupby(days.to_period("W-FRI"))
        .max()
        .sort_values()
        .dt.date.tolist()
    )
    return weekly


def trading_days(start: date, end: date, calendar: str = "XNYS") -> list[date]:
    cal = xcals.get_calendar(calendar)
    bounds = _clamp_sessions_range(cal, start, end)
    if bounds is None:
        return []
    sessions = cal.sessions_in_range(bounds[0], bounds[1])
    if len(sessions) == 0:
        return []
    return pd.DatetimeIndex(sessions).tz_localize(None).date.tolist()


def build_horizon_map(
    asof_dates: list[date],
    trading_day_list: list[date],
    horizon_days: int = 252,
) -> dict[date, date | None]:
    lookup = {d: i for i, d in enumerate(trading_day_list)}
    out: dict[date, date | None] = {}
    for asof in asof_dates:
        idx = lookup.get(asof)
        if idx is None:
            out[asof] = None
            continue
        target_idx = idx + horizon_days
        out[asof] = trading_day_list[target_idx] if target_idx < len(trading_day_list) else None
    return out


def _normalize_history(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame is None or frame.empty:
        return (
            pd.DataFrame(columns=["date", "close"]),
            pd.DataFrame(columns=["date", "split_ratio"]),
        )

    local = frame.copy()
    local.index = pd.to_datetime(local.index).tz_localize(None)
    local["date"] = local.index.date

    prices = local[["date", "Close"]].rename(columns={"Close": "close"})
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna(subset=["close"]).reset_index(drop=True)

    split_col = "Stock Splits" if "Stock Splits" in local.columns else None
    if split_col is None:
        splits = pd.DataFrame(columns=["date", "split_ratio"])
    else:
        splits = local[["date", split_col]].rename(columns={split_col: "split_ratio"})
        splits["split_ratio"] = pd.to_numeric(splits["split_ratio"], errors="coerce").fillna(0.0)
        splits = splits[splits["split_ratio"] > 0].reset_index(drop=True)

    return prices, splits


def fetch_price_history(
    ticker: str,
    out_price_dir: str | Path,
    out_split_dir: str | Path,
    start: date,
    end: date,
) -> PriceFetchResult:
    out_price = Path(out_price_dir)
    out_split = Path(out_split_dir)
    out_price.mkdir(parents=True, exist_ok=True)
    out_split.mkdir(parents=True, exist_ok=True)

    try:
        frame = yf.Ticker(ticker).history(
            start=(start - timedelta(days=400)).isoformat(),
            end=(end + timedelta(days=400)).isoformat(),
            auto_adjust=False,
            actions=True,
        )
        prices, splits = _normalize_history(frame)
        prices.to_parquet(out_price / f"{ticker}.parquet", index=False)
        splits.to_parquet(out_split / f"{ticker}.parquet", index=False)
        return PriceFetchResult(ticker=ticker, status="ok", rows=len(prices))
    except Exception as exc:
        return PriceFetchResult(ticker=ticker, status="error", rows=0, error=str(exc))


def fetch_prices_for_universe(
    tickers: list[str],
    out_price_dir: str | Path,
    out_split_dir: str | Path,
    start: date,
    end: date,
    workers: int = 8,
) -> list[PriceFetchResult]:
    results: list[PriceFetchResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_price_history, t, out_price_dir, out_split_dir, start, end): t for t in tickers
        }
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda r: r.ticker)
