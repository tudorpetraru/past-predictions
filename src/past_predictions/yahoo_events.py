from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from .utils import normalize_analyst_key

EVENT_COLUMNS = [
    "ticker",
    "event_ts",
    "event_date",
    "analyst_key",
    "target_price",
    "event_source",
    "raw_path",
]


def empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def choose_provider(yahoo_valid_events: int, fmp_valid_events: int) -> str:
    if yahoo_valid_events > 0:
        return "yahoo"
    if fmp_valid_events > 0:
        return "fmp"
    return "none"


def _coerce_raw(raw: pd.DataFrame, ticker: str, raw_path: Path) -> pd.DataFrame:
    if raw is None or raw.empty:
        return empty_events()

    frame = raw.copy()
    if frame.index.name:
        frame = frame.reset_index().rename(columns={frame.index.name: "event_ts"})
    elif "event_ts" not in frame.columns:
        frame = frame.reset_index().rename(columns={"index": "event_ts"})

    if "event_ts" not in frame.columns:
        return empty_events()

    if "Firm" not in frame.columns or "currentPriceTarget" not in frame.columns:
        return empty_events()

    events = pd.DataFrame()
    events["ticker"] = ticker
    events["event_ts"] = pd.to_datetime(frame["event_ts"], errors="coerce", utc=True).dt.tz_convert(None)
    events["event_date"] = events["event_ts"].dt.date
    events["analyst_key"] = frame["Firm"].map(normalize_analyst_key)
    events["target_price"] = pd.to_numeric(frame["currentPriceTarget"], errors="coerce")
    events["event_source"] = "yahoo"
    events["raw_path"] = str(raw_path)

    events = events.dropna(subset=["event_ts", "event_date", "target_price"]).copy()
    events = events[(events["target_price"] > 0) & (events["analyst_key"] != "")]
    events = events.sort_values("event_ts").reset_index(drop=True)
    return events[EVENT_COLUMNS]


def fetch_yahoo_events(
    ticker: str,
    raw_dir: str | Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    out_dir = Path(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{ticker}.parquet"

    if raw_path.exists() and not force_refresh:
        try:
            cached = pd.read_parquet(raw_path)
            missing = [c for c in EVENT_COLUMNS if c not in cached.columns]
            if not missing:
                return cached[EVENT_COLUMNS]
        except Exception:
            pass

    raw = yf.Ticker(ticker).get_upgrades_downgrades()
    events = _coerce_raw(raw, ticker=ticker, raw_path=raw_path)
    events.to_parquet(raw_path, index=False)
    return events
