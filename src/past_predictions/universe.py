from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from .utils import normalize_ticker, ticker_to_fmp, ticker_to_yahoo

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"


@dataclass(frozen=True)
class UniverseEntry:
    ticker_raw: str
    ticker_norm: str
    ticker_yahoo: str
    ticker_fmp: str
    source_index: str
    as_of_date: str


def _fetch_html(url: str) -> str:
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (PastPredictionsBot/1.0)"},
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def _parse_sp500(html: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html))
    for table in tables:
        cols = {str(c).lower(): c for c in table.columns}
        if "symbol" in cols and "security" in cols:
            output = table[[cols["symbol"]]].copy()
            output.columns = ["ticker_raw"]
            return output
    raise ValueError("Unable to locate S&P 500 constituents table")


def _parse_nasdaq100(html: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html))
    for table in tables:
        cols = [str(c) for c in table.columns]
        lowered = {c.lower(): c for c in cols}
        if "ticker" in lowered and "company" in lowered and 80 <= len(table) <= 140:
            output = table[[lowered["ticker"]]].copy()
            output.columns = ["ticker_raw"]
            return output
    raise ValueError("Unable to locate Nasdaq-100 constituents table")


def _normalize_frame(frame: pd.DataFrame, source_index: str, as_of_date: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["ticker_raw"] = frame["ticker_raw"].astype(str).str.strip()
    frame["ticker_norm"] = frame["ticker_raw"].map(normalize_ticker)
    frame["ticker_yahoo"] = frame["ticker_norm"].map(ticker_to_yahoo)
    frame["ticker_fmp"] = frame["ticker_norm"].map(ticker_to_fmp)
    frame["source_index"] = source_index
    frame["as_of_date"] = as_of_date
    return frame


def build_universe(out_dir: str | Path) -> pd.DataFrame:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    as_of_date = datetime.now(timezone.utc).date().isoformat()

    sp500_html = _fetch_html(SP500_URL)
    nasdaq_html = _fetch_html(NASDAQ100_URL)

    sp500 = _normalize_frame(_parse_sp500(sp500_html), "sp500", as_of_date)
    nasdaq100 = _normalize_frame(_parse_nasdaq100(nasdaq_html), "nasdaq100", as_of_date)

    sp500.to_csv(out_path / "sp500.csv", index=False)
    nasdaq100.to_csv(out_path / "nasdaq100.csv", index=False)

    union = (
        pd.concat([sp500, nasdaq100], ignore_index=True)
        .sort_values(["ticker_norm", "source_index"])
        .drop_duplicates(subset=["ticker_norm"], keep="first")
        .reset_index(drop=True)
    )
    union.to_csv(out_path / "universe_union.csv", index=False)

    metadata = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [SP500_URL, NASDAQ100_URL],
        "counts": {
            "sp500": int(sp500["ticker_norm"].nunique()),
            "nasdaq100": int(nasdaq100["ticker_norm"].nunique()),
            "union": int(union["ticker_norm"].nunique()),
        },
        "schema": list(asdict(UniverseEntry("", "", "", "", "", "")).keys()),
    }
    (out_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return union
