from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import requests

from .utils import normalize_analyst_key

PRICE_TARGET_ENDPOINT = "/api/v4/price-target"


@dataclass(frozen=True)
class FMPRequestConfig:
    base_url: str
    api_key: str
    timeout_seconds: int = 25
    max_retries: int = 5
    backoff_seconds: float = 1.0


def _params_hash(params: Dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _request_with_retries(url: str, params: Dict[str, Any], cfg: FMPRequestConfig) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(cfg.max_retries):
        try:
            resp = requests.get(url, params=params, timeout=cfg.timeout_seconds)
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                wait = cfg.backoff_seconds * (2 ** attempt)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_error = exc
            wait = cfg.backoff_seconds * (2 ** attempt)
            time.sleep(wait)
    if last_error:
        raise last_error
    raise RuntimeError("FMP request failed without a concrete exception")


def _extract_value(payload: Dict[str, Any], candidates: Iterable[str]) -> Any:
    for key in candidates:
        if key in payload:
            return payload[key]
    return None


def parse_fmp_events(payload: Any, ticker: str, raw_path: Path) -> pd.DataFrame:
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame(
            columns=[
                "ticker",
                "event_ts",
                "event_date",
                "analyst_key",
                "target_price",
                "event_source",
                "raw_path",
            ]
        )

    rows = []
    for row in payload:
        if not isinstance(row, dict):
            continue

        date_raw = _extract_value(row, ["publishedDate", "date", "updatedDate", "newsURLDate"])
        analyst_raw = _extract_value(row, ["analystName", "analyst", "analystCompany", "publisher", "newsPublisher"])
        target_raw = _extract_value(row, ["priceTarget", "targetPrice", "target", "price_target"])

        ts = pd.to_datetime(date_raw, errors="coerce", utc=True)
        target = pd.to_numeric(target_raw, errors="coerce")
        analyst_key = normalize_analyst_key(analyst_raw)

        if pd.isna(ts) or pd.isna(target) or float(target) <= 0 or analyst_key == "":
            continue

        ts = ts.tz_convert(None)
        rows.append(
            {
                "ticker": ticker,
                "event_ts": ts,
                "event_date": ts.date(),
                "analyst_key": analyst_key,
                "target_price": float(target),
                "event_source": "fmp",
                "raw_path": str(raw_path),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "event_ts",
                "event_date",
                "analyst_key",
                "target_price",
                "event_source",
                "raw_path",
            ]
        )
    return frame.sort_values("event_ts").reset_index(drop=True)


def fetch_fmp_price_targets(
    ticker_fmp: str,
    ticker_norm: str,
    raw_dir: str | Path,
    cfg: FMPRequestConfig,
    force_refresh: bool = False,
) -> pd.DataFrame:
    out_dir = Path(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = {"symbol": ticker_fmp, "apikey": cfg.api_key}
    phash = _params_hash(params)

    raw_path = out_dir / f"{ticker_fmp}.json"
    meta_path = out_dir / f"{ticker_fmp}.meta.json"

    payload: Any
    if raw_path.exists() and meta_path.exists() and not force_refresh:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        url = f"{cfg.base_url.rstrip('/')}{PRICE_TARGET_ENDPOINT}"
        response = _request_with_retries(url, params=params, cfg=cfg)
        payload = response.json()

        raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        metadata = {
            "ticker": ticker_norm,
            "ticker_fmp": ticker_fmp,
            "endpoint": PRICE_TARGET_ENDPOINT,
            "params_hash": phash,
            "fetched_at_utc": datetime.utcnow().isoformat() + "Z",
            "http_status": response.status_code,
            "count": len(payload) if isinstance(payload, list) else 0,
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return parse_fmp_events(payload=payload, ticker=ticker_norm, raw_path=raw_path)
