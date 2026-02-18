from __future__ import annotations

import re
from datetime import date, datetime


def normalize_ticker(value: str) -> str:
    token = value.strip().upper().replace(" ", "")
    return token.replace("-", ".")


def ticker_to_fmp(value: str) -> str:
    return normalize_ticker(value).replace(".", "-")


def ticker_to_yahoo(value: str) -> str:
    return normalize_ticker(value)


def normalize_analyst_key(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def to_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return ts.date()
    except ValueError:
        return None
