from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class RunWindow:
    start_date: date
    end_date: date
    calendar: str = "XNYS"


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_api_key(value: str | None) -> str | None:
    if not value:
        return None
    if value.startswith("ENV:"):
        return os.getenv(value.split(":", 1)[1])
    return value


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def config_hash(payload: Dict[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()
