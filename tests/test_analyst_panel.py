from __future__ import annotations

import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from past_predictions.analyst_panel import compute_analyst_weekly_dataset
from past_predictions.prices import build_horizon_map, trading_days, weekly_asof_dates


def _price_frame(start: date, end: date, base: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    days = trading_days(start, end, calendar="XNYS")
    closes = [base + i * step for i in range(len(days))]
    return pd.DataFrame({"date": days, "close": closes})


def _write_inputs(
    tmp_path: Path,
    events: pd.DataFrame,
    prices: pd.DataFrame,
    splits: pd.DataFrame | None = None,
) -> dict[str, Path]:
    universe = pd.DataFrame(
        [{"ticker_norm": "AAA", "ticker_fmp": "AAA", "ticker_yahoo": "AAA"}]
    )
    provider = pd.DataFrame([{"ticker": "AAA", "provider": "yahoo"}])

    universe_path = tmp_path / "universe.csv"
    events_path = tmp_path / "events.parquet"
    provider_path = tmp_path / "provider.csv"

    price_dir = tmp_path / "prices"
    split_dir = tmp_path / "splits"
    price_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    universe.to_csv(universe_path, index=False)
    provider.to_csv(provider_path, index=False)
    events.to_parquet(events_path, index=False)

    prices.to_parquet(price_dir / "AAA.parquet", index=False)
    if splits is None:
        splits = pd.DataFrame(columns=["date", "split_ratio"])
    splits.to_parquet(split_dir / "AAA.parquet", index=False)

    return {
        "universe": universe_path,
        "events": events_path,
        "provider": provider_path,
        "price_dir": price_dir,
        "split_dir": split_dir,
    }


def _base_events(*, targets: list[tuple[str, str, float]]) -> pd.DataFrame:
    rows = []
    for analyst_key, ts, target in targets:
        dt = pd.to_datetime(ts)
        rows.append(
            {
                "ticker": "AAA",
                "event_ts": dt,
                "event_date": dt.date(),
                "analyst_key": analyst_key,
                "target_price": target,
                "event_source": "yahoo",
                "raw_path": "x",
            }
        )
    return pd.DataFrame(rows)


def test_analyst_row_grain_only_active(tmp_path: Path) -> None:
    events = _base_events(targets=[("firm a", "2024-01-02 10:00:00", 100.0)])
    prices = _price_frame(date(2024, 1, 1), date(2025, 12, 31), base=100, step=0.1)
    paths = _write_inputs(tmp_path, events, prices)

    out = compute_analyst_weekly_dataset(
        universe=pd.read_csv(paths["universe"]),
        events=pd.read_parquet(paths["events"]),
        provider_selection=pd.read_csv(paths["provider"]),
        price_dir=paths["price_dir"],
        split_dir=paths["split_dir"],
        pred_start=date(2024, 1, 1),
        pred_end=date(2024, 2, 15),
        actual_start=date(2024, 1, 1),
        actual_end=date(2025, 12, 31),
        ttl_days=365,
        calendar="XNYS",
        horizon_days=1,
    )

    assert not out.empty
    assert set(out["analyst_key"]) == {"firm a"}
    assert out["predicted"].notna().all()
    assert not out.duplicated(subset=["ticker", "date", "analyst_key"]).any()


def test_analyst_latest_per_analyst_ttl(tmp_path: Path) -> None:
    events = _base_events(
        targets=[
            ("firm a", "2024-01-02 10:00:00", 100.0),
            ("firm a", "2024-01-20 10:00:00", 130.0),
        ]
    )
    prices = _price_frame(date(2024, 1, 1), date(2025, 12, 31), base=90, step=0.2)
    paths = _write_inputs(tmp_path, events, prices)

    out = compute_analyst_weekly_dataset(
        universe=pd.read_csv(paths["universe"]),
        events=pd.read_parquet(paths["events"]),
        provider_selection=pd.read_csv(paths["provider"]),
        price_dir=paths["price_dir"],
        split_dir=paths["split_dir"],
        pred_start=date(2024, 1, 1),
        pred_end=date(2024, 2, 20),
        actual_start=date(2024, 1, 1),
        actual_end=date(2025, 12, 31),
        ttl_days=10,
        calendar="XNYS",
        horizon_days=1,
    )

    assert not out.empty
    jan26 = out[out["date"] == "2024-01-26"]
    assert not jan26.empty
    assert jan26.iloc[0]["predicted"] == pytest.approx(130.0)
    assert pd.to_datetime(out["date"]).dt.date.max() <= date(2024, 1, 26)


def test_analyst_window_enforcement(tmp_path: Path) -> None:
    pred_start = date(2024, 1, 1)
    pred_end = date(2024, 1, 31)
    actual_start = date(2024, 1, 10)
    actual_end = date(2024, 1, 17)

    events = _base_events(targets=[("firm a", "2024-01-02 10:00:00", 100.0)])
    prices = _price_frame(date(2024, 1, 1), date(2025, 12, 31), base=100, step=0.0)
    paths = _write_inputs(tmp_path, events, prices)

    out = compute_analyst_weekly_dataset(
        universe=pd.read_csv(paths["universe"]),
        events=pd.read_parquet(paths["events"]),
        provider_selection=pd.read_csv(paths["provider"]),
        price_dir=paths["price_dir"],
        split_dir=paths["split_dir"],
        pred_start=pred_start,
        pred_end=pred_end,
        actual_start=actual_start,
        actual_end=actual_end,
        ttl_days=365,
        calendar="XNYS",
        horizon_days=1,
    )

    weekly = weekly_asof_dates(pred_start, pred_end, calendar="XNYS")
    trading = trading_days(pred_start, max(actual_end, pred_end), calendar="XNYS")
    hmap = build_horizon_map(weekly, trading, horizon_days=1)
    allowed_asof = {d for d in weekly if hmap.get(d) is not None and actual_start <= hmap[d] <= actual_end}

    observed_asof = set(pd.to_datetime(out["date"]).dt.date.tolist())
    assert observed_asof <= allowed_asof
    assert observed_asof


def test_analyst_horizon_mapping_252(tmp_path: Path) -> None:
    pred_start = date(2023, 3, 1)
    pred_end = date(2023, 3, 31)
    actual_end = date(2025, 12, 31)

    events = _base_events(targets=[("firm a", "2023-01-03 10:00:00", 200.0)])
    prices = _price_frame(date(2022, 1, 1), date(2025, 12, 31), base=50.0, step=1.0)
    paths = _write_inputs(tmp_path, events, prices)

    out = compute_analyst_weekly_dataset(
        universe=pd.read_csv(paths["universe"]),
        events=pd.read_parquet(paths["events"]),
        provider_selection=pd.read_csv(paths["provider"]),
        price_dir=paths["price_dir"],
        split_dir=paths["split_dir"],
        pred_start=pred_start,
        pred_end=pred_end,
        actual_start=date(2023, 1, 1),
        actual_end=actual_end,
        ttl_days=365,
        calendar="XNYS",
        horizon_days=252,
    )

    assert not out.empty
    first = out.iloc[0]

    weekly = weekly_asof_dates(pred_start, pred_end, calendar="XNYS")
    trading = trading_days(pred_start, max(actual_end, pred_end + timedelta(days=500)), calendar="XNYS")
    hmap = build_horizon_map(weekly, trading, horizon_days=252)
    asof_date = pd.to_datetime(first["date"]).date()
    horizon_date = hmap[asof_date]

    price_lookup = dict(zip(pd.to_datetime(prices["date"]).dt.date, prices["close"]))
    assert first["actual_12m"] == pytest.approx(float(price_lookup[horizon_date]))


def test_analyst_split_adjustment_consistency(tmp_path: Path) -> None:
    events = _base_events(targets=[("firm a", "2024-05-02 10:00:00", 200.0)])
    prices = _price_frame(date(2024, 5, 1), date(2024, 12, 31), base=100.0, step=0.0)
    splits = pd.DataFrame([{"date": "2024-06-10", "split_ratio": 2.0}])
    paths = _write_inputs(tmp_path, events, prices, splits=splits)

    out = compute_analyst_weekly_dataset(
        universe=pd.read_csv(paths["universe"]),
        events=pd.read_parquet(paths["events"]),
        provider_selection=pd.read_csv(paths["provider"]),
        price_dir=paths["price_dir"],
        split_dir=paths["split_dir"],
        pred_start=date(2024, 5, 1),
        pred_end=date(2024, 5, 31),
        actual_start=date(2024, 1, 1),
        actual_end=date(2024, 12, 31),
        ttl_days=365,
        calendar="XNYS",
        horizon_days=1,
    )

    assert not out.empty
    row = out.iloc[0]
    assert row["predicted"] == pytest.approx(100.0)
    assert row["actual"] == pytest.approx(50.0)
    assert row["actual_12m"] == pytest.approx(50.0)
    assert "SPLIT_ADJUSTED" in row["data_quality_flags"]


def test_analyst_direction_hit_logic(tmp_path: Path) -> None:
    events = _base_events(
        targets=[
            ("bull firm", "2024-01-02 10:00:00", 120.0),
            ("bear firm", "2024-01-02 10:00:00", 80.0),
        ]
    )
    pred_start = date(2024, 1, 1)
    pred_end = date(2024, 1, 31)
    days = trading_days(pred_start, date(2024, 2, 1), calendar="XNYS")
    prices = pd.DataFrame({"date": days, "close": [100.0 for _ in days]})
    weekly = weekly_asof_dates(pred_start, pred_end, calendar="XNYS")
    hmap = build_horizon_map(weekly, days, horizon_days=1)
    first_horizon = hmap[weekly[0]]
    prices.loc[prices["date"] == first_horizon, "close"] = 110.0

    paths = _write_inputs(tmp_path, events, prices)

    out = compute_analyst_weekly_dataset(
        universe=pd.read_csv(paths["universe"]),
        events=pd.read_parquet(paths["events"]),
        provider_selection=pd.read_csv(paths["provider"]),
        price_dir=paths["price_dir"],
        split_dir=paths["split_dir"],
        pred_start=pred_start,
        pred_end=pred_end,
        actual_start=date(2024, 1, 1),
        actual_end=date(2025, 12, 31),
        ttl_days=365,
        calendar="XNYS",
        horizon_days=1,
    )

    first_date = sorted(out["date"].unique())[0]
    sample = out[out["date"] == first_date].set_index("analyst_key")
    assert bool(sample.loc["bull firm", "hit_direction"]) is True
    assert bool(sample.loc["bear firm", "hit_direction"]) is False


def test_analyst_cli_smoke(tmp_path: Path) -> None:
    events = _base_events(targets=[("firm a", "2024-01-02 10:00:00", 100.0)])
    prices = _price_frame(date(2024, 1, 1), date(2025, 12, 31), base=100.0, step=0.5)
    paths = _write_inputs(tmp_path, events, prices)

    out_parquet = tmp_path / "analyst_weekly.parquet"
    out_manifest = tmp_path / "analyst_manifest.json"
    out_csv = tmp_path / "analyst_weekly.csv"

    compute_cmd = [
        sys.executable,
        "-m",
        "past_predictions.cli",
        "compute-analyst-weekly",
        "--pred-start",
        "2024-01-01",
        "--pred-end",
        "2024-01-31",
        "--actual-start",
        "2024-01-01",
        "--actual-end",
        "2025-12-31",
        "--ttl-days",
        "365",
        "--calendar",
        "XNYS",
        "--horizon-days",
        "1",
        "--universe",
        str(paths["universe"]),
        "--events",
        str(paths["events"]),
        "--provider-selection",
        str(paths["provider"]),
        "--price-dir",
        str(paths["price_dir"]),
        "--split-dir",
        str(paths["split_dir"]),
        "--out",
        str(out_parquet),
        "--manifest-out",
        str(out_manifest),
    ]
    subprocess.run(compute_cmd, check=True)

    export_cmd = [
        sys.executable,
        "-m",
        "past_predictions.cli",
        "export-analyst-csv",
        "--source",
        str(out_parquet),
        "--out",
        str(out_csv),
    ]
    subprocess.run(export_cmd, check=True)

    assert out_parquet.exists()
    assert out_manifest.exists()
    assert out_csv.exists()

    exported = pd.read_csv(out_csv)
    sorted_copy = exported.sort_values(["ticker", "date", "analyst_key"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(exported.reset_index(drop=True), sorted_copy)
