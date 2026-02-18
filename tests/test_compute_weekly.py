from datetime import date

import pandas as pd

from past_predictions.consensus import compute_weekly_dataset


def test_compute_weekly_basic(tmp_path) -> None:
    universe = pd.DataFrame(
        [{"ticker_norm": "AAA", "ticker_fmp": "AAA", "ticker_yahoo": "AAA"}]
    )
    events = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "event_ts": "2025-01-02T10:00:00",
                "event_date": "2025-01-02",
                "analyst_key": "firm a",
                "target_price": 100.0,
                "event_source": "yahoo",
                "raw_path": "x",
            }
        ]
    )
    provider = pd.DataFrame([{"ticker": "AAA", "provider": "yahoo"}])

    price_dir = tmp_path / "prices"
    split_dir = tmp_path / "splits"
    price_dir.mkdir()
    split_dir.mkdir()

    prices = pd.DataFrame(
        [
            {"date": "2025-01-03", "close": 95.0},
            {"date": "2025-01-10", "close": 98.0},
        ]
    )
    splits = pd.DataFrame(columns=["date", "split_ratio"])
    prices.to_parquet(price_dir / "AAA.parquet", index=False)
    splits.to_parquet(split_dir / "AAA.parquet", index=False)

    out = compute_weekly_dataset(
        universe=universe,
        events=events,
        provider_selection=provider,
        price_dir=price_dir,
        split_dir=split_dir,
        start=date(2025, 1, 1),
        end=date(2025, 1, 10),
        ttl_days=365,
        min_analysts=1,
        calendar="XNYS",
        horizon_days=1,
    )

    assert not out.empty
    assert "SOURCE_YAHOO" in out.iloc[0]["data_quality_flags"]
