from datetime import datetime, date

import pandas as pd

from past_predictions.consensus import aggregate_targets, select_active_targets


def test_ttl_and_latest_per_analyst() -> None:
    events = pd.DataFrame(
        [
            {"analyst_key": "a", "event_ts": datetime(2025, 1, 1), "event_date": date(2025, 1, 1), "target_price_adj": 100},
            {"analyst_key": "a", "event_ts": datetime(2025, 3, 1), "event_date": date(2025, 3, 1), "target_price_adj": 120},
            {"analyst_key": "b", "event_ts": datetime(2024, 1, 1), "event_date": date(2024, 1, 1), "target_price_adj": 80},
        ]
    )
    active = select_active_targets(events, asof_date=date(2025, 3, 15), ttl_days=365)
    assert len(active) == 1
    assert active.iloc[0]["analyst_key"] == "a"
    assert active.iloc[0]["target_price_adj"] == 120


def test_aggregation_correctness() -> None:
    active = pd.DataFrame(
        [
            {"target_price_adj": 100.0},
            {"target_price_adj": 120.0},
            {"target_price_adj": 140.0},
        ]
    )
    mn, avg, mx, n = aggregate_targets(active)
    assert mn == 100.0
    assert avg == 120.0
    assert mx == 140.0
    assert n == 3
