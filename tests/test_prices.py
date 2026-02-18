from datetime import date

from past_predictions.prices import build_horizon_map, weekly_asof_dates


def test_weekly_asof_holiday_week() -> None:
    asof = weekly_asof_dates(start=date(2025, 4, 14), end=date(2025, 4, 18), calendar="XNYS")
    assert asof[-1].isoformat() == "2025-04-17"


def test_horizon_lookup_252_like() -> None:
    trading = [date(2025, 1, d) for d in range(1, 11)]
    mapping = build_horizon_map([trading[0], trading[5]], trading_day_list=trading, horizon_days=3)
    assert mapping[trading[0]] == trading[3]
    assert mapping[trading[5]] == trading[8]
