from past_predictions.yahoo_events import choose_provider


def test_source_priority_switch() -> None:
    assert choose_provider(5, 0) == "yahoo"
    assert choose_provider(0, 3) == "fmp"
    assert choose_provider(0, 0) == "none"
