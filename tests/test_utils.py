from past_predictions.utils import normalize_ticker, ticker_to_fmp, ticker_to_yahoo


def test_ticker_normalization_share_class() -> None:
    assert normalize_ticker("brk-b") == "BRK.B"
    assert normalize_ticker("BF.B") == "BF.B"
    assert ticker_to_fmp("BF.B") == "BF-B"
    assert ticker_to_yahoo("BRK-B") == "BRK.B"
