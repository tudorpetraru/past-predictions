from datetime import date

import pandas as pd

from past_predictions.adjustments import SplitAdjuster


def test_split_adjustment_math() -> None:
    splits = pd.DataFrame(
        [
            {"date": "2024-06-10", "split_ratio": 4.0},
            {"date": "2025-01-15", "split_ratio": 2.0},
        ]
    )
    adjuster = SplitAdjuster.from_frame(splits, end_date=date(2025, 12, 31))

    factor_pre = adjuster.factor_to_end(date(2024, 1, 1))
    factor_mid = adjuster.factor_to_end(date(2024, 12, 1))
    factor_post = adjuster.factor_to_end(date(2025, 2, 1))

    assert factor_pre == 8.0
    assert factor_mid == 2.0
    assert factor_post == 1.0

    assert adjuster.adjust_value(800.0, date(2024, 1, 1)) == 100.0
