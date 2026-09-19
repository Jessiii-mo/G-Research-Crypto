import unittest

import numpy as np
import pandas as pd

from run_experiment import NUMERIC_FEATURES, make_features, time_split


def synthetic_frame(periods: int = 100) -> pd.DataFrame:
    rows = []
    for asset in (0, 1):
        for index in range(periods):
            close = 100 + asset * 20 + index * 0.1
            rows.append({
                "timestamp": 1_600_000_000 + index * 60,
                "Asset_ID": asset,
                "Count": 20 + index,
                "Open": close - 0.05,
                "High": close + 0.2,
                "Low": close - 0.2,
                "Close": close,
                "Volume": 1_000 + index * 3,
                "VWAP": close - 0.01,
                "Target": np.sin(index / 10) / 100,
            })
    return pd.DataFrame(rows)


class PipelineTests(unittest.TestCase):
    def test_future_rows_do_not_change_past_features(self):
        full = synthetic_frame(100)
        past = full.loc[full["timestamp"] < 1_600_000_000 + 80 * 60]
        past_features = make_features(past).set_index(["timestamp", "Asset_ID"])
        full_features = make_features(full).set_index(["timestamp", "Asset_ID"])
        shared = past_features.index
        pd.testing.assert_frame_equal(
            past_features.loc[shared, NUMERIC_FEATURES],
            full_features.loc[shared, NUMERIC_FEATURES],
        )

    def test_time_splits_are_ordered_and_disjoint(self):
        train, validation, test = time_split(make_features(synthetic_frame()))
        self.assertLess(train["timestamp"].max(), validation["timestamp"].min())
        self.assertLess(validation["timestamp"].max(), test["timestamp"].min())
        self.assertTrue(set(train.index).isdisjoint(validation.index))
        self.assertTrue(set(validation.index).isdisjoint(test.index))


if __name__ == "__main__":
    unittest.main()
