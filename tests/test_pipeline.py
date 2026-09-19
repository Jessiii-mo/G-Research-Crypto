import unittest

import numpy as np
import pandas as pd

from run_experiment import NUMERIC_FEATURES, evaluate_long_short_strategy, make_features, time_split


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

    def test_strategy_charges_turnover_cost(self):
        data = make_features(synthetic_frame(20)).sort_values(["timestamp", "Asset_ID"])
        data["forward_return_20m"] = data.groupby("Asset_ID")["Close"].transform(
            lambda series: series.shift(-2) / series - 1
        )
        predictions = np.tile([-1.0, 1.0], len(data) // 2)
        curve, summary = evaluate_long_short_strategy(data, predictions, "test", 10.0)
        self.assertTrue((curve["net_return"] <= curve["gross_return"]).all())
        self.assertEqual(summary["transaction_cost_bps"], 10.0)


if __name__ == "__main__":
    unittest.main()
