"""Leakage-aware benchmark for Kaggle G-Research Crypto Forecasting data."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import TransformedTargetRegressor


RAW_COLUMNS = [
    "timestamp", "Asset_ID", "Count", "Open", "High", "Low", "Close",
    "Volume", "VWAP", "Target",
]
NUMERIC_FEATURES = [
    "return_1", "log_return_1", "close_open_return", "high_low_range",
    "vwap_close_gap", "volume_change_1", "count_change_1",
    "return_mean_5", "return_std_5", "volume_ratio_5", "momentum_5",
    "return_mean_15", "return_std_15", "volume_ratio_15", "momentum_15",
    "return_mean_60", "return_std_60", "volume_ratio_60", "momentum_60",
]


def load_window(path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """Read only a UTC date window from the large CSV using bounded memory."""
    start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp())
    end_ts = int((pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    dtypes = {
        "timestamp": "int64", "Asset_ID": "int8", "Count": "float32",
        "Open": "float32", "High": "float32", "Low": "float32",
        "Close": "float32", "Volume": "float32", "VWAP": "float32",
        "Target": "float32",
    }
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=RAW_COLUMNS, dtype=dtypes, chunksize=1_000_000):
        selected = chunk.loc[(chunk["timestamp"] >= start_ts) & (chunk["timestamp"] < end_ts)]
        if not selected.empty:
            pieces.append(selected)
    if not pieces:
        raise ValueError(f"No rows found between {start_date} and {end_date}.")
    return pd.concat(pieces, ignore_index=True)


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.drop_duplicates(["timestamp", "Asset_ID"], keep="last").copy()
    valid_price = (data[["Open", "High", "Low", "Close", "VWAP"]] > 0).all(axis=1)
    data = data.loc[valid_price & (data["Volume"] >= 0) & data["Target"].notna()]
    return data.sort_values(["Asset_ID", "timestamp"]).reset_index(drop=True)


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features from information available at or before each row."""
    data = frame.sort_values(["Asset_ID", "timestamp"]).copy()
    group = data.groupby("Asset_ID", sort=False)
    data["return_1"] = group["Close"].pct_change(fill_method=None)
    data["log_return_1"] = group["Close"].transform(lambda s: np.log(s).diff())
    data["close_open_return"] = data["Close"] / data["Open"] - 1
    data["high_low_range"] = (data["High"] - data["Low"]) / data["Close"]
    data["vwap_close_gap"] = data["VWAP"] / data["Close"] - 1
    data["volume_change_1"] = group["Volume"].pct_change(fill_method=None)
    data["count_change_1"] = group["Count"].pct_change(fill_method=None)

    for window in (5, 15, 60):
        by_asset = data.groupby("Asset_ID", sort=False)
        data[f"return_mean_{window}"] = by_asset["return_1"].transform(
            lambda s: s.rolling(window, min_periods=window).mean()
        )
        data[f"return_std_{window}"] = by_asset["return_1"].transform(
            lambda s: s.rolling(window, min_periods=window).std()
        )
        volume_mean = by_asset["Volume"].transform(
            lambda s: s.rolling(window, min_periods=window).mean()
        )
        data[f"volume_ratio_{window}"] = data["Volume"] / volume_mean
        data[f"momentum_{window}"] = by_asset["Close"].pct_change(
            periods=window, fill_method=None
        )

    data[NUMERIC_FEATURES] = data[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)
    return data.sort_values(["timestamp", "Asset_ID"]).reset_index(drop=True)


def sample_timestamps(frame: pd.DataFrame, every: int) -> pd.DataFrame:
    timestamps = np.sort(frame["timestamp"].unique())
    selected = set(timestamps[::every].tolist())
    return frame.loc[frame["timestamp"].isin(selected)].reset_index(drop=True)


def time_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timestamps = np.sort(frame["timestamp"].unique())
    train_end = timestamps[int(len(timestamps) * 0.70)]
    val_end = timestamps[int(len(timestamps) * 0.85)]
    train = frame.loc[frame["timestamp"] < train_end].copy()
    validation = frame.loc[(frame["timestamp"] >= train_end) & (frame["timestamp"] < val_end)].copy()
    test = frame.loc[frame["timestamp"] >= val_end].copy()
    return train, validation, test


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("asset", categorical, ["Asset_ID"]),
    ])


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if np.std(y_pred) > 0 and np.std(y_true) > 0:
        correlation = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        correlation = float("nan")
    direction = (
        float(np.mean(np.sign(y_true) == np.sign(y_pred)))
        if np.any(y_pred != 0)
        else float("nan")
    )
    if len(np.unique(y_pred)) > 1:
        low, high = np.quantile(y_pred, [0.1, 0.9])
        spread = float(y_true[y_pred >= high].mean() - y_true[y_pred <= low].mean())
    else:
        spread = float("nan")
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson_ic": correlation,
        "direction_accuracy": direction,
        "decile_target_spread": spread,
    }


def fit_and_evaluate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    knn_max_train: int,
    transaction_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_columns = NUMERIC_FEATURES + ["Asset_ID"]
    preprocessor = build_preprocessor()
    x_train = preprocessor.fit_transform(train[feature_columns]).astype("float32")
    x_validation = preprocessor.transform(validation[feature_columns]).astype("float32")
    x_test = preprocessor.transform(test[feature_columns]).astype("float32")
    y_train = train["Target"].to_numpy(dtype="float32")
    y_validation = validation["Target"].to_numpy(dtype="float32")
    y_test = test["Target"].to_numpy(dtype="float32")

    x_fit = np.concatenate([x_train, x_validation])
    y_fit = np.concatenate([y_train, y_validation])
    models = {
        "Linear Regression": LinearRegression(n_jobs=-1),
        "Ridge": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=50, random_state=seed),
        "KNN": KNeighborsRegressor(n_neighbors=20, weights="distance", n_jobs=-1),
        "Neural Network": TransformedTargetRegressor(
            regressor=MLPRegressor(
                hidden_layer_sizes=(64, 32), learning_rate_init=0.001,
                max_iter=10, batch_size=512, random_state=seed,
            ),
            transformer=StandardScaler(),
        ),
    }
    results: list[dict[str, float | str | int]] = []
    per_asset: list[dict[str, float | str | int]] = []
    strategy_summaries: list[dict[str, float | str | int]] = []
    strategy_series: list[pd.DataFrame] = []

    predictions: dict[str, np.ndarray] = {"Zero Baseline": np.zeros_like(y_test)}
    runtimes: dict[str, float] = {"Zero Baseline": 0.0}
    train_sizes: dict[str, int] = {"Zero Baseline": len(x_fit)}
    for name, model in models.items():
        model_x, model_y = x_fit, y_fit
        if name == "KNN" and len(model_x) > knn_max_train:
            indices = np.linspace(0, len(model_x) - 1, knn_max_train, dtype=int)
            model_x, model_y = model_x[indices], model_y[indices]
        started = time.perf_counter()
        model.fit(model_x, model_y)
        predictions[name] = model.predict(x_test).astype("float32")
        runtimes[name] = time.perf_counter() - started
        train_sizes[name] = len(model_x)

    for name, predicted in predictions.items():
        row: dict[str, float | str | int] = {
            "model": name,
            "train_rows": train_sizes[name],
            "test_rows": len(y_test),
            "runtime_seconds": runtimes[name],
        }
        row.update(score_predictions(y_test, predicted))
        results.append(row)
        for asset_id in sorted(test["Asset_ID"].unique()):
            mask = test["Asset_ID"].to_numpy() == asset_id
            asset_row: dict[str, float | str | int] = {
                "model": name, "Asset_ID": int(asset_id), "rows": int(mask.sum())
            }
            asset_row.update(score_predictions(y_test[mask], predicted[mask]))
            per_asset.append(asset_row)

        if name != "Zero Baseline":
            curve, summary = evaluate_long_short_strategy(
                test, predicted, name, transaction_cost_bps
            )
            strategy_series.append(curve)
            strategy_summaries.append(summary)
    return (
        pd.DataFrame(results),
        pd.DataFrame(per_asset),
        pd.DataFrame(strategy_summaries),
        pd.concat(strategy_series, ignore_index=True),
    )


def evaluate_long_short_strategy(
    test: pd.DataFrame,
    predictions: np.ndarray,
    model_name: str,
    transaction_cost_bps: float,
) -> tuple[pd.DataFrame, dict[str, float | str | int]]:
    """Evaluate a non-overlapping, cross-sectional long-short signal prototype."""
    strategy = test[["timestamp", "Asset_ID", "forward_return_20m"]].copy()
    strategy["prediction"] = predictions
    rebalance_times = np.sort(strategy["timestamp"].unique())[::2]
    strategy = strategy.loc[
        strategy["timestamp"].isin(rebalance_times) & strategy["forward_return_20m"].notna()
    ]

    rows: list[dict[str, float | int | str]] = []
    previous_weights: dict[int, float] = {}
    cost_rate = transaction_cost_bps / 10_000
    for timestamp, cross_section in strategy.groupby("timestamp", sort=True):
        ranked = cross_section.sort_values("prediction")
        side_size = max(1, int(np.ceil(len(ranked) * 0.20)))
        short = ranked.head(side_size)
        long = ranked.tail(side_size)
        weights = {
            **{int(asset): -0.5 / side_size for asset in short["Asset_ID"]},
            **{int(asset): 0.5 / side_size for asset in long["Asset_ID"]},
        }
        gross_return = 0.5 * (
            float(long["forward_return_20m"].mean())
            - float(short["forward_return_20m"].mean())
        )
        assets = set(previous_weights) | set(weights)
        turnover = sum(abs(weights.get(asset, 0.0) - previous_weights.get(asset, 0.0)) for asset in assets)
        net_return = gross_return - turnover * cost_rate
        rows.append({
            "model": model_name,
            "timestamp": int(timestamp),
            "gross_return": gross_return,
            "turnover": turnover,
            "net_return": net_return,
        })
        previous_weights = weights

    curve = pd.DataFrame(rows)
    curve["gross_cumulative_return"] = (1 + curve["gross_return"]).cumprod() - 1
    curve["net_cumulative_return"] = (1 + curve["net_return"]).cumprod() - 1
    running_peak = (1 + curve["net_cumulative_return"]).cummax()
    drawdown = (1 + curve["net_cumulative_return"]) / running_peak - 1
    summary: dict[str, float | str | int] = {
        "model": model_name,
        "rebalances": len(curve),
        "transaction_cost_bps": transaction_cost_bps,
        "mean_turnover": float(curve["turnover"].mean()),
        "gross_total_return": float(curve["gross_cumulative_return"].iloc[-1]),
        "net_total_return": float(curve["net_cumulative_return"].iloc[-1]),
        "net_hit_rate": float((curve["net_return"] > 0).mean()),
        "net_max_drawdown": float(drawdown.min()),
    }
    return curve, summary


def save_chart(metrics: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(metrics["model"], metrics["rmse"], color="#4059AD")
    axes[0].set_title("Test RMSE (lower is better)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(metrics["model"], metrics["pearson_ic"].fillna(0), color="#2A9D8F")
    axes[1].set_title("Test Pearson IC")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_strategy_chart(strategy: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for model_name, model_curve in strategy.groupby("model", sort=False):
        datetime = pd.to_datetime(model_curve["timestamp"], unit="s", utc=True)
        axes[0].plot(datetime, model_curve["gross_cumulative_return"], label=model_name)
        axes[1].plot(datetime, model_curve["net_cumulative_return"], label=model_name)
    axes[0].set_title("Gross cumulative return")
    axes[1].set_title("Net cumulative return after 10 bps turnover cost")
    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("Cumulative return")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--start-date", default="2021-03-25")
    parser.add_argument("--end-date", default="2021-09-21")
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--knn-max-train", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_window(args.data_path, args.start_date, args.end_date)
    featured = make_features(clean_data(raw))
    sampled = sample_timestamps(featured, args.sample_every)
    sampled["forward_return_20m"] = sampled.groupby("Asset_ID")["Close"].transform(
        lambda series: series.shift(-2) / series - 1
    )
    train, validation, test = time_split(sampled)
    metrics, per_asset, strategy_metrics, strategy_series = fit_and_evaluate(
        train, validation, test, seed=args.seed, knn_max_train=args.knn_max_train,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    metrics.insert(1, "source_window_rows", len(raw))
    metrics.insert(2, "modeled_rows", len(sampled))
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    per_asset.to_csv(args.output_dir / "per_asset_metrics.csv", index=False)
    strategy_metrics.to_csv(args.output_dir / "strategy_metrics.csv", index=False)
    strategy_series.to_csv(args.output_dir / "strategy_timeseries.csv", index=False)
    save_chart(metrics, args.output_dir / "model_comparison.png")
    save_strategy_chart(strategy_series, args.output_dir / "strategy_cumulative_return.png")
    print(metrics.to_string(index=False))
    print("\nLong-short strategy prototype:\n")
    print(strategy_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
