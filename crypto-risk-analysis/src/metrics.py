from __future__ import annotations

import numpy as np
import pandas as pd

from .config import REGIME_ORDER
from .utils import annualized_volatility, compute_drawdown, max_drawdown, mean_absolute_return

ASSET_CONFIG = {
    "BTC": {"return_col": "btc_return", "price_col": "btc_close"},
    "ETH": {"return_col": "eth_return", "price_col": "eth_close"},
}

METRIC_ROWS = [
    ("Mean Absolute Return", "Mean Absolute Return"),
    ("Volatility (Annualized)", "Volatility"),
    ("Max Drawdown", "Max Drawdown"),
]


def _iter_regime_subsets(merged: pd.DataFrame):
    yield "Overall", merged
    for regime in REGIME_ORDER:
        yield regime, merged[merged["regime"] == regime]


def _asset_metrics(df: pd.DataFrame, return_col: str, price_col: str) -> dict[str, float]:
    returns = df[return_col].dropna()
    prices = df[price_col].dropna()
    # Với từng regime, MDD phải tính lại trên chính chuỗi giá của subset đó.
    drawdown = compute_drawdown(prices) if not prices.empty else pd.Series(dtype="float64")
    return {
        "Mean Absolute Return": mean_absolute_return(returns),
        "Volatility": annualized_volatility(returns),
        "Max Drawdown": max_drawdown(drawdown),
    }


def compute_regime_summary(merged: pd.DataFrame) -> dict[str, dict[str, dict[str, float] | float]]:
    summary: dict[str, dict[str, dict[str, float] | float]] = {}

    for regime, subset in _iter_regime_subsets(merged):
        regime_summary: dict[str, dict[str, float] | float] = {}
        for asset, config in ASSET_CONFIG.items():
            regime_summary[asset] = _asset_metrics(subset, config["return_col"], config["price_col"])

        # Bảng correlation dùng một giá trị Pearson correlation cho cả subset regime, không dùng rolling corr.
        valid = subset[["btc_return", "eth_return"]].dropna()
        regime_summary["Correlation"] = float(valid.corr().iloc[0, 1]) if len(valid) >= 2 else np.nan
        summary[regime] = regime_summary

    return summary


def compute_metrics_table(merged: pd.DataFrame) -> pd.DataFrame:
    summary = compute_regime_summary(merged)
    columns = [
        "Metric",
        "BTC Overall",
        "ETH Overall",
        *[f"BTC {regime}" for regime in REGIME_ORDER],
        *[f"ETH {regime}" for regime in REGIME_ORDER],
    ]

    records = []
    for label, field in METRIC_ROWS:
        row = {"Metric": label}
        row["BTC Overall"] = summary["Overall"]["BTC"][field]
        row["ETH Overall"] = summary["Overall"]["ETH"][field]
        for regime in REGIME_ORDER:
            row[f"BTC {regime}"] = summary[regime]["BTC"][field]
            row[f"ETH {regime}"] = summary[regime]["ETH"][field]
        records.append(row)

    return pd.DataFrame(records, columns=columns)


def compute_correlation_table(merged: pd.DataFrame) -> pd.DataFrame:
    summary = compute_regime_summary(merged)
    row = {"Metric": "BTC-ETH Correlation", "Overall": summary["Overall"]["Correlation"]}
    for regime in REGIME_ORDER:
        row[regime] = summary[regime]["Correlation"]
    return pd.DataFrame([row], columns=["Metric", "Overall", *REGIME_ORDER])


def table_to_console(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    out = df.copy()

    if "Metric" in out.columns:
        # MAR, Volatility và MDD được in theo %, còn correlation giữ dạng số thập phân.
        percent_metrics = {"Mean Absolute Return", "Volatility (Annualized)", "Max Drawdown"}
        metric_series = out["Metric"]
        for col in out.select_dtypes(include=["float", "float64", "float32"]).columns:
            formatted = []
            for idx, value in out[col].items():
                if pd.isna(value):
                    formatted.append("NaN")
                    continue
                metric_name = metric_series.loc[idx]
                if metric_name in percent_metrics:
                    formatted.append(f"{value * 100:.2f}%")
                else:
                    formatted.append(float_fmt.format(value))
            out[col] = formatted
    else:
        for col in out.select_dtypes(include=["float", "float64", "float32"]).columns:
            out[col] = out[col].map(lambda x: "NaN" if pd.isna(x) else float_fmt.format(x))

    return out.to_string(index=False)
