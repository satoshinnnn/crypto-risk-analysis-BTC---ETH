from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd
from dateutil.relativedelta import relativedelta

from src.config import DATA_DIR, OUTPUT_DIR, PLOTS_DIR, TABLES_DIR, REGIME_ORDER
from src.data_fetcher import BinanceDataFetcher
from src.metrics import compute_correlation_table, compute_metrics_table, table_to_console
from src.plots import create_dashboard, plot_correlation, plot_drawdown, plot_price_volume, plot_volatility
from src.processor import build_processed_asset_views, preprocess_data
from src.regimes import classify_regimes


def _prepare_output_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _parse_utc_day(date_str: str) -> datetime:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)


def _latest_closed_utc_day() -> datetime:
    # Nến daily của Binance đóng theo mốc ngày UTC.
    now = datetime.now(timezone.utc)
    current_utc_day = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    return current_utc_day - relativedelta(days=1)


def _compute_times(lookback_years: int | None, start_str: str | None = None, end_str: str | None = None):
    if start_str or end_str:
        if not start_str or not end_str:
            raise ValueError("You must provide both --start and --end together.")

        start_day = _parse_utc_day(start_str)
        end_day_inclusive = _parse_utc_day(end_str)
        latest_closed_day = _latest_closed_utc_day()

        # Nếu user nhập end quá mới, ép về ngày UTC gần nhất đã đóng để tránh lấy nến chưa hoàn tất.
        effective_end_day = min(end_day_inclusive, latest_closed_day)
        end_day = effective_end_day + relativedelta(days=1)

        if start_day >= end_day:
            raise ValueError(
                f"--start must be earlier than the effective --end. Latest closed UTC day is "
                f"{latest_closed_day.strftime('%Y-%m-%d')}."
            )

        return (
            int(start_day.timestamp() * 1000),
            int(end_day.timestamp() * 1000),
            effective_end_day,
            effective_end_day != end_day_inclusive,
        )

    now = datetime.now(timezone.utc)
    end_day = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    start_day = end_day - relativedelta(years=lookback_years)
    return (
        int(start_day.timestamp() * 1000),
        int(end_day.timestamp() * 1000),
        end_day - relativedelta(days=1),
        False,
    )


def _save_tables(metrics_df: pd.DataFrame, corr_df: pd.DataFrame):
    metrics_path = TABLES_DIR / "metrics_table_1.csv"
    corr_path = TABLES_DIR / "correlation_table_2.csv"
    metrics_pretty_path = TABLES_DIR / "metrics_table_1_pretty.txt"
    corr_pretty_path = TABLES_DIR / "correlation_table_2_pretty.txt"

    metrics_df.to_csv(metrics_path, index=False)
    corr_df.to_csv(corr_path, index=False)

    # Lưu thêm bản text để mở nhanh trong editor mà không cần nhìn CSV thô.
    metrics_pretty_path.write_text(table_to_console(metrics_df), encoding="utf-8")
    corr_pretty_path.write_text(table_to_console(corr_df), encoding="utf-8")
    return metrics_path, corr_path


def _save_processed_data(btc_df: pd.DataFrame, eth_df: pd.DataFrame):
    btc_path = DATA_DIR / "btc.csv"
    eth_path = DATA_DIR / "eth.csv"
    btc_df.to_csv(btc_path, index=False)
    eth_df.to_csv(eth_path, index=False)
    return btc_path, eth_path


def _print_insights(metrics_df: pd.DataFrame, corr_df: pd.DataFrame):
    print("\nInsights")

    def metric_value(metric: str, column: str):
        series = metrics_df.loc[metrics_df["Metric"] == metric, column]
        return series.iloc[0] if not series.empty else pd.NA

    for regime in ["Overall", *REGIME_ORDER]:
        btc_col = f"BTC {regime}"
        eth_col = f"ETH {regime}"
        btc_vol = metric_value("Volatility (Annualized)", btc_col)
        eth_vol = metric_value("Volatility (Annualized)", eth_col)
        btc_mdd = metric_value("Max Drawdown", btc_col)
        eth_mdd = metric_value("Max Drawdown", eth_col)
        corr_col = "Overall" if regime == "Overall" else regime
        corr_val = corr_df[corr_col].iloc[0]

        lines = []
        if pd.notna(btc_vol) and pd.notna(eth_vol):
            if btc_vol > eth_vol:
                lines.append("BTC volatility is higher than ETH.")
            else:
                lines.append("ETH volatility is higher than BTC.")
        if pd.notna(btc_mdd) and pd.notna(eth_mdd):
            deeper = "BTC" if btc_mdd < eth_mdd else "ETH"
            lines.append(f"{deeper} experienced the deeper max drawdown.")
        if pd.notna(corr_val):
            if corr_val >= 0.8:
                lines.append(f"Correlation was very tight (corr={corr_val:.2f}).")
            elif corr_val >= 0.5:
                lines.append(f"Correlation stayed strong (corr={corr_val:.2f}).")
            elif corr_val >= 0:
                lines.append(f"Correlation was moderate (corr={corr_val:.2f}).")
            else:
                lines.append(f"Correlation turned negative (corr={corr_val:.2f}).")

        print(f"- {regime}: " + " ".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Crypto risk analysis for BTC and ETH using Binance Spot data.")
    parser.add_argument("--lookback_years", type=int, default=4, help="Historical lookback window in years (1-10).")
    parser.add_argument("--start", type=str, default=None, help="Custom start date in YYYY-MM-DD.")
    parser.add_argument("--end", type=str, default=None, help="Custom end date in YYYY-MM-DD.")
    args = parser.parse_args()

    lookback_years = None if args.start and args.end else max(1, min(10, args.lookback_years))

    _prepare_output_dirs()

    start_ms, end_ms, effective_end_day, end_was_clamped = _compute_times(
        lookback_years, args.start, args.end
    )

    if args.start and args.end:
        if end_was_clamped:
            print(
                f"Fetching data from {args.start} to {effective_end_day.strftime('%Y-%m-%d')} "
                f"(requested end {args.end} exceeds the latest closed UTC day)..."
            )
        else:
            print(f"Fetching data from {args.start} to {args.end}...")
    else:
        print(f"Fetching {lookback_years} year daily data from Binance...")

    fetcher = BinanceDataFetcher()
    btc_raw, eth_raw = fetcher.fetch_crypto_pair(start_ms, end_ms)

    prepared = preprocess_data(btc_raw, eth_raw)
    merged = prepared.merged.copy()
    print(f"BTC actual data start: {btc_raw['timestamp'].min().strftime('%Y-%m-%d')}")
    print(f"ETH actual data start: {eth_raw['timestamp'].min().strftime('%Y-%m-%d')}")
    print(f"Final overlapping analysis start: {merged['timestamp'].min().strftime('%Y-%m-%d')}")

    # Bỏ các dòng đầu chưa có return vì log return cần ít nhất 2 mốc giá.
    merged = merged.dropna(subset=["btc_return", "eth_return"]).reset_index(drop=True)
    if merged.empty:
        raise ValueError("Not enough overlapping BTC/ETH return data after preprocessing.")

    # Regime chỉ được phân từ BTC rolling volatility, sau đó dùng chung cho metrics và plots.
    merged["regime"] = classify_regimes(merged["btc_rolling_vol"])

    metrics_df = compute_metrics_table(merged)
    corr_df = compute_correlation_table(merged)

    print("\nMetrics Table")
    print(table_to_console(metrics_df))

    print("\nCorrelation Table")
    print(table_to_console(corr_df))

    print("\nGenerating dashboard and plots...")
    plot_volatility(merged)
    plot_drawdown(merged)
    plot_correlation(merged)
    plot_price_volume(merged, "BTC")
    plot_price_volume(merged, "ETH")
    dashboard_path = create_dashboard(merged)

    print("\nSaving tables and processed data...")
    _save_tables(metrics_df, corr_df)
    btc_processed, eth_processed = build_processed_asset_views(merged)
    _save_processed_data(btc_processed, eth_processed)

    print(f"Dashboard saved to: {dashboard_path}")
    _print_insights(metrics_df, corr_df)


if __name__ == "__main__":
    main()
