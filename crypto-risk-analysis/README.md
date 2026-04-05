# Crypto Risk Analysis

This project analyzes BTC and ETH risk behavior using Binance Spot daily kline data.

## What it does

- Fetches closed daily candles for `BTCUSDT` and `ETHUSDT`
- Computes log returns, 30-day BTC/ETH rolling volatility, drawdown, rolling correlation, and regime labels
- Classifies market regimes using BTC volatility percentiles
- Builds finance-style metrics tables using overall and BTC-regime-filtered subsets
- Exports:
  - processed data to `data/btc.csv` and `data/eth.csv`
  - plots to `output/plots/`
  - tables to `output/tables/`
  - combined dashboard to `output/dashboard.png`

## Project structure

```text
crypto-risk-analysis/
├── data/
│   ├── btc.csv
│   └── eth.csv
├── output/
│   ├── plots/
│   ├── tables/
│   └── dashboard.png
├── src/
├── main.py
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Optional:

```bash
python main.py --lookback_years 1
python main.py --lookback_years 10
python main.py --start 2021-06-01 --end 2026-04-03
python main.py --start 2010-01-01 --end 2026-01-01

```

## Notes

- Only processed data is written to `data/`.
- `btc.csv` and `eth.csv` are overwritten on every run.
- Processed asset files keep only analysis columns: timestamp, close, volume, return, drawdown, rolling volatility, and regime.
- All analysis uses aligned timestamps before computing correlation and regime-based metrics.
- Binance public market data is fetched from the Spot REST kline endpoint.
- `--end YYYY-MM-DD` is treated as inclusive and only closed daily candles are requested.
- If `--end` exceeds the latest closed UTC day, it is clamped automatically to avoid using a still-open daily candle.
