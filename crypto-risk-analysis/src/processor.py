from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ROLLING_WINDOW
from .utils import compute_drawdown, to_log_return


@dataclass
class PreparedData:
    merged: pd.DataFrame


def _rename_asset_columns(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    # Chỉ giữ các cột thật sự dùng trong pipeline phân tích.
    out = df[["timestamp", "close", "volume"]].copy()
    return out.rename(columns={
        "close": f"{asset}_close",
        "volume": f"{asset}_volume",
    })


def preprocess_data(btc_raw: pd.DataFrame, eth_raw: pd.DataFrame) -> PreparedData:
    required_cols = {"timestamp", "close", "volume"}
    for asset_name, df in [("BTC", btc_raw), ("ETH", eth_raw)]:
        missing = required_cols.difference(df.columns)
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"{asset_name} data is missing required columns: {missing_cols}")
        if df.empty:
            raise ValueError(f"{asset_name} data is empty. Unable to preprocess.")

    btc = _rename_asset_columns(btc_raw, "btc")
    eth = _rename_asset_columns(eth_raw, "eth")

    btc["btc_return"] = to_log_return(btc["btc_close"])
    eth["eth_return"] = to_log_return(eth["eth_close"])

    merged = pd.merge(
        btc[["timestamp", "btc_close", "btc_volume", "btc_return"]],
        eth[["timestamp", "eth_close", "eth_volume", "eth_return"]],
        on="timestamp",
        how="inner",
    ).sort_values("timestamp").reset_index(drop=True)

    if merged.empty:
        raise ValueError("BTC and ETH data have no overlapping timestamps after alignment.")

    # Drawdown phụ thuộc vào đường đi của giá, nên phải tính trên toàn bộ chuỗi giá đã căn theo thời gian.
    merged["btc_drawdown"] = compute_drawdown(merged["btc_close"])
    merged["eth_drawdown"] = compute_drawdown(merged["eth_close"])

    # Các đại lượng rolling vừa dùng để gán regime, vừa dùng để vẽ chart theo thời gian.
    merged["btc_rolling_vol"] = merged["btc_return"].rolling(ROLLING_WINDOW).std(ddof=1) * np.sqrt(365)
    merged["eth_rolling_vol"] = merged["eth_return"].rolling(ROLLING_WINDOW).std(ddof=1) * np.sqrt(365)
    merged["rolling_corr"] = merged["btc_return"].rolling(ROLLING_WINDOW).corr(merged["eth_return"])

    return PreparedData(merged=merged)


def build_processed_asset_views(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Lưu dữ liệu đã xử lý theo dạng dễ đọc và dễ kiểm tra lại.
    btc = merged[[
        "timestamp",
        "btc_close",
        "btc_volume",
        "btc_return",
        "btc_drawdown",
        "btc_rolling_vol",
        "regime",
    ]].copy()
    eth = merged[[
        "timestamp",
        "eth_close",
        "eth_volume",
        "eth_return",
        "eth_drawdown",
        "eth_rolling_vol",
        "regime",
    ]].copy()
    return btc, eth
