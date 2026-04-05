from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"

BINANCE_BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"

SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
}

INTERVAL = "1d"
INTERVAL_MS = 24 * 60 * 60 * 1000
MAX_LIMIT = 1000
ROLLING_WINDOW = 30

REGIME_ORDER = ["Deep Calm", "Calm", "Turbulent", "Stress Turbulent"]

REGIME_COLORS = {
    "Deep Calm": "#dceeff",
    "Calm": "#b9dbff",
    "Turbulent": "#ffd9b3",
    "Stress Turbulent": "#ffb3b3",
}

BTC_COLOR = "#f59e0b"
ETH_COLOR = "#2f80ed"
CORR_COLOR = "#8b5cf6"
VOL_COLOR = "#64748b"
