"""Central configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = {
    "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI", "Russell 2000": "^RUT",
    "VIX": "^VIX", "US 10Y": "^TNX", "DXY": "DX-Y.NYB", "Gold": "GC=F",
    "WTI Oil": "CL=F", "Bitcoin": "BTC-USD", "EUR/USD": "EURUSD=X",
}
START = "2007-01-01"   # covers GFC; dot-com is NOT covered by this free feed

# FRED public CSV endpoint (no key). value = (series id, publication lag in days)
FRED = {"curve": ("T10Y2Y", 1), "credit": ("BAMLH0A0HYM2", 1), "unrate": ("UNRATE", 35)}

# forward-drawdown event definition per horizon: min close in next h days <= thr
HORIZONS = {1: -0.015, 3: -0.025, 5: -0.03, 10: -0.04}

REGIMES = ["STRONG BULL", "BULL", "SIDEWAYS", "HIGH VOLATILITY", "BEAR", "CRISIS"]
REGIME_COLS = ["ret20", "vol20", "dd", "skew60", "vix", "mom20"]

EW_LEVELS = [(20, "LOW RISK", "#2e8b57"), (40, "WATCH", "#c9a227"), (60, "ELEVATED", "#e08a1e"),
             (80, "HIGH RISK", "#d9534f"), (100, "CRITICAL", "#a40000")]
REGIME_COLORS = {"STRONG BULL": "#1b7f4c", "BULL": "#7bc47f", "SIDEWAYS": "#9aa5b1",
                 "HIGH VOLATILITY": "#e0b23a", "BEAR": "#e07b39", "CRISIS": "#b3261e"}
SENSITIVITY = {"Aggressive Warning": 1.15, "Balanced": 1.0, "Conservative Warning": 0.85}

DISCLAIMER = ("This system is for research and educational purposes only. It does not provide "
              "financial advice, investment recommendations, or guaranteed predictions.")
