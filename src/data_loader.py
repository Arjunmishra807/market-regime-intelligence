"""Free data collection with disk-cache fallback. Never fabricates values."""
from __future__ import annotations
import io, logging
from datetime import datetime
import pandas as pd
import requests
from config import CACHE_DIR, FRED, START

log = logging.getLogger(__name__)


def _stamp(path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def _flat(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy(); df.columns = df.columns.get_level_values(0)
    return df


def load_prices(names: dict, start: str = START):
    """Returns (close_df keyed by asset name, status, last_updated).
    status: DELAYED (free feed, ~15 min delay) | CACHED | UNAVAILABLE."""
    path = CACHE_DIR / "prices.csv"
    try:
        import yfinance as yf
        raw = yf.download(list(names.values()), start=start, auto_adjust=False,
                          progress=False, threads=True)
        close = raw["Close"] if "Close" in raw else pd.DataFrame()
        if isinstance(close, pd.Series):
            close = close.to_frame(list(names.values())[0])
        close = close.dropna(how="all")
        if close.empty:
            raise RuntimeError("empty download")
        close = close.rename(columns={v: k for k, v in names.items()})
        close.to_csv(path)
        return close, "DELAYED", datetime.now().strftime("%Y-%m-%d %H:%M")
    except Exception as e:  # noqa
        log.warning("price download failed: %s", e)
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df[[c for c in df.columns if c in names]], "CACHED", _stamp(path)
    return pd.DataFrame(), "UNAVAILABLE", "n/a"


def load_ohlc(ticker: str, period: str = "6mo") -> pd.DataFrame:
    try:
        import yfinance as yf
        df = _flat(yf.download(ticker, period=period, auto_adjust=False, progress=False))
        return df.dropna(subset=["Close"]) if not df.empty else pd.DataFrame()
    except Exception as e:  # noqa
        log.warning("ohlc failed: %s", e); return pd.DataFrame()


def load_macro():
    """FRED public CSVs. Index is shifted by a publication lag so values only
    appear after they would have been known (no look-ahead).
    Returns (macro_df, status, last_obs dict)."""
    cols, last_obs, ok = {}, {}, False
    for key, (sid, lag) in FRED.items():
        path = CACHE_DIR / f"fred_{sid}.csv"
        s = None
        try:
            r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",
                             timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            d = pd.read_csv(io.StringIO(r.text))
            s = pd.to_numeric(d.iloc[:, 1], errors="coerce")
            s.index = pd.to_datetime(d.iloc[:, 0]); s = s.dropna(); s.to_csv(path); ok = True
        except Exception as e:  # noqa
            log.warning("FRED %s failed: %s", sid, e)
            if path.exists():
                s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        if s is not None and len(s):
            last_obs[key] = s.index[-1].strftime("%Y-%m-%d")
            s.index = s.index + pd.Timedelta(days=lag)
            cols[key] = s
    if not cols:
        return pd.DataFrame(), "UNAVAILABLE", {}
    return pd.DataFrame(cols).sort_index().ffill(), ("DELAYED" if ok else "CACHED"), last_obs
