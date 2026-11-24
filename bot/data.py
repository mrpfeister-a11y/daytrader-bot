"""Download intraday price bars.

Uses Alpaca when API keys are set (years of history, free IEX feed).
Falls back to Yahoo Finance otherwise (only the last ~60 days of 15-min bars).
Returns one long table: symbol, ts, open, high, low, close, volume.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import config

NY = "America/New_York"
COLS = ["symbol", "ts", "open", "high", "low", "close", "volume"]


def _regular_hours(df: pd.DataFrame) -> pd.DataFrame:
    t = df["ts"].dt.tz_convert(NY)
    df = df.assign(ts=t)
    minutes = t.dt.hour * 60 + t.dt.minute
    return df[(minutes >= 9 * 60 + 30) & (minutes < 16 * 60)].reset_index(drop=True)


def from_alpaca(symbols, days: int) -> pd.DataFrame:
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(config.ALPACA_KEY, config.ALPACA_SECRET)
    end = datetime.now(timezone.utc)
    req = StockBarsRequest(
        symbol_or_symbols=list(symbols),
        timeframe=TimeFrame(config.BAR_MINUTES, TimeFrameUnit.Minute),
        start=end - timedelta(days=days),
        end=end,
        feed=DataFeed.IEX,
    )
    df = client.get_stock_bars(req).df.reset_index()
    df = df.rename(columns={"timestamp": "ts"})[COLS]
    return _regular_hours(df)


def from_yahoo(symbols, days: int) -> pd.DataFrame:
    import yfinance as yf

    days = min(days, 59)  # Yahoo limit for 15m bars
    raw = yf.download(
        list(symbols), period=f"{days}d", interval=f"{config.BAR_MINUTES}m",
        group_by="ticker", auto_adjust=True, prepost=False, progress=False,
        threads=True,
    )
    frames = []
    for sym in symbols:
        if sym not in raw.columns.get_level_values(0):
            continue
        d = raw[sym].dropna().reset_index()
        d.columns = [c.lower() for c in d.columns]
        d = d.rename(columns={"datetime": "ts", "date": "ts"})
        d["symbol"] = sym
        frames.append(d[COLS])
    df = pd.concat(frames, ignore_index=True)
    return _regular_hours(df)


def get_bars(symbols=None, days: int = 365) -> pd.DataFrame:
    symbols = symbols or config.UNIVERSE
    if config.ALPACA_KEY and config.ALPACA_SECRET:
        df = from_alpaca(symbols, days)
        src = "Alpaca"
    else:
        df = from_yahoo(symbols, days)
        src = "Yahoo (no Alpaca keys found, limited to ~60 days)"
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    print(f"[data] {len(df):,} bars for {df.symbol.nunique()} symbols from {src}")
    return df
