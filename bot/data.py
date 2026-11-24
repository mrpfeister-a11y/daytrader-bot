"""Download intraday price bars.

Uses Alpaca when API keys are set: for learning, the full-market (SIP) history
back to 2016; for live decisions, the free real-time IEX feed.
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


def _alpaca_bars(symbols, start, end, feed) -> pd.DataFrame:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(config.ALPACA_KEY, config.ALPACA_SECRET)
    req = StockBarsRequest(
        symbol_or_symbols=list(symbols),
        timeframe=TimeFrame(config.BAR_MINUTES, TimeFrameUnit.Minute),
        start=start, end=end, feed=feed, adjustment="all",
    )
    df = client.get_stock_bars(req).df
    if df.empty:
        return pd.DataFrame(columns=COLS)
    df = df.reset_index().rename(columns={"timestamp": "ts"})[COLS]
    return _regular_hours(df)


def from_alpaca(symbols, days: int) -> pd.DataFrame:
    """Recent bars for live decisions (free real-time IEX feed)."""
    from alpaca.data.enums import DataFeed
    end = datetime.now(timezone.utc)
    return _alpaca_bars(symbols, end - timedelta(days=days), end, DataFeed.IEX)


def history_from_alpaca(symbols, start: str) -> pd.DataFrame:
    """All history since `start`, one year at a time (full-market SIP feed; the
    free plan allows it for anything older than 15 minutes). Falls back to IEX."""
    from alpaca.data.enums import DataFeed
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    t = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    frames = []
    while t < end:
        stop = min(t + timedelta(days=365), end)
        try:
            part = _alpaca_bars(symbols, t, stop, DataFeed.SIP)
        except Exception as e:  # plan doesn't allow SIP -> IEX (starts later, less volume info)
            print(f"[data] SIP not available ({str(e)[:80]}), using IEX for {t.year}")
            part = _alpaca_bars(symbols, t, stop, DataFeed.IEX)
        print(f"[data] {t.date()} -> {stop.date()}: {len(part):,} bars")
        frames.append(part)
        t = stop
    return pd.concat(frames, ignore_index=True).drop_duplicates(["symbol", "ts"])


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


def get_bars(symbols=None, start: str = None, days: int = None) -> pd.DataFrame:
    """Learning data. Default: everything since config.HISTORY_START."""
    symbols = symbols or config.UNIVERSE
    if config.ALPACA_KEY and config.ALPACA_SECRET:
        if days:
            start = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
        df = history_from_alpaca(symbols, start or config.HISTORY_START)
        src = "Alpaca"
    else:
        df = from_yahoo(symbols, days or 59)
        src = "Yahoo (no Alpaca keys found, limited to ~60 days)"
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    print(f"[data] {len(df):,} bars for {df.symbol.nunique()} symbols from {src}")
    return df
