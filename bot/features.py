"""Turn raw bars into the signals the model learns from.

Every feature only uses information available at the close of that bar,
so there is no peeking into the future.
"""
import numpy as np
import pandas as pd

from . import config

FEATURES = [
    "ret_1", "ret_2", "ret_4", "ret_8",
    "z_1", "z_4",
    "gap", "ret_since_open", "vwap_dist", "range_pos",
    "vol_8", "vol_ratio", "volu_z", "rsi_14",
    "time_of_day",
    "mkt_ret_1", "mkt_ret_4", "rel_ret_4",
]


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    c = g["close"]
    lr = np.log(c).diff()
    day = g["ts"].dt.date
    g["day"] = day

    for n in (1, 2, 4, 8):
        g[f"ret_{n}"] = np.log(c).diff(n)
    g["vol_8"] = lr.rolling(8).std()
    vol_64 = lr.rolling(64).std()
    g["vol_ratio"] = g["vol_8"] / vol_64
    g["z_1"] = g["ret_1"] / vol_64
    g["z_4"] = g["ret_4"] / (vol_64 * 2)

    by_day = g.groupby(day)
    day_open = by_day["open"].transform("first")
    # previous day's last close, broadcast across today
    last_close = by_day["close"].last().shift(1)
    g["gap"] = np.log(day_open / day.map(last_close))
    g["ret_since_open"] = np.log(c / day_open)

    typical = (g["high"] + g["low"] + c) / 3
    pv = (typical * g["volume"]).groupby(day).cumsum()
    vv = g["volume"].groupby(day).cumsum().replace(0, np.nan)
    g["vwap_dist"] = np.log(c / (pv / vv))

    hi = g["high"].groupby(day).cummax()
    lo = g["low"].groupby(day).cummin()
    g["range_pos"] = (c - lo) / (hi - lo).replace(0, np.nan)

    g["volu_z"] = np.log1p(g["volume"]) - np.log1p(g["volume"]).rolling(64).mean()
    g["rsi_14"] = _rsi(c)

    mins = g["ts"].dt.hour * 60 + g["ts"].dt.minute - (9 * 60 + 30)
    g["time_of_day"] = mins / 390.0

    # Target: return over the next HORIZON_BARS, only within the same trading
    # day (a day trader never holds overnight).
    h = config.HORIZON_BARS
    g["fwd_ret"] = np.where(day.shift(-h) == day, c.shift(-h) / c - 1, np.nan)
    # 1-bar return actually earned while holding (used for P&L, not learning)
    g["next_ret"] = np.where(day.shift(-1) == day, c.shift(-1) / c - 1, np.nan)
    return g


def build(bars: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([_per_symbol(g) for _, g in bars.groupby("symbol")])

    # Market context: what SPY is doing at the same moment.
    mkt = df[df.symbol == "SPY"][["ts", "ret_1", "ret_4"]].rename(
        columns={"ret_1": "mkt_ret_1", "ret_4": "mkt_ret_4"})
    df = df.merge(mkt, on="ts", how="left")
    df["rel_ret_4"] = df["ret_4"] - df["mkt_ret_4"]

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.sort_values(["ts", "symbol"]).reset_index(drop=True)
