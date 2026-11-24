"""Turn raw bars into the signals the model learns from.

Every feature only uses information available at the close of that bar,
so there is no peeking into the future.

Besides the basic price/volume signals, it adds effects that published
research has found in intraday data (they are only *candidates*; the
walk-forward test decides whether they actually help):
  * Intraday momentum: the first half hour's move (plus the overnight gap)
    tends to carry into the rest of the day (Gao, Han, Li & Zhou, 2018).
  * Short-term reversal / momentum from the last days (daily context).
  * Cross-sectional ranks: how a market is doing compared with the other 19
    right now (relative strength, used by the market-neutral "relative" style).
  * Risk-on / risk-off context: S&P 500 and bonds (TLT) at the same moment.
"""
import numpy as np
import pandas as pd

from . import config

FEATURES = [
    "ret_1", "ret_2", "ret_4", "ret_8",
    "z_1", "z_4",
    "gap", "ret_since_open", "vwap_dist", "range_pos",
    "vol_8", "vol_ratio", "volu_z", "rsi_14",
    "time_of_day", "dow",
    "first30", "prev_day_ret", "ret_5d", "ret_20d", "dvol_20", "dist_20d_high",
    "mkt_ret_1", "mkt_ret_4", "mkt_ret_since_open", "mkt_vol_8", "bond_ret_since_open",
    "rel_ret_4",
    "rank_ret_4", "rank_ret_since_open", "rank_gap", "rank_z_1",
]


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("ts").copy()
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
    daily_close = by_day["close"].last()              # one value per day
    g["gap"] = np.log(day_open / day.map(daily_close.shift(1)))
    g["ret_since_open"] = np.log(c / day_open)

    # first half hour (two 15-min bars), known from the 2nd bar's close on
    n_in_day = by_day.cumcount()
    second_close = c.where(n_in_day == 1).groupby(day).transform("max")
    g["first30"] = np.where(n_in_day >= 1, np.log(second_close / day_open), np.nan)

    # daily context, using only days BEFORE today
    dlr = np.log(daily_close).diff()
    ctx = pd.DataFrame({
        "prev_day_ret": dlr.shift(1),
        "ret_5d": np.log(daily_close).diff(5).shift(1),
        "ret_20d": np.log(daily_close).diff(20).shift(1),
        "dvol_20": dlr.rolling(20).std().shift(1),
        "dist_20d_high": (daily_close / daily_close.rolling(20).max() - 1).shift(1),
    })
    for col in ctx:
        g[col] = day.map(ctx[col]).astype(float)

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
    g["dow"] = g["ts"].dt.dayofweek.astype(float)

    # Targets: return over the next h bars (each horizon the backtest tries), only
    # within the same trading day (a day trader never holds overnight).
    for h in config.HORIZONS:
        g[f"fwd_ret_{h}"] = np.where(day.shift(-h) == day, c.shift(-h) / c - 1, np.nan)
    # 1-bar return actually earned while holding (used for P&L, not learning)
    g["next_ret"] = np.where(day.shift(-1) == day, c.shift(-1) / c - 1, np.nan)
    return g


def build(bars: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([_per_symbol(g) for _, g in bars.groupby("symbol")])

    # Market context: what SPY (stocks) and TLT (bonds) are doing right now.
    def ctx(sym, cols, names):
        d = df[df.symbol == sym][["ts"] + cols]
        return d.rename(columns=dict(zip(cols, names)))

    mkt = ctx("SPY", ["ret_1", "ret_4", "ret_since_open", "vol_8"],
              ["mkt_ret_1", "mkt_ret_4", "mkt_ret_since_open", "mkt_vol_8"])
    df = df.merge(mkt, on="ts", how="left")
    if (df.symbol == "TLT").any():
        df = df.merge(ctx("TLT", ["ret_since_open"], ["bond_ret_since_open"]), on="ts", how="left")
    else:
        df["bond_ret_since_open"] = np.nan
    df["rel_ret_4"] = df["ret_4"] - df["mkt_ret_4"]

    # Cross-sectional: rank 0..1 among all markets at the same moment.
    by_ts = df.groupby("ts")
    for col in ("ret_4", "ret_since_open", "gap", "z_1"):
        df[f"rank_{col}"] = by_ts[col].rank(pct=True)
    # Target for the market-neutral style: beat the average of all markets?
    for h in config.HORIZONS:
        df[f"fwd_rel_{h}"] = df[f"fwd_ret_{h}"] - by_ts[f"fwd_ret_{h}"].transform("mean")

    df = df.replace([np.inf, -np.inf], np.nan)
    num = df.select_dtypes("float64").columns
    df[num] = df[num].astype("float32")          # half the memory for years of data
    return df.sort_values(["ts", "symbol"]).reset_index(drop=True)
