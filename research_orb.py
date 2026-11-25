"""Stocks-in-play opening-range breakout, tested the way the paper describes it:
Zarattini & Aziz (2023), "Can Day Trading Really Be Profitable? Evidence of Sustainable
Long-term Profits from Opening Range Breakout (ORB) Day Trading Strategy".

Rules (from the paper):
  * Universe: ALL US stocks (including ones that later disappeared, so the test isn't rigged
    by hindsight), price > $5, 14-day average volume > 1M shares, 14-day ATR > $0.50.
  * "In play": first 5-minute volume vs. its own 14-day average (relative volume). Each day
    take the 20 with the highest relative volume (at least 1.0).
  * First 5-min candle up -> buy-stop at its high; down -> sell-stop at its low; flat -> skip.
  * Stop loss 10% of the 14-day ATR away from entry. Otherwise exit at the close.
  * Size: risk 1% of the account per trade, total capped at 4x the account (BORROWED money).
    Also shown capped at 1x (no borrowing), which is what this bot is allowed to do.
  * Costs: $0.0035 per share commission, plus extra slippage (shown with 0 and 5 bps).

    python research_orb.py   # needs Alpaca keys; downloads a lot (about 1-2 hours)
"""
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from bot import config

CACHE = config.ROOT / "data_cache"
NY = "America/New_York"
START = "2016-01-01"
TOP_N = 20


def _client():
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(config.ALPACA_KEY, config.ALPACA_SECRET)


def _bars(client, symbols, start, end, minutes=None):
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    tf = TimeFrame.Day if minutes is None else TimeFrame(minutes, TimeFrameUnit.Minute)
    for attempt in range(5):
        try:
            df = client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=list(symbols), timeframe=tf, start=start, end=end,
                feed=DataFeed.SIP, adjustment="all")).df
            return df.reset_index() if not df.empty else pd.DataFrame()
        except Exception as e:
            print(f"  retry {attempt + 1}: {str(e)[:100]}", flush=True)
            time.sleep(10 * (attempt + 1))
    return pd.DataFrame()


def universe():
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass
    from alpaca.trading.requests import GetAssetsRequest
    tc = TradingClient(config.ALPACA_KEY, config.ALPACA_SECRET, paper=True)
    assets = tc.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY))
    ok_ex = {"NYSE", "NASDAQ", "AMEX"}          # stocks (ARCA/BATS are mostly ETFs)
    syms = sorted({a.symbol for a in assets if str(getattr(a.exchange, "value", a.exchange)) in ok_ex
                   and a.symbol.isalpha() and len(a.symbol) <= 5})
    print(f"[orb] universe: {len(syms):,} US stocks/ETFs (active and delisted)", flush=True)
    return syms


def daily(client, syms):
    path = CACHE / "orb_daily.pkl"
    if path.exists():
        return pd.read_pickle(path)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    frames = []
    for i in range(0, len(syms), 400):
        df = _bars(client, syms[i:i + 400], datetime.fromisoformat(START).replace(tzinfo=timezone.utc), end)
        if not df.empty:
            frames.append(df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]])
        print(f"[orb] daily bars {min(i + 400, len(syms))}/{len(syms)}", flush=True)
    d = pd.concat(frames, ignore_index=True)
    d["day"] = d["timestamp"].dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()
    d = d.drop(columns="timestamp")
    for c in ("open", "high", "low", "close", "volume"):
        d[c] = d[c].astype("float32")
    d["symbol"] = d["symbol"].astype("category")
    CACHE.mkdir(exist_ok=True)
    d.to_pickle(path)
    return d


def candidates(d: pd.DataFrame) -> pd.DataFrame:
    """Rows that pass the paper's filters, using only information from before that day."""
    d = d.sort_values(["symbol", "day"]).copy()
    g = d.groupby("symbol", observed=True)
    pc = g["close"].shift(1)
    tr = np.maximum(d["high"] - d["low"], np.maximum((d["high"] - pc).abs(), (d["low"] - pc).abs()))
    d["atr"] = tr.groupby(d["symbol"], observed=True).transform(lambda x: x.rolling(14).mean().shift(1))
    d["avgvol"] = g["volume"].transform(lambda x: x.rolling(14).mean().shift(1))
    d["prev_close"] = pc
    ok = (d["prev_close"] > 5) & (d["avgvol"] > 1e6) & (d["atr"] > 0.5)
    return d.loc[ok, ["symbol", "day", "atr", "prev_close"]]


def opening_bars(client, cand: pd.DataFrame) -> pd.DataFrame:
    """First 5-minute bar for every candidate (and the days around it, for its 14-day average)."""
    path = CACHE / "orb_open5.pkl"
    have = pd.read_pickle(path) if path.exists() else pd.DataFrame()
    done = set(have["day"].unique()) if len(have) else set()
    days = sorted(cand["day"].unique())
    by_day = cand.groupby("day")["symbol"].apply(lambda s: set(s.astype(str)))
    frames = [have] if len(have) else []
    t0 = time.time()
    for k, day in enumerate(days):
        if day in done:
            continue
        need = set()
        for dd in days[k:k + 15]:          # symbols that are candidates today or in the next 14 days
            need |= by_day.get(dd, set())
        need = sorted(need)
        o = pd.Timestamp(f"{pd.Timestamp(day).date()} 09:30", tz=NY).tz_convert("UTC").to_pydatetime()
        for i in range(0, len(need), 500):
            df = _bars(client, need[i:i + 500], o, o + timedelta(minutes=5), minutes=5)
            if not df.empty:
                df = df[df["timestamp"] == pd.Timestamp(o)]
                frames.append(pd.DataFrame({"symbol": df["symbol"].values, "day": day,
                                            "o": df["open"].values, "h": df["high"].values,
                                            "l": df["low"].values, "c": df["close"].values,
                                            "v": df["volume"].values}))
        if k % 100 == 0:
            print(f"[orb] opening bars {k}/{len(days)} days ({(time.time() - t0) / 60:.0f} min)", flush=True)
            pd.concat(frames, ignore_index=True).to_pickle(path)
    out = pd.concat(frames, ignore_index=True)
    out.to_pickle(path)
    return out


def pick(cand: pd.DataFrame, ob: pd.DataFrame) -> pd.DataFrame:
    ob = ob.sort_values(["symbol", "day"]).copy()
    ob["avg_v"] = ob.groupby("symbol")["v"].transform(lambda x: x.rolling(14, min_periods=10).mean().shift(1))
    ob["rvol"] = ob["v"] / ob["avg_v"]
    c = cand.assign(symbol=cand["symbol"].astype(str))
    m = c.merge(ob, on=["symbol", "day"], how="inner")
    m = m[(m["rvol"] >= 1) & (m["c"] != m["o"])]
    m = m.sort_values(["day", "rvol"], ascending=[True, False]).groupby("day").head(TOP_N)
    m["side"] = np.where(m["c"] > m["o"], 1, -1)
    return m


def day_bars(client, picks: pd.DataFrame) -> pd.DataFrame:
    """1-MINUTE bars for each picked stock-day (the paper's resolution: with 5-min bars a
    tight stop and the entry often fall in the same bar and the order is unknowable)."""
    path = CACHE / "orb_daybars1m.pkl"
    have = pd.read_pickle(path) if path.exists() else pd.DataFrame()
    done = set(have["day"].unique()) if len(have) else set()
    frames = [have] if len(have) else []
    days = sorted(picks["day"].unique())
    for k, (day, grp) in enumerate(picks.groupby("day")):
        if day in done:
            continue
        o = pd.Timestamp(f"{pd.Timestamp(day).date()} 09:30", tz=NY).tz_convert("UTC").to_pydatetime()
        df = _bars(client, sorted(grp["symbol"]), o, o + timedelta(hours=6, minutes=30), minutes=1)
        if not df.empty:
            df["day"] = day
            df = df[["symbol", "day", "timestamp", "open", "high", "low", "close"]]
            for col in ("open", "high", "low", "close"):
                df[col] = df[col].astype("float32")
            frames.append(df)
        if k % 200 == 0:
            print(f"[orb] day bars {k}/{len(days)}", flush=True)
            pd.concat(frames, ignore_index=True).to_pickle(path)
    out = pd.concat(frames, ignore_index=True)
    out.to_pickle(path)
    return out


def simulate(picks, bars, max_lev, slip_bps, start_equity=100_000.0):
    bars = bars.sort_values(["day", "symbol", "timestamp"])
    by = {k: g for k, g in bars.groupby(["day", "symbol"])}
    eq, rows = start_equity, []
    for day, grp in picks.groupby("day"):
        trades = []
        for r in grp.itertuples():
            b = by.get((day, r.symbol))
            if b is None or len(b) < 10:
                continue
            t0 = pd.Timestamp(f"{pd.Timestamp(day).date()} 09:35", tz=NY).tz_convert("UTC")
            after = b[b["timestamp"] >= t0]                     # minutes after the first 5 minutes
            if after.empty:
                continue
            entry = r.h if r.side == 1 else r.l
            hit = (after["high"] > entry) if r.side == 1 else (after["low"] < entry)
            if not hit.any():
                continue
            j = int(np.argmax(hit.to_numpy()))
            bar_open = float(after["open"].iloc[j])     # jumped past the level -> filled at the open
            entry = max(entry, bar_open) if r.side == 1 else min(entry, bar_open)
            risk = 0.1 * r.atr
            stop = entry - risk if r.side == 1 else entry + risk
            # entry minute: stopped only if it CLOSED beyond the stop (the order inside one
            # minute is unknowable); later minutes: any touch of the stop counts
            first_close = float(after["close"].iloc[j])
            same = (first_close <= stop) if r.side == 1 else (first_close >= stop)
            rest = after.iloc[j + 1:]
            stopped = (rest["low"] <= stop) if r.side == 1 else (rest["high"] >= stop)
            if same or stopped.any():
                exit_px = stop
            else:
                exit_px = float(rest["close"].iloc[-1]) if len(rest) else first_close
            trades.append((entry, exit_px, r.side, risk))
        if not trades:
            rows.append((day, 0.0, 0))
            continue
        shares = np.array([0.01 * eq / t[3] for t in trades])            # 1% risk per trade
        notional = shares * np.array([t[0] for t in trades])
        cap = max_lev * eq
        if notional.sum() > cap:
            shares *= cap / notional.sum()
        pnl = 0.0
        for sh, (en, ex, side, _) in zip(shares, trades):
            pnl += sh * (ex - en) * side - 2 * sh * 0.0035 - 2 * sh * en * slip_bps / 1e4
        ret = pnl / eq
        eq *= 1 + ret
        rows.append((day, ret, len(trades)))
    return pd.DataFrame(rows, columns=["day", "ret", "n"]).set_index("day")


def stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    yrs = max(len(r) / 252, 1e-9)
    tot = float(eq.iloc[-1] - 1) if len(r) else 0.0
    return {"yearly_pct": round(((1 + tot) ** (1 / yrs) - 1) * 100, 1) if tot > -1 else -100.0,
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2) if r.std() > 0 else 0.0,
            "max_dd_pct": round(float((eq / eq.cummax() - 1).min() * 100), 1),
            "avg_day_pct": round(float(r.mean() * 100), 3),
            "beat_spy_days_pct": None}


def main():
    t0 = time.time()
    client = _client()
    syms = universe()
    d = daily(client, syms)
    print(f"[orb] {len(d):,} daily rows ({(time.time() - t0) / 60:.0f} min)", flush=True)
    # the S&P 500 fund is only the yardstick (it's an ETF, so it isn't in the stock universe)
    s = _bars(client, ["SPY"], datetime.fromisoformat(START).replace(tzinfo=timezone.utc),
              datetime.now(timezone.utc) - timedelta(minutes=20))
    s["day"] = s["timestamp"].dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()
    spy = s.set_index("day")["close"].astype(float).pct_change().fillna(0)
    if spy.empty:
        raise RuntimeError("could not download the S&P 500 (SPY) prices for the comparison")
    cand = candidates(d)
    print(f"[orb] {len(cand):,} candidate stock-days, ~{len(cand) / cand['day'].nunique():.0f} per day", flush=True)
    ob = opening_bars(client, cand)
    picks = pick(cand, ob)
    print(f"[orb] {len(picks):,} in-play picks ({(time.time() - t0) / 60:.0f} min)", flush=True)
    bars = day_bars(client, picks)
    days = sorted(spy.index)
    hold = min(config.HOLDOUT_DAYS, len(days) // 3)
    cut = days[-hold]
    lines = ["=" * 78, " STOCKS-IN-PLAY OPENING-RANGE BREAKOUT (Zarattini & Aziz 2023), all US stocks",
             f" dev {days[0]:%Y-%m-%d} .. {days[-hold - 1]:%Y-%m-%d}  |  final exam {cut:%Y-%m-%d} .. "
             f"{days[-1]:%Y-%m-%d}", "=" * 78,
             f" {'version':38} {'DEV/yr':>8} {'Sharpe':>7} {'EXAM/yr':>8} {'Sharpe':>7} {'maxDD':>7} {'beats SPY days':>8}"]
    out = {}
    for label, lev, slip in [("paper: up to 4x BORROWED, no slippage", 4, 0),
                             ("4x BORROWED + 5 bps slippage", 4, 5),
                             ("1x (no borrowing) + 5 bps slippage", 1, 5)]:
        r = simulate(picks, bars, lev, slip)["ret"].reindex(days).fillna(0)
        out[label] = r
        dv, ex = stats(r[r.index < cut]), stats(r[r.index >= cut])
        beat = float((r > spy.reindex(days).fillna(0)).mean() * 100)
        lines.append(f" {label:38} {dv['yearly_pct']:>7}% {dv['sharpe']:>7} {ex['yearly_pct']:>7}% "
                     f"{ex['sharpe']:>7} {ex['max_dd_pct']:>6}% {beat:>7.0f}%")
    s_dv, s_ex = stats(spy[spy.index < cut]), stats(spy[spy.index >= cut])
    lines.append(f" {'BENCHMARK buy & hold S&P 500':38} {s_dv['yearly_pct']:>7}% {s_dv['sharpe']:>7} "
                 f"{s_ex['yearly_pct']:>7}% {s_ex['sharpe']:>7} {s_ex['max_dd_pct']:>6}%")
    lines.append("-" * 78)
    for label, r in out.items():
        yrs = r.groupby([d_.year for d_ in r.index]).apply(lambda x: (1 + x).prod() - 1)
        lines.append(f" {label}: " + "  ".join(f"{y}:{v * 100:+.0f}%" for y, v in yrs.items()))
    sy = spy.groupby([d_.year for d_ in spy.index]).apply(lambda x: (1 + x).prod() - 1)
    lines.append(" S&P 500: " + "  ".join(f"{y}:{v * 100:+.0f}%" for y, v in sy.items()))
    lines.append(f" (took {(time.time() - t0) / 60:.0f} min)")
    lines.append("=" * 78)
    rep = "\n".join(lines)
    print(rep)
    config.LOG_DIR.mkdir(exist_ok=True)
    (config.LOG_DIR / "research_orb_report.txt").write_text(rep + "\n", encoding="utf-8")
    return rep


if __name__ == "__main__":
    main()
