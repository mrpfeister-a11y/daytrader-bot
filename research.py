"""Research lab: backtest many known trading ideas on all 15-min history since 2016.

    python research.py            # Alpaca keys -> all history; otherwise ~60 days of Yahoo data

The honest rules (so it finds real effects, not luck):
  * Each idea has a few settings. The best setting is chosen on the DEVELOPMENT years
    (2016 .. a year ago) only.
  * Then it is graded on the FINAL EXAM: the last ~252 trading days, never used to choose.
  * Results are per 100% of the account (no borrowing), after trading costs.
  * Same-bar ambiguity (a 15-min bar that hits both entry and stop) counts as a loss.

Ideas tested (all published / widely studied):
  1. Intraday momentum (Gao, Han, Li & Zhou 2018): the first half hour predicts the last half hour.
  2. Opening-range breakout (e.g. Zarattini & Aziz 2023): trade the break of the first 15/30 min range.
  3. Gap fade / gap-and-go: trade against (or with) the overnight gap.
  4. Cross-sectional reversal / momentum: buy the day's losers / winners among the 20 markets.
  5. Overnight effect: hold from close to next open (NOT a day trade - shown for comparison).
  Benchmarks: buy-and-hold S&P 500 (SPY), open-to-close.
"""
import itertools
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bot import config, data

COST = config.COST_BPS_PER_SIDE / 1e4      # per side
INDEXES = ["SPY", "QQQ", "IWM", "DIA"]
STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "JPM", "NFLX"]
CACHE = config.ROOT / "data_cache" / "bars.pkl"
SLOTS = 26                                 # 15-min bars 9:30 .. 15:45


# ------------------------------------------------------------------ data ----
def load_bars() -> pd.DataFrame:
    if CACHE.exists() and time.time() - CACHE.stat().st_mtime < 6 * 86400:
        print("[research] using cached bars")
        return pd.read_pickle(CACHE)
    bars = data.get_bars()
    CACHE.parent.mkdir(exist_ok=True)
    bars.to_pickle(CACHE)
    return bars


class Panel:
    """Prices as arrays [symbol, day, 15-min slot]."""

    def __init__(self, bars: pd.DataFrame):
        b = bars.copy()
        b["day"] = b["ts"].dt.date
        b["slot"] = ((b["ts"].dt.hour * 60 + b["ts"].dt.minute) - 570) // 15
        self.syms = sorted(b["symbol"].unique())
        self.days = np.array(sorted(b["day"].unique()))
        idx = pd.MultiIndex.from_product([self.syms, self.days], names=["symbol", "day"])
        S, D = len(self.syms), len(self.days)
        for f in ("open", "high", "low", "close", "volume"):
            p = b.pivot_table(index=["symbol", "day"], columns="slot", values=f, aggfunc="last")
            p = p.reindex(index=idx, columns=range(SLOTS))
            setattr(self, f[0].upper(), p.to_numpy(dtype=float).reshape(S, D, SLOTS))
        filled = pd.DataFrame(self.C.reshape(S * D, SLOTS)).ffill(axis=1).to_numpy().reshape(S, D, SLOTS)
        self.close = filled[:, :, -1]                     # day close (works on half days too)
        self.open = self.O[:, :, 0]
        self.prev_close = np.concatenate([np.full((S, 1), np.nan), self.close[:, :-1]], axis=1)
        self.year = np.array([d.year for d in self.days])

    def rows(self, names):
        return [self.syms.index(s) for s in names if s in self.syms]

    def at(self, hhmm: str):
        """Price at a clock time (close of the bar ending then); 16:00 = day close."""
        h, m = map(int, hhmm.split(":"))
        if (h, m) == (16, 0):
            return self.close
        return self.C[:, :, (h * 60 + m - 585) // 15]


def first_true(a):
    """Index of the first True along the last axis, or the length if none."""
    i = a.argmax(axis=-1)
    i[~a.any(axis=-1)] = a.shape[-1]
    return i


def portfolio(R, n):
    """Per-symbol returns [S, D] (nan = no trade) -> account daily return, each symbol 1/n."""
    return np.nansum(R, axis=0) / n


# ---------------------------------------------------------------- ideas ----
def intraday_momentum(P, uni, entry, confirm, min_move):
    r = P.rows(uni)
    first30 = P.at("10:00")[r] / P.prev_close[r] - 1
    sig = np.sign(first30)
    sig[np.abs(first30) < min_move] = 0
    if confirm:   # the 30 min before entry must point the same way
        h, m = map(int, entry.split(":"))
        before = f"{h if m >= 30 else h - 1}:{(m - 30) % 60:02d}"
        sig[np.sign(P.at(entry)[r] / P.at(before)[r] - 1) != sig] = 0
    px = P.at(entry)[r]
    R = sig * (P.close[r] / px - 1) - np.abs(sig) * 2 * COST
    R[(sig == 0) | np.isnan(R)] = np.nan
    return portfolio(R, len(r))


def orb(P, uni, n_bars, both_sides, stop_mid, rvol_min):
    r = P.rows(uni)
    H, L, V = P.H[r], P.L[r], P.V[r]
    rh, rl = np.nanmax(H[:, :, :n_bars], axis=2), np.nanmin(L[:, :, :n_bars], axis=2)
    Ha, La = H[:, :, n_bars:], L[:, :, n_bars:]
    with np.errstate(invalid="ignore"):
        fu = first_true(Ha > rh[..., None])
        fd = first_true(La < rl[..., None])
    T = Ha.shape[-1]
    j = np.arange(T)
    R = np.full(rh.shape, np.nan)
    for side in ((1, -1) if both_sides else (1,)):
        if side == 1:
            go, e, entry = (fu < T) & (fu <= fd), fu, rh
            stop = (rh + rl) / 2 if stop_mid else rl
            with np.errstate(invalid="ignore"):
                hit = first_true((La < stop[..., None]) & (j >= e[..., None]))
        else:
            go, e, entry = (fd < T) & (fd < fu), fd, rl
            stop = (rh + rl) / 2 if stop_mid else rh
            with np.errstate(invalid="ignore"):
                hit = first_true((Ha > stop[..., None]) & (j >= e[..., None]))
        exit_px = np.where(hit < T, stop, P.close[r])
        ret = side * (exit_px / entry - 1) - 2 * COST
        R = np.where(go, ret, R)
    if rvol_min:   # "in play": first range volume vs. its 14-day average
        v0 = np.nansum(V[:, :, :n_bars], axis=2)
        avg = pd.DataFrame(v0.T).rolling(14).mean().shift(1).to_numpy().T
        with np.errstate(invalid="ignore", divide="ignore"):
            R[~(v0 / avg >= rvol_min)] = np.nan
    return portfolio(R, len(r))


def gap(P, uni, min_gap, exit_at, fade):
    r = P.rows(uni)
    g = P.open[r] / P.prev_close[r] - 1
    sig = np.where(np.abs(g) >= min_gap, -np.sign(g) if fade else np.sign(g), 0)
    R = sig * (P.at(exit_at)[r] / P.open[r] - 1) - np.abs(sig) * 2 * COST
    R[(sig == 0) | np.isnan(R)] = np.nan
    return portfolio(R, len(r))


def cross_section(P, k, at, reversal, uni):
    r = P.rows(uni)
    score = P.at(at)[r] / P.prev_close[r] - 1
    ranks = pd.DataFrame(score).rank(axis=0, method="first").to_numpy()   # 1 = biggest loser
    n_ok = np.sum(~np.isnan(score), axis=0)
    low, high = ranks <= k, ranks > (n_ok - k)
    sig = np.where(low, 1, np.where(high, -1, 0)) * (1 if reversal else -1)
    sig[np.isnan(score)] = 0
    R = sig * (P.close[r] / P.at(at)[r] - 1) - np.abs(sig) * 2 * COST
    R[(sig == 0) | np.isnan(R)] = np.nan
    return portfolio(R, len(r))


def overnight(P, uni):
    r = P.rows(uni)
    R = np.full(P.close[r].shape, np.nan)
    R[:, 1:] = P.open[r][:, 1:] / P.close[r][:, :-1] - 1 - 2 * COST
    return portfolio(R, len(r))


def open_to_close(P, uni):
    r = P.rows(uni)
    return portfolio(P.close[r] / P.open[r] - 1 - 2 * COST, len(r))


def buy_hold_spy(P):
    r = P.rows(["SPY"])[0]
    out = P.close[r] / P.prev_close[r] - 1
    return np.nan_to_num(out)


ALL = config.UNIVERSE
IDEAS = {
    "1 Intraday momentum": (intraday_momentum, dict(
        uni=[INDEXES, ALL], entry=["15:00", "15:30"], confirm=[False, True], min_move=[0.0, 0.003])),
    "2 Opening-range breakout": (orb, dict(
        uni=[ALL, STOCKS], n_bars=[1, 2], both_sides=[True, False], stop_mid=[False, True],
        rvol_min=[0, 1.5])),
    "3 Gap fade / gap-and-go": (gap, dict(
        uni=[INDEXES, ALL], min_gap=[0.003, 0.01], exit_at=["10:30", "12:00", "16:00"], fade=[True, False])),
    "4 Losers/winners of the day": (cross_section, dict(
        k=[3, 5], at=["09:45", "10:00", "11:00"], reversal=[True, False], uni=[ALL])),
    "5 Overnight hold (NOT day trading)": (overnight, dict(uni=[["SPY"], INDEXES, ALL])),
}


# -------------------------------------------------------------- scoring ----
def stats(d: np.ndarray) -> dict:
    d = np.nan_to_num(d)
    eq = np.cumprod(1 + d)
    years = max(len(d) / 252, 1e-9)
    tot = eq[-1] - 1 if len(d) else 0
    traded = d != 0
    return {
        "avg_daily_pct": round(float(d.mean() * 100), 4),
        "avg_traded_day_pct": round(float(d[traded].mean() * 100), 4) if traded.any() else 0.0,
        "yearly_pct": round(float(((1 + tot) ** (1 / years) - 1) * 100), 2) if tot > -1 else -100.0,
        "total_pct": round(float(tot * 100), 2),
        "sharpe": round(float(d.mean() / d.std() * np.sqrt(252)), 2) if d.std() > 0 else 0.0,
        "max_dd_pct": round(float((eq / np.maximum.accumulate(eq) - 1).min() * 100), 2) if len(d) else 0.0,
        "days_traded_pct": round(float(traded.mean() * 100), 1),
        "win_pct": round(float((d[traded] > 0).mean() * 100), 1) if traded.any() else 0.0,
    }


def per_year(d, years):
    return {int(y): round(float((np.prod(1 + np.nan_to_num(d[years == y])) - 1) * 100), 1)
            for y in np.unique(years)}


def main():
    P = Panel(load_bars())
    n = len(P.days)
    hold = min(config.HOLDOUT_DAYS, n // 3)
    dev, exam = slice(0, n - hold), slice(n - hold, n)
    print(f"[research] {len(P.syms)} markets, {n} days: dev {P.days[0]}..{P.days[n - hold - 1]}, "
          f"exam {P.days[n - hold]}..{P.days[-1]}")
    results = []
    for name, (fn, grid) in IDEAS.items():
        keys = list(grid)
        tried = []
        for combo in itertools.product(*grid.values()):
            kw = dict(zip(keys, combo))
            d = fn(P, **kw)
            tried.append((stats(d[dev])["sharpe"], kw, d))
        tried.sort(key=lambda t: -t[0])
        _, kw, d = tried[0]
        results.append({
            "idea": name, "settings_tried": len(tried),
            "best_settings": {k: (v if not isinstance(v, list) else
                                  ("indexes" if v == INDEXES else "stocks" if v == STOCKS else
                                   "all 20" if v == ALL else ",".join(v))) for k, v in kw.items()},
            "dev": stats(d[dev]), "exam": stats(d[exam]), "years": per_year(d, P.year),
        })
        print(f"[research] {name}: dev Sharpe {results[-1]['dev']['sharpe']}, "
              f"exam Sharpe {results[-1]['exam']['sharpe']}", flush=True)
    bench = {"Buy & hold S&P 500 (SPY)": buy_hold_spy(P), "Open-to-close, all 20": open_to_close(P, ALL)}
    for name, d in bench.items():
        results.append({"idea": "BENCHMARK " + name, "settings_tried": 1, "best_settings": {},
                        "dev": stats(d[dev]), "exam": stats(d[exam]), "years": per_year(d, P.year)})

    lines = ["=" * 78, " RESEARCH LAB - every idea tuned on DEV years only, graded on the FINAL EXAM",
             f" dev {P.days[0]} .. {P.days[n - hold - 1]}   |   exam {P.days[n - hold]} .. {P.days[-1]}",
             f" Per 100% of the account, no borrowing, after {config.COST_BPS_PER_SIDE:.0f} bps cost per side",
             "=" * 78,
             f" {'idea':36} {'DEV/yr':>8} {'Sharpe':>7} {'EXAM/yr':>8} {'Sharpe':>7} {'avg/day':>8} {'maxDD':>7}"]
    for r in results:
        lines.append(f" {r['idea'][:36]:36} {r['dev']['yearly_pct']:>7}% {r['dev']['sharpe']:>7} "
                     f"{r['exam']['yearly_pct']:>7}% {r['exam']['sharpe']:>7} "
                     f"{r['exam']['avg_daily_pct']:>7}% {r['exam']['max_dd_pct']:>6}%")
    lines.append("-" * 78)
    for r in results:
        lines.append(f" {r['idea']}: " + ", ".join(f"{k}={v}" for k, v in r["best_settings"].items()))
        lines.append("    by year: " + "  ".join(f"{y}:{v:+}%" for y, v in r["years"].items()))
        lines.append(f"    exam: traded {r['exam']['days_traded_pct']}% of days, won {r['exam']['win_pct']}% "
                     f"of those, avg {r['exam']['avg_traded_day_pct']}% per traded day")
    lines.append("=" * 78)
    rep = "\n".join(lines)
    print(rep)
    config.LOG_DIR.mkdir(exist_ok=True)
    (config.LOG_DIR / "research_report.txt").write_text(rep + "\n", encoding="utf-8")
    (config.LOG_DIR / "research.json").write_text(json.dumps(results, indent=1, default=str))
    return rep


if __name__ == "__main__":
    main()
