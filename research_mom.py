"""Honest re-test of Miles's "12-month momentum" swing rule (candidate for bot #10).

The rule (as Miles wrote it):
  * Stocks priced $5-$25 that trade at least $5M a day and are above their 200-day average.
  * Rank by the rise from 12 months ago to 1 month ago. Buy the top 5, equal dollars, at the
    open on the first trading day of the week.
  * Stop-loss 8% below the buy price. Sell everything at the close of the 10th trading day.
    Next round starts the following week.

Why re-test: the original test used TODAY's S&P 1500 lists, so every stock in it survived
to 2026. Here the same rule runs on Alpaca's full SIP history, which includes companies that
were delisted or went bankrupt, with realistic stop fills (a stock that opens below the stop
is sold at that open, not at -8%) and trading costs.

    python run.py research_mom   # needs Alpaca keys; uses data_cache/orb_daily.pkl if present
Writes reports/momentum_retest.md. Never trades.
"""
import io
import time
import traceback
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from bot import config
import research_orb as orb

OUT = config.ROOT / "reports" / "momentum_retest.md"
TOP_N, HOLD, STOP = 5, 10, 0.08
ETF_WORDS = ("ETF", "ETN", "PROSHARES", "DIREXION", "ISHARES", "SPDR", "INVESCO", " FUND",
             "LEVERAGED", " 2X", " 3X", "ULTRA", "BULL ", "BEAR ", "VOLATILITY", "TRUST UNIT")


# ---------------------------------------------------------------- data
def assets():
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass
    from alpaca.trading.requests import GetAssetsRequest
    tc = TradingClient(config.ALPACA_KEY, config.ALPACA_SECRET, paper=True)
    rows = []
    for a in tc.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY)):
        ex = str(getattr(a.exchange, "value", a.exchange))
        if ex in {"NYSE", "NASDAQ", "AMEX"} and a.symbol.isalpha() and len(a.symbol) <= 5:
            rows.append((a.symbol, (a.name or "").upper()))
    df = pd.DataFrame(rows, columns=["symbol", "name"]).drop_duplicates("symbol")
    df["fund"] = df["name"].apply(lambda n: any(w in n for w in ETF_WORDS))
    return df


def sp1500_today() -> set:
    out = set()
    for page in ("List_of_S%26P_500_companies", "List_of_S%26P_400_companies",
                 "List_of_S%26P_600_companies"):
        req = urllib.request.Request("https://en.wikipedia.org/wiki/" + page,
                                     headers={"User-Agent": "Mozilla/5.0 research-script"})
        html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
        for t in pd.read_html(io.StringIO(html)):
            col = next((c for c in t.columns if "symbol" in str(c).lower()
                        or "ticker" in str(c).lower()), None)
            if col is not None and len(t) > 300:
                out |= set(t[col].astype(str).str.strip())
                break
    return out


def wide(d: pd.DataFrame):
    d = d.assign(symbol=d["symbol"].astype(str)).drop_duplicates(["day", "symbol"])
    m = {c: d.pivot(index="day", columns="symbol", values=c) for c in ("open", "low", "close", "volume")}
    return m["open"], m["low"], m["close"], m["volume"]


# ---------------------------------------------------------------- the rule
def signals(C, V, allowed: pd.Index):
    """Score and eligibility for every day, using only that day's close and earlier."""
    C, V = C[allowed], V[allowed]
    mom = C.shift(21) / C.shift(252) - 1
    sma = C.rolling(200, min_periods=180).mean()
    dv = (C * V).rolling(20, min_periods=15).mean()
    jump = np.log(C / C.shift(1)).abs().rolling(252, min_periods=1).max()   # ticker reuse / bad data
    ok = (C >= 5) & (C <= 25) & (dv >= 5e6) & (C > sma) & mom.notna() & (jump < np.log(5))
    return mom.where(ok)


def rounds(days):
    """(buy day index, sell day index): buy on the week's first trading day, sell on day 10."""
    out, i = [], 0
    wk = pd.Index(days).isocalendar()
    key = list(zip(wk.year, wk.week))
    while i < len(days):
        if i > 0 and key[i] == key[i - 1]:
            i += 1
            continue
        j = i + HOLD - 1
        if j >= len(days):
            break
        out.append((i, j))
        i = j + 1
    return out


def simulate(score, O, L, C, days, slip_bps=10, stop=True, ideal_stop=False, top=TOP_N,
             rng=None, start=None):
    """Daily equity curve + trade list. Positions are equal dollars of the equity at the buy."""
    s = slip_bps / 1e4
    eq, curve, trades = 1.0, {}, []
    idx = {d: k for k, d in enumerate(days)}
    O, L, C = O.reindex(days), L.reindex(days), C.reindex(days)
    for i, j in rounds(days):
        if i == 0 or (start is not None and days[i] < start):
            continue
        sc = score.loc[days[i - 1]].dropna()
        sc = sc[O.loc[days[i], sc.index].notna()]          # must trade at the buy open
        if rng is not None:
            pick = list(rng.choice(sc.index, size=min(top, len(sc)), replace=False)) if len(sc) else []
        else:
            pick = list(sc.sort_values(ascending=False).index[:top])
        if not curve:
            curve[days[i - 1]] = eq                           # starting point (day-1 gains count)
        alloc = eq / top
        vals = np.zeros(j - i + 1)
        cash = eq - alloc * len(pick)
        for sym in pick:
            buy = O.at[days[i], sym] * (1 + s)
            stp = buy * (1 - STOP)
            sh, out_val, path = alloc / buy, None, []
            last = buy
            for k in range(i, j + 1):
                o, lo, c = O.iat[k, O.columns.get_loc(sym)], L.iat[k, L.columns.get_loc(sym)], \
                    C.iat[k, C.columns.get_loc(sym)]
                if np.isnan(c):                               # halted or delisted: keep last price
                    path.append(sh * last)
                    continue
                if stop and k > i and o <= stp:
                    out_val = sh * (stp if ideal_stop else o) * (1 - s)
                elif stop and lo <= stp:
                    out_val = sh * stp * (1 - s)
                if out_val is not None:
                    path += [out_val] * (j - k + 1)
                    break
                last = c
                path.append(sh * c)
            if out_val is None:
                out_val = sh * last * (1 - s)
                path[-1] = out_val
            vals += np.array(path)
            trades.append((days[i], sym, out_val / alloc - 1))
        for k in range(i, j + 1):
            curve[days[k]] = cash + vals[k - i]
        eq = curve[days[j]]
    r = pd.Series(curve).pct_change().fillna(0)
    return r, pd.DataFrame(trades, columns=["day", "symbol", "ret"])


def stats(r: pd.Series, t: pd.DataFrame | None = None) -> dict:
    yrs = max(len(r) / 252, 1e-9)
    eq = (1 + r).cumprod()
    out = {"cagr": (eq.iloc[-1] ** (1 / yrs) - 1) * 100 if len(r) else 0.0,
           "dd": float((eq / eq.cummax() - 1).min() * 100) if len(r) else 0.0,
           "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0}
    if t is not None and len(t):
        out["win"] = float((t["ret"] > 0).mean() * 100)
        out["tpy"] = len(t) / yrs
    return out


# ---------------------------------------------------------------- report
def main() -> str:
    t0 = time.time()
    client = orb._client()
    a = assets()
    d = orb.daily(client, sorted(a["symbol"]))
    print(f"[mom] {len(d):,} daily rows, {d['symbol'].nunique():,} symbols", flush=True)
    O, L, C, V = wide(d)
    spy_df = orb._bars(client, ["SPY"], datetime(2016, 1, 1, tzinfo=timezone.utc),
                       datetime.now(timezone.utc) - timedelta(minutes=20))
    spy_df["day"] = spy_df["timestamp"].dt.tz_convert(orb.NY).dt.tz_localize(None).dt.normalize()
    spy = spy_df.set_index("day")["close"].astype(float)
    days = [x for x in spy.index if x in C.index]
    last_seen = C.apply(lambda col: col.last_valid_index())
    alive = last_seen[last_seen >= days[-5]].index

    stocks = C.columns.intersection(a.loc[~a["fund"], "symbol"])
    try:
        sp = C.columns.intersection(sorted(sp1500_today()))
    except Exception as e:
        print("[mom] S&P lists unavailable:", e)
        sp = pd.Index([])
    print(f"[mom] stocks {len(stocks):,} (still trading {len(stocks.intersection(alive)):,}), "
          f"today's S&P 1500 found {len(sp):,}", flush=True)

    sc_all = signals(C, V, stocks)
    sc_alive = signals(C, V, stocks.intersection(alive))
    sc_sp = signals(C, V, sp) if len(sp) else None
    start = days[260]
    year_ago = days[-252]
    spy_r = spy.pct_change().reindex(days).fillna(0)
    spy_r = spy_r[spy_r.index >= start]

    rows, curves = [], {}

    def add(label, score, **kw):
        if score is None:
            rows.append(f"| {label} | n/a | | | | | |")
            return
        r, t = simulate(score, O, L, C, days, start=start, **kw)
        r = r[r.index >= start]
        f, ly = stats(r, t), stats(r[r.index >= year_ago])
        curves[label] = r
        rows.append(f"| {label} | {f['cagr']:+.1f}% | {f['dd']:.0f}% | {f['sharpe']:.2f} | "
                    f"{f.get('win', 0):.0f}% | {f.get('tpy', 0):.0f} | {ly['cagr']:+.1f}% |")
        print("[mom]", rows[-1], f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        return t

    add("A. Original style: today's S&P 1500, no costs, stops fill at exactly -8%", sc_sp,
        slip_bps=0, ideal_stop=True)
    add("B. Today's S&P 1500, real stop fills + 0.10% costs", sc_sp)
    add("C. All US stocks that still trade today (survivors only)", sc_alive)
    t_main = add("D. HONEST: all US stocks incl. delisted/bankrupt", sc_all)
    add("E. Honest, higher costs (0.25% per trade)", sc_all, slip_bps=25)
    add("F. Honest, no stop-loss", sc_all, stop=False)
    add("G. Honest, top 10 instead of top 5", sc_all, top=10)

    rand = []
    for seed in range(20):
        r, _ = simulate(sc_all, O, L, C, days, start=start, rng=np.random.default_rng(seed))
        rand.append(stats(r[r.index >= start])["cagr"])
    honest = stats(curves["D. HONEST: all US stocks incl. delisted/bankrupt"])["cagr"]
    s_full, s_ly = stats(spy_r), stats(spy_r[spy_r.index >= year_ago])

    years = sorted({x.year for x in spy_r.index})
    yr_lines = ["| Year | " + " | ".join(k[:2].strip(". ") for k in curves) + " | S&P 500 |",
                "|---" * (len(curves) + 2) + "|"]
    for y in years:
        cells = [f"{((1 + r[r.index.year == y]).prod() - 1) * 100:+.0f}%" for r in curves.values()]
        sy = (1 + spy_r[spy_r.index.year == y]).prod() - 1
        yr_lines.append(f"| {y} | " + " | ".join(cells) + f" | {sy * 100:+.0f}% |")

    worst = t_main.nsmallest(8, "ret") if t_main is not None else pd.DataFrame()
    best = t_main.nlargest(5, "ret") if t_main is not None else pd.DataFrame()
    fmt = lambda t: ", ".join(f"{r.symbol} {r.day:%Y-%m} {r.ret * 100:+.0f}%" for r in t.itertuples())

    rep = "\n".join([
        "# 12-month momentum rule: honest re-test",
        f"Run {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. Tested {start:%Y-%m-%d} to {days[-1]:%Y-%m-%d} "
        f"(the first year of data is needed to measure 12-month momentum). Alpaca SIP daily prices, "
        f"split/dividend adjusted. Top {TOP_N}, equal dollars, buy at the week's first open, "
        f"{STOP:.0%} stop, sell at the close of day {HOLD}. Paper research only.",
        "",
        "| Version | Per year | Worst drop | Sharpe | Trades won | Trades/yr | Last 12 months |",
        "|---|---|---|---|---|---|---|",
        *rows,
        f"| S&P 500 (SPY) buy & hold | {s_full['cagr']:+.1f}% | {s_full['dd']:.0f}% | "
        f"{s_full['sharpe']:.2f} | | | {s_ly['cagr']:+.1f}% |",
        "",
        f"Random 5 stocks from the same honest list (20 tries): median {np.median(rand):+.1f}%/yr, "
        f"range {min(rand):+.1f}% to {max(rand):+.1f}%. The rule beat {sum(x < honest for x in rand)} of 20.",
        "",
        "## Year by year",
        *yr_lines,
        "",
        f"Worst honest trades: {fmt(worst)}",
        f"Best honest trades: {fmt(best)}",
        "",
        f"Universe: {len(stocks):,} US stocks ({len(stocks.intersection(alive)):,} still trading), funds "
        f"removed by name; today's S&P 1500 matched {len(sp):,}. Prices are dividend-adjusted, so the "
        f"$5-$25 filter is slightly approximate in early years. Took {(time.time() - t0) / 60:.0f} min.",
    ])
    return rep


def run() -> None:
    """Called by run.py. Never raises (a failed research run must not trigger the bot's restart)."""
    try:
        rep = main()
    except Exception:
        rep = "# Momentum re-test FAILED\n\n```\n" + traceback.format_exc()[-3000:] + "\n```"
    print(rep)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(rep + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
