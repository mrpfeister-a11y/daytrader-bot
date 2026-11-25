"""End of day: learn from today's own decisions and trades, then write the report.

What it learns from its own trading (logs/learned.json):
  * Live accuracy: every 15 min it logged a guess for each market, even while only
    watching. Afterwards it checks those guesses against what really happened.
  * Real trading cost: compares the price it decided at with the price it actually got,
    and the next honesty test uses that cost if it's worse than assumed.
  * Live check: after 20 trading days, if real results are losing AND its guesses are
    wrong more than right, it pauses itself (watch only) until a new test passes.
"""
import csv
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import config, data, live

LEARNED = config.LOG_DIR / "learned.json"
DAYS = config.LOG_DIR / "live_days.csv"


def load_learned() -> dict:
    try:
        return json.loads(LEARNED.read_text())
    except (OSError, ValueError):
        return {}


def _fmt(pct, usd=None):
    s = f"{pct:+.2f}%"
    return s + (f" ({'+' if usd >= 0 else '-'}${abs(usd):,.2f})" if usd is not None else "")


def _todays_orders(c, day: str):
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    start = datetime.fromisoformat(day).replace(tzinfo=live.NY)
    orders = c.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=start,
                                           symbols=config.UNIVERSE, limit=500))
    return [o for o in orders if o.filled_at is not None and float(o.filled_qty or 0) > 0]


def _decisions(day: str) -> pd.DataFrame:
    path = config.LOG_DIR / "decisions.csv"
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path)
    d["bar_ts"] = pd.to_datetime(d["bar_ts"], utc=True).dt.tz_convert(live.NY)
    return d[d["bar_ts"].dt.date.astype(str) == day]


def shadow_check(day: str, horizon: int, who: str = "champion") -> dict:
    """Were today's guesses right? (works in watch mode too). who = champion or challenger."""
    dec = _decisions(day)
    if dec.empty:
        return {"n": 0}
    dec = dec[(dec["mode"] == "challenger") if who == "challenger" else (dec["mode"] != "challenger")]
    dec = dec[dec["signal"] != 0]
    if dec.empty:
        return {"n": 0}
    bars = data.from_alpaca(config.UNIVERSE, days=4)
    bars = bars[bars["ts"].dt.date.astype(str) == day].sort_values(["symbol", "ts"])
    bars["fwd"] = bars.groupby("symbol")["close"].shift(-horizon) / bars["close"] - 1
    m = dec.merge(bars[["symbol", "ts", "fwd"]], left_on=["symbol", "bar_ts"],
                  right_on=["symbol", "ts"], how="inner").dropna(subset=["fwd"])
    if m.empty:
        return {"n": 0}
    gain = np.sign(m["signal"]) * m["fwd"]
    return {"n": int(len(m)), "hits": int((gain > 0).sum()),
            "avg_bps": round(float(gain.mean() * 1e4), 2)}


def slippage(orders, day: str) -> list:
    """bps lost between the decision price and the real fill (positive = cost)."""
    path = config.LOG_DIR / "trades.csv"
    if not path.exists() or not orders:
        return []
    want = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["order_id"] and r["decision_price"]:
                want[r["order_id"]] = float(r["decision_price"])
    out = []
    for o in orders:
        p0 = want.get(str(o.id))
        if p0 and o.filled_avg_price:
            fill = float(o.filled_avg_price)
            sign = 1 if o.side.value == "buy" else -1
            out.append(sign * (fill / p0 - 1) * 1e4)
    return out


def learn(day: str, mode: str, day_pct: float, shadow: dict, slips: list) -> dict:
    row = [day, mode, round(day_pct, 4), shadow.get("n", 0), shadow.get("hits", 0),
           shadow.get("avg_bps", ""), round(float(np.median(slips)), 2) if slips else ""]
    rows = []
    if DAYS.exists():
        rows = [r for r in csv.reader(DAYS.open()) if r and r[0] != "day" and r[0] != day]
    rows.append([str(x) for x in row])
    with DAYS.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "mode", "day_pct", "guesses", "right", "avg_gain_bps", "slippage_bps"])
        w.writerows(rows)

    L = load_learned()
    since = L.get("review_from", "")
    recent = [r for r in rows if r[0] >= since]
    n = sum(int(r[3]) for r in recent)
    hits = sum(int(r[4]) for r in recent)
    slip = [float(r[6]) for r in rows if r[6] not in ("", None)]
    traded = [r for r in recent if r[1] == "trade"]
    live_ret = float(np.prod([1 + float(r[2]) / 100 for r in traded]) - 1) if traded else 0.0
    L.update({
        "updated": day,
        "live_guesses": n,
        "live_hit_rate_pct": round(hits / n * 100, 1) if n else None,
        "live_trading_days": len(traded),
        "live_return_pct": round(live_ret * 100, 2),
        "slippage_bps_per_side": round(float(np.median(slip)), 2) if slip else None,
    })
    if (len(traded) >= config.LIVE_REVIEW_MIN_DAYS and live_ret < 0
            and n and hits / n < 0.5 and not L.get("paused")):
        L["paused"] = True
        L["paused_on"] = day
        L["just_paused"] = True
    LEARNED.write_text(json.dumps(L, indent=1))
    return L


def contest(day: str, bundle, champ_today: dict) -> str:
    """Champion vs. challenger on live guesses. The better one is kept, so the bot can only
    get better. Returns a line for the report ('' if there is no challenger)."""
    from . import model
    ch = model.load_challenger()
    if not ch:
        return ""
    L = load_learned()
    c = L.get("contest")
    if not c or c.get("challenger_trained") != ch["trained_at"]:
        c = {"challenger_trained": ch["trained_at"], "days": 0,
             "champ": [0, 0, 0.0], "chall": [0, 0, 0.0]}           # guesses, right, total bps
    try:
        b = shadow_check(day, ch["horizon"], "challenger")
    except Exception:
        b = {"n": 0}
    for key, s in (("champ", champ_today), ("chall", b)):
        if s.get("n"):
            c[key][0] += s["n"]
            c[key][1] += s["hits"]
            c[key][2] += s["avg_bps"] * s["n"]
    c["days"] += 1
    avg = lambda k: c[k][2] / c[k][0] if c[k][0] else float("-inf")
    line = (f"🏆 Champion vs. challenger, day {c['days']}: current model {avg('champ'):+.1f} bps per guess "
            f"({c['champ'][0]} guesses) vs. new model {avg('chall'):+.1f} bps ({c['chall'][0]} guesses)")
    line = line.replace("-inf", "n/a")
    better = avg("chall") > avg("champ") and c["chall"][0] >= config.CONTEST_MIN_GUESSES
    allowed = ch["edge_passed"] or not (bundle and bundle["edge_passed"])
    if c["days"] >= config.CONTEST_MIN_DAYS and better and allowed:
        config.CHALLENGER_PATH.replace(config.MODEL_PATH)
        line += " -> **the new model won and now trades**"
        L.pop("contest", None)
        L["promoted"] = L.get("promoted", 0) + 1
    elif c["days"] >= config.CONTEST_MAX_DAYS:
        config.CHALLENGER_PATH.unlink(missing_ok=True)
        line += " -> the current model was as good or better, so it stays"
        L.pop("contest", None)
        L["kept"] = L.get("kept", 0) + 1
    else:
        L["contest"] = c
    LEARNED.write_text(json.dumps(L, indent=1))
    return line


def sip_mine(sym):
    return sym not in config.UNIVERSE


def sip_today(day: str) -> dict:
    """Bot #9's results file. Bot #9 runs as a separate job, so read the newest copy
    from GitHub (without touching this job's own files); fall back to the local copy."""
    import subprocess
    rel = "logs/sip_today.json"
    try:
        subprocess.run(["git", "fetch", "-q", "origin", "main"], cwd=config.ROOT, timeout=60,
                       capture_output=True, check=True)
        txt = subprocess.run(["git", "show", f"origin/main:{rel}"], cwd=config.ROOT, timeout=30,
                             capture_output=True, text=True, check=True).stdout
        d = json.loads(txt)
    except Exception:
        try:
            d = json.loads((config.LOG_DIR / "sip_today.json").read_text())
        except (OSError, ValueError):
            d = {}
    return d if d.get("day") == day else {}


def bot9_section(day: str, t9: dict, hist: dict) -> tuple:
    """(markdown, problems) for bot #9 in the shared daily report."""
    b9 = t9.get("budget") or 0
    head = f"## 🚀 Bot #9 - stocks in play (budget ${b9:,.0f})" if b9 else "## 🚀 Bot #9 - stocks in play"
    if not t9:
        return head + "\n\n⚠️ Its results for today aren't in yet (the watchdog will check).", \
            ["bot #9 results missing"]
    res = [r for r in t9.get("results", []) if r.get("triggered")]
    pnl = sum(r["pnl"] for r in res)
    practice = t9["mode"] == "practice"
    lines = [head, ""]
    if practice:
        lines.append(f"🧪 **PRACTICE** (no orders yet): it would have made {'+' if pnl >= 0 else '-'}${abs(pnl):,.2f} "
                     f"({pnl / b9 * 100:+.2f}% of its budget)" if b9 else "🧪 **PRACTICE**")
    else:
        tot9 = sum(hist["sip"].values())
        lines.append(f"Today {_fmt(pnl / b9 * 100 if b9 else 0, pnl)} · since start "
                     f"{'+' if tot9 >= 0 else '-'}${abs(tot9):,.2f}")
    picks = t9.get("results", [])
    lines.append(f"Picked {len(picks)} stocks in play, {len(res)} broke out and were "
                 + ("traded" if not practice else "practice-traded")
                 + f", {sum(1 for r in res if r['pnl'] > 0)} won")
    if res:
        rows = ["| Stock | Way | Result |", "|---|---|---|"]
        for r in sorted(res, key=lambda r: -r["pnl"])[:12]:
            dot = "🟢" if r["pnl"] > 0 else "🔴" if r["pnl"] < 0 else "⚪"
            rows.append(f"| {r['symbol']} | {'buy' if r['side'] == 1 else 'short'} | "
                        f"{dot} {'+' if r['pnl'] >= 0 else '-'}${abs(r['pnl']):,.2f} |")
        lines.append("\n".join(rows))
    L = t9.get("learned", {})
    bits = []
    if L.get("slippage_bps") is not None:
        bits.append(f"real cost {L['slippage_bps']} bps per trade")
    if L.get("real_days"):
        bits.append(f"{L['real_days']} real days, {L.get('real_total_pct', 0):+.2f}% in total")
    p = t9.get("params", {})
    bits.append(f"settings: top {p.get('top_n')}, stop {p.get('stop_atr', 0) * 100:.0f}% of ATR, "
                f"borrowing up to {t9.get('leverage', 1)}x")
    lines.append("📚 " + "; ".join(bits))
    if L.get("paused"):
        lines.append("⏸️ It paused itself (real results were losing after 20+ days): practice only "
                     "until its monthly re-test passes.")
    bad = [f"bot #9: Alpaca refused {len(t9['refused'])} order(s)"] if t9.get("refused") else []
    if not picks:
        bad.append("bot #9 picked no stocks today (possible price-data problem)")
    return "\n\n".join(x for x in lines if x is not None), bad


def daily_report(c, bundle, st: dict, day: str) -> tuple:
    """Returns (title, markdown, urgent)."""
    acct = c.get_account()
    equity, last = float(acct.equity), float(acct.last_equity)
    start = float(st.get("start_equity") or equity)
    day_usd, tot_usd = equity - last, equity - start              # whole account (both bots)
    acct_day_pct = day_usd / last * 100 if last else 0.0
    tot_pct = tot_usd / start * 100 if start else 0.0
    # bot #8's own part (it trades only its 20 markets, so its fills are exactly its trades)
    b8 = last * config.BUDGET_SHARE["dt"]
    pnl8 = live.own_day_pnl(c, day, live.is_dt)
    day_pct = pnl8 / b8 * 100 if b8 else 0.0
    hist = st.setdefault("pnl", {"dt": {}, "sip": {}})
    hist["dt"][day] = round(pnl8, 2)
    L0 = load_learned()
    trading = bool(bundle and bundle.get("edge_passed")) and not L0.get("paused") and st.get("guard_ok", True)
    mode = "trade" if trading else "watch"

    orders = _todays_orders(c, day)
    horizon = bundle.get("horizon", config.HORIZON_BARS) if bundle else config.HORIZON_BARS
    try:
        shadow = shadow_check(day, horizon)
    except Exception as e:
        shadow = {"n": 0, "error": str(e)[:100]}
    slips = slippage(orders, day)
    L = learn(day, mode, day_pct, shadow, slips)

    cycles = st.get("cycles", {}).get(day, 0)
    restarts = st.get("restarts", {}).get(day, 0)
    problems = []
    if cycles < 15:
        problems.append(f"only {cycles} of ~24 checks ran today")
    if restarts:
        problems.append(f"restarted itself {restarts}x after a problem")
    if not st.get("guard_ok", True):
        problems.append("SAFETY: " + st.get("guard_reason", "account check failed"))
    refused = st.get("refused", {}).get(day, [])
    if refused:
        problems.append(f"Alpaca refused {len(refused)} order(s) ({', '.join(sorted(set(refused))[:6])}); skipped them")

    t9 = sip_today(day)
    if t9.get("mode") == "trade":
        pnl9 = live.own_day_pnl(c, day, sip_mine)
        hist["sip"][day] = round(pnl9, 2)
    out = []
    out.append(f"💰 **Whole account** (both bots): today {_fmt(acct_day_pct, day_usd)} · "
               f"since start {_fmt(tot_pct, tot_usd)}")
    tot8 = sum(hist["dt"].values())
    out.append(f"## 🤖 Bot #8 - winners vs. losers (budget ${b8:,.0f})\n\n"
               f"Today {_fmt(day_pct, pnl8)} · since start {'+' if tot8 >= 0 else '-'}${abs(tot8):,.2f}")
    out.append(("✅ Ran all day" if not problems else "⚠️ " + "; ".join(problems))
               + f" ({cycles} checks)")
    out.append("🧪 Mode: " + ("**TRADING (paper)** - its honesty test passed" if trading else
                             "**WATCH ONLY** - " + (
                                 "paused itself: live results were worse than its test" if L0.get("paused")
                                 else "safety check failed" if not st.get("guard_ok", True)
                                 else "no model yet" if not bundle
                                 else "no proven edge yet, so no orders")))
    # today's trades: the bot is flat at the end of the day, so money in vs. money out = result
    if orders:
        by = {}
        for o in orders:
            q, p = float(o.filled_qty), float(o.filled_avg_price)
            b = by.setdefault(o.symbol, {"n": 0, "cash": 0.0, "gross": 0.0})
            b["n"] += 1
            b["cash"] += q * p * (1 if o.side.value == "sell" else -1)
            b["gross"] += q * p
        rows = ["| Stock | Orders | Traded | Gain/loss |", "|---|---|---|---|"]
        for sym, b in sorted(by.items(), key=lambda kv: -kv[1]["cash"]):
            dot = "🟢" if b["cash"] > 0 else "🔴" if b["cash"] < 0 else "⚪"
            rows.append(f"| {sym} | {b['n']} | ${b['gross']:,.0f} | {dot} {'+' if b['cash'] >= 0 else '-'}${abs(b['cash']):,.2f} |")
        out.append("🔁 **Today's trades**\n\n" + "\n".join(rows))
    else:
        out.append("🔁 Trades today: none" + (" (watching only)" if not trading else ""))

    if shadow.get("n"):
        out.append(f"🧠 Today's guesses: {shadow['hits']} of {shadow['n']} right "
                   f"({shadow['hits'] / shadow['n']:.0%}), average {shadow['avg_bps']:+.1f} bps per guess "
                   "before costs" + (" - this is what it *would* have done" if not trading else ""))
    lr = L.get("live_hit_rate_pct")
    if lr is not None:
        bt = bundle["stats"]["hit_rate_pct"] if bundle else None
        out.append(f"📚 Learned so far: right {lr}% of {L['live_guesses']} live guesses"
                   + (f" (backtest said {bt}%)" if bt is not None else "")
                   + (f"; real trading cost {L['slippage_bps_per_side']} bps per trade"
                      if L.get("slippage_bps_per_side") is not None else ""))
    try:
        duel = contest(day, bundle, shadow)
    except Exception as e:
        duel = f"🏆 champion vs. challenger check failed (ignored): {str(e)[:80]}"
    if duel:
        out.append(duel)
    if L.get("just_paused"):
        out.append("⏸️ **It paused itself today**: after 20+ trading days, its real results were "
                   "losing and its guesses were wrong more than right. It will only watch until a new "
                   "weekly test passes.")
        L2 = load_learned()          # re-read: the contest above may have just saved changes
        L2.pop("just_paused", None)
        LEARNED.write_text(json.dumps(L2, indent=1))
    if bundle:
        ex = bundle["stats"]
        traded = f", traded on {ex['days_traded']} of them" if "days_traded" in ex else ""
        out.append(f"🧪 Last honesty test ({bundle['trained_at'][:10]}): final exam "
                   f"{ex['total_return_pct']:+.2f}% over {ex['days_tested']} days{traded}, Sharpe {ex['sharpe']} -> "
                   + ("PASSED" if bundle["edge_passed"] else "not good enough to trade"))
    sec9, bad9 = bot9_section(day, t9, hist)
    out.append(sec9)
    problems += bad9
    out.append("Paper money - not financial advice.")

    title = "report " + datetime.fromisoformat(day).strftime("%a %b %d")
    md = "\n\n".join(out)
    config.REPORT_DIR.mkdir(exist_ok=True)
    (config.REPORT_DIR / f"{day}.md").write_text(f"# Day-trader {title}\n\n{md}\n", encoding="utf-8")
    (config.REPORT_DIR / "latest.md").write_text(f"# Day-trader {title}\n\n{md}\n", encoding="utf-8")
    return title, md, bool(problems)
