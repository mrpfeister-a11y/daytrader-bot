"""Bot #9: stocks-in-play opening-range breakout (Zarattini & Aziz 2023), live.

Every trading day:
  1. Before the open: list all US stocks with price > $5, 14-day avg volume > 1M shares,
     14-day ATR > $0.50 - EXCEPT bot #8's 20 markets (so the two bots never share a stock).
  2. 9:35 ET: the 20 with the most unusual volume in the first 5 minutes ("in play").
     First 5-min candle up -> buy-stop at its high; down -> sell-stop at its low.
  3. When an entry fills, a stop-loss goes in 10% of the stock's daily range (ATR) away.
  4. 3:45 ET: cancel what's left and sell/cover everything it holds. Never overnight.
Sizing: risk 1% of its budget per trade; total capped at SIP_MAX_LEVERAGE x its budget
(1 = no borrowing). Before SIP_TRADE_FROM it only practices (no orders) and records what
would have happened, so it learns before risking anything.

Learning from its mistakes (logs/sip_learned.json, logs/sip_trades.csv):
  * every trade (practice or real) is logged with the slippage it really got;
  * the monthly re-test (research_orb.retune) uses the real slippage and may switch to
    better settings (how many stocks, how tight the stop) - only if they also win on the
    last year it never used to choose;
  * after 20 real trading days with a loss, it goes back to practice until a re-test passes.
"""
import csv
import json
import logging
import math
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import config, live

log = logging.getLogger("bot")
NY = live.NY
STATE = config.LOG_DIR / "sip_state.json"
TRADES = config.LOG_DIR / "sip_trades.csv"
LEARNED = config.LOG_DIR / "sip_learned.json"
TODAY = config.LOG_DIR / "sip_today.json"
ERR = config.LOG_DIR / "sip_last_error.txt"
CHAIN = config.LOG_DIR / "sip_chain.flag"
SLIP_BPS_DEFAULT = 5.0


def mine(sym: str) -> bool:
    return sym not in config.UNIVERSE


def load_json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def save_json(path, obj):
    config.LOG_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


def params() -> dict:
    p = {"top_n": config.SIP_TOP_N, "stop_atr": config.SIP_STOP_ATR}
    p.update({k: v for k, v in load_json(config.SIP_PARAMS, {}).items() if k in p})
    return p


def practicing(day: str) -> bool:
    return day < config.SIP_TRADE_FROM or bool(load_json(LEARNED, {}).get("paused"))


# ------------------------------------------------------------------ data ----
def _data():
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(config.ALPACA_KEY, config.ALPACA_SECRET)


def _bars(symbols, start, end, minutes=None, feed="iex", adjustment="split") -> pd.DataFrame:
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    tf = TimeFrame.Day if minutes is None else TimeFrame(minutes, TimeFrameUnit.Minute)
    frames = []
    for i in range(0, len(symbols), 400):
        for attempt in range(3):
            try:
                df = _data().get_stock_bars(StockBarsRequest(
                    symbol_or_symbols=list(symbols[i:i + 400]), timeframe=tf, start=start, end=end,
                    feed=DataFeed.SIP if feed == "sip" else DataFeed.IEX, adjustment=adjustment)).df
                if not df.empty:
                    frames.append(df.reset_index())
                break
            except Exception as e:
                log.warning("  price download retry %d: %s", attempt + 1, str(e)[:120])
                time.sleep(5 * (attempt + 1))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def stock_list(c) -> dict:
    """{symbol: (shortable and easy to borrow)} for tradable US stocks, minus bot #8's markets."""
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest
    assets = c.get_all_assets(GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY))
    ok_ex = {"NYSE", "NASDAQ", "AMEX"}
    return {a.symbol: bool(a.shortable and a.easy_to_borrow) for a in assets
            if a.tradable and str(getattr(a.exchange, "value", a.exchange)) in ok_ex
            and a.symbol.isalpha() and len(a.symbol) <= 5 and mine(a.symbol)}


def candidates(c, day: str) -> pd.DataFrame:
    """Stocks passing the paper's filters, from daily bars up to yesterday."""
    stocks = stock_list(c)
    start = live.day_start_ny(day) - timedelta(days=40)
    d = _bars(sorted(stocks), start, live.day_start_ny(day), feed="sip")
    if d.empty:
        return pd.DataFrame()
    d = d.sort_values(["symbol", "timestamp"])
    g = d.groupby("symbol")
    pc = g["close"].shift(1)
    tr = np.maximum(d["high"] - d["low"], np.maximum((d["high"] - pc).abs(), (d["low"] - pc).abs()))
    d["tr"] = tr
    last = d.groupby("symbol").tail(14)
    s = last.groupby("symbol").agg(atr=("tr", "mean"), avgvol=("volume", "mean"), n=("tr", "size"))
    s["prev_close"] = d.groupby("symbol")["close"].last()
    s = s[(s["n"] >= 14) & (s["prev_close"] > config.SIP_MIN_PRICE) & (s["avgvol"] > config.SIP_MIN_AVG_VOL)
          & (s["atr"] > config.SIP_MIN_ATR)].copy()
    s["easy_short"] = [stocks.get(x, False) for x in s.index]
    s["days"] = [sorted(d["timestamp"].dt.tz_convert(NY).dt.date.astype(str).unique())[-14:]] * len(s)
    log.info("Bot #9: %d candidate stocks (of %d)", len(s), len(stocks))
    return s


def _open_bar(symbols, day: str) -> pd.DataFrame:
    o = live.day_start_ny(day) + timedelta(hours=9, minutes=30)
    b = _bars(symbols, o, o + timedelta(minutes=5), minutes=5)
    if b.empty:
        return b
    b = b[b["timestamp"] == pd.Timestamp(o).tz_convert("UTC")]
    return b.set_index("symbol")


def opening_volume_avg(cands: pd.DataFrame) -> pd.Series:
    """Average first-5-minute volume over the last 14 days (same data feed as today's)."""
    syms = list(cands.index)
    vols = []
    for d in cands["days"].iloc[0] if len(cands) else []:
        b = _open_bar(syms, d)
        if not b.empty:
            vols.append(b["volume"].rename(d))
    if not vols:
        return pd.Series(dtype=float)
    v = pd.concat(vols, axis=1)
    return v.mean(axis=1, skipna=True)[v.notna().sum(axis=1) >= 10]


def pick(cands, avg_vol, day, top_n) -> pd.DataFrame:
    ob = _open_bar(list(cands.index), day)
    if ob.empty:
        return pd.DataFrame()
    m = cands.join(ob[["open", "high", "low", "close", "volume"]], how="inner")
    m["rvol"] = m["volume"] / avg_vol.reindex(m.index)
    m = m[(m["rvol"] >= 1) & (m["close"] != m["open"])].sort_values("rvol", ascending=False)
    m["side"] = np.where(m["close"] > m["open"], 1, -1)
    m = m[(m["side"] == 1) | m["easy_short"]]               # shorts only if easy to borrow
    return m.head(top_n)


def plan_sizes(picks: pd.DataFrame, budget: float, stop_atr: float) -> pd.Series:
    level = np.where(picks["side"] == 1, picks["high"], picks["low"])
    risk_sh = config.SIP_RISK_PCT * budget / (stop_atr * picks["atr"])
    cap_sh = config.SIP_MAX_POSITION_PCT * budget * config.SIP_MAX_LEVERAGE / level
    sh = np.minimum(risk_sh, cap_sh)
    total = float((sh * level).sum())
    cap = config.SIP_MAX_LEVERAGE * budget
    if total > cap:
        sh = sh * cap / total
    return np.floor(sh).astype(int)


# --------------------------------------------------------------- orders ----
def place_entries(c, picks, sizes, st):
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import StopOrderRequest
    for (sym, r), qty in zip(picks.iterrows(), sizes):
        if qty < 1:
            continue
        level = round(float(r["high"] if r["side"] == 1 else r["low"]), 2)
        side = OrderSide.BUY if r["side"] == 1 else OrderSide.SELL
        try:
            o = c.submit_order(StopOrderRequest(symbol=sym, qty=int(qty), side=side,
                                                time_in_force=TimeInForce.DAY, stop_price=level))
            st["orders"][sym] = {"side": int(r["side"]), "qty": int(qty), "level": level,
                                 "atr": float(r["atr"]), "entry_id": str(o.id), "stop_id": None,
                                 "fill": None}
            log.info("  bot #9 %s-stop %d %s at %.2f", side.value, qty, sym, level)
        except Exception as e:
            log.warning("  bot #9 entry for %s refused: %s", sym, str(e)[:150])
            st.setdefault("refused", []).append(sym)


def manage(c, st, stop_atr):
    """Entry filled -> put in its stop-loss."""
    from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
    from alpaca.trading.requests import GetOrdersRequest, StopOrderRequest
    if not st["orders"]:
        return
    orders = {str(o.id): o for o in c.get_orders(GetOrdersRequest(
        status=QueryOrderStatus.ALL, after=live.day_start_ny(st["day"]), limit=500))}
    for sym, t in st["orders"].items():
        o = orders.get(t["entry_id"])
        if t["stop_id"] or o is None or o.status.value != "filled":
            continue
        fill = float(o.filled_avg_price)
        qty = int(float(o.filled_qty))
        stop = round(fill - stop_atr * t["atr"] if t["side"] == 1 else fill + stop_atr * t["atr"], 2)
        side = OrderSide.SELL if t["side"] == 1 else OrderSide.BUY
        try:
            so = c.submit_order(StopOrderRequest(symbol=sym, qty=qty, side=side,
                                                 time_in_force=TimeInForce.DAY, stop_price=stop))
            t.update(stop_id=str(so.id), fill=fill, stop=stop, filled_qty=qty)
            log.info("  bot #9 %s filled at %.2f, stop-loss at %.2f", sym, fill, stop)
        except Exception as e:
            log.warning("  bot #9 stop-loss for %s refused (will retry): %s", sym, str(e)[:150])


def flatten(c, reason="end of day"):
    """Cancel bot #9's open orders and close bot #9's positions (never bot #8's 20 markets)."""
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    for o in c.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)):
        if mine(o.symbol):
            try:
                c.cancel_order_by_id(o.id)
            except Exception:
                pass
    time.sleep(2)
    for p in c.get_all_positions():
        if mine(p.symbol):
            try:
                c.close_position(p.symbol)
                log.info("  bot #9 closed %s (%s) - %s", p.symbol, p.qty, reason)
            except Exception as e:
                log.warning("  bot #9 could not close %s: %s", p.symbol, e)


# ------------------------------------------------- practice / results ----
def simulate(picks: pd.DataFrame, sizes, day: str, stop_atr: float) -> list:
    """What the day's picks did, minute by minute (practice mode), until 3:45 PM."""
    if picks.empty:
        return []
    o = live.day_start_ny(day) + timedelta(hours=9, minutes=35)
    b = _bars(list(picks.index), o, o + timedelta(hours=6, minutes=10), minutes=1)
    out = []
    for (sym, r), qty in zip(picks.iterrows(), sizes):
        x = b[b["symbol"] == sym] if not b.empty else b
        if qty < 1 or x.empty:
            continue
        side = int(r["side"])
        level = float(r["high"] if side == 1 else r["low"])
        hit = (x["high"] > level) if side == 1 else (x["low"] < level)
        if not hit.any():
            out.append({"symbol": sym, "side": side, "level": level, "triggered": False, "pnl": 0.0})
            continue
        j = int(np.argmax(hit.to_numpy()))
        fill = max(level, float(x["open"].iloc[j])) if side == 1 else min(level, float(x["open"].iloc[j]))
        stop = fill - stop_atr * r["atr"] if side == 1 else fill + stop_atr * r["atr"]
        c0 = float(x["close"].iloc[j])
        rest = x.iloc[j + 1:]
        hit_stop = (c0 <= stop if side == 1 else c0 >= stop) or \
            ((rest["low"] <= stop).any() if side == 1 else (rest["high"] >= stop).any())
        exit_px = stop if hit_stop else float(x["close"].iloc[-1])
        slip = SLIP_BPS_DEFAULT / 1e4 * (fill + exit_px) * qty
        out.append({"symbol": sym, "side": side, "level": level, "triggered": True, "fill": round(fill, 4),
                    "exit": round(exit_px, 4), "stopped": bool(hit_stop), "qty": int(qty),
                    "pnl": round(side * (exit_px - fill) * qty - slip, 2)})
    return out


def real_results(c, st) -> list:
    """Per-stock results of today's real trades (from Alpaca fills)."""
    by = {}
    for o in live.fills_today(c, st["day"]):
        if not mine(o.symbol):
            continue
        q, p = float(o.filled_qty), float(o.filled_avg_price)
        b = by.setdefault(o.symbol, {"symbol": o.symbol, "cash": 0.0, "bought": 0.0, "sold": 0.0})
        b["cash"] += q * p if o.side.value == "sell" else -q * p
    out = []
    for sym, b in by.items():
        t = st["orders"].get(sym, {})
        slip = None
        if t.get("fill") and t.get("level"):
            slip = round(t["side"] * (t["fill"] / t["level"] - 1) * 1e4, 2)    # + = paid more than planned
        out.append({"symbol": sym, "side": t.get("side"), "level": t.get("level"), "fill": t.get("fill"),
                    "triggered": True, "pnl": round(b["cash"], 2), "slippage_bps": slip})
    return out


def learn(day: str, mode: str, results: list, budget_amt: float):
    new = not TRADES.exists()
    with TRADES.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["day", "mode", "symbol", "side", "level", "fill", "exit", "pnl", "slippage_bps"])
        for r in results:
            if r.get("triggered"):
                w.writerow([day, mode, r["symbol"], r["side"], r.get("level"), r.get("fill"),
                            r.get("exit", ""), r["pnl"], r.get("slippage_bps", "")])
    L = load_json(LEARNED, {})
    days = L.setdefault("days", {})
    pnl = round(sum(r["pnl"] for r in results), 2)
    days[day] = {"mode": mode, "pnl": pnl, "pct": round(pnl / budget_amt * 100, 3) if budget_amt else 0,
                 "trades": sum(1 for r in results if r.get("triggered")),
                 "won": sum(1 for r in results if r.get("pnl", 0) > 0)}
    real = [v for k, v in days.items() if v["mode"] == "trade" and k >= L.get("review_from", "")]
    slips = []
    if TRADES.exists():
        for r in csv.DictReader(TRADES.open()):
            if r["slippage_bps"] not in ("", None):
                slips.append(float(r["slippage_bps"]))
    L["slippage_bps"] = round(float(np.median(slips)), 2) if slips else None
    L["real_days"] = len(real)
    L["real_total_pct"] = round(sum(v["pct"] for v in real), 2)
    if len(real) >= config.LIVE_REVIEW_MIN_DAYS and L["real_total_pct"] < 0 and not L.get("paused"):
        L.update(paused=True, paused_on=day, just_paused=True)
    save_json(LEARNED, L)
    return L


# ---------------------------------------------------------------- session ----
def session(watch_flag=False):
    c = live.client()
    st = load_json(STATE, {})
    day = live.today_ny()
    if st.get("day") != day:
        st = {"day": day, "orders": {}, "picked": False, "done": False, "restarts": st.get("restarts", {})}
    ok, why = live.guard(c)
    p = params()
    practice = watch_flag or practicing(day) or not ok
    log.info("Bot #9 (stocks in play): %s, top %d, stop %.0f%% of ATR, max %.1fx of its budget",
             "PRACTICE (no orders)" if practice else "PAPER TRADING", p["top_n"], p["stop_atr"] * 100,
             config.SIP_MAX_LEVERAGE)
    t0 = time.monotonic()
    cands = avg_vol = None
    while True:
        clock = c.get_clock()
        now = clock.timestamp
        if not clock.is_open:
            today_open = clock.next_open.astimezone(NY).date().isoformat() == day
            if today_open and clock.next_open - now < timedelta(minutes=100):
                if cands is None:
                    cands = candidates(c, day)
                    avg_vol = opening_volume_avg(cands) if len(cands) else pd.Series(dtype=float)
                    log.info("Bot #9: opening-volume history for %d stocks", len(avg_vol))
                time.sleep(min(max((clock.next_open - now).total_seconds(), 5), 600))
                continue
            break
        if st["done"]:
            break
        ny = now.astimezone(NY)
        if cands is None and not st["picked"]:  # started late: prepare now
            cands = candidates(c, day)
            avg_vol = opening_volume_avg(cands) if len(cands) else pd.Series(dtype=float)
        if not st["picked"] and ny >= ny.replace(hour=9, minute=35, second=15, microsecond=0):
            picks = pick(cands, avg_vol, day, p["top_n"]) if len(cands) else pd.DataFrame()
            b = live.budget(c, "sip")
            sizes = plan_sizes(picks, b, p["stop_atr"]) if len(picks) else []
            st["picks"] = [{"symbol": s, "side": int(r["side"]), "high": float(r["high"]), "low": float(r["low"]),
                            "atr": float(r["atr"]), "rvol": round(float(r["rvol"]), 2), "qty": int(q)}
                           for (s, r), q in zip(picks.iterrows(), sizes)]
            st["budget"] = b
            st["picked"] = True
            log.info("Bot #9 picks: %s", ", ".join(f"{x['symbol']}{'+' if x['side'] == 1 else '-'}"
                                                 for x in st["picks"]) or "none")
            if not practice and len(picks):
                place_entries(c, picks, sizes, st)
            save_json(STATE, st)
        if not practice:
            try:
                manage(c, st, p["stop_atr"])
                pnl = live.own_day_pnl(c, day, mine)
                if st.get("budget") and pnl <= -config.SIP_DAILY_STOP_PCT * st["budget"]:
                    log.warning("Bot #9 daily loss stop hit (%.2f): closing its trades for today.", pnl)
                    flatten(c, "daily loss stop")
                    st.update(done=True, halted=True)
                    save_json(STATE, st)
                    break
            except Exception as e:
                log.warning("Bot #9 manage failed (retrying): %s", str(e)[:150])
        if clock.next_close - now <= timedelta(minutes=config.FLATTEN_MINUTES_BEFORE_CLOSE):
            if not practice:
                flatten(c)
            st["done"] = True
            save_json(STATE, st)
            break
        if (time.monotonic() - t0) / 60 > config.SESSION_MAX_MINUTES:
            CHAIN.write_text("sip")
            save_json(STATE, st)
            return
        save_json(STATE, st)
        time.sleep(20)
    finish(c, st, practice)


def finish(c, st, practice):
    """After its last trade of the day: results + learning (once per day)."""
    if not st.get("picked") or st.get("finished"):
        save_json(STATE, st)
        return
    day = st["day"]
    p = params()
    if practice:
        picks = pd.DataFrame(st["picks"]).set_index("symbol") if st["picks"] else pd.DataFrame()
        sizes = [x["qty"] for x in st["picks"]]
        try:
            results = simulate(picks, sizes, day, p["stop_atr"])
        except Exception as e:
            log.warning("practice results failed: %s", e)
            results = []
    else:
        results = real_results(c, st)
    L = learn(day, "practice" if practice else "trade", results, st.get("budget") or live.budget(c, "sip"))
    save_json(TODAY, {"day": day, "mode": "practice" if practice else "trade", "results": results,
                      "budget": st.get("budget"), "params": p, "leverage": config.SIP_MAX_LEVERAGE,
                      "learned": {k: v for k, v in L.items() if k != "days"},
                      "refused": st.get("refused", [])})
    st["finished"] = True
    save_json(STATE, st)
