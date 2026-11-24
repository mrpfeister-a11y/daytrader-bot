"""Everything that talks to the Alpaca account while trading.

Isolation rules (so it can never mess with the other bots):
  * Refuses to trade in the other bots' account (fingerprint check).
  * Refuses to trade if the account holds anything outside its own 20 markets.
  * Only ever buys/sells/closes its own 20 markets - never "close everything".
"""
import csv
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from . import config, data, features, model

log = logging.getLogger("bot")
NY = ZoneInfo("America/New_York")
STATE = config.LOG_DIR / "state.json"


# ---------------------------------------------------------------- state ----
def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def save_state(st: dict):
    config.LOG_DIR.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1, default=str))


def today_ny() -> str:
    return datetime.now(NY).date().isoformat()


# --------------------------------------------------------------- client ----
def client():
    if not (config.ALPACA_KEY and config.ALPACA_SECRET):
        raise SystemExit("No Alpaca keys (GitHub secrets ALPACA_API_KEY / ALPACA_SECRET_KEY).")
    if not config.PAPER and not config.LIVE_CONFIRMED:
        raise SystemExit("ALPACA_PAPER=false but I_UNDERSTAND_LIVE_TRADING_RISK is not 'yes'. Refusing.")
    from alpaca.trading.client import TradingClient
    return TradingClient(config.ALPACA_KEY, config.ALPACA_SECRET, paper=config.PAPER)


def guard(c) -> tuple:
    """(ok, reason). ok=False means: do not place any order in this account."""
    acct = c.get_account()
    if config.account_fingerprint(acct.account_number) in config.BLOCKED_ACCOUNT_HASHES:
        return False, ("these keys belong to the account of your other 7 bots. This bot sells "
                       "everything it holds every afternoon, so it will NOT trade there. "
                       "Use keys from the separate Alpaca paper account.")
    if not config.PAPER:
        return False, "not a paper account"
    foreign = sorted(p.symbol for p in c.get_all_positions() if p.symbol not in config.UNIVERSE)
    if foreign:
        return False, ("the account holds stocks this bot never trades (" + ", ".join(foreign[:8]) +
                       "). It looks shared with another bot, so this bot won't trade in it.")
    return True, ""


def own_positions(c) -> dict:
    return {p.symbol: int(float(p.qty)) for p in c.get_all_positions() if p.symbol in config.UNIVERSE}


def close_own(c, reason: str):
    """Sell/cover only this bot's own markets (never anything else in the account)."""
    for sym, qty in own_positions(c).items():
        try:
            c.close_position(sym)
            record_trade(sym, "close", qty, None, reason)
            log.info("  closed %s (%d) - %s", sym, qty, reason)
        except Exception as e:
            log.warning("  could not close %s: %s", sym, e)


def cancel_own_orders(c):
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    for o in c.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=config.UNIVERSE)):
        try:
            c.cancel_order_by_id(o.id)
        except Exception:
            pass


# ------------------------------------------------------------- journals ----
def _append(name, header, row):
    path = config.LOG_DIR / name
    config.LOG_DIR.mkdir(exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def record_trade(sym, side, qty, price, reason, order_id=""):
    _append("trades.csv", ["time_utc", "symbol", "side", "qty", "decision_price", "reason", "order_id"],
            [datetime.now(timezone.utc).isoformat(), sym, side, qty, price or "", reason, order_id])


def record_decisions(latest, mode):
    for sym, ts, p, sig, price in zip(latest["symbol"], latest["ts"], latest["prob"],
                                      latest["signal"], latest["close"]):
        _append("decisions.csv", ["bar_ts", "symbol", "prob", "signal", "price", "mode"],
                [ts.isoformat(), sym, round(float(p), 4), int(sig), round(float(price), 4), mode])


# --------------------------------------------------------------- decide ----
def latest_features(universe):
    bars = data.from_alpaca(universe, days=45)   # 45 days: enough for the 20-day signals
    now = datetime.now(timezone.utc)
    finished = bars["ts"] + timedelta(minutes=config.BAR_MINUTES) <= now
    feats = features.build(bars[finished])
    latest = feats[feats["ts"] == feats["ts"].max()].copy()
    return latest.dropna(subset=["close"])


def decide(latest, bundle) -> None:
    latest["prob"] = bundle["model"].predict_proba(latest[model.FEATURES])[:, 1]
    latest["signal"] = model.positions(latest, bundle["style"], bundle["entry_prob"])


def targets(latest, equity: float, st: dict, horizon: int = config.HORIZON_BARS) -> dict:
    """{symbol: target shares}. Keeps a trade for the full hour it predicted (same as the
    backtest) unless the opposite signal appears; everything flat from 3:30 PM ET."""
    bar_ts = latest["ts"].max()
    late = float(latest["time_of_day"].max()) >= config.NO_NEW_TRADES_TOD
    entries = st.setdefault("entries", {})
    want = {}
    for sym, sig in zip(latest["symbol"], latest["signal"]):
        sig = int(sig)
        e = entries.get(sym)
        if sig == 0 and e and not late:
            age = (bar_ts - datetime.fromisoformat(e["bar"])) / timedelta(minutes=config.BAR_MINUTES)
            if age < horizon:
                sig = e["side"]           # still inside its hour: keep holding
        want[sym] = 0 if late else sig
    n = sum(1 for s in want.values() if s)
    per = config.POSITION_PCT if not n or n * config.POSITION_PCT <= config.MAX_GROSS_PCT \
        else config.MAX_GROSS_PCT / n
    price = dict(zip(latest["symbol"], latest["close"]))
    return {s: int(np.sign(v)) * math.floor(equity * per / float(price[s])) for s, v in want.items()}


def rebalance(c, tgt: dict, latest, st: dict, watch_only: bool):
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    held = own_positions(c)
    price = dict(zip(latest["symbol"], latest["close"]))
    bar = latest["ts"].max().isoformat()
    entries = st.setdefault("entries", {})
    for sym in sorted(set(tgt) | set(held)):
        want, have = tgt.get(sym, 0), held.get(sym, 0)
        # Only act when the DIRECTION changes (same rule as the backtest - keeps costs down).
        if np.sign(want) == np.sign(have):
            continue
        if watch_only:
            log.info("  [watch] would move %s from %d to %d shares", sym, have, want)
            if want:
                entries[sym] = {"side": int(np.sign(want)), "bar": bar}
            else:
                entries.pop(sym, None)
            continue
        try:   # one refused order must not stop the others (or the whole day)
            if have != 0:
                c.close_position(sym)
                record_trade(sym, "close", have, price.get(sym), "signal changed")
                entries.pop(sym, None)
                log.info("  closed %s (%d)", sym, have)
            if want < 0:
                asset = c.get_asset(sym)
                if not (asset.shortable and asset.easy_to_borrow):
                    log.info("  %s not shortable right now, staying flat", sym)
                    continue
            if want != 0:
                side = OrderSide.BUY if want > 0 else OrderSide.SELL
                o = c.submit_order(MarketOrderRequest(symbol=sym, qty=abs(want), side=side,
                                                      time_in_force=TimeInForce.DAY))
                record_trade(sym, side.value, abs(want), price.get(sym), "new signal", o.id)
                entries[sym] = {"side": int(np.sign(want)), "bar": bar}
                log.info("  %s %d %s", side.value.upper(), abs(want), sym)
        except Exception as e:
            log.warning("  order for %s refused, skipping it this time: %s", sym, str(e)[:150])
            st.setdefault("refused", {}).setdefault(today_ny(), []).append(sym)


def cycle(c, bundle, st: dict, watch_only: bool) -> str:
    acct = c.get_account()
    equity, start = float(acct.equity), float(acct.last_equity)
    day_pnl = equity / start - 1 if start else 0.0
    log.info("Equity $%.2f  (today %+.2f%%)", equity, day_pnl * 100)
    st.setdefault("start_equity", equity)

    if day_pnl <= -config.DAILY_LOSS_LIMIT_PCT:
        log.warning("Daily loss limit hit. Closing its positions, done for today.")
        if not watch_only:
            cancel_own_orders(c)
            close_own(c, "daily loss limit")
        st["halted"] = today_ny()
        return "halted"

    latest = latest_features(bundle["universe"])
    decide(latest, bundle)
    log.info("Bar %s, %d markets, signals: %s", latest["ts"].max(), len(latest),
             {s: int(v) for s, v in zip(latest["symbol"], latest["signal"]) if v} or "none")
    record_decisions(latest, "watch" if watch_only else "trade")
    try:   # this week's challenger model guesses too (practice only, never trades)
        ch = model.load_challenger()
        if ch:
            lc = latest.copy()
            decide(lc, ch)
            record_decisions(lc, "challenger")
    except Exception as e:
        log.warning("challenger guess failed (ignored): %s", e)
    rebalance(c, targets(latest, equity, st, bundle["horizon"]), latest, st, watch_only)
    st["last_cycle"] = datetime.now(timezone.utc).isoformat()
    st.setdefault("cycles", {})
    st["cycles"][today_ny()] = st["cycles"].get(today_ny(), 0) + 1
    return "ok"
