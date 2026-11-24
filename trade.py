"""The bot. Runs all day, every trading day, on Alpaca.

    python trade.py           # run forever (paper account by default)
    python trade.py --once    # run one decision cycle and exit (for testing)
    python trade.py --watch   # never place orders, only log what it would do

Safety rules built in:
  * Paper account unless you change TWO settings in .env.
  * Only trades if the model passed the walk-forward edge test.
  * Max 5% of the account per stock, 50% total.
  * Stops for the day after a 2% daily loss.
  * Closes everything 10 minutes before the market closes. Never holds overnight.
  * Retrains itself every 7 days on fresh data, and re-tests the edge.
"""
import argparse
import csv
import logging
import math
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from bot import config, data, features, model

log = logging.getLogger("bot")


def setup_logging():
    config.LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(config.LOG_DIR / "bot.log")])


def safety_check():
    if not (config.ALPACA_KEY and config.ALPACA_SECRET):
        raise SystemExit("No Alpaca keys. Copy .env.example to .env and fill in your PAPER keys.")
    if not config.PAPER and not config.LIVE_CONFIRMED:
        raise SystemExit("ALPACA_PAPER=false but I_UNDERSTAND_LIVE_TRADING_RISK is not 'yes'. Refusing to trade real money.")
    log.info("Mode: %s", "PAPER (fake money)" if config.PAPER else "*** LIVE - REAL MONEY ***")


def model_is_stale(bundle) -> bool:
    if bundle is None:
        return True
    age = datetime.now(timezone.utc) - datetime.fromisoformat(bundle["trained_at"])
    return age > timedelta(days=config.RETRAIN_DAYS)


def retrain():
    import train
    log.info("Retraining on fresh data and re-testing the edge...")
    return train.run(days=365)


# --------------------------------------------------------------------------
# Decision: pure function, easy to test.
# --------------------------------------------------------------------------
def decide(latest, bundle, equity: float) -> dict:
    """latest: one row per symbol (most recent finished bar). Returns
    {symbol: target_shares} where negative means short."""
    prob = bundle["model"].predict_proba(latest[features.FEATURES])[:, 1]
    pos = model.signals(prob)
    # flat for the last hour-ish so we don't open trades we must dump at the bell
    pos = np.where(latest["time_of_day"].values >= 0.95, 0, pos)

    n_active = int((pos != 0).sum())
    per = config.POSITION_PCT
    if n_active and n_active * per > config.MAX_GROSS_PCT:
        per = config.MAX_GROSS_PCT / n_active

    targets = {}
    for sym, p, price, pr in zip(latest["symbol"], pos, latest["close"], prob):
        shares = math.floor(equity * per / price) if p else 0
        targets[sym] = int(p * shares)
        log.info("  %-5s p(up)=%.3f -> %s", sym, pr,
                 {1: "LONG", -1: "SHORT", 0: "flat"}[int(p)])
    return targets


# --------------------------------------------------------------------------
# Talking to Alpaca
# --------------------------------------------------------------------------
def current_positions(client) -> dict:
    return {p.symbol: int(float(p.qty)) for p in client.get_all_positions()}


def record(sym, side, qty, reason):
    path = config.LOG_DIR / "trades.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time_utc", "symbol", "side", "qty", "reason"])
        w.writerow([datetime.now(timezone.utc).isoformat(), sym, side, qty, reason])


def rebalance(client, targets: dict, watch_only: bool):
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    held = current_positions(client)
    for sym in sorted(set(targets) | set(held)):
        want, have = targets.get(sym, 0), held.get(sym, 0)
        # Only act when the DIRECTION changes. Small size drift is ignored,
        # which keeps trading costs down (same rule as the backtest).
        if np.sign(want) == np.sign(have):
            continue
        if watch_only:
            log.info("  [watch] would move %s from %d to %d shares", sym, have, want)
            continue
        if have != 0:
            client.close_position(sym)
            record(sym, "close", have, "signal changed")
            log.info("  closed %s (%d)", sym, have)
        if want < 0:
            asset = client.get_asset(sym)
            if not (asset.shortable and asset.easy_to_borrow):
                log.info("  %s not shortable right now, staying flat", sym)
                continue
        if want != 0:
            side = OrderSide.BUY if want > 0 else OrderSide.SELL
            client.submit_order(MarketOrderRequest(
                symbol=sym, qty=abs(want), side=side, time_in_force=TimeInForce.DAY))
            record(sym, side.value, abs(want), "new signal")
            log.info("  %s %d %s", side.value.upper(), abs(want), sym)


def run_cycle(client, bundle, watch_only: bool):
    acct = client.get_account()
    equity, start = float(acct.equity), float(acct.last_equity)
    day_pnl = equity / start - 1
    log.info("Equity $%.2f  (today %+.2f%%)", equity, day_pnl * 100)

    if day_pnl <= -config.DAILY_LOSS_LIMIT_PCT:
        log.warning("Daily loss limit hit. Closing everything, done for today.")
        if not watch_only:
            client.close_all_positions(cancel_orders=True)
        return "halted"

    bars = data.from_alpaca(bundle["universe"], days=10)
    now = datetime.now(timezone.utc)
    finished = bars["ts"] + timedelta(minutes=config.BAR_MINUTES) <= now
    feats = features.build(bars[finished])
    latest = feats[feats["ts"] == feats["ts"].max()]
    latest = latest.dropna(subset=features.FEATURES, how="all")
    log.info("Bar %s, %d symbols", latest["ts"].max(), len(latest))

    targets = decide(latest, bundle, equity)
    if not bundle["edge_passed"]:
        log.info("Model has NO proven edge -> watching only, no orders.")
        watch_only = True
    rebalance(client, targets, watch_only)
    return "ok"


def sleep_until(t: datetime, cap_s: float = 900):
    secs = (t - datetime.now(timezone.utc)).total_seconds()
    if secs > 0:
        time.sleep(min(secs, cap_s))


def next_bar_time() -> datetime:
    now = datetime.now(timezone.utc)
    m = config.BAR_MINUTES
    nxt = now.replace(second=0, microsecond=0) + timedelta(minutes=m - now.minute % m)
    return nxt + timedelta(seconds=30)  # let the bar finish and arrive


def cloud_run(client, watch_only: bool):
    """One stateless pass. A scheduler (GitHub Actions) calls this every 15
    minutes. Cloud schedulers can start runs late, so the flatten window is
    wider than when running on your own computer."""
    clock = client.get_clock()
    bundle = model.load()
    if not clock.is_open:
        log.info("Market closed. Next open %s", clock.next_open)
        if model_is_stale(bundle):
            retrain()
        return
    if client.get_all_positions() and \
            clock.next_close - clock.timestamp <= timedelta(minutes=config.CLOUD_FLATTEN_MINUTES):
        log.info("Within %d min of the close: flattening everything.", config.CLOUD_FLATTEN_MINUTES)
        if not watch_only:
            client.close_all_positions(cancel_orders=True)
        return
    if clock.next_close - clock.timestamp <= timedelta(minutes=config.CLOUD_FLATTEN_MINUTES):
        log.info("Too close to the close to open new trades.")
        return
    if model_is_stale(bundle):
        bundle = retrain()
    log.info("Model edge test: %s, stats: %s",
             "PASSED" if bundle["edge_passed"] else "FAILED", bundle["stats"])
    run_cycle(client, bundle, watch_only)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--cloud", action="store_true",
                   help="one check-and-trade, then exit (for scheduled cloud runs)")
    args = p.parse_args()

    setup_logging()
    safety_check()
    from alpaca.trading.client import TradingClient
    client = TradingClient(config.ALPACA_KEY, config.ALPACA_SECRET, paper=config.PAPER)

    if args.cloud:
        cloud_run(client, args.watch)
        return

    bundle = model.load()
    if model_is_stale(bundle):
        bundle = retrain()
    log.info("Model trained %s, edge test: %s, stats: %s",
             bundle["trained_at"][:16], "PASSED" if bundle["edge_passed"] else "FAILED",
             bundle["stats"])

    if args.once:
        run_cycle(client, bundle, args.watch)
        return

    halted_day = None
    while True:
        try:
            clock = client.get_clock()
            if not clock.is_open:
                if model_is_stale(bundle):
                    bundle = retrain()
                log.info("Market closed. Next open %s", clock.next_open)
                sleep_until(clock.next_open)
                continue

            today = clock.timestamp.date()
            to_close = clock.next_close - clock.timestamp
            if to_close <= timedelta(minutes=config.FLATTEN_MINUTES_BEFORE_CLOSE):
                log.info("Near the close: flattening all positions.")
                if not args.watch:
                    client.close_all_positions(cancel_orders=True)
                sleep_until(clock.next_close + timedelta(minutes=1))
                continue
            if halted_day == today:
                sleep_until(clock.next_close - timedelta(minutes=config.FLATTEN_MINUTES_BEFORE_CLOSE))
                continue

            if run_cycle(client, bundle, args.watch) == "halted":
                halted_day = today
            sleep_until(min(next_bar_time(),
                            clock.next_close - timedelta(minutes=config.FLATTEN_MINUTES_BEFORE_CLOSE)))
        except KeyboardInterrupt:
            log.info("Stopped by you. Open positions (if any) are left as they are.")
            break
        except Exception as e:  # network blips etc. - keep going
            log.exception("Error: %s. Retrying in 60s.", e)
            time.sleep(60)


if __name__ == "__main__":
    main()
