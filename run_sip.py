"""What the cloud runs for bot #9 (stocks in play). One command per job:

    python run_sip.py session   # practice or trade the whole day (all decisions inside)
    python run_sip.py recover <status>   # after a crash: restart (max 3 a day) + email
    python run_sip.py chain     # hand a long day to a fresh job
    python run_sip.py check     # can it see the stock list and prices? (email)
    python run_sip.py push      # command from requests/sip.txt (backup starter)

Bot #9 sends no email of its own on a normal day: its results go into the day-trader's
daily report (one email for both bots).
"""
import logging
import sys
import traceback

from bot import config, live, notify, sip

log = logging.getLogger("bot")


def setup_logging():
    config.LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(config.LOG_DIR / "sip.log")])


def session(watch=False):
    try:
        sip.session(watch)
    except Exception:
        sip.ERR.write_text(traceback.format_exc()[-3000:])
        raise


def recover():
    status = sys.argv[2] if len(sys.argv) > 2 else "failure"
    if status == "cancelled":
        print("cancelled by hand - nothing to fix")
        return
    st = sip.load_json(sip.STATE, {})
    day = live.today_ny()
    n = st.setdefault("restarts", {}).get(day, 0)
    err = sip.ERR.read_text()[-1500:] if sip.ERR.exists() else "(no error text - stopped or timed out)"
    try:
        c = live.client()
        market_open = c.get_clock().is_open
    except Exception:
        market_open, c = True, None
    if market_open and n < config.MAX_RESTARTS_PER_DAY:
        st["restarts"][day] = n + 1
        sip.save_json(sip.STATE, st)
        started = notify.start_run("session", workflow="sip.yml")
        notify.send("bot #9 hiccup - " + ("fixed itself" if started else "could NOT restart"),
                    f"Bot #9 (stocks in play) broke and {'restarted' if started else 'could not restart'} "
                    f"(restart {n + 1} of {config.MAX_RESTARTS_PER_DAY} today). Its trades and stop-losses "
                    f"stay in place at Alpaca meanwhile.\n\n```\n{err}\n```", urgent=not started)
    else:
        if c is not None and not market_open:
            try:
                sip.flatten(c, "safety sweep after a crash")   # nothing may stay overnight
            except Exception:
                pass
        notify.send("bot #9 problem - the watchdog will look",
                    f"Bot #9 broke and did not restart ({'market closed' if not market_open else '3 restarts used'}). "
                    f"Bot #8 is not affected.\n\n```\n{err}\n```", urgent=True)


def chain():
    if sip.CHAIN.exists():
        sip.CHAIN.unlink()
        notify.start_run("session", workflow="sip.yml")


def check():
    c = live.client()
    ok, why = live.guard(c)
    stocks = sip.stock_list(c)
    lines = [("✅ Account check OK" if ok else "🛑 " + why),
             f"✅ {len(stocks):,} US stocks it may trade (bot #8's 20 markets excluded)",
             f"Mode on {live.today_ny()}: " + ("PRACTICE (no orders)" if sip.practicing(live.today_ny())
                                               else "PAPER TRADING"),
             f"Settings: {sip.params()}, borrowing up to {config.SIP_MAX_LEVERAGE}x its budget"]
    notify.send("bot #9 health check", "\n\n".join(lines) + "\n\nPaper money - not financial advice.",
                urgent=not ok)


def dryrun():
    """Do today's whole morning scan + pick on real data (no orders, no email) - a test."""
    import time as _t
    c = live.client()
    day = live.today_ny()
    t0 = _t.time()
    cands = sip.candidates(c, day)
    avg = sip.opening_volume_avg(cands)
    picks = sip.pick(cands, avg, day, sip.params()["top_n"])
    sizes = sip.plan_sizes(picks, live.budget(c, "sip"), sip.params()["stop_atr"]) if len(picks) else []
    print(f"DRY RUN {day}: {len(cands)} candidates, {len(avg)} with opening-volume history, "
          f"{len(picks)} picks in {_t.time() - t0:.0f}s")
    for (s, r), q in zip(picks.iterrows(), sizes):
        print(f"  {s:6} {'BUY ' if r['side'] == 1 else 'SHORT'} rvol {r['rvol']:.1f}  level "
              f"{r['high'] if r['side'] == 1 else r['low']:.2f}  atr {r['atr']:.2f}  shares {q}")
    res = sip.simulate(picks, list(sizes), day, sip.params()["stop_atr"]) if len(picks) else []
    print(f"  practice result today: ${sum(x['pnl'] for x in res):,.2f} from "
          f"{sum(1 for x in res if x['triggered'])} triggered trades")


def ordertest():
    """Real orders in the shared paper account that can NEVER fill, then cancelled:
    proves Alpaca accepts both bots' order types and that each bot cleans up only its own."""
    import time as _t
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest, StopOrderRequest
    c = live.client()
    ok, why = live.guard(c)
    if not ok:
        print("SAFETY STOP:", why)
        return
    before = sorted(p.symbol for p in c.get_all_positions())
    print("positions before:", before or "none")
    d = sip._bars(["F"], live.day_start_ny(live.today_ny()) - __import__("datetime").timedelta(days=10),
                  live.day_start_ny(live.today_ny()), feed="sip")
    px = float(d["close"].iloc[-1])
    res = {}
    o9a = c.submit_order(StopOrderRequest(symbol="F", qty=1, side=OrderSide.BUY,
                                          time_in_force=TimeInForce.DAY, stop_price=round(px * 3, 2)))
    res["bot #9 buy-stop (F, 3x price)"] = o9a.id
    try:
        o9b = c.submit_order(StopOrderRequest(symbol="F", qty=1, side=OrderSide.SELL,
                                              time_in_force=TimeInForce.DAY, stop_price=round(px * 0.3, 2)))
        res["bot #9 sell-stop / short (F, 0.3x price)"] = o9b.id
    except Exception as e:
        print("  short-entry stop refused (would be skipped by the bot):", str(e)[:120])
    o8 = c.submit_order(LimitOrderRequest(symbol="SPY", qty=1, side=OrderSide.BUY,
                                          time_in_force=TimeInForce.DAY, limit_price=1.00))
    res["bot #8 order (SPY limit $1)"] = o8.id
    _t.sleep(3)
    for k, i in res.items():
        print(f"  placed {k}: {c.get_order_by_id(i).status.value}")
    sip.flatten(c, "order test")                   # bot #9's clean-up
    _t.sleep(3)
    st8 = c.get_order_by_id(o8.id).status.value
    print(f"  after bot #9 clean-up: bot #8's order is '{st8}' (must still be open)")
    for k, i in res.items():
        if "bot #9" in k:
            print(f"  after bot #9 clean-up: {k} is '{c.get_order_by_id(i).status.value}' (must be canceled)")
    live.cancel_own_orders(c)                      # bot #8's clean-up
    _t.sleep(3)
    print(f"  after bot #8 clean-up: its order is '{c.get_order_by_id(o8.id).status.value}' (must be canceled)")
    after = sorted(p.symbol for p in c.get_all_positions())
    print("positions after:", after or "none", "- unchanged" if after == before else "- CHANGED!")
    day = live.today_ny()
    print(f"  bot #8 result today ${live.own_day_pnl(c, day, live.is_dt):,.2f}, "
          f"bot #9 ${live.own_day_pnl(c, day, sip.mine):,.2f}, budgets ${live.budget(c, 'dt'):,.0f} / "
          f"${live.budget(c, 'sip'):,.0f}")
    print("ORDER TEST DONE")


def main():
    setup_logging()
    mode = sys.argv[1] if len(sys.argv) > 1 else "session"
    if mode == "push":
        try:
            mode = (config.ROOT / "requests" / "sip.txt").read_text(encoding="utf-8-sig").split()[0]
        except (OSError, IndexError):
            mode = "check"
    log.info("run_sip.py %s", mode)
    {"session": session, "recover": recover, "chain": chain, "check": check,
     "dryrun": dryrun, "ordertest": ordertest}.get(mode, check)()


if __name__ == "__main__":
    main()
