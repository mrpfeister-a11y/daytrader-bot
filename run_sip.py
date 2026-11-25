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


def main():
    setup_logging()
    mode = sys.argv[1] if len(sys.argv) > 1 else "session"
    if mode == "push":
        try:
            mode = (config.ROOT / "requests" / "sip.txt").read_text(encoding="utf-8-sig").split()[0]
        except (OSError, IndexError):
            mode = "check"
    log.info("run_sip.py %s", mode)
    {"session": session, "recover": recover, "chain": chain, "check": check}.get(mode, check)()


if __name__ == "__main__":
    main()
