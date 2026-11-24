"""What the cloud runs. One command per job:

    python run.py session   # trade (or watch) all day, then send the daily report
    python run.py report    # send today's report if it hasn't gone out yet
    python run.py train     # download all history, re-run the honesty test, email the result
    python run.py check     # health check (keys, account safety, data, model) + test email
    python run.py recover   # after a crash: email what happened and restart itself
    python run.py chain     # hand the trading day over to a fresh job if asked to
    python run.py push      # read the command from requests/run.txt (backup starter)

Self-fixing: each 15-min check retries on errors; a crashed job emails you and starts
a new one (up to 3 times a day); a day longer than one cloud job is handed to a
fresh job; a missed report is sent by the next run; an old/missing model triggers
retraining after the close.
"""
import logging
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone

from bot import config, live, model, notify, review

log = logging.getLogger("bot")
CHAIN = config.LOG_DIR / "chain.flag"
ERR = config.LOG_DIR / "last_error.txt"


def setup_logging():
    config.LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(config.LOG_DIR / "bot.log")])


def model_age_days(bundle) -> float:
    if not bundle:
        return 1e9
    return (datetime.now(timezone.utc) - datetime.fromisoformat(bundle["trained_at"])).total_seconds() / 86400


def is_trading_day(c, day: str) -> bool:
    from alpaca.trading.requests import GetCalendarRequest
    d = datetime.fromisoformat(day).date()
    return bool(c.get_calendar(GetCalendarRequest(start=d, end=d)))


def check_guard(c, st) -> bool:
    ok, why = live.guard(c)
    first_today = st.get("guard_day") != live.today_ny()
    st.update(guard_ok=ok, guard_reason=why, guard_day=live.today_ny())
    if not ok and first_today:
        notify.send("safety stop - not trading", "🛑 " + why +
                    "\n\nNothing was bought or sold. The other 7 bots are not affected.", urgent=True)
    return ok


def after_close(c, st, bundle):
    """Report once per trading day, then retrain if the model is a week old."""
    day = live.today_ny()
    if st.get("report_sent") == day or not is_trading_day(c, day):
        return
    title, md, urgent = review.daily_report(c, bundle, st, day)
    notify.send(title, md, urgent=urgent)
    st["report_sent"] = day
    live.save_state(st)
    if model_age_days(bundle) >= config.RETRAIN_DAYS and st.get("train_started") != day:
        st["train_started"] = day
        notify.start_run("train")


def session(watch_flag=False):
    c = live.client()
    st = live.load_state()
    t0 = time.monotonic()
    bundle = model.load()
    L = review.load_learned()
    safe = check_guard(c, st)
    watch = (watch_flag or not safe or not bundle or not bundle["edge_passed"] or bool(L.get("paused")))
    if not bundle:
        log.info("No model yet - watching only; training starts after the close.")
    else:
        log.info("Model: %s, hold %d min, %.0f%% sure, honesty test %s %s",
                 bundle["style"], bundle["horizon"] * 15, bundle["entry_prob"] * 100,
                 "PASSED" if bundle["edge_passed"] else "FAILED", bundle["stats"])
    log.info("Mode: %s", "WATCH ONLY" if watch else "PAPER TRADING")
    errors = 0
    while True:
        clock = c.get_clock()
        now = clock.timestamp
        if not clock.is_open:
            until_open = clock.next_open - now
            if clock.next_open.astimezone(live.NY).date().isoformat() == live.today_ny() \
                    and until_open < timedelta(minutes=100):
                log.info("Waiting for the open at %s", clock.next_open)
                time.sleep(min(until_open.total_seconds() + 5, 900))
                continue
            break                                     # done for today
        if st.get("morning_sent") != live.today_ny():
            st["morning_sent"] = live.today_ny()
            notify.send("started " + datetime.now(live.NY).strftime("%a %b %d"),
                        "✅ Running. Mode: " + ("**WATCH ONLY** (no orders)" if watch else "**PAPER TRADING**")
                        + ". Report after the close.")
        if clock.next_close - now <= timedelta(minutes=config.FLATTEN_MINUTES_BEFORE_CLOSE):
            log.info("Near the close: making sure it holds nothing.")
            if safe:
                live.cancel_own_orders(c)
                live.close_own(c, "end of day")
            st["entries"] = {}
            live.save_state(st)
            time.sleep(max(0, (clock.next_close - now).total_seconds()) + 60)
            continue
        if (time.monotonic() - t0) / 60 > config.SESSION_MAX_MINUTES:
            log.info("This job is getting old - handing the day over to a fresh one.")
            CHAIN.write_text("session")
            live.save_state(st)
            return
        if st.get("halted") == live.today_ny() or not bundle:
            pass
        else:
            try:
                live.cycle(c, bundle, st, watch)
                errors = 0
            except Exception as e:
                errors += 1
                log.exception("Check failed (%d in a row): %s", errors, e)
                ERR.write_text(traceback.format_exc()[-3000:])
                if errors >= 5:
                    if safe and not watch:
                        live.close_own(c, "too many errors")
                    live.save_state(st)
                    raise
                time.sleep(60)
                continue
        live.save_state(st)
        nxt = datetime.now(timezone.utc)
        m = config.BAR_MINUTES
        wake = nxt.replace(second=0, microsecond=0) + timedelta(minutes=m - nxt.minute % m, seconds=30)
        flat_at = clock.next_close - timedelta(minutes=config.FLATTEN_MINUTES_BEFORE_CLOSE)
        time.sleep(max(5, (min(wake, flat_at) - nxt).total_seconds()))
    after_close(c, st, bundle)
    live.save_state(st)


def train():
    import train as t
    b = t.run()
    L = review.load_learned()
    if b["edge_passed"] and L.get("paused"):          # a fresh pass lifts a self-pause
        L.update(paused=False, review_from=live.today_ny())
        review.LEARNED.write_text(review.json.dumps(L, indent=1))
    rep = (config.LOG_DIR / "backtest_report.txt").read_text(encoding="utf-8")
    notify.send("weekly honesty test: " + ("PASSED - it may paper trade" if b["edge_passed"]
                                           else "no edge - keeps watching only"),
                "It re-learned from all history and tested itself on data it had never seen.\n\n"
                "```\n" + rep + "```")


def report():
    c = live.client()
    st = live.load_state()
    clock = c.get_clock()
    if clock.is_open:
        print("Market still open - the report comes after the close.")
        return
    after_close(c, st, model.load())


def check():
    lines = []
    c = live.client()
    acct = c.get_account()
    lines.append(f"✅ Alpaca keys work (paper: {config.PAPER}), equity ${float(acct.equity):,.2f}")
    st = live.load_state()
    ok = check_guard(c, st)
    lines.append("✅ Separate account - the other 7 bots can't be touched" if ok
                 else "🛑 " + st["guard_reason"])
    latest = live.latest_features(config.UNIVERSE)
    lines.append(f"✅ Live prices OK ({len(latest)} markets, last bar {latest['ts'].max():%b %d %H:%M} ET)")
    b = model.load()
    lines.append("✅ Model: " + (f"{b['style']}, hold {b['horizon'] * 15} min, trained {b['trained_at'][:10]}, test "
                               + ("PASSED" if b["edge_passed"] else "not passed -> watch only")
                               if b else "none yet -> training"))
    live.save_state(st)
    notify.send("health check", "\n\n".join(lines) + "\n\nPaper money - not financial advice.",
                urgent=not ok)
    if not b:
        notify.start_run("train")


def recover():
    """The job failed or was cut off. Tell Miles, and restart (limited per day)."""
    st = live.load_state()
    day = live.today_ny()
    n = st.setdefault("restarts", {}).get(day, 0)
    err = ERR.read_text()[-1500:] if ERR.exists() else "(no error text - the job was stopped or timed out)"
    try:
        c = live.client()
        market_open = c.get_clock().is_open
    except Exception:
        market_open = True
    if market_open and n < config.MAX_RESTARTS_PER_DAY:
        st["restarts"][day] = n + 1
        live.save_state(st)
        started = notify.start_run("session")
        notify.send("hiccup - fixed itself" if started else "hiccup - could NOT restart",
                    f"Something broke during trading, so it {'started a fresh run' if started else 'tried to restart but could not'}"
                    f" (restart {n + 1} of {config.MAX_RESTARTS_PER_DAY} today). Its open trades are kept.\n\n"
                    f"What broke:\n```\n{err}\n```", urgent=not started)
    else:
        why = "the market is closed" if not market_open else "it already restarted 3 times today"
        notify.send("problem - needs a look",
                    f"Something broke and it did not restart because {why}. "
                    f"The 7 other bots are not affected.\n\nWhat broke:\n```\n{err}\n```", urgent=True)
        if not market_open:
            try:
                c = live.client()
                after_close(c, st, model.load())   # still try to send the daily report
            except Exception as e:
                print("report after failure also failed:", e)


def chain():
    if CHAIN.exists():
        CHAIN.unlink()
        notify.start_run("session")


def main():
    setup_logging()
    mode = sys.argv[1] if len(sys.argv) > 1 else "session"
    watch = "--watch" in sys.argv
    if mode == "push":
        try:
            words = (config.ROOT / "requests" / "run.txt").read_text(encoding="utf-8-sig").split()
            mode = words[0] if words else "check"
        except OSError:
            mode = "check"
    log.info("run.py %s%s", mode, " (watch only)" if watch else "")
    {"session": lambda: session(watch), "report": report, "train": train, "check": check,
     "recover": recover, "chain": chain}.get(mode, check)()


if __name__ == "__main__":
    main()
