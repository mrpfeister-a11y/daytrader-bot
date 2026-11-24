"""Run the bot on your own computer or a server (the cloud uses run.py instead).

    python trade.py           # trade every market day, forever (paper account by default)
    python trade.py --watch   # never place orders, only log what it would do

Same rules as the cloud: paper only, trades only after passing its honesty test,
max 5% per stock / 50% total, stops at -2% for the day, never holds overnight,
retrains weekly, and never touches anything outside its own 20 markets.
"""
import argparse
import logging
import time
from datetime import datetime, timedelta, timezone

import run
from bot import live, model

log = logging.getLogger("bot")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true")
    args = p.parse_args()
    run.setup_logging()
    while True:
        try:
            if run.model_age_days(model.load()) >= 7 and not live.client().get_clock().is_open:
                run.train()
            run.session(args.watch)
        except KeyboardInterrupt:
            log.info("Stopped by you. Open positions (if any) are left as they are.")
            return
        except Exception as e:  # network blips etc. - keep going
            log.exception("Error: %s. Retrying in 60s.", e)
            time.sleep(60)
            continue
        if run.CHAIN.exists():
            run.CHAIN.unlink()
            continue
        nxt = live.client().get_clock().next_open
        wait = (nxt - datetime.now(timezone.utc) - timedelta(minutes=20)).total_seconds()
        log.info("Done for today. Next open %s", nxt)
        time.sleep(max(60, min(wait, 6 * 3600)))


if __name__ == "__main__":
    main()
