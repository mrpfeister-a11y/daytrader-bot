# Day-Trading Bot (Alpaca paper, learns from 20 markets)

**Read this first.** No bot can be guaranteed to make money, and most day-trading
strategies lose money. This one is built to protect you: it tests itself on data it
has never seen and **refuses to trade unless it finds a real edge**. Paper money only.

## What it does
- Learns from **all 15-minute price history since 2016** for 20 markets (S&P 500,
  Nasdaq, small caps, Dow, banks, energy, tech, health, bonds, gold and 10 big stocks).
- Research-based signals: intraday momentum (first half hour + overnight gap),
  last days' moves, relative strength vs. the other markets, stocks vs. bonds mood,
  volume, time of day.
- **Backtests 16 ways of trading** (up/down per market or market-neutral; hold 1 or 2
  hours; 55-66% confidence), picks the best on older years, then grades it on the
  last year it never used to choose (the "final exam"). It trades only if the final
  exam was profitable too (Sharpe >= 1, positive return). Otherwise it only watches.
- **Learns from its own trading**: logs every guess (also while watching), checks them
  after the close, measures its real trading costs (used in the next test), and pauses
  itself if live results are worse than the test after 20 trading days.
- Re-learns every Saturday (and whenever its model is a week old).

## Safety
| Rule | Setting |
|---|---|
| Account | Paper only (`ALPACA_PAPER: "true"` in the workflow) |
| Other bots | Refuses the other 7 bots' account; refuses any account holding other stocks; only ever touches its own 20 markets |
| Per stock / total | 5% / 50% of the account |
| Daily loss stop | -2%, then it closes its trades and stops for the day |
| Overnight | Never. Flat from 3:30 PM New York, safety sweep at 3:45 |
| No proven edge | Watch only |

## How it runs (GitHub Actions, free for public repos)
One job per day trades from the open to the close (a check every 15 minutes), then emails
the daily report. Emails come from GitHub (a closed issue that @mentions you).
**This repo is public, so those issues are public too** - paper numbers only, never keys
or account numbers.

Self-fixing: retries each check; a crashed job emails you and restarts itself (up to 3x a
day); a day longer than one cloud job is handed to a fresh job; extra timer starts during
the day and after the close restart it or send a missed report; a Claude watchdog
routine is the last backup.

## Files
- `run.py` - what the cloud runs (`session`, `report`, `train`, `check`, `recover`, `chain`)
- `train.py` - downloads all history, backtests the 16 ways, saves `models/model.joblib`
- `bot/features.py` signals - `bot/model.py` learning + honesty test - `bot/live.py`
  trading + safety guard - `bot/review.py` daily report + learning from its own trades -
  `bot/notify.py` emails - `bot/config.py` all settings
- `logs/backtest_report.txt` latest honesty test, `reports/latest.md` latest daily report,
  `logs/learned.json` what it learned from live trading
- `tests/test_bot.py` - offline tests with a fake account (`python tests/test_bot.py`)
- `trade.py` - run it on your own computer instead (`python trade.py --watch`)

## Keys
GitHub secrets `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` from a **separate** Alpaca paper
account. Never paste keys into a chat.
