# Day-Trading Bot (Alpaca, learns from 20 markets)

**Read this first.** This bot is NOT guaranteed to make money, and most
day-trading strategies lose money. It's built to protect you: it tests
itself on data it has never seen and **refuses to trade unless it finds a
real edge**. On the 60 days of free data available today it found **no
edge** (-5.7% after costs), so right now it will only *watch* and log what
it would do. With your Alpaca keys it can test on 1+ year of data, which
tells you a lot more.

## What it does
- Every 15 minutes it looks at 20 markets (S&P 500, Nasdaq, small caps,
  Dow, banks, energy, tech, health, bonds, gold and 10 big stocks).
- A machine-learning model trained on all of them together guesses
  whether each will go up or down over the next hour.
- It only buys (or shorts) when the model is fairly confident.
- It retrains itself every week and re-runs the honesty test each time.

## Built-in safety
| Rule | Default |
|---|---|
| Account | **Paper (fake money)** |
| Max per stock | 5% of account |
| Max total invested | 50% of account |
| Daily loss stop | -2%, then it stops for the day |
| Overnight holds | Never. Closes everything 10 min before the bell |
| No proven edge | Watches only, no orders |

## Step 1 — Alpaca paper keys (you do this)
1. Sign up at alpaca.markets.
2. Switch to **Paper Trading**, go to **API Keys** and generate a key pair.
3. Copy `.env.example` to `.env` and paste the two keys in.
   Never share `.env` or paste your keys into a chat.

## Step 2 — See whether it has an edge
```
./start.sh --watch --once
```
First run installs everything, downloads 1 year of data, trains, and prints
the backtest report (also saved to `logs/backtest_report.txt`).

## Step 3a — Run it on a computer
```
./start.sh
```
Leave it running. The computer must stay on and awake during market hours.

## Step 3b — Free cloud: GitHub Actions (what we set up)
GitHub runs `trade.py --cloud` every 15 minutes on weekdays from the
`.github/workflows/bot.yml` schedule. Public repo = unlimited free minutes.
Your keys live in the repo's Secrets, never in the code.
- Check on it: repo -> **Actions** tab -> click a run -> "Run one cycle".
- Turn it off: Actions -> day-trading-bot -> "..." -> **Disable workflow**.
- GitHub pauses schedules in repos with no changes for 60 days and emails
  you first. Click **Enable workflow** to restart it.
- Run logs in a public repo are visible to others (they show the paper
  account balance and trades, never your keys).

## Step 3c — Paid cloud server (alternative)
The `Dockerfile` lets any cloud server run it 24/7. Cheapest simple route,
about $5-6/month:
1. Create a small Linux server (1 GB RAM is enough) at a provider such as
   DigitalOcean, Hetzner, or Linode.
2. Copy this folder to it (without `.env`), then create `.env` on the
   server with your keys.
3. On the server:
   ```
   curl -fsSL https://get.docker.com | sh
   docker build -t daytrader .
   docker run -d --name daytrader --restart always --env-file .env -v $PWD/logs:/app/logs -v $PWD/models:/app/models daytrader
   ```
4. Watch it: `docker logs -f daytrader`

`--restart always` brings it back after crashes or server reboots.

## Files it writes
- `logs/bot.log` — everything it thinks and does
- `logs/trades.csv` — every order
- `logs/backtest_report.txt` — latest honesty test

## Before ever using real money
Run paper for at least a month. Compare `logs/trades.csv` results with the
backtest. Only if both are positive, consider real money, and only an
amount you can afford to lose completely. Switching needs two settings in
`.env` (see `.env.example`). This isn't financial advice.
