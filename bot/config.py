"""All settings in one place. Secrets come from the environment / .env file, never from code."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- Alpaca -----------------------------------------------------------------
ALPACA_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "")
# Paper trading unless BOTH of these say otherwise.
PAPER = os.getenv("ALPACA_PAPER", "true").lower() != "false"
LIVE_CONFIRMED = os.getenv("I_UNDERSTAND_LIVE_TRADING_RISK", "no").lower() == "yes"

# Accounts this bot must NEVER trade in (salted PBKDF2 fingerprint of the account
# number, so the number itself isn't published). This is the account of the 7 other
# bots: this bot sells everything it holds every afternoon, which would wreck them.
BLOCKED_ACCOUNT_HASHES = {
    "b5507b625190f7dac6c0d4f99b81a1489d1252f0fc9dfd35667cd8d4ce336e01",
}


def account_fingerprint(account_number: str) -> str:
    import hashlib
    return hashlib.pbkdf2_hmac("sha256", str(account_number).encode(), b"daytrader-guard", 200_000).hex()

# --- What it trades ---------------------------------------------------------
# A spread of index ETFs, sectors, bonds/commodities and liquid large caps.
# The model is trained on ALL of them together, so it learns patterns that
# show up across markets instead of memorizing one stock.
UNIVERSE = os.getenv(
    "UNIVERSE",
    "SPY,QQQ,IWM,DIA,XLF,XLE,XLK,XLV,TLT,GLD,"
    "AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA,AMD,JPM,NFLX",
).split(",")
BAR_MINUTES = 15            # one decision every 15 minutes
# Predict the move over the next N bars (4 x 15min = 1 hour). Longer
# horizons mean fewer trades and less money lost to costs.
HORIZON_BARS = int(os.getenv("HORIZON_BARS", "4"))
HORIZONS = (4, 8)           # the backtest tries holding 1 hour and 2 hours

# --- History used for learning ---------------------------------------------
# "Everything available": Alpaca's full-market (SIP) 15-min history starts in 2016.
HISTORY_START = os.getenv("HISTORY_START", "2016-01-01")
TRAIN_WINDOW_DAYS = int(os.getenv("TRAIN_WINDOW_DAYS", "756"))   # learn from the last ~3 years at each step
TEST_STEP_DAYS = int(os.getenv("TEST_STEP_DAYS", "63"))          # then trade the next ~3 months unseen
HOLDOUT_DAYS = int(os.getenv("HOLDOUT_DAYS", "252"))             # last ~year: final exam, never used to choose settings

# --- Costs (be pessimistic) -------------------------------------------------
COST_BPS_PER_SIDE = float(os.getenv("COST_BPS_PER_SIDE", "2"))  # slippage+spread
# (after live trades, the bot measures its REAL slippage and uses that if worse)

# --- Model / edge gate ------------------------------------------------------
STYLES = ("direction", "relative")   # up/down guess per market, or beat-the-others (market neutral)
ENTRY_PROBS = (0.55, 0.58, 0.62, 0.66)  # how sure before trading (picked on the validation years)
RELATIVE_TOP_K = 3                   # relative style: long the best 3, short the worst 3
ALLOW_SHORTS = os.getenv("ALLOW_SHORTS", "true").lower() == "true"
MIN_SHARPE_TO_TRADE = float(os.getenv("MIN_SHARPE_TO_TRADE", "1.0"))
RETRAIN_DAYS = int(os.getenv("RETRAIN_DAYS", "7"))

# --- Risk limits ------------------------------------------------------------
POSITION_PCT = float(os.getenv("POSITION_PCT", "0.05"))    # 5% of equity each
MAX_GROSS_PCT = float(os.getenv("MAX_GROSS_PCT", "0.50"))  # 50% total exposure
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.02"))  # stop at -2%
NO_NEW_TRADES_TOD = 0.92            # from 3:30 PM ET: everything flat (same in the backtest)
FLATTEN_MINUTES_BEFORE_CLOSE = 15   # safety net: sell anything still open
CLOUD_FLATTEN_MINUTES = 45          # old one-shot cloud mode only

# --- Learning from its own live results ------------------------------------
LIVE_REVIEW_MIN_DAYS = 20           # after this many trading days, compare live vs. backtest
SESSION_MAX_MINUTES = 300           # one cloud job (limit 340 min); then it hands over to a fresh one
MAX_RESTARTS_PER_DAY = 3

MODEL_PATH = ROOT / "models" / "model.joblib"            # the champion (trades)
CHALLENGER_PATH = ROOT / "models" / "challenger.joblib"  # this week's new model (practice guesses only)
CONTEST_MIN_DAYS = 5     # champion vs. challenger: compare live guesses for at least 5 trading days
CONTEST_MAX_DAYS = 10    # then decide; by 10 days the challenger is promoted or dropped
CONTEST_MIN_GUESSES = 20
LOG_DIR = ROOT / "logs"
REPORT_DIR = ROOT / "reports"
