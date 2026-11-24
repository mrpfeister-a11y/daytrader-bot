"""All settings in one place. Secrets come from the .env file, never from code."""
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

# --- Costs (be pessimistic) -------------------------------------------------
COST_BPS_PER_SIDE = float(os.getenv("COST_BPS_PER_SIDE", "2"))  # slippage+spread

# --- Model / edge gate ------------------------------------------------------
ENTRY_PROB = float(os.getenv("ENTRY_PROB", "0.57"))  # how sure before trading
ALLOW_SHORTS = os.getenv("ALLOW_SHORTS", "true").lower() == "true"
MIN_SHARPE_TO_TRADE = float(os.getenv("MIN_SHARPE_TO_TRADE", "1.0"))
RETRAIN_DAYS = int(os.getenv("RETRAIN_DAYS", "7"))

# --- Risk limits ------------------------------------------------------------
POSITION_PCT = float(os.getenv("POSITION_PCT", "0.05"))    # 5% of equity each
MAX_GROSS_PCT = float(os.getenv("MAX_GROSS_PCT", "0.50"))  # 50% total exposure
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.02"))  # stop at -2%
FLATTEN_MINUTES_BEFORE_CLOSE = 10
CLOUD_FLATTEN_MINUTES = 45  # cloud runs can start late, so stop earlier

MODEL_PATH = ROOT / "models" / "model.joblib"
LOG_DIR = ROOT / "logs"
