"""Offline tests with a fake Alpaca account (no keys, no orders). Run:  python tests/test_bot.py
Needs a trained model in models/ (python train.py works without keys, on Yahoo data)."""
import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run  # noqa: E402
from bot import config, data, features, live, model, notify, review  # noqa: E402

config.LOG_DIR = ROOT / "tests" / "_logs"
config.REPORT_DIR = ROOT / "tests" / "_reports"
config.FIRST_TRADING_DAY = "2000-01-01"   # the fake days are older than the real first day
live.STATE = config.LOG_DIR / "state.json"
review.LEARNED = config.LOG_DIR / "learned.json"
review.DAYS = config.LOG_DIR / "live_days.csv"
run.CHAIN = config.LOG_DIR / "chain.flag"
run.ERR = config.LOG_DIR / "last_error.txt"
for d in (config.LOG_DIR, config.REPORT_DIR):
    d.mkdir(exist_ok=True)
    for f in d.iterdir():
        f.unlink()

SENT, STARTED = [], []
notify.send = lambda title, msg, urgent=False: SENT.append((title, msg, urgent)) or True
notify.start_run = lambda mode, workflow="bot.yml": STARTED.append(mode) or True
run.time.sleep = lambda s: None

passed = 0


def ok(cond, what):
    global passed
    assert cond, "FAILED: " + what
    passed += 1
    print("  ok -", what)


# ------------------------------------------------------------ fake Alpaca ----
class Obj(types.SimpleNamespace):
    pass


class FakeClient:
    def __init__(self, account="PA-TEST-123", positions=None):
        live.client = lambda: self  # every new fake account becomes the active one
        self.account = account
        self.pos = dict(positions or {})
        self.orders, self.closed = [], []
        self.clocks = []
        self.equity = 100000.0

    def get_account(self):
        return Obj(account_number=self.account, equity=str(self.equity), last_equity="100000")

    def get_all_positions(self):
        return [Obj(symbol=s, qty=str(q), market_value=str(q * 100.0)) for s, q in self.pos.items()]

    def close_position(self, sym):
        self.closed.append(sym)
        self.pos.pop(sym, None)

    def submit_order(self, req):
        q = req.qty if req.side.value == "buy" else -req.qty
        self.pos[req.symbol] = self.pos.get(req.symbol, 0) + q
        o = Obj(id=f"o{len(self.orders)}", symbol=req.symbol, side=req.side, filled_qty=str(req.qty),
                filled_avg_price="100.0", filled_at=datetime.now(timezone.utc))
        self.orders.append(o)
        return o

    def get_asset(self, sym):
        return Obj(shortable=True, easy_to_borrow=True)

    def get_orders(self, req):
        return [] if getattr(req, "status", None) and req.status.value == "open" else self.orders

    def cancel_order_by_id(self, i):
        pass

    def get_calendar(self, req):
        return [1]

    def get_clock(self):
        return self.clocks.pop(0) if len(self.clocks) > 1 else self.clocks[0]


def clock(is_open, now, next_open, next_close):
    return Obj(is_open=is_open, timestamp=now, next_open=next_open, next_close=next_close)


# ---------------------------------------------------------------- data ----
bundle = model.load()
assert bundle, "train a model first: python train.py"
bars = data.from_yahoo(config.UNIVERSE, 30)
feats = features.build(bars)
last_day = feats["day"].max()
day_rows = feats[feats["day"] == last_day]
mid_ts = sorted(day_rows["ts"].unique())[8]           # a mid-morning bar
latest_mid = day_rows[day_rows["ts"] == mid_ts].copy()
live.latest_features = lambda universe: latest_mid.copy()
data.from_alpaca = lambda symbols, days: bars
live.today_ny = lambda: str(last_day)
review.sip_today = lambda day: {}
review_day = str(last_day)

print("1. account safety guard")
c = FakeClient()
ok(live.guard(c)[0], "separate account with no other stocks -> allowed")
fp = config.account_fingerprint("PA-OTHER-BOTS")
config.BLOCKED_ACCOUNT_HASHES.add(fp)
okk, why = live.guard(FakeClient(account="PA-OTHER-BOTS"))
ok(not okk and "other 7 bots" in why, "the other bots' account -> refused")
okk, why = live.guard(FakeClient(positions={"COST": 5}))
ok(okk, "account shared with bot #9 (other stocks) -> allowed")

print("2. only closes its own markets")
c = FakeClient(positions={"SPY": 10, "AAPL": -3})
live.close_own(c, "test")
ok(sorted(c.closed) == ["AAPL", "SPY"], "closes only its own symbols")

print("3. decisions + hold-for-horizon")
st = {}
lm = latest_mid.copy()
live.decide(lm, bundle)
ok(set(np.unique(lm["signal"])) <= {-1, 0, 1}, "signals are -1/0/+1")
lm["signal"] = 0
lm.loc[lm["symbol"] == "SPY", "signal"] = 1
t = live.targets(lm, 100000, st, horizon=4)
ok(t["SPY"] > 0 and sum(1 for v in t.values() if v) == 1, "buys SPY only, 5% size")
ok(abs(t["SPY"] * float(lm[lm.symbol == "SPY"].close.iloc[0]) - 5000) < 1000, "position ~5% of account")
st["entries"] = {"SPY": {"side": 1, "bar": (mid_ts - pd.Timedelta(minutes=30)).isoformat()}}
lm["signal"] = 0
ok(live.targets(lm, 100000, st, 4)["SPY"] > 0, "keeps the trade inside its hour")
st["entries"]["SPY"]["bar"] = (mid_ts - pd.Timedelta(minutes=60)).isoformat()
ok(live.targets(lm, 100000, st, 4)["SPY"] == 0, "lets go after the hour")
late = lm.copy()
late["time_of_day"] = 0.95
late["signal"] = 1
ok(all(v == 0 for v in live.targets(late, 100000, {}, 4).values()), "no trades after 3:30 PM")
pos = model.positions(late.assign(prob=0.99), "direction", 0.55)
ok((pos == 0).all(), "backtest uses the same 3:30 PM rule")

print("4. a full fake trading day")
b2 = dict(bundle, edge_passed=True)            # pretend the test passed so it trades
model.load = lambda path=None: b2 if path in (None, config.MODEL_PATH) else None
c = FakeClient()
ny = live.NY
d0 = datetime.fromisoformat(review_day).replace(tzinfo=ny)
open_t, close_t = d0.replace(hour=9, minute=30), d0.replace(hour=16)
c.clocks = [clock(False, d0.replace(hour=9, minute=10), open_t, close_t),        # waits for the open
            clock(True, d0.replace(hour=10, minute=1), open_t + timedelta(days=1), close_t),
            clock(True, d0.replace(hour=11, minute=1), open_t + timedelta(days=1), close_t),
            clock(True, d0.replace(hour=15, minute=50), open_t + timedelta(days=1), close_t),  # flatten
            clock(False, d0.replace(hour=16, minute=2), open_t + timedelta(days=1), close_t + timedelta(days=1))]
orig_cycle = live.cycle
forced = {"SPY": 1, "QQQ": -1}


def cycle_forcing(cl, bun, st, watch):
    lm2 = latest_mid.copy()
    live.decide(lm2, bun)
    lm2["signal"] = lm2["symbol"].map(forced).fillna(0).astype(int)
    live.record_decisions(lm2, "trade")
    live.rebalance(cl, live.targets(lm2, 100000, st, bun["horizon"]), lm2, st, watch)
    st.setdefault("cycles", {})[live.today_ny()] = 24
    st.setdefault("start_equity", 100000.0)


live.cycle = cycle_forcing
run.session()
ok(not any("started" in s[0] for s in SENT), "no morning email (only the report)")
ok({o.symbol for o in c.orders} == {"SPY", "QQQ"}, "bought SPY, shorted QQQ")
ok(c.pos == {}, "holds nothing at the end of the day")
rep = [s for s in SENT if s[0].startswith("report")]
ok(len(rep) == 1 and "Today's trades" in rep[0][1] and "TRADING" in rep[0][1], "daily report emailed")
ok("Paper money" in rep[0][1], "report says paper money")
ok((config.REPORT_DIR / "latest.md").exists(), "report saved to reports/latest.md")
L = json.loads(review.LEARNED.read_text())
ok(L.get("live_guesses", 0) >= 0 and "slippage_bps_per_side" in L, "learned.json updated")
n_reports = len(rep)
c.clocks = [clock(False, d0.replace(hour=18), open_t + timedelta(days=1), close_t + timedelta(days=1))]
run.session()
ok(len([s for s in SENT if s[0].startswith("report")]) == n_reports, "no second report the same day")

print("5. watch-only never orders")
live.today_ny = lambda: "2099-01-02"
c = FakeClient()
c.clocks = [clock(True, d0.replace(hour=10, minute=1), open_t + timedelta(days=1), close_t),
            clock(False, d0.replace(hour=16, minute=2), open_t + timedelta(days=1), close_t + timedelta(days=1))]
model.load = lambda path=None: dict(bundle, edge_passed=False) if path in (None, config.MODEL_PATH) else None
run.session()
ok(c.orders == [] and c.closed == [], "no orders when the honesty test did not pass")
live.today_ny = lambda: str(last_day)

print("6. wrong account -> no orders, urgent email")
live.today_ny = lambda: "2099-01-03"
c = FakeClient(account="PA-OTHER-BOTS", positions={"COST": 5, "SPY": 2})
c.clocks = [clock(True, d0.replace(hour=10, minute=1), open_t + timedelta(days=1), close_t),
            clock(True, d0.replace(hour=15, minute=50), open_t + timedelta(days=1), close_t),
            clock(False, d0.replace(hour=16, minute=2), open_t + timedelta(days=1), close_t + timedelta(days=1))]
model.load = lambda path=None: b2 if path in (None, config.MODEL_PATH) else None
run.session()
ok(c.orders == [] and c.closed == [] and c.pos == {"COST": 5, "SPY": 2},
   "other bots' account untouched (not even at the close)")
ok(any(s[2] and "safety stop" in s[0] for s in SENT), "urgent safety email")
live.today_ny = lambda: str(last_day)

print("7. self-fixing")
live.today_ny = lambda: "2099-01-04"
c = FakeClient()
live.client = lambda: c
c.clocks = [clock(True, d0.replace(hour=10, minute=1), open_t + timedelta(days=1), close_t)]


def boom(*a):
    raise RuntimeError("fake network error")


live.cycle = boom
try:
    run.session()
    ok(False, "should have stopped after 5 errors")
except RuntimeError:
    ok(True, "stops after 5 errors in a row (after retrying)")
ok("fake network error" in run.ERR.read_text(), "error saved for the email")
STARTED.clear()
run.recover()
ok(STARTED == ["session"] and any("fixed itself" in s[0] for s in SENT), "restarts itself + emails")
for _ in range(3):
    run.recover()
ok(STARTED.count("session") == 3 and any("needs a look" in s[0] for s in SENT),
   "max 3 restarts a day, then asks for help")
n_sent = len(SENT)
STARTED.clear()
sys.argv = ["run.py", "recover", "research_orb", "cancelled"]
run.recover()
ok(len(SENT) == n_sent and STARTED == [], "a run cancelled by hand: no alarm, no restart")
sys.argv = ["run.py", "recover", "research", "failure"]
run.recover()
ok(len(SENT) == n_sent + 1 and not SENT[-1][2] and STARTED == [],
   "a research job failing: one calm note, no restart, no report")
sys.argv = ["tests/test_bot.py"]
run.CHAIN.write_text("session")
STARTED.clear()
run.chain()
ok(STARTED == ["session"] and not run.CHAIN.exists(), "long day handed to a fresh job")

print("8. champion vs. challenger")
import shutil  # noqa: E402
real_load = model.__dict__.get("_real_load") or model.load
config.MODEL_PATH = config.LOG_DIR / "champ.joblib"
config.CHALLENGER_PATH = config.LOG_DIR / "chall.joblib"
import joblib  # noqa: E402
joblib.dump(dict(bundle, edge_passed=True, trained_at="2026-01-01T00:00:00+00:00"), config.MODEL_PATH)
joblib.dump(dict(bundle, edge_passed=True, trained_at="2026-02-01T00:00:00+00:00"), config.CHALLENGER_PATH)
model.load = lambda path=None: joblib.load(path or config.MODEL_PATH) if (path or config.MODEL_PATH).exists() else None
model.load_challenger = lambda: model.load(config.CHALLENGER_PATH)
champ = {"n": 30, "hits": 14, "avg_bps": -1.0}
review.shadow_check = lambda day, h, who="champion": {"n": 30, "hits": 18, "avg_bps": 3.0}
for i in range(config.CONTEST_MIN_DAYS - 1):
    line = review.contest(f"2026-03-0{i + 1}", model.load(), champ)
    ok(config.CHALLENGER_PATH.exists() and "day" in line, f"contest day {i + 1}: still competing")
line = review.contest("2026-03-09", model.load(), champ)
ok("won" in line and not config.CHALLENGER_PATH.exists(), "better challenger is promoted after 5 days")
ok(model.load()["trained_at"].startswith("2026-02"), "the promoted model now trades")
joblib.dump(dict(bundle, edge_passed=True, trained_at="2026-04-01T00:00:00+00:00"), config.CHALLENGER_PATH)
review.shadow_check = lambda day, h, who="champion": {"n": 30, "hits": 10, "avg_bps": -5.0}
for i in range(config.CONTEST_MAX_DAYS):
    line = review.contest(f"2026-04-{i + 10}", model.load(), champ)
ok("stays" in line and not config.CHALLENGER_PATH.exists()
   and model.load()["trained_at"].startswith("2026-02"), "worse challenger is dropped, champion kept")
STARTED.clear()
run.train()
ok(True, "weekly training runs when no contest is going on")  # (trains on Yahoo data offline)
ok(config.CHALLENGER_PATH.exists(), "new weekly model becomes a challenger, not the champion")
ok(model.load()["trained_at"].startswith("2026-02"), "champion untouched until the challenger proves itself")

print("9. bot #9 (stocks in play) in the shared account")
from bot import sip  # noqa: E402
for name in ("STATE", "TRADES", "LEARNED", "TODAY", "ERR", "CHAIN"):
    setattr(sip, name, config.LOG_DIR / getattr(sip, name).name)
config.SIP_PARAMS = config.LOG_DIR / "sip_params.json"
sip.time.sleep = lambda s: None
DAY9 = "2026-12-01"
live.today_ny = lambda: DAY9


class FakeOrder(Obj):
    pass


class SipClient:
    """Fake account shared by both bots: SPY belongs to bot #8, the rest to bot #9."""
    def __init__(self):
        self.pos = {"SPY": 10}                       # bot #8's position - must never be touched
        self.orders = {}
        self.cancelled, self.closed = [], []

    def get_all_assets(self, req):
        mk = lambda s, ex="NASDAQ", short=True: Obj(symbol=s, tradable=True, exchange=Obj(value=ex),
                                                    shortable=short, easy_to_borrow=short)
        return [mk("NVDA"), mk("ZZZA"), mk("ZZZB", short=False), mk("ZZZC", "OTC"), mk("BRK.B"), mk("SPY", "ARCA")]

    def get_account(self):
        return Obj(account_number="PA-TEST-9", equity="100000", last_equity="100000")

    def get_all_positions(self):
        return [Obj(symbol=s, qty=str(q), market_value=str(q * 50.0)) for s, q in self.pos.items()]

    def submit_order(self, req):
        o = FakeOrder(id=f"s{len(self.orders)}", symbol=req.symbol, side=req.side, qty=req.qty,
                      stop_price=req.stop_price, status=Obj(value="accepted"), filled_qty="0",
                      filled_avg_price=None, filled_at=None)
        self.orders[o.id] = o
        return o

    def fill(self, sym, price):
        for o in self.orders.values():
            if o.symbol == sym and o.status.value == "accepted":
                o.status, o.filled_qty, o.filled_avg_price = Obj(value="filled"), str(o.qty), str(price)
                o.filled_at = datetime.now(timezone.utc)
                q = o.qty if o.side.value == "buy" else -o.qty
                self.pos[sym] = self.pos.get(sym, 0) + q
                return

    def get_orders(self, req):
        st = req.status.value
        if st == "open":
            return [o for o in self.orders.values() if o.status.value == "accepted"]
        if st == "closed":
            return [o for o in self.orders.values() if o.status.value == "filled"]
        return list(self.orders.values())

    def cancel_order_by_id(self, i):
        self.cancelled.append(self.orders[str(i)].symbol)
        self.orders[str(i)].status = Obj(value="canceled")

    def close_position(self, sym):
        self.closed.append(sym)
        self.pos.pop(sym, None)


sc = SipClient()
stocks = sip.stock_list(sc)
ok(set(stocks) == {"ZZZA", "ZZZB"}, "stock list: no bot #8 markets (NVDA, SPY), no OTC, no odd symbols")
ok(stocks["ZZZA"] and not stocks["ZZZB"], "knows which stocks it may short")

# candidates + picks from fake prices
days14 = [f"2026-11-{d:02d}" for d in range(10, 24)]
cands = pd.DataFrame({"atr": [2.0, 1.5], "avgvol": [3e6, 2e6], "n": [14, 14], "prev_close": [50.0, 30.0],
                      "easy_short": [True, False], "days": [days14, days14]}, index=["ZZZA", "ZZZB"])
avg = pd.Series({"ZZZA": 10000.0, "ZZZB": 10000.0})
sip._open_bar = lambda symbols, day: pd.DataFrame(
    {"open": [50.0, 30.0], "high": [51.0, 30.5], "low": [49.5, 29.0], "close": [50.8, 29.2],
     "volume": [30000.0, 25000.0]}, index=["ZZZA", "ZZZB"])
picks = sip.pick(cands, avg, DAY9, 20)
ok(list(picks.index) == ["ZZZA"], "ZZZB (falling, not shortable) skipped; ZZZA (rising, 3x volume) picked")
sizes = sip.plan_sizes(picks, 50000, 0.10)
ok(int(sizes.iloc[0]) == 245, "size: 1%-risk size (2,500 sh) capped at 25% of its $50k budget = 245 shares")
config.SIP_MAX_LEVERAGE = 1.0
big = sip.plan_sizes(picks.assign(atr=0.05), 50000, 0.10)
ok(float(big.iloc[0]) * 51.0 <= 50000 * 0.25 + 51, "never more than 25% of its budget in one stock (no borrowing)")

st9 = {"day": DAY9, "orders": {}, "picked": True, "done": False}
sip.place_entries(sc, picks, sizes, st9)
o = list(sc.orders.values())[0]
ok(o.symbol == "ZZZA" and o.side.value == "buy" and o.stop_price == 51.0, "buy-stop placed at the 5-minute high")
sc.fill("ZZZA", 51.02)
sip.manage(sc, st9, 0.10)
stops = [x for x in sc.orders.values() if x.side.value == "sell"]
ok(len(stops) == 1 and abs(stops[0].stop_price - 50.82) < 0.01, "after the fill: stop-loss at fill - 10% ATR")
sip.manage(sc, st9, 0.10)
ok(len([x for x in sc.orders.values() if x.side.value == "sell"]) == 1, "no duplicate stop-loss")
sip.flatten(sc)
ok(sc.closed == ["ZZZA"] and "SPY" not in sc.cancelled and sc.pos == {"SPY": 10},
   "end of day: closes only its own stock - bot #8's SPY untouched")
ok("ZZZA" in sc.cancelled, "end of day: its open stop-loss is cancelled")
pnl8 = live.own_day_pnl(sc, DAY9, live.is_dt)
ok(abs(pnl8 - 500.0) < 1e-6, "bot #8's daily result counts only its own stocks")

# practice day: one winner, one stopped out
mins = pd.date_range(f"{DAY9} 09:35", periods=5, freq="1min", tz="America/New_York").tz_convert("UTC")
sip._bars = lambda symbols, start, end, minutes=None, feed="iex", adjustment="split": pd.DataFrame({
    "symbol": ["ZZZA"] * 5 + ["ZZZQ"] * 5, "timestamp": list(mins) * 2,
    "open": [50.9, 51.1, 51.5, 52.0, 52.5, 30.0, 29.9, 29.6, 30.4, 30.9],
    "high": [51.0, 51.6, 52.1, 52.6, 53.0, 30.1, 29.95, 29.7, 30.5, 31.0],
    "low": [50.8, 51.0, 51.4, 51.9, 52.4, 29.8, 29.5, 29.5, 30.3, 30.8],
    "close": [50.95, 51.5, 52.0, 52.5, 53.0, 29.9, 29.6, 30.4, 30.9, 30.9]})
pk = pd.DataFrame({"side": [1, -1], "high": [51.0, 30.2], "low": [49.5, 29.6], "atr": [2.0, 1.5]},
                  index=["ZZZA", "ZZZQ"])
res = sip.simulate(pk, [100, 100], DAY9, 0.10)
win = next(r for r in res if r["symbol"] == "ZZZA")
loss = next(r for r in res if r["symbol"] == "ZZZQ")
ok(win["triggered"] and not win["stopped"] and win["pnl"] > 150, "practice: breakout that keeps going = a win")
ok(loss["triggered"] and loss["stopped"] and loss["pnl"] < 0, "practice: short that turns back = stopped out")
L9 = sip.learn(DAY9, "practice", res, 50000)
ok(L9["days"][DAY9]["trades"] == 2 and sip.TRADES.exists(), "practice trades logged for learning")
for k in range(config.LIVE_REVIEW_MIN_DAYS):
    L9 = sip.learn(f"2026-12-{k + 2:02d}", "trade", [{"symbol": "ZZZA", "side": 1, "triggered": True,
                                                      "pnl": -100.0, "slippage_bps": 8.0}], 50000)
ok(L9["paused"] and L9["slippage_bps"] == 8.0, "losing 20 real days -> pauses itself; real cost measured (8 bps)")
ok(sip.practicing("2026-12-30"), "while paused it only practices")

# a full practice session: no orders at all
sip.save_json(sip.LEARNED, {})
config.SIP_TRADE_FROM = "2099-01-01"
sc2 = SipClient()
live.client = lambda: sc2
sip.candidates = lambda c, day: cands
sip.opening_volume_avg = lambda c: avg
ny9 = datetime.fromisoformat(DAY9).replace(tzinfo=live.NY)
clocks9 = [clock(True, ny9.replace(hour=9, minute=36), ny9 + timedelta(days=1), ny9.replace(hour=16)),
           clock(True, ny9.replace(hour=15, minute=50), ny9 + timedelta(days=1), ny9.replace(hour=16))]
sc2.get_clock = lambda: clocks9.pop(0) if len(clocks9) > 1 else clocks9[0]
sip.session()
t9 = json.loads(sip.TODAY.read_text())
ok(sc2.orders == {} and sc2.closed == [] and sc2.pos == {"SPY": 10}, "practice day: no orders, nothing touched")
ok(t9["mode"] == "practice" and t9["day"] == DAY9, "practice results saved for the shared report")
sec, bad = review.bot9_section(DAY9, t9, {"dt": {}, "sip": {}})
ok("PRACTICE" in sec and "Bot #9" in sec, "shared daily report has a bot #9 section")
config.SIP_TRADE_FROM = "2026-11-27"

print("10. backtest pieces")
fr = pd.DataFrame({"ts": pd.date_range("2026-01-05 10:00", periods=8, freq="15min", tz="America/New_York"),
                   "symbol": "SPY", "day": "2026-01-05", "time_of_day": 0.3})
held = model.hold_for_horizon(fr, np.array([1, 0, 0, 0, 0, -1, 0, 0]), 4)
ok(list(held) == [1, 1, 1, 1, 0, -1, -1, -1], "backtest holds a trade for its horizon")
ok(len(model.variants()) * len(config.ENTRY_PROBS) == 16, "16 ways of trading are compared")

print("11. hand-over keeps the morning's logs")
for wf in ("bot.yml", "sip.yml"):
    txt = (config.ROOT / ".github" / "workflows" / wf).read_text(encoding="utf-8")
    after = txt.split("actions/checkout@v4", 1)[1][:200]
    ok("ref: main" in after, f"{wf}: every run loads the newest saved logs (bug of Nov 25)")

print(f"\nALL {passed} TESTS PASSED")
