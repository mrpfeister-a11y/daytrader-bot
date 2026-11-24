"""The learning part: train, honestly test, decide whether to trade at all.

How the honesty test works (walk-forward):
  1. Learn from ~3 years, trade the next ~3 months it has never seen, move on, repeat
     across all history since 2016.
  2. The older results ("validation") are used to choose the most profitable way to
     trade among 16: style (up/down per market vs. market-neutral "relative"),
     how long to hold (1 or 2 hours) and how sure it must be (55-66%).
  3. The last ~year ("final exam") is NEVER used to choose anything. The bot may
     trade only if the chosen way also made money there (Sharpe >= 1, positive return).
     Picking the best of many on the same data would just find luck; the exam catches that.
"""
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config
from .features import FEATURES


def target(style: str, h: int) -> str:
    return f"fwd_ret_{h}" if style == "direction" else f"fwd_rel_{h}"


def make_model():
    # Small, heavily regularized trees: market data is noisy, and big models
    # mostly learn the noise.
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=500, l2_regularization=1.0, random_state=0,
    )


def fit(df: pd.DataFrame, style: str, h: int):
    col = target(style, h)
    d = df.dropna(subset=[col])
    X = d[FEATURES].copy()
    empty = X.columns[X.isna().all()]
    X[empty] = 0.0  # a signal with no data yet (e.g. 20-day stats in the first weeks)
    return make_model().fit(X, (d[col] > 0).astype(int))


def signals(prob_up: np.ndarray, entry_prob: float = 0.57) -> np.ndarray:
    pos = np.where(prob_up > entry_prob, 1, 0)
    if config.ALLOW_SHORTS:
        pos = np.where(prob_up < 1 - entry_prob, -1, pos)
    return pos


def positions(frame: pd.DataFrame, style: str, entry_prob: float, prob_col: str = "prob") -> np.ndarray:
    """Turn predictions into +1 / -1 / 0 per row. Used by the backtest AND the live bot,
    so both follow exactly the same rules."""
    p = frame[prob_col].to_numpy()
    if style == "direction":
        pos = signals(p, entry_prob)
    else:  # relative: best K long, worst K short, only if the model is sure enough
        g = frame.groupby("ts")[prob_col]
        top = g.rank(ascending=False, method="first").to_numpy() <= config.RELATIVE_TOP_K
        bot = g.rank(ascending=True, method="first").to_numpy() <= config.RELATIVE_TOP_K
        pos = np.where(top & (p > entry_prob), 1, 0)
        if config.ALLOW_SHORTS:
            pos = np.where(bot & (p < 1 - entry_prob), -1, pos)
    late = frame["time_of_day"].to_numpy() >= config.NO_NEW_TRADES_TOD
    return np.where(late | np.isnan(p), 0, pos)


def hold_for_horizon(frame: pd.DataFrame, pos: np.ndarray, h: int) -> np.ndarray:
    """Once in a trade, keep it for the whole horizon it predicted unless the opposite
    signal shows up. Cuts costly in-and-out trading. The live bot does the same with
    the time each trade was opened."""
    s = pd.Series(np.where(pos == 0, np.nan, pos), index=frame.index, dtype=float)
    order = frame.sort_values("ts").index
    keys = [frame["symbol"].loc[order], frame["day"].loc[order]]
    s = s.loc[order].groupby(keys).ffill(limit=h - 1).reindex(frame.index)
    late = frame["time_of_day"].to_numpy() >= config.NO_NEW_TRADES_TOD
    return np.where(late, 0, s.fillna(0).to_numpy()).astype(int)


def variants():
    return [(s, h) for s in config.STYLES for h in config.HORIZONS]


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """Train on the past, predict the next stretch it has never seen, repeat."""
    days = np.array(sorted(df["day"].unique()))
    idx = df["day"].map({d: i for i, d in enumerate(days)}).to_numpy()
    n = len(days)
    first = min(config.TRAIN_WINDOW_DAYS, max(20, n // 3))
    step = min(config.TEST_STEP_DAYS, max(5, n // 12))
    keep = ["ts", "symbol", "day", "time_of_day", "next_ret"]
    out = []
    for i in range(first, n, step):
        tr = df[(idx >= max(0, i - config.TRAIN_WINDOW_DAYS)) & (idx < i)]
        te_mask = (idx >= i) & (idx < i + step)
        te = df.loc[te_mask, keep].copy()
        for s, h in variants():
            te[f"p_{s}_{h}"] = fit(tr, s, h).predict_proba(df.loc[te_mask, FEATURES])[:, 1]
        out.append(te)
        print(f"[walk-forward] tested {days[i]} .. {days[min(i + step, n) - 1]}", flush=True)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def score(trades: pd.DataFrame, cost_bps: float = None) -> dict:
    """Turn bar-by-bar positions into account-level results, after costs."""
    cost_bps = config.COST_BPS_PER_SIDE if cost_bps is None else cost_bps
    t = trades.sort_values(["symbol", "ts"]).copy()
    t.loc[t["next_ret"].isna(), "pos"] = 0  # flat into the close
    prev = t.groupby(["symbol", "day"])["pos"].shift(1).fillna(0)
    turnover = (t["pos"] - prev).abs()
    # exit cost for whatever is still open on the day's last bar
    last = t.groupby(["symbol", "day"]).cumcount(ascending=False) == 0
    turnover += np.where(last, t["pos"].abs(), 0)
    cost = turnover * cost_bps / 1e4

    t["w"] = t["pos"] * config.POSITION_PCT
    gross = t.groupby("ts")["w"].transform(lambda w: w.abs().sum())
    scale = np.minimum(1.0, config.MAX_GROSS_PCT / gross.replace(0, np.nan)).fillna(1)
    t["w"] *= scale
    t["pnl"] = t["w"] * t["next_ret"].fillna(0) - cost * config.POSITION_PCT * scale

    daily = t.groupby("day")["pnl"].sum()
    equity = (1 + daily).cumprod()
    dd = (equity / equity.cummax() - 1).min() if len(equity) else 0
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0
    active = t[t["pos"] != 0]
    hit = (np.sign(active["pos"]) == np.sign(active["next_ret"])).mean() if len(active) else 0
    years = max(len(daily) / 252, 1e-9)
    total = float(equity.iloc[-1] - 1) if len(equity) else 0.0
    return {
        "days_tested": int(len(daily)),
        "from": str(daily.index.min()) if len(daily) else "",
        "to": str(daily.index.max()) if len(daily) else "",
        "total_return_pct": round(total * 100, 2),
        "yearly_return_pct": round(((1 + total) ** (1 / years) - 1) * 100, 2) if total > -1 else -100.0,
        "sharpe": round(float(sharpe), 2),
        "max_drawdown_pct": round(float(dd) * 100, 2),
        "days_traded": int((daily != 0).sum()),
        "winning_days_pct": round(float((daily[daily != 0] > 0).mean()) * 100, 1) if (daily != 0).any() else 0,
        "trades": int((turnover > 0).sum()),
        "hit_rate_pct": round(float(hit) * 100, 1),
        "cost_paid_pct": round(float((cost * config.POSITION_PCT * scale).sum()) * 100, 2),
        "daily": daily,
    }


def choose(preds: pd.DataFrame, cost_bps: float) -> dict:
    """Pick the most profitable way on the validation years; grade it on the final exam."""
    days = np.array(sorted(preds["day"].unique()))
    hold = min(config.HOLDOUT_DAYS, len(days) // 3)
    cut = days[-hold] if hold else days[-1]
    val, exam = preds[preds["day"] < cut], preds[preds["day"] >= cut]

    def pos_for(frame, s, h, thr):
        return hold_for_horizon(frame, positions(frame, s, thr, f"p_{s}_{h}"), h)

    tried = []
    for s, h in variants():
        for thr in config.ENTRY_PROBS:
            st = score(val.assign(pos=pos_for(val, s, h, thr)), cost_bps)
            ex = score(exam.assign(pos=pos_for(exam, s, h, thr)), cost_bps)  # shown, never used to pick
            tried.append({"style": s, "horizon": h, "entry_prob": thr,
                          "val_sharpe": st["sharpe"], "val_return_pct": st["total_return_pct"],
                          "val_trades": st["trades"],
                          "exam_sharpe": ex["sharpe"], "exam_return_pct": ex["total_return_pct"]})
    best = max(tried, key=lambda r: r["val_sharpe"])
    s, h, thr = best["style"], best["horizon"], best["entry_prob"]
    return {"style": s, "horizon": h, "entry_prob": thr, "tried": tried,
            "validation": score(val.assign(pos=pos_for(val, s, h, thr)), cost_bps),
            "exam": score(exam.assign(pos=pos_for(exam, s, h, thr)), cost_bps)}


def edge_passed(choice: dict) -> bool:
    ex, va = choice["exam"], choice["validation"]
    return (ex["sharpe"] >= config.MIN_SHARPE_TO_TRADE and ex["total_return_pct"] > 0
            and va["sharpe"] > 0)


def train_and_save(df: pd.DataFrame, choice: dict, cost_bps: float):
    days = np.array(sorted(df["day"].unique()))
    recent = df[df["day"] >= days[max(0, len(days) - config.TRAIN_WINDOW_DAYS)]]
    m = fit(recent, choice["style"], choice["horizon"])
    clean = lambda st: {k: v for k, v in st.items() if k != "daily"}
    bundle = {
        "model": m,
        "style": choice["style"],
        "horizon": choice["horizon"],
        "entry_prob": choice["entry_prob"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "edge_passed": edge_passed(choice),
        "stats": clean(choice["exam"]),
        "validation": clean(choice["validation"]),
        "tried": choice["tried"],
        "cost_bps": cost_bps,
        "history": {"from": str(days[0]), "to": str(days[-1]), "days": int(len(days))},
        "universe": sorted(df["symbol"].unique()),
        "features": FEATURES,
    }
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.MODEL_PATH, compress=3)
    return bundle


def load():
    if not config.MODEL_PATH.exists():
        return None
    try:
        b = joblib.load(config.MODEL_PATH)
    except Exception:
        return None
    if b.get("features") != FEATURES or "horizon" not in b:   # an old model from before the upgrade
        return None
    return b
