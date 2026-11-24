"""The learning part: train, honestly test, decide whether to trade at all."""
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config
from .features import FEATURES


def make_model():
    # Small, heavily regularized trees: market data is noisy, and big models
    # mostly learn the noise.
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=300, l2_regularization=1.0, random_state=0,
    )


def _xy(df):
    d = df.dropna(subset=["fwd_ret"])
    return d[FEATURES], (d["fwd_ret"] > 0).astype(int), d


def signals(prob_up: np.ndarray) -> np.ndarray:
    pos = np.where(prob_up > config.ENTRY_PROB, 1, 0)
    if config.ALLOW_SHORTS:
        pos = np.where(prob_up < 1 - config.ENTRY_PROB, -1, pos)
    return pos


def walk_forward(df: pd.DataFrame, min_train_days=20, test_days=5) -> pd.DataFrame:
    """Train on the past, trade the next few days it has never seen, repeat."""
    days = np.array(sorted(df["day"].unique()))
    out = []
    for i in range(min_train_days, len(days), test_days):
        train = df[df["day"] < days[i]]
        test = df[df["day"].isin(days[i:i + test_days])].copy()
        X, y, _ = _xy(train)
        m = make_model().fit(X, y)
        test["prob_up"] = m.predict_proba(test[FEATURES])[:, 1]
        test["pos"] = signals(test["prob_up"].values)
        test.loc[test["next_ret"].isna(), "pos"] = 0  # flat into the close
        out.append(test)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def score(trades: pd.DataFrame) -> dict:
    """Turn bar-by-bar positions into account-level results, after costs."""
    t = trades.sort_values(["symbol", "ts"]).copy()
    prev = t.groupby(["symbol", "day"])["pos"].shift(1).fillna(0)
    turnover = (t["pos"] - prev).abs()
    # exit cost for whatever is still open on the day's last bar
    last = t.groupby(["symbol", "day"]).cumcount(ascending=False) == 0
    turnover += np.where(last, t["pos"].abs(), 0)
    cost = turnover * config.COST_BPS_PER_SIDE / 1e4

    t["w"] = t["pos"] * config.POSITION_PCT
    gross = t.groupby("ts")["w"].transform(lambda w: w.abs().sum())
    scale = np.minimum(1.0, config.MAX_GROSS_PCT / gross.replace(0, np.nan)).fillna(1)
    t["w"] *= scale
    t["pnl"] = t["w"] * t["next_ret"].fillna(0) - cost * config.POSITION_PCT * scale

    daily = t.groupby("day")["pnl"].sum()
    equity = (1 + daily).cumprod()
    dd = (equity / equity.cummax() - 1).min()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0
    active = t[t["pos"] != 0]
    hit = (np.sign(active["pos"]) == np.sign(active["next_ret"])).mean() if len(active) else 0
    return {
        "days_tested": int(len(daily)),
        "total_return_pct": round(float(equity.iloc[-1] - 1) * 100, 2) if len(equity) else 0,
        "sharpe": round(float(sharpe), 2),
        "max_drawdown_pct": round(float(dd) * 100, 2) if len(equity) else 0,
        "winning_days_pct": round(float((daily > 0).mean()) * 100, 1),
        "trades": int((turnover > 0).sum()),
        "hit_rate_pct": round(float(hit) * 100, 1),
        "cost_paid_pct": round(float((cost * config.POSITION_PCT * scale).sum()) * 100, 2),
        "daily": daily,
    }


def train_and_save(df: pd.DataFrame, stats: dict):
    X, y, _ = _xy(df)
    m = make_model().fit(X, y)
    passed = (stats["sharpe"] >= config.MIN_SHARPE_TO_TRADE
              and stats["total_return_pct"] > 0)
    bundle = {
        "model": m,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "edge_passed": bool(passed),
        "stats": {k: v for k, v in stats.items() if k != "daily"},
        "universe": sorted(df["symbol"].unique()),
    }
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.MODEL_PATH)
    return bundle


def load():
    if not config.MODEL_PATH.exists():
        return None
    return joblib.load(config.MODEL_PATH)
