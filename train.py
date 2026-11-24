"""Download data, learn, test honestly, and save the model.

    python train.py            # uses Alpaca if keys are in .env, else Yahoo
    python train.py --days 730 # more history (Alpaca only)
"""
import argparse

from bot import config, data, features, model


def run(days: int = 365) -> dict:
    bars = data.get_bars(days=days)
    df = features.build(bars)
    print(f"[train] {df['day'].nunique()} trading days, walk-forward testing...")

    trades = model.walk_forward(df)
    if trades.empty:
        raise SystemExit("Not enough history to test. Need at least ~25 trading days.")
    stats = model.score(trades)
    bundle = model.train_and_save(df, stats)

    lines = [
        "=" * 56,
        " WALK-FORWARD BACKTEST (only data the model had NOT seen)",
        "=" * 56,
        f" Days tested          {stats['days_tested']}",
        f" Total return         {stats['total_return_pct']}%",
        f" Sharpe (yearly)      {stats['sharpe']}",
        f" Max drawdown         {stats['max_drawdown_pct']}%",
        f" Winning days         {stats['winning_days_pct']}%",
        f" Trades               {stats['trades']}",
        f" Right direction      {stats['hit_rate_pct']}% of bars in a trade",
        f" Costs paid           {stats['cost_paid_pct']}% of account",
        "-" * 56,
        (" EDGE FOUND -> bot is allowed to trade (paper first!)"
         if bundle["edge_passed"] else
         " NO RELIABLE EDGE -> bot will watch but NOT trade"),
        f" (needs Sharpe >= {config.MIN_SHARPE_TO_TRADE} and a positive return)",
        "=" * 56,
    ]
    report = "\n".join(lines)
    print(report)
    config.LOG_DIR.mkdir(exist_ok=True)
    (config.LOG_DIR / "backtest_report.txt").write_text(report + "\n")
    stats["daily"].rename("daily_return").to_csv(config.LOG_DIR / "backtest_daily.csv")
    return bundle


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=365)
    run(p.parse_args().days)
