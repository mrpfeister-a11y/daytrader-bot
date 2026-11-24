"""Download all history, learn, test honestly, and save the model.

    python train.py              # everything since 2016 (Alpaca keys) or ~60 days (Yahoo)
    python train.py --days 365   # just the last year (quicker)
"""
import argparse
import json

from bot import config, data, features, model


def cost_to_use() -> float:
    """Pessimistic trading cost: the default, or what live trading actually cost if worse."""
    try:
        learned = json.loads((config.LOG_DIR / "learned.json").read_text())
        measured = float(learned.get("slippage_bps_per_side") or 0)
    except (OSError, ValueError):
        measured = 0.0
    return max(config.COST_BPS_PER_SIDE, measured)


def style_name(s: str) -> str:
    return {"direction": "up/down guess per market",
            "relative": "market-neutral (best 3 long, worst 3 short)"}[s]


def report_lines(bundle: dict) -> list:
    ex, va, h = bundle["stats"], bundle["validation"], bundle["history"]
    w = 18
    period = lambda s: s["from"][:7] + ".." + s["to"][:7]
    tried = sorted(bundle["tried"], key=lambda r: -r["val_sharpe"])
    return [
        "=" * 62,
        " HONESTY TEST (walk-forward: only data the model had NOT seen)",
        "=" * 62,
        f" History used         {h['from']} .. {h['to']} ({h['days']} trading days)",
        f" Best way found       {style_name(bundle['style'])}, hold {bundle['horizon'] * 15} min,",
        f"                      trade when >= {bundle['entry_prob']:.0%} sure",
        f" Trading cost used    {bundle['cost_bps']:.1f} bps per side",
        "-" * 62,
        f" {'':20} {'Validation':>{w}} {'FINAL EXAM':>{w}}",
        f" {'Period':20} {period(va):>{w}} {period(ex):>{w}}",
        f" {'Total return':20} {va['total_return_pct']:>{w - 1}}% {ex['total_return_pct']:>{w - 1}}%",
        f" {'Per year':20} {va['yearly_return_pct']:>{w - 1}}% {ex['yearly_return_pct']:>{w - 1}}%",
        f" {'Sharpe (yearly)':20} {va['sharpe']:>{w}} {ex['sharpe']:>{w}}",
        f" {'Max drawdown':20} {va['max_drawdown_pct']:>{w - 1}}% {ex['max_drawdown_pct']:>{w - 1}}%",
        f" {'Days it traded':20} {str(va['days_traded']) + '/' + str(va['days_tested']):>{w}} "
        f"{str(ex['days_traded']) + '/' + str(ex['days_tested']):>{w}}",
        f" {'Won (days traded)':20} {va['winning_days_pct']:>{w - 1}}% {ex['winning_days_pct']:>{w - 1}}%",
        f" {'Trades':20} {va['trades']:>{w}} {ex['trades']:>{w}}",
        f" {'Right direction':20} {va['hit_rate_pct']:>{w - 1}}% {ex['hit_rate_pct']:>{w - 1}}%",
        f" {'Costs paid':20} {va['cost_paid_pct']:>{w - 1}}% {ex['cost_paid_pct']:>{w - 1}}%",
        "-" * 62,
        f" All {len(tried)} ways tried (best validation first):",
        "   style       hold  sure   valid. Sharpe/return   exam Sharpe/return",
    ] + [
        f"   {r['style']:<10} {r['horizon'] * 15:>4}m  {r['entry_prob']:.0%}   "
        f"{r['val_sharpe']:>6} / {r['val_return_pct']:>7}%    {r['exam_sharpe']:>6} / {r['exam_return_pct']:>7}%"
        for r in tried
    ] + [
        "-" * 62,
        (" EDGE FOUND -> bot is allowed to paper trade"
         if bundle["edge_passed"] else
         " NO RELIABLE EDGE -> bot will watch but NOT trade"),
        f" (needs a final-exam Sharpe >= {config.MIN_SHARPE_TO_TRADE}, a positive final-exam"
        " return, and a positive validation Sharpe)",
        "=" * 60,
    ]


def run(days: int = None) -> dict:
    bars = data.get_bars(days=days)
    df = features.build(bars)
    del bars
    print(f"[train] {df['day'].nunique()} trading days, walk-forward testing...", flush=True)

    preds = model.walk_forward(df)
    if preds.empty:
        raise SystemExit("Not enough history to test. Need at least ~25 trading days.")
    cost = cost_to_use()
    choice = model.choose(preds, cost)
    bundle = model.train_and_save(df, choice, cost)

    report = "\n".join(report_lines(bundle))
    print(report)
    config.LOG_DIR.mkdir(exist_ok=True)
    (config.LOG_DIR / "backtest_report.txt").write_text(report + "\n", encoding="utf-8")
    choice["exam"]["daily"].rename("daily_return").to_csv(config.LOG_DIR / "backtest_exam_daily.csv")
    summary = {k: bundle[k] for k in ("style", "horizon", "entry_prob", "trained_at", "edge_passed", "stats",
                                      "validation", "tried", "cost_bps", "history")}
    (config.LOG_DIR / "last_training.json").write_text(json.dumps(summary, indent=1, default=str))
    return bundle


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=None)
    run(p.parse_args().days)
