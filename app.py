"""
G3B Next-Day Predictor Dashboard
Run:  PYTHONPATH=. python app.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _base_xgb_direction(features: pd.DataFrame) -> float:
    """Fallback XGBoost probability for days with no high-conviction rule match."""
    try:
        df = features.copy()
        df["target"] = (df["ret1"].shift(-1) > 0).astype(int)
        model_cols = [c for c in df.columns if c != "target"]
        train = df.dropna(subset=model_cols + ["target"])
        if len(train) < 100:
            return 0.5
        X = train[model_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = train["target"]
        model = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.8,
            min_child_weight=4,
            reg_alpha=0.15,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=2,
            verbosity=0,
        )
        model.fit(X, y)
        last = df[model_cols].iloc[[-1]].replace([np.inf, -np.inf], np.nan).fillna(0)
        return float(model.predict_proba(last)[0, 1])
    except Exception as exc:
        logger.warning("Base XGB fallback failed: %s", exc)
        return 0.5


def run_predictor(cost_bps: float = 4.0) -> str:
    try:
        from data.data_fetcher import load_g3b_data, make_bank_weighted_return
        from factors.high_conviction import (
            build_features,
            evaluate_high_conviction,
        )
        from backtest.high_conviction_backtest import run_backtest

        g3b, data = load_g3b_data(period="5y")
        if g3b is None or g3b.empty:
            return "Could not fetch G3B data (rate limit). Please try again in a minute."

        bank_dfs = {t: df for t, df in data.items() if t in ["D05.SI", "O39.SI", "U11.SI"]}
        bank_weight = make_bank_weighted_return(bank_dfs)
        macro = {t: df for t, df in data.items() if t not in ["G3B.SI", "D05.SI", "O39.SI", "U11.SI"]}

        features = build_features(g3b, macro, bank_weight)
        latest = features.iloc[-1]
        as_of = str(features.index[-1].date())
        last_close = float(g3b["Close"].iloc[-1])

        signal = evaluate_high_conviction(latest)

        if signal is None:
            proba = _base_xgb_direction(features)
            if proba >= 0.6:
                direction, confidence = "UP", f"{proba:.1%} (base model)"
                signal_text = "LONG / BASE MODEL"
            elif proba <= 0.4:
                direction, confidence = "DOWN", f"{1 - proba:.1%} (base model)"
                signal_text = "SHORT / BASE MODEL"
            else:
                direction, confidence = "SIDEWAYS", "50.0%"
                signal_text = "NEUTRAL / NO HIGH-CONVICTION SIGNAL"
            reasons = [
                "No high-conviction rule triggered today.",
                f"Base XGBoost up-probability: {proba:.1%}",
                "Fading macro/technical conditions — no edge strong enough to override.",
            ]
            rule_name = "None"
        else:
            if signal.direction == "LONG":
                direction, signal_text = "UP", "LONG"
            else:
                direction, signal_text = "DOWN", "SHORT"
            confidence = f"{signal.confidence:.1%}"
            reasons = signal.reasons
            rule_name = signal.rule_name

        # run backtest for the dashboard summary
        bt = run_backtest(start="2024-01-01", end="2026-08-08", cost_bps=cost_bps)

        lines = [
            "=" * 55,
            "  G3B NEXT-DAY PREDICTOR",
            "  Amundi Singapore STI ETF (G3B.SI)",
            "=" * 55,
            f"  As-of date          : {as_of}",
            f"  G3B last close      : SGD {last_close:.4f}",
            f"  Direction bias      : {direction}",
            f"  Signal              : {signal_text}",
            f"  Rule triggered      : {rule_name}",
            f"  Confidence          : {confidence}",
            "-" * 55,
            "  Reasons:",
        ]
        for r in reasons:
            lines.append(f"    • {r}")
        lines += [
            "-" * 55,
            f"  High-conviction backtest {bt.start} -> {bt.end}",
            f"    Directional accuracy : {bt.directional_accuracy:6.1%}",
            f"    Long-only accuracy   : {bt.long_accuracy:6.1%}" if bt.long_accuracy else "",
            f"    Short-only accuracy  : {bt.short_accuracy:6.1%}" if bt.short_accuracy else "",
            f"    Trades taken         : {bt.n_trades}",
            f"    Hit rate after cost  : {bt.hit_rate_after_cost:6.1%}",
            f"    Avg return / trade   : {bt.avg_return_per_trade:+.3%}",
            f"    Sharpe (annualised)  : {bt.sharpe:6.2f}",
            f"    Profit factor        : {bt.profit_factor:6.2f}",
            f"    Max drawdown         : {bt.max_drawdown:6.1%}",
            f"    Total return         : {bt.total_return:+.2%}",
            "-" * 55,
            "  Research signal only — not financial advice.",
            f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 55,
        ]
        return "\n".join(line for line in lines if line)

    except Exception as e:
        logger.exception("Predictor failed")
        return f"Error: {type(e).__name__}: {e}\n\nTry again shortly (Yahoo rate limits are common)."


if __name__ == "__main__":
    with gr.Blocks(title="G3B Next-Day Predictor") as demo:
        gr.Markdown(
            """
            # G3B Next-Day Predictor
            Amundi Singapore STI ETF — high-conviction rule + macro overlay

            G3B is ~58% DBS + OCBC + UOB. This dashboard uses actual G3B.SI prices,
            weighted bank returns, and cross-asset macro factors.
            """
        )

        btn = gr.Button("Run Prediction", variant="primary")
        output = gr.Textbox(label="Prediction Card", lines=25)

        btn.click(fn=run_predictor, inputs=[], outputs=output)

        gr.Markdown(
            "Repo: [eewnah1/SGX-G3B-Predictor](https://github.com/eewnah1/SGX-G3B-Predictor) · Research only"
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
