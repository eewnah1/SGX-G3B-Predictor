"""
G3B Next-Day + Multi-Horizon Predictor Dashboard
Run: PYTHONPATH=. python app.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

predictor = None


def _get_predictor():
    global predictor
    if predictor is None:
        from factors.predictor import G3BPredictor

        predictor = G3BPredictor()
        predictor.load_data()
        predictor.build_features()
        predictor.fit()
    return predictor


def _format_bucket_probs(probs: dict) -> str:
    if not probs:
        return "n/a"
    parts = []
    for b in ["Strong Down", "Weak Down", "Flat", "Weak Up", "Strong Up"]:
        p = probs.get(b, probs.get(str(b), 0.0))
        parts.append(f"{b}: {p:.1%}")
    return " | ".join(parts)


def _format_horizon(pred) -> list:
    lines = [
        f"### {pred.horizon}-day horizon",
        f"**Bucket:** {pred.bucket}",
        f"**Bucket probabilities:** {_format_bucket_probs(pred.bucket_probs)}",
        f"**Direction:** {pred.direction}  ·  **Confidence:** {pred.confidence:.1%}",
    ]
    if pred.expected_return_pct != 0.0:
        lines.append(f"**Expected return:** {pred.expected_return_pct:+.2f}%")
    lines.append(f"**Signal source:** {pred.signal_type.replace('_', ' ')}")
    lines.append(f"**XGB up-probability:** {pred.model_proba_up:.1%}")
    if pred.reasons:
        lines.append("**Reasons:**")
        for r in pred.reasons:
            lines.append(f"  • {r}")
    if pred.top_features:
        feats = ", ".join(
            [f"{name} ({imp:.2f})" for name, imp in pred.top_features[:5]]
        )
        lines.append(f"**Top factors:** {feats}")
    return lines


def run_predictor(cost_bps: float = 4.0) -> str:
    try:
        p = _get_predictor()
        pred = p.predict()
        bt = p.backtest(start="2024-01-01", end="2026-08-08", cost_bps=cost_bps)

        lines = [
            "=" * 60,
            "  G3B NEXT-DAY + MULTI-HORIZON PREDICTOR",
            "  Amundi Singapore STI ETF (G3B.SI)",
            "=" * 60,
            f"  As-of date          : {pred['as_of']}",
            f"  G3B last close      : SGD {pred['last_close']:.4f}",
            f"  Latest data date    : {pred['latest_date']}",
            "",
        ]

        for h in sorted(pred["horizons"].keys()):
            lines.extend(_format_horizon(pred["horizons"][h]))
            lines.append("")

        lines += [
            "-" * 60,
            "  Data health",
            f"    Status              : {pred['data_health']['status']}",
            f"    Latest date         : {pred['data_health']['latest_date']}",
            f"    Missing tickers     : "
            f"{pred['data_health']['missing_tickers'] or 'none'}",
            f"    Cached bars         : {pred['data_health']['n_rows']}",
            "",
            "-" * 60,
            "  Upcoming earnings (next 20 days, top holdings)",
        ]
        if pred["earnings"]:
            for e in pred["earnings"][:10]:
                sur = (
                    f" | surprise {e['surprise_pct']:+.1f}%"
                    if e.get("surprise_pct") is not None
                    and not np.isnan(e["surprise_pct"])
                    else ""
                )
                lines.append(
                    f"    {e['ticker']:>8}  {e['date']}  ({e['days_to']}d){sur}"
                )
        else:
            lines.append("    None within the next 20 days.")

        lines += [
            "",
            "-" * 60,
            "  Out-of-sample backtest 2024-01-01 -> 2026-08-08",
        ]
        for h, m in bt["horizons"].items():
            hc_n = m["high_conviction_n"]
            hc_a = m["high_conviction_accuracy"]
            hc = f" | HC {hc_n:>3} @ {hc_a:.1%}" if hc_n else " | HC n/a"
            da = m["directional_accuracy"]
            ar = m["avg_return_per_trade"]
            lines.append(
                f"    h={h:>2}  all={m['n']:>3} dir_acc={da:>6.1%}{hc} avg={ar:>+.3%}"
            )

        if pred["top_features_global"]:
            lines += [
                "",
                "-" * 60,
                "  Top global factors (averaged bucket-model importances)",
            ]
            for name, imp in pred["top_features_global"][:8]:
                lines.append(f"    {name:.<45} {imp:.3f}")

        lines += [
            "",
            "-" * 60,
            "  Research signal only — not financial advice.",
            f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 60,
        ]
        return "\n".join(lines)

    except Exception as e:
        logger.exception("Predictor failed")
        return (
            f"Error: {type(e).__name__}: {e}\n\n"
            "Try again shortly (Yahoo rate limits or memory issues are common)."
        )


if __name__ == "__main__":
    # Pre-load and fit so the Run Prediction button is fast.
    logger.info("Pre-loading predictor...")
    _get_predictor()
    logger.info("Predictor ready.")

    with gr.Blocks(title="G3B Next-Day + Multi-Horizon Predictor") as demo:
        gr.Markdown(
            """
            # G3B Next-Day + Multi-Horizon Predictor
            **Amundi Singapore STI ETF (G3B.SI)**

            Combines per-horizon XGBoost + LightGBM + LSTM bucket models with
            interpretable high-conviction rules, weighted bank/sector baskets, macro/FX
            proxies, and upcoming STI earnings-calendar features. Forecasts are shown as
            directional magnitude buckets (Strong/Weak Up/Down/Flat) for horizons 1, 2,
            3, 5, 10, 20 and 60 days.
            """
        )

        btn = gr.Button("Run Prediction", variant="primary")
        output = gr.Textbox(label="Prediction Card", lines=35)

        btn.click(fn=run_predictor, inputs=[], outputs=output)

        repo_url = "https://github.com/eewnah1/SGX-G3B-Predictor"
        gr.Markdown(f"Repo: [eewnah1/SGX-G3B-Predictor]({repo_url}) · Research only")

    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
