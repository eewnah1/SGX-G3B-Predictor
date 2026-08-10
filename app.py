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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_predictor(confidence: float = 0.55) -> str:
    try:
        from data.data_fetcher import fetch_universe, make_proxy_g3b
        from factors.factors import compute_all_factors, composite_score, detect_regime

        data = fetch_universe(["D05.SI", "O39.SI", "U11.SI"], period="1y")
        if len(data) < 2:
            return "Could not fetch bank data (rate limit). Please try again in a minute."

        price = make_proxy_g3b(data)
        factors = compute_all_factors(price, data)
        comp = composite_score(factors)
        regime = detect_regime(price)

        last_close = float(price["Close"].iloc[-1])
        as_of = str(price.index[-1].date())

        if comp >= 0.35:
            direction, signal, conviction = "UP", "LONG", "HIGH"
        elif comp <= -0.35:
            direction, signal, conviction = "DOWN", "SHORT", "HIGH"
        else:
            direction = "UP" if comp > 0 else "DOWN"
            signal, conviction = "FLAT / NO TRADE", "LOW"

        lines = [
            "=" * 50,
            "  G3B NEXT-DAY PREDICTOR",
            "=" * 50,
            f"  As-of date          : {as_of}",
            f"  Proxy last close    : {last_close:.3f}",
            f"  Regime              : {regime}",
            f"  Composite score     : {comp:+.3f}",
            f"  Direction bias      : {direction}",
            f"  Signal              : {signal}",
            f"  Conviction          : {conviction}",
            "-" * 50,
            "  Factor Breakdown:",
        ]
        for f in factors:
            lines.append(f"    {f.name:22s}  {f.score:+.2f}   (w={f.weight:.2f})  {f.description}")
        lines += [
            "-" * 50,
            "  Note: Uses DBS+OCBC+UOB weighted proxy (~57.6% of G3B).",
            "  Research signal only — not financial advice.",
            f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 50,
        ]
        return "\n".join(lines)

    except Exception as e:
        logger.exception("Predictor failed")
        return f"Error: {type(e).__name__}: {e}\n\nTry again shortly (Yahoo rate limits are common)."


BACKTEST_MD = """
**Latest Walk-Forward Backtest** (DBS+OCBC+UOB proxy)

| Metric | Value |
|--------|-------|
| Directional Accuracy | 52.2% |
| Precision UP / DOWN | 57.1% / 46.5% |
| Hit rate after costs | 49.8% |
| Sharpe | 0.36 |
| Max Drawdown | -12.3% |
| Profit Factor | 1.06 |
| Total Return | +4.2% |
"""


with gr.Blocks(title="G3B Next-Day Predictor") as demo:
    gr.Markdown(
        """
        # G3B Next-Day Predictor
        Amova Singapore STI ETF — regime-aware factors + bank concentration model

        G3B is ~58% DBS + OCBC + UOB. This dashboard uses a weighted bank proxy.
        """
    )

    with gr.Row():
        conf = gr.Slider(0.50, 0.70, value=0.55, step=0.01, label="Confidence threshold")
        btn = gr.Button("Run Predictor", variant="primary")

    output = gr.Textbox(label="Prediction Card", lines=20)

    btn.click(fn=run_predictor, inputs=[conf], outputs=output)

    with gr.Accordion("Backtest Results", open=False):
        gr.Markdown(BACKTEST_MD)

    gr.Markdown(
        "Repo: [eewnah1/SGX-G3B-Predictor](https://github.com/eewnah1/SGX-G3B-Predictor) · Research only"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
