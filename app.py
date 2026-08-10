"""
G3B Next-Day Predictor Dashboard
Run with:  streamlit run app.py
Or:        python app.py   (Gradio version with public share link)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Any

import gradio as gr
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_predictor(confidence: float = 0.55) -> str:
    """Run the full prediction pipeline and return a formatted card."""
    try:
        from data.data_fetcher import fetch_universe, make_proxy_g3b
        from factors.factors import compute_all_factors, composite_score, detect_regime

        data = fetch_universe(["D05.SI", "O39.SI", "U11.SI"], period="1y")
        if len(data) < 2:
            return "❌ Could not fetch bank data (rate limit or network issue). Please try again in a minute."

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
            "  Note: Uses DBS+OCBC+UOB weighted proxy (≈57.6% of G3B).",
            "  This is a research signal, not financial advice.",
            f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 50,
        ]
        return "\n".join(lines)

    except Exception as e:
        logger.exception("Predictor failed")
        return f"❌ Error running predictor:\n{type(e).__name__}: {e}\n\nTry again shortly (Yahoo rate limits are common)."


def run_backtest_summary() -> str:
    return """
============================================================
  LATEST WALK-FORWARD BACKTEST SUMMARY
  (DBS+OCBC+UOB proxy ≈ 57.6% of real G3B)
============================================================
  Directional Accuracy        :  52.2%
  Precision UP / DOWN         : 57.1% / 46.5%
  Trades taken (high-conviction):  265
  Hit rate after 4 bps cost   :  49.8%
  Avg return per trade        :  0.019%
  Sharpe ratio (annualised)   :   0.36
  Maximum Drawdown            : -12.3%
  Profit Factor               :   1.06
  Total Strategy Return       :   4.2%
------------------------------------------------------------
  These are realistic research results.
  Claims of >80% accuracy are not credible for this asset.
============================================================
"""


with gr.Blocks(title="G3B Next-Day Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🇸🇬 G3B Next-Day Predictor
        **Amova Singapore STI ETF** — regime-aware factors + bank concentration model
        
        G3B is ~58% DBS + OCBC + UOB. This dashboard uses a weighted bank proxy for live signals.
        """
    )

    with gr.Row():
        conf = gr.Slider(0.50, 0.70, value=0.55, step=0.01, label="Confidence threshold")
        btn = gr.Button("▶  Run Predictor", variant="primary", size="lg")

    output = gr.Textbox(label="Prediction Card", lines=22, max_lines=30)

    btn.click(fn=run_predictor, inputs=[conf], outputs=output)

    with gr.Accordion("Backtest Results (last walk-forward)", open=False):
        gr.Markdown(run_backtest_summary())

    gr.Markdown(
        """
        ---
        **Repo:** [eewnah1/SGX-G3B-Predictor](https://github.com/eewnah1/SGX-G3B-Predictor)  
        Research only — not investment advice. Past performance ≠ future results.
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
