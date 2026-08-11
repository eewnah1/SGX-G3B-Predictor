"""FastAPI dashboard for SGX G3B (Amundi STI ETF) Multi-Horizon Predictor."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from anyio.to_thread import run_sync
from datetime import date, datetime


def _clean_json(obj):
    import math
    try:
        import numpy as np
    except Exception:
        np = None
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean_json(v) for v in obj]
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return _clean_json(obj.model_dump())
    if hasattr(obj, "dict") and callable(obj.dict):
        return _clean_json(obj.dict())
    if np is not None:
        if isinstance(obj, np.ndarray):
            return _clean_json(obj.tolist())
        if isinstance(obj, np.generic):
            obj = obj.item()
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return obj


import logging
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO)
from factors.predictor import G3BPredictor
_predictor = None
def _get_predictor():
    global _predictor
    if _predictor is None:
        p = G3BPredictor()
        p.load_data()
        p.build_features()
        p.fit()
        _predictor = p
    return _predictor

_latest_prediction = None
_latest_backtest = None

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>__TITLE__</title>
  <style>
    :root { --bg:#0f172a; --card:#1e293b; --text:#e2e8f0; --muted:#94a3b8; --negative:#f43f5e; --low:#22c55e; --mid:#f59e0b; --high:#ef4444; --accent:#3b82f6; }
    * { box-sizing: border-box; }
    body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 24px; }
    h1 { margin: 0 0 8px; font-size: 1.5rem; }
    .subtitle { color: var(--muted); margin-bottom: 24px; }
    button { background: var(--accent); color: white; border: none; padding: 12px 20px; border-radius: 8px; font-size: 1rem; cursor: pointer; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .status { margin-top: 12px; color: var(--muted); font-size: 0.9rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }
    .card { background: var(--card); border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    .card h2 { margin: 0 0 8px; font-size: 0.9rem; color: var(--muted); text-transform: uppercase; }
    .big { font-size: 2.2rem; font-weight: 700; }
    .bar { height: 10px; border-radius: 5px; background: #334155; margin-top: 12px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 5px; }
    .banner { border-radius: 12px; padding: 24px; margin-bottom: 24px; text-align: center; display: none; }
    .rec-up { background: #14532d; color: #86efac; }
    .rec-down { background: #450a0a; color: #fecaca; }
    .rec-neutral { background: #334155; color: #cbd5e1; }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.9rem; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #334155; }
    th { color: var(--muted); font-weight: 500; }
    pre { background: #0b1220; padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 0.85rem; }
    details { background: #0b1220; border-radius: 8px; padding: 12px; margin-top: 16px; }
    summary { cursor: pointer; color: var(--muted); }
    .ok { color: var(--low); }
    .stale { color: var(--mid); }
    .bad { color: var(--negative); }
    .error { color: var(--negative); }
    ul { color: var(--muted); padding-left: 20px; }
  </style>
</head>
<body>
  <h1>__TITLE__</h1>
  <div class="subtitle">FastAPI live predictor. Click <strong>Run prediction</strong> to fetch the latest forecast.</div>
  <div style="margin-bottom: 24px;">
    <button id="runBtn" onclick="runPrediction()">Run prediction now</button>
    <div class="status" id="status">Ready.</div>
  </div>
  <div id="banner" class="banner rec-neutral">
    <div class="big" id="bannerText">-</div>
    <div id="bannerSub">-</div>
  </div>
  <div class="grid" id="metrics"></div>
  <div class="grid" id="probs"></div>
  <div class="card" style="margin-bottom: 24px;">
    <h2>Reasons / interpretation</h2>
    <ul id="reasons"><li style="color:var(--muted)">No prediction yet.</li></ul>
  </div>
  <div class="card" style="margin-bottom: 24px;">
    <h2>Backtest summary</h2>
    <div id="backtest" style="color:var(--muted);">Click Run prediction to generate a backtest summary.</div>
  </div>
  <div class="grid">
    <div class="card">
      <h2>Data health / source status</h2>
      <table id="health"><thead><tr><th>Source</th><th>Status</th><th>Value</th></tr></thead><tbody></tbody></table>
    </div>
    <div class="card">
      <h2>Factors / features</h2>
      <table id="factors"><thead><tr><th>Factor</th><th>Value</th></tr></thead><tbody></tbody></table>
    </div>
  </div>
  <details>
    <summary>Raw JSON</summary>
    <pre id="raw">{}</pre>
  </details>

  <script>
    const $ = id => document.getElementById(id);
    const API = window.location.protocol + '//' + window.location.host + '/api/v1';

    async function api(path, method='GET', body=null) {
      const ctrl = new AbortController();
      const to = setTimeout(() => ctrl.abort(), 120000);
      const headers = body ? { 'Content-Type': 'application/json' } : {};
      try {
        const res = await fetch(API + path, { method, headers, body: body ? JSON.stringify(body) : null, signal: ctrl.signal });
        clearTimeout(to);
        if (!res.ok) { const t = await res.text(); throw new Error(`${res.status} ${res.statusText}: ${t}`); }
        return await res.json();
      } catch (e) { clearTimeout(to); throw e; }
    }

    function fmtPct(v) { return (Number(v || 0) * 100).toFixed(1) + '%'; }
    function fmtNum(v) { if (v === null || v === undefined) return '-'; if (typeof v === 'number') return Number.isFinite(v) ? v.toFixed(4) : v; return v; }

    function directionClass(text) {
      const t = (text || '').toLowerCase();
      if (t.includes('buy') || t.includes('up') || t.includes('bull') || t.includes('long')) return 'rec-up';
      if (t.includes('sell') || t.includes('down') || t.includes('bear') || t.includes('short')) return 'rec-down';
      return 'rec-neutral';
    }

    function barColor(label) {
      const l = (label || '').toLowerCase();
      if (l.includes('down') || l.includes('sell') || l.includes('negative')) return 'var(--negative)';
      if (l.includes('up') || l.includes('buy') || l.includes('positive')) return 'var(--low)';
      if (l.includes('flat') || l.includes('mid') || l.includes('neutral')) return 'var(--mid)';
      return 'var(--accent)';
    }

    function renderMetrics(p) {
      const cards = [];
      if (p.ticker || p.symbol) cards.push(['Ticker', p.ticker || p.symbol || '-']);
      if (p.date || p.as_of) cards.push(['Date', p.date || p.as_of || '-']);
      if (p.close !== undefined || p.price !== undefined) cards.push(['Close / price', fmtNum(p.close !== undefined ? p.close : p.price)]);
      if (p.confidence !== undefined) cards.push(['Confidence', fmtPct(p.confidence)]);
      if (p.signal || p.prediction) cards.push(['Signal', p.signal || p.prediction]);
      if (p.backtest_accuracy !== undefined) cards.push(['Backtest accuracy', fmtPct(p.backtest_accuracy)]);
      $('metrics').innerHTML = cards.map(([k, v]) => `<div class="card"><h2>${k}</h2><div class="big">${v}</div></div>`).join('');
    }

    function renderProbs(p) {
      let probs = p.probabilities || p.bucket_probs;
      let html = '';
      if (probs && typeof probs === 'object') {
        const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
        html = entries.map(([k, v]) => `<div class="card"><h2>${k}</h2><div class="big" style="color:${barColor(k)}">${fmtPct(v)}</div><div class="bar"><div class="bar-fill" style="width:${Math.min(Number(v||0)*100,100).toFixed(1)}%;background:${barColor(k)}"></div></div></div>`).join('');
      } else if (p.proba_up !== undefined) {
        const up = Number(p.proba_up || 0);
        const down = p.proba_down !== undefined ? Number(p.proba_down) : 1 - up;
        html = `<div class="card"><h2>Probability UP</h2><div class="big" style="color:var(--low)">${fmtPct(up)}</div><div class="bar"><div class="bar-fill" style="width:${(up*100).toFixed(1)}%;background:var(--low)"></div></div></div>`;
        html += `<div class="card"><h2>Probability DOWN</h2><div class="big" style="color:var(--negative)">${fmtPct(down)}</div><div class="bar"><div class="bar-fill" style="width:${(down*100).toFixed(1)}%;background:var(--negative)"></div></div></div>`;
      } else if (p.blended_proba !== undefined) {
        const v = Number(p.blended_proba || 0);
        html = `<div class="card"><h2>Blended proba</h2><div class="big" style="color:${barColor(v > 0.5 ? 'up' : 'down')}">${fmtPct(v)}</div><div class="bar"><div class="bar-fill" style="width:${(v*100).toFixed(1)}%;background:${barColor(v > 0.5 ? 'up' : 'down')}"></div></div></div>`;
      }
      $('probs').innerHTML = html || '<div class="card" style="grid-column:1/-1;color:var(--muted)">No probability data.</div>';
    }

    function renderList(id, arr) {
      const el = $(id);
      if (!arr || !arr.length) { el.innerHTML = '<li style="color:var(--muted)">None.</li>'; return; }
      el.innerHTML = arr.map(x => `<li>${x}</li>`).join('');
    }

    function renderHealth(p) {
      const tbody = $('health').querySelector('tbody');
      let rows = [];
      if (p.source_status && Array.isArray(p.source_status)) rows = p.source_status.map(s => [s.name, s.status, s.value]);
      else if (p.data_health && typeof p.data_health === 'object') rows = Object.entries(p.data_health).map(([k, v]) => [k, v, '']);
      else if (p.data_quality_flags && typeof p.data_quality_flags === 'object') rows = Object.entries(p.data_quality_flags).map(([k, v]) => [k, v, '']);
      if (!rows.length) { tbody.innerHTML = '<tr><td colspan="3" style="color:var(--muted)">No health data.</td></tr>'; return; }
      tbody.innerHTML = rows.map(r => {
        const status = (r[1] || '').toString().toLowerCase();
        const cls = status === 'ok' || status === 'good' || status === true || status === 'true' ? 'ok' : (status === 'stale' ? 'stale' : 'bad');
        return `<tr><td>${r[0]}</td><td class="${cls}">${r[1]}</td><td style="color:var(--muted)">${r[2] || '-'}</td></tr>`;
      }).join('');
    }

    function renderFactors(p) {
      const tbody = $('factors').querySelector('tbody');
      let rows = [];
      if (p.factor_contributions && Array.isArray(p.factor_contributions)) rows = p.factor_contributions.map(f => [f.name, f.score]);
      else if (p.factor_scores && typeof p.factor_scores === 'object') rows = Object.entries(p.factor_scores);
      else if (p.top_features && Array.isArray(p.top_features)) rows = p.top_features.map(f => Array.isArray(f) ? f : [f.feature, f.importance]);
      else if (p.features && typeof p.features === 'object') rows = Object.entries(p.features).slice(0, 30);
      if (!rows.length) { tbody.innerHTML = '<tr><td colspan="2" style="color:var(--muted)">No factor data.</td></tr>'; return; }
      tbody.innerHTML = rows.map(r => `<tr><td>${r[0]}</td><td>${fmtNum(r[1])}</td></tr>`).join('');
    }

    function renderPrediction(data) {
      const p = data.prediction || data;
      window._lastPrediction = p;
      $('banner').style.display = 'block';
      const dirText = p.direction || p.prediction || p.signal || 'NEUTRAL';
      $('banner').className = 'banner ' + directionClass(dirText);
      $('bannerText').textContent = dirText.toString().toUpperCase();
      const conf = (Number(p.confidence || 0) * 100).toFixed(1);
      $('bannerSub').textContent = `${p.ticker || p.symbol || ''} · Confidence ${conf}%`;
      renderMetrics(p);
      renderProbs(p);
      renderList('reasons', p.reasons || (p.interpretation ? [p.interpretation] : undefined));
      renderHealth(p);
      renderFactors(p);
      $('raw').textContent = JSON.stringify(data, null, 2);
    }

    function renderBacktest(bt) {
      const el = $('backtest');
      if (!bt) { el.textContent = 'No backtest summary available.'; return; }
      if (bt.error || bt.message) { el.textContent = bt.error || bt.message; return; }
      let html = '';
      if (bt.accuracy !== undefined) {
        html += `<div class="grid"><div class="card"><h2>Direction accuracy</h2><div class="big" style="color:var(--low)">${fmtPct(bt.accuracy)}</div></div>`;
        if (bt.n_samples !== undefined) html += `<div class="card"><h2>Samples</h2><div class="big">${bt.n_samples}</div></div>`;
        if (bt.trades !== undefined) html += `<div class="card"><h2>Trades</h2><div class="big">${bt.trades}</div></div>`;
        html += '</div>';
      }
      if (bt.horizons && typeof bt.horizons === 'object') {
        html += '<table><thead><tr><th>Horizon</th><th>Signals</th><th>Direction accuracy</th><th>Avg return / trade</th></tr></thead><tbody>';
        for (const [h, s] of Object.entries(bt.horizons)) {
          html += `<tr><td><strong>${h}-day</strong></td><td>${s.n !== undefined ? s.n : (s.high_conviction_n !== undefined ? s.high_conviction_n : '-')}</td><td>${(s.directional_accuracy !== undefined ? s.directional_accuracy : s.high_conviction_accuracy) ? fmtPct((s.directional_accuracy !== undefined ? s.directional_accuracy : s.high_conviction_accuracy)) : '-'}</td><td>${s.avg_return_per_trade !== undefined ? s.avg_return_per_trade.toFixed(3) + '%' : '-'}</td></tr>`;
        }
        html += '</tbody></table>';
      }
      if (bt.overall) { html += '<pre>' + JSON.stringify(bt.overall, null, 2) + '</pre>'; }
      el.innerHTML = html || 'Backtest summary loaded.';
    }

    async function runPrediction() {
      const btn = $('runBtn');
      btn.disabled = true;
      $('status').textContent = 'Fetching market data and running model (can take 30–90s)...';
      $('status').className = 'status';
      try {
        const data = await api('/predict', 'POST', {});
        renderPrediction(data);
        renderBacktest(data.backtest);
        $('status').textContent = 'Prediction ready.';
      } catch (e) {
        $('status').textContent = 'Error: ' + e.message;
        $('status').className = 'status error';
      } finally {
        btn.disabled = false;
      }
    }

    async function loadBacktest() {
      try { const bt = await api('/backtest/summary'); renderBacktest(bt); }
      catch (e) { $('backtest').textContent = 'Backtest summary unavailable.'; }
    }

    loadBacktest();
  </script>
</body>
</html>
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="SGX G3B (Amundi STI ETF) Multi-Horizon Predictor", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _run_prediction():
    global _latest_prediction, _latest_backtest
    p = _get_predictor()
    pred = p.predict()
    bt = p.backtest(start='2024-01-01', end='2026-08-08', cost_bps=4.0)
    if isinstance(pred, dict) and isinstance(pred.get('horizons'), dict):
        h1 = pred['horizons'].get('1') or pred['horizons'].get(1)
        if h1 is not None and hasattr(h1, '__dict__'):
            h1 = vars(h1)
        if h1:
            pred['ticker'] = 'G3B.SI'
            pred['direction'] = h1.get('direction') or pred.get('direction')
            pred['confidence'] = h1.get('confidence') if h1.get('confidence') is not None else pred.get('confidence')
            pred['bucket_probs'] = h1.get('bucket_probs') or pred.get('bucket_probs')
            pred['bucket'] = h1.get('bucket') or pred.get('bucket')
            pred['reasons'] = h1.get('reasons') or pred.get('reasons', [])
            pred['top_features'] = h1.get('top_features') or pred.get('top_features_global') or pred.get('top_features', [])
            pred['signal_type'] = h1.get('signal_type') or pred.get('signal_type')
            bp = h1.get('bucket_probs') or {}
            proba_up = bp.get('Weak Up', 0) + bp.get('Strong Up', 0) + 0.5 * bp.get('Flat', 0)
            if not bp:
                proba_up = h1.get('model_proba_up')
            pred['proba_up'] = proba_up if proba_up is not None else 0.5
            pred['proba_down'] = 1 - pred['proba_up']
    def _to_dict(x):
        return vars(x) if hasattr(x, '__dict__') else x
    bt_summary = {'start': _to_dict(bt).get('start'), 'end': _to_dict(bt).get('end'), 'cost_bps': _to_dict(bt).get('cost_bps'), 'generated_at': _to_dict(bt).get('generated_at')}
    if isinstance(_to_dict(bt).get('horizons'), dict):
        bt_summary['horizons'] = {}
        for h, s in _to_dict(bt)['horizons'].items():
            sd = _to_dict(s)
            bt_summary['horizons'][h] = {k: _to_dict(v) if k == 'trades' else v for k, v in sd.items()}
            bt_summary['horizons'][h].pop('trades', None)
    else:
        bt_summary['horizons'] = _to_dict(bt).get('horizons')
    # Headline backtest accuracy uses next-day high-conviction signal accuracy when available.
    h1 = (bt_summary.get('horizons') or {}).get(1) or (bt_summary.get('horizons') or {}).get('1') or {}
    bt_summary['accuracy'] = h1.get('high_conviction_accuracy') if h1.get('high_conviction_n', 0) >= 10 else h1.get('directional_accuracy')
    bt_summary['n_samples'] = h1.get('n') if h1 else 0
    bt_summary['trades'] = h1.get('high_conviction_n') if h1 else 0
    return {'prediction': pred, 'backtest': bt_summary}

def _get_backtest_summary():
    p = _get_predictor(); return p.backtest(start='2024-01-01', end='2026-08-08', cost_bps=4.0)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/v1/predict")
async def predict():
    try:
        result = await run_sync(_run_prediction)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=503, detail=result["error"])
        global _latest_prediction, _latest_backtest
        _latest_prediction = result.get("prediction")
        _latest_backtest = result.get("backtest")
        return _clean_json(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/api/v1/backtest/summary")
async def backtest_summary():
    try:
        if _latest_backtest is not None:
            return _clean_json(_latest_backtest)
        bt = await run_sync(_get_backtest_summary)
        if bt is None:
            return {"message": "Run a prediction first to generate the backtest summary."}
        return _clean_json(bt)
    except Exception as exc:
        return {"error": str(exc)}

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML.replace("__TITLE__", "SGX G3B (Amundi STI ETF) Multi-Horizon Predictor")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8060)