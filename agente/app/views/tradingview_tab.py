"""
TradingView Tab — Aba de Market Profile / Value Area via tv.cmd CDP.

Quando TradingView Desktop está aberto com CDP (--remote-debugging-port=9222),
permite consultar dados ao vivo: OHLCV, indicadores, Market Profile (VAH/VAL/POC/VWAP).

Quando offline: mostra painel de status + instrução para conectar.
"""
from __future__ import annotations

import json
import subprocess
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("views.tradingview_tab")


# ── CDP check ────────────────────────────────────────────────────────────────

def _tv_status() -> dict:
    """Chama tv.cmd status com timeout de 6s. Retorna dict com success + error."""
    try:
        result = subprocess.run(
            ["tv.cmd", "status"],
            capture_output=True, text=True, timeout=7,
        )
        return json.loads(result.stdout.strip()) if result.stdout.strip() else {"success": False, "error": result.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _tv_quote(ticker: str) -> dict | None:
    """Busca quote em tempo real via tv.cmd quote."""
    try:
        result = subprocess.run(
            ["tv.cmd", "quote", ticker],
            capture_output=True, text=True, timeout=10,
        )
        if result.stdout.strip():
            return json.loads(result.stdout.strip())
    except Exception:
        pass
    return None


def _tv_values(ticker: str | None = None) -> dict | None:
    """Busca valores do data window (indicadores ativos) via tv.cmd values."""
    try:
        cmd = ["tv.cmd", "values"]
        if ticker:
            cmd += ["--symbol", ticker]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.stdout.strip():
            return json.loads(result.stdout.strip())
    except Exception:
        pass
    return None


def _tv_state() -> dict | None:
    """Estado atual do chart (symbol, TF, studies)."""
    try:
        result = subprocess.run(
            ["tv.cmd", "state"],
            capture_output=True, text=True, timeout=10,
        )
        if result.stdout.strip():
            return json.loads(result.stdout.strip())
    except Exception:
        pass
    return None


# ── Renderers ────────────────────────────────────────────────────────────────

def _offline_panel() -> str:
    return """
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;min-height:400px;gap:24px;padding:48px">
  <div style="font-size:48px">📉</div>
  <div style="font-size:18px;font-weight:700;color:#e2e8f0">TradingView Offline</div>
  <div style="font-size:13px;color:#6b7280;text-align:center;max-width:480px;line-height:1.7">
    Para ativar esta aba, abra o TradingView Desktop com o modo CDP habilitado:
  </div>
  <div style="background:#0d1117;border:1px solid #1e293b;border-radius:8px;padding:16px 24px;font-family:monospace;font-size:13px;color:#38bdf8">
    tv.cmd launch
  </div>
  <div style="font-size:12px;color:#475569;text-align:center;max-width:400px;line-height:1.6">
    Depois rode <code style="color:#818cf8">agente run desk</code> novamente para carregar os dados de Market Profile, VWAP e Value Area.
  </div>
  <div style="display:flex;gap:12px;flex-wrap:wrap;justify-content:center;margin-top:8px">
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:8px 16px;font-size:11px;color:#64748b">
      <span style="color:#ef4444">●</span> CDP desconectado
    </div>
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:8px 16px;font-size:11px;color:#64748b">
      Porta: 9222
    </div>
  </div>
</div>"""


def _state_panel(state: dict) -> str:
    symbol = state.get("symbol", "—")
    tf = state.get("timeframe", "—")
    studies = state.get("studies", [])
    studies_html = "".join(
        f'<span style="background:#1e293b;border-radius:3px;padding:2px 6px;font-size:10px;color:#94a3b8">{s}</span>'
        for s in (studies[:8] if isinstance(studies, list) else [])
    )
    return f"""
<div style="display:flex;gap:12px;align-items:center;padding:12px 20px;background:#0d1117;border-bottom:1px solid #1e293b;flex-wrap:wrap">
  <div style="display:flex;align-items:center;gap:8px">
    <span style="font-size:10px;color:#22c55e">● ONLINE</span>
    <span style="font-size:13px;font-weight:700;color:#e2e8f0">{symbol}</span>
    <span style="font-size:11px;color:#64748b;background:#1e293b;padding:2px 6px;border-radius:3px">{tf}</span>
  </div>
  <div style="display:flex;gap:6px;flex-wrap:wrap">{studies_html}</div>
</div>"""


def _quote_card(ticker: str, q: dict) -> str:
    price = q.get("price") or q.get("last") or q.get("close", 0)
    chg = q.get("change_pct") or q.get("change_percent", 0)
    color = "#22c55e" if (chg or 0) >= 0 else "#ef4444"
    sign = "+" if (chg or 0) >= 0 else ""
    return f"""
<div style="background:#0d1117;border:1px solid #1e293b;border-radius:8px;padding:12px 16px;min-width:140px">
  <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.06em">{ticker}</div>
  <div style="font-size:20px;font-weight:700;color:#e2e8f0;margin:4px 0">{price:,.2f}</div>
  <div style="font-size:12px;color:{color};font-weight:600">{sign}{(chg or 0):.2f}%</div>
</div>"""


def _values_panel(values: dict) -> str:
    """Renderiza data window values — incluindo Market Profile se disponível."""
    if not values:
        return '<div style="color:#64748b;font-size:12px;padding:12px">Sem indicadores no data window.</div>'

    # Procura Market Profile / VWAP / Value Area
    mp_keys = ["vah", "val", "poc", "vwap", "volume_area_high", "volume_area_low",
                "point_of_control", "value_area_high", "value_area_low"]
    mp_data = {k: v for k, v in values.items() if any(mk in k.lower() for mk in mp_keys)}
    other_data = {k: v for k, v in values.items() if k not in mp_data}

    mp_html = ""
    if mp_data:
        rows = "".join(
            f'<tr><td style="color:#64748b;padding:4px 8px;font-size:11px">{k}</td>'
            f'<td style="color:#e2e8f0;padding:4px 8px;font-size:12px;font-weight:600;text-align:right">{v}</td></tr>'
            for k, v in mp_data.items()
        )
        mp_html = f"""
<div style="margin-bottom:16px">
  <div style="font-size:10px;font-weight:700;color:#38bdf8;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">Market Profile · Value Area</div>
  <table style="border-collapse:collapse;width:100%"><tbody>{rows}</tbody></table>
</div>"""

    other_html = ""
    if other_data:
        rows = "".join(
            f'<tr><td style="color:#64748b;padding:3px 8px;font-size:11px">{k}</td>'
            f'<td style="color:#94a3b8;padding:3px 8px;font-size:11px;text-align:right">{v}</td></tr>'
            for k, v in list(other_data.items())[:20]
        )
        other_html = f"""
<div>
  <div style="font-size:10px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">Outros Indicadores</div>
  <table style="border-collapse:collapse;width:100%"><tbody>{rows}</tbody></table>
</div>"""

    return mp_html + other_html


# ── Main render ───────────────────────────────────────────────────────────────

_WATCHLIST = ["SPY", "QQQ", "GLD", "TLT", "BTC-USD", "CL=F", "^VIX"]

def render_tradingview_tab(market_prices: dict | None = None) -> str:
    status = _tv_status()
    online = status.get("success", False)

    if not online:
        return _offline_panel()

    # TV online — coleta dados
    state = _tv_state() or {}
    current_symbol = state.get("symbol", "SPY")

    # Quote do símbolo atual
    q = _tv_quote(current_symbol)

    # Data window values (Market Profile, VWAP, etc.)
    values_raw = _tv_values()
    values = {}
    if isinstance(values_raw, dict):
        values = values_raw.get("values") or values_raw.get("data") or values_raw

    _log.info("tv_tab_data_collected",
              symbol=current_symbol,
              has_quote=q is not None,
              n_values=len(values))

    # Header de status
    state_html = _state_panel(state)

    # Quote card
    quote_html = ""
    if q:
        quote_html = f"""
<div style="padding:16px 20px;border-bottom:1px solid #1e293b">
  <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px">Quote em tempo real</div>
  <div style="display:flex;gap:12px;flex-wrap:wrap">{_quote_card(current_symbol, q)}</div>
</div>"""

    # Values / Market Profile
    values_html = f"""
<div style="padding:16px 20px;border-bottom:1px solid #1e293b">
  <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px">Data Window — {current_symbol}</div>
  {_values_panel(values)}
</div>"""

    # Instrução para Market Profile
    mp_guide = """
<div style="padding:16px 20px">
  <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px">Market Profile · Guia de Uso</div>
  <div style="font-size:12px;color:#475569;line-height:1.8;max-width:600px">
    Para ativar dados de Market Profile no data window:<br>
    1. Abra o TradingView Desktop com layout <strong style="color:#94a3b8">Ultimate Profile</strong><br>
    2. Certifique-se que o indicador <strong style="color:#94a3b8">Volume Profile / Market Profile</strong> está ativo<br>
    3. Os valores VAH, VAL, POC e VWAP aparecerão automaticamente aqui<br>
    4. O filtro de "ficar dentro da value area" usa esses dados para ajustar conviction nas posições
  </div>
  <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">
    <div style="background:#0d1117;border:1px solid #1e293b;border-radius:6px;padding:8px 14px;font-size:11px">
      <span style="color:#22c55e;font-weight:700">VAH</span> <span style="color:#64748b">— Value Area High (70% do volume acima)</span>
    </div>
    <div style="background:#0d1117;border:1px solid #1e293b;border-radius:6px;padding:8px 14px;font-size:11px">
      <span style="color:#ef4444;font-weight:700">VAL</span> <span style="color:#64748b">— Value Area Low (70% do volume abaixo)</span>
    </div>
    <div style="background:#0d1117;border:1px solid #1e293b;border-radius:6px;padding:8px 14px;font-size:11px">
      <span style="color:#f59e0b;font-weight:700">POC</span> <span style="color:#64748b">— Point of Control (maior volume)</span>
    </div>
    <div style="background:#0d1117;border:1px solid #1e293b;border-radius:6px;padding:8px 14px;font-size:11px">
      <span style="color:#38bdf8;font-weight:700">VWAP</span> <span style="color:#64748b">— Volume Weighted Average Price</span>
    </div>
  </div>
</div>"""

    return f"""
<div style="display:flex;flex-direction:column;height:100%">
  {state_html}
  <div style="flex:1;overflow-y:auto">
    {quote_html}
    {values_html}
    {mp_guide}
  </div>
</div>"""
