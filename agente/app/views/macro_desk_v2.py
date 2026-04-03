"""
MacroDesk v2 — Grafo orgânico interativo com Cytoscape.js

Modo único: rede de mercado que se expande como uma molécula.
  • 10 ativos com dados reais são os nós iniciais, conectados pelo MST de Mantegna.
  • Clique [+] num nó para expandir seus filhos na hierarquia (setores, stocks).
  • Clique [-] para recolher.
  • Layout force-directed anima suavemente a cada expansão.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone as _tz
from pathlib import Path
from typing import Any, TYPE_CHECKING

from app.storage.paths import workspace
from app.audit.logger import get_logger

if TYPE_CHECKING:
    from app.models.daily_ingestion_bundle import DailyIngestionBundle
    from app.curation.models import CurationResult

_log = get_logger("views.macro_desk_v2")


# ── Sidebar helpers ───────────────────────────────────────────────────────────

def _score_bar_html(label: str, value: float | None) -> str:
    if value is None:
        return (f'<div class="score-row"><span class="score-label">{label}</span>'
                f'<span class="score-na">—</span></div>')
    pct = (value + 2) / 4 * 100
    color = "#22c55e" if value >= 0.5 else ("#ef4444" if value <= -0.5 else "#f59e0b")
    bar = f'<div class="score-bar-fill" style="width:{pct:.1f}%;background:{color}"></div>'
    sign = "+" if value >= 0 else ""
    return (f'<div class="score-row">'
            f'<span class="score-label">{label}</span>'
            f'<div class="score-bar">{bar}</div>'
            f'<span class="score-val" style="color:{color}">{sign}{value:.2f}</span>'
            f'</div>')


def _regime_badge(regime_data: dict[str, Any]) -> str:
    regime  = regime_data.get("regime", "unknown")
    conf    = regime_data.get("confidence", 0)
    avg_c   = regime_data.get("avg_correlation", 0)
    entropy = regime_data.get("corr_entropy", 0)
    colors  = {
        "risk_on":    ("#065f46", "#4ade80"),
        "risk_off":   ("#7f1d1d", "#f87171"),
        "transition": ("#78350f", "#fcd34d"),
        "chaotic":    ("#581c87", "#c084fc"),
        "unknown":    ("#1f2937", "#9ca3af"),
    }
    bg, fg = colors.get(regime, colors["unknown"])
    icons  = {"risk_on": "&#9650;", "risk_off": "&#9660;", "transition": "&#8596;",
               "chaotic": "~", "unknown": "?"}
    labels = {"risk_on": "RISK ON", "risk_off": "RISK OFF", "transition": "TRANSITION",
               "chaotic": "CHAOTIC", "unknown": "UNKNOWN"}
    return (f'<div class="regime-badge" style="background:{bg};border-color:{fg}">'
            f'<div class="regime-icon" style="color:{fg}">{icons.get(regime,"?")}</div>'
            f'<div><div class="regime-label" style="color:{fg}">{labels.get(regime,regime.upper())}</div>'
            f'<div class="regime-meta">conf {conf:.0%} &middot; &rho;={avg_c:.2f} &middot; H={entropy:.2f}</div>'
            f'</div></div>')


def _flow_panel_html(flow_pred: dict[str, Any]) -> str:
    """
    Sidebar section: Fluxo Mecânico EOD (GEX + LETF Rebalancing).
    Baseado em Barbon et al. — sinais mecânicos previsíveis no close.
    """
    if not flow_pred:
        return '<div style="font-size:11px;color:#64748b">BQL não disponível</div>'

    # Sinal líquido
    direction = flow_pred.get("direction", "flat")
    mag_bn    = flow_pred.get("magnitude_bn", 0.0) or 0.0
    conviction= flow_pred.get("conviction", "low")
    summary   = flow_pred.get("summary", "")
    error     = flow_pred.get("error", "")

    if error and not summary:
        return f'<div style="font-size:11px;color:#f87171">{error}</div>'

    dir_color = {"buy": "#4ade80", "sell": "#f87171", "flat": "#94a3b8"}.get(direction, "#94a3b8")
    dir_arrow = {"buy": "&#9650; BUY", "sell": "&#9660; SELL", "flat": "&#9670; FLAT"}.get(direction, "&#9670;")
    conv_color = {"high": "#fbbf24", "medium": "#60a5fa", "low": "#475569"}.get(conviction, "#475569")

    # GEX
    gex = flow_pred.get("gex", {})
    gex_spx = gex.get("spx", {})
    gex_bn    = gex_spx.get("gex_bn", 0.0) or 0.0
    gamma_reg = gex_spx.get("gamma_regime", "flat")
    flip_lvl  = gex_spx.get("flip_level")
    gex_color = "#4ade80" if gamma_reg == "long" else ("#f87171" if gamma_reg == "short" else "#94a3b8")
    gex_label = {"long": "LONG γ — amortece", "short": "SHORT γ — amplifica", "flat": "NEUTRO"}.get(gamma_reg, gamma_reg)
    flip_str  = f" flip@{flip_lvl:,.0f}" if flip_lvl else ""

    # LETF flows
    letf = flow_pred.get("letf", {})
    spx_flow = (letf.get("spx", {}) or {}).get("flow_usd", 0.0) or 0.0
    ndx_flow = (letf.get("ndx", {}) or {}).get("flow_usd", 0.0) or 0.0
    spx_r    = letf.get("spx_r")
    spx_r_str = f"{spx_r:+.2%}" if spx_r is not None else "—"

    def _flow_row(label: str, usd: float) -> str:
        bn = usd / 1e9
        col = "#4ade80" if bn > 0 else ("#f87171" if bn < 0 else "#94a3b8")
        return (f'<div style="display:flex;justify-content:space-between;font-size:11px;'
                f'padding:1px 0">'
                f'<span style="color:#64748b">{label}</span>'
                f'<span style="color:{col};font-weight:600">{bn:+.2f}B</span></div>')

    # Top members por flow (top 5 buys + top 5 sells)
    per_member = flow_pred.get("per_member", {})
    members_sorted = sorted(
        [(t, v.get("total_usd", 0.0) or 0.0) for t, v in (per_member or {}).items()],
        key=lambda x: -abs(x[1])
    )[:8]
    member_rows = ""
    for ticker, total in members_sorted:
        if abs(total) < 1e5:
            continue
        mn = total / 1e6
        col = "#4ade80" if total > 0 else "#f87171"
        arr = "&#9650;" if total > 0 else "&#9660;"
        member_rows += (f'<div style="display:flex;justify-content:space-between;'
                        f'font-size:10px;padding:0.5px 0">'
                        f'<span style="color:#64748b">{ticker}</span>'
                        f'<span style="color:{col}">{arr} ${mn:+.1f}M</span></div>')

    return f"""
<div style="margin-bottom:6px">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="color:{dir_color};font-size:12px;font-weight:800">{dir_arrow}</span>
    <span style="color:{dir_color};font-size:10px">${mag_bn:+.2f}B</span>
    <span style="font-size:10px;color:{conv_color};border:1px solid {conv_color};
                 border-radius:3px;padding:0 4px">{conviction.upper()}</span>
  </div>
</div>
<div style="font-size:10px;color:#64748b;margin-bottom:4px">SPX retorno: {spx_r_str}</div>

<div style="font-size:11px;color:#475569;margin:4px 0 2px;font-weight:600">GEX (Barbon Γ^HP)</div>
<div style="font-size:11px;color:{gex_color}">{gex_label}{flip_str} ({gex_bn:+.1f}B)</div>

<div style="font-size:11px;color:#475569;margin:4px 0 2px;font-weight:600">LETF Rebal (Ω^LETF)</div>
{_flow_row("SPX (SPXL/SPXS)", spx_flow)}
{_flow_row("NDX (TQQQ/SQQQ)", ndx_flow)}

{('<div style="font-size:11px;color:#475569;margin:4px 0 2px;font-weight:600">Top membros</div>' + member_rows) if member_rows else ""}
"""


def _live_network_html(live: dict[str, Any]) -> str:
    """Sidebar section: SRI gauge + top contagion sources + corr breaks."""
    if not live:
        return '<div style="font-size:11px;color:#64748b">—</div>'

    sri       = live.get("sri", 0.0)
    sri_lbl   = live.get("sri_label", "—")
    sri_col   = live.get("sri_color", "#6b7280")
    top       = live.get("top_sources", [])
    rolling   = live.get("rolling_corr", [])

    # ── SRI gauge ────────────────────────────────────────────────────────────
    pct = int(sri * 100)
    gauge = (
        f'<div style="display:flex;justify-content:space-between;font-size:11px">'
        f'<span style="color:{sri_col};font-weight:700">{sri_lbl}</span>'
        f'<span style="color:#64748b">{sri:.3f}</span></div>'
        f'<div class="sri-bar"><div class="sri-fill" style="width:{pct}%;background:{sri_col}"></div></div>'
        f'<div class="sri-labels"><span>0</span><span>0.25</span><span>0.50</span><span>0.75</span><span>1</span></div>'
    )

    # ── Top contagion sources ─────────────────────────────────────────────────
    source_rows = ""
    for s in top[:5]:
        sc  = s.get("score", 0)
        tk  = s.get("ticker") or s.get("node_id", "")
        bar_w = int(sc * 100)
        bar_c = "#f87171" if sc > 0.5 else ("#f59e0b" if sc > 0.2 else "#4ade80")
        source_rows += (
            f'<div class="nd-row">'
            f'<span class="nd-key">{tk[:10]}</span>'
            f'<div style="flex:1;margin:0 4px;height:4px;background:#1a2535;border-radius:2px">'
            f'<div style="width:{bar_w}%;height:100%;background:{bar_c};border-radius:2px"></div></div>'
            f'<span class="nd-val" style="color:{bar_c}">{sc:.2f}</span>'
            f'</div>'
        )
    if not source_rows:
        source_rows = '<div style="font-size:11px;color:#475569">Sem dados MST</div>'

    # ── Rolling corr breaks (top 4 por divergência) ───────────────────────────
    corr_rows = ""
    for rc in rolling[:4]:
        cs   = rc.get("corr_short", 0)
        cl   = rc.get("corr_long",  0)
        div  = rc.get("divergence", 0)
        lbl  = rc.get("label", rc.get("pair", ""))[:18]
        trend = rc.get("trend", "stable")
        arrow = "↑" if trend == "rising" else ("↓" if trend == "falling" else "→")
        ac    = "#4ade80" if trend == "rising" else ("#f87171" if trend == "falling" else "#475569")
        cs_c  = "#4ade80" if cs > 0 else "#f87171"
        corr_rows += (
            f'<div class="corr-row">'
            f'<span class="corr-label">{lbl}</span>'
            f'<span class="corr-vals">'
            f'<span style="color:{cs_c}">{cs:+.2f}</span>'
            f'<span class="corr-arrow" style="color:{ac}">{arrow}</span>'
            f'<span style="color:#64748b">{cl:+.2f}</span>'
            f'</span></div>'
        )
    if not corr_rows:
        corr_rows = '<div style="font-size:11px;color:#475569">—</div>'

    return (
        f'<div style="font-size:10px;font-weight:700;color:#64748b;letter-spacing:1px;margin-bottom:3px">SRI</div>'
        f'{gauge}'
        f'<div style="font-size:10px;font-weight:700;color:#64748b;letter-spacing:1px;margin:6px 0 3px">TOP CONTÁGIO</div>'
        f'{source_rows}'
        f'<div style="font-size:10px;font-weight:700;color:#64748b;letter-spacing:1px;margin:6px 0 3px">'
        f'CORR 20d → 60d</div>'
        f'{corr_rows}'
    )


def _risk_heatmap_html(graph_data: dict[str, Any]) -> str:
    """
    Mini heatmap de tail score para os nós com dados probabilísticos.
    Células coloridas de verde (score=0) a vermelho (score=1).
    """
    nodes = graph_data.get("elements", {}).get("nodes", [])
    cells = []
    for n in nodes:
        d = n.get("data", {})
        prob = d.get("prob") or {}
        ts   = prob.get("tail_score")
        if ts is None:
            continue
        label = d.get("label", d.get("id", ""))[:8]
        # Interpola verde → amarelo → vermelho
        if ts <= 0.5:
            r = int(ts * 2 * 255)
            g = 200
        else:
            r = 200
            g = int((1 - ts) * 2 * 200)
        color = f"rgb({r},{g},40)"
        cells.append(
            f'<div class="risk-cell" style="background:{color}" '
            f'data-tip="{label}: {ts:.2f}"></div>'
        )
    if not cells:
        return '<div style="font-size:11px;color:#64748b">—</div>'
    return f'<div class="risk-grid">{"".join(cells[:30])}</div>'


def _vix_term_html(vix_term: dict[str, Any]) -> str:
    if not vix_term:
        return '<div style="font-size:11px;color:#64748b">—</div>'
    vix9d  = vix_term.get("vix9d")
    vix30d = vix_term.get("vix30d")
    vix3m  = vix_term.get("vix3m")
    slope  = vix_term.get("ts_slope")
    contango = vix_term.get("contango")

    def vv(v: float | None) -> str:
        return f"{v:.1f}" if v is not None else "—"

    slope_color = "#4ade80" if contango else "#f87171"
    slope_str   = (f'<span style="color:{slope_color}">{slope:+.1%}</span>' if slope is not None else "—")
    ct_label    = "Contango" if contango else "Backwdtn"
    ct_color    = "#4ade80" if contango else "#f87171"

    rows = (
        f'<div class="nd-row"><span class="nd-key">VIX9D</span><span class="nd-val">{vv(vix9d)}</span></div>'
        f'<div class="nd-row"><span class="nd-key">VIX30D</span><span class="nd-val">{vv(vix30d)}</span></div>'
        f'<div class="nd-row"><span class="nd-key">VIX3M</span><span class="nd-val">{vv(vix3m)}</span></div>'
        f'<div class="nd-row"><span class="nd-key">TS Slope</span><span class="nd-val">{slope_str}</span></div>'
        f'<div class="nd-row"><span class="nd-key">Structure</span>'
        f'<span class="nd-val" style="color:{ct_color}">{ct_label}</span></div>'
    )
    return f'<div id="nd-rows">{rows}</div>'


def _market_table_rows(market_prices: dict[str, Any]) -> str:
    priority = ["^GSPC","^NDX","^RUT","^VIX","TLT","HYG","GLD","CL=F","BTC-USD","DX-Y.NYB"]
    rows = []
    for sym in priority:
        mp = market_prices.get(sym)
        if not mp:
            continue
        price = mp.get("price")
        daily = mp.get("daily_return")
        name  = (mp.get("name") or sym).split("(")[0].strip()[:18]
        ps    = f"{price:,.2f}" if price else "—"
        if daily is not None:
            color = "#4ade80" if daily >= 0 else "#f87171"
            sign  = "+" if daily >= 0 else ""
            ds = f'<span style="color:{color}">{sign}{daily*100:.2f}%</span>'
        else:
            ds = '<span style="color:#6b7280">—</span>'
        rows.append(
            f'<tr><td class="mp-name">{name}</td>'
            f'<td class="mp-price">{ps}</td>'
            f'<td class="mp-ret">{ds}</td></tr>'
        )
    return "\n".join(rows)


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #060a12;
       color: #e2e8f0; height: 100vh; overflow: hidden;
       display: flex; flex-direction: column; font-size: 12px; }

#topbar { display: flex; align-items: center; gap: 10px; padding: 8px 16px;
          background: #0a0f1a; border-bottom: 1px solid #1a2535;
          flex-shrink: 0; height: 52px; }
.brand { font-size: 14px; font-weight: 700; color: #38bdf8;
         letter-spacing: 2.5px; white-space: nowrap; }
.run-date { font-size: 11px; color: #64748b; white-space: nowrap; }
.spacer { flex: 1; }
.ctrl-btn { padding: 4px 12px; border-radius: 5px; border: 1px solid #2d3f55;
            background: transparent; color: #64748b; cursor: pointer;
            font-size: 12px; font-weight: 600; transition: all 0.15s;
            white-space: nowrap; }
.ctrl-btn:hover { border-color: #38bdf8; color: #94a3b8; }
.ctrl-btn.danger:hover { border-color: #ef4444; color: #ef4444; }
.sep { width: 1px; height: 20px; background: #2d3f55; }
#expand-count { font-size: 11px; color: #4a6380; min-width: 60px; text-align: right; }

#desk-view { flex-direction: row; }
#main { display: flex; flex: 1; overflow: hidden; width: 100%; }

#sidebar { width: 240px; min-width: 240px; flex-shrink: 0; background: #0a0f1a;
           border-right: 1px solid #1a2535; overflow-y: auto; padding: 10px;
           display: flex; flex-direction: column; gap: 10px; }
.s-section { border: 1px solid #1e2d42; border-radius: 5px; padding: 10px; }
.s-title { font-size: 10px; font-weight: 700; color: #4a6380;
           letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px; }

.regime-badge { display: flex; align-items: center; gap: 8px;
                padding: 7px 8px; border-radius: 5px; border: 1px solid; }
.regime-icon { font-size: 18px; font-weight: 700; }
.regime-label { font-size: 12px; font-weight: 700; }
.regime-meta { font-size: 11px; color: #9ca3af; margin-top: 1px; }

.score-row { display: flex; align-items: center; gap: 5px; margin-bottom: 5px; }
.score-label { width: 80px; font-size: 11px; color: #94a3b8; flex-shrink: 0; }
.score-bar { flex: 1; height: 6px; background: #1a2535; border-radius: 3px; overflow: hidden; }
.score-bar-fill { height: 100%; border-radius: 3px; }
.score-val { width: 36px; font-size: 11px; text-align: right; font-weight: 700; flex-shrink: 0; }
.score-na { font-size: 11px; color: #4a5568; }

.mp-table { width: 100%; border-collapse: collapse; font-size: 11px; }
.mp-table tr { border-bottom: 1px solid #0d1520; }
.mp-name { color: #4b6278; padding: 2px 0; }
.mp-price { text-align: right; color: #94a3b8; font-variant-numeric: tabular-nums; padding: 2px 3px; }
.mp-ret { text-align: right; font-variant-numeric: tabular-nums; padding: 2px 0; }

.net-meta { font-size: 11px; color: #64748b; line-height: 1.9; }
.net-meta strong { color: #4b5563; }

#cy-container { flex: 1; position: relative; background: #060a12; }
#cy { width: 100%; height: 100%; }

/* Node detail panel */
#node-detail { position: absolute; top: 10px; right: 10px; width: 224px;
               background: #0a0f1a; border: 1px solid #1a2535;
               border-radius: 7px; padding: 10px 10px 8px; display: none; z-index: 200; }
#node-detail.visible { display: block; }
#nd-close { position: absolute; top: 6px; right: 8px; cursor: pointer;
            color: #64748b; font-size: 15px; line-height: 1; }
#nd-close:hover { color: #94a3b8; }
#nd-label { font-size: 13px; font-weight: 700; margin-bottom: 2px; padding-right: 16px; }
#nd-sub { font-size: 11px; color: #64748b; margin-bottom: 7px; }

/* Anatomy tabs */
.nd-tabs { display: flex; gap: 3px; margin-bottom: 9px; }
.nd-tab { font-size: 10px; font-weight: 700; padding: 3px 9px;
          border-radius: 3px; border: 1px solid #1e293b;
          background: transparent; color: #64748b; cursor: pointer;
          letter-spacing: 0.5px; }
.nd-tab.active { background: #0f172a; border-color: #38bdf8; color: #38bdf8; }
.nd-tab-panel { display: none; }
.nd-tab-panel.active { display: block; }

#nd-rows .nd-row, .nd-tab-panel .nd-row {
  display: flex; justify-content: space-between;
  border-bottom: 1px solid #0d1520; padding: 4px 0; font-size: 12px; }
#nd-rows .nd-key, .nd-tab-panel .nd-key { color: #64748b; }
#nd-rows .nd-val, .nd-tab-panel .nd-val { font-weight: 700; color: #e2e8f0; }
.nd-na { color: #64748b; font-size: 11px; }

/* Hint bar */
#hint-bar { position: absolute; bottom: 10px; left: 10px; right: 10px;
            display: flex; gap: 14px; align-items: center;
            background: rgba(10,15,26,0.92); border: 1px solid #2d3f55;
            border-radius: 5px; padding: 6px 12px; font-size: 12px;
            color: #64748b; pointer-events: none; }
.leg { display: flex; align-items: center; gap: 5px; }
.leg-dot { width: 9px; height: 9px; border-radius: 50%; }
.leg-line { width: 16px; height: 2px; border-radius: 1px; }
#hint-text { flex: 1; text-align: right; color: #4a6380; font-size: 11px; }

/* Contagion pulse animation */
@keyframes contagion-pulse {
  0%   { box-shadow: 0 0 0 0 rgba(239,68,68,0.55); }
  70%  { box-shadow: 0 0 0 9px rgba(239,68,68,0); }
  100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
}
.contagion-high { animation: contagion-pulse 1.6s ease-out infinite; border-radius: 50%; }

/* SRI gauge */
.sri-bar { height: 6px; background: #1a2535; border-radius: 3px; overflow: hidden; margin: 4px 0 2px; }
.sri-fill { height: 100%; border-radius: 3px; transition: width 0.4s; }
.sri-labels { display: flex; justify-content: space-between; font-size: 10px; color: #64748b; }

/* Rolling corr table */
.corr-row { display: flex; justify-content: space-between; align-items: center;
            border-bottom: 1px solid #0d1520; padding: 2px 0; font-size: 11px; }
.corr-label { color: #4b6278; flex: 1; }
.corr-vals  { display: flex; gap: 5px; align-items: center; }
.corr-arrow { font-size: 10px; }

/* Risk heatmap */
.risk-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 2px; margin-top: 4px; }
.risk-cell { height: 14px; border-radius: 2px; position: relative; cursor: default; }
.risk-cell:hover::after { content: attr(data-tip);
  position: absolute; bottom: 18px; left: 50%; transform: translateX(-50%);
  background: #0f172a; border: 1px solid #1e293b; color: #94a3b8;
  font-size: 10px; padding: 2px 4px; border-radius: 3px; white-space: nowrap;
  pointer-events: none; z-index: 999; }

::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: #0a0f1a; }
::-webkit-scrollbar-thumb { background: #1a2535; border-radius: 2px; }

/* Hover tooltip */
#cy-tooltip {
  position: absolute; display: none; pointer-events: none; z-index: 500;
  background: #0c1624; border: 1px solid #1e3a5f; border-radius: 6px;
  padding: 7px 10px; font-size: 10px; min-width: 128px;
  box-shadow: 0 6px 20px rgba(0,0,0,0.6);
}
.tt-name { font-weight: 700; color: #e2e8f0; margin-bottom: 4px; font-size: 11px; }
.tt-ticker { font-size: 11px; color: #38bdf8; margin-bottom: 3px; }
.tt-row  { display: flex; justify-content: space-between; gap: 10px;
           color: #475569; font-size: 11px; padding: 1.5px 0; border-bottom: 1px solid #0d1520; }
.tt-row:last-child { border-bottom: none; }
.tt-val  { font-weight: 600; color: #94a3b8; }

/* Search input */
#search-input {
  background: #0d1520; border: 1px solid #1e293b; border-radius: 4px;
  color: #94a3b8; font-size: 10px; padding: 3px 8px; width: 110px;
  outline: none;
}
#search-input:focus { border-color: #38bdf8; }
#search-input::placeholder { color: #64748b; }

/* Corr threshold slider */
#corr-slider {
  -webkit-appearance: none; width: 76px; height: 3px;
  border-radius: 2px; background: #1a2535; outline: none; cursor: pointer;
  vertical-align: middle;
}
#corr-slider::-webkit-slider-thumb {
  -webkit-appearance: none; width: 10px; height: 10px;
  border-radius: 50%; background: #38bdf8; cursor: pointer;
}
#corr-label { font-size: 11px; color: #64748b; min-width: 28px; }

/* Color mode buttons */
.mode-btn { padding: 4px 12px; border-radius: 3px; border: 1px solid #2d3f55;
            background: transparent; color: #94a3b8; cursor: pointer;
            font-size: 12px; font-weight: 700; letter-spacing: 0.5px;
            transition: all 0.15s; }
.mode-btn:hover { border-color: #38bdf8; color: #cbd5e1; }
.mode-btn.active { background: #0f172a; border-color: #38bdf8; color: #38bdf8; }

/* Isolated mode banner */
#iso-banner {
  position: absolute; top: 8px; left: 50%; transform: translateX(-50%);
  background: rgba(56,189,248,0.1); border: 1px solid #1e3a5f;
  border-radius: 4px; padding: 3px 12px; font-size: 11px; color: #38bdf8;
  pointer-events: none; display: none; z-index: 100; white-space: nowrap;
}

/* Keyboard hint */
#kbd-hints { position: absolute; bottom: 10px; right: 10px;
             font-size: 10px; color: #4a6380; pointer-events: none;
             line-height: 1.7; text-align: right; }

/* ── Top-level main tabs ─────────────────────────────────────────────────── */
#main-tabs-bar { display: flex; gap: 2px; align-items: center;
                 padding: 0 14px; background: #070c17;
                 border-bottom: 1px solid #2d3f55; flex-shrink: 0; height: 36px; }
.main-tab { padding: 5px 20px; border-radius: 4px 4px 0 0;
            border: 1px solid transparent; border-bottom: none;
            background: transparent; color: #64748b; cursor: pointer;
            font-size: 13px; font-weight: 700; letter-spacing: 0.8px;
            text-transform: uppercase; transition: all 0.15s; }
.main-tab:hover { color: #cbd5e1; border-color: #64748b; }
.main-tab.active { background: #060a12; border-color: #2d4a6f;
                   color: #38bdf8; }
.main-view { display: none; flex: 1; overflow: hidden; min-height: 0; }
.main-view.active { display: flex; }

/* ── Portfolio tab content ───────────────────────────────────────────────── */
#portfolio-view { flex-direction: column; overflow-y: auto;
                  padding: 18px 24px; gap: 16px; background: #060a12; }
.pt-section { background: #0a0f1a; border: 1px solid #1a2535;
              border-radius: 7px; padding: 14px; }
.pt-title { font-size: 11px; font-weight: 700; color: #64748b;
            letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 10px; }
.pt-badge { display: inline-block; padding: 3px 10px; border-radius: 5px;
            font-size: 11px; font-weight: 700; border: 1px solid; }
.pt-table { width: 100%; border-collapse: collapse; font-size: 11px; }
.pt-table th { background: #070c17; color: #64748b; font-weight: 700;
               padding: 5px 8px; text-align: left; border-bottom: 1px solid #1a2535;
               font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;
               position: sticky; top: 0; }
.pt-table td { padding: 6px 8px; border-bottom: 1px solid #0d1520;
               vertical-align: top; }
.pt-table tr:hover td { background: #0d1520; }
.pt-dir-long  { color: #10b981; font-weight: 700; }
.pt-dir-short { color: #ef4444; font-weight: 700; }
.pt-empty { color: #64748b; font-size: 12px; text-align: center; padding: 40px; }
.gex-bar { display: flex; gap: 12px; flex-wrap: wrap; align-items: center;
           padding: 10px 14px; background: #070c17; border-radius: 5px;
           border: 1px solid #1a2535; margin-bottom: 12px; font-size: 11px; }
.gex-item { display: flex; flex-direction: column; gap: 2px; }
.gex-lbl { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
.gex-val { font-weight: 700; }
"""


# ── JavaScript (raw string — sem escaping {{ }}) ──────────────────────────────

_JS = r"""
const GD = __GRAPH_DATA_JSON__;

// ── Mapas de lookup ───────────────────────────────────────────────────────────
const N = {};           // node_id → data
const CHILDREN = {};    // node_id → [child_id, ...]
GD.elements.nodes.forEach(n => {
  N[n.data.id] = n.data;
  const pid = n.data.parent_id;
  if (pid) {
    if (!CHILDREN[pid]) CHILDREN[pid] = [];
    CHILDREN[pid].push(n.data.id);
  }
});

// IDs dos nós com dados de mercado (camada base da rede)
const DATA_IDS = new Set(GD.elements.nodes.filter(n => n.data.has_data).map(n => n.data.id));

// ── Estado de visualização ─────────────────────────────────────────────────
let colorMode  = 'default';  // 'default' | 'risk' | 'regime'
let corrThresh = 0.3;        // |correlation| mínimo para exibir aresta MST
let isolateId  = null;       // node ID em isolamento (right-click)

// ── Color helper ──────────────────────────────────────────────────────────
function getNodeBg(d) {
  if (colorMode === 'risk') {
    const ts = d.prob && d.prob.tail_score;
    if (ts != null) {
      const t = Math.min(1, Math.max(0, ts));
      const r = Math.round(t * 200 + 40);
      const g = Math.round((1 - t) * 150 + 20);
      return 'rgb(' + r + ',' + g + ',40)';
    }
  }
  if (colorMode === 'regime') {
    const bull = d.prob && d.prob.regime_prob_bull;
    if (bull != null) {
      const t = Math.min(1, Math.max(0, bull));
      const r = Math.round((1 - t) * 200 + 20);
      const g = Math.round(t * 150 + 20);
      return 'rgb(' + r + ',' + g + ',50)';
    }
  }
  return (d.has_data && d.bg_color) ? d.bg_color : (d.color || '#374151');
}

// ── Estado — pilha de navegação ───────────────────────────────────────────────
// Cada elemento da pilha = { nodeId, label }
// Pilha vazia = modo rede (todos os data nodes + MST)
// Pilha = [spx]         = universo do SPX (setores)
// Pilha = [spx, energy] = universo de Energy (stocks)
const navStack = [];

function currentFocus() {
  return navStack.length > 0 ? navStack[navStack.length - 1].nodeId : null;
}

// ── Breadcrumb ────────────────────────────────────────────────────────────────
function updateBreadcrumb() {
  const parts = ['Network', ...navStack.map(f => f.label)];
  document.getElementById('expand-count').textContent = parts.join(' \u203a ');
  document.getElementById('btn-back').style.display = navStack.length > 0 ? 'inline-block' : 'none';
}

// ══════════════════════════════════════════════════════════════════════════════
// ELEMENTOS VISÍVEIS
// ══════════════════════════════════════════════════════════════════════════════

function currentElements() {
  const focus = currentFocus();

  if (!focus) {
    // ── Modo rede: data nodes + MST ──────────────────────────────────────────
    // MST é o backbone — exibe todas as arestas independente de corrThresh
    const mst = GD.elements.edges.filter(e =>
      e.data.type === 'mst'
      && DATA_IDS.has(e.data.source)
      && DATA_IDS.has(e.data.target)
    );
    // Só mostrar nós que aparecem em pelo menos uma aresta MST
    const mstNodeIds = new Set();
    mst.forEach(e => { mstNodeIds.add(e.data.source); mstNodeIds.add(e.data.target); });

    // Se isolamento ativo: manter só nó + 1-hop
    const isoNeighbors = isolateId != null
      ? new Set(mst.filter(e => e.data.source === isolateId || e.data.target === isolateId)
          .flatMap(e => [e.data.source, e.data.target]))
      : null;

    const nodes = GD.elements.nodes.filter(n => {
      if (!n.data.has_data) return false;
      if (!mstNodeIds.has(n.data.id)) return false;  // remove nós sem conexão MST
      if (isoNeighbors != null) return isoNeighbors.has(n.data.id);
      return true;
    }).map(n => {
      const d = Object.assign({}, n.data);
      delete d.parent;
      d._bg = getNodeBg(d);
      if ((CHILDREN[d.id] || []).length > 0) {
        d.expandable = true;
        d.label = '[+] ' + (n.data.label || n.data.id);
      }
      return { data: d };
    });
    return [...nodes, ...mst];
  }

  // ── Drill-down: nó atual + filhos diretos ─────────────────────────────────
  const kids = CHILDREN[focus] || [];
  const visible = [focus, ...kids];

  const nodes = visible.map(id => {
    const src = N[id];
    if (!src) return null;
    const d = Object.assign({}, src);
    delete d.parent;
    d._bg = getNodeBg(d);

    if (id === focus) {
      // Raiz da camada atual: clique volta para camada anterior
      d.isFocusRoot = true;
      d.label = '\u21a9 ' + (src.label || id);
    } else if ((CHILDREN[id] || []).length > 0) {
      // Filho com filhos: clique entra na próxima camada
      d.expandable = true;
      d.label = '[+] ' + (src.label || id);
    }
    return { data: d };
  }).filter(Boolean);

  // Arestas da raiz para cada filho
  const edges = kids.map(cid => ({
    data: { id: 'h_' + focus + '_' + cid, source: focus, target: cid, type: 'hierarchy' }
  }));

  return [...nodes, ...edges];
}

// ══════════════════════════════════════════════════════════════════════════════
// ESTILOS
// ══════════════════════════════════════════════════════════════════════════════

const STYLE = [
  // ── Base: todos os nós são círculos, tamanho data-driven ──────────────────
  { selector: 'node', style: {
    'shape': 'ellipse',
    'background-color': 'data(_bg)',
    'label': 'data(label)',
    'color': '#f1f5f9',
    'font-size': 9,
    'font-weight': 'bold',
    'text-valign': 'center',
    'text-halign': 'center',
    'text-wrap': 'wrap',
    'text-max-width': '64px',
    'text-background-color': '#060a12',
    'text-background-opacity': 0.78,
    'text-background-padding': '2px',
    'border-width': 1.5,
    'border-color': 'data(border_color)',
    'width': 'data(size)', 'height': 'data(size)',
    'transition-property': 'background-color, border-color, width, height',
    'transition-duration': '0.3s',
  } },

  // ── Nós com dados de mercado (rede base) ───────────────────────────────────
  { selector: 'node[?has_data]', style: {
    'font-size': 10, 'font-weight': 'bold',
    'border-width': 2,
    'z-index': 20,
    'text-background-opacity': 0.88,
  } },

  // ── Hub: borda dourada ────────────────────────────────────────────────────
  { selector: 'node[?is_hub]', style: {
    'border-color': '#f59e0b',
    'border-width': 3,
  } },

  // ── Contagion: borda laranja/vermelha pulsante para alto contágio ──────────
  { selector: 'node[?has_data]', style: {
    'border-color': function(n) {
      const c = n.data('contagion');
      if (c == null) return n.data('border_color') || '#6b7280';
      if (c > 0.5)  return '#ef4444';
      if (c > 0.25) return '#f97316';
      return n.data('border_color') || '#6b7280';
    },
    'border-width': function(n) {
      const c = n.data('contagion');
      if (c > 0.5)  return 4;
      if (c > 0.25) return 3;
      return 2;
    },
  } },

  // ── Level 4 (setores) — círculo, tamanho dinâmico ─────────────────────────
  { selector: 'node[level=4]', style: {
    'font-size': 8,
    'text-background-opacity': 0.75,
    'color': '#ffffff',
    'border-width': 1.5,
    'z-index': 5,
  } },

  // ── Level 5 (stocks/assets) ───────────────────────────────────────────────
  { selector: 'node[level=5][!has_data]', style: {
    'font-size': 7,
    'font-weight': 'normal',
    'border-width': 1,
    'text-background-opacity': 0.7,
    'z-index': 3,
  } },

  // ── Expandível ────────────────────────────────────────────────────────────
  { selector: 'node[?expandable]', style: { 'cursor': 'pointer' } },

  // ── Raiz da camada atual (clique volta) ───────────────────────────────────
  { selector: 'node[?isFocusRoot]', style: {
    'font-size': 11, 'font-weight': 'bold',
    'border-color': '#38bdf8', 'border-width': 3,
    'cursor': 'pointer',
    'z-index': 30,
  } },

  // ── Selecionado ───────────────────────────────────────────────────────────
  { selector: 'node:selected', style: {
    'border-color': '#fbbf24', 'border-width': 3, 'overlay-opacity': 0,
  } },

  // ── MST edges ─────────────────────────────────────────────────────────────
  { selector: 'edge[type="mst"]', style: {
    'line-color': 'data(color)',
    'width': 'data(width)',
    'curve-style': 'straight',
    'opacity': 0.9,
    'z-index': 15,
    'label': function(e) {
      const c = e.data('correlation');
      return c != null ? c.toFixed(2) : '';
    },
    'font-size': 8,
    'color': '#475569',
    'text-background-color': '#060a12',
    'text-background-opacity': 0.75,
    'text-background-padding': '1px',
  } },

  // ── Hierarchy edges ────────────────────────────────────────────────────────
  { selector: 'edge[type="hierarchy"]', style: {
    'line-color': '#1e3a5f',
    'width': 1,
    'curve-style': 'bezier',
    'opacity': 0.5,
    'target-arrow-shape': 'none',
    'z-index': 2,
  } },

  { selector: 'node:active', style: { 'overlay-opacity': 0.07 } },
];

// ══════════════════════════════════════════════════════════════════════════════
// INIT CYTOSCAPE
// ══════════════════════════════════════════════════════════════════════════════

// Inicializa vazio — rebuild() vai popular logo abaixo
const cy = cytoscape({
  container: document.getElementById('cy'),
  elements: [],
  style: STYLE,
  minZoom: 0.04, maxZoom: 10, wheelSensitivity: 0.18,
  boxSelectionEnabled: false,
});

// ══════════════════════════════════════════════════════════════════════════════
// REBUILD — reconstrói o grafo e faz fit
// ══════════════════════════════════════════════════════════════════════════════

function rebuild() {
  cy.elements().remove();
  cy.add(currentElements());

  // Layout diferente por nível
  const focus = currentFocus();
  const layoutOpts = focus
    ? {   // Drill-down: força-dirigida para poucos nós
        name: 'cose', animate: true, animationDuration: 420,
        randomize: true,
        nodeRepulsion: function(n) { return n.data('isFocusRoot') ? 90000 : 18000; },
        idealEdgeLength: 120, edgeElasticity: 220,
        gravity: 1.8, numIter: 400,
        fit: true, padding: 80,
      }
    : {   // Modo rede: cose built-in (sem CDN, roda offline)
        name: 'cose',
        animate: true, animationDuration: 500, randomize: true,
        nodeRepulsion: 35000,
        idealEdgeLength: 100,
        edgeElasticity: 100,
        nestingFactor: 5,
        gravity: 60,
        numIter: 500,
        initialTemp: 200,
        coolingFactor: 0.95,
        minTemp: 1.0,
        fit: true, padding: 60,
      };

  // Fallback para cose se fcose não estiver disponível
  function runLayout(opts) {
    try {
      cy.layout(opts).run();
    } catch(e) {
      cy.layout({
        name: 'cose', animate: true, animationDuration: 600,
        randomize: true,
        nodeRepulsion: 45000,
        idealEdgeLength: 100,
        edgeElasticity: 100,
        nestingFactor: 5,
        gravity: 80,
        numIter: 1000,
        initialTemp: 200,
        coolingFactor: 0.95,
        minTemp: 1.0,
        fit: true, padding: 70,
      }).run();
    }
  }
  runLayout(layoutOpts);
  updateBreadcrumb();
}

// ══════════════════════════════════════════════════════════════════════════════
// EVENTOS
// ══════════════════════════════════════════════════════════════════════════════

cy.on('tap', 'node', e => {
  const d    = e.target.data();
  const focus = currentFocus();

  if (d.isFocusRoot) {
    // ── Clicou na raiz da camada atual → volta uma camada ─────────────────
    navStack.pop();
    rebuild();
    return;
  }

  const kids = CHILDREN[d.id] || [];
  if (kids.length > 0) {
    // ── Entra na próxima camada (drill-down) ──────────────────────────────
    navStack.push({ nodeId: d.id, label: N[d.id] ? (N[d.id].label || d.id) : d.id });
    rebuild();
    return;
  }

  // ── Nó folha → painel de detalhe ─────────────────────────────────────────
  showDetail(d);
});

cy.on('tap', e => {
  if (e.target === cy) {
    document.getElementById('node-detail').classList.remove('visible');
  }
});

// ── Hover: tooltip rico + dim ────────────────────────────────────────────
const CY_CONT = document.getElementById('cy-container');
const TT = document.getElementById('cy-tooltip');

function fmtP(v) {
  if (v == null) return null;
  const c = v >= 0 ? '#4ade80' : '#f87171';
  return '<span style="color:' + c + '">' + (v >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%</span>';
}
function ttRow(label, val) {
  return '<div class="tt-row"><span>' + label + '</span><span class="tt-val">' + val + '</span></div>';
}

cy.on('mouseover', 'node', e => {
  const d   = e.target.data();
  const box = CY_CONT.getBoundingClientRect();
  const ev  = e.originalEvent;

  let html = '<div class="tt-name">' + (d.label || d.id).replace(/^\[.\] /, '') + '</div>';
  if (d.ticker) html += '<div class="tt-ticker">' + d.ticker + '</div>';
  if (d.price  != null) html += ttRow('Price', d.price.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2}));
  if (d.daily  != null) html += ttRow('1D', fmtP(d.daily) || '—');
  if (d.weekly != null) html += ttRow('1W', fmtP(d.weekly) || '—');
  const ann_vol = d.prob && d.prob.ann_vol;
  if (ann_vol   != null) html += ttRow('Vol ann.', (ann_vol * 100).toFixed(1) + '%');
  const pe = d.anatomy && d.anatomy.pe;
  if (pe        != null) html += ttRow('P/E', pe.toFixed(1) + 'x');
  const mktcap = d.anatomy && d.anatomy.mktcap_b;
  if (mktcap    != null) html += ttRow('Mkt Cap', mktcap >= 1000 ? '$' + (mktcap/1000).toFixed(1) + 'T' : '$' + mktcap.toFixed(0) + 'B');
  if (d.contagion != null && d.contagion > 0.01) {
    const cc = d.contagion > 0.5 ? '#ef4444' : '#f97316';
    html += ttRow('Contagion', '<span style="color:' + cc + '">' + d.contagion.toFixed(3) + '</span>');
  }

  TT.innerHTML = html;
  TT.style.display = 'block';
  const mx = (ev.clientX - box.left) + 16;
  const my = (ev.clientY - box.top)  - 10;
  TT.style.left = (mx + 150 > box.width ? mx - 166 : mx) + 'px';
  TT.style.top  = (my + 140 > box.height ? my - 145 : my) + 'px';

  cy.elements().not(e.target.closedNeighborhood()).style('opacity', 0.18);
});
cy.on('mousemove', 'node', e => {
  if (TT.style.display === 'none') return;
  const box = CY_CONT.getBoundingClientRect();
  const ev  = e.originalEvent;
  const mx = (ev.clientX - box.left) + 16;
  const my = (ev.clientY - box.top)  - 10;
  TT.style.left = (mx + 150 > box.width ? mx - 166 : mx) + 'px';
  TT.style.top  = (my + 140 > box.height ? my - 145 : my) + 'px';
});
cy.on('mouseout', 'node', () => {
  TT.style.display = 'none';
  cy.elements().style('opacity', null);
});

// Duplo-clique: zoom na vizinhança
cy.on('dblclick', 'node', e => {
  cy.animate({ fit: { eles: e.target.closedNeighborhood(), padding: 80 }, duration: 350 });
});

cy.ready(() => rebuild());

// ── Botões ────────────────────────────────────────────────────────────────────
document.getElementById('btn-back').addEventListener('click', () => {
  navStack.length = 0;  // volta direto ao início (limpa toda a pilha)
  rebuild();
});
document.getElementById('btn-fit').addEventListener('click', () =>
  cy.animate({ fit: { padding: 70 }, duration: 300 })
);
document.getElementById('btn-relayout').addEventListener('click', () => rebuild());
document.getElementById('btn-reset').addEventListener('click', () => {
  navStack.length = 0;
  rebuild();
  document.getElementById('node-detail').classList.remove('visible');
});

// ══════════════════════════════════════════════════════════════════════════════
// PAINEL DE DETALHE
// ══════════════════════════════════════════════════════════════════════════════

// ── Formatters ────────────────────────────────────────────────────────────────
function fNum(v, dec) {
  if (v == null) return '<span class="nd-na">\u2014</span>';
  const d = dec != null ? dec : 2;
  return v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
}
function fPct(v) {
  if (v == null) return '<span class="nd-na">\u2014</span>';
  const c = v >= 0 ? '#4ade80' : '#f87171';
  return '<span style="color:' + c + '">' + (v >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%</span>';
}
function fScore(v) {
  if (v == null) return '<span class="nd-na">\u2014</span>';
  const c = v >= 0.3 ? '#4ade80' : (v <= -0.3 ? '#f87171' : '#f59e0b');
  return '<span style="color:' + c + '">' + (v >= 0 ? '+' : '') + v.toFixed(3) + '</span>';
}
function fMult(v) {
  if (v == null) return '<span class="nd-na">\u2014</span>';
  return v.toFixed(1) + 'x';
}
function fB(v) {
  // Formata bilhões: 2850 → $2.85T, 45 → $45B
  if (v == null) return '<span class="nd-na">\u2014</span>';
  if (v >= 1000) return '$' + (v / 1000).toFixed(2) + 'T';
  return '$' + v.toFixed(1) + 'B';
}
function ndRow(k, v) {
  return '<div class="nd-row"><span class="nd-key">' + k + '</span><span class="nd-val">' + v + '</span></div>';
}

function showDetail(d) {
  document.getElementById('nd-label').textContent = d.label || d.id;
  document.getElementById('nd-label').style.color = d.color || '#e2e8f0';
  document.getElementById('nd-sub').textContent =
    (d.ticker ? '[' + d.ticker + ']  ' : '') + 'Level ' + (d.level != null ? d.level : '?');

  // ── Tab: Price ──────────────────────────────────────────────────────────────
  const priceRows = [];
  if (d.price   != null) priceRows.push(ndRow('Price',    fNum(d.price)));
  if (d.daily   != null) priceRows.push(ndRow('1D',       fPct(d.daily)));
  if (d.weekly  != null) priceRows.push(ndRow('1W',       fPct(d.weekly)));
  if (d.ytd     != null) priceRows.push(ndRow('YTD',      fPct(d.ytd)));
  if (d.momentum != null) priceRows.push(ndRow('Momentum', fScore(d.momentum)));
  if (d.contagion != null && d.contagion > 0) {
    const cc = d.contagion > 0.5 ? '#ef4444' : (d.contagion > 0.25 ? '#f97316' : '#4ade80');
    priceRows.push(ndRow('Contagion',
      '<span style="color:' + cc + '">' + d.contagion.toFixed(3) + '</span>'));
  }
  if (d.propagated_shock != null && Math.abs(d.propagated_shock) > 0.0001) {
    priceRows.push(ndRow('Prop. Shock', fPct(d.propagated_shock)));
  }
  if (d.is_hub) priceRows.push(ndRow('Role', '<span style="color:#f59e0b">Hub \u2605</span>'));
  document.getElementById('nd-rows').innerHTML = priceRows.join('');

  // ── Tab: Valuation ─────────────────────────────────────────────────────────
  const a = d.anatomy || {};
  const valRows = [];
  valRows.push(ndRow('P/E',          fMult(a.pe)));
  valRows.push(ndRow('Fwd P/E',      fMult(a.forward_pe)));
  valRows.push(ndRow('P/S',          fMult(a.ps)));
  valRows.push(ndRow('P/B',          fMult(a.pb)));
  valRows.push(ndRow('EV/EBITDA',    fMult(a.ev_ebitda)));
  valRows.push(ndRow('Beta',         a.beta != null ? a.beta.toFixed(2) : '<span class="nd-na">\u2014</span>'));
  valRows.push(ndRow('Mkt Cap',      fB(a.mktcap_b)));
  if (a.hi_52w != null) valRows.push(ndRow('52W High',   fNum(a.hi_52w)));
  if (a.lo_52w != null) valRows.push(ndRow('52W Low',    fNum(a.lo_52w)));
  if (a.drawdown_52w != null) valRows.push(ndRow('DD 52W',    fPct(a.drawdown_52w)));
  document.getElementById('nd-rows-val').innerHTML = valRows.join('');

  // ── Tab: Quality ───────────────────────────────────────────────────────────
  const qualRows = [];
  qualRows.push(ndRow('ROE',          a.roe  != null ? fPct(a.roe)  : '<span class="nd-na">\u2014</span>'));
  qualRows.push(ndRow('ROA',          a.roa  != null ? fPct(a.roa)  : '<span class="nd-na">\u2014</span>'));
  qualRows.push(ndRow('Net Margin',   a.profit_margin != null ? fPct(a.profit_margin) : '<span class="nd-na">\u2014</span>'));
  qualRows.push(ndRow('Debt/Equity',  a.debt_equity != null ? a.debt_equity.toFixed(2) : '<span class="nd-na">\u2014</span>'));
  qualRows.push(ndRow('Div. Yield',   a.dividend_yield != null ? fPct(a.dividend_yield) : '<span class="nd-na">\u2014</span>'));
  document.getElementById('nd-rows-qual').innerHTML = qualRows.join('');

  // ── Tab: Options ────────────────────────────────────────────────────────────
  const o = d.options || {};
  const optRows = [];
  if (o.atm_iv    != null) optRows.push(ndRow('ATM IV',     fPct(o.atm_iv)));
  if (o.skew_5pct != null) optRows.push(ndRow('Skew 5%',   fPct(o.skew_5pct)));
  if (o.pcr_oi    != null) optRows.push(ndRow('Put/Call',  o.pcr_oi.toFixed(2)));
  if (o.gex_b     != null) {
    const gc = o.gex_b >= 0 ? '#4ade80' : '#f87171';
    optRows.push(ndRow('GEX', '<span style="color:' + gc + '">' + (o.gex_b >= 0 ? '+' : '') + o.gex_b.toFixed(2) + 'B</span>'));
  }
  if (o.next_expiry) optRows.push(ndRow('Next Exp.', o.next_expiry));
  // Term structure
  const ts = o.term_structure || {};
  const tsKeys = Object.keys(ts).sort((a,b) => parseInt(a) - parseInt(b));
  tsKeys.forEach(k => {
    if (ts[k] != null) optRows.push(ndRow(k.replace('iv_','') + ' IV', fPct(ts[k])));
  });
  if (optRows.length === 0) {
    optRows.push('<div style="font-size:11px;color:#475569;padding:4px 0">Sem dados de opções</div>');
  }
  document.getElementById('nd-rows-opt').innerHTML = optRows.join('');

  // ── Tab: Risk (Probabilistic) ───────────────────────────────────────────────
  const p = d.prob || {};
  const riskRows = [];

  // VaR / CVaR
  if (p.var_95  != null) riskRows.push(ndRow('VaR 95%',  fPct(p.var_95)));
  if (p.cvar_95 != null) riskRows.push(ndRow('CVaR 95%', fPct(p.cvar_95)));
  if (p.var_99  != null) riskRows.push(ndRow('VaR 99%',  fPct(p.var_99)));
  if (p.cvar_99 != null) riskRows.push(ndRow('CVaR 99%', fPct(p.cvar_99)));
  if (p.ann_vol != null) riskRows.push(ndRow('Vol ann.',  fPct(p.ann_vol)));

  // Distribuição
  if (p.skewness        != null) riskRows.push(ndRow('Skewness', p.skewness.toFixed(3)));
  if (p.excess_kurtosis != null) riskRows.push(ndRow('Ex. Kurt.',  p.excess_kurtosis.toFixed(2)));
  if (p.tail_score      != null) {
    const tc = p.tail_score > 0.6 ? '#f87171' : (p.tail_score > 0.3 ? '#f59e0b' : '#4ade80');
    riskRows.push(ndRow('Tail Score',
      '<div style="display:flex;align-items:center;gap:4px">'
      + '<div style="flex:1;height:4px;background:#1a2535;border-radius:2px">'
      + '<div style="width:' + (p.tail_score * 100).toFixed(0) + '%;height:100%;background:' + tc + ';border-radius:2px"></div>'
      + '</div><span style="color:' + tc + '">' + p.tail_score.toFixed(2) + '</span></div>'
    ));
  }
  if (p.dist && p.dist.df != null) riskRows.push(ndRow('t-dist df', p.dist.df.toFixed(1)));

  // FFT ciclo dominante
  if (p.dominant_cycle != null) {
    riskRows.push(ndRow('Dom. Cycle', p.dominant_cycle + 'd'));
  }

  // Regime HMM
  if (p.regime_prob_bull != null) {
    const bull = p.regime_prob_bull;
    const bc   = bull > 0.6 ? '#4ade80' : (bull < 0.4 ? '#f87171' : '#f59e0b');
    const bl   = bull > 0.6 ? 'Bull' : (bull < 0.4 ? 'Bear' : 'Mixed');
    riskRows.push(ndRow('Regime',
      '<div style="display:flex;align-items:center;gap:4px">'
      + '<div style="flex:1;height:4px;background:#1a2535;border-radius:2px">'
      + '<div style="width:' + (bull * 100).toFixed(0) + '%;height:100%;background:' + bc + ';border-radius:2px"></div>'
      + '</div><span style="color:' + bc + '">' + bl + ' ' + (bull * 100).toFixed(0) + '%</span></div>'
    ));
  }

  if (riskRows.length === 0) {
    riskRows.push('<div style="font-size:11px;color:#475569;padding:4px 0">Sem dados probabilísticos</div>');
  }
  document.getElementById('nd-rows-risk').innerHTML = riskRows.join('');

  // ── Tab: Flow (GEX + LETF) ───────────────────────────────────────────────────
  const fl = d.flow || {};
  const flowRows = [];
  if (fl.total_usd != null) {
    const mn = fl.total_usd / 1e6;
    const fc = mn > 0 ? '#4ade80' : (mn < 0 ? '#f87171' : '#94a3b8');
    const fa = mn > 0 ? '▲' : (mn < 0 ? '▼' : '◆');
    flowRows.push(ndRow('EOD Flow Total', '<span style="color:'+fc+'">'+fa+' $'+mn.toFixed(1)+'M</span>'));
  }
  if (fl.letf_flow_usd != null && fl.letf_flow_usd !== 0) {
    const lm = fl.letf_flow_usd / 1e6;
    const lc = lm > 0 ? '#4ade80' : '#f87171';
    flowRows.push(ndRow('LETF Rebal', '<span style="color:'+lc+'">$'+lm.toFixed(1)+'M</span>'));
  }
  if (fl.gex_flow_usd != null && fl.gex_flow_usd !== 0) {
    const gm = fl.gex_flow_usd / 1e6;
    const gc2 = gm > 0 ? '#4ade80' : '#f87171';
    flowRows.push(ndRow('GEX Hedge', '<span style="color:'+gc2+'">$'+gm.toFixed(1)+'M</span>'));
  }
  if (fl.direction) {
    const dc = {buy:'#4ade80',sell:'#f87171',flat:'#94a3b8'}[fl.direction] || '#94a3b8';
    flowRows.push(ndRow('Direção', '<span style="color:'+dc+'">'+fl.direction.toUpperCase()+'</span>'));
  }
  if (flowRows.length === 0) {
    flowRows.push('<div style="font-size:11px;color:#475569;padding:4px 0">Sem dados de fluxo mecânico</div>');
  }
  document.getElementById('nd-rows-flow').innerHTML = flowRows.join('');

  // Ativa aba Price por padrão ao abrir
  document.querySelectorAll('.nd-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.nd-tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelector('.nd-tab[data-tab="price"]').classList.add('active');
  document.getElementById('nd-tab-price').classList.add('active');

  document.getElementById('node-detail').classList.add('visible');
}

// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll('.nd-tab').forEach(function(btn) {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.nd-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.nd-tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('nd-tab-' + btn.dataset.tab).classList.add('active');
  });
});

document.getElementById('nd-close').addEventListener('click', () =>
  document.getElementById('node-detail').classList.remove('visible')
);

// ── Right-click: isolar nó + vizinhança 1-hop ──────────────────────────────
cy.on('cxttap', 'node', e => {
  if (currentFocus() !== null) return; // só em modo rede
  const id = e.target.data('id');
  if (isolateId === id) {
    isolateId = null;
    document.getElementById('iso-banner').style.display = 'none';
  } else {
    isolateId = id;
    const lbl = (e.target.data('label') || id).replace(/^\[.\] /, '');
    const iso = document.getElementById('iso-banner');
    iso.textContent = '⬡ ' + lbl + ' — clique direito novamente para sair';
    iso.style.display = 'block';
  }
  rebuild();
});

// ── Search / jump-to-ticker ────────────────────────────────────────────────
const searchEl = document.getElementById('search-input');
searchEl.addEventListener('input', e => {
  const q = e.target.value.trim().toUpperCase();
  if (!q) { cy.elements().style('opacity', null); return; }
  let first = null;
  cy.nodes().forEach(n => {
    const d = n.data();
    const match = (d.ticker || '').toUpperCase().includes(q)
               || (d.label  || '').toUpperCase().includes(q);
    n.style('opacity', match ? 1 : 0.08);
    if (match && !first) first = n;
  });
  if (first) cy.animate({ fit: { eles: first.closedNeighborhood(), padding: 90 }, duration: 400 });
});
searchEl.addEventListener('keydown', e => {
  if (e.key === 'Escape') { searchEl.value = ''; cy.elements().style('opacity', null); searchEl.blur(); }
});

// ── Color mode buttons ─────────────────────────────────────────────────────
document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    colorMode = btn.dataset.mode;
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    // Re-aplica cores sem full rebuild
    cy.nodes().forEach(n => {
      const d = Object.assign({}, n.data());
      n.data('_bg', getNodeBg(d));
    });
    cy.style(STYLE);
  });
});

// ── Correlation threshold slider ───────────────────────────────────────────
document.getElementById('corr-slider').addEventListener('input', e => {
  corrThresh = parseFloat(e.target.value);
  document.getElementById('corr-label').textContent = corrThresh.toFixed(2);
  if (currentFocus() === null) rebuild();
});

// ── Keyboard shortcuts ─────────────────────────────────────────────────────
function activateTab(name) {
  const panel = document.getElementById('node-detail');
  if (!panel.classList.contains('visible')) return;
  document.querySelectorAll('.nd-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.nd-tab-panel').forEach(p => p.classList.remove('active'));
  const btn = document.querySelector('.nd-tab[data-tab="' + name + '"]');
  if (btn) { btn.classList.add('active'); document.getElementById('nd-tab-' + name).classList.add('active'); }
}
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  switch (e.key) {
    case 'f': case 'F':
      cy.animate({ fit: { padding: 70 }, duration: 300 }); break;
    case 'r': case 'R':
      rebuild(); break;
    case 'Escape':
      if (document.getElementById('node-detail').classList.contains('visible')) {
        document.getElementById('node-detail').classList.remove('visible');
      } else if (isolateId !== null) {
        isolateId = null;
        document.getElementById('iso-banner').style.display = 'none';
        rebuild();
      } else if (navStack.length > 0) {
        navStack.pop(); rebuild();
      }
      break;
    case '1': activateTab('price');     break;
    case '2': activateTab('valuation'); break;
    case '3': activateTab('quality');   break;
    case '4': activateTab('options');   break;
    case '5': activateTab('risk');      break;
  }
});
"""


# ── Portfolio tab renderer ────────────────────────────────────────────────────

def _render_portfolio_tab(portfolio, market_prices: "dict | None", flow_pred) -> str:
    """Render the Portfolio top-level tab HTML (inner content only)."""

    # ── GEX + LETF summary bar ──────────────────────────────────────────────
    gex_html = ""
    if flow_pred is not None:
        try:
            dir_color = {"buy": "#10b981", "sell": "#ef4444", "flat": "#6b7280"}.get(
                getattr(flow_pred, "direction", "flat"), "#6b7280"
            )
            conv_color = {"high": "#10b981", "medium": "#f59e0b", "low": "#6b7280"}.get(
                getattr(flow_pred, "conviction", "low"), "#6b7280"
            )
            gamma_regime = getattr(getattr(flow_pred, "gex", None), "gamma_regime", "—") or "—"
            gamma_color  = {"long": "#10b981", "short": "#ef4444", "flat": "#6b7280"}.get(gamma_regime, "#6b7280")
            magnitude_bn = getattr(flow_pred, "magnitude_bn", 0.0) or 0.0
            gex_bn       = getattr(getattr(flow_pred, "gex", None), "gex_bn", 0.0) or 0.0
            direction    = getattr(flow_pred, "direction", "flat") or "flat"
            conviction   = getattr(flow_pred, "conviction", "low") or "low"
            gex_html = (
                f'<div class="gex-bar">'
                f'<div class="gex-item"><div class="gex-lbl">Fluxo Direcional</div>'
                f'<div class="gex-val pt-badge" style="color:{dir_color};border-color:{dir_color}22;background:{dir_color}11">'
                f'{direction.upper()}</div></div>'
                f'<div class="gex-item"><div class="gex-lbl">Magnitude</div>'
                f'<div class="gex-val" style="color:#94a3b8">${magnitude_bn:+.1f}Bn</div></div>'
                f'<div class="gex-item"><div class="gex-lbl">Convicção</div>'
                f'<div class="gex-val" style="color:{conv_color}">{conviction.upper()}</div></div>'
                f'<div class="gex-item"><div class="gex-lbl">Gamma Regime</div>'
                f'<div class="gex-val" style="color:{gamma_color}">{gamma_regime.upper()}</div></div>'
                f'<div class="gex-item"><div class="gex-lbl">GEX SPX</div>'
                f'<div class="gex-val" style="color:#94a3b8">${gex_bn:+.1f}Bn</div></div>'
                f'</div>'
            )
        except Exception:
            gex_html = ""

    # ── Positions table ─────────────────────────────────────────────────────
    if portfolio is None or not getattr(portfolio, "positions", None):
        positions_html = f'<div class="pt-empty">Sem posições abertas</div>'
    else:
        mp = market_prices or {}
        rows = ""
        for pos in portfolio.positions:
            ticker    = getattr(pos, "ticker", "—")
            direction = getattr(pos, "direction", "neutral")
            alloc_pct = getattr(pos, "allocation_pct", 0.0) or 0.0
            entry     = getattr(pos, "entry_price", 0.0) or 0.0
            stop      = getattr(pos, "stop_loss", 0.0) or 0.0
            target    = getattr(pos, "take_profit", 0.0) or 0.0
            rr        = getattr(pos, "risk_reward", 0.0) or 0.0
            rationale = getattr(pos, "rationale", []) or []
            first_rat = rationale[0] if rationale else "—"

            dir_class = "pt-dir-long" if direction == "long" else ("pt-dir-short" if direction == "short" else "")
            dir_label = {"long": "LONG ▲", "short": "SHORT ▼", "neutral": "—"}.get(direction, direction)

            # P&L vs current price
            cur_price = mp.get(ticker, {}).get("price") if mp else None
            pnl_html = "—"
            if cur_price and entry and entry > 0:
                if direction == "long":
                    pnl_pct = (cur_price - entry) / entry
                else:
                    pnl_pct = (entry - cur_price) / entry
                pnl_color = "#10b981" if pnl_pct >= 0 else "#ef4444"
                pnl_html  = f'<span style="color:{pnl_color};font-weight:700">{pnl_pct:+.1%}</span>'

            rows += (
                f"<tr>"
                f'<td style="font-weight:800;color:#f1f5f9;font-size:14px;letter-spacing:.5px">{ticker}</td>'
                f'<td class="{dir_class}" style="font-size:13px;font-weight:700">{dir_label}</td>'
                f'<td style="color:#cbd5e1;font-size:13px">{abs(alloc_pct)*100:.1f}%</td>'
                f'<td style="color:#cbd5e1;font-size:13px">{f"${entry:,.2f}" if entry else "—"}</td>'
                f'<td style="color:#f87171;font-size:13px;font-weight:600">{f"${stop:,.2f}" if stop else "—"}</td>'
                f'<td style="color:#34d399;font-size:13px;font-weight:600">{f"${target:,.2f}" if target else "—"}</td>'
                f'<td style="color:#fbbf24;font-size:13px;font-weight:700">{f"{rr:.1f}x" if rr else "—"}</td>'
                f'<td style="font-size:13px">{pnl_html}</td>'
                f'<td style="color:#64748b;font-size:12px;max-width:300px">{first_rat}</td>'
                f"</tr>"
            )
        positions_html = (
            '<div style="overflow-x:auto">'
            '<table class="pt-table" style="font-size:13px">'
            '<thead><tr>'
            '<th style="font-size:11px">Ticker</th><th style="font-size:11px">Direction</th>'
            '<th style="font-size:11px">Alloc%</th><th style="font-size:11px">Entry</th>'
            '<th style="font-size:11px">Stop</th><th style="font-size:11px">Target</th>'
            '<th style="font-size:11px">R/R</th><th style="font-size:11px">P&amp;L</th>'
            '<th style="font-size:11px">Rationale</th>'
            '</tr></thead>'
            f'<tbody>{rows}</tbody>'
            '</table></div>'
        )

    return f"""
<div class="pt-section">
  <div class="pt-title">GEX + LETF Mechanical Flow</div>
  {gex_html if gex_html else '<div style="color:#64748b;font-size:11px">Sem dados de fluxo disponíveis</div>'}
</div>
<div class="pt-section" style="flex:1;min-height:0">
  <div class="pt-title">Posições do Portfolio</div>
  {positions_html}
</div>
"""


# ── HTML builder ──────────────────────────────────────────────────────────────

def _load_editorial_html(bundle: "DailyIngestionBundle") -> str:
    """
    Carrega o editorial diário (brief HTML) para embutir na aba Informações de Mercado.
    Procura o _brief.html mais recente do dia no diretório do bundle.
    Retorna HTML completo do body do brief, ou string vazia se não encontrado.
    """
    import re as _re
    try:
        from app.storage.paths import workspace
        from pathlib import Path as _P
        bundle_dir = _P(workspace.bundles) / str(bundle.run_date)
        # Prefere _brief.html mais recente, depois qualquer _week_ahead_brief.html
        candidates = sorted(
            list(bundle_dir.glob("*_brief.html")) + list(bundle_dir.glob("*_week_ahead_brief.html")),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if not candidates:
            return ""
        raw = candidates[0].read_text(encoding="utf-8")
        # Extrai apenas o conteúdo entre <body> e </body>
        m = _re.search(r"<body[^>]*>(.*?)</body>", raw, _re.DOTALL | _re.IGNORECASE)
        content = m.group(1).strip() if m else raw
        # Remove scripts de auto-refresh e navegação independente
        content = _re.sub(r"<script[^>]*>.*?</script>", "", content, flags=_re.DOTALL | _re.IGNORECASE)
        content = _re.sub(r"<style[^>]*>.*?</style>", "", content, flags=_re.DOTALL | _re.IGNORECASE)
        return content
    except Exception:
        return ""


def _banco_bloomberg_status_html() -> str:
    """
    Banner de status do banco Bloomberg.
    - Verde: dados frescos (< STALE_MINUTES)
    - Amarelo: banco tem dados mas desatualizados — exibe último snapshot
    - Vermelho: banco não existe ou está vazio
    """
    try:
        from app.query_layer import BloombergQueryLayer
        ql = BloombergQueryLayer()
        snap = ql.get_snapshot_info()

        if not snap["has_data"]:
            # Banco vazio ou inexistente — erro real
            return (
                '<div style="background:#450a0a;border:1px solid #ef4444;border-radius:6px;'
                'padding:10px 14px;font-size:12px;color:#fca5a5;margin-bottom:8px">'
                '<strong>[!] Banco Bloomberg sem dados.</strong> '
                'Execute o Bloomberg Agent para popular o banco. '
                '<span style="color:#94a3b8;font-size:10px">'
                'agente run ingest</span>'
                '</div>'
            )

        # Banco tem dados — formata timestamp legível
        last_upd = snap.get("last_update", "")
        age = snap.get("age_minutes")
        tickers = snap.get("tickers_count", 0)

        try:
            from datetime import datetime as _dt, timezone as _tz
            dt = _dt.fromisoformat(last_upd.replace("Z", "+00:00"))
            last_str = dt.astimezone().strftime("%d/%m %H:%M")
        except Exception:
            last_str = last_upd[:16] if last_upd else "—"

        age_str = f"{age:.0f} min" if age is not None else "—"

        if snap["is_fresh"]:
            # Dados frescos
            return (
                f'<div style="background:#064e3b;border:1px solid #059669;border-radius:6px;'
                f'padding:7px 14px;font-size:11px;color:#34d399;margin-bottom:8px">'
                f'Bloomberg: {tickers} tickers — atualizado há {age_str} ({last_str})'
                f'</div>'
            )
        else:
            # Snapshot válido mas não fresco
            return (
                f'<div style="background:#1c1505;border:1px solid #78350f;border-radius:6px;'
                f'padding:7px 14px;font-size:11px;color:#fbbf24;margin-bottom:8px">'
                f'Bloomberg: snapshot de {last_str} ({age_str} atrás) &middot; {tickers} tickers &middot; '
                f'<span style="color:#94a3b8">execute bql_export.py para atualizar</span>'
                f'</div>'
            )
    except Exception:
        return ""


def generate_macro_desk_v2_html(
    bundle: "DailyIngestionBundle",
    graph_data: "dict | None" = None,
    curation_result: "CurationResult | None" = None,
    live_mode: bool = False,
    portfolio=None,   # PortfolioResult | None
    flow_pred=None,   # FlowPrediction | None
    editorial_html: str | None = None,  # override; None = carrega automaticamente
) -> str:
    if graph_data is None:
        from app.desk.graph_engine import build_from_bundle
        graph_data = build_from_bundle(bundle, curation_result)
    regime   = graph_data.get("regime", {})
    mst_meta = graph_data.get("mst_meta", {})
    rmt_meta = graph_data.get("rmt_meta", {})
    scores   = graph_data.get("agent_scores") or {}
    stats    = graph_data.get("stats", {})
    mp       = dict(bundle.market_prices or {})
    # Fallback: se bundle nao tem precos, carrega do banco (snapshot mais recente)
    if not any(k for k in mp if not k.startswith("__")):
        try:
            from app.query_layer import BloombergQueryLayer
            _ql = BloombergQueryLayer()
            if _ql.has_any_data():
                _db_prices = _ql.get_latest_prices()
                if _db_prices:
                    mp.update(_db_prices)
        except Exception:
            pass
    _bbg_status_html = _banco_bloomberg_status_html()
    vix_term       = graph_data.get("vix_term") or {}
    live_network   = graph_data.get("live_network") or {}
    flow_pred_panel = graph_data.get("flow_pred") or {}
    # flow_pred param takes precedence for portfolio tab; fallback to graph_data entry
    _pt_flow_pred = flow_pred if flow_pred is not None else (
        flow_pred_panel if isinstance(flow_pred_panel, object) and not isinstance(flow_pred_panel, dict) else None
    )

    run_date = str(bundle.run_date)
    gen_time = datetime.now(_tz.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Data da última coleta Bloomberg do banco
    _bbg_last_coleta = ""
    try:
        from app.query_layer import BloombergQueryLayer as _BQL
        _snap = _BQL().get_snapshot_info()
        if _snap.get("last_update"):
            from datetime import datetime as _dtt, timezone as _tzz
            _dt = _dtt.fromisoformat(_snap["last_update"].replace("Z", "+00:00"))
            _bbg_last_coleta = _dt.astimezone().strftime("%d/%m %H:%M")
    except Exception:
        pass

    regime_html  = _regime_badge(regime)
    score_labels = {"rational":"Rational","behavioral":"Behavioral",
                    "entropy":"Entropy","arbitration":"Arbitration","allocation":"Allocation"}
    scores_html  = "".join(_score_bar_html(lbl, scores.get(k)) for k, lbl in score_labels.items()) \
        if scores else '<div style="font-size:11px;color:#64748b">Scores indisponiveis</div>'
    market_rows  = _market_table_rows(mp)
    vix_term_html     = _vix_term_html(vix_term)
    risk_heatmap_html = _risk_heatmap_html(graph_data)
    live_network_html = _live_network_html(live_network)
    flow_panel_html   = _flow_panel_html(flow_pred_panel)

    portfolio_tab_html = _render_portfolio_tab(portfolio, bundle.market_prices if bundle else None, _pt_flow_pred)

    # ── Editorial diário — aba Informações de Mercado ─────────────────────────
    if editorial_html is None:
        editorial_html = _load_editorial_html(bundle)

    if editorial_html:
        editorial_content = editorial_html
    else:
        editorial_content = (
            '<div style="color:#475569;font-size:13px;padding:40px;text-align:center">'
            'Editorial diário não disponível.<br>'
            '<span style="font-size:11px">Rode <code>agente writer</code> para gerar o conteúdo.</span>'
            '</div>'
        )

    top_hubs = mst_meta.get("top_hubs") or []
    hubs_str = ", ".join(f"{t}({d})" for t, d in top_hubs[:3]) if top_hubs else "—"
    avg_corr = mst_meta.get("avg_corr")
    n_signal = rmt_meta.get("n_signal_factors")

    graph_json = json.dumps(graph_data, ensure_ascii=False, default=str)
    js_code    = _JS.replace("__GRAPH_DATA_JSON__", graph_json)

    # Carrega Cytoscape.js inline (evita bloqueio de CDN em file://)
    import pathlib as _pl
    _cy_path = _pl.Path(__file__).parent.parent / "static" / "cytoscape.min.js"
    cytoscape_js = _cy_path.read_text(encoding="utf-8") if _cy_path.exists() else \
        'document.write("<p style=\\"color:red\\">Cytoscape.js não encontrado</p>")'

    # JS setTimeout funciona em file:// (meta refresh bloqueado em browsers modernos)
    live_meta   = r'<script>setTimeout(()=>{const u=location.pathname;location.replace("file:///"+u.replace(/^\//,"")+"?_t="+Date.now());},90000);</script>' if live_mode else ""
    live_badge  = ('<span style="color:#22c55e;animation:contagion-pulse 2s infinite" '
                   'title="Auto-refresh a cada 90s">&#9679; LIVE</span>') if live_mode else \
                  '<span style="color:#f59e0b" title="Snapshot — não é streaming">&#9679; snapshot</span>'
    refreshed   = bundle.market_prices.get("__refreshed_at__", gen_time) if live_mode else gen_time

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
{live_meta}
<title>MacroDesk {'LIVE' if live_mode else ''} &mdash; {run_date}</title>
<style>{_CSS}</style>
</head>
<body>

{_bbg_status_html}
<div id="topbar">
  <span class="brand">MACRO DESK</span>
  <span class="run-date">{run_date} &middot; {refreshed} &middot; {live_badge}</span>
  {'<span style="font-size:10px;color:#334155;white-space:nowrap">BBG ' + _bbg_last_coleta + '</span>' if _bbg_last_coleta else ''}
  <div class="sep"></div>
  <input id="search-input" type="text" placeholder="&#128269; ticker / label" autocomplete="off" spellcheck="false">
  <div class="sep"></div>
  <span style="font-size:11px;color:#64748b;white-space:nowrap">&rho;&ge;</span>
  <input id="corr-slider" type="range" min="0" max="0.8" step="0.05" value="0.3">
  <span id="corr-label" style="font-size:11px;color:#64748b;min-width:26px">0.30</span>
  <div class="sep"></div>
  <button class="mode-btn active" data-mode="default">Base</button>
  <button class="mode-btn" data-mode="risk">Risk</button>
  <button class="mode-btn" data-mode="regime">Regime</button>
  <div class="sep"></div>
  <span class="spacer"></span>
  <span id="expand-count" style="font-size:10px;color:#64748b"></span>
  <div class="sep"></div>
  <button class="ctrl-btn" id="btn-back" style="display:none;border-color:#38bdf8;color:#38bdf8">&larr; Back</button>
  <button class="ctrl-btn" id="btn-fit">Fit <span style="color:#4a6380">[F]</span></button>
  <button class="ctrl-btn" id="btn-relayout">Layout <span style="color:#4a6380">[R]</span></button>
  <button class="ctrl-btn danger" id="btn-reset">Reset</button>
</div>

<div id="main-tabs-bar">
  <button class="main-tab active" onclick="switchMainTab('desk',this)">Desk Portfólio</button>
  <button class="main-tab" onclick="switchMainTab('portfolio',this)">Alocação</button>
  <button class="main-tab" onclick="switchMainTab('editorial',this)">Informações de Mercado</button>
</div>

<div id="desk-view" class="main-view active">
  <div id="main">
    <div id="sidebar">

      <div class="s-section">
        <div class="s-title">Market Regime</div>
        {regime_html}
      </div>

      <div class="s-section">
        <div class="s-title">Investment Agent</div>
        {scores_html}
      </div>

      <div class="s-section">
        <div class="s-title">Market Prices &middot; 1D</div>
        <table class="mp-table"><tbody>{market_rows}</tbody></table>
      </div>

      <div class="s-section">
        <div class="s-title">VIX Term Structure</div>
        {vix_term_html}
      </div>

      <div class="s-section">
        <div class="s-title">Tail Risk Map</div>
        <div style="font-size:10px;color:#64748b;margin-bottom:3px">Verde=baixo &middot; Vermelho=alto</div>
        {risk_heatmap_html}
      </div>

      <div class="s-section">
        <div class="s-title">Network</div>
        <div class="net-meta">
          <div><strong>Regime:</strong> {regime.get("regime","—")}</div>
          <div><strong>Avg &rho;:</strong> {f"{avg_corr:.3f}" if avg_corr is not None else "—"}</div>
          <div><strong>Signals:</strong> {n_signal if n_signal is not None else "—"} factors</div>
          <div><strong>Hubs:</strong> {hubs_str}</div>
          <div><strong>MST edges:</strong> {stats.get("mst_edges",0)}</div>
        </div>
      </div>

      <div class="s-section">
        <div class="s-title">Live Network</div>
        {live_network_html}
      </div>

      <div class="s-section">
        <div class="s-title">Fluxo Mecânico EOD</div>
        {flow_panel_html}
      </div>

    </div>

    <div id="cy-container">
      <div id="cy"></div>
      <div id="cy-tooltip"></div>
      <div id="iso-banner"></div>

      <div id="node-detail">
        <span id="nd-close">&times;</span>
        <div id="nd-label">—</div>
        <div id="nd-sub">—</div>
        <div class="nd-tabs">
          <button class="nd-tab active" data-tab="price">Price</button>
          <button class="nd-tab" data-tab="valuation">Val.</button>
          <button class="nd-tab" data-tab="quality">Quality</button>
          <button class="nd-tab" data-tab="options">Options</button>
          <button class="nd-tab" data-tab="risk">Risk</button>
          <button class="nd-tab" data-tab="flow">Flow</button>
        </div>
        <div id="nd-tab-price"     class="nd-tab-panel active"><div id="nd-rows"></div></div>
        <div id="nd-tab-valuation" class="nd-tab-panel"><div id="nd-rows-val"></div></div>
        <div id="nd-tab-quality"   class="nd-tab-panel"><div id="nd-rows-qual"></div></div>
        <div id="nd-tab-options"   class="nd-tab-panel"><div id="nd-rows-opt"></div></div>
        <div id="nd-tab-risk"      class="nd-tab-panel"><div id="nd-rows-risk"></div></div>
        <div id="nd-tab-flow"      class="nd-tab-panel"><div id="nd-rows-flow"></div></div>
      </div>

      <div id="hint-bar">
        <div class="leg"><div class="leg-line" style="background:#22c55e"></div> +&rho;</div>
        <div class="leg"><div class="leg-line" style="background:#ef4444"></div> &minus;&rho;</div>
        <div class="leg"><div class="leg-dot" style="background:#f59e0b"></div> Hub</div>
        <div class="leg"><div class="leg-dot" style="background:#818cf8"></div> Expan.</div>
        <div id="hint-text">
          [+] drill &middot; dbl-click zoom &middot; hover=stats &middot; right-click=isolate
        </div>
      </div>
      <div id="kbd-hints">
        F fit &nbsp; R layout &nbsp; Esc fechar<br>
        1–5 tabs &nbsp; ⌘+scroll zoom
      </div>
    </div>
  </div>
</div>

<div id="portfolio-view" class="main-view" style="flex-direction:column;overflow-y:auto;padding:18px 24px;gap:16px;background:#060a12">
  {portfolio_tab_html}
</div>

<div id="editorial-view" class="main-view" style="flex-direction:column;overflow-y:auto;background:#060a12">
  <div style="max-width:900px;margin:0 auto;padding:24px 32px;width:100%">
    {editorial_content}
  </div>
</div>

<script>
{cytoscape_js}
</script>
<script>
{js_code}
</script>
<script>
function switchMainTab(name, btn) {{
  document.querySelectorAll('.main-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.main-view').forEach(v => v.classList.remove('active'));
  if (btn) btn.classList.add('active');
  const view = document.getElementById(name + '-view');
  if (view) view.classList.add('active');
}}
</script>
</body>
</html>"""


# ── Save ──────────────────────────────────────────────────────────────────────

def save_macro_desk_v2(
    bundle: "DailyIngestionBundle",
    curation_result: "CurationResult | None" = None,
    graph_data: "dict | None" = None,
    live_mode: bool = False,
) -> Path:
    try:
        html = generate_macro_desk_v2_html(bundle, graph_data, curation_result, live_mode=live_mode)
    except Exception as exc:
        _log.error("macro_desk_v2_failed", error=str(exc), exc_info=True)
        raise

    out_dir  = workspace.bundles / str(bundle.run_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{bundle.run_id}_desk_v2.html"
    out_path.write_text(html, encoding="utf-8")
    _log.info("macro_desk_v2_saved", path=str(out_path))
    return out_path
