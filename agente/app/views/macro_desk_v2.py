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
    _meta_parts = [f"conf {conf:.0%}"]
    if avg_c != 0 or entropy != 0:
        _meta_parts += [f"&rho;={avg_c:.2f}", f"H={entropy:.2f}"]
    return (f'<div class="regime-badge" style="background:{bg};border-color:{fg}">'
            f'<div class="regime-icon" style="color:{fg}">{icons.get(regime,"?")}</div>'
            f'<div><div class="regime-label" style="color:{fg}">{labels.get(regime,regime.upper())}</div>'
            f'<div class="regime-meta">{" &middot; ".join(_meta_parts)}</div>'
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

    # GEX — suporta dois formatos: {gex: {gex_bn, gamma_regime}} e {gex: {spx: {gex_bn, ...}}}
    gex = flow_pred.get("gex", {})
    # Formato novo: GEXData serializado diretamente
    if "gex_bn" in gex:
        gex_spx = gex
    else:
        gex_spx = gex.get("spx", {})
    gex_bn    = gex_spx.get("gex_bn", 0.0) or 0.0
    gamma_reg = gex_spx.get("gamma_regime", "flat")
    flip_lvl  = gex_spx.get("flip_level")
    gex_color = "#4ade80" if gamma_reg == "long" else ("#f87171" if gamma_reg == "short" else "#94a3b8")
    gex_label = {"long": "LONG γ — amortece", "short": "SHORT γ — amplifica", "flat": "NEUTRO"}.get(gamma_reg, gamma_reg)
    flip_str  = f" flip@{flip_lvl:,.0f}" if flip_lvl else ""

    # LETF flows — suporta formato FlowPrediction.asdict() (spx/ndx/sox como dicts diretos)
    # e formato legado {letf: {spx: {flow_usd: ...}}}
    letf_raw = flow_pred.get("letf", {})
    # Formato novo: spx/ndx/sox como attrs de FlowPrediction (não aninhados em "letf")
    spx_flow_obj = flow_pred.get("spx", letf_raw.get("spx", {})) or {}
    ndx_flow_obj = flow_pred.get("ndx", letf_raw.get("ndx", {})) or {}
    spx_flow = (spx_flow_obj.get("flow_usd") or 0.0)
    ndx_flow = (ndx_flow_obj.get("flow_usd") or 0.0)
    spx_r    = spx_flow_obj.get("ret") or letf_raw.get("spx_r")
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

#topbar { display: flex; align-items: center; gap: 8px; padding: 8px 12px;
          background: #0a0f1a; border-bottom: 1px solid #1a2535;
          flex-shrink: 0; height: 52px; overflow-x: auto; overflow-y: hidden; }
.brand { font-size: 14px; font-weight: 700; color: #38bdf8;
         letter-spacing: 2.5px; white-space: nowrap; }
.run-date { font-size: 11px; color: #64748b; white-space: nowrap; }
.spacer { flex: 1; }
.ctrl-btn { padding: 4px 9px; border-radius: 5px; border: 1px solid #2d3f55;
            background: transparent; color: #64748b; cursor: pointer;
            font-size: 12px; font-weight: 600; transition: all 0.15s;
            white-space: nowrap; }
.ctrl-btn:hover { border-color: #38bdf8; color: #94a3b8; }
.ctrl-btn.danger:hover { border-color: #ef4444; color: #ef4444; }
/* Layer toggle buttons */
.layer-btn { padding: 3px 9px; border-radius: 4px; border: 1px solid #1e3a5f;
             background: transparent; color: #4a6380; cursor: pointer;
             font-size: 11px; font-weight: 700; transition: all 0.15s; letter-spacing: 0.3px; }
.layer-btn:hover { opacity: 0.9; }
.layer-btn.active[data-layer="structure"] { border-color: #818cf8; color: #818cf8; background: #818cf820; }
.layer-btn.active[data-layer="flow"]      { border-color: #22c55e; color: #22c55e; background: #22c55e20; }
.layer-btn.active[data-layer="convexity"] { border-color: #f59e0b; color: #f59e0b; background: #f59e0b20; }
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
#node-detail { position: absolute; top: 10px; right: 10px; width: 280px;
               background: #070d18; border: 1px solid #1e3148;
               border-radius: 8px; padding: 12px 12px 10px; display: none; z-index: 200;
               box-shadow: 0 4px 24px rgba(0,0,0,0.7); }
#node-detail.visible { display: block; }
#nd-close { position: absolute; top: 7px; right: 10px; cursor: pointer;
            color: #475569; font-size: 16px; line-height: 1; transition: color 0.15s; }
#nd-close:hover { color: #94a3b8; }
#nd-header { display: flex; align-items: flex-start; gap: 8px; margin-bottom: 9px; }
#nd-ticker-badge { font-size: 13px; font-weight: 800; background: #0f1f38;
                   border: 1px solid #1e3a5f; border-radius: 5px;
                   padding: 3px 8px; color: #38bdf8; white-space: nowrap; }
#nd-header-right { flex: 1; min-width: 0; }
#nd-label { font-size: 12px; font-weight: 700; color: #e2e8f0; white-space: nowrap;
            overflow: hidden; text-overflow: ellipsis; padding-right: 18px; }
#nd-sub { font-size: 10px; color: #4b6278; margin-top: 1px; }
#nd-quadrant-badge { display: inline-block; font-size: 9px; font-weight: 700;
                     padding: 1px 6px; border-radius: 3px; margin-top: 3px;
                     text-transform: uppercase; letter-spacing: 0.4px; }

/* Anatomy tabs */
.nd-tabs { display: flex; gap: 2px; margin-bottom: 10px; border-bottom: 1px solid #1a2d42; padding-bottom: 6px; }
.nd-tab { font-size: 9px; font-weight: 800; padding: 4px 8px;
          border-radius: 4px; border: none;
          background: transparent; color: #4b6278; cursor: pointer;
          letter-spacing: 0.6px; text-transform: uppercase; transition: all 0.15s; flex: 1; text-align: center; }
.nd-tab:hover { color: #94a3b8; background: #0d1a2a; }
.nd-tab.active { background: #0f2235; border-bottom: 2px solid #38bdf8;
                 color: #38bdf8; border-radius: 4px 4px 0 0; }
.nd-tab-panel { display: none; }
.nd-tab-panel.active { display: block; }

.nd-row { display: flex; justify-content: space-between; align-items: center;
  border-bottom: 1px solid #0d1825; padding: 5px 0; font-size: 11.5px; gap: 6px; }
.nd-key { color: #4b6278; flex: 1; }
.nd-val { font-weight: 700; color: #cbd5e1; text-align: right; }
.nd-na { color: #334155; font-size: 11px; }
.nd-section { font-size: 9px; font-weight: 800; color: #1e3a5f; text-transform: uppercase;
              letter-spacing: 0.8px; padding: 8px 0 3px; }
.nd-mini-bar { height: 4px; background: #0d1825; border-radius: 2px; overflow: hidden; margin: 2px 0; }
.nd-mini-fill { height: 100%; border-radius: 2px; transition: width 0.4s; }

/* Contagion panel */
#contagion-panel {
  position: absolute; top: 0; right: 0; bottom: 0; width: 300px;
  background: rgba(6,10,18,0.97); border-left: 1px solid #1a2535;
  display: flex; flex-direction: column; z-index: 300;
  transform: translateX(100%); transition: transform 0.25s ease;
  pointer-events: none;
}
#contagion-panel.open {
  transform: translateX(0); pointer-events: all;
}
#cp-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 12px 8px; border-bottom: 1px solid #1a2535;
}
#cp-title { font-size: 12px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: .06em; }
#cp-asset { font-size: 16px; font-weight: 700; color: #38bdf8; margin-bottom: 2px; }
#cp-close { background: none; border: none; color: #64748b; font-size: 18px; cursor: pointer; padding: 0; line-height: 1; }
#cp-close:hover { color: #e2e8f0; }
#cp-body { flex: 1; overflow-y: auto; padding: 10px 12px; }
.cp-hop { font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: .06em; margin: 10px 0 6px; }
.cp-row {
  display: flex; align-items: center; gap: 6px;
  padding: 4px 0; border-bottom: 1px solid #0d1520;
  font-size: 12px;
}
.cp-row-label { width: 70px; color: #94a3b8; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex-shrink: 0; }
.cp-bar-wrap { flex: 1; height: 8px; background: #0d1520; border-radius: 4px; overflow: hidden; }
.cp-bar { height: 100%; border-radius: 4px; transition: width 0.3s; }
.cp-rho { width: 38px; text-align: right; font-variant-numeric: tabular-nums; font-size: 11px; flex-shrink: 0; }
#cp-toggle { padding: 4px 12px; border-radius: 5px; border: 1px solid #2d3f55;
  background: transparent; color: #64748b; font-size: 12px; cursor: pointer; font-family: inherit; }
#cp-toggle.active { border-color: #38bdf8; color: #38bdf8; background: #0f1f2e; }
#cp-toggle:hover { border-color: #38bdf8; color: #94a3b8; }

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


/* ── Top-level main tabs ─────────────────────────────────────────────────── */
#main-tabs-bar { display: flex; gap: 2px; align-items: center;
                 padding: 0 14px; background: #070c17;
                 border-bottom: 1px solid #2d3f55; flex-shrink: 0; height: 36px;
                 overflow-x: auto; overflow-y: visible; }
.main-tab { padding: 5px 14px; border-radius: 4px 4px 0 0;
            border: 1px solid transparent; border-bottom: none;
            background: transparent; color: #64748b; cursor: pointer;
            font-size: 12px; font-weight: 700; letter-spacing: 0.6px;
            text-transform: uppercase; transition: all 0.15s; white-space: nowrap; }
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
    const hasMst = mst.length > 0;

    // Se isolamento ativo: manter só nó + 1-hop (MST + hierarquia)
    const isoNeighbors = isolateId != null
      ? new Set(GD.elements.edges
          .filter(e => (e.data.type === 'mst' || e.data.type === 'hierarchy')
                    && (e.data.source === isolateId || e.data.target === isolateId))
          .flatMap(e => [e.data.source, e.data.target]))
      : null;

    // Mostrar: hierarquia completa (level 0-4 = até setores) + nós com dados
    // level 4 = sectors — necessário para conectar indices aos ativos individuais
    const visIds = new Set(
      GD.elements.nodes
        .filter(n => n.data.level <= 4 || n.data.has_data)
        .map(n => n.data.id)
    );

    const nodes = GD.elements.nodes.filter(n => {
      if (!visIds.has(n.data.id)) return false;
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

    // Arestas de hierarquia entre nós visíveis + MST overlay
    const hierEdges = GD.elements.edges.filter(e =>
      e.data.type === 'hierarchy'
      && visIds.has(e.data.source)
      && visIds.has(e.data.target)
    );

    // RRG edges — rotação relativa vs SPX (somente entre nós visíveis)
    const rrgEdges = GD.elements.edges.filter(e =>
      e.data.type === 'rrg'
      && visIds.has(e.data.source)
      && visIds.has(e.data.target)
    );

    return [...nodes, ...hierEdges, ...mst, ...rrgEdges];
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
// HELPERS DE COR — gradiente bolinha
// ══════════════════════════════════════════════════════════════════════════════

function _hexToRgb(hex) {
  hex = hex.replace('#','');
  if (hex.length === 3) hex = hex.split('').map(c=>c+c).join('');
  return [parseInt(hex.substr(0,2),16), parseInt(hex.substr(2,2),16), parseInt(hex.substr(4,2),16)];
}
function _rgbToHex(r,g,b) {
  return '#'+[r,g,b].map(v=>Math.min(255,Math.max(0,Math.round(v))).toString(16).padStart(2,'0')).join('');
}
function _lighten(hex, pct) {
  if (!hex || !hex.startsWith('#')) return hex || '#374151';
  const [r,g,b] = _hexToRgb(hex);
  return _rgbToHex(r+(255-r)*pct, g+(255-g)*pct, b+(255-b)*pct);
}
function _darken(hex, pct) {
  if (!hex || !hex.startsWith('#')) return hex || '#374151';
  const [r,g,b] = _hexToRgb(hex);
  return _rgbToHex(r*(1-pct), g*(1-pct), b*(1-pct));
}

// ══════════════════════════════════════════════════════════════════════════════
// ESTILOS
// ══════════════════════════════════════════════════════════════════════════════

const STYLE = [
  // ── Base: todos os nós são círculos, tamanho data-driven ──────────────────
  { selector: 'node', style: {
    'shape': 'ellipse',
    // Gradiente radial simulado: top-left claro → bottom-right cor base
    'background-gradient-direction': 'to-bottom-right',
    'background-gradient-stop-colors': function(n) {
      const c = n.data('_bg') || '#374151';
      return _lighten(c, 0.5) + ' ' + c;
    },
    'background-gradient-stop-positions': '0% 85%',
    // Glow/sombra colorida
    'shadow-blur': 8,
    'shadow-color': function(n) { return n.data('_bg') || '#374151'; },
    'shadow-opacity': 0.55,
    'shadow-offset-x': 1,
    'shadow-offset-y': 2,
    // Borda fina mais clara que a cor
    'border-width': 2,
    'border-color': function(n) { return _lighten(n.data('_bg') || '#374151', 0.35); },
    'border-opacity': 0.9,
    // Label dentro da bolinha
    'label': 'data(label)',
    'color': '#ffffff',
    'font-size': 9,
    'font-weight': 'bold',
    'text-valign': 'center',
    'text-halign': 'center',
    'text-wrap': 'wrap',
    'text-max-width': '64px',
    'text-outline-color': function(n) { return _darken(n.data('_bg') || '#374151', 0.4); },
    'text-outline-width': 1.5,
    'text-outline-opacity': 0.85,
    'width': 'data(size)', 'height': 'data(size)',
    'transition-property': 'background-color, border-color, width, height, shadow-blur',
    'transition-duration': '0.25s',
  } },

  // ── Nós com dados de mercado — bolinha maior e mais brilhante ─────────────
  { selector: 'node[?has_data]', style: {
    'font-size': 10,
    'shadow-blur': 14,
    'shadow-opacity': 0.7,
    'border-width': 2.5,
    'z-index': 20,
  } },

  // ── Hub: borda dourada com glow forte ─────────────────────────────────────
  { selector: 'node[?is_hub]', style: {
    'border-color': '#f59e0b',
    'border-width': 3.5,
    'shadow-color': '#f59e0b',
    'shadow-blur': 18,
    'shadow-opacity': 0.8,
  } },

  // ── Contagion: borda + glow laranja/vermelho para alto contágio ───────────
  { selector: 'node[?has_data]', style: {
    'border-color': function(n) {
      const c = n.data('contagion');
      if (c > 0.5)  return '#ef4444';
      if (c > 0.25) return '#f97316';
      return _lighten(n.data('_bg') || '#374151', 0.35);
    },
    'border-width': function(n) {
      const c = n.data('contagion');
      if (c > 0.5)  return 4;
      if (c > 0.25) return 3;
      return 2.5;
    },
  } },

  // ── Level 0-2 (World, Asset Class, Region) ────────────────────────────────
  { selector: 'node[level < 3]', style: {
    'shadow-blur': 20,
    'shadow-opacity': 0.8,
    'border-width': 3,
    'font-size': 10,
    'z-index': 30,
  } },

  // ── Level 3 (índices) ─────────────────────────────────────────────────────
  { selector: 'node[level=3]', style: {
    'shadow-blur': 12,
    'shadow-opacity': 0.65,
    'border-width': 2.5,
    'font-size': 9,
    'z-index': 15,
  } },

  // ── Level 4 (setores) ─────────────────────────────────────────────────────
  { selector: 'node[level=4]', style: {
    'font-size': 8,
    'shadow-blur': 7,
    'shadow-opacity': 0.45,
    'border-width': 1.5,
    'color': '#ffffff',
    'z-index': 5,
  } },

  // ── Level 5 sem dados ─────────────────────────────────────────────────────
  { selector: 'node[level=5][!has_data]', style: {
    'font-size': 7,
    'shadow-blur': 4,
    'shadow-opacity': 0.3,
    'border-width': 1,
    'z-index': 3,
  } },

  // ── Expandível ────────────────────────────────────────────────────────────
  { selector: 'node[?expandable]', style: { 'cursor': 'pointer' } },

  // ── Raiz da camada atual ──────────────────────────────────────────────────
  { selector: 'node[?isFocusRoot]', style: {
    'font-size': 11,
    'border-color': '#38bdf8', 'border-width': 4,
    'shadow-color': '#38bdf8', 'shadow-blur': 20, 'shadow-opacity': 0.9,
    'cursor': 'pointer', 'z-index': 30,
  } },

  // ── Selecionado ───────────────────────────────────────────────────────────
  { selector: 'node:selected', style: {
    'border-color': '#fbbf24', 'border-width': 4,
    'shadow-color': '#fbbf24', 'shadow-blur': 22, 'shadow-opacity': 0.9,
    'overlay-opacity': 0,
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

  // ── RRG edges — rotação relativa vs benchmark (SPX) ───────────────────────
  // LEADING (verde) / IMPROVING (lima): ativo → sp500
  // WEAKENING (laranja) / LAGGING (vermelho): sp500 → ativo
  { selector: 'edge[type="rrg"]', style: {
    'line-color': 'data(color)',
    'width': 'data(width)',
    'curve-style': 'unbundled-bezier',
    'control-point-distances': [40],
    'control-point-weights': [0.5],
    'opacity': 0.75,
    'target-arrow-shape': 'triangle',
    'target-arrow-color': 'data(color)',
    'arrow-scale': 1.2,
    'z-index': 20,
    'label': function(e) {
      const q = e.data('quadrant') || '';
      const r = e.data('rs_ratio');
      if (!q) return '';
      const icons = {leading:'▲', improving:'↑', weakening:'↓', lagging:'▼'};
      return (icons[q] || q[0].toUpperCase()) + (r != null ? ' ' + r.toFixed(0) : '');
    },
    'font-size': 7,
    'color': 'data(color)',
    'text-background-color': '#060a12',
    'text-background-opacity': 0.8,
    'text-background-padding': '1px',
  } },
  { selector: 'edge[type="rrg"][dashed=true]', style: {
    'line-style': 'dashed',
    'line-dash-pattern': [6, 3],
    'opacity': 0.55,
  } },

  { selector: 'node:active', style: { 'overlay-opacity': 0.07 } },

  // ── Convexity layer — halo colorido por IV rank + fragility ──────────────
  // Verde = IV barata + oportunidade | Azul = IV barata | Roxo = IV cara | Vermelho = IV cara + frágil
  { selector: 'node[convexity]', style: {
    'shadow-color': function(n) {
      const cv = n.data('convexity');
      return (cv && cv.halo_color) ? cv.halo_color : (n.data('_bg') || '#374151');
    },
    'shadow-blur': function(n) {
      const cv = n.data('convexity');
      return (cv && cv.halo_color) ? 22 : 8;
    },
    'shadow-opacity': function(n) {
      const cv = n.data('convexity');
      return (cv && cv.halo_color) ? 0.70 : 0.55;
    },
  } },

  // ── Structure edges — mais finos e discretos ───────────────────────────────
  { selector: 'edge[layer="structure"]', style: {
    'opacity': 0.50,
  } },

  // ── Flow edges — mais brilhantes e espessos ────────────────────────────────
  { selector: 'edge[layer="flow"]', style: {
    'opacity': 0.85,
  } },
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
    : {   // Modo rede: breadthfirst — SPX como âncora gravitacional
        name: 'breadthfirst',
        animate: true, animationDuration: 600,
        directed: true,
        roots: (() => {
          // SPX como âncora: "o mercado tem um centro gravitacional"
          const spx = cy.nodes('#sp500');
          if (spx.length) return spx;
          // Fallback: nós raiz da hierarquia (level 0, sem parent)
          return cy.nodes().filter(n => !n.data('parent_id') && n.data('level') === 0);
        })(),
        spacingFactor: 1.6,
        avoidOverlap: true,
        fit: true, padding: 50,
      };

  // Fallback para cose se layout falhar
  function runLayout(opts) {
    try {
      cy.layout(opts).run();
    } catch(e) {
      cy.layout({
        name: 'breadthfirst', animate: true, animationDuration: 600,
        directed: true,
        roots: cy.nodes().filter(n => !n.data('parent_id') && n.data('level') === 0),
        spacingFactor: 1.6, avoidOverlap: true,
        fit: true, padding: 70,
      }).run();
    }
  }
  runLayout(layoutOpts);
  updateBreadcrumb();
}

// ══════════════════════════════════════════════════════════════════════════════
// LAYER TOGGLES — Structure · Flow · Convexity
// ══════════════════════════════════════════════════════════════════════════════

const layerState = { structure: true, flow: true, convexity: true };

function applyLayerVisibility() {
  // Structure edges: hierarchy + mst + rmt
  cy.edges('[layer="structure"]').style('display', layerState.structure ? 'element' : 'none');
  // Flow edges: rrg + shadow_flow
  cy.edges('[layer="flow"]').style('display', layerState.flow ? 'element' : 'none');
  // Convexity: toggle halo on nodes that have convexity data
  cy.nodes().forEach(function(n) {
    const cv = n.data('convexity');
    if (!cv || !cv.halo_color) return;
    if (layerState.convexity) {
      n.style({ 'shadow-color': cv.halo_color, 'shadow-blur': 22, 'shadow-opacity': 0.70 });
    } else {
      n.style({ 'shadow-color': n.data('_bg') || '#374151', 'shadow-blur': 8, 'shadow-opacity': 0.55 });
    }
  });
}

document.querySelectorAll('.layer-btn').forEach(function(btn) {
  btn.addEventListener('click', function() {
    const layer = btn.dataset.layer;
    layerState[layer] = !layerState[layer];
    btn.classList.toggle('active', layerState[layer]);
    applyLayerVisibility();
  });
});

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
    // Se contagion ativo: mostra painel de contagio sem fazer drill-down
    if (contagionEnabled) { showContagionPanel(d.id); return; }
    // ── Entra na próxima camada (drill-down) ──────────────────────────────
    navStack.push({ nodeId: d.id, label: N[d.id] ? (N[d.id].label || d.id) : d.id });
    rebuild();
    return;
  }

  // ── Nó folha → painel de detalhe + contagion ─────────────────────────────
  showDetail(d);
  if (contagionEnabled) showContagionPanel(d.id);
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
document.getElementById('btn-tree-layout').addEventListener('click', () => {
  // Force-directed alternativo (agrupa por correlação, não por hierarquia)
  cy.layout({
    name: 'cose', animate: true, animationDuration: 600,
    randomize: true,
    nodeRepulsion: 28000,
    idealEdgeLength: 80,
    edgeElasticity: 120,
    nestingFactor: 1.5,
    gravity: 50,
    numIter: 600,
    initialTemp: 180,
    coolingFactor: 0.95,
    minTemp: 1.0,
    fit: true, padding: 50,
  }).run();
});
document.getElementById('btn-reset').addEventListener('click', () => {
  navStack.length = 0;
  rebuild();
  document.getElementById('node-detail').classList.remove('visible');
  closeContagionPanel();
});

// ══════════════════════════════════════════════════════════════════════════════
// CONTAGION PANEL
// ══════════════════════════════════════════════════════════════════════════════

const CP_PANEL  = document.getElementById('contagion-panel');
const CP_ASSET  = document.getElementById('cp-asset');
const CP_BODY   = document.getElementById('cp-body');
const CP_TOGGLE = document.getElementById('cp-toggle');
let contagionEnabled = false;
let lastContagionId  = null;

function closeContagionPanel() {
  CP_PANEL.classList.remove('open');
  CP_TOGGLE.classList.remove('active');
}

function buildContagionRows(peers, label) {
  if (!peers.length) return '';
  const maxAbs = Math.max(...peers.map(p => Math.abs(p.rho)), 0.01);
  let html = `<div class="cp-hop">${label}</div>`;
  peers.forEach(p => {
    const pct  = Math.round(Math.abs(p.rho) / maxAbs * 100);
    const col  = p.rho >= 0 ? '#22c55e' : '#ef4444';
    const sign = p.rho >= 0 ? '+' : '−';
    const abs  = Math.abs(p.rho).toFixed(2);
    html += `
    <div class="cp-row" title="${p.id}">
      <span class="cp-row-label">${p.label}</span>
      <div class="cp-bar-wrap">
        <div class="cp-bar" style="width:${pct}%;background:${col}"></div>
      </div>
      <span class="cp-rho" style="color:${col}">${sign}${abs}</span>
    </div>`;
  });
  return html;
}

function showContagionPanel(nodeId) {
  const label = (N[nodeId] && N[nodeId].label) ? N[nodeId].label : nodeId;
  CP_ASSET.textContent = label;
  lastContagionId = nodeId;

  // 1st hop: direct MST neighbors
  const hop1 = [];
  const hop1Ids = new Set([nodeId]);
  GD.elements.edges.forEach(e => {
    if (e.data.type !== 'mst') return;
    let peer = null, rho = e.data.correlation || 0;
    if (e.data.source === nodeId) peer = e.data.target;
    else if (e.data.target === nodeId) peer = e.data.source;
    if (peer) { hop1.push({ id: peer, label: (N[peer]&&N[peer].label)||peer, rho }); hop1Ids.add(peer); }
  });
  hop1.sort((a,b) => Math.abs(b.rho) - Math.abs(a.rho));

  // 2nd hop: neighbors of neighbors (excluding already shown)
  const hop2 = [];
  const hop2Ids = new Set();
  hop1.forEach(h => {
    GD.elements.edges.forEach(e => {
      if (e.data.type !== 'mst') return;
      let peer = null, rho = e.data.correlation || 0;
      if (e.data.source === h.id) peer = e.data.target;
      else if (e.data.target === h.id) peer = e.data.source;
      if (peer && !hop1Ids.has(peer) && !hop2Ids.has(peer)) {
        hop2Ids.add(peer);
        // Effective contagion: product of correlations (attenuated)
        hop2.push({ id: peer, label: (N[peer]&&N[peer].label)||peer, rho: h.rho * rho * 0.7 });
      }
    });
  });
  hop2.sort((a,b) => Math.abs(b.rho) - Math.abs(a.rho));

  CP_BODY.innerHTML = hop1.length
    ? buildContagionRows(hop1, '1° grau — direto') +
      (hop2.length ? buildContagionRows(hop2.slice(0,8), '2° grau — indireto') : '')
    : '<div style="font-size:12px;color:#64748b;padding:16px 0">Sem conexões MST para este ativo.<br><small>MST requer dados históricos (IBKR ou outra fonte).</small></div>';

  CP_PANEL.classList.add('open');
}

CP_TOGGLE.addEventListener('click', () => {
  contagionEnabled = !contagionEnabled;
  CP_TOGGLE.classList.toggle('active', contagionEnabled);
  if (!contagionEnabled) {
    closeContagionPanel();
  } else if (lastContagionId) {
    showContagionPanel(lastContagionId);
  }
});

document.getElementById('cp-close').addEventListener('click', () => {
  contagionEnabled = false;
  closeContagionPanel();
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

function ndSection(label) {
  return '<div class="nd-section">' + label + '</div>';
}
function ndBar(v, color) {
  // v in [0,1], renders a mini bar
  const w = Math.min(100, Math.max(0, v * 100)).toFixed(0);
  return '<div class="nd-mini-bar"><div class="nd-mini-fill" style="width:'+w+'%;background:'+color+'"></div></div>';
}

function showDetail(d) {
  const ticker = d.ticker || d.id || '—';
  const label  = d.label || d.id || '—';

  // ── Header ──────────────────────────────────────────────────────────────────
  document.getElementById('nd-ticker-badge').textContent = ticker;
  document.getElementById('nd-label').textContent = label !== ticker ? label : '';
  document.getElementById('nd-label').style.color = d.color || '#e2e8f0';
  document.getElementById('nd-sub').textContent =
    (d.level != null ? 'Level ' + d.level : '') + (d.is_hub ? '  ★ Hub' : '');

  // Quadrant badge
  const qBadge = document.getElementById('nd-quadrant-badge');
  const quad_colors = {leading:'#22c55e', improving:'#3b82f6', weakening:'#f97316', lagging:'#ef4444'};
  const quad_bg     = {leading:'#052e16', improving:'#1e3a5f', weakening:'#431407', lagging:'#450a0a'};
  const quad_icons  = {leading:'&#9650;', improving:'&#8593;', weakening:'&#8595;', lagging:'&#9660;'};
  const quad_labels = {leading:'LEADING', improving:'IMPROVING', weakening:'WEAKENING', lagging:'LAGGING'};
  if (d.rrg_quadrant && quad_colors[d.rrg_quadrant]) {
    const qc = quad_colors[d.rrg_quadrant];
    const qb = quad_bg[d.rrg_quadrant];
    qBadge.innerHTML = (quad_icons[d.rrg_quadrant]||'') + ' ' + (quad_labels[d.rrg_quadrant]||d.rrg_quadrant.toUpperCase());
    qBadge.style.cssText = 'display:inline-block;color:'+qc+';background:'+qb+';border:1px solid '+qc+
      ';font-size:9px;font-weight:800;padding:1px 6px;border-radius:3px;margin-top:3px;letter-spacing:0.4px;text-transform:uppercase';
  } else {
    qBadge.style.display = 'none';
  }

  const a  = d.anatomy   || {};
  const o  = d.options   || {};
  const p  = d.prob      || {};
  const fl = d.flow      || {};
  const cv = d.convexity || {};

  // ── PAINEL — preço, retornos, valuation, qualidade ─────────────────────────
  const painelRows = [];
  // Preço
  if (d.price  != null) painelRows.push(ndRow('Preço', fNum(d.price)));
  if (d.daily  != null) painelRows.push(ndRow('1D',    fPct(d.daily)));
  if (d.weekly != null) painelRows.push(ndRow('1W',    fPct(d.weekly)));
  if (d.ytd    != null) painelRows.push(ndRow('YTD',   fPct(d.ytd)));
  if (d.momentum != null) painelRows.push(ndRow('Momentum', fScore(d.momentum)));
  // Valuation
  painelRows.push(ndSection('Valuation'));
  if (a.pe       != null) painelRows.push(ndRow('P/E',       fMult(a.pe)));
  if (a.forward_pe != null) painelRows.push(ndRow('Fwd P/E', fMult(a.forward_pe)));
  if (a.ev_ebitda != null) painelRows.push(ndRow('EV/EBITDA',fMult(a.ev_ebitda)));
  if (a.mktcap_b != null) painelRows.push(ndRow('Mkt Cap',   fB(a.mktcap_b)));
  if (a.drawdown_52w != null) painelRows.push(ndRow('DD 52W', fPct(a.drawdown_52w)));
  // Quality
  painelRows.push(ndSection('Qualidade'));
  if (a.roe  != null) painelRows.push(ndRow('ROE',        fPct(a.roe)));
  if (a.roa  != null) painelRows.push(ndRow('ROA',        fPct(a.roa)));
  if (a.profit_margin != null) painelRows.push(ndRow('Margem', fPct(a.profit_margin)));
  if (a.debt_equity != null) painelRows.push(ndRow('D/E', a.debt_equity.toFixed(2)));
  if (a.dividend_yield != null) painelRows.push(ndRow('Div Yield', fPct(a.dividend_yield)));
  document.getElementById('nd-rows-painel').innerHTML = painelRows.join('');

  // ── RISCO — probabilístico + volatilidade + tail ────────────────────────────
  const riscoRows = [];
  if (p.ann_vol  != null) riscoRows.push(ndRow('Vol Anual',  fPct(p.ann_vol)));
  if (p.var_95   != null) riscoRows.push(ndRow('VaR 95%',   fPct(p.var_95)));
  if (p.cvar_95  != null) riscoRows.push(ndRow('CVaR 95%',  fPct(p.cvar_95)));
  if (p.var_99   != null) riscoRows.push(ndRow('VaR 99%',   fPct(p.var_99)));
  if (p.tail_score != null) {
    const tc = p.tail_score > 0.6 ? '#ef4444' : (p.tail_score > 0.3 ? '#f59e0b' : '#4ade80');
    riscoRows.push(ndRow('Tail Score',
      '<div style="display:flex;align-items:center;gap:5px;min-width:90px">'
      + ndBar(p.tail_score, tc)
      + '<span style="color:'+tc+';font-size:11px">'+p.tail_score.toFixed(2)+'</span></div>'));
  }
  riscoRows.push(ndSection('Distribuição'));
  if (p.skewness        != null) riscoRows.push(ndRow('Skewness',  p.skewness.toFixed(3)));
  if (p.excess_kurtosis != null) riscoRows.push(ndRow('Ex. Kurt.', p.excess_kurtosis.toFixed(2)));
  if (p.dist && p.dist.df != null) riscoRows.push(ndRow('t-dist df', p.dist.df.toFixed(1)));
  if (p.dominant_cycle  != null) riscoRows.push(ndRow('Ciclo dom.', p.dominant_cycle + 'd'));
  riscoRows.push(ndSection('Regime HMM'));
  if (p.regime_prob_bull != null) {
    const bull = p.regime_prob_bull;
    const bc   = bull > 0.6 ? '#4ade80' : (bull < 0.4 ? '#ef4444' : '#f59e0b');
    const bl   = bull > 0.6 ? 'Bull' : (bull < 0.4 ? 'Bear' : 'Mixed');
    riscoRows.push(ndRow('Regime',
      '<div style="display:flex;align-items:center;gap:5px;min-width:100px">'
      + ndBar(bull, bc)
      + '<span style="color:'+bc+';font-size:11px">'+bl+' '+(bull*100).toFixed(0)+'%</span></div>'));
  }
  if (d.contagion != null && d.contagion > 0) {
    const cc = d.contagion > 0.5 ? '#ef4444' : (d.contagion > 0.25 ? '#f97316' : '#4ade80');
    riscoRows.push(ndRow('Contágio', '<span style="color:'+cc+'">'+d.contagion.toFixed(3)+'</span>'));
  }
  if (d.propagated_shock != null && Math.abs(d.propagated_shock) > 0.0001)
    riscoRows.push(ndRow('Shock prop.', fPct(d.propagated_shock)));
  if (riscoRows.filter(r => !r.includes('nd-section')).length === 0)
    riscoRows.push('<div class="nd-na">Sem dados probabilísticos</div>');
  document.getElementById('nd-rows-risco').innerHTML = riscoRows.join('');

  // ── GREGAS — opções, IV, skew, term structure, oportunidade ─────────────────
  const gregasRows = [];
  const ivRank = cv.iv_rank;
  if (ivRank != null) {
    const ic = ivRank < 0.35 ? '#4ade80' : (ivRank > 0.75 ? '#ef4444' : '#f59e0b');
    gregasRows.push(ndRow('IV Rank',
      '<div style="display:flex;align-items:center;gap:5px;min-width:100px">'
      + ndBar(ivRank, ic)
      + '<span style="color:'+ic+';font-size:11px">'+(ivRank*100).toFixed(0)+'%ile</span></div>'));
  }
  if (o.atm_iv    != null) gregasRows.push(ndRow('ATM IV',    fPct(o.atm_iv)));
  if (o.skew_5pct != null || cv.skew != null) {
    const sv = o.skew_5pct ?? cv.skew;
    const sc = sv < -0.02 ? '#4ade80' : (sv > 0.04 ? '#ef4444' : '#94a3b8');
    gregasRows.push(ndRow('Skew 5%', '<span style="color:'+sc+'">'+(sv>0?'+':'')+sv.toFixed(4)+'</span>'));
  }
  if (o.pcr_oi    != null) gregasRows.push(ndRow('Put/Call OI', o.pcr_oi.toFixed(2)));
  if (o.gex_b     != null) {
    const gc = o.gex_b >= 0 ? '#4ade80' : '#ef4444';
    gregasRows.push(ndRow('GEX', '<span style="color:'+gc+'">'+(o.gex_b>=0?'+':'')+o.gex_b.toFixed(2)+'B</span>'));
  }
  if (o.next_expiry) gregasRows.push(ndRow('Próx. Exp.', o.next_expiry));
  // Term structure
  const ts = o.term_structure || {};
  const tsKeys = Object.keys(ts).sort((a2,b2) => parseInt(a2) - parseInt(b2));
  if (tsKeys.length > 0) {
    gregasRows.push(ndSection('Term Structure'));
    tsKeys.forEach(k => {
      if (ts[k] != null) gregasRows.push(ndRow(k.replace('iv_','').replace('d',' dias'), fPct(ts[k])));
    });
  }
  // Convexity signals
  gregasRows.push(ndSection('Convexidade'));
  if (cv.hidden_opp != null) {
    const oc = cv.hidden_opp > 0.20 ? '#4ade80' : (cv.hidden_opp < -0.1 ? '#ef4444' : '#94a3b8');
    gregasRows.push(ndRow('Oportunidade', '<span style="color:'+oc+'">'+(cv.hidden_opp>0?'+':'')+cv.hidden_opp.toFixed(3)+'</span>'));
  }
  if (cv.fragility != null) {
    const fc2 = cv.fragility > 0.5 ? '#ef4444' : (cv.fragility > 0.25 ? '#f97316' : '#4ade80');
    gregasRows.push(ndRow('Fragilidade', '<span style="color:'+fc2+'">'+cv.fragility.toFixed(3)+'</span>'));
  }
  if (cv.halo_color) {
    const halo_labels = {'#22c55e':'IV barata + upside','#38bdf8':'IV barata',
                         '#c084fc':'IV cara','#ef4444':'IV cara + frágil','#f97316':'Put skew elevado'};
    gregasRows.push(ndRow('Sinal', '<span style="color:'+cv.halo_color+'">'+(halo_labels[cv.halo_color]||'Sinal ativo')+'</span>'));
  }
  if (gregasRows.filter(r => !r.includes('nd-section')).length === 0)
    gregasRows.push('<div class="nd-na">Sem dados de opções</div>');
  document.getElementById('nd-rows-gregas').innerHTML = gregasRows.join('');

  // ── ESTRUTURA — rede, RRG, MST, cluster ─────────────────────────────────────
  const estrutRows = [];
  // RRG
  estrutRows.push(ndSection('RRG — Força Relativa'));
  if (d.rrg_quadrant) {
    const qc2 = quad_colors[d.rrg_quadrant] || '#94a3b8';
    estrutRows.push(ndRow('Quadrante', '<span style="color:'+qc2+'">'+(quad_icons[d.rrg_quadrant]||'')+' '+
      (quad_labels[d.rrg_quadrant]||d.rrg_quadrant)+'</span>'));
  }
  if (d.rrg_rs_ratio != null) estrutRows.push(ndRow('RS-Ratio',   d.rrg_rs_ratio.toFixed(2)));
  if (d.rrg_rs_mom   != null) estrutRows.push(ndRow('RS-Mom',     d.rrg_rs_mom.toFixed(2)));
  if (d.rrg_alpha    != null) {
    const ac = d.rrg_alpha > 0 ? '#4ade80' : (d.rrg_alpha < 0 ? '#ef4444' : '#94a3b8');
    estrutRows.push(ndRow('Alpha RS', '<span style="color:'+ac+'">'+(d.rrg_alpha>0?'+':'')+d.rrg_alpha.toFixed(3)+'</span>'));
  }
  // Rede
  estrutRows.push(ndSection('Rede MST'));
  if (d.parent_id) estrutRows.push(ndRow('Cluster', d.parent_id));
  if (d.level != null) estrutRows.push(ndRow('Level', d.level));
  if (d.is_hub) estrutRows.push(ndRow('Centralidade', '<span style="color:#f59e0b">Hub &#9733;</span>'));
  if (d.weight != null) estrutRows.push(ndRow('Peso índice', (d.weight*100).toFixed(2)+'%'));
  if (a.beta != null) estrutRows.push(ndRow('Beta', a.beta.toFixed(2)));
  if (d.contagion != null && d.contagion > 0) {
    const cc = d.contagion > 0.5 ? '#ef4444' : (d.contagion > 0.25 ? '#f97316' : '#4ade80');
    estrutRows.push(ndRow('Contágio', '<span style="color:'+cc+'">'+d.contagion.toFixed(3)+'</span>'));
  }
  document.getElementById('nd-rows-estrutura').innerHTML = estrutRows.join('');

  // ── CTA — posicionamento + fluxo mecânico EOD ────────────────────────────────
  const ctaRows = [];
  // RRG momentum para contexto
  if (d.momentum != null) ctaRows.push(ndRow('Momentum Score', fScore(d.momentum)));
  // GEX + LETF flow
  ctaRows.push(ndSection('Fluxo Mecânico EOD'));
  if (fl.total_usd != null) {
    const mn = fl.total_usd / 1e6;
    const fc2 = mn > 0 ? '#4ade80' : (mn < 0 ? '#ef4444' : '#94a3b8');
    const fa  = mn > 0 ? '&#9650;' : (mn < 0 ? '&#9660;' : '&#9670;');
    ctaRows.push(ndRow('Fluxo Total EOD',
      '<span style="color:'+fc2+'">'+fa+' $'+Math.abs(mn).toFixed(1)+'M</span>'));
  }
  if (fl.letf_flow_usd != null && fl.letf_flow_usd !== 0) {
    const lm = fl.letf_flow_usd / 1e6;
    const lc = lm > 0 ? '#4ade80' : '#ef4444';
    ctaRows.push(ndRow('LETF Rebal.', '<span style="color:'+lc+'">'+(lm>0?'+':'')+lm.toFixed(1)+'M</span>'));
  }
  if (fl.gex_flow_usd != null && fl.gex_flow_usd !== 0) {
    const gm = fl.gex_flow_usd / 1e6;
    const gc3 = gm > 0 ? '#4ade80' : '#ef4444';
    ctaRows.push(ndRow('GEX Hedge', '<span style="color:'+gc3+'">'+(gm>0?'+':'')+gm.toFixed(1)+'M</span>'));
  }
  if (fl.direction) {
    const dc = {buy:'#4ade80', sell:'#ef4444', flat:'#94a3b8'}[fl.direction] || '#94a3b8';
    ctaRows.push(ndRow('Direção EOD', '<span style="color:'+dc+'">'+fl.direction.toUpperCase()+'</span>'));
  }
  // Dark pool / shadow flow
  if (d.dark_pool_score != null) {
    ctaRows.push(ndSection('Dark Pool'));
    const dpc = d.dark_pool_score > 0.2 ? '#4ade80' : (d.dark_pool_score < -0.2 ? '#ef4444' : '#94a3b8');
    ctaRows.push(ndRow('Dark Pool Score',
      '<div style="display:flex;align-items:center;gap:5px;min-width:100px">'
      + ndBar((d.dark_pool_score + 1) / 2, dpc)
      + '<span style="color:'+dpc+';font-size:11px">'+(d.dark_pool_score>0?'+':'')+d.dark_pool_score.toFixed(2)+'</span></div>'));
    if (d.dark_pct != null) ctaRows.push(ndRow('Dark Pool %', (d.dark_pct*100).toFixed(1)+'%'));
    if (d.dark_pct_delta != null && Math.abs(d.dark_pct_delta) > 0.01) {
      const dd = d.dark_pct_delta;
      const dc2 = dd > 0 ? '#4ade80' : '#ef4444';
      ctaRows.push(ndRow('Dark &Delta; 1W', '<span style="color:'+dc2+'">'+(dd>0?'+':''+(dd*100).toFixed(1))+'%</span>'));
    }
  }
  // CTA score
  if (d.cta_score != null) {
    ctaRows.push(ndSection('CTA Positioning'));
    const ctac = d.cta_score > 0.3 ? '#4ade80' : (d.cta_score < -0.3 ? '#ef4444' : '#94a3b8');
    const crowding_label = d.cta_crowding || '';
    ctaRows.push(ndRow('CTA Score',
      '<div style="display:flex;align-items:center;gap:5px;min-width:100px">'
      + ndBar((d.cta_score + 1) / 2, ctac)
      + '<span style="color:'+ctac+';font-size:11px">'+(d.cta_score>0?'+':'')+d.cta_score.toFixed(2)+'</span></div>'));
    if (crowding_label) {
      const cwc = {'extreme_long':'#ef4444','long':'#f97316','neutral':'#94a3b8','short':'#60a5fa','extreme_short':'#818cf8'}[crowding_label] || '#94a3b8';
      ctaRows.push(ndRow('Crowding', '<span style="color:'+cwc+'">'+crowding_label.replace('_',' ').toUpperCase()+'</span>'));
    }
  }
  if (ctaRows.filter(r => !r.includes('nd-section')).length === 0)
    ctaRows.push('<div class="nd-na">Sem dados de fluxo/posicionamento</div>');
  document.getElementById('nd-rows-cta').innerHTML = ctaRows.join('');

  // ── Summary text ─────────────────────────────────────────────────────────────
  const summaryEl = document.getElementById('nd-summary');
  const summaryText = buildNodeSummary(d);
  if (summaryText) {
    summaryEl.innerHTML = summaryText;
    summaryEl.style.display = 'block';
  } else {
    summaryEl.style.display = 'none';
  }

  // Ativa aba PAINEL por padrão
  document.querySelectorAll('.nd-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.nd-tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelector('.nd-tab[data-tab="painel"]').classList.add('active');
  document.getElementById('nd-tab-painel').classList.add('active');

  document.getElementById('node-detail').classList.add('visible');
}

function buildNodeSummary(d) {
  const parts = [];
  const cv = d.convexity || {};
  if (cv.iv_rank != null && cv.iv_rank < 0.35 && cv.hidden_opp > 0.2) {
    parts.push('&#128994; IV barata (' + Math.round(cv.iv_rank*100) + '%ile) &mdash; convexidade atrativa.');
  }
  if (cv.fragility != null && cv.fragility > 0.5) {
    parts.push('&#128308; Estrutura frágil &mdash; risco de reversão.');
  }
  if ((d.contagion || 0) > 0.4) {
    parts.push('&#9889; Contágio elevado da rede (' + d.contagion.toFixed(2) + ').');
  }
  if (d.is_hub) parts.push('&#11088; Hub MST &mdash; alta conectividade sistêmica.');
  return parts.join('  ');
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
    case 'h': case 'H':
      document.getElementById('btn-tree-layout').click(); break;
    case 'c': case 'C':
      CP_TOGGLE.click(); break;
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
    case '1': activateTab('painel');    break;
    case '2': activateTab('risco');     break;
    case '3': activateTab('gregas');    break;
    case '4': activateTab('estrutura'); break;
    case '5': activateTab('cta');       break;
  }
});
"""


# ── Portfolio tab renderer ────────────────────────────────────────────────────

_INITIAL_CAPITAL = 100_000.0   # capital inicial configurável


def _render_pnl_section(market_prices: "dict | None") -> str:
    """P&L tracker: capital strip + equity curve + trade history."""
    import json as _json
    from pathlib import Path as _P

    trade_log_path = _P.home() / "agente-workspace" / "portfolio" / "trade_log.json"
    if not trade_log_path.exists():
        return (
            '<div class="pt-section" style="color:#475569;font-size:11px;padding:18px">'
            'Nenhum trade registrado ainda — rode o pipeline para abrir posições.'
            '</div>'
        )
    try:
        trades: list = _json.loads(trade_log_path.read_text(encoding="utf-8"))
    except Exception:
        return '<div class="pt-section" style="color:#ef4444;font-size:11px">Erro ao ler trade_log.json</div>'

    mp = market_prices or {}

    # Normaliza ticker: remove sufixos Bloomberg ("AAPL US Equity" → "AAPL")
    def _norm_ticker(tk: str) -> str:
        for sfx in (" US Equity", " US EQUITY", " Equity", " Index", " Comdty", " Curncy"):
            if tk.endswith(sfx):
                return tk[: -len(sfx)].strip()
        return tk.strip()

    def _get_price(ticker: str) -> float:
        p = (mp.get(ticker) or mp.get(_norm_ticker(ticker)) or {}).get("price") or 0
        return float(p)

    # ── Computa equity curve ────────────────────────────────────────────────
    closed = [t for t in trades if t.get("status") == "closed" and t.get("exit_date")]
    # Deduplica abertos: mantém apenas o trade mais recente por ticker
    _seen: dict[str, dict] = {}
    for t in sorted(trades, key=lambda x: (x.get("entry_date",""), x.get("entry_time",""))):
        if t.get("status") == "open":
            _seen[_norm_ticker(t.get("ticker",""))] = t
    opened = list(_seen.values())

    closed_sorted = sorted(closed, key=lambda t: (t.get("exit_date") or "", t.get("exit_time") or ""))

    # P&L acumulado por data
    equity_by_date: dict[str, float] = {}
    running = _INITIAL_CAPITAL
    for t in closed_sorted:
        pnl = float(t.get("pnl_realized") or 0)
        running += pnl
        dt = t["exit_date"]
        equity_by_date[dt] = running

    # Calcula P&L não realizado das posições abertas
    pnl_unreal = 0.0
    for t in opened:
        entry  = float(t.get("entry_price") or 0)
        shares = float(t.get("shares") or 0)
        ticker = t.get("ticker", "")
        cur    = _get_price(ticker)
        if cur > 0 and entry > 0 and shares > 0:
            if t.get("direction") == "long":
                pnl_unreal += (cur - entry) * shares
            else:
                pnl_unreal += (entry - cur) * shares

    pnl_real_total = running - _INITIAL_CAPITAL
    equity_now     = running + pnl_unreal
    total_pnl      = equity_now - _INITIAL_CAPITAL

    # Métricas de performance
    n_closed  = len(closed)
    n_wins    = sum(1 for t in closed if float(t.get("pnl_realized") or 0) > 0)
    win_rate  = n_wins / n_closed if n_closed > 0 else 0
    avg_win   = sum(float(t.get("pnl_realized") or 0) for t in closed if float(t.get("pnl_realized") or 0) > 0) / max(n_wins, 1)
    avg_loss  = sum(float(t.get("pnl_realized") or 0) for t in closed if float(t.get("pnl_realized") or 0) <= 0) / max(n_closed - n_wins, 1)

    # ── Capital strip ───────────────────────────────────────────────────────
    def cap_card(label: str, val: str, color: str = "#94a3b8") -> str:
        return (
            f"<div style='flex:1;min-width:110px;background:#0a0f1a;border:1px solid #1a2535;"
            f"border-radius:7px;padding:10px 14px'>"
            f"<div style='font-size:9px;color:#475569;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px'>{label}</div>"
            f"<div style='font-size:20px;font-weight:900;font-family:monospace;color:{color}'>{val}</div>"
            f"</div>"
        )

    pnl_color  = "#10b981" if total_pnl >= 0 else "#ef4444"
    real_color = "#10b981" if pnl_real_total >= 0 else "#ef4444"
    ur_color   = "#10b981" if pnl_unreal >= 0 else "#ef4444"

    capital_strip = (
        "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px'>"
        + cap_card("Capital Inicial", f"${_INITIAL_CAPITAL:,.0f}", "#64748b")
        + cap_card("Equity Atual", f"${equity_now:,.0f}", pnl_color)
        + cap_card("P&amp;L Total", f"{'+' if total_pnl>=0 else ''}{total_pnl/1000:.1f}k ({total_pnl/_INITIAL_CAPITAL:+.1%})", pnl_color)
        + cap_card("Realizado", f"{'+' if pnl_real_total>=0 else ''}{pnl_real_total/1000:.1f}k", real_color)
        + cap_card("Não Realizado", f"{'+' if pnl_unreal>=0 else ''}{pnl_unreal/1000:.1f}k", ur_color)
        + cap_card("Win Rate", f"{win_rate:.0%} ({n_wins}/{n_closed})", "#f59e0b")
        + "</div>"
    )

    # ── Equity curve (Canvas JS) ────────────────────────────────────────────
    # Ponto inicial + um ponto por date com fechamento + ponto atual (com unrealized)
    from datetime import date as _date
    curve_dates = ["Start"] + list(equity_by_date.keys())
    curve_vals  = [_INITIAL_CAPITAL] + list(equity_by_date.values())
    # Adiciona hoje com unrealized
    today_str = str(_date.today())
    if today_str not in equity_by_date:
        curve_dates.append(today_str + "*")
        curve_vals.append(equity_now)

    curve_json  = _json.dumps(curve_vals)
    labels_json = _json.dumps(curve_dates)

    equity_chart = f"""
<div class="pt-section" style="margin-bottom:16px">
  <div class="pt-title">Equity Curve — P&amp;L Acumulado</div>
  <canvas id="equityCanvas" style="width:100%;height:220px;display:block;background:#0a0f1a;border-radius:4px"></canvas>
  <div style="font-size:9px;color:#334155;margin-top:4px">* inclui P&amp;L não realizado</div>
</div>
<script>
window._drawEquityCurve = function() {{
  const canvas = document.getElementById('equityCanvas');
  if (!canvas) return;
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  if (W < 10 || H < 10) return;  // aba ainda oculta — tentar novamente depois
  const dpr = window.devicePixelRatio || 1;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const pad = {{l:60, r:20, t:16, b:36}};
  const data = {curve_json};
  const labels = {labels_json};
  const N = data.length;
  if (N < 2) {{ ctx.fillStyle='#0a0f1a'; ctx.fillRect(0,0,W,H); return; }}
  const minV = Math.min(...data) * 0.999;
  const maxV = Math.max(...data) * 1.001;
  const rng  = maxV - minV || 1;

  function xOf(i) {{ return pad.l + i / Math.max(N-1,1) * (W - pad.l - pad.r); }}
  function yOf(v) {{ return pad.t + (1 - (v - minV) / rng) * (H - pad.t - pad.b); }}

  // Background
  ctx.fillStyle = '#0a0f1a'; ctx.fillRect(0,0,W,H);

  // Zero line ($100k)
  const y0 = yOf({_INITIAL_CAPITAL});
  ctx.setLineDash([4,4]);
  ctx.strokeStyle = '#1e3a5f'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, y0); ctx.lineTo(W-pad.r, y0); ctx.stroke();
  ctx.setLineDash([]);

  // Gradient fill
  const lastAbove = data[N-1] >= {_INITIAL_CAPITAL};
  const grad = ctx.createLinearGradient(0, pad.t, 0, H-pad.b);
  grad.addColorStop(0, lastAbove ? 'rgba(16,185,129,.30)' : 'rgba(239,68,68,.25)');
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.beginPath();
  ctx.moveTo(xOf(0), yOf(data[0]));
  for (let i=1; i<N; i++) ctx.lineTo(xOf(i), yOf(data[i]));
  ctx.lineTo(xOf(N-1), H-pad.b);
  ctx.lineTo(xOf(0), H-pad.b);
  ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();

  // Line
  ctx.beginPath();
  ctx.moveTo(xOf(0), yOf(data[0]));
  for (let i=1; i<N; i++) {{
    const isUnreal = labels[i].endsWith('*');
    ctx.setLineDash(isUnreal ? [5,4] : []);
    ctx.strokeStyle = data[i] >= {_INITIAL_CAPITAL} ? '#10b981' : '#ef4444';
    ctx.lineWidth = 2;
    ctx.lineTo(xOf(i), yOf(data[i]));
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(xOf(i), yOf(data[i]));
  }}
  ctx.setLineDash([]);

  // Dots
  for (let i=0; i<N; i++) {{
    ctx.beginPath();
    ctx.arc(xOf(i), yOf(data[i]), i===N-1?4:2.5, 0, 2*Math.PI);
    ctx.fillStyle = data[i] >= {_INITIAL_CAPITAL} ? '#10b981' : '#ef4444';
    ctx.fill();
  }}

  // Y axis labels
  ctx.fillStyle = '#475569'; ctx.font = '10px monospace'; ctx.textAlign = 'right';
  for (let s=0; s<=5; s++) {{
    const v = minV + (maxV-minV)*s/5;
    ctx.fillText('$' + (v/1000).toFixed(1) + 'k', pad.l-4, yOf(v)+3);
  }}

  // X axis labels
  ctx.textAlign = 'center'; ctx.fillStyle = '#334155';
  const step = Math.ceil(N / Math.min(N, 8));
  for (let i=0; i<N; i+=step) {{
    const lbl = labels[i].replace('Start','Início').replace('*','');
    ctx.fillText(lbl.length > 10 ? lbl.slice(5) : lbl, xOf(i), H-pad.b+14);
  }}
  canvas._drawn = true;
}};
// Tenta desenhar agora; se a aba estiver oculta (offsetWidth=0), usa ResizeObserver
(function() {{
  function tryDraw() {{
    const c = document.getElementById('equityCanvas');
    if (c && c.offsetWidth > 10) {{ window._drawEquityCurve(); return; }}
    // Aba oculta — observa quando ficar visível
    if (typeof ResizeObserver !== 'undefined' && c && !c._equityObs) {{
      c._equityObs = new ResizeObserver(function(entries) {{
        if (entries[0].contentRect.width > 10) {{
          window._drawEquityCurve();
          c._equityObs.disconnect();
        }}
      }});
      c._equityObs.observe(c);
    }}
  }}
  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', tryDraw);
  }} else {{
    requestAnimationFrame(tryDraw);
  }}
}})();
</script>
"""

    # ── Trade history table ─────────────────────────────────────────────────
    all_trades = sorted(trades, key=lambda t: (t.get("exit_date") or t.get("entry_date",""), t.get("exit_time") or t.get("entry_time","")), reverse=True)

    def trade_row(t: dict) -> str:
        status  = t.get("status","open")
        ticker  = t.get("ticker","?")
        direc   = t.get("direction","—")
        e_date  = t.get("entry_date","—")
        x_date  = t.get("exit_date") or "aberto"
        e_price = t.get("entry_price")
        x_price = t.get("exit_price")
        pnl_r   = t.get("pnl_realized")
        pnl_pct = t.get("pnl_pct")
        target  = t.get("target_usd") or 0
        conv    = t.get("conviction","—")

        # P&L não realizado para abertas
        if status == "open":
            cur = _get_price(ticker)
            shares = float(t.get("shares") or 0)
            ep = float(e_price or 0)
            if cur > 0 and ep > 0 and shares > 0:
                unreal = (cur - ep) * shares if direc == "long" else (ep - cur) * shares
                pnl_r  = unreal
                pnl_pct = unreal / abs(target) if target else 0

        pnl_color = "#10b981" if (pnl_r or 0) >= 0 else "#ef4444"
        dir_color = "#10b981" if direc == "long" else "#ef4444"
        status_badge = (
            '<span style="background:#10b98122;color:#10b981;border-radius:3px;padding:1px 6px;font-size:9px">ABERTO</span>'
            if status == "open" else
            '<span style="background:#33415522;color:#64748b;border-radius:3px;padding:1px 6px;font-size:9px">FECHADO</span>'
        )
        pnl_str = f"${pnl_r:+,.0f} ({pnl_pct:+.1%})" if pnl_r is not None else "—"

        return (
            f"<tr>"
            f"<td style='font-weight:800;color:#f1f5f9;font-size:13px'>{ticker}</td>"
            f"<td style='color:{dir_color};font-weight:700;font-size:12px'>{direc.upper()}</td>"
            f"<td>{status_badge}</td>"
            f"<td style='color:#64748b;font-size:11px;font-family:monospace'>{e_date}</td>"
            f"<td style='color:#64748b;font-size:11px;font-family:monospace'>{x_date}</td>"
            f"<td style='font-family:monospace;font-size:12px'>${float(e_price or 0):,.2f}</td>"
            f"<td style='font-family:monospace;font-size:12px'>{f'${float(x_price):,.2f}' if x_price else '—'}</td>"
            f"<td style='font-size:12px;font-weight:700;color:{pnl_color};font-family:monospace'>{pnl_str}</td>"
            f"<td style='color:#475569;font-size:11px'>{f'${abs(target):,.0f}'}</td>"
            f"<td style='color:#94a3b8;font-size:10px'>{conv}</td>"
            f"</tr>"
        )

    history_rows = "".join(trade_row(t) for t in all_trades[:50])
    history_html = (
        "<div class='pt-section' style='margin-bottom:16px'>"
        "<div class='pt-title'>Histórico de Trades</div>"
        "<div style='overflow-x:auto'>"
        "<table class='pt-table' style='font-size:12px'>"
        "<thead><tr>"
        "<th>Ticker</th><th>Dir</th><th>Status</th>"
        "<th>Entrada</th><th>Saída</th>"
        "<th>Px Entrada</th><th>Px Saída</th>"
        "<th>P&amp;L</th><th>Capital</th><th>Conv</th>"
        "</tr></thead>"
        f"<tbody>{history_rows}</tbody>"
        "</table></div></div>"
    ) if all_trades else ""

    # ── Performance stats ───────────────────────────────────────────────────
    stats_html = (
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px'>"
        + "".join(
            f"<div style='background:#0a0f1a;border:1px solid #1a2535;border-radius:6px;padding:8px 14px;text-align:center'>"
            f"<div style='font-size:9px;color:#475569;letter-spacing:1px;margin-bottom:3px'>{l}</div>"
            f"<div style='font-size:15px;font-weight:700;font-family:monospace;color:{c}'>{v}</div>"
            f"</div>"
            for l, v, c in [
                ("Trades Fechados",  str(n_closed), "#94a3b8"),
                ("Vencedores",       str(n_wins), "#10b981"),
                ("Win Rate",         f"{win_rate:.0%}", "#f59e0b"),
                ("Avg Win",          f"${avg_win:+,.0f}" if n_wins else "—", "#10b981"),
                ("Avg Loss",         f"${avg_loss:+,.0f}" if n_closed-n_wins else "—", "#ef4444"),
                ("Posições Abertas", str(len(opened)), "#38bdf8"),
            ]
        )
        + "</div>"
    )

    return capital_strip + stats_html + equity_chart + history_html


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

    pnl_section = _render_pnl_section(market_prices)

    return f"""
{pnl_section}
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
    Carrega o editorial diário para a aba Informações de Mercado.
    Usa <iframe srcdoc> com o conteúdo completo do _brief.html para preservar
    CSS e layout integralmente sem problemas de same-origin file://.
    Retorna uma tag <iframe srcdoc=...>, ou string vazia se não encontrado.
    """
    try:
        from app.storage.paths import workspace
        from pathlib import Path as _P
        bundle_dir = _P(workspace.bundles) / str(bundle.run_date)
        # Prefere _brief.html mais recente, depois qualquer _week_ahead_brief.html
        candidates = sorted(
            list(bundle_dir.glob("*_brief.html")) + list(bundle_dir.glob("*_week_ahead_brief.html")),
            key=lambda p: p.stat().st_size, reverse=True,  # maior brief = mais conteúdo
        )
        if not candidates:
            return ""
        raw = candidates[0].read_text(encoding="utf-8")
        # srcdoc precisa de & e " escapados
        srcdoc = raw.replace("&", "&amp;").replace('"', "&quot;")
        return f'<iframe srcdoc="{srcdoc}" style="width:100%;flex:1;border:none;display:block;background:#06080f;min-height:0"></iframe>'
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
    rrg_result=None,  # RRGResult | None
    desk_intel=None,  # DeskIntelligenceResult | None
    options_snapshot=None,  # OptionsSnapshot | None — importado via options-import
) -> str:
    # Resolve desk_intel e rrg antecipadamente para passá-los ao build_from_bundle
    _desk_intel_early = desk_intel or getattr(portfolio, "_desk_intel", None)
    _rrg_early        = rrg_result  or getattr(portfolio, "_rrg_result", None)
    if graph_data is None:
        from app.desk.graph_engine import build_from_bundle
        _cached_opts = getattr(portfolio, "_options_map", None) if portfolio else None
        graph_data = build_from_bundle(
            bundle, curation_result,
            rrg_result=_rrg_early,
            desk_intel=_desk_intel_early,
            cached_options=_cached_opts,
        )
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
    # ── Normaliza tickers Bloomberg → yfinance/IBKR ──────────────────────────
    _BBG_SUFFIXES_MP = (" US EQUITY", " US Equity", " US equity", " INDEX", " Index",
                        " COMDTY", " Comdty", " CURNCY", " Curncy", " EQUITY", " Equity")
    _BBG_YF_MAP_MP = {
        "SPX": "^GSPC", "VIX": "^VIX", "NDX": "^NDX", "RTY": "^RUT",
        "DXY": "DX-Y.NYB", "XBT": "BTC-USD", "GC1": "GC=F", "CL1": "CL=F",
    }
    _mp_norm: dict = {}
    for _k, _v in mp.items():
        if _k.startswith("__"):
            _mp_norm[_k] = _v
            continue
        _t = _k
        for _sfx in _BBG_SUFFIXES_MP:
            if _t.endswith(_sfx):
                _t = _t[: -len(_sfx)].strip()
                break
        _t = _t.replace("/", "-")
        _t = _BBG_YF_MAP_MP.get(_t, _t)
        _mp_norm[_t] = _v
    mp = _mp_norm
    # ── Enriquece daily_return via yfinance para tickers sem retorno ──────────
    try:
        import yfinance as _yf_mp
        _need_ret_mp = [t for t, d in mp.items()
                        if not t.startswith("__") and isinstance(d, dict) and not d.get("daily_return")]
        if _need_ret_mp:
            _yfd = _yf_mp.download(_need_ret_mp, period="2d", progress=False, auto_adjust=True)
            _cl = _yfd["Close"] if "Close" in getattr(_yfd, "columns", []) else _yfd
            if hasattr(_cl, "columns"):
                for _t in _need_ret_mp:
                    if _t in _cl.columns:
                        _s = _cl[_t].dropna()
                        if len(_s) >= 2:
                            _r = (_s.iloc[-1] - _s.iloc[-2]) / _s.iloc[-2]
                            mp[_t]["daily_return"] = round(float(_r), 6)
                            mp[_t]["price"] = round(float(_s.iloc[-1]), 4)
            elif len(_need_ret_mp) == 1:
                _s = _cl.dropna()
                if len(_s) >= 2:
                    _r = (_s.iloc[-1] - _s.iloc[-2]) / _s.iloc[-2]
                    mp[_need_ret_mp[0]]["daily_return"] = round(float(_r), 6)
                    mp[_need_ret_mp[0]]["price"] = round(float(_s.iloc[-1]), 4)
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

    # ── Regime badge: usa desk_intel como fallback se network retornou unknown ──
    _regime_for_badge = regime
    if (not regime or regime.get("regime", "unknown") == "unknown") and _desk_intel_early:
        _di = _desk_intel_early
        _mr = getattr(_di, "market_regime", None) or ""
        # Mapeia market_regime do desk_intel → risk_on / risk_off / transition
        _regime_map = {
            "risk_on_momentum":   "risk_on",
            "risk_off_defensive": "risk_off",
            "vol_squeeze":        "transition",
            "narrative_driven":   "transition",
            "mechanical_passive": "transition",
            "dispersed_rotation": "transition",
            "stress":             "risk_off",
        }
        _mapped = _regime_map.get(_mr, "unknown")
        _conf   = getattr(_di, "regime_confidence", 0) or 0
        _regime_for_badge = {"regime": _mapped, "confidence": _conf, "avg_correlation": 0, "corr_entropy": 0}
    regime_html  = _regime_badge(_regime_for_badge)

    score_labels = {"rational":"Rational","behavioral":"Behavioral",
                    "entropy":"Entropy","arbitration":"Arbitration","allocation":"Allocation"}
    if scores:
        scores_html = "".join(_score_bar_html(lbl, scores.get(k)) for k, lbl in score_labels.items())
    elif _desk_intel_early and getattr(_desk_intel_early, "regime_adj_scores", None):
        # Fallback: mostra top scores ajustados do desk_intel
        _adj = _desk_intel_early.regime_adj_scores
        _top5 = sorted(_adj.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        scores_html = "".join(_score_bar_html(t, v) for t, v in _top5)
    else:
        scores_html = '<div style="font-size:11px;color:#475569;padding:6px 0">Aguardando análise</div>'
    market_rows  = _market_table_rows(mp)
    vix_term_html     = _vix_term_html(vix_term)
    risk_heatmap_html = _risk_heatmap_html(graph_data)
    live_network_html = _live_network_html(live_network)
    flow_panel_html   = _flow_panel_html(flow_pred_panel)

    portfolio_tab_html = _render_portfolio_tab(portfolio, mp or (bundle.market_prices if bundle else None), _pt_flow_pred)

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

    # ── Desk Radar tab ────────────────────────────────────────────────────────
    # Reutiliza os valores já resolvidos no início
    _desk_intel = _desk_intel_early
    _rrg_result = _rrg_early
    try:
        from app.views.desk_radar import render_desk_radar_tab
        from app.analysis.alpha_signals import AssetSignal as _AS
        _signals_for_radar = {}
        # Tenta extrair signals do portfolio_pipeline (armazenado no objeto ou reconstruído)
        if hasattr(portfolio, "_signals"):
            _signals_for_radar = portfolio._signals or {}
        _network_for_radar = (graph_data or {}).get("_network_result")
        radar_tab_html = render_desk_radar_tab(
            result=_desk_intel,
            signals=_signals_for_radar,
            rrg_result=_rrg_result,
            market_prices=mp,
            network_result=_network_for_radar,
        )
    except Exception as _exc:
        _log.warning("desk_radar_render_failed", error=str(_exc))
        radar_tab_html = (
            '<div style="padding:40px;text-align:center;color:#6b7280">'
            f'Desk Radar indisponível: {str(_exc)[:120]}'
            '</div>'
        )

    # ── TradingView tab ──────────────────────────────────────────────────────
    try:
        from app.views.tradingview_tab import render_tradingview_tab
        tradingview_tab_html = render_tradingview_tab(market_prices=mp)
    except Exception as _exc:
        _log.warning("tradingview_tab_render_failed", error=str(_exc))
        tradingview_tab_html = (
            '<div style="padding:40px;text-align:center;color:#6b7280">'
            f'TradingView indisponível: {str(_exc)[:120]}'
            '</div>'
        )

    # ── Options tab ──────────────────────────────────────────────────────────
    _options_snap = options_snapshot
    if _options_snap is None:
        try:
            from app.providers.options_store import options_store as _os
            _options_snap = _os.load_latest()
        except Exception:
            pass

    try:
        from app.views.options_desk import render_options_tab
        _jarvis_html = None
        if _options_snap is not None:
            try:
                from app.providers.options_store import options_store as _os2
                _jarvis_html = _os2.load_jarvis_html(_options_snap)
            except Exception:
                pass
        _cta_result   = getattr(portfolio, "_cta_result",   None) if portfolio else None
        _shadow_flow  = getattr(portfolio, "_shadow_flow",   None) if portfolio else None
        _vol_regime_o = getattr(portfolio, "_vol_regime",    None) if portfolio else None
        _signals_o    = getattr(portfolio, "_signals",       {})   if portfolio else {}
        options_tab_html = render_options_tab(
            _options_snap, _jarvis_html,
            cta_result=_cta_result,
            shadow_flow=_shadow_flow,
            vol_regime=_vol_regime_o,
            signals=_signals_o,
        )
    except Exception as _exc:
        _log.warning("options_tab_render_failed", error=str(_exc))
        options_tab_html = (
            '<div style="padding:40px;text-align:center;color:#6b7280">'
            f'Opções indisponível: {str(_exc)[:120]}'
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
    refreshed   = mp.get("__refreshed_at__", gen_time) if live_mode else gen_time

    html = f"""<!DOCTYPE html>
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
  <button class="layer-btn active" data-layer="structure" title="Camada estrutural — hierarquia, MST, RMT">Struct</button>
  <button class="layer-btn active" data-layer="flow" title="Camada de fluxo — rotação RRG">Flow</button>
  <button class="layer-btn active" data-layer="convexity" title="Camada de convexidade — IV rank, skew, fragility">Convex</button>
  <div class="sep"></div>
  <span class="spacer"></span>
  <span id="expand-count" style="font-size:10px;color:#64748b"></span>
  <div class="sep"></div>
  <button class="ctrl-btn" id="btn-back" style="display:none;border-color:#38bdf8;color:#38bdf8">&larr; Back</button>
  <button class="ctrl-btn" id="btn-fit">Fit <span style="color:#4a6380">[F]</span></button>
  <button class="ctrl-btn" id="btn-relayout">Layout <span style="color:#4a6380">[R]</span></button>
  <button class="ctrl-btn" id="btn-tree-layout">Force <span style="color:#4a6380">[H]</span></button>
  <button id="cp-toggle">Contagion <span style="color:#4a6380">[C]</span></button>
  <button class="ctrl-btn danger" id="btn-reset">Reset</button>
</div>

<div id="main-tabs-bar">
  <button class="main-tab active" onclick="switchMainTab('desk',this)">Desk Portfólio</button>
  <button class="main-tab" onclick="switchMainTab('portfolio',this)">Alocação</button>
  <button class="main-tab" onclick="switchMainTab('editorial',this)">Informações de Mercado</button>
  <button class="main-tab" onclick="switchMainTab('radar',this)" style="color:#818cf8;border-color:#818cf840">Desk Radar ◈</button>
  <button class="main-tab" onclick="switchMainTab('options',this)" style="color:#00d4e8;border-color:#00d4e840">Opções ◈</button>
  <button class="main-tab" onclick="switchMainTab('tradingview',this)" style="color:#2962ff;border-color:#2962ff40">TradingView ◈</button>
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
        <div id="nd-header">
          <div id="nd-ticker-badge">—</div>
          <div id="nd-header-right">
            <div id="nd-label">—</div>
            <div id="nd-sub">—</div>
            <div id="nd-quadrant-badge" style="display:none"></div>
          </div>
        </div>
        <div id="nd-summary" style="font-size:10px;color:#7dd3fc;margin-bottom:8px;line-height:1.45;
             background:#071525;border-left:2px solid #1e4a7a;border-radius:0 4px 4px 0;
             padding:5px 7px;display:none"></div>
        <div class="nd-tabs">
          <button class="nd-tab active" data-tab="painel">PAINEL</button>
          <button class="nd-tab" data-tab="risco">RISCO</button>
          <button class="nd-tab" data-tab="gregas">GREGAS</button>
          <button class="nd-tab" data-tab="estrutura">ESTRU.</button>
          <button class="nd-tab" data-tab="cta">CTA</button>
        </div>
        <div id="nd-tab-painel"    class="nd-tab-panel active"><div id="nd-rows-painel"></div></div>
        <div id="nd-tab-risco"     class="nd-tab-panel"><div id="nd-rows-risco"></div></div>
        <div id="nd-tab-gregas"    class="nd-tab-panel"><div id="nd-rows-gregas"></div></div>
        <div id="nd-tab-estrutura" class="nd-tab-panel"><div id="nd-rows-estrutura"></div></div>
        <div id="nd-tab-cta"       class="nd-tab-panel"><div id="nd-rows-cta"></div></div>

        <!-- Keep legacy IDs for backward compat with JS that may reference them -->
        <div id="nd-rows" style="display:none"></div>
        <div id="nd-rows-val" style="display:none"></div>
        <div id="nd-rows-qual" style="display:none"></div>
        <div id="nd-rows-opt" style="display:none"></div>
        <div id="nd-rows-risk" style="display:none"></div>
        <div id="nd-rows-flow" style="display:none"></div>
        <div id="nd-rows-struct" style="display:none"></div>
        <div id="nd-rows-nodeflow" style="display:none"></div>
        <div id="nd-rows-convex" style="display:none"></div>
      </div>

      <!-- Contagion side panel -->
      <div id="contagion-panel">
        <div id="cp-header">
          <div>
            <div id="cp-title">Contagion Map</div>
            <div id="cp-asset">—</div>
          </div>
          <button id="cp-close">&times;</button>
        </div>
        <div id="cp-body"></div>
      </div>

      <div id="hint-bar">
        <div class="leg"><div class="leg-line" style="background:#22c55e"></div> +&rho;</div>
        <div class="leg"><div class="leg-line" style="background:#ef4444"></div> &minus;&rho;</div>
        <div class="leg"><div class="leg-dot" style="background:#f59e0b"></div> Hub</div>
        <div class="leg"><div class="leg-dot" style="background:#818cf8"></div> Expan.</div>
        <div id="hint-text">
          [+] drill &middot; dbl-click zoom &middot; hover=stats &middot; right-click=isolate &middot; F fit &middot; R layout &middot; H hier. &middot; C contágio &middot; 1-5 tabs
        </div>
      </div>
    </div>
  </div>
</div>

<div id="portfolio-view" class="main-view" style="flex-direction:column;overflow-y:auto;padding:18px 24px;gap:16px;background:#060a12">
  {portfolio_tab_html}
</div>

<div id="editorial-view" class="main-view" style="overflow:hidden;background:#06080f">
  {editorial_content}
</div>

"""

    # Concatena tabs com CSS/JS próprio fora do f-string para evitar
    # conflito de chaves {} com a sintaxe de f-string.
    # IMPORTANTE: usar + explícito em cada linha para evitar implicit string concat.
    _radar_open   = '<div id="radar-view" class="main-view" style="flex-direction:column;overflow-y:auto;background:#060a12;padding:0;width:100%;align-items:stretch">\n'
    _radar_close  = '\n</div>\n\n'
    _opts_open    = '<div id="options-view" class="main-view" style="flex-direction:column;overflow-y:auto;background:#020810;padding:0;width:100%;align-items:stretch">\n'
    _opts_close   = '\n</div>\n\n'
    _tv_open      = '<div id="tradingview-view" class="main-view" style="overflow-y:auto;background:#060a12;padding:0">\n'
    _tv_close     = '\n</div>\n\n'
    _switch_js    = (
        '<script>\n'
        'function switchMainTab(name, btn) {\n'
        '  document.querySelectorAll(\'.main-tab\').forEach(t => t.classList.remove(\'active\'));\n'
        '  document.querySelectorAll(\'.main-view\').forEach(v => v.classList.remove(\'active\'));\n'
        '  if (btn) btn.classList.add(\'active\');\n'
        '  const view = document.getElementById(name + \'-view\');\n'
        '  if (view) view.classList.add(\'active\');\n'
        '  // Renderiza equity curve ao abrir aba Alocação\n'
        '  if (name === \'portfolio\') {\n'
        '    requestAnimationFrame(function() {\n'
        '      if (typeof window._drawEquityCurve === \'function\') window._drawEquityCurve();\n'
        '    });\n'
        '  }\n'
        '}\n'
        '</script>\n'
        '</body>\n</html>'
    )
    html = (html
            + _radar_open
            + radar_tab_html
            + _radar_close
            + _opts_open
            + options_tab_html
            + _opts_close
            + _tv_open
            + tradingview_tab_html
            + _tv_close
            + '<script>\n'
            + cytoscape_js
            + '\n</script>\n'
            + '<script>\n'
            + js_code
            + '\n</script>\n'
            + _switch_js)
    return html


# ── Save ──────────────────────────────────────────────────────────────────────

def save_macro_desk_v2(
    bundle: "DailyIngestionBundle",
    curation_result: "CurationResult | None" = None,
    graph_data: "dict | None" = None,
    live_mode: bool = False,
    portfolio=None,   # PortfolioResult | None
    rrg_result=None,  # RRGResult | None
    desk_intel=None,  # DeskIntelligenceResult | None
    options_snapshot=None,  # OptionsSnapshot | None
) -> Path:
    # Extrai rrg e desk_intel do portfolio se não foram passados explicitamente
    _rrg    = rrg_result   or getattr(portfolio, "_rrg_result",  None)
    _dintel = desk_intel   or getattr(portfolio, "_desk_intel",  None)
    try:
        html = generate_macro_desk_v2_html(
            bundle, graph_data, curation_result,
            live_mode=live_mode, portfolio=portfolio,
            rrg_result=_rrg, desk_intel=_dintel,
            options_snapshot=options_snapshot,
        )
    except Exception as exc:
        _log.error("macro_desk_v2_failed", error=str(exc), exc_info=True)
        raise

    out_dir  = workspace.bundles / str(bundle.run_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{bundle.run_id}_desk_v2.html"
    out_path.write_text(html, encoding="utf-8")
    _log.info("macro_desk_v2_saved", path=str(out_path))
    return out_path
