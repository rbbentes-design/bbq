"""
Week Ahead Brief — HTML standalone para segunda-feira.

Gera um relatório visual com:
  1. Header: data + narrativa primária
  2. Calendário econômico da semana (tabela com badges de impacto)
  3. Dashboard macro FRED: 6 categorias, sparklines SVG, variação colorida
  4. Tabela de preços de mercado (principais ativos)
  5. Texto editorial week_ahead (do writer)

Output: self-contained HTML, sem dependências externas (SVG inline, CSS puro).
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.audit.logger import get_logger
from app.models.daily_ingestion_bundle import DailyIngestionBundle
from app.storage.paths import workspace

_log = get_logger("views.week_ahead_brief")


# ── Helpers SVG ───────────────────────────────────────────────────────────────

def _sparkline(values: list[float | None], width: int = 80, height: int = 28,
               color: str = "#38bdf8") -> str:
    """Gera sparkline SVG inline a partir de lista de valores."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return f'<svg width="{width}" height="{height}"></svg>'

    mn, mx = min(vals), max(vals)
    rng = mx - mn or 1

    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        x = int(i / (n - 1) * (width - 4)) + 2
        y = int((1 - (v - mn) / rng) * (height - 6)) + 3
        pts.append(f"{x},{y}")

    polyline = " ".join(pts)

    # cor depende da direção
    up = vals[-1] >= vals[0]
    stroke = "#22c55e" if up else "#ef4444"

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'style="overflow:visible">'
        f'<polyline points="{polyline}" fill="none" stroke="{stroke}" '
        f'stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{pts[-1].split(",")[0]}" cy="{pts[-1].split(",")[1]}" '
        f'r="2.5" fill="{stroke}"/>'
        f'</svg>'
    )


def _fmt_change(change: float | None, unit: str = "") -> str:
    if change is None:
        return '<span class="dim">—</span>'
    sign = "+" if change > 0 else ""
    cls = "up" if change > 0 else ("down" if change < 0 else "flat")
    arrow = "▲" if change > 0 else ("▼" if change < 0 else "─")
    return f'<span class="{cls}">{arrow} {sign}{change:+.3g}</span>'


def _fmt_val(val: float | None, unit: str = "") -> str:
    if val is None:
        return "—"
    # formata por unidade
    if unit in ("%", "pp"):
        return f"{val:.2f}{unit}"
    if "USD bi" in unit:
        return f"${val:,.0f}B"
    if "USD mi" in unit:
        return f"${val:,.0f}M"
    if "mil" in unit:
        return f"{val:,.0f}k"
    if "índice" in unit:
        return f"{val:.2f}"
    return f"{val:.2f}"


def _pct_str(v: float | None) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v*100:.2f}%"


def _color_pct(v: float | None) -> str:
    if v is None:
        return "#9ca3af"
    return "#22c55e" if v >= 0 else "#ef4444"


# ── Calendário econômico ──────────────────────────────────────────────────────

_IMPACT_MAP = {
    "employment situation": ("🔴", "Alto impacto"),
    "nonfarm payroll": ("🔴", "Alto impacto"),
    "consumer price": ("🔴", "Alto impacto"),
    "fomc": ("🔴", "Alto impacto"),
    "gdp": ("🔴", "Alto impacto"),
    "pce price": ("🔴", "Alto impacto"),
    "adp national": ("🟠", "Médio impacto"),
    "ism manufactur": ("🟠", "Médio impacto"),
    "ism non-manufactur": ("🟠", "Médio impacto"),
    "jolts": ("🟠", "Médio impacto"),
    "retail sales": ("🟠", "Médio impacto"),
    "ppi": ("🟠", "Médio impacto"),
    "michigan consumer": ("🟡", "Impacto moderado"),
    "housing starts": ("🟡", "Impacto moderado"),
    "chicago pmi": ("🟡", "Impacto moderado"),
    "existing home": ("🟡", "Impacto moderado"),
    "treasury": ("⚪", "Técnico"),
}

def _get_impact(name: str) -> tuple[str, str]:
    low = name.lower()
    for kw, val in _IMPACT_MAP.items():
        if kw in low:
            return val
    return ("⚪", "Monitorar")


def _build_calendar_html(calendar: list[dict]) -> str:
    if not calendar:
        return "<p class='dim'>Sem eventos no período.</p>"

    rows = []
    prev_date = None
    for item in sorted(calendar, key=lambda x: x.get("date", "")):
        d = item.get("date", "")
        name = item.get("release_name", "")
        emoji, label = _get_impact(name)

        # formata data
        try:
            dt = date.fromisoformat(d)
            day_str = dt.strftime("%a %d/%m")
            is_today = dt == date.today()
            is_tomorrow = (dt - date.today()).days == 1
            if is_today:
                day_label = f'<span class="badge-today">HOJE</span> {day_str}'
            elif is_tomorrow:
                day_label = f'<span class="badge-tomorrow">AMANHÃ</span> {day_str}'
            else:
                day_label = day_str
        except Exception:
            day_label = d

        sep = "<tr class='date-sep'></tr>" if d != prev_date and prev_date else ""
        prev_date = d

        rows.append(f"""
        {sep}
        <tr>
          <td class="cal-date">{day_label}</td>
          <td class="cal-event">{name}</td>
          <td class="cal-impact" title="{label}">{emoji} <span class="impact-label">{label}</span></td>
        </tr>""")

    return f"""
    <table class="cal-table">
      <thead>
        <tr><th>Data</th><th>Evento</th><th>Impacto</th></tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>"""


# ── Dashboard FRED ────────────────────────────────────────────────────────────

# Séries prioritárias para o week_ahead com label curto e formatação especial
_PRIORITY_SERIES = {
    "Política Monetária": [
        ("DFF",       "Fed Funds",         None),
        ("T10Y2Y",    "Curva 10y-2y",      None),
        ("T10YIE",    "Breakeven 10y",     None),
        ("WALCL",     "Balanço Fed",       None),
    ],
    "Inflação": [
        ("CPIAUCSL",  "CPI Total",         None),
        ("CPILFESL",  "CPI Core",          None),
        ("PCEPI",     "PCE Total",         None),
        ("PCEPILFE",  "PCE Core",          None),
        ("MICH",      "Michigan Inf. Exp", None),
    ],
    "Mercado de Trabalho": [
        ("UNRATE",    "Desemprego",        None),
        ("PAYEMS",    "Payroll",           None),
        ("JTSJOL",    "JOLTS Vagas",       None),
        ("CIVPART",   "Participação",      None),
    ],
    "Crédito e Condições Financeiras": [
        ("BAMLH0A0HYM2", "HY Spread",     None),
        ("BAMLC0A0CM",   "IG Spread",     None),
        ("NFCI",         "NFCI",          None),
        ("MORTGAGE30US", "Hipoteca 30a",  None),
    ],
    "Crescimento": [
        ("GDPCA",     "PIB (YoY%)",        None),
        ("INDPRO",    "Prod. Industrial",  None),
        ("PCE",       "Consumo Pessoal",   None),
    ],
    "Dólar e Externo": [
        ("DTWEXBGS",  "DXY Amplo",         None),
        ("BOPGSTB",   "Balança Comercial", None),
    ],
}

# Ícones por categoria
_CAT_ICONS = {
    "Política Monetária": "🏛️",
    "Inflação": "📈",
    "Mercado de Trabalho": "👷",
    "Crédito e Condições Financeiras": "💳",
    "Crescimento": "📊",
    "Dólar e Externo": "🌐",
}


def _build_swaggy_section(swaggy_result) -> str:
    """Renderiza WSB Top Mentions + Squeeze Candidates + Sentiment Gauge."""
    if swaggy_result is None:
        return ""
    wsb = getattr(swaggy_result, "wsb_mentions", [])
    squeeze = getattr(swaggy_result, "squeeze_candidates", [])
    market_bull_pct = getattr(swaggy_result, "market_bull_pct", None)
    if not wsb and not squeeze and market_bull_pct is None:
        return ""

    # Gauge de sentimento geral
    gauge_html = ""
    if market_bull_pct is not None:
        pct = market_bull_pct if market_bull_pct <= 1 else market_bull_pct / 100
        pct_display = f"{pct*100:.0f}%"
        bar_color = "#22c55e" if pct > 0.55 else ("#ef4444" if pct < 0.45 else "#f59e0b")
        label = "Bullish" if pct > 0.55 else ("Bearish" if pct < 0.45 else "Neutral")
        gauge_html = f"""
        <div style="margin-bottom:16px;padding:12px 16px;background:rgba(255,255,255,0.03);border-radius:8px;border:1px solid rgba(255,255,255,0.08)">
          <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">Sentimento Geral do Mercado</div>
          <div style="display:flex;align-items:center;gap:12px">
            <div style="font-size:24px;font-weight:700;color:{bar_color}">{pct_display}</div>
            <div style="flex:1">
              <div style="height:8px;background:rgba(255,255,255,0.08);border-radius:4px;overflow:hidden">
                <div style="height:100%;width:{pct*100:.0f}%;background:{bar_color};border-radius:4px;transition:width .3s"></div>
              </div>
              <div style="font-size:11px;color:{bar_color};margin-top:4px;font-weight:600">{label}</div>
            </div>
          </div>
        </div>"""

    wsb_rows = ""
    for m in wsb[:15]:
        sent_color = "#22c55e" if m.sentiment > 0.55 else "#ef4444" if m.sentiment < 0.45 else "#f59e0b"
        sent_label = getattr(m, "sentiment_label", "neutral")
        wsb_rows += (
            f"<tr><td>#{m.rank}</td><td><b>{m.ticker}</b></td>"
            f"<td>{m.mentions:,}</td>"
            f"<td style='color:{sent_color}'>{sent_label}</td>"
            f"<td>{m.attention_score:.2f}</td></tr>"
        )

    squeeze_rows = ""
    for s in squeeze[:10]:
        score_color = "#ef4444" if s.squeeze_score > 0.6 else "#f59e0b" if s.squeeze_score > 0.3 else "#6b7280"
        squeeze_rows += (
            f"<tr><td><b>{s.ticker}</b></td>"
            f"<td>{s.short_interest_pct:.1f}%</td>"
            f"<td>{s.days_to_cover:.1f}d</td>"
            f"<td>{s.borrow_rate_pct:.1f}%</td>"
            f"<td style='color:{score_color}'>{s.squeeze_score:.2f}</td></tr>"
        )

    wsb_table = ""
    if wsb_rows:
        wsb_table = f"""
        <div style="flex:1;min-width:280px">
          <div style="font-size:11px;font-weight:700;color:#38bdf8;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">🧵 WSB Top Mentions</div>
          <table style="width:100%;border-collapse:collapse;font-size:12px">
            <thead><tr style="color:#6b7280;border-bottom:1px solid rgba(255,255,255,0.08)">
              <th style="text-align:left;padding:3px 6px">#</th>
              <th style="text-align:left;padding:3px 6px">Ticker</th>
              <th style="text-align:right;padding:3px 6px">Mentions</th>
              <th style="text-align:left;padding:3px 6px">Sentiment</th>
              <th style="text-align:right;padding:3px 6px">Score</th>
            </tr></thead>
            <tbody>{wsb_rows}</tbody>
          </table>
        </div>"""

    squeeze_table = ""
    if squeeze_rows:
        squeeze_table = f"""
        <div style="flex:1;min-width:280px">
          <div style="font-size:11px;font-weight:700;color:#c084fc;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">🔥 Squeeze Candidates</div>
          <table style="width:100%;border-collapse:collapse;font-size:12px">
            <thead><tr style="color:#6b7280;border-bottom:1px solid rgba(255,255,255,0.08)">
              <th style="text-align:left;padding:3px 6px">Ticker</th>
              <th style="text-align:right;padding:3px 6px">SI%</th>
              <th style="text-align:right;padding:3px 6px">DTC</th>
              <th style="text-align:right;padding:3px 6px">Borrow</th>
              <th style="text-align:right;padding:3px 6px">Score</th>
            </tr></thead>
            <tbody>{squeeze_rows}</tbody>
          </table>
        </div>"""

    if not wsb_table and not squeeze_table:
        return ""

    return f"""
<div class="card" style="margin-bottom:20px">
  <div class="card-header">📱 WSB &amp; Squeeze Monitor</div>
  <div class="card-body">
    {gauge_html}
    <div style="display:flex;gap:24px;flex-wrap:wrap">
      {wsb_table}
      {squeeze_table}
    </div>
  </div>
</div>"""


def _build_tv_zones_section(signals: dict) -> str:
    """Renderiza TradingView Zone Signals para os ativos com setup ideal/acceptable."""
    if not signals:
        return ""

    rows_ideal = []
    rows_accept = []
    rows_avoid = []

    for ticker, sig in signals.items():
        zs = getattr(sig, "_zone_signal", None)
        if zs is None:
            continue
        q = zs.entry_quality
        price_str = f"${zs.price:,.2f}" if zs.price else "—"
        vwap_str  = f"${zs.vwap:,.2f}" if zs.vwap else "—"
        vah_str   = f"${zs.vah:,.2f}" if zs.vah else "—"
        val_str   = f"${zs.val:,.2f}" if zs.val else "—"
        rsi_str   = f"{zs.rsi:.0f}" if zs.rsi else "—"
        regime_color = "#22c55e" if zs.tv_regime == "bullish" else "#ef4444" if zs.tv_regime == "bearish" else "#6b7280"
        row = (
            f"<tr><td><b>{ticker}</b></td>"
            f"<td>{price_str}</td>"
            f"<td>{vwap_str}</td>"
            f"<td>{vah_str}</td>"
            f"<td>{val_str}</td>"
            f"<td>{rsi_str}</td>"
            f"<td style='color:{regime_color}'>{zs.tv_regime}</td>"
            f"<td style='font-size:11px;color:#9ca3af'>{zs.rationale[:60]}</td></tr>"
        )
        if q == "ideal":
            rows_ideal.append(row)
        elif q in ("acceptable",):
            rows_accept.append(row)
        elif q == "avoid":
            rows_avoid.append(row)

    if not rows_ideal and not rows_accept and not rows_avoid:
        return ""

    def _table(rows: list[str], label: str, color: str) -> str:
        if not rows:
            return ""
        return f"""
        <div style="margin-bottom:16px">
          <div style="font-size:11px;font-weight:700;color:{color};text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">{label} ({len(rows)})</div>
          <table style="width:100%;border-collapse:collapse;font-size:12px">
            <thead><tr style="color:#6b7280;border-bottom:1px solid rgba(255,255,255,0.08)">
              <th style="text-align:left;padding:3px 6px">Ticker</th>
              <th style="text-align:right;padding:3px 6px">Preço</th>
              <th style="text-align:right;padding:3px 6px">VWAP</th>
              <th style="text-align:right;padding:3px 6px">VAH</th>
              <th style="text-align:right;padding:3px 6px">VAL</th>
              <th style="text-align:right;padding:3px 6px">RSI</th>
              <th style="text-align:left;padding:3px 6px">Regime</th>
              <th style="text-align:left;padding:3px 6px">Rationale</th>
            </tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>"""

    body = (
        _table(rows_ideal,  "✅ Ideal Entry",      "#22c55e")
        + _table(rows_accept, "🟡 Acceptable",       "#f59e0b")
        + _table(rows_avoid,  "🚫 Avoid",            "#ef4444")
    )

    return f"""
<div class="card" style="margin-bottom:20px">
  <div class="card-header">📊 TradingView — Value Area · VWAP · Setup Quality</div>
  <div class="card-body">{body}</div>
</div>"""


def _build_fred_section(series_data: dict) -> str:
    """Constrói o dashboard FRED com cards por categoria."""
    blocks = []

    for cat, priority in _PRIORITY_SERIES.items():
        cat_data = series_data.get(cat, [])
        # indexa por series_id
        by_id = {s["series_id"]: s for s in cat_data}

        rows = []
        for sid, short_label, _ in priority:
            s = by_id.get(sid)
            if not s:
                continue

            val = s.get("value")
            chg = s.get("change")
            unit = s.get("unit", "")
            dt = s.get("date", "")
            hist = [h["value"] for h in s.get("history", []) if h.get("value") is not None]

            spark = _sparkline(hist) if len(hist) >= 3 else ""
            val_str = _fmt_val(val, unit)
            chg_html = _fmt_change(chg, unit)

            rows.append(f"""
            <tr>
              <td class="metric-name">{short_label}</td>
              <td class="metric-val">{val_str} <span class="metric-unit">{unit}</span></td>
              <td class="metric-chg">{chg_html}</td>
              <td class="metric-spark">{spark}</td>
              <td class="metric-date dim">{dt}</td>
            </tr>""")

        if not rows:
            continue

        icon = _CAT_ICONS.get(cat, "📌")
        blocks.append(f"""
        <div class="fred-card">
          <div class="fred-card-title">{icon} {cat}</div>
          <table class="fred-table">
            <thead>
              <tr>
                <th>Série</th><th>Valor</th><th>Δ</th><th>Tendência</th><th>Data</th>
              </tr>
            </thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>""")

    return "\n".join(blocks)


# ── Tabela de preços de mercado ───────────────────────────────────────────────

_MARKET_ORDER = [
    ("^GSPC",    "S&P 500"),
    ("^NDX",     "Nasdaq 100"),
    ("^RUT",     "Russell 2000"),
    ("^VIX",     "VIX"),
    ("TLT",      "TLT 20yr"),
    ("HYG",      "HY Credit"),
    ("GLD",      "Gold"),
    ("CL=F",     "WTI Crude"),
    ("BTC-USD",  "Bitcoin"),
    ("DX-Y.NYB", "DXY"),
]

def _build_charts_html(artifact_paths: dict, mode: str = "week_ahead") -> str:
    """
    Embeds key charts using iframe srcdoc (self-contained).
    Selects charts based on mode:
      week_ahead/week_recap → kline GSPC, NDX, CLF, GLD, VIX + risk_radar
      flow_show            → kline GSPC, VIX + mc_fan GSPC, NDX
      growth               → kline NDX + mc_fan NDX, RUT
    """
    # Quais charts incluir por modo
    _MODE_CHARTS: dict[str, list[str]] = {
        "week_ahead":  ["chart_kline_GSPC", "chart_kline_NDX", "chart_kline_CLF", "chart_kline_GLD", "chart_kline_VIX"],
        "week_recap":  ["chart_kline_GSPC", "chart_kline_NDX", "chart_kline_CLF", "chart_kline_GLD"],
        "flow_show":   ["chart_kline_GSPC", "chart_kline_VIX", "chart_mc_fan_GSPC", "chart_mc_fan_NDX"],
        "growth":      ["chart_kline_NDX", "chart_mc_fan_NDX", "chart_mc_fan_RUT"],
        "tese":        ["chart_kline_GSPC", "chart_risk_radar"],
    }
    _LABELS: dict[str, str] = {
        "chart_kline_GSPC":    "S&P 500",
        "chart_kline_NDX":     "Nasdaq 100",
        "chart_kline_CLF":     "WTI Crude",
        "chart_kline_GLD":     "Gold",
        "chart_kline_VIX":     "VIX",
        "chart_risk_radar":    "Risk Radar",
        "chart_mc_fan_GSPC":   "Monte Carlo S&P 500",
        "chart_mc_fan_NDX":    "Monte Carlo Nasdaq",
        "chart_mc_fan_RUT":    "Monte Carlo Russell 2K",
    }

    keys = _MODE_CHARTS.get(mode, _MODE_CHARTS["week_ahead"])
    frames = []

    for key in keys:
        path_str = artifact_paths.get(key)
        if not path_str:
            continue
        p = Path(path_str)
        if not p.exists():
            continue
        label = _LABELS.get(key, key)
        try:
            chart_html = p.read_text(encoding="utf-8")
            # Extrai apenas o <body> do chart HTML para embutir inline
            # (evita iframe aninhado que não renderiza dentro de srcdoc)
            import re as _re
            body_m = _re.search(r'<body[^>]*>(.*?)</body>', chart_html, _re.DOTALL | _re.IGNORECASE)
            body_content = body_m.group(1).strip() if body_m else chart_html
            # Extrai <style> do <head> se existir
            style_m = _re.search(r'<style[^>]*>(.*?)</style>', chart_html, _re.DOTALL | _re.IGNORECASE)
            style_tag = f'<style>{style_m.group(1)}</style>' if style_m else ''
            # Extrai <script> tags do head/body
            scripts = _re.findall(r'<script[^>]*>.*?</script>', chart_html, _re.DOTALL | _re.IGNORECASE)
            scripts_html = '\n'.join(scripts)
            frames.append(f"""
            <div class="chart-frame">
              <div class="chart-label">{label}</div>
              <div style="width:100%;height:320px;overflow:hidden;border-radius:6px;background:#0d1117;position:relative">
                {style_tag}
                {body_content}
                {scripts_html}
              </div>
            </div>""")
        except Exception:
            continue

    if not frames:
        return ""

    # Grade: 2 colunas para kline, 1 para risk_radar
    return f"""
    <div class="section-title">📈 Gráficos de Mercado</div>
    <div class="charts-grid">{''.join(frames)}</div>"""


def _build_scenarios_html(scenarios: dict) -> str:
    """Cards Bull / Base / Bear com probabilidades."""
    if not scenarios or "bull" not in scenarios:
        return ""

    def card(kind: str, label: str) -> str:
        s = scenarios.get(kind, {})
        prob = s.get("probability", 0)
        catalyst = s.get("catalyst", "")
        narrative = s.get("narrative", "")
        spx = s.get("spx_target")
        horizon = s.get("time_horizon", "")
        spx_str = f"SPX target: {spx:,}" if spx else ""
        return f"""
        <div class="scenario-card {kind}">
          <div class="scenario-label">{label}</div>
          <div class="scenario-prob">{prob:.0%}</div>
          <div class="scenario-catalyst">{catalyst}</div>
          <div class="scenario-narrative">{narrative}</div>
          <div class="scenario-meta">
            <span>{spx_str}</span><span>{horizon}</span>
          </div>
        </div>"""

    narrative = scenarios.get("narrative", "")
    note = f'<div class="dim" style="font-size:12px;margin-bottom:12px;padding:10px 14px;background:var(--surface2);border-radius:8px;border-left:3px solid var(--cyan)">Cenários gerados por LLM para: <em>{narrative}</em>. Probabilidades são estimativas qualitativas.</div>' if narrative else ""

    return f"""
    <div class="section-title">🎯 Cenários Bull / Base / Bear</div>
    {note}
    <div class="scenarios-grid">
      {card("bull", "BULL")}
      {card("base", "BASE")}
      {card("bear", "BEAR")}
    </div>"""


def _build_polymarket_html(markets: list) -> str:
    """Tabela Polymarket."""
    if not markets:
        return ""

    rows = ""
    for m in markets:
        prob = m.get("probability", 0)
        prob_pct = f"{prob:.0%}"
        color = "#22c55e" if prob >= 0.6 else ("#ef4444" if prob <= 0.3 else "#f59e0b")
        vol = m.get("volume_usd", 0)
        vol_str = f"${vol/1e6:.1f}M" if vol else "—"
        expiry = m.get("end_date", m.get("expiry", ""))[:10] if m.get("end_date") or m.get("expiry") else "—"
        q = m.get("question", m.get("title", ""))
        rows += f"""
        <tr>
          <td>{q}</td>
          <td class="poly-prob" style="color:{color}">{prob_pct}</td>
          <td class="poly-vol">{vol_str}</td>
          <td class="poly-vol">{expiry}</td>
        </tr>"""

    return f"""
    <div class="card">
      <div class="card-header">🎲 Polymarket — Mercados de Predição</div>
      <div class="card-body" style="padding:0">
        <table class="poly-table">
          <thead><tr><th>Evento</th><th>Prob.</th><th>Volume</th><th>Vencimento</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </div>"""


def _build_risk_html(risk_data: dict) -> str:
    """Tabela Risk Radar: Mag 7 + índices principais."""
    tickers_data = risk_data.get("tickers", {})
    if not tickers_data:
        return ""

    # Magnificent 7 + índices macro principais — nada mais
    _INDICES = {
        # Mag 7
        "AAPL":      "Apple",
        "MSFT":      "Microsoft",
        "AMZN":      "Amazon",
        "GOOGL":     "Alphabet",
        "META":      "Meta",
        "NVDA":      "Nvidia",
        # Índices
        "^GSPC":     "S&P 500",
        "^NDX":      "Nasdaq 100",
        "^DJI":      "Dow Jones",
        "^RUT":      "Russell 2000",
        "^VIX":      "VIX",
        "DX-Y.NYB":  "DXY",
        "^TNX":      "TNX 10Y",
    }
    _LABELS = _INDICES

    # Filtro: só índices; se nenhum encontrado, silencioso
    filtered = {t: v for t, v in tickers_data.items() if t in _INDICES}
    if not filtered:
        return ""

    def risk_color(v, invert=False):
        if v is None: return "#9ca3af"
        if invert:
            return "#22c55e" if v > 0 else "#ef4444"
        return "#ef4444" if v < -0.05 else ("#f59e0b" if v < -0.02 else "#22c55e")

    rows = ""
    for ticker, r in filtered.items():
        label = _LABELS.get(ticker, ticker)
        var = r.get("var_95")
        cvar = r.get("cvar_95")
        dd = r.get("max_drawdown")
        sharpe = r.get("sharpe")

        def fmt_pct(v):
            if v is None: return "—"
            return f"{v*100:.1f}%"
        def fmt_num(v):
            if v is None: return "—"
            return f"{v:.2f}"

        rows += f"""
        <tr>
          <td>{label}</td>
          <td style="color:{risk_color(var)}">{fmt_pct(var)}</td>
          <td style="color:{risk_color(cvar)}">{fmt_pct(cvar)}</td>
          <td style="color:{risk_color(dd)}">{fmt_pct(dd)}</td>
          <td style="color:{risk_color(sharpe, invert=True)}">{fmt_num(sharpe)}</td>
        </tr>"""

    return f"""
    <div class="card">
      <div class="card-header">🛡️ Risk Radar (60d)</div>
      <div class="card-body" style="padding:0">
        <table class="risk-table">
          <thead><tr><th>Ativo</th><th>VaR 95%</th><th>CVaR 95%</th><th>Max DD</th><th>Sharpe</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </div>"""


def _build_monte_carlo_html(mc_data: dict) -> str:
    """Tabela Monte Carlo GBM — Mag 7 + índices principais."""
    if not mc_data:
        return ""

    # Magnificent 7 + índices macro — mesma seleção do Risk Radar
    _INDICES = {
        # Mag 7
        "AAPL":      "Apple",
        "MSFT":      "Microsoft",
        "AMZN":      "Amazon",
        "GOOGL":     "Alphabet",
        "META":      "Meta",
        "NVDA":      "Nvidia",
        # Índices
        "^GSPC":     "S&P 500",
        "^NDX":      "Nasdaq 100",
        "^DJI":      "Dow Jones",
        "^RUT":      "Russell 2000",
        "^VIX":      "VIX",
        "DX-Y.NYB":  "DXY",
        "^TNX":      "TNX 10Y",
    }
    _LABELS = _INDICES

    # Filtra apenas índices
    mc_filtered = {t: v for t, v in mc_data.items() if t in _INDICES}
    if not mc_filtered:
        return ""

    # Info do primeiro ticker
    sample = next(iter(mc_filtered.values()), {})
    horizon = sample.get("horizon_days", 20)
    paths = sample.get("paths_count", 500)
    subtitle = f'<div class="dim" style="font-size:11.5px;padding:8px 14px;background:var(--surface2);border-bottom:1px solid var(--border)">GBM · {paths} caminhos · {horizon} dias úteis · P50 = retorno mediano esperado</div>'

    rows = ""
    for ticker, d in mc_filtered.items():
        label = _LABELS.get(ticker, ticker)
        price = d.get("current_price", 0)
        prob_up = d.get("prob_up", 0)
        prob_up5 = d.get("prob_up_5pct", 0)
        prob_dn5 = d.get("prob_down_5pct", 0)
        p50_raw = d.get("percentiles", {}).get("p50") or d.get("percentiles", {}).get("50")
        p50 = p50_raw[-1] if isinstance(p50_raw, list) and p50_raw else (p50_raw if isinstance(p50_raw, (int, float)) else None)
        p50_ret = ((p50 / price) - 1) if (p50 and price) else None

        def pct_cell(v, threshold_green=0.5):
            if v is None: return "—"
            color = "#22c55e" if v >= threshold_green else ("#ef4444" if v < 0.3 else "#f59e0b")
            return f'<span style="color:{color};font-weight:600">{v:.0%}</span>'

        p50_str = f'<span style="color:{"#22c55e" if (p50_ret or 0)>=0 else "#ef4444"};font-weight:600">{(p50_ret or 0)*100:+.2f}%</span>' if p50_ret is not None else "—"

        rows += f"""
        <tr>
          <td><strong>{ticker}</strong> <span style="color:#8b949e;font-weight:400">{label}</span></td>
          <td style="text-align:right">{price:,.2f}</td>
          <td style="text-align:right">{pct_cell(prob_up, 0.5)}</td>
          <td style="text-align:right">{pct_cell(prob_up5, 0.3)}</td>
          <td style="text-align:right">{pct_cell(prob_dn5, 0.4)}</td>
          <td style="text-align:right">{p50_str}</td>
        </tr>"""

    return f"""
    <div class="section-title">🎲 Monte Carlo — GBM ({horizon} dias, {paths} caminhos)</div>
    <div class="card" style="margin-bottom:20px">
      {subtitle}
      <div class="card-body" style="padding:0">
        <table class="mc-table">
          <thead>
            <tr>
              <th style="text-align:left">Ativo</th>
              <th>Preço Atual</th>
              <th>Prob Up</th>
              <th>Prob +5%</th>
              <th>Prob −5%</th>
              <th>P50 ({horizon}d)</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </div>"""


def _img_src(img_path: str, out_dir: Path) -> str:
    """Retorna data URI base64 para imagens locais (HTML auto-contido)."""
    import base64, mimetypes
    p = Path(img_path)
    if p.exists():
        try:
            mime = mimetypes.guess_type(str(p))[0] or "image/png"
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f"data:{mime};base64,{b64}"
        except Exception:
            pass
    return ""  # arquivo não encontrado — retorna vazio para não gerar buraco no HTML


def _build_media_gallery(bundle: DailyIngestionBundle, out_dir: Path) -> str:
    """
    Market Gossip — ZeroHedge Market Ear + X Timeline com imagens.
    Sem nome de autor. Foco em imagem + comentário editorial curto.
    """
    _gallery_seen: set[str] = set()
    all_cards: list[str] = []

    # Padrões de banner/logo que não são gráficos reais
    _BANNER_PATTERNS = (
        "themarketary", "marketary", "market_ear", "marketear",
        "logo", "banner", "header", "favicon", "placeholder",
    )
    _MIN_CHART_BYTES = 25_000  # < 25KB → provável logo/banner
    _MAX_BANNER_RATIO = 3.5    # width/height > 3.5 → banner/logo horizontal

    def _is_chart(img_path: str) -> bool:
        """True se parece um gráfico real (não logo/banner)."""
        p = Path(img_path)
        name_lower = p.name.lower()
        if any(pat in name_lower for pat in _BANNER_PATTERNS):
            return False
        try:
            if p.stat().st_size < _MIN_CHART_BYTES:
                return False
        except OSError:
            pass
        # Verifica dimensões: banners têm aspect ratio muito largo
        try:
            from PIL import Image as _PILImage
            with _PILImage.open(p) as im:
                w, h = im.size
                if h > 0 and w / h > _MAX_BANNER_RATIO:
                    return False
                # Imagens muito pequenas em altura também são logos
                if h < 80:
                    return False
        except Exception:
            pass
        return True

    # ── ZeroHedge blocks com imagem ──────────────────────────────────────────
    for block in bundle.market_ear_blocks:
        unique_refs = list(dict.fromkeys(block.image_refs))
        local_imgs = [
            p for p in unique_refs
            if Path(p).exists() and not p.startswith("http") and _is_chart(p)
        ]
        if not local_imgs:
            continue  # sem gráfico local → sem card

        imgs_html = ""
        _MAX_IMGS_PER_BLOCK = 2  # evita cards gigantes que criam buracos no grid
        _block_count = 0
        for img_path in local_imgs:
            if _block_count >= _MAX_IMGS_PER_BLOCK:
                break
            key = Path(img_path).name if Path(img_path).exists() else img_path
            if key in _gallery_seen:
                continue
            _gallery_seen.add(key)
            src = _img_src(img_path, out_dir)
            if src:
                imgs_html += f'<img src="{src}" class="mg-img" loading="lazy" alt="">'
                _block_count += 1

        if not imgs_html:
            continue

        title_html = f'<div class="mg-block-title">{block.title}</div>' if block.title else ""
        body_snippet = (block.body_text[:280].strip() + "…") if len(block.body_text) > 280 else block.body_text.strip()
        body_html = f'<p class="mg-block-body">{body_snippet}</p>' if body_snippet else ""

        all_cards.append(f"""
        <div class="mg-block">
          {title_html}
          {imgs_html}
          {body_html}
        </div>""")

    if not all_cards:
        return ""

    return f"""
    <div class="mg-wrap">
      <div class="section-title">📰 Market Gossip</div>
      <div class="mg-zh-grid">{''.join(all_cards)}</div>
    </div>"""


def _build_market_heatmap(market_prices: dict) -> str:
    """Heatmap visual de retornos diários — tiles coloridos estilo Bloomberg."""
    _TILES = [
        ("^GSPC", "S&P 500"), ("^NDX", "Nasdaq"), ("^RUT", "Russell 2K"),
        ("TLT", "TLT 20yr"), ("HYG", "HY Credit"), ("GLD", "Gold"),
        ("CL=F", "WTI Crude"), ("BTC-USD", "Bitcoin"), ("DX-Y.NYB", "DXY"), ("^VIX", "VIX"),
    ]

    def _heat_color(v: float | None) -> tuple[str, str]:
        """Retorna (background_color, text_color)."""
        if v is None:
            return "#374151", "#9ca3af"
        if v >= 0.04:   return "#064e3b", "#34d399"
        if v >= 0.02:   return "#065f46", "#6ee7b7"
        if v >= 0.005:  return "#14532d", "#86efac"
        if v >= -0.005: return "#1f2937", "#9ca3af"
        if v >= -0.02:  return "#7f1d1d", "#fca5a5"
        if v >= -0.04:  return "#991b1b", "#f87171"
        return "#450a0a", "#ef4444"

    tiles = ""
    for ticker, label in _TILES:
        mp = market_prices.get(ticker, {})
        if not mp:
            continue
        daily = mp.get("daily_return")
        price = mp.get("price")
        bg, fg = _heat_color(daily)
        ret_str = _pct_str(daily)
        price_str = f"{price:,.2f}" if price and price < 10000 else (f"{price:,.0f}" if price else "—")
        tiles += f"""
        <div class="heatmap-tile" style="background:{bg};color:{fg}">
          <span class="ht-ticker">{ticker}</span>
          <span class="ht-name">{label}</span>
          <span class="ht-ret">{ret_str}</span>
          <span class="ht-price">{price_str}</span>
        </div>"""

    return f"""
    <div class="heatmap-wrap">
      <div class="heatmap-header">🌡️ Mercado — Retorno Diário (Heatmap)</div>
      <div class="heatmap-grid">{tiles}</div>
    </div>"""


def _build_market_table(market_prices: dict) -> str:
    rows = []
    for ticker, label in _MARKET_ORDER:
        mp = market_prices.get(ticker, {})
        if not mp:
            continue
        price = mp.get("price")
        daily = mp.get("daily_return")
        weekly = mp.get("weekly_return")
        ytd = mp.get("ytd_return")

        def fmt_price(p):
            if p is None: return "—"
            if p > 10000: return f"{p:,.0f}"
            if p > 100: return f"{p:,.2f}"
            return f"{p:.2f}"

        def pct_cell(v):
            if v is None: return "<td>—</td>"
            color = _color_pct(v)
            return f'<td style="color:{color};font-weight:500">{_pct_str(v)}</td>'

        rows.append(f"""
        <tr>
          <td class="mk-ticker">{ticker}</td>
          <td class="mk-name">{label}</td>
          <td class="mk-price">{fmt_price(price)}</td>
          {pct_cell(daily)}
          {pct_cell(weekly)}
          {pct_cell(ytd)}
        </tr>""")

    return f"""
    <table class="market-table">
      <thead>
        <tr>
          <th>Ticker</th><th>Ativo</th><th>Preço</th>
          <th>1D</th><th>1W</th><th>YTD</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


# ── Texto editorial ───────────────────────────────────────────────────────────

def _extract_written_sections(curation_path: str | None) -> dict[str, list[dict]]:
    """
    Extrai seções do texto escrito.
    Prefere o raw .txt (com sentinels <<<IMG:path|legenda>>>) se disponível,
    senão cai no .docx (sem as imagens contextuais).
    Retorna dict[seção] → lista de {type: 'heading'|'body'|'callout'|'image', text/path/caption: str}
    """
    if not curation_path:
        return {}

    try:
        import json as _j
        c = _j.loads(Path(curation_path).read_text(encoding="utf-8"))
        written = c.get("artifact_paths", {}).get("written", "")
        if not written:
            return {}
        # Procura o raw .txt ao lado do .docx — preserva sentinels de imagem
        raw_txt = Path(written).with_name(Path(written).stem + "_raw.txt")
        if not raw_txt.exists():
            # Também tenta padrão sem o sufixo duplicado
            alt = Path(written).parent / (Path(written).stem.replace("_written_", "_written_") + "_raw.txt")
            raw_txt = alt if alt.exists() else raw_txt
        if raw_txt.exists():
            return _parse_written_raw_text(raw_txt.read_text(encoding="utf-8"))
        # Fallback: parseia o .docx (imagens serão mecanicamente intercaladas)
        if not Path(written).exists():
            return {}
        from docx import Document
        doc = Document(written)
        paragraphs = doc.paragraphs
    except Exception as exc:
        _log.warning("written_extract_error", error=str(exc))
        return {}

    sections: dict[str, list[dict]] = {}
    current_key: str | None = None
    current_items: list[dict] = []

    _SEP = re.compile(r"^[━—\-=]{4,}")

    for p in paragraphs:
        raw = p.text
        if not raw.strip():
            continue

        style = p.style.name if p.style else "Normal"
        is_italic = any(run.italic for run in p.runs if run.text.strip())

        for line in raw.split("\n"):
            text = line.strip()
            if not text:
                continue

            m = re.match(r"=== (.+?) ===", text)
            if m:
                if current_key is not None:
                    sections[current_key] = current_items
                current_key = m.group(1).strip()
                current_items = []
                continue

            if current_key is None:
                continue

            if _SEP.match(text):
                continue

            if "Heading" in style:
                current_items.append({"type": "heading", "text": text})
                continue

            is_short = len(text) < 140
            has_colon = ":" in text[:60]
            if is_italic or (is_short and has_colon and not text.endswith(".")):
                current_items.append({"type": "callout", "text": text})
                continue

            current_items.append({"type": "body", "text": text})

    if current_key is not None:
        sections[current_key] = current_items

    return sections


def _parse_written_raw_text(raw: str) -> dict[str, list[dict]]:
    """
    Parseia o texto raw do writer, preservando ordem contextual das imagens.
    Reconhece:
      === SEÇÃO ===        → marcador de seção
      <<<IMG:path|legenda>>> → item imagem (na posição exata onde o LLM escolheu)
      ━━━...                → separador visual (ignorado)
      linhas normais       → parágrafo body / callout / heading
    """
    import re as _re
    sections: dict[str, list[dict]] = {}
    current_key: str | None = None
    current_items: list[dict] = []

    _SEP = _re.compile(r"^[━—\-=]{4,}")
    _IMG_RE = _re.compile(r"<<<IMG:(.+?)>>>")
    _SECTION_RE = _re.compile(r"===\s*(.+?)\s*===")

    for line in raw.splitlines():
        text = line.strip()
        if not text:
            continue

        m = _SECTION_RE.match(text)
        if m:
            if current_key is not None:
                sections[current_key] = current_items
            current_key = m.group(1).strip()
            current_items = []
            continue

        if current_key is None:
            continue

        if _SEP.match(text):
            continue

        # Linha pode ter texto + imagem (ou só imagem, ou só texto)
        # Split em pedaços alternando texto / imagem
        pos = 0
        for img_m in _IMG_RE.finditer(text):
            pre = text[pos:img_m.start()].strip()
            if pre:
                _append_text_item(current_items, pre)
            payload = img_m.group(1)
            if "|" in payload:
                path, caption = payload.split("|", 1)
                current_items.append({"type": "image", "path": path.strip(), "caption": caption.strip()})
            else:
                current_items.append({"type": "image", "path": payload.strip(), "caption": ""})
            pos = img_m.end()
        tail = text[pos:].strip()
        if tail:
            _append_text_item(current_items, tail)

    if current_key is not None:
        sections[current_key] = current_items

    return sections


def _append_text_item(items: list[dict], text: str) -> None:
    """Classifica a linha como heading/callout/body e adiciona à lista."""
    is_short = len(text) < 140
    has_colon = ":" in text[:60]
    # Callout: curto, termina com : ou é pergunta
    if is_short and has_colon and not text.endswith("."):
        items.append({"type": "callout", "text": text})
        return
    items.append({"type": "body", "text": text})


def _extract_written_text(curation_path: str | None) -> dict[str, str]:
    """Compat: retorna texto plano por seção (para publicacao.html)."""
    rich = _extract_written_sections(curation_path)
    result: dict[str, str] = {}
    for key, items in rich.items():
        result[key] = "\n\n".join(it["text"] for it in items)
    return result


def _rich_to_html(items: list[dict], inline_images: list[str] | None = None,
                  out_dir: Path | None = None) -> str:
    """
    Converte lista de itens ricos em HTML formatado.

    Se items contém entradas do tipo 'image' (vindas do raw text com sentinels
    <<<IMG:path|legenda>>>), renderiza elas em sua posição exata — contexto
    do LLM é preservado.

    Caso contrário, cai no fallback de intercalar `inline_images` mecanicamente
    a cada par de parágrafos.
    """
    if not items:
        return ""

    # Detecta se o LLM posicionou imagens contextualmente
    has_contextual_imgs = any(it.get("type") == "image" for it in items)

    parts = []
    img_idx = 0
    body_count = 0
    imgs = inline_images or []

    for it in items:
        t = it.get("type", "body")
        if t == "heading":
            parts.append(f'<h2 class="editorial-title">{it["text"]}</h2>')
        elif t == "callout":
            parts.append(f'<div class="editorial-callout">{it["text"]}</div>')
        elif t == "image":
            path = it.get("path", "")
            caption = it.get("caption", "")
            src = _img_src(path, out_dir) if out_dir else path
            if src:
                cap_html = (f'<div class="inline-img-caption">{caption}</div>'
                            if caption else '')
                parts.append(
                    f'<div class="inline-img-wrap">'
                    f'<img src="{src}" class="inline-img" loading="lazy">'
                    f'{cap_html}'
                    f'</div>'
                )
        else:
            parts.append(f'<p>{it["text"]}</p>')
            body_count += 1
            # Fallback: só intercala mecanicamente se o LLM NÃO posicionou imagens
            if not has_contextual_imgs:
                while img_idx < len(imgs) and not imgs[img_idx]:
                    img_idx += 1
                if body_count % 2 == 1 and img_idx < len(imgs):
                    img_src = imgs[img_idx]
                    img_idx += 1
                    if img_src:
                        parts.append(
                            f'<div class="inline-img-wrap">'
                            f'<img src="{img_src}" class="inline-img" loading="lazy">'
                            f'</div>'
                        )
    return "\n".join(parts)


def _paragraphs_to_html(text: str) -> str:
    """Converte texto corrido em parágrafos HTML (usado na publicação)."""
    if not text:
        return ""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return "".join(f"<p>{p}</p>" for p in paras)


# ── CSS e template HTML ───────────────────────────────────────────────────────

_CSS = """
:root {
  --bg: #06080f;
  --surface: #0d1117;
  --surface2: #161b22;
  --border: #21262d;
  --text: #e6edf3;
  --muted: #8b949e;
  --cyan: #38bdf8;
  --green: #22c55e;
  --red: #ef4444;
  --yellow: #f59e0b;
  --orange: #f97316;
  --purple: #a78bfa;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
  font-size: 13.5px;
  line-height: 1.6;
}

/* ── Layout ── */
.page { max-width: 1280px; margin: 0 auto; padding: 24px 20px 60px; }

/* ── Header ── */
.hero {
  background: linear-gradient(135deg, #0d1f35 0%, #111827 60%, #0d1117 100%);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 28px 32px;
  margin-bottom: 24px;
}
.hero-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }
.hero-badge {
  background: rgba(56,189,248,0.12);
  border: 1px solid rgba(56,189,248,0.3);
  color: var(--cyan);
  border-radius: 6px;
  padding: 4px 12px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  white-space: nowrap;
}
.hero-date { color: var(--muted); font-size: 12px; margin-top: 4px; }
.hero-narrative {
  font-size: 22px;
  font-weight: 700;
  color: var(--text);
  margin: 12px 0 6px;
  line-height: 1.3;
}
.hero-secondary { color: var(--muted); font-size: 13px; margin-top: 6px; }
.hero-conf {
  display: flex;
  gap: 16px;
  margin-top: 14px;
  flex-wrap: wrap;
}
.conf-chip {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 5px 12px;
  font-size: 12px;
  color: var(--muted);
}
.conf-chip strong { color: var(--text); }

/* ── Grid de seções ── */
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 20px; }
@media (max-width: 900px) {
  .grid-2, .grid-3 { grid-template-columns: 1fr; }
}

/* ── Cards ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
}
.card-header {
  background: var(--surface2);
  border-bottom: 1px solid var(--border);
  padding: 10px 16px;
  font-size: 12px;
  font-weight: 600;
  color: var(--cyan);
  letter-spacing: 0.05em;
  text-transform: uppercase;
  display: flex;
  align-items: center;
  gap: 8px;
}
.card-body { padding: 12px 16px; }

/* ── Calendário ── */
.cal-table { width: 100%; border-collapse: collapse; }
.cal-table thead th {
  font-size: 11px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 6px 8px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}
.cal-table tbody tr:hover { background: rgba(255,255,255,0.02); }
.cal-date {
  padding: 8px 8px;
  white-space: nowrap;
  font-size: 12px;
  color: var(--muted);
  min-width: 110px;
}
.cal-event { padding: 8px; font-size: 13px; }
.cal-impact { padding: 8px; font-size: 13px; white-space: nowrap; }
.impact-label { font-size: 11px; color: var(--muted); }
.badge-today {
  background: rgba(239,68,68,0.2);
  color: var(--red);
  border-radius: 4px;
  padding: 1px 6px;
  font-size: 10px;
  font-weight: 700;
  margin-right: 4px;
  letter-spacing: 0.05em;
}
.badge-tomorrow {
  background: rgba(245,158,11,0.2);
  color: var(--yellow);
  border-radius: 4px;
  padding: 1px 6px;
  font-size: 10px;
  font-weight: 700;
  margin-right: 4px;
}
.date-sep td { border-top: 1px solid var(--border); }

/* ── FRED dashboard ── */
.fred-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 20px;
}
@media (max-width: 900px) { .fred-grid { grid-template-columns: 1fr; } }

.fred-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
}
.fred-card-title {
  background: var(--surface2);
  border-bottom: 1px solid var(--border);
  padding: 9px 14px;
  font-size: 11.5px;
  font-weight: 600;
  color: var(--cyan);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.fred-table { width: 100%; border-collapse: collapse; }
.fred-table thead th {
  font-size: 10px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 6px 10px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}
.fred-table tbody tr { border-bottom: 1px solid rgba(33,38,45,0.5); }
.fred-table tbody tr:last-child { border-bottom: none; }
.fred-table tbody tr:hover { background: rgba(255,255,255,0.02); }
.metric-name { padding: 7px 10px; font-size: 12.5px; color: var(--text); white-space: nowrap; }
.metric-val { padding: 7px 8px; font-size: 13px; font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; }
.metric-unit { font-size: 10px; color: var(--muted); font-weight: 400; }
.metric-chg { padding: 7px 8px; font-size: 12px; white-space: nowrap; }
.metric-spark { padding: 4px 8px; }
.metric-date { padding: 7px 8px; font-size: 10.5px; white-space: nowrap; }

/* ── Mercado ── */
.market-table { width: 100%; border-collapse: collapse; }
.market-table thead th {
  font-size: 11px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 7px 10px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}
.market-table thead th:not(:first-child):not(:nth-child(2)) { text-align: right; }
.market-table tbody tr { border-bottom: 1px solid rgba(33,38,45,0.4); }
.market-table tbody tr:last-child { border-bottom: none; }
.market-table tbody tr:hover { background: rgba(255,255,255,0.02); }
.mk-ticker { padding: 8px 10px; font-size: 11.5px; color: var(--muted); font-weight: 600; }
.mk-name { padding: 8px 10px; font-size: 12.5px; }
.mk-price { padding: 8px 10px; text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; }
.market-table tbody td:not(.mk-ticker):not(.mk-name):not(.mk-price) {
  text-align: right; padding: 8px 10px; font-variant-numeric: tabular-nums; font-size: 12.5px;
}

/* ── Heatmap de mercado ── */
.heatmap-wrap {
  margin-bottom: 20px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
}
.heatmap-header {
  background: var(--surface2);
  border-bottom: 1px solid var(--border);
  padding: 9px 16px;
  font-size: 11.5px;
  font-weight: 600;
  color: var(--cyan);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.heatmap-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  padding: 12px;
}
.heatmap-tile {
  border-radius: 6px;
  padding: 10px 12px;
  min-width: 90px;
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 3px;
  position: relative;
  overflow: hidden;
}
.heatmap-tile::before {
  content: '';
  position: absolute;
  inset: 0;
  opacity: 0.12;
  background: currentColor;
}
.ht-ticker { font-size: 11px; font-weight: 700; color: rgba(255,255,255,0.6); position: relative; }
.ht-name { font-size: 10px; color: rgba(255,255,255,0.45); position: relative; }
.ht-ret { font-size: 17px; font-weight: 800; position: relative; letter-spacing: -0.02em; }
.ht-price { font-size: 10.5px; color: rgba(255,255,255,0.5); position: relative; }

/* ── Texto editorial ── */
.editorial {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
  margin-bottom: 20px;
}
.editorial-header {
  background: var(--surface2);
  border-bottom: 1px solid var(--border);
  padding: 10px 20px;
  font-size: 12px;
  font-weight: 600;
  color: var(--purple);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.editorial-body { padding: 24px 28px; }
.editorial-title {
  font-size: 20px;
  font-weight: 800;
  color: var(--text);
  margin: 0 0 18px;
  line-height: 1.25;
  letter-spacing: -0.02em;
}
.editorial-body p {
  margin-bottom: 16px;
  color: #cdd5e0;
  font-size: 14.5px;
  line-height: 1.8;
  max-width: 820px;
}
.editorial-body p:last-child { margin-bottom: 0; }
.editorial-callout {
  border-left: 3px solid var(--cyan);
  background: rgba(56,189,248,0.06);
  border-radius: 0 6px 6px 0;
  padding: 10px 16px;
  margin: 16px 0;
  font-size: 13px;
  color: #94c8e0;
  font-style: italic;
  line-height: 1.6;
  max-width: 820px;
}
.inline-img-wrap {
  margin: 20px 0;
  border-radius: 8px;
  overflow: hidden;
  max-width: 820px;
  border: 1px solid var(--border);
}
.inline-img-caption {
  padding: 8px 12px;
  font-size: 12px;
  color: var(--muted);
  font-style: italic;
  text-align: center;
  background: rgba(0,0,0,0.15);
  border-top: 1px solid var(--border);
}
.inline-img {
  width: 100%;
  max-height: 400px;
  object-fit: contain;
  display: block;
  background: #111;
}

/* ── Micro posts ── */
.microposts { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }
.micropost {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
  font-size: 13px;
  line-height: 1.5;
  color: var(--text);
}
.micropost-num { font-size: 11px; color: var(--cyan); font-weight: 600; margin-bottom: 6px; }

/* ── Charts grid ── */
.charts-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-bottom: 20px;
}
@media (max-width: 900px) { .charts-grid { grid-template-columns: 1fr; } }
.chart-frame {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
}
.chart-label {
  background: var(--surface2);
  border-bottom: 1px solid var(--border);
  padding: 7px 14px;
  font-size: 11.5px;
  font-weight: 600;
  color: var(--cyan);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

/* ── Cenários ── */
.scenarios-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; margin-bottom: 20px; }
@media (max-width: 900px) { .scenarios-grid { grid-template-columns: 1fr; } }
.scenario-card {
  border-radius: 10px;
  padding: 16px;
  border: 1px solid;
}
.scenario-card.bull { background: rgba(34,197,94,0.05); border-color: rgba(34,197,94,0.25); }
.scenario-card.base { background: rgba(245,158,11,0.05); border-color: rgba(245,158,11,0.25); }
.scenario-card.bear { background: rgba(239,68,68,0.05); border-color: rgba(239,68,68,0.25); }
.scenario-label { font-size: 10px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 6px; }
.scenario-card.bull .scenario-label { color: #22c55e; }
.scenario-card.base .scenario-label { color: #f59e0b; }
.scenario-card.bear .scenario-label { color: #ef4444; }
.scenario-prob { font-size: 28px; font-weight: 800; margin-bottom: 8px; }
.scenario-catalyst { font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 8px; line-height: 1.35; }
.scenario-narrative { font-size: 12px; color: var(--muted); line-height: 1.55; margin-bottom: 10px; }
.scenario-meta { font-size: 11px; color: var(--muted); }
.scenario-meta span { margin-right: 12px; }

/* ── Polymarket ── */
.poly-table { width: 100%; border-collapse: collapse; }
.poly-table thead th { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; padding: 7px 10px; text-align: left; border-bottom: 1px solid var(--border); }
.poly-table tbody tr { border-bottom: 1px solid rgba(33,38,45,0.5); }
.poly-table tbody tr:last-child { border-bottom: none; }
.poly-table tbody tr:hover { background: rgba(255,255,255,0.02); }
.poly-table td { padding: 9px 10px; font-size: 13px; }
.poly-prob { font-weight: 700; font-size: 14px; }
.poly-vol { color: var(--muted); font-size: 12px; }

/* ── Monte Carlo ── */
.mc-table { width: 100%; border-collapse: collapse; }
.mc-table thead th { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; padding: 7px 10px; text-align: right; border-bottom: 1px solid var(--border); }
.mc-table thead th:first-child { text-align: left; }
.mc-table tbody tr { border-bottom: 1px solid rgba(33,38,45,0.4); }
.mc-table tbody tr:last-child { border-bottom: none; }
.mc-table tbody tr:hover { background: rgba(255,255,255,0.02); }
.mc-table td { padding: 8px 10px; font-size: 12.5px; text-align: right; font-variant-numeric: tabular-nums; }
.mc-table td:first-child { text-align: left; font-weight: 600; color: var(--muted); font-size: 12px; }

/* ── Risk Radar (tabela) ── */
.risk-table { width: 100%; border-collapse: collapse; }
.risk-table thead th { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; padding: 7px 10px; text-align: right; border-bottom: 1px solid var(--border); }
.risk-table thead th:first-child { text-align: left; }
.risk-table tbody tr { border-bottom: 1px solid rgba(33,38,45,0.4); }
.risk-table tbody tr:last-child { border-bottom: none; }
.risk-table td { padding: 8px 10px; font-size: 12.5px; text-align: right; font-variant-numeric: tabular-nums; }
.risk-table td:first-child { text-align: left; font-weight: 600; color: var(--muted); font-size: 12px; }

/* ── Utilities ── */
.up { color: var(--green); }
.down { color: var(--red); }
.flat { color: var(--muted); }
.dim { color: var(--muted); }
.section-title {
  font-size: 11px;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin: 24px 0 12px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}
.verdict-pass { color: var(--green); }
.verdict-warn { color: var(--yellow); }
.verdict-fail { color: var(--red); }

/* ── Media Gallery — ZeroHedge + X ── */
.mg-wrap { margin-bottom: 24px; }
.mg-zh-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  grid-auto-rows: max-content;
  gap: 16px;
  margin-bottom: 24px;
  align-items: start;
}
.mg-block {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  max-height: 480px;
}
.mg-block .mg-img {
  max-height: 280px;
  object-fit: contain;
}
.mg-block-title {
  padding: 10px 14px 8px;
  font-size: 13px;
  font-weight: 700;
  color: var(--text);
  line-height: 1.35;
  border-bottom: 1px solid var(--border);
}
.mg-img {
  width: 100%;
  display: block;
  background: #111;
}
.mg-block-body {
  padding: 10px 14px;
  font-size: 12.5px;
  color: var(--muted);
  line-height: 1.6;
  margin: 0;
  flex: 1;
}
.mg-x-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.mg-x-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
}
.mg-x-author {
  display: block;
  padding: 8px 12px 6px;
  font-size: 11.5px;
  font-weight: 700;
  color: var(--cyan);
}
.mg-x-text {
  padding: 8px 12px 10px;
  font-size: 12.5px;
  color: var(--muted);
  line-height: 1.55;
  margin: 0;
}

/* ── Footer ── */
.footer { margin-top: 40px; text-align: center; font-size: 11px; color: var(--muted); }
"""

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Week Ahead — {run_date}</title>
<style>{css}</style>
</head>
<body>
<div class="page">

  <!-- Hero -->
  <div class="hero">
    <div class="hero-top">
      <div>
        <div class="hero-badge">📅 Week Ahead · Segunda-feira</div>
        <div class="hero-date">{run_date_full}</div>
      </div>
      <div style="text-align:right">
        <div class="hero-badge" style="background:rgba(167,139,250,0.1);border-color:rgba(167,139,250,0.3);color:#a78bfa">
          Verificação: <span class="verdict-{verdict_cls}">{verdict_icon} {verdict}</span>
        </div>
      </div>
    </div>
    <div class="hero-narrative">{primary_label}</div>
    <div class="hero-secondary">{secondary_label}</div>
    <div class="hero-conf">
      <div class="conf-chip">Confiança primária: <strong>{primary_conf}</strong></div>
      <div class="conf-chip">Itens pontuados: <strong>{scored_items}</strong></div>
      <div class="conf-chip">Modo: <strong>{mode}</strong></div>
    </div>
  </div>

  <!-- Heatmap visual -->
  {heatmap_html}

  <!-- Calendário + Mercado (tabela detalhada) -->
  <div class="grid-2">
    <div class="card">
      <div class="card-header">📅 Agenda Econômica — Próximos 14 dias</div>
      <div class="card-body" style="padding:0">
        {calendar_html}
      </div>
    </div>
    <div class="card">
      <div class="card-header">📊 Mercado — Preços e Retornos</div>
      <div class="card-body" style="padding:0">
        {market_html}
      </div>
    </div>
  </div>

  <!-- FRED Dashboard -->
  <div class="section-title">🏛️ Dashboard Macro — FRED</div>
  <div class="fred-grid">
    {fred_html}
  </div>

  <!-- Cenários Bull/Base/Bear -->
  {scenarios_html}

  <!-- Polymarket + Risk numa linha -->
  <div class="grid-2">
    {polymarket_html}
    {risk_html}
  </div>

  <!-- Monte Carlo -->
  {monte_carlo_html}

  <!-- Galeria de imagens — ZeroHedge + X -->
  {media_gallery_html}

  <!-- Texto principal -->
  {editorial_main_html}

  <!-- Consolidação -->
  {consolidacao_html}

  <div class="footer">
    Gerado em {generated_at} · Agente Editorial · bundle {run_id}
  </div>
</div>
</body>
</html>"""


# ── Função principal ───────────────────────────────────────────────────────────

def save_week_ahead_brief(
    bundle: DailyIngestionBundle,
    curation_path: str | None = None,
) -> Path:
    """
    Gera e salva o Week Ahead Brief HTML.

    Args:
        bundle: DailyIngestionBundle do dia
        curation_path: caminho do JSON de curação (opcional, para texto editorial)

    Returns:
        Path do HTML gerado
    """
    run_date = bundle.run_date
    fred_data = bundle.fred_data or {}
    market_prices = bundle.market_prices or {}

    # ── Output dir (necessário antes da galeria de imagens) ─────────────────
    out_dir = workspace.bundles / str(run_date)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Calendário ─────────────────────────────────────────────────────────
    calendar = fred_data.get("calendar", [])
    calendar_html = _build_calendar_html(calendar)

    # ── Mercado ─────────────────────────────────────────────────────────────
    market_html = _build_market_table(market_prices)
    heatmap_html = _build_market_heatmap(market_prices)

    # ── FRED ────────────────────────────────────────────────────────────────
    fred_series = fred_data.get("series", {})
    fred_html = _build_fred_section(fred_series)

    # ── Curação ──────────────────────────────────────────────────────────────
    primary_label = "—"
    secondary_label = ""
    primary_conf = "—"
    scored_items = "—"
    verdict = "—"
    verdict_cls = "warn"
    verdict_icon = "?"
    mode = "week_ahead"

    if curation_path and Path(curation_path).exists():
        try:
            c = json.loads(Path(curation_path).read_text(encoding="utf-8"))
            narr = c.get("narrative", {})
            ps = narr.get("primary_signal", {})
            ss = narr.get("secondary_signals", [])
            primary_label = ps.get("label", "—")
            primary_conf = f"{ps.get('confidence', 0):.0%}"
            scored_items = str(len(c.get("scored_items", [])))
            verdict = c.get("verification", {}).get("overall_verdict", "—")
            verdict_cls = {"pass": "pass", "warn": "warn", "fail": "fail"}.get(verdict, "warn")
            verdict_icon = {"pass": "✓", "warn": "⚠", "fail": "✕"}.get(verdict, "?")
            if ss:
                s0 = ss[0]
                secondary_label = f"↳ {s0.get('label', '')} ({s0.get('confidence', 0):.0%})"
            ap = c.get("artifact_paths", {})
            mode = ap.get("written_mode", "week_ahead")
        except Exception as exc:
            _log.warning("curation_parse_error", error=str(exc))

    # ── Cenários / Monte Carlo / Risk / Polymarket / Gráficos ─────────────────
    scenarios_html = ""
    polymarket_html = ""
    monte_carlo_html = ""
    risk_html = ""

    # Carrega enrichment artifacts do curation mais recente
    enrichment_data: dict = {}
    enrichment_ap: dict = {}
    if curation_path and Path(curation_path).exists():
        try:
            enrichment_ap = json.loads(Path(curation_path).read_text(encoding="utf-8")).get("artifact_paths", {})
            for key in ("scenarios", "monte_carlo", "risk"):
                p = enrichment_ap.get(key)
                if p and Path(p).exists():
                    enrichment_data[key] = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception as exc:
            _log.warning("enrichment_load_error", error=str(exc))

    if enrichment_data.get("scenarios"):
        scenarios_html = _build_scenarios_html(enrichment_data["scenarios"])

    polymarket_html = _build_polymarket_html(bundle.polymarket_markets or [])

    if enrichment_data.get("monte_carlo"):
        monte_carlo_html = _build_monte_carlo_html(enrichment_data["monte_carlo"])

    if enrichment_data.get("risk"):
        risk_html = _build_risk_html(enrichment_data["risk"])

    # ── Galeria de imagens ────────────────────────────────────────────────────
    media_gallery_html = _build_media_gallery(bundle, out_dir)

    # ── Texto editorial ───────────────────────────────────────────────────────
    rich_sections = _extract_written_sections(curation_path)
    # também extrai texto plano para publicacao.html
    sections = {k: "\n\n".join(it["text"] for it in v) for k, v in rich_sections.items()}

    # Coleta imagens locais do ZeroHedge para intercalar no editorial (sem duplicatas, base64)
    _seen_imgs: set[str] = set()
    _zh_inline_imgs: list[str] = []
    for bl in bundle.market_ear_blocks:
        for ref in bl.image_refs:
            key = Path(ref).name if Path(ref).exists() else ref
            if key in _seen_imgs:
                continue
            _seen_imgs.add(key)
            src = _img_src(ref, out_dir)
            if src:
                _zh_inline_imgs.append(src)

    editorial_main_html = ""
    if rich_sections.get("TEXTO PRINCIPAL"):
        body = _rich_to_html(rich_sections["TEXTO PRINCIPAL"], inline_images=_zh_inline_imgs, out_dir=out_dir)
        editorial_main_html = f"""
        <div class="editorial">
          <div class="editorial-header">✍️ Texto Principal — Week Ahead</div>
          <div class="editorial-body">{body}</div>
        </div>"""

    # Consolidação
    consolidacao_html = ""
    if sections.get("CONSOLIDAÇÃO ESTRUTURADA"):
        raw = sections["CONSOLIDAÇÃO ESTRUTURADA"]
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        rows = ""
        for line in lines:
            if ":" in line:
                k, _, v = line.partition(":")
                rows += f"<tr><td class='dim' style='padding:6px 10px;white-space:nowrap'>{k.strip()}</td><td style='padding:6px 10px'>{v.strip()}</td></tr>"
        if rows:
            consolidacao_html = f"""
            <div class="card" style="margin-bottom:20px">
              <div class="card-header">📋 Consolidação Estruturada</div>
              <div class="card-body" style="padding:0">
                <table style="width:100%;border-collapse:collapse">{rows}</table>
              </div>
            </div>"""

    # ── Monta HTML ────────────────────────────────────────────────────────────
    try:
        run_date_full = datetime.strptime(str(run_date), "%Y-%m-%d").strftime("%A, %d de %B de %Y")
    except Exception:
        run_date_full = str(run_date)

    html = _HTML_TEMPLATE.format(
        css=_CSS,
        run_date=run_date,
        run_date_full=run_date_full,
        primary_label=primary_label,
        secondary_label=secondary_label,
        primary_conf=primary_conf,
        scored_items=scored_items,
        verdict=verdict,
        verdict_cls=verdict_cls,
        verdict_icon=verdict_icon,
        mode=mode,
        heatmap_html=heatmap_html,
        calendar_html=calendar_html,
        market_html=market_html,
        fred_html=fred_html,
        media_gallery_html=media_gallery_html,
        editorial_main_html=editorial_main_html,
        consolidacao_html=consolidacao_html,
        scenarios_html=scenarios_html,
        polymarket_html=polymarket_html,
        monte_carlo_html=monte_carlo_html,
        risk_html=risk_html,
        generated_at=datetime.now().strftime("%d/%m/%Y %H:%M"),
        run_id=bundle.run_id[:12],
    )

    # ── Salva brief ───────────────────────────────────────────────────────────
    path = out_dir / f"{bundle.run_id}_week_ahead_brief.html"
    path.write_text(html, encoding="utf-8")
    _log.info("week_ahead_brief_saved", path=str(path))

    # ── Gera publicacao separada (Texto Gratuito + Micro Posts) ───────────────
    pub_path = _save_publicacao(
        sections=sections,
        run_date=run_date,
        run_date_full=run_date_full,
        primary_label=primary_label,
        run_id=bundle.run_id,
        out_dir=out_dir,
    )
    if pub_path:
        _log.info("publicacao_saved", path=str(pub_path))

    return path


# ── Publicação separada: Texto Gratuito + Micro Posts ─────────────────────────

_PUB_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #06080f;
  color: #e6edf3;
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.7;
  min-height: 100vh;
}
.page { max-width: 760px; margin: 0 auto; padding: 32px 20px 60px; }

/* Header */
.pub-header {
  border-bottom: 1px solid #21262d;
  padding-bottom: 20px;
  margin-bottom: 28px;
}
.pub-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(34,197,94,0.1);
  border: 1px solid rgba(34,197,94,0.3);
  color: #22c55e;
  border-radius: 6px;
  padding: 4px 12px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 12px;
}
.pub-title {
  font-size: 13px;
  color: #8b949e;
  margin-bottom: 4px;
}
.pub-narrative {
  font-size: 18px;
  font-weight: 700;
  color: #e6edf3;
  line-height: 1.35;
}
.pub-date { font-size: 12px; color: #8b949e; margin-top: 8px; }

/* Texto gratuito */
.free-section { margin-bottom: 36px; }
.section-label {
  font-size: 11px;
  font-weight: 700;
  color: #f59e0b;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: #21262d;
}
.free-text p {
  margin-bottom: 16px;
  color: #e6edf3;
  font-size: 15px;
  line-height: 1.8;
}
.free-text p:last-child { margin-bottom: 0; }

/* Copy button */
.copy-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(56,189,248,0.08);
  border: 1px solid rgba(56,189,248,0.25);
  color: #38bdf8;
  border-radius: 6px;
  padding: 6px 14px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  margin-top: 14px;
  transition: background 0.15s;
}
.copy-btn:hover { background: rgba(56,189,248,0.16); }
.copy-btn.copied { color: #22c55e; border-color: rgba(34,197,94,0.3); background: rgba(34,197,94,0.08); }

/* Micro posts */
.posts-section { margin-bottom: 36px; }
.post-card {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 10px;
  padding: 16px 18px;
  margin-bottom: 12px;
  position: relative;
}
.post-card:last-child { margin-bottom: 0; }
.post-num {
  font-size: 10px;
  font-weight: 700;
  color: #38bdf8;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 8px;
}
.post-text {
  font-size: 14px;
  line-height: 1.65;
  color: #e6edf3;
  white-space: pre-wrap;
}
.post-actions {
  display: flex;
  gap: 8px;
  margin-top: 12px;
}
.post-char {
  font-size: 11px;
  color: #8b949e;
  align-self: center;
  margin-left: auto;
}

/* Whatsapp */
.wa-section { margin-bottom: 36px; }
.wa-card {
  background: linear-gradient(135deg, #0a1f12 0%, #0d1117 100%);
  border: 1px solid rgba(34,197,94,0.2);
  border-radius: 10px;
  padding: 16px 18px;
}
.wa-text {
  font-size: 14px;
  line-height: 1.7;
  color: #e6edf3;
  white-space: pre-wrap;
}

/* Footer */
.footer { margin-top: 40px; text-align: center; font-size: 11px; color: #8b949e; border-top: 1px solid #21262d; padding-top: 20px; }
"""

_PUB_JS = """
function copyText(id, btn) {
  const el = document.getElementById(id);
  navigator.clipboard.writeText(el.innerText).then(() => {
    btn.classList.add('copied');
    btn.innerHTML = '✓ Copiado';
    setTimeout(() => { btn.classList.remove('copied'); btn.innerHTML = '⎘ Copiar'; }, 2000);
  });
}
"""

_PUB_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Publicação — {run_date}</title>
<style>{css}</style>
</head>
<body>
<div class="page">

  <div class="pub-header">
    <div class="pub-badge">🔓 Conteúdo para publicação</div>
    <div class="pub-title">Week Ahead · {run_date_full}</div>
    <div class="pub-narrative">{primary_label}</div>
  </div>

  {free_section}
  {wa_section}
  {posts_section}

  <div class="footer">
    Agente Editorial · {run_date} · {run_id}
  </div>
</div>
<script>{js}</script>
</body>
</html>"""


def _save_publicacao(
    sections: dict[str, str],
    run_date: object,
    run_date_full: str,
    primary_label: str,
    run_id: str,
    out_dir: Path,
) -> Path | None:
    """Gera HTML separado com Texto Gratuito + WhatsApp + Micro Posts."""

    # Texto gratuito desativado — somente versão paga/premium é oficial
    wa_text = sections.get("VERSÃO WHATSAPP", "")
    micro_raw = sections.get("MICRO POSTS", "")

    if not micro_raw and not wa_text:
        return None

    free_section = ""  # desativado

    # ── WhatsApp ──────────────────────────────────────────────────────────────
    wa_section = ""
    if wa_text:
        wa_clean = wa_text.strip()
        wa_section = f"""
        <div class="wa-section">
          <div class="section-label">💬 Versão WhatsApp</div>
          <div class="wa-card">
            <div class="wa-text" id="wa-text">{wa_clean}</div>
          </div>
          <button class="copy-btn" onclick="copyText('wa-text', this)">⎘ Copiar</button>
        </div>"""

    # ── Micro Posts ───────────────────────────────────────────────────────────
    posts_section = ""
    if micro_raw:
        raw_posts = re.split(r"POST\s*\d+\s*[:.]", micro_raw)
        posts = [p.strip() for p in raw_posts if p.strip()]
        cards = ""
        for i, post in enumerate(posts, 1):
            char_count = len(post)
            color = "#ef4444" if char_count > 280 else "#8b949e"
            cards += f"""
            <div class="post-card">
              <div class="post-num">Post {i}</div>
              <div class="post-text" id="post-{i}">{post}</div>
              <div class="post-actions">
                <button class="copy-btn" onclick="copyText('post-{i}', this)">⎘ Copiar</button>
                <span class="post-char" style="color:{color}">{char_count} chars</span>
              </div>
            </div>"""

        posts_section = f"""
        <div class="posts-section">
          <div class="section-label">📱 Micro Posts — X / Substack Notes / WhatsApp</div>
          {cards}
        </div>"""

    html = _PUB_TEMPLATE.format(
        css=_PUB_CSS,
        js=_PUB_JS,
        run_date=run_date,
        run_date_full=run_date_full,
        primary_label=primary_label,
        free_section=free_section,
        wa_section=wa_section,
        posts_section=posts_section,
        run_id=run_id[:12],
    )

    path = out_dir / f"{run_id}_publicacao.html"
    path.write_text(html, encoding="utf-8")
    return path


# ── Writer Brief universal (todos os modos) ────────────────────────────────────

_MODE_META: dict[str, tuple[str, str, str]] = {
    "week_ahead":     ("📅", "Week Ahead",       "Segunda-feira"),
    "growth":         ("📈", "Growth Stocks",    "Terça-feira"),
    "flow_show":      ("🌊", "The Flow Show",    "Quarta-feira"),
    "tese":           ("🎯", "Tese do Dia",      "Quinta-feira"),
    "week_recap":     ("📊", "Week Recap",       "Sexta-feira"),
    "podcast_sabado": ("🎙️", "Podcast Sábado",  "Sábado"),
    "tese_livre":     ("🖊️", "Tese Livre",       "Domingo"),
    "morning_call":   ("☀️", "Morning Call",     ""),
}


def _build_editorial_sections_html(
    rich_sections: dict[str, list[dict]],
    mode: str,
    zh_inline_imgs: list[str],
    out_dir: Path | None = None,
) -> str:
    """Renderiza todas as seções editoriais do writer em HTML."""
    parts: list[str] = []

    # ── Texto Principal ───────────────────────────────────────────────────────
    # Each mode may use a different section name as main body
    _MAIN_ALIASES = {
        "week_recap":     "WEEK RECAP",
        "week_ahead":     "WEEK AHEAD",
        "growth":         "GROWTH STOCKS",
        "flow_show":      "THE FLOW SHOW",
        "tese":           "TESE DO DIA",
        "tese_livre":     "TESE LIVRE",
        "morning_call":   "MORNING CALL",
    }
    sec = rich_sections.get("TEXTO PRINCIPAL") or rich_sections.get(_MAIN_ALIASES.get(mode, ""))
    if sec:
        body = _rich_to_html(sec, inline_images=zh_inline_imgs, out_dir=out_dir)
        emoji, title, _ = _MODE_META.get(mode, ("✍️", mode.title(), ""))
        parts.append(f"""
        <div class="editorial" style="margin-bottom:24px">
          <div class="editorial-header">{emoji} Texto Principal — {title}</div>
          <div class="editorial-body">{body}</div>
        </div>""")

    # Texto Gratuito desativado — apenas versão paga/premium é exibida no relatório

    # ── Podcast: Título + Descrição + Script ──────────────────────────────────
    if mode == "podcast_sabado":
        titulo = rich_sections.get("TÍTULO DO EPISÓDIO")
        if titulo:
            text = " ".join(it["text"] for it in titulo)
            parts.append(f"""
            <div class="card" style="margin-bottom:20px">
              <div class="card-header">🎙️ Título do Episódio</div>
              <div class="card-body" style="font-size:18px;font-weight:700;color:var(--cyan)">{text}</div>
            </div>""")

        desc = rich_sections.get("DESCRIÇÃO DO EPISÓDIO")
        if desc:
            body = _rich_to_html(desc)
            parts.append(f"""
            <div class="card" style="margin-bottom:20px">
              <div class="card-header">📝 Descrição do Episódio</div>
              <div class="card-body editorial-body">{body}</div>
            </div>""")

        script = rich_sections.get("SCRIPT PODCAST")
        if script:
            text = "\n".join(it["text"] for it in script)
            parts.append(f"""
            <div class="card" style="margin-bottom:20px">
              <div class="card-header">🎤 Script do Podcast</div>
              <div class="card-body" style="white-space:pre-wrap;font-size:13px;line-height:1.8;color:var(--text)">{text}</div>
            </div>""")

        comentario = rich_sections.get("COMENTÁRIO DO POST")
        if comentario:
            body = _rich_to_html(comentario)
            parts.append(f"""
            <div class="card" style="margin-bottom:20px;border-color:rgba(56,189,248,0.3)">
              <div class="card-header">💬 Comentário Fixado</div>
              <div class="card-body editorial-body">{body}</div>
            </div>""")

        notes = rich_sections.get("SUBSTACK NOTES")
        if notes:
            body = _rich_to_html(notes)
            parts.append(f"""
            <div class="card" style="margin-bottom:20px">
              <div class="card-header">📰 Substack Notes</div>
              <div class="card-body editorial-body">{body}</div>
            </div>""")

    # ── Consolidação Estruturada ───────────────────────────────────────────────
    cons = rich_sections.get("CONSOLIDAÇÃO ESTRUTURADA")
    if cons:
        raw_lines = [it["text"] for it in cons]
        rows = ""
        for line in raw_lines:
            if ":" in line:
                k, _, v = line.partition(":")
                rows += (
                    f"<tr><td style='padding:7px 12px;white-space:nowrap;"
                    f"color:var(--muted);font-size:12px'>{k.strip()}</td>"
                    f"<td style='padding:7px 12px;font-size:13px'>{v.strip()}</td></tr>"
                )
        if rows:
            parts.append(f"""
            <div class="card" style="margin-bottom:20px">
              <div class="card-header">📋 Consolidação Estruturada</div>
              <div class="card-body" style="padding:0">
                <table style="width:100%;border-collapse:collapse">{rows}</table>
              </div>
            </div>""")

    # ── Ângulos ────────────────────────────────────────────────────────────────
    ang_p = rich_sections.get("ÂNGULO PRINCIPAL")
    ang_s = rich_sections.get("ÂNGULOS SECUNDÁRIOS")
    if ang_p or ang_s:
        cols = ""
        if ang_p:
            text = " ".join(it["text"] for it in ang_p)
            cols += f"""
            <div style="flex:1;min-width:240px;background:var(--surface);border:1px solid rgba(56,189,248,0.2);
                        border-radius:10px;padding:14px 16px">
              <div style="font-size:11px;font-weight:700;color:var(--cyan);letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px">
                🎯 Ângulo Principal
              </div>
              <div style="font-size:13.5px;line-height:1.6">{text}</div>
            </div>"""
        if ang_s:
            text = " ".join(it["text"] for it in ang_s)
            cols += f"""
            <div style="flex:1;min-width:240px;background:var(--surface);border:1px solid var(--border);
                        border-radius:10px;padding:14px 16px">
              <div style="font-size:11px;font-weight:700;color:var(--muted);letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px">
                ↳ Ângulos Secundários
              </div>
              <div style="font-size:13px;line-height:1.6;color:var(--muted)">{text}</div>
            </div>"""
        parts.append(f"""
        <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px">{cols}</div>""")

    # ── Micro Posts ────────────────────────────────────────────────────────────
    micro_raw_items = rich_sections.get("MICRO POSTS")
    if micro_raw_items:
        raw_text = "\n".join(it["text"] for it in micro_raw_items)
        raw_posts = re.split(r"POST\s*\d+\s*[:.]", raw_text)
        posts = [p.strip() for p in raw_posts if p.strip()]
        if posts:
            cards = ""
            for i, post in enumerate(posts, 1):
                color = "#ef4444" if len(post) > 280 else "#8b949e"
                cards += f"""
                <div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 14px;margin-bottom:10px">
                  <div style="font-size:10px;font-weight:700;color:var(--muted);text-transform:uppercase;margin-bottom:6px">Post {i}</div>
                  <div style="font-size:13.5px;line-height:1.6">{post}</div>
                  <div style="font-size:11px;color:{color};margin-top:6px">{len(post)} chars</div>
                </div>"""
            parts.append(f"""
            <div class="card" style="margin-bottom:20px">
              <div class="card-header">📱 Micro Posts — X / Substack Notes</div>
              <div class="card-body">{cards}</div>
            </div>""")

    # ── Versão WhatsApp ────────────────────────────────────────────────────────
    wa = rich_sections.get("VERSÃO WHATSAPP")
    if wa:
        text = "\n".join(it["text"] for it in wa)
        parts.append(f"""
        <div class="card" style="margin-bottom:20px;border-color:rgba(34,197,94,0.2);
                                  background:linear-gradient(135deg,#0a1f12 0%,#0d1117 100%)">
          <div class="card-header" style="color:var(--green)">💬 Versão WhatsApp</div>
          <div class="card-body" style="white-space:pre-wrap;font-size:14px;line-height:1.7">{text}</div>
        </div>""")

    return "\n".join(parts)


def save_writer_brief(
    bundle: "DailyIngestionBundle",
    curation_path: str | None = None,
    swaggy_result=None,   # SwaggyResult | None
    signals: dict | None = None,  # dict[str, AssetSignal] com _zone_signal
) -> Path:
    """
    Gera o Writer Brief HTML para QUALQUER modo editorial.

    Usa o mesmo CSS/design do Week Ahead Brief mas adapta:
    - Hero badge e título ao modo do dia
    - Mostra calendário + FRED apenas para week_ahead / week_recap
    - Renderiza todas as seções editoriais do writer output

    Args:
        bundle: DailyIngestionBundle do dia
        curation_path: caminho do JSON de curação (opcional)

    Returns:
        Path do HTML gerado
    """
    run_date = bundle.run_date
    fred_data = bundle.fred_data or {}
    market_prices = bundle.market_prices or {}

    out_dir = workspace.bundles / str(run_date)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Modo e metadados ──────────────────────────────────────────────────────
    mode = "morning_call"
    primary_label = "—"
    secondary_label = ""
    primary_conf = "—"
    scored_items = "—"
    verdict = "—"
    verdict_cls = "warn"
    verdict_icon = "?"

    if curation_path and Path(curation_path).exists():
        try:
            c = json.loads(Path(curation_path).read_text(encoding="utf-8"))
            narr = c.get("narrative", {})
            ps = narr.get("primary_signal", {})
            ss = narr.get("secondary_signals", [])
            primary_label = ps.get("label", "—")
            primary_conf = f"{ps.get('confidence', 0):.0%}"
            scored_items = str(len(c.get("scored_items", [])))
            verdict = c.get("verification", {}).get("overall_verdict", "—")
            verdict_cls = {"pass": "pass", "warn": "warn", "fail": "fail"}.get(verdict, "warn")
            verdict_icon = {"pass": "✓", "warn": "⚠", "fail": "✕"}.get(verdict, "?")
            if ss:
                s0 = ss[0]
                secondary_label = f"↳ {s0.get('label', '')} ({s0.get('confidence', 0):.0%})"
            mode = c.get("artifact_paths", {}).get("written_mode", mode)
        except Exception as exc:
            _log.warning("writer_brief_curation_parse", error=str(exc))

    emoji, mode_title, mode_day = _MODE_META.get(mode, ("✍️", mode.title(), ""))

    try:
        run_date_full = datetime.strptime(str(run_date), "%Y-%m-%d").strftime("%A, %d de %B de %Y")
    except Exception:
        run_date_full = str(run_date)

    # ── Heatmap + Mercado ─────────────────────────────────────────────────────
    heatmap_html = _build_market_heatmap(market_prices)
    market_html = _build_market_table(market_prices)

    # ── Calendário + FRED — sempre visível ───────────────────────────────────
    show_macro = True
    calendar_html = _build_calendar_html(fred_data.get("calendar", []))
    fred_html = _build_fred_section(fred_data.get("series", {}))

    # ── Enrichment ────────────────────────────────────────────────────────────
    enrichment_data: dict = {}
    ap: dict = {}
    if curation_path and Path(curation_path).exists():
        try:
            ap = json.loads(Path(curation_path).read_text(encoding="utf-8")).get("artifact_paths", {})
            for key in ("scenarios", "monte_carlo", "risk"):
                p = ap.get(key)
                if p and Path(p).exists():
                    enrichment_data[key] = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            pass

    scenarios_html = _build_scenarios_html(enrichment_data["scenarios"]) if enrichment_data.get("scenarios") else ""
    polymarket_html = _build_polymarket_html(bundle.polymarket_markets or [])
    monte_carlo_html = _build_monte_carlo_html(enrichment_data["monte_carlo"]) if enrichment_data.get("monte_carlo") else ""
    risk_html = _build_risk_html(enrichment_data["risk"]) if enrichment_data.get("risk") else ""

    # ── WSB + Squeeze ─────────────────────────────────────────────────────────
    swaggy_html = _build_swaggy_section(swaggy_result)

    # ── TradingView — Value Area zones ───────────────────────────────────────
    tv_zones_html = _build_tv_zones_section(signals or {})

    # ── Galeria de imagens ────────────────────────────────────────────────────
    media_gallery_html = _build_media_gallery(bundle, out_dir)

    # ── Texto editorial — todas as seções ────────────────────────────────────
    rich_sections = _extract_written_sections(curation_path)

    _seen_imgs: set[str] = set()
    _zh_inline_imgs: list[str] = []
    for bl in bundle.market_ear_blocks:
        for ref in bl.image_refs:
            key = Path(ref).name if Path(ref).exists() else ref
            if key in _seen_imgs:
                continue
            _seen_imgs.add(key)
            src = _img_src(ref, out_dir)
            if src:
                _zh_inline_imgs.append(src)

    editorial_html = _build_editorial_sections_html(rich_sections, mode, _zh_inline_imgs, out_dir=out_dir)

    # ── Monta HTML ────────────────────────────────────────────────────────────
    day_suffix = f" · {mode_day}" if mode_day else ""
    macro_grid = ""
    if show_macro:
        macro_grid = f"""
  <div class="grid-2">
    <div class="card">
      <div class="card-header">📅 Agenda Econômica — Próximos 14 dias</div>
      <div class="card-body" style="padding:0">{calendar_html}</div>
    </div>
    <div class="card">
      <div class="card-header">📊 Mercado — Preços e Retornos</div>
      <div class="card-body" style="padding:0">{market_html}</div>
    </div>
  </div>
  <div class="section-title">🏛️ Dashboard Macro — FRED</div>
  <div class="fred-grid">{fred_html}</div>"""
    else:
        macro_grid = f"""
  <div class="card" style="margin-bottom:20px">
    <div class="card-header">📊 Mercado — Preços e Retornos</div>
    <div class="card-body" style="padding:0">{market_html}</div>
  </div>"""

    poly_risk_html = ""
    if polymarket_html or risk_html:
        poly_risk_html = f'<div class="grid-2">{polymarket_html}{risk_html}</div>'

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{mode_title} — {run_date}</title>
<style>{_CSS}
.editorial {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
  margin-bottom: 24px;
}}
.editorial-header {{
  background: var(--surface2);
  border-bottom: 1px solid var(--border);
  padding: 10px 18px;
  font-size: 12px;
  font-weight: 600;
  color: var(--cyan);
  letter-spacing: .05em;
  text-transform: uppercase;
}}
.editorial-body {{
  padding: 18px 20px;
  font-size: 14.5px;
  line-height: 1.8;
  color: var(--text);
}}
.editorial-body p {{ margin-bottom: 1rem; }}
.editorial-body h2.editorial-title {{
  font-size: 18px;
  font-weight: 700;
  color: var(--text);
  margin: 1.4rem 0 .6rem;
  line-height: 1.3;
}}
.editorial-body .editorial-callout {{
  background: var(--surface2);
  border-left: 3px solid var(--cyan);
  border-radius: 0 6px 6px 0;
  padding: .6rem 1rem;
  margin: .8rem 0;
  font-size: 13.5px;
  color: var(--muted);
  line-height: 1.6;
}}
.editorial-body .inline-img-wrap {{ margin: 1.2rem 0; text-align: center; }}
.editorial-body .inline-img {{ max-width: 100%; border-radius: 8px;
                               border: 1px solid var(--border); }}
.section-title {{
  font-size: 11px;
  font-weight: 700;
  color: var(--muted);
  letter-spacing: .08em;
  text-transform: uppercase;
  margin: 24px 0 12px;
}}
</style>
</head>
<body>
<div class="page">

  <!-- Hero -->
  <div class="hero">
    <div class="hero-top">
      <div>
        <div class="hero-badge">{emoji} {mode_title}{day_suffix}</div>
        <div class="hero-date">{run_date_full}</div>
      </div>
      <div style="text-align:right">
        <div class="hero-badge" style="background:rgba(167,139,250,0.1);border-color:rgba(167,139,250,0.3);color:#a78bfa">
          Verificação: <span class="verdict-{verdict_cls}">{verdict_icon} {verdict}</span>
        </div>
      </div>
    </div>
    <div class="hero-narrative">{primary_label}</div>
    <div class="hero-secondary">{secondary_label}</div>
    <div class="hero-conf">
      <div class="conf-chip">Confiança: <strong>{primary_conf}</strong></div>
      <div class="conf-chip">Itens pontuados: <strong>{scored_items}</strong></div>
      <div class="conf-chip">Modo: <strong>{mode}</strong></div>
    </div>
  </div>

  <!-- Heatmap -->
  {heatmap_html}

  <!-- Calendário / Mercado -->
  {macro_grid}

  <!-- Cenários -->
  {scenarios_html}

  <!-- Polymarket + Risk -->
  {poly_risk_html}

  <!-- Monte Carlo -->
  {monte_carlo_html}

  <!-- WSB & Squeeze -->
  {swaggy_html}

  <!-- TradingView — Value Area Zones -->
  {tv_zones_html}

  <!-- Galeria de imagens -->
  {media_gallery_html}

  <!-- Texto editorial — todas as seções -->
  {editorial_html}

  <div class="footer">
    Gerado em {datetime.now().strftime("%d/%m/%Y %H:%M")} · Agente Editorial · {mode_title} · bundle {bundle.run_id[:12]}
  </div>
</div>
</body>
</html>"""

    path = out_dir / f"{bundle.run_id}_brief.html"
    path.write_text(html, encoding="utf-8")
    _log.info("writer_brief_saved", mode=mode, path=str(path))

    # Também gera a publicação separada (Texto Gratuito + Micro Posts + WhatsApp)
    sections_text = {k: "\n\n".join(it["text"] for it in v) for k, v in rich_sections.items()}
    _save_publicacao(
        sections=sections_text,
        run_date=run_date,
        run_date_full=run_date_full,
        primary_label=primary_label,
        run_id=bundle.run_id,
        out_dir=out_dir,
    )

    return path
