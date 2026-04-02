"""
Week Ahead Brief — PDF export via Playwright.

Gera um relatório PDF profissional com branding Gulfstream Capital:
  - Cover page com logo + título + data
  - Snapshot de mercado (heatmap)
  - Calendário econômico
  - Dashboard macro FRED
  - Cenários Bull/Base/Bear
  - Texto editorial

Suporta PT (padrão) e EN (clientes internacionais).

Uso:
    from app.views.week_ahead_pdf import save_week_ahead_pdf
    path = save_week_ahead_pdf(bundle, curation_path, lang="en")
"""
from __future__ import annotations

import base64
import json
import re
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.audit.logger import get_logger
from app.models.daily_ingestion_bundle import DailyIngestionBundle
from app.storage.paths import workspace

_log = get_logger("views.week_ahead_pdf")

# ── Brand colors Gulfstream Capital ──────────────────────────────────────────
_NAVY   = "#0D192B"
_NAVY2  = "#14202e"
_GOLD   = "#D0A044"
_GOLD2  = "#BF9F5F"
_LIGHT  = "#F6F7F9"
_MUTED  = "#8a9bb0"
_WHITE  = "#ffffff"
_GREEN  = "#22c55e"
_RED    = "#ef4444"
_YELLOW = "#f59e0b"

# ── Logo Gulfstream (base64 cached) ───────────────────────────────────────────
_LOGO_URL = "https://gulfstreamcapital.com.br/wp-content/uploads/2026/01/cropped-favicon-1.png"
_logo_b64: str | None = None


def _get_logo_b64() -> str:
    global _logo_b64
    if _logo_b64:
        return _logo_b64
    # try local cache first
    cache = workspace.bundles.parent / "gulfstream_logo.png"
    if cache.exists():
        _logo_b64 = "data:image/png;base64," + base64.b64encode(cache.read_bytes()).decode()
        return _logo_b64
    try:
        req = urllib.request.Request(_LOGO_URL, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=10).read()
        cache.write_bytes(data)
        _logo_b64 = "data:image/png;base64," + base64.b64encode(data).decode()
    except Exception:
        _logo_b64 = ""
    return _logo_b64


# ── Translations ──────────────────────────────────────────────────────────────
_T: dict[str, dict[str, str]] = {
    "pt": {
        "report_title":   "Week Ahead Brief",
        "report_subtitle":"Análise Macro Semanal",
        "prepared_for":   "Preparado para",
        "confidential":   "CONFIDENCIAL — USO INTERNO",
        "generated":      "Gerado em",
        "narrative":      "Narrativa Principal",
        "confidence":     "Confiança",
        "verification":   "Verificação",
        "scored":         "Itens Analisados",
        "mode":           "Modo",
        "market_snapshot":"Snapshot de Mercado",
        "daily":          "1D",
        "weekly":         "1S",
        "ytd":            "YTD",
        "price":          "Preço",
        "asset":          "Ativo",
        "calendar":       "Agenda Econômica — Próximos 14 dias",
        "date_col":       "Data",
        "event":          "Evento",
        "impact":         "Impacto",
        "fred_title":     "Dashboard Macro — FRED",
        "metric":         "Métrica",
        "value":          "Valor",
        "change":         "Variação",
        "last_date":      "Referência",
        "scenarios":      "Cenários Bull / Base / Bear",
        "probability":    "Probabilidade",
        "catalyst":       "Catalisador",
        "target":         "Alvo SPX",
        "horizon":        "Horizonte",
        "editorial":      "Texto Editorial",
        "poly_title":     "Prediction Markets — Polymarket",
        "question":       "Questão",
        "prob":           "Prob.",
        "volume":         "Volume",
        "risk_title":     "Métricas de Risco",
        "ticker":         "Ticker",
        "var95":          "VaR 95%",
        "cvar95":         "CVaR 95%",
        "maxdd":          "Max DD",
        "sharpe":         "Sharpe",
        "mc_title":       "Monte Carlo — 20 dias",
        "prob_up":        "Prob ↑",
        "prob_p5":        "Prob +5%",
        "prob_m5":        "Prob −5%",
        "p50":            "P50 (20d)",
        "market_gossip":  "Market Gossip",
        "page_of":        "de",
        "disclaimer":     (
            "Este relatório é produzido pela Gulfstream Capital para uso interno e distribuição a clientes qualificados. "
            "Não constitui oferta de compra ou venda de ativos. Desempenho passado não é garantia de resultados futuros."
        ),
    },
    "en": {
        "report_title":   "Week Ahead Brief",
        "report_subtitle":"Weekly Macro Analysis",
        "prepared_for":   "Prepared for",
        "confidential":   "CONFIDENTIAL — INTERNAL USE",
        "generated":      "Generated on",
        "narrative":      "Primary Narrative",
        "confidence":     "Confidence",
        "verification":   "Verification",
        "scored":         "Items Analyzed",
        "mode":           "Mode",
        "market_snapshot":"Market Snapshot",
        "daily":          "1D",
        "weekly":         "1W",
        "ytd":            "YTD",
        "price":          "Price",
        "asset":          "Asset",
        "calendar":       "Economic Calendar — Next 14 Days",
        "date_col":       "Date",
        "event":          "Event",
        "impact":         "Impact",
        "fred_title":     "Macro Dashboard — FRED",
        "metric":         "Metric",
        "value":          "Value",
        "change":         "Change",
        "last_date":      "As of",
        "scenarios":      "Bull / Base / Bear Scenarios",
        "probability":    "Probability",
        "catalyst":       "Catalyst",
        "target":         "SPX Target",
        "horizon":        "Horizon",
        "editorial":      "Editorial",
        "poly_title":     "Prediction Markets — Polymarket",
        "question":       "Question",
        "prob":           "Prob.",
        "volume":         "Volume",
        "risk_title":     "Risk Metrics",
        "ticker":         "Ticker",
        "var95":          "VaR 95%",
        "cvar95":         "CVaR 95%",
        "maxdd":          "Max DD",
        "sharpe":         "Sharpe",
        "mc_title":       "Monte Carlo — 20 days",
        "prob_up":        "Prob ↑",
        "prob_p5":        "Prob +5%",
        "prob_m5":        "Prob −5%",
        "p50":            "P50 (20d)",
        "market_gossip":  "Market Gossip",
        "page_of":        "of",
        "disclaimer":     (
            "This report is produced by Gulfstream Capital for internal use and distribution to qualified clients. "
            "It does not constitute an offer to buy or sell any asset. Past performance is not indicative of future results."
        ),
    },
}

# ── Market order ──────────────────────────────────────────────────────────────
_MARKET_ORDER = [
    ("^GSPC", "S&P 500"), ("^NDX", "Nasdaq 100"), ("^RUT", "Russell 2000"),
    ("TLT", "TLT 20yr Bond"), ("HYG", "HY Credit"), ("GLD", "Gold"),
    ("CL=F", "WTI Crude"), ("BTC-USD", "Bitcoin"), ("DX-Y.NYB", "DXY"), ("^VIX", "VIX"),
]

# ── Impact map (bilingual) ────────────────────────────────────────────────────
_IMPACT_MAP = {
    "employment situation": ("🔴", "High impact", "Alto impacto"),
    "nonfarm payroll":      ("🔴", "High impact", "Alto impacto"),
    "consumer price":       ("🔴", "High impact", "Alto impacto"),
    "fomc":                 ("🔴", "High impact", "Alto impacto"),
    "gdp":                  ("🔴", "High impact", "Alto impacto"),
    "pce price":            ("🔴", "High impact", "Alto impacto"),
    "adp national":         ("🟠", "Medium impact", "Médio impacto"),
    "ism manufactur":       ("🟠", "Medium impact", "Médio impacto"),
    "ism non-manufactur":   ("🟠", "Medium impact", "Médio impacto"),
    "jolts":                ("🟠", "Medium impact", "Médio impacto"),
    "retail sales":         ("🟠", "Medium impact", "Médio impacto"),
    "ppi":                  ("🟠", "Medium impact", "Médio impacto"),
    "michigan consumer":    ("🟡", "Moderate", "Moderado"),
    "housing starts":       ("🟡", "Moderate", "Moderado"),
    "chicago pmi":          ("🟡", "Moderate", "Moderado"),
    "existing home":        ("🟡", "Moderate", "Moderado"),
    "treasury":             ("⚪", "Technical", "Técnico"),
}

_PRIORITY_SERIES = {
    "Monetary Policy / Política Monetária": [
        ("DFF", "Fed Funds Rate", "Fed Funds"),
        ("T10Y2Y", "10y-2y Spread", "Curva 10y-2y"),
        ("T10YIE", "10y Breakeven", "Breakeven 10y"),
        ("WALCL", "Fed Balance Sheet", "Balanço Fed"),
    ],
    "Inflation / Inflação": [
        ("CPIAUCSL", "CPI Total", "CPI Total"),
        ("CPILFESL", "CPI Core", "CPI Core"),
        ("PCEPI", "PCE Total", "PCE Total"),
        ("PCEPILFE", "PCE Core", "PCE Core"),
    ],
    "Labor Market / Mercado de Trabalho": [
        ("UNRATE", "Unemployment", "Desemprego"),
        ("PAYEMS", "Nonfarm Payroll", "Payroll"),
        ("JTSJOL", "JOLTS Openings", "JOLTS Vagas"),
    ],
    "Credit / Crédito": [
        ("BAMLH0A0HYM2", "HY Spread", "HY Spread"),
        ("BAMLC0A0CM", "IG Spread", "IG Spread"),
        ("NFCI", "NFCI", "NFCI"),
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(v: float | None) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v*100:.2f}%"


def _color_pct(v: float | None) -> str:
    if v is None:
        return _MUTED
    return _GREEN if v >= 0 else _RED


def _fmt_price(p: float | None) -> str:
    if p is None:
        return "—"
    if p > 10000:
        return f"{p:,.0f}"
    if p > 100:
        return f"{p:,.2f}"
    return f"{p:.4f}"


def _get_impact(name: str, lang: str) -> tuple[str, str]:
    low = name.lower()
    for kw, (emoji, en, pt) in _IMPACT_MAP.items():
        if kw in low:
            return emoji, (en if lang == "en" else pt)
    return "⚪", ("Monitor" if lang == "en" else "Monitorar")


def _sparkline_svg(values: list, w: int = 60, h: int = 20) -> str:
    vals = [v for v in values if isinstance(v, (int, float)) and v is not None]
    if len(vals) < 2:
        return ""
    mn, mx = min(vals), max(vals)
    rng = mx - mn or 1
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        x = int(i / (n - 1) * (w - 4)) + 2
        y = int((1 - (v - mn) / rng) * (h - 4)) + 2
        pts.append(f"{x},{y}")
    stroke = _GREEN if vals[-1] >= vals[0] else _RED
    return (
        f'<svg width="{w}" height="{h}" style="vertical-align:middle">'
        f'<polyline points="{" ".join(pts)}" fill="none" stroke="{stroke}" '
        f'stroke-width="1.5" stroke-linejoin="round"/>'
        f'</svg>'
    )


# ── Section builders ──────────────────────────────────────────────────────────

def _build_market_table(market_prices: dict, t: dict) -> str:
    rows = []
    for ticker, label in _MARKET_ORDER:
        mp = market_prices.get(ticker, {})
        if not mp:
            continue
        price = mp.get("price")
        daily = mp.get("daily_return")
        weekly = mp.get("weekly_return")
        ytd = mp.get("ytd_return")

        def pct_td(v: float | None) -> str:
            color = _color_pct(v)
            return f'<td style="color:{color};text-align:right;font-weight:600">{_pct(v)}</td>'

        rows.append(f"""
        <tr>
          <td style="color:{_MUTED};font-size:10px;font-weight:700">{ticker}</td>
          <td>{label}</td>
          <td style="text-align:right;font-weight:600">{_fmt_price(price)}</td>
          {pct_td(daily)}{pct_td(weekly)}{pct_td(ytd)}
        </tr>""")

    return f"""
    <table class="data-table">
      <thead>
        <tr>
          <th>Ticker</th><th>{t["asset"]}</th><th style="text-align:right">{t["price"]}</th>
          <th style="text-align:right">{t["daily"]}</th>
          <th style="text-align:right">{t["weekly"]}</th>
          <th style="text-align:right">{t["ytd"]}</th>
        </tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def _build_heatmap(market_prices: dict) -> str:
    _TILES = [
        ("^GSPC","S&P 500"),("^NDX","Nasdaq"),("^RUT","Russell 2K"),
        ("TLT","TLT 20yr"),("HYG","HY Credit"),("GLD","Gold"),
        ("CL=F","WTI"),("BTC-USD","Bitcoin"),("DX-Y.NYB","DXY"),("^VIX","VIX"),
    ]
    def _heat(v):
        if v is None: return "#374151","#9ca3af"
        if v >= 0.04:  return "#064e3b","#34d399"
        if v >= 0.02:  return "#065f46","#6ee7b7"
        if v >= 0.005: return "#14532d","#86efac"
        if v >= -0.005:return "#1a2234","#9ca3af"
        if v >= -0.02: return "#7f1d1d","#fca5a5"
        if v >= -0.04: return "#991b1b","#f87171"
        return "#450a0a","#ef4444"

    tiles = ""
    for ticker, label in _TILES:
        mp = market_prices.get(ticker, {})
        if not mp: continue
        daily = mp.get("daily_return")
        price = mp.get("price")
        bg, fg = _heat(daily)
        ret_str = _pct(daily)
        price_str = _fmt_price(price)
        tiles += f"""
        <div style="background:{bg};color:{fg};border-radius:6px;padding:8px 10px;
                    min-width:80px;flex:1;display:flex;flex-direction:column;gap:2px">
          <span style="font-size:9px;font-weight:700;opacity:0.7">{ticker}</span>
          <span style="font-size:9px;opacity:0.5">{label}</span>
          <span style="font-size:15px;font-weight:800;letter-spacing:-0.02em">{ret_str}</span>
          <span style="font-size:9px;opacity:0.5">{price_str}</span>
        </div>"""

    return f'<div style="display:flex;flex-wrap:wrap;gap:4px;padding:4px 0">{tiles}</div>'


def _build_calendar(calendar: list[dict], t: dict, lang: str) -> str:
    if not calendar:
        return "<p style='color:#666'>No events / Sem eventos</p>"
    rows = []
    prev = None
    for item in sorted(calendar, key=lambda x: x.get("date", "")):
        d = item.get("date", "")
        name = item.get("release_name", "")
        emoji, label = _get_impact(name, lang)
        try:
            dt = date.fromisoformat(d)
            day_str = dt.strftime("%a %d/%m")
            is_today = dt == date.today()
            badge = ' <span style="background:#D0A044;color:#000;font-size:8px;font-weight:800;padding:1px 4px;border-radius:2px">TODAY</span>' if is_today else ""
        except Exception:
            day_str = d
            badge = ""
        sep = "<tr><td colspan='3' style='border-top:1px solid #ddd;padding:0'></td></tr>" if d != prev and prev else ""
        prev = d
        rows.append(f"""
        {sep}
        <tr>
          <td style="white-space:nowrap;color:#555;font-size:10px">{day_str}{badge}</td>
          <td>{name}</td>
          <td style="white-space:nowrap;text-align:right">{emoji} <span style="font-size:9px;color:#777">{label}</span></td>
        </tr>""")
    return f"""
    <table class="data-table" style="font-size:11px">
      <thead><tr><th>{t["date_col"]}</th><th>{t["event"]}</th><th>{t["impact"]}</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def _build_fred(series_data: dict, t: dict, lang: str) -> str:
    blocks = []
    for cat, rows_def in _PRIORITY_SERIES.items():
        cat_label, _, cat_pt = cat.partition(" / ")
        display = cat_label if lang == "en" else cat_pt
        cat_data = series_data.get(cat_pt, []) or series_data.get(cat_label, [])
        # try both keys
        if not cat_data:
            for key in series_data:
                if cat_pt in key or cat_label in key:
                    cat_data = series_data[key]
                    break
        by_id = {s["series_id"]: s for s in cat_data}

        rows = []
        for sid, en_label, pt_label in rows_def:
            s = by_id.get(sid)
            if not s:
                continue
            label = en_label if lang == "en" else pt_label
            val = s.get("value")
            chg = s.get("change")
            unit = s.get("unit", "")
            history = s.get("history", [])
            last_date = s.get("date", "")

            val_str = f"{val:.2f}{unit}" if val is not None and "%" in unit else (f"{val:,.2f}" if val is not None else "—")
            chg_color = _GREEN if (chg or 0) > 0 else (_RED if (chg or 0) < 0 else _MUTED)
            chg_str = (f"+{chg:.3g}" if chg and chg > 0 else f"{chg:.3g}") if chg is not None else "—"
            spark = _sparkline_svg(history) if history else ""

            rows.append(f"""
            <tr>
              <td style="font-size:11px">{label}</td>
              <td style="text-align:right;font-weight:600;font-size:11px">{val_str}</td>
              <td style="text-align:right;color:{chg_color};font-size:10px">{chg_str}</td>
              <td style="text-align:center">{spark}</td>
              <td style="color:#999;font-size:9px;text-align:right">{last_date}</td>
            </tr>""")

        if not rows:
            continue

        blocks.append(f"""
        <div style="margin-bottom:16px">
          <div style="font-size:10px;font-weight:700;color:{_GOLD};text-transform:uppercase;
                      letter-spacing:0.06em;margin-bottom:6px;padding-bottom:4px;
                      border-bottom:1px solid #e2e8f0">{display}</div>
          <table class="data-table" style="font-size:11px">
            <thead><tr>
              <th>{t["metric"]}</th>
              <th style="text-align:right">{t["value"]}</th>
              <th style="text-align:right">{t["change"]}</th>
              <th style="text-align:center">Chart</th>
              <th style="text-align:right">{t["last_date"]}</th>
            </tr></thead>
            <tbody>{"".join(rows)}</tbody>
          </table>
        </div>""")

    return "".join(blocks) if blocks else "<p style='color:#999'>No FRED data available.</p>"


def _build_scenarios(scenarios: dict, t: dict, lang: str) -> str:
    cards = []
    for key, color, border in [("bull","#14532d","#22c55e"),
                                ("base","#451a03","#f59e0b"),
                                ("bear","#450a0a","#ef4444")]:
        sc = scenarios.get(key, {})
        if not sc:
            continue
        prob = sc.get("probability", 0)
        catalyst = sc.get("catalyst", "")
        narrative = sc.get("narrative", "")
        spx_target = sc.get("spx_target", "")
        horizon = sc.get("horizon_weeks", "")

        label_map = {"bull":"BULL","base":"BASE","bear":"BEAR"}
        cards.append(f"""
        <div style="background:{color};border:1px solid {border};border-radius:8px;
                    padding:12px;flex:1;min-width:180px">
          <div style="font-size:9px;font-weight:800;letter-spacing:0.1em;color:{border};margin-bottom:4px">{label_map[key]}</div>
          <div style="font-size:24px;font-weight:800;color:{border};margin-bottom:6px">{prob:.0%}</div>
          <div style="font-size:11px;font-weight:600;color:#e5e7eb;margin-bottom:6px;line-height:1.35">{catalyst}</div>
          <div style="font-size:10px;color:#9ca3af;line-height:1.5;margin-bottom:8px">{narrative}</div>
          <div style="font-size:10px;color:#6b7280">
            {t["target"]}: <strong style="color:{border}">{spx_target}</strong>
            &nbsp;·&nbsp; {t["horizon"]}: {horizon}w
          </div>
        </div>""")

    if not cards:
        return ""
    return f'<div style="display:flex;gap:10px;flex-wrap:wrap">{"".join(cards)}</div>'


def _build_editorial_html(curation_path: str | None, lang: str) -> str:
    """Extrai texto editorial do .docx e formata para PDF."""
    if not curation_path or not Path(curation_path).exists():
        return ""
    try:
        c = json.loads(Path(curation_path).read_text(encoding="utf-8"))
        written = c.get("artifact_paths", {}).get("written", "")
        if not written or not Path(written).exists():
            return ""
        from docx import Document
        doc = Document(written)
    except Exception:
        return ""

    _SEP = re.compile(r"^[━—\-=]{4,}")
    current_key = None
    items = []

    for p in doc.paragraphs:
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
                current_key = m.group(1).strip()
                continue
            if current_key != "TEXTO PRINCIPAL":
                continue
            if _SEP.match(text):
                continue
            if "Heading" in style:
                items.append(("heading", text))
            elif is_italic or (len(text) < 140 and ":" in text[:60] and not text.endswith(".")):
                items.append(("callout", text))
            else:
                items.append(("body", text))

    if not items:
        return ""

    parts = []
    for kind, text in items:
        if kind == "heading":
            parts.append(f'<h3 style="font-size:15px;font-weight:800;color:{_NAVY};margin:16px 0 8px;line-height:1.25">{text}</h3>')
        elif kind == "callout":
            parts.append(
                f'<div style="border-left:3px solid {_GOLD};background:#fffbeb;'
                f'border-radius:0 4px 4px 0;padding:8px 12px;margin:12px 0;'
                f'font-size:11px;color:#92400e;font-style:italic;line-height:1.6">{text}</div>'
            )
        else:
            parts.append(f'<p style="font-size:12px;line-height:1.8;color:#374151;margin:0 0 10px">{text}</p>')

    return "".join(parts)


def _build_polymarket(markets: list[dict], t: dict) -> str:
    if not markets:
        return ""
    rows = []
    for m in markets[:8]:
        q = m.get("question", "")[:80]
        prob = m.get("probability")
        vol = m.get("volume")
        prob_str = f"{prob:.0%}" if prob is not None else "—"
        prob_color = _GREEN if (prob or 0) > 0.6 else (_RED if (prob or 0) < 0.3 else _YELLOW)
        vol_str = f"${vol:,.0f}" if vol else "—"
        rows.append(f"""
        <tr>
          <td style="font-size:11px">{q}</td>
          <td style="text-align:right;font-weight:700;color:{prob_color}">{prob_str}</td>
          <td style="text-align:right;color:#9ca3af;font-size:10px">{vol_str}</td>
        </tr>""")
    return f"""
    <table class="data-table">
      <thead><tr><th>{t["question"]}</th><th style="text-align:right">{t["prob"]}</th><th style="text-align:right">{t["volume"]}</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def _build_risk(risk_data: dict, t: dict) -> str:
    rows = []
    for ticker, rd in risk_data.items():
        var95 = rd.get("var_95")
        cvar95 = rd.get("cvar_95")
        maxdd = rd.get("max_drawdown")
        sharpe = rd.get("sharpe_ratio")
        rows.append(f"""
        <tr>
          <td style="font-weight:600;font-size:11px;color:{_MUTED}">{ticker}</td>
          <td style="text-align:right;color:{_RED};font-size:11px">{_pct(var95) if var95 else '—'}</td>
          <td style="text-align:right;color:{_RED};font-size:11px">{_pct(cvar95) if cvar95 else '—'}</td>
          <td style="text-align:right;color:{_RED};font-size:11px">{_pct(maxdd) if maxdd else '—'}</td>
          <td style="text-align:right;font-size:11px">{f"{sharpe:.2f}" if sharpe else "—"}</td>
        </tr>""")
    if not rows:
        return ""
    return f"""
    <table class="data-table">
      <thead><tr>
        <th>{t["ticker"]}</th>
        <th style="text-align:right">{t["var95"]}</th>
        <th style="text-align:right">{t["cvar95"]}</th>
        <th style="text-align:right">{t["maxdd"]}</th>
        <th style="text-align:right">{t["sharpe"]}</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def _build_monte_carlo(mc_data: dict, t: dict) -> str:
    rows = []
    for ticker, md in mc_data.items():
        prob_up = md.get("prob_up")
        prob_p5 = md.get("prob_plus_5pct")
        prob_m5 = md.get("prob_minus_5pct")
        p50_raw = md.get("p50")
        p50 = p50_raw[-1] if isinstance(p50_raw, list) and p50_raw else (p50_raw if isinstance(p50_raw, (int, float)) else None)
        current = md.get("current_price")
        rows.append(f"""
        <tr>
          <td style="font-weight:600;font-size:11px;color:{_MUTED}">{ticker}</td>
          <td style="text-align:right;font-size:11px">{_fmt_price(current)}</td>
          <td style="text-align:right;color:{_color_pct(prob_up)};font-size:11px">{_pct(prob_up) if prob_up else '—'}</td>
          <td style="text-align:right;color:{_GREEN};font-size:11px">{_pct(prob_p5) if prob_p5 else '—'}</td>
          <td style="text-align:right;color:{_RED};font-size:11px">{_pct(prob_m5) if prob_m5 else '—'}</td>
          <td style="text-align:right;font-size:11px">{_fmt_price(p50)}</td>
        </tr>""")
    if not rows:
        return ""
    return f"""
    <table class="data-table">
      <thead><tr>
        <th>{t["ticker"]}</th>
        <th style="text-align:right">{t["price"]}</th>
        <th style="text-align:right">{t["prob_up"]}</th>
        <th style="text-align:right">{t["prob_p5"]}</th>
        <th style="text-align:right">{t["prob_m5"]}</th>
        <th style="text-align:right">{t["p50"]}</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def _build_gossip(bundle: DailyIngestionBundle, out_dir: Path) -> str:
    """ZeroHedge Market Gossip — imagens + título + snippet."""
    cards = []
    seen: set[str] = set()
    for block in bundle.market_ear_blocks:
        unique = list(dict.fromkeys(block.image_refs))
        local = [p for p in unique if Path(p).exists() and not p.startswith("http")]
        if not local:
            continue
        img_path = local[0]
        p = Path(img_path)
        # embed as base64 for self-contained PDF
        try:
            b64 = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
            if b64 in seen:
                continue
            seen.add(b64)
        except Exception:
            continue
        title = block.title or ""
        snippet = block.body_text[:200].strip() if block.body_text else ""
        if len(block.body_text) > 200:
            snippet += "…"
        cards.append(f"""
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:6px;
                    overflow:hidden;break-inside:avoid;margin-bottom:10px">
          {"<div style='font-size:11px;font-weight:700;padding:8px 10px;border-bottom:1px solid #f0f4f8;color:" + _NAVY + "'>" + title + "</div>" if title else ""}
          <img src="{b64}" style="width:100%;max-height:200px;object-fit:cover;display:block">
          {"<p style='font-size:10px;color:#6b7280;padding:8px 10px;margin:0;line-height:1.5'>" + snippet + "</p>" if snippet else ""}
        </div>""")

    if not cards:
        return ""
    return "".join(cards)


# ── CSS ───────────────────────────────────────────────────────────────────────
_PDF_CSS = f"""
@page {{
  size: A4;
  margin: 0;
}}

* {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  font-family: 'Segoe UI', 'Arial', sans-serif;
  font-size: 12px;
  color: {_NAVY};
  background: white;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}}

.cover {{
  width: 210mm;
  height: 297mm;
  background: {_NAVY};
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  page-break-after: always;
  position: relative;
  overflow: hidden;
}}

.cover-accent {{
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 6px;
  background: {_GOLD};
}}

.cover-top-bar {{
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: {_GOLD};
}}

.cover-logo {{
  width: 90px;
  height: 90px;
  object-fit: contain;
  margin-bottom: 32px;
  filter: brightness(1.1);
}}

.cover-title {{
  font-size: 32px;
  font-weight: 800;
  color: {_WHITE};
  letter-spacing: -0.02em;
  margin-bottom: 6px;
  text-align: center;
}}

.cover-subtitle {{
  font-size: 14px;
  color: {_GOLD};
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 48px;
  text-align: center;
}}

.cover-date-box {{
  border: 1px solid rgba(208,160,68,0.4);
  border-radius: 8px;
  padding: 16px 32px;
  text-align: center;
  margin-bottom: 40px;
}}

.cover-date {{
  font-size: 18px;
  font-weight: 700;
  color: {_WHITE};
  margin-bottom: 4px;
}}

.cover-date-label {{
  font-size: 10px;
  color: {_GOLD};
  text-transform: uppercase;
  letter-spacing: 0.1em;
}}

.cover-firm {{
  position: absolute;
  bottom: 24px;
  font-size: 11px;
  color: rgba(255,255,255,0.4);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}}

.content {{
  padding: 20mm 18mm 16mm;
}}

.page-break {{ page-break-before: always; }}

.section {{
  margin-bottom: 20px;
  break-inside: avoid;
}}

.section-header {{
  background: {_NAVY};
  color: {_WHITE};
  padding: 7px 12px;
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  border-radius: 4px 4px 0 0;
  display: flex;
  align-items: center;
  gap: 6px;
}}

.section-header .gold-bar {{
  width: 3px;
  height: 12px;
  background: {_GOLD};
  border-radius: 2px;
  display: inline-block;
}}

.section-body {{
  border: 1px solid #e2e8f0;
  border-top: none;
  border-radius: 0 0 4px 4px;
  padding: 12px;
  background: white;
}}

.data-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
}}

.data-table thead th {{
  font-size: 9px;
  font-weight: 700;
  color: {_MUTED};
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 5px 8px;
  text-align: left;
  border-bottom: 1px solid #e2e8f0;
  background: #f8fafc;
}}

.data-table tbody tr {{ border-bottom: 1px solid #f0f4f8; }}
.data-table tbody tr:last-child {{ border-bottom: none; }}
.data-table tbody td {{ padding: 6px 8px; vertical-align: middle; }}

.narrative-box {{
  background: linear-gradient(135deg, {_NAVY} 0%, {_NAVY2} 100%);
  border-radius: 6px;
  padding: 14px 18px;
  margin-bottom: 16px;
  color: white;
}}

.narrative-label {{
  font-size: 16px;
  font-weight: 800;
  color: {_WHITE};
  margin-bottom: 8px;
  line-height: 1.25;
}}

.narrative-meta {{
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}}

.meta-chip {{
  font-size: 10px;
  color: {_GOLD};
  font-weight: 600;
}}

.meta-chip span {{
  color: rgba(255,255,255,0.8);
  font-weight: 400;
  margin-left: 4px;
}}

.grid-2 {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}}

.footer-strip {{
  position: fixed;
  bottom: 0;
  left: 0; right: 0;
  height: 28px;
  background: {_NAVY};
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 18mm;
  font-size: 8px;
  color: rgba(255,255,255,0.4);
}}

.footer-strip .gold-dot {{
  width: 4px;
  height: 4px;
  background: {_GOLD};
  border-radius: 50%;
  display: inline-block;
  margin: 0 6px;
}}

.disclaimer {{
  margin-top: 24px;
  padding: 10px 14px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-left: 3px solid {_GOLD};
  border-radius: 0 4px 4px 0;
  font-size: 9px;
  color: {_MUTED};
  line-height: 1.6;
}}

.gossip-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}}
"""


# ── HTML template ─────────────────────────────────────────────────────────────

def _build_html(
    bundle: DailyIngestionBundle,
    curation_path: str | None,
    lang: str,
    out_dir: Path,
) -> str:
    t = _T[lang]
    market_prices = bundle.market_prices or {}
    fred_data = bundle.fred_data or {}

    # Curation data
    primary_label = "—"
    secondary_label = ""
    primary_conf = "—"
    scored = "—"
    verdict = "—"
    verdict_color = _MUTED
    verdict_icon = "?"
    mode = "week_ahead"

    enrichment_data: dict = {}
    enrichment_ap: dict = {}

    if curation_path and Path(curation_path).exists():
        try:
            c = json.loads(Path(curation_path).read_text(encoding="utf-8"))
            narr = c.get("narrative", {})
            ps = narr.get("primary_signal", {})
            ss = narr.get("secondary_signals", [])
            primary_label = ps.get("label", "—")
            primary_conf = f"{ps.get('confidence', 0):.0%}"
            scored = str(len(c.get("scored_items", [])))
            verdict = c.get("verification", {}).get("overall_verdict", "—")
            verdict_color = {
                "pass": _GREEN, "warn": _YELLOW, "fail": _RED
            }.get(verdict, _MUTED)
            verdict_icon = {"pass": "✓", "warn": "⚠", "fail": "✕"}.get(verdict, "?")
            if ss:
                s0 = ss[0]
                secondary_label = f"↳ {s0.get('label', '')} ({s0.get('confidence', 0):.0%})"
            ap = c.get("artifact_paths", {})
            mode = ap.get("written_mode", "week_ahead")
            enrichment_ap = ap
            for key in ("scenarios", "monte_carlo", "risk"):
                p = enrichment_ap.get(key)
                if p and Path(p).exists():
                    enrichment_data[key] = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception as exc:
            _log.warning("pdf_curation_parse_error", error=str(exc))

    # Date formatting
    run_date = bundle.run_date
    try:
        if lang == "en":
            run_date_full = datetime.strptime(str(run_date), "%Y-%m-%d").strftime("%B %d, %Y")
        else:
            run_date_full = datetime.strptime(str(run_date), "%Y-%m-%d").strftime("%d de %B de %Y")
    except Exception:
        run_date_full = str(run_date)

    logo_b64 = _get_logo_b64()
    logo_img = f'<img src="{logo_b64}" class="cover-logo">' if logo_b64 else (
        f'<div style="width:90px;height:90px;border:2px solid {_GOLD};border-radius:50%;'
        f'display:flex;align-items:center;justify-content:center;margin-bottom:32px;'
        f'font-size:28px;font-weight:900;color:{_GOLD}">G</div>'
    )

    # ── Sections ──────────────────────────────────────────────────────────────
    market_table = _build_market_table(market_prices, t)
    heatmap = _build_heatmap(market_prices)
    calendar_html = _build_calendar(fred_data.get("calendar", []), t, lang)
    fred_html = _build_fred(fred_data.get("series", {}), t, lang)
    scenarios_html = _build_scenarios(enrichment_data.get("scenarios", {}), t, lang) if enrichment_data.get("scenarios") else ""
    polymarket_html = _build_polymarket(bundle.polymarket_markets or [], t) if bundle.polymarket_markets else ""
    risk_html = _build_risk(enrichment_data.get("risk", {}), t) if enrichment_data.get("risk") else ""
    mc_html = _build_monte_carlo(enrichment_data.get("monte_carlo", {}), t) if enrichment_data.get("monte_carlo") else ""
    editorial_html = _build_editorial_html(curation_path, lang)
    gossip_html = _build_gossip(bundle, out_dir)

    # ── Assemble ──────────────────────────────────────────────────────────────
    generated_at = datetime.now().strftime("%d/%m/%Y %H:%M")

    content_sections = f"""
    <!-- Narrative -->
    <div class="narrative-box">
      <div class="narrative-label">{primary_label}</div>
      {"<div style='font-size:11px;color:rgba(255,255,255,0.5);margin-bottom:8px'>" + secondary_label + "</div>" if secondary_label else ""}
      <div class="narrative-meta">
        <div class="meta-chip">{t["confidence"]}: <span>{primary_conf}</span></div>
        <div class="meta-chip">{t["verification"]}: <span style="color:{verdict_color}">{verdict_icon} {verdict}</span></div>
        <div class="meta-chip">{t["scored"]}: <span>{scored}</span></div>
        <div class="meta-chip">{t["mode"]}: <span>{mode}</span></div>
      </div>
    </div>

    <!-- Market Heatmap -->
    <div class="section">
      <div class="section-header"><span class="gold-bar"></span>{t["market_snapshot"]}</div>
      <div class="section-body">{heatmap}</div>
    </div>

    <!-- Calendar + Market grid -->
    <div class="grid-2" style="margin-bottom:20px">
      <div class="section">
        <div class="section-header"><span class="gold-bar"></span>📅 {t["calendar"]}</div>
        <div class="section-body" style="padding:0">{calendar_html}</div>
      </div>
      <div class="section">
        <div class="section-header"><span class="gold-bar"></span>📊 {t["market_snapshot"]}</div>
        <div class="section-body" style="padding:0">{market_table}</div>
      </div>
    </div>

    <!-- FRED Dashboard — new page -->
    <div class="page-break"></div>

    <div class="section">
      <div class="section-header"><span class="gold-bar"></span>🏛️ {t["fred_title"]}</div>
      <div class="section-body">{fred_html}</div>
    </div>

    {"<!-- Scenarios --><div class='section'><div class='section-header'><span class='gold-bar'></span>📊 " + t["scenarios"] + "</div><div class='section-body'>" + scenarios_html + "</div></div>" if scenarios_html else ""}

    {"<div class='grid-2' style='margin-bottom:20px'>" + ("<div class='section'><div class='section-header'><span class='gold-bar'></span>🔮 " + t["poly_title"] + "</div><div class='section-body' style='padding:0'>" + polymarket_html + "</div></div>" if polymarket_html else "<div></div>") + ("<div class='section'><div class='section-header'><span class='gold-bar'></span>⚠️ " + t["risk_title"] + "</div><div class='section-body' style='padding:0'>" + risk_html + "</div></div>" if risk_html else "<div></div>") + "</div>" if (polymarket_html or risk_html) else ""}

    {"<!-- Monte Carlo --><div class='section'><div class='section-header'><span class='gold-bar'></span>🎲 " + t["mc_title"] + "</div><div class='section-body' style='padding:0'>" + mc_html + "</div></div>" if mc_html else ""}

    <!-- Editorial — new page -->
    {"<div class='page-break'></div><div class='section'><div class='section-header'><span class='gold-bar'></span>✍️ " + t["editorial"] + "</div><div class='section-body'>" + editorial_html + "</div></div>" if editorial_html else ""}

    <!-- Market Gossip -->
    {"<div class='page-break'></div><div style='font-size:10px;font-weight:700;color:" + _GOLD + ";text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;padding-bottom:6px;border-bottom:2px solid " + _GOLD + "'>📰 " + t["market_gossip"] + "</div><div class='gossip-grid'>" + gossip_html + "</div>" if gossip_html else ""}

    <div class="disclaimer">{t["disclaimer"]}</div>
    """

    return f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{t["report_title"]} — {run_date}</title>
<style>{_PDF_CSS}</style>
</head>
<body>

<!-- COVER PAGE -->
<div class="cover">
  <div class="cover-top-bar"></div>
  {logo_img}
  <div class="cover-title">{t["report_title"]}</div>
  <div class="cover-subtitle">{t["report_subtitle"]} · Gulfstream Capital</div>
  <div class="cover-date-box">
    <div class="cover-date">{run_date_full}</div>
    <div class="cover-date-label">{t["generated"]} {generated_at}</div>
  </div>
  <div class="cover-firm">Gulfstream Capital · {t["confidential"]}</div>
  <div class="cover-accent"></div>
</div>

<!-- CONTENT -->
<div class="content">
  {content_sections}
</div>

<!-- FOOTER (fixed on each page) -->
<div class="footer-strip">
  <span>Gulfstream Capital</span>
  <span><span class="gold-dot"></span>{t["report_title"]} · {run_date}<span class="gold-dot"></span></span>
  <span>{t["confidential"]}</span>
</div>

</body>
</html>"""


# ── Main export function ───────────────────────────────────────────────────────

def save_week_ahead_pdf(
    bundle: DailyIngestionBundle,
    curation_path: str | None = None,
    lang: str = "pt",
) -> Path:
    """
    Converte o Week Ahead Brief HTML em PDF via Playwright (= Chrome Print to PDF).

    Usa o HTML gerado por save_week_ahead_brief() diretamente — sem redesign,
    sem template separado. O PDF é fiel ao que aparece no browser.

    Args:
        bundle: DailyIngestionBundle do dia
        curation_path: JSON de curação (opcional)
        lang: ignorado por ora (futuro: EN translation)

    Returns:
        Path do PDF gerado
    """
    from playwright.sync_api import sync_playwright
    from app.views.week_ahead_brief import save_week_ahead_brief

    run_date = bundle.run_date
    out_dir = workspace.bundles / str(run_date)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log.info("pdf_generation_start", lang=lang, run_id=bundle.run_id)

    # 1. Garante que o HTML mais recente existe
    html_path = save_week_ahead_brief(bundle, curation_path)

    # 2. Playwright imprime o HTML exatamente como o browser renderiza
    pdf_path = out_dir / f"{bundle.run_id}_week_ahead_brief_{lang}.pdf"

    logo_b64 = _get_logo_b64()
    logo_img = f'<img src="{logo_b64}" style="height:28px;vertical-align:middle;margin-right:8px">' if logo_b64 else ""

    footer_html = f"""
    <div style="width:100%;display:flex;align-items:center;justify-content:space-between;
                padding:0 20px;font-family:Arial,sans-serif;font-size:8px;
                color:#8a9bb0;border-top:1px solid #21262d;height:100%">
      <span>{logo_img}Gulfstream Capital</span>
      <span>Week Ahead Brief · {run_date}</span>
      <span style="color:#D0A044"><span class="pageNumber"></span> / <span class="totalPages"></span></span>
    </div>"""

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(html_path.as_uri(), wait_until="networkidle", timeout=60000)
        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            margin={"top": "10mm", "bottom": "12mm", "left": "10mm", "right": "10mm"},
            display_header_footer=True,
            header_template="<div></div>",
            footer_template=footer_html,
            scale=0.78,  # escala para caber o layout 1440px em A4
        )
        browser.close()

    _log.info("pdf_saved", path=str(pdf_path), size_kb=pdf_path.stat().st_size // 1024)
    return pdf_path
