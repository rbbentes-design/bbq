"""FlowPatrol Charts — visualizacoes do SpotGamma FlowPatrol parsed."""

from __future__ import annotations

from app.analysis.flow_patrol_parser import FlowPatrolParsed, FlowPatrolTrade


_COLORS = {
    "bullish":  "#22c55e", "bearish":  "#ef4444", "neutral":  "#94a3b8",
    "bg":       "#0a1628", "border":   "#1e293b",
    "label":    "#94a3b8", "value":    "#e2e8f0",
    "accent":   "#818cf8", "sell":     "#f87171", "buy":      "#4ade80",
}


def _classify_trade(t: FlowPatrolTrade) -> str:
    if t.dominant_action in ("BTO", "BTC"):
        return "bullish" if t.option_type == "C" else "bearish"
    elif t.dominant_action in ("STO", "STC"):
        return "bearish" if t.option_type == "C" else "bullish"
    return "neutral"


def _fmt_number(n: int) -> str:
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def _fmt_usd_m(v: float) -> str:
    if abs(v) >= 1000:
        return f"${v/1000:+.1f}B"
    return f"${v:+.0f}M"


def render_top_trades_table(trades: list[FlowPatrolTrade], title: str, limit: int = 8) -> str:
    if not trades:
        return ""
    sorted_trades = sorted(trades, key=lambda t: t.total_volume, reverse=True)[:limit]
    rows_html = []
    for t in sorted_trades:
        sentiment = _classify_trade(t)
        color = _COLORS[sentiment]
        type_color = _COLORS["bullish"] if t.option_type == "C" else _COLORS["bearish"]
        dom_color = {"BTO": _COLORS["buy"], "BTC": _COLORS["accent"],
                     "STO": _COLORS["sell"], "STC": _COLORS["neutral"]}.get(t.dominant_action, _COLORS["neutral"])
        rows_html.append(
            f'<tr>'
            f'<td style="color:{_COLORS["value"]};font-weight:600">{t.ticker}</td>'
            f'<td style="text-align:right;color:{_COLORS["value"]}">${t.strike:.0f}</td>'
            f'<td style="text-align:center;color:{type_color};font-weight:700">{t.option_type}</td>'
            f'<td style="color:{_COLORS["label"]};font-size:10px">{t.expiry}</td>'
            f'<td style="text-align:right;color:{_COLORS["label"]};font-size:10px">{t.iv_pct:.1f}%</td>'
            f'<td style="text-align:center;color:{dom_color};font-weight:700;font-size:10px">{t.dominant_action}</td>'
            f'<td style="text-align:right;color:{color};font-weight:600">{_fmt_number(t.total_volume)}</td>'
            f'</tr>'
        )
    return f'''
<div style="background:{_COLORS["bg"]};border:1px solid {_COLORS["border"]};border-radius:8px;padding:12px 14px">
  <div style="font-size:11px;font-weight:700;color:{_COLORS["label"]};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">
    {title} <span style="color:{_COLORS["accent"]}">({len(trades)} trades)</span>
  </div>
  <table style="width:100%;font-size:11px;border-collapse:collapse">
    <thead>
      <tr style="color:{_COLORS["label"]};font-size:10px;border-bottom:1px solid {_COLORS["border"]}">
        <td style="padding:4px 0">Ticker</td><td style="text-align:right">Strike</td>
        <td style="text-align:center">Type</td><td>Exp</td>
        <td style="text-align:right">IV</td><td style="text-align:center">Action</td>
        <td style="text-align:right">Vol</td>
      </tr>
    </thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
</div>'''


def render_sector_flows(sectors: list, max_items: int = 8) -> str:
    if not sectors:
        return ""
    sorted_secs = sorted(sectors, key=lambda s: abs(s.delta_usd_m), reverse=True)[:max_items]
    max_abs = max((abs(s.delta_usd_m) for s in sorted_secs), default=1)
    rows_html = []
    for s in sorted_secs:
        is_bull = s.delta_usd_m > 0
        color = _COLORS["bullish"] if is_bull else _COLORS["bearish"]
        width = abs(s.delta_usd_m) / max_abs * 100
        if is_bull:
            bar_html = (f'<div style="display:flex;align-items:center;height:18px">'
                        f'<div style="flex:1"></div><div style="width:{width:.0f}%;max-width:50%;height:12px;'
                        f'background:{color};border-radius:2px 0 0 2px"></div></div>')
        else:
            bar_html = (f'<div style="display:flex;align-items:center;height:18px">'
                        f'<div style="width:{width:.0f}%;max-width:50%;height:12px;'
                        f'background:{color};border-radius:0 2px 2px 0;margin-left:auto"></div>'
                        f'<div style="flex:1"></div></div>')
        rows_html.append(
            f'<tr>'
            f'<td style="color:{_COLORS["value"]};font-size:11px;padding:3px 6px 3px 0">{s.sector}</td>'
            f'<td style="width:60%">{bar_html}</td>'
            f'<td style="text-align:right;color:{color};font-weight:600;font-size:11px;padding:0 0 0 6px">{_fmt_usd_m(s.delta_usd_m)}</td>'
            f'<td style="color:{_COLORS["label"]};font-size:10px;padding:0 0 0 6px">{s.percentile:.0f}%ile</td>'
            f'</tr>'
        )
    return f'''
<div style="background:{_COLORS["bg"]};border:1px solid {_COLORS["border"]};border-radius:8px;padding:12px 14px;margin-top:12px">
  <div style="font-size:11px;font-weight:700;color:{_COLORS["label"]};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">
    Sector Delta Flow
  </div>
  <table style="width:100%;border-collapse:collapse">{"".join(rows_html)}</table>
</div>'''


def render_highlights(highlights: list, limit: int = 6) -> str:
    if not highlights:
        return ""
    order = {"extreme_bearish": 0, "extreme_bullish": 1, "bearish": 2, "bullish": 3}
    sorted_h = sorted(highlights, key=lambda h: (order.get(h.sentiment, 9), -abs(h.value_usd_m)))[:limit]
    cards = []
    for h in sorted_h:
        is_bull = "bullish" in h.sentiment
        color = _COLORS["bullish"] if is_bull else _COLORS["bearish"]
        is_extreme = "extreme" in h.sentiment
        badge = "EXTREME" if is_extreme else h.sentiment.upper()
        cards.append(
            f'<div style="background:{_COLORS["bg"]};border:1px solid {color}40;'
            f'border-left:3px solid {color};border-radius:6px;padding:10px 12px;'
            f'display:flex;flex-direction:column;gap:4px;min-width:0">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;gap:8px">'
            f'<span style="color:{_COLORS["value"]};font-weight:700;font-size:13px">{h.ticker}</span>'
            f'<span style="color:{color};font-size:9px;font-weight:700;letter-spacing:0.05em">{badge}</span>'
            f'</div>'
            f'<div style="color:{color};font-weight:600;font-size:14px">{_fmt_usd_m(h.value_usd_m)}</div>'
            f'<div style="color:{_COLORS["label"]};font-size:10px">{h.metric} · {h.percentile:.0f}%ile</div>'
            f'</div>'
        )
    return f'''
<div style="margin-bottom:12px">
  <div style="font-size:11px;font-weight:700;color:{_COLORS["label"]};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">
    Executive Flow Highlights
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px">
    {"".join(cards)}
  </div>
</div>'''


def render_flow_patrol_panel(parsed: FlowPatrolParsed) -> str:
    if not parsed or (not parsed.trades_index_etfs and
                      not parsed.trades_single_stocks and
                      not parsed.highlights):
        return ""
    parts = [f'''
<div style="margin-bottom:20px">
  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:12px;
              padding-bottom:8px;border-bottom:2px solid {_COLORS["accent"]}">
    <h3 style="color:{_COLORS["value"]};font-size:16px;font-weight:700;margin:0">
      🛰 SpotGamma FlowPatrol
    </h3>
    <span style="color:{_COLORS["label"]};font-size:11px">
      Report date: {parsed.report_date or "—"} · {len(parsed.all_trades)} trades parsed
    </span>
  </div>
''']
    if parsed.highlights:
        parts.append(render_highlights(parsed.highlights, limit=8))
    if parsed.sectors:
        parts.append(render_sector_flows(parsed.sectors))
    tables_row = []
    if parsed.trades_index_etfs:
        tables_row.append(render_top_trades_table(parsed.trades_index_etfs, "Index ETFs — Largest Trades", limit=10))
    if parsed.trades_single_stocks:
        tables_row.append(render_top_trades_table(parsed.trades_single_stocks, "Single Stocks — Largest Trades", limit=10))
    if parsed.trades_asset_etfs:
        tables_row.append(render_top_trades_table(parsed.trades_asset_etfs, "Asset ETFs — Largest Trades", limit=10))
    if tables_row:
        parts.append(
            f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));'
            f'gap:12px;margin-top:12px">{"".join(tables_row)}</div>'
        )
    parts.append('</div>')
    return "\n".join(parts)
