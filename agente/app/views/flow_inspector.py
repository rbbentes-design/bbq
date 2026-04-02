"""
Flow Inspector — Portfolio Allocation HTML View

Janela interativa com:
  - Portfolio allocation table + breakdown por regime
  - Scatter plot: risco × retorno esperado (tamanho = alocação)
  - Signal heatmap: momentum / mean_rev / vol_edge / options_flow por ativo
  - Network colorido por alpha signal (verde = long, vermelho = short)
  - Watchlist de oportunidades próximas do threshold

Auto-refresh via JavaScript setTimeout (funciona em file://).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("views.flow_inspector")

_REGIME_COLORS = {
    "bull":    "#10b981",   # green
    "neutral": "#f59e0b",   # amber
    "bear":    "#ef4444",   # red
}

_DIRECTION_COLORS = {
    "long":    "#10b981",
    "short":   "#ef4444",
    "neutral": "#6b7280",
}


def _pct(v: float) -> str:
    return f"{v:+.1%}" if v is not None else "—"

def _pct0(v: float) -> str:
    return f"{v:.1%}" if v is not None else "—"

def _usd(v: float) -> str:
    return f"${v:,.0f}" if v is not None else "—"

def _fmt(v: float | None, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}" if v is not None else "—"


def generate_flow_inspector_html(
    portfolio,           # PortfolioResult
    signals: dict,       # dict[str, AssetSignal]
    bundle_date: str = "",
    live_mode: bool = False,
    refresh_interval: int = 62,
    options_strategy=None,   # OptionsStrategyResult | None
    vol_regime=None,         # VolRegimeResult | None
    pairs_result=None,       # PairsResult | None
    rrg_result=None,         # RRGResult | None
    market_prices: dict | None = None,  # {ticker: {price, ...}}
    tv_map: dict | None = None,         # {ticker: TV snapshot} from tradingview provider
) -> str:
    from app.analysis.portfolio_optimizer import PortfolioResult, PositionResult
    from app.analysis.alpha_signals import AssetSignal

    por: PortfolioResult = portfolio
    regime_color = _REGIME_COLORS.get(por.regime_mode, "#6b7280")
    _tv = tv_map or {}

    def _tv_chart_badge(ticker: str, tv: dict | None) -> str:
        """Render a compact Chart Structure badge for a position row."""
        if not tv:
            return ""
        snap = tv.get(ticker)
        if not snap:
            return ""
        regime = snap.get("regime", "neutral")
        regime_dot = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(regime, "⚪")
        quality = snap.get("setup_quality", "")
        quality_color = {"strong": "#10b981", "moderate": "#f59e0b", "weak": "#6b7280"}.get(quality, "#4b5563")
        rsi = snap.get("rsi")
        vwap = snap.get("vwap")
        val  = snap.get("val")
        vah  = snap.get("vah")
        poc  = snap.get("poc")
        pvw  = snap.get("price_vs_vwap", 0)
        items: list[str] = [f'{regime_dot} <b style="color:{quality_color}">{quality}</b>']
        if rsi is not None:
            rsi_c = "#10b981" if rsi > 55 else ("#ef4444" if rsi < 45 else "#f59e0b")
            items.append(f'RSI <span style="color:{rsi_c}">{rsi:.0f}</span>')
        if vwap:
            sign = "+" if pvw >= 0 else ""
            pvw_c = "#10b981" if pvw >= 0 else "#ef4444"
            items.append(f'VWAP <span style="color:{pvw_c}">{sign}{pvw*100:.1f}%</span>')
        if val and vah:
            items.append(f'VA ${val:.2f}–${vah:.2f}')
        elif poc:
            items.append(f'POC ${poc:.2f}')
        return (
            f'<div style="margin-top:3px;font-size:9.5px;color:#6b7280;'
            f'background:#1f293744;padding:2px 5px;border-radius:3px;'
            f'border-left:2px solid {quality_color}">'
            + " &nbsp;·&nbsp; ".join(items)
            + "</div>"
        )

    # ── Positions table rows ────────────────────────────────────────────────
    def direction_badge(d: str) -> str:
        c = _DIRECTION_COLORS.get(d, "#6b7280")
        label = {"long": "LONG ▲", "short": "SHORT ▼", "neutral": "—"}.get(d, d)
        return f'<span style="color:{c};font-weight:700;font-size:11px">{label}</span>'

    def conviction_badge(c: str) -> str:
        colors = {"high": "#10b981", "medium": "#f59e0b", "low": "#6b7280"}
        return f'<span style="background:{colors.get(c,"#6b7280")}22;color:{colors.get(c,"#6b7280")};padding:1px 6px;border-radius:4px;font-size:11px;font-weight:600">{c.upper()}</span>'

    def signal_bar(v: float, width: int = 60) -> str:
        """Mini horizontal bar for signal strength [-1, 1]."""
        pct = int((v + 1) / 2 * 100)  # map [-1,1] → [0,100]
        color = "#10b981" if v >= 0 else "#ef4444"
        mid = width // 2
        bar_w = int(abs(v) * mid)
        if v >= 0:
            bar_html = f'<div style="position:absolute;left:{mid}px;width:{bar_w}px;height:10px;background:{color};top:2px;border-radius:2px"></div>'
        else:
            bar_html = f'<div style="position:absolute;left:{mid-bar_w}px;width:{bar_w}px;height:10px;background:{color};top:2px;border-radius:2px"></div>'
        return (
            f'<div style="position:relative;width:{width}px;height:14px;background:#1f2937;'
            f'border-radius:4px;display:inline-block">'
            f'<div style="position:absolute;left:{mid}px;width:1px;height:14px;background:#4b5563"></div>'
            f'{bar_html}</div>'
        )

    # ── Tracking: P&L atual + trade history ─────────────────────────────────
    pnl_summary = None
    try:
        from app.analysis.portfolio_tracker import get_pnl_summary
        pnl_summary = get_pnl_summary()
    except Exception:
        pass

    # Trade history for Historico tab
    trade_history_rows = ""
    try:
        from app.analysis.portfolio_tracker import get_trade_history
        trade_history = get_trade_history()[:50]
        for t in trade_history:
            dc = "#10b981" if t.get("direction") == "long" else "#ef4444"
            status = t.get("status", "open")
            pnl_r = t.get("pnl_realized")
            pnl_u = t.get("pnl_unrealized")
            pnl_v = pnl_r if status == "closed" else pnl_u
            pnl_pct_v = t.get("pnl_pct", 0) or 0
            pnl_color_t = "#10b981" if (pnl_v or 0) >= 0 else "#ef4444"
            status_badge = (
                '<span style="background:#10b98133;color:#10b981;font-size:10px;padding:1px 5px;border-radius:3px">OPEN</span>'
                if status == "open" else
                '<span style="background:#6b728033;color:#9ca3af;font-size:10px;padding:1px 5px;border-radius:3px">CLOSED</span>'
            )
            why_o = t.get("why_opened", "—")[:80]
            why_c = t.get("why_closed") or "—"
            why_c = why_c[:80] if why_c != "—" else "—"
            exit_info = f"{t.get('exit_date','')}" if status == "closed" else "—"
            trade_history_rows += f"""
            <tr>
              <td style="color:#9ca3af;font-size:10px">{t.get('trade_id','')}</td>
              <td style="font-weight:700;color:#e5e7eb">{t.get('ticker','')}</td>
              <td style="color:{dc};font-size:11px">{t.get('direction','').upper()}</td>
              <td>{status_badge}</td>
              <td style="color:#9ca3af;font-size:11px">{t.get('entry_date','')} {t.get('entry_time','')[:5]}</td>
              <td style="color:#e5e7eb">${t.get('entry_price',0):,.2f}</td>
              <td style="color:#9ca3af;font-size:11px">{exit_info}</td>
              <td style="color:#9ca3af">${t.get('exit_price') or 0:,.2f}</td>
              <td style="color:{pnl_color_t};font-weight:600">${(pnl_v or 0):+,.0f}</td>
              <td style="color:{pnl_color_t}">{pnl_pct_v:+.1%}</td>
              <td style="font-size:10px;color:#6b7280;max-width:180px">{why_o}</td>
              <td style="font-size:10px;color:#6b7280;max-width:180px">{why_c}</td>
            </tr>"""
    except Exception:
        trade_history_rows = '<tr><td colspan="12" style="color:#6b7280;text-align:center;padding:20px">Sem historico de trades</td></tr>'

    pnl_unrealized = pnl_summary.get("total_pnl", 0)   if pnl_summary else 0
    pnl_realized   = pnl_summary.get("realized_pnl", 0) if pnl_summary else 0
    pnl_total      = pnl_unrealized + pnl_realized
    pnl_history    = pnl_summary.get("history", [])    if pnl_summary else []
    pnl_by_ticker  = pnl_summary.get("pnl_by_ticker", {}) if pnl_summary else {}
    pnl_color      = "#10b981" if pnl_total >= 0 else "#ef4444"
    pnl_sign       = "+" if pnl_total >= 0 else ""
    pnl_history_js = json.dumps([h["pnl_pct"] * 100 for h in pnl_history])
    # Equity cumulativa = budget + realized P&L (reflete trades fechados)
    equity = por.budget + pnl_realized

    def pnl_cell(ticker: str) -> str:
        v = pnl_by_ticker.get(ticker)
        if v is None:
            return '<td style="color:#6b7280">—</td>'
        color = "#10b981" if v >= 0 else "#ef4444"
        return f'<td style="color:{color};font-weight:600">${v:+,.0f}</td>'

    # Mapa de last price: market_prices injetado ou pnl_by_ticker para derivar
    _mp = market_prices or {}

    def _last_price(ticker: str, entry: float, direction: str) -> float | None:
        # 1. market_prices direto
        mp_entry = _mp.get(ticker)
        if mp_entry and isinstance(mp_entry, dict):
            p = mp_entry.get("price")
            if p and float(p) > 0:
                return float(p)
        # 2. deriva do P&L
        pnl = pnl_by_ticker.get(ticker)
        if pnl is not None and entry > 0:
            from app.analysis.portfolio_tracker import _load_active
            active = _load_active()
            if active:
                for t in active.get("trades", []):
                    if t["ticker"] == ticker and t["status"] == "open":
                        shares = t.get("shares", 0)
                        if shares and shares > 0:
                            if direction == "long":
                                return entry + pnl / shares
                            else:
                                return entry - pnl / shares
        return None

    pos_rows = ""
    for p in por.positions:
        alloc_color = "#10b981" if p.allocation_pct > 0 else "#ef4444"
        entry_val  = getattr(p, "entry_price", 0) or 0
        stop_val   = getattr(p, "stop_loss", 0) or 0
        entry_str  = f"${entry_val:,.2f}"  if entry_val > 0 else "—"
        stop_str   = f"${stop_val:,.2f}"   if stop_val > 0 else "—"
        target_str = f"${p.take_profit:,.2f}" if getattr(p, "take_profit", 0) > 0 else "—"
        rr_val     = getattr(p, "risk_reward", 0)
        stop_pct_v = getattr(p, "stop_pct", 0)
        rr_str     = f"{rr_val:.1f}:1" if rr_val > 0 else "—"
        rr_color   = "#10b981" if rr_val >= 2.0 else ("#f59e0b" if rr_val >= 1.5 else "#ef4444")
        stop_color = "#ef4444" if p.direction == "long" else "#10b981"
        tgt_color  = "#10b981" if p.direction == "long" else "#ef4444"

        # Last price + stop hit detection
        last = _last_price(p.ticker, entry_val, p.direction)
        if last is not None:
            stop_hit = (
                (p.direction == "long"  and stop_val > 0 and last <= stop_val) or
                (p.direction == "short" and stop_val > 0 and last >= stop_val)
            )
            if stop_hit:
                last_cell = (
                    f'<td style="color:#ef4444;font-weight:700;font-size:12px">'
                    f'${last:,.2f}'
                    f'<div style="font-size:9px;background:#ef444422;color:#ef4444;'
                    f'padding:1px 4px;border-radius:3px;margin-top:2px">⚠ STOP HIT</div></td>'
                )
            else:
                chg = ((last - entry_val) / entry_val) if entry_val > 0 else 0
                if p.direction == "short":
                    chg = -chg
                chg_color = "#10b981" if chg >= 0 else "#ef4444"
                last_cell = (
                    f'<td style="color:#60a5fa;font-size:12px">'
                    f'${last:,.2f}'
                    f'<div style="font-size:10px;color:{chg_color}">{chg:+.1%}</div></td>'
                )
        else:
            last_cell = '<td style="color:#4b5563;font-size:11px">—</td>'

        # Se stop hit, destaca a linha inteira
        row_style = ' style="background:#ef444408;border-left:2px solid #ef4444"' if (
            last is not None and stop_val > 0 and (
                (p.direction == "long"  and last <= stop_val) or
                (p.direction == "short" and last >= stop_val)
            )
        ) else ""

        pos_rows += f"""
        <tr class="pos-row" data-cluster="{p.cluster_id}" data-direction="{p.direction}"{row_style}>
          <td><span style="font-weight:700;color:#e5e7eb">{p.ticker}</span>
              <div style="font-size:10px;color:#9ca3af;max-width:100px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{p.name[:18]}</div></td>
          <td>{direction_badge(p.direction)}</td>
          <td>{conviction_badge(p.conviction)}</td>
          <td style="color:{alloc_color};font-weight:700">{_pct0(abs(p.allocation_pct))}</td>
          <td style="color:{alloc_color}">{_usd(abs(p.allocation_usd))}</td>
          {pnl_cell(p.ticker)}
          <td style="color:#9ca3af;font-size:12px">{entry_str}</td>
          {last_cell}
          <td style="color:{stop_color};font-size:12px;font-weight:600">{stop_str}
              <div style="font-size:10px;color:#6b7280">{_pct0(stop_pct_v) if stop_pct_v else ''}</div></td>
          <td style="color:{tgt_color};font-size:12px;font-weight:600">{target_str}</td>
          <td style="color:{rr_color};font-weight:700">{rr_str}</td>
          <td style="color:#e5e7eb">{_pct(p.expected_return_ann)}</td>
          <td style="color:{'#10b981' if p.sharpe_implied > 0.5 else '#ef4444' if p.sharpe_implied < 0 else '#f59e0b'}">{_fmt(p.sharpe_implied)}</td>
          <td style="max-width:220px">
            <div style="font-size:10px;color:#9ca3af">{' · '.join(p.rationale[:2]) if p.rationale else '—'}</div>
            {_tv_chart_badge(p.ticker, tv_map)}
          </td>
        </tr>"""

    # ── Watchlist rows ──────────────────────────────────────────────────────
    watch_rows = ""
    for w in por.watchlist:
        c = "#f59e0b"
        watch_rows += f"""
        <tr>
          <td style="color:#e5e7eb;font-weight:600">{w['ticker']}</td>
          <td style="color:{c};font-size:11px">WATCH</td>
          <td style="color:#9ca3af;font-size:11px">{_fmt(w['composite'],3)}</td>
          <td style="color:#9ca3af;font-size:11px;max-width:200px">{' · '.join(w.get('rationale',[''])[:2])}</td>
        </tr>"""

    # ── Scatter data (JS) ───────────────────────────────────────────────────
    scatter_points = []
    for p in por.positions:
        scatter_points.append({
            "ticker": p.ticker,
            "x": round(p.risk_score * 100, 1),
            "y": round(p.expected_return_ann * 100, 1),
            "r": max(4, min(20, int(abs(p.allocation_pct) * 120))),
            "color": "#10b981" if p.direction == "long" else "#ef4444",
            "alloc": round(p.allocation_pct * 100, 1),
            "conviction": p.conviction,
        })

    # ── Signal heatmap data ─────────────────────────────────────────────────
    # Top 20 assets by |composite|
    top_signals = sorted(
        signals.values(),
        key=lambda s: abs(s.composite), reverse=True
    )[:20]

    heatmap_rows = ""
    for s in top_signals:
        row_color = _DIRECTION_COLORS.get(s.direction, "#6b7280")
        # Normalize score to display
        def score_cell(v: float) -> str:
            color = "#10b981" if v > 0.05 else ("#ef4444" if v < -0.05 else "#6b7280")
            intensity = min(1.0, abs(v) * 3)
            bg = f"rgba({'16,185,129' if v > 0 else '239,68,68'},{intensity:.2f})"
            return f'<td style="background:{bg};color:#fff;font-size:11px;text-align:center">{v:+.2f}</td>'

        heatmap_rows += f"""
        <tr>
          <td style="color:{row_color};font-weight:700;white-space:nowrap">{s.ticker}</td>
          {score_cell(s.momentum_score)}
          {score_cell(s.mean_rev_score)}
          {score_cell(s.vol_edge_score)}
          {score_cell(s.options_flow_score)}
          {score_cell(s.composite)}
          <td style="color:#9ca3af;font-size:11px">{s.conviction}</td>
        </tr>"""

    # ── Options strategy rows ───────────────────────────────────────────────
    _STRAT_LABELS = {
        "long_put":   ("PUT",       "#ef4444"),
        "put_spread": ("PUT SPD",   "#f87171"),
        "long_call":  ("CALL",      "#10b981"),
        "call_spread":("CALL SPD",  "#34d399"),
        "spy_hedge":  ("SPY HEDGE", "#8b5cf6"),
        "collar":     ("COLLAR",    "#f59e0b"),
    }

    opts_rows = ""
    opts_total_cost = 0.0
    opts_max_profit = 0.0
    opts_max_loss = 0.0

    if options_strategy and options_strategy.recommendations:
        for rec in options_strategy.recommendations:
            label, color = _STRAT_LABELS.get(rec.strategy, (rec.strategy.upper(), "#9ca3af"))
            if rec.buy_strike and rec.sell_strike:
                strike_str = f"{rec.buy_strike:.0f}/{rec.sell_strike:.0f}"
            elif rec.strike:
                strike_str = f"{rec.strike:.0f}"
            else:
                strike_str = "—"
            cost_str   = f"${rec.debit_credit:,.0f}" if rec.debit_credit else "—"
            profit_str = f"${rec.max_profit:,.0f}" if rec.max_profit else "∞"
            loss_str   = f"${rec.max_loss:,.0f}"   if rec.max_loss else "—"
            be_str     = f"{rec.breakeven:.2f}"     if rec.breakeven else "—"
            iv_str     = f"{rec.iv_used:.0%}"       if rec.iv_used else "—"
            ivp_str    = f"{rec.iv_percentile:.0%}" if rec.iv_percentile else "—"
            kelly_str  = f"{rec.kelly_fraction:.1%}" if rec.kelly_fraction else "—"
            opts_rows += f"""
        <tr>
          <td style="font-weight:700;color:#e5e7eb">{rec.ticker}</td>
          <td><span style="background:{color}22;color:{color};padding:2px 6px;border-radius:4px;font-size:11px;font-weight:700">{label}</span></td>
          <td style="color:#9ca3af">{rec.expiry}<br><span style="font-size:10px">{rec.dte}DTE</span></td>
          <td style="color:#e5e7eb">{strike_str}</td>
          <td style="color:#f59e0b;font-weight:600">{cost_str}</td>
          <td style="color:#10b981">{profit_str}</td>
          <td style="color:#ef4444">{loss_str}</td>
          <td style="color:#e5e7eb">{be_str}</td>
          <td style="color:#9ca3af">{rec.contracts}x</td>
          <td style="color:#f59e0b;font-weight:600">${rec.total_cost_usd:,.0f}</td>
          <td style="color:#9ca3af;font-size:10px">{iv_str} ({ivp_str}ile)</td>
          <td style="color:#6b7280;font-size:10px">{kelly_str}</td>
          <td style="color:#6b7280;font-size:10px;max-width:180px">{rec.rationale[:60]}</td>
        </tr>"""
            opts_total_cost  += rec.total_cost_usd
            opts_max_profit  += rec.max_profit or 0
            opts_max_loss    += rec.max_loss or 0

    opts_empty = (
        '<tr><td colspan="13" style="color:#6b7280;text-align:center;padding:20px">'
        'Sem recomendacoes de opcoes — rode portfolio allocation primeiro</td></tr>'
    )

    # ── RRG data for JS visualization ───────────────────────────────────────
    rrg_points = []
    rrg_rows = ""
    if rrg_result and rrg_result.signals:
        _quad_colors = {
            "leading":   "#10b981",
            "improving": "#3b82f6",
            "weakening": "#f59e0b",
            "lagging":   "#ef4444",
        }
        # Sort by RS percentile desc
        sorted_sigs = sorted(
            rrg_result.signals.items(),
            key=lambda x: x[1].rs_percentile, reverse=True
        )
        for ticker, rs_sig in sorted_sigs[:40]:
            qc = _quad_colors.get(rs_sig.quadrant, "#6b7280")
            in_portfolio = ticker in {p.ticker for p in portfolio.positions}
            rrg_points.append({
                "ticker": ticker,
                "x": rs_sig.rs_ratio,
                "y": rs_sig.rs_momentum,
                "quadrant": rs_sig.quadrant,
                "color": qc,
                "pct": rs_sig.rs_percentile,
                "in_portfolio": in_portfolio,
                "tail_x": rs_sig.tail_rs_ratio,
                "tail_y": rs_sig.tail_rs_momentum,
                "rotation": rs_sig.rotation_direction,
                "perf_1m": round(rs_sig.perf_1m_vs_bench * 100, 1) if rs_sig.perf_1m_vs_bench else None,
            })
            badge = '<span style="background:#10b98133;color:#10b981;font-size:10px;padding:1px 4px;border-radius:3px">PORT</span>' if in_portfolio else ""
            perf_1m = f"{rs_sig.perf_1m_vs_bench:+.1%}" if rs_sig.perf_1m_vs_bench else "—"
            rrg_rows += f"""
        <tr>
          <td style="font-weight:700;color:{qc}">{ticker} {badge}</td>
          <td><span style="color:{qc};font-weight:700;font-size:11px">{rs_sig.quadrant.upper()}</span></td>
          <td style="color:#e5e7eb">{rs_sig.rs_ratio:.1f}</td>
          <td style="color:#e5e7eb">{rs_sig.rs_momentum:.1f}</td>
          <td style="color:{'#10b981' if rs_sig.rs_percentile >= 50 else '#ef4444'};font-weight:600">{rs_sig.rs_percentile:.0f}</td>
          <td style="color:{'#10b981' if (rs_sig.perf_1m_vs_bench or 0) >= 0 else '#ef4444'}">{perf_1m}</td>
          <td style="color:#9ca3af;font-size:11px">{rs_sig.rotation_direction}</td>
          <td style="font-size:10px;color:#6b7280">{' · '.join(rs_sig.rationale[:2])}</td>
        </tr>"""

    rrg_js = json.dumps(rrg_points)

    # ── Pairs trading rows ──────────────────────────────────────────────────
    pairs_rows = ""
    if pairs_result and pairs_result.active_pairs:
        for p in pairs_result.active_pairs:
            z_color = "#10b981" if p.z_score < 0 else "#ef4444"
            pairs_rows += f"""
        <tr>
          <td style="color:#10b981;font-weight:700">{p.long_ticker}</td>
          <td style="color:#ef4444;font-weight:700">{p.short_ticker}</td>
          <td style="color:{z_color};font-weight:700">{p.z_score:+.2f}σ</td>
          <td style="color:#9ca3af">{f"{p.half_life_days:.0f}d" if p.half_life_days is not None else "—"}</td>
          <td style="color:#e5e7eb">{(p.correlation or 0):.2f}</td>
          <td style="color:#f59e0b">{p.allocation_pct:.1%}</td>
          <td style="color:#9ca3af;font-size:11px">{p.rationale[:60]}</td>
        </tr>"""

    pairs_empty = '<tr><td colspan="7" style="color:#6b7280;text-align:center;padding:20px">Nenhum par ativo — sem divergencia estatistica suficiente</td></tr>'

    # ── Vol regime HTML card ─────────────────────────────────────────────────
    if vol_regime:
        _vr_colors = {"calm": "#10b981", "elevated": "#f59e0b", "stress": "#f97316", "crisis": "#ef4444", "unknown": "#6b7280"}
        vr_color = _vr_colors.get(vol_regime.regime, "#6b7280")
        vr_label = vol_regime.regime.upper()
        vr_vix    = f"{vol_regime.vix:.1f}" if vol_regime.vix else "—"
        vr_ts     = f"{vol_regime.term_structure_ratio:.2f}" if vol_regime.term_structure_ratio else "—"
        vr_vvix   = f"{vol_regime.vvix:.0f}" if vol_regime.vvix else "—"
        vr_rv     = f"{vol_regime.realized_vol_10d:.1f}%" if vol_regime.realized_vol_10d else "—"
        vr_vrp    = f"{vol_regime.vol_risk_premium:+.1f}" if vol_regime.vol_risk_premium else "—"
        vr_scalar = f"{vol_regime.position_scalar:.0%}"
        vr_reasons = "<br>".join(f"• {r}" for r in vol_regime.reasons[:4])
        vol_card_html = f"""
    <div class="card">
      <div class="card-title">Vol Regime <span style="color:{vr_color};font-weight:700">{vr_label}</span></div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px;font-size:12px">
        <div>VIX: <span style="color:{vr_color};font-weight:700">{vr_vix}</span></div>
        <div>VIX/VIX3M: <span style="color:#e5e7eb">{vr_ts}</span></div>
        <div>VVIX: <span style="color:#e5e7eb">{vr_vvix}</span></div>
        <div>RV 10d: <span style="color:#e5e7eb">{vr_rv}</span></div>
        <div>VRP: <span style="color:#e5e7eb">{vr_vrp}</span></div>
        <div>Pos Scale: <span style="color:{'#10b981' if vol_regime.position_scalar >= 1.0 else '#ef4444'}">{vr_scalar}</span></div>
      </div>
      <div style="font-size:11px;color:#9ca3af;line-height:1.6">{vr_reasons}</div>
      {'<div style="margin-top:6px;font-size:11px;color:#ef4444;font-weight:600">⚠ HEDGE OBRIGATORIO (' + vol_regime.hedge_asset + ')</div>' if vol_regime.hedge_required else ''}
    </div>"""
    else:
        vol_card_html = ""

    # ── Asset class breakdown ───────────────────────────────────────────────
    ac_bars = ""
    ac_colors = {
        "equity": "#3b82f6", "bonds": "#10b981", "commodities": "#f59e0b",
        "fx": "#8b5cf6", "crypto": "#ec4899", "intl_equity": "#06b6d4",
    }
    for ac, pct in sorted(por.by_asset_class.items(), key=lambda x: -x[1]):
        color = ac_colors.get(ac, "#6b7280")
        bar_w = int(pct * 200)
        ac_bars += f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
          <div style="width:80px;color:#9ca3af;font-size:12px;text-align:right">{ac}</div>
          <div style="width:{bar_w}px;height:14px;background:{color};border-radius:3px;min-width:2px"></div>
          <div style="color:#e5e7eb;font-size:12px">{_pct0(pct)}</div>
        </div>"""

    # ── Direction breakdown ─────────────────────────────────────────────────
    long_pct  = por.by_direction.get("long_pct", 0)
    short_pct = por.by_direction.get("short_pct", 0)
    cash_pct  = por.by_direction.get("cash_pct", 0)

    # ── Regime card data ────────────────────────────────────────────────────
    regime_upper = por.regime_mode.upper()
    regime_desc = {
        "bull":    "Momentum risk-on · equities sobre-representadas",
        "neutral": "Balanceado · exposição mista",
        "bear":    "Defensivo · redução de equities · bonds/commodities preferidos",
    }.get(por.regime_mode, "")

    now_str = datetime.now().strftime("%H:%M:%S")
    live_badge = (
        '<span style="background:#10b98133;color:#10b981;padding:2px 8px;border-radius:4px;'
        'font-size:11px;font-weight:700;animation:pulse 2s infinite">● LIVE</span>'
        if live_mode else
        '<span style="background:#f59e0b22;color:#f59e0b;padding:2px 8px;border-radius:4px;'
        'font-size:11px">◉ SNAPSHOT</span>'
    )

    auto_refresh_js = ""
    if live_mode:
        auto_refresh_js = f"""
    // Auto-refresh (JS-based, works on file://)
    setTimeout(() => location.reload(), {refresh_interval * 1000});
    console.log('Auto-refresh in {refresh_interval}s');"""

    scatter_js = json.dumps(scatter_points)
    budget_usd = _usd(equity)

    html = f"""<!DOCTYPE html>
<html lang="pt">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flow Inspector — Portfolio {bundle_date}</title>
<style>
  :root {{
    --bg: #0f1117; --surface: #1a1d27; --border: #2d3142;
    --text: #e5e7eb; --muted: #9ca3af; --accent: #3b82f6;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'SF Mono', 'Consolas', monospace; font-size: 13px; }}

  .header {{ background: #0d1117; border-bottom: 1px solid var(--border); padding: 12px 20px;
             display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 100; }}
  .header-title {{ font-size: 16px; font-weight: 700; color: var(--text); }}
  .header-meta {{ display: flex; gap: 12px; align-items: center; font-size: 12px; color: var(--muted); }}

  .layout {{ display: grid; grid-template-columns: 1fr 320px; gap: 0; height: calc(100vh - 50px); overflow: hidden; }}
  .main {{ overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 16px; }}
  .sidebar {{ border-left: 1px solid var(--border); overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 14px; }}

  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }}
  .card-title {{ font-size: 11px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px; }}

  /* Metrics row */
  .metrics {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }}
  .metric {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; text-align: center; }}
  .metric-val {{ font-size: 20px; font-weight: 700; }}
  .metric-lbl {{ font-size: 10px; color: var(--muted); margin-top: 2px; text-transform: uppercase; }}

  /* Tables */
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{ background: #0d1117; color: var(--muted); font-weight: 600; padding: 6px 8px; text-align: left;
        border-bottom: 1px solid var(--border); position: sticky; top: 0; font-size: 10px; text-transform: uppercase; }}
  td {{ padding: 5px 8px; border-bottom: 1px solid #1e2230; vertical-align: middle; }}
  tr:hover td {{ background: #1e2230; }}

  /* Tabs */
  .tabs {{ display: flex; gap: 2px; background: #0d1117; padding: 4px; border-radius: 8px; margin-bottom: 14px; }}
  .tab {{ padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 12px; color: var(--muted); transition: all 0.2s; }}
  .tab.active {{ background: var(--accent); color: white; font-weight: 600; }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}

  /* Chart canvas */
  #scatter-canvas {{ width: 100%; height: 280px; background: #0d1117; border-radius: 6px; }}

  /* Regime badge */
  .regime-badge {{ padding: 4px 12px; border-radius: 6px; font-size: 12px; font-weight: 700;
                   border: 1px solid; display: inline-block; }}

  /* Filter buttons */
  .filter-btn {{ padding: 4px 10px; border: 1px solid var(--border); border-radius: 4px; cursor: pointer;
                 font-size: 11px; color: var(--muted); background: transparent; }}
  .filter-btn.active {{ background: var(--accent); border-color: var(--accent); color: white; }}

  @keyframes pulse {{ 0%,100% {{ opacity:1 }} 50% {{ opacity:.4 }} }}

  /* Scrollbar */
  ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
  ::-webkit-scrollbar-track {{ background: var(--bg); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}

  /* Direction exposure bar */
  .exposure-bar {{ height: 24px; border-radius: 4px; overflow: hidden; display: flex; }}
  .exposure-seg {{ display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; color: white; transition: width 0.5s; }}
</style>
</head>
<body>

<div class="header">
  <div style="display:flex;align-items:center;gap:12px">
    <span class="header-title">Flow Inspector</span>
    <span style="color:#6b7280">|</span>
    <span style="color:#9ca3af">{bundle_date}</span>
    {live_badge}
  </div>
  <div class="header-meta">
    <span>Capital: <strong style="color:#e5e7eb">{budget_usd}</strong></span>
    {'<span style="background:' + pnl_color + '22;color:' + pnl_color + ';padding:2px 8px;border-radius:4px;font-size:12px;font-weight:700">P&amp;L: ' + pnl_sign + '$' + f'{abs(pnl_total):,.0f}' + ' (' + pnl_sign + f'{pnl_pct:.2%})</span>' if pnl_total != 0 else ''}
    <span>Atualizado: <strong id="refresh-time">{now_str}</strong></span>
    <span style="background:{regime_color}22;color:{regime_color};padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700">{regime_upper}</span>
  </div>
</div>

<div class="layout">
  <!-- MAIN CONTENT -->
  <div class="main">
    <!-- Metrics -->
    <div class="metrics">
      <div class="metric">
        <div class="metric-val" style="color:{'#10b981' if por.expected_return_ann > 0 else '#ef4444'}">{_pct(por.expected_return_ann)}</div>
        <div class="metric-lbl">Retorno Esperado</div>
      </div>
      <div class="metric">
        <div class="metric-val" style="color:#f59e0b">{_pct0(por.portfolio_vol)}</div>
        <div class="metric-lbl">Volatilidade</div>
      </div>
      <div class="metric">
        <div class="metric-val" style="color:{'#10b981' if por.sharpe > 0.5 else '#ef4444' if por.sharpe < 0 else '#f59e0b'}">{_fmt(por.sharpe)}</div>
        <div class="metric-lbl">Sharpe</div>
      </div>
      <div class="metric">
        <div class="metric-val" style="color:#ef4444">{_pct0(por.max_drawdown_est)}</div>
        <div class="metric-lbl">Max DD Est.</div>
      </div>
      <div class="metric">
        <div class="metric-val" style="color:#3b82f6">{len(por.positions)}</div>
        <div class="metric-lbl">Posições ({por.by_conviction.get('high',0)}H/{por.by_conviction.get('medium',0)}M)</div>
      </div>
    </div>

    <!-- Tabs -->
    <div class="card" style="padding:14px">
      <div class="tabs">
        <div class="tab active" onclick="switchTab('portfolio')">Portfolio</div>
        <div class="tab" onclick="switchTab('signals')">Sinais Alpha</div>
        <div class="tab" onclick="switchTab('rrg')">RRG ◎</div>
        <div class="tab" onclick="switchTab('scatter')">Risco × Retorno</div>
        <div class="tab" onclick="switchTab('options')">Opcoes 🎯</div>
        <div class="tab" onclick="switchTab('pairs')">Pairs ↔</div>
        <div class="tab" onclick="switchTab('historico')">Historico 📋</div>
        <div class="tab" onclick="switchTab('watchlist')">Watchlist</div>
      </div>

      <!-- Portfolio tab -->
      <div id="tab-portfolio" class="tab-content active">
        <div style="display:flex;gap:8px;margin-bottom:10px">
          <button class="filter-btn active" onclick="filterPositions('all',this)">Todos</button>
          <button class="filter-btn" onclick="filterPositions('long',this)">Long</button>
          <button class="filter-btn" onclick="filterPositions('short',this)">Short</button>
          <button class="filter-btn" onclick="filterPositions('high',this)">High Conv.</button>
        </div>
        <div style="overflow-x:auto">
        <table id="pos-table">
          <thead>
            <tr>
              <th>Ativo</th><th>Dir.</th><th>Conv.</th>
              <th>Alloc%</th><th>USD</th>
              <th style="color:#10b981">P&amp;L $</th>
              <th>Entrada</th>
              <th style="color:#60a5fa">Last</th>
              <th style="color:#ef4444">Stop</th>
              <th style="color:#10b981">Target</th>
              <th>R:R</th>
              <th>E[R]</th><th>Sharpe</th>
              <th>Sinal</th>
            </tr>
          </thead>
          <tbody>
            {pos_rows}
          </tbody>
        </table>
        </div>
      </div>

      <!-- Signals tab -->
      <div id="tab-signals" class="tab-content">
        <div style="overflow-x:auto">
        <table>
          <thead>
            <tr>
              <th>Ativo</th>
              <th style="color:#3b82f6">Momentum</th>
              <th style="color:#10b981">MeanRev</th>
              <th style="color:#f59e0b">VolEdge</th>
              <th style="color:#8b5cf6">OptFlow</th>
              <th style="color:#e5e7eb">Composite</th>
              <th>Conv.</th>
            </tr>
          </thead>
          <tbody>
            {heatmap_rows}
          </tbody>
        </table>
        </div>
      </div>

      <!-- RRG tab -->
      <div id="tab-rrg" class="tab-content">
        <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px">
          <div style="display:flex;align-items:center;gap:6px;font-size:12px">
            <div style="width:12px;height:12px;background:#10b981;border-radius:50%"></div><span>Leading (comprar)</span>
          </div>
          <div style="display:flex;align-items:center;gap:6px;font-size:12px">
            <div style="width:12px;height:12px;background:#3b82f6;border-radius:50%"></div><span>Improving (monitorar entrada)</span>
          </div>
          <div style="display:flex;align-items:center;gap:6px;font-size:12px">
            <div style="width:12px;height:12px;background:#f59e0b;border-radius:50%"></div><span>Weakening (preparar saida)</span>
          </div>
          <div style="display:flex;align-items:center;gap:6px;font-size:12px">
            <div style="width:12px;height:12px;background:#ef4444;border-radius:50%"></div><span>Lagging (short)</span>
          </div>
        </div>
        <canvas id="rrg-canvas" style="width:100%;max-width:700px;height:500px;cursor:crosshair"></canvas>
        <div id="rrg-tooltip" style="position:fixed;background:#1a1d27;border:1px solid #3d4460;border-radius:6px;padding:8px 12px;font-size:12px;pointer-events:none;display:none;z-index:999"></div>
        <div style="margin-top:14px;overflow-x:auto">
        <table>
          <thead>
            <tr>
              <th>Ativo</th><th>Quadrante</th>
              <th>RS-Ratio</th><th>RS-Mom</th>
              <th>RS%ile</th><th>1M vs SPY</th>
              <th>Rotacao</th><th>Sinal</th>
            </tr>
          </thead>
          <tbody>{rrg_rows or '<tr><td colspan="8" style="color:#6b7280;text-align:center;padding:20px">Calculando RRG...</td></tr>'}</tbody>
        </table>
        </div>
      </div>

      <!-- Scatter tab -->
      <div id="tab-scatter" class="tab-content">
        <div style="color:#9ca3af;font-size:11px;margin-bottom:8px">Eixo X = Risco anual (vol%) · Eixo Y = Retorno esperado% · Tamanho = alocação</div>
        <canvas id="scatter-canvas"></canvas>
        <div id="scatter-tooltip" style="position:fixed;background:#1a1d27;border:1px solid #3d4460;border-radius:6px;padding:8px 12px;font-size:12px;pointer-events:none;display:none;z-index:999"></div>
      </div>

      <!-- Options tab -->
      <div id="tab-options" class="tab-content">
        <div style="display:flex;gap:20px;margin-bottom:12px;flex-wrap:wrap">
          <div class="metric">
            <div class="metric-val" style="color:#f59e0b">${opts_total_cost:,.0f}</div>
            <div class="metric-lbl">Custo Total</div>
          </div>
          <div class="metric">
            <div class="metric-val" style="color:#10b981">${opts_max_profit:,.0f}</div>
            <div class="metric-lbl">Max Profit</div>
          </div>
          <div class="metric">
            <div class="metric-val" style="color:#ef4444">${opts_max_loss:,.0f}</div>
            <div class="metric-lbl">Max Loss</div>
          </div>
          <div class="metric">
            <div class="metric-val" style="color:#3b82f6">{len(options_strategy.recommendations) if options_strategy else 0}</div>
            <div class="metric-lbl">Estrategias</div>
          </div>
        </div>
        <div style="color:#9ca3af;font-size:11px;margin-bottom:10px">
          Recomendacoes baseadas em Kelly fraction (1/4 Kelly), IV percentile e regime {por.regime_mode.upper()}.
          Strikes teoricos Black-Scholes. Verificar preco de mercado antes de executar.
        </div>
        <div style="overflow-x:auto">
        <table>
          <thead>
            <tr>
              <th>Ativo</th><th>Estrategia</th><th>Expiry</th>
              <th>Strike(s)</th><th>Debit/Cont</th>
              <th style="color:#10b981">Max Profit</th>
              <th style="color:#ef4444">Max Loss</th>
              <th>Breakeven</th><th>Contratos</th>
              <th style="color:#f59e0b">Custo Total</th>
              <th>IV (pctile)</th><th>Kelly f*</th><th>Rationale</th>
            </tr>
          </thead>
          <tbody>
            {opts_rows or opts_empty}
          </tbody>
        </table>
        </div>
        {f'<div style="color:#ef4444;font-size:11px;margin-top:8px">Erros: {"; ".join(options_strategy.errors[:3])}</div>' if options_strategy and options_strategy.errors else ''}
      </div>

      <!-- Pairs tab -->
      <div id="tab-pairs" class="tab-content">
        <div style="color:#9ca3af;font-size:11px;margin-bottom:10px">
          Pares estatisticamente cointegrados que divergiram alem de 1.8σ.
          Estrategia market-neutral: Long o underperformer, Short o overperformer.
          Half-life indica velocidade de convergencia esperada.
        </div>
        <div style="overflow-x:auto">
        <table>
          <thead>
            <tr>
              <th style="color:#10b981">Long</th>
              <th style="color:#ef4444">Short</th>
              <th>Z-Score</th>
              <th>Half-Life</th>
              <th>Correlacao</th>
              <th>Alloc %</th>
              <th>Rationale</th>
            </tr>
          </thead>
          <tbody>
            {pairs_rows or pairs_empty}
          </tbody>
        </table>
        </div>
        {f'<div style="color:#9ca3af;font-size:11px;margin-top:8px">Pares analisados: {pairs_result.total_pairs_analyzed} | Ativos: {len(pairs_result.active_pairs)}</div>' if pairs_result else ''}
      </div>

      <!-- Historico tab -->
      <div id="tab-historico" class="tab-content">
        <div style="color:#9ca3af;font-size:11px;margin-bottom:10px">
          Ledger completo de todas as posicoes abertas e fechadas — entrada, saida, P&amp;L e raciocinio.
        </div>
        <div style="overflow-x:auto">
        <table>
          <thead>
            <tr>
              <th>ID</th><th>Ativo</th><th>Dir.</th><th>Status</th>
              <th>Entrada</th><th>Preco Entrada</th>
              <th>Saida</th><th>Preco Saida</th>
              <th style="color:#10b981">P&amp;L $</th>
              <th>P&amp;L %</th>
              <th>Por que entrou</th>
              <th>Por que saiu</th>
            </tr>
          </thead>
          <tbody>
            {trade_history_rows}
          </tbody>
        </table>
        </div>
      </div>

      <!-- Watchlist tab -->
      <div id="tab-watchlist" class="tab-content">
        <div style="color:#9ca3af;font-size:11px;margin-bottom:8px">Ativos próximos do threshold — monitorar para entrada</div>
        <table>
          <thead>
            <tr><th>Ativo</th><th>Status</th><th>Score</th><th>Fatores</th></tr>
          </thead>
          <tbody>
            {watch_rows or '<tr><td colspan="4" style="color:#6b7280;text-align:center;padding:20px">Nenhum ativo no watchlist</td></tr>'}
          </tbody>
        </table>
      </div>
    </div>

    <!-- Summary bar -->
    <div class="card">
      <div class="card-title">Resumo do Run</div>
      <div style="color:#9ca3af;font-size:12px;line-height:1.6">{por.summary_text}</div>
    </div>
  </div>

  <!-- SIDEBAR -->
  <div class="sidebar">
    <!-- Regime -->
    <div class="card">
      <div class="card-title">Regime Macro</div>
      <div style="margin-bottom:8px">
        <span class="regime-badge" style="color:{regime_color};border-color:{regime_color}44;background:{regime_color}11">
          {regime_upper}
        </span>
      </div>
      <div style="color:#9ca3af;font-size:11px;line-height:1.5">{regime_desc}</div>
    </div>

    {vol_card_html}

    <!-- Exposure bars -->
    <div class="card">
      <div class="card-title">Exposicao Direcional</div>
      <div class="exposure-bar" style="margin-bottom:8px">
        <div class="exposure-seg" style="background:#10b981;width:{int(long_pct*100)}%">{_pct0(long_pct)}</div>
        <div class="exposure-seg" style="background:#ef4444;width:{int(short_pct*100)}%">{_pct0(short_pct)}</div>
        <div class="exposure-seg" style="background:#374151;width:{int(cash_pct*100)}%">{_pct0(cash_pct)}</div>
      </div>
      <div style="display:flex;gap:12px;font-size:11px">
        <span style="color:#10b981">■ Long</span>
        <span style="color:#ef4444">■ Short</span>
        <span style="color:#6b7280">■ Cash</span>
      </div>
    </div>

    <!-- Asset class breakdown -->
    <div class="card">
      <div class="card-title">Por Asset Class</div>
      {ac_bars or '<div style="color:#6b7280;font-size:12px">—</div>'}
    </div>

    <!-- Top longs -->
    <div class="card">
      <div class="card-title">Top Long</div>
      {''.join(
        f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:12px">'
        f'<span style="color:#10b981;font-weight:700">{p.ticker}</span>'
        f'<span style="color:#e5e7eb">{_pct0(p.allocation_pct)}</span>'
        f'<span style="color:#9ca3af">{_usd(p.allocation_usd)}</span>'
        f'</div>'
        for p in por.positions if p.direction == "long"
      )[:600] or '<div style="color:#6b7280;font-size:12px">—</div>'}
    </div>

    <!-- Top shorts -->
    <div class="card">
      <div class="card-title">Top Short</div>
      {''.join(
        f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:12px">'
        f'<span style="color:#ef4444;font-weight:700">{p.ticker}</span>'
        f'<span style="color:#ef4444">{_pct0(abs(p.allocation_pct))}</span>'
        f'<span style="color:#9ca3af">{_usd(abs(p.allocation_usd))}</span>'
        f'</div>'
        for p in por.positions if p.direction == "short"
      )[:400] or '<div style="color:#6b7280;font-size:12px">—</div>'}
    </div>

    <!-- Conviction breakdown -->
    <div class="card">
      <div class="card-title">Por Convicção</div>
      <div style="display:flex;gap:8px;flex-direction:column;font-size:12px">
        <div style="display:flex;justify-content:space-between">
          <span style="color:#10b981">High</span>
          <span>{por.by_conviction.get('high', 0)} posições</span>
        </div>
        <div style="display:flex;justify-content:space-between">
          <span style="color:#f59e0b">Medium</span>
          <span>{por.by_conviction.get('medium', 0)} posições</span>
        </div>
        <div style="display:flex;justify-content:space-between">
          <span style="color:#6b7280">Low</span>
          <span>{por.by_conviction.get('low', 0)} posições</span>
        </div>
      </div>
    </div>

    <!-- Diversification -->
    <div class="card">
      <div class="card-title">Diversificação</div>
      <div style="font-size:24px;font-weight:700;color:#3b82f6;margin-bottom:4px">
        {int(por.diversification_score * 100)}%
      </div>
      <div style="color:#9ca3af;font-size:11px">Score (1 - HHI de pesos)</div>
    </div>

    <!-- Portfolio P&L Tracking -->
    <div class="card">
      <div class="card-title">Tracking P&amp;L</div>
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px">
        <div style="font-size:22px;font-weight:700;color:{pnl_color}">{pnl_sign}${abs(pnl_total):,.0f}</div>
        <div style="font-size:14px;color:{pnl_color}">{pnl_sign}{pnl_pct:.2%}</div>
      </div>
      <canvas id="pnl-chart" width="280" height="70" style="width:100%;height:70px;background:#0d1117;border-radius:4px"></canvas>
      <div style="color:#6b7280;font-size:10px;margin-top:4px;text-align:right">{len(pnl_history)} snapshots</div>
    </div>
  </div>
</div>

<script>
// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(name) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
  if (name === 'scatter') drawScatter();
  if (name === 'rrg') drawRRG();
}}

// ── Position filter ───────────────────────────────────────────────────────────
function filterPositions(filter, btn) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.pos-row').forEach(row => {{
    const dir  = row.dataset.direction;
    const conv = row.querySelector('span[style*="font-size:11px"]')?.textContent?.trim().toLowerCase() || '';
    let show = true;
    if (filter === 'long')   show = dir === 'long';
    if (filter === 'short')  show = dir === 'short';
    if (filter === 'high')   show = conv.includes('high');
    row.style.display = show ? '' : 'none';
  }});
}}

// ── Scatter plot ──────────────────────────────────────────────────────────────
const scatterData = {scatter_js};
let scatterDrawn = false;

function drawScatter() {{
  if (scatterDrawn) return;
  scatterDrawn = true;
  const canvas = document.getElementById('scatter-canvas');
  const tooltip = document.getElementById('scatter-tooltip');
  if (!canvas || !scatterData.length) return;

  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const pad = {{ l: 50, r: 20, t: 20, b: 40 }};
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  // Ranges
  const xs = scatterData.map(d => d.x), ys = scatterData.map(d => d.y);
  const xmin = Math.min(...xs) - 2, xmax = Math.max(...xs) + 2;
  const ymin = Math.min(...ys) - 2, ymax = Math.max(...ys) + 2;

  function toCanvas(x, y) {{
    return [
      pad.l + (x - xmin) / (xmax - xmin) * pw,
      pad.t + ph - (y - ymin) / (ymax - ymin) * ph
    ];
  }}

  // Grid
  ctx.strokeStyle = '#1f2937'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {{
    const x = pad.l + i * pw / 4;
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ph); ctx.stroke();
    const y = pad.t + i * ph / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke();
  }}

  // Zero line
  if (ymin < 0 && ymax > 0) {{
    const [, y0] = toCanvas(0, 0);
    ctx.strokeStyle = '#374151'; ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(pad.l, y0); ctx.lineTo(pad.l + pw, y0); ctx.stroke();
    ctx.setLineDash([]);
  }}

  // Axes labels
  ctx.fillStyle = '#6b7280'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
  ctx.fillText('Risco (vol %)', pad.l + pw / 2, H - 4);
  ctx.save(); ctx.translate(12, pad.t + ph / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText('Retorno Esperado %', 0, 0); ctx.restore();

  // Points
  scatterData.forEach(d => {{
    const [cx, cy] = toCanvas(d.x, d.y);
    ctx.beginPath();
    ctx.arc(cx, cy, d.r, 0, Math.PI * 2);
    ctx.fillStyle = d.color + '99';
    ctx.fill();
    ctx.strokeStyle = d.color;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = '#e5e7eb'; ctx.font = `bold ${{Math.min(11, d.r)}}px monospace`;
    ctx.textAlign = 'center';
    ctx.fillText(d.ticker, cx, cy + d.r + 12);
  }});

  // Hover
  const points = scatterData.map(d => {{ const [cx, cy] = toCanvas(d.x, d.y); return {{ ...d, cx, cy }}; }});
  canvas.addEventListener('mousemove', e => {{
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const hit = points.find(p => Math.hypot(p.cx - mx, p.cy - my) < p.r + 6);
    if (hit) {{
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX + 12) + 'px';
      tooltip.style.top  = (e.clientY - 30) + 'px';
      tooltip.innerHTML = `<strong style="color:${{hit.color}}">${{hit.ticker}}</strong><br>
        Risco: ${{hit.x.toFixed(1)}}% | Retorno: ${{hit.y > 0 ? '+' : ''}}${{hit.y.toFixed(1)}}%<br>
        Alocação: ${{hit.alloc > 0 ? '+' : ''}}${{hit.alloc.toFixed(1)}}%`;
    }} else {{
      tooltip.style.display = 'none';
    }}
  }});
}}

{auto_refresh_js}

// ── RRG (Relative Rotation Graph) ─────────────────────────────────────────────
const rrgData = {rrg_js};

function drawRRG() {{
  const canvas = document.getElementById('rrg-canvas');
  if (!canvas || rrgData.length === 0) return;
  const W = canvas.clientWidth || 700;
  const H = 500;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');

  // Background
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, W, H);

  const PAD = {{ l: 50, r: 20, t: 20, b: 50 }};
  const pw = W - PAD.l - PAD.r;
  const ph = H - PAD.t - PAD.b;

  // Data range (centered on 100, 100)
  const allX = rrgData.map(d => d.x).concat([95, 105]);
  const allY = rrgData.map(d => d.y).concat([95, 105]);
  const minX = Math.min(...allX) - 1, maxX = Math.max(...allX) + 1;
  const minY = Math.min(...allY) - 1, maxY = Math.max(...allY) + 1;

  const toCanvasX = x => PAD.l + (x - minX) / (maxX - minX) * pw;
  const toCanvasY = y => PAD.t + ph - (y - minY) / (maxY - minY) * ph;

  // Pivot lines (RS-Ratio = 100, RS-Mom = 100)
  const px0 = toCanvasX(100), py0 = toCanvasY(100);

  // Quadrant backgrounds
  const quads = [
    {{ x0: px0, y0: PAD.t, w: W-PAD.r-px0, h: py0-PAD.t, color: '#10b98108', label: 'LEADING',   lx: px0+20, ly: PAD.t+16 }},
    {{ x0: PAD.l, y0: PAD.t, w: px0-PAD.l, h: py0-PAD.t, color: '#3b82f608', label: 'IMPROVING', lx: PAD.l+8, ly: PAD.t+16 }},
    {{ x0: PAD.l, y0: py0,   w: px0-PAD.l, h: H-PAD.b-py0, color: '#ef444408', label: 'LAGGING',   lx: PAD.l+8, ly: H-PAD.b-8 }},
    {{ x0: px0, y0: py0,   w: W-PAD.r-px0, h: H-PAD.b-py0, color: '#f59e0b08', label: 'WEAKENING', lx: px0+20, ly: H-PAD.b-8 }},
  ];
  quads.forEach(q => {{
    ctx.fillStyle = q.color;
    ctx.fillRect(q.x0, q.y0, q.w, q.h);
    ctx.fillStyle = '#ffffff18';
    ctx.font = 'bold 11px monospace';
    ctx.fillText(q.label, q.lx, q.ly);
  }});

  // Grid lines (pivot)
  ctx.strokeStyle = '#3d4460';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(px0, PAD.t); ctx.lineTo(px0, H-PAD.b); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(PAD.l, py0); ctx.lineTo(W-PAD.r, py0); ctx.stroke();
  ctx.setLineDash([]);

  // Axes labels
  ctx.fillStyle = '#6b7280';
  ctx.font = '10px monospace';
  ctx.fillText('RS-Ratio →', W-PAD.r-60, H-PAD.b+14);
  ctx.save(); ctx.translate(14, PAD.t+ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('RS-Momentum ↑', -40, 0); ctx.restore();

  // Axis tick labels
  [minX, 100, maxX].forEach(v => {{
    const cx = toCanvasX(v);
    ctx.fillStyle = '#9ca3af';
    ctx.font = '9px monospace';
    ctx.fillText(v.toFixed(0), cx-10, H-PAD.b+12);
  }});
  [minY, 100, maxY].forEach(v => {{
    const cy = toCanvasY(v);
    ctx.fillText(v.toFixed(0), 4, cy+3);
  }});

  // Draw tails + dots for each asset
  const tooltip = document.getElementById('rrg-tooltip');
  const hitBoxes = [];

  rrgData.forEach(d => {{
    if (!d.x || !d.y) return;
    const cx = toCanvasX(d.x);
    const cy = toCanvasY(d.y);

    // Draw tail (path of last 5 periods)
    if (d.tail_x && d.tail_x.length > 1) {{
      ctx.strokeStyle = d.color + '55';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      d.tail_x.forEach((tx, i) => {{
        const ty = d.tail_y[i];
        if (!tx || !ty) return;
        const tx_c = toCanvasX(tx), ty_c = toCanvasY(ty);
        i === 0 ? ctx.moveTo(tx_c, ty_c) : ctx.lineTo(tx_c, ty_c);
      }});
      ctx.stroke();
      // Arrow at end of tail
      if (d.tail_x.length >= 2) {{
        const tx1 = toCanvasX(d.tail_x[d.tail_x.length-2]);
        const ty1 = toCanvasY(d.tail_y[d.tail_y.length-2]);
        const angle = Math.atan2(cy - ty1, cx - tx1);
        ctx.fillStyle = d.color + '88';
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(angle)*8, cy + Math.sin(angle)*8);
        ctx.lineTo(cx + Math.cos(angle-2.5)*4, cy + Math.sin(angle-2.5)*4);
        ctx.lineTo(cx + Math.cos(angle+2.5)*4, cy + Math.sin(angle+2.5)*4);
        ctx.fill();
      }}
    }}

    // Dot
    const r = d.in_portfolio ? 8 : 5;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI*2);
    ctx.fillStyle = d.color;
    ctx.fill();
    if (d.in_portfolio) {{
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }}

    // Label
    ctx.fillStyle = '#e5e7eb';
    ctx.font = `${{d.in_portfolio ? 'bold ' : ''}}10px monospace`;
    ctx.fillText(d.ticker, cx + r + 2, cy + 3);

    hitBoxes.push({{ d, cx, cy, r: r + 12 }});
  }});

  // Hover
  canvas.onmousemove = (e) => {{
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (W / rect.width);
    const my = (e.clientY - rect.top) * (H / rect.height);
    let hit = null;
    for (const h of hitBoxes) {{
      if (Math.hypot(mx-h.cx, my-h.cy) < h.r) {{ hit = h; break; }}
    }}
    if (hit) {{
      const d = hit.d;
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX + 12) + 'px';
      tooltip.style.top  = (e.clientY - 10) + 'px';
      tooltip.innerHTML = `<b style="color:${{d.color}}">${{d.ticker}}</b> — ${{d.quadrant.toUpperCase()}}<br>
        RS-Ratio: ${{d.x?.toFixed(2)}}<br>
        RS-Mom: ${{d.y?.toFixed(2)}}<br>
        RS%ile: ${{d.pct?.toFixed(0)}}<br>
        1M vs SPY: ${{d.perf_1m != null ? (d.perf_1m >= 0 ? '+' : '') + d.perf_1m?.toFixed(1) + '%' : '—'}}<br>
        Rotacao: ${{d.rotation}}`;
    }} else {{
      tooltip.style.display = 'none';
    }}
  }};
  canvas.onmouseleave = () => {{ tooltip.style.display = 'none'; }};
}}

// ── P&L Mini Chart ────────────────────────────────────────────────────────────
const pnlData = {pnl_history_js};
(function() {{
  const canvas = document.getElementById('pnl-chart');
  if (!canvas || pnlData.length < 2) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.clientWidth || 280, H = canvas.clientHeight || 70;
  canvas.width = W; canvas.height = H;
  const mn = Math.min(...pnlData), mx = Math.max(...pnlData);
  const range = mx - mn || 0.001;
  const pad = {{ l: 4, r: 4, t: 6, b: 6 }};
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;

  // Zero line
  const y0 = pad.t + ph - (0 - mn) / range * ph;
  ctx.strokeStyle = '#374151'; ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(pad.l, y0); ctx.lineTo(pad.l + pw, y0); ctx.stroke();
  ctx.setLineDash([]);

  // Fill
  const last = pnlData[pnlData.length - 1];
  const isPos = last >= 0;
  ctx.beginPath();
  pnlData.forEach((v, i) => {{
    const x = pad.l + i / (pnlData.length - 1) * pw;
    const y = pad.t + ph - (v - mn) / range * ph;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.lineTo(pad.l + pw, y0); ctx.lineTo(pad.l, y0); ctx.closePath();
  ctx.fillStyle = isPos ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)';
  ctx.fill();

  // Line
  ctx.beginPath();
  pnlData.forEach((v, i) => {{
    const x = pad.l + i / (pnlData.length - 1) * pw;
    const y = pad.t + ph - (v - mn) / range * ph;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.strokeStyle = isPos ? '#10b981' : '#ef4444';
  ctx.lineWidth = 1.5; ctx.stroke();
}})();
</script>
</body>
</html>"""

    return html


def save_flow_inspector(
    portfolio,
    signals: dict,
    bundle,
    out_dir: Path | None = None,
    live_mode: bool = False,
    options_strategy=None,
    vol_regime=None,
    pairs_result=None,
    rrg_result=None,
    tv_map: dict | None = None,
) -> Path:
    """Salva o HTML do Flow Inspector e retorna o path."""
    from app.storage.paths import workspace

    bundle_date = str(getattr(bundle, "run_date", ""))
    run_id = getattr(bundle, "run_id", "unknown")

    market_prices = getattr(bundle, "market_prices", None) or {}

    html = generate_flow_inspector_html(
        portfolio, signals,
        bundle_date=bundle_date,
        live_mode=live_mode,
        options_strategy=options_strategy,
        vol_regime=vol_regime,
        pairs_result=pairs_result,
        rrg_result=rrg_result,
        market_prices=market_prices,
        tv_map=tv_map,
    )

    if out_dir is None:
        from app.storage.bundle_store import bundle_store
        bundles = [
            p for p in bundle_store.list_bundles()
            if bundle_date in str(p) and "_" not in p.stem
        ]
        if bundles:
            out_dir = bundles[0].parent
        else:
            out_dir = Path(workspace.reports_dir) if hasattr(workspace, "reports_dir") else Path(".")

    out_path = out_dir / f"{run_id}_flow_inspector.html"
    out_path.write_text(html, encoding="utf-8")
    _log.info("flow_inspector_saved", path=str(out_path))
    return out_path
