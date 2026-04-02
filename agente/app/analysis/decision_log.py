"""
Decision Log — Registro de Raciocinio das Decisoes de Portfolio

Grava em JSON e gera HTML explicativo com o raciocinio por tras de cada decisao:
  - Por que cada ativo foi selecionado (ou rejeitado)
  - Por que foi Long vs Short
  - Qual regime macro foi detectado e como afetou a alocacao
  - Quais sinais dominaram (momentum, mean_rev, vol_edge, options_flow)
  - Por que certa opcao foi recomendada
  - Metricas de qualidade do portfolio (Sharpe, diversificacao)

Arquivo salvo em: workspace/reports/YYYY-MM-DD_decision_log.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("analysis.decision_log")


@dataclass
class SignalDecision:
    ticker: str
    name: str
    direction: str          # long | short | neutral
    conviction: str
    composite: float
    momentum_score: float
    mean_rev_score: float
    vol_edge_score: float
    options_flow_score: float
    contagion_penalty: float
    tail_penalty: float
    rationale: list[str]
    dominant_signal: str    # qual sinal mais contribuiu


@dataclass
class PositionDecision:
    ticker: str
    name: str
    direction: str
    conviction: str
    allocation_pct: float
    allocation_usd: float
    expected_return_ann: float
    risk_score: float
    sharpe_implied: float
    why_selected: list[str]
    why_not_neutral: str


@dataclass
class RegimeDecision:
    regime: str                  # bull | neutral | bear
    regime_bull_avg: float       # P(bull) medio
    tickers_used: list[str]
    effect: str                  # como afetou o portfolio
    safe_haven_injected: list[str]  # se bear, quais foram forcados


@dataclass
class OptionsDecision:
    ticker: str
    strategy: str
    rationale: str
    kelly_fraction: float | None
    iv_percentile: float | None
    why_this_strategy: str


@dataclass
class PortfolioDecisionLog:
    run_id: str
    timestamp: str
    bundle_date: str

    # Regime
    regime: RegimeDecision

    # Signals (todos os ativos analisados, nao so posicoes)
    signals_considered: list[SignalDecision]
    signals_selected: list[str]     # tickers que viraram posicoes
    signals_rejected: list[dict]    # {ticker, reason}

    # Posicoes
    positions: list[PositionDecision]

    # Portfolio metrics
    sharpe: float
    expected_return_ann: float
    portfolio_vol: float
    diversification: float
    max_drawdown_est: float

    # Options
    options_decisions: list[OptionsDecision]
    options_total_cost: float

    # Resumo narrativo
    narrative: str


def _dominant_signal(sig) -> str:
    """Qual sinal mais contribuiu para o composite."""
    scores = {
        "momentum":     abs(sig.momentum_score)    * 0.35,
        "mean_rev":     abs(sig.mean_rev_score)    * 0.20,
        "vol_edge":     abs(sig.vol_edge_score)    * 0.20,
        "options_flow": abs(sig.options_flow_score) * 0.25,
    }
    return max(scores, key=scores.get)


def _why_direction(sig) -> str:
    d = sig.direction
    c = sig.composite
    if d == "long":
        return (
            f"Composite +{c:.3f} (acima do threshold +0.08): "
            f"momentum={sig.momentum_score:+.2f}, "
            f"mean_rev={sig.mean_rev_score:+.2f}, "
            f"options_flow={sig.options_flow_score:+.2f}"
        )
    elif d == "short":
        return (
            f"Composite {c:.3f} (abaixo do threshold -0.08): "
            f"momentum={sig.momentum_score:+.2f}, "
            f"mean_rev={sig.mean_rev_score:+.2f}, "
            f"options_flow={sig.options_flow_score:+.2f}"
        )
    else:
        return f"Composite {c:.3f} entre -0.08 e +0.08 — sem direcao clara"


def _why_selected(pos, sig=None) -> list[str]:
    reasons = []
    if abs(pos.composite) > 0.25:
        reasons.append(f"Alta convicao: composite={pos.composite:+.3f}")
    elif abs(pos.composite) > 0.12:
        reasons.append(f"Media convicao: composite={pos.composite:+.3f}")
    if sig:
        dom = _dominant_signal(sig)
        reasons.append(f"Sinal dominante: {dom} ({abs(getattr(sig, dom+'_score', 0.0)):.2f})")
        if sig.contagion_penalty and sig.contagion_penalty > 0.3:
            reasons.append(f"Penalidade contagio aplicada: -{sig.contagion_penalty:.2f} (risco sistemico)")
        if sig.tail_penalty and sig.tail_penalty > 0.3:
            reasons.append(f"Penalidade tail risk: -{sig.tail_penalty:.2f}")
    if pos.sharpe_implied and pos.sharpe_implied > 0.5:
        reasons.append(f"Sharpe implícito atrativo: {pos.sharpe_implied:.2f}")
    return reasons or ["Selecionado pelo otimizador de Sharpe com constraints de alocacao"]


def _regime_effect(regime: str) -> str:
    return {
        "bear":    "Regime BEAR: portfolio defensivo — maxima 40% equities, minimo 30% bonds/commodities. Safe-havens injetados (SHY/IEF/BIL) para longs forcados.",
        "neutral": "Regime NEUTRAL: exposicao mista sem restricao adicional. Otimizador livre para maximizar Sharpe.",
        "bull":    "Regime BULL: equities ate 70% do portfolio. Bias para momentum risk-on.",
    }.get(regime, "Regime desconhecido.")


def _options_why(rec) -> str:
    strategy = rec.strategy
    iv_pct = rec.iv_percentile or 0.5
    if strategy == "put_spread":
        return (
            f"Put spread escolhido: IV a {iv_pct:.0%} do percentile historico — "
            f"spread reduz custo vs long put outright; posicao SHORT do ativo precisa de protecao em caso de rally."
        )
    elif strategy == "long_put":
        return (
            f"Long put direto: IV acima de 75%ile ({iv_pct:.0%}) — "
            f"IV muito rica para vender, preferivel comprar protecao outright."
        )
    elif strategy == "call_spread":
        return (
            f"Call spread: IV a {iv_pct:.0%}ile — custo reduzido vs call outright; "
            f"alavanca posicao LONG com risco limitado ao debit."
        )
    elif strategy == "long_call":
        return (
            f"Long call: IV barata ({iv_pct:.0%}ile) — opcao de alavancagem low-cost; "
            f"complementa posicao LONG."
        )
    elif strategy == "spy_hedge":
        return (
            f"SPY put spread: hedge de portfolio — cobre 50% do long exposure em queda de 5-10%; "
            f"custo ~1% do portfolio."
        )
    return "Estrategia selecionada pelo engine de opcoes."


def build_decision_log(
    portfolio,
    signals: dict,
    bundle,
    options_strategy=None,
) -> PortfolioDecisionLog:
    """Constrói o log de decisoes a partir dos resultados do pipeline."""
    from app.analysis.portfolio_optimizer import PortfolioResult
    from app.analysis.alpha_signals import AssetSignal

    por: PortfolioResult = portfolio
    bundle_date = str(getattr(bundle, "run_date", ""))
    run_id = getattr(bundle, "run_id", "unknown")

    # ── Regime ────────────────────────────────────────────────────────────────
    safe_havens = [p.ticker for p in por.positions
                   if p.ticker in ("SHY", "IEF", "BIL", "TLT") and p.direction == "long"]
    regime_tickers = [t for t in ("SPY", "QQQ", "IWM", "TLT", "GLD") if t in signals]
    regime_bull_avg = 0.0
    if regime_tickers:
        regime_bull_avg = sum(
            signals[t].regime_bull or 0.5 for t in regime_tickers
        ) / len(regime_tickers)

    regime_dec = RegimeDecision(
        regime=por.regime_mode,
        regime_bull_avg=regime_bull_avg,
        tickers_used=regime_tickers,
        effect=_regime_effect(por.regime_mode),
        safe_haven_injected=safe_havens,
    )

    # ── Signal decisions ───────────────────────────────────────────────────────
    selected_tickers = {p.ticker for p in por.positions}
    signals_considered = []
    signals_rejected = []

    for ticker, sig in sorted(signals.items(), key=lambda x: abs(x[1].composite), reverse=True):
        sd = SignalDecision(
            ticker=ticker,
            name=sig.name or ticker,
            direction=sig.direction,
            conviction=sig.conviction,
            composite=sig.composite,
            momentum_score=sig.momentum_score or 0.0,
            mean_rev_score=sig.mean_rev_score or 0.0,
            vol_edge_score=sig.vol_edge_score or 0.0,
            options_flow_score=sig.options_flow_score or 0.0,
            contagion_penalty=sig.contagion_penalty or 0.0,
            tail_penalty=sig.tail_penalty or 0.0,
            rationale=sig.rationale or [],
            dominant_signal=_dominant_signal(sig),
        )
        signals_considered.append(sd)
        if ticker not in selected_tickers and sig.direction != "neutral":
            # Razao de rejeicao
            reason = "nao passou no filtro do otimizador"
            if sig.conviction == "low":
                reason = f"baixa convicao (|composite|={abs(sig.composite):.3f} < 0.12)"
            elif sig.direction == "neutral":
                reason = "sinal neutro (composite entre -0.08 e +0.08)"
            signals_rejected.append({"ticker": ticker, "reason": reason, "composite": sig.composite})

    # ── Position decisions ─────────────────────────────────────────────────────
    position_decs = []
    for p in por.positions:
        sig = signals.get(p.ticker)
        why = _why_selected(p, sig)
        why_dir = _why_direction(sig) if sig else f"Injetado pelo regime {por.regime_mode} como hedge"
        pd = PositionDecision(
            ticker=p.ticker,
            name=p.name,
            direction=p.direction,
            conviction=p.conviction,
            allocation_pct=p.allocation_pct,
            allocation_usd=p.allocation_usd,
            expected_return_ann=p.expected_return_ann,
            risk_score=p.risk_score,
            sharpe_implied=p.sharpe_implied,
            why_selected=why,
            why_not_neutral=why_dir,
        )
        position_decs.append(pd)

    # ── Options decisions ──────────────────────────────────────────────────────
    opts_decs = []
    opts_total = 0.0
    if options_strategy:
        for rec in options_strategy.recommendations:
            od = OptionsDecision(
                ticker=rec.ticker,
                strategy=rec.strategy,
                rationale=rec.rationale,
                kelly_fraction=rec.kelly_fraction,
                iv_percentile=rec.iv_percentile,
                why_this_strategy=_options_why(rec),
            )
            opts_decs.append(od)
        opts_total = options_strategy.total_options_cost

    # ── Narrative ──────────────────────────────────────────────────────────────
    n_long  = sum(1 for p in por.positions if p.direction == "long")
    n_short = sum(1 for p in por.positions if p.direction == "short")
    top3 = [p.ticker for p in por.positions[:3]]

    narrative = (
        f"Regime {por.regime_mode.upper()} detectado "
        f"(P(bull) medio = {regime_bull_avg:.0%}). "
        f"Portfolio com {len(por.positions)} posicoes: "
        f"{n_long} long, {n_short} short. "
        f"Top holdings: {', '.join(top3)}. "
        f"E[R] anual = {por.expected_return_ann:+.1%}, "
        f"Vol = {por.portfolio_vol:.1%}, "
        f"Sharpe = {por.sharpe:.2f}. "
        f"{len(signals_rejected)} ativos rejeitados por baixa convicao. "
        f"{len(opts_decs)} estrategias de opcoes recomendadas "
        f"(custo total ~${opts_total:,.0f})."
    )

    return PortfolioDecisionLog(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        bundle_date=bundle_date,
        regime=regime_dec,
        signals_considered=signals_considered,
        signals_selected=list(selected_tickers),
        signals_rejected=signals_rejected[:20],  # top 20 rejections
        positions=position_decs,
        sharpe=por.sharpe,
        expected_return_ann=por.expected_return_ann,
        portfolio_vol=por.portfolio_vol,
        diversification=por.diversification_score,
        max_drawdown_est=por.max_drawdown_est,
        options_decisions=opts_decs,
        options_total_cost=opts_total,
        narrative=narrative,
    )


def save_decision_log(
    log: PortfolioDecisionLog,
    out_dir: Path | None = None,
) -> Path:
    """Salva o log em JSON e HTML."""
    if out_dir is None:
        from app.storage.paths import workspace
        out_dir = Path(getattr(workspace, "reports_dir", "."))

    json_path = out_dir / f"{log.run_id}_decision_log.json"
    html_path = out_dir / f"{log.run_id}_decision_log.html"

    # JSON
    def _ser(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return str(obj)

    log_dict = asdict(log)
    json_path.write_text(json.dumps(log_dict, indent=2, default=_ser), encoding="utf-8")

    # HTML
    html = _render_html(log)
    html_path.write_text(html, encoding="utf-8")

    _log.info("decision_log_saved", json=str(json_path), html=str(html_path))
    return html_path


def _render_html(log: PortfolioDecisionLog) -> str:
    regime_colors = {"bull": "#10b981", "neutral": "#f59e0b", "bear": "#ef4444"}
    rc = regime_colors.get(log.regime.regime, "#6b7280")
    dir_colors = {"long": "#10b981", "short": "#ef4444", "neutral": "#6b7280"}

    # Positions rows
    pos_rows = ""
    for p in log.positions:
        dc = dir_colors.get(p.direction, "#6b7280")
        why_html = "".join(f"<li>{w}</li>" for w in p.why_selected)
        pos_rows += f"""
        <tr>
          <td style="font-weight:700;color:#e5e7eb">{p.ticker}</td>
          <td style="font-size:11px;color:#9ca3af;max-width:120px">{p.name[:20]}</td>
          <td style="color:{dc};font-weight:700">{p.direction.upper()}</td>
          <td style="color:#e5e7eb">{abs(p.allocation_pct):.1%}</td>
          <td style="color:#e5e7eb">${abs(p.allocation_usd):,.0f}</td>
          <td style="color:{'#10b981' if p.expected_return_ann >= 0 else '#ef4444'}">{p.expected_return_ann:+.1%}</td>
          <td style="color:#9ca3af">{p.risk_score:.1%}</td>
          <td style="color:{'#10b981' if p.sharpe_implied > 0.5 else '#ef4444' if p.sharpe_implied < 0 else '#f59e0b'}">{p.sharpe_implied:.2f}</td>
          <td style="font-size:11px;color:#9ca3af;max-width:250px">
            <ul style="margin:0;padding-left:14px;line-height:1.5">{why_html}</ul>
            <div style="margin-top:4px;color:#6b7280;font-size:10px">{p.why_not_neutral[:120]}</div>
          </td>
        </tr>"""

    # Signals rows
    sig_rows = ""
    for s in log.signals_considered[:30]:
        dc = dir_colors.get(s.direction, "#6b7280")
        dom_color = {"momentum": "#3b82f6", "mean_rev": "#10b981", "vol_edge": "#f59e0b", "options_flow": "#8b5cf6"}.get(s.dominant_signal, "#9ca3af")
        selected_badge = '<span style="background:#10b98133;color:#10b981;font-size:10px;padding:1px 5px;border-radius:3px">SELECTED</span>' if s.ticker in log.signals_selected else ""
        sig_rows += f"""
        <tr>
          <td style="font-weight:700;color:{dc}">{s.ticker} {selected_badge}</td>
          <td style="color:{dc};font-size:11px">{s.direction}</td>
          <td style="color:#e5e7eb">{s.composite:+.3f}</td>
          <td style="color:#3b82f6">{s.momentum_score:+.2f}</td>
          <td style="color:#10b981">{s.mean_rev_score:+.2f}</td>
          <td style="color:#f59e0b">{s.vol_edge_score:+.2f}</td>
          <td style="color:#8b5cf6">{s.options_flow_score:+.2f}</td>
          <td style="color:#ef4444">{s.contagion_penalty:.2f}</td>
          <td style="color:{dom_color};font-size:11px">{s.dominant_signal}</td>
          <td style="font-size:10px;color:#9ca3af;max-width:200px">{' · '.join(s.rationale[:2])}</td>
        </tr>"""

    # Rejections
    rej_rows = ""
    for r in log.signals_rejected[:15]:
        dc = "#f87171" if r.get("composite", 0) < 0 else "#34d399"
        rej_rows += f"""
        <tr>
          <td style="color:#e5e7eb">{r['ticker']}</td>
          <td style="color:{dc}">{r.get('composite', 0):+.3f}</td>
          <td style="color:#9ca3af;font-size:11px">{r['reason']}</td>
        </tr>"""

    # Options rows
    opt_rows = ""
    for o in log.options_decisions:
        strat_colors = {
            "long_put": "#ef4444", "put_spread": "#f87171",
            "long_call": "#10b981", "call_spread": "#34d399",
            "spy_hedge": "#8b5cf6",
        }
        sc = strat_colors.get(o.strategy, "#9ca3af")
        kf = f"{o.kelly_fraction:.1%}" if o.kelly_fraction else "—"
        ivp = f"{o.iv_percentile:.0%}" if o.iv_percentile else "—"
        opt_rows += f"""
        <tr>
          <td style="font-weight:700;color:#e5e7eb">{o.ticker}</td>
          <td><span style="color:{sc};font-size:11px;font-weight:700">{o.strategy.upper()}</span></td>
          <td style="color:#9ca3af;font-size:11px">{ivp}</td>
          <td style="color:#f59e0b;font-size:11px">{kf}</td>
          <td style="font-size:11px;color:#9ca3af;max-width:280px">{o.why_this_strategy[:150]}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="pt">
<head>
<meta charset="UTF-8">
<title>Decision Log — {log.bundle_date}</title>
<style>
  :root {{ --bg:#0f1117;--surface:#1a1d27;--border:#2d3142;--text:#e5e7eb;--muted:#9ca3af; }}
  * {{ box-sizing:border-box;margin:0;padding:0 }}
  body {{ background:var(--bg);color:var(--text);font-family:'SF Mono','Consolas',monospace;font-size:13px;padding:24px }}
  h1 {{ font-size:18px;font-weight:700;margin-bottom:6px }}
  h2 {{ font-size:14px;font-weight:700;color:{rc};margin:24px 0 10px }}
  .card {{ background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:16px }}
  table {{ width:100%;border-collapse:collapse;font-size:12px }}
  th {{ text-align:left;padding:6px 8px;border-bottom:1px solid var(--border);color:var(--muted);font-weight:600 }}
  td {{ padding:5px 8px;border-bottom:1px solid #1f2937 }}
  tr:hover td {{ background:#1f2937 }}
  .narrative {{ background:#1a1d27;border-left:3px solid {rc};padding:12px 16px;border-radius:4px;line-height:1.7;color:#d1d5db }}
  .regime-chip {{ display:inline-block;background:{rc}22;color:{rc};border:1px solid {rc}44;padding:4px 12px;border-radius:6px;font-weight:700;font-size:14px }}
  .metric-grid {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:12px 0 }}
  .metric {{ background:#0d1117;border-radius:6px;padding:10px;text-align:center }}
  .metric-val {{ font-size:20px;font-weight:700 }}
  .metric-lbl {{ font-size:11px;color:var(--muted);margin-top:3px }}
</style>
</head>
<body>
<h1>Decision Log — Portfolio {log.bundle_date}</h1>
<div style="color:#6b7280;font-size:12px;margin-bottom:16px">Run: {log.run_id} · {log.timestamp[:19]}</div>

<!-- Narrative -->
<div class="narrative">{log.narrative}</div>

<!-- Regime -->
<h2>Regime Macro</h2>
<div class="card">
  <div style="margin-bottom:10px"><span class="regime-chip">{log.regime.regime.upper()}</span></div>
  <div style="color:#9ca3af;font-size:12px;line-height:1.6;margin-bottom:10px">{log.regime.effect}</div>
  <div style="font-size:12px">P(bull) medio: <span style="color:{rc};font-weight:700">{log.regime.regime_bull_avg:.0%}</span>
    — tickers referencia: {', '.join(log.regime.tickers_used)}
  </div>
  {f'<div style="margin-top:8px;font-size:12px;color:#10b981">Safe-havens injetados: {", ".join(log.regime.safe_haven_injected)}</div>' if log.regime.safe_haven_injected else ''}
</div>

<!-- Portfolio metrics -->
<h2>Metricas do Portfolio</h2>
<div class="metric-grid">
  <div class="metric"><div class="metric-val" style="color:{'#10b981' if log.expected_return_ann >= 0 else '#ef4444'}">{log.expected_return_ann:+.1%}</div><div class="metric-lbl">E[R] Anual</div></div>
  <div class="metric"><div class="metric-val" style="color:#9ca3af">{log.portfolio_vol:.1%}</div><div class="metric-lbl">Vol Anual</div></div>
  <div class="metric"><div class="metric-val" style="color:{'#10b981' if log.sharpe > 0.5 else '#f59e0b'}">{log.sharpe:.2f}</div><div class="metric-lbl">Sharpe</div></div>
  <div class="metric"><div class="metric-val" style="color:#3b82f6">{log.diversification:.2f}</div><div class="metric-lbl">Diversificacao</div></div>
  <div class="metric"><div class="metric-val" style="color:#ef4444">{log.max_drawdown_est:.1%}</div><div class="metric-lbl">Max DD Est.</div></div>
</div>

<!-- Positions with rationale -->
<h2>Posicoes — Por que cada ativo foi selecionado</h2>
<div class="card" style="overflow-x:auto">
<table>
  <thead>
    <tr><th>Ticker</th><th>Nome</th><th>Dir.</th><th>Alloc%</th><th>USD</th><th>E[R]</th><th>Vol</th><th>Sharpe</th><th>Raciocinio</th></tr>
  </thead>
  <tbody>{pos_rows or '<tr><td colspan="9" style="color:#6b7280;text-align:center;padding:20px">Sem posicoes</td></tr>'}</tbody>
</table>
</div>

<!-- All signals -->
<h2>Todos os Sinais Analisados (top 30 por |composite|)</h2>
<div class="card" style="overflow-x:auto">
<table>
  <thead>
    <tr><th>Ticker</th><th>Dir.</th><th>Composite</th><th style="color:#3b82f6">Momentum</th><th style="color:#10b981">MeanRev</th><th style="color:#f59e0b">VolEdge</th><th style="color:#8b5cf6">OptFlow</th><th style="color:#ef4444">Contagio</th><th>Dominante</th><th>Fatores</th></tr>
  </thead>
  <tbody>{sig_rows or '<tr><td colspan="10" style="color:#6b7280;text-align:center;padding:20px">Sem sinais</td></tr>'}</tbody>
</table>
</div>

<!-- Rejections -->
<h2>Ativos Rejeitados (nao viraram posicao)</h2>
<div class="card" style="overflow-x:auto">
<table>
  <thead><tr><th>Ticker</th><th>Composite</th><th>Motivo da Rejeicao</th></tr></thead>
  <tbody>{rej_rows or '<tr><td colspan="3" style="color:#6b7280;text-align:center;padding:20px">Nenhum rejeitado</td></tr>'}</tbody>
</table>
</div>

<!-- Options -->
<h2>Estrategias de Opcoes — Raciocinio</h2>
<div class="card" style="overflow-x:auto">
<table>
  <thead><tr><th>Ticker</th><th>Estrategia</th><th>IV %ile</th><th>Kelly f*</th><th>Por que essa estrategia?</th></tr></thead>
  <tbody>{opt_rows or '<tr><td colspan="5" style="color:#6b7280;text-align:center;padding:20px">Sem recomendacoes de opcoes</td></tr>'}</tbody>
</table>
</div>

</body>
</html>"""
