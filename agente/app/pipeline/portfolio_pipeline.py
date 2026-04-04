"""
Portfolio Pipeline — Orquestrador de Alocação Autônoma

Fluxo:
  1. Carrega bundle do dia (ou recebe como parâmetro)
  2. Roda análise de rede (MST + correlações)
  3. Coleta options + prob (reutiliza se já disponíveis)
  4. Computa sinais alpha (alpha_signals.compute_signals)
  5. Otimiza portfolio (portfolio_optimizer.optimize_portfolio)
  6. Gera HTML (flow_inspector.save_flow_inspector)
  7. Retorna (PortfolioResult, path_html)

Reutilizável tanto pelo CLI quanto pelo live loop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.audit.logger import get_logger

_log = get_logger("pipeline.portfolio")


def _mst_to_adj(mst_result: dict) -> dict[str, list[str]]:
    """Converte edges do MST em adjacency list. Suporta dict {from, to} e tuple/list."""
    adj: dict[str, list[str]] = {}
    for edge in mst_result.get("edges", []):
        if isinstance(edge, dict):
            t1, t2 = edge.get("from", ""), edge.get("to", "")
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            t1, t2 = edge[0], edge[1]
        else:
            continue
        if t1 and t2:
            adj.setdefault(t1, []).append(t2)
            adj.setdefault(t2, []).append(t1)
    return adj


def _corr_to_matrix(corr_dict: dict, tickers: list[str]) -> np.ndarray | None:
    """Converte corr_clean dict → np.ndarray alinhado com tickers."""
    if not corr_dict or not tickers:
        return None
    n = len(tickers)
    mat = np.eye(n)
    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            val = corr_dict.get(ti, {}).get(tj) or corr_dict.get(tj, {}).get(ti)
            if val is not None:
                mat[i, j] = mat[j, i] = float(val)
    return mat


def run_portfolio_pipeline(
    bundle: Any,
    options_map: dict | None = None,
    prob_map: dict | None = None,
    network_result: dict | None = None,
    budget: float = 100_000.0,
    regime_override: str | None = None,
    save_html: bool = True,
    live_mode: bool = False,
    out_dir: Path | None = None,
) -> tuple[Any, dict, Path | None]:
    """
    Roda o pipeline completo de alocação.

    Returns:
        (PortfolioResult, signals_dict, html_path)
    """
    market_prices: dict[str, Any] = {
        k: v for k, v in (bundle.market_prices or {}).items()
        if not k.startswith("__") and isinstance(v, dict) and v.get("price")
    }

    if not market_prices:
        _log.warning("no_market_prices_for_portfolio")
        from app.analysis.portfolio_optimizer import PortfolioResult
        return PortfolioResult(budget=budget, regime_mode="neutral"), {}, None

    # ── 1. Rede: MST + correlações ────────────────────────────────────────────
    if network_result is None:
        try:
            from app.analysis.network import analyze as net_analyze
            network_result = net_analyze(market_prices) or {}
        except Exception as exc:
            _log.warning("network_failed", error=str(exc))
            network_result = {}

    mst_data  = network_result.get("mst", {})
    rmt_data  = network_result.get("rmt", {})
    mst_adj   = _mst_to_adj(mst_data)
    tickers_ordered = list(market_prices.keys())
    corr_matrix = _corr_to_matrix(rmt_data.get("corr_clean", {}), tickers_ordered)

    # ── 2. Options (reutiliza se fornecido) ───────────────────────────────────
    if options_map is None:
        try:
            from app.providers.options import collect as options_collect
            options_map = options_collect() or {}
        except Exception as exc:
            _log.warning("options_failed", error=str(exc))
            options_map = {}

    # ── 3. Probabilistic (reutiliza se fornecido) ─────────────────────────────
    if prob_map is None:
        try:
            from app.analysis.probabilistic import analyze_from_registry
            prob_map = analyze_from_registry() or {}
        except Exception as exc:
            _log.warning("prob_failed", error=str(exc))
            prob_map = {}

    # ── 3b. Load options snapshot (Greeks Dashboard BBQ — mais recente disponível) ──
    options_snapshot = None
    try:
        from app.providers.options_store import options_store as _opt_store
        options_snapshot = _opt_store.load_latest()
        if options_snapshot:
            _log.info("options_snapshot_loaded",
                      ticker=options_snapshot.ticker,
                      ts=options_snapshot.ts,
                      gex=options_snapshot.gex_net_bn,
                      squeeze=options_snapshot.squeeze_score)
    except Exception as exc:
        _log.warning("options_snapshot_failed", error=str(exc))

    # ── 3c. Vol regime (enriquecido com dados BBQ se disponíveis) ─────────────
    vol_regime = None
    try:
        from app.analysis.vol_regime import compute_vol_regime
        vol_regime = compute_vol_regime(
            market_prices=market_prices,
            options_snapshot=options_snapshot,
        )
        _log.info("vol_regime", regime=vol_regime.regime, stress=round(vol_regime.stress_score, 3))
    except Exception as exc:
        _log.warning("vol_regime_failed", error=str(exc))

    # ── 3d. CTA positioning ───────────────────────────────────────────────────
    cta_result = None
    try:
        from app.providers.cta_positioning import compute_cta_positioning
        cta_result = compute_cta_positioning(tickers_extra=list(market_prices.keys())[:20])
    except Exception as exc:
        _log.warning("cta_positioning_failed", error=str(exc))

    # ── 3e. Relative Strength / RRG ──────────────────────────────────────────
    rrg_result = None
    try:
        from app.analysis.relative_strength import compute_relative_strength
        rrg_result = compute_relative_strength(
            tickers=list(market_prices.keys()),
            benchmark="SPY",
        )
        _log.info("rrg_done",
                  leading=rrg_result.leading[:3],
                  lagging=rrg_result.lagging[:3])
    except Exception as exc:
        _log.warning("rrg_failed", error=str(exc))

    # ── 3f. Shadow Flow / Dark Pool ───────────────────────────────────────────
    shadow_flow_result = None
    try:
        from app.providers.shadow_flow import collect_shadow_flow
        shadow_flow_result = collect_shadow_flow(
            tickers=list(market_prices.keys())[:25],
            market_prices=market_prices,
        )
    except Exception as exc:
        _log.warning("shadow_flow_failed", error=str(exc))

    # ── 3h. Narrative alpha (DeepVue themes + X sentiment) ───────────────────
    narrative_result = None
    try:
        from app.analysis.narrative_alpha import compute_narrative_alpha
        narrative_result = compute_narrative_alpha(
            bundle=bundle,
            known_tickers=set(market_prices.keys()),
        )
        _log.info("narrative_alpha",
                  long=narrative_result.top_narrative_long[:3],
                  short=narrative_result.top_narrative_short[:3],
                  deepvue_themes=len(narrative_result.deepvue_themes_parsed))
    except Exception as exc:
        _log.warning("narrative_alpha_failed", error=str(exc))

    # ── 3g. SwaggyStocks — WSB mentions + short squeeze list ─────────────────
    swaggy_result = None
    try:
        from app.providers.swaggy_stocks import collect as swaggy_collect
        swaggy_result = swaggy_collect(max_wsb=50, max_squeeze=30)
        _log.info("swaggy_done",
                  wsb=len(swaggy_result.wsb_mentions),
                  squeeze=len(swaggy_result.squeeze_candidates),
                  top=swaggy_result.top_mentions[:5])
    except Exception as exc:
        _log.warning("swaggy_failed", error=str(exc))

    # ── 3i. TradingView — Value Area / VWAP / zonas para UNIVERSO COMPLETO ─────
    # Roda antes do alpha_signals para que o zone_filter ajuste composite + conviction
    # antes da função objetivo do optimizer.
    # tv_map_universe = {ticker: snapshot} para todos os ativos no universo
    tv_map_universe: dict = {}
    zone_signals: dict = {}
    try:
        from app.providers.tradingview import collect_for_positions as tv_collect
        # Limita a 20 ativos para não ultrapassar ~60s de coleta (2.5s + 1s por ativo)
        _tv_universe = list(market_prices.keys())[:20]
        tv_map_universe = tv_collect(
            _tv_universe,
            layout="ultimate profile",
            timeframe="D",
        )
        if tv_map_universe:
            _log.info("tv_universe_collected", tickers=len(tv_map_universe))
    except Exception as exc:
        _log.warning("tv_universe_failed", error=str(exc))

    # ── 4. Alpha signals ──────────────────────────────────────────────────────
    from app.analysis.alpha_signals import compute_signals
    signals = compute_signals(
        market_prices=market_prices,
        prob_map=prob_map,
        options_map=options_map,
        mst_adj=mst_adj,
        corr_matrix=corr_matrix,
        tickers_ordered=tickers_ordered,
        vol_regime=vol_regime,
        cta_result=cta_result,
        rrg_result=rrg_result,
        shadow_flow=shadow_flow_result,
        narrative_result=narrative_result,
    )

    # ── 4b. TV Zone Filter — ajuste fino de composite + conviction ────────────
    # Aplica position_bias (delta no composite) e atualiza conviction
    # ANTES do optimizer para que as zonas TV entrem na função objetivo
    if tv_map_universe:
        try:
            from app.analysis.tv_zone_filter import (
                compute_zone_signals, apply_zone_signals_to_signals,
            )
            zone_signals = compute_zone_signals(tv_map_universe, signals)
            if zone_signals:
                signals = apply_zone_signals_to_signals(signals, zone_signals)
                _log.info(
                    "tv_zone_applied",
                    tickers=len(zone_signals),
                    ideal=sum(1 for z in zone_signals.values() if z.entry_quality == "ideal"),
                    avoid=sum(1 for z in zone_signals.values() if z.entry_quality == "avoid"),
                )
        except Exception as exc:
            _log.warning("tv_zone_filter_failed", error=str(exc))

    # ── 5. Portfolio optimization ─────────────────────────────────────────────
    from app.analysis.portfolio_optimizer import optimize_portfolio

    # Computa vol_options_regime antecipadamente (antes do optimizer)
    # para que GEX + IV ajustem alocações; desk_intel vai reusá-lo depois
    _vol_options_regime_early = None
    try:
        from app.analysis.vol_options_regime import compute_vol_options_regime
        _vol_options_regime_early = compute_vol_options_regime(options_snapshot, vol_regime)
    except Exception as exc:
        _log.warning("vol_options_regime_pre_failed", error=str(exc))

    portfolio = optimize_portfolio(
        signals=signals,
        market_prices=market_prices,
        budget=budget,
        regime_override=regime_override,
        vol_regime=vol_regime,
        vol_options_regime=_vol_options_regime_early,
        zone_signals=zone_signals or None,
    )

    # ── 5a. Earnings filter ───────────────────────────────────────────────────
    try:
        from app.analysis.earnings_filter import compute_event_risks, apply_event_scalars
        event_risks = compute_event_risks([p.ticker for p in portfolio.positions])
        portfolio.positions = apply_event_scalars(portfolio.positions, event_risks)
    except Exception as exc:
        _log.warning("earnings_filter_failed", error=str(exc))

    # ── 5b. Pairs trading ─────────────────────────────────────────────────────
    pairs_result = None
    try:
        from app.analysis.pairs_trading import compute_pairs_signals
        pairs_result = compute_pairs_signals(
            mst_adj=mst_adj,
            universe_tickers=list(market_prices.keys()),
            budget=budget,
        )
        if pairs_result.active_pairs:
            _log.info("pairs_found", n=len(pairs_result.active_pairs),
                      pairs=[(p.long_ticker, p.short_ticker) for p in pairs_result.active_pairs[:3]])
    except Exception as exc:
        _log.warning("pairs_failed", error=str(exc))

    # ── 5b. Options strategy ──────────────────────────────────────────────────
    options_strategy = None
    try:
        from app.analysis.options_strategy import compute_options_strategy
        options_strategy = compute_options_strategy(
            positions=portfolio.positions,
            market_prices=market_prices,
            budget=budget,
        )
    except Exception as exc:
        _log.warning("options_strategy_failed", error=str(exc))

    # ── 5c. TradingView enrichment — completa para posições não coletadas ──────
    tv_map: dict = dict(tv_map_universe)   # reutiliza coleta do universo
    try:
        from app.providers.tradingview import collect_for_positions
        position_tickers = [p.ticker for p in portfolio.positions]
        missing_tickers = [t for t in position_tickers if t not in tv_map]
        if missing_tickers:
            extra = collect_for_positions(
                missing_tickers,
                layout="ultimate profile",
                timeframe="D",
            )
            tv_map.update(extra)

        if tv_map:
            from app.analysis.technical_tv import enrich_positions_with_tv
            portfolio.positions = enrich_positions_with_tv(portfolio.positions, tv_map)
            _log.info("tv_enrichment_done", tickers=len(tv_map))
    except Exception as exc:
        _log.warning("tv_enrichment_failed", error=str(exc))

    # ── 6. HTML output ────────────────────────────────────────────────────────
    html_path: Path | None = None
    if save_html:
        try:
            from app.views.flow_inspector import save_flow_inspector
            html_path = save_flow_inspector(
                portfolio=portfolio,
                signals=signals,
                bundle=bundle,
                out_dir=out_dir,
                live_mode=live_mode,
                options_strategy=options_strategy,
                vol_regime=vol_regime,
                pairs_result=pairs_result,
                rrg_result=rrg_result,
                tv_map=tv_map or None,
            )
        except Exception as exc:
            _log.warning("flow_inspector_save_failed", error=str(exc))

    # ── 7. Persiste portfolio — abre/fecha posições com preços de entrada ────
    if save_html and portfolio.positions:
        try:
            from app.analysis.portfolio_tracker import open_portfolio
            open_portfolio(portfolio, market_prices)
        except Exception as exc:
            _log.warning("portfolio_save_failed", error=str(exc))

    # ── 8. Decision log ───────────────────────────────────────────────────────
    if save_html:
        try:
            from app.analysis.decision_log import build_decision_log, save_decision_log
            dec_log = build_decision_log(
                portfolio=portfolio,
                signals=signals,
                bundle=bundle,
                options_strategy=options_strategy,
            )
            save_decision_log(dec_log, out_dir=out_dir)
        except Exception as exc:
            _log.warning("decision_log_failed", error=str(exc))

    # Armazena resultados no portfolio para uso pelo live loop e Desk Radar
    portfolio._rrg_result   = rrg_result
    portfolio._pairs_result = pairs_result
    portfolio._signals      = signals   # dict[str, AssetSignal] para desk_radar

    # ── Desk Intelligence — motor de inferência regime-aware ──────────────────
    portfolio._desk_intel = None
    try:
        from app.analysis.desk_intelligence import compute_desk_intelligence
        portfolio._desk_intel = compute_desk_intelligence(
            signals=signals,
            rrg_result=rrg_result,
            vol_regime=vol_regime,
            narrative_result=narrative_result,
            cta_result=cta_result,
            shadow_flow=shadow_flow_result,
            options_map=options_map,
            network_result=network_result,
            market_prices=market_prices,
            swaggy_result=swaggy_result,
            options_snapshot=options_snapshot,
        )
    except Exception as exc:
        _log.warning("desk_intelligence_failed", error=str(exc))

    _log.info("portfolio_pipeline_done",
              n_signals=len(signals),
              n_positions=len(portfolio.positions),
              sharpe=portfolio.sharpe,
              expected_return=portfolio.expected_return_ann,
              regime=portfolio.regime_mode)

    return portfolio, signals, html_path
