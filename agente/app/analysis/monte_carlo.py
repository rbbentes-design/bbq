"""
Analysis: Monte Carlo Simulation

Simula distribuição de retornos futuros com Monte Carlo (GBM).
Gera confidence intervals, fan charts e probabilidades de cenários.

Requer: numpy, pandas

Uso:
    from app.analysis.monte_carlo import simulate, format_summary
    result = simulate(prices, days=20, n_paths=1000)
"""

from __future__ import annotations

from typing import Any

from app.audit.logger import get_logger

_log = get_logger("analysis.monte_carlo")


def simulate(
    prices: list[float],
    days: int = 20,
    n_paths: int = 1000,
    confidence_levels: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95),
) -> dict[str, Any]:
    """
    Simulação Monte Carlo por Geometric Brownian Motion.

    Args:
        prices:             Série histórica de preços.
        days:               Horizonte de projeção (dias úteis).
        n_paths:            Número de simulações.
        confidence_levels:  Percentis a calcular.

    Returns:
        {
          "current_price": float,
          "horizon_days": int,
          "paths_count": int,
          "percentiles": {
            "p5": [lista de preços no percentil 5],
            "p50": [...],
            ...
          },
          "final_distribution": {
            "mean": float,
            "std": float,
            "p5": float,
            "p95": float,
          },
          "prob_up": float,   # probabilidade de retorno positivo
          "prob_up_5pct": float,
        }
    """
    try:
        import numpy as np
        import pandas as pd

        s = np.array(prices, dtype=float)
        log_returns = np.diff(np.log(s))
        mu = log_returns.mean()
        sigma = log_returns.std()

        rng = np.random.default_rng(seed=42)
        dt = 1
        shocks = rng.normal(
            loc=(mu - 0.5 * sigma ** 2) * dt,
            scale=sigma * np.sqrt(dt),
            size=(n_paths, days),
        )
        # Acumula retornos log
        cum_log = np.cumsum(shocks, axis=1)
        # Paths de preços
        paths = s[-1] * np.exp(cum_log)  # shape: (n_paths, days)

        # Percentis ao longo do tempo
        perc_labels = [f"p{int(c * 100)}" for c in confidence_levels]
        percentiles_over_time: dict[str, list[float]] = {}
        for label, cl in zip(perc_labels, confidence_levels):
            percentiles_over_time[label] = [
                round(float(np.percentile(paths[:, d], cl * 100)), 2)
                for d in range(days)
            ]

        # Distribuição final
        final = paths[:, -1]
        final_dist = {
            "mean":  round(float(final.mean()), 2),
            "std":   round(float(final.std()), 2),
            "p5":    round(float(np.percentile(final, 5)), 2),
            "p25":   round(float(np.percentile(final, 25)), 2),
            "p75":   round(float(np.percentile(final, 75)), 2),
            "p95":   round(float(np.percentile(final, 95)), 2),
        }
        prob_up = float((final > s[-1]).mean())
        prob_up5 = float((final > s[-1] * 1.05).mean())
        prob_down5 = float((final < s[-1] * 0.95).mean())

        return {
            "current_price":      round(float(s[-1]), 2),
            "horizon_days":       days,
            "paths_count":        n_paths,
            "mu_daily":           round(float(mu), 6),
            "sigma_daily":        round(float(sigma), 6),
            "percentiles":        percentiles_over_time,
            "final_distribution": final_dist,
            "prob_up":            round(prob_up, 3),
            "prob_up_5pct":       round(prob_up5, 3),
            "prob_down_5pct":     round(prob_down5, 3),
        }

    except Exception as exc:
        _log.warning("monte_carlo_error", error=str(exc))
        return {}


def run_for_portfolio(
    market_prices_raw: dict[str, Any],
    days: int = 20,
    n_paths: int = 500,
) -> dict[str, Any]:
    """
    Roda Monte Carlo para cada ticker do portfolio.

    Args:
        market_prices_raw: output de market_prices.collect()
        days:  horizonte
        n_paths: simulações por ticker

    Returns:
        {sym: simulate_result, ...}
    """
    try:
        import yfinance as yf
    except ImportError:
        _log.warning("yfinance_not_installed")
        return {}

    results: dict[str, Any] = {}
    for sym in market_prices_raw:
        try:
            hist = yf.Ticker(sym).history(period="60d", auto_adjust=True)
            if hist.empty or len(hist) < 30:
                continue
            prices = hist["Close"].tolist()
            mc = simulate(prices, days=days, n_paths=n_paths)
            if mc:
                results[sym] = mc
        except Exception as exc:
            _log.debug("mc_ticker_error", sym=sym, error=str(exc))

    _log.info("monte_carlo_portfolio_done", tickers=len(results))
    return results


def format_summary(mc_results: dict[str, Any], market_prices: dict[str, Any]) -> str:
    """Formata projeções Monte Carlo para inclusão no contexto do LLM."""
    if not mc_results:
        return ""
    lines = ["=== PROJEÇÃO MONTE CARLO (20 dias úteis) ==="]
    for sym, mc in mc_results.items():
        name = market_prices.get(sym, {}).get("name", sym)
        fd = mc.get("final_distribution", {})
        prob_up = mc.get("prob_up", 0)
        prob_up5 = mc.get("prob_up_5pct", 0)
        prob_down5 = mc.get("prob_down_5pct", 0)
        lines.append(
            f"  {name}: p5={fd.get('p5', 'N/A')} | p50={fd.get('mean', 'N/A')} | p95={fd.get('p95', 'N/A')} "
            f"| P(alta)={prob_up:.0%} | P(+5%)={prob_up5:.0%} | P(-5%)={prob_down5:.0%}"
        )
    return "\n".join(lines)
