"""
Analysis: Risk Metrics

VaR histórico, CVaR, Max Drawdown, Correlação e Sharpe Ratio
calculados com pandas/numpy a partir de dados de mercado.

Uso:
    from app.analysis.risk import analyze_portfolio
    metrics = analyze_portfolio(market_prices)
"""

from __future__ import annotations

from typing import Any

from app.audit.logger import get_logger

_log = get_logger("analysis.risk")


def historical_var(returns: list[float], confidence: float = 0.95) -> float:
    """Value at Risk histórico (percentil de perdas)."""
    try:
        import numpy as np
        arr = np.array(returns)
        return float(np.percentile(arr, (1 - confidence) * 100))
    except Exception:
        return 0.0


def cvar(returns: list[float], confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall) — média das perdas além do VaR."""
    try:
        import numpy as np
        arr = np.array(returns)
        var = historical_var(returns, confidence)
        tail = arr[arr <= var]
        return float(tail.mean()) if len(tail) > 0 else var
    except Exception:
        return 0.0


def max_drawdown(prices: list[float]) -> float:
    """Maximum Drawdown — queda máxima de pico a vale."""
    try:
        import numpy as np
        arr = np.array(prices)
        peak = np.maximum.accumulate(arr)
        drawdown = (arr - peak) / peak
        return float(drawdown.min())
    except Exception:
        return 0.0


def sharpe_ratio(returns: list[float], risk_free_daily: float = 0.0001) -> float:
    """Sharpe Ratio anualizado (252 dias de trading)."""
    try:
        import numpy as np
        arr = np.array(returns)
        excess = arr - risk_free_daily
        if excess.std() == 0:
            return 0.0
        return float((excess.mean() / excess.std()) * (252 ** 0.5))
    except Exception:
        return 0.0


def correlation_matrix(series_dict: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Matriz de correlação entre séries de retornos."""
    try:
        import pandas as pd
        df = pd.DataFrame(series_dict)
        corr = df.corr()
        return {
            col: {row: round(float(corr.loc[row, col]), 3) for row in corr.index}
            for col in corr.columns
        }
    except Exception:
        return {}


def analyze_portfolio(market_prices_raw: dict[str, Any]) -> dict[str, Any]:
    """
    Calcula métricas de risco para cada ticker e matriz de correlação.

    Args:
        market_prices_raw: output de market_prices.collect()

    Returns:
        {
          "tickers": {
            "^GSPC": {var_95, cvar_95, max_drawdown, sharpe},
            ...
          },
          "correlations": {  ...matriz de correlação...  }
        }
    """
    try:
        import yfinance as yf
    except ImportError:
        _log.warning("yfinance_not_installed")
        return {}

    ticker_returns: dict[str, list[float]] = {}
    ticker_prices: dict[str, list[float]] = {}

    for sym in market_prices_raw:
        try:
            hist = yf.Ticker(sym).history(period="60d", auto_adjust=True)
            if hist.empty or len(hist) < 10:
                continue
            closes = hist["Close"].tolist()
            rets = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
            ticker_returns[sym] = rets
            ticker_prices[sym] = closes
        except Exception as exc:
            _log.debug("risk_ticker_error", sym=sym, error=str(exc))

    tickers_metrics: dict[str, dict] = {}
    for sym, rets in ticker_returns.items():
        tickers_metrics[sym] = {
            "var_95":       round(historical_var(rets, 0.95), 4),
            "cvar_95":      round(cvar(rets, 0.95), 4),
            "max_drawdown": round(max_drawdown(ticker_prices[sym]), 4),
            "sharpe":       round(sharpe_ratio(rets), 2),
        }

    corr = correlation_matrix(ticker_returns) if len(ticker_returns) > 1 else {}

    _log.info("risk_analysis_done", tickers=len(tickers_metrics))
    return {"tickers": tickers_metrics, "correlations": corr}


def format_summary(risk: dict[str, Any], market_prices: dict[str, Any]) -> str:
    """Formata métricas de risco para inclusão no contexto do LLM."""
    tickers = risk.get("tickers", {})
    if not tickers:
        return ""
    lines = ["=== MÉTRICAS DE RISCO (60d) ==="]
    for sym, m in tickers.items():
        name = market_prices.get(sym, {}).get("name", sym)
        lines.append(
            f"  {name}: VaR95={m['var_95']:+.1%} | CVaR={m['cvar_95']:+.1%} "
            f"| MaxDD={m['max_drawdown']:+.1%} | Sharpe={m['sharpe']:.2f}"
        )
    return "\n".join(lines)
