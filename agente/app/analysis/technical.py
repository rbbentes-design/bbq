"""
Analysis: Technical Indicators

RSI, MACD e Bollinger Bands calculados via pandas/numpy puro.
Não requer bibliotecas de TA externas — apenas pandas + numpy.

Uso:
    from app.analysis.technical import analyze
    signals = analyze(market_prices)
"""

from __future__ import annotations

from typing import Any

from app.audit.logger import get_logger

_log = get_logger("analysis.technical")


# ── Indicadores individuais ────────────────────────────────────────────────────

def rsi(prices: list[float], period: int = 14) -> float | None:
    """
    Relative Strength Index (RSI).
    Retorna valor 0-100. None se dados insuficientes.
    """
    try:
        import pandas as pd
        s = pd.Series(prices)
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi_series = 100 - (100 / (1 + rs))
        val = rsi_series.iloc[-1]
        return round(float(val), 1) if not pd.isna(val) else None
    except Exception:
        return None


def macd(
    prices: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, float | None]:
    """
    MACD: retorna {macd_line, signal_line, histogram}.
    """
    try:
        import pandas as pd
        s = pd.Series(prices)
        ema_fast = s.ewm(span=fast, adjust=False).mean()
        ema_slow = s.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {
            "macd_line":   round(float(macd_line.iloc[-1]), 4),
            "signal_line": round(float(signal_line.iloc[-1]), 4),
            "histogram":   round(float(histogram.iloc[-1]), 4),
        }
    except Exception:
        return {"macd_line": None, "signal_line": None, "histogram": None}


def bollinger_bands(
    prices: list[float],
    period: int = 20,
    std_mult: float = 2.0,
) -> dict[str, float | None]:
    """
    Bollinger Bands: retorna {upper, middle, lower, pct_b}.
    pct_b indica posição do preço dentro das bandas (0=lower, 1=upper).
    """
    try:
        import pandas as pd
        s = pd.Series(prices)
        middle = s.rolling(period).mean()
        std = s.rolling(period).std()
        upper = middle + std_mult * std
        lower = middle - std_mult * std
        current = s.iloc[-1]
        band_width = float(upper.iloc[-1]) - float(lower.iloc[-1])
        pct_b = (current - float(lower.iloc[-1])) / band_width if band_width else 0.5
        return {
            "upper":  round(float(upper.iloc[-1]), 2),
            "middle": round(float(middle.iloc[-1]), 2),
            "lower":  round(float(lower.iloc[-1]), 2),
            "pct_b":  round(pct_b, 3),
        }
    except Exception:
        return {"upper": None, "middle": None, "lower": None, "pct_b": None}


def _signal_label(rsi_val: float | None, macd_dict: dict, bb_dict: dict) -> str:
    """Classifica o sinal técnico consolidado."""
    signals = []
    if rsi_val is not None:
        if rsi_val < 30:
            signals.append("oversold")
        elif rsi_val > 70:
            signals.append("overbought")
    if macd_dict.get("histogram") is not None:
        signals.append("bullish_momentum" if macd_dict["histogram"] > 0 else "bearish_momentum")
    pct_b = bb_dict.get("pct_b")
    if pct_b is not None:
        if pct_b > 1.0:
            signals.append("above_upper_band")
        elif pct_b < 0.0:
            signals.append("below_lower_band")
    return ", ".join(signals) if signals else "neutral"


# ── Análise por ticker a partir de histórico yfinance ─────────────────────────

def analyze_from_prices(market_prices_raw: dict[str, Any]) -> dict[str, dict]:
    """
    Calcula indicadores técnicos para cada ticker usando histórico yfinance.

    Args:
        market_prices_raw: output de market_prices.collect()
                           (não contém série histórica, apenas snapshot)

    Returns:
        dict {sym: {rsi, macd, bollinger, signal}}

    Nota: Busca histórico de 60 dias do yfinance para calcular os indicadores.
    """
    try:
        import yfinance as yf
    except ImportError:
        _log.warning("yfinance_not_installed")
        return {}

    results: dict[str, dict] = {}
    for sym in market_prices_raw:
        try:
            hist = yf.Ticker(sym).history(period="60d", auto_adjust=True)
            if hist.empty or len(hist) < 26:
                continue
            closes = hist["Close"].tolist()
            r = rsi(closes)
            m = macd(closes)
            bb = bollinger_bands(closes)
            results[sym] = {
                "rsi": r,
                "macd": m,
                "bollinger": bb,
                "signal": _signal_label(r, m, bb),
            }
        except Exception as exc:
            _log.debug("technical_ticker_error", sym=sym, error=str(exc))

    _log.info("technical_analysis_done", tickers=len(results))
    return results


def format_summary(technical: dict[str, dict], market_prices: dict[str, Any]) -> str:
    """Formata indicadores técnicos para inclusão no contexto do LLM."""
    if not technical:
        return ""
    lines = ["=== INDICADORES TÉCNICOS ==="]
    for sym, ind in technical.items():
        name = market_prices.get(sym, {}).get("name", sym)
        rsi_val = ind.get("rsi")
        signal = ind.get("signal", "")
        bb = ind.get("bollinger", {})
        pct_b = bb.get("pct_b")

        rsi_str = f"RSI {rsi_val}" if rsi_val is not None else ""
        bb_str = f"%B {pct_b:.2f}" if pct_b is not None else ""
        parts = [p for p in [rsi_str, bb_str, signal] if p]
        lines.append(f"  {name}: {' | '.join(parts)}")
    return "\n".join(lines)
