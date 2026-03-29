"""
Provider: Market Prices

Coleta cotações, retornos diários, semanais e YTD de ativos-chave via yfinance.
Requer: pip install yfinance
"""

from __future__ import annotations

from typing import Any

from app.audit.logger import get_logger

_log = get_logger("providers.market_prices")

SOURCE_NAME = "market_prices"

# Tickers padrão — cobre equities, bonds, commodities, cripto, vol e dólar
DEFAULT_TICKERS: dict[str, str] = {
    "^GSPC":    "S&P 500",
    "^NDX":     "Nasdaq 100",
    "^RUT":     "Russell 2000",
    "TLT":      "Treasury 20yr (TLT)",
    "HYG":      "High Yield (HYG)",
    "GLD":      "Gold (GLD)",
    "CL=F":     "WTI Crude Oil",
    "BTC-USD":  "Bitcoin",
    "DX-Y.NYB": "US Dollar Index",
    "^VIX":     "VIX",
}


def collect(
    tickers: dict[str, str] | None = None,
    period: str = "5d",
    include_ytd: bool = True,
) -> dict[str, Any]:
    """
    Retorna preços e retornos para os tickers configurados.

    Args:
        tickers: dict {symbol: friendly_name}. None usa DEFAULT_TICKERS.
        period:  período para cálculo de retorno semanal (ex: "5d", "1wk").
        include_ytd: se True, busca retorno YTD separadamente.

    Returns:
        {
          "^GSPC": {
            "name": "S&P 500",
            "price": 5100.23,
            "daily_return": -0.0122,
            "weekly_return": -0.034,
            "ytd_return": -0.085,   # se include_ytd=True
          },
          ...
        }
    """
    try:
        import yfinance as yf
    except ImportError:
        _log.warning("yfinance_not_installed", hint="pip install yfinance")
        return {}

    ticker_map = tickers or DEFAULT_TICKERS
    results: dict[str, Any] = {}

    for sym, name in ticker_map.items():
        try:
            t = yf.Ticker(sym)
            hist = t.history(period=period, auto_adjust=True)
            if hist.empty or len(hist) < 2:
                _log.debug("ticker_no_data", sym=sym)
                continue

            price = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[-2])
            week_open = float(hist["Close"].iloc[0])

            daily = (price - prev_close) / prev_close if prev_close else 0.0
            weekly = (price - week_open) / week_open if week_open else 0.0

            entry: dict[str, Any] = {
                "name": name,
                "price": round(price, 2),
                "daily_return": round(daily, 4),
                "weekly_return": round(weekly, 4),
            }

            if include_ytd:
                try:
                    ytd_hist = t.history(start="2026-01-01", auto_adjust=True)
                    if not ytd_hist.empty and len(ytd_hist) > 1:
                        ytd_open = float(ytd_hist["Close"].iloc[0])
                        ytd_close = float(ytd_hist["Close"].iloc[-1])
                        entry["ytd_return"] = round(
                            (ytd_close - ytd_open) / ytd_open if ytd_open else 0.0, 4
                        )
                except Exception:
                    pass

            results[sym] = entry
            _log.debug("ticker_ok", sym=sym, price=price)

        except Exception as exc:
            _log.debug("ticker_error", sym=sym, error=str(exc))

    _log.info("market_prices_done", tickers=len(results))
    return results


def format_summary(prices: dict[str, Any]) -> str:
    """Formata retorno legível para inclusão no contexto do LLM."""
    if not prices:
        return ""
    lines = ["=== PREÇOS E RETORNOS DE MERCADO ==="]
    for sym, d in prices.items():
        name = d.get("name", sym)
        price = d.get("price", "N/A")
        d1 = d.get("daily_return")
        w1 = d.get("weekly_return")
        ytd = d.get("ytd_return")

        d1_str = f"{d1:+.1%}" if d1 is not None else ""
        w1_str = f"{w1:+.1%}" if w1 is not None else ""
        ytd_str = f"YTD {ytd:+.1%}" if ytd is not None else ""

        parts = [p for p in [d1_str, w1_str, ytd_str] if p]
        lines.append(f"  {name}: ${price}  {' | '.join(parts)}")
    return "\n".join(lines)
