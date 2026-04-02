"""
Provider: Market Prices

Coleta cotações, retornos diários, semanais e YTD de ativos-chave.

Fonte primária : IBKR (TWS/Gateway via ib_insync) — quando disponível.
Fonte de fallback: yfinance — usado quando IBKR indisponível ou ticker ausente.

Tickers são extraídos dinamicamente do node_registry (todos os nós com ticker).
DEFAULT_TICKERS é mantido apenas como cobertura mínima de emergência.
"""

from __future__ import annotations

from typing import Any

from app.audit.logger import get_logger

_log = get_logger("providers.market_prices")

SOURCE_NAME = "market_prices"

# Cobertura mínima de emergência (usado quando node_registry falhar)
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

# Ano de início para cálculo YTD — atualiza automaticamente via datetime
import datetime as _dt
_YTD_START = f"{_dt.date.today().year}-01-01"


def _registry_tickers() -> dict[str, str]:
    """Extrai {ticker: label} de todos os nós do node_registry que tenham ticker."""
    try:
        from app.desk.node_registry import NODES
        result: dict[str, str] = {}
        for node in NODES.values():
            tk = node.get("ticker") or node.get("meta", {}).get("yf_symbol")
            if tk and isinstance(tk, str):
                result[tk] = node.get("label", tk)
        _log.debug("registry_tickers_loaded", count=len(result))
        return result
    except Exception as exc:
        _log.warning("registry_tickers_failed", error=str(exc))
        return {}


def _collect_ibkr(
    ticker_map: dict[str, str],
    include_ytd: bool,
) -> dict[str, Any]:
    """Tenta coletar via IBKR. Retorna {} se IBKR indisponível."""
    try:
        from app.providers import ibkr
        if not ibkr.is_available():
            _log.info("ibkr_unavailable", hint="TWS/Gateway não detectado em 7497")
            return {}
        _log.info("ibkr_collecting", tickers=len(ticker_map))
        return ibkr.collect(list(ticker_map.keys()), include_ytd=include_ytd)
    except Exception as exc:
        _log.warning("ibkr_collect_failed", error=str(exc))
        return {}


def _collect_yfinance(
    ticker_map: dict[str, str],
    period: str,
    include_ytd: bool,
) -> dict[str, Any]:
    """Coleta via yfinance em batch. Fallback confiável."""
    try:
        import yfinance as yf
    except ImportError:
        _log.warning("yfinance_not_installed", hint="pip install yfinance")
        return {}

    symbols = list(ticker_map.keys())
    results: dict[str, Any] = {}

    # Batch download — muito mais rápido que Ticker-por-Ticker
    try:
        raw = yf.download(
            symbols,
            period=period if not include_ytd else "1y",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        close = raw["Close"] if "Close" in raw.columns else raw

        # YTD window
        ytd_start_ts = _dt.datetime.strptime(_YTD_START, "%Y-%m-%d")

        for sym in symbols:
            try:
                if sym not in close.columns:
                    continue
                s = close[sym].dropna()
                if len(s) < 2:
                    continue

                price       = float(s.iloc[-1])
                prev_close  = float(s.iloc[-2])
                week_slice  = s.iloc[-6:] if len(s) >= 6 else s
                week_open   = float(week_slice.iloc[0])

                daily  = (price - prev_close) / prev_close if prev_close else 0.0
                weekly = (price - week_open)  / week_open  if week_open  else 0.0

                entry: dict[str, Any] = {
                    "name":          ticker_map.get(sym, sym),
                    "price":         round(price, 4),
                    "daily_return":  round(daily,  4),
                    "weekly_return": round(weekly, 4),
                }

                if include_ytd:
                    ytd_slice = s[s.index >= ytd_start_ts]
                    if len(ytd_slice) >= 2:
                        y0 = float(ytd_slice.iloc[0])
                        yn = float(ytd_slice.iloc[-1])
                        entry["ytd_return"] = round((yn - y0) / y0 if y0 else 0.0, 4)

                results[sym] = entry
                _log.debug("yf_ticker_ok", sym=sym, price=price)

            except Exception as exc:
                _log.debug("yf_ticker_error", sym=sym, error=str(exc))

    except Exception as exc:
        _log.warning("yf_batch_failed", error=str(exc))
        # Fallback individual se batch falhar
        for sym, name in ticker_map.items():
            try:
                t = yf.Ticker(sym)
                hist = t.history(period=period, auto_adjust=True)
                if hist.empty or len(hist) < 2:
                    continue
                price      = float(hist["Close"].iloc[-1])
                prev_close = float(hist["Close"].iloc[-2])
                week_open  = float(hist["Close"].iloc[0])
                daily  = (price - prev_close) / prev_close if prev_close else 0.0
                weekly = (price - week_open)  / week_open  if week_open  else 0.0
                entry = {
                    "name": name,
                    "price": round(price, 2),
                    "daily_return":  round(daily,  4),
                    "weekly_return": round(weekly, 4),
                }
                if include_ytd:
                    try:
                        ytd_hist = t.history(start=_YTD_START, auto_adjust=True)
                        if not ytd_hist.empty and len(ytd_hist) > 1:
                            y0 = float(ytd_hist["Close"].iloc[0])
                            yn = float(ytd_hist["Close"].iloc[-1])
                            entry["ytd_return"] = round((yn - y0) / y0 if y0 else 0.0, 4)
                    except Exception:
                        pass
                results[sym] = entry
            except Exception:
                pass

    _log.info("yf_collect_done", tickers=len(results))
    return results


def collect(
    tickers: dict[str, str] | None = None,
    period: str = "5d",
    include_ytd: bool = True,
) -> dict[str, Any]:
    """
    Retorna preços e retornos para os tickers configurados.

    Ordem de prioridade:
      1. IBKR (TWS em localhost:7497) — dados precisos, intra-day
      2. yfinance batch — fallback quando IBKR ausente
      3. DEFAULT_TICKERS — cobertura mínima de emergência

    Args:
        tickers:     dict {symbol: friendly_name}. None = node_registry + DEFAULT_TICKERS.
        period:      período para cálculo de retorno semanal (ex: "5d").
        include_ytd: se True, calcula retorno desde 1-Jan do ano corrente.

    Returns:
        {
          "^GSPC": {
            "name":          "S&P 500",
            "price":         5100.23,
            "daily_return":  -0.0122,
            "weekly_return": -0.034,
            "ytd_return":    -0.085,   # se include_ytd=True
          },
          ...
        }
    """
    # Monta mapa de tickers: explícito > registry > default
    if tickers is not None:
        ticker_map = tickers
    else:
        ticker_map = {**DEFAULT_TICKERS, **_registry_tickers()}

    _log.info("market_prices_start", tickers=len(ticker_map), ibkr_primary=True)

    # ── IBKR — fonte primária e única (BQL via Bloomberg não disponível aqui) ───
    results = _collect_ibkr(ticker_map, include_ytd)

    if not results:
        _log.warning("market_prices_empty", hint="IBKR indisponível e yfinance desabilitado — sem dados de preço")

    _log.info("market_prices_done", total=len(results), from_registry=len(ticker_map))
    return results


def format_summary(prices: dict[str, Any]) -> str:
    """Formata retorno legível para inclusão no contexto do LLM."""
    if not prices:
        return ""
    lines = ["=== PREÇOS E RETORNOS DE MERCADO ==="]
    for sym, d in prices.items():
        name  = d.get("name", sym)
        price = d.get("price", "N/A")
        d1    = d.get("daily_return")
        w1    = d.get("weekly_return")
        ytd   = d.get("ytd_return")

        d1_str  = f"{d1:+.1%}"      if d1  is not None else ""
        w1_str  = f"{w1:+.1%}"      if w1  is not None else ""
        ytd_str = f"YTD {ytd:+.1%}" if ytd is not None else ""

        parts = [p for p in [d1_str, w1_str, ytd_str] if p]
        lines.append(f"  {name}: ${price}  {' | '.join(parts)}")
    return "\n".join(lines)
