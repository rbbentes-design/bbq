"""
Provider: Market Prices Live

Refresh rápido de preços para o loop live (agente live).
Não recalcula YTD/weekly — atualiza só price e daily_return no bundle existente.

Prioridade:
  1. IBKR reqMktData snapshot  (~1 tick delay, requer TWS ativo)
  2. yfinance fast_info         (15min delay, sem histórico)
"""

from __future__ import annotations

import datetime as _dt
from datetime import timezone as _tz
from typing import Any

# Tickers yfinance não consegue resolver — prefixos inválidos ou interna IBKR
_YF_INVALID_PREFIXES = ("$",)
# Máximo de workers paralelos
_MAX_WORKERS = 15

from app.audit.logger import get_logger

_log = get_logger("providers.market_prices_live")


def _from_ibkr(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """Snapshot via IBKR reqMktData. Retorna {} se indisponível."""
    try:
        from app.providers import ibkr
        if not ibkr.is_available():
            return {}
        raw = ibkr.snapshot(tickers)
        results: dict[str, dict[str, Any]] = {}
        for sym, d in raw.items():
            price = d.get("last")
            if not price:
                continue
            results[sym] = {
                "price":  round(float(price), 4),
                "source": "ibkr_live",
            }
        _log.info("ibkr_live_ok", tickers=len(results))
        return results
    except Exception as exc:
        _log.warning("ibkr_live_error", error=str(exc))
        return {}


def _from_yfinance_fast(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """
    Snapshot via yfinance.
    Estratégia:
      1. yf.download() em batch (1d) — mais robusto para crumb
      2. Fallback individual com fast_info para o que sobrou
    """
    try:
        import yfinance as yf
    except ImportError:
        return {}

    # Filtra tickers inválidos para yfinance (ex: $CADE de formato IBKR)
    valid = [s for s in tickers if not any(s.startswith(p) for p in _YF_INVALID_PREFIXES)]
    if not valid:
        return {}

    results: dict[str, dict[str, Any]] = {}

    # ── Caminho 1: batch download (mais estável, lida melhor com crumb) ───────
    try:
        raw = yf.download(
            valid,
            period="2d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if not raw.empty:
            close = raw["Close"] if "Close" in raw.columns else raw
            for sym in valid:
                try:
                    col = close[sym] if sym in close.columns else None
                    if col is None:
                        continue
                    s = col.dropna()
                    if len(s) < 1:
                        continue
                    price = float(s.iloc[-1])
                    prev  = float(s.iloc[-2]) if len(s) >= 2 else price
                    if price <= 0:
                        continue
                    daily = (price - prev) / prev if prev > 0 else 0.0
                    results[sym] = {
                        "price":        round(price, 4),
                        "daily_return": round(daily, 4),
                        "source":       "yfinance_fast",
                    }
                except Exception:
                    pass
    except Exception as exc:
        err_str = str(exc)
        if "401" in err_str or "Crumb" in err_str or "Unauthorized" in err_str:
            _log.warning("yf_crumb_batch", hint="crumb expirado no batch — tenta individual")
        else:
            _log.warning("yf_batch_failed", error=err_str[:120])

    # ── Caminho 2: fast_info individual para o que não veio no batch ──────────
    missing = [s for s in valid if s not in results]
    if missing:
        from concurrent.futures import ThreadPoolExecutor

        def _fetch_one(sym: str) -> tuple[str, dict | None]:
            try:
                fi = yf.Ticker(sym).fast_info
                price = getattr(fi, "last_price", None) or getattr(fi, "previous_close", None)
                prev  = getattr(fi, "previous_close", None)
                if not price or float(price) <= 0:
                    return sym, None
                daily = (float(price) - float(prev)) / float(prev) if prev and float(prev) > 0 else 0.0
                return sym, {
                    "price":        round(float(price), 4),
                    "daily_return": round(daily, 4),
                    "source":       "yfinance_fast",
                }
            except Exception as exc:
                err_str = str(exc)
                if "401" in err_str or "Crumb" in err_str or "Unauthorized" in err_str:
                    _log.debug("yf_crumb_individual", sym=sym)
                return sym, None

        try:
            with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
                for sym, d in [f.result() for f in [pool.submit(_fetch_one, s) for s in missing]]:
                    if d:
                        results[sym] = d
        except Exception as exc:
            _log.warning("yf_fast_pool_error", error=str(exc))

    _log.info("yf_fast_ok", tickers=len(results), of=len(valid))
    return results


def refresh(
    bundle_prices: dict[str, Any],
) -> dict[str, Any]:
    """
    Atualiza price + daily_return no dict de preços do bundle.

    Mantém weekly_return, ytd_return, name e outros campos do bundle original.
    Retorna o dict atualizado (in-place também).

    Args:
        bundle_prices: bundle.market_prices (modificado in-place)

    Returns:
        dict atualizado com timestamp
    """
    tickers = [k for k in bundle_prices.keys() if not k.startswith("__")]
    _log.info("live_refresh_start", tickers=len(tickers))

    # 1. Bloomberg CSV (BQuant export a cada 3min) — fonte primária
    fresh: dict[str, dict[str, Any]] = {}
    try:
        from app.providers.bql_csv import load_prices
        bbg = load_prices()
        if bbg:
            for sym, d in bbg.items():
                if sym in tickers and "price" in d:
                    fresh[sym] = {**d, "source": "bloomberg_csv"}
            _log.info("live_bloomberg_csv", tickers=len(fresh))
    except Exception as exc:
        _log.warning("live_bloomberg_csv_error", error=str(exc))

    # 2. IBKR para o que faltou
    missing = [t for t in tickers if t not in fresh]
    if missing:
        fresh.update(_from_ibkr(missing))

    # 3. yfinance fast_info para o que ainda faltou
    missing = [t for t in tickers if t not in fresh]
    if missing:
        fresh.update(_from_yfinance_fast(missing))

    # Mescla: só sobrescreve price e daily_return
    updated = 0
    for sym, new_d in fresh.items():
        if sym in bundle_prices:
            if "price" in new_d:
                bundle_prices[sym]["price"] = new_d["price"]
            if "daily_return" in new_d:
                bundle_prices[sym]["daily_return"] = new_d["daily_return"]
            bundle_prices[sym]["source"] = new_d.get("source", "live")
            updated += 1

    bundle_prices["__refreshed_at__"] = _dt.datetime.now(_tz.utc).strftime("%H:%M:%S UTC")
    _log.info("live_refresh_done", updated=updated, of=len(tickers))
    return bundle_prices
