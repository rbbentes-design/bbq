"""
Provider: Options Data

Coleta snapshot de opções para ativos líquidos:
  - ATM IV (implied volatility at-the-money)
  - Skew: IV_put_OTM5% − IV_call_OTM5%  (proxy de demanda por proteção)
  - Put/Call OI ratio
  - GEX (Gamma Exposure) estimado em $B
  - Term structure: IV a 7d, 30d, 60d (expirations mais próximas)

Fonte primária : IBKR (reqSecDefOptParams + reqMktData snapshot)
Fonte fallback  : yfinance (gratuito, sem auth) — quando IBKR indisponível

Tickers cobertos: índices proxy + top liquid stocks.
Não cobre todos os 200+ tickers — foco em liquidez de opções.
"""

from __future__ import annotations

import bisect
import math
import time
from datetime import date, timedelta
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("providers.options")

# Tickers com liquidez de opções suficiente para análise
# Usamos ETF proxies para índices (SPY, QQQ, IWM) pois têm mais liquidez que futuros
OPTIONS_UNIVERSE: dict[str, str] = {
    # Índices US (ETF proxies)
    "SPY":  "S&P 500",
    "QQQ":  "Nasdaq 100",
    "IWM":  "Russell 2000",
    # Mega-caps (maior OI de opções)
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
    "META": "Meta",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "NFLX": "Netflix",
    "AMD":  "AMD",
    "INTC": "Intel",
    "CRM":  "Salesforce",
    "ORCL": "Oracle",
    # Financials
    "JPM":  "JPMorgan",
    "GS":   "Goldman Sachs",
    "BAC":  "Bank of America",
    "XLF":  "Financials ETF",
    # Energia / Commodities
    "XLE":  "Energy ETF",
    "XOM":  "Exxon",
    "CVX":  "Chevron",
    "CL":   "Colgate",
    # Macro / rates / vol
    "GLD":  "Gold",
    "SLV":  "Silver",
    "TLT":  "Treasuries 20yr",
    "IEF":  "Treasuries 7-10yr",
    "HYG":  "High Yield",
    "LQD":  "IG Corp Bonds",
    "DXY":  "Dollar Index",
    "EEM":  "Emerging Markets",
    # Volatility
    "VIXY": "VIX Short-Term",
}

# Mapeamento ticker de índice → ETF proxy (para opções)
INDEX_TO_ETF: dict[str, str] = {
    "^GSPC": "SPY",
    "^NDX":  "QQQ",
    "^RUT":  "IWM",
    "^VIX":  "VIXY",
}


def _nearest_expiries(expiry_list: list[str], n: int = 3) -> list[str]:
    """Retorna as n expirations mais próximas de hoje."""
    today = date.today().isoformat()
    future = sorted(e for e in expiry_list if e >= today)
    return future[:n]


def _atm_iv(chain_df: Any, spot: float) -> float | None:
    """IV do strike mais próximo ao spot."""
    try:
        df = chain_df.copy()
        df = df[df["impliedVolatility"] > 0.001]
        if df.empty:
            return None
        idx = (df["strike"] - spot).abs().idxmin()
        return round(float(df.loc[idx, "impliedVolatility"]), 4)
    except Exception:
        return None


def _skew(calls: Any, puts: Any, spot: float) -> float | None:
    """
    Skew = IV_put(spot × 0.95) − IV_call(spot × 1.05)
    Positivo → mercado pagando mais por proteção (bearish fear premium).
    """
    try:
        otm_put_strike  = spot * 0.95
        otm_call_strike = spot * 1.05

        p = puts[puts["impliedVolatility"] > 0.001].copy()
        c = calls[calls["impliedVolatility"] > 0.001].copy()
        if p.empty or c.empty:
            return None

        pi = (p["strike"] - otm_put_strike).abs().idxmin()
        ci = (c["strike"] - otm_call_strike).abs().idxmin()

        put_iv  = float(p.loc[pi,  "impliedVolatility"])
        call_iv = float(c.loc[ci,  "impliedVolatility"])
        return round(put_iv - call_iv, 4)
    except Exception:
        return None


def _pcr(calls: Any, puts: Any) -> float | None:
    """Put/Call open interest ratio."""
    try:
        put_oi  = float(puts["openInterest"].sum())
        call_oi = float(calls["openInterest"].sum())
        if call_oi < 1:
            return None
        return round(put_oi / call_oi, 3)
    except Exception:
        return None


def _gex(calls: Any, puts: Any, spot: float, expiry: str | None = None) -> float | None:
    """
    Gamma Exposure estimado em $Bilhões.

    GEX = Σ_calls(OI × gamma × 100 × spot²) − Σ_puts(OI × gamma × 100 × spot²)

    Gamma via Black-Scholes. T calculado da data de expiração do contrato.
    Positivo = dealers long gamma (estabilizador).
    Negativo = dealers short gamma (amplificador).
    """
    try:
        import math

        def bs_gamma(S: float, K: float, T: float, sigma: float) -> float:
            if T <= 0 or sigma <= 0:
                return 0.0
            d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
            return math.exp(-0.5 * d1**2) / (math.sqrt(2 * math.pi) * S * sigma * math.sqrt(T))

        # T correto = dias até expiração (não lastTradeDate)
        today = date.today()
        if expiry:
            try:
                T_default = max((date.fromisoformat(expiry) - today).days, 1) / 365.0
            except Exception:
                T_default = 30 / 365.0
        else:
            T_default = 30 / 365.0

        gex_total = 0.0
        for df, sign in [(calls, 1), (puts, -1)]:
            df = df[df["impliedVolatility"] > 0.001].copy()
            for _, row in df.iterrows():
                try:
                    # OI é EOD no yfinance — usa volume como proxy quando OI=0
                    oi  = float(row.get("openInterest", 0) or 0)
                    vol = float(row.get("volume", 0) or 0)
                    effective_oi = oi if oi > 0 else vol * 0.1
                    if effective_oi <= 0:
                        continue
                    iv = float(row["impliedVolatility"])
                    K  = float(row["strike"])
                    g  = bs_gamma(spot, K, T_default, iv)
                    gex_total += sign * effective_oi * g * 100 * spot * spot
                except Exception:
                    continue

        return round(gex_total / 1e9, 3)  # em bilhões
    except Exception:
        return None


def _collect_ticker_yf(sym: str, label: str) -> dict[str, Any] | None:
    """Coleta snapshot de opções de um único ticker via yfinance."""
    try:
        import yfinance as yf
        t = yf.Ticker(sym)
        expiries = t.options
        if not expiries:
            return None

        # Preço spot
        info  = t.fast_info
        spot  = float(getattr(info, "last_price", None) or getattr(info, "previousClose", 0))
        if spot <= 0:
            hist = t.history(period="2d")
            if hist.empty:
                return None
            spot = float(hist["Close"].iloc[-1])

        nearest = _nearest_expiries(list(expiries), n=3)
        if not nearest:
            return None

        result: dict[str, Any] = {
            "label":        label,
            "spot":         round(spot, 4),
            "next_expiry":  nearest[0],
            "expiries":     nearest,
        }

        # IV por expiração (term structure)
        term_ivs: dict[str, float | None] = {}
        for exp in nearest:
            try:
                chain = t.option_chain(exp)
                iv = _atm_iv(chain.calls, spot)
                if iv is not None:
                    days = (date.fromisoformat(exp) - date.today()).days
                    term_ivs[f"iv_{max(days,1)}d"] = iv
            except Exception:
                continue

        result["term_structure"] = term_ivs

        # ATM IV, skew, PCR, GEX — da 1ª expiração (mais líquida)
        try:
            chain0 = t.option_chain(nearest[0])
            calls, puts = chain0.calls, chain0.puts

            result["atm_iv"]    = _atm_iv(calls, spot)
            result["skew_5pct"] = _skew(calls, puts, spot)
            result["pcr_oi"]    = _pcr(calls, puts)
            result["gex_b"]     = _gex(calls, puts, spot, expiry=nearest[0])
        except Exception as exc:
            _log.debug("options_chain_error", sym=sym, error=str(exc))

        _log.debug("options_ok", sym=sym, atm_iv=result.get("atm_iv"))
        return result

    except Exception as exc:
        _log.debug("options_ticker_error", sym=sym, error=str(exc))
        return None


def _collect_ticker_ibkr(
    ib: Any,
    sym: str,
    label: str,
    spot: float,
) -> dict[str, Any] | None:
    """
    Coleta opções via IBKR para um ticker.
    Usa reqSecDefOptParams + reqMktData snapshot para IV ATM e skew.
    """
    try:
        from ib_insync import Option
        from app.providers.ibkr import _make_contract

        contract = _make_contract(sym)
        if contract is None:
            return None
        try:
            ib.qualifyContracts(contract)
        except Exception:
            pass

        # Parâmetros da chain (expirations + strikes)
        chains = ib.reqSecDefOptParams(
            contract.symbol, "",
            contract.secType,
            contract.conId,
        )
        if not chains:
            return None

        chain = chains[0]
        expirations = sorted(e for e in chain.expirations
                             if e >= date.today().strftime("%Y%m%d"))[:3]
        if not expirations:
            return None

        strikes = sorted(chain.strikes)
        atm_strike = min(strikes, key=lambda s: abs(s - spot))
        # strikes OTM 5%
        put_otm_strike  = min(strikes, key=lambda s: abs(s - spot * 0.95))
        call_otm_strike = min(strikes, key=lambda s: abs(s - spot * 1.05))

        result: dict[str, Any] = {
            "label":       label,
            "spot":        round(spot, 4),
            "next_expiry": expirations[0],
            "expiries":    [
                f"{e[:4]}-{e[4:6]}-{e[6:]}" for e in expirations
            ],
        }

        term_ivs: dict[str, float] = {}
        exp0_calls_oi = 0.0
        exp0_puts_oi  = 0.0
        atm_iv_vals: list[float] = []
        skew_put_iv: float | None = None
        skew_call_iv: float | None = None

        for exp in expirations:
            exp_date = date(int(exp[:4]), int(exp[4:6]), int(exp[6:]))
            days = max((exp_date - date.today()).days, 1)

            for strike, right, role in [
                (atm_strike,       "C", "atm_call"),
                (atm_strike,       "P", "atm_put"),
                (put_otm_strike,   "P", "otm_put"),
                (call_otm_strike,  "C", "otm_call"),
            ]:
                # Só coleta skew na primeira expiração
                if exp != expirations[0] and role in ("otm_put", "otm_call"):
                    continue
                try:
                    opt = Option(contract.symbol, exp, strike, right, chain.exchange or "SMART")
                    qualified = ib.qualifyContracts(opt)
                    if not qualified:
                        continue
                    # "106" (IV) não é suportado em snapshot=True no IBKR (Error 321)
                    # Usa genericTickList="" e obtém IV via modelGreeks
                    ticker = ib.reqMktData(opt, "", snapshot=True, regulatorySnapshot=False)
                    ib.sleep(1.5)

                    iv = None
                    if ticker.modelGreeks and ticker.modelGreeks.impliedVol:
                        iv = float(ticker.modelGreeks.impliedVol)
                    elif ticker.impliedVolatility and ticker.impliedVolatility > 0:
                        iv = float(ticker.impliedVolatility)

                    oi = float(getattr(ticker, "openInterest", None) or 0)
                    ib.cancelMktData(opt)

                    if iv and iv > 0.001:
                        if role in ("atm_call", "atm_put"):
                            atm_iv_vals.append(iv)
                            term_ivs[f"iv_{days}d"] = round(iv, 4)
                        elif role == "otm_put":
                            skew_put_iv = iv
                        elif role == "otm_call":
                            skew_call_iv = iv

                    if right == "C" and exp == expirations[0]:
                        exp0_calls_oi += oi
                    elif right == "P" and exp == expirations[0]:
                        exp0_puts_oi  += oi

                except Exception as exc:
                    _log.debug("ibkr_opt_tick_err", sym=sym, exp=exp,
                               strike=strike, right=right, error=str(exc)[:60])

        result["term_structure"] = term_ivs
        result["atm_iv"] = round(sum(atm_iv_vals) / len(atm_iv_vals), 4) if atm_iv_vals else None

        if skew_put_iv is not None and skew_call_iv is not None:
            result["skew_5pct"] = round(skew_put_iv - skew_call_iv, 4)

        if exp0_calls_oi > 0:
            result["pcr_oi"] = round(exp0_puts_oi / exp0_calls_oi, 3)

        if not result.get("atm_iv") and not term_ivs:
            return None

        _log.debug("ibkr_options_ok", sym=sym, atm_iv=result.get("atm_iv"),
                   skew=result.get("skew_5pct"), pcr=result.get("pcr_oi"))
        return result

    except Exception as exc:
        _log.debug("ibkr_options_ticker_failed", sym=sym, error=str(exc)[:100])
        return None


def collect(
    tickers: list[str] | None = None,
    max_workers: int = 8,
) -> dict[str, dict[str, Any]]:
    """
    Coleta snapshot de opções.
    Primário: IBKR (reqSecDefOptParams + reqMktData snapshot).
    Fallback: yfinance quando IBKR indisponível.

    Args:
        tickers:     lista de símbolos. None = OPTIONS_UNIVERSE.
        max_workers: threads paralelas para fallback yfinance.

    Returns:
        {ticker: {atm_iv, skew_5pct, pcr_oi, gex_b, term_structure, ...}}
    """
    universe = tickers or list(OPTIONS_UNIVERSE.keys())
    _log.info("options_start", tickers=len(universe))

    results: dict[str, dict[str, Any]] = {}

    # ── Tenta IBKR primeiro ────────────────────────────────────────────────────
    try:
        from app.providers.ibkr import is_available, _connect_ib, _ensure_event_loop
        _ensure_event_loop()

        if is_available():
            ib = _connect_ib()
            if ib is not None:
                _log.info("options_ibkr_session", tickers=len(universe))
                # Precisa de preços spot — tenta fast_info ou market_prices do bundle
                spot_map: dict[str, float] = {}
                try:
                    import yfinance as yf
                    for sym in universe:
                        try:
                            fi = yf.Ticker(sym).fast_info
                            p = getattr(fi, "last_price", None)
                            if p and float(p) > 0:
                                spot_map[sym] = float(p)
                        except Exception:
                            pass
                        time.sleep(0.05)
                except Exception:
                    pass

                try:
                    for sym in universe:
                        label = OPTIONS_UNIVERSE.get(sym, sym)
                        spot = spot_map.get(sym, 0.0)
                        if spot <= 0:
                            continue
                        data = _collect_ticker_ibkr(ib, sym, label, spot)
                        if data:
                            results[sym] = data
                        time.sleep(0.1)
                finally:
                    try:
                        ib.disconnect()
                    except Exception:
                        pass

                _log.info("options_ibkr_collected", collected=len(results))
                if results:
                    return results
                # IBKR conectou mas sem dados (sem subscrição) — usa yfinance

    except Exception as exc:
        _log.warning("ibkr_options_failed", error=str(exc))

    # ── Fallback: yfinance ─────────────────────────────────────────────────────
    _log.info("options_yfinance_fallback", tickers=len(universe))
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch(sym: str) -> tuple[str, dict | None]:
        label = OPTIONS_UNIVERSE.get(sym, sym)
        return sym, _collect_ticker_yf(sym, label)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch, sym): sym for sym in universe}
        for fut in as_completed(futures):
            sym, data = fut.result()
            if data:
                results[sym] = data

    _log.info("options_done", collected=len(results), source="yfinance")
    return results


def vix_term_structure(market_prices: dict[str, Any]) -> dict[str, Any]:
    """
    Extrai estrutura a prazo da volatilidade dos nós VIX já coletados.

    Usa preços de ^VIX9D, ^VIX (30d), VIXY como proxy de VIX3M.

    Returns:
        {
          "vix9d":  float | None,
          "vix30d": float | None,
          "vix3m":  float | None,  # proxy via VIXY ou VIX3M
          "contango": bool | None,  # True = mercado em contango (VIX9D < VIX30d)
          "ts_slope": float | None, # (vix30d - vix9d) / vix9d
        }
    """
    vix9d  = (market_prices.get("^VIX9D") or {}).get("price")
    vix30d = (market_prices.get("^VIX")   or {}).get("price")
    # VIX3M proxy: ^VIX3M se disponível, senão VIXY
    vix3m  = (market_prices.get("^VIX3M") or market_prices.get("VIXY") or {}).get("price")

    contango = None
    ts_slope = None
    if vix9d and vix30d and vix9d > 0:
        contango = vix30d > vix9d
        ts_slope = round((vix30d - vix9d) / vix9d, 4)

    return {
        "vix9d":    round(vix9d,  2) if vix9d  else None,
        "vix30d":   round(vix30d, 2) if vix30d else None,
        "vix3m":    round(vix3m,  2) if vix3m  else None,
        "contango": contango,
        "ts_slope": ts_slope,
    }
