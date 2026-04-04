"""
Provider: Options Data

Coleta snapshot de opções para ativos líquidos:
  - ATM IV (implied volatility at-the-money)
  - Skew: IV_put_OTM5% − IV_call_OTM5%  (proxy de demanda por proteção)
  - Put/Call OI ratio
  - GEX (Gamma Exposure) estimado em $B
  - Term structure: IV a 7d, 30d, 60d (expirations mais próximas)

Hierarquia de fontes:
  1. IBKR          — Greeks completos, live chain (requer TWS)
  2. OCC           — PCR + OI diário (gratuito, sem credenciais)
  3. Cboe All Access — Greeks completos (OAuth, requer CBOE_API_KEY)

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
                    # OI é EOD — usa volume como proxy quando OI=0
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
) -> dict[str, dict[str, Any]]:
    """
    Coleta snapshot de opções.
    Hierarquia: IBKR → OCC (free, PCR/OI) → Cboe (Greeks completos).

    Args:
        tickers: lista de símbolos. None = OPTIONS_UNIVERSE.

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
                # Precisa de preços spot — usa Bloomberg DB ou IBKR collect
                spot_map: dict[str, float] = {}
                try:
                    from app.providers.market_prices import collect as _prices
                    raw_prices = _prices()
                    for sym in universe:
                        etf = INDEX_TO_ETF.get(sym, sym)
                        p = (raw_prices.get(sym) or raw_prices.get(etf) or {}).get("price")
                        if p and float(p) > 0:
                            spot_map[sym] = float(p)
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
                # IBKR conectou mas sem dados (sem subscrição) — cai para OCC/Cboe

    except Exception as exc:
        _log.warning("ibkr_options_failed", error=str(exc))

    # ── Fallback tier 2: OCC (gratuito — PCR + OI, sem Greeks) ───────────────
    _log.info("options_occ_fallback", tickers=len(universe))
    try:
        from app.providers.occ import collect_daily_stats
        occ_stats = collect_daily_stats()
        for sym in universe:
            key = INDEX_TO_ETF.get(sym, sym)
            if key in occ_stats and sym not in results:
                results[sym] = {**occ_stats[key], "label": OPTIONS_UNIVERSE.get(sym, sym)}
        if occ_stats:
            _log.info("options_occ_collected", collected=len([s for s in universe if s in results]))
    except Exception as exc:
        _log.warning("occ_options_failed", error=str(exc))

    # ── Fallback tier 3: Cboe All Access (Greeks completos, requer credenciais) ─
    from app.config.settings import settings
    if settings.cboe_api_key and settings.cboe_api_secret:
        missing = [s for s in universe if s not in results]
        if missing:
            _log.info("options_cboe_fallback", tickers=len(missing))
            try:
                from app.providers.cboe import collect as cboe_collect
                cboe_data = cboe_collect(missing)
                for sym, d in cboe_data.items():
                    if sym not in results:
                        results[sym] = d
                if cboe_data:
                    _log.info("options_cboe_collected", collected=len(cboe_data))
            except Exception as exc:
                _log.warning("cboe_options_failed", error=str(exc))

    _log.info("options_done", collected=len(results))
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
