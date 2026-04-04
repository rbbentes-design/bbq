"""
Provider: Cboe All Access — Opções com Greeks, IV term structure, VIX

Autenticação: OAuth 2.0 client_credentials
  Base API : https://api.livevol.com/v1
  Auth URL : https://id.livevol.com/connect/token

Credenciais via .env:
  CBOE_API_KEY=<client_id>
  CBOE_API_SECRET=<client_secret>

Dados disponíveis:
  - Option chains com IV, Delta, Gamma, Vega, Theta
  - VIX term structure (7d, 30d, 60d, 90d)
  - Put/Call ratios por underlying
  - ATM IV e skew

Tier gratuito: 14 dias trial, dados não-SIP.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("providers.cboe")

_BASE_URL = "https://api.livevol.com/v1"
_AUTH_URL = "https://id.livevol.com/connect/token"

# Token cache: {token, expires_at}
_token_cache: dict[str, Any] = {}


def _get_token(api_key: str, api_secret: str) -> str | None:
    """OAuth client_credentials flow. Reutiliza token até expirar."""
    now = time.time()
    cached = _token_cache.get("token")
    expires = _token_cache.get("expires_at", 0)
    if cached and now < expires - 30:
        return cached

    data = urllib.parse.urlencode({
        "grant_type":    "client_credentials",
        "client_id":     api_key,
        "client_secret": api_secret,
    }).encode()

    try:
        req = urllib.request.Request(
            _AUTH_URL,
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = json.loads(r.read())
        token = resp.get("access_token")
        if not token:
            _log.warning("cboe_auth_no_token", resp=resp)
            return None
        _token_cache["token"] = token
        _token_cache["expires_at"] = now + resp.get("expires_in", 3600)
        _log.info("cboe_auth_ok", expires_in=resp.get("expires_in"))
        return token
    except Exception as exc:
        _log.warning("cboe_auth_error", error=str(exc))
        return None


def _api_get(path: str, token: str, params: dict | None = None) -> dict | list | None:
    """GET request autenticado à Cboe API."""
    url = f"{_BASE_URL}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            return json.loads(r.read())
    except Exception as exc:
        _log.debug("cboe_api_error", path=path, error=str(exc))
        return None


def _collect_ticker(ticker: str, token: str) -> dict[str, Any] | None:
    """
    Coleta snapshot de opções de um ticker via Cboe All Access.
    Endpoint: /live/allaccess/market/option-and-underlying-quotes
    """
    # Normaliza ticker (Cboe usa formato OCC sem ^ e sem =X)
    sym = ticker.replace("^", "").replace("=X", "").replace("-USD", "")

    data = _api_get(
        "/live/allaccess/market/option-and-underlying-quotes",
        token,
        params={"symbol": sym, "limit": 200},
    )
    if not data:
        return None

    # A resposta é uma lista de quotes de opções
    rows = data if isinstance(data, list) else data.get("data", [])
    if not rows:
        return None

    try:
        spot = None
        calls, puts = [], []
        for row in rows:
            option_type = (row.get("option_type") or row.get("type") or "").upper()
            iv    = row.get("iv") or row.get("implied_volatility")
            oi    = row.get("open_interest") or 0
            vol   = row.get("volume") or 0
            strike = row.get("strike") or row.get("strike_price")
            und_price = row.get("underlying_bid") or row.get("underlying_last") or row.get("last")

            if und_price and spot is None:
                try:
                    spot = float(und_price)
                except Exception:
                    pass

            if option_type == "C":
                calls.append({"iv": iv, "oi": oi, "vol": vol, "strike": strike})
            elif option_type == "P":
                puts.append({"iv": iv, "oi": oi, "vol": vol, "strike": strike})

        if spot is None or not calls or not puts:
            return None

        # ATM IV (call mais próxima ao spot)
        calls_valid = [c for c in calls if c["iv"] and c["strike"]]
        puts_valid  = [p for p in puts  if p["iv"] and p["strike"]]

        atm_iv = None
        if calls_valid:
            atm_call = min(calls_valid, key=lambda c: abs(float(c["strike"]) - spot))
            atm_iv = round(float(atm_call["iv"]), 4)

        # Skew 5%
        skew = None
        if calls_valid and puts_valid:
            otm_put_strike  = spot * 0.95
            otm_call_strike = spot * 1.05
            try:
                p = min(puts_valid,  key=lambda x: abs(float(x["strike"]) - otm_put_strike))
                c = min(calls_valid, key=lambda x: abs(float(x["strike"]) - otm_call_strike))
                skew = round(float(p["iv"]) - float(c["iv"]), 4)
            except Exception:
                pass

        # PCR OI
        total_call_oi = sum(float(c["oi"] or 0) for c in calls)
        total_put_oi  = sum(float(p["oi"] or 0) for p in puts)
        pcr_oi = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else None

        result: dict[str, Any] = {
            "spot":     round(spot, 4),
            "atm_iv":   atm_iv,
            "skew_5pct": skew,
            "pcr_oi":   pcr_oi,
            "source":   "cboe",
        }
        _log.debug("cboe_ticker_ok", sym=ticker, atm_iv=atm_iv)
        return result

    except Exception as exc:
        _log.debug("cboe_parse_error", sym=ticker, error=str(exc))
        return None


def collect(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """
    Coleta snapshot de opções via Cboe All Access.

    Requer CBOE_API_KEY e CBOE_API_SECRET no .env.
    Retorna {} se credenciais ausentes ou API indisponível.
    """
    from app.config.settings import settings
    api_key    = settings.cboe_api_key
    api_secret = settings.cboe_api_secret

    if not api_key or not api_secret:
        _log.debug("cboe_no_credentials")
        return {}

    token = _get_token(api_key, api_secret)
    if not token:
        return {}

    results: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        data = _collect_ticker(ticker, token)
        if data:
            results[ticker] = data
        time.sleep(0.1)  # respeita rate limit

    _log.info("cboe_done", collected=len(results), of=len(tickers))
    return results


def collect_vix_term_structure(token: str | None = None) -> dict[str, float]:
    """
    Retorna a estrutura a termo do VIX: {7d, 30d, 60d, 90d}.
    Usado para análise de vol regime.
    """
    from app.config.settings import settings
    if token is None:
        token = _get_token(settings.cboe_api_key, settings.cboe_api_secret)
    if not token:
        return {}

    data = _api_get("/live/allaccess/market/vix-term-structure", token)
    if not data:
        return {}

    result: dict[str, float] = {}
    try:
        rows = data if isinstance(data, list) else data.get("data", [])
        for row in rows:
            days = row.get("days_to_expiration") or row.get("dte")
            iv   = row.get("iv") or row.get("value")
            if days and iv:
                result[f"vix_{int(days)}d"] = round(float(iv), 4)
    except Exception as exc:
        _log.debug("cboe_vix_ts_error", error=str(exc))

    return result
