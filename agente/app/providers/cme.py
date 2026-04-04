"""
Provider: CME Group Market Data — Futuros, Referência e Posicionamento

Autenticação: OAuth 2.0 client_credentials
  Token URL : https://auth.cmegroup.com/as/token.oauth2
  API Base  : https://api.cmegroup.com/v1  (Reference Data API v3)

Credenciais via .env:
  CME_API_ID=<client_id>      # gerado no CME DataServices Portal
  CME_API_SECRET=<secret>

Dados disponíveis:
  - Especificações de contratos futuros (tick size, contract size, exchange)
  - Preços de liquidação do front-month
  - Séries de opções sobre futuros

Repositório de referência: https://github.com/CMEGroupPublic/datamine_python
Documentação: https://cmegroupclientsite.atlassian.net/wiki/display/EPICSANDBOX/CME+Reference+Data+API+Version+3
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("providers.cme")

_TOKEN_URL = "https://auth.cmegroup.com/as/token.oauth2"
_BASE_URL  = "https://api.cmegroup.com/v1"

# Token cache
_token_cache: dict[str, Any] = {}

# Mapeamento ticker padrão → símbolo CME
_TICKER_TO_CME: dict[str, str] = {
    "CL=F": "CL",   # WTI Crude Oil
    "NG=F": "NG",   # Natural Gas
    "GC=F": "GC",   # Gold
    "SI=F": "SI",   # Silver
    "HG=F": "HG",   # Copper
    "ZW=F": "ZW",   # Wheat
    "ZC=F": "ZC",   # Corn
    "ZS=F": "ZS",   # Soybeans
    "ES":   "ES",   # E-mini S&P 500
    "NQ":   "NQ",   # E-mini Nasdaq
    "RTY":  "RTY",  # E-mini Russell 2000
    "ZB=F": "ZB",   # 30yr T-Bond
    "ZN=F": "ZN",   # 10yr T-Note
    "6E=F": "6E",   # Euro FX
    "6J=F": "6J",   # Japanese Yen
    "DX-Y.NYB": "DX",  # Dollar Index
}


def _get_token(api_id: str, api_secret: str) -> str | None:
    """OAuth client_credentials. Reutiliza token até expirar."""
    now = time.time()
    cached  = _token_cache.get("token")
    expires = _token_cache.get("expires_at", 0)
    if cached and now < expires - 30:
        return cached

    data = urllib.parse.urlencode({
        "grant_type":    "client_credentials",
        "client_id":     api_id,
        "client_secret": api_secret,
    }).encode()

    try:
        req = urllib.request.Request(
            _TOKEN_URL,
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = json.loads(r.read())

        token = resp.get("access_token")
        if not token:
            _log.warning("cme_auth_no_token", resp=resp)
            return None

        _token_cache["token"] = token
        _token_cache["expires_at"] = now + resp.get("expires_in", 3600)
        _log.info("cme_auth_ok")
        return token

    except Exception as exc:
        _log.warning("cme_auth_error", error=str(exc))
        return None


def _api_get(path: str, token: str, params: dict | None = None) -> dict | list | None:
    """GET autenticado à CME Reference Data API."""
    url = f"{_BASE_URL}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept":        "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            return json.loads(r.read())
    except Exception as exc:
        _log.debug("cme_api_error", path=path, error=str(exc))
        return None


def collect_futures_reference(
    symbols: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Retorna especificações e preço de liquidação do front-month.

    Args:
        symbols: lista de símbolos CME ("CL", "GC", "ES") ou tickers padrão ("CL=F").
                 None = todos os futuros do _TICKER_TO_CME.

    Returns:
        {symbol: {cme_code, description, exchange, tick_size, contract_size, settle_price}}
        Retorna {} se credenciais ausentes ou API indisponível.
    """
    from app.config.settings import settings
    api_id     = settings.cme_api_id
    api_secret = settings.cme_api_secret

    if not api_id or not api_secret:
        _log.debug("cme_no_credentials")
        return {}

    token = _get_token(api_id, api_secret)
    if not token:
        return {}

    # Normaliza símbolos para formato CME
    if symbols is None:
        cme_syms = list(set(_TICKER_TO_CME.values()))
    else:
        cme_syms = []
        for s in symbols:
            cme = _TICKER_TO_CME.get(s, s.replace("=F", "").upper())
            cme_syms.append(cme)

    results: dict[str, dict[str, Any]] = {}

    for cme_sym in cme_syms:
        # Reference Data v3: /refdata/productslates/futureOptions?productCode=CL
        data = _api_get(
            "/refdata/productslates/futures",
            token,
            params={"productCode": cme_sym, "pageSize": 5},
        )
        if not data:
            time.sleep(0.2)
            continue

        products = data if isinstance(data, list) else data.get("products", data.get("data", []))
        if not products:
            time.sleep(0.2)
            continue

        # Pega o front-month (primeiro ativo)
        try:
            p = products[0] if isinstance(products, list) else products
            entry: dict[str, Any] = {
                "cme_code":      cme_sym,
                "description":   p.get("productName") or p.get("name") or "",
                "exchange":      p.get("exchange") or p.get("venue") or "",
                "tick_size":     p.get("tickIncrement") or p.get("tick_size"),
                "contract_size": p.get("contractSize") or p.get("unitOfMeasureQty"),
                "settle_price":  p.get("settlementPrice") or p.get("settle") or p.get("last"),
                "source":        "cme",
            }
            # Remove None
            entry = {k: v for k, v in entry.items() if v is not None}
            results[cme_sym] = entry
            _log.debug("cme_futures_ok", sym=cme_sym)
        except Exception as exc:
            _log.debug("cme_parse_error", sym=cme_sym, error=str(exc))

        time.sleep(0.15)

    _log.info("cme_futures_done", collected=len(results), of=len(cme_syms))
    return results


def collect_positioning() -> dict[str, dict[str, Any]]:
    """
    Posicionamento de grandes traders (COT-equivalent via CME DataMine).

    Retorna {asset: {long_pct, short_pct, net, net_change}}
    Retorna {} se subscrição não cobrir ou credenciais ausentes.

    Nota: COT real requer assinatura CME DataMine separada.
          Este endpoint tenta via Reference Data como proxy.
    """
    from app.config.settings import settings
    api_id     = settings.cme_api_id
    api_secret = settings.cme_api_secret

    if not api_id or not api_secret:
        return {}

    token = _get_token(api_id, api_secret)
    if not token:
        return {}

    results: dict[str, dict[str, Any]] = {}

    # Tenta endpoint de COT/positioning — requer DataMine subscription
    data = _api_get("/refdata/commitments-of-traders", token)
    if not data:
        _log.debug("cme_cot_not_available")
        return {}

    rows = data if isinstance(data, list) else data.get("data", [])
    for row in rows:
        try:
            asset = row.get("productCode") or row.get("symbol") or ""
            if not asset:
                continue
            long_pct  = row.get("longPercent") or row.get("long_pct")
            short_pct = row.get("shortPercent") or row.get("short_pct")
            net       = row.get("netPositions") or row.get("net")
            results[asset] = {
                "long_pct":  float(long_pct)  if long_pct  else None,
                "short_pct": float(short_pct) if short_pct else None,
                "net":       float(net)        if net        else None,
                "source":    "cme_cot",
            }
        except Exception:
            pass

    _log.info("cme_positioning_done", assets=len(results))
    return results
