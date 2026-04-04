"""
Provider: Node Anatomy — Bloomberg BQL

Coleta dados fundamentais para os nós do grafo via Bloomberg BQL (BQuant Python 3.11).
O agente roda em Python 3.14 (incompatível com blpapi), então usa subprocess para
invocar o script scripts/bql_fetch.py sob o Python do BQuant.

Campos retornados por ticker:
  pe            — P/E trailing
  beta          — beta Bloomberg
  mktcap_b      — market cap em bilhões USD
  roe           — Return on Equity (decimal)
  profit_margin — margem líquida (decimal)
  debt_equity   — dívida/patrimônio
  dividend_yield — dividend yield (decimal)
  price         — preço corrente
  hi_52w        — máxima 52 semanas
  lo_52w        — mínima 52 semanas
  drawdown_52w  — preço atual vs hi_52w (decimal negativo)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("providers.node_anatomy")

_BQNT_PYTHON = Path("C:/blp/bqnt/environments/bqnt-3/python.exe")
_BQL_SCRIPT  = Path(__file__).parent.parent.parent / "scripts" / "bql_fetch.py"

_SKIP_PATTERN = re.compile(
    r"(\^|=X$|=F$|-USD$|-EUR$|\.NYB$|\.SS$|\.HK$|\.BO$|\.NS$)"
)


def _is_equity(ticker: str) -> bool:
    return not bool(_SKIP_PATTERN.search(ticker))


def _collect_bql(
    tickers: list[str],
    price_map: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """
    Chama scripts/bql_fetch.py via BQuant Python 3.11 como subprocess.
    Retorna dict {ticker: {pe, beta, mktcap_b, ...}}.
    """
    if not _BQNT_PYTHON.exists():
        _log.warning("bqnt_python_not_found", path=str(_BQNT_PYTHON))
        return {}

    if not _BQL_SCRIPT.exists():
        _log.warning("bql_script_not_found", path=str(_BQL_SCRIPT))
        return {}

    _log.info("bql_fetch_start", tickers=len(tickers))

    # Lotes de 50 para não sobrecarregar o BQL
    BATCH = 50
    results: dict[str, dict[str, Any]] = {}

    for i in range(0, len(tickers), BATCH):
        batch = tickers[i : i + BATCH]
        try:
            proc = subprocess.run(
                [str(_BQNT_PYTHON), str(_BQL_SCRIPT)] + batch,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0:
                _log.debug("bql_batch_stderr", stderr=proc.stderr[:200])

            if proc.stdout.strip():
                batch_data: dict = json.loads(proc.stdout.strip())
                results.update(batch_data)
                _log.debug("bql_batch_ok", batch_start=i, got=len(batch_data))

        except subprocess.TimeoutExpired:
            _log.warning("bql_batch_timeout", batch_start=i)
        except json.JSONDecodeError as exc:
            _log.warning("bql_batch_json_error", error=str(exc))
        except Exception as exc:
            _log.warning("bql_batch_error", error=str(exc))

    # Adiciona drawdown se price_map tem preço mais fresco
    for sym, entry in results.items():
        price = price_map.get(sym) or entry.get("price")
        hi = entry.get("hi_52w")
        if price and hi and hi > 0:
            entry["drawdown_52w"] = round((float(price) - hi) / hi, 4)

    _log.info("bql_fetch_done", collected=len(results), of=len(tickers))
    return results


# ── API pública ───────────────────────────────────────────────────────────────

def collect(
    tickers: list[str],
    price_map: dict[str, float] | None = None,
    max_workers: int = 1,
) -> dict[str, dict[str, Any]]:
    """
    Coleta fundamentais via Bloomberg BQL.

    Args:
        tickers:   lista de símbolos
        price_map: {ticker: price} para drawdown atualizado
    Returns:
        {ticker: {pe, beta, mktcap_b, roe, profit_margin, debt_equity,
                  dividend_yield, price, hi_52w, lo_52w, drawdown_52w}}
    """
    price_map = price_map or {}
    equity_tickers = [t for t in tickers if _is_equity(t)]
    _log.info("anatomy_start", total=len(tickers), equity=len(equity_tickers))

    # 1. Tenta DB Bloomberg (bql_latest) — dados já ingeridos pelo pipeline
    try:
        from app.query_layer import BloombergQueryLayer
        ql = BloombergQueryLayer()
        if ql.has_any_data():
            funds = ql.get_fundamentals()
            if funds:
                # Adiciona drawdown com preço atualizado
                for sym, entry in funds.items():
                    price = price_map.get(sym) or entry.get("price")
                    hi = entry.get("hi_52w")
                    if price and hi and hi > 0:
                        entry["drawdown_52w"] = round((float(price) - hi) / hi, 4)
                _log.info("anatomy_from_db", collected=len(funds))
                return funds
    except Exception as exc:
        _log.debug("anatomy_db_failed", error=str(exc))

    # 2. Tenta BQuant subprocess (script BQL)
    result = _collect_bql(equity_tickers, price_map)
    if result:
        return result

    # No data available — return empty
    _log.warning("anatomy_no_data", tickers=len(equity_tickers))
    return {}


_OPTIONS_FIELDS = {"atm_iv", "skew_5pct", "pcr_oi"}
_ANATOMY_FIELDS = {"pe", "mktcap_b", "beta", "profit_margin", "debt_equity",
                   "roe", "dividend_yield", "price", "hi_52w", "lo_52w",
                   "drawdown_52w", "forward_pe", "ps", "pb", "ev_ebitda", "roa"}


def collect_from_registry(
    price_map: dict[str, float] | None = None,
    tickers_filter: list[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """
    Coleta fundamentais + options via BQL (primário) ou DB Bloomberg (fallback).

    Args:
        price_map:      {ticker: price} para drawdown atualizado
        tickers_filter: se fornecido, limita a coleta a esses tickers

    Returns:
        (anatomy_map, options_map)
    """
    price_map = price_map or {}

    if tickers_filter:
        tickers = [t for t in tickers_filter if _is_equity(t)]
    else:
        try:
            from app.desk.node_registry import NODES
            tickers = list({
                n.get("ticker")
                for n in NODES.values()
                if n.get("ticker") and n.get("level", 99) == 5
            })
        except Exception as exc:
            _log.warning("registry_load_failed", error=str(exc))
            tickers = []

    # Coleta via BQL (Bloomberg)
    raw = collect(tickers, price_map=price_map)

    anatomy_map: dict[str, dict[str, Any]] = {}
    options_map: dict[str, dict[str, Any]] = {}

    for ticker, data in raw.items():
        anat = {k: v for k, v in data.items() if k not in _OPTIONS_FIELDS}
        opts = {k: v for k, v in data.items() if k in _OPTIONS_FIELDS}
        if anat:
            anatomy_map[ticker] = anat
        if opts:
            options_map[ticker] = opts

    _log.info("bql_split", anatomy=len(anatomy_map), options=len(options_map))
    return anatomy_map, options_map
