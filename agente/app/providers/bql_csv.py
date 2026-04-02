"""
Provider: BQL CSV Reader

Lê os CSVs exportados pelo scripts/bql_export.py (que roda no BQuant).
Substitui todas as chamadas BQL diretas no app Python 3.14.

Arquivos esperados em BQL_DATA_DIR:
    fundamentals.csv  — PE, beta, mktcap, ROE, etc.
    options_iv.csv    — ATM IV, skew 25D, put/call OI
    gex_summary.csv   — GEX SPX agregado
    gex_spx.csv       — GEX por strike
    letf_flows.csv    — NAV e AUM dos LETFs
    meta.csv          — timestamp de geração
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("providers.bql_csv")

BQL_DATA_DIR = Path(r"C:\Users\rafael bentes\agente-workspace\bql_data")

# Alerta se CSV tiver mais de X minutos
_STALE_MINUTES = 30


def _read_csv(prefix: str) -> list[dict[str, str]]:
    """Lê o CSV mais recente com nome prefix_YYYY-MM-DD.csv ou prefix.csv (fallback)."""
    # Procura arquivos datados primeiro: fundamentals_2026-04-02.csv
    dated = sorted(BQL_DATA_DIR.glob(f"{prefix}_*.csv"), reverse=True)
    if dated:
        path = dated[0]
    else:
        # fallback: nome fixo (sem data)
        path = BQL_DATA_DIR / f"{prefix}.csv"
    if not path.exists():
        _log.warning("bql_csv_missing", prefix=prefix)
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(val: str | None) -> float | None:
    if not val:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _check_freshness() -> bool:
    """True se os dados foram gerados nos últimos STALE_MINUTES minutos."""
    rows = _read_csv("meta")
    if not rows:
        return False
    ts = rows[0].get("generated_at", "")
    try:
        generated = datetime.fromisoformat(ts)
        age = datetime.now() - generated
        if age > timedelta(minutes=_STALE_MINUTES):
            _log.warning("bql_csv_stale", age_min=round(age.total_seconds() / 60, 1))
            return False
        return True
    except Exception:
        return False


# ── Fundamentals ──────────────────────────────────────────────────────────────

def load_fundamentals() -> dict[str, dict[str, Any]]:
    """
    Retorna {ticker: {pe, beta, mktcap_b, roe, profit_margin, ...}}.
    Formato idêntico ao node_anatomy.collect() para drop-in replacement.
    """
    rows = _read_csv("fundamentals.csv")
    if not rows:
        return {}

    result: dict[str, dict[str, Any]] = {}
    for r in rows:
        ticker = r.get("ticker", "").strip()
        if not ticker:
            continue
        entry: dict[str, Any] = {}
        for field in ["pe", "forward_pe", "pb", "ev_ebitda", "beta", "mktcap_b",
                      "roe", "profit_margin", "debt_equity", "dividend_yield",
                      "price", "hi_52w", "lo_52w", "drawdown_52w"]:
            v = _float(r.get(field))
            if v is not None:
                entry[field] = v
        if entry:
            result[ticker] = entry

    _log.info("bql_csv_fundamentals_loaded", tickers=len(result))
    return result


# ── Options IV ───────────────────────────────────────────────────────────────

def load_options_iv() -> dict[str, dict[str, Any]]:
    """
    Retorna {ticker: {atm_iv, skew_25d, pcr_oi}}.
    """
    rows = _read_csv("options_iv.csv")
    if not rows:
        return {}

    result: dict[str, dict[str, Any]] = {}
    for r in rows:
        ticker = r.get("ticker", "").strip()
        if not ticker:
            continue
        entry: dict[str, Any] = {}
        for field in ["atm_iv", "skew_25d", "pcr_oi"]:
            v = _float(r.get(field))
            if v is not None:
                entry[field] = v
        if entry:
            result[ticker] = entry

    _log.info("bql_csv_options_loaded", tickers=len(result))
    return result


# ── GEX ──────────────────────────────────────────────────────────────────────

def load_gex_summary() -> dict[str, Any]:
    """
    Retorna o resumo do GEX SPX (gex_total_bn, direction, gamma_regime, etc.).
    Compatível com o formato de FlowPrediction.gex.spx usado no dashboard.
    """
    rows = _read_csv("gex_summary.csv")
    if not rows:
        return {}
    r = rows[0]
    return {
        "date":           r.get("date", ""),
        "spot":           _float(r.get("spot")),
        "gex_bn":         _float(r.get("gex_total_bn")) or 0.0,
        "gex_call_bn":    _float(r.get("gex_call_bn")) or 0.0,
        "gex_put_bn":     _float(r.get("gex_put_bn")) or 0.0,
        "direction":      r.get("direction", "flat"),
        "gamma_regime":   r.get("gamma_regime", "flat"),
        "n_options":      int(r.get("n_options", 0) or 0),
    }


def load_gex_by_strike() -> list[dict[str, Any]]:
    """Lista de {strike, put_call, open_int, gamma, gex_bn} para heat map."""
    rows = _read_csv("gex_spx.csv")
    result = []
    for r in rows:
        result.append({
            "expiry":   r.get("expiry", ""),
            "strike":   _float(r.get("strike")),
            "put_call": r.get("put_call", ""),
            "open_int": int(r.get("open_int", 0) or 0),
            "gamma":    _float(r.get("gamma")),
            "gex_bn":   _float(r.get("gex_bn")),
        })
    return result


# ── LETF Flows ───────────────────────────────────────────────────────────────

def load_letf_flows() -> list[dict[str, Any]]:
    """Lista de {ticker, leverage, index, nav, aum_b}."""
    rows = _read_csv("letf_flows.csv")
    result = []
    for r in rows:
        result.append({
            "ticker":   r.get("ticker", ""),
            "leverage": int(r.get("leverage", 0) or 0),
            "index":    r.get("index", ""),
            "nav":      _float(r.get("nav")),
            "aum_b":    _float(r.get("aum_b")),
        })
    return result


# ── API unificada ────────────────────────────────────────────────────────────

def load_all() -> dict[str, Any]:
    """
    Carrega todos os CSVs de uma vez.
    Retorna dict com fundamentals, options_iv, gex, letf.
    Logga aviso se dados estiverem velhos.
    """
    fresh = _check_freshness()
    if not fresh:
        _log.warning("bql_csv_not_fresh", hint="Rode: python scripts/bql_export.py --loop")

    return {
        "fundamentals": load_fundamentals(),
        "options_iv":   load_options_iv(),
        "gex_summary":  load_gex_summary(),
        "gex_strikes":  load_gex_by_strike(),
        "letf_flows":   load_letf_flows(),
        "fresh":        fresh,
    }
