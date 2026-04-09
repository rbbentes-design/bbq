"""
Provider: BQL CSV / SQLite Reader
===================================

SINAPSE OFICIAL entre os dados Bloomberg e o MacroDesk.

Fonte primária: banco SQLite (data/database/macrodesk.db) via query_layer.
Fonte secundária: arquivos CSV em bql_data/ (fallback enquanto banco estiver vazio).

Quando o banco estiver populado pelo bloomberg_main_agent, a leitura é 100% SQLite.
O fallback CSV garante compatibilidade durante a transição.

API pública (inalterada — drop-in replacement):
    load_fundamentals()     → {ticker: {pe, beta, mktcap_b, roe, ...}}
    load_options_iv()       → {ticker: {atm_iv, skew_25d, pcr_oi}}
    load_gex_summary()      → {date, spot, gex_bn, ...}
    load_gex_by_strike()    → [{strike, put_call, open_int, gamma, gex_bn}]
    load_letf_flows()       → [{ticker, leverage, index, nav, aum_b}]
    load_prices()           → {yf_ticker: {name, price, daily_return, ...}}
    load_price_history()    → {yf_ticker: [close_0, ..., close_N]}
    load_macro_series()     → {bbg_ticker: {field: value, ...}}   ← NOVO
    load_all()              → unified dict
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("providers.bql_csv")

BQL_DATA_DIR = Path(r"C:\Users\rafael bentes\bbg\agente\bql_data")

_STALE_MINUTES = 60   # dados mais velhos que isso disparam aviso


# ══════════════════════════════════════════════════════════════════════════════
# CAMADA 1 — SQLite (fonte oficial)
# ══════════════════════════════════════════════════════════════════════════════

def _try_load_from_db() -> "BloombergQueryLayer | None":
    """
    Retorna a query layer se o banco existir e tiver dados.
    Retorna None silenciosamente se banco vazio ou indisponível.
    """
    try:
        import sys
        from pathlib import Path as _P
        _root = _P(__file__).parent.parent.parent
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from app.query_layer import BloombergQueryLayer
        ql = BloombergQueryLayer()
        # Verifica se o banco tem qualquer dado (não precisa ser fresh)
        status = ql.get_last_ingestion_status()
        if status and status.get("rows_ingested", 0) > 0:
            return ql
        return None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CAMADA 2 — CSV fallback (compatibilidade durante transição)
# ══════════════════════════════════════════════════════════════════════════════

def _read_csv(prefix: str) -> list[dict[str, str]]:
    """Lê o CSV mais recente com nome prefix_YYYYMMDD_*.csv ou prefix.csv."""
    # Procura arquivos com timestamp (novo padrão): prices_20260403_143015_a3f7.csv
    stamped = sorted(BQL_DATA_DIR.glob(f"{prefix}_*.csv"), reverse=True)
    if stamped:
        path = stamped[0]
    else:
        path = BQL_DATA_DIR / f"{prefix}.csv"
    if not path.exists():
        _log.debug("bql_csv_missing", prefix=prefix)
        return []
    try:
        with open(path, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        _log.warning("bql_csv_read_error", prefix=prefix, error=str(exc))
        return []


def _float(val: Any) -> float | None:
    if val is None or str(val).strip() in ("", "NaN", "nan", "null", "None", "N/A"):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _check_freshness() -> bool:
    rows = _read_csv("meta")
    if not rows:
        return False
    ts = rows[0].get("generated_at", "") or rows[0].get("value", "")
    try:
        generated = datetime.fromisoformat(ts)
        age = datetime.now() - generated.replace(tzinfo=None)
        if age > timedelta(minutes=_STALE_MINUTES):
            _log.warning("bql_csv_stale", age_min=round(age.total_seconds() / 60, 1))
            return False
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# API PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════

def load_fundamentals() -> dict[str, dict[str, Any]]:
    """
    Retorna {ticker: {pe, beta, mktcap_b, roe, profit_margin, ...}}.
    Fonte: SQLite → CSV fallback.
    """
    # ── Fonte primária: SQLite ─────────────────────────────────────────────
    ql = _try_load_from_db()
    if ql:
        data = ql.get_fundamentals()
        if data:
            _log.info("fundamentals_from_db", tickers=len(data))
            return data

    # ── Fallback: CSV ─────────────────────────────────────────────────────
    rows = _read_csv("fundamentals")
    if not rows:
        return {}

    result: dict[str, dict[str, Any]] = {}
    for r in rows:
        ticker = (r.get("ticker") or r.get("bbg_ticker") or "").strip()
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

    _log.info("fundamentals_from_csv", tickers=len(result))
    return result


def load_options_iv() -> dict[str, dict[str, Any]]:
    """
    Retorna {ticker: {atm_iv, skew_25d, pcr_oi, iv_percentile}}.
    Fonte: SQLite → CSV fallback.
    Enriquece com iv_percentile via iv_history CSV se disponível.
    """
    ql = _try_load_from_db()
    if ql:
        data = ql.get_options_iv()
        if data:
            _log.info("options_iv_from_db", tickers=len(data))
            # Enriquece com iv_percentile do histórico
            _enrich_iv_percentile(data)
            return data

    rows = _read_csv("options_iv")
    if not rows:
        return {}

    result: dict[str, dict[str, Any]] = {}
    for r in rows:
        ticker = (r.get("ticker") or "").strip()
        if not ticker:
            continue
        entry: dict[str, Any] = {}
        for field in ["atm_iv", "skew_25d", "pcr_oi"]:
            v = _float(r.get(field))
            if v is not None:
                entry[field] = v
        if entry:
            result[ticker] = entry

    _enrich_iv_percentile(result)
    _log.info("options_iv_from_csv", tickers=len(result))
    return result


def _enrich_iv_percentile(options_iv: dict[str, dict[str, Any]]) -> None:
    """
    Enriquece options_iv com iv_percentile calculado via histórico de IV Bloomberg.
    Modifica options_iv in-place.
    Fonte primária: tabela iv_history no SQLite (via query_layer).
    Fonte secundária: iv_history_*.csv (fallback legado).
    """
    # Fonte primária: SQLite via query_layer
    iv_by_ticker: dict[str, list[float]] = {}
    ql = _try_load_from_db()
    if ql:
        try:
            iv_by_ticker = ql.get_iv_history()
        except AttributeError:
            pass  # versão antiga sem get_iv_history

    # Fallback: CSV direto
    if not iv_by_ticker:
        iv_hist_rows = _read_csv("iv_history")
        if iv_hist_rows:
            from collections import defaultdict
            _iv_map: dict[str, list[float]] = defaultdict(list)
            for r in iv_hist_rows:
                tk = (r.get("yf_ticker") or r.get("ticker") or "").strip()
                iv = _float(r.get("iv"))
                if tk and iv is not None and iv > 0:
                    _iv_map[tk].append(iv)
            iv_by_ticker = dict(_iv_map)

    if not iv_by_ticker:
        return

    enriched = 0
    for ticker, entry in options_iv.items():
        cur_iv = entry.get("atm_iv")
        if not cur_iv or cur_iv <= 0:
            continue
        # Normaliza ticker para lookup (BBG suffix strip)
        tk_norm = ticker.replace(" US Equity", "").strip()
        hist = iv_by_ticker.get(ticker) or iv_by_ticker.get(tk_norm)
        if hist and len(hist) >= 20:
            iv_pct = sum(1 for v in hist if v <= cur_iv) / len(hist)
            entry["iv_percentile"] = round(iv_pct, 3)
            entry["iv_52w_low"]    = round(min(hist), 4)
            entry["iv_52w_high"]   = round(max(hist), 4)
            enriched += 1

    if enriched:
        _log.info("iv_percentile_enriched_bloomberg", tickers=enriched)


def load_gex_summary() -> dict[str, Any]:
    """
    Retorna o resumo do GEX SPX.
    Fonte: SQLite → CSV fallback.
    """
    ql = _try_load_from_db()
    if ql:
        data = ql.get_gex_summary()
        if data:
            _log.info("gex_summary_from_db")
            return data

    rows = _read_csv("gex_summary")
    if not rows:
        return {}
    r = rows[0]
    return {
        "date":         r.get("date", ""),
        "spot":         _float(r.get("spot")),
        "gex_bn":       _float(r.get("gex_total_bn")) or 0.0,
        "gex_call_bn":  _float(r.get("gex_call_bn")) or 0.0,
        "gex_put_bn":   _float(r.get("gex_put_bn")) or 0.0,
        "direction":    r.get("direction", "flat"),
        "gamma_regime": r.get("gamma_regime", "flat"),
        "n_options":    int(r.get("n_options", 0) or 0),
    }


def load_gex_by_strike() -> list[dict[str, Any]]:
    """
    Lista de {strike, put_call, open_int, gamma, gex_bn}.
    Fonte: SQLite → CSV fallback.
    """
    ql = _try_load_from_db()
    if ql:
        data = ql.get_gex_by_strike()
        if data:
            _log.info("gex_by_strike_from_db", rows=len(data))
            return data

    rows = _read_csv("gex_spx")
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


def load_letf_flows() -> list[dict[str, Any]]:
    """
    Lista de {ticker, leverage, index, nav, aum_b}.
    Fonte: SQLite → CSV fallback.
    """
    ql = _try_load_from_db()
    if ql:
        data = ql.get_letf_flows()
        if data:
            _log.info("letf_flows_from_db", rows=len(data))
            return data

    rows = _read_csv("letf_flows")
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


def load_skew_tails() -> dict[str, dict[str, Any]]:
    """
    Lê skew_tails_*.csv (3 tenores × 11 campos por ticker).

    Para cada ticker, retorna:
      atm_<T>, put25_<T>, call25_<T>, put10_<T>, call10_<T>,
      call_skew_<T>, put_skew_<T>, skew_25d_<T>, rr_25d_<T>,
      skew_10d_<T>, tail_premium_<T>
    onde T ∈ {30D, 90D, 180D}.

    Conveniências (tenor 30D = principal):
      skew_5pct → skew_25d_30D
      put_skew  → put_skew_30D
      call_skew → call_skew_30D
      tail_premium → tail_premium_30D
    """
    rows = _read_csv("skew_tails")
    result: dict[str, dict[str, Any]] = {}
    if not rows:
        return result

    _NUMERIC_COLS = [
        "atm_30D", "atm_90D", "atm_180D",
        "put25_30D", "put25_90D", "put25_180D",
        "call25_30D", "call25_90D", "call25_180D",
        "put10_30D", "put10_90D", "put10_180D",
        "call10_30D", "call10_90D", "call10_180D",
        "call_skew_30D", "put_skew_30D", "skew_25d_30D",
        "rr_25d_30D", "skew_10d_30D", "tail_premium_30D",
        "call_skew_90D", "put_skew_90D", "skew_25d_90D",
        "rr_25d_90D", "skew_10d_90D", "tail_premium_90D",
        "call_skew_180D", "put_skew_180D", "skew_25d_180D",
        "rr_25d_180D", "skew_10d_180D", "tail_premium_180D",
    ]

    for r in rows:
        tk = (r.get("ticker") or "").strip()
        if not tk:
            continue
        entry: dict[str, Any] = {}
        for col in _NUMERIC_COLS:
            entry[col] = _float(r.get(col))
        # Convenience aliases (tenor 30D = principal)
        entry["skew_5pct"]    = entry.get("skew_25d_30D")
        entry["put_skew"]     = entry.get("put_skew_30D")
        entry["call_skew"]    = entry.get("call_skew_30D")
        entry["tail_premium"] = entry.get("tail_premium_30D")
        entry["rr_25d"]       = entry.get("rr_25d_30D")
        entry["atm_iv"]       = entry.get("atm_30D")
        result[tk] = entry

    if result:
        _log.info("skew_tails_loaded", n=len(result))
    return result


def load_positioning_models() -> dict[str, dict[str, Any]]:
    """
    Lê positioning_models_*.csv (CTA + VolCtrl + Risk Parity por ticker).

    Retorna {ticker: {price, rv_5d, rv_30d, rv_60d,
                      cta_score, cta_leverage, cta_notional_b,
                      volctrl_score, volctrl_leverage, volctrl_notional_b,
                      rp_score, rp_weight, rp_notional_b,
                      flow_total_b, flow_direction}}.
    """
    rows = _read_csv("positioning_models")
    result: dict[str, dict[str, Any]] = {}
    for r in rows:
        tk = (r.get("ticker") or "").strip()
        if not tk:
            continue
        result[tk] = {
            "price":              _float(r.get("price")),
            "rv_5d":              _float(r.get("rv_5d")),
            "rv_30d":             _float(r.get("rv_30d")),
            "rv_60d":             _float(r.get("rv_60d")),
            "cta_sig_20d":        _float(r.get("cta_sig_20d")),
            "cta_sig_60d":        _float(r.get("cta_sig_60d")),
            "cta_sig_120d":       _float(r.get("cta_sig_120d")),
            "cta_score":          _float(r.get("cta_score")),
            "cta_leverage":       _float(r.get("cta_leverage")),
            "cta_notional_b":     _float(r.get("cta_notional_b")),
            "volctrl_score":      _float(r.get("volctrl_score")),
            "volctrl_leverage":   _float(r.get("volctrl_leverage")),
            "volctrl_notional_b": _float(r.get("volctrl_notional_b")),
            "rp_score":           _float(r.get("rp_score")),
            "rp_weight":          _float(r.get("rp_weight")),
            "rp_notional_b":      _float(r.get("rp_notional_b")),
            "flow_total_b":       _float(r.get("flow_total_b")),
            "flow_direction":     (r.get("flow_direction") or "").strip(),
        }
    if result:
        _log.info("positioning_models_loaded", n=len(result))
    return result


def load_prices() -> dict[str, Any]:
    """
    Retorna {yf_ticker: {name, price, daily_return, weekly_return, ytd_return}}.
    Fonte: SQLite → CSV fallback.
    Nota: market_prices.py também usa query_layer diretamente.
    """
    ql = _try_load_from_db()
    if ql:
        data = ql.get_latest_prices()
        if data:
            _log.info("prices_from_db", tickers=len(data))
            return data

    rows = _read_csv("prices")
    if not rows:
        return {}
    result: dict[str, Any] = {}
    for r in rows:
        tk = (r.get("yf_ticker") or r.get("bbg_ticker") or "").strip()
        if not tk:
            continue
        entry: dict[str, Any] = {"name": r.get("name", tk)}
        for field in ["price", "daily_return", "weekly_return", "ytd_return"]:
            v = _float(r.get(field))
            if v is not None:
                entry[field] = v
        result[tk] = entry
    _log.info("prices_from_csv", tickers=len(result))
    return result


def load_price_history(days: int = 252) -> dict[str, list[float]]:
    """
    Retorna {yf_ticker: [close_0, ..., close_N]} ordenado por data asc.
    Fonte: SQLite → CSV fallback.
    """
    ql = _try_load_from_db()
    if ql:
        raw = ql.get_price_history(days=days)
        if raw:
            result: dict[str, list[float]] = {}
            for ticker, series in raw.items():
                prices = [p["price"] for p in series if p.get("price") is not None]
                if prices:
                    result[ticker] = prices
            if result:
                _log.info("price_history_from_db", tickers=len(result))
                return result

    rows = _read_csv("price_history")
    if not rows:
        return {}
    from collections import defaultdict
    hist: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in rows:
        tk  = (r.get("yf_ticker") or r.get("ticker") or "").strip()
        dt  = r.get("date", "").strip()
        px  = _float(r.get("price"))
        if tk and dt and px is not None:
            hist[tk].append((dt, px))
    result = {}
    for tk, entries in hist.items():
        entries.sort(key=lambda x: x[0])
        result[tk] = [px for _, px in entries]
    _log.info("price_history_from_csv", tickers=len(result))
    return result


def load_macro_series() -> dict[str, dict[str, Any]]:
    """
    Retorna séries macroeconômicas do banco SQLite.
    Inclui curva de juros, spreads de crédito, VIX term structure, DXY, etc.
    Disponível apenas quando collect_macro_series.py foi rodado no BQuant.

    Returns:
        {
          "USGG10YR": {"px_last": 4.25, "date": "2026-04-03"},
          "USGG2YR":  {"px_last": 4.10, "date": "2026-04-03"},
          "LUACOAS":  {"px_last": 92.3, "date": "2026-04-03"},
          ...
        }
    """
    ql = _try_load_from_db()
    if ql:
        try:
            data = ql.get_macro_series()
            if data:
                _log.info("macro_series_from_db", tickers=len(data))
                return data
        except AttributeError:
            pass  # versão antiga do query_layer — sem get_macro_series

    # Fallback CSV
    rows = _read_csv("macro_series")
    if not rows:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for r in rows:
        ticker = (r.get("bbg_ticker") or r.get("ticker") or "").strip()
        if not ticker:
            continue
        entry: dict[str, Any] = {}
        for field in ["px_last", "yield", "spread", "value"]:
            v = _float(r.get(field))
            if v is not None:
                entry["px_last"] = v
                break
        if r.get("date"):
            entry["date"] = r["date"]
        if entry:
            result[ticker] = entry
    _log.info("macro_series_from_csv", tickers=len(result))
    return result


def load_all() -> dict[str, Any]:
    """
    Carrega todos os dados de uma vez.
    Usa SQLite quando disponível, CSV como fallback.
    """
    ql = _try_load_from_db()
    source = "db" if ql else "csv"
    _log.info("load_all_start", source=source)

    return {
        "fundamentals": load_fundamentals(),
        "options_iv":   load_options_iv(),
        "gex_summary":  load_gex_summary(),
        "gex_strikes":  load_gex_by_strike(),
        "letf_flows":   load_letf_flows(),
        "macro_series": load_macro_series(),
        "source":       source,
        "fresh":        _check_freshness(),
    }
