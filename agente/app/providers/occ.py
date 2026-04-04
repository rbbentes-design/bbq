"""
Provider: OCC (Options Clearing Corporation) — Acesso público, sem autenticação

Dados disponíveis via batch processing:
  - Open Interest diário por underlying (calls + puts)
  - Volume diário (calls + puts)
  - Put/Call Ratio (OI e volume)

URL: https://marketdata.theocc.com/daily-open-interest
Formato: CSV, download direto, sem credenciais.

Uso: fallback gratuito para options.py quando IBKR indisponível.
     Fornece PCR e OI total, mas SEM Greeks/IV.
"""

from __future__ import annotations

import csv
import io
import time
import urllib.request
from datetime import date, timedelta
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("providers.occ")

_BASE_URL = "https://marketdata.theocc.com/daily-open-interest"

# Cache em memória: evita múltiplos downloads no mesmo dia
_cache: dict[str, dict[str, dict]] = {}   # date_str → {ticker: stats}


def _fetch_csv(report_date: str) -> str | None:
    """
    Baixa o CSV de open interest do dia.
    report_date: "MM/DD/YYYY"
    """
    url = f"{_BASE_URL}?reportDate={report_date}&action=download&format=csv"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            content = r.read().decode("utf-8", errors="replace")
        return content
    except Exception as exc:
        _log.warning("occ_fetch_error", date=report_date, error=str(exc))
        return None


def _parse_csv(raw: str) -> dict[str, dict[str, Any]]:
    """
    Parseia o CSV do OCC.
    Colunas típicas: Symbol, Call OI, Put OI, Call Volume, Put Volume, ...
    Retorna {symbol: {call_oi, put_oi, call_vol, put_vol, pcr_oi, pcr_vol}}
    """
    results: dict[str, dict[str, Any]] = {}
    try:
        reader = csv.DictReader(io.StringIO(raw))
        for row in reader:
            # Normaliza nomes de coluna (OCC muda às vezes)
            row_lower = {k.strip().lower(): v.strip() for k, v in row.items() if k}

            sym = (
                row_lower.get("symbol")
                or row_lower.get("underlying symbol")
                or row_lower.get("class symbol")
                or ""
            ).strip().upper()
            if not sym:
                continue

            def _int(key: str) -> int:
                for k in [key, key.replace(" ", "_"), key.replace("_", " ")]:
                    v = row_lower.get(k, "").replace(",", "").strip()
                    if v:
                        try:
                            return int(float(v))
                        except ValueError:
                            pass
                return 0

            call_oi  = _int("call open int") or _int("call_oi") or _int("calls oi")
            put_oi   = _int("put open int")  or _int("put_oi")  or _int("puts oi")
            call_vol = _int("call volume")   or _int("call_vol")
            put_vol  = _int("put volume")    or _int("put_vol")

            entry: dict[str, Any] = {
                "call_oi":  call_oi,
                "put_oi":   put_oi,
                "call_vol": call_vol,
                "put_vol":  put_vol,
                "pcr_oi":   round(put_oi  / call_oi,  3) if call_oi  > 0 else None,
                "pcr_vol":  round(put_vol / call_vol, 3) if call_vol > 0 else None,
                "source":   "occ",
            }
            results[sym] = entry

    except Exception as exc:
        _log.warning("occ_parse_error", error=str(exc))
    return results


def collect_daily_stats(
    report_date: str | None = None,
    retry_days: int = 3,
) -> dict[str, dict[str, Any]]:
    """
    Retorna estatísticas diárias de options por underlying.

    Args:
        report_date: "MM/DD/YYYY" — padrão: hoje. Se não disponível, tenta dias anteriores.
        retry_days:  quantos dias anteriores tentar se hoje não estiver disponível.

    Returns:
        {ticker: {call_oi, put_oi, call_vol, put_vol, pcr_oi, pcr_vol, source}}
        Retorna {} se download falhar.
    """
    if report_date is None:
        target = date.today()
    else:
        target = date.strptime(report_date, "%m/%d/%Y") if isinstance(report_date, str) else report_date

    for delta in range(retry_days + 1):
        d = target - timedelta(days=delta)
        # Pula fins de semana
        if d.weekday() >= 5:
            continue
        date_str = d.strftime("%m/%d/%Y")

        if date_str in _cache:
            _log.debug("occ_cache_hit", date=date_str)
            return _cache[date_str]

        raw = _fetch_csv(date_str)
        if raw and len(raw) > 200:
            parsed = _parse_csv(raw)
            if parsed:
                _cache[date_str] = parsed
                _log.info("occ_stats_loaded", date=date_str, tickers=len(parsed))
                return parsed

        time.sleep(0.5)

    _log.warning("occ_no_data", attempted_date=target.isoformat())
    return {}


def get_pcr(ticker: str, stats: dict[str, dict] | None = None) -> dict[str, float | None]:
    """
    Retorna PCR por OI e por volume para um ticker.
    Utilitário para uso em análises.
    """
    if stats is None:
        stats = collect_daily_stats()
    entry = stats.get(ticker.upper(), {})
    return {
        "pcr_oi":  entry.get("pcr_oi"),
        "pcr_vol": entry.get("pcr_vol"),
        "call_oi": entry.get("call_oi"),
        "put_oi":  entry.get("put_oi"),
    }
