"""
Provider: FRED (Federal Reserve Economic Data)

Coleta séries históricas macro via API do St. Louis Fed.
Documentação: https://fred.stlouisfed.org/docs/api/fred/

Séries coletadas por categoria:
  - Política monetária: Fed Funds, balanço Fed, curva de juros
  - Inflação: CPI, PCE, PPI, expectativas de inflação
  - Trabalho: desemprego, payroll, JOLTS, participação
  - Crescimento: PIB, PMI, consumo, investimento
  - Crédito: spreads, condições financeiras, defaults
  - Externo: dólar, comércio
"""

from __future__ import annotations

import json
import urllib.request
import urllib.parse
from datetime import date, timedelta
from typing import Any

from app.audit.logger import get_logger
from app.config.settings import settings

_log = get_logger("providers.fred")

_BASE_URL = "https://api.stlouisfed.org/fred"

# ── Séries por categoria ───────────────────────────────────────────────────────
# (series_id, label, unidade)
SERIES_CATALOG: dict[str, list[tuple[str, str, str]]] = {
    "Política Monetária": [
        ("FEDFUNDS",    "Fed Funds Rate",                    "%"),
        ("DFF",         "Fed Funds Rate (diário)",           "%"),
        ("T10Y2Y",      "Curva 10y-2y (spread)",             "pp"),
        ("T10YIE",      "Breakeven Inflação 10y",            "%"),
        ("WALCL",       "Balanço do Fed",                    "USD bi"),
    ],
    "Inflação": [
        ("CPIAUCSL",    "CPI (total)",                       "YoY%"),
        ("CPILFESL",    "CPI Core (ex-food/energy)",         "YoY%"),
        ("PCEPI",       "PCE (total)",                       "YoY%"),
        ("PCEPILFE",    "PCE Core",                          "YoY%"),
        ("PPIFIS",      "PPI (indústria)",                   "YoY%"),
        ("MICH",        "Expectativa Inflação Michigan",     "%"),
    ],
    "Mercado de Trabalho": [
        ("UNRATE",      "Taxa de Desemprego",                "%"),
        ("PAYEMS",      "Payroll não-agrícola",              "mil"),
        ("U6RATE",      "Desemprego amplo (U6)",             "%"),
        ("JTSJOL",      "JOLTS — Vagas abertas",             "mil"),
        ("CIVPART",     "Taxa de Participação",              "%"),
        ("AHETOT",      "Salário médio por hora",            "USD"),
    ],
    "Crescimento": [
        ("GDP",         "PIB nominal",                       "USD bi"),
        ("GDPC1",       "PIB real",                          "USD bi"),
        ("GDPCA",       "PIB real (crescimento anual)",      "%"),
        ("PCE",         "Consumo pessoal",                   "USD bi"),
        ("INDPRO",      "Produção industrial",               "índice"),
        ("RETAILSL",    "Vendas no varejo",                  "USD mi"),
    ],
    "Crédito e Condições Financeiras": [
        ("BAMLH0A0HYM2", "Spread High Yield (OAS)",         "pp"),
        ("BAMLC0A0CM",   "Spread Investment Grade (OAS)",   "pp"),
        ("NFCI",         "Índice Cond. Financeiras (NFCI)", "índice"),
        ("DRCCLACBS",    "Taxa default cartão crédito",     "%"),
        ("MORTGAGE30US", "Taxa hipoteca 30 anos",           "%"),
    ],
    "Dólar e Externo": [
        ("DTWEXBGS",    "Índice Dólar (DXY amplo)",          "índice"),
        ("BOPGSTB",     "Balança comercial",                 "USD mi"),
    ],
}


def collect(
    lookback_days: int = 365,
    series_filter: list[str] | None = None,
) -> dict[str, Any]:
    """
    Coleta últimas observações das séries FRED.

    Args:
        lookback_days: janela histórica em dias
        series_filter: lista de series_id para restringir (None = todas)

    Returns:
        Dict por categoria → lista de {series_id, label, unit, value, date, prev_value, change}
    """
    api_key = settings.fred_api_key
    if not api_key:
        _log.warning("fred_api_key_missing")
        return {}

    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    output: dict[str, list[dict]] = {}

    for category, series_list in SERIES_CATALOG.items():
        cat_results = []
        for series_id, label, unit in series_list:
            if series_filter and series_id not in series_filter:
                continue
            obs = _fetch_observations(series_id, start, api_key)
            if obs:
                latest = obs[-1]
                prev = obs[-2] if len(obs) >= 2 else None
                cat_results.append({
                    "series_id": series_id,
                    "label": label,
                    "unit": unit,
                    "value": latest["value"],
                    "date": latest["date"],
                    "prev_value": prev["value"] if prev else None,
                    "prev_date": prev["date"] if prev else None,
                    "change": _change(latest["value"], prev["value"] if prev else None),
                    "history": obs[-12:],  # últimas 12 observações
                })
        if cat_results:
            output[category] = cat_results

    _log.info("fred_done", categories=len(output),
              series=sum(len(v) for v in output.values()))
    return output


def collect_release_calendar(days_ahead: int = 14) -> list[dict]:
    """
    Coleta agenda de releases econômicos do FRED para os próximos dias.
    """
    api_key = settings.fred_api_key
    if not api_key:
        return []

    today = date.today()
    end = today + timedelta(days=days_ahead)

    url = (
        f"{_BASE_URL}/releases/dates"
        f"?api_key={api_key}"
        f"&realtime_start={today.isoformat()}"
        f"&realtime_end={end.isoformat()}"
        f"&include_release_dates_with_no_data=false"
        f"&file_type=json"
    )

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        _log.warning("fred_calendar_error", error=str(exc))
        return []

    releases = []
    for rd in data.get("release_dates", []):
        releases.append({
            "date": rd.get("date"),
            "release_name": rd.get("release_name"),
            "release_id": rd.get("release_id"),
        })

    _log.info("fred_calendar_done", releases=len(releases))
    return releases


# ── Internos ──────────────────────────────────────────────────────────────────

def _fetch_observations(series_id: str, start: str, api_key: str) -> list[dict]:
    """Busca observações de uma série FRED."""
    params = urllib.parse.urlencode({
        "series_id": series_id,
        "api_key": api_key,
        "observation_start": start,
        "sort_order": "asc",
        "file_type": "json",
    })
    url = f"{_BASE_URL}/series/observations?{params}"

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        obs = [
            {"date": o["date"], "value": _parse_value(o["value"])}
            for o in data.get("observations", [])
            if o.get("value") not in (".", "")
        ]
        return obs
    except Exception as exc:
        _log.warning("fred_series_error", series=series_id, error=str(exc))
        return []


def _parse_value(v: str) -> float | None:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _change(current: float | None, prev: float | None) -> float | None:
    if current is None or prev is None or prev == 0:
        return None
    return round(current - prev, 4)
