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
    Retorna agenda de releases econômicos dos próximos dias.

    Usa calendário computado baseado nos padrões de divulgação conhecidos
    dos principais indicadores econômicos dos EUA (NFP, CPI, FOMC, etc.).
    O endpoint FRED /releases/dates só retorna datas históricas — não futuras.
    """
    releases = _compute_scheduled_calendar(days_ahead)
    _log.info("fred_calendar_done", releases=len(releases))
    return releases


# ── Calendário econômico computado ────────────────────────────────────────────

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Retorna a n-ésima ocorrência (1-based) do weekday (0=Mon…6=Sun) no mês."""
    first = date(year, month, 1)
    diff = (weekday - first.weekday()) % 7
    d = first + timedelta(days=diff + (n - 1) * 7)
    return d


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Retorna a última ocorrência do weekday no mês."""
    import calendar as _cal
    last_day = _cal.monthrange(year, month)[1]
    last = date(year, month, last_day)
    diff = (last.weekday() - weekday) % 7
    return last - timedelta(days=diff)


def _first_business_day(year: int, month: int) -> date:
    """Retorna o primeiro dia útil do mês."""
    d = date(year, month, 1)
    while d.weekday() >= 5:  # 5=Sáb, 6=Dom
        d += timedelta(days=1)
    return d


def _nth_business_day(year: int, month: int, n: int) -> date:
    """Retorna o n-ésimo dia útil do mês."""
    d = date(year, month, 1)
    count = 0
    while True:
        if d.weekday() < 5:
            count += 1
            if count == n:
                return d
        d += timedelta(days=1)


def _compute_scheduled_calendar(days_ahead: int = 14) -> list[dict]:
    """
    Computa as datas aproximadas dos principais releases econômicos dos EUA
    para o período today … today+days_ahead.

    Padrões usados (aproximações dos calendários do BLS, BEA, Fed, ISM):
      - ISM Manufacturing    : 1º dia útil do mês
      - ISM Non-Manufacturing: 3º dia útil do mês
      - ADP Employment       : quarta-feira antes do NFP
      - NFP / Employment     : 1ª sexta-feira do mês
      - JOLTS                : 5ª semana após mês de referência (~terça da 5ª semana)
      - CPI                  : ~12-14 dias após fim do mês (terça/quarta da 3ª semana)
      - PPI                  : ~13-15 dias após fim do mês (dia seguinte ao CPI)
      - Retail Sales         : ~15-17 dias após fim do mês (quarta da 3ª semana)
      - Housing Starts       : ~3ª semana do mês
      - PCE / Income         : último dia útil do mês corrente (mês anterior)
      - GDP Advance          : ~últimos dias do mês seguinte ao fim do trimestre
      - Michigan Sentiment   : 2ª sexta do mês (preliminar), última sexta (final)
      - FOMC (2026)          : datas fixas hardcoded
    """
    today = date.today()
    window_end = today + timedelta(days=days_ahead)

    # Coletar releases para o mês atual e próximo
    months_to_check = set()
    d = today.replace(day=1)
    while d <= window_end + timedelta(days=35):
        months_to_check.add((d.year, d.month))
        # avança 1 mês
        if d.month == 12:
            d = date(d.year + 1, 1, 1)
        else:
            d = date(d.year, d.month + 1, 1)

    events: list[tuple[date, str]] = []

    for yr, mo in sorted(months_to_check):
        import calendar as _cal
        last_day = _cal.monthrange(yr, mo)[1]

        # Mês anterior (para referência de JOLTS, PCE)
        if mo == 1:
            prev_yr, prev_mo = yr - 1, 12
        else:
            prev_yr, prev_mo = yr, mo - 1

        # ── ISM Manufacturing (1º dia útil) ─────────────────────────────────
        events.append((_first_business_day(yr, mo), "ISM Manufacturing PMI"))

        # ── ISM Non-Manufacturing (3º dia útil) ──────────────────────────────
        events.append((_nth_business_day(yr, mo, 3), "ISM Non-Manufacturing PMI (Services)"))

        # ── NFP — 1ª sexta-feira ──────────────────────────────────────────────
        nfp_date = _nth_weekday(yr, mo, 4, 1)  # 4=Sexta
        events.append((nfp_date, "Employment Situation (Nonfarm Payrolls / NFP)"))

        # ── ADP — quarta antes do NFP ─────────────────────────────────────────
        adp_date = nfp_date - timedelta(days=2)
        events.append((adp_date, "ADP National Employment Report"))

        # ── JOLTS — ~terça da 5ª semana do mês (referência: mês anterior) ────
        # Tipicamente divulgado ~5 semanas após o mês de referência
        jolts_candidate = _nth_weekday(yr, mo, 1, 2)  # 2ª terça
        events.append((jolts_candidate, "JOLTS Job Openings (referência mês anterior)"))

        # ── CPI — terça ou quarta da 3ª semana ───────────────────────────────
        # Tipicamente ao redor do dia 12 do mês
        cpi_anchor = date(yr, mo, 12)
        while cpi_anchor.weekday() >= 5 or cpi_anchor.weekday() == 0:  # não sáb/dom/seg
            cpi_anchor += timedelta(days=1)
        events.append((cpi_anchor, "Consumer Price Index (CPI)"))

        # ── PPI — dia seguinte ao CPI ─────────────────────────────────────────
        ppi_date = cpi_anchor + timedelta(days=1)
        if ppi_date.weekday() >= 5:
            ppi_date += timedelta(days=2)
        events.append((ppi_date, "Producer Price Index (PPI)"))

        # ── Retail Sales — ~15-17 do mês ─────────────────────────────────────
        retail_anchor = date(yr, mo, 16)
        while retail_anchor.weekday() >= 5:
            retail_anchor += timedelta(days=1)
        events.append((retail_anchor, "Retail Sales"))

        # ── Housing Starts — 3ª semana ────────────────────────────────────────
        hs_date = date(yr, mo, 17)
        while hs_date.weekday() >= 5:
            hs_date += timedelta(days=1)
        events.append((hs_date, "Housing Starts & Building Permits"))

        # ── PCE / Personal Income — último dia útil do mês ───────────────────
        last_bd = date(yr, mo, last_day)
        while last_bd.weekday() >= 5:
            last_bd -= timedelta(days=1)
        events.append((last_bd, "PCE Price Index / Personal Income & Spending"))

        # ── GDP Advance — meses de Jan, Apr, Jul, Oct (fim de trimestre +1 mês)
        if mo in (1, 4, 7, 10):
            gdp_date = date(yr, mo, 28)
            while gdp_date.weekday() >= 5:
                gdp_date += timedelta(days=1)
            events.append((gdp_date, "GDP Advance Estimate"))

        # ── Michigan Consumer Sentiment ───────────────────────────────────────
        mich_prelim = _nth_weekday(yr, mo, 4, 2)  # 2ª sexta (preliminar)
        mich_final = _last_weekday(yr, mo, 4)     # última sexta (final)
        events.append((mich_prelim, "Michigan Consumer Sentiment (Preliminary)"))
        if mich_final != mich_prelim:
            events.append((mich_final, "Michigan Consumer Sentiment (Final)"))

        # ── Existing Home Sales — ~3ª semana ─────────────────────────────────
        ehs_date = date(yr, mo, 21)
        while ehs_date.weekday() >= 5:
            ehs_date += timedelta(days=1)
        events.append((ehs_date, "Existing Home Sales"))

        # ── Chicago PMI — último dia útil do mês ─────────────────────────────
        events.append((last_bd, "Chicago PMI"))

    # ── FOMC 2026 (datas oficiais) ───────────────────────────────────────────
    _FOMC_2026 = [
        date(2026, 1, 29), date(2026, 3, 19), date(2026, 5, 7),
        date(2026, 6, 18), date(2026, 7, 30), date(2026, 9, 17),
        date(2026, 10, 29), date(2026, 12, 10),
    ]
    # FOMC 2025 (para contexto histórico)
    _FOMC_2025 = [
        date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
        date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
        date(2025, 10, 29), date(2025, 12, 10),
    ]
    for fd in _FOMC_2026 + _FOMC_2025:
        events.append((fd, "FOMC Meeting Decision (Federal Reserve)"))
        events.append((fd - timedelta(days=1), "FOMC Meeting Day 1"))

    # ── Treasury Auctions — 2ª e 4ª quarta-feira (10yr) ─────────────────────
    for yr, mo in sorted(months_to_check):
        events.append((_nth_weekday(yr, mo, 2, 2), "Treasury 10yr Auction"))
        events.append((_nth_weekday(yr, mo, 2, 4), "Treasury 10yr Auction"))

    # ── Filtra janela ─────────────────────────────────────────────────────────
    releases = []
    seen = set()
    for ev_date, ev_name in sorted(events):
        if today <= ev_date <= window_end:
            key = f"{ev_date}|{ev_name}"
            if key not in seen:
                seen.add(key)
                releases.append({
                    "date": ev_date.isoformat(),
                    "release_name": ev_name,
                    "release_id": None,
                    "source": "computed",
                })

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
