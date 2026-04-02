"""
Earnings Calendar Filter

Identifica ativos com eventos binarios iminentes (earnings, FDA, FOMC, etc.)
e ajusta o tamanho das posicoes para evitar risco de gap.

Logica:
  - 0-2 dias antes de earnings: excluir da otimizacao (too risky)
  - 3-5 dias antes de earnings: reduzir posicao em 60%
  - 6-10 dias antes: reduzir posicao em 30%
  - Pos-earnings (1-2 dias): aumentar posicao se sinal confirmar
    (o mercado ja precificou, volatilidade cai = comprar premium em opcoes)

Fontes:
  - yfinance: ticker.calendar (proximo earnings)
  - FOMC calendar: hardcoded para 2026 (reunioes publicas)
  - CPI/NFP: hardcoded para 2026

Output: dict[ticker, EventRisk] com fator de escala [0, 1]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("analysis.earnings_filter")


@dataclass
class EventRisk:
    ticker: str
    event_type: str         # "earnings" | "fomc" | "cpi" | "nfp" | "fda"
    event_date: str | None  # YYYY-MM-DD
    days_to_event: int | None
    position_scalar: float  # [0, 1] — multiplicar tamanho da posicao
    avoid: bool             # True = nao abrir posicao
    rationale: str


# ── FOMC Meetings 2026 ─────────────────────────────────────────────────────────
# Federal Reserve meeting dates (FOMC statement day = quarta-feira)
_FOMC_2026 = [
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
]

# ── CPI Release 2026 (aprox. 2a semana de cada mes) ───────────────────────────
_CPI_2026 = [
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-10",
    "2026-05-13", "2026-06-10", "2026-07-14", "2026-08-12",
    "2026-09-09", "2026-10-14", "2026-11-12", "2026-12-10",
]

# ── NFP 2026 (1a sexta-feira de cada mes) ─────────────────────────────────────
_NFP_2026 = [
    "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03",
    "2026-05-08", "2026-06-05", "2026-07-10", "2026-08-07",
    "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04",
]


def _days_to_nearest(dates: list[str], today: date | None = None) -> int | None:
    """Retorna dias ate o proximo evento na lista (ou None se nao houver)."""
    if today is None:
        today = date.today()
    min_days = None
    for d_str in dates:
        try:
            d = date.fromisoformat(d_str)
            diff = (d - today).days
            if -1 <= diff <= 14:  # janela de relevancia: ontem ate 14 dias
                if min_days is None or abs(diff) < abs(min_days):
                    min_days = diff
        except Exception:
            continue
    return min_days


def _fetch_earnings_date(ticker: str) -> date | None:
    """Busca proximo earnings date via yfinance."""
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is None:
            return None
        # yfinance retorna um dict com 'Earnings Date' como Timestamp ou lista
        if hasattr(cal, "get"):
            ed = cal.get("Earnings Date")
        elif hasattr(cal, "loc"):
            # DataFrame
            try:
                ed = cal.loc["Earnings Date"]
            except Exception:
                return None
        else:
            return None

        if ed is None:
            return None

        # Pode ser Timestamp, list, ou string
        if hasattr(ed, "__iter__") and not isinstance(ed, str):
            ed = list(ed)[0] if len(list(ed)) > 0 else None
        if ed is None:
            return None

        if hasattr(ed, "date"):
            return ed.date()
        return date.fromisoformat(str(ed)[:10])
    except Exception as exc:
        _log.debug("earnings_fetch_failed", ticker=ticker, error=str(exc)[:60])
        return None


def compute_event_risks(
    tickers: list[str],
    today: date | None = None,
    use_macro_calendar: bool = True,
) -> dict[str, EventRisk]:
    """
    Calcula o risco de evento para cada ativo.
    Returns: dict[ticker, EventRisk]
    """
    if today is None:
        today = date.today()

    result: dict[str, EventRisk] = {}

    # ── Macro calendar events ─────────────────────────────────────────────────
    fomc_days = _days_to_nearest(_FOMC_2026, today) if use_macro_calendar else None
    cpi_days  = _days_to_nearest(_CPI_2026, today)  if use_macro_calendar else None
    nfp_days  = _days_to_nearest(_NFP_2026, today)  if use_macro_calendar else None

    # Event mais proximo do calendario macro
    macro_days: int | None = None
    macro_type = "macro"
    for dtype, days in [("fomc", fomc_days), ("cpi", cpi_days), ("nfp", nfp_days)]:
        if days is not None and (macro_days is None or abs(days) < abs(macro_days)):
            macro_days = days
            macro_type = dtype

    # ── Per-ticker earnings ───────────────────────────────────────────────────
    for ticker in tickers:
        # Skip macro-insensitive tickers (options, vol ETFs)
        if ticker in ("VXX", "UVXY", "SVXY", "BIL", "SHY", "IEF", "TLT"):
            result[ticker] = EventRisk(
                ticker=ticker,
                event_type="none",
                event_date=None,
                days_to_event=None,
                position_scalar=1.0,
                avoid=False,
                rationale="ETF sem evento binario individual",
            )
            continue

        earnings_date = _fetch_earnings_date(ticker)
        earnings_days: int | None = None
        if earnings_date:
            earnings_days = (earnings_date - today).days

        # Determina o risco dominante
        scalar = 1.0
        avoid = False
        event_type = "none"
        event_date_str = None
        days_to_event = None
        rationale = "Sem evento iminente"

        # Check earnings (mais relevante para acoes individuais)
        if earnings_days is not None and -1 <= earnings_days <= 14:
            days_to_event = earnings_days
            event_type = "earnings"
            event_date_str = earnings_date.isoformat() if earnings_date else None

            if earnings_days <= 1:  # amanha ou ontem (dia do earnings)
                avoid = True
                scalar = 0.0
                rationale = f"EARNINGS {earnings_date} em {earnings_days}d — AVOID (risco binario)"
            elif earnings_days <= 2:
                avoid = True
                scalar = 0.0
                rationale = f"EARNINGS {earnings_date} em {earnings_days}d — AVOID"
            elif earnings_days <= 5:
                scalar = 0.35
                rationale = f"EARNINGS {earnings_date} em {earnings_days}d — posicao 35%"
            elif earnings_days <= 10:
                scalar = 0.60
                rationale = f"EARNINGS {earnings_date} em {earnings_days}d — posicao 60%"
            else:
                scalar = 0.85
                rationale = f"EARNINGS {earnings_date} em {earnings_days}d — posicao 85%"

        # Check macro (afeta todos os ativos beta-alto)
        elif macro_days is not None and abs(macro_days) <= 2:
            days_to_event = macro_days
            event_type = macro_type
            avoid = False  # macro nao e tao binario quanto earnings individuais
            scalar = 0.70
            rationale = f"{macro_type.upper()} em {macro_days}d — reduz posicao para 70%"

        result[ticker] = EventRisk(
            ticker=ticker,
            event_type=event_type,
            event_date=event_date_str,
            days_to_event=days_to_event,
            position_scalar=scalar,
            avoid=avoid,
            rationale=rationale,
        )

        if avoid:
            _log.info("event_avoid", ticker=ticker, event_type=event_type, days=days_to_event)

    return result


def apply_event_scalars(
    positions: list,  # list[PositionResult]
    event_risks: dict[str, EventRisk],
) -> list:
    """
    Aplica os fatores de evento nas alocacoes.
    Retorna posicoes com allocation_usd e allocation_pct ajustados.
    """
    adjusted = []
    for pos in positions:
        risk = event_risks.get(pos.ticker)
        if risk is None or risk.avoid:
            if risk and risk.avoid:
                _log.info("position_excluded_event", ticker=pos.ticker, reason=risk.rationale)
            elif not risk:
                adjusted.append(pos)
            continue

        if risk.position_scalar < 1.0:
            import copy
            p = copy.copy(pos)
            p.allocation_usd = pos.allocation_usd * risk.position_scalar
            p.allocation_pct = pos.allocation_pct * risk.position_scalar
            if hasattr(p, "rationale") and p.rationale:
                p.rationale = p.rationale + [f"Evento: {risk.rationale}"]
            adjusted.append(p)
            _log.info("position_scaled_event",
                      ticker=pos.ticker,
                      scalar=risk.position_scalar,
                      reason=risk.rationale)
        else:
            adjusted.append(pos)

    return adjusted
