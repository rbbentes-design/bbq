"""
FINRA ADF/TRF — Dark Pool Volume por Ativo (T+1, gratuito)

Fonte: FINRA OTC Transparency Data
  https://api.finra.org/data/group/otcMarket/name/weeklySummary

Métricas extraídas por ticker:
  - otc_volume        : volume off-exchange (dark pool + ATSs)
  - total_volume      : volume total reportado à FINRA
  - dark_pct          : otc_volume / total_volume  ← % em dark pool
  - dark_pool_score   : [-1, 1] derivado do dark_pct vs média histórica
  - trade_count       : número de trades off-exchange

Por que isso importa:
  - dark_pct > 50% = institucional preferindo dark pools → acumulação ou distribuição sigilosa
  - dark_pct subindo semana a semana = fluxo se intensificando
  - Combinado com direção do preço: sinaliza acumulação (↑price + ↑dark%) ou distribuição
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("providers.finra_dark_pool")

_BASE = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"
_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; MacroDesk/1.0)",
}
_CACHE_TTL_H = 20  # dados são semanais, mas FINRA atualiza intraday


@dataclass
class FinraDarkPoolSignal:
    ticker: str
    otc_volume: int = 0
    total_volume: int = 0
    dark_pct: float = 0.0          # [0, 1] — % do volume em dark pool
    dark_pct_prev: float = 0.0     # semana anterior (se disponível)
    dark_pct_delta: float = 0.0    # mudança semana a semana
    trade_count: int = 0
    week_ending: str = ""
    dark_pool_score: float = 0.0   # [-1, 1] derivado vs baseline
    signal: str = "neutral"        # "accumulation" | "distribution" | "neutral"
    rationale: list[str] = field(default_factory=list)
    source: str = "finra_trf"
    timestamp: str = ""


@dataclass
class FinraDarkPoolResult:
    signals: dict[str, FinraDarkPoolSignal] = field(default_factory=dict)
    market_dark_pct: float = 0.0   # média ponderada do mercado
    top_dark: list[str] = field(default_factory=list)   # maior % dark pool
    top_light: list[str] = field(default_factory=list)  # menor % dark pool
    timestamp: str = ""
    errors: list[str] = field(default_factory=list)


# ── Fetch FINRA API ──────────────────────────────────────────────────────────

def _fetch_ticker(session, ticker: str, limit: int = 2) -> list[dict]:
    """Busca últimas N semanas de dados OTC para um ticker."""
    try:
        params = {
            "compareFilters": f"issueSymbolIdentifier:eq:{ticker.upper()}",
            "fields": "weeklyStartDate,weeklyEndDate,issueSymbolIdentifier,"
                      "otcShareVolume,totalWeeklyShareVolume,otcTradeCount",
            "limit": limit,
            "sortFields": "-weeklyEndDate",
        }
        resp = session.get(_BASE, params=params, headers=_HEADERS, timeout=10)
        if resp.status_code == 200:
            return resp.json() if isinstance(resp.json(), list) else []
    except Exception as exc:
        _log.debug("finra_fetch_failed", ticker=ticker, error=str(exc)[:60])
    return []


def _build_signal(ticker: str, rows: list[dict]) -> FinraDarkPoolSignal:
    sig = FinraDarkPoolSignal(ticker=ticker, timestamp=datetime.now().isoformat())
    if not rows:
        return sig

    # Semana mais recente
    r0 = rows[0]
    otc   = int(r0.get("otcShareVolume") or 0)
    total = int(r0.get("totalWeeklyShareVolume") or 1)
    sig.otc_volume   = otc
    sig.total_volume = total
    sig.dark_pct     = otc / total if total > 0 else 0.0
    sig.trade_count  = int(r0.get("otcTradeCount") or 0)
    sig.week_ending  = str(r0.get("weeklyEndDate") or "")

    # Semana anterior para delta
    if len(rows) > 1:
        r1 = rows[1]
        t1 = int(r1.get("totalWeeklyShareVolume") or 1)
        o1 = int(r1.get("otcShareVolume") or 0)
        sig.dark_pct_prev  = o1 / t1 if t1 > 0 else 0.0
        sig.dark_pct_delta = sig.dark_pct - sig.dark_pct_prev

    # Score: baseline ~40% dark pool é normal para large caps
    # Acima de 50% + subindo = acumulação; abaixo de 30% = transparência (ou distribuição detectável)
    baseline = 0.40
    z = (sig.dark_pct - baseline) / 0.10  # normaliza por stddev ~10%
    score = max(-1.0, min(1.0, z * 0.5))

    # Ajuste pelo delta semana a semana
    if abs(sig.dark_pct_delta) > 0.03:
        score += 0.2 * (1 if sig.dark_pct_delta > 0 else -1)
        score = max(-1.0, min(1.0, score))

    sig.dark_pool_score = round(score, 3)

    # Sinal qualitativo
    reasons = []
    if sig.dark_pct > 0.55:
        reasons.append(f"Dark pool {sig.dark_pct:.0%} do volume — institucional off-exchange elevado")
        sig.signal = "accumulation"
    elif sig.dark_pct > 0.45:
        reasons.append(f"Dark pool {sig.dark_pct:.0%} — acima da média ({baseline:.0%})")
        sig.signal = "neutral"
    elif sig.dark_pct < 0.30:
        reasons.append(f"Dark pool baixo {sig.dark_pct:.0%} — volume concentrado em exchanges visíveis")
        sig.signal = "distribution"
    else:
        sig.signal = "neutral"

    if abs(sig.dark_pct_delta) > 0.05:
        direction = "subindo" if sig.dark_pct_delta > 0 else "caindo"
        reasons.append(f"Dark% {direction} {sig.dark_pct_delta:+.1%} vs semana anterior")

    sig.rationale = reasons
    return sig


# ── API pública ───────────────────────────────────────────────────────────────

def collect(tickers: list[str]) -> FinraDarkPoolResult:
    """
    Coleta dark pool % via FINRA OTC Transparency para a lista de tickers.
    Gratuito, sem autenticação, delay T+1 (dados da semana anterior).
    """
    import requests
    result = FinraDarkPoolResult(timestamp=datetime.now().isoformat())

    if not tickers:
        return result

    # Normaliza tickers (remove sufixos Bloomberg se houver)
    clean = []
    for t in tickers:
        for sfx in [" US Equity", " US EQUITY", " Equity"]:
            if t.endswith(sfx):
                t = t[:-len(sfx)].strip()
        clean.append(t.upper())

    session = requests.Session()
    dark_pcts = []

    for ticker in clean[:30]:  # max 30 para não sobrecarregar
        rows = _fetch_ticker(session, ticker)
        sig = _build_signal(ticker, rows)
        result.signals[ticker] = sig
        if sig.dark_pct > 0:
            dark_pcts.append((ticker, sig.dark_pct))
        time.sleep(0.15)  # rate limit gentil

    if dark_pcts:
        result.market_dark_pct = sum(p for _, p in dark_pcts) / len(dark_pcts)
        sorted_dark = sorted(dark_pcts, key=lambda x: x[1], reverse=True)
        result.top_dark  = [t for t, _ in sorted_dark[:5]]
        result.top_light = [t for t, _ in sorted_dark[-5:]]

    _log.info("finra_dark_pool_done",
              tickers=len(result.signals),
              market_dark_pct=round(result.market_dark_pct, 3))
    return result
