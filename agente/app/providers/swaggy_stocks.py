"""
Provider: SwaggyStocks — Reddit/WSB Sentiment + Short Squeeze List

Coleta:
  1. WSB Ticker Sentiment — menções, rank, sentimento (bull/bear), score
  2. Short Squeeze List — short interest, days to cover, borrow rate

URL: https://swaggystocks.com/dashboard/wallstreetbets/ticker-sentiment

Por que isso importa para o motor de narrativa:
  - WSB mentions = proxy de atenção de varejo (early signal de narrativa emergente)
  - Spike de menções + sentimento positivo = narrativa começando a se propagar
  - Menções em pico + preço já alto = sinal de narrativa madura/exaurida
  - Short squeeze list = pool de ativos com combustível mecânico (posicionamento adversarial)
  - Alto short interest + narrativa emergente = assimetria de squeeze

Output: SwaggyResult com wsb_mentions e squeeze_candidates
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import requests

from app.audit.logger import get_logger

_log = get_logger("providers.swaggy_stocks")

_BASE_URL = "https://swaggystocks.com"
_WSB_API  = "https://swaggystocks.com/api/wsb/tickers"   # JSON endpoint (public)
_HEADERS  = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/html",
    "Referer": "https://swaggystocks.com/dashboard/wallstreetbets/ticker-sentiment",
}
_TIMEOUT = 15


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class WSBMention:
    ticker: str
    mentions: int            # total de menções (24h)
    rank: int                # posição no ranking (1 = mais mencionado)
    sentiment: float         # bull ratio [0,1] — >0.5 = bullish
    sentiment_label: str     # "bullish" | "bearish" | "neutral"
    # Scores derivados
    attention_score: float = 0.0   # [0,1] normalizado pelo máximo do dia
    momentum_score: float  = 0.0   # variação vs período anterior se disponível


@dataclass
class SqueezeCanditate:
    ticker: str
    short_interest_pct: float    # % do float vendido a descoberto
    days_to_cover: float         # dias para cobrir ao volume médio
    borrow_rate_pct: float       # custo anual de aluguel
    squeeze_score: float         # [0,1] — maior = mais combustível de squeeze


@dataclass
class SwaggyResult:
    wsb_mentions: list[WSBMention] = field(default_factory=list)
    squeeze_candidates: list[SqueezeCanditate] = field(default_factory=list)
    # Lookups rápidos
    mention_map: dict[str, WSBMention] = field(default_factory=dict)
    squeeze_map: dict[str, SqueezeCanditate] = field(default_factory=dict)
    top_mentions: list[str] = field(default_factory=list)   # top 20 tickers por menções
    top_squeeze: list[str] = field(default_factory=list)    # top 10 por squeeze score


# ── Coleta principal ───────────────────────────────────────────────────────────

def collect(max_wsb: int = 50, max_squeeze: int = 30) -> SwaggyResult:
    """Coleta WSB mentions e short squeeze list do SwaggyStocks."""
    result = SwaggyResult()

    wsb = _collect_wsb_mentions(max_wsb)
    result.wsb_mentions = wsb
    result.mention_map  = {m.ticker: m for m in wsb}
    result.top_mentions = [m.ticker for m in wsb[:20]]

    squeeze = _collect_squeeze_list(max_squeeze)
    result.squeeze_candidates = squeeze
    result.squeeze_map = {s.ticker: s for s in squeeze}
    result.top_squeeze = [s.ticker for s in squeeze[:10]]

    _log.info("swaggy_done",
              wsb_tickers=len(wsb),
              squeeze_tickers=len(squeeze),
              top_mention=wsb[0].ticker if wsb else None)
    return result


# ── WSB Mentions ───────────────────────────────────────────────────────────────

def _collect_wsb_mentions(max_results: int) -> list[WSBMention]:
    """Tenta API JSON; fallback para scraping HTML da página."""
    try:
        return _wsb_api(max_results)
    except Exception as exc:
        _log.warning("swaggy_wsb_api_failed", error=str(exc))
    try:
        return _wsb_scrape(max_results)
    except Exception as exc:
        _log.warning("swaggy_wsb_scrape_failed", error=str(exc))
    return []


def _wsb_api(max_results: int) -> list[WSBMention]:
    """Tenta endpoint JSON do SwaggyStocks."""
    resp = requests.get(_WSB_API, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    # Normaliza diferentes formatos possíveis
    rows: list[dict] = []
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("data") or data.get("tickers") or data.get("results") or []

    if not rows:
        raise ValueError("API returned empty data")

    mentions = []
    max_mentions = max(int(r.get("mentions", r.get("count", 1)) or 1) for r in rows[:1]) or 1

    for i, row in enumerate(rows[:max_results]):
        ticker = str(row.get("ticker", row.get("symbol", ""))).upper().strip()
        if not ticker or len(ticker) > 5:
            continue
        count = int(row.get("mentions", row.get("count", row.get("mention_count", 0))) or 0)
        bull  = float(row.get("bullish", row.get("bull", row.get("positive", 0.5))) or 0.5)
        if bull > 1:
            bull = bull / 100  # percentual → ratio

        label = "bullish" if bull > 0.55 else ("bearish" if bull < 0.45 else "neutral")
        attention = min(count / max_mentions, 1.0) if max_mentions > 0 else 0.0

        mentions.append(WSBMention(
            ticker=ticker,
            mentions=count,
            rank=i + 1,
            sentiment=bull,
            sentiment_label=label,
            attention_score=attention,
        ))

    _log.info("swaggy_wsb_api_ok", tickers=len(mentions))
    return mentions


def _wsb_scrape(max_results: int) -> list[WSBMention]:
    """Scraping HTML da página de sentiment."""
    url = f"{_BASE_URL}/dashboard/wallstreetbets/ticker-sentiment"
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    html = resp.text

    # Procura tabela ou JSON embutido no HTML
    # Padrão: JSON no window.__NEXT_DATA__ ou similar
    m = re.search(r'"tickers"\s*:\s*(\[.*?\])', html, re.DOTALL)
    if not m:
        m = re.search(r'"wsb"\s*:\s*\{.*?"data"\s*:\s*(\[.*?\])', html, re.DOTALL)
    if not m:
        raise ValueError("No ticker data found in HTML")

    import json
    rows = json.loads(m.group(1))
    return _wsb_api.__wrapped__(rows, max_results) if hasattr(_wsb_api, '__wrapped__') else []


# ── Short Squeeze List ─────────────────────────────────────────────────────────

def _collect_squeeze_list(max_results: int) -> list[SqueezeCanditate]:
    """Coleta short squeeze candidates. Fallback para yfinance se SwaggyStocks falhar."""
    try:
        return _squeeze_swaggy(max_results)
    except Exception as exc:
        _log.warning("swaggy_squeeze_failed", error=str(exc))
    try:
        return _squeeze_yfinance_fallback(max_results)
    except Exception as exc2:
        _log.warning("swaggy_squeeze_yfinance_failed", error=str(exc2))
    return []


def _squeeze_swaggy(max_results: int) -> list[SqueezeCanditate]:
    """Tenta endpoint de squeeze do SwaggyStocks."""
    url = f"{_BASE_URL}/api/wsb/squeeze"
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    rows: list[dict] = data if isinstance(data, list) else (
        data.get("data") or data.get("results") or []
    )
    if not rows:
        raise ValueError("empty squeeze data")

    candidates = []
    for row in rows[:max_results]:
        ticker = str(row.get("ticker", row.get("symbol", ""))).upper().strip()
        if not ticker:
            continue
        si   = float(row.get("short_interest", row.get("shortInterest", row.get("si_pct", 0))) or 0)
        dtc  = float(row.get("days_to_cover", row.get("daysToCover", row.get("dtc", 0))) or 0)
        borrow = float(row.get("borrow_rate", row.get("borrowRate", row.get("fee", 0))) or 0)

        # Squeeze score composto
        score = _compute_squeeze_score(si, dtc, borrow)
        candidates.append(SqueezeCanditate(
            ticker=ticker,
            short_interest_pct=si,
            days_to_cover=dtc,
            borrow_rate_pct=borrow,
            squeeze_score=score,
        ))

    candidates.sort(key=lambda x: x.squeeze_score, reverse=True)
    _log.info("swaggy_squeeze_ok", tickers=len(candidates))
    return candidates


def _squeeze_yfinance_fallback(max_results: int) -> list[SqueezeCanditate]:
    """Fallback: busca short interest via yfinance para universe padrão."""
    import yfinance as yf

    UNIVERSE = [
        "GME","AMC","BBBY","MSTR","RIVN","LCID","SOFI","PLTR","HOOD","AFRM",
        "UPST","CLOV","WISH","IRNT","SPCE","WKHS","RIDE","NKLA","CLNE","RKT",
        "TLRY","SNDL","ACB","CGC","CRON","HUGE","JMIA","XPEV","NIO","LI",
    ]

    candidates = []
    for ticker in UNIVERSE[:max_results]:
        try:
            info = yf.Ticker(ticker).info
            si   = float(info.get("shortPercentOfFloat") or 0) * 100
            dtc  = float(info.get("shortRatio") or 0)
            if si < 5:
                continue
            score = _compute_squeeze_score(si, dtc, 0)
            candidates.append(SqueezeCanditate(
                ticker=ticker,
                short_interest_pct=si,
                days_to_cover=dtc,
                borrow_rate_pct=0,
                squeeze_score=score,
            ))
        except Exception:
            continue

    candidates.sort(key=lambda x: x.squeeze_score, reverse=True)
    return candidates


def _compute_squeeze_score(si_pct: float, dtc: float, borrow_rate: float) -> float:
    """
    Score de squeeze [0,1].
    - SI > 30% = extremo
    - DTC > 5 = alto risco de squeeze
    - Borrow rate alto = custo crescente para shorts manterem posição
    """
    s = 0.0
    # Short interest: 0-40%+ mapeado para 0-0.5
    s += min(si_pct / 40, 1.0) * 0.45
    # Days to cover: 0-10 mapeado para 0-0.35
    s += min(dtc / 10, 1.0) * 0.35
    # Borrow rate: 0-100%+ mapeado para 0-0.20
    s += min(borrow_rate / 100, 1.0) * 0.20
    return round(min(s, 1.0), 3)


# ── Helpers para integração com desk_intelligence ──────────────────────────────

def get_wsb_attention(result: SwaggyResult | None, ticker: str) -> float:
    """Retorna attention_score [0,1] do ticker no WSB. 0 se não encontrado."""
    if not result:
        return 0.0
    m = result.mention_map.get(ticker)
    return m.attention_score if m else 0.0


def get_wsb_sentiment(result: SwaggyResult | None, ticker: str) -> float | None:
    """Retorna bull ratio [0,1] do ticker. None se não encontrado."""
    if not result:
        return None
    m = result.mention_map.get(ticker)
    return m.sentiment if m else None


def get_squeeze_score(result: SwaggyResult | None, ticker: str) -> float:
    """Retorna squeeze_score [0,1] do ticker. 0 se não encontrado."""
    if not result:
        return 0.0
    s = result.squeeze_map.get(ticker)
    return s.squeeze_score if s else 0.0
