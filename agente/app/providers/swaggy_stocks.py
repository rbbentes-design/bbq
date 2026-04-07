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
_WSB_API  = "https://swaggystocks.com/api/wsb/tickers"   # JSON endpoint (gated)
_APEWISDOM_WSB_URL = "https://apewisdom.io/api/v1.0/filter/wallstreetbets"
_APEWISDOM_ALL_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks"
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
    market_bull_pct: float | None = None                    # sentimento geral do mercado [0,1]


_MARKET_SENTIMENT_URL = f"{_BASE_URL}/dashboard/stocks/market-sentiment"


def collect_with_page(page, max_results: int = 30) -> SwaggyResult:
    """
    Coleta sentimento de mercado via Playwright (página renderizada client-side).
    URL: https://swaggystocks.com/dashboard/stocks/market-sentiment

    Extrai:
      - Top stocks por sentimento (ticker, bull%, mentions)
      - Sentimento geral do mercado (bull_pct agregado)
    """
    import json as _json

    result = SwaggyResult()

    try:
        page.goto(_MARKET_SENTIMENT_URL, timeout=30_000, wait_until="domcontentloaded")
        # Aguarda conteúdo dinâmico carregar
        page.wait_for_timeout(4000)

        # Tenta capturar via __NEXT_DATA__ (Next.js SSR/ISR)
        next_data = page.evaluate("""
            () => {
                const el = document.getElementById('__NEXT_DATA__');
                return el ? el.textContent : null;
            }
        """)

        rows = []
        bull_pct_global = None

        if next_data:
            try:
                data = _json.loads(next_data)
                # Navega pela estrutura do Next.js para encontrar os dados
                props = data.get("props", {}).get("pageProps", {})
                # Tenta diferentes chaves possíveis
                for key in ("stocks", "tickers", "data", "sentiment", "stockSentiment"):
                    if key in props:
                        candidate = props[key]
                        if isinstance(candidate, list) and candidate:
                            rows = candidate
                            break
                        elif isinstance(candidate, dict):
                            for sub in ("stocks", "tickers", "data", "items"):
                                if isinstance(candidate.get(sub), list):
                                    rows = candidate[sub]
                                    break
                # Sentimento geral
                for key in ("marketSentiment", "overall", "bullPct", "bullish_pct"):
                    val = props.get(key)
                    if isinstance(val, (int, float)):
                        bull_pct_global = float(val)
                        break
            except Exception as exc:
                _log.warning("swaggy_next_data_parse", error=str(exc))

        # Fallback: extrai dados via DOM (tabela ou cards)
        if not rows:
            rows_raw = page.evaluate("""
                () => {
                    // Tenta extrair de qualquer tabela com dados de ticker
                    const results = [];
                    const rows = document.querySelectorAll('table tbody tr');
                    for (const row of rows) {
                        const cells = Array.from(row.querySelectorAll('td'));
                        if (cells.length >= 2) {
                            results.push(cells.map(c => c.innerText.trim()));
                        }
                    }
                    return results;
                }
            """)
            # Heurística: primeira coluna = ticker, segunda+ = valores
            for i, cells in enumerate(rows_raw[:max_results]):
                if not cells:
                    continue
                ticker = cells[0].upper().strip()
                if not ticker or len(ticker) > 6 or not ticker.isalpha():
                    continue
                # Tenta extrair bull% de alguma célula com %
                bull = 0.5
                for cell in cells[1:]:
                    if "%" in cell:
                        try:
                            bull = float(cell.replace("%", "").strip()) / 100
                            break
                        except ValueError:
                            pass
                rows.append({"ticker": ticker, "bullish": bull, "rank": i + 1})

        # Processa rows → WSBMention
        mentions = []
        max_count = 1
        for row in rows[:max_results]:
            ticker = str(row.get("ticker", row.get("symbol", row.get("stock", "")))).upper().strip()
            if not ticker or len(ticker) > 6:
                continue
            bull = float(row.get("bullish", row.get("bull", row.get("bullPct", row.get("bull_pct", 0.5)))) or 0.5)
            if bull > 1:
                bull = bull / 100
            count = int(row.get("mentions", row.get("count", row.get("totalMentions", 0))) or 0)
            max_count = max(max_count, count)
            label = "bullish" if bull > 0.55 else ("bearish" if bull < 0.45 else "neutral")
            mentions.append(WSBMention(
                ticker=ticker,
                mentions=count,
                rank=len(mentions) + 1,
                sentiment=bull,
                sentiment_label=label,
                attention_score=min(count / max_count, 1.0) if max_count > 0 else 0.0,
            ))

        # Normaliza attention scores
        if mentions and max_count > 1:
            for m in mentions:
                m.attention_score = m.mentions / max_count

        result.wsb_mentions = mentions
        result.mention_map  = {m.ticker: m for m in mentions}
        result.top_mentions = [m.ticker for m in mentions[:20]]
        result.market_bull_pct = bull_pct_global  # pode ser None

        _log.info("swaggy_playwright_ok",
                  tickers=len(mentions),
                  bull_pct_global=bull_pct_global,
                  top=result.top_mentions[:5])

    except Exception as exc:
        _log.warning("swaggy_playwright_failed", error=str(exc))

    return result


# ── Coleta principal ───────────────────────────────────────────────────────────

def collect(max_wsb: int = 50, max_squeeze: int = 30) -> SwaggyResult:
    """
    Coleta WSB mentions via ApeWisdom (free API) + short squeeze via yfinance fallback.
    ApeWisdom agrega menções de r/wallstreetbets + outros subs de Reddit.
    """
    result = SwaggyResult()

    wsb = _collect_wsb_apewisdom(max_wsb)
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


# ── WSB Mentions via ApeWisdom ────────────────────────────────────────────────

def _collect_wsb_apewisdom(max_results: int) -> list[WSBMention]:
    """
    Coleta ticker mentions de r/wallstreetbets via ApeWisdom (gratuito, sem auth).
    Endpoint: https://apewisdom.io/api/v1.0/filter/wallstreetbets
    Campos: rank, ticker, name, mentions, upvotes, rank_24h_ago, mentions_24h_ago
    """
    try:
        resp = requests.get(
            _APEWISDOM_WSB_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("results", [])
        if not rows:
            raise ValueError("ApeWisdom returned empty results")

        max_mentions = max(int(r.get("mentions", 1) or 1) for r in rows[:1]) or 1
        mentions = []
        for row in rows[:max_results]:
            ticker = str(row.get("ticker", "")).upper().strip()
            if not ticker or len(ticker) > 6:
                continue
            # Strip .X suffix (crypto tickers)
            if ticker.endswith(".X"):
                continue
            count = int(row.get("mentions", 0) or 0)
            upvotes = int(row.get("upvotes", 0) or 0)
            count_24h = int(row.get("mentions_24h_ago", 0) or 0)
            rank = int(row.get("rank", len(mentions) + 1))
            # ApeWisdom doesn't provide bull/bear — estimate from upvotes ratio
            # Use rank momentum as proxy: improving rank → bullish signal
            rank_prev = int(row.get("rank_24h_ago", rank) or rank)
            rank_delta = rank_prev - rank  # positive = moved up (more bullish attention)
            # Sentiment: no direct bull/bear, use 0.5 + momentum adjustment
            sentiment = min(max(0.5 + rank_delta * 0.02, 0.0), 1.0)
            label = "bullish" if sentiment > 0.55 else ("bearish" if sentiment < 0.45 else "neutral")
            attention = min(count / max_mentions, 1.0) if max_mentions > 0 else 0.0
            mentions.append(WSBMention(
                ticker=ticker,
                mentions=count,
                rank=rank,
                sentiment=sentiment,
                sentiment_label=label,
                attention_score=attention,
                momentum_score=float(rank_delta) / 10,
            ))

        _log.info("apewisdom_wsb_ok", tickers=len(mentions), top=mentions[0].ticker if mentions else None)
        return mentions

    except Exception as exc:
        _log.warning("apewisdom_wsb_failed", error=str(exc))
        # Fallback to old scraping approach
        return _collect_wsb_mentions(max_results)


# ── WSB Mentions (legacy fallback) ────────────────────────────────────────────

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


# ── Serialização para bundle ──────────────────────────────────────────────────

def swaggy_result_to_dict(result: SwaggyResult) -> dict:
    """Serializa SwaggyResult para dict JSON-safe (para bundle)."""
    return {
        "wsb_mentions": [
            {
                "ticker": m.ticker, "mentions": m.mentions, "rank": m.rank,
                "sentiment": m.sentiment, "sentiment_label": m.sentiment_label,
                "attention_score": m.attention_score,
            }
            for m in result.wsb_mentions
        ],
        "squeeze_candidates": [
            {
                "ticker": s.ticker, "short_interest_pct": s.short_interest_pct,
                "days_to_cover": s.days_to_cover, "borrow_rate_pct": s.borrow_rate_pct,
                "squeeze_score": s.squeeze_score,
            }
            for s in result.squeeze_candidates
        ],
        "top_mentions": result.top_mentions,
        "top_squeeze": result.top_squeeze,
        "market_bull_pct": result.market_bull_pct,
    }


def dict_to_swaggy_result(data: dict) -> SwaggyResult:
    """Reconstrói SwaggyResult a partir de dict serializado."""
    if not data:
        return SwaggyResult()
    mentions = [
        WSBMention(
            ticker=m["ticker"], mentions=m.get("mentions", 0), rank=m.get("rank", i + 1),
            sentiment=m.get("sentiment", 0.5), sentiment_label=m.get("sentiment_label", "neutral"),
            attention_score=m.get("attention_score", 0.0),
        )
        for i, m in enumerate(data.get("wsb_mentions", []))
    ]
    squeeze = [
        SqueezeCanditate(
            ticker=s["ticker"], short_interest_pct=s.get("short_interest_pct", 0),
            days_to_cover=s.get("days_to_cover", 0), borrow_rate_pct=s.get("borrow_rate_pct", 0),
            squeeze_score=s.get("squeeze_score", 0),
        )
        for s in data.get("squeeze_candidates", [])
    ]
    result = SwaggyResult(
        wsb_mentions=mentions,
        squeeze_candidates=squeeze,
        mention_map={m.ticker: m for m in mentions},
        squeeze_map={s.ticker: s for s in squeeze},
        top_mentions=data.get("top_mentions", [m.ticker for m in mentions[:20]]),
        top_squeeze=data.get("top_squeeze", [s.ticker for s in squeeze[:10]]),
        market_bull_pct=data.get("market_bull_pct"),
    )
    return result


def collect_playwright(max_wsb: int = 50) -> SwaggyResult:
    """Abre uma sessão Playwright standalone e coleta SwaggyStocks."""
    try:
        from playwright.sync_api import sync_playwright
        from app.auth.browser_profile import open_context
        with sync_playwright() as pw:
            ctx = open_context(pw, headless=True)
            page = ctx.new_page()
            result = collect_with_page(page, max_results=max_wsb)
            ctx.close()
        return result
    except Exception as exc:
        _log.warning("swaggy_playwright_standalone_failed", error=str(exc))
        return SwaggyResult()


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
