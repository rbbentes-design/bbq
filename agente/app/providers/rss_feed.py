"""
Provider: RSS Feeds

Coleta artigos de blogs macro via RSS/Atom.
Configurado por settings.rss_feeds (lista de URLs).
"""

from __future__ import annotations

import ssl
import urllib.request
from datetime import datetime, timezone
from typing import Any

import feedparser

from app.audit.logger import get_logger
from app.models.rss_item import RSSItem

# Windows may fail SSL cert verification for some sites; use unverified context
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE
_HTTPS_HANDLER = urllib.request.HTTPSHandler(context=_ssl_ctx)

_log = get_logger("providers.rss_feed")

SOURCE_NAME = "rss"

# Feeds padrão — edite em .env com RSS_FEEDS="url1,url2"
DEFAULT_FEEDS: list[str] = [
    # ── Macro News ────────────────────────────────────────────────────────────
    # Investing.com — breaking financial news
    "https://www.investing.com/rss/news.rss",
    # SeekingAlpha — market currents (real-time)
    "https://seekingalpha.com/market_currents.xml",
    # CNBC Top News
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    # Bloomberg Markets (public)
    "https://feeds.bloomberg.com/markets/news.rss",

    # ── Substack — Macro / Rates / Credit ─────────────────────────────────────
    # Capital Wars (Luke Gromen territory — liquidity, macro)
    "https://capitalwars.substack.com/feed",
    # Junk Bond Investor — credit markets weekly
    "https://www.junkbondinvestor.com/feed",
    # Clouded Judgement (Jamin Ball) — SaaS/growth multiples
    "https://cloudedjudgement.substack.com/feed",
    # Software Analyst — enterprise tech
    "https://softwareanalyst.substack.com/feed",
    # Doomberg — energy & macro (free posts)
    "https://doomberg.substack.com/feed",
    # The Macro Compass (Alfonso Peccatiello / Alf) — rates, liquidity
    "https://themacrocompass.substack.com/feed",
    # Quoth the Raven — contrarian macro & credit
    "https://quoththeraven.substack.com/feed",
    # Concoda — monetary plumbing, Fed, rates
    "https://concoda.substack.com/feed",
    # Apricitas Economics — data-driven macro
    "https://apricitas.substack.com/feed",

    # ── Analysis Blogs ────────────────────────────────────────────────────────
    # The Diff (Byrne Hobart) — inflections in finance & tech
    "https://www.thediff.co/feed",
    # Adventures in Capitalism (Kuppy) — special sits, macro
    "https://adventuresincapitalism.com/feed/",
    # Real Investment Advice — all posts (daily commentary + bull/bear)
    "https://realinvestmentadvice.com/feed/",
    # Lance Roberts Substack — macro, technicals, portfolio strategy
    "https://lanceroberts.substack.com/feed",
    # SpotGamma Blog — gamma/options analysis (público)
    "https://spotgamma.com/feed/",

    # ── Dados Econômicos / Calendário Macro ──────────────────────────────────
    # FRED Blog — Federal Reserve Bank of St. Louis, análise de dados econômicos
    "https://fredblog.stlouisfed.org/feed/",
    # BEA press releases — GDP, PCE, trade
    "https://www.bea.gov/rss/rss.xml",
    # Calculated Risk — cobertura sistemática de releases econômicas
    "https://feeds.feedburner.com/CalculatedRisk",
    # Nick Timiraos (WSJ Fed reporter) via substack / blog se disponível
    # EconBrowser (Hamilton & Chinn) — análise de dados macro
    "https://econbrowser.com/feed",
]


_MAX_AGE_DAYS = 7  # ignora artigos mais velhos que isso


def collect(feed_urls: list[str] | None = None, max_per_feed: int = 10) -> list[RSSItem]:
    """Coleta itens de todos os feeds configurados, filtrando por janela de 7 dias."""
    from app.config.settings import settings

    urls = feed_urls or _get_configured_feeds(settings)
    if not urls:
        _log.info("rss_no_feeds_configured")
        return []

    cutoff = datetime.now(timezone.utc) - __import__("datetime").timedelta(days=_MAX_AGE_DAYS)
    items: list[RSSItem] = []
    for url in urls:
        try:
            feed_items = _fetch_feed(url, max_per_feed, cutoff=cutoff)
            items.extend(feed_items)
            _log.info("rss_feed_fetched", url=url[:60], items=len(feed_items))
        except Exception as exc:
            _log.warning("rss_feed_error", url=url[:60], error=str(exc))

    _log.info("rss_collect_done", total=len(items), feeds=len(urls))
    return items


_ARTICLE_MAX_CHARS = 4000   # limita corpo para não explodir contexto do LLM
_FETCH_TIMEOUT    = 12       # segundos por artigo


def _fetch_feed(url: str, max_items: int, cutoff: datetime | None = None) -> list[RSSItem]:
    parsed = feedparser.parse(
        url,
        handlers=[_HTTPS_HANDLER],
        request_headers={"User-Agent": "Mozilla/5.0"},
    )
    feed_title = parsed.feed.get("title", url)
    items: list[RSSItem] = []

    for entry in parsed.entries[:max_items]:
        published = _parse_date(entry)
        # Filtra artigos fora da janela — se não tem data, aceita (melhor incluir que perder)
        if cutoff and published and published < cutoff:
            continue

        article_url = entry.get("link", "")
        rss_summary = entry.get("summary", "") or entry.get("description", "")
        import re
        rss_summary = re.sub(r"<[^>]+>", " ", rss_summary).strip()

        # Tenta buscar conteúdo completo; cai no resumo do RSS se falhar
        body = _fetch_article_content(article_url) if article_url else ""
        summary = (body or rss_summary)[:_ARTICLE_MAX_CHARS]

        items.append(RSSItem(
            source_name=feed_title,
            feed_url=url,
            title=entry.get("title", "").strip(),
            summary=summary,
            url=article_url,
            published_at=published,
            author=entry.get("author", ""),
            tags=[t.get("term", "") for t in entry.get("tags", [])],
        ))

    return items


def _fetch_article_content(url: str) -> str:
    """Extrai texto principal do artigo via trafilatura. Retorna '' se falhar."""
    try:
        import trafilatura
        html = trafilatura.fetch_url(url, config=_trafilatura_config())
        if not html:
            return ""
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        return (text or "").strip()
    except Exception:
        return ""


def _trafilatura_config():
    """Config trafilatura com timeout curto para não travar o pipeline."""
    try:
        from trafilatura.settings import use_config
        cfg = use_config()
        cfg.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(_FETCH_TIMEOUT))
        return cfg
    except Exception:
        return None


def _parse_date(entry: Any) -> datetime | None:
    import time
    struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if struct:
        try:
            return datetime.fromtimestamp(time.mktime(struct), tz=timezone.utc)
        except Exception:
            pass
    return None


def _get_configured_feeds(settings: Any) -> list[str]:
    raw = getattr(settings, "rss_feeds", "") or ""
    if raw:
        return [u.strip() for u in raw.split(",") if u.strip()]
    return DEFAULT_FEEDS
