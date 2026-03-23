"""
Provider: X (Twitter) Timeline

Coleta tweets do timeline do usuario via Playwright (requer sessao auth_token).
Faz scroll para carregar mais tweets conforme o limite configurado.

Estrutura do DOM do X (inspecionada em 2026-03):
- Tweets em [data-testid="tweet"]
- Autor em [data-testid="User-Name"] (primeiro = autor principal)
- Texto em [data-testid="tweetText"]
- Timestamp em time[datetime]
- Engajamento em [data-testid="reply|retweet|like"]
- URL em a[href*="/status/"]
"""

from __future__ import annotations

import re
import time

from playwright.sync_api import Locator, Page

from app.audit.logger import get_logger
from app.config.settings import settings
from app.models.source_document import SourceDocument
from app.models.x_timeline_item import EngagementInfo, XTimelineItem
from app.utils.timestamps import new_ulid

_log = get_logger("providers.x_timeline")

SOURCE_NAME = "x"
_X_BASE = "https://x.com"
_TIMELINE_URL = "https://x.com/i/timeline"


def collect(page: Page, source_doc: SourceDocument) -> list[XTimelineItem]:
    """
    Coleta tweets do timeline do usuario.

    Args:
        page: Page Playwright com sessao X ativa (auth_token).
        source_doc: SourceDocument ja criado pelo pipeline.

    Returns:
        Lista de XTimelineItem (ate settings.x_timeline_limit itens).
    """
    limit = settings.x_timeline_limit
    _log.info("collect_start", source=SOURCE_NAME, limit=limit)

    page.goto(_TIMELINE_URL, timeout=30_000)
    page.wait_for_load_state("domcontentloaded", timeout=15_000)
    time.sleep(5)  # Aguarda carregamento inicial do React

    items: list[XTimelineItem] = []
    seen_urls: set[str] = set()
    scroll_attempts = 0
    max_scrolls = 20

    while len(items) < limit and scroll_attempts < max_scrolls:
        tweets = page.locator('[data-testid="tweet"]')
        count = tweets.count()

        for i in range(count):
            if len(items) >= limit:
                break
            try:
                item = _parse_tweet(tweets.nth(i), source_doc)
                if item and item.url not in seen_urls:
                    seen_urls.add(item.url)
                    items.append(item)
            except Exception as exc:
                _log.warning("parse_tweet_error", idx=i, error=str(exc))

        prev_count = count
        page.keyboard.press("End")
        time.sleep(2)
        scroll_attempts += 1

        new_count = page.locator('[data-testid="tweet"]').count()
        if new_count == prev_count:
            # Nenhum novo tweet carregou — fim do feed
            break

    _log.info("collect_done", source=SOURCE_NAME, items=len(items), scrolls=scroll_attempts)
    return items[:limit]


def collect_html(page: Page) -> str:
    """Retorna o HTML da timeline apos carregamento inicial."""
    page.goto(_TIMELINE_URL, timeout=30_000)
    page.wait_for_load_state("domcontentloaded", timeout=15_000)
    time.sleep(5)
    return page.content()


# ── Parsing ────────────────────────────────────────────────────────────────────

def _parse_tweet(tweet: Locator, source_doc: SourceDocument) -> XTimelineItem | None:
    # Autor: primeiro User-Name (evita quoted tweet author)
    author = _extract_handle(tweet)
    if not author:
        return None

    # URL do tweet
    tweet_url = _extract_tweet_url(tweet, author)
    if not tweet_url:
        return None

    # Texto
    text_el = tweet.locator('[data-testid="tweetText"]').first
    text = text_el.inner_text().strip() if text_el.count() > 0 else ""

    # Timestamp
    created_at = None
    time_el = tweet.locator("time").first
    if time_el.count() > 0:
        dt_str = time_el.get_attribute("datetime")
        if dt_str:
            try:
                from datetime import datetime, timezone
                created_at = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

    # Engajamento
    engagement = _extract_engagement(tweet)

    # Midia: imagens no tweet
    media_refs: list[str] = []
    imgs = tweet.locator('img[src*="pbs.twimg.com/media"]')
    for j in range(min(imgs.count(), 4)):
        src = imgs.nth(j).get_attribute("src")
        if src:
            media_refs.append(src)

    return XTimelineItem(
        id=new_ulid(),
        author=author,
        text=text,
        url=tweet_url,
        created_at=created_at,
        engagement_info=engagement,
        media_refs=media_refs,
        raw_source_document_id=source_doc.id,
    )


def _extract_handle(tweet: Locator) -> str:
    """Extrai @handle do primeiro User-Name do tweet."""
    try:
        author_el = tweet.locator('[data-testid="User-Name"]').first
        if author_el.count() == 0:
            return ""
        raw = author_el.inner_text()
        # Formato: "Display Name\n@handle\n..."
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("@"):
                return line
        # Fallback: link com href="/handle"
        links = tweet.locator('[data-testid="User-Name"] a[href^="/"]')
        if links.count() > 0:
            href = links.first.get_attribute("href") or ""
            match = re.match(r"^/([^/]+)$", href)
            if match:
                return f"@{match.group(1)}"
        return ""
    except Exception:
        return ""


def _extract_tweet_url(tweet: Locator, author: str) -> str:
    """Extrai a URL canonica do tweet (https://x.com/handle/status/id)."""
    try:
        handle = author.lstrip("@")
        links = tweet.locator(f'a[href*="/{handle}/status/"]')
        if links.count() > 0:
            href = links.first.get_attribute("href") or ""
            if href.startswith("/"):
                return f"{_X_BASE}{href}"
            return href
        # Fallback: qualquer link /status/
        links2 = tweet.locator('a[href*="/status/"]')
        if links2.count() > 0:
            href = links2.first.get_attribute("href") or ""
            if href.startswith("/"):
                return f"{_X_BASE}{href}"
        return ""
    except Exception:
        return ""


def _extract_engagement(tweet: Locator) -> EngagementInfo:
    """Extrai metricas de engajamento do tweet."""
    def _parse_count(text: str) -> int:
        text = text.strip().replace(",", "")
        if not text:
            return 0
        # "1.2K" -> 1200, "3.5M" -> 3500000
        m = re.match(r"^([\d.]+)([KkMm]?)$", text)
        if m:
            val = float(m.group(1))
            suffix = m.group(2).upper()
            if suffix == "K":
                val *= 1000
            elif suffix == "M":
                val *= 1_000_000
            return int(val)
        return 0

    def _get_metric(testid: str) -> int:
        el = tweet.locator(f'[data-testid="{testid}"]')
        if el.count() == 0:
            return 0
        try:
            return _parse_count(el.first.inner_text())
        except Exception:
            return 0

    return EngagementInfo(
        replies=_get_metric("reply"),
        reposts=_get_metric("retweet"),
        likes=_get_metric("like"),
    )
