"""
Provider: X (Twitter) — Notificacoes de novos posts

Estrategia:
  1. Navega para /notifications e extrai handles das contas com sino ativo
     ("New post notifications for X and N others")
  2. Para cada handle navega ate o perfil e coleta tweets recentes
  3. Deduplica por URL e limita ao settings.x_timeline_limit

Estrutura do DOM do X (inspecionada em 2026-03):
- Notificacoes em [data-testid="notification"]
- Texto de notificacao de novo post: contem "New post notification"
- Handles: links a[href^="/"] dentro do bloco de notificacao
- Tweets em [data-testid="tweet"]
- Autor em [data-testid="User-Name"]
- Texto em [data-testid="tweetText"]
- Timestamp em time[datetime]
- Engajamento em [data-testid="reply|retweet|like"]
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
_NOTIFICATIONS_URL = "https://x.com/notifications"
# Mantido para collect_html (snapshot da pagina de notificacoes)
_TIMELINE_URL = _NOTIFICATIONS_URL

# Contas prioritárias — sempre coletadas independentemente do feed de notificações
PRIORITY_ACCOUNTS: list[str] = [
    "LizAnnSonders",    # Chief Investment Strategist Schwab — economic data analysis
    "LanceRoberts",     # RealInvestmentAdvice — macro, technicals, portfolio
    "dailychartbook",   # Charts quantitativos diários
    "SubuTrade",        # Quant data, flow analysis
    "BlacklionCTA",     # CTA/trend signals, contrarian
    "SethCL",           # Stock picking, special sits
    "saxena_puru",      # Volatility, options flow
    "AndreasSteno",     # Macro liquidity, rates (Steno Research)
    "JonahLupton",      # Macro, EM
    "profplum99",       # Options, derivatives
    "jam_croissant",    # Macro, rates
    "Mayhem4Markets",   # Market structure, flow
    "WisemanCap",       # Capital markets
    "steve_donze",      # Head of Investments Pictet Japan — global liquidity lag effect
]
_PRIORITY_TWEETS_PER_ACCOUNT = 3  # últimos N tweets de cada conta prioritária


def collect(page: Page, source_doc: SourceDocument) -> list[XTimelineItem]:
    """
    Coleta tweets das contas com notificacao ativa (sino) + contas prioritárias.

    1. Abre /notifications
    2. Clica no item "New post notifications for X and N others"
    3. Scrape os tweets do feed resultante
    4. Visita cada conta prioritária e coleta tweets recentes (garante inclusão)
    """
    limit = settings.x_timeline_limit
    _log.info("collect_start", source=SOURCE_NAME, mode="notifications_click", limit=limit)

    page.goto(_NOTIFICATIONS_URL, timeout=30_000)
    page.wait_for_load_state("domcontentloaded", timeout=15_000)
    time.sleep(4)

    # Localiza e clica no item de notificacao de novo post
    clicked = _click_new_post_notification(page)
    if not clicked:
        _log.warning("notification_not_found", msg="Item 'New post notifications' nao encontrado")
        items: list[XTimelineItem] = []
        seen_urls: set[str] = set()
    else:
        # Aguarda feed carregar
        page.wait_for_load_state("domcontentloaded", timeout=15_000)
        time.sleep(4)
        _log.info("notification_clicked", url=page.url)

        # Scrape tweets do feed aberto
        items = []
        seen_urls: set[str] = set()
        scroll_attempts = 0
        max_scrolls = 15

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

            page.keyboard.press("End")
            time.sleep(2)
            scroll_attempts += 1

            new_count = page.locator('[data-testid="tweet"]').count()
            if new_count == count:
                break

    _log.info("notifications_done", items=len(items))

    # ── Contas prioritárias (sempre coletadas) ──────────────────────────────────
    priority_items = _collect_priority_accounts(page, source_doc, seen_urls)
    _log.info("priority_done", priority_items=len(priority_items))

    all_items = priority_items + items  # prioritárias primeiro
    _log.info("collect_done", source=SOURCE_NAME, total=len(all_items), scrolls=scroll_attempts if clicked else 0)
    return all_items[:limit + len(priority_items)]  # não corta as prioritárias pelo limit


def _collect_priority_accounts(
    page: Page, source_doc: SourceDocument, seen_urls: set[str]
) -> list[XTimelineItem]:
    """Visita cada conta prioritária e coleta os N tweets mais recentes."""
    collected: list[XTimelineItem] = []

    for handle in PRIORITY_ACCOUNTS:
        try:
            profile_url = f"{_X_BASE}/{handle}"
            page.goto(profile_url, timeout=25_000)
            page.wait_for_load_state("domcontentloaded", timeout=10_000)
            time.sleep(3)

            tweets = page.locator('[data-testid="tweet"]')
            found = 0
            for i in range(min(tweets.count(), 10)):
                if found >= _PRIORITY_TWEETS_PER_ACCOUNT:
                    break
                try:
                    item = _parse_tweet(tweets.nth(i), source_doc)
                    if item and item.url not in seen_urls:
                        seen_urls.add(item.url)
                        collected.append(item)
                        found += 1
                except Exception as exc:
                    _log.warning("priority_parse_error", handle=handle, idx=i, error=str(exc))

            _log.info("priority_account_done", handle=handle, found=found)
        except Exception as exc:
            _log.warning("priority_account_error", handle=handle, error=str(exc))

    return collected


# Contas cujo histórico semanal é coletado nas sextas-feiras
WEEKLY_RECAP_ACCOUNTS: list[str] = [
    "LizAnnSonders",   # economic data commentary — semana inteira
    "LanceRoberts",    # macro weekly wrap
    "AndreasSteno",    # liquidity & rates weekly view
]
_WEEKLY_DAYS_BACK = 7    # janela de dias para coleta semanal
_WEEKLY_MAX_TWEETS = 25  # máximo de tweets por conta


def collect_profile_week(
    page: Page,
    source_doc: SourceDocument,
    handle: str,
    days_back: int = _WEEKLY_DAYS_BACK,
    max_tweets: int = _WEEKLY_MAX_TWEETS,
) -> list[XTimelineItem]:
    """
    Coleta tweets de um perfil específico nos últimos N dias.

    Estratégia:
      - Visita x.com/{handle}
      - Scrola a página coletando tweets
      - Para quando encontra tweets mais velhos que `days_back` dias
        ou quando atinge `max_tweets`

    Args:
        page:       Playwright page já autenticada.
        source_doc: SourceDocument de referência.
        handle:     Handle sem @ (ex: "LizAnnSonders").
        days_back:  Janela de dias para trás.
        max_tweets: Máximo de tweets a coletar.

    Returns:
        Lista de XTimelineItem dentro da janela de tempo.
    """
    from datetime import datetime, timezone, timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    profile_url = f"{_X_BASE}/{handle}"
    items: list[XTimelineItem] = []
    seen_urls: set[str] = set()

    try:
        page.goto(profile_url, timeout=30_000)
        page.wait_for_load_state("domcontentloaded", timeout=15_000)
        time.sleep(3)
    except Exception as exc:
        _log.warning("profile_week_goto_error", handle=handle, error=str(exc))
        return []

    scroll_attempts = 0
    max_scrolls = 20
    stop_collecting = False

    while not stop_collecting and len(items) < max_tweets and scroll_attempts < max_scrolls:
        tweets = page.locator('[data-testid="tweet"]')
        count = tweets.count()

        for i in range(count):
            if len(items) >= max_tweets:
                stop_collecting = True
                break
            try:
                item = _parse_tweet(tweets.nth(i), source_doc)
                if not item or item.url in seen_urls:
                    continue
                # Verifica se está dentro da janela de tempo
                if item.created_at and item.created_at < cutoff:
                    stop_collecting = True  # chegou em tweets antigos demais
                    break
                seen_urls.add(item.url)
                items.append(item)
            except Exception:
                pass

        if stop_collecting:
            break

        # Scroll
        prev_count = count
        page.keyboard.press("End")
        time.sleep(2)
        scroll_attempts += 1

        new_count = page.locator('[data-testid="tweet"]').count()
        if new_count == prev_count:
            break  # página não carregou mais conteúdo

    _log.info(
        "profile_week_done",
        handle=handle,
        found=len(items),
        days_back=days_back,
        scrolls=scroll_attempts,
    )
    return items


def collect_html(page: Page) -> str:
    """Retorna o HTML da pagina de notificacoes apos carregamento."""
    page.goto(_NOTIFICATIONS_URL, timeout=30_000)
    page.wait_for_load_state("domcontentloaded", timeout=15_000)
    time.sleep(4)
    return page.content()


# ── Notificacoes ───────────────────────────────────────────────────────────────

def _click_new_post_notification(page: Page) -> bool:
    """
    Localiza e clica no primeiro item 'New post notifications for ...'
    na pagina de notificacoes ja carregada.

    Retorna True se clicou, False se nao encontrou.
    """
    notifs = page.locator('[data-testid="notification"]')
    count = notifs.count()

    for i in range(count):
        try:
            notif = notifs.nth(i)
            text = notif.inner_text()
            if "post notification" in text.lower() or "new post" in text.lower():
                _log.info("notification_found", idx=i, text=text[:80])
                notif.click()
                return True
        except Exception as exc:
            _log.warning("notif_click_error", idx=i, error=str(exc))

    return False


# ── Parsing ────────────────────────────────────────────────────────────────────

def _parse_tweet(tweet: Locator, source_doc: SourceDocument) -> XTimelineItem | None:
    author = _extract_handle(tweet)
    if not author:
        return None

    tweet_url = _extract_tweet_url(tweet, author)
    if not tweet_url:
        return None

    text_el = tweet.locator('[data-testid="tweetText"]').first
    text = text_el.inner_text().strip() if text_el.count() > 0 else ""

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

    engagement = _extract_engagement(tweet)

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
    try:
        author_el = tweet.locator('[data-testid="User-Name"]').first
        if author_el.count() == 0:
            return ""
        raw = author_el.inner_text()
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("@"):
                return line
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
    try:
        handle = author.lstrip("@")
        links = tweet.locator(f'a[href*="/{handle}/status/"]')
        if links.count() > 0:
            href = links.first.get_attribute("href") or ""
            if href.startswith("/"):
                return f"{_X_BASE}{href}"
            return href
        links2 = tweet.locator('a[href*="/status/"]')
        if links2.count() > 0:
            href = links2.first.get_attribute("href") or ""
            if href.startswith("/"):
                return f"{_X_BASE}{href}"
        return ""
    except Exception:
        return ""


def _extract_engagement(tweet: Locator) -> EngagementInfo:
    def _parse_count(text: str) -> int:
        text = text.strip().replace(",", "")
        if not text:
            return 0
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
