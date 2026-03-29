"""
Provider: Spectra Markets Library (am/FX)

Coleta os N artigos mais recentes da biblioteca via Playwright.
Os links dos artigos estão em <a class="w-grid-item-anchor"> na página /library/.

Autenticação:
  Perfil persistente ~/agente-workspace/state/browser.
  Primeiro uso: python -m app.cli.auth login spectra
"""

from __future__ import annotations

import re
import time
from datetime import date, datetime

from bs4 import BeautifulSoup
from playwright.sync_api import Page

from app.audit.logger import get_logger
from app.models.rss_item import RSSItem
from app.utils.timestamps import new_ulid, utcnow

_log = get_logger("providers.spectra")

SOURCE_NAME = "spectra"
_BASE = "https://www.spectramarkets.com"
_LIBRARY_URL = f"{_BASE}/library/"
_SPEEDRUN_URL = f"{_BASE}/friday-speedrun/"  # alvo principal


def collect(page: Page, max_articles: int = 2) -> list[RSSItem]:
    """
    Coleta os N Friday Speedruns mais recentes da Spectra Markets.
    Fallback para a library geral se a página do Speedrun não tiver artigos.
    """
    _log.info("collect_start", source=SOURCE_NAME, max=max_articles)

    # Tenta primeiro a página do Friday Speedrun
    items = _collect_from_url(page, _SPEEDRUN_URL, max_articles, section="Friday Speedrun")
    if items:
        return items

    # Fallback: library geral
    _log.info("speedrun_empty_fallback_to_library")
    return _collect_from_url(page, _LIBRARY_URL, max_articles, section="Library")


def _collect_from_url(page: Page, url: str, max_articles: int, section: str) -> list[RSSItem]:
    page.goto(url, timeout=30_000)
    page.wait_for_load_state("networkidle", timeout=20_000)
    time.sleep(3)
    _accept_cookies(page)

    soup = BeautifulSoup(page.content(), "lxml")
    article_links = soup.find_all("a", class_="w-grid-item-anchor")

    if not article_links:
        _log.warning("no_articles_found", source=SOURCE_NAME, section=section)
        return []

    _log.info("articles_found", count=len(article_links), section=section)

    items: list[RSSItem] = []
    for a in article_links[:max_articles]:
        href = a.get("href", "")
        title = a.get("aria-label", "").strip()
        if not href:
            continue
        try:
            item = _fetch_article(page, href, title)
            if item:
                items.append(item)
        except Exception as exc:
            _log.warning("article_error", url=href[:60], error=str(exc))

    _log.info("collect_done", source=SOURCE_NAME, section=section, collected=len(items))
    return items


def _accept_cookies(page: Page) -> None:
    try:
        btn = page.locator('button:has-text("Accept All")')
        if btn.count() > 0:
            btn.click(timeout=3_000)
            time.sleep(1)
    except Exception:
        pass


def _fetch_article(page: Page, url: str, title: str) -> RSSItem | None:
    page.goto(url, timeout=20_000)
    page.wait_for_load_state("networkidle", timeout=15_000)
    time.sleep(2)

    # Extrai texto visível (cookies já aceitos na library — persistem no domínio)
    body_text = page.inner_text("body")

    # Detecta paywall: conteúdo substituído por upsell de livro/assinatura
    _PAYWALL_MARKERS = [
        "A comprehensive guide to foreign exchange",
        "Choose your plan",
        "Start your free trial",
    ]
    if any(m in body_text for m in _PAYWALL_MARKERS):
        _log.info("article_paywalled", url=url[:60])
        # Retorna apenas o teaser (título + subtítulo da listagem)
        summary = f"[Subscriber content] {title}"
        published_at = None
    else:
        # Conteúdo livre — extrai após o subtítulo/data
        lines = [l.strip() for l in body_text.splitlines() if l.strip()]
        # Encontra onde começa o artigo (após a data ou título)
        start = 0
        for i, line in enumerate(lines):
            if re.match(r"\w+ \d+, 20\d\d", line) or (title[:10].lower() in line.lower() and i < 20):
                start = i
                break
        summary = "\n".join(lines[start:start + 80])[:2000]
        soup = BeautifulSoup(page.content(), "lxml")
        published_at = _extract_date(soup)

    soup = BeautifulSoup(page.content(), "lxml")
    category = _extract_category(soup)
    source_name = f"Spectra Markets — {category}" if category else "Spectra Markets"

    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else url.split("/")[-2].replace("-", " ").title()

    _log.info("article_fetched", title=title[:60], chars=len(summary),
              paywalled="[Subscriber" in summary)

    return RSSItem(
        id=new_ulid(),
        source_name=source_name,
        feed_url=_LIBRARY_URL,
        title=title,
        summary=summary,
        url=url,
        published_at=published_at if not isinstance(published_at, type(None)) else None,
    )


def _extract_date(soup: BeautifulSoup) -> datetime | None:
    # Tenta <time datetime="...">
    time_el = soup.find("time", datetime=True)
    if time_el:
        try:
            return datetime.fromisoformat(str(time_el["datetime"]).replace("Z", "+00:00"))
        except (ValueError, KeyError):
            pass
    # Tenta texto de data (ex: "March 20, 2026")
    for el in soup.find_all(string=re.compile(r"\w+ \d+, 20\d\d")):
        m = re.search(r"(\w+ \d+, 20\d\d)", el)
        if m:
            try:
                return datetime.strptime(m.group(1), "%B %d, %Y")
            except ValueError:
                pass
    return None


def _extract_category(soup: BeautifulSoup) -> str:
    # Tenta breadcrumb ou badge de categoria
    for el in soup.find_all(class_=re.compile(r"category|tag|badge|label", re.I)):
        t = el.get_text(strip=True)
        if t and len(t) < 30:
            return t
    return ""
