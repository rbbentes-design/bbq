"""
Provider: ZeroHedge — Market Ear + Main Page (Hi-Lo Red)

Coleta os blocos editoriais da pagina /the-market-ear via Playwright + BeautifulSoup.
O conteudo e acessivel com sessao autenticada (cookie coral_talk_sess).

Estrutura da pagina (inspecionada em 2026-03):
- Artigos em div[class*="nonStickyContainer"]
- Titulo em h2 direto dentro do container
- Texto em div[class*="body"] > p
- Imagens em img[src]
"""

from __future__ import annotations

import time
from datetime import datetime

from bs4 import BeautifulSoup, Tag
from playwright.sync_api import Page

from app.audit.logger import get_logger
from app.config.settings import settings
from app.models.market_ear_block import MarketEarBlock
from app.models.source_document import SourceDocument
from app.utils.text import normalize_whitespace
from app.utils.timestamps import new_ulid

_log = get_logger("providers.zerohedge")

SOURCE_NAME = "zerohedge"
_MAIN_URL = "https://www.zerohedge.com/"


def collect(page: Page, source_doc: SourceDocument) -> list[MarketEarBlock]:
    """
    Coleta blocos do Market Ear com scroll para atingir zerohedge_blocks_limit.
    """
    limit = settings.zerohedge_blocks_limit
    _log.info("collect_start", source=SOURCE_NAME,
              url=settings.zerohedge_market_ear_url, limit=limit)

    page.goto(settings.zerohedge_market_ear_url, timeout=30_000)
    page.wait_for_load_state("networkidle", timeout=20_000)
    time.sleep(3)

    # Scroll até ter blocos suficientes (max 10 tentativas)
    for _ in range(10):
        html = page.content()
        count = html.count("nonStickyContainer")
        if count >= limit:
            break
        page.keyboard.press("End")
        time.sleep(2)

    html = page.content()
    blocks = _parse_blocks(html, source_doc)[:limit]

    _log.info("collect_done", source=SOURCE_NAME, blocks=len(blocks))
    return blocks


def collect_html(page: Page) -> str:
    """Retorna HTML bruto da pagina apos scroll para zerohedge_blocks_limit."""
    limit = settings.zerohedge_blocks_limit
    page.goto(settings.zerohedge_market_ear_url, timeout=30_000)
    page.wait_for_load_state("networkidle", timeout=20_000)
    time.sleep(3)

    for _ in range(10):
        count = page.content().count("nonStickyContainer")
        if count >= limit:
            break
        page.keyboard.press("End")
        time.sleep(2)

    return page.content()


# ── Main page (sábado — artigos mais comentados/polêmicos) ────────────────────

def collect_main_page(page: Page, max_articles: int = 5) -> list["RSSItem"]:
    """
    Coleta os artigos mais recentes e comentados da página principal do ZeroHedge.
    Usado no sábado para o podcast — fonte de temas polêmicos e de alto engajamento.
    Retorna artigos com título, resumo, URL e contagem de comentários (proxy de engajamento).
    """
    from app.models.rss_item import RSSItem

    _log.info("collect_main_page_start", source=SOURCE_NAME, max=max_articles)
    page.goto(_MAIN_URL, timeout=30_000)
    page.wait_for_load_state("networkidle", timeout=20_000)
    time.sleep(3)

    soup = BeautifulSoup(page.content(), "lxml")

    # Artigos na página principal: cada item tem título, link, excerpt e comment count
    articles: list[dict] = []
    seen_urls: set[str] = set()

    for a_tag in soup.find_all("a", href=True):
        href = str(a_tag["href"])
        # Links de artigos: /[categoria]/[data]/[slug]
        if not (href.startswith("/") and href.count("/") >= 3):
            continue
        if any(skip in href for skip in ["/user/", "/tags/", "/category/", "/search", "#"]):
            continue
        url = f"https://www.zerohedge.com{href}" if href.startswith("/") else href
        if url in seen_urls:
            continue
        seen_urls.add(url)

        # Sobe na árvore para encontrar o container do artigo
        container = a_tag.find_parent(["article", "div"])
        if not container:
            continue

        title = a_tag.get_text(strip=True)
        if len(title) < 15 or title.lower() in ("read more", "more", "continue"):
            continue

        # Excerpt: texto do container excluindo o título
        excerpt_parts = [
            t.strip() for t in container.stripped_strings
            if t.strip() and t.strip() != title and len(t.strip()) > 30
        ]
        excerpt = " ".join(excerpt_parts[:4])[:600]

        # Contagem de comentários como proxy de engajamento
        comment_count = 0
        comment_el = container.find(string=lambda s: s and s.strip().isdigit() and
                                    int(s.strip()) > 0)
        if comment_el:
            try:
                comment_count = int(comment_el.strip())
            except ValueError:
                pass

        articles.append({
            "title": title,
            "url": url,
            "excerpt": excerpt,
            "comments": comment_count,
        })

        if len(articles) >= max_articles * 4:  # coleta mais para ordenar
            break

    # Ordena por engajamento (comentários) e pega os top N
    articles.sort(key=lambda x: x["comments"], reverse=True)
    top = articles[:max_articles]

    # Para os top artigos, busca o texto completo
    results: list[RSSItem] = []
    for art in top:
        try:
            full_text = _fetch_article_text(page, art["url"])
            summary = full_text[:3000] if full_text else art["excerpt"]
            results.append(RSSItem(
                source_name="ZeroHedge — Main",
                feed_url=_MAIN_URL,
                title=art["title"],
                summary=summary,
                url=art["url"],
                tags=[f"comments:{art['comments']}"],
            ))
            _log.info("main_article_fetched", title=art["title"][:60],
                      comments=art["comments"], chars=len(summary))
        except Exception as exc:
            _log.warning("main_article_error", url=art["url"][:60], error=str(exc))

    _log.info("collect_main_page_done", articles=len(results))
    return results


def _fetch_article_text(page: Page, url: str) -> str:
    """Busca o texto completo de um artigo do ZeroHedge."""
    page.goto(url, timeout=20_000)
    page.wait_for_load_state("domcontentloaded", timeout=15_000)
    time.sleep(2)
    soup = BeautifulSoup(page.content(), "lxml")

    # Corpo do artigo: div com class article-content ou similar
    body = (
        soup.select_one('[class*="article-content"]') or
        soup.select_one('[class*="article__content"]') or
        soup.select_one("article") or
        soup.select_one("main")
    )
    if not body:
        return ""

    paras = [normalize_whitespace(p.get_text()) for p in body.find_all("p") if p.get_text(strip=True)]
    return "\n\n".join(paras)


# ── Parsing ────────────────────────────────────────────────────────────────────

def _parse_blocks(html: str, source_doc: SourceDocument) -> list[MarketEarBlock]:
    soup = BeautifulSoup(html, "lxml")
    containers = soup.select('[class*="nonStickyContainer"]')

    blocks: list[MarketEarBlock] = []
    for idx, container in enumerate(containers):
        block = _parse_one(container, idx, source_doc)
        if block is not None:
            blocks.append(block)

    return blocks


def _parse_one(container: Tag, idx: int, source_doc: SourceDocument) -> MarketEarBlock | None:
    try:
        # Titulo: primeiro h2 direto dentro do container
        h2 = container.find("h2")
        title = normalize_whitespace(h2.get_text()) if h2 else ""

        # Corpo: paragrafos dentro do div body
        body_div = container.select_one('[class*="body"]')
        if body_div:
            paras = body_div.find_all("p")
            body_text = "\n\n".join(
                normalize_whitespace(p.get_text())
                for p in paras
                if p.get_text(strip=True)
            )
        else:
            body_text = ""

        # Imagens: src absolutas
        imgs = container.find_all("img", src=True)
        image_refs = [
            str(img["src"])
            for img in imgs
            if str(img.get("src", "")).startswith("http")
        ]

        # Timestamp: elemento time com atributo datetime
        time_el = container.find("time")
        published_at: datetime | None = None
        if time_el and time_el.get("datetime"):
            try:
                published_at = datetime.fromisoformat(
                    str(time_el["datetime"]).replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        # Descarta containers sem titulo e sem imagem (estruturais/vazios)
        if not title and not image_refs and not body_text:
            return None

        return MarketEarBlock(
            id=new_ulid(),
            title=title,
            body_text=body_text,
            image_refs=image_refs,
            source_url=settings.zerohedge_market_ear_url,
            published_at=published_at,
            raw_source_document_id=source_doc.id,
            position_index=idx,
        )
    except Exception as exc:
        _log.warning("parse_block_error", idx=idx, error=str(exc))
        return None
