"""
Provider: ZeroHedge The Market Ear

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


def collect(page: Page, source_doc: SourceDocument) -> list[MarketEarBlock]:
    """
    Coleta todos os blocos do Market Ear a partir de uma Page ja autenticada.

    Args:
        page: Page Playwright com sessao ativa.
        source_doc: SourceDocument ja criado pelo pipeline (fornece o run_id/id).

    Returns:
        Lista de MarketEarBlock ordenada por posicao (0 = topo).
    """
    _log.info("collect_start", source=SOURCE_NAME, url=settings.zerohedge_market_ear_url)

    page.goto(settings.zerohedge_market_ear_url, timeout=30_000)
    page.wait_for_load_state("networkidle", timeout=20_000)
    time.sleep(3)  # JS rendering

    html = page.content()
    blocks = _parse_blocks(html, source_doc)

    _log.info("collect_done", source=SOURCE_NAME, blocks=len(blocks))
    return blocks


def collect_html(page: Page) -> str:
    """Retorna apenas o HTML bruto da pagina (para RawStore)."""
    page.goto(settings.zerohedge_market_ear_url, timeout=30_000)
    page.wait_for_load_state("networkidle", timeout=20_000)
    time.sleep(3)
    return page.content()


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
