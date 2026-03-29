"""
Provider: DeepVue — Theme Tracker

Coleta o ranking de ETFs temáticos por período (Today, 1W, 1M, 3M, YTD)
navegando pelo dashboard e clicando em cada botão de período.

Autenticação:
  Perfil persistente ~/agente-workspace/state/browser.
  Primeiro uso: python -m app.cli.auth login deepvue
"""

from __future__ import annotations

import re
import time
from datetime import date

from playwright.sync_api import Page

from app.audit.logger import get_logger
from app.models.rss_item import RSSItem
from app.utils.timestamps import new_ulid, utcnow

_log = get_logger("providers.deepvue")

SOURCE_NAME = "deepvue"
_DASHBOARD_URL = "https://app.deepvue.com/dashboard"

_PERIODS = ["Today", "1W", "1M", "3M", "YTD"]
_N_THEMES = 32


def collect(page: Page) -> list[RSSItem]:
    """
    Coleta Theme Tracker do DeepVue para todos os períodos.
    Retorna um único RSSItem com o resumo completo formatado.
    """
    _log.info("collect_start", source=SOURCE_NAME)

    page.goto(_DASHBOARD_URL, timeout=30_000)
    page.wait_for_load_state("domcontentloaded", timeout=15_000)
    time.sleep(6)

    if "/login" in page.url:
        _log.warning("not_authenticated", url=page.url)
        return []

    period_data: dict[str, dict[str, str]] = {}

    for period in _PERIODS:
        try:
            btn = page.locator(f'button:has-text("{period}")')
            if btn.count() == 0:
                # fallback: qualquer elemento com texto exato
                btn = page.locator(f'text="{period}"').first
            btn.click(timeout=4_000)
            time.sleep(2)
        except Exception as exc:
            _log.warning("period_click_failed", period=period, error=str(exc))
            continue

        themes = _parse_themes(page)
        period_data[period] = themes
        _log.info("period_parsed", period=period, themes=len(themes))

    if not period_data:
        return []

    summary = _format_summary(period_data)
    today_str = str(date.today())

    item = RSSItem(
        id=new_ulid(),
        source_name="DeepVue — Theme Tracker",
        feed_url=_DASHBOARD_URL,
        title=f"DeepVue Theme Tracker — {today_str}",
        summary=summary,
        url=_DASHBOARD_URL,
        published_at=utcnow(),
    )

    _log.info("collect_done", source=SOURCE_NAME, periods=len(period_data))
    return [item]


# ── Parsing ────────────────────────────────────────────────────────────────────

def _parse_themes(page: Page) -> dict[str, str]:
    """
    Extrai {tema: pct_change} do body text atual.
    Estratégia: após os botões de período, pares (nome, +X.XX%) até '32 results'.
    """
    text = page.inner_text("body")
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    themes: dict[str, str] = {}
    in_themes = False
    i = 0
    while i < len(lines):
        line = lines[i]

        # Começa a capturar após os botões de período
        if not in_themes and line in _PERIODS:
            in_themes = True
            i += 1
            continue

        if in_themes:
            if line == f"{_N_THEMES} results":
                break
            # Par: nome seguido de porcentagem
            if i + 1 < len(lines):
                pct = lines[i + 1]
                if re.match(r"^[+-]?\d+\.\d+%$", pct):
                    themes[line] = pct
                    i += 2
                    continue
        i += 1

    return themes


def _format_summary(period_data: dict[str, dict[str, str]]) -> str:
    """
    Formata resumo multi-período ordenado do maior para o menor retorno.
    """
    sections: list[str] = []

    for period in _PERIODS:
        themes = period_data.get(period)
        if not themes:
            continue

        def _sort_key(item: tuple[str, str]) -> float:
            try:
                return float(item[1].replace("%", ""))
            except ValueError:
                return 0.0

        sorted_themes = sorted(themes.items(), key=_sort_key, reverse=True)
        lines = [f"  {name}: {pct}" for name, pct in sorted_themes]
        sections.append(f"[{period}]\n" + "\n".join(lines))

    return "\n\n".join(sections)
