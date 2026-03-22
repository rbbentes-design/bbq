"""
Validadores de sessao por site.

Cada funcao recebe uma Page aberta e retorna True se a sessao esta ativa,
False se o usuario esta deslogado (ex: pagina de login apareceu).
"""

from __future__ import annotations

from playwright.sync_api import Page

from app.config.settings import settings


def is_zerohedge_logged_in(page: Page) -> bool:
    """
    Navega para ZeroHedge Market Ear e verifica se ha sessao ativa.

    Indicadores de sessao ativa: ausencia de link /login, presenca de conteudo editorial.
    """
    try:
        page.goto(settings.zerohedge_market_ear_url, timeout=settings.auth_timeout_ms)
        page.wait_for_load_state("domcontentloaded", timeout=15_000)

        # Se aparece formulario de login, sessao nao esta ativa
        login_form = page.locator("form[action*='login'], input[name='pass']")
        if login_form.count() > 0:
            return False

        # Verifica se tem conteudo editorial (qualquer artigo/bloco)
        content = page.locator("article, .node--type-zh-blog-entry, .market-ear")
        return content.count() > 0
    except Exception:
        return False


def is_x_logged_in(page: Page) -> bool:
    """
    Navega para X (Twitter) e verifica se ha sessao ativa.

    Indicadores de sessao ativa: ausencia de botao Sign In na home, presenca de timeline.
    """
    try:
        page.goto("https://x.com/home", timeout=settings.auth_timeout_ms)
        page.wait_for_load_state("domcontentloaded", timeout=15_000)

        # Se redirecionou para /i/flow/login ou login page, nao esta logado
        if "/login" in page.url or "/i/flow/" in page.url:
            return False

        # Verifica presenca de elementos da timeline autenticada
        timeline = page.locator(
            '[data-testid="primaryColumn"], [aria-label="Timeline: Your Home Timeline"]'
        )
        return timeline.count() > 0
    except Exception:
        return False
