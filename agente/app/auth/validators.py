"""
Validadores de sessao por site.

ZeroHedge: usa Coral Talk para auth — detecta por cookie coral_talk_sess.
X: detecta por cookie auth_token.
"""

from __future__ import annotations

import time

from playwright.sync_api import Page

from app.config.settings import settings


def is_zerohedge_logged_in(page: Page) -> bool:
    """
    Navega para ZeroHedge e verifica sessao por cookie coral_talk_sess.

    ZeroHedge usa o sistema Coral Talk para autenticacao de usuarios.
    O cookie coral_talk_sess (JWT, ~1200+ chars) e criado no login e
    persiste no perfil do browser.
    """
    try:
        page.goto(settings.zerohedge_market_ear_url, timeout=settings.auth_timeout_ms)
        page.wait_for_load_state("domcontentloaded", timeout=15_000)
        time.sleep(3)

        url = page.url
        if "zerohedge.com" not in url:
            return False
        if "/user/login" in url:
            return False

        cookies = page.context.cookies()
        for c in cookies:
            if c.get("domain", "").endswith("zerohedge.com"):
                if c.get("name") == "coral_talk_sess" and len(c.get("value", "")) > 100:
                    return True
        return False
    except Exception:
        return False


def is_x_logged_in(page: Page) -> bool:
    """
    Navega para X e verifica sessao por cookie auth_token.
    """
    try:
        page.goto("https://x.com/i/timeline", timeout=settings.auth_timeout_ms)
        page.wait_for_load_state("domcontentloaded", timeout=15_000)
        time.sleep(3)

        url = page.url
        if "/login" in url or "/i/flow/" in url:
            return False

        cookies = page.context.cookies()
        for c in cookies:
            domain = c.get("domain", "")
            if domain.endswith("x.com") or domain.endswith("twitter.com"):
                if c.get("name") == "auth_token" and c.get("value"):
                    return True

        return page.locator('[data-testid="primaryColumn"]').count() > 0
    except Exception:
        return False
