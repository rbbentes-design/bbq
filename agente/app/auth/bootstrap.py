"""
Bootstrap de autenticacao manual.

ZeroHedge: detecta login por cookie coral_talk_sess (JWT do Coral Talk auth).
X: detecta login por cookie auth_token.
"""

from __future__ import annotations

import time

from playwright.sync_api import BrowserContext, Page

from app.audit.logger import get_logger
from app.config.settings import settings

_log = get_logger("auth.bootstrap")


# ── Verificadores sem navegacao ───────────────────────────────────────────────

def _zh_has_coral_session(page: Page) -> bool:
    """
    Verifica sessao ZeroHedge por cookie coral_talk_sess.
    Esse cookie (JWT ~1200+ chars) e criado quando o usuario faz login no ZeroHedge.
    """
    try:
        url = page.url
        if any(p in url for p in ["/user/login", "/user/register", "about:blank"]):
            return False
        cookies = page.context.cookies()
        for c in cookies:
            if c.get("domain", "").endswith("zerohedge.com"):
                if c.get("name") == "coral_talk_sess" and len(c.get("value", "")) > 100:
                    return True
        return False
    except Exception:
        return False


def _x_has_auth_token(page: Page) -> bool:
    """Verifica cookie auth_token do X (sem navegar)."""
    try:
        url = page.url
        if any(p in url for p in ["/login", "/i/flow/", "about:blank"]):
            return False
        cookies = page.context.cookies()
        for c in cookies:
            domain = c.get("domain", "")
            if domain.endswith("x.com") or domain.endswith("twitter.com"):
                if c.get("name") == "auth_token" and c.get("value"):
                    return True
        return False
    except Exception:
        return False


# ── Waiter generico ───────────────────────────────────────────────────────────

def _wait_for_login(
    page: Page,
    site: str,
    check_fn: "callable[[Page], bool]",
    timeout_s: int = 300,
    min_wait_s: int = 8,
) -> bool:
    """
    Aguarda login (polling a cada 3s, sem navegar).
    Espera min_wait_s antes de iniciar verificacao.
    """
    print(f"[{site}] Aguardando pagina carregar...", end="", flush=True)
    time.sleep(min_wait_s)
    print(" ok")

    deadline = time.time() + (timeout_s - min_wait_s)
    dots = 0
    while time.time() < deadline:
        try:
            if check_fn(page):
                print(f"\r[{site}] Login detectado!                           ")
                return True
        except Exception:
            pass
        dots = (dots + 1) % 4
        remaining = int(deadline - time.time())
        print(f"\r[{site}] Aguardando{'.' * dots}   {remaining}s restantes  ", end="", flush=True)
        time.sleep(3)
    print()
    return False


# ── Bootstrap por site ────────────────────────────────────────────────────────

def login_zerohedge(context: BrowserContext) -> bool:
    """
    Abre ZeroHedge Market Ear e aguarda login via botao no header.
    Detecta por cookie coral_talk_sess (token JWT do sistema de auth).
    """
    page = context.new_page()
    try:
        _log.info("bootstrap_start", site="zerohedge")
        page.goto(settings.zerohedge_market_ear_url, timeout=30_000)
        page.wait_for_load_state("domcontentloaded")

        print(f"\n[ZeroHedge] Pagina aberta: {settings.zerohedge_market_ear_url}")
        print("[ZeroHedge] Clique em 'LOG IN' no canto superior direito da pagina.")
        print("[ZeroHedge] Faca o login com seu usuario/senha. O script detecta o cookie coral_talk_sess.\n")

        ok = _wait_for_login(page, "ZeroHedge", _zh_has_coral_session)
        _log.info("bootstrap_ok" if ok else "bootstrap_timeout", site="zerohedge")
        return ok
    finally:
        page.close()


def login_x(context: BrowserContext) -> bool:
    """
    Abre X.com login e aguarda autenticacao por cookie auth_token.
    """
    page = context.new_page()
    try:
        _log.info("bootstrap_start", site="x")
        page.goto("https://x.com/login", timeout=30_000)
        page.wait_for_load_state("domcontentloaded")

        print(f"\n[X] Pagina de login aberta.")
        print("[X] Entre com seu usuario/senha e complete 2FA se necessario.")
        print("[X] O script detecta o cookie auth_token automaticamente.\n")

        ok = _wait_for_login(page, "X", _x_has_auth_token)
        _log.info("bootstrap_ok" if ok else "bootstrap_timeout", site="x")
        return ok
    finally:
        page.close()
