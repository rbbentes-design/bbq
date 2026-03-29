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


def _sg_has_session(page: Page) -> bool:
    """
    Verifica sessão SpotGamma.
    Detecta login bem-sucedido pela URL — após autenticar o browser redireciona
    para o dashboard (sai de /login). Cookie name varia por provedor de auth.
    """
    try:
        url = page.url
        # Ainda na página de login/auth → não logado
        if any(p in url for p in ["/login", "/signin", "/callback", "about:blank", "auth0", "cognito"]):
            return False
        # Redirecionou para qualquer página do dashboard → logado
        if "dashboard.spotgamma.com" in url:
            return True
        # Fallback: qualquer cookie com valor longo no domínio spotgamma
        for c in page.context.cookies():
            if "spotgamma" in c.get("domain", "") and len(c.get("value", "")) > 20:
                return True
        return False
    except Exception:
        return False


def login_spectra(context: BrowserContext) -> bool:
    """Abre Spectra Markets e aguarda login."""
    page = context.new_page()
    try:
        _log.info("bootstrap_start", site="spectra")
        page.goto("https://www.spectramarkets.com/my-account/", timeout=30_000)
        page.wait_for_load_state("domcontentloaded")
        print("\n[Spectra] Pagina de login aberta.")
        print("[Spectra] Entre com seu email/senha.")
        print("[Spectra] O script detecta o redirecionamento para o dashboard.\n")
        ok = _wait_for_login(page, "Spectra", _spectra_has_session)
        _log.info("bootstrap_ok" if ok else "bootstrap_timeout", site="spectra")
        return ok
    finally:
        page.close()


def _spectra_has_session(page: Page) -> bool:
    try:
        url = page.url
        if "my-account" in url and "login" not in url:
            # Verifica se mostra menu de usuário logado (não formulário de login)
            content = page.content()
            return "Log Out" in content or "logout" in content
        return False
    except Exception:
        return False


def login_spotgamma(context: BrowserContext) -> bool:
    """
    Abre o SpotGamma dashboard e aguarda login.
    Detecta por cookie de sessão ou URL autenticada.
    """
    page = context.new_page()
    try:
        _log.info("bootstrap_start", site="spotgamma")
        page.goto("https://dashboard.spotgamma.com/login", timeout=30_000)
        page.wait_for_load_state("domcontentloaded")

        print("\n[SpotGamma] Pagina de login aberta.")
        print("[SpotGamma] Entre com seu email/senha.")
        print("[SpotGamma] O script detecta o cookie de sessao automaticamente.\n")

        ok = _wait_for_login(page, "SpotGamma", _sg_has_session)
        _log.info("bootstrap_ok" if ok else "bootstrap_timeout", site="spotgamma")
        return ok
    finally:
        page.close()


def _deepvue_has_session(page: Page) -> bool:
    """
    Detecta login no DeepVue verificando localStorage (auth token JWT).
    DeepVue armazena sessão em localStorage, não em cookies.
    """
    try:
        # Verifica se há token de auth no localStorage
        token = page.evaluate("""
            () => {
                for (let k of Object.keys(localStorage)) {
                    let v = localStorage.getItem(k);
                    if (v && v.length > 50 && (
                        k.toLowerCase().includes('token') ||
                        k.toLowerCase().includes('auth') ||
                        k.toLowerCase().includes('user') ||
                        k.toLowerCase().includes('session') ||
                        (v.startsWith('ey') && v.includes('.'))  // JWT
                    )) return v.substring(0, 20);
                }
                return null;
            }
        """)
        if token:
            return True
        # Fallback: URL já saiu do login
        url = page.url
        if "app.deepvue.com" in url and "/login" not in url:
            return True
        return False
    except Exception:
        return False


def login_deepvue(context: BrowserContext) -> bool:
    """Abre DeepVue e aguarda login."""
    page = context.new_page()
    try:
        _log.info("bootstrap_start", site="deepvue")
        page.goto("https://app.deepvue.com/login", timeout=30_000)
        page.wait_for_load_state("domcontentloaded")
        print("\n[DeepVue] Pagina de login aberta.")
        print("[DeepVue] Entre com seu email/senha.")
        print("[DeepVue] O script detecta o redirecionamento para o dashboard.\n")
        ok = _wait_for_login(page, "DeepVue", _deepvue_has_session, timeout_s=600)
        _log.info("bootstrap_ok" if ok else "bootstrap_timeout", site="deepvue")
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
