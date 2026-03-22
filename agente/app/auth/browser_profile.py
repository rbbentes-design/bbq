"""
Gerenciamento do perfil Chrome para Playwright.

Usa launch_persistent_context com o perfil Chrome existente do usuario,
evitando qualquer necessidade de login manual — as sessoes ja estao ativas.

IMPORTANTE: Chrome deve estar FECHADO antes de chamar open_context().
O Playwright nao consegue abrir um perfil que outro processo Chrome ja detem.
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import BrowserContext, Playwright

from app.config.settings import settings


def chrome_profile_dir() -> Path:
    """Retorna o caminho completo do perfil Chrome selecionado."""
    return settings.chrome_user_data_dir / settings.chrome_profile


def open_context(
    playwright: Playwright,
    *,
    headless: bool | None = None,
    slow_mo: int = 0,
) -> BrowserContext:
    """
    Abre um BrowserContext Playwright reutilizando o perfil Chrome do usuario.

    O perfil ja contem cookies/sessoes de ZeroHedge e X — nao precisa logar.

    Args:
        playwright: instancia sync Playwright (do `with sync_playwright() as p`).
        headless: sobrescreve settings.playwright_headless se fornecido.
        slow_mo: milissegundos de delay entre acoes (util para debug).

    Returns:
        BrowserContext pronto para uso.

    Raises:
        RuntimeError: se o user_data_dir nao existir.
    """
    user_data_dir = settings.chrome_user_data_dir
    if not user_data_dir.exists():
        raise RuntimeError(
            f"Chrome user_data_dir nao encontrado: {user_data_dir}\n"
            "Verifique CHROME_USER_DATA_DIR no .env"
        )

    _headless = headless if headless is not None else settings.playwright_headless

    context = playwright.chromium.launch_persistent_context(
        user_data_dir=str(user_data_dir),
        channel="chrome",
        headless=_headless,
        slow_mo=slow_mo,
        args=[
            f"--profile-directory={settings.chrome_profile}",
            "--disable-blink-features=AutomationControlled",
        ],
        ignore_default_args=["--enable-automation"],
    )
    return context
