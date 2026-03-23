"""
Gerenciamento do browser Playwright para o agente.

Usa um perfil persistente proprio no workspace (nao o Chrome do sistema),
armazenado em ~/agente-workspace/state/browser/.
Cada sessao e preservada entre execucoes via cookies persistentes do perfil.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import BrowserContext, Playwright

from app.config.settings import settings


def _browser_profile_dir() -> Path:
    """Diretorio do perfil Playwright proprio (dentro do workspace)."""
    return settings.resolved_state_dir() / "browser"


def open_context(
    playwright: Playwright,
    *,
    headless: bool | None = None,
    slow_mo: int = 0,
) -> BrowserContext:
    """
    Abre um BrowserContext Playwright com perfil persistente no workspace.

    O perfil armazena cookies/sessoes entre execucoes. Na primeira vez o
    browser abre sem sessao; use `agente auth login` para autenticar.

    Args:
        playwright: instancia sync Playwright.
        headless: sobrescreve settings.playwright_headless se fornecido.
        slow_mo: milissegundos de delay entre acoes (util para debug).

    Returns:
        BrowserContext pronto para uso.
    """
    profile_dir = _browser_profile_dir()
    profile_dir.mkdir(parents=True, exist_ok=True)

    _headless = headless if headless is not None else settings.playwright_headless

    context = playwright.chromium.launch_persistent_context(
        user_data_dir=str(profile_dir),
        headless=_headless,
        slow_mo=slow_mo,
        args=["--disable-blink-features=AutomationControlled"],
        ignore_default_args=["--enable-automation"],
        viewport={"width": 1280, "height": 900},
    )
    return context


def profile_dir() -> Path:
    """Expoe o diretorio do perfil (para info/debug)."""
    return _browser_profile_dir()
