"""
SessionManager — ponto central para abrir/fechar o browser e verificar sessoes.

Uso:
    from app.auth.session_manager import SessionManager

    with SessionManager() as sm:
        zh_ok = sm.check_zerohedge()
        x_ok  = sm.check_x()
        # ... coletar dados ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from playwright.sync_api import BrowserContext, Page, sync_playwright

from app.audit.logger import get_logger
from app.auth.browser_profile import open_context
from app.auth.validators import is_x_logged_in, is_zerohedge_logged_in

_log = get_logger("auth.session_manager")


@dataclass
class SessionStatus:
    zerohedge: bool = False
    x: bool = False
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def all_ok(self) -> bool:
        return self.zerohedge and self.x

    def summary(self) -> str:
        parts = [
            f"ZeroHedge={'OK' if self.zerohedge else 'FAIL'}",
            f"X={'OK' if self.x else 'FAIL'}",
        ]
        if self.errors:
            parts.append(f"errors={list(self.errors.keys())}")
        return " | ".join(parts)


class SessionManager:
    """
    Gerencia o ciclo de vida do BrowserContext Playwright.

    Pode ser usado como context manager (recomendado) ou manualmente:
        sm = SessionManager(); sm.open(); ...; sm.close()
    """

    def __init__(self, headless: bool | None = None) -> None:
        self._headless = headless
        self._playwright: Any = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    # ── Context Manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "SessionManager":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── Ciclo de vida ─────────────────────────────────────────────────────────

    def open(self) -> None:
        """Abre o Playwright e o BrowserContext com o perfil Chrome."""
        self._playwright = sync_playwright().start()
        self._context = open_context(self._playwright, headless=self._headless)
        self._page = self._context.new_page()
        _log.info("browser_opened", profile="Chrome/Default")

    def close(self) -> None:
        """Fecha o contexto e o Playwright."""
        try:
            if self._context:
                self._context.close()
            if self._playwright:
                self._playwright.stop()
        except Exception as exc:
            _log.warning("browser_close_error", error=str(exc))
        finally:
            self._context = None
            self._playwright = None
            self._page = None
        _log.info("browser_closed")

    # ── Acesso a paginas ──────────────────────────────────────────────────────

    @property
    def page(self) -> Page:
        """Pagina principal (reutilizada entre chamadas)."""
        if self._page is None:
            raise RuntimeError("SessionManager nao esta aberto. Use .open() ou como context manager.")
        return self._page

    @property
    def context(self) -> BrowserContext:
        if self._context is None:
            raise RuntimeError("SessionManager nao esta aberto.")
        return self._context

    def new_page(self) -> Page:
        """Abre uma nova aba no contexto atual."""
        return self.context.new_page()

    # ── Validacao de sessoes ──────────────────────────────────────────────────

    def check_zerohedge(self) -> bool:
        """Verifica se a sessao ZeroHedge esta ativa."""
        result = is_zerohedge_logged_in(self.page)
        _log.info("session_check", site="zerohedge", ok=result)
        return result

    def check_x(self) -> bool:
        """Verifica se a sessao X esta ativa."""
        result = is_x_logged_in(self.page)
        _log.info("session_check", site="x", ok=result)
        return result

    def check_all(self) -> SessionStatus:
        """Verifica todas as sessoes e retorna um SessionStatus."""
        status = SessionStatus()

        try:
            status.zerohedge = self.check_zerohedge()
        except Exception as exc:
            status.errors["zerohedge"] = str(exc)
            _log.error("session_check_error", site="zerohedge", error=str(exc))

        try:
            status.x = self.check_x()
        except Exception as exc:
            status.errors["x"] = str(exc)
            _log.error("session_check_error", site="x", error=str(exc))

        _log.info("session_check_summary", **{
            "zerohedge": status.zerohedge,
            "x": status.x,
        })
        return status
