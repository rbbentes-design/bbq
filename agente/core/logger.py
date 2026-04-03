"""
MacroDesk Bloomberg Ecosystem — Logger
=======================================

Logger estruturado para o pipeline de dados Bloomberg.
Grava simultâneamente em:
  - console (Rich, colorido)
  - arquivo de log em data/logs/ (rotação diária)

Uso:
    from core.logger import get_logger
    log = get_logger("bloomberg_agent")
    log.info("zip_found", zip_name="bql_data_20260403.zip")
    log.error("parse_failed", file="prices.csv", reason="encoding error")
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


# ── Cores para console (ANSI) ─────────────────────────────────────────────────
_COLORS = {
    "DEBUG":   "\033[36m",   # cyan
    "INFO":    "\033[32m",   # green
    "WARNING": "\033[33m",   # yellow
    "ERROR":   "\033[31m",   # red
    "RESET":   "\033[0m",
}


class _StructuredFormatter(logging.Formatter):
    """Formata log como JSON de uma linha — ideal para arquivos."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        data: dict[str, Any] = {
            "ts":    datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "level": record.levelname,
            "name":  record.name,
            "event": record.getMessage(),
        }
        # Campos extras adicionados via log.info("event", key=val)
        extras = getattr(record, "_extra", {})
        if extras:
            data.update(extras)
        return json.dumps(data, ensure_ascii=False)


class _ConsoleFormatter(logging.Formatter):
    """Formata log legível para console, com cores ANSI."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        color = _COLORS.get(record.levelname, "")
        reset = _COLORS["RESET"]
        ts    = datetime.now().strftime("%H:%M:%S")
        extras = getattr(record, "_extra", {})
        extra_str = "  " + "  ".join(f"{k}={v}" for k, v in extras.items()) if extras else ""
        return f"{color}[{ts}] {record.levelname:<8} {record.name} — {record.getMessage()}{extra_str}{reset}"


class BoundLogger:
    """
    Logger com interface key=value para facilitar structured logging.

    Exemplo:
        log.info("zip_extracted", files=3, dest="bql_data/")
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _emit(self, level: int, event: str, **kw: Any) -> None:
        record = self._logger.makeRecord(
            self._logger.name, level, "(none)", 0, event, (), None
        )
        record._extra = kw  # type: ignore[attr-defined]
        self._logger.handle(record)

    def debug(self, event: str, **kw: Any) -> None:
        self._emit(logging.DEBUG, event, **kw)

    def info(self, event: str, **kw: Any) -> None:
        self._emit(logging.INFO, event, **kw)

    def warning(self, event: str, **kw: Any) -> None:
        self._emit(logging.WARNING, event, **kw)

    def error(self, event: str, **kw: Any) -> None:
        self._emit(logging.ERROR, event, **kw)


def get_logger(name: str, logs_dir: Path | None = None) -> BoundLogger:
    """
    Retorna um BoundLogger configurado para o componente.

    Args:
        name:     Nome do componente (ex: "bloomberg_agent", "zip_extractor").
        logs_dir: Pasta onde o arquivo de log será gravado.
                  Se None, usa o padrão de settings.py.
    """
    # Evita duplicar handlers se já configurado
    existing = logging.getLogger(name)
    if existing.handlers:
        return BoundLogger(existing)

    if logs_dir is None:
        try:
            import sys as _sys
            from pathlib import Path as _P
            _root = _P(__file__).parent.parent
            _sys.path.insert(0, str(_root))
            from config.settings import LOGS_DIR
            logs_dir = LOGS_DIR
        except Exception:
            logs_dir = Path("data/logs")

    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Handler de console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_ConsoleFormatter())
    logger.addHandler(ch)

    # Handler de arquivo (rotação diária via sufixo no nome)
    date_str = datetime.now().strftime("%Y%m%d")
    log_file = logs_dir / f"bloomberg_agent_{date_str}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_StructuredFormatter())
    logger.addHandler(fh)

    return BoundLogger(logger)
