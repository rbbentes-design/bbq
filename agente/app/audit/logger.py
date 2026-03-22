from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import structlog

from app.models.audit_record import AuditRecord

# Campos que NUNCA devem aparecer em logs — redação automática
_SENSITIVE_KEYS = frozenset({
    "password", "passwd", "token", "cookie", "cookies",
    "storage_state", "secret", "authorization", "api_key",
    "access_token", "refresh_token", "session",
})


def _redact(obj: Any, depth: int = 0) -> Any:
    """Remove recursivamente campos sensíveis de dicts antes de logar."""
    if depth > 5:
        return obj
    if isinstance(obj, dict):
        return {
            k: "[REDACTED]" if k.lower() in _SENSITIVE_KEYS else _redact(v, depth + 1)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(i, depth + 1) for i in obj]
    return obj


def configure_logging(log_level: str = "INFO") -> None:
    """Configura structlog para output JSON estruturado."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(message)s",
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


class AuditLogger:
    """
    Grava AuditRecords em arquivo JSONL e em structlog simultaneamente.
    Cada linha do JSONL é um AuditRecord serializado — permite replay e análise.
    Nunca expõe campos sensíveis.
    """

    def __init__(self, log_path: Path) -> None:
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._log = get_logger("audit")

    def write(self, record: AuditRecord) -> None:
        safe_meta = _redact(record.metadata)
        line = record.model_copy(update={"metadata": safe_meta})

        # Append ao JSONL
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line.model_dump_json() + "\n")

        # Log estruturado
        log_fn = getattr(self._log, "warning" if record.status == "warning" else record.status, self._log.info)
        if record.status == "error":
            self._log.error(
                record.action, stage=record.stage, source=record.source,
                run_id=record.run_id, error=record.error_message, **safe_meta,
            )
        elif record.status == "warning":
            self._log.warning(
                record.action, stage=record.stage, source=record.source,
                run_id=record.run_id, warning=record.error_message, **safe_meta,
            )
        else:
            self._log.info(
                record.action, stage=record.stage, source=record.source,
                run_id=record.run_id, **safe_meta,
            )

    def read_all(self) -> list[AuditRecord]:
        if not self._path.exists():
            return []
        records = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(AuditRecord.model_validate_json(line))
        return records
