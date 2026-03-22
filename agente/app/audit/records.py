from __future__ import annotations

from typing import Any, Literal

from app.models.audit_record import AuditRecord
from app.utils.timestamps import new_ulid, utcnow


def make_record(
    run_id: str,
    stage: str,
    action: str,
    status: Literal["ok", "warning", "error", "skipped"],
    source: str = "",
    error_message: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AuditRecord:
    """Cria um AuditRecord preenchendo id e timestamp automaticamente."""
    return AuditRecord(
        id=new_ulid(),
        run_id=run_id,
        stage=stage,
        action=action,
        source=source,
        timestamp=utcnow(),
        status=status,
        error_message=error_message,
        metadata=metadata or {},
    )


def ok(run_id: str, stage: str, action: str, source: str = "", **meta: Any) -> AuditRecord:
    return make_record(run_id, stage, action, "ok", source=source, metadata=meta)


def warning(run_id: str, stage: str, action: str, msg: str, source: str = "", **meta: Any) -> AuditRecord:
    return make_record(run_id, stage, action, "warning", source=source, error_message=msg, metadata=meta)


def error(run_id: str, stage: str, action: str, msg: str, source: str = "", **meta: Any) -> AuditRecord:
    return make_record(run_id, stage, action, "error", source=source, error_message=msg, metadata=meta)


def skipped(run_id: str, stage: str, action: str, msg: str, source: str = "", **meta: Any) -> AuditRecord:
    return make_record(run_id, stage, action, "skipped", source=source, error_message=msg, metadata=meta)
