from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Retorna datetime atual em UTC com tzinfo explícito."""
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    """Retorna ISO 8601 UTC atual."""
    return utcnow().isoformat()


def new_ulid() -> str:
    """Gera um ULID único (lexicograficamente ordenável por tempo)."""
    from ulid import ULID
    return str(ULID())
