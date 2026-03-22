from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class AuditRecord(BaseModel):
    """
    Registro de auditoria de uma ação do sistema.
    Toda etapa relevante do pipeline gera um AuditRecord.
    Permite responder: o que foi feito, quando, onde, com qual resultado.
    """

    id: str = Field(description="ULID único do registro")
    run_id: str = Field(description="ID da execução do pipeline")
    stage: str = Field(description="Etapa do pipeline: 'auth' | 'collect' | 'parse' | 'rank' | 'bundle'")
    action: str = Field(description="Ação específica executada (ex: 'save_raw_html')")
    source: str = Field(default="", description="Fonte envolvida (ex: 'zerohedge', 'x')")
    timestamp: datetime = Field(description="Timestamp UTC da ação")
    status: Literal["ok", "warning", "error", "skipped"] = Field(description="Resultado da ação")
    error_message: str | None = Field(default=None, description="Mensagem de erro se status='error'")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Dados contextuais — nunca incluir segredos aqui",
    )
