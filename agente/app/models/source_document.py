from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """
    Representa o documento bruto coletado de uma fonte.
    Vínculo entre o artefato físico (HTML salvo em disco) e os metadados da coleta.
    """

    id: str = Field(description="ULID único da coleta")
    source_name: str = Field(description="Nome da fonte: 'zerohedge' | 'x'")
    source_url: str = Field(description="URL exata acessada")
    collected_at: datetime = Field(description="Timestamp UTC da coleta")
    access_method: str = Field(description="Método: 'playwright_chrome_profile' | 'api'")
    raw_content_path: str = Field(description="Caminho absoluto do HTML/JSON bruto salvo")
    content_hash: str = Field(description="SHA-256 do conteúdo bruto — detecção de duplicatas")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadados livres da coleta")
