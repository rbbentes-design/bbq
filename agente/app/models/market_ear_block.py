from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class MarketEarBlock(BaseModel):
    """
    Um bloco editorial extraído da página The Market Ear.
    Cada bloco corresponde a um item/post da página.
    """

    id: str = Field(description="ULID único do bloco")
    title: str = Field(default="", description="Título do bloco (pode estar vazio)")
    subtitle: str = Field(default="", description="Subtítulo ou lead")
    body_text: str = Field(default="", description="Texto completo do corpo")
    image_refs: list[str] = Field(default_factory=list, description="URLs das imagens referenciadas")
    source_url: str = Field(description="URL canônica do bloco ou da página")
    published_at: datetime | None = Field(default=None, description="Data de publicação se detectada")
    detected_topics: list[str] = Field(default_factory=list, description="Tópicos extraídos por heurística")
    relevance_score_preliminary: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score preliminar de relevância (0–1)",
    )
    raw_source_document_id: str = Field(description="ID do SourceDocument de origem")
    position_index: int = Field(description="Posição do bloco na página (0 = topo)")
