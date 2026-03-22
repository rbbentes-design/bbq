from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class EngagementInfo(BaseModel):
    replies: int = 0
    reposts: int = 0
    likes: int = 0
    views: int | None = None


class XTimelineItem(BaseModel):
    """
    Um item coletado do timeline do X.
    Preserva origem, método de acesso e métricas de tração.
    """

    id: str = Field(description="ULID único do item")
    author: str = Field(description="Handle do autor (@username)")
    text: str = Field(description="Texto completo do post")
    url: str = Field(description="URL do post no X")
    created_at: datetime | None = Field(default=None, description="Timestamp de criação do post")
    engagement_info: EngagementInfo = Field(default_factory=EngagementInfo)
    media_refs: list[str] = Field(default_factory=list, description="URLs de mídia anexada")
    detected_topics: list[str] = Field(default_factory=list, description="Tópicos extraídos por heurística")
    traction_score_preliminary: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score preliminar de tração (0–1)",
    )
    raw_source_document_id: str = Field(description="ID do SourceDocument de origem")
    extra: dict[str, Any] = Field(default_factory=dict, description="Campos extras do provider")
