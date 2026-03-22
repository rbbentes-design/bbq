from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """
    Evidência individual que suporta um SignalCandidate.
    Pode ser um bloco do Market Ear, um item do X ou outro artefato.
    """

    id: str = Field(description="ULID único da evidência")
    kind: Literal["market_ear_block", "x_item", "external"] = Field(
        description="Tipo da evidência"
    )
    source: str = Field(description="Nome da fonte: 'zerohedge' | 'x' | outro")
    reference_id: str = Field(description="ID do objeto referenciado (bloco, item etc.)")
    summary: str = Field(description="Resumo textual da evidência")
    url: str = Field(default="", description="URL de origem da evidência")
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confiança nesta evidência (0–1)",
    )
