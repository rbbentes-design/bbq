from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.models.evidence_item import EvidenceItem


class SignalCandidate(BaseModel):
    """
    Sinal editorial candidato gerado a partir dos dados coletados.
    Representa um tema ou narrativa que pode vir a ser um artigo.
    Nesta etapa: identificação e ranking apenas — sem escrita.
    """

    id: str = Field(description="ULID único do sinal")
    label: str = Field(description="Rótulo curto do sinal (ex: 'CTA positioning extremo')")
    summary: str = Field(description="Resumo do sinal em 1–3 frases")
    origin_sources: list[str] = Field(
        default_factory=list,
        description="Fontes que originaram o sinal: ['zerohedge', 'x']",
    )
    supporting_items: list[EvidenceItem] = Field(
        default_factory=list,
        description="Evidências que suportam este sinal",
    )
    detected_topics: list[str] = Field(
        default_factory=list,
        description="Tópicos associados ao sinal",
    )
    confidence_preliminary: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confiança preliminar no sinal (0–1)",
    )
    status: Literal["candidate", "discarded", "promoted"] = Field(
        default="candidate",
        description="Status do sinal nesta etapa",
    )
