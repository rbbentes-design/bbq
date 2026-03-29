"""
Model: ISQ Signal (Investment Signal Qualification)

Inspirado no framework ISQ do AlphaEar / Awesome-finance-skills.
Representa um sinal de investimento estruturado com:
  - Cadeia de transmissão causal (nó a nó)
  - Tickers impactados com pesos e direção
  - Score de sentimento, confiança e intensidade
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TransmissionNode(BaseModel):
    """Um nó na cadeia de transmissão causal do sinal."""
    node: str = Field(description="Nome do nó (ex: 'Tensão Iran-EUA', 'Alta do petróleo')")
    impact: Literal["positive", "negative", "neutral"] = Field(
        description="Impacto deste nó no próximo da cadeia"
    )
    reasoning: str = Field(description="Justificativa do impacto")


class ImpactTicker(BaseModel):
    """Ativo impactado pelo sinal."""
    ticker: str = Field(description="Símbolo do ativo (ex: 'CL=F', 'TLT', '^GSPC')")
    name: str = Field(default="", description="Nome amigável do ativo")
    direction: Literal["long", "short", "neutral"] = Field(
        description="Direção sugerida pela narrativa"
    )
    weight: float = Field(ge=0.0, le=1.0, description="Relevância relativa 0-1")
    reasoning: str = Field(description="Por que este ativo é impactado")


class ISQSignal(BaseModel):
    """
    Investment Signal Qualification — representação estruturada
    de um sinal de mercado com cadeia causal completa.
    """
    run_id: str
    title: str = Field(description="Título conciso do sinal (max 80 chars)")
    transmission_chain: list[TransmissionNode] = Field(
        default_factory=list,
        description="Cadeia de transmissão causal, do gatilho ao efeito final",
    )
    impact_tickers: list[ImpactTicker] = Field(
        default_factory=list,
        description="Ativos impactados, ordenados por peso descendente",
    )
    sentiment_score: float = Field(
        ge=-1.0, le=1.0,
        description="Score de sentimento do sinal: -1 (muito bearish) a +1 (muito bullish)",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confiança no sinal 0-1")
    intensity: int = Field(ge=1, le=5, description="Intensidade do sinal 1-5 (5 = extremo)")
    reasoning: str = Field(description="Análise qualitativa do sinal")
    generated_at: datetime = Field(default_factory=lambda: datetime.utcnow())
