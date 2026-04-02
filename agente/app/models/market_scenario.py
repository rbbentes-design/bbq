"""
Model: Market Scenario

Representação estruturada dos cenários Bull / Base / Bear
gerados pelo analysis.scenarios.
"""

from __future__ import annotations

from datetime import datetime, timezone as _tz
from typing import Any

from pydantic import BaseModel, Field


class ScenarioCase(BaseModel):
    """Um cenário individual (bull, base ou bear)."""
    probability: float = Field(ge=0.0, le=1.0, description="Probabilidade estimada 0-1")
    catalyst: str = Field(description="Catalisador principal do cenário")
    narrative: str = Field(description="Descrição do cenário em 2-3 frases")
    spx_target: float | None = Field(default=None, description="Nível-alvo do S&P 500")
    time_horizon: str = Field(default="2-4 semanas", description="Horizonte de tempo")
    key_levels: dict[str, float] = Field(
        default_factory=dict,
        description="Níveis-chave para outros ativos {sym: level}",
    )


class MarketScenario(BaseModel):
    """Conjunto de cenários Bull / Base / Bear para o período."""
    run_id: str
    narrative: str = Field(description="Narrativa central que originou os cenários")
    bull: ScenarioCase
    base: ScenarioCase
    bear: ScenarioCase
    generated_at: datetime = Field(default_factory=lambda: datetime.now(_tz.utc))

    @classmethod
    def from_dict(cls, data: dict[str, Any], run_id: str, narrative: str) -> "MarketScenario":
        """Cria MarketScenario a partir do output do LLM (dict)."""
        return cls(
            run_id=run_id,
            narrative=narrative,
            bull=ScenarioCase(**data["bull"]),
            base=ScenarioCase(**data["base"]),
            bear=ScenarioCase(**data["bear"]),
        )
