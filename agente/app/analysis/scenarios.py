"""
Analysis: Market Scenario Generator

Gera cenários Bull / Base / Bear com probabilidades ponderadas,
catalisadores e níveis-alvo usando LLM + dados quantitativos.

Inspirado no Claude Equity Research skill (quant-sentiment-ai).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("analysis.scenarios")

_MODEL = "claude-haiku-4-5-20251001"


def generate_scenarios(
    narrative_label: str,
    narrative_description: str,
    market_prices: dict[str, Any],
    risk_metrics: dict[str, Any],
    monte_carlo: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """
    Gera cenários Bull/Base/Bear com probabilidades e catalisadores.

    Args:
        narrative_label:       Narrativa primária curada.
        narrative_description: Descrição da narrativa.
        market_prices:         Preços e retornos atuais.
        risk_metrics:          VaR, CVaR, drawdown por ticker.
        monte_carlo:           Projeções Monte Carlo.
        run_id:                ID do run.

    Returns:
        {
          "run_id": str,
          "narrative": str,
          "bull":  {"probability": 0.30, "catalyst": str, "narrative": str,
                    "spx_target": float, "time_horizon": str},
          "base":  {...},
          "bear":  {...},
          "generated_at": str,
        }
    """
    from app.curation.llm_client import call_claude
    from app.providers.market_prices import format_summary as fmt_prices
    from app.analysis.risk import format_summary as fmt_risk
    from app.analysis.monte_carlo import format_summary as fmt_mc

    context_blocks = []
    if market_prices:
        context_blocks.append(fmt_prices(market_prices))
    if risk_metrics:
        context_blocks.append(fmt_risk(risk_metrics, market_prices))
    if monte_carlo:
        context_blocks.append(fmt_mc(monte_carlo, market_prices))

    context = "\n\n".join(b for b in context_blocks if b)

    system_prompt = (
        "Você é um analista quantitativo sênior especializado em análise de cenários de mercado. "
        "Produza análises concisas, baseadas em dados, sem floreio. "
        "Sempre responda em JSON puro, sem markdown."
    )

    user_prompt = f"""Narrativa central do mercado hoje:
{narrative_label}

{narrative_description}

{context}

Com base na narrativa e nos dados quantitativos acima, gere 3 cenários de mercado para as próximas 2-4 semanas.
As probabilidades devem somar 100%.

Responda APENAS com JSON válido neste formato:
{{
  "bull": {{
    "probability": 0.25,
    "catalyst": "frase curta descrevendo o catalisador principal",
    "narrative": "2-3 frases descrevendo o cenário",
    "spx_target": 5400,
    "time_horizon": "2-3 semanas"
  }},
  "base": {{
    "probability": 0.50,
    "catalyst": "...",
    "narrative": "...",
    "spx_target": 5100,
    "time_horizon": "2-4 semanas"
  }},
  "bear": {{
    "probability": 0.25,
    "catalyst": "...",
    "narrative": "...",
    "spx_target": 4800,
    "time_horizon": "2-3 semanas"
  }}
}}"""

    try:
        raw = call_claude(
            prompt_system=system_prompt,
            prompt_user=user_prompt,
            model=_MODEL,
            max_tokens=1000,
            temperature=0.2,
        )
        # Extrai JSON da resposta
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        data["run_id"] = run_id
        data["narrative"] = narrative_label
        data["generated_at"] = datetime.now(timezone.utc).isoformat()
        _log.info("scenarios_done", run_id=run_id)
        return data
    except Exception as exc:
        _log.warning("scenarios_error", error=str(exc))
        return {}


def format_summary(scenarios: dict[str, Any]) -> str:
    """Formata cenários para inclusão no contexto do LLM."""
    if not scenarios:
        return ""
    lines = ["=== CENÁRIOS DE MERCADO ==="]
    for label in ("bull", "base", "bear"):
        s = scenarios.get(label, {})
        if not s:
            continue
        prob = s.get("probability", 0)
        catalyst = s.get("catalyst", "")
        spx = s.get("spx_target", "")
        lines.append(f"  [{label.upper()} {prob:.0%}] {catalyst} — SPX target: {spx}")
        if s.get("narrative"):
            lines.append(f"    {s['narrative']}")
    return "\n".join(lines)
