"""
Curation: ISQ Signal Extractor

Extrai um sinal ISQ (Investment Signal Qualification) estruturado
a partir da narrativa curada. Baseado no framework ISQ do AlphaEar.

O ISQ inclui:
  - Cadeia de transmissão causal (nó a nó, do gatilho ao efeito final)
  - Lista de ativos impactados com direção e peso
  - Score de sentimento, confiança e intensidade
  - Justificativa qualitativa completa
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from app.audit.logger import get_logger
from app.curation.models import CurationResult
from app.models.isq_signal import ISQSignal, ImpactTicker, TransmissionNode

_log = get_logger("curation.isq")

_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM_PROMPT = """\
Você é um analista quantitativo especializado em qualificação de sinais de investimento.
Sua tarefa é extrair um ISQ (Investment Signal Qualification) estruturado e preciso.
Responda APENAS com JSON válido, sem markdown, sem explicações adicionais.\
"""


def extract_isq(result: CurationResult, run_id: str) -> ISQSignal | None:
    """
    Extrai ISQ estruturado a partir de uma CurationResult.

    Args:
        result: Resultado da curadoria LLM.
        run_id: ID do run atual.

    Returns:
        ISQSignal populado, ou None se falhar.
    """
    from app.curation.llm_client import call_claude

    primary = result.narrative.primary_signal
    secondary = result.narrative.secondary_signals[:2] if result.narrative.secondary_signals else []

    secondary_text = ""
    if secondary:
        secondary_text = "\n".join(
            f"  - {s.label} (confiança: {s.confidence:.0%})" for s in secondary
        )
        secondary_text = f"\nSinais secundários:\n{secondary_text}"

    evidence_text = ""
    if primary.evidence_quotes:
        evidence_text = "\n".join(f"  - {q}" for q in primary.evidence_quotes[:5])
        evidence_text = f"\nEvidências:\n{evidence_text}"

    user_prompt = f"""Narrativa primária detectada hoje:
{primary.label}

Descrição:
{primary.description}

Confiança: {primary.confidence:.0%}
{secondary_text}
{evidence_text}

Gere um ISQ completo para este sinal. O ISQ deve capturar:
1. A cadeia causal: do gatilho até os efeitos nos mercados (máximo 6 nós)
2. Os ativos mais impactados com direção e peso (máximo 6 tickers)
3. Um score de sentimento geral do sinal para o mercado

Responda APENAS com JSON neste formato exato:
{{
  "title": "Título conciso do sinal (max 80 chars)",
  "transmission_chain": [
    {{
      "node": "Nome do nó causal",
      "impact": "positive" | "negative" | "neutral",
      "reasoning": "Por que este nó leva ao próximo"
    }}
  ],
  "impact_tickers": [
    {{
      "ticker": "símbolo Yahoo Finance",
      "name": "nome amigável",
      "direction": "long" | "short" | "neutral",
      "weight": 0.0-1.0,
      "reasoning": "por que este ativo é impactado"
    }}
  ],
  "sentiment_score": -1.0 a 1.0,
  "confidence": 0.0 a 1.0,
  "intensity": 1 a 5,
  "reasoning": "Análise qualitativa do sinal em 3-5 frases"
}}

Use tickers reais do Yahoo Finance (^GSPC, CL=F, GLD, TLT, ^VIX, BTC-USD, etc.).\
"""

    try:
        raw = call_claude(
            prompt_system=_SYSTEM_PROMPT,
            prompt_user=user_prompt,
            model=_MODEL,
            max_tokens=2000,
            temperature=0.1,
        )
        # Limpa markdown se presente
        raw = raw.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw.strip())

        nodes = [TransmissionNode(**n) for n in data.get("transmission_chain", [])]
        tickers = [ImpactTicker(**t) for t in data.get("impact_tickers", [])]

        signal = ISQSignal(
            run_id=run_id,
            title=data.get("title", primary.label[:80]),
            transmission_chain=nodes,
            impact_tickers=tickers,
            sentiment_score=float(data.get("sentiment_score", 0.0)),
            confidence=float(data.get("confidence", primary.confidence)),
            intensity=int(data.get("intensity", 3)),
            reasoning=data.get("reasoning", ""),
            generated_at=datetime.now(timezone.utc),
        )

        _log.info(
            "isq_done",
            run_id=run_id,
            title=signal.title[:60],
            nodes=len(nodes),
            tickers=len(tickers),
            intensity=signal.intensity,
        )
        return signal

    except Exception as exc:
        _log.warning("isq_error", run_id=run_id, error=str(exc))
        return None


def save_isq(signal: ISQSignal, output_path: "Path") -> None:  # type: ignore[name-defined]
    """Persiste o ISQ como JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(signal.model_dump_json(indent=2), encoding="utf-8")
    _log.info("isq_saved", path=str(output_path))
