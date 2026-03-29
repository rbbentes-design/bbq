from __future__ import annotations

import re
from datetime import datetime, timezone

from app.audit.logger import get_logger
from app.curation.llm_client import call_claude
from app.curation.models import CorrectionEntry, Narrative, NarrativeSignal
from app.utils.timestamps import new_ulid

_log = get_logger("curation.narrative_detector")

_MODEL = "claude-sonnet-4-6"

_SYSTEM = """\
You are a financial narrative analyst. Read the full day's market content \
and identify the 1-3 most significant emergent narratives or signals.

Rules:
- Do NOT invent themes — every claim must be grounded in verbatim quotes from the corpus.
- If no strong narrative exists, say so with confidence < 0.40.
- Quotes must be EXACT substrings from the corpus items.
- Be specific: name assets, rates, spreads, timeframes.

Respond in EXACTLY this format (no extra text outside the markers):
---BEGIN---
PRIMARY NARRATIVE: <short label, max 10 words>
CONFIDENCE: <0.00-1.00>
DESCRIPTION: <2-4 sentences synthesis>
SUPPORTING_IDS: <comma-separated corpus IDs like ME-01J9X, X-01JA2>
EVIDENCE_QUOTES:
  - "<exact verbatim quote>"
  - "<exact verbatim quote>"

SECONDARY NARRATIVE: <label or NONE>
CONFIDENCE: <0.00-1.00>
DESCRIPTION: <2-4 sentences or NONE>
SUPPORTING_IDS: <IDs or NONE>
EVIDENCE_QUOTES:
  - "<quote or NONE>"
---END---"""


def detect_narrative(
    corpus_text: str,
    run_id: str,
    run_date: str,
    few_shot_examples: list[CorrectionEntry] | None = None,
) -> Narrative:
    from app.curation.corrections import format_few_shot_block

    few_shot_block = format_few_shot_block(few_shot_examples or [])

    user_prompt = f"Today is {run_date}.\n"
    if few_shot_block:
        user_prompt += f"\n{few_shot_block}\n"
    user_prompt += f"\nAnalyze the following corpus and detect today's narrative(s):\n\n{corpus_text}"

    raw = call_claude(_SYSTEM, user_prompt, model=_MODEL, max_tokens=2048)
    _log.debug("narrative_raw", chars=len(raw))

    primary, secondary = _parse_response(raw)

    return Narrative(
        id=new_ulid(),
        run_id=run_id,
        run_date=run_date,
        primary_signal=primary,
        secondary_signals=[secondary] if secondary else [],
        raw_llm_response=raw,
        detection_model=_MODEL,
        detected_at=datetime.now(timezone.utc),
    )


def _parse_response(raw: str) -> tuple[NarrativeSignal, NarrativeSignal | None]:
    # Extrai bloco entre ---BEGIN--- e ---END---
    m = re.search(r"---BEGIN---(.*?)---END---", raw, re.DOTALL)
    text = m.group(1).strip() if m else raw.strip()

    def _extract(section: str, key: str) -> str:
        pattern = rf"{re.escape(key)}:\s*(.+?)(?=\n[A-Z_]+:|$)"
        m = re.search(pattern, section, re.DOTALL)
        return m.group(1).strip() if m else ""

    def _extract_quotes(section: str) -> list[str]:
        return re.findall(r'- "(.+?)"', section)

    def _extract_ids(section: str) -> list[str]:
        raw_ids = _extract(section, "SUPPORTING_IDS")
        if not raw_ids or raw_ids.upper() == "NONE":
            return []
        return [i.strip() for i in raw_ids.split(",") if i.strip()]

    # Split em PRIMARY e SECONDARY
    parts = re.split(r"\nSECONDARY NARRATIVE:", text, maxsplit=1)
    primary_text = parts[0]
    secondary_text = parts[1] if len(parts) > 1 else ""

    def _parse_signal(section: str, is_secondary: bool = False) -> NarrativeSignal | None:
        if is_secondary:
            label = _extract(section, "")
            # primeira linha é o label
            label = section.strip().splitlines()[0].strip() if section.strip() else ""
        else:
            label = _extract(section, "PRIMARY NARRATIVE")

        if not label or label.upper() == "NONE":
            return None

        try:
            confidence = float(_extract(section, "CONFIDENCE") or "0.5")
        except ValueError:
            confidence = 0.5

        return NarrativeSignal(
            id=new_ulid(),
            label=label,
            description=_extract(section, "DESCRIPTION"),
            confidence=min(max(confidence, 0.0), 1.0),
            supporting_item_ids=_extract_ids(section),
            evidence_quotes=_extract_quotes(section),
        )

    primary = _parse_signal(primary_text) or NarrativeSignal(
        id=new_ulid(),
        label="Narrative unclear",
        description="Could not detect a clear narrative from today's corpus.",
        confidence=0.3,
    )

    secondary = _parse_signal(secondary_text, is_secondary=True) if secondary_text else None
    return primary, secondary
