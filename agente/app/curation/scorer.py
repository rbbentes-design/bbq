from __future__ import annotations

import json
import re

from app.audit.logger import get_logger
from app.curation.corpus import build_item_index, item_to_snippet
from app.curation.llm_client import call_claude
from app.curation.models import Narrative, ScoredItem
from app.models.daily_ingestion_bundle import DailyIngestionBundle

_log = get_logger("curation.scorer")
_MODEL = "claude-sonnet-4-6"

_SYSTEM = """\
You are a financial relevance scorer. Given a detected narrative and a list of corpus items, \
score each item's relevance to the narrative.

For each item output a JSON line with exactly this format:
{"item_id": "<id>", "narrative_relevance": <0.00-1.00>, "relevance_reason": "<1 sentence>"}

Rules:
- Score 0.0 if the item has NO connection to the narrative.
- Score 1.0 only if the item is a primary source/driver of the narrative.
- Output ONE JSON line per item. Do NOT output any text before or after the JSON lines.
- Keep relevance_reason under 20 words.
- Use double quotes in JSON. No trailing commas."""


def score_items(
    narrative: Narrative,
    bundle: DailyIngestionBundle,
) -> list[ScoredItem]:
    item_index = build_item_index(bundle)
    item_ids = list(item_index.keys())
    # Use short IDs only (ME-xxx / X-xxx), skip full ULIDs (duplicates)
    short_ids = [sid for sid in item_ids if sid.startswith(("ME-", "X-"))]

    if not short_ids:
        return []

    signal = narrative.primary_signal
    all_signals = [signal] + narrative.secondary_signals

    # Build prompt
    signal_block = "\n".join(
        f"- [{s.label}] (confidence {s.confidence:.2f}): {s.description}"
        for s in all_signals
        if s is not None
    )

    items_block = "\n\n".join(
        f"{sid}:\n{item_to_snippet(item_index[sid])}"
        for sid in short_ids
    )

    user_prompt = (
        f"DETECTED NARRATIVE(S):\n{signal_block}\n\n"
        f"ITEMS TO SCORE:\n{items_block}"
    )

    _log.info("scoring_items", n_items=len(short_ids), narrative=signal.label[:40])

    try:
        raw = call_claude(_SYSTEM, user_prompt, model=_MODEL, max_tokens=8192)
    except Exception as exc:
        _log.warning("scorer_error", error=str(exc))
        return []

    parsed = _parse_scores(raw, all_signals)
    if not parsed:
        _log.warning("scorer_parse_zero", raw_sample=repr(raw[:300]))
    return parsed


def _parse_scores(raw: str, signals: list) -> list[ScoredItem]:
    results: list[ScoredItem] = []
    signal_labels = [s.label for s in signals if s is not None]

    for line in raw.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        # Fix invalid JSON escapes the LLM sometimes produces
        line = line.replace("\\'", "'")
        try:
            obj = json.loads(line)
            item_id = obj.get("item_id", "")
            relevance = float(obj.get("narrative_relevance", 0.0))
            reason = obj.get("relevance_reason", "")
            if not item_id:
                continue
            results.append(ScoredItem(
                item_id=item_id,
                item_type="market_ear_block" if item_id.startswith("ME-") else "x_item",
                narrative_relevance=min(max(relevance, 0.0), 1.0),
                relevance_reason=reason,
                signal_ids=signal_labels,
            ))
        except (json.JSONDecodeError, ValueError):
            continue

    _log.info("scoring_done", scored=len(results))
    return results
