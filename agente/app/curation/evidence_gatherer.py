from __future__ import annotations

from app.audit.logger import get_logger
from app.curation.corpus import item_to_snippet
from app.curation.llm_client import call_claude
from app.curation.models import (
    EvidenceGatheringTrace,
    EvidenceIteration,
    NarrativeSignal,
)
from app.models.market_ear_block import MarketEarBlock
from app.models.x_timeline_item import XTimelineItem

_log = get_logger("curation.evidence_gatherer")
_MODEL = "claude-sonnet-4-6"

_SYSTEM = """\
You are a financial evidence analyst. Assess whether the corpus items below \
provide additional evidence for a given signal.
Be conservative: only increase confidence when there is clear textual grounding.

Respond EXACTLY in this format:
ADDS_CONFIDENCE: yes | no
NEW_SUPPORTING_IDS: <comma-separated IDs or none>
UPDATED_CONFIDENCE: <0.00-1.00>
REASONING: <1-2 sentences>"""


def gather_evidence(
    signal: NarrativeSignal,
    item_index: dict[str, MarketEarBlock | XTimelineItem],
    threshold: float = 0.65,
    max_iterations: int = 3,
) -> tuple[NarrativeSignal, EvidenceGatheringTrace]:
    trace = EvidenceGatheringTrace(signal_id=signal.id)

    if signal.confidence >= threshold:
        trace.final_confidence = signal.confidence
        trace.terminated_reason = "threshold_met"
        return signal, trace

    examined: set[str] = set(signal.supporting_item_ids)
    remaining = [sid for sid in item_index if sid not in examined and not sid.startswith(tuple(examined))]

    for iteration in range(max_iterations):
        batch_ids = remaining[:10]
        if not batch_ids:
            trace.terminated_reason = "inconclusive"
            break

        batch_text = "\n\n".join(
            item_to_snippet(item_index[sid]) for sid in batch_ids if sid in item_index
        )

        user_prompt = (
            f"SIGNAL: {signal.label}\n"
            f"CURRENT CONFIDENCE: {signal.confidence:.2f}\n"
            f"DESCRIPTION: {signal.description}\n\n"
            f"EXISTING EVIDENCE:\n" + "\n".join(f'- "{q}"' for q in signal.evidence_quotes[:5]) +
            f"\n\nNEW ITEMS TO EXAMINE:\n{batch_text}"
        )

        conf_before = signal.confidence
        try:
            raw = call_claude(_SYSTEM, user_prompt, model=_MODEL, max_tokens=512)
            new_ids, new_conf = _parse_response(raw)
        except Exception as exc:
            _log.warning("evidence_gather_error", iteration=iteration, error=str(exc))
            break

        # Atualiza signal
        new_items = [sid for sid in new_ids if sid in item_index and sid not in examined]
        examined.update(new_ids)
        remaining = [sid for sid in remaining if sid not in examined]

        signal = signal.model_copy(update={
            "supporting_item_ids": signal.supporting_item_ids + new_items,
            "confidence": new_conf,
        })

        trace.iterations.append(EvidenceIteration(
            iteration=iteration,
            query_used=signal.label,
            items_examined=len(batch_ids),
            confidence_before=conf_before,
            confidence_after=new_conf,
            new_evidence_found=new_items,
        ))

        _log.info("evidence_iteration", signal=signal.label[:40],
                  iteration=iteration, conf_before=conf_before, conf_after=new_conf)

        if signal.confidence >= threshold:
            trace.terminated_reason = "threshold_met"
            break
    else:
        trace.terminated_reason = "max_iterations"

    if trace.terminated_reason == "inconclusive" or signal.confidence < threshold:
        signal = signal.model_copy(update={"status": "inconclusive",
                                           "inconclusive_reason": f"Confidence {signal.confidence:.2f} below threshold {threshold}"})

    trace.final_confidence = signal.confidence
    return signal, trace


def _parse_response(raw: str) -> tuple[list[str], float]:
    import re
    ids_m = re.search(r"NEW_SUPPORTING_IDS:\s*(.+)", raw)
    conf_m = re.search(r"UPDATED_CONFIDENCE:\s*([\d.]+)", raw)

    raw_ids = ids_m.group(1).strip() if ids_m else ""
    new_ids = [] if not raw_ids or raw_ids.lower() == "none" else [i.strip() for i in raw_ids.split(",")]

    try:
        conf = float(conf_m.group(1)) if conf_m else 0.5
    except ValueError:
        conf = 0.5

    return new_ids, min(max(conf, 0.0), 1.0)
