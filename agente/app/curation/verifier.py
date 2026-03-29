from __future__ import annotations

import re
from datetime import datetime, timezone

from app.audit.logger import get_logger
from app.curation.corpus import build_item_index
from app.curation.llm_client import call_claude
from app.curation.models import (
    Narrative,
    SignalVerification,
    VerificationResult,
)
from app.models.daily_ingestion_bundle import DailyIngestionBundle
from app.utils.timestamps import new_ulid

_log = get_logger("curation.verifier")
_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM = """\
You are a financial fact-checker. Verify whether each narrative signal is \
grounded in the provided corpus items.

For each signal, output EXACTLY:
SIGNAL_ID: <id>
VERDICT: confirmed | weak | hallucinated
GROUNDING_SCORE: <0.00-1.00>
NOTES: <1-2 sentences explaining your verdict>
---

Rules:
- "confirmed": the theme/claim is clearly supported by the corpus, even if not verbatim.
- "weak": signal is plausible and partially supported but evidence is thin or indirect.
- "hallucinated": the claim contradicts the corpus OR refers to events/assets not mentioned at all.
- Accept paraphrasing and thematic support — you are checking for factual grounding, not exact quotes.
- Secondary narratives are often subtler signals; apply a lower bar than primary narratives.
- Only use "hallucinated" when the corpus actively contradicts the signal or there is zero thematic connection."""


def verify_narrative(
    narrative: Narrative,
    bundle: DailyIngestionBundle,
    run_id: str,
) -> VerificationResult:
    item_index = build_item_index(bundle)

    signals_to_verify = [narrative.primary_signal] + [
        s for s in narrative.secondary_signals if s is not None
    ]

    # Build corpus snapshot: full text of the supporting items only
    all_supporting = set()
    for sig in signals_to_verify:
        all_supporting.update(sig.supporting_item_ids)

    corpus_items = []
    for sid in all_supporting:
        if sid in item_index:
            corpus_items.append(f"[{sid}] {_full_text(item_index[sid])}")
        else:
            # Try matching by prefix in the index
            for key in item_index:
                if key.startswith(sid[:8]) or sid.startswith(key[:8]):
                    corpus_items.append(f"[{sid}] {_full_text(item_index[key])}")
                    break

    # Fallback: include all short-ID items (truncated) if no supporting IDs
    if not corpus_items:
        corpus_items = [
            f"[{sid}] {_short_text(item_index[sid])}"
            for sid in item_index
            if sid.startswith(("ME-", "X-"))
        ]

    corpus_snippet = "\n".join(corpus_items)

    signal_blocks = []
    for i, sig in enumerate(signals_to_verify):
        quotes_str = "\n".join(f'  - "{q}"' for q in sig.evidence_quotes[:5])
        ids_str = ", ".join(sig.supporting_item_ids[:10])
        signal_blocks.append(
            f"SIGNAL: SIG{i}\n"
            f"LABEL: {sig.label}\n"
            f"SUPPORTING_IDS: {ids_str or 'none'}\n"
            f"EVIDENCE_QUOTES:\n{quotes_str or '  (none)'}"
        )

    user_prompt = (
        "CORPUS (ID + excerpt):\n"
        + corpus_snippet
        + "\n\nSIGNALS TO VERIFY:\n\n"
        + "\n\n".join(signal_blocks)
    )

    _log.info("verifying_narrative", n_signals=len(signals_to_verify))

    try:
        raw = call_claude(_SYSTEM, user_prompt, model=_MODEL, max_tokens=1024)
    except Exception as exc:
        _log.warning("verifier_error", error=str(exc))
        # Return a default pass-through result on error
        return _default_result(run_id, signals_to_verify)

    verifications = _parse_verifications(raw)
    # Map SIG0/SIG1 back to real signal labels
    sig_map = {f"SIG{i}": s.label for i, s in enumerate(signals_to_verify)}
    verifications = [
        v.model_copy(update={"signal_id": sig_map.get(v.signal_id, v.signal_id)})
        for v in verifications
    ]

    hallucination_flags = [
        v.signal_id for v in verifications if v.verdict == "hallucinated"
    ]

    # fail only if the PRIMARY signal is hallucinated
    primary_label = signals_to_verify[0].label if signals_to_verify else ""
    primary_hallucinated = any(
        v.verdict == "hallucinated" and v.signal_id == primary_label
        for v in verifications
    )

    overall = "pass"
    if primary_hallucinated:
        overall = "fail"
    elif hallucination_flags or any(v.verdict == "weak" for v in verifications):
        overall = "warn"

    _log.info("verification_done", verdict=overall, flags=hallucination_flags)

    return VerificationResult(
        run_id=run_id,
        verified_at=datetime.now(timezone.utc),
        verification_model=_MODEL,
        signal_verifications=verifications,
        overall_verdict=overall,
        hallucination_flags=hallucination_flags,
    )


def _short_text(item) -> str:
    from app.models.market_ear_block import MarketEarBlock
    if isinstance(item, MarketEarBlock):
        return item.body_text[:300].replace("\n", " ")
    return item.text[:300].replace("\n", " ")


def _full_text(item) -> str:
    from app.models.market_ear_block import MarketEarBlock
    if isinstance(item, MarketEarBlock):
        return item.body_text.replace("\n", " ")
    return item.text.replace("\n", " ")


def _parse_verifications(raw: str) -> list[SignalVerification]:
    results: list[SignalVerification] = []
    blocks = re.split(r"\n---\n?", raw)
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        sig_m = re.search(r"SIGNAL_ID:\s*(.+)", block)
        verdict_m = re.search(r"VERDICT:\s*(confirmed|weak|hallucinated)", block)
        score_m = re.search(r"GROUNDING_SCORE:\s*([\d.]+)", block)
        notes_m = re.search(r"NOTES:\s*(.+)", block, re.DOTALL)

        if not sig_m:
            continue

        try:
            score = float(score_m.group(1)) if score_m else 0.5
        except ValueError:
            score = 0.5

        results.append(SignalVerification(
            signal_id=sig_m.group(1).strip(),
            verdict=verdict_m.group(1) if verdict_m else "weak",
            grounding_score=min(max(score, 0.0), 1.0),
            notes=notes_m.group(1).strip() if notes_m else "",
        ))

    return results


def _default_result(run_id: str, signals) -> VerificationResult:
    verifications = [
        SignalVerification(
            signal_id=s.id,
            verdict="weak",
            grounding_score=0.5,
            notes="Verification skipped due to API error.",
        )
        for s in signals
    ]
    return VerificationResult(
        run_id=run_id,
        verified_at=datetime.now(timezone.utc),
        verification_model=_MODEL,
        signal_verifications=verifications,
        overall_verdict="warn",
        hallucination_flags=[],
    )
