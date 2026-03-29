from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class NarrativeSignal(BaseModel):
    id: str
    label: str
    description: str
    confidence: float
    supporting_item_ids: list[str] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(default_factory=list)
    status: Literal["confirmed", "inconclusive", "discarded"] = "confirmed"
    inconclusive_reason: str | None = None


class Narrative(BaseModel):
    id: str
    run_id: str
    run_date: str
    primary_signal: NarrativeSignal
    secondary_signals: list[NarrativeSignal] = Field(default_factory=list)
    raw_llm_response: str = ""
    detection_model: str = "claude-sonnet-4-6"
    detected_at: datetime


class ScoredItem(BaseModel):
    item_id: str
    item_type: Literal["market_ear_block", "x_item"]
    narrative_relevance: float
    relevance_reason: str = ""
    signal_ids: list[str] = Field(default_factory=list)


class SignalVerification(BaseModel):
    signal_id: str
    verdict: Literal["confirmed", "weak", "hallucinated"]
    grounding_score: float
    notes: str = ""


class VerificationResult(BaseModel):
    run_id: str
    verified_at: datetime
    verification_model: str = "claude-haiku-4-5-20251001"
    signal_verifications: list[SignalVerification] = Field(default_factory=list)
    overall_verdict: Literal["pass", "warn", "partial", "fail"] = "pass"
    hallucination_flags: list[str] = Field(default_factory=list)


class EvidenceIteration(BaseModel):
    iteration: int
    query_used: str
    items_examined: int
    confidence_before: float
    confidence_after: float
    new_evidence_found: list[str] = Field(default_factory=list)


class EvidenceGatheringTrace(BaseModel):
    signal_id: str
    iterations: list[EvidenceIteration] = Field(default_factory=list)
    final_confidence: float = 0.0
    terminated_reason: Literal["threshold_met", "max_iterations", "inconclusive"] = "inconclusive"


class CurationResult(BaseModel):
    id: str
    run_id: str
    run_date: str
    narrative: Narrative
    scored_items: list[ScoredItem] = Field(default_factory=list)
    verification: VerificationResult
    evidence_traces: list[EvidenceGatheringTrace] = Field(default_factory=list)
    corrections_applied: int = 0
    curated_at: datetime
    artifact_paths: dict[str, Any] = Field(default_factory=dict)


# ── Corrections / Learning ────────────────────────────────────────────────────

class CorrectionEntry(BaseModel):
    date: str
    correction_type: Literal["wrong_narrative", "missed_signal", "wrong_score", "hallucinated_quote"]
    original_label: str | None = None
    corrected_label: str
    example_input: str
    example_output: str
    notes: str = ""


class CorrectionsFile(BaseModel):
    version: int = 1
    corrections: list[CorrectionEntry] = Field(default_factory=list)
