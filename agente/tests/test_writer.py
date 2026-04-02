"""
Teste do writer com dados mock realistas.
Roda: python -m tests.test_writer
"""

from datetime import datetime, timezone

from app.curation.models import (
    CurationResult,
    EvidenceGatheringTrace,
    Narrative,
    NarrativeSignal,
    VerificationResult,
    SignalVerification,
)
from app.curation.writer import write
from app.utils.timestamps import new_ulid


def _mock_result() -> CurationResult:
    run_id = new_ulid()
    run_date = "2026-03-24"

    primary = NarrativeSignal(
        id=new_ulid(),
        label="Fed credibilidade em xeque — mercado questiona trajetória de cortes",
        description=(
            "Os dados de CPI acima do esperado, combinados com falas hawkish de membros do Fed, "
            "estão forçando uma reprecificação agressiva da curva. O mercado que precificava "
            "três cortes em 2026 agora vê menos de um. A ironia é que o próprio Fed criou a "
            "expectativa que agora está destruindo — e o custo de capital está sendo recalibrado "
            "em tempo real pelos dealers de Treasuries."
        ),
        confidence=0.87,
        supporting_item_ids=["ME-01", "X-01", "RSS-01"],
        evidence_quotes=[
            "Fed's inflation expectations becoming unanchored, real yields repricing fast",
            "Dealers are short gamma on the 10y — any move gets amplified, not absorbed",
            "The CPI print wasn't a surprise in magnitude, it was a surprise in composition: services sticky, goods bouncing back",
            "Cut pricing collapsed from 3x to 0.8x this week — that's not an adjustment, that's a regime change",
        ],
        status="confirmed",
    )

    secondary = NarrativeSignal(
        id=new_ulid(),
        label="Crédito high yield começa a separar — spreads HY vs IG divergindo",
        description=(
            "Com o custo de capital subindo, o crédito mais alavancado começa a mostrar stress. "
            "Spreads de HY abrindo enquanto IG permanece comprimido. Pode ser o canário — "
            "ou pode ser seletividade técnica de fim de trimestre."
        ),
        confidence=0.61,
        supporting_item_ids=["RSS-02", "X-02"],
        evidence_quotes=[
            "HY spreads +40bps ytd, IG flat — divergence not seen since Q4 2022",
        ],
        status="confirmed",
    )

    narrative = Narrative(
        id=new_ulid(),
        run_id=run_id,
        run_date=run_date,
        primary_signal=primary,
        secondary_signals=[secondary],
        detection_model="claude-sonnet-4-6",
        detected_at=datetime.now(timezone.utc),
    )

    verification = VerificationResult(
        run_id=run_id,
        verified_at=datetime.now(timezone.utc),
        signal_verifications=[
            SignalVerification(
                signal_id=primary.id,
                verdict="confirmed",
                grounding_score=0.91,
                notes="Todas as quotes verificadas no corpus",
            )
        ],
        overall_verdict="pass",
        hallucination_flags=[],
    )

    return CurationResult(
        id=new_ulid(),
        run_id=run_id,
        run_date=run_date,
        narrative=narrative,
        scored_items=[],
        verification=verification,
        evidence_traces=[],
        corrections_applied=0,
        curated_at=datetime.now(timezone.utc),
    )


if __name__ == "__main__":
    print("Construindo mock CurationResult...")
    result = _mock_result()
    print(f"Narrativa: {result.narrative.primary_signal.label}")
    print(f"Confiança: {result.narrative.primary_signal.confidence:.0%}")
    print("\nRodando writer (2 turnos LLM)...\n")
    print("=" * 70)

    output = write(result)

    print(f"\n[MODO ESCOLHIDO]: {output.mode}")
    print(f"[RACIONAL]: {output.rationale}")
    print(f"[FOCO]: {output.focus}")
    print("\n" + "=" * 70)
    print("\n" + output.text)
    print("\n" + "=" * 70)
