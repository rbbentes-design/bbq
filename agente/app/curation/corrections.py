from __future__ import annotations

from typing import Literal

import yaml

from app.audit.logger import get_logger
from app.curation.models import CorrectionEntry, CorrectionsFile
from app.storage.paths import workspace

_log = get_logger("curation.corrections")


def load_corrections() -> CorrectionsFile:
    path = workspace.corrections_path()
    if not path.exists():
        return CorrectionsFile()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return CorrectionsFile.model_validate(data)
    except Exception as exc:
        _log.warning("corrections_load_error", error=str(exc))
        return CorrectionsFile()


def get_few_shot_examples(
    corrections: CorrectionsFile,
    correction_type: Literal["wrong_narrative", "missed_signal", "wrong_score", "hallucinated_quote"],
    max_examples: int = 3,
) -> list[CorrectionEntry]:
    filtered = [c for c in corrections.corrections if c.correction_type == correction_type]
    filtered.sort(key=lambda c: c.date, reverse=True)
    return filtered[:max_examples]


def format_few_shot_block(examples: list[CorrectionEntry]) -> str:
    if not examples:
        return ""
    lines = ["PAST CORRECTIONS (learn from these mistakes):\n"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i} ({ex.date}):")
        lines.append(f"INPUT:\n{ex.example_input.strip()}")
        lines.append(f"CORRECT OUTPUT:\n{ex.example_output.strip()}")
        if ex.notes:
            lines.append(f"NOTE: {ex.notes}")
        lines.append("")
    return "\n".join(lines)


def create_corrections_template(path_hint: str = "") -> str:
    """Returns a YAML template string to bootstrap the corrections file."""
    return """\
version: 1
corrections: []

# Como usar:
# Adicione entradas abaixo apos revisar um relatorio.
# O sistema injeta as correcoes como exemplos no proximo run.
#
# correction_type pode ser:
#   wrong_narrative   - narrativa detectada estava errada
#   missed_signal     - sinal real nao foi detectado
#   wrong_score       - item pontuado incorretamente
#   hallucinated_quote - citacao inventada pelo modelo
#
# Exemplo:
# - date: "2026-03-23"
#   correction_type: wrong_narrative
#   original_label: "Tech earnings optimism"
#   corrected_label: "Credit stress bleeding into equities"
#   example_input: |
#     MarketEar: "IG spreads at 3-month wides..."
#     X @credittrader: "This isn't equity vol — credit contagion."
#   example_output: |
#     PRIMARY NARRATIVE: Credit stress bleeding into equities
#     CONFIDENCE: 0.87
#     REASONING: Three independent sources point to credit spreads as causal driver.
"""
