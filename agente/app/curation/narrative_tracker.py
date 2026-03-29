"""
Rastreamento de narrativas ao longo do tempo.

Compara a narrativa do dia com os ultimos 7 dias e detecta:
- Persistencia: mesma narrativa continua
- Evolucao: narrativa similar mas com nova informacao
- Inversao: narrativa oposta emergiu
- Nova: tema completamente diferente
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from app.storage.paths import workspace


@dataclass
class DayEntry:
    date: str
    label: str
    confidence: float
    verdict: str  # pass | warn | fail
    secondary_label: str = ""


@dataclass
class NarrativeTrend:
    entries: list[DayEntry] = field(default_factory=list)   # oldest first
    trend: Literal["persisting", "evolving", "reversed", "new", "unknown"] = "unknown"
    trend_note: str = ""

    @property
    def today(self) -> DayEntry | None:
        return self.entries[-1] if self.entries else None

    @property
    def history(self) -> list[DayEntry]:
        return self.entries[:-1]


def load_trend(days: int = 7) -> NarrativeTrend:
    """Carrega resultados de curação dos últimos N dias e calcula tendência."""
    today = date.today()
    entries: list[DayEntry] = []

    for delta in range(days - 1, -1, -1):   # oldest → newest
        d = today - timedelta(days=delta)
        entry = _load_day(d)
        if entry:
            entries.append(entry)

    trend = NarrativeTrend(entries=entries)
    if len(entries) >= 2:
        _compute_trend(trend)
    return trend


def _load_day(d: date) -> DayEntry | None:
    folder = workspace.bundles / d.isoformat()
    if not folder.exists():
        return None

    # Pega o curation mais recente do dia
    curation_files = sorted(folder.glob("*_curation.json"), reverse=True)
    if not curation_files:
        return None

    try:
        data = json.loads(curation_files[0].read_text(encoding="utf-8"))
        primary = data["narrative"]["primary_signal"]
        secondary = data["narrative"].get("secondary_signals", [])
        verdict = data["verification"]["overall_verdict"]
        return DayEntry(
            date=d.isoformat(),
            label=primary["label"],
            confidence=primary["confidence"],
            verdict=verdict,
            secondary_label=secondary[0]["label"] if secondary else "",
        )
    except Exception:
        return None


def _compute_trend(trend: NarrativeTrend) -> None:
    if not trend.today or not trend.history:
        return

    today_words = set(trend.today.label.lower().split())
    recent = trend.history[-1]
    recent_words = set(recent.label.lower().split())

    # Remover stopwords
    stop = {"the", "a", "an", "and", "or", "of", "in", "to", "is", "are",
            "signals", "drive", "drives", "on", "as", "with", "for", "by"}
    today_kw = today_words - stop
    recent_kw = recent_words - stop

    overlap = len(today_kw & recent_kw) / max(len(today_kw | recent_kw), 1)

    # Detectar inversão: palavras opostas
    bullish = {"rally", "surge", "gain", "rise", "buy", "long", "squeeze"}
    bearish = {"plunge", "drop", "fall", "sell", "short", "crash", "decline"}

    today_bull = bool(today_kw & bullish)
    today_bear = bool(today_kw & bearish)
    recent_bull = bool(recent_kw & bullish)
    recent_bear = bool(recent_kw & bearish)
    inverted = (today_bull and recent_bear) or (today_bear and recent_bull)

    if inverted and overlap < 0.4:
        trend.trend = "reversed"
        trend.trend_note = f"Inversão: '{recent.label[:40]}' → '{trend.today.label[:40]}'"
    elif overlap >= 0.5:
        trend.trend = "persisting"
        trend.trend_note = f"Narrativa persistindo há {len(trend.entries)} dias"
    elif overlap >= 0.2:
        trend.trend = "evolving"
        trend.trend_note = f"Evoluindo de '{recent.label[:40]}'"
    else:
        trend.trend = "new"
        trend.trend_note = "Nova narrativa — tema diferente de ontem"
