from __future__ import annotations

from datetime import date as Date
from datetime import datetime

from pydantic import BaseModel, Field

from app.utils.timestamps import new_ulid


class ReportSection(BaseModel):
    heading: str = ""
    body_text: str = ""
    image_refs: list[str] = Field(default_factory=list)


class SpotGammaReport(BaseModel):
    """
    Relatório SpotGamma coletado via Playwright (requer autenticação).
    Tipicamente o FlowPatrol diário com dados de fluxo de opções.
    """

    id: str = Field(default_factory=new_ulid)
    report_type: str = ""          # ex: "FlowPatrol", "FoundersNotes"
    report_date: Date | None = None
    title: str = ""
    source_url: str = ""
    sections: list[ReportSection] = Field(default_factory=list)
    image_refs: list[str] = Field(default_factory=list)
    raw_text: str = ""             # texto completo concatenado para curation
    collected_at: datetime | None = None
