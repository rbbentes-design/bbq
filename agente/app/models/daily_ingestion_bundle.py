from __future__ import annotations

from datetime import date as Date
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.audit_record import AuditRecord
from app.models.market_ear_block import MarketEarBlock
from app.models.signal_candidate import SignalCandidate
from app.models.spotgamma_report import SpotGammaReport
from app.models.x_timeline_item import XTimelineItem
from app.models.rss_item import RSSItem


# ── Tipos para dados enrichment ───────────────────────────────────────────────
PolymarketMarket = dict[str, Any]
MarketPriceEntry = dict[str, Any]


class AuditSummary(BaseModel):
    total_records: int = 0
    ok: int = 0
    warnings: int = 0
    errors: int = 0
    error_messages: list[str] = Field(default_factory=list)


class DailyIngestionBundle(BaseModel):
    """
    Bundle diário completo — output final do pipeline de ingestão.
    Agrega todos os artefatos processados de uma execução.
    """

    run_id: str = Field(description="ID único da execução")
    run_date: Date = Field(description="Data de referência do bundle")
    created_at: datetime = Field(description="Timestamp UTC de criação do bundle")
    market_ear_blocks: list[MarketEarBlock] = Field(default_factory=list)
    x_items: list[XTimelineItem] = Field(default_factory=list)
    rss_items: list[RSSItem] = Field(default_factory=list)
    spotgamma_reports: list[SpotGammaReport] = Field(default_factory=list)
    candidate_signals: list[SignalCandidate] = Field(default_factory=list)
    # ── Enrichment (providers adicionais) ─────────────────────────────────────
    polymarket_markets: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Mercados de predição macro do Polymarket",
    )
    fred_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Séries macro FRED + agenda econômica da semana",
    )
    damodaran_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Dados Damodaran: ERP implícito, country risk premiums, WACC por setor",
    )
    global_liquidity: dict[str, Any] = Field(
        default_factory=dict,
        description="Liquidez global: Net Fed Liquidity, MMF, ECB BS, M2 G4, balanços BCx",
    )
    market_prices: dict[str, Any] = Field(
        default_factory=dict,
        description="Preços e retornos de ativos-chave via yfinance",
    )
    swaggy_data: dict[str, Any] = Field(
        default_factory=dict,
        description="SwaggyStocks: WSB mentions + squeeze candidates (serializado como dict)",
    )
    audit_summary: AuditSummary = Field(default_factory=AuditSummary)
    artifact_paths: dict[str, str] = Field(
        default_factory=dict,
        description="Caminhos dos artefatos gerados: {'bundle': '...', 'markdown': '...', 'json': '...'}",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
