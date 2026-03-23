"""
Fixtures compartilhadas para todos os testes.

A fixture `tmp_workspace` redireciona todos os caminhos de storage
para um diretório temporário, isolando os testes do workspace real.
"""
from __future__ import annotations

from datetime import date, timezone, datetime
from pathlib import Path

import pytest

from app.models.market_ear_block import MarketEarBlock
from app.models.x_timeline_item import EngagementInfo, XTimelineItem
from app.utils.timestamps import new_ulid


@pytest.fixture()
def tmp_workspace(tmp_path, monkeypatch):
    """Redireciona workspace para diretório temporário isolado."""
    from app.config.settings import settings
    monkeypatch.setattr(settings, "workspace_dir", tmp_path)
    return tmp_path


@pytest.fixture()
def sample_zh_block() -> MarketEarBlock:
    return MarketEarBlock(
        id=new_ulid(),
        title="Hello Stagflation",
        body_text="Markets are pricing stagflation.",
        image_refs=["https://example.com/img1.jpg", "https://example.com/img2.jpg"],
        source_url="https://www.zerohedge.com/the-market-ear",
        raw_source_document_id=new_ulid(),
        position_index=0,
    )


@pytest.fixture()
def sample_x_item() -> XTimelineItem:
    return XTimelineItem(
        id=new_ulid(),
        author="@zerohedge",
        text="Gold hits all-time high as dollar weakens.",
        url="https://x.com/zerohedge/status/123456",
        created_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
        engagement_info=EngagementInfo(replies=10, reposts=50, likes=200),
        raw_source_document_id=new_ulid(),
    )


@pytest.fixture()
def sample_bundle(sample_zh_block, sample_x_item):
    from app.models.daily_ingestion_bundle import AuditSummary, DailyIngestionBundle
    from app.utils.timestamps import utcnow
    return DailyIngestionBundle(
        run_id=new_ulid(),
        run_date=date(2026, 3, 22),
        created_at=utcnow(),
        market_ear_blocks=[sample_zh_block],
        x_items=[sample_x_item],
        candidate_signals=[],
        audit_summary=AuditSummary(total_records=2, errors=0),
        artifact_paths={},
    )
