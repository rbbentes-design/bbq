"""Tests para os modelos Pydantic."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from app.models.daily_ingestion_bundle import AuditSummary, DailyIngestionBundle
from app.models.market_ear_block import MarketEarBlock
from app.models.x_timeline_item import EngagementInfo, XTimelineItem
from app.utils.timestamps import new_ulid, utcnow


# ── MarketEarBlock ────────────────────────────────────────────────────────────


def test_market_ear_block_defaults():
    block = MarketEarBlock(
        id=new_ulid(),
        source_url="https://example.com",
        raw_source_document_id=new_ulid(),
        position_index=0,
    )
    assert block.title == ""
    assert block.body_text == ""
    assert block.image_refs == []
    assert block.relevance_score_preliminary == 0.0


def test_market_ear_block_relevance_bounds():
    with pytest.raises(Exception):
        MarketEarBlock(
            id=new_ulid(),
            source_url="https://example.com",
            raw_source_document_id=new_ulid(),
            position_index=0,
            relevance_score_preliminary=1.5,  # > 1.0
        )


def test_market_ear_block_serialization(sample_zh_block):
    json_str = sample_zh_block.model_dump_json()
    restored = MarketEarBlock.model_validate_json(json_str)
    assert restored.id == sample_zh_block.id
    assert restored.title == sample_zh_block.title


# ── XTimelineItem ─────────────────────────────────────────────────────────────


def test_x_timeline_item_defaults():
    item = XTimelineItem(
        id=new_ulid(),
        author="@test",
        text="hello",
        url="https://x.com/test/status/1",
        raw_source_document_id=new_ulid(),
    )
    assert item.created_at is None
    assert item.engagement_info.likes == 0
    assert item.media_refs == []


def test_engagement_info_parse():
    eng = EngagementInfo(replies=5, reposts=10, likes=100)
    assert eng.likes == 100


def test_x_item_serialization(sample_x_item):
    json_str = sample_x_item.model_dump_json()
    restored = XTimelineItem.model_validate_json(json_str)
    assert restored.id == sample_x_item.id
    assert restored.engagement_info.likes == 200


# ── AuditSummary ──────────────────────────────────────────────────────────────


def test_audit_summary_defaults():
    s = AuditSummary(total_records=10, errors=2)
    assert s.error_messages == []


def test_audit_summary_with_messages():
    s = AuditSummary(total_records=5, errors=1, error_messages=["ZH failed"])
    assert len(s.error_messages) == 1


# ── DailyIngestionBundle ──────────────────────────────────────────────────────


def test_bundle_defaults(sample_bundle):
    assert sample_bundle.candidate_signals == []
    assert sample_bundle.artifact_paths == {}


def test_bundle_serialization(sample_bundle):
    json_str = sample_bundle.model_dump_json()
    restored = DailyIngestionBundle.model_validate_json(json_str)
    assert restored.run_id == sample_bundle.run_id
    assert len(restored.market_ear_blocks) == 1
    assert len(restored.x_items) == 1
    assert restored.audit_summary.total_records == 2


def test_bundle_model_copy_update(sample_bundle):
    updated = sample_bundle.model_copy(update={"artifact_paths": {"bundle": "/tmp/x.json"}})
    assert updated.artifact_paths["bundle"] == "/tmp/x.json"
    assert sample_bundle.artifact_paths == {}  # original unchanged
