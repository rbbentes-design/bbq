"""Tests para app.views.report."""
from __future__ import annotations

import json

import pytest

from app.views.report import generate_markdown, generate_json_summary, save_reports


# ── generate_markdown ─────────────────────────────────────────────────────────


def test_markdown_contains_run_date(sample_bundle):
    md = generate_markdown(sample_bundle)
    assert "2026-03-22" in md


def test_markdown_contains_zh_block_title(sample_bundle):
    md = generate_markdown(sample_bundle)
    assert "Hello Stagflation" in md


def test_markdown_contains_x_author(sample_bundle):
    md = generate_markdown(sample_bundle)
    assert "@zerohedge" in md


def test_markdown_contains_x_text(sample_bundle):
    md = generate_markdown(sample_bundle)
    assert "Gold hits all-time high" in md


def test_markdown_shows_image_count(sample_bundle):
    md = generate_markdown(sample_bundle)
    assert "2 imagem" in md


def test_markdown_no_errors_section_when_clean(sample_bundle):
    md = generate_markdown(sample_bundle)
    assert "## Erros" not in md


def test_markdown_errors_section_when_errors(sample_bundle):
    from app.models.daily_ingestion_bundle import AuditSummary
    bundle = sample_bundle.model_copy(update={
        "audit_summary": AuditSummary(
            total_records=2, errors=1, error_messages=["ZH failed"]
        )
    })
    md = generate_markdown(bundle)
    assert "## Erros" in md
    assert "ZH failed" in md


def test_markdown_empty_zh_blocks(sample_bundle):
    bundle = sample_bundle.model_copy(update={"market_ear_blocks": []})
    md = generate_markdown(bundle)
    assert "Nenhum bloco coletado" in md


def test_markdown_empty_x_items(sample_bundle):
    bundle = sample_bundle.model_copy(update={"x_items": []})
    md = generate_markdown(bundle)
    assert "Nenhum tweet coletado" in md


def test_markdown_x_sorted_by_likes(sample_bundle):
    from datetime import timezone, datetime
    from app.models.x_timeline_item import EngagementInfo, XTimelineItem
    from app.utils.timestamps import new_ulid

    item_low = XTimelineItem(
        id=new_ulid(), author="@low", text="low likes", url="https://x.com/l/status/1",
        engagement_info=EngagementInfo(likes=5),
        raw_source_document_id=new_ulid(),
    )
    item_high = XTimelineItem(
        id=new_ulid(), author="@high", text="high likes", url="https://x.com/h/status/2",
        engagement_info=EngagementInfo(likes=9999),
        raw_source_document_id=new_ulid(),
    )
    bundle = sample_bundle.model_copy(update={"x_items": [item_low, item_high]})
    md = generate_markdown(bundle)
    assert md.index("@high") < md.index("@low")


# ── generate_json_summary ─────────────────────────────────────────────────────


def test_json_summary_structure(sample_bundle):
    summary = generate_json_summary(sample_bundle)
    assert summary["run_id"] == sample_bundle.run_id
    assert summary["run_date"] == "2026-03-22"
    assert "stats" in summary
    assert summary["stats"]["zh_blocks"] == 1
    assert summary["stats"]["x_items"] == 1
    assert summary["stats"]["errors"] == 0


def test_json_summary_zh_headlines(sample_bundle):
    summary = generate_json_summary(sample_bundle)
    headlines = summary["zh_headlines"]
    assert len(headlines) == 1
    assert headlines[0]["title"] == "Hello Stagflation"
    assert headlines[0]["images"] == 2


def test_json_summary_x_top10(sample_bundle):
    summary = generate_json_summary(sample_bundle)
    top10 = summary["x_top10"]
    assert len(top10) == 1
    assert top10[0]["author"] == "@zerohedge"
    assert top10[0]["likes"] == 200


def test_json_summary_serializable(sample_bundle):
    summary = generate_json_summary(sample_bundle)
    # Must serialize without errors
    s = json.dumps(summary, ensure_ascii=False)
    assert len(s) > 10


# ── save_reports ──────────────────────────────────────────────────────────────


def test_save_reports_creates_files(tmp_workspace, sample_bundle):
    md_path, json_path = save_reports(sample_bundle)
    assert md_path.exists()
    assert json_path.exists()


def test_save_reports_md_content(tmp_workspace, sample_bundle):
    md_path, _ = save_reports(sample_bundle)
    content = md_path.read_text(encoding="utf-8")
    assert "Hello Stagflation" in content


def test_save_reports_json_valid(tmp_workspace, sample_bundle):
    _, json_path = save_reports(sample_bundle)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["run_id"] == sample_bundle.run_id
