"""Tests para app.storage (raw, normalized, bundle)."""
from __future__ import annotations

from datetime import date

import pytest

from app.models.market_ear_block import MarketEarBlock
from app.models.x_timeline_item import XTimelineItem
from app.storage.raw_store import RawStore
from app.storage.normalized_store import NormalizedStore
from app.storage.bundle_store import BundleStore
from app.utils.timestamps import new_ulid


# ── RawStore ──────────────────────────────────────────────────────────────────


def test_raw_store_save_and_load_html(tmp_workspace):
    store = RawStore()
    run_id = new_ulid()
    html = "<html><body>Test</body></html>"
    store.save_html("zerohedge", run_id, html)
    loaded = store.load_html("zerohedge", run_id)
    assert loaded == html


def test_raw_store_build_document(tmp_workspace):
    store = RawStore()
    html = "<html><body>ZH content</body></html>"
    doc, html_path = store.build_document(
        source_name="zerohedge",
        source_url="https://www.zerohedge.com/the-market-ear",
        access_method="playwright",
        html=html,
    )
    assert html_path.exists()
    assert doc.source_name == "zerohedge"
    assert len(doc.content_hash) == 64  # sha256 hex
    assert doc.raw_content_path == str(html_path)


def test_raw_store_load_document(tmp_workspace):
    store = RawStore()
    html = "<html>test</html>"
    doc, _ = store.build_document(
        source_name="x",
        source_url="https://x.com/home",
        access_method="playwright",
        html=html,
    )
    loaded = store.load_document("x", doc.id)
    assert loaded.id == doc.id
    assert loaded.source_url == "https://x.com/home"


def test_raw_store_content_hash_consistent(tmp_workspace):
    store = RawStore()
    html = "<html>same content</html>"
    doc1, _ = store.build_document("zh", "https://zh.com", "playwright", html)
    doc2, _ = store.build_document("zh", "https://zh.com", "playwright", html)
    assert doc1.content_hash == doc2.content_hash


# ── NormalizedStore ───────────────────────────────────────────────────────────


def test_normalized_store_write_and_read_all(tmp_workspace, sample_zh_block):
    store = NormalizedStore()
    run_id = new_ulid()
    store.write_all("zerohedge", run_id, [sample_zh_block])
    loaded = store.read_all("zerohedge", run_id, MarketEarBlock)
    assert len(loaded) == 1
    assert loaded[0].id == sample_zh_block.id
    assert loaded[0].title == sample_zh_block.title


def test_normalized_store_append(tmp_workspace, sample_zh_block):
    store = NormalizedStore()
    run_id = new_ulid()
    store.append("zerohedge", run_id, sample_zh_block)
    store.append("zerohedge", run_id, sample_zh_block)
    loaded = store.read_all("zerohedge", run_id, MarketEarBlock)
    assert len(loaded) == 2


def test_normalized_store_empty_returns_list(tmp_workspace):
    store = NormalizedStore()
    result = store.read_all("zerohedge", new_ulid(), MarketEarBlock)
    assert result == []


def test_normalized_store_x_items(tmp_workspace, sample_x_item):
    store = NormalizedStore()
    run_id = new_ulid()
    store.write_all("x", run_id, [sample_x_item])
    loaded = store.read_all("x", run_id, XTimelineItem)
    assert len(loaded) == 1
    assert loaded[0].author == sample_x_item.author
    assert loaded[0].engagement_info.likes == 200


# ── BundleStore ───────────────────────────────────────────────────────────────


def test_bundle_store_save_and_load(tmp_workspace, sample_bundle):
    store = BundleStore()
    store.save(sample_bundle)
    loaded = store.load(sample_bundle.run_date, sample_bundle.run_id)
    assert loaded.run_id == sample_bundle.run_id
    assert len(loaded.market_ear_blocks) == 1
    assert len(loaded.x_items) == 1


def test_bundle_store_list_bundles(tmp_workspace, sample_bundle):
    store = BundleStore()
    store.save(sample_bundle)
    bundles = store.list_bundles()
    assert len(bundles) == 1
    assert sample_bundle.run_id in bundles[0].stem


def test_bundle_store_list_empty(tmp_workspace):
    store = BundleStore()
    assert store.list_bundles() == []


def test_bundle_store_roundtrip_preserves_data(tmp_workspace, sample_bundle):
    store = BundleStore()
    store.save(sample_bundle)
    loaded = store.load(sample_bundle.run_date, sample_bundle.run_id)
    assert loaded.audit_summary.errors == 0
    assert loaded.audit_summary.total_records == 2
    assert loaded.x_items[0].engagement_info.likes == 200
