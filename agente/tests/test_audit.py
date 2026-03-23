"""Tests para app.audit (records, logger)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.audit import records as rec
from app.audit.logger import AuditLogger


# ── records ───────────────────────────────────────────────────────────────────


def test_ok_record_structure():
    r = rec.ok("run1", "pipeline", "start")
    assert r.status == "ok"
    assert r.run_id == "run1"
    assert r.stage == "pipeline"
    assert r.action == "start"
    assert r.timestamp is not None


def test_error_record_structure():
    r = rec.error("run1", "collect", "zh_failed", msg="timeout")
    assert r.status == "error"
    assert r.error_message == "timeout"


def test_warning_record_structure():
    r = rec.warning("run1", "parse", "empty_block", msg="empty")
    assert r.status == "warning"
    assert r.error_message == "empty"


def test_skipped_record_structure():
    r = rec.skipped("run1", "collect", "x_skipped", msg="no data")
    assert r.status == "skipped"


def test_ok_with_extra_kwargs():
    r = rec.ok("run1", "parse", "done", blocks=15)
    assert r.metadata["blocks"] == 15


def test_record_ts_is_iso():
    r = rec.ok("r", "s", "e")
    # timestamp deve ser datetime aware
    from datetime import timezone
    assert r.timestamp.tzinfo is not None


# ── AuditLogger ───────────────────────────────────────────────────────────────


def test_audit_logger_writes_jsonl(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path)
    logger.write(rec.ok("r1", "pipeline", "start"))
    logger.write(rec.ok("r1", "pipeline", "done"))

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert obj["run_id"] == "r1"


def test_audit_logger_creates_parent_dir(tmp_path):
    log_path = tmp_path / "nested" / "dir" / "audit.jsonl"
    logger = AuditLogger(log_path)
    logger.write(rec.ok("r", "s", "e"))
    assert log_path.exists()


def test_audit_logger_appends(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    AuditLogger(log_path).write(rec.ok("r", "s", "e1"))
    AuditLogger(log_path).write(rec.ok("r", "s", "e2"))
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_audit_logger_redacts_sensitive(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path)
    logger.write(rec.ok("r", "s", "e", password="secret", token="tok123"))
    line = json.loads(log_path.read_text().strip())
    # Sensitive keys in metadata should be redacted
    meta = line.get("metadata", {})
    assert meta.get("password") != "secret"
    assert meta.get("token") != "tok123"
