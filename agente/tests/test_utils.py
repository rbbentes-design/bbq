"""Tests para app.utils (hashing, timestamps, text)."""
from __future__ import annotations

import time

import pytest

from app.utils.hashing import sha256_of_str, sha256_of_bytes
from app.utils.text import normalize_whitespace, truncate, word_count, slugify, extract_urls
from app.utils.timestamps import utcnow, new_ulid


# ── hashing ───────────────────────────────────────────────────────────────────


def test_sha256_of_str_deterministic():
    assert sha256_of_str("hello") == sha256_of_str("hello")


def test_sha256_of_str_different():
    assert sha256_of_str("hello") != sha256_of_str("world")


def test_sha256_of_str_length():
    assert len(sha256_of_str("x")) == 64  # hex digest


def test_sha256_of_bytes_matches_str():
    assert sha256_of_bytes(b"hello") == sha256_of_str("hello")


# ── text ──────────────────────────────────────────────────────────────────────


def test_normalize_whitespace_basic():
    assert normalize_whitespace("  hello   world  ") == "hello world"


def test_normalize_whitespace_newlines():
    # normalize_whitespace preserva newlines simples mas colapsa tabs/spaces
    result = normalize_whitespace("a\n\nb\t c")
    assert "b c" in result  # tabs viram espaço
    assert "a" in result


def test_normalize_whitespace_empty():
    assert normalize_whitespace("") == ""


def test_truncate_short():
    assert truncate("hello", 10) == "hello"


def test_truncate_exact():
    assert truncate("hello", 5) == "hello"


def test_truncate_long():
    result = truncate("hello world", 5)
    assert result.endswith("...")
    assert len(result) <= 8  # 5 + "..."


def test_truncate_custom_suffix():
    result = truncate("hello world", 5, suffix="…")
    assert result.endswith("…")


def test_word_count_basic():
    assert word_count("one two three") == 3


def test_word_count_empty():
    assert word_count("") == 0


def test_word_count_extra_spaces():
    assert word_count("  a  b  ") == 2


def test_slugify_basic():
    assert slugify("Hello World") == "hello-world"


def test_slugify_special_chars():
    result = slugify("Foo & Bar! 2024")
    assert " " not in result
    assert "&" not in result
    assert "!" not in result


def test_slugify_empty():
    assert slugify("") == ""


def test_extract_urls_finds_http():
    urls = extract_urls("visit https://example.com for more")
    assert "https://example.com" in urls


def test_extract_urls_none():
    assert extract_urls("no urls here") == []


def test_extract_urls_multiple():
    text = "go to https://a.com and http://b.org"
    urls = extract_urls(text)
    assert len(urls) == 2


# ── timestamps ────────────────────────────────────────────────────────────────


def test_utcnow_returns_aware():
    from datetime import timezone
    dt = utcnow()
    assert dt.tzinfo is not None
    assert dt.tzinfo == timezone.utc


def test_new_ulid_is_string():
    uid = new_ulid()
    assert isinstance(uid, str)
    assert len(uid) == 26


def test_new_ulid_unique():
    assert new_ulid() != new_ulid()


def test_new_ulid_monotonic():
    a = new_ulid()
    time.sleep(0.01)
    b = new_ulid()
    assert b > a  # ULIDs are lexicographically sortable by time
