"""
MacroDesk Bloomberg Ecosystem — ZIP Scanner
============================================

Responsabilidade única:
  Varrer a pasta Downloads em busca de arquivos .zip exportados pelo Bloomberg
  (padrão: bql_data_*.zip) e retornar apenas os que ainda não foram processados.

O controle de "já processado" é feito consultando processed_zip_registry no banco.
A chave de unicidade é (zip_name, zip_hash) — assim o mesmo nome de arquivo com
conteúdo diferente (zip sobreescrito) ainda será detectado como novo.

Uso:
    scanner = ZipScanner(downloads_dir, db_path)
    new_zips = scanner.find_new_zips()
    for zp in new_zips:
        print(zp.path, zp.zip_hash)
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from core.logger import get_logger

_log = get_logger("zip_scanner")


@dataclass
class ZipInfo:
    """Informações sobre um arquivo .zip descoberto em Downloads."""
    path:        Path
    zip_name:    str
    zip_hash:    str    # sha256 hexdigest
    zip_size:    int    # bytes
    detected_at: str    # ISO-8601 UTC


def _sha256(path: Path) -> str:
    """Calcula SHA-256 do arquivo em blocos de 64 KB."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class ZipScanner:
    """
    Detecta novos arquivos .zip Bloomberg em Downloads.

    Args:
        downloads_dir: Pasta onde o Bloomberg salva os exports.
        db_path:       Caminho para macrodesk.db.
        zip_pattern:   Glob pattern dos zips Bloomberg (default: bql_data_*.zip).
    """

    def __init__(
        self,
        downloads_dir: Path,
        db_path: Path,
        zip_pattern: str = "bql_data_*.zip",
    ) -> None:
        self._downloads = downloads_dir
        self._db_path   = db_path
        self._pattern   = zip_pattern

    # ── API Pública ───────────────────────────────────────────────────────────

    def find_new_zips(self) -> list[ZipInfo]:
        """
        Retorna lista de zips Bloomberg em Downloads que ainda não foram processados.
        Ordena por mtime (mais antigo primeiro) para ingestão cronológica.
        """
        candidates = sorted(
            self._downloads.glob(self._pattern),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            _log.info("no_zips_found", dir=str(self._downloads), pattern=self._pattern)
            return []

        _log.info("zips_found_in_downloads", count=len(candidates))

        already_processed = self._load_processed_hashes()
        new_zips: list[ZipInfo] = []

        for p in candidates:
            try:
                stat     = p.stat()
                zip_hash = _sha256(p)
                key      = (p.name, zip_hash)

                if key in already_processed:
                    _log.debug("zip_already_processed", zip_name=p.name)
                    continue

                zi = ZipInfo(
                    path        = p,
                    zip_name    = p.name,
                    zip_hash    = zip_hash,
                    zip_size    = stat.st_size,
                    detected_at = datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
                new_zips.append(zi)
                _log.info("new_zip_detected", zip_name=p.name, size_kb=round(stat.st_size / 1024, 1))

            except Exception as exc:
                _log.error("zip_scan_error", zip=str(p), error=str(exc))

        _log.info(
            "scan_complete",
            total_found=len(candidates),
            already_processed=len(candidates) - len(new_zips),
            new=len(new_zips),
        )
        return new_zips

    def register_zip_as_pending(self, zi: ZipInfo) -> int:
        """
        Insere o zip no processed_zip_registry com status 'pending'.
        Retorna o id do registro inserido.
        """
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO processed_zip_registry
                    (zip_name, zip_path, zip_hash, zip_size, detected_at, status)
                VALUES (?, ?, ?, ?, ?, 'pending')
                """,
                (zi.zip_name, str(zi.path), zi.zip_hash, zi.zip_size, zi.detected_at),
            )
            conn.commit()
            return cur.lastrowid or 0

    def mark_zip_done(self, zip_id: int, status: str = "ok", error: str | None = None) -> None:
        """Atualiza status do zip no registro (ok ou error)."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                UPDATE processed_zip_registry
                SET status = ?, processed_at = ?, error_message = ?
                WHERE id = ?
                """,
                (status, now, error, zip_id),
            )
            conn.commit()

    # ── Privados ──────────────────────────────────────────────────────────────

    def _load_processed_hashes(self) -> set[tuple[str, str]]:
        """Carrega (zip_name, zip_hash) de todos os zips com status = 'ok'."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT zip_name, zip_hash FROM processed_zip_registry WHERE status = 'ok'"
                ).fetchall()
            return {(r[0], r[1]) for r in rows}
        except Exception as exc:
            _log.warning("load_processed_hashes_failed", error=str(exc))
            return set()
