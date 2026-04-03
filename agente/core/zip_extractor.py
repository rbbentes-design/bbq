"""
MacroDesk Bloomberg Ecosystem — ZIP Extractor
==============================================

Responsabilidade única:
  Abrir um arquivo .zip do Bloomberg, extrair os CSVs internos,
  renomear cada um com timestamp para evitar sobrescrita,
  salvar em bql_data/ e registrar em extracted_file_registry.

Convenção de nomeação obrigatória:
  {nome_original}_{YYYYMMDD}_{HHMMSS}_{hashcurto}.csv

  Exemplos:
    prices_20260403_143015_a3f7.csv
    fundamentals_20260403_143017_b8c2.csv

Regras:
  - Nunca sobrescrever arquivo existente
  - Nunca extrair arquivo fora da pasta de destino (path traversal)
  - Registrar cada arquivo em extracted_file_registry
  - Calcular hash SHA-256 do CSV extraído para trilha de auditoria

Uso:
    extractor = ZipExtractor(bql_data_dir, db_path)
    extracted = extractor.extract(zip_info, run_id="run-001")
    for ef in extracted:
        print(ef.renamed_csv_name)
"""

from __future__ import annotations

import hashlib
import sqlite3
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.logger import get_logger
from core.zip_scanner import ZipInfo

_log = get_logger("zip_extractor")


@dataclass
class ExtractedFile:
    """Informações sobre um CSV extraído e renomeado."""
    original_csv_name: str
    renamed_csv_name:  str
    source_zip_name:   str
    saved_path:        Path
    extracted_at:      str   # ISO-8601
    file_hash:         str   # sha256 do CSV


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _short_hash(data: bytes) -> str:
    """4 caracteres hex do SHA-256 — suficiente para unicidade dentro do segundo."""
    return hashlib.sha256(data).hexdigest()[:4]


class ZipExtractor:
    """
    Extrai CSVs de um zip Bloomberg, renomeia com timestamp e salva em bql_data.

    Args:
        bql_data_dir: Pasta de destino dos CSVs (normalmente .../agente/bql_data/).
        db_path:      Caminho para macrodesk.db (para registrar os arquivos).
    """

    def __init__(self, bql_data_dir: Path, db_path: Path) -> None:
        self._bql_data = bql_data_dir
        self._db_path  = db_path
        self._bql_data.mkdir(parents=True, exist_ok=True)

    # ── API Pública ───────────────────────────────────────────────────────────

    def extract(self, zi: ZipInfo, run_id: str) -> list[ExtractedFile]:
        """
        Extrai todos os CSVs do zip, renomeia e salva.

        Args:
            zi:     ZipInfo com path e metadados do zip.
            run_id: ID da execução atual (para rastreabilidade).

        Returns:
            Lista de ExtractedFile para cada CSV extraído com sucesso.
        """
        extracted: list[ExtractedFile] = []

        if not zi.path.exists():
            _log.error("zip_not_found", zip_path=str(zi.path))
            return extracted

        try:
            with zipfile.ZipFile(zi.path, "r") as zf:
                members = zf.namelist()
                csv_members = [
                    m for m in members
                    if Path(m).suffix.lower() == ".csv"
                    and len(Path(m).parts) <= 2  # evita subdirs profundos
                ]

                _log.info("zip_opened", zip_name=zi.zip_name, total_members=len(members), csv_count=len(csv_members))

                for member in csv_members:
                    try:
                        ef = self._extract_member(zf, member, zi, run_id)
                        if ef:
                            extracted.append(ef)
                    except Exception as exc:
                        _log.error("member_extract_failed", member=member, error=str(exc))

        except zipfile.BadZipFile as exc:
            _log.error("bad_zip_file", zip_name=zi.zip_name, error=str(exc))
        except Exception as exc:
            _log.error("zip_extract_error", zip_name=zi.zip_name, error=str(exc))

        _log.info("zip_extracted", zip_name=zi.zip_name, csvs_extracted=len(extracted))
        return extracted

    # ── Privados ──────────────────────────────────────────────────────────────

    def _extract_member(
        self,
        zf: zipfile.ZipFile,
        member: str,
        zi: ZipInfo,
        run_id: str,
    ) -> ExtractedFile | None:
        """Extrai um único membro CSV e registra no banco."""

        original_name = Path(member).name   # descarta qualquer subdir do zip
        data          = zf.read(member)
        file_hash     = _sha256_bytes(data)
        now           = datetime.now(timezone.utc)
        ts_str        = now.strftime("%Y%m%d_%H%M%S")
        short         = _short_hash(data)

        # Monta nome renomeado: nome_sem_ext_{YYYYMMDD}_{HHMMSS}_{hashcurto}.csv
        stem    = Path(original_name).stem   # ex: "prices"
        renamed = f"{stem}_{ts_str}_{short}.csv"

        dest = self._bql_data / renamed

        # Nunca sobrescrever — se (por algum acidente de milissegundo) já existir,
        # adiciona sufixo incremental
        if dest.exists():
            for i in range(1, 100):
                dest = self._bql_data / f"{stem}_{ts_str}_{short}_{i}.csv"
                renamed = dest.name
                if not dest.exists():
                    break

        # Segurança: verifica que o destino final está dentro de bql_data
        try:
            dest.resolve().relative_to(self._bql_data.resolve())
        except ValueError:
            _log.error(
                "path_traversal_blocked",
                original=original_name,
                dest=str(dest),
            )
            return None

        # Grava o arquivo
        dest.write_bytes(data)
        extracted_at = now.isoformat(timespec="seconds")

        ef = ExtractedFile(
            original_csv_name = original_name,
            renamed_csv_name  = renamed,
            source_zip_name   = zi.zip_name,
            saved_path        = dest,
            extracted_at      = extracted_at,
            file_hash         = file_hash,
        )

        self._register_file(ef)
        _log.info(
            "csv_extracted",
            original=original_name,
            renamed=renamed,
            size_kb=round(len(data) / 1024, 1),
        )
        return ef

    def _register_file(self, ef: ExtractedFile) -> None:
        """Insere o arquivo em extracted_file_registry."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO extracted_file_registry
                        (original_csv_name, renamed_csv_name, source_zip_name,
                         saved_path, extracted_at, file_hash, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'ok')
                    """,
                    (
                        ef.original_csv_name,
                        ef.renamed_csv_name,
                        ef.source_zip_name,
                        str(ef.saved_path),
                        ef.extracted_at,
                        ef.file_hash,
                    ),
                )
                conn.commit()
        except Exception as exc:
            _log.warning("register_file_failed", file=ef.renamed_csv_name, error=str(exc))
