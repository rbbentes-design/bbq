"""
MacroDesk Bloomberg Ecosystem — Database Writer
================================================

Responsabilidade única:
  Gravar registros normalizados no banco SQLite com as seguintes garantias:
  - Deduplicação: UNIQUE(ticker, field, date) — INSERT OR IGNORE em timeseries
  - Atualização de latest: UPSERT — sempre mantém o valor mais recente por (ticker, field)
  - NULL real: Python None → SQLite NULL (nunca string "None" ou "NaN")
  - Rastreabilidade: log de ausências em missing_data_log

Uso:
    writer = DbWriter(db_path)
    stats = writer.write_batch(run_id, ts_records, missing_records, source_file)
    print(stats.rows_inserted, stats.rows_skipped)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.logger import get_logger

_log = get_logger("db_writer")


@dataclass
class WriteStats:
    """Contadores de uma operação de escrita."""
    rows_inserted:  int = 0
    rows_skipped:   int = 0   # duplicatas ignoradas
    latest_updated: int = 0
    missing_logged: int = 0
    errors:         int = 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DbWriter:
    """
    Gravador de dados Bloomberg no banco SQLite.

    Args:
        db_path: Caminho para macrodesk.db.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    # ── API Pública ───────────────────────────────────────────────────────────

    def write_batch(
        self,
        run_id: str,
        ts_records: list[dict],
        missing_records: list[dict],
        source_file: str,
    ) -> WriteStats:
        """
        Grava um lote de registros de timeseries e ausências no banco.

        Args:
            run_id:          ID da execução atual.
            ts_records:      Lista de dicts no formato do DataNormalizer.
            missing_records: Lista de dicts de dados ausentes/inválidos.
            source_file:     Nome do CSV de origem (para log).

        Returns:
            WriteStats com contagem de inserções, duplicatas e erros.
        """
        stats = WriteStats()

        with sqlite3.connect(self._db_path) as conn:
            # Grava timeseries
            _log.info("writing_timeseries", source_file=source_file, records=len(ts_records))
            for rec in ts_records:
                try:
                    self._write_timeseries_record(conn, rec, stats)
                except Exception as exc:
                    stats.errors += 1
                    _log.error("timeseries_write_error", rec=rec, error=str(exc))

            # Atualiza bql_latest para cada registro inserido com sucesso
            try:
                self._refresh_latest_from_timeseries(conn, source_file, stats)
            except Exception as exc:
                stats.errors += 1
                _log.error("latest_refresh_error", error=str(exc))

            # Grava missing_data_log
            if missing_records:
                try:
                    self._write_missing_records(conn, run_id, missing_records, stats)
                except Exception as exc:
                    stats.errors += 1
                    _log.error("missing_log_write_error", error=str(exc))

            conn.commit()

        _log.info(
            "batch_written",
            source_file=source_file,
            inserted=stats.rows_inserted,
            skipped=stats.rows_skipped,
            latest_updated=stats.latest_updated,
            missing_logged=stats.missing_logged,
            errors=stats.errors,
        )
        return stats

    def write_macro_series(
        self,
        run_id: str,
        snapshot_records: list[dict],
        history_records: list[dict],
    ) -> tuple[int, int]:
        """
        Grava séries macro no banco.

        Args:
            run_id:           ID da execução.
            snapshot_records: Lista de {bbg_ticker, description, category, px_last, date}.
            history_records:  Lista de {date, bbg_ticker, description, category, value}.

        Returns:
            (snapshot_upserted, history_inserted)
        """
        snap_count = 0
        hist_count = 0
        now = _now_iso()

        with sqlite3.connect(self._db_path) as conn:
            # UPSERT em macro_series_latest
            for rec in snapshot_records:
                if not rec.get("bbg_ticker"):
                    continue
                conn.execute(
                    """
                    INSERT INTO macro_series_latest
                        (bbg_ticker, description, category, px_last, date, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(bbg_ticker) DO UPDATE SET
                        px_last    = excluded.px_last,
                        date       = excluded.date,
                        updated_at = excluded.updated_at,
                        description = COALESCE(excluded.description, macro_series_latest.description),
                        category   = COALESCE(excluded.category,    macro_series_latest.category)
                    WHERE excluded.date >= macro_series_latest.date
                    """,
                    (
                        rec["bbg_ticker"],
                        rec.get("description"),
                        rec.get("category"),
                        rec.get("px_last"),
                        rec.get("date", now[:10]),
                        now,
                    ),
                )
                snap_count += 1

            # INSERT OR IGNORE em macro_series_history (dedup por bbg_ticker + date)
            for rec in history_records:
                if not rec.get("bbg_ticker") or not rec.get("date"):
                    continue
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO macro_series_history
                        (bbg_ticker, description, category, date, value, source_file, ingestion_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rec["bbg_ticker"],
                        rec.get("description"),
                        rec.get("category"),
                        rec["date"],
                        rec.get("value"),
                        rec.get("source_file", ""),
                        now,
                    ),
                )
                if cur.rowcount == 1:
                    hist_count += 1

            conn.commit()

        _log.info("macro_series_written", snapshot=snap_count, history=hist_count)
        return snap_count, hist_count

    def start_run(self, run_id: str) -> None:
        """Registra início de execução em ingestion_log."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO ingestion_log
                    (run_id, started_at, status)
                VALUES (?, ?, 'running')
                """,
                (run_id, _now_iso()),
            )
            conn.commit()

    def finish_run(self, run_id: str, stats: dict[str, Any], status: str = "ok", error: str | None = None) -> None:
        """Atualiza ingestion_log com resultado final da execução."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                UPDATE ingestion_log SET
                    finished_at    = ?,
                    zips_found     = ?,
                    zips_processed = ?,
                    csvs_extracted = ?,
                    rows_ingested  = ?,
                    rows_skipped   = ?,
                    status         = ?,
                    error_message  = ?
                WHERE run_id = ?
                """,
                (
                    _now_iso(),
                    stats.get("zips_found",     0),
                    stats.get("zips_processed", 0),
                    stats.get("csvs_extracted", 0),
                    stats.get("rows_ingested",  0),
                    stats.get("rows_skipped",   0),
                    status,
                    error,
                    run_id,
                ),
            )
            conn.commit()

    # ── Privados ──────────────────────────────────────────────────────────────

    def _write_timeseries_record(
        self, conn: sqlite3.Connection, rec: dict, stats: WriteStats
    ) -> None:
        """
        Insere um registro em bql_timeseries.
        INSERT OR IGNORE garante que duplicatas (mesmo ticker/field/date) são silenciadas.
        value = None gravado como NULL real.
        """
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO bql_timeseries
                (ticker, field, date, value, frequency, source_file, ingestion_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec["ticker"],
                rec["field"],
                rec["date"],
                rec.get("value"),            # None → NULL
                rec.get("frequency", "D"),
                rec["source_file"],
                rec["ingestion_timestamp"],
            ),
        )
        if cur.rowcount == 1:
            stats.rows_inserted += 1
        else:
            stats.rows_skipped += 1

    def _refresh_latest_from_timeseries(
        self, conn: sqlite3.Connection, source_file: str, stats: WriteStats
    ) -> None:
        """
        Atualiza bql_latest para todos os (ticker, field) deste source_file.
        Pega o registro com a data mais recente em timeseries.
        Usa UPSERT (INSERT OR REPLACE) para garantir sempre o valor mais novo.
        """
        # Seleciona o valor mais recente por (ticker, field) do source_file atual
        rows = conn.execute(
            """
            SELECT ticker, field,
                   MAX(date)                         AS latest_date,
                   value,                            -- valor na data mais recente
                   source_file,
                   MAX(ingestion_timestamp)          AS updated_at
            FROM bql_timeseries
            WHERE source_file = ?
            GROUP BY ticker, field
            """,
            (source_file,),
        ).fetchall()

        # Para cada (ticker, field), verifica se a data deste batch é mais recente
        for row in rows:
            ticker, fld, latest_date, value, src, updated_at = (
                row[0], row[1], row[2], row[3], row[4], row[5]
            )

            # Verifica se o banco já tem uma data mais recente para este (ticker, field)
            existing = conn.execute(
                "SELECT latest_date FROM bql_latest WHERE ticker = ? AND field = ?",
                (ticker, fld),
            ).fetchone()

            if existing is None or latest_date >= existing[0]:
                conn.execute(
                    """
                    INSERT INTO bql_latest (ticker, field, latest_date, latest_value, source_file, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, field) DO UPDATE SET
                        latest_date  = excluded.latest_date,
                        latest_value = excluded.latest_value,
                        source_file  = excluded.source_file,
                        updated_at   = excluded.updated_at
                    WHERE excluded.latest_date >= bql_latest.latest_date
                    """,
                    (ticker, fld, latest_date, value, src, updated_at),
                )
                stats.latest_updated += 1

    def _write_missing_records(
        self, conn: sqlite3.Connection, run_id: str, records: list[dict], stats: WriteStats
    ) -> None:
        """Insere registros de dados ausentes em missing_data_log."""
        conn.executemany(
            """
            INSERT INTO missing_data_log
                (run_id, source_file, row_number, ticker, field,
                 reference_date, issue_type, issue_description, detected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    r["source_file"],
                    r.get("row_number"),
                    r.get("ticker"),
                    r.get("field"),
                    r.get("reference_date"),
                    r["issue_type"],
                    r.get("issue_description"),
                    r["detected_at"],
                )
                for r in records
            ],
        )
        stats.missing_logged += len(records)
