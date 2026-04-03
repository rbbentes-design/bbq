"""
MacroDesk Bloomberg Ecosystem — Agente Principal
=================================================

Este é o núcleo do ecossistema MacroDesk.

Responsabilidade:
  Orquestrar todo o pipeline de dados Bloomberg desde a captura dos
  arquivos exportados pelo BQuant até a gravação no banco SQLite.

Fluxo completo:
  1. Varrer Downloads por arquivos .zip do Bloomberg (bql_data_*.zip)
  2. Identificar quais arquivos ainda não foram processados
  3. Registrar cada zip como 'pending' no banco
  4. Extrair os CSVs de cada zip
  5. Renomear cada CSV com timestamp para evitar sobrescrita
  6. Salvar os CSVs renomeados em bql_data/
  7. Parsear cada CSV (detecta tipo, lê DataFrame)
  8. Normalizar para formato canônico (ticker, field, date, value)
  9. Gravar no banco SQLite (timeseries + latest + missing_log)
  10. Registrar log completo da execução

Regras absolutas:
  - Apenas dados Bloomberg são processados
  - Sem Yahoo Finance, sem IBKR, sem fallback externo
  - Se o dado não entrou no banco, não existe para o MacroDesk
  - Dados ausentes → NULL real (nunca zero ou string fake)
  - Sem duplicação de registros

Uso:
    # Via linha de comando
    python -m core.bloomberg_main_agent

    # Programático
    from core.bloomberg_main_agent import BloombergMainAgent
    agent = BloombergMainAgent()
    result = agent.run()
    print(result.status, result.rows_ingested)

    # Com callback de progresso (para UI)
    agent.run(on_progress=minha_funcao_callback)
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

# ── Garante que o root do projeto está no sys.path ────────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import (
    BQL_DATA_DIR,
    DATABASE_PATH,
    DOWNLOADS_DIR,
    ZIP_PATTERN,
)
from core.csv_parser import CsvParser
from core.data_normalizer import DataNormalizer
from core.db_schema import create_schema
from core.db_writer import DbWriter, WriteStats
from core.logger import get_logger
from core.zip_extractor import ZipExtractor
from core.zip_scanner import ZipInfo, ZipScanner

_log = get_logger("bloomberg_agent")


# ── Resultado da execução ─────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """Resumo do resultado de uma execução do agente."""
    run_id:          str
    started_at:      str
    finished_at:     str = ""
    status:          str = "running"   # running | ok | partial | error | no_new_data
    zips_found:      int = 0
    zips_processed:  int = 0
    csvs_extracted:  int = 0
    rows_ingested:   int = 0
    rows_skipped:    int = 0
    missing_logged:  int = 0
    errors:          int = 0
    error_message:   str = ""
    messages:        list[str] = field(default_factory=list)

    def add_message(self, msg: str) -> None:
        self.messages.append(msg)
        _log.info("agent_message", msg=msg)


# ── Agente Principal ──────────────────────────────────────────────────────────

class BloombergMainAgent:
    """
    Agente principal do ecossistema MacroDesk Bloomberg.

    Este agente substitui todos os agentes anteriores de captura, extração,
    renomeação e ingestão de dados Bloomberg.

    Args:
        downloads_dir: Pasta Downloads (default: ~/Downloads).
        bql_data_dir:  Pasta bql_data para CSVs extraídos.
        db_path:       Caminho para macrodesk.db.
        zip_pattern:   Padrão glob para zips Bloomberg.
    """

    def __init__(
        self,
        downloads_dir: Path | None = None,
        bql_data_dir:  Path | None = None,
        db_path:       Path | None = None,
        zip_pattern:   str  = ZIP_PATTERN,
    ) -> None:
        self._downloads = downloads_dir or DOWNLOADS_DIR
        self._bql_data  = bql_data_dir  or BQL_DATA_DIR
        self._db_path   = db_path       or DATABASE_PATH
        self._pattern   = zip_pattern

        # Garante que o schema do banco existe
        create_schema(self._db_path)

        # Componentes internos
        self._scanner    = ZipScanner(self._downloads, self._db_path, self._pattern)
        self._extractor  = ZipExtractor(self._bql_data, self._db_path)
        self._parser     = CsvParser()
        self._normalizer = DataNormalizer()
        self._writer     = DbWriter(self._db_path)

    # ── API Pública ───────────────────────────────────────────────────────────

    def run(
        self,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> AgentResult:
        """
        Executa o pipeline completo de ingestão Bloomberg.

        Args:
            on_progress: Callback opcional chamado a cada etapa com mensagem de status.
                         Útil para atualizar UI (ex: Tkinter label).

        Returns:
            AgentResult com estatísticas completas da execução.
        """
        run_id     = str(uuid.uuid4())[:8]
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        result     = AgentResult(run_id=run_id, started_at=started_at)

        def _notify(msg: str) -> None:
            result.add_message(msg)
            if on_progress:
                try:
                    on_progress(msg)
                except Exception:
                    pass

        _log.info("agent_started", run_id=run_id)
        _notify("Agente iniciado. Procurando arquivos .zip em Downloads...")

        # Registra início no banco
        self._writer.start_run(run_id)

        try:
            # ── Passo 1: Varrer Downloads ─────────────────────────────────────
            new_zips = self._scanner.find_new_zips()
            result.zips_found = len(new_zips)

            if not new_zips:
                msg = "Nenhum novo arquivo .zip Bloomberg encontrado em Downloads."
                _notify(msg)
                result.status = "no_new_data"
                self._writer.finish_run(run_id, _result_to_stats(result), status="no_new_data")
                result.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                return result

            _notify(f"{result.zips_found} arquivo(s) .zip novo(s) encontrado(s).")

            # ── Passos 2-9: Processar cada zip ────────────────────────────────
            all_write_stats = WriteStats()

            for zi in new_zips:
                zip_stats = self._process_zip(zi, run_id, _notify)
                result.csvs_extracted += zip_stats["csvs_extracted"]
                all_write_stats.rows_inserted  += zip_stats["rows_inserted"]
                all_write_stats.rows_skipped   += zip_stats["rows_skipped"]
                all_write_stats.missing_logged += zip_stats["missing_logged"]
                all_write_stats.errors         += zip_stats["errors"]

                if zip_stats["success"]:
                    result.zips_processed += 1

            result.rows_ingested  = all_write_stats.rows_inserted
            result.rows_skipped   = all_write_stats.rows_skipped
            result.missing_logged = all_write_stats.missing_logged
            result.errors         = all_write_stats.errors

            # ── Status final ──────────────────────────────────────────────────
            if result.errors == 0 and result.zips_processed == result.zips_found:
                result.status = "ok"
                _notify(f"Banco de dados atualizado com sucesso. {result.rows_ingested} linhas ingeridas.")
            elif result.zips_processed > 0:
                result.status = "partial"
                _notify(
                    f"Banco atualizado parcialmente. "
                    f"{result.zips_processed}/{result.zips_found} zips processados. "
                    f"Verifique o log de erros."
                )
            else:
                result.status = "error"
                _notify("Falha na ingestão. Nenhum zip foi processado com sucesso. Verifique o log.")

        except Exception as exc:
            result.status = "error"
            result.error_message = str(exc)
            _log.error("agent_fatal_error", run_id=run_id, error=str(exc))
            _notify(f"Erro fatal no agente: {exc}")

        finally:
            result.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            try:
                self._writer.finish_run(
                    run_id,
                    _result_to_stats(result),
                    status=result.status,
                    error=result.error_message or None,
                )
            except Exception:
                pass

        _notify("Ingestão finalizada.")
        _log.info(
            "agent_finished",
            run_id=run_id,
            status=result.status,
            zips_processed=result.zips_processed,
            rows_ingested=result.rows_ingested,
            errors=result.errors,
        )
        return result

    # ── Pipeline por Zip ─────────────────────────────────────────────────────

    def _process_zip(
        self,
        zi: ZipInfo,
        run_id: str,
        notify: Callable[[str], None],
    ) -> dict:
        """
        Processa um único zip: extrai CSVs, parseia, normaliza e grava no banco.
        Retorna dicionário de estatísticas.
        """
        stats = {
            "csvs_extracted": 0,
            "rows_inserted":  0,
            "rows_skipped":   0,
            "missing_logged": 0,
            "errors":         0,
            "success":        False,
        }

        notify(f"Processando: {zi.zip_name}")

        # Registra zip como pendente
        zip_id = self._scanner.register_zip_as_pending(zi)

        try:
            # ── Passo 3-7: Extrair e renomear CSVs ───────────────────────────
            extracted_files = self._extractor.extract(zi, run_id)
            stats["csvs_extracted"] = len(extracted_files)

            if not extracted_files:
                notify(f"  Nenhum CSV encontrado em {zi.zip_name}.")
                self._scanner.mark_zip_done(zip_id, status="error", error="Nenhum CSV no zip")
                stats["errors"] += 1
                return stats

            notify(
                f"  {len(extracted_files)} CSV(s) extraído(s) e renomeados com timestamp."
            )

            # ── Passos 8-10: Parsear, Normalizar e Gravar ────────────────────
            for ef in extracted_files:
                file_stats = self._process_csv(ef.saved_path, run_id)
                stats["rows_inserted"]  += file_stats.rows_inserted
                stats["rows_skipped"]   += file_stats.rows_skipped
                stats["missing_logged"] += file_stats.missing_logged
                stats["errors"]         += file_stats.errors

            self._scanner.mark_zip_done(zip_id, status="ok")
            stats["success"] = True

            notify(
                f"  {zi.zip_name} | {stats['rows_inserted']} linhas ingeridas, "
                f"{stats['rows_skipped']} duplicatas ignoradas."
            )

        except Exception as exc:
            stats["errors"] += 1
            error_msg = str(exc)
            _log.error("zip_processing_failed", zip_name=zi.zip_name, error=error_msg)
            self._scanner.mark_zip_done(zip_id, status="error", error=error_msg)
            notify(f"  ERRO ao processar {zi.zip_name}: {exc}")

        return stats

    def _process_csv(self, csv_path: Path, run_id: str) -> WriteStats:
        """
        Parseia um CSV, normaliza e grava no banco.
        Retorna WriteStats da operação.
        """
        empty_stats = WriteStats()

        # Parseia
        dataset_type, df = self._parser.parse(csv_path)
        if df is None or df.empty:
            _log.warning("csv_empty_or_unparseable", file=csv_path.name)
            return empty_stats

        if dataset_type == "unknown":
            _log.info("csv_type_unknown_skipped", file=csv_path.name)
            return empty_stats

        # macro_series: grava também em macro_series_latest (tabela dedicada)
        if dataset_type == "macro_series":
            snap = self._extract_macro_snapshot(df)
            if snap:
                self._writer.write_macro_series(run_id, snap, [])

        # Normaliza para bql_timeseries / bql_latest
        ts_records, missing_records = self._normalizer.normalize(
            dataset_type, df, csv_path.name
        )

        if not ts_records and not missing_records:
            return empty_stats

        # Grava no banco
        write_stats = self._writer.write_batch(
            run_id, ts_records, missing_records, csv_path.name
        )
        return write_stats

    def _extract_macro_snapshot(self, df: "Any") -> list[dict]:
        """
        Extrai snapshot de macro_series_*.csv para gravar em macro_series_latest.
        Formato esperado: bbg_ticker, description, category, px_last
        """
        from datetime import date
        from core.data_normalizer import _to_str, _to_float

        cols = {c.lower(): c for c in df.columns}
        ticker_col = cols.get("bbg_ticker") or cols.get("ticker")
        desc_col   = cols.get("description")
        cat_col    = cols.get("category")
        val_col    = cols.get("px_last") or cols.get("value")
        date_col   = cols.get("date")
        today      = date.today().isoformat()

        snap = []
        for _, row in df.iterrows():
            ticker = _to_str(row.get(ticker_col)) if ticker_col else None
            if not ticker:
                continue
            snap.append({
                "bbg_ticker":  ticker,
                "description": _to_str(row.get(desc_col))  if desc_col else ticker,
                "category":    _to_str(row.get(cat_col))   if cat_col  else "other",
                "px_last":     _to_float(row.get(val_col)) if val_col  else None,
                "date":        _to_str(row.get(date_col))  if date_col else today,
            })
        return snap


# ── Helpers ───────────────────────────────────────────────────────────────────

def _result_to_stats(result: AgentResult) -> dict:
    return {
        "zips_found":     result.zips_found,
        "zips_processed": result.zips_processed,
        "csvs_extracted": result.csvs_extracted,
        "rows_ingested":  result.rows_ingested,
        "rows_skipped":   result.rows_skipped,
    }


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Ponto de entrada para execução via linha de comando."""
    print("=" * 60)
    print(" MacroDesk Bloomberg Agent")
    print("=" * 60)

    agent  = BloombergMainAgent()
    result = agent.run(on_progress=print)

    print()
    print("=" * 60)
    print(f" Status:          {result.status.upper()}")
    print(f" Zips encontrados: {result.zips_found}")
    print(f" Zips processados: {result.zips_processed}")
    print(f" CSVs extraídos:   {result.csvs_extracted}")
    print(f" Linhas ingeridas: {result.rows_ingested}")
    print(f" Duplicatas:       {result.rows_skipped}")
    print(f" Ausências log:    {result.missing_logged}")
    print(f" Erros:            {result.errors}")
    print("=" * 60)

    sys.exit(0 if result.status in ("ok", "no_new_data") else 1)


if __name__ == "__main__":
    main()
