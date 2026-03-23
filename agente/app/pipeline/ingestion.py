"""
Pipeline de Ingestao Diaria.

Orquestra a coleta de ZeroHedge e X, persistindo:
  - HTML bruto (RawStore)
  - Blocos normalizados (NormalizedStore)
  - DailyIngestionBundle completo (BundleStore)
  - Log de auditoria (AuditLogger)

Uso:
    from app.pipeline.ingestion import run_ingestion
    bundle = run_ingestion()
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

from app.audit import records as rec
from app.audit.logger import AuditLogger, get_logger
from app.auth.browser_profile import open_context
from app.config.settings import settings
from app.models.daily_ingestion_bundle import AuditSummary, DailyIngestionBundle
from app.models.market_ear_block import MarketEarBlock
from app.models.x_timeline_item import XTimelineItem
from app.providers import x_timeline as x_prov
from app.providers import zerohedge as zh_prov
from app.storage.bundle_store import bundle_store
from app.storage.normalized_store import normalized_store
from app.storage.paths import workspace
from app.storage.raw_store import raw_store
from app.utils.timestamps import new_ulid, utcnow

_log = get_logger("pipeline.ingestion")


def run_ingestion(headless: bool | None = None) -> DailyIngestionBundle:
    """
    Executa o pipeline de ingestao completo.

    1. Abre sessao Playwright com perfil do workspace
    2. Coleta ZeroHedge Market Ear -> raw HTML + blocos normalizados
    3. Coleta X timeline -> raw HTML + itens normalizados
    4. Monta e persiste o DailyIngestionBundle
    5. Grava log de auditoria

    Returns:
        DailyIngestionBundle com todos os dados coletados.
    """
    run_id = new_ulid()
    run_date = date.today()
    workspace.ensure_all()

    audit_path = workspace.audit_log_path(run_date)
    audit = AuditLogger(audit_path)

    _log.info("pipeline_start", run_id=run_id, run_date=str(run_date))
    audit.write(rec.ok(run_id, "pipeline", "start"))

    zh_blocks: list[MarketEarBlock] = []
    x_items: list[XTimelineItem] = []
    artifact_paths: dict[str, str] = {}
    errors: list[str] = []

    with sync_playwright() as p:
        ctx = open_context(p, headless=headless)
        try:
            # ── ZeroHedge ──────────────────────────────────────────────────────
            try:
                page = ctx.new_page()
                html = zh_prov.collect_html(page)
                page.close()

                zh_doc, html_path = raw_store.build_document(
                    source_name=zh_prov.SOURCE_NAME,
                    source_url=settings.zerohedge_market_ear_url,
                    access_method="playwright",
                    html=html,
                )
                artifact_paths["zh_html"] = str(html_path)
                audit.write(rec.ok(run_id, "collect", "zh_html_saved",
                                   source=zh_prov.SOURCE_NAME,
                                   html_bytes=len(html)))

                from bs4 import BeautifulSoup
                zh_blocks = zh_prov._parse_blocks(html, zh_doc)
                norm_path = normalized_store.write_all(
                    zh_prov.SOURCE_NAME, run_id, zh_blocks
                )
                artifact_paths["zh_blocks"] = str(norm_path)
                audit.write(rec.ok(run_id, "parse", "zh_blocks_saved",
                                   source=zh_prov.SOURCE_NAME,
                                   blocks=len(zh_blocks)))

            except Exception as exc:
                msg = f"ZeroHedge collect failed: {exc}"
                errors.append(msg)
                _log.error("zh_collect_error", error=str(exc))
                audit.write(rec.error(run_id, "collect", "zh_failed",
                                      msg=msg, source=zh_prov.SOURCE_NAME))

            # ── X Timeline ─────────────────────────────────────────────────────
            try:
                page2 = ctx.new_page()
                x_html = x_prov.collect_html(page2)
                page2.close()

                x_doc, x_html_path = raw_store.build_document(
                    source_name=x_prov.SOURCE_NAME,
                    source_url="https://x.com/home",
                    access_method="playwright",
                    html=x_html,
                )
                artifact_paths["x_html"] = str(x_html_path)
                audit.write(rec.ok(run_id, "collect", "x_html_saved",
                                   source=x_prov.SOURCE_NAME,
                                   html_bytes=len(x_html)))

                page3 = ctx.new_page()
                x_items = x_prov.collect(page3, x_doc)
                page3.close()

                x_norm_path = normalized_store.write_all(
                    x_prov.SOURCE_NAME, run_id, x_items
                )
                artifact_paths["x_items"] = str(x_norm_path)
                audit.write(rec.ok(run_id, "parse", "x_items_saved",
                                   source=x_prov.SOURCE_NAME,
                                   items=len(x_items)))

            except Exception as exc:
                msg = f"X collect failed: {exc}"
                errors.append(msg)
                _log.error("x_collect_error", error=str(exc))
                audit.write(rec.error(run_id, "collect", "x_failed",
                                      msg=msg, source=x_prov.SOURCE_NAME))

        finally:
            ctx.close()

    # ── Bundle ─────────────────────────────────────────────────────────────────
    audit_summary = AuditSummary(
        total_records=len(zh_blocks) + len(x_items),
        errors=len(errors),
        error_messages=errors,
    )

    bundle = DailyIngestionBundle(
        run_id=run_id,
        run_date=run_date,
        created_at=utcnow(),
        market_ear_blocks=zh_blocks,
        x_items=x_items,
        candidate_signals=[],
        audit_summary=audit_summary,
        artifact_paths=artifact_paths,
    )

    bundle_path = bundle_store.save(bundle)
    artifact_paths["bundle"] = str(bundle_path)
    audit.write(rec.ok(run_id, "pipeline", "bundle_saved",
                       bundle_path=str(bundle_path),
                       zh_blocks=len(zh_blocks),
                       x_items=len(x_items)))

    # ── Relatorios ─────────────────────────────────────────────────────────────
    from app.views.report import save_reports
    md_path, json_path, html_path = save_reports(bundle)
    artifact_paths["markdown"] = str(md_path)
    artifact_paths["json_summary"] = str(json_path)
    artifact_paths["html_report"] = str(html_path)
    # Atualiza bundle com paths dos relatorios
    bundle = bundle.model_copy(update={"artifact_paths": artifact_paths})
    bundle_store.save(bundle)  # re-salva com paths completos
    audit.write(rec.ok(run_id, "pipeline", "reports_saved",
                       markdown=str(md_path), json=str(json_path), html=str(html_path)))

    _log.info("pipeline_done", run_id=run_id,
              zh_blocks=len(zh_blocks), x_items=len(x_items),
              errors=len(errors))

    return bundle
