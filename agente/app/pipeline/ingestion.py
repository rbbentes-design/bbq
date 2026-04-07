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
from app.providers import deepvue as deepvue_prov
from app.providers import spectra as spectra_prov
from app.providers import spotgamma as sg_prov
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

    from app.models.rss_item import RSSItem
    from app.models.spotgamma_report import SpotGammaReport
    zh_blocks: list[MarketEarBlock] = []
    x_items: list[XTimelineItem] = []
    sg_reports: list[SpotGammaReport] = []
    rss_items: list[RSSItem] = []
    polymarket_markets: list[dict] = []
    fred_data: dict = {}
    damodaran_data: dict = {}
    global_liquidity_data: dict = {}
    market_prices_data: dict = {}
    swaggy_data: dict = {}
    artifact_paths: dict[str, str] = {}
    errors: list[str] = []
    _x_source_doc = None  # usado para coleta semanal de perfis

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
                zh_blocks = zh_prov._parse_blocks(html, zh_doc)[:settings.zerohedge_blocks_limit]
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
                    source_url=x_prov._TIMELINE_URL,
                    access_method="playwright",
                    html=x_html,
                )
                _x_source_doc = x_doc  # guardado para coleta semanal
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

            # ── SpotGamma FlowPatrol ────────────────────────────────────────────
            try:
                sg_page = ctx.new_page()
                sg_report = sg_prov.collect_flow_patrol(sg_page)
                sg_page.close()
                if sg_report:
                    sg_reports.append(sg_report)
                    audit.write(rec.ok(run_id, "collect", "spotgamma_flowpatrol_done",
                                       source=sg_prov.SOURCE_NAME,
                                       sections=len(sg_report.sections)))
                else:
                    _log.warning("spotgamma_no_report", date=str(run_date))
            except Exception as exc:
                msg = f"SpotGamma FlowPatrol failed: {exc}"
                errors.append(msg)
                _log.warning("spotgamma_flowpatrol_error", error=str(exc))
                audit.write(rec.error(run_id, "collect", "spotgamma_flowpatrol_failed",
                                      msg=msg, source=sg_prov.SOURCE_NAME))

            # ── Spectra Markets am/FX ──────────────────────────────────────────
            try:
                spectra_page = ctx.new_page()
                spectra_items = spectra_prov.collect(spectra_page, max_articles=3)
                spectra_page.close()
                rss_items.extend(spectra_items)
                audit.write(rec.ok(run_id, "collect", "spectra_done",
                                   source=spectra_prov.SOURCE_NAME,
                                   items=len(spectra_items)))
            except Exception as exc:
                msg = f"Spectra collect failed: {exc}"
                errors.append(msg)
                _log.warning("spectra_collect_error", error=str(exc))

            # ── DeepVue ───────────────────────────────────────────────────────────
            try:
                dv_page = ctx.new_page()
                dv_items = deepvue_prov.collect(dv_page)
                dv_page.close()
                rss_items.extend(dv_items)
                if dv_items:
                    audit.write(rec.ok(run_id, "collect", "deepvue_done",
                                       source=deepvue_prov.SOURCE_NAME,
                                       items=len(dv_items)))
                else:
                    _log.info("deepvue_no_items_or_not_authenticated")
            except Exception as exc:
                msg = f"DeepVue collect failed: {exc}"
                errors.append(msg)
                _log.warning("deepvue_collect_error", error=str(exc))

            # ── SpotGamma Founder's Notes ───────────────────────────────────────
            try:
                fn_page = ctx.new_page()
                fn_reports = sg_prov.collect_founders_notes(fn_page, max_notes=2)
                fn_page.close()
                sg_reports.extend(fn_reports)
                audit.write(rec.ok(run_id, "collect", "spotgamma_founders_done",
                                   source=sg_prov.SOURCE_NAME,
                                   notes=len(fn_reports)))
            except Exception as exc:
                msg = f"SpotGamma Founders Notes failed: {exc}"
                errors.append(msg)
                _log.warning("spotgamma_founders_error", error=str(exc))
                audit.write(rec.error(run_id, "collect", "spotgamma_founders_failed",
                                      msg=msg, source=sg_prov.SOURCE_NAME))

            # ── ZeroHedge Main Page (somente sábado — podcast) ────────────────
            if run_date.weekday() == 5:  # 5 = sábado
                try:
                    zh_main_page = ctx.new_page()
                    zh_main_items = zh_prov.collect_main_page(zh_main_page, max_articles=5)
                    zh_main_page.close()
                    rss_items.extend(zh_main_items)
                    if zh_main_items:
                        audit.write(rec.ok(run_id, "collect", "zh_main_done",
                                           articles=len(zh_main_items)))
                        _log.info("zh_main_done", articles=len(zh_main_items))
                except Exception as exc:
                    msg = f"ZeroHedge Main collect failed: {exc}"
                    errors.append(msg)
                    _log.warning("zh_main_error", error=str(exc))

            # ── Perfis semanais (somente sexta-feira) ──────────────────────────
            if run_date.weekday() == 4:  # 4 = sexta-feira
                try:
                    weekly_page = ctx.new_page()
                    from app.providers.x_timeline import (
                        collect_profile_week,
                        WEEKLY_RECAP_ACCOUNTS,
                    )
                    from app.models.source_document import SourceDocument as _SD
                    from app.utils.timestamps import new_ulid as _ulid
                    # Usa x_doc capturado anteriormente, ou cria documento mínimo
                    _weekly_doc = _x_source_doc or _SD(
                        id=_ulid(),
                        source_name="x_weekly",
                        source_url="https://x.com",
                        access_method="playwright",
                        html="",
                    )
                    existing_urls = {it.url for it in x_items}
                    weekly_items: list[XTimelineItem] = []
                    for handle in WEEKLY_RECAP_ACCOUNTS:
                        w_items = collect_profile_week(
                            weekly_page, _weekly_doc, handle, days_back=7, max_tweets=25
                        )
                        # Insere no início de x_items (maior prioridade no contexto)
                        for it in w_items:
                            if it.url not in existing_urls:
                                existing_urls.add(it.url)
                                weekly_items.append(it)
                    weekly_page.close()
                    # Prepend — tweets semanais aparecem primeiro no bundle
                    x_items = weekly_items + x_items
                    audit.write(rec.ok(run_id, "collect", "weekly_profiles_done",
                                       accounts=len(WEEKLY_RECAP_ACCOUNTS),
                                       new_items=len(weekly_items)))
                    _log.info("weekly_profiles_done",
                              accounts=len(WEEKLY_RECAP_ACCOUNTS),
                              new_items=len(weekly_items))
                except Exception as exc:
                    msg = f"Weekly profiles collect failed: {exc}"
                    errors.append(msg)
                    _log.warning("weekly_profiles_error", error=str(exc))

            # ── SwaggyStocks — WSB mentions via ApeWisdom + squeeze ──────────────
            try:
                from app.providers.swaggy_stocks import collect as swaggy_collect, swaggy_result_to_dict
                sw_result = swaggy_collect(max_wsb=50, max_squeeze=30)
                swaggy_data = swaggy_result_to_dict(sw_result)
                _log.info("swaggy_ingest_done",
                          wsb=len(sw_result.wsb_mentions),
                          squeeze=len(sw_result.squeeze_candidates),
                          top=sw_result.top_mentions[:5])
                audit.write(rec.ok(run_id, "collect", "swaggy_done",
                                   wsb=len(sw_result.wsb_mentions),
                                   squeeze=len(sw_result.squeeze_candidates)))
            except Exception as exc:
                msg = f"SwaggyStocks collect failed: {exc}"
                errors.append(msg)
                _log.warning("swaggy_ingest_error", error=str(exc))

        finally:
            ctx.close()

    # ── RSS Feeds ──────────────────────────────────────────────────────────────
    try:
        from app.providers.rss_feed import collect as rss_collect
        from app.cli.sources import _load as load_sources
        src = load_sources()
        rss_urls = src.get("rss_feeds") or None   # None = usa defaults
        rss_items.extend(rss_collect(feed_urls=rss_urls))
        audit.write(rec.ok(run_id, "collect", "rss_done", items=len(rss_items)))
    except Exception as exc:
        msg = f"RSS collect failed: {exc}"
        errors.append(msg)
        _log.warning("rss_collect_error", error=str(exc))

    # ── Polymarket Prediction Markets ──────────────────────────────────────────
    try:
        from app.providers.polymarket import collect as poly_collect
        polymarket_markets = poly_collect(max_results=10)
        audit.write(rec.ok(run_id, "collect", "polymarket_done",
                           markets=len(polymarket_markets)))
        _log.info("polymarket_done", markets=len(polymarket_markets))
    except Exception as exc:
        msg = f"Polymarket collect failed: {exc}"
        errors.append(msg)
        _log.warning("polymarket_error", error=str(exc))

    # ── FRED — Séries macro + agenda econômica ─────────────────────────────────
    try:
        from app.providers.fred import collect as fred_collect, collect_release_calendar
        from app.config.settings import settings as _s
        if _s.fred_api_key:
            fred_series = fred_collect(lookback_days=180)
            fred_calendar = collect_release_calendar(days_ahead=10)
            fred_data = {"series": fred_series, "calendar": fred_calendar}
            audit.write(rec.ok(run_id, "collect", "fred_done",
                               categories=len(fred_series),
                               calendar_items=len(fred_calendar)))
            _log.info("fred_done", categories=len(fred_series), calendar_items=len(fred_calendar))
    except Exception as exc:
        _log.warning("fred_error", error=str(exc))

    # ── Damodaran — ERP, Country Risk, WACC ───────────────────────────────────
    try:
        from app.providers.damodaran import collect as damo_collect
        damodaran_data = damo_collect()
        audit.write(rec.ok(run_id, "collect", "damodaran_done",
                           sectors=len(damodaran_data.get("wacc_sectors", [])),
                           countries=len(damodaran_data.get("country_risk", []))))
        _log.info("damodaran_done",
                  erp=damodaran_data.get("erp_current"),
                  sectors=len(damodaran_data.get("wacc_sectors", [])),
                  countries=len(damodaran_data.get("country_risk", [])))
    except Exception as exc:
        _log.warning("damodaran_error", error=str(exc))

    # ── Global Liquidity ───────────────────────────────────────────────────────
    try:
        from app.providers.global_liquidity import collect as liq_collect
        global_liquidity_data = liq_collect(lookback_days=365)
        summary = global_liquidity_data.get("summary", {})
        nfl = summary.get("net_fed_liquidity", {})
        mmf = summary.get("money_market_total", {})
        _log.info("global_liquidity_done",
                  net_fed_liquidity=nfl.get("value"),
                  mmf_total=mmf.get("value") if mmf else None,
                  ecb_ok=bool(global_liquidity_data.get("ecb")))
        audit.write(rec.ok(run_id, "collect", "global_liquidity_done",
                           net_fed=nfl.get("value"),
                           ecb=bool(global_liquidity_data.get("ecb"))))
    except Exception as exc:
        _log.warning("global_liquidity_error", error=str(exc))

    # ── Market Prices ─────────────────────────────────────────────────────────
    try:
        from app.providers.market_prices import collect as prices_collect
        market_prices_data = prices_collect()
        audit.write(rec.ok(run_id, "collect", "market_prices_done",
                           tickers=len(market_prices_data)))
        _log.info("market_prices_done", tickers=len(market_prices_data))
    except Exception as exc:
        msg = f"Market prices collect failed: {exc}"
        errors.append(msg)
        _log.warning("market_prices_error", error=str(exc))

    # ── Download de imagens ────────────────────────────────────────────────────
    try:
        from app.utils.image_downloader import download_images
        img_dir = workspace.bundles / str(run_date) / "images"

        all_img_urls = [u for b in zh_blocks for u in b.image_refs]
        all_img_urls += [u for it in x_items for u in it.media_refs]

        img_map = download_images(list(dict.fromkeys(all_img_urls)), img_dir)  # preserva ordem, deduplica

        # Atualiza refs nos blocos para paths locais
        zh_blocks = [
            b.model_copy(update={"image_refs": [
                str(img_map[u]) if u in img_map else u for u in b.image_refs
            ]}) for b in zh_blocks
        ]
        x_items = [
            it.model_copy(update={"media_refs": [
                str(img_map[u]) if u in img_map else u for u in it.media_refs
            ]}) for it in x_items
        ]
        artifact_paths["images_dir"] = str(img_dir)
        audit.write(rec.ok(run_id, "pipeline", "images_downloaded",
                           total=len(all_img_urls), saved=len(img_map)))
    except Exception as exc:
        msg = f"Image download failed: {exc}"
        errors.append(msg)
        _log.warning("image_download_error", error=str(exc))

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
        rss_items=rss_items,
        spotgamma_reports=sg_reports,
        candidate_signals=[],
        polymarket_markets=polymarket_markets,
        fred_data=fred_data,
        damodaran_data=damodaran_data,
        global_liquidity=global_liquidity_data,
        market_prices=market_prices_data,
        swaggy_data=swaggy_data,
        audit_summary=audit_summary,
        artifact_paths=artifact_paths,
    )

    bundle_path = bundle_store.save(bundle)
    artifact_paths["bundle"] = str(bundle_path)
    audit.write(rec.ok(run_id, "pipeline", "bundle_saved",
                       bundle_path=str(bundle_path),
                       zh_blocks=len(zh_blocks),
                       x_items=len(x_items)))

    # ── Curação LLM ────────────────────────────────────────────────────────────
    curation_result = None
    if settings.curation_enabled:
        try:
            from app.curation.orchestrator import run_curation
            curation_result = run_curation(bundle, run_id, str(run_date))
            artifact_paths["curation"] = str(
                workspace.curation_result_path(run_date, run_id)
            )
            audit.write(rec.ok(run_id, "curation", "done",
                               verdict=curation_result.verification.overall_verdict,
                               primary=curation_result.narrative.primary_signal.label,
                               confidence=curation_result.narrative.primary_signal.confidence))
        except Exception as exc:
            msg = f"Curation failed: {exc}"
            errors.append(msg)
            _log.warning("curation_error", error=str(exc))
            audit.write(rec.error(run_id, "curation", "failed", msg=msg))

    # ── Tracking de narrativas ─────────────────────────────────────────────────
    trend = None
    try:
        from app.curation.narrative_tracker import load_trend
        trend = load_trend(days=7)
    except Exception as exc:
        _log.warning("trend_load_error", error=str(exc))

    # ── Relatorios ─────────────────────────────────────────────────────────────
    from app.views.report import save_reports
    md_path, json_path, html_path = save_reports(bundle, curation_result=curation_result, trend=trend)
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
