from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.audit.logger import get_logger
from app.config.settings import settings
from app.curation.corpus import build_corpus_text, build_item_index
from app.curation.corrections import get_few_shot_examples, load_corrections
from app.curation.evidence_gatherer import gather_evidence
from app.curation.models import CurationResult, EvidenceGatheringTrace
from app.curation.narrative_detector import detect_narrative
from app.curation.scorer import score_items
from app.curation.verifier import verify_narrative
from app.curation.writer import to_docx, write as write_editorial
from app.models.daily_ingestion_bundle import DailyIngestionBundle
from app.storage.paths import workspace
from app.utils.timestamps import new_ulid

_log = get_logger("curation.orchestrator")


def run_curation(
    bundle: DailyIngestionBundle,
    run_id: str,
    run_date: str,
    mode_override: str | None = None,
    tema_hint: str | None = None,
) -> CurationResult:
    _log.info("curation_start", run_id=run_id, run_date=run_date)

    # 1. Build corpus
    corpus_text = build_corpus_text(bundle)
    item_index = build_item_index(bundle)

    # 2. Load corrections (few-shot examples)
    corrections = load_corrections()
    few_shot = get_few_shot_examples(corrections, "wrong_narrative") + \
               get_few_shot_examples(corrections, "missed_signal")

    # 3. Detect narrative
    _log.info("detecting_narrative")
    narrative = detect_narrative(corpus_text, run_id, run_date, few_shot_examples=few_shot)
    _log.info(
        "narrative_detected",
        label=narrative.primary_signal.label,
        confidence=narrative.primary_signal.confidence,
    )

    # 4. Agentic evidence gathering (if confidence below threshold)
    threshold = settings.curation_confidence_threshold
    max_iters = settings.curation_max_evidence_iterations
    evidence_traces: list[EvidenceGatheringTrace] = []

    updated_primary, trace_primary = gather_evidence(
        narrative.primary_signal,
        item_index,
        threshold=threshold,
        max_iterations=max_iters,
    )
    evidence_traces.append(trace_primary)

    updated_secondary = []
    for sec in narrative.secondary_signals:
        if sec is None:
            continue
        updated_sec, trace_sec = gather_evidence(
            sec, item_index, threshold=threshold, max_iterations=max_iters
        )
        updated_secondary.append(updated_sec)
        evidence_traces.append(trace_sec)

    # Rebuild narrative with updated signals
    narrative = narrative.model_copy(update={
        "primary_signal": updated_primary,
        "secondary_signals": updated_secondary,
    })

    # 5. Score all items against narrative
    _log.info("scoring_items")
    scored_items = score_items(narrative, bundle)

    # 6. Verify (anti-hallucination double-check with Haiku)
    _log.info("verifying_narrative")
    verification = verify_narrative(narrative, bundle, run_id)
    _log.info("verification_result", verdict=verification.overall_verdict)

    # 7. Assemble result
    result = CurationResult(
        id=new_ulid(),
        run_id=run_id,
        run_date=run_date,
        narrative=narrative,
        scored_items=scored_items,
        verification=verification,
        evidence_traces=evidence_traces,
        corrections_applied=len(few_shot),
        curated_at=datetime.now(timezone.utc),
    )

    # 8. Write editorial text
    _log.info("writing_editorial")
    output = None
    try:
        output = write_editorial(result, bundle, mode_override=mode_override, tema_hint=tema_hint)
        written_path = _save_written(output, run_date, run_id)
        result.artifact_paths["written"] = str(written_path)
        result.artifact_paths["written_mode"] = output.mode
        _log.info("writing_done", mode=output.mode, chars=len(output.text), path=str(written_path))
    except Exception as exc:
        _log.warning("writing_failed", error=str(exc))

    # 8b. TTS — gera áudio MP3 via ElevenLabs (só podcast de sábado)
    if output is not None and output.mode == "podcast_sabado" and settings.elevenlabs_api_key:
        try:
            from app.providers.elevenlabs_tts import generate_audio
            audio_dir = workspace.bundles / run_date
            audio_paths = generate_audio(output.text, output.mode, audio_dir, run_id)
            result.artifact_paths.update(audio_paths)
            _log.info("tts_done", files=list(audio_paths.keys()))
        except Exception as exc:
            _log.warning("tts_failed", error=str(exc))

    # 9b. Enrichment: ISQ + Technical + Risk + Monte Carlo + Scenarios + Web Search
    _run_enrichment(result, bundle, run_date, run_id)

    # 9c. Persist (depois do writer e enrichment para incluir artifact_paths completo)
    _save_result(result, run_date, run_id)

    _log.info(
        "curation_done",
        run_id=run_id,
        verdict=verification.overall_verdict,
        scored=len(scored_items),
        primary_conf=updated_primary.confidence,
    )
    return result


def _run_enrichment(
    result: CurationResult,
    bundle: DailyIngestionBundle,
    run_date: str,
    run_id: str,
) -> None:
    """
    Executa módulos de enrichment em sequência após a curadoria:
      - ISQ Signal Qualification
      - Análise técnica (RSI, MACD, Bollinger)
      - Métricas de risco (VaR, CVaR, MaxDD)
      - Monte Carlo
      - Cenários Bull/Base/Bear
      - Web Search complementar
      - Diagrama de transmissão causal (HTML)
      - Charts (K-line, heatmap, radar, fan)
      - Export Excel

    Todos os erros são absorvidos — enrichment nunca bloqueia o pipeline.
    """
    from datetime import date as date_type

    try:
        d = date_type.fromisoformat(run_date)
    except ValueError:
        d = __import__("datetime").date.today()

    out_dir = workspace.bundles / d.isoformat()
    enrichment_dir = out_dir / "enrichment"
    enrichment_dir.mkdir(parents=True, exist_ok=True)

    market_prices = bundle.market_prices or {}
    primary = result.narrative.primary_signal

    # ── ISQ ───────────────────────────────────────────────────────────────────
    isq_signal = None
    try:
        from app.curation.isq import extract_isq, save_isq
        isq_signal = extract_isq(result, run_id)
        if isq_signal:
            isq_path = enrichment_dir / f"{run_id}_isq.json"
            save_isq(isq_signal, isq_path)
            result.artifact_paths["isq"] = str(isq_path)
    except Exception as exc:
        _log.warning("enrichment_isq_error", error=str(exc))

    # ── Análise Técnica ────────────────────────────────────────────────────────
    technical: dict = {}
    try:
        from app.analysis.technical import analyze_from_prices
        if market_prices:
            technical = analyze_from_prices(market_prices)
    except Exception as exc:
        _log.warning("enrichment_technical_error", error=str(exc))

    # ── Métricas de Risco ─────────────────────────────────────────────────────
    risk_metrics: dict = {}
    try:
        from app.analysis.risk import analyze_portfolio
        if market_prices:
            risk_metrics = analyze_portfolio(market_prices)
    except Exception as exc:
        _log.warning("enrichment_risk_error", error=str(exc))

    # ── Monte Carlo ────────────────────────────────────────────────────────────
    monte_carlo: dict = {}
    try:
        from app.analysis.monte_carlo import run_for_portfolio
        if market_prices:
            monte_carlo = run_for_portfolio(market_prices, days=20, n_paths=500)
    except Exception as exc:
        _log.warning("enrichment_mc_error", error=str(exc))

    # ── Cenários Bull/Base/Bear ────────────────────────────────────────────────
    scenarios: dict = {}
    try:
        from app.analysis.scenarios import generate_scenarios
        scenarios = generate_scenarios(
            narrative_label=primary.label,
            narrative_description=primary.description,
            market_prices=market_prices,
            risk_metrics=risk_metrics,
            monte_carlo=monte_carlo,
            run_id=run_id,
        )
        if scenarios:
            sc_path = enrichment_dir / f"{run_id}_scenarios.json"
            import json
            sc_path.write_text(json.dumps(scenarios, ensure_ascii=False, indent=2), encoding="utf-8")
            result.artifact_paths["scenarios"] = str(sc_path)
    except Exception as exc:
        _log.warning("enrichment_scenarios_error", error=str(exc))

    # ── Web Search ────────────────────────────────────────────────────────────
    try:
        from app.providers.web_search import search as web_search
        search_results = web_search(primary.label, max_results=6, timelimit="d")
        if search_results:
            import json
            ws_path = enrichment_dir / f"{run_id}_web_search.json"
            ws_path.write_text(
                json.dumps(search_results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            result.artifact_paths["web_search"] = str(ws_path)
    except Exception as exc:
        _log.warning("enrichment_websearch_error", error=str(exc))

    # ── Diagrama ISQ (Draw.io → HTML) ─────────────────────────────────────────
    try:
        if isq_signal:
            from app.views.logic_diagram import generate_diagram
            diag_path = enrichment_dir / f"{run_id}_isq_diagram.html"
            generate_diagram(isq_signal, diag_path)
            result.artifact_paths["isq_diagram"] = str(diag_path)
    except Exception as exc:
        _log.warning("enrichment_diagram_error", error=str(exc))

    # ── Charts (pyecharts) ────────────────────────────────────────────────────
    try:
        if market_prices:
            from app.analysis.charts import generate_all_charts
            charts_dir = enrichment_dir / "charts"
            chart_paths = generate_all_charts(
                market_prices=market_prices,
                risk_metrics=risk_metrics,
                monte_carlo=monte_carlo,
                output_dir=charts_dir,
            )
            result.artifact_paths.update({f"chart_{k}": v for k, v in chart_paths.items()})
    except Exception as exc:
        _log.warning("enrichment_charts_error", error=str(exc))

    # ── Salva technical/risk/mc como JSON (para uso no HTML report) ───────────
    import json as _json

    if technical:
        try:
            tech_path = enrichment_dir / f"{run_id}_technical.json"
            tech_path.write_text(_json.dumps(technical, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
            result.artifact_paths["technical"] = str(tech_path)
        except Exception as exc:
            _log.warning("enrichment_save_technical_error", error=str(exc))

    if risk_metrics:
        try:
            risk_path = enrichment_dir / f"{run_id}_risk.json"
            risk_path.write_text(_json.dumps(risk_metrics, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
            result.artifact_paths["risk"] = str(risk_path)
        except Exception as exc:
            _log.warning("enrichment_save_risk_error", error=str(exc))

    if monte_carlo:
        try:
            mc_path = enrichment_dir / f"{run_id}_monte_carlo.json"
            mc_path.write_text(_json.dumps(monte_carlo, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
            result.artifact_paths["monte_carlo"] = str(mc_path)
        except Exception as exc:
            _log.warning("enrichment_save_mc_error", error=str(exc))

    # ── Export Excel ──────────────────────────────────────────────────────────
    try:
        from app.views.excel_export import export_to_excel
        xlsx_path = out_dir / f"{run_id}_analysis.xlsx"
        export_to_excel(
            bundle=bundle,
            curation_result=result,
            enrichment={
                "market_prices": market_prices,
                "technical": technical,
                "risk": risk_metrics,
                "scenarios": scenarios,
                "isq_signal": isq_signal,
                "polymarket": bundle.polymarket_markets or [],
                "monte_carlo": monte_carlo,
            },
            output_path=xlsx_path,
        )
        result.artifact_paths["excel"] = str(xlsx_path)
    except Exception as exc:
        _log.warning("enrichment_excel_error", error=str(exc))

    _log.info(
        "enrichment_done",
        run_id=run_id,
        has_isq=isq_signal is not None,
        has_scenarios=bool(scenarios),
        tickers_technical=len(technical),
        charts=len([k for k in result.artifact_paths if k.startswith("chart_")]),
    )


def _save_written(output: object, run_date: str, run_id: str) -> Path:
    from datetime import date as date_type
    try:
        d = date_type.fromisoformat(run_date)
    except ValueError:
        d = datetime.now().date()

    base = workspace.bundles / d.isoformat()
    base.mkdir(parents=True, exist_ok=True)

    # Sempre salva o raw text (com sentinels <<<IMG:path>>>) para o brief conseguir
    # reconstruir a ordem contextual das imagens dentro do texto
    raw_path = base / f"{run_id}_written_{output.mode}_raw.txt"
    try:
        raw_path.write_text(output.text, encoding="utf-8")
    except Exception as exc:
        _log.warning("raw_text_save_error", error=str(exc))

    # Tenta gerar .docx com imagens
    try:
        docx_bytes = to_docx(output.text, output.mode, run_date)
        path = base / f"{run_id}_written_{output.mode}.docx"
        path.write_bytes(docx_bytes)
        return path
    except Exception as exc:
        _log.warning("docx_failed_fallback_txt", error=str(exc))

    # Fallback: .txt
    path = base / f"{run_id}_written_{output.mode}.txt"
    path.write_text(
        f"# [{output.mode.upper()}] {run_date}\n\n{output.text}\n",
        encoding="utf-8",
    )
    return path


def _save_result(result: CurationResult, run_date: str, run_id: str) -> Path:
    from datetime import date as date_type
    try:
        d = date_type.fromisoformat(run_date)
    except ValueError:
        d = datetime.now().date()

    path = workspace.curation_result_path(d, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    _log.info("curation_saved", path=str(path))
    return path
