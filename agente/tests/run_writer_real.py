"""
Roda o writer com o CurationResult + bundle mais recentes em disco.
Uso: python tests/run_writer_real.py [YYYY-MM-DD]
  Se data for fornecida, sobrepõe o run_date (útil para testar outro modo do calendário).
"""
import json
import sys
from pathlib import Path
from app.curation.models import CurationResult
from app.models.daily_ingestion_bundle import DailyIngestionBundle
from app.curation.writer import to_docx, write

# Data override via argumento (ex: 2026-03-27 para testar sexta)
# Tema opcional como segundo argumento (ex: "After Hormuz — o novo regime energético")
date_override = sys.argv[1] if len(sys.argv) > 1 else None
tema_override = sys.argv[2] if len(sys.argv) > 2 else None

bundles_dir = Path(r"C:\Users\rafael bentes\agente-workspace\bundles")

# Curation result mais recente
curation_files = sorted(bundles_dir.rglob("*_curation.json"), key=lambda p: p.stat().st_mtime)
if not curation_files:
    print("Nenhum curation result encontrado.")
    exit(1)

latest_curation = curation_files[-1]
print(f"Curation: {latest_curation.name}\n")
result = CurationResult.model_validate(json.loads(latest_curation.read_text(encoding="utf-8")))
if date_override:
    result.run_date = date_override
    print(f"Data override: {date_override}")

# Bundle correspondente (mesmo run_id)
run_id = result.run_id
bundle_file = latest_curation.parent / f"{run_id}.json"
bundle = None
if bundle_file.exists():
    try:
        bundle = DailyIngestionBundle.model_validate(json.loads(bundle_file.read_text(encoding="utf-8")))
        sg_count = len(bundle.spotgamma_reports)
        dv_count = len([r for r in bundle.rss_items if "DeepVue" in r.source_name])
        print(f"Bundle: SpotGamma={sg_count} reports, DeepVue={dv_count} items")
    except Exception as e:
        print(f"Bundle load error: {e}")
else:
    print("Bundle não encontrado, rodando sem dados estruturados.")

print(f"Narrativa: {result.narrative.primary_signal.label}")
print(f"Confiança: {result.narrative.primary_signal.confidence:.0%}")
print(f"Secundária: {result.narrative.secondary_signals[0].label if result.narrative.secondary_signals else 'nenhuma'}")

# ── Para podcast_sabado: busca ZH Main ao vivo se não está no bundle ──────────
_mode = date_override if date_override and not date_override[0].isdigit() else None
from datetime import date as _date
_run_date = date_override if date_override else (result.run_date if result.run_date else str(_date.today()))
try:
    _d = _date.fromisoformat(_run_date)
except (ValueError, TypeError):
    _d = _date.today()

_is_podcast = (_d.weekday() == 5) or (date_override in ("podcast_sabado",))

if _is_podcast and bundle:
    zh_main_existing = [r for r in bundle.rss_items if r.source_name == "ZeroHedge — Main"]
    if not zh_main_existing:
        print("\n[ZH MAIN] Bundle não tem artigos do ZeroHedge Main — buscando ao vivo...")
        try:
            from playwright.sync_api import sync_playwright
            from app.providers.zerohedge import collect_main_page
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=False)
                ctx = browser.new_context()
                page = ctx.new_page()
                zh_articles = collect_main_page(page, max_articles=5)
                browser.close()
            bundle.rss_items.extend(zh_articles)
            print(f"[ZH MAIN] {len(zh_articles)} artigos adicionados ao bundle:")
            for a in zh_articles:
                print(f"  - {a.title[:70]}")
        except Exception as e:
            print(f"[ZH MAIN] Erro ao buscar ao vivo: {e}")

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

print("\nRodando writer...\n" + "=" * 70)

output = write(result, bundle, tema_hint=tema_override)

print(f"\n[MODO]: {output.mode}")
print(f"[FOCO]: {output.focus}")
print(f"[ÂNGULO]: {output.angle}")
print("\n" + "=" * 70 + "\n")
print(output.text)
print("\n" + "=" * 70)

# Salva como .docx
stem = latest_curation.stem.replace("_curation", "")
date_tag = date_override.replace("-", "") if date_override else ""
suffix = f"_{date_tag}" if date_tag else ""
try:
    docx_bytes = to_docx(output.text, output.mode, result.run_date)
    out_path = latest_curation.parent / f"{stem}_written_{output.mode}{suffix}.docx"
    out_path.write_bytes(docx_bytes)
    print(f"\nSalvo em: {out_path.name}")
except Exception as e:
    print(f"\nErro ao gerar docx ({e}), salvando .txt")
    out_path = latest_curation.parent / f"{stem}_written_{output.mode}{suffix}.txt"
    out_path.write_text(f"# [{output.mode.upper()}] {result.run_date}\n\n{output.text}\n", encoding="utf-8")
    print(f"Salvo em: {out_path.name}")

# ── TTS ───────────────────────────────────────────────────────────────────────
try:
    from app.providers.elevenlabs_tts import generate_audio
    from app.config.settings import settings
    if settings.elevenlabs_api_key:
        print("\n" + "=" * 70)
        print("Gerando áudio (ElevenLabs)...")
        audio_paths = generate_audio(
            text=output.text,
            mode=output.mode,
            out_dir=latest_curation.parent,
            run_id=run_id,
        )
        for k, v in audio_paths.items():
            print(f"  [{k}] → {v}")
    else:
        print("\n[TTS] ELEVENLABS_API_KEY não configurado — pulando.")
except Exception as e:
    print(f"\n[TTS] Erro: {e}")
