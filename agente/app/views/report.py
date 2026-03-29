"""
Gerador de relatorio do DailyIngestionBundle.

Produz:
  - Markdown legivel para revisao humana
  - JSON resumido para consumo programatico
  - HTML estilizado para visualizacao no browser

O relatorio e salvo no workspace/bundles/<data>/ junto ao bundle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from app.models.daily_ingestion_bundle import DailyIngestionBundle
from app.models.market_ear_block import MarketEarBlock
from app.models.x_timeline_item import XTimelineItem
from app.storage.paths import workspace

if TYPE_CHECKING:
    from app.curation.models import CurationResult
    from app.curation.narrative_tracker import NarrativeTrend


def generate_markdown(bundle: DailyIngestionBundle) -> str:
    """Gera o relatorio completo em Markdown."""
    lines: list[str] = []

    # ── Cabecalho ──────────────────────────────────────────────────────────────
    lines += [
        f"# Relatorio de Ingestao — {bundle.run_date}",
        "",
        f"**run_id:** `{bundle.run_id}`  ",
        f"**gerado em:** {bundle.created_at.strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**ZeroHedge blocos:** {len(bundle.market_ear_blocks)}  ",
        f"**X tweets:** {len(bundle.x_items)}  ",
        f"**Erros:** {bundle.audit_summary.errors}  ",
        "",
        "---",
        "",
    ]

    # ── The Market Ear ─────────────────────────────────────────────────────────
    lines += [
        "## The Market Ear (ZeroHedge)",
        "",
    ]

    if not bundle.market_ear_blocks:
        lines.append("_Nenhum bloco coletado._")
        lines.append("")
    else:
        for i, b in enumerate(bundle.market_ear_blocks):
            lines.append(f"### {i + 1}. {b.title or '(sem titulo)'}")
            lines.append("")

            if b.published_at:
                lines.append(f"*{b.published_at.strftime('%Y-%m-%d %H:%M UTC')}*  ")
                lines.append("")

            if b.body_text:
                lines.append(b.body_text)
                lines.append("")

            if b.image_refs:
                lines.append(f"*{len(b.image_refs)} imagem(ns) anexada(s)*")
                lines.append("")

            lines.append("---")
            lines.append("")

    # ── X Timeline ─────────────────────────────────────────────────────────────
    lines += [
        "## X Timeline",
        "",
    ]

    if not bundle.x_items:
        lines.append("_Nenhum tweet coletado._")
        lines.append("")
    else:
        # Ordena por likes desc para destacar os mais relevantes
        sorted_items = sorted(
            bundle.x_items,
            key=lambda it: it.engagement_info.likes,
            reverse=True,
        )
        for it in sorted_items[:30]:  # Top 30 no relatorio
            eng = it.engagement_info
            ts = it.created_at.strftime("%Y-%m-%d %H:%M") if it.created_at else "?"
            lines.append(f"**{it.author}** · {ts}")
            lines.append("")
            if it.text:
                lines.append(f"> {it.text.replace(chr(10), ' ')}")
                lines.append("")
            lines.append(
                f"[Link]({it.url}) · "
                f"Replies: {eng.replies} · "
                f"Reposts: {eng.reposts} · "
                f"Likes: {eng.likes}"
            )
            lines.append("")
            lines.append("---")
            lines.append("")

    # ── Erros ──────────────────────────────────────────────────────────────────
    if bundle.audit_summary.error_messages:
        lines += ["## Erros", ""]
        for msg in bundle.audit_summary.error_messages:
            lines.append(f"- {msg}")
        lines.append("")

    return "\n".join(lines)


def generate_json_summary(bundle: DailyIngestionBundle) -> dict:
    """Gera um JSON resumido (sem body completo) para consumo programatico."""
    return {
        "run_id": bundle.run_id,
        "run_date": str(bundle.run_date),
        "created_at": bundle.created_at.isoformat(),
        "stats": {
            "zh_blocks": len(bundle.market_ear_blocks),
            "x_items": len(bundle.x_items),
            "errors": bundle.audit_summary.errors,
        },
        "zh_headlines": [
            {"title": b.title, "images": len(b.image_refs), "body_chars": len(b.body_text)}
            for b in bundle.market_ear_blocks
        ],
        "x_top10": [
            {
                "author": it.author,
                "text": it.text[:200],
                "url": it.url,
                "likes": it.engagement_info.likes,
                "reposts": it.engagement_info.reposts,
            }
            for it in sorted(bundle.x_items, key=lambda x: x.engagement_info.likes, reverse=True)[:10]
        ],
        "artifact_paths": bundle.artifact_paths,
    }


def _img_src(url_or_path: str) -> str:
    """Retorna URI usavel em <img src>: file:/// para paths locais, URL para remotas."""
    p = Path(url_or_path)
    if p.exists():
        return p.as_uri()
    return url_or_path


def generate_html(
    bundle: DailyIngestionBundle,
    curation_result: "CurationResult | None" = None,
    trend: "NarrativeTrend | None" = None,
    report_dir: "Path | None" = None,
) -> str:
    """Gera o relatorio completo em HTML estilizado, com imagens."""
    from html import escape

    parts: list[str] = []

    # ── CSS + cabecalho ────────────────────────────────────────────────────────
    gerado = bundle.created_at.strftime("%Y-%m-%d %H:%M UTC")
    parts.append(f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Relatorio {bundle.run_date}</title>
<style>
  :root {{
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --text: #e2e8f0; --muted: #8892a4; --accent: #60a5fa;
    --green: #34d399; --red: #f87171; --yellow: #fbbf24;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 15px; line-height: 1.7; padding: 2rem;
    max-width: 960px; margin: 0 auto;
  }}
  h1 {{ font-size: 1.6rem; color: var(--accent); margin-bottom: 0.5rem; }}
  h2 {{ font-size: 1.2rem; color: var(--accent); margin: 2.5rem 0 1rem;
        padding-bottom: 0.4rem; border-bottom: 1px solid var(--border); }}
  h3 {{ font-size: 1rem; color: var(--text); margin: 1.8rem 0 0.5rem; }}
  p {{ margin-bottom: 0.8rem; }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  code {{ background: var(--surface); padding: 0.1em 0.4em;
          border-radius: 4px; font-size: 0.85em; color: var(--yellow); }}
  hr {{ border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }}
  .meta {{
    display: flex; gap: 1.5rem; flex-wrap: wrap;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 2rem;
    font-size: 0.85rem; color: var(--muted);
  }}
  .meta span {{ display: flex; align-items: center; gap: 0.4rem; }}
  .meta strong {{ color: var(--text); }}
  .block {{ margin-bottom: 2rem; }}
  .block-ts {{ font-size: 0.82rem; color: var(--muted); margin-bottom: 0.6rem; }}
  .block-body {{ color: var(--text); white-space: pre-wrap; margin-bottom: 0.8rem; }}
  .imgs {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.8rem 0; }}
  .imgs img {{ max-width: 320px; max-height: 240px; border-radius: 6px;
               border: 1px solid var(--border); object-fit: cover; }}
  .tweet {{ background: var(--surface); border: 1px solid var(--border);
            border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 1rem; }}
  .tweet-author {{ font-weight: 600; color: var(--accent); }}
  .tweet-ts {{ font-size: 0.82rem; color: var(--muted); margin-left: 0.5rem; }}
  .tweet-text {{ margin: 0.5rem 0; color: var(--text); }}
  .tweet-eng {{ font-size: 0.82rem; color: var(--muted); margin-top: 0.5rem; }}
  .tweet-eng a {{ color: var(--muted); }}
  .tweet-media {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.6rem; }}
  .tweet-media img {{ max-width: 280px; max-height: 200px; border-radius: 6px;
                      border: 1px solid var(--border); object-fit: cover; }}
  ul.errors {{ padding-left: 1.4rem; }}
  ul.errors li {{ color: var(--red); margin-bottom: 0.3rem; }}
  .narrative-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 1.2rem 1.4rem; margin-bottom: 1.5rem;
  }}
  .narrative-label {{ font-size: 1.1rem; font-weight: 700; color: var(--accent); }}
  .confidence-bar-wrap {{ display: flex; align-items: center; gap: 0.8rem; margin: 0.5rem 0; }}
  .confidence-bar {{ flex: 1; height: 8px; background: var(--border); border-radius: 4px; }}
  .confidence-fill {{ height: 100%; border-radius: 4px; background: var(--green); }}
  .verdict-badge {{
    display: inline-block; padding: 0.15em 0.6em; border-radius: 999px;
    font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
  }}
  .verdict-pass {{ background: #064e3b; color: var(--green); }}
  .verdict-warn {{ background: #451a03; color: var(--yellow); }}
  .verdict-fail {{ background: #450a0a; color: var(--red); }}
  .score-badge {{
    display: inline-block; padding: 0.1em 0.45em; border-radius: 4px;
    font-size: 0.78rem; font-weight: 600; color: #000;
    margin-left: 0.5rem;
  }}
  .score-high {{ background: var(--green); }}
  .score-mid  {{ background: var(--yellow); }}
  .score-low  {{ background: #64748b; color: #fff; }}
  .trend-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 0.8rem; }}
  .trend-table th {{ text-align: left; color: var(--muted); font-weight: 500;
                     padding: 0.3rem 0.6rem; border-bottom: 1px solid var(--border); }}
  .trend-table td {{ padding: 0.35rem 0.6rem; border-bottom: 1px solid var(--border); color: var(--text); }}
  .trend-table tr.today td {{ color: var(--accent); font-weight: 600; }}
  .trend-badge {{ display: inline-block; padding: 0.15em 0.55em; border-radius: 999px;
                  font-size: 0.75rem; font-weight: 600; }}
  .trend-persisting {{ background: #064e3b; color: var(--green); }}
  .trend-evolving   {{ background: #1e3a5f; color: #93c5fd; }}
  .trend-reversed   {{ background: #451a03; color: var(--yellow); }}
  .trend-new        {{ background: #2d1b69; color: #c4b5fd; }}
  .conf-bar {{ display: inline-block; width: 60px; height: 6px; background: var(--border);
               border-radius: 3px; vertical-align: middle; margin-left: 4px; }}
  .conf-fill {{ height: 100%; border-radius: 3px; background: var(--green); }}
  .quote {{ border-left: 3px solid var(--accent); padding: 0.4rem 0.8rem;
           margin: 0.4rem 0; color: var(--muted); font-style: italic; font-size: 0.9rem; }}
  /* ── Enrichment ── */
  .data-table {{ width:100%; border-collapse:collapse; font-size:0.85rem; margin:0.8rem 0; }}
  .data-table th {{ text-align:left; color:var(--muted); font-weight:500; padding:0.35rem 0.6rem;
                    border-bottom:1px solid var(--border); white-space:nowrap; }}
  .data-table td {{ padding:0.35rem 0.6rem; border-bottom:1px solid var(--border); color:var(--text); }}
  .data-table tr:hover td {{ background:var(--surface); }}
  .pos {{ color:var(--green); font-weight:600; }}
  .neg {{ color:var(--red); font-weight:600; }}
  .neu {{ color:var(--muted); }}
  .scenario-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin:1rem 0; }}
  .scenario-card {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:1rem; }}
  .scenario-card.bull {{ border-color:#34d399; }}
  .scenario-card.base {{ border-color:#fbbf24; }}
  .scenario-card.bear {{ border-color:#f87171; }}
  .scenario-label {{ font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em; margin-bottom:.3rem; }}
  .scenario-card.bull .scenario-label {{ color:var(--green); }}
  .scenario-card.base .scenario-label {{ color:var(--yellow); }}
  .scenario-card.bear .scenario-label {{ color:var(--red); }}
  .scenario-prob {{ font-size:1.3rem; font-weight:700; margin-bottom:.4rem; }}
  .scenario-catalyst {{ font-size:0.82rem; font-weight:600; color:var(--text); margin-bottom:.4rem; }}
  .scenario-narrative {{ font-size:0.82rem; color:var(--muted); line-height:1.5; }}
  .scenario-spx {{ font-size:0.78rem; color:var(--accent); margin-top:.4rem; }}
  .chain-node {{ display:flex; align-items:flex-start; gap:.8rem; margin:.5rem 0; }}
  .chain-dot {{ width:10px; height:10px; border-radius:50%; flex-shrink:0; margin-top:.35rem; }}
  .chain-dot.positive {{ background:var(--green); }}
  .chain-dot.negative {{ background:var(--red); }}
  .chain-dot.neutral {{ background:var(--muted); }}
  .chain-text {{ font-size:0.85rem; }}
  .chain-node-name {{ font-weight:600; }}
  .chain-arrow {{ color:var(--muted); font-size:0.78rem; padding:.2rem 0 .2rem 1.8rem; }}
  .ticker-grid {{ display:flex; flex-wrap:wrap; gap:.5rem; margin:.6rem 0; }}
  .ticker-chip {{ display:inline-flex; align-items:center; gap:.35rem; padding:.2rem .6rem;
                  border-radius:4px; font-size:0.8rem; border:1px solid var(--border); }}
  .ticker-chip.long {{ border-color:var(--green); color:var(--green); }}
  .ticker-chip.short {{ border-color:var(--red); color:var(--red); }}
  .ticker-chip.neutral {{ border-color:var(--muted); color:var(--muted); }}
  .poly-row-high {{ color:var(--green); }}
  .poly-row-low {{ color:var(--red); }}
  .section-comment {{ background:var(--surface); border-left:3px solid var(--accent);
                      border-radius:0 6px 6px 0; padding:.7rem 1rem; margin:.8rem 0;
                      font-size:0.88rem; color:var(--muted); line-height:1.6; }}
</style>
</head>
<body>""")

    # ── Meta ──────────────────────────────────────────────────────────────────
    sg = bundle.spotgamma_reports
    rss = bundle.rss_items
    spectra_n = sum(1 for r in rss if "Spectra" in r.source_name)
    deepvue_n = sum(1 for r in rss if "DeepVue" in r.source_name)
    feed_n = sum(1 for r in rss if "Spectra" not in r.source_name and "DeepVue" not in r.source_name)
    total = len(bundle.market_ear_blocks) + len(bundle.x_items) + len(sg) + len(rss)

    parts.append(f"""
<h1>Relatorio de Ingestao &mdash; {bundle.run_date}</h1>
<div class="meta">
  <span>&#128196; run_id: <strong><code>{bundle.run_id[:20]}...</code></strong></span>
  <span>&#128336; <strong>{gerado}</strong></span>
  <span>&#128200; ZeroHedge: <strong>{len(bundle.market_ear_blocks)} blocos</strong></span>
  <span>&#128038; X tweets: <strong>{len(bundle.x_items)}</strong></span>
  <span>&#127381; SpotGamma: <strong>{len(sg)} reports</strong></span>
  <span>&#127758; Spectra: <strong>{spectra_n}</strong></span>
  <span>&#128202; DeepVue: <strong>{deepvue_n}</strong></span>
  <span>&#128240; RSS: <strong>{feed_n} itens</strong></span>
  <span>&#8721; Total: <strong>{total}</strong></span>
  <span>&#9888; Erros: <strong>{bundle.audit_summary.errors}</strong></span>
</div>""")

    # ── Curação ───────────────────────────────────────────────────────────────
    if curation_result is not None:
        parts.append(_render_curation_html(curation_result, escape))
    if trend is not None:
        parts.append(_render_trend_html(trend, escape))

    # ── Áudio (ElevenLabs TTS) ────────────────────────────────────────────────
    if curation_result is not None:
        ap = curation_result.artifact_paths
        audio_principal = ap.get("audio_principal")
        audio_gratuito = ap.get("audio_gratuito")
        if audio_principal or audio_gratuito:
            parts.append("<h2>Áudio</h2>")
            if audio_principal:
                rel = _rel_audio(audio_principal, report_dir)
                parts.append(
                    f'<div style="margin-bottom:1.2rem">'
                    f'<p style="font-size:0.85rem;color:var(--muted);margin-bottom:0.4rem">Texto Principal (assinantes)</p>'
                    f'<audio controls style="width:100%;accent-color:var(--accent)">'
                    f'<source src="{rel}" type="audio/mpeg">Seu browser não suporta áudio.</audio>'
                    f'</div>'
                )
            if audio_gratuito:
                rel = _rel_audio(audio_gratuito, report_dir)
                parts.append(
                    f'<div style="margin-bottom:1.2rem">'
                    f'<p style="font-size:0.85rem;color:var(--muted);margin-bottom:0.4rem">Texto Gratuito (público)</p>'
                    f'<audio controls style="width:100%;accent-color:var(--accent)">'
                    f'<source src="{rel}" type="audio/mpeg">Seu browser não suporta áudio.</audio>'
                    f'</div>'
                )

    # ── Enrichment (Mercado, Técnicos, Risco, Cenários, ISQ, Polymarket, MC, Charts) ──
    if curation_result is not None:
        parts.append(_render_enrichment_html(curation_result, bundle, escape, report_dir))

    # ── ZeroHedge ─────────────────────────────────────────────────────────────
    parts.append("<h2>The Market Ear (ZeroHedge)</h2>")
    scored_items = curation_result.scored_items if curation_result else []

    if not bundle.market_ear_blocks:
        parts.append("<p><em>Nenhum bloco coletado.</em></p>")
    else:
        for i, b in enumerate(bundle.market_ear_blocks):
            me_id = f"ME-{b.id[:8]}"
            badge = _score_badge_html(me_id, scored_items)
            parts.append(f'<div class="block">')
            parts.append(f"<h3>{i + 1}. {escape(b.title or '(sem titulo)')}{badge}</h3>")
            if b.published_at:
                parts.append(f'<div class="block-ts">{b.published_at.strftime("%Y-%m-%d %H:%M UTC")}</div>')
            if b.body_text:
                parts.append(f'<div class="block-body">{escape(b.body_text)}</div>')
            # Imagens — deduplica mantendo ordem
            seen: set[str] = set()
            unique_imgs = [u for u in b.image_refs if u not in seen and not seen.add(u)]  # type: ignore[func-returns-value]
            if unique_imgs:
                parts.append('<div class="imgs">')
                for url in unique_imgs:
                    parts.append(f'<img src="{escape(_img_src(url))}" loading="lazy" alt="">')
                parts.append("</div>")
            parts.append("<hr></div>")

    # ── X Timeline ────────────────────────────────────────────────────────────
    parts.append("<h2>X Timeline</h2>")
    if not bundle.x_items:
        parts.append("<p><em>Nenhum tweet coletado.</em></p>")
    else:
        sorted_items = sorted(bundle.x_items, key=lambda it: it.engagement_info.likes, reverse=True)
        for it in sorted_items[:30]:
            eng = it.engagement_info
            ts = it.created_at.strftime("%Y-%m-%d %H:%M") if it.created_at else "?"
            x_id = f"X-{it.id[:8]}"
            badge = _score_badge_html(x_id, scored_items)
            parts.append(f'<div class="tweet">')
            parts.append(
                f'<span class="tweet-author">{escape(it.author)}</span>'
                f'<span class="tweet-ts">{ts}</span>{badge}'
            )
            if it.text:
                parts.append(f'<div class="tweet-text">{escape(it.text)}</div>')
            if it.media_refs:
                parts.append('<div class="tweet-media">')
                for url in it.media_refs[:4]:
                    parts.append(f'<img src="{escape(_img_src(url))}" loading="lazy" alt="">')
                parts.append("</div>")
            parts.append(
                f'<div class="tweet-eng">'
                f'<a href="{escape(it.url)}" target="_blank">Link</a> &nbsp;'
                f'&#128172; {eng.replies} &nbsp;'
                f'&#128257; {eng.reposts} &nbsp;'
                f'&#10084; {eng.likes}'
                f"</div>"
            )
            parts.append("</div>")

    # ── SpotGamma ─────────────────────────────────────────────────────────────
    if bundle.spotgamma_reports:
        parts.append("<h2>SpotGamma</h2>")
        for rpt in bundle.spotgamma_reports:
            date_str = str(rpt.report_date) if rpt.report_date else ""
            parts.append(f'<div class="block">')
            parts.append(f'<h3>{escape(rpt.report_type)}: {escape(rpt.title)}</h3>')
            if date_str:
                parts.append(f'<div class="block-ts">{date_str}</div>')
            if rpt.raw_text:
                preview = rpt.raw_text[:2000]
                parts.append(f'<div class="block-body">{escape(preview)}{"…" if len(rpt.raw_text) > 2000 else ""}</div>')
            parts.append("<hr></div>")

    # ── Spectra Markets ───────────────────────────────────────────────────────
    spectra_items = [r for r in bundle.rss_items if "Spectra" in r.source_name]
    if spectra_items:
        parts.append("<h2>Spectra Markets</h2>")
        for it in spectra_items:
            parts.append(f'<div class="block">')
            parts.append(f'<h3><a href="{escape(it.url)}" target="_blank">{escape(it.title)}</a></h3>')
            if it.summary:
                parts.append(f'<div class="block-body">{escape(it.summary[:1500])}</div>')
            parts.append("<hr></div>")

    # ── DeepVue ───────────────────────────────────────────────────────────────
    deepvue_items = [r for r in bundle.rss_items if "DeepVue" in r.source_name]
    if deepvue_items:
        parts.append("<h2>DeepVue — Theme Tracker</h2>")
        for it in deepvue_items:
            parts.append(f'<div class="block">')
            parts.append(f'<div class="block-body" style="font-family:monospace;font-size:0.85rem">{escape(it.summary)}</div>')
            parts.append("<hr></div>")

    # ── RSS Feeds ─────────────────────────────────────────────────────────────
    feed_items = [r for r in bundle.rss_items if "Spectra" not in r.source_name and "DeepVue" not in r.source_name]
    if feed_items:
        parts.append(f"<h2>RSS Feeds ({len(feed_items)} itens)</h2>")
        # Agrupa por source
        from collections import defaultdict
        by_source: dict[str, list] = defaultdict(list)
        for it in feed_items:
            by_source[it.source_name].append(it)
        for source, items in by_source.items():
            parts.append(f'<h3>{escape(source)}</h3>')
            for it in items[:5]:
                ts = it.published_at.strftime("%Y-%m-%d %H:%M") if it.published_at else ""
                parts.append(
                    f'<div style="margin-bottom:0.6rem">'
                    f'<a href="{escape(it.url)}" target="_blank"><strong>{escape(it.title)}</strong></a>'
                    f'{"  <span style=\'color:var(--muted);font-size:0.8rem\'>" + ts + "</span>" if ts else ""}'
                    f'{"<p style=\'color:var(--muted);font-size:0.85rem;margin-top:0.2rem\'>" + escape(it.summary[:200]) + "</p>" if it.summary else ""}'
                    f'</div>'
                )
            parts.append("<hr>")

    # ── Erros ─────────────────────────────────────────────────────────────────
    if bundle.audit_summary.error_messages:
        parts.append("<h2>Erros</h2><ul class='errors'>")
        for msg in bundle.audit_summary.error_messages:
            parts.append(f"<li>{escape(msg)}</li>")
        parts.append("</ul>")

    parts.append("</body></html>")
    return "\n".join(parts)


def _render_curation_html(curation_result: "CurationResult", escape) -> str:  # type: ignore[type-arg]
    parts: list[str] = ["<h2>Narrativa do Dia (Curation LLM)</h2>"]
    narrative = curation_result.narrative

    def _signal_card(sig, title: str) -> str:
        if sig is None:
            return ""
        conf_pct = int(sig.confidence * 100)
        status_note = ""
        if sig.status == "inconclusive":
            status_note = f'<span style="color:var(--yellow);font-size:0.82rem"> ⚠ {escape(sig.inconclusive_reason or "inconclusive")}</span>'

        quotes_html = "".join(
            f'<div class="quote">{escape(q)}</div>' for q in sig.evidence_quotes[:3]
        )
        return (
            f'<div class="narrative-card">'
            f'<div class="narrative-label">{escape(title)}: {escape(sig.label)}</div>'
            f'{status_note}'
            f'<div class="confidence-bar-wrap">'
            f'  <div class="confidence-bar"><div class="confidence-fill" style="width:{conf_pct}%"></div></div>'
            f'  <span style="font-size:0.85rem;color:var(--muted)">Confiança: <strong style="color:var(--text)">{sig.confidence:.0%}</strong></span>'
            f'</div>'
            f'<p style="color:var(--muted);font-size:0.9rem">{escape(sig.description)}</p>'
            f'{quotes_html}'
            f'</div>'
        )

    parts.append(_signal_card(narrative.primary_signal, "Narrativa Primária"))
    for sec in narrative.secondary_signals:
        parts.append(_signal_card(sec, "Narrativa Secundária"))

    # Verification badge
    v = curation_result.verification
    verdict_class = {"pass": "verdict-pass", "warn": "verdict-warn", "fail": "verdict-fail"}.get(
        v.overall_verdict, "verdict-warn"
    )
    verdict_label = {"pass": "✓ Verificado", "warn": "⚠ Fraco", "fail": "✗ Alucinação"}.get(
        v.overall_verdict, v.overall_verdict
    )
    parts.append(
        f'<p>Verificação (Haiku): <span class="verdict-badge {verdict_class}">{verdict_label}</span> '
        f'<span style="font-size:0.82rem;color:var(--muted)">modelo: {escape(v.verification_model)}</span></p>'
    )

    return "\n".join(parts)


def _render_trend_html(trend: "NarrativeTrend", escape) -> str:  # type: ignore[type-arg]
    if not trend.entries:
        return ""

    trend_cls = {
        "persisting": "trend-persisting",
        "evolving": "trend-evolving",
        "reversed": "trend-reversed",
        "new": "trend-new",
        "unknown": "",
    }.get(trend.trend, "")

    trend_label = {
        "persisting": "↻ Persistindo",
        "evolving": "→ Evoluindo",
        "reversed": "⇄ Invertida",
        "new": "★ Nova",
        "unknown": "?",
    }.get(trend.trend, "")

    today_date = trend.today.date if trend.today else ""

    rows = ""
    for e in trend.entries:
        is_today = e.date == today_date
        cls = ' class="today"' if is_today else ""
        conf_pct = int(e.confidence * 100)
        verdict_icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}.get(e.verdict, "?")
        verdict_color = {"pass": "color:var(--green)", "warn": "color:var(--yellow)",
                         "fail": "color:var(--red)"}.get(e.verdict, "")
        rows += (
            f'<tr{cls}>'
            f'<td>{e.date}</td>'
            f'<td>{escape(e.label[:55])}</td>'
            f'<td><span class="conf-bar"><span class="conf-fill" style="width:{conf_pct}%"></span></span>'
            f' {conf_pct}%</td>'
            f'<td style="{verdict_color}">{verdict_icon}</td>'
            f'</tr>'
        )

    badge = f'<span class="trend-badge {trend_cls}">{trend_label}</span>' if trend_cls else ""
    note = f' <span style="font-size:0.8rem;color:var(--muted)">{escape(trend.trend_note)}</span>'

    return (
        f'<h2>Histórico de Narrativas (7 dias)</h2>'
        f'<div class="narrative-card">'
        f'{badge}{note}'
        f'<table class="trend-table"><thead>'
        f'<tr><th>Data</th><th>Narrativa</th><th>Confiança</th><th>OK</th></tr>'
        f'</thead><tbody>{rows}</tbody></table>'
        f'</div>'
    )


def _score_badge_html(item_id: str, scored_items: list) -> str:
    """Returns score badge HTML for a given item_id, or empty string."""
    for s in scored_items:
        if s.item_id == item_id:
            score = s.narrative_relevance
            if score >= 0.65:
                cls = "score-high"
            elif score >= 0.35:
                cls = "score-mid"
            else:
                return ""  # low relevance — don't clutter
            return f'<span class="score-badge {cls}">{score:.0%}</span>'
    return ""


def _rel_audio(abs_path_str: str, report_dir: "Path | None") -> str:
    """Retorna caminho relativo para o arquivo de áudio (ou URI absoluta)."""
    p = Path(abs_path_str)
    if report_dir and p.exists():
        try:
            return str(p.relative_to(report_dir)).replace("\\", "/")
        except ValueError:
            pass
    return p.as_uri() if p.exists() else ""


def _render_enrichment_html(
    curation_result: "CurationResult",
    bundle: DailyIngestionBundle,
    escape,  # type: ignore[type-arg]
    report_dir: "Path | None" = None,
) -> str:
    """Renderiza seção de enrichment: preços, técnicos, risco, cenários, ISQ, Polymarket, MC, charts."""
    import json as _json

    parts: list[str] = []
    ap = curation_result.artifact_paths if curation_result else {}

    def _load_json(key: str) -> dict:
        p = ap.get(key)
        if not p:
            return {}
        try:
            return _json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _rel_path(abs_path_str: str) -> str:
        """Retorna caminho relativo a report_dir, ou URI absoluta."""
        if not abs_path_str:
            return ""
        p = Path(abs_path_str)
        if report_dir and p.exists():
            try:
                return str(p.relative_to(report_dir)).replace("\\", "/")
            except ValueError:
                return p.as_uri()
        return p.as_uri() if p.exists() else ""

    def _fmt_pct(v, *, na: str = "—") -> str:
        if v is None:
            return f'<span class="neu">{na}</span>'
        fv = float(v) * 100
        cls = "pos" if fv >= 0 else "neg"
        sign = "+" if fv >= 0 else ""
        return f'<span class="{cls}">{sign}{fv:.2f}%</span>'

    def _fmt_risk_pct(v) -> str:
        if v is None:
            return '<span class="neu">—</span>'
        fv = float(v) * 100
        cls = "neg" if fv < -3 else "pos" if fv > -1 else "neu"
        return f'<span class="{cls}">{fv:.2f}%</span>'

    # ── Market Prices ──────────────────────────────────────────────────────────
    market_prices = bundle.market_prices or {}
    if market_prices:
        parts.append("<h2>Mercado &amp; Enriquecimento</h2>")
        parts.append("<h3>Preços de Mercado</h3>")
        parts.append(
            '<div class="section-comment">Retornos diário, semanal e YTD dos principais ativos. '
            "Dados coletados via yfinance no início do pipeline.</div>"
        )
        parts.append(
            '<table class="data-table"><thead><tr>'
            "<th>Ativo</th><th>Preço</th><th>1D</th><th>1W</th><th>YTD</th>"
            "</tr></thead><tbody>"
        )
        for sym, data in market_prices.items():
            price = data.get("price") or data.get("close")
            name = data.get("name", sym)
            price_str = f"{float(price):.2f}" if price is not None else "—"
            parts.append(
                f"<tr>"
                f"<td><strong>{escape(sym)}</strong>"
                f" <span style='color:var(--muted);font-size:0.8rem'>{escape(name)}</span></td>"
                f"<td>{price_str}</td>"
                f"<td>{_fmt_pct(data.get('daily_return'))}</td>"
                f"<td>{_fmt_pct(data.get('weekly_return'))}</td>"
                f"<td>{_fmt_pct(data.get('ytd_return'))}</td>"
                f"</tr>"
            )
        parts.append("</tbody></table>")

    # ── Technical Analysis ─────────────────────────────────────────────────────
    technical = _load_json("technical")
    if technical:
        parts.append("<h3>Análise Técnica (60d)</h3>")
        parts.append(
            '<div class="section-comment">'
            "RSI(14): abaixo de 30 = sobrevendido (bullish), acima de 70 = sobrecomprado (bearish). "
            "MACD Histograma positivo = momentum de alta. "
            "Sinal gerado combinando RSI + posição nas Bandas de Bollinger(20,2).</div>"
        )
        parts.append(
            '<table class="data-table"><thead><tr>'
            "<th>Ticker</th><th>RSI(14)</th><th>MACD Hist.</th><th>BB Width</th><th>Signal</th>"
            "</tr></thead><tbody>"
        )
        for sym, t in technical.items():
            rsi_val = t.get("rsi")
            macd_data = t.get("macd", {})
            macd_hist = macd_data.get("histogram") if isinstance(macd_data, dict) else None
            bb_data = t.get("bollinger", {})
            bb_width = bb_data.get("width") if isinstance(bb_data, dict) else None
            signal = str(t.get("signal", "hold")).lower()

            rsi_str = f"{float(rsi_val):.1f}" if rsi_val is not None else "—"
            rsi_cls = "neg" if (rsi_val and float(rsi_val) > 70) else "pos" if (rsi_val and float(rsi_val) < 30) else "neu"
            macd_str = f"{float(macd_hist):+.4f}" if macd_hist is not None else "—"
            macd_cls = "pos" if (macd_hist and float(macd_hist) > 0) else "neg" if (macd_hist and float(macd_hist) < 0) else "neu"
            bb_str = f"{float(bb_width):.4f}" if bb_width is not None else "—"
            sig_cls = {"buy": "pos", "sell": "neg", "hold": "neu"}.get(signal, "neu")

            parts.append(
                f"<tr>"
                f"<td><strong>{escape(sym)}</strong></td>"
                f"<td><span class='{rsi_cls}'>{rsi_str}</span></td>"
                f"<td><span class='{macd_cls}'>{macd_str}</span></td>"
                f"<td>{bb_str}</td>"
                f"<td><span class='{sig_cls}'>{escape(signal.upper())}</span></td>"
                f"</tr>"
            )
        parts.append("</tbody></table>")

    # ── Risk Metrics ───────────────────────────────────────────────────────────
    risk = _load_json("risk")
    risk_tickers = risk.get("tickers", {}) if risk else {}
    if risk_tickers:
        parts.append("<h3>Métricas de Risco (60d)</h3>")
        parts.append(
            '<div class="section-comment">'
            "VaR 95%: perda máxima esperada em 95% dos dias. CVaR: perda média nos piores 5% dos dias. "
            "Max Drawdown: maior queda do pico ao vale. Sharpe &gt; 1 = bom retorno ajustado ao risco.</div>"
        )
        parts.append(
            '<table class="data-table"><thead><tr>'
            "<th>Ticker</th><th>VaR 95%</th><th>CVaR 95%</th><th>Max DD</th><th>Sharpe</th>"
            "</tr></thead><tbody>"
        )
        for sym, m in risk_tickers.items():
            sharpe = m.get("sharpe")
            sharpe_fv = float(sharpe) if sharpe is not None else None
            sharpe_cls = "pos" if (sharpe_fv and sharpe_fv > 1) else "neg" if (sharpe_fv and sharpe_fv < 0) else "neu"
            sharpe_str = f"{sharpe_fv:.2f}" if sharpe_fv is not None else "—"

            parts.append(
                f"<tr>"
                f"<td><strong>{escape(sym)}</strong></td>"
                f"<td>{_fmt_risk_pct(m.get('var_95'))}</td>"
                f"<td>{_fmt_risk_pct(m.get('cvar_95'))}</td>"
                f"<td>{_fmt_risk_pct(m.get('max_drawdown'))}</td>"
                f"<td><span class='{sharpe_cls}'>{sharpe_str}</span></td>"
                f"</tr>"
            )
        parts.append("</tbody></table>")

    # ── Scenarios ─────────────────────────────────────────────────────────────
    scenarios = _load_json("scenarios")
    if scenarios:
        bull = scenarios.get("bull", {})
        base = scenarios.get("base", {})
        bear = scenarios.get("bear", {})
        narrative_label = scenarios.get("narrative", "")

        parts.append("<h3>Cenários Bull / Base / Bear</h3>")
        parts.append(
            f'<div class="section-comment">Cenários gerados por LLM para: '
            f"<em>{escape(narrative_label)}</em>. "
            "As probabilidades são estimativas qualitativas baseadas na narrativa — não são previsões de mercado.</div>"
        )
        parts.append('<div class="scenario-grid">')
        for label, sc, cls in [("BULL", bull, "bull"), ("BASE", base, "base"), ("BEAR", bear, "bear")]:
            prob = sc.get("probability", 0)
            catalyst = sc.get("catalyst", "")
            narrative_text = sc.get("narrative", "")
            spx = sc.get("spx_target")
            horizon = sc.get("time_horizon", "")
            spx_line = ""
            if spx:
                spx_line = f'<div class="scenario-spx">SPX target: {int(spx):,} &nbsp;·&nbsp; {escape(horizon)}</div>'
            parts.append(
                f'<div class="scenario-card {cls}">'
                f'<div class="scenario-label">{label}</div>'
                f'<div class="scenario-prob">{prob:.0%}</div>'
                f'<div class="scenario-catalyst">{escape(catalyst)}</div>'
                f'<div class="scenario-narrative">{escape(narrative_text[:300])}</div>'
                f"{spx_line}"
                f"</div>"
            )
        parts.append("</div>")

    # ── ISQ Signal Qualification ───────────────────────────────────────────────
    isq = _load_json("isq")
    if isq:
        title = isq.get("title", "")
        sentiment = isq.get("sentiment_score", 0)
        intensity = isq.get("intensity", 0)
        confidence = isq.get("confidence", 0)
        reasoning = isq.get("reasoning", "")
        chain = isq.get("transmission_chain", [])
        tickers = isq.get("impact_tickers", [])

        sent_cls = "pos" if float(sentiment) > 0.1 else "neg" if float(sentiment) < -0.1 else "neu"
        parts.append("<h3>ISQ — Investment Signal Qualification</h3>")
        parts.append(
            f'<div class="narrative-card">'
            f'<div class="narrative-label">{escape(title)}</div>'
            f'<div style="margin:.5rem 0;font-size:0.85rem;color:var(--muted)">'
            f'Sentimento: <span class="{sent_cls}"><strong>{float(sentiment):+.2f}</strong></span>'
            f" &nbsp; Intensidade: <strong>{intensity}/5</strong>"
            f" &nbsp; Confiança: <strong>{float(confidence):.0%}</strong>"
            f"</div>"
        )
        if reasoning:
            parts.append(f'<div class="section-comment">{escape(reasoning)}</div>')

        if chain:
            parts.append('<div style="margin:1rem 0">')
            for i, node in enumerate(chain):
                node_name = node.get("node", "")
                impact = node.get("impact", "neutral")
                node_reason = node.get("reasoning", "")
                reason_html = (
                    f"<br><span style='font-size:0.78rem;color:var(--muted)'>{escape(node_reason)}</span>"
                    if node_reason else ""
                )
                parts.append(
                    f'<div class="chain-node">'
                    f'<div class="chain-dot {impact}"></div>'
                    f'<div class="chain-text">'
                    f'<span class="chain-node-name">{escape(node_name)}</span>'
                    f"{reason_html}"
                    f"</div></div>"
                )
                if i < len(chain) - 1:
                    parts.append('<div class="chain-arrow">↓</div>')
            parts.append("</div>")

        if tickers:
            parts.append('<div class="ticker-grid">')
            for t in tickers:
                tk = t.get("ticker", "")
                tk_name = t.get("name", tk)
                direction = t.get("direction", "neutral")
                weight = t.get("weight", 0)
                t_reason = escape(t.get("reasoning", ""))
                dir_arrow = {"long": "↑", "short": "↓", "neutral": "→"}.get(direction, "→")
                parts.append(
                    f'<div class="ticker-chip {direction}" title="{t_reason}">'
                    f"{dir_arrow} <strong>{escape(tk)}</strong> "
                    f"<span style='font-size:0.75rem'>{escape(tk_name[:20])}</span> "
                    f"<span style='font-size:0.75rem;opacity:0.7'>{float(weight):.0%}</span>"
                    f"</div>"
                )
            parts.append("</div>")
        parts.append("</div>")

    # ── Polymarket ─────────────────────────────────────────────────────────────
    poly = bundle.polymarket_markets or []
    if poly:
        parts.append("<h3>Polymarket — Mercados de Predição Macro</h3>")
        parts.append(
            '<div class="section-comment">'
            "Probabilidades implícitas de mercados de predição para eventos macro. "
            "Verde &gt; 60% (consensus), vermelho &lt; 30% (contra-consenso). "
            "Volume USD indica liquidez e nível de confiança do mercado.</div>"
        )
        parts.append(
            '<table class="data-table"><thead><tr>'
            "<th>Evento</th><th>Prob.</th><th>Volume (USD)</th><th>Vencimento</th>"
            "</tr></thead><tbody>"
        )
        for m in poly[:15]:
            q = m.get("question", "")
            prob = m.get("probability")
            vol = m.get("volume_usd")
            end = str(m.get("end_date", ""))[:10]
            prob_str = f"{float(prob):.0%}" if prob is not None else "—"
            prob_cls = "pos" if (prob and float(prob) > 0.6) else "neg" if (prob and float(prob) < 0.3) else "neu"
            vol_str = f"${float(vol):,.0f}" if vol else "—"
            parts.append(
                f"<tr>"
                f"<td>{escape(q[:90])}</td>"
                f"<td><span class='{prob_cls}'><strong>{prob_str}</strong></span></td>"
                f"<td style='font-size:0.82rem'>{vol_str}</td>"
                f"<td style='font-size:0.82rem;color:var(--muted)'>{escape(end)}</td>"
                f"</tr>"
            )
        parts.append("</tbody></table>")

    # ── Monte Carlo Summary ────────────────────────────────────────────────────
    mc = _load_json("monte_carlo")
    if mc:
        parts.append("<h3>Monte Carlo — GBM (20 dias, 500 caminhos)</h3>")
        parts.append(
            '<div class="section-comment">'
            "Simulação Browniana Geométrica com 500 caminhos e horizonte de 20 dias úteis. "
            "<em>Prob Up</em>: % de simulações com retorno positivo. "
            "<em>+5% / −5%</em>: probabilidade de move maior que 5% para cada lado. "
            "P50 mostra o retorno mediano esperado ao final dos 20 dias.</div>"
        )
        parts.append(
            '<table class="data-table"><thead><tr>'
            "<th>Ticker</th><th>Preço Atual</th><th>Prob Up</th><th>Prob +5%</th><th>Prob −5%</th><th>P50 (20d)</th>"
            "</tr></thead><tbody>"
        )
        for sym, mc_data in mc.items():
            if not isinstance(mc_data, dict):
                continue
            current_price = mc_data.get("current_price")
            prob_up = mc_data.get("prob_up")
            prob_up5 = mc_data.get("prob_up_5pct")
            prob_dn5 = mc_data.get("prob_down_5pct")
            final_dist = mc_data.get("final_distribution", {})
            p50_price = final_dist.get("mean") if isinstance(final_dist, dict) else None

            price_str = f"{float(current_price):.2f}" if current_price is not None else "—"
            p50_ret_str = "—"
            if p50_price is not None and current_price:
                p50_ret = (float(p50_price) - float(current_price)) / float(current_price)
                p50_cls = "pos" if p50_ret >= 0 else "neg"
                sign = "+" if p50_ret >= 0 else ""
                p50_ret_str = f'<span class="{p50_cls}">{sign}{p50_ret * 100:.2f}%</span>'

            def _prob_cell(v) -> str:
                if v is None:
                    return '<span class="neu">—</span>'
                fv = float(v)
                cls = "pos" if fv >= 0.5 else "neg"
                return f'<span class="{cls}">{fv:.0%}</span>'

            parts.append(
                f"<tr>"
                f"<td><strong>{escape(sym)}</strong></td>"
                f"<td>{price_str}</td>"
                f"<td>{_prob_cell(prob_up)}</td>"
                f"<td>{_prob_cell(prob_up5)}</td>"
                f"<td>{_prob_cell(prob_dn5)}</td>"
                f"<td>{p50_ret_str}</td>"
                f"</tr>"
            )
        parts.append("</tbody></table>")

    # ── Charts Interativos ─────────────────────────────────────────────────────
    chart_keys = {k: v for k, v in ap.items() if k.startswith("chart_")}
    if chart_keys:
        parts.append("<h3>Charts Interativos</h3>")
        parts.append(
            '<div class="section-comment">'
            "Charts gerados via pyecharts. Interativos: zoom, tooltip, legenda clicável.</div>"
        )

        klines = {k: v for k, v in chart_keys.items() if "kline_" in k}
        mc_fans = {k: v for k, v in chart_keys.items() if "mc_fan_" in k}
        others = {k: v for k, v in chart_keys.items() if "kline_" not in k and "mc_fan_" not in k}

        def _iframe_section(title: str, abs_path: str, height: int = 440) -> str:
            rel = _rel_path(abs_path)
            if not rel:
                return ""
            return (
                f'<div style="margin:1rem 0">'
                f'<div style="font-size:0.85rem;color:var(--muted);margin-bottom:.3rem">{escape(title)}</div>'
                f'<iframe src="{rel}" width="100%" height="{height}" frameborder="0" '
                f'style="border:1px solid var(--border);border-radius:6px;background:#fff">'
                f"</iframe></div>"
            )

        if klines:
            parts.append(
                "<h4 style='color:var(--muted);font-size:0.9rem;margin:1.2rem 0 .4rem'>K-Line Candlestick</h4>"
            )
            for k, v in klines.items():
                label = k.replace("chart_kline_", "").replace("_", " ").upper()
                parts.append(_iframe_section(label, v, height=440))

        if "chart_risk_radar" in others:
            parts.append(
                "<h4 style='color:var(--muted);font-size:0.9rem;margin:1.2rem 0 .4rem'>Risk Radar</h4>"
            )
            parts.append(_iframe_section("Risk Radar (VaR / CVaR / MaxDD / Sharpe)", others["chart_risk_radar"], height=520))

        if "chart_correlation_heatmap" in others:
            parts.append(
                "<h4 style='color:var(--muted);font-size:0.9rem;margin:1.2rem 0 .4rem'>Correlation Heatmap</h4>"
            )
            parts.append(_iframe_section("Correlação entre Ativos (60d)", others["chart_correlation_heatmap"], height=480))

        if mc_fans:
            parts.append(
                "<h4 style='color:var(--muted);font-size:0.9rem;margin:1.2rem 0 .4rem'>Monte Carlo Fan Charts</h4>"
            )
            priority_order = ["chart_mc_fan_GSPC", "chart_mc_fan_NDX", "chart_mc_fan_CLF", "chart_mc_fan_GLD", "chart_mc_fan_VIX"]
            shown: set[str] = set()
            for k in priority_order:
                if k in mc_fans:
                    label = k.replace("chart_mc_fan_", "").replace("_", " ").upper()
                    parts.append(_iframe_section(f"MC Fan — {label}", mc_fans[k]))
                    shown.add(k)
            for k, v in mc_fans.items():
                if k not in shown:
                    label = k.replace("chart_mc_fan_", "").replace("_", " ").upper()
                    parts.append(_iframe_section(f"MC Fan — {label}", v))

    if not parts:
        return ""
    return "\n".join(parts)


def generate_macro_desk_html(
    bundle: DailyIngestionBundle,
    curation_result: "CurationResult | None" = None,
) -> str:
    """Gera HTML standalone do Macro Desk — scoreboard + narrativa."""
    from html import escape

    # ── Roda o diagnóstico ────────────────────────────────────────────────────
    from app.curation.investment_agent import diagnose
    result = diagnose(bundle, curation_result)
    scores    = result.get("scores", {})
    narrative = result.get("narrative", "")
    run_date  = result.get("run_date", str(bundle.run_date))

    # ── Sinal do writer ───────────────────────────────────────────────────────
    writer_signal = ""
    if curation_result:
        sig = curation_result.narrative.primary_signal
        writer_signal = f"{sig.label} ({sig.confidence:.0%})"

    # ── Scores ────────────────────────────────────────────────────────────────
    score_labels = {
        "rational":          "Rational Engine",
        "behavioral":        "Behavioral Engine",
        "entropy":           "Entropy Engine",
        "valuation_gap":     "Valuation Gap",
        "regime_confidence": "Regime Confidence",
    }

    def score_color(v: int) -> str:
        return {2: "#34d399", 1: "#6ee7b7", 0: "#fbbf24", -1: "#f87171", -2: "#ef4444"}.get(v, "#8892a4")

    def score_bar_html(v: int | None) -> str:
        if v is None:
            return "<span style='color:#8892a4'>—</span>"
        pct = int((v + 2) / 4 * 100)
        color = score_color(v)
        sign = f"+{v}" if v > 0 else str(v)
        return (
            f"<div style='display:flex;align-items:center;gap:10px'>"
            f"<div style='flex:1;background:#1a1d27;border-radius:4px;height:10px;overflow:hidden'>"
            f"<div style='width:{pct}%;background:{color};height:100%;border-radius:4px;transition:width .3s'></div>"
            f"</div>"
            f"<span style='font-size:1rem;font-weight:700;color:{color};min-width:28px;text-align:right'>{sign}</span>"
            f"</div>"
        )

    scores_html = ""
    for key, label in score_labels.items():
        val = scores.get(key)
        scores_html += (
            f"<div style='display:grid;grid-template-columns:180px 1fr;align-items:center;"
            f"gap:12px;padding:8px 0;border-bottom:1px solid #2a2d3a'>"
            f"<span style='color:#8892a4;font-size:0.85rem'>{label}</span>"
            f"{score_bar_html(val)}"
            f"</div>"
        )

    # ── Preços de mercado ─────────────────────────────────────────────────────
    prices_html = ""
    if bundle.market_prices:
        rows = ""
        for ticker, info in list(bundle.market_prices.items())[:10]:
            if not isinstance(info, dict):
                continue
            name  = info.get("name", ticker)
            price = info.get("price")
            ret1d = info.get("return_1d")
            if price is None:
                continue
            ret_color = "#34d399" if (ret1d or 0) >= 0 else "#f87171"
            ret_str = f"<span style='color:{ret_color}'>{ret1d:+.1f}%</span>" if ret1d is not None else ""
            rows += (
                f"<tr><td style='color:#e2e8f0'>{escape(name)}</td>"
                f"<td style='text-align:right;font-variant-numeric:tabular-nums'>{price:.2f}</td>"
                f"<td style='text-align:right'>{ret_str}</td></tr>"
            )
        prices_html = (
            f"<h2 style='font-size:0.9rem;color:#60a5fa;margin:2rem 0 0.8rem;"
            f"text-transform:uppercase;letter-spacing:.08em'>Mercado</h2>"
            f"<table style='width:100%;border-collapse:collapse;font-size:0.85rem'>"
            f"<thead><tr style='color:#8892a4;border-bottom:1px solid #2a2d3a'>"
            f"<th style='text-align:left;padding:4px 0'>Ativo</th>"
            f"<th style='text-align:right'>Preço</th>"
            f"<th style='text-align:right'>1d</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    # ── Narrativa formatada ───────────────────────────────────────────────────
    import re
    # Bold markdown **texto** → <strong>
    narrative_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escape(narrative))
    # Parágrafos
    paragraphs = [p.strip() for p in narrative_html.split("\n\n") if p.strip()]
    narrative_html = "".join(f"<p>{p}</p>" for p in paragraphs)

    from datetime import datetime
    now = datetime.now().strftime("%H:%M")

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Macro Desk — {run_date}</title>
<style>
  :root {{
    --bg:#0f1117; --surface:#1a1d27; --border:#2a2d3a;
    --text:#e2e8f0; --muted:#8892a4; --accent:#22d3ee;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0 }}
  body {{ background:var(--bg); color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    font-size:15px; line-height:1.7; padding:2rem;
    max-width:860px; margin:0 auto }}
  h2 {{ font-size:0.8rem; color:var(--accent); margin:2rem 0 0.8rem;
       text-transform:uppercase; letter-spacing:.08em }}
  p {{ margin-bottom:0.9rem; color:var(--text) }}
  strong {{ color:#fbbf24 }}
</style>
</head>
<body>

<div style="display:flex;align-items:baseline;gap:1rem;margin-bottom:0.3rem">
  <h1 style="font-size:1.8rem;font-weight:900;color:var(--accent);
             letter-spacing:-.02em">MACRO DESK</h1>
  <span style="color:var(--muted);font-size:0.85rem">{run_date} &nbsp; {now}</span>
</div>

{f'<p style="color:#8892a4;font-size:0.85rem;margin-bottom:1.5rem">Writer signal: <span style="color:#60a5fa">{escape(writer_signal)}</span></p>' if writer_signal else ""}

<div style="background:var(--surface);border:1px solid var(--border);
            border-radius:10px;padding:1.2rem 1.5rem;margin-bottom:2rem">
  <h2 style="margin-top:0">Engines</h2>
  {scores_html}
</div>

<h2>Diagnóstico</h2>
<div style="background:var(--surface);border:1px solid var(--border);
            border-radius:10px;padding:1.5rem 1.8rem">
  {narrative_html}
</div>

{prices_html}

</body></html>"""

    return html


def save_macro_desk(
    bundle: DailyIngestionBundle,
    curation_result: "CurationResult | None" = None,
) -> Path:
    """Gera e salva o HTML do Macro Desk. Retorna o path do arquivo."""
    html_path = workspace.html_report_path(bundle.run_date, bundle.run_id)
    desk_path = html_path.parent / f"{bundle.run_id}_macro_desk.html"
    desk_path.write_text(
        generate_macro_desk_html(bundle, curation_result),
        encoding="utf-8",
    )
    return desk_path


def save_reports(
    bundle: DailyIngestionBundle,
    curation_result: "CurationResult | None" = None,
    trend: "NarrativeTrend | None" = None,
) -> tuple[Path, Path, Path]:
    """
    Salva markdown, JSON resumido e HTML no workspace.

    Returns:
        (markdown_path, json_path, html_path)
    """
    md_path = workspace.markdown_report_path(bundle.run_date, bundle.run_id)
    json_path = workspace.json_report_path(bundle.run_date, bundle.run_id)
    html_path = workspace.html_report_path(bundle.run_date, bundle.run_id)

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(generate_markdown(bundle), encoding="utf-8")
    json_path.write_text(
        json.dumps(generate_json_summary(bundle), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    html_path.write_text(
        generate_html(bundle, curation_result=curation_result, trend=trend, report_dir=html_path.parent),
        encoding="utf-8",
    )

    return md_path, json_path, html_path
