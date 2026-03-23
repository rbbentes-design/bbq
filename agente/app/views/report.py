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

from app.models.daily_ingestion_bundle import DailyIngestionBundle
from app.models.market_ear_block import MarketEarBlock
from app.models.x_timeline_item import XTimelineItem
from app.storage.paths import workspace


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


def generate_html(bundle: DailyIngestionBundle) -> str:
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
</style>
</head>
<body>""")

    # ── Meta ──────────────────────────────────────────────────────────────────
    parts.append(f"""
<h1>Relatorio de Ingestao &mdash; {bundle.run_date}</h1>
<div class="meta">
  <span>&#128196; run_id: <strong><code>{bundle.run_id[:20]}...</code></strong></span>
  <span>&#128336; <strong>{gerado}</strong></span>
  <span>&#128200; ZeroHedge: <strong>{len(bundle.market_ear_blocks)} blocos</strong></span>
  <span>&#128038; X tweets: <strong>{len(bundle.x_items)}</strong></span>
  <span>&#9888; Erros: <strong>{bundle.audit_summary.errors}</strong></span>
</div>""")

    # ── ZeroHedge ─────────────────────────────────────────────────────────────
    parts.append("<h2>The Market Ear (ZeroHedge)</h2>")
    if not bundle.market_ear_blocks:
        parts.append("<p><em>Nenhum bloco coletado.</em></p>")
    else:
        for i, b in enumerate(bundle.market_ear_blocks):
            parts.append(f'<div class="block">')
            parts.append(f"<h3>{i + 1}. {escape(b.title or '(sem titulo)')}</h3>")
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
            parts.append(f'<div class="tweet">')
            parts.append(
                f'<span class="tweet-author">{escape(it.author)}</span>'
                f'<span class="tweet-ts">{ts}</span>'
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

    # ── Erros ─────────────────────────────────────────────────────────────────
    if bundle.audit_summary.error_messages:
        parts.append("<h2>Erros</h2><ul class='errors'>")
        for msg in bundle.audit_summary.error_messages:
            parts.append(f"<li>{escape(msg)}</li>")
        parts.append("</ul>")

    parts.append("</body></html>")
    return "\n".join(parts)


def save_reports(bundle: DailyIngestionBundle) -> tuple[Path, Path, Path]:
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
    html_path.write_text(generate_html(bundle), encoding="utf-8")

    return md_path, json_path, html_path
