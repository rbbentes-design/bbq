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


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Relatorio {date}</title>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3a;
    --text: #e2e8f0;
    --muted: #8892a4;
    --accent: #60a5fa;
    --green: #34d399;
    --red: #f87171;
    --yellow: #fbbf24;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 15px;
    line-height: 1.6;
    padding: 2rem;
    max-width: 900px;
    margin: 0 auto;
  }}
  h1 {{ font-size: 1.6rem; color: var(--accent); margin-bottom: 0.5rem; }}
  h2 {{ font-size: 1.2rem; color: var(--accent); margin: 2rem 0 1rem;
        padding-bottom: 0.4rem; border-bottom: 1px solid var(--border); }}
  h3 {{ font-size: 1rem; color: var(--text); margin: 1.5rem 0 0.4rem; }}
  p {{ margin-bottom: 0.8rem; color: var(--text); }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  code {{ background: var(--surface); padding: 0.1em 0.4em; border-radius: 4px;
          font-size: 0.85em; color: var(--yellow); }}
  blockquote {{
    border-left: 3px solid var(--accent);
    padding: 0.5rem 1rem;
    margin: 0.8rem 0;
    background: var(--surface);
    border-radius: 0 6px 6px 0;
    color: var(--text);
  }}
  hr {{ border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }}
  .meta {{
    display: flex; gap: 1.5rem; flex-wrap: wrap;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 2rem;
    font-size: 0.85rem; color: var(--muted);
  }}
  .meta span {{ display: flex; align-items: center; gap: 0.4rem; }}
  .meta strong {{ color: var(--text); }}
  em {{ color: var(--muted); font-style: normal; font-size: 0.85rem; }}
  ul {{ padding-left: 1.5rem; }}
  li {{ margin-bottom: 0.3rem; color: var(--red); }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def generate_html(bundle: DailyIngestionBundle) -> str:
    """Gera o relatorio completo em HTML estilizado."""
    import markdown as md_lib

    raw_md = generate_markdown(bundle)

    # Substitui cabecalho por bloco de meta cards
    lines = raw_md.splitlines()
    h1_line = lines[0] if lines else ""
    # Extrai bloco de meta (linhas 2-7) e converte resto para HTML
    meta_lines = [l for l in lines[2:8] if l.strip()]
    content_lines = lines[8:]

    def _meta_val(label: str) -> str:
        for l in meta_lines:
            if label in l:
                import re
                m = re.search(r"\*\*[^*]+\*\*\s*`?([^`<\n]+)`?", l)
                return m.group(1).strip() if m else ""
        return ""

    run_id_short = _meta_val("run_id")[:16] + "..."
    gerado = _meta_val("gerado em")
    zh_count = _meta_val("ZeroHedge blocos")
    x_count = _meta_val("X tweets")
    erros = _meta_val("Erros")

    meta_html = f"""<h1>{h1_line.lstrip("# ")}</h1>
<div class="meta">
  <span>&#128196; run_id: <strong><code>{run_id_short}</code></strong></span>
  <span>&#128336; <strong>{gerado}</strong></span>
  <span>&#128200; ZeroHedge: <strong>{zh_count} blocos</strong></span>
  <span>&#128038; X tweets: <strong>{x_count}</strong></span>
  <span>&#9888; Erros: <strong>{erros}</strong></span>
</div>"""

    body_md = "\n".join(content_lines)
    body_html = md_lib.markdown(body_md, extensions=["nl2br"])

    full_body = meta_html + "\n" + body_html
    date_str = str(bundle.run_date)
    return _HTML_TEMPLATE.format(date=date_str, body=full_body)


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
