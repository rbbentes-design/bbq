"""
Gerador de relatorio do DailyIngestionBundle.

Produz:
  - Markdown legivel para revisao humana
  - JSON resumido para consumo programatico

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


def save_reports(bundle: DailyIngestionBundle) -> tuple[Path, Path]:
    """
    Salva markdown e JSON resumido no workspace.

    Returns:
        (markdown_path, json_path)
    """
    md_path = workspace.markdown_report_path(bundle.run_date, bundle.run_id)
    json_path = workspace.json_report_path(bundle.run_date, bundle.run_id)

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(generate_markdown(bundle), encoding="utf-8")
    json_path.write_text(
        json.dumps(generate_json_summary(bundle), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return md_path, json_path
