from __future__ import annotations

from app.models.daily_ingestion_bundle import DailyIngestionBundle
from app.models.market_ear_block import MarketEarBlock
from app.models.x_timeline_item import XTimelineItem

# Use 16 chars (10 timestamp + 6 random) to avoid ULID prefix collisions
# among items collected within the same millisecond window.
_ID_LEN = 16


def _me_sid(item_id: str) -> str:
    return f"ME-{item_id[:_ID_LEN]}"


def _x_sid(item_id: str) -> str:
    return f"X-{item_id[:_ID_LEN]}"


def build_corpus_text(bundle: DailyIngestionBundle, max_chars_per_item: int = 800) -> str:
    lines: list[str] = [f"=== THE MARKET EAR ({len(bundle.market_ear_blocks)} blocks) ===\n"]
    for b in bundle.market_ear_blocks:
        sid = _me_sid(b.id)
        ts = b.published_at.strftime("%Y-%m-%d %H:%M UTC") if b.published_at else "unknown"
        body = b.body_text[:max_chars_per_item].replace("\n", " ").strip()
        lines.append(f"[{sid}] {b.title} | {ts}")
        if body:
            lines.append(body)
        lines.append("")

    lines.append(f"=== X TIMELINE ({len(bundle.x_items)} items) ===\n")
    for it in bundle.x_items:
        sid = _x_sid(it.id)
        ts = it.created_at.strftime("%Y-%m-%d %H:%M UTC") if it.created_at else "unknown"
        eng = it.engagement_info
        text = it.text[:max_chars_per_item].replace("\n", " ").strip()
        lines.append(f"[{sid}] {it.author} | {ts} | likes:{eng.likes} reposts:{eng.reposts}")
        if text:
            lines.append(text)
        lines.append("")

    if bundle.rss_items:
        lines.append(f"=== RSS FEEDS ({len(bundle.rss_items)} items) ===\n")
        for it in bundle.rss_items:
            sid = f"RSS-{it.id[:_ID_LEN]}"
            ts = it.published_at.strftime("%Y-%m-%d %H:%M UTC") if it.published_at else "unknown"
            text = it.summary[:max_chars_per_item].replace("\n", " ").strip() or it.title
            lines.append(f"[{sid}] {it.source_name} | {ts}")
            lines.append(f"{it.title}")
            if text and text != it.title:
                lines.append(text)
            lines.append("")

    if bundle.spotgamma_reports:
        lines.append(f"=== SPOTGAMMA ({len(bundle.spotgamma_reports)} reports) ===\n")
        for rpt in bundle.spotgamma_reports:
            sid = f"SG-{rpt.id[:_ID_LEN]}"
            date_str = str(rpt.report_date) if rpt.report_date else "unknown"
            lines.append(f"[{sid}] SpotGamma {rpt.report_type} | {date_str}")
            lines.append(rpt.title)
            # Inclui o texto bruto do report (já limitado a 8000 chars na coleta)
            if rpt.raw_text:
                lines.append(rpt.raw_text[:max_chars_per_item * 4])
            lines.append("")

    return "\n".join(lines)


def build_item_index(bundle: DailyIngestionBundle) -> dict[str, MarketEarBlock | XTimelineItem]:
    index: dict[str, MarketEarBlock | XTimelineItem] = {}
    for b in bundle.market_ear_blocks:
        index[_me_sid(b.id)] = b
        index[b.id] = b
    for it in bundle.x_items:
        index[_x_sid(it.id)] = it
        index[it.id] = it
    return index


def item_to_snippet(item: MarketEarBlock | XTimelineItem, max_chars: int = 400) -> str:
    if isinstance(item, MarketEarBlock):
        sid = _me_sid(item.id)
        ts = item.published_at.strftime("%Y-%m-%d %H:%M UTC") if item.published_at else ""
        body = item.body_text[:max_chars].replace("\n", " ")
        return f"[{sid}] {item.title} | {ts}\n{body}"
    else:
        sid = _x_sid(item.id)
        ts = item.created_at.strftime("%Y-%m-%d %H:%M UTC") if item.created_at else ""
        eng = item.engagement_info
        return f"[{sid}] {item.author} | {ts} | likes:{eng.likes}\n{item.text[:max_chars]}"
