"""
Provider: SpotGamma FlowPatrol

Coleta o relatório FlowPatrol mais recente via Playwright:
  1. Navega para /reports (redireciona automaticamente para o mais recente)
  2. Clica em "Download PDF" e intercepta o download
  3. Extrai texto + tabelas do PDF com pdfplumber

Autenticação:
  Perfil persistente ~/agente-workspace/state/browser.
  Primeiro uso: python -m app.cli.auth login spotgamma
"""

from __future__ import annotations

import re
import time
from datetime import date
from pathlib import Path

from playwright.sync_api import Page

from app.audit.logger import get_logger
from app.config.settings import settings
from app.models.spotgamma_report import ReportSection, SpotGammaReport
from app.utils.timestamps import new_ulid, utcnow

_log = get_logger("providers.spotgamma")

SOURCE_NAME = "spotgamma"
_REPORTS_URL = "https://dashboard.spotgamma.com/reports"
_FOUNDERS_URL = "https://dashboard.spotgamma.com/foundersNotes"


# ── Public API ─────────────────────────────────────────────────────────────────

def collect_flow_patrol(page: Page) -> SpotGammaReport | None:
    """
    Abre /reports (redireciona para o mais recente), baixa o PDF e extrai conteúdo.
    Retorna None se não autenticado ou PDF indisponível.
    """
    _log.info("collect_start", source=SOURCE_NAME)

    page.goto(_REPORTS_URL, timeout=30_000)
    page.wait_for_load_state("domcontentloaded", timeout=15_000)

    # Checa redirecionamento para login ANTES do networkidle (evita timeout desnecessário)
    if _is_login_page(page.url):
        _log.warning("not_authenticated", url=page.url[:80])
        return None

    try:
        page.wait_for_load_state("networkidle", timeout=20_000)
    except Exception:
        pass  # SPA pode nunca atingir networkidle — continua mesmo assim

    time.sleep(3)  # SPA precisa renderizar

    report_url = page.url
    _log.info("report_url", url=report_url[:100])

    # Re-verifica após renderização (pode ter redirecionado depois do domcontentloaded)
    if _is_login_page(report_url) or report_url == _REPORTS_URL:
        _log.warning("not_authenticated", url=report_url[:80])
        return None

    # Extrai título e data da URL / página
    title = _extract_title_from_url(report_url)
    report_date = _extract_date_from_url(report_url)

    # Extrai texto visível já renderizado (fallback e metadados)
    body_text = page.inner_text("body")

    # Download do PDF
    pdf_path = _download_pdf(page, report_url)

    if pdf_path:
        sections, raw_text, image_count = _parse_pdf(pdf_path)
        # Remove PDF temporário após extração
        try:
            pdf_path.unlink()
        except Exception:
            pass
    else:
        # Fallback: usa texto visível da página
        _log.warning("pdf_fallback_to_page_text", source=SOURCE_NAME)
        sections, raw_text = _parse_page_text(body_text)
        image_count = 0

    # Título da página se não extraído da URL
    if not title:
        for line in body_text.splitlines():
            line = line.strip()
            if len(line) > 20 and "FlowPatrol" not in line and line[0].isupper():
                title = line
                break

    report = SpotGammaReport(
        id=new_ulid(),
        report_type="FlowPatrol",
        report_date=report_date,
        title=title,
        source_url=report_url,
        sections=sections,
        image_refs=[],   # PDF não expõe URLs de imagem; charts ficam no PDF
        raw_text=raw_text[:10_000],
        collected_at=utcnow(),
    )

    _log.info("collect_done", source=SOURCE_NAME,
              sections=len(sections), text_chars=len(raw_text),
              pdf_pages=image_count)
    return report


def collect_founders_notes(page: Page, max_notes: int = 2) -> list[SpotGammaReport]:
    """
    Coleta as notas mais recentes do Founder's Notes (PM Notes + Founder's Notes).
    Por padrão pega as 2 mais recentes (normalmente hoje).

    A página carrega automaticamente a nota mais recente no conteúdo principal.
    Para as demais, clica nos itens da sidebar.
    """
    _log.info("founders_notes_start", source=SOURCE_NAME, max_notes=max_notes)

    page.goto(_FOUNDERS_URL, timeout=30_000)
    page.wait_for_load_state("domcontentloaded", timeout=15_000)

    if _is_login_page(page.url):
        _log.warning("not_authenticated_founders", url=page.url[:80])
        return []

    try:
        page.wait_for_load_state("networkidle", timeout=20_000)
    except Exception:
        pass  # SPA — continua mesmo sem networkidle

    time.sleep(3)

    if _is_login_page(page.url):
        _log.warning("not_authenticated_founders", url=page.url[:80])
        return []

    # Descobre itens da sidebar
    sidebar = page.locator('[class*="MuiListItemButton"]')
    n_items = sidebar.count()
    if n_items == 0:
        _log.warning("founders_sidebar_empty")
        return []

    _log.info("founders_sidebar_found", items=n_items)

    results: list[SpotGammaReport] = []

    # O item [0] já está carregado na página principal
    for i in range(min(max_notes, n_items)):
        try:
            if i > 0:
                # Clica no item da sidebar para carregar
                sidebar.nth(i).click()
                time.sleep(2)

            body_text = page.inner_text("body")
            note_title, note_body = _extract_note_content(body_text)
            if not note_body:
                continue

            report_type = _classify_note_type(note_title)
            report_date = _parse_note_date(note_title)

            sections = [ReportSection(heading="", body_text=note_body)]
            report = SpotGammaReport(
                id=new_ulid(),
                report_type=report_type,
                report_date=report_date,
                title=note_title,
                source_url=_FOUNDERS_URL,
                sections=sections,
                raw_text=note_body[:8000],
                collected_at=utcnow(),
            )
            results.append(report)
            _log.info("founders_note_collected", idx=i, type=report_type,
                      title=note_title[:60], chars=len(note_body))

        except Exception as exc:
            _log.warning("founders_note_error", idx=i, error=str(exc))

    _log.info("founders_notes_done", source=SOURCE_NAME, collected=len(results))
    return results


def collect_html(page: Page) -> str:
    """Retorna HTML bruto da página de reports (para RawStore)."""
    page.goto(_REPORTS_URL, timeout=30_000)
    page.wait_for_load_state("networkidle", timeout=20_000)
    time.sleep(5)
    return page.content()


# ── PDF download ───────────────────────────────────────────────────────────────

def _download_pdf(page: Page, report_url: str) -> Path | None:
    """Clica em 'Download PDF' e retorna o path do arquivo baixado."""
    try:
        btn = page.locator('button:has-text("Download PDF"), a:has-text("Download PDF")')
        if btn.count() == 0:
            _log.warning("pdf_button_not_found")
            return None

        # Destino temporário no workspace
        dest_dir = settings.resolved_raw_dir() / "spotgamma"
        dest_dir.mkdir(parents=True, exist_ok=True)

        with page.expect_download(timeout=30_000) as dl_info:
            btn.first.click()

        dl = dl_info.value
        # Usa nome do arquivo original do servidor quando possível
        filename = dl.suggested_filename or f"flowpatrol_{date.today()}.pdf"
        dest = dest_dir / filename
        dl.save_as(str(dest))

        _log.info("pdf_downloaded", path=str(dest), bytes=dest.stat().st_size)
        return dest

    except Exception as exc:
        _log.warning("pdf_download_error", error=str(exc))
        return None


# ── PDF parsing ────────────────────────────────────────────────────────────────

def _parse_pdf(pdf_path: Path) -> tuple[list[ReportSection], str, int]:
    """
    Extrai seções do PDF FlowPatrol com pdfplumber.
    Estratégia: uma seção por página (evita problema da tabela de conteúdo).
    Retorna (sections, raw_text, n_pages).
    """
    import pdfplumber

    sections: list[ReportSection] = []
    all_text_parts: list[str] = []
    n_pages = 0

    # Seções de alto valor para extrair separadamente
    key_sections: dict[str, str] = {}

    with pdfplumber.open(str(pdf_path)) as pdf:
        n_pages = len(pdf.pages)

        for i, pg in enumerate(pdf.pages):
            page_text = (pg.extract_text() or "").strip()
            if not page_text:
                continue

            # Remove cabeçalho repetido "FlowPatrolTM" de cada página
            page_text = re.sub(r"^FlowPatrol\s*TM\s*\n?", "", page_text, flags=re.IGNORECASE)

            # Tabelas estruturadas desta página
            table_parts: list[str] = []
            for tbl in pg.extract_tables():
                rows = []
                for row in tbl:
                    cells = [str(c or "").strip() for c in row]
                    if any(c for c in cells if c):
                        rows.append(" | ".join(cells))
                if rows:
                    table_parts.append("\n".join(rows))

            # Detecta heading principal da página (primeira linha que é heading)
            heading = ""
            body_lines: list[str] = []
            for line in page_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if not heading and _is_heading(line):
                    heading = line
                else:
                    body_lines.append(line)

            body_text = "\n".join(body_lines).strip()
            table_text = "\n\n".join(table_parts)
            combined = "\n\n".join(filter(None, [body_text, table_text]))

            if combined:
                sections.append(ReportSection(heading=heading, body_text=combined))
                chunk = f"### {heading}\n{combined}" if heading else combined
                all_text_parts.append(chunk.strip())

                # Captura seções-chave por nome
                if heading and not heading.startswith("FlowPatrol"):
                    key_sections.setdefault(heading, combined)

    raw_text = "\n\n".join(all_text_parts)
    return sections, raw_text, n_pages


def _is_heading(line: str) -> bool:
    """Detecta linhas de cabeçalho no FlowPatrol PDF."""
    # Linha curta sem ponto final que parece um título de seção
    if len(line) > 80:
        return False
    heading_patterns = [
        r"^(Single Stock Positioning|Sector Breakdown|Directional Positioning|"
        r"Gamma Positioning|Volatility Positioning|Top Position Changes|"
        r"Largest Premium Trades|Largest Index Trades|Top Index Trades|"
        r"Unusual Options Positions|Statistically Significant|Heavy DayTrading|"
        r"What Traders Should Know|Executive Summary|Index ETF|Sector ETF)",
    ]
    for pat in heading_patterns:
        if re.match(pat, line, re.IGNORECASE):
            return True
    return False


def _flush(
    sections: list[ReportSection],
    raw_parts: list[str],
    heading: str,
    body: list[str],
    tables: list[str],
) -> None:
    body_text = "\n".join(body).strip()
    table_text = "\n\n".join(tables).strip()
    combined = "\n\n".join(filter(None, [body_text, table_text]))
    if heading or combined:
        sections.append(ReportSection(
            heading=heading,
            body_text=combined,
        ))
        chunk = f"### {heading}\n{combined}" if heading else combined
        if chunk.strip():
            raw_parts.append(chunk.strip())


# ── Fallback: texto da página ──────────────────────────────────────────────────

def _parse_page_text(body_text: str) -> tuple[list[ReportSection], str]:
    """Extrai seções do texto visível renderizado pelo browser (fallback)."""
    sections: list[ReportSection] = []
    raw_parts: list[str] = []
    current_heading = ""
    current_body: list[str] = []

    for line in body_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _is_heading(line):
            if current_heading or current_body:
                body = "\n".join(current_body).strip()
                sections.append(ReportSection(heading=current_heading, body_text=body))
                raw_parts.append(f"### {current_heading}\n{body}")
            current_heading = line
            current_body = []
        else:
            current_body.append(line)

    if current_heading or current_body:
        body = "\n".join(current_body).strip()
        sections.append(ReportSection(heading=current_heading, body_text=body))
        raw_parts.append(f"### {current_heading}\n{body}")

    return sections, "\n\n".join(raw_parts)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_login_page(url: str) -> bool:
    """Retorna True se a URL indica página de login (não autenticado).
    NÃO inclui /callback nem auth0/cognito — o fluxo OAuth autenticado passa por
    essas URLs antes de chegar na página do report. Checar essas URLs causaria
    falso-negativo no fluxo autenticado.
    """
    return "/login" in url or "/signin" in url or not url or url == "about:blank"


def _extract_note_content(body_text: str) -> tuple[str, str]:
    """
    Extrai o título e corpo da nota do texto visível da página.

    Estrutura do body:
      ... nav/header ...
      Review
      Print
      <TÍTULO DA NOTA>      ← primeira linha após Print
      <CORPO>               ← até o início da lista da sidebar (nota repetida)
    """
    lines = [l.strip() for l in body_text.splitlines() if l.strip()]

    # Encontra onde começa a nota (após "Print")
    start = -1
    for i, line in enumerate(lines):
        if line == "Print":
            start = i + 1
            break

    if start < 0 or start >= len(lines):
        return "", ""

    note_title = lines[start]

    # O corpo vai até onde a sidebar repete o título (loop)
    # Ou até atingir ~80 linhas (evita capturar sidebar inteira)
    body_lines: list[str] = []
    for line in lines[start + 1:]:
        # A sidebar repete o título exatamente — para aqui
        if line == note_title:
            break
        # Indicador de início da lista lateral
        if re.match(r"^(PM Note|Founder'?s? Note):.*(ET|EDT|EST)$", line):
            break
        body_lines.append(line)

    return note_title, "\n".join(body_lines).strip()


def _classify_note_type(title: str) -> str:
    if title.lower().startswith("pm note"):
        return "PMNote"
    if "founder" in title.lower():
        return "FoundersNote"
    return "Note"


def _parse_note_date(title: str) -> date | None:
    """Extrai data de 'PM Note: Mon, March 23, 2026 at 5:31 PM ET'"""
    import datetime
    m = re.search(r"(\w+ \d+, \d{4})", title)
    if m:
        try:
            return datetime.datetime.strptime(m.group(1), "%B %d, %Y").date()
        except ValueError:
            pass
    return None


def _extract_title_from_url(url: str) -> str:
    m = re.search(r"FlowPatrol[^/]*", url)
    if m:
        return m.group(0).replace("%20", " ").replace("___", " | ")
    return "FlowPatrol"


def _extract_date_from_url(url: str) -> date | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", url)
    if m:
        try:
            return date.fromisoformat(m.group(1))
        except ValueError:
            pass
    return None
