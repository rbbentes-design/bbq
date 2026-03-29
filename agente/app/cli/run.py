"""
CLI: agente run

Executa ingestao + curacao + abre relatorio HTML.

Uso:
    python -m app.cli.run
    python -m app.cli.run --headless
    python -m app.cli.run --no-open
"""

from __future__ import annotations

import webbrowser
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app.audit.logger import configure_logging
from app.pipeline.ingestion import run_ingestion

app = typer.Typer(name="run", help="Executa pipeline completo: ingestao + curacao + relatorio.")
console = Console()


@app.command()
def ingest(
    headless: bool = typer.Option(False, "--headless", help="Browser sem janela."),
    no_open: bool = typer.Option(False, "--no-open", help="Nao abre o HTML no browser."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Coleta ZeroHedge + X, roda curacao LLM e abre relatorio."""
    configure_logging("DEBUG" if verbose else "INFO")

    console.print(Panel.fit("[bold cyan]Agente Editorial[/bold cyan] — iniciando run diario"))

    with console.status("[cyan]Coletando fontes...[/cyan]"):
        bundle = run_ingestion(headless=headless)

    # ── Tabela de coleta ───────────────────────────────────────────────────────
    t = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    t.add_column("Fonte")
    t.add_column("Itens", justify="right")
    t.add_column("Erros", justify="right")

    t.add_row("ZeroHedge Market Ear", str(len(bundle.market_ear_blocks)), "")
    t.add_row("X Timeline", str(len(bundle.x_items)), "")

    # SpotGamma
    sg = bundle.spotgamma_reports
    fp = [r for r in sg if r.report_type == "FlowPatrol"]
    notes = [r for r in sg if r.report_type in ("PMNote", "FoundersNote", "Note")]
    if fp:
        t.add_row("SpotGamma FlowPatrol", str(len(fp)), "")
    if notes:
        t.add_row("SpotGamma Notes", str(len(notes)), "")

    # RSS items (inclui Spectra + DeepVue + feeds)
    rss = bundle.rss_items
    spectra_items = [r for r in rss if "Spectra" in r.source_name]
    deepvue_items = [r for r in rss if "DeepVue" in r.source_name]
    feed_items = [r for r in rss if "Spectra" not in r.source_name and "DeepVue" not in r.source_name]

    if spectra_items:
        t.add_row("Spectra Markets", str(len(spectra_items)), "")
    if deepvue_items:
        t.add_row("DeepVue Theme Tracker", str(len(deepvue_items)), "")
    if feed_items:
        t.add_row(f"RSS ({len(set(r.feed_url for r in feed_items))} feeds)", str(len(feed_items)), "")

    total = (
        len(bundle.market_ear_blocks) + len(bundle.x_items)
        + len(sg) + len(rss)
    )
    t.add_row(
        "Total",
        str(total),
        str(bundle.audit_summary.errors) if bundle.audit_summary.errors else "[green]0[/green]",
    )
    console.print(t)

    # ── Curação ───────────────────────────────────────────────────────────────
    curation_path = bundle.artifact_paths.get("curation")
    if curation_path:
        _print_curation_summary(curation_path)
    else:
        console.print("[yellow]Curação desabilitada ou falhou.[/yellow]")

    # ── Erros ─────────────────────────────────────────────────────────────────
    if bundle.audit_summary.error_messages:
        for m in bundle.audit_summary.error_messages:
            console.print(f"  [red][X] {m}[/red]")

    # ── HTML Relatório ────────────────────────────────────────────────────────
    html_path = bundle.artifact_paths.get("html_report")
    if html_path and not no_open:
        p = Path(html_path)
        if p.exists():
            console.print(f"\n[green]Abrindo relatorio:[/green] {p.name}")
            webbrowser.open(p.as_uri())

    # ── Macro Desk HTML ───────────────────────────────────────────────────────
    curation_obj = None
    curation_path = bundle.artifact_paths.get("curation")
    if curation_path:
        try:
            import json as _json
            from app.curation.models import CurationResult
            curation_obj = CurationResult.model_validate(
                _json.loads(Path(curation_path).read_text(encoding="utf-8"))
            )
        except Exception:
            pass

    try:
        from app.views.report import save_macro_desk
        with console.status("[cyan]Macro Desk processando...[/cyan]"):
            desk_path = save_macro_desk(bundle, curation_obj)
        console.print(f"[green]Abrindo Macro Desk:[/green] {desk_path.name}")
        if not no_open:
            webbrowser.open(desk_path.as_uri())
    except Exception as exc:
        console.print(f"[yellow]Macro Desk falhou: {exc}[/yellow]")

    console.print(Panel.fit(
        f"[bold green]Run concluido[/bold green] — {bundle.run_date} | id={bundle.run_id[:12]}",
        border_style="green",
    ))

    if bundle.audit_summary.errors > 0:
        raise typer.Exit(1)


def _print_curation_summary(curation_path: str) -> None:
    import json
    try:
        data = json.loads(Path(curation_path).read_text(encoding="utf-8"))
        primary = data["narrative"]["primary_signal"]
        secondary = data["narrative"].get("secondary_signals", [])
        verdict = data["verification"]["overall_verdict"]
        scored = len(data.get("scored_items", []))

        verdict_color = {"pass": "green", "warn": "yellow", "fail": "red"}.get(verdict, "white")
        verdict_icon = {"pass": "[OK]", "warn": "[!]", "fail": "[X]"}.get(verdict, "?")

        written_mode = data.get("artifact_paths", {}).get("written_mode", "")
        written_path = data.get("artifact_paths", {}).get("written", "")

        console.print()
        console.print(Panel(
            Text.assemble(
                ("Narrativa primária\n", "bold"),
                (f"  {primary['label']}\n", "cyan"),
                (f"  Confiança: {primary['confidence']:.0%}", ""),
                (f"   Verificação: [{verdict_color}]{verdict_icon} {verdict}[/{verdict_color}]\n", ""),
                (f"  {scored} itens pontuados\n", "dim"),
                *([
                    ("Narrativa secundária\n", "bold"),
                    (f"  {secondary[0]['label']} ({secondary[0]['confidence']:.0%})\n", "dim cyan"),
                ] if secondary else []),
                *([ ("Texto gerado\n", "bold"),
                    (f"  Modo: {written_mode}  |  {Path(written_path).name}\n", "green"),
                ] if written_mode else [("Texto\n", "bold"), ("  não gerado\n", "dim")]),
            ),
            title="[bold]Curação LLM[/bold]",
            border_style="cyan",
        ))
    except Exception:
        console.print("[dim]Resumo de curação indisponivel.[/dim]")


@app.command()
def desk(
    date: str = typer.Option(None, "--date", "-d", help="Data YYYY-MM-DD (padrão: mais recente)"),
    focus: str = typer.Option(None, "--focus", "-f", help="Instrução específica do operador"),
    save: bool = typer.Option(False, "--save", "-s", help="Salva diagnóstico em .txt"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Macro Desk — diagnóstico macro acionável sobre o bundle do dia."""
    from app.cli.invest import main as desk_main
    desk_main(date=date, focus=focus, save=save, log_level="DEBUG" if verbose else "WARNING")


if __name__ == "__main__":
    app()
