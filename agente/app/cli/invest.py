"""
CLI: agente invest

Diagnóstico de investimento sobre o bundle mais recente.
Não coleta dados — opera sobre o pipeline já executado.

Uso:
    python -m app.cli.invest
    python -m app.cli.invest --date 2026-03-29
    python -m app.cli.invest --focus "foque em crédito HY e curva de juros"
    python -m app.cli.invest --save
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from app.audit.logger import configure_logging

app = typer.Typer(name="invest", help="Diagnóstico de investimento sobre o bundle do dia.")
console = Console()


@app.command()
def main(
    date: str = typer.Option(None, "--date", "-d", help="Data YYYY-MM-DD (padrão: mais recente)"),
    focus: str = typer.Option(None, "--focus", "-f", help="Instrução específica (ex: 'foque em crédito HY')"),
    save: bool = typer.Option(False, "--save", "-s", help="Salva output em .txt no diretório do bundle"),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    configure_logging(log_level)

    from app.curation.models import CurationResult
    from app.curation.investment_agent import diagnose
    from app.models.daily_ingestion_bundle import DailyIngestionBundle
    from app.storage.paths import workspace

    bundles_dir = workspace.bundles

    # ── Localiza bundle ────────────────────────────────────────────────────────
    if date:
        target_dir = bundles_dir / date
        if not target_dir.exists():
            console.print(f"[red]Bundle não encontrado para {date}[/red]")
            raise typer.Exit(1)
    else:
        # Pega o mais recente
        dirs = sorted([d for d in bundles_dir.iterdir() if d.is_dir()], reverse=True)
        if not dirs:
            console.print("[red]Nenhum bundle encontrado no workspace.[/red]")
            raise typer.Exit(1)
        target_dir = dirs[0]

    bundle_files = sorted(target_dir.glob("*.json"))
    bundle_files = [f for f in bundle_files if not any(
        s in f.name for s in ("_curation", "_summary", "_written")
    )]
    if not bundle_files:
        console.print(f"[red]Bundle JSON não encontrado em {target_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[dim]Bundle: {target_dir.name} / {bundle_files[-1].name}[/dim]")

    try:
        bundle = DailyIngestionBundle.model_validate(
            json.loads(bundle_files[-1].read_text(encoding="utf-8"))
        )
    except Exception as exc:
        console.print(f"[red]Erro ao carregar bundle: {exc}[/red]")
        raise typer.Exit(1)

    # ── Curation result (opcional) ─────────────────────────────────────────────
    curation = None
    curation_files = sorted(target_dir.glob("*_curation.json"), reverse=True)
    if curation_files:
        try:
            curation = CurationResult.model_validate(
                json.loads(curation_files[0].read_text(encoding="utf-8"))
            )
        except Exception:
            pass

    # ── Diagnóstico ────────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]INVESTMENT AGENT[/bold cyan]"))
    console.print(f"[dim]Analisando bundle de {bundle.run_date}...[/dim]")
    if focus:
        console.print(f"[yellow]Foco: {focus}[/yellow]")
    console.print()

    result = diagnose(bundle, curation, focus=focus)

    console.print(Panel(result, title=f"[bold]Diagnóstico — {bundle.run_date}[/bold]",
                        border_style="cyan", expand=True))

    # ── Salva ──────────────────────────────────────────────────────────────────
    if save:
        run_id = bundle_files[-1].stem
        out_path = target_dir / f"{run_id}_investment_call.txt"
        out_path.write_text(result, encoding="utf-8")
        console.print(f"\n[green]Salvo: {out_path.name}[/green]")
