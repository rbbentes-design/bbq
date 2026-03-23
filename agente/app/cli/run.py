"""
CLI: agente run

Executa o pipeline de ingestao completo e exibe o resumo.

Uso:
    python -m app.cli.run
    python -m app.cli.run --headless
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from app.audit.logger import configure_logging
from app.pipeline.ingestion import run_ingestion

app = typer.Typer(name="run", help="Executa o pipeline de ingestao.")
console = Console()


@app.command()
def ingest(
    headless: bool = typer.Option(False, "--headless", help="Browser sem janela."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Coleta ZeroHedge Market Ear + X timeline e salva bundle diario."""
    configure_logging("DEBUG" if verbose else "INFO")

    console.print("[bold]Iniciando pipeline de ingestao...[/bold]")

    bundle = run_ingestion(headless=headless)

    table = Table(title=f"Bundle {bundle.run_date} | run_id={bundle.run_id[:8]}...",
                  show_header=True, header_style="bold")
    table.add_column("Metrica", style="cyan")
    table.add_column("Valor", justify="right")

    table.add_row("ZeroHedge blocos", str(len(bundle.market_ear_blocks)))
    table.add_row("X tweets", str(len(bundle.x_items)))
    table.add_row("Erros", str(bundle.audit_summary.errors))

    for key, path in bundle.artifact_paths.items():
        table.add_row(f"  {key}", path[-60:])

    console.print(table)

    if bundle.audit_summary.errors == 0:
        console.print("\n[green]Pipeline concluido sem erros.[/green]")
    else:
        msgs = bundle.audit_summary.error_messages
        for m in msgs:
            console.print(f"[red]  ERRO: {m}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
