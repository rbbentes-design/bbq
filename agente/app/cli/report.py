"""
CLI: agente report

Gera ou exibe o relatorio do ultimo bundle (ou de um especifico).

Uso:
    python -m app.cli.report show
    python -m app.cli.report show --run-id 01KMC19...
    python -m app.cli.report list
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.markdown import Markdown

from app.storage.bundle_store import bundle_store
from app.views.report import generate_markdown, generate_json_summary, generate_html

app = typer.Typer(name="report", help="Gera e exibe relatorios de ingestao.")
console = Console()


@app.command()
def show(
    run_id: str = typer.Option("", "--run-id", help="ID do run (vazio = ultimo)"),
    fmt: str = typer.Option("md", "--format", "-f", help="Formato: md | json | html"),
) -> None:
    """Exibe o relatorio do bundle mais recente (ou do run_id especificado)."""
    bundles = bundle_store.list_bundles()
    if not bundles:
        console.print("[red]Nenhum bundle encontrado. Rode: python -m app.cli.run[/red]")
        raise typer.Exit(1)

    if run_id:
        matches = [p for p in bundles if run_id in p.stem]
        if not matches:
            console.print(f"[red]run_id {run_id!r} nao encontrado.[/red]")
            raise typer.Exit(1)
        bundle_path = matches[0]
    else:
        # Bundle mais recente: ordena por data da pasta + ULID (ambos lexicograficos)
        bundle_path = max(bundles, key=lambda p: (p.parent.name, p.stem))

    from datetime import date
    stem = bundle_path.stem
    parent = bundle_path.parent.name  # data: 2026-03-22
    try:
        run_date = date.fromisoformat(parent)
    except ValueError:
        run_date = date.today()

    bundle = bundle_store.load(run_date, stem)

    if fmt == "json":
        summary = generate_json_summary(bundle)
        console.print_json(json.dumps(summary, ensure_ascii=False, indent=2))
    elif fmt == "html":
        import webbrowser
        from app.storage.paths import workspace
        html_path = workspace.html_report_path(bundle.run_date, bundle.run_id)
        if not html_path.exists():
            # Gera se ainda nao existe (bundle antigo)
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(generate_html(bundle), encoding="utf-8")
        console.print(f"[green]Abrindo:[/green] {html_path}")
        webbrowser.open(html_path.as_uri())
    else:
        md = generate_markdown(bundle)
        console.print(Markdown(md))


@app.command(name="list")
def list_bundles() -> None:
    """Lista todos os bundles salvos."""
    bundles = bundle_store.list_bundles()
    if not bundles:
        console.print("[yellow]Nenhum bundle encontrado.[/yellow]")
        return

    from rich.table import Table
    table = Table(title="Bundles Salvos", show_header=True, header_style="bold")
    table.add_column("Data", style="cyan")
    table.add_column("run_id")
    table.add_column("Tamanho")

    for p in bundles:
        size_kb = p.stat().st_size // 1024
        table.add_row(p.parent.name, p.stem[:20] + "...", f"{size_kb} KB")

    console.print(table)


if __name__ == "__main__":
    app()
