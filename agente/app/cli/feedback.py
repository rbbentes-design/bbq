"""
CLI: agente feedback

Loop de aprendizagem — corrige narrativas e ensina o modelo.

Uso:
    python -m app.cli.feedback open
    python -m app.cli.feedback add --wrong "Iran narrative" --correct "Tariff escalation"
    python -m app.cli.feedback add --missed "Tariff escalation" --type missed_signal
    python -m app.cli.feedback list
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date

import typer
import yaml
from rich.console import Console
from rich.table import Table

from app.curation.models import CorrectionEntry, CorrectionsFile
from app.storage.paths import workspace

app = typer.Typer(name="feedback", help="Gerencia correcoes para aprendizagem do modelo.")
console = Console()


@app.command("open")
def open_editor() -> None:
    """Abre o corrections.yaml no editor padrao."""
    path = workspace.corrections_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _bootstrap(path)

    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if editor:
        subprocess.run([editor, str(path)])
    elif sys.platform == "win32":
        os.startfile(str(path))
    else:
        subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(path)])

    console.print(f"[dim]Arquivo: {path}[/dim]")


@app.command("add")
def add_correction(
    wrong: str = typer.Option(None, "--wrong", "-w", help="Label errado detectado."),
    correct: str = typer.Option(None, "--correct", "-c", help="Label correto."),
    missed: str = typer.Option(None, "--missed", "-m", help="Sinal que nao foi detectado."),
    correction_type: str = typer.Option(None, "--type", "-t",
                                         help="Tipo: wrong_narrative | missed_signal | wrong_score | hallucinated_quote"),
    note: str = typer.Option("", "--note", "-n", help="Nota adicional."),
) -> None:
    """Adiciona uma correcao diretamente via CLI."""
    path = workspace.corrections_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _bootstrap(path)

    # Inferir tipo e labels
    if missed:
        ctype = "missed_signal"
        original = None
        corrected = missed
        example_out = f"PRIMARY NARRATIVE: {missed}\nCONFIDENCE: 0.80\nDESCRIPTION: <descreva o sinal perdido>"
    elif wrong and correct:
        ctype = correction_type or "wrong_narrative"
        original = wrong
        corrected = correct
        example_out = f"PRIMARY NARRATIVE: {correct}\nCONFIDENCE: 0.85\nDESCRIPTION: <descricao correta>"
    else:
        console.print("[red]Use --wrong + --correct ou --missed.[/red]")
        raise typer.Exit(1)

    entry = CorrectionEntry(
        date=date.today().isoformat(),
        correction_type=ctype,  # type: ignore[arg-type]
        original_label=original,
        corrected_label=corrected,
        example_input="<cole o corpus relevante aqui>",
        example_output=example_out,
        notes=note,
    )

    # Carrega e adiciona
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cf = CorrectionsFile.model_validate(data) if data.get("corrections") is not None else CorrectionsFile()
    cf = cf.model_copy(update={"corrections": cf.corrections + [entry]})

    path.write_text(_to_yaml(cf), encoding="utf-8")

    console.print(f"[green]Correcao adicionada:[/green] [{ctype}] {corrected}")
    console.print(f"[dim]Edite o example_input em: {path}[/dim]")


@app.command("list")
def list_corrections(
    last: int = typer.Option(10, "--last", "-n", help="Ultimas N correcoes."),
) -> None:
    """Lista as correcoes registradas."""
    path = workspace.corrections_path()
    if not path.exists():
        console.print("[dim]Nenhuma correcao ainda.[/dim]")
        return

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cf = CorrectionsFile.model_validate(data)
    entries = sorted(cf.corrections, key=lambda c: c.date, reverse=True)[:last]

    if not entries:
        console.print("[dim]Nenhuma correcao ainda.[/dim]")
        return

    t = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    t.add_column("Data")
    t.add_column("Tipo")
    t.add_column("Original")
    t.add_column("Correto")
    t.add_column("Nota")

    for e in entries:
        t.add_row(
            e.date,
            e.correction_type,
            (e.original_label or "—")[:35],
            e.corrected_label[:35],
            (e.notes or "")[:30],
        )

    console.print(t)
    console.print(f"\n[dim]Total: {len(cf.corrections)} correcoes | {path}[/dim]")


def _bootstrap(path) -> None:
    from app.curation.corrections import create_corrections_template
    path.write_text(create_corrections_template(), encoding="utf-8")


def _to_yaml(cf: CorrectionsFile) -> str:
    data = {
        "version": cf.version,
        "corrections": [
            {k: v for k, v in e.model_dump().items() if v is not None and v != ""}
            for e in cf.corrections
        ],
    }
    return yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    app()
