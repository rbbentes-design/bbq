"""
CLI: Macro Desk

Diagnóstico macro acionável sobre o bundle mais recente.
Identidade: Macro Desk — segundo agente do sistema.

Uso:
    python -m app.cli.invest
    python -m app.cli.invest --date 2026-03-29
    python -m app.cli.invest --focus "foque em crédito HY e curva de juros"
    python -m app.cli.invest --save
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from app.audit.logger import configure_logging

app = typer.Typer(name="desk", help="Macro Desk — diagnóstico macro acionável.")
# Força UTF-8 no Windows para suportar caracteres Unicode do Rich
_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") \
    if hasattr(sys.stdout, "buffer") else sys.stdout
console = Console(file=_stdout, highlight=False)

# Scores: -2 a +2
_SCORE_LABELS = {
    "rational":          "Rational Engine   ",
    "behavioral":        "Behavioral Engine ",
    "entropy":           "Entropy Engine    ",
    "valuation_gap":     "Valuation Gap     ",
    "regime_confidence": "Regime Confidence ",
}

_SCORE_COLORS = {
    2:  "bold green",
    1:  "green",
    0:  "yellow",
    -1: "red",
    -2: "bold red",
}


def _score_bar(value: int) -> Text:
    """Renderiza barra visual para score -2 a +2."""
    t = Text()
    for i in range(-2, 3):
        if i == value:
            color = _SCORE_COLORS.get(value, "white")
            t.append("█", style=color)
        else:
            t.append("░", style="dim")
        if i < 2:
            t.append(" ")
    t.append(f"  {value:+d}", style=_SCORE_COLORS.get(value, "white"))
    return t


def _render_scoreboard(scores: dict[str, int]) -> Panel:
    """Painel com os 5 scores do Macro Desk."""
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", min_width=20)
    t.add_column(min_width=14)

    for key, label in _SCORE_LABELS.items():
        val = scores.get(key)
        if val is not None:
            t.add_row(label, _score_bar(val))
        else:
            t.add_row(label, Text("—", style="dim"))

    return Panel(t, title="[bold cyan]Engines[/bold cyan]", border_style="cyan", expand=False)


def _render_header(run_date: str, focus: str | None) -> None:
    """Cabeçalho de identidade do Macro Desk."""
    now = datetime.now().strftime("%H:%M")
    title = Text()
    title.append("MACRO DESK", style="bold white on dark_cyan")
    title.append(f"  {run_date}  {now}", style="dim")
    if focus:
        title.append(f"\n  foco: {focus}", style="italic yellow")
    console.print()
    console.print(Rule(title, style="cyan"))
    console.print()


@app.command()
def main(
    date: str = typer.Option(None, "--date", "-d", help="Data YYYY-MM-DD (padrão: mais recente)"),
    focus: str = typer.Option(None, "--focus", "-f", help="Instrução específica do operador"),
    save: bool = typer.Option(False, "--save", "-s", help="Salva diagnóstico em .txt"),
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
        dirs = sorted([d for d in bundles_dir.iterdir() if d.is_dir()], reverse=True)
        if not dirs:
            console.print("[red]Nenhum bundle no workspace.[/red]")
            raise typer.Exit(1)
        target_dir = dirs[0]

    bundle_files = sorted(target_dir.glob("*.json"))
    bundle_files = [f for f in bundle_files if not any(
        s in f.name for s in ("_curation", "_summary", "_written", "_investment", "_decision_log", "_flow_inspector", "_macro_desk", "_desk_v2", "_analysis", "_isq_diagram")
    )]
    if not bundle_files:
        console.print(f"[red]Bundle JSON não encontrado em {target_dir}[/red]")
        raise typer.Exit(1)

    try:
        bundle = DailyIngestionBundle.model_validate(
            json.loads(bundle_files[-1].read_text(encoding="utf-8"))
        )
    except Exception as exc:
        console.print(f"[red]Erro ao carregar bundle: {exc}[/red]")
        raise typer.Exit(1)

    # ── Curation (opcional) ───────────────────────────────────────────────────
    curation = None
    for cf in sorted(target_dir.glob("*_curation.json"), reverse=True):
        try:
            curation = CurationResult.model_validate(
                json.loads(cf.read_text(encoding="utf-8"))
            )
            break
        except Exception:
            pass

    # ── Sinal do writer (se existir) ──────────────────────────────────────────
    if curation:
        sig = curation.narrative.primary_signal
        console.print(
            f"[dim]Writer signal:[/dim] [cyan]{sig.label}[/cyan] "
            f"[dim]({sig.confidence:.0%})[/dim]"
        )

    # ── Cabeçalho ─────────────────────────────────────────────────────────────
    _render_header(str(bundle.run_date), focus)

    with console.status("[cyan]Macro Desk processando...[/cyan]"):
        result = diagnose(bundle, curation, focus=focus)

    scores    = result.get("scores", {})
    narrative = result.get("narrative", "")

    # ── Scoreboard ────────────────────────────────────────────────────────────
    if scores:
        console.print(_render_scoreboard(scores))
        console.print()

    # ── Narrativa ─────────────────────────────────────────────────────────────
    console.print(Panel(
        narrative,
        title=f"[bold]Diagnóstico — {bundle.run_date}[/bold]",
        border_style="cyan",
        expand=True,
        padding=(1, 2),
    ))

    # ── Salva ──────────────────────────────────────────────────────────────────
    if save:
        run_id = bundle_files[-1].stem
        out = target_dir / f"{run_id}_macro_desk.txt"
        out.write_text(
            f"MACRO DESK — {bundle.run_date}\n"
            f"scores: {json.dumps(scores)}\n\n"
            f"{narrative}",
            encoding="utf-8",
        )
        console.print(f"\n[green]Salvo:[/green] {out.name}")


if __name__ == "__main__":
    app()
