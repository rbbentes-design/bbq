"""
CLI: agente auth check

Verifica se as sessoes do Chrome estao ativas para ZeroHedge e X.
Abre o browser com o perfil Chrome existente e testa navegacao.

Uso:
    cd agente/
    python -m app.cli.auth check
    python -m app.cli.auth check --headless
    python -m app.cli.auth info
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from app.audit.logger import configure_logging
from app.auth.browser_profile import chrome_profile_dir
from app.auth.session_manager import SessionManager
from app.config.settings import settings

app = typer.Typer(name="auth", help="Gerencia sessoes de autenticacao do browser.")
console = Console()


@app.command()
def check(
    headless: bool = typer.Option(False, "--headless", help="Rodar browser sem janela."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log detalhado."),
) -> None:
    """Verifica se as sessoes de ZeroHedge e X estao ativas."""
    configure_logging("DEBUG" if verbose else "WARNING")

    console.print("[bold]Verificando sessoes de autenticacao...[/bold]")
    console.print(f"  Chrome: [cyan]{settings.chrome_user_data_dir}[/cyan]")
    console.print(f"  Perfil: [cyan]{settings.chrome_profile}[/cyan]")
    console.print()

    profile_dir = chrome_profile_dir()
    if not settings.chrome_user_data_dir.exists():
        console.print(f"[red]ERRO: Chrome user_data_dir nao encontrado:[/red] {settings.chrome_user_data_dir}")
        raise typer.Exit(1)

    console.print("[yellow]Abrindo Chrome... (feche o Chrome antes de rodar)[/yellow]")

    with SessionManager(headless=headless) as sm:
        status = sm.check_all()

    # Tabela de resultado
    table = Table(title="Status das Sessoes", show_header=True, header_style="bold")
    table.add_column("Site", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Detalhes")

    for site, ok in [("ZeroHedge", status.zerohedge), ("X (Twitter)", status.x)]:
        icon = "[green]OK[/green]" if ok else "[red]FALHA[/red]"
        detail = status.errors.get(site.lower().split()[0], "")
        table.add_row(site, icon, detail)

    console.print(table)

    if status.all_ok:
        console.print("\n[green]Todas as sessoes ativas. Pronto para coletar.[/green]")
    else:
        console.print("\n[red]Uma ou mais sessoes falharam.[/red]")
        console.print("Dica: abra o Chrome manualmente, logue nos sites, feche o Chrome e rode novamente.")
        raise typer.Exit(1)


@app.command()
def info() -> None:
    """Mostra informacoes do perfil Chrome configurado."""
    table = Table(title="Configuracao do Chrome", show_header=True, header_style="bold")
    table.add_column("Campo", style="cyan")
    table.add_column("Valor")

    profile_dir = chrome_profile_dir()
    exists_icon = "[green]existe[/green]" if settings.chrome_user_data_dir.exists() else "[red]NAO encontrado[/red]"
    profile_exists = "[green]existe[/green]" if profile_dir.exists() else "[red]NAO encontrado[/red]"

    table.add_row("CHROME_USER_DATA_DIR", str(settings.chrome_user_data_dir))
    table.add_row("  -> existe?", exists_icon)
    table.add_row("CHROME_PROFILE", settings.chrome_profile)
    table.add_row("  -> dir do perfil", str(profile_dir))
    table.add_row("  -> existe?", profile_exists)
    table.add_row("PLAYWRIGHT_HEADLESS", str(settings.playwright_headless))

    console.print(table)


if __name__ == "__main__":
    app()
