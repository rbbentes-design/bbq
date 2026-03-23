"""
CLI: agente auth

Comandos:
  check   - verifica se as sessoes estao ativas (usa perfil salvo)
  login   - abre browser para login manual (primeira vez ou sessao expirada)
  info    - mostra configuracao do perfil

Uso:
    cd agente/
    python -m app.cli.auth login
    python -m app.cli.auth check
    python -m app.cli.auth info
"""

from __future__ import annotations

import typer
from playwright.sync_api import sync_playwright
from rich.console import Console
from rich.table import Table

from app.audit.logger import configure_logging
from app.auth.bootstrap import login_x, login_zerohedge
from app.auth.browser_profile import open_context, profile_dir
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

    console.print("[bold]Verificando sessoes...[/bold]")
    console.print(f"  Perfil: [cyan]{profile_dir()}[/cyan]")

    with SessionManager(headless=headless) as sm:
        status = sm.check_all()

    table = Table(title="Status das Sessoes", show_header=True, header_style="bold")
    table.add_column("Site", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Detalhes")

    zh_icon = "[green]OK[/green]" if status.zerohedge else "[red]FALHA[/red]"
    x_icon = "[green]OK[/green]" if status.x else "[red]FALHA[/red]"
    table.add_row("ZeroHedge", zh_icon, status.errors.get("zerohedge", ""))
    table.add_row("X (Twitter)", x_icon, status.errors.get("x", ""))

    console.print(table)

    if status.all_ok:
        console.print("\n[green]Todas as sessoes ativas. Pronto para coletar.[/green]")
    else:
        console.print("\n[yellow]Sessoes inativas. Rode:[/yellow] python -m app.cli.auth login")
        raise typer.Exit(1)


@app.command()
def login(
    site: str = typer.Argument(
        default="all",
        help="Site para logar: 'zerohedge', 'x', ou 'all' (padrao).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log detalhado."),
) -> None:
    """
    Abre o browser para login manual.

    O browser abre com o perfil do workspace (nao o Chrome do sistema).
    Faca o login normalmente — os cookies ficam salvos para proximas execucoes.
    """
    configure_logging("DEBUG" if verbose else "WARNING")

    valid_sites = {"zerohedge", "x", "all"}
    if site not in valid_sites:
        console.print(f"[red]Site invalido:[/red] {site}. Use: zerohedge, x, ou all")
        raise typer.Exit(1)

    console.print(f"[bold]Login manual — site: {site}[/bold]")
    console.print(f"  Perfil: [cyan]{profile_dir()}[/cyan]")
    console.print("[yellow]O browser vai abrir. Faca o login e aguarde a confirmacao.[/yellow]\n")

    results: dict[str, bool] = {}

    with sync_playwright() as p:
        context = open_context(p, headless=False)
        try:
            if site in ("zerohedge", "all"):
                results["zerohedge"] = login_zerohedge(context)

            if site in ("x", "all"):
                results["x"] = login_x(context)
        finally:
            context.close()

    # Resultado
    table = Table(title="Resultado do Login", show_header=True, header_style="bold")
    table.add_column("Site", style="cyan")
    table.add_column("Status", justify="center")

    all_ok = True
    for s, ok in results.items():
        icon = "[green]OK[/green]" if ok else "[red]FALHA[/red]"
        table.add_row(s, icon)
        if not ok:
            all_ok = False

    console.print(table)

    if all_ok:
        console.print("\n[green]Login concluido! Rode `auth check` para confirmar.[/green]")
    else:
        console.print("\n[red]Um ou mais logins falharam. Tente novamente.[/red]")
        raise typer.Exit(1)


@app.command()
def info() -> None:
    """Mostra informacoes do perfil Playwright configurado."""
    table = Table(title="Configuracao do Browser", show_header=True, header_style="bold")
    table.add_column("Campo", style="cyan")
    table.add_column("Valor")

    pd = profile_dir()
    exists = "[green]existe[/green]" if pd.exists() else "[yellow]sera criado no primeiro uso[/yellow]"

    table.add_row("Perfil (workspace)", str(pd))
    table.add_row("  -> existe?", exists)
    table.add_row("PLAYWRIGHT_HEADLESS", str(settings.playwright_headless))
    table.add_row("Workspace", str(settings.workspace_dir))

    console.print(table)


if __name__ == "__main__":
    app()
