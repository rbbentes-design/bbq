"""
CLI: agente sources

Gerencia filtros de fontes — contas X, feeds RSS.

Uso:
    python -m app.cli.sources x list
    python -m app.cli.sources x add @SquawkAlpha
    python -m app.cli.sources x remove @SquawkAlpha
    python -m app.cli.sources rss list
    python -m app.cli.sources rss add https://...
    python -m app.cli.sources rss remove https://...
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from app.storage.paths import workspace

app = typer.Typer(name="sources", help="Gerencia fontes de dados (X accounts, RSS feeds).")
x_app = typer.Typer(help="Contas X seguidas via notificacoes.")
rss_app = typer.Typer(help="Feeds RSS/Atom.")
app.add_typer(x_app, name="x")
app.add_typer(rss_app, name="rss")

console = Console()


def _sources_path() -> Path:
    return workspace.workspace / "sources.json"


def _load() -> dict:
    p = _sources_path()
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"x_accounts": [], "rss_feeds": []}


def _save(data: dict) -> None:
    p = _sources_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── X accounts ────────────────────────────────────────────────────────────────

@x_app.command("list")
def x_list() -> None:
    """Lista contas X monitoradas via notificacoes."""
    data = _load()
    accounts = data.get("x_accounts", [])
    if not accounts:
        console.print("[dim]Nenhuma conta configurada. O agente usa o filtro de notificacoes do X.[/dim]")
        return
    t = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    t.add_column("#")
    t.add_column("Handle")
    t.add_column("Nota")
    for i, a in enumerate(accounts, 1):
        t.add_row(str(i), a.get("handle", ""), a.get("note", ""))
    console.print(t)


@x_app.command("add")
def x_add(
    handle: str = typer.Argument(..., help="Handle da conta, ex: @SquawkAlpha"),
    note: str = typer.Option("", "--note", "-n"),
) -> None:
    """Adiciona uma conta X para ativar notificacoes."""
    if not handle.startswith("@"):
        handle = f"@{handle}"
    data = _load()
    accounts = data.setdefault("x_accounts", [])
    if any(a["handle"].lower() == handle.lower() for a in accounts):
        console.print(f"[yellow]{handle} ja esta na lista.[/yellow]")
        return
    accounts.append({"handle": handle, "note": note})
    _save(data)
    console.print(f"[green]Adicionado:[/green] {handle}")
    console.print("[dim]Lembre de ativar notificacoes para esta conta no X.[/dim]")


@x_app.command("remove")
def x_remove(handle: str = typer.Argument(..., help="Handle a remover")) -> None:
    """Remove uma conta X da lista."""
    if not handle.startswith("@"):
        handle = f"@{handle}"
    data = _load()
    before = len(data.get("x_accounts", []))
    data["x_accounts"] = [a for a in data.get("x_accounts", [])
                           if a["handle"].lower() != handle.lower()]
    if len(data["x_accounts"]) < before:
        _save(data)
        console.print(f"[green]Removido:[/green] {handle}")
    else:
        console.print(f"[yellow]{handle} nao encontrado.[/yellow]")


# ── RSS feeds ─────────────────────────────────────────────────────────────────

@rss_app.command("list")
def rss_list() -> None:
    """Lista feeds RSS configurados."""
    from app.providers.rss_feed import DEFAULT_FEEDS
    data = _load()
    feeds = data.get("rss_feeds", []) or DEFAULT_FEEDS

    t = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    t.add_column("#")
    t.add_column("Feed URL")
    t.add_column("Tipo")
    for i, f in enumerate(feeds, 1):
        is_default = f in DEFAULT_FEEDS
        t.add_row(str(i), f[:70], "[dim]default[/dim]" if is_default else "custom")
    console.print(t)


@rss_app.command("add")
def rss_add(url: str = typer.Argument(..., help="URL do feed RSS/Atom")) -> None:
    """Adiciona um feed RSS."""
    data = _load()
    feeds = data.setdefault("rss_feeds", [])
    if url in feeds:
        console.print(f"[yellow]Feed ja existe.[/yellow]")
        return
    feeds.append(url)
    _save(data)
    console.print(f"[green]Feed adicionado:[/green] {url[:70]}")


@rss_app.command("remove")
def rss_remove(url: str = typer.Argument(..., help="URL a remover")) -> None:
    """Remove um feed RSS."""
    data = _load()
    before = len(data.get("rss_feeds", []))
    data["rss_feeds"] = [f for f in data.get("rss_feeds", []) if f != url]
    if len(data["rss_feeds"]) < before:
        _save(data)
        console.print(f"[green]Removido:[/green] {url[:70]}")
    else:
        console.print(f"[yellow]Feed nao encontrado.[/yellow]")


if __name__ == "__main__":
    app()
