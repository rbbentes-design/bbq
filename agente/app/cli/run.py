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

# Registra sub-app live
from app.cli.live import app as live_app  # noqa: E402
app.add_typer(live_app, name="live")
console = Console()


@app.command()
def ingest(
    headless:  bool = typer.Option(False, "--headless",  help="Browser sem janela."),
    no_open:   bool = typer.Option(False, "--no-open",   help="Nao abre o HTML no browser."),
    no_live:   bool = typer.Option(False, "--no-live",   help="Nao inicia loop de precos live apos ingestao."),
    interval:  int  = typer.Option(60,   "--interval",  help="Segundos entre refreshes de preco (live loop)."),
    verbose:   bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Coleta ZeroHedge + X, roda curacao LLM, abre relatorio e mantém precos atualizados."""
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
    desk2_path = None
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
        console.print(f"[green]Macro Desk (v1):[/green] {desk_path.name}")
        if not no_open:
            webbrowser.open(desk_path.as_uri())
    except Exception as exc:
        console.print(f"[yellow]Macro Desk v1 falhou: {exc}[/yellow]")

    try:
        from app.views.macro_desk_v2 import save_macro_desk_v2
        with console.status("[cyan]MacroDesk v2 (grafo interativo)...[/cyan]"):
            desk2_path = save_macro_desk_v2(bundle, curation_obj)
        console.print(f"[green]MacroDesk v2:[/green] {desk2_path.name}")
        if not no_open:
            webbrowser.open(desk2_path.as_uri())
    except Exception as exc:
        console.print(f"[yellow]MacroDesk v2 falhou: {exc}[/yellow]")

    # Week Ahead Brief — gerado automaticamente às segundas
    import datetime as _dt
    if _dt.date.today().weekday() == 0:  # segunda-feira
        try:
            from app.views.week_ahead_brief import save_week_ahead_brief
            with console.status("[cyan]Week Ahead Brief...[/cyan]"):
                brief_path = save_week_ahead_brief(bundle, curation_path)
            console.print(f"[green]Week Ahead Brief:[/green] {brief_path.name}")
            if not no_open:
                webbrowser.open(brief_path.as_uri())
        except Exception as exc:
            console.print(f"[yellow]Week Ahead Brief falhou: {exc}[/yellow]")

    # ── Portfolio Allocation ──────────────────────────────────────────────────
    fi_path, portfolio, signals = _run_portfolio_after_ingest(bundle)

    # Abre o Flow Inspector — usa path do pipeline ou busca o mais recente no disco
    # Retorna o path efetivamente aberto (pode ser fallback do disco)
    opened_path = _open_flow_inspector(fi_path, bundle, open_browser=not no_open)

    console.print(Panel.fit(
        f"[bold green]Run concluido[/bold green] — {bundle.run_date} | id={bundle.run_id[:12]}",
        border_style="green",
    ))

    if bundle.audit_summary.errors > 0:
        raise typer.Exit(1)

    # ── Live price loop ───────────────────────────────────────────────────────
    # Usa o path aberto (pipeline ou disco) para o live loop reescrever o arquivo
    live_path = opened_path or fi_path
    if not no_live and portfolio and live_path:
        _run_live_loop(bundle, live_path, portfolio, signals, interval=interval, desk2_path=desk2_path)


def _run_portfolio_after_ingest(bundle):
    """Roda o pipeline de portfolio. Retorna (html_path, portfolio, signals)."""
    try:
        from app.pipeline.portfolio_pipeline import run_portfolio_pipeline
        with console.status("[cyan]Portfolio: computando sinais alpha + otimizacao...[/cyan]"):
            portfolio, signals, html_path = run_portfolio_pipeline(
                bundle, save_html=True, live_mode=True,
            )
        _print_portfolio_summary(portfolio)
        return html_path, portfolio, signals
    except Exception as exc:
        import traceback
        console.print(f"[red]Portfolio pipeline falhou: {exc}[/red]")
        console.print(f"[dim]{traceback.format_exc()[-600:]}[/dim]")
        return None, None, {}


def _open_flow_inspector(
    fi_path: "Path | None",
    bundle,
    open_browser: bool = True,
) -> "Path | None":
    """
    Abre o Flow Inspector no browser.
    Fallback: busca HTML mais recente no disco.
    Retorna o Path efetivamente usado (para o live loop reescrever).
    """
    target = Path(fi_path) if fi_path and Path(fi_path).exists() else None

    if target is None:
        # Busca o flow_inspector mais recente no workspace
        try:
            from app.storage.paths import workspace
            ws_path = Path(workspace.bundles)
            candidates = sorted(
                ws_path.rglob("*flow_inspector*.html"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                target = candidates[0]
                console.print(f"[dim]Flow Inspector (disco): {target.name}[/dim]")
        except Exception:
            pass

    if target and target.exists():
        if open_browser:
            console.print(f"[green]Abrindo Flow Inspector:[/green] {target.name}")
            webbrowser.open(target.as_uri())
        return target
    else:
        console.print("[yellow]Flow Inspector HTML nao encontrado.[/yellow]")
        return None


def _run_live_loop(
    bundle,
    fi_path: "Path | None",
    portfolio,
    signals: dict,
    interval: int = 60,
    desk2_path: "Path | None" = None,
) -> None:
    """
    Loop de refresh de precos — roda até Ctrl+C.
    Usa portfolio/signals já computados (sem re-rodar o pipeline).
    Rebuild completo a cada 15 ciclos (~15min).
    """
    import time as _time

    console.print(
        f"\n[bold cyan]Live loop ativo[/bold cyan] — refresh a cada [bold]{interval}s[/bold] | "
        f"[dim]Ctrl+C para parar[/dim]\n"
    )

    from app.providers.market_prices_live import refresh as refresh_prices

    # Usa portfolio/signals já computados — sem nova chamada ao pipeline
    _cached_portfolio = portfolio
    _cached_signals   = signals
    _cached_rrg       = getattr(portfolio, "_rrg_result",   None)
    _cached_pairs     = getattr(portfolio, "_pairs_result", None)

    # Constrói graph_data uma vez (network/MST pesado — não rebuild a cada ciclo)
    _cached_graph_data = None
    if desk2_path:
        try:
            from app.desk.graph_engine import build_from_bundle
            _cached_graph_data = build_from_bundle(
                bundle, skip_anatomy=True, skip_options=True, skip_prob=True
            )
        except Exception as exc:
            console.print(f"[yellow]Graph cache falhou: {exc}[/yellow]")

    # Computa options uma vez (já temos os preços)
    _cached_options = None
    if _cached_portfolio and _cached_portfolio.positions:
        try:
            from app.analysis.options_strategy import compute_options_strategy
            _mkt = {k: v for k, v in bundle.market_prices.items()
                    if not k.startswith("__") and isinstance(v, dict)}
            _cached_options = compute_options_strategy(
                _cached_portfolio.positions, _mkt, budget=100_000,
            )
        except Exception as exc:
            console.print(f"[yellow]Options cache falhou: {exc}[/yellow]")

    cycle = 0
    try:
        while True:
            cycle += 1
            t0 = _time.time()

            # Refresh precos (IBKR snapshot ou yfinance fast_info)
            refresh_prices(bundle.market_prices)
            refreshed_at = bundle.market_prices.get("__refreshed_at__", "?")

            # Atualiza P&L snapshot
            pnl_str = ""
            try:
                from app.analysis.portfolio_tracker import update_snapshot
                snap = update_snapshot(bundle.market_prices)
                if snap:
                    color = "green" if snap["total_pnl"] >= 0 else "red"
                    pnl_str = (
                        f" | P&L=[{color}]${snap['total_pnl']:+,.0f} "
                        f"({snap['pnl_pct']:+.2%})[/{color}]"
                    )
            except Exception:
                pass

            # Regenera flow inspector HTML com preco/P&L atualizados
            if fi_path and _cached_portfolio and _cached_signals:
                try:
                    from app.views.flow_inspector import generate_flow_inspector_html
                    fi_html = generate_flow_inspector_html(
                        _cached_portfolio, _cached_signals,
                        bundle_date=str(bundle.run_date),
                        live_mode=True,
                        refresh_interval=interval + 2,
                        options_strategy=_cached_options,
                        rrg_result=_cached_rrg,
                        pairs_result=_cached_pairs,
                        market_prices=bundle.market_prices,
                    )
                    fi_path.write_text(fi_html, encoding="utf-8")
                except Exception:
                    pass

            # Regenera MacroDesk v2 com precos atualizados (usa graph_data cacheado)
            if desk2_path and _cached_graph_data:
                try:
                    from app.views.macro_desk_v2 import generate_macro_desk_v2_html
                    _d2_html = generate_macro_desk_v2_html(bundle, graph_data=_cached_graph_data, live_mode=True)
                    desk2_path.write_text(_d2_html, encoding="utf-8")
                except Exception:
                    pass

            elapsed = _time.time() - t0
            console.print(
                f"[dim]#{cycle:03d}[/dim] {refreshed_at} "
                f"[dim]{elapsed:.1f}s[/dim]{pnl_str}"
            )

            # A cada 15 ciclos (~15min): rebuild completo de sinais + portfolio
            if cycle % 15 == 0 and fi_path:
                console.print("[yellow]Rebuild completo (sinais + portfolio)...[/yellow]")
                try:
                    from app.pipeline.portfolio_pipeline import run_portfolio_pipeline
                    _cached_portfolio, _cached_signals, _ = run_portfolio_pipeline(
                        bundle, save_html=False,
                    )
                    _cached_rrg   = getattr(_cached_portfolio, "_rrg_result",   None)
                    _cached_pairs = getattr(_cached_portfolio, "_pairs_result",  None)
                    if _cached_portfolio and _cached_portfolio.positions:
                        from app.analysis.options_strategy import compute_options_strategy
                        _mkt2 = {k: v for k, v in bundle.market_prices.items()
                                 if not k.startswith("__") and isinstance(v, dict)}
                        _cached_options = compute_options_strategy(
                            _cached_portfolio.positions, _mkt2, budget=100_000,
                        )
                    console.print("[green]Rebuild done[/green]")
                except Exception as exc:
                    console.print(f"[yellow]Rebuild falhou: {exc}[/yellow]")

            _time.sleep(max(1, interval - elapsed))

    except KeyboardInterrupt:
        console.print("\n[yellow]Live loop encerrado.[/yellow]")


def _print_portfolio_summary(portfolio) -> None:
    from rich.table import Table as RTable
    regime_color = {"bull": "green", "neutral": "yellow", "bear": "red"}.get(
        portfolio.regime_mode, "white")

    t = RTable(show_header=True, header_style="bold magenta", box=None, padding=(0, 1))
    t.add_column("Ativo")
    t.add_column("Dir.", justify="center")
    t.add_column("Alloc%", justify="right")
    t.add_column("USD", justify="right")
    t.add_column("Entrada", justify="right")
    t.add_column("Stop", justify="right", style="red")
    t.add_column("Target", justify="right", style="green")
    t.add_column("R:R", justify="right")
    t.add_column("E[R]", justify="right")

    cash_pct = portfolio.by_direction.get("cash_pct", 0)

    for p in portfolio.positions[:12]:
        dir_color = "green" if p.direction == "long" else "red"
        entry  = f"${p.entry_price:,.2f}"  if getattr(p, "entry_price", 0) > 0 else "—"
        stop   = f"${p.stop_loss:,.2f}"   if getattr(p, "stop_loss", 0) > 0 else "—"
        target = f"${p.take_profit:,.2f}" if getattr(p, "take_profit", 0) > 0 else "—"
        rr     = getattr(p, "risk_reward", 0)
        rr_str = f"{rr:.1f}:1" if rr > 0 else "—"
        t.add_row(
            f"[bold]{p.ticker}[/bold]",
            f"[{dir_color}]{'L' if p.direction=='long' else 'S'}[/{dir_color}]",
            f"{abs(p.allocation_pct):.1%}",
            f"${abs(p.allocation_usd):,.0f}",
            entry, stop, target, rr_str,
            f"{p.expected_return_ann:+.1%}",
        )

    console.print()
    cash_str = f" | Cash={cash_pct:.0%}" if cash_pct > 0.02 else ""
    console.print(Panel(
        t,
        title=f"[bold magenta]Portfolio[/bold magenta] [{regime_color}]{portfolio.regime_mode.upper()}[/{regime_color}] | E[R]={portfolio.expected_return_ann:+.1%} | Sharpe={portfolio.sharpe:.2f}{cash_str}",
        border_style="magenta",
    ))


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
def pdf(
    date_str: str = typer.Option(None, "--date", "-d", help="Data YYYY-MM-DD (padrão: hoje)"),
    lang: str = typer.Option("pt", "--lang", "-l", help="Idioma: pt (padrão) ou en (clientes internacionais)"),
    both: bool = typer.Option(False, "--both", help="Gera PT e EN juntos"),
    no_open: bool = typer.Option(False, "--no-open", help="Não abre o PDF."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Exporta Week Ahead Brief em PDF com branding Gulfstream Capital."""
    import json as _json
    from datetime import date as _date
    from pathlib import Path as _Path

    configure_logging("DEBUG" if verbose else "INFO")

    target_date = _date.fromisoformat(date_str) if date_str else _date.today()

    console.print(Panel.fit(
        f"[bold cyan]PDF Export[/bold cyan] — Gulfstream Capital · [yellow]{target_date}[/yellow]"
    ))

    from app.storage.bundle_store import bundle_store
    bundles = [
        p for p in bundle_store.list_bundles()
        if str(target_date) in str(p) and "_" not in p.stem
    ]
    if not bundles:
        console.print(f"[red]Nenhum bundle encontrado para {target_date}.[/red]")
        raise typer.Exit(1)

    bundle_path = bundles[0]
    from app.models.daily_ingestion_bundle import DailyIngestionBundle
    bundle = DailyIngestionBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))

    # Curation mais recente
    curation_dir = bundle_path.parent
    cur_files = sorted(curation_dir.glob("*_curation.json"), reverse=True)
    curation_path = str(cur_files[0]) if cur_files else None
    if curation_path:
        console.print(f"[dim]Curação: {_Path(curation_path).name}[/dim]")
    else:
        console.print("[yellow]Sem curação disponível — exportando sem texto editorial.[/yellow]")

    from app.views.week_ahead_pdf import save_week_ahead_pdf

    langs = ["pt", "en"] if both else [lang]
    for lng in langs:
        with console.status(f"[cyan]Gerando PDF [{lng.upper()}]...[/cyan]"):
            try:
                pdf_path = save_week_ahead_pdf(bundle, curation_path, lang=lng)
                console.print(f"[green]PDF [{lng.upper()}]:[/green] {pdf_path.name} ({pdf_path.stat().st_size // 1024} KB)")
                if not no_open:
                    import webbrowser as _wb
                    _wb.open(pdf_path.as_uri())
            except Exception as exc:
                console.print(f"[red]PDF [{lng.upper()}] falhou: {exc}[/red]")


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


@app.command()
def writer(
    date_str: str = typer.Option(None, "--date", "-d", help="Data YYYY-MM-DD (padrão: hoje)"),
    mode: str = typer.Option(None, "--mode", "-m", help="Força modo editorial (week_ahead, growth, etc.)"),
    no_open: bool = typer.Option(False, "--no-open", help="Não abre o HTML no browser."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Roda só writer + curação no bundle mais recente do dia (sem re-coletar dados)."""
    import json as _json
    from datetime import date as _date
    from pathlib import Path as _Path

    configure_logging("DEBUG" if verbose else "INFO")

    target_date = _date.fromisoformat(date_str) if date_str else _date.today()

    console.print(Panel.fit(
        f"[bold cyan]Writer[/bold cyan] — carregando bundle de [yellow]{target_date}[/yellow]"
    ))

    # ── Carrega bundle mais recente do dia ─────────────────────────────────────
    from app.storage.bundle_store import bundle_store
    # Bundles reais: ULID.json sem underscore no stem (ex: 01KMZ5D....json)
    bundles = [
        p for p in bundle_store.list_bundles()
        if str(target_date) in str(p) and "_" not in p.stem
    ]
    if not bundles:
        console.print(f"[red]Nenhum bundle encontrado para {target_date}. Rode 'agente run ingest' primeiro.[/red]")
        raise typer.Exit(1)

    bundle_path = bundles[0]  # mais recente
    console.print(f"[dim]Bundle: {bundle_path.name}[/dim]")

    from app.models.daily_ingestion_bundle import DailyIngestionBundle
    bundle = DailyIngestionBundle.model_validate_json(
        bundle_path.read_text(encoding="utf-8")
    )

    # ── Curação + Writer ───────────────────────────────────────────────────────
    from app.curation.orchestrator import run_curation
    from app.utils.timestamps import new_ulid
    run_id = new_ulid()

    with console.status("[cyan]Rodando curação + writer...[/cyan]"):
        try:
            curation_result = run_curation(bundle, run_id, str(target_date), mode_override=mode or None)
        except Exception as exc:
            console.print(f"[red]Curação falhou: {exc}[/red]")
            raise typer.Exit(1)

    # ── Salva resultado de curação ────────────────────────────────────────────
    from app.storage.paths import workspace
    curation_path = workspace.curation_result_path(target_date, run_id)
    curation_path.parent.mkdir(parents=True, exist_ok=True)
    curation_path.write_text(
        curation_result.model_dump_json(indent=2), encoding="utf-8"
    )
    console.print(f"[green]Curação salva:[/green] {curation_path.name}")

    # ── Exibe resumo ──────────────────────────────────────────────────────────
    _print_curation_summary(str(curation_path))

    # ── Writer Brief (todos os modos) ─────────────────────────────────────────
    try:
        from app.views.week_ahead_brief import save_writer_brief
        with console.status("[cyan]Writer Brief...[/cyan]"):
            brief_path = save_writer_brief(bundle, str(curation_path))
        console.print(f"[green]Writer Brief:[/green] {brief_path.name}")
        if not no_open:
            import webbrowser as _wb2
            _wb2.open(brief_path.as_uri())
    except Exception as exc:
        console.print(f"[yellow]Writer Brief falhou: {exc}[/yellow]")

    # ── Abre relatórios ───────────────────────────────────────────────────────
    if not no_open:
        import webbrowser as _wb
        for key in ("html_report", "macro_desk"):
            p_str = bundle.artifact_paths.get(key)
            if p_str and _Path(p_str).exists():
                _wb.open(_Path(p_str).as_uri())


@app.command()
def allocate(
    date_str: str = typer.Option(None, "--date", "-d", help="Data YYYY-MM-DD (padrão: hoje)"),
    budget: float = typer.Option(100_000.0, "--budget", "-b", help="Capital disponível (padrão: $100,000)"),
    regime: str = typer.Option(None, "--regime", "-r", help="Regime override: bull|neutral|bear"),
    no_open: bool = typer.Option(False, "--no-open", help="Não abre o HTML."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Roda modelo de alocação autonomo no bundle mais recente do dia."""
    from datetime import date as _date

    configure_logging("DEBUG" if verbose else "INFO")

    target_date = _date.fromisoformat(date_str) if date_str else _date.today()

    console.print(Panel.fit(
        f"[bold magenta]Portfolio Allocation[/bold magenta] — [yellow]{target_date}[/yellow] | Capital: ${budget:,.0f}"
    ))

    from app.storage.bundle_store import bundle_store
    bundles = [
        p for p in bundle_store.list_bundles()
        if str(target_date) in str(p) and "_" not in p.stem
    ]
    if not bundles:
        console.print(f"[red]Nenhum bundle para {target_date}. Rode 'agente run ingest' primeiro.[/red]")
        raise typer.Exit(1)

    bundle_path = bundles[0]
    console.print(f"[dim]Bundle: {bundle_path.name}[/dim]")

    from app.models.daily_ingestion_bundle import DailyIngestionBundle
    bundle = DailyIngestionBundle.model_validate_json(
        bundle_path.read_text(encoding="utf-8")
    )

    with console.status("[cyan]Computando sinais alpha + otimizacao de portfolio...[/cyan]"):
        from app.pipeline.portfolio_pipeline import run_portfolio_pipeline
        portfolio, signals, html_path = run_portfolio_pipeline(
            bundle,
            budget=budget,
            regime_override=regime or None,
            save_html=True,
            live_mode=True,
            out_dir=bundle_path.parent,
        )

    _print_portfolio_summary(portfolio)

    if html_path:
        console.print(f"\n[green]Flow Inspector:[/green] {html_path.name}")
        if not no_open:
            webbrowser.open(html_path.as_uri())


@app.command()
def positions(
    history: bool = typer.Option(False, "--history", "-h", help="Mostra histórico de trades fechados."),
    perf: bool = typer.Option(False, "--perf", "-p", help="Mostra resumo de performance."),
) -> None:
    """Mostra posições abertas com entrada, preço atual e P&L."""
    from app.analysis.portfolio_tracker import (
        get_trade_history, get_performance_summary, print_open_positions
    )

    if perf:
        p = get_performance_summary()
        from rich.table import Table as RTable
        t = RTable(show_header=False, box=None, padding=(0, 2))
        t.add_column("Métrica")
        t.add_column("Valor", justify="right")
        for k, v in p.items():
            color = "green" if isinstance(v, (int, float)) and v > 0 else "red" if isinstance(v, (int, float)) and v < 0 else "white"
            formatted = f"${v:,.2f}" if "pnl" in k or "profit" in k or "loss" in k or "win" in k or "loss" in k else (f"{v:.1%}" if k == "win_rate" else str(v))
            t.add_row(k.replace("_", " ").title(), f"[{color}]{formatted}[/{color}]")
        console.print(Panel(t, title="[bold]Performance Summary[/bold]", border_style="cyan"))
        return

    if history:
        trades = get_trade_history()
        closed = [t for t in trades if t["status"] == "closed"]
        if not closed:
            console.print("[dim]Sem trades fechados.[/dim]")
            return
        from rich.table import Table as RTable
        t = RTable(show_header=True, header_style="bold", box=None, padding=(0, 1))
        t.add_column("ID"); t.add_column("Ticker"); t.add_column("Dir.")
        t.add_column("Entrada"); t.add_column("Data"); t.add_column("Saída")
        t.add_column("Data Saída"); t.add_column("Shares"); t.add_column("P&L $", justify="right")
        t.add_column("P&L %", justify="right")
        for tr in closed[:50]:
            pnl = tr.get("pnl_realized", 0) or 0
            c = "green" if pnl >= 0 else "red"
            t.add_row(
                tr["trade_id"], f"[bold]{tr['ticker']}[/bold]",
                f"{'▲' if tr['direction']=='long' else '▼'}",
                f"${tr['entry_price']:.2f}", f"{tr['entry_date']} {tr['entry_time']}",
                f"${tr.get('exit_price',0):.2f}", f"{tr.get('exit_date','')} {tr.get('exit_time','')}",
                f"{tr['shares']:.2f}",
                f"[{c}]${pnl:+,.0f}[/{c}]",
                f"[{c}]{tr.get('pnl_pct',0):+.2%}[/{c}]",
            )
        console.print(Panel(t, title=f"[bold]Trade History[/bold] — {len(closed)} trades", border_style="cyan"))
        return

    print_open_positions(console)


if __name__ == "__main__":
    app()
