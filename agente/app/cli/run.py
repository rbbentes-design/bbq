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

    # ── Curation object (compartilhado por brief e MacroDesk) ─────────────────
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

    # ── Writer Brief — gerado ANTES do MacroDesk para embutir na aba editorial ──
    brief_path = None
    try:
        from app.views.week_ahead_brief import save_writer_brief
        with console.status("[cyan]HTML diario (Wiki Recap)...[/cyan]"):
            brief_path = save_writer_brief(bundle, curation_path)
        console.print(f"[green]HTML diario:[/green] {brief_path.name}")
    except Exception as exc:
        console.print(f"[yellow]HTML diario falhou: {exc}[/yellow]")

    # ── MacroDesk v2 ─────────────────────────────────────────────────────────
    # Flow Inspector está embutido no MacroDesk — não abre separado
    desk2_path = None
    try:
        from app.views.macro_desk_v2 import save_macro_desk_v2
        with console.status("[cyan]MacroDesk v2 (grafo interativo)...[/cyan]"):
            desk2_path = save_macro_desk_v2(bundle, curation_obj)
        console.print(f"[green]MacroDesk v2:[/green] {desk2_path.name}")
    except Exception as exc:
        import traceback
        console.print(f"[red]MacroDesk v2 falhou: {exc}[/red]")
        console.print(f"[dim]{traceback.format_exc()[-800:]}[/dim]")

    # ── Verificação Bloomberg Live ─────────────────────────────────────────────
    # Obrigatório antes de prosseguir — emite aviso explícito se indisponível
    _bloomberg_live_ok = _check_bloomberg_live(bundle)

    # ── Portfolio Allocation (sem abrir Flow Inspector standalone) ────────────
    fi_path, portfolio, signals = _run_portfolio_after_ingest(bundle)

    # ── Regenera MacroDesk com portfolio + options (versão completa) ──────────
    if portfolio is not None:
        try:
            from app.providers.options_store import options_store as _opts_s
            _opts_snap = _opts_s.load_latest()
            from app.views.macro_desk_v2 import save_macro_desk_v2 as _smv2
            with console.status("[cyan]MacroDesk v2 (completo com portfolio)...[/cyan]"):
                desk2_path = _smv2(bundle, curation_obj, portfolio=portfolio, options_snapshot=_opts_snap)
            console.print(f"[green]MacroDesk v2 (completo):[/green] {desk2_path.name}")
        except Exception as exc:
            console.print(f"[yellow]MacroDesk v2 (full) falhou: {exc}[/yellow]")

    # ── Abre UM único HTML — MacroDesk tem prioridade (tem auto-refresh) ────────
    _final_html = desk2_path or brief_path
    # Fallback: se nenhum HTML foi gerado hoje, abre o mais recente do workspace
    if not _final_html:
        try:
            from app.storage.paths import workspace
            _all = sorted(
                list(Path(workspace.bundles).glob("**/*_desk_v2.html")) +
                list(Path(workspace.bundles).glob("**/*_brief.html")),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if _all:
                _final_html = _all[0]
                console.print(f"[yellow]Abrindo ultimo HTML disponivel: {_final_html.name}[/yellow]")
        except Exception:
            pass
    if _final_html and not no_open:
        webbrowser.open(_final_html.as_uri())

    console.print(Panel.fit(
        f"[bold green]Run concluido[/bold green] — {bundle.run_date} | id={bundle.run_id[:12]}",
        border_style="green",
    ))

    if bundle.audit_summary.errors > 0:
        raise typer.Exit(1)

    # ── Live price loop — MacroDesk como alvo principal (tem auto-refresh) ──────
    _live_target = desk2_path or brief_path or _final_html
    if not no_live and _live_target:
        _run_live_loop(bundle, fi_path, portfolio, signals, interval=interval, desk2_path=_live_target)


def _check_bloomberg_live(bundle) -> bool:
    """
    Verifica se os preços Bloomberg foram carregados no bundle.
    Emite aviso explícito se indisponível — nunca falha silenciosamente.
    """
    prices = bundle.market_prices or {}
    bbg_count = sum(
        1 for v in prices.values()
        if isinstance(v, dict) and v.get("source") == "bloomberg_csv"
    )
    if bbg_count >= 10:
        console.print(f"[dim]Bloomberg Live: {bbg_count} tickers carregados[/dim]")
        return True
    else:
        console.print(Panel(
            "[bold yellow][!] Macro Desk Live nao carregado automaticamente.[/bold yellow]\n"
            "Necessário puxar cotações Bloomberg antes de continuar a análise completa.\n\n"
            "[dim]Para corrigir: rode o script BQuant (bql_export.py) e aguarde o zip ser extraído.[/dim]",
            border_style="yellow",
            title="[yellow]Aviso Operacional[/yellow]",
        ))
        return False


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


# Flow Inspector está embutido no MacroDesk — não existe mais como janela separada


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

    # Carrega options snapshot uma vez (para incluir na aba Opções do live)
    _cached_options_snap = None
    try:
        from app.providers.options_store import options_store as _opts_live
        _cached_options_snap = _opts_live.load_latest()
    except Exception:
        pass

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

    # Rastreia mtime dos CSVs BQL para detectar novos dados Bloomberg
    def _bql_current_mtime() -> float:
        try:
            from app.providers.bql_csv import BQL_DATA_DIR
            from pathlib import Path as _P
            _dated = sorted(_P(BQL_DATA_DIR).glob("meta_*.csv"), reverse=True)
            _meta = _dated[0] if _dated else _P(BQL_DATA_DIR) / "meta.csv"
            return _meta.stat().st_mtime if _meta.exists() else 0.0
        except Exception:
            return 0.0

    _last_bbg_mtime = _bql_current_mtime()

    cycle = 0
    try:
        while True:
            cycle += 1
            t0 = _time.time()

            # Refresh precos (IBKR snapshot ou yfinance fast_info)
            refresh_prices(bundle.market_prices)
            refreshed_at = bundle.market_prices.get("__refreshed_at__", "?")

            # Detecta se Bloomberg CSV foi atualizado neste ciclo
            _cur_bbg_mtime = _bql_current_mtime()
            _bbg_updated = _cur_bbg_mtime > _last_bbg_mtime
            if _bbg_updated:
                _last_bbg_mtime = _cur_bbg_mtime

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

            # Regenera MacroDesk com precos + portfolio atualizados todo ciclo
            if desk2_path and _cached_graph_data:
                try:
                    from app.views.macro_desk_v2 import generate_macro_desk_v2_html
                    _d2_html = generate_macro_desk_v2_html(
                        bundle,
                        graph_data=_cached_graph_data,
                        live_mode=True,
                        portfolio=_cached_portfolio,
                        options_snapshot=_cached_options_snap,
                    )
                    desk2_path.write_text(_d2_html, encoding="utf-8")
                except Exception:
                    pass

            elapsed = _time.time() - t0
            _bbg_tag = " [green]BBG↑[/green]" if _bbg_updated else ""
            console.print(
                f"[dim]#{cycle:03d}[/dim] {refreshed_at} "
                f"[dim]{elapsed:.1f}s[/dim]{pnl_str}{_bbg_tag}"
            )

            # A cada 15 ciclos (~15min): rebuild completo de sinais + portfolio
            if cycle % 15 == 0 and desk2_path:
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
    no_open: bool = typer.Option(False, "--no-open", help="Não abre HTML no browser."),
) -> None:
    """Macro Desk — diagnóstico macro acionável + HTML interativo."""
    from app.cli.invest import main as desk_main
    desk_main(date=date, focus=focus, save=save, log_level="DEBUG" if verbose else "WARNING")

    # ── Regenera MacroDesk HTML com os fixes mais recentes ────────────────────
    try:
        from datetime import date as _date
        from app.storage.bundle_store import bundle_store
        from app.storage.paths import workspace
        from pathlib import Path as _P

        target = _date.fromisoformat(date) if date else _date.today()
        def _valid_bundle(p):
            """Apenas arquivos _summary.json ou ULID.json (bundle completo)."""
            stem = p.stem
            return stem.endswith("_summary") or (len(stem) == 26 and stem.isalnum())
        def _bundle_sort_key(p):
            # Prioriza ULID puro (tem market_prices) sobre _summary (sem preços)
            stem = p.stem
            is_full = len(stem) == 26 and stem.isalnum()
            return (str(p.parent), is_full, stem)
        all_bundles = sorted(
            [p for p in bundle_store.list_bundles() if _valid_bundle(p)],
            key=_bundle_sort_key,
            reverse=True,
        )
        bundles = [p for p in all_bundles if str(target) in str(p)]
        # Fallback para bundle mais recente se não há bundle do dia
        if not bundles:
            bundles = all_bundles
        if not bundles:
            return

        from app.models.daily_ingestion_bundle import DailyIngestionBundle
        import json as _json
        bundle = DailyIngestionBundle.model_validate_json(
            bundles[0].read_text(encoding="utf-8")
        )

        # Carrega curação se disponível
        curation_obj = None
        curation_path = bundle.artifact_paths.get("curation")
        if curation_path:
            try:
                from app.curation.models import CurationResult
                curation_obj = CurationResult.model_validate(
                    _json.loads(_P(curation_path).read_text(encoding="utf-8"))
                )
            except Exception:
                pass

        # ── Auto-import greeks ZIP de ~/Downloads se mais recente que último snapshot ──
        try:
            import glob as _glob
            from pathlib import Path as _P2
            from app.providers.options_store import options_store as _opts_auto
            _downloads = _P2.home() / "Downloads"
            _zips = sorted(_glob.glob(str(_downloads / "greeks_*.zip")),
                           key=lambda p: _P2(p).stat().st_mtime, reverse=True)
            if _zips:
                _latest_zip = _P2(_zips[0])
                _latest_snap = _opts_auto.load_latest()
                _snap_ts = _latest_snap.imported_at if _latest_snap else None
                _zip_mtime = _latest_zip.stat().st_mtime
                import datetime as _dt
                _snap_epoch = _dt.datetime.fromisoformat(_snap_ts).timestamp() if _snap_ts else 0
                if _zip_mtime > _snap_epoch:
                    with console.status(f"[cyan]Auto-import {_latest_zip.name}...[/cyan]"):
                        _opts_auto.import_from_zip(_latest_zip)
                    console.print(f"[green]Options auto-imported:[/green] {_latest_zip.name}")
        except Exception as _exc_ai:
            _log.debug("options_auto_import_skipped", error=str(_exc_ai))

        # ── Auto-extract bql_data ZIP de ~/Downloads se mais recente que CSVs locais ──
        _bql_extracted = False
        try:
            import glob as _glob2, zipfile as _zf, datetime as _dt2
            from pathlib import Path as _P3
            _bql_dir  = _P3(r"C:\Users\rafael bentes\bbg\agente\bql_data")
            _bql_dir.mkdir(parents=True, exist_ok=True)
            _dl2      = _P3.home() / "Downloads"
            _bql_zips = sorted(_glob2.glob(str(_dl2 / "bql_data_*.zip")),
                                key=lambda p: _P3(p).stat().st_mtime, reverse=True)
            if _bql_zips:
                _bql_zip = _P3(_bql_zips[0])
                _zip_mt  = _bql_zip.stat().st_mtime
                # Compara com o CSV mais recente na pasta bql_data
                _existing = list(_bql_dir.glob("*.csv"))
                _csv_mt   = max((f.stat().st_mtime for f in _existing), default=0)
                if _zip_mt > _csv_mt:
                    with console.status(f"[cyan]BQL data: extraindo {_bql_zip.name}...[/cyan]"):
                        with _zf.ZipFile(_bql_zip, "r") as _z:
                            _z.extractall(_bql_dir)
                    console.print(f"[green]BQL data extraido:[/green] {_bql_zip.name} -> bql_data/")
                    _bql_extracted = True
        except Exception as _exc_bql:
            _log.debug("bql_data_auto_extract_skipped", error=str(_exc_bql))

        # ── Auto-ingest: roda bloomberg_main_agent se CSV foi extraído ou banco desatualizado ──
        try:
            import sys as _sys
            from pathlib import Path as _P4
            _bbg_root = _P4(r"C:\Users\rafael bentes\bbg\agente")
            if str(_bbg_root) not in _sys.path:
                _sys.path.insert(0, str(_bbg_root))
            from app.query_layer import BloombergQueryLayer as _BQL
            _ql = _BQL()
            _status = _ql.get_last_ingestion_status()
            _age = _status.get("age_minutes") if _status else None
            _needs_ingest = _bql_extracted or _age is None or _age > 60
            if _needs_ingest:
                with console.status("[cyan]Bloomberg DB: ingerindo dados...[/cyan]"):
                    from core.bloomberg_main_agent import BloombergMainAgent
                    _agent = BloombergMainAgent()
                    _result = _agent.run()
                console.print(
                    f"[green]Bloomberg DB:[/green] {_result.rows_ingested} linhas "
                    f"({'OK' if _result.status == 'ok' else _result.status})"
                )
        except Exception as _exc_ing:
            _log.debug("bql_auto_ingest_skipped", error=str(_exc_ing))

        # ── Refresh market_prices do bundle a partir do BBG DB atual ──────────
        # O bundle salvo no JSON pode ter snapshot antigo (106 tickers); o BBG DB
        # foi atualizado depois (274). Re-popula bundle.market_prices direto do DB.
        try:
            from app.providers.market_prices import collect as _collect_mp
            _fresh_mp = _collect_mp()
            if _fresh_mp:
                bundle.market_prices = _fresh_mp
                console.print(f"[dim]market_prices refreshed: {len(_fresh_mp)} tickers do BBG DB[/dim]")
        except Exception as _exc_mp:
            console.print(f"[yellow]market_prices refresh falhou: {_exc_mp}[/yellow]")

        # Roda portfolio pipeline para a aba Alocação (antes do brief — passa swaggy + zones)
        portfolio = None
        rrg_result = None
        _pipeline_signals = {}
        _swaggy_result = None
        try:
            from app.pipeline.portfolio_pipeline import run_portfolio_pipeline
            with console.status("[cyan]Portfolio pipeline...[/cyan]"):
                portfolio, _pipeline_signals, _ = run_portfolio_pipeline(bundle)
            rrg_result = getattr(portfolio, "_rrg_result", None)
            _swaggy_result = getattr(portfolio, "_swaggy_result", None)
        except Exception as exc:
            console.print(f"[yellow]Portfolio pipeline falhou: {exc}[/yellow]")

        # Gera brief DEPOIS do pipeline para incluir swaggy + TradingView zones
        try:
            from app.views.week_ahead_brief import save_writer_brief
            with console.status("[cyan]Writer Brief...[/cyan]"):
                save_writer_brief(bundle, curation_path,
                                  swaggy_result=_swaggy_result,
                                  signals=_pipeline_signals or {})
        except Exception:
            pass

        # Gera MacroDesk HTML completo
        with console.status("[cyan]Gerando MacroDesk HTML...[/cyan]"):
            from app.views.macro_desk_v2 import save_macro_desk_v2
            from app.providers.options_store import options_store as _opts_s
            _opts_snap = _opts_s.load_latest()
            desk_path = save_macro_desk_v2(bundle, curation_obj, portfolio=portfolio, rrg_result=rrg_result, options_snapshot=_opts_snap)

        console.print(f"[green]MacroDesk HTML:[/green] {desk_path.name}")

        # ── Netlify deploy DESATIVADO (sem credito) ───────────────────────────
        # O HTML fica local em workspace/bundles/{date}/{ulid}_desk_v2.html

        if not no_open:
            import webbrowser
            webbrowser.open(desk_path.as_uri())

    except Exception as exc:
        console.print(f"[yellow]HTML geração falhou: {exc}[/yellow]")

    # ── Bloomberg ZIP Watcher — roda enquanto a janela estiver aberta ─────────
    # Detecta novos bql_data_*.zip em ~/Downloads e dispara a sequência completa:
    #   1. ingest do ZIP no BBG DB
    #   2. portfolio pipeline (sinais alpha + RRG + desk intel)
    #   3. writer (curação LLM + texto + brief HTML) — opcional, só se LLM ok
    #   4. MacroDesk HTML
    #   (Netlify deploy DESATIVADO — sem credito)
    def _regenerate_full_pipeline() -> None:
        """Re-roda TUDO com market_prices fresh: pipeline → writer → desk → netlify."""
        try:
            from app.providers.market_prices import collect as _mp_collect
            from app.pipeline.portfolio_pipeline import run_portfolio_pipeline as _rpp
            from app.views.macro_desk_v2 import save_macro_desk_v2 as _save_md
            from app.providers.options_store import options_store as _os_re

            # 1. Refresh market_prices direto do BBG DB
            _fresh = _mp_collect()
            if _fresh:
                bundle.market_prices = _fresh
                console.print(f"[dim]market_prices: {len(_fresh)} tickers fresh do BBG DB[/dim]")

            # 2. Portfolio pipeline (sinais alpha + RRG + desk intel)
            console.print("[cyan]Portfolio pipeline...[/cyan]")
            _portf, _sigs, _ = _rpp(bundle)
            _rrg = getattr(_portf, "_rrg_result", None)
            _swaggy = getattr(_portf, "_swaggy_result", None)

            # 3. Writer (curação LLM + texto + brief HTML)
            _new_curation_path = curation_path
            _new_curation_obj = curation_obj
            try:
                from app.curation.orchestrator import run_curation as _run_cur
                from app.utils.timestamps import new_ulid as _ulid
                from datetime import date as _date_w
                from pathlib import Path as _Pw
                from app.storage.paths import workspace as _ws

                _run_id_w = _ulid()
                _today_w = _date_w.today().isoformat()
                console.print("[cyan]Writer (curação + texto + brief)...[/cyan]")
                _cur_result = _run_cur(bundle, run_id=_run_id_w, run_date=_today_w)
                _cur_path = _ws.bundles / _today_w / f"{_run_id_w}_curation.json"
                _cur_path.write_text(_cur_result.model_dump_json(indent=2), encoding="utf-8")
                _new_curation_path = str(_cur_path)
                _new_curation_obj = _cur_result
                console.print(f"[green]Writer ok:[/green] {len(_cur_result.scored_items)} items")

                # Brief HTML (texto + tabelas)
                try:
                    from app.views.week_ahead_brief import save_writer_brief as _save_br
                    _save_br(bundle, _new_curation_path,
                             swaggy_result=_swaggy, signals=_sigs or {})
                except Exception as _exc_br:
                    console.print(f"[yellow]Brief falhou: {_exc_br}[/yellow]")
            except Exception as _exc_w:
                console.print(f"[yellow]Writer falhou (segue sem texto novo): {_exc_w}[/yellow]")

            # 4. MacroDesk HTML
            _opts = _os_re.load_latest()
            _new_path = _save_md(
                bundle, _new_curation_obj,
                portfolio=_portf, rrg_result=_rrg,
                options_snapshot=_opts,
            )
            console.print(f"[green]MacroDesk regenerado:[/green] {_new_path.name}")
            # Netlify deploy DESATIVADO (sem credito)
        except Exception as _exc_rd:
            console.print(f"[yellow]Pipeline regen falhou: {_exc_rd}[/yellow]")

    _watch_dl   = Path.home() / "Downloads"
    _watch_seen = {str(p) for p in _watch_dl.glob("bql_data_*.zip")}
    console.print()
    console.print("[bold cyan]Bloomberg Watcher ativo[/bold cyan] — ZIP novo dispara ingest + desk regen")
    console.print("[dim]Feche esta janela para parar.[/dim]")
    import time as _time
    while True:
        _time.sleep(20)
        try:
            _current = {str(p) for p in _watch_dl.glob("bql_data_*.zip")}
            _novos   = _current - _watch_seen
            if _novos:
                for _zp in sorted(_novos):
                    console.print(f"[green]Novo ZIP:[/green] {Path(_zp).name} — ingerindo...")
                    try:
                        from core.bloomberg_main_agent import BloombergMainAgent as _BBA
                        _r = _BBA().run()
                        console.print(
                            f"[green]Bloomberg DB:[/green] {_r.rows_ingested} linhas "
                            f"({'OK' if _r.status == 'ok' else _r.status})"
                        )
                        if _r.status == "ok" and _r.rows_ingested > 0:
                            console.print("[bold cyan]>> Pipeline completo: pipeline + writer + desk + netlify[/bold cyan]")
                            _regenerate_full_pipeline()
                    except Exception as _e:
                        console.print(f"[yellow]Bloomberg ingest erro: {_e}[/yellow]")
                _watch_seen = _current
        except KeyboardInterrupt:
            console.print("[dim]Watcher encerrado.[/dim]")
            break
        except Exception:
            pass


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

    def _valid_bundle_stem(stem: str) -> bool:
        return stem.endswith("_summary") or (len(stem) == 26 and stem.replace("_", "").isalnum())

    bundles = sorted(
        [p for p in bundle_store.list_bundles()
         if str(target_date) in str(p) and _valid_bundle_stem(p.stem)],
        key=lambda p: p.stat().st_size,  # maior bundle = mais conteúdo (X items, ZeroHedge, etc.)
        reverse=True,
    )
    if not bundles:
        console.print(f"[red]Nenhum bundle encontrado para {target_date}. Rode 'agente run ingest' primeiro.[/red]")
        raise typer.Exit(1)

    bundle_path = bundles[0]  # maior = mais conteúdo
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

    # Writer Brief já aberto acima — não reabrir html_report nem macro_desk separado


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
        console.print(f"\n[green]Flow Inspector gerado:[/green] {html_path.name}")
        console.print("[dim]Flow Inspector embutido no MacroDesk — use 'agente run live' para visualizar.[/dim]")


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


@app.command(name="options-import")
def options_import(
    zip_path: str = typer.Argument(
        None,
        help="Caminho para o ZIP ou diretorio (usa o ZIP mais recente). Padrao: ~/Downloads",
    ),
    no_open: bool = typer.Option(False, "--no-open", help="Nao regenera o MacroDesk apos importar."),
    date: str = typer.Option(None, "--date", "-d", help="Data YYYY-MM-DD do bundle a regenerar (padrao: hoje)."),
) -> None:
    """Importa snapshot do Greeks Dashboard (BQuant ZIP) para o workspace e regenera o MacroDesk."""
    import glob as _glob
    from pathlib import Path as _Path

    from app.providers.options_store import options_store

    # Resolve caminho: None ou diretorio -> ZIP greeks_*.zip mais recente
    resolved: str = zip_path or ""
    if not resolved or _Path(resolved).is_dir():
        search_dir = _Path(resolved) if resolved else _Path.home() / "Downloads"
        candidates = sorted(
            _glob.glob(str(search_dir / "greeks_*.zip")),
            key=lambda p: _Path(p).stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            console.print(f"[red]Nenhum greeks_*.zip encontrado em {search_dir}[/red]")
            raise typer.Exit(1)
        resolved = candidates[0]
        console.print(f"[dim]ZIP mais recente: {_Path(resolved).name}[/dim]")

    console.print(Panel.fit(
        f"[bold cyan]Options Import[/bold cyan]\n[dim]{resolved}[/dim]",
        border_style="cyan",
    ))

    with console.status("[cyan]Importando ZIP...[/cyan]"):
        try:
            snap = options_store.import_from_zip(resolved)
        except Exception as exc:
            console.print(f"[red]Erro ao importar ZIP: {exc}[/red]")
            raise typer.Exit(1)

    console.print(f"[green]OK[/green] Importado: [bold]{snap.ticker}[/bold] spot={snap.spot:,.0f}  ts={snap.ts}")
    console.print(f"   GEX: {snap.gex_net_bn:+.1f}B  |  IV 30D: {snap.iv_30d*100:.2f}%  |  Squeeze: {snap.squeeze_score:.0f}/100")
    console.print(f"   Gamma Flip: {snap.gamma_flip:,.0f}  |  Call Wall: {snap.call_wall:,.0f}  |  Put Wall: {snap.put_wall:,.0f}")
    if snap.has_jarvis_html:
        console.print("   [dim]jarvis.html incluído[/dim]")

    if no_open:
        return

    # Regenera MacroDesk com nova aba Opções
    from datetime import date as _date
    from pathlib import Path as _P
    import json as _json

    target = _date.fromisoformat(date) if date else _date.today()
    try:
        from app.storage.bundle_store import bundle_store

        def _valid_bundle(p):
            stem = p.stem
            return stem.endswith("_summary") or (len(stem) == 26 and stem.isalnum())

        def _bundle_sort_key_oi(p):
            stem = p.stem
            is_full = len(stem) == 26 and stem.isalnum()
            return (str(p.parent), is_full, stem)

        all_bundles = sorted(
            [p for p in bundle_store.list_bundles() if _valid_bundle(p)],
            key=_bundle_sort_key_oi,
            reverse=True,
        )
        bundles = [p for p in all_bundles if str(target) in str(p)]
        if not bundles:
            bundles = all_bundles  # fallback ao mais recente disponível
        if not bundles:
            console.print("[yellow]Sem bundle disponivel — MacroDesk não regenerado.[/yellow]")
            return

        from app.models.daily_ingestion_bundle import DailyIngestionBundle
        bundle = DailyIngestionBundle.model_validate_json(
            bundles[0].read_text(encoding="utf-8")
        )

        curation_obj = None
        curation_path = bundle.artifact_paths.get("curation")
        if curation_path:
            try:
                from app.curation.models import CurationResult
                curation_obj = CurationResult.model_validate(
                    _json.loads(_P(curation_path).read_text(encoding="utf-8"))
                )
            except Exception:
                pass

        # Roda portfolio pipeline para incluir abas Alocação + Desk Radar
        portfolio = None
        rrg_result = None
        try:
            from app.pipeline.portfolio_pipeline import run_portfolio_pipeline
            with console.status("[cyan]Portfolio pipeline...[/cyan]"):
                portfolio, _, _ = run_portfolio_pipeline(bundle)
            rrg_result = getattr(portfolio, "_rrg_result", None)
        except Exception as exc2:
            console.print(f"[yellow]Portfolio pipeline pulado: {exc2}[/yellow]")

        with console.status("[cyan]Regenerando MacroDesk...[/cyan]"):
            from app.views.macro_desk_v2 import save_macro_desk_v2
            from app.providers.options_store import options_store as _opts_s
            _opts_snap = _opts_s.load_latest()
            desk_path = save_macro_desk_v2(
                bundle, curation_obj,
                portfolio=portfolio, rrg_result=rrg_result,
                options_snapshot=_opts_snap,
            )

        console.print(f"[green]MacroDesk atualizado:[/green] {desk_path.name}")
        import webbrowser as _wb
        _wb.open(desk_path.as_uri())

    except Exception as exc:
        console.print(f"[yellow]MacroDesk regeneration falhou: {exc}[/yellow]")


@app.command(name="all")
def run_all(
    date: str = typer.Option(None, "--date", "-d", help="Data YYYY-MM-DD (padrão: hoje)"),
    no_open: bool = typer.Option(False, "--no-open", help="Não abre HTML no browser."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Tudo em um: ingest -> portfolio pipeline -> writer -> MacroDesk HTML."""
    from datetime import date as _date
    configure_logging("DEBUG" if verbose else "INFO")

    target_date = _date.fromisoformat(date) if date else _date.today()

    console.print(Panel.fit(
        f"[bold cyan]Agente All-in-One[/bold cyan] — {target_date} "
        f"| ingest -> pipeline -> writer -> desk"
    ))

    # ── 1. Ingest (fontes: X, ZeroHedge, SpotGamma...) ────────────────────────
    console.print("\n[bold]1/4[/bold] Coletando fontes...")
    with console.status("[cyan]Ingest...[/cyan]"):
        bundle = run_ingestion(headless=True)
    console.print(f"[green]Ingest ok:[/green] {len(bundle.x_items)} X · "
                  f"{len(bundle.market_ear_blocks)} ZH/ME · "
                  f"{len(bundle.spotgamma_reports)} SpotGamma")

    # ── 2. Portfolio pipeline (mercado, charts, sinais alpha, desk intel) ──────
    console.print("\n[bold]2/4[/bold] Portfolio pipeline...")
    portfolio, signals, html_path = _run_portfolio_after_ingest(bundle)

    # ── 3. Writer (curação LLM + texto + brief HTML) ──────────────────────────
    console.print("\n[bold]3/4[/bold] Curação + Writer...")
    curation_path = None
    try:
        from app.curation.orchestrator import run_curation
        from app.utils.timestamps import new_ulid
        from pathlib import Path as _P
        from app.storage.paths import workspace

        run_id = new_ulid()
        with console.status("[cyan]Writer: narrativa + texto...[/cyan]"):
            curation_result = run_curation(bundle, run_id=run_id, run_date=str(target_date))

        import json as _json
        curation_path = str(
            workspace.bundles / str(target_date) / f"{run_id}_curation.json"
        )
        _P(curation_path).write_text(
            curation_result.model_dump_json(indent=2), encoding="utf-8"
        )
        console.print(f"[green]Writer ok:[/green] "
                      f"{len(curation_result.scored_items)} items · "
                      f"modo={getattr(curation_result, 'written_mode', '?')}")
    except Exception as exc:
        console.print(f"[yellow]Writer falhou: {exc}[/yellow]")

    # ── 4. MacroDesk HTML ──────────────────────────────────────────────────────
    console.print("\n[bold]4/4[/bold] MacroDesk HTML...")
    desk_path = None
    try:
        from app.views.macro_desk_v2 import save_macro_desk_v2
        from app.providers.options_store import options_store as _opts_s
        import json as _json2
        from pathlib import Path as _P2

        curation_obj = None
        if curation_path and _P2(curation_path).exists():
            try:
                from app.curation.models import CurationResult
                curation_obj = CurationResult.model_validate(
                    _json2.loads(_P2(curation_path).read_text(encoding="utf-8"))
                )
            except Exception:
                pass

        opts_snap = _opts_s.load_latest()
        with console.status("[cyan]MacroDesk HTML...[/cyan]"):
            desk_path = save_macro_desk_v2(
                bundle, curation_obj,
                portfolio=portfolio,
                options_snapshot=opts_snap,
            )
        console.print(f"[green]MacroDesk ok:[/green] {desk_path.name}")
    except Exception as exc:
        console.print(f"[yellow]MacroDesk falhou: {exc}[/yellow]")

    console.print(Panel.fit(
        "[bold green]All done![/bold green]",
        subtitle=str(target_date),
    ))

    if not no_open and desk_path:
        import webbrowser as _wb
        _wb.open(desk_path.as_uri())


if __name__ == "__main__":
    app()
