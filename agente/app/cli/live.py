"""
CLI: agente live

Loop de atualização de preços em tempo real (near real-time).
  1. Carrega o bundle mais recente do dia
  2. Atualiza preços via IBKR snapshot (ou yfinance fast_info)
  3. Rebuilda o grafo e salva o HTML
  4. Dorme {interval}s e repete

O HTML gerado tem meta-refresh automático no browser.

Uso:
    python -m app.cli.live
    python -m app.cli.live --interval 30
    python -m app.cli.live --date 2026-03-31
"""

from __future__ import annotations

import os
import time
import webbrowser
from datetime import date as _date
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from app.audit.logger import configure_logging, get_logger

app  = typer.Typer(name="live", help="Loop de preços near real-time + MacroDesk v2 auto-refresh.")
console = Console()
_log = get_logger("cli.live")


def _load_portfolio_from_tracker():
    """Reconstrói PortfolioResult do active_portfolio.json sem rodar pipeline."""
    import json as _json
    from dataclasses import dataclass, field
    from app.analysis.portfolio_optimizer import PortfolioResult, PositionResult
    from app.storage.paths import workspace

    path = Path(workspace.workspace) / "portfolio" / "active_portfolio.json"
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        data = _json.load(f)

    open_trades = [t for t in data.get("trades", []) if t.get("status") == "open"]
    positions = []
    for t in open_trades:
        shares = t.get("shares", 0) or 0
        entry = t.get("entry_price", 0) or 0
        alloc_usd = t.get("target_usd") or t.get("allocation_usd") or (shares * entry)
        budget = data.get("budget", 100_000)
        positions.append(PositionResult(
            ticker=t["ticker"],
            name=t.get("name", t["ticker"]),
            direction=t.get("direction", "long"),
            conviction=t.get("conviction", "medium"),
            allocation_pct=alloc_usd / budget if budget else 0,
            allocation_usd=alloc_usd,
            shares_approx=shares,
            expected_return_ann=t.get("expected_return_ann", 0.10),
            risk_score=0.5,
            sharpe_implied=2.0,
            composite=0.5,
            asset_class=t.get("asset_class", "equity"),
            cluster_id=t.get("cluster_id", ""),
            rationale=t.get("rationale", []),
            entry_price=entry,
            stop_loss=t.get("stop_loss", 0),
            take_profit=t.get("take_profit", 0),
            stop_pct=t.get("stop_pct", 0),
            target_pct=t.get("target_pct", 0),
            risk_reward=t.get("risk_reward", 2.0),
        ))

    return PortfolioResult(
        budget=data.get("budget", 100_000),
        regime_mode=data.get("regime_at_entry", "neutral"),
        positions=positions,
        expected_return_ann=data.get("expected_return", 0),
        sharpe=data.get("sharpe_at_entry", 0),
    )


def _load_bundle(date_str: str | None):
    """Carrega bundle mais recente do dia."""
    from app.storage.bundle_store import bundle_store
    from app.models.daily_ingestion_bundle import DailyIngestionBundle

    target = _date.fromisoformat(date_str) if date_str else _date.today()
    bundles = [
        p for p in bundle_store.list_bundles()
        if str(target) in str(p) and "_" not in p.stem
    ]
    if not bundles:
        return None, None
    path = bundles[0]
    bundle = DailyIngestionBundle.model_validate_json(path.read_text(encoding="utf-8"))
    return bundle, path


def _build_and_save(
    bundle, out_path: Path, live_mode: bool = True,
    cached_network: "dict | None" = None,
    anatomy_map: "dict | None" = None,
    options_map: "dict | None" = None,
    prob_map: "dict | None" = None,
    flow_pred=None,
    portfolio=None,
) -> int:
    """Rebuilda grafo e salva HTML. Retorna tamanho em KB."""
    from app.desk.graph_engine import build_from_bundle
    from app.views.macro_desk_v2 import generate_macro_desk_v2_html

    graph = build_from_bundle(
        bundle,
        skip_anatomy=bool(anatomy_map),
        skip_options=bool(options_map),
        skip_prob=bool(prob_map),
        skip_flow=True,  # sempre usa cached_flow; nunca re-coleta no loop
        skip_network=bool(cached_network), cached_network=cached_network,
        cached_anatomy=anatomy_map or None,
        cached_options=options_map or None,
        cached_prob=prob_map or None,
        cached_flow=flow_pred,
    )
    html = generate_macro_desk_v2_html(
        bundle, graph, live_mode=live_mode,
        portfolio=portfolio, flow_pred=flow_pred,
    )
    out_path.write_text(html, encoding="utf-8")
    return len(html) // 1024


@app.command()
def run(
    date_str:  str  = typer.Option(None,  "--date",     "-d", help="Data YYYY-MM-DD (padrão: hoje)"),
    interval:  int  = typer.Option(60,    "--interval", "-i", help="Segundos entre updates (padrão: 60)"),
    no_open:   bool = typer.Option(False, "--no-open",        help="Não abre o browser."),
    verbose:   bool = typer.Option(False, "--verbose",  "-v"),
) -> None:
    """Loop de preços near real-time + MacroDesk v2 auto-refresh."""
    configure_logging("DEBUG" if verbose else "WARNING")

    console.print("[bold cyan]MacroDesk Live[/bold cyan] — iniciando")

    bundle, bundle_path = _load_bundle(date_str)
    if bundle is None:
        console.print("[red]Nenhum bundle encontrado. Rode 'agente run ingest' primeiro.[/red]")
        raise typer.Exit(1)

    out_dir  = bundle_path.parent
    # Nome fixo — macroDesk.html — fácil de abrir pelo bat sem depender do run_id
    out_path = out_dir / "macroDesk.html"

    console.print(f"[dim]Bundle: {bundle_path.name}[/dim]")
    console.print(f"[dim]Desk v2: {out_path.name}[/dim]")
    console.print(f"[dim]Interval: {interval}s[/dim]")

    # Build inicial — network pesada (yfinance 200+ tickers) feita uma vez aqui
    console.print("[cyan]Build inicial — network + precos...[/cyan]")
    _init_network: dict = {}
    try:
        from app.analysis.network import analyze as _net_analyze
        _init_network = _net_analyze(
            {k: v for k, v in (bundle.market_prices or {}).items() if not k.startswith("__")},
            universe="spx",  # ~50 tickers do SPX core — rápido e legível
        ) or {}
        console.print(f"[dim]Network: {len(_init_network.get('mst', {}).get('edges', []))} MST edges[/dim]")
    except Exception as exc:
        console.print(f"[yellow]Network análise falhou: {exc}[/yellow]")

    # BQL via CSV (exportado pelo scripts/bql_export.py no BQuant)
    _anatomy_map: dict = {}
    _options_map: dict = {}
    _cached_flow = None
    _bql_mtime: float = 0.0  # controla quando o CSV mudou

    def _load_bql_csvs() -> float:
        """Lê os CSVs BQL e retorna o mtime do meta.csv."""
        nonlocal _anatomy_map, _options_map, _cached_flow
        try:
            from app.providers.bql_csv import load_all, BQL_DATA_DIR
            from pathlib import Path
            _meta_dated = sorted(Path(BQL_DATA_DIR).glob("meta_*.csv"), reverse=True)
            meta = _meta_dated[0] if _meta_dated else Path(BQL_DATA_DIR) / "meta.csv"
            mtime = meta.stat().st_mtime if meta.exists() else 0.0
            data = load_all()
            _anatomy_map = data.get("fundamentals", {})
            _options_map = data.get("options_iv", {})
            # Reconstrói flow_pred a partir do GEX CSV
            gex_s = data.get("gex_summary", {})
            letf  = data.get("letf_flows", [])
            if gex_s or letf:
                from app.providers.gex_letf import FlowPrediction, GEXResult, GEXSummary
                spx_gex = GEXSummary(
                    gex_bn=gex_s.get("gex_bn", 0),
                    gamma_regime=gex_s.get("gamma_regime", "flat"),
                    flip_level=None,
                    n_strikes=gex_s.get("n_options", 0),
                )
                mag = abs(gex_s.get("gex_bn", 0))
                direction = gex_s.get("direction", "flat")
                conviction = "high" if mag > 2 else ("medium" if mag > 0.5 else "low")
                _cached_flow = FlowPrediction(
                    direction=direction,
                    magnitude_bn=gex_s.get("gex_bn", 0),
                    conviction=conviction,
                    gex=GEXResult(spx=spx_gex),
                    per_etf=letf,
                    summary=f"GEX {direction} ${gex_s.get('gex_bn',0):+.2f}B | regime={gex_s.get('gamma_regime','flat')}",
                )
            return mtime
        except Exception as exc:
            console.print(f"[yellow]BQL CSV: {exc}[/yellow]")
            return 0.0

    console.print("[cyan]BQL: carregando CSVs do BQuant...[/cyan]")
    _bql_mtime = _load_bql_csvs()
    console.print(f"[dim]BQL CSV: {len(_anatomy_map)} fundamentais, {len(_options_map)} options, flow={'ok' if _cached_flow else 'sem dados'}[/dim]")
    if not _anatomy_map:
        console.print("[yellow dim]  → Rode no BQuant: python scripts/bql_export.py --loop[/yellow dim]")

    # Prob: VaR, CVaR, tail score
    _prob_map: dict = {}
    try:
        console.print("[cyan]Prob: VaR/CVaR/tail via IBKR histórico...[/cyan]")
        from app.analysis.probabilistic import analyze_from_registry
        _prob_map = analyze_from_registry() or {}
        console.print(f"[dim]Prob: {len(_prob_map)} tickers[/dim]")
    except Exception as exc:
        console.print(f"[yellow]Prob falhou: {exc}[/yellow]")

    # Portfolio deve ser carregado ANTES do primeiro build
    _cached_portfolio = None
    try:
        _cached_portfolio = _load_portfolio_from_tracker()
        if _cached_portfolio:
            console.print(f"[dim]Portfolio: {len(_cached_portfolio.positions)} posições carregadas do tracker[/dim]")
    except Exception as exc:
        console.print(f"[yellow]Portfolio load falhou: {exc}[/yellow]")

    kb = _build_and_save(bundle, out_path, live_mode=True, cached_network=_init_network, anatomy_map=_anatomy_map, options_map=_options_map, prob_map=_prob_map, flow_pred=_cached_flow, portfolio=_cached_portfolio)
    console.print(f"[green]HTML gerado: {out_path.name} ({kb}KB)[/green]")

    console.print(f"[bold green]Desk v2:[/bold green] {out_path}")

    if not no_open:
        import subprocess
        try:
            subprocess.Popen(["cmd", "/c", "start", "", str(out_path)])
        except Exception:
            try:
                os.startfile(str(out_path))
            except Exception:
                webbrowser.open(out_path.as_uri())

    from app.providers.market_prices_live import refresh as refresh_prices

    # Cache graph_data inicial (reutiliza network já computada)
    _cached_graph: dict = {}
    try:
        from app.desk.graph_engine import build_from_bundle as _bgraph
        _cached_graph = _bgraph(
            bundle,
            skip_anatomy=bool(_anatomy_map), skip_options=bool(_options_map), skip_prob=bool(_prob_map),
            skip_flow=True, cached_flow=_cached_flow,
            skip_network=True, cached_network=_init_network,
            cached_anatomy=_anatomy_map or None,
            cached_options=_options_map or None,
            cached_prob=_prob_map or None,
        ) or {}
    except Exception as exc:
        console.print(f"[yellow]Graph cache inicial falhou: {exc}[/yellow]")

    cycle = 0
    console.print(f"\n[bold]Loop live ativo[/bold] — Ctrl+C para parar\n")

    try:
        while True:
            cycle += 1
            t0 = time.time()

            # Atualiza só preços (rápido: IBKR snapshot ou yfinance fast_info)
            refresh_prices(bundle.market_prices)
            refreshed_at = bundle.market_prices.get("__refreshed_at__", "?")

            # Atualiza P&L snapshot antes de gerar o HTML
            pnl_str = ""
            try:
                from app.analysis.portfolio_tracker import update_snapshot
                snap = update_snapshot(bundle.market_prices)
                if snap:
                    pnl_color = "green" if snap["total_pnl"] >= 0 else "red"
                    pnl_str = f" P&L=[{pnl_color}]${snap['total_pnl']:+,.0f}({snap['pnl_pct']:+.2%})[/{pnl_color}]"
            except Exception:
                pass

            # Sincroniza CSVs do GitHub Gist (publicado pelo BQuant cloud)
            try:
                from app.providers.bql_gist import sync_from_gist
                if sync_from_gist():
                    _bql_mtime = _load_bql_csvs()
                    console.print(f"[green]Gist sincronizado — {len(_anatomy_map)} fundamentais[/green]")
            except Exception:
                pass

            # Recarrega CSVs BQL se o arquivo foi atualizado pelo BQuant
            try:
                from app.providers.bql_csv import BQL_DATA_DIR
                from pathlib import Path as _Path
                # Suporta tanto meta_YYYY-MM-DD.csv (com data) quanto meta.csv (legado)
                _meta_dated = sorted(_Path(BQL_DATA_DIR).glob("meta_*.csv"), reverse=True)
                _meta = _meta_dated[0] if _meta_dated else _Path(BQL_DATA_DIR) / "meta.csv"
                if _meta.exists():
                    _new_mtime = _meta.stat().st_mtime
                    if _new_mtime > _bql_mtime:
                        _bql_mtime = _load_bql_csvs()
                        console.print(f"[green]BQL CSV atualizado — {len(_anatomy_map)} fundamentais, {len(_options_map)} options[/green]")
            except Exception:
                pass

            # Regenera HTML unificado com graph_data cacheado + precos atualizados + portfolio
            from app.views.macro_desk_v2 import generate_macro_desk_v2_html
            html = generate_macro_desk_v2_html(
                bundle, _cached_graph or None,
                live_mode=True,
                portfolio=_cached_portfolio,
                flow_pred=_cached_flow,
            )
            out_path.write_text(html, encoding="utf-8")

            elapsed = time.time() - t0
            src_keys = [k for k in bundle.market_prices.keys() if not k.startswith("__")]
            src = bundle.market_prices.get(src_keys[0], {}).get("source", "?") if src_keys else "?"

            console.print(
                f"[dim]#{cycle:03d}[/dim] [{refreshed_at}] "
                f"src=[cyan]{src}[/cyan] "
                f"[dim]{elapsed:.1f}s[/dim]"
                f"{pnl_str}"
            )

            # A cada 15 ciclos (~15min), refaz o graph_data (sem network re-download)
            if cycle % 15 == 0:
                console.print("[yellow]Ciclo 15 — rebuild graph...[/yellow]")
                try:
                    from app.desk.graph_engine import build_from_bundle as _bgraph
                    _cached_graph = _bgraph(
                        bundle,
                        skip_anatomy=bool(_anatomy_map), skip_options=bool(_options_map), skip_prob=bool(_prob_map),
                        skip_flow=True, cached_flow=_cached_flow,
                        skip_network=True, cached_network=_init_network,
                        cached_anatomy=_anatomy_map or None,
                        cached_options=_options_map or None,
                        cached_prob=_prob_map or None,
                    ) or {}
                    kb = _build_and_save(bundle, out_path, live_mode=True, cached_network=_init_network, anatomy_map=_anatomy_map, options_map=_options_map, prob_map=_prob_map, flow_pred=_cached_flow, portfolio=_cached_portfolio)
                    console.print(f"[green]Rebuild: {kb}KB[/green]")
                except Exception as exc:
                    console.print(f"[yellow]Rebuild falhou: {exc}[/yellow]")

            time.sleep(max(1, interval - elapsed))

    except KeyboardInterrupt:
        console.print("\n[yellow]Live loop encerrado.[/yellow]")


if __name__ == "__main__":
    app()
