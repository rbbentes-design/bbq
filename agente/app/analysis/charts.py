"""
Analysis: Chart Generation

Gera charts HTML interativos via pyecharts:
  - K-line (candlestick) com volume
  - Correlation heatmap
  - Radar chart de métricas de risco
  - Fan chart para Monte Carlo

Requer: pip install pyecharts yfinance

Uso:
    from app.analysis.charts import generate_all_charts
    paths = generate_all_charts(market_prices, risk, monte_carlo, output_dir)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("analysis.charts")


def kline_chart(
    sym: str,
    name: str,
    output_path: Path,
    period: str = "60d",
) -> Path | None:
    """
    Gera K-line (candlestick) com volume em HTML.
    """
    try:
        import yfinance as yf
        from pyecharts import options as opts
        from pyecharts.charts import Bar, Kline, Grid
        from pyecharts.commons.utils import JsCode

        hist = yf.Ticker(sym).history(period=period, auto_adjust=True)
        if hist.empty:
            return None

        dates = [str(d.date()) for d in hist.index]
        ohlc = [[
            round(float(row["Open"]), 2),
            round(float(row["Close"]), 2),
            round(float(row["Low"]), 2),
            round(float(row["High"]), 2),
        ] for _, row in hist.iterrows()]
        volumes = [round(float(row["Volume"]), 0) for _, row in hist.iterrows()]
        closes = [round(float(row["Close"]), 2) for _, row in hist.iterrows()]

        kline = (
            Kline()
            .add_xaxis(dates)
            .add_yaxis(
                series_name=name,
                y_axis=ohlc,
                itemstyle_opts=opts.ItemStyleOpts(
                    color="#ef232a", color0="#14b143",
                    border_color="#ef232a", border_color0="#14b143",
                ),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"{name} — {period}"),
                xaxis_opts=opts.AxisOpts(is_scale=True),
                yaxis_opts=opts.AxisOpts(is_scale=True, splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                )),
                datazoom_opts=[
                    opts.DataZoomOpts(is_show=False, type_="inside", range_start=70, range_end=100),
                    opts.DataZoomOpts(pos_bottom="-2%", range_start=70, range_end=100),
                ],
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            )
        )

        bar = (
            Bar()
            .add_xaxis(dates)
            .add_yaxis(
                series_name="Volume",
                y_axis=volumes,
                itemstyle_opts=opts.ItemStyleOpts(
                    color=JsCode(
                        "function(params) { var c = ['#ef232a', '#14b143'];"
                        "return c[params.dataIndex % 2]; }"
                    )
                ),
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show=False)),
            )
        )

        grid = (
            Grid(init_opts=opts.InitOpts(width="900px", height="500px"))
            .add(kline, grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", height="55%"))
            .add(bar,   grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", pos_top="70%", height="16%"))
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid.render(str(output_path))
        _log.info("kline_chart_saved", sym=sym, path=str(output_path))
        return output_path

    except Exception as exc:
        _log.warning("kline_chart_error", sym=sym, error=str(exc))
        return None


def correlation_heatmap(
    correlations: dict[str, dict[str, float]],
    market_prices: dict[str, Any],
    output_path: Path,
) -> Path | None:
    """Gera heatmap de correlação em HTML."""
    try:
        from pyecharts import options as opts
        from pyecharts.charts import HeatMap

        if not correlations:
            return None

        symbols = list(correlations.keys())
        names = [market_prices.get(s, {}).get("name", s) for s in symbols]

        data = []
        for i, sym_i in enumerate(symbols):
            for j, sym_j in enumerate(symbols):
                val = correlations.get(sym_i, {}).get(sym_j, 0)
                data.append([j, i, round(val, 3)])

        heatmap = (
            HeatMap(init_opts=opts.InitOpts(width="700px", height="600px"))
            .add_xaxis(names)
            .add_yaxis(
                series_name="Correlação",
                yaxis_data=names,
                value=data,
                label_opts=opts.LabelOpts(is_show=True, position="inside"),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="Matriz de Correlação (60d)"),
                visualmap_opts=opts.VisualMapOpts(
                    min_=-1, max_=1,
                    is_calculable=True,
                    orient="horizontal",
                    pos_left="center",
                    pos_bottom="5%",
                    range_color=["#313695", "#f7f7f7", "#d73027"],
                ),
                tooltip_opts=opts.TooltipOpts(formatter="{b}: {c}"),
            )
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        heatmap.render(str(output_path))
        _log.info("heatmap_saved", path=str(output_path))
        return output_path

    except Exception as exc:
        _log.warning("heatmap_error", error=str(exc))
        return None


def risk_radar_chart(
    risk_metrics: dict[str, Any],
    market_prices: dict[str, Any],
    output_path: Path,
) -> Path | None:
    """Gera radar chart com métricas de risco normalizadas."""
    try:
        from pyecharts import options as opts
        from pyecharts.charts import Radar

        tickers_data = risk_metrics.get("tickers", {})
        if not tickers_data:
            return None

        # Normaliza métricas para escala 0-100 (invertido para VaR/drawdown)
        schema = [
            opts.RadarIndicatorItem(name="VaR 95%", max_=5),
            opts.RadarIndicatorItem(name="CVaR 95%", max_=5),
            opts.RadarIndicatorItem(name="Max Drawdown", max_=50),
            opts.RadarIndicatorItem(name="Sharpe Ratio", max_=3),
        ]

        series_data = []
        for sym, m in list(tickers_data.items())[:6]:  # max 6 tickers no radar
            name = market_prices.get(sym, {}).get("name", sym)
            series_data.append(
                opts.RadarItem(
                    name=name,
                    value=[
                        abs(m.get("var_95", 0)) * 100,
                        abs(m.get("cvar_95", 0)) * 100,
                        abs(m.get("max_drawdown", 0)) * 100,
                        max(0, m.get("sharpe", 0)),
                    ],
                )
            )

        radar = (
            Radar(init_opts=opts.InitOpts(width="600px", height="500px"))
            .add_schema(schema=schema)
            .set_global_opts(title_opts=opts.TitleOpts(title="Risk Radar (60d)"))
        )
        for item in series_data:
            radar.add("", [item])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        radar.render(str(output_path))
        _log.info("radar_chart_saved", path=str(output_path))
        return output_path

    except Exception as exc:
        _log.warning("radar_chart_error", error=str(exc))
        return None


def monte_carlo_fan_chart(
    sym: str,
    name: str,
    mc_result: dict[str, Any],
    output_path: Path,
) -> Path | None:
    """Gera fan chart (percentis Monte Carlo) em HTML."""
    try:
        from pyecharts import options as opts
        from pyecharts.charts import Line

        percs = mc_result.get("percentiles", {})
        days = mc_result.get("horizon_days", 20)
        if not percs:
            return None

        x_axis = [f"D+{i+1}" for i in range(days)]

        line = Line(init_opts=opts.InitOpts(width="800px", height="400px"))
        line.add_xaxis(x_axis)

        styles = {
            "p5":  ("#c0392b", 1, "dashed"),
            "p25": ("#e67e22", 1, "dashed"),
            "p50": ("#2980b9", 2, "solid"),
            "p75": ("#27ae60", 1, "dashed"),
            "p95": ("#1abc9c", 1, "dashed"),
        }
        for key, (color, width, style) in styles.items():
            if key in percs:
                line.add_yaxis(
                    series_name=key.upper(),
                    y_axis=percs[key],
                    linestyle_opts=opts.LineStyleOpts(width=width, type_=style, color=color),
                    label_opts=opts.LabelOpts(is_show=False),
                    is_symbol_show=False,
                )

        line.set_global_opts(
            title_opts=opts.TitleOpts(title=f"Monte Carlo — {name} ({days}d)"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_bottom="0%"),
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        line.render(str(output_path))
        _log.info("fan_chart_saved", sym=sym, path=str(output_path))
        return output_path

    except Exception as exc:
        _log.warning("fan_chart_error", sym=sym, error=str(exc))
        return None


def mst_network_chart(
    mst_data: dict,
    rmt_data: dict,
    market_prices: dict,
    output_path: Path,
) -> Path | None:
    """
    Gera grafo de rede force-directed do MST de Mantegna.
    Nós = ativos, arestas = correlações reais (grafo esparso RMT).
    Tamanho do nó = grau no MST (hub = maior).
    Cor = categoria do ativo.
    """
    try:
        from pyecharts import options as opts
        from pyecharts.charts import Graph

        tickers   = rmt_data.get("tickers", [])
        edges_mst = mst_data.get("edges", [])
        edges_gl  = rmt_data.get("edges", [])   # GraphicalLasso edges
        hubs      = {t: d for t, d in mst_data.get("hubs", [])}

        if not tickers or not edges_mst:
            return None

        # ── Categorias por tipo de ativo ──────────────────────────────────────
        _CATEGORIES = {
            "^GSPC": ("Equity US", "#60a5fa"),
            "^NDX":  ("Equity US", "#60a5fa"),
            "^RUT":  ("Equity US", "#60a5fa"),
            "^VIX":  ("Volatility", "#f87171"),
            "TLT":   ("Bonds",     "#34d399"),
            "HYG":   ("Bonds",     "#6ee7b7"),
            "GLD":   ("Commodity", "#fbbf24"),
            "CL=F":  ("Commodity", "#f97316"),
            "BTC-USD": ("Crypto",  "#a78bfa"),
            "DX-Y.NYB": ("FX",    "#e879f9"),
        }
        cat_names = list({v[0] for v in _CATEGORIES.values()})
        cat_map   = {c: i for i, c in enumerate(cat_names)}

        nodes = []
        for t in tickers:
            name   = market_prices.get(t, {}).get("name", t)
            degree = hubs.get(t, 0)
            cat, color = _CATEGORIES.get(t, ("Other", "#8892a4"))
            ret1d = market_prices.get(t, {}).get("return_1d", 0) or 0
            nodes.append(opts.GraphNode(
                name=name,
                symbol_size=12 + degree * 8,
                value=round(ret1d, 2),
                category=cat_map[cat],
                label_opts=opts.LabelOpts(is_show=True, position="right", font_size=11),
                itemstyle_opts=opts.ItemStyleOpts(color=color),
            ))

        # ── Arestas MST ────────────────────────────────────────────────────────
        links = []
        name_map = {t: market_prices.get(t, {}).get("name", t) for t in tickers}
        for e in edges_mst:
            rho = e["correlation"]
            links.append(opts.GraphLink(
                source=name_map.get(e["from"], e["from"]),
                target=name_map.get(e["to"], e["to"]),
                value=abs(rho),
                linestyle_opts=opts.LineStyleOpts(
                    width=max(1, abs(rho) * 3),
                    color="#34d399" if rho > 0 else "#f87171",
                    opacity=0.7,
                ),
                label_opts=opts.LabelOpts(
                    is_show=True,
                    formatter=f"{rho:+.2f}",
                    font_size=9,
                    color="#8892a4",
                ),
            ))

        # ── Arestas GraphicalLasso (mais finas, tracejadas) ───────────────────
        for src, tgt, w in edges_gl[:20]:
            links.append(opts.GraphLink(
                source=name_map.get(src, src),
                target=name_map.get(tgt, tgt),
                value=abs(w),
                linestyle_opts=opts.LineStyleOpts(
                    width=1, type_="dashed", color="#4a5568", opacity=0.4,
                ),
            ))

        categories = [opts.GraphCategory(name=c) for c in cat_names]

        graph = (
            Graph(init_opts=opts.InitOpts(
                width="100%", height="520px",
                bg_color="#0f1117",
            ))
            .add(
                series_name="",
                nodes=nodes,
                links=links,
                categories=categories,
                layout="force",
                is_roam=True,
                is_focusnode=True,
                gravity=0.3,
                repulsion=300,
                edge_length=[60, 180],
                linestyle_opts=opts.LineStyleOpts(curve=0.2),
                label_opts=opts.LabelOpts(is_show=True, font_size=11),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="Rede de Ativos — MST de Mantegna",
                    subtitle="nós = ativos · arestas = correlação real · tamanho = grau no MST",
                    title_textstyle_opts=opts.TextStyleOpts(color="#e2e8f0", font_size=13),
                    subtitle_textstyle_opts=opts.TextStyleOpts(color="#8892a4", font_size=10),
                ),
                legend_opts=opts.LegendOpts(
                    is_show=True,
                    pos_right="5%",
                    orient="vertical",
                    textstyle_opts=opts.TextStyleOpts(color="#e2e8f0"),
                ),
                tooltip_opts=opts.TooltipOpts(
                    formatter="Retorno 1d: {c}%",
                ),
            )
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        graph.render(str(output_path))
        _log.info("mst_network_chart_saved", edges=len(links), path=str(output_path))
        return output_path

    except Exception as exc:
        _log.warning("mst_network_chart_error", error=str(exc))
        return None


def engine_scores_radar(
    scores: dict[str, int],
    output_path: Path,
) -> Path | None:
    """
    Radar chart dos 5 scores dos motores do Macro Desk (-2 a +2).
    """
    try:
        from pyecharts import options as opts
        from pyecharts.charts import Radar

        labels = {
            "rational":          "Rational",
            "behavioral":        "Behavioral",
            "entropy":           "Entropy",
            "valuation_gap":     "Valuation Gap",
            "regime_confidence": "Regime Conf.",
        }

        schema = [
            opts.RadarIndicatorItem(name=label, min_=-2, max_=2)
            for label in labels.values()
        ]
        values = [scores.get(k, 0) for k in labels]

        # Cor baseada na média dos scores
        avg = sum(values) / len(values) if values else 0
        color = "#34d399" if avg > 0 else "#f87171" if avg < -0.5 else "#fbbf24"

        radar = (
            Radar(init_opts=opts.InitOpts(
                width="100%", height="420px",
                bg_color="#0f1117",
            ))
            .add_schema(
                schema=schema,
                shape="polygon",
                center=["50%", "55%"],
                radius="65%",
                angleaxis_opts=opts.AngleAxisOpts(
                    axislabel_opts=opts.LabelOpts(
                        font_size=11, color="#e2e8f0",
                    )
                ),
                textstyle_opts=opts.TextStyleOpts(color="#e2e8f0", font_size=11),
                splitline_opt=opts.SplitLineOpts(
                    linestyle_opts=opts.LineStyleOpts(color="#2a2d3a")
                ),
                splitarea_opt=opts.SplitAreaOpts(
                    areastyle_opts=opts.AreaStyleOpts(color=["#1a1d27", "#0f1117"])
                ),
                axisline_opt=opts.AxisLineOpts(
                    linestyle_opts=opts.LineStyleOpts(color="#2a2d3a")
                ),
            )
            .add(
                series_name="Engines",
                data=[values],
                color=color,
                areastyle_opts=opts.AreaStyleOpts(opacity=0.25, color=color),
                linestyle_opts=opts.LineStyleOpts(width=2, color=color),
                label_opts=opts.LabelOpts(
                    is_show=True, font_size=12, color=color,
                    formatter=lambda p: f"{p.value:+d}",
                ),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="Engine Scores",
                    title_textstyle_opts=opts.TextStyleOpts(color="#e2e8f0", font_size=13),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
                tooltip_opts=opts.TooltipOpts(),
            )
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        radar.render(str(output_path))
        _log.info("engine_radar_saved", path=str(output_path))
        return output_path

    except Exception as exc:
        _log.warning("engine_radar_error", error=str(exc))
        return None


def rmt_correlation_heatmap(
    corr_clean: dict[str, dict[str, float]],
    market_prices: dict,
    output_path: Path,
) -> Path | None:
    """
    Heatmap da correlação limpa pelo RMT (Marchenko-Pastur).
    """
    try:
        from pyecharts import options as opts
        from pyecharts.charts import HeatMap

        symbols = list(corr_clean.keys())
        if not symbols:
            return None
        names = [market_prices.get(s, {}).get("name", s) for s in symbols]

        data = []
        for i, si in enumerate(symbols):
            for j, sj in enumerate(symbols):
                v = corr_clean.get(si, {}).get(sj, 0)
                data.append([j, i, round(float(v), 3)])

        heatmap = (
            HeatMap(init_opts=opts.InitOpts(
                width="100%", height="480px",
                bg_color="#0f1117",
            ))
            .add_xaxis(names)
            .add_yaxis(
                series_name="Correlação RMT",
                yaxis_data=names,
                value=data,
                label_opts=opts.LabelOpts(
                    is_show=True, position="inside",
                    font_size=9, color="#e2e8f0",
                    formatter="{c}",
                ),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="Correlação Limpa (RMT)",
                    subtitle="ruído de Marchenko-Pastur removido",
                    title_textstyle_opts=opts.TextStyleOpts(color="#e2e8f0", font_size=13),
                    subtitle_textstyle_opts=opts.TextStyleOpts(color="#8892a4", font_size=10),
                ),
                visualmap_opts=opts.VisualMapOpts(
                    min_=-1, max_=1,
                    is_calculable=True,
                    orient="horizontal",
                    pos_left="center",
                    pos_bottom="2%",
                    range_color=["#ef4444", "#1a1d27", "#34d399"],
                    textstyle_opts=opts.TextStyleOpts(color="#8892a4"),
                ),
                xaxis_opts=opts.AxisOpts(
                    axislabel_opts=opts.LabelOpts(
                        rotate=30, font_size=10, color="#8892a4"
                    )
                ),
                yaxis_opts=opts.AxisOpts(
                    axislabel_opts=opts.LabelOpts(font_size=10, color="#8892a4")
                ),
                tooltip_opts=opts.TooltipOpts(formatter="{b}: {c}"),
            )
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        heatmap.render(str(output_path))
        _log.info("rmt_heatmap_saved", path=str(output_path))
        return output_path

    except Exception as exc:
        _log.warning("rmt_heatmap_error", error=str(exc))
        return None


def generate_all_charts(
    market_prices: dict[str, Any],
    risk_metrics: dict[str, Any],
    monte_carlo: dict[str, Any],
    output_dir: Path,
    kline_tickers: list[str] | None = None,
) -> dict[str, str]:
    """
    Gera todos os charts e retorna {chart_type: html_path}.

    Args:
        market_prices: output de market_prices.collect()
        risk_metrics:  output de risk.analyze_portfolio()
        monte_carlo:   output de monte_carlo.run_for_portfolio()
        output_dir:    diretório de saída
        kline_tickers: subset de tickers para K-line (default: SPX + NDX + Oil)
    """
    paths: dict[str, str] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    # K-line charts para tickers principais
    tickers_for_kline = kline_tickers or ["^GSPC", "^NDX", "CL=F", "GLD", "^VIX"]
    for sym in tickers_for_kline:
        if sym in market_prices:
            name = market_prices[sym].get("name", sym)
            safe_sym = sym.replace("^", "").replace("=", "").replace("-", "_")
            p = kline_chart(sym, name, output_dir / f"kline_{safe_sym}.html")
            if p:
                paths[f"kline_{safe_sym}"] = str(p)

    # Heatmap de correlação
    corr = risk_metrics.get("correlations", {})
    if corr:
        p = correlation_heatmap(corr, market_prices, output_dir / "correlation_heatmap.html")
        if p:
            paths["correlation_heatmap"] = str(p)

    # Radar de risco
    if risk_metrics.get("tickers"):
        p = risk_radar_chart(risk_metrics, market_prices, output_dir / "risk_radar.html")
        if p:
            paths["risk_radar"] = str(p)

    # Fan charts Monte Carlo
    for sym, mc in monte_carlo.items():
        if sym in market_prices:
            name = market_prices[sym].get("name", sym)
            safe_sym = sym.replace("^", "").replace("=", "").replace("-", "_")
            p = monte_carlo_fan_chart(sym, name, mc, output_dir / f"mc_fan_{safe_sym}.html")
            if p:
                paths[f"mc_fan_{safe_sym}"] = str(p)

    _log.info("charts_done", count=len(paths))
    return paths
