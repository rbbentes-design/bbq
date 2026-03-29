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
