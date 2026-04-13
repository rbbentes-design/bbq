"""Flow Patrol visualization: gauges, bar charts, grids, COT plots."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as wd

try:
    from .config import _C, DASH_TEMPLATE, FLOW_FIG_LAYOUT, HAS_BQPLOT, HAS_DATAGRID, wd
except ImportError:
    import sys, os as _os
    _dir = _os.path.dirname(_os.path.abspath(__file__)) if '__file__' in dir() else _os.getcwd()
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
    from config import _C, DASH_TEMPLATE, FLOW_FIG_LAYOUT, HAS_BQPLOT, HAS_DATAGRID, wd

try:
    import ipydatagrid as ipd
    from ipydatagrid import DataGrid, BarRenderer, TextRenderer
except ImportError:
    pass

try:
    import bqplot as bqp
except ImportError:
    pass


def _flow_border(fig_widget):
    return wd.VBox([fig_widget], layout={
        'border': f'1px solid {_C["border"]}', 'margin': '6px',
        'padding': '10px', 'width': '98%',
        'border_radius': '8px'})


def fp_plot_score_gauge(score):
    """Gauge chart do score combinado de fluxo."""
    prob = score.get('prob_up', 0.5)
    direction = score.get('direction', 'NEUTRAL')
    colors = {'BULLISH': _C['green'], 'BEARISH': _C['red'], 'NEUTRAL': _C['text_muted']}
    fig = go.FigureWidget(go.Indicator(
        mode="gauge+number+delta", value=prob * 100,
        title={'text': f"Flow Score: {direction}", 'font': {'color': _C['text_muted']}},
        number={'font': {'color': _C['text']}},
        delta={'reference': 50, 'increasing': {'color': _C['green']},
               'decreasing': {'color': _C['red']}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': _C['text_muted'],
                     'tickfont': {'color': _C['text_muted']}},
            'bar': {'color': colors.get(direction, _C['text_muted']), 'thickness': 0.3},
            'bgcolor': _C['card2'],
            'borderwidth': 1, 'bordercolor': _C['border'],
            'steps': [
                {'range': [0, 30], 'color': '#3a1a1a'},
                {'range': [30, 70], 'color': '#3a3520'},
                {'range': [70, 100], 'color': '#1a3a2a'}],
            'threshold': {'line': {'color': _C['text'], 'width': 3},
                          'thickness': 0.8, 'value': prob * 100}
        }))
    fig.update_layout(height=280, margin=dict(t=50, b=20, l=20, r=20),
                      **{k: v for k, v in FLOW_FIG_LAYOUT.items()
                         if k != 'margin'})
    return fig


def fp_plot_components_bar(score):
    """Barras dos componentes do flow score."""
    components = {
        'CTA': score.get('z_cta', 0),
        'Dealer/MM': score.get('z_dealer', 0),
        'Vol Ctrl': score.get('z_volctrl', 0),
        'Risk Parity': score.get('z_rp', 0),
        'ETFs Alav.': score.get('z_leveraged', 0),
        'ETFs Passivos': score.get('z_passive_etf', 0),
        'Buyback': score.get('z_buyback', 0),
        'COT': score.get('z_cot', 0),
    }
    weights = score.get('weights', {})
    names = list(components.keys())
    values = list(components.values())
    w_vals = [weights.get(k, 0) for k in ['cta', 'dealer', 'volctrl', 'rp', 'leveraged', 'passive_etf', 'buyback', 'cot']]
    colors_bar = [_C['accent'] if v >= 0 else _C['red'] for v in values]
    fig = go.FigureWidget()
    fig.add_trace(go.Bar(x=names, y=values, marker_color=colors_bar,
                         name='Z-Score', text=[f'{v:+.2f}' for v in values],
                         textposition='outside'))
    fig.add_trace(go.Scatter(x=names, y=w_vals, name='Peso',
                             yaxis='y2', mode='markers+text',
                             text=[f'{w:.0%}' for w in w_vals],
                             textposition='top center',
                             marker=dict(size=10, color=_C['text_muted'])))
    fig.update_layout(
        title='Componentes do Flow Score',
        yaxis_title='Z-Score',
        yaxis2=dict(overlaying='y', side='right',
                    title='Peso', range=[0, 1]),
        xaxis=dict(tickangle=-20, automargin=True),
        **{**FLOW_FIG_LAYOUT, 'margin': dict(t=55, r=40, b=110, l=50)},
    )
    return fig


def fp_plot_flow_history(flow_hist):
    """Série histórica de fluxo vs retorno."""
    if flow_hist.empty:
        fig = go.FigureWidget()
        fig.update_layout(title="Sem histórico de fluxo")
        return fig
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=flow_hist.index, y=flow_hist['LevETF_Flow'],
                             name='Lev ETF Flow', line=dict(color=_C['orange'], width=1.5)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=flow_hist.index, y=flow_hist['Return'],
                             name='Return', line=dict(color=_C['accent'], width=1)),
                  secondary_y=True)
    fig.update_layout(title='Fluxo ETFs Alavancados vs Retorno',
                      hovermode='x unified', **FLOW_FIG_LAYOUT)
    fig.update_yaxes(title_text='Flow ($)', secondary_y=False)
    fig.update_yaxes(title_text='Return', secondary_y=True)
    return go.FigureWidget(fig)


def fp_plot_positions_basket(df_cot, basket_col='Price', data_col='Positions'):
    """Dual-axis: positions + preço."""
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(
        x=df_cot.index, y=df_cot[data_col], name=data_col,
        yaxis='y1', line=dict(width=1.5), marker_color=_C['orange']))
    if basket_col in df_cot.columns:
        fig.add_trace(go.Scatter(
            x=df_cot.index, y=df_cot[basket_col], name=basket_col,
            yaxis='y2', line=dict(width=1), marker_color=_C['accent']))
    fig.update_yaxes(zeroline=False, showgrid=False)
    fig.layout.yaxis.update(title_text=data_col)
    fig.layout.yaxis2.update(title_text=basket_col)
    fig.update_layout(title='Positions & Basket Price', **FLOW_FIG_LAYOUT)
    return _flow_border(go.FigureWidget(fig))


def fp_plot_long_short_net(df_cot):
    """Barras Long/Short + linha Net."""
    fig = go.FigureWidget()
    for name, color in [('Long', _C['teal']), ('Short', _C['text_muted'])]:
        col = f'Positions - {name}'
        if col in df_cot.columns:
            fig.add_trace(go.Bar(x=df_cot.index, y=df_cot[col],
                                 name=name, marker_color=color))
    if 'Positions - Net' in df_cot.columns:
        fig.add_trace(go.Scatter(x=df_cot.index, y=df_cot['Positions - Net'],
                                 name='Net', marker_color=_C['orange']))
    fig.update_layout(barmode='relative',
                      title='Long, Short & Net Positions',
                      yaxis_title='Positions', hovermode='x unified',
                      **FLOW_FIG_LAYOUT)
    return _flow_border(fig)


def fp_plot_correlation(df_cot, window=26):
    """Rolling correlation entre preço e net positions."""
    if 'Price' not in df_cot.columns or 'Positions - Net' not in df_cot.columns:
        return wd.VBox([wd.HTML("<p>Sem dados para correlação.</p>")])
    corr = (df_cot[['Price', 'Positions - Net']].pct_change()
            .rolling(window).corr().unstack()[('Price', 'Positions - Net')])
    fig = go.FigureWidget(go.Scatter(
        x=corr.index, y=corr.values, name='Corr',
        line=dict(width=1.5), marker_color=_C['orange']))
    fig.update_yaxes(zeroline=False, showgrid=False)
    fig.layout.yaxis.update(title_text='Correlation')
    fig.update_layout(title='Rolling Correlation: Price Δ vs Net Length Δ',
                      **FLOW_FIG_LAYOUT)
    return _flow_border(fig)


def fp_plot_long_short_ratio(df_cot):
    """Long/Short ratio."""
    if 'Positions - Long' not in df_cot.columns or 'Positions - Short' not in df_cot.columns:
        return wd.VBox([wd.HTML("<p>Sem dados L/S.</p>")])
    ratio = df_cot['Positions - Long'] / df_cot['Positions - Short'] * -1
    fig = go.FigureWidget(go.Scatter(
        x=ratio.index, y=ratio.values, name='L/S Ratio',
        line=dict(width=1.5, color=_C['orange'])))
    fig.update_yaxes(zeroline=False, showgrid=False)
    fig.layout.yaxis.update(title_text='Ratio')
    fig.update_layout(title='Long/Short Ratio', **FLOW_FIG_LAYOUT)
    return _flow_border(fig)


def fp_plot_multi_year(df_cot):
    """Gráfico sazonal multi-ano."""
    if 'week' not in df_cot.columns or 'year' not in df_cot.columns:
        return wd.VBox([wd.HTML("<p>Sem dados semanais.</p>")])
    pivot = df_cot.pivot(columns='year', index='week', values='Positions')
    pivot = pivot.iloc[:, -6:]
    fig = go.FigureWidget()
    colors_yr = [_C['text_dim'], _C['text_muted'], '#8b949e', _C['yellow'], _C['orange'], _C['teal']]
    for col_name, color in zip(pivot.columns, colors_yr):
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[col_name], mode='lines',
            line=dict(color=color), name=str(col_name)))
    fig.update_layout(hovermode='x unified', yaxis_title='Positions',
                      title='Seasonal Analysis', **FLOW_FIG_LAYOUT)
    return _flow_border(fig)


def fp_plot_dispersion(seasonality_df, df_cot, col='Positions'):
    """Dispersion chart: min/max band + mean + current year."""
    seas = seasonality_df[col].copy() if col in seasonality_df.columns else pd.DataFrame()
    if seas.empty:
        return wd.VBox([wd.HTML("<p>Sem dados de sazonalidade.</p>")])
    current_year = pd.Timestamp.now().year
    yr_data = df_cot[df_cot.index.year == current_year] if hasattr(df_cot.index, 'year') else pd.DataFrame()
    if not yr_data.empty and 'week' in yr_data.columns and col in yr_data.columns:
        seas['Current'] = yr_data.set_index('week')[col]
    fig = go.FigureWidget()
    for name, fill in [('Max', None), ('Min', 'tonexty')]:
        if name in seas.columns:
            fig.add_trace(go.Scatter(
                x=seas.index, y=seas[name], mode='lines', name=name,
                fill=fill, fillcolor='rgba(88,166,255,0.08)',
                line=dict(color=_C['border'], width=0), showlegend=False))
    for name, color in [('Mean', _C['orange']), ('Current', _C['teal'])]:
        if name in seas.columns:
            fig.add_trace(go.Scatter(
                x=seas.index, y=seas[name], mode='lines+markers',
                name=name, line=dict(color=color)))
    fig.update_layout(title='5Y Dispersion', xaxis_title='Weeks',
                      yaxis_title=col, hovermode='x unified',
                      xaxis=dict(range=[1, 53]), **FLOW_FIG_LAYOUT)
    return _flow_border(fig)


def fp_bqp_flow_bar_line(flow_df, bar_col='LevETF_Flow', line_col='Return'):
    """Bar + line overlay usando bqplot."""
    if not HAS_BQPLOT or flow_df.empty:
        return wd.HTML("<p>bqplot não disponível ou sem dados.</p>")
    scale_x = bqp.DateScale()
    scale_y = bqp.LinearScale()
    mark_bar = bqp.Bars(
        x=flow_df.index, y=flow_df[bar_col],
        scales={'x': scale_x, 'y': scale_y}, colors=[_C['accent']],
        tooltip=bqp.Tooltip(fields=['y', 'x'], show_labels=False,
                            formats=['.0f', '%Y/%m/%d']))
    marks = [mark_bar]
    if line_col in flow_df.columns:
        scale_y2 = bqp.LinearScale()
        mark_line = bqp.Lines(
            x=flow_df.index, y=flow_df[line_col], stroke_width=2,
            scales={'x': scale_x, 'y': scale_y2}, colors=[_C['purple']])
        marks.append(mark_line)
    ax_y = bqp.Axis(scale=scale_y, orientation='vertical', label='Flow ($)')
    ax_x = bqp.Axis(scale=scale_x)
    return bqp.Figure(
        marks=marks, axes=[ax_x, ax_y],
        title='Fluxo de ETFs Alavancados',
        title_style={'font-size': '18px'}, padding_y=0,
        fig_margin={'top': 50, 'bottom': 50, 'left': 60, 'right': 50},
        layout={'width': 'auto', 'height': '400px'})


def fp_bqp_scatter(flow_df):
    """Scatter plot: flow vs return (bqplot)."""
    if not HAS_BQPLOT or flow_df.empty:
        return wd.HTML("<p>bqplot não disponível ou sem dados.</p>")
    scale_x = bqp.LinearScale()
    scale_y = bqp.LinearScale()
    tooltip = bqp.Tooltip(fields=['x', 'y'], labels=['Flow', 'Return'],
                          formats=['.0f', '.4f'])
    mark = bqp.Scatter(
        x=flow_df['LevETF_Flow'], y=flow_df['Return'],
        tooltip=tooltip, scales={'x': scale_x, 'y': scale_y},
        default_size=32, colors=[_C['orange']])
    ax_x = bqp.Axis(scale=scale_x, label='Flow ($)')
    ax_y = bqp.Axis(scale=scale_y, orientation='vertical', label='Return')
    return bqp.Figure(
        marks=[mark], axes=[ax_x, ax_y],
        title='Flow vs Return', title_style={'font-size': '18px'},
        padding_x=0.05, padding_y=0.05,
        layout={'width': '100%', 'height': '400px'})


def fp_grid_flow_score(score):
    """Tabela interativa do flow score."""
    if not HAS_DATAGRID:
        rows_html = [f"<tr><td>{k}</td><td>{v:.3f}</td></tr>"
                     for k, v in score.items() if isinstance(v, (int, float))]
        return wd.HTML(f"<table>{''.join(rows_html)}</table>")
    data = {
        'Componente': ['CTA Trend', 'Dealer/MM', 'Vol Control',
                        'Risk Parity', 'ETFs Alavancados', 'ETFs Passivos',
                        'Buyback', 'COT', 'Score Combinado'],
        'Z-Score': [score.get('z_cta', 0),
                    score.get('z_dealer', 0), score.get('z_volctrl', 0),
                    score.get('z_rp', 0),
                    score.get('z_leveraged', 0), score.get('z_passive_etf', 0),
                    score.get('z_buyback', 0),
                    score.get('z_cot', 0),
                    score.get('combined_score', 0)],
        'Peso (%)': [score.get('weights', {}).get('cta', 0) * 100,
                     score.get('weights', {}).get('dealer', 0) * 100,
                     score.get('weights', {}).get('volctrl', 0) * 100,
                     score.get('weights', {}).get('rp', 0) * 100,
                     score.get('weights', {}).get('leveraged', 0) * 100,
                     score.get('weights', {}).get('passive_etf', 0) * 100,
                     score.get('weights', {}).get('buyback', 0) * 100,
                     score.get('weights', {}).get('cot', 0) * 100,
                     100],
    }
    df_s = pd.DataFrame(data).set_index('Componente')
    try:
        linear_scale = bqp.LinearScale(min=-3, max=3)
        color_scale = bqp.ColorScale(min=-3, max=3,
                                     colors=[_C['red'], _C['card2'], _C['green']])
        renderers = {
            'Z-Score': BarRenderer(bar_value=linear_scale, format='.2f',
                                   bar_color=color_scale,
                                   horizontal_alignment='center'),
        }
    except Exception:
        renderers = {}
    return DataGrid(df_s, renderers=renderers, base_column_size=150,
                    layout={'height': '200px'})


def fp_grid_cot_stats(stats):
    """Tabela de estatísticas COT."""
    if stats.empty:
        return wd.HTML("<p>Sem dados COT.</p>")
    df_s = stats.to_frame('Value')
    if not HAS_DATAGRID:
        return wd.HTML(df_s.to_html())
    try:
        from ipydatagrid import VegaExpr
        renderers = {
            'Value': TextRenderer(
                background_color=VegaExpr(
                    f"cell.value < -1 ? '{_C['red']}' : "
                    f"cell.value > 1 ? '{_C['green']}' : 'transparent'"),
                format='.2f')
        }
    except Exception:
        renderers = {}
    return DataGrid(df_s, renderers=renderers, base_column_size=150,
                    base_row_header_size=200, layout={'height': '300px'})


def fp_grid_buyback(buyback_df):
    """Tabela de buyback por empresa."""
    if buyback_df.empty:
        return wd.HTML("<p>Sem dados de buyback.</p>")
    if not HAS_DATAGRID:
        return wd.HTML(buyback_df.head(20).to_html())
    try:
        scale = bqp.LinearScale(min=0, max=buyback_df['daily_est'].max())
        renderers = {
            'daily_est': BarRenderer(bar_value=scale, format='$,.0f',
                                     bar_color=_C['green'],
                                     horizontal_alignment='right'),
        }
    except Exception:
        renderers = {}
    return DataGrid(buyback_df.head(30), renderers=renderers,
                    base_column_size=120, base_row_header_size=140,
                    layout={'height': '400px'})
