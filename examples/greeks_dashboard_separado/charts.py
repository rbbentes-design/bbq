"""Charts and analysis: macro, skew, dealer scenarios, gamma, squeeze, vol smile, dynamic book."""

import numpy as np
import pandas as pd
import math
import traceback
import warnings
import json as _de_json
import uuid as _de_uuid
import bql
from datetime import datetime, timedelta
from datetime import datetime as _de_dt, date as _de_date, timedelta as _de_td
from dataclasses import dataclass as _de_dc, field as _de_field, asdict as _de_asdict
from enum import Enum as _de_Enum
from scipy.stats import norm, t as student_t
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as wd
from IPython.display import display, HTML

try:
    from .config import (bq, _C, DASH_TEMPLATE, FLOW_FIG_LAYOUT, HAS_DATAGRID,
                         TRADING_DAYS, FUTURES_MULTIPLIER, GREEK_CONFIGS, INDEX_PROXY,
                         MAG7, _greek_cache, _snapshot,
                         SPX_ANNUAL_BUYBACK_EST, MM_VOLUME_SHARES,
                         VOL_CTRL_AUM, VOL_CTRL_MAX_LEV, VOL_CTRL_MIN_EXP, _vc_exposure)
    from .data import _bql_ts, _bql_ts_df, fetch_options_chain
    from .exposure import compute_strike_exposures, compute_walls
    from .greeks import calculate_all_greeks, black_scholes_price_vec, fmt_value
    from .ui import _hud_panel, _svg_ring_html, create_gauge, create_symmetric_gauge
except ImportError:
    import sys, os as _os
    _dir = _os.path.dirname(_os.path.abspath(__file__)) if '__file__' in dir() else _os.getcwd()
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
    from config import (bq, _C, DASH_TEMPLATE, FLOW_FIG_LAYOUT, HAS_DATAGRID,
                        TRADING_DAYS, FUTURES_MULTIPLIER, GREEK_CONFIGS, INDEX_PROXY,
                        MAG7, _greek_cache, _snapshot,
                        SPX_ANNUAL_BUYBACK_EST, MM_VOLUME_SHARES,
                        VOL_CTRL_AUM, VOL_CTRL_MAX_LEV, VOL_CTRL_MIN_EXP, _vc_exposure)
    from data import _bql_ts, _bql_ts_df, fetch_options_chain
    from exposure import compute_strike_exposures, compute_walls
    from greeks import calculate_all_greeks, black_scholes_price_vec, fmt_value
    from ui import _hud_panel, _svg_ring_html, create_gauge, create_symmetric_gauge

def build_mbad_summary_cards(prices_df, weights=None, spx_chg_pct=None):
    """
    Breadth summary cards:
    - SPX Last Price + %chg
    - Stocks Advancing count + %
    - Stocks Declining count + %
    - Breadth Gauge (strong/weak)
    Retorna HTML widget.
    """
    rets_1d = (prices_df.iloc[-1] / prices_df.iloc[-2] - 1) * 100
    rets = rets_1d.dropna()
    n_up = int((rets >= 0).sum())
    n_down = int((rets < 0).sum())
    n_total = n_up + n_down
    pct_up = n_up / n_total * 100 if n_total > 0 else 0
    pct_down = n_down / n_total * 100 if n_total > 0 else 0

    # Breadth gauge: ratio-based
    breadth_ratio = n_up / n_total if n_total > 0 else 0.5
    if breadth_ratio >= 0.65:
        gauge_label, gauge_color = 'Strong', '#22c55e'
    elif breadth_ratio >= 0.55:
        gauge_label, gauge_color = 'Moderate', '#fbbf24'
    elif breadth_ratio >= 0.45:
        gauge_label, gauge_color = 'Neutral', '#8b949e'
    elif breadth_ratio >= 0.35:
        gauge_label, gauge_color = 'Weak', '#f97316'
    else:
        gauge_label, gauge_color = 'Very Weak', '#dc2626'

    # SPX price from last index column if available
    spx_cols = [c for c in prices_df.columns if 'Index' in c or 'SPX' in c]
    spx_px = prices_df[spx_cols[0]].iloc[-1] if spx_cols else 0
    spx_chg = spx_chg_pct if spx_chg_pct is not None else (
        (prices_df[spx_cols[0]].iloc[-1] / prices_df[spx_cols[0]].iloc[-2] - 1) * 100
        if spx_cols and len(prices_df) >= 2 else 0)
    spx_chg_color = '#22c55e' if spx_chg >= 0 else '#dc2626'

    # Gauge SVG (speedometer arc)
    angle = int(180 * breadth_ratio)  # 0-180 degrees
    gauge_svg = (
        f"<svg viewBox='0 0 120 70' width='120' height='70'>"
        f"<path d='M 10 60 A 50 50 0 0 1 110 60' fill='none' stroke='#30363d' stroke-width='8'/>"
        f"<path d='M 10 60 A 50 50 0 0 1 110 60' fill='none' stroke='url(#gaugeGrad)' "
        f"stroke-width='8' stroke-dasharray='{angle * 1.74} 314'/>"
        f"<defs><linearGradient id='gaugeGrad'>"
        f"<stop offset='0%' stop-color='#dc2626'/>"
        f"<stop offset='50%' stop-color='#fbbf24'/>"
        f"<stop offset='100%' stop-color='#22c55e'/>"
        f"</linearGradient></defs>"
        f"<text x='60' y='55' text-anchor='middle' fill='{gauge_color}' "
        f"font-size='12' font-weight='bold'>{gauge_label}</text>"
        f"<text x='60' y='68' text-anchor='middle' fill='#8b949e' "
        f"font-size='8'>{pct_up:.0f}% / {pct_down:.0f}%</text>"
        f"</svg>"
    )

    card_style = ("display:inline-block; background:#161b22; padding:12px 20px; "
                  "border-radius:8px; margin:4px; text-align:center; min-width:140px; "
                  "border:1px solid #30363d;")

    html = (
        f"<div style='display:flex; flex-wrap:wrap; justify-content:center; gap:8px; margin:8px 0;'>"
        # SPX Price card
        f"<div style='{card_style}'>"
        f"<div style='color:#8b949e; font-size:10px;'>Last Price</div>"
        f"<div style='color:#c9d1d9; font-size:24px; font-weight:bold;'>{spx_px:,.2f}</div>"
        f"<div style='color:{spx_chg_color}; font-size:13px;'>{spx_chg:+.2f}%</div></div>"
        # Advancing card
        f"<div style='{card_style} border-color:#22c55e;'>"
        f"<div style='color:#8b949e; font-size:10px;'>Stocks Advancing</div>"
        f"<div style='color:#22c55e; font-size:28px; font-weight:bold;'>{n_up}</div>"
        f"<div style='color:#22c55e; font-size:13px;'>▲{pct_up:.1f}%</div></div>"
        # Declining card
        f"<div style='{card_style}'>"
        f"<div style='color:#8b949e; font-size:10px;'>Stocks Declining</div>"
        f"<div style='color:#dc2626; font-size:28px; font-weight:bold;'>{n_down}</div>"
        f"<div style='color:#dc2626; font-size:13px;'>▼{pct_down:.1f}%</div></div>"
        # Gauge card
        f"<div style='{card_style}'>{gauge_svg}</div>"
        f"</div>"
    )
    return wd.HTML(html)


def fetch_spx_eq_weight_correlation(lookback=2520):
    """
    Busca SPX e SPW Index (S&P 500 Equal Weight) via BQL.
    Calcula correlação rolling 3M (63 dias úteis).
    Retorna (correlation_series, fig_widget).
    SPX vs Equal-Weight SPX Rolling Correlation (3M window).
    """
    bq_svc = bql.Service()
    dt_rng = bq_svc.func.range(f'-{lookback}d', '0d')

    # SPX vs S&P 500 Equal Weight Index (SPW Index on Bloomberg)
    tickers = ['SPX Index', 'SPW Index']
    prices = {}
    for tk in tickers:
        try:
            req = bql.Request(tk, {'px': bq_svc.data.px_last(fill='PREV', dates=dt_rng)})
            s = _bql_ts(bq_svc.execute(req)[0], 'px').dropna()
            if not s.empty:
                prices[tk] = s
        except Exception as e:
            print(f"⚠️ EQ Weight Corr — {tk}: {e}")

    if len(prices) < 2:
        return pd.Series(dtype=float), wd.HTML(
            '<p style="color:#8b949e;">Sem dados SPX/EW para correlação.</p>')

    df = pd.DataFrame(prices).dropna()
    if len(df) < 63:
        return pd.Series(dtype=float), wd.HTML(
            '<p style="color:#8b949e;">Dados insuficientes para correlação 3M.</p>')

    # Log returns
    log_rets = np.log(df / df.shift(1)).dropna()

    # Rolling 63-day (3M) correlation
    corr_3m = log_rets.iloc[:, 0].rolling(63).corr(log_rets.iloc[:, 1])
    corr_3m = corr_3m.dropna()

    # Build chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=corr_3m.index, y=corr_3m.values,
        mode='lines', name='SPX/Equal Weight 3M Correlation',
        line=dict(color='#2dd4bf', width=1.5),
        fill='tozeroy', fillcolor='rgba(45,212,191,0.08)',
    ))

    # Reference lines
    fig.add_hline(y=0.7, line=dict(color='#8b949e', dash='dash', width=1),
                  annotation_text='0.70', annotation_position='right',
                  annotation_font=dict(size=10, color='#8b949e'))

    # Current value annotation
    curr_val = corr_3m.iloc[-1] if len(corr_3m) > 0 else 0
    fig.add_annotation(
        x=0.02, y=0.05, xref='paper', yref='paper',
        text=f"Atual: <b>{curr_val:.4f}</b>",
        showarrow=False, font=dict(size=12, color='#2dd4bf'),
        bgcolor='rgba(22,27,34,0.8)', bordercolor='#30363d',
        borderpad=6, xanchor='left', yanchor='bottom')

    fig.update_layout(
        title='SPX and Equal-Weight SPX Rolling Correlation (3M)',
        template=DASH_TEMPLATE,
        height=380,
        margin=dict(l=50, r=30, t=45, b=40),
        xaxis_title='', yaxis_title='Correlation',
        yaxis=dict(range=[0.5, 1.02]),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.18,
                    xanchor='center', x=0.5),
    )
    return corr_3m, go.FigureWidget(fig)


def fetch_odte_volume_pct(lookback=2000):
    """
    Busca volume de opções 0DTE do SPX como % do volume total via BQL.
    Usa SPX Index options volume histórico.
    Retorna (series, fig_widget).
    0DTE SPX Option Volume as a Percentage of Total Volume.
    """
    bq_svc = bql.Service()
    dt_rng = bq_svc.func.range(f'-{lookback}d', '0d')

    # Fetch total SPX options volume and 0DTE proxy
    # Bloomberg fields: OPT_VOL (total), OPT_VOL_0DTE may not exist
    # Use VOLUME_CALL + VOLUME_PUT for total, and SHORT_TERM_OPTIONS_VOLUME as proxy
    vol_data = {}
    # Try fetching SPX total options volume
    for tk, label in [('SPX Index', 'spx_vol'), ('VIX Index', 'vix')]:
        try:
            req = bql.Request(tk, {
                'vol': bq_svc.data.px_volume(fill='PREV', dates=dt_rng)
            })
            s = _bql_ts(bq_svc.execute(req)[0], 'vol').dropna()
            if not s.empty:
                vol_data[label] = s
        except Exception:
            pass

    # Try Bloomberg 0DTE percentage series (SPXW)
    # SPX 0DTE is tracked via SPXW (weekly) volume vs total
    try:
        # Attempt SPXW as 0DTE proxy —  SPX Weeklys
        req = bql.Request('SPXW Index', {
            'vol': bq_svc.data.px_volume(fill='PREV', dates=dt_rng)
        })
        s = _bql_ts(bq_svc.execute(req)[0], 'vol').dropna()
        if not s.empty:
            vol_data['spxw_vol'] = s
    except Exception:
        pass

    if 'spx_vol' not in vol_data:
        # Fallback: use OPT_IMPLIED_VOL_AVG_7DAY / VIX as proxy for activity
        return pd.Series(dtype=float), wd.HTML(
            '<p style="color:#8b949e;">Sem dados de volume de opções SPX.</p>')

    # Calculate ratio
    if 'spxw_vol' in vol_data and 'spx_vol' in vol_data:
        df = pd.DataFrame({'total': vol_data['spx_vol'],
                           'short': vol_data['spxw_vol']}).dropna()
        if not df.empty and (df['total'] > 0).any():
            ratio = (df['short'] / df['total']).clip(0, 1) * 100
        else:
            ratio = pd.Series(dtype=float)
    else:
        # Synthetic estimate: use VIX as activity proxy
        # 0DTE has grown from ~10% (2018) to ~60% (2025)
        ratio = pd.Series(dtype=float)

    if ratio.empty:
        return pd.Series(dtype=float), wd.HTML(
            '<p style="color:#8b949e;">Sem dados 0DTE/SPXW volume.</p>')

    # 1M moving average
    ma_1m = ratio.rolling(21, min_periods=5).mean()

    fig = go.Figure()
    # Daily bars (cyan, semi-transparent)
    fig.add_trace(go.Bar(
        x=ratio.index, y=ratio.values,
        name='0DTE as % of Total Volume',
        marker=dict(color='rgba(45,212,191,0.5)'),
    ))
    # 1M moving average line (orange)
    fig.add_trace(go.Scatter(
        x=ma_1m.index, y=ma_1m.values,
        mode='lines', name='0DTE % 1M Avg',
        line=dict(color='#f97316', width=2.5),
    ))

    # Key event annotations
    events = [
        ('2022-05-01', 'Tues/Thurs OpEx'),
        ('2025-01-01', 'Robinhood Launched\nSPX Options'),
    ]
    for edt, elbl in events:
        try:
            edt_ts = pd.Timestamp(edt)
            if ratio.index.min() <= edt_ts <= ratio.index.max():
                fig.add_vline(x=edt_ts, line=dict(color='#dc2626', dash='dash', width=1))
                fig.add_annotation(
                    x=edt_ts, y=ratio.max() * 0.9,
                    text=elbl, showarrow=False,
                    font=dict(size=9, color='#dc2626'),
                    textangle=-90)
        except Exception:
            pass

    # 60% reference line
    fig.add_hline(y=60, line=dict(color='#8b949e', dash='dash', width=1))

    fig.update_layout(
        title='0DTE SPX Option Volume as a Percentage of Total Volume',
        template=DASH_TEMPLATE,
        height=400,
        margin=dict(l=50, r=60, t=45, b=40),
        xaxis_title='', yaxis_title='0DTE % of SPX Total Volume',
        yaxis2=dict(title='0DTE % SPX Total Volume', overlaying='y',
                    side='right', showgrid=False),
        bargap=0,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.18,
                    xanchor='center', x=0.5),
    )
    return ratio, go.FigureWidget(fig)


def build_buyback_blackout_chart(blackout_curve, earnings_df=None, buyback_annual=None):
    """
    Buyback blackout chart:
    - Teal area: % of S&P 500 in blackout period
    - Orange bars: earnings reports/day
    - Purple line: estimated daily buyback flow ($B) modulated by blackout
    - "Today" marker with current pct annotation
    Retorna FigureWidget.
    """
    if blackout_curve.empty:
        return wd.HTML('<p style="color:#8b949e;">Sem dados de blackout.</p>')

    _bc = blackout_curve.copy()
    _bc['pct'] = _bc['pct_blackout'] * 100

    # Estimate daily buyback flow across full year modulated by blackout openness
    annual_bb = buyback_annual if buyback_annual and buyback_annual > 0 else SPX_ANNUAL_BUYBACK_EST
    _bc['open_pct'] = 1.0 - _bc['pct_blackout']
    # Weight each day's flow by proportion of market open for buybacks
    # Normalize so total across year = annual_bb * execution_rate
    open_sum = _bc['open_pct'].sum()
    if open_sum > 0:
        _bc['daily_flow'] = annual_bb * 0.80 * _bc['open_pct'] / open_sum
    else:
        _bc['daily_flow'] = annual_bb * 0.80 / len(_bc)

    # Rolling 5d for smoother display
    _bc['flow_5d'] = _bc['daily_flow'].rolling(5, min_periods=1, center=True).mean()

    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]])

    # Area fill: % in blackout (teal/dark cyan)
    fig.add_trace(go.Scatter(
        x=_bc['date'], y=_bc['pct'],
        mode='lines', fill='tozeroy',
        fillcolor='rgba(45,212,191,0.25)',
        line=dict(color='#2dd4bf', width=1.5),
        name='% S&P 500 in Blackout',
    ), secondary_y=False)

    # Earnings reports/day as bars (if we have earnings data)
    if earnings_df is not None and not earnings_df.empty:
        earn_counts = (earnings_df['earn_dt'].dt.normalize()
                       .value_counts().sort_index())
        mask = (earn_counts.index >= _bc['date'].min()) & (earn_counts.index <= _bc['date'].max())
        earn_counts = earn_counts[mask]
        if not earn_counts.empty:
            fig.add_trace(go.Bar(
                x=earn_counts.index, y=earn_counts.values,
                name='Earnings Reports/Day',
                marker=dict(color='rgba(249,115,22,0.7)'),
                opacity=0.6,
            ), secondary_y=True)

    # Buyback flow line (purple) on secondary axis
    fig.add_trace(go.Scatter(
        x=_bc['date'], y=_bc['flow_5d'] / 1e9,
        mode='lines', name='Est. Buyback Flow ($B/dia)',
        line=dict(color='#bc8cff', width=2, dash='solid'),
    ), secondary_y=True)

    # Mark "Hoje" (today)
    _today = pd.Timestamp.now().normalize()
    _today_row = _bc.loc[_bc['date'] == _today]
    if not _today_row.empty:
        today_pct = float(_today_row['pct'].iloc[0])
        today_flow = float(_today_row['flow_5d'].iloc[0]) / 1e9
        fig.add_vline(x=_today, line=dict(color='#dc2626', dash='dash', width=1.5))
        fig.add_annotation(
            x=_today, y=today_pct + 5,
            text=f"<b>{today_pct:.1f}%</b> in Blackout<br>Flow: ${today_flow:.2f}B/dia",
            showarrow=True, arrowhead=2, arrowcolor='#dc2626',
            font=dict(size=11, color='white'),
            bgcolor='rgba(139,25,25,0.8)', bordercolor='#dc2626',
            borderpad=5)
        fig.add_trace(go.Scatter(
            x=[_today], y=[today_pct],
            mode='markers', showlegend=False,
            marker=dict(size=8, color='#dc2626'),
        ), secondary_y=False)

    fig.update_layout(
        title=f"S&P 500 — Buyback Blackout Window (12M) vs. Earnings + Flow Estimado",
        template=DASH_TEMPLATE,
        height=420,
        margin=dict(l=50, r=60, t=45, b=40),
        xaxis_title='',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.18,
                    xanchor='center', x=0.5),
    )
    fig.update_yaxes(title_text='Percent of tickers in blackout', range=[0, 105],
                     secondary_y=False)
    fig.update_yaxes(title_text='Reports/Day | Flow ($B)',
                     showgrid=False, secondary_y=True)

    return go.FigureWidget(fig)


def build_spy_intraday_candlestick(ticker='SPY US Equity', lookback_days=5):
    """
    SPY candlestick chart via BQL OHLC data.
    Se OHLC intraday não disponível, usa barras diárias recentes.
    Inclui análise de padrões nos últimos 5 candles (blended).
    """
    bq_svc = bql.Service()
    dt_rng = bq_svc.func.range(f'-{lookback_days}d', '0d')

    ohlc = {}
    for field_name, field_label in [
        ('px_open', 'open'), ('px_high', 'high'),
        ('px_low', 'low'), ('px_last', 'close')]:
        try:
            fld = getattr(bq_svc.data, field_name)
            req = bql.Request(ticker, {field_label: fld(fill='PREV', dates=dt_rng)})
            s = _bql_ts(bq_svc.execute(req)[0], field_label).dropna()
            if not s.empty:
                ohlc[field_label] = s
        except Exception:
            pass

    if len(ohlc) < 4:
        return wd.HTML(f'<p style="color:#8b949e;">Sem dados OHLC para {ticker}.</p>')

    df = pd.DataFrame(ohlc).dropna()
    if df.empty:
        return wd.HTML(f'<p style="color:#8b949e;">OHLC vazio para {ticker}.</p>')

    # ── Candlestick pattern analysis (last 5 candles) ──
    patterns = []
    n = len(df)
    if n >= 2:
        o, h, lo, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
        body = c - o
        rng = h - lo
        for i in range(max(0, n - 5), n):
            b = body[i]
            r = rng[i] if rng[i] > 0 else 1e-9
            upper_shadow = h[i] - max(o[i], c[i])
            lower_shadow = min(o[i], c[i]) - lo[i]
            body_pct = abs(b) / r
            # Doji
            if body_pct < 0.1:
                patterns.append(('Doji', i, '#fbbf24'))
            # Hammer (small body top, long lower shadow)
            elif lower_shadow > abs(b) * 2 and upper_shadow < abs(b) * 0.5:
                patterns.append(('Hammer' if b >= 0 else 'Inverted Hammer', i, '#22c55e'))
            # Shooting star (small body bottom, long upper shadow)
            elif upper_shadow > abs(b) * 2 and lower_shadow < abs(b) * 0.5:
                patterns.append(('Shooting Star', i, '#dc2626'))
            # Engulfing
            if i > 0 and abs(b) > abs(body[i-1]) * 1.3:
                if b > 0 and body[i-1] < 0:
                    patterns.append(('Bullish Engulfing', i, '#22c55e'))
                elif b < 0 and body[i-1] > 0:
                    patterns.append(('Bearish Engulfing', i, '#dc2626'))

        # Blended candle analysis (combine last 5 into one)
        last5 = df.iloc[-min(5, n):]
        blended_open = last5['open'].iloc[0]
        blended_close = last5['close'].iloc[-1]
        blended_high = last5['high'].max()
        blended_low = last5['low'].min()
        blended_body = blended_close - blended_open
        blended_range = blended_high - blended_low
        blended_pct = abs(blended_body) / blended_range if blended_range > 0 else 0

        if blended_pct < 0.1:
            blended_signal = '⚖️ Indecisão (Doji Blended)'
            blended_color = '#fbbf24'
        elif blended_body > 0 and blended_pct > 0.6:
            blended_signal = '🟢 Forte Alta (Marubozu Blended)'
            blended_color = '#22c55e'
        elif blended_body < 0 and blended_pct > 0.6:
            blended_signal = '🔴 Forte Baixa (Marubozu Blended)'
            blended_color = '#dc2626'
        elif blended_body > 0:
            blended_signal = '🟢 Leve Alta (Blended)'
            blended_color = '#22c55e'
        else:
            blended_signal = '🔴 Leve Baixa (Blended)'
            blended_color = '#dc2626'

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name=ticker.split(' ')[0],
        increasing=dict(line=dict(color='#22c55e'), fillcolor='rgba(34,197,94,0.4)'),
        decreasing=dict(line=dict(color='#dc2626'), fillcolor='rgba(220,38,38,0.4)'),
    ))

    # Annotate detected patterns
    for pname, idx, pcolor in patterns:
        fig.add_annotation(
            x=df.index[idx], y=df['high'].iloc[idx],
            text=pname, showarrow=True, arrowhead=2,
            font=dict(size=8, color=pcolor),
            bgcolor='rgba(22,27,34,0.85)', borderpad=2,
            ay=-25)

    # Add blended candle annotation
    if n >= 2:
        pct_chg = (blended_close / blended_open - 1) * 100
        fig.add_annotation(
            x=0.02, y=0.98, xref='paper', yref='paper',
            text=(f"<b>Blended {min(5,n)} Candles:</b> {blended_signal}<br>"
                  f"O:{blended_open:.2f} H:{blended_high:.2f} "
                  f"L:{blended_low:.2f} C:{blended_close:.2f} "
                  f"({pct_chg:+.2f}%)"),
            showarrow=False, font=dict(size=10, color=blended_color),
            bgcolor='rgba(22,27,34,0.85)', bordercolor='#30363d',
            borderwidth=1, borderpad=6, align='left',
            xanchor='left', yanchor='top')

    fig.update_layout(
        title=f'{ticker.split(" ")[0]} — Candlestick (Last {lookback_days}D) + Padrões',
        template=DASH_TEMPLATE,
        height=400,
        margin=dict(l=50, r=30, t=45, b=30),
        xaxis_title='', yaxis_title='Price',
        xaxis_rangeslider_visible=False,
    )
    return go.FigureWidget(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5H — SKEW MONITOR + TAIL ANALYTICS + DEALER BOOK MC + OPEX
# ═══════════════════════════════════════════════════════════════════════════════

# ── Mag8 constituents (Mag7 + AVGO) ──
MAG8 = [
    'AAPL US Equity', 'MSFT US Equity', 'GOOGL US Equity', 'AMZN US Equity',
    'NVDA US Equity', 'META US Equity', 'TSLA US Equity', 'AVGO US Equity',
]

# ── Skew Monitor ─────────────────────────────────────────────────

def fetch_skew_metrics(ticker='SPX Index', lookback=756):
    """
    Busca métricas de skew via BQL: 25d put IV, 25d call IV, ATM IV.
    Calcula: Risk Reversal (25dP - 25dC), Put Skew (25dP/ATM), Call Skew (25dC/ATM).
    Lookback padrão: 756 dias (~3 anos) para percentis mais significativos.
    Retorna DataFrame com colunas: atm_iv, put25d_iv, call25d_iv, risk_reversal,
    put_skew, call_skew + percentis.
    """
    from scipy.stats import percentileofscore as _pctof

    bq = bql.Service()
    dt_range = bq.func.range('-{}d'.format(lookback), '0d')
    try:
        # Tentar buscar ATM + 25d put/call IV em queries separadas (mais robusto)
        req_atm = bql.Request(ticker, {
            'atm_iv': bq.data.implied_volatility(
                expiry='30D', pct_moneyness='100', fill='PREV', dates=dt_range),
        })
        resp_atm = bq.execute(req_atm)
        df = _bql_ts_df(resp_atm[0])
        # Tentar 25d put/call IV separadamente
        try:
            req_p = bql.Request(ticker, {
                'put25d': bq.data.implied_volatility(
                    expiry='30D', delta='25', put_call='PUT', fill='PREV', dates=dt_range),
            })
            df['put25d'] = _bql_ts(bq.execute(req_p)[0], 'put25d')
        except Exception:
            df['put25d'] = np.nan
        try:
            req_c = bql.Request(ticker, {
                'call25d': bq.data.implied_volatility(
                    expiry='30D', delta='25', put_call='CALL', fill='PREV', dates=dt_range),
            })
            df['call25d'] = _bql_ts(bq.execute(req_c)[0], 'call25d')
        except Exception:
            df['call25d'] = np.nan
        # Se ambos falharam, tentar SKEW Index como proxy
        if df['put25d'].isna().all() and df['call25d'].isna().all():
            try:
                req_skew = bql.Request('SKEW Index', {
                    'px': bq.data.px_last(fill='PREV', dates=dt_range),
                })
                skew_s = _bql_ts(bq.execute(req_skew)[0], 'px')
                # SKEW Index ~100=no skew, >100=more put skew
                # Mapear para risk_reversal proxy: (SKEW - 100) / 100 como fraction
                df['skew_index'] = skew_s
            except Exception:
                pass
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    df['risk_reversal'] = df['put25d'] - df['call25d']
    atm = df['atm_iv'].replace(0, np.nan)
    df['put_skew'] = df['put25d'] / atm
    df['call_skew'] = df['call25d'] / atm

    # Percentis: rank do último valor vs série histórica inteira (scipy percentileofscore)
    for col in ['atm_iv', 'put25d', 'call25d', 'risk_reversal', 'put_skew', 'call_skew']:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 20:
                last = vals.iloc[-1]
                pctile = float(_pctof(vals.values, last, kind='rank'))
                df['{}_pctile'.format(col)] = np.nan
                df.loc[df.index[-1], '{}_pctile'.format(col)] = pctile
    return df


def compute_skew_summary(skew_df):
    """Resumo das métricas de skew atuais + percentis."""
    if skew_df.empty:
        return {}
    last = skew_df.iloc[-1]
    summary = {}
    for col in ['atm_iv', 'put25d', 'call25d', 'risk_reversal', 'put_skew', 'call_skew']:
        if col in last.index and not np.isnan(last.get(col, np.nan)):
            summary[col] = round(float(last[col]), 2)
            pctile_col = '{}_pctile'.format(col)
            if pctile_col in last.index:
                summary[pctile_col] = round(float(last[pctile_col]), 0)
    return summary


def build_skew_chart(skew_df):
    """
    Gráfico 4-panel: Risk Reversal, ATM Vol, Call Skew, Put Skew.
    """
    if skew_df.empty or len(skew_df) < 10:
        fig = go.Figure()
        fig.add_annotation(text='Sem dados de skew', x=0.5, y=0.5,
                           xref='paper', yref='paper', showarrow=False,
                           font=dict(color='white', size=14))
        fig.update_layout(template='plotly_dark',
                          paper_bgcolor='#0d1117', plot_bgcolor='#161b22')
        return go.FigureWidget(fig)

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[
                            'Risk Reversal (25dP - 25dC)',
                            'ATM Implied Volatility',
                            'Call Skew (25dC / ATM)',
                            'Put Skew (25dP / ATM)',
                        ],
                        vertical_spacing=0.12, horizontal_spacing=0.08)

    def _winsor(s, n_sigma=4):
        """Remove outliers além de n_sigma desvios-padrão (dados ruins do BQL)."""
        if len(s) < 10:
            return s
        mu, sd = s.mean(), s.std()
        if sd == 0:
            return s
        return s.where((s - mu).abs() <= n_sigma * sd)

    if 'risk_reversal' in skew_df.columns:
        rr = _winsor(skew_df['risk_reversal'].dropna())
        fig.add_trace(go.Scatter(x=rr.index, y=rr, name='Risk Reversal',
                                 line=dict(color='#da3633', width=1.5)), row=1, col=1)

    if 'atm_iv' in skew_df.columns:
        atm = _winsor(skew_df['atm_iv'].dropna())
        fig.add_trace(go.Scatter(x=atm.index, y=atm, name='ATM IV',
                                 line=dict(color='#8b949e', width=1.5)), row=1, col=2)

    if 'call_skew' in skew_df.columns:
        cs = _winsor(skew_df['call_skew'].dropna())
        fig.add_trace(go.Scatter(x=cs.index, y=cs, name='Call Skew 25dC/ATM',
                                 line=dict(color='#3fb950', width=1.5)), row=2, col=1)

    if 'put_skew' in skew_df.columns:
        ps = _winsor(skew_df['put_skew'].dropna())
        fig.add_trace(go.Scatter(x=ps.index, y=ps, name='Put Skew 25dP/ATM',
                                 line=dict(color='#f0883e', width=1.5)), row=2, col=2)

    fig.update_layout(
        template='plotly_dark', height=480,
        paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9', size=11),
        showlegend=False,
        margin=dict(l=50, r=30, t=50, b=30),
    )
    return go.FigureWidget(fig)


# ── Spot-Up-Vol-Up Tracker ───────────────────────────────────────

def compute_spot_up_vol_up(log_returns, vix_changes):
    """
    Conta dias consecutivos onde Spot sobe E Vol sobe (raro, sinal de euforia).
    Retorna dict com: current_streak, max_streak, history (series de streaks),
    total_occurrences, pct_of_days.
    """
    n = min(len(log_returns), len(vix_changes))
    if n < 20:
        return {'current_streak': 0, 'max_streak': 0, 'history': pd.Series(dtype=int),
                'total_days': 0, 'pct_up_up': 0}

    spot_up = np.asarray(log_returns.iloc[-n:]) > 0
    vol_up = np.asarray(vix_changes.iloc[-n:]) > 0
    both_up = spot_up & vol_up

    streaks = []
    current = 0
    dates = log_returns.index[-n:]
    streak_dates = []
    streak_vals = []
    for i in range(n):
        if both_up[i]:
            current += 1
        else:
            if current > 0:
                streak_dates.append(dates[i - 1])
                streak_vals.append(current)
            current = 0
    if current > 0:
        streak_dates.append(dates[-1])
        streak_vals.append(current)

    current_streak = current
    max_streak = max(streak_vals) if streak_vals else 0
    total_up_up = int(both_up.sum())

    return {
        'current_streak': current_streak,
        'max_streak': max_streak,
        'history': pd.Series(streak_vals, index=streak_dates) if streak_dates else pd.Series(dtype=int),
        'total_days': total_up_up,
        'pct_up_up': round(total_up_up / n * 100, 1) if n > 0 else 0,
    }


def compute_vix_spx_regression(spx_returns, vix_changes, window_years=2):
    """
    Regressão VIX Move vs SPX Move (1M rolling).
    Retorna dict com: slope, intercept, r2, prediction, scatter_data.
    """
    n = min(len(spx_returns), len(vix_changes))
    if n < 30:
        return {}

    spx_1m = spx_returns.rolling(21).sum().dropna()
    vix_1m = vix_changes.rolling(21).sum().dropna()
    common = spx_1m.index.intersection(vix_1m.index)
    spx_1m = spx_1m.reindex(common).values * 100
    vix_1m = vix_1m.reindex(common).values

    window = min(window_years * 252, len(spx_1m))
    x = spx_1m[-window:]
    y = vix_1m[-window:]
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]

    if len(x) < 20:
        return {}

    mean_x, mean_y = np.mean(x), np.mean(y)
    cov_xy = np.mean((x - mean_x) * (y - mean_y))
    var_x = np.mean((x - mean_x) ** 2)
    slope = cov_xy / var_x if var_x > 1e-12 else 0
    intercept = mean_y - slope * mean_x
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - mean_y) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0

    last_spx_1m = float(x[-1]) if len(x) > 0 else 0
    predicted_vix = slope * last_spx_1m + intercept

    return {
        'slope': round(float(slope), 3),
        'intercept': round(float(intercept), 2),
        'r2': round(float(r2), 3),
        'predicted_vix_move': round(float(predicted_vix), 1),
        'last_spx_1m': round(float(last_spx_1m), 1),
        'x': x, 'y': y,
    }


# ── Enhanced Dealer Book + Per-Dealer Monte Carlo ────────────────

def run_dealer_monte_carlo(spot, df, risk_params, n_sims=10000, n_days=5, r=0.0):
    """
    Monte Carlo por dealer individual com dinâmica de vol.
    Simula evolução de spot + vol, recalcula Greeks em cada passo para
    capturar gamma hedging e convexity. Retorna dict por dealer.
    """
    oi100 = df.OI.values * 100
    call_sign = np.where(df.Type.values == 'Call', 1, -1)
    strikes = df.Strike.values
    base_iv = df.IV.values.copy()
    base_tte = df.Tte.values.copy()
    types_arr = df.Type.values

    day_rets_all = np.empty((n_days, n_sims), dtype=float)
    for d in range(n_days):
        day_rets_all[d] = student_t.rvs(
            risk_params['tdf'], loc=risk_params['tloc'],
            scale=risk_params['tscale'], size=n_sims)

    # Simulate total book P&L first (full path with Greeks recomputation)
    # For each path: recompute Greeks at each step
    total_cum_pnl = np.zeros(n_sims)
    # Batch: compute average path then scale by sims for speed
    # Use representative scenarios: compute Greeks at percentile spots
    pctiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    cum_spots = np.full(n_sims, spot)
    for d in range(n_days):
        new_spots = cum_spots * (1 + day_rets_all[d])
        ds = new_spots - cum_spots
        # Vol response: spot-vol correlation (empirical: ~-0.5 to -0.8 for SPX)
        spot_chg_pct = day_rets_all[d]
        vol_chg = -0.5 * spot_chg_pct  # ~50% inverse correlation
        sim_iv = np.clip(base_iv + vol_chg.mean(), 0.001, None)
        sim_tte = np.clip(base_tte - (d + 1) / TRADING_DAYS, 1.0 / TRADING_DAYS, None)

        # Recompute Greeks at current mean spot for better P&L estimation
        mean_spot = float(np.mean(new_spots))
        g = calculate_all_greeks(mean_spot, strikes, sim_iv, sim_tte, types_arr, r=r)

        total_dex = (g['delta'] * oi100).sum()
        total_gex = (g['gamma'] * call_sign * oi100).sum()
        total_theta = (g['theta'] * oi100).sum()
        total_vanna = (g['vanna'] * oi100).sum()
        vol_chg_total = float(np.mean(vol_chg)) * 100  # in vol pts

        daily_pnl = (-(total_dex * ds + 0.5 * total_gex * ds ** 2)
                     + total_theta
                     - total_vanna * vol_chg_total * mean_spot / 100)
        total_cum_pnl += daily_pnl
        cum_spots = new_spots

    # Per-dealer: scale from total
    greeks = calculate_all_greeks(spot, strikes, base_iv, base_tte, types_arr, r=r)
    total_dex0 = (greeks['delta'] * oi100).sum()
    total_gex0 = (greeks['gamma'] * call_sign * oi100).sum()
    total_theta0 = (greeks['theta'] * oi100).sum()
    total_vanna0 = (greeks['vanna'] * oi100).sum()

    results = {}
    total_book = {
        'dex': total_dex0, 'gex': total_gex0,
        'theta': total_theta0, 'vanna': total_vanna0,
    }

    all_dealers = list(MM_VOLUME_SHARES.items()) + [('TOTAL', 1.0)]
    for mm_name, share in all_dealers:
        if mm_name == 'TOTAL':
            cum_pnl = total_cum_pnl
        else:
            cum_pnl = total_cum_pnl * share

        results[mm_name] = {
            'share': share,
            'var_95': float(np.percentile(cum_pnl, 5)),
            'var_99': float(np.percentile(cum_pnl, 1)),
            'cvar_95': float(np.mean(cum_pnl[cum_pnl <= np.percentile(cum_pnl, 5)])),
            'cvar_99': float(np.mean(cum_pnl[cum_pnl <= np.percentile(cum_pnl, 1)])),
            'mean_pnl': float(np.mean(cum_pnl)),
            'median_pnl': float(np.median(cum_pnl)),
            'max_loss': float(np.min(cum_pnl)),
            'max_gain': float(np.max(cum_pnl)),
            'win_pct': float((cum_pnl > 0).mean() * 100),
            'mc_pnl': cum_pnl,
        }

    results['_book'] = total_book
    return results


def compute_dealer_scenario_matrix(spot, df, greeks_now):
    """
    Matriz de cenários: dealer buy/sell por nível de spot (estilo dos slides).
    Inclui SPX, QQQ proxies, Mag8, e Vol Control.
    Retorna DataFrame com cenários ±3% a ±20%.
    """
    oi100 = df.OI.values * 100
    call_sign = np.where(df.Type.values == 'Call', 1, -1)
    gex_per_pt = (greeks_now['gamma'] * call_sign * oi100).sum()
    dex = (greeks_now['delta'] * oi100).sum()
    vanna_not = np.nansum(greeks_now['vanna'] * oi100) * spot
    # Charm: daily $ delta decay — dealers need to unwind this delta overnight
    charm_daily = np.nansum(greeks_now['charm'] * oi100) * spot / 365.0

    moves = [0.20, 0.15, 0.10, 0.05, 0.03, -0.03, -0.05, -0.10, -0.15, -0.20]
    rows = []
    for m in moves:
        ds = spot * m
        dealer_flow = -(dex * ds + 0.5 * gex_per_pt * ds ** 2)
        vol_chg = max(0, -m) * 150
        vanna_flow = -vanna_not * vol_chg / 100.0
        charm_flow = -charm_daily  # dealers unwind decayed delta
        total = dealer_flow + vanna_flow + charm_flow
        rows.append({
            'Move': '{:+.0%}'.format(m),
            'SPX Level': round(spot * (1 + m), 0),
            'Dealer Gamma ($B)': round(dealer_flow / 1e9, 1),
            'Vanna Flow ($B)': round(vanna_flow / 1e9, 1),
            'Charm Flow ($B)': round(charm_flow / 1e9, 1),
            'Total ($B)': round(total / 1e9, 1),
        })
    return pd.DataFrame(rows)


def compute_mag8_dealer_scenarios(spot, df, greeks_now):
    """
    Projeta rebalance de dealers por Mag8 stock (estilo slides: dealer buy/sell).
    Assume que Mag8 representa ~35% do GEX total ponderado por market cap.
    """
    oi100 = df.OI.values * 100
    call_sign = np.where(df.Type.values == 'Call', 1, -1)
    total_gex = (greeks_now['gamma'] * call_sign * oi100).sum()

    mag8_weights = {
        'MSFT': 0.065, 'NVDA': 0.060, 'TSLA': 0.035, 'AAPL': 0.070,
        'META': 0.040, 'GOOG': 0.045, 'AMZN': 0.050, 'AVGO': 0.030,
    }
    moves = [0.15, 0.10, 0.05, 0.03, -0.03, -0.05, -0.10, -0.15]
    rows = []
    for m in moves:
        ds = spot * m
        row = {'Move': '{:+.0%}'.format(m)}
        total = 0.0
        for stock, wt in mag8_weights.items():
            stock_gex = total_gex * wt
            flow = -0.5 * stock_gex * ds ** 2
            flow_b = flow / 1e9
            row[stock] = round(flow_b, 1)
            total += flow_b
        row['Total'] = round(total, 1)
        rows.append(row)
    return pd.DataFrame(rows)


# ── OPEX Analysis ────────────────────────────────────────────────

def compute_opex_dates(year_start=2020, year_end=2026):
    """
    Gera datas de OPEX (3ª sexta de cada mês) + VIX expiration (30d antes SPX OPEX).
    Retorna DataFrame com opex_date, vix_exp_date, vix_before_opex (bool).
    """
    from datetime import timedelta
    opex_dates = []
    for y in range(year_start, year_end + 1):
        for mth in range(1, 13):
            first_day = datetime(y, mth, 1)
            dow = first_day.weekday()
            first_friday = first_day + timedelta(days=(4 - dow) % 7)
            third_friday = first_friday + timedelta(days=14)
            opex_dates.append(third_friday)

    rows = []
    for opex in opex_dates:
        vix_exp = opex - timedelta(days=30)
        vix_exp_dow = vix_exp.weekday()
        if vix_exp_dow == 5:
            vix_exp -= timedelta(days=1)
        elif vix_exp_dow == 6:
            vix_exp -= timedelta(days=2)
        rows.append({
            'opex_date': opex,
            'vix_exp_date': vix_exp,
            'vix_before_opex': vix_exp < opex,
        })
    return pd.DataFrame(rows)


def compute_opex_stats(log_returns, lookback_years=5):
    """
    Estatísticas de OPEX: return flip probability, RV impact.
    Inspirado nos frameworks de gamma exposure.
    """
    idx = pd.to_datetime(log_returns.index)
    if len(idx) < 252:
        return {}

    opex_df = compute_opex_dates(
        year_start=idx[0].year, year_end=idx[-1].year)

    week_before_returns = []
    week_after_returns = []
    flip_count = 0
    total_opex = 0

    rv_into_5d = []
    rv_out_5d = []
    rv_into_10d = []
    rv_out_10d = []

    for _, row in opex_df.iterrows():
        opex = row['opex_date']
        pos = idx.searchsorted(opex)
        if pos < 15 or pos >= len(idx) - 10:
            continue

        ret_before_5d = float(log_returns.iloc[pos - 5:pos].sum())
        ret_after_5d = float(log_returns.iloc[pos:pos + 5].sum())

        week_before_returns.append(ret_before_5d)
        week_after_returns.append(ret_after_5d)

        if (ret_before_5d > 0 and ret_after_5d < 0) or \
           (ret_before_5d < 0 and ret_after_5d > 0):
            flip_count += 1
        total_opex += 1

        rv5_into = float(log_returns.iloc[pos - 5:pos].std() * np.sqrt(252) * 100)
        rv5_out = float(log_returns.iloc[pos:pos + 5].std() * np.sqrt(252) * 100)
        rv_into_5d.append(rv5_into)
        rv_out_5d.append(rv5_out)

        if pos >= 10 and pos + 10 < len(idx):
            rv10_into = float(log_returns.iloc[pos - 10:pos].std() * np.sqrt(252) * 100)
            rv10_out = float(log_returns.iloc[pos:pos + 10].std() * np.sqrt(252) * 100)
            rv_into_10d.append(rv10_into)
            rv_out_10d.append(rv10_out)

    flip_pct = flip_count / total_opex * 100 if total_opex > 0 else 0

    return {
        'total_opex': total_opex,
        'flip_count': flip_count,
        'flip_pct': round(flip_pct, 1),
        'avg_ret_before': round(np.mean(week_before_returns) * 100, 2) if week_before_returns else 0,
        'avg_ret_after': round(np.mean(week_after_returns) * 100, 2) if week_after_returns else 0,
        'rv5_delta_into': round(np.mean(rv_into_5d), 1) if rv_into_5d else 0,
        'rv5_delta_out': round(np.mean(rv_out_5d), 1) if rv_out_5d else 0,
        'rv10_delta_into': round(np.mean(rv_into_10d), 1) if rv_into_10d else 0,
        'rv10_delta_out': round(np.mean(rv_out_10d), 1) if rv_out_10d else 0,
    }


# ── Gamma Index → Realized Vol Model ────────────────────────────

def compute_gamma_vol_relationship(gex_series, rv_series, window=21):
    """
    Modela relação Gamma Exposure → Realized Vol prevista.
    Quanto maior o GEX positivo, menor a vol.
    Retorna: slope, r2, predicted_vol, scatter (for plotting).
    """
    n = min(len(gex_series), len(rv_series))
    if n < 30:
        return {}
    gex = np.asarray(gex_series[-n:], dtype=float)
    rv = np.asarray(rv_series[-n:], dtype=float)
    valid = ~(np.isnan(gex) | np.isnan(rv))
    gex, rv = gex[valid], rv[valid]
    if len(gex) < 20:
        return {}
    mean_g, mean_r = np.mean(gex), np.mean(rv)
    cov = np.mean((gex - mean_g) * (rv - mean_r))
    var_g = np.mean((gex - mean_g) ** 2)
    slope = cov / var_g if var_g > 1e-12 else 0
    intercept = mean_r - slope * mean_g
    predicted = slope * gex[-1] + intercept
    y_pred = slope * gex + intercept
    ss_res = np.sum((rv - y_pred) ** 2)
    ss_tot = np.sum((rv - mean_r) ** 2)
    r2 = max(0, 1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0
    return {
        'slope': round(float(slope), 6),
        'intercept': round(float(intercept), 4),
        'r2': round(float(r2), 3),
        'predicted_rv': round(float(predicted) * 100, 1),
        'current_gex': float(gex[-1]),
        'gex': gex, 'rv': rv * 100,
    }


# ── Vol Control + Leveraged ETF Scenario Projections ──

def compute_vol_rebalance_projection(rv_current, spot, gex_per_pt=0,
                                      vanna_notional=0, dex=0):
    """
    Projeção de rebalanceamento combinado (Vol Control + Dealer + LevETF).
    Estilo dos slides: tabela por cenário ±3% a ±20%.
    """
    moves = [0.20, 0.15, 0.10, 0.05, 0.03, -0.03, -0.05, -0.10, -0.15, -0.20]
    rows = []
    for m in moves:
        ds = spot * m
        rv_shock = rv_current * (1 + max(0, -m) * 5)
        rv_shock = min(rv_shock, 0.80)

        vc_flow = 0
        for tv in [5, 10, 15]:
            tv_dec = tv / 100.0
            exp_cur = _vc_exposure(tv_dec, rv_current)
            exp_shock = _vc_exposure(tv_dec, rv_shock)
            vc_flow += VOL_CTRL_AUM.get(tv, 100e9) * (exp_shock - exp_cur)

        dealer_flow = -(dex * ds + 0.5 * gex_per_pt * ds ** 2) if gex_per_pt != 0 else 0

        lev_flow = 0
        for _lev, _aum in [('3x', 15e9), ('2x', 25e9), ('-3x', 8e9), ('-1x', 32e9)]:
            mult = 3 if '3x' in _lev else (2 if '2x' in _lev else (-3 if '-3x' in _lev else -1))
            rebal = _aum * mult * m
            if '-' in _lev:
                rebal = -rebal
            lev_flow += rebal

        total = vc_flow + dealer_flow + lev_flow
        rows.append({
            'Move': '{:+.0%}'.format(m),
            'Vol Ctrl ($B)': round(vc_flow / 1e9, 1),
            'Dealer ($B)': round(dealer_flow / 1e9, 1),
            'Lev ETF ($B)': round(lev_flow / 1e9, 1),
            'Total ($B)': round(total / 1e9, 1),
        })
    return pd.DataFrame(rows)


# ── Tail Risk Probabilistic Gauge ────────────────────────────────

def compute_tail_risk_gauge(log_returns, iv_30d=None, rv_30d=None,
                            skew_summary=None, spot_vol_up_streak=0):
    """
    Computa um score probabilístico de risco caudal (0-100).
    Combina múltiplos fatores:
    - Kurtosis excess (caudas pesadas)
    - Skew negativo (assimetria left-tail)
    - IV/RV ratio (fear premium)
    - Put skew level (proteção downside demandada)
    - Risk reversal magnitude
    - Spot-up-vol-up streak (raro, sinal de stress próximo)
    Retorna: score (0-100), components dict, interpretation string.
    """
    rets = np.asarray(log_returns, dtype=float)
    rets = rets[~np.isnan(rets)]
    if len(rets) < 50:
        return 50, {}, 'Dados insuficientes'

    std_r = np.std(rets)
    if std_r < 1e-10:
        return 50, {}, 'Vol zero'

    kurtosis = float(np.mean((rets - np.mean(rets)) ** 4) / std_r ** 4)
    skewness = float(np.mean((rets - np.mean(rets)) ** 3) / std_r ** 3)

    components = {}

    kurt_score = min(25, max(0, (kurtosis - 3) * 5))
    components['kurtosis'] = {'value': round(kurtosis, 2), 'score': round(kurt_score, 1),
                              'label': 'Excess Kurtosis'}

    skew_score = min(20, max(0, abs(min(0, skewness)) * 10))
    components['skewness'] = {'value': round(skewness, 3), 'score': round(skew_score, 1),
                              'label': 'Left Skew'}

    iv_rv_score = 0
    if iv_30d is not None and rv_30d is not None and rv_30d > 1e-6:
        ratio = iv_30d / rv_30d
        if ratio > 1.3:
            iv_rv_score = min(15, (ratio - 1.0) * 10)
        else:
            iv_rv_score = max(0, min(15, (1.3 - ratio) * 15))
        components['iv_rv_ratio'] = {'value': round(ratio, 2), 'score': round(iv_rv_score, 1),
                                     'label': 'IV/RV Ratio'}

    put_skew_score = 0
    rr_score = 0
    if skew_summary:
        ps = skew_summary.get('put_skew', 1.0)
        if ps > 1.15:
            put_skew_score = min(15, (ps - 1.0) * 50)
        components['put_skew'] = {'value': round(ps, 3), 'score': round(put_skew_score, 1),
                                  'label': 'Put Skew 25d/ATM'}

        rr = skew_summary.get('risk_reversal', 0)
        rr_score = min(10, abs(rr) * 1.5)
        components['risk_reversal'] = {'value': round(rr, 2), 'score': round(rr_score, 1),
                                       'label': 'Risk Reversal'}

    suvu_score = min(15, spot_vol_up_streak * 3)
    components['spot_vol_up'] = {'value': spot_vol_up_streak, 'score': round(suvu_score, 1),
                                 'label': 'Spot Up Vol Up Streak'}

    total = kurt_score + skew_score + iv_rv_score + put_skew_score + rr_score + suvu_score
    total = min(100, max(0, total))

    if total >= 75:
        interp = 'EXTREMO — Proteção caudal recomendada'
    elif total >= 50:
        interp = 'ELEVADO — Monitorar sinais de stress'
    elif total >= 25:
        interp = 'MODERADO — Complacência relativa'
    else:
        interp = 'BAIXO — Ambiente de baixo risco caudal'

    return round(total, 1), components, interp


def build_gamma_levels_chart(prices, spot, call_wall, put_wall, gamma_flip,
                              iv_30d, lookback=30):
    """
    Gráfico diário de preço (últimos N dias) com linhas horizontais dos
    níveis de gamma: Call Wall, Put Wall, Vol Trigger, Est Move 1d/5d.
    Estilo similar ao TradingView com linhas de referência coloridas.
    """
    px_tail = prices.iloc[-lookback:] if len(prices) >= lookback else prices
    dates = px_tail.index
    vals  = px_tail.values

    fig = go.Figure()

    # Linha de preço
    fig.add_trace(go.Scatter(
        x=dates, y=vals, mode='lines', name='SPX',
        line=dict(color='#c9d1d9', width=1.5)))

    # Ponto atual
    fig.add_trace(go.Scatter(
        x=[dates[-1]], y=[spot], mode='markers', name=f'Spot {spot:,.0f}',
        marker=dict(color='#f0883e', size=8, symbol='circle')))

    # Níveis de gamma — linhas horizontais
    _levels = []
    if call_wall:
        _levels.append(('Call Wall', call_wall, _C['green'], 'dash'))
    if put_wall:
        _levels.append(('Put Wall', put_wall, _C['red'], 'dash'))
    if gamma_flip:
        _levels.append(('Vol Trigger / Zero Gamma', gamma_flip, _C['yellow'], 'dot'))

    # Est Move 1d e 5d
    if pd.notna(iv_30d) and iv_30d > 0:
        move_1d = spot * iv_30d * math.sqrt(1 / TRADING_DAYS)
        move_5d = spot * iv_30d * math.sqrt(5 / TRADING_DAYS)
        _levels.append((f'1D Est Move + ({move_1d:+.0f})', spot + move_1d, '#58a6ff', 'dashdot'))
        _levels.append((f'1D Est Move - ({-move_1d:+.0f})', spot - move_1d, '#da3633', 'dashdot'))
        _levels.append((f'5D Est Move - ({-move_5d:+.0f})', spot - move_5d, '#f85149', 'dot'))
        _levels.append((f'5D Est Move + ({move_5d:+.0f})', spot + move_5d, '#3fb950', 'dot'))

    for name, level, color, dash in _levels:
        fig.add_hline(y=level, line=dict(color=color, dash=dash, width=1.2),
                      annotation_text=name,
                      annotation_font=dict(color=color, size=10),
                      annotation_position='right')

    fig.update_layout(
        title=f'Níveis de Gamma — SPX (últimos {lookback} dias)',
        template=DASH_TEMPLATE,
        height=360,
        margin=dict(l=50, r=140, t=40, b=30),
        xaxis=dict(showgrid=False),
        yaxis_title='SPX Level',
        legend=dict(orientation='h', yanchor='bottom', y=-0.25,
                    xanchor='center', x=0.5),
        showlegend=True,
    )
    return go.FigureWidget(fig)


def build_tail_gauge(score, interpretation):
    """Cria gauge widget para tail risk score."""
    color = '#3fb950' if score < 25 else '#d29922' if score < 50 else '#f0883e' if score < 75 else '#da3633'
    return create_gauge(
        score, 'Tail Risk Score', 0, 100, color, '',
        steps=[
            {'range': [0, 25], 'color': '#1a3a2a'},
            {'range': [25, 50], 'color': '#2a2a1a'},
            {'range': [50, 75], 'color': '#3a2a1a'},
            {'range': [75, 100], 'color': '#3a1a1a'},
        ], width=280, height=220)


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — MATRIZES DE SENSIBILIDADE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sensitivity_matrices(df, spot, r=0.0):
    """
    Calcula matrizes de sensibilidade (preço × vol shift) para cada grega.
    Retorna dict: {greek_name: pd.DataFrame}.
    """
    spot_range = np.linspace(spot * 0.97, spot * 1.03, 7)
    vol_shifts = np.linspace(-0.03, 0.03, 5)

    cols = [f"{s:,.0f}" for s in spot_range]
    idx = [f"{vs:+.1%}" for vs in vol_shifts]

    greek_keys = ['delta', 'gamma', 'vega', 'vanna', 'theta', 'charm', 'zomma', 'speed']
    matrices = {k: pd.DataFrame(np.nan, index=idx, columns=cols, dtype=float) for k in greek_keys}

    strikes = df['Strike'].values
    base_ivs = df['IV'].values
    ttes = df['Tte'].values
    types = df['Type'].values
    ois = df['OI'].values

    for i, iv_shift in enumerate(vol_shifts):
        shifted_ivs = np.clip(base_ivs + iv_shift, 0.001, None)
        for j, s in enumerate(spot_range):
            greeks = calculate_all_greeks(s, strikes, shifted_ivs, ttes, types, r=r)
            oi_100 = ois * 100.0
            is_call = types == 'Call'

            # Todas as matrizes em $ Mn (÷1e6) para escala comparável
            matrices['delta'].iloc[i, j] = np.nansum(greeks['delta'] * oi_100 * s) / 1e6
            matrices['gamma'].iloc[i, j] = np.nansum(
                greeks['gamma'] * np.where(is_call, 1, -1) * oi_100 * (s**2) * 0.01) / 1e6
            matrices['vega'].iloc[i, j] = np.nansum(greeks['vega'] * oi_100) / 1e6
            matrices['vanna'].iloc[i, j] = np.nansum(greeks['vanna'] * oi_100 * s) / 1e6
            matrices['theta'].iloc[i, j] = np.nansum(greeks['theta'] * oi_100) / 1e6
            matrices['charm'].iloc[i, j] = np.nansum(greeks['charm'] * oi_100 * s / 365.0) / 1e6
            matrices['zomma'].iloc[i, j] = np.nansum(greeks['zomma'] * oi_100 * (s**2) * 0.01) / 1e6
            matrices['speed'].iloc[i, j] = np.nansum(greeks['speed'] * oi_100 * s) / 1e6

    for k in matrices:
        matrices[k].index.name = 'Vol Shift'

    return matrices


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 6B — MODELO GAMMA SQUEEZE / SHORT SQUEEZE
# ═══════════════════════════════════════════════════════════════════════════════

# Eventos históricos que desencadearam squeezes (mercado pessimista → rali abrupto)
GAMMA_SQUEEZE_EVENTS = [
    {'date': '2020-03-23', 'label': 'COVID Bottom', 'type': 'bottom',
     'desc': 'Mínima do COVID (-34% em 33 dias). GEX profundamente negativo. '
             'Reversão de 70% em 5 meses.'},
    {'date': '2021-01-27', 'label': 'GME Squeeze', 'type': 'squeeze',
     'desc': 'Short squeeze massivo (GME +1.700%). Contágio para SPX: '
             'dealers cobertos em calls OTM forçaram delta-hedge buy.'},
    {'date': '2022-10-13', 'label': 'CPI Reversal Oct22', 'type': 'reversal',
     'desc': 'CPI acima do esperado → SPX abriu -3%, fechou +2.6% (+5.6% intraday). '
             'GEX fortemente negativo amplificou o rali pós-cobertura de puts.'},
    {'date': '2023-03-13', 'label': 'SVB Crisis', 'type': 'bottom',
     'desc': 'Crise SVB/Signature. SPX caiu 5% em 3 dias. GEX negativo. '
             'Reversão de 9% em 2 semanas após Fed backstop.'},
    {'date': '2018-02-05', 'label': 'VolPocalypse', 'type': 'vix_spike',
     'desc': 'XIV implodiu, VIX saltou de 13 para 37. GEX negativo forçou '
             'dealers a vender → queda acelerada de 12% em 10 dias.'},
    {'date': '2024-08-05', 'label': 'Yen Carry Unwind', 'type': 'squeeze',
     'desc': 'Unwind do carry trade iene. VIX atingiu 65 intraday. SPX caiu 3% '
             'e recuperou 90% em 10 pregões. GEX extremamente negativo.'},
]


def compute_gamma_squeeze_score(net_gex_bn, pc_ratio, iv_30d, rv_30d, gamma_flip,
                                spot, skew, put_wall, call_wall,
                                spxsk3_current=None, spxsk3_hist=None,
                                pc_hist=None, vix9d=None,
                                vvix=None, vix_skew_c25=None, vix_skew_p25=None,
                                sdex_cur=None, tdex_cur=None,
                                vix_call_oi=None, vix_put_oi=None):
    """
    Calcula o Gamma Squeeze Score (0–100).

    Score alto = condições favoráveis para um short squeeze induzido por gamma:
    - Dealers SHORT gamma (GEX muito negativo) → compram quando mercado sobe
    - Posicionamento bearish excessivo (P/C ratio alto, skew alto)
    - Mercado próximo ao gamma flip → qualquer alta cruza o flip e acelera
    - Vol realizada bem abaixo da implícita (mais espaço para rali surpresa)
    - Tail Skew (SPXSK3 / ATM_IV*10) < 0.40 → tail extremo confirmado

    Retorna dict com score total, sub-scores, interpretação e nível de alerta.
    """
    components = {}

    # Sanitize inputs que podem chegar como None
    gamma_flip = gamma_flip if (gamma_flip is not None and not (isinstance(gamma_flip, float) and np.isnan(gamma_flip))) else None
    iv_30d     = iv_30d  if pd.notna(iv_30d)  else 0.20
    rv_30d     = rv_30d  if pd.notna(rv_30d)  else 0.15
    skew       = skew    if pd.notna(skew)    else 0.0

    # 1. GEX negativity (0–30): quanto mais negativo, maior o score
    _gex_score = min(30, max(0, abs(min(0, net_gex_bn)) / 20.0 * 30))
    components['gex'] = {
        'label': 'GEX Negatividade',
        'value': f'{net_gex_bn:+.2f}B',
        'score': round(_gex_score, 1),
        'max': 30,
        'desc': 'Dealers SHORT gamma → amplificam movimentos de alta',
    }

    # 2. Put/Call ratio (0–25): usa percentil histórico quando disponível
    if pc_hist is not None and len(pc_hist) > 20:
        _pc_pct = float(np.mean(np.asarray(pc_hist) < pc_ratio))  # fração abaixo do atual
        _pc_score = min(25, _pc_pct * 25)
        _pc_hist_label = f' (p{_pc_pct*100:.0f}h)'
    else:
        # faixa ampliada: começa a pontuar de 0.75 (não só >1.0)
        _pc_score = min(25, max(0, (pc_ratio - 0.75) / 1.75 * 25))
        _pc_hist_label = ''
    components['pc_ratio'] = {
        'label': 'P/C OI Ratio',
        'value': f'{pc_ratio:.2f}x{_pc_hist_label}',
        'score': round(_pc_score, 1),
        'max': 25,
        'desc': 'Posicionamento pessimista → squeeze mais intenso se mercado subir',
    }

    # 3. Proximidade ao gamma flip (0–25)
    _flip_dist = abs(gamma_flip - spot) / spot if (gamma_flip and spot) else 0.05
    _flip_score = min(25, max(0, (1 - _flip_dist / 0.05) * 25))
    _flip_side = 'ACIMA' if (gamma_flip and gamma_flip > spot) else 'ABAIXO'
    components['flip_proximity'] = {
        'label': 'Distância Gamma Flip',
        'value': f'{gamma_flip:,.0f} ({_flip_dist:.1%} do spot) — Flip {_flip_side}' if gamma_flip else f'N/A ({_flip_dist:.1%} do spot)',
        'score': round(_flip_score, 1),
        'max': 25,
        'desc': 'Próximo ao flip → qualquer rali pode cruzar e virar auto-reforçado',
    }

    # 4. Vol premium IV−RV (0–20)
    _vol_gap = (iv_30d - rv_30d) if (pd.notna(iv_30d) and pd.notna(rv_30d)) else 0
    _vol_score = min(20, max(0, _vol_gap * 100 * 4))
    components['vol_premium'] = {
        'label': 'Prêmio de Vol (IV−RV)',
        'value': f'{_vol_gap*100:+.1f} vol pts',
        'score': round(_vol_score, 1),
        'max': 20,
        'desc': 'IV > RV = medo embutido → mais proteção para ser revertida',
    }

    # 5. Tail Skew — SPXSK3 / (ATM_IV × 10) (0–20)
    #    Fórmula BBG: .SPXSK3 G ÷ (SPX 30D IVOL at 100 × 0.1)
    #    <0.40 = TAIL EXTREMO (Volmageddon / COVID / End-2022)
    #    >0.65 = tail normal
    _sk3_ratio = None
    _sk3_score = 0.0
    if spxsk3_current is not None and iv_30d and iv_30d > 0:
        _sk3_ratio = spxsk3_current / (iv_30d * 10)   # mesmo cálculo do BBG chart
        # score baseado no nível absoluto
        _sk3_level_score = min(20, max(0, (0.65 - _sk3_ratio) / (0.65 - 0.40) * 20))
        # percentil histórico reforça (quanto mais raro → mais score)
        if spxsk3_hist is not None and len(spxsk3_hist) > 20:
            _sk3_arr = np.asarray(spxsk3_hist)
            _sk3_hist_ratios = _sk3_arr / (iv_30d * 10)
            _hist_pct = float(np.mean(_sk3_hist_ratios > _sk3_ratio))  # fração acima do atual
            _sk3_score = max(_sk3_level_score, _hist_pct * 20)
        else:
            _sk3_score = _sk3_level_score
        # confirmação VIX9D
        _vix9d_str = ''
        if vix9d is not None and vix9d > 20 and _sk3_ratio < 0.55:
            _vix9d_str = f' | VIX9D={vix9d:.1f} ✓'
        _sk3_label = ('⚠ EXTREMO' if _sk3_ratio < 0.40 else
                      'ELEVADO'   if _sk3_ratio < 0.55 else 'normal')
        components['tail_skew'] = {
            'label': 'Tail Skew (SPXSK3/IV)',
            'value': f'{_sk3_ratio:.3f} — {_sk3_label}{_vix9d_str}',
            'score': round(_sk3_score, 1),
            'max': 20,
            'desc': 'Ratio <0.40 = tail extremo → squeeze/vol spike iminente (Volmageddon, COVID)',
        }

    # 6. VVIX — vol da vol (0–15)
    if vvix is not None:
        _vvix_score = min(15, max(0, (vvix - 80) / (150 - 80) * 15))
        _vvix_level = ('EXTREMO ⚠' if vvix > 130 else
                       'ELEVADO'   if vvix > 100 else 'normal')
        components['vvix'] = {
            'label': 'VVIX (Vol da Vol)',
            'value': f'{vvix:.1f} — {_vvix_level}',
            'score': round(_vvix_score, 1),
            'max': 15,
            'desc': '>130 = panic vol-of-vol → squeeze/spike iminente',
        }

    total_raw = sum(c['score'] for c in components.values())
    total = min(100, round(total_raw, 1))  # cap em 100

    # Interpretação
    if total >= 75:
        interp = 'RISCO MUITO ALTO de Gamma Squeeze — condições extremas'
        alert = 'critical'
    elif total >= 55:
        interp = 'RISCO ELEVADO — monitorar flip + call OTM'
        alert = 'warning'
    elif total >= 35:
        interp = 'Risco moderado — mercado sensível a surpresas positivas'
        alert = 'moderate'
    else:
        interp = 'Risco baixo — posicionamento não sugere squeeze iminente'
        alert = 'low'

    _squeeze_mag = abs(net_gex_bn) * 0.15 * (pc_ratio / 1.5)
    _squeeze_mag_pct = min(15, _squeeze_mag / (spot * 0.01))

    return {
        'score': total,
        'alert': alert,
        'interp': interp,
        'components': components,
        'squeeze_mag_pct': round(_squeeze_mag_pct, 1),
        'flip_above': gamma_flip is not None and gamma_flip > spot,
        'flip_dist_pct': round(_flip_dist * 100, 2),
        'sk3_ratio': round(_sk3_ratio, 4) if _sk3_ratio is not None else None,
        'vix9d': vix9d,
        # vol-of-vol pass-through
        'vvix': vvix,
        'vix_skew_c25': vix_skew_c25,
        'vix_skew_p25': vix_skew_p25,
        'sdex_cur': sdex_cur,
        'tdex_cur': tdex_cur,
        'vix_call_oi': vix_call_oi,
        'vix_put_oi': vix_put_oi,
    }


def build_vol_smile_chart(df_orig, spot, ticker=''):
    """
    Gráfico interativo de vol smile por expiry — estilo SpotGamma.
    X = Strike, Y = IV%
    Banda = intervalo entre IV de puts e IV de calls no mesmo strike.
    Linha vertical = spot atual.
    Dropdown para selecionar o vencimento.
    """
    import plotly.graph_objects as go
    from IPython.display import display as _disp

    expiries = sorted(df_orig['Exp'].dt.normalize().unique())

    out_smile = wd.Output()

    w_exp = wd.Dropdown(
        options=[(pd.Timestamp(e).strftime('%d/%b/%Y — %d DTE' if True else ''),
                  pd.Timestamp(e))
                 for e in expiries],
        description='Vencimento:',
        style={'description_width': '100px'},
        layout=wd.Layout(width='280px'))

    # Build option list properly
    today = pd.Timestamp('today').normalize()
    w_exp.options = [
        (f"{pd.Timestamp(e).strftime('%d/%b/%Y')}  ({(pd.Timestamp(e)-today).days}d)",
         pd.Timestamp(e))
        for e in expiries
    ]
    # default: nearest expiry
    if expiries:
        w_exp.value = pd.Timestamp(expiries[0])

    def _draw_smile(_):
        sel_exp = w_exp.value
        sub = df_orig[df_orig['Exp'].dt.normalize() == sel_exp.normalize()].copy()
        if sub.empty:
            with out_smile:
                out_smile.clear_output(wait=True)
                _disp(wd.HTML("<p style='color:#f85149;'>Sem dados para este vencimento.</p>"))
            return

        calls = sub[sub['Type'] == 'Call'].sort_values('Strike')
        puts  = sub[sub['Type'] == 'Put' ].sort_values('Strike')

        # Pivot: para cada strike, obter call IV e put IV
        c_iv = calls.set_index('Strike')['IV'] * 100
        p_iv = puts.set_index('Strike')['IV']  * 100
        all_k = sorted(set(c_iv.index) | set(p_iv.index))

        c_vals = [c_iv.get(k, np.nan) for k in all_k]
        p_vals = [p_iv.get(k, np.nan) for k in all_k]

        # Band = fill between put IV e call IV (mesma estrutura do SpotGamma)
        # Linha mid = média dos dois onde ambos existem
        mid_vals = [np.nanmean([c, p]) for c, p in zip(c_vals, p_vals)]
        band_lo  = [min(c, p) if not (np.isnan(c) or np.isnan(p)) else np.nan
                    for c, p in zip(c_vals, p_vals)]
        band_hi  = [max(c, p) if not (np.isnan(c) or np.isnan(p)) else np.nan
                    for c, p in zip(c_vals, p_vals)]

        dte = (sel_exp - today).days
        fig = go.Figure()

        # Banda (fill)
        fig.add_trace(go.Scatter(
            x=all_k + all_k[::-1],
            y=band_hi + band_lo[::-1],
            fill='toself',
            fillcolor='rgba(0,180,160,.22)',
            line=dict(color='rgba(0,0,0,0)'),
            hoverinfo='skip',
            showlegend=False,
            name='Banda Put/Call'))

        # Linha mid IV
        fig.add_trace(go.Scatter(
            x=all_k, y=mid_vals,
            mode='lines',
            line=dict(color='rgba(0,212,232,.9)', width=2),
            name='Mid IV',
            hovertemplate='K=%{x}<br>IV=%{y:.2f}%<extra></extra>'))

        # Calls pontilhado
        fig.add_trace(go.Scatter(
            x=list(c_iv.index), y=list(c_iv.values),
            mode='lines',
            line=dict(color='rgba(0,212,232,.45)', width=1, dash='dot'),
            name='Call IV',
            hovertemplate='K=%{x}<br>Call IV=%{y:.2f}%<extra></extra>'))

        # Puts pontilhado
        fig.add_trace(go.Scatter(
            x=list(p_iv.index), y=list(p_iv.values),
            mode='lines',
            line=dict(color='rgba(248,81,73,.45)', width=1, dash='dot'),
            name='Put IV',
            hovertemplate='K=%{x}<br>Put IV=%{y:.2f}%<extra></extra>'))

        # Linha vertical — spot atual
        iv_at_spot = np.interp(spot, all_k,
                               [v if not np.isnan(v) else 0 for v in mid_vals])
        fig.add_vline(x=spot,
                      line=dict(color='rgba(0,212,232,.6)', width=1.5, dash='dash'),
                      annotation_text=f'Spot ${spot:,.2f}',
                      annotation_font=dict(color='rgba(0,212,232,.8)', size=10),
                      annotation_position='top right')

        # Layout escuro igual ao resto do dashboard
        fig.update_layout(
            title=dict(
                text=f'<b>Vol Smile — {sel_exp.strftime("%d/%b/%Y")}  ({dte} DTE) &nbsp;|&nbsp; {ticker}</b>',
                font=dict(color='rgba(0,212,232,.85)', size=13),
                x=0.5, xanchor='center'),
            paper_bgcolor='rgba(12,15,20,1)',
            plot_bgcolor='rgba(12,15,20,1)',
            font=dict(family="'Courier New',monospace", color='rgba(200,200,200,.7)', size=10),
            xaxis=dict(
                title='Strike', showgrid=True,
                gridcolor='rgba(255,255,255,.05)',
                zerolinecolor='rgba(255,255,255,.08)',
                tickformat=',d'),
            yaxis=dict(
                title='Implied Vol (%)', showgrid=True,
                gridcolor='rgba(255,255,255,.05)',
                ticksuffix='%'),
            legend=dict(
                bgcolor='rgba(0,0,0,0)', font=dict(size=10),
                orientation='h', y=-0.15),
            margin=dict(t=50, b=60, l=60, r=20),
            height=520,
        )

        with out_smile:
            out_smile.clear_output(wait=True)
            _disp(fig)

    w_exp.observe(lambda c: _draw_smile(None) if c['name'] == 'value' else None)
    _draw_smile(None)

    return wd.VBox([
        wd.HTML("<p style='color:rgba(0,212,232,.7);font-size:11px;margin:12px 0 4px;"
                "letter-spacing:.5px;'>VOL SMILE — IV% por strike e vencimento</p>"),
        w_exp,
        out_smile,
    ])


def build_dynamic_book_tab(df_orig, spot, rfr, ticker='', dealer_aum_bn=0.0):
    """
    Aba Ajuste Dinâmico do Book.

    Lógica (3 níveis):
      1. Repricing por instrumento via BS após choque de mercado
      2. Agregação por vencimento (expiry bucket)
      3. Consolidado total do book

    Inputs de cenário: ΔSpot, ΔVol (pp), ΔRate (bp), Dias à frente.
    Hedge Adj (por expiry) = −(Δpos_after − Δpos_before)
    onde Δpos = delta × OI × 100.

    Limitações:
      - Vol surface: choque uniforme (flat shift). Sem smile/skew diferencial.
      - Modelo: Black-Scholes europeu (mesmo modelo do dashboard).
      - OI usado como posição proxy — não há quantity/multiplier separado.
      - Opções com Tte < 1 dia após dt são tratadas como expiradas (valor intrínseco).
    """
    from IPython.display import display as _disp

    _TBL_CSS = (
        "border-collapse:collapse;width:100%;font-family:'Courier New',monospace;"
        "font-size:11px;"
    )
    _TH_CSS  = (
        "padding:3px 10px;text-align:right;border-bottom:1px solid rgba(0,212,232,.2);"
        "color:rgba(0,212,232,.65);white-space:nowrap;"
    )
    _TD_CSS  = "padding:2px 10px;text-align:right;white-space:nowrap;"
    _ROW_CSS = "border-bottom:1px solid rgba(255,255,255,.04);"

    def _html_table(dff, num_cols=None):
        """Render DataFrame as styled HTML table. num_cols: set of col names to color by sign."""
        num_cols = num_cols or set()
        hdrs = ''.join(f"<th style='{_TH_CSS}'>{c}</th>" for c in dff.columns)
        rows_html = ''
        for _, row in dff.iterrows():
            cells = ''
            for c in dff.columns:
                v = row[c]
                if c in num_cols and isinstance(v, float):
                    color = ('rgba(0,212,232,.95)' if v > 0
                             else ('rgba(248,81,73,.95)' if v < 0 else 'rgba(255,255,255,.4)'))
                    txt = f'{v:+,.2f}' if abs(v) < 10000 else f'{v:+,.0f}'
                else:
                    color = 'rgba(220,220,220,.85)'
                    txt = (f'{v:,.0f}' if isinstance(v, float) and abs(v) >= 100
                           else (f'{v:.4f}' if isinstance(v, float) else str(v)))
                cells += f"<td style='{_TD_CSS}color:{color};'>{txt}</td>"
            rows_html += f"<tr style='{_ROW_CSS}'>{cells}</tr>"
        return (f"<div style='overflow-x:auto;'>"
                f"<table style='{_TBL_CSS}'>"
                f"<thead><tr>{hdrs}</tr></thead>"
                f"<tbody>{rows_html}</tbody></table></div>")

    def _card(label, value, color='rgba(0,212,232,.9)', sub=''):
        _sub_p = ('<p style="color:rgba(255,255,255,.35);font-size:9px;margin:2px 0 0;">'
                  + sub + '</p>') if sub else ''
        return (f"<div style='background:rgba(0,212,232,.06);"
                f"border:1px solid rgba(0,212,232,.18);border-radius:6px;"
                f"padding:10px 18px;min-width:150px;'>"
                f"<p style='color:rgba(0,212,232,.55);font-size:9px;"
                f"letter-spacing:1.2px;margin:0;'>{label}</p>"
                f"<p style='color:{color};font-size:17px;font-weight:700;"
                f"margin:4px 0 0;font-family:monospace;'>{value}</p>"
                f"{_sub_p}"
                f"</div>")

    # ── Widgets de cenário ────────────────────────────────────────────────────
    # Vol surface: choque independente por asa de moneyness (K/S)
    #   Put wing : 50%–100%  (K/S in [0.50, 1.00))
    #   Call wing: 100%–150% (K/S in [1.00, 1.50])
    #   Fora do range: sem choque de vol
    _lyt  = wd.Layout(width='210px')
    _lytw = wd.Layout(width='225px')
    _sty  = {'description_width': '120px'}
    w_dspot    = wd.FloatText(value=0.0, description='ΔSpot ($):',        layout=_lyt,  style=_sty)
    w_dvol_put = wd.FloatText(value=0.0, description='ΔVol Put (pp):',    layout=_lytw, style=_sty,
                              tooltip='Asa de put: moneyness 50%–100% (strikes abaixo do spot)')
    w_dvol_call= wd.FloatText(value=0.0, description='ΔVol Call (pp):',   layout=_lytw, style=_sty,
                              tooltip='Asa de call: moneyness 100%–150% (strikes acima do spot)')
    w_drate    = wd.FloatText(value=0.0, description='ΔRate (bp):',        layout=_lyt,  style=_sty)
    w_days     = wd.IntText( value=0,    description='Dias à frente:',     layout=_lyt,  style=_sty)
    # AUM dos dealers: soma do book — escala o tamanho real da posição
    # Pré-preenchido com delta_bn total da cadeia (build_greek_overview)
    # scale = dealer_aum_$ / (total_OI × 100 × spot) → ajuste por instrumento
    _mkt_notional_bn = float(df_orig['OI'].sum() * 100 * spot / 1e9)
    w_aum  = wd.FloatText(
        value=round(abs(dealer_aum_bn), 2) if dealer_aum_bn else round(_mkt_notional_bn, 2),
        description='AUM Dealers ($B):',
        layout=wd.Layout(width='230px'), style={'description_width': '130px'},
        tooltip='Soma do book dos dealers em $B. Escala o tamanho de posição '
                'proporcional ao OI. 0 = usa OI×100 bruto (mercado total).')
    w_btn  = wd.Button(description='▶ Aplicar', button_style='primary',
                       layout=wd.Layout(width='120px', height='34px', margin='2px 0 0 0'))
    w_reset= wd.Button(description='↺ Reset',  button_style='',
                       layout=wd.Layout(width='90px',  height='34px', margin='2px 0 0 0'))

    out_cards       = wd.Output()
    out_agg         = wd.Output()
    out_inst        = wd.Output()
    out_sensitivity = wd.Output()

    def _compute_and_render(_):
        S_new        = spot + w_dspot.value
        dvol_put_dec = w_dvol_put.value  / 100.0   # pp → decimal, asa de put
        dvol_call_dec= w_dvol_call.value / 100.0   # pp → decimal, asa de call
        r_new        = rfr + w_drate.value / 10000.0  # bp → decimal
        dt_yr        = max(int(w_days.value), 0) / float(TRADING_DAYS)

        df    = df_orig.copy()
        mness = df['Strike'].values / spot   # moneyness K/S

        # Vol shift por asa de moneyness:
        #   Put  wing: K/S in [0.50, 1.00)  → dvol_put
        #   Call wing: K/S in [1.00, 1.50]  → dvol_call
        #   Fora do range (deep OTM além de 50%/150%): sem choque
        dvol_arr = np.where(
            (mness >= 0.50) & (mness < 1.00), dvol_put_dec,
            np.where((mness >= 1.00) & (mness <= 1.50), dvol_call_dec, 0.0)
        )

        T_after  = np.maximum(df['Tte'].values - dt_yr, 0.0)

        # Sticky-delta: quando spot muda, cada strike usa a IV da NOVA moneyness K/S_new
        # interpolada do smile original — sem isso o choque de spot ignora o skew.
        if S_new != spot:
            iv_sd = df['IV'].values.copy()
            for _exp in df['Exp'].unique():
                _m = df['Exp'].values == _exp
                _k   = df.loc[_m, 'Strike'].values
                _iv  = df.loc[_m, 'IV'].values
                _mn_orig = _k / spot
                _mn_new  = _k / S_new
                _ord = np.argsort(_mn_orig)
                iv_sd[_m] = np.interp(_mn_new,
                                      _mn_orig[_ord], _iv[_ord],
                                      left=_iv[_ord[0]], right=_iv[_ord[-1]])
            vol_aft = np.maximum(iv_sd + dvol_arr, 0.001)
        else:
            vol_aft = np.maximum(df['IV'].values + dvol_arr, 0.001)

        # ── Posição por instrumento ───────────────────────────────────────────
        # Se AUM dos dealers fornecido: escala oi100 para refletir o book real.
        # Distribuição proporcional ao OI (preserva a estrutura do livro).
        # scale = dealer_aum_$ / (total_OI × 100 × spot)
        _oi_raw  = df['OI'].values * 100.0
        _aum_val = w_aum.value
        if _aum_val > 0:
            _mkt_total = float(_oi_raw.sum() * spot)
            _scale     = (_aum_val * 1e9) / _mkt_total if _mkt_total > 0 else 1.0
            oi100      = _oi_raw * _scale
        else:
            oi100      = _oi_raw

        # ── Gregas e preços: base e cenário ──────────────────────────────────
        g_b  = calculate_all_greeks(spot, df['Strike'].values, df['IV'].values,
                                    df['Tte'].values, df['Type'].values, r=rfr)
        px_b = black_scholes_price_vec(spot, df['Strike'].values, df['IV'].values,
                                       df['Tte'].values, df['Type'].values, r=rfr)

        g_a  = calculate_all_greeks(S_new, df['Strike'].values, vol_aft,
                                    T_after, df['Type'].values, r=r_new)
        px_a = black_scholes_price_vec(S_new, df['Strike'].values, vol_aft,
                                       T_after, df['Type'].values, r=r_new)

        # Δposição = delta unitário × OI × 100
        dpos_b = g_b['delta'] * oi100
        dpos_a = g_a['delta'] * oi100
        pnl    = (px_a - px_b) * oi100
        # Ajuste incremental: quanto comprar/vender do ativo para rebalancear
        # Sinal: positivo = comprar underlying, negativo = vender
        # Componentes: delta puro + vanna (sensibilidade ao vol) + charm (decay temporal)
        _vanna_hedge = -(g_a['vanna'] - g_b['vanna']) * oi100 * dvol_arr
        _charm_hedge = -g_b['charm'] * oi100 * dt_yr
        hedge_adj = -(dpos_a - dpos_b) + _vanna_hedge + _charm_hedge

        # ── Tabela por instrumento ────────────────────────────────────────────
        expired_flag = T_after <= 0
        # Rótulo da asa aplicada por instrumento
        wing_lbl = np.where(
            (mness >= 0.50) & (mness < 1.00), 'PUT',
            np.where((mness >= 1.00) & (mness <= 1.50), 'CALL', 'OTM-deep')
        )
        df_inst = pd.DataFrame({
            'Expiry':        df['Exp'].dt.strftime('%d/%b/%y').values,
            'Strike':        df['Strike'].values.astype(int),
            'Mnss%':         np.round(mness * 100, 1),
            'Asa':           wing_lbl,
            'IV Base%':      np.round(df['IV'].values * 100, 2),
            'IV Cen.%':      np.round(vol_aft * 100, 2),
            'Type':          df['Type'].values,
            'OI':            df['OI'].values.astype(int),
            # ── Delta ────────────────────────────────────────────────────────
            'Δ Base':        np.round(g_b['delta'], 4),
            'Δ Cen.':        np.round(g_a['delta'], 4),
            'ΔΔ':            np.round(g_a['delta'] - g_b['delta'], 4),
            # ── Gamma ────────────────────────────────────────────────────────
            'Γ Base':        np.round(g_b['gamma'], 6),
            'Γ Cen.':        np.round(g_a['gamma'], 6),
            'ΔΓ':            np.round(g_a['gamma'] - g_b['gamma'], 6),
            # ── Vega ─────────────────────────────────────────────────────────
            'Vega Base':     np.round(g_b['vega'],  3),
            'Vega Cen.':     np.round(g_a['vega'],  3),
            'ΔVega':         np.round(g_a['vega'] - g_b['vega'], 3),
            # ── Theta ────────────────────────────────────────────────────────
            'Theta/d':       np.round(g_b['theta'] / TRADING_DAYS, 3),
            # ── Vanna ────────────────────────────────────────────────────────
            'Vanna Base':    np.round(g_b['vanna'], 4),
            'Vanna Cen.':    np.round(g_a['vanna'], 4),
            # ── Charm ────────────────────────────────────────────────────────
            'Charm/d':       np.round(g_b['charm'] / 365.0, 6),
            # ── P&L e Hedge ──────────────────────────────────────────────────
            'P&L ($)':       np.round(pnl, 0),
            'Hedge Adj (Δ)': np.round(hedge_adj, 1),
            '_Exp':          df['Exp'].values,
            '_exp_flag':     expired_flag,
        })
        # Marca expiradas
        df_inst['Type'] = np.where(expired_flag,
                                   df_inst['Type'] + '✗',
                                   df_inst['Type'])
        df_inst = (df_inst
                   .sort_values(['_Exp', 'OI'], ascending=[True, False])
                   .drop(columns=['_Exp', '_exp_flag']))

        INST_COLS  = ['Expiry', 'Strike', 'Mnss%', 'Asa', 'IV Base%', 'IV Cen.%',
                      'Type', 'OI',
                      'Δ Base', 'Δ Cen.', 'ΔΔ',
                      'Γ Base', 'Γ Cen.', 'ΔΓ',
                      'Vega Base', 'Vega Cen.', 'ΔVega',
                      'Theta/d', 'Vanna Base', 'Vanna Cen.', 'Charm/d',
                      'P&L ($)', 'Hedge Adj (Δ)']
        SIGN_COLS  = {'ΔΔ', 'ΔΓ', 'ΔVega', 'P&L ($)', 'Hedge Adj (Δ)',
                      'Vanna Base', 'Vanna Cen.', 'Charm/d', 'Theta/d'}
        MAX_ROWS   = 60

        # ── Agregação por vencimento ──────────────────────────────────────────
        df_agg_src = df_orig.copy()
        df_agg_src['_db']    = dpos_b
        df_agg_src['_da']    = dpos_a
        df_agg_src['_pnl']   = pnl
        df_agg_src['_adj']   = hedge_adj
        df_agg_src['_vb']    = g_b['vega']  * oi100
        df_agg_src['_va']    = g_a['vega']  * oi100
        df_agg_src['_gb']    = g_b['gamma'] * oi100
        df_agg_src['_ga']    = g_a['gamma'] * oi100
        df_agg_src['_vannb'] = g_b['vanna'] * oi100
        df_agg_src['_charmb']= g_b['charm'] * oi100 / 365.0

        grp = df_agg_src.groupby('Exp').agg(
            _db=('_db', 'sum'),   _da=('_da', 'sum'),
            _pnl=('_pnl', 'sum'), _adj=('_adj', 'sum'),
            _vb=('_vb', 'sum'),   _va=('_va', 'sum'),
            _gb=('_gb', 'sum'),   _ga=('_ga', 'sum'),
            _vannb=('_vannb', 'sum'),
            _charmb=('_charmb', 'sum'),
            n=('OI', 'count'),
        ).reset_index()

        df_agg = pd.DataFrame({
            'Vencimento':    pd.to_datetime(grp['Exp']).dt.strftime('%d/%b/%Y'),
            'Δpos Base':     grp['_db'].round(1),
            'Δpos Cen.':     grp['_da'].round(1),
            'ΔΔ Net':        (grp['_da'] - grp['_db']).round(1),
            'Γ Base':        grp['_gb'].round(2),
            'Γ Cen.':        grp['_ga'].round(2),
            'ΔΓ':            (grp['_ga'] - grp['_gb']).round(2),
            'Vega Base':     grp['_vb'].round(1),
            'Vega Cen.':     grp['_va'].round(1),
            'ΔVega':         (grp['_va'] - grp['_vb']).round(1),
            'Vanna (pos)':   grp['_vannb'].round(2),
            'Charm/d (pos)': grp['_charmb'].round(2),
            'P&L ($)':       grp['_pnl'].round(0),
            'Hedge Adj (Δ)': grp['_adj'].round(1),
            '# Strikes':     grp['n'],
        })
        df_agg = df_agg.sort_values('Vencimento')
        AGG_SIGN = {'ΔΔ Net', 'ΔΓ', 'ΔVega', 'P&L ($)', 'Hedge Adj (Δ)',
                    'Δpos Base', 'Δpos Cen.', 'Vanna (pos)', 'Charm/d (pos)'}

        # ── Consolidado ───────────────────────────────────────────────────────
        tot_db  = dpos_b.sum()
        tot_da  = dpos_a.sum()
        tot_pnl = pnl.sum()
        tot_adj = hedge_adj.sum()
        tot_vb  = (g_b['vega'] * oi100).sum()
        tot_va  = (g_a['vega'] * oi100).sum()
        n_exp   = grp.shape[0]

        adj_lbl  = ('▲ Comprar' if tot_adj > 0 else ('▼ Vender' if tot_adj < 0 else '—'))
        col_pnl  = 'rgba(0,212,232,.95)'  if tot_pnl >= 0 else 'rgba(248,81,73,.95)'
        col_adj  = 'rgba(245,166,35,.95)' if abs(tot_adj) > 0 else 'rgba(255,255,255,.35)'
        col_dd   = 'rgba(0,212,232,.95)'  if (tot_da-tot_db) > 0 else 'rgba(248,81,73,.95)'

        _scale_lbl = f'{_scale:.3f}×' if _aum_val > 0 else 'OI bruto'
        scenario_lbl = (f"ΔSpot {w_dspot.value:+.0f}  |  "
                        f"ΔVol Put {w_dvol_put.value:+.1f}pp  |  "
                        f"ΔVol Call {w_dvol_call.value:+.1f}pp  |  "
                        f"ΔRate {w_drate.value:+.0f}bp  |  "
                        f"+{w_days.value}d  |  "
                        f"AUM ${_aum_val:.1f}B [{_scale_lbl}]")

        cards_html = (
            f"<div style='margin:8px 0 12px;'>"
            f"<p style='color:rgba(255,255,255,.3);font-size:10px;margin:0 0 8px;"
            f"letter-spacing:.5px;'>CENÁRIO: {scenario_lbl}</p>"
            f"<div style='display:flex;gap:10px;flex-wrap:wrap;'>"
            + _card('ΔPOS BASE',     f'{tot_db:+,.1f}')
            + _card('ΔPOS CENÁRIO',  f'{tot_da:+,.1f}', color=col_dd)
            + _card('ΔΔ TOTAL',      f'{tot_da-tot_db:+,.1f}', color=col_dd,
                    sub='Δpos_after − Δpos_before')
            + _card('HEDGE ADJ TOTAL', f'{adj_lbl} {abs(tot_adj):,.1f}Δ', color=col_adj,
                    sub='−(ΔΔ) por vencimento')
            + _card('P&L ESTIMADO', f'${tot_pnl:+,.0f}', color=col_pnl)
            + _card('VEGA BASE→CEN', f'{tot_vb:+,.1f} → {tot_va:+,.1f}',
                    sub=f'{n_exp} vencimentos')
            + _card('AUM DEALERS', f'${_aum_val:.1f}B',
                    color='rgba(245,166,35,.9)',
                    sub=f'scale {_scale_lbl} vs OI bruto')
            + "</div></div>"
        )

        # ── Convenção de sinal ────────────────────────────────────────────────
        convention_html = (
            "<div style='background:rgba(245,166,35,.06);border-left:3px solid rgba(245,166,35,.4);"
            "border-radius:3px;padding:8px 14px;margin:8px 0;font-size:10px;"
            "color:rgba(255,255,255,.5);font-family:monospace;'>"
            "<b style='color:rgba(245,166,35,.8);'>Convenção de sinal</b> &nbsp;|&nbsp; "
            "Δpos = δ × OI × 100 &nbsp;·&nbsp; "
            "Hedge Adj = −(Δpos_after − Δpos_before) &nbsp;·&nbsp; "
            "+ = comprar underlying &nbsp;·&nbsp; − = vender &nbsp;·&nbsp; "
            "OI como proxy de posição (sem quantity/multiplier separado) &nbsp;·&nbsp; "
            "Vol: choque por asa — Put wing K/S∈[50%,100%) / Call wing K/S∈[100%,150%] / deep OTM sem choque &nbsp;·&nbsp; "
            "IV base já reflete o smile completo da surface (per-instrument BQL) &nbsp;·&nbsp; "
            "Opções com ✗ = expiraram no horizonte"
            "</div>"
        )

        # ── Render ────────────────────────────────────────────────────────────
        with out_cards:
            out_cards.clear_output(wait=True)
            _disp(wd.HTML(cards_html + convention_html))

        with out_agg:
            out_agg.clear_output(wait=True)
            _disp(wd.HTML(
                "<p style='color:rgba(0,212,232,.7);font-size:11px;margin:4px 0 6px;"
                "letter-spacing:.5px;'>AGREGADO POR VENCIMENTO</p>"
                + _html_table(df_agg, num_cols=AGG_SIGN)
            ))

        # ── Heatmap compacto da vol surface (expiry × strike bucket) ─────────
        # Agrupa IV base por expiry e faixas de strike arredondadas
        _stride = max(int(round((df_orig['Strike'].max() - df_orig['Strike'].min()) / 20)), 5)
        _sbins  = np.arange(df_orig['Strike'].min() // _stride * _stride,
                            df_orig['Strike'].max() + _stride, _stride)
        _surf   = df_orig.copy()
        _surf['_sb'] = ((_surf['Strike'] // _stride) * _stride).astype(int)
        _surf_piv = (_surf.groupby(['Exp', '_sb'])['IV']
                     .mean()
                     .unstack('_sb') * 100)

        # Monta HTML do heatmap
        def _vol_color(v, lo=15, hi=40):
            # cyan (baixo) → laranja (alto)
            t = max(0.0, min(1.0, (v - lo) / (hi - lo))) if pd.notna(v) else 0.5
            r = int(t * 245 + (1-t) * 0)
            g = int(t * 166 + (1-t) * 212)
            b = int(t * 35  + (1-t) * 232)
            return f'rgba({r},{g},{b},{0.5 + t*0.4:.2f})'

        _hm_strikes = sorted(_surf_piv.columns.tolist())
        _th_s = ''.join(f"<th style='padding:3px 8px;font-size:11px;color:rgba(0,212,232,.7);text-align:center;'>{int(k)}</th>"
                        for k in _hm_strikes)
        _hm_rows = ''
        for exp, row in _surf_piv.iterrows():
            _exp_lbl = pd.Timestamp(exp).strftime('%d/%b')
            _hm_rows += f"<tr><td style='padding:3px 8px;font-size:11px;color:rgba(255,255,255,.7);white-space:nowrap;font-weight:600;'>{_exp_lbl}</td>"
            for k in _hm_strikes:
                v = row.get(k, np.nan)
                bg = _vol_color(v) if pd.notna(v) else 'transparent'
                txt = f'{v:.1f}' if pd.notna(v) else ''
                _hm_rows += (f"<td style='padding:3px 7px;font-size:11px;text-align:center;"
                             f"background:{bg};color:rgba(0,0,0,.85);font-weight:700;'>{txt}</td>")
            _hm_rows += '</tr>'

        _hm_html = (
            "<div style='margin:12px 0 6px;'>"
            "<p style='color:rgba(0,212,232,.7);font-size:12px;margin:0 0 8px;letter-spacing:.5px;'>"
            "VOL SURFACE — IV% base por expiry × strike bucket (média por faixa)</p>"
            "<div style='overflow-x:auto;'><table style='border-collapse:collapse;font-family:monospace;'>"
            f"<thead><tr><th style='padding:3px 8px;font-size:11px;'></th>{_th_s}</tr></thead>"
            f"<tbody>{_hm_rows}</tbody></table></div>"
            "<p style='color:rgba(255,255,255,.3);font-size:10px;margin:4px 0 0;'>"
            f"Cor: baixa IV = ciano, alta IV = laranja &nbsp;·&nbsp; faixa de strike: {_stride}pts</p></div>"
        )

        with out_inst:
            out_inst.clear_output(wait=True)
            _disp(wd.HTML(_hm_html))
            shown = df_inst[INST_COLS].head(MAX_ROWS)
            suffix = (f" — exibindo {MAX_ROWS} de {len(df_inst)}, ordenado por expiry/OI desc"
                      if len(df_inst) > MAX_ROWS else '')
            _disp(wd.HTML(
                f"<p style='color:rgba(0,212,232,.7);font-size:11px;margin:12px 0 6px;"
                f"letter-spacing:.5px;'>POR INSTRUMENTO{suffix}</p>"
                + _html_table(shown, num_cols=SIGN_COLS)
            ))

        # ── Sensitivity Matrix ────────────────────────────────────────────────
        # Grid: ΔSpot(%) × ΔVol(pp) → Hedge Adj total e P&L
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots as _msp

        _sp_pcts  = np.array([-0.03, -0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02, 0.03])
        _vv_pps   = np.array([-15, -10, -5, 0, 5, 10, 15, 20])
        _K_s      = df_orig['Strike'].values
        _iv_s     = df_orig['IV'].values
        _T_s      = df_orig['Tte'].values
        _typ_s    = df_orig['Type'].values
        _oi_s_raw = df_orig['OI'].values * 100.0
        if _aum_val > 0:
            _mkt_s = float(_oi_s_raw.sum() * spot)
            _sc_s  = (_aum_val * 1e9) / _mkt_s if _mkt_s > 0 else 1.0
            _oi_s  = _oi_s_raw * _sc_s
        else:
            _oi_s  = _oi_s_raw
        _g_base_s = calculate_all_greeks(spot, _K_s, _iv_s, _T_s, _typ_s, r=rfr)
        _px_bs    = black_scholes_price_vec(spot, _K_s, _iv_s, _T_s, _typ_s, r=rfr)

        _sens_adj = np.zeros((len(_sp_pcts), len(_vv_pps)))
        _sens_pnl = np.zeros_like(_sens_adj)

        for _ii, _dp in enumerate(_sp_pcts):
            _S_i = spot * (1.0 + _dp)
            # Sticky-delta IV por vencimento
            if _dp != 0:
                _iv_i = _iv_s.copy()
                for _exp in df_orig['Exp'].unique():
                    _msk_e = df_orig['Exp'].values == _exp
                    _k_e  = _K_s[_msk_e]; _iv_e = _iv_s[_msk_e]
                    _mo   = _k_e / spot;  _mn   = _k_e / _S_i
                    _ord  = np.argsort(_mo)
                    _iv_i[_msk_e] = np.interp(_mn, _mo[_ord], _iv_e[_ord],
                                               left=_iv_e[_ord[0]], right=_iv_e[_ord[-1]])
            else:
                _iv_i = _iv_s.copy()
            for _jj, _dv_pp in enumerate(_vv_pps):
                _dv_dec = _dv_pp / 100.0
                _mn_i   = _K_s / spot
                _dvol_i = np.where((_mn_i >= 0.50) & (_mn_i < 1.00), _dv_dec,
                          np.where((_mn_i >= 1.00) & (_mn_i <= 1.50), _dv_dec * 0.7, 0.0))
                _vol_i  = np.maximum(_iv_i + _dvol_i, 0.001)
                _g_s_i  = calculate_all_greeks(_S_i, _K_s, _vol_i, _T_s, _typ_s, r=rfr)
                _px_s_i = black_scholes_price_vec(_S_i, _K_s, _vol_i, _T_s, _typ_s, r=rfr)
                _dp_b   = _g_base_s['delta'] * _oi_s
                _dp_a   = _g_s_i['delta']    * _oi_s
                _va_h   = -(_g_s_i['vanna'] - _g_base_s['vanna']) * _oi_s * _dvol_i
                _sens_adj[_ii, _jj] = (-((_dp_a - _dp_b)) + _va_h).sum()
                _sens_pnl[_ii, _jj] = ((_px_s_i - _px_bs) * _oi_s).sum()

        _sp_lbls = [f'{p*100:+.1f}%' for p in _sp_pcts]
        _vv_lbls = [f'{v:+d}pp'      for v in _vv_pps]

        _fig_sens = _msp(rows=1, cols=2,
                         subplot_titles=['Hedge Adj (Δ) — ΔSpot × ΔVol',
                                         'P&L ($) — ΔSpot × ΔVol'],
                         horizontal_spacing=0.10)
        _fig_sens.add_trace(go.Heatmap(
            z=_sens_adj, x=_vv_lbls, y=_sp_lbls, colorscale='RdYlGn', zmid=0,
            text=[[f'{v:,.0f}' for v in row] for row in _sens_adj],
            texttemplate='%{text}', showscale=True,
            colorbar=dict(x=0.44, thickness=10, len=0.85, title='Δ')), row=1, col=1)
        _fig_sens.add_trace(go.Heatmap(
            z=_sens_pnl, x=_vv_lbls, y=_sp_lbls, colorscale='RdYlGn', zmid=0,
            text=[[f'${v/1e6:.1f}M' for v in row] for row in _sens_pnl],
            texttemplate='%{text}', showscale=True,
            colorbar=dict(x=1.01, thickness=10, len=0.85, title='$')), row=1, col=2)
        _fig_sens.update_layout(
            height=430, paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
            font=dict(color='#e0e0e0', size=10),
            title=dict(text='Matrix de Sensibilidade — Hedge Adj e P&L por Cenário',
                       x=0.5, font=dict(size=12, color='#00d4e8')),
            margin=dict(l=60, r=60, t=55, b=30))
        _fig_sens.update_xaxes(title_text='ΔVol (put e call wing)', tickfont=dict(size=9))
        _fig_sens.update_yaxes(title_text='ΔSpot', tickfont=dict(size=9))
        # ── Valor central (spot=0, vol=0) → referência
        _adj_zero = float(_sens_adj[len(_sp_pcts)//2, list(_vv_pps).index(0)])
        _pnl_zero = float(_sens_pnl[len(_sp_pcts)//2, list(_vv_pps).index(0)])
        _sens_guide = (
            "<div style='background:#0d1520;border:1px solid rgba(0,212,232,.2);"
            "border-radius:6px;padding:12px 16px;margin:8px 0;font-size:11px;"
            "font-family:monospace;line-height:1.7;'>"
            "<span style='color:#00d4e8;font-size:10px;letter-spacing:.8px;'>COMO LER A MATRIX</span><br>"
            "<b style='color:#fff;'>Hedge Adj (Δ)</b> — quantidade de contratos do ativo que o dealer precisa negociar para rebalancear o delta hedge.<br>"
            "&nbsp;&nbsp;<span style='color:#ff4444;'>■ Vermelho = Vender</span> &nbsp;"
            "<span style='color:#44ff44;'>■ Verde = Comprar</span> &nbsp;·&nbsp; "
            "Linha 0%/+0pp = cenário atual sem choque.<br>"
            "<b style='color:#fff;'>P&L ($)</b> — resultado estimado da carteira de opções no cenário (antes do hedge).<br><br>"
            "<b style='color:#00d4e8;'>O que fazer:</b><br>"
            "① Identifique o cenário mais provável (ex: spot −1%, vol +5pp) e veja qual ajuste será necessário.<br>"
            "② Cells vermelhas intensas = você vai <b>vender</b> o ativo — prepare liquidez ou ordens limitadas.<br>"
            "③ Cells verdes intensas = você vai <b>comprar</b> — útil para pré-posicionar stops de compra.<br>"
            "④ Use a coluna +0pp como baseline: qualquer choque de vol puro move o hedge na horizontal.<br>"
            f"⑤ Cenário atual sem choque: Hedge Adj = <b style='color:#00d4e8;'>{_adj_zero:,.0f}Δ</b> &nbsp;·&nbsp; "
            f"P&L = <b style='color:#ff6b35;'>${_pnl_zero/1e6:.1f}M</b>"
            "</div>")
        with out_sensitivity:
            out_sensitivity.clear_output(wait=True)
            _disp(go.FigureWidget(_fig_sens))
            _disp(wd.HTML(_sens_guide))

    def _reset(_):
        w_dspot.value     = 0.0
        w_dvol_put.value  = 0.0
        w_dvol_call.value = 0.0
        w_drate.value     = 0.0
        w_days.value      = 0
        w_aum.value       = round(abs(dealer_aum_bn), 2) if dealer_aum_bn else round(_mkt_notional_bn, 2)
        _compute_and_render(None)

    # ── Predictive Analytics ─────────────────────────────────────────────────
    # HAR (Heterogeneous Autoregressive) para RV vs IV
    # CatBoost/GBM para sinal direcional de vol
    # PCA da surface para fatores de nível/skew/curvatura
    out_predict = wd.Output()
    w_pred_btn  = wd.Button(description='📊 Calibrar Modelos',
                            button_style='info',
                            layout=wd.Layout(width='170px', height='34px', margin='2px 0 0 0'),
                            tooltip='HAR (RV vs IV) · CatBoost signal · Surface PCA')

    def _run_predict(_):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots as _msp3
        _msgs = []

        # ── Dados da surface atual ────────────────────────────────────────
        _df_p   = df_orig.copy()
        _K_p    = _df_p['Strike'].values
        _iv_p   = _df_p['IV'].values
        _T_p    = _df_p['Tte'].values
        _atm_iv = float(_iv_p[np.argmin(np.abs(_K_p - spot))])  # ATM IV decimal

        # ══════════════════════════════════════════════════════════════════
        # 1. HAR — Heterogeneous Autoregressive Model
        #    RV_{t+1} = α + β_d·RV_t + β_w·RV̄_{t-5} + β_m·RV̄_{t-22} + ε
        #    Benchmark obrigatório para RV vs IV
        # ══════════════════════════════════════════════════════════════════
        _har_ok   = False
        _rv_fore  = None
        _rv_hist  = None
        _iv_proxy = None
        try:
            _hist_req = bql.Request(ticker, {
                'px': bq.data.px_last(dates=bq.func.range('-40d', '0d'),
                                      per='D', fill='PREV')})
            _hist_resp = bq.execute(_hist_req)
            _px_hist   = _hist_resp[0].df()['px'].dropna().values.astype(float)
            if len(_px_hist) >= 25:
                _rets    = np.diff(np.log(_px_hist))
                _rv_day  = _rets ** 2                        # variance diária
                N        = len(_rv_day)
                _rv_w    = np.array([_rv_day[max(0, i-5):i].mean()  for i in range(N)])
                _rv_m    = np.array([_rv_day[max(0, i-22):i].mean() for i in range(N)])
                _start   = 22
                _Y       = _rv_day[_start:]
                _X       = np.column_stack([
                    np.ones(len(_Y)),
                    _rv_day[_start-1:-1],
                    _rv_w[_start-1:-1],
                    _rv_m[_start-1:-1]])
                _coef, _, _, _ = np.linalg.lstsq(_X, _Y, rcond=None)
                _x_new   = np.array([1.0, _rv_day[-1], _rv_w[-1], _rv_m[-1]])
                _rv_fore_var = float(np.clip(_coef @ _x_new, 1e-10, None))
                _rv_fore     = np.sqrt(_rv_fore_var * 252) * 100   # % anualizado
                _rv_hist     = np.sqrt(_rv_day * 252) * 100        # série histórica %
                _iv_proxy    = _atm_iv * 100                       # ATM IV em %
                _har_ok      = True
                _msgs.append(f'HAR: β_d={_coef[1]:.3f} β_w={_coef[2]:.3f} β_m={_coef[3]:.3f}')
        except Exception as _e:
            _msgs.append(f'HAR: dados históricos indisponíveis ({_e})')

        # ══════════════════════════════════════════════════════════════════
        # 2. Surface PCA — Estágio 1 do modelo em dois estágios
        #    Fatores: nível (PC1), inclinação/skew (PC2), curvatura (PC3)
        # ══════════════════════════════════════════════════════════════════
        _pca_ok = False
        _pc_scores = None
        _pc_expvar = None
        _pca_fig   = None
        try:
            from sklearn.decomposition import PCA
            # Bucket de moneyness: 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15
            _mn_bkts = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
            _exps_u  = sorted(_df_p['Exp'].unique())[:8]   # até 8 vencimentos
            _surface = []
            _exp_lbls = []
            for _e in _exps_u:
                _row = _df_p[_df_p['Exp'] == _e]
                _mn_e  = _row['Strike'].values / spot
                _iv_e  = _row['IV'].values
                _ord_e = np.argsort(_mn_e)
                _iv_interp = np.interp(_mn_bkts, _mn_e[_ord_e], _iv_e[_ord_e],
                                        left=_iv_e[_ord_e[0]], right=_iv_e[_ord_e[-1]])
                _surface.append(_iv_interp * 100)
                _dte = int(float(_row['Tte'].mean()) * 365)
                _exp_lbls.append(f'{_dte}d')
            _surface = np.array(_surface)           # shape: (n_exp, n_moneyness)
            if _surface.shape[0] >= 3:
                _pca = PCA(n_components=min(3, _surface.shape[0]))
                _pca.fit(_surface)
                _pc_scores  = _pca.transform(_surface)
                _pc_expvar  = _pca.explained_variance_ratio_ * 100
                _loadings   = _pca.components_              # (3, 7)
                _pca_ok     = True
                _msgs.append(f'PCA surface: PC1={_pc_expvar[0]:.1f}% PC2={_pc_expvar[1]:.1f}% PC3={_pc_expvar[2]:.1f}%')
        except Exception as _e:
            _msgs.append(f'PCA: {_e}')

        # ══════════════════════════════════════════════════════════════════
        # 3. CatBoost / GBM — Sinal direcional de IV
        #    Features: IV level, IV change 1d/5d, RV-IV spread, IV momentum
        #    Label: IV sobe (1) ou cai (0) no próximo dia
        # ══════════════════════════════════════════════════════════════════
        _gb_ok   = False
        _gb_pred = None
        _gb_proba = None
        _feat_imp = None
        _feat_names = None
        try:
            # Busca histórico de IV (usa frontmonth ATM IV se disponível,
            # senão usa VIX como proxy de IV implícita)
            _iv_ticker = 'VIX Index' if 'SPX' in ticker.upper() else ticker
            _iv_req  = bql.Request(_iv_ticker, {
                'iv_px': bq.data.px_last(dates=bq.func.range('-90d', '0d'),
                                          per='D', fill='PREV')})
            _iv_resp  = bq.execute(_iv_req)
            _iv_hist_raw = _iv_resp[0].df()['iv_px'].dropna().values.astype(float)

            if len(_iv_hist_raw) >= 40 and _har_ok:
                # Alinhar com série de preços (igual comprimento)
                N_iv = min(len(_iv_hist_raw), len(_rv_hist)) if _rv_hist is not None else len(_iv_hist_raw)
                _iv_s_   = _iv_hist_raw[-N_iv:]
                _rv_s_   = _rv_hist[-N_iv:] if _rv_hist is not None else np.full(N_iv, _atm_iv * 100)

                # Feature engineering
                _iv_chg1 = np.diff(_iv_s_,   prepend=_iv_s_[0])
                _iv_chg5 = np.diff(_iv_s_, n=5, prepend=_iv_s_[:5])
                _rv_iv   = _rv_s_ - _iv_s_               # RV - IV spread
                _iv_mom  = _iv_s_ - np.array([_iv_s_[max(0,i-10):i+1].mean()
                                               for i in range(len(_iv_s_))])  # mean-reversion
                _feat_names = ['IV_level', 'IV_chg_1d', 'IV_chg_5d', 'RV-IV_spread', 'IV_momentum']
                _feats = np.column_stack([_iv_s_, _iv_chg1, _iv_chg5, _rv_iv, _iv_mom])

                # Label: IV sobe amanhã?
                _labels = (np.diff(_iv_s_, append=_iv_s_[-1]) > 0).astype(int)

                # Treina nos primeiros 80%, prediz nos últimos 20%
                _split = int(len(_feats) * 0.80)
                _X_tr, _X_te = _feats[:_split], _feats[_split:]
                _y_tr, _y_te = _labels[:_split], _labels[_split:]

                try:
                    from catboost import CatBoostClassifier
                    _gb = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05,
                                             verbose=0, random_seed=42)
                    _gb.fit(_X_tr, _y_tr)
                    _feat_imp = _gb.get_feature_importance()
                    _model_name = 'CatBoost'
                except ImportError:
                    from sklearn.ensemble import GradientBoostingClassifier
                    _gb = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                     learning_rate=0.05, random_state=42)
                    _gb.fit(_X_tr, _y_tr)
                    _feat_imp = _gb.feature_importances_
                    _model_name = 'GBM (sklearn fallback)'

                # Predição atual
                _cur_iv_chg1 = float(_iv_hist_raw[-1] - _iv_hist_raw[-2])
                _cur_iv_chg5 = float(_iv_hist_raw[-1] - _iv_hist_raw[-6]) if len(_iv_hist_raw) >= 6 else 0
                _cur_rv_iv   = (float(_rv_hist[-1]) if _rv_hist is not None else _atm_iv * 100) - float(_iv_hist_raw[-1])
                _cur_iv_mom  = float(_iv_hist_raw[-1]) - float(_iv_hist_raw[-10:].mean())
                _x_cur = np.array([[float(_iv_hist_raw[-1]), _cur_iv_chg1,
                                    _cur_iv_chg5, _cur_rv_iv, _cur_iv_mom]])
                _gb_proba = float(_gb.predict_proba(_x_cur)[0][1])   # P(IV sobe)
                _gb_pred  = 'Sobe' if _gb_proba > 0.55 else ('Cai' if _gb_proba < 0.45 else 'Neutro')
                _gb_ok    = True
                _msgs.append(f'{_model_name}: P(IV↑)={_gb_proba*100:.1f}% → {_gb_pred}')
        except Exception as _e:
            _msgs.append(f'CatBoost/GBM: {_e}')

        # ── Montagem dos charts ───────────────────────────────────────────
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots as _msp3

        n_cols = (1 + int(_pca_ok) + int(_gb_ok and _feat_imp is not None))
        _fig_p  = _msp3(rows=1, cols=n_cols,
                        subplot_titles=(
                            ['HAR: RV Forecast vs ATM IV']
                            + (['PCA Surface — Fatores'] if _pca_ok else [])
                            + ([f'{"CatBoost" if "CatBoost" in str(_msgs) else "GBM"} — Feature Importance'] if _gb_ok and _feat_imp is not None else [])
                        ),
                        horizontal_spacing=0.08)

        # Chart 1: HAR
        _col = 1
        if _har_ok and _rv_hist is not None:
            _days_axis = list(range(len(_rv_hist)))
            _fig_p.add_trace(go.Scatter(x=_days_axis, y=_rv_hist,
                mode='lines', line=dict(color='#00d4e8', width=1.5),
                name='RV Realizada (HAR)'), row=1, col=_col)
            _fig_p.add_hline(y=_atm_iv * 100, line_dash='dash',
                             line_color='#ff6b35', row=1, col=_col,
                             annotation_text=f'ATM IV {_atm_iv*100:.1f}%',
                             annotation_font_size=9)
            _fig_p.add_hline(y=_rv_fore, line_dash='dot',
                             line_color='#44ff44', row=1, col=_col,
                             annotation_text=f'HAR forecast {_rv_fore:.1f}%',
                             annotation_font_size=9)
        else:
            _fig_p.add_annotation(text='Dados históricos<br>indisponíveis',
                                  xref='paper', yref='paper', x=0.1, y=0.5,
                                  showarrow=False, font=dict(color='#aaa'), row=1, col=_col)
        _col += 1

        # Chart 2: PCA loadings
        if _pca_ok:
            _mn_lbls_pca = [f'{int(m*100)}%' for m in _mn_bkts]
            _colors_pca  = ['#00d4e8', '#ff6b35', '#44ff44']
            for _pc_i in range(min(3, len(_pca.components_))):
                _fig_p.add_trace(go.Scatter(
                    x=_mn_lbls_pca, y=_pca.components_[_pc_i] * 100,
                    mode='lines+markers',
                    line=dict(color=_colors_pca[_pc_i], width=2),
                    name=f'PC{_pc_i+1} ({_pc_expvar[_pc_i]:.1f}%)'), row=1, col=_col)
            _col += 1

        # Chart 3: Feature Importance
        if _gb_ok and _feat_imp is not None and _feat_names is not None:
            _fi_sorted = sorted(zip(_feat_names, _feat_imp), key=lambda x: x[1])
            _fig_p.add_trace(go.Bar(
                x=[v for _, v in _fi_sorted],
                y=[n for n, _ in _fi_sorted],
                orientation='h',
                marker_color='#00d4e8',
                name='Importância'), row=1, col=_col)

        _fig_p.update_layout(
            height=400, paper_bgcolor='#0f1117', plot_bgcolor='#131722',
            font=dict(color='#e0e0e0', size=10), showlegend=True,
            legend=dict(orientation='h', y=-0.15, font=dict(size=9)),
            margin=dict(l=50, r=40, t=55, b=60))

        # Painel de signal
        _iv_rv_spread = (_atm_iv * 100 - _rv_fore) if _rv_fore else None
        _signal_color = '#00ff99' if (_iv_rv_spread and _iv_rv_spread > 2) else \
                        '#ff4444' if (_iv_rv_spread and _iv_rv_spread < -2) else '#aaaaaa'
        _signal_text  = ('Vender Vol — IV > HAR RV'  if _iv_rv_spread and _iv_rv_spread > 2 else
                         'Comprar Vol — IV < HAR RV' if _iv_rv_spread and _iv_rv_spread < -2 else
                         'IV ≈ RV — sem sinal claro')
        _gb_color     = '#00ff99' if _gb_pred == 'Sobe' else '#ff4444' if _gb_pred == 'Cai' else '#aaa'
        # ── Ações sugeridas por modelo ────────────────────────────────────
        _har_action = (
            'Venda de vol tem vantagem estatística (IV cara): short strangle / short straddle / venda de puts cobertas. '
            f'Prêmio de vol = {_iv_rv_spread:+.1f}pp — quanto maior, maior a margem de segurança.'
            if _rv_fore and _iv_rv_spread > 2 else
            'Compra de vol tem vantagem estatística (IV barata): long straddle / long calls. '
            f'IV está {abs(_iv_rv_spread):.1f}pp abaixo do RV esperado.'
            if _rv_fore and _iv_rv_spread < -2 else
            'Sem vantagem clara: IV ≈ RV esperado. Foque em estruturas com carry positivo (spreads).'
            if _rv_fore else 'Dados históricos indisponíveis.')
        _gb_action = (
            f'P(IV↑)={_gb_proba*100:.1f}% — alta confiança de alta de vol. '
            'Posições longas em vol (long straddle, compra de calls/puts, ratio spreads) têm vantagem no curto prazo. '
            'Atenção: sinal é direcional de vol, não de spot.'
            if _gb_ok and _gb_pred == 'Sobe' else
            f'P(IV↑)={_gb_proba*100:.1f}% — modelo aponta queda de vol. '
            'Venda de vol (short strangle, iron condor, venda de puts) tem vantagem. '
            'Confirme com HAR antes de executar.'
            if _gb_ok and _gb_pred == 'Cai' else
            'Sinal inconclusivo — aguarde confirmação ou reduza tamanho.'
            if _gb_ok else 'Modelo não calibrado.')
        _pca_action = (
            f'PC1 domina com {_pc_expvar[0]:.1f}% da variância — surface se move principalmente em nível (vol up/down paralelo). '
            + (f'PC2 (skew={_pc_expvar[1]:.1f}%) é relevante — inclinação put vs call está variando, monitore risk reversal. ' if _pca_ok and _pc_expvar[1] > 5 else
               f'PC2 (skew={_pc_expvar[1]:.1f}%) baixo — surface está relativamente plana entre puts e calls. ' if _pca_ok else '')
            + (f'PC3 (curv={_pc_expvar[2]:.1f}%) — curvatura/smile insignificante.' if _pca_ok and len(_pc_expvar) > 2 and _pc_expvar[2] < 2 else
               f'PC3 (curv={_pc_expvar[2]:.1f}%) — smile pronunciado, butterfly spreads podem ser caros.' if _pca_ok and len(_pc_expvar) > 2 else '')
            if _pca_ok else 'PCA não disponível.')

        _signal_html  = (
            # Cards de sinal
            f"<div style='display:flex;gap:12px;margin:10px 0 6px;flex-wrap:wrap;'>"
            + (f"<div style='background:#1a2035;padding:10px 18px;border-radius:6px;"
               f"border-left:3px solid {_signal_color};min-width:200px;'>"
               f"<div style='color:#aaa;font-size:9px;letter-spacing:.8px;'>HAR SIGNAL</div>"
               f"<div style='color:{_signal_color};font-size:14px;font-weight:bold;'>{_signal_text}</div>"
               f"<div style='color:#aaa;font-size:10px;'>IV={_atm_iv*100:.1f}% · RV HAR={_rv_fore:.1f}% · spread={_iv_rv_spread:+.1f}pp</div>"
               f"</div>" if _rv_fore else '')
            + (f"<div style='background:#1a2035;padding:10px 18px;border-radius:6px;"
               f"border-left:3px solid {_gb_color};min-width:180px;'>"
               f"<div style='color:#aaa;font-size:9px;letter-spacing:.8px;'>CATBOOST/GBM SIGNAL</div>"
               f"<div style='color:{_gb_color};font-size:14px;font-weight:bold;'>IV {_gb_pred}</div>"
               f"<div style='color:#aaa;font-size:10px;'>P(IV↑) = {_gb_proba*100:.1f}%</div>"
               f"</div>" if _gb_ok else '')
            + (f"<div style='background:#1a2035;padding:10px 18px;border-radius:6px;"
               f"border-left:3px solid #00d4e8;min-width:220px;'>"
               f"<div style='color:#aaa;font-size:9px;letter-spacing:.8px;'>SURFACE PCA</div>"
               f"<div style='color:#00d4e8;font-size:13px;'>"
               f"PC1 nível: {_pc_expvar[0]:.1f}% &nbsp;·&nbsp; PC2 skew: {_pc_expvar[1]:.1f}%"
               + (f" &nbsp;·&nbsp; PC3 curv: {_pc_expvar[2]:.1f}%" if len(_pc_expvar) > 2 else '')
               + f"</div></div>" if _pca_ok else '')
            + f"</div>"
            # Guia de interpretação e ação
            + f"<div style='background:#0d1520;border:1px solid rgba(0,212,232,.15);"
            f"border-radius:6px;padding:12px 16px;margin:6px 0;font-size:11px;"
            f"font-family:monospace;line-height:1.8;'>"
            f"<span style='color:#00d4e8;font-size:10px;letter-spacing:.8px;'>O QUE FAZER COM ESSES SINAIS</span><br><br>"
            # HAR
            f"<b style='color:{_signal_color};'>① HAR (Realized Vol Forecast)</b><br>"
            f"&nbsp;&nbsp;Modelo: RV_{{t+1}} = α + β_d·RV_t + β_w·RV̄_5d + β_m·RV̄_22d &nbsp;·&nbsp; "
            f"Captura clustering de vol em horizonte diário/semanal/mensal.<br>"
            f"&nbsp;&nbsp;→ {_har_action}<br><br>"
            # CatBoost
            f"<b style='color:{_gb_color};'>② CatBoost/GBM (Sinal Direcional de IV)</b><br>"
            f"&nbsp;&nbsp;Features: nível de IV, variação 1d/5d, spread RV-IV, momentum de IV.<br>"
            f"&nbsp;&nbsp;→ {_gb_action}<br><br>"
            # PCA
            f"<b style='color:#00d4e8;'>③ Surface PCA (Estrutura dos Fatores)</b><br>"
            f"&nbsp;&nbsp;PC1=nível · PC2=skew put/call · PC3=curvatura/smile. "
            f"Cada fator independente — movimentos misturados são raros.<br>"
            f"&nbsp;&nbsp;→ {_pca_action}<br><br>"
            # Consenso
            f"<b style='color:#fff;'>④ Consenso dos modelos</b><br>"
            + (f"&nbsp;&nbsp;<span style='color:#00ff99;'>✓ HAR e GBM alinham</span> — "
               f"sinal reforçado. Execute com maior convicção."
               if _rv_fore and _gb_ok and
                  ((_iv_rv_spread > 2 and _gb_pred == 'Sobe') or (_iv_rv_spread < -2 and _gb_pred == 'Cai'))
               else
               f"&nbsp;&nbsp;<span style='color:#ffaa00;'>⚠ HAR e GBM divergem</span> — "
               f"sinais conflitantes. Reduza tamanho ou aguarde próximo pregão."
               if _rv_fore and _gb_ok else
               f"&nbsp;&nbsp;Apenas um modelo disponível — use com cautela.")
            + f"</div>"
            + f"<p style='color:rgba(255,255,255,.25);font-size:9px;margin:4px 0 0;'>"
            + ' &nbsp;|&nbsp; '.join(_msgs) + f"</p>")

        with out_predict:
            out_predict.clear_output(wait=True)
            _disp(go.FigureWidget(_fig_p))
            _disp(wd.HTML(_signal_html))

    w_pred_btn.on_click(_run_predict)
    w_btn.on_click(_compute_and_render)
    w_reset.on_click(_reset)
    _compute_and_render(None)  # render cenário zero ao carregar

    header = wd.HTML(
        f"<div style='padding:10px 0 2px;'>"
        f"<h3 style='color:#00d4e8;margin:0 0 2px;font-size:15px;'>"
        f"Ajuste Dinâmico do Book — {ticker}</h3>"
        f"<p style='color:rgba(255,255,255,.3);font-size:10px;margin:0;'>"
        f"Repricing BS por instrumento → agregação por expiry → hedge adjustment por vencimento &nbsp;·&nbsp; "
        f"Vol surface: choque independente por asa (put wing 50–100% / call wing 100–150%)"
        f"</p></div>"
    )
    vol_label = wd.HTML(
        "<p style='color:rgba(0,212,232,.5);font-size:9px;margin:0 0 2px;"
        "letter-spacing:.8px;font-family:monospace;'>"
        "VOL SURFACE SHOCK</p>",
        layout=wd.Layout(margin='6px 0 0 0'))
    input_row = wd.VBox([
        wd.HBox([w_dspot, w_drate, w_days, w_aum, w_btn, w_reset],
                layout=wd.Layout(flex_flow='row wrap', gap='8px')),
        wd.HBox([vol_label, w_dvol_put, w_dvol_call],
                layout=wd.Layout(flex_flow='row wrap', gap='8px', align_items='center')),
    ], layout=wd.Layout(margin='4px 0 10px 0'))

    smile_widget = build_vol_smile_chart(df_orig, spot, ticker=ticker)

    pred_row = wd.HBox(
        [w_pred_btn],
        layout=wd.Layout(margin='10px 0 4px 0', align_items='center'))
    return wd.VBox([header, input_row, out_cards, smile_widget, out_agg, out_inst,
                    out_sensitivity, pred_row, out_predict])


def build_squeeze_tab(squeeze_result, net_gex_bn, spot, gamma_flip,
                      iv_30d, rv_30d, pc_ratio, _C, vvol_data=None):
    """Monta widget da aba Gamma Squeeze."""
    import plotly.graph_objects as go

    score = squeeze_result['score']
    alert = squeeze_result['alert']
    interp = squeeze_result['interp']
    comps = squeeze_result['components']

    alert_colors = {
        'critical': '#ff4444',
        'warning': '#ffaa00',
        'moderate': '#88aaff',
        'low': '#3fb950',
    }
    alert_color = alert_colors.get(alert, _C['text'])

    # ── Gauge do score ──
    gauge_fig = go.FigureWidget(go.Indicator(
        mode='gauge+number',
        value=score,
        number={'font': {'color': alert_color, 'size': 36}},
        title={'text': 'Gamma Squeeze Risk', 'font': {'color': _C['text'], 'size': 14}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': _C['text_muted'],
                     'tickfont': {'color': _C['text_muted'], 'size': 9}},
            'bar': {'color': alert_color, 'thickness': 0.3},
            'bgcolor': _C['card2'],
            'steps': [
                {'range': [0, 35],  'color': '#1a3a2a'},
                {'range': [35, 55], 'color': '#3a3020'},
                {'range': [55, 75], 'color': '#3a2510'},
                {'range': [75, 100], 'color': '#3a1a1a'},
            ],
            'threshold': {'line': {'color': alert_color, 'width': 3},
                          'thickness': 0.75, 'value': score},
        }
    ))
    gauge_fig.update_layout(
        height=250, width=280, template='plotly_dark',
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor=_C['card'], plot_bgcolor=_C['card'])

    # ── Barra de componentes ──
    bar_labels = [c['label'] for c in comps.values()]
    bar_scores = [c['score'] for c in comps.values()]
    bar_maxes  = [c['max']   for c in comps.values()]
    bar_colors = [alert_color if s / m > 0.6 else _C['accent']
                  for s, m in zip(bar_scores, bar_maxes)]

    bar_fig = go.FigureWidget()
    bar_fig.add_trace(go.Bar(
        y=bar_labels, x=bar_scores, orientation='h',
        marker_color=bar_colors, name='Score',
        text=[f"{s:.0f}/{m}" for s, m in zip(bar_scores, bar_maxes)],
        textposition='outside'))
    bar_fig.update_layout(
        title='Componentes do Score',
        xaxis=dict(range=[0, 35], title='Score'),
        height=max(250, len(bar_labels) * 38 + 50), template='plotly_dark',
        margin=dict(t=35, b=20, l=5, r=70),
        paper_bgcolor=_C['card'], plot_bgcolor=_C['card'], showlegend=False)

    # ── Resumo textual ──
    _sq_mag = squeeze_result['squeeze_mag_pct']
    _fd = squeeze_result['flip_dist_pct']
    _flip_dir = 'ACIMA' if squeeze_result['flip_above'] else 'ABAIXO'

    # pré-calcula cor do tail skew (evita aspas duplas dentro de f-string)
    _sk3 = squeeze_result.get('sk3_ratio')
    _sk3_col = '#f85149' if (_sk3 and _sk3 < 0.40) else '#ffaa00' if (_sk3 and _sk3 < 0.55) else '#3fb950'
    _sk3_tag = ' ⚠ TAIL EXTREMO' if (_sk3 and _sk3 < 0.40) else ''
    _sk3_row = (f"<tr><td>Tail Skew (SPXSK3/IV×10)</td>"
                f"<td><b style='color:{_sk3_col};'>{_sk3:.3f}{_sk3_tag}</b></td></tr>"
                if _sk3 is not None else "")
    _v9d = squeeze_result.get('vix9d')
    _v9d_row = (f"<tr><td>VIX9D</td><td><b>{_v9d:.1f}</b></td></tr>"
                if _v9d is not None else "")

    summary_html = (
        f"<div class='mm-dash'><div class='mm-card'>"
        f"<h3 style='color:{alert_color}'>Gamma Squeeze Score: {score:.0f}/100</h3>"
        f"<p><b style='color:{alert_color}'>{interp}</b></p>"
        f"<table class='mm-table' style='width:auto;font-size:12px;'>"
        f"<tr><td>GEX NET</td><td><b>{net_gex_bn:+.2f}B</b></td></tr>"
        f"<tr><td>Gamma Flip</td><td><b>{f'{gamma_flip:,.0f}' if gamma_flip else 'N/A'}</b> ({_fd:.1f}% {_flip_dir} do spot)</td></tr>"
        f"<tr><td>P/C OI Ratio</td><td><b>{pc_ratio:.2f}x</b></td></tr>"
        f"<tr><td>IV-RV Gap</td><td><b>{(iv_30d-rv_30d)*100:+.1f} vol pts</b></td></tr>"
        + _sk3_row + _v9d_row
        + f"<tr><td>Magnitude estimada</td><td><b>~{_sq_mag:.1f}%</b> se flip cruzado</td></tr>"
        f"</table>"
    )
    # Componentes detalhados
    for _k, _cv in comps.items():
        _pct = _cv['score'] / _cv['max'] * 100
        _bar = '█' * int(_pct / 10) + '░' * (10 - int(_pct / 10))
        summary_html += (
            f"<p style='margin:4px 0;font-size:12px;'>"
            f"<b>{_cv['label']}</b>: {_cv['value']} "
            f"[{_bar}] {_cv['score']:.0f}/{_cv['max']} — "
            f"<i>{_cv['desc']}</i></p>")
    summary_html += "</div></div>"

    # ── Eventos históricos ──
    _evts_html = (
        "<div class='mm-dash'><div class='mm-card'>"
        "<h3>Eventos Históricos — Gamma Squeeze Triggers</h3>"
        "<table class='mm-table' style='width:100%;font-size:12px;'>"
        "<tr style='background:#161b22;'>"
        "<th>Data</th><th>Evento</th><th>Tipo</th><th>Descrição</th></tr>")
    _type_colors = {
        'squeeze': '#ffaa00', 'bottom': '#3fb950',
        'reversal': '#58a6ff', 'vix_spike': '#ff6b6b'}
    for ev in GAMMA_SQUEEZE_EVENTS:
        _tc = _type_colors.get(ev['type'], '#8b949e')
        _evts_html += (
            f"<tr><td>{ev['date']}</td>"
            f"<td><b>{ev['label']}</b></td>"
            f"<td style='color:{_tc}'>{ev['type'].upper()}</td>"
            f"<td style='font-size:11px;'>{ev['desc']}</td></tr>")
    _evts_html += "</table></div></div>"

    # ── Vol-of-Vol & Tail Risk Panel ─────────────────────────────────────────
    _vvol = vvol_data or {}
    _vvix      = squeeze_result.get('vvix') or _vvol.get('vvix_cur')
    _c25       = squeeze_result.get('vix_skew_c25') or _vvol.get('vix_skew_c25')
    _p25       = squeeze_result.get('vix_skew_p25') or _vvol.get('vix_skew_p25')
    _sdex      = squeeze_result.get('sdex_cur') or _vvol.get('sdex_cur')
    _tdex      = squeeze_result.get('tdex_cur') or _vvol.get('tdex_cur')
    _call_oi   = squeeze_result.get('vix_call_oi') or _vvol.get('vix_call_oi')
    _put_oi    = squeeze_result.get('vix_put_oi') or _vvol.get('vix_put_oi')

    def _cell(val, fmt, lo=None, hi=None, lo_color='#3fb950', hi_color='#f85149'):
        """Formata célula com cor opcional baseada em thresholds."""
        if val is None:
            return "<td>—</td>"
        txt = fmt.format(val)
        col = _C['text']
        if hi is not None and val >= hi:
            col = hi_color
        elif lo is not None and val <= lo:
            col = lo_color
        return f"<td><b style='color:{col};'>{txt}</b></td>"

    def _pct_color_level(cur, hist_s, label_low='🔴 guarda baixa', label_mid='🟡 moderado', label_hi='🟢 defensivo'):
        """Percentil baixo = barato/plano = guard down (vermelho). Alto = caro/íngreme = defensivo (verde)."""
        if hist_s is not None and len(hist_s) > 20:
            _arr = np.asarray(hist_s.dropna())
            _pct = float(np.mean(_arr < cur))
            _col = ('#f85149' if _pct < 0.20 else
                    '#ffaa00' if _pct < 0.40 else
                    '#3fb950' if _pct > 0.70 else '#8b949e')
            _lvl = (label_low  if _pct < 0.20 else
                    label_mid  if _pct < 0.40 else
                    label_hi   if _pct > 0.70 else '🔵 neutro')
            _pct_str = f' p{_pct*100:.0f}'
        else:
            _col, _lvl, _pct_str = '#8b949e', '— sem histórico', ''
        return _col, _lvl, _pct_str

    # ── SPX IV & Skew context vars ──────────────────────────────────────────
    _spx_iv_cur  = _vvol.get('spx_iv_cur')
    _spx_iv_pct  = _vvol.get('spx_iv_pct')
    _spx_sk_cur  = _vvol.get('spx_skew_cur')
    _spx_sk_pct  = _vvol.get('spx_skew_pct')

    _vvol_html = (
        "<div class='mm-dash'><div class='mm-card'>"
        "<h3 style='margin:0 0 8px;'>Vol-of-Vol &amp; Tail Risk Indicators</h3>"
        "<table class='mm-table' style='width:100%;font-size:12px;'>"
        "<tr style='background:#161b22;'>"
        "<th>Indicador</th><th>Valor</th><th>Percentil 1Y</th><th>Descrição</th></tr>"
    )

    # SPX 30d ATM IV + Percentile
    if _spx_iv_cur is not None:
        _iv_col = ('#f85149' if (_spx_iv_pct or 0) > 80 else
                   '#ffaa00' if (_spx_iv_pct or 0) > 60 else
                   '#3fb950' if (_spx_iv_pct or 0) < 30 else '#8b949e')
        _iv_lvl = ('🔴 IV alta — fear' if (_spx_iv_pct or 0) > 80 else
                   '🟡 IV elevada'     if (_spx_iv_pct or 0) > 60 else
                   '🟢 IV baixa — complacência' if (_spx_iv_pct or 0) < 30 else '🔵 IV neutra')
        _pct_bar = f'p{_spx_iv_pct:.0f}' if _spx_iv_pct is not None else '—'
        _vvol_html += (
            f"<tr><td>SPX 30d ATM IV</td>"
            f"<td><b style='color:{_iv_col};'>{_spx_iv_cur:.1f}%</b></td>"
            f"<td style='color:{_iv_col};'><b>{_pct_bar}</b></td>"
            f"<td style='font-size:11px;'>IV implícita 30d ATM — percentil relativo a 1 ano</td></tr>")

    # SPX 30d Skew (ATM − 90% moneyness)
    if _spx_sk_cur is not None:
        _sk_col, _sk_lvl, _sk_pct_str = _pct_color_level(
            _spx_sk_cur, _vvol.get('spx_skew_hist'),
            label_low='🟢 skew plano — complacência',
            label_mid='🟡 skew moderado',
            label_hi='🔴 skew íngreme — proteção cara')
        _vvol_html += (
            f"<tr><td>SPX Skew 30d (ATM−90%)</td>"
            f"<td><b style='color:{_sk_col};'>{_spx_sk_cur:.2f} vol pts</b></td>"
            f"<td style='color:{_sk_col};'><b>p{_spx_sk_pct:.0f}</b></td>"
            f"<td style='font-size:11px;'>Skew 30d: IV ATM − IV 90% OTM — alto = mercado pagando por downside</td></tr>"
            if _spx_sk_pct is not None else
            f"<tr><td>SPX Skew 30d (ATM−90%)</td>"
            f"<td><b style='color:{_sk_col};'>{_spx_sk_cur:.2f} vol pts</b></td>"
            f"<td>—</td>"
            f"<td style='font-size:11px;'>Skew 30d: IV ATM − IV 90% OTM</td></tr>")

    # VVIX
    if _vvix is not None:
        _vv_lvl = '🔴 EXTREMO' if _vvix > 130 else ('🟡 ELEVADO' if _vvix > 100 else '🟢 normal')
        _vv_col = '#f85149' if _vvix > 130 else ('#ffaa00' if _vvix > 100 else '#3fb950')
        _vvol_html += (f"<tr><td>VVIX (Vol-of-Vol)</td>"
                       f"<td><b style='color:{_vv_col};'>{_vvix:.1f}</b></td>"
                       f"<td style='color:{_vv_col};'>{_vv_lvl}</td>"
                       f"<td style='font-size:11px;'>Vol da vol do SPX — >130 = panic, >100 = elevado</td></tr>")

    # VIX 25d Call/Put skew
    if _c25 is not None or _p25 is not None:
        _c25_str = f"{_c25:.3f}x" if _c25 else "—"
        _p25_str = f"{_p25:.3f}x" if _p25 else "—"
        _c25_col = '#ffaa00' if (_c25 and _c25 > 1.10) else _C['text']
        _p25_col = '#f85149' if (_p25 and _p25 > 1.20) else _C['text']
        _vvol_html += (
            f"<tr><td>VIX 25Δ Call IV / ATM</td>"
            f"<td><b style='color:{_c25_col};'>{_c25_str}</b></td>"
            f"<td style='color:{_c25_col};'>{'⬆ acima ATM' if (_c25 and _c25 > 1.0) else 'normal'}</td>"
            f"<td style='font-size:11px;'>Inst. comprando upside de vol → squeeze amplifier</td></tr>"
            f"<tr><td>VIX 25Δ Put IV / ATM</td>"
            f"<td><b style='color:{_p25_col};'>{_p25_str}</b></td>"
            f"<td style='color:{_p25_col};'>{'⬆ skew elevado' if (_p25 and _p25 > 1.10) else 'normal'}</td>"
            f"<td style='font-size:11px;'>Skew de put do VIX → demanda por tail protection</td></tr>")

    # VRP = VIX − RV 10D SPX
    _vrp_cur  = _vvol.get('vrp_cur')
    _rv10_cur = _vvol.get('rv10_cur')
    _vrp_hist = _vvol.get('vrp_hist')
    if _vrp_cur is not None:
        _vrp_col, _vrp_lvl, _vrp_pct = _pct_color_level(
            _vrp_cur, _vrp_hist,
            label_low='🔴 VRP baixo — IV comprimida',
            label_mid='🟡 VRP moderado',
            label_hi='🟢 VRP alto — medo embutido')
        _rv10_str = f' | RV10D: {_rv10_cur:.1f}%' if _rv10_cur is not None else ''
        _vix_ref  = _vvol.get('vvix_cur')  # usa VVIX como proxy do VIX se disponível
        _vvol_html += (
            f"<tr><td>VRP (VIX − RV 10D)</td>"
            f"<td><b style='color:{_vrp_col};'>{_vrp_cur:+.1f} vol pts{_vrp_pct}{_rv10_str}</b></td>"
            f"<td style='color:{_vrp_col};'>{_vrp_lvl}</td>"
            f"<td style='font-size:11px;'>Prêmio de vol: IV implícita vs realizada 10D — alto = medo excessivo = squeeze fuel</td></tr>")

    # VIX OI Call vs Put
    if _call_oi is not None or _put_oi is not None:
        _oi_ratio = (_call_oi / _put_oi) if (_call_oi and _put_oi and _put_oi > 0) else None
        _oi_col = '#ffaa00' if (_oi_ratio and _oi_ratio > 2.0) else _C['text']
        _oi_str = (f"{_call_oi:.1f}M calls / {_put_oi:.1f}M puts"
                   + (f" = {_oi_ratio:.1f}x call-heavy" if _oi_ratio else ""))
        _vvol_html += (
            f"<tr><td>VIX OI (Call vs Put)</td>"
            f"<td><b style='color:{_oi_col};'>{_oi_str}</b></td>"
            f"<td style='color:{_oi_col};'>{'⚠ call heavy' if (_oi_ratio and _oi_ratio > 2.0) else 'balanceado'}</td>"
            f"<td style='font-size:11px;'>>2x calls = inst. comprando upside de vol / hedges de curto prazo</td></tr>")

    # SDEX — inclinação do skew SPY (OTM/ATM)
    if _sdex is not None:
        _sd_hist = _vvol.get('sdex_hist')
        _sd_col, _sd_lvl, _sd_pct = _pct_color_level(
            _sdex, _sd_hist,
            label_low='🔴 skew plano — guarda baixa',
            label_mid='🟡 skew moderado',
            label_hi='🟢 skew íngreme — defensivo')
        _vvol_html += (
            f"<tr><td>SDEX (CBOE Skew SPY)</td>"
            f"<td><b style='color:{_sd_col};'>{_sdex:.2f}{_sd_pct}</b></td>"
            f"<td style='color:{_sd_col};'>{_sd_lvl}</td>"
            f"<td style='font-size:11px;'>Inclinação OTM/ATM do SPY — baixo = skew plano = mercado de guarda baixa</td></tr>")

    # TDEX — custo absoluto de OTM tail risk
    if _tdex is not None:
        _td_hist = _vvol.get('tdex_hist')
        _td_col, _td_lvl, _td_pct = _pct_color_level(
            _tdex, _td_hist,
            label_low='🔴 tail barato — guarda baixa',
            label_mid='🟡 tail moderado',
            label_hi='🟢 tail caro — mercado defensivo')
        _vvol_html += (
            f"<tr><td>TDEX (CBOE Tail Risk)</td>"
            f"<td><b style='color:{_td_col};'>{_tdex:.2f}{_td_pct}</b></td>"
            f"<td style='color:{_td_col};'>{_td_lvl}</td>"
            f"<td style='font-size:11px;'>Custo de OTM tail risk — baixo = tail options baratas = mercado de guarda baixa</td></tr>")

    # ── Funding & Liquidity Stress ────────────────────────────────────────────
    _axwa      = _vvol.get('axwa_cur')
    _axwa_hist = _vvol.get('axwa_hist')
    _axwa_vol  = _vvol.get('axwa_vol')
    _fed1      = _vvol.get('fedpsor1_cur')
    _fed1_hist = _vvol.get('fedpsor1_hist')
    _es_ba     = _vvol.get('es_bid_ask_cur')

    if any(v is not None for v in [_axwa, _fed1, _es_ba]):
        _vvol_html += (
            "<tr style='background:#0d1117;'>"
            "<td colspan='4' style='color:#58a6ff;font-weight:bold;padding:4px 0 2px;'>"
            "── Funding &amp; Liquidity Stress ──</td></tr>")

    # AXWA — equity funding spread
    if _axwa is not None:
        _ax_col, _ax_lvl, _ax_pct = _pct_color_level(
            _axwa, _axwa_hist,
            label_low='🟢 funding barato',
            label_mid='🟡 funding moderado',
            label_hi='🔴 funding caro — stress')
        # para AXWA: HIGH = stress (inverte lógica — alto pct = vermelho)
        _ax_col = ('#f85149' if _ax_col == '#3fb950' else
                   '#3fb950' if _ax_col == '#f85149' else _ax_col)
        _ax_lvl = ('🔴 funding caro — stress' if '🟢' in _ax_lvl else
                   '🟢 funding barato'        if '🔴' in _ax_lvl else _ax_lvl)
        _vol_str = f' | vol: {int(_axwa_vol):,}' if _axwa_vol else ''
        _vvol_html += (
            f"<tr><td>AXWA (SPX Funding Spread)</td>"
            f"<td><b style='color:{_ax_col};'>{_axwa:.1f}{_ax_pct}{_vol_str}</b></td>"
            f"<td style='color:{_ax_col};'>{_ax_lvl}</td>"
            f"<td style='font-size:11px;'>Custo de financiamento de equity SPX — alto = funding squeeze</td></tr>")

    # FEDPSOR1 — Primary Dealer equity repo
    if _fed1 is not None:
        _fd_col, _fd_lvl, _fd_pct = _pct_color_level(
            _fed1, _fed1_hist,
            label_low='🟢 repo baixo',
            label_mid='🟡 repo moderado',
            label_hi='🔴 repo ATH — alavancagem extrema')
        # alto repo = mais alavancagem = stress (inverte)
        _fd_col = ('#f85149' if _fd_col == '#3fb950' else
                   '#3fb950' if _fd_col == '#f85149' else _fd_col)
        _fd_lvl = ('🔴 repo ATH — alavancagem extrema' if '🟢' in _fd_lvl else
                   '🟢 repo baixo'                     if '🔴' in _fd_lvl else _fd_lvl)
        _vvol_html += (
            f"<tr><td>FEDPSOR1 (PD Equity Repo)</td>"
            f"<td><b style='color:{_fd_col};'>{_fed1:.1f}B{_fd_pct}</b></td>"
            f"<td style='color:{_fd_col};'>{_fd_lvl}</td>"
            f"<td style='font-size:11px;'>Primary Dealer repo equities outstanding — ATH = funding squeeze sistêmico</td></tr>")

    # ES1 bid-ask spread
    if _es_ba is not None:
        _ba_col = '#f85149' if _es_ba > 2.0 else '#ffaa00' if _es_ba > 1.5 else '#3fb950'
        _ba_lvl = '🔴 liquidez ruim' if _es_ba > 2.0 else '🟡 spread moderado' if _es_ba > 1.5 else '🟢 liquidez ok'
        _vvol_html += (
            f"<tr><td>ES1 Bid-Ask Spread</td>"
            f"<td><b style='color:{_ba_col};'>{_es_ba:.2f} ticks</b></td>"
            f"<td style='color:{_ba_col};'>{_ba_lvl}</td>"
            f"<td style='font-size:11px;'>Spread bid-ask do futuro ES — >2 ticks = custo de trade elevado = stress de liquidez</td></tr>")

    # ── Vol-Control Fund Exposure (SPLV*TE) ──────────────────────────────────
    _splv_items = [
        ('splv5ute',  'SPLV5UTE',  '5%'),
        ('splv10te',  'SPLV10TE', '10%'),
        ('splv12te',  'SPLV12TE', '12%'),
        ('splv15te',  'SPLV15TE', '15%'),
    ]
    _splv_any = any(_vvol.get(f'{k}_cur') is not None for k, _, _ in _splv_items)
    if _splv_any:
        _vvol_html += (
            "<tr style='background:#0d1117;'>"
            "<td colspan='4' style='color:#58a6ff;font-weight:bold;padding:4px 0 2px;'>"
            "── Vol-Control Fund Exposure ──</td></tr>")
        # RV multi-window na mesma seção
        _rv_m = _vvol.get('rv_multi')
        if _rv_m:
            _rv_parts = ' | '.join(f"RV{w}D: {_rv_m[w]:.1f}%" for w in (10, 15, 21, 30) if w in _rv_m)
            _vvol_html += (
                f"<tr><td>RV SPX (multi-janela)</td>"
                f"<td colspan='2'><b style='color:#e6b430;'>{_rv_parts}</b></td>"
                f"<td style='font-size:11px;'>Volatilidade realizada — vol-ctrl funds reduzem equity quando RV sobe</td></tr>")
        for _splv_key, _splv_lbl, _splv_tgt in _splv_items:
            _cur = _vvol.get(f'{_splv_key}_cur')
            _hist = _vvol.get(f'{_splv_key}_hist')
            _pct = _vvol.get(f'{_splv_key}_pct')
            if _cur is None:
                continue
            # Exposição baixa = fundos leves = venda forçada iminente se RV sobe
            _pct_str = f' p{_pct:.0f}' if _pct is not None else ''
            if _pct is not None:
                if _pct <= 20:
                    _sc = '#f85149'; _sl = '🔴 exposição MÍNIMA — venda forçada iminente'
                elif _pct <= 40:
                    _sc = '#ffaa00'; _sl = '🟡 exposição baixa — buffer limitado'
                elif _pct >= 80:
                    _sc = '#3fb950'; _sl = '🟢 exposição MÁXIMA — fundos comprados'
                elif _pct >= 60:
                    _sc = '#3fb950'; _sl = '🟢 exposição elevada'
                else:
                    _sc = '#8b949e'; _sl = '— exposição neutra'
            else:
                _sc = '#8b949e'; _sl = '—'
            _vvol_html += (
                f"<tr><td>{_splv_lbl} (vol-ctrl {_splv_tgt})</td>"
                f"<td><b style='color:{_sc};'>{_cur:.1f}%{_pct_str}</b></td>"
                f"<td style='color:{_sc};'>{_sl}</td>"
                f"<td style='font-size:11px;'>Exposição equity do fundo vol-target {_splv_tgt} — baixo = deslavaancagem ao menor spike de RV</td></tr>")

    # LAGIDBMA — Conference Board margin level (ambos os extremos são risco)
    # Alto  → alavancagem excessiva → realizações forçadas se mercado cair
    # Baixo → guarda baixa / sem proteção
    _lag = _vvol.get('lagidbma_cur')
    _lag_hist = _vvol.get('lagidbma_hist')
    if _lag is not None:
        if _lag_hist is not None and len(_lag_hist) > 20:
            _lag_arr = np.asarray(_lag_hist.dropna())
            _lg_pct  = float(np.mean(_lag_arr < _lag))   # percentil histórico
            _lg_pct_str = f' p{_lg_pct*100:.0f}'
            # ambos os extremos = alerta
            if _lg_pct >= 0.80:
                _lg_col = '#f85149'
                _lg_lvl = '🔴 margin ALTA — risco de realizações forçadas'
            elif _lg_pct >= 0.60:
                _lg_col = '#ffaa00'
                _lg_lvl = '🟡 margin elevada — alavancagem crescente'
            elif _lg_pct <= 0.20:
                _lg_col = '#f85149'
                _lg_lvl = '🔴 margin BAIXA — guarda baixa / sem proteção'
            elif _lg_pct <= 0.40:
                _lg_col = '#ffaa00'
                _lg_lvl = '🟡 margin baixa — pouca alavancagem'
            else:
                _lg_col = '#3fb950'
                _lg_lvl = '🟢 margin neutra'
        else:
            _lg_pct_str = ''
            _lg_col, _lg_lvl = '#8b949e', '— sem histórico'
        _vvol_html += (
            f"<tr><td>LAGIDBMA (Conference Board Margin)</td>"
            f"<td><b style='color:{_lg_col};'>{_lag:.2f}{_lg_pct_str}</b></td>"
            f"<td style='color:{_lg_col};'>{_lg_lvl}</td>"
            f"<td style='font-size:11px;'>Margin alta → realizações forçadas se mercado cair; baixa → guarda baixa</td></tr>")

    # Se nenhum indicador disponível (nenhuma seção)
    _splv_any2 = any(_vvol.get(f'{k}_cur') is not None for k, _, _ in _splv_items)
    if all(v is None for v in [_vvix, _c25, _p25, _sdex, _tdex, _call_oi, _put_oi,
                                _axwa, _fed1, _es_ba, _lag]) and not _splv_any2:
        _vvol_html += "<tr><td colspan='4' style='color:#8b949e;'>Indicadores não disponíveis (falha no fetch BBG)</td></tr>"

    _vvol_html += "</table></div></div>"

    # SPX IV + Skew sparklines
    _spx_iv_chart = None
    _spx_iv_h = _vvol.get('spx_iv_hist')
    _spx_sk_h = _vvol.get('spx_skew_hist')
    if _spx_iv_h is not None and len(_spx_iv_h) > 10:
        _fig_iv = go.FigureWidget()
        _fig_iv.add_trace(go.Scatter(
            x=_spx_iv_h.index, y=_spx_iv_h.values,
            mode='lines', line=dict(color='#61afef', width=1.5),
            fill='tozeroy', fillcolor='rgba(97,175,239,0.08)', name='ATM IV'))
        if _spx_sk_h is not None and len(_spx_sk_h) > 10:
            _fig_iv.add_trace(go.Scatter(
                x=_spx_sk_h.index, y=_spx_sk_h.values,
                mode='lines', line=dict(color='#e5c07b', width=1.2, dash='dot'),
                name='Skew (ATM−90%)', yaxis='y2'))
            _fig_iv.update_layout(yaxis2=dict(
                overlaying='y', side='right', showgrid=False,
                tickfont=dict(size=8, color='#e5c07b'), title='Skew'))
        _fig_iv.update_layout(
            title='SPX 30d ATM IV + Skew (1Y)', height=180, template='plotly_dark',
            margin=dict(t=30, b=20, l=40, r=50),
            paper_bgcolor=_C['card'], plot_bgcolor=_C['card'],
            legend=dict(font=dict(size=9), x=0, y=1),
            yaxis_title='IV (%)')
        _spx_iv_chart = _fig_iv

    # VVIX history sparkline (se disponível)
    _vvix_hist = _vvol.get('vvix_hist')
    _vvix_chart = None
    if _vvix_hist is not None and len(_vvix_hist) > 10:
        _vvix_fig = go.FigureWidget()
        _vvix_fig.add_trace(go.Scatter(
            x=_vvix_hist.index, y=_vvix_hist.values,
            mode='lines', line=dict(color='#58a6ff', width=1.5),
            fill='tozeroy', fillcolor='rgba(88,166,255,0.08)',
            name='VVIX'))
        _vvix_fig.add_hline(y=130, line_dash='dash', line_color='#f85149', line_width=1,
                            annotation_text='130 Extreme', annotation_font_color='#f85149')
        _vvix_fig.add_hline(y=100, line_dash='dot',  line_color='#ffaa00', line_width=1,
                            annotation_text='100 Elevated', annotation_font_color='#ffaa00')
        _vvix_fig.update_layout(
            title='VVIX — Vol da Vol (1Y)', height=180, template='plotly_dark',
            margin=dict(t=30, b=20, l=40, r=10),
            paper_bgcolor=_C['card'], plot_bgcolor=_C['card'],
            showlegend=False, yaxis_title='VVIX')
        _vvix_chart = _vvix_fig

    # VRP sparkline
    _vrp_chart = None
    _vrp_hist2 = _vvol.get('vrp_hist')
    if _vrp_hist2 is not None and len(_vrp_hist2) > 10:
        _vrp_fig = go.FigureWidget()
        _vrp_fig.add_trace(go.Scatter(
            x=_vrp_hist2.index, y=_vrp_hist2.values,
            mode='lines', line=dict(color='#f0a500', width=1.5),
            fill='tozeroy', fillcolor='rgba(240,165,0,0.08)',
            name='VRP'))
        _vrp_fig.add_hline(y=0, line_dash='dash', line_color='#8b949e', line_width=1)
        _vrp_fig.update_layout(
            title='VRP = VIX − RV 10D SPX (1Y)', height=180, template='plotly_dark',
            margin=dict(t=30, b=20, l=40, r=10),
            paper_bgcolor=_C['card'], plot_bgcolor=_C['card'],
            showlegend=False, yaxis_title='Vol pts')
        _vrp_chart = _vrp_fig

    # AXWA funding sparkline
    _axwa_chart = None
    _axwa_hist2 = _vvol.get('axwa_hist')
    if _axwa_hist2 is not None and len(_axwa_hist2) > 10:
        _axwa_fig = go.FigureWidget()
        _axwa_fig.add_trace(go.Scatter(
            x=_axwa_hist2.index, y=_axwa_hist2.values,
            mode='lines', line=dict(color='#e06c75', width=1.5),
            fill='tozeroy', fillcolor='rgba(224,108,117,0.08)',
            name='AXWA'))
        _axwa_fig.update_layout(
            title='AXWA — SPX Funding Spread (1Y)', height=180, template='plotly_dark',
            margin=dict(t=30, b=20, l=40, r=10),
            paper_bgcolor=_C['card'], plot_bgcolor=_C['card'],
            showlegend=False, yaxis_title='Spread')
        _axwa_chart = _axwa_fig

    # FEDPSOR1 sparkline
    _fed1_chart = None
    _fed1_hist2 = _vvol.get('fedpsor1_hist')
    if _fed1_hist2 is not None and len(_fed1_hist2) > 10:
        _fed1_fig = go.FigureWidget()
        _fed1_fig.add_trace(go.Scatter(
            x=_fed1_hist2.index, y=_fed1_hist2.values,
            mode='lines', line=dict(color='#c678dd', width=1.5),
            fill='tozeroy', fillcolor='rgba(198,120,221,0.08)',
            name='PD Repo'))
        _fed1_fig.update_layout(
            title='FEDPSOR1 — Primary Dealer Equity Repo (1Y)', height=180,
            template='plotly_dark', margin=dict(t=30, b=20, l=40, r=10),
            paper_bgcolor=_C['card'], plot_bgcolor=_C['card'],
            showlegend=False, yaxis_title='$B')
        _fed1_chart = _fed1_fig

    children = [
        wd.HTML(summary_html),
        wd.HBox([gauge_fig, bar_fig],
                layout={'align_items': 'flex-start'}),
        wd.HTML(_vvol_html),
    ]
    # linha 0: SPX IV + Skew (contexto base)
    if _spx_iv_chart is not None:
        children.append(_spx_iv_chart)
    # linha 1: VVIX + VRP
    _row1 = [c for c in [_vvix_chart, _vrp_chart] if c is not None]
    if _row1:
        children.append(wd.HBox(_row1, layout={'align_items': 'flex-start'}))
    # linha 2: AXWA funding + PD Repo
    _row2 = [c for c in [_axwa_chart, _fed1_chart] if c is not None]
    if _row2:
        children.append(wd.HBox(_row2, layout={'align_items': 'flex-start'}))
    children.append(wd.HTML(_evts_html))
    return wd.VBox(children)


def build_squeeze_mini_panel(squeeze_result, _C):
    """Painel compacto do Gamma Squeeze para a Visão Geral (sem eventos históricos)."""
    score  = squeeze_result['score']
    alert  = squeeze_result['alert']
    interp = squeeze_result['interp']
    comps  = squeeze_result['components']
    alert_colors = {'critical': '#ff4444', 'warning': '#ffaa00',
                    'moderate': '#88aaff', 'low': '#3fb950'}
    ac = alert_colors.get(alert, _C['text'])
    alert_labels = {'critical': '🔴 CRÍTICO', 'warning': '🟡 ALERTA',
                    'moderate': '🔵 MODERADO', 'low': '🟢 BAIXO'}

    gauge_fig = go.FigureWidget(go.Indicator(
        mode='gauge+number', value=score,
        number={'font': {'color': ac, 'size': 30}, 'valueformat': '.0f'},
        title={'text': 'Gamma Squeeze Risk', 'font': {'color': _C['text_muted'], 'size': 12}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': _C['text_muted'],
                     'tickfont': {'color': _C['text_muted'], 'size': 8}},
            'bar': {'color': ac, 'thickness': 0.3},
            'bgcolor': _C['card2'],
            'steps': [
                {'range': [0, 35],   'color': '#1a3a2a'},
                {'range': [35, 55],  'color': '#3a3020'},
                {'range': [55, 75],  'color': '#3a2510'},
                {'range': [75, 100], 'color': '#3a1a1a'},
            ],
            'threshold': {'line': {'color': ac, 'width': 3},
                          'thickness': 0.75, 'value': score},
        }))
    gauge_fig.update_layout(
        height=210, width=240, template='plotly_dark',
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor=_C['card'], plot_bgcolor=_C['card'])

    bar_labels = [c['label'] for c in comps.values()]
    bar_scores = [c['score'] for c in comps.values()]
    bar_maxes  = [c['max']   for c in comps.values()]
    bar_colors = [ac if s / m > 0.6 else _C['accent']
                  for s, m in zip(bar_scores, bar_maxes)]
    bar_fig = go.FigureWidget()
    bar_fig.add_trace(go.Bar(
        y=bar_labels, x=bar_scores, orientation='h',
        marker_color=bar_colors,
        text=[f"{s:.0f}/{m}" for s, m in zip(bar_scores, bar_maxes)],
        textposition='outside'))
    bar_fig.update_layout(
        title=dict(text='Componentes do Score', font=dict(size=11, color=_C['text_muted'])),
        xaxis=dict(range=[0, 30], title='Score', tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
        height=210, template='plotly_dark',
        margin=dict(t=30, b=20, l=5, r=60),
        paper_bgcolor=_C['card'], plot_bgcolor=_C['card'], showlegend=False)

    badge_html = (
        f"<div style='text-align:center;padding:2px 0 0;'>"
        f"<span style='font-size:11px;color:{_C['text_muted']};'>{interp}</span>"
        f"</div>")
    # Retorna tupla: (gauge_widget, components_widget, badge_html_str, alert_color)
    return gauge_fig, bar_fig, badge_html, ac


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 7 — VISUALIZAÇÃO (Gauges, Gráficos, Tabelas)
# ═══════════════════════════════════════════════════════════════════════════════


import json as _de_json
import uuid as _de_uuid
from datetime import datetime as _de_dt, date as _de_date, timedelta as _de_td
from dataclasses import dataclass as _de_dc, field as _de_field, asdict as _de_asdict
from enum import Enum as _de_Enum

# ── DE constants ──────────────────────────────────────────────────────────────
_DE_ES_PV   = 50.0   # USD/point ES full
_DE_MES_PV  = 5.0    # USD/point MES micro
_DE_OPT_MULT= 100    # SPX option multiplier
_DE_OPEN    = (9, 30)
_DE_CLOSE   = (16, 0)
_DE_FLATTEN = 15     # min before close to flatten
_DE_LAST_EN = 30     # min before close to block new entries

# ── Enums ─────────────────────────────────────────────────────────────────────
class _DE_Regime(str, _de_Enum):
    DIRECTIONAL_LONG  = 'directional_long'
    DIRECTIONAL_SHORT = 'directional_short'
    NEUTRAL           = 'neutral'
    LONG_VOL          = 'long_vol'
    SHORT_VOL         = 'short_vol'
    NO_TRADE          = 'no_trade'

class _DE_StructureType(str, _de_Enum):
    LONG_CALL         = 'long_call'
    LONG_CALL_SPREAD  = 'long_call_spread'
    SHORT_CALL_SPREAD = 'short_call_spread'
    LONG_PUT          = 'long_put'
    LONG_PUT_SPREAD   = 'long_put_spread'
    SHORT_PUT_SPREAD  = 'short_put_spread'
    IRON_CONDOR       = 'iron_condor'
    CALL_FLY          = 'call_fly'
    IRON_BUTTERFLY    = 'iron_butterfly'
    PUT_FLY           = 'put_fly'
    LONG_STRADDLE     = 'long_straddle'
    ES_LONG           = 'es_long'
    ES_SHORT          = 'es_short'
    MES_LONG          = 'mes_long'
    MES_SHORT         = 'mes_short'
    NONE              = 'none'

_DE_REGIME_STRUCTS = {
    _DE_Regime.DIRECTIONAL_LONG:  [_DE_StructureType.LONG_CALL_SPREAD,  _DE_StructureType.LONG_CALL,        _DE_StructureType.SHORT_PUT_SPREAD, _DE_StructureType.ES_LONG,  _DE_StructureType.MES_LONG],
    _DE_Regime.DIRECTIONAL_SHORT: [_DE_StructureType.LONG_PUT_SPREAD,   _DE_StructureType.LONG_PUT,         _DE_StructureType.SHORT_CALL_SPREAD,_DE_StructureType.ES_SHORT, _DE_StructureType.MES_SHORT],
    _DE_Regime.NEUTRAL:           [_DE_StructureType.IRON_CONDOR,       _DE_StructureType.IRON_BUTTERFLY,   _DE_StructureType.CALL_FLY,         _DE_StructureType.PUT_FLY],
    _DE_Regime.LONG_VOL:          [_DE_StructureType.LONG_STRADDLE,     _DE_StructureType.LONG_CALL_SPREAD, _DE_StructureType.LONG_PUT_SPREAD],
    _DE_Regime.SHORT_VOL:         [_DE_StructureType.IRON_CONDOR,       _DE_StructureType.SHORT_PUT_SPREAD, _DE_StructureType.SHORT_CALL_SPREAD,_DE_StructureType.IRON_BUTTERFLY],
    _DE_Regime.NO_TRADE:          [],
}

# ── Dataclasses ───────────────────────────────────────────────────────────────
@_de_dc
class _DE_RiskConfig:
    max_risk_per_trade_pct:   float = 1.0
    max_daily_loss_pct:       float = 3.0
    max_total_exposure_pct:   float = 15.0
    max_margin_usage_pct:     float = 50.0
    reserve_cash_pct:         float = 10.0
    max_positions_open:       int   = 4
    max_contracts_per_trade:  int   = 10
    min_cash_buffer:          float = 5000.0
    min_confidence_to_trade:  float = 0.55
    last_entry_min_before_close: int = _DE_LAST_EN
    flatten_min_before_close: int   = _DE_FLATTEN
    paper_mode:               bool  = True
    force_override:           bool  = False   # ignora trava NO_TRADE e paper_mode

@_de_dc
class _DE_AccountState:
    net_liquidation:  float = 100_000.0
    available_cash:   float = 100_000.0
    buying_power:     float = 200_000.0
    available_margin: float = 100_000.0
    unrealized_pnl:   float = 0.0
    realized_pnl_day: float = 0.0
    margin_used:      float = 0.0
    risk_used_day:    float = 0.0
    source:           str   = 'manual'

@_de_dc
class _DE_Leg:
    instrument:  str
    side:        int
    option_type: object = None
    strike:      object = None
    iv:          object = None
    tte:         object = None
    px:          object = None
    delta:       float  = 0.0
    gamma:       float  = 0.0
    vega:        float  = 0.0
    theta:       float  = 0.0
    vanna:       float  = 0.0
    charm:       float  = 0.0
    multiplier:  float  = 100.0

@_de_dc
class _DE_Structure:
    structure_type:    object
    legs:              object
    net_debit:         float = 0.0
    max_loss:          float = 0.0
    max_gain:          float = 0.0
    net_delta:         float = 0.0
    net_gamma:         float = 0.0
    net_vega:          float = 0.0
    net_theta:         float = 0.0
    net_vanna:         float = 0.0
    net_charm:         float = 0.0
    estimated_margin:  float = 0.0

@_de_dc
class _DE_TradeDecision:
    decision_id:    str   = _de_field(default_factory=lambda: str(_de_uuid.uuid4())[:8])
    timestamp:      str   = _de_field(default_factory=lambda: _de_dt.utcnow().isoformat())
    action:         str   = 'no_trade'
    instrument:     str   = 'ES'
    structure:      object = _de_field(default_factory=lambda: _DE_StructureType.NONE)
    regime:         object = _de_field(default_factory=lambda: _DE_Regime.NO_TRADE)
    confidence:     float = 0.0
    regime_proba:   dict  = _de_field(default_factory=dict)
    entry_price:    object = None
    stop_loss:      object = None
    take_profit:    object = None
    quantity:       int   = 0
    expiry:         object = None
    strikes:        dict  = _de_field(default_factory=dict)
    rationale:      str   = ''
    risk_metrics:   dict  = _de_field(default_factory=dict)
    flatten_time:   object = None
    execution_ready: bool = False
    block_reason:   object = None
    capital_available:          float = 0.0
    margin_available:           float = 0.0
    risk_budget_trade:          float = 0.0
    risk_budget_day_remaining:  float = 0.0
    estimated_trade_cost:       float = 0.0
    estimated_max_loss:         float = 0.0
    estimated_margin_usage:     float = 0.0
    allowed_size:               int   = 0
    size_block_reason:          object = None
    structure_obj:              object = None

# ── MarketState ───────────────────────────────────────────────────────────────
class _DE_MarketState:
    def __init__(self, ticker, spot, df):
        self.ticker = ticker; self.spot = spot; self.df = df
        self.features = {}

    def compute(self):
        f = {}
        df = self.df; spot = self.spot
        if df is None or df.empty:
            return f
        is_call = df['Type'] == 'Call'; is_put = df['Type'] == 'Put'
        K = df['Strike'].values; iv = df['IV'].values
        oi = df['OI'].values;    tte = df['Tte'].values

        # ATM IV
        atm_idx = int(np.argmin(np.abs(K - spot)))
        f['atm_iv_0dte'] = float(iv[atm_idx]) if len(iv) > 0 else 0.15

        # 0DTE skew
        zero_mask = tte < (2.0 / TRADING_DAYS)
        if zero_mask.sum() > 4:
            df0 = df[zero_mask].copy()
            try:
                g0 = calculate_all_greeks(spot, df0['Strike'].values,
                                          df0['IV'].values, df0['Tte'].values,
                                          df0['Type'].values)
                c_mask = df0['Type'].values == 'Call'
                p_mask = df0['Type'].values == 'Put'
                if g0 and c_mask.sum() > 0 and p_mask.sum() > 0:
                    c_d = np.abs(g0['delta'][c_mask])
                    p_d = np.abs(g0['delta'][p_mask])
                    iv0v = df0['IV'].values
                    c25 = float(iv0v[c_mask][np.argmin(np.abs(c_d - 0.25))]) if c_mask.sum() > 0 else f['atm_iv_0dte']
                    p25 = float(iv0v[p_mask][np.argmin(np.abs(p_d - 0.25))]) if p_mask.sum() > 0 else f['atm_iv_0dte']
                    f['skew_0dte'] = p25 - c25
                    f['iv_25d_call_0dte'] = c25; f['iv_25d_put_0dte'] = p25
            except Exception:
                f['skew_0dte'] = 0.0
        else:
            f['skew_0dte'] = 0.0

        # OI
        oi_c = np.where(is_call, oi, 0); oi_p = np.where(is_put, oi, 0)
        tot  = oi.sum() + 1e-9
        f['total_oi']    = float(oi.sum())
        f['oi_call_pct'] = float(oi_c.sum() / tot)
        f['pc_oi_ratio'] = float(oi_p.sum() / (oi_c.sum() + 1e-9))
        atm_mask = np.abs(K - spot) <= spot * 0.01
        f['atm_oi_pct']  = float(oi[atm_mask].sum() / tot)

        # Walls via existing dashboard function
        try:
            greeks_all = calculate_all_greeks(spot, K, iv, tte, df['Type'].values)
            agg = compute_strike_exposures(df, greeks_all, spot)
            cw, pw = compute_walls(agg)
            f['call_wall'] = float(cw) if cw else spot * 1.02
            f['put_wall']  = float(pw) if pw else spot * 0.98
            f['dist_to_call_wall'] = (f['call_wall'] - spot) / spot
            f['dist_to_put_wall']  = (spot - f['put_wall'])  / spot
            f['wall_range']        = (f['call_wall'] - f['put_wall']) / spot
            f['net_gex'] = float((agg.get('Call_gamma', pd.Series([0])).sum()
                                   - agg.get('Put_gamma',  pd.Series([0])).sum()))
        except Exception:
            f['call_wall'] = spot * 1.02; f['put_wall'] = spot * 0.98
            f['dist_to_call_wall'] = 0.02; f['dist_to_put_wall'] = 0.02
            f['wall_range'] = 0.04; f['net_gex'] = 0.0
        f.setdefault('net_vanna', 0.0); f.setdefault('net_charm', 0.0)

        # Time
        now = _de_dt.now()
        open_t  = now.replace(hour=_DE_OPEN[0],  minute=_DE_OPEN[1],  second=0, microsecond=0)
        close_t = now.replace(hour=_DE_CLOSE[0], minute=_DE_CLOSE[1], second=0, microsecond=0)
        elapsed = max(0, (now - open_t).total_seconds() / 60)
        remain  = max(0, (close_t - now).total_seconds() / 60)
        total   = (close_t - open_t).total_seconds() / 60
        f['time_of_day_pct']  = float(elapsed / (total + 1e-9))
        f['minutes_to_close'] = float(remain)
        f['minutes_elapsed']  = float(elapsed)
        f['theta_compression']= float(np.exp(-remain / 60.0))
        self.features = f
        return f

# ── DecisionModel (heuristic — no external ML lib required) ───────────────────
class _DE_DecisionModel:
    FEATURE_NAMES = [
        'atm_iv_0dte','skew_0dte','pc_oi_ratio','atm_oi_pct',
        'dist_to_call_wall','dist_to_put_wall','wall_range',
        'net_gex','net_vanna','net_charm',
        'time_of_day_pct','minutes_to_close','theta_compression',
        'flow_score','squeeze_score','tail_score','iv_rv_spread','skew_level',
    ]

    def predict(self, f, ext):
        flow   = ext.get('flow_score', 50)
        squeeze= ext.get('squeeze_score', 0)
        tail   = ext.get('tail_score', 0)
        iv_rv  = ext.get('iv_rv_spread', 0)
        skew   = f.get('skew_0dte', 0)
        gex    = f.get('net_gex', 0)
        min_cl = f.get('minutes_to_close', 390)

        sc = {r: 0.0 for r in _DE_Regime}
        if flow > 65:    sc[_DE_Regime.DIRECTIONAL_LONG]  += 0.30
        elif flow < 35:  sc[_DE_Regime.DIRECTIONAL_SHORT] += 0.30
        if gex > 0:
            sc[_DE_Regime.NEUTRAL]   += 0.20
            sc[_DE_Regime.SHORT_VOL] += 0.15
        else:
            sc[_DE_Regime.LONG_VOL]  += 0.20
        if iv_rv > 3:    sc[_DE_Regime.SHORT_VOL]         += 0.25
        elif iv_rv < -3: sc[_DE_Regime.LONG_VOL]          += 0.25
        if squeeze > 70: sc[_DE_Regime.LONG_VOL]          += 0.20
        if tail > 70:    sc[_DE_Regime.NO_TRADE]           += 0.40
        if skew > 0.03:  sc[_DE_Regime.DIRECTIONAL_SHORT] += 0.10
        if min_cl < _DE_LAST_EN:
            sc[_DE_Regime.NO_TRADE] += 1.0

        total = sum(sc.values()) or 1.0
        proba = {k.value: v / total for k, v in sc.items()}
        best  = max(sc, key=sc.get)
        conf  = sc[best] / total
        return best, conf, proba

# ── StrategySelector ──────────────────────────────────────────────────────────
class _DE_StrategySelector:
    def __init__(self, df, spot, rfr=0.05, minutes_to_close=390):
        self.df = df; self.spot = spot; self.rfr = rfr
        self.mtc = minutes_to_close
        self.df0 = df[df['Tte'] < 2.0 / TRADING_DAYS].copy() if df is not None and not df.empty else pd.DataFrame()

    def _atm(self):
        K = self.df0['Strike'].values if not self.df0.empty else np.array([self.spot])
        return float(K[np.argmin(np.abs(K - self.spot))])

    def _near(self, t):
        K = self.df0['Strike'].values if not self.df0.empty else np.array([self.spot])
        return float(K[np.argmin(np.abs(K - t))])

    def _get_iv(self, k, t):
        if self.df0.empty: return 0.15
        sub = self.df0[(np.abs(self.df0['Strike'] - k) < 1e-3) &
                       (self.df0['Type'] == ('Call' if t == 'C' else 'Put'))]
        return float(sub['IV'].values[0]) if not sub.empty else 0.15

    def _tte(self):
        return float(self.df0['Tte'].mean()) if not self.df0.empty else 0.5 / TRADING_DAYS

    def _price_leg(self, strike, opt_type, sign):
        iv  = self._get_iv(strike, opt_type)
        tte = self._tte()
        try:
            px = float(black_scholes_price_vec(
                self.spot, np.array([strike]), np.array([iv]),
                np.array([tte]), np.array([opt_type]), r=self.rfr)[0])
        except Exception:
            px = 0.0
        try:
            g = calculate_all_greeks(self.spot, np.array([strike]), np.array([iv]),
                                     np.array([tte]), np.array([opt_type]), r=self.rfr)
        except Exception:
            g = {}
        return _DE_Leg(
            instrument='option', side=sign, option_type=opt_type,
            strike=strike, iv=iv, tte=tte, px=px,
            delta=float(g.get('delta', [0])[0]) if g else 0.0,
            gamma=float(g.get('gamma', [0])[0]) if g else 0.0,
            vega =float(g.get('vega',  [0])[0]) if g else 0.0,
            theta=float(g.get('theta', [0])[0]) if g else 0.0,
            vanna=float(g.get('vanna', [0])[0]) if g else 0.0,
            charm=float(g.get('charm', [0])[0]) if g else 0.0,
            multiplier=_DE_OPT_MULT)

    def _build(self, legs, stype):
        nd = sum(l.side * (l.px or 0) * l.multiplier for l in legs)
        # max_loss
        if stype in (_DE_StructureType.LONG_CALL, _DE_StructureType.LONG_PUT,
                     _DE_StructureType.LONG_STRADDLE,
                     _DE_StructureType.LONG_CALL_SPREAD, _DE_StructureType.LONG_PUT_SPREAD,
                     _DE_StructureType.CALL_FLY, _DE_StructureType.PUT_FLY,
                     _DE_StructureType.IRON_BUTTERFLY):
            ml = abs(nd)
        elif stype in (_DE_StructureType.SHORT_CALL_SPREAD, _DE_StructureType.SHORT_PUT_SPREAD):
            sk = sorted([l.strike for l in legs if l.strike])
            ml = abs(sk[-1] - sk[0]) * _DE_OPT_MULT - abs(nd) if len(sk) >= 2 else abs(nd)
        elif stype == _DE_StructureType.IRON_CONDOR:
            cl = [l for l in legs if l.option_type == 'C']
            pl = [l for l in legs if l.option_type == 'P']
            cw = abs(cl[0].strike - cl[1].strike) * _DE_OPT_MULT if len(cl) == 2 else 500
            pw = abs(pl[0].strike - pl[1].strike) * _DE_OPT_MULT if len(pl) == 2 else 500
            ml = max(cw, pw) - abs(nd)
        else:
            ml = abs(nd) * 2
        return _DE_Structure(
            structure_type=stype, legs=legs, net_debit=nd, max_loss=max(ml, 0.01),
            net_delta=sum(l.side*l.delta for l in legs),
            net_gamma=sum(l.side*l.gamma for l in legs),
            net_vega =sum(l.side*l.vega  for l in legs),
            net_theta=sum(l.side*l.theta for l in legs),
            estimated_margin=max(ml, 0.01))

    def build(self, stype):
        atm = self._atm()
        sd  = (self.spot * self._get_iv(atm, 'C') *
               np.sqrt(max(self.mtc / 390.0, 0.01) / TRADING_DAYS))
        oc1 = self._near(self.spot + sd * 0.5)
        oc2 = self._near(self.spot + sd * 1.0)
        op1 = self._near(self.spot - sd * 0.5)
        op2 = self._near(self.spot - sd * 1.0)
        try:
            if stype == _DE_StructureType.LONG_CALL:
                legs = [self._price_leg(atm, 'C', +1)]
            elif stype == _DE_StructureType.LONG_PUT:
                legs = [self._price_leg(atm, 'P', +1)]
            elif stype == _DE_StructureType.LONG_STRADDLE:
                legs = [self._price_leg(atm,'C',+1), self._price_leg(atm,'P',+1)]
            elif stype == _DE_StructureType.LONG_CALL_SPREAD:
                legs = [self._price_leg(atm,'C',+1), self._price_leg(oc1,'C',-1)]
            elif stype == _DE_StructureType.LONG_PUT_SPREAD:
                legs = [self._price_leg(atm,'P',+1), self._price_leg(op1,'P',-1)]
            elif stype == _DE_StructureType.SHORT_CALL_SPREAD:
                legs = [self._price_leg(oc1,'C',-1), self._price_leg(oc2,'C',+1)]
            elif stype == _DE_StructureType.SHORT_PUT_SPREAD:
                legs = [self._price_leg(op1,'P',-1), self._price_leg(op2,'P',+1)]
            elif stype == _DE_StructureType.IRON_CONDOR:
                legs = [self._price_leg(op1,'P',-1), self._price_leg(op2,'P',+1),
                        self._price_leg(oc1,'C',-1), self._price_leg(oc2,'C',+1)]
            elif stype == _DE_StructureType.IRON_BUTTERFLY:
                legs = [self._price_leg(op1,'P',+1), self._price_leg(atm,'P',-1),
                        self._price_leg(atm,'C',-1), self._price_leg(oc1,'C',+1)]
            elif stype == _DE_StructureType.CALL_FLY:
                legs = [self._price_leg(atm,'C',+1), self._price_leg(oc1,'C',-2),
                        self._price_leg(oc2,'C',+1)]
            elif stype == _DE_StructureType.PUT_FLY:
                legs = [self._price_leg(atm,'P',+1), self._price_leg(op1,'P',-2),
                        self._price_leg(op2,'P',+1)]
            else:
                return None
            return self._build(legs, stype)
        except Exception:
            return None

    def select_best(self, regime, confidence, cfg):
        candidates = _DE_REGIME_STRUCTS.get(regime, [])
        if self.mtc < 60:
            simple = {_DE_StructureType.LONG_CALL, _DE_StructureType.LONG_PUT,
                      _DE_StructureType.LONG_CALL_SPREAD, _DE_StructureType.LONG_PUT_SPREAD,
                      _DE_StructureType.ES_LONG, _DE_StructureType.ES_SHORT,
                      _DE_StructureType.MES_LONG, _DE_StructureType.MES_SHORT}
            candidates = [c for c in candidates if c in simple]
        if confidence < 0.65:
            cx = {_DE_StructureType.IRON_CONDOR, _DE_StructureType.CALL_FLY,
                  _DE_StructureType.PUT_FLY,     _DE_StructureType.IRON_BUTTERFLY}
            candidates = [c for c in candidates if c not in cx]
        for stype in candidates:
            if stype in (_DE_StructureType.ES_LONG,  _DE_StructureType.ES_SHORT,
                         _DE_StructureType.MES_LONG, _DE_StructureType.MES_SHORT):
                return None, stype
            s = self.build(stype)
            if s and s.max_loss > 0:
                return s, stype
        return None, _DE_StructureType.NONE

# ── RiskEngine ────────────────────────────────────────────────────────────────
class _DE_RiskEngine:
    def __init__(self, cfg, acc):
        self.cfg = cfg; self.acc = acc

    @property
    def _cap(self):   return self.acc.net_liquidation
    @property
    def _rb_trade(self): return self._cap * self.cfg.max_risk_per_trade_pct / 100
    @property
    def _rb_day(self):   return self._cap * self.cfg.max_daily_loss_pct / 100
    @property
    def _rb_day_rem(self):
        return max(0.0, self._rb_day - abs(min(0.0, self.acc.realized_pnl_day)))

    def size(self, struct, stop_pts, instrument='option', minutes_to_close=390):
        if instrument in ('ES', 'MES'):
            pv = _DE_ES_PV if instrument == 'ES' else _DE_MES_PV
            ml_lot = abs(stop_pts) * pv
            mg_lot = ml_lot * 2; cost_lot = ml_lot
        elif struct:
            ml_lot  = abs(struct.max_loss)
            mg_lot  = abs(struct.estimated_margin)
            cost_lot= abs(struct.net_debit) if struct.net_debit > 0 else mg_lot
        else:
            return {'allowed_size': 0, 'block_reason': 'no structure'}

        if ml_lot <= 0:
            return {'allowed_size': 0, 'block_reason': 'max_loss=0',
                    'capital': self._cap, 'risk_budget_trade': self._rb_trade,
                    'risk_budget_day_remaining': self._rb_day_rem,
                    'available_margin': self.acc.available_margin,
                    'available_cash': self.acc.available_cash,
                    'estimated_trade_cost': 0, 'estimated_max_loss': 0, 'estimated_margin': 0}

        n_rt = int(self._rb_trade        / ml_lot)
        n_rd = int(self._rb_day_rem      / ml_lot)
        n_mg = int(self.acc.available_margin / (mg_lot + 1e-9))
        n_ca = int((self.acc.available_cash - self.cfg.min_cash_buffer) / (cost_lot + 1e-9))
        n_mx = self.cfg.max_contracts_per_trade
        final = max(0, min(n_rt, n_rd, n_mg, n_ca, n_mx))
        block = None
        if final == 0:
            if   n_rd == 0: block = 'Budget diário esgotado'
            elif n_rt == 0: block = 'Budget por trade insuficiente'
            elif n_mg == 0: block = 'Margem insuficiente'
            elif n_ca == 0: block = 'Caixa insuficiente'
        return {'allowed_size': final, 'block_reason': block,
                'capital': self._cap, 'risk_budget_trade': self._rb_trade,
                'risk_budget_day_remaining': self._rb_day_rem,
                'available_margin': self.acc.available_margin,
                'available_cash': self.acc.available_cash,
                'estimated_trade_cost': final * cost_lot,
                'estimated_max_loss':   final * ml_lot,
                'estimated_margin':     final * mg_lot}

    def validate(self, decision, struct, minutes_to_close):
        # Sempre calcula sizing primeiro — garante que campos de capital ficam populados
        stop_pts = abs((decision.entry_price or 0) - (decision.stop_loss or 0))
        sz = self.size(struct, stop_pts, decision.instrument, minutes_to_close)
        decision.allowed_size              = sz['allowed_size']
        decision.size_block_reason         = sz.get('block_reason')
        decision.estimated_trade_cost      = sz['estimated_trade_cost']
        decision.estimated_max_loss        = sz['estimated_max_loss']
        decision.estimated_margin_usage    = sz['estimated_margin']
        decision.capital_available         = sz['capital']
        decision.margin_available          = sz['available_margin']
        decision.risk_budget_trade         = sz['risk_budget_trade']
        decision.risk_budget_day_remaining = sz['risk_budget_day_remaining']
        # flatten time
        now   = _de_dt.now()
        close = now.replace(hour=_DE_CLOSE[0], minute=_DE_CLOSE[1], second=0, microsecond=0)
        ft    = close - _de_td(minutes=self.cfg.flatten_min_before_close)
        decision.flatten_time = ft.strftime('%H:%M ET')

        # Hard blocks — ignorados quando force_override está ativo
        if not self.cfg.force_override:
            if self.acc.available_cash < self.cfg.min_cash_buffer:
                decision.block_reason = f'Caixa abaixo do buffer: ${self.acc.available_cash:,.0f}'
                decision.execution_ready = False; return decision
            if self._rb_day_rem <= 0:
                decision.block_reason = 'Budget diário esgotado'
                decision.execution_ready = False; return decision
            if minutes_to_close < self.cfg.last_entry_min_before_close:
                decision.block_reason = f'Muito perto do fechamento: {minutes_to_close:.0f} min'
                decision.execution_ready = False; return decision

        decision.execution_ready = (sz['allowed_size'] > 0
                                    and (not self.cfg.paper_mode or self.cfg.force_override)
                                    and decision.stop_loss is not None)
        return decision

# ── Orchestrator ──────────────────────────────────────────────────────────────
class _DE_Orchestrator:
    def __init__(self, ticker, spot, rfr, cfg, acc):
        self.ticker = ticker; self.spot = spot; self.rfr = rfr
        self.cfg = cfg; self.acc = acc
        self.model = _DE_DecisionModel()
        self._last: object = None

    def run(self, df, ext=None):
        ext = ext or {}
        now = _de_dt.now()
        open_t  = now.replace(hour=_DE_OPEN[0],  minute=_DE_OPEN[1],  second=0, microsecond=0)
        close_t = now.replace(hour=_DE_CLOSE[0], minute=_DE_CLOSE[1], second=0, microsecond=0)
        mtc = max(0, (close_t - now).total_seconds() / 60)

        ms       = _DE_MarketState(self.ticker, self.spot, df)
        features = ms.compute()
        features['minutes_to_close'] = mtc

        regime, conf, proba = self.model.predict(features, ext)
        d = _DE_TradeDecision(regime=regime, confidence=conf, regime_proba=proba)

        if regime == _DE_Regime.NO_TRADE or conf < self.cfg.min_confidence_to_trade:
            if not self.cfg.force_override:
                d.action = 'no_trade'
                d.block_reason = ('Regime: no_trade' if regime == _DE_Regime.NO_TRADE
                                  else f'Confiança {conf:.1%} < {self.cfg.min_confidence_to_trade:.1%}')
                self._last = d; return d
            else:
                # Override: proba tem chaves string (k.value), filtrar por .value
                _no_val = _DE_Regime.NO_TRADE.value
                _alt = {r: p for r, p in proba.items() if r != _no_val}
                if _alt:
                    _best_str = max(_alt, key=_alt.get)
                    regime = _DE_Regime(_best_str)   # reconverte para enum
                    conf   = _alt[_best_str]
                    d.regime = regime; d.confidence = conf
                d.block_reason = '[OVERRIDE] original: no_trade'

        sel = _DE_StrategySelector(df, self.spot, self.rfr, mtc)
        struct, stype = sel.select_best(regime, conf, self.cfg)
        d.structure = stype; d.structure_obj = struct
        d.instrument = ('ES'  if stype in (_DE_StructureType.ES_LONG,  _DE_StructureType.ES_SHORT)  else
                        'MES' if stype in (_DE_StructureType.MES_LONG, _DE_StructureType.MES_SHORT) else
                        'SPX_OPT')
        d.action = ('buy'  if regime in (_DE_Regime.DIRECTIONAL_LONG,  _DE_Regime.LONG_VOL)  else
                    'sell' if regime in (_DE_Regime.DIRECTIONAL_SHORT, _DE_Regime.SHORT_VOL) else 'buy')

        if struct:
            # Custo líquido da estrutura em pontos (sempre positivo = prêmio pago/recebido)
            ep = abs(struct.net_debit) / _DE_OPT_MULT
            # Stop = perde 50% do prêmio pago (long) ou lucra 50% (short/crédito)
            # Alvo = ganha 100% do prêmio ou perde 50% (crédito)
            _is_debit = (struct.net_debit >= 0)   # long/debit structure
            sp = round(ep * 0.50, 2) if _is_debit else round(ep * 1.50, 2)  # stop sempre < entry para long
            tp = round(ep * 2.00, 2) if _is_debit else round(ep * 0.25, 2)  # alvo > entry para long
            d.entry_price = round(ep, 2); d.stop_loss = sp; d.take_profit = tp
            d.strikes = {f'leg{i+1}': l.strike for i, l in enumerate(struct.legs) if l.strike}
            d.risk_metrics = {'net_delta': struct.net_delta, 'net_gamma': struct.net_gamma,
                              'net_vega': struct.net_vega, 'net_theta': struct.net_theta}
            try:
                exp_rows = df[df['Tte'] < 2.0 / TRADING_DAYS]
                d.expiry = str(exp_rows['Exp'].min()) if not exp_rows.empty else ''
            except Exception:
                d.expiry = ''
        elif stype in (_DE_StructureType.ES_LONG, _DE_StructureType.ES_SHORT,
                       _DE_StructureType.MES_LONG, _DE_StructureType.MES_SHORT):
            sp_pts = features.get('atm_iv_0dte', 0.01) * self.spot * 0.5
            d.entry_price = round(self.spot, 2)
            d.stop_loss   = round(self.spot - sp_pts if 'LONG' in stype.value.upper() else self.spot + sp_pts, 2)
            d.take_profit = round(self.spot + sp_pts * 2 if 'LONG' in stype.value.upper() else self.spot - sp_pts * 2, 2)

        d.rationale = (
            f"Regime={regime.value} conf={conf:.1%} | "
            f"Flow={ext.get('flow_score',0):.0f} Sq={ext.get('squeeze_score',0):.0f} "
            f"Tail={ext.get('tail_score',0):.0f} | "
            f"ATM_IV={features.get('atm_iv_0dte',0)*100:.1f}% "
            f"Skew={features.get('skew_0dte',0)*100:.1f}pp "
            f"GEX={features.get('net_gex',0):.1e} | "
            f"Walls call={features.get('call_wall',self.spot):.0f} "
            f"put={features.get('put_wall',self.spot):.0f}"
        )

        risk = _DE_RiskEngine(self.cfg, self.acc)
        sz = risk.size(struct, abs((d.entry_price or 0) - (d.stop_loss or 0)),
                       d.instrument, mtc)
        d.quantity = sz['allowed_size']
        d = risk.validate(d, struct, mtc)
        self._last = d
        return d

    def execute_paper(self):
        if not self._last: return {'status': 'no_decision'}
        return {'status': 'paper_logged', 'decision_id': self._last.decision_id,
                'structure': self._last.structure.value,
                'quantity': self._last.quantity, 'entry': self._last.entry_price,
                'stop': self._last.stop_loss, 'target': self._last.take_profit,
                'timestamp': _de_dt.utcnow().isoformat()}


# ── Tab builder ───────────────────────────────────────────────────────────────
def _build_decision_engine_tab_inline(df, spot, rfr, ticker, external_scores=None):
    """Tab 14 — Decision Engine 0DTE (fully inline, no external file needed)."""
    from IPython.display import display as _disp, HTML as _HTML
    ext = external_scores or {}

    # ── Config widgets ────────────────────────────────────────────────────────
    w_cash   = wd.FloatText(value=100_000, description='Cash ($):',
                            layout=wd.Layout(width='200px'), style={'description_width':'80px'})
    w_nlv    = wd.FloatText(value=100_000, description='NLV ($):',
                            layout=wd.Layout(width='200px'), style={'description_width':'80px'})
    w_bp     = wd.FloatText(value=200_000, description='Buying Power:',
                            layout=wd.Layout(width='220px'), style={'description_width':'100px'})
    w_margin = wd.FloatText(value=100_000, description='Avail Margin:',
                            layout=wd.Layout(width='220px'), style={'description_width':'100px'})
    w_rpnl   = wd.FloatText(value=0,       description='Real PnL $:',
                            layout=wd.Layout(width='200px'), style={'description_width':'80px'})
    w_risk   = wd.FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1,
                              description='Risk/Trade %:', readout_format='.1f',
                              layout=wd.Layout(width='300px'), style={'description_width':'100px'})
    w_daily  = wd.FloatSlider(value=3.0, min=0.5, max=10.0, step=0.5,
                              description='Daily Loss %:', readout_format='.1f',
                              layout=wd.Layout(width='300px'), style={'description_width':'100px'})
    w_maxpos = wd.IntSlider(value=4, min=1, max=10, description='Max Pos:',
                            layout=wd.Layout(width='260px'), style={'description_width':'80px'})
    w_paper  = wd.ToggleButton(value=True, description='PAPER MODE ON',
                               button_style='warning', icon='shield',
                               layout=wd.Layout(width='160px', height='36px'))
    w_run    = wd.Button(description='▶ Gerar Decisão', button_style='primary',
                         layout=wd.Layout(width='160px', height='36px'))
    w_pex    = wd.Button(description='📋 Paper Execute', button_style='warning',
                         layout=wd.Layout(width='150px', height='36px'))
    w_override = wd.Button(
        description='🔒 Trava ON',
        button_style='',
        layout=wd.Layout(width='130px', height='36px'))
    _override_state = [False]  # estado manual (Button não tem .value bool)

    out_d = wd.Output()
    orch  = [None]

    def _make_orch():
        cfg = _DE_RiskConfig(
            max_risk_per_trade_pct=w_risk.value,
            max_daily_loss_pct=w_daily.value,
            max_positions_open=w_maxpos.value,
            paper_mode=w_paper.value,
            force_override=_override_state[0])
        acc = _DE_AccountState(
            net_liquidation=w_nlv.value, available_cash=w_cash.value,
            buying_power=w_bp.value, available_margin=w_margin.value,
            realized_pnl_day=w_rpnl.value)
        return _DE_Orchestrator(ticker, spot, rfr, cfg, acc)

    def _render(d):
        ac = ('#00ff99' if d.action == 'buy' else
              '#ff4444' if d.action == 'sell' else '#aaaaaa')
        bc = '#ff4444' if d.block_reason else '#00ff99'
        cb = int(d.confidence * 180)
        rb = ''.join(
            f"<div style='display:flex;align-items:center;margin:1px 0;'>"
            f"<span style='color:#aaa;font-size:9px;width:130px;'>{k}</span>"
            f"<div style='background:#00d4e8;height:7px;width:{int(v*110)}px;border-radius:2px;'></div>"
            f"<span style='color:#aaa;font-size:9px;margin-left:4px;'>{v*100:.0f}%</span></div>"
            for k, v in sorted(d.regime_proba.items(), key=lambda x: -x[1]))
        sk = ''.join(f"<b>{k}</b>: {v:.0f} &nbsp;" for k, v in d.strikes.items())
        return f"""
<div style='background:#0d1520;border:1px solid rgba(0,212,232,.25);border-radius:8px;
            padding:16px;font-family:monospace;margin:6px 0;'>
  <div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px;'>
    <div style='background:#1a2035;border-left:4px solid {ac};padding:10px 18px;border-radius:6px;min-width:160px;'>
      <div style='color:#aaa;font-size:9px;'>DECISÃO</div>
      <div style='color:{ac};font-size:20px;font-weight:bold;'>{d.action.upper()}</div>
      <div style='color:#aaa;font-size:10px;'>{d.instrument} · {d.structure.value}</div>
    </div>
    <div style='background:#1a2035;padding:10px 18px;border-radius:6px;min-width:160px;'>
      <div style='color:#aaa;font-size:9px;'>CONFIANÇA</div>
      <div style='color:#fff;font-size:18px;font-weight:bold;'>{d.confidence*100:.1f}%</div>
      <div style='background:#333;height:7px;width:160px;border-radius:4px;margin-top:4px;'>
        <div style='background:#00d4e8;height:7px;width:{cb}px;border-radius:4px;'></div></div>
    </div>
    <div style='background:#1a2035;padding:10px 18px;border-radius:6px;min-width:180px;'>
      <div style='color:#aaa;font-size:9px;'>PREÇOS</div>
      <div style='font-size:11px;line-height:1.8;'>
        <span style='color:#aaa;'>Entrada:</span> <b style='color:#fff;'>{d.entry_price}</b><br>
        <span style='color:#aaa;'>Stop:</span> <b style='color:#ff4444;'>{d.stop_loss}</b><br>
        <span style='color:#aaa;'>Alvo:</span> <b style='color:#00ff99;'>{d.take_profit}</b>
      </div>
    </div>
    <div style='background:#1a2035;padding:10px 18px;border-radius:6px;min-width:180px;'>
      <div style='color:#aaa;font-size:9px;'>SIZING & RISCO</div>
      <div style='font-size:11px;line-height:1.8;'>
        <span style='color:#aaa;'>Qtde:</span> <b style='color:#00d4e8;'>{d.quantity}</b><br>
        <span style='color:#aaa;'>Perda máx:</span> <b style='color:#ff6b35;'>${d.estimated_max_loss:,.0f}</b><br>
        <span style='color:#aaa;'>Custo est.:</span> <b>${d.estimated_trade_cost:,.0f}</b>
      </div>
    </div>
    <div style='background:#1a2035;padding:10px 18px;border-radius:6px;min-width:180px;'>
      <div style='color:#aaa;font-size:9px;'>CAPITAL</div>
      <div style='font-size:11px;line-height:1.8;'>
        <span style='color:#aaa;'>NLV:</span> <b>${d.capital_available:,.0f}</b><br>
        <span style='color:#aaa;'>Margem:</span> <b>${d.margin_available:,.0f}</b><br>
        <span style='color:#aaa;'>Budget dia:</span> <b>${d.risk_budget_day_remaining:,.0f}</b>
      </div>
    </div>
    <div style='background:#1a2035;border-left:4px solid {bc};padding:10px 18px;border-radius:6px;min-width:180px;'>
      <div style='color:#aaa;font-size:9px;'>STATUS</div>
      <div style='color:{bc};font-size:12px;font-weight:bold;'>
        {'✓ Pronto' if d.execution_ready else '✗ Bloqueado'}</div>
      <div style='color:#aaa;font-size:10px;'>{d.block_reason or 'PAPER MODE'}</div>
      <div style='color:#ffaa00;font-size:10px;'>Flatten: {d.flatten_time}</div>
    </div>
  </div>
  {'<div style="font-size:10px;color:#aaa;margin-bottom:6px;">Strikes: ' + sk + '</div>' if d.strikes else ''}
  <div style='margin-bottom:8px;'>
    <div style='color:#00d4e8;font-size:9px;letter-spacing:.8px;margin-bottom:3px;'>REGIME PROBA</div>
    {rb}
  </div>
  <div style='background:#0a0f1a;padding:7px 10px;border-radius:4px;font-size:10px;color:#aaa;'>
    {d.rationale}
  </div>
</div>"""

    def _guide():
        return """
<div style='background:#0d1520;border:1px solid rgba(0,212,232,.15);border-radius:6px;
            padding:12px 16px;font-size:11px;font-family:monospace;line-height:1.8;margin:6px 0;'>
<span style='color:#00d4e8;font-size:10px;letter-spacing:.8px;'>COMO USAR</span><br><br>
<b>① Configure a conta</b> — Cash, NLV, Buying Power, Margem disponível com valores reais da IB.<br>
<b>② Ajuste risco</b> — Risk/Trade % = % do NLV por operação (ex: 1% de $100k = $1.000 por trade).<br>
<b>③ Gerar Decisão</b> — lê mercado (chain 0DTE, GEX, walls, IV) e produz a decisão de regime.<br>
<b>④ Leia o output</b>: <span style='color:#00ff99;'>BUY</span> = long / <span style='color:#ff4444;'>SELL</span> = short / <span style='color:#aaa;'>NO_TRADE</span> = sem sinal<br>
<b>⑤ Paper Execute</b> — loga no journal sem enviar ordens reais. Valide o fluxo primeiro.<br>
<b style='color:#ffaa00;'>⚠ Regra central:</b> SOMENTE day trade 0DTE. Toda posição é encerrada no mesmo dia.
</div>"""

    # ── Auto-refresh (5 min fixo, re-busca Bloomberg a cada ciclo) ───────────
    import threading as _threading
    _REFRESH_INTERVAL = 300  # 5 minutos em segundos

    w_auto = wd.ToggleButton(
        value=False,
        description='🔴 AUTO OFF',
        button_style='danger',
        layout=wd.Layout(width='140px', height='40px',
                         border='2px solid #f85149'))
    w_last_upd = wd.HTML(
        "<span style='color:#484f58;font-size:10px;font-family:monospace;"
        "margin-left:8px;'>● aguardando</span>")

    _stop_evt = [_threading.Event()]
    _thread   = [None]
    _live_df  = [df]   # cache do df mais recente

    def _fetch_fresh_data():
        """Re-busca options chain do Bloomberg."""
        try:
            # reutiliza os mesmos parâmetros do run_analysis que gerou o df original
            new_df = fetch_options_chain(
                ticker, spot,
                min_dte=0, max_dte=5,
                mny_low=-0.10, mny_high=0.10)  # monthly_only auto via ticker
            if new_df is not None and not new_df.empty:
                _live_df[0] = new_df
        except Exception as _fe:
            pass  # mantém df anterior se BBG falhar

    def _run_and_render():
        try:
            w_last_upd.value = ("<span style='color:#d29922;font-size:10px;"
                                "font-family:monospace;margin-left:8px;'>"
                                "⟳ buscando dados...</span>")
            _fetch_fresh_data()
            orch[0] = _make_orch()
            d = orch[0].run(_live_df[0], ext)
            ts = _de_dt.now().strftime('%H:%M:%S')
            with out_d:
                out_d.clear_output(wait=True)
                _disp(_HTML(_render(d)))
                _disp(_HTML(_guide()))
            w_last_upd.value = (
                f"<span style='color:#3fb950;font-size:10px;font-family:monospace;"
                f"margin-left:8px;'>✓ {ts}</span>")
        except Exception as _re:
            w_last_upd.value = (
                f"<span style='color:#f85149;font-size:10px;font-family:monospace;"
                f"margin-left:8px;'>⚠ {_re}</span>")

    def _auto_loop():
        while not _stop_evt[0].is_set():
            _run_and_render()
            # espera 5 min ou até ser parado
            _stop_evt[0].wait(timeout=_REFRESH_INTERVAL)

    def _on_run(_):
        _run_and_render()

    def _on_auto_toggle(change):
        if change['new']:
            w_auto.description  = '🟢 AUTO ON'
            w_auto.button_style = 'success'
            w_auto.layout.border = '2px solid #3fb950'
            _stop_evt[0].clear()
            _thread[0] = _threading.Thread(target=_auto_loop, daemon=True)
            _thread[0].start()
        else:
            w_auto.description  = '🔴 AUTO OFF'
            w_auto.button_style = 'danger'
            w_auto.layout.border = '2px solid #f85149'
            _stop_evt[0].set()
            w_last_upd.value = ("<span style='color:#484f58;font-size:10px;"
                                "font-family:monospace;margin-left:8px;'>"
                                "● parado</span>")

    def _on_pex(_):
        if orch[0]:
            res = orch[0].execute_paper()
            with out_d:
                out_d.clear_output(wait=True)
                _disp(_HTML(f"<pre style='color:#00d4e8;font-size:10px;'>"
                            f"PAPER EXECUTE:\n{_de_json.dumps(res, indent=2)}</pre>"))

    def _on_toggle(change):
        w_paper.description  = 'PAPER MODE ON' if change['new'] else '⚠ LIVE MODE'
        w_paper.button_style = 'warning'        if change['new'] else 'danger'

    def _on_override_click(_):
        _override_state[0] = not _override_state[0]
        if _override_state[0]:
            w_override.description  = '🔓 Trava OFF'
            w_override.button_style = 'danger'
        else:
            w_override.description  = '🔒 Trava ON'
            w_override.button_style = ''

    w_run.on_click(_on_run); w_pex.on_click(_on_pex)
    w_paper.observe(_on_toggle, names='value')
    w_override.on_click(_on_override_click)
    w_auto.observe(_on_auto_toggle, names='value')

    with out_d:
        _disp(_HTML(_guide()))

    header = wd.HTML(
        "<div style='padding:6px 0 2px;'>"
        "<h3 style='color:#00d4e8;margin:0 0 2px;font-size:15px;'>"
        "Decision Engine — 0DTE Intraday</h3>"
        "<p style='color:rgba(255,255,255,.3);font-size:10px;margin:0;'>"
        "Day trade only · 0DTE structures · ES / MES / SPX Options · "
        "Capital-constrained sizing · Flatten antes do fechamento</p></div>")
    acc_row = wd.VBox([
        wd.HTML("<p style='color:rgba(0,212,232,.5);font-size:9px;margin:4px 0 2px;"
                "letter-spacing:.8px;font-family:monospace;'>CONTA & CAPITAL</p>"),
        wd.HBox([w_cash, w_nlv, w_bp, w_margin, w_rpnl],
                layout=wd.Layout(flex_flow='row wrap', gap='8px')),
    ])
    risk_row = wd.VBox([
        wd.HTML("<p style='color:rgba(0,212,232,.5);font-size:9px;margin:8px 0 2px;"
                "letter-spacing:.8px;font-family:monospace;'>LIMITES DE RISCO</p>"),
        wd.HBox([w_risk, w_daily, w_maxpos],
                layout=wd.Layout(flex_flow='row wrap', gap='8px')),
    ])
    btn_row = wd.HBox(
        [w_paper, w_run, w_pex, w_auto, w_override, w_last_upd],
        layout=wd.Layout(gap='8px', margin='10px 0 6px 0',
                         align_items='center', flex_flow='row wrap'))
    return wd.VBox([header, acc_row, risk_row, btn_row, out_d])


