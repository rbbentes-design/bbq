"""UI helpers: HUD panels, SVG gauges, badges."""

import math as _m

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from .config import _C, wd, HTML, display
except ImportError:
    import sys, os as _os
    _dir = _os.path.dirname(_os.path.abspath(__file__)) if '__file__' in dir() else _os.getcwd()
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
    from config import _C, wd, HTML, display


def _hud_panel(content, title='', scan=True, glow=False):
    """Wrap HTML content in a HUD panel with L-corner brackets, glow border and scan line."""
    _scan   = "<div class='hud-scan-bar'></div>" if scan else ''
    _title  = f"<span class='hud-label'>{title}</span>" if title else ''
    _gcls   = ' hud-panel-glow' if glow else ''
    return (
        f"<div class='mm-dash hud-panel hud-scanlines{_gcls}'>"
        f"<span class='hud-c tl'></span><span class='hud-c tr'></span>"
        f"<span class='hud-c bl'></span><span class='hud-c br'></span>"
        f"{_scan}{_title}{content}"
        f"</div>")


def _svg_ring_html(value, vmin, vmax, label, unit='',
                   color1='#00c8ff', color2=None,
                   w=260, h=272, label2=''):
    """Stark Industries / Iron Man arc reactor gauge.
    Multi-ring: outer 360-tick ring + 270-deg value arc + inner decoration.
    """
    if color2 is None:
        color2 = '#ff8c00'

    cx = w // 2
    cy = (h - 28) // 2
    r_out  = min(cx, cy) - 6
    r_val  = r_out - 18
    r_dec  = r_val - 16
    r_core = r_dec - 10

    circ_val = 2 * _m.pi * r_val
    a_len = circ_val * 0.75
    g_len = circ_val - a_len
    pct  = max(0., min(1., (value - vmin) / (vmax - vmin))) if vmax != vmin else 0.5
    fill = a_len * pct

    gid  = str(abs(hash(label + str(round(value, 2)))) % 99999)
    gx1, gy1 = cx - r_val, cy - r_val
    gx2, gy2 = cx + r_val, cy + r_val

    # 24-tick outer ring (major every 6)
    ticks = ''
    for i in range(24):
        ang    = _m.radians(i * 15)
        is_maj = (i % 6 == 0)
        ri = r_out - (7 if is_maj else 4)
        x1 = cx + ri * _m.cos(ang);       y1 = cy + ri * _m.sin(ang)
        x2 = cx + r_out * _m.cos(ang);    y2 = cy + r_out * _m.sin(ang)
        sw = '2' if is_maj else '1'
        op = '0.7' if is_maj else '0.28'
        clr = color1 if is_maj else 'rgba(0,200,255,0.6)'
        ticks += (f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
                  f"stroke='{clr}' stroke-width='{sw}' opacity='{op}'/>")

    def _ep(deg):
        a = _m.radians(deg)
        return cx + (r_val + 24) * _m.cos(a), cy + (r_val + 24) * _m.sin(a)
    mx, my = _ep(-225)
    Mx, My = _ep(45)
    vmin_s = f'{vmin:.0f}' if abs(vmin) >= 1 else f'{vmin:.1f}'
    vmax_s = f'{vmax:.0f}' if abs(vmax) >= 1 else f'{vmax:.1f}'

    av = abs(value)
    vs = (f'{value:.0f}' if av >= 100 else f'{value:.1f}' if av >= 10 else f'{value:.2f}')
    if unit:
        vs += unit

    # Hexagonal grid inside core (6 lines)
    _hex = ''
    for i in range(6):
        ang = _m.radians(i * 30)
        x1 = cx + r_core * 0.72 * _m.cos(ang)
        y1 = cy + r_core * 0.72 * _m.sin(ang)
        x2 = cx - r_core * 0.72 * _m.cos(ang)
        y2 = cy - r_core * 0.72 * _m.sin(ang)
        _hex += (f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
                 f"stroke='{color1}' stroke-width='0.6' opacity='0.1'/>")

    # Stark angled corner markers + corner dots
    bk = 15
    br = (
        f"<line x1='0' y1='{bk}' x2='{bk}' y2='0' stroke='{color1}' stroke-width='2' opacity='0.9'/>"
        f"<line x1='0' y1='0' x2='{bk}' y2='0' stroke='{color1}' stroke-width='1.5' opacity='0.5'/>"
        f"<line x1='0' y1='0' x2='0' y2='{bk}' stroke='{color1}' stroke-width='1.5' opacity='0.5'/>"
        f"<line x1='{w}' y1='{bk}' x2='{w-bk}' y2='0' stroke='{color2}' stroke-width='2' opacity='0.9'/>"
        f"<line x1='{w-bk}' y1='0' x2='{w}' y2='0' stroke='{color2}' stroke-width='1.5' opacity='0.5'/>"
        f"<line x1='{w}' y1='0' x2='{w}' y2='{bk}' stroke='{color2}' stroke-width='1.5' opacity='0.5'/>"
        f"<line x1='0' y1='{h-bk}' x2='{bk}' y2='{h}' stroke='{color2}' stroke-width='2' opacity='0.9'/>"
        f"<line x1='0' y1='{h}' x2='{bk}' y2='{h}' stroke='{color2}' stroke-width='1.5' opacity='0.5'/>"
        f"<line x1='0' y1='{h-bk}' x2='0' y2='{h}' stroke='{color2}' stroke-width='1.5' opacity='0.5'/>"
        f"<line x1='{w}' y1='{h-bk}' x2='{w-bk}' y2='{h}' stroke='{color1}' stroke-width='2' opacity='0.9'/>"
        f"<line x1='{w-bk}' y1='{h}' x2='{w}' y2='{h}' stroke='{color1}' stroke-width='1.5' opacity='0.5'/>"
        f"<line x1='{w}' y1='{h-bk}' x2='{w}' y2='{h}' stroke='{color1}' stroke-width='1.5' opacity='0.5'/>"
        f"<circle cx='0' cy='0' r='3' fill='{color1}' opacity='0.7'/>"
        f"<circle cx='{w}' cy='0' r='3' fill='{color2}' opacity='0.7'/>"
        f"<circle cx='0' cy='{h}' r='3' fill='{color2}' opacity='0.7'/>"
        f"<circle cx='{w}' cy='{h}' r='3' fill='{color1}' opacity='0.7'/>"
    )

    sub = (f"<text x='{cx}' y='{cy+21}' text-anchor='middle' "
           f"font-family=\"'Orbitron','Courier New',monospace\" font-size='9' "
           f"fill='rgba(200,230,255,0.45)'>{str(label2)[:16]}</text>" if label2 else '')

    _defs = (
        f"<defs>"
        f"<linearGradient id='g{gid}' gradientUnits='userSpaceOnUse' "
        f"x1='{gx1:.0f}' y1='{gy1:.0f}' x2='{gx2:.0f}' y2='{gy2:.0f}'>"
        f"<stop offset='0%' stop-color='{color1}'/>"
        f"<stop offset='100%' stop-color='{color2}'/>"
        f"</linearGradient>"
        f"<filter id='fa{gid}' x='-60%' y='-60%' width='220%' height='220%'>"
        f"<feGaussianBlur stdDeviation='5' result='b'/>"
        f"<feMerge><feMergeNode in='b'/><feMergeNode in='SourceGraphic'/></feMerge>"
        f"</filter>"
        f"<filter id='ft{gid}' x='-50%' y='-50%' width='200%' height='200%'>"
        f"<feGaussianBlur stdDeviation='3' result='b'/>"
        f"<feMerge><feMergeNode in='b'/><feMergeNode in='SourceGraphic'/></feMerge>"
        f"</filter>"
        f"</defs>"
    )
    _o_ring = (f"<circle cx='{cx}' cy='{cy}' r='{r_out-4}' fill='none' "
               f"stroke='{color1}' stroke-width='0.5' opacity='0.18'/>")
    _track  = (f"<circle cx='{cx}' cy='{cy}' r='{r_val}' fill='none' "
               f"stroke='rgba(0,200,255,0.07)' stroke-width='14' "
               f"stroke-dasharray='{a_len:.2f} {g_len:.2f}' "
               f"transform='rotate(-225 {cx} {cy})' stroke-linecap='round'/>")
    _varc   = (f"<circle cx='{cx}' cy='{cy}' r='{r_val}' fill='none' "
               f"stroke='url(#g{gid})' stroke-width='14' "
               f"stroke-dasharray='{fill:.2f} {circ_val-fill:.2f}' "
               f"transform='rotate(-225 {cx} {cy})' stroke-linecap='round' filter='url(#fa{gid})'/>")
    _sheen  = (f"<circle cx='{cx}' cy='{cy}' r='{r_val}' fill='none' "
               f"stroke='rgba(255,255,255,0.14)' stroke-width='5' "
               f"stroke-dasharray='{fill:.2f} {circ_val-fill:.2f}' "
               f"transform='rotate(-225 {cx} {cy})' stroke-linecap='round'/>")
    _i_ring = (f"<circle cx='{cx}' cy='{cy}' r='{r_dec}' fill='none' "
               f"stroke='{color1}' stroke-width='1' opacity='0.2'/>"
               f"<circle cx='{cx}' cy='{cy}' r='{r_dec-4}' fill='none' "
               f"stroke='{color2}' stroke-width='0.5' opacity='0.12'/>")
    _core   = f"<circle cx='{cx}' cy='{cy}' r='{r_core}' fill='rgba(2,5,18,0.93)'/>"
    _cdot   = (f"<circle cx='{cx}' cy='{cy}' r='5' fill='{color1}' opacity='0.5' filter='url(#fa{gid})'/>"
               f"<circle cx='{cx}' cy='{cy}' r='2' fill='white' opacity='0.7'/>")
    _val_txt = (f"<text x='{cx}' y='{cy+3}' text-anchor='middle' dominant-baseline='middle' "
                f"font-family=\"'Orbitron','Courier New',monospace\" font-size='26' font-weight='900' "
                f"fill='{color1}' filter='url(#ft{gid})'>{vs}</text>")
    _minmax = (f"<text x='{mx:.1f}' y='{my+5:.1f}' text-anchor='middle' "
               f"font-family=\"'Orbitron','Courier New',monospace\" font-size='8' fill='rgba(0,200,255,0.32)'>{vmin_s}</text>"
               f"<text x='{Mx:.1f}' y='{My+5:.1f}' text-anchor='middle' "
               f"font-family=\"'Orbitron','Courier New',monospace\" font-size='8' fill='rgba(0,200,255,0.32)'>{vmax_s}</text>")
    _lbl    = (f"<text x='{cx}' y='{h-8}' text-anchor='middle' "
               f"font-family=\"'Orbitron','Courier New',monospace\" font-size='8' font-weight='700' "
               f"letter-spacing='3' fill='rgba(0,200,255,0.55)'>{label[:18].upper()}</text>")
    return (
        f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}' "
        f"style='background:linear-gradient(160deg,#010810,#020d1f);display:block;'>"
        + _defs + _o_ring + ticks + _track + _varc + _sheen + _i_ring + _core + _hex
        + _cdot + _val_txt + sub + _minmax + _lbl + br +
        f"</svg>"
    )


def create_gauge(value, title, range_min, range_max, bar_color, suffix,
                 steps=None, width=220, height=190):
    """Cria indicador gauge do Plotly."""
    if pd.isna(value):
        value = 0
    gauge_cfg = {
        'axis': {'range': [range_min, range_max],
                 'tickcolor': 'rgba(0,200,255,0.4)',
                 'tickfont': {'color': 'rgba(0,200,255,0.5)', 'size': 9,
                              'family': "'Orbitron','Courier New',monospace"}},
        'bar': {'color': bar_color, 'thickness': 0.3},
        'bgcolor': '#020d1f',
        'borderwidth': 1,
        'bordercolor': 'rgba(0,200,255,0.25)',
    }
    if steps:
        gauge_cfg['steps'] = steps
    return go.FigureWidget(
        go.Indicator(
            mode="gauge+number", value=value,
            title={'text': title, 'font': {'size': 11, 'color': 'rgba(0,200,255,0.55)',
                                           'family': "'Orbitron','Courier New',monospace"}},
            number={'suffix': suffix, 'font': {'size': 18, 'color': '#ff8c00',
                                               'family': "'Orbitron','Courier New',monospace"},
                    'valueformat': '.2f'},
            gauge=gauge_cfg),
        layout=go.Layout(width=width, height=height,
                         paper_bgcolor='#010810',
                         font=dict(color='#cce8ff',
                                   family="'Orbitron','Courier New',monospace"),
                         margin=dict(l=18, r=18, t=42, b=12)))


def create_symmetric_gauge(value, title, scale, unit='$Bn', width=220, height=195):
    """
    Gauge simétrico: zero no centro, vermelho à esquerda (negativo), verde à direita (positivo).
    scale = metade do range total (ex: 10 → vai de -10 a +10).
    value deve estar em $Bn já arredondado.
    """
    if pd.isna(value):
        value = 0.0
    value = round(float(value), 2)
    rng   = round(max(abs(value) * 1.35, float(scale)), 1)
    bar_color = _C['green'] if value >= 0 else _C['red']
    steps = [
        {'range': [-rng, 0],   'color': 'rgba(248,81,73,0.15)'},
        {'range': [0,    rng], 'color': 'rgba(63,185,80,0.15)'},
    ]
    # Ticks: só 3 (min, 0, max) — evita poluição visual
    tick_lo = f'{-rng:.1f}'
    tick_hi = f'{rng:.1f}'
    num_text = f'{value:.2f} {unit}'
    return go.FigureWidget(
        go.Indicator(
            mode="gauge",
            value=value,
            title={'text': title, 'font': {'size': 11, 'color': 'rgba(0,200,255,0.55)',
                                           'family': "'Orbitron','Courier New',monospace"}},
            domain={'x': [0, 1], 'y': [0.25, 1.0]},
            gauge={
                'axis': {'range': [-rng, rng],
                         'tickvals': [-rng, 0, rng],
                         'ticktext': [tick_lo, '0', tick_hi],
                         'tickcolor': 'rgba(0,200,255,0.4)',
                         'tickfont': {'color': 'rgba(0,200,255,0.5)', 'size': 9,
                                      'family': "'Orbitron','Courier New',monospace"}},
                'bar': {'color': bar_color, 'thickness': 0.30},
                'bgcolor': '#020d1f',
                'borderwidth': 1,
                'bordercolor': 'rgba(0,200,255,0.25)',
                'steps': steps,
                'threshold': {'line': {'color': '#8b949e', 'width': 2},
                              'thickness': 0.80, 'value': 0},
            }),
        layout=go.Layout(
            width=width, height=height,
            paper_bgcolor='#010810',
            font=dict(color='#cce8ff',
                      family="'Orbitron','Courier New',monospace"),
            margin=dict(l=20, r=20, t=44, b=12),
            annotations=[dict(
                text=num_text,
                x=0.5, y=0.08, xref='paper', yref='paper',
                showarrow=False,
                font=dict(size=16, color=bar_color),
                xanchor='center', yanchor='bottom',
            )]))
