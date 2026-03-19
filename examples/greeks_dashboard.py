"""
MARKET MAKER DASHBOARD — Versão Unificada v1.0
Consolidação completa de todas as análises anteriores de gregas, risco e posicionamento.

Organizado em seções:
  0. Imports e Configuração
  1. Motor de Gregas (Black-Scholes)
  2. Pipeline de Dados (BQL)
  3. Cálculos de Exposição e Curvas Modelo
  4. Modelos de Risco (VaR, Monte Carlo, P&L)
  5. Rebalanceamento de ETFs
  6. Matrizes de Sensibilidade
  7. Visualização (Gauges, Gráficos, Tabelas)
  8. Callback Principal e Montagem do Dashboard
  9. Interface de Widgets

Uso: Copie tudo para uma célula Jupyter no BQuant e execute.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 0 — IMPORTS E CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

from IPython.display import display
import ipywidgets as wd
from IPython.display import display, clear_output, HTML
import pandas as pd
import numpy as np
import os
from datetime import datetime
from functools import lru_cache
from typing import Optional, Dict, List, Tuple
import math
import traceback
from scipy.stats import norm, t as student_t
from scipy.optimize import minimize as sp_minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import bql
import random as _random
import warnings

try:
    import ipydatagrid as ipd
    from ipydatagrid import DataGrid, BarRenderer, TextRenderer
    HAS_DATAGRID = True
except ImportError:
    HAS_DATAGRID = False

try:
    import bqplot as bqp
    HAS_BQPLOT = True
except ImportError:
    HAS_BQPLOT = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as SkPipeline
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#e6edf3',
    'text.color': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.5,
    'font.family': 'sans-serif',
    'font.size': 11,
})

bq = bql.Service()

# ── Design System (cores, CSS, template Plotly) ─────────────────────────────
_C = {
    'bg': '#0d1117', 'card': '#161b22', 'card2': '#1c2333',
    'border': '#30363d', 'border_light': '#21262d',
    'text': '#e6edf3', 'text_muted': '#8b949e', 'text_dim': '#484f58',
    'accent': '#58a6ff', 'teal': '#00d4aa', 'orange': '#f0883e',
    'green': '#3fb950', 'red': '#f85149', 'purple': '#bc8cff',
    'yellow': '#d29922', 'pink': '#f778ba',
}

DASH_CSS = (
"<style>\n"
"@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&display=swap');\n"
"/* =========================================================================\n"
"   STARK ANALYTICS -- JARVIS INTERFACE v3\n"
"   Inspired by Tony Stark / Iron Man HUD aesthetic\n"
"   ========================================================================= */\n"
"\n"
"/* -- Base font -- */\n"
".mm-dash { font-family: 'Orbitron','Courier New',monospace; color:#cce8ff; }\n"
"\n"
"/* -- Keyframe animations -- */\n"
"@keyframes hud-pulse-glow {\n"
"  0%,100% { box-shadow:0 0 6px 1px rgba(0,212,255,0.55),0 0 18px 4px rgba(0,212,255,0.2); }\n"
"  50%      { box-shadow:0 0 8px 2px rgba(0,212,255,0.7), 0 0 28px 8px rgba(0,212,255,0.3); }\n"
"}\n"
"@keyframes hud-dot-blink { 0%,100%{opacity:1;} 50%{opacity:0.2;} }\n"
"@keyframes hud-scan-sweep {\n"
"  0%   { top:-4px;  opacity:0.8; }\n"
"  95%  { opacity:0.6; }\n"
"  100% { top:100%;  opacity:0; }\n"
"}\n"
"\n"
"/* -- Circuit board background -- */\n"
".hud-circuit {\n"
"  background-color:#040818;\n"
"  background-image:\n"
"    repeating-linear-gradient(0deg,transparent,transparent 29px,rgba(0,212,255,0.07) 29px,rgba(0,212,255,0.07) 30px),\n"
"    repeating-linear-gradient(90deg,transparent,transparent 29px,rgba(0,212,255,0.07) 29px,rgba(0,212,255,0.07) 30px),\n"
"    repeating-linear-gradient(0deg,transparent,transparent 149px,rgba(0,212,255,0.13) 149px,rgba(0,212,255,0.13) 150px),\n"
"    repeating-linear-gradient(90deg,transparent,transparent 149px,rgba(0,212,255,0.13) 149px,rgba(0,212,255,0.13) 150px);\n"
"  background-size:30px 30px,30px 30px,150px 150px,150px 150px;\n"
"}\n"
"\n"
"/* -- HUD panel -- */\n"
".hud-panel {\n"
"  position:relative;\n"
"  background:linear-gradient(145deg,rgba(2,15,30,0.96) 0%,rgba(5,20,40,0.92) 100%);\n"
"  border:1px solid rgba(0,212,255,0.25);\n"
"  padding:14px 16px; margin:3px; overflow:hidden;\n"
"  clip-path:polygon(0 0,calc(100% - 14px) 0,100% 14px,100% 100%,0 100%);\n"
"}\n"
".hud-panel::before {\n"
"  content:''; position:absolute; top:0; left:0; right:0; height:1px;\n"
"  background:linear-gradient(90deg,rgba(0,212,255,0) 0%,rgba(0,212,255,0.9) 40%,rgba(0,255,200,0.9) 60%,rgba(0,212,255,0) 100%);\n"
"}\n"
".hud-panel::after {\n"
"  content:''; position:absolute; inset:0; pointer-events:none;\n"
"  background:radial-gradient(ellipse at 50% -20%,rgba(0,212,255,0.07) 0%,transparent 65%);\n"
"}\n"
".hud-panel-glow {\n"
"  box-shadow:0 0 6px 1px rgba(0,212,255,0.55),0 0 18px 4px rgba(0,212,255,0.2),inset 0 0 30px rgba(0,212,255,0.04);\n"
"  animation:hud-pulse-glow 3s ease-in-out infinite;\n"
"}\n"
"\n"
"/* -- L-corner brackets -- */\n"
".hud-c { position:absolute; width:14px; height:14px; z-index:5; }\n"
".hud-c.tl { top:-1px; left:-1px;   border-top:2px solid #00d4ff; border-left:2px solid #00d4ff; }\n"
".hud-c.tr { top:-1px; right:-1px;  border-top:2px solid #00d4ff; border-right:2px solid #00d4ff; }\n"
".hud-c.bl { bottom:-1px; left:-1px; border-bottom:2px solid #00d4ff; border-left:2px solid #00d4ff; }\n"
".hud-c.br { bottom:-1px; right:-1px; border-bottom:2px solid #00d4ff; border-right:2px solid #00d4ff; }\n"
"\n"
"/* -- Scan line -- */\n"
".hud-scanlines { position:relative; overflow:hidden; }\n"
".hud-scanlines::after {\n"
"  content:''; position:absolute; inset:0; pointer-events:none; z-index:3;\n"
"  background:repeating-linear-gradient(to bottom,transparent 0px,transparent 3px,rgba(0,0,0,0.14) 3px,rgba(0,0,0,0.14) 4px);\n"
"}\n"
".hud-scan-bar {\n"
"  position:absolute; left:0; right:0; height:3px; z-index:4; pointer-events:none;\n"
"  background:linear-gradient(to bottom,transparent,rgba(0,212,255,0.5) 50%,transparent);\n"
"  animation:hud-scan-sweep 5s linear infinite;\n"
"}\n"
"\n"
"/* -- Panel label -- */\n"
".hud-label { font-size:9px; font-weight:700; letter-spacing:2.5px;\n"
"  text-transform:uppercase; color:rgba(0,212,255,0.6); margin-bottom:6px; display:block; }\n"
"\n"
"/* -- Live dot -- */\n"
".hud-dot-live { display:inline-block; width:7px; height:7px; border-radius:50%;\n"
"  background:#00d4ff; box-shadow:0 0 6px 2px rgba(0,212,255,0.8);\n"
"  animation:hud-dot-blink 1.8s ease-in-out infinite; vertical-align:middle; margin-right:6px; }\n"
"\n"
"/* -- Section header (Stark orange accent) -- */\n"
".mm-section-hdr {\n"
"  display:flex; align-items:center; gap:10px;\n"
"  background:linear-gradient(90deg,rgba(255,140,0,0.12) 0%,rgba(255,140,0,0.03) 55%,transparent 100%);\n"
"  border-left:3px solid #ff8c00;\n"
"  padding:5px 14px 5px 10px; margin:10px 0 3px; position:relative;\n"
"  clip-path:polygon(0 0,calc(100% - 10px) 0,100% 50%,calc(100% - 10px) 100%,0 100%);\n"
"}\n"
".mm-section-hdr::before {\n"
"  content:''; position:absolute; inset:0; pointer-events:none;\n"
"  background:repeating-linear-gradient(90deg,transparent 0px,transparent 5px,rgba(255,140,0,0.04) 5px,rgba(255,140,0,0.04) 6px);\n"
"}\n"
".mm-dot { width:7px; height:7px; border-radius:50%; background:#ff8c00;\n"
"  box-shadow:0 0 5px 2px rgba(255,140,0,0.8);\n"
"  animation:hud-dot-blink 2s ease-in-out infinite; flex-shrink:0; }\n"
".mm-hdr-title { font-size:9px; font-weight:700; color:#ff8c00;\n"
"  text-transform:uppercase; letter-spacing:2.5px; font-family:'Orbitron','Courier New',monospace; }\n"
".mm-hdr-sub { font-size:8px; color:rgba(0,200,255,0.45); letter-spacing:0.5px; }\n"
"\n"
"/* -- Status bar -- */\n"
".mm-statusbar {\n"
"  display:flex; flex-wrap:wrap; align-items:center;\n"
"  padding:10px 20px; margin:0 0 2px; position:relative; overflow:hidden;\n"
"  background:linear-gradient(90deg,#010d1a 0%,#020f1f 50%,#010d1a 100%);\n"
"  border:1px solid rgba(0,212,255,0.3);\n"
"  clip-path:polygon(0 0,calc(100% - 20px) 0,100% 20px,100% 100%,0 100%);\n"
"  box-shadow:0 0 20px rgba(0,212,255,0.12),inset 0 0 40px rgba(0,212,255,0.04);\n"
"}\n"
".mm-statusbar::before {\n"
"  content:''; position:absolute; top:0; left:0; right:0; height:1px;\n"
"  background:linear-gradient(90deg,transparent,#00d4ff 30%,#00ffcc 60%,transparent);\n"
"}\n"
".mm-statusbar::after {\n"
"  content:''; position:absolute; inset:0; pointer-events:none;\n"
"  background:repeating-linear-gradient(0deg,transparent 0px,transparent 3px,rgba(0,0,0,0.08) 3px,rgba(0,0,0,0.08) 4px);\n"
"}\n"
".mm-cmd-title { font-size:12px; font-weight:900; color:#ff8c00; letter-spacing:4px;\n"
"  text-transform:uppercase; margin-right:28px; font-family:'Orbitron','Courier New',monospace;\n"
"  text-shadow:0 0 12px rgba(255,140,0,0.8),0 0 28px rgba(255,140,0,0.4); }\n"
".mm-stat-item { display:flex; flex-direction:column; align-items:center;\n"
"  padding:2px 16px; border-right:1px solid rgba(0,212,255,0.15); position:relative; z-index:1; }\n"
".mm-stat-item:last-child { border-right:none; }\n"
".mm-stat-label { font-size:7px; color:rgba(0,212,255,0.45); text-transform:uppercase;\n"
"  letter-spacing:1.8px; font-weight:700; }\n"
".mm-stat-value { font-size:14px; font-weight:700; font-family:'Courier New',monospace;\n"
"  line-height:1.2; text-shadow:0 0 8px currentColor; }\n"
"\n"
"/* -- Card (legacy compatibility) -- */\n"
".mm-card { position:relative; overflow:hidden;\n"
"  background:linear-gradient(145deg,rgba(1,8,20,0.97) 0%,rgba(2,12,30,0.93) 100%);\n"
"  border:1px solid rgba(0,200,255,0.15);\n"
"  border-top:2px solid rgba(255,140,0,0.65);\n"
"  padding:12px 14px; margin:3px;\n"
"  clip-path:polygon(0 0,calc(100% - 10px) 0,100% 10px,100% 100%,0 100%);\n"
"}\n"
".mm-card::before { content:''; position:absolute; top:0; left:0; right:0; height:1px;\n"
"  background:linear-gradient(90deg,transparent,rgba(255,140,0,0.7) 50%,transparent); }\n"
".mm-card h3 { margin:0 0 8px; font-size:10px; font-weight:700; color:#ff8c00;\n"
"  text-transform:uppercase; letter-spacing:2px;\n"
"  font-family:'Orbitron','Courier New',monospace; }\n"
".mm-card h4 { margin:6px 0 4px; font-size:9px; font-weight:700; color:#00c8ff; letter-spacing:1px; }\n"
".mm-card p, .mm-card span { font-size:11px; color:rgba(180,210,255,0.75); line-height:1.6; }\n"
".mm-card b { color:#cce8ff; }\n"
"\n"
"/* -- Badges -- */\n"
".mm-badge { display:inline-block; padding:1px 8px; font-size:9px; font-weight:700;\n"
"  letter-spacing:1.5px; text-transform:uppercase;\n"
"  font-family:'Orbitron','Courier New',monospace;\n"
"  clip-path:polygon(4px 0,100% 0,calc(100% - 4px) 100%,0 100%); }\n"
"\n"
"/* =========================================================================\n"
"   JARVIS HUD -- boot, particles, scanlines, ticker, arc reactor\n"
"   ========================================================================= */\n"
"@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');\n"
"\n"
"@keyframes spin-cw  { from { transform:rotate(0deg) }   to { transform:rotate(360deg) } }\n"
"@keyframes spin-ccw { from { transform:rotate(360deg) } to { transform:rotate(0deg) } }\n"
".jarvis-reactor { flex-shrink:0; margin-right:10px; }\n"
".jarvis-r1 { animation:spin-cw  14s linear infinite; transform-origin:21px 21px; }\n"
".jarvis-r2 { animation:spin-ccw  9s linear infinite; transform-origin:21px 21px; }\n"
".jarvis-r3 { animation:spin-cw   5s linear infinite; transform-origin:21px 21px; }\n"
"\n"
"</style>\n"
)



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
    import math as _m
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


# Template Plotly -- Stark Industries Iron Man palette
_PLOTLY_DARK = go.layout.Template()
_PLOTLY_DARK.layout = go.Layout(
    paper_bgcolor='#010810',
    plot_bgcolor='#020d1f',
    font=dict(family="'Orbitron','Courier New',monospace", size=11, color='#cce8ff'),
    title=dict(font=dict(size=13, color='#ff8c00', family="'Orbitron','Courier New',monospace")),
    xaxis=dict(gridcolor='rgba(0,200,255,0.08)', zerolinecolor='rgba(0,200,255,0.2)',
               linecolor='rgba(0,200,255,0.15)', tickcolor='rgba(0,200,255,0.4)',
               tickfont=dict(color='rgba(0,200,255,0.55)', size=10)),
    yaxis=dict(gridcolor='rgba(0,200,255,0.08)', zerolinecolor='rgba(0,200,255,0.2)',
               linecolor='rgba(0,200,255,0.15)', tickcolor='rgba(0,200,255,0.4)',
               tickfont=dict(color='rgba(0,200,255,0.55)', size=10)),
    legend=dict(bgcolor='rgba(1,8,20,0.8)', font=dict(size=10, color='#cce8ff'),
                bordercolor='rgba(0,200,255,0.2)', borderwidth=1),
    colorway=['#00c8ff', '#ff8c00', '#00ff88', '#ff4757', '#b44aff',
              '#ffd32a', '#ff6b9d', '#7efff5'],
    hoverlabel=dict(bgcolor='#010810', font_size=11, font_color='#cce8ff',
                    bordercolor='rgba(0,200,255,0.4)'),
)
DASH_TEMPLATE = _PLOTLY_DARK
BQL_PARAMS = {'fill': 'prev'}
TRADING_DAYS = 252
FUTURES_TICKER = 'ES1 Index'
FUTURES_MULTIPLIER = 50

# ── Snapshot state (usado pelo botão de export) ────────────────────────────
_greek_cache = {}  # populated by build_greek_overview()
_snapshot = {'sections': [], 'ticker': '', 'spot': 0, 'ts': '', 'metrics': {}}

# Configuração dos 7 greeks para gráficos de exposição.
# Cada entrada define: nome, chave no dict de gregas, unidade de exibição,
# função de escala (converte greek*OI*100 → dólares), divisor de exibição,
# e operação para combinar call/put no total líquido.
GREEK_CONFIGS = [
    {'name': 'Delta',  'key': 'delta',  'unit': '$ Mn',            'scale': lambda L: L,              'div': 1e6, 'op': np.add},
    {'name': 'Gamma',  'key': 'gamma',  'unit': '$ Mn / 1% move',  'scale': lambda L: (L**2) * 0.01,  'div': 1e6, 'op': np.subtract},
    {'name': 'Vega',   'key': 'vega',   'unit': '$ Mn / 1% vol',   'scale': lambda L: 1,              'div': 1e6, 'op': np.add},
    {'name': 'Vanna',  'key': 'vanna',  'unit': '$ Mn / 1% vol',   'scale': lambda L: L,              'div': 1e6, 'op': np.subtract},
    {'name': 'Theta',  'key': 'theta',  'unit': '$ Mn / dia',      'scale': lambda L: 1.0/TRADING_DAYS, 'div': 1e6, 'op': np.add},
    {'name': 'Charm',  'key': 'charm',  'unit': '$ Mn / dia',      'scale': lambda L: L / 365.0,      'div': 1e6, 'op': np.add},
    {'name': 'Zomma',  'key': 'zomma',  'unit': '$ Mn',            'scale': lambda L: 1,              'div': 1e6, 'op': np.subtract},
    {'name': 'Speed',  'key': 'speed',  'unit': '$ Mn',            'scale': lambda L: 1,              'div': 1e6, 'op': np.subtract},
]

# ETFs passivos e alavancados
INDEX_PROXY = 'B500 Index'
PASSIVE_ETFS = ['VOO US Equity', 'SPY US Equity', 'IVV US Equity']
LEVERAGED_ETFS = [
    {'ticker': 'SPXL US Equity', 'name': 'SPXL', 'leverage': 3},
    {'ticker': 'UPRO US Equity', 'name': 'UPRO', 'leverage': 3},
    {'ticker': 'SSO US Equity',  'name': 'SSO',  'leverage': 2},
    {'ticker': 'SH US Equity',   'name': 'SH',   'leverage': -1},
    {'ticker': 'SDS US Equity',  'name': 'SDS',  'leverage': -2},
    {'ticker': 'SPXS US Equity', 'name': 'SPXS', 'leverage': -3},
    {'ticker': 'SPXU US Equity', 'name': 'SPXU', 'leverage': -3},
]

# ETFs alavancados estendidos (para flow predictor — inclui NDX e SOX)
LEVERAGED_ETFS_EXT = LEVERAGED_ETFS + [
    {'ticker': 'TQQQ US Equity', 'name': 'TQQQ', 'leverage': 3, 'under': 'NDX Index'},
    {'ticker': 'SQQQ US Equity', 'name': 'SQQQ', 'leverage': -3, 'under': 'NDX Index'},
    {'ticker': 'SOXL US Equity', 'name': 'SOXL', 'leverage': 3, 'under': 'SOX Index'},
    {'ticker': 'SOXS US Equity', 'name': 'SOXS', 'leverage': -3, 'under': 'SOX Index'},
]

# Mapeamento: ticker → futures com dados COT
COT_FUTURES_MAP = {
    'SPX Index': 'ES1 Index', 'NDX Index': 'NQ1 Index',
    'RTY Index': 'RA1 Index', 'INDU Index': 'DM1 Index',
    'SPXL US Equity': 'ES1 Index', 'SPXS US Equity': 'ES1 Index',
    'TQQQ US Equity': 'NQ1 Index', 'SQQQ US Equity': 'NQ1 Index',
    'NG1 Comdty': 'NG1 Comdty', 'CL1 Comdty': 'CL1 Comdty',
    'CO1 Comdty': 'CO1 Comdty', 'GC1 Comdty': 'GC1 Comdty',
    'SI1 Comdty': 'SI1 Comdty', 'HG1 Comdty': 'HG1 Comdty',
    'W 1 Comdty': 'W 1 Comdty', 'S 1 Comdty': 'S 1 Comdty',
    'C 1 Comdty': 'C 1 Comdty',
}

COT_CONTRACTS = {
    'Equity Indices': [
        ('S&P 500 E-mini', 'ES1 Index'),
        ('Nasdaq 100 E-mini', 'NQ1 Index'),
        ('Russell 2000 E-mini', 'RA1 Index'),
    ],
    'Energy': [
        ('WTI Crude', 'CL1 Comdty'),
        ('Brent Crude', 'CO1 Comdty'),
        ('Natural Gas', 'NG1 Comdty'),
    ],
    'Oil': [
        ('Brent', 'CO1 Comdty'),
        ('Gasoil', 'QS1 Comdty'),
        ('WTI', 'CL1 Comdty'),
        ('RBOB', 'XB1 Comdty'),
        ('ULSD', 'HO1 Comdty'),
    ],
    'Power': [
        ('German Power', 'DEBM1 Comdty'),
        ('French Power', 'FRBM1 Comdty'),
        ('Italian Power', 'ITBM1 Comdty'),
        ('ERCOT', 'ERHN1 Comdty'),
    ],
    'Metals': [
        ('Gold', 'GC1 Comdty'),
        ('Silver', 'SI1 Comdty'),
        ('Copper', 'HG1 Comdty'),
    ],
    'Agriculture': [
        ('Wheat', 'W 1 Comdty'),
        ('Soybeans', 'S 1 Comdty'),
        ('Corn', 'C 1 Comdty'),
    ],
}

# Estimativas padrão de AUM ($ USD) para cálculo rápido de fluxo
DEFAULT_AUM = {
    'SPXL': 5e9, 'UPRO': 3e9, 'SSO': 4e9,
    'SH': 2e9, 'SDS': 0.8e9, 'SPXS': 1e9, 'SPXU': 0.8e9,
    'TQQQ': 20e9, 'SQQQ': 5e9, 'SOXL': 10e9, 'SOXS': 3e9,
}

# Mapeamento: trader type → report type correto para COT
COT_TRADER_REPORT_MAP = {
    'Managed Money': 'CFTC_Disaggregated',
    'Swap Dealers': 'CFTC_Disaggregated',
    'Commercial': 'CFTC_Legacy',
    'Non-Commercial': 'CFTC_Legacy',
    'Asset Manager': 'CFTC_TFF',
    'Leveraged Funds': 'CFTC_TFF',
    'Total': None,
}

# Estimativa anual de buyback do SPX (fallback quando BQL não retorna)
SPX_ANNUAL_BUYBACK_EST = 1.0e12  # ~$1T USD

# ── Market Maker volume shares (OVME, avg daily contracts, latest month) ──
# Fonte: Bloomberg OVME → Options Volume Matrix → By Exchange
MM_VOLUME_SHARES = {
    'Citadel Securities':       0.34,
    'Dash Financial/IMC':       0.26,
    'Jane Street':              0.15,
    'Wolverine Execution':      0.11,
    'Global Execution/Susque':  0.10,
    'NASDAQ':                   0.025,
    'MIAX':                     0.015,
    'Others':                   0.01,
}

# ── Options trade description breakdown (OVME, % of total avg daily contracts) ──
# Fonte: Bloomberg OVME → By Trade Description (as Percent %)
OPTIONS_TRADE_DESC = [
    ('Electronic',                         53.16),
    ('Single Leg Auction Non ISO',         23.42),
    ('Multi Leg Electronic',                9.57),
    ('Intermarket Sweep',                   4.23),
    ('Multi Leg Auction',                   3.82),
    ('Multi Leg vs Single, Electronic',     1.92),
    ('Multi Leg vs Single, Floor Trade',    0.84),
    ('Open Trade / Late & Out of Seq',      0.79),
    ('Single Leg Floor Trade',              0.58),
    ('Multi Leg Floor Trade',               0.54),
    ('Single Leg Cross ISO',                0.46),
    ('Multi Leg Cross',                     0.18),
    ('Other',                               0.49),
]
# Total avg daily contracts (latest month, OVME)
OPTIONS_TOTAL_ADC = 21_685_696

# ── Dispersion Trade Constants ──
MAG7 = [
    'AAPL US Equity', 'MSFT US Equity', 'GOOGL US Equity', 'AMZN US Equity',
    'NVDA US Equity', 'META US Equity', 'TSLA US Equity',
]
DISP_CORR_WINDOWS = {'1M': 21, '3M': 63, '6M': 126}
DISP_IV_FIELD = 'implied_volatility'  # BQL field
DISP_SPXSK3 = '.SPXSK3 G Index'  # CBOE S&P 500 3M Implied Corr Index
DISP_VIX9D = 'VIX9D Index'
DISP_COR1M = 'COR1M Index'       # 1M 50-Delta Implied Correlation
DISP_DSPX = 'DSPX Index'         # CBOE S&P 500 Dispersion Index
DISP_VIXEQ = 'VIXEQ Index'       # Single Stock Vol Premium (VIX - realized eq vol)
DISP_TOP_N = 10                   # Top N members by weight for dispersion
DISP_EXCLUDE = {'BRK/B US Equity'}  # Tickers excluídos da análise de dispersão

# Resolve path do CSV de gamma — busca em cwd direto, data/, e subindo na árvore
def _find_gamma_csv():
    base = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
    search = os.path.abspath(base)
    for _ in range(6):  # sobe até 6 níveis
        # tenta na raiz do diretório atual (ex: /gamma_history.csv no BQuant)
        for candidate in [
            os.path.join(search, 'gamma_history.csv'),
            os.path.join(search, 'data', 'gamma_history.csv'),
        ]:
            if os.path.exists(candidate):
                return os.path.normpath(candidate)
        parent = os.path.dirname(search)
        if parent == search:
            break
        search = parent
    # fallback: usa cwd/data/ (será criado no append se não existir)
    return os.path.normpath(os.path.join(os.getcwd(), 'data', 'gamma_history.csv'))

GAMMA_HISTORY_PATH = _find_gamma_csv()
print(f"[GAMMA DB] Path resolvido: {GAMMA_HISTORY_PATH} (exists={os.path.exists(GAMMA_HISTORY_PATH)})")
# ...existing code...

# Layout padrão Plotly (dark elegante para todos os gráficos)
FLOW_FIG_LAYOUT = {
    'template': DASH_TEMPLATE,
    'font': {'family': '-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica', 'size': 12},
    'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': -0.3,
               'xanchor': 'center', 'x': 0.5},
    'margin': {'t': 40, 'b': 40, 'l': 50, 'r': 20},
}


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — MOTOR DE GREGAS (BLACK-SCHOLES)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_all_greeks(S, K, vol, T, option_types, r=0.0):
    """
    Calcula todas as gregas de Black-Scholes de forma vetorial.

    Args:
        S: Preço spot do ativo subjacente (escalar).
        K: Array de strikes.
        vol: Array de volatilidade implícita (decimal, ex: 0.20 para 20%).
        T: Array de tempo até expiração em anos.
        option_types: Array de strings 'Call' ou 'Put'.
        r: Taxa livre de risco (default 0).

    Returns:
        Dict com arrays: delta, gamma, vega, vanna, zomma, theta, charm, speed.
        Cada valor é o greek por opção (não multiplicado por OI).
    """
    K = np.asarray(K, dtype=float)
    vol = np.asarray(vol, dtype=float)
    T = np.asarray(T, dtype=float)
    option_types = np.asarray(option_types)

    mask = (vol <= 0) | (T <= 0) | np.isnan(vol) | (K <= 0) | (S <= 0)
    greeks = {k: np.zeros_like(vol) for k in
              ['delta', 'gamma', 'vega', 'vanna', 'zomma', 'theta', 'charm', 'speed']}
    d1 = np.zeros_like(vol)

    with np.errstate(divide='ignore', invalid='ignore'):
        v = ~mask
        if not np.any(v):
            return greeks

        K_v, vol_v, T_v = K[v], vol[v], T[v]
        types_v = option_types[v]
        sqrt_T = np.sqrt(T_v)

        d1_v = (np.log(S / K_v) + (r + 0.5 * vol_v**2) * T_v) / (vol_v * sqrt_T)
        d2_v = d1_v - vol_v * sqrt_T

        pdf_d1 = norm.pdf(d1_v)
        cdf_d1 = norm.cdf(d1_v)
        cdf_d2 = norm.cdf(d2_v)

        delta_call = cdf_d1
        gamma_v = pdf_d1 / (S * vol_v * sqrt_T)

        # Delta
        greeks['delta'][v] = np.where(types_v == 'Call', delta_call, delta_call - 1)

        # Gamma
        greeks['gamma'][v] = gamma_v

        # Vega (por 1% de mudança na vol — divide por 100)
        greeks['vega'][v] = S * pdf_d1 * sqrt_T / 100.0

        # Vanna (∂Delta/∂Vol)
        greeks['vanna'][v] = -pdf_d1 * d2_v / vol_v

        # Zomma (∂Gamma/∂Vol)
        greeks['zomma'][v] = gamma_v * ((d1_v * d2_v - 1) / vol_v)

        # Theta (decaimento anualizado do preço da opção)
        exp_rT = np.exp(-r * T_v)
        theta_call = -(S * pdf_d1 * vol_v) / (2 * sqrt_T) - r * K_v * exp_rT * cdf_d2
        theta_put = -(S * pdf_d1 * vol_v) / (2 * sqrt_T) + r * K_v * exp_rT * (1 - cdf_d2)
        greeks['theta'][v] = np.where(types_v == 'Call', theta_call, theta_put)

        # Charm (∂Delta/∂T — decaimento anualizado do delta)
        charm_num = 2 * r * T_v - d2_v * vol_v * sqrt_T
        charm_den = 2 * T_v * vol_v * sqrt_T
        greeks['charm'][v] = -pdf_d1 * charm_num / charm_den

        # Speed (∂Gamma/∂S)
        greeks['speed'][v] = -gamma_v / S * (d1_v / (vol_v * sqrt_T) + 1)

    return greeks


def black_scholes_price_vec(S, K, vol, T, option_types, r=0.0):
    """
    Preço Black-Scholes vetorizado (europeu).
    Mesma fórmula d1/d2 de calculate_all_greeks — retorna array de preços.
    Opções com T <= 0 retornam valor intrínseco (max(S-K,0) ou max(K-S,0)).
    """
    K_a    = np.asarray(K,            dtype=float)
    vol_a  = np.asarray(vol,          dtype=float)
    T_a    = np.asarray(T,            dtype=float)
    types_a = np.asarray(option_types)
    prices = np.zeros(len(K_a))

    # opções expiradas → valor intrínseco
    expired = T_a <= 0
    if expired.any():
        prices[expired] = np.where(
            types_a[expired] == 'Call',
            np.maximum(S - K_a[expired], 0.0),
            np.maximum(K_a[expired] - S, 0.0))

    v = (vol_a > 0) & (T_a > 0) & (K_a > 0) & (S > 0) & ~np.isnan(vol_a)
    if not v.any():
        return prices
    with np.errstate(divide='ignore', invalid='ignore'):
        sq   = np.sqrt(T_a[v])
        d1   = (np.log(S / K_a[v]) + (r + 0.5 * vol_a[v] ** 2) * T_a[v]) / (vol_a[v] * sq)
        d2   = d1 - vol_a[v] * sq
        disc = np.exp(-r * T_a[v])
        c_px = S * norm.cdf(d1) - K_a[v] * disc * norm.cdf(d2)
        p_px = K_a[v] * disc * norm.cdf(-d2) - S * norm.cdf(-d1)
        prices[v] = np.where(types_a[v] == 'Call', c_px, p_px)
    return prices


def calculate_flip(levels, curve):
    """Encontra o ponto onde a curva cruza zero (interpolação linear)."""
    sign_changes = np.where(np.diff(np.sign(curve)))[0]
    if sign_changes.size > 0:
        try:
            i = sign_changes[0]
            x1, y1 = levels[i], curve[i]
            x2, y2 = levels[i + 1], curve[i + 1]
            if (y2 - y1) == 0:
                return None
            return x1 - y1 * (x2 - x1) / (y2 - y1)
        except (IndexError, ZeroDivisionError):
            return None
    return None


def implied_move_pct(iv_annual, days=1):
    """Retorna o movimento implícito diário em % dado IV anualizada (decimal)."""
    return iv_annual * 100 * math.sqrt(days / TRADING_DAYS)


def fmt_value(value, decimals=2):
    """Formata valor grande em Bi/Mi/K."""
    if pd.isna(value):
        return "N/A"
    a = abs(value)
    if a >= 1e9:
        return f"{value / 1e9:,.{decimals}f} Bi"
    if a >= 1e6:
        return f"{value / 1e6:,.{decimals}f} Mi"
    if a >= 1e3:
        return f"{value / 1e3:,.{decimals}f} K"
    return f"{value:,.{decimals}f}"


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — PIPELINE DE DADOS (BQL)
# ═══════════════════════════════════════════════════════════════════════════════


def _bql_ts(resp_item, field):
    """
    Extrai série temporal de um BQL response item para single-ticker.
    Usa reset_index() para robustez contra variações de MultiIndex.
    Retorna pd.Series com DatetimeIndex.
    """
    df = resp_item.df().reset_index()
    # Encontrar coluna de data
    date_col = None
    for c in df.columns:
        if str(c).upper() == 'DATE':
            date_col = c
            break
    if date_col is None:
        # Tentar detectar coluna com datas
        for c in df.columns:
            if c == field or str(c).upper() == 'ID':
                continue
            try:
                sample = df[c].dropna().head(5)
                if len(sample) > 0:
                    pd.to_datetime(sample)
                    date_col = c
                    break
            except Exception:
                continue
    if date_col is not None:
        s = df.set_index(date_col)[field]
        s.index = pd.to_datetime(s.index)
    else:
        # Fallback: tentar converter index original
        s = resp_item.df()[field]
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            if isinstance(s.index, pd.MultiIndex):
                s.index = s.index.droplevel(0)
                s.index = pd.to_datetime(s.index)
    return s


def _bql_ts_df(resp_item):
    """
    Extrai DataFrame de um BQL response item para single-ticker.
    Retorna DataFrame com DatetimeIndex (sem coluna ID).
    """
    df = resp_item.df().reset_index()
    date_col = None
    for c in df.columns:
        if str(c).upper() == 'DATE':
            date_col = c
            break
    if date_col is not None:
        id_col = [c for c in df.columns if str(c).upper() == 'ID']
        if id_col:
            df = df.drop(columns=id_col)
        df = df.set_index(date_col)
        df.index = pd.to_datetime(df.index)
    else:
        orig = resp_item.df()
        if isinstance(orig.index, pd.MultiIndex):
            orig.index = orig.index.droplevel(0)
        orig.index = pd.to_datetime(orig.index)
        df = orig
    return df


def fetch_market_data(ticker):
    """Busca spot, IV 30d, RV 30d, skew, volume médio em dólares, risk-free rate, MOVE Index."""
    spot = pd.to_numeric(
        bq.execute(bql.Request(ticker, {'Value': bq.data.px_last()}))[0].df()['Value'].iloc[-1],
        errors='coerce')

    iv_30d = pd.to_numeric(
        bq.execute(bql.Request(ticker, {'Value': bq.data.implied_volatility(
            expiry='30D', pct_moneyness='100')}))[0].df()['Value'].iloc[-1],
        errors='coerce') / 100.0

    rv_30d = pd.to_numeric(
        bq.execute(bql.Request(ticker, {'Value': bq.data.volatility_30d_calc()}))[0].df()['Value'].iloc[-1],
        errors='coerce') / 100.0

    put_iv = pd.to_numeric(
        bq.execute(bql.Request(ticker, {'Value': bq.data.implied_volatility(
            expiry='30D', delta='25', put_call='PUT')}))[0].df()['Value'].iloc[-1],
        errors='coerce')
    call_iv = pd.to_numeric(
        bq.execute(bql.Request(ticker, {'Value': bq.data.implied_volatility(
            expiry='30D', delta='25', put_call='CALL')}))[0].df()['Value'].iloc[-1],
        errors='coerce')
    skew = (put_iv - call_iv) / 100.0 if pd.notna(put_iv) and pd.notna(call_iv) else 0.0

    avg_dollar_volume_item = bq.func.avg(
        bq.data.px_volume(dates=bq.func.range('-30D', '0D')) *
        bq.data.px_last(dates=bq.func.range('-30D', '0D')))
    avg_dollar_volume = pd.to_numeric(
        bq.execute(bql.Request(ticker, {'Value': avg_dollar_volume_item}))[0].df()['Value'].iloc[-1],
        errors='coerce')

    # Risk-free rate: 3M US T-Bill yield
    rfr = 0.0
    try:
        rfr = pd.to_numeric(
            bq.execute(bql.Request('USGG3M Index', {'Value': bq.data.px_last()}))[0].df()['Value'].iloc[-1],
            errors='coerce') / 100.0  # BQL retorna em %
        if pd.isna(rfr):
            rfr = 0.0
    except Exception:
        pass

    # MOVE Index (bond vol proxy) para Risk Parity
    move_idx = np.nan
    try:
        move_idx = pd.to_numeric(
            bq.execute(bql.Request('MOVE Index', {'Value': bq.data.px_last()}))[0].df()['Value'].iloc[-1],
            errors='coerce')
    except Exception:
        pass

    return {
        'spot': spot, 'iv_30d': iv_30d, 'rv_30d': rv_30d,
        'skew': skew, 'avg_dollar_volume': avg_dollar_volume,
        'risk_free_rate': rfr, 'move_index': move_idx,
    }


def fetch_options_chain(ticker, spot, min_dte, max_dte, mny_low, mny_high):
    """Busca a cadeia de opções e retorna DataFrame processado."""
    from_strike = (1 + mny_low) * spot
    to_strike = (1 + mny_high) * spot

    conditions = (
        bq.data.expire_dt() >= f'{min_dte}d'
    ).and_(
        bq.data.expire_dt() <= f'{max_dte}d'
    ).and_(
        bq.data.strike_px() > from_strike
    ).and_(
        bq.data.strike_px() < to_strike
    )
    univ = bq.univ.filter(bq.univ.options([ticker]), conditions)

    items = {
        'Expire': bq.data.expire_dt(),
        'Strike': bq.data.strike_px(),
        'Type':   bq.data.put_call(),
        'IV':     bq.data.ivol()['Value'],
        'OI':     bq.data.open_int()['Value'],
    }
    data = pd.concat(
        [r.df() for r in bq.execute(bql.Request(univ, items))], axis=1
    ).dropna()

    if data.empty:
        raise ValueError("Nenhum dado de opção encontrado para os filtros.")

    df = pd.DataFrame({
        'Exp':    pd.to_datetime(data['Expire']),
        'Strike': pd.to_numeric(data['Strike'], errors='coerce'),
        'Type':   data['Type'],
    })
    df['IV'] = pd.to_numeric(data['IV'], errors='coerce')
    df['OI'] = pd.to_numeric(data['OI'], errors='coerce')
    df.dropna(subset=['IV', 'OI', 'Strike'], inplace=True)
    df['IV'] /= 100.0

    today = np.datetime64(datetime.utcnow().date(), 'D')
    bus = np.busday_count(today, df.Exp.dt.normalize().values.astype('datetime64[D]'))
    df['Tte'] = np.maximum(bus, 1) / float(TRADING_DAYS)

    return df, from_strike, to_strike


def fetch_historical(ticker, period='-2Y'):
    """Busca preços históricos e retorna log-retornos."""
    hist_req = bql.Request(ticker, {'Value': bq.data.px_last(
        dates=bq.func.range(period, '0D'), fill='PREV')})
    prices = pd.to_numeric(
        _bql_ts(bq.execute(hist_req)[0], 'Value'), errors='coerce'
    ).dropna()
    if prices.empty:
        raise ValueError("Histórico de preços vazio.")
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if log_returns.empty:
        raise ValueError("Retornos insuficientes.")
    return prices, log_returns


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — CÁLCULOS DE EXPOSIÇÃO E CURVAS MODELO
# ═══════════════════════════════════════════════════════════════════════════════

def compute_strike_exposures(df, greeks, spot):
    """
    Adiciona colunas de exposição por tipo (Call/Put) ao DataFrame
    e retorna DataFrame agregado por strike.
    """
    is_call = (df['Type'] == 'Call').values
    is_put = (df['Type'] == 'Put').values
    oi_100 = df['OI'].values * 100.0

    for cfg in GREEK_CONFIGS:
        key = cfg['key']
        raw = greeks[key]
        df[f'Call_{key}'] = np.where(is_call, raw * oi_100, 0.0)
        df[f'Put_{key}']  = np.where(is_put,  raw * oi_100, 0.0)

    # OI bruto por tipo — necessário para identificar Call/Put Wall por concentração de OI
    df['Call_OI'] = np.where(is_call, df['OI'].values, 0.0)
    df['Put_OI']  = np.where(is_put,  df['OI'].values, 0.0)

    # Agregar por strike — exposições de gregas + OI bruto por tipo
    exp_cols = []
    for cfg in GREEK_CONFIGS:
        exp_cols += [f'Call_{cfg["key"]}', f'Put_{cfg["key"]}']
    exp_cols += ['Call_OI', 'Put_OI']
    agg = df.groupby('Strike')[exp_cols].sum()

    # Computar totais líquidos
    for cfg in GREEK_CONFIGS:
        key = cfg['key']
        scale_val = cfg['scale'](spot)
        call_scaled = agg[f'Call_{key}'] * scale_val
        put_scaled = agg[f'Put_{key}'] * scale_val
        agg[f'Total_{key}'] = cfg['op'](call_scaled, put_scaled) / cfg['div']

    return agg


def compute_model_curves(df, levels, configs=None, r=0.0):
    """
    Calcula curvas modelo (exposure vs. preço spot) para todas as gregas
    em uma única passagem por nível de preço.

    Retorna dict: {greek_name: np.array com a curva}.
    """
    if configs is None:
        configs = GREEK_CONFIGS

    results = {c['name']: [] for c in configs}
    strikes = df['Strike'].values
    ivs = df['IV'].values
    ttes = df['Tte'].values
    ois = df['OI'].values * 100.0
    types = df['Type'].values
    is_call = types == 'Call'
    is_put = types == 'Put'

    for L in levels:
        greeks = calculate_all_greeks(L, strikes, ivs, ttes, types, r=r)
        for cfg in configs:
            key = cfg['key']
            raw = greeks[key]
            call_exp = np.nansum(raw[is_call] * ois[is_call])
            put_exp = np.nansum(raw[is_put] * ois[is_put])
            total = cfg['op'](call_exp, put_exp) * cfg['scale'](L)
            results[cfg['name']].append(total)

    return {name: np.array(curve) for name, curve in results.items()}


def compute_walls(agg):
    """
    Identifica Call Wall e Put Wall pelo maior Open Interest por strike.
    GEX (gamma) é usado apenas como critério secundário em caso de empate
    ou ambiguidade entre strikes com OI muito próximo (dentro de 2% do máximo).
    """
    def _wall(oi_col, gamma_col):
        # Fallback para gamma se OI não estiver disponível
        if oi_col not in agg.columns or agg[oi_col].max() <= 0:
            return agg[gamma_col].idxmax() if gamma_col in agg.columns else None
        max_oi = agg[oi_col].max()
        # Zona de ambiguidade: strikes dentro de 2% do OI máximo
        candidates = agg[agg[oi_col] >= max_oi * 0.98]
        if len(candidates) == 1 or gamma_col not in agg.columns:
            return int(candidates[oi_col].idxmax())
        # Tiebreaker: maior GEX entre os candidatos empatados
        return int(candidates[gamma_col].idxmax())

    call_wall = _wall('Call_OI', 'Call_gamma')
    put_wall  = _wall('Put_OI',  'Put_gamma')
    return call_wall, put_wall


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — MODELOS DE RISCO (VaR, Monte Carlo, P&L)
# ═══════════════════════════════════════════════════════════════════════════════

def fit_risk_model(log_returns):
    """Ajusta distribuição t-Student e calcula VaR/CVaR a 95% e 99%."""
    arr = np.asarray(log_returns, dtype=float)
    tdf, tloc, tscale = student_t.fit(arr)
    print(f"[RISK] t-Student fit: df={tdf:.2f}, loc={tloc:.6f}, scale={tscale:.6f}, n={len(arr)}")

    # Sanity check: se tscale for degenerado, usa parâmetros empíricos
    if tscale < 1e-6 or not np.isfinite(tscale):
        print(f"[RISK] ⚠️ tscale degenerado ({tscale}), usando fallback empírico")
        tloc = float(np.mean(arr))
        tscale = float(np.std(arr))
        tdf = 4.0  # fat-tail conservador

    var_95 = student_t.ppf(0.05, tdf, tloc, tscale)
    cvar_95 = student_t.expect(
        lambda x: x, args=(tdf,), loc=tloc, scale=tscale,
        lb=-np.inf, ub=var_95) / 0.05
    var_99 = student_t.ppf(0.01, tdf, tloc, tscale)
    cvar_99 = student_t.expect(
        lambda x: x, args=(tdf,), loc=tloc, scale=tscale,
        lb=-np.inf, ub=var_99) / 0.01
    print(f"[RISK] VaR 95%={var_95:.4f} ({var_95:.2%}), VaR 99%={var_99:.4f} ({var_99:.2%})")
    return {'tdf': tdf, 'tloc': tloc, 'tscale': tscale,
            'var_95': var_95, 'cvar_95': cvar_95,
            'var_99': var_99, 'cvar_99': cvar_99}


def run_monte_carlo(spot, df, risk_params, n_sims=10000, n_days=5, r=0.0):
    """Simula P&L acumulado do livro do market maker ao longo de n_days."""
    greeks = calculate_all_greeks(spot, df.Strike.values, df.IV.values,
                                  df.Tte.values, df.Type.values, r=r)
    oi100 = df.OI.values * 100
    call_sign = np.where(df.Type.values == 'Call', 1, -1)
    dex = (greeks['delta'] * oi100).sum()
    gex_per_pt = (greeks['gamma'] * call_sign * oi100).sum()
    theta_tot = (greeks['theta'] * oi100).sum()

    cum_pnl = np.zeros(n_sims)
    cur_spot = np.full(n_sims, spot)
    for _ in range(n_days):
        day_rets = student_t.rvs(risk_params['tdf'],
                                 loc=risk_params['tloc'],
                                 scale=risk_params['tscale'], size=n_sims)
        new_spot = cur_spot * (1 + day_rets)
        ds = new_spot - cur_spot
        daily_pnl = -(dex * ds + 0.5 * gex_per_pt * ds ** 2) + theta_tot
        cum_pnl += daily_pnl
        cur_spot = new_spot
    return cum_pnl, cur_spot


def compute_pnl_curves(greeks_now, df, spot, levels, skew, r=0.0):
    """
    Calcula curvas de P&L comparativas:
    - Simplificada (Delta + Gamma)
    - Completa (Delta + Gamma + Vega + Vanna + Zomma)
    - Market vs. Dealer
    """
    oi = df['OI'].values
    delta_vals = greeks_now['delta']
    gamma_vals = greeks_now['gamma']
    vega_vals = greeks_now['vega']
    vanna_vals = greeks_now['vanna']
    zomma_vals = greeks_now['zomma']

    pnl_simple = []
    pnl_complete = []
    market_pnl = []
    hedge_demand = []

    for s in levels:
        dS = s - spot
        dVol = -np.sign(dS) * abs(skew) if skew != 0 else 0

        # Simplificado
        p_s = np.nansum((delta_vals * dS + 0.5 * gamma_vals * dS**2) * oi * 100)

        # Completo (com efeito Zomma no gamma)
        gamma_adj = gamma_vals + zomma_vals * dVol
        p_c = np.nansum(
            (delta_vals * dS
             + 0.5 * gamma_adj * dS**2
             + vega_vals * (dVol * 100)
             + vanna_vals * dS * (dVol * 100)) * oi * 100)

        pnl_simple.append(p_s)
        pnl_complete.append(p_c)
        market_pnl.append(p_s)

        # Demanda de hedge (contratos futuros para ficar delta-neutro)
        greeks_at_s = calculate_all_greeks(s, df.Strike.values, df.IV.values,
                                           df.Tte.values, df.Type.values, r=r)
        mkt_delta = np.nansum(greeks_at_s['delta'] * oi * 100)
        hedge_demand.append(mkt_delta / FUTURES_MULTIPLIER)

    return {
        'simple': np.array(pnl_simple),
        'complete': np.array(pnl_complete),
        'market': np.array(market_pnl),
        'dealer': -np.array(market_pnl),
        'hedge_demand': np.array(hedge_demand),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — REBALANCEAMENTO DE ETFs (passivos + alavancados)
# ═══════════════════════════════════════════════════════════════════════════════

def _last_spx_rebal_date():
    """Última data de rebalanceamento trimestral do S&P 500 (3ª sexta de Mar/Jun/Set/Dez)."""
    today = pd.Timestamp.now().normalize()
    rebal_months = [3, 6, 9, 12]
    candidates = []
    for year in [today.year - 1, today.year]:
        for month in rebal_months:
            first = pd.Timestamp(year, month, 1)
            dow = first.dayofweek  # Monday=0, Friday=4
            days_to_fri = (4 - dow) % 7
            first_friday = first + pd.Timedelta(days=days_to_fri)
            third_friday = first_friday + pd.Timedelta(weeks=2)
            if third_friday < today:
                candidates.append(third_friday)
    return max(candidates) if candidates else today - pd.Timedelta(days=90)


def _float_weights(idx, as_of=None):
    """Retorna pesos float-adjusted normalizados."""
    uni = bq.univ.members([idx], dates=[as_of]) if as_of else bq.univ.members([idx])
    cap = bq.data.cur_mkt_cap(dates=[as_of]) if as_of else bq.data.cur_mkt_cap()
    df_cap = bq.execute(bql.Request(uni, [cap], with_params=BQL_PARAMS)
                        )[0].df().select_dtypes('number').rename(columns=lambda c: 'Cap')
    df_ff = bq.execute(bql.Request(bq.univ.members([idx]),
                                   [bq.data.eqy_free_float_pct()],
                                   with_params=BQL_PARAMS)
                       )[0].df().select_dtypes('number').rename(columns=lambda c: 'FF')
    df_ = df_cap.join(df_ff, how='inner')
    df_['FMC'] = df_['Cap'] * df_['FF'] / 100
    df_['Weight'] = df_['FMC'] / df_['FMC'].sum()
    return df_['Weight']


def _adv5_usd(tickers_list):
    """Média dos últimos 5 dias de $-volume."""
    usd_val = (bq.data.px_volume(dates=bq.func.range('-5D', '-1D'))
               * bq.data.px_last(dates=bq.func.range('-5D', '-1D')))
    adv_item = bq.func.avg(usd_val)
    return bq.execute(bql.Request(bq.univ.List(tickers_list),
                                  {'ADV5': adv_item},
                                  with_params=BQL_PARAMS))[0].df()['ADV5']


def compute_full_etf_flows(index_proxy=INDEX_PROXY, start_date_override=None):
    """
    Calcula fluxos de rebalanceamento por ETF passivo + combinado.
    Retorna (flows_dict, summary_df, start_date).
    flows_dict: {'Combined': df, 'VOO US Equity': df, ...}
    """
    if start_date_override:
        start_date = pd.Timestamp(start_date_override).strftime('%Y-%m-%d')
    else:
        start_date = _last_spx_rebal_date().strftime('%Y-%m-%d')

    w0 = _float_weights(index_proxy, start_date)
    w1 = _float_weights(index_proxy)
    delta = pd.DataFrame({'Start': w0, 'Now': w1}).dropna()
    delta['Delta'] = delta['Now'] - delta['Start']

    adv5 = _adv5_usd(delta.index.tolist()).fillna(1)

    aum = bq.execute(bql.Request(
        bq.univ.List(PASSIVE_ETFS),
        {'AUM': bq.data.fund_total_assets()},
        with_params=BQL_PARAMS))[0].df().select_dtypes('number').iloc[:, 0]

    flows = {}
    for etf, aum_val in aum.items():
        t = delta.copy()
        t['Flow_$'] = t['Delta'] * aum_val
        t['PctADV'] = (t['Flow_$'] / adv5.reindex(t.index).fillna(1)) * 100
        flows[etf] = t[t['Flow_$'] != 0].sort_values('Flow_$', ascending=False)

    combo = delta.copy()
    combo['Flow_$'] = sum(flows[etf].reindex(delta.index, fill_value=0)['Flow_$']
                          for etf in flows)
    combo['PctADV'] = (combo['Flow_$'] / adv5.reindex(combo.index).fillna(1)) * 100
    flows['Combined'] = combo[combo['Flow_$'] != 0].sort_values('Flow_$', ascending=False)

    summary_rows = []
    for etf in PASSIVE_ETFS:
        df_f = flows.get(etf, pd.DataFrame(columns=['Flow_$']))
        buy = df_f.loc[df_f['Flow_$'] > 0, 'Flow_$'].sum() if not df_f.empty else 0
        sell = -df_f.loc[df_f['Flow_$'] < 0, 'Flow_$'].sum() if not df_f.empty else 0
        summary_rows.append({'ETF': etf.split()[0], 'Buy_$': buy, 'Sell_$': sell, 'Net_$': buy - sell})
    summary = pd.DataFrame(summary_rows).set_index('ETF')

    return flows, summary, start_date


def compute_leveraged_flows(daily_return):
    """
    Calcula fluxo de rebalanceamento end-of-day dos ETFs alavancados.
    Fórmula: Rebalance_$ = AUM × L × (L-1) × r / (1 + L×r)
    """
    tickers = [e['ticker'] for e in LEVERAGED_ETFS]
    print(f"[LEV] daily_return={daily_return:.6f}")
    aum_df = None
    try:
        aum_df = bq.execute(bql.Request(
            tickers,
            {'AUM': bq.data.fund_total_assets()},
            with_params=BQL_PARAMS))[0].df()
        print(f"[LEV] AUM fetched: {aum_df.shape}")
    except Exception as e:
        print(f"[LEV] AUM fetch failed: {e}")

    rows = []
    for etf in LEVERAGED_ETFS:
        tk = etf['ticker']
        L = etf['leverage']
        r = daily_return
        try:
            aum = pd.to_numeric(aum_df.loc[tk].iloc[0], errors='coerce') if aum_df is not None else np.nan
        except (KeyError, TypeError, IndexError):
            aum = np.nan
        if pd.isna(aum):
            aum = DEFAULT_AUM.get(etf['name'], 1e9)

        if pd.notna(aum) and (1 + L * r) != 0:
            rebal = aum * L * (L - 1) * r / (1 + L * r)
        else:
            rebal = np.nan

        rows.append({
            'ETF': etf['name'],
            'Leverage': f"{L:+d}x",
            'AUM_$': aum,
            'Rebalance_$': rebal,
            'Direção': ('COMPRAR' if pd.notna(rebal) and rebal > 0
                        else 'VENDER' if pd.notna(rebal) and rebal < 0
                        else 'N/A')
        })

    result = pd.DataFrame(rows).set_index('ETF')
    total_flow = result['Rebalance_$'].sum()
    print(f"[LEV] Total flow: ${total_flow:,.0f}")
    return result, total_flow


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5B — PREVISÃO DE REBALANCEAMENTO DO S&P 500 (inclusão/exclusão)
# ═══════════════════════════════════════════════════════════════════════════════

def build_spx_prediction():
    """
    Constrói modelo de previsão de inclusão/exclusão do S&P 500.
    Usa 4 variáveis: FMC, FALR (annual liquidity ratio), FREE_FLOAT_PCT, NET_INC_TTM.
    Retorna (top_in, top_out, model_df).
    Requer sklearn.
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn não disponível. Instale com: pip install scikit-learn")

    def _fetch(u, item, name):
        df_ = bq.execute(bql.Request(u, item))[0].df()
        df_ = df_.select_dtypes('number').iloc[:, 0].to_frame(name)
        return df_.loc[~df_.index.duplicated(keep='first')]

    # 1 — Universos
    spx_ids = bq.execute(
        bql.Request(bq.univ.members(['SPX Index']), bq.data.id())
    )[0].df().index.tolist()

    # S&P 500: market cap mínimo ~$18B (regra atualizada)
    base_ids = bq.execute(
        bql.Request(
            bq.univ.filter(
                bq.univ.equitiesuniv(['active', 'primary']),
                bq.func.and_(
                    bq.data.cntry_of_domicile() == 'US',
                    bq.data.cur_mkt_cap() > 1.8e10)),
            bq.data.id())
    )[0].df().index.tolist()

    neg_ids = [t for t in base_ids if t not in spx_ids]
    _random.seed(42)
    neg_sample = _random.sample(neg_ids, k=min(len(neg_ids), len(spx_ids) * 2))
    universe = spx_ids + neg_sample

    # 2 — ADVT anual
    advt_series = (
        bq.data.px_volume(dates=bq.func.range('-365D', '0D')) *
        bq.data.px_last(dates=bq.func.range('-365D', '0D')))
    advt_item = bq.func.sum(advt_series)

    # 3 — Coleta em batches
    records = []
    batch_size = 50
    for i in range(0, len(universe), batch_size):
        u = bq.univ.List(universe[i:i + batch_size])
        dfb = pd.concat([
            _fetch(u, bq.data.cur_mkt_cap(), 'CUR_MKT_CAP'),
            _fetch(u, bq.data.eqy_free_float_pct(), 'FREE_FLOAT_PCT'),
            _fetch(u, advt_item, 'ADVT'),
            _fetch(u, bq.data.net_income(fa_period_offset='0'), 'NI_Q0'),
            _fetch(u, bq.data.net_income(fa_period_offset='1'), 'NI_Q1'),
            _fetch(u, bq.data.net_income(fa_period_offset='2'), 'NI_Q2'),
            _fetch(u, bq.data.net_income(fa_period_offset='3'), 'NI_Q3'),
        ], axis=1, join='inner')
        records.append(dfb)

    df_ = pd.concat(records)

    # 4 — Feature engineering
    df_['FMC'] = df_['CUR_MKT_CAP'] * df_['FREE_FLOAT_PCT'] / 100
    df_['FALR'] = df_['ADVT'] / df_['FMC']
    df_['NET_INC_TTM'] = df_[['NI_Q0', 'NI_Q1', 'NI_Q2', 'NI_Q3']].sum(axis=1)

    roots_all = df_.index.to_series().str.split().str[0]
    roots_spx = pd.Series(spx_ids).str.split().str[0]
    df_['IN_SPX'] = roots_all.isin(set(roots_spx)).astype(int)
    df_ = df_.dropna()

    # 4b — Flag de elegibilidade S&P 500
    # Regras: Free float >= 50%, lucro TTM > 0, último trimestre > 0, FALR >= 0.75
    df_['ELIGIBLE'] = (
        (df_['FREE_FLOAT_PCT'] >= 50) &
        (df_['NET_INC_TTM'] > 0) &
        (df_['NI_Q0'] > 0) &
        (df_['FALR'] >= 0.75) &
        (df_['CUR_MKT_CAP'] >= 1.8e10)
    ).astype(int)

    # 5 — Modelo LogisticRegression
    features = ['FMC', 'FALR', 'FREE_FLOAT_PCT', 'NET_INC_TTM']
    X, y = df_[features], df_['IN_SPX']
    pipe = SkPipeline([
        ('scaler', StandardScaler()),
        ('logit', LogisticRegression(solver='liblinear'))
    ])
    pipe.fit(X, y)
    df_['Prob_In'] = pipe.predict_proba(X)[:, 1]

    # 6 — ExitScore (penalidade por cap, FMC, lucro, liquidez)
    #     Thresholds atualizados conforme metodologia S&P
    cap_thr = 1.8e10   # mínimo de market cap para permanência
    fmc_thr = 0.9e10   # mínimo de float-adjusted market cap
    cap_pen = ((cap_thr - df_['CUR_MKT_CAP']).clip(lower=0)) / cap_thr
    fmc_pen = ((fmc_thr - df_['FMC']).clip(lower=0)) / fmc_thr
    earn_pen = ((df_['NET_INC_TTM'] <= 0) | (df_['NI_Q0'] <= 0)).astype(float)
    float_pen = ((50 - df_['FREE_FLOAT_PCT']).clip(lower=0)) / 50
    liq_pen = ((0.75 - df_['FALR']).clip(lower=0)) / 0.75
    df_['ExitScore'] = (cap_pen + fmc_pen + earn_pen + float_pen + liq_pen) / 5

    # 7 — Rankings
    allowed = df_.index.str.endswith((' US Equity', ' UW Equity', ' UQ Equity'))

    def _dedup_root(d, col):
        t = d.copy()
        t['root'] = t.index.str.split().str[0]
        return t.loc[t.groupby('root')[col].idxmax()]

    # Inclusão: somente elegíveis (FREE_FLOAT>=50, lucro positivo, FALR>=0.75)
    top_in = (df_[(df_['IN_SPX'] == 0) & allowed & (df_['ELIGIBLE'] == 1)]
              .pipe(_dedup_root, 'Prob_In')
              .sort_values('Prob_In', ascending=False)
              .head(30))

    top_out = (df_[(df_['IN_SPX'] == 1) & allowed]
               .pipe(_dedup_root, 'ExitScore')
               .sort_values('ExitScore', ascending=False)
               .head(30))

    return top_in, top_out, df_


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5C — COT ENGINE (Commitment of Traders)
# ═══════════════════════════════════════════════════════════════════════════════

def _fp_last_numeric_col(df_in):
    for c in df_in.columns[::-1]:
        if pd.api.types.is_numeric_dtype(df_in[c]):
            return c
    return None


def _fp_get_data(universe, data_items, with_params=None, preferences=None):
    """Executa BQL Request e retorna DataFrame com multi-index [ID, Date]."""
    if isinstance(universe, str):
        universe = [universe]
    req = bql.Request(universe, data_items,
                      with_params=with_params, preferences=preferences)
    response = bq.execute(req)
    df_r = pd.concat([di.df() for di in response], axis=1)
    df_r = df_r.loc[:, ~df_r.columns.duplicated()]
    if 'DATE' in df_r.columns:
        df_r = df_r.set_index('DATE', append=True)
    # Selecionar colunas disponíveis (ignorar as não retornadas)
    avail = [k for k in data_items.keys() if k in df_r.columns]
    if avail:
        df_r = df_r[avail]
    df_r.index.names = ['ID', 'Date']
    return df_r


# ── Helpers COT v4 ──
QUARTERLY_MONTHS = {3: 'H', 6: 'M', 9: 'U', 12: 'Z'}

SPEC_TRADER_TYPES = {
    'cftc_disaggregated': ['MANAGED_MONEY'],
    'cftc_tff': ['ASSET_MANAGER', 'LEVERAGED_FUNDS'],
    'cftc_legacy': ['NON_COMMERCIAL'],
}


def _concat_bql_response(resp):
    """Concat DataItems de um BQL response, usando integer index."""
    dfs = [di.df().reset_index() for di in resp]
    raw = pd.concat([d.reset_index(drop=True) for d in dfs], axis=1)
    return raw.loc[:, ~raw.columns.duplicated()]


def _has_positions_data(df):
    """Verifica se o BQL result tem dados reais de posição."""
    if df is None or df.empty:
        return False
    for c in df.columns:
        cl = str(c).lower()
        if 'value' in cl or 'position' in cl:
            vals = pd.to_numeric(df[c], errors='coerce')
            if vals.notna().any() and (vals != 0).any():
                return True
    return False


def resolve_specific_contract(generic_ticker):
    """ES1 Index → ['ESH6 Index', 'ESM6 Index'] baseado na data atual."""
    parts = generic_ticker.split()
    if len(parts) < 2:
        return []
    root_num = parts[0]
    suffix = ' '.join(parts[1:])
    root = root_num.rstrip('0123456789')
    if not root:
        return []
    now = pd.Timestamp.now()
    month, year = now.month, now.year
    sorted_qm = sorted(QUARTERLY_MONTHS.keys())
    candidates = []
    for qm in sorted_qm:
        if month <= qm:
            code = QUARTERLY_MONTHS[qm]
            candidates.append(f"{root}{code}{year % 10} {suffix}")
            if len(candidates) >= 2:
                break
    if len(candidates) < 2:
        next_year = (year + 1) % 10
        for qm in sorted_qm:
            code = QUARTERLY_MONTHS[qm]
            candidates.append(f"{root}{code}{next_year} {suffix}")
            if len(candidates) >= 2:
                break
    print(f"[resolve] {generic_ticker} → candidatos: {candidates}")
    return candidates


def _to_bql_date(s):
    """Converte data YYYYMMDD (do widget) para formato BQL relativo."""
    if not s or not isinstance(s, str):
        return s
    if s in ('0D', '0d') or (s.startswith('-') and s[-1] in 'YyDd'):
        return s
    try:
        dt = pd.to_datetime(s)
        days = (pd.Timestamp.now() - dt).days
        if days <= 0:
            return '0D'
        years = days / 365.25
        if years >= 1:
            return f'-{int(years) + 1}Y'
        return f'-{days}D'
    except Exception:
        return s


def _try_cot_query(ticker, rpt, start, end, with_dates=True):
    """Tenta uma query COT. Retorna DataFrame ou None."""
    try:
        bql_s, bql_e = _to_bql_date(start), _to_bql_date(end)
        if with_dates:
            q = f"""
let(#p = cot_position(report_type={rpt}, direction=all,
                      trader_type=all, commitment_type=futures,
                      dates=range({bql_s},{bql_e}));)
for('{ticker}') get(#p().date, #p().trader_type, #p().direction,
                    #p().value, #p().change)
"""
        else:
            q = f"""
let(#p = cot_position(report_type={rpt}, direction=all,
                      trader_type=all, commitment_type=futures);)
for('{ticker}') get(#p().date, #p().trader_type, #p().direction,
                    #p().value, #p().change)
"""
        label = f"{rpt} + {ticker}" + (" + dates" if with_dates else "")
        print(f"[COT] Tentando {label}…")
        resp = bq.execute(q)
        raw = _concat_bql_response(resp)
        print(f"[COT]   shape={raw.shape}")
        if not _has_positions_data(raw):
            print(f"[COT]   → sem dados de posição reais")
            return None
        print(f"[COT]   ✓ has real data")
        return raw
    except Exception as e:
        print(f"[COT]   ✗ {e}")
        return None


def has_cot(ticker):
    """Verifica se o ticker possui dados COT. Retorna (True, futures_ticker) ou (False, None)."""
    if ticker in COT_FUTURES_MAP:
        return True, COT_FUTURES_MAP[ticker]
    t = ticker.strip()
    if t.endswith('Comdty') or t.endswith('Index'):
        return True, ticker
    return False, None


def fetch_cot_data(futures_ticker, start='-2Y', end='0D'):
    """Busca COT histórico: tenta genérico + específico, filtra spec traders."""
    tickers = [futures_ticker] if isinstance(futures_ticker, str) else list(futures_ticker)

    # Build lista de tickers para tentar: genérico + contratos específicos
    tickers_to_try = list(tickers)
    for t in tickers:
        specific = resolve_specific_contract(t)
        tickers_to_try.extend(specific)

    raw = None
    used_rpt = ''
    used_ticker = ''

    for t in tickers_to_try:
        for rpt in ('cftc_disaggregated', 'cftc_tff', 'cftc_legacy'):
            result = _try_cot_query(t, rpt, start, end, with_dates=True)
            if result is None:
                result = _try_cot_query(t, rpt, start, end, with_dates=False)
            if result is not None:
                raw = result
                used_rpt = rpt
                used_ticker = t
                break
        if raw is not None:
            break

    if raw is None:
        print("[COT] Nenhum dado retornado")
        return pd.DataFrame()

    # ── Renomear colunas ──
    col_map = {}
    for c in raw.columns:
        cl = str(c).lower()
        if 'date' in cl:
            col_map[c] = 'Date'
        elif 'trader_type' in cl:
            col_map[c] = 'TraderType'
        elif 'direction' in cl:
            col_map[c] = 'Direction'
        elif 'change' in cl:
            col_map[c] = 'Pos_Chg'
        elif 'value' in cl:
            col_map[c] = 'Positions'
    raw = raw.rename(columns=col_map)
    if 'ID' not in raw.columns:
        raw['ID'] = used_ticker
    if 'Date' in raw.columns:
        raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce')
    for nc in ('Positions', 'Pos_Chg'):
        if nc in raw.columns:
            raw[nc] = pd.to_numeric(raw[nc], errors='coerce')

    if 'Direction' in raw.columns:
        raw['Direction'] = raw['Direction'].astype(str).str.strip().str.upper()
    if 'TraderType' in raw.columns:
        raw['TraderType'] = raw['TraderType'].astype(str).str.strip().str.upper()

    print(f"[COT] Rename: cols={list(raw.columns)}, report={used_rpt}, ticker={used_ticker}")

    # ── Filtrar trader types especulativos ──
    spec_types = SPEC_TRADER_TYPES.get(used_rpt, [])
    if 'TraderType' in raw.columns and spec_types:
        n_before = len(raw)
        raw = raw[raw['TraderType'].isin(spec_types)]
        print(f"[COT] Filtro spec ({spec_types}): {n_before} → {len(raw)} rows")
    if raw.empty:
        print("[COT] Vazio após filtro de trader types")
        return pd.DataFrame()

    # ── Drop rows com Positions NaN/zero (dias sem report) ──
    if 'Positions' in raw.columns:
        raw = raw.dropna(subset=['Positions'])
        raw = raw[raw['Positions'] != 0]
        print(f"[COT] Após drop zeros/NaN: {len(raw)} rows")
    if raw.empty:
        print("[COT] Vazio após drop zeros")
        return pd.DataFrame()

    # ── Filtrar datas ──
    def _pdt(s):
        if not s:
            return None
        s = str(s)
        if s in ('0D', '0d'):
            return pd.Timestamp.now()
        if s.startswith('-') and s[-1] in 'Yy':
            return pd.Timestamp.now() - pd.DateOffset(years=int(s[1:-1]))
        if s.startswith('-') and s[-1] in 'Dd':
            return pd.Timestamp.now() - pd.Timedelta(days=int(s[1:-1]))
        return pd.to_datetime(s, errors='coerce')
    if 'Date' in raw.columns:
        dt_s, dt_e = _pdt(start), _pdt(end)
        if dt_s is not None:
            raw = raw[raw['Date'] >= dt_s]
        if dt_e is not None:
            raw = raw[raw['Date'] <= dt_e]
    if raw.empty:
        print("[COT] Vazio após filtro de datas")
        return pd.DataFrame()

    # ── Pivot: spec traders por Date × Direction ──
    idx_cols = [c for c in ['ID', 'Date'] if c in raw.columns]
    if not idx_cols or 'Positions' not in raw.columns:
        print(f"[COT] Colunas insuficientes: {list(raw.columns)}")
        return pd.DataFrame()

    if 'Direction' in raw.columns:
        grouped = raw.groupby(idx_cols + ['Direction'])['Positions'].sum().reset_index()
        piv = grouped.pivot_table(index=idx_cols, columns='Direction',
                                  values='Positions', aggfunc='sum')
        piv.columns = [f'Positions - {c.title()}' for c in piv.columns]
        df = piv
    else:
        df = raw.groupby(idx_cols)['Positions'].sum().to_frame()

    # ── Compute Net = Long + Short ──
    if 'Positions - Net' in df.columns:
        df['Positions'] = df['Positions - Net']
    elif 'Positions - Long' in df.columns and 'Positions - Short' in df.columns:
        df['Positions'] = df['Positions - Long'] + df['Positions - Short']
    elif 'Positions - Long' in df.columns:
        df['Positions'] = df['Positions - Long']
    else:
        df['Positions'] = df.iloc[:, 0]

    df = df[df['Positions'] != 0]
    if df.empty:
        print("[COT] Vazio após drop net == 0")
        return pd.DataFrame()

    # ── Price & Open Interest (usa genérico para preço) ──
    price_ticker = futures_ticker if isinstance(futures_ticker, str) else futures_ticker[0]
    try:
        dates_bql = bq.func.range(_to_bql_date(start), _to_bql_date(end), frq='d')
        px_items = {'Price': bq.data.px_last(fill='prev'),
                    'Open Interest': bq.data.fut_aggte_open_int()}
        px_df = _fp_get_data(price_ticker, px_items,
                             with_params={'currency': 'usd', 'dates': dates_bql})
        px_sr = px_df.droplevel('ID').sort_index()
        cot_dates = df.index.get_level_values('Date')
        px_a = px_sr.reindex(cot_dates, method='ffill')
        for c in ['Price', 'Open Interest']:
            if c in px_a.columns:
                df[c] = px_a[c].values
    except Exception:
        df['Price'] = np.nan
        df['Open Interest'] = np.nan

    # ── week/year ──
    dt_idx = pd.to_datetime(df.index.get_level_values('Date'))
    iso = dt_idx.isocalendar()
    df['week'] = iso.week.values
    df['year'] = iso.year.values
    if 'Positions - Short' in df.columns:
        df['Positions - Short'] = df['Positions - Short'] * -1
    print(f"[COT] Final: {df.shape}, cols: {list(df.columns)}")
    return df.dropna(subset=['Positions'])


def aggregate_cot(df_cot):
    """Agrega COT por data (caso de múltiplos contratos)."""
    cols_skip = ['year', 'week', 'Price']
    aggs = {c: 'sum' for c in df_cot.columns if c not in cols_skip}
    out = df_cot.groupby('Date').agg(
        **{**{c: (c, agg) for c, agg in aggs.items()},
           'week': ('week', 'first'),
           'year': ('year', 'first'),
           'Price': ('Price', 'mean')})
    out['Basket Returns'] = (1 + out['Price'].pct_change()).cumprod() - 1
    return out


def cot_seasonality(df_cot):
    """Estatísticas semanais de sazonalidade COT."""
    cols = ['Positions', 'Open Interest', 'week']
    avail = [c for c in cols if c in df_cot.columns]
    return (df_cot[avail].groupby('week')
            .agg(['mean', 'max', 'sum', 'min'])
            .rename(columns=lambda x: x.title()))


def cot_summary_stats(df_cot):
    """Estatísticas resumo: último valor, WoW change, percentil, mediana, z-score."""
    summary = pd.Series(dtype=float)
    for col in ['Positions']:
        if col not in df_cot.columns:
            continue
        s = df_cot[col].dropna()
        if len(s) < 3:
            continue
        stats = pd.Series({
            col: s.iloc[-1],
            f'{col} WoW Change': s.iloc[-1] - s.iloc[-2] if len(s) >= 2 else np.nan,
            f'{col} Percentile': (s.rank(pct=True).iloc[-1] * 100),
            f'{col} Median': s.median(),
            f'{col} Z-Score': (s.iloc[-1] - s.mean()) / max(s.std(), 1e-9),
        })
        summary = pd.concat([summary, stats])
    return summary


def safe_fetch_cot(ticker, start='-2Y', end='0D'):
    """Tenta buscar COT. Retorna None se não disponível."""
    ok, fut = has_cot(ticker)
    if not ok:
        return None
    try:
        df_r = fetch_cot_data(fut, start=start, end=end)
        if df_r is None or df_r.empty:
            return None
        return aggregate_cot(df_r)
    except Exception as e:
        print(f"⚠️ COT fetch error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5D — ESTIMATIVA DE BUYBACK + BLACKOUT WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

# Blackout: empresas não podem recomprar ações de ~28 dias antes do balanço
# até ~2 dias úteis após a divulgação. Quando muitas estão em blackout, o
# fluxo de buyback cai significativamente.
BLACKOUT_DAYS_BEFORE = 28  # ~4 semanas antes do earnings
BLACKOUT_DAYS_AFTER = 2    # ~2 dias após divulgação


def fetch_earnings_dates(index_ticker='SPX Index'):
    """Busca datas de earnings esperadas dos membros do índice via BQL.
    Retorna DataFrame com coluna 'earn_dt' (datetime)."""
    try:
        uni = bq.univ.members(index_ticker)
        # Tentar EXPECTED_REPORT_DT primeiro, depois EARN_ANN_DT_NEXT_ACTUAL
        for fld_name in ['expected_report_dt', 'earn_ann_dt_next_actual',
                         'next_announce_dt']:
            try:
                fld = getattr(bq.data, fld_name, None)
                if fld is None:
                    continue
                req = bql.Request(uni, {'earn_dt': fld()})
                resp = bq.execute(req)
                df_r = resp[0].df()
                if df_r is not None and not df_r.empty:
                    # Encontrar coluna de data
                    for c in df_r.columns:
                        if 'earn' in str(c).lower() or 'dt' in str(c).lower() or 'announce' in str(c).lower():
                            df_r['earn_dt'] = pd.to_datetime(df_r[c], errors='coerce')
                            break
                    else:
                        df_r['earn_dt'] = pd.to_datetime(df_r.iloc[:, 0], errors='coerce')
                    df_r = df_r.dropna(subset=['earn_dt'])
                    if len(df_r) > 10:
                        return df_r[['earn_dt']]
            except Exception:
                continue
    except Exception:
        pass
    return pd.DataFrame()


def compute_blackout_curve(earnings_df, n_days_forward=365):
    """Calcula curva de blackout: para os próximos N dias, quantas empresas do
    SPX estão em janela de restrição de buyback.

    BQL retorna apenas o PRÓXIMO earnings de cada empresa. Para projetar o ano
    inteiro, replicamos cada data trimestralmente (~91 dias) para frente.

    Returns:
        DataFrame com colunas: date, n_blackout, pct_blackout
    """
    if earnings_df.empty:
        return pd.DataFrame()

    today = pd.Timestamp.now().normalize()
    horizon = today + pd.Timedelta(days=n_days_forward)

    # Projetar earnings trimestrais a partir da data conhecida
    projected = []
    for dt in earnings_df['earn_dt']:
        if pd.isna(dt):
            continue
        d = pd.Timestamp(dt).normalize()
        # Gerar datas trimestrais para trás e para frente
        for q in range(-1, 5):  # -1 trimestre atrás até +4 à frente
            qd = d + pd.Timedelta(days=91 * q)
            if (today - pd.Timedelta(days=60)) <= qd <= horizon:
                projected.append(qd.to_datetime64())
    earn_dates_all = np.array(projected)
    total_companies = len(earnings_df)

    dates = pd.date_range(today - pd.Timedelta(days=30),
                          today + pd.Timedelta(days=n_days_forward))
    records = []
    for d in dates:
        # Uma empresa está em blackout se: earn_dt - 28 <= d <= earn_dt + 2
        in_blackout = np.sum(
            (earn_dates_all >= (d - pd.Timedelta(days=BLACKOUT_DAYS_AFTER)).to_datetime64()) &
            (earn_dates_all <= (d + pd.Timedelta(days=BLACKOUT_DAYS_BEFORE)).to_datetime64())
        )
        # Limitar ao total (pode ter duplicatas por projeção)
        in_blackout = min(int(in_blackout), total_companies)
        records.append({
            'date': d,
            'n_blackout': in_blackout,
            'pct_blackout': in_blackout / total_companies if total_companies > 0 else 0
        })
    return pd.DataFrame(records)


def blackout_pct_today(earnings_df):
    """Retorna % de empresas atualmente em blackout."""
    if earnings_df.empty:
        return 0.0, 0, 0
    today = pd.Timestamp.now().normalize()
    earn_dates = earnings_df['earn_dt'].values
    total = len(earnings_df)
    in_blackout = int(np.sum(
        (earn_dates >= (today - pd.Timedelta(days=BLACKOUT_DAYS_AFTER)).to_datetime64()) &
        (earn_dates <= (today + pd.Timedelta(days=BLACKOUT_DAYS_BEFORE)).to_datetime64())
    ))
    in_blackout = min(in_blackout, total)
    return in_blackout / total if total > 0 else 0, in_blackout, total

@lru_cache(maxsize=128)
def fetch_buyback_data(ticker):
    """Busca dados de buyback via BQL. Retorna dict com campos disponíveis."""
    fields_try = {
        'announced': 'ANNOUNCED_BUYBACK_AMT',
        'mkt_cap': 'CUR_MKT_CAP',
        'sh_out': 'EQY_SH_OUT',
        'px': 'PX_LAST',
        'adv20': 'VOLUME_AVG_20D',
    }
    result = {}
    for key, field_name in fields_try.items():
        try:
            fld = getattr(bq.data, field_name.lower(), None)
            if fld is None:
                continue
            req = bql.Request([ticker], fld())
            df_r = bq.execute(req)[0].df()
            if df_r is not None and not df_r.empty:
                vcol = _fp_last_numeric_col(df_r)
                if vcol:
                    val = pd.to_numeric(df_r[vcol], errors='coerce').dropna()
                    if len(val) > 0:
                        result[key] = float(val.iloc[-1])
        except Exception:
            continue
    return result


def estimate_buyback_flow(ticker, horizon_days=252):
    """
    Estima fluxo diário de buyback.
    Usa _adv5_usd para % ADV (mesma lógica do rebalanceamento).
    """
    data = fetch_buyback_data(ticker)
    announced = data.get('announced', 0)
    if not announced or announced <= 0:
        return {'daily_est': 0, 'pct_adv_est': 0,
                'confidence': 'none', 'announced': 0}
    execution_rate = 0.80
    daily_est = (announced * execution_rate) / max(horizon_days, 1)
    try:
        adv5 = _adv5_usd([ticker])
        adv_usd = float(adv5.iloc[0]) if len(adv5) > 0 else 0
    except Exception:
        adv_usd = 0
    pct_adv = (announced / adv_usd * 100) if adv_usd > 0 else np.nan
    return {'daily_est': daily_est, 'pct_adv_est': pct_adv,
            'confidence': 'low', 'announced': announced,
            'mkt_cap': data.get('mkt_cap', np.nan)}


def estimate_index_buyback_flow(index_ticker='SPX Index', top_n=50):
    """Estima fluxo de buyback para os maiores membros de um índice.

    Tenta buscar announced_buyback_amt via BQL. Se não disponível,
    faz fallback para estimativa baseada em market cap (~2% cap/ano).
    Usa _adv5_usd (mesmo ADV do rebalanceamento) para calcular % ADV.
    """
    try:
        uni = bq.univ.members(index_ticker)
        items = {'cap': bq.data.cur_mkt_cap()}

        # Tentar múltiplos campos de buyback
        bb_field_name = None
        for fname in ['announced_buyback_amt', 'bs_sh_repurchase']:
            try:
                fld = getattr(bq.data, fname, None)
                if fld is not None:
                    items['buyback_raw'] = fld()
                    bb_field_name = fname
                    break
            except Exception:
                continue

        req = bql.Request(uni, items, with_params=BQL_PARAMS)
        df_r = bq.execute(req)[0].df()
        if df_r is None or df_r.empty:
            return pd.DataFrame()

        # Renomear colunas robustamente
        rename = {}
        for c in df_r.columns:
            cl = str(c).lower()
            if 'mkt' in cl and 'cap' in cl:
                rename[c] = 'cap'
            elif 'buyback' in cl or 'repurchase' in cl:
                rename[c] = 'buyback'
        df_r = df_r.rename(columns=rename)

        if 'cap' in df_r.columns:
            df_r['cap'] = pd.to_numeric(df_r['cap'], errors='coerce')
            df_r = df_r.dropna(subset=['cap'])
            df_r = df_r.nlargest(top_n, 'cap')

        # Verificar se dados de buyback reais estão disponíveis
        has_bb = ('buyback' in df_r.columns and
                  pd.to_numeric(df_r['buyback'], errors='coerce').abs().sum() > 0)

        if has_bb:
            df_r['buyback'] = pd.to_numeric(df_r['buyback'], errors='coerce').fillna(0).abs()
            df_r['confidence'] = 'low'
        else:
            df_r['buyback'] = df_r['cap'].fillna(0) * 0.02
            df_r['confidence'] = 'estimated'

        df_r['daily_est'] = df_r['buyback'] * 0.80 / TRADING_DAYS

        # % ADV usando _adv5_usd (mesma lógica do rebalanceamento)
        try:
            tickers_list = df_r.index.get_level_values('ID').unique().tolist()
            adv5 = _adv5_usd(tickers_list)
            df_r['pct_adv_est'] = (df_r['buyback']
                                   / adv5.reindex(df_r.index).replace(0, np.nan)) * 100
        except Exception:
            df_r['pct_adv_est'] = np.nan

        out_cols = [c for c in ['cap', 'buyback', 'daily_est', 'pct_adv_est', 'confidence']
                    if c in df_r.columns]
        return df_r[out_cols]
    except Exception:
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5E — FLOW SCORING & AGREGAÇÃO PREDITIVA
# ═══════════════════════════════════════════════════════════════════════════════

def flow_zscore(current, history):
    """Z-score do valor atual relativo ao histórico."""
    if history is None or len(history) < 5:
        return np.nan
    mu = history.mean()
    sigma = history.std()
    if sigma < 1e-12:
        return 0.0
    return (current - mu) / sigma


def compute_leveraged_flow_simple(daily_return, aum_estimates=None):
    """Calcula fluxo total de rebalanceamento sem BQL call."""
    if aum_estimates is None:
        aum_estimates = DEFAULT_AUM
    total = 0
    for etf in LEVERAGED_ETFS_EXT:
        L = etf['leverage']
        r = daily_return
        aum = aum_estimates.get(etf['name'], 1e9)
        denom = 1 + L * r
        if abs(denom) > 1e-12:
            total += aum * L * (L - 1) * r / denom
    return total


def build_flow_history(ticker='SPX Index', lookback=252):
    """Constrói série histórica de fluxo de rebalanceamento de ETFs alavancados."""
    item = bq.data.px_last(dates=bq.func.range(f'-{lookback + 10}D', '0D'), fill='PREV')
    try:
        df_r = bq.execute(bql.Request([ticker], item))[0].df()
        if df_r is None or df_r.empty:
            return pd.DataFrame()
        dcol = next((c for c in df_r.columns if 'date' in str(c).lower()), None)
        vcol = _fp_last_numeric_col(df_r)
        if dcol is None or vcol is None:
            return pd.DataFrame()
        px = df_r.set_index(dcol)[vcol].astype(float).sort_index()
        px.index = pd.to_datetime(px.index)
    except Exception:
        return pd.DataFrame()
    rets = px.pct_change().dropna()
    flows = rets.apply(compute_leveraged_flow_simple)
    flows.name = 'LevETF_Flow'
    return pd.DataFrame({'Return': rets, 'LevETF_Flow': flows})


def compute_flow_score(leveraged_flow, buyback_daily=0, cot_net_change=0,
                       passive_etf_flow=0, history_leveraged=None,
                       history_cot=None, dealer_flow=0, volctrl_flow=0,
                       cta_flow=0, rp_flow=0,
                       history_dealer=None, history_volctrl=None):
    """
    Computa score combinado de fluxo contratado — 8 componentes.
    Pesos: CTA 22%, Dealer 18%, VolCtrl 12%, Risk Parity 12%,
           ETFs Alav 12%, Buyback 8%, COT 10%, ETFs Passivos 6%.
    """
    z_lev = flow_zscore(leveraged_flow, history_leveraged) if history_leveraged is not None else 0
    z_cot = flow_zscore(cot_net_change, history_cot) if history_cot is not None else 0
    z_buyback = np.clip(buyback_daily / 1e8, -3, 3) if buyback_daily else 0
    z_passive = np.clip(passive_etf_flow / 1e9, -3, 3) if passive_etf_flow else 0
    z_dealer = np.clip(dealer_flow / 1e9, -3, 3) if dealer_flow else 0
    z_volctrl = np.clip(volctrl_flow / 1e9, -3, 3) if volctrl_flow else 0
    z_cta = np.clip(cta_flow / 1e9, -3, 3) if cta_flow else 0
    z_rp = np.clip(rp_flow / 1e9, -3, 3) if rp_flow else 0

    w_cta, w_deal, w_vc, w_rp = 0.22, 0.18, 0.12, 0.12
    w_lev, w_buy, w_cot, w_passive = 0.12, 0.08, 0.10, 0.06
    if history_cot is None or len(history_cot) < 5:
        w_cot = 0.0
        w_cta, w_deal, w_rp = 0.26, 0.20, 0.14
        w_vc, w_lev, w_buy, w_passive = 0.14, 0.14, 0.08, 0.04

    combined = (w_lev * z_lev + w_buy * z_buyback + w_cot * z_cot
                + w_passive * z_passive + w_deal * z_dealer + w_vc * z_volctrl
                + w_cta * z_cta + w_rp * z_rp)

    if combined > 0.5:
        direction = "BULLISH"
    elif combined < -0.5:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    prob_up = 1.0 / (1.0 + math.exp(-combined))
    score_0_100 = round(prob_up * 100, 1)
    return {
        'z_leveraged': z_lev, 'z_buyback': z_buyback,
        'z_cot': z_cot, 'z_passive_etf': z_passive,
        'z_dealer': z_dealer, 'z_volctrl': z_volctrl,
        'z_cta': z_cta, 'z_rp': z_rp,
        'combined_score': combined, 'direction': direction,
        'prob_up': prob_up, 'prob_down': 1.0 - prob_up,
        'score': score_0_100, 'score_total': score_0_100,
        'weights': {'leveraged': w_lev, 'buyback': w_buy,
                    'cot': w_cot, 'passive_etf': w_passive,
                    'dealer': w_deal, 'volctrl': w_vc,
                    'cta': w_cta, 'rp': w_rp},
    }


def compute_dealer_hedging_flow(gex_per_pt, daily_price_change, spot):
    """
    Estima fluxo de hedging de dealers/market makers em opções.
    Quando dealers estão short gamma, um move de +1% os força a comprar
    (delta hedge), amplificando o movimento. Short gamma → pro-cíclico.
    Flow ≈ -GEX_per_pt × ΔS (dealers hedgeiam no sentido oposto à gamma).
    """
    if pd.isna(gex_per_pt) or pd.isna(daily_price_change) or pd.isna(spot):
        return 0
    flow = -gex_per_pt * daily_price_change
    return flow


def fetch_options_volume_bql(ticker='SPX Index'):
    """
    Busca volume total de opções e P/C ratio via BQL.
    P/C ratio: usa PCUSEQTR Index (código BBG oficial do Put/Call ratio).
    Retorna dict com total_adc, put_vol, call_vol, pc_ratio.
    """
    bq = bql.Service()
    # P/C ratio — PCUSEQTR Index é o código oficial no terminal Bloomberg
    pcr = 0.0
    try:
        pc_req = bql.Request('PCUSEQTR Index', {'v': bq.data.px_last(fill='PREV')})
        pc_resp = bq.execute(pc_req)
        pcr = float(pc_resp[0].df()['v'].iloc[-1] or 0)
        print(f"[PC] PCUSEQTR Index = {pcr:.2f}")
    except Exception as _pce:
        print(f"⚠️ PCUSEQTR fetch: {_pce}")

    # Volume de calls/puts
    cv, pv = 0.0, 0.0
    try:
        req = bql.Request(ticker, {
            'call_vol': bq.data.call_opt_volume(),
            'put_vol':  bq.data.put_opt_volume(),
        })
        resp = bq.execute(req)
        row = resp[0].df().iloc[0] if len(resp[0].df()) > 0 else {}
        cv = float(row.get('call_vol', 0) or 0)
        pv = float(row.get('put_vol',  0) or 0)
    except Exception as _ve:
        print(f"⚠️ Options volume fetch: {_ve}")

    total = cv + pv
    return {
        'total_adc': total if total > 0 else OPTIONS_TOTAL_ADC,
        'call_vol': cv,
        'put_vol': pv,
        'pc_ratio': pcr,
        'source': 'BQL' if pcr > 0 else 'fallback',
    }


def estimate_mm_var_by_book(gex_per_pt, spot, risk_params, oi_total):
    """
    Estima VaR 95%/99% por market maker, proporcional ao volume share.
    Usa distribuição t-Student calibrada dos log returns.
    Para cada MM: assume que detém fração proporcional do GEX e OI.
    VaR_mm = share × VaR_total (linear porque gamma exposure escala com OI).
    Retorna lista de dicts com VaR por MM.
    """
    var_95 = risk_params.get('var_95', 0)
    var_99 = risk_params.get('var_99', 0)
    cvar_95 = risk_params.get('cvar_95', 0)

    # Total portfolio 1-day VaR in $ terms
    ds_95 = spot * abs(var_95)
    ds_99 = spot * abs(var_99)
    ds_cvar = spot * abs(cvar_95)

    # Total P&L at VaR level: delta-neutral MM → mostly gamma exposure
    # PnL ≈ 0.5 × GEX × ΔS²  (dealers are short the book → negative)
    pnl_var95 = abs(0.5 * gex_per_pt * ds_95 ** 2)
    pnl_var99 = abs(0.5 * gex_per_pt * ds_99 ** 2)
    pnl_cvar95 = abs(0.5 * gex_per_pt * ds_cvar ** 2)

    results = []
    for mm_name, share in MM_VOLUME_SHARES.items():
        mm_gex = gex_per_pt * share
        mm_oi = oi_total * share
        results.append({
            'name': mm_name,
            'share': share,
            'gex_per_pt': mm_gex,
            'oi_contracts': mm_oi,
            'var_95': pnl_var95 * share,
            'var_99': pnl_var99 * share,
            'cvar_95': pnl_cvar95 * share,
            'daily_theta': 0,  # Will be filled later if available
        })
    return results, {'pnl_var95': pnl_var95, 'pnl_var99': pnl_var99,
                     'pnl_cvar95': pnl_cvar95}


# ── Volatility Control Fund Flows ─────────────────────────────────
# Vol-targeting strategies adjust equity exposure inversely to realized vol.
# When RV rises, they sell; when it falls, they buy.
# Typical global AUM: ~$350B-$500B across pension and systematic funds.
VOL_CTRL_AUM = {5: 100e9, 10: 150e9, 15: 100e9}  # target_vol% → AUM estimate
VOL_CTRL_MAX_LEV = 2.0  # Maximum leverage cap
VOL_CTRL_MIN_EXP = 0.20  # Piso mínimo de exposição (fundos não vão a 0%)
VOL_CTRL_DAILY_ADJ = 0.25  # Ajuste máximo por dia (~25% do delta)


def _vc_exposure(target_dec, rv):
    """Calcula exposure com piso e teto."""
    if rv < 1e-6:
        return VOL_CTRL_MAX_LEV
    return max(min(target_dec / rv, VOL_CTRL_MAX_LEV), VOL_CTRL_MIN_EXP)


def compute_vol_control_flow(rv_current, rv_prev, target_vols=None):
    """
    Estima fluxo dos fundos de controle de volatilidade.
    rv_current e rv_prev devem ser vol anualizada (ex: 0.15 = 15%).
    Quando vol sobe → exposure cai → fundos vendem (flow negativo).
    Inclui piso de exposição (20%) e ajuste gradual (25%/dia).
    """
    if target_vols is None:
        target_vols = [5, 10, 15]
    if pd.isna(rv_current) or pd.isna(rv_prev) or rv_current < 1e-6 or rv_prev < 1e-6:
        return {'total': 0, 'detail': {}}

    detail = {}
    total = 0
    for tv in target_vols:
        tv_dec = tv / 100.0
        exp_new = _vc_exposure(tv_dec, rv_current)
        exp_old = _vc_exposure(tv_dec, rv_prev)
        aum = VOL_CTRL_AUM.get(tv, 100e9)
        # Fluxo total necessário vs fluxo diário (ajuste gradual)
        full_flow = aum * (exp_new - exp_old)
        daily_flow = full_flow * VOL_CTRL_DAILY_ADJ
        detail[f'{tv}%'] = {'exposure_new': exp_new, 'exposure_old': exp_old,
                            'flow': full_flow, 'daily_flow': daily_flow, 'aum': aum}
        total += full_flow
    daily_total = total * VOL_CTRL_DAILY_ADJ
    return {'total': total, 'daily_total': daily_total, 'detail': detail}


def compute_vol_control_scenarios(rv_current, target_vols=None):
    """
    Cenários de stress: quanto vendem se vol sobe.
    Retorna lista com total flow e daily flow (ajuste gradual).
    """
    if target_vols is None:
        target_vols = [5, 10, 15]
    if pd.isna(rv_current) or rv_current < 1e-6:
        return []
    shock_vols = [0.15, 0.20, 0.25, 0.30, 0.40]
    scenarios = []
    for sv in shock_vols:
        if sv <= rv_current * 1.05:
            continue
        total = 0
        for tv in target_vols:
            tv_dec = tv / 100.0
            exp_cur = _vc_exposure(tv_dec, rv_current)
            exp_shock = _vc_exposure(tv_dec, sv)
            aum = VOL_CTRL_AUM.get(tv, 100e9)
            total += aum * (exp_shock - exp_cur)
        scenarios.append({'rv_shock': sv, 'flow': total,
                          'daily_flow': total * VOL_CTRL_DAILY_ADJ})
    return scenarios


def compute_combined_flow_scenarios(rv_current, prices=None, gex_per_pt=0,
                                    spot=0, vanna_notional=0, vega_notional=0,
                                    charm_notional=0):
    """
    Cenários combinados: para cada nível de vol shock, estima fluxo de
    Vol Control + Risk Parity + CTA + Dealer + Vanna + Charm.
    vanna_notional = sum(vanna * OI * 100) * spot  ($ de delta por 1% vol)
    vega_notional = sum(vega * OI * 100)           ($ por 1% vol)
    charm_notional = sum(charm * OI * 100) * spot / 365  ($ de delta decay diário)
    """
    if pd.isna(rv_current) or rv_current < 1e-6:
        return []
    scenarios = [
        ('Leve (-3%, vol 18%)', -0.03, 0.18),
        ('Moderado (-5%, vol 22%)', -0.05, 0.22),
        ('Forte (-8%, vol 28%)', -0.08, 0.28),
        ('Crash (-12%, vol 35%)', -0.12, 0.35),
        ('Pânico (-20%, vol 50%)', -0.20, 0.50),
    ]
    results = []
    for name, spx_move, rv_shock in scenarios:
        if rv_shock <= rv_current * 1.05:
            continue
        # Vol Control (com piso)
        vc = 0
        for tv in [5, 10, 15]:
            tv_dec = tv / 100.0
            exp_cur = _vc_exposure(tv_dec, rv_current)
            exp_shock = _vc_exposure(tv_dec, rv_shock)
            vc += VOL_CTRL_AUM.get(tv, 100e9) * (exp_shock - exp_cur)

        # Risk Parity
        rp_result = compute_risk_parity_flow(rv_shock, rv_current)
        rp = rp_result['total']

        # CTA (trend reversal)
        cta = 0
        if prices is not None and len(prices) > 200:
            trend_now = compute_cta_trend_strength(prices)
            shocked = prices.copy()
            for i in range(1, 6):
                shocked.iloc[-i] = shocked.iloc[-i] * (1 + spx_move / 5)
            trend_shock = compute_cta_trend_strength(shocked)
            cta_rv = max(rv_shock, rv_current)
            pos_now = np.clip(trend_now * (0.10 / rv_current), -2, 2)
            pos_shock = np.clip(trend_shock * (0.10 / cta_rv), -2, 2)
            cta = CTA_AUM * CTA_EQUITY_ALLOC * (pos_shock - pos_now)

        # Dealer (short gamma amplifica sell-off)
        dealer = 0
        if gex_per_pt != 0 and spot > 0:
            daily_chg = spot * spx_move
            dealer = -gex_per_pt * daily_chg

        # Vanna flow: quando vol sobe + spot cai, dealers com vanna positivo
        # precisam vender delta. Flow ≈ -vanna_notional × Δvol_pts
        # vanna_notional já é em $ de delta por 1% vol
        vol_chg_pts = (rv_shock - rv_current) * 100  # em pontos de vol
        vanna = -vanna_notional * vol_chg_pts if vanna_notional != 0 else 0

        # Charm flow: decay diário do delta — dealers precisam rebalancear overnight.
        # charm_notional já é em $ de delta por dia (positivo = dealers precisam vender)
        charm = -charm_notional if charm_notional != 0 else 0

        total = vc + rp + cta + dealer + vanna + charm
        results.append({
            'name': name, 'spx_move': spx_move, 'rv_shock': rv_shock,
            'vol_ctrl': vc, 'risk_parity': rp, 'cta': cta,
            'dealer': dealer, 'vanna': vanna, 'charm': charm, 'total': total,
        })
    return results


# ── Risk Parity Model ──────────────────────────────────────────────
# Baseado em BofA Systematic Flows Monitor:
# - 3 asset classes: Equities (SPX), Bonds (10Y UST), Commodities (GSCI)
# - Alocação inversamente proporcional à volatilidade (↓vol → ↑alocação)
# - Rebalanceamento mensal usando 3M de dados para vol e correlação
# - AUM estimado: $200B–$750B (BofA usa $200B nos exhibits)
# - Risk targets: 10%, 12%, 15% vol; max leverage 1.5x/1.5x/3.0x
RISK_PARITY_AUM = 200e9  # $200B total (BofA Exhibit 3)
RISK_PARITY_TARGETS = {
    10: {'aum_share': 0.33, 'max_lev': 1.5},
    12: {'aum_share': 0.34, 'max_lev': 1.5},
    15: {'aum_share': 0.33, 'max_lev': 3.0},
}


def compute_risk_parity_flow(rv_equity, rv_equity_prev,
                             rv_bonds=None, rv_bonds_prev=None,
                             rv_commod=None, rv_commod_prev=None):
    """
    Estima fluxo de risk parity para equities.
    Alocação equity ∝ 1/vol_equity. Quando vol equity sobe,
    alocação cai → vendem equities.
    Se vol de bonds/commodities disponível, usa pesos relativos;
    caso contrário, usa só equity vol como proxy.
    """
    if pd.isna(rv_equity) or pd.isna(rv_equity_prev) or rv_equity < 1e-6:
        return {'total': 0, 'detail': {}, 'eq_alloc_new': 0, 'eq_alloc_old': 0}

    def _eq_weight(rv_eq, rv_bd, rv_cm):
        """Peso equity = (1/vol_eq) / sum(1/vol_i)."""
        inv_eq = 1.0 / max(rv_eq, 0.01)
        inv_bd = 1.0 / max(rv_bd, 0.01) if rv_bd and rv_bd > 1e-6 else inv_eq * 2.5
        inv_cm = 1.0 / max(rv_cm, 0.01) if rv_cm and rv_cm > 1e-6 else inv_eq * 0.8
        total_inv = inv_eq + inv_bd + inv_cm
        return inv_eq / total_inv if total_inv > 0 else 0.33

    eq_w_new = _eq_weight(rv_equity, rv_bonds, rv_commod)
    eq_w_old = _eq_weight(rv_equity_prev, rv_bonds_prev, rv_commod_prev)

    detail = {}
    total = 0
    _bv_new = rv_bonds if rv_bonds and rv_bonds > 1e-6 else 0.05
    _bv_old = rv_bonds_prev if rv_bonds_prev and rv_bonds_prev > 1e-6 else 0.05
    for tv, params in RISK_PARITY_TARGETS.items():
        aum = RISK_PARITY_AUM * params['aum_share']
        tv_dec = tv / 100.0
        # Portfolio vol ≈ weighted avg of component vols (simplificação)
        port_vol_new = rv_equity * eq_w_new + _bv_new * (1 - eq_w_new)
        port_vol_old = rv_equity_prev * eq_w_old + _bv_old * (1 - eq_w_old)
        lev_new = min(tv_dec / max(port_vol_new, 0.01), params['max_lev'])
        lev_old = min(tv_dec / max(port_vol_old, 0.01), params['max_lev'])
        eq_exp_new = lev_new * eq_w_new
        eq_exp_old = lev_old * eq_w_old
        flow = aum * (eq_exp_new - eq_exp_old)
        detail[f'{tv}%'] = {
            'aum': aum, 'leverage_new': lev_new, 'leverage_old': lev_old,
            'eq_alloc_new': eq_w_new, 'eq_alloc_old': eq_w_old,
            'eq_exposure_new': eq_exp_new, 'eq_exposure_old': eq_exp_old,
            'flow': flow}
        total += flow

    return {'total': total, 'detail': detail,
            'eq_alloc_new': eq_w_new, 'eq_alloc_old': eq_w_old}


# ── CTA Trend Following Model ─────────────────────────────────────
# Baseado em BofA Systematic Flows Monitor:
# - Trend strength via coleção de cruzamentos de médias móveis (near vs far)
# - Position sizing: trend_strength / volatility (trend/vol ratio)
# - AUM ~$340B (BarclayHedge 2024-Q4), ~25% alocação em equities
# - CTAs ajustam diariamente, são os mais rápidos a responder
CTA_AUM = 340e9
CTA_EQUITY_ALLOC = 0.25  # ~25% da carteira em equities
CTA_MA_PAIRS = [(5, 20), (5, 60), (10, 60), (20, 120), (20, 200)]
# Pesos por janela: sinais longos movem mais notional (estilo GS/BofA)
CTA_MA_WEIGHTS = {(5, 20): 0.10, (5, 60): 0.15, (10, 60): 0.20,
                  (20, 120): 0.25, (20, 200): 0.30}


def compute_cta_trend_strength(prices, ma_pairs=None, use_weights=None):
    """
    Calcula trend strength usando cruzamentos de médias móveis.
    Para cada par (curta, longa): spread contínuo em [-1, +1].
    use_weights=True → pesos por janela (sinais longos têm mais peso, estilo GS/BofA).
    use_weights=False → média simples (igual peso por janela).
    use_weights=None → lê o toggle cta_weight_w se disponível.
    Ref: BofA "Trends aren't going out of fashion" (2017).
    """
    if ma_pairs is None:
        ma_pairs = CTA_MA_PAIRS
    if use_weights is None:
        try:
            use_weights = cta_weight_w.value
        except NameError:
            use_weights = False
    if len(prices) < max(p[1] for p in ma_pairs):
        return 0
    scores, weights = [], []
    for short_w, long_w in ma_pairs:
        ma_short = prices.rolling(short_w).mean().iloc[-1]
        ma_long = prices.rolling(long_w).mean().iloc[-1]
        if pd.isna(ma_short) or pd.isna(ma_long) or ma_long == 0:
            continue
        spread = (ma_short - ma_long) / ma_long
        scores.append(np.clip(spread * 100, -1, 1))
        weights.append(CTA_MA_WEIGHTS.get((short_w, long_w), 1.0))
    if not scores:
        return 0
    if use_weights and len(weights) > 0:
        w = np.array(weights)
        return float(np.average(scores, weights=w / w.sum()))
    return float(np.mean(scores))


def compute_cta_flow(prices, rv_current, target_vol=0.10):
    """
    Estima fluxo de CTA trend followers para equities.
    Position = trend_strength × (target_vol / realized_vol), capped em [-2, +2].
    Flow = AUM × equity_alloc × Δposition.
    """
    if len(prices) < 201 or rv_current < 1e-6:
        return {'flow': 0, 'trend_today': 0, 'trend_prev': 0,
                'pos_today': 0, 'pos_prev': 0}

    trend_today = compute_cta_trend_strength(prices)
    trend_prev = compute_cta_trend_strength(prices.iloc[:-1])

    pos_today = np.clip(trend_today * (target_vol / rv_current), -2, 2)

    # RV do dia anterior (rolling 63d para capturar ~3M de vol)
    rets_prev = prices.iloc[:-1].pct_change().dropna()
    rv_prev = rets_prev.iloc[-63:].std() * np.sqrt(252) if len(rets_prev) >= 63 else rv_current
    pos_prev = np.clip(trend_prev * (target_vol / max(rv_prev, 1e-6)), -2, 2)

    flow = CTA_AUM * CTA_EQUITY_ALLOC * (pos_today - pos_prev)

    return {
        'flow': flow,
        'trend_today': trend_today,
        'trend_prev': trend_prev,
        'pos_today': pos_today,
        'pos_prev': pos_prev,
    }

def _trend_from_array(vals, use_weights=None):
    """Fast trend strength from raw numpy array (no pandas overhead)."""
    if use_weights is None:
        try:
            use_weights = cta_weight_w.value
        except NameError:
            use_weights = False
    n = len(vals)
    scores, weights = [], []
    for short_w, long_w in CTA_MA_PAIRS:
        if n < long_w:
            continue
        ms = vals[-short_w:].mean()
        ml = vals[-long_w:].mean()
        if ml == 0:
            continue
        spread = (ms - ml) / ml
        scores.append(np.clip(spread * 100, -1, 1))
        weights.append(CTA_MA_WEIGHTS.get((short_w, long_w), 1.0))
    if not scores:
        return 0.0
    if use_weights:
        w = np.array(weights)
        return float(np.average(scores, weights=w / w.sum()))
    return float(np.mean(scores))


def compute_cta_scenario_flows(prices, rv_current, spot, horizon_days=5,
                                annualized_vol=None):
    """
    Calcula fluxos de CTA em diferentes cenários (Flat / Up / Down) ao estilo GS.
    Versão rápida: usa numpy arrays, sem pd.concat nem pd.Timedelta.
    """
    if len(prices) < 201 or rv_current < 1e-6:
        return []
    if annualized_vol is None:
        annualized_vol = rv_current

    daily_vol = annualized_vol / np.sqrt(252)
    move_1sigma = daily_vol * np.sqrt(horizon_days)

    scenarios = [
        ('Flat', 0.0),
        ('Up 1\u03c3', move_1sigma),
        ('Up 2\u03c3', 2 * move_1sigma),
        ('Down 1\u03c3', -move_1sigma),
        ('Down 2\u03c3', -2 * move_1sigma),
        ('Down 2.5\u03c3', -2.5 * move_1sigma),
    ]

    vals = np.asarray(prices.values, dtype=float)
    trend_now = _trend_from_array(vals)
    pos_now = np.clip(trend_now * (0.10 / rv_current), -2, 2)
    current_notional = CTA_AUM * CTA_EQUITY_ALLOC * pos_now

    results = []
    for name, pct_move in scenarios:
        daily_step = pct_move / max(horizon_days, 1)
        sim = np.empty(len(vals) + horizon_days, dtype=float)
        sim[:len(vals)] = vals
        for d in range(horizon_days):
            sim[len(vals) + d] = sim[len(vals) + d - 1] * (1 + daily_step)

        trend_end = _trend_from_array(sim)
        rv_end = rv_current * (1 + max(0, -pct_move) * 3)
        pos_end = np.clip(trend_end * (0.10 / max(rv_end, 1e-6)), -2, 2)
        end_notional = CTA_AUM * CTA_EQUITY_ALLOC * pos_end
        flow = end_notional - current_notional

        results.append({
            'name': name,
            'spx_end': spot * (1 + pct_move),
            'pct_move': pct_move,
            'flow_total': flow,
            'trend_end': trend_end,
            'pos_end': pos_end,
            'pos_now': pos_now,
        })
    return results


def compute_cta_pivot_levels(prices, spot, rv_current):
    """
    Calcula níveis de preço onde sinais de trend MA flip (pivot levels).
    Usa numpy arrays para robustez (sem dependência de index type).
    """
    vals = np.asarray(prices.values, dtype=float)
    if len(vals) < 201:
        return []

    pivots = []
    labels = {
        (5, 20): 'Curto prazo',
        (5, 60): 'Curto-médio',
        (10, 60): 'Médio prazo',
        (20, 120): 'Médio-longo',
        (20, 200): 'Longo prazo',
    }

    for short_w, long_w in CTA_MA_PAIRS:
        ma_short_now = vals[-short_w:].mean()
        ma_long_now = vals[-long_w:].mean()

        if np.isnan(ma_short_now) or np.isnan(ma_long_now):
            continue

        above = ma_short_now > ma_long_now

        # Pivot: price X such that new MA_short = MA_long
        sum_recent = vals[-short_w + 1:].sum() if short_w > 1 else 0.0
        pivot_px = ma_long_now * short_w - sum_recent

        if pivot_px <= 0 or pivot_px > spot * 2:
            continue

        dist_pct = (pivot_px - spot) / spot
        signal_type = 'SELL trigger' if above else 'BUY trigger'
        label = labels.get((short_w, long_w), f'MA{short_w}/{long_w}')

        pivots.append({
            'label': label,
            'ma_pair': f'{short_w}/{long_w}',
            'level': pivot_px,
            'type': signal_type,
            'distance_pct': dist_pct,
            'above_now': above,
        })

    # Sort by proximity to spot
    pivots.sort(key=lambda x: abs(x['distance_pct']))
    return pivots


def compute_cta_historical_positions(prices, rv_series=None, lookback=252):
    """
    Calcula série histórica de trend strength, position sizing e notional.
    Versão vetorizada com numpy → rápido.
    """
    vals = np.asarray(prices.values, dtype=float)
    if len(vals) < 201:
        return pd.DataFrame()

    # Pre-compute all rolling MAs as numpy arrays
    ma_arrays = {}
    for short_w, long_w in CTA_MA_PAIRS:
        ms = pd.Series(vals).rolling(short_w).mean().values
        ml = pd.Series(vals).rolling(long_w).mean().values
        ma_arrays[(short_w, long_w)] = (ms, ml)

    # Rolling realized vol (63d)
    rets = np.diff(vals) / vals[:-1]
    rets = np.concatenate([[np.nan], rets])
    rv_arr = pd.Series(rets).rolling(63).std().values * np.sqrt(252)

    start = max(201, len(vals) - lookback)
    idx_range = range(start, len(vals))

    # Try to get dates from prices index
    try:
        dates = prices.index
    except Exception:
        dates = list(range(len(vals)))

    records = []
    for i in idx_range:
        scores = []
        for short_w, long_w in CTA_MA_PAIRS:
            ms_v, ml_v = ma_arrays[(short_w, long_w)]
            msv = ms_v[i]
            mlv = ml_v[i]
            if np.isnan(msv) or np.isnan(mlv) or mlv == 0:
                continue
            spread = (msv - mlv) / mlv
            scores.append(np.clip(spread * 100, -1, 1))
        trend = float(np.mean(scores)) if scores else 0.0

        rv = rv_arr[i] if not np.isnan(rv_arr[i]) else 0.15
        pos = np.clip(trend * (0.10 / max(rv, 1e-6)), -2, 2)
        notional = CTA_AUM * CTA_EQUITY_ALLOC * pos
        records.append({
            'date': dates[i],
            'trend': trend,
            'position': pos,
            'notional': notional,
        })
    return pd.DataFrame(records)


def build_cta_gs_chart(fp_cta_hist, fp_cta_scenarios_1w, fp_cta_scenarios_1m,
                       spot):
    """
    Constroi chart: histórico de posição CTA com fan de
    cenários projetados para 1W e 1M à frente.
    Retorna go.Figure (converter para FigureWidget externamente).
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=[
            'CTA Estimates — S&P 500 (Notional $B)',
            'CTA Scenario Flows ($B) — 1 Week vs 1 Month',
        ])

    # ── Top panel: Historical notional + scenario fan ──
    if not fp_cta_hist.empty and len(fp_cta_hist) > 5:
        hist_dates = fp_cta_hist['date']
        hist_notional = fp_cta_hist['notional'] / 1e9

        # Historical line
        fig.add_trace(go.Scatter(
            x=hist_dates, y=hist_notional,
            name='CTA Notional (Hist)',
            mode='lines',
            line=dict(color='#4A90D9', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(74,144,217,0.10)'),
            row=1, col=1)

        # Scenario fan from last date
        last_date = pd.Timestamp(hist_dates.iloc[-1])
        last_notional = hist_notional.iloc[-1]

        # Build forward points for 1W and 1M scenarios
        scenario_colors = {
            'Flat': '#AAAAAA',
            'Up 1\u03c3': '#00C853',
            'Up 2\u03c3': '#00E676',
            'Down 1\u03c3': '#FF5252',
            'Down 2\u03c3': '#FF1744',
            'Down 2.5\u03c3': '#D50000',
        }

        for s1w, s1m in zip(fp_cta_scenarios_1w, fp_cta_scenarios_1m):
            name = s1w['name']
            end_notional_1w = CTA_AUM * CTA_EQUITY_ALLOC * s1w['pos_end'] / 1e9
            end_notional_1m = CTA_AUM * CTA_EQUITY_ALLOC * s1m['pos_end'] / 1e9
            color = scenario_colors.get(name, '#888888')

            try:
                d1w = last_date + pd.Timedelta(days=7)
                d1m = last_date + pd.Timedelta(days=30)
            except Exception:
                d1w = last_date + pd.Timedelta(days=5)
                d1m = last_date + pd.Timedelta(days=21)

            fig.add_trace(go.Scatter(
                x=[last_date, d1w, d1m],
                y=[last_notional, end_notional_1w, end_notional_1m],
                name=name,
                mode='lines+markers',
                line=dict(color=color, width=2, dash='dot'),
                marker=dict(size=6, color=color),
                legendgroup=name),
                row=1, col=1)

        fig.add_hline(y=0, line_dash='dash', line_color='rgba(150,150,150,0.5)',
                      row=1, col=1)

    # ── Bottom panel: Scenario flow bar chart (grouped 1W vs 1M) ──
    if fp_cta_scenarios_1w and fp_cta_scenarios_1m:
        names = [s['name'] for s in fp_cta_scenarios_1w]
        flows_1w = [s['flow_total'] / 1e9 for s in fp_cta_scenarios_1w]
        flows_1m = [s['flow_total'] / 1e9 for s in fp_cta_scenarios_1m]

        bar_colors_1w = ['#00C853' if f > 0 else '#FF5252' for f in flows_1w]
        bar_colors_1m = ['#00E676' if f > 0 else '#FF1744' for f in flows_1m]

        fig.add_trace(go.Bar(
            x=names, y=flows_1w, name='1 Week Flow',
            marker_color=bar_colors_1w,
            text=[f'${f:+.1f}B' for f in flows_1w],
            textposition='outside',
            textfont=dict(size=10)),
            row=2, col=1)

        fig.add_trace(go.Bar(
            x=names, y=flows_1m, name='1 Month Flow',
            marker_color=bar_colors_1m,
            text=[f'${f:+.1f}B' for f in flows_1m],
            textposition='outside',
            textfont=dict(size=10),
            opacity=0.7),
            row=2, col=1)

        fig.add_hline(y=0, line_dash='dash', line_color='rgba(150,150,150,0.5)',
                      row=2, col=1)

    fig.update_layout(
        paper_bgcolor=_C['card'], plot_bgcolor=_C['card'],
        font=dict(color=_C['text'], size=11),
        legend=dict(orientation='h', y=-0.08, x=0.5, xanchor='center',
                    font=dict(color=_C['text_muted'], size=10)),
        height=620, margin=dict(l=55, r=30, t=40, b=40),
        barmode='group')
    for ax in ['yaxis', 'yaxis2', 'xaxis', 'xaxis2']:
        fig.update_layout(**{ax: dict(
            gridcolor=_C['border'], zerolinecolor=_C['border'],
            tickfont=dict(color=_C['text_muted']))})
    # Make subplot titles use theme color
    for ann in fig.layout.annotations:
        ann.font.color = _C['text']
        ann.font.size = 13

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5F — VISUALIZAÇÕES DO FLOW PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5G — DISPERSION TRADE + TAIL RISK
# ═══════════════════════════════════════════════════════════════════════════════

def _bql_fetch_member_data(index_ticker='SPX Index', lookback_days=252):
    """
    Busca preços e IV dos membros de um índice via BQL.
    Retorna (prices_df, iv_df, weights_dict).
    """
    bq = bql.Service()
    univ = bq.univ.members(index_ticker)
    dt_range = bq.func.range('-{}d'.format(lookback_days), '0d')
    req_px = bql.Request(univ, {
        'px': bq.data.px_last(fill='PREV', dates=dt_range),
    })
    req_iv = bql.Request(univ, {
        'iv': bq.data.implied_volatility(fill='PREV', dates=dt_range),
    })

    # Pesos: tentar idx_wt_val, senão cur_mkt_cap como proxy
    try:
        req_wt = bql.Request(univ, {
            'wt': bq.data.idx_wt_val(fill='PREV'),
        })
        resp_wt = bq.execute(req_wt)
        wt_df = resp_wt[0].df()
    except Exception:
        try:
            req_wt = bql.Request(univ, {
                'wt': bq.data.cur_mkt_cap(),
            })
            resp_wt = bq.execute(req_wt)
            wt_df = resp_wt[0].df()
        except Exception:
            wt_df = pd.DataFrame()

    resp_px = bq.execute(req_px)
    resp_iv = bq.execute(req_iv)

    df_px = resp_px[0].df().reset_index()
    # Encontrar coluna de data para usar como index do pivot
    _dt_col_px = next((c for c in df_px.columns if str(c).upper() == 'DATE'), None)
    if _dt_col_px:
        df_px[_dt_col_px] = pd.to_datetime(df_px[_dt_col_px])
        prices_df = df_px.pivot(index=_dt_col_px, columns='ID', values='px')
    else:
        prices_df = df_px.pivot(columns='ID', values='px')

    df_iv = resp_iv[0].df().reset_index()
    _dt_col_iv = next((c for c in df_iv.columns if str(c).upper() == 'DATE'), None)
    if _dt_col_iv:
        df_iv[_dt_col_iv] = pd.to_datetime(df_iv[_dt_col_iv])
        iv_df = df_iv.pivot(index=_dt_col_iv, columns='ID', values='iv')
    else:
        iv_df = df_iv.pivot(columns='ID', values='iv')

    weights = {}
    if not wt_df.empty:
        if 'ID' not in wt_df.columns:
            wt_df = wt_df.reset_index()
        total_wt = 0
        for _, row in wt_df.iterrows():
            ticker = row.get('ID', '')
            w = row.get('wt', 0.0)
            if ticker and w and not np.isnan(w):
                weights[ticker] = float(w)
                total_wt += float(w)
        if total_wt > 0:
            for k in weights:
                weights[k] = weights[k] / total_wt
    return prices_df, iv_df, weights


def _bql_fetch_index_iv(index_ticker='SPX Index', lookback_days=252):
    """Busca IV histórica do índice."""
    bq = bql.Service()
    dt_range = bq.func.range('-{}d'.format(lookback_days), '0d')
    req = bql.Request(index_ticker, {
        'iv': bq.data.implied_volatility(fill='PREV', dates=dt_range),
    })
    resp = bq.execute(req)
    return _bql_ts(resp[0], 'iv')


def _bql_fetch_impl_corr(lookback_days=252):
    """Busca o CBOE S&P 500 3M Implied Correlation Index (.SPXSK3 G Index)."""
    bq = bql.Service()
    dt_range = bq.func.range('-{}d'.format(lookback_days), '0d')
    try:
        req = bql.Request(DISP_SPXSK3, {
            'px': bq.data.px_last(fill='PREV', dates=dt_range),
        })
        resp = bq.execute(req)
        return _bql_ts(resp[0], 'px')
    except Exception:
        return pd.Series(dtype=float)


def compute_realized_correlation(prices_df, windows=None):
    """
    Calcula correlação realizada média (pairwise) para cada janela.
    Retorna DataFrame: index=datas, cols=window labels.
    """
    if windows is None:
        windows = DISP_CORR_WINDOWS
    log_rets = np.log(prices_df / prices_df.shift(1)).dropna()
    result = {}
    for label, w in windows.items():
        corrs = []
        dates_out = []
        for end in range(w, len(log_rets)):
            window_rets = log_rets.iloc[end - w:end]
            corr_mat = window_rets.corr()
            n = len(corr_mat)
            if n < 2:
                continue
            upper = corr_mat.values[np.triu_indices(n, k=1)]
            avg_corr = float(np.nanmean(upper))
            corrs.append(avg_corr)
            dates_out.append(log_rets.index[end])
        result[label] = pd.Series(corrs, index=dates_out)
    return pd.DataFrame(result)


def compute_implied_correlation(sigma_idx, sigmas_i, weights):
    """
    Calcula correlação implícita (CBOE methodology).
    ρ_impl = (σ²_idx - Σ wi² σi²) / (Σ_{i≠j} wi wj σi σj)
    Inputs: escalares (para um dia).
    """
    tickers = list(weights.keys())
    n = len(tickers)
    if n < 2:
        return np.nan
    w = np.array([weights[t] for t in tickers])
    s = np.array([sigmas_i.get(t, 0.0) for t in tickers])
    var_idx = sigma_idx ** 2
    sum_wi2_si2 = np.sum(w ** 2 * s ** 2)
    cross = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            cross += w[i] * w[j] * s[i] * s[j]
    denom = 2.0 * cross
    if abs(denom) < 1e-12:
        return np.nan
    rho = (var_idx - sum_wi2_si2) / denom
    return float(np.clip(rho, -1.0, 1.0))


def compute_implied_corr_series(index_iv, member_iv_df, weights):
    """
    Calcula série temporal de correlação implícita.
    index_iv: Series (dates → IV), member_iv_df: DataFrame (dates × tickers).
    """
    common_dates = index_iv.index.intersection(member_iv_df.index)
    impl_corrs = []
    dates_out = []
    for dt in common_dates:
        sig_idx = index_iv.loc[dt]
        if np.isnan(sig_idx) or sig_idx <= 0:
            continue
        sigmas = {}
        for t in weights:
            if t in member_iv_df.columns:
                val = member_iv_df.loc[dt, t]
                if not np.isnan(val) and val > 0:
                    sigmas[t] = val / 100.0
        sub_w = {t: weights[t] for t in sigmas if t in weights}
        total_w = sum(sub_w.values())
        if total_w < 0.5:
            continue
        sub_w = {t: v / total_w for t, v in sub_w.items()}
        rho = compute_implied_correlation(sig_idx / 100.0, sigmas, sub_w)
        if not np.isnan(rho):
            impl_corrs.append(rho)
            dates_out.append(dt)
    return pd.Series(impl_corrs, index=dates_out, name='impl_corr')


def compute_dispersion_signal(impl_corr_series, real_corr_df, window='3M'):
    """
    Gera sinal de dispersion trade.
    Spread = Impl Corr - Realized Corr → positivo = vender vol do índice.
    Retorna DataFrame com colunas: impl_corr, real_corr, spread, z_score, signal.
    """
    real_col = real_corr_df[window] if window in real_corr_df.columns else real_corr_df.iloc[:, 0]
    common = impl_corr_series.index.intersection(real_col.index)
    if len(common) < 20:
        return pd.DataFrame()
    impl = impl_corr_series.reindex(common)
    real = real_col.reindex(common)
    spread = impl - real
    roll_mean = spread.rolling(63, min_periods=20).mean()
    roll_std = spread.rolling(63, min_periods=20).std()
    z = (spread - roll_mean) / roll_std.replace(0, np.nan)
    signal = pd.Series('NEUTRAL', index=common)
    signal[z > 1.0] = 'SHORT INDEX VOL'
    signal[z < -1.0] = 'LONG INDEX VOL'
    return pd.DataFrame({
        'impl_corr': impl, 'real_corr': real,
        'spread': spread, 'z_score': z, 'signal': signal,
    })


def optimize_tracking_basket(index_prices, member_prices, n_stocks=10):
    """
    Seleciona N ações que melhor replicam o índice (minimize tracking error).
    Retorna (selected_tickers, weights_dict, tracking_error).
    """
    idx_ret = np.log(index_prices / index_prices.shift(1)).dropna()
    mem_ret = np.log(member_prices / member_prices.shift(1)).dropna()
    common = idx_ret.index.intersection(mem_ret.index)
    idx_ret = idx_ret.reindex(common).values
    mem_ret = mem_ret.reindex(common)
    tickers = list(mem_ret.columns)
    mem_arr = mem_ret.values

    corrs = np.array([np.corrcoef(idx_ret, mem_arr[:, i])[0, 1]
                       for i in range(mem_arr.shape[1])])
    valid = ~np.isnan(corrs)
    top_k = min(max(n_stocks * 3, 30), int(valid.sum()))
    top_idx = np.argsort(-np.abs(np.where(valid, corrs, 0)))[:top_k]
    sub_tickers = [tickers[i] for i in top_idx]
    sub_arr = mem_arr[:, top_idx]

    def objective(w):
        port_ret = sub_arr @ w
        te = np.std(port_ret - idx_ret)
        return te

    n = len(sub_tickers)
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    res = sp_minimize(objective, w0, method='SLSQP',
                      bounds=bounds, constraints=cons,
                      options={'maxiter': 500, 'ftol': 1e-10})
    opt_w = res.x
    ranked = np.argsort(-opt_w)[:n_stocks]
    sel_tickers = [sub_tickers[i] for i in ranked]
    sel_weights = {sub_tickers[i]: float(opt_w[i]) for i in ranked}
    total = sum(sel_weights.values())
    if total > 0:
        sel_weights = {t: v / total for t, v in sel_weights.items()}
    te_final = float(objective(opt_w))
    return sel_tickers, sel_weights, te_final


def compute_mag7_dispersion(prices_df):
    """
    Calcula dispersão entre pares Mag7.
    Retorna DataFrame com IV spread e correlation para cada par.
    """
    mag7_in_df = [t for t in MAG7 if t in prices_df.columns]
    if len(mag7_in_df) < 2:
        return pd.DataFrame()
    log_rets = np.log(prices_df[mag7_in_df] / prices_df[mag7_in_df].shift(1)).dropna()
    corr_mat = log_rets.tail(63).corr()
    pairs = []
    for i in range(len(mag7_in_df)):
        for j in range(i + 1, len(mag7_in_df)):
            t1 = mag7_in_df[i]
            t2 = mag7_in_df[j]
            short_1 = t1.split(' ')[0]
            short_2 = t2.split(' ')[0]
            corr_val = corr_mat.loc[t1, t2]
            vol1 = float(log_rets[t1].std() * np.sqrt(252) * 100)
            vol2 = float(log_rets[t2].std() * np.sqrt(252) * 100)
            pairs.append({
                'Par': '{}/{}'.format(short_1, short_2),
                'Corr 3M': round(corr_val, 3),
                'RVol1 (%)': round(vol1, 1),
                'RVol2 (%)': round(vol2, 1),
                'Vol Spread': round(abs(vol1 - vol2), 1),
            })
    df = pd.DataFrame(pairs)
    df = df.sort_values('Vol Spread', ascending=False).reset_index(drop=True)
    return df


def find_best_2x2_dispersion(prices_df, iv_df=None):
    """
    Encontra melhor combo NxN entre Mag7 para dispersion trade.
    Testa 2x2, 3x3, ... até floor(len/2) x floor(len/2).
    Para cada tamanho N: top-N high vol vs bottom-N low vol.
    Retorna lista com melhor combo de cada tamanho + o ótimo geral.
    """
    from itertools import combinations

    mag7_in = [t for t in MAG7 if t in prices_df.columns]
    if len(mag7_in) < 4:
        return []
    log_rets = np.log(prices_df[mag7_in] / prices_df[mag7_in].shift(1)).dropna()
    vols = {}
    for t in mag7_in:
        vols[t] = float(log_rets[t].tail(63).std() * np.sqrt(252))

    if iv_df is not None:
        for t in mag7_in:
            if t in iv_df.columns:
                last_iv = iv_df[t].dropna()
                if len(last_iv) > 0:
                    vols[t] = float(last_iv.iloc[-1]) / 100.0

    sorted_by_vol = sorted(mag7_in, key=lambda t: vols.get(t, 0), reverse=True)
    max_n = len(mag7_in) // 2  # max group size

    combos = []
    best_spread = -np.inf
    best_combo = None

    for n in range(2, max_n + 1):
        # Try all combinations of n from high-vol pool vs n from low-vol pool
        # For efficiency: top 2*n candidates, test all combos of n from each half
        pool_high = sorted_by_vol[:min(2 * n, len(sorted_by_vol))]
        pool_low = sorted_by_vol[max(0, len(sorted_by_vol) - 2 * n):]

        local_best_spread = -np.inf
        local_best_high = sorted_by_vol[:n]
        local_best_low = sorted_by_vol[-n:]

        for high_combo in combinations(pool_high, n):
            remaining = [t for t in mag7_in if t not in high_combo]
            if len(remaining) < n:
                continue
            for low_combo in combinations(remaining, n):
                avg_high = np.mean([vols[t] for t in high_combo])
                avg_low = np.mean([vols[t] for t in low_combo])
                spread = avg_high - avg_low
                if spread > local_best_spread:
                    local_best_spread = spread
                    local_best_high = list(high_combo)
                    local_best_low = list(low_combo)

        combo_entry = {
            'Combo': f'{n}x{n}',
            'Long Vol (Buy)': ', '.join(t.split(' ')[0] for t in local_best_high),
            'Short Vol (Sell)': ', '.join(t.split(' ')[0] for t in local_best_low),
            'Avg IV High': round(np.mean([vols[t] for t in local_best_high]) * 100, 1),
            'Avg IV Low': round(np.mean([vols[t] for t in local_best_low]) * 100, 1),
            'Spread (pp)': round(local_best_spread * 100, 1),
        }
        combos.append(combo_entry)

        if local_best_spread > best_spread:
            best_spread = local_best_spread
            best_combo = combo_entry

    # Mark the best overall combo
    if best_combo:
        for c in combos:
            c['Ótimo'] = '⭐' if c['Combo'] == best_combo['Combo'] else ''

    return combos


def find_best_pair_combos(prices_df, iv_df=None, straddle_data=None, max_pairs=3):
    """
    Encontra melhores combinações de até max_pairs pares para dispersion trade.
    Cada par = (long vol ticker, short vol ticker).
    Score = soma dos vol spreads * (1 - abs(corr)) para diversificação.
    Retorna list of dicts com detalhes de cada combo (1-par, 2-pares, 3-pares).
    """
    from itertools import combinations

    mag7_in = [t for t in MAG7 if t in prices_df.columns]
    if len(mag7_in) < 2:
        return []
    log_rets = np.log(prices_df[mag7_in] / prices_df[mag7_in].shift(1)).dropna()
    corr_mat = log_rets.tail(63).corr()
    vols = {}
    for t in mag7_in:
        vols[t] = float(log_rets[t].tail(63).std() * np.sqrt(252))
    if iv_df is not None:
        for t in mag7_in:
            if t in iv_df.columns:
                last_iv = iv_df[t].dropna()
                if len(last_iv) > 0:
                    vols[t] = float(last_iv.iloc[-1]) / 100.0

    # Build all possible pairs
    all_pairs = []
    for t1, t2 in combinations(mag7_in, 2):
        v1, v2 = vols[t1], vols[t2]
        if v1 < v2:
            t1, t2 = t2, t1
            v1, v2 = v2, v1
        spread = v1 - v2
        corr = corr_mat.loc[t1, t2] if t1 in corr_mat.index and t2 in corr_mat.index else 0
        # Straddle cost info
        s1_pct = straddle_data.get(t1, {}).get('straddle_pct', np.nan) if straddle_data else np.nan
        s2_pct = straddle_data.get(t2, {}).get('straddle_pct', np.nan) if straddle_data else np.nan
        all_pairs.append({
            'long': t1, 'short': t2,
            'spread': spread, 'corr': corr,
            'score': spread * (1 - abs(corr)),
            'straddle_long': s1_pct, 'straddle_short': s2_pct,
        })
    all_pairs.sort(key=lambda x: x['score'], reverse=True)

    results = []
    for n_pairs in range(1, min(max_pairs + 1, len(all_pairs) + 1)):
        # For n_pairs: find best non-overlapping combination
        best_score = -np.inf
        best_selection = None
        # Try top candidates (limit search space for speed)
        top_pool = all_pairs[:min(15, len(all_pairs))]
        for combo in combinations(range(len(top_pool)), n_pairs):
            selected = [top_pool[i] for i in combo]
            # Ensure no ticker overlap
            tickers_used = set()
            overlap = False
            for p in selected:
                if p['long'] in tickers_used or p['short'] in tickers_used:
                    overlap = True
                    break
                tickers_used.add(p['long'])
                tickers_used.add(p['short'])
            if overlap:
                continue
            total_score = sum(p['score'] for p in selected)
            if total_score > best_score:
                best_score = total_score
                best_selection = selected

        if best_selection is None:
            continue

        pair_strs = []
        for p in best_selection:
            l_short = p['long'].split(' ')[0]
            s_short = p['short'].split(' ')[0]
            pair_strs.append(f"{l_short}/{s_short}")

        avg_spread = np.mean([p['spread'] for p in best_selection])
        avg_corr = np.mean([p['corr'] for p in best_selection])
        straddle_costs = []
        for p in best_selection:
            if not np.isnan(p['straddle_long']) and not np.isnan(p['straddle_short']):
                straddle_costs.append(p['straddle_long'] + p['straddle_short'])

        entry = {
            'N Pares': n_pairs,
            'Pares': ' + '.join(pair_strs),
            'Spread Médio (pp)': round(avg_spread * 100, 1),
            'Corr Média': round(avg_corr, 3),
            'Score': round(best_score * 100, 2),
        }
        if straddle_costs:
            entry['Custo Straddle (%)'] = round(np.mean(straddle_costs), 2)
        results.append(entry)

    return results


# ── Tail Risk (EVT + Conditional Expectations) ──

def compute_tail_risk(log_returns, threshold_pct=5):
    """
    Calcula métricas de risco caudal usando EVT simplificada.
    - Exceedances abaixo do percentil threshold_pct
    - Hill estimator para tail index
    - Conditional Tail Expectation (CTE)
    - Expected Shortfall beyond threshold
    Retorna dict com métricas.
    """
    rets = np.asarray(log_returns, dtype=float)
    rets = rets[~np.isnan(rets)]
    n = len(rets)
    if n < 50:
        return {}

    threshold = np.percentile(rets, threshold_pct)
    exceedances = rets[rets <= threshold]
    n_exceed = len(exceedances)

    cte = float(np.mean(exceedances)) if n_exceed > 0 else np.nan

    abs_exceed = np.abs(exceedances - threshold)
    abs_exceed = abs_exceed[abs_exceed > 0]
    if len(abs_exceed) >= 5:
        log_excess = np.log(abs_exceed / abs_exceed.min())
        hill_alpha = float(1.0 / np.mean(log_excess)) if np.mean(log_excess) > 0 else np.nan
    else:
        hill_alpha = np.nan

    max_loss = float(np.min(rets))
    p1 = np.percentile(rets, 1)
    p5 = np.percentile(rets, 5)
    below_1 = rets[rets <= p1]
    cte_1 = float(np.mean(below_1)) if len(below_1) > 0 else np.nan

    kurtosis_val = float(
        np.mean((rets - np.mean(rets)) ** 4) / (np.std(rets) ** 4)
    ) if np.std(rets) > 0 else np.nan
    skew_val = float(
        np.mean((rets - np.mean(rets)) ** 3) / (np.std(rets) ** 3)
    ) if np.std(rets) > 0 else np.nan

    return {
        'threshold_pct': threshold_pct,
        'threshold_ret': round(float(threshold) * 100, 2),
        'n_exceedances': n_exceed,
        'pct_exceedances': round(n_exceed / n * 100, 1),
        'CTE (5%)': round(float(cte) * 100, 2),
        'CTE (1%)': round(float(cte_1) * 100, 2),
        'Max Loss': round(float(max_loss) * 100, 2),
        'Hill Alpha': round(hill_alpha, 2) if not np.isnan(hill_alpha) else 'N/A',
        'Kurtosis': round(kurtosis_val, 2),
        'Skewness': round(skew_val, 3),
        'VaR 1%': round(float(p1) * 100, 2),
        'VaR 5%': round(float(p5) * 100, 2),
    }


# ── Dispersion Dashboard Visualizations ──

def build_dispersion_chart(disp_signal_df, impl_corr_cboe=None):
    """
    Gráfico principal de dispersion: Impl Corr vs Realized Corr + Spread.
    """
    if disp_signal_df.empty:
        fig = go.Figure()
        fig.add_annotation(text='Sem dados de dispersão', x=0.5, y=0.5,
                           xref='paper', yref='paper', showarrow=False,
                           font=dict(color='white', size=16))
        fig.update_layout(template='plotly_dark',
                          paper_bgcolor='#0d1117', plot_bgcolor='#161b22')
        return go.FigureWidget(fig)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45],
                        subplot_titles=['Correlação Implícita vs Realizada',
                                        'Spread (Impl - Real) + Z-Score'],
                        vertical_spacing=0.14)

    fig.add_trace(go.Scatter(
        x=disp_signal_df.index, y=disp_signal_df['impl_corr'],
        name='Impl Corr (calc)', line=dict(color='#f0883e', width=2.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=disp_signal_df.index, y=disp_signal_df['real_corr'],
        name='Real Corr 3M', line=dict(color='#58a6ff', width=2.5),
    ), row=1, col=1)

    if impl_corr_cboe is not None and len(impl_corr_cboe) > 0:
        cboe_scaled = impl_corr_cboe / 100.0
        fig.add_trace(go.Scatter(
            x=cboe_scaled.index, y=cboe_scaled,
            name='CBOE Impl Corr', line=dict(color='#d29922',
                                              width=1.5, dash='dot'),
        ), row=1, col=1)

    # --- Row 2: Spread bars + Z-score on secondary y ---
    colors = ['#238636' if v > 0 else '#da3633'
              for v in disp_signal_df['spread'].values]
    fig.add_trace(go.Bar(
        x=disp_signal_df.index, y=disp_signal_df['spread'],
        name='Spread', marker_color=colors, opacity=0.55,
    ), row=2, col=1)

    # Z-Score on secondary y-axis for row 2
    fig.add_trace(go.Scatter(
        x=disp_signal_df.index, y=disp_signal_df['z_score'],
        name='Z-Score', line=dict(color='#bc8cff', width=2),
    ), row=2, col=1)

    fig.add_hline(y=1.0, line_dash='dash', line_color='#238636',
                  opacity=0.5, row=2, col=1)
    fig.add_hline(y=-1.0, line_dash='dash', line_color='#da3633',
                  opacity=0.5, row=2, col=1)

    # --- Layout ---
    fig.update_layout(
        template='plotly_dark', height=660,
        paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9', size=12),
        legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center',
                    font=dict(size=12), bgcolor='rgba(13,17,23,0.7)'),
        margin=dict(l=60, r=60, t=60, b=30),
    )
    # Axis labels
    fig.update_yaxes(title_text='Correlação', row=1, col=1, title_font_size=12)
    fig.update_yaxes(title_text='Spread / Z-Score', row=2, col=1, title_font_size=12)
    fig.update_xaxes(tickfont=dict(size=10), row=1, col=1)
    fig.update_xaxes(tickfont=dict(size=10), row=2, col=1)

    return go.FigureWidget(fig)


def build_corr_regime_chart(real_corr_df):
    """Gráfico de correlação realizada em múltiplos timeframes."""
    if real_corr_df.empty:
        fig = go.Figure()
        fig.update_layout(template='plotly_dark',
                          paper_bgcolor='#0d1117', plot_bgcolor='#161b22')
        return go.FigureWidget(fig)

    palette = {'1M': '#58a6ff', '3M': '#f0883e', '6M': '#bc8cff'}
    fig = go.Figure()
    for col in real_corr_df.columns:
        fig.add_trace(go.Scatter(
            x=real_corr_df.index, y=real_corr_df[col],
            name='Corr {}'.format(col),
            line=dict(color=palette.get(col, '#8b949e'), width=1.8),
        ))
    fig.add_hline(y=0.5, line_dash='dash', line_color='#8b949e', opacity=0.4)
    fig.update_layout(
        template='plotly_dark', height=300,
        title='Correlação Realizada Média (Pairwise)',
        paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9', size=11),
        yaxis_title='Avg Correlation',
        legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'),
        margin=dict(l=60, r=40, t=50, b=30),
    )
    return go.FigureWidget(fig)


def build_tail_risk_chart(log_returns, tail_metrics):
    """Histograma de retornos com cauda marcada + métricas."""
    rets = np.asarray(log_returns, dtype=float)
    rets = rets[~np.isnan(rets)]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rets * 100, nbinsx=80, name='Retornos (%)',
        marker_color='#58a6ff', opacity=0.7,
    ))
    var5 = tail_metrics.get('VaR 5%', None)
    var1 = tail_metrics.get('VaR 1%', None)
    if var5 is not None:
        fig.add_vline(x=var5, line_dash='dash', line_color='#f0883e',
                      annotation_text='VaR 5%', annotation_font_color='#f0883e')
    if var1 is not None:
        fig.add_vline(x=var1, line_dash='dash', line_color='#da3633',
                      annotation_text='VaR 1%', annotation_font_color='#da3633')
    fig.update_layout(
        template='plotly_dark', height=300,
        title='Distribuição de Retornos + Tail Risk',
        paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9', size=11),
        xaxis_title='Retorno (%)', yaxis_title='Frequência',
        margin=dict(l=60, r=40, t=50, b=30),
    )
    return go.FigureWidget(fig)


def _disp_table_widget(df, title='', height='250px'):
    """Helper: cria widget de tabela para dispersion data."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return wd.HTML('<p style="color:#8b949e;">{}: Sem dados</p>'.format(title))
    html = '<div style="color:#c9d1d9; font-size:12px;">'
    html += '<b>{}</b>'.format(title)
    html += '<table style="border-collapse:collapse; width:100%; margin-top:5px;">'
    html += '<tr>'
    for c in df.columns:
        html += '<th style="border-bottom:1px solid #30363d; padding:4px 8px; '
        html += 'text-align:left; color:#58a6ff;">{}</th>'.format(c)
    html += '</tr>'
    for _, row in df.iterrows():
        html += '<tr>'
        for c in df.columns:
            val = row[c]
            html += '<td style="padding:3px 8px; border-bottom:1px solid #21262d; '
            html += 'color:#c9d1d9;">{}</td>'.format(val)
        html += '</tr>'
    html += '</table></div>'
    return wd.HTML(html)


# ── Gamma History Database ──────────────────────────────────────────────────
def load_gamma_history(path=None):
    """
    Carrega histórico de gamma do CSV e calcula RV 21d a partir do Ref Px.
    Retorna DataFrame com colunas: date, px, gamma, delta, call_wall, put_wall,
    vol_trigger, rv21d.
    """
    fpath = path or GAMMA_HISTORY_PATH
    print(f"[GAMMA DB] Loading from: {fpath} (exists={os.path.exists(fpath)})")
    if not os.path.exists(fpath):
        return pd.DataFrame()
    df = pd.read_csv(fpath)
    print(f"[GAMMA DB] CSV columns: {list(df.columns)}, shape={df.shape}")
    # Normalize column names (strip whitespace, handle variations)
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        cl = c.lower().replace(' ', '_')
        if cl in ('trade_date', 'date'):
            col_map[c] = 'date'
        elif cl in ('ref_px', 'price', 'px', 'spot'):
            col_map[c] = 'px'
        elif cl in ('net_gamma', 'gamma', 'gamma_index'):
            col_map[c] = 'gamma'
        elif cl in ('net_delta', 'delta'):
            col_map[c] = 'delta'
        elif cl in ('call_wall',):
            col_map[c] = 'call_wall'
        elif cl in ('put_wall',):
            col_map[c] = 'put_wall'
        elif cl in ('vol_trigger',):
            col_map[c] = 'vol_trigger'
        elif cl in ('data_release',):
            col_map[c] = 'data_release'
    df = df.rename(columns=col_map)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
    # Remove linhas corrompidas: px ou gamma não numérico (linhas mescladas no CSV)
    for _col in ['px', 'gamma']:
        if _col in df.columns:
            df[_col] = pd.to_numeric(df[_col], errors='coerce')
    df = df.dropna(subset=[c for c in ['px', 'gamma'] if c in df.columns])
    df = df.sort_values('date').reset_index(drop=True)
    # RV 21d (annualized) from Ref Px
    if 'px' in df.columns:
        df['px'] = pd.to_numeric(df['px'], errors='coerce')
        log_ret = np.log(df['px'] / df['px'].shift(1))
        df['rv21d'] = log_ret.rolling(21).std() * np.sqrt(252)
    print(f"[GAMMA DB] After processing: {len(df)} rows, cols={list(df.columns)}")
    return df


def append_gamma_snapshot(spot, gamma_idx, net_delta, call_wall, put_wall,
                          vol_trigger, trade_date=None, path=None):
    """Append today's snapshot to gamma history CSV.
    Same day → replaces last row for that date. New day → appends new row.
    Numbers formatted with fixed decimal places to avoid CSV corruption."""
    fpath = path or GAMMA_HISTORY_PATH
    dt = trade_date or datetime.now().strftime('%Y-%m-%d')
    new_row = (f"{dt},{int(round(spot))},{float(gamma_idx):.4f},"
               f"{int(round(net_delta))},{int(round(call_wall))},"
               f"{int(round(put_wall))},{int(round(vol_trigger))},{dt}\n")
    header = "Trade Date,Ref Px,Net Gamma,Net Delta,Call Wall,Put Wall,Vol Trigger,Data Release\n"

    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            lines = f.readlines()
        # Ensure every line ends with \n to prevent row merging on write
        lines = [ln if ln.endswith('\n') else ln + '\n' for ln in lines]
        kept = [lines[0]] if lines else [header]  # keep header
        for line in lines[1:]:
            # Keep only rows that are NOT today's date
            if not line.startswith(dt + ','):
                kept.append(line)
        kept.append(new_row)
        with open(fpath, 'w') as f:
            f.writelines(kept)
    else:
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, 'w') as f:
            f.write(header)
            f.write(new_row)
    print(f"[GAMMA DB] Snapshot saved for {dt}")


def build_rv_gamma_chart(gamma_hist, current_gamma=None, current_rv=None):
    """
    Scatter: RV 21d (y) vs Gamma Index (x) com regressão OLS + teste de hipótese.
    Colore por regime: gamma positivo (verde), gamma negativo (vermelho).
    Destaca ponto atual + forecasted RV.
    """
    from scipy import stats as sp_stats

    df = gamma_hist.dropna(subset=['rv21d', 'gamma']).copy()
    if df.empty:
        return wd.HTML('<p style="color:#8b949e;">Sem dados históricos de gamma.</p>')

    pos = df[df['gamma'] >= 0]
    neg = df[df['gamma'] < 0]

    fig = go.Figure()

    # Positive gamma regime (green)
    if not pos.empty:
        fig.add_trace(go.Scatter(
            x=pos['gamma'], y=pos['rv21d'] * 100,
            mode='markers', name='γ ≥ 0 (Longo Gamma)',
            marker=dict(color='rgba(63,185,80,0.45)', size=5),
            text=pos['date'].dt.strftime('%Y-%m-%d') + '<br>Px: ' + pos['px'].astype(str),
            hovertemplate='<b>%{text}</b><br>Gamma: %{x:.2f}<br>RV21d: %{y:.1f}%<extra></extra>',
        ))

    # Negative gamma regime (red)
    if not neg.empty:
        fig.add_trace(go.Scatter(
            x=neg['gamma'], y=neg['rv21d'] * 100,
            mode='markers', name='γ < 0 (Curto Gamma)',
            marker=dict(color='rgba(248,81,73,0.45)', size=5),
            text=neg['date'].dt.strftime('%Y-%m-%d') + '<br>Px: ' + neg['px'].astype(str),
            hovertemplate='<b>%{text}</b><br>Gamma: %{x:.2f}<br>RV21d: %{y:.1f}%<extra></extra>',
        ))

    # ── OLS Regression line + hypothesis test ──
    x_all = df['gamma'].values.astype(float)
    y_all = (df['rv21d'] * 100).values.astype(float)
    slope, intercept, r_val, p_val, std_err = sp_stats.linregress(x_all, y_all)
    r2 = r_val ** 2
    n = len(x_all)
    f_stat = (r2 / 1) / ((1 - r2) / (n - 2)) if r2 < 1 and n > 2 else np.inf
    f_pval = 1 - sp_stats.f.cdf(f_stat, 1, n - 2) if n > 2 else 1.0

    x_fit = np.linspace(x_all.min(), x_all.max(), 200)
    y_fit = slope * x_fit + intercept
    fig.add_trace(go.Scatter(
        x=x_fit, y=y_fit,
        mode='lines', name=f'OLS (R²={r2:.3f}, p={p_val:.2e})',
        line=dict(color='#58a6ff', width=2.5, dash='solid'),
    ))

    # ── Confidence band (95%) ──
    x_mean = x_all.mean()
    se_fit = std_err * np.sqrt(1/n + (x_fit - x_mean)**2 / ((x_all - x_mean)**2).sum())
    t_crit = sp_stats.t.ppf(0.975, n - 2)
    y_upper = y_fit + t_crit * se_fit
    y_lower = y_fit - t_crit * se_fit
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_fit, x_fit[::-1]]),
        y=np.concatenate([y_upper, y_lower[::-1]]),
        fill='toself', fillcolor='rgba(88,166,255,0.12)',
        line=dict(width=0), showlegend=False,
        hoverinfo='skip',
    ))

    # Forecasted RV at current gamma
    forecasted_rv = None
    if current_gamma is not None:
        forecasted_rv = slope * current_gamma + intercept
        fig.add_trace(go.Scatter(
            x=[current_gamma], y=[forecasted_rv],
            mode='markers', name=f'Forecast: {forecasted_rv:.1f}%',
            marker=dict(color='#58a6ff', size=12, symbol='diamond',
                        line=dict(width=2, color='white')),
            hovertemplate=f'<b>FORECAST</b><br>Gamma: {current_gamma:.2f}<br>'
                          f'RV Prevista: {forecasted_rv:.1f}%<extra></extra>',
        ))

    # Último dia histórico do CSV (marcador destacado)
    last_row = df.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[last_row['gamma']], y=[last_row['rv21d'] * 100],
        mode='markers+text', name=f"Último ({last_row['date'].strftime('%Y-%m-%d')})",
        marker=dict(color='#ffffff', size=13, symbol='circle',
                    line=dict(width=2.5, color='#f0883e')),
        text=[last_row['date'].strftime('%m/%d')],
        textposition='top center',
        textfont=dict(size=9, color='#f0883e'),
        hovertemplate=(f"<b>ÚLTIMO HISTÓRICO</b><br>"
                       f"Data: {last_row['date'].strftime('%Y-%m-%d')}<br>"
                       f"Gamma: %{{x:.2f}}<br>RV21d: %{{y:.1f}}%<extra></extra>"),
    ))

    # Current point
    if current_gamma is not None and current_rv is not None:
        fig.add_trace(go.Scatter(
            x=[current_gamma], y=[current_rv * 100],
            mode='markers', name='HOJE (Realizado)',
            marker=dict(color='#f0883e', size=14, symbol='star',
                        line=dict(width=2, color='white')),
            hovertemplate='<b>HOJE</b><br>Gamma: %{x:.2f}<br>RV21d: %{y:.1f}%<extra></extra>',
        ))

    # Add zero gamma vertical line
    fig.add_vline(x=0, line=dict(color='#8b949e', dash='dash', width=1),
                  annotation_text='Gamma Flip', annotation_position='top')

    # RV percentiles as horizontal lines
    all_rv = df['rv21d'].dropna() * 100
    for pct, lbl, clr in [(0.75, '75th', '#d29922'), (0.90, '90th', '#f85149')]:
        val = all_rv.quantile(pct)
        fig.add_hline(y=val, line=dict(color=clr, dash='dot', width=1),
                      annotation_text=f'RV {lbl}: {val:.0f}%',
                      annotation_position='right')

    # Annotation with regression stats
    sig_txt = '✅ Significativo' if p_val < 0.05 else '❌ Não significativo'
    ann_text = (f"OLS: RV = {slope:.3f}·γ + {intercept:.1f}<br>"
                f"R² = {r2:.4f} | p = {p_val:.2e}<br>"
                f"F = {f_stat:.1f} (p={f_pval:.2e}) | n = {n}<br>"
                f"{sig_txt}")
    if forecasted_rv is not None and current_rv is not None:
        delta = current_rv * 100 - forecasted_rv
        ann_text += f"<br>Forecast: {forecasted_rv:.1f}% | Δ: {delta:+.1f}pp"
    fig.add_annotation(
        x=0.02, y=0.98, xref='paper', yref='paper',
        text=ann_text, showarrow=False,
        font=dict(size=10, color='#c9d1d9'),
        bgcolor='rgba(22,27,34,0.85)', bordercolor='#30363d',
        borderwidth=1, borderpad=6, align='left',
        xanchor='left', yanchor='top')

    fig.update_layout(
        title='RV Realizada 21d vs Gamma Index — Regressão + Teste de Hipótese',
        template=DASH_TEMPLATE,
        height=480,
        margin=dict(l=50, r=80, t=45, b=40),
        xaxis_title='Gamma Index (Net GEX Bn)',
        yaxis_title='Realized Vol 21d (%)',
        legend=dict(orientation='h', yanchor='bottom', y=-0.22,
                    xanchor='center', x=0.5),
    )
    return go.FigureWidget(fig)


def build_gamma_ts_chart(gamma_hist):
    """
    Time-series: Gamma Index + Call Wall + Put Wall + Vol Trigger + Ref Px.
    Dual axis: left = gamma/walls, right = price.
    """
    df = gamma_hist.dropna(subset=['gamma']).copy()
    if df.empty:
        return wd.HTML('<p style="color:#8b949e;">Sem dados históricos de gamma.</p>')

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['gamma'],
        name='Gamma Index', line=dict(color=_C['accent'], width=1.5),
        fill='tozeroy', fillcolor='rgba(88,166,255,0.08)'),
        secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['call_wall'],
        name='Call Wall', line=dict(color=_C['green'], width=1, dash='dot')),
        secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['put_wall'],
        name='Put Wall', line=dict(color=_C['red'], width=1, dash='dot')),
        secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['vol_trigger'],
        name='Vol Trigger (Gamma Flip)', line=dict(color=_C['yellow'], width=1.2, dash='dash')),
        secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['px'],
        name='SPX', line=dict(color='#ffffff', width=1)),
        secondary_y=True)

    fig.update_layout(
        title='Gamma Index + Walls + Vol Trigger vs SPX',
        template=DASH_TEMPLATE,
        height=420,
        margin=dict(l=50, r=60, t=45, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=-0.25,
                    xanchor='center', x=0.5),
    )
    fig.update_yaxes(title_text='Gamma Index (Bn)', secondary_y=False)
    fig.update_yaxes(title_text='SPX Level', secondary_y=True)

    return go.FigureWidget(fig)


def build_dispersion_index_chart(cor1m, dspx, vixeq):
    """
    Gráfico estilo Bloomberg G 1059: COR1M + DSPX + VIXEQ (single stock vol premium).
    3 séries no mesmo chart, eixo Y compartilhado.
    """
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=1)

    if not cor1m.empty:
        fig.add_trace(go.Scatter(
            x=cor1m.index, y=cor1m.values,
            name='Impl Corr 1M 50Δ (COR1M)',
            line=dict(color='#FFFFFF', width=1.5)))
    if not dspx.empty:
        fig.add_trace(go.Scatter(
            x=dspx.index, y=dspx.values,
            name='CBOE S&P 500 Dispersion (DSPX)',
            line=dict(color='#58a6ff', width=1.5)))
    if not vixeq.empty:
        fig.add_trace(go.Scatter(
            x=vixeq.index, y=vixeq.values,
            name='Single Stock Vol Premium (VIXEQ)',
            line=dict(color='#f0883e', width=1.5)))

    fig.update_layout(
        title='Low Implied Correlation and Rising Dispersion for S&P 500',
        template=DASH_TEMPLATE,
        height=380,
        margin=dict(l=50, r=50, t=40, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=-0.25,
                    xanchor='center', x=0.5),
        yaxis_title='Index Level',
    )
    return go.FigureWidget(fig)


def _tail_metrics_widget(metrics):
    """Widget de métricas de tail risk."""
    if not metrics:
        return wd.HTML('<p style="color:#8b949e;">Sem dados de tail risk.</p>')
    html = '<div style="color:#c9d1d9; font-size:12px;">'
    html += '<b>Tail Risk Metrics (EVT)</b>'
    html += '<table style="border-collapse:collapse; margin-top:5px;">'
    for k, v in metrics.items():
        color = '#da3633' if 'Loss' in str(k) or 'CTE' in str(k) else '#c9d1d9'
        html += '<tr>'
        html += '<td style="padding:3px 10px; color:#8b949e;">{}</td>'.format(k)
        html += '<td style="padding:3px 10px; color:{}; font-weight:600;">'.format(color)
        html += '{}</td>'.format(v)
        html += '</tr>'
    html += '</table></div>'
    return wd.HTML(html)


# ── Multi-Window Correlation Matrix + Dispersion Trade Engine ─────────────

def compute_multi_window_correlations(prices_df, windows=None):
    """
    Calcula matrizes de correlação para múltiplas janelas temporais.
    windows: dict {label: n_days} — e.g., {'1D': 1, '5D': 5, '1M_roll5D': (21,5)}
    Para tupla (W, R): rolling(R) sobre retornos de W dias.
    Retorna dict {label: corr_matrix (DataFrame)}.
    """
    if windows is None:
        windows = {'Intraday (1D)': 1, '5D': 5, '1M rolling 5D': (21, 5)}

    log_rets_1d = np.log(prices_df / prices_df.shift(1)).dropna()
    result = {}

    for label, spec in windows.items():
        if isinstance(spec, tuple):
            # Rolling window: compute W-day returns, then rolling R-day corr
            w_days, r_days = spec
            rets_w = np.log(prices_df / prices_df.shift(w_days)).dropna()
            if len(rets_w) >= r_days:
                result[label] = rets_w.tail(r_days * 4).corr()
        elif spec == 1:
            # Last N days of daily returns for intraday proxy
            result[label] = log_rets_1d.tail(21).corr()
        else:
            # Simple N-day return correlation
            rets_n = np.log(prices_df / prices_df.shift(spec)).dropna()
            if len(rets_n) >= 21:
                result[label] = rets_n.tail(63).corr()

    return result


def build_correlation_heatmap(corr_matrix, title='Correlation Matrix'):
    """Heatmap interativo de correlação com Plotly."""
    tickers = [t.split(' ')[0] for t in corr_matrix.columns]
    z = corr_matrix.values

    fig = go.Figure(data=go.Heatmap(
        z=z, x=tickers, y=tickers,
        colorscale=[
            [0.0, '#da3633'], [0.25, '#f85149'],
            [0.5, '#161b22'], [0.75, '#238636'],
            [1.0, '#3fb950']],
        zmin=-1, zmax=1,
        text=np.round(z, 2), texttemplate='%{text}',
        textfont=dict(size=9),
        hovertemplate='%{x} vs %{y}: %{z:.3f}<extra></extra>',
    ))
    fig.update_layout(
        title=title,
        template=DASH_TEMPLATE,
        height=400, width=450,
        margin=dict(l=60, r=20, t=40, b=60),
        xaxis=dict(side='bottom', tickangle=45),
    )
    return go.FigureWidget(fig)


def find_dispersion_pairs(corr_matrices, iv_latest, n_pairs=8):
    """
    Identifica os melhores pares para dispersion trade via straddle/strangle.
    Critérios:
    1. Baixa correlação (ou negativa) entre os ativos
    2. Alto diferencial de IV
    3. Consistência entre janelas temporais
    Retorna DataFrame com pares rankeados.
    """
    # Collect pairwise metrics across windows
    pair_scores = {}
    tickers = None

    for label, corr in corr_matrices.items():
        if tickers is None:
            tickers = list(corr.columns)
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                t1, t2 = tickers[i], tickers[j]
                key = (t1, t2)
                if key not in pair_scores:
                    pair_scores[key] = {'corrs': [], 'labels': []}
                pair_scores[key]['corrs'].append(corr.iloc[i, j])
                pair_scores[key]['labels'].append(label)

    if not pair_scores:
        return pd.DataFrame()

    rows = []
    for (t1, t2), info in pair_scores.items():
        avg_corr = np.mean(info['corrs'])
        min_corr = np.min(info['corrs'])
        iv1 = iv_latest.get(t1, np.nan)
        iv2 = iv_latest.get(t2, np.nan)
        iv_spread = abs(iv1 - iv2) if not (np.isnan(iv1) or np.isnan(iv2)) else 0

        # Score: low correlation + high IV spread = better dispersion opportunity
        # Normalize: corr contributes negatively (lower = better), IV spread positively
        disp_score = (1 - avg_corr) * 0.4 + (1 - min_corr) * 0.3 + iv_spread * 100 * 0.3

        rows.append({
            'Pair': f"{t1.split(' ')[0]}/{t2.split(' ')[0]}",
            'Ticker1': t1, 'Ticker2': t2,
            'Avg Corr': round(avg_corr, 3),
            'Min Corr': round(min_corr, 3),
            'IV1 (%)': round(iv1 * 100, 1) if not np.isnan(iv1) else np.nan,
            'IV2 (%)': round(iv2 * 100, 1) if not np.isnan(iv2) else np.nan,
            'IV Spread (pp)': round(iv_spread * 100, 1),
            'Disp Score': round(disp_score, 2),
        })

    df = pd.DataFrame(rows).sort_values('Disp Score', ascending=False).head(n_pairs)
    return df.reset_index(drop=True)


def fetch_straddle_prices(tickers, expiry_range='30D'):
    """
    Busca preços de straddle ATM (50-delta) + strangle 25-delta para tickers via BQL.
    Para cada ticker: cadeia de opções → ATM call+put (straddle) + 25Δ put + 25Δ call (strangle).
    Retorna dict {ticker: {call_iv, put_iv, straddle_iv, straddle_px, straddle_pct,
                           strangle_iv, strangle_px, strangle_pct, p25_iv, c25_iv, ...}}.
    """
    bq = bql.Service()
    results = {}

    for tk in tickers:
        try:
            # Spot price
            spot_req = bql.Request(tk, {'px': bq.data.px_last()})
            spot_val = bq.execute(spot_req)[0].df().reset_index()
            spot_v = float(spot_val.iloc[0]['px']) if len(spot_val) > 0 else np.nan
            if np.isnan(spot_v):
                continue

            # Options universe: 15-45 DTE
            opt_univ = bq.univ.filter(
                bq.univ.options(tk),
                bq.func.and_(
                    bq.data.expire_dt() >= bq.func.today() + '15D',
                    bq.data.expire_dt() <= bq.func.today() + '45D',
                ))

            req = bql.Request(opt_univ, {
                'strike': bq.data.strike_px(),
                'pc': bq.data.put_call(),
                'iv': bq.data.ivol(),
                'bid': bq.data.px_bid(),
                'ask': bq.data.px_ask(),
                'delta': bq.data.delta(),
                'expiry': bq.data.expire_dt(),
            })
            resp = bq.execute(req)

            # Build DataFrame from all response items
            dfs = []
            for item in resp:
                _d = item.df().reset_index()
                if not _d.empty:
                    dfs.append(_d)
            if not dfs:
                continue

            df = pd.concat(dfs, axis=1)
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.dropna(subset=['strike', 'pc'])

            # Ensure numeric delta
            df['delta'] = pd.to_numeric(df['delta'], errors='coerce')

            # ── ATM Straddle ──
            df['dist'] = (df['strike'] - spot_v).abs()
            atm_strike = df.loc[df['dist'].idxmin(), 'strike']

            atm = df[df['strike'] == atm_strike]
            calls_atm = atm[atm['pc'] == 'Call']
            puts_atm = atm[atm['pc'] == 'Put']

            if calls_atm.empty or puts_atm.empty:
                continue

            call_row = calls_atm.iloc[0]
            put_row = puts_atm.iloc[0]

            call_mid = (call_row.get('bid', 0) + call_row.get('ask', 0)) / 2
            put_mid = (put_row.get('bid', 0) + put_row.get('ask', 0)) / 2
            call_iv = float(call_row.get('iv', np.nan))
            put_iv = float(put_row.get('iv', np.nan))

            straddle_px = call_mid + put_mid
            straddle_iv = (call_iv + put_iv) / 2 if not (np.isnan(call_iv) or np.isnan(put_iv)) else np.nan

            # ── 25-Delta Strangle ──
            calls_all = df[df['pc'] == 'Call'].copy()
            puts_all = df[df['pc'] == 'Put'].copy()

            c25_iv, p25_iv, strangle_px, strangle_iv = np.nan, np.nan, np.nan, np.nan
            c25_mid, p25_mid = 0.0, 0.0

            # 25-delta call: delta closest to 0.25
            if not calls_all.empty and calls_all['delta'].notna().any():
                calls_all['d25_dist'] = (calls_all['delta'].abs() - 0.25).abs()
                c25_row = calls_all.loc[calls_all['d25_dist'].idxmin()]
                c25_iv = float(c25_row.get('iv', np.nan))
                c25_mid = (c25_row.get('bid', 0) + c25_row.get('ask', 0)) / 2

            # 25-delta put: delta closest to -0.25
            if not puts_all.empty and puts_all['delta'].notna().any():
                puts_all['d25_dist'] = (puts_all['delta'].abs() - 0.25).abs()
                p25_row = puts_all.loc[puts_all['d25_dist'].idxmin()]
                p25_iv = float(p25_row.get('iv', np.nan))
                p25_mid = (p25_row.get('bid', 0) + p25_row.get('ask', 0)) / 2

            strangle_px = c25_mid + p25_mid
            if not (np.isnan(c25_iv) or np.isnan(p25_iv)):
                strangle_iv = (c25_iv + p25_iv) / 2

            results[tk] = {
                'spot': spot_v,
                'strike': float(atm_strike),
                'expiry': str(call_row.get('expiry', '')),
                'call_iv': call_iv,
                'put_iv': put_iv,
                'straddle_iv': straddle_iv,
                'call_mid': call_mid,
                'put_mid': put_mid,
                'straddle_px': straddle_px,
                'straddle_pct': straddle_px / spot_v * 100 if spot_v > 0 else np.nan,
                'c25_iv': c25_iv,
                'p25_iv': p25_iv,
                'strangle_iv': strangle_iv,
                'strangle_px': strangle_px,
                'strangle_pct': strangle_px / spot_v * 100 if spot_v > 0 else np.nan,
            }
        except Exception as e:
            print(f"⚠️ Straddle {tk}: {e}")

    return results


def fetch_historical_straddle_iv(tickers, lookback=252):
    """
    Busca IV ATM histórica para avaliar se straddle está caro/barato.
    Usa implied_volatility(expiry='30D', pct_moneyness='100').
    Retorna DataFrame com IV histórica de cada ticker.
    """
    bq = bql.Service()
    dt_range = bq.func.range(f'-{lookback}d', '0d')
    iv_hist = {}

    for tk in tickers:
        try:
            req = bql.Request(tk, {
                'iv': bq.data.implied_volatility(
                    expiry='30D', pct_moneyness='100', fill='PREV', dates=dt_range),
            })
            s = _bql_ts(bq.execute(req)[0], 'iv')
            if not s.empty:
                iv_hist[tk] = s
        except Exception:
            pass

    if not iv_hist:
        return pd.DataFrame()
    return pd.DataFrame(iv_hist)


def compute_straddle_richness(straddle_data, iv_hist_df, rv_df=None):
    """
    Avalia se cada straddle está caro ou barato.
    - Percentil da IV atual vs histórico
    - IV vs RV spread (se rv_df disponível)
    Retorna DataFrame com métricas de richness.
    """
    rows = []
    for tk, data in straddle_data.items():
        iv_now = data.get('straddle_iv', np.nan)
        if np.isnan(iv_now):
            continue

        short_name = tk.split(' ')[0]

        # Percentil vs histórico
        hist = iv_hist_df[tk].dropna() if tk in iv_hist_df.columns else pd.Series(dtype=float)
        if len(hist) > 20:
            pct = (hist < iv_now).sum() / len(hist) * 100
            avg_iv = hist.mean()
            std_iv = hist.std()
            z_score = (iv_now - avg_iv) / std_iv if std_iv > 0 else 0
        else:
            pct, avg_iv, std_iv, z_score = np.nan, np.nan, np.nan, np.nan

        # IV-RV spread
        rv_now = np.nan
        if rv_df is not None and tk in rv_df.columns:
            rv_series = rv_df[tk].dropna()
            if len(rv_series) > 0:
                rv_now = rv_series.iloc[-1]

        iv_rv_spread = iv_now - rv_now if not np.isnan(rv_now) else np.nan

        rows.append({
            'Ticker': short_name,
            'IV Atual (%)': round(iv_now * 100, 1) if iv_now < 1 else round(iv_now, 1),
            'IV Média (%)': round(avg_iv * 100, 1) if not np.isnan(avg_iv) and avg_iv < 1 else round(avg_iv, 1) if not np.isnan(avg_iv) else np.nan,
            'Percentil': round(pct, 0) if not np.isnan(pct) else np.nan,
            'Z-Score': round(z_score, 2) if not np.isnan(z_score) else np.nan,
            'RV 21d (%)': round(rv_now * 100, 1) if not np.isnan(rv_now) and rv_now < 1 else round(rv_now, 1) if not np.isnan(rv_now) else np.nan,
            'IV-RV (pp)': round(iv_rv_spread * 100, 1) if not np.isnan(iv_rv_spread) and abs(iv_rv_spread) < 1 else round(iv_rv_spread, 1) if not np.isnan(iv_rv_spread) else np.nan,
            'Straddle (%)': round(data['straddle_pct'], 2),
            'Sinal': 'CARO' if not np.isnan(pct) and pct > 75 else ('BARATO' if not np.isnan(pct) and pct < 25 else 'NEUTRO'),
        })
    return pd.DataFrame(rows)


def build_atm_vol_matrix(straddle_data):
    """
    Constrói matriz de ATM vol: Call/Put IV, Straddle IV, Strangle 25Δ IV,
    Straddle %, Strangle % para Mag8 + SPX.
    Retorna (DataFrame, FigureWidget).
    """
    rows = []
    for tk, data in straddle_data.items():
        short = tk.split(' ')[0]
        call_iv = data.get('call_iv', np.nan)
        put_iv = data.get('put_iv', np.nan)
        strl_iv = data.get('straddle_iv', np.nan)
        c25_iv = data.get('c25_iv', np.nan)
        p25_iv = data.get('p25_iv', np.nan)
        strg_iv = data.get('strangle_iv', np.nan)
        straddle_pct = data.get('straddle_pct', np.nan)
        strangle_pct = data.get('strangle_pct', np.nan)
        # Normalize to % if value is in decimal form (< 1)
        if not np.isnan(call_iv) and call_iv < 1:
            call_iv *= 100
        if not np.isnan(put_iv) and put_iv < 1:
            put_iv *= 100
        if not np.isnan(strl_iv) and strl_iv < 1:
            strl_iv *= 100
        if not np.isnan(c25_iv) and c25_iv < 1:
            c25_iv *= 100
        if not np.isnan(p25_iv) and p25_iv < 1:
            p25_iv *= 100
        if not np.isnan(strg_iv) and strg_iv < 1:
            strg_iv *= 100
        skew = put_iv - call_iv if not (np.isnan(put_iv) or np.isnan(call_iv)) else np.nan
        rows.append({
            'Ticker': short,
            'Call IV (%)': round(call_iv, 1) if not np.isnan(call_iv) else np.nan,
            'Put IV (%)': round(put_iv, 1) if not np.isnan(put_iv) else np.nan,
            'Straddle IV (%)': round(strl_iv, 1) if not np.isnan(strl_iv) else np.nan,
            'Straddle (%)': round(straddle_pct, 2) if not np.isnan(straddle_pct) else np.nan,
            '25Δ Call IV (%)': round(c25_iv, 1) if not np.isnan(c25_iv) else np.nan,
            '25Δ Put IV (%)': round(p25_iv, 1) if not np.isnan(p25_iv) else np.nan,
            'Strangle IV (%)': round(strg_iv, 1) if not np.isnan(strg_iv) else np.nan,
            'Strangle (%)': round(strangle_pct, 2) if not np.isnan(strangle_pct) else np.nan,
            'Skew (pp)': round(skew, 1) if not np.isnan(skew) else np.nan,
        })
    matrix_df = pd.DataFrame(rows)
    if matrix_df.empty:
        return matrix_df, wd.HTML('<p style="color:#8b949e;">Sem dados de ATM vol.</p>')

    # Build grouped bar chart: Straddle IV vs Strangle IV + skew overlay
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=matrix_df['Ticker'], y=matrix_df['Straddle IV (%)'],
        name='Straddle IV', marker_color='rgba(63,185,80,0.75)',
        text=matrix_df['Straddle IV (%)'], textposition='outside',
        textfont=dict(size=10),
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=matrix_df['Ticker'], y=matrix_df['Strangle IV (%)'],
        name='Strangle 25Δ IV', marker_color='rgba(88,166,255,0.75)',
        text=matrix_df['Strangle IV (%)'], textposition='outside',
        textfont=dict(size=10),
    ), secondary_y=False)

    # Add straddle % as spot-normalized cost
    fig.add_trace(go.Scatter(
        x=matrix_df['Ticker'], y=matrix_df['Straddle (%)'],
        name='Straddle Cost (%Spot)', mode='markers+lines',
        marker=dict(color=_C['purple'], size=9, symbol='circle'),
        line=dict(color=_C['purple'], width=1.5, dash='dot'),
    ), secondary_y=True)

    # Skew diamonds
    fig.add_trace(go.Scatter(
        x=matrix_df['Ticker'], y=matrix_df['Skew (pp)'],
        name='Put-Call Skew', mode='markers',
        marker=dict(color=_C['yellow'], size=10, symbol='diamond'),
    ), secondary_y=True)

    fig.update_layout(
        title='ATM Vol Matrix — Straddle & Strangle (Mag8 + SPX)',
        template=DASH_TEMPLATE,
        height=450,
        barmode='group',
        margin=dict(l=50, r=60, t=50, b=40),
        xaxis_title='',
        legend=dict(orientation='h', yanchor='bottom', y=-0.20,
                    xanchor='center', x=0.5, font=dict(size=11)),
    )
    fig.update_yaxes(title_text='IV (%)', secondary_y=False)
    fig.update_yaxes(title_text='Cost (%) / Skew (pp)', secondary_y=True, showgrid=False)
    return matrix_df, go.FigureWidget(fig)


def build_intraday_mag8_chart(prices_df, index_ticker='SPX Index'):
    """
    Gráfico intraday (última sessão) de Mag8 + SPX normalizados como % change.
    Usa dados BQL de preço intraday 1-min bars.
    Se intraday não disponível, usa últimos 5 dias normalized.
    """
    tickers = [t for t in MAG8 if t in prices_df.columns]
    if index_ticker in prices_df.columns:
        tickers.append(index_ticker)

    if not tickers or len(prices_df) < 2:
        return wd.HTML('<p style="color:#8b949e;">Sem dados intraday.</p>')

    # Use last 5 trading days for near-intraday view
    df = prices_df[tickers].tail(5).copy()
    # Normalize: % change from start of window
    base = df.iloc[0]
    pct_change = (df / base - 1) * 100

    fig = go.Figure()
    colors = ['#3fb950', '#f85149', '#58a6ff', '#bc8cff',
              '#d29922', '#f0883e', '#8b949e', '#da3633', 'white']

    for i, tk in enumerate(tickers):
        short = tk.split(' ')[0]
        is_index = 'Index' in tk
        fig.add_trace(go.Scatter(
            x=pct_change.index, y=pct_change[tk],
            name=short,
            line=dict(color=colors[i % len(colors)],
                      width=3 if is_index else 1.5,
                      dash='solid' if is_index else 'solid'),
            opacity=1.0 if is_index else 0.8,
        ))

    fig.add_hline(y=0, line=dict(color='#30363d', width=1))

    fig.update_layout(
        title='Mag8 + SPX — Retorno Recente (% Change, últimos 5D)',
        template=DASH_TEMPLATE,
        height=400,
        margin=dict(l=50, r=30, t=45, b=30),
        xaxis_title='', yaxis_title='% Change',
        legend=dict(orientation='h', yanchor='bottom', y=-0.22,
                    xanchor='center', x=0.5),
    )
    return go.FigureWidget(fig)


def build_dispersion_trade_recommendations(pair_df, richness_df, straddle_data):
    """
    Gera recomendações de trade de dispersão com interpretação clara e detalhada.
    Para cada par: operação exata, direção, preços, IV, percentil.
    """
    if pair_df.empty or richness_df.empty:
        return pd.DataFrame(), ''

    richness_map = {}
    for _, row in richness_df.iterrows():
        richness_map[row['Ticker']] = row

    trades = []
    interp_lines = []
    for idx, (_, pair) in enumerate(pair_df.iterrows()):
        t1_short = pair['Ticker1'].split(' ')[0]
        t2_short = pair['Ticker2'].split(' ')[0]
        r1 = richness_map.get(t1_short, {})
        r2 = richness_map.get(t2_short, {})

        if not r1 or not r2:
            continue

        pct1 = r1.get('Percentil', 50)
        pct2 = r2.get('Percentil', 50)
        if pd.isna(pct1):
            pct1 = 50
        if pd.isna(pct2):
            pct2 = 50

        # Straddle data for each leg
        s1 = straddle_data.get(pair['Ticker1'], {})
        s2 = straddle_data.get(pair['Ticker2'], {})

        # Determine direction: buy cheap vol, sell expensive vol
        if pct1 > pct2:
            long_leg, short_leg = t2_short, t1_short
            long_pct, short_pct = pct2, pct1
            long_s, short_s = s2, s1
            long_r, short_r = r2, r1
        else:
            long_leg, short_leg = t1_short, t2_short
            long_pct, short_pct = pct1, pct2
            long_s, short_s = s1, s2
            long_r, short_r = r1, r2

        spread_pct = short_pct - long_pct
        edge = pair['Disp Score'] * (spread_pct / 100) if spread_pct > 0 else 0

        # Build detailed interpretation
        long_iv = long_r.get('IV Atual (%)', '?')
        short_iv = short_r.get('IV Atual (%)', '?')
        long_strd_pct = long_s.get('straddle_pct', 0)
        short_strd_pct = short_s.get('straddle_pct', 0)
        long_expiry = long_s.get('expiry', '30D')
        short_expiry = short_s.get('expiry', '30D')

        op_tipo = 'DISPERSÃO' if pair['Avg Corr'] < 0.5 else 'RELATIVE VALUE'

        # Clear leg descriptions
        long_desc = f"COMPRAR Straddle ATM {long_leg} (IV={long_iv}%, pctl={long_pct:.0f}th — BARATO)"
        short_desc = f"VENDER Straddle ATM {short_leg} (IV={short_iv}%, pctl={short_pct:.0f}th — CARO)"

        trades.append({
            'Operação': f'{long_leg}/{short_leg}',
            'Leg 1 (Long)': long_desc,
            'Leg 2 (Short)': short_desc,
            'Corr': pair['Avg Corr'],
            'Long IV Pctl': round(long_pct, 0),
            'Short IV Pctl': round(short_pct, 0),
            'Straddle Long (%)': round(long_strd_pct, 2) if long_strd_pct else np.nan,
            'Straddle Short (%)': round(short_strd_pct, 2) if short_strd_pct else np.nan,
            'Edge Score': round(edge, 2),
            'Tipo': op_tipo,
        })

        if idx < 3:  # Top 3 detailed interpretation
            interp_lines.append(
                f"<b style='color:#58a6ff;'>Trade {idx + 1}: {op_tipo} — {long_leg} vs {short_leg}</b><br>"
                f"&nbsp;&nbsp;📈 {long_desc}<br>"
                f"&nbsp;&nbsp;📉 {short_desc}<br>"
                f"&nbsp;&nbsp;↔ Correlação: {pair['Avg Corr']:.3f} | Edge: {edge:.2f} | "
                f"Spread Pctl: {spread_pct:.0f}pp"
            )

    df = pd.DataFrame(trades).sort_values('Edge Score', ascending=False)
    df = df.head(8).reset_index(drop=True)

    # Build interpretation HTML
    interpretation = ''
    if interp_lines:
        interpretation = (
            "<div style='background:#161b22; padding:12px; border-radius:6px; "
            "margin:5px 0; border-left:3px solid #58a6ff;'>"
            "<b style='color:#f0883e; font-size:13px;'>🎯 Interpretação — "
            "O que fazer exatamente:</b><br><br>"
            + "<br><br>".join(interp_lines)
            + "<br><br><small style='color:#8b949e;'>"
            "Long straddle = comprar ATM call + ATM put (aposta que vol sobe ou "
            "ativo se move). Short straddle = vender ATM call + ATM put (aposta "
            "que vol cai ou ativo fica parado). Spread de dispersão lucra quando "
            "a vol individual diverge da vol do índice.</small></div>"
        )

    return df, interpretation


def train_dispersion_model(prices_df, iv_df, lookback=126):
    """
    Modelo de ML simples (Logistic Regression) para prever dispersão futura.
    Features: IV spread, correlação rolling, RV ratio, momentum divergence.
    Target: retorno da dispersão foi positivo nos próximos 5 dias?
    Retorna (model, feature_names, accuracy, feature_importance).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit

    tickers = list(prices_df.columns)
    if len(tickers) < 2:
        return None, [], 0, {}

    log_rets = np.log(prices_df / prices_df.shift(1)).dropna()

    # Build features for each day
    features_list = []
    targets = []

    for end_idx in range(63, len(log_rets) - 5):
        window = log_rets.iloc[end_idx - 21:end_idx]
        window_long = log_rets.iloc[end_idx - 63:end_idx]

        # Feature 1: Average pairwise correlation (21d)
        corr_21 = window.corr()
        mask = np.triu(np.ones(corr_21.shape, dtype=bool), k=1)
        avg_corr = corr_21.values[mask].mean()

        # Feature 2: Std of individual vols (cross-sectional vol dispersion)
        vol_cs = window.std() * np.sqrt(252)
        vol_disp = vol_cs.std()

        # Feature 3: Correlation change (21d vs 63d)
        corr_63 = window_long.corr()
        avg_corr_63 = corr_63.values[mask[:corr_63.shape[0], :corr_63.shape[1]]].mean() \
            if corr_63.shape == corr_21.shape else avg_corr
        corr_chg = avg_corr - avg_corr_63

        # Feature 4: Mean IV level (if available)
        mean_iv = 0
        if iv_df is not None and not iv_df.empty:
            iv_slice = iv_df.iloc[min(end_idx, len(iv_df) - 1)]
            mean_iv = iv_slice.mean() if not iv_slice.isna().all() else 0

        # Feature 5: Return dispersion (cross-sectional std of returns)
        ret_disp = window.iloc[-1].std()

        # Feature 6: Momentum divergence (range of cumulative returns)
        cum_21 = window.sum()
        mom_div = cum_21.max() - cum_21.min()

        features_list.append([avg_corr, vol_disp, corr_chg, mean_iv, ret_disp, mom_div])

        # Target: was cross-sectional vol dispersion higher in next 5 days?
        future = log_rets.iloc[end_idx:end_idx + 5]
        future_disp = future.std().std() * np.sqrt(252)
        current_disp = vol_disp
        targets.append(1 if future_disp > current_disp else 0)

    if len(features_list) < 50:
        return None, [], 0, {}

    X = np.array(features_list)
    y = np.array(targets)

    # Remove NaN rows
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]

    if len(X) < 50:
        return None, [], 0, {}

    feature_names = ['Avg Corr 21d', 'Vol Dispersion', 'Corr Change',
                     'Mean IV', 'Return Dispersion', 'Momentum Divergence']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    accuracies = []
    for train_idx, test_idx in tscv.split(X_scaled):
        model = LogisticRegression(max_iter=500, C=0.1, random_state=42)
        model.fit(X_scaled[train_idx], y[train_idx])
        acc = model.score(X_scaled[test_idx], y[test_idx])
        accuracies.append(acc)

    # Train final model on all data
    final_model = LogisticRegression(max_iter=500, C=0.1, random_state=42)
    final_model.fit(X_scaled, y)

    importance = dict(zip(feature_names, np.abs(final_model.coef_[0])))

    # Store scaler for prediction
    final_model._scaler = scaler
    final_model._feature_names = feature_names

    return final_model, feature_names, np.mean(accuracies), importance


def predict_dispersion(model, prices_df, iv_df=None):
    """
    Usa modelo treinado para prever probabilidade de dispersão futura.
    Retorna (probability, features_dict).
    """
    if model is None:
        return 0.5, {}

    log_rets = np.log(prices_df / prices_df.shift(1)).dropna()
    if len(log_rets) < 63:
        return 0.5, {}

    window21 = log_rets.tail(21)
    window63 = log_rets.tail(63)

    corr_21 = window21.corr()
    mask = np.triu(np.ones(corr_21.shape, dtype=bool), k=1)
    avg_corr = corr_21.values[mask].mean()

    vol_cs = window21.std() * np.sqrt(252)
    vol_disp = vol_cs.std()

    corr_63 = window63.corr()
    mask_63 = np.triu(np.ones(corr_63.shape, dtype=bool), k=1)
    avg_corr_63 = corr_63.values[mask_63].mean()
    corr_chg = avg_corr - avg_corr_63

    mean_iv = 0
    if iv_df is not None and not iv_df.empty:
        iv_last = iv_df.iloc[-1]
        mean_iv = iv_last.mean() if not iv_last.isna().all() else 0

    ret_disp = window21.iloc[-1].std()
    cum_21 = window21.sum()
    mom_div = cum_21.max() - cum_21.min()

    X = np.array([[avg_corr, vol_disp, corr_chg, mean_iv, ret_disp, mom_div]])
    X_scaled = model._scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0, 1]

    features_dict = dict(zip(model._feature_names, X[0]))
    return float(prob), features_dict


def build_kde_distribution_chart(prices_df, weights=None):
    """
    Chart KDE de distribuição de retornos dos constituintes — estilo Tier1 Alpha.
    Dual display: distribuição individual (equi-weighted, red/green fill) +
    ponderada por peso no índice (yellow line on twin axis).
    Top-10 labels posicionados por peso (y = weight_adj * scale) no eixo secundário.
    Retorna (FigureWidget, HTML_interpretacao).
    """
    from scipy.stats import gaussian_kde

    log_rets_1d = (prices_df.iloc[-1] / prices_df.iloc[-2] - 1) * 100
    rets = log_rets_1d.dropna()

    if len(rets) < 5:
        return wd.HTML('<p style="color:#8b949e;">Dados insuficientes para KDE.</p>'), ''
    if rets.std() == 0:
        return wd.HTML('<p style="color:#8b949e;">Retornos sem variação (possível feriado/dia sem pregão).</p>'), ''

    n_down = int((rets < 0).sum())
    n_up = int((rets >= 0).sum())
    avg_all = rets.mean()
    avg_up = rets[rets >= 0].mean() if (rets >= 0).any() else 0
    avg_dn = rets[rets < 0].mean() if (rets < 0).any() else 0

    # ── Equi-weighted KDE ──
    kde = gaussian_kde(rets)
    x = np.linspace(rets.min() - 2, rets.max() + 2, 1000)
    y = kde(x)

    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # ── Fill areas: red (neg) / green (pos) — equi-weighted on primary y ──
    neg_mask = x <= 0
    pos_mask = x >= 0

    # Outline (grey, thin)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines',
        line=dict(color='rgba(150,150,150,0.5)', width=1),
        showlegend=False), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=x[neg_mask], y=y[neg_mask],
        fill='tozeroy', fillcolor='rgba(220,38,38,0.50)',
        line=dict(width=0),
        name=f'Down: {n_down} stocks',
        showlegend=False), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=x[pos_mask], y=y[pos_mask],
        fill='tozeroy', fillcolor='rgba(34,197,94,0.50)',
        line=dict(width=0),
        name=f'Up: {n_up} stocks',
        showlegend=False), secondary_y=False)

    # ── Weighted KDE (yellow line on secondary y) ──
    has_weighted = False
    if weights:
        w_arr = np.array([weights.get(t, 0) for t in rets.index])
        w_arr = w_arr / w_arr.sum() if w_arr.sum() > 0 else w_arr
        non_zero = w_arr > 0
        if non_zero.sum() > 3 and rets[non_zero].std() > 0:
            try:
                kde_w = gaussian_kde(rets[non_zero], weights=w_arr[non_zero])
                y_w = kde_w(x)
                fig.add_trace(go.Scatter(
                    x=x, y=y_w,
                    line=dict(color='#fbbf24', width=2.5),
                    name='Weighted distribution',
                    opacity=0.95), secondary_y=True)
                has_weighted = True
            except Exception:
                pass

    # ── Average lines ──
    fig.add_vline(x=0, line=dict(color='#6b7280', dash='solid', width=1))
    fig.add_vline(x=avg_all, line=dict(color='white', dash='dash', width=1.5))
    fig.add_vline(x=avg_up, line=dict(color='#22c55e', dash='dash', width=2))
    fig.add_vline(x=avg_dn, line=dict(color='#dc2626', dash='dash', width=2))

    # ── Top-10 by weight: scatter on SECONDARY y using weight as y-position ──
    # This naturally separates labels vertically (heavier stocks = higher y)
    interp_heavy = []
    if weights:
        w_series = pd.Series(weights)
        top10 = w_series.nlargest(10)
        # Compute weight-adjusted position like the reference: w_adj * scale_factor
        # w_adj = (price * w) / sum(price * w) for proper positioning
        top10_data = []
        for tk, w in top10.items():
            if tk in rets.index:
                top10_data.append((tk, w, rets[tk]))
        if top10_data:
            # Scale y: map weight to secondary-axis range for visibility
            max_y_secondary = max(y) * 1.1 if not has_weighted else 0
            if has_weighted:
                max_y_secondary = max(kde_w(x)) if 'kde_w' in dir() else max(y)
            scale = max_y_secondary * 3.5  # scale factor like reference code

            for tk, w, ret_val in top10_data:
                short_name = tk.split(' ')[0]
                # Remove common suffixes
                for sfx in ['UW', 'UQ', 'UN', 'US']:
                    short_name = short_name.replace(sfx, '')
                y_pos = w * scale
                fig.add_trace(go.Scatter(
                    x=[ret_val], y=[y_pos],
                    mode='markers+text',
                    marker=dict(color='#7dd3fc', size=7,
                                line=dict(width=0.5, color='white')),
                    text=[short_name],
                    textposition='top center',
                    textfont=dict(size=12, color='white'),
                    showlegend=False,
                    hovertemplate=f'<b>{short_name}</b><br>Ret: {ret_val:+.1f}%<br>'
                                  f'Peso: {w*100:.1f}%<extra></extra>',
                ), secondary_y=True)

                # Mean-reversion analysis
                w_pct = w * 100
                if tk in prices_df.columns and len(prices_df) > 21:
                    recent = (prices_df[tk].iloc[-21:] / prices_df[tk].iloc[-22:-1].values - 1) * 100
                    r_mean = recent.mean()
                    r_std = recent.std()
                    z = (ret_val - r_mean) / r_std if r_std > 0 else 0
                    if abs(z) > 1.5:
                        direction = 'ABAIXO da média 21d' if z < -1.5 else 'ACIMA da média 21d'
                        interp_heavy.append(
                            f"<b>{short_name}</b> (peso {w_pct:.1f}%): ret 1D = {ret_val:+.1f}%, "
                            f"Z-score 21d = {z:.1f} → {direction} — "
                            f"<b style='color:#fbbf24;'>possível reversão à média</b>")

    # ── Legend entries for stats ──
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='white', dash='dash', width=1.5),
        name=f'Avg. Daily Return: {avg_all:+.1f}%'), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='#22c55e', dash='dash', width=2),
        name=f'Avg. Gain: {avg_up:+.1f}%'), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='#dc2626', dash='dash', width=2),
        name=f'Avg. Decline: {avg_dn:+.1f}%'), secondary_y=False)

    # ── Stats text (upper left) ──
    fig.add_annotation(
        x=0.01, y=0.98, xref='paper', yref='paper',
        text=(f"Stocks Up: {n_up}<br>Stocks Down: {n_down}"),
        showarrow=False, font=dict(size=14, color='white'),
        bgcolor='rgba(0,0,0,0.0)', align='left',
        xanchor='left', yanchor='top')

    # ── Date annotation (bottom center) ──
    _date_str = pd.Timestamp.now().strftime('%Y/%m/%d')
    fig.add_annotation(
        x=0.5, y=-0.08, xref='paper', yref='paper',
        text=f'Date: {_date_str}',
        showarrow=False, font=dict(size=12, color='white'),
        xanchor='center', yanchor='top')

    title_text = ('1D% Return Distribution For SPX, Individual Constituents')

    # Dynamic x range: pad by 2 or at least ±8%
    _x_lo = min(rets.min() - 2, -8)
    _x_hi = max(rets.max() + 2, 8)

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16, color='white')),
        template=DASH_TEMPLATE,
        height=650,
        margin=dict(l=50, r=50, t=50, b=60),
        xaxis=dict(
            title='', zeroline=False,
            tickformat='.1f', ticksuffix='%',
            gridcolor='rgba(128,128,128,0.2)', gridwidth=0.5,
            range=[_x_lo, _x_hi],
            dtick=2.5,
        ),
        yaxis=dict(showgrid=False, title='', showticklabels=True),
        yaxis2=dict(showgrid=False, title='', showticklabels=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1,
                    font=dict(size=11, color='white'),
                    bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
    )

    chart_widget = go.FigureWidget(fig)

    # ── Build interpretation HTML ──
    interp_html = ''
    if interp_heavy or weights:
        parts = []
        parts.append(
            f"<b style='color:#58a6ff;'>📊 Análise de Distribuição</b><br>"
            f"Total: {n_down + n_up} stocks | ↓ {n_down} caindo ({100*n_down/(n_down+n_up):.0f}%) "
            f"| ↑ {n_up} subindo ({100*n_up/(n_down+n_up):.0f}%)<br>"
            f"Média geral: {avg_all:+.2f}% | Média ganhos: +{avg_up:.2f}% "
            f"| Média perdas: {avg_dn:.2f}%"
        )
        if interp_heavy:
            parts.append(
                "<br><br><b style='color:#fbbf24;'>🔄 Candidatos à Reversão à Média "
                "(Heavy Stocks com Z > 1.5):</b><br>"
                + "<br>".join(interp_heavy)
            )
        interp_html = (
            "<div style='background:#161b22; padding:12px; border-radius:6px; "
            "margin:5px 0; border-left:3px solid #fbbf24;'>"
            + "<br>".join(parts)
            + "</div>"
        )

    return chart_widget, interp_html


def build_straddle_richness_chart(richness_df):
    """Bar chart horizontal: IV percentil de cada ativo (caro vs barato)."""
    if richness_df.empty:
        return wd.HTML('<p style="color:#8b949e;">Sem dados de straddle.</p>')

    df = richness_df.sort_values('Percentil', ascending=True)
    colors = ['#da3633' if p > 75 else '#3fb950' if p < 25 else '#8b949e'
              for p in df['Percentil']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df['Ticker'], x=df['Percentil'],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{p:.0f}th | IV: {iv}" for p, iv in zip(df['Percentil'], df['IV Atual (%)'])],
        textposition='outside',
        hovertemplate='%{y}: IV Pctl %{x:.0f}th<extra></extra>',
    ))

    fig.add_vline(x=50, line=dict(color='#8b949e', dash='dash', width=1))
    fig.add_vline(x=25, line=dict(color=_C['green'], dash='dot', width=1),
                  annotation_text='Barato', annotation_position='top')
    fig.add_vline(x=75, line=dict(color=_C['red'], dash='dot', width=1),
                  annotation_text='Caro', annotation_position='top')

    fig.update_layout(
        title='Straddle ATM — Percentil de IV (Caro vs Barato)',
        template=DASH_TEMPLATE,
        height=max(250, len(df) * 35),
        margin=dict(l=80, r=60, t=45, b=30),
        xaxis_title='Percentil IV (%)',
        xaxis=dict(range=[0, 105]),
    )
    return go.FigureWidget(fig)


def build_dispersion_ml_widget(model_accuracy, feature_importance, disp_prob, features_dict):
    """Widget HTML com output do modelo ML de dispersão."""
    prob_color = _C['green'] if disp_prob > 0.6 else (_C['red'] if disp_prob < 0.4 else _C['yellow'])
    signal = 'LONG DISPERSÃO' if disp_prob > 0.6 else ('SHORT DISPERSÃO' if disp_prob < 0.4 else 'NEUTRO')

    html = (
        f"<div style='background:#161b22; padding:12px; border-radius:6px; margin:5px 0;'>"
        f"<b style='color:#58a6ff; font-size:14px;'>🤖 ML Dispersion Model</b><br>"
        f"<span style='color:#c9d1d9; font-size:13px;'>"
        f"Acurácia CV: <b>{model_accuracy:.1%}</b> │ "
        f"P(dispersão ↑): <b style='color:{prob_color}'>{disp_prob:.1%}</b> │ "
        f"Sinal: <b style='color:{prob_color}'>{signal}</b></span><br>"
        f"<div style='margin-top:6px;'>"
    )

    # Feature importance bars
    if feature_importance:
        sorted_fi = sorted(feature_importance.items(), key=lambda x: -x[1])
        max_fi = max(v for _, v in sorted_fi) if sorted_fi else 1
        for fname, fval in sorted_fi:
            bar_w = int(fval / max_fi * 150)
            current = features_dict.get(fname, 0)
            html += (
                f"<div style='display:flex; align-items:center; margin:2px 0;'>"
                f"<span style='color:#8b949e; font-size:11px; width:160px;'>{fname}</span>"
                f"<div style='background:#58a6ff; height:10px; width:{bar_w}px; "
                f"border-radius:3px; margin:0 8px;'></div>"
                f"<span style='color:#c9d1d9; font-size:11px;'>{current:.4f}</span>"
                f"</div>"
            )
    html += "</div></div>"
    return wd.HTML(html)


def run_dispersion_analysis(index_ticker='SPX Index', lookback=252):
    """
    Executa análise completa de dispersion trade.
    Usa top-N membros por peso (Mag8 + próximos maiores) para correlação mais robusta.
    Retorna dict com todos os resultados.
    """
    result = {
        'error': None,
        'disp_signal': pd.DataFrame(),
        'real_corr': pd.DataFrame(),
        'impl_corr_cboe': pd.Series(dtype=float),
        'mag7_pairs': pd.DataFrame(),
        'best_2x2': [],
        'best_pairs': [],
        'optimal_basket': {},
        'tail_risk': {},
        'index_returns': np.array([]),
        'hyp_test': {},
        'cor1m': pd.Series(dtype=float),
        'dspx': pd.Series(dtype=float),
        'vixeq': pd.Series(dtype=float),
    }

    try:
        prices_df, iv_df, weights = _bql_fetch_member_data(index_ticker, lookback)
    except Exception as e:
        result['error'] = 'Erro ao buscar membros: {}'.format(str(e))
        return result

    # ── Filtrar top-N membros por peso (mais robusto que 500 stocks) ──
    # Excluir tickers indesejados (ex: BRK/B)
    if weights:
        weights = {t: w for t, w in weights.items() if t not in DISP_EXCLUDE}
    if weights and len(weights) > DISP_TOP_N:
        top_tickers = sorted(weights, key=lambda t: -weights[t])[:DISP_TOP_N]
        top_weights = {t: weights[t] for t in top_tickers}
        tw_sum = sum(top_weights.values())
        if tw_sum > 0:
            top_weights = {t: v / tw_sum for t, v in top_weights.items()}
        top_cols = [c for c in top_tickers if c in prices_df.columns]
        prices_top = prices_df[top_cols] if top_cols else prices_df
        iv_top = iv_df[[c for c in top_cols if c in iv_df.columns]] if top_cols else iv_df
        weights_top = top_weights
        print(f"[DISP] Usando top-{len(top_cols)} membros por peso: "
              f"{[t.split(' ')[0] for t in top_cols]}")
    else:
        prices_top = prices_df
        iv_top = iv_df
        weights_top = weights

    try:
        index_iv = _bql_fetch_index_iv(index_ticker, lookback)
    except Exception as e:
        result['error'] = 'Erro ao buscar IV do índice: {}'.format(str(e))
        return result

    try:
        result['real_corr'] = compute_realized_correlation(prices_top)
    except Exception:
        pass

    try:
        impl_corr_ts = compute_implied_corr_series(index_iv, iv_top, weights_top)
        if not impl_corr_ts.empty and not result['real_corr'].empty:
            result['disp_signal'] = compute_dispersion_signal(
                impl_corr_ts, result['real_corr'], window='3M')
    except Exception:
        pass

    try:
        result['impl_corr_cboe'] = _bql_fetch_impl_corr(lookback)
    except Exception:
        pass

    # ── Fetch Bloomberg dispersion indices: COR1M, DSPX, VIXEQ ──
    bq_svc = bql.Service()
    dt_rng = bq_svc.func.range('-{}d'.format(lookback), '0d')
    for tk, key in [(DISP_COR1M, 'cor1m'), (DISP_DSPX, 'dspx'), (DISP_VIXEQ, 'vixeq')]:
        try:
            req = bql.Request(tk, {'px': bq_svc.data.px_last(fill='PREV', dates=dt_rng)})
            result[key] = _bql_ts(bq_svc.execute(req)[0], 'px').dropna()
        except Exception as _idx_err:
            print(f"⚠️ Disp index {tk}: {_idx_err}")

    try:
        result['mag7_pairs'] = compute_mag7_dispersion(prices_df)
    except Exception:
        pass

    try:
        result['best_2x2'] = find_best_2x2_dispersion(prices_df, iv_df)
    except Exception:
        pass

    try:
        idx_prices = prices_top.mean(axis=1)
        sel, wts, te = optimize_tracking_basket(idx_prices, prices_top, n_stocks=min(10, len(prices_top.columns)))
        result['optimal_basket'] = {'tickers': sel, 'weights': wts,
                                    'tracking_error': te}
    except Exception:
        pass

    # ── Hypothesis test: F-test para R² do modelo de dispersão ──
    try:
        if not result['disp_signal'].empty:
            ds = result['disp_signal'].dropna(subset=['impl_corr', 'real_corr'])
            if len(ds) > 30:
                from scipy import stats as sp_stats
                y = ds['real_corr'].values
                x = ds['impl_corr'].values
                slope, intercept, r_val, p_val, std_err = sp_stats.linregress(x, y)
                r2 = r_val ** 2
                n = len(x)
                k = 1  # 1 regressor
                f_stat = (r2 / k) / ((1 - r2) / (n - k - 1)) if r2 < 1 else np.inf
                f_pval = 1 - sp_stats.f.cdf(f_stat, k, n - k - 1)
                result['hyp_test'] = {
                    'R²': round(r2, 4),
                    'R² adj': round(1 - (1 - r2) * (n - 1) / (n - k - 1), 4),
                    'F-stat': round(f_stat, 2),
                    'p-value': f'{f_pval:.2e}',
                    'slope': round(slope, 4),
                    'intercept': round(intercept, 4),
                    'n_obs': n,
                    'significant': p_val < 0.05,
                }
    except Exception:
        pass

    try:
        log_rets = np.log(prices_top.mean(axis=1) /
                          prices_top.mean(axis=1).shift(1)).dropna().values
        result['index_returns'] = log_rets
        result['tail_risk'] = compute_tail_risk(log_rets)
    except Exception:
        pass

    # ── Multi-window correlation matrices ──
    try:
        corr_matrices = compute_multi_window_correlations(prices_top)
        result['corr_matrices'] = corr_matrices

        # Latest IV for pair scoring
        iv_latest = {}
        if iv_top is not None and not iv_top.empty:
            for col in iv_top.columns:
                last_val = iv_top[col].dropna()
                if len(last_val) > 0:
                    iv_latest[col] = float(last_val.iloc[-1])
                    if iv_latest[col] > 1:
                        iv_latest[col] /= 100.0

        result['dispersion_pairs'] = find_dispersion_pairs(corr_matrices, iv_latest)
        print(f"[DISP] {len(result['dispersion_pairs'])} dispersion pairs identified")
    except Exception as _mwc_err:
        print(f"⚠️ Multi-window corr: {_mwc_err}")
        result['corr_matrices'] = {}
        result['dispersion_pairs'] = pd.DataFrame()

    # ── Straddle pricing (Mag8 + SPX) ──
    # Always include all MAG8 + index — do NOT filter by prices_df.columns
    straddle_tickers = list(MAG8) + [index_ticker]
    try:
        result['straddle_data'] = fetch_straddle_prices(straddle_tickers)
        print(f"[DISP] Straddle prices for {len(result['straddle_data'])} tickers")
        # Fallback: for any MAG8 ticker missing, try BQL implied_volatility directly
        _bq_sd = bql.Service()
        for _mag_tk in MAG8:
            if _mag_tk not in result['straddle_data']:
                try:
                    _iv_req = bql.Request(_mag_tk, {
                        'atm_iv': _bq_sd.data.implied_volatility(
                            expiry='30D', pct_moneyness='100'),
                        'px': _bq_sd.data.px_last(),
                    })
                    _iv_resp = _bq_sd.execute(_iv_req)
                    _iv_df = pd.concat([r.df() for r in _iv_resp], axis=1).reset_index()
                    _atm = float(_iv_df.iloc[0]['atm_iv'])
                    _px = float(_iv_df.iloc[0]['px'])
                    if not np.isnan(_atm) and not np.isnan(_px):
                        result['straddle_data'][_mag_tk] = {
                            'spot': _px, 'strike': _px, 'expiry': '30D',
                            'call_iv': _atm, 'put_iv': _atm,
                            'straddle_iv': _atm,
                            'call_mid': 0, 'put_mid': 0,
                            'straddle_px': 0, 'straddle_pct': 0,
                            'c25_iv': np.nan, 'p25_iv': np.nan,
                            'strangle_iv': np.nan, 'strangle_px': 0,
                            'strangle_pct': 0,
                        }
                        print(f"[DISP] Fallback IV for {_mag_tk}: {_atm:.4f}")
                except Exception:
                    pass
    except Exception as _strd_err:
        print(f"⚠️ Straddle prices: {_strd_err}")
        result['straddle_data'] = {}

    # ── Best pair combos (1-pair, 2-pair, 3-pair) ──
    try:
        result['best_pairs'] = find_best_pair_combos(
            prices_df, iv_df,
            straddle_data=result.get('straddle_data'),
            max_pairs=3)
    except Exception:
        result['best_pairs'] = []

    # ── Historical IV for richness ──
    try:
        iv_hist = fetch_historical_straddle_iv(straddle_tickers, lookback=lookback)
        # RV from prices for IV-RV spread
        rv_df = pd.DataFrame()
        for tk in straddle_tickers:
            if tk in prices_df.columns:
                p = prices_df[tk].dropna()
                lr = np.log(p / p.shift(1))
                rv_df[tk] = lr.rolling(21).std() * np.sqrt(252)

        result['straddle_richness'] = compute_straddle_richness(
            result['straddle_data'], iv_hist, rv_df)
        print(f"[DISP] Richness computed for {len(result['straddle_richness'])} tickers")
    except Exception as _rich_err:
        print(f"⚠️ Straddle richness: {_rich_err}")
        result['straddle_richness'] = pd.DataFrame()

    # ── Trade recommendations ──
    try:
        recs_df, recs_interp = build_dispersion_trade_recommendations(
            result.get('dispersion_pairs', pd.DataFrame()),
            result.get('straddle_richness', pd.DataFrame()),
            result.get('straddle_data', {}))
        result['trade_recs'] = recs_df
        result['trade_interp'] = recs_interp
    except Exception:
        result['trade_recs'] = pd.DataFrame()
        result['trade_interp'] = ''

    # ── ML Dispersion Model ──
    try:
        model, feat_names, accuracy, feat_imp = train_dispersion_model(
            prices_top, iv_top, lookback=min(lookback, 126))
        if model is not None:
            disp_prob, feat_dict = predict_dispersion(model, prices_top, iv_top)
            result['ml_model'] = {
                'accuracy': accuracy,
                'feature_importance': feat_imp,
                'disp_prob': disp_prob,
                'features': feat_dict,
            }
            print(f"[DISP] ML model: accuracy={accuracy:.1%}, P(disp↑)={disp_prob:.1%}")
    except Exception as _ml_err:
        print(f"⚠️ ML Dispersion: {_ml_err}")

    # ── KDE distribution data ──
    result['prices_df'] = prices_df
    result['weights'] = weights

    # ── ATM vol matrix (from straddle data already fetched) ──
    if result.get('straddle_data'):
        try:
            atm_matrix_df, atm_matrix_chart = build_atm_vol_matrix(result['straddle_data'])
            result['atm_vol_matrix'] = atm_matrix_df
            result['atm_vol_chart'] = atm_matrix_chart
        except Exception as _atm_err:
            print(f"⚠️ ATM Vol Matrix: {_atm_err}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5G-B — BREADTH, CORRELATION, 0DTE, BLACKOUT CHARTS
# ═══════════════════════════════════════════════════════════════════════════════


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
                                spot, skew, put_wall, call_wall):
    """
    Calcula o Gamma Squeeze Score (0–100).

    Score alto = condições favoráveis para um short squeeze induzido por gamma:
    - Dealers SHORT gamma (GEX muito negativo) → compram quando mercado sobe
    - Posicionamento bearish excessivo (P/C ratio alto, skew alto)
    - Mercado próximo ao gamma flip → qualquer alta cruza o flip e acelera
    - Vol realizada bem abaixo da implícita (mais espaço para rali surpresa)

    Retorna dict com score total, sub-scores, interpretação e nível de alerta.
    """
    components = {}

    # 1. GEX negativity (0–30): quanto mais negativo, maior o score
    # Normalizado por -20Bn como referência de mercado muito short gamma
    _gex_score = min(30, max(0, abs(min(0, net_gex_bn)) / 20.0 * 30))
    components['gex'] = {
        'label': 'GEX Negatividade',
        'value': f'{net_gex_bn:+.2f}B',
        'score': round(_gex_score, 1),
        'max': 30,
        'desc': 'Dealers SHORT gamma → amplificam movimentos de alta',
    }

    # 2. Put/Call ratio (0–25): >1.5 = bearish extremo
    _pc_score = min(25, max(0, (pc_ratio - 1.0) / 1.5 * 25)) if pc_ratio > 1.0 else 0
    components['pc_ratio'] = {
        'label': 'P/C OI Ratio',
        'value': f'{pc_ratio:.2f}x',
        'score': round(_pc_score, 1),
        'max': 25,
        'desc': 'Posicionamento muito pessimista → squeeze mais intenso se mercado subir',
    }

    # 3. Proximidade ao gamma flip (0–25): quanto mais perto, mais perigoso
    _flip_dist = abs(gamma_flip - spot) / spot if (gamma_flip and spot) else 0.05
    _flip_score = min(25, max(0, (1 - _flip_dist / 0.05) * 25))  # max score se dist < 0.5%
    _flip_side = 'ACIMA' if (gamma_flip and gamma_flip > spot) else 'ABAIXO'
    components['flip_proximity'] = {
        'label': 'Distância Gamma Flip',
        'value': f'{gamma_flip:,.0f} ({_flip_dist:.1%} do spot) — Flip {_flip_side}',
        'score': round(_flip_score, 1),
        'max': 25,
        'desc': 'Próximo ao flip → qualquer rali pode cruzar e virar auto-reforçado',
    }

    # 4. Vol premium invertida (0–20): IV muito > RV = mercado pagando por proteção
    # Mas para squeeze, queremos IV alta (medo) + mercado que ainda não se moveu
    _vol_gap = (iv_30d - rv_30d) if (pd.notna(iv_30d) and pd.notna(rv_30d)) else 0
    _vol_score = min(20, max(0, _vol_gap * 100 * 4))  # 5 vol pts gap → score 20
    components['vol_premium'] = {
        'label': 'Prêmio de Vol (IV−RV)',
        'value': f'{_vol_gap*100:+.1f} vol pts',
        'score': round(_vol_score, 1),
        'max': 20,
        'desc': 'IV > RV = medo embutido → mais proteção para ser revertida',
    }

    total = sum(c['score'] for c in components.values())

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

    # Estimativa de magnitude do squeeze se o flip for cruzado
    _squeeze_mag = abs(net_gex_bn) * 0.15 * (pc_ratio / 1.5)  # rough: 15% do GEX por 1% de alta
    _squeeze_mag_pct = min(15, _squeeze_mag / (spot * 0.01))   # % de alta esperado

    return {
        'score': round(total, 1),
        'alert': alert,
        'interp': interp,
        'components': components,
        'squeeze_mag_pct': round(_squeeze_mag_pct, 1),
        'flip_above': gamma_flip is not None and gamma_flip > spot,
        'flip_dist_pct': round(_flip_dist * 100, 2),
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
        _th_s = ''.join(f"<th style='padding:2px 6px;font-size:9px;color:rgba(0,212,232,.5);text-align:center;'>{int(k)}</th>"
                        for k in _hm_strikes)
        _hm_rows = ''
        for exp, row in _surf_piv.iterrows():
            _exp_lbl = pd.Timestamp(exp).strftime('%d/%b')
            _hm_rows += f"<tr><td style='padding:2px 6px;font-size:9px;color:rgba(255,255,255,.5);white-space:nowrap;'>{_exp_lbl}</td>"
            for k in _hm_strikes:
                v = row.get(k, np.nan)
                bg = _vol_color(v) if pd.notna(v) else 'transparent'
                txt = f'{v:.1f}' if pd.notna(v) else ''
                _hm_rows += (f"<td style='padding:1px 5px;font-size:9px;text-align:center;"
                             f"background:{bg};color:rgba(0,0,0,.7);font-weight:600;'>{txt}</td>")
            _hm_rows += '</tr>'

        _hm_html = (
            "<div style='margin:12px 0 6px;'>"
            "<p style='color:rgba(0,212,232,.7);font-size:11px;margin:0 0 6px;letter-spacing:.5px;'>"
            "VOL SURFACE — IV% base por expiry × strike bucket (média por faixa)</p>"
            "<div style='overflow-x:auto;'><table style='border-collapse:collapse;font-family:monospace;'>"
            f"<thead><tr><th style='padding:2px 6px;font-size:9px;'></th>{_th_s}</tr></thead>"
            f"<tbody>{_hm_rows}</tbody></table></div>"
            "<p style='color:rgba(255,255,255,.25);font-size:9px;margin:4px 0 0;'>"
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
                      iv_30d, rv_30d, pc_ratio, _C):
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
        xaxis=dict(range=[0, 30], title='Score'),
        height=220, template='plotly_dark',
        margin=dict(t=35, b=20, l=5, r=60),
        paper_bgcolor=_C['card'], plot_bgcolor=_C['card'], showlegend=False)

    # ── Resumo textual ──
    _sq_mag = squeeze_result['squeeze_mag_pct']
    _fd = squeeze_result['flip_dist_pct']
    _flip_dir = 'ACIMA' if squeeze_result['flip_above'] else 'ABAIXO'
    summary_html = (
        f"<div class='mm-dash'><div class='mm-card'>"
        f"<h3 style='color:{alert_color}'>Gamma Squeeze Score: {score:.0f}/100</h3>"
        f"<p><b style='color:{alert_color}'>{interp}</b></p>"
        f"<table class='mm-table' style='width:auto;font-size:12px;'>"
        f"<tr><td>GEX NET</td><td><b>{net_gex_bn:+.2f}B</b></td></tr>"
        f"<tr><td>Gamma Flip</td><td><b>{gamma_flip:,.0f}</b> ({_fd:.1f}% {_flip_dir} do spot)</td></tr>"
        f"<tr><td>P/C OI Ratio</td><td><b>{pc_ratio:.2f}x</b></td></tr>"
        f"<tr><td>IV−RV Gap</td><td><b>{(iv_30d-rv_30d)*100:+.1f} vol pts</b></td></tr>"
        f"<tr><td>Magnitude estimada</td><td><b>~{_sq_mag:.1f}%</b> se flip cruzado</td></tr>"
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

    children = [
        wd.HTML(summary_html),
        wd.HBox([gauge_fig, bar_fig],
                layout={'align_items': 'flex-start'}),
        wd.HTML(_evts_html),
    ]
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 14 — DECISION ENGINE (inline loader, no external import)
# ══════════════════════════════════════════════════════════════════════════════

def _build_decision_engine_tab_inline(df, spot, rfr, ticker, external_scores=None):
    """
    Loads decision_engine.py via importlib (avoids ModuleNotFoundError in
    BQuant kernel where sys.path doesn't include the project directory).
    Falls back to a minimal informational tab if the file cannot be found.
    """
    import importlib.util as _ilu
    import sys as _sys
    import os as _os
    import json as _json
    import traceback as _tb

    ext = external_scores or {}

    # ── Step 1: try to get already-loaded module ──────────────────────────────
    _de_mod = _sys.modules.get('decision_engine', None)

    # ── Step 2: search for decision_engine.py ────────────────────────────────
    if _de_mod is None:
        _de_candidates = []
        # Same directory as a known absolute path (BBG local)
        for _base in [
            _os.path.expanduser('~/bbg/examples'),
            _os.path.expanduser('~/examples'),
            '/bbg/examples',
            '/home/user/bbg/examples',
            '/home/user/examples',
            _os.getcwd(),
            _os.path.join(_os.getcwd(), '..'),
            _os.path.join(_os.getcwd(), '..', 'examples'),
            # Windows path (for local dev)
            r'C:\Users\rafael bentes\bbg\examples',
            '/c/Users/rafael bentes/bbg/examples',
        ]:
            _p = _os.path.join(_base, 'decision_engine.py')
            if _os.path.isfile(_p) and _p not in _de_candidates:
                _de_candidates.append(_p)

        for _path in _de_candidates:
            try:
                _spec = _ilu.spec_from_file_location('decision_engine', _path)
                _mod  = _ilu.module_from_spec(_spec)
                _sys.modules['decision_engine'] = _mod
                _spec.loader.exec_module(_mod)
                _de_mod = _mod
                print(f"✓ decision_engine carregado de: {_path}")
                break
            except Exception as _load_err:
                _sys.modules.pop('decision_engine', None)
                print(f"⚠ decision_engine load failed ({_path}): {_load_err}")
                continue

    # ── Step 3: delegate to build_decision_engine_tab ────────────────────────
    if _de_mod is not None:
        try:
            return _de_mod.build_decision_engine_tab(df, spot, rfr, ticker, ext)
        except Exception as _render_err:
            _err_html = (
                "<div style='background:#0d1520;border:1px solid #f85149;"
                "border-radius:8px;padding:16px;font-family:monospace;'>"
                "<h3 style='color:#f85149;'>Decision Engine — erro ao renderizar</h3>"
                f"<pre style='font-size:10px;color:#aaa;white-space:pre-wrap;'>"
                f"{_tb.format_exc()}</pre></div>"
            )
            return wd.VBox([wd.HTML(_err_html)])

    # ── Step 4: fallback tab when file not found ──────────────────────────────
    _scores_html = ''.join(
        f"<tr><td style='color:#aaa;padding:4px 12px 4px 0;'>{k}</td>"
        f"<td style='color:#00d4e8;font-weight:bold;'>{v:.1f}</td></tr>"
        for k, v in ext.items()
    )
    _fallback_html = f"""
<div style='background:#0d1520;border:1px solid rgba(0,212,232,.25);
            border-radius:8px;padding:20px;font-family:monospace;max-width:700px;'>
  <h3 style='color:#00d4e8;margin:0 0 8px;'>Decision Engine — 0DTE Intraday</h3>
  <p style='color:#f85149;margin:0 0 12px;'>
    ⚠ <b>decision_engine.py não encontrado</b> no path do kernel BQuant.
  </p>
  <p style='color:#aaa;font-size:11px;margin:0 0 8px;'>
    Verifique que <code>decision_engine.py</code> está na mesma pasta que
    <code>greeks_dashboard.py</code> e que o kernel tem acesso ao diretório.
  </p>
  <p style='color:rgba(0,212,232,.6);font-size:10px;letter-spacing:.8px;
             margin:12px 0 4px;'>SCORES EXTERNOS RECEBIDOS</p>
  <table style='font-size:11px;border-collapse:collapse;'>
    {_scores_html}
  </table>
  <p style='color:#aaa;font-size:10px;margin:16px 0 0;'>
    Paths tentados: {", ".join(_de_candidates) if "_de_candidates" in dir() else "nenhum"}
  </p>
</div>"""
    return wd.VBox([wd.HTML(_fallback_html)])


def build_greek_overview(greeks_now, df, spot, etf_flows=None):
    """
    Seção de termômetros das gregas + fluxo por ação (Mag8) para a Visão Geral.
    Gregas em $Bn usando a mesma escala das abas de exposição (GREEK_CONFIGS).
    etf_flows: dict de DataFrames com Flow_$ e PctADV por ação (compute_full_etf_flows).
    """
    oi_100  = df['OI'].values * 100.0
    is_call = df['Type'].values == 'Call'
    is_put  = df['Type'].values == 'Put'

    # ── Gregas em $Bn — mesma escala do GREEK_CONFIGS ─────────────
    # Delta:  op=add,      scale=L (spot)
    delta_bn = float(np.nansum(greeks_now['delta'] * oi_100) * spot / 1e9)
    # Gamma:  op=subtract, scale=L²×0.01
    gamma_bn = float((np.nansum(greeks_now['gamma'][is_call] * oi_100[is_call]) -
                      np.nansum(greeks_now['gamma'][is_put]  * oi_100[is_put])) * spot**2 * 0.01 / 1e9)
    # Vanna:  op=subtract, scale=1  (sem spot)
    vanna_bn = float((np.nansum(greeks_now['vanna'][is_call] * oi_100[is_call]) -
                      np.nansum(greeks_now['vanna'][is_put]  * oi_100[is_put])) / 1e9)
    # Charm:  op=add,      scale=L/365
    charm_bn = float(np.nansum(greeks_now['charm'] * oi_100) * spot / 365.0 / 1e9)

    # Cache module-level para exportação JARVIS (div/10 = escala BBG)
    _greek_cache['delta_bn'] = delta_bn / 10
    _greek_cache['vanna_bn'] = vanna_bn
    _greek_cache['charm_bn'] = charm_bn / 10

    # Escala dinâmica mínima por grega (SPX típico)
    g_delta = create_symmetric_gauge(delta_bn, 'Δ Delta Nocional',  max(5.0,  abs(delta_bn) * 1.5))
    g_gamma = create_symmetric_gauge(gamma_bn, 'Γ Gamma (GEX Net)', max(0.5,  abs(gamma_bn) * 1.5))
    g_vanna = create_symmetric_gauge(vanna_bn, 'V Vanna',           max(2.0,  abs(vanna_bn) * 1.5))
    g_charm = create_symmetric_gauge(charm_bn, 'C Charm (diário)',  max(0.5,  abs(charm_bn) * 1.5))

    # ── Interpretação textual ──────────────────────────────────────
    def _badge(positive, txt_pos, txt_neg, val, thr=0.1):
        if abs(val) < thr:
            return f"<span style='color:#8b949e;'>⚪ Neutro</span>"
        color = _C['green'] if (val > 0) == positive else _C['red']
        txt   = txt_pos if val > 0 else txt_neg
        return f"<span style='color:{color};'>{txt}</span>"

    interp_html = (
        f"<div class='mm-dash'><div class='mm-card' style='padding:10px 16px;min-width:280px;'>"
        f"<div class='mm-section-label' style='margin-top:0;'>Leitura das Gregas</div>"
        f"<p style='margin:4px 0;font-size:12px;'><b>Δ Delta:</b> "
        + _badge(False, '🔴 Dealers comprados → venda', '🟢 Dealers vendidos → compra', delta_bn, 0.5)
        + f"</p><p style='margin:4px 0;font-size:12px;'><b>Γ Gamma:</b> "
        + _badge(True, '🟢 GEX+ estabiliza mercado', '🔴 GEX− acelera movimentos', gamma_bn, 0.3)
        + f"</p><p style='margin:4px 0;font-size:12px;'><b>V Vanna:</b> "
        + _badge(False, '🔴 Vol↑ → dealers vendem', '🟢 Vol↑ → dealers compram', vanna_bn, 0.2)
        + f"</p><p style='margin:4px 0;font-size:12px;'><b>C Charm:</b> "
        + _badge(False, '🔴 Delta decai → desfaz hedge', '🟢 Delta decai → reforça hedge', charm_bn, 0.05)
        + "</p></div></div>"
    )

    row_gauges = wd.HBox(
        [g_delta, g_gamma, g_vanna, g_charm, wd.HTML(interp_html)],
        layout={'justify_content': 'flex-start', 'align_items': 'center', 'flex_wrap': 'wrap'})

    # ── Fluxo por ação — dados reais do ETF rebalanceamento ────────
    combo = pd.DataFrame()
    if etf_flows and 'Combined' in etf_flows:
        combo = etf_flows['Combined'].copy()
        # Normalizar ticker: 'AAPL US Equity' → 'AAPL'
        combo.index = combo.index.str.split().str[0]

    def _stock_bar(name, val_bn, pct_adv, flow_max_bn):
        pct_bar = min(abs(val_bn) / flow_max_bn * 100, 100) if flow_max_bn > 0 else 0
        color   = _C['green'] if val_bn >= 0 else _C['red']
        arrow   = '▲' if val_bn >= 0 else '▼'
        adv_str = f' | {pct_adv:.1f}% ADV' if pd.notna(pct_adv) else ''
        return (
            f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>"
            f"<span style='font-size:12px;font-weight:700;color:{_C['text']};width:48px;'>{name}</span>"
            f"<div style='flex:1;background:{_C['card2']};border-radius:3px;height:14px;'>"
            f"<div style='width:{pct_bar:.0f}%;height:100%;background:{color};"
            f"border-radius:3px;opacity:0.75;'></div></div>"
            f"<span style='font-size:11px;color:{color};min-width:110px;text-align:right;'>"
            f"{arrow} ${val_bn:.2f}Bn{adv_str}</span>"
            f"</div>"
        )

    if not combo.empty and 'Flow_$' in combo.columns:
        combo['Flow_Bn'] = combo['Flow_$'] / 1e9
        buys_df  = combo[combo['Flow_Bn'] >= 0].nlargest(4, 'Flow_Bn')
        sells_df = combo[combo['Flow_Bn'] <  0].nsmallest(4, 'Flow_Bn')
        flow_max = max(combo['Flow_Bn'].abs().max(), 1.0)

        buy_rows  = ''.join(_stock_bar(s, row['Flow_Bn'],
                            row.get('PctADV', np.nan), flow_max)
                            for s, row in buys_df.iterrows())
        sell_rows = ''.join(_stock_bar(s, row['Flow_Bn'],
                            row.get('PctADV', np.nan), flow_max)
                            for s, row in sells_df.iterrows())
        data_note = 'Dados reais do rebalanceamento de ETFs passivos (VOO/SPY/IVV).'

        # ── Overhang — mesmos dados das barras, top 2 por %ADV ───
        # Usa buys_df/sells_df (já filtrados e normalizados) para
        # garantir que os valores batam exatamente com as barras acima.
        overhang_section = wd.HTML('')
        if 'PctADV' in combo.columns:
            # Limpa inf/nan nas colunas PctADV das mesmas tabelas usadas acima
            buys_adv  = buys_df[buys_df['PctADV'].replace([np.inf,-np.inf],np.nan).notna()].copy()
            sells_adv = sells_df[sells_df['PctADV'].replace([np.inf,-np.inf],np.nan).notna()].copy()

            top2_buy  = buys_adv.nlargest(2, 'PctADV')   # maior %ADV positivo
            top2_sell = sells_adv.nsmallest(2, 'PctADV') # menor %ADV (mais negativo)

            all_adv_vals = (list(top2_buy['PctADV']) + list(top2_sell['PctADV']))
            adv_scale = max((abs(v) for v in all_adv_vals if pd.notna(v)), default=5.0) * 1.15

            overhang_gauges = []
            for tkr, row in list(top2_buy.iterrows()) + list(top2_sell.iterrows()):
                pct = float(row['PctADV'])
                overhang_gauges.append(
                    create_symmetric_gauge(
                        round(pct, 2), tkr,
                        adv_scale, unit='% ADV',
                        width=200, height=178))

            overhang_row   = wd.HBox(overhang_gauges,
                                     layout={'justify_content': 'flex-start',
                                             'flex_wrap': 'wrap'})
            overhang_title = wd.HTML(
                f"<div class='mm-section-label' style='margin:10px 0 4px;padding:0 8px;'>"
                f"Overhang — Impacto do Rebalanceamento (% ADV)</div>"
                f"<div style='font-size:11px;color:{_C['text_dim']};padding:0 8px 6px;'>"
                f"Top 2 maiores compra e Top 2 maiores venda — "
                f"quanto o flow representa do volume médio diário do ativo (5d ADV). "
                f"Mesmo dado das barras acima (BQL).</div>")
            overhang_section = wd.VBox([overhang_title, overhang_row])
    else:
        # Fallback sem dados ETF
        buy_rows = sell_rows = "<p style='color:#8b949e;font-size:12px;'>Dados ETF não disponíveis.</p>"
        data_note = 'Execute com ETF rebalancing ativo para ver dados reais.'
        overhang_section = wd.HTML('')

    flow_html = (
        f"<div class='mm-dash'><div class='mm-card' style='padding:12px 16px;'>"
        f"<div class='mm-section-label' style='margin:0 0 8px;'>"
        f"Rebalanceamento ETF Passivo (VOO / SPY / IVV) — Top 4 Compra + Top 4 Venda</div>"
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:20px;'>"
        f"<div>"
        f"<div style='font-size:11px;font-weight:700;color:{_C['green']};margin-bottom:6px;"
        f"text-transform:uppercase;letter-spacing:0.8px;'>▲ Fluxo de Compra</div>"
        f"{buy_rows}</div>"
        f"<div>"
        f"<div style='font-size:11px;font-weight:700;color:{_C['red']};margin-bottom:6px;"
        f"text-transform:uppercase;letter-spacing:0.8px;'>▼ Fluxo de Venda</div>"
        f"{sell_rows}</div>"
        f"</div>"
        f"<p style='font-size:10px;color:{_C['text_dim']};margin:8px 0 0;'>{data_note}</p>"
        f"</div></div>"
    )

    return wd.VBox([
        row_gauges,
        wd.HBox([wd.HTML(flow_html), overhang_section],
                layout={'align_items': 'flex-start', 'flex_wrap': 'wrap'}),
    ])


def plot_exposure_charts(agg, df, spot, from_strike, to_strike,
                         levels, model_curves, flip_points,
                         call_wall, put_wall):
    """Gera todos os gráficos de exposição (Combo + Absolute + Model) para cada grega."""
    strikes = agg.index.values
    step = np.median(np.diff(np.sort(strikes))) if len(strikes) > 1 else 5
    bar_w = step * 0.8

    for cfg in GREEK_CONFIGS:
        name = cfg['name']
        key = cfg['key']
        unit = cfg['unit']
        div = cfg['div']
        scale_val = cfg['scale'](spot)
        combo_op = cfg['op']
        flip = flip_points.get(name)

        call_col = f'Call_{key}'
        put_col = f'Put_{key}'
        call_data = agg[call_col] * scale_val
        put_data = agg[put_col] * scale_val
        combo_data = combo_op(call_data, put_data) / div
        total_val = combo_data.sum()

        # ── Gráfico COMBO ──
        plt.figure(figsize=(18, 5))
        colors = [_C['green'] if v >= 0 else _C['red'] for v in combo_data.values]
        plt.bar(strikes, combo_data, width=bar_w, color=colors, edgecolor='k', lw=0.3)
        plt.axvline(spot, color='r', ls='--', lw=1.5, label=f'Spot {spot:,.0f}')
        if name == 'Gamma':
            if call_wall:
                plt.axvline(call_wall, color='g', ls=':', lw=2, label=f'Call Wall {call_wall:,.0f}')
            if put_wall:
                plt.axvline(put_wall, color='b', ls=':', lw=2, label=f'Put Wall {put_wall:,.0f}')
        if flip:
            plt.axvline(flip, color='orange', ls='--', lw=2.5, label=f'{name} Flip {flip:,.0f}')
        plt.title(f'COMBO {name.upper()} EXPOSURE = {total_val:,.2f} ({unit})',
                  fontsize=16, weight='bold')
        plt.ylabel(f'Exposure ({unit})', fontsize=12)
        plt.xlim(from_strike, to_strike)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # ── Gráfico ABSOLUTE ──
        put_sign = 1 if combo_op is np.add else -1
        plt.figure(figsize=(18, 5))
        plt.bar(strikes, call_data / div, width=bar_w, color=_C['green'],
                edgecolor='k', lw=0.3, label=f'Call {name}')
        plt.bar(strikes, put_sign * put_data / div, width=bar_w, color=_C['red'],
                edgecolor='k', lw=0.3, label=f'Put {name}')
        plt.axvline(spot, color='r', ls='--', lw=1.5, label=f'Spot {spot:,.0f}')
        if name == 'Gamma':
            if call_wall:
                plt.axvline(call_wall, color='g', ls=':', lw=2, label=f'Call Wall {call_wall:,.0f}')
            if put_wall:
                plt.axvline(put_wall, color='b', ls=':', lw=2, label=f'Put Wall {put_wall:,.0f}')
        if flip:
            plt.axvline(flip, color='orange', ls='--', lw=2.5, label=f'{name} Flip {flip:,.0f}')
        plt.title(f'ABSOLUTE {name.upper()} EXPOSURE', fontsize=16, weight='bold')
        plt.ylabel(f'Exposure ({unit})', fontsize=12)
        plt.xlim(from_strike, to_strike)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # ── Gráfico MODEL ──
        curve = model_curves[name] / div
        current_val = curve[np.argmin(np.abs(levels - spot))]
        plt.figure(figsize=(18, 5))
        plt.plot(levels, curve, lw=2, color=_C['accent'])
        plt.fill_between(levels, 0, curve, where=curve >= 0,
                         alpha=0.15, color='green', interpolate=True)
        plt.fill_between(levels, 0, curve, where=curve <= 0,
                         alpha=0.15, color='red', interpolate=True)
        plt.axvline(spot, color='r', ls='--', lw=1.5, label=f'Spot {spot:,.0f}')
        if flip:
            plt.axvline(flip, color='orange', ls='--', lw=2.5, label=f'{name} Flip {flip:,.0f}')
        plt.axhline(0, color='k', lw=0.5)
        plt.title(f'{name.upper()} EXPOSURE MODEL = {current_val:,.2f} ({unit})',
                  fontsize=16, weight='bold')
        plt.xlabel('Preço do Ativo', fontsize=12)
        plt.ylabel(f'Exposure ({unit})', fontsize=12)
        plt.xlim(from_strike, to_strike)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def style_sensitivity_matrix(matrix_df, cmap='viridis'):
    """Aplica estilo visual a uma matriz de sensibilidade."""
    styled = matrix_df.style.format(fmt_value)
    styled = styled.set_properties(**{
        'width': '110px', 'text-align': 'center', 'font-size': '12px',
        'padding': '6px 8px', 'border': f'1px solid {_C["border_light"]}',
        'color': _C['text'], 'background-color': _C['card'],
    })
    styled = styled.set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', _C['card2']), ('color', _C['text_muted']),
            ('font-size', '11px'), ('text-transform', 'uppercase'),
            ('letter-spacing', '0.5px'), ('padding', '8px'),
            ('border', f'1px solid {_C["border"]}')]},
    ])
    numeric = matrix_df.select_dtypes(include=np.number).dropna(how='all')
    if not numeric.empty and numeric.max().max() != numeric.min().min():
        styled = styled.background_gradient(cmap=cmap, axis=None)
    return styled.to_html()


_JARVIS_EXPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>J.A.R.V.I.S — Intelligence Core</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#020810;
  --c:#00d4e8;              /* cyan — primary/neutral */
  --c60:rgba(0,212,232,.88);
  --c40:rgba(0,212,232,.72);
  --c20:rgba(0,212,232,.52);
  --c08:rgba(0,212,232,.16);
  --c04:rgba(0,212,232,.08);
  --brd:rgba(0,212,232,.25);
  --dim:#041420;
  --txt:rgba(0,212,232,.92);
  --lbl:rgba(0,212,232,.62);
  --a:#f5a623;              /* amber — warning/caution */
  --a60:rgba(245,166,35,.88);
  --a40:rgba(245,166,35,.72);
  --a20:rgba(245,166,35,.5);
  --a08:rgba(245,166,35,.16);
  --r:#f85149;              /* red — danger/bearish */
  --r60:rgba(248,81,73,.88);
  --r40:rgba(248,81,73,.72);
  --r20:rgba(248,81,73,.5);
  --r08:rgba(248,81,73,.16);
}
html,body{width:100%;height:100vh;overflow:hidden;background:var(--bg);
  font-family:'Share Tech Mono',monospace;color:var(--txt)}
canvas#bg{position:fixed;inset:0;z-index:0;opacity:.5}
.scl{position:fixed;inset:0;z-index:2;pointer-events:none;
  background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,.04) 3px,rgba(0,0,0,.04) 4px)}

/* ── BOOT ── */
#boot{position:fixed;inset:0;z-index:999;background:var(--bg);
  display:flex;flex-direction:column;align-items:center;justify-content:center;transition:opacity .7s}
#boot.gone{opacity:0;pointer-events:none}
.brw{position:relative;width:68px;height:68px;display:flex;align-items:center;justify-content:center;margin-bottom:20px}
.bri{position:absolute;border-radius:50%;border:1px solid transparent}
.bri1{inset:0;border-top-color:var(--c);border-right-color:var(--c);animation:sp 1s linear infinite}
.bri2{inset:10px;border-bottom-color:var(--c40);border-left-color:var(--c40);animation:sp 1.8s linear reverse infinite}
.bri3{inset:20px;border-top-color:var(--c20);animation:sp 3s linear infinite}
.bcore{width:8px;height:8px;border-radius:50%;background:var(--c);box-shadow:0 0 10px var(--c),0 0 20px var(--c)}
.btitle{font-family:'Orbitron',sans-serif;font-size:9px;letter-spacing:6px;color:var(--c);margin-bottom:14px}
#blog{font-size:9px;color:var(--c40);line-height:2.2;text-align:center;min-height:60px}
@keyframes sp{to{transform:rotate(360deg)}}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.1}}

/* ── APP ── */
#app{position:relative;z-index:10;width:100vw;height:100vh;
  display:flex;flex-direction:column;opacity:0;transition:opacity .8s}
#app.on{opacity:1}

/* ── CMD STRIP ── */
.cmd{display:flex;align-items:center;gap:8px;padding:6px 24px;
  background:rgba(0,4,10,.98);border-bottom:1px solid var(--c08);flex-shrink:0}
.cmd-id{font-family:'Orbitron',sans-serif;font-size:10px;font-weight:700;letter-spacing:3px;
  color:var(--c);display:flex;align-items:center;gap:6px;white-space:nowrap;margin-right:6px}
.cmd-id::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--c);
  box-shadow:0 0 8px var(--c);animation:blink 1.4s ease-in-out infinite;flex-shrink:0}
.dvd{width:1px;height:18px;background:var(--c08);flex-shrink:0;margin:0 6px}
.cs{display:flex;flex-direction:column;gap:1px;flex-shrink:0}
.csl{font-size:9px;letter-spacing:2px;color:rgba(0,200,220,.65)}
.csv{font-family:'Orbitron',sans-serif;font-size:13px;font-weight:700;color:var(--c)}

/* ── HEADER ── */
.hdr{display:flex;align-items:center;padding:8px 24px;gap:16px;
  background:linear-gradient(180deg,rgba(0,8,18,.98),rgba(0,4,10,.9));
  border-bottom:1px solid var(--c08);flex-shrink:0;position:relative}
.hdr::after{content:'';position:absolute;bottom:0;left:0;width:15%;height:1px;
  background:linear-gradient(90deg,transparent,var(--c60),transparent);animation:hln 7s linear infinite}
@keyframes hln{0%{left:-15%}100%{left:115%}}
.reactor{width:30px;height:30px;position:relative;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.ri{position:absolute;border-radius:50%;border:1px solid transparent}
.ri1{inset:0;border-top-color:var(--c);border-right-color:var(--c);animation:sp 2s linear infinite}
.ri2{inset:5px;border-bottom-color:var(--c40);border-left-color:var(--c40);animation:sp 3.5s linear reverse infinite}
.ri3{inset:11px;border-top-color:var(--c20);animation:sp 5s linear infinite}
.rcore{width:6px;height:6px;border-radius:50%;background:var(--c);
  box-shadow:0 0 6px var(--c),0 0 14px var(--c);animation:rpc 2.2s ease-in-out infinite}
@keyframes rpc{0%,100%{box-shadow:0 0 6px var(--c),0 0 12px var(--c)}
  50%{box-shadow:0 0 10px var(--c),0 0 22px var(--c),0 0 36px var(--c20)}}
.brand{flex-shrink:0}
.bn{font-family:'Orbitron',sans-serif;font-size:18px;font-weight:900;letter-spacing:5px;
  color:var(--c);text-shadow:0 0 18px var(--c20)}
.bs{font-size:9px;color:var(--lbl);letter-spacing:3px;margin-top:2px}
.tabs{display:flex;align-items:stretch;gap:0;flex:1;justify-content:center}
.tb{font-family:'Orbitron',sans-serif;font-size:11px;font-weight:600;letter-spacing:3px;
  color:var(--lbl);padding:0 20px;cursor:pointer;border:none;background:none;
  border-bottom:2px solid transparent;transition:all .2s;white-space:nowrap}
.tb:hover{color:var(--c40)}
.tb.act{color:var(--c);border-bottom-color:var(--c)}
.clkw{text-align:right;flex-shrink:0}
.cll{font-size:9px;color:var(--lbl);letter-spacing:2px}
.clv{font-family:'Orbitron',sans-serif;font-size:15px;font-weight:700;color:var(--c)}

/* ── CONTENT ── */
.content{flex:1;overflow-y:auto;overflow-x:hidden;min-height:0;
  scrollbar-width:thin;scrollbar-color:var(--c08) transparent}
.content::-webkit-scrollbar{width:4px}
.content::-webkit-scrollbar-thumb{background:var(--c20)}
.tp{display:none;flex-direction:column;gap:16px;padding:20px 28px}
.tp.act{display:flex}

/* ── PANEL ── */
.p{background:linear-gradient(145deg,rgba(0,10,20,.98),rgba(0,4,10,.99));
  border:1px solid var(--brd);position:relative;
  clip-path:polygon(0 0,calc(100% - 12px) 0,100% 12px,100% 100%,12px 100%,0 calc(100% - 12px))}
.p::before,.p::after,.p .cb,.p .ct{content:'';position:absolute;width:9px;height:9px;pointer-events:none}
.p::before{top:-1px;left:-1px;border-top:1px solid var(--c40);border-left:1px solid var(--c40)}
.p::after{bottom:-1px;right:-1px;border-bottom:1px solid var(--c40);border-right:1px solid var(--c40)}
.p .cb{bottom:-1px;left:-1px;border-bottom:1px solid var(--c40);border-left:1px solid var(--c40)}
.p .ct{top:-1px;right:-1px;border-top:1px solid var(--c40);border-right:1px solid var(--c40)}
.ph{font-family:'Orbitron',sans-serif;font-size:11px;font-weight:700;letter-spacing:3px;
  color:var(--c);padding-bottom:10px;border-bottom:1px solid var(--c08);margin-bottom:14px;
  display:flex;align-items:center;gap:9px}
.phd{width:5px;height:5px;background:var(--c);box-shadow:0 0 7px var(--c);flex-shrink:0}

/* ── ARC GAUGE ── */
.gc{clip-path:polygon(0 0,calc(100% - 8px) 0,100% 8px,100% 100%,8px 100%,0 calc(100% - 8px));
  background:linear-gradient(145deg,rgba(0,10,20,.98),rgba(0,4,10,.99));
  border:1px solid var(--brd);padding:14px 10px 12px;
  display:flex;flex-direction:column;align-items:center}
.gw{position:relative}
.gv{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.gn{font-family:'Orbitron',sans-serif;font-weight:900;line-height:1}
.gst{font-size:10px;letter-spacing:2px;margin-top:4px}
.gl{font-family:'Orbitron',sans-serif;font-size:10px;font-weight:600;letter-spacing:2px;
  text-transform:uppercase;margin-top:6px;color:var(--lbl);text-align:center;line-height:1.4}
.gmm{display:flex;justify-content:space-between;font-size:9px;color:var(--c08);margin-top:2px}

/* ── SEMI GAUGE ── */
.sc{clip-path:polygon(0 0,calc(100% - 7px) 0,100% 7px,100% 100%,0 100%);
  background:linear-gradient(145deg,rgba(0,10,20,.98),rgba(0,4,10,.99));
  border:1px solid var(--brd);padding:12px 12px 14px;
  display:flex;flex-direction:column;align-items:center;gap:4px}
.sl2{font-family:'Orbitron',sans-serif;font-size:10px;letter-spacing:3px;color:var(--lbl);text-align:center}
.sv{font-family:'Orbitron',sans-serif;font-size:16px;font-weight:700;text-align:center;line-height:1.2;margin-top:4px}

/* ── CARD ── */
.card{clip-path:polygon(0 0,calc(100% - 6px) 0,100% 6px,100% 100%,0 100%);
  background:linear-gradient(145deg,rgba(0,10,20,.98),rgba(0,4,10,.99));
  border:1px solid var(--brd);padding:13px 16px;display:flex;flex-direction:column;gap:5px}
.cdl{font-size:10px;letter-spacing:3px;color:var(--lbl);text-transform:uppercase}
.cdv{font-family:'Orbitron',sans-serif;font-size:22px;font-weight:700;line-height:1.1;color:var(--c)}
.secl{font-family:'Orbitron',sans-serif;font-size:10px;letter-spacing:4px;color:var(--lbl);
  padding:4px 0 3px;display:flex;align-items:center;gap:8px}
.secl::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--c08),transparent)}

/* ── RISK TABLE ── */
.rt{width:100%;border-collapse:collapse}
.rt th{font-family:'Orbitron',sans-serif;font-size:10px;letter-spacing:2px;color:var(--lbl);
  padding:5px 8px;border-bottom:1px solid var(--c08);text-align:left}
.rt td{font-size:12px;padding:7px 8px;border-bottom:1px solid var(--c04);color:var(--txt)}
.rt td:last-child{text-align:right;font-family:'Orbitron',sans-serif;font-size:13px;font-weight:700;color:var(--c)}

/* ── COMP BARS ── */
.cbar{margin-bottom:12px}
.cbh{display:flex;justify-content:space-between;margin-bottom:5px}
.cbn{font-size:11px;color:var(--txt)}
.cbs{font-family:'Orbitron',sans-serif;font-size:12px;font-weight:700;color:var(--c)}
.cbt{height:5px;background:var(--c04);border-radius:2px;overflow:hidden}
.cbf{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--c08),var(--c))}

/* ── SCORE BAR ── */
.sbw{margin-top:14px}
.sbl2{display:flex;justify-content:space-between;font-size:11px;color:var(--lbl);margin-bottom:6px}
.sbt{height:5px;background:var(--c04);border-radius:2px;overflow:hidden}
.sbf{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--c08),var(--c));
  box-shadow:0 0 10px var(--c20)}

/* ── FLOW ITEMS ── */
.flow-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.fhdr{font-family:'Orbitron',sans-serif;font-size:11px;letter-spacing:3px;color:var(--c60);
  padding-bottom:8px;border-bottom:1px solid var(--c08);margin-bottom:10px;display:flex;align-items:center;gap:6px}
.fi{display:flex;align-items:center;gap:10px;padding:8px 12px;
  background:var(--c04);border:1px solid var(--brd);margin-bottom:6px}
.fi-t{font-family:'Orbitron',sans-serif;font-size:13px;font-weight:700;color:var(--c);width:52px;flex-shrink:0}
.fi-bar{flex:1;height:5px;background:var(--c08);border-radius:2px;overflow:hidden}
.fi-fill{height:100%;border-radius:2px}
.fi-v{font-family:'Orbitron',sans-serif;font-size:12px;font-weight:700;color:var(--c);white-space:nowrap;margin-left:8px}
.fi-s{font-size:10px;color:var(--lbl);white-space:nowrap;margin-left:4px}

/* ── LEITURA ── */
.li{display:flex;align-items:center;gap:10px;padding:9px 0;border-bottom:1px solid var(--c08)}
.lg{font-family:'Orbitron',sans-serif;font-size:14px;font-weight:700;color:var(--c);width:18px;flex-shrink:0}
.ld{width:9px;height:9px;border-radius:50%;flex-shrink:0;background:var(--c)}
.lt{font-size:12px;color:var(--txt);line-height:1.5}

/* ── CHART WRAP ── */
.cw{position:relative;width:100%}

/* ── TICKER ── */
.ticker{flex-shrink:0;border-top:1px solid var(--c08);background:rgba(0,2,6,.9);overflow:hidden;padding:6px 0}
.ti{display:inline-block;white-space:nowrap;animation:tck 42s linear infinite}
.ti s{font-size:11px;color:var(--c40);margin:0 22px;text-decoration:none}
.up{color:rgba(0,212,232,1);text-shadow:0 0 6px rgba(0,212,232,.4)}.dn{color:rgba(248,81,73,.9);text-shadow:0 0 6px rgba(248,81,73,.3)}
@keyframes tck{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
</style>
<script src="https://unpkg.com/zdog@1/dist/zdog.dist.min.js"></script>
</head>
<body>
<canvas id="bg"></canvas>
<div class="scl"></div>

<!-- BOOT -->
<div id="boot">
  <canvas id="boot-reactor" width="80" height="80" style="display:block;margin-bottom:16px"></canvas>
  <div class="btitle">INICIALIZANDO J.A.R.V.I.S</div>
  <div id="blog"></div>
</div>

<!-- APP -->
<div id="app">

  <!-- CMD: aparece UMA vez, resumo topo -->
  <div class="cmd">
    <div class="cmd-id">SPX MARKET COMMAND</div>
    <div class="dvd"></div>
    <div class="cs"><div class="csl">SPOT</div><div class="csv" style="color:rgba(0,212,232,1);text-shadow:0 0 8px rgba(0,212,232,.4)">__JV_SPOT__</div></div>
    <div class="dvd"></div>
    <div class="cs"><div class="csl">GAMMA FLIP</div><div class="csv" style="color:var(--a60)">__JV_FLIP__</div></div>
    <div class="dvd"></div>
    <div class="cs"><div class="csl">GEX NET</div><div class="csv" style="color:__JV_C_GEX__">__JV_GEX__</div></div>
    <div class="dvd"></div>
    <div class="cs"><div class="csl">P/C RATIO</div><div class="csv" style="color:var(--a60)">__JV_PC__</div></div>
    <div class="dvd"></div>
    <div class="cs"><div class="csl">IV−RV</div><div class="csv" style="color:__JV_C_IVRV__">__JV_IVRV__</div></div>
    <div class="dvd"></div>
    <div class="cs"><div class="csl">SQUEEZE RISK</div><div class="csv" style="color:__JV_C_SQ__">__JV_SQ__</div></div>
    <div class="dvd"></div>
    <div class="cs"><div class="csl">TAIL RISK</div><div class="csv" style="color:__JV_C_TAIL__">__JV_TAIL__</div></div>
  </div>

  <!-- HEADER -->
  <div class="hdr">
    <canvas id="hdr-reactor" width="32" height="32" style="flex-shrink:0;cursor:grab"></canvas>
    <div class="brand">
      <div class="bn">J.A.R.V.I.S</div>
      <div class="bs">JUST A RATHER VERY INTELLIGENT SYSTEM  ·  OPTIONS CORE  ·  v4.2</div>
    </div>
    <div class="tabs">
      <button class="tb act" data-t="painel">PAINEL</button>
      <button class="tb" data-t="risco">RISCO</button>
      <button class="tb" data-t="gregas">GREGAS</button>
      <button class="tb" data-t="estrutura">ESTRUTURA</button>
      <button class="tb" data-t="cta">CTA</button>
    </div>
    <div class="clkw" style="text-align:right;flex-shrink:0;min-width:180px">
      <div id="clk-session" style="font-family:'Orbitron',sans-serif;font-size:9px;letter-spacing:2px;color:var(--lbl);margin-bottom:2px">● CARREGANDO...</div>
      <div style="font-family:'Orbitron',sans-serif;font-size:20px;font-weight:700;color:var(--c);line-height:1;letter-spacing:2px" id="clk-time">--:--:--</div>
      <div style="display:flex;justify-content:flex-end;gap:14px;margin-top:3px">
        <div style="font-size:9px;color:var(--lbl);letter-spacing:1px" id="clk-date">---</div>
        <div style="font-size:9px;color:var(--a40);letter-spacing:1px" id="clk-ny">NY --:--</div>
      </div>
      <div style="font-size:8px;color:rgba(0,212,232,.3);letter-spacing:1px;margin-top:2px">DATA: __JV_TS__</div>
    </div>
  </div>

  <!-- CONTENT -->
  <div class="content">

    <!-- ══ PAINEL ══ -->
    <div class="tp act" id="tab-painel">
      <!-- 4 gauges top -->
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px" id="gr1"></div>
      <!-- 3 gauges bottom -->
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px" id="gr2"></div>
      <!-- Vol metrics — PRÊMIO IV-RV e SKEW já estão nos gauges acima -->
      <div class="secl">Volatilidade Implícita</div>
      <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:16px">
        <div class="card"><div class="cdl">IV 30D ATM</div><div class="cdv">__JV_IV30__</div></div>
        <div class="card"><div class="cdl">RV 30D REALIZADA</div><div class="cdv" style="opacity:.7">__JV_RV30__</div></div>
      </div>
      <!-- Key levels -->
      <div class="secl">Níveis Chave</div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px">
        <div class="card"><div class="cdl">GAMMA FLIP</div><div class="cdv">~__JV_FLIP__</div></div>
        <div class="card"><div class="cdl">CALL WALL</div><div class="cdv">__JV_CW__</div></div>
        <div class="card"><div class="cdl">PUT WALL</div><div class="cdv" style="opacity:.55">__JV_PW__</div></div>
      </div>
      <div style="height:8px"></div>
    </div>

    <!-- ══ RISCO ══ -->
    <div class="tp" id="tab-risco">
      <div style="display:grid;grid-template-columns:1fr 1.2fr 0.9fr;gap:16px">

        <!-- Tail Risk table -->
        <div class="p" style="padding:18px 20px;display:flex;flex-direction:column">
          <div class="cb"></div><div class="ct"></div>
          <div class="ph"><div class="phd"></div>TAIL RISK — __JV_TAIL_INT__/100</div>
          <div style="font-size:11px;color:var(--lbl);margin-bottom:12px;letter-spacing:1px">ELEVADO — Monitorar sinais de stress</div>
          <div style="font-size:11px;color:var(--lbl);margin:8px 0 12px;line-height:1.8">
            Score calculado via BQL: excess kurtosis, skew, IV/RV ratio, put skew e risk reversal da superfície de vol do SPX.
          </div>
          <div class="sbw" style="margin-top:auto">
            <div class="sbl2"><span>SCORE TOTAL</span><span style="font-family:'Orbitron',sans-serif;font-size:16px;color:var(--c);font-weight:700">__JV_TAIL_NUM__ / 100</span></div>
            <div class="sbt"><div class="sbf" style="width:__JV_TAIL_PCT__%"></div></div>
          </div>
        </div>

        <!-- Flow Z-Score chart -->
        <div class="p" style="padding:18px 20px;display:flex;flex-direction:column">
          <div class="cb"></div><div class="ct"></div>
          <div class="ph"><div class="phd"></div>COMPONENTES FLOW SCORE — Z-SCORE</div>
          <div class="cw" style="flex:1;min-height:520px"><canvas id="flowChart"></canvas></div>
        </div>

        <!-- Gamma Squeeze -->
        <div class="p" style="padding:18px 20px;display:flex;flex-direction:column">
          <div class="cb"></div><div class="ct"></div>
          <div class="ph"><div class="phd" style="animation:blink 1s infinite"></div>GAMMA SQUEEZE — __JV_SQ_NUM__/100</div>
          <div style="font-size:11px;color:var(--lbl);margin-bottom:14px;letter-spacing:1px">RISCO MUITO ALTO — condições extremas</div>
          <div id="gc" style="flex:1"></div>
          <div class="sbw" style="border-top:1px solid var(--c08);padding-top:10px">
            <div class="sbl2"><span>SCORE TOTAL</span><span style="font-family:'Orbitron',sans-serif;font-size:16px;color:var(--c);font-weight:700">__JV_SQ_NUM__ / 100</span></div>
            <div class="sbt"><div class="sbf" style="width:__JV_SQ_PCT__%"></div></div>
          </div>
        </div>
      </div>
      <div style="height:8px"></div>
    </div>

    <!-- ══ GREGAS ══ -->
    <div class="tp" id="tab-gregas">

      <!-- 4 semi gauges + leitura -->
      <div style="display:grid;grid-template-columns:repeat(4,1fr) 260px;gap:16px">
        <div id="sgr" style="display:contents"></div>
        <div class="p" style="padding:18px 20px">
          <div class="cb"></div><div class="ct"></div>
          <div class="ph"><div class="phd"></div>LEITURA DAS GREGAS</div>
          <div style="display:flex;flex-direction:column;gap:1px">
            <div class="li"><span class="lg">Δ</span><span class="ld" style="opacity:1"></span><span class="lt">Dealers vendidos → compra</span></div>
            <div class="li"><span class="lg">Γ</span><span class="ld" style="opacity:.35"></span><span class="lt">GEX− acelera movimentos</span></div>
            <div class="li"><span class="lg">V</span><span class="ld" style="opacity:.2"></span><span class="lt">Vanna: Neutro</span></div>
            <div class="li"><span class="lg">C</span><span class="ld" style="opacity:.9"></span><span class="lt">Delta decai → reforça hedge</span></div>
          </div>
        </div>
      </div>


      <div style="height:8px"></div>
    </div>

    <!-- ══ ESTRUTURA ══ -->
    <div class="tp" id="tab-estrutura">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
        <div class="p" style="padding:18px 20px;display:flex;flex-direction:column">
          <div class="cb"></div><div class="ct"></div>
          <div class="ph"><div class="phd"></div>GAMMA EXPOSURE (GEX) — POR STRIKE</div>
          <div style="font-size:11px;color:var(--lbl);margin-bottom:8px">$ Bi por 1% de movimento — Spot __JV_SPOT__ | G-Flip __JV_FLIP__</div>
          <div class="cw" style="flex:1;min-height:380px"><canvas id="gexChart"></canvas></div>
        </div>
        <div class="p" style="padding:18px 20px;display:flex;flex-direction:column">
          <div class="cb"></div><div class="ct"></div>
          <div class="ph"><div class="phd"></div>NÍVEIS DE GAMMA — SPOT vs REFERÊNCIAS</div>
          <div style="font-size:11px;color:var(--lbl);margin-bottom:8px;line-height:1.7">
            <span style="color:rgba(0,212,232,.9);font-weight:700">▲ Call Wall</span> = strike com maior Open Interest de Calls (resistência técnica) &nbsp;·&nbsp;
            <span style="color:rgba(245,166,35,.95);font-weight:700">◆ Gamma Flip</span> = divisor crítico: acima=vol amortecida/mercado pinado, abaixo=vol amplificada &nbsp;·&nbsp;
            <span style="color:rgba(0,212,232,.5);font-weight:700">▼ Put Wall</span> = strike com maior Open Interest de Puts (suporte técnico) &nbsp;·&nbsp;
            linhas tracejadas finas = projeção de movimento esperado pela IV em 5 dias
          </div>
          <div class="cw" style="flex:1;min-height:380px"><canvas id="levChart"></canvas></div>
        </div>
      </div>
      <div style="height:8px"></div>
    </div>

    <!-- ══ CTA ══ -->
    <div class="tp" id="tab-cta">
      <div class="p" style="padding:18px 20px;display:flex;flex-direction:column">
        <div class="cb"></div><div class="ct"></div>
        <div class="ph"><div class="phd"></div>CTA ESTIMATES — S&P 500 (NOTIONAL $B)</div>
        <div style="font-size:11px;color:var(--lbl);margin-bottom:8px">
          Dados do modelo CTA — estimativas baseadas em posicionamento histórico via BQL.
          CTAs (Commodity Trading Advisors) são fundos sistemáticos que seguem tendência.
          Posição em $B: <span style="color:rgba(0,212,232,.9)">positivo = comprado</span> · <span style="color:rgba(248,81,73,.9)">negativo = vendido</span>.
        </div>
        <div class="cw" style="min-height:380px"><canvas id="ctaLine"></canvas></div>
      </div>
      <div class="p" style="padding:18px 20px;display:flex;flex-direction:column">
        <div class="cb"></div><div class="ct"></div>
        <div class="ph"><div class="phd"></div>CTA SCENARIO FLOWS ($B) — 1 SEMANA vs 1 MÊS</div>
        <div class="cw" style="min-height:360px"><canvas id="ctaBar"></canvas></div>
      </div>
      <div style="height:8px"></div>
    </div>

  </div><!-- /content -->

  <div class="ticker"><div class="ti" id="ti"></div></div>
</div><!-- /app -->

<script>
// ── PARTICLES
const cv=document.getElementById('bg'),ctx2=cv.getContext('2d');
let W,H,pts=[];
function rsz(){W=cv.width=innerWidth;H=cv.height=innerHeight;
  pts=[];const n=Math.floor(W*H/18000);
  for(let i=0;i<n;i++)pts.push({x:Math.random()*W,y:Math.random()*H,
    vx:(Math.random()-.5)*.1,vy:(Math.random()-.5)*.1,r:Math.random()*.8+.3,a:Math.random()*.25+.08})}
function drawBg(){ctx2.clearRect(0,0,W,H);
  ctx2.strokeStyle='rgba(0,60,80,.04)';ctx2.lineWidth=.5;
  const s=52;for(let x=0;x<W;x+=s)for(let y=0;y<H;y+=s)ctx2.strokeRect(x,y,s,s);
  pts.forEach((p,i)=>{p.x+=p.vx;p.y+=p.vy;
    if(p.x<0||p.x>W)p.vx*=-1;if(p.y<0||p.y>H)p.vy*=-1;
    ctx2.beginPath();ctx2.arc(p.x,p.y,p.r,0,Math.PI*2);
    ctx2.fillStyle=`rgba(0,180,210,${p.a})`;ctx2.fill();
    for(let j=i+1;j<pts.length;j++){
      const dx=pts[j].x-p.x,dy=pts[j].y-p.y,d=Math.sqrt(dx*dx+dy*dy);
      if(d<80){ctx2.beginPath();ctx2.moveTo(p.x,p.y);ctx2.lineTo(pts[j].x,pts[j].y);
        ctx2.strokeStyle=`rgba(0,140,180,${.05*(1-d/80)})`;ctx2.lineWidth=.4;ctx2.stroke()}}});
  requestAnimationFrame(drawBg)}
window.addEventListener('resize',rsz);rsz();drawBg();

// ── CLOCK
function _upClock(){
  const now=new Date();
  document.getElementById('clk-time').textContent=
    now.toLocaleTimeString('pt-BR',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
  document.getElementById('clk-date').textContent=
    now.toLocaleDateString('pt-BR',{weekday:'short',day:'2-digit',month:'short'});
  const nyFmt=new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',
    hour:'2-digit',minute:'2-digit',hour12:false});
  const nyStr=nyFmt.format(now);
  document.getElementById('clk-ny').textContent='NY '+nyStr;
  const nyD=new Date(now.toLocaleString('en-US',{timeZone:'America/New_York'}));
  const tot=nyD.getHours()*60+nyD.getMinutes(),wd=nyD.getDay();
  let ses,sc;
  if(wd===0||wd===6){ses='■ FIM DE SEMANA';sc='rgba(0,212,232,.35)'}
  else if(tot<240){ses='■ FECHADO';sc='rgba(0,212,232,.35)'}
  else if(tot<570){ses='◐ PRÉ-MERCADO';sc='rgba(245,166,35,.85)'}
  else if(tot<960){ses='● AO VIVO — NYSE/NASDAQ';sc='rgba(0,212,232,1)'}
  else if(tot<1200){ses='◑ PÓS-MERCADO';sc='rgba(245,166,35,.75)'}
  else{ses='■ FECHADO';sc='rgba(0,212,232,.35)'}
  const sel=document.getElementById('clk-session');
  sel.textContent=ses;sel.style.color=sc;
}
setInterval(_upClock,1000);_upClock();

// ── TABS
document.querySelectorAll('.tb').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('.tb').forEach(x=>x.classList.remove('act'));
  document.querySelectorAll('.tp').forEach(x=>x.classList.remove('act'));
  b.classList.add('act');
  document.getElementById('tab-'+b.dataset.t).classList.add('act');
}));

// ── ZDOG REACTORS ──────────────────────────────────────────────────────────
(function(){
  if(typeof Zdog==='undefined') return;
  function mkR(id,sz,drag){
    const cvs=document.getElementById(id); if(!cvs) return;
    cvs.width=sz; cvs.height=sz;
    const illo=new Zdog.Illustration({element:'#'+id,resize:false,zoom:sz/90,dragRotate:!!drag});
    const c1='rgba(0,212,232,.95)',c2='rgba(0,212,232,.55)',c3='rgba(0,212,232,.22)';
    new Zdog.Ellipse({addTo:illo,diameter:62,stroke:3,color:c1,fill:false,rotate:{x:Zdog.TAU/4}});
    new Zdog.Ellipse({addTo:illo,diameter:44,stroke:2.2,color:c2,fill:false,rotate:{x:Zdog.TAU/6,y:Zdog.TAU/8}});
    new Zdog.Ellipse({addTo:illo,diameter:26,stroke:1.5,color:c3,fill:false,rotate:{x:-Zdog.TAU/5,y:-Zdog.TAU/6}});
    new Zdog.Shape({addTo:illo,stroke:10,color:'rgba(0,212,232,1)',translate:{z:5}});
    new Zdog.Shape({addTo:illo,stroke:20,color:'rgba(0,212,232,.1)',translate:{z:3}});
    let t=0;
    (function anim(){t+=0.007;illo.rotate.y=t;illo.updateRenderGraph();requestAnimationFrame(anim)})();
  }
  mkR('boot-reactor',80,false);
  mkR('hdr-reactor',32,true);
})();

// ── BOOT VOICE ────────────────────────────────────────────────────────────────
function _jvSpeak(txt){
  if(!('speechSynthesis' in window)) return;
  speechSynthesis.cancel();
  const u=new SpeechSynthesisUtterance(txt);
  u.lang='en-US'; u.pitch=0.72; u.rate=0.88; u.volume=0.72;
  function go(){speechSynthesis.speak(u);}
  if(speechSynthesis.getVoices().length>0) go();
  else { speechSynthesis.onvoiceschanged=function(){speechSynthesis.onvoiceschanged=null;go();};
         setTimeout(go,250); }
}

// ── BOOT
const BL=['Inicializando núcleo de risco...','Carregando superfície de vol...','Conectando feed OI...','Compilando GEX matrix...','Calibrando modelos de cauda...','Sincronizando posicionamento CTA...','Sistema operacional — ONLINE'];
let bi=0;const bel=document.getElementById('blog');
(function nb(){if(bi<BL.length){bel.innerHTML+=BL[bi++]+'<br>';setTimeout(nb,260)}
 else setTimeout(()=>{document.getElementById('boot').classList.add('gone');
   document.getElementById('app').classList.add('on');buildAll();_jvSpeak('Welcome trader. J A R V I S online.')},500);})();

// ── ARC GAUGE — intensity via opacity only (monochromatic)
function arcGauge(container,{v,mn,mx,label,unit='',state='',intensity=1,size=140}){
  const el=document.createElement('div');el.className='gc';
  const alpha=0.42+intensity*0.58;
  const col=intensity>0.78?`rgba(248,81,73,${alpha})`:
             intensity>0.48?`rgba(245,166,35,${alpha})`:
             `rgba(0,212,232,${alpha})`;
  const glw=intensity>0.78?`rgba(248,81,73,${alpha*.5})`:
             intensity>0.48?`rgba(245,166,35,${alpha*.5})`:
             `rgba(0,212,232,${alpha*.5})`;
  const R=size*.43,CX=size/2,CY=size/2,circ=2*Math.PI*R,sw=.75,tr=sw*circ;
  const pct=Math.max(0,Math.min(1,(v-mn)/(mx-mn)));
  const fill=pct*tr;const id='a'+Math.random().toString(36).slice(2,8);
  el.innerHTML=`
    <div class="gw" style="width:${size}px;height:${size}px">
      <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
        <defs><filter id="${id}"><feGaussianBlur stdDeviation="2.5" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
        <circle cx="${CX}" cy="${CY}" r="${R+8}" fill="none" stroke="rgba(0,212,232,.06)" stroke-width="1" stroke-dasharray="3 8">
          <animateTransform attributeName="transform" type="rotate" from="0 ${CX} ${CY}" to="360 ${CX} ${CY}" dur="${14+pct*9}s" repeatCount="indefinite"/>
        </circle>
        <circle cx="${CX}" cy="${CY}" r="${R}" fill="none" stroke="rgba(0,30,44,.95)" stroke-width="7"
          stroke-dasharray="${tr} ${circ-tr}" stroke-dashoffset="0" transform="rotate(-225 ${CX} ${CY})" stroke-linecap="round"/>
        <circle cx="${CX}" cy="${CY}" r="${R}" fill="none" stroke="${col}" stroke-width="10" opacity=".1"
          stroke-dasharray="${fill} ${circ-fill}" stroke-dashoffset="0" transform="rotate(-225 ${CX} ${CY})" stroke-linecap="round"/>
        <circle cx="${CX}" cy="${CY}" r="${R}" fill="none" stroke="${col}" stroke-width="6"
          stroke-dasharray="${fill} ${circ-fill}" stroke-dashoffset="0" transform="rotate(-225 ${CX} ${CY})" stroke-linecap="round" filter="url(#${id})"/>
        <circle cx="${CX}" cy="${CY}" r="${R*.6}" fill="none" stroke="rgba(0,212,232,.1)" stroke-width="1" stroke-dasharray="2 6">
          <animateTransform attributeName="transform" type="rotate" from="0 ${CX} ${CY}" to="-360 ${CX} ${CY}" dur="11s" repeatCount="indefinite"/>
        </circle>
      </svg>
      <div class="gv">
        <span class="gn" style="color:${col};text-shadow:0 0 14px ${glw};font-size:${size*.17}px">${unit==='%'?v+'%':v}</span>
        ${state?`<span class="gst" style="color:${col};opacity:.8">${state}</span>`:''}
      </div>
    </div>
    <div class="gmm" style="width:${size}px"><span>${mn}${unit==='%'?'%':''}</span><span>${mx}${unit==='%'?'%':''}</span></div>
    <div class="gl">${label}</div>`;
  container.appendChild(el);
}

// ── SEMI GAUGE
function semiGauge(container,{v,mn,mx,label,unit='$Bn',intensity=0.7}){
  const el=document.createElement('div');el.className='sc';
  const alpha=0.42+intensity*0.58;
  const col=intensity>0.78?`rgba(248,81,73,${alpha})`:
             intensity>0.48?`rgba(245,166,35,${alpha})`:
             `rgba(0,212,232,${alpha})`;
  const W2=180,H2=100,R=70,CX=90,CY=90;
  const pct=Math.max(0,Math.min(1,(v-mn)/(mx-mn)));
  const sa=Math.PI,ea=2*Math.PI,fa=sa+pct*(ea-sa);
  const tx=a=>CX+R*Math.cos(a),ty=a=>CY+R*Math.sin(a);
  const nz=pct>0.001;
  const trackD=`M ${tx(sa)} ${ty(sa)} A ${R} ${R} 0 1 1 ${tx(ea)} ${ty(ea)}`;
  const fillD=nz?`M ${tx(sa)} ${ty(sa)} A ${R} ${R} 0 0 1 ${tx(fa)} ${ty(fa)}`:'';
  const id='s'+Math.random().toString(36).slice(2,8);
  el.innerHTML=`
    <div class="sl2">${label}</div>
    <svg width="${W2}" height="${H2}" viewBox="0 0 ${W2} ${H2}" style="display:block">
      <defs><filter id="${id}"><feGaussianBlur stdDeviation="2" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
      <path d="${trackD}" fill="none" stroke="rgba(0,30,44,.9)" stroke-width="8" stroke-linecap="round"/>
      ${nz?`<path d="${fillD}" fill="none" stroke="${col}" stroke-width="8" stroke-linecap="round" filter="url(#${id})"/>
            <path d="${fillD}" fill="none" stroke="${col}" stroke-width="14" stroke-linecap="round" opacity=".1"/>`:``}
      <text x="4" y="96" font-size="9" fill="rgba(0,100,130,.5)" font-family="Share Tech Mono">${mn}</text>
      <text x="${W2-12}" y="96" font-size="9" fill="rgba(0,100,130,.5)" font-family="Share Tech Mono" text-anchor="end">${mx}</text>
    </svg>
    <div class="sv" style="color:${col};text-shadow:0 0 10px rgba(0,212,232,${alpha*.4})">${v} <span style="font-size:12px;opacity:.6">${unit}</span></div>`;
  container.appendChild(el);
}

// ── CHART DEFAULTS
Chart.defaults.color='rgba(0,140,170,.65)';
Chart.defaults.font={family:"'Share Tech Mono',monospace",size:11};
const G='rgba(0,50,70,.25)';
const TT={backgroundColor:'rgba(0,4,10,.97)',borderColor:'rgba(0,80,100,.3)',borderWidth:1,
  titleColor:'rgba(0,200,220,.8)',bodyColor:'rgba(0,140,170,.85)',padding:8};

function buildAll(){

  // PAINEL gauges row 1
  const g1=document.getElementById('gr1');
  [{v:__JV_V_FRAG__,mn:0,mx:20,label:'FRAGILIDADE',unit:'%',state:'ALTO',intensity:0.95},
   {v:__JV_V_IVRV__,mn:0,mx:10,label:'PRÊMIO VOL',state:'pp',intensity:0.55},
   {v:__JV_V_SKEW__,mn:-15,mx:15,label:'SKEW P25-C25',unit:'%',intensity:0.65},
   {v:__JV_V_MOVE__,mn:0,mx:5,label:'MOV ESP 10',unit:'%',intensity:0.35}
  ].forEach(g=>arcGauge(g1,g));

  const g2=document.getElementById('gr2');
  [{v:__JV_V_TAIL__,mn:0,mx:100,label:'TAIL RISK',state:'ELEVADO',intensity:0.65},
   {v:__JV_V_FLOW__,mn:0,mx:100,label:'FLOW SCORE',state:'NEUTRO',intensity:0.5},
   {v:__JV_V_SQ__,mn:0,mx:100,label:'GAMMA SQUEEZE',state:'CRÍTICO',intensity:1}
  ].forEach(g=>arcGauge(g2,g));

  // RISCO — gamma comps
  __JV_SQ_COMPS__.forEach(c=>{const p=(c.s/c.m)*100;
    document.getElementById('gc').innerHTML+=`
    <div class="cbar">
      <div class="cbh">
        <span class="cbn">${c.n}</span>
        <span class="cbs" style="opacity:${c.i}">${c.s}<span style="opacity:.4;font-size:10px">/${c.m}</span></span>
      </div>
      <div class="cbt"><div class="cbf" style="width:${p}%;opacity:${c.i}"></div></div>
    </div>`});

  // RISCO — Flow Z-Score
  new Chart(document.getElementById('flowChart'),{
    type:'bar',
    data:{
      labels:['CTA','Dealer/MM','Vol Ctrl',['Risk','Parity'],['ETFs','Alav.'],['ETFs','Passivos'],'Buyback','COT'],
      datasets:[{
        label:'Z-Score',
        data:[__JV_FLOW_DATA__],
        backgroundColor:d=>d.raw>=0?'rgba(88,166,255,.78)':'rgba(248,81,73,.82)',
        borderColor:d=>d.raw>=0?'rgba(120,200,255,1)':'rgba(255,120,120,1)',
        borderWidth:1,borderRadius:2,
        order:2
      },{
        type:'line',
        label:'Peso',
        data:[__JV_FLOW_W_DATA__],
        yAxisID:'y1',
        showLine:false,
        pointRadius:4,
        pointHoverRadius:4,
        pointBackgroundColor:'rgba(180,190,205,.95)',
        pointBorderColor:'rgba(40,55,70,.95)',
        pointBorderWidth:1,
        order:1
      }]
    },
    options:{responsive:true,maintainAspectRatio:false,
      layout:{padding:{top:6,bottom:12,left:6,right:6}},
      plugins:{
        legend:{
          display:true,
          position:'bottom',
          labels:{color:'rgba(180,200,220,.85)',boxWidth:10,font:{size:9}}
        },
        tooltip:TT
      },
      scales:{
        x:{
          grid:{color:G},
          ticks:{
            color:'rgba(0,200,220,.9)',
            font:{size:9,family:"'Orbitron',sans-serif"},
            autoSkip:false,
            maxRotation:0,
            minRotation:0,
            padding:8
          },
          border:{color:'rgba(0,80,100,.2)'}
        },
        y:{grid:{color:G},ticks:{color:'rgba(0,180,200,.7)'},min:-3.5,max:4,
          title:{display:true,text:'Z-Score',color:'rgba(0,120,150,.5)',font:{size:8}}},
        y1:{
          position:'right',
          min:0,max:1,
          grid:{drawOnChartArea:false},
          ticks:{
            color:'rgba(170,180,190,.8)',
            callback:(v)=>`${Math.round(v*100)}%`
          },
          title:{display:true,text:'Peso',color:'rgba(140,150,165,.65)',font:{size:8}}
        }
      }
    }
  });

  // GREGAS — semis
  const sg=document.getElementById('sgr');
  [{v:__JV_V_DELTA__,mn:__JV_V_DELTA_MIN__,mx:__JV_V_DELTA_MAX__,label:'Δ DELTA NOCIONAL',intensity:1},
   {v:__JV_V_GEX_SEMI__,mn:-40,mx:40,label:'Γ GAMMA (GEX NET)',intensity:0.4},
   {v:__JV_V_VANNA__,mn:__JV_V_VANNA_MIN__,mx:__JV_V_VANNA_MAX__,label:'V VANNA',intensity:0.25},
   {v:__JV_V_CHARM__,mn:__JV_V_CHARM_MIN__,mx:__JV_V_CHARM_MAX__,label:'C CHARM (DIÁRIO)',intensity:0.8}
  ].forEach(g=>semiGauge(sg,g));

  // ESTRUTURA — GEX curve
  const strikes=[],gex=[];
  for(let s=6400;s<=7000;s+=10){
    strikes.push(s);
    const x=(s-__JV_FLIP_NUM__)/200;
    gex.push(40*(2/(1+Math.exp(-x*3))-1));
  }
  new Chart(document.getElementById('gexChart'),{
    type:'line',
    data:{labels:strikes,datasets:[
      {label:'GEX $Bi/1%',data:gex,
       borderColor:'rgba(0,212,232,.8)',backgroundColor:'rgba(0,80,120,.1)',
       borderWidth:1.5,fill:true,pointRadius:0,tension:0.4},
      {label:'Spot __JV_SPOT__',
       data:strikes.map((s,i)=>Math.abs(s-__JV_SPOT_R10__)<6?gex[i]:null),
       type:'scatter',pointRadius:6,
       pointBackgroundColor:'rgba(0,212,232,1)',pointBorderColor:'rgba(0,212,232,.3)',pointBorderWidth:3},
      {label:'G-Flip __JV_FLIP__',
       data:strikes.map((s,i)=>Math.abs(s-__JV_FLIP_R10__)<6?gex[i]:null),
       type:'scatter',pointRadius:6,
       pointBackgroundColor:'rgba(0,212,232,.4)',pointBorderColor:'rgba(0,212,232,.2)',pointBorderWidth:3}
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'rgba(0,140,170,.7)',boxWidth:8,font:{size:8}}},tooltip:TT},
      scales:{
        x:{grid:{color:G},ticks:{color:'rgba(0,140,170,.5)',maxTicksLimit:8},
          title:{display:true,text:'SPX Strike',color:'rgba(0,120,150,.4)',font:{size:8}}},
        y:{grid:{color:G},ticks:{color:'rgba(0,140,170,.5)'},
          title:{display:true,text:'$ Bi / 1% move',color:'rgba(0,120,150,.4)',font:{size:8}}}
      }
    }
  });

  // ESTRUTURA — Gamma Levels (snapshot — sem dados históricos falsos)
  const lvSpot=__JV_SPOT_NUM__;
  const lvCW=__JV_CW_NUM__;
  const lvPW=__JV_PW_NUM__;
  const lvFlip=__JV_FLIP_NUM__;
  const lvEstUp=__JV_EST_UP_NUM__;
  const lvEstDn=__JV_EST_DN_NUM__;
  const lvRange=Math.max(200,(lvCW-lvPW)*0.35,Math.abs(lvEstUp-lvSpot)*1.8);
  const lvMin=Math.min(lvPW,lvEstDn,lvSpot)-lvRange;
  const lvMax=Math.max(lvCW,lvEstUp,lvSpot)+lvRange;

  // Determina a zona atual do SPX
  const zoneLabel=lvSpot>lvCW?'ACIMA CALL WALL — Mercado empurrado':
                  lvSpot>lvFlip?'ACIMA GAMMA FLIP — Vol baixa, dealers compram dip':
                  lvSpot>lvPW?'ABAIXO FLIP — Vol pode acelerar, zona de risco':
                  'ABAIXO PUT WALL — Suporte em risco, vol extrema';
  const zoneColor=lvSpot>lvFlip?'rgba(0,212,232,.9)':'rgba(248,81,73,.9)';

  // Plugin: desenha zonas coloridas + labels diretamente no gráfico
  const lvDraw={id:'lvDraw',afterDraw(chart){
    const {ctx,chartArea:{left,right,top,bottom},scales:{y}}=chart;
    // Zonas de fundo coloridas por região
    [
      [Math.max(lvCW,lvEstUp),lvMax,'rgba(0,212,232,.05)'],   // acima call wall
      [lvFlip,Math.max(lvCW,lvEstUp),'rgba(0,212,232,.09)'],  // entre flip e call wall
      [lvPW,lvFlip,'rgba(245,166,35,.06)'],                    // entre put wall e flip
      [Math.min(lvPW,lvEstDn),lvPW,'rgba(248,81,73,.09)'],    // abaixo put wall
    ].forEach(([lo,hi,c])=>{
      const y1=Math.min(bottom,Math.max(top,y.getPixelForValue(hi)));
      const y2=Math.min(bottom,Math.max(top,y.getPixelForValue(lo)));
      if(y1>=y2) return;
      ctx.save();ctx.fillStyle=c;ctx.fillRect(left,y1,right-left,y2-y1);ctx.restore();
    });
    // Labels das linhas de referência — lado direito
    const lblPad=right-8;
    [
      [lvCW,  '▲ CALL WALL  '+lvCW.toLocaleString('pt-BR'),  'rgba(0,212,232,.95)', 'bold 11px'],
      [lvFlip,'◆ GAMMA FLIP  '+lvFlip.toLocaleString('pt-BR'),'rgba(245,166,35,1)', 'bold 12px'],
      [lvPW,  '▼ PUT WALL  '+lvPW.toLocaleString('pt-BR'),   'rgba(0,212,232,.65)', 'bold 11px'],
      [lvEstUp,'↑ +5d IV  '+lvEstUp.toLocaleString('pt-BR'), 'rgba(0,212,232,.55)', '10px'],
      [lvEstDn,'↓ −5d IV  '+lvEstDn.toLocaleString('pt-BR'), 'rgba(248,81,73,.55)', '10px'],
      [lvSpot, '● SPX  '+lvSpot.toLocaleString('pt-BR'),     'rgba(0,212,232,1)',   'bold 13px'],
    ].forEach(([v,lbl,col,fnt])=>{
      const yp=y.getPixelForValue(v);
      if(yp<top-2||yp>bottom+2) return;
      const yc=Math.max(top+10,Math.min(bottom-10,yp));
      ctx.save();
      ctx.fillStyle=col;
      ctx.font=`${fnt} 'Share Tech Mono',monospace`;
      ctx.textAlign='right';ctx.textBaseline='middle';
      // fundo semi-transparente para legibilidade
      const w=ctx.measureText(lbl).width;
      ctx.fillStyle='rgba(2,8,16,.7)';
      ctx.fillRect(lblPad-w-6,yc-8,w+12,16);
      ctx.fillStyle=col;
      ctx.fillText(lbl,lblPad,yc);
      ctx.restore();
    });
    // Status da zona — canto superior esquerdo
    ctx.save();
    ctx.fillStyle='rgba(2,8,16,.75)';ctx.fillRect(left+6,top+6,420,22);
    ctx.fillStyle=zoneColor;
    ctx.font=`bold 11px 'Share Tech Mono',monospace`;
    ctx.textAlign='left';ctx.textBaseline='middle';
    ctx.fillText('ZONA ATUAL: '+zoneLabel,left+12,top+17);
    ctx.restore();
  }};

  new Chart(document.getElementById('levChart'),{
    type:'line',
    data:{labels:['HOJE','PROJ +5d'],datasets:[
      // Fan de projeção — zona preenchida
      {data:[lvSpot,lvEstUp],backgroundColor:'rgba(0,212,232,.07)',
       borderColor:'transparent',borderWidth:0,pointRadius:0,fill:'+1',tension:0},
      {data:[lvSpot,lvEstDn],backgroundColor:'rgba(248,81,73,.05)',
       borderColor:'transparent',borderWidth:0,pointRadius:0,fill:false,tension:0},
      // Linhas de referência — horizontais
      {data:[lvCW,lvCW],  borderColor:'rgba(0,212,232,.9)',   borderWidth:2.5,borderDash:[14,6],pointRadius:0,fill:false},
      {data:[lvFlip,lvFlip],borderColor:'rgba(245,166,35,1)', borderWidth:3,  borderDash:[10,5],pointRadius:0,fill:false},
      {data:[lvPW,lvPW],  borderColor:'rgba(0,212,232,.5)',   borderWidth:2,  borderDash:[14,6],pointRadius:0,fill:false},
      // Projeção +/− 5d
      {data:[lvSpot,lvEstUp],borderColor:'rgba(0,212,232,.5)',borderWidth:2,borderDash:[5,6],
       pointRadius:[0,8],pointBackgroundColor:'rgba(0,212,232,.7)',fill:false,tension:0},
      {data:[lvSpot,lvEstDn],borderColor:'rgba(248,81,73,.5)',borderWidth:2,borderDash:[5,6],
       pointRadius:[0,8],pointBackgroundColor:'rgba(248,81,73,.7)',fill:false,tension:0},
      // Spot
      {data:[lvSpot,null],type:'scatter',
       pointRadius:16,pointHoverRadius:18,
       pointBackgroundColor:'rgba(0,212,232,1)',
       pointBorderColor:'rgba(0,212,232,.2)',pointBorderWidth:6}
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{
        legend:{display:false},
        tooltip:{...TT,callbacks:{
          title:()=>'',
          label:ctx=>{
            const v=ctx.parsed.y; if(v==null) return null;
            return `${v.toLocaleString('pt-BR')}`;
          }
        }}
      },
      scales:{
        x:{grid:{color:G},
          ticks:{color:'rgba(0,200,220,.85)',font:{size:14,family:"'Orbitron',sans-serif"},
            padding:10}},
        y:{grid:{color:G},
          ticks:{color:'rgba(0,140,170,.7)',font:{size:11},
            callback:v=>v.toLocaleString('pt-BR')},
          min:lvMin,max:lvMax}
      }
    },
    plugins:[lvDraw]
  });

  // CTA — Line
  const ctaDates=['Dez 1','Dez 8','Dez 15','Dez 22','Jan 1','Jan 8','Jan 15','Jan 22','Fev 1','Fev 8','Fev 15','Mar 1','Mar 8','Mar 15'];
  const ctaHist=[65,22,-15,8,28,52,72,83,88,90,86,35,16,-68];
  const projDates=[...ctaDates,'Mar 22','Mar 29','Abr 5','Abr 12'];
  const nh=ctaDates.length; // 14
  // pad: 13 nulls + 5-point array = 18 items matching projDates
  const pad=arr=>[...Array(nh-1).fill(null),...arr];

  // inline plugin: $B labels on bars
  const barLabels={id:'barLabels',afterDatasetsDraw(chart){
    const {ctx}=chart;
    chart.data.datasets.forEach((ds,i)=>{
      chart.getDatasetMeta(i).data.forEach((bar,j)=>{
        const v=ds.data[j]; if(v==null) return;
        const lbl=(v>=0?'+':'')+v.toFixed(1)+'B';
        ctx.save(); ctx.fillStyle='rgba(0,212,232,1)';
        ctx.font='bold 11px monospace'; ctx.textAlign='center';
        ctx.textBaseline=v>=0?'bottom':'top';
        ctx.fillText(lbl,bar.x,v>=0?bar.y-4:bar.y+4); ctx.restore();
      });
    });
  }};

  new Chart(document.getElementById('ctaLine'),{
    type:'line',
    data:{labels:projDates,datasets:[
      {label:'CTA Notional (Hist)',
       data:[...ctaHist,...Array(4).fill(null)],
       borderColor:'rgba(0,212,232,.9)',backgroundColor:'rgba(0,60,100,.12)',
       borderWidth:2,fill:true,pointRadius:0,tension:0.35},
      {label:'Flat',
       data:pad([-68,-68,-68,-68,-68]),
       borderColor:'rgba(0,212,232,.3)',borderWidth:1.5,borderDash:[5,4],
       pointRadius:0,fill:false,spanGaps:true},
      {label:'Up 1σ  →+37B',
       data:pad([-68,-38,-8,18,37]),
       borderColor:'rgba(0,212,232,1)',borderWidth:2.5,borderDash:[5,3],
       pointRadius:[...Array(nh-1).fill(0),5,3,3,3,8],
       pointBackgroundColor:'rgba(0,212,232,1)',
       fill:false,tension:0.3,spanGaps:true},
      {label:'Up 2σ  →+75B',
       data:pad([-68,-20,15,48,75]),
       borderColor:'rgba(0,212,232,.6)',borderWidth:2,borderDash:[5,3],
       pointRadius:[...Array(nh-1).fill(0),4,3,3,3,7],
       pointBackgroundColor:'rgba(0,212,232,.6)',
       fill:false,tension:0.3,spanGaps:true},
      {label:'Down 1σ  →-75B',
       data:pad([-68,-70,-73,-74,-75]),
       borderColor:'rgba(248,81,73,.9)',borderWidth:2,borderDash:[5,3],
       pointRadius:[...Array(nh-1).fill(0),4,3,3,3,7],
       pointBackgroundColor:'rgba(248,81,73,.9)',
       fill:false,tension:0.2,spanGaps:true},
      {label:'Down 2.5σ  →-85B',
       data:pad([-68,-73,-78,-82,-85]),
       borderColor:'rgba(248,81,73,.45)',borderWidth:1.5,borderDash:[3,5],
       pointRadius:[...Array(nh-1).fill(0),3,2,2,2,6],
       pointBackgroundColor:'rgba(248,81,73,.45)',
       fill:false,tension:0.2,spanGaps:true},
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{
        legend:{
          display:true,position:'top',
          labels:{color:'rgba(0,212,232,.85)',boxWidth:20,padding:14,
            font:{size:11,family:"'Share Tech Mono',monospace"}}
        },
        tooltip:{...TT,callbacks:{
          label:ctx=>`${ctx.dataset.label}: $${ctx.parsed.y}B`
        }}
      },
      scales:{
        x:{grid:{color:G},ticks:{color:'rgba(0,212,232,.55)',maxTicksLimit:12,font:{size:10}}},
        y:{grid:{color:G},ticks:{color:'rgba(0,212,232,.55)',font:{size:10}},
          title:{display:true,text:'$B Notional (CTA positioning)',color:'rgba(0,212,232,.5)',font:{size:11}}}
      }
    }
  });

  // CTA — Scenario Bar
  new Chart(document.getElementById('ctaBar'),{
    type:'bar',
    plugins:[barLabels],
    data:{
      labels:['Flat','Up 1σ','Up 2σ','Down 1σ','Down 2σ','Down 2.5σ'],
      datasets:[
        {label:'1 Semana',data:[0.5,14.5,38.3,-6.8,-7.9,-6.4],
         backgroundColor:d=>d.raw>=0?'rgba(0,212,232,.55)':'rgba(0,212,232,.18)',
         borderColor:'rgba(0,212,232,.8)',borderWidth:1,borderRadius:2},
        {label:'1 Mês',data:[1.5,105.8,143.8,-7.7,-2.0,0.5],
         backgroundColor:d=>d.raw>=0?'rgba(0,212,232,.3)':'rgba(0,212,232,.1)',
         borderColor:'rgba(0,212,232,.5)',borderWidth:1,borderRadius:2},
      ]
    },
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'rgba(0,212,232,1)',boxWidth:8,font:{size:8}}},tooltip:TT},
      scales:{
        x:{grid:{color:G},ticks:{color:'rgba(0,212,232,.5)'}},
        y:{grid:{color:G},ticks:{color:'rgba(0,212,232,.5)'},
          title:{display:true,text:'$B',color:'rgba(0,212,232,.4)',font:{size:8}}}
      }
    }
  });
}

// TICKER
const td=[['SPX','__JV_SPOT__',1],['GEX','__JV_GEX_T__',0],['GAMMA FLIP','__JV_FLIP__',0],['VIX','__JV_VIX__',0],
  ['IV 30D','__JV_IV30__',0],['RV 30D','__JV_RV30__',1],['P/C','__JV_PC_T__',0],['CTA','−$68B',0],
  ['PUT WALL','__JV_PW__',0],['CALL WALL','__JV_CW__',1],['TAIL RISK','__JV_TAIL_NUM__',0],['SQUEEZE','__JV_SQ_T__',0]];
const th=td.map(([n,v,u])=>`<s>${n} <span class="${u?'up':'dn'}">${u?'▲':'▼'} ${v}</span></s>`).join('');
document.getElementById('ti').innerHTML=th+th;
</script>
</body>
</html>

"""

def _export_dashboard_html():
    """Exporta JARVIS HUD HTML standalone — 100% fiel ao jarvis_final v2."""
    if not _snapshot.get('ts'):
        return None

    m    = _snapshot.get('metrics', {})
    spot = _snapshot['spot']

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _f(k, default=0):
        v = m.get(k, default)
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    def _sym_range(v, factor=1.5, min_abs=1.0):
        """Symmetric range centred on 0 for semi-gauges."""
        mag = max(min_abs, abs(v) * factor)
        return round(-mag, 1), round(mag, 1)

    # ── Metric strings ────────────────────────────────────────────────────────
    _spot_s      = f"{spot:,.0f}"
    _flip_raw    = _f('gamma_flip')
    _flip_s      = f"{_flip_raw:,.0f}" if _flip_raw else "N/A"
    _flip_num    = round(_flip_raw)
    _spot_r10    = round(spot / 10) * 10
    _flip_r10    = round(_flip_num / 10) * 10

    _gex_raw     = _f('gex_net_bn')
    _gex_sign    = "\u2212" if _gex_raw < 0 else "+"
    _gex_s       = f"{_gex_sign}{abs(_gex_raw):.1f}B"
    _gex_t       = f"{_gex_sign}${abs(_gex_raw):.1f}B"

    _pc_raw      = _f('pc_ratio')
    _pc_s        = f"{_pc_raw:.2f}\u00d7" if _pc_raw > 0 else 'N/D'

    _ivrv_raw    = _f('iv_rv_pp')
    _ivrv_s      = f"{_ivrv_raw:+.1f}pp"
    _ivrv_prem_s = f"{_ivrv_raw:+.2f}%"

    _sq_raw      = _f('squeeze_score')
    _sq_s        = f"{_sq_raw:.0f}/100"
    _sq_int_s    = f"{_sq_raw:.0f}"

    _tail_raw    = _f('tail_score')
    _tail_s      = f"{_tail_raw:.1f}/100"
    _tail_num_s  = f"{_tail_raw:.1f}"
    _tail_int_s  = f"{_tail_raw:.0f}"

    _iv30_raw    = _f('iv_30d')
    _rv30_raw    = _f('rv_30d')
    _iv30_s      = f"{_iv30_raw*100:.2f}%"
    _rv30_s      = f"{_rv30_raw*100:.2f}%"

    _cw_raw      = _f('call_wall')
    _pw_raw      = _f('put_wall')
    _cw_s        = f"{_cw_raw:,.0f}" if _cw_raw else "N/A"
    _pw_s        = f"{_pw_raw:,.0f}" if _pw_raw else "N/A"

    # ── JS gauge values ───────────────────────────────────────────────────────
    _frag_raw    = _f('fragility')
    _frag_v      = round(_frag_raw * 100, 2) if abs(_frag_raw) <= 1.0 else round(abs(_frag_raw), 2)
    _move_raw    = _f('daily_move')
    _move_v      = round(abs(_move_raw) * 100, 2) if abs(_move_raw) <= 1.0 else round(abs(_move_raw), 2)
    _ivrv_v      = round(abs(_ivrv_raw), 2)
    _sq_v        = round(_sq_raw, 1)
    _tail_v      = round(_tail_raw, 1)

    # ── Greek semi-gauge values (real BBG) ────────────────────────────────────
    _delta_v     = round(_f('delta_bn'), 2)
    _delta_min, _delta_max = _sym_range(_delta_v, factor=1.5, min_abs=5.0)
    _vanna_v     = round(_f('vanna_bn'), 3)
    _vanna_min, _vanna_max = _sym_range(_vanna_v, factor=2.0, min_abs=0.5)
    _charm_v     = round(_f('charm_bn'), 3)
    _charm_min, _charm_max = _sym_range(_charm_v, factor=2.0, min_abs=0.2)
    _gex_semi    = round(_gex_raw, 2)

    # ── Apply replacements ────────────────────────────────────────────────────
    _html = _JARVIS_EXPORT_TEMPLATE
    _html = _html.replace('__JV_SPOT__',       _spot_s)
    _html = _html.replace('__JV_FLIP__',       _flip_s)
    _html = _html.replace('__JV_GEX__',        _gex_s)
    _html = _html.replace('__JV_PC__',         _pc_s)
    _html = _html.replace('__JV_IVRV__',       _ivrv_s)
    _html = _html.replace('__JV_SQ__',         _sq_s)
    _html = _html.replace('__JV_TAIL__',       _tail_s)
    _html = _html.replace('__JV_IV30__',       _iv30_s)
    _html = _html.replace('__JV_RV30__',       _rv30_s)
    _html = _html.replace('__JV_IVRV_PREM__',  _ivrv_prem_s)
    _html = _html.replace('__JV_CW__',         _cw_s)
    _html = _html.replace('__JV_PW__',         _pw_s)
    _html = _html.replace('__JV_TAIL_INT__',   _tail_int_s)
    _html = _html.replace('__JV_TAIL_NUM__',   _tail_num_s)
    _html = _html.replace('__JV_TAIL_PCT__',   _tail_num_s)
    _html = _html.replace('__JV_SQ_NUM__',     _sq_int_s)
    _html = _html.replace('__JV_SQ_PCT__',     _sq_int_s)
    _html = _html.replace('__JV_V_FRAG__',     str(_frag_v))
    _html = _html.replace('__JV_V_IVRV__',     str(_ivrv_v))
    _html = _html.replace('__JV_V_MOVE__',     str(_move_v))
    _html = _html.replace('__JV_V_TAIL__',     str(_tail_v))
    _html = _html.replace('__JV_V_SQ__',       str(_sq_v))
    _html = _html.replace('__JV_FLIP_NUM__',   str(_flip_num))
    _html = _html.replace('__JV_SPOT_R10__',   str(_spot_r10))
    _html = _html.replace('__JV_FLIP_R10__',   str(_flip_r10))
    _html = _html.replace('__JV_GEX_T__',      _gex_t)
    _html = _html.replace('__JV_PC_T__',       _pc_s)
    _html = _html.replace('__JV_SQ_T__',       _sq_s)
    # ── Semantic colors for CMD strip ─────────────────────────────────────────
    _c_gex   = 'rgba(0,212,232,1)' if _gex_raw >= 0 else 'rgba(248,81,73,.9)'
    _c_ivrv  = 'rgba(248,81,73,.95)' if _ivrv_raw > 2 else ('rgba(245,166,35,.9)' if _ivrv_raw > 0 else 'rgba(0,212,232,.85)')
    _c_sq    = 'rgba(248,81,73,1)' if _sq_raw > 70 else ('rgba(245,166,35,.95)' if _sq_raw > 40 else 'rgba(0,212,232,.85)')
    _c_tail  = 'rgba(248,81,73,1)' if _tail_raw > 70 else ('rgba(245,166,35,.95)' if _tail_raw > 40 else 'rgba(0,212,232,.85)')
    _html = _html.replace('__JV_C_GEX__',    _c_gex)
    _html = _html.replace('__JV_C_IVRV__',   _c_ivrv)
    _html = _html.replace('__JV_C_SQ__',     _c_sq)
    _html = _html.replace('__JV_C_TAIL__',   _c_tail)
    # Timestamp
    import datetime as _dt
    _ts_str = _snapshot.get('ts', _dt.datetime.now().strftime('%Y-%m-%d %H:%M'))
    _html = _html.replace('__JV_TS__', str(_ts_str)[:16])
    # Greek semi-gauges (real BBG)
    _html = _html.replace('__JV_V_GEX_SEMI__',  str(_gex_semi))
    _html = _html.replace('__JV_V_DELTA__',      str(_delta_v))
    _html = _html.replace('__JV_V_DELTA_MIN__',  str(_delta_min))
    _html = _html.replace('__JV_V_DELTA_MAX__',  str(_delta_max))
    _html = _html.replace('__JV_V_VANNA__',      str(_vanna_v))
    _html = _html.replace('__JV_V_VANNA_MIN__',  str(_vanna_min))
    _html = _html.replace('__JV_V_VANNA_MAX__',  str(_vanna_max))
    _html = _html.replace('__JV_V_CHARM__',      str(_charm_v))
    _html = _html.replace('__JV_V_CHARM_MIN__',  str(_charm_min))
    _html = _html.replace('__JV_V_CHARM_MAX__',  str(_charm_max))
    # ── Gamma levels chart numeric values ─────────────────────────────────────
    _spot_num    = round(spot)
    _cw_num      = round(_cw_raw) if _cw_raw else round(spot * 1.03)
    _pw_num      = round(_pw_raw) if _pw_raw else round(spot * 0.97)
    _est_move_5d = round(spot * (_iv30_raw if _iv30_raw > 0 else 0.15) * (5/252)**0.5)
    _est_up_num  = round(spot) + _est_move_5d
    _est_dn_num  = round(spot) - _est_move_5d
    _html = _html.replace('__JV_SPOT_NUM__',   str(_spot_num))
    _html = _html.replace('__JV_CW_NUM__',     str(_cw_num))
    _html = _html.replace('__JV_PW_NUM__',     str(_pw_num))
    _html = _html.replace('__JV_EST_UP_NUM__', str(_est_up_num))
    _html = _html.replace('__JV_EST_DN_NUM__', str(_est_dn_num))

    # Flow score — 8 real BBG components
    import json as _json
    _flow_data = _json.dumps([
        round(_f('z_cta'), 2),
        round(_f('z_dealer'), 2),
        round(_f('z_volctrl'), 2),
        round(_f('z_rp'), 2),
        round(_f('z_leveraged'), 2),
        round(_f('z_passive_etf'), 2),
        round(_f('z_buyback'), 2),
        round(_f('z_cot'), 2),
    ])
    _flow_w_data = _json.dumps([
        round(_f('w_cta', 0), 4),
        round(_f('w_dealer', 0), 4),
        round(_f('w_volctrl', 0), 4),
        round(_f('w_rp', 0), 4),
        round(_f('w_leveraged', 0), 4),
        round(_f('w_passive_etf', 0), 4),
        round(_f('w_buyback', 0), 4),
        round(_f('w_cot', 0), 4),
    ])
    _html = _html.replace('[__JV_FLOW_DATA__]', _flow_data)
    _html = _html.replace('[__JV_FLOW_W_DATA__]', _flow_w_data)

    # ── SKEW 25d (put-call IV spread, real BBG) ───────────────────────────────
    _skew_v = round(_f('skew_25d', 0), 2)
    _html = _html.replace('__JV_V_SKEW__', str(_skew_v))

    # ── FLOW SCORE total (from fp_score via BBG) ──────────────────────────────
    _flow_score_v = round(_f('flow_score_total', 50), 1)
    _html = _html.replace('__JV_V_FLOW__', str(_flow_score_v))

    # ── VIX current (fetched from BBG time series, last value) ────────────────
    _vix_raw = _f('vix', 0)
    _vix_s = f"{_vix_raw:.1f}" if _vix_raw > 0 else 'N/D'
    _html = _html.replace('__JV_VIX__', _vix_s)

    # ── Gamma Squeeze component bars (real BBG-derived) ───────────────────────
    import json as _json2
    _sq_comps_raw = m.get('squeeze_components', {})
    _sq_comps_list = []
    _comp_order = [
        ('vol_premium',     'Prêmio de Vol (IV−RV)', 20),
        ('flip_proximity',  'Distância Gamma Flip',   25),
        ('pc_ratio',        'P/C OI Ratio',           25),
        ('gex',             'GEX Negatividade',       30),
    ]
    for _ck, _cn, _cm in _comp_order:
        _cv = _sq_comps_raw.get(_ck, {})
        _cs = float(_cv.get('score', 0)) if isinstance(_cv, dict) else 0.0
        _ci = min(1.0, max(0.1, _cs / _cm)) if _cm > 0 else 0.5
        _sq_comps_list.append({'n': _cn, 's': round(_cs, 1), 'm': _cm, 'i': round(_ci, 2)})
    _html = _html.replace('__JV_SQ_COMPS__', _json2.dumps(_sq_comps_list))

    return _html



def _collect_widget_content(widget):
    """Recursivamente extrai conteúdo (Plotly figs + HTML) de um widget tree."""
    items = []
    if isinstance(widget, go.FigureWidget):
        items.append({'type': 'plotly', 'data': go.Figure(widget)})
    elif isinstance(widget, wd.HTML):
        val = widget.value.strip()
        if val:
            items.append({'type': 'html', 'data': val})
    elif isinstance(widget, wd.Output):
        # Output widgets contem matplotlib — não é fácil capturar retroativamente
        items.append({'type': 'html', 'data': '<p><i>[Conteúdo de Output widget — ver aba Exposições no dashboard]</i></p>'})
    elif hasattr(widget, 'children'):
        for child in widget.children:
            items.extend(_collect_widget_content(child))
    return items


def _capture_matplotlib_figures(func, *args, **kwargs):
    """Executa func que gera plt.show() e captura todas as figuras como base64 PNGs."""
    import io, base64
    plt.close('all')
    _orig_show = plt.show
    _figs_collected = []

    def _capture_show(*a, **kw):
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        _figs_collected.append(base64.b64encode(buf.read()).decode())
        plt.close(fig)

    plt.show = _capture_show
    try:
        func(*args, **kwargs)
    finally:
        plt.show = _orig_show
    return [{'type': 'matplotlib', 'data': b64} for b64 in _figs_collected]


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 8 — CALLBACK PRINCIPAL E MONTAGEM DO DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

# Widget definitions (global)
ticker_w = wd.Text(value='SPX Index', description='Ativo:',
                   layout={'width': '250px'})
dte_w = wd.IntRangeSlider(value=[0, 30], min=0, max=90, step=1,
                          description='DTE (dias):',
                          layout={'width': '400px'})
mny_w = wd.FloatRangeSlider(value=[-0.05, 0.05], min=-0.30, max=0.30,
                            step=0.01, readout_format='.0%',
                            description='% MNY:',
                            layout={'width': '400px'})
run_btn = wd.Button(description='Gerar Análise Completa',
                    button_style='success', icon='cogs')
spx_pred_w = wd.Checkbox(value=False, description='Incluir Previsão SPX',
                         indent=False, layout={'width': '250px'})
flow_pred_w = wd.Checkbox(value=True, description='Incluir Flow Predictor',
                          indent=False, layout={'width': '250px'})
disp_w = wd.Checkbox(value=False, description='Incluir Dispersão',
                     indent=False, layout={'width': '250px'})
cta_weight_w = wd.ToggleButton(value=False, description='CTA: Peso por Janela',
                                tooltip='Ativo: sinais longos têm mais peso (estilo GS/BofA). '
                                        'Inativo: peso igual entre todas as janelas.',
                                button_style='info', icon='balance-scale',
                                layout={'width': '220px'})
cot_type_w = wd.Dropdown(
    options=list(COT_CONTRACTS.keys()),
    value='Equity Indices', description='COT Categoria:',
    layout={'width': '280px'})
cot_contract_w = wd.SelectMultiple(
    description='Contratos:', layout={'min_width': '300px', 'height': '80px'})
cot_trader_w = wd.Dropdown(
    options=['Total', 'Managed Money', 'Commercial', 'Non-Commercial',
             'Swap Dealers', 'Leveraged Funds', 'Asset Manager',
             'Dealer Intermediary', 'Other Reportables'],
    value='Total', description='Trader Type:',
    layout={'width': '280px'})
cot_start_w = wd.DatePicker(
    value=pd.Timestamp.now().date() - pd.Timedelta(days=2*365),
    description='COT Start:', layout={'width': '260px'})
cot_end_w = wd.DatePicker(
    value=pd.Timestamp.now().date(),
    description='COT End:', layout={'width': '260px'})

rebal_date_w = wd.DatePicker(
    value=_last_spx_rebal_date().date(),
    description='Último Rebalance:', layout={'width': '300px'})


def _update_cot_contracts(change=None):
    opts = COT_CONTRACTS.get(cot_type_w.value, [])
    cot_contract_w.options = opts
    if opts:
        cot_contract_w.value = (opts[0][1],)

cot_type_w.observe(_update_cot_contracts, names='value')
_update_cot_contracts()

out_main = wd.Output()
out_cot_reload = wd.Output()

# Botões de recarga parcial
cot_reload_btn = wd.Button(description='⟳ Recarregar COT',
                           button_style='info', icon='refresh',
                           layout={'width': '180px'})
etf_reload_btn = wd.Button(description='⟳ Recarregar ETFs',
                           button_style='info', icon='refresh',
                           layout={'width': '180px'})


def _reload_cot(_):
    """Recarrega apenas dados COT e exibe resultado."""
    with out_cot_reload:
        clear_output(wait=True)
        display(wd.HTML(DASH_CSS + "<div class='mm-dash mm-loading'>⏳ Recarregando COT...</div>"))
        ticker = ticker_w.value.strip() or 'SPX Index'
        _cot_start = cot_start_w.value.strftime('%Y%m%d') if cot_start_w.value else '-2Y'
        _cot_end = cot_end_w.value.strftime('%Y%m%d') if cot_end_w.value else '0D'
        try:
            cot_df = safe_fetch_cot(ticker, start=_cot_start, end=_cot_end)
            sel_cots = list(cot_contract_w.value)
            sel_df = None
            if sel_cots:
                try:
                    raw = fetch_cot_data(
                        sel_cots[0] if len(sel_cots) == 1 else sel_cots,
                        start=_cot_start, end=_cot_end)
                    if raw is not None and not raw.empty:
                        sel_df = aggregate_cot(raw)
                except Exception as e:
                    print(f"⚠️ COT selected: {e}")
            clear_output(wait=True)
            children = [wd.HTML(
                f"<div class='mm-dash'><div class='mm-card'>"
                f"<h3>COT — Recarga Rápida</h3>"
                f"<p>Dados agregados (todas trader types somadas). "
                f"Report type: auto (tenta disaggregated → tff → legacy)</p>"
                f"</div></div>")]
            ok, fut = has_cot(ticker)
            if ok and cot_df is not None and not cot_df.empty:
                stats = cot_summary_stats(cot_df)
                children.append(wd.HTML(f"<p>Futures: <b>{fut}</b> — {len(cot_df)} registros</p>"))
                children.append(fp_grid_cot_stats(stats))
                seas = cot_seasonality(cot_df)
                children.append(wd.HBox([
                    fp_plot_positions_basket(cot_df),
                    fp_plot_dispersion(seas, cot_df)]))
                children.append(fp_plot_long_short_net(cot_df))
            elif ok:
                children.append(wd.HTML(f"<p>COT disponível para {fut}, mas sem dados.</p>"))
            if sel_df is not None and not sel_df.empty:
                sel_label = ', '.join(sel_cots)
                children.append(wd.HTML(f"<hr><h4>COT: {sel_label} — {len(sel_df)} registros</h4>"))
                children.append(fp_plot_positions_basket(sel_df))
            display(wd.VBox(children))
        except Exception as e:
            clear_output(wait=True)
            print(f"Erro COT: {e}")
            import traceback as _tb_mod; _tb_mod.print_exc()


def _reload_etfs(_):
    """Recarrega apenas fluxo de ETFs alavancados."""
    with out_cot_reload:
        clear_output(wait=True)
        display(wd.HTML(DASH_CSS + "<div class='mm-dash mm-loading'>⏳ Recarregando ETFs...</div>"))
        ticker = ticker_w.value.strip() or 'SPX Index'
        try:
            prices, lr = fetch_historical(ticker)
            nz = lr[lr != 0]
            dr = float(nz.iloc[-1]) if len(nz) > 0 else 0
            flows, total = compute_leveraged_flows(dr)
            clear_output(wait=True)
            html = (
                f"<div class='mm-dash'><div class='mm-card'>"
                f"<h4>ETFs Alavancados (Retorno diário: {dr*100:+.2f}%)</h4>" +
                flows[['Leverage', 'AUM_$', 'Rebalance_$', 'Direção']]
                .style.format({'AUM_$': '${:,.0f}', 'Rebalance_$': '${:,.0f}'})
                .to_html() +
                f"<p><b>Fluxo Direcional Total: ${total:,.0f}</b></p>"
                f"</div></div>")
            display(wd.HTML(html))
        except Exception as e:
            clear_output(wait=True)
            print(f"Erro ETF: {e}")


cot_reload_btn.on_click(_reload_cot)
etf_reload_btn.on_click(_reload_etfs)


def run_analysis(_):
    """Callback principal: busca dados, calcula tudo, monta dashboard."""
    with out_main:
        clear_output(wait=True)
        loading = wd.HTML(DASH_CSS + "<div class='mm-dash mm-loading'>⏳ Inicializando...</div>")
        display(loading)

        ticker = ticker_w.value.strip() or 'SPX Index'
        min_dte, max_dte = dte_w.value
        mny_low, mny_high = mny_w.value

        try:
            # ── 1. Dados de Mercado ──────────────────────────────────────
            loading.value = "<h4>1/16: Buscando dados de mercado...</h4>"
            mkt = fetch_market_data(ticker)
            spot = mkt['spot']
            iv_30d = mkt['iv_30d']
            rv_30d = mkt['rv_30d']
            skew = mkt['skew']
            avg_vol = mkt['avg_dollar_volume']
            rfr = mkt.get('risk_free_rate', 0.0)
            move_index = mkt.get('move_index', np.nan)
            print(f"[MKT] risk-free rate = {rfr:.4f} (USGG3M), MOVE = {move_index}")

            if pd.isna(spot):
                raise ValueError(f"Spot inválido para {ticker}")

            # ── 2. Histórico + Modelo de Risco ───────────────────────────
            loading.value = "<h4>2/16: Modelagem de risco (t-Student)...</h4>"
            prices, log_returns = fetch_historical(ticker)
            risk = fit_risk_model(log_returns)

            # ── 3. Cadeia de Opções ──────────────────────────────────────
            loading.value = "<h4>3/16: Buscando cadeia de opções...</h4>"
            df, from_strike, to_strike = fetch_options_chain(
                ticker, spot, min_dte, max_dte, mny_low, mny_high)

            # ── 4. Gregas + Exposições ───────────────────────────────────
            loading.value = f"<h4>4/16: Calculando gregas para {len(df)} opções...</h4>"
            greeks_now = calculate_all_greeks(
                spot, df.Strike.values, df.IV.values, df.Tte.values, df.Type.values, r=rfr)
            agg = compute_strike_exposures(df, greeks_now, spot)
            call_wall, put_wall = compute_walls(agg)

            if call_wall == put_wall and call_wall is not None:
                print(f"⚠️ Call Wall = Put Wall = {call_wall:,.0f}")

            # ── 5. Curvas Modelo ─────────────────────────────────────────
            loading.value = "<h4>5/16: Calculando curvas modelo (100 níveis × 7 gregas)...</h4>"
            levels = np.linspace(from_strike, to_strike, 100)
            model_curves = compute_model_curves(df, levels, r=rfr)

            # Flip points
            flip_points = {}
            for cfg in GREEK_CONFIGS:
                curve = model_curves[cfg['name']]
                flip_points[cfg['name']] = calculate_flip(levels, curve)

            gamma_flip = flip_points.get('Gamma')
            gamma_curve = model_curves['Gamma']

            # ── 6. Matrizes de Sensibilidade ─────────────────────────────
            loading.value = "<h4>6/16: Matrizes de sensibilidade (7×5×7)...</h4>"
            sens_matrices = compute_sensitivity_matrices(df, spot, r=rfr)

            # ── 7. Monte Carlo ───────────────────────────────────────────
            loading.value = "<h4>7/16: Simulação Monte Carlo (10k cenários, 5 dias)...</h4>"
            mc_n_days = 5
            mc_pnl, mc_prices = run_monte_carlo(spot, df, risk, n_days=mc_n_days, r=rfr)

            # ── 8. Curvas de P&L ─────────────────────────────────────────
            loading.value = "<h4>8/16: Curvas de P&L e hedge demand...</h4>"
            pnl_curves = compute_pnl_curves(greeks_now, df, spot, levels, skew, r=rfr)

            # ── 9. Rebalanceamento ETFs ──────────────────────────────────
            loading.value = "<h4>9/16: Fluxo de rebalanceamento ETFs passivos...</h4>"
            try:
                _rebal_dt = rebal_date_w.value
                etf_flows, etf_summary, etf_start = compute_full_etf_flows(
                    start_date_override=_rebal_dt)
                etf_ok = True
            except Exception:
                etf_flows, etf_summary, etf_start, etf_ok = {}, None, None, False

            # ── 10. ETFs Alavancados ─────────────────────────────────────
            loading.value = "<h4>10/16: Fluxo de ETFs alavancados...</h4>"
            try:
                # Usar último retorno não-zero (evitar 0 de weekend/feriado)
                nz_rets = log_returns[log_returns != 0]
                daily_ret = float(nz_rets.iloc[-1]) if len(nz_rets) > 0 else 0
                lev_flows, lev_total = compute_leveraged_flows(daily_ret)
                lev_ok = True
            except Exception:
                lev_flows, lev_total, lev_ok = None, 0, False
                daily_ret = 0

            # ── 11. Previsão SPX (opcional — pesado) ─────────────────────
            loading.value = "<h4>11/16: Previsão de rebalanceamento SPX...</h4>"
            spx_pred_ok = False
            top_in_spx, top_out_spx = None, None
            if spx_pred_w.value and HAS_SKLEARN:
                try:
                    top_in_spx, top_out_spx, _ = build_spx_prediction()
                    spx_pred_ok = True
                except Exception as pred_err:
                    print(f"⚠️ Previsão SPX falhou: {pred_err}")

            # ── 12. Flow Predictor — Histórico + Buyback ─────────────────
            fp_ok = False
            fp_flow_hist = pd.DataFrame()
            fp_buyback = {'daily_est': 0, 'pct_adv_est': 0,
                          'confidence': 'none', 'announced': 0}
            fp_score = None
            fp_cot_df = None
            fp_cot_stats = pd.Series(dtype=float)
            fp_selected_cot_df = None

            if flow_pred_w.value:
                loading.value = "<h4>12/16: Flow Predictor — histórico + buyback...</h4>"
                try:
                    fp_flow_hist = build_flow_history(ticker, lookback=252)
                    fp_today_flow = (compute_leveraged_flow_simple(
                        float(fp_flow_hist['Return'].iloc[-1]))
                        if not fp_flow_hist.empty else 0)
                except Exception as fp_err:
                    print(f"⚠️ Flow hist: {fp_err}")
                    fp_today_flow = 0

                # Buyback: para índices, usar agregação por membros
                try:
                    if ticker.strip().endswith('Index'):
                        bb_df = estimate_index_buyback_flow(ticker, top_n=30)
                        fp_buyback_daily = float(bb_df['daily_est'].sum()) if (
                            not bb_df.empty and 'daily_est' in bb_df.columns) else 0
                        fp_buyback = {
                            'daily_est': fp_buyback_daily,
                            'pct_adv_est': 0,
                            'confidence': 'estimated',
                            'announced': float(bb_df['buyback'].sum()) if (
                                not bb_df.empty and 'buyback' in bb_df.columns) else 0,
                        }
                    else:
                        fp_buyback = estimate_buyback_flow(ticker)
                        fp_buyback_daily = fp_buyback.get('daily_est', 0)
                except Exception as fp_err:
                    print(f"⚠️ Buyback: {fp_err}")
                    fp_buyback_daily = 0

                # Blackout window: buscar datas de earnings e calcular % em restrição
                fp_earnings_df = pd.DataFrame()
                fp_blackout_pct = 0.0
                fp_blackout_n = 0
                fp_blackout_total = 0
                fp_blackout_curve = pd.DataFrame()
                try:
                    loading.value = "<h4>12b/16: Blackout window — earnings dates...</h4>"
                    fp_earnings_df = fetch_earnings_dates(ticker if ticker.strip().endswith('Index') else 'SPX Index')
                    if not fp_earnings_df.empty:
                        fp_blackout_pct, fp_blackout_n, fp_blackout_total = blackout_pct_today(fp_earnings_df)
                        fp_blackout_curve = compute_blackout_curve(fp_earnings_df, n_days_forward=365)
                        # Ajustar buyback diário pela janela de blackout
                        fp_buyback['blackout_pct'] = fp_blackout_pct
                        fp_buyback['blackout_n'] = fp_blackout_n
                        fp_buyback['blackout_total'] = fp_blackout_total
                        fp_buyback['daily_est_open'] = fp_buyback.get('daily_est', 0)
                        fp_buyback['daily_est'] = fp_buyback['daily_est_open'] * (1 - fp_blackout_pct)
                        fp_buyback_daily = fp_buyback['daily_est']
                except Exception as bo_err:
                    print(f"⚠️ Blackout: {bo_err}")

                # ── 13. Flow Predictor — COT ─────────────────────────────
                loading.value = "<h4>13/16: Flow Predictor — COT...</h4>"
                cot_ok_fp, cot_fut_fp = has_cot(ticker)
                cot_net_change = 0
                history_cot = None

                _cot_start = (cot_start_w.value.strftime('%Y%m%d')
                              if cot_start_w.value else '-2Y')
                _cot_end = (cot_end_w.value.strftime('%Y%m%d')
                            if cot_end_w.value else '0D')

                if cot_ok_fp:
                    try:
                        fp_cot_df = safe_fetch_cot(ticker,
                                                   start=_cot_start,
                                                   end=_cot_end)
                        if fp_cot_df is not None and not fp_cot_df.empty:
                            fp_cot_stats = cot_summary_stats(fp_cot_df)
                            # Pega net change de qualquer coluna disponível
                            _net_col = None
                            for _nc in ('Positions - Net', 'Positions'):
                                if _nc in fp_cot_df.columns:
                                    _net_col = _nc
                                    break
                            if _net_col:
                                net = fp_cot_df[_net_col].dropna()
                                if len(net) >= 2:
                                    cot_net_change = float(
                                        net.iloc[-1] - net.iloc[-2])
                                    history_cot = net.diff().dropna()
                    except Exception:
                        pass

                # COT de contratos selecionados
                selected_cots = list(cot_contract_w.value)
                if selected_cots:
                    try:
                        sel_df = fetch_cot_data(
                            selected_cots[0] if len(selected_cots) == 1
                            else selected_cots,
                            start=_cot_start, end=_cot_end)
                        if not sel_df.empty:
                            fp_selected_cot_df = aggregate_cot(sel_df)
                    except Exception:
                        pass

                # ── 14. Flow Score Combinado ─────────────────────────────
                loading.value = "<h4>14/16: Calculando flow score...</h4>"

                # Dealer/MM hedging flow (baseado em GEX)
                fp_dealer_flow = 0
                try:
                    _gex_per_pt = (greeks_now['gamma']
                                   * np.where(df.Type.values == 'Call', 1, -1)
                                   * df['OI'].values * 100).sum()
                    _daily_chg = spot * daily_ret
                    fp_dealer_flow = compute_dealer_hedging_flow(
                        _gex_per_pt, _daily_chg, spot)
                    print(f"[FLOW] Dealer flow: ${fp_dealer_flow:,.0f} "
                          f"(GEX/pt={_gex_per_pt:,.0f}, ΔS={_daily_chg:,.1f})")
                except Exception as e:
                    print(f"⚠️ Dealer flow: {e}")

                # Market Maker VaR by individual book
                fp_mm_var = []
                fp_mm_var_totals = {}
                fp_vol_data = {'total_adc': OPTIONS_TOTAL_ADC, 'source': 'fallback',
                               'call_vol': 0, 'put_vol': 0, 'pc_ratio': 0}
                try:
                    fp_vol_data = fetch_options_volume_bql(ticker)
                    print(f"[FLOW] Options volume: {fp_vol_data['total_adc']:,.0f} "
                          f"(source={fp_vol_data['source']})")
                except Exception as e:
                    print(f"⚠️ Options volume BQL: {e}")
                try:
                    _oi_total = df['OI'].sum()
                    _theta_total = (greeks_now['theta'] * df['OI'].values * 100).sum()
                    fp_mm_var, fp_mm_var_totals = estimate_mm_var_by_book(
                        _gex_per_pt, spot, risk, _oi_total)
                    # Fill theta per MM
                    for mm in fp_mm_var:
                        mm['daily_theta'] = _theta_total * mm['share']
                    print(f"[FLOW] MM VaR: {len(fp_mm_var)} MMs, "
                          f"VaR95=${fp_mm_var_totals.get('pnl_var95', 0):,.0f}")
                except Exception as e:
                    print(f"⚠️ MM VaR: {e}")

                # Vol control fund flows (5%, 10%, 15%)
                fp_volctrl = {'total': 0, 'detail': {}}
                fp_vc_scenarios = []
                fp_combined_scenarios = []
                try:
                    _rv_window = 21
                    _rets = log_returns.iloc[-_rv_window * 2:]
                    _rv_cur = _rets.iloc[-_rv_window:].std() * np.sqrt(252)
                    _rv_prev = _rets.iloc[-_rv_window * 2:-_rv_window].std() * np.sqrt(252)
                    fp_volctrl = compute_vol_control_flow(_rv_cur, _rv_prev)
                    fp_vc_scenarios = compute_vol_control_scenarios(_rv_cur)
                    print(f"[FLOW] Vol ctrl: ${fp_volctrl['total']:,.0f} "
                          f"(RV cur={_rv_cur:.2%}, prev={_rv_prev:.2%})")
                except Exception as e:
                    print(f"⚠️ Vol ctrl: {e}")

                # CTA trend following flow
                fp_cta = {'flow': 0, 'trend_today': 0, 'pos_today': 0, 'pos_prev': 0}
                fp_cta_scenarios_1w = []
                fp_cta_scenarios_1m = []
                fp_cta_pivots = []
                fp_cta_hist = pd.DataFrame()
                try:
                    # Reconstruir preços a partir de log returns
                    _px_series = np.exp(np.cumsum(log_returns))
                    _px_series = _px_series * (spot / _px_series.iloc[-1])
                    _cta_rv = log_returns.iloc[-63:].std() * np.sqrt(252) if len(log_returns) >= 63 else rv_30d
                    fp_cta = compute_cta_flow(_px_series, _cta_rv)
                    print(f"[FLOW] CTA: ${fp_cta['flow']:,.0f} "
                          f"(trend={fp_cta['trend_today']:+.3f}, "
                          f"pos={fp_cta['pos_today']:+.3f}→{fp_cta['pos_prev']:+.3f})")
                except Exception as e:
                    print(f"⚠️ CTA flow: {e}\n{traceback.format_exc()}")

                # CTA: scenarios, pivots, historical
                try:
                    if len(_px_series) < 201:
                        _px_series = np.exp(np.cumsum(log_returns))
                        _px_series = _px_series * (spot / _px_series.iloc[-1])
                except Exception:
                    _px_series = np.exp(np.cumsum(log_returns))
                    _px_series = _px_series * (spot / _px_series.iloc[-1])
                try:
                    if '_cta_rv' not in dir() or _cta_rv < 1e-6:
                        _cta_rv = log_returns.iloc[-63:].std() * np.sqrt(252) if len(log_returns) >= 63 else rv_30d
                except Exception:
                    _cta_rv = rv_30d
                try:
                    print(f"[FLOW] CTA GS: px_series len={len(_px_series)}, rv={_cta_rv:.4f}")
                    fp_cta_scenarios_1w = compute_cta_scenario_flows(
                        _px_series, _cta_rv, spot, horizon_days=5)
                    print(f"[FLOW] CTA scenarios 1W: {len(fp_cta_scenarios_1w)}")

                    fp_cta_scenarios_1m = compute_cta_scenario_flows(
                        _px_series, _cta_rv, spot, horizon_days=21)
                    print(f"[FLOW] CTA scenarios 1M: {len(fp_cta_scenarios_1m)}")

                    fp_cta_pivots = compute_cta_pivot_levels(_px_series, spot, _cta_rv)
                    print(f"[FLOW] CTA pivots: {len(fp_cta_pivots)}")

                    loading.value = "<h4>14b/16: CTA — histórico de posições...</h4>"
                    fp_cta_hist = compute_cta_historical_positions(_px_series, lookback=126)
                    print(f"[FLOW] CTA hist: {len(fp_cta_hist)} rows")
                except Exception as e:
                    print(f"⚠️ CTA: {e}\n{traceback.format_exc()}")

                # Risk Parity flow
                fp_rp = {'total': 0, 'detail': {}, 'eq_alloc_new': 0, 'eq_alloc_old': 0}
                try:
                    _rv_window = 21
                    _rets_rp = log_returns.iloc[-_rv_window * 2:]
                    _rv_eq_cur = _rets_rp.iloc[-_rv_window:].std() * np.sqrt(252)
                    _rv_eq_prev = _rets_rp.iloc[-_rv_window * 2:-_rv_window].std() * np.sqrt(252)
                    # Bond vol from MOVE Index (yield vol → price vol, ~7yr duration)
                    _bond_vol = (move_index / 10000) * 7 if move_index and move_index > 0 else None
                    fp_rp = compute_risk_parity_flow(_rv_eq_cur, _rv_eq_prev,
                                                     rv_bonds=_bond_vol, rv_bonds_prev=_bond_vol)
                    print(f"[FLOW] Risk Parity: ${fp_rp['total']:,.0f} "
                          f"(eq_alloc={fp_rp['eq_alloc_new']:.1%}→{fp_rp['eq_alloc_old']:.1%})")
                except Exception as e:
                    print(f"⚠️ Risk Parity: {e}")

                # Combined flow scenarios (vol spike → all components)
                try:
                    _px_s = _px_series if '_px_series' in dir() else None
                    _gex = _gex_per_pt if '_gex_per_pt' in dir() else 0
                    _oi_100 = df['OI'].values * 100.0
                    _vanna_not = np.nansum(greeks_now['vanna'] * _oi_100) * spot
                    _vega_not = np.nansum(greeks_now['vega'] * _oi_100)
                    _charm_not = np.nansum(greeks_now['charm'] * _oi_100) * spot / 365.0
                    fp_combined_scenarios = compute_combined_flow_scenarios(
                        _rv_cur, prices=_px_s, gex_per_pt=_gex, spot=spot,
                        vanna_notional=_vanna_not, vega_notional=_vega_not,
                        charm_notional=_charm_not)
                except Exception as e:
                    print(f"⚠️ Combined scenarios: {e}")

                # Passive ETF flow — net rebalancing from VOO/SPY/IVV
                fp_passive_etf_flow = 0
                try:
                    if etf_ok and etf_flows:
                        combo = etf_flows.get('Combined', pd.DataFrame())
                        if not combo.empty and 'Flow_$' in combo.columns:
                            fp_passive_etf_flow = float(combo['Flow_$'].sum())
                            print(f"[FLOW] ETFs Passivos: ${fp_passive_etf_flow:,.0f} "
                                  f"(buy={combo.loc[combo['Flow_$']>0, 'Flow_$'].sum():,.0f}, "
                                  f"sell={combo.loc[combo['Flow_$']<0, 'Flow_$'].sum():,.0f})")
                except Exception as e:
                    print(f"⚠️ ETFs Passivos: {e}")

                try:
                    lev_history = (fp_flow_hist['LevETF_Flow']
                                   if not fp_flow_hist.empty else None)
                    fp_score = compute_flow_score(
                        leveraged_flow=fp_today_flow,
                        buyback_daily=fp_buyback_daily,
                        cot_net_change=cot_net_change,
                        passive_etf_flow=fp_passive_etf_flow,
                        history_leveraged=lev_history,
                        history_cot=history_cot,
                        dealer_flow=fp_dealer_flow,
                        volctrl_flow=fp_volctrl['total'],
                        cta_flow=fp_cta['flow'],
                        rp_flow=fp_rp['total'])
                    fp_ok = True
                except Exception as fp_err:
                    print(f"⚠️ Flow score: {fp_err}")

            # ── P/C OI Ratio (standalone, always fetched from BBG) ───────
            # fp_vol_data may already be populated by Flow Predictor block;
            # if not (FP disabled or query failed), fetch it now so that
            # pc_ratio is always a real BBG value, never hardcoded.
            if not (isinstance(fp_vol_data, dict) and fp_vol_data.get('pc_ratio', 0) > 0):
                try:
                    _pc_tmp = fetch_options_volume_bql(ticker)
                    if _pc_tmp.get('pc_ratio', 0) > 0:
                        fp_vol_data = _pc_tmp
                        print(f"[PC] P/C OI ratio (standalone): {_pc_tmp['pc_ratio']:.2f}")
                except Exception as _pc_err:
                    print(f"⚠️ P/C ratio standalone fetch: {_pc_err}")

            # ── 15. Dispersion Trade + Tail Risk ─────────────────────────
            disp_result = {
                'error': None, 'disp_signal': pd.DataFrame(),
                'real_corr': pd.DataFrame(),
                'impl_corr_cboe': pd.Series(dtype=float),
                'mag7_pairs': pd.DataFrame(), 'best_2x2': [], 'best_pairs': [],
                'optimal_basket': {}, 'tail_risk': {},
                'index_returns': np.array([]),
                'hyp_test': {},
                'cor1m': pd.Series(dtype=float),
                'dspx': pd.Series(dtype=float),
                'vixeq': pd.Series(dtype=float),
            }
            disp_ok = False
            if disp_w.value:
                loading.value = "<h4>15/16: Dispersion Trade + Tail Risk (BQL)...</h4>"
                try:
                    disp_result = run_dispersion_analysis(
                        index_ticker=ticker, lookback=252)
                    if disp_result['error'] is None:
                        disp_ok = True
                    else:
                        print(f"⚠️ Dispersion: {disp_result['error']}")
                except Exception as disp_err:
                    print(f"⚠️ Dispersion: {disp_err}")

            # ── 16. Advanced Analytics (Skew, Tail Gauge, Dealer MC, OPEX) ──
            analytics = {
                'skew_df': pd.DataFrame(),
                'skew_summary': {},
                'spot_vol_up': {'current_streak': 0, 'max_streak': 0, 'total_days': 0, 'pct_up_up': 0, 'history': pd.Series(dtype=int)},
                'vix_reg': {},
                'dealer_mc': {},
                'opex_stats': {},
                'tail_score': 50, 'tail_components': {}, 'tail_interp': '',
                'dealer_scenarios': pd.DataFrame(),
                'mag8_scenarios': pd.DataFrame(),
                'vol_rebal': pd.DataFrame(),
                'gamma_vol': {},
            }
            try:
                loading.value = "<h4>16/16: Advanced Analytics...</h4>"

                # Skew monitor
                try:
                    analytics['skew_df'] = fetch_skew_metrics(ticker, lookback=252)
                    analytics['skew_summary'] = compute_skew_summary(analytics['skew_df'])
                except Exception as _sk_err:
                    print(f"⚠️ Skew: {_sk_err}")

                # Spot-Up-Vol-Up: fetch VIX
                try:
                    bq = bql.Service()
                    dt_rng = bq.func.range('-504d', '0d')
                    vix_req = bql.Request('VIX Index', {
                        'px': bq.data.px_last(fill='PREV', dates=dt_rng),
                    })
                    vix_resp = bq.execute(vix_req)
                    vix_s = _bql_ts(vix_resp[0], 'px').dropna()
                    vix_changes = vix_s.diff().dropna()
                    analytics['spot_vol_up'] = compute_spot_up_vol_up(log_returns, vix_changes)
                    analytics['vix_reg'] = compute_vix_spx_regression(log_returns, vix_changes)
                except Exception as _vix_err:
                    print(f"⚠️ VIX/Spot-Vol-Up: {_vix_err}")

                # Per-dealer MC
                try:
                    analytics['dealer_mc'] = run_dealer_monte_carlo(
                        spot, df, risk, n_sims=10000, n_days=mc_n_days, r=rfr)
                except Exception as _dmc_err:
                    print(f"⚠️ Dealer MC: {_dmc_err}")

                # OPEX stats
                try:
                    analytics['opex_stats'] = compute_opex_stats(log_returns, lookback_years=5)
                except Exception as _opex_err:
                    print(f"⚠️ OPEX: {_opex_err}")

                # Dealer scenario matrices
                try:
                    analytics['dealer_scenarios'] = compute_dealer_scenario_matrix(
                        spot, df, greeks_now)
                    analytics['mag8_scenarios'] = compute_mag8_dealer_scenarios(
                        spot, df, greeks_now)
                except Exception as _dsm_err:
                    print(f"⚠️ Dealer Scenarios: {_dsm_err}")

                # Vol Control rebalance projection
                oi100_vc = df.OI.values * 100
                cs_vc = np.where(df.Type.values == 'Call', 1, -1)
                _gex_pt = (greeks_now['gamma'] * cs_vc * oi100_vc).sum()
                _dex_vc = (greeks_now['delta'] * oi100_vc).sum()
                _vanna_not = float(np.nansum(greeks_now['vanna'] * oi100_vc) * spot)
                try:
                    analytics['vol_rebal'] = compute_vol_rebalance_projection(
                        rv_30d if pd.notna(rv_30d) else 0.15, spot,
                        gex_per_pt=_gex_pt, vanna_notional=_vanna_not, dex=_dex_vc)
                except Exception as _vr_err:
                    print(f"⚠️ Vol Rebal: {_vr_err}")

                # Tail risk gauge
                suvu_streak = analytics['spot_vol_up'].get('current_streak', 0)
                analytics['tail_score'], analytics['tail_components'], analytics['tail_interp'] = \
                    compute_tail_risk_gauge(
                        log_returns,
                        iv_30d=iv_30d if pd.notna(iv_30d) else None,
                        rv_30d=rv_30d if pd.notna(rv_30d) else None,
                        skew_summary=analytics['skew_summary'],
                        spot_vol_up_streak=suvu_streak)

            except Exception as _analytics_err:
                print(f"⚠️ Analytics: {_analytics_err}")

            # ── Gamma History (CSV database) ─────────────────────────────
            gamma_hist = pd.DataFrame()
            try:
                gamma_hist = load_gamma_history()
                if not gamma_hist.empty:
                    print(f"[GAMMA DB] {len(gamma_hist)} rows loaded "
                          f"({gamma_hist['date'].iloc[0].strftime('%Y-%m-%d')} → "
                          f"{gamma_hist['date'].iloc[-1].strftime('%Y-%m-%d')})")
                else:
                    print(f"[GAMMA DB] Empty — path={GAMMA_HISTORY_PATH}, exists={os.path.exists(GAMMA_HISTORY_PATH)}")
            except Exception as _gh_err:
                print(f"⚠️ Gamma History load: {_gh_err}")
            # Atualização do banco de dados desativada — preencher manualmente no CSV.

            clear_output(wait=True)

            # ═════════════════════════════════════════════════════════════
            # MONTAGEM DAS ABAS DO DASHBOARD
            # ═════════════════════════════════════════════════════════════

            title_html = wd.HTML(
                DASH_CSS +
                f"<div class='mm-dash'>"
                f"<div class='mm-title'>Market Maker Dashboard "
                f"<small>{ticker} @ {spot:,.2f} │ {datetime.now().strftime('%Y-%m-%d %H:%M')}</small>"
                f"</div></div>")

            # ─── ABA 1: VISÃO GERAL ─────────────────────────────────────
            total_gex = gamma_curve[np.argmin(np.abs(levels - spot))]
            # Fragilidade = GEX / (notional diário de opções SPX + futuros ES)
            # Opções: total de contratos ADV × 100 shares × spot
            _opt_adc = fp_vol_data.get('total_adc', OPTIONS_TOTAL_ADC)
            # options_notional: volume ADC em lotes × spot (sem ×100 — ADC já em unidades de share equiv.)
            _opt_notional = _opt_adc * spot                  # ~$145B @ SPX=6700, ADC=21.7M
            # Futuros ES: ADV ~400k contratos × $50/ponto × spot = ~$134B @ SPX=6700
            _fut_notional = 400_000 * 50 * spot
            _daily_flow_cap = _opt_notional + _fut_notional
            # Fragilidade = GEX ($ / 1%move) como % do fluxo diário total (opções + futuros)
            fragility = (abs(total_gex) / _daily_flow_cap * 100
                         if _daily_flow_cap > 0 else 0)

            # Diagnóstico GEX modelo completo (todos vencimentos)
            _gc_is_call = df.Type.values == 'Call'
            _gc_is_put  = df.Type.values == 'Put'
            _gc_oi100   = df['OI'].values * 100.0
            _gc_g       = greeks_now['gamma']
            _gc_calls_abs = float(np.nansum(_gc_g[_gc_is_call] * _gc_oi100[_gc_is_call] * spot**2 * 0.01)) / 1e9
            _gc_puts_abs  = float(np.nansum(_gc_g[_gc_is_put]  * _gc_oi100[_gc_is_put]  * spot**2 * 0.01)) / 1e9
            print(f"[GEX FULL] OI calls: {df[_gc_is_call].OI.sum():,.0f} | OI puts: {df[_gc_is_put].OI.sum():,.0f} | "
                  f"Expirations: {df['Tte'].nunique() if 'Tte' in df.columns else '?'} | "
                  f"Options total: {len(df)}")
            print(f"[GEX FULL] Gamma absoluto (bbg-like): {_gc_calls_abs + _gc_puts_abs:+.3f}Bn | "
                  f"Calls: {_gc_calls_abs:+.3f}Bn | Puts: {_gc_puts_abs:+.3f}Bn | "
                  f"NET (calls-puts): {_gc_calls_abs - _gc_puts_abs:+.3f}Bn | "
                  f"Curva no spot: {total_gex/1e9:+.3f}Bn")
            daily_move = implied_move_pct(iv_30d) if pd.notna(iv_30d) else 0
            vol_premium = (iv_30d - rv_30d) * 100 if pd.notna(iv_30d) and pd.notna(rv_30d) else 0

            _frag_max = max(20, round(fragility * 1.5 / 5) * 5)  # arredonda p/ múltiplo de 5
            g_frag = create_gauge(fragility, "Fragilidade GEX/Fluxo",
                                  0, _frag_max, _C['red'], "%",
                                  steps=[
                                      {'range': [0,           _frag_max * 0.25], 'color': '#1a3a2a'},
                                      {'range': [_frag_max * 0.25, _frag_max * 0.6],  'color': '#3a3520'},
                                      {'range': [_frag_max * 0.6,  _frag_max],        'color': '#3a1a1a'},
                                  ])
            _vol_hi = max(5, round(abs(vol_premium) * 1.4))
            g_vol = create_gauge(vol_premium, "Prêmio Vol (IV-RV)",
                                 -_vol_hi, _vol_hi, _C['orange'], "%")
            _skew_raw = skew * 100
            # Clamp outliers de BQL (dados anômalos excedem ±25pp)
            _skew_val = float(np.clip(_skew_raw, -25, 25))
            _skew_hi = max(15, abs(_skew_val) * 1.3)
            g_skew = create_gauge(_skew_val, "Skew (P25-C25)",
                                  -_skew_hi, _skew_hi, _C['teal'], "%")
            _move_hi = max(5, daily_move * 1.3)
            g_move = create_gauge(daily_move, "Mov. Esperado 1D",
                                  0, _move_hi, _C['green'], "%")

            # GEX curve (Plotly)
            fig_gex = go.FigureWidget()
            fig_gex.add_trace(go.Scatter(
                x=levels, y=gamma_curve / 1e10, mode='lines',
                fill='tozeroy', line_color=_C['accent'],
                fillcolor='rgba(88,166,255,0.15)', name='GEX'))
            fig_gex.add_vline(x=spot, line_dash="dash", line_color=_C['red'],
                              annotation_text=f"Spot {spot:,.0f}")
            if gamma_flip:
                fig_gex.add_vline(x=gamma_flip, line_color=_C['orange'],
                                  annotation_text=f"G-Flip {gamma_flip:,.0f}")
            fig_gex.update_layout(title="Gamma Exposure (GEX)",
                                  yaxis_title="$ Bi / 1% move",
                                  height=350, width=480, template=DASH_TEMPLATE,
                                  margin=dict(t=35, b=25), showlegend=False)

            # Return distribution (Plotly)
            fig_dist = go.FigureWidget()
            fig_dist.add_trace(go.Histogram(
                x=log_returns, histnorm='probability density',
                name='Reais', marker_color=_C['accent'], opacity=0.6,
                xbins=dict(size=log_returns.std() / 4)))
            x_pdf = np.linspace(log_returns.min(), log_returns.max(), 500)
            pdf_vals = student_t.pdf(x_pdf, risk['tdf'], risk['tloc'], risk['tscale'])
            fig_dist.add_trace(go.Scatter(
                x=x_pdf, y=pdf_vals, mode='lines',
                name='t-Student', line_color=_C['orange']))
            fig_dist.add_vline(x=risk['var_95'], line_dash="dash",
                               line_color=_C['orange'],
                               annotation_text=f"VaR 95% ({risk['var_95']:.2%})")
            fig_dist.add_vline(x=risk['var_99'], line_dash="dash",
                               line_color=_C['red'],
                               annotation_text=f"VaR 99% ({risk['var_99']:.2%})")
            _xlo = max(log_returns.quantile(0.005), -0.08)
            _xhi = min(log_returns.quantile(0.995), 0.08)
            fig_dist.update_layout(title="Distribuição de Retornos",
                                   yaxis_title="Prob.",
                                   xaxis_tickformat=".1%",
                                   xaxis_range=[_xlo, _xhi],
                                   height=350, width=480, template=DASH_TEMPLATE,
                                   margin=dict(t=35, b=25))

            # Sumário de vol e risco
            summary_html = f"""
            <div class='mm-dash'><div class='mm-card'>
                <div class='mm-kpi-row'>
                    <div class='mm-kpi'><div class='kpi-label'>IV 30d ATM</div><div class='kpi-value' style='color:{_C["accent"]}'>{iv_30d:.2%}</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>RV 30d</div><div class='kpi-value' style='color:{_C["teal"]}'>{rv_30d:.2%}</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>Prêmio</div><div class='kpi-value' style='color:{_C["orange"]}'>{vol_premium:+.2f}%</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>Skew</div><div class='kpi-value' style='color:{_C["purple"]}'>{skew:+.2%}</div></div>
                </div>
                <div class='mm-section-label'>Risco Caudal</div>
                <div class='mm-kpi-row'>
                    <div class='mm-kpi'><div class='kpi-label'>VaR 95%</div><div class='kpi-value' style='color:{_C["yellow"]}'>{risk['var_95']:.2%}</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>CVaR 95%</div><div class='kpi-value' style='color:{_C["orange"]}'>{risk['cvar_95']:.2%}</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>VaR 99%</div><div class='kpi-value' style='color:{_C["red"]}'>{risk['var_99']:.2%}</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>CVaR 99%</div><div class='kpi-value' style='color:{_C["red"]}'>{risk['cvar_99']:.2%}</div></div>
                </div>
                <div class='mm-section-label'>Posicionamento</div>
                <div class='mm-kpi-row'>
                    <div class='mm-kpi'><div class='kpi-label'>Gamma Flip</div><div class='kpi-value'>~{f'{gamma_flip:,.0f}' if gamma_flip else 'N/A'}</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>Call Wall</div><div class='kpi-value' style='color:{_C["green"]}'>{f'{call_wall:,.0f}' if call_wall else 'N/A'}</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>Put Wall</div><div class='kpi-value' style='color:{_C["red"]}'>{f'{put_wall:,.0f}' if put_wall else 'N/A'}</div></div>
                </div>
            </div></div>"""

            # ══ Dimensões fixas — NUNCA alterar sem redesign completo ═
            _GW, _GH   = 210, 190   # gauge  width × height
            _DH        = 250        # detail panel height

            # Helper: envolve qualquer widget numa célula de tamanho fixo
            def _cell(widget, w, h=None):
                kw = dict(width=f'{w}px', min_width=f'{w}px', max_width=f'{w}px',
                          overflow='hidden')
                if h:
                    kw.update(height=f'{h}px', min_height=f'{h}px', max_height=f'{h}px')
                return wd.Box([widget], layout=wd.Layout(**kw))

            # ── Tail Risk ──────────────────────────────────────────────
            _home_tail_gauge = build_tail_gauge(
                analytics.get('tail_score', 50),
                analytics.get('tail_interp', ''))
            _home_tail_gauge.update_layout(width=_GW, height=_GH,
                                           margin=dict(t=40, b=8, l=15, r=15))
            _tail_score_val = analytics.get('tail_score', 50)
            _tail_interp    = analytics.get('tail_interp', '')
            _tail_rows = []
            for _ck, _cv in analytics.get('tail_components', {}).items():
                _tail_rows.append(
                    f"<tr><td style='color:{_C['text_muted']};padding:3px 8px;font-size:11px;'>"
                    f"{_cv.get('label', _ck)}</td>"
                    f"<td style='color:{_C['text']};padding:3px 8px;font-size:11px;font-weight:700;'>"
                    f"{_cv.get('value', 0)}</td>"
                    f"<td style='color:{_C['yellow']};padding:3px 8px;font-size:11px;'>"
                    f"{_cv.get('score', 0):.1f}</td></tr>")
            _tail_detail_html = (
                f"<div class='mm-dash'><div class='mm-card' style='min-width:280px;'>"
                f"<h3>Tail Risk — {_tail_score_val:.0f}/100</h3>"
                f"<p style='margin:0 0 8px;'>{_tail_interp}</p>"
                f"<table style='width:100%;border-collapse:collapse;'>"
                f"<tr><th style='font-size:9px;color:{_C['text_dim']};padding:3px 8px;"
                f"text-align:left;border-bottom:1px solid {_C['border']};'>Componente</th>"
                f"<th style='font-size:9px;color:{_C['text_dim']};padding:3px 8px;"
                f"border-bottom:1px solid {_C['border']};'>Valor</th>"
                f"<th style='font-size:9px;color:{_C['text_dim']};padding:3px 8px;"
                f"border-bottom:1px solid {_C['border']};'>Score</th></tr>"
                + ''.join(_tail_rows)
                + f"</table></div></div>")

            # ── Flow Predictor ─────────────────────────────────────────
            _fp_gauge_w = wd.HTML(
                f"<div class='mm-dash'><div class='mm-card' style='width:{_GW}px;height:{_GH}px;"
                f"display:flex;align-items:center;justify-content:center;'>"
                f"<p style='color:{_C['text_muted']};font-size:11px;'>Flow N/A</p></div></div>")
            _fp_comps_w = wd.HTML(
                f"<div class='mm-dash'><div class='mm-card'>"
                f"<p style='color:{_C['text_muted']};font-size:11px;'>Ative o Flow Predictor</p>"
                f"</div></div>")
            if fp_ok and fp_score is not None:
                try:
                    _fp_gauge_w = fp_plot_score_gauge(fp_score)
                    _fp_gauge_w.update_layout(width=_GW, height=_GH,
                                              margin=dict(t=40, b=8, l=15, r=15))
                    _fp_comps_w = fp_plot_components_bar(fp_score)
                    _fp_comps_w.update_layout(height=_DH,
                                              margin=dict(t=32, b=40, l=10, r=20))
                except Exception:
                    pass

            # ── CTA Chart ─────────────────────────────────────────────
            _home_cta = wd.HTML(
                f"<div class='mm-dash'><div class='mm-card'>"
                f"<p style='color:{_C['text_muted']};'>CTA: ative o Flow Predictor</p>"
                f"</div></div>")
            if fp_ok and not fp_cta_hist.empty:
                try:
                    _cta_fig = build_cta_gs_chart(
                        fp_cta_hist, fp_cta_scenarios_1w, fp_cta_scenarios_1m, spot)
                    _home_cta = wd.Output()
                    with _home_cta:
                        _cta_fig.show()
                except Exception:
                    pass

            # ── Gregas + ETF flow ──────────────────────────────────────
            try:
                _greek_overview = build_greek_overview(
                    greeks_now, df, spot,
                    etf_flows=etf_flows if etf_ok else {})
            except Exception as _go_err:
                print(f"⚠️ Greek overview: {_go_err}")
                _greek_overview = wd.HTML('')

            # ── Gamma Squeeze ──────────────────────────────────────────
            _sq_gauge_w    = wd.HTML('')
            _sq_comps_w    = wd.HTML('')
            _sq_badge_w    = wd.HTML('')
            _sq_score_disp = 'N/A'
            _sq_ac         = _C['text_muted']
            try:
                _sq_pc_v1  = fp_vol_data.get('pc_ratio', 0) or 0  # 0 = BBG unavailable
                _sq_gex_v1 = total_gex_val / 1e9 if 'total_gex_val' in dir() else (
                              total_gex / 1e9    if 'total_gex'     in dir() else 0)
                _sq_result_v1 = compute_gamma_squeeze_score(
                    net_gex_bn=_sq_gex_v1, pc_ratio=_sq_pc_v1,
                    iv_30d=iv_30d, rv_30d=rv_30d, gamma_flip=gamma_flip,
                    spot=spot, skew=skew, put_wall=put_wall, call_wall=call_wall)
                _sq_gauge_w, _sq_comps_w, _sq_badge_str, _sq_ac = \
                    build_squeeze_mini_panel(_sq_result_v1, _C)
                _sq_gauge_w.update_layout(width=_GW, height=_GH,
                                          margin=dict(t=38, b=8, l=18, r=18))
                _sq_comps_w.update_layout(height=_DH,
                                          margin=dict(t=30, b=20, l=5, r=60))
                _sq_badge_w    = wd.HTML(_sq_badge_str)
                _sq_score_disp = f"{_sq_result_v1['score']:.0f}"
            except Exception as _sqm_err:
                print(f"⚠️ Squeeze mini: {_sqm_err}")

            # ── Status bar ─────────────────────────────────────────────
            _flip_str = f"{gamma_flip:,.0f}"  if gamma_flip      else "N/A"
            _gex_disp = _sq_gex_v1 * 0.1 if '_sq_gex_v1' in dir() else None
            _gex_str  = f"{_gex_disp:+.1f}B" if _gex_disp is not None else "N/A"
            _pc_str   = (f"{_sq_pc_v1:.2f}×" if ('_sq_pc_v1' in dir() and _sq_pc_v1 > 0) else "N/D")
            _ivrv_str = f"{(iv_30d - rv_30d)*100:+.1f}pp"

            def _stat(label, value, color):
                return (f"<div class='mm-stat-item'>"
                        f"<span class='mm-stat-label'>{label}</span>"
                        f"<span class='mm-stat-value' style='color:{color};'>{value}</span>"
                        f"</div>")

            _status_bar = wd.HTML(
                f"<div class='mm-dash mm-statusbar'>"
                "<div class='jarvis-reactor'>"
                "<svg width='42' height='42' viewBox='0 0 42 42'>"
                "<defs><filter id='rfglow'><feGaussianBlur stdDeviation='2' result='b'/>"
                "<feMerge><feMergeNode in='b'/><feMergeNode in='SourceGraphic'/></feMerge></filter></defs>"
                "<circle cx='21' cy='21' r='18' fill='none' stroke='rgba(0,200,255,.22)' stroke-width='1' stroke-dasharray='6 4' class='jarvis-r1'/>"
                "<circle cx='21' cy='21' r='13' fill='none' stroke='rgba(0,200,255,.42)' stroke-width='1' stroke-dasharray='4 3' class='jarvis-r2'/>"
                "<circle cx='21' cy='21' r='7'  fill='none' stroke='rgba(0,200,255,.62)' stroke-width='1' stroke-dasharray='2 2' class='jarvis-r3'/>"
                "<circle cx='21' cy='21' r='3.5' fill='rgba(0,200,255,.95)' filter='url(#rfglow)'/>"
                "</svg></div>"
                f"<span class='mm-cmd-title'>⬡ SPX&nbsp;MARKET&nbsp;COMMAND</span>"
                f"<div style='display:flex;flex-wrap:wrap;align-items:stretch;'>"
                + _stat('Spot',            f"{spot:,.0f}",        _C['text'])
                + _stat('Gamma&nbsp;Flip', _flip_str,             _C['orange'])
                + _stat('GEX&nbsp;Net',    _gex_str,              _C['accent'])
                + _stat('P/C&nbsp;Ratio',  _pc_str,               _C['purple'])
                + _stat('IV−RV',           _ivrv_str,             _C['yellow'])
                + _stat('Squeeze&nbsp;Risk', f"{_sq_score_disp}/100", _sq_ac)
                + f"</div></div>")

            # ── Section header helper ──────────────────────────────────
            def _sh(title, sub=''):
                _s = f"<span class='mm-hdr-sub'>· {sub}</span>" if sub else ''
                return wd.HTML(
                    f"<div class='mm-dash mm-section-hdr'>"
                    f"<div class='mm-dot'></div>"
                    f"<span class='mm-hdr-title'>{title}</span>{_s}"
                    f"</div>")

            # ══ LAYOUT TAB 1 ═══════════════════════════════════════════
            # ── 4-column symmetric ring grid (2 rows × 4 cols) ─────────
            _GW, _GH = 260, 272   # Stark arc reactor ring size

            # -- Dynamic gradient colors per ring --
            def _risk_grad(score):
                """Return (color1, color2) based on 0-100 risk score."""
                if score < 30:   return '#2ed573', '#00d4ff'
                if score < 60:   return '#ffd32a', '#ffa502'
                if score < 80:   return '#ffa502', '#ff6b35'
                return '#ff4757', '#b84040'

            _tail_g1, _tail_g2 = _risk_grad(_tail_score_val)
            _tail_svg_lbl2 = (_tail_interp.split('—')[0].strip()[:16]
                              if _tail_interp else '')

            _fp_svg_val   = 50.0
            _fp_svg_lbl2  = ''
            _fp_g1, _fp_g2 = '#00d4ff', '#7efff5'
            if fp_ok and fp_score is not None:
                try:
                    _fp_svg_val  = float(fp_score.get('score', 50)
                                         if isinstance(fp_score, dict) else 50)
                    _fp_svg_lbl2 = (str(fp_score.get('interpretation', '')).split()[0][:14]
                                    if isinstance(fp_score, dict) else '')
                    _fp_g1, _fp_g2 = _risk_grad(100 - _fp_svg_val)  # invert: high flow = good
                except Exception:
                    pass

            _sq_svg_val  = 50.0
            _sq_svg_lbl2 = ''
            _sq_g1, _sq_g2 = '#b44aff', '#ff6b9d'
            try:
                _sq_svg_val  = float(_sq_result_v1.get('score', 50))
                _sq_svg_lbl2 = str(_sq_result_v1.get('label', ''))[:16]
                _sq_g1, _sq_g2 = _risk_grad(_sq_svg_val)
            except Exception:
                pass

            # Row 1: 4 market metric rings
            _r_frag = wd.HTML(_svg_ring_html(
                fragility,    0, _frag_max,  'Fragilidade',   '%', '#ff4757', '#ff7843', _GW, _GH))
            _r_vol  = wd.HTML(_svg_ring_html(
                vol_premium, -_vol_hi, _vol_hi, 'Premio Vol', '%', '#ffa502', '#ff6b81', _GW, _GH))
            _r_skew = wd.HTML(_svg_ring_html(
                _skew_val,   -_skew_hi, _skew_hi, 'Skew P25-C25','%','#00d4ff','#7efff5',_GW,_GH))
            _r_move = wd.HTML(_svg_ring_html(
                daily_move,  0, _move_hi,   'Mov Esp 1D',    '%', '#2ed573', '#00d4ff', _GW, _GH))

            # Row 2: 3 risk score rings + 1 compact KPI panel
            _r_tail = wd.HTML(_svg_ring_html(
                _tail_score_val, 0, 100, 'Tail Risk', '',
                _tail_g1, _tail_g2, _GW, _GH, label2=_tail_svg_lbl2))
            _r_fp   = wd.HTML(_svg_ring_html(
                _fp_svg_val, 0, 100, 'Flow Score', '',
                _fp_g1, _fp_g2, _GW, _GH, label2=_fp_svg_lbl2))
            _r_sq   = wd.HTML(_svg_ring_html(
                _sq_svg_val, 0, 100, 'Gamma Squeeze', '',
                _sq_g1, _sq_g2, _GW, _GH, label2=_sq_svg_lbl2))

            # 4th slot row 2: compact KPI status card (pure SVG)
            _kpi_svg = (
                f"<svg viewBox='0 0 {_GW} {_GH}' width='{_GW}' height='{_GH}' "
                f"style='background:linear-gradient(145deg,#04081e,#080c28);display:block;'>"
                # corner brackets
                f"<path d='M 2,15 L 2,2 L 15,2' fill='none' stroke='#00d4ff' stroke-width='2' opacity='0.7'/>"
                f"<path d='M {_GW-15},2 L {_GW-2},2 L {_GW-2},15' fill='none' stroke='#00d4ff' stroke-width='2' opacity='0.7'/>"
                f"<path d='M 2,{_GH-15} L 2,{_GH-2} L 15,{_GH-2}' fill='none' stroke='#00d4ff' stroke-width='2' opacity='0.7'/>"
                f"<path d='M {_GW-15},{_GH-2} L {_GW-2},{_GH-2} L {_GW-2},{_GH-15}' fill='none' stroke='#00d4ff' stroke-width='2' opacity='0.7'/>"
                # title
                f"<text x='{_GW//2}' y='28' text-anchor='middle' "
                f"font-family=\"'Courier New',monospace\" font-size='9' font-weight='700' "
                f"letter-spacing='2.5' fill='rgba(0,212,255,0.55)'>MARKET STATUS</text>"
                f"<line x1='20' y1='38' x2='{_GW-20}' y2='38' stroke='rgba(0,212,255,0.18)' stroke-width='1'/>"
                # Row A: SPOT | GAMMA FLIP
                f"<text x='62' y='68' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='8' fill='rgba(0,212,255,0.4)' letter-spacing='1'>SPOT</text>"
                f"<text x='62' y='90' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='20' font-weight='700' fill='#e0f0ff'>{spot:,.0f}</text>"
                f"<text x='{_GW-62}' y='68' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='8' fill='rgba(0,212,255,0.4)' letter-spacing='1'>GAMMA FLIP</text>"
                f"<text x='{_GW-62}' y='90' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='20' font-weight='700' fill='#ffa502'>{_flip_str}</text>"
                f"<line x1='20' y1='104' x2='{_GW-20}' y2='104' stroke='rgba(0,212,255,0.08)' stroke-width='1'/>"
                # Row B: IV-RV | P/C RATIO
                f"<text x='62' y='130' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='8' fill='rgba(0,212,255,0.4)' letter-spacing='1'>IV-RV</text>"
                f"<text x='62' y='152' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='20' font-weight='700' fill='#ffd32a'>{_ivrv_str}</text>"
                f"<text x='{_GW-62}' y='130' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='8' fill='rgba(0,212,255,0.4)' letter-spacing='1'>P/C RATIO</text>"
                f"<text x='{_GW-62}' y='152' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='20' font-weight='700' fill='#b44aff'>{_pc_str}</text>"
                f"<line x1='20' y1='166' x2='{_GW-20}' y2='166' stroke='rgba(0,212,255,0.08)' stroke-width='1'/>"
                # Row C: GEX NET | SQUEEZE
                f"<text x='62' y='193' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='8' fill='rgba(0,212,255,0.4)' letter-spacing='1'>GEX NET</text>"
                f"<text x='62' y='215' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='20' font-weight='700' fill='#00d4ff'>{_gex_str}</text>"
                f"<text x='{_GW-62}' y='193' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='8' fill='rgba(0,212,255,0.4)' letter-spacing='1'>SQUEEZE</text>"
                f"<text x='{_GW-62}' y='215' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='20' font-weight='700' fill='{_sq_g1}'>{_sq_score_disp}/100</text>"
                f"</svg>"
            )
            _r_kpi = wd.HTML(_kpi_svg)

            # 4×2 grid: wraps naturally into 2 rows of 4
            _gauge_grid = wd.GridBox(
                [_cell(_r_frag, _GW, _GH), _cell(_r_vol,  _GW, _GH),
                 _cell(_r_skew, _GW, _GH), _cell(_r_move, _GW, _GH),
                 _cell(_r_tail, _GW, _GH), _cell(_r_fp,   _GW, _GH),
                 _cell(_r_sq,   _GW, _GH), _cell(_r_kpi,  _GW, _GH)],
                layout=wd.Layout(
                    grid_template_columns=f'repeat(4, {_GW}px)',
                    gap='6px',
                    width='fit-content'))

            # ── Linha 2: 3 painéis de detalhe em GridBox ───────────────
            _tail_html_w  = wd.HTML(_tail_detail_html)
            _sq_detail_vb = wd.VBox([_sq_badge_w, _sq_comps_w])
            _detail_grid  = wd.GridBox(
                [_cell(_tail_html_w,  None, _DH),
                 _cell(_fp_comps_w,   None, _DH),
                 _cell(_sq_detail_vb, None, _DH)],
                layout=wd.Layout(
                    grid_template_columns='repeat(3, minmax(420px, 1fr))',
                    grid_template_rows=f'{_DH}px',
                    gap='6px',
                    width='100%',
                    overflow_x='auto'))

            tab1 = wd.VBox([
                _status_bar,
                _sh('Painel de Controle',
                    'Fragilidade · Vol · Skew · Move · Tail Risk · Flow Score · Gamma Squeeze'),
                _gauge_grid,
                _detail_grid,
                _sh('Exposição das Gregas',
                    'Delta · Gamma · Vanna · Charm + Rebalanceamento ETF Passivo'),
                _greek_overview,
                _sh('Estrutura de Mercado',
                    'GEX por Strike · Distribuição de Retornos'),
                wd.HBox([fig_gex, fig_dist],
                        layout={'flex_wrap': 'nowrap', 'align_items': 'flex-start', 'overflow_x': 'auto', 'width': '100%'}),
                _sh('CTA Estimado & Resumo Narrativo'),
                _home_cta,
                wd.HTML(summary_html),
            ])

            # ─── ABA 2: EXPOSIÇÕES POR STRIKE ───────────────────────────
            exp_output = wd.Output()
            with exp_output:
                plot_exposure_charts(
                    agg, df, spot, from_strike, to_strike,
                    levels, model_curves, flip_points,
                    call_wall, put_wall)
            tab2 = exp_output

            # ─── ABA 3: SENSIBILIDADE ────────────────────────────────────
            cmap_map = {
                'delta': 'RdBu_r', 'gamma': 'viridis', 'vega': 'YlGnBu',
                'vanna': 'PuOr', 'theta': 'Greens', 'charm': 'RdYlGn',
                'zomma': 'plasma', 'speed': 'coolwarm'
            }
            titles_map = {
                'delta': 'Delta Nocional ($ Mn)',
                'gamma': 'Gamma — GEX NET ($ Mn / 1% move)',
                'vega': 'Vega ($ Mn / 1 vol pt)',
                'vanna': 'Vanna ($ Mn / 1pt × 1 vol pt)',
                'theta': 'Theta — Decaimento ($ Mn / dia)',
                'charm': 'Charm — Decay do Delta ($ Mn / dia)',
                'zomma': 'Zomma ($ Mn / 1% move²)',
                'speed': 'Speed ($ Mn / 1pt)',
            }
            sens_html_parts = []
            for key in ['delta', 'gamma', 'vega', 'vanna', 'theta', 'charm', 'zomma', 'speed']:
                styled = style_sensitivity_matrix(sens_matrices[key], cmap_map[key])
                sens_html_parts.append(f"<h4>{titles_map[key]}</h4>{styled}<br>")
            tab3 = wd.VBox([
                wd.HTML("<div class='mm-dash'><div class='mm-card'><h3>Matrizes de Sensibilidade (Preço × Vol Shift)</h3></div></div>"),
                wd.HTML("".join(sens_html_parts))
            ])

            # ─── ABA 4: ANÁLISE DE P&L ──────────────────────────────────
            # P&L comparativo
            fig_pnl = go.FigureWidget()
            fig_pnl.add_trace(go.Scatter(
                x=levels, y=pnl_curves['simple'] / 1e6,
                mode='lines', name='Simplificado (Δ+Γ)',
                line=dict(color=_C['orange'], dash='dot')))
            fig_pnl.add_trace(go.Scatter(
                x=levels, y=pnl_curves['complete'] / 1e6,
                mode='lines', name='Completo (+Vega+Vanna+Zomma)',
                line=dict(color=_C['accent']), fill='tonexty',
                fillcolor='rgba(88,166,255,0.08)'))
            fig_pnl.add_vline(x=spot, line_dash="dash", line_color=_C['red'])
            fig_pnl.add_hline(y=0, line_width=0.5, line_color=_C['text_dim'])
            fig_pnl.update_layout(title="P&L Comparativo: Modelo Completo vs. Simplificado",
                                  yaxis_title="P&L ($ Mi)", xaxis_title="Preço do Ativo",
                                  height=380, template=DASH_TEMPLATE)

            # Dealer P&L
            fig_dealer = go.FigureWidget()
            fig_dealer.add_trace(go.Scatter(
                x=levels, y=pnl_curves['dealer'] / 1e6,
                mode='lines', name='P&L Dealer (Total)', line_color=_C['purple'],
                line=dict(width=2.5),
                fill='tozeroy', fillcolor='rgba(188,140,255,0.08)'))
            # Per-dealer P&L curves
            _dealer_colors = ['#58a6ff', '#3fb950', '#f0883e', '#da3633',
                              '#d29922', '#bc8cff', '#8b949e', '#e6edf3']
            for _di, (_mm, _share) in enumerate(MM_VOLUME_SHARES.items()):
                _dlr_pnl = pnl_curves['dealer'] * _share / 1e6
                fig_dealer.add_trace(go.Scatter(
                    x=levels, y=_dlr_pnl,
                    mode='lines', name=_mm,
                    line=dict(color=_dealer_colors[_di % len(_dealer_colors)],
                              width=1, dash='dot'),
                    visible='legendonly'))
            fig_dealer.add_vline(x=spot, line_dash="dash", line_color=_C['red'])
            fig_dealer.add_hline(y=0, line_width=0.5, line_color=_C['text_dim'])
            fig_dealer.update_layout(title="P&L Estimado — Total + Por Dealer",
                                     yaxis_title="P&L ($ Mi)", xaxis_title="Preço do Ativo",
                                     height=420, template=DASH_TEMPLATE,
                                     legend=dict(font=dict(size=9), y=1.02, orientation='h'))

            # Dealer book summary table
            _oi100_pnl = df.OI.values * 100
            _cs_pnl = np.where(df.Type.values == 'Call', 1, -1)
            _total_dex = float(np.nansum(greeks_now['delta'] * _oi100_pnl))
            _total_gex = float(np.nansum(greeks_now['gamma'] * _cs_pnl * _oi100_pnl))
            _total_theta = float(np.nansum(greeks_now['theta'] * _oi100_pnl))
            _total_vanna = float(np.nansum(greeks_now['vanna'] * _oi100_pnl))
            _book_rows = []
            for _mm, _share in list(MM_VOLUME_SHARES.items()) + [('TOTAL', 1.0)]:
                _book_rows.append({
                    'Dealer': _mm,
                    'Share': '{:.0%}'.format(_share),
                    'DEX ($M)': '{:,.1f}'.format(_total_dex * _share * spot / 1e6),
                    'GEX/pt ($M)': '{:,.1f}'.format(_total_gex * _share / 1e6),
                    'Theta ($K/d)': '{:,.0f}'.format(_total_theta * _share / 1e3),
                    'Vanna': '{:,.0f}'.format(_total_vanna * _share),
                })
            _book_html = pd.DataFrame(_book_rows).to_html(
                classes='mm-table', index=False, border=0)
            dealer_summary_w = wd.HTML(
                "<div class='mm-dash'><div class='mm-card'>"
                "<h3>Book dos Dealers (Estimado)</h3>"
                "{}</div></div>".format(_book_html))

            # Hedge demand — per dealer
            fig_hedge = go.FigureWidget()
            _dealer_colors_h = ['#58a6ff', '#3fb950', '#f0883e', '#da3633',
                                '#d29922', '#bc8cff', '#8b949e', '#e6edf3']
            # Total hedge demand (main curve)
            fig_hedge.add_trace(go.Scatter(
                x=levels, y=pnl_curves['hedge_demand'],
                mode='lines', line=dict(color=_C['teal'], width=2.5),
                name='Total'))
            # Per-dealer hedge demand curves
            for _dhi, (_mm_h, _share_h) in enumerate(MM_VOLUME_SHARES.items()):
                _dlr_hedge = pnl_curves['hedge_demand'] * _share_h
                fig_hedge.add_trace(go.Scatter(
                    x=levels, y=_dlr_hedge,
                    mode='lines', name=_mm_h,
                    line=dict(color=_dealer_colors_h[_dhi % len(_dealer_colors_h)],
                              width=1, dash='dot'),
                    visible='legendonly'))
            fig_hedge.add_vline(x=spot, line_dash="dash", line_color=_C['red'])
            fig_hedge.add_hline(y=0, line_width=0.5, line_color=_C['text_dim'])
            fig_hedge.update_layout(
                title=f"Demanda de Hedge em Futuros ({FUTURES_TICKER})",
                yaxis_title="Número de Contratos",
                xaxis_title="Preço do Ativo",
                height=380, template=DASH_TEMPLATE)

            tab4 = wd.VBox([fig_pnl, wd.HBox([fig_dealer, fig_hedge]),
                            dealer_summary_w])

            # ─── ABA 5: MONTE CARLO ──────────────────────────────────────
            sim_var_95 = np.percentile(mc_pnl, 5)
            sim_cvar_95 = mc_pnl[mc_pnl <= sim_var_95].mean() if np.any(mc_pnl <= sim_var_95) else sim_var_95
            sim_var_99 = np.percentile(mc_pnl, 1)
            sim_cvar_99 = mc_pnl[mc_pnl <= sim_var_99].mean() if np.any(mc_pnl <= sim_var_99) else sim_var_99
            mc_p1 = np.percentile(mc_pnl, 1)
            mc_p99 = np.percentile(mc_pnl, 99)
            mc_iqr = mc_p99 - mc_p1

            fig_mc_hist = go.FigureWidget()
            fig_mc_hist.add_trace(go.Histogram(
                x=mc_pnl / 1e6, nbinsx=120,
                marker_color=_C['accent'], opacity=0.7, name='P&L'))
            fig_mc_hist.add_vline(x=sim_var_99 / 1e6, line_dash='dash',
                                  line_color=_C['red'],
                                  annotation_text=f'VaR 99% ${sim_var_99/1e6:,.1f}M',
                                  annotation_position='top left')
            fig_mc_hist.add_vline(x=sim_var_95 / 1e6, line_dash='dash',
                                  line_color=_C['orange'],
                                  annotation_text=f'VaR 95% ${sim_var_95/1e6:,.1f}M',
                                  annotation_position='bottom left')
            fig_mc_hist.add_vline(x=0, line_width=0.5, line_color=_C['text_dim'])
            mc_xlo = (mc_p1 - mc_iqr * 0.15) / 1e6
            mc_xhi = (mc_p99 + mc_iqr * 0.15) / 1e6
            fig_mc_hist.update_layout(
                title=f'Distribuição de P&L do Livro (10k Sim. t-Student, {mc_n_days} Dias)',
                xaxis_title='P&L ($ Mi)', yaxis_title='Frequência',
                xaxis_range=[mc_xlo, mc_xhi],
                height=420, template=DASH_TEMPLATE)

            mc_win_pct = (mc_pnl > 0).mean() * 100
            mc_max_loss = mc_pnl.min()
            mc_max_gain = mc_pnl.max()
            mc_table = pd.DataFrame({
                'Métrica': ['P&L Médio', 'P&L Mediano', '% Cenários Positivos',
                            'VaR 95% (Sim.)', 'CVaR 95% (Sim.)',
                            'VaR 99% (Sim.)', 'CVaR 99% (Sim.)',
                            'Perda Máxima', 'Ganho Máximo',
                            f'Horizonte ({mc_n_days}d)'],
                'Valor': [f'${np.mean(mc_pnl)/1e6:,.2f} Mi',
                          f'${np.median(mc_pnl)/1e6:,.2f} Mi',
                          f'{mc_win_pct:.1f}%',
                          f'${sim_var_95/1e6:,.2f} Mi',
                          f'${sim_cvar_95/1e6:,.2f} Mi',
                          f'${sim_var_99/1e6:,.2f} Mi',
                          f'${sim_cvar_99/1e6:,.2f} Mi',
                          f'${mc_max_loss/1e6:,.2f} Mi',
                          f'${mc_max_gain/1e6:,.2f} Mi',
                          f'{mc_n_days} dias úteis']
            }).to_html(classes='mm-table', index=False, border=0)

            # Per-dealer MC table
            dealer_mc_html = ''
            if analytics.get('dealer_mc') and len(analytics['dealer_mc']) > 1:
                dmc = analytics['dealer_mc']
                dmc_rows = []
                for mm_name in list(MM_VOLUME_SHARES.keys()) + ['TOTAL']:
                    if mm_name in dmc:
                        d = dmc[mm_name]
                        dmc_rows.append({
                            'Dealer': mm_name,
                            'Share': '{:.0%}'.format(d.get('share', 0)),
                            'Mean P&L ($M)': '{:,.1f}'.format(d['mean_pnl'] / 1e6),
                            'VaR 95% ($M)': '{:,.1f}'.format(d['var_95'] / 1e6),
                            'VaR 99% ($M)': '{:,.1f}'.format(d['var_99'] / 1e6),
                            'CVaR 95% ($M)': '{:,.1f}'.format(d['cvar_95'] / 1e6),
                            'Win %': '{:.1f}'.format(d['win_pct']),
                            'Max Loss ($M)': '{:,.1f}'.format(d['max_loss'] / 1e6),
                        })
                if dmc_rows:
                    dealer_mc_html = pd.DataFrame(dmc_rows).to_html(
                        classes='mm-table', index=False, border=0)

            mc_dealer_widget = wd.HTML(
                "<div class='mm-dash'><div class='mm-card'>"
                "<h3>Monte Carlo por Dealer</h3>"
                "{}</div></div>".format(dealer_mc_html if dealer_mc_html else
                                        '<p style="color:#8b949e;">Sem dados de dealer MC</p>'))

            # Per-dealer histograms (overlay top 4 dealers)
            fig_mc_dealers = go.FigureWidget()
            _mc_d_colors = ['#58a6ff', '#3fb950', '#f0883e', '#da3633']
            _mc_top4 = list(MM_VOLUME_SHARES.keys())[:4]
            dmc = analytics.get('dealer_mc', {})
            for _mci, _mmn in enumerate(_mc_top4):
                if _mmn in dmc and 'mc_pnl' in dmc[_mmn]:
                    fig_mc_dealers.add_trace(go.Histogram(
                        x=dmc[_mmn]['mc_pnl'] / 1e6, nbinsx=80,
                        marker_color=_mc_d_colors[_mci], opacity=0.5,
                        name=_mmn))
            fig_mc_dealers.update_layout(
                title='Distribuição P&L por Dealer (Top 4)',
                xaxis_title='P&L ($ Mi)', yaxis_title='Frequência',
                barmode='overlay', height=350, template=DASH_TEMPLATE)

            tab5 = wd.VBox([
                wd.HTML("<div class='mm-dash'><div class='mm-card'>"
                        f"<h3>Simulação Monte Carlo (t-Student, {mc_n_days} Dias)</h3></div></div>"),
                wd.HBox([fig_mc_hist, wd.HTML(
                    f"<div class='mm-dash'><div class='mm-card'>{mc_table}</div></div>")]),
                mc_dealer_widget,
                fig_mc_dealers,
            ])

            # ─── ABA 6: REBALANCEAMENTO ETFs + ALAVANCADOS ─────────────
            if etf_ok and etf_flows:
                # Dropdown para selecionar ETF
                etf_dd = wd.Dropdown(
                    options=['Combined'] + PASSIVE_ETFS,
                    value='Combined', description='ETF:',
                    layout={'width': '300px'})
                flow_html = wd.HTML()
                summary_html = wd.HTML()

                def _render_etf(change=None):
                    key = etf_dd.value
                    df_f = etf_flows.get(key, pd.DataFrame())
                    if df_f.empty:
                        flow_html.value = "<p>Sem dados.</p>"
                        return
                    top = df_f.head(30)
                    flow_html.value = (
                        top[['Start', 'Now', 'Delta', 'Flow_$', 'PctADV']]
                        .style.format({
                            'Start': '{:.4f}', 'Now': '{:.4f}', 'Delta': '{:+.4f}',
                            'Flow_$': '${:,.0f}', 'PctADV': '{:.1f}%'})
                        .background_gradient(cmap='RdYlGn', subset=['Flow_$'])
                        .to_html())
                etf_dd.observe(_render_etf, names='value')
                _render_etf()

                if etf_summary is not None:
                    summary_html.value = (
                        "<h4>Resumo por ETF</h4>" +
                        etf_summary.style.format('${:,.0f}')
                            .background_gradient(cmap='RdYlGn', subset=['Net_$'])
                            .to_html())

                # Seção de ETFs alavancados
                if lev_ok and lev_flows is not None:
                    lev_html_str = (
                        f"<h4>ETFs Alavancados (Retorno diário: {daily_ret*100:+.2f}%)</h4>" +
                        lev_flows[['Leverage', 'AUM_$', 'Rebalance_$', 'Direção']]
                        .style.format({
                            'AUM_$': '${:,.0f}', 'Rebalance_$': '${:,.0f}'})
                        .to_html() +
                        f"<p><b>Fluxo Direcional Total: ${lev_total:,.0f}</b></p>")
                else:
                    lev_html_str = "<p>ETFs alavancados não disponíveis.</p>"

                # Top 30 que ganham fluxo de compra e top 30 que levam fluxo de venda
                combo_flow = etf_flows.get('Combined', pd.DataFrame())
                flow_ranking = wd.HTML()
                if not combo_flow.empty and 'Flow_$' in combo_flow.columns:
                    top30_buy = combo_flow.nlargest(30, 'Flow_$')
                    top30_sell = combo_flow.nsmallest(30, 'Flow_$')
                    fmt_dict = {'Delta': '{:+.4f}', 'Flow_$': '${:,.0f}', 'PctADV': '{:.1f}%'}
                    buy_html = (top30_buy[['Delta', 'Flow_$', 'PctADV']]
                        .style.format(fmt_dict)
                        .background_gradient(cmap='Greens', subset=['Flow_$'])
                        .to_html())
                    sell_html = (top30_sell[['Delta', 'Flow_$', 'PctADV']]
                        .style.format(fmt_dict)
                        .background_gradient(cmap='Reds', subset=['Flow_$'])
                        .to_html())
                    flow_ranking = wd.HBox([
                        wd.VBox([wd.HTML("<div class='mm-dash'><div class='mm-card'><h4>Top 30 — Fluxo de Compra</h4></div></div>"),
                                 wd.HTML(buy_html)]),
                        wd.VBox([wd.HTML("<div class='mm-dash'><div class='mm-card'><h4>Top 30 — Fluxo de Venda</h4></div></div>"),
                                 wd.HTML(sell_html)])
                    ])

                tab6 = wd.VBox([
                    wd.HTML(f"<div class='mm-dash'><div class='mm-card'><h3>Rebalanceamento ETFs Passivos desde {etf_start}</h3></div></div>"),
                    etf_dd, flow_html, summary_html,
                    flow_ranking,
                    wd.HTML(lev_html_str)
                ])
            else:
                tab6 = wd.VBox([wd.HTML(
                    "<h3>Rebalanceamento ETFs</h3>"
                    "<p>Dados de rebalanceamento não disponíveis.</p>")])

            # ─── ABA 7: PREVISÃO SPX ────────────────────────────────────
            if spx_pred_ok and top_in_spx is not None:
                cols_in = ['Prob_In', 'CUR_MKT_CAP', 'FMC', 'FALR', 'FREE_FLOAT_PCT', 'NET_INC_TTM']
                cols_out = ['ExitScore', 'CUR_MKT_CAP', 'FMC', 'FALR', 'FREE_FLOAT_PCT', 'NET_INC_TTM']
                in_html = (top_in_spx[[c for c in cols_in if c in top_in_spx.columns]]
                    .style.format({
                        'Prob_In': '{:.2%}', 'CUR_MKT_CAP': '${:,.0f}',
                        'FMC': '${:,.0f}', 'FALR': '{:.4f}',
                        'FREE_FLOAT_PCT': '{:.1f}%'})
                    .background_gradient(cmap='Greens', subset=['Prob_In'])
                    .to_html())
                out_html = (top_out_spx[[c for c in cols_out if c in top_out_spx.columns]]
                    .style.format({
                        'ExitScore': '{:.2%}', 'CUR_MKT_CAP': '${:,.0f}',
                        'FMC': '${:,.0f}', 'FALR': '{:.4f}',
                        'FREE_FLOAT_PCT': '{:.1f}%', 'NET_INC_TTM': '${:,.0f}'})
                    .background_gradient(cmap='Reds', subset=['ExitScore'])
                    .to_html())
                spx_rules_html = (
                    "<div class='mm-dash'><div class='mm-card' style='padding:10px;'>"
                    "<span class='mm-section-label'>Critérios S&P 500</span> "
                    f"<span class='mm-cot-label'>Market Cap ≥ $18B</span> "
                    f"<span class='mm-cot-label'>Free Float ≥ 50%</span> "
                    f"<span class='mm-cot-label'>FALR ≥ 0.75</span> "
                    f"<span class='mm-cot-label'>Lucro TTM > 0</span> "
                    f"<span class='mm-cot-label'>NI Q0 > 0</span>"
                    "</div></div>")
                tab7 = wd.VBox([
                    wd.HTML("<div class='mm-dash'><div class='mm-card'><h3>Previsão de Rebalanceamento S&P 500</h3></div></div>"),
                    wd.HTML(spx_rules_html),
                    wd.HBox([
                        wd.VBox([wd.HTML("<div class='mm-dash'><div class='mm-card'><h4>Top 30 — Candidatos a Entrar</h4></div></div>"),
                                 wd.HTML(in_html)]),
                        wd.VBox([wd.HTML("<div class='mm-dash'><div class='mm-card'><h4>Top 30 — Candidatos a Sair</h4></div></div>"),
                                 wd.HTML(out_html)])
                    ])
                ])
            else:
                reason = "Marque 'Incluir Previsão SPX' e rode novamente." if not spx_pred_w.value else (
                    "sklearn não disponível." if not HAS_SKLEARN else "Erro na execução.")
                tab7 = wd.VBox([wd.HTML(
                    f"<h3>Previsão SPX</h3><p>{reason}</p>")])

            # ─── ABA 8: SIMULADOR INTERATIVO ─────────────────────────────
            vol_slider = wd.FloatSlider(
                value=0, min=-10, max=10, step=1,
                description='Shift Vol (pts):', continuous_update=False,
                layout={'width': '350px'})
            dte_slider = wd.IntSlider(
                value=0, min=0, max=20, step=1,
                description='Dias a Frente:', continuous_update=False,
                layout={'width': '350px'})
            spot_slider = wd.FloatSlider(
                value=0, min=-10, max=10, step=0.5,
                description='Spot Move (%):', continuous_update=False,
                layout={'width': '350px'})

            fig_sim_dex = go.FigureWidget()
            fig_sim_dex.add_trace(go.Scatter(x=levels, y=np.zeros(len(levels)),
                                             mode='lines', line_color=_C['accent'],
                                             name='Delta'))
            fig_sim_dex.add_trace(go.Scatter(x=[spot], y=[0], mode='markers',
                                             marker=dict(color=_C['red'], size=10, symbol='x'),
                                             name='Spot'))
            fig_sim_dex.update_layout(title="Delta Nocional", yaxis_title="$ Bi",
                                      height=320, template=DASH_TEMPLATE,
                                      margin=dict(t=30, b=20))

            fig_sim_gex = go.FigureWidget()
            fig_sim_gex.add_trace(go.Scatter(x=levels, y=np.zeros(len(levels)),
                                             mode='lines', line_color=_C['red'],
                                             name='Gamma'))
            fig_sim_gex.add_trace(go.Scatter(x=[spot], y=[0], mode='markers',
                                             marker=dict(color='#d29922', size=10, symbol='diamond'),
                                             name='Flip'))
            fig_sim_gex.update_layout(title="Gamma (GEX) + Gamma Flip",
                                      yaxis_title="$ Bi / 1% move",
                                      height=320, template=DASH_TEMPLATE,
                                      margin=dict(t=30, b=20))

            fig_sim_vega = go.FigureWidget()
            fig_sim_vega.add_trace(go.Scatter(x=levels, y=np.zeros(len(levels)),
                                              mode='lines', line_color=_C['purple'],
                                              name='Vega'))
            fig_sim_vega.update_layout(title="Vega Nocional", yaxis_title="$ Mi",
                                       height=320, template=DASH_TEMPLATE,
                                       margin=dict(t=30, b=20))

            # Flow adjustment chart (how vol ctrl / RP / CTA / dealer adjust)
            fig_sim_flows = go.FigureWidget()
            fig_sim_flows.add_trace(go.Bar(x=['Vol Ctrl', 'Risk Parity', 'CTA', 'Dealer', 'Vanna', 'Charm'],
                                           y=[0, 0, 0, 0, 0, 0],
                                           marker_color=[_C['accent'], _C['teal'], _C['orange'],
                                                         _C['purple'], _C['pink'], _C['yellow']],
                                           name='Flow ($B)'))
            fig_sim_flows.update_layout(
                title="Fluxo Estimado por Componente ($B)",
                yaxis_title="$ Bi", height=320, template=DASH_TEMPLATE,
                margin=dict(t=30, b=20))

            sim_info = wd.HTML('')

            def _update_simulator(change=None):
                v_shift = vol_slider.value / 100.0
                d_shift = dte_slider.value
                s_move = spot_slider.value / 100.0
                sim_vol = np.clip(df.IV.values + v_shift, 0.001, None)
                sim_tte = np.clip(df.Tte.values - d_shift / TRADING_DAYS, 1.0 / TRADING_DAYS, None)
                types_arr = df.Type.values
                new_spot = spot * (1 + s_move)

                dex_c, gex_c, vex_c = [], [], []
                _flip_level = None
                _prev_gex = None
                for L in levels:
                    g = calculate_all_greeks(L, df.Strike.values, sim_vol, sim_tte, types_arr, r=rfr)
                    oi_100 = df.OI.values * 100.0
                    dex_c.append(np.nansum(g['delta'] * oi_100 * L))
                    _gex_val = np.nansum(g['gamma'] * np.where(types_arr == 'Call', 1, -1)
                                         * oi_100 * (L**2) * 0.01)
                    gex_c.append(_gex_val)
                    vex_c.append(np.nansum(g['vega'] * oi_100))
                    if _prev_gex is not None and _flip_level is None:
                        if (_prev_gex < 0 and _gex_val >= 0) or (_prev_gex > 0 and _gex_val <= 0):
                            _flip_level = L
                    _prev_gex = _gex_val

                dex_arr = np.array(dex_c) / 1e9
                gex_arr = np.array(gex_c) / 1e9
                vex_arr = np.array(vex_c) / 1e6

                with fig_sim_dex.batch_update():
                    fig_sim_dex.data[0].y = dex_arr
                    fig_sim_dex.data[1].x = [new_spot]
                    idx_ns = np.argmin(np.abs(levels - new_spot))
                    fig_sim_dex.data[1].y = [dex_arr[idx_ns]] if idx_ns < len(dex_arr) else [0]
                with fig_sim_gex.batch_update():
                    fig_sim_gex.data[0].y = gex_arr
                    if _flip_level is not None:
                        fig_sim_gex.data[1].x = [_flip_level]
                        fig_sim_gex.data[1].y = [0]
                    else:
                        fig_sim_gex.data[1].x = [new_spot]
                        idx_ns2 = np.argmin(np.abs(levels - new_spot))
                        fig_sim_gex.data[1].y = [gex_arr[idx_ns2]] if idx_ns2 < len(gex_arr) else [0]
                with fig_sim_vega.batch_update():
                    fig_sim_vega.data[0].y = vex_arr

                # Compute combined flow for that scenario
                ds = new_spot - spot
                rv_now = rv_30d if pd.notna(rv_30d) else 0.15
                # Vol slider shifts IV; treat it as realized vol shock too
                vol_shift_dec = v_shift  # already in decimal (slider / 100)
                rv_shock = max(rv_now + vol_shift_dec, rv_now * (1 + max(0, -s_move) * 5))
                rv_shock = min(rv_shock, 0.80)

                # Vol Control: responds to BOTH spot crash and vol rise
                vc_flow = 0
                for _tv in [5, 10, 15]:
                    _tv_d = _tv / 100.0
                    _e0 = _vc_exposure(_tv_d, rv_now)
                    _e1 = _vc_exposure(_tv_d, rv_shock)
                    vc_flow += VOL_CTRL_AUM.get(_tv, 100e9) * (_e1 - _e0)

                # Risk Parity: responds to vol changes (inverse-vol allocation)
                rp_result = compute_risk_parity_flow(rv_shock, rv_now)
                rp_flow = rp_result['total']

                # CTA: responds to spot trends
                cta_flow = 0
                if s_move < -0.03:
                    cta_flow = s_move * CTA_AUM * CTA_EQUITY_ALLOC * 0.5
                elif s_move > 0.03:
                    cta_flow = s_move * CTA_AUM * CTA_EQUITY_ALLOC * 0.3

                _oi_sim = df.OI.values * 100
                _cs_sim = np.where(types_arr == 'Call', 1, -1)
                g_new = calculate_all_greeks(new_spot, df.Strike.values, sim_vol, sim_tte, types_arr, r=rfr)
                _gex_new = float(np.nansum(g_new['gamma'] * _cs_sim * _oi_sim))
                _dex_new = float(np.nansum(g_new['delta'] * _oi_sim))
                dealer_flow = -(_dex_new * ds + 0.5 * _gex_new * ds ** 2)

                vanna_not = float(np.nansum(g_new['vanna'] * _oi_sim) * new_spot)
                vol_chg_dec = rv_shock - rv_now  # decimal (0.02 = 2 pts)
                vanna_flow = -vanna_not * vol_chg_dec if vanna_not != 0 else 0

                charm_not_sim = float(np.nansum(g_new['charm'] * _oi_sim) * new_spot / 365.0)
                charm_flow = -charm_not_sim  # dealers unwind decayed delta overnight

                flows = [vc_flow / 1e9, rp_flow / 1e9, cta_flow / 1e9,
                         dealer_flow / 1e9, vanna_flow / 1e9, charm_flow / 1e9]
                total_flow = sum(flows)

                with fig_sim_flows.batch_update():
                    fig_sim_flows.data[0].y = flows

                # Info panel
                regime = 'ESTABILIDADE (GEX+)' if _flip_level and new_spot > _flip_level else 'ACELERAÇÃO (GEX−)'
                flip_str = '{:,.0f}'.format(_flip_level) if _flip_level else 'N/A'
                sim_info.value = (
                    "<div class='mm-dash'><div class='mm-card'>"
                    "<h4>Cenário: SPX {:+.1f}% → {:,.0f} | Vol {:+.0f} pts | "
                    "{} dias à frente</h4>"
                    "<p>Gamma Flip: <b>{}</b> | Regime: <b>{}</b></p>"
                    "<p>Fluxo Total Estimado: <b style='color:{};'>${:+.1f}B</b> "
                    "(VC: ${:.1f}B, RP: ${:.1f}B, CTA: ${:.1f}B, "
                    "Dealer: ${:.1f}B, Vanna: ${:.1f}B, Charm: ${:.1f}B)</p>"
                    "</div></div>".format(
                        s_move * 100, new_spot, vol_slider.value, d_shift,
                        flip_str, regime,
                        '#3fb950' if total_flow > 0 else '#da3633', total_flow,
                        flows[0], flows[1], flows[2], flows[3], flows[4], flows[5]))

            vol_slider.observe(_update_simulator, names='value')
            dte_slider.observe(_update_simulator, names='value')
            spot_slider.observe(_update_simulator, names='value')
            _update_simulator()

            tab8 = wd.VBox([
                wd.HTML("<div class='mm-dash'><div class='mm-card'>"
                        "<h3>Simulador Interativo — Cenários + Fluxos</h3>"
                        "<p>Ajuste spot, vol e tempo para ver inflexões, "
                        "gamma flip e como dealers/vol-ctrl/RP/CTA ajustam.</p>"
                        "</div></div>"),
                wd.HBox([spot_slider, vol_slider, dte_slider]),
                sim_info,
                wd.HBox([fig_sim_dex, fig_sim_gex]),
                wd.HBox([fig_sim_vega, fig_sim_flows]),
            ])

            # ─── ABA 9: RELATÓRIO DE RISCO ───────────────────────────────
            oi_100 = df['OI'].values * 100.0
            delta_notional = np.nansum(greeks_now['delta'] * oi_100) * spot
            total_vega = np.nansum(greeks_now['vega'] * oi_100)
            total_zomma = np.nansum(greeks_now['zomma'] * oi_100)
            total_speed = np.nansum(greeks_now['speed'] * oi_100)
            total_charm = np.nansum(greeks_now['charm'] * oi_100) / 365.0
            total_gex_val = np.nansum(
                greeks_now['gamma'] * np.where(df.Type.values == 'Call', 1, -1)
                * oi_100 * (spot**2) * 0.01)

            vanna_impact = np.nansum(greeks_now['vanna'] * oi_100) * spot
            vanna_action = "VENDER" if vanna_impact > 0 else "COMPRAR"
            zomma_action = "MAIS INSTÁVEL" if total_zomma < 0 else "MAIS ESTÁVEL"

            gamma_regime = "N/A"
            if gamma_flip:
                gamma_regime = ("ESTABILIDADE (acima do Flip)" if spot > gamma_flip
                                else "ACELERAÇÃO (abaixo do Flip)")

            hedge_contracts = -np.nansum(greeks_now['delta'] * oi_100) / FUTURES_MULTIPLIER
            hedge_action = "COMPRAR" if hedge_contracts > 0 else "VENDER"

            # ── Lógica de decisão do MM ──
            # Bias direcional
            if abs(delta_notional) < 50e6:
                delta_signal = '🟢 NEUTRO'
                delta_rec = 'Sem ajuste necessário.'
            elif delta_notional > 0:
                delta_signal = '🔴 COMPRADO'
                delta_rec = (f'VENDER {abs(hedge_contracts):,.0f} {FUTURES_TICKER} '
                             f'ou comprar puts para reduzir exposição.')
            else:
                delta_signal = '🔵 VENDIDO'
                delta_rec = (f'COMPRAR {abs(hedge_contracts):,.0f} {FUTURES_TICKER} '
                             f'ou comprar calls para reduzir exposição.')

            # Regime de gamma
            if total_gex_val > 0:
                gex_signal = '🟢 GAMMA POSITIVO'
                gex_rec = ('Mercado tende a se estabilizar. MM vende em alta, compra em '
                           'baixa → menor volatilidade realizada. Pode VENDER vol (straddles/strangles).')
            else:
                gex_signal = '🔴 GAMMA NEGATIVO'
                gex_rec = ('Mercado tende a se ACELERAR. MM compra em alta, vende em '
                           'baixa → maior volatilidade realizada. REDUZIR tamanho e '
                           'colocar stops mais apertados.')

            # Vol trade
            if vol_premium > 2:
                vol_signal = '🟢 IV > RV (prêmio alto)'
                vol_rec = 'Oportunidade de VENDER vol — spreads de crédito, iron condors.'
            elif vol_premium < -2:
                vol_signal = '🔴 RV > IV (vol barata)'
                vol_rec = 'Oportunidade de COMPRAR vol — straddles, calendars.'
            else:
                vol_signal = '🟡 VOL NEUTRA'
                vol_rec = 'Sem edge claro em vol. Preferir posições direcionais.'

            # Skew trade
            if skew > 0.03:
                skew_signal = '🔴 SKEW ELEVADO'
                skew_rec = ('Puts caras vs calls. Considerar risk reversals '
                            '(vender put OTM, comprar call OTM) ou put spreads.')
            elif skew < 0.01:
                skew_signal = '🟢 SKEW COMPRIMIDO'
                skew_rec = 'Proteção barata. Comprar puts OTM como hedge de cauda.'
            else:
                skew_signal = '🟡 SKEW NORMAL'
                skew_rec = 'Sem distorção significativa no skew.'

            # Vanna/Zomma trade
            if vanna_impact > 0:
                vanna_rec = ('Vol subindo → dealers precisam VENDER o ativo (pressão baixista). '
                             'Cuidado com posições compradas.')
            else:
                vanna_rec = ('Vol subindo → dealers precisam COMPRAR o ativo (pressão altista). '
                             'Suporte adicional em quedas.')

            # Theta vs Gamma
            daily_theta = np.nansum(greeks_now['theta'] * oi_100) / TRADING_DAYS
            gamma_pnl_1pct = total_gex_val
            theta_gamma_ratio = abs(daily_theta / gamma_pnl_1pct) if gamma_pnl_1pct != 0 else 0
            if theta_gamma_ratio > 1:
                tg_signal = '🟢 THETA DOMINANTE'
                tg_rec = ('Decaimento diário supera risco de gamma. '
                          'Posições vendidas em vol são favorecidas.')
            else:
                tg_signal = '🔴 GAMMA DOMINANTE'
                tg_rec = ('Risco de gamma supera o decaimento. '
                          'Movimentos grandes serão caros. Reduzir exposição vendida.')

            # Urgência geral
            urgency_score = 0
            if abs(delta_notional) > 200e6: urgency_score += 2
            if total_gex_val < 0: urgency_score += 2
            if abs(vol_premium) > 3: urgency_score += 1
            if skew > 0.04: urgency_score += 1
            if urgency_score >= 4:
                urgency = '🔴 ALTA — Ajustes imediatos recomendados'
            elif urgency_score >= 2:
                urgency = '🟡 MÉDIA — Monitorar e preparar ordens'
            else:
                urgency = '🟢 BAIXA — Posição confortável'

            report_html = f"""
            <div class='mm-dash'>
            <div class='mm-card'>
            <h3 style='font-size:18px;'>COCKPIT DO MARKET MAKER</h3>
            <p>Análise: {datetime.now().strftime('%Y-%m-%d %H:%M')} │ Spot: ${spot:,.2f} │
               {len(df)} opções │ {urgency}</p>
            </div>

            <div class='mm-card'>
            <h3>DECISÕES RECOMENDADAS</h3>
            <table class='mm-table'>
            <tr><th>Dimensão</th><th>Sinal</th><th>Ação Recomendada</th></tr>
            <tr><td><b>Delta (Direção)</b></td><td style='text-align:center;'>{delta_signal}</td><td>{delta_rec}</td></tr>
            <tr><td><b>Gamma (Regime)</b></td><td style='text-align:center;'>{gex_signal}</td><td>{gex_rec}</td></tr>
            <tr><td><b>Vol Premium</b></td><td style='text-align:center;'>{vol_signal}</td><td>{vol_rec}</td></tr>
            <tr><td><b>Skew</b></td><td style='text-align:center;'>{skew_signal}</td><td>{skew_rec}</td></tr>
            <tr><td><b>Vanna (Vol→Spot)</b></td><td style='text-align:center;'>{'🔴' if vanna_impact > 0 else '🟢'} {vanna_action}</td><td>{vanna_rec}</td></tr>
            <tr><td><b>Theta vs Gamma</b></td><td style='text-align:center;'>{tg_signal}</td><td>{tg_rec}</td></tr>
            </table>
            </div>

            <div class='mm-card'>
            <h3>POSIÇÃO ATUAL</h3>
            <table class='mm-table'>
            <tr><th>Exposição</th><th style='text-align:right;'>Valor</th><th>Interpretação</th></tr>
            <tr><td>Delta Nocional</td><td style='text-align:right;'>{fmt_value(delta_notional)}</td><td>Hedge: {hedge_action} {abs(hedge_contracts):,.0f} {FUTURES_TICKER}</td></tr>
            <tr><td>GEX Total</td><td style='text-align:right;'>{fmt_value(total_gex_val)}</td><td>Flip: ~{f'{gamma_flip:,.0f}' if gamma_flip else 'N/A'} │ {gamma_regime}</td></tr>
            <tr><td>Vega</td><td style='text-align:right;'>{fmt_value(total_vega)}</td><td>P&L por 1% de aumento na vol</td></tr>
            <tr><td>Vanna</td><td style='text-align:right;'>{fmt_value(vanna_impact)}</td><td>Fluxo de rebalanceamento se vol mudar</td></tr>
            <tr><td>Zomma</td><td style='text-align:right;'>{fmt_value(total_zomma)}</td><td>Regime ficaria {zomma_action} com vol</td></tr>
            <tr><td>Speed</td><td style='text-align:right;'>{fmt_value(total_speed)}</td><td>Aceleração do gamma por $1 no spot</td></tr>
            <tr><td>Charm (diário)</td><td style='text-align:right;'>{fmt_value(total_charm)}</td><td>Decaimento do delta overnight</td></tr>
            </table>
            </div>

            <div class='mm-card'>
            <h3>RISCO CAUDAL</h3>
            <table class='mm-table' style='width:70%;'>
            <tr><th>Métrica</th><th>Paramétrico</th><th>Monte Carlo</th></tr>
            <tr><td>VaR 95%</td><td style='color:{_C['yellow']}'>{risk['var_95']:.2%}</td><td style='color:{_C['yellow']}'>${sim_var_95/1e6:,.2f} Mi</td></tr>
            <tr><td>CVaR 95%</td><td style='color:{_C['orange']}'>{risk['cvar_95']:.2%}</td><td style='color:{_C['orange']}'>${sim_cvar_95/1e6:,.2f} Mi</td></tr>
            <tr><td>VaR 99%</td><td style='color:{_C['red']}'>{risk['var_99']:.2%}</td><td style='color:{_C['red']}'>${sim_var_99/1e6:,.2f} Mi</td></tr>
            <tr><td>CVaR 99%</td><td style='color:{_C['red']}'>{risk['cvar_99']:.2%}</td><td style='color:{_C['red']}'>${sim_cvar_99/1e6:,.2f} Mi</td></tr>
            </table>
            </div>

            <div class='mm-card'>
            <h3>NÍVEIS-CHAVE</h3>
            <div class='mm-kpi-row'>
                <div class='mm-kpi'><div class='kpi-label'>Call Wall</div><div class='kpi-value' style='color:{_C['green']}'>{f'{call_wall:,.0f}' if call_wall else 'N/A'}</div><div style='font-size:11px;color:{_C['text_muted']}'>Resistência forte</div></div>
                <div class='mm-kpi'><div class='kpi-label'>Put Wall</div><div class='kpi-value' style='color:{_C['red']}'>{f'{put_wall:,.0f}' if put_wall else 'N/A'}</div><div style='font-size:11px;color:{_C['text_muted']}'>Suporte forte</div></div>
                <div class='mm-kpi'><div class='kpi-label'>Gamma Flip</div><div class='kpi-value' style='color:{_C['yellow']}'>{f'{gamma_flip:,.0f}' if gamma_flip else 'N/A'}</div><div style='font-size:11px;color:{_C['text_muted']}'>Acima = estabilidade</div></div>
                <div class='mm-kpi'><div class='kpi-label'>Mov. Implícito 1D</div><div class='kpi-value' style='color:{_C['accent']}'>±{daily_move:.2f}%</div><div style='font-size:11px;color:{_C['text_muted']}'>±${spot * daily_move / 100:,.0f}</div></div>
            </div>
            </div>
            </div>"""

            tab9 = wd.VBox([wd.HTML(report_html)])

            # ─── ABA 10: FLOW PREDICTOR ──────────────────────────────────
            try:
              if fp_ok and fp_score is not None:
                # Score summary header
                fp_title = wd.HTML(
                    f"<div class='mm-dash'><div class='mm-card'>"
                    f"<h3 style='font-size:18px;'>Flow Predictor — {ticker}</h3>"
                    f"<p>Direction: <b style='color:"
                    f"{_C['green'] if fp_score['direction'] == 'BULLISH' else _C['red'] if fp_score['direction'] == 'BEARISH' else _C['text_muted']}'>"
                    f"{fp_score['direction']}</b> │ "
                    f"P(Up): {fp_score['prob_up']:.1%} │ "
                    f"Score: {fp_score['combined_score']:+.2f}</p>"
                    f"</div></div>")

                # Sub-tab A: Score
                st_a = wd.VBox([
                    fp_title,
                    wd.HBox([fp_plot_score_gauge(fp_score),
                             fp_plot_components_bar(fp_score)]),
                    fp_grid_flow_score(fp_score)
                ])

                # Sub-tab B: Histórico
                st_b_children = [wd.HTML("<div class='mm-dash'><div class='mm-card'><h3>Histórico de Fluxo — ETFs Alavancados</h3></div></div>")]
                if not fp_flow_hist.empty:
                    st_b_children.append(fp_plot_flow_history(fp_flow_hist))
                    st_b_children.append(fp_bqp_flow_bar_line(fp_flow_hist.tail(60)))
                    st_b_children.append(fp_bqp_scatter(fp_flow_hist))
                st_b = wd.VBox(st_b_children)

                # Sub-tab C: Buyback + Blackout
                _bo_pct = fp_buyback.get('blackout_pct', 0)
                _bo_n = fp_buyback.get('blackout_n', 0)
                _bo_total = fp_buyback.get('blackout_total', 0)
                _bo_open = 1 - _bo_pct
                _bo_color = _C['red'] if _bo_pct > 0.4 else _C['yellow'] if _bo_pct > 0.2 else _C['green']
                _bo_signal = ('🔴 BLACKOUT PESADO' if _bo_pct > 0.4
                              else '🟡 BLACKOUT MODERADO' if _bo_pct > 0.2
                              else '🟢 JANELA ABERTA')

                st_c_children = [
                    wd.HTML(
                        f"<div class='mm-dash'><div class='mm-card'>"
                        f"<h3>Estimativa de Buyback + Blackout Window</h3>"
                        f"<p>{_bo_signal} — <b style='color:{_bo_color}'>"
                        f"{_bo_pct:.0%}</b> das empresas em blackout "
                        f"({_bo_n}/{_bo_total})</p>"
                        f"<p><small>Empresas não podem recomprar ações ~{BLACKOUT_DAYS_BEFORE} dias "
                        f"antes do balanço até ~{BLACKOUT_DAYS_AFTER} dias após divulgação. "
                        f"Quando muitas estão em restrição, o fluxo de buyback cai "
                        f"significativamente.</small></p>"
                        f"</div></div>")]

                bb_pct_adv = fp_buyback.get('pct_adv_est', 0)
                bb_pct_str = 'N/A' if (bb_pct_adv is None or pd.isna(bb_pct_adv)) else f'{bb_pct_adv:.2f}%'
                _daily_open = fp_buyback.get('daily_est_open', fp_buyback.get('daily_est', 0))
                _daily_adj = fp_buyback.get('daily_est', 0)
                bb_html = (
                    f"<div class='mm-dash'><div class='mm-card'>"
                    f"<table class='mm-table' style='width:auto;'>"
                    f"<tr><td>Anunciado:</td><td style='text-align:right;'>"
                    f"${fp_buyback.get('announced', 0):,.0f}</td></tr>"
                    f"<tr><td>Estimativa diária (sem blackout):</td>"
                    f"<td style='text-align:right;'>${_daily_open:,.0f}</td></tr>"
                    f"<tr><td>Blackout ({_bo_pct:.0%} restrito):</td>"
                    f"<td style='text-align:right;color:{_bo_color}'>-{_bo_pct:.0%}</td></tr>"
                    f"<tr><td><b>Estimativa diária ajustada:</b></td>"
                    f"<td style='text-align:right;font-weight:bold;'>${_daily_adj:,.0f}</td></tr>"
                    f"<tr><td>% ADV estimado:</td><td style='text-align:right;'>"
                    f"{bb_pct_str}</td></tr>"
                    f"<tr><td>Confiança:</td><td style='text-align:right;'>"
                    f"{fp_buyback.get('confidence', 'N/A')}</td></tr>"
                    f"</table></div></div>")
                st_c_children.append(wd.HTML(bb_html))

                # Blackout curve chart
                if not fp_blackout_curve.empty:
                    _bb_annual = fp_buyback.get('announced', 0)
                    if not _bb_annual or _bb_annual <= 0:
                        _bb_annual = SPX_ANNUAL_BUYBACK_EST
                    st_c_children.append(
                        build_buyback_blackout_chart(
                            fp_blackout_curve, fp_earnings_df,
                            buyback_annual=_bb_annual))

                try:
                    bb_df = estimate_index_buyback_flow(ticker, top_n=30)
                    if not bb_df.empty:
                        st_c_children.append(wd.HTML("<h4>Top Buybacks do Índice</h4>"))
                        st_c_children.append(fp_grid_buyback(bb_df))
                except Exception:
                    pass
                st_c = wd.VBox(st_c_children)

                # Sub-tab D: COT
                st_d_children = [wd.HTML(
                    f"<div class='mm-dash'><div class='mm-card'>"
                    f"<h3>COT — Commitment of Traders</h3>"
                    f"<p>Dados agregados (soma de todas trader types). "
                    f"Report type: auto. "
                    f"<span class='mm-cot-label'>Period: <b>{cot_start_w.value} → {cot_end_w.value}</b></span></p>"
                    f"</div></div>")]
                cot_ok_fp2, cot_fut_fp2 = has_cot(ticker)
                if cot_ok_fp2 and fp_cot_df is not None and not fp_cot_df.empty:
                    st_d_children.append(
                        wd.HTML(f"<p>Futures: <b>{cot_fut_fp2}</b> — "
                                f"<b>{len(fp_cot_df)}</b> registros — "
                                f"WoW Δ Net: <b>{cot_net_change:+,.0f}</b></p>"))
                    st_d_children.append(fp_grid_cot_stats(fp_cot_stats))
                    seas = cot_seasonality(fp_cot_df)
                    st_d_children.append(wd.HBox([
                        fp_plot_positions_basket(fp_cot_df),
                        fp_plot_dispersion(seas, fp_cot_df)
                    ]))
                    st_d_children.append(fp_plot_long_short_net(fp_cot_df))
                    st_d_children.append(wd.HBox([
                        fp_plot_long_short_ratio(fp_cot_df),
                        fp_plot_correlation(fp_cot_df)
                    ]))
                    st_d_children.append(fp_plot_multi_year(fp_cot_df))
                elif cot_ok_fp2:
                    st_d_children.append(
                        wd.HTML(f"<p>COT disponível para {cot_fut_fp2}, "
                                "mas sem dados retornados.</p>"))
                else:
                    st_d_children.append(
                        wd.HTML(f"<p>{ticker} não possui dados COT vinculados. "
                                "Use os contratos selecionados abaixo.</p>"))

                if fp_selected_cot_df is not None and not fp_selected_cot_df.empty:
                    sel_label = ', '.join(selected_cots)
                    st_d_children.append(wd.HTML(f"<hr><h4>COT: {sel_label}</h4>"))
                    sel_stats = cot_summary_stats(fp_selected_cot_df)
                    st_d_children.append(fp_grid_cot_stats(sel_stats))
                    sel_seas = cot_seasonality(fp_selected_cot_df)
                    st_d_children.append(wd.HBox([
                        fp_plot_positions_basket(fp_selected_cot_df),
                        fp_plot_long_short_net(fp_selected_cot_df)
                    ]))
                    st_d_children.append(wd.HBox([
                        fp_plot_dispersion(sel_seas, fp_selected_cot_df),
                        fp_plot_multi_year(fp_selected_cot_df)
                    ]))
                st_d = wd.VBox(st_d_children)

                # Sub-tab E: Regressão OLS — LevETF Flow vs Retorno D+1
                from scipy import stats as _scipy_stats
                st_e_children = []
                if not fp_flow_hist.empty and len(fp_flow_hist) >= 20:
                    _df_reg = fp_flow_hist[['Return', 'LevETF_Flow']].dropna().copy()
                    _y_all = _df_reg['Return'].shift(-1) * 100   # next-day return %
                    _x_all = _df_reg['LevETF_Flow'] / 1e9         # $B
                    _mask  = _y_all.notna() & _x_all.notna()
                    _x_reg = _x_all[_mask].values
                    _y_reg = _y_all[_mask].values

                    _slope, _intc, _r, _pval, _se = _scipy_stats.linregress(_x_reg, _y_reg)
                    _r2    = _r ** 2
                    _tstat = _slope / (_se + 1e-10)
                    _sig   = '***' if _pval < 0.01 else '**' if _pval < 0.05 else '*' if _pval < 0.10 else 'n.s.'
                    _sig_color = _C['green'] if _pval < 0.05 else _C['orange'] if _pval < 0.10 else _C['text_muted']

                    # Scatter + regression line
                    _x_line = np.linspace(_x_reg.min(), _x_reg.max(), 100)
                    _y_line = _slope * _x_line + _intc
                    _scatter_fig = go.FigureWidget()
                    _scatter_fig.add_trace(go.Scatter(
                        x=_x_reg, y=_y_reg, mode='markers',
                        marker=dict(color=_C['accent'], size=4, opacity=0.5),
                        name='Observações'))
                    _scatter_fig.add_trace(go.Scatter(
                        x=_x_line, y=_y_line, mode='lines',
                        line=dict(color=_C['red'], width=2),
                        name=f'OLS fit (R²={_r2:.3f})'))
                    _scatter_fig.add_hline(y=0, line_color=_C['border'], line_width=1)
                    _scatter_fig.add_vline(x=0, line_color=_C['border'], line_width=1)
                    _scatter_fig.update_layout(
                        title='LevETF Flow ($B) [t] vs Retorno SPX D+1 (%)',
                        xaxis_title='Fluxo LevETF ($B) — dia t',
                        yaxis_title='Retorno SPX (%) — dia t+1',
                        height=380, template=DASH_TEMPLATE,
                        margin=dict(t=40, b=30),
                        legend=dict(font=dict(size=10)))

                    # Stats card
                    _stats_html = (
                        "<div class='mm-dash'><div class='mm-card'>"
                        "<h3>Regressão OLS: LevETF Flow → Retorno SPX D+1</h3>"
                        "<p><b>Variáveis:</b> "
                        "X = fluxo estimado de ETFs alavancados no dia t ($B) | "
                        "Y = retorno do SPX no dia seguinte t+1 (%)</p>"
                        "<table class='mm-table' style='width:auto;font-size:13px;'>"
                        f"<tr><td>N observações</td><td><b>{len(_x_reg)}</b></td></tr>"
                        f"<tr><td>R²</td><td><b>{_r2:.4f}</b> "
                        f"({_r2*100:.1f}% da variância explicada)</td></tr>"
                        f"<tr><td>Coeficiente β (slope)</td><td><b>{_slope:+.4f}</b> "
                        f"pp por $B de fluxo</td></tr>"
                        f"<tr><td>Intercepto α</td><td><b>{_intc:+.4f}%</b></td></tr>"
                        f"<tr><td>t-estatístico</td><td><b>{_tstat:+.2f}</b></td></tr>"
                        f"<tr><td>p-valor</td>"
                        f"<td><b style='color:{_sig_color}'>{_pval:.4f} ({_sig})</b></td></tr>"
                        f"<tr><td>Correlação (r)</td><td><b>{_r:+.3f}</b></td></tr>"
                        "</table>"
                        "<p><small>Sinal negativo no β é esperado: fluxo de LevETF é "
                        "contra-tendência (mean-reverting). ETF compra quando mercado caiu "
                        "→ retorno D+1 tende a ser positivo.</small></p>"
                        "</div></div>")
                    st_e_children.append(wd.HTML(_stats_html))
                    st_e_children.append(_scatter_fig)
                else:
                    st_e_children.append(wd.HTML(
                        "<p style='color:#8b949e;'>Histórico insuficiente (&lt;20 obs).</p>"))
                st_e = wd.VBox(st_e_children)

                # Sub-tab F: Fluxos Sistemáticos (CTA + Dealer + Vol Control + Risk Parity)
                print("[UI] st_f: INICIO")
                st_f_children = [wd.HTML(
                    "<div class='mm-dash'><div class='mm-card'>"
                    "<h3>Fluxos Sistemáticos — CTA, Dealer/MM, Vol Control, Risk Parity</h3>"
                    "<p><small>Ref: BofA Systematic Flows Monitor methodology</small></p>"
                    "</div></div>")]

                # CTA Trend Following
                print(f"[UI] CTA data: flow={fp_cta.get('flow',0):.0f}, "
                      f"scenarios_1w={len(fp_cta_scenarios_1w)}, "
                      f"scenarios_1m={len(fp_cta_scenarios_1m)}, "
                      f"pivots={len(fp_cta_pivots)}, "
                      f"hist={len(fp_cta_hist) if not fp_cta_hist.empty else 0}")
                try:
                    _cta_flow = fp_cta.get('flow', 0)
                    _cta_trend = fp_cta.get('trend_today', 0)
                    _cta_pos = fp_cta.get('pos_today', 0)
                    _cta_pos_prev = fp_cta.get('pos_prev', 0)
                    _cta_color = _C['green'] if _cta_flow > 0 else _C['red'] if _cta_flow < 0 else _C['text_muted']
                    _cta_dir = 'COMPRA' if _cta_flow > 0 else 'VENDA' if _cta_flow < 0 else 'FLAT'
                    _trend_bar = '█' * max(1, int(abs(_cta_trend) * 10))
                    _trend_color = _C['green'] if _cta_trend > 0 else _C['red'] if _cta_trend < 0 else _C['text_muted']

                    # ── 1. Current status summary ──
                    _cta_html = (
                        f"<div class='mm-dash'><div class='mm-card'>"
                        f"<h4>CTA / Trend Following (AUM: ~$340B, ~25% equity = ~$85B)</h4>"
                        f"<table class='mm-table' style='width:auto;'>"
                        f"<tr><td>Trend Strength:</td>"
                        f"<td style='color:{_trend_color}'><b>{_cta_trend:+.3f}</b> "
                        f"<span style='font-size:10px;'>{_trend_bar}</span></td></tr>"
                        f"<tr><td>Posição CTA:</td>"
                        f"<td>{_cta_pos_prev:+.3f}x → <b>{_cta_pos:+.3f}x</b></td></tr>"
                        f"<tr><td>Fluxo Estimado (hoje):</td>"
                        f"<td style='color:{_cta_color}'><b>${_cta_flow/1e9:,.2f}B</b> ({_cta_dir})</td></tr>"
                        f"</table>")

                    # ── 2. Scenario table (1W + 1M) ──
                    if fp_cta_scenarios_1w and fp_cta_scenarios_1m:
                        _card2 = _C['card2']
                        _border = _C['border']
                        _cta_html += (
                            f"<h4 style='margin-top:16px;'>📊 CTA Estimated Flows by Scenario</h4>"
                            f"<table class='mm-table' style='width:100%;font-size:12px;'>"
                            f"<tr style='background:{_card2};'>"
                            f"<th rowspan='2'>Cenário</th>"
                            f"<th colspan='3' style='text-align:center;border-bottom:1px solid {_border};'>1 Week</th>"
                            f"<th colspan='3' style='text-align:center;border-bottom:1px solid {_border};'>1 Month</th></tr>"
                            f"<tr style='background:{_card2};'>"
                            f"<th>SPX End</th><th>Flow ($B)</th><th>Pos End</th>"
                            f"<th>SPX End</th><th>Flow ($B)</th><th>Pos End</th></tr>")
                        for s1w, s1m in zip(fp_cta_scenarios_1w, fp_cta_scenarios_1m):
                            _f1w = s1w['flow_total']
                            _f1m = s1m['flow_total']
                            _fc1w = _C['green'] if _f1w > 0 else _C['red'] if _f1w < 0 else _C['text_muted']
                            _fc1m = _C['green'] if _f1m > 0 else _C['red'] if _f1m < 0 else _C['text_muted']
                            _row_bg = ''
                            if 'Down' in s1w['name']:
                                _row_bg = f" style='background:rgba(255,77,77,0.08);'"
                            elif 'Up' in s1w['name']:
                                _row_bg = f" style='background:rgba(0,200,100,0.08);'"
                            _cta_html += (
                                f"<tr{_row_bg}>"
                                f"<td><b>{s1w['name']}</b></td>"
                                f"<td>{s1w['spx_end']:,.0f} ({s1w['pct_move']:+.1%})</td>"
                                f"<td style='color:{_fc1w}'><b>${_f1w/1e9:,.1f}B</b></td>"
                                f"<td>{s1w['pos_end']:+.2f}x</td>"
                                f"<td>{s1m['spx_end']:,.0f} ({s1m['pct_move']:+.1%})</td>"
                                f"<td style='color:{_fc1m}'><b>${_f1m/1e9:,.1f}B</b></td>"
                                f"<td>{s1m['pos_end']:+.2f}x</td>"
                                f"</tr>")
                        _cta_html += f"</table>"

                    # ── 3. Pivot levels ──
                    if fp_cta_pivots:
                        _card2 = _C['card2']
                        _cta_html += (
                            f"<h4 style='margin-top:16px;'>🎯 CTA Pivot Levels — Trigger Thresholds</h4>"
                            f"<table class='mm-table' style='width:auto;font-size:12px;'>"
                            f"<tr style='background:{_card2};'>"
                            f"<th>Horizonte</th><th>MA Pair</th><th>Nível</th>"
                            f"<th>Tipo</th><th>Distância</th><th>Posição Atual</th></tr>")
                        for pv in fp_cta_pivots:
                            _pv_color = _C['red'] if 'SELL' in pv['type'] else _C['green']
                            _pv_icon = '🔻' if 'SELL' in pv['type'] else '🔺'
                            # Posição atual: above_now=True → fast > slow → COMPRADO
                            _pos_now = pv.get('above_now', None)
                            if _pos_now is True:
                                _pos_label = '🟢 COMPRADO'
                                _pos_color = _C['green']
                            elif _pos_now is False:
                                _pos_label = '🔴 VENDIDO'
                                _pos_color = _C['red']
                            else:
                                _pos_label = '—'
                                _pos_color = _C['text']
                            _cta_html += (
                                f"<tr>"
                                f"<td><b>{pv['label']}</b></td>"
                                f"<td>{pv['ma_pair']}</td>"
                                f"<td><b>{pv['level']:,.0f}</b></td>"
                                f"<td style='color:{_pv_color}'>{_pv_icon} {pv['type']}</td>"
                                f"<td>{pv['distance_pct']:+.1%}</td>"
                                f"<td style='color:{_pos_color};font-weight:bold'>{_pos_label}</td>"
                                f"</tr>")
                        _cta_html += (
                            f"</table>"
                            f"<p><small>Nível de preço que causaria flip do sinal de MA cross. "
                            f"Posição Atual = lado corrente do CTA. "
                            f"Mais próximo do spot = maior risco de trigger.</small></p>")

                    _cta_html += (
                        f"<p><small>Trend = média de MA crosses (5/20, 5/60, 10/60, 20/120, 20/200). "
                        f"Sizing = trend/vol. CTAs ajustam diariamente. "
                        f"Vol spike em cenários Down: RV_end = RV × (1 + |move| × 3).</small></p>"
                        f"</div></div>")
                    st_f_children.append(wd.HTML(_cta_html))

                    # ── 4. CTA chart: historical + scenario fan ──
                    _has_hist = (not fp_cta_hist.empty and len(fp_cta_hist) > 5)
                    _has_scen = bool(fp_cta_scenarios_1w and fp_cta_scenarios_1m)
                    if _has_hist or _has_scen:
                        _gs_fig = build_cta_gs_chart(
                            fp_cta_hist, fp_cta_scenarios_1w,
                            fp_cta_scenarios_1m, spot)
                        st_f_children.append(_flow_border(go.FigureWidget(_gs_fig)))
                except Exception as _cta_err:
                    _tb = traceback.format_exc()
                    print(f"⚠️ CTA rendering: {_cta_err}\n{_tb}")
                    st_f_children.append(wd.HTML(
                        f"<div class='mm-dash'><div class='mm-card'>"
                        f"<h4>CTA / Trend Following</h4>"
                        f"<p style='color:red'>Erro ao renderizar CTA: {_cta_err}</p>"
                        f"</div></div>"))

                # Dealer flow + MM VaR by book + Options volume
                _dl_color = _C['green'] if fp_dealer_flow > 0 else _C['red'] if fp_dealer_flow < 0 else _C['text_muted']
                _dl_dir = 'COMPRA' if fp_dealer_flow > 0 else 'VENDA' if fp_dealer_flow < 0 else 'NEUTRO'
                _card2 = _C['card2']

                _dl_html = (
                    f"<div class='mm-dash'><div class='mm-card'>"
                    f"<h4>Dealer/Market Maker Delta Hedging</h4>"
                    f"<p>GEX-implied flow: <b style='color:{_dl_color}'>"
                    f"${fp_dealer_flow:,.0f}</b> ({_dl_dir})</p>"
                    f"<p><small>Baseado em GEX × ΔS. Dados de gamma derivados de "
                    f"signed volume (firms + MMs + flex). Short gamma → pro-cíclico.</small></p>")

                # ── MM VaR by book ──
                if fp_mm_var:
                    _v95_total = fp_mm_var_totals.get('pnl_var95', 0)
                    _v99_total = fp_mm_var_totals.get('pnl_var99', 0)
                    _dl_html += (
                        f"<h4 style='margin-top:16px;'>📊 VaR por Market Maker (proporcional ao volume)</h4>"
                        f"<p><small>VaR total (gamma exposure): "
                        f"<b>95%: ${_v95_total/1e6:,.1f}M</b> | "
                        f"<b>99%: ${_v99_total/1e6:,.1f}M</b></small></p>"
                        f"<table class='mm-table' style='width:100%;font-size:12px;'>"
                        f"<tr style='background:{_card2};'>"
                        f"<th>Market Maker</th>"
                        f"<th>Share</th>"
                        f"<th>OI (K cts)</th>"
                        f"<th>GEX/pt</th>"
                        f"<th>VaR 95%</th>"
                        f"<th>VaR 99%</th>"
                        f"<th>CVaR 95%</th>"
                        f"<th>Theta/dia</th></tr>")
                    for mm in fp_mm_var:
                        _mm_theta = mm['daily_theta']
                        _th_c = _C['green'] if _mm_theta > 0 else _C['red']
                        _dl_html += (
                            f"<tr>"
                            f"<td><b>{mm['name']}</b></td>"
                            f"<td style='text-align:center;'>{mm['share']:.0%}</td>"
                            f"<td style='text-align:right;'>{mm['oi_contracts']/1e3:,.0f}</td>"
                            f"<td style='text-align:right;'>{mm['gex_per_pt']:,.0f}</td>"
                            f"<td style='text-align:right;color:{_C['yellow']}'>"
                            f"${mm['var_95']/1e6:,.1f}M</td>"
                            f"<td style='text-align:right;color:{_C['red']}'>"
                            f"${mm['var_99']/1e6:,.1f}M</td>"
                            f"<td style='text-align:right;color:{_C['red']}'>"
                            f"${mm['cvar_95']/1e6:,.1f}M</td>"
                            f"<td style='text-align:right;color:{_th_c}'>"
                            f"${_mm_theta/1e6:,.2f}M</td>"
                            f"</tr>")
                    _dl_html += (
                        f"</table>"
                        f"<p><small>Volume shares: OVME (Bloomberg). "
                        f"VaR = 0.5 × GEX × ΔS² na pior perda (t-Student). "
                        f"Proporcional ao share de cada MM.</small></p>")

                # ── Options Volume by Trade Description ──
                _total_adc = fp_vol_data.get('total_adc', OPTIONS_TOTAL_ADC)
                _vol_src = fp_vol_data.get('source', 'fallback')
                _cv = fp_vol_data.get('call_vol', 0)
                _pv = fp_vol_data.get('put_vol', 0)
                _pcr = fp_vol_data.get('pc_ratio', 0)
                _vol_extra = ''
                if _vol_src == 'BQL':
                    _vol_extra = (
                        f" │ Call: {_cv:,.0f} │ Put: {_pv:,.0f}"
                        f" │ P/C: {_pcr:.2f}")
                _dl_html += (
                    f"<h4 style='margin-top:16px;'>📈 Mercado de Opções — Volume por Trade Type</h4>"
                    f"<p>Total Avg Daily Contracts: <b>{_total_adc:,.0f}</b>"
                    f" <small>({_vol_src}{_vol_extra})</small></p>"
                    f"<table class='mm-table' style='width:100%;font-size:12px;'>"
                    f"<tr style='background:{_card2};'>"
                    f"<th>Trade Description</th>"
                    f"<th>%</th>"
                    f"<th>Avg Daily Contracts</th></tr>")
                for _td_name, _td_pct in OPTIONS_TRADE_DESC:
                    _td_vol = _total_adc * _td_pct / 100
                    _bar_w = max(1, int(_td_pct * 2))
                    _dl_html += (
                        f"<tr>"
                        f"<td>{_td_name}</td>"
                        f"<td style='text-align:right;'><b>{_td_pct:.2f}%</b></td>"
                        f"<td style='text-align:right;'>{_td_vol:,.0f}"
                        f"<span style='display:inline-block;width:{_bar_w}px;"
                        f"height:10px;background:{_C['green']};margin-left:6px;"
                        f"vertical-align:middle;border-radius:2px;'></span></td>"
                        f"</tr>")
                _dl_html += (
                    f"</table>"
                    f"<p><small>Fonte: Bloomberg OVME. Electronic inclui 0DTE + 10-14DTE. "
                    f"Multi-leg = spreads, combos, butterflies.</small></p>")

                _dl_html += f"</div></div>"
                st_f_children.append(wd.HTML(_dl_html))

                # Vol control
                vc_total = fp_volctrl.get('total', 0)
                _vc_color = _C['green'] if vc_total > 0 else _C['red'] if vc_total < 0 else _C['text_muted']
                vc_rows = ""
                for tv_k, tv_v in fp_volctrl.get('detail', {}).items():
                    tv_flow = tv_v.get('flow', 0)
                    tv_c = _C['green'] if tv_flow > 0 else _C['red'] if tv_flow < 0 else _C['text_muted']
                    _tv_daily = tv_v.get('daily_flow', 0)
                    _tv_dc = _C['green'] if _tv_daily > 0 else _C['red'] if _tv_daily < 0 else _C['text_muted']
                    vc_rows += (
                        f"<tr><td>Target {tv_k}</td>"
                        f"<td style='text-align:right;'>${tv_v.get('aum', 0)/1e9:,.0f}B</td>"
                        f"<td style='text-align:right;'>{tv_v.get('exposure_old', 0):.2f}x</td>"
                        f"<td style='text-align:right;'>{tv_v.get('exposure_new', 0):.2f}x</td>"
                        f"<td style='text-align:right;color:{tv_c}'>${tv_flow/1e9:,.2f}B</td>"
                        f"<td style='text-align:right;color:{_tv_dc}'>${_tv_daily/1e9:,.2f}B</td></tr>")
                _vc_daily_total = fp_volctrl.get('daily_total', 0)
                _vc_dtc = _C['green'] if _vc_daily_total > 0 else _C['red'] if _vc_daily_total < 0 else _C['text_muted']
                st_f_children.append(wd.HTML(
                    f"<div class='mm-dash'><div class='mm-card'>"
                    f"<h4>Equity Vol Control (AUM: ~$300B)</h4>"
                    f"<p>Fluxo total estimado: <b style='color:{_vc_color}'>"
                    f"${vc_total/1e9:,.2f}B</b> | Fluxo/dia: <b style='color:{_vc_dtc}'>"
                    f"${_vc_daily_total/1e9:,.2f}B</b></p>"
                    f"<table class='mm-table'>"
                    f"<tr><th>Target Vol</th><th>AUM Est.</th>"
                    f"<th>Exp. Anterior</th><th>Exp. Atual</th><th>Fluxo Total</th><th>Fluxo/Dia</th></tr>"
                    f"{vc_rows}</table>"
                    f"<p><small>Leverage = target_vol / realized_vol (21d). Piso mínimo 20% exposição. "
                    f"Vol sobe → exposure cai → vendem. Ajuste ~25%/dia (~4 dias para completar).</small></p>"
                    f"</div></div>"))

                # Vol control stress scenarios
                if fp_vc_scenarios:
                    sc_rows = ""
                    for sc in fp_vc_scenarios:
                        sc_c = _C['red'] if sc['flow'] < 0 else _C['green']
                        _sc_df = sc.get('daily_flow', 0)
                        _sc_dc = _C['red'] if _sc_df < 0 else _C['green']
                        sc_rows += (
                            f"<tr><td style='text-align:center;'>{sc['rv_shock']:.0%}</td>"
                            f"<td style='text-align:right;color:{sc_c}'>"
                            f"${sc['flow']/1e9:,.1f}B</td>"
                            f"<td style='text-align:right;color:{_sc_dc}'>"
                            f"${_sc_df/1e9:,.1f}B</td></tr>")
                    st_f_children.append(wd.HTML(
                        f"<div class='mm-dash'><div class='mm-card'>"
                        f"<h4>⚡ Vol Spike Scenarios — Venda Forçada</h4>"
                        f"<p><small>Se realized vol subir para esses níveis, "
                        f"quanto os fundos vol-control precisam vender:</small></p>"
                        f"<table class='mm-table'>"
                        f"<tr><th>RV Spike Para</th><th>Fluxo Total</th><th>Fluxo/Dia</th></tr>"
                        f"{sc_rows}</table>"
                        f"<p><small>Piso mínimo 20% exposição. "
                        f"Ajuste gradual ~25%/dia (~4 dias para completar).</small></p>"
                        f"</div></div>"))

                # Risk Parity
                rp_total = fp_rp.get('total', 0)
                _rp_color = _C['green'] if rp_total > 0 else _C['red'] if rp_total < 0 else _C['text_muted']
                rp_eq_new = fp_rp.get('eq_alloc_new', 0)
                rp_eq_old = fp_rp.get('eq_alloc_old', 0)
                rp_rows = ""
                for rp_k, rp_v in fp_rp.get('detail', {}).items():
                    rp_f = rp_v.get('flow', 0)
                    rp_c = _C['green'] if rp_f > 0 else _C['red'] if rp_f < 0 else _C['text_muted']
                    rp_rows += (
                        f"<tr><td>Target {rp_k}</td>"
                        f"<td style='text-align:right;'>${rp_v.get('aum', 0)/1e9:,.0f}B</td>"
                        f"<td style='text-align:right;'>{rp_v.get('leverage_old', 0):.2f}x</td>"
                        f"<td style='text-align:right;'>{rp_v.get('leverage_new', 0):.2f}x</td>"
                        f"<td style='text-align:right;'>{rp_v.get('eq_alloc_old', 0):.1%}</td>"
                        f"<td style='text-align:right;'>{rp_v.get('eq_alloc_new', 0):.1%}</td>"
                        f"<td style='text-align:right;color:{rp_c}'>${rp_f/1e9:,.2f}B</td></tr>")
                st_f_children.append(wd.HTML(
                    f"<div class='mm-dash'><div class='mm-card'>"
                    f"<h4>Risk Parity (AUM: ~$200B, equity alloc: {rp_eq_old:.1%} → {rp_eq_new:.1%})</h4>"
                    f"<p>Fluxo equity estimado: <b style='color:{_rp_color}'>"
                    f"${rp_total/1e9:,.2f}B</b></p>"
                    f"<table class='mm-table'>"
                    f"<tr><th>Target Vol</th><th>AUM Est.</th>"
                    f"<th>Lev. Ant.</th><th>Lev. Atual</th>"
                    f"<th>Eq% Ant.</th><th>Eq% Atual</th><th>Fluxo</th></tr>"
                    f"{rp_rows}</table>"
                    f"<p><small>Alocação ∝ 1/vol por asset class (equities, bonds, commodities). "
                    f"Rebalanceamento mensal. Vol↑ → equity alloc↓ → vendem.</small></p>"
                    f"</div></div>"))

                # Combined flow scenarios table
                if fp_combined_scenarios:
                    cs_rows = ""
                    for cs in fp_combined_scenarios:
                        _cs_color = _C['red'] if cs['total'] < 0 else _C['green']
                        _cs_vanna = cs.get('vanna', 0)
                        _cs_charm = cs.get('charm', 0)
                        cs_rows += (
                            f"<tr>"
                            f"<td>{cs['name']}</td>"
                            f"<td style='text-align:center;'>{cs['spx_move']:+.0%}</td>"
                            f"<td style='text-align:center;'>{cs['rv_shock']:.0%}</td>"
                            f"<td style='text-align:right;color:{_C['red'] if cs['vol_ctrl'] < 0 else _C['text_muted']}'>"
                            f"${cs['vol_ctrl']/1e9:,.1f}B</td>"
                            f"<td style='text-align:right;color:{_C['red'] if cs['risk_parity'] < 0 else _C['text_muted']}'>"
                            f"${cs['risk_parity']/1e9:,.1f}B</td>"
                            f"<td style='text-align:right;color:{_C['red'] if cs['cta'] < 0 else _C['text_muted']}'>"
                            f"${cs['cta']/1e9:,.1f}B</td>"
                            f"<td style='text-align:right;color:{_C['red'] if cs['dealer'] < 0 else _C['text_muted']}'>"
                            f"${cs['dealer']/1e9:,.1f}B</td>"
                            f"<td style='text-align:right;color:{_C['red'] if _cs_vanna < 0 else _C['text_muted']}'>"
                            f"${_cs_vanna/1e9:,.1f}B</td>"
                            f"<td style='text-align:right;color:{_C['red'] if _cs_charm < 0 else _C['text_muted']}'>"
                            f"${_cs_charm/1e9:,.1f}B</td>"
                            f"<td style='text-align:right;color:{_cs_color};font-weight:bold'>"
                            f"${cs['total']/1e9:,.1f}B</td>"
                            f"</tr>")
                    st_f_children.append(wd.HTML(
                        f"<div class='mm-dash'><div class='mm-card'>"
                        f"<h4>⚡ Cenários de Stress — Fluxo Combinado por Componente</h4>"
                        f"<p><small>Estimativa de venda forçada se SPX cair e vol subir. "
                        f"Mostra fluxo de cada estratégia sistemática + vanna de opções:</small></p>"
                        f"<table class='mm-table'>"
                        f"<tr><th>Cenário</th><th>SPX</th><th>RV</th>"
                        f"<th>Vol Ctrl</th><th>Risk Parity</th>"
                        f"<th>CTA</th><th>Dealer</th><th>Vanna</th><th>Charm</th>"
                        f"<th>TOTAL</th></tr>"
                        f"{cs_rows}</table>"
                        f"<p><small>Vol Ctrl/RP: ajuste ~25%/dia (~4d). CTA: ajustam diariamente. "
                        f"Dealer: instantâneo (delta hedge). Vanna: -vanna_notional × ΔVol. "
                        f"Charm: -charm_notional (decay diário do delta, constante). "
                        f"Valores estimados, não garantidos.</small></p>"
                        f"</div></div>"))

                print(f"[UI] st_f: {len(st_f_children)} children")
                st_f = wd.VBox(st_f_children)

                # ─── Sub-tab Dispersão (st_g) ────────────────────────────
                st_g_children = []
                if disp_ok:
                    try:
                        # Dispersion signal chart
                        disp_chart = build_dispersion_chart(
                            disp_result['disp_signal'],
                            disp_result.get('impl_corr_cboe'))
                        st_g_children.append(disp_chart)

                        # Correlation regime chart
                        if not disp_result['real_corr'].empty:
                            corr_chart = build_corr_regime_chart(
                                disp_result['real_corr'])
                            st_g_children.append(corr_chart)

                        # Current signal
                        if not disp_result['disp_signal'].empty:
                            last_row = disp_result['disp_signal'].iloc[-1]
                            sig_color = '#238636' if 'SHORT' in str(last_row.get('signal', '')) else (
                                '#da3633' if 'LONG' in str(last_row.get('signal', '')) else '#8b949e')
                            _ic = last_row.get('impl_corr', 0)
                            _rc = last_row.get('real_corr', 0)
                            _sp = last_row.get('spread', 0)
                            _zs = last_row.get('z_score', 0)
                            _sg = last_row.get('signal', 'N/A')
                            st_g_children.append(wd.HTML(
                                f"<div style='background:#161b22; padding:10px; "
                                f"border-radius:6px; margin:5px 0;'>"
                                f"<b style='color:{sig_color}; font-size:14px;'>"
                                f"Sinal Atual: {_sg}</b><br>"
                                f"<span style='color:#c9d1d9; font-size:12px;'>"
                                f"Impl Corr: {_ic:.3f} │ Real Corr: {_rc:.3f} │ "
                                f"Spread: {_sp:.3f} │ Z-Score: {_zs:.2f}</span>"
                                f"</div>"))

                        # Mag7 pairs dispersion table
                        if not disp_result['mag7_pairs'].empty:
                            st_g_children.append(_disp_table_widget(
                                disp_result['mag7_pairs'],
                                title='Mag7 — Dispersão por Par'))

                        # Best NxN dispersion trade (2x2, 3x3, etc.)
                        if disp_result['best_2x2']:
                            b22_df = pd.DataFrame(disp_result['best_2x2'])
                            st_g_children.append(_disp_table_widget(
                                b22_df,
                                title='Melhor Trade NxN (Mag7) — ⭐ = Combo Ótimo'))

                        # Best pair combos (1-pair, 2-pair, 3-pair)
                        if disp_result.get('best_pairs'):
                            bp_df = pd.DataFrame(disp_result['best_pairs'])
                            st_g_children.append(_disp_table_widget(
                                bp_df,
                                title='Melhores Combinações de Pares (até 3)'))

                        # Optimal tracking basket
                        ob = disp_result.get('optimal_basket', {})
                        if ob.get('weights'):
                            ob_df = pd.DataFrame([
                                {'Ticker': t.split(' ')[0],
                                 'Peso': round(w * 100, 1)}
                                for t, w in sorted(ob['weights'].items(),
                                                   key=lambda x: -x[1])
                            ])
                            te_val = ob.get('tracking_error', 0)
                            st_g_children.append(_disp_table_widget(
                                ob_df,
                                title='Basket Ótimo ({} stocks, TE={:.4f})'.format(
                                    len(ob_df), te_val)))

                        # Hypothesis test for R² (quality of dispersion model)
                        ht = disp_result.get('hyp_test', {})
                        if ht:
                            _r2 = ht.get('R²', 0)
                            _r2_color = _C['green'] if _r2 >= 0.5 else (
                                '#f0883e' if _r2 >= 0.3 else _C['red'])
                            _sig_txt = '✅ Significativo' if ht.get('significant') else '❌ Não significativo'
                            _sig_color = _C['green'] if ht.get('significant') else _C['red']
                            st_g_children.append(wd.HTML(
                                f"<div style='background:#161b22; padding:10px; "
                                f"border-radius:6px; margin:5px 0;'>"
                                f"<b style='color:#58a6ff;'>📊 Teste de Hipótese — Modelo de Dispersão</b><br>"
                                f"<span style='color:#c9d1d9; font-size:12px;'>"
                                f"R² = <b style='color:{_r2_color}'>{_r2:.4f}</b> │ "
                                f"R² adj = {ht.get('R² adj', 0):.4f} │ "
                                f"F-stat = {ht.get('F-stat', 0):.2f} │ "
                                f"p-value = {ht.get('p-value', 'N/A')} │ "
                                f"<b style='color:{_sig_color}'>{_sig_txt}</b><br>"
                                f"n = {ht.get('n_obs', 0)} │ "
                                f"slope = {ht.get('slope', 0):.4f} │ "
                                f"intercept = {ht.get('intercept', 0):.4f}</span>"
                                f"<br><small style='color:#8b949e;'>Top-{DISP_TOP_N} membros por peso. "
                                f"R² &lt; 0.30 → modelo fraco, considere usar Mag8 / top 5.</small>"
                                f"</div>"))

                        # Bloomberg-style COR1M + DSPX + VIXEQ chart
                        _cor1m = disp_result.get('cor1m', pd.Series(dtype=float))
                        _dspx = disp_result.get('dspx', pd.Series(dtype=float))
                        _vixeq = disp_result.get('vixeq', pd.Series(dtype=float))
                        if not _cor1m.empty or not _dspx.empty or not _vixeq.empty:
                            st_g_children.append(build_dispersion_index_chart(
                                _cor1m, _dspx, _vixeq))

                        # ── Multi-window correlation heatmaps ──
                        corr_mats = disp_result.get('corr_matrices', {})
                        if corr_mats:
                            heatmap_widgets = []
                            for lbl, cmat in corr_mats.items():
                                if cmat is not None and not cmat.empty:
                                    heatmap_widgets.append(
                                        build_correlation_heatmap(cmat, title=f'Correlação — {lbl}'))
                            if heatmap_widgets:
                                st_g_children.append(wd.HBox(heatmap_widgets))

                        # ── Dispersion pairs (multi-window scoring) ──
                        dp = disp_result.get('dispersion_pairs', pd.DataFrame())
                        if not dp.empty:
                            st_g_children.append(_disp_table_widget(
                                dp[['Pair', 'Avg Corr', 'Min Corr', 'IV1 (%)',
                                    'IV2 (%)', 'IV Spread (pp)', 'Disp Score']],
                                title='Top Pares para Dispersão (Multi-Window)'))

                        # ── Straddle richness chart ──
                        rich_df = disp_result.get('straddle_richness', pd.DataFrame())
                        if not rich_df.empty:
                            st_g_children.append(build_straddle_richness_chart(rich_df))
                            st_g_children.append(_disp_table_widget(
                                rich_df,
                                title='Straddle ATM — Caro vs Barato (Mag8 + SPX)'))

                        # ── ATM Vol Matrix (Call vs Put IV) ──
                        atm_chart = disp_result.get('atm_vol_chart')
                        atm_matrix = disp_result.get('atm_vol_matrix')
                        if atm_chart is not None:
                            st_g_children.append(atm_chart)
                        if atm_matrix is not None and not atm_matrix.empty:
                            st_g_children.append(_disp_table_widget(
                                atm_matrix,
                                title='ATM Vol Matrix — Call IV vs Put IV (Mag8 + SPX)'))

                        # ── Intraday / Recent Price Chart (Mag8 + SPX) ──
                        _px_df_all = disp_result.get('prices_df', pd.DataFrame())
                        if not _px_df_all.empty and len(_px_df_all) >= 5:
                            st_g_children.append(build_intraday_mag8_chart(_px_df_all))

                        # ── Trade recommendations + interpretation ──
                        trade_recs = disp_result.get('trade_recs', pd.DataFrame())
                        trade_interp = disp_result.get('trade_interp', '')
                        if trade_interp:
                            st_g_children.append(wd.HTML(trade_interp))
                        if not trade_recs.empty:
                            st_g_children.append(_disp_table_widget(
                                trade_recs,
                                title='🎯 Recomendações de Dispersão — Straddle/Strangle'))

                        # ── ML Dispersion Model ──
                        ml = disp_result.get('ml_model', {})
                        if ml:
                            st_g_children.append(build_dispersion_ml_widget(
                                ml['accuracy'], ml['feature_importance'],
                                ml['disp_prob'], ml['features']))

                        # ── Breadth Summary Cards + KDE return distribution ──
                        _px_df = disp_result.get('prices_df', pd.DataFrame())
                        _wts = disp_result.get('weights', {})
                        if not _px_df.empty and len(_px_df) >= 2:
                            # Breadth summary cards
                            try:
                                st_g_children.append(
                                    build_mbad_summary_cards(_px_df, _wts))
                            except Exception:
                                pass

                            # KDE distribution
                            _kde_result = build_kde_distribution_chart(_px_df, _wts)
                            if isinstance(_kde_result, tuple):
                                kde_chart, kde_interp = _kde_result
                                st_g_children.append(kde_chart)
                                if kde_interp:
                                    st_g_children.append(wd.HTML(kde_interp))
                            else:
                                st_g_children.append(_kde_result)

                        # ── RV 1M × Gamma Scatter ──
                        if not gamma_hist.empty:
                            try:
                                _cur_rv_d = rv_30d if pd.notna(rv_30d) else None
                                _cur_gex_d = total_gex_val / 1e9 if 'total_gex_val' in dir() else None
                                st_g_children.append(
                                    build_rv_gamma_chart(gamma_hist,
                                                         current_gamma=_cur_gex_d,
                                                         current_rv=_cur_rv_d))
                            except Exception:
                                pass

                        # ── SPY Intraday Candlestick ──
                        try:
                            st_g_children.append(
                                build_spy_intraday_candlestick('SPY US Equity', lookback_days=5))
                        except Exception:
                            pass

                    except Exception as _dsp_err:
                        st_g_children.append(wd.HTML(
                            f"<p style='color:#da3633;'>Erro na UI de dispersão: "
                            f"{_dsp_err}</p>"))
                else:
                    if disp_w.value:
                        _de = disp_result.get('error', 'Erro desconhecido')
                        st_g_children.append(wd.HTML(
                            f"<p style='color:#8b949e;'>Dispersão: {_de}</p>"))
                    else:
                        st_g_children.append(wd.HTML(
                            "<p style='color:#8b949e;'>Marque 'Incluir Dispersão' "
                            "para analisar.</p>"))
                st_g = wd.VBox(st_g_children)

                fp_tabs = wd.Tab()
                fp_tabs.children = [st_a, st_b, st_c, st_d, st_e, st_f, st_g]
                for idx_t, nm in enumerate(['Score', 'Histórico', 'Buyback',
                                            'COT', 'Correlação', 'Sistemáticos',
                                            'Dispersão']):
                    fp_tabs.set_title(idx_t, nm)
                tab10 = fp_tabs
              else:
                reason = ("Marque 'Incluir Flow Predictor' e rode novamente."
                          if not flow_pred_w.value else "Erro na execução.")
                tab10 = wd.VBox([wd.HTML(
                    f"<h3>Flow Predictor</h3><p>{reason}</p>")])
            except Exception as _fp_ui_err:
                _fp_tb = traceback.format_exc()
                print(f"⚠️ Flow Predictor UI: {_fp_ui_err}\n{_fp_tb}")
                tab10 = wd.VBox([wd.HTML(
                    f"<h3>Flow Predictor</h3>"
                    f"<p style='color:red;'>Erro ao montar UI: {_fp_ui_err}</p>"
                    f"<pre style='font-size:10px;color:#aaa;white-space:pre-wrap;'>{_fp_tb}</pre>")])

            # ─── ABA 11: ANALYTICS AVANÇADO ─────────────────────────────
            try:
                analytics_children = []

                # ── Row 1: Tail Risk Gauge + Skew Summary ──
                tail_gauge = build_tail_gauge(
                    analytics['tail_score'], analytics['tail_interp'])
                tail_info_parts = ['<h3>Tail Risk</h3>']
                tail_info_parts.append(
                    '<p><b>Score:</b> {:.0f}/100 — {}</p>'.format(
                        analytics['tail_score'], analytics['tail_interp']))
                for comp_key, comp_val in analytics['tail_components'].items():
                    label = comp_val.get('label', comp_key)
                    val = comp_val.get('value', 0)
                    scr = comp_val.get('score', 0)
                    tail_info_parts.append(
                        '<p style="margin:2px 0;font-size:12px;">'
                        '{}: <b>{}</b> (contrib: {:.1f})</p>'.format(label, val, scr))
                tail_info_html = wd.HTML(
                    "<div class='mm-dash'><div class='mm-card'>"
                    "{}</div></div>".format(''.join(tail_info_parts)))

                skew_summary_parts = ['<h3>Skew Monitor</h3>']
                sk = analytics['skew_summary']
                if sk:
                    for key in ['atm_iv', 'put25d', 'call25d', 'risk_reversal', 'put_skew', 'call_skew']:
                        if key in sk:
                            pctile_key = '{}_pctile'.format(key)
                            if pctile_key in sk:
                                raw_pct = sk[pctile_key]
                                # call_skew: percentil invertido (baixo = calls baratas = extremo)
                                disp_pct = 100 - raw_pct if key == 'call_skew' else raw_pct
                                pctile_str = ' (pctile: {:.0f}%)'.format(disp_pct)
                            else:
                                pctile_str = ''
                            skew_summary_parts.append(
                                '<p style="margin:2px 0;font-size:12px;">'
                                '{}: <b>{}</b>{}</p>'.format(key.replace('_', ' ').title(), sk[key], pctile_str))
                else:
                    skew_summary_parts.append('<p style="color:#8b949e;">Sem dados de skew</p>')
                skew_summary_widget = wd.HTML(
                    "<div class='mm-dash'><div class='mm-card'>"
                    "{}</div></div>".format(''.join(skew_summary_parts)))

                analytics_children.append(wd.HBox([tail_gauge, tail_info_html, skew_summary_widget]))

                # ── Row 2: Skew 4-Panel Chart ──
                skew_chart = build_skew_chart(analytics['skew_df'])
                analytics_children.append(skew_chart)

                # ── Row 3: Spot-Up-Vol-Up + VIX Regression ──
                suvu = analytics['spot_vol_up']
                suvu_parts = [
                    '<h3>Spot Up / Vol Up</h3>',
                    '<p>Streak Atual: <b style="color:#f0883e;">{}</b> dias</p>'.format(
                        suvu.get('current_streak', 0)),
                    '<p>Streak Máximo: <b>{}</b></p>'.format(suvu.get('max_streak', 0)),
                    '<p>Total dias Up/Up: {} ({:.1f}%)</p>'.format(
                        suvu.get('total_days', 0), suvu.get('pct_up_up', 0)),
                ]
                suvu_widget = wd.HTML(
                    "<div class='mm-dash'><div class='mm-card'>"
                    "{}</div></div>".format(''.join(suvu_parts)))

                vix_reg = analytics['vix_reg']
                vix_parts = ['<h3>VIX vs SPX Regressão (1M)</h3>']
                if vix_reg:
                    vix_parts.append(
                        '<p>R²: <b>{:.1%}</b></p>'.format(vix_reg.get('r2', 0)))
                    vix_parts.append(
                        '<p>Slope: {} | Intercept: {}</p>'.format(
                            vix_reg.get('slope', 0), vix_reg.get('intercept', 0)))
                    vix_parts.append(
                        '<p>SPX 1M: {:.1f}% → VIX pred: <b>{:+.1f} pts</b></p>'.format(
                            vix_reg.get('last_spx_1m', 0), vix_reg.get('predicted_vix_move', 0)))
                else:
                    vix_parts.append('<p style="color:#8b949e;">Sem dados VIX</p>')
                vix_widget = wd.HTML(
                    "<div class='mm-dash'><div class='mm-card'>"
                    "{}</div></div>".format(''.join(vix_parts)))

                # VIX scatter chart
                if vix_reg and 'x' in vix_reg and len(vix_reg['x']) > 10:
                    fig_vix_sc = go.FigureWidget()
                    fig_vix_sc.add_trace(go.Scatter(
                        x=vix_reg['x'], y=vix_reg['y'],
                        mode='markers', name='1M obs',
                        marker=dict(color=_C['accent'], size=4, opacity=0.5)))
                    x_line = np.array([min(vix_reg['x']), max(vix_reg['x'])])
                    y_line = vix_reg['slope'] * x_line + vix_reg['intercept']
                    fig_vix_sc.add_trace(go.Scatter(
                        x=x_line, y=y_line, mode='lines',
                        line=dict(color=_C['red'], width=2), name='Reg'))
                    fig_vix_sc.update_layout(
                        title='VIX Move vs SPX Move (1M, R²={:.1%})'.format(vix_reg['r2']),
                        xaxis_title='SPX 1M (%)', yaxis_title='VIX 1M (pts)',
                        height=320, template=DASH_TEMPLATE)
                    analytics_children.append(wd.HBox([suvu_widget, vix_widget, fig_vix_sc]))
                else:
                    analytics_children.append(wd.HBox([suvu_widget, vix_widget]))

                # ── Row 4: OPEX Analysis ──
                opex = analytics['opex_stats']
                if opex:
                    opex_parts = [
                        '<h3>OPEX Price Action</h3>',
                        '<p>Total OPEX analisados: <b>{}</b></p>'.format(opex.get('total_opex', 0)),
                        '<p>Performance Flip: <b>{:.1f}%</b> ({} de {})</p>'.format(
                            opex.get('flip_pct', 0), opex.get('flip_count', 0), opex.get('total_opex', 0)),
                        '<p>Avg Ret Antes 5d: {:.2f}% | Depois 5d: {:.2f}%</p>'.format(
                            opex.get('avg_ret_before', 0), opex.get('avg_ret_after', 0)),
                        '<h4>Realized Vol Impact</h4>',
                        '<p>RV 5d Into OPEX: {:.1f}% | Out: {:.1f}%</p>'.format(
                            opex.get('rv5_delta_into', 0), opex.get('rv5_delta_out', 0)),
                        '<p>RV 10d Into: {:.1f}% | Out: {:.1f}%</p>'.format(
                            opex.get('rv10_delta_into', 0), opex.get('rv10_delta_out', 0)),
                    ]
                    opex_widget = wd.HTML(
                        "<div class='mm-dash'><div class='mm-card'>"
                        "{}</div></div>".format(''.join(opex_parts)))

                    # OPEX RV bar chart
                    _opex_fig = go.FigureWidget()
                    _opex_cats = ['RV 5d', 'RV 10d']
                    _opex_fig.add_trace(go.Bar(
                        x=_opex_cats,
                        y=[opex.get('rv5_delta_into', 0), opex.get('rv10_delta_into', 0)],
                        name='Into OPEX', marker_color=_C['accent']))
                    _opex_fig.add_trace(go.Bar(
                        x=_opex_cats,
                        y=[opex.get('rv5_delta_out', 0), opex.get('rv10_delta_out', 0)],
                        name='Out of OPEX', marker_color=_C['orange']))
                    _opex_fig.update_layout(
                        title='Realized Vol: Into vs Out of OPEX',
                        yaxis_title='RV Anualizada (%)', barmode='group',
                        height=300, template=DASH_TEMPLATE,
                        margin=dict(t=35, b=25))
                    analytics_children.append(wd.HBox([opex_widget, _opex_fig]))

                # ── Row 5: Dealer Scenario Matrix ──
                if not analytics['dealer_scenarios'].empty:
                    _ds_df = analytics['dealer_scenarios']
                    ds_html = _ds_df.to_html(
                        classes='mm-table', index=False, border=0)

                    # Dealer scenario bar chart
                    _ds_fig = go.FigureWidget()
                    _ds_fig.add_trace(go.Bar(
                        x=_ds_df['Move'], y=_ds_df['Dealer Gamma ($B)'],
                        name='Gamma', marker_color=_C['accent']))
                    _ds_fig.add_trace(go.Bar(
                        x=_ds_df['Move'], y=_ds_df['Vanna Flow ($B)'],
                        name='Vanna', marker_color=_C['purple']))
                    _ds_fig.add_trace(go.Bar(
                        x=_ds_df['Move'], y=_ds_df['Charm Flow ($B)'],
                        name='Charm', marker_color=_C['yellow']))
                    _ds_fig.add_trace(go.Scatter(
                        x=_ds_df['Move'], y=_ds_df['Total ($B)'],
                        name='Total', mode='lines+markers',
                        line=dict(color=_C['red'], width=2)))
                    _ds_fig.update_layout(
                        title='Dealer Scenario: Gamma + Vanna Flow',
                        yaxis_title='$B', barmode='group',
                        height=340, template=DASH_TEMPLATE,
                        margin=dict(t=35, b=25))

                    analytics_children.append(wd.HTML(
                        "<div class='mm-dash'><div class='mm-card'>"
                        "<h3>Dealer Scenario Matrix</h3>"
                        "{}</div></div>".format(ds_html)))
                    analytics_children.append(_ds_fig)

                # ── Row 6: Mag8 Rebalance Projection ──
                if not analytics['mag8_scenarios'].empty:
                    _m8_df = analytics['mag8_scenarios']
                    m8_html = _m8_df.to_html(
                        classes='mm-table', index=False, border=0)

                    # Mag8 bar chart (stacked by stock)
                    _m8_fig = go.FigureWidget()
                    _m8_stocks = [c for c in _m8_df.columns if c not in ('Move', 'Total')]
                    _m8_colors = ['#58a6ff', '#3fb950', '#f0883e', '#da3633',
                                  '#d29922', '#bc8cff', '#8b949e', '#e6edf3']
                    for _si, _stk in enumerate(_m8_stocks):
                        _m8_fig.add_trace(go.Bar(
                            x=_m8_df['Move'], y=_m8_df[_stk],
                            name=_stk, marker_color=_m8_colors[_si % len(_m8_colors)]))
                    _m8_fig.update_layout(
                        title='Mag8 Dealer Rebalance by Stock ($B)',
                        yaxis_title='$B', barmode='relative',
                        height=340, template=DASH_TEMPLATE,
                        margin=dict(t=35, b=25),
                        legend=dict(font=dict(size=9), orientation='h', y=1.05))

                    analytics_children.append(wd.HTML(
                        "<div class='mm-dash'><div class='mm-card'>"
                        "<h3>Mag8 Dealer Rebalance Projection ($B)</h3>"
                        "{}</div></div>".format(m8_html)))
                    analytics_children.append(_m8_fig)

                # ── Row 7: Vol Control Rebalance Projection ──
                if not analytics['vol_rebal'].empty:
                    _vr_df = analytics['vol_rebal']
                    vr_html = _vr_df.to_html(
                        classes='mm-table', index=False, border=0)

                    # Vol rebal stacked bar
                    _vr_fig = go.FigureWidget()
                    _vr_fig.add_trace(go.Bar(
                        x=_vr_df['Move'], y=_vr_df['Vol Ctrl ($B)'],
                        name='Vol Ctrl', marker_color=_C['accent']))
                    _vr_fig.add_trace(go.Bar(
                        x=_vr_df['Move'], y=_vr_df['Dealer ($B)'],
                        name='Dealer', marker_color=_C['purple']))
                    _vr_fig.add_trace(go.Bar(
                        x=_vr_df['Move'], y=_vr_df['Lev ETF ($B)'],
                        name='Lev ETF', marker_color=_C['orange']))
                    _vr_fig.add_trace(go.Scatter(
                        x=_vr_df['Move'], y=_vr_df['Total ($B)'],
                        name='Total', mode='lines+markers',
                        line=dict(color=_C['red'], width=2)))
                    _vr_fig.update_layout(
                        title='Vol Ctrl + Dealer + LevETF Rebalance',
                        yaxis_title='$B', barmode='relative',
                        height=340, template=DASH_TEMPLATE,
                        margin=dict(t=35, b=25))

                    analytics_children.append(wd.HTML(
                        "<div class='mm-dash'><div class='mm-card'>"
                        "<h3>Vol Ctrl + Dealer + LevETF Rebalance Projection</h3>"
                        "{}</div></div>".format(vr_html)))
                    analytics_children.append(_vr_fig)

                # ── Row 8: Tail Risk EVT (movido da Dispersão) ──
                if disp_ok and disp_result.get('tail_risk'):
                    analytics_children.append(
                        _tail_metrics_widget(disp_result['tail_risk']))
                    if len(disp_result.get('index_returns', [])) > 50:
                        analytics_children.append(
                            build_tail_risk_chart(
                                disp_result['index_returns'],
                                disp_result['tail_risk']))

                # ── Row 9: COR1M + DSPX + VIXEQ Bloomberg-style chart ──
                if disp_ok:
                    _cor1m_a = disp_result.get('cor1m', pd.Series(dtype=float))
                    _dspx_a = disp_result.get('dspx', pd.Series(dtype=float))
                    _vixeq_a = disp_result.get('vixeq', pd.Series(dtype=float))
                    if not _cor1m_a.empty or not _dspx_a.empty or not _vixeq_a.empty:
                        analytics_children.append(
                            build_dispersion_index_chart(_cor1m_a, _dspx_a, _vixeq_a))

                # ── Row 10: RV 21d vs Gamma Index (scatter) ──
                if not gamma_hist.empty:
                    _cur_rv = rv_30d if pd.notna(rv_30d) else None
                    # Net GEX em bilhões — clamp ao range do histórico CSV para
                    # evitar que o ponto apareça fora do scatter quando o CSV foi
                    # construído com 0DTE (escala menor que o GEX full-chain)
                    if 'total_gex_val' in dir() and total_gex_val is not None:
                        _cur_gex_bn = total_gex_val / 1e9
                        _hist_gex = gamma_hist['gamma'].dropna()
                        if not _hist_gex.empty:
                            _gex_lo = _hist_gex.quantile(0.02)
                            _gex_hi = _hist_gex.quantile(0.98)
                            _cur_gex_bn = float(np.clip(_cur_gex_bn, _gex_lo, _gex_hi))
                    else:
                        _cur_gex_bn = None
                    analytics_children.append(
                        build_rv_gamma_chart(gamma_hist,
                                             current_gamma=_cur_gex_bn,
                                             current_rv=_cur_rv))

                # ── Row 11: Gamma Index + Walls time-series ──
                if not gamma_hist.empty:
                    analytics_children.append(build_gamma_ts_chart(gamma_hist))

                # ── Row 12: SPX vs Equal-Weight SPX Rolling Correlation ──
                try:
                    _ew_corr, _ew_chart = fetch_spx_eq_weight_correlation(lookback=2520)
                    analytics_children.append(_ew_chart)
                except Exception as _ew_err:
                    print(f"⚠️ EW Correlation: {_ew_err}")

                # ── Row 13: 0DTE Volume as % of Total ──
                try:
                    _odte_ratio, _odte_chart = fetch_odte_volume_pct(lookback=2000)
                    analytics_children.append(_odte_chart)
                except Exception as _odte_err:
                    print(f"⚠️ 0DTE Volume: {_odte_err}")

                if not analytics_children:
                    analytics_children.append(wd.HTML(
                        '<p style="color:#8b949e;">Sem dados de analytics.</p>'))

                tab11 = wd.VBox(analytics_children)

            except Exception as _an_ui_err:
                _an_tb = traceback.format_exc()
                print(f"⚠️ Analytics UI: {_an_ui_err}\n{_an_tb}")
                tab11 = wd.VBox([wd.HTML(
                    "<h3>Analytics</h3>"
                    f"<p style='color:red;'>Erro: {_an_ui_err}</p>"
                    f"<pre style='font-size:10px;color:#aaa;white-space:pre-wrap;'>"
                    f"{_an_tb}</pre>")])

            # ─── ABA 12: GAMMA SQUEEZE MODEL ──────────────────────────
            try:
                _sq_pc = fp_vol_data.get('pc_ratio', 1.5) or 1.5
                _sq_gex_bn = total_gex_val / 1e9 if 'total_gex_val' in dir() else (
                    total_gex / 1e9 if 'total_gex' in dir() else 0)
                _sq_result = compute_gamma_squeeze_score(
                    net_gex_bn=_sq_gex_bn,
                    pc_ratio=_sq_pc,
                    iv_30d=iv_30d,
                    rv_30d=rv_30d,
                    gamma_flip=gamma_flip,
                    spot=spot,
                    skew=skew,
                    put_wall=put_wall,
                    call_wall=call_wall)
                tab12 = build_squeeze_tab(
                    _sq_result, _sq_gex_bn, spot, gamma_flip,
                    iv_30d, rv_30d, _sq_pc, _C)
            except Exception as _sq_err:
                _sq_tb = traceback.format_exc()
                print(f"⚠️ Gamma Squeeze tab: {_sq_err}\n{_sq_tb}")
                tab12 = wd.VBox([wd.HTML(
                    f"<h3>Gamma Squeeze</h3>"
                    f"<p style='color:red;'>Erro: {_sq_err}</p>")])

            # ── Tab 13: Ajuste Dinâmico do Book ──────────────────────────
            try:
                # dealer_aum_bn: delta_bn raw (antes do /10 de escala BBG)
                # = sum(delta × OI × 100) × spot / 1e9 — proxy do notional total do livro
                _dealer_aum = abs(_greek_cache.get('delta_bn', 0)) * 10  # reverte escala BBG
                tab13 = build_dynamic_book_tab(df, spot, rfr, ticker=ticker,
                                               dealer_aum_bn=_dealer_aum)
            except Exception as _db_err:
                print(f"⚠️ Ajuste Dinâmico tab: {_db_err}")
                tab13 = wd.VBox([wd.HTML(
                    f"<h3 style='color:#00d4e8;'>Ajuste Dinâmico do Book</h3>"
                    f"<p style='color:#f85149;'>Erro: {_db_err}</p>")])

            # ── Tab 14: Decision Engine (0DTE Intraday) ───────────────────
            try:
                _ext_scores = {
                    'flow_score':    float(fp_score.get('score', 50)) if isinstance(fp_score, dict) else 50.0,
                    'squeeze_score': float(_sq_result_v1['score']) if '_sq_result_v1' in dir() and _sq_result_v1 else 0.0,
                    'tail_score':    float(analytics.get('tail_score', 0)) if analytics else 0.0,
                    'iv_rv_spread':  float((iv_30d - rv_30d) * 100) if pd.notna(iv_30d) and pd.notna(rv_30d) else 0.0,
                    'skew_level':    float(iv_30d * 100) if pd.notna(iv_30d) else 0.0,
                }
                tab14 = _build_decision_engine_tab_inline(df, spot, rfr, ticker, _ext_scores)
            except Exception as _de_err:
                print(f"⚠️ Decision Engine tab: {_de_err}\n{traceback.format_exc()}")
                tab14 = wd.VBox([wd.HTML(
                    f"<h3 style='color:#00d4e8;'>Decision Engine</h3>"
                    f"<p style='color:#f85149;'>Erro: {_de_err}</p>"
                    f"<pre style='font-size:10px;color:#aaa;'>{traceback.format_exc()}</pre>")])

            # ═════════════════════════════════════════════════════════════
            # MONTAGEM FINAL
            # ═════════════════════════════════════════════════════════════
            dashboard = wd.Tab()
            dashboard.children = [tab1, tab2, tab3, tab4, tab5, tab6, tab7,
                                   tab8, tab9, tab10, tab11, tab12, tab13, tab14]
            tab_names = [
                'Visão Geral', 'Exposições', 'Sensibilidade', 'Análise P&L',
                'Monte Carlo', 'Rebalanceamento', 'Previsão SPX',
                'Simulador', 'Relatório', 'Flow Predictor', 'Analytics',
                'Gamma Squeeze', 'Ajuste Dinâmico', 'Decision Engine',
            ]
            for i, name in enumerate(tab_names):
                dashboard.set_title(i, name)

            # ── Snapshot: captura conteúdo de todas as abas ──
            _snapshot['ticker'] = ticker
            _snapshot['spot'] = spot
            _snapshot['ts'] = datetime.now().strftime('%Y-%m-%d %H:%M')
            _snapshot['sections'] = []
            _snapshot['metrics'] = {
                'gamma_flip':    gamma_flip,
                'gex_net_bn':    (_sq_gex_v1 * 0.1) if '_sq_gex_v1' in dir() else 0,
                'pc_ratio':      _sq_pc_v1  if '_sq_pc_v1'  in dir() else 0,
                'iv_rv_pp':      (iv_30d - rv_30d) * 100 if pd.notna(iv_30d) and pd.notna(rv_30d) else 0,
                'iv_30d':        iv_30d if pd.notna(iv_30d) else 0,
                'rv_30d':        rv_30d if pd.notna(rv_30d) else 0,
                'squeeze_score': (_sq_result_v1['score'] if '_sq_result_v1' in dir() and _sq_result_v1 else 0),
                'tail_score':    analytics.get('tail_score', 0) if analytics else 0,
                'call_wall':     call_wall,
                'put_wall':      put_wall,
                'daily_move':    daily_move if 'daily_move' in dir() else 0,
                'fragility':     fragility  if 'fragility'  in dir() else 0,
                'delta_bn':      _greek_cache.get('delta_bn', 0),
                'vanna_bn':      _greek_cache.get('vanna_bn', 0),
                'charm_bn':      _greek_cache.get('charm_bn', 0),
                # Flow score z-components (real BBG)
                'z_cta':         fp_score.get('z_cta', 0)        if isinstance(fp_score, dict) else 0,
                'z_dealer':      fp_score.get('z_dealer', 0)     if isinstance(fp_score, dict) else 0,
                'z_volctrl':     fp_score.get('z_volctrl', 0)    if isinstance(fp_score, dict) else 0,
                'z_rp':          fp_score.get('z_rp', 0)         if isinstance(fp_score, dict) else 0,
                'z_leveraged':   fp_score.get('z_leveraged', 0)  if isinstance(fp_score, dict) else 0,
                'z_passive_etf': fp_score.get('z_passive_etf', 0) if isinstance(fp_score, dict) else 0,
                'z_buyback':     fp_score.get('z_buyback', 0)    if isinstance(fp_score, dict) else 0,
                'z_cot':         fp_score.get('z_cot', 0)        if isinstance(fp_score, dict) else 0,
                'w_cta':         fp_score.get('weights', {}).get('cta', 0) if isinstance(fp_score, dict) else 0,
                'w_dealer':      fp_score.get('weights', {}).get('dealer', 0) if isinstance(fp_score, dict) else 0,
                'w_volctrl':     fp_score.get('weights', {}).get('volctrl', 0) if isinstance(fp_score, dict) else 0,
                'w_rp':          fp_score.get('weights', {}).get('rp', 0) if isinstance(fp_score, dict) else 0,
                'w_leveraged':   fp_score.get('weights', {}).get('leveraged', 0) if isinstance(fp_score, dict) else 0,
                'w_passive_etf': fp_score.get('weights', {}).get('passive_etf', 0) if isinstance(fp_score, dict) else 0,
                'w_buyback':     fp_score.get('weights', {}).get('buyback', 0) if isinstance(fp_score, dict) else 0,
                'w_cot':         fp_score.get('weights', {}).get('cot', 0) if isinstance(fp_score, dict) else 0,
                # Skew 25d (P25 put IV − C25 call IV), from BBG implied_volatility BQL
                'skew_25d':      round(skew * 100, 2) if pd.notna(skew) else 0,
                # Flow score total (0-100), from fp_score BBG-derived composite
                'flow_score_total': fp_score.get('score_total', 0) if isinstance(fp_score, dict) else 0,
                # VIX last price from BBG time series (fetched in analytics block)
                'vix':           float(vix_s.iloc[-1]) if ('vix_s' in dir() and not vix_s.empty) else 0,
                # Gamma Squeeze component breakdown (for bar chart)
                'squeeze_components': (_sq_result_v1['components'] if '_sq_result_v1' in dir() and _sq_result_v1 else {}),
            }

            # Tab 2 (Exposições) usa matplotlib — captura separadamente
            try:
                mpl_items = _capture_matplotlib_figures(
                    plot_exposure_charts, agg, df, spot, from_strike,
                    to_strike, levels, model_curves, flip_points,
                    call_wall, put_wall)
                _snapshot['sections'].append({'name': 'Exposições', 'content': mpl_items})
            except Exception:
                pass

            # Todas as outras abas (recursivo via widget tree)
            for idx, (tab_w, tname) in enumerate(zip(dashboard.children, tab_names)):
                if tname == 'Exposições':
                    continue  # já capturado acima (matplotlib)
                try:
                    items = _collect_widget_content(tab_w)
                    if items:
                        _snapshot['sections'].append({'name': tname, 'content': items})
                except Exception:
                    pass

            display(title_html, dashboard)

        except Exception as e:
            clear_output(wait=True)
            print(f"ERRO NA ANÁLISE: {e}")
            traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 9 — INTERFACE DE WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

run_btn.on_click(run_analysis)

# Botão de export / screenshot
export_btn = wd.Button(description='📸 Exportar HTML',
                       button_style='warning', icon='camera',
                       layout={'width': '180px'})
out_export = wd.Output()


def _on_export(_):
    with out_export:
        clear_output(wait=True)
        if not _snapshot['sections']:
            print("⚠️ Rode a análise primeiro antes de exportar.")
            return
        try:
            html_content = _export_dashboard_html()
            if not html_content:
                print("⚠️ Nenhum conteúdo para exportar.")
                return
            import base64
            fname = f"dashboard_{_snapshot['ticker'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
            b64 = base64.b64encode(html_content.encode('utf-8')).decode('ascii')
            size_mb = len(html_content) / (1024 * 1024)
            n_sections = len(_snapshot['sections'])
            n_items = sum(len(s['content']) for s in _snapshot['sections'])
            # Trigger browser download via JavaScript
            js_code = (
                f"var a = document.createElement('a');"
                f"a.href = 'data:text/html;base64,{b64}';"
                f"a.download = '{fname}';"
                f"document.body.appendChild(a);"
                f"a.click();"
                f"document.body.removeChild(a);")
            display(HTML(f"<script>{js_code}</script>"))
            display(wd.HTML(
                f"<div class='mm-dash'><div class='mm-card'>"
                f"<p>✅ Download iniciado: <b>{fname}</b> ({size_mb:.1f} MB)</p>"
                f"<p><small>{n_sections} abas │ {n_items} itens (gráficos + tabelas)</small></p>"
                f"</div></div>"))
        except Exception as e:
            print(f"❌ Erro ao exportar: {e}")

export_btn.on_click(_on_export)

_ctrl_box_layout = wd.Layout(
    border=f'1px solid {_C["border"]}',
    border_radius='8px',
    padding='12px',
    margin='4px 0',
)

display(wd.HTML(DASH_CSS))
display(wd.VBox([
    wd.HTML(f"<div class='mm-dash'><div class='mm-section-label'>Parâmetros da Análise</div></div>"),
    wd.VBox([
        wd.HBox([ticker_w, dte_w]),
        mny_w,
        wd.HBox([run_btn, spx_pred_w, flow_pred_w, disp_w, cta_weight_w, export_btn]),
        wd.HBox([rebal_date_w]),
    ], layout=_ctrl_box_layout),
    wd.HTML(f"<div class='mm-dash'><div class='mm-section-label'>COT Controls</div></div>"),
    wd.VBox([
        wd.HBox([cot_type_w, cot_contract_w]),
        wd.HBox([cot_trader_w]),
        wd.HBox([cot_start_w, cot_end_w]),
        wd.HBox([cot_reload_btn, etf_reload_btn]),
    ], layout=_ctrl_box_layout),
    out_cot_reload,
    out_export,
    out_main
]))
