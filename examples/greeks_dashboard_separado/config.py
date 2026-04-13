"""Config, constants, CSS and shared state for greeks_dashboard_separado."""

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


# _hud_panel e _svg_ring_html → ui.py


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
    'HWAA Index': 'ES1 Index',  # Micro SPX (1/10 do SPX)
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
MAG8 = [
    'AAPL US Equity', 'MSFT US Equity', 'GOOGL US Equity', 'AMZN US Equity',
    'NVDA US Equity', 'META US Equity', 'TSLA US Equity', 'AVGO US Equity',
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

# ── Vol-Control Fund Constants ─────────────────────────────────────────────
VOL_CTRL_AUM = {5: 100e9, 10: 150e9, 15: 100e9}  # target_vol% → AUM estimate
VOL_CTRL_MAX_LEV = 2.0  # Maximum leverage cap
VOL_CTRL_MIN_EXP = 0.20  # Piso mínimo de exposição (fundos não vão a 0%)
VOL_CTRL_DAILY_ADJ = 0.25  # Ajuste máximo por dia (~25% do delta)


def _vc_exposure(target_dec, rv):
    """Calcula exposure com piso e teto."""
    if rv < 1e-6:
        return VOL_CTRL_MAX_LEV
    return max(min(target_dec / rv, VOL_CTRL_MAX_LEV), VOL_CTRL_MIN_EXP)
