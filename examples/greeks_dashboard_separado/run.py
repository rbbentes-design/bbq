"""
Launcher para BQuant — executa todos os modulos na ordem correta.

Uso no BQuant (uma unica celula):
    %run /path/to/greeks_dashboard_separado/run.py

Ordem de execucao (dependencias de cima pra baixo):
    1. config.py        — imports, constantes, CSS, _C, bq, templates
    2. ui.py            — _hud_panel, _svg_ring_html, create_gauge, create_symmetric_gauge
    3. greeks.py        — calculate_all_greeks, black_scholes_price_vec, fmt_value
    4. data.py          — _bql_ts, fetch_market_data, fetch_options_chain, fetch_historical
    5. exposure.py      — compute_strike_exposures, compute_model_curves, compute_walls, fit_risk_model, run_monte_carlo
    6. flows.py         — ETF rebalancing, COT, buyback, flow scoring, CTA, dealer, vol control, risk parity
    7. flow_charts.py   — fp_plot_score_gauge, fp_plot_components_bar, fp_grid_*, COT plots
    8. dispersion.py    — correlacao, pairs, straddles, ML, gamma history
    9. charts.py        — skew, squeeze, vol smile, dynamic book, decision engine, macro charts
   10. dashboard.py     — build_greek_overview, plot_exposure_charts, run_analysis, widgets, display()
"""
import sys, os

_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
if _dir not in sys.path:
    sys.path.insert(0, _dir)
os.chdir(_dir)

# 1. Base
from config import *
# 2. UI helpers
from ui import *
# 3. Greeks engine
from greeks import *
# 4. BQL data pipeline
from data import *
# 5. Exposure + risk models
from exposure import *
# 6. Flow analysis
from flows import *
# 7. Flow charts
from flow_charts import *
# 8. Dispersion analysis
from dispersion import *
# 9. Charts + decision engine
from charts import *
# 10. Dashboard (widgets + run_analysis + display)
from dashboard import *
