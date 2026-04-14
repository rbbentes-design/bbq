"""
session_stats.py — Intraday Session Stats + Nomura Framework (BQuant-native)

Modulo standalone no padrao do greeks_dashboard.py:
  - BQL nativo (bq.data / bq.func / bql.Request)
  - Plotly + DASH_TEMPLATE HUD (cores _C)
  - Widgets ipywidgets + botao "Iniciar Analise" + botao "Export ZIP"
  - ZIP download via base64 + JS (sem tocar o filesystem)

Uso no BQuant:
    %run session_stats.py
    # O painel aparece direto. Configura ticker, anos, clica em Iniciar.
    # No final, 📦 Export ZIP gera metrics.json + session_stats.html + CSVs.

Para plugar como tab no greeks_dashboard.py no futuro:
    from session_stats import compute_session_stats, build_section_widgets
    section = build_section_widgets(ticker, years)

Se rodar fora do BQuant, cai em yfinance como fallback (pra testes locais).
"""
from __future__ import annotations
import base64
import io
import json
import math
import logging
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("session_stats")

# =============================================================================
# 1. BQUANT DETECTION — BQL nativo com fallback yfinance
# =============================================================================
try:
    import bql
    bq = bql.Service()
    HAS_BQL = True
    log.info("BQL detectado — modo BQuant")
except Exception:
    bq = None
    HAS_BQL = False
    try:
        import yfinance as yf
        HAS_YF = True
        log.info("yfinance detectado — modo local fallback")
    except Exception:
        HAS_YF = False

# Plotly + widgets
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as wd
from IPython.display import display, HTML, clear_output

try:
    from scipy import stats as scistats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# =============================================================================
# 2. STYLE — _C dict + DASH_TEMPLATE + DASH_CSS (padrao greeks_dashboard)
# =============================================================================
_C = {
    'bg': '#0d1117', 'card': '#161b22', 'card2': '#1c2333',
    'border': '#30363d', 'border_light': '#21262d',
    'text': '#e6edf3', 'text_muted': '#8b949e', 'text_dim': '#484f58',
    'accent': '#58a6ff', 'teal': '#00d4aa', 'orange': '#f0883e',
    'green': '#3fb950', 'red': '#f85149', 'purple': '#bc8cff',
    'yellow': '#d29922', 'pink': '#f778ba',
}

DASH_TEMPLATE = go.layout.Template()
DASH_TEMPLATE.layout = go.Layout(
    paper_bgcolor='#010810',
    plot_bgcolor='#020d1f',
    font=dict(family="'Orbitron','Courier New',monospace", size=11, color='#cce8ff'),
    title=dict(font=dict(size=13, color='#ff8c00',
                          family="'Orbitron','Courier New',monospace")),
    xaxis=dict(gridcolor='rgba(0,200,255,0.08)',
                zerolinecolor='rgba(0,200,255,0.2)',
                linecolor='rgba(0,200,255,0.15)',
                tickcolor='rgba(0,200,255,0.4)',
                tickfont=dict(color='rgba(0,200,255,0.55)', size=10)),
    yaxis=dict(gridcolor='rgba(0,200,255,0.08)',
                zerolinecolor='rgba(0,200,255,0.2)',
                linecolor='rgba(0,200,255,0.15)',
                tickcolor='rgba(0,200,255,0.4)',
                tickfont=dict(color='rgba(0,200,255,0.55)', size=10)),
    legend=dict(bgcolor='rgba(1,8,20,0.8)',
                 font=dict(size=10, color='#cce8ff'),
                 bordercolor='rgba(0,200,255,0.2)', borderwidth=1),
    colorway=['#00c8ff', '#ff8c00', '#00ff88', '#ff4757', '#b44aff',
              '#ffd32a', '#ff6b9d', '#7efff5'],
    hoverlabel=dict(bgcolor='#010810', font_size=11, font_color='#cce8ff',
                     bordercolor='rgba(0,200,255,0.4)'),
)

DASH_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&display=swap');
.mm-dash { font-family:'Orbitron','Courier New',monospace; color:#cce8ff; }
.mm-card { background:linear-gradient(145deg,rgba(1,8,20,0.97),rgba(2,13,31,0.95));
           border:1px solid rgba(0,200,255,0.15); padding:12px 14px;
           border-radius:4px; margin:8px 0; }
.mm-section-label { font-size:10px; font-weight:700; letter-spacing:2.5px;
                    text-transform:uppercase; color:rgba(0,212,255,0.7);
                    margin:14px 0 6px 0; }
.mm-loading { color:#ff8c00; padding:16px; font-size:13px; letter-spacing:1.5px; }
.mm-metric { display:inline-block; margin:0 18px 8px 0; vertical-align:top; }
.mm-metric-lbl { color:#8b949e; font-size:9px; text-transform:uppercase;
                 letter-spacing:1.5px; display:block; }
.mm-metric-val { color:#ff8c00; font-weight:700; font-size:15px; }
.mm-metric-val.up { color:#3fb950; }
.mm-metric-val.down { color:#f85149; }
.mm-table { border-collapse:collapse; width:100%; font-size:11px; }
.mm-table th { background:rgba(0,200,255,0.08); color:#cce8ff; padding:6px 10px;
               text-align:left; border-bottom:1px solid rgba(0,200,255,0.25);
               font-weight:600; text-transform:uppercase; letter-spacing:1px; }
.mm-table td { padding:5px 10px; border-bottom:1px solid rgba(0,200,255,0.08);
               color:#cce8ff; }
.mm-table tr:hover td { background:rgba(0,200,255,0.05); }
.mm-note { color:#8b949e; font-size:10px; font-style:italic; margin-top:6px; }
.mm-flag { color:#ff8c00; font-weight:600; }

.mm-divider {
  margin: 40px 0 24px 0;
  padding: 18px 24px;
  background: linear-gradient(90deg, rgba(0,200,255,0.04), rgba(0,200,255,0.08), rgba(0,200,255,0.04));
  border-top: 2px solid rgba(0,200,255,0.35);
  border-bottom: 2px solid rgba(0,200,255,0.35);
  text-align: center;
}
.mm-divider-title {
  font-family: 'Orbitron', monospace;
  font-size: 18px;
  font-weight: 900;
  color: #ff8c00;
  letter-spacing: 4px;
  text-transform: uppercase;
  text-shadow: 0 0 12px rgba(255,140,0,0.4);
}
.mm-divider-subtitle {
  color: #cce8ff;
  font-size: 11px;
  letter-spacing: 2px;
  margin-top: 6px;
  opacity: 0.75;
}
</style>"""


def _big_divider(title: str, subtitle: str = '') -> str:
    """Divisor visual grande pra separar grandes blocos do relatorio."""
    sub_html = f"<div class='mm-divider-subtitle'>{subtitle}</div>" if subtitle else ''
    return f"""
    <div class='mm-dash'>
      <div class='mm-divider'>
        <div class='mm-divider-title'>◆ ◆ ◆ &nbsp; {title} &nbsp; ◆ ◆ ◆</div>
        {sub_html}
      </div>
    </div>
    """

# Layout comum pros gráficos.
# Altura default aumentada de 380 -> 460 pra melhor legibilidade.
# Margens maiores pra nao cortar labels. Legenda com mais espaco.
_FIG_LAYOUT = dict(
    template=DASH_TEMPLATE,
    height=460,
    margin=dict(l=70, r=50, t=55, b=55),
    legend=dict(orientation='h', yanchor='bottom', y=-0.22,
                 xanchor='center', x=0.5, font=dict(size=10)),
    hoverlabel=dict(font_size=12),
)

# Layouts especificos pra charts que precisam de mais altura
_FIG_LAYOUT_TALL = {**_FIG_LAYOUT, 'height': 560}
_FIG_LAYOUT_XTALL = {**_FIG_LAYOUT, 'height': 680}


# =============================================================================
# 3. DATA LOADER (BQL nativo + yfinance fallback)
# =============================================================================

YF_TO_BBG = {
    'SPY': 'SPY US Equity', 'QQQ': 'QQQ US Equity', 'IWM': 'IWM US Equity',
    'ES=F': 'ES1 Index', 'NQ=F': 'NQ1 Index', 'RTY=F': 'RTY1 Index',
    '^VIX': 'VIX Index', '^SKEW': 'SKEW Index', '^SPX': 'SPX Index',
}


def _bql_ts(response_item, col_name='Value'):
    """
    Extrai serie temporal de um bql response item.

    Em BQL o df retornado tem o ticker como INDEX e as datas numa coluna
    'DATE' quando a query usa dates=range. Precisa re-indexar por DATE
    antes de converter pra datetime (senao pd.to_datetime quebra com
    'Unknown string format: SPX Index').
    """
    df = response_item.df()
    # Re-indexa por DATE se existir
    for date_col in ('DATE', 'date', 'Date'):
        if date_col in df.columns:
            df = df.reset_index(drop=True).set_index(date_col)
            break
    # Extrai valor
    if col_name in df.columns:
        s = df[col_name]
    elif response_item.name in df.columns:
        s = df[response_item.name]
    elif 'VALUE' in df.columns:
        s = df['VALUE']
    else:
        # ultima coluna numerica como fallback
        num = df.select_dtypes(include=[np.number])
        if len(num.columns) == 0:
            raise ValueError(f"Sem coluna numerica no response: {df.columns.tolist()}")
        s = num.iloc[:, -1]
    s.index = pd.to_datetime(s.index, errors='coerce')
    s = s[s.index.notna()]
    return s


def _bql_one_field(ticker: str, bq_field, period: str = '-5Y') -> pd.Series:
    """
    Executa uma request de 1 field (padrao exato do greeks_dashboard.fetch_historical).
    Cap automatico em -20Y (limite do Bloomberg).
    """
    # Parse period e cap em 20Y
    if period.endswith('Y') or period.endswith('y'):
        try:
            yrs = int(period.replace('-', '').replace('Y', '').replace('y', ''))
            if yrs > 20:
                period = '-20Y'
        except Exception:
            pass
    req = bql.Request(ticker, {'Value': bq_field(
        dates=bq.func.range(period, '0D'), fill='PREV')})
    s = _bql_ts(bq.execute(req)[0], 'Value')
    s = pd.to_numeric(s, errors='coerce').dropna()
    return s


def load_daily_bql(ticker: str, years: int = 5) -> pd.DataFrame:
    """
    OHLCV diario via BQL. Faz 1 request por field (padrao do greeks_dashboard).
    Se open/high/low falharem (comum em indices puros como SPX Index),
    degrada graciosamente: usa close como proxy e desabilita features que
    dependem de OHLC completo (range, gap).

    NOTA: Bloomberg trava com janelas > 20Y. Capamos automaticamente.
    """
    years = min(years, 20)  # cap BQL em 20Y (limite do Bloomberg)
    period = f'-{years}Y'
    out = {}
    # Close e volume sao obrigatorios
    out['close'] = _bql_one_field(ticker, bq.data.px_last, period)
    try:
        out['volume'] = _bql_one_field(ticker, bq.data.px_volume, period)
    except Exception as e:
        log.warning(f"[{ticker}] volume falhou: {e} — usando 0")
        out['volume'] = pd.Series(0.0, index=out['close'].index)
    # OHLC best-effort. Alguns indices/tickers nao tem.
    for name, field in [('open', bq.data.px_open),
                         ('high', bq.data.px_high),
                         ('low', bq.data.px_low)]:
        try:
            out[name] = _bql_one_field(ticker, field, period)
        except Exception as e:
            log.warning(f"[{ticker}] {name} falhou ({e}) — fallback close")
            out[name] = out['close'].copy()
    df = pd.DataFrame(out).dropna(subset=['close']).astype(float)
    df.index.name = 'date'
    return df[['open', 'high', 'low', 'close', 'volume']]


def load_daily_yf(ticker: str, years: int = 5) -> pd.DataFrame:
    """Fallback yfinance com mapeamento de ticker."""
    inv = {v: k for k, v in YF_TO_BBG.items()}
    yft = inv.get(ticker, ticker)
    df = yf.download(yft, period=f'{years}y', progress=False,
                       auto_adjust=False, threads=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[['open', 'high', 'low', 'close', 'volume']]
    df.index = pd.to_datetime(df.index)
    df.index.name = 'date'
    return df.astype(float)


def load_daily(ticker: str, years: int = 5) -> pd.DataFrame:
    """Entry point: BQL se disponivel, senao yfinance."""
    return load_daily_bql(ticker, years) if HAS_BQL else load_daily_yf(ticker, years)


def load_vol_indices(years: int = 5) -> pd.DataFrame:
    """VIX + SKEW diarios alinhados."""
    vix = load_daily('VIX Index', years)['close']
    skew = load_daily('SKEW Index', years)['close']
    return pd.DataFrame({'vix': vix, 'skew': skew}).dropna()


# =============================================================================
# 4. FEATURE ENGINEERING (RTH return, gap, range, MA, vol)
# =============================================================================

def build_session_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Monta DataFrame dia-a-dia com separacao explicita:
      rth_return = open[t] -> close[t]           (regular trading hours, 9:30-16:00 NY)
      eth_return = close[t] -> open[t+1]         (extended hours = AH + overnight + pre)
                   atribuido ao dia t (quem entra no close do dia t e sai no open do dia t+1)
      total_return = close[t-1] -> close[t]      (dia cheio)
    """
    df = daily.copy()
    # Retorno RTH (open -> close do mesmo dia)
    df['rth_return'] = (df['close'] - df['open']) / df['open']
    df['rth_return_pct'] = df['rth_return'] * 100
    df['rth_return_pts'] = df['close'] - df['open']
    # ETH: close do dia atual -> open do proximo dia. Atribuimos ao dia atual.
    next_open = df['open'].shift(-1)
    df['eth_return'] = (next_open - df['close']) / df['close']
    df['eth_return_pct'] = df['eth_return'] * 100
    # Total daily return (close-to-close) pra comparacao
    df['total_return'] = df['close'].pct_change()
    df['total_return_pct'] = df['total_return'] * 100
    # Overnight/AH (mantido como alias — ponto de vista de quem entrou ontem no close)
    prev_close = df['close'].shift(1)
    df['overnight_return'] = (df['open'] - prev_close) / prev_close
    df['ah_return'] = df['overnight_return']  # alias legado
    # Range
    df['range'] = df['high'] - df['low']
    df['range_pct'] = df['range'] / df['open']
    # Weekday
    df['weekday'] = df.index.weekday
    df['weekday_name'] = df.index.strftime('%A')
    df['day_type'] = np.where(df['rth_return'] > 0, 'up',
                      np.where(df['rth_return'] < 0, 'down', 'flat'))
    return df.dropna(subset=['rth_return'])


def add_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    prev_close = df['close'].shift(1)
    df = df.copy()
    df['gap'] = df['open'] - prev_close
    df['gap_pct'] = df['gap'] / prev_close
    df['gap_type'] = np.where(df['gap_pct'] > 0.001, 'up',
                      np.where(df['gap_pct'] < -0.001, 'down', 'flat'))
    df['gap_closed'] = np.where(
        df['gap_type'] == 'up', df['low'] <= prev_close,
        np.where(df['gap_type'] == 'down', df['high'] >= prev_close, False)
    )
    return df


MA_WINDOWS = [5, 20, 50, 200]

def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for w in MA_WINDOWS:
        df[f'ma_{w}'] = df['close'].rolling(w, min_periods=max(5, w // 4)).mean()
        df[f'open_above_ma{w}'] = df['open'] > df[f'ma_{w}'].shift(1)
        df[f'close_above_ma{w}'] = df['close'] > df[f'ma_{w}']
    return df


def compute_volatility(df: pd.DataFrame, atr_w: int = 14, rv_w: int = 21) -> pd.DataFrame:
    df = df.copy()
    prev_close = df['close'].shift(1)
    tr = pd.concat([df['high'] - df['low'],
                     (df['high'] - prev_close).abs(),
                     (df['low'] - prev_close).abs()], axis=1).max(axis=1)
    df['true_range'] = tr
    df[f'atr_{atr_w}'] = tr.rolling(atr_w, min_periods=5).mean()
    log_ret = np.log(df['close'] / prev_close)
    df['log_return'] = log_ret
    df[f'rv_{rv_w}'] = log_ret.rolling(rv_w, min_periods=5).std() * math.sqrt(252)
    df['vol_up'] = df[f'atr_{atr_w}'] > df[f'atr_{atr_w}'].shift(1)
    return df


# =============================================================================
# 5. STATS ENGINES
# =============================================================================

WEEKDAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


def _pct(x):
    """Converte fracao em pct (2 casas decimais). NaN-safe."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return np.nan
    return round(float(x) * 100, 2)


def _round2(x):
    """Arredonda pra 2 casas (NaN-safe)."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return np.nan
    return round(float(x), 2)


def _usd_bn(x):
    """Formata valor em USD bn (2 casas)."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return np.nan
    return round(float(x) / 1e9, 2)


def weekday_stats(df: pd.DataFrame, col: str = 'rth_return') -> pd.DataFrame:
    rows = []
    g = df.groupby('weekday_name')[col]
    for name in WEEKDAY_ORDER:
        if name not in g.groups:
            continue
        s = g.get_group(name)
        n = len(s); wins = s[s > 0]; losses = s[s < 0]
        hit = len(wins) / n if n else np.nan
        payoff = (wins.mean() / abs(losses.mean())) if (len(losses) and losses.mean() != 0) else np.nan
        pf = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.nan
        exp_ = (hit * (wins.mean() if len(wins) else 0)
                - (1 - hit) * (abs(losses.mean()) if len(losses) else 0))
        skew = kurt = np.nan
        if HAS_SCIPY and n >= 10:
            try:
                skew = scistats.skew(s, bias=False)
                kurt = scistats.kurtosis(s, bias=False, fisher=True)
            except Exception:
                pass
        rows.append({
            'weekday': name, 'n': n,
            'hit_rate_pct': _pct(hit),
            'loss_rate_pct': _pct(1 - hit) if not np.isnan(hit) else np.nan,
            'mean_pct': _pct(s.mean()),
            'median_pct': _pct(s.median()),
            'std_pct': _pct(s.std()),
            'best_pct': _pct(s.max()),
            'worst_pct': _pct(s.min()),
            'avg_win_pct': _pct(wins.mean()) if len(wins) else np.nan,
            'avg_loss_pct': _pct(losses.mean()) if len(losses) else np.nan,
            'payoff': round(payoff, 2) if pd.notna(payoff) else np.nan,
            'profit_factor': round(pf, 2) if pd.notna(pf) else np.nan,
            'expectancy_pct': _pct(exp_),
            'skew': round(skew, 2) if pd.notna(skew) else np.nan,
            'kurtosis': round(kurt, 2) if pd.notna(kurt) else np.nan,
            'cum_return_pct': _pct((1 + s).prod() - 1),
        })
    return pd.DataFrame(rows)


def up_down_by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in WEEKDAY_ORDER:
        s = df[df['weekday_name'] == name]['rth_return']
        if len(s) == 0:
            continue
        up = s[s > 0]; dn = s[s < 0]
        rows.append({
            'weekday': name, 'n_total': len(s),
            'n_up': len(up), 'n_down': len(dn),
            'pct_up': _pct(len(up) / len(s)),
            'pct_down': _pct(len(dn) / len(s)),
            'avg_up_pct': _pct(up.mean()) if len(up) else np.nan,
            'avg_down_pct': _pct(dn.mean()) if len(dn) else np.nan,
        })
    return pd.DataFrame(rows)


def _compute_streaks(flags: np.ndarray) -> dict:
    if len(flags) == 0:
        return {'max_positive': 0, 'max_negative': 0, 'current_length': 0}
    max_pos = max_neg = cur_pos = cur_neg = 0
    for f in flags:
        if f:
            cur_pos += 1; cur_neg = 0
            max_pos = max(max_pos, cur_pos)
        else:
            cur_neg += 1; cur_pos = 0
            max_neg = max(max_neg, cur_neg)
    cur = cur_pos if flags[-1] else cur_neg
    return {'max_positive': max_pos, 'max_negative': max_neg, 'current_length': cur}


def ma_residency_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w in MA_WINDOWS:
        sub = df.dropna(subset=[f'ma_{w}'])
        if len(sub) == 0:
            continue
        above = sub[f'close_above_ma{w}'].sum()
        streaks = _compute_streaks(sub[f'close_above_ma{w}'].astype(int).values)
        cur = int(sub[f'close_above_ma{w}'].iloc[-1])
        cur_signed = streaks['current_length'] * (1 if cur else -1)
        r = sub['rth_return']
        rows.append({
            'ma': f'MA_{w}', 'n': len(sub),
            'days_above': int(above),
            'days_below': int(len(sub) - above),
            'pct_above': _pct(above / len(sub)),
            'pct_below': _pct(1 - above / len(sub)),
            'current_streak_signed': cur_signed,
            'max_streak_above': streaks['max_positive'],
            'max_streak_below': streaks['max_negative'],
            'rth_ret_open_above_pct': _pct(r[sub[f'open_above_ma{w}']].mean()),
            'rth_ret_open_below_pct': _pct(r[~sub[f'open_above_ma{w}']].mean()),
            'rth_ret_close_above_pct': _pct(r[sub[f'close_above_ma{w}']].mean()),
            'rth_ret_close_below_pct': _pct(r[~sub[f'close_above_ma{w}']].mean()),
        })
    return pd.DataFrame(rows)


def streak_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    flags = (df['rth_return'] > 0).astype(int).values
    s = _compute_streaks(flags)
    rows.append({'serie': 'rth_updown',
                  'max_streak_up': s['max_positive'],
                  'max_streak_down': s['max_negative'],
                  'current_streak_signed': s['current_length'] * (1 if flags[-1] else -1),
                  'n': len(df)})
    if 'vol_up' in df.columns:
        vu = df['vol_up'].dropna().astype(int).values
        if len(vu):
            s2 = _compute_streaks(vu)
            rows.append({'serie': 'vol_atr_updown',
                          'max_streak_up': s2['max_positive'],
                          'max_streak_down': s2['max_negative'],
                          'current_streak_signed': s2['current_length'] * (1 if vu[-1] else -1),
                          'n': len(vu)})
    return pd.DataFrame(rows)


def conditional_after_streaks(df: pd.DataFrame, max_k: int = 5) -> pd.DataFrame:
    r = df['rth_return'].values; n = len(r)
    up_flags = (r > 0).astype(int)
    out = []
    for k in range(2, max_k + 1):
        mu = np.zeros(n, bool); md = np.zeros(n, bool)
        for t in range(k, n):
            win = up_flags[t - k:t]
            if win.sum() == k: mu[t] = True
            if win.sum() == 0: md[t] = True
        out.append({
            'k_days': k,
            'n_after_up': int(mu.sum()),
            'rth_after_up_pct': _pct(r[mu].mean()) if mu.sum() else np.nan,
            'n_after_down': int(md.sum()),
            'rth_after_down_pct': _pct(r[md].mean()) if md.sum() else np.nan,
        })
    return pd.DataFrame(out)


@dataclass
class BacktestResult:
    equity: pd.Series
    drawdown: pd.Series
    metrics: dict


def backtest_rth(df: pd.DataFrame) -> BacktestResult:
    r = df['rth_return'].fillna(0.0)
    equity = (1 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    n = len(r); years = n / 252 if n else 0
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * math.sqrt(252) if n > 1 else np.nan
    sharpe = (r.mean() * 252) / vol if vol and vol > 0 else np.nan
    dn = r[r < 0].std() * math.sqrt(252)
    sortino = (r.mean() * 252) / dn if dn and dn > 0 else np.nan
    max_dd = dd.min() if len(dd) else np.nan
    calmar = cagr / abs(max_dd) if max_dd and max_dd != 0 else np.nan
    wins = r[r > 0]; losses = r[r < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan
    rec_days = np.nan
    if len(dd) and max_dd < 0:
        trough = dd.idxmin()
        post = equity.loc[trough:]
        pk = peak.loc[trough]
        rec = post[post >= pk]
        if len(rec): rec_days = (rec.index[0] - trough).days
    metrics = {
        'n_days': n,
        'total_return_pct': _pct(equity.iloc[-1] - 1) if n else np.nan,
        'cagr_pct': _pct(cagr),
        'ann_vol_pct': _pct(vol),
        'sharpe': round(sharpe, 2) if pd.notna(sharpe) else np.nan,
        'sortino': round(sortino, 2) if pd.notna(sortino) else np.nan,
        'calmar': round(calmar, 2) if pd.notna(calmar) else np.nan,
        'max_drawdown_pct': _pct(max_dd),
        'recovery_days': rec_days,
        'profit_factor': round(pf, 2) if pd.notna(pf) else np.nan,
        'hit_rate_pct': _pct((r > 0).sum() / n) if n else np.nan,
        'best_day_pct': _pct(r.max()),
        'worst_day_pct': _pct(r.min()),
    }
    return BacktestResult(equity=equity, drawdown=dd, metrics=metrics)


def regime_stats(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=['ma_20', 'ma_50']).copy()
    cu = (d['close'] > d['ma_20']) & (d['ma_20'] > d['ma_50'])
    cd = (d['close'] < d['ma_20']) & (d['ma_20'] < d['ma_50'])
    d['regime'] = np.where(cu, 'uptrend', np.where(cd, 'downtrend', 'sideways'))
    rows = []
    for reg, s in d.groupby('regime')['rth_return']:
        rows.append({
            'regime': reg, 'n': len(s),
            'mean_pct': _pct(s.mean()),
            'median_pct': _pct(s.median()),
            'std_pct': _pct(s.std()),
            'hit_rate_pct': _pct((s > 0).sum() / len(s)),
        })
    return pd.DataFrame(rows)


def gap_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gt in ['up', 'down', 'flat']:
        s = df[df['gap_type'] == gt]['rth_return']
        if len(s) < 3: continue
        closed = df[(df['gap_type'] == gt) & df['gap_closed']]
        rows.append({
            'gap_type': gt, 'n': len(s),
            'rth_mean_pct': _pct(s.mean()),
            'rth_hit_pct': _pct((s > 0).sum() / len(s)),
            'gap_close_rate_pct': _pct(len(closed) / len(s)),
        })
    return pd.DataFrame(rows)


def monthly_stats(df: pd.DataFrame):
    d = df.copy()
    d['month'] = d.index.month
    by_month = d.groupby('month')['rth_return'].agg(['count', 'mean', 'std']).reset_index()
    by_month.columns = ['month', 'n', 'mean_pct', 'std_pct']
    by_month['mean_pct'] = (by_month['mean_pct'] * 100).round(2)
    by_month['std_pct'] = (by_month['std_pct'] * 100).round(2)
    by_month['hit_rate_pct'] = by_month['month'].map(
        lambda m: _pct((d[d['month'] == m]['rth_return'] > 0).mean()))
    return by_month


# =============================================================================
# 6. NOMURA SECTION (Options PnL + Skew Percentiles + Dynamic AUM Flows)
# =============================================================================

def approximate_25d_ivs(atm_iv: pd.Series, skew_idx: pd.Series) -> pd.DataFrame:
    skew_prem = (skew_idx - 100) / 10.0
    iv_25dP = atm_iv * (1 + 0.05 * skew_prem)
    iv_25dC = atm_iv * (1 - 0.02 * skew_prem)
    return pd.DataFrame({
        'atm_iv': atm_iv, 'iv_25dP': iv_25dP, 'iv_25dC': iv_25dC,
        'skew_25dP_25dC': iv_25dP / iv_25dC,
        'skew_25dC_atm': iv_25dC / atm_iv,
    })


def _ncdf(x): return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x * x / math.pi)))


def bs_price(S, K, T, r, sig, kind='C'):
    if T <= 0 or sig <= 0:
        return max(S - K, 0) if kind == 'C' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if kind == 'C':
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def strike_25d(S, T, sig, kind='P'):
    return S * math.exp((-1 if kind == 'P' else 1) * 0.67 * sig * math.sqrt(T))


STRATEGY_LABELS = [
    'Selling Daily ATM Straddle', 'Selling Daily ATM Call', 'Selling Daily ATM Put',
    'Selling Daily Strangle', 'Selling Daily 25d Call', 'Selling Daily 25d Put',
    'Selling Daily Straddle, Long Strangle',
    'Sell 25d Put, Buy 25d Call', 'Sell 25d Call, Buy 25d Put',
    'Stock (Long)',
]


def compute_daily_options_pnl(spot_df, iv_df, r=0.04):
    T = 1 / 252.0
    d = spot_df[['open', 'close']].join(iv_df, how='inner').dropna()
    rows = []
    for dt, row in d.iterrows():
        S0, ST = row['open'], row['close']
        iv_a = row['atm_iv'] / 100; iv_p = row['iv_25dP'] / 100; iv_c = row['iv_25dC'] / 100
        p_c = bs_price(S0, S0, T, r, iv_a, 'C'); p_p = bs_price(S0, S0, T, r, iv_a, 'P')
        K_p = strike_25d(S0, T, iv_p, 'P'); K_c = strike_25d(S0, T, iv_c, 'C')
        p_25p = bs_price(S0, K_p, T, r, iv_p, 'P'); p_25c = bs_price(S0, K_c, T, r, iv_c, 'C')
        pay_c = max(ST - S0, 0); pay_p = max(S0 - ST, 0)
        pay_25p = max(K_p - ST, 0); pay_25c = max(ST - K_c, 0)
        rows.append({
            'date': dt,
            'Selling Daily ATM Straddle': ((p_c + p_p) - (pay_c + pay_p)) / S0,
            'Selling Daily ATM Call': (p_c - pay_c) / S0,
            'Selling Daily ATM Put': (p_p - pay_p) / S0,
            'Selling Daily Strangle': ((p_25p + p_25c) - (pay_25p + pay_25c)) / S0,
            'Selling Daily 25d Call': (p_25c - pay_25c) / S0,
            'Selling Daily 25d Put': (p_25p - pay_25p) / S0,
            'Selling Daily Straddle, Long Strangle':
                (((p_c + p_p) - (pay_c + pay_p)) - ((p_25p + p_25c) - (pay_25p + pay_25c))) / S0,
            'Sell 25d Put, Buy 25d Call': ((p_25p - pay_25p) - (p_25c - pay_25c)) / S0,
            'Sell 25d Call, Buy 25d Put': ((p_25c - pay_25c) - (p_25p - pay_25p)) / S0,
            'Stock (Long)': (ST - S0) / S0,
        })
    return pd.DataFrame(rows).set_index('date')


def options_pnl_summary(pnl: pd.DataFrame) -> pd.DataFrame:
    if len(pnl) == 0: return pd.DataFrame()
    cols = [c for c in STRATEGY_LABELS if c in pnl.columns]
    today = pnl.index[-1]
    ytd = pd.Timestamp(year=today.year, month=1, day=1)
    if pnl.index.tz is not None:
        ytd = ytd.tz_localize(pnl.index.tz)
    horizons = {'1d': pnl.tail(1), '10d': pnl.tail(10), '20d': pnl.tail(20),
                '60d': pnl.tail(60), 'ytd': pnl[pnl.index >= ytd], '1y': pnl.tail(252)}
    out = pd.DataFrame(index=cols, columns=list(horizons.keys()))
    for h, sub in horizons.items():
        out[h] = [_pct(sub[c].sum()) for c in cols]
    out.index.name = 'Strategy'
    return out


def options_sharpe(pnl: pd.DataFrame) -> pd.DataFrame:
    if len(pnl) == 0: return pd.DataFrame()
    cols = [c for c in STRATEGY_LABELS if c in pnl.columns]
    today = pnl.index[-1]
    ytd = pd.Timestamp(year=today.year, month=1, day=1)
    if pnl.index.tz is not None:
        ytd = ytd.tz_localize(pnl.index.tz)
    hz = {'10d': 10, '20d': 20, '60d': 60, 'ytd': None, '1y': 252}
    out = pd.DataFrame(index=cols, columns=list(hz.keys()))
    for h, n in hz.items():
        sub = pnl[pnl.index >= ytd] if h == 'ytd' else pnl.tail(n)
        for c in cols:
            s = sub[c].dropna()
            out.at[c, h] = np.nan if len(s) < 5 or s.std() == 0 else round(s.mean() / s.std() * math.sqrt(252), 2)
    out.index.name = 'Strategy'
    return out


def skew_percentiles(iv_df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    out = iv_df.copy()
    for c in ['skew_25dP_25dC', 'skew_25dC_atm', 'atm_iv']:
        out[f'{c}_pctile'] = out[c].rolling(window, min_periods=60).apply(
            lambda x: x.rank(pct=True).iloc[-1] * 100, raw=False)
    last = out.dropna().iloc[-1] if len(out.dropna()) else None
    if last is not None:
        ltp = last['skew_25dP_25dC_pctile']; rtp = last['skew_25dC_atm_pctile']
        out.attrs['left_tail'] = ('OVERHEDGED FOR LEFT-TAIL' if ltp >= 80
                                    else 'UNDERHEDGED FOR LEFT-TAIL' if ltp <= 20
                                    else 'NEUTRAL LEFT-TAIL')
        out.attrs['right_tail'] = ('UNDERHEDGED FOR RIGHT-TAIL' if rtp <= 20
                                     else 'OVERHEDGED FOR RIGHT-TAIL' if rtp >= 80
                                     else 'NEUTRAL RIGHT-TAIL')
    return out


@dataclass
class FlowConfig:
    vc_aum_base: float = 150e9
    vc_target_vol: float = 0.10
    vc_max_lev: float = 2.0
    vc_floor: float = 0.20
    cta_aum_base: float = 85e9
    rp_aum_base: float = 200e9
    rp_eq_weight: float = 0.35


def compute_dynamic_flows(spot_df: pd.DataFrame, cfg: FlowConfig = None) -> pd.DataFrame:
    """Flows com AUM que evolui com PnL (nao e constante)."""
    cfg = cfg or FlowConfig()
    df = spot_df.copy()
    r = df['close'].pct_change().fillna(0)
    rv = (r.rolling(21, min_periods=5).std() * math.sqrt(252)).ffill().fillna(0.15)
    # garantia anti-NaN
    rv = rv.replace([np.inf, -np.inf], 0.15).fillna(0.15)

    vc_exp = (cfg.vc_target_vol / rv).clip(lower=cfg.vc_floor, upper=cfg.vc_max_lev).fillna(1.0)
    # smooth (cap 25%/dia)
    vc_arr = vc_exp.values.copy()
    for i in range(1, len(vc_arr)):
        delta = vc_arr[i] - vc_arr[i - 1]
        if abs(delta) > 0.25:
            vc_arr[i] = vc_arr[i - 1] + np.sign(delta) * 0.25
    vc_exp = pd.Series(vc_arr, index=vc_exp.index)

    cta = pd.Series(0.0, index=df.index)
    for s_w, l_w in [(5, 20), (10, 60), (20, 120), (50, 200)]:
        s = df['close'].rolling(s_w, min_periods=2).mean()
        l = df['close'].rolling(l_w, min_periods=2).mean()
        cta = cta + np.sign(s - l) / 4
    cta_exp = (cta * (0.15 / rv)).clip(-2.0, 2.0).fillna(0.0)

    inv = 1.0 / rv
    inv_ma = inv.rolling(60, min_periods=10).mean().ffill().bfill()
    rp_exp = (cfg.rp_eq_weight * (inv / inv_ma)).clip(
        0.3 * cfg.rp_eq_weight, 2.0 * cfg.rp_eq_weight).fillna(cfg.rp_eq_weight)

    out = pd.DataFrame(index=df.index)
    out['vc_exposure'] = vc_exp
    out['cta_exposure'] = cta_exp
    out['rp_exposure'] = rp_exp

    aum_vc = np.full(len(df), cfg.vc_aum_base)
    aum_cta = np.full(len(df), cfg.cta_aum_base)
    aum_rp = np.full(len(df), cfg.rp_aum_base)
    f_vc = np.zeros(len(df)); f_cta = np.zeros(len(df)); f_rp = np.zeros(len(df))
    f_vc_s = np.zeros(len(df)); f_cta_s = np.zeros(len(df)); f_rp_s = np.zeros(len(df))

    for i in range(1, len(df)):
        dv = vc_exp.iloc[i] - vc_exp.iloc[i - 1]
        dc = cta_exp.iloc[i] - cta_exp.iloc[i - 1]
        dr = rp_exp.iloc[i] - rp_exp.iloc[i - 1]
        f_vc[i] = aum_vc[i - 1] * dv; f_cta[i] = aum_cta[i - 1] * dc; f_rp[i] = aum_rp[i - 1] * dr
        f_vc_s[i] = cfg.vc_aum_base * dv; f_cta_s[i] = cfg.cta_aum_base * dc; f_rp_s[i] = cfg.rp_aum_base * dr
        aum_vc[i] = aum_vc[i - 1] * (1 + vc_exp.iloc[i] * r.iloc[i])
        aum_cta[i] = aum_cta[i - 1] * (1 + cta_exp.iloc[i] * r.iloc[i])
        aum_rp[i] = aum_rp[i - 1] * (1 + rp_exp.iloc[i] * r.iloc[i])

    out['aum_vc'] = aum_vc; out['aum_cta'] = aum_cta; out['aum_rp'] = aum_rp
    out['flow_vc'] = f_vc; out['flow_cta'] = f_cta; out['flow_rp'] = f_rp
    out['flow_total'] = f_vc + f_cta + f_rp
    out['flow_total_static'] = f_vc_s + f_cta_s + f_rp_s
    out['aum_total'] = aum_vc + aum_cta + aum_rp
    out['aum_divergence_pct'] = (out['aum_total'] / (cfg.vc_aum_base + cfg.cta_aum_base + cfg.rp_aum_base) - 1) * 100
    return out


# =============================================================================
# 7. PLOTLY CHARTS (padrao _C + DASH_TEMPLATE)
# =============================================================================

def fig_weekday_bars(wstats: pd.DataFrame, ticker: str) -> go.Figure:
    d = wstats.set_index('weekday').reindex([x for x in WEEKDAY_ORDER if x in wstats['weekday'].values])
    colors = [_C['green'] if v > 0 else _C['red'] for v in d['mean_pct']]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d.index, y=d['mean_pct'], marker_color=colors,
                          text=[f"{v:.2f}%<br>n={int(n)}" for v, n in zip(d['mean_pct'], d['n'])],
                          textposition='outside', showlegend=False))
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.6)
    fig.update_layout(title=f'{ticker} — Retorno medio RTH por weekday (%)',
                       yaxis_title='% medio', **_FIG_LAYOUT)
    return fig


def fig_weekday_hitrate(wstats: pd.DataFrame, ticker: str) -> go.Figure:
    d = wstats.set_index('weekday').reindex([x for x in WEEKDAY_ORDER if x in wstats['weekday'].values])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d.index, y=d['hit_rate_pct'], name='Hit %', marker_color=_C['green']))
    fig.add_trace(go.Bar(x=d.index, y=d['loss_rate_pct'], name='Loss %', marker_color=_C['red']))
    fig.add_hline(y=50, line_color=_C['text_muted'], line_dash='dash', line_width=0.8)
    fig.update_layout(title=f'{ticker} — Frequencia alta vs queda por weekday',
                       barmode='group', yaxis_title='%', **_FIG_LAYOUT)
    return fig


def fig_equity_dd(bt: BacktestResult, ticker: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                         vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=bt.equity.index, y=bt.equity.values,
                              name='Equity', line=dict(color=_C['accent'], width=1.4),
                              fill='tozeroy', fillcolor='rgba(88,166,255,0.05)'),
                   row=1, col=1)
    fig.add_trace(go.Scatter(x=bt.drawdown.index, y=bt.drawdown.values * 100,
                              name='DD %', line=dict(color=_C['red'], width=1),
                              fill='tozeroy', fillcolor='rgba(248,81,73,0.3)'),
                   row=2, col=1)
    m = bt.metrics
    title = (f'{ticker} — Equity RTH | Sharpe={m["sharpe"]} '
             f'CAGR={m["cagr_pct"]}% MaxDD={m["max_drawdown_pct"]}%')
    fig.update_layout(title=title, **{**_FIG_LAYOUT, 'height': 520})
    fig.update_yaxes(title_text='Equity', row=1, col=1)
    fig.update_yaxes(title_text='DD %', row=2, col=1)
    return fig


def fig_histogram(df: pd.DataFrame, ticker: str) -> go.Figure:
    r = df['rth_return'] * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r, nbinsx=60, marker_color=_C['accent'],
                                 marker_line_color=_C['border'], opacity=0.85))
    fig.add_vline(x=r.mean(), line_color=_C['orange'], line_dash='dash',
                   annotation_text=f'mean={r.mean():.2f}%')
    fig.add_vline(x=r.median(), line_color=_C['green'], line_dash='dash',
                   annotation_text=f'median={r.median():.2f}%')
    fig.update_layout(title=f'{ticker} — Distribuicao retorno RTH (%)',
                       xaxis_title='Retorno %', yaxis_title='Frequencia', **_FIG_LAYOUT)
    return fig


def fig_ma_residency(ma_df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ma_df['ma'], y=ma_df['pct_above'],
                          name='% acima', marker_color=_C['green']))
    fig.add_trace(go.Bar(x=ma_df['ma'], y=ma_df['pct_below'],
                          name='% abaixo', marker_color=_C['red']))
    fig.add_hline(y=50, line_color=_C['text_muted'], line_dash='dash', line_width=0.8)
    fig.update_layout(title=f'{ticker} — Permanencia acima/abaixo das medias',
                       barmode='group', yaxis_title='%', **_FIG_LAYOUT)
    return fig


def fig_heatmap_wkd_month(df: pd.DataFrame, ticker: str) -> go.Figure:
    d = df.copy(); d['month'] = d.index.month
    pv = d.pivot_table(index='weekday_name', columns='month',
                        values='rth_return', aggfunc='mean') * 100
    pv = pv.reindex([x for x in WEEKDAY_ORDER if x in pv.index])
    vmax = max(abs(pv.min().min()), abs(pv.max().max()))
    fig = go.Figure(go.Heatmap(
        z=pv.values, x=[f'M{c}' for c in pv.columns], y=pv.index,
        colorscale=[[0, _C['red']], [0.5, '#0d1117'], [1, _C['green']]],
        zmin=-vmax, zmax=vmax,
        text=[[f'{v:.2f}' if pd.notna(v) else '' for v in row] for row in pv.values],
        texttemplate='%{text}', textfont=dict(size=9, color='#cce8ff'),
        colorbar=dict(title='%', tickfont=dict(size=9)),
    ))
    fig.update_layout(title=f'{ticker} — Heatmap retorno medio (%) weekday x mes',
                       **_FIG_LAYOUT)
    return fig


def fig_streak_distribution(df: pd.DataFrame, ticker: str) -> go.Figure:
    flags = (df['rth_return'] > 0).astype(int).values
    runs_up = []; runs_dn = []; cur = 1
    for i in range(1, len(flags)):
        if flags[i] == flags[i - 1]: cur += 1
        else:
            (runs_up if flags[i - 1] == 1 else runs_dn).append(cur)
            cur = 1
    (runs_up if flags[-1] == 1 else runs_dn).append(cur)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=runs_up, name=f'Up (n={len(runs_up)})',
                                 marker_color=_C['green'], opacity=0.75))
    fig.add_trace(go.Histogram(x=runs_dn, name=f'Down (n={len(runs_dn)})',
                                 marker_color=_C['red'], opacity=0.75))
    fig.update_layout(title=f'{ticker} — Duracao de sequencias RTH',
                       barmode='overlay', xaxis_title='Dias consecutivos',
                       **_FIG_LAYOUT)
    return fig


def fig_nomura_strategies_equity(pnl: pd.DataFrame,
                                    tenor_label: str = '0DTE (T=1d)') -> go.Figure:
    """
    Curvas de PnL cumulativo de cada estrategia Nomura ao longo do tempo.
    10 linhas num so painel — o mesmo tipo de plot que gera os numeros
    da tabela, mas agora mostrando a evolucao temporal.
    """
    if len(pnl) == 0:
        return go.Figure().update_layout(title='Nomura Strategies — sem dados',
                                           **_FIG_LAYOUT)
    # Abrevia os nomes longos
    def _short(label):
        rep = {
            'Selling Daily ATM Straddle': 'Sell ATM Strd',
            'Selling Daily ATM Call': 'Sell ATM Call',
            'Selling Daily ATM Put': 'Sell ATM Put',
            'Selling Daily Strangle': 'Sell 25d Strg',
            'Selling Daily 25d Call': 'Sell 25dC',
            'Selling Daily 25d Put': 'Sell 25dP',
            'Selling Daily Straddle, Long Strangle': 'Sell Strd/Lg Strg',
            'Sell 25d Put, Buy 25d Call': 'Sell25dP/Buy25dC',
            'Sell 25d Call, Buy 25d Put': 'Sell25dC/Buy25dP',
            'Stock (Long)': 'Stock Long',
        }
        return rep.get(label, label)

    strat_cols = [c for c in STRATEGY_LABELS if c in pnl.columns]
    # Cumulative return em % (sum do daily PnL em % do spot)
    cum = (pnl[strat_cols].cumsum() * 100)

    # Paleta distintiva
    palette = ['#00c8ff', '#ff8c00', '#00ff88', '#ff4757', '#b44aff',
                '#ffd32a', '#ff6b9d', '#7efff5', '#3fb950', '#cce8ff']

    fig = go.Figure()
    # Ordena por retorno final (maior em cima na legenda)
    final_sorted = cum.iloc[-1].sort_values(ascending=False)
    for i, strat in enumerate(final_sorted.index):
        final_val = cum[strat].iloc[-1]
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum[strat].values,
            name=f'{_short(strat)}: {final_val:+.1f}%',
            line=dict(color=color, width=1.4),
            hovertemplate=f'<b>{_short(strat)}</b><br>%{{x|%Y-%m-%d}}: %{{y:.2f}}%<extra></extra>'))
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.6, line_dash='dot')
    fig.update_layout(
        title=f'Nomura — Cumulative PnL por Estrategia ({tenor_label})',
        yaxis_title='% cumulativo do spot', yaxis_ticksuffix='%',
        **{**_FIG_LAYOUT, 'height': 560,
           'legend': dict(orientation='v', yanchor='middle', y=0.5,
                           xanchor='left', x=1.02, font=dict(size=9.5)),
           'margin': dict(l=60, r=240, t=55, b=50)})
    return fig


def fig_nomura_strategies_rolling(pnl: pd.DataFrame, window: int = 60,
                                     tenor_label: str = '0DTE (T=1d)') -> go.Figure:
    """
    Rolling N-day return de cada estrategia — mostra quando cada uma esta
    performando e quando nao. Util pra ver regimes.
    """
    if len(pnl) == 0:
        return go.Figure().update_layout(title='Rolling — sem dados', **_FIG_LAYOUT)

    def _short(label):
        rep = {
            'Selling Daily ATM Straddle': 'Sell ATM Strd',
            'Selling Daily ATM Call': 'Sell ATM Call',
            'Selling Daily ATM Put': 'Sell ATM Put',
            'Selling Daily Strangle': 'Sell 25d Strg',
            'Selling Daily 25d Call': 'Sell 25dC',
            'Selling Daily 25d Put': 'Sell 25dP',
            'Selling Daily Straddle, Long Strangle': 'Sell Strd/Lg Strg',
            'Sell 25d Put, Buy 25d Call': 'Sell25dP/Buy25dC',
            'Sell 25d Call, Buy 25d Put': 'Sell25dC/Buy25dP',
            'Stock (Long)': 'Stock Long',
        }
        return rep.get(label, label)

    strat_cols = [c for c in STRATEGY_LABELS if c in pnl.columns]
    # Rolling sum (N-day cumulative return em pts absolutos do spot) em %
    rolling = (pnl[strat_cols].rolling(window).sum() * 100)

    palette = ['#00c8ff', '#ff8c00', '#00ff88', '#ff4757', '#b44aff',
                '#ffd32a', '#ff6b9d', '#7efff5', '#3fb950', '#cce8ff']

    fig = go.Figure()
    for i, strat in enumerate(strat_cols):
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=rolling.index, y=rolling[strat].values,
            name=_short(strat),
            line=dict(color=color, width=1.1),
            hovertemplate=f'<b>{_short(strat)}</b><br>%{{x|%Y-%m-%d}}: %{{y:.2f}}%<extra></extra>'))
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.6, line_dash='dot')
    fig.update_layout(
        title=f'Nomura — Rolling {window}-day PnL ({tenor_label})',
        yaxis_title=f'% ultimos {window} dias',
        **{**_FIG_LAYOUT, 'height': 500,
           'legend': dict(orientation='v', yanchor='middle', y=0.5,
                           xanchor='left', x=1.02, font=dict(size=9)),
           'margin': dict(l=60, r=240, t=55, b=50)})
    return fig


def fig_options_pnl_heatmap(summary: pd.DataFrame, sharpe: pd.DataFrame,
                               tenor_label: str = '0DTE (T=1d)') -> go.Figure:
    """Dois heatmaps empilhados (um em cima do outro) — evita label overflow."""
    # Abrevia labels longos pra nao invadir adjacente
    def _short(label):
        rep = {
            'Selling Daily ATM Straddle': 'Sell ATM Straddle',
            'Selling Daily ATM Call': 'Sell ATM Call',
            'Selling Daily ATM Put': 'Sell ATM Put',
            'Selling Daily Strangle': 'Sell 25d Strangle',
            'Selling Daily 25d Call': 'Sell 25d Call',
            'Selling Daily 25d Put': 'Sell 25d Put',
            'Selling Daily Straddle, Long Strangle': 'Sell Strd / Long Strg',
            'Sell 25d Put, Buy 25d Call': 'Sell 25dP / Buy 25dC',
            'Sell 25d Call, Buy 25d Put': 'Sell 25dC / Buy 25dP',
            'Stock (Long)': 'Stock (Long)',
        }
        return rep.get(label, label)
    summary = summary.rename(index=_short)
    sharpe = sharpe.rename(index=_short)
    fig = make_subplots(rows=2, cols=1, row_heights=[0.55, 0.45],
                         vertical_spacing=0.12,
                         subplot_titles=(
                             f'Options PnL Cumulative (%) — {tenor_label}',
                             f'Sharpe Ratio Annualized — {tenor_label}'))
    for idx, data in enumerate([summary, sharpe]):
        v = data.values.astype(float)
        vmax = np.nanmax(np.abs(v)) if v.size else 1
        fig.add_trace(go.Heatmap(
            z=v, x=list(data.columns), y=list(data.index),
            colorscale=[[0, _C['red']], [0.5, '#0d1117'], [1, _C['green']]],
            zmin=-vmax, zmax=vmax, showscale=False,
            text=[[f'{x:.1f}' if pd.notna(x) else '' for x in row] for row in v],
            texttemplate='%{text}', textfont=dict(size=10, color='#cce8ff'),
            hovertemplate='%{y} | %{x}<br>value: %{z:.2f}<extra></extra>',
        ), row=idx + 1, col=1)
    fig.update_layout(title=f'Nomura — SPX Options PnL Summary ({tenor_label})',
                       **{**_FIG_LAYOUT, 'height': 820,
                          'margin': dict(l=220, r=50, t=60, b=50)})
    return fig


def fig_skew_percentiles(sp: pd.DataFrame) -> go.Figure:
    d = sp.dropna()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                         subplot_titles=(
                             f'3M Skew 25dP/25dC (percentile) — {sp.attrs.get("left_tail", "")}',
                             f'3M Call Skew 25dC/ATM (percentile) — {sp.attrs.get("right_tail", "")}'))
    fig.add_trace(go.Scatter(x=d.index, y=d['skew_25dP_25dC_pctile'],
                              line=dict(color=_C['accent'], width=1),
                              name='25dP/25dC', showlegend=False), row=1, col=1)
    fig.add_hline(y=80, line_color=_C['red'], line_dash='dash', line_width=0.7, row=1, col=1)
    fig.add_hline(y=20, line_color=_C['green'], line_dash='dash', line_width=0.7, row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d['skew_25dC_atm_pctile'],
                              line=dict(color=_C['orange'], width=1),
                              name='25dC/ATM', showlegend=False), row=2, col=1)
    fig.add_hline(y=80, line_color=_C['red'], line_dash='dash', line_width=0.7, row=2, col=1)
    fig.add_hline(y=20, line_color=_C['green'], line_dash='dash', line_width=0.7, row=2, col=1)
    fig.update_yaxes(range=[0, 100], title_text='pctile', row=1, col=1)
    fig.update_yaxes(range=[0, 100], title_text='pctile', row=2, col=1)
    fig.update_layout(title='Nomura — SPX Historical Percentiles (Skew / Volatility)',
                       **{**_FIG_LAYOUT, 'height': 500})
    return fig


def fig_systematic_flows(flows: pd.DataFrame) -> go.Figure:
    # NAO fazer dropna agressivo — um NaN em qualquer coluna apagava tudo.
    # Remove so os primeiros dias onde flow_total e inicializado em 0.
    f = flows.copy()
    # Mascara dias em que AUM ainda nao comecou a evoluir
    f = f[f.index >= f.index[min(30, len(f) - 1)]]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                         row_heights=[0.42, 0.28, 0.30],
                         subplot_titles=(
                             'Vol Control + CTA + Risk Parity (USD bn, AUM dinamico)',
                             'Breakdown por estrategia',
                             'Divergencia AUM dinamico vs estatico (%)'))
    fig.add_trace(go.Scatter(x=f.index, y=f['flow_total'] / 1e9,
                              line=dict(color=_C['text'], width=1.2),
                              name='Total (dinamico)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=f.index, y=f['flow_total_static'] / 1e9,
                              line=dict(color=_C['text_muted'], width=0.8, dash='dash'),
                              name='Total (estatico)'), row=1, col=1)
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.5, row=1, col=1)

    fig.add_trace(go.Scatter(x=f.index, y=f['flow_vc'] / 1e9,
                              line=dict(color=_C['accent'], width=0.9),
                              name='Vol Control'), row=2, col=1)
    fig.add_trace(go.Scatter(x=f.index, y=f['flow_cta'] / 1e9,
                              line=dict(color=_C['orange'], width=0.9),
                              name='CTA'), row=2, col=1)
    fig.add_trace(go.Scatter(x=f.index, y=f['flow_rp'] / 1e9,
                              line=dict(color=_C['green'], width=0.9),
                              name='Risk Parity'), row=2, col=1)
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.5, row=2, col=1)

    fig.add_trace(go.Scatter(x=f.index, y=f['aum_divergence_pct'],
                              line=dict(color=_C['yellow'], width=1),
                              fill='tozeroy', fillcolor='rgba(210,153,34,0.2)',
                              name='AUM divergence', showlegend=False),
                   row=3, col=1)
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.5, row=3, col=1)

    fig.update_yaxes(title_text='USD bn', row=1, col=1)
    fig.update_yaxes(title_text='USD bn', row=2, col=1)
    fig.update_yaxes(title_text='%', row=3, col=1)
    fig.update_layout(title='Nomura — US Equities Systematic Flows',
                       **{**_FIG_LAYOUT, 'height': 760})
    return fig


# ---- graficos adicionais (tudo que era so tabela agora tem chart) ----

def fig_updown_weekday(updown: pd.DataFrame, ticker: str) -> go.Figure:
    """Barras empilhadas up/down por weekday com retorno medio como linha."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=updown['weekday'], y=updown['pct_up'],
                          name='% up', marker_color=_C['green']),
                   secondary_y=False)
    fig.add_trace(go.Bar(x=updown['weekday'], y=updown['pct_down'],
                          name='% down', marker_color=_C['red']),
                   secondary_y=False)
    fig.add_trace(go.Scatter(x=updown['weekday'], y=updown['avg_up_pct'],
                              name='avg up %', mode='lines+markers',
                              line=dict(color=_C['teal'], width=2, dash='dot')),
                   secondary_y=True)
    fig.add_trace(go.Scatter(x=updown['weekday'], y=updown['avg_down_pct'],
                              name='avg down %', mode='lines+markers',
                              line=dict(color=_C['orange'], width=2, dash='dot')),
                   secondary_y=True)
    fig.update_layout(title=f'{ticker} — Subiu / Caiu por weekday (barra=%, linha=ret medio)',
                       barmode='group', **_FIG_LAYOUT)
    fig.update_yaxes(title_text='% dos dias', secondary_y=False)
    fig.update_yaxes(title_text='retorno medio %', secondary_y=True)
    return fig


def fig_streaks(streaks: pd.DataFrame, ticker: str) -> go.Figure:
    """Estado atual + max historico de sequencias."""
    if len(streaks) == 0:
        return go.Figure().update_layout(title='Sequencias — sem dados', **_FIG_LAYOUT)
    x = streaks['serie']
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=streaks['max_streak_up'], name='max up',
                          marker_color=_C['green']))
    fig.add_trace(go.Bar(x=x, y=streaks['max_streak_down'], name='max down',
                          marker_color=_C['red']))
    fig.add_trace(go.Bar(x=x, y=streaks['current_streak_signed'],
                          name='atual (signed)', marker_color=_C['accent']))
    fig.update_layout(title=f'{ticker} — Sequencias RTH e Vol',
                       barmode='group', yaxis_title='dias', **_FIG_LAYOUT)
    return fig


def fig_conditional_after_streaks(cond: pd.DataFrame, ticker: str) -> go.Figure:
    """Retorno RTH medio apos N dias seguidos de alta/queda."""
    if len(cond) == 0:
        return go.Figure().update_layout(title='Conditional — sem dados', **_FIG_LAYOUT)
    x = cond['k_days'].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=cond['rth_after_up_pct'],
                          name='RTH apos N alta (%)', marker_color=_C['green'],
                          text=[f"n={n}" for n in cond['n_after_up']],
                          textposition='outside'))
    fig.add_trace(go.Bar(x=x, y=cond['rth_after_down_pct'],
                          name='RTH apos N queda (%)', marker_color=_C['red'],
                          text=[f"n={n}" for n in cond['n_after_down']],
                          textposition='outside'))
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.5)
    fig.update_layout(title=f'{ticker} — Retorno RTH apos N dias seguidos',
                       barmode='group', xaxis_title='N dias',
                       yaxis_title='retorno medio %', **_FIG_LAYOUT)
    return fig


def fig_regime(regime: pd.DataFrame, ticker: str) -> go.Figure:
    """Bars por regime: n, mean return, hit rate."""
    if len(regime) == 0:
        return go.Figure().update_layout(title='Regime — sem dados', **_FIG_LAYOUT)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = {'uptrend': _C['green'], 'downtrend': _C['red'], 'sideways': _C['neutral'] if 'neutral' in _C else _C['text_muted']}
    bar_colors = [colors.get(r, _C['accent']) for r in regime['regime']]
    fig.add_trace(go.Bar(x=regime['regime'], y=regime['mean_pct'],
                          name='Ret medio %', marker_color=bar_colors,
                          text=[f"n={n}" for n in regime['n']],
                          textposition='outside'),
                   secondary_y=False)
    fig.add_trace(go.Scatter(x=regime['regime'], y=regime['hit_rate_pct'],
                              name='Hit rate %', mode='lines+markers',
                              line=dict(color=_C['yellow'], width=2),
                              marker=dict(size=12)),
                   secondary_y=True)
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.5)
    fig.update_layout(title=f'{ticker} — Regime (uptrend/sideways/downtrend)',
                       **_FIG_LAYOUT)
    fig.update_yaxes(title_text='Ret medio RTH %', secondary_y=False)
    fig.update_yaxes(title_text='Hit rate %', range=[0, 100], secondary_y=True)
    return fig


def fig_gap(gaps: pd.DataFrame, ticker: str) -> go.Figure:
    """Gap analysis: retorno + hit + fechamento do gap."""
    if len(gaps) == 0:
        return go.Figure().update_layout(title='Gap — sem dados', **_FIG_LAYOUT)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = {'up': _C['green'], 'down': _C['red'], 'flat': _C['text_muted']}
    bar_colors = [colors.get(g, _C['accent']) for g in gaps['gap_type']]
    fig.add_trace(go.Bar(x=gaps['gap_type'], y=gaps['rth_mean_pct'],
                          name='RTH mean %', marker_color=bar_colors,
                          text=[f"n={n}" for n in gaps['n']],
                          textposition='outside'),
                   secondary_y=False)
    fig.add_trace(go.Scatter(x=gaps['gap_type'], y=gaps['rth_hit_pct'],
                              name='Hit %', mode='lines+markers',
                              line=dict(color=_C['yellow'], width=2)),
                   secondary_y=True)
    fig.add_trace(go.Scatter(x=gaps['gap_type'], y=gaps['gap_close_rate_pct'],
                              name='Gap close rate %', mode='lines+markers',
                              line=dict(color=_C['purple'], width=2, dash='dash')),
                   secondary_y=True)
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.5)
    fig.update_layout(title=f'{ticker} — Gap analysis', **_FIG_LAYOUT)
    fig.update_yaxes(title_text='Ret medio RTH %', secondary_y=False)
    fig.update_yaxes(title_text='%', range=[0, 100], secondary_y=True)
    return fig


def fig_monthly(by_month: pd.DataFrame, ticker: str) -> go.Figure:
    """Sazonalidade mes-a-mes."""
    if len(by_month) == 0:
        return go.Figure().update_layout(title='Monthly — sem dados', **_FIG_LAYOUT)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    xs = [month_names[m - 1] for m in by_month['month']]
    bar_colors = [_C['green'] if v > 0 else _C['red'] for v in by_month['mean_pct']]
    fig.add_trace(go.Bar(x=xs, y=by_month['mean_pct'], name='Ret medio %',
                          marker_color=bar_colors,
                          text=[f"n={n}" for n in by_month['n']],
                          textposition='outside'),
                   secondary_y=False)
    fig.add_trace(go.Scatter(x=xs, y=by_month['hit_rate_pct'],
                              name='Hit rate %', mode='lines+markers',
                              line=dict(color=_C['yellow'], width=2)),
                   secondary_y=True)
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.5)
    fig.update_layout(title=f'{ticker} — Sazonalidade por mes', **_FIG_LAYOUT)
    fig.update_yaxes(title_text='Ret medio RTH %', secondary_y=False)
    fig.update_yaxes(title_text='Hit rate %', range=[0, 100], secondary_y=True)
    return fig


def skew_percentiles_multi(iv_df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Calcula skew percentiles em TRES convencoes (na duvida, coloca todas):

      Nomura (ratio):
        left_tail_ratio = iv_25dP / iv_25dC    -> alto = put caro vs call
        right_tail_ratio = iv_25dC / atm_iv    -> alto = call caro vs atm

      Normalized diff (academico):
        put_skew_norm = (iv_25dP - atm_iv) / atm_iv   -> positivo = put caro
        call_skew_norm = (iv_25dC - atm_iv) / atm_iv  -> negativo = call desconta

      Absoluto (simples):
        skew_abs = iv_25dP - iv_25dC           -> diferenca pura em pts de vol

    Todas com percentil rolling 1y.
    """
    out = iv_df.copy()
    out['left_tail_ratio'] = out['iv_25dP'] / out['iv_25dC']
    out['right_tail_ratio'] = out['iv_25dC'] / out['atm_iv']
    out['put_skew_norm'] = (out['iv_25dP'] - out['atm_iv']) / out['atm_iv']
    out['call_skew_norm'] = (out['iv_25dC'] - out['atm_iv']) / out['atm_iv']
    out['skew_abs'] = out['iv_25dP'] - out['iv_25dC']
    for c in ['left_tail_ratio', 'right_tail_ratio', 'put_skew_norm',
              'call_skew_norm', 'skew_abs', 'atm_iv']:
        out[f'{c}_pctile'] = out[c].rolling(window, min_periods=60).apply(
            lambda x: x.rank(pct=True).iloc[-1] * 100, raw=False)
    last = out.dropna().iloc[-1] if len(out.dropna()) else None
    if last is not None:
        ltp = last['left_tail_ratio_pctile']
        rtp = last['right_tail_ratio_pctile']
        out.attrs['left_tail'] = ('OVERHEDGED FOR LEFT-TAIL' if ltp >= 80
                                    else 'UNDERHEDGED FOR LEFT-TAIL' if ltp <= 20
                                    else 'NEUTRAL LEFT-TAIL')
        out.attrs['right_tail'] = ('UNDERHEDGED FOR RIGHT-TAIL' if rtp <= 20
                                     else 'OVERHEDGED FOR RIGHT-TAIL' if rtp >= 80
                                     else 'NEUTRAL RIGHT-TAIL')
    return out


def fig_skew_multi(sp: pd.DataFrame) -> go.Figure:
    """
    4 paineis com todas as convencoes de skew.
    Deixa o usuario ver qual faz mais sentido.
    """
    d = sp.dropna(subset=['atm_iv'])
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                         subplot_titles=(
                             f'Nomura — 3M Skew 25dP/25dC (percentile) — {sp.attrs.get("left_tail", "")}',
                             f'Nomura — 3M Call Skew 25dC/ATM (percentile) — {sp.attrs.get("right_tail", "")}',
                             'SpotGamma-style — Put skew (25dP - ATM) / ATM (raw %)',
                             'SpotGamma-style — Call skew (25dC - ATM) / ATM (raw %)'))
    # 1. Nomura left tail
    fig.add_trace(go.Scatter(x=d.index, y=d['left_tail_ratio_pctile'],
                              line=dict(color=_C['accent'], width=1),
                              showlegend=False), row=1, col=1)
    fig.add_hline(y=80, line_color=_C['red'], line_dash='dash', line_width=0.7, row=1, col=1)
    fig.add_hline(y=20, line_color=_C['green'], line_dash='dash', line_width=0.7, row=1, col=1)
    # 2. Nomura right tail
    fig.add_trace(go.Scatter(x=d.index, y=d['right_tail_ratio_pctile'],
                              line=dict(color=_C['orange'], width=1),
                              showlegend=False), row=2, col=1)
    fig.add_hline(y=80, line_color=_C['red'], line_dash='dash', line_width=0.7, row=2, col=1)
    fig.add_hline(y=20, line_color=_C['green'], line_dash='dash', line_width=0.7, row=2, col=1)
    # 3. Put skew normalized (raw)
    fig.add_trace(go.Scatter(x=d.index, y=d['put_skew_norm'] * 100,
                              line=dict(color=_C['yellow'], width=1),
                              showlegend=False, fill='tozeroy',
                              fillcolor='rgba(210,153,34,0.12)'), row=3, col=1)
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.5, row=3, col=1)
    # 4. Call skew normalized (raw)
    fig.add_trace(go.Scatter(x=d.index, y=d['call_skew_norm'] * 100,
                              line=dict(color=_C['teal'], width=1),
                              showlegend=False, fill='tozeroy',
                              fillcolor='rgba(0,212,170,0.12)'), row=4, col=1)
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.5, row=4, col=1)
    fig.update_yaxes(range=[0, 100], title_text='pctile', row=1, col=1)
    fig.update_yaxes(range=[0, 100], title_text='pctile', row=2, col=1)
    fig.update_yaxes(title_text='%', row=3, col=1)
    fig.update_yaxes(title_text='%', row=4, col=1)
    fig.update_layout(title='Skew — convencoes Nomura + SpotGamma (na duvida)',
                       **{**_FIG_LAYOUT, 'height': 900})
    return fig


def backtest_eth(df: pd.DataFrame) -> BacktestResult:
    """
    Estrategia ETH: entra no close de t, sai no open de t+1.
    Reutiliza mesma mecanica do backtest_rth, so troca a serie de retorno.
    """
    r = df['eth_return'].fillna(0.0)
    equity = (1 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    n = len(r); years = n / 252 if n else 0
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * math.sqrt(252) if n > 1 else np.nan
    sharpe = (r.mean() * 252) / vol if vol and vol > 0 else np.nan
    dn = r[r < 0].std() * math.sqrt(252)
    sortino = (r.mean() * 252) / dn if dn and dn > 0 else np.nan
    max_dd = dd.min() if len(dd) else np.nan
    calmar = cagr / abs(max_dd) if max_dd and max_dd != 0 else np.nan
    wins = r[r > 0]; losses = r[r < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan
    metrics = {
        'n_days': n,
        'total_return_pct': _pct(equity.iloc[-1] - 1) if n else np.nan,
        'cagr_pct': _pct(cagr),
        'ann_vol_pct': _pct(vol),
        'sharpe': _round2(sharpe),
        'sortino': _round2(sortino),
        'calmar': _round2(calmar),
        'max_drawdown_pct': _pct(max_dd),
        'profit_factor': _round2(pf),
        'hit_rate_pct': _pct((r > 0).sum() / n) if n else np.nan,
        'best_day_pct': _pct(r.max()),
        'worst_day_pct': _pct(r.min()),
    }
    return BacktestResult(equity=equity, drawdown=dd, metrics=metrics)


def fig_rth_vs_eth_equity(bt_rth: BacktestResult, bt_eth: BacktestResult,
                            ticker: str) -> go.Figure:
    """
    3 paineis empilhados comparando RTH vs ETH vs Buy&Hold:
      1. Equity normalizada (norm=1) — quanto rendeu $1 em cada estrategia
      2. Drawdown das duas estrategias
      3. Rolling 60d return — mostra em qual regime cada uma performa
    """
    # Buy & Hold = manter 24/7 = (1 + rth) * (1 + eth) compondo
    rth_r = bt_rth.equity.pct_change().fillna(0)
    eth_r = bt_eth.equity.pct_change().fillna(0)
    bh_eq = ((1 + rth_r) * (1 + eth_r)).cumprod()

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                         row_heights=[0.50, 0.22, 0.28],
                         subplot_titles=(
                             f'{ticker} — Equity RTH vs ETH vs Buy & Hold (norm=1)',
                             'Drawdown (%)',
                             'Rolling 60d cumulative return (%)'))

    # 1. Equity
    fig.add_trace(go.Scatter(
        x=bt_rth.equity.index, y=bt_rth.equity.values,
        name=f'RTH open→close (Sharpe={bt_rth.metrics["sharpe"]}, '
             f'CAGR={bt_rth.metrics["cagr_pct"]}%)',
        line=dict(color=_C['accent'], width=1.6),
        fill='tozeroy', fillcolor='rgba(88,166,255,0.05)',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=bt_eth.equity.index, y=bt_eth.equity.values,
        name=f'ETH close→open (Sharpe={bt_eth.metrics["sharpe"]}, '
             f'CAGR={bt_eth.metrics["cagr_pct"]}%)',
        line=dict(color=_C['orange'], width=1.6),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=bh_eq.index, y=bh_eq.values,
        name='Buy & Hold 24/7 (close→close)',
        line=dict(color=_C['text_muted'], width=1.2, dash='dash'),
    ), row=1, col=1)
    fig.add_hline(y=1.0, line_color=_C['text_muted'], line_width=0.5, row=1, col=1)

    # 2. Drawdown
    fig.add_trace(go.Scatter(
        x=bt_rth.drawdown.index, y=bt_rth.drawdown.values * 100,
        name='RTH DD', line=dict(color=_C['accent'], width=1),
        fill='tozeroy', fillcolor='rgba(88,166,255,0.2)',
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=bt_eth.drawdown.index, y=bt_eth.drawdown.values * 100,
        name='ETH DD', line=dict(color=_C['orange'], width=1),
        showlegend=False,
    ), row=2, col=1)

    # 3. Rolling 60d return (%)
    win = 60
    rth_roll = ((1 + rth_r).rolling(win).apply(lambda x: x.prod(), raw=True) - 1) * 100
    eth_roll = ((1 + eth_r).rolling(win).apply(lambda x: x.prod(), raw=True) - 1) * 100
    fig.add_trace(go.Scatter(
        x=rth_roll.index, y=rth_roll.values,
        name='RTH 60d', line=dict(color=_C['accent'], width=1),
        showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=eth_roll.index, y=eth_roll.values,
        name='ETH 60d', line=dict(color=_C['orange'], width=1),
        showlegend=False,
    ), row=3, col=1)
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.5, row=3, col=1)

    fig.update_yaxes(title_text='Equity', row=1, col=1)
    fig.update_yaxes(title_text='DD %', row=2, col=1)
    fig.update_yaxes(title_text='Ret 60d %', row=3, col=1)
    fig.update_layout(
        **{**_FIG_LAYOUT, 'height': 780,
           'legend': dict(orientation='h', yanchor='bottom', y=1.02,
                           xanchor='center', x=0.5, font=dict(size=10))})
    return fig


def fig_rth_vs_eth_scatter(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Scatter RTH x ETH dia-a-dia.
      - x = retorno RTH (%)
      - y = retorno ETH (%)
      - cor = sinal combinado (RTH+ETH)
      - linhas de referencia em x=0 e y=0 dividem em 4 quadrantes
    """
    d = df[['rth_return', 'eth_return']].dropna() * 100
    if len(d) == 0:
        return go.Figure().update_layout(title='Scatter RTH x ETH — sem dados',
                                            **_FIG_LAYOUT)

    corr = d.corr().iloc[0, 1]
    # Contagem por quadrante
    q1 = ((d['rth_return'] > 0) & (d['eth_return'] > 0)).sum()  # ambos up
    q2 = ((d['rth_return'] < 0) & (d['eth_return'] > 0)).sum()  # RTH- ETH+
    q3 = ((d['rth_return'] < 0) & (d['eth_return'] < 0)).sum()  # ambos down
    q4 = ((d['rth_return'] > 0) & (d['eth_return'] < 0)).sum()  # RTH+ ETH-
    total = len(d)

    # Cores pontos por quadrante
    colors = np.where((d['rth_return'] > 0) & (d['eth_return'] > 0), _C['green'],
              np.where((d['rth_return'] < 0) & (d['eth_return'] < 0), _C['red'],
                        _C['yellow']))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d['rth_return'], y=d['eth_return'],
        mode='markers',
        marker=dict(color=colors, size=4, opacity=0.55,
                     line=dict(width=0.3, color=_C['border'])),
        name='dia', showlegend=False,
        hovertemplate='RTH=%{x:.2f}%<br>ETH=%{y:.2f}%<extra></extra>'))
    # Linhas de referencia nos eixos
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.6)
    fig.add_vline(x=0, line_color=_C['text_muted'], line_width=0.6)
    # Linha de regressao (ajuste linear simples)
    if len(d) > 5:
        b, a = np.polyfit(d['rth_return'].values, d['eth_return'].values, 1)
        xs = np.array([d['rth_return'].min(), d['rth_return'].max()])
        fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode='lines',
                                   line=dict(color=_C['accent'], width=1.3, dash='dot'),
                                   name=f'fit: y={a:+.2f}{b:+.3f}x', showlegend=True))
    # Anotacoes de quadrantes
    xmax = d['rth_return'].max(); xmin = d['rth_return'].min()
    ymax = d['eth_return'].max(); ymin = d['eth_return'].min()
    annotations = [
        dict(x=xmax * 0.75, y=ymax * 0.85, showarrow=False,
             text=f"Ambos UP<br>{q1} ({100*q1/total:.1f}%)",
             font=dict(color=_C['green'], size=10)),
        dict(x=xmin * 0.75, y=ymax * 0.85, showarrow=False,
             text=f"RTH-<br>ETH+<br>{q2} ({100*q2/total:.1f}%)",
             font=dict(color=_C['yellow'], size=10)),
        dict(x=xmin * 0.75, y=ymin * 0.85, showarrow=False,
             text=f"Ambos DOWN<br>{q3} ({100*q3/total:.1f}%)",
             font=dict(color=_C['red'], size=10)),
        dict(x=xmax * 0.75, y=ymin * 0.85, showarrow=False,
             text=f"RTH+<br>ETH-<br>{q4} ({100*q4/total:.1f}%)",
             font=dict(color=_C['yellow'], size=10)),
    ]
    fig.update_layout(
        title=f'{ticker} — RTH vs ETH daily scatter (corr={corr:.3f}, n={total})',
        xaxis_title='RTH return %  (open→close)',
        yaxis_title='ETH return %  (close→open)',
        annotations=annotations,
        **{**_FIG_LAYOUT, 'height': 620})
    return fig


def rth_eth_bottom_line(bt_rth: BacktestResult, bt_eth: BacktestResult,
                          initial: float = 10000.0) -> dict:
    """Retorna o bottom-line: quanto $initial vira em cada estrategia."""
    rth_r = bt_rth.equity.pct_change().fillna(0)
    eth_r = bt_eth.equity.pct_change().fillna(0)
    bh_eq = ((1 + rth_r) * (1 + eth_r)).cumprod()
    n_days = len(bt_rth.equity)
    years = n_days / 252 if n_days else 0
    start = bt_rth.equity.index[0].strftime('%Y-%m-%d') if n_days else 'N/A'
    end = bt_rth.equity.index[-1].strftime('%Y-%m-%d') if n_days else 'N/A'
    return {
        'initial_usd': initial,
        'period_start': start, 'period_end': end,
        'n_days': n_days, 'years': round(years, 2),
        'rth_final_usd': round(initial * float(bt_rth.equity.iloc[-1]), 2),
        'rth_return_pct': bt_rth.metrics.get('total_return_pct'),
        'rth_cagr_pct': bt_rth.metrics.get('cagr_pct'),
        'rth_sharpe': bt_rth.metrics.get('sharpe'),
        'eth_final_usd': round(initial * float(bt_eth.equity.iloc[-1]), 2),
        'eth_return_pct': bt_eth.metrics.get('total_return_pct'),
        'eth_cagr_pct': bt_eth.metrics.get('cagr_pct'),
        'eth_sharpe': bt_eth.metrics.get('sharpe'),
        'bh_final_usd': round(initial * float(bh_eq.iloc[-1]), 2),
        'bh_return_pct': _pct(bh_eq.iloc[-1] - 1),
    }


def fig_rth_eth_cumret(bt_rth: BacktestResult, bt_eth: BacktestResult,
                         ticker: str) -> go.Figure:
    """
    Retorno cumulativo em % das duas estrategias — grafico simples, 1 painel.
    Comeca em 0%, linha do tempo, valor final destacado na legenda.
    """
    rth_cum = (bt_rth.equity - 1) * 100   # % desde o inicio
    eth_cum = (bt_eth.equity - 1) * 100

    rth_final = rth_cum.iloc[-1] if len(rth_cum) else np.nan
    eth_final = eth_cum.iloc[-1] if len(eth_cum) else np.nan

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rth_cum.index, y=rth_cum.values,
        name=f'RTH (open→close) → final {rth_final:+.2f}%',
        line=dict(color=_C['accent'], width=2.2),
        fill='tozeroy', fillcolor='rgba(88,166,255,0.06)',
        hovertemplate='%{x|%Y-%m-%d}<br>%{y:+.2f}%<extra>RTH</extra>'))
    fig.add_trace(go.Scatter(
        x=eth_cum.index, y=eth_cum.values,
        name=f'ETH (close→open) → final {eth_final:+.2f}%',
        line=dict(color=_C['orange'], width=2.2),
        hovertemplate='%{x|%Y-%m-%d}<br>%{y:+.2f}%<extra>ETH</extra>'))
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.8,
                   line_dash='dot')
    fig.update_layout(
        title=f'{ticker} — Retorno cumulativo (%): RTH vs ETH',
        yaxis_title='Retorno cumulativo %',
        yaxis_ticksuffix='%',
        **{**_FIG_LAYOUT, 'height': 460})
    return fig


def fig_rth_eth_simple(bt_rth: BacktestResult, bt_eth: BacktestResult,
                        ticker: str, initial: float = 10000.0) -> go.Figure:
    """
    Grafico simples de 1 painel so: $initial virou quanto em cada estrategia.
    Eixo Y em dolares absolutos, nao normalizado. Facil de ler.
    """
    rth_r = bt_rth.equity.pct_change().fillna(0)
    eth_r = bt_eth.equity.pct_change().fillna(0)
    bh_eq = ((1 + rth_r) * (1 + eth_r)).cumprod()

    rth_usd = bt_rth.equity * initial
    eth_usd = bt_eth.equity * initial
    bh_usd = bh_eq * initial

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rth_usd.index, y=rth_usd.values,
        name=f'RTH — final ${rth_usd.iloc[-1]:,.0f}',
        line=dict(color=_C['accent'], width=2.2),
        fill='tozeroy', fillcolor='rgba(88,166,255,0.06)',
        hovertemplate='%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>RTH</extra>'))
    fig.add_trace(go.Scatter(
        x=eth_usd.index, y=eth_usd.values,
        name=f'ETH — final ${eth_usd.iloc[-1]:,.0f}',
        line=dict(color=_C['orange'], width=2.2),
        hovertemplate='%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>ETH</extra>'))
    fig.add_trace(go.Scatter(
        x=bh_usd.index, y=bh_usd.values,
        name=f'Buy & Hold 24/7 — final ${bh_usd.iloc[-1]:,.0f}',
        line=dict(color=_C['text_muted'], width=1.3, dash='dash'),
        hovertemplate='%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>B&H</extra>'))
    fig.add_hline(y=initial, line_color=_C['text_muted'], line_width=0.6,
                   line_dash='dot',
                   annotation_text=f'inicial ${initial:,.0f}',
                   annotation_font_size=9,
                   annotation_position='right')
    fig.update_layout(
        title=f'{ticker} — ${initial:,.0f} investido viraram quanto?  RTH vs ETH vs B&H',
        yaxis_title='USD',
        yaxis_tickformat='$,.0f',
        **{**_FIG_LAYOUT, 'height': 500})
    return fig


def _bottom_line_html(bl: dict, ticker: str) -> str:
    """Card grande com o bottom line — aparece no topo do relatorio."""
    def _fmt_usd(v): return f"${v:,.0f}" if pd.notna(v) else '—'
    def _fmt_pct(v, sign=True):
        if pd.isna(v):
            return '—'
        s = f"{'+' if sign and v > 0 else ''}{v:.2f}%"
        return s
    def _color(v):
        return _C['green'] if v and v > 0 else _C['red']

    rth_color = _color(bl['rth_return_pct'])
    eth_color = _color(bl['eth_return_pct'])
    bh_color = _color(bl['bh_return_pct'])

    return f"""
    <div class='mm-dash'>
      <div class='mm-card' style='padding:20px 24px'>
        <div style='font-size:11px; color:#8b949e; letter-spacing:2px;
                    text-transform:uppercase; margin-bottom:4px;'>
          Bottom line — {ticker}
        </div>
        <div style='font-size:13px; color:#cce8ff; margin-bottom:18px;'>
          Se voce investisse <b style='color:#ff8c00'>{_fmt_usd(bl['initial_usd'])}</b>
          em <b>{bl['period_start']}</b> ({bl['years']}y, {bl['n_days']} dias uteis)
          e zerasse no final de <b>{bl['period_end']}</b>:
        </div>
        <table style='width:100%; border-collapse:collapse;'>
          <tr>
            <td style='padding:12px 14px; border-left:3px solid {_C["accent"]};
                       background:rgba(88,166,255,0.04);'>
              <div style='font-size:10px; color:#8b949e; text-transform:uppercase;
                          letter-spacing:1.5px;'>RTH (open → close)</div>
              <div style='font-size:24px; color:{_C["accent"]}; font-weight:700;
                          margin:4px 0;'>{_fmt_usd(bl['rth_final_usd'])}</div>
              <div style='font-size:13px; color:{rth_color}; font-weight:600;'>
                {_fmt_pct(bl['rth_return_pct'])} total
              </div>
              <div style='font-size:10px; color:#8b949e; margin-top:6px;'>
                CAGR {_fmt_pct(bl['rth_cagr_pct'])} &nbsp;|&nbsp;
                Sharpe {bl['rth_sharpe']}
              </div>
            </td>
            <td style='padding:12px 14px; border-left:3px solid {_C["orange"]};
                       background:rgba(240,136,62,0.04);'>
              <div style='font-size:10px; color:#8b949e; text-transform:uppercase;
                          letter-spacing:1.5px;'>ETH (close → open)</div>
              <div style='font-size:24px; color:{_C["orange"]}; font-weight:700;
                          margin:4px 0;'>{_fmt_usd(bl['eth_final_usd'])}</div>
              <div style='font-size:13px; color:{eth_color}; font-weight:600;'>
                {_fmt_pct(bl['eth_return_pct'])} total
              </div>
              <div style='font-size:10px; color:#8b949e; margin-top:6px;'>
                CAGR {_fmt_pct(bl['eth_cagr_pct'])} &nbsp;|&nbsp;
                Sharpe {bl['eth_sharpe']}
              </div>
            </td>
            <td style='padding:12px 14px; border-left:3px solid {_C["text_muted"]};
                       background:rgba(139,148,158,0.04);'>
              <div style='font-size:10px; color:#8b949e; text-transform:uppercase;
                          letter-spacing:1.5px;'>Buy & Hold 24/7</div>
              <div style='font-size:24px; color:{_C["text_muted"]}; font-weight:700;
                          margin:4px 0;'>{_fmt_usd(bl['bh_final_usd'])}</div>
              <div style='font-size:13px; color:{bh_color}; font-weight:600;'>
                {_fmt_pct(bl['bh_return_pct'])} total
              </div>
              <div style='font-size:10px; color:#8b949e; margin-top:6px;'>
                = (1+RTH) × (1+ETH) compondo
              </div>
            </td>
          </tr>
        </table>
      </div>
    </div>
    """


def rth_vs_eth_summary(bt_rth: BacktestResult, bt_eth: BacktestResult,
                        df: pd.DataFrame) -> pd.DataFrame:
    """Tabela comparativa RTH x ETH lado a lado."""
    d = df[['rth_return', 'eth_return']].dropna()
    corr = d.corr().iloc[0, 1] if len(d) else np.nan
    rows = []
    keys = [('n_days', 'n'), ('total_return_pct', 'total_return_pct'),
            ('cagr_pct', 'cagr_pct'), ('ann_vol_pct', 'ann_vol_pct'),
            ('sharpe', 'sharpe'), ('sortino', 'sortino'),
            ('calmar', 'calmar'), ('max_drawdown_pct', 'max_drawdown_pct'),
            ('hit_rate_pct', 'hit_rate_pct'),
            ('best_day_pct', 'best_day_pct'),
            ('worst_day_pct', 'worst_day_pct'),
            ('profit_factor', 'profit_factor')]
    for label, key in keys:
        rows.append({
            'metric': label,
            'RTH (open→close)': bt_rth.metrics.get(key, np.nan),
            'ETH (close→open)': bt_eth.metrics.get(key, np.nan),
        })
    # Adiciona correlacao no fim
    rows.append({
        'metric': 'correlation_rth_eth',
        'RTH (open→close)': _round2(corr),
        'ETH (close→open)': _round2(corr),
    })
    return pd.DataFrame(rows)


def fig_histogram_eth(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Histograma dos retornos ETH (close -> open)."""
    r = (df['eth_return'] * 100).dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r, nbinsx=60, marker_color=_C['orange'],
                                 marker_line_color=_C['border'], opacity=0.85,
                                 showlegend=False))
    fig.add_vline(x=r.mean(), line_color=_C['yellow'], line_dash='dash',
                   annotation_text=f'mean={r.mean():.2f}%')
    fig.add_vline(x=r.median(), line_color=_C['green'], line_dash='dash',
                   annotation_text=f'median={r.median():.2f}%')
    fig.update_layout(title=f'{ticker} — Distribuicao retorno ETH (close→open, %)',
                       xaxis_title='Retorno %', yaxis_title='Frequencia', **_FIG_LAYOUT)
    return fig


def fig_histogram_rth_vs_eth(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Overlay dos histogramas RTH e ETH no mesmo painel pra comparar
    distribuicao (skew, kurtosis, fat tails, etc).
    """
    rth = (df['rth_return'] * 100).dropna()
    eth = (df['eth_return'] * 100).dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rth, nbinsx=60, name=f'RTH (μ={rth.mean():.2f}%, σ={rth.std():.2f}%)',
                                 marker_color=_C['accent'], opacity=0.55,
                                 histnorm='probability'))
    fig.add_trace(go.Histogram(x=eth, nbinsx=60, name=f'ETH (μ={eth.mean():.2f}%, σ={eth.std():.2f}%)',
                                 marker_color=_C['orange'], opacity=0.55,
                                 histnorm='probability'))
    fig.add_vline(x=0, line_color=_C['text_muted'], line_width=0.6, line_dash='dash')
    fig.update_layout(title=f'{ticker} — Distribuicao RTH vs ETH (normalizado em probabilidade)',
                       xaxis_title='Retorno %', yaxis_title='Probabilidade',
                       barmode='overlay', **_FIG_LAYOUT)
    return fig


def fig_eth_weekday_bars(wstats_eth: pd.DataFrame, ticker: str) -> go.Figure:
    d = wstats_eth.set_index('weekday').reindex([x for x in WEEKDAY_ORDER if x in wstats_eth['weekday'].values])
    colors = [_C['green'] if v > 0 else _C['red'] for v in d['mean_pct']]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d.index, y=d['mean_pct'], marker_color=colors,
                          text=[f"{v:.2f}%<br>n={int(n)}" for v, n in zip(d['mean_pct'], d['n'])],
                          textposition='outside', showlegend=False))
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.6)
    fig.update_layout(title=f'{ticker} — Retorno medio ETH (close→open) por weekday (%)',
                       yaxis_title='% medio', **_FIG_LAYOUT)
    return fig


def fig_iv_rank(vol_df: pd.DataFrame) -> go.Figure:
    """IV Rank + Skew Rank estilo SpotGamma — mas pro indice (VIX/SKEW)."""
    if len(vol_df) == 0:
        return go.Figure().update_layout(title='IV/Skew rank — sem dados', **_FIG_LAYOUT)
    d = vol_df.copy()
    d['vix_rank'] = d['vix'].rolling(252, min_periods=60).apply(
        lambda x: x.rank(pct=True).iloc[-1] * 100, raw=False)
    d['skew_rank'] = d['skew'].rolling(252, min_periods=60).apply(
        lambda x: x.rank(pct=True).iloc[-1] * 100, raw=False)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                         subplot_titles=('IV Rank — VIX vs 1y history',
                                          'Skew Rank — SKEW index vs 1y history'))
    fig.add_trace(go.Scatter(x=d.index, y=d['vix_rank'],
                              line=dict(color=_C['teal'], width=1),
                              fill='tozeroy', fillcolor='rgba(0,212,170,0.15)',
                              showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d['skew_rank'],
                              line=dict(color=_C['pink'], width=1),
                              fill='tozeroy', fillcolor='rgba(247,120,186,0.15)',
                              showlegend=False), row=2, col=1)
    for row in [1, 2]:
        fig.add_hline(y=50, line_color=_C['text_muted'], line_dash='dash',
                       line_width=0.6, row=row, col=1)
    fig.update_yaxes(range=[0, 100], title_text='rank', row=1, col=1)
    fig.update_yaxes(range=[0, 100], title_text='rank', row=2, col=1)
    fig.update_layout(title='Market-Wide IV & Skew Ranks (estilo SpotGamma)',
                       **{**_FIG_LAYOUT, 'height': 600})
    return fig


# =============================================================================
# 7b. GS-STYLE INDICATORS (inspirado nos reports do Goldman Sachs)
# =============================================================================

def consecutive_days_streak_table(daily: pd.DataFrame, target_streak: int = None) -> pd.DataFrame:
    """
    Estilo Cullen Morgan / GS Vol Color:
    Encontra no historico todos os momentos onde houve N dias consecutivos de alta
    no fechamento, e calcula forward returns em varios horizontes.

    Retorna tabela com uma linha por evento historico + linha agregada
    (Average, Median, Max, Min, % Positive).
    """
    close = daily['close']
    ret = close.pct_change()
    # Flag dia positivo
    up = (ret > 0).astype(int)
    # Conta streak atual em cada ponto do tempo
    streak = up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)
    # streak so conta o ultimo grupo positivo; se 0 (dia down), streak = 0
    streak = streak.where(up == 1, 0)

    # Target streak: se nao passado, usa o atual
    if target_streak is None:
        target_streak = int(streak.iloc[-1]) if streak.iloc[-1] > 0 else 6

    # Encontra todas as datas com exatamente target_streak dias seguidos de alta
    event_dates = streak[streak == target_streak].index

    # Forward horizons em dias uteis
    horizons = [('+1d', 1), ('+2d', 2), ('+3d', 3), ('+4d', 4),
                 ('+1w', 5), ('+2w', 10), ('+1m', 21), ('+3m', 63), ('+6m', 126)]

    rows = []
    for dt in event_dates:
        row = {'Date': dt.strftime('%Y-%m-%d') if hasattr(dt, 'strftime') else str(dt)}
        try:
            idx = close.index.get_loc(dt)
        except Exception:
            continue
        for label, n in horizons:
            future_idx = idx + n
            if future_idx < len(close):
                fwd_ret = (close.iloc[future_idx] / close.iloc[idx] - 1) * 100
                row[f'{label} Return'] = round(fwd_ret, 2)
            else:
                row[f'{label} Return'] = np.nan
        rows.append(row)

    if len(rows) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Linhas de agregacao
    agg_rows = []
    for stat_name, func in [('Average', np.nanmean), ('Median', np.nanmedian),
                              ('Max', np.nanmax), ('Min', np.nanmin)]:
        agg = {'Date': stat_name}
        for label, _ in horizons:
            col = f'{label} Return'
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce').dropna()
                agg[col] = round(func(vals), 2) if len(vals) else np.nan
        agg_rows.append(agg)
    # % Positive
    pct_row = {'Date': '% Positive'}
    for label, _ in horizons:
        col = f'{label} Return'
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            pct_row[col] = round((vals > 0).sum() / len(vals) * 100, 1) if len(vals) else np.nan
    agg_rows.append(pct_row)

    return pd.concat([df, pd.DataFrame([{'Date': '---'}]), pd.DataFrame(agg_rows)],
                       ignore_index=True)


def fig_consecutive_days_bar(streak_table: pd.DataFrame, ticker: str,
                                target_streak: int) -> go.Figure:
    """Grafico de barras com a estatistica agregada (Average/% Positive) por horizonte."""
    if len(streak_table) == 0:
        return go.Figure().update_layout(title='Consecutive days — sem dados', **_FIG_LAYOUT)

    horizons = [c for c in streak_table.columns if 'Return' in c]
    avg = streak_table[streak_table['Date'] == 'Average'].iloc[0]
    pct = streak_table[streak_table['Date'] == '% Positive'].iloc[0]
    n_events = len(streak_table) - 6  # subtracts header + 5 agg rows

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = [_C['green'] if avg[h] > 0 else _C['red'] for h in horizons]
    fig.add_trace(go.Bar(
        x=[h.replace(' Return', '') for h in horizons],
        y=[avg[h] for h in horizons],
        name='Average Return %', marker_color=colors,
        text=[f"{avg[h]:.2f}%" for h in horizons],
        textposition='outside'), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=[h.replace(' Return', '') for h in horizons],
        y=[pct[h] for h in horizons],
        name='% Positive', mode='lines+markers',
        line=dict(color=_C['yellow'], width=2),
        marker=dict(size=12)), secondary_y=True)
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.5)
    fig.update_layout(
        title=f'{ticker} — Forward returns apos {target_streak} dias seguidos de alta '
              f'(n={n_events} eventos historicos)',
        **_FIG_LAYOUT)
    fig.update_yaxes(title_text='Ret medio %', secondary_y=False)
    fig.update_yaxes(title_text='% Positive', range=[0, 100], secondary_y=True)
    return fig


def fig_seasonality_by_year(daily: pd.DataFrame, ticker: str,
                              max_years: int = 10) -> go.Figure:
    """
    Estilo Goldman (Soria): cada ano como linha separada, performance YTD cumulativa.
    Mostra como o ano atual se compara com os anteriores.
    """
    df = daily.copy()
    df['year'] = df.index.year
    df['doy'] = df.index.dayofyear

    years = sorted(df['year'].unique())[-max_years:]
    current_year = years[-1] if years else None

    fig = go.Figure()
    # Paleta — ano atual destaque em vermelho, outros com transparencia
    palette = ['#f85149', '#ff8c00', '#d29922', '#3fb950', '#00d4aa',
                '#58a6ff', '#bc8cff', '#f778ba', '#8b949e', '#6e7681']

    for i, y in enumerate(years):
        sub = df[df['year'] == y].copy()
        if len(sub) < 5:
            continue
        # Cumulative return normalized to 100 at start of year
        sub['cum'] = ((1 + sub['close'].pct_change().fillna(0)).cumprod() * 100)
        color = palette[i % len(palette)]
        is_current = (y == current_year)
        fig.add_trace(go.Scatter(
            x=sub['doy'], y=sub['cum'],
            name=str(y) + (' (atual)' if is_current else ''),
            line=dict(color=color,
                       width=2.4 if is_current else 1.0),
            opacity=1.0 if is_current else 0.55,
        ))
    # Avg linha
    pivot = df.pivot_table(index='doy', columns='year', values='close',
                             aggfunc='first')
    pivot = pivot.apply(lambda s: (1 + s.pct_change().fillna(0)).cumprod() * 100)
    avg = pivot.mean(axis=1)
    fig.add_trace(go.Scatter(
        x=avg.index, y=avg.values, name='Avg',
        line=dict(color='#ffffff', width=1.3, dash='dash')))

    fig.add_hline(y=100, line_color=_C['text_muted'], line_width=0.5, line_dash='dot')
    # X tick marks nos meses
    tick_days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    tick_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig.update_xaxes(tickmode='array', tickvals=tick_days, ticktext=tick_labels)
    fig.update_layout(
        title=f'{ticker} — Seasonality por ano (cada linha = 1 ano, ano atual destacado)',
        yaxis_title='Performance YTD (norm=100)',
        xaxis_title='Dia do ano',
        **{**_FIG_LAYOUT, 'height': 540})
    return fig


def fig_rolling_sharpe(daily: pd.DataFrame, ticker: str) -> go.Figure:
    """
    3m / 6m / 12m rolling Sharpe (estilo GS factor report).
    """
    r = daily['close'].pct_change().dropna()
    windows = [('3m', 63), ('6m', 126), ('12m', 252)]
    fig = go.Figure()
    colors = [_C['accent'], _C['orange'], _C['green']]
    for (label, w), c in zip(windows, colors):
        mean = r.rolling(w, min_periods=max(20, w // 3)).mean() * 252
        std = r.rolling(w, min_periods=max(20, w // 3)).std() * math.sqrt(252)
        sharpe = mean / std
        fig.add_trace(go.Scatter(
            x=sharpe.index, y=sharpe.values,
            name=f'{label} Sharpe', line=dict(color=c, width=1.2)))
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.5)
    fig.add_hline(y=1, line_color=_C['green'], line_dash='dot', line_width=0.6,
                   opacity=0.5)
    fig.update_layout(
        title=f'{ticker} — Rolling Sharpe 3m / 6m / 12m',
        yaxis_title='Sharpe (anualizado)',
        **_FIG_LAYOUT)
    return fig


def fig_rsi(daily: pd.DataFrame, ticker: str, period: int = 14) -> go.Figure:
    """RSI com zonas overbought (70) e oversold (30)."""
    close = daily['close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi.values,
                              name=f'RSI({period})',
                              line=dict(color=_C['accent'], width=1.1)))
    fig.add_hline(y=70, line_color=_C['red'], line_dash='dash', line_width=0.8,
                   annotation_text='Overbought')
    fig.add_hline(y=30, line_color=_C['green'], line_dash='dash', line_width=0.8,
                   annotation_text='Oversold')
    fig.add_hline(y=50, line_color=_C['text_muted'], line_width=0.5, line_dash='dot')
    fig.update_layout(
        title=f'{ticker} — RSI({period}) com zonas overbought/oversold',
        yaxis_title='RSI', yaxis_range=[0, 100],
        **_FIG_LAYOUT)
    return fig


def fig_rolling_return_pctile(daily: pd.DataFrame, ticker: str,
                                 window: int = 5) -> go.Figure:
    """
    Estilo Marquee PlotTool: retorno N-day com banda de 10/90 percentiles.
    Mostra se o momento atual e extremo vs historico.
    """
    close = daily['close']
    roll = close.pct_change(window) * 100
    p10 = roll.rolling(252, min_periods=60).quantile(0.10)
    p90 = roll.rolling(252, min_periods=60).quantile(0.90)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p90.index, y=p90.values,
                              line=dict(color=_C['red'], width=0.8, dash='dash'),
                              name='90th %ile'))
    fig.add_trace(go.Scatter(x=p10.index, y=p10.values,
                              line=dict(color=_C['green'], width=0.8, dash='dash'),
                              name='10th %ile',
                              fill='tonexty', fillcolor='rgba(139,148,158,0.08)'))
    fig.add_trace(go.Scatter(x=roll.index, y=roll.values,
                              line=dict(color=_C['accent'], width=1.3),
                              name=f'{window}d Return %'))
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.5)
    fig.update_layout(
        title=f'{ticker} — {window}d % change vs historical band (10/90 %ile rolling 1y)',
        yaxis_title=f'{window}d return %',
        **_FIG_LAYOUT)
    return fig


def fig_vol_panic_proxy(daily: pd.DataFrame, vol_indices: pd.DataFrame,
                          ticker: str) -> go.Figure:
    """
    Proxy do GS Vol Panic Index (0-10):
      componente 1: VIX percentil 252d     (0 baixo, 10 alto)
      componente 2: realized vol percentil  (0 baixo, 10 alto)
      componente 3: drawdown 20d            (0 sem, 10 profundo)
      componente 4: z-score retorno 5d      (0 calmo, 10 extremo)
    Indice = media simples, re-escalado 0-10.
    """
    close = daily['close']
    r = close.pct_change().fillna(0)

    # VIX percentil
    vix_pct = pd.Series(index=daily.index, dtype=float)
    if vol_indices is not None and 'vix' in vol_indices.columns:
        vix = vol_indices['vix'].reindex(daily.index).ffill()
        vix_pct = vix.rolling(252, min_periods=60).apply(
            lambda x: x.rank(pct=True).iloc[-1] * 10, raw=False)

    # RV percentil
    rv = r.rolling(21, min_periods=10).std() * math.sqrt(252)
    rv_pct = rv.rolling(252, min_periods=60).apply(
        lambda x: x.rank(pct=True).iloc[-1] * 10, raw=False)

    # Drawdown 20d
    peak = close.rolling(20, min_periods=5).max()
    dd = (close / peak - 1)
    dd_score = (-dd * 100).clip(0, 10)  # 10% DD → score 10

    # Z-score 5d return
    ret5 = close.pct_change(5) * 100
    z5 = (ret5 - ret5.rolling(60, min_periods=20).mean()) / \
         ret5.rolling(60, min_periods=20).std()
    z_score = (z5.abs().clip(0, 3) / 3) * 10  # |z|=3 → 10

    panic = pd.concat([vix_pct, rv_pct, dd_score, z_score], axis=1).mean(axis=1)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         row_heights=[0.35, 0.65],
                         subplot_titles=(
                             f'{ticker} Close',
                             'Vol Panic Proxy (0=calmo, 10=panico) | Avg(VIX%, RV%, DD20d, |z5d|)'))
    fig.add_trace(go.Scatter(x=close.index, y=close.values,
                              line=dict(color=_C['accent'], width=1.2),
                              name='Close', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=panic.index, y=panic.values,
                              line=dict(color=_C['red'], width=1.2),
                              fill='tozeroy', fillcolor='rgba(248,81,73,0.15)',
                              name='Panic score', showlegend=False), row=2, col=1)
    fig.add_hline(y=8, line_color=_C['red'], line_dash='dash', line_width=0.8,
                   row=2, col=1)
    fig.add_hline(y=5, line_color=_C['yellow'], line_dash='dot', line_width=0.6,
                   row=2, col=1)
    fig.update_yaxes(title_text='USD', row=1, col=1)
    fig.update_yaxes(title_text='Panic (0-10)', range=[0, 10], row=2, col=1)
    fig.update_layout(**{**_FIG_LAYOUT, 'height': 560})
    return fig


# =============================================================================
# 7c. GS FACTOR MONITOR (Barra pair indices + Thematic baskets)
# =============================================================================
#
# Universo inspirado no GS US Factor Monitor (Soria et al).
# Bloomberg libera o HISTORICO (px_last) dos Barra pair indices e das
# thematic baskets da GS quando colocamos "Index" no final do ticker.
# A carteira (constituents) nao e liberada, mas o level da serie esta.
# =============================================================================

GS_FACTOR_UNIVERSE = {
    'Barra Factors': [
        ('GSP1MOMO', 'Momentum'),
        ('GSP1BETA', 'Beta'),
        ('GSP1LEVG', 'Leverage'),
        ('GSP1PROF', 'Profitability'),
        ('GSP1ERNY', 'Earnings Yield'),
        ('GSP1RSVL', 'Res Vol'),
        ('GSP1GRWT', 'Growth'),
        ('GSPUGRVA', 'Growth vs Value'),
        ('GSP1VALU', 'Value'),
        ('GSP1SIZE', 'Size'),
        ('GSP1QUAL', 'Quality'),
    ],
    'Momentum': [
        ('GSPRHIMO', 'High Beta Momo'),
        ('GSPRHMO6', '6m Momo'),
        ('GSPRHMO3', '3m Momo'),
        ('GSTMTMOM', 'TMT Momo'),
        ('GSPUMOXX', 'Cross-Sector ex AI'),
        ('GSFIMOMO', 'Fins Momo'),
        ('GSENEMOM', 'Energy Momo'),
        ('GSINMOMO', 'Indus Momo'),
        ('GSHCMOMO', 'Health Momo'),
        ('GSCNMOMO', 'Consumer Momo'),
    ],
    'BBG Themes': [
        # AI / Tech
        ('BAIAT',   'AI'),
        ('BCLAT',   'Cloud'),
        ('BCYAT',   'Cybersecurity'),
        ('BFTAT',   'Frontier Tech'),
        ('BGDMTST', 'DigiTech'),
        ('BSEMISCT', 'Semiconductors'),
        ('BNEXTT',  'Next Generation'),
        ('BROAT',   'Robotics'),
        ('BAVAT',   'Autonomous Veh'),
        ('BM7T',    'Magnificent 7'),
        ('BFAANGT', 'FAANG 2.0'),
        # Fintech
        ('BPAYT',   'Digital Payments'),
        ('BEFAT',   'Enterp Fintech'),
        ('BFFAT',   'Future Fintech'),
        ('BDEAT',   'Decentral Eng'),
        # Energy / Materials / Defense
        ('BNUAT',   'Nuclear'),
        ('BHYAT',   'Hydrogen'),
        ('BSOAT',   'Solar'),
        ('BBFAT',   'Biofuels'),
        ('BGSCET',  'Clean Energy'),
        ('BCMAT',   'Circular Materials'),
        ('BFMAT',   'Future Materials'),
        ('BTMAT',   'Transition Metals'),
        ('BCCAET',  'CCUS'),
        ('BDST',    'Defense & Sec'),
        ('BMDAT',   'Modern Defense'),
        ('BWAAT',   'Water'),
        # Health / Bio
        ('BOBAT',   'Obesity'),
        ('BIBFT',   'Inno Biopharma'),
        ('BGEAT',   'Genomics'),
        ('BNMAT',   'Neuro Mental Health'),
        ('BPRAT',   'Prepare & Repair'),
        ('BRNDT',   'R&D Leaders'),
        ('BPLANTT', 'Plant-Based Food'),
        # EV / Transportation
        ('BBEVT',   'Elec Vehicles'),
        ('BFVAT',   'Future Vehicles'),
        # Infra / Digital
        ('BFDAT',   'Digital Infra'),
        ('BGTAT',   'Grid Tech'),
        ('BDMLINFT', 'Listed Infra'),
        # Ideias / Style
        ('B5GAT',   '5G'),
        ('BMETAT',  'Metaverse'),
        ('BSPAT',   'Sports'),
        ('BMVPT',   'MVP'),
        ('BBIDAT',  'Biodiversity'),
        ('BBIST',   'Billionaires'),
        # Factor-ish
        ('BSHARPT', 'Shareholder Yield'),
        ('BPPUST',  'Pricing Power'),
        ('B1GDT',   '1000 Div Growth'),
        ('BINFLST', 'Infl Sens'),
        ('BCORET',  'NC Core'),
        ('BANRT',   'ANR Improvers'),
        ('BMULTIT', 'MULTI'),
        ('DMCPTR',  'DM Comm Prod'),
        # Emerging / LatAm
        ('BFREET',  'EM Freedom'),
        ('LATAMMET', 'LATAM M&E Cap'),
    ],
    'GS Themes': [
        ('GSXUMEME', 'Memes'),
        ('GSXUNPTC', 'Non-Profit Tech'),
        ('GSXUQNTM', 'Quantum Comp'),
        ('GSXURFAV', 'Retail Favorites'),
        ('GSXUROBO', 'Robotics/Auto'),
        ('GSTHREPO', 'Buyback'),
        ('GSXUDFNS', 'Defense'),
        ('GSXUMOXL', 'High Mo ex AI/BTC'),
        ('GSXURANI', 'Uranium'),
        ('GSXUMFML', 'Marquee Momo'),
        ('GSXUHOME', 'Housing'),
        ('GSXUCYCL', 'Cyclicals'),
        ('GSXUHICN', 'Hi Income Cons'),
        ('GSXUCADR', 'China ADRs'),
        ('GSXUCOND', 'Consumer Disc'),
        ('GSXULOCN', 'Lo Income Cons'),
        ('GSXUCOMP', 'Defensive Comp'),
        ('GSXURNEW', 'Renewables'),
        ('GSXUMIDC', 'Mid Income Cons'),
        ('GSXUEDEF', 'Expensive Def'),
        ('GSXURETL', 'Retail'),
        ('GSPUCRWD', 'Crowding'),
        ('GSPUWSHI', 'Short Interest'),
        ('GSPUOILY', 'Oil Sensitivity'),
    ],
}


def fetch_gs_factors(years: int = 2) -> pd.DataFrame:
    """
    Busca close history de todos os tickers do universo GS Factor em batch.
    Retorna DataFrame wide: index=date, cols=ticker.

    Requer BQL (BQuant). Fora do BQuant retorna DataFrame vazio.
    """
    if not HAS_BQL:
        log.warning('[gs_factors] sem BQL — pulando')
        return pd.DataFrame()

    all_items = []
    for cat, items in GS_FACTOR_UNIVERSE.items():
        for tk, name in items:
            all_items.append((tk, name, cat))

    log.info(f'[gs_factors] buscando {len(all_items)} factors via BQL...')
    frames = {}
    for tk, name, cat in all_items:
        try:
            bbg = f'{tk} Index'
            s = _bql_one_field(bbg, bq.data.px_last, period=f'-{years}Y')
            if len(s) > 50:
                frames[tk] = s
        except Exception as e:
            log.warning(f'[gs_factors] {tk} falhou: {e}')
    if not frames:
        return pd.DataFrame()
    out = pd.DataFrame(frames).sort_index()
    return out


def factor_monitor_table(history: pd.DataFrame) -> pd.DataFrame:
    """
    Tabela estilo GS Factor Monitor:
      Ticker | Name | Category | 1D % | 5D % | 1M % | YTD % | 1Y Return | Std Dev 252d

    Positivos em verde implicito (cor fica na viz, nao na tabela).
    """
    if len(history) == 0:
        return pd.DataFrame()

    # Mapa ticker -> (name, category)
    meta = {}
    for cat, items in GS_FACTOR_UNIVERSE.items():
        for tk, name in items:
            meta[tk] = (name, cat)

    today = history.index[-1]
    ytd_start = pd.Timestamp(year=today.year, month=1, day=1)
    if history.index.tz is not None:
        ytd_start = ytd_start.tz_localize(history.index.tz)

    rows = []
    for tk in history.columns:
        s = history[tk].dropna()
        if len(s) < 10:
            continue
        name, cat = meta.get(tk, (tk, 'Other'))
        last = s.iloc[-1]
        # 1D
        d1 = ((last / s.iloc[-2] - 1) * 100) if len(s) > 1 else np.nan
        # 5D
        d5 = ((last / s.iloc[-6] - 1) * 100) if len(s) > 5 else np.nan
        # 1M (21 dias uteis)
        d21 = ((last / s.iloc[-22] - 1) * 100) if len(s) > 21 else np.nan
        # YTD
        ytd_s = s[s.index >= ytd_start]
        ytd = ((last / ytd_s.iloc[0] - 1) * 100) if len(ytd_s) > 0 else np.nan
        # 1Y
        d252 = ((last / s.iloc[-253] - 1) * 100) if len(s) > 252 else np.nan
        # Std dev (anualizado)
        ret = s.pct_change().dropna()
        stdev = ret.rolling(252, min_periods=60).std().iloc[-1] * math.sqrt(252) * 100 \
                if len(ret) > 60 else np.nan
        rows.append({
            'Ticker': tk,
            'Name': name,
            'Category': cat,
            '1D_pct': round(d1, 2) if pd.notna(d1) else np.nan,
            '5D_pct': round(d5, 2) if pd.notna(d5) else np.nan,
            '1M_pct': round(d21, 2) if pd.notna(d21) else np.nan,
            'YTD_pct': round(ytd, 2) if pd.notna(ytd) else np.nan,
            '1Y_pct': round(d252, 2) if pd.notna(d252) else np.nan,
            'Std_Dev_pct': round(stdev, 2) if pd.notna(stdev) else np.nan,
        })
    df = pd.DataFrame(rows)
    # Sort por YTD desc
    df = df.sort_values('YTD_pct', ascending=False, na_position='last').reset_index(drop=True)
    return df


def fig_factor_heatmap(table: pd.DataFrame) -> go.Figure:
    """Heatmap Ticker x Horizonte, cor pela performance."""
    if len(table) == 0:
        return go.Figure().update_layout(title='Factor Monitor — sem dados',
                                           **_FIG_LAYOUT)
    cols = ['1D_pct', '5D_pct', '1M_pct', 'YTD_pct', '1Y_pct']
    labels = ['1D', '5D', '1M', 'YTD', '1Y']
    z = table[cols].values.astype(float)
    cat_abbr = {'Barra Factors': 'Barra', 'Momentum': 'Momo',
                 'GS Themes': 'GS', 'BBG Themes': 'BBG'}
    y = [f"[{cat_abbr.get(r['Category'], r['Category'][:4])}] {r['Name']}"
         for _, r in table.iterrows()]
    vmax = np.nanpercentile(np.abs(z[~np.isnan(z)]), 95) if np.any(~np.isnan(z)) else 20
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=y,
        colorscale=[[0, _C['red']], [0.5, '#0d1117'], [1, _C['green']]],
        zmin=-vmax, zmax=vmax,
        text=[[f'{v:+.2f}%' if pd.notna(v) else '' for v in row] for row in z],
        texttemplate='%{text}', textfont=dict(size=10, color='#cce8ff'),
        colorbar=dict(title='%', tickfont=dict(size=9))))
    fig.update_layout(
        title=f'GS Factor Monitor — {len(table)} factors/themes (ordenado por YTD)',
        **{**_FIG_LAYOUT, 'height': max(520, len(table) * 18),
           'margin': dict(l=210, r=60, t=55, b=40)})
    return fig


def fig_factor_leaderboard(table: pd.DataFrame, horizon: str = 'YTD_pct',
                             top_n: int = 10) -> go.Figure:
    """Top N winners + losers por horizonte."""
    if len(table) == 0:
        return go.Figure().update_layout(title='Leaderboard — sem dados',
                                           **_FIG_LAYOUT)
    df = table.dropna(subset=[horizon]).copy()
    top = df.head(top_n)
    bot = df.tail(top_n).iloc[::-1]
    combined = pd.concat([top, bot])
    colors = [_C['green'] if v > 0 else _C['red'] for v in combined[horizon]]
    # Nome principal + categoria abreviada (sem ticker, que ja aparece no hover)
    cat_abbr = {'Barra Factors': 'Barra', 'Momentum': 'Momo',
                 'GS Themes': 'GS', 'BBG Themes': 'BBG'}
    labels = [f"{r['Name']} · {cat_abbr.get(r['Category'], r['Category'][:4])}"
              for _, r in combined.iterrows()]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=combined[horizon], y=labels, orientation='h',
        marker_color=colors,
        text=[f'{v:+.2f}%' for v in combined[horizon]],
        textposition='outside', showlegend=False))
    fig.add_vline(x=0, line_color=_C['text_muted'], line_width=0.6)
    fig.update_layout(
        title=f'GS Factor Leaderboard — Top/Bottom {top_n} ({horizon.replace("_pct", "")})',
        xaxis_title='%',
        yaxis=dict(autorange='reversed'),
        **{**_FIG_LAYOUT, 'height': max(520, top_n * 40 + 100),
           'margin': dict(l=200, r=60, t=55, b=40)})
    return fig


def fig_factor_category_avg(table: pd.DataFrame) -> go.Figure:
    """Performance media por categoria (Barra / Momentum / Themes)."""
    if len(table) == 0:
        return go.Figure().update_layout(title='Categoria — sem dados',
                                           **_FIG_LAYOUT)
    grp = table.groupby('Category')[['1D_pct', '5D_pct', '1M_pct',
                                        'YTD_pct', '1Y_pct']].mean().round(2)
    fig = go.Figure()
    hz_colors = {'1D_pct': _C['accent'], '5D_pct': _C['orange'],
                  '1M_pct': _C['yellow'], 'YTD_pct': _C['green'],
                  '1Y_pct': _C['purple']}
    for col in grp.columns:
        hz_label = col.replace('_pct', '')
        fig.add_trace(go.Bar(
            x=grp.index, y=grp[col], name=hz_label,
            marker_color=hz_colors.get(col, _C['accent']),
            text=[f'{hz_label}: {v:+.2f}%' for v in grp[col]],
            textposition='outside', textfont=dict(size=10),
            hovertemplate=f'<b>%{{x}}</b><br>{hz_label}: %{{y:.2f}}%<extra></extra>'))
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.5)
    fig.update_layout(
        title='GS Factor Monitor — Media por Categoria e Horizonte',
        barmode='group', yaxis_title='%',
        **{**_FIG_LAYOUT, 'height': 500})
    return fig


def fig_factor_rolling(history: pd.DataFrame, tickers: list = None,
                         max_lines: int = 8) -> go.Figure:
    """
    Rolling relative performance dos top N factors (YTD) vs inicio da janela.
    Estilo "Performance of factors across regimes" do report Soria.
    """
    if len(history) == 0:
        return go.Figure().update_layout(title='Rolling — sem dados',
                                           **_FIG_LAYOUT)
    # Normaliza tudo em 100 no inicio
    norm = history.apply(lambda s: s / s.dropna().iloc[0] * 100 if len(s.dropna()) else s)
    # Pega os top N por retorno final
    final = norm.iloc[-1].dropna().sort_values(ascending=False)
    picked = tickers or list(final.index[:max_lines])
    fig = go.Figure()
    palette = ['#00c8ff', '#ff8c00', '#00ff88', '#ff4757', '#b44aff',
                '#ffd32a', '#ff6b9d', '#7efff5']
    for i, tk in enumerate(picked):
        if tk not in norm.columns:
            continue
        s = norm[tk].dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=tk,
            line=dict(color=palette[i % len(palette)], width=1.2)))
    fig.add_hline(y=100, line_color=_C['text_muted'], line_width=0.5, line_dash='dot')
    fig.update_layout(
        title=f'GS Factor Monitor — Rolling performance (norm=100) Top {max_lines}',
        yaxis_title='Level (norm=100)',
        **{**_FIG_LAYOUT, 'height': 500})
    return fig


def compute_factor_weekday_stats(history: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada factor, retorno medio por weekday.
    Retorna DataFrame wide: rows=factor, cols=weekday_name.
    """
    if len(history) == 0:
        return pd.DataFrame()
    ret = history.pct_change()
    ret = ret.assign(weekday=ret.index.strftime('%A'))
    ret_melt = ret.reset_index().melt(id_vars=['date', 'weekday'],
                                         var_name='ticker', value_name='ret')
    ret_melt = ret_melt.dropna(subset=['ret'])
    pv = ret_melt.pivot_table(index='ticker', columns='weekday',
                                values='ret', aggfunc='mean') * 100
    order = [d for d in WEEKDAY_ORDER if d in pv.columns]
    pv = pv[order]
    return pv.round(3)


def fig_factor_weekday_heatmap(pv: pd.DataFrame, universe_meta: dict) -> go.Figure:
    """Heatmap Factor × Weekday (mean daily return %)."""
    if len(pv) == 0:
        return go.Figure().update_layout(title='Weekday — sem dados', **_FIG_LAYOUT)
    # Ordena por YTD proxy (somar todos os dias)
    pv = pv.loc[pv.sum(axis=1).sort_values(ascending=False).index]
    # Labels curtos com categoria
    def _label(tk):
        name, cat = universe_meta.get(tk, (tk, 'Other'))
        cat_abbr = {'Barra Factors': 'Barra', 'Momentum': 'Momo',
                     'GS Themes': 'GS', 'BBG Themes': 'BBG'}.get(cat, cat[:4])
        return f'[{cat_abbr}] {name}'
    y_labels = [_label(tk) for tk in pv.index]

    z = pv.values
    vmax = np.nanpercentile(np.abs(z[~np.isnan(z)]), 95) if np.any(~np.isnan(z)) else 1
    fig = go.Figure(go.Heatmap(
        z=z, x=list(pv.columns), y=y_labels,
        colorscale=[[0, _C['red']], [0.5, '#0d1117'], [1, _C['green']]],
        zmin=-vmax, zmax=vmax,
        text=[[f'{v:+.2f}%' if pd.notna(v) else '' for v in row] for row in z],
        texttemplate='%{text}', textfont=dict(size=9, color='#cce8ff'),
        colorbar=dict(title='%', tickfont=dict(size=9))))
    fig.update_layout(
        title=f'Factor Monitor — Weekday Effect (retorno medio diario %, {len(pv)} factors)',
        **{**_FIG_LAYOUT, 'height': max(560, len(pv) * 18),
           'margin': dict(l=230, r=60, t=55, b=40)})
    return fig


def fig_factor_weekday_best(pv: pd.DataFrame, universe_meta: dict,
                                top_n: int = 20) -> go.Figure:
    """Top N factors com maior spread entre melhor e pior weekday."""
    if len(pv) == 0:
        return go.Figure().update_layout(title='Best Weekday — sem dados', **_FIG_LAYOUT)
    # Melhor dia e spread
    best_day = pv.idxmax(axis=1)
    best_val = pv.max(axis=1)
    worst_val = pv.min(axis=1)
    spread = best_val - worst_val
    df = pd.DataFrame({
        'best_day': best_day, 'best_val': best_val,
        'worst_val': worst_val, 'spread': spread,
    }).sort_values('spread', ascending=False).head(top_n)

    def _label(tk):
        name, cat = universe_meta.get(tk, (tk, 'Other'))
        return f'{name} ({df.loc[tk, "best_day"][:3]} +{df.loc[tk, "best_val"]:.2f}%)'
    labels = [_label(tk) for tk in df.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['spread'], y=labels, orientation='h',
                          marker_color=_C['accent'],
                          text=[f'spread {s:.2f}%' for s in df['spread']],
                          textposition='outside', showlegend=False))
    fig.update_layout(
        title=f'Top {top_n} factors com maior spread entre melhor e pior weekday',
        xaxis_title='best-worst spread %',
        yaxis=dict(autorange='reversed'),
        **{**_FIG_LAYOUT, 'height': max(500, top_n * 30 + 100),
           'margin': dict(l=280, r=60, t=55, b=40)})
    return fig


def compute_breadth(history: pd.DataFrame) -> pd.DataFrame:
    """
    Breadth do universo de factors:
      - % positivos 1D, 5D, 1M, 3M, YTD, 1Y
      - % acima da propria MA20, MA50, MA200
    """
    if len(history) == 0:
        return pd.DataFrame()
    close = history
    last = close.iloc[-1]
    rows = []
    for label, n in [('1D', 1), ('5D', 5), ('1M', 21), ('3M', 63), ('6M', 126), ('1Y', 252)]:
        if n >= len(close): continue
        past = close.iloc[-n - 1]
        chg = (last / past - 1) * 100
        rows.append({'horizon': label,
                      'pct_positive': round((chg > 0).sum() / chg.notna().sum() * 100, 2),
                      'pct_negative': round((chg < 0).sum() / chg.notna().sum() * 100, 2),
                      'avg_return_pct': round(chg.mean(), 2),
                      'median_return_pct': round(chg.median(), 2),
                      'n_factors': int(chg.notna().sum())})
    # Acima das MAs
    for w in [20, 50, 200]:
        if w >= len(close): continue
        ma = close.rolling(w, min_periods=max(5, w // 4)).mean().iloc[-1]
        above = (last > ma).sum()
        tot = ma.notna().sum()
        rows.append({'horizon': f'>MA{w}',
                      'pct_positive': round(above / tot * 100, 2) if tot else np.nan,
                      'pct_negative': round((1 - above / tot) * 100, 2) if tot else np.nan,
                      'avg_return_pct': np.nan, 'median_return_pct': np.nan,
                      'n_factors': int(tot)})
    return pd.DataFrame(rows)


def fig_breadth(breadth_df: pd.DataFrame) -> go.Figure:
    """Barras agrupadas: % positivos vs negativos por horizonte."""
    if len(breadth_df) == 0:
        return go.Figure().update_layout(title='Breadth — sem dados', **_FIG_LAYOUT)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=breadth_df['horizon'], y=breadth_df['pct_positive'],
                          name='% positivos', marker_color=_C['green'],
                          text=[f'{v:.0f}%' for v in breadth_df['pct_positive']],
                          textposition='outside'))
    fig.add_trace(go.Bar(x=breadth_df['horizon'], y=breadth_df['pct_negative'],
                          name='% negativos', marker_color=_C['red'],
                          text=[f'{v:.0f}%' for v in breadth_df['pct_negative']],
                          textposition='outside'))
    fig.add_hline(y=50, line_color=_C['text_muted'], line_dash='dash', line_width=0.7)
    fig.update_layout(
        title=f'Breadth — {int(breadth_df["n_factors"].max())} factors, '
              f'distribuicao positivos/negativos por horizonte',
        barmode='group', yaxis_title='%', yaxis_range=[0, 105],
        **_FIG_LAYOUT)
    return fig


def compute_rs_ratio(price: pd.Series, benchmark: pd.Series,
                       lookback: int = 63, momentum_period: int = 20) -> pd.DataFrame:
    """
    Simplified JdK RS-Ratio + RS-Momentum.
    - RS = price/benchmark centralizado em 100 usando media de lookback dias
    - Momentum = mudanca nos ultimos momentum_period dias
    """
    common_idx = price.index.intersection(benchmark.index)
    p = price.reindex(common_idx).ffill()
    b = benchmark.reindex(common_idx).ffill()
    rel = p / b
    rel_ma = rel.rolling(lookback, min_periods=max(10, lookback // 3)).mean()
    rs_ratio = 100 + (rel / rel_ma - 1) * 100
    rs_momentum = 100 + (rs_ratio - rs_ratio.shift(momentum_period))
    return pd.DataFrame({'RS_Ratio': rs_ratio, 'RS_Momentum': rs_momentum},
                          index=common_idx)


def fig_rs_ratio_quadrant(history: pd.DataFrame, benchmark: pd.Series,
                             universe_meta: dict, top_n: int = 30,
                             tail_length: int = 10) -> go.Figure:
    """
    RRG-style chart:
      x = RS-Ratio (100 = equal to benchmark)
      y = RS-Momentum (100 = no change in RS-Ratio)
    4 quadrantes:
      Q1 (top-right): LEADING (RS>100, momentum up)
      Q2 (top-left):  IMPROVING (RS<100, momentum up)
      Q3 (bottom-left): LAGGING (RS<100, momentum down)
      Q4 (bottom-right): WEAKENING (RS>100, momentum down)
    Pontos = factors. Trails = ultimas tail_length semanas.
    """
    if len(history) == 0 or len(benchmark) == 0:
        return go.Figure().update_layout(title='RS-Ratio — sem dados', **_FIG_LAYOUT)

    # Seleciona top N mais liquidos/relevantes (todos se <top_n)
    tickers = list(history.columns)[:top_n]
    # Semanal (toda sexta ou ultimo do periodo)
    weekly_idx = history.index[-tail_length * 5::5]  # aprox semanal
    if len(weekly_idx) < 2:
        weekly_idx = history.index[-tail_length:]

    fig = go.Figure()
    # Zonas sombreadas
    fig.add_shape(type='rect', x0=100, x1=130, y0=100, y1=130,
                   fillcolor='rgba(63,185,80,0.08)', line_width=0, layer='below')
    fig.add_shape(type='rect', x0=70, x1=100, y0=100, y1=130,
                   fillcolor='rgba(88,166,255,0.08)', line_width=0, layer='below')
    fig.add_shape(type='rect', x0=70, x1=100, y0=70, y1=100,
                   fillcolor='rgba(248,81,73,0.08)', line_width=0, layer='below')
    fig.add_shape(type='rect', x0=100, x1=130, y0=70, y1=100,
                   fillcolor='rgba(240,136,62,0.08)', line_width=0, layer='below')

    cat_colors = {'Barra Factors': _C['accent'], 'Momentum': _C['orange'],
                   'GS Themes': _C['purple'], 'BBG Themes': _C['teal']}

    for tk in tickers:
        try:
            rs_df = compute_rs_ratio(history[tk].dropna(), benchmark)
            rs_df = rs_df.dropna()
            if len(rs_df) < 5:
                continue
        except Exception:
            continue

        name, cat = universe_meta.get(tk, (tk, 'Other'))
        color = cat_colors.get(cat, _C['text'])
        # Trail: ultimas tail_length semanas
        trail = rs_df.iloc[-tail_length * 5::5] if len(rs_df) > tail_length * 5 else rs_df.tail(tail_length)
        # Ponto atual
        cur = rs_df.iloc[-1]

        # Trail como linha tracejada com opacidade crescente
        if len(trail) >= 2:
            fig.add_trace(go.Scatter(
                x=trail['RS_Ratio'], y=trail['RS_Momentum'],
                mode='lines+markers',
                line=dict(color=color, width=1, dash='dot'),
                marker=dict(size=3, color=color),
                showlegend=False, opacity=0.4, hoverinfo='skip'))

        # Ponto atual com nome
        fig.add_trace(go.Scatter(
            x=[cur['RS_Ratio']], y=[cur['RS_Momentum']],
            mode='markers+text',
            marker=dict(size=11, color=color,
                         line=dict(color='#fff', width=1.2)),
            text=[name[:14]], textposition='top center',
            textfont=dict(size=9, color=color),
            name=f'[{cat}] {name}',
            showlegend=False,
            hovertemplate=f'<b>{name} ({tk})</b><br>'
                           f'RS-Ratio: %{{x:.2f}}<br>'
                           f'RS-Momentum: %{{y:.2f}}<extra></extra>'))

    # Linhas de eixo em 100
    fig.add_hline(y=100, line_color=_C['text_muted'], line_width=0.7, line_dash='dash')
    fig.add_vline(x=100, line_color=_C['text_muted'], line_width=0.7, line_dash='dash')
    # Quadrante labels
    fig.add_annotation(x=125, y=128, text='<b>LEADING</b>',
                        showarrow=False, font=dict(color=_C['green'], size=14),
                        xanchor='right')
    fig.add_annotation(x=75, y=128, text='<b>IMPROVING</b>',
                        showarrow=False, font=dict(color=_C['accent'], size=14),
                        xanchor='left')
    fig.add_annotation(x=75, y=72, text='<b>LAGGING</b>',
                        showarrow=False, font=dict(color=_C['red'], size=14),
                        xanchor='left')
    fig.add_annotation(x=125, y=72, text='<b>WEAKENING</b>',
                        showarrow=False, font=dict(color=_C['orange'], size=14),
                        xanchor='right')

    fig.update_layout(
        title=f'RS-Ratio RRG vs benchmark — {len(tickers)} factors '
              f'(trails = ultimas {tail_length}s semanais)',
        xaxis=dict(title='RS-Ratio (>100 = outperforming)', range=[70, 130]),
        yaxis=dict(title='RS-Momentum (>100 = RS acelerando)', range=[70, 130]),
        **{**_FIG_LAYOUT, 'height': 680})
    return fig


def fig_regime_weekday_matrix(df: pd.DataFrame, regime_df: pd.DataFrame,
                                 dim: str = 'trend') -> go.Figure:
    """
    Heatmap: weekday × regime_state, celula = RTH retorno medio %.
    Mostra que weekday funciona dentro de cada regime state.
    """
    if regime_df is None or len(regime_df) == 0:
        return go.Figure().update_layout(title=f'Regime×Weekday — sem dados',
                                           **_FIG_LAYOUT)
    col = f'{dim}_regime'
    if col not in regime_df.columns or 'rth_return' not in df.columns:
        return go.Figure().update_layout(title=f'Regime×Weekday — sem dados',
                                           **_FIG_LAYOUT)
    m = pd.DataFrame({
        'rth': df['rth_return'].reindex(regime_df.index),
        'weekday': df.index.strftime('%A'),
        'regime': regime_df[col],
    }).dropna()
    pv = m.pivot_table(index='regime', columns='weekday',
                        values='rth', aggfunc='mean') * 100
    order = [d for d in WEEKDAY_ORDER if d in pv.columns]
    pv = pv[order]
    n_pv = m.pivot_table(index='regime', columns='weekday',
                           values='rth', aggfunc='count')[order]

    z = pv.values
    vmax = np.nanpercentile(np.abs(z[~np.isnan(z)]), 95) if np.any(~np.isnan(z)) else 1
    text = [[f'{v:+.2f}%<br>n={int(n_pv.iloc[i, j]) if pd.notna(n_pv.iloc[i, j]) else 0}'
             if pd.notna(v) else '' for j, v in enumerate(row)]
            for i, row in enumerate(z)]
    fig = go.Figure(go.Heatmap(
        z=z, x=list(pv.columns), y=list(pv.index),
        colorscale=[[0, _C['red']], [0.5, '#0d1117'], [1, _C['green']]],
        zmin=-vmax, zmax=vmax,
        text=text, texttemplate='%{text}',
        textfont=dict(size=11, color='#cce8ff'),
        colorbar=dict(title='RTH %')))
    fig.update_layout(
        title=f'Regime × Weekday — retorno medio RTH por {dim} × dia da semana (%)',
        **{**_FIG_LAYOUT, 'height': 400})
    return fig


def run_gs_factor_section(years: int = 2, benchmark_ticker: str = 'SPY US Equity',
                             session_frame: pd.DataFrame = None,
                             regime_df: pd.DataFrame = None) -> dict:
    """Pipeline completo da secao GS Factors. Retorna dict com tabela + figs."""
    history = fetch_gs_factors(years=years)
    if len(history) == 0:
        return {}
    table = factor_monitor_table(history)

    # Meta pra labels
    universe_meta = {}
    for cat, items in GS_FACTOR_UNIVERSE.items():
        for tk, name in items:
            universe_meta[tk] = (name, cat)

    # Weekday stats
    wd_stats = compute_factor_weekday_stats(history)

    # Breadth
    breadth = compute_breadth(history)

    # RS-Ratio — baixa benchmark separado
    bench_series = None
    try:
        bench_bundle = load_daily(benchmark_ticker, years=max(years, 2))
        bench_series = bench_bundle['close'] if bench_bundle is not None else None
    except Exception as e:
        log.warning(f'[gs_factors] benchmark {benchmark_ticker} falhou: {e}')

    figs = {
        'heatmap': fig_factor_heatmap(table),
        'leaderboard_ytd': fig_factor_leaderboard(table, 'YTD_pct'),
        'leaderboard_1m': fig_factor_leaderboard(table, '1M_pct'),
        'category_avg': fig_factor_category_avg(table),
        'rolling_top': fig_factor_rolling(history, max_lines=8),
        'weekday_heatmap': fig_factor_weekday_heatmap(wd_stats, universe_meta),
        'weekday_best': fig_factor_weekday_best(wd_stats, universe_meta),
        'breadth': fig_breadth(breadth),
    }
    if bench_series is not None and len(bench_series) > 50:
        figs['rs_ratio_rrg'] = fig_rs_ratio_quadrant(
            history, bench_series, universe_meta, top_n=30, tail_length=10)

    # Regime × Weekday (se tiver os dados)
    if session_frame is not None and regime_df is not None:
        for dim in ['trend', 'vol', 'voldir']:
            try:
                figs[f'regime_wd_{dim}'] = fig_regime_weekday_matrix(
                    session_frame, regime_df, dim=dim)
            except Exception as e:
                log.warning(f'regime_wd {dim} falhou: {e}')

    return {
        'history': history,
        'table': table,
        'weekday_stats': wd_stats,
        'breadth': breadth,
        'benchmark_ticker': benchmark_ticker,
        'universe_meta': universe_meta,
        'figs': figs,
    }


# =============================================================================
# 7d. PASSIVE BREAKS THE MARKET (Green/Krishnan/Sturm SSRN 2025)
# =============================================================================
#
# Implementacao do modelo:
#
#   dS(t) = kappa*(1-p(t))*(F(t) - S(t))*dt + sigma*sqrt(F(t)*S(t))*dW(t)
#
# com F(t) = F0*exp(r*t), p(t) logistica em (0,1), W brownian.
#
# Thresholds criticos (para kappa=0.0909, sigma=0.1247 calibrados por eles):
#   Estavel:          p <= 1 - 3*sigma^2/(4*kappa) = 0.87
#   Lyapunov (cubic): 0.87 < p <= 1 - sigma^2/(2*kappa) = 0.91  (vol explode)
#   Feller (collapse): p > 0.91  (probabilidade positiva de S -> 0)
#
# Ajuste Haddad (HHL25): replace p com p_hat*p = p/(1 + chi*(1-p)), chi=3.
# Equivale a shift de ln(1+chi)/alpha = 13.08 anos no tempo.
#
# Volatilidade instantanea: V(t) = sigma * sqrt(F(t)/S(t))
# Vol estavel (regime 1):  V_stable = 2*sigma * sqrt((kappa*(1-p)+r)/(4*kappa*(1-p) - 3*sigma^2))
# =============================================================================

@dataclass
class PassiveBreaksConfig:
    """Parametros do modelo Green/Krishnan/Sturm (calibrados em 1926-1994 S&P 500)."""
    kappa: float = 0.0909       # mean reversion speed (anualizado)
    sigma: float = 0.1247       # vol parameter (anualizado)
    r: float = 0.0917           # growth rate de F
    alpha: float = 0.106        # logistic growth of passive share
    t0: float = 2021.0          # ano onde p(t)=0.5 (inflection da logistica)
    chi: float = 3.0            # Haddad strategic response (1/(1+chi*(1-p)))


def passive_share(t_year: float, alpha: float = 0.106, t0: float = 2021.0) -> float:
    """Logistic passive share at given year. Varia de 0 a 1."""
    return 1.0 / (1.0 + math.exp(-alpha * (t_year - t0)))


def haddad_effective_share(p: float, chi: float = 3.0) -> float:
    """p_effetiva apos ajuste Haddad: p_hat * p, where p_hat = 1/(1 + chi*(1-p))."""
    return p / (1.0 + chi * (1.0 - p))


def critical_thresholds(kappa: float, sigma: float) -> dict:
    """
    Thresholds criticos para passive share:
      - lyapunov: acima deste p, vol cresce em cubic speed
      - feller: acima deste p, S(t) tem prob>0 de colapsar a 0
    """
    lyapunov = 1.0 - (3.0 * sigma * sigma) / (4.0 * kappa)
    feller = 1.0 - (sigma * sigma) / (2.0 * kappa)
    return {
        'lyapunov_threshold': max(0, min(1, lyapunov)),
        'feller_threshold': max(0, min(1, feller)),
    }


def stable_volatility(p: float, kappa: float, sigma: float, r: float) -> float:
    """V_stable in regime estavel (p < lyapunov). NaN se fora."""
    num = kappa * (1.0 - p) + r
    den = 4.0 * kappa * (1.0 - p) - 3.0 * sigma * sigma
    if den <= 0:
        return np.nan
    return 2.0 * sigma * math.sqrt(num / den)


def calibrate_passive_breaks(close: pd.Series, r_override: float = None,
                                use_paper_defaults: bool = True) -> PassiveBreaksConfig:
    """
    Calibra kappa, sigma, r do modelo.

    use_paper_defaults=True (default): usa os valores fixos do paper
      calibrados em 1926-1994 (era limpa, sem passive relevante):
      kappa=0.0909, sigma=0.1247, r=0.0917.
      Isso e o RECOMENDADO — calibrar em dados modernos (2005+) da valores
      contaminados pela propria distorcao que o modelo tenta capturar.

    use_paper_defaults=False: calibra via regressao eq (6) do paper:
        dS/sqrt(F*S)  =  kappa*dt * (F-S)/sqrt(F*S) + sigma*sqrt(dt)*eps

    Retorna PassiveBreaksConfig.
    """
    if use_paper_defaults:
        log.info('[passive_breaks] usando parametros fixos do paper (1926-1994): '
                 'kappa=0.0909 sigma=0.1247 r=0.0917')
        return PassiveBreaksConfig()

    s = close.dropna().astype(float)
    if len(s) < 500:
        log.warning('[passive_breaks] dados insuficientes pra calibracao, '
                     'caindo em defaults do paper')
        return PassiveBreaksConfig()

    # tempo em anos (252 uteis/ano)
    dt = 1.0 / 252.0
    n = len(s)
    T_years = (n - 1) * dt

    # r: ajusta F0*exp(r*T) = S_final, com F0=S_inicial
    S0, ST = float(s.iloc[0]), float(s.iloc[-1])
    if r_override is not None:
        r = r_override
    else:
        r = math.log(ST / S0) / T_years if T_years > 0 and S0 > 0 else 0.09

    # F(t) pra cada ponto
    t_arr = np.arange(n) * dt
    F = S0 * np.exp(r * t_arr)
    S = s.values

    # Discreto (eq 6): y = kappa * dt * (F-S)/sqrt(F*S) + sigma*sqrt(dt)*eps
    # y = dS / sqrt(F*S)
    dS = np.diff(S)
    sqrt_FS = np.sqrt(F[:-1] * S[:-1])
    # filtra pontos ruins
    mask = (sqrt_FS > 0) & np.isfinite(sqrt_FS)
    y = dS[mask] / sqrt_FS[mask]
    x = dt * (F[:-1][mask] - S[:-1][mask]) / sqrt_FS[mask]

    # Regressao sem intercepto: kappa = (x'y)/(x'x)
    if len(x) < 100:
        return PassiveBreaksConfig(r=r)
    kappa = float((x * y).sum() / (x * x).sum())
    kappa = max(0.001, kappa)  # anti-negativo

    # sigma dos residuos
    resid = y - kappa * x
    sigma = float(np.std(resid) / math.sqrt(dt))

    # Sanity checks: se parametros absurdos, cai em defaults do paper
    if not (0.01 < kappa < 1.0):
        log.warning(f'[passive_breaks] kappa calibrado {kappa:.4f} fora do razoavel, '
                     f'usando default 0.0909')
        kappa = 0.0909
    if not (0.01 < sigma < 1.0):
        log.warning(f'[passive_breaks] sigma calibrado {sigma:.4f} fora do razoavel, '
                     f'usando default 0.1247')
        sigma = 0.1247

    log.info(f'[passive_breaks] calibrado: kappa={kappa:.4f} sigma={sigma:.4f} r={r:.4f} '
              f'(T={T_years:.1f}y, n={n})')

    return PassiveBreaksConfig(kappa=kappa, sigma=sigma, r=r)


def simulate_passive_breaks(S0: float, F0: float, cfg: PassiveBreaksConfig,
                              n_paths: int = 100, horizon_years: float = 20.0,
                              steps_per_year: int = 52,
                              p_fn=None, include_haddad: bool = True,
                              seed: int = 42) -> np.ndarray:
    """
    Monte Carlo da equacao (2) do paper.

    p_fn(t_year) -> retorna p(t). Se None, usa logistica default do cfg.
    steps_per_year=52 (semanal) pra balancear velocidade e precisao.

    Retorna array shape (n_paths, n_steps+1).
    """
    np.random.seed(seed)
    n_steps = int(horizon_years * steps_per_year)
    dt = 1.0 / steps_per_year
    sqrt_dt = math.sqrt(dt)
    t0_year = cfg.t0 - (horizon_years / 2)

    # Pre-calcula passive share e F ao longo do tempo (vetorizado)
    t_arr = np.arange(n_steps) * dt
    years = t0_year + t_arr
    if p_fn is None:
        p_raw_arr = np.array([passive_share(y, cfg.alpha, cfg.t0) for y in years])
    else:
        p_raw_arr = np.array([p_fn(y) for y in years])
    if include_haddad:
        p_eff_arr = p_raw_arr / (1.0 + cfg.chi * (1.0 - p_raw_arr))
    else:
        p_eff_arr = p_raw_arr
    F_arr = F0 * np.exp(cfg.r * t_arr)

    paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = S0

    # Gera TODOS os randoms de uma vez
    rnd = np.random.randn(n_paths, n_steps)

    for i in range(n_steps):
        p_eff = p_eff_arr[i]
        F_t = F_arr[i]
        S_prev = paths[:, i]
        S_safe = np.maximum(S_prev, 1e-6)
        drift = cfg.kappa * (1.0 - p_eff) * (F_t - S_prev) * dt
        diffusion = cfg.sigma * np.sqrt(F_t * S_safe) * sqrt_dt * rnd[:, i]
        paths[:, i + 1] = np.maximum(S_prev + drift + diffusion, 0.0)

    return paths


def compute_passive_breaks_state(close: pd.Series, cfg: PassiveBreaksConfig,
                                    current_year: float = None) -> dict:
    """
    Snapshot do estado atual do mercado no framework do paper.
    Retorna dict com p_atual, thresholds, distancia, vol estavel, zona, etc.
    """
    if current_year is None:
        current_year = datetime.now().year + (datetime.now().month - 1) / 12
    p_now = passive_share(current_year, cfg.alpha, cfg.t0)
    p_eff = haddad_effective_share(p_now, cfg.chi)
    thr = critical_thresholds(cfg.kappa, cfg.sigma)
    lyap, feller = thr['lyapunov_threshold'], thr['feller_threshold']

    # Zona
    if p_now < lyap:
        zone = 'ESTAVEL'
        zone_color = _C['green']
    elif p_now < feller:
        zone = 'LYAPUNOV (vol cresce em cubic speed)'
        zone_color = _C['yellow']
    else:
        zone = 'FELLER (risco de colapso a 0)'
        zone_color = _C['red']

    # Vol estavel se aplicavel
    V_stable = stable_volatility(p_now, cfg.kappa, cfg.sigma, cfg.r)

    # F/S ratio atual (proxy: fit log-trend aos ultimos T anos)
    s = close.dropna()
    if len(s) > 500:
        n = len(s)
        t_arr = np.arange(n) / 252.0
        # F: ajusta F(0)*exp(r*t) pelo primeiro ponto
        F0 = float(s.iloc[0])
        F_series = F0 * np.exp(cfg.r * t_arr)
        F_now = float(F_series[-1])
        S_now = float(s.iloc[-1])
        fs_ratio = F_now / S_now
        current_vol = cfg.sigma * math.sqrt(fs_ratio)
    else:
        F_now = np.nan; S_now = np.nan; fs_ratio = np.nan; current_vol = np.nan

    return {
        'current_year': round(current_year, 2),
        'p_raw_pct': round(p_now * 100, 2),
        'p_effective_haddad_pct': round(p_eff * 100, 2),
        'lyapunov_threshold_pct': round(lyap * 100, 2),
        'feller_threshold_pct': round(feller * 100, 2),
        'distance_to_lyapunov_pct': round((lyap - p_now) * 100, 2),
        'distance_to_feller_pct': round((feller - p_now) * 100, 2),
        'zone': zone,
        'zone_color': zone_color,
        'V_stable_pct': round(V_stable * 100, 2) if pd.notna(V_stable) else np.nan,
        'current_vol_pct': round(current_vol * 100, 2) if pd.notna(current_vol) else np.nan,
        'F_current': round(F_now, 2) if pd.notna(F_now) else np.nan,
        'S_current': round(S_now, 2) if pd.notna(S_now) else np.nan,
        'F_over_S': round(fs_ratio, 3) if pd.notna(fs_ratio) else np.nan,
        'kappa': round(cfg.kappa, 4),
        'sigma': round(cfg.sigma, 4),
        'r': round(cfg.r, 4),
        'alpha': cfg.alpha,
        't0': cfg.t0,
    }


# ---- Plotly figs ----

def project_threshold_hit(cfg: PassiveBreaksConfig, target_p: float,
                             alpha_sigma_pct: float = 15,
                             us_equity_mktcap_usd: float = 60e12) -> pd.DataFrame:
    """
    Projeta quando passive share chega a target_p sob varios cenarios de alpha.

    alpha_sigma_pct: desvio padrao do alpha em % (15% default, baseado na
                      sensibilidade do fit logistico Haddad 1995-2019).
    us_equity_mktcap_usd: total mkt cap US equities pra estimar AUM adicional.

    Retorna DataFrame com 5 cenarios: -2sig, -1sig, base, +1sig, +2sig.
    """
    now_year = datetime.now().year + (datetime.now().month - 1) / 12
    p_now = passive_share(now_year, cfg.alpha, cfg.t0)
    aum_now = p_now * us_equity_mktcap_usd
    aum_target = target_p * us_equity_mktcap_usd
    aum_needed = aum_target - aum_now

    rows = []
    scenarios = [
        ('-2σ (fluxo mais lento)', -2),
        ('-1σ (fluxo mais lento)', -1),
        ('Base (α calibrado)', 0),
        ('+1σ (fluxo mais rapido)', 1),
        ('+2σ (fluxo mais rapido)', 2),
    ]
    for label, mult in scenarios:
        alpha_adj = cfg.alpha * (1 + mult * alpha_sigma_pct / 100)
        if alpha_adj <= 0 or target_p <= p_now:
            t_hit = float('inf')
        else:
            t_hit = cfg.t0 - math.log((1 - target_p) / target_p) / alpha_adj
        yrs = t_hit - now_year if math.isfinite(t_hit) else np.nan
        days = yrs * 365.25 if pd.notna(yrs) else np.nan
        rows.append({
            'cenario': label,
            'alpha': round(alpha_adj, 4),
            'data_hit': f'{int(t_hit)}-{int((t_hit - int(t_hit)) * 12) + 1:02d}'
                         if math.isfinite(t_hit) else 'nunca',
            'years_ate_hit': round(yrs, 1) if pd.notna(yrs) else np.nan,
            'days_ate_hit': int(round(days)) if pd.notna(days) else np.nan,
            'aum_adicional_usd_tn': round(aum_needed / 1e12, 2) if aum_needed > 0 else 0,
            'aum_adicional_por_dia_usd_bn': round(aum_needed / 1e9 / days, 3)
                                                 if pd.notna(days) and days > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def fig_threshold_projection(cfg: PassiveBreaksConfig,
                                us_equity_mktcap_usd: float = 60e12) -> go.Figure:
    """
    Curva logistica da passive share + projecoes em 5 cenarios ate Lyapunov e Feller.
    Linhas verticais marcando as datas de hit em cada cenario.
    """
    thr = critical_thresholds(cfg.kappa, cfg.sigma)
    lyap_p = thr['lyapunov_threshold']
    feller_p = thr['feller_threshold']
    now_year = datetime.now().year + (datetime.now().month - 1) / 12
    p_now = passive_share(now_year, cfg.alpha, cfg.t0)

    years = np.linspace(1990, 2060, 701)
    alpha_sigma_pct = 15
    scenarios = [
        ('-2σ', -2, _C['green']),
        ('-1σ', -1, _C['teal']),
        ('Base', 0, _C['accent']),
        ('+1σ', 1, _C['orange']),
        ('+2σ', 2, _C['red']),
    ]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         row_heights=[0.7, 0.3],
                         subplot_titles=(
                             'Projecao Passive Share — 5 cenarios (alpha ±1σ, ±2σ)',
                             'Distancia aos thresholds (pp) — quanto falta pra Lyapunov / Feller'))

    for label, mult, color in scenarios:
        alpha_adj = cfg.alpha * (1 + mult * alpha_sigma_pct / 100)
        p = np.array([passive_share(y, alpha_adj, cfg.t0) for y in years]) * 100
        fig.add_trace(go.Scatter(
            x=years, y=p, name=label,
            line=dict(color=color, width=2.0 if mult == 0 else 1.2,
                       dash='solid' if mult == 0 else ('dot' if abs(mult) == 2 else 'dash'))),
            row=1, col=1)

    # Thresholds como bandas horizontais
    fig.add_hline(y=lyap_p * 100, line_color=_C['yellow'], line_dash='dash',
                   line_width=1.3, row=1, col=1,
                   annotation_text=f'Lyapunov {lyap_p*100:.1f}%',
                   annotation_font_color=_C['yellow'], annotation_position='left')
    fig.add_hline(y=feller_p * 100, line_color=_C['red'], line_dash='dash',
                   line_width=1.3, row=1, col=1,
                   annotation_text=f'Feller {feller_p*100:.1f}%',
                   annotation_font_color=_C['red'], annotation_position='left')

    # Linha vertical NOW
    fig.add_vline(x=now_year, line_color='#fff', line_width=1.5, line_dash='dot',
                   row=1, col=1,
                   annotation_text=f'HOJE {p_now*100:.1f}%',
                   annotation_font_color='#fff')

    # Paineel 2: distancia pp aos thresholds ao longo do tempo (base scenario)
    p_base = np.array([passive_share(y, cfg.alpha, cfg.t0) for y in years]) * 100
    dist_lyap = lyap_p * 100 - p_base
    dist_feller = feller_p * 100 - p_base
    fig.add_trace(go.Scatter(x=years, y=dist_lyap, name='dist Lyapunov',
                              line=dict(color=_C['yellow'], width=1.3),
                              showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=years, y=dist_feller, name='dist Feller',
                              line=dict(color=_C['red'], width=1.3),
                              showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.6, row=2, col=1)
    fig.add_vline(x=now_year, line_color='#fff', line_width=1.5, line_dash='dot',
                   row=2, col=1)

    fig.update_yaxes(range=[0, 100], title_text='Passive Share %', row=1, col=1)
    fig.update_yaxes(title_text='pp ate threshold', row=2, col=1)
    fig.update_xaxes(range=[2010, 2055])
    fig.update_layout(**{**_FIG_LAYOUT, 'height': 720})
    return fig


def fig_passive_share_curve(cfg: PassiveBreaksConfig) -> go.Figure:
    """Curva logistica da passive share + linhas de threshold + ponto atual."""
    years = np.linspace(1990, 2050, 601)
    p = np.array([passive_share(y, cfg.alpha, cfg.t0) for y in years]) * 100
    thr = critical_thresholds(cfg.kappa, cfg.sigma)
    lyap = thr['lyapunov_threshold'] * 100
    feller = thr['feller_threshold'] * 100
    current_year = datetime.now().year + (datetime.now().month - 1) / 12
    p_now = passive_share(current_year, cfg.alpha, cfg.t0) * 100

    fig = go.Figure()
    # Thresholds como banda colorida (em vez de hline com annotation)
    fig.add_hrect(y0=0, y1=lyap, fillcolor='rgba(63,185,80,0.06)', line_width=0,
                   layer='below')
    fig.add_hrect(y0=lyap, y1=feller, fillcolor='rgba(210,153,34,0.10)',
                   line_width=0, layer='below',
                   annotation_text=f'LYAPUNOV {lyap:.1f}%',
                   annotation_position='top left',
                   annotation_font_color=_C['yellow'])
    fig.add_hrect(y0=feller, y1=100, fillcolor='rgba(248,81,73,0.10)',
                   line_width=0, layer='below',
                   annotation_text=f'FELLER {feller:.1f}%',
                   annotation_position='top left',
                   annotation_font_color=_C['red'])

    # Linha principal Passive Share
    fig.add_trace(go.Scatter(x=years, y=p, name='Passive Share',
                              line=dict(color=_C['accent'], width=2.4)))
    # Haddad-ajustada
    p_eff = np.array([haddad_effective_share(passive_share(y, cfg.alpha, cfg.t0), cfg.chi)
                        for y in years]) * 100
    fig.add_trace(go.Scatter(x=years, y=p_eff, name='Haddad-ajustada (p_eff)',
                              line=dict(color=_C['purple'], width=1.4, dash='dash')))

    # Current point — marker grande + label sem overlap
    fig.add_trace(go.Scatter(x=[current_year], y=[p_now],
                              mode='markers',
                              marker=dict(color='#fff', size=18,
                                          line=dict(color=_C['orange'], width=3)),
                              name=f'Hoje {p_now:.2f}%', showlegend=True))
    # Linha vertical no now
    fig.add_vline(x=current_year, line_color=_C['orange'], line_width=1,
                   line_dash='dot')

    fig.update_layout(
        title='Passive Share Logistica — trajetoria historica + extrapolacao',
        xaxis_title='Ano', yaxis_title='Passive Share %',
        yaxis_range=[0, 100],
        xaxis_range=[1995, 2050],
        **{**_FIG_LAYOUT, 'height': 500,
           'legend': dict(orientation='h', yanchor='bottom', y=-0.18,
                           xanchor='center', x=0.5, font=dict(size=11))})
    return fig


def fig_passive_state_gauge(state: dict) -> go.Figure:
    """Gauge horizontal limpo: zonas coloridas + marcador atual."""
    p = state['p_raw_pct']
    lyap = state['lyapunov_threshold_pct']
    feller = state['feller_threshold_pct']

    fig = go.Figure()
    # Zonas como shapes (nao bars) pra nao poluir
    fig.add_shape(type='rect', x0=0, x1=lyap, y0=0, y1=1,
                   fillcolor='rgba(63,185,80,0.55)', line_width=0)
    fig.add_shape(type='rect', x0=lyap, x1=feller, y0=0, y1=1,
                   fillcolor='rgba(210,153,34,0.55)', line_width=0)
    fig.add_shape(type='rect', x0=feller, x1=100, y0=0, y1=1,
                   fillcolor='rgba(248,81,73,0.55)', line_width=0)

    # Labels das zonas (so texto, sem barra duplicada)
    fig.add_annotation(x=lyap / 2, y=0.5, text=f'<b>ESTAVEL</b><br>0 → {lyap:.1f}%',
                        showarrow=False, font=dict(color='#fff', size=13))
    fig.add_annotation(x=(lyap + feller) / 2, y=0.5,
                        text=f'<b>LYAPUNOV</b><br>{lyap:.1f} → {feller:.1f}%',
                        showarrow=False, font=dict(color='#fff', size=13))
    fig.add_annotation(x=(feller + 100) / 2, y=0.5,
                        text=f'<b>FELLER</b><br>&gt; {feller:.1f}%',
                        showarrow=False, font=dict(color='#fff', size=13))

    # Marker atual — linha branca grossa + seta + texto acima
    fig.add_shape(type='line', x0=p, x1=p, y0=0, y1=1,
                   line=dict(color='#fff', width=4))
    fig.add_annotation(x=p, y=1.35, text=f'<b>ATUAL<br>{p:.2f}%</b>',
                        showarrow=True, arrowhead=2, arrowcolor='#fff',
                        arrowwidth=2, ax=0, ay=-30,
                        font=dict(color='#fff', size=14),
                        bgcolor='rgba(1,8,20,0.9)', bordercolor='#fff', borderwidth=1)

    fig.update_xaxes(range=[0, 100], title='Passive Share %',
                      tickmode='array', tickvals=[0, 25, 50, 75, 100],
                      ticktext=['0%', '25%', '50%', '75%', '100%'])
    fig.update_yaxes(range=[0, 2], visible=False)
    fig.update_layout(
        title=f'Passive Breaks Zone — {state["zone"]}',
        showlegend=False,
        **{**_FIG_LAYOUT, 'height': 260, 'margin': dict(l=40, r=40, t=80, b=60)})
    return fig


def fig_s_vs_f(close: pd.Series, cfg: PassiveBreaksConfig, ticker: str) -> go.Figure:
    """S(t) vs F(t) no estilo da Figura 2 do paper (log scale)."""
    s = close.dropna()
    if len(s) < 50:
        return go.Figure().update_layout(title='S vs F — sem dados', **_FIG_LAYOUT)
    n = len(s)
    t_arr = np.arange(n) / 252.0
    F0 = float(s.iloc[0])
    F = F0 * np.exp(cfg.r * t_arr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, name='S(t) market',
                              line=dict(color=_C['accent'], width=1.3)))
    fig.add_trace(go.Scatter(x=s.index, y=F, name=f'F(t) = F0·exp(r·t), r={cfg.r:.4f}',
                              line=dict(color=_C['orange'], width=1.3, dash='dash')))
    # Spread F/S em subplot
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                          row_heights=[0.7, 0.3],
                          subplot_titles=(f'{ticker} — S(t) vs F(t) (log scale)',
                                           'F/S ratio (alto = subvalorizado)'))
    fig2.add_trace(go.Scatter(x=s.index, y=s.values, name='S(t)',
                                line=dict(color=_C['accent'], width=1.3)),
                    row=1, col=1)
    fig2.add_trace(go.Scatter(x=s.index, y=F, name='F(t)',
                                line=dict(color=_C['orange'], width=1.3, dash='dash')),
                    row=1, col=1)
    fig2.update_yaxes(type='log', row=1, col=1)
    fs = F / s.values
    fig2.add_trace(go.Scatter(x=s.index, y=fs, name='F/S',
                                line=dict(color=_C['yellow'], width=1),
                                fill='tozeroy', fillcolor='rgba(210,153,34,0.15)',
                                showlegend=False),
                    row=2, col=1)
    fig2.add_hline(y=1, line_color=_C['text_muted'], line_width=0.6, line_dash='dot',
                    row=2, col=1)
    fig2.update_layout(**{**_FIG_LAYOUT, 'height': 560})
    return fig2


def fig_passive_breaks_monte_carlo(paths: np.ndarray, horizon_years: float,
                                      title_suffix: str = '') -> go.Figure:
    """Plot 100 simulated paths Figure 3/4 do paper."""
    n_paths, n_steps_plus = paths.shape
    t_years = np.linspace(0, horizon_years, n_steps_plus)
    fig = go.Figure()
    # Plot all paths em cinza transparente
    max_show = min(n_paths, 100)
    for i in range(max_show):
        fig.add_trace(go.Scatter(x=t_years, y=paths[i],
                                    line=dict(color='rgba(0,200,255,0.15)', width=0.6),
                                    showlegend=False, hoverinfo='skip'))
    # Median + percentiles
    median = np.median(paths, axis=0)
    p10 = np.percentile(paths, 10, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    fig.add_trace(go.Scatter(x=t_years, y=p90, name='p90',
                              line=dict(color=_C['red'], width=1.4, dash='dash')))
    fig.add_trace(go.Scatter(x=t_years, y=median, name='mediana',
                              line=dict(color=_C['orange'], width=2)))
    fig.add_trace(go.Scatter(x=t_years, y=p10, name='p10',
                              line=dict(color=_C['green'], width=1.4, dash='dash')))
    fig.update_layout(
        title=f'Monte Carlo — {n_paths} paths, horizonte {horizon_years:.0f}y {title_suffix}',
        xaxis_title='anos', yaxis_title='S(t), S(0)=100',
        **{**_FIG_LAYOUT, 'height': 500})
    return fig


def fig_mc_forward_histogram(paths: np.ndarray, horizon_years: float,
                                horizon_target: float = 10) -> go.Figure:
    """Histograma do valor em T=horizon_target (Figure 5/8 do paper)."""
    steps_per_year = (paths.shape[1] - 1) / horizon_years
    idx = int(horizon_target * steps_per_year)
    if idx >= paths.shape[1]:
        idx = paths.shape[1] - 1
    vals = paths[:, idx]
    median = np.median(vals)
    q10, q90 = np.percentile(vals, [10, 90])
    n_neg = int((vals < 100).sum())
    n_collapsed = int((vals < 50).sum())
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=vals, nbinsx=40,
                                 marker_color=_C['accent'],
                                 marker_line_color=_C['border'],
                                 opacity=0.85, showlegend=False))
    fig.add_vline(x=100, line_color='#fff', line_dash='dash', line_width=1.2,
                   annotation_text='S0=100')
    fig.add_vline(x=median, line_color=_C['orange'], line_dash='dot',
                   annotation_text=f'median {median:.0f}')
    fig.update_layout(
        title=f'Distribuicao S(T={horizon_target:.0f}y) — '
              f'{n_neg}/{paths.shape[0]} abaixo de S0, '
              f'{n_collapsed} quase colapso',
        xaxis_title='Nivel do indice', yaxis_title='Frequencia',
        **{**_FIG_LAYOUT, 'height': 460})
    return fig


def fig_critical_vs_r(cfg: PassiveBreaksConfig) -> go.Figure:
    """
    Figure 6 do paper: thresholds criticos (Lyapunov e Feller) como funcao de r.
    Aqui modelamos recalibracao: assume que para cada r, kappa e sigma
    mudam ligeiramente pra manter E[S(T)]=S(T) com F(T)=S(T).

    No paper, thresholds sao concavos em r porque a recalibracao captura
    efeitos compostos. Pra simplificar, mantemos kappa/sigma constantes
    e plotamos os thresholds (que nao mudam com r isoladamente) — o grafico
    e uma visualizacao da faixa r in [8%, 12%] anualizado.
    """
    r_range = np.linspace(0.05, 0.12, 15)
    # Nesse modelo simplificado, thresholds sao funcao apenas de kappa/sigma.
    # Plot mostra valor constante com anotacao do r calibrado.
    thr = critical_thresholds(cfg.kappa, cfg.sigma)
    lyap_pct = thr['lyapunov_threshold'] * 100
    feller_pct = thr['feller_threshold'] * 100
    r_labels = [f'{r*100:.1f}%' for r in r_range]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=r_labels, y=[lyap_pct] * len(r_range),
                          name=f'Lyapunov ({lyap_pct:.1f}%)',
                          marker_color=_C['yellow'], opacity=0.7))
    fig.add_trace(go.Bar(x=r_labels, y=[feller_pct] * len(r_range),
                          name=f'Feller ({feller_pct:.1f}%)',
                          marker_color=_C['red'], opacity=0.7))
    # Highlight da barra do r calibrado (usando index, nao string, pra add_vline)
    r_idx = int(np.argmin(np.abs(r_range - cfg.r)))
    fig.add_vline(x=r_idx, line_color='#fff', line_width=2.5,
                   annotation_text=f'r={cfg.r*100:.2f}%',
                   annotation_font_color='#fff')
    fig.update_layout(
        title=f'Thresholds criticos (Figura 6) | kappa={cfg.kappa}, sigma={cfg.sigma}',
        xaxis_title='r (growth de F, %)', yaxis_title='threshold passive share %',
        barmode='group', yaxis_range=[0, 100],
        **{**_FIG_LAYOUT, 'height': 460})
    return fig


def run_passive_breaks_section(close: pd.Series, ticker: str = 'SPX Index',
                                 horizon_years: float = 20,
                                 n_paths: int = 100,
                                 use_paper_defaults: bool = True) -> dict:
    """
    Pipeline completo da secao Passive Breaks.

    use_paper_defaults=True (RECOMENDADO): usa kappa/sigma/r fixos do paper
      (calibrados em 1926-1994 — era sem contaminacao de passive).
      Nao depende de tamanho do historico — 2y ja basta pra plotar S vs F.

    use_paper_defaults=False: auto-calibra nos dados atuais (risco de
      bias por passive — so use se entender as implicacoes).
    """
    cfg = calibrate_passive_breaks(close, use_paper_defaults=use_paper_defaults)
    state = compute_passive_breaks_state(close, cfg)

    # Simula dois cenarios: (a) no passive share, (b) logistic passive share
    S0 = 100.0; F0 = 100.0
    paths_no_p = simulate_passive_breaks(
        S0, F0, cfg, n_paths=n_paths, horizon_years=horizon_years,
        p_fn=lambda t: 0.0, include_haddad=False)
    paths_logistic = simulate_passive_breaks(
        S0, F0, cfg, n_paths=n_paths, horizon_years=horizon_years,
        p_fn=None, include_haddad=False)
    paths_haddad = simulate_passive_breaks(
        S0, F0, cfg, n_paths=n_paths, horizon_years=horizon_years,
        p_fn=None, include_haddad=True)

    # Projecao temporal ate thresholds (5 cenarios)
    thr = critical_thresholds(cfg.kappa, cfg.sigma)
    proj_lyap = project_threshold_hit(cfg, thr['lyapunov_threshold'])
    proj_feller = project_threshold_hit(cfg, thr['feller_threshold'])

    return {
        'config': cfg,
        'state': state,
        'projection_lyapunov': proj_lyap,
        'projection_feller': proj_feller,
        'paths_no_passive': paths_no_p,
        'paths_logistic': paths_logistic,
        'paths_haddad': paths_haddad,
        'horizon_years': horizon_years,
        'figs': {
            'threshold_projection': fig_threshold_projection(cfg),
            'passive_curve': fig_passive_share_curve(cfg),
            'state_gauge': fig_passive_state_gauge(state),
            's_vs_f': fig_s_vs_f(close, cfg, ticker),
            'mc_no_passive': fig_passive_breaks_monte_carlo(
                paths_no_p, horizon_years, '(p=0 baseline)'),
            'mc_logistic': fig_passive_breaks_monte_carlo(
                paths_logistic, horizon_years, '(p logistica, SEM Haddad)'),
            'mc_haddad': fig_passive_breaks_monte_carlo(
                paths_haddad, horizon_years, '(p logistica + Haddad ajuste)'),
            'hist_10y': fig_mc_forward_histogram(paths_logistic, horizon_years, 10),
            'hist_20y': fig_mc_forward_histogram(paths_logistic, horizon_years,
                                                    min(20, horizon_years)),
            'critical_vs_r': fig_critical_vs_r(cfg),
        }
    }


def _passive_opinion_html(state: dict, proj_lyap: pd.DataFrame = None,
                             proj_feller: pd.DataFrame = None) -> str:
    """
    Card de INTERPRETACAO — em linguagem direta:
    - O que cada limite significa
    - Quanto falta (pp + anos + USD tn)
    - O que esperar em cada zona
    - Onde estamos e pra onde vamos
    """
    p = state['p_raw_pct']
    lyap = state['lyapunov_threshold_pct']
    feller = state['feller_threshold_pct']
    dist_lyap = state['distance_to_lyapunov_pct']
    dist_feller = state['distance_to_feller_pct']

    # Anos base ate os limites
    yrs_lyap_base = None; yrs_feller_base = None
    aum_lyap = None; aum_feller = None
    if proj_lyap is not None and len(proj_lyap) > 0:
        base = proj_lyap[proj_lyap['cenario'].str.contains('Base')]
        if len(base) > 0:
            yrs_lyap_base = base.iloc[0]['years_ate_hit']
            aum_lyap = base.iloc[0]['aum_adicional_usd_tn']
    if proj_feller is not None and len(proj_feller) > 0:
        base = proj_feller[proj_feller['cenario'].str.contains('Base')]
        if len(base) > 0:
            yrs_feller_base = base.iloc[0]['years_ate_hit']
            aum_feller = base.iloc[0]['aum_adicional_usd_tn']

    zone_emoji = '🟢' if p < lyap else ('🟡' if p < feller else '🔴')

    # Explicacao contextual por zona
    if p < lyap:
        interp = f"""
            <b>Mercado ainda funcional.</b> O passive share ({p:.2f}%) esta
            <b>{dist_lyap:.1f} pp abaixo</b> do limiar de Lyapunov. Active managers
            ainda conseguem puxar os precos de volta ao valor fundamental F(t) apos
            choques — mean reversion opera. Volatilidade mean-reverta pra nivel
            estavel V_stable ≈ {state.get('V_stable_pct', 'N/A')}%.
            <br><br>
            <b>O que vigiar:</b> ritmo do passive share. Ao ritmo atual (α calibrado
            pelo Haddad), chegamos no <b>Lyapunov em ~{yrs_lyap_base:.1f} anos</b>
            (+{aum_lyap:.1f} trilhoes USD em AUM passivo incremental).
        """ if yrs_lyap_base else ""
    elif p < feller:
        interp = f"""
            <b>Zona de transicao (LYAPUNOV).</b> Acima de {lyap:.1f}%, o paper prova
            matematicamente que a volatilidade comeca a crescer em <b>cubic speed</b> —
            ou seja, dV ~ V^3. Isso significa que shocks pequenos se amplificam
            exponencialmente. Mean reversion k*(1-p) esta fraco demais pra controlar.
            <br><br>
            <b>O que esperar:</b> caudas mais gordas, periodos longos com F/S
            sustentadamente elevado (subvalorizacao estrutural), drawdowns mais
            profundos e recuperacoes mais lentas. Vol target e vol control sistematicos
            viram foguete amplificador.
        """
    else:
        interp = f"""
            <b>Zona FELLER — colapso possivel.</b> Acima de {feller:.1f}%, a condicao
            2·k·(1-p) ≥ σ² e quebrada. Matematicamente existe probabilidade estritamente
            positiva de S(t) → 0 em tempo finito. Na pratica, circuit breakers e
            intervencao de banco central impedem o colapso real, mas a volatilidade
            realizada fica impossivel de controlar.
            <br><br>
            <b>Implicacao:</b> o premio de risco equity some, liquidez dos indices
            principais vira questao sistemica, e a engine que sustenta o passive
            (retorno consistente) deixa de existir.
        """

    lyap_projection_html = f"""
          <div style='margin:14px 0 6px 0;'>
            <span style='color:#d29922; font-weight:700; letter-spacing:2px;
                         font-size:12px;'>🟡 LYAPUNOV ({lyap:.1f}%)</span>
          </div>
          <div style='color:#cce8ff; font-size:13px; line-height:1.6'>
            Falta <b>{dist_lyap:.1f}pp</b> de passive share. No ritmo atual do α:
            <b>~{yrs_lyap_base:.1f} anos</b> ({int(yrs_lyap_base * 365.25)} dias uteis)
            e <b>+{aum_lyap:.1f} trilhoes USD</b> de inflow passivo adicional.
            Quando cruzar: volatilidade comeca a crescer em velocidade cubica.
          </div>
    """ if yrs_lyap_base else ""

    feller_projection_html = f"""
          <div style='margin:14px 0 6px 0;'>
            <span style='color:#f85149; font-weight:700; letter-spacing:2px;
                         font-size:12px;'>🔴 FELLER ({feller:.1f}%)</span>
          </div>
          <div style='color:#cce8ff; font-size:13px; line-height:1.6'>
            Falta <b>{dist_feller:.1f}pp</b>. No ritmo atual:
            <b>~{yrs_feller_base:.1f} anos</b> ({int(yrs_feller_base * 365.25)} dias uteis)
            e <b>+{aum_feller:.1f} trilhoes USD</b>. Quando cruzar: probabilidade positiva
            de colapso em tempo finito. O paper chama de "critical systems analysis".
          </div>
    """ if yrs_feller_base else ""

    return f"""
    <div class='mm-dash'>
      <div class='mm-card' style='padding:18px 22px; border-left:3px solid {state['zone_color']};'>
        <div style='font-size:11px; color:#8b949e; letter-spacing:2px;
                    text-transform:uppercase; margin-bottom:10px;'>
          {zone_emoji} Interpretacao — o que os limites significam
        </div>

        <div style='color:#cce8ff; font-size:13px; line-height:1.65; margin-bottom:12px;'>
          {interp}
        </div>

        <div style='border-top:1px solid rgba(0,200,255,0.15); padding-top:12px;'>
          <div style='font-size:11px; color:#8b949e; letter-spacing:2px;
                      text-transform:uppercase; margin-bottom:6px;'>
            Quanto falta pra cada limite (cenario base, α=0.106)
          </div>
          {lyap_projection_html}
          {feller_projection_html}
        </div>

        <div style='margin-top:12px; font-size:10px; color:#8b949e; font-style:italic;
                    line-height:1.5;'>
          Premissa AUM: mkt cap US equity = $60tn. A projecao usa logistica calibrada em
          Haddad 1995-2019 (α=0.106). Cenarios ±1σ/±2σ no grafico assumem 15% de incerteza
          no α — valores baixos ({dist_lyap > 0 and '-' or '+'}) sao mais lentos, altos
          sao mais rapidos. Se active managers desertarem (virarem seguidores de tendencia),
          α real pode ser maior que o calibrado (+σ ou +2σ).
        </div>
      </div>
    </div>
    """


def _passive_state_card_html(state: dict) -> str:
    """Card HUD com o estado atual — pra por no topo do relatorio."""
    p = state['p_raw_pct']; lyap = state['lyapunov_threshold_pct']
    feller = state['feller_threshold_pct']
    dist_lyap = state['distance_to_lyapunov_pct']
    dist_feller = state['distance_to_feller_pct']
    zone = state['zone']
    zone_color = state['zone_color']
    return f"""
    <div class='mm-dash'>
      <div class='mm-card' style='padding:18px 22px'>
        <div style='font-size:11px; color:#8b949e; letter-spacing:2px;
                    text-transform:uppercase; margin-bottom:6px;'>
          Passive Breaks Model — Green/Krishnan/Sturm SSRN 2025
        </div>
        <div style='display:flex; gap:20px; flex-wrap:wrap;'>
          <div>
            <div class='mm-metric-lbl'>Passive Share atual (ano {state['current_year']})</div>
            <div style='font-size:28px; color:{zone_color}; font-weight:700;'>
              {p:.2f}%
            </div>
          </div>
          <div>
            <div class='mm-metric-lbl'>Zona</div>
            <div style='font-size:16px; color:{zone_color}; font-weight:700;
                        padding-top:6px;'>{zone}</div>
          </div>
          <div>
            <div class='mm-metric-lbl'>Lyapunov threshold</div>
            <div style='font-size:20px; color:{_C["yellow"]}; font-weight:700;'>
              {lyap:.1f}% (dist: {dist_lyap:+.1f}pp)
            </div>
          </div>
          <div>
            <div class='mm-metric-lbl'>Feller threshold</div>
            <div style='font-size:20px; color:{_C["red"]}; font-weight:700;'>
              {feller:.1f}% (dist: {dist_feller:+.1f}pp)
            </div>
          </div>
        </div>
        <div style='margin-top:12px; display:flex; gap:18px; flex-wrap:wrap;
                    font-size:11px; color:#8b949e;'>
          <span>kappa = {state['kappa']}</span>
          <span>sigma = {state['sigma']}</span>
          <span>r = {state['r']}</span>
          <span>alpha = {state['alpha']}</span>
          <span>t0 = {state['t0']}</span>
          <span>F/S = {state.get('F_over_S', 'N/A')}</span>
          <span>V_stable = {state.get('V_stable_pct', 'N/A')}%</span>
        </div>
      </div>
    </div>
    """


# =============================================================================
# 7e. REGIME DETECTION ENGINE
# =============================================================================
#
# Filosofia: 3 semanas (15 dias uteis) no mesmo modus operandi = PATTERN.
# Identificar cedo = vantagem. Alem disso:
#   - 45+ dias = ENTRENCHED (risco de reversao alto)
#   - transicoes abruptas sao os momentos mais caros
#
# 5 dimensoes de regime:
#   1. Trend (posicao relativa as MAs)
#   2. Volatility (VIX ou RV percentile)
#   3. Session (RTH vs ETH dominance rolling 20d)
#   4. Direction (sinal do retorno 20d)
#   5. Vol Direction (ATR subindo ou descendo)
# =============================================================================

REGIME_DURATION_THRESHOLDS = {
    'forming': 0,       # < 15d
    'confirmed': 15,    # 15-44d (3 semanas)
    'entrenched': 45,   # >= 45d
}

REGIME_COLORS = {
    # trend
    'uptrend': '#3fb950', 'downtrend': '#f85149', 'sideways': '#8b949e',
    # vol
    'low_vol': '#3fb950', 'medium_vol': '#d29922', 'high_vol': '#f85149',
    # session
    'rth_led': '#58a6ff', 'eth_led': '#f0883e', 'balanced': '#8b949e',
    # direction
    'up_month': '#3fb950', 'down_month': '#f85149', 'flat_month': '#8b949e',
    # vol direction
    'vol_rising': '#f85149', 'vol_falling': '#3fb950', 'vol_stable': '#8b949e',
    # status
    'forming': '#d29922', 'confirmed': '#58a6ff', 'entrenched': '#bc8cff',
}


def _compute_state_duration(series: pd.Series) -> pd.Series:
    """Dias consecutivos no estado atual."""
    result = []
    current = None
    count = 0
    for val in series:
        if val == current:
            count += 1
        else:
            current = val
            count = 1
        result.append(count)
    return pd.Series(result, index=series.index)


def _duration_status(d) -> str:
    if pd.isna(d):
        return 'na'
    d = int(d)
    if d < 15:
        return 'forming'
    if d < 45:
        return 'confirmed'
    return 'entrenched'


def compute_regimes(df: pd.DataFrame, vol_indices: pd.DataFrame = None) -> pd.DataFrame:
    """
    Calcula series temporais de regime em 5 dimensoes + duracao + status.

    Retorna DataFrame com:
      trend_regime, vol_regime, session_regime, direction_regime, voldir_regime
      + {dim}_duration (dias no estado)
      + {dim}_status (forming/confirmed/entrenched)
    """
    out = pd.DataFrame(index=df.index)
    close = df['close']

    # --- 1. Trend ---
    ma20 = close.rolling(20, min_periods=10).mean()
    ma50 = close.rolling(50, min_periods=20).mean()
    trend = np.where((close > ma20) & (ma20 > ma50), 'uptrend',
             np.where((close < ma20) & (ma20 < ma50), 'downtrend', 'sideways'))
    out['trend_regime'] = trend

    # --- 2. Volatility (VIX pctile se disponivel, senao RV) ---
    if (vol_indices is not None and len(vol_indices) > 0
          and 'vix' in vol_indices.columns):
        vix = vol_indices['vix'].reindex(df.index).ffill()
        vol_pct = vix.rolling(252, min_periods=60).apply(
            lambda x: x.rank(pct=True).iloc[-1], raw=False)
    else:
        rv = df['close'].pct_change().rolling(21, min_periods=10).std()
        vol_pct = rv.rolling(252, min_periods=60).apply(
            lambda x: x.rank(pct=True).iloc[-1], raw=False)
    vol_regime = np.where(vol_pct < 0.3, 'low_vol',
                  np.where(vol_pct > 0.7, 'high_vol', 'medium_vol'))
    out['vol_regime'] = vol_regime

    # --- 3. Session dominance (RTH vs ETH cum 20d) ---
    if 'rth_return' in df.columns and 'eth_return' in df.columns:
        rth_cum = df['rth_return'].rolling(20, min_periods=10).sum()
        eth_cum = df['eth_return'].rolling(20, min_periods=10).sum()
        diff = rth_cum - eth_cum
        session = np.where(diff > 0.01, 'rth_led',
                    np.where(diff < -0.01, 'eth_led', 'balanced'))
    else:
        session = ['balanced'] * len(df)
    out['session_regime'] = session

    # --- 4. Direction (retorno 20d) ---
    ret20 = close.pct_change(20)
    direction = np.where(ret20 > 0.02, 'up_month',
                  np.where(ret20 < -0.02, 'down_month', 'flat_month'))
    out['direction_regime'] = direction

    # --- 5. Vol Direction (ATR subindo/caindo vs MA60 de ATR) ---
    if 'true_range' in df.columns:
        tr = df['true_range']
    else:
        prev_c = df['close'].shift(1)
        tr = pd.concat([df['high'] - df['low'],
                         (df['high'] - prev_c).abs(),
                         (df['low'] - prev_c).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=5).mean()
    atr60 = tr.rolling(60, min_periods=20).mean()
    vd_ratio = atr14 / atr60
    voldir = np.where(vd_ratio > 1.1, 'vol_rising',
              np.where(vd_ratio < 0.9, 'vol_falling', 'vol_stable'))
    out['voldir_regime'] = voldir

    # --- Duracao + status ---
    for dim in ['trend', 'vol', 'session', 'direction', 'voldir']:
        col = f'{dim}_regime'
        out[f'{dim}_duration'] = _compute_state_duration(out[col])
        out[f'{dim}_status'] = out[f'{dim}_duration'].apply(_duration_status)

    return out


def regime_snapshot(regime_df: pd.DataFrame) -> dict:
    """Estado atual (ultima linha) de todas as dimensoes."""
    if len(regime_df) == 0:
        return {}
    last = regime_df.iloc[-1]
    dims = ['trend', 'vol', 'session', 'direction', 'voldir']
    out = {}
    for d in dims:
        out[d] = {
            'state': last[f'{d}_regime'],
            'duration_days': int(last[f'{d}_duration']),
            'status': last[f'{d}_status'],
            'color': REGIME_COLORS.get(last[f'{d}_regime'], _C['text_muted']),
        }
    return out


def similar_historical_regimes(df: pd.DataFrame, regime_df: pd.DataFrame,
                                  n_forward: int = 20) -> dict:
    """
    Acha instancias historicas com o MESMO combo de regimes do estado atual
    (trend + vol + direction + voldir — session intencionalmente omitido pq
    tem menos amostra). Retorna:
      - numero de matches historicos
      - forward 20d: mean, median, hit rate, best, worst
      - matches list (datas onde combinou)
    """
    if len(regime_df) < n_forward + 50:
        return {'n_matches': 0, 'error': 'historia curta'}

    last = regime_df.iloc[-1]
    combo_cols = ['trend_regime', 'vol_regime', 'direction_regime', 'voldir_regime']
    current = {c: last[c] for c in combo_cols}

    mask = pd.Series(True, index=regime_df.index)
    for c, v in current.items():
        mask &= (regime_df[c] == v)
    # Exclui ultimos n_forward dias (pra ter dados forward)
    mask.iloc[-n_forward:] = False
    match_dates = regime_df.index[mask]

    if len(match_dates) < 3:
        return {
            'n_matches': len(match_dates),
            'current_combo': current,
            'note': 'poucas ocorrencias historicas (< 3)',
        }

    # Forward returns pra cada match
    close = df['close']
    forward_rets = []
    for dt in match_dates:
        try:
            i = close.index.get_loc(dt)
        except KeyError:
            continue
        if i + n_forward < len(close):
            fwd = (close.iloc[i + n_forward] / close.iloc[i] - 1) * 100
            forward_rets.append(fwd)

    forward_rets = pd.Series(forward_rets)
    return {
        'n_matches': len(forward_rets),
        'current_combo': current,
        'fwd_n_days': n_forward,
        'fwd_mean_pct': round(forward_rets.mean(), 2),
        'fwd_median_pct': round(forward_rets.median(), 2),
        'fwd_std_pct': round(forward_rets.std(), 2),
        'fwd_hit_rate_pct': round((forward_rets > 0).sum() / len(forward_rets) * 100, 2),
        'fwd_best_pct': round(forward_rets.max(), 2),
        'fwd_worst_pct': round(forward_rets.min(), 2),
        'forward_returns_series': forward_rets,
    }


def regime_duration_stats(regime_df: pd.DataFrame) -> pd.DataFrame:
    """
    Quantos dias cada estado costuma durar historicamente.
    Util pra saber: se ja estou 40 dias em uptrend, esta esticado vs historico?
    """
    rows = []
    dims = ['trend', 'vol', 'session', 'direction', 'voldir']
    for dim in dims:
        col = f'{dim}_regime'
        dur = f'{dim}_duration'
        # Duracoes de cada bloco continuo
        states = regime_df[col]
        durations = regime_df[dur]
        # Acha o ultimo dia de cada bloco (onde o estado vai mudar)
        change = states != states.shift(-1)
        block_ends = durations[change]
        block_states = states[change]
        for state in states.dropna().unique():
            mask = block_states == state
            if mask.sum() < 2:
                continue
            dd = block_ends[mask]
            rows.append({
                'dimension': dim,
                'state': state,
                'n_episodes': int(mask.sum()),
                'avg_duration_days': round(dd.mean(), 1),
                'median_duration_days': int(dd.median()),
                'max_duration_days': int(dd.max()),
                'p90_duration_days': int(dd.quantile(0.9)),
            })
    return pd.DataFrame(rows)


def fig_regime_timeline(regime_df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    5 faixas empilhadas mostrando o regime ao longo do tempo (cor por estado).
    Visual rapido de quando cada dimensao mudou.
    """
    if len(regime_df) == 0:
        return go.Figure().update_layout(title='Regime timeline — sem dados',
                                           **_FIG_LAYOUT)
    dims = [('trend', 'Trend'), ('vol', 'Volatility'),
            ('session', 'Session'), ('direction', 'Direction'),
            ('voldir', 'Vol Direction')]

    fig = make_subplots(rows=len(dims), cols=1, shared_xaxes=True,
                         vertical_spacing=0.03,
                         subplot_titles=[d[1] for d in dims])

    for i, (dim, label) in enumerate(dims):
        col = f'{dim}_regime'
        states = regime_df[col].unique()
        for state in states:
            if pd.isna(state): continue
            mask = regime_df[col] == state
            # Series binaria 1 onde estado=state, senao nan
            y_vals = np.where(mask, 1, np.nan)
            fig.add_trace(go.Scatter(
                x=regime_df.index, y=y_vals,
                mode='lines', name=f'{label} · {state}',
                line=dict(color=REGIME_COLORS.get(state, _C['text_muted']),
                          width=12),
                hovertemplate=f'<b>{label}: {state}</b><br>%{{x|%Y-%m-%d}}<extra></extra>',
                showlegend=(i == 0)),
                row=i + 1, col=1)
        fig.update_yaxes(visible=False, range=[0.5, 1.5], row=i + 1, col=1)

    fig.update_layout(
        title=f'{ticker} — Regime Timeline (5 dimensoes, cor por estado)',
        **{**_FIG_LAYOUT, 'height': 620,
           'legend': dict(orientation='h', yanchor='bottom', y=-0.12,
                           xanchor='center', x=0.5, font=dict(size=9))})
    return fig


def fig_regime_duration_dist(regime_df: pd.DataFrame, ticker: str) -> go.Figure:
    """Distribuicao historica da duracao dos regimes (box plot por dimensao)."""
    if len(regime_df) == 0:
        return go.Figure().update_layout(title='Duration — sem dados', **_FIG_LAYOUT)
    dims = [('trend', 'Trend'), ('vol', 'Vol'), ('session', 'Session'),
            ('direction', 'Direction'), ('voldir', 'VolDir')]
    fig = go.Figure()
    for dim, label in dims:
        col = f'{dim}_regime'
        dur = f'{dim}_duration'
        states = regime_df[col].unique()
        change = regime_df[col] != regime_df[col].shift(-1)
        block_ends = regime_df.loc[change, dur]
        block_states = regime_df.loc[change, col]
        for state in states:
            if pd.isna(state): continue
            vals = block_ends[block_states == state].values
            if len(vals) < 2: continue
            fig.add_trace(go.Box(
                y=vals, name=f'{label}·{state}',
                marker_color=REGIME_COLORS.get(state, _C['text_muted']),
                boxmean=True, showlegend=False))
    fig.add_hline(y=15, line_color=_C['yellow'], line_dash='dash',
                   annotation_text='15d = pattern confirmado')
    fig.add_hline(y=45, line_color=_C['purple'], line_dash='dot',
                   annotation_text='45d = entrenched')
    fig.update_layout(
        title=f'{ticker} — Distribuicao historica de duracao por regime',
        yaxis_title='dias no estado',
        **{**_FIG_LAYOUT, 'height': 520,
           'margin': dict(l=60, r=40, t=55, b=120)})
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))
    return fig


def fig_regime_forward_histogram(similar_data: dict, ticker: str) -> go.Figure:
    """Histograma dos forward returns quando o combo atual aconteceu antes."""
    if similar_data.get('n_matches', 0) < 3:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Poucas ocorrencias historicas<br>({similar_data.get('n_matches', 0)} matches)",
            showarrow=False, font=dict(size=16, color=_C['text_muted']),
            x=0.5, y=0.5, xref='paper', yref='paper')
        fig.update_layout(title='Forward Returns — sem amostra suficiente',
                           **_FIG_LAYOUT)
        return fig

    rets = similar_data['forward_returns_series']
    mean = similar_data['fwd_mean_pct']
    median = similar_data['fwd_median_pct']
    hit = similar_data['fwd_hit_rate_pct']

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rets, nbinsx=30,
        marker_color=_C['accent'], marker_line_color=_C['border'],
        opacity=0.85, showlegend=False))
    fig.add_vline(x=0, line_color=_C['text_muted'], line_dash='dash')
    fig.add_vline(x=mean, line_color=_C['orange'], line_dash='dot',
                   annotation_text=f'mean {mean:+.2f}%')
    fig.add_vline(x=median, line_color=_C['green'], line_dash='dot',
                   annotation_text=f'median {median:+.2f}%')
    fig.update_layout(
        title=f'{ticker} — Forward {similar_data["fwd_n_days"]}d apos combo atual '
              f'({similar_data["n_matches"]} matches historicos, hit {hit:.1f}%)',
        xaxis_title='retorno %', yaxis_title='frequencia',
        **_FIG_LAYOUT)
    return fig


def _regime_card_html(snap: dict, similar: dict = None) -> str:
    """Card HUD com o regime atual em cada dimensao + status."""
    STATUS_EMOJI = {'forming': '🟡', 'confirmed': '🔵', 'entrenched': '🟣',
                      'na': '⚪'}
    STATE_LABELS = {
        'uptrend': 'UPTREND', 'downtrend': 'DOWNTREND', 'sideways': 'SIDEWAYS',
        'low_vol': 'LOW VOL', 'medium_vol': 'MED VOL', 'high_vol': 'HIGH VOL',
        'rth_led': 'RTH-LED', 'eth_led': 'ETH-LED', 'balanced': 'BALANCED',
        'up_month': 'UP MONTH', 'down_month': 'DOWN MONTH', 'flat_month': 'FLAT MONTH',
        'vol_rising': 'VOL ↑', 'vol_falling': 'VOL ↓', 'vol_stable': 'VOL →',
    }
    dim_labels = {
        'trend': 'TREND', 'vol': 'VOLATILITY', 'session': 'SESSION',
        'direction': 'DIRECTION', 'voldir': 'VOL DIR',
    }

    rows_html = ''
    for dim in ['trend', 'vol', 'session', 'direction', 'voldir']:
        d = snap.get(dim, {})
        state = d.get('state', 'na')
        duration = d.get('duration_days', 0)
        status = d.get('status', 'na')
        color = d.get('color', _C['text_muted'])
        rows_html += f"""
        <tr>
          <td style='padding:8px 14px; color:#8b949e; font-size:10px;
                     letter-spacing:2px;'>{dim_labels[dim]}</td>
          <td style='padding:8px 14px; color:{color}; font-weight:700;
                     font-size:16px;'>{STATE_LABELS.get(state, state.upper())}</td>
          <td style='padding:8px 14px; color:#cce8ff; font-size:14px;
                     font-weight:600;'>{duration} dias</td>
          <td style='padding:8px 14px; color:{REGIME_COLORS.get(status, _C["text_muted"])};
                     font-size:11px; font-weight:700; letter-spacing:1.5px;'>
                     {STATUS_EMOJI.get(status, "")} {status.upper()}</td>
        </tr>
        """

    similar_html = ''
    if similar and similar.get('n_matches', 0) >= 3:
        color = _C['green'] if similar['fwd_mean_pct'] > 0 else _C['red']
        similar_html = f"""
        <div style='margin-top:16px; padding-top:14px;
                    border-top:1px solid rgba(0,200,255,0.2);'>
          <div style='font-size:10px; color:#8b949e; letter-spacing:2px;
                      text-transform:uppercase; margin-bottom:8px;'>
            Historico — combo atual apareceu {similar['n_matches']} vezes
          </div>
          <div style='display:flex; gap:20px; flex-wrap:wrap;'>
            <div>
              <div class='mm-metric-lbl'>Forward {similar['fwd_n_days']}d mean</div>
              <div style='font-size:20px; color:{color}; font-weight:700;'>
                {similar['fwd_mean_pct']:+.2f}%
              </div>
            </div>
            <div>
              <div class='mm-metric-lbl'>Hit rate</div>
              <div style='font-size:20px; color:#cce8ff; font-weight:700;'>
                {similar['fwd_hit_rate_pct']:.1f}%
              </div>
            </div>
            <div>
              <div class='mm-metric-lbl'>Melhor / pior</div>
              <div style='font-size:14px; color:#cce8ff; font-weight:700; padding-top:4px;'>
                {similar['fwd_best_pct']:+.2f}% / {similar['fwd_worst_pct']:+.2f}%
              </div>
            </div>
            <div>
              <div class='mm-metric-lbl'>Std dev</div>
              <div style='font-size:14px; color:#cce8ff; font-weight:700; padding-top:4px;'>
                {similar['fwd_std_pct']:.2f}%
              </div>
            </div>
          </div>
        </div>
        """
    elif similar:
        similar_html = f"""
        <div style='margin-top:12px; color:{_C['text_muted']}; font-size:11px; font-style:italic;'>
          {similar.get('note', 'Sem amostra historica suficiente pra este combo especifico.')}
        </div>
        """

    return f"""
    <div class='mm-dash'>
      <div class='mm-card' style='padding:18px 22px;'>
        <div style='font-size:11px; color:#8b949e; letter-spacing:2px;
                    text-transform:uppercase; margin-bottom:12px;'>
          🔍 Regime Detection — 3 semanas no mesmo estado = pattern confirmado
        </div>
        <table style='width:100%; border-collapse:collapse;'>
          {rows_html}
        </table>
        {similar_html}
      </div>
    </div>
    """


def detect_recent_patterns(df: pd.DataFrame,
                              min_hit_rate: float = 0.70,
                              min_n: int = 3) -> pd.DataFrame:
    """
    Pattern miner: detecta regularidades recentes no retorno por weekday.

    Exemplos: "ultimo 4w: segunda sobe 4/4 (100%)", "desde 1-mar: quinta cai 5/6 (83%)".

    Scanneia varias janelas:
      - last_4w (20 dias uteis)
      - last_8w (40)
      - last_12w (60)
      - last_6m (126)
      - since_month_start (desde dia 1 do mes corrente)
      - since_year_start (YTD)
      - last_30d_cal (30 dias calendario)

    Retorna DataFrame ordenado por forca do padrao (hit_rate * sqrt(n)).
    Filtra: hit_rate >= min_hit_rate e n >= min_n.
    """
    if 'rth_return' not in df.columns:
        return pd.DataFrame()
    last_date = df.index[-1]

    def _range_since(year, month, day):
        try:
            start = pd.Timestamp(year=year, month=month, day=day)
            if df.index.tz is not None:
                start = start.tz_localize(df.index.tz)
            return df[df.index >= start]
        except Exception:
            return df

    windows = {
        'last_4w (20d)': df.tail(20),
        'last_8w (40d)': df.tail(40),
        'last_12w (60d)': df.tail(60),
        'last_6m (126d)': df.tail(126),
        'since_month_start': _range_since(last_date.year, last_date.month, 1),
        'since_year_start': _range_since(last_date.year, 1, 1),
    }

    rows = []
    for win_name, sub in windows.items():
        if len(sub) < min_n:
            continue
        for wd in WEEKDAY_ORDER:
            r = sub[sub['weekday_name'] == wd]['rth_return']
            n = len(r)
            if n < min_n:
                continue
            up = (r > 0).sum()
            down = (r < 0).sum()
            hit_up = up / n
            hit_down = down / n
            mean_ret = r.mean() * 100

            # Padrao de ALTA
            if hit_up >= min_hit_rate:
                strength = hit_up * math.sqrt(n)
                rows.append({
                    'window': win_name, 'weekday': wd, 'pattern': 'UP',
                    'hit_rate_pct': round(hit_up * 100, 1),
                    'n_occurrences': n, 'n_positive': int(up),
                    'mean_return_pct': round(mean_ret, 2),
                    'strength': round(strength, 2),
                })
            # Padrao de BAIXA
            if hit_down >= min_hit_rate:
                strength = hit_down * math.sqrt(n)
                rows.append({
                    'window': win_name, 'weekday': wd, 'pattern': 'DOWN',
                    'hit_rate_pct': round(hit_down * 100, 1),
                    'n_occurrences': n, 'n_positive': int(up),
                    'mean_return_pct': round(mean_ret, 2),
                    'strength': round(strength, 2),
                })

    return pd.DataFrame(rows).sort_values('strength', ascending=False).reset_index(drop=True)


def _patterns_card_html(patterns: pd.DataFrame) -> str:
    """Card HUD destacando os padroes mais fortes."""
    if len(patterns) == 0:
        return (
            "<div class='mm-dash'><div class='mm-card'>"
            "<p class='mm-metric-lbl'>🔥 Pattern Miner</p>"
            "<p style='color:#8b949e;'>Nenhum padrao forte (hit>=70%) detectado nas "
            "janelas 4w/8w/12w/6m/mes/YTD.</p></div></div>")

    # Agrupa por janela pra card organizado
    cards = ''
    seen_windows = []
    for win in patterns['window'].unique():
        if len(seen_windows) >= 5: break
        sub = patterns[patterns['window'] == win].head(5)
        if len(sub) == 0: continue
        seen_windows.append(win)
        rows_html = ''
        for _, row in sub.iterrows():
            emoji = '🟢' if row['pattern'] == 'UP' else '🔴'
            color = _C['green'] if row['pattern'] == 'UP' else _C['red']
            rows_html += f"""
            <div style='margin:5px 0; font-size:13px; color:#cce8ff;'>
              {emoji} <b style='color:{color}'>{row['weekday'].upper()}</b>:
              {row['pattern']} &nbsp;
              <span style='color:#fff;'>{int(row['n_positive']) if row['pattern']=='UP' else (row['n_occurrences']-int(row['n_positive']))}/{int(row['n_occurrences'])}</span>
              <span style='color:#8b949e;'>({row['hit_rate_pct']}%,
              mean {row['mean_return_pct']:+.2f}%)</span>
            </div>
            """
        cards += f"""
          <div style='margin-bottom:14px;'>
            <div style='color:#d29922; font-size:11px; letter-spacing:2px;
                        text-transform:uppercase; margin-bottom:6px;'>
              📅 {win}
            </div>
            {rows_html}
          </div>
        """

    return f"""
    <div class='mm-dash'>
      <div class='mm-card' style='padding:18px 22px;'>
        <div style='font-size:11px; color:#8b949e; letter-spacing:2px;
                    text-transform:uppercase; margin-bottom:12px;'>
          🔥 Pattern Miner — weekday patterns com hit &gt;= 70%
        </div>
        {cards}
        <div style='margin-top:10px; font-size:10px; color:#8b949e; font-style:italic;'>
          Ordenado por <b>strength</b> (hit_rate × √n). Padroes com n pequeno sao
          coincidencias; padroes com n &gt;= 6 ja sao relevantes. Ainda nao
          significa edge — so que o padrao EXISTE no recorte.
        </div>
      </div>
    </div>
    """


def fig_pattern_weekday_recent(df: pd.DataFrame, n_days: int = 40,
                                  ticker: str = '') -> go.Figure:
    """Grid dia-a-dia das ultimas N sessoes, colorido por weekday + RTH return."""
    sub = df.tail(n_days).copy()
    if len(sub) == 0:
        return go.Figure().update_layout(title='Pattern — sem dados', **_FIG_LAYOUT)
    sub['week'] = sub.index.isocalendar().week
    sub['weekday_num'] = sub.index.weekday  # 0=Mon
    sub['ret_pct'] = sub['rth_return'] * 100

    # Pivot week × weekday
    pv = sub.pivot_table(index='week', columns='weekday_num', values='ret_pct',
                          aggfunc='last')
    # Reordena index (semanas recentes em cima)
    pv = pv.sort_index(ascending=False)
    # Cria text com data + %
    dates_pv = sub.pivot_table(index='week', columns='weekday_num',
                                 values='rth_return',
                                 aggfunc=lambda x: x.index.strftime('%d/%m')[-1] if len(x) else '')
    dates_pv = dates_pv.reindex(pv.index)

    vmax = np.nanpercentile(np.abs(pv.values[~np.isnan(pv.values)]), 95) \
           if np.any(~np.isnan(pv.values)) else 1
    text = [[f'{d}<br>{v:+.2f}%' if pd.notna(v) else ''
             for d, v in zip(drow, vrow)]
            for drow, vrow in zip(dates_pv.values, pv.values)]

    fig = go.Figure(go.Heatmap(
        z=pv.values, x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
        y=[f'W{int(w)}' for w in pv.index],
        colorscale=[[0, _C['red']], [0.5, '#0d1117'], [1, _C['green']]],
        zmin=-vmax, zmax=vmax,
        text=text, texttemplate='%{text}',
        textfont=dict(size=9, color='#cce8ff'),
        colorbar=dict(title='RTH %')))
    fig.update_layout(
        title=f'{ticker} — RTH diario ultimas {n_days} sessoes (grid semana x weekday)',
        **{**_FIG_LAYOUT, 'height': max(400, len(pv) * 22 + 100)})
    return fig


def run_regime_section(df: pd.DataFrame, ticker: str,
                         vol_indices: pd.DataFrame = None) -> dict:
    """Pipeline completo: computa regimes + snapshots + similar + figs + pattern miner."""
    regime_df = compute_regimes(df, vol_indices)
    snapshot = regime_snapshot(regime_df)
    similar = similar_historical_regimes(df, regime_df, n_forward=20)
    duration_stats = regime_duration_stats(regime_df)
    patterns = detect_recent_patterns(df, min_hit_rate=0.70, min_n=3)

    return {
        'regime_df': regime_df,
        'snapshot': snapshot,
        'similar': similar,
        'duration_stats': duration_stats,
        'patterns': patterns,
        'figs': {
            'timeline': fig_regime_timeline(regime_df, ticker),
            'duration_dist': fig_regime_duration_dist(regime_df, ticker),
            'forward_hist': fig_regime_forward_histogram(similar, ticker),
            'pattern_grid_40d': fig_pattern_weekday_recent(df, n_days=40, ticker=ticker),
            'pattern_grid_60d': fig_pattern_weekday_recent(df, n_days=60, ticker=ticker),
        }
    }


# =============================================================================
# 8. ORQUESTRADOR + WIDGETS + EXPORT ZIP
# =============================================================================

def compute_session_stats(ticker: str, years: int = 5,
                            include_nomura: bool = True,
                            nomura_ticker: str = 'SPY US Equity',
                            include_gs_factors: bool = False,
                            include_passive_breaks: bool = False,
                            pb_years: int = 30,
                            pb_ticker: str = None,
                            benchmark_ticker: str = 'SPY US Equity') -> dict:
    """
    Computa TUDO e retorna dict com tabelas + figs + metrics.
    Isso e o que sera importado pelo greeks_dashboard no futuro.
    """
    daily = load_daily(ticker, years)
    df = build_session_frame(daily)
    df = add_gap_features(df)
    df = compute_moving_averages(df)
    df = compute_volatility(df)

    tables = {
        'session_frame': df,
        'weekday_stats_rth': weekday_stats(df, col='rth_return'),
        'weekday_stats_eth': weekday_stats(df, col='eth_return'),
        'weekday_stats': weekday_stats(df, col='rth_return'),  # alias default = RTH
        'updown_by_weekday': up_down_by_weekday(df),
        'ma_residency': ma_residency_stats(df),
        'streaks': streak_stats(df),
        'conditional_streaks': conditional_after_streaks(df),
        'regime': regime_stats(df),
        'gap_stats': gap_stats(df),
        'monthly_stats': monthly_stats(df),
    }
    bt = backtest_rth(df)
    bt_eth = backtest_eth(df)
    tables['equity_curve_rth'] = bt.equity.to_frame('equity')
    tables['drawdown_curve_rth'] = bt.drawdown.to_frame('drawdown')
    tables['equity_curve_eth'] = bt_eth.equity.to_frame('equity')
    tables['drawdown_curve_eth'] = bt_eth.drawdown.to_frame('drawdown')
    tables['rth_vs_eth_summary'] = rth_vs_eth_summary(bt, bt_eth, df)
    bottom_line = rth_eth_bottom_line(bt, bt_eth, initial=10000.0)

    # Regime detection — calcula sempre (default on)
    try:
        regime = run_regime_section(df, ticker, vol_indices=None)
    except Exception as e:
        log.warning(f'Regime detection falhou: {e}')
        regime = {}

    # GS-style streak analysis: usa o streak atual como target
    up_flags = (daily['close'].pct_change() > 0).astype(int)
    cur_up_streak = 0
    for v in reversed(up_flags.values):
        if v == 1: cur_up_streak += 1
        else: break
    streak_tgt = max(cur_up_streak, 3)   # se menor que 3, usa 3 como default
    tables['consecutive_days_streak'] = consecutive_days_streak_table(daily, streak_tgt)

    figs = {
        'weekday_bars_rth': fig_weekday_bars(tables['weekday_stats_rth'], f'{ticker} RTH'),
        'weekday_bars_eth': fig_eth_weekday_bars(tables['weekday_stats_eth'], ticker),
        'weekday_hitrate': fig_weekday_hitrate(tables['weekday_stats_rth'], ticker),
        'updown_weekday': fig_updown_weekday(tables['updown_by_weekday'], ticker),
        'equity_dd_rth': fig_equity_dd(bt, f'{ticker} RTH'),
        'equity_dd_eth': fig_equity_dd(bt_eth, f'{ticker} ETH'),
        'rth_eth_cumret': fig_rth_eth_cumret(bt, bt_eth, ticker),
        'rth_eth_simple': fig_rth_eth_simple(bt, bt_eth, ticker),
        'rth_vs_eth': fig_rth_vs_eth_equity(bt, bt_eth, ticker),
        'rth_vs_eth_scatter': fig_rth_vs_eth_scatter(df, ticker),
        'histogram': fig_histogram(df, ticker),
        'histogram_eth': fig_histogram_eth(df, ticker),
        'histogram_rth_vs_eth': fig_histogram_rth_vs_eth(df, ticker),
        'ma_residency': fig_ma_residency(tables['ma_residency'], ticker),
        'heatmap_wkd_month': fig_heatmap_wkd_month(df, ticker),
        'streak_distribution': fig_streak_distribution(df, ticker),
        'streaks': fig_streaks(tables['streaks'], ticker),
        'conditional_streaks': fig_conditional_after_streaks(
            tables['conditional_streaks'], ticker),
        'regime': fig_regime(tables['regime'], ticker),
        'gap': fig_gap(tables['gap_stats'], ticker),
        'monthly': fig_monthly(tables['monthly_stats'], ticker),
        # GS-style charts
        'consecutive_days_bar': fig_consecutive_days_bar(
            tables['consecutive_days_streak'], ticker, streak_tgt),
        'seasonality_by_year': fig_seasonality_by_year(daily, ticker),
        'rolling_sharpe': fig_rolling_sharpe(daily, ticker),
        'rsi': fig_rsi(daily, ticker),
        'rolling_return_5d': fig_rolling_return_pctile(daily, ticker, window=5),
        'rolling_return_20d': fig_rolling_return_pctile(daily, ticker, window=20),
    }

    nomura = {}
    if include_nomura:
        try:
            nd = load_daily(nomura_ticker, years)
            vol = load_vol_indices(years)
            iv = approximate_25d_ivs(vol['vix'], vol['skew'])

            # Roda 2 tenores: 0DTE (T=1d) e 30d monthly
            pnl_0dte = compute_daily_options_pnl(nd, iv, tenor_days=1)
            pnl_30d = compute_daily_options_pnl(nd, iv, tenor_days=21)  # 21 uteis ~ 30 cal
            summary_0dte = options_pnl_summary(pnl_0dte)
            sharpe_0dte = options_sharpe(pnl_0dte)
            summary_30d = options_pnl_summary(pnl_30d)
            sharpe_30d = options_sharpe(pnl_30d)

            sp = skew_percentiles_multi(iv)       # 3 convencoes
            flows = compute_dynamic_flows(nd)
            nomura = {
                # Default pnl refere-se ao 0DTE (compat com CSVs antigos)
                'pnl_daily': pnl_0dte, 'pnl_summary': summary_0dte, 'sharpe': sharpe_0dte,
                # Nova divisao tenor
                'pnl_0dte_daily': pnl_0dte, 'pnl_0dte_summary': summary_0dte,
                'sharpe_0dte': sharpe_0dte,
                'pnl_30d_daily': pnl_30d, 'pnl_30d_summary': summary_30d,
                'sharpe_30d': sharpe_30d,
                'skew_pctiles': sp, 'flows': flows, 'vol_indices': vol,
                'figs': {}
            }
            # Constroi figs individualmente pra falha em uma nao mata as outras
            nomura_fig_builders = [
                ('options_pnl_0dte', lambda: fig_options_pnl_heatmap(
                    summary_0dte, sharpe_0dte, tenor_label='0DTE (T=1d, daily rolado)')),
                ('options_pnl_30d', lambda: fig_options_pnl_heatmap(
                    summary_30d, sharpe_30d, tenor_label='30d Monthly (T=21d, mensal)')),
                ('strategies_equity_0dte', lambda: fig_nomura_strategies_equity(
                    pnl_0dte, tenor_label='0DTE (T=1d)')),
                ('strategies_equity_30d', lambda: fig_nomura_strategies_equity(
                    pnl_30d, tenor_label='30d Monthly (T=21d)')),
                ('strategies_rolling_0dte', lambda: fig_nomura_strategies_rolling(
                    pnl_0dte, window=60, tenor_label='0DTE (T=1d)')),
                ('strategies_rolling_30d', lambda: fig_nomura_strategies_rolling(
                    pnl_30d, window=60, tenor_label='30d Monthly (T=21d)')),
                ('skew_pctiles', lambda: fig_skew_multi(sp)),
                ('iv_rank', lambda: fig_iv_rank(vol)),
                ('flows', lambda: fig_systematic_flows(flows)),
                ('vol_panic', lambda: fig_vol_panic_proxy(daily, vol, ticker)),
            ]
            for fig_name, builder in nomura_fig_builders:
                try:
                    nomura['figs'][fig_name] = builder()
                except Exception as fig_e:
                    log.warning(f'[nomura] fig {fig_name} falhou: {fig_e}')
                    import traceback
                    log.warning(traceback.format_exc())
            log.info(f'[nomura] OK ({len(nomura["figs"])} figs geradas)')
        except Exception as e:
            import traceback
            err_trace = traceback.format_exc()
            log.warning(f'Nomura section falhou: {e}')
            log.warning(err_trace)
            nomura = {'error': str(e), 'traceback': err_trace}

    gs_factors = {}
    if include_gs_factors:
        try:
            gs_factors = run_gs_factor_section(
                years=max(2, min(years, 5)),
                benchmark_ticker=benchmark_ticker,
                session_frame=df,
                regime_df=regime.get('regime_df') if regime else None)
            if not gs_factors:
                gs_factors = {'error': 'run_gs_factor_section retornou {} (BQL pode ter falhado pra todos os tickers)'}
        except Exception as e:
            import traceback
            err_trace = traceback.format_exc()
            log.warning(f'GS Factor section falhou: {e}')
            log.warning(err_trace)
            gs_factors = {'error': str(e), 'traceback': err_trace}

    passive_breaks = {}
    if include_passive_breaks:
        try:
            # Passive Breaks usa janela propria — tipicamente 25y pra ver o
            # S vs F de longa duracao (paper usa 1926-1994 pra calibracao, mas
            # pra visualizacao atual 25-30y do ticker ja e o suficiente).
            pb_tk = pb_ticker if pb_ticker else ticker
            log.info(f'[passive_breaks] carregando {pb_years}y de {pb_tk}...')
            if pb_tk == ticker and pb_years <= years:
                pb_daily = daily  # reusa dado ja carregado
            else:
                pb_daily = load_daily(pb_tk, pb_years)
            if pb_daily is None or len(pb_daily) < 50:
                raise ValueError(
                    f'Historico muito curto: {len(pb_daily) if pb_daily is not None else 0} dias. '
                    f'Precisa de 50+ dias pra visualizacao. Aumente "PB Years" ou troque de ticker.')
            log.info(f'[passive_breaks] simulando com {len(pb_daily)} dias de {pb_tk}...')
            passive_breaks = run_passive_breaks_section(
                pb_daily['close'], ticker=pb_tk, horizon_years=20, n_paths=100,
                use_paper_defaults=True)
            passive_breaks['window_info'] = {
                'ticker': pb_tk, 'years': pb_years, 'n_days': len(pb_daily),
                'start': pb_daily.index[0].strftime('%Y-%m-%d'),
                'end': pb_daily.index[-1].strftime('%Y-%m-%d'),
            }
            log.info('[passive_breaks] OK')
        except Exception as e:
            import traceback
            err_trace = traceback.format_exc()
            log.warning(f'Passive Breaks falhou: {e}')
            log.warning(err_trace)
            passive_breaks = {'error': str(e), 'traceback': err_trace}

    return {
        'ticker': ticker, 'years': years,
        'ts': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'bt': bt, 'bt_eth': bt_eth,
        'bottom_line': bottom_line,
        'regime': regime,
        'tables': tables, 'figs': figs, 'nomura': nomura,
        'gs_factors': gs_factors,
        'passive_breaks': passive_breaks,
    }


def _df_to_html_table(df, max_rows=None) -> str:
    """Converte DataFrame pra HTML no estilo mm-table."""
    if df is None or len(df) == 0:
        return "<div class='mm-card'>(sem dados)</div>"
    d = df.head(max_rows) if max_rows else df
    # float_format como callable (string quebra em alguns pandas quando
    # coluna e object com numpy floats / np.nan misturado)
    try:
        html = d.to_html(classes='mm-table', border=0, index=True,
                          float_format=lambda x: f'{x:.2f}' if pd.notna(x) else '')
    except Exception:
        # fallback: sem float_format
        html = d.to_html(classes='mm-table', border=0, index=True, na_rep='')
    return f"<div class='mm-card' style='overflow-x:auto'>{html}</div>"


def _metrics_html(metrics: dict) -> str:
    """Header metrics estilo mm-metric."""
    items = []
    for k, v in metrics.items():
        cls = 'mm-metric-val'
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if 'return' in k.lower() or 'cagr' in k.lower():
                cls += ' up' if v > 0 else ' down' if v < 0 else ''
        items.append(f"<span class='mm-metric'><span class='mm-metric-lbl'>{k}</span>"
                      f"<span class='{cls}'>{v}</span></span>")
    return f"<div class='mm-card'>{''.join(items)}</div>"


# Snapshot global — usado pelo botao Export ZIP
_snapshot = {}


def _shorten_tab_title(full_title: str) -> str:
    """'Parte I — Quant Session Stats' -> 'I · Quant'."""
    import re
    m = re.match(r'Parte\s+([IVX]+)\s*[—-]\s*(.+)', full_title, re.I)
    if m:
        roman = m.group(1)
        rest = m.group(2).strip()
        # Mapeamento direto pra evitar duplicacao
        mapping = {
            'Quant Session Stats': 'Quant',
            'Regime Detection + Pattern Miner': 'Regime+Pattern',
            'Regime Detection': 'Regime',
            'Structural Market Models': 'Structural',
            'Cross-Sectional Factor Monitor': 'Factors',
            'Nomura Options Framework': 'Nomura',
        }
        short = mapping.get(rest)
        if short is None:
            # Fallback: primeira palavra
            short = rest.split()[0] if rest.split() else rest[:15]
        return f'{roman} · {short}'
    return full_title[:20]


def _split_sections_into_tabs(sections: list) -> 'wd.Tab':
    """
    Divide a lista plana de widgets em ABAS usando os _big_divider como
    marcadores. Cada divider inicia uma nova aba. Tudo antes do primeiro
    divider vai pra aba 'Overview'.
    """
    import re
    tabs_data = []
    current_title = '📊 Overview'
    current_widgets = []

    for w in sections:
        is_divider = False
        if isinstance(w, wd.HTML):
            html = w.value
            if 'mm-divider' in html and 'mm-divider-title' in html:
                is_divider = True

        if is_divider:
            # Salva aba corrente
            if current_widgets:
                tabs_data.append((current_title, current_widgets))
            # Extrai titulo do divider
            m = re.search(r'◆ ◆ ◆\s*&nbsp;\s*(.*?)\s*&nbsp;\s*◆ ◆ ◆', w.value)
            if m:
                current_title = _shorten_tab_title(m.group(1))
            else:
                current_title = 'Extra'
            current_widgets = []
            continue  # divider nao vai pra aba

        current_widgets.append(w)

    if current_widgets:
        tabs_data.append((current_title, current_widgets))

    if not tabs_data:
        tabs_data = [('Empty', [wd.HTML("<div class='mm-card'>Sem dados</div>")])]

    tabs = wd.Tab()
    tabs.children = [wd.VBox(ws, layout=wd.Layout(overflow='auto',
                                                       max_height='85vh'))
                      for _, ws in tabs_data]
    for i, (title, _) in enumerate(tabs_data):
        tabs.set_title(i, title)
    # Aba inicial = primeira
    tabs.selected_index = 0
    return tabs


def build_section_widgets(result: dict) -> list:
    """
    Converte o resultado de compute_session_stats em lista de widgets/HTML
    prontos pra display. Reusavel quando greeks_dashboard plugar como tab.
    """
    sec = []
    ticker = result['ticker']
    sec.append(wd.HTML(f"<div class='mm-section-label'>Session Stats — {ticker}"
                         f" ({result['years']}y) — {result['ts']}</div>"))

    # --- BOTTOM LINE bem no topo (direto ao ponto: $10k virou quanto?) ---
    if result.get('bottom_line'):
        sec.append(wd.HTML(_bottom_line_html(result['bottom_line'], ticker)))
    # Retorno cumulativo % — grafico simples, direto ao ponto
    sec.append(go.FigureWidget(result['figs']['rth_eth_cumret']))
    # Em dolares absolutos
    sec.append(go.FigureWidget(result['figs']['rth_eth_simple']))

    # Header com metricas RTH e ETH lado a lado
    rth_m = {f'RTH {k}': v for k, v in result['bt'].metrics.items()}
    eth_m = {f'ETH {k}': v for k, v in result['bt_eth'].metrics.items()}
    # So os principais no header pra nao poluir
    key_metrics = ['sharpe', 'cagr_pct', 'max_drawdown_pct', 'hit_rate_pct', 'n_days']
    header = {}
    for k in key_metrics:
        header[f'RTH {k}'] = result['bt'].metrics.get(k, np.nan)
    for k in key_metrics:
        header[f'ETH {k}'] = result['bt_eth'].metrics.get(k, np.nan)
    sec.append(wd.HTML(_metrics_html(header)))

    # Legenda unica — deixa claro % vs $ vs ratio
    sec.append(wd.HTML(
        "<div class='mm-note'>"
        "<b>Legenda unidades:</b> "
        "colunas terminadas em <b>_pct</b> = percentual (%) &nbsp;|&nbsp; "
        "<b>_bn</b> = USD bilhoes &nbsp;|&nbsp; "
        "<b>ratio/signed/rank</b> = escalar puro &nbsp;|&nbsp; "
        "<b>n</b> = numero de observacoes. "
        "Todos os valores com 2 casas decimais."
        "</div>"))

    # ====== PART I: QUANT SESSION STATS ======
    sec.append(wd.HTML(_big_divider(
        'Parte I — Quant Session Stats',
        f'Estatistica intradiaria de {ticker} | RTH vs ETH | Weekday | Sequencias | Regime | Gap | Sazonalidade | Backtest')))

    # --- RTH (Regular Trading Hours) ---
    sec.append(wd.HTML("<div class='mm-section-label'>RTH — Weekday Stats (open→close, %)</div>"))
    sec.append(wd.HTML(_df_to_html_table(result['tables']['weekday_stats_rth'])))
    sec.append(go.FigureWidget(result['figs']['weekday_bars_rth']))
    sec.append(go.FigureWidget(result['figs']['weekday_hitrate']))

    # --- ETH (Extended Trading Hours) ---
    sec.append(wd.HTML("<div class='mm-section-label'>ETH — Weekday Stats "
                         "(close→open, overnight + pre + AH, %)</div>"))
    sec.append(wd.HTML(_df_to_html_table(result['tables']['weekday_stats_eth'])))
    sec.append(go.FigureWidget(result['figs']['weekday_bars_eth']))

    sec.append(wd.HTML("<div class='mm-section-label'>Subiu / Caiu por Weekday (RTH, %)</div>"))
    sec.append(wd.HTML(_df_to_html_table(result['tables']['updown_by_weekday'])))
    sec.append(go.FigureWidget(result['figs']['updown_weekday']))

    # --- Backtests ---
    sec.append(wd.HTML("<div class='mm-section-label'>Comparacao RTH vs ETH — Equity, DD, Rolling 60d</div>"))
    sec.append(go.FigureWidget(result['figs']['rth_vs_eth']))

    sec.append(wd.HTML("<div class='mm-section-label'>Tabela comparativa RTH vs ETH (metricas lado a lado)</div>"))
    sec.append(wd.HTML(_df_to_html_table(result['tables']['rth_vs_eth_summary'])))

    sec.append(wd.HTML("<div class='mm-section-label'>Scatter RTH x ETH dia-a-dia (correlacao + quadrantes)</div>"))
    sec.append(go.FigureWidget(result['figs']['rth_vs_eth_scatter']))

    sec.append(wd.HTML("<div class='mm-section-label'>Equity + Drawdown individual</div>"))
    sec.append(go.FigureWidget(result['figs']['equity_dd_rth']))
    sec.append(go.FigureWidget(result['figs']['equity_dd_eth']))

    sec.append(wd.HTML("<div class='mm-section-label'>Distribuicao Retornos — RTH vs ETH (%)</div>"))
    sec.append(go.FigureWidget(result['figs']['histogram_rth_vs_eth']))
    sec.append(go.FigureWidget(result['figs']['histogram']))
    sec.append(go.FigureWidget(result['figs']['histogram_eth']))

    sec.append(wd.HTML("<div class='mm-section-label'>Heatmap Weekday x Mes (%)</div>"))
    sec.append(go.FigureWidget(result['figs']['heatmap_wkd_month']))

    sec.append(wd.HTML("<div class='mm-section-label'>Permanencia Medias Moveis (%)</div>"))
    sec.append(wd.HTML(_df_to_html_table(result['tables']['ma_residency'])))
    sec.append(go.FigureWidget(result['figs']['ma_residency']))

    sec.append(wd.HTML("<div class='mm-section-label'>Sequencias (dias consecutivos)</div>"))
    sec.append(wd.HTML(_df_to_html_table(result['tables']['streaks'])))
    sec.append(go.FigureWidget(result['figs']['streaks']))
    sec.append(go.FigureWidget(result['figs']['streak_distribution']))

    sec.append(wd.HTML("<div class='mm-section-label'>Retorno RTH apos N dias seguidos</div>"))
    sec.append(wd.HTML(_df_to_html_table(result['tables']['conditional_streaks'])))
    sec.append(go.FigureWidget(result['figs']['conditional_streaks']))

    sec.append(wd.HTML("<div class='mm-section-label'>Regime de Tendencia</div>"))
    sec.append(wd.HTML(_df_to_html_table(result['tables']['regime'])))
    sec.append(go.FigureWidget(result['figs']['regime']))

    sec.append(wd.HTML("<div class='mm-section-label'>Gap Analysis (% open-to-open)</div>"))
    sec.append(wd.HTML(_df_to_html_table(result['tables']['gap_stats'])))
    sec.append(go.FigureWidget(result['figs']['gap']))

    sec.append(wd.HTML("<div class='mm-section-label'>Sazonalidade por Mes (%)</div>"))
    sec.append(wd.HTML(_df_to_html_table(result['tables']['monthly_stats'])))
    sec.append(go.FigureWidget(result['figs']['monthly']))

    # --- GS-style indicators (inspirado nos reports da Goldman Sachs) ---
    sec.append(wd.HTML(
        "<div class='mm-section-label'>GS-Style Indicators — Momentum, Sazonalidade, Vol Regime</div>"))

    sec.append(wd.HTML(
        "<div class='mm-section-label'>Forward returns apos N dias seguidos de alta "
        "(estilo Cullen Morgan / GS Vol Color)</div>"))
    sec.append(go.FigureWidget(result['figs']['consecutive_days_bar']))
    sec.append(wd.HTML(_df_to_html_table(result['tables']['consecutive_days_streak'])))

    sec.append(wd.HTML("<div class='mm-section-label'>Seasonality por Ano "
                         "(cada linha = 1 ano, ano atual em destaque)</div>"))
    sec.append(go.FigureWidget(result['figs']['seasonality_by_year']))

    sec.append(wd.HTML("<div class='mm-section-label'>Rolling Sharpe 3m / 6m / 12m</div>"))
    sec.append(go.FigureWidget(result['figs']['rolling_sharpe']))

    sec.append(wd.HTML("<div class='mm-section-label'>RSI(14) com zonas overbought/oversold</div>"))
    sec.append(go.FigureWidget(result['figs']['rsi']))

    sec.append(wd.HTML("<div class='mm-section-label'>Rolling 5d e 20d return vs "
                         "band historica (p10/p90)</div>"))
    sec.append(go.FigureWidget(result['figs']['rolling_return_5d']))
    sec.append(go.FigureWidget(result['figs']['rolling_return_20d']))

    # ====== PARTE II: REGIME DETECTION + PATTERN MINER ======
    if result.get('regime') and result['regime'].get('snapshot'):
        r_data = result['regime']
        sec.append(wd.HTML(_big_divider(
            'Parte II — Regime Detection + Pattern Miner',
            '5 dimensoes | 15d = pattern confirmado | weekday patterns em 4w/8w/12w/6m/mes/YTD | forward returns historicos')))
        sec.append(wd.HTML(_regime_card_html(r_data['snapshot'],
                                                 r_data.get('similar'))))

        # Pattern Miner — destaque, logo apos o regime card
        patterns = r_data.get('patterns')
        if patterns is not None:
            sec.append(wd.HTML(_patterns_card_html(patterns)))
            if len(patterns) > 0:
                sec.append(wd.HTML("<div class='mm-section-label'>Tabela completa "
                                     "dos padroes detectados (ordem por strength)</div>"))
                sec.append(wd.HTML(_df_to_html_table(patterns)))

        # Grid visual das ultimas sessoes
        if 'pattern_grid_40d' in r_data['figs']:
            sec.append(wd.HTML("<div class='mm-section-label'>Grid visual — "
                                 "ultimas 40 sessoes por semana × weekday</div>"))
            sec.append(go.FigureWidget(r_data['figs']['pattern_grid_40d']))
        if 'pattern_grid_60d' in r_data['figs']:
            sec.append(wd.HTML("<div class='mm-section-label'>Grid visual — "
                                 "ultimas 60 sessoes por semana × weekday</div>"))
            sec.append(go.FigureWidget(r_data['figs']['pattern_grid_60d']))

        sec.append(wd.HTML("<div class='mm-section-label'>Timeline — quando "
                             "cada dimensao mudou</div>"))
        sec.append(go.FigureWidget(r_data['figs']['timeline']))
        sec.append(wd.HTML("<div class='mm-section-label'>Forward 20d apos combo "
                             "atual (historicamente)</div>"))
        sec.append(go.FigureWidget(r_data['figs']['forward_hist']))
        sec.append(wd.HTML("<div class='mm-section-label'>Duracao tipica de cada "
                             "regime (box plot)</div>"))
        sec.append(go.FigureWidget(r_data['figs']['duration_dist']))
        if r_data.get('duration_stats') is not None and len(r_data['duration_stats']) > 0:
            sec.append(wd.HTML("<div class='mm-section-label'>Tabela — duracao "
                                 "media/mediana/max por regime</div>"))
            sec.append(wd.HTML(_df_to_html_table(r_data['duration_stats'])))

    # ====== PART III: STRUCTURAL MODELS (Passive Breaks) ======
    pb_data = result.get('passive_breaks') or {}
    if pb_data.get('error') or pb_data.get('state'):
        sec.append(wd.HTML(_big_divider(
            'Parte III — Structural Market Models',
            'Passive Breaks Model (Green/Krishnan/Sturm SSRN 2025) | nao e quant trading, e analise estrutural')))

    if pb_data.get('error'):
        sec.append(wd.HTML(
            f"<div class='mm-section-label'>Passive Breaks Model — ERRO</div>"
            f"<div class='mm-card'>"
            f"<p class='mm-flag'>❌ Falhou: {pb_data['error']}</p>"
            f"<pre style='color:#8b949e;font-size:10px;max-height:200px;overflow:auto'>"
            f"{pb_data.get('traceback', '')}</pre>"
            f"</div>"))
    elif pb_data.get('state'):
        pb = pb_data
        # Info da janela usada
        if pb.get('window_info'):
            wi = pb['window_info']
            sec.append(wd.HTML(
                f"<div class='mm-note'>"
                f"Janela Passive Breaks: <b>{wi['ticker']}</b> "
                f"({wi['years']}y, {wi['n_days']} dias, "
                f"{wi['start']} → {wi['end']}) | "
                f"parametros <b>fixos do paper (1926-1994)</b>: "
                f"kappa=0.0909, sigma=0.1247, r=0.0917"
                f"</div>"))
        sec.append(wd.HTML(
            "<div class='mm-section-label'>Passive Breaks the Market "
            "(Green/Krishnan/Sturm SSRN 2025)</div>"))
        sec.append(wd.HTML(_passive_state_card_html(pb['state'])))
        sec.append(wd.HTML(_passive_opinion_html(pb['state'],
                                                    pb.get('projection_lyapunov'),
                                                    pb.get('projection_feller'))))
        sec.append(go.FigureWidget(pb['figs']['state_gauge']))
        sec.append(go.FigureWidget(pb['figs']['passive_curve']))

        # Projecao temporal
        sec.append(wd.HTML(
            "<div class='mm-section-label'>Projecao temporal — "
            "quando chegamos em cada threshold? (±1σ / ±2σ do α)</div>"))
        sec.append(go.FigureWidget(pb['figs']['threshold_projection']))
        if pb.get('projection_lyapunov') is not None and len(pb['projection_lyapunov']) > 0:
            sec.append(wd.HTML(f"<div class='mm-section-label'>"
                                 f"Ate o Lyapunov ({pb['state']['lyapunov_threshold_pct']}%)</div>"))
            sec.append(wd.HTML(_df_to_html_table(pb['projection_lyapunov'])))
        if pb.get('projection_feller') is not None and len(pb['projection_feller']) > 0:
            sec.append(wd.HTML(f"<div class='mm-section-label'>"
                                 f"Ate o Feller ({pb['state']['feller_threshold_pct']}%)</div>"))
            sec.append(wd.HTML(_df_to_html_table(pb['projection_feller'])))

        sec.append(go.FigureWidget(pb['figs']['s_vs_f']))
        sec.append(wd.HTML("<div class='mm-section-label'>Monte Carlo — 3 cenarios</div>"))
        sec.append(go.FigureWidget(pb['figs']['mc_no_passive']))
        sec.append(go.FigureWidget(pb['figs']['mc_logistic']))
        sec.append(go.FigureWidget(pb['figs']['mc_haddad']))
        sec.append(wd.HTML("<div class='mm-section-label'>Distribuicao forward</div>"))
        sec.append(go.FigureWidget(pb['figs']['hist_10y']))
        sec.append(go.FigureWidget(pb['figs']['hist_20y']))
        sec.append(go.FigureWidget(pb['figs']['critical_vs_r']))
        # Tabela do state
        state_df = pd.DataFrame([pb['state']]).T
        state_df.columns = ['value']
        sec.append(wd.HTML("<div class='mm-section-label'>State snapshot (valores)</div>"))
        sec.append(wd.HTML(_df_to_html_table(state_df)))

    # ====== PART IV: CROSS-SECTIONAL MONITOR (GS Factor Monitor) ======
    # Sempre mostra o divider; conteudo varia se data presente ou nao
    sec.append(wd.HTML(_big_divider(
        'Parte IV — Cross-Sectional Factor Monitor',
        'GS Barra Pair Indices + Momentum + GS Themes + BBG Themes | scan 1D/5D/1M/YTD/1Y | rotacao de factors')))
    gs_factors_data = result.get('gs_factors') or {}
    if gs_factors_data.get('error'):
        tb = gs_factors_data.get('traceback', '')
        sec.append(wd.HTML(
            f"<div class='mm-card'>"
            f"<p class='mm-flag'>❌ Factor Monitor falhou:</p>"
            f"<p style='color:#cce8ff; font-size:13px;'><b>{gs_factors_data['error']}</b></p>"
            f"<pre style='color:#8b949e; font-size:10px; max-height:300px; "
            f"overflow:auto; background:rgba(0,0,0,0.3); padding:10px;'>{tb}</pre>"
            f"</div>"))
    elif not gs_factors_data:
        sec.append(wd.HTML(
            "<div class='mm-card'>"
            "<p class='mm-flag'>⚠ Factor Monitor nao foi gerado.</p>"
            "<p style='color:#8b949e; font-size:11px;'>"
            "Checkbox 'Incluir GS Factor Monitor' provavelmente nao esta marcado."
            "</p></div>"))
    elif gs_factors_data.get('table') is None or len(gs_factors_data.get('table', [])) == 0:
        sec.append(wd.HTML(
            "<div class='mm-card'>"
            "<p class='mm-flag'>⚠ Factor Monitor retornou tabela vazia.</p>"
            "<p style='color:#8b949e; font-size:11px;'>"
            "Nenhum dos 98 tickers retornou dados via BQL. Verifique entitlements "
            "BQuant pra GS basket indices (GSP*, GSXU*) e BBG themes (BAIAT, BNUAT, etc)."
            "</p></div>"))
    else:
        gf = gs_factors_data
        sec.append(wd.HTML(
            "<div class='mm-section-label'>GS Factor Monitor — Barra Pair Indices + "
            "Momentum + Thematic Baskets (historico via BQL)</div>"))
        sec.append(go.FigureWidget(gf['figs']['category_avg']))
        sec.append(go.FigureWidget(gf['figs']['leaderboard_ytd']))
        sec.append(go.FigureWidget(gf['figs']['leaderboard_1m']))
        sec.append(go.FigureWidget(gf['figs']['heatmap']))
        sec.append(go.FigureWidget(gf['figs']['rolling_top']))

        # Breadth
        if 'breadth' in gf['figs']:
            sec.append(wd.HTML("<div class='mm-section-label'>Breadth do universo — "
                                 "% de factors positivos/negativos por horizonte</div>"))
            sec.append(go.FigureWidget(gf['figs']['breadth']))
            if gf.get('breadth') is not None and len(gf['breadth']) > 0:
                sec.append(wd.HTML(_df_to_html_table(gf['breadth'])))

        # RS-Ratio RRG
        if 'rs_ratio_rrg' in gf['figs']:
            sec.append(wd.HTML("<div class='mm-section-label'>RS-Ratio RRG (JdK-style) — "
                                 f"factors vs {gf.get('benchmark_ticker', 'benchmark')} | "
                                 "Leading/Improving/Lagging/Weakening</div>"))
            sec.append(go.FigureWidget(gf['figs']['rs_ratio_rrg']))

        # Weekday effect por factor
        if 'weekday_heatmap' in gf['figs']:
            sec.append(wd.HTML("<div class='mm-section-label'>Weekday Effect — "
                                 "retorno medio diario por factor × dia da semana</div>"))
            sec.append(go.FigureWidget(gf['figs']['weekday_heatmap']))
        if 'weekday_best' in gf['figs']:
            sec.append(wd.HTML("<div class='mm-section-label'>Top 20 factors com "
                                 "maior spread best-worst weekday</div>"))
            sec.append(go.FigureWidget(gf['figs']['weekday_best']))

        # Regime × Weekday
        for dim, label in [('trend', 'Trend'), ('vol', 'Volatility'),
                             ('voldir', 'Vol Direction')]:
            fkey = f'regime_wd_{dim}'
            if fkey in gf['figs']:
                sec.append(wd.HTML(f"<div class='mm-section-label'>Regime × Weekday — "
                                     f"{label} regime vs weekday (RTH %)</div>"))
                sec.append(go.FigureWidget(gf['figs'][fkey]))

        sec.append(wd.HTML("<div class='mm-section-label'>Factor Monitor table "
                             "(ordenado por YTD, %)</div>"))
        sec.append(wd.HTML(_df_to_html_table(gf['table'])))

    # ====== PARTE V: NOMURA OPTIONS FRAMEWORK ======
    # Sempre mostra divider
    sec.append(wd.HTML(_big_divider(
        'Parte V — Nomura Options Framework',
        'Daily Options PnL Summary | Skew Percentiles | Systematic Flows (AUM dinamico) | Vol Panic Proxy')))
    nomura_data = result.get('nomura') or {}
    if nomura_data.get('error'):
        tb = nomura_data.get('traceback', '')
        sec.append(wd.HTML(
            f"<div class='mm-card'>"
            f"<p class='mm-flag'>❌ Nomura falhou:</p>"
            f"<p style='color:#cce8ff; font-size:13px;'><b>{nomura_data['error']}</b></p>"
            f"<pre style='color:#8b949e; font-size:10px; max-height:300px; "
            f"overflow:auto; background:rgba(0,0,0,0.3); padding:10px;'>{tb}</pre>"
            f"</div>"))
    elif not nomura_data:
        sec.append(wd.HTML(
            "<div class='mm-card'>"
            "<p class='mm-flag'>⚠ Nomura nao foi gerado.</p>"
            "<p style='color:#8b949e; font-size:11px;'>"
            "Checkbox 'Incluir Nomura' provavelmente nao esta marcado."
            "</p></div>"))
    elif not nomura_data.get('figs'):
        sec.append(wd.HTML(
            "<div class='mm-card'>"
            "<p class='mm-flag'>⚠ Nomura inicializou mas sem figs.</p>"
            "<p style='color:#8b949e; font-size:11px;'>"
            "Todos os fig builders falharam individualmente. Confira logs.</p></div>"))
    else:
        n = nomura_data
        sec.append(wd.HTML(
            "<div class='mm-note'>"
            "<b>Tenores:</b> o paper da Nomura usa <b>0DTE</b> (daily rolled, T=1d). "
            "Adicionamos tambem a versao <b>30d monthly</b> (T=21d) pra comparacao — "
            "vol premium embutido nos strikes cresce com √T, entao as magnitudes "
            "sao muito maiores. Valores em % do spot, 2 casas."
            "</div>"))

        # --- 0DTE ---
        sec.append(wd.HTML("<div class='mm-section-label'>Options PnL — 0DTE "
                             "(T=1 dia, rolado diariamente)</div>"))
        sec.append(go.FigureWidget(n['figs']['options_pnl_0dte']))
        if 'strategies_equity_0dte' in n['figs']:
            sec.append(wd.HTML("<div class='mm-section-label'>Equity curves 0DTE — "
                                 "performance cumulativa de cada estrategia</div>"))
            sec.append(go.FigureWidget(n['figs']['strategies_equity_0dte']))
        if 'strategies_rolling_0dte' in n['figs']:
            sec.append(wd.HTML("<div class='mm-section-label'>Rolling 60d 0DTE — "
                                 "em qual regime cada estrategia performa</div>"))
            sec.append(go.FigureWidget(n['figs']['strategies_rolling_0dte']))
        sec.append(wd.HTML("<div class='mm-section-label'>Tabela 0DTE — Cumulative (%)</div>"))
        sec.append(wd.HTML(_df_to_html_table(n.get('pnl_0dte_summary', n['pnl_summary']))))
        sec.append(wd.HTML("<div class='mm-section-label'>Tabela 0DTE — Sharpe Annualized</div>"))
        sec.append(wd.HTML(_df_to_html_table(n.get('sharpe_0dte', n['sharpe']))))

        # --- 30d Monthly ---
        if 'options_pnl_30d' in n['figs']:
            sec.append(wd.HTML("<div class='mm-section-label'>Options PnL — 30d Monthly "
                                 "(T=21 dias uteis, ~1 mes)</div>"))
            sec.append(go.FigureWidget(n['figs']['options_pnl_30d']))
            if 'strategies_equity_30d' in n['figs']:
                sec.append(wd.HTML("<div class='mm-section-label'>Equity curves 30d — "
                                     "performance cumulativa de cada estrategia</div>"))
                sec.append(go.FigureWidget(n['figs']['strategies_equity_30d']))
            if 'strategies_rolling_30d' in n['figs']:
                sec.append(wd.HTML("<div class='mm-section-label'>Rolling 60d 30d — "
                                     "em qual regime cada estrategia performa</div>"))
                sec.append(go.FigureWidget(n['figs']['strategies_rolling_30d']))
            sec.append(wd.HTML("<div class='mm-section-label'>Tabela 30d — Cumulative (%)</div>"))
            sec.append(wd.HTML(_df_to_html_table(n['pnl_30d_summary'])))
            sec.append(wd.HTML("<div class='mm-section-label'>Tabela 30d — Sharpe Annualized</div>"))
            sec.append(wd.HTML(_df_to_html_table(n['sharpe_30d'])))

        sec.append(wd.HTML("<div class='mm-section-label'>Skew Percentiles — "
                             "Nomura (ratio) + SpotGamma (normalized diff %)</div>"))
        sec.append(go.FigureWidget(n['figs']['skew_pctiles']))

        sec.append(wd.HTML("<div class='mm-section-label'>IV Rank + Skew Rank "
                             "(estilo SpotGamma, 1y rolling)</div>"))
        sec.append(go.FigureWidget(n['figs']['iv_rank']))

        sec.append(wd.HTML("<div class='mm-section-label'>Vol Panic Proxy "
                             "(estilo GS Vol Color — VIX+RV+DD+|z5d|, 0-10)</div>"))
        sec.append(go.FigureWidget(n['figs']['vol_panic']))

        sec.append(wd.HTML("<div class='mm-section-label'>Systematic Flows — "
                             "VC + CTA + RP (USD bn, AUM dinamico)</div>"))
        sec.append(go.FigureWidget(n['figs']['flows']))

    sec.append(wd.HTML(
        "<div class='mm-note'>"
        "<b>Obs:</b> estatistica nao e edge automatico. Linhas com n&lt;30 "
        "tem leitura fraca. 25d IVs aproximados via SKEW index — plugar chain "
        "real (OVDV/OptionMetrics) pra precisao total. GS Factor Monitor usa "
        "apenas o nivel (px_last) — a carteira interna nao e liberada."
        "</div>"))
    return sec


def _snapshot_html() -> str:
    """HTML standalone com todas as figs + tabelas do snapshot."""
    if not _snapshot.get('result'):
        return ""
    r = _snapshot['result']
    parts = [DASH_CSS,
              f"<h1 class='mm-dash'>Session Stats — {r['ticker']} "
              f"({r['years']}y) — {r['ts']}</h1>",
              _metrics_html(r['bt'].metrics),
              "<h2 class='mm-dash'>Tabelas principais</h2>"]
    # Tabelas
    table_labels = [
        ('weekday_stats', 'Weekday Stats (%)'),
        ('updown_by_weekday', 'Subiu / Caiu por Weekday (%)'),
        ('ma_residency', 'Permanencia Medias Moveis (%)'),
        ('streaks', 'Sequencias'),
        ('conditional_streaks', 'Retorno RTH apos N dias seguidos (%)'),
        ('regime', 'Regime de Tendencia (%)'),
        ('gap_stats', 'Gap Analysis (%)'),
        ('monthly_stats', 'Sazonalidade por Mes (%)'),
    ]
    for key, label in table_labels:
        if key in r['tables']:
            parts.append(f"<h3 class='mm-dash'>{label}</h3>")
            parts.append(_df_to_html_table(r['tables'][key]))
    # Figs principais
    parts.append("<h2 class='mm-dash'>Graficos</h2>")
    for key, fig in r['figs'].items():
        parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    # Nomura
    if r.get('nomura'):
        n = r['nomura']
        parts.append("<h2 class='mm-dash'>Nomura Framework</h2>")
        parts.append("<h3 class='mm-dash'>Options PnL Summary (%)</h3>")
        parts.append(_df_to_html_table(n['pnl_summary']))
        parts.append("<h3 class='mm-dash'>Sharpe Ratio Annualized</h3>")
        parts.append(_df_to_html_table(n['sharpe']))
        for fig in n['figs'].values():
            parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
    return "\n".join(parts)


# =============================================================================
# 9. UI (widgets + callbacks)
# =============================================================================

ticker_w = wd.Text(value='SPX Index', description='Ticker:',
                    layout=wd.Layout(width='280px'))
years_w = wd.IntSlider(value=5, min=1, max=20, step=1, description='Years:',
                        layout=wd.Layout(width='300px'),
                        tooltip='Janela principal (BBG cap = 20y)')
nomura_ticker_w = wd.Text(value='SPY US Equity', description='Nomura spot:',
                            layout=wd.Layout(width='280px'))
nomura_chk_w = wd.Checkbox(value=True,
                             description='Incluir Nomura (Options PnL + Skew + Flows)',
                             layout=wd.Layout(width='400px'))
gs_factors_chk_w = wd.Checkbox(value=False,
                                  description='Incluir GS Factor Monitor (lento: ~50 BQL queries)',
                                  layout=wd.Layout(width='400px'))
benchmark_w = wd.Text(value='SPY US Equity', description='Benchmark:',
                         layout=wd.Layout(width='280px'),
                         tooltip='Benchmark pro RS-Ratio (default SPY US Equity)')
passive_breaks_chk_w = wd.Checkbox(value=False,
                                      description='Incluir Passive Breaks Model (Green/Krishnan 2025)',
                                      layout=wd.Layout(width='450px'))
pb_ticker_w = wd.Text(value='SPY US Equity', description='PB spot:',
                         layout=wd.Layout(width='280px'))
pb_years_w = wd.IntSlider(value=20, min=5, max=20, step=1,
                             description='PB Years:',
                             layout=wd.Layout(width='320px'),
                             tooltip='Janela de historia (BBG cap = 20y maximo)')

run_btn = wd.Button(description='▶ Iniciar Analise', button_style='success',
                     icon='play', layout=wd.Layout(width='180px'))
zip_btn = wd.Button(description='📦 Export ZIP', button_style='info',
                     icon='download', layout=wd.Layout(width='160px'))

out_main = wd.Output()
out_zip = wd.Output()


def _run_analysis(_):
    with out_main:
        clear_output(wait=True)
        loading = wd.HTML(DASH_CSS + "<div class='mm-dash'><div class='mm-card mm-loading'>"
                                       "⏳ Inicializando...</div></div>")
        display(loading)
        try:
            ticker = ticker_w.value.strip() or 'SPX Index'
            years = int(years_w.value)
            include_n = bool(nomura_chk_w.value)
            include_gs = bool(gs_factors_chk_w.value)
            include_pb = bool(passive_breaks_chk_w.value)
            nom_tk = nomura_ticker_w.value.strip() or 'SPY US Equity'
            pb_tk = pb_ticker_w.value.strip() or 'SPY US Equity'
            pb_yrs = int(pb_years_w.value)
            bench_tk = benchmark_w.value.strip() or 'SPY US Equity'

            loading.value = DASH_CSS + ("<div class='mm-dash'><div class='mm-card mm-loading'>"
                                           "Processando... (BQL: 1 query spot + VIX/SKEW + "
                                           f"{'~50 factors' if include_gs else 'sem factors'})"
                                           "</div></div>")

            result = compute_session_stats(ticker, years,
                                             include_nomura=include_n,
                                             nomura_ticker=nom_tk,
                                             include_gs_factors=include_gs,
                                             include_passive_breaks=include_pb,
                                             pb_years=pb_yrs,
                                             pb_ticker=pb_tk,
                                             benchmark_ticker=bench_tk)

            loading.value = DASH_CSS + f"<div class='mm-dash'><div class='mm-card mm-loading'>Montando widgets em abas...</div></div>"
            sections = build_section_widgets(result)
            tabs = _split_sections_into_tabs(sections)

            # Persiste snapshot pro Export ZIP
            _snapshot.clear()
            _snapshot['result'] = result

            clear_output(wait=True)
            display(wd.HTML(DASH_CSS))
            display(tabs)

        except Exception as e:
            import traceback
            clear_output(wait=True)
            display(wd.HTML(DASH_CSS + f"<div class='mm-dash'><div class='mm-card'>"
                              f"<p class='mm-flag'>❌ Erro: {e}</p>"
                              f"<pre style='color:#8b949e;font-size:10px'>{traceback.format_exc()}</pre>"
                              f"</div></div>"))


def _export_zip(_):
    with out_zip:
        clear_output(wait=True)
        if not _snapshot.get('result'):
            display(wd.HTML(DASH_CSS + "<div class='mm-dash'><div class='mm-card'>"
                              "<p class='mm-flag'>⚠️ Rode a analise primeiro.</p></div></div>"))
            return
        try:
            r = _snapshot['result']
            ticker = r['ticker']
            ts_safe = r['ts'].replace(':', '').replace(' ', '_').replace('-', '')
            zip_name = f"session_stats_{ticker.replace(' ', '_')}_{ts_safe}.zip"

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                # 1. metrics.json
                payload = {
                    'ticker': ticker, 'years': r['years'], 'ts': r['ts'],
                    'metrics': r['bt'].metrics,
                }
                if r.get('nomura'):
                    sp = r['nomura'].get('skew_pctiles')
                    if sp is not None:
                        payload['nomura_tails'] = {
                            'left_tail': sp.attrs.get('left_tail'),
                            'right_tail': sp.attrs.get('right_tail'),
                        }
                zf.writestr('metrics.json', json.dumps(payload, indent=2, default=str))
                # 2. session_stats.html (dashboard completo)
                zf.writestr('session_stats.html', _snapshot_html())
                # 3. CSVs
                for name, df in r['tables'].items():
                    try:
                        zf.writestr(f'data/{name}.csv', df.to_csv())
                    except Exception as ex:
                        log.warning(f'csv {name} skip: {ex}')
                if r.get('nomura'):
                    nm = r['nomura']
                    csv_list = [
                        ('options_pnl_0dte_daily', nm.get('pnl_0dte_daily', nm.get('pnl_daily'))),
                        ('options_pnl_0dte_summary', nm.get('pnl_0dte_summary', nm.get('pnl_summary'))),
                        ('options_sharpe_0dte', nm.get('sharpe_0dte', nm.get('sharpe'))),
                        ('options_pnl_30d_daily', nm.get('pnl_30d_daily')),
                        ('options_pnl_30d_summary', nm.get('pnl_30d_summary')),
                        ('options_sharpe_30d', nm.get('sharpe_30d')),
                        ('skew_pctiles', nm.get('skew_pctiles')),
                        ('systematic_flows', nm.get('flows')),
                    ]
                    for name, df in csv_list:
                        if df is None:
                            continue
                        try: zf.writestr(f'data/nomura_{name}.csv', df.to_csv())
                        except Exception as ex: log.warning(f'csv nomura {name} skip: {ex}')
                if r.get('gs_factors') and r['gs_factors']:
                    try:
                        zf.writestr('data/gs_factor_monitor.csv',
                                     r['gs_factors']['table'].to_csv(index=False))
                        zf.writestr('data/gs_factor_history.csv',
                                     r['gs_factors']['history'].to_csv())
                    except Exception as ex:
                        log.warning(f'csv gs_factors skip: {ex}')

            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode('ascii')
            js = (f"var a=document.createElement('a');"
                  f"a.href='data:application/zip;base64,{b64}';"
                  f"a.download='{zip_name}';"
                  f"document.body.appendChild(a);a.click();"
                  f"document.body.removeChild(a);")
            display(HTML(f"<script>{js}</script>"))
            display(wd.HTML(DASH_CSS + f"<div class='mm-dash'><div class='mm-card'>"
                              f"<p>✅ ZIP gerado: <b class='mm-flag'>{zip_name}</b></p>"
                              f"<p class='mm-note'>metrics.json + session_stats.html + data/*.csv</p>"
                              f"</div></div>"))
        except Exception as e:
            import traceback
            display(wd.HTML(DASH_CSS + f"<div class='mm-dash'><div class='mm-card'>"
                              f"<p class='mm-flag'>❌ Erro ZIP: {e}</p>"
                              f"<pre style='color:#8b949e;font-size:10px'>{traceback.format_exc()}</pre>"
                              f"</div></div>"))


run_btn.on_click(_run_analysis)
zip_btn.on_click(_export_zip)


def launch():
    """Entry point — monta e exibe o painel completo."""
    display(wd.HTML(DASH_CSS))
    display(wd.VBox([
        wd.HTML("<div class='mm-dash'><div class='mm-section-label'>"
                 "Session Stats — Intraday RTH Engine + Nomura Framework</div></div>"),
        wd.VBox([
            wd.HBox([ticker_w, years_w]),
            wd.HBox([nomura_ticker_w, nomura_chk_w]),
            wd.HBox([gs_factors_chk_w, benchmark_w]),
            wd.HBox([passive_breaks_chk_w, pb_ticker_w, pb_years_w]),
            wd.HBox([run_btn, zip_btn]),
        ]),
        out_zip,
        out_main,
    ]))


# Auto-launch quando o arquivo for executado via %run session_stats.py
try:
    ip = get_ipython()  # noqa — existe em Jupyter/BQuant
    launch()
except NameError:
    # Rodando via "python session_stats.py" — nao faz launch automatico
    if __name__ == '__main__':
        print('session_stats.py carregado. Use `launch()` num notebook.')
