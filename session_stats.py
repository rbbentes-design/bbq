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
</style>"""

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
    """Executa uma request de 1 field (padrao exato do greeks_dashboard.fetch_historical)."""
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
    """
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


def fig_options_pnl_heatmap(summary: pd.DataFrame, sharpe: pd.DataFrame) -> go.Figure:
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
                         subplot_titles=('SPX Daily Options PnL — Cumulative (%)',
                                          'Sharpe Ratio Annualized'))
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
    fig.update_layout(title='Nomura — SPX Daily Options PnL Summary',
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
        ('GSP1MOMO', 'Momentum (Hi/Lo)'),
        ('GSP1BETA', 'Beta (Hi/Lo)'),
        ('GSP1LEVG', 'Leverage (Hi/Lo)'),
        ('GSP1PROF', 'Profitability (Hi/Lo)'),
        ('GSP1ERNY', 'Earnings Yield (Hi/Lo)'),
        ('GSP1RSVL', 'Res Vol (Hi/Lo)'),
        ('GSP1GRWT', 'Growth (Hi/Lo)'),
        ('GSPUGRVA', 'Growth vs Value'),
        ('GSP1VALU', 'Value (Cheap/Expensive)'),
        ('GSP1SIZE', 'Size (Large/Small)'),
        ('GSP1QUAL', 'Quality (Hi/Lo)'),
    ],
    'Momentum': [
        ('GSPRHIMO', 'High Beta Momentum'),
        ('GSPRHMO6', '6m Momentum'),
        ('GSPRHMO3', '3m Momentum'),
        ('GSTMTMOM', 'TMT Momentum'),
        ('GSPUMOXX', 'Cross Sector Momo ex AI'),
        ('GSFIMOMO', 'Financials Momentum'),
        ('GSENEMOM', 'Energy Momentum'),
        ('GSINMOMO', 'Industrial Momentum'),
        ('GSHCMOMO', 'Health Care Momentum'),
        ('GSCNMOMO', 'Consumer Momentum'),
    ],
    'BBG Themes': [
        # AI / Tech
        ('BAIAT',   'AI TR'),
        ('BCLAT',   'Cloud TR'),
        ('BCYAT',   'Cybersecurity TR'),
        ('BFTAT',   'Frontier Tech TR'),
        ('BGDMTST', 'Glb DigiTech TR'),
        ('BSEMISCT', 'Glb Semicon TR'),
        ('BNEXTT',  'Next Generation TR'),
        ('BROAT',   'Robotics TR'),
        ('BAVAT',   'Autonomous Veh TR'),
        ('BM7T',    'MAGNIFICENT 7 TR'),
        ('BFAANGT', 'FAANG 2.0 TR'),
        # Fintech / Finance
        ('BPAYT',   'Digital Payments TR'),
        ('BEFAT',   'Enterp Fintech TR'),
        ('BFFAT',   'Future Fintech TR'),
        ('BDEAT',   'Decentral Eng TR'),
        # Energy / Materials / Defense
        ('BNUAT',   'Nuclear TR'),
        ('BHYAT',   'Hydrogen TR'),
        ('BSOAT',   'Solar TR'),
        ('BBFAT',   'Biofuels TR'),
        ('BGSCET',  'GS Clean Energy TR'),
        ('BCMAT',   'Circular Mat TR'),
        ('BFMAT',   'Future Materials TR'),
        ('BTMAT',   'Transition Metals TR'),
        ('BCCAET',  'CCUS EW TR'),
        ('BDST',    'Def & Sec TR'),
        ('BMDAT',   'Modern Glb Def TR'),
        ('BWAAT',   'Water TR'),
        # Health / Bio
        ('BOBAT',   'Obesity TR'),
        ('BIBFT',   'Inno Biopharma TR'),
        ('BGEAT',   'Genomics TR'),
        ('BNMAT',   'Neuro Mental Hlth TR'),
        ('BPRAT',   'Prepare & Repair TR'),
        ('BRNDT',   'R&D Leaders TR'),
        ('BPLANTT', 'Plant-B Food Scr TR'),
        # EV / Transportation
        ('BBEVT',   'Elec Vehicles TR'),
        ('BFVAT',   'Future Vehicles TR'),
        # Infra / Digital
        ('BFDAT',   'Future Dig Infra TR'),
        ('BGTAT',   'Grid Tech TR'),
        ('BDMLINFT', 'DM Listed Infra TR'),
        # Ideias / Style
        ('B5GAT',   '5G TR'),
        ('BMETAT',  'Metaverse TR'),
        ('BSPAT',   'Sports TR'),
        ('BMVPT',   'MVP TR'),
        ('BBIDAT',  'Biodiversity TR'),
        ('BBIST',   'Billionaires Inv Select TR'),
        # Factor-ish
        ('BSHARPT', 'Shareholder Yield TR'),
        ('BPPUST',  'Pricing Power TR'),
        ('B1GDT',   'US 1000 Div Growth TR'),
        ('BINFLST', 'Infl Sens TR'),
        ('BCORET',  'NC Core TR'),
        ('BANRT',   'ANR Improvers TR'),
        ('BMULTIT', 'MULTI TR'),
        ('DMCPTR',  'DM Comm Prod L/M TR'),
        # Emerging / LatAm
        ('BFREET',  'EM Freedom TR'),
        ('LATAMMET', 'LATAM M&E Cap TR'),
    ],
    'GS Themes': [
        ('GSXUMEME', 'US Memes Stocks'),
        ('GSXUNPTC', 'Non Profitable Tech'),
        ('GSXUQNTM', 'Quantum Computing'),
        ('GSXURFAV', 'Retail Favorites'),
        ('GSXUROBO', 'Robotics/Automation'),
        ('GSTHREPO', 'Buyback (US)'),
        ('GSXUDFNS', 'US Defense'),
        ('GSXUMOXL', 'High Mo ex AI & BTC'),
        ('GSXURANI', 'US Uranium'),
        ('GSXUMFML', 'Marquee Momentum'),
        ('GSXUHOME', 'US Housing'),
        ('GSXUCYCL', 'US Cyclicals'),
        ('GSXUHICN', 'High Income Consumer'),
        ('GSXUCADR', 'China ADRs'),
        ('GSXUCOND', 'US Consumer Disc'),
        ('GSXULOCN', 'Low Income Consumer'),
        ('GSXUCOMP', 'Defensive Compounder'),
        ('GSXURNEW', 'US Renewables'),
        ('GSXUMIDC', 'Mid Income Consumer'),
        ('GSXUEDEF', 'Expensive Defensives'),
        ('GSXURETL', 'US Retail'),
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
    y = [f"{r['Ticker']}  {r['Name']}" for _, r in table.iterrows()]
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
           'margin': dict(l=280, r=60, t=55, b=40)})
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
    labels = [f"{r['Ticker']} · {r['Name'][:22]}" for _, r in combined.iterrows()]
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
           'margin': dict(l=280, r=60, t=55, b=40)})
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
        fig.add_trace(go.Bar(
            x=grp.index, y=grp[col], name=col.replace('_pct', ''),
            marker_color=hz_colors.get(col, _C['accent']),
            text=[f'{v:+.2f}%' for v in grp[col]],
            textposition='outside'))
    fig.add_hline(y=0, line_color=_C['text_muted'], line_width=0.5)
    fig.update_layout(
        title='GS Factor Monitor — Media por Categoria',
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


def run_gs_factor_section(years: int = 2) -> dict:
    """Pipeline completo da secao GS Factors. Retorna dict com tabela + figs."""
    history = fetch_gs_factors(years=years)
    if len(history) == 0:
        return {}
    table = factor_monitor_table(history)
    return {
        'history': history,
        'table': table,
        'figs': {
            'heatmap': fig_factor_heatmap(table),
            'leaderboard_ytd': fig_factor_leaderboard(table, 'YTD_pct'),
            'leaderboard_1m': fig_factor_leaderboard(table, '1M_pct'),
            'category_avg': fig_factor_category_avg(table),
            'rolling_top': fig_factor_rolling(history, max_lines=8),
        }
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


def calibrate_passive_breaks(close: pd.Series, r_override: float = None) -> PassiveBreaksConfig:
    """
    Calibra kappa, sigma, r do modelo usando serie diaria de close do indice.

    Metodo do paper (eq 6): regressao com intercepto=0 de:
        dS/sqrt(F*S)  =  kappa*dt * (F-S)/sqrt(F*S) + sigma*sqrt(dt)*eps

    Assume r tal que F(T) = S(T) (boundary condition no final da janela).
    Retorna PassiveBreaksConfig calibrado.
    """
    s = close.dropna().astype(float)
    if len(s) < 500:
        log.warning('[passive_breaks] dados insuficientes pra calibracao')
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

    log.info(f'[passive_breaks] calibrado: kappa={kappa:.4f} sigma={sigma:.4f} r={r:.4f} '
              f'(T={T_years:.1f}y, n={n})')

    return PassiveBreaksConfig(kappa=kappa, sigma=sigma, r=r)


def simulate_passive_breaks(S0: float, F0: float, cfg: PassiveBreaksConfig,
                              n_paths: int = 100, horizon_years: float = 20.0,
                              steps_per_year: int = 252,
                              p_fn=None, include_haddad: bool = True,
                              seed: int = 42) -> np.ndarray:
    """
    Monte Carlo da equacao (2) do paper.

    p_fn(t_year) -> retorna p(t). Se None, usa logistica default do cfg.

    Retorna array shape (n_paths, n_steps+1).
    """
    np.random.seed(seed)
    n_steps = int(horizon_years * steps_per_year)
    dt = 1.0 / steps_per_year
    sqrt_dt = math.sqrt(dt)
    t0_year = cfg.t0 - (horizon_years / 2)  # arbitrario — vai ser mascarado

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    for i in range(n_steps):
        t = i * dt
        t_year = t0_year + t
        if p_fn is None:
            p_raw = passive_share(t_year, cfg.alpha, cfg.t0)
        else:
            p_raw = p_fn(t_year)
        p_eff = haddad_effective_share(p_raw, cfg.chi) if include_haddad else p_raw

        F_t = F0 * math.exp(cfg.r * t)
        drift = cfg.kappa * (1.0 - p_eff) * (F_t - paths[:, i]) * dt
        S_safe = np.maximum(paths[:, i], 1e-6)
        diffusion = cfg.sigma * np.sqrt(F_t * S_safe) * sqrt_dt * np.random.randn(n_paths)
        paths[:, i + 1] = np.maximum(paths[:, i] + drift + diffusion, 0.0)

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
    fig.add_trace(go.Scatter(x=years, y=p, name='Passive Share %',
                              line=dict(color=_C['accent'], width=2),
                              fill='tozeroy', fillcolor='rgba(88,166,255,0.08)'))
    # Thresholds
    fig.add_hline(y=lyap, line_color=_C['yellow'], line_dash='dash', line_width=1.2,
                   annotation_text=f'Lyapunov {lyap:.1f}%',
                   annotation_font_color=_C['yellow'])
    fig.add_hline(y=feller, line_color=_C['red'], line_dash='dash', line_width=1.2,
                   annotation_text=f'Feller {feller:.1f}%',
                   annotation_font_color=_C['red'])
    # Current point
    fig.add_trace(go.Scatter(x=[current_year], y=[p_now],
                              mode='markers+text',
                              marker=dict(color=_C['orange'], size=16,
                                          line=dict(color='#fff', width=2)),
                              text=[f'ATUAL<br>{p_now:.1f}%'],
                              textposition='top center',
                              textfont=dict(color=_C['orange'], size=11),
                              name='Atual', showlegend=False))
    # Haddad effective
    p_eff = np.array([haddad_effective_share(passive_share(y, cfg.alpha, cfg.t0), cfg.chi)
                        for y in years]) * 100
    fig.add_trace(go.Scatter(x=years, y=p_eff, name='Passive Share (Haddad-ajustada)',
                              line=dict(color=_C['purple'], width=1.2, dash='dot')))

    fig.update_layout(
        title='Passive Share Logistica — distancia ate os thresholds criticos',
        xaxis_title='Ano', yaxis_title='Passive Share %',
        yaxis_range=[0, 100],
        **{**_FIG_LAYOUT, 'height': 500})
    return fig


def fig_passive_state_gauge(state: dict) -> go.Figure:
    """Gauge/bar mostrando onde o mercado esta entre 0% e thresholds."""
    p = state['p_raw_pct']
    lyap = state['lyapunov_threshold_pct']
    feller = state['feller_threshold_pct']

    fig = go.Figure()
    # Zonas empilhadas
    fig.add_trace(go.Bar(x=[lyap], y=['Estado'], orientation='h',
                          marker_color=_C['green'], name=f'Estavel 0-{lyap:.1f}%',
                          text=f'Estavel <{lyap:.1f}%', textposition='inside'))
    fig.add_trace(go.Bar(x=[feller - lyap], y=['Estado'], orientation='h',
                          marker_color=_C['yellow'],
                          name=f'Lyapunov {lyap:.1f}-{feller:.1f}%',
                          text=f'Lyapunov', textposition='inside'))
    fig.add_trace(go.Bar(x=[100 - feller], y=['Estado'], orientation='h',
                          marker_color=_C['red'],
                          name=f'Feller >{feller:.1f}%',
                          text=f'Feller', textposition='inside'))
    # Current marker
    fig.add_vline(x=p, line_color='#fff', line_width=3,
                   annotation_text=f'ATUAL {p:.1f}%',
                   annotation_font_color='#fff', annotation_font_size=13)
    fig.update_layout(
        title=f'Passive Breaks Zone — {state["zone"]}',
        barmode='stack', xaxis=dict(range=[0, 100], title='Passive Share %'),
        yaxis=dict(showticklabels=False),
        **{**_FIG_LAYOUT, 'height': 220})
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
    Mostra quao sensivel o modelo e ao growth rate assumido.
    """
    r_range = np.linspace(0.05, 0.12, 50)
    lyaps = []; fellers = []
    for r in r_range:
        # kappa e sigma sao recalibrados se r muda (aprox): usar cfg base
        # (paper mostra que thresholds dependem so de kappa e sigma, nao de r
        # diretamente — mas simula shift da relacao)
        thr = critical_thresholds(cfg.kappa, cfg.sigma)
        lyaps.append(thr['lyapunov_threshold'] * 100)
        fellers.append(thr['feller_threshold'] * 100)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f'{r*100:.2f}%' for r in r_range], y=lyaps,
                          name='Lyapunov', marker_color=_C['yellow'], opacity=0.8))
    fig.add_trace(go.Bar(x=[f'{r*100:.2f}%' for r in r_range], y=fellers,
                          name='Feller', marker_color=_C['red'], opacity=0.8))
    fig.add_vline(x=f'{cfg.r*100:.2f}%', line_color='#fff', line_width=2,
                   annotation_text=f'calibrado r={cfg.r*100:.2f}%')
    fig.update_layout(
        title='Thresholds criticos vs growth rate r (estilo Figura 6)',
        xaxis_title='r (growth de F, %)', yaxis_title='threshold passive share %',
        barmode='group', yaxis_range=[0, 100],
        **{**_FIG_LAYOUT, 'height': 460})
    return fig


def run_passive_breaks_section(close: pd.Series, ticker: str = 'SPX Index',
                                 horizon_years: float = 20,
                                 n_paths: int = 100) -> dict:
    """Pipeline completo da secao Passive Breaks."""
    cfg = calibrate_passive_breaks(close)
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

    return {
        'config': cfg,
        'state': state,
        'paths_no_passive': paths_no_p,
        'paths_logistic': paths_logistic,
        'paths_haddad': paths_haddad,
        'horizon_years': horizon_years,
        'figs': {
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
# 8. ORQUESTRADOR + WIDGETS + EXPORT ZIP
# =============================================================================

def compute_session_stats(ticker: str, years: int = 5,
                            include_nomura: bool = True,
                            nomura_ticker: str = 'SPY US Equity',
                            include_gs_factors: bool = False,
                            include_passive_breaks: bool = False) -> dict:
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
            pnl = compute_daily_options_pnl(nd, iv)
            summary = options_pnl_summary(pnl)
            sharpe = options_sharpe(pnl)
            sp = skew_percentiles_multi(iv)       # agora com 3 convencoes
            flows = compute_dynamic_flows(nd)
            nomura = {
                'pnl_daily': pnl, 'pnl_summary': summary, 'sharpe': sharpe,
                'skew_pctiles': sp, 'flows': flows, 'vol_indices': vol,
                'figs': {
                    'options_pnl': fig_options_pnl_heatmap(summary, sharpe),
                    'skew_pctiles': fig_skew_multi(sp),     # 4 paineis
                    'iv_rank': fig_iv_rank(vol),            # IV + Skew rank
                    'flows': fig_systematic_flows(flows),
                    'vol_panic': fig_vol_panic_proxy(daily, vol, ticker),
                }
            }
        except Exception as e:
            log.warning(f'Nomura section falhou: {e}')
            import traceback
            log.warning(traceback.format_exc())

    gs_factors = {}
    if include_gs_factors:
        try:
            gs_factors = run_gs_factor_section(years=max(2, min(years, 5)))
        except Exception as e:
            log.warning(f'GS Factor section falhou: {e}')
            import traceback
            log.warning(traceback.format_exc())

    passive_breaks = {}
    if include_passive_breaks:
        try:
            # Pra passive breaks, quanto mais historia melhor.
            # Tenta carregar ate 30y do mesmo ticker pra calibrar kappa/sigma.
            pb_years = max(years, 15)
            pb_daily = load_daily(ticker, pb_years) if pb_years > years else daily
            passive_breaks = run_passive_breaks_section(
                pb_daily['close'], ticker=ticker, horizon_years=20, n_paths=100)
        except Exception as e:
            log.warning(f'Passive Breaks falhou: {e}')
            import traceback
            log.warning(traceback.format_exc())

    return {
        'ticker': ticker, 'years': years,
        'ts': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'bt': bt, 'bt_eth': bt_eth,
        'bottom_line': bottom_line,
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

    if result.get('nomura'):
        n = result['nomura']
        sec.append(wd.HTML("<div class='mm-section-label'>Nomura — Options PnL Summary "
                             "(valores em % do spot, 2 casas)</div>"))
        sec.append(go.FigureWidget(n['figs']['options_pnl']))
        sec.append(wd.HTML("<div class='mm-section-label'>Options PnL — Cumulative (%)</div>"))
        sec.append(wd.HTML(_df_to_html_table(n['pnl_summary'])))
        sec.append(wd.HTML("<div class='mm-section-label'>Sharpe Ratio Annualized</div>"))
        sec.append(wd.HTML(_df_to_html_table(n['sharpe'])))

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

    # --- Passive Breaks Model ---
    if result.get('passive_breaks') and result['passive_breaks'].get('state'):
        pb = result['passive_breaks']
        sec.append(wd.HTML(
            "<div class='mm-section-label'>Passive Breaks the Market "
            "(Green/Krishnan/Sturm SSRN 2025)</div>"))
        sec.append(wd.HTML(_passive_state_card_html(pb['state'])))
        sec.append(go.FigureWidget(pb['figs']['state_gauge']))
        sec.append(go.FigureWidget(pb['figs']['passive_curve']))
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

    # --- GS Factor Monitor ---
    if result.get('gs_factors') and result['gs_factors'].get('table') is not None \
       and len(result['gs_factors']['table']) > 0:
        gf = result['gs_factors']
        sec.append(wd.HTML(
            "<div class='mm-section-label'>GS Factor Monitor — Barra Pair Indices + "
            "Momentum + Thematic Baskets (historico via BQL)</div>"))
        sec.append(go.FigureWidget(gf['figs']['category_avg']))
        sec.append(go.FigureWidget(gf['figs']['leaderboard_ytd']))
        sec.append(go.FigureWidget(gf['figs']['leaderboard_1m']))
        sec.append(go.FigureWidget(gf['figs']['heatmap']))
        sec.append(go.FigureWidget(gf['figs']['rolling_top']))
        sec.append(wd.HTML("<div class='mm-section-label'>Factor Monitor table "
                             "(ordenado por YTD, %)</div>"))
        sec.append(wd.HTML(_df_to_html_table(gf['table'])))

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
years_w = wd.IntSlider(value=5, min=1, max=10, step=1, description='Years:',
                        layout=wd.Layout(width='300px'))
nomura_ticker_w = wd.Text(value='SPY US Equity', description='Nomura spot:',
                            layout=wd.Layout(width='280px'))
nomura_chk_w = wd.Checkbox(value=True,
                             description='Incluir Nomura (Options PnL + Skew + Flows)',
                             layout=wd.Layout(width='400px'))
gs_factors_chk_w = wd.Checkbox(value=False,
                                  description='Incluir GS Factor Monitor (lento: ~50 BQL queries)',
                                  layout=wd.Layout(width='400px'))
passive_breaks_chk_w = wd.Checkbox(value=False,
                                      description='Incluir Passive Breaks Model (Green/Krishnan 2025 — 15y+ history)',
                                      layout=wd.Layout(width='500px'))

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

            loading.value = DASH_CSS + ("<div class='mm-dash'><div class='mm-card mm-loading'>"
                                           "Processando... (BQL: 1 query spot + VIX/SKEW + "
                                           f"{'~50 factors' if include_gs else 'sem factors'})"
                                           "</div></div>")

            result = compute_session_stats(ticker, years,
                                             include_nomura=include_n,
                                             nomura_ticker=nom_tk,
                                             include_gs_factors=include_gs,
                                             include_passive_breaks=include_pb)

            loading.value = DASH_CSS + f"<div class='mm-dash'><div class='mm-card mm-loading'>Montando widgets...</div></div>"
            sections = build_section_widgets(result)

            # Persiste snapshot pro Export ZIP
            _snapshot.clear()
            _snapshot['result'] = result

            clear_output(wait=True)
            display(wd.HTML(DASH_CSS))
            display(wd.VBox(sections))

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
                    for name, df in [('options_pnl_daily', r['nomura']['pnl_daily']),
                                       ('options_pnl_summary', r['nomura']['pnl_summary']),
                                       ('options_sharpe', r['nomura']['sharpe']),
                                       ('skew_pctiles', r['nomura']['skew_pctiles']),
                                       ('systematic_flows', r['nomura']['flows'])]:
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
            wd.HBox([gs_factors_chk_w]),
            wd.HBox([passive_breaks_chk_w]),
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
