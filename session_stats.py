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

# Layout comum pros gráficos (mesma dimensão/margin em todos)
_FIG_LAYOUT = dict(
    template=DASH_TEMPLATE,
    height=380,
    margin=dict(l=50, r=40, t=45, b=40),
    legend=dict(orientation='h', yanchor='bottom', y=-0.25,
                 xanchor='center', x=0.5, font=dict(size=9)),
)


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
    fig.update_layout(title=title, **{**_FIG_LAYOUT, 'height': 480})
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
                       **{**_FIG_LAYOUT, 'height': 720,
                          'margin': dict(l=200, r=40, t=55, b=40)})
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
                       **{**_FIG_LAYOUT, 'height': 620})
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
                       **{**_FIG_LAYOUT, 'height': 780})
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
    """Compara equity RTH (open->close) vs ETH (close->open) no mesmo ativo."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt_rth.equity.index, y=bt_rth.equity.values,
                              name=f'RTH (Sharpe={bt_rth.metrics["sharpe"]})',
                              line=dict(color=_C['accent'], width=1.4),
                              fill='tozeroy', fillcolor='rgba(88,166,255,0.04)'))
    fig.add_trace(go.Scatter(x=bt_eth.equity.index, y=bt_eth.equity.values,
                              name=f'ETH (Sharpe={bt_eth.metrics["sharpe"]})',
                              line=dict(color=_C['orange'], width=1.4)))
    fig.add_hline(y=1.0, line_color=_C['text_muted'], line_dash='dash', line_width=0.6)
    fig.update_layout(
        title=f'{ticker} — Equity RTH vs ETH (qual sessao paga mais?)',
        yaxis_title='Equity (norm=1)',
        **{**_FIG_LAYOUT, 'height': 440})
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
                       **{**_FIG_LAYOUT, 'height': 500})
    return fig


# =============================================================================
# 8. ORQUESTRADOR + WIDGETS + EXPORT ZIP
# =============================================================================

def compute_session_stats(ticker: str, years: int = 5,
                            include_nomura: bool = True,
                            nomura_ticker: str = 'SPY US Equity') -> dict:
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

    figs = {
        'weekday_bars_rth': fig_weekday_bars(tables['weekday_stats_rth'], f'{ticker} RTH'),
        'weekday_bars_eth': fig_eth_weekday_bars(tables['weekday_stats_eth'], ticker),
        'weekday_hitrate': fig_weekday_hitrate(tables['weekday_stats_rth'], ticker),
        'updown_weekday': fig_updown_weekday(tables['updown_by_weekday'], ticker),
        'equity_dd_rth': fig_equity_dd(bt, f'{ticker} RTH'),
        'equity_dd_eth': fig_equity_dd(bt_eth, f'{ticker} ETH'),
        'rth_vs_eth': fig_rth_vs_eth_equity(bt, bt_eth, ticker),
        'histogram': fig_histogram(df, ticker),
        'ma_residency': fig_ma_residency(tables['ma_residency'], ticker),
        'heatmap_wkd_month': fig_heatmap_wkd_month(df, ticker),
        'streak_distribution': fig_streak_distribution(df, ticker),
        'streaks': fig_streaks(tables['streaks'], ticker),
        'conditional_streaks': fig_conditional_after_streaks(
            tables['conditional_streaks'], ticker),
        'regime': fig_regime(tables['regime'], ticker),
        'gap': fig_gap(tables['gap_stats'], ticker),
        'monthly': fig_monthly(tables['monthly_stats'], ticker),
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
                }
            }
        except Exception as e:
            log.warning(f'Nomura section falhou: {e}')
            import traceback
            log.warning(traceback.format_exc())

    return {
        'ticker': ticker, 'years': years,
        'ts': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'bt': bt, 'bt_eth': bt_eth,
        'tables': tables, 'figs': figs, 'nomura': nomura,
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
    sec.append(wd.HTML("<div class='mm-section-label'>Backtest Equity + Drawdown — RTH vs ETH</div>"))
    sec.append(go.FigureWidget(result['figs']['rth_vs_eth']))
    sec.append(go.FigureWidget(result['figs']['equity_dd_rth']))
    sec.append(go.FigureWidget(result['figs']['equity_dd_eth']))

    sec.append(wd.HTML("<div class='mm-section-label'>Distribuicao Retornos RTH (%)</div>"))
    sec.append(go.FigureWidget(result['figs']['histogram']))

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

        sec.append(wd.HTML("<div class='mm-section-label'>Systematic Flows — "
                             "VC + CTA + RP (USD bn, AUM dinamico)</div>"))
        sec.append(go.FigureWidget(n['figs']['flows']))

    sec.append(wd.HTML(
        "<div class='mm-note'>"
        "<b>Obs:</b> estatistica nao e edge automatico. Linhas com n&lt;30 "
        "tem leitura fraca. 25d IVs aproximados via SKEW index — plugar chain "
        "real (OVDV/OptionMetrics) pra precisao total."
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
            nom_tk = nomura_ticker_w.value.strip() or 'SPY US Equity'

            steps = [
                '1/4: Carregando diario via BQL...',
                '2/4: Calculando features + stats...',
                '3/4: Gerando graficos Plotly...',
                '4/4: Secao Nomura (Options PnL + Skew + Flows)...',
            ]
            loading.value = DASH_CSS + f"<div class='mm-dash'><div class='mm-card mm-loading'>{steps[0]}</div></div>"

            result = compute_session_stats(ticker, years,
                                             include_nomura=include_n,
                                             nomura_ticker=nom_tk)

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
