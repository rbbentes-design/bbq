"""
DASHBOARD UNIFICADO — ANÁLISE DE ETFs ALAVANCADOS E IMPACTO DE MERCADO
======================================================================

Consolida:
  • Motor de rebalanceamento diário (16+ ETFs alavancados / inversos)
  • Distribuição de fluxo por membro do índice (peso por float-adjusted cap)
  • Tabela resumo estilo BofA (AUM, Flow/1%, percentil, %ADV)
  • Séries históricas de flow-per-1% e AUM
  • Gráficos interativos com tooltip/hover (requer %matplotlib widget)
  • Dashboard com modos, presets, CSV export

Uso: Execute todas as células em sequência no BQuant.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 0 — IMPORTS E CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

import math, os, time, threading, warnings
from datetime import datetime, timedelta
from functools import lru_cache
from collections import namedtuple
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------- Backend interativo (antes de importar pyplot) ----------
import matplotlib
try:
    import ipympl
    matplotlib.use("module://ipympl.backend_nbagg")
except Exception:
    pass

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, MaxNLocator, AutoMinorLocator
import ipywidgets as wd
from IPython.display import display

try:
    from IPython import get_ipython
    _ip = get_ipython()
    if _ip is not None:
        _ip.run_line_magic("matplotlib", "widget")
except Exception:
    pass

try:
    import bql
except ImportError:
    raise ImportError("Este notebook requer Bloomberg BQL para Python.")

bq = bql.Service()

TRADING_DAYS = 252.0
PARAMS = {'fill': 'PREV'}

BACKEND_NAME = str(plt.get_backend()).lower()
INTERACTIVE_BACKEND = any(x in BACKEND_NAME for x in ("ipympl", "nbagg", "widget"))

plt.rcParams.update({
    "figure.dpi": 120,
    "figure.figsize": (11, 5),
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "semibold",
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

FIGSIZE_LINE = (9.4, 3.6)
FIGSIZE_AREA = (9.4, 3.8)


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — TIMER E STATUS
# ═══════════════════════════════════════════════════════════════════════════════

class LapTimer:
    def __init__(self, interval=0.25):
        self.t0 = time.perf_counter()
        self.last = self.t0
        self.stage = "iniciando…"
        self._stop = threading.Event()
        self.interval = interval
        self.widget = wd.HTML(value="")
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        while not self._stop.is_set():
            total = time.perf_counter() - self.t0
            self.widget.value = f"⏱ <b>{total:6.2f}s</b> | etapa: {self.stage}"
            time.sleep(self.interval)

    def lap(self, stage: str):
        now = time.perf_counter()
        lap_t = now - self.last
        total = now - self.t0
        print(f"[timer] {stage}: lap={lap_t:.2f}s | total={total:.2f}s")
        self.last = now
        self.stage = stage

    def stop(self, stage: str = "timer parado"):
        self.stage = stage
        self._stop.set()
        try:
            self._th.join(timeout=1.0)
        except Exception:
            pass
        total = time.perf_counter() - self.t0
        print(f"[timer] STOP ({stage}) total={total:.2f}s")


status = wd.Output()
display(status)
TIMER = LapTimer()
with status:
    display(wd.HTML("<b>Performance:</b>"))
    display(wd.HBox([
        TIMER.widget,
        wd.HTML(f"<span style='margin-left:12px;color:#666'>Backend: {plt.get_backend()}</span>")
    ]))

TIMER.lap("BQL iniciado")


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — HELPERS BQL (normalização, séries, lote)
# ═══════════════════════════════════════════════════════════════════════════════

def _last_numeric_col(df: pd.DataFrame):
    """Retorna a última coluna numérica do DataFrame."""
    for c in df.columns[::-1]:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def _norm_cols(df):
    if df is None or len(df) == 0:
        return df, {}
    cols = list(df.columns)
    lower = {str(c).lower(): c for c in cols}
    alias = {}
    dcol = lower.get("date")
    if dcol is None:
        for c in cols:
            if "date" in str(c).lower():
                dcol = c
                break
    if dcol is not None:
        alias["date"] = dcol
    vcol = lower.get("value")
    if vcol is None:
        candidates = [c for c in cols if "(" in str(c) or str(c).lower() in
                      ("px_last", "fund_net_asset_val", "px_volume",
                       "cur_mkt_cap", "eqy_free_float_pct")]
        if candidates:
            vcol = candidates[0]
        else:
            num = df.select_dtypes("number")
            if num.shape[1] == 1:
                vcol = num.columns[0]
    if vcol is not None:
        alias["value"] = vcol
    return df, alias


def _to_series(df, value_key=("value", "px_last", "fund_net_asset_val"),
               index_key="date", name=None):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    df2, alias = _norm_cols(df)
    dcol = (alias.get(index_key, alias.get("date"))
            or next((c for c in df2.columns if "date" in str(c).lower()), None))
    vcol = None
    if isinstance(value_key, (list, tuple)):
        for vk in value_key:
            if vk in alias:
                vcol = alias[vk]
                break
        if vcol is None:
            for vk in value_key:
                mm = next((c for c in df2.columns if str(c).lower() == vk.lower()), None)
                if mm is not None:
                    vcol = mm
                    break
    else:
        vcol = alias.get(value_key)
    if vcol is None:
        num = df2.select_dtypes("number")
        if num.shape[1] == 1:
            vcol = num.columns[0]
    if dcol is None or vcol is None:
        return pd.Series(dtype=float)
    s = df2.set_index(dcol)[vcol].astype(float).sort_index()
    s.index = pd.to_datetime(s.index)
    if name:
        s.name = name
    return s


# ---------- Pivot robusto (long → wide) ----------

def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["|".join(str(x) for x in tup if x is not None).strip()
                      for tup in df.columns.values]
    else:
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
    return df


def _pivot_timeseries(df: pd.DataFrame, value_col: str):
    if df is None or df.empty:
        return {}
    df = _flatten_cols(df)
    datec = next((c for c in df.columns if "date" in c.lower()), None)
    if datec is None and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={df.index.name or "index": "DATE"})
        datec = "DATE"
    id_cands = [c for c in df.columns
                if c.upper() in ("ID", "SECURITY", "TICKER", "BBG_TICKER")]
    if not id_cands and any(df.index.names):
        if any(n and "id" in str(n).lower() for n in df.index.names):
            df = df.reset_index()
            id_cands = [c for c in df.columns
                        if c.upper() in ("ID", "SECURITY", "TICKER", "BBG_TICKER")]
    valc = next((c for c in df.columns if value_col.lower() in c.lower()), None)
    if valc is None:
        num_cols = df.select_dtypes("number").columns.tolist()
        num_cols = [c for c in num_cols if c not in id_cands + ([datec] if datec else [])]
        valc = num_cols[0] if num_cols else None

    out = {}
    if datec and id_cands and valc:
        idc = id_cands[0]
        dfx = df[[idc, datec, valc]].dropna().copy()
        dfx.columns = ["ID", "DATE", "VALUE"]
        dfx["DATE"] = pd.to_datetime(dfx["DATE"])
        for tk, sub in dfx.groupby("ID", sort=False):
            s = sub.set_index("DATE")["VALUE"].astype(float).sort_index()
            s.name = tk
            out[tk] = s
        if out:
            return out

    # Fallback: wide
    if datec and datec in df.columns:
        df = df.copy()
        df[datec] = pd.to_datetime(df[datec])
        df = df.set_index(datec).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception:
            return {}
    dfn = df.select_dtypes(include=[np.number])
    for c in dfn.columns:
        s = dfn[c].astype(float).dropna()
        if not s.empty:
            s.name = c
            out[c] = s.sort_index()
    return out


# ---------- BQL genérico com cache (robusto a formatos long/wide + fallbacks) ----------

@lru_cache(maxsize=256)
def bdh_cached(tickers: tuple, field: str, start: str,
               end: str = None, strict: bool = True) -> pd.DataFrame:
    """
    Retorna DataFrame wide (index=DATE, colunas=tickers) via BQL.
    Estratégia: batch via bq.univ.list, fallback 1-a-1.
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    fld = getattr(bq.data, field)
    uni = bq.univ.list(list(tickers))
    req = bql.Request(uni, fld(dates=bq.func.range(start, end)))
    res = bq.execute(req)
    if not res:
        raise RuntimeError("BQL retornou vazio no batch BDH.")

    got = {}
    if len(res) == 1:
        df0 = res[0].df().copy()
        if not df0.empty:
            cols_map = {c.lower(): c for c in df0.columns}
            date_col = cols_map.get("date") or next(
                (c for c in df0.columns if "date" in c.lower()), None)
            id_col = (cols_map.get("id") or cols_map.get("security")
                      or cols_map.get("ticker")
                      or next((c for c in df0.columns
                               if c.lower() in ("id", "security", "ticker")), None))
            val_col = _last_numeric_col(df0)
            if date_col and id_col and val_col:
                wide0 = df0.pivot(index=date_col, columns=id_col,
                                  values=val_col).sort_index()
                _norm = lambda s: str(s).strip().upper().replace("  ", " ")
                got_norm = {_norm(c): c for c in wide0.columns}
                for t in tickers:
                    key = _norm(t)
                    if key in got_norm:
                        got[t] = wide0[got_norm[key]]
                    else:
                        cands = [k for k in got_norm if k.startswith(key)]
                        if len(cands) == 1:
                            got[t] = wide0[got_norm[cands[0]]]

    missing = [t for t in tickers if t not in got]
    for t in missing:
        try:
            req_i = bql.Request(bq.univ.list([t]),
                                fld(dates=bq.func.range(start, end)))
            res_i = bq.execute(req_i)
            if not res_i:
                continue
            dfi = res_i[0].df().copy()
            if dfi.empty:
                continue
            dc = next((c for c in dfi.columns if "date" in c.lower()), None)
            vc = _last_numeric_col(dfi)
            if dc and vc:
                got[t] = dfi.set_index(dc)[vc].sort_index()
        except Exception:
            continue

    if not got:
        raise RuntimeError("Nenhuma série válida (nem batch, nem fallback).")
    wide = pd.DataFrame(got).sort_index().reindex(columns=list(tickers))
    empty_cols = [c for c in wide.columns if wide[c].dropna().empty]
    if empty_cols and strict:
        raise KeyError(f"Colunas vazias do BQL: {empty_cols}")
    if empty_cols:
        wide = wide.drop(columns=empty_cols)
    return wide


def sanity_check_tickers(tickers: List[str], start: str, end: str) -> Dict[str, str]:
    """Verifica acesso BQL para uma lista de tickers."""
    out = {}
    for t in tickers:
        try:
            r = bq.execute(bql.Request(
                bq.univ.list([t]),
                bq.data.px_last(dates=bq.func.range(start, end))))
            out[t] = "ok" if r and not r[0].df().empty else "vazio"
        except Exception as e:
            out[t] = f"ERRO: {e}"
    return out


# ---------- Fetch em lote ----------

def fetch_nav_bulk(etfs, start, end):
    item = bq.data.fund_net_asset_val(dates=bq.func.range(start, end), fill='PREV')
    req = bql.Request(etfs, item, with_params=PARAMS)
    df = bq.execute(req)[0].df()
    return _pivot_timeseries(df, 'fund_net_asset_val')


def fetch_px_bulk(tickers, start, end):
    item = bq.data.px_last(dates=bq.func.range(start, end), fill='PREV')
    req = bql.Request(tickers, item, with_params=PARAMS)
    df = bq.execute(req)[0].df()
    return _pivot_timeseries(df, 'px_last')


def fetch_px_vol_bulk(tickers, start, end):
    item_px = bq.data.px_last(dates=bq.func.range(start, end), fill='PREV')
    df_px = bq.execute(bql.Request(tickers, item_px, with_params=PARAMS))[0].df()
    px_dict = _pivot_timeseries(df_px, 'px_last')

    item_vol = bq.data.px_volume(dates=bq.func.range(start, end), fill='PREV')
    df_vol = bq.execute(bql.Request(tickers, item_vol, with_params=PARAMS))[0].df()
    vol_dict = _pivot_timeseries(df_vol, 'px_volume')
    return px_dict, vol_dict


# ---------- Séries individuais (com cache) ----------

@lru_cache(maxsize=256)
def px_last_series(ticker: str, start="-1000D", end="0D"):
    item = bq.data.px_last(dates=bq.func.range(start, end), fill='PREV')
    df = bq.execute(bql.Request([ticker], item, with_params=PARAMS))[0].df()
    return _to_series(df, value_key=("px_last", "value"), name=ticker)


@lru_cache(maxsize=256)
def nav_series(etf: str, start="-1000D", end="0D"):
    item = bq.data.fund_net_asset_val(dates=bq.func.range(start, end), fill='PREV')
    df = bq.execute(bql.Request([etf], item, with_params=PARAMS))[0].df()
    return _to_series(df, value_key=("fund_net_asset_val", "value"), name=etf)


@lru_cache(maxsize=256)
def px_volume_series(ticker: str, start="-1000D", end="0D"):
    item = bq.data.px_volume(dates=bq.func.range(start, end), fill='PREV')
    df = bq.execute(bql.Request([ticker], item, with_params=PARAMS))[0].df()
    return _to_series(df, value_key=("px_volume", "value"), name=ticker)


@lru_cache(maxsize=256)
def fund_expense_ratio(etf: str):
    try:
        df = bq.execute(bql.Request(
            [etf], bq.data.fund_expense_ratio(), with_params=PARAMS))[0].df()
        s = _to_series(df, value_key=("fund_expense_ratio", "value"))
        return float(s.iloc[-1]) if len(s) > 0 else np.nan
    except Exception:
        return np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — PESOS DE ÍNDICE E ADV
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=64)
def index_float_weights(index_ticker: str, as_of: str = None):
    """Peso ≈ Market Cap × Free Float% (normalizado). Múltiplos fallbacks."""
    univ = (bq.univ.members([index_ticker], dates=[as_of])
            if as_of else bq.univ.members([index_ticker]))
    try:
        df_cap = bq.execute(bql.Request(
            univ, bq.data.cur_mkt_cap(), with_params=PARAMS))[0].df()
    except Exception:
        df_cap = pd.DataFrame()
    try:
        df_ff = bq.execute(bql.Request(
            univ, bq.data.eqy_free_float_pct(), with_params=PARAMS))[0].df()
    except Exception:
        df_ff = pd.DataFrame()

    def _one(df, keyname):
        if df is None or df.empty:
            return pd.DataFrame()
        if "ID" in df.columns:
            df = df.set_index("ID")
        num = df.select_dtypes("number")
        if num.shape[1] == 0:
            v = next((c for c in df.columns
                      if str(c).lower() in ("value", "cur_mkt_cap",
                                             "eqy_free_float_pct")), None)
            if v is None:
                return pd.DataFrame()
            return pd.to_numeric(df[v], errors="coerce").to_frame(keyname)
        col = num.columns[-1] if num.shape[1] > 1 else num.columns[0]
        return pd.to_numeric(df[col], errors="coerce").to_frame(keyname)

    cap = _one(df_cap, "cap")
    ff = _one(df_ff, "ff_pct")
    df = cap.join(ff, how="outer")

    # fallback cap = PX_LAST × shares_out
    if ("cap" not in df.columns) or df["cap"].isna().all():
        try:
            items = {"px": bq.data.px_last()}
            try:
                items["sh"] = bq.data.eqy_sh_out()
            except Exception:
                items["sh"] = bq.data.bs_sh_out(
                    fa_period_type='Q', fa_period_offset='0')
            dff = bq.execute(bql.Request(univ, items, with_params=PARAMS))[0].df()
            if "ID" in dff.columns:
                dff = dff.set_index("ID")
            pxc = next((c for c in dff.columns
                        if "px_last" in str(c).lower() or str(c).lower() in ("px", "value")), None)
            shc = next((c for c in dff.columns
                        if "sh_out" in str(c).lower() or str(c).lower() == "sh"), None)
            if pxc and shc:
                cap2 = (pd.to_numeric(dff[pxc], errors="coerce")
                        * pd.to_numeric(dff[shc], errors="coerce"))
                df["cap"] = df["cap"].fillna(cap2) if "cap" in df.columns else cap2
        except Exception:
            pass

    if "ff_pct" not in df.columns:
        df["ff_pct"] = 100.0
    df["ff_pct"] = pd.to_numeric(df["ff_pct"], errors="coerce").fillna(100.0)
    if "cap" not in df.columns or df["cap"].isna().all():
        return pd.Series(dtype=float, name="weight")

    df = df[~pd.to_numeric(df["cap"], errors="coerce").isna()].copy()
    df["cap"] = pd.to_numeric(df["cap"], errors="coerce")
    df["adj"] = df["cap"] * (df["ff_pct"] / 100.0)
    total = df["adj"].sum()
    if not np.isfinite(total) or total <= 0:
        return pd.Series(dtype=float, name="weight")
    w = (df["adj"] / total).rename("weight").sort_values(ascending=False)
    w.index.name = "ID"
    return w


_ADV_CACHE = {}


def _adv5d_usd_bulk(tickers, start="-7D", end="0D"):
    key = (tuple(sorted(tickers)), start, end)
    if key in _ADV_CACHE:
        return _ADV_CACHE[key]
    px_dict, vol_dict = fetch_px_vol_bulk(tickers, start, end)
    adv = {}
    for tk in tickers:
        px = px_dict.get(tk, pd.Series(dtype=float))
        vol = vol_dict.get(tk, pd.Series(dtype=float))
        idx = px.index.intersection(vol.index)
        if len(idx) >= 5:
            usd5 = (px.loc[idx] * vol.loc[idx]).tail(5).mean()
            adv[tk] = float(usd5) if pd.notna(usd5) else np.nan
        else:
            adv[tk] = np.nan
    s = pd.Series(adv, dtype=float)
    _ADV_CACHE[key] = s
    return s


@lru_cache(maxsize=256)
def _adv_last_window_usd(symbol: str, minutes: int, lookback_days=20):
    """ADV em USD para os últimos N minutos do dia (30 ou 5). Fallback proporcional."""
    try:
        start_t = "15:30" if minutes == 30 else "15:55"
        bars = bq.data.intraday_bar(
            event="TRADE", interval="MIN",
            time=bq.func.time_range(start_t, "16:00"),
            dates=bq.func.range(f"-{lookback_days}D", "0D"))
        df = bq.execute(bql.Request([symbol], bars))[0].df()
        if df is not None and not df.empty:
            df2, alias = _norm_cols(df)
            dcol = alias.get("date") or next(
                c for c in df2.columns if "date" in str(c).lower())
            pcol = next((c for c in df2.columns
                         if "price" in str(c).lower() or "px_last" in str(c).lower()), None)
            vcol = next((c for c in df2.columns if "volume" in str(c).lower()), None)
            if dcol and pcol and vcol:
                dfi = df2[[dcol, pcol, vcol]].copy()
                dfi.columns = ["DATE", "PX", "VOL"]
                dfi["USD"] = (pd.to_numeric(dfi["PX"], errors="coerce")
                              * pd.to_numeric(dfi["VOL"], errors="coerce"))
                adv_val = float(dfi.groupby("DATE")["USD"].sum().tail(lookback_days).mean())
                return adv_val, False
    except Exception:
        pass
    # Fallback: proporcional ao dia inteiro
    try:
        px = px_last_series(symbol, start=f"-{lookback_days + 5}D", end="0D")
        vol = px_volume_series(symbol, start=f"-{lookback_days + 5}D", end="0D")
        idx = px.index.intersection(vol.index)
        usd = px.loc[idx] * vol.loc[idx]
        adv_val = float(usd.tail(lookback_days).mean() * (minutes / 390.0))
        return adv_val, True
    except Exception:
        return np.nan, True


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — ESPECIFICAÇÕES DE ETFs E MAPEAMENTOS
# ═══════════════════════════════════════════════════════════════════════════════

ETFSpec = namedtuple("ETFSpec", "etf under_ticker leverage expense_ratio fin_mode")


def default_specs():
    _MAP = {
        "TQQQ US Equity": ("NDX Index", +3.0, "futures"),
        "SQQQ US Equity": ("NDX Index", -3.0, "futures"),
        "SPXL US Equity": ("SPX Index", +3.0, "futures"),
        "SPXS US Equity": ("SPX Index", -3.0, "futures"),
        "SOXL US Equity": ("SOX Index", +3.0, "swaps"),
        "SOXS US Equity": ("SOX Index", -3.0, "swaps"),
        "TSLL US Equity": ("TSLA UW Equity", +1.5, "swaps"),
        "TSLQ US Equity": ("TSLA UW Equity", -1.0, "swaps"),
        "NVDU US Equity": ("NVDA UW Equity", +1.5, "swaps"),
        "NVD  US Equity": ("NVDA UW Equity", -1.0, "swaps"),
        "NVOX US Equity": ("NVDA UW Equity", +1.25, "options"),
        "MSTU US Equity": ("MSTR US Equity", +1.5, "swaps"),
        "MSTZ US Equity": ("MSTR US Equity", -1.0, "swaps"),
        "GGLL US Equity": ("GOOGL UW Equity", +1.5, "swaps"),
        "GGLS US Equity": ("GOOGL UW Equity", -1.0, "swaps"),
        "LABU US Equity": ("XBI US Equity", +3.0, "swaps"),
    }
    specs = {}
    for etf, (under, lev, mode) in _MAP.items():
        er = fund_expense_ratio(etf)
        if not np.isfinite(er):
            er = 0.0
        specs[etf] = ETFSpec(etf=etf, under_ticker=under, leverage=lev,
                             expense_ratio=er, fin_mode=mode)
    return specs


IDX_ETFS = {
    "SPX": ["SPXL US Equity", "SPXS US Equity"],
    "NDX": ["TQQQ US Equity", "SQQQ US Equity"],
}

TARGET_MAP = {
    "TSLL US Equity": {"type": "equity", "symbol": "TSLA UW Equity"},
    "TSLQ US Equity": {"type": "equity", "symbol": "TSLA UW Equity"},
    "NVDU US Equity": {"type": "equity", "symbol": "NVDA UW Equity"},
    "NVD  US Equity": {"type": "equity", "symbol": "NVDA UW Equity"},
    "NVOX US Equity": {"type": "equity", "symbol": "NVDA UW Equity"},
    "MSTU US Equity": {"type": "equity", "symbol": "MSTR US Equity"},
    "MSTZ US Equity": {"type": "equity", "symbol": "MSTR US Equity"},
    "GGLL US Equity": {"type": "equity", "symbol": "GOOGL UW Equity"},
    "GGLS US Equity": {"type": "equity", "symbol": "GOOGL UW Equity"},
    "SOXL US Equity": {"type": "etf_proxy", "symbol": "SOXX US Equity"},
    "SOXS US Equity": {"type": "etf_proxy", "symbol": "SOXX US Equity"},
}

INDEX_ROWS = {
    "Index:SPX": {"fut": "ES1 Index", "etfs": IDX_ETFS.get("SPX", [])},
    "Index:NDX": {"fut": "NQ1 Index", "etfs": IDX_ETFS.get("NDX", [])},
}


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — MOTOR DE REBALANCEAMENTO
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rebalance_series_vector(spec: ETFSpec, under_px: pd.Series,
                                    nav: pd.Series) -> pd.DataFrame:
    """Calcula série diária de rebalanceamento para um ETF alavancado."""
    if under_px is None or nav is None or len(under_px) == 0 or len(nav) == 0:
        return pd.DataFrame()
    idx = under_px.index.union(nav.index)
    under_px = under_px.reindex(idx).ffill()
    nav = nav.reindex(idx).ffill()
    r_under = under_px.pct_change().fillna(0.0)
    shares = pd.Series(1e8, index=idx, dtype=float)
    shares_tm1 = shares.shift(1).ffill()
    L = spec.leverage
    E_target = L * nav * shares_tm1
    E_pre = (E_target.shift(1) * (1.0 + r_under)).fillna(E_target.iloc[0])
    trade = (E_target - E_pre).astype(float)
    trade.iloc[0] = 0.0
    df = pd.DataFrame({
        "under_px": under_px, "r_under": r_under,
        "NAV_new": nav, "Shares_t-1": shares_tm1,
        "TradeFund_$": trade
    })
    df.index.name = "DATE"
    return df


@lru_cache(maxsize=64)
def _index_weights_cached(index_ticker: str, as_of: str = None):
    return index_float_weights(index_ticker, as_of=as_of)


def distribute_trade_by_names(spec: ETFSpec, reb_df: pd.DataFrame,
                              as_of=None) -> pd.DataFrame:
    """Distribui o trade do último dia pelas ações do índice (por peso)."""
    if reb_df is None or reb_df.empty:
        return pd.DataFrame()
    last = reb_df.index.max()
    trade_today = float(reb_df.loc[last, "TradeFund_$"])
    under = spec.under_ticker
    if under.endswith(("UW Equity", "UN Equity", "US Equity")):
        flows = pd.DataFrame(index=[under], data={"Weight": [1.0]})
    else:
        w = _index_weights_cached(under, as_of=as_of)
        if len(w) == 0:
            return pd.DataFrame()
        flows = pd.DataFrame({"Weight": w})
    flows["Flow_$"] = flows["Weight"] * trade_today
    adv = _adv5d_usd_bulk(flows.index.tolist(), start="-7D", end="0D")
    flows["%ADV-5D"] = (flows["Flow_$"].abs()
                        / adv.reindex(flows.index).replace(0, np.nan)) * 100.0
    flows["%ADV-5D"] = flows["%ADV-5D"].fillna(0.0)
    return flows.sort_values("Flow_$", ascending=False)


def compute_books_fast(specs: dict, start="-190D", end="0D"):
    """Calcula reb_df para todos os ETFs em lote (rápido)."""
    etfs = list(specs.keys())
    unders = sorted({specs[e].under_ticker for e in etfs})
    nav_map = fetch_nav_bulk(etfs, start, end)
    under_map = fetch_px_bulk(unders, start, end)
    out = {}
    for etf, spec in specs.items():
        nav = nav_map.get(etf)
        if nav is None or len(nav) == 0:
            nav = nav_series(etf, start, end)
        upx = under_map.get(spec.under_ticker)
        if upx is None or len(upx) == 0:
            upx = px_last_series(spec.under_ticker, start, end)
        reb = compute_rebalance_series_vector(spec, upx, nav)
        out[etf] = (reb, None)  # distribuição lazy
    return out


def reload_books(days_window=2000, lookahead=10):
    start = f"-{int(days_window) + int(lookahead)}D"
    end = "0D"
    global books
    books = compute_books_fast(SPECS, start, end)
    print(f"[reload] carregadas {len(books)} ETFs | janela {start}→{end}")


def aggregate_by_name(books_dict, specs_dict):
    """Agrega fluxos de rebalanceamento por nome subjacente (todos os ETFs)."""
    frames = []
    for etf, (reb, _) in books_dict.items():
        if reb is None or reb.empty:
            continue
        spec = specs_dict.get(etf)
        if spec is None:
            continue
        book = distribute_trade_by_names(spec, reb)
        if book is not None and not book.empty:
            frames.append(book[["Flow_$"]].assign(ETF=etf))
    if not frames:
        return pd.DataFrame()
    all_flows = pd.concat(frames)
    agg = all_flows.groupby(level=0).agg(
        **{"Flow_$": ("Flow_$", "sum"), "ETFs": ("ETF", "nunique")})
    adv = _adv5d_usd_bulk(agg.index.tolist())
    agg["%ADV-5D"] = (agg["Flow_$"].abs()
                      / adv.reindex(agg.index).replace(0, np.nan)) * 100.0
    agg["|Flow|"] = agg["Flow_$"].abs()
    return agg.sort_values("|Flow|", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — ANÁLISE DE FLUXO E IMPACTO
# ═══════════════════════════════════════════════════════════════════════════════

def _flow_per_1pct_series(reb_df: pd.DataFrame, signed=True):
    if reb_df is None or reb_df.empty:
        return pd.Series(dtype=float)
    trade = pd.to_numeric(reb_df["TradeFund_$"], errors="coerce")
    if not signed:
        trade = trade.abs()
    r = pd.to_numeric(reb_df["r_under"], errors="coerce").abs().replace(0, np.nan)
    s = trade * (0.01 / r)
    s.name = "Flow_$ per 1%"
    return s.dropna()


def series_index_flow_per_1pct(books_dict, index_key, signed=True):
    pieces = []
    for etf in IDX_ETFS.get(index_key, []):
        reb, _ = books_dict.get(etf, (None, None))
        s = _flow_per_1pct_series(reb, signed=signed)
        if len(s) > 0:
            pieces.append(s)
    if not pieces:
        return pd.Series(dtype=float)
    out = pd.concat(pieces, axis=1).sum(axis=1).sort_index()
    out.name = f"{index_key} Flow per 1%"
    return out


def series_index_aum(books_dict, index_key):
    pieces = []
    for etf in IDX_ETFS.get(index_key, []):
        reb, _ = books_dict.get(etf, (None, None))
        if reb is None or reb.empty:
            continue
        s = (pd.to_numeric(reb["NAV_new"], errors="coerce")
             * pd.to_numeric(reb["Shares_t-1"], errors="coerce"))
        if len(s) > 0:
            pieces.append(s)
    if not pieces:
        return pd.Series(dtype=float)
    out = pd.concat(pieces, axis=1).sum(axis=1).sort_index()
    out.name = f"{index_key} AUM_$"
    return out


def compute_index_impact(index_ticker: str, pct_move: float = 0.01,
                         as_of: Optional[str] = None) -> pd.DataFrame:
    """Contribuição de cada membro do índice para um dado % move."""
    w = index_float_weights(index_ticker, as_of=as_of)
    if len(w) == 0:
        return pd.DataFrame()
    df = w.to_frame().copy()
    df["impact_1pct"] = df["weight"] * pct_move
    return df.sort_values("impact_1pct", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 7 — TABELA RESUMO (estilo BofA)
# ═══════════════════════════════════════════════════════════════════════════════

def make_bofa_table(books_dict):
    rows = []
    # ── Índices ──
    for name, info in INDEX_ROWS.items():
        etfs = info["etfs"]
        fut = info["fut"]
        aums, flows = [], []
        for e in etfs:
            reb, _ = books_dict.get(e, (None, None))
            if reb is None or reb.empty:
                continue
            aums.append(pd.to_numeric(reb["NAV_new"], errors="coerce")
                        * pd.to_numeric(reb["Shares_t-1"], errors="coerce"))
            flows.append(_flow_per_1pct_series(reb, signed=False))
        aum_s = pd.concat(aums, axis=1).sum(axis=1) if aums else pd.Series(dtype=float)
        flow_s = pd.concat(flows, axis=1).sum(axis=1) if flows else pd.Series(dtype=float)
        aum_t = float(aum_s.iloc[-1]) if len(aum_s) > 0 else np.nan
        aum_1wchg = np.nan
        if len(aum_s) > 0:
            prev = aum_s.loc[aum_s.index <= aum_s.index.max() - pd.Timedelta(days=7)]
            if len(prev) > 0:
                aum_1wchg = float(aum_t - prev.iloc[-1])
        flow_t = float(flow_s.iloc[-1]) if len(flow_s) > 0 else np.nan
        pctile = np.nan
        if len(flow_s) > 5:
            win = flow_s.loc[flow_s.index >= flow_s.index.max() - pd.Timedelta(days=365)]
            if len(win) > 5:
                pctile = float((win <= flow_t).mean() * 100.0)
        adv30, _ = _adv_last_window_usd(fut, minutes=30)
        adv05, _ = _adv_last_window_usd(fut, minutes=5)
        pct_05 = (float(abs(flow_t) / adv05 * 100.0)
                  if pd.notna(adv05) and adv05 > 0 and pd.notna(flow_t) else np.nan)
        rows.append({
            "Ticker": name,
            "AUM ($mm)": aum_t / 1e6 if pd.notna(aum_t) else np.nan,
            "1w AUM chg ($mm)": aum_1wchg / 1e6 if pd.notna(aum_1wchg) else np.nan,
            "Rebalance/1% move ($mm)": flow_t / 1e6 if pd.notna(flow_t) else np.nan,
            "1yr %ile of Rebalance": pctile,
            "Rebalance Amt / Last 5m notional (%)": pct_05
        })

    # ── ETFs ──
    for etf, (reb, _) in books_dict.items():
        if reb is None or reb.empty:
            continue
        aum_s = (pd.to_numeric(reb["NAV_new"], errors="coerce")
                 * pd.to_numeric(reb["Shares_t-1"], errors="coerce"))
        aum_t = float(aum_s.iloc[-1]) if len(aum_s) > 0 else np.nan
        aum_1wchg = np.nan
        if len(aum_s) > 0:
            prev = aum_s.loc[aum_s.index <= aum_s.index.max() - pd.Timedelta(days=7)]
            if len(prev) > 0:
                aum_1wchg = float(aum_t - prev.iloc[-1])
        flow_s = _flow_per_1pct_series(reb, signed=False)
        flow_t = float(flow_s.iloc[-1]) if len(flow_s) > 0 else np.nan
        pctile = np.nan
        if len(flow_s) > 5:
            win = flow_s.loc[flow_s.index >= flow_s.index.max() - pd.Timedelta(days=365)]
            if len(win) > 5:
                pctile = float((win <= flow_t).mean() * 100.0)
        tgt = TARGET_MAP.get(etf, {})
        sym = tgt.get("symbol")
        adv05 = np.nan
        if sym:
            adv05, _ = _adv_last_window_usd(sym, minutes=5)
        pct_05 = (float(abs(flow_t) / adv05 * 100.0)
                  if pd.notna(adv05) and adv05 > 0 and pd.notna(flow_t) else np.nan)
        rows.append({
            "Ticker": etf,
            "AUM ($mm)": aum_t / 1e6 if pd.notna(aum_t) else np.nan,
            "1w AUM chg ($mm)": aum_1wchg / 1e6 if pd.notna(aum_1wchg) else np.nan,
            "Rebalance/1% move ($mm)": flow_t / 1e6 if pd.notna(flow_t) else np.nan,
            "1yr %ile of Rebalance": pctile,
            "Rebalance Amt / Last 5m notional (%)": pct_05
        })

    df = pd.DataFrame(rows).set_index("Ticker").sort_values("AUM ($mm)", ascending=False)
    sty = (df.style
           .format({
               "AUM ($mm)": "{:,.0f}",
               "1w AUM chg ($mm)": "{:,.0f}",
               "Rebalance/1% move ($mm)": "{:,.0f}",
               "1yr %ile of Rebalance": "{:,.0f}",
               "Rebalance Amt / Last 5m notional (%)": "{:,.1f}",
           })
           .background_gradient(cmap="RdYlGn_r", subset=["1yr %ile of Rebalance"])
           .background_gradient(cmap="RdYlGn_r",
                                subset=["Rebalance Amt / Last 5m notional (%)"]))
    return df, sty


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 8 — MOTOR DE PLOTAGEM
# ═══════════════════════════════════════════════════════════════════════════════

def _scale(series, units):
    if series is None or len(series) == 0:
        return series, FuncFormatter(lambda x, _: f"{x:,.2f}"), "", 1.0
    if units == "Billions":
        return (series / 1e9, FuncFormatter(lambda y, _: f"{y:,.1f}"),
                " ($bn)", 1e9)
    if units == "Millions":
        return (series / 1e6, FuncFormatter(lambda y, _: f"{y:,.0f}"),
                " ($mm)", 1e6)
    return series, FuncFormatter(lambda y, _: f"${y:,.0f}"), " ($)", 1.0


def _format_date_axis(ax):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.tick_params(axis='x', rotation=0)
    ax.margins(x=0.02)


def _apply_bofa_axes(ax, units, s=None):
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.margins(y=0.05)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both',
                                           steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    if s is not None:
        _format_date_axis(ax)


# ---------- Overlay interativo (hover com crosshair + tooltip) ----------

_INTERACTIVE_HANDLES = []


def _nearest_idx(index, xnum):
    if xnum is None or len(index) == 0:
        return None
    xdt = pd.Timestamp(mdates.num2date(xnum)).tz_localize(None)
    pos = index.get_indexer([xdt], method='nearest')
    return int(pos[0])


def _fmt_value(val, units):
    if val is None or not np.isfinite(val):
        return "–"
    if units == "Billions":
        return f"{val:,.2f}"
    if units == "Millions":
        return f"{val:,.0f}"
    return f"{val:,.0f}"


def enable_overlay(ax, units, series_map):
    """series_map: dict{name -> (DatetimeIndex, np.array)} já na escala do plot."""
    if not INTERACTIVE_BACKEND:
        ax.text(0.99, 0.02, "Ative '%matplotlib widget' p/ hover",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#777")
        return None
    vline = ax.axvline(ax.get_xlim()[0], lw=0.8, alpha=0.6, color="#999")
    hline = ax.axhline(ax.get_ylim()[0], lw=0.8, alpha=0.6, color="#999")
    markers = {}
    for name, (idx, vals) in series_map.items():
        (pt,) = ax.plot(
            [idx[-1] if len(idx) > 0 else []],
            [vals[-1] if len(vals) > 0 else []],
            marker='o', ms=4, linestyle='None')
        markers[name] = pt
    ann = ax.annotate("", xy=(1, 1), xycoords="axes fraction",
                      xytext=(-8, -8), textcoords="offset points",
                      ha="right", va="top", fontsize=9,
                      bbox=dict(boxstyle="round,pad=0.3", fc="w",
                                ec="0.5", alpha=0.9))
    ann.set_visible(False)

    def on_move(event):
        if event.inaxes != ax or event.xdata is None:
            ann.set_visible(False)
            plt.gcf().canvas.draw_idle()
            return
        xnum = event.xdata
        y = event.ydata
        vline.set_xdata([xnum, xnum])
        hline.set_ydata([y, y])
        first_key = next(iter(series_map))
        idx_ref = series_map[first_key][0]
        if len(idx_ref) == 0:
            ann.set_visible(False)
            plt.gcf().canvas.draw_idle()
            return
        i = _nearest_idx(idx_ref, xnum)
        lines = [idx_ref[i].strftime("%d %b %Y")]
        for name, (idx, vals) in series_map.items():
            if len(idx) == 0:
                continue
            j = min(max(i, 0), len(idx) - 1)
            xj = mdates.date2num(idx[j].to_pydatetime())
            yv = float(vals[j])
            markers[name].set_data([xj], [yv])
            lines.append(f"{name}: {_fmt_value(yv, units)}")
        ann.set_text("\n".join(lines))
        ann.set_visible(True)
        plt.gcf().canvas.draw_idle()

    cid = plt.gcf().canvas.mpl_connect("motion_notify_event", on_move)
    _INTERACTIVE_HANDLES.append((cid, vline, hline, markers, ann))
    return cid


# ---------- Funções de plot ----------

def plot_line_bofa(series, title, units="Billions", color="#0b2e59",
                   interativo=False):
    s, yfmt, suf, _ = _scale(series, units)
    plt.figure(figsize=FIGSIZE_LINE, constrained_layout=True)
    ax = plt.gca()
    ax.plot(s.index, s.values, linewidth=2.0, color=color, label=title)
    if len(s) > 0:
        ax.scatter([s.index[-1]], [s.values[-1]], s=30, zorder=5, color=color)
        label = f"{s.values[-1]:,.2f}" if units != "Millions" else f"{s.values[-1]:,.0f}"
        ax.annotate(label, xy=(s.index[-1], s.values[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9)
    ax.set_title(title + suf)
    ax.yaxis.set_major_formatter(yfmt)
    _apply_bofa_axes(ax, units, s)
    if interativo and len(s) > 0:
        enable_overlay(ax, units, {title: (s.index, s.values)})
    plt.show()
    return s


def plot_multi_bofa(series_dict, title, units="Billions", interativo=False):
    colors = ["#0b2e59", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    plt.figure(figsize=FIGSIZE_LINE, constrained_layout=True)
    ax = plt.gca()
    overlay_map = {}
    yfmt = None
    for i, (name, series) in enumerate(series_dict.items()):
        s, yfmt, suf, _ = _scale(series, units)
        ax.plot(s.index, s.values, linewidth=2.0, label=name,
                color=colors[i % len(colors)])
        if len(s) > 0:
            ax.scatter([s.index[-1]], [s.values[-1]], s=28, zorder=5,
                       color=colors[i % len(colors)])
        overlay_map[name] = (s.index, s.values)
    ax.set_title(title + (" ($bn)" if units == "Billions"
                          else " ($mm)" if units == "Millions" else " ($)"))
    if yfmt is None:
        yfmt = FuncFormatter(lambda y, _: f"{y:,.2f}")
    ax.yaxis.set_major_formatter(yfmt)
    _apply_bofa_axes(ax, units, True)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    if interativo and any(len(v[0]) > 0 for v in overlay_map.values()):
        enable_overlay(ax, units, overlay_map)
    plt.show()


def plot_area_aum(spx_aum, ndx_aum,
                  title="S&P 500 and NASDAQ-100 leveraged & inverse ETF AUM",
                  interativo=False):
    idx = spx_aum.index.union(ndx_aum.index)
    spx = spx_aum.reindex(idx).ffill().fillna(0) / 1e9
    ndx = ndx_aum.reindex(idx).ffill().fillna(0) / 1e9
    plt.figure(figsize=FIGSIZE_AREA, constrained_layout=True)
    ax = plt.gca()
    ax.fill_between(idx, spx, alpha=1.0, label="S&P 500")
    ax.fill_between(idx, spx + ndx, spx, alpha=0.6, label="NASDAQ-100")
    ax.set_title(title)
    ax.set_ylabel("AUM ($bn)")
    _apply_bofa_axes(ax, "Billions")
    _format_date_axis(ax)
    ax.legend(frameon=False)
    if interativo and len(idx) > 0:
        enable_overlay(ax, "Billions", {
            "S&P 500": (idx, spx.values),
            "NASDAQ-100 cum": (idx, (spx + ndx).values)
        })
    plt.show()


def plot_heatmap(df: pd.DataFrame, title: str):
    """Heatmap simples (útil para impacto de membros do índice)."""
    fig, ax = plt.subplots()
    im = ax.imshow(df.values, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df.index.astype(str))
    fig.colorbar(im, ax=ax)
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 9 — DASHBOARD INTERATIVO
# ═══════════════════════════════════════════════════════════════════════════════

# ---------- Carga inicial ----------
books = {}
SPECS = default_specs()

TIMER.lap("carregando books")
reload_books(2000)
TIMER.lap("books carregados")

# ---------- Widgets ----------
mode_dd = wd.Dropdown(options=[
    "Índice • Flow por 1%",
    "Índice • AUM",
    "ETF • Flow por 1%",
    "ETF • AUM",
    "Tabela (BofA-like)",
    "Fluxo por Nome",
], value="Índice • Flow por 1%", description="View:")

index_dd = wd.Dropdown(options=list(IDX_ETFS.keys()), value="NDX",
                       description="Índice:")
etf_dd = wd.Dropdown(options=sorted(list(SPECS.keys())),
                     value="TQQQ US Equity", description="ETF:")
etf_multi = wd.SelectMultiple(
    options=sorted(list(SPECS.keys())),
    value=("TQQQ US Equity", "SQQQ US Equity"),
    description="ETFs (multi):", rows=6)

# Nome: Aggregate ou ETF individual
name_dd = wd.Dropdown(
    options=["Aggregate"] + sorted(list(SPECS.keys())),
    value="Aggregate", description="Target:")
side_rb = wd.RadioButtons(
    options=["Buys", "Sells", "All"], value="Sells", description="Side:")
topn_sl = wd.IntSlider(value=20, min=5, max=50, step=5, description="Top N:")

signed_sw = wd.ToggleButtons(
    options=[("Com Sinal", "signed"), ("Magnitude (abs)", "abs")],
    value="signed", description="Notional:")
units_dd = wd.Dropdown(options=["Billions", "Millions"],
                       value="Billions", description="Unidades:")
compare_cb = wd.Checkbox(value=False, description="Comparar")
inter_cb = wd.Checkbox(value=True, description="Hover interativo")

presets = wd.ToggleButtons(
    options=[("1M", "1M"), ("3M", "3M"), ("6M", "6M"),
             ("YTD", "YTD"), ("1Y", "1Y"), ("Max", "MAX")],
    description="Presets:")
days_sl = wd.IntSlider(value=180, min=30, max=2000, step=10,
                       description="Dias:")

btn_plot = wd.Button(description="Plotar / Atualizar", button_style="primary")
btn_reload = wd.Button(description="Recarregar BQL")
btn_csv = wd.Button(description="Exportar CSV")
btn_timer_stop = wd.Button(description="Parar Timer")

out = wd.Output(layout={'border': '1px solid #777', 'height': '460px',
                        'overflow_y': 'auto'})


# ---------- Preset logic ----------

def _apply_preset(p):
    today = pd.Timestamp.today().normalize()
    if p == "1M":
        days_sl.value = 30
    elif p == "3M":
        days_sl.value = 90
    elif p == "6M":
        days_sl.value = 180
    elif p == "1Y":
        days_sl.value = 365
    elif p == "YTD":
        jan1 = pd.Timestamp(today.year, 1, 1)
        days_sl.value = max(30, (today - jan1).days + 1)
    elif p == "MAX":
        days_sl.value = days_sl.max


def _on_preset_change(change):
    if change['name'] == "value" and change['new'] is not None:
        _apply_preset(change['new'])
        _do_plot()

presets.observe(_on_preset_change, names='value')


# ---------- Toggle visibilidade ----------

def _toggle_vis(*_):
    for w in [index_dd, etf_dd, etf_multi, signed_sw,
              name_dd, side_rb, topn_sl]:
        w.layout.display = "none"

    if mode_dd.value == "Fluxo por Nome":
        name_dd.layout.display = "block"
        side_rb.layout.display = "block"
        topn_sl.layout.display = "block"
        return

    if "Flow" in mode_dd.value:
        signed_sw.layout.display = "block"

    if mode_dd.value.startswith("Índice"):
        index_dd.layout.display = "block"
    elif mode_dd.value.startswith("ETF"):
        if compare_cb.value:
            etf_multi.layout.display = "block"
        else:
            etf_dd.layout.display = "block"

mode_dd.observe(_toggle_vis, "value")
compare_cb.observe(_toggle_vis, "value")
_toggle_vis()

_LAST = {"name": None, "raw": None, "display": None}


def _subset_days(s, days):
    if s is None or len(s) == 0:
        return s
    s = s.copy().sort_index()
    last = s.index.max()
    return s.loc[s.index >= (last - pd.Timedelta(days=int(days) - 1))]


# ---------- Lógica de plot principal ----------

def _do_plot(_=None):
    TIMER.lap(f"plot start: {mode_dd.value}")
    out.clear_output()
    with out:
        interativo = inter_cb.value

        # ── Índice: Flow por 1% ──
        if mode_dd.value == "Índice • Flow por 1%":
            signed = (signed_sw.value == "signed")
            if compare_cb.value:
                sd = {}
                for k in IDX_ETFS:
                    s = _subset_days(
                        series_index_flow_per_1pct(books, k, signed=signed),
                        days_sl.value)
                    if s is not None and len(s) > 0:
                        sd[k] = s
                if not sd:
                    print("Sem dados.")
                else:
                    plot_multi_bofa(sd, "Flow por 1% — Índices",
                                   units_dd.value, interativo=interativo)
                    idx = pd.Index([])
                    for v in sd.values():
                        idx = idx.union(v.index)
                    raw = pd.DataFrame({k: v.reindex(idx) for k, v in sd.items()})
                    _LAST.update(name="indices_flow1pct_compare", raw=raw, display=raw)
            else:
                s = _subset_days(
                    series_index_flow_per_1pct(books, index_dd.value, signed=signed),
                    days_sl.value)
                if s is None or len(s) == 0:
                    print("Sem dados.")
                else:
                    disp = plot_line_bofa(s, f"{index_dd.value} — Flow por 1%",
                                         units_dd.value, interativo=interativo)
                    _LAST.update(name=f"{index_dd.value}_flow1pct",
                                 raw=s.to_frame("Flow_$"),
                                 display=disp.to_frame("Flow_units"))

        # ── Índice: AUM ──
        elif mode_dd.value == "Índice • AUM":
            spx = _subset_days(series_index_aum(books, "SPX"), days_sl.value)
            ndx = _subset_days(series_index_aum(books, "NDX"), days_sl.value)
            if (spx is None or len(spx) == 0) and (ndx is None or len(ndx) == 0):
                print("Sem dados.")
            else:
                spx = spx if spx is not None else pd.Series(dtype=float)
                ndx = ndx if ndx is not None else pd.Series(dtype=float)
                plot_area_aum(spx, ndx, interativo=interativo)
                idx = spx.index.union(ndx.index) if len(spx) > 0 else ndx.index
                raw = pd.DataFrame({
                    "SPX_AUM_$": spx.reindex(idx) if len(spx) > 0 else np.nan,
                    "NDX_AUM_$": ndx.reindex(idx) if len(ndx) > 0 else np.nan
                })
                _LAST.update(name="AUM_SPX_NDX", raw=raw, display=raw / 1e9)

        # ── ETF: Flow por 1% ──
        elif mode_dd.value == "ETF • Flow por 1%":
            signed = (signed_sw.value == "signed")
            if compare_cb.value:
                sel = list(etf_multi.value)[:5]
                if not sel:
                    print("Selecione ao menos uma ETF.")
                else:
                    sd = {}
                    for et in sel:
                        reb, _ = books.get(et, (None, None))
                        s = _subset_days(_flow_per_1pct_series(reb, signed=signed),
                                         days_sl.value)
                        if s is not None and len(s) > 0:
                            sd[et] = s
                    if not sd:
                        print("Sem dados.")
                    else:
                        plot_multi_bofa(sd, "Flow por 1% — ETFs",
                                       units_dd.value, interativo=interativo)
                        all_idx = pd.Index([])
                        for v in sd.values():
                            all_idx = all_idx.union(v.index)
                        raw = pd.DataFrame({k: v.reindex(all_idx) for k, v in sd.items()})
                        _LAST.update(name="ETF_flow1pct_compare", raw=raw, display=raw)
            else:
                reb, _ = books.get(etf_dd.value, (None, None))
                s = _subset_days(_flow_per_1pct_series(reb, signed=signed),
                                 days_sl.value)
                if s is None or len(s) == 0:
                    print("Sem dados.")
                else:
                    disp = plot_line_bofa(s, f"{etf_dd.value} — Flow por 1%",
                                         units_dd.value, interativo=interativo)
                    _LAST.update(name=f"{etf_dd.value}_flow1pct",
                                 raw=s.to_frame("Flow_$"),
                                 display=disp.to_frame("Flow_units"))

        # ── ETF: AUM ──
        elif mode_dd.value == "ETF • AUM":
            if compare_cb.value:
                sel = list(etf_multi.value)[:5]
                if not sel:
                    print("Selecione ao menos uma ETF.")
                else:
                    sd = {}
                    for et in sel:
                        reb, _ = books.get(et, (None, None))
                        if reb is None or reb.empty:
                            continue
                        s = (pd.to_numeric(reb["NAV_new"], errors="coerce")
                             * pd.to_numeric(reb["Shares_t-1"], errors="coerce"))
                        s = _subset_days(s, days_sl.value)
                        if s is not None and len(s) > 0:
                            s.name = et
                            sd[et] = s
                    if not sd:
                        print("Sem dados.")
                    else:
                        plot_multi_bofa(sd, "AUM — ETFs", units_dd.value,
                                       interativo=interativo)
                        all_idx = pd.Index([])
                        for v in sd.values():
                            all_idx = all_idx.union(v.index)
                        raw = pd.DataFrame({k: v.reindex(all_idx) for k, v in sd.items()})
                        _LAST.update(name="ETF_AUM_compare", raw=raw, display=raw)
            else:
                reb, _ = books.get(etf_dd.value, (None, None))
                if reb is None or reb.empty:
                    print("Sem dados.")
                else:
                    s = (pd.to_numeric(reb["NAV_new"], errors="coerce")
                         * pd.to_numeric(reb["Shares_t-1"], errors="coerce"))
                    s = _subset_days(s, days_sl.value)
                    disp = plot_line_bofa(s, f"{etf_dd.value} — AUM",
                                         units_dd.value, interativo=interativo)
                    _LAST.update(name=f"{etf_dd.value}_AUM",
                                 raw=s.to_frame("AUM_$"),
                                 display=disp.to_frame("AUM_units"))

        # ── Tabela BofA ──
        elif mode_dd.value == "Tabela (BofA-like)":
            df, sty = make_bofa_table(books)
            display(sty)
            _LAST.update(name="bofa_table_all", raw=df, display=df)

        # ── Fluxo por Nome (aggregado ou por ETF) ──
        elif mode_dd.value == "Fluxo por Nome":
            target = name_dd.value
            side = side_rb.value
            topn = topn_sl.value
            if target == "Aggregate":
                df = aggregate_by_name(books, SPECS)
            else:
                spec = SPECS.get(target)
                reb, _ = books.get(target, (None, None))
                if spec and reb is not None and not reb.empty:
                    df = distribute_trade_by_names(spec, reb)
                else:
                    df = pd.DataFrame()
            if df.empty:
                print("Sem dados de fluxo por nome.")
            else:
                if side == "Buys":
                    df = df[df["Flow_$"] > 0].sort_values("Flow_$", ascending=False)
                elif side == "Sells":
                    df = df[df["Flow_$"] < 0].sort_values("Flow_$")
                else:
                    df = df.reindex(df["Flow_$"].abs().sort_values(ascending=False).index)
                df = df.head(topn)
                fmt_cols = {"Flow_$": "${:,.0f}", "%ADV-5D": "{:.2f}%"}
                if "Weight" in df.columns:
                    fmt_cols["Weight"] = "{:.4f}"
                if "|Flow|" in df.columns:
                    fmt_cols["|Flow|"] = "${:,.0f}"
                sty = (df.style.format(fmt_cols)
                       .background_gradient(cmap="RdYlGn",
                                           subset=["%ADV-5D"]))
                display(sty)
                _LAST.update(name=f"name_flows_{target}_{side}", raw=df, display=df)

    TIMER.lap("plot fim")


# ---------- Callbacks utilitários ----------

def _click_reload(_):
    TIMER.lap("reload start")
    out.clear_output(wait=True)
    with out:
        reload_books(days_sl.value)
    TIMER.lap("reload fim")


def _export_csv(_=None):
    raw = _LAST.get("raw")
    name = _LAST.get("name", "export")
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        with out:
            print("Nada para exportar (plote algo antes).")
        return
    os.makedirs("./exports/csv", exist_ok=True)
    stamp = pd.Timestamp.today().strftime("%Y%m%d_%H%M%S")
    p1 = f"./exports/csv/{name}_{stamp}.csv"
    raw.to_csv(p1)
    with out:
        print(f"CSV salvo: {os.path.abspath(p1)}")
    TIMER.lap("export CSV")


def _stop_timer(_):
    TIMER.stop("usuário")


btn_plot.on_click(_do_plot)
btn_reload.on_click(_click_reload)
btn_csv.on_click(_export_csv)
btn_timer_stop.on_click(_stop_timer)


# ---------- Layout ----------
row_mode = wd.HBox([mode_dd, compare_cb, inter_cb, units_dd])
row_select = wd.HBox([index_dd, etf_dd, etf_multi, signed_sw,
                      name_dd, side_rb, topn_sl])
row_btns = wd.HBox([btn_plot, btn_reload, btn_csv, btn_timer_stop])

TIMER.lap("widgets criados")

display(row_mode, row_select, row_btns, out)

row_time = wd.HBox([presets, days_sl])
display(row_time)

TIMER.lap("widgets exibidos")

# ---------- Render inicial ----------
TIMER.lap("plot inicial start")
_do_plot()
TIMER.stop("UI pronta (plot inicial)")
