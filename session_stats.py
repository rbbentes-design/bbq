"""
session_stats.py — Engine de estatistica intradiaria da sessao regular.

Standalone. Nao depende do codigo-mae. Roda em 2 ambientes:

  1) LOCAL (fora do Bloomberg):
     pip install yfinance pandas numpy matplotlib scipy
     python session_stats.py

  2) BQUANT (Bloomberg Terminal):
     %run session_stats.py
     # ou em celula:
     #   from session_stats import analyze_ticker, run_nomura_section, main
     #   main()
     # O script detecta o objeto `bq` / modulo `bql` e usa BQL automaticamente.
     # Tickers yfinance sao mapeados via YF_TO_BBG (SPY -> "SPY US Equity", etc).

Uso basico:
    python session_stats.py                     # roda universo default
    python session_stats.py SPY QQQ             # tickers custom
    python session_stats.py --years 10 SPY      # janela de 10 anos

Saida: pasta session_stats_out/ + arquivo session_stats_<timestamp>.zip
contendo CSVs de todas as tabelas, PNGs de todos os graficos e summary.txt.

Estrategia base:
    RTH  = entra no primeiro minuto do pregao regular, sai no ultimo minuto
    AH   = entra no fechamento do RTH, sai no 1 min antes do RTH do dia seguinte
           (equivalente a 4:01 ate 9:29 para futuros)

Convencoes:
    - Dados diarios: RTH return = (Close - Open) / Open
                     AH/overnight return = (Open_t - Close_{t-1}) / Close_{t-1}
    - Dados minuto (quando disponiveis): usa o primeiro e ultimo minuto da sessao
    - Tudo em timezone America/New_York, tratando DST automaticamente via pandas

Filosofia: estatistica nao e edge. Amostra sempre reportada. Sem conclusoes
exageradas quando N < 30 (flag explicito no relatorio).
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import sys
import warnings
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Matplotlib backend (nao interativo) antes do pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    from scipy import stats as scistats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

# BQL detection: em BQuant o objeto `bq` e injetado no namespace do kernel.
# Fora de BQuant nao existe — fallback e yfinance.
try:
    import bql                         # noqa: F401
    _bq = bql.Service()
    HAS_BQL = True
except Exception:
    try:
        # Em BQuant o objeto ja vem injetado no namespace builtins
        import builtins
        _bq = getattr(builtins, "bq", None)
        HAS_BQL = _bq is not None
    except Exception:
        _bq = None
        HAS_BQL = False

# =============================================================================
# CONFIG
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("session_stats")

NY_TZ = "America/New_York"

# Default: equities + futuros principais
DEFAULT_UNIVERSE = ["SPY", "QQQ", "IWM", "ES=F", "NQ=F", "RTY=F"]

# Regras de sessao por classe de ativo
# RTH = regular trading hours; AH = after hours (inclui pre-market do dia seguinte)
SESSION_RULES = {
    "equity": {
        "rth_open": time(9, 30),
        "rth_close": time(16, 0),
        "description": "Equities/ETFs: 09:30-16:00 NY",
    },
    "futures": {
        "rth_open": time(9, 30),
        "rth_close": time(16, 0),
        "ah_open": time(16, 1),    # abre o AH logo apos o close RTH
        "ah_close": time(9, 29),   # fecha 1 min antes do RTH do proximo dia
        "description": "Futures SPX/NDX/RTY: RTH 09:30-16:00, AH 16:01-09:29",
    },
}

def classify_asset(ticker: str) -> str:
    """Classifica ticker em equity ou futures baseado em convencao yfinance."""
    return "futures" if ticker.endswith("=F") else "equity"


# Cores usadas nos graficos (paleta HUD discreta, sem exageros)
COLORS = {
    "bg": "#0E1117",
    "fg": "#E6E6E6",
    "grid": "#2A2E36",
    "up": "#26A69A",
    "down": "#EF5350",
    "neutral": "#90A4AE",
    "accent": "#FFB300",
    "accent2": "#42A5F5",
}

def _apply_style():
    """Aplica estilo escuro consistente a todos os plots."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["bg"],
        "savefig.facecolor": COLORS["bg"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["fg"],
        "axes.titlecolor": COLORS["fg"],
        "xtick.color": COLORS["fg"],
        "ytick.color": COLORS["fg"],
        "text.color": COLORS["fg"],
        "grid.color": COLORS["grid"],
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "font.size": 10,
        "axes.titleweight": "bold",
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
    })

_apply_style()

# =============================================================================
# 1. DATA LOADER
# =============================================================================

@dataclass
class DataBundle:
    """Container para dados de um ticker."""
    ticker: str
    asset_class: str
    daily: pd.DataFrame              # OHLCV diario, index tz-aware NY
    minute: Optional[pd.DataFrame]   # OHLCV minuto, index tz-aware NY (se disponivel)
    meta: dict = field(default_factory=dict)

    def __repr__(self):
        n = len(self.daily) if self.daily is not None else 0
        m = len(self.minute) if self.minute is not None else 0
        return f"DataBundle({self.ticker}, daily={n}, minute={m})"


# ---- Mapeamento yfinance <-> Bloomberg tickers (BQuant) ----
# Se rodar no BQuant, mapeamos o simbolo yfinance para o BBG equivalente.
YF_TO_BBG = {
    "SPY": "SPY US Equity",
    "QQQ": "QQQ US Equity",
    "IWM": "IWM US Equity",
    "DIA": "DIA US Equity",
    "ES=F": "ES1 Index",
    "NQ=F": "NQ1 Index",
    "RTY=F": "RTY1 Index",
    "YM=F": "DM1 Index",
    "^VIX": "VIX Index",
    "^SKEW": "SKEW Index",
    "^SPX": "SPX Index",
    "^NDX": "NDX Index",
    "^RUT": "RTY Index",
}


def _bbg_ticker(yf_ticker: str) -> str:
    """Converte simbolo yfinance para Bloomberg ticker."""
    return YF_TO_BBG.get(yf_ticker, yf_ticker)


def load_ticker_bql(ticker: str, years: int = 5) -> Optional[DataBundle]:
    """
    Carrega OHLCV diario via BQL (Bloomberg BQuant).
    Nao suporta minuto (BQL intraday exige outro pipeline via UDC).
    """
    if not HAS_BQL or _bq is None:
        return None
    bbg = _bbg_ticker(ticker)
    asset_class = classify_asset(ticker)
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=int(years * 365.25))).strftime("%Y-%m-%d")
    log.info(f"[{ticker}] BQL fetch {bbg} ({start} -> {end})")
    try:
        u = _bq.univ.list([bbg])
        dr = _bq.func.range(start, end)
        flds = {
            "open": _bq.data.px_open(dates=dr, fill="prev"),
            "high": _bq.data.px_high(dates=dr, fill="prev"),
            "low": _bq.data.px_low(dates=dr, fill="prev"),
            "close": _bq.data.px_last(dates=dr, fill="prev"),
            "volume": _bq.data.px_volume(dates=dr, fill="prev"),
        }
        req = bql.Request(u, flds)
        res = _bq.execute(req)
        # Monta DataFrame alinhado por date
        frames = []
        for item in res:
            df = item.df().reset_index()
            name = item.name
            df = df[["DATE", name]].rename(columns={"DATE": "date", name: name})
            frames.append(df.set_index("date"))
        daily = pd.concat(frames, axis=1).dropna(how="all")
        daily.columns = [c.lower() for c in daily.columns]
        if daily.index.tz is None:
            daily.index = pd.to_datetime(daily.index).tz_localize(NY_TZ)
        else:
            daily.index = daily.index.tz_convert(NY_TZ)
        daily = daily[["open", "high", "low", "close", "volume"]].astype(float)
        meta = {
            "ticker": ticker, "bbg_ticker": bbg, "asset_class": asset_class,
            "source": "bql",
            "daily_start": daily.index.min().isoformat(),
            "daily_end": daily.index.max().isoformat(),
            "n_days": len(daily), "n_minutes": 0,
        }
        return DataBundle(ticker=ticker, asset_class=asset_class,
                           daily=daily, minute=None, meta=meta)
    except Exception as e:
        log.warning(f"[{ticker}] BQL falhou: {e}")
        return None


def load_ticker(ticker: str, years: int = 5, fetch_minute: bool = False) -> Optional[DataBundle]:
    """
    Carrega dados de um ticker.

    Prioridade:
      1. BQL (se rodar em BQuant/Bloomberg Terminal)
      2. yfinance (fallback fora do BBG)

    Args:
        ticker: simbolo yfinance (SPY, ES=F, ^VIX, ...). Mapeado pra BBG via YF_TO_BBG.
        years: janela de dados diarios
        fetch_minute: tentar minuto (so yfinance, max 30 dias)
    """
    # Prioridade: BQL
    if HAS_BQL:
        bundle = load_ticker_bql(ticker, years=years)
        if bundle is not None:
            return bundle
        log.info(f"[{ticker}] BQL retornou vazio, tentando yfinance")

    if not HAS_YF:
        log.error("Nem BQL nem yfinance disponiveis. Instale: pip install yfinance")
        return None

    asset_class = classify_asset(ticker)
    end = datetime.now()
    start = end - timedelta(days=int(years * 365.25))

    log.info(f"[{ticker}] carregando {years}y diario + minuto={fetch_minute}")

    try:
        daily = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as e:
        log.warning(f"[{ticker}] falha no fetch diario: {e}")
        return None

    if daily is None or len(daily) == 0:
        log.warning(f"[{ticker}] sem dados diarios")
        return None

    # yfinance as vezes retorna MultiIndex columns quando ha 1 ticker so
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)

    daily = daily.rename(columns=str.lower).rename(columns={
        "adj close": "adj_close",
    })
    # Index para NY timezone (se vier naive, assume UTC midnight -> converte para NY)
    if daily.index.tz is None:
        daily.index = daily.index.tz_localize("UTC").tz_convert(NY_TZ)
    else:
        daily.index = daily.index.tz_convert(NY_TZ)
    daily.index.name = "date"

    # Normaliza data do index para date (sem horario) para facilitar joins
    daily = daily[["open", "high", "low", "close", "volume"]].astype(float)

    minute = None
    if fetch_minute:
        try:
            mdf = yf.download(
                ticker, period="30d", interval="1m",
                auto_adjust=False, progress=False, threads=False,
            )
            if mdf is not None and len(mdf) > 0:
                if isinstance(mdf.columns, pd.MultiIndex):
                    mdf.columns = mdf.columns.get_level_values(0)
                mdf = mdf.rename(columns=str.lower)
                if mdf.index.tz is None:
                    mdf.index = mdf.index.tz_localize("UTC").tz_convert(NY_TZ)
                else:
                    mdf.index = mdf.index.tz_convert(NY_TZ)
                minute = mdf[["open", "high", "low", "close", "volume"]].astype(float)
        except Exception as e:
            log.warning(f"[{ticker}] falha no fetch minuto: {e}")

    meta = {
        "ticker": ticker,
        "asset_class": asset_class,
        "source": "yfinance",
        "daily_start": daily.index.min().isoformat(),
        "daily_end": daily.index.max().isoformat(),
        "n_days": len(daily),
        "n_minutes": len(minute) if minute is not None else 0,
    }

    return DataBundle(ticker=ticker, asset_class=asset_class,
                       daily=daily, minute=minute, meta=meta)


# =============================================================================
# 2. SESSION HANDLER
# =============================================================================

def extract_rth_from_minute(minute: pd.DataFrame, asset_class: str) -> pd.DataFrame:
    """
    A partir de barras de minuto, extrai para cada dia:
      rth_open, rth_close, rth_high, rth_low, rth_volume, ah_return

    Se nao houver minuto, retorna None (caller usa fallback diario).
    """
    if minute is None or len(minute) == 0:
        return None

    rule = SESSION_RULES[asset_class]
    rto = rule["rth_open"]
    rtc = rule["rth_close"]

    # Filtra apenas RTH (inclusive open, exclusive close+1min para pegar 15:59)
    m = minute.copy()
    m["t"] = m.index.time
    rth_mask = (m["t"] >= rto) & (m["t"] <= rtc)
    rth = m[rth_mask].copy()
    if len(rth) == 0:
        return None

    rth["date"] = rth.index.date
    agg = rth.groupby("date").agg(
        rth_open=("open", "first"),
        rth_close=("close", "last"),
        rth_high=("high", "max"),
        rth_low=("low", "min"),
        rth_volume=("volume", "sum"),
    )
    agg.index = pd.to_datetime(agg.index).tz_localize(NY_TZ)
    return agg


def build_session_frame(bundle: DataBundle) -> pd.DataFrame:
    """
    Gera o dataframe canonico dia-a-dia com colunas:
      open, high, low, close, volume (diario oficial)
      rth_return       = (close - open) / open
      rth_return_pts   = close - open
      overnight_return = (open_t - close_{t-1}) / close_{t-1}
      ah_return        = mesma coisa que overnight para equity;
                         para futures e o gap 16:00 -> 09:29 (aproximado por
                         overnight ja que yfinance daily para futuros ja inclui
                         sessao estendida)
      ext_return       = retorno extended hours (overnight)
      weekday          = 0=Mon ... 4=Fri
      day_type         = 'up' / 'down' / 'flat' (baseado em rth_return)

    Preferimos minuto quando disponivel (mais fiel), senao usamos daily OHLC.
    """
    daily = bundle.daily.copy()
    rth_from_minute = extract_rth_from_minute(bundle.minute, bundle.asset_class)

    df = daily.copy()
    # Integra RTH do minuto quando disponivel (sobrepoe open/close RTH-truncados)
    if rth_from_minute is not None and len(rth_from_minute) > 0:
        df = df.join(rth_from_minute, how="left")
        # Onde temos RTH do minuto, usa; onde nao, fallback para daily open/close
        df["rth_open_final"] = df["rth_open"].fillna(df["open"])
        df["rth_close_final"] = df["rth_close"].fillna(df["close"])
        df["source_rth"] = df["rth_open"].notna().map({True: "minute", False: "daily"})
    else:
        df["rth_open_final"] = df["open"]
        df["rth_close_final"] = df["close"]
        df["source_rth"] = "daily"

    df["rth_return"] = (df["rth_close_final"] - df["rth_open_final"]) / df["rth_open_final"]
    df["rth_return_pts"] = df["rth_close_final"] - df["rth_open_final"]
    df["rth_return_pct"] = df["rth_return"] * 100.0

    # Overnight/AH: do close anterior ate o open do dia atual
    prev_close = df["close"].shift(1)
    df["overnight_return"] = (df["open"] - prev_close) / prev_close
    df["overnight_return_pct"] = df["overnight_return"] * 100.0
    df["ah_return"] = df["overnight_return"]      # alias
    df["ext_return"] = df["overnight_return"]     # alias semantico para extended hours

    # Range
    df["range"] = df["high"] - df["low"]
    df["range_pct"] = df["range"] / df["open"]

    # Classificacao
    df["weekday"] = df.index.weekday                     # 0=Mon
    df["weekday_name"] = df.index.strftime("%A")
    df["day_type"] = np.where(df["rth_return"] > 0, "up",
                     np.where(df["rth_return"] < 0, "down", "flat"))

    # Magnitude em bps para ordenacao
    df["rth_abs_bps"] = (df["rth_return"].abs() * 10000.0).round(1)

    return df.dropna(subset=["rth_return"])


# =============================================================================
# 3. FEATURE ENGINEERING (gaps, ranges, dias consecutivos)
# =============================================================================

def add_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gap de abertura, classificacao e se fechou o gap."""
    prev_close = df["close"].shift(1)
    df = df.copy()
    df["gap"] = df["open"] - prev_close
    df["gap_pct"] = df["gap"] / prev_close
    df["gap_type"] = np.where(df["gap_pct"] > 0.001, "up",
                     np.where(df["gap_pct"] < -0.001, "down", "flat"))
    # Gap fechou se o low (gap up) ou high (gap down) tocou o close anterior
    gap_closed = np.where(
        df["gap_type"] == "up", df["low"] <= prev_close,
        np.where(df["gap_type"] == "down", df["high"] >= prev_close, False)
    )
    df["gap_closed"] = gap_closed
    return df


def add_range_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Expansao / compressao de range vs media movel."""
    df = df.copy()
    df[f"range_ma{window}"] = df["range"].rolling(window, min_periods=5).mean()
    df["range_expansion"] = df["range"] > df[f"range_ma{window}"]
    df["range_compression"] = df["range"] < df[f"range_ma{window}"] * 0.75
    return df


# =============================================================================
# 4. MOVING AVERAGE ENGINE
# =============================================================================

MA_WINDOWS = [5, 20, 50, 200]

def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona MA_5, MA_20, MA_50, MA_200 e flags acima/abaixo."""
    df = df.copy()
    for w in MA_WINDOWS:
        df[f"ma_{w}"] = df["close"].rolling(w, min_periods=max(5, w // 4)).mean()
        # Abre acima da MA (ponto de vista intraday entry)
        df[f"open_above_ma{w}"] = df["open"] > df[f"ma_{w}"].shift(1)
        df[f"close_above_ma{w}"] = df["close"] > df[f"ma_{w}"]
    return df


def ma_residency_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada MA (5, 20, 50, 200) calcula:
      dias acima, dias abaixo, % acima, % abaixo,
      sequencia atual (com sinal), maior seq acima, maior seq abaixo,
      retorno RTH medio quando abre/fecha acima/abaixo da MA
    """
    rows = []
    for w in MA_WINDOWS:
        col_close_above = f"close_above_ma{w}"
        col_open_above = f"open_above_ma{w}"
        sub = df.dropna(subset=[f"ma_{w}"])
        if len(sub) == 0:
            continue
        above = sub[col_close_above].sum()
        below = len(sub) - above
        pct_above = above / len(sub) if len(sub) else np.nan

        # Sequencias usando close_above_ma
        streaks = _compute_streaks(sub[col_close_above].astype(int).values)
        # streak atual: pega o ultimo
        cur_above = int(sub[col_close_above].iloc[-1])
        cur_len = streaks["current_length"] * (1 if cur_above else -1)
        max_above = streaks["max_positive"]
        max_below = streaks["max_negative"]

        # Retornos condicionais
        rth = sub["rth_return"]
        ret_open_above = rth[sub[col_open_above]].mean()
        ret_open_below = rth[~sub[col_open_above]].mean()
        ret_close_above = rth[sub[col_close_above]].mean()
        ret_close_below = rth[~sub[col_close_above]].mean()

        rows.append({
            "ma": f"MA_{w}",
            "n_obs": len(sub),
            "days_above": int(above),
            "days_below": int(below),
            "pct_above": round(pct_above * 100, 2),
            "pct_below": round((1 - pct_above) * 100, 2),
            "current_streak_signed": cur_len,
            "max_streak_above": max_above,
            "max_streak_below": max_below,
            "rth_ret_open_above": _pct(ret_open_above),
            "rth_ret_open_below": _pct(ret_open_below),
            "rth_ret_close_above": _pct(ret_close_above),
            "rth_ret_close_below": _pct(ret_close_below),
        })
    return pd.DataFrame(rows)


# =============================================================================
# 5. VOLATILITY ENGINE
# =============================================================================

def compute_volatility(df: pd.DataFrame, atr_window: int = 14,
                        rv_window: int = 21) -> pd.DataFrame:
    """
    Varios proxies de volatilidade. Modular para permitir trocar/adicionar.
      - true_range
      - atr (media movel de TR)
      - realized_vol (desvio padrao anualizado dos log-returns diarios)
      - close_to_close (abs dos retornos diarios)
      - range_intraday (high - low / open)
    """
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["true_range"] = tr
    df[f"atr_{atr_window}"] = tr.rolling(atr_window, min_periods=5).mean()

    log_ret = np.log(df["close"] / prev_close)
    df["log_return"] = log_ret
    df[f"rv_{rv_window}"] = log_ret.rolling(rv_window, min_periods=5).std() * math.sqrt(252)

    df["c2c_abs"] = log_ret.abs()
    df["range_intraday"] = (df["high"] - df["low"]) / df["open"]

    # Proxy default de vol "sobe/cai" = ATR
    df["vol_up"] = df[f"atr_{atr_window}"] > df[f"atr_{atr_window}"].shift(1)
    df["vol_down"] = df[f"atr_{atr_window}"] < df[f"atr_{atr_window}"].shift(1)
    return df


# =============================================================================
# 6. STREAK ENGINE
# =============================================================================

def _compute_streaks(flags: np.ndarray) -> dict:
    """
    Dada uma serie binaria (0/1), calcula:
      max_positive: maior sequencia de 1s
      max_negative: maior sequencia de 0s
      current_length: tamanho da sequencia atual (positiva se termina em 1)
    """
    if len(flags) == 0:
        return {"max_positive": 0, "max_negative": 0, "current_length": 0}
    max_pos = max_neg = 0
    cur_pos = cur_neg = 0
    for f in flags:
        if f:
            cur_pos += 1
            cur_neg = 0
            max_pos = max(max_pos, cur_pos)
        else:
            cur_neg += 1
            cur_pos = 0
            max_neg = max(max_neg, cur_neg)
    current_length = cur_pos if flags[-1] else cur_neg
    return {"max_positive": max_pos, "max_negative": max_neg,
            "current_length": current_length}


def streak_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sequencias de:
      - dias de alta / baixa do proprio ativo (com base em rth_return)
      - dias de vol subindo / caindo (com base em ATR)
    Alem de: retorno RTH medio apos N dias seguidos de alta/queda.
    """
    rows = []

    # Sequencias de dias up/down (RTH)
    flags_up = (df["rth_return"] > 0).astype(int).values
    s_up = _compute_streaks(flags_up)
    # streak atual com sinal
    cur_signed_up = s_up["current_length"] * (1 if flags_up[-1] else -1)

    rows.append({
        "serie": "rth_updown",
        "max_streak_up": s_up["max_positive"],
        "max_streak_down": s_up["max_negative"],
        "current_streak_signed": cur_signed_up,
        "n_obs": len(df),
    })

    # Sequencias de ATR subindo/caindo
    if "vol_up" in df.columns:
        vu = df["vol_up"].dropna().astype(int).values
        if len(vu) > 0:
            s_vol = _compute_streaks(vu)
            cur_signed_vol = s_vol["current_length"] * (1 if vu[-1] else -1)
            rows.append({
                "serie": "vol_atr_updown",
                "max_streak_up": s_vol["max_positive"],
                "max_streak_down": s_vol["max_negative"],
                "current_streak_signed": cur_signed_vol,
                "n_obs": len(vu),
            })
    return pd.DataFrame(rows)


def conditional_after_streaks(df: pd.DataFrame, max_k: int = 5) -> pd.DataFrame:
    """
    Retorno RTH medio do dia T+1 apos N dias seguidos de alta/queda em T-N..T.
    Tambem cruza com vol subindo/caindo.
    """
    r = df["rth_return"].values
    n = len(r)
    out = []
    # Constroi sequencia de dias "up" (diarios, mesmo valor)
    up_flags = (r > 0).astype(int)

    for k in range(2, max_k + 1):
        # Dias t onde os k dias anteriores (t-k..t-1) foram todos up
        mask_up = np.zeros(n, dtype=bool)
        mask_dn = np.zeros(n, dtype=bool)
        for t in range(k, n):
            win = up_flags[t - k:t]
            if win.sum() == k:
                mask_up[t] = True
            if win.sum() == 0:
                mask_dn[t] = True
        ret_after_up = r[mask_up].mean() if mask_up.sum() > 0 else np.nan
        ret_after_dn = r[mask_dn].mean() if mask_dn.sum() > 0 else np.nan
        out.append({
            "k_days": k,
            "n_after_up": int(mask_up.sum()),
            "rth_ret_after_up_pct": _pct(ret_after_up),
            "n_after_down": int(mask_dn.sum()),
            "rth_ret_after_down_pct": _pct(ret_after_dn),
        })

    # Mesmo cruzamento com vol
    if "vol_up" in df.columns and df["vol_up"].notna().any():
        vu = df["vol_up"].fillna(False).astype(int).values
        for k in range(2, 4):
            mask_volup = np.zeros(n, dtype=bool)
            mask_voldn = np.zeros(n, dtype=bool)
            for t in range(k, n):
                win = vu[t - k:t]
                if win.sum() == k:
                    mask_volup[t] = True
                if win.sum() == 0:
                    mask_voldn[t] = True
            ret_vu = r[mask_volup].mean() if mask_volup.sum() > 0 else np.nan
            ret_vd = r[mask_voldn].mean() if mask_voldn.sum() > 0 else np.nan
            out.append({
                "k_days": f"vol_up_{k}",
                "n_after_up": int(mask_volup.sum()),
                "rth_ret_after_up_pct": _pct(ret_vu),
                "n_after_down": int(mask_voldn.sum()),
                "rth_ret_after_down_pct": _pct(ret_vd),
            })
    return pd.DataFrame(out)


# =============================================================================
# 7. STATS ENGINE (weekday + distribuicao)
# =============================================================================

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

def _pct(x) -> float:
    """Converte fracao em pct com 2 decimais, tratando nan."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return np.nan
    return round(float(x) * 100, 3)


def weekday_stats(df: pd.DataFrame, col: str = "rth_return") -> pd.DataFrame:
    """Estatisticas completas por dia da semana."""
    g = df.groupby("weekday_name")[col]
    rows = []
    for name in WEEKDAY_ORDER:
        if name not in g.groups:
            continue
        s = g.get_group(name)
        n = len(s)
        wins = s[s > 0]
        losses = s[s < 0]
        hit = len(wins) / n if n else np.nan
        payoff = (wins.mean() / abs(losses.mean())) if (len(losses) and losses.mean() != 0) else np.nan
        profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.nan
        expectancy = hit * (wins.mean() if len(wins) else 0) - (1 - hit) * (abs(losses.mean()) if len(losses) else 0)
        skew = kurt = np.nan
        if HAS_SCIPY and n >= 10:
            try:
                skew = scistats.skew(s, bias=False)
                kurt = scistats.kurtosis(s, bias=False, fisher=True)
            except Exception:
                pass
        rows.append({
            "weekday": name,
            "n": n,
            "hit_rate_pct": _pct(hit),
            "loss_rate_pct": _pct(1 - hit) if not np.isnan(hit) else np.nan,
            "mean_pct": _pct(s.mean()),
            "median_pct": _pct(s.median()),
            "std_pct": _pct(s.std()),
            "best_pct": _pct(s.max()),
            "worst_pct": _pct(s.min()),
            "avg_win_pct": _pct(wins.mean()) if len(wins) else np.nan,
            "avg_loss_pct": _pct(losses.mean()) if len(losses) else np.nan,
            "payoff_ratio": round(payoff, 3) if pd.notna(payoff) else np.nan,
            "profit_factor": round(profit_factor, 3) if pd.notna(profit_factor) else np.nan,
            "expectancy_pct": _pct(expectancy),
            "skew": round(skew, 3) if pd.notna(skew) else np.nan,
            "kurtosis": round(kurt, 3) if pd.notna(kurt) else np.nan,
            "cum_return_pct": _pct((1 + s).prod() - 1),
        })
    return pd.DataFrame(rows)


def up_down_by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela clara: quantos subiram/cairam por dia da semana."""
    rows = []
    for name in WEEKDAY_ORDER:
        s = df[df["weekday_name"] == name]["rth_return"]
        if len(s) == 0:
            continue
        up = s[s > 0]
        dn = s[s < 0]
        rows.append({
            "weekday": name,
            "n_total": len(s),
            "n_up": len(up),
            "n_down": len(dn),
            "pct_up": _pct(len(up) / len(s)),
            "pct_down": _pct(len(dn) / len(s)),
            "avg_up_pct": _pct(up.mean()) if len(up) else np.nan,
            "avg_down_pct": _pct(dn.mean()) if len(dn) else np.nan,
        })
    return pd.DataFrame(rows)


def rolling_weekday_average(df: pd.DataFrame, windows: list = [20, 60, 252]) -> pd.DataFrame:
    """
    Media rolante do retorno RTH por weekday para detectar mudanca de padrao.
    Retorna dataframe longo: date, weekday, window, mean_return.
    """
    frames = []
    for w in windows:
        for name in WEEKDAY_ORDER:
            sub = df[df["weekday_name"] == name].copy()
            if len(sub) == 0:
                continue
            # Rolling mean ao longo dos dias desse weekday (nao dias totais)
            sub = sub.sort_index()
            sub[f"rolling_mean"] = sub["rth_return"].rolling(max(3, w // 5), min_periods=3).mean()
            sub["weekday"] = name
            sub["window_days"] = w
            frames.append(sub[["weekday", "window_days", "rolling_mean"]].reset_index())
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


# =============================================================================
# 8. BACKTEST ENGINE (equity, DD, metricas)
# =============================================================================

@dataclass
class BacktestResult:
    equity: pd.Series
    drawdown: pd.Series
    metrics: dict


def backtest_rth(df: pd.DataFrame) -> BacktestResult:
    """
    Estrategia: long no rth_open, sai no rth_close. 1x alavancagem.
    PnL composto diariamente. Equity = produto cumulativo de (1 + rth_return).
    """
    r = df["rth_return"].fillna(0.0)
    equity = (1 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0

    n = len(r)
    years = n / 252.0 if n else 0.0
    total_return = equity.iloc[-1] - 1 if n else 0.0
    cagr = (equity.iloc[-1]) ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = r.std() * math.sqrt(252) if n > 1 else np.nan
    sharpe = (r.mean() * 252) / ann_vol if ann_vol and ann_vol > 0 else np.nan
    downside = r[r < 0].std() * math.sqrt(252)
    sortino = (r.mean() * 252) / downside if downside and downside > 0 else np.nan
    max_dd = dd.min() if len(dd) else np.nan
    calmar = cagr / abs(max_dd) if max_dd and max_dd != 0 else np.nan
    wins = r[r > 0]
    losses = r[r < 0]
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan
    hit = (r > 0).sum() / n if n else np.nan

    # Tempo de recuperacao do max DD
    recovery_days = np.nan
    if len(dd) and max_dd < 0:
        trough_idx = dd.idxmin()
        post = equity.loc[trough_idx:]
        peak_at_trough = peak.loc[trough_idx]
        recovered = post[post >= peak_at_trough]
        if len(recovered) > 0:
            recovery_days = (recovered.index[0] - trough_idx).days

    metrics = {
        "n_days": n,
        "total_return_pct": _pct(total_return),
        "cagr_pct": _pct(cagr),
        "ann_vol_pct": _pct(ann_vol),
        "sharpe": round(sharpe, 3) if pd.notna(sharpe) else np.nan,
        "sortino": round(sortino, 3) if pd.notna(sortino) else np.nan,
        "calmar": round(calmar, 3) if pd.notna(calmar) else np.nan,
        "max_drawdown_pct": _pct(max_dd),
        "avg_drawdown_pct": _pct(dd[dd < 0].mean()) if (dd < 0).any() else np.nan,
        "recovery_days_from_max_dd": recovery_days,
        "profit_factor": round(profit_factor, 3) if pd.notna(profit_factor) else np.nan,
        "hit_rate_pct": _pct(hit),
        "best_day_pct": _pct(r.max()),
        "worst_day_pct": _pct(r.min()),
    }
    return BacktestResult(equity=equity, drawdown=dd, metrics=metrics)


# =============================================================================
# 9. REGIME / Z-SCORE / GAP / MONTH
# =============================================================================

def regime_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regime simples via posicao relativa:
      uptrend: close > MA20 > MA50
      downtrend: close < MA20 < MA50
      sideways: caso contrario
    Retorna stats de rth_return por regime.
    """
    d = df.dropna(subset=["ma_20", "ma_50"]).copy()
    cond_up = (d["close"] > d["ma_20"]) & (d["ma_20"] > d["ma_50"])
    cond_dn = (d["close"] < d["ma_20"]) & (d["ma_20"] < d["ma_50"])
    d["regime"] = np.where(cond_up, "uptrend",
                   np.where(cond_dn, "downtrend", "sideways"))
    g = d.groupby("regime")["rth_return"]
    out = g.agg(["count", "mean", "median", "std"]).reset_index()
    out.columns = ["regime", "n", "mean_pct", "median_pct", "std_pct"]
    for c in ["mean_pct", "median_pct", "std_pct"]:
        out[c] = (out[c] * 100).round(3)
    out["hit_rate_pct"] = [
        _pct((g.get_group(r) > 0).sum() / len(g.get_group(r)))
        for r in out["regime"]
    ]
    return out


def zscore_continuation_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score do retorno diario (rolling 60d).
    Dias com |z| > 2 sao classificados como extremos.
    Mede retorno RTH do dia seguinte condicionado ao sinal do extremo.
    """
    r = df["rth_return"]
    mu = r.rolling(60, min_periods=20).mean()
    sd = r.rolling(60, min_periods=20).std()
    z = (r - mu) / sd
    next_r = r.shift(-1)
    buckets = [
        ("extreme_up (z>2)", z > 2),
        ("extreme_down (z<-2)", z < -2),
        ("strong_up (1<z<=2)", (z > 1) & (z <= 2)),
        ("strong_down (-2<=z<-1)", (z < -1) & (z >= -2)),
    ]
    rows = []
    for name, mask in buckets:
        s = next_r[mask].dropna()
        if len(s) < 3:
            continue
        rows.append({
            "bucket": name,
            "n": len(s),
            "next_day_mean_pct": _pct(s.mean()),
            "next_day_median_pct": _pct(s.median()),
            "hit_rate_pct": _pct((s > 0).sum() / len(s)),
        })
    return pd.DataFrame(rows)


def gap_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Retorno RTH condicionado ao tipo de gap de abertura."""
    rows = []
    for gt in ["up", "down", "flat"]:
        s = df[df["gap_type"] == gt]["rth_return"]
        if len(s) < 3:
            continue
        closed = df[(df["gap_type"] == gt) & df["gap_closed"]]
        rows.append({
            "gap_type": gt,
            "n": len(s),
            "rth_mean_pct": _pct(s.mean()),
            "rth_median_pct": _pct(s.median()),
            "rth_hit_pct": _pct((s > 0).sum() / len(s)),
            "gap_close_rate_pct": _pct(len(closed) / len(s)) if len(s) else np.nan,
        })
    return pd.DataFrame(rows)


def monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Estatisticas por mes do ano + primeiro/ultimo dia util."""
    d = df.copy()
    d["month"] = d.index.month
    d["is_first_bday"] = d.index.to_series().groupby([d.index.year, d.index.month]).transform(
        lambda s: s == s.min())
    d["is_last_bday"] = d.index.to_series().groupby([d.index.year, d.index.month]).transform(
        lambda s: s == s.max())

    by_month = d.groupby("month")["rth_return"].agg(["count", "mean", "std"]).reset_index()
    by_month.columns = ["month", "n", "mean_pct", "std_pct"]
    by_month["mean_pct"] = (by_month["mean_pct"] * 100).round(3)
    by_month["std_pct"] = (by_month["std_pct"] * 100).round(3)
    by_month["hit_rate_pct"] = by_month["month"].map(
        lambda m: _pct((d[d["month"] == m]["rth_return"] > 0).mean())
    )

    # Primeiro e ultimo dia util
    first = d[d["is_first_bday"]]["rth_return"]
    last = d[d["is_last_bday"]]["rth_return"]
    turn_rows = pd.DataFrame([
        {"bucket": "first_bday_of_month", "n": len(first),
         "mean_pct": _pct(first.mean()), "hit_rate_pct": _pct((first > 0).mean())},
        {"bucket": "last_bday_of_month", "n": len(last),
         "mean_pct": _pct(last.mean()), "hit_rate_pct": _pct((last > 0).mean())},
    ])
    return by_month, turn_rows


# =============================================================================
# 10. VISUALIZATION
# =============================================================================

def save_plot(fig, path: Path):
    fig.savefig(path)
    plt.close(fig)


def plot_weekday_bars(wdf: pd.DataFrame, ticker: str, out: Path):
    """Bar chart de retorno medio por weekday."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    d = wdf.set_index("weekday").reindex([x for x in WEEKDAY_ORDER if x in wdf["weekday"].values])
    colors = [COLORS["up"] if v > 0 else COLORS["down"] for v in d["mean_pct"]]
    ax.bar(d.index, d["mean_pct"], color=colors, edgecolor=COLORS["grid"])
    ax.axhline(0, color=COLORS["fg"], linewidth=0.8)
    ax.set_title(f"{ticker} | Retorno medio RTH por weekday (%)")
    ax.set_ylabel("% medio")
    for i, (idx, row) in enumerate(d.iterrows()):
        ax.text(i, row["mean_pct"], f"{row['mean_pct']:.2f}%\nn={int(row['n'])}",
                ha="center",
                va="bottom" if row["mean_pct"] >= 0 else "top",
                fontsize=8, color=COLORS["fg"])
    ax.grid(True, axis="y")
    save_plot(fig, out)


def plot_weekday_hitrate(wdf: pd.DataFrame, ticker: str, out: Path):
    """Bar chart frequencia de alta/queda por weekday."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    d = wdf.set_index("weekday").reindex([x for x in WEEKDAY_ORDER if x in wdf["weekday"].values])
    x = np.arange(len(d))
    ax.bar(x - 0.2, d["hit_rate_pct"], width=0.4, label="Hit rate (%)",
           color=COLORS["up"])
    ax.bar(x + 0.2, d["loss_rate_pct"], width=0.4, label="Loss rate (%)",
           color=COLORS["down"])
    ax.axhline(50, color=COLORS["neutral"], linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(d.index)
    ax.set_title(f"{ticker} | Frequencia de alta vs queda por weekday")
    ax.set_ylabel("%")
    ax.legend()
    ax.grid(True, axis="y")
    save_plot(fig, out)


def plot_equity_curve(bt: BacktestResult, ticker: str, out: Path):
    """Equity curve da estrategia RTH."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(bt.equity.index, bt.equity.values, color=COLORS["accent2"], linewidth=1.3)
    ax1.set_title(f"{ticker} | Equity curve RTH (open->close) | "
                  f"Sharpe={bt.metrics['sharpe']} | "
                  f"MaxDD={bt.metrics['max_drawdown_pct']}%")
    ax1.set_ylabel("Equity (norm=1)")
    ax1.grid(True)

    ax2.fill_between(bt.drawdown.index, bt.drawdown.values * 100, 0,
                      color=COLORS["down"], alpha=0.6)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(True)

    plt.tight_layout()
    save_plot(fig, out)


def plot_histogram(df: pd.DataFrame, ticker: str, out: Path):
    """Histograma dos retornos RTH."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    r = df["rth_return"] * 100
    ax.hist(r, bins=60, color=COLORS["accent2"], edgecolor=COLORS["grid"], alpha=0.85)
    ax.axvline(r.mean(), color=COLORS["accent"], linestyle="--", label=f"Mean {r.mean():.2f}%")
    ax.axvline(r.median(), color=COLORS["up"], linestyle="--", label=f"Median {r.median():.2f}%")
    ax.set_title(f"{ticker} | Distribuicao retorno RTH (%)")
    ax.set_xlabel("Retorno %")
    ax.legend()
    ax.grid(True, axis="y")
    save_plot(fig, out)


def plot_heatmap_weekday_month(df: pd.DataFrame, ticker: str, out: Path):
    """Heatmap retorno medio por weekday x mes."""
    d = df.copy()
    d["month"] = d.index.month
    pivot = d.pivot_table(index="weekday_name", columns="month",
                           values="rth_return", aggfunc="mean") * 100
    pivot = pivot.reindex([x for x in WEEKDAY_ORDER if x in pivot.index])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    cmap = LinearSegmentedColormap.from_list("rg", [COLORS["down"], "#222", COLORS["up"]])
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{ticker} | Heatmap retorno medio RTH (%) — weekday x mes")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=COLORS["fg"], fontsize=8)
    fig.colorbar(im, ax=ax, label="% medio")
    save_plot(fig, out)


def plot_ma_residency(ma_df: pd.DataFrame, ticker: str, out: Path):
    """Bar chart do % de dias acima/abaixo de cada MA."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(ma_df))
    ax.bar(x - 0.2, ma_df["pct_above"], width=0.4, color=COLORS["up"], label="% acima")
    ax.bar(x + 0.2, ma_df["pct_below"], width=0.4, color=COLORS["down"], label="% abaixo")
    ax.set_xticks(x)
    ax.set_xticklabels(ma_df["ma"])
    ax.axhline(50, color=COLORS["neutral"], linestyle="--", linewidth=0.8)
    ax.set_title(f"{ticker} | Permanencia acima/abaixo das medias")
    ax.set_ylabel("%")
    ax.legend()
    ax.grid(True, axis="y")
    save_plot(fig, out)


def plot_streak_distribution(df: pd.DataFrame, ticker: str, out: Path):
    """Histograma de duracao das sequencias up e down."""
    flags = (df["rth_return"] > 0).astype(int).values
    # conta runs
    runs_up, runs_dn = [], []
    cur = 1
    for i in range(1, len(flags)):
        if flags[i] == flags[i - 1]:
            cur += 1
        else:
            (runs_up if flags[i - 1] == 1 else runs_dn).append(cur)
            cur = 1
    (runs_up if flags[-1] == 1 else runs_dn).append(cur)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.arange(1, max(max(runs_up or [1]), max(runs_dn or [1])) + 2) - 0.5
    ax.hist(runs_up, bins=bins, alpha=0.7, color=COLORS["up"], label=f"Runs up (n={len(runs_up)})")
    ax.hist(runs_dn, bins=bins, alpha=0.7, color=COLORS["down"], label=f"Runs down (n={len(runs_dn)})")
    ax.set_title(f"{ticker} | Distribuicao de duracao de sequencias RTH")
    ax.set_xlabel("Dias consecutivos")
    ax.legend()
    ax.grid(True, axis="y")
    save_plot(fig, out)


# =============================================================================
# 11. REPORT BUILDER (ZIP)
# =============================================================================

def build_summary(ticker: str, bundle: DataBundle, df: pd.DataFrame,
                   bt: BacktestResult, wstats: pd.DataFrame,
                   updown: pd.DataFrame, ma_stats: pd.DataFrame,
                   streaks: pd.DataFrame) -> str:
    """Texto-resumo honesto, com tamanho de amostra e flags de baixa significancia."""
    lines = []
    push = lines.append
    push("=" * 72)
    push(f"SESSION STATS REPORT | {ticker}")
    push(f"Asset class: {bundle.asset_class}")
    push(f"Periodo: {bundle.meta['daily_start'][:10]} -> {bundle.meta['daily_end'][:10]}")
    push(f"Observacoes diarias: {bundle.meta['n_days']}")
    push(f"Fonte RTH: {df['source_rth'].iloc[-1] if 'source_rth' in df.columns else 'daily'}")
    push("=" * 72)
    push("")
    push("[BACKTEST RTH open->close]")
    for k, v in bt.metrics.items():
        push(f"  {k:30s}: {v}")
    push("")
    push("[WEEKDAY STATS — RTH]")
    push(wstats.to_string(index=False))
    push("")
    push("[SUBIU / CAIU POR WEEKDAY]")
    push(updown.to_string(index=False))
    push("")
    push("[MEDIAS MOVEIS — PERMANENCIA]")
    push(ma_stats.to_string(index=False))
    push("")
    push("[SEQUENCIAS]")
    push(streaks.to_string(index=False))
    push("")
    # Flags honestos
    push("[NOTAS DE SIGNIFICANCIA]")
    for _, row in wstats.iterrows():
        if row["n"] < 30:
            push(f"  ! {row['weekday']}: amostra pequena (n={row['n']}), leitura fraca")
    push("")
    push("OBS: estatistica nao e edge automaticamente. Use com contexto.")
    push("=" * 72)
    return "\n".join(lines)


def export_zip(out_dir: Path, zip_path: Path):
    """Empacota tudo da pasta de saida em um zip."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.relative_to(out_dir))
    log.info(f"ZIP gerado: {zip_path}")


# =============================================================================
# 13. NOMURA SECTION — Options PnL + Skew Percentiles + Dynamic AUM Flows
# =============================================================================
#
# Replica o framework do "Nomura Cross Asset TAKES":
#   1. SPX Daily Options PnL Summary (short straddle/strangle/25d combos)
#   2. SPX Historical Percentiles (skew left-tail e right-tail)
#   3. US Equities Systematic Flows (Vol Control + CTA + Risk Parity)
#      com AUM DINAMICO — AUM evolui com PnL, nao e constante.
#
# Observacao sobre AUM dinamico (ponto crucial levantado pelo usuario):
#   Modelo tradicional assume AUM fixo (ex: $150B pra Vol Control sempre).
#   Na pratica: AUM_{t+1} = AUM_t * (1 + exposure_t * r_t)
#   Isso muda materialmente o tamanho dos flows ao longo do tempo.
# =============================================================================

# ---- 13.1 — Options data loader (VIX, SKEW indices) ----

def load_vol_indices(years: int = 5) -> pd.DataFrame:
    """
    Carrega VIX e SKEW.
      BQuant: VIX Index, SKEW Index
      Local:  ^VIX, ^SKEW via yfinance
      - VIX: ATM IV 30d (%)
      - SKEW: indice CBOE de tail risk. SKEW=100 -> sem skew.
              Geralmente 100-150. Maior = mais tail risk no put.
    Retorna dataframe diario alinhado.
    """
    # Tenta BQL primeiro (BQuant)
    if HAS_BQL:
        b1 = load_ticker_bql("^VIX", years=years)
        b2 = load_ticker_bql("^SKEW", years=years)
        if b1 is not None and b2 is not None:
            out = pd.DataFrame({
                "vix": b1.daily["close"],
                "skew": b2.daily["close"],
            }).dropna()
            return out

    if not HAS_YF:
        return pd.DataFrame()
    end = datetime.now()
    start = end - timedelta(days=int(years * 365.25))
    frames = {}
    for t, name in [("^VIX", "vix"), ("^SKEW", "skew")]:
        try:
            d = yf.download(t, start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"),
                             interval="1d", progress=False, threads=False,
                             auto_adjust=False)
            if d is None or len(d) == 0:
                continue
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            frames[name] = d["Close"].astype(float)
        except Exception as e:
            log.warning(f"falha ao baixar {t}: {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.DataFrame(frames)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC").tz_convert(NY_TZ)
    else:
        out.index = out.index.tz_convert(NY_TZ)
    return out


def approximate_25d_ivs(atm_iv: pd.Series, skew_idx: pd.Series) -> pd.DataFrame:
    """
    Aproxima 25d put IV e 25d call IV a partir de ATM IV (VIX) e SKEW index.

    Formulacao pragmatica (CBOE SKEW convertido em delta de IV):
      skew_premium = (skew_index - 100) / 10    # ~0 a ~5
      iv_25dP = atm_iv * (1 + 0.05 * skew_premium)     # put enriquece com SKEW
      iv_25dC = atm_iv * (1 - 0.02 * skew_premium)     # call desconta de leve

    Calibracao baseada em empiricos SPX 2015-2024. Nao e exato — e um proxy
    razoavel quando nao temos chain completa. Se tiver dados reais da curva
    (Bloomberg OVDV ou OptionMetrics), plugar aqui no lugar.
    """
    skew_prem = (skew_idx - 100) / 10.0
    iv_25dP = atm_iv * (1 + 0.05 * skew_prem)
    iv_25dC = atm_iv * (1 - 0.02 * skew_prem)
    return pd.DataFrame({
        "atm_iv": atm_iv,
        "iv_25dP": iv_25dP,
        "iv_25dC": iv_25dC,
        "skew_25dP_25dC": iv_25dP / iv_25dC,
        "skew_25dC_atm": iv_25dC / atm_iv,
    })


# ---- 13.2 — Black-Scholes mini-engine (standalone, nao toca no monolito) ----

def _norm_cdf(x):
    return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x * x / math.pi)))

def _norm_pdf(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def bs_price(S, K, T, r, sigma, kind="C"):
    """Black-Scholes para 1 opcao. T em anos, sigma anualizada."""
    if T <= 0 or sigma <= 0:
        if kind == "C":
            return max(S - K, 0)
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "C":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def strike_from_delta(S, T, r, sigma, delta_target, kind="P"):
    """
    Inverte BS para achar strike correspondente ao delta target.
    Usa aproximacao por iteracao simples. 25d put => delta=-0.25.
    """
    if T <= 0 or sigma <= 0:
        return S
    # Chute inicial: strike ~= S * exp(sign * 0.67 * sigma * sqrt(T))
    sign = -1 if kind == "P" else 1
    K = S * math.exp(sign * 0.67 * sigma * math.sqrt(T))
    return float(K)


# ---- 13.3 — Daily Options Strategy PnL ----

STRATEGY_LABELS = [
    "Selling Daily ATM Straddle",
    "Selling Daily ATM Call",
    "Selling Daily ATM Put",
    "Selling Daily Strangle",
    "Selling Daily 25d Call",
    "Selling Daily 25d Put",
    "Selling Daily Straddle, Long Strangle",
    "Sell 25d Put, Buy 25d Call",
    "Sell 25d Call, Buy 25d Put",
    "Stock (Long)",
]


def compute_daily_options_pnl(spx_df: pd.DataFrame,
                                iv_df: pd.DataFrame,
                                tenor_days: int = 1,
                                r: float = 0.04) -> pd.DataFrame:
    """
    Calcula PnL diario de cada estrategia em % do spot.

    Regras:
      - "Daily" = tenor de 1 dia util (T = 1/252). Rolled diariamente.
      - Premium vendido no open, PnL marcado no close do mesmo dia.
      - ATM: strike = S_open. 25d: strike do delta 0.25.
      - PnL_short_opt = premium_collected - payoff_intrinsic(K, S_close)
      - PnL_long_opt  = -premium_paid + payoff_intrinsic(K, S_close)
      - Todos os valores em % do spot_open.
    """
    if len(spx_df) == 0 or len(iv_df) == 0:
        return pd.DataFrame()

    # Alinha IV com spot
    df = spx_df[["open", "close"]].copy()
    df = df.join(iv_df, how="inner").dropna()
    if len(df) == 0:
        return pd.DataFrame()

    T = tenor_days / 252.0
    rows = []
    for dt, row in df.iterrows():
        S0 = row["open"]
        ST = row["close"]
        # ATM
        iv_atm = row["atm_iv"] / 100.0
        iv_p25 = row["iv_25dP"] / 100.0
        iv_c25 = row["iv_25dC"] / 100.0

        # Premia ATM
        p_atm_call = bs_price(S0, S0, T, r, iv_atm, "C")
        p_atm_put = bs_price(S0, S0, T, r, iv_atm, "P")
        p_atm_straddle = p_atm_call + p_atm_put

        # Strikes 25d
        K_25dP = strike_from_delta(S0, T, r, iv_p25, 0.25, "P")
        K_25dC = strike_from_delta(S0, T, r, iv_c25, 0.25, "C")
        p_25dP = bs_price(S0, K_25dP, T, r, iv_p25, "P")
        p_25dC = bs_price(S0, K_25dC, T, r, iv_c25, "C")
        p_25d_strangle = p_25dP + p_25dC

        # Payoffs no close
        pay_atm_call = max(ST - S0, 0)
        pay_atm_put = max(S0 - ST, 0)
        pay_atm_straddle = pay_atm_call + pay_atm_put
        pay_25dP = max(K_25dP - ST, 0)
        pay_25dC = max(ST - K_25dC, 0)
        pay_25d_strangle = pay_25dP + pay_25dC

        # PnL por estrategia (em pontos de spot)
        pnl_atm_straddle = p_atm_straddle - pay_atm_straddle
        pnl_atm_call = p_atm_call - pay_atm_call
        pnl_atm_put = p_atm_put - pay_atm_put
        pnl_strangle = p_25d_strangle - pay_25d_strangle
        pnl_25dC = p_25dC - pay_25dC
        pnl_25dP = p_25dP - pay_25dP
        # Short straddle + long strangle (short vol no centro, long asas)
        pnl_iron_butterfly_short = pnl_atm_straddle - (p_25d_strangle - pay_25d_strangle)
        # Risk reversal (bullish): sell 25dP, buy 25dC
        pnl_rr_bull = pnl_25dP - (p_25dC - pay_25dC)
        # Risk reversal (bearish): sell 25dC, buy 25dP
        pnl_rr_bear = pnl_25dC - (p_25dP - pay_25dP)
        # Long stock
        pnl_stock = ST - S0

        # Normaliza para % do spot open
        rows.append({
            "date": dt,
            "S0": S0, "ST": ST,
            "Selling Daily ATM Straddle": pnl_atm_straddle / S0,
            "Selling Daily ATM Call": pnl_atm_call / S0,
            "Selling Daily ATM Put": pnl_atm_put / S0,
            "Selling Daily Strangle": pnl_strangle / S0,
            "Selling Daily 25d Call": pnl_25dC / S0,
            "Selling Daily 25d Put": pnl_25dP / S0,
            "Selling Daily Straddle, Long Strangle": pnl_iron_butterfly_short / S0,
            "Sell 25d Put, Buy 25d Call": pnl_rr_bull / S0,
            "Sell 25d Call, Buy 25d Put": pnl_rr_bear / S0,
            "Stock (Long)": pnl_stock / S0,
        })

    pnl = pd.DataFrame(rows).set_index("date")
    return pnl


def options_pnl_summary_table(pnl: pd.DataFrame) -> pd.DataFrame:
    """
    Tabela no formato Nomura: linhas = estrategias, colunas = [1d, 10d, 20d, 60d, ytd, 1y].
    Valores em % (cumulativo simples, nao composto — e consistente com PnL em % do notional).
    """
    if len(pnl) == 0:
        return pd.DataFrame()
    strat_cols = [c for c in STRATEGY_LABELS if c in pnl.columns]
    today = pnl.index[-1]
    ytd_start = pd.Timestamp(year=today.year, month=1, day=1, tz=today.tz)
    horizons = {
        "1d": pnl.tail(1),
        "10d": pnl.tail(10),
        "20d": pnl.tail(20),
        "60d": pnl.tail(60),
        "ytd": pnl[pnl.index >= ytd_start],
        "1y": pnl.tail(252),
    }
    out = pd.DataFrame(index=strat_cols, columns=list(horizons.keys()))
    for h, sub in horizons.items():
        out[h] = [_pct(sub[c].sum()) for c in strat_cols]
    out.index.name = "Strategy"
    return out


def options_sharpe_table(pnl: pd.DataFrame) -> pd.DataFrame:
    """
    Sharpe anualizado por horizonte e estrategia.
    Usa retornos diarios da estrategia / std * sqrt(252).
    """
    if len(pnl) == 0:
        return pd.DataFrame()
    strat_cols = [c for c in STRATEGY_LABELS if c in pnl.columns]
    horizons = {"10d": 10, "20d": 20, "60d": 60, "ytd": None, "1y": 252}
    today = pnl.index[-1]
    ytd_start = pd.Timestamp(year=today.year, month=1, day=1, tz=today.tz)
    out = pd.DataFrame(index=strat_cols, columns=list(horizons.keys()))
    for h, n in horizons.items():
        sub = pnl[pnl.index >= ytd_start] if h == "ytd" else pnl.tail(n)
        for c in strat_cols:
            s = sub[c].dropna()
            if len(s) < 5 or s.std() == 0:
                out.at[c, h] = np.nan
            else:
                out.at[c, h] = round(s.mean() / s.std() * math.sqrt(252), 2)
    out.index.name = "Strategy"
    return out


# ---- 13.4 — Skew Percentiles (left-tail / right-tail) ----

def skew_percentiles(iv_df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Calcula percentil rolante de:
      - skew_25dP_25dC (left-tail indicator)
      - skew_25dC_atm  (right-tail indicator)
      - atm_iv
    Percentil alto em 25dP/25dC = OVERHEDGED FOR LEFT-TAIL.
    Percentil baixo em 25dC/ATM = UNDERHEDGED FOR RIGHT-TAIL.
    """
    if len(iv_df) == 0:
        return pd.DataFrame()
    out = iv_df.copy()
    for c in ["skew_25dP_25dC", "skew_25dC_atm", "atm_iv"]:
        pct = out[c].rolling(window, min_periods=60).apply(
            lambda x: (x.rank(pct=True).iloc[-1]) * 100, raw=False
        )
        out[f"{c}_pctile"] = pct
    # Classificacao textual do estado atual
    last = out.dropna().iloc[-1] if len(out.dropna()) else None
    if last is not None:
        ltp = last["skew_25dP_25dC_pctile"]
        rtp = last["skew_25dC_atm_pctile"]
        if ltp >= 80:
            out.attrs["left_tail_status"] = "OVERHEDGED FOR LEFT-TAIL"
        elif ltp <= 20:
            out.attrs["left_tail_status"] = "UNDERHEDGED FOR LEFT-TAIL"
        else:
            out.attrs["left_tail_status"] = "NEUTRAL LEFT-TAIL"
        if rtp <= 20:
            out.attrs["right_tail_status"] = "UNDERHEDGED FOR RIGHT-TAIL"
        elif rtp >= 80:
            out.attrs["right_tail_status"] = "OVERHEDGED FOR RIGHT-TAIL"
        else:
            out.attrs["right_tail_status"] = "NEUTRAL RIGHT-TAIL"
    return out


# ---- 13.5 — Dynamic AUM Systematic Flows ----

@dataclass
class FlowConfig:
    """Parametros dos fundos sistematicos (base, nao AUM dinamico)."""
    vc_aum_base: float = 150e9        # Vol Control (target 10%)
    vc_target_vol: float = 0.10
    vc_max_lev: float = 2.0
    vc_floor: float = 0.20
    cta_aum_base: float = 85e9        # CTA equity allocation (340B * 25%)
    rp_aum_base: float = 200e9        # Risk Parity total
    rp_eq_weight: float = 0.35         # equity share (~35% do RP notional)


def compute_dynamic_flows(spx_df: pd.DataFrame,
                           config: FlowConfig = None,
                           rv_window: int = 21) -> pd.DataFrame:
    """
    Calcula flows DIARIOS com AUM EVOLUINDO por PnL.

    Fundamentalmente: cada fundo comeca com AUM_base e a cada dia:
      1. Calcula exposure alvo (em % de AUM)
      2. Flow_t = AUM_t * (exposure_t - exposure_{t-1})
      3. PnL_t = AUM_t * exposure_t * r_equity_t
      4. AUM_{t+1} = AUM_t + PnL_t + external_flows (externo=0 aqui)

    Estrategias (exposure em % da carteira):
      - Vol Control: target_vol / realized_vol, capado
      - CTA: trend score de cruzamentos de MA (-2 a +2)
      - Risk Parity: inversamente proporcional a vol, sem reversao de sinal
    """
    if config is None:
        config = FlowConfig()
    df = spx_df.copy()
    r = df["close"].pct_change().fillna(0)
    # Realized vol (anualizada)
    rv = r.rolling(rv_window, min_periods=5).std() * math.sqrt(252)
    rv = rv.fillna(method="ffill").fillna(0.15)

    # ---- Vol Control ----
    vc_exp = (config.vc_target_vol / rv).clip(
        lower=config.vc_floor, upper=config.vc_max_lev
    )
    # Ajuste diario capado a 25%
    vc_exp_smooth = vc_exp.copy()
    for i in range(1, len(vc_exp)):
        prev = vc_exp_smooth.iloc[i - 1]
        cur = vc_exp_smooth.iloc[i]
        delta = cur - prev
        if abs(delta) > 0.25:
            vc_exp_smooth.iloc[i] = prev + np.sign(delta) * 0.25

    # ---- CTA ----
    ma_pairs = [(5, 20), (10, 60), (20, 120), (50, 200)]
    cta_score = pd.Series(0.0, index=df.index)
    for short, long in ma_pairs:
        s = df["close"].rolling(short, min_periods=2).mean()
        l = df["close"].rolling(long, min_periods=2).mean()
        cta_score = cta_score + np.sign(s - l) / len(ma_pairs)
    # Normaliza pelo vol target (inverse vol sizing)
    cta_target_vol = 0.15
    cta_exp = (cta_score * (cta_target_vol / rv)).clip(-2.0, 2.0)

    # ---- Risk Parity ----
    # Equity weight inversamente proporcional a vol. Simplificacao: RP_exposure =
    # (inv_vol_eq / inv_vol_total) onde inv_vol_total e constante (assumindo bonds+comm ~= static)
    rp_base_weight = config.rp_eq_weight
    inv_vol = 1.0 / rv
    inv_vol_ma = inv_vol.rolling(60, min_periods=10).mean()
    rp_exp = rp_base_weight * (inv_vol / inv_vol_ma).clip(0.3, 2.0)

    # ---- Loop de AUM dinamico ----
    flows = pd.DataFrame(index=df.index)
    flows["rv"] = rv
    flows["vc_exposure"] = vc_exp_smooth
    flows["cta_exposure"] = cta_exp
    flows["rp_exposure"] = rp_exp

    # Inicializa AUMs
    aum_vc = np.full(len(df), config.vc_aum_base)
    aum_cta = np.full(len(df), config.cta_aum_base)
    aum_rp = np.full(len(df), config.rp_aum_base)

    flow_vc = np.zeros(len(df))
    flow_cta = np.zeros(len(df))
    flow_rp = np.zeros(len(df))

    # Tambem rastreamos versao com AUM ESTATICO pra comparar
    flow_vc_static = np.zeros(len(df))
    flow_cta_static = np.zeros(len(df))
    flow_rp_static = np.zeros(len(df))

    for i in range(1, len(df)):
        # Flow do dia = AUM * delta_exposure
        d_vc = vc_exp_smooth.iloc[i] - vc_exp_smooth.iloc[i - 1]
        d_cta = cta_exp.iloc[i] - cta_exp.iloc[i - 1]
        d_rp = rp_exp.iloc[i] - rp_exp.iloc[i - 1]

        flow_vc[i] = aum_vc[i - 1] * d_vc
        flow_cta[i] = aum_cta[i - 1] * d_cta
        flow_rp[i] = aum_rp[i - 1] * d_rp

        flow_vc_static[i] = config.vc_aum_base * d_vc
        flow_cta_static[i] = config.cta_aum_base * d_cta
        flow_rp_static[i] = config.rp_aum_base * d_rp

        # PnL e evolucao do AUM
        aum_vc[i] = aum_vc[i - 1] * (1 + vc_exp_smooth.iloc[i] * r.iloc[i])
        aum_cta[i] = aum_cta[i - 1] * (1 + cta_exp.iloc[i] * r.iloc[i])
        aum_rp[i] = aum_rp[i - 1] * (1 + rp_exp.iloc[i] * r.iloc[i])

    flows["aum_vc"] = aum_vc
    flows["aum_cta"] = aum_cta
    flows["aum_rp"] = aum_rp
    flows["flow_vc"] = flow_vc
    flows["flow_cta"] = flow_cta
    flows["flow_rp"] = flow_rp
    flows["flow_total"] = flow_vc + flow_cta + flow_rp
    flows["flow_vc_static"] = flow_vc_static
    flows["flow_cta_static"] = flow_cta_static
    flows["flow_rp_static"] = flow_rp_static
    flows["flow_total_static"] = flow_vc_static + flow_cta_static + flow_rp_static
    flows["aum_total_dynamic"] = aum_vc + aum_cta + aum_rp
    flows["aum_total_static"] = (config.vc_aum_base + config.cta_aum_base
                                   + config.rp_aum_base)
    flows["aum_divergence_pct"] = (flows["aum_total_dynamic"]
                                      / flows["aum_total_static"] - 1) * 100

    # Percentil do flow total em rolling 1y
    flows["flow_total_pctile"] = flows["flow_total"].rolling(
        252, min_periods=60
    ).apply(lambda x: x.rank(pct=True).iloc[-1] * 100)

    return flows


# ---- 13.6 — Visualizacoes Nomura ----

def plot_options_pnl_table(summary: pd.DataFrame, sharpe: pd.DataFrame,
                             out: Path):
    """Heatmap estilo Nomura: PnL summary + Sharpe ratio lado a lado."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                     gridspec_kw={"width_ratios": [6, 5]})
    cmap = LinearSegmentedColormap.from_list("rg", [COLORS["down"], "#222", COLORS["up"]])

    # PnL
    v = summary.values.astype(float)
    vmax = np.nanmax(np.abs(v))
    im1 = ax1.imshow(v, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax1.set_xticks(range(len(summary.columns)))
    ax1.set_xticklabels(summary.columns)
    ax1.set_yticks(range(len(summary.index)))
    ax1.set_yticklabels(summary.index, fontsize=9)
    ax1.set_title("SPX Daily Options PnL Summary — Cumulative PnL (%)")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if pd.notna(v[i, j]):
                ax1.text(j, i, f"{v[i, j]:.1f}", ha="center", va="center",
                          fontsize=8, color=COLORS["fg"])

    # Sharpe
    s = sharpe.values.astype(float)
    smax = np.nanmax(np.abs(s)) if s.size else 1
    im2 = ax2.imshow(s, cmap=cmap, vmin=-smax, vmax=smax, aspect="auto")
    ax2.set_xticks(range(len(sharpe.columns)))
    ax2.set_xticklabels(sharpe.columns)
    ax2.set_yticks(range(len(sharpe.index)))
    ax2.set_yticklabels(sharpe.index, fontsize=9)
    ax2.set_title("Sharpe Ratio")
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            if pd.notna(s[i, j]):
                ax2.text(j, i, f"{s[i, j]:.1f}", ha="center", va="center",
                          fontsize=8, color=COLORS["fg"])
    plt.tight_layout()
    save_plot(fig, out)


def plot_skew_percentiles(sp_df: pd.DataFrame, out: Path):
    """Duas series: left-tail percentile e right-tail percentile."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    sp = sp_df.dropna()
    ax1.plot(sp.index, sp["skew_25dP_25dC_pctile"],
              color=COLORS["fg"], linewidth=0.9)
    ax1.axhline(80, color=COLORS["down"], linestyle="--", linewidth=0.8,
                 alpha=0.6)
    ax1.axhline(20, color=COLORS["up"], linestyle="--", linewidth=0.8,
                 alpha=0.6)
    ax1.set_title("SPX 3M Skew - 25dP/25dC (percentile) | LEFT-TAIL")
    ax1.set_ylabel("percentile")
    ax1.set_ylim(0, 100)
    ax1.grid(True)
    # Anota status
    if "left_tail_status" in sp_df.attrs:
        ax1.text(0.98, 0.92, sp_df.attrs["left_tail_status"],
                  transform=ax1.transAxes, ha="right", va="top",
                  fontsize=10, color=COLORS["accent"],
                  bbox=dict(boxstyle="round,pad=0.3", fc=COLORS["bg"],
                             ec=COLORS["accent"]))

    ax2.plot(sp.index, sp["skew_25dC_atm_pctile"],
              color=COLORS["down"], linewidth=0.9)
    ax2.axhline(80, color=COLORS["down"], linestyle="--", linewidth=0.8,
                 alpha=0.6)
    ax2.axhline(20, color=COLORS["up"], linestyle="--", linewidth=0.8,
                 alpha=0.6)
    ax2.set_title("SPX 3M Call Skew - 25dC/ATM (percentile) | RIGHT-TAIL")
    ax2.set_ylabel("percentile")
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    if "right_tail_status" in sp_df.attrs:
        ax2.text(0.98, 0.92, sp_df.attrs["right_tail_status"],
                  transform=ax2.transAxes, ha="right", va="top",
                  fontsize=10, color=COLORS["accent"],
                  bbox=dict(boxstyle="round,pad=0.3", fc=COLORS["bg"],
                             ec=COLORS["accent"]))
    plt.tight_layout()
    save_plot(fig, out)


def plot_systematic_flows(flows: pd.DataFrame, out: Path):
    """
    Grafico principal: flow total com percentil.
    Subplot: comparacao AUM dinamico vs estatico.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 2, 2]})
    f = flows.dropna()
    # flow total dinamico em bilhoes
    ft_dyn = f["flow_total"] / 1e9
    ft_sta = f["flow_total_static"] / 1e9

    ax1.plot(f.index, ft_dyn, color=COLORS["fg"], linewidth=0.9,
              label="Vol Control + CTA + Risk Parity (AUM dinamico)")
    ax1.plot(f.index, ft_sta, color=COLORS["neutral"], linewidth=0.7,
              linestyle="--", alpha=0.7, label="mesmo (AUM estatico)")
    # Marca percentis
    p10 = f["flow_total"].quantile(0.1) / 1e9
    p90 = f["flow_total"].quantile(0.9) / 1e9
    ax1.axhline(p10, color=COLORS["down"], linestyle=":", linewidth=0.7,
                 alpha=0.6, label=f"10%ile ({p10:.1f}B)")
    ax1.axhline(p90, color=COLORS["up"], linestyle=":", linewidth=0.7,
                 alpha=0.6, label=f"90%ile ({p90:.1f}B)")
    ax1.axhline(0, color=COLORS["fg"], linewidth=0.4)
    ax1.set_title("US Equities Systematic Flows — Vol Control + CTA + Risk Parity (USD bn)")
    ax1.set_ylabel("Flow (USD bn)")
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(True)

    # Break-down por estrategia
    ax2.plot(f.index, f["flow_vc"] / 1e9, color=COLORS["accent2"],
              linewidth=0.8, label="Vol Control")
    ax2.plot(f.index, f["flow_cta"] / 1e9, color=COLORS["accent"],
              linewidth=0.8, label="CTA")
    ax2.plot(f.index, f["flow_rp"] / 1e9, color=COLORS["up"],
              linewidth=0.8, label="Risk Parity")
    ax2.axhline(0, color=COLORS["fg"], linewidth=0.4)
    ax2.set_title("Flow por estrategia (AUM dinamico)")
    ax2.set_ylabel("USD bn")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True)

    # Divergencia AUM dinamico vs estatico
    ax3.fill_between(f.index, f["aum_divergence_pct"], 0,
                      where=f["aum_divergence_pct"] >= 0,
                      color=COLORS["up"], alpha=0.4, label=">0 (AUM cresceu)")
    ax3.fill_between(f.index, f["aum_divergence_pct"], 0,
                      where=f["aum_divergence_pct"] < 0,
                      color=COLORS["down"], alpha=0.4, label="<0 (AUM encolheu)")
    ax3.set_title("Divergencia AUM dinamico vs estatico (%) — aqui mora o erro comum")
    ax3.set_ylabel("%")
    ax3.legend(loc="lower left", fontsize=8)
    ax3.grid(True)
    plt.tight_layout()
    save_plot(fig, out)


# ---- 13.7 — Orquestracao da secao Nomura ----

def run_nomura_section(out_root: Path, spx_ticker: str = "SPY",
                        years: int = 5) -> dict:
    """
    Pipeline completo Nomura:
      - Carrega SPY + VIX + SKEW
      - Calcula options PnL + sharpe tables
      - Calcula skew percentiles
      - Calcula systematic flows com AUM dinamico
      - Dump CSVs + PNGs na pasta nomura/
    """
    ndir = out_root / "nomura"
    ndir.mkdir(parents=True, exist_ok=True)

    log.info(f"[nomura] carregando {spx_ticker} + indices de vol ({years}y)")
    bundle = load_ticker(spx_ticker, years=years, fetch_minute=False)
    vol_idx = load_vol_indices(years=years)

    if bundle is None or len(vol_idx) == 0:
        log.warning("[nomura] dados insuficientes, pulando")
        return {}

    # 25d IVs aproximados via VIX + SKEW
    iv_df = approximate_25d_ivs(vol_idx["vix"], vol_idx["skew"])

    # Options PnL
    pnl = compute_daily_options_pnl(bundle.daily, iv_df)
    summary = options_pnl_summary_table(pnl)
    sharpe = options_sharpe_table(pnl)

    # Skew percentiles
    sp = skew_percentiles(iv_df)

    # Flows dinamicos
    flows = compute_dynamic_flows(bundle.daily)

    # Dump CSVs
    pnl.to_csv(ndir / "options_pnl_daily.csv")
    summary.to_csv(ndir / "options_pnl_summary.csv")
    sharpe.to_csv(ndir / "options_sharpe_ratio.csv")
    sp.to_csv(ndir / "skew_percentiles.csv")
    flows.to_csv(ndir / "systematic_flows_dynamic_aum.csv")

    # Plots
    if len(summary) > 0:
        plot_options_pnl_table(summary, sharpe,
                                 ndir / "fig_options_pnl_table.png")
    if len(sp) > 0:
        plot_skew_percentiles(sp, ndir / "fig_skew_percentiles.png")
    if len(flows) > 0:
        plot_systematic_flows(flows, ndir / "fig_systematic_flows.png")

    # Summary txt
    lines = ["=" * 72, "NOMURA CROSS ASSET FRAMEWORK — REPORT", "=" * 72, ""]
    lines.append(f"Fonte spot: {spx_ticker} | Fonte vol: ^VIX + ^SKEW (aproximacao 25d)")
    lines.append(f"Periodo: {bundle.meta['daily_start'][:10]} -> {bundle.meta['daily_end'][:10]}")
    lines.append("")
    lines.append("[OPTIONS PnL SUMMARY — % cumulativo]")
    lines.append(summary.to_string())
    lines.append("")
    lines.append("[SHARPE RATIO]")
    lines.append(sharpe.to_string())
    lines.append("")
    lines.append("[SKEW PERCENTILES — estado atual]")
    lines.append(f"  Left-tail (25dP/25dC): {sp.attrs.get('left_tail_status', 'N/A')}")
    lines.append(f"  Right-tail (25dC/ATM): {sp.attrs.get('right_tail_status', 'N/A')}")
    if len(sp.dropna()) > 0:
        last = sp.dropna().iloc[-1]
        lines.append(f"  25dP/25dC pctile: {last['skew_25dP_25dC_pctile']:.0f}")
        lines.append(f"  25dC/ATM  pctile: {last['skew_25dC_atm_pctile']:.0f}")
        lines.append(f"  ATM IV    pctile: {last['atm_iv_pctile']:.0f}")
    lines.append("")
    lines.append("[SYSTEMATIC FLOWS — snapshot]")
    if len(flows.dropna()) > 0:
        fl = flows.dropna().iloc[-1]
        lines.append(f"  Flow total (dinamico): {fl['flow_total'] / 1e9:+.2f} USD bn")
        lines.append(f"  Flow total (estatico): {fl['flow_total_static'] / 1e9:+.2f} USD bn")
        lines.append(f"  AUM divergence: {fl['aum_divergence_pct']:+.2f}%  <-- aqui mora o erro")
        lines.append(f"  VC exposure:  {fl['vc_exposure']:.2f}x")
        lines.append(f"  CTA exposure: {fl['cta_exposure']:+.2f}x")
        lines.append(f"  RP exposure:  {fl['rp_exposure']:.2f}x")
        lines.append(f"  Flow percentile 1y: {fl['flow_total_pctile']:.0f}")
    lines.append("")
    lines.append("OBS: 25d IVs sao aproximados via SKEW index. Para precisao total,")
    lines.append("     plugar chain real (Bloomberg OVDV / OptionMetrics).")
    lines.append("=" * 72)
    (ndir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    log.info(f"[nomura] OK — output em {ndir}")
    return {
        "dir": str(ndir),
        "n_pnl_days": len(pnl),
        "left_tail": sp.attrs.get("left_tail_status"),
        "right_tail": sp.attrs.get("right_tail_status"),
    }


# =============================================================================
# 12. MAIN ORCHESTRATOR
# =============================================================================

def analyze_ticker(ticker: str, out_root: Path, years: int = 5,
                    fetch_minute: bool = False) -> Optional[dict]:
    """Pipeline completo para 1 ticker. Retorna dict com paths + metricas."""
    bundle = load_ticker(ticker, years=years, fetch_minute=fetch_minute)
    if bundle is None:
        log.warning(f"[{ticker}] pulado (sem dados)")
        return None

    # 1. Session frame
    df = build_session_frame(bundle)
    if len(df) < 30:
        log.warning(f"[{ticker}] poucos dados ({len(df)}), pulado")
        return None

    # 2. Features
    df = add_gap_features(df)
    df = add_range_features(df)
    df = compute_moving_averages(df)
    df = compute_volatility(df)

    # 3. Stats
    wstats = weekday_stats(df)
    updown = up_down_by_weekday(df)
    roll = rolling_weekday_average(df)
    ma_stats = ma_residency_stats(df)
    streaks = streak_stats(df)
    cond = conditional_after_streaks(df)
    regime = regime_stats(df)
    zstats = zscore_continuation_stats(df)
    gaps = gap_stats(df)
    by_month, turn_rows = monthly_stats(df)

    # 4. Backtest
    bt = backtest_rth(df)

    # 5. Diretorio de saida por ticker
    tdir = out_root / ticker.replace("=", "_")
    tdir.mkdir(parents=True, exist_ok=True)

    # 6. Dump CSVs
    df.to_csv(tdir / "session_frame.csv")
    wstats.to_csv(tdir / "weekday_stats.csv", index=False)
    updown.to_csv(tdir / "updown_by_weekday.csv", index=False)
    roll.to_csv(tdir / "rolling_weekday_avg.csv", index=False)
    ma_stats.to_csv(tdir / "ma_residency.csv", index=False)
    streaks.to_csv(tdir / "streak_stats.csv", index=False)
    cond.to_csv(tdir / "conditional_after_streaks.csv", index=False)
    regime.to_csv(tdir / "regime_stats.csv", index=False)
    zstats.to_csv(tdir / "zscore_continuation.csv", index=False)
    gaps.to_csv(tdir / "gap_stats.csv", index=False)
    by_month.to_csv(tdir / "monthly_stats.csv", index=False)
    turn_rows.to_csv(tdir / "month_turn_stats.csv", index=False)
    bt.equity.to_frame("equity").to_csv(tdir / "equity_curve.csv")
    bt.drawdown.to_frame("drawdown").to_csv(tdir / "drawdown_curve.csv")
    pd.Series(bt.metrics).to_frame("value").to_csv(tdir / "backtest_metrics.csv")

    # 7. Plots
    plot_weekday_bars(wstats, ticker, tdir / "fig_weekday_bars.png")
    plot_weekday_hitrate(wstats, ticker, tdir / "fig_weekday_hitrate.png")
    plot_equity_curve(bt, ticker, tdir / "fig_equity_drawdown.png")
    plot_histogram(df, ticker, tdir / "fig_histogram.png")
    plot_heatmap_weekday_month(df, ticker, tdir / "fig_heatmap_weekday_month.png")
    plot_ma_residency(ma_stats, ticker, tdir / "fig_ma_residency.png")
    plot_streak_distribution(df, ticker, tdir / "fig_streak_distribution.png")

    # 8. Summary text
    summary = build_summary(ticker, bundle, df, bt, wstats, updown, ma_stats, streaks)
    (tdir / "summary.txt").write_text(summary, encoding="utf-8")
    (tdir / "meta.json").write_text(json.dumps(bundle.meta, indent=2, default=str),
                                     encoding="utf-8")

    log.info(f"[{ticker}] OK — Sharpe={bt.metrics['sharpe']} "
             f"MaxDD={bt.metrics['max_drawdown_pct']}% "
             f"n={bt.metrics['n_days']}")
    return {"ticker": ticker, "dir": str(tdir), "metrics": bt.metrics}


def _in_notebook() -> bool:
    """Detecta se esta rodando dentro de Jupyter/IPython/BQuant."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ != "TerminalInteractiveShell"
    except Exception:
        return False


def main(tickers=None, years=5, fetch_minute=False, out="session_stats_out",
          no_nomura=False, nomura_ticker="SPY"):
    """
    Entry point. Aceita chamada programatica (Python/BQuant) ou CLI.

    BQuant:   main()   ou   main(tickers=["SPY","QQQ"], years=5)
    CLI:      python session_stats.py SPY QQQ --years 10
    """
    # Se chamado programaticamente (em notebook ou import), usa kwargs.
    # Se chamado via CLI direto, parseia sys.argv.
    if _in_notebook() or tickers is not None:
        class _A: pass
        args = _A()
        args.tickers = tickers if tickers is not None else DEFAULT_UNIVERSE
        args.years = years
        args.minute = fetch_minute
        args.out = out
        args.no_nomura = no_nomura
        args.nomura_ticker = nomura_ticker
    else:
        parser = argparse.ArgumentParser(description="Session Stats Engine")
        parser.add_argument("tickers", nargs="*", default=DEFAULT_UNIVERSE,
                             help="Tickers yfinance (default: universo preset)")
        parser.add_argument("--years", type=int, default=5,
                             help="Janela diaria em anos")
        parser.add_argument("--minute", action="store_true",
                             help="Tentar baixar barras de 1 minuto (ultimos 30d)")
        parser.add_argument("--out", type=str, default="session_stats_out",
                             help="Diretorio de saida")
        parser.add_argument("--no-nomura", action="store_true",
                             help="Pular secao Nomura (options PnL + skew + flows)")
        parser.add_argument("--nomura-ticker", type=str, default="SPY",
                             help="Ticker spot para secao Nomura (default SPY)")
        args = parser.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log.info(f"Output: {out_root}")
    log.info(f"Universo: {args.tickers}")

    results = []
    for t in args.tickers:
        try:
            r = analyze_ticker(t, out_root, years=args.years, fetch_minute=args.minute)
            if r:
                results.append(r)
        except Exception as e:
            log.exception(f"[{t}] erro: {e}")

    # Consolida: um CSV mestre com metricas de todos os tickers
    if results:
        mdf = pd.DataFrame([{"ticker": r["ticker"], **r["metrics"]} for r in results])
        mdf.to_csv(out_root / "ALL_metrics.csv", index=False)
        log.info(f"Metricas consolidadas: {out_root / 'ALL_metrics.csv'}")

    # Nomura section (options PnL + skew percentiles + dynamic AUM flows)
    if not args.no_nomura:
        try:
            run_nomura_section(out_root, spx_ticker=args.nomura_ticker,
                                years=args.years)
        except Exception as e:
            log.exception(f"[nomura] erro: {e}")

    # ZIP final
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = out_root.parent / f"session_stats_{ts}.zip"
    export_zip(out_root, zip_path)
    print(f"\n  Concluido.\n  Pasta: {out_root}\n  ZIP:   {zip_path}\n")


if __name__ == "__main__":
    main()
