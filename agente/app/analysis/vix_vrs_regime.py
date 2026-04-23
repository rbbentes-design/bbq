"""
VIX × VRS Quadrant — Krishnamurthy CMT JOTA 71

Computa regime atual baseado em VIX + VRS (Variance from Reference Structure).

VRS = -(spread_atual - spread_medio_referencia)
  spread = VIX6M - VIX (ou VIX3M - VIX fallback)

4 regimes:
  R1 Contango + VIX baixo       (RALLY — bull OK)
  R2 VIX spike + Contango        (ENTRY — setup raro e rentavel)
  R3 VIX spike + Backwardation   (BEAR — stress sistemico)
  R4 VIX baixo + Backwardation   (RARO — anomalia transicional)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("analysis.vix_vrs")


@dataclass
class VixVrsResult:
    regime: int = 0                   # 1-4
    regime_label: str = ""
    vix: float = 0.0
    vix_threshold: float = 17.0
    spread: float = 0.0               # VIX6M (ou VIX3M) - VIX
    ref_mean: float = 0.0             # spread medio historico
    vrs: float = 0.0                  # -(spread - ref_mean)
    vrs_threshold: float = 0.0        # zero
    horizon_label: str = ""           # "6M-30d" ou "3M-30d"
    action: str = ""
    structures: list[str] = field(default_factory=list)
    interpretation: str = ""
    trail_60d: list[dict] = field(default_factory=list)  # [{"date","vix","vrs"}]
    error: str = ""


_REGIME_META: dict[int, dict[str, Any]] = {
    1: {
        "label": "R1 — RALLY (Contango + VIX baixo)",
        "action": "Manter long exposure. Credit spreads OTM favoravel.",
        "structures": ["Bull Put Credit Spread OTM", "Covered Call", "Long SPY com collar largo"],
        "interpretation": "Contango saudavel. Historicamente regime de rally sustentavel.",
    },
    2: {
        "label": "R2 — ENTRY (VIX spike + Contango)",
        "action": "SINAL RARO de entrada. Comprar dip com bull put spread.",
        "structures": ["Bull Put Credit Spread (premium inflado)", "Long SPX com ATM call", "Debit bullish"],
        "interpretation": "VIX spike mas curva ainda em contango — medo concentrado no curto, nao sistemico. Paper Krishnamurthy: melhor setup historicamente.",
    },
    3: {
        "label": "R3 — BEAR (Backwardation)",
        "action": "Reduzir long. Hedges com puts. NAO compre o dip.",
        "structures": ["Put Debit Spread OTM", "Long VIX calls", "Collar com put ITM"],
        "interpretation": "Backwardation = stress sistemico. Paper: pior regime, retornos 5d negativos.",
    },
    4: {
        "label": "R4 — RARO (VIX baixo + Backwardation)",
        "action": "Cautela, regime transicional. Neutralidade.",
        "structures": ["Iron Butterfly", "Calendar long vega", "Reducao de leverage"],
        "interpretation": "Anomalia. Raramente persiste — resolve para R1 ou R3.",
    },
}


def compute_vix_vrs_regime(
    vix_now: float | None = None,
    vix3m_now: float | None = None,
    vix6m_now: float | None = None,
    history_df=None,
    vix_threshold: float = 17.0,
    reference_days: int = 252,
) -> VixVrsResult:
    """
    Calcula regime atual VIX × VRS.

    Args:
        vix_now: VIX atual. Se None, pega ultima linha do history_df.
        vix3m_now / vix6m_now: curva atual.
        history_df: DataFrame com colunas VIX, VIX3M, VIX6M (indexado por data).
            Usado pra calcular ref_mean (spread medio historico) + trail 60d.
        vix_threshold: ponto de corte de VIX para alto/baixo.
        reference_days: janela para calcular spread_medio_referencia.

    Returns:
        VixVrsResult com regime 1-4 + contexto.
    """
    result = VixVrsResult(vix_threshold=vix_threshold)

    if history_df is None or len(history_df) < 30:
        result.error = "historico insuficiente (<30 dias)"
        _log.warning("vix_vrs_insufficient_history",
                     rows=0 if history_df is None else len(history_df))
        return result

    try:
        import pandas as pd
        df = history_df.sort_index()

        # Determina horizon (6M ou 3M)
        has_6m = "VIX6M" in df.columns and df["VIX6M"].notna().sum() >= 30
        has_3m = "VIX3M" in df.columns and df["VIX3M"].notna().sum() >= 30

        if has_6m:
            long_col = "VIX6M"
            result.horizon_label = "6M-30d"
        elif has_3m:
            long_col = "VIX3M"
            result.horizon_label = "3M-30d"
        else:
            result.error = "VIX3M/VIX6M indisponiveis"
            return result

        spread_ts = (df[long_col] - df["VIX"]).dropna()
        if len(spread_ts) < 30:
            result.error = "spread serie curta"
            return result

        # Reference: media do spread nos ultimos N dias (exceto o mais recente)
        ref_window = spread_ts.iloc[-reference_days:-1] if len(spread_ts) > reference_days else spread_ts.iloc[:-1]
        ref_mean = float(ref_window.mean())

        # Valores atuais
        if vix_now is None:
            vix_now = float(df["VIX"].dropna().iloc[-1])
        current_long = vix6m_now if (has_6m and vix6m_now is not None) else (vix3m_now if has_3m else None)
        if current_long is None:
            current_long = float(df[long_col].dropna().iloc[-1])

        spread_now = current_long - vix_now
        vrs = -(spread_now - ref_mean)  # formula do paper (invertida pro plot)

        result.vix = vix_now
        result.spread = spread_now
        result.ref_mean = ref_mean
        result.vrs = vrs

        # Regime:
        # R1: VIX < thr, VRS < 0 (contango forte, vix baixo)
        # R2: VIX > thr, VRS < 0 (spike mas contango mantido)
        # R3: VIX > thr, VRS > 0 (backwardation)
        # R4: VIX < thr, VRS > 0 (raro)
        vix_high = vix_now > vix_threshold
        vrs_high = vrs > 0
        if not vix_high and not vrs_high:
            regime = 1
        elif vix_high and not vrs_high:
            regime = 2
        elif vix_high and vrs_high:
            regime = 3
        else:
            regime = 4

        result.regime = regime
        meta = _REGIME_META[regime]
        result.regime_label = meta["label"]
        result.action = meta["action"]
        result.structures = list(meta["structures"])
        result.interpretation = meta["interpretation"]

        # Trail 60d (para scatter plot)
        trail_df = df.iloc[-60:].copy()
        for idx, row in trail_df.iterrows():
            v = row.get("VIX")
            long_v = row.get(long_col)
            if v is None or long_v is None:
                continue
            try:
                vrs_pt = -((float(long_v) - float(v)) - ref_mean)
                result.trail_60d.append({
                    "date": str(idx)[:10],
                    "vix": float(v),
                    "vrs": vrs_pt,
                })
            except Exception:
                continue

        _log.info("vix_vrs_computed", regime=regime, vix=vix_now,
                  vrs=vrs, spread=spread_now, horizon=result.horizon_label)

    except Exception as exc:
        result.error = str(exc)[:120]
        _log.warning("vix_vrs_error", error=str(exc)[:120])

    return result


def load_vix_history_from_db(days: int = 365):
    """
    Carrega historico VIX/VIX3M/VIX6M via yfinance (yfinance tem ^VIX, ^VIX3M, ^VIX6M).
    Fallback: DB Bloomberg bql_timeseries se yfinance falhar.
    Retorna DataFrame com colunas VIX, VIX3M, VIX6M indexado por data.
    """
    import pandas as pd
    # Tenta yfinance primeiro (tem historico completo)
    try:
        import yfinance as yf
        yr = max(1, days // 365 + 1)
        tickers = {"^VIX": "VIX", "^VIX3M": "VIX3M", "^VIX9D": "VIX9D"}
        frames = {}
        for yf_tk, col_name in tickers.items():
            try:
                df_tk = yf.download(yf_tk, period=f"{yr}y", progress=False, auto_adjust=False)
                if df_tk is None or df_tk.empty:
                    continue
                if isinstance(df_tk.columns, pd.MultiIndex):
                    df_tk.columns = df_tk.columns.get_level_values(0)
                s = df_tk["Close"].dropna()
                s.name = col_name
                frames[col_name] = s
            except Exception:
                continue
        if frames:
            df = pd.DataFrame(frames)
            df.index = pd.to_datetime(df.index)
            _log.info("vix_history_yfinance", cols=list(df.columns), rows=len(df))
            return df.dropna(how="all")
    except Exception as exc:
        _log.warning("vix_history_yf_error", error=str(exc)[:100])

    # Fallback DB
    try:
        import sqlite3
        from app.query_layer import BloombergQueryLayer
        ql = BloombergQueryLayer()
        conn = sqlite3.connect(ql._db_path)
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        tickers_map = {"^VIX": "VIX", "VIX Index": "VIX", "VIX3M Index": "VIX3M",
                       "VIX6M Index": "VIX6M", "VIX9D Index": "VIX9D"}
        frames = {}
        for tk, col_name in tickers_map.items():
            rows = conn.execute(
                "SELECT date, value FROM bql_timeseries WHERE ticker=? AND field='price' AND date>=? ORDER BY date",
                (tk, cutoff),
            ).fetchall()
            if rows and len(rows) > 5:
                s = pd.Series({r[0]: r[1] for r in rows if r[1] is not None}, name=col_name)
                s.index = pd.to_datetime(s.index)
                if col_name not in frames:
                    frames[col_name] = s
        conn.close()
        if not frames:
            return None
        df = pd.DataFrame(frames)
        return df.dropna(how="all")
    except Exception as exc:
        _log.warning("vix_history_load_error", error=str(exc)[:100])
        return None
