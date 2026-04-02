"""
GEX + LETF Flow — BQL script

Roda sob BQuant Python 3.11 (C:/blp/bqnt/environments/bqnt-3/python.exe)

Calcula dois sinais mecânicos de fluxo de fim de dia (Barbon et al.):

  1. Γ^HP  — Gamma Hedging Pressure (GEX do SPX)
             Dealers short-gamma precisam vender quando mercado sobe → pressão de venda
             Dealers long-gamma precisam comprar quando mercado sobe  → amortece moves

  2. Ω^LETF — LETF Rebalancing Flow
              AUM × L × (L-1) × r / (1 + L×r)
              ETFs 3x bull: se SPX subiu 1%, precisam comprar ~6% de exposição adicional
              ETFs 3x bear: se SPX subiu 1%, precisam vender ~6%

Saída: JSON para stdout
{
  "letf": {
    "spx": {"flow_usd": ..., "direction": "buy"|"sell"|"flat"},
    "ndx": {"flow_usd": ..., "direction": ...},
    "per_etf": {"SPXL": {"flow_usd": ..., "aum": ..., "leverage": 3, "return": 0.01}},
    "spx_r": 0.012,   # retorno do dia SPX
    "ndx_r": 0.008
  },
  "gex": {
    "spx": {"gex_usd": ..., "gamma_regime": "long"|"short"|"flat", "flip_level": ...},
    "per_strike": [[strike, gex], ...]
  },
  "flow_per_member": {
    "AAPL": {"letf_flow_usd": ..., "gex_flow_usd": ..., "total": ..., "direction": "buy"|"sell"},
    ...
  },
  "net_signal": {
    "direction": "buy"|"sell"|"flat",
    "magnitude_usd": ...,
    "conviction": "high"|"medium"|"low",
    "summary": "..."
  },
  "timestamp": "2026-04-01T15:30:00"
}

Uso:
    bqnt_python bql_gex_flow.py
    bqnt_python bql_gex_flow.py --spx-only
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from datetime import datetime, date

warnings.filterwarnings("ignore")

# ── LETF Specs ────────────────────────────────────────────────────────────────
# Baseado em etf_rebalancing_dashboard.py do repo rbbentes-design/bbq

# (etf_bbg, underlying_bbg, leverage, index_key, etf_type)
# etf_type: "index" = usa futuros/índice como underlying
#            "single" = single-stock leveraged
LETF_SPECS = [
    # S&P 500
    ("SPXL US Equity",  "SPX Index",       3.0,  "spx",   "index"),
    ("SPXS US Equity",  "SPX Index",      -3.0,  "spx",   "index"),
    # Nasdaq-100
    ("TQQQ US Equity",  "NDX Index",       3.0,  "ndx",   "index"),
    ("SQQQ US Equity",  "NDX Index",      -3.0,  "ndx",   "index"),
    # Semiconductors
    ("SOXL US Equity",  "SOX Index",       3.0,  "sox",   "index"),
    ("SOXS US Equity",  "SOX Index",      -3.0,  "sox",   "index"),
    # Biotech
    ("LABU US Equity",  "SPSIBITO Index",  3.0,  "biotech","index"),
    # Single-stock
    ("TSLL US Equity",  "TSLA UW Equity",  1.5,  "tsla",  "single"),
    ("TSLQ US Equity",  "TSLA UW Equity", -1.0,  "tsla",  "single"),
    ("NVDU US Equity",  "NVDA UW Equity",  2.0,  "nvda",  "single"),
    ("NVD US Equity",   "NVDA UW Equity", -2.0,  "nvda",  "single"),
    ("GGLL US Equity",  "GOOGL UW Equity", 2.0,  "googl", "single"),
    ("GGLS US Equity",  "GOOGL UW Equity",-1.0,  "googl", "single"),
    ("MSTU US Equity",  "MSTR US Equity",  2.0,  "mstr",  "single"),
    ("MSTZ US Equity",  "MSTR US Equity", -2.0,  "mstr",  "single"),
    ("NVOX US Equity",  "NVO US Equity",   2.0,  "nvo",   "single"),
]

# Índices que mapeamos para membros (para distribuição de fluxo)
INDEX_TICKERS = {
    "spx": "SPX Index",
    "ndx": "NDX Index",
}

# Proxy de execução para índices (futuros)
TARGET_FUTURES = {
    "SPX Index": "ES1 Index",
    "NDX Index": "NQ1 Index",
}

SPX_BBG = "SPX Index"
NDX_BBG = "NDX Index"

# ── BQL helpers ───────────────────────────────────────────────────────────────

def _first_numeric_col(df) -> str | None:
    """Retorna nome da primeira coluna numérica do DataFrame."""
    import pandas as pd
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return df.columns[0] if len(df.columns) else None


def _bql_px_last(tickers: list[str]) -> dict[str, float]:
    """Retorna {bbg_ticker: preco_atual} para lista de tickers."""
    import bql
    bq = bql.Service()
    results = {}
    BATCH = 50
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i+BATCH]
        try:
            req = bql.Request(batch, bq.data.px_last())
            resp = bq.execute(req)
            df = bql.combined_df(resp).groupby(level=0).last()
            col = _first_numeric_col(df)
            if not col:
                continue
            for t, row in df.iterrows():
                v = row.get(col)
                if v is not None and v == v:
                    results[t] = float(v)
        except Exception as e:
            print(f"[gex_flow] px_last batch {i}: {e}", file=sys.stderr)
    return results


def _bql_aum(tickers: list[str]) -> dict[str, float]:
    """Retorna {bbg_ticker: AUM total em USD} — fund_total_assets para LETFs."""
    import bql
    bq = bql.Service()
    results = {}
    try:
        req = bql.Request(tickers, bq.data.fund_total_assets())
        resp = bq.execute(req)
        df = bql.combined_df(resp).groupby(level=0).last()
        col = next((c for c in df.columns if "ASSET" in c.upper() or "AUM" in c.upper()), None) or _first_numeric_col(df)
        if col:
            for t, row in df.iterrows():
                v = row.get(col)
                if v is not None and v == v:
                    results[t] = float(v)  # Bloomberg retorna em USD
    except Exception as e:
        print(f"[gex_flow] aum error: {e}", file=sys.stderr)
    return results


def _bql_shares_out(tickers: list[str]) -> dict[str, float]:
    """Retorna {bbg_ticker: shares_outstanding (em unidades)}."""
    import bql
    bq = bql.Service()
    results = {}
    try:
        req = bql.Request(tickers, bq.data.eqy_sh_out())
        resp = bq.execute(req)
        df = bql.combined_df(resp).groupby(level=0).last()
        col = next((c for c in df.columns if "SH_OUT" in c.upper() or "SHARE" in c.upper()), None) or _first_numeric_col(df)
        if col:
            for t, row in df.iterrows():
                v = row.get(col)
                if v is not None and v == v:
                    # Bloomberg retorna em milhares
                    results[t] = float(v) * 1000
    except Exception as e:
        print(f"[gex_flow] shares_out error: {e}", file=sys.stderr)
    return results


def _bql_px_prev(tickers: list[str]) -> dict[str, float]:
    """Retorna preço de ontem para calcular retorno do dia."""
    import bql
    bq = bql.Service()
    results = {}
    BATCH = 50
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i+BATCH]
        try:
            item = bq.data.px_last(dates=bq.func.range("-2D", "-1D"), fill="PREV")
            req = bql.Request(batch, item)
            resp = bq.execute(req)
            df = bql.combined_df(resp).groupby(level=0).last()
            col = _first_numeric_col(df)
            if not col:
                continue
            for t, row in df.iterrows():
                v = row.get(col)
                if v is not None and v == v:
                    results[t] = float(v)
        except Exception as e:
            print(f"[gex_flow] px_prev batch {i}: {e}", file=sys.stderr)
    return results


def _bql_index_weights(index_bbg: str, top_n: int = 100) -> dict[str, float]:
    """
    Retorna {member_ticker: float_adj_weight} para membros do índice.
    Peso = mktcap × free_float% / total
    """
    import bql
    bq = bql.Service()
    try:
        univ = bq.univ.members([index_bbg])
        items = {
            "mktcap":    bq.data.cur_mkt_cap(),
            "free_float": bq.data.eqy_free_float_pct(),
        }
        req = bql.Request(univ, items)
        resp = bq.execute(req)
        df = bql.combined_df(resp).groupby(level=0).last()

        weights = {}
        for t, row in df.iterrows():
            mc = row.get("CUR_MKT_CAP")
            ff = row.get("EQY_FREE_FLOAT_PCT")
            if mc is not None and mc == mc and ff is not None and ff == ff:
                weights[t] = float(mc) * float(ff) / 100.0

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # Ordena por peso e retorna top_n
        return dict(sorted(weights.items(), key=lambda x: -x[1])[:top_n])

    except Exception as e:
        print(f"[gex_flow] index_weights {index_bbg}: {e}", file=sys.stderr)
        return {}


# ── LETF Rebalancing Flow ─────────────────────────────────────────────────────

def compute_letf_flows(price_now: dict[str, float], price_prev: dict[str, float]) -> dict:
    """
    Calcula fluxo de rebalanceamento para cada LETF.

    Formula (Barbon et al. / Goldman Sachs):
        Rebal_$ = AUM × L × (L-1) × r / (1 + L×r)

    onde:
        AUM = NAV × shares_out
        L   = leverage (positivo para bull, negativo para bear)
        r   = retorno do dia do underlying

    Retorno positivo + L>0 (bull 3x): fluxo positivo (compra)
    Retorno positivo + L<0 (bear 3x): fluxo negativo (venda)
    """
    etf_tickers = [s[0] for s in LETF_SPECS]
    under_tickers = list(set(s[1] for s in LETF_SPECS))

    aum_map   = _bql_aum(etf_tickers)
    share_map = _bql_shares_out(etf_tickers)  # fallback se AUM indisponível

    per_etf: dict[str, dict] = {}
    flow_by_index: dict[str, float] = {}

    for (etf, under, lev, idx_key, etf_type) in LETF_SPECS:
        p1 = price_now.get(under)
        p0 = price_prev.get(under)

        if p1 is None or p0 is None or p0 == 0:
            continue

        r = (p1 - p0) / p0

        aum = aum_map.get(etf)

        # Fallback: preço × shares se fund_total_assets indisponível
        if aum is None:
            px     = price_now.get(etf)
            shares = share_map.get(etf)
            if px and shares:
                aum = px * shares

        if aum is None or aum == 0:
            continue

        denom = 1 + lev * r
        if abs(denom) < 1e-12:
            continue

        flow_usd = aum * lev * (lev - 1) * r / denom
        etf_px = price_now.get(etf)

        per_etf[etf.replace(" US Equity", "")] = {
            "flow_usd": round(flow_usd, 0),
            "aum":      round(aum, 0),
            "leverage": lev,
            "return":   round(r, 6),
            "nav":      round(etf_px, 4) if etf_px else None,
        }

        flow_by_index[idx_key] = flow_by_index.get(idx_key, 0.0) + flow_usd

    # Agrega por índice
    def _fmt_index(idx_key: str) -> dict:
        f = flow_by_index.get(idx_key, 0.0)
        if abs(f) < 1e6:
            direction = "flat"
        else:
            direction = "buy" if f > 0 else "sell"
        return {"flow_usd": round(f, 0), "direction": direction}

    # Retorno do dia dos índices principais
    def _index_return(bbg: str) -> float | None:
        p1 = price_now.get(bbg)
        p0 = price_prev.get(bbg)
        if p1 and p0 and p0 != 0:
            return (p1 - p0) / p0
        return None

    return {
        "spx":     _fmt_index("spx"),
        "ndx":     _fmt_index("ndx"),
        "sox":     _fmt_index("sox"),
        "per_etf": per_etf,
        "spx_r":   _index_return(SPX_BBG),
        "ndx_r":   _index_return(NDX_BBG),
    }


# ── GEX (Gamma Exposure) ──────────────────────────────────────────────────────

def _black_scholes_gamma(S: float, K: float, T: float, vol: float, r: float = 0.0) -> float:
    """Gamma Black-Scholes — comum a calls e puts."""
    if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * vol * vol) * T) / (vol * math.sqrt(T))
        gamma = math.exp(-0.5 * d1 * d1) / (math.sqrt(2 * math.pi) * S * vol * math.sqrt(T))
        return gamma
    except Exception:
        return 0.0


def compute_gex_spx(spot: float) -> dict:
    """
    Calcula GEX do SPX via chain de opções BQL.

    GEX = Σ_strikes [gamma × OI × 100 × (sign)]
        sign = +1 para calls (dealers long gamma se compraram calls)
               -1 para puts (dealers long gamma se venderam puts → short)

    Convenção: dealers são sempre contrapartida do investidor.
    - Investidor compra call → dealer vende call → dealer short gamma → sign = -1
    - Investidor compra put  → dealer vende put  → dealer short gamma → sign = -1
    - Investidor vende call  → dealer compra call → dealer long gamma → sign = +1

    Na prática usamos a convenção de mercado: call GEX positivo, put GEX negativo.
    GEX total positivo = dealers long gamma → amortece moves (mean-reversion)
    GEX total negativo = dealers short gamma → amplifica moves (trending)
    """
    import bql
    bq = bql.Service()

    per_strike: list[list] = []
    gex_total = 0.0
    flip_level = None

    try:
        # Filtra opções próximas (DTE 0-10, moneyness ±3%)
        min_strike = spot * 0.97
        max_strike = spot * 1.03

        # Busca VIX como proxy de IV (evita cota de ivol() da chain inteira)
        vix_iv = 0.20  # fallback 20%
        try:
            vix_req = bql.Request(["VIX Index"], bq.data.px_last())
            vix_resp = bq.execute(vix_req)
            vix_df = bql.combined_df(vix_resp)
            vix_col = _first_numeric_col(vix_df)
            if vix_col:
                vix_val = vix_df[vix_col].iloc[0]
                if vix_val and vix_val == vix_val:
                    vix_iv = float(vix_val) / 100.0
        except Exception:
            pass

        # Metadados (STRIKE_PX, PUT_CALL, EXPIRE_DT) já vêm do universo de opções
        # Pede só OI — IV usa VIX como proxy (evita cota BQL de ivol())
        items = {
            "OI": bq.data.open_int(),
        }

        req = bql.Request(
            bq.univ.options(["SPX Index"]),
            items,
        )
        resp = bq.execute(req)
        df = bql.combined_df(resp)

        today = date.today()
        # Normaliza nomes de colunas (universo já traz STRIKE_PX, PUT_CALL, EXPIRE_DT)
        if "STRIKE_PX" in df.columns:
            df = df.rename(columns={"STRIKE_PX": "Strike", "PUT_CALL": "Type", "EXPIRE_DT": "Expire"})

        # Filtra em Python: moneyness ±3%, DTE ≤ 10 dias, OI > 0
        if "Strike" in df.columns:
            dte_cutoff = today + __import__("datetime").timedelta(days=10)
            expire_col = df.get("Expire") if hasattr(df, "get") else df["Expire"] if "Expire" in df.columns else None
            mask = (
                (df["Strike"] >= min_strike) &
                (df["Strike"] <= max_strike) &
                (df["OI"].fillna(0) > 0)
            )
            if "Expire" in df.columns:
                mask = mask & (df["Expire"].apply(
                    lambda e: (e.date() if hasattr(e, "date") else e) <= dte_cutoff
                ))
            df = df[mask]

        gex_by_strike: dict[float, float] = {}

        for _, row in df.iterrows():
            strike = row.get("Strike")
            opt_type = row.get("Type")
            oi = row.get("OI")
            expire = row.get("Expire")

            if any(v is None or (v != v) for v in [strike, opt_type, oi]):
                continue

            try:
                exp_date = expire.date() if hasattr(expire, "date") else expire
                T = max((exp_date - today).days / 365.0, 1/365)
            except Exception:
                T = 30 / 365

            gamma = _black_scholes_gamma(spot, float(strike), T, vix_iv)
            contract_gex = gamma * float(oi) * 100 * (spot ** 2) * 0.01

            # Convenção: call = positivo, put = negativo
            sign = 1.0 if str(opt_type).upper().startswith("C") else -1.0
            gex_contrib = contract_gex * sign

            gex_total += gex_contrib
            k = round(float(strike), 0)
            gex_by_strike[k] = gex_by_strike.get(k, 0.0) + gex_contrib

        # Encontra gamma flip (strike onde GEX muda de sinal)
        if gex_by_strike:
            strikes_sorted = sorted(gex_by_strike.keys())
            per_strike = [[k, round(gex_by_strike[k] / 1e9, 4)] for k in strikes_sorted]

            # Flip: strike onde GEX acumulado muda de sinal
            cumulative = 0.0
            for k in strikes_sorted:
                prev = cumulative
                cumulative += gex_by_strike[k]
                if prev * cumulative < 0:
                    flip_level = k
                    break

    except Exception as e:
        print(f"[gex_flow] gex_spx error: {e}", file=sys.stderr)

    if abs(gex_total) < 1e8:
        gamma_regime = "flat"
    elif gex_total > 0:
        gamma_regime = "long"   # dealers long gamma → amortece
    else:
        gamma_regime = "short"  # dealers short gamma → amplifica

    return {
        "gex_usd":      round(gex_total, 0),
        "gex_bn":       round(gex_total / 1e9, 3),
        "gamma_regime": gamma_regime,
        "flip_level":   flip_level,
        "per_strike":   per_strike[:60],  # top 60 strikes
    }


# ── Per-member flow attribution ───────────────────────────────────────────────

def compute_per_member_flow(
    letf_spx_flow: float,
    letf_ndx_flow: float,
    gex_total: float,
    spx_r: float | None,
    spx_weights: dict[str, float],
    ndx_weights: dict[str, float],
) -> dict[str, dict]:
    """
    Distribui fluxo de LETF + GEX hedging para membros do índice.

    LETF flow → proporcional ao float-adj weight no índice
    GEX hedging flow → proporcional ao peso, na direção oposta ao GEX
                       (dealers short gamma compram quando mercado cai e vendem quando sobe)
    """
    result: dict[str, dict] = {}

    # Todos os membros únicos
    all_members = set(spx_weights) | set(ndx_weights)

    gex_hedge_flow = 0.0
    if spx_r is not None and gex_total != 0:
        # Dealers short gamma precisam delta-hedge na mesma direção do mercado
        # mas escalam com gamma × ΔS (simplificado: gex × return)
        gex_hedge_flow = -gex_total * spx_r  # short gamma → vende em alta

    for member in all_members:
        spx_w = spx_weights.get(member, 0.0)
        ndx_w = ndx_weights.get(member, 0.0)

        letf_flow = letf_spx_flow * spx_w + letf_ndx_flow * ndx_w
        gex_flow  = gex_hedge_flow * spx_w

        total = letf_flow + gex_flow

        # Normaliza ticker: "AAPL UW Equity" → "AAPL"
        ticker = member.split()[0]

        result[ticker] = {
            "letf_flow_usd": round(letf_flow, 0),
            "gex_flow_usd":  round(gex_flow, 0),
            "total":         round(total, 0),
            "direction":     "buy" if total > 1e4 else ("sell" if total < -1e4 else "flat"),
        }

    return result


# ── Net Signal ────────────────────────────────────────────────────────────────

def compute_net_signal(letf_data: dict, gex_data: dict, spx_r: float | None) -> dict:
    """
    Combina LETF flow + GEX regime num sinal direcional de EOD.

    Lógica (Barbon et al.):
    - LETF flow: mecânico, direcional → compra perto do close
    - GEX: se short gamma + mercado subiu → dealers vão comprar mais (amplifica)
            se long gamma + mercado subiu → dealers vão vender (amortece)
    """
    letf_spx = letf_data.get("spx", {}).get("flow_usd", 0.0) or 0.0
    letf_ndx = letf_data.get("ndx", {}).get("flow_usd", 0.0) or 0.0
    total_letf = letf_spx + letf_ndx

    gex_usd   = gex_data.get("gex_usd", 0.0) or 0.0
    gamma_reg = gex_data.get("gamma_regime", "flat")

    # GEX hedging pressure: quando dealers short gamma, amplificam o move do dia
    gex_pressure = 0.0
    if spx_r is not None and gex_usd != 0:
        gex_pressure = -gex_usd * spx_r  # short gamma (gex<0) + alta → positive buy pressure

    total_flow = total_letf + gex_pressure

    if abs(total_flow) < 50e6:
        direction = "flat"
        conviction = "low"
    elif abs(total_flow) < 500e6:
        direction = "buy" if total_flow > 0 else "sell"
        conviction = "low"
    elif abs(total_flow) < 2e9:
        direction = "buy" if total_flow > 0 else "sell"
        conviction = "medium"
    else:
        direction = "buy" if total_flow > 0 else "sell"
        conviction = "high"

    # Summary legível
    letf_str = f"LETF: ${total_letf/1e9:+.2f}B"
    gex_str  = f"GEX {gamma_reg} (${gex_usd/1e9:.1f}B)"
    press_str = f"hedge press: ${gex_pressure/1e9:+.2f}B" if gex_pressure != 0 else ""
    parts = [letf_str, gex_str]
    if press_str:
        parts.append(press_str)

    return {
        "direction":     direction,
        "magnitude_usd": round(total_flow, 0),
        "magnitude_bn":  round(total_flow / 1e9, 3),
        "letf_usd":      round(total_letf, 0),
        "gex_pressure":  round(gex_pressure, 0),
        "conviction":    conviction,
        "summary":       " | ".join(parts),
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────

def run() -> dict:
    import bql
    bq = bql.Service()

    # Todos os tickers que precisamos
    under_tickers = list(set(s[1] for s in LETF_SPECS))
    etf_tickers   = [s[0] for s in LETF_SPECS]

    print("[gex_flow] fetching prices...", file=sys.stderr)
    price_now  = _bql_px_last(under_tickers + etf_tickers)
    price_prev = _bql_px_prev(under_tickers + etf_tickers)

    print("[gex_flow] computing LETF flows...", file=sys.stderr)
    letf_data = compute_letf_flows(price_now, price_prev)

    spx_r = letf_data.get("spx_r")
    spot  = price_now.get(SPX_BBG, 5000.0)

    print("[gex_flow] computing GEX (SPX options chain)...", file=sys.stderr)
    gex_data = compute_gex_spx(spot)

    print("[gex_flow] fetching index weights...", file=sys.stderr)
    spx_weights = _bql_index_weights(SPX_BBG, top_n=100)
    ndx_weights = _bql_index_weights(NDX_BBG, top_n=100)

    print("[gex_flow] computing per-member flow...", file=sys.stderr)
    letf_spx_flow = letf_data.get("spx", {}).get("flow_usd", 0.0) or 0.0
    letf_ndx_flow = letf_data.get("ndx", {}).get("flow_usd", 0.0) or 0.0
    gex_usd = gex_data.get("gex_usd", 0.0) or 0.0

    flow_per_member = compute_per_member_flow(
        letf_spx_flow, letf_ndx_flow, gex_usd, spx_r,
        spx_weights, ndx_weights,
    )

    net_signal = compute_net_signal(letf_data, gex_data, spx_r)

    return {
        "letf":            letf_data,
        "gex":             {"spx": gex_data},
        "flow_per_member": flow_per_member,
        "net_signal":      net_signal,
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
    }


if __name__ == "__main__":
    try:
        result = run()
        print(json.dumps(result, ensure_ascii=False, default=str))
    except Exception as e:
        print(f"[gex_flow] fatal: {e}", file=sys.stderr)
        print("{}")
