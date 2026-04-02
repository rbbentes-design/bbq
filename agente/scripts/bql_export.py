"""
BQL Export — roda no terminal BQuant (Python 3.11)

Exporta todos os dados necessários para o MacroDesk em CSVs.
Nosso app Python 3.14 lê esses CSVs sem precisar do BQL.

Uso no terminal BQuant:
    python bql_export.py

Saída (em BQL_OUT_DIR):
    fundamentals.csv  — PE, beta, mktcap, ROE, margin, D/E, yield, price, 52w hi/lo
    options_iv.csv    — ATM IV 30D, skew 25D delta, put/call OI ratio
    gex_spx.csv       — GEX SPX por strike (OI × gamma × spot²)
    letf_flows.csv    — NAV e flows estimados dos principais LETFs
    meta.csv          — timestamp de geração

Tickers cobertos: top 20 SPX por peso + referências de índice/macro
"""

from __future__ import annotations

import csv
import os
import warnings
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Destino dos CSVs ──────────────────────────────────────────────────────────
BQL_OUT_DIR = Path(r"C:\Users\rafael bentes\agente-workspace\bql_data")
BQL_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Universo SPX Core ─────────────────────────────────────────────────────────
# Top 20 por peso no S&P 500 (abr/2026)
SPX_STOCKS = [
    "AAPL US Equity", "MSFT US Equity", "NVDA US Equity", "AMZN US Equity", "META US Equity",
    "GOOGL US Equity", "TSLA US Equity", "BRK/B US Equity", "AVGO US Equity", "JPM US Equity",
    "LLY US Equity", "UNH US Equity", "XOM US Equity", "COST US Equity", "V US Equity",
    "MA US Equity", "WMT US Equity", "NFLX US Equity", "JNJ US Equity", "PG US Equity",
]
SPX_ALL = SPX_STOCKS + [
    "SPY US Equity", "QQQ US Equity",
    "GLD US Equity", "TLT US Equity", "HYG US Equity",
]
# Mapa bbg → ticker limpo para os CSVs
BBG_TO_TICKER = {
    "AAPL US Equity": "AAPL", "MSFT US Equity": "MSFT", "NVDA US Equity": "NVDA",
    "AMZN US Equity": "AMZN", "META US Equity": "META", "GOOGL US Equity": "GOOGL",
    "TSLA US Equity": "TSLA", "BRK/B US Equity": "BRK-B", "AVGO US Equity": "AVGO",
    "JPM US Equity": "JPM", "LLY US Equity": "LLY", "UNH US Equity": "UNH",
    "XOM US Equity": "XOM", "COST US Equity": "COST", "V US Equity": "V",
    "MA US Equity": "MA", "WMT US Equity": "WMT", "NFLX US Equity": "NFLX",
    "JNJ US Equity": "JNJ", "PG US Equity": "PG",
    "SPY US Equity": "SPY", "QQQ US Equity": "QQQ",
    "GLD US Equity": "GLD", "TLT US Equity": "TLT", "HYG US Equity": "HYG",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _first_col(df, keys: list[str]):
    """Retorna nome da coluna que contém alguma das keys (case-insensitive)."""
    for k in keys:
        for c in df.columns:
            if k.upper() in str(c).upper():
                return c
    return None


def _val(row, col):
    """Valor seguro: None se NaN/NA."""
    if col is None:
        return None
    v = row.get(col)
    if v is None:
        return None
    try:
        import math
        if math.isnan(float(v)):
            return None
    except Exception:
        pass
    return v


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  → {path.name}: {len(rows)} linhas")


# ── 1. Fundamentais ───────────────────────────────────────────────────────────

def export_fundamentals(bq):
    print("Fundamentais (PE, beta, mktcap, ROE, margens, D/E, yield, price)...")
    tstr = ", ".join(f'"{t}"' for t in SPX_ALL)

    req = bq.execute(f"""
        get(
          PE_RATIO,
          CUR_MKT_CAP,
          BETA,
          PROF_MARGIN,
          TOT_DEBT_TO_TOT_EQY,
          RETURN_COM_EQY,
          EQY_DVD_YLD_IND,
          PX_LAST,
          EQY_SH_OUT,
          SALES_PER_SH,
          BOOK_VAL_PER_SH,
          EBITDA_TO_REVENUE
        ) for([{tstr}])
    """)

    import bql
    df = bql.combined_df(req)
    df = df.groupby(level=0).last()
    df.columns = [c.split("(")[0].strip() for c in df.columns]

    req2 = bq.execute(f"""
        get(
          PX_HIGH(dates=range(-365D,0D), frq=Y),
          PX_LOW(dates=range(-365D,0D), frq=Y),
          PX_TO_BOOK_RATIO,
          TRAIL_12M_EPS,
          EV_TO_T12M_EBITDA
        ) for([{tstr}])
    """)
    df2 = bql.combined_df(req2)
    df2 = df2.groupby(level=0).last()
    df2.columns = [c.split("(")[0].strip() for c in df2.columns]

    rows = []
    for bbg_t in SPX_ALL:
        ticker = BBG_TO_TICKER.get(bbg_t, bbg_t)
        r1 = df.loc[bbg_t] if bbg_t in df.index else {}
        r2 = df2.loc[bbg_t] if bbg_t in df2.index else {}

        def g1(k): return _val(r1, k)
        def g2(k): return _val(r2, k)

        price = g1("PX_LAST")
        hi52  = g2("PX_HIGH")
        mktcap = g1("CUR_MKT_CAP")
        rows.append({
            "ticker":         ticker,
            "pe":             round(float(g1("PE_RATIO")), 2)           if g1("PE_RATIO") else "",
            "forward_pe":     "",  # BQL PE_RATIO é trailing
            "pb":             round(float(g2("PX_TO_BOOK_RATIO")), 2)   if g2("PX_TO_BOOK_RATIO") else "",
            "ev_ebitda":      round(float(g2("EV_TO_T12M_EBITDA")), 2)  if g2("EV_TO_T12M_EBITDA") else "",
            "beta":           round(float(g1("BETA")), 3)               if g1("BETA") else "",
            "mktcap_b":       round(float(mktcap) / 1e9, 2)            if mktcap else "",
            "roe":            round(float(g1("RETURN_COM_EQY")) / 100, 4) if g1("RETURN_COM_EQY") else "",
            "profit_margin":  round(float(g1("PROF_MARGIN")) / 100, 4) if g1("PROF_MARGIN") else "",
            "debt_equity":    round(float(g1("TOT_DEBT_TO_TOT_EQY")), 2) if g1("TOT_DEBT_TO_TOT_EQY") else "",
            "dividend_yield": round(float(g1("EQY_DVD_YLD_IND")) / 100, 4) if g1("EQY_DVD_YLD_IND") else "",
            "price":          round(float(price), 4)                    if price else "",
            "hi_52w":         round(float(hi52), 4)                     if hi52 else "",
            "lo_52w":         round(float(g2("PX_LOW")), 4)             if g2("PX_LOW") else "",
            "drawdown_52w":   round((float(price) - float(hi52)) / float(hi52), 4) if price and hi52 and float(hi52) > 0 else "",
        })

    fields = ["ticker","pe","forward_pe","pb","ev_ebitda","beta","mktcap_b",
              "roe","profit_margin","debt_equity","dividend_yield",
              "price","hi_52w","lo_52w","drawdown_52w"]
    _write_csv(BQL_OUT_DIR / "fundamentals.csv", rows, fields)


# ── 2. Options IV / Skew / PCR ────────────────────────────────────────────────

def export_options_iv(bq):
    print("Options IV (ATM 30D, skew 25D delta, put/call OI)...")
    estr = ", ".join(f'"{t}"' for t in SPX_STOCKS)

    import bql

    # IV ATM 30D + skew 25D
    atm_item   = bq.data.implied_volatility(expiry="30D", pct_moneyness="100")
    put25_item = bq.data.implied_volatility(expiry="30D", delta="25", put_call="PUT")
    cal25_item = bq.data.implied_volatility(expiry="30D", delta="25", put_call="CALL")
    req3 = bql.Request(SPX_STOCKS, {"atm_iv": atm_item, "put25": put25_item, "call25": cal25_item})
    resp3 = bq.execute(req3)
    df3 = bql.combined_df(resp3).groupby(level=0).last()

    # Put/Call OI ratio
    try:
        req4 = bql.Request(SPX_STOCKS, bq.data.put_call_open_interest_ratio())
        resp4 = bq.execute(req4)
        df4 = bql.combined_df(resp4).groupby(level=0).last()
        pcr_col = next((c for c in df4.columns if "PUT_CALL" in c.upper() or "pcr" in c.lower()), df4.columns[0])
    except Exception:
        df4 = None
        pcr_col = None

    rows = []
    for bbg_t in SPX_STOCKS:
        ticker = BBG_TO_TICKER.get(bbg_t, bbg_t)
        r3 = df3.loc[bbg_t] if bbg_t in df3.index else {}
        v_atm = _val(r3, "atm_iv")
        v_put = _val(r3, "put25")
        v_cal = _val(r3, "call25")
        pcr   = None
        if df4 is not None and bbg_t in df4.index and pcr_col:
            pcr = _val(df4.loc[bbg_t], pcr_col)

        rows.append({
            "ticker":    ticker,
            "atm_iv":    round(float(v_atm) / 100, 4)                         if v_atm else "",
            "skew_25d":  round((float(v_put) - float(v_cal)) / 100, 4)        if v_put and v_cal else "",
            "pcr_oi":    round(float(pcr), 3)                                  if pcr else "",
        })

    _write_csv(BQL_OUT_DIR / "options_iv.csv", rows, ["ticker", "atm_iv", "skew_25d", "pcr_oi"])


# ── 3. GEX SPX ────────────────────────────────────────────────────────────────

def export_gex_spx(bq):
    print("GEX SPX (gamma exposure por strike)...")
    import bql
    from datetime import timedelta

    spot_req = bq.execute('get(PX_LAST) for(["SPX Index"])')
    spot_df = bql.combined_df(spot_req)
    spot_df.columns = [c.split("(")[0].strip() for c in spot_df.columns]
    spot = float(spot_df.iloc[0]["PX_LAST"])
    print(f"  SPX spot: {spot:.0f}")

    lo, hi = spot * 0.97, spot * 1.03
    today = date.today()
    exp_limit = today + timedelta(days=10)

    univ = bq.univ.options(
        "SPX Index",
        include_expired=False,
        strike_range=(lo, hi),
    )
    req = bq.execute(
        "get(PX_LAST, OPEN_INT, DELTA, GAMMA) for(@universe)",
        {"@universe": univ}
    )
    df = bql.combined_df(req)
    df = df.reset_index()
    df.columns = [c.split("(")[0].strip() for c in df.columns]

    # Filtra por expiração ≤ 10 dias
    date_col = next((c for c in df.columns if "DATE" in c.upper() or "EXPIRE" in c.upper()), None)
    pc_col   = next((c for c in df.columns if "PUT_CALL" in c.upper() or "PUT" == c.upper()), None)
    stk_col  = next((c for c in df.columns if "STRIKE" in c.upper()), None)

    rows = []
    for _, row in df.iterrows():
        try:
            exp_str = str(row.get(date_col, ""))[:10]
            exp_dt  = date.fromisoformat(exp_str)
            if exp_dt > exp_limit:
                continue
        except Exception:
            continue
        oi    = row.get("OPEN_INT") or 0
        gamma = row.get("GAMMA")    or 0
        pc    = str(row.get(pc_col, "")).upper()
        stk   = row.get(stk_col)   or 0
        gex   = float(oi) * float(gamma) * float(spot) ** 2 / 1e9 * (1 if pc == "CALL" else -1)
        rows.append({
            "expiry":    exp_str,
            "strike":    round(float(stk), 2),
            "put_call":  pc,
            "open_int":  int(oi),
            "gamma":     round(float(gamma), 6),
            "gex_bn":    round(gex, 4),
        })

    _write_csv(BQL_OUT_DIR / "gex_spx.csv", rows, ["expiry","strike","put_call","open_int","gamma","gex_bn"])

    # Resumo GEX
    total_gex = sum(r["gex_bn"] for r in rows)
    call_gex  = sum(r["gex_bn"] for r in rows if r["put_call"] == "CALL")
    put_gex   = sum(r["gex_bn"] for r in rows if r["put_call"] == "PUT")
    direction = "buy" if total_gex > 0.5 else ("sell" if total_gex < -0.5 else "flat")
    regime    = "long" if total_gex > 0 else ("short" if total_gex < 0 else "flat")
    print(f"  GEX total: ${total_gex:+.2f}B | calls={call_gex:+.2f}B puts={put_gex:+.2f}B | regime={regime}")
    _write_csv(BQL_OUT_DIR / "gex_summary.csv", [{
        "date":       str(today),
        "spot":       round(spot, 2),
        "gex_total_bn": round(total_gex, 4),
        "gex_call_bn":  round(call_gex, 4),
        "gex_put_bn":   round(put_gex, 4),
        "direction":  direction,
        "gamma_regime": regime,
        "n_options":  len(rows),
    }], ["date","spot","gex_total_bn","gex_call_bn","gex_put_bn","direction","gamma_regime","n_options"])


# ── 4. LETF Flows ────────────────────────────────────────────────────────────

def export_letf_flows(bq):
    print("LETF NAV + flows (TQQQ, SQQQ, UPRO, SPXS, SOXL, SOXS)...")
    import bql

    LETFS = {
        "TQQQ US Equity": ("TQQQ", "QQQ US Equity", 3, "NDX"),
        "SQQQ US Equity": ("SQQQ", "QQQ US Equity", -3, "NDX"),
        "UPRO US Equity": ("UPRO", "SPY US Equity", 3, "SPX"),
        "SPXS US Equity": ("SPXS", "SPY US Equity", -3, "SPX"),
        "SOXL US Equity": ("SOXL", "SOXX US Equity", 3, "SOX"),
        "SOXS US Equity": ("SOXS", "SOXX US Equity", -3, "SOX"),
    }
    letf_list = list(LETFS.keys())
    lstr = ", ".join(f'"{t}"' for t in letf_list)

    req = bq.execute(f'get(PX_LAST, FUND_TOTAL_ASSETS, FUND_NET_ASSET_VAL) for([{lstr}])')
    df = bql.combined_df(req).groupby(level=0).last()
    df.columns = [c.split("(")[0].strip() for c in df.columns]

    rows = []
    for bbg_t, (sym, under, lev, idx) in LETFS.items():
        r = df.loc[bbg_t] if bbg_t in df.index else {}
        aum = _val(r, "FUND_TOTAL_ASSETS")
        nav = _val(r, "FUND_NET_ASSET_VAL") or _val(r, "PX_LAST")
        rows.append({
            "ticker":    sym,
            "leverage":  lev,
            "index":     idx,
            "nav":       round(float(nav), 4) if nav else "",
            "aum_b":     round(float(aum) / 1e9, 4) if aum else "",
        })

    _write_csv(BQL_OUT_DIR / "letf_flows.csv", rows, ["ticker","leverage","index","nav","aum_b"])


# ── 5. Meta ───────────────────────────────────────────────────────────────────

def export_meta():
    _write_csv(BQL_OUT_DIR / "meta.csv", [{
        "generated_at": datetime.now().isoformat(),
        "date":         str(date.today()),
        "universe":     "SPX_TOP20",
        "tickers":      len(SPX_ALL),
    }], ["generated_at", "date", "universe", "tickers"])


# ── Main ──────────────────────────────────────────────────────────────────────

def run_once(bq):
    errors = []
    for name, fn in [
        ("fundamentals", lambda: export_fundamentals(bq)),
        ("options_iv",   lambda: export_options_iv(bq)),
        ("gex_spx",      lambda: export_gex_spx(bq)),
        ("letf_flows",   lambda: export_letf_flows(bq)),
    ]:
        try:
            fn()
        except Exception as exc:
            print(f"  [ERRO] {name}: {exc}")
            errors.append(name)
    export_meta()
    return errors


if __name__ == "__main__":
    import sys
    import time as _time

    # --loop FLAG: atualiza a cada 3 minutos (para rodar em background no BQuant)
    loop_mode  = "--loop" in sys.argv
    interval   = 180  # segundos

    print(f"=== BQL Export → {BQL_OUT_DIR} ===")
    print(f"Data: {date.today()}")
    if loop_mode:
        print(f"Modo loop: atualiza a cada {interval}s (Ctrl+C para parar)")
    print()

    import bql
    bq = bql.Service()
    print("BQL conectado.\n")

    cycle = 0
    while True:
        cycle += 1
        t0 = _time.time()
        print(f"--- Ciclo {cycle} [{datetime.now().strftime('%H:%M:%S')}] ---")
        errors = run_once(bq)
        elapsed = _time.time() - t0
        if errors:
            print(f"Erros: {errors}")
        else:
            print(f"OK em {elapsed:.0f}s — arquivos em {BQL_OUT_DIR}")

        if not loop_mode:
            break

        sleep = max(10, interval - elapsed)
        print(f"Próxima atualização em {sleep:.0f}s...\n")
        try:
            _time.sleep(sleep)
        except KeyboardInterrupt:
            print("\nLoop encerrado.")
            break
