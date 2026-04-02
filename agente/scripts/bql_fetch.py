"""
Script BQL — roda sob BQuant Python 3.11 (C:/blp/bqnt/environments/bqnt-3/python.exe)

Coleta fundamentais + options (IV, skew, put/call) via Bloomberg BQL.

Uso:
    bqnt_python bql_fetch.py AAPL MSFT JNJ ...
    bqnt_python bql_fetch.py --file tickers.txt

Saída: JSON para stdout
  {
    "AAPL": {"pe": 32.2, "mktcap_b": 3741.0, "beta": 1.2,
             "atm_iv": 0.28, "skew_5pct": 0.04, ...},
    ...
  }
"""

from __future__ import annotations

import json
import sys
import warnings

warnings.filterwarnings("ignore")

# ── Ticker normalização ────────────────────────────────────────────────────────

_SUFFIX_MAP = {
    "^GSPC":   "SPX Index",
    "^NDX":    "NDX Index",
    "^VIX":    "VIX Index",
    "^RUT":    "RTY Index",
    "GC=F":    "GC1 Comdty",
    "CL=F":    "CL1 Comdty",
    "SI=F":    "SI1 Comdty",
    "ZB=F":    "US1 Comdty",
    "BTC-USD": "XBT Curncy",
}

_SKIP = {"=X", "=F", "-USD", "-EUR", ".HK", ".SS", ".BO", ".NS", ".NYB"}


def _to_bbg(ticker: str) -> str | None:
    if ticker in _SUFFIX_MAP:
        return _SUFFIX_MAP[ticker]
    for skip in _SKIP:
        if ticker.endswith(skip):
            return None
    if ticker.startswith("^"):
        return None
    return f"{ticker} US Equity"


# ── BQL queries ────────────────────────────────────────────────────────────────

def fetch(tickers: list[str]) -> dict[str, dict]:
    bbg_map: dict[str, str] = {}  # bbg_ticker → original_ticker
    for t in tickers:
        b = _to_bbg(t)
        if b:
            bbg_map[b] = t

    if not bbg_map:
        return {}

    bbg_list = list(bbg_map.keys())
    # Apenas equities para queries que não fazem sentido para índices/FX/Comdty
    equity_list = [b for b in bbg_list if b.endswith("Equity")]
    t_str      = ", ".join(f'"{t}"' for t in bbg_list)
    eq_str     = ", ".join(f'"{t}"' for t in equity_list)

    import bql
    bq = bql.Service()

    results: dict[str, dict] = {}

    # ── Query 1: fundamentais ─────────────────────────────────────────────────
    try:
        req = bq.execute(
            f"get("
            f"  PE_RATIO,"
            f"  CUR_MKT_CAP,"
            f"  BETA,"
            f"  PROF_MARGIN,"
            f"  TOT_DEBT_TO_TOT_EQY,"
            f"  RETURN_COM_EQY,"
            f"  EQY_DVD_YLD_IND,"
            f"  PX_LAST"
            f") for([{t_str}])"
        )
        df = bql.combined_df(req)
        df = df.groupby(level=0).last()
        # BQL retorna colunas como "PE_RATIO()" — normaliza removendo parênteses
        df.columns = [c.split("(")[0].strip() for c in df.columns]

        FIELDS = {
            "PE_RATIO":            "pe",
            "CUR_MKT_CAP":         "mktcap_b",
            "BETA":                "beta",
            "PROF_MARGIN":         "profit_margin",
            "TOT_DEBT_TO_TOT_EQY": "debt_equity",
            "RETURN_COM_EQY":      "roe",
            "EQY_DVD_YLD_IND":     "dividend_yield",
            "PX_LAST":             "price",
        }

        for bbg_t, row in df.iterrows():
            orig = bbg_map.get(bbg_t, bbg_t)
            entry: dict = {}
            for bql_f, key in FIELDS.items():
                val = row.get(bql_f)
                if val is not None and val == val:
                    f = float(val)
                    if key == "mktcap_b":
                        f = round(f * 1e-9, 4)
                    elif key in ("profit_margin", "roe", "dividend_yield"):
                        f = round(f * 0.01, 6)
                    else:
                        f = round(f, 4)
                    entry[key] = f
            if entry:
                results[orig] = entry

    except Exception as exc:
        print(f"[bql_fetch] fundamentals error: {exc}", file=sys.stderr)

    # ── Query 2: 52w high/low ─────────────────────────────────────────────────
    try:
        req2 = bq.execute(
            f"get("
            f"  PX_HIGH(dates=range(-365D,0D), frq=Y),"
            f"  PX_LOW(dates=range(-365D,0D), frq=Y)"
            f") for([{t_str}])"
        )
        df2 = bql.combined_df(req2)
        df2 = df2.groupby(level=0).last()
        df2.columns = [c.split("(")[0].strip() for c in df2.columns]
        cols = df2.columns.tolist()
        hi_col = next((c for c in cols if "HIGH" in c or "high" in c), None)
        lo_col = next((c for c in cols if "LOW" in c or "low" in c), None)

        for bbg_t, row in df2.iterrows():
            orig = bbg_map.get(bbg_t, bbg_t)
            entry = results.setdefault(orig, {})
            if hi_col:
                v = row.get(hi_col)
                if v == v and v is not None:
                    entry["hi_52w"] = round(float(v), 4)
            if lo_col:
                v = row.get(lo_col)
                if v == v and v is not None:
                    entry["lo_52w"] = round(float(v), 4)
            price = entry.get("price")
            hi = entry.get("hi_52w")
            if price and hi and hi > 0:
                entry["drawdown_52w"] = round((price - hi) / hi, 4)

    except Exception as exc:
        print(f"[bql_fetch] 52w error: {exc}", file=sys.stderr)

    # ── Query 3: options — IV 30D ATM, skew 25D, put/call ratio ──────────────
    # Só equities têm opções líquidas no Bloomberg
    if equity_list:
        try:
            # Campos corretos do repo rbbentes-design/bbq (greeks_dashboard.py)
            atm_item   = bq.data.implied_volatility(expiry="30D", pct_moneyness="100")
            put25_item = bq.data.implied_volatility(expiry="30D", delta="25", put_call="PUT")
            cal25_item = bq.data.implied_volatility(expiry="30D", delta="25", put_call="CALL")
            req3 = bql.Request(equity_list, {"atm_iv": atm_item, "put25": put25_item, "call25": cal25_item})
            resp3 = bq.execute(req3)
            df3 = bql.combined_df(resp3).groupby(level=0).last()

            for bbg_t, row in df3.iterrows():
                orig = bbg_map.get(bbg_t, bbg_t)
                entry = results.setdefault(orig, {})
                v_atm = row.get("atm_iv")
                if v_atm is not None and v_atm == v_atm:
                    entry["atm_iv"] = round(float(v_atm) * 0.01, 4)  # % → decimal
                v_put = row.get("put25")
                v_cal = row.get("call25")
                if (v_put is not None and v_put == v_put and
                        v_cal is not None and v_cal == v_cal):
                    # Skew: put25 delta IV - call25 delta IV (put skew premium)
                    entry["skew_5pct"] = round((float(v_put) - float(v_cal)) * 0.01, 4)

        except Exception as exc:
            print(f"[bql_fetch] options IV error: {exc}", file=sys.stderr)

        # ── Query 4: put/call open interest ratio ─────────────────────────────
        try:
            req4 = bql.Request(equity_list, bq.data.put_call_open_interest_ratio())
            resp4 = bq.execute(req4)
            df4 = bql.combined_df(resp4).groupby(level=0).last()
            for bbg_t, row in df4.iterrows():
                orig = bbg_map.get(bbg_t, bbg_t)
                entry = results.setdefault(orig, {})
                col = next((c for c in df4.columns if "PUT_CALL" in c.upper() or "pcr" in c.lower()), None)
                if col:
                    v = row.get(col)
                    if v is not None and v == v:
                        entry["pcr_oi"] = round(float(v), 4)
        except Exception as exc:
            print(f"[bql_fetch] pcr error: {exc}", file=sys.stderr)

    return results


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("{}")
        sys.exit(0)

    if args[0] == "--file":
        with open(args[1]) as f:
            tickers = [l.strip() for l in f if l.strip()]
    else:
        tickers = args

    data = fetch(tickers)
    print(json.dumps(data, ensure_ascii=False))
