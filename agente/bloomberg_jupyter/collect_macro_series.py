"""
Bloomberg BQuant — Coleta de Séries Macroeconômicas
=====================================================

Cole este código em uma célula Jupyter do BQuant e execute.
Ou rode: bqnt_python bloomberg_jupyter/collect_macro_series.py

O que coleta:
  CURVA DE JUROS EUA
    - US 1M, 3M, 6M, 1yr, 2yr, 5yr, 10yr, 30yr
    - Spread 2yr-10yr (calculado)

  SPREADS DE CRÉDITO
    - IG OAS Spread (LUACOAS)
    - HY OAS Spread (LF98OAS)
    - CDX IG 5yr
    - CDX HY 5yr

  VOLATILIDADE / RISCO
    - VIX Spot, VIX9D, VIX3M (term structure)
    - VVIX (vol do vol)
    - MOVE Index (vol de bonds)

  LIQUIDEZ / BANCOS CENTRAIS
    - Fed Funds Rate (FDFD)
    - SOFR
    - US 10yr Breakeven (inflação implícita)
    - US 2yr Breakeven

  GLOBAL
    - MSCI EM, MSCI World
    - EuroStoxx 50
    - DXY
    - Gold, Brent

  HISTÓRICO
    - Todas as séries com histórico de 252 dias

Saídas:
  - macro_series_{today}.csv     → último valor por série
  - macro_history_{today}.csv    → histórico diário
  - meta_{today}.csv             → timestamp

Formato macro_series CSV:
  bbg_ticker, description, category, px_last, date

Formato macro_history CSV:
  date, bbg_ticker, description, category, value
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from export_utils import (
    MACRO_UNIVERSE,
    ensure_output_dir,
    log,
    make_zip,
    safe_float,
    today_str,
    write_csv,
    write_meta_csv,
)

try:
    import bql
    bq = bql.Service()
    log("BQL Service iniciado com sucesso.")
except ImportError:
    raise SystemExit("Execute no ambiente BQuant/BQNT.")


# ── Categorias para enriquecer o dado no banco ────────────────────────────────
MACRO_CATEGORIES: dict[str, str] = {
    "USGG1M Index":   "rates_usd",
    "USGG3M Index":   "rates_usd",
    "USGG6M Index":   "rates_usd",
    "USGG1YR Index":  "rates_usd",
    "USGG2YR Index":  "rates_usd",
    "USGG5YR Index":  "rates_usd",
    "USGG10YR Index": "rates_usd",
    "USGG30YR Index": "rates_usd",
    "LUACOAS Index":  "credit_spread",
    "LF98OAS Index":  "credit_spread",
    "CDXIG Index":    "credit_spread",
    "CDXHY Index":    "credit_spread",
    "VIX Index":      "volatility",
    "VIX9D Index":    "volatility",
    "VIX3M Index":    "volatility",
    "VVIX Index":     "volatility",
    "MOVE Index":     "volatility",
    "DXY Curncy":     "fx",
    "FDFD Index":     "monetary",
    "SOFRRATE Index": "monetary",
    "USSWIT10 Index": "inflation",
    "USSWIT2 Index":  "inflation",
    "MXEF Index":     "global_equity",
    "MXWO Index":     "global_equity",
    "SX5E Index":     "global_equity",
    "XAU Curncy":     "commodity",
    "CO1 Comdty":     "commodity",
}


def collect_macro_snapshot() -> list[dict]:
    """Coleta o último valor de cada série macro."""
    log("Coletando snapshot de séries macro...")
    bbg_tickers = [t[0] for t in MACRO_UNIVERSE]
    desc_map    = {t[0]: t[2] for t in MACRO_UNIVERSE}
    today       = today_str()

    items = {
        "px_last": bq.data.px_last(dates=bq.func.range("0D", "0D"), fill="PREV"),
    }

    log("  Executando query BQL...")
    resp = bq.execute(bql.Request(bbg_tickers, items))

    px_map: dict[str, float | None] = {}
    try:
        df = resp["px_last"].df()
        for ticker, row in df.iterrows():
            px_map[str(ticker).strip()] = safe_float(row.get("px_last") or row.iloc[0])
    except Exception as exc:
        log(f"  AVISO: extração falhou: {exc}")

    rows = []
    for bbg, field, desc in MACRO_UNIVERSE:
        rows.append({
            "bbg_ticker":  bbg,
            "description": desc,
            "category":    MACRO_CATEGORIES.get(bbg, "other"),
            "px_last":     px_map.get(bbg),
            "date":        today,
        })

    # Derivados calculados
    # Spread 2yr-10yr
    us2y  = px_map.get("USGG2YR Index")
    us10y = px_map.get("USGG10YR Index")
    if us2y is not None and us10y is not None:
        rows.append({
            "bbg_ticker":  "US_2Y10Y_SPREAD",
            "description": "US 2yr-10yr Spread (calculado)",
            "category":    "rates_derived",
            "px_last":     round(us10y - us2y, 4),
            "date":        today,
        })

    # Spread 5yr-30yr
    us5y  = px_map.get("USGG5YR Index")
    us30y = px_map.get("USGG30YR Index")
    if us5y is not None and us30y is not None:
        rows.append({
            "bbg_ticker":  "US_5Y30Y_SPREAD",
            "description": "US 5yr-30yr Spread (calculado)",
            "category":    "rates_derived",
            "px_last":     round(us30y - us5y, 4),
            "date":        today,
        })

    # VIX term structure: contango/backwardation
    vix9d  = px_map.get("VIX9D Index")
    vix3m  = px_map.get("VIX3M Index")
    vix_sp = px_map.get("VIX Index")
    if vix9d and vix_sp and vix_sp != 0:
        rows.append({
            "bbg_ticker":  "VIX_TERM_9D_SP",
            "description": "VIX 9D / VIX Spot (term structure)",
            "category":    "volatility_derived",
            "px_last":     round(vix9d / vix_sp, 4),
            "date":        today,
        })
    if vix3m and vix_sp and vix_sp != 0:
        rows.append({
            "bbg_ticker":  "VIX_TERM_3M_SP",
            "description": "VIX 3M / VIX Spot (term structure)",
            "category":    "volatility_derived",
            "px_last":     round(vix3m / vix_sp, 4),
            "date":        today,
        })

    ok_count = sum(1 for r in rows if r["px_last"] is not None)
    log(f"  {ok_count}/{len(rows)} séries com valor disponível.")
    return rows


def collect_macro_history(days: int = 252) -> list[dict]:
    """Coleta histórico diário de todas as séries macro."""
    log(f"Coletando histórico macro de {days} dias...")
    bbg_tickers = [t[0] for t in MACRO_UNIVERSE]
    desc_map    = {t[0]: t[2] for t in MACRO_UNIVERSE}

    items = {
        "px_last": bq.data.px_last(
            dates=bq.func.range(f"-{days}D", "0D"),
            fill="PREV",
            frq="D",
        )
    }

    log("  Executando query histórica BQL...")
    resp = bq.execute(bql.Request(bbg_tickers, items))

    rows = []
    try:
        df = resp["px_last"].df()
        for (ticker, date), row in df.iterrows():
            bbg  = str(ticker).strip()
            val  = safe_float(row.get("px_last") or row.iloc[0])
            rows.append({
                "date":        str(date)[:10],
                "bbg_ticker":  bbg,
                "description": desc_map.get(bbg, bbg),
                "category":    MACRO_CATEGORIES.get(bbg, "other"),
                "value":       val,
            })
    except Exception as exc:
        log(f"  AVISO: histórico multi-index falhou: {exc}")
        # Fallback: por ticker
        for bbg, field, desc in MACRO_UNIVERSE:
            try:
                df_t = resp["px_last"].df().xs(bbg, level=0)
                for date, row in df_t.iterrows():
                    rows.append({
                        "date":        str(date)[:10],
                        "bbg_ticker":  bbg,
                        "description": desc,
                        "category":    MACRO_CATEGORIES.get(bbg, "other"),
                        "value":       safe_float(row.iloc[0]),
                    })
            except Exception:
                continue

    log(f"  {len(rows)} linhas de histórico macro coletadas.")
    return rows


def main() -> None:
    out = ensure_output_dir()
    log(f"Pasta de saída: {out}")
    csv_files = []

    snapshot_rows = collect_macro_snapshot()
    if snapshot_rows:
        csv_files.append(
            write_csv(out, "macro_series", snapshot_rows,
                      fieldnames=["bbg_ticker", "description", "category", "px_last", "date"])
        )

    history_rows = collect_macro_history(days=252)
    if history_rows:
        csv_files.append(
            write_csv(out, "macro_history", history_rows,
                      fieldnames=["date", "bbg_ticker", "description", "category", "value"])
        )

    csv_files.append(write_meta_csv(out, {
        "script":  "collect_macro_series.py",
        "series":  str(len(MACRO_UNIVERSE)),
    }))

    make_zip(csv_files, out, zip_prefix="bql_macro")
    log("collect_macro_series.py concluído.")


if __name__ == "__main__":
    main()
