"""
Bloomberg BQuant — Coleta de Preços
======================================

Cole este código em uma célula Jupyter do BQuant e execute.
Ou rode: bqnt_python bloomberg_jupyter/collect_prices.py

O que coleta:
  - Preço atual (px_last)
  - Fechamento anterior (px_last D-1)
  - Preço 5 dias atrás (px_last D-5, para retorno semanal)
  - Preço 1-Jan do ano corrente (para retorno YTD)
  - Histórico diário dos últimos 252 dias (para gráficos e RRG)
  - Volume médio 20 dias
  - Máxima e mínima 52 semanas

Saídas:
  - prices_{today}.csv          → preços e retornos atuais
  - price_history_{today}.csv   → histórico de fechamento diário
  - meta_{today}.csv            → timestamp do export

Formato prices CSV:
  bbg_ticker, yf_ticker, name, price, prev_price, price_w, price_ytd,
  daily_return, weekly_return, ytd_return, vol_20d, hi_52w, lo_52w

Formato price_history CSV:
  date, bbg_ticker, yf_ticker, price
"""

# ── Imports ────────────────────────────────────────────────────────────────
import sys
from datetime import datetime
from pathlib import Path

# Adiciona bloomberg_jupyter ao path para import de export_utils
sys.path.insert(0, str(Path(__file__).parent))
from export_utils import (
    PRICE_UNIVERSE,
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
    BQL_AVAILABLE = True
    log("BQL Service iniciado com sucesso.")
except ImportError:
    BQL_AVAILABLE = False
    log("AVISO: bql não disponível. Este script deve rodar no BQuant.")
    raise SystemExit("Execute no ambiente BQuant/BQNT.")


def collect_prices() -> list[dict]:
    """
    Coleta preços atuais, prevs e retornos calculados.
    Retorna lista de dicts para o CSV.
    """
    log("Coletando preços atuais...")
    bbg_tickers = [t[0] for t in PRICE_UNIVERSE]
    ticker_map  = {t[0]: t for t in PRICE_UNIVERSE}

    today_date = datetime.now().strftime("%Y-%m-%d")
    ytd_start  = f"{datetime.now().year}-01-01"

    # ── Consultas BQL ─────────────────────────────────────────────────────
    items = {
        "px_now":    bq.data.px_last(dates=bq.func.range("0D", "0D"), fill="PREV"),
        "px_prev":   bq.data.px_last(dates=bq.func.range("-1D", "-1D"), fill="PREV"),
        "px_week":   bq.data.px_last(dates=bq.func.range("-6D", "-6D"), fill="PREV"),
        "px_ytd":    bq.data.px_last(dates=bq.func.range(ytd_start, ytd_start), fill="PREV"),
        "vol_20d":   bq.data.volatility_20d(dates=bq.func.range("0D", "0D"), fill="PREV"),
        "px_hi_52w": bq.data.px_high(dates=bq.func.range("-365D", "0D"), fill="PREV"),
        "px_lo_52w": bq.data.px_low(dates=bq.func.range("-365D", "0D"), fill="PREV"),
    }

    log("  Executando queries BQL...")
    resp = bq.execute(bql.Request(bbg_tickers, items))

    # ── Extrai dados ───────────────────────────────────────────────────────
    def _get_field(field_name: str) -> dict[str, float | None]:
        try:
            return {
                str(k).strip(): safe_float(v)
                for k, v in resp[field_name].df()[field_name].items()
            }
        except Exception as exc:
            log(f"  AVISO: campo {field_name} falhou: {exc}")
            return {}

    px_now    = _get_field("px_now")
    px_prev   = _get_field("px_prev")
    px_week   = _get_field("px_week")
    px_ytd    = _get_field("px_ytd")
    vol_20d   = _get_field("vol_20d")
    px_hi_52w = _get_field("px_hi_52w")
    px_lo_52w = _get_field("px_lo_52w")

    # ── Monta CSV ─────────────────────────────────────────────────────────
    rows = []
    for bbg, yf, name in PRICE_UNIVERSE:
        price   = px_now.get(bbg)
        prev    = px_prev.get(bbg)
        pw      = px_week.get(bbg)
        pytd    = px_ytd.get(bbg)
        hi52    = px_hi_52w.get(bbg)
        lo52    = px_lo_52w.get(bbg)
        vol     = vol_20d.get(bbg)

        # Retornos calculados (None se referência indisponível)
        daily_ret  = round((price - prev)  / prev,  6) if (price and prev  and prev  != 0) else None
        weekly_ret = round((price - pw)    / pw,    6) if (price and pw    and pw    != 0) else None
        ytd_ret    = round((price - pytd)  / pytd,  6) if (price and pytd  and pytd  != 0) else None
        dd_52w     = round((price - hi52)  / hi52,  6) if (price and hi52  and hi52  != 0) else None

        rows.append({
            "bbg_ticker":    bbg,
            "yf_ticker":     yf,
            "name":          name,
            "date":          today_date,
            "price":         price,
            "prev_price":    prev,
            "price_w":       pw,
            "price_ytd":     pytd,
            "daily_return":  daily_ret,
            "weekly_return": weekly_ret,
            "ytd_return":    ytd_ret,
            "vol_20d":       vol,
            "hi_52w":        hi52,
            "lo_52w":        lo52,
            "drawdown_52w":  dd_52w,
        })

    ok_count = sum(1 for r in rows if r["price"] is not None)
    log(f"  {ok_count}/{len(rows)} tickers com preço disponível.")
    return rows


def collect_price_history(days: int = 252) -> list[dict]:
    """
    Coleta histórico de fechamento diário para todos os tickers do universo.
    Retorna lista de dicts para o CSV.
    """
    log(f"Coletando histórico de {days} dias...")
    bbg_tickers = [t[0] for t in PRICE_UNIVERSE]
    ticker_map  = {t[0]: (t[1], t[2]) for t in PRICE_UNIVERSE}

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
            val = safe_float(row.get("px_last") or row.iloc[0])
            yf, name = ticker_map.get(str(ticker).strip(), (str(ticker), str(ticker)))
            rows.append({
                "date":       str(date)[:10],
                "bbg_ticker": str(ticker).strip(),
                "yf_ticker":  yf,
                "name":       name,
                "price":      val,
            })
    except Exception as exc:
        log(f"  AVISO: histórico falhou com MultiIndex — tentando por ticker: {exc}")
        # Fallback: processa ticker por ticker (mais lento mas robusto)
        for bbg, yf, name in PRICE_UNIVERSE:
            try:
                df_t = resp["px_last"].df().xs(bbg, level=0)
                for date, row in df_t.iterrows():
                    val = safe_float(row.iloc[0])
                    rows.append({
                        "date":       str(date)[:10],
                        "bbg_ticker": bbg,
                        "yf_ticker":  yf,
                        "name":       name,
                        "price":      val,
                    })
            except Exception:
                continue

    log(f"  {len(rows)} linhas de histórico coletadas.")
    return rows


def main() -> None:
    out = ensure_output_dir()
    log(f"Pasta de saída: {out}")
    csv_files = []

    # Coleta preços
    prices_rows = collect_prices()
    if prices_rows:
        csv_files.append(
            write_csv(out, "prices", prices_rows, fieldnames=[
                "bbg_ticker", "yf_ticker", "name", "date", "price", "prev_price",
                "price_w", "price_ytd", "daily_return", "weekly_return", "ytd_return",
                "vol_20d", "hi_52w", "lo_52w", "drawdown_52w",
            ])
        )

    # Coleta histórico
    history_rows = collect_price_history(days=252)
    if history_rows:
        csv_files.append(
            write_csv(out, "price_history", history_rows,
                      fieldnames=["date", "bbg_ticker", "yf_ticker", "name", "price"])
        )

    # Meta
    csv_files.append(write_meta_csv(out, {
        "script":  "collect_prices.py",
        "tickers": str(len(PRICE_UNIVERSE)),
        "history": "252 dias",
    }))

    # Empacota em ZIP
    make_zip(csv_files, out, zip_prefix="bql_data")
    log("collect_prices.py concluído.")


if __name__ == "__main__":
    main()
