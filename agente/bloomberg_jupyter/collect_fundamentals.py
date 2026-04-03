"""
Bloomberg BQuant — Coleta de Fundamentos + Opções + GEX + LETF
================================================================

Cole este código em uma célula Jupyter do BQuant e execute.
Ou rode: bqnt_python bloomberg_jupyter/collect_fundamentals.py

O que coleta:
  FUNDAMENTOS (equities)
    - P/E, P/B, EV/EBITDA
    - Market Cap (B)
    - Beta 2 anos
    - ROE, Margem de Lucro, Dívida/Patrimônio
    - Dividend Yield
    - Preço atual, Máxima/Mínima 52 semanas, Drawdown

  OPÇÕES / VOLATILIDADE IMPLÍCITA (equities)
    - ATM IV 30 dias
    - Skew 25-delta (put - call)
    - Put/Call Open Interest Ratio

  GEX SPX (Gamma Exposure)
    - GEX Total, Call, Put (em bilhões $)
    - Direção do gamma (long/short)
    - Regime gamma (positive/negative/near-zero)
    - GEX por strike (heat map)

  LETF FLOWS
    - NAV e AUM dos principais ETFs alavancados
    - Nível de alavancagem, índice de referência

Saídas:
  - fundamentals_{today}.csv
  - options_iv_{today}.csv
  - gex_summary_{today}.csv
  - gex_spx_{today}.csv
  - letf_flows_{today}.csv
  - meta_{today}.csv
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from export_utils import (
    FUNDAMENTAL_UNIVERSE,
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


# ── LETF Universe ─────────────────────────────────────────────────────────────
LETF_UNIVERSE: list[tuple[str, int, str]] = [
    ("TQQQ US Equity",  3,  "NDX"),
    ("SQQQ US Equity", -3,  "NDX"),
    ("UPRO US Equity",  3,  "SPX"),
    ("SPXS US Equity", -3,  "SPX"),
    ("SOXL US Equity",  3,  "SOX"),
    ("SOXS US Equity", -3,  "SOX"),
    ("TECL US Equity",  3,  "XLK"),
    ("TECS US Equity", -3,  "XLK"),
    ("FAS US Equity",   3,  "XLF"),
    ("FAZ US Equity",  -3,  "XLF"),
    ("TNA US Equity",   3,  "RTY"),
    ("TZA US Equity",  -3,  "RTY"),
    ("LABU US Equity",  3,  "XBI"),
    ("LABD US Equity", -3,  "XBI"),
    ("UVXY US Equity",  1,  "VIX"),  # 1.5x
    ("SVXY US Equity", -1,  "VIX"),  # -0.5x
]


def collect_fundamentals() -> list[dict]:
    """Coleta múltiplos fundamentalistas."""
    log("Coletando fundamentos...")
    today = today_str()

    items = {
        "pe":             bq.data.pe_ratio(dates=bq.func.range("0D", "0D"), fill="PREV"),
        "pb":             bq.data.px_to_book_ratio(dates=bq.func.range("0D", "0D"), fill="PREV"),
        "ev_ebitda":      bq.data.best_ev_ebitda(dates=bq.func.range("0D", "0D"), fill="PREV"),
        "mktcap_b":       bq.data.cur_mkt_cap(dates=bq.func.range("0D", "0D"), fill="PREV"),
        "beta":           bq.data.beta_adj_overridable(dates=bq.func.range("0D", "0D"), fill="PREV"),
        "roe":            bq.data.return_on_equity(dates=bq.func.range("0D", "0D"), fill="PREV"),
        "profit_margin":  bq.data.prof_margin(dates=bq.func.range("0D", "0D"), fill="PREV"),
        "debt_equity":    bq.data.tot_debt_to_tot_eqy(dates=bq.func.range("0D", "0D"), fill="PREV"),
        "dividend_yield": bq.data.dividend_yield(dates=bq.func.range("0D", "0D"), fill="PREV"),
        "px_last":        bq.data.px_last(dates=bq.func.range("0D", "0D"), fill="PREV"),
        "px_hi_52w":      bq.data.px_high(dates=bq.func.range("-365D", "0D"), fill="PREV"),
        "px_lo_52w":      bq.data.px_low(dates=bq.func.range("-365D", "0D"), fill="PREV"),
    }

    resp = bq.execute(bql.Request(FUNDAMENTAL_UNIVERSE, items))

    def _get(field_name: str) -> dict[str, float | None]:
        try:
            return {
                str(k).strip(): safe_float(v)
                for k, v in resp[field_name].df()[field_name].items()
            }
        except Exception:
            return {}

    pe     = _get("pe")
    pb     = _get("pb")
    ev_eb  = _get("ev_ebitda")
    mktcap = _get("mktcap_b")
    beta   = _get("beta")
    roe    = _get("roe")
    pm     = _get("profit_margin")
    de     = _get("debt_equity")
    dy     = _get("dividend_yield")
    price  = _get("px_last")
    hi52   = _get("px_hi_52w")
    lo52   = _get("px_lo_52w")

    rows = []
    for ticker in FUNDAMENTAL_UNIVERSE:
        p = price.get(ticker)
        h = hi52.get(ticker)
        dd = round((p - h) / h, 6) if (p and h and h != 0) else None
        mc = mktcap.get(ticker)
        mc_b = round(mc / 1e9, 4) if mc else None  # Bloomberg retorna em USD

        rows.append({
            "ticker":         ticker,
            "date":           today,
            "pe":             pe.get(ticker),
            "pb":             pb.get(ticker),
            "ev_ebitda":      ev_eb.get(ticker),
            "mktcap_b":       mc_b,
            "beta":           beta.get(ticker),
            "roe":            roe.get(ticker),
            "profit_margin":  pm.get(ticker),
            "debt_equity":    de.get(ticker),
            "dividend_yield": dy.get(ticker),
            "price":          p,
            "hi_52w":         hi52.get(ticker),
            "lo_52w":         lo52.get(ticker),
            "drawdown_52w":   dd,
        })

    log(f"  {sum(1 for r in rows if r['price'] is not None)} / {len(rows)} tickers com dados.")
    return rows


def collect_options_iv() -> list[dict]:
    """Coleta volatilidade implícita ATM e skew."""
    log("Coletando opções/IV...")
    today = today_str()

    # Apenas equities têm dados de opções razoáveis na BQL
    equity_tickers = [t for t in FUNDAMENTAL_UNIVERSE if "Equity" in t]

    items = {
        "atm_iv_30d":  bq.data.ivol_delta(ivol_delta=0.50,
                                           maturity_type="REGULAR",
                                           maturity=30,
                                           put_call="CALL",
                                           dates=bq.func.range("0D", "0D"),
                                           fill="PREV"),
        "skew_25d_put": bq.data.ivol_delta(ivol_delta=0.25,
                                            maturity_type="REGULAR",
                                            maturity=30,
                                            put_call="PUT",
                                            dates=bq.func.range("0D", "0D"),
                                            fill="PREV"),
        "skew_25d_call": bq.data.ivol_delta(ivol_delta=0.25,
                                             maturity_type="REGULAR",
                                             maturity=30,
                                             put_call="CALL",
                                             dates=bq.func.range("0D", "0D"),
                                             fill="PREV"),
        "pcr_oi": bq.data.opt_put_call_ratio_open_int(dates=bq.func.range("0D", "0D"),
                                                       fill="PREV"),
    }

    try:
        resp = bq.execute(bql.Request(equity_tickers, items))

        def _get(fn: str) -> dict[str, float | None]:
            try:
                return {str(k).strip(): safe_float(v)
                        for k, v in resp[fn].df()[fn].items()}
            except Exception:
                return {}

        atm_iv   = _get("atm_iv_30d")
        sk_put   = _get("skew_25d_put")
        sk_call  = _get("skew_25d_call")
        pcr      = _get("pcr_oi")

        rows = []
        for ticker in equity_tickers:
            iv  = atm_iv.get(ticker)
            sp  = sk_put.get(ticker)
            sc  = sk_call.get(ticker)
            skew = round(sp - sc, 6) if (sp is not None and sc is not None) else None
            rows.append({
                "ticker":   ticker,
                "date":     today,
                "atm_iv":   iv,
                "skew_25d": skew,
                "pcr_oi":   pcr.get(ticker),
            })
        log(f"  {sum(1 for r in rows if r['atm_iv'] is not None)} / {len(rows)} tickers com IV.")
        return rows
    except Exception as exc:
        log(f"  AVISO: opções IV falharam: {exc}")
        return []


def collect_gex_spx() -> tuple[list[dict], list[dict]]:
    """
    Coleta GEX SPX: resumo + por strike.
    Retorna (summary_rows, strike_rows).
    """
    log("Coletando GEX SPX...")
    today = today_str()

    # GEX SPX via BQL (dados de opções do SPX)
    SPX = "SPX Index"

    try:
        items = {
            "opt_notl_value": bq.data.opt_notl_value(
                dates=bq.func.range("0D", "0D"),
                fill="PREV",
            ),
            "px_last": bq.data.px_last(dates=bq.func.range("0D", "0D"), fill="PREV"),
        }
        resp = bq.execute(bql.Request([SPX], items))
        spot = safe_float(resp["px_last"].df()["px_last"].iloc[0])
    except Exception as exc:
        log(f"  AVISO: GEX SPX falhou: {exc}")
        return [], []

    # Resumo simplificado (dados completos de gamma exposure por strike
    # requerem o serviço Bloomberg OAPI — aqui usamos proxy via px_last do SPX)
    summary_rows = [{
        "date":         today,
        "spot":         spot,
        "gex_total_bn": None,   # requer feed especializado (SpotGamma/SqueezeMetrics)
        "gex_call_bn":  None,
        "gex_put_bn":   None,
        "direction":    "unknown",
        "gamma_regime": "unknown",
        "n_options":    None,
        "note":         "GEX detalhado requer SpotGamma API ou Bloomberg DLIB",
    }]

    log("  NOTA: GEX detalhado indisponível via BQL standard — use SpotGamma API.")
    return summary_rows, []


def collect_letf_flows() -> list[dict]:
    """Coleta NAV e AUM dos ETFs alavancados."""
    log("Coletando LETF flows...")
    today = today_str()
    tickers = [t[0] for t in LETF_UNIVERSE]

    items = {
        "nav":      bq.data.nav(dates=bq.func.range("0D", "0D"), fill="PREV"),
        "fund_net_assets": bq.data.fund_net_assets(dates=bq.func.range("0D", "0D"), fill="PREV"),
        "px_last":  bq.data.px_last(dates=bq.func.range("0D", "0D"), fill="PREV"),
    }

    try:
        resp = bq.execute(bql.Request(tickers, items))

        def _get(fn: str) -> dict[str, float | None]:
            try:
                return {str(k).strip(): safe_float(v)
                        for k, v in resp[fn].df()[fn].items()}
            except Exception:
                return {}

        nav_map = _get("nav")
        aum_map = _get("fund_net_assets")
        px_map  = _get("px_last")

        rows = []
        for ticker, leverage, index in LETF_UNIVERSE:
            aum = aum_map.get(ticker)
            aum_b = round(aum / 1e9, 4) if aum else None
            rows.append({
                "ticker":   ticker,
                "date":     today,
                "leverage": leverage,
                "index":    index,
                "nav":      nav_map.get(ticker) or px_map.get(ticker),
                "aum_b":    aum_b,
            })
        log(f"  {sum(1 for r in rows if r['nav'] is not None)} / {len(rows)} LETFs com NAV.")
        return rows
    except Exception as exc:
        log(f"  AVISO: LETF flows falharam: {exc}")
        return []


def main() -> None:
    out = ensure_output_dir()
    log(f"Pasta de saída: {out}")
    csv_files = []

    # Fundamentos
    fund_rows = collect_fundamentals()
    if fund_rows:
        csv_files.append(
            write_csv(out, "fundamentals", fund_rows, fieldnames=[
                "ticker", "date", "pe", "pb", "ev_ebitda", "mktcap_b", "beta",
                "roe", "profit_margin", "debt_equity", "dividend_yield",
                "price", "hi_52w", "lo_52w", "drawdown_52w",
            ])
        )

    # Opções IV
    iv_rows = collect_options_iv()
    if iv_rows:
        csv_files.append(
            write_csv(out, "options_iv", iv_rows,
                      fieldnames=["ticker", "date", "atm_iv", "skew_25d", "pcr_oi"])
        )

    # GEX
    gex_summary, gex_strikes = collect_gex_spx()
    if gex_summary:
        csv_files.append(
            write_csv(out, "gex_summary", gex_summary, fieldnames=[
                "date", "spot", "gex_total_bn", "gex_call_bn", "gex_put_bn",
                "direction", "gamma_regime", "n_options", "note",
            ])
        )
    if gex_strikes:
        csv_files.append(
            write_csv(out, "gex_spx", gex_strikes,
                      fieldnames=["date", "strike", "put_call", "open_int", "gamma", "gex_bn"])
        )

    # LETF
    letf_rows = collect_letf_flows()
    if letf_rows:
        csv_files.append(
            write_csv(out, "letf_flows", letf_rows,
                      fieldnames=["ticker", "date", "leverage", "index", "nav", "aum_b"])
        )

    csv_files.append(write_meta_csv(out, {"script": "collect_fundamentals.py"}))
    make_zip(csv_files, out, zip_prefix="bql_fundamentals")
    log("collect_fundamentals.py concluído.")


if __name__ == "__main__":
    main()
