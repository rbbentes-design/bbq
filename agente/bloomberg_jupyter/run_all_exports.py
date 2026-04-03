"""
Bloomberg BQuant — Orquestrador Completo de Exportação
========================================================

Cole este código em uma célula Jupyter do BQuant e execute.
OU rode: bqnt_python bloomberg_jupyter/run_all_exports.py

Este script executa TODOS os coletores em sequência e empacota
tudo em um único ZIP final chamado bql_data_{date}.zip.

Fluxo:
  1. collect_prices.py        → prices, price_history
  2. collect_fundamentals.py  → fundamentals, options_iv, gex_summary, letf_flows
  3. collect_macro_series.py  → macro_series, macro_history
  4. meta.csv
  5. bql_data_{date}.zip      → ZIP com TUDO (este é o que o agente local detecta)

Modo loop:
  Execute com LOOP_MODE = True para rodar continuamente a cada INTERVAL_SEC segundos.
  Útil para deixar rodando durante o pregão.

Configuração:
  LOOP_MODE    = False   → roda uma vez e sai
  LOOP_MODE    = True    → roda em loop até interromper (Kernel → Interrupt)
  INTERVAL_SEC = 300     → 5 minutos entre cada rodada
"""

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from export_utils import ensure_output_dir, log, make_zip, write_meta_csv

# ── Configuração ─────────────────────────────────────────────────────────────
LOOP_MODE    = False   # True para loop contínuo durante o pregão
INTERVAL_SEC = 300     # 5 minutos entre exportações

# O que coletar (desative comentando a linha)
RUN_PRICES       = True
RUN_FUNDAMENTALS = True
RUN_MACRO        = True


def run_once() -> list[Path]:
    """Executa todos os coletores e retorna lista de CSVs gerados."""
    out = ensure_output_dir()
    log("=" * 60)
    log(f"Iniciando exportação completa Bloomberg — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    csv_files: list[Path] = []
    errors: list[str] = []

    # ── 1. Preços ─────────────────────────────────────────────────────────
    if RUN_PRICES:
        log("\n[1/3] PREÇOS E HISTÓRICO")
        try:
            from collect_prices import collect_prices, collect_price_history
            from export_utils import write_csv

            prices_rows = collect_prices()
            if prices_rows:
                csv_files.append(
                    write_csv(out, "prices", prices_rows, fieldnames=[
                        "bbg_ticker", "yf_ticker", "name", "date", "price", "prev_price",
                        "price_w", "price_ytd", "daily_return", "weekly_return", "ytd_return",
                        "vol_20d", "hi_52w", "lo_52w", "drawdown_52w",
                    ])
                )

            history_rows = collect_price_history(days=252)
            if history_rows:
                csv_files.append(
                    write_csv(out, "price_history", history_rows,
                              fieldnames=["date", "bbg_ticker", "yf_ticker", "name", "price"])
                )
        except Exception as exc:
            errors.append(f"Preços: {exc}")
            log(f"  ERRO em preços: {exc}")

    # ── 2. Fundamentos ────────────────────────────────────────────────────
    if RUN_FUNDAMENTALS:
        log("\n[2/3] FUNDAMENTOS + OPÇÕES + GEX + LETF")
        try:
            from collect_fundamentals import (
                collect_fundamentals,
                collect_gex_spx,
                collect_letf_flows,
                collect_options_iv,
            )
            from export_utils import write_csv

            fund_rows = collect_fundamentals()
            if fund_rows:
                csv_files.append(
                    write_csv(out, "fundamentals", fund_rows, fieldnames=[
                        "ticker", "date", "pe", "pb", "ev_ebitda", "mktcap_b", "beta",
                        "roe", "profit_margin", "debt_equity", "dividend_yield",
                        "price", "hi_52w", "lo_52w", "drawdown_52w",
                    ])
                )

            iv_rows = collect_options_iv()
            if iv_rows:
                csv_files.append(
                    write_csv(out, "options_iv", iv_rows,
                              fieldnames=["ticker", "date", "atm_iv", "skew_25d", "pcr_oi"])
                )

            gex_summary, gex_strikes = collect_gex_spx()
            if gex_summary:
                csv_files.append(
                    write_csv(out, "gex_summary", gex_summary, fieldnames=[
                        "date", "spot", "gex_total_bn", "gex_call_bn", "gex_put_bn",
                        "direction", "gamma_regime", "n_options", "note",
                    ])
                )

            letf_rows = collect_letf_flows()
            if letf_rows:
                csv_files.append(
                    write_csv(out, "letf_flows", letf_rows,
                              fieldnames=["ticker", "date", "leverage", "index", "nav", "aum_b"])
                )
        except Exception as exc:
            errors.append(f"Fundamentos: {exc}")
            log(f"  ERRO em fundamentos: {exc}")

    # ── 3. Macro Series ───────────────────────────────────────────────────
    if RUN_MACRO:
        log("\n[3/3] SÉRIES MACROECONÔMICAS")
        try:
            from collect_macro_series import collect_macro_history, collect_macro_snapshot
            from export_utils import write_csv

            macro_rows = collect_macro_snapshot()
            if macro_rows:
                csv_files.append(
                    write_csv(out, "macro_series", macro_rows,
                              fieldnames=["bbg_ticker", "description", "category", "px_last", "date"])
                )

            history_rows = collect_macro_history(days=252)
            if history_rows:
                csv_files.append(
                    write_csv(out, "macro_history", history_rows,
                              fieldnames=["date", "bbg_ticker", "description", "category", "value"])
                )
        except Exception as exc:
            errors.append(f"Macro: {exc}")
            log(f"  ERRO em macro: {exc}")

    # ── Meta + ZIP ────────────────────────────────────────────────────────
    meta_extra = {
        "script":   "run_all_exports.py",
        "csvs":     str(len(csv_files)),
        "errors":   str(len(errors)),
    }
    if errors:
        meta_extra["error_list"] = "; ".join(errors)

    csv_files.append(write_meta_csv(out, meta_extra))

    # ZIP FINAL — este é o arquivo que o agente local detecta em Downloads
    log("\nEmpacotando ZIP final...")
    zip_path = make_zip(csv_files, out, zip_prefix="bql_data")

    log("\n" + "=" * 60)
    log(f"EXPORTAÇÃO CONCLUÍDA")
    log(f"  CSVs gerados:  {len(csv_files)}")
    log(f"  Erros:         {len(errors)}")
    log(f"  ZIP:           {zip_path.name}")
    log("=" * 60)

    if errors:
        log("\nERROS encontrados:")
        for e in errors:
            log(f"  - {e}")

    return csv_files


def main() -> None:
    if LOOP_MODE:
        log(f"Modo loop ativo — exportação a cada {INTERVAL_SEC}s. Kernel → Interrupt para parar.")
        run_count = 0
        while True:
            run_count += 1
            log(f"\n{'='*60}")
            log(f"RODADA #{run_count}")
            run_once()
            log(f"Aguardando {INTERVAL_SEC}s até a próxima exportação...")
            time.sleep(INTERVAL_SEC)
    else:
        run_once()


if __name__ == "__main__":
    main()
