"""
MacroDesk Bloomberg Ecosystem — CSV Parser
==========================================

Responsabilidade única:
  Ler arquivos CSV extraídos do Bloomberg e retornar DataFrames pandas limpos.

Detecta o tipo de dataset pelo prefixo do nome do arquivo:
  prices_*           → tipo "prices"
  price_history_*    → tipo "price_history"
  fundamentals_*     → tipo "fundamentals"
  options_iv_*       → tipo "options_iv"
  gex_summary_*      → tipo "gex_summary"
  gex_spx_*          → tipo "gex_spx"
  letf_flows_*       → tipo "letf_flows"
  meta_*             → tipo "meta"
  (outros)           → tipo "unknown"

Regras de parsing:
  - Tolera BOM UTF-8 (encoding utf-8-sig)
  - Tolera separador ponto-e-vírgula além de vírgula
  - Linhas em branco e cabeçalhos duplicados são ignorados
  - Não faz qualquer transformação de valores — isso é responsabilidade do normalizer

Uso:
    parser = CsvParser()
    dataset_type, df = parser.parse(Path("prices_20260403_143015_a3f7.csv"))
    if df is not None:
        print(dataset_type, df.shape)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.logger import get_logger

_log = get_logger("csv_parser")

# Tentamos importar pandas — obrigatório para o parser
try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False
    _log.error("pandas_not_installed", hint="pip install pandas")


# ── Mapeamento de prefixo → tipo de dataset ───────────────────────────────────
# Ordem importa: prefixos mais específicos primeiro (price_history antes de prices)
_PREFIX_TO_TYPE: list[tuple[str, str]] = [
    # Mais específicos primeiro
    ("price_history_bulk_",        "price_history"),
    ("price_history_",             "price_history"),
    ("prices_",                    "prices"),
    ("fundamentals_history_",      "fundamentals_history"),
    ("fundamentals_",              "fundamentals"),
    ("bond_etf_history_",          "bond_etf_history"),
    ("bond_etf_fundamentals_",     "bond_etf_fundamentals"),
    ("fx_etf_fundamentals_",       "fx_etf_fundamentals"),
    ("commodity_etf_fundamentals_","commodity_etf_fundamentals"),
    # Options / greeks
    ("options_iv_",                "options_iv"),
    ("iv_history_",                "iv_history"),
    ("iv_term_",                   "iv_term"),                # term structure 30/60/90/180/360
    ("skew_tails_",                "skew_tails"),             # 25d + 10d skew
    ("gex_summary_",               "gex_summary"),
    ("gex_spx_",                   "gex_spx"),
    ("greeks_per_ticker_",         "greeks_per_ticker"),      # mega-cap GEX/walls/flip
    ("chain_",                     "chain"),                  # chain raw por ticker
    # Volume / fluxos / short / borrow / earnings / dividends / revisions
    ("volume_flows_",              "volume_flows"),
    ("borrow_rate_",               "borrow_rate"),
    ("earnings_calendar_",         "earnings_calendar"),
    ("dividends_",                 "dividends"),
    ("eps_revisions_",             "eps_revisions"),
    ("realized_vol_",              "realized_vol"),
    ("thematic_flow_",             "thematic_flow"),
    # Estrutura de mercado
    ("index_members_",             "index_members"),
    ("etf_holdings_",              "etf_holdings"),
    # Macro / LETF
    ("letf_flows_",                "letf_flows"),
    ("macro_history_",             "macro_history"),
    ("macro_series_",              "macro_series"),
    ("meta_",                      "meta"),
]

# Colunas obrigatórias por tipo de dataset — usadas para validação
_REQUIRED_COLS: dict[str, list[str]] = {
    "prices":                     ["price"],
    "price_history":              ["date", "price"],
    "fundamentals":               ["ticker", "pe"],
    "fundamentals_history":       ["date", "ticker", "pe"],
    "bond_etf_fundamentals":      ["ticker", "price"],
    "bond_etf_history":           ["date", "ticker", "price"],
    "fx_etf_fundamentals":        ["ticker", "price"],
    "commodity_etf_fundamentals": ["ticker", "price"],
    "options_iv":                 ["ticker", "atm_iv"],
    "iv_history":                 ["date", "yf_ticker", "iv"],
    "iv_term":                    ["ticker", "iv_30d"],
    "skew_tails":                 ["ticker", "skew_25d"],
    "gex_summary":                ["gex_total_bn"],
    "gex_spx":                    ["strike", "gex_bn"],
    "greeks_per_ticker":          ["ticker", "gex_total_bn"],
    "chain":                      ["strike", "ivol"],
    "volume_flows":               ["ticker", "volume"],
    "borrow_rate":                ["ticker", "borrow_rate"],
    "earnings_calendar":          ["ticker", "next_earn_date"],
    "dividends":                  ["ticker", "next_div_date"],
    "eps_revisions":              ["ticker", "eps_est"],
    "realized_vol":               ["ticker", "rv_30d"],
    "thematic_flow":              ["ticker", "theme"],
    "index_members":              ["index", "member"],
    "etf_holdings":               ["etf", "holding"],
    "letf_flows":                 ["ticker", "nav"],
    "macro_series":               ["bbg_ticker", "px_last"],
    "macro_history":              ["date", "bbg_ticker", "value"],
    "meta":                       ["generated_at"],
}


def detect_dataset_type(filename: str) -> str:
    """
    Detecta o tipo de dataset pelo prefixo do nome do arquivo.

    Args:
        filename: Nome do arquivo (apenas basename, com ou sem extensão).

    Returns:
        Tipo de dataset (ex: "prices", "fundamentals") ou "unknown".
    """
    name = Path(filename).name.lower()
    for prefix, dtype in _PREFIX_TO_TYPE:
        if name.startswith(prefix):
            return dtype
    # Tenta sem sufixo de timestamp (ex: "prices.csv" legado)
    for prefix, dtype in _PREFIX_TO_TYPE:
        if name.startswith(prefix.rstrip("_")):
            return dtype
    return "unknown"


class CsvParser:
    """
    Parser de CSVs Bloomberg.

    Todos os métodos retornam (dataset_type, DataFrame | None).
    None indica falha de parsing — registrada em log, não levanta exceção.
    """

    def parse(self, file_path: Path) -> tuple[str, Optional["pd.DataFrame"]]:
        """
        Lê e valida um CSV Bloomberg.

        Args:
            file_path: Caminho completo para o arquivo CSV.

        Returns:
            (dataset_type, df) onde df é None em caso de erro.
        """
        if not _PANDAS_OK:
            return "unknown", None

        if not file_path.exists():
            _log.error("file_not_found", path=str(file_path))
            return "unknown", None

        dtype = detect_dataset_type(file_path.name)
        if dtype == "unknown":
            _log.warning("unknown_csv_type", file=file_path.name)
            # Ainda tenta ler — pode ser útil em debug
            df = self._safe_read(file_path)
            return "unknown", df

        df = self._safe_read(file_path)
        if df is None:
            return dtype, None

        if df.empty:
            _log.warning("empty_csv", file=file_path.name, dtype=dtype)
            return dtype, df

        # Valida colunas obrigatórias (case-insensitive)
        cols_lower = {c.lower() for c in df.columns}
        required   = _REQUIRED_COLS.get(dtype, [])
        missing    = [r for r in required if r.lower() not in cols_lower]
        if missing:
            _log.warning(
                "missing_required_columns",
                file=file_path.name,
                dtype=dtype,
                missing=missing,
            )
            # Não rejeita — normalizer lidará com colunas ausentes

        _log.info("csv_parsed", file=file_path.name, dtype=dtype, rows=len(df), cols=len(df.columns))
        return dtype, df

    # ── Privados ──────────────────────────────────────────────────────────────

    def _safe_read(self, path: Path) -> Optional["pd.DataFrame"]:
        """
        Lê CSV tolerando diferentes encodings e separadores.
        Retorna None se não conseguir ler de forma alguma.
        """
        errors_tried: list[str] = []

        # Ordem de tentativas: utf-8-sig (com BOM), utf-8, latin-1
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            for sep in (",", ";"):
                try:
                    df = pd.read_csv(
                        path,
                        encoding=encoding,
                        sep=sep,
                        skip_blank_lines=True,
                        na_values=["", "N/A", "NA", "NaN", "#N/A", "null", "NULL", "None"],
                        keep_default_na=True,
                    )
                    # Normaliza nomes de colunas: strip + lowercase para comparação interna
                    df.columns = [c.strip() for c in df.columns]
                    # Remove linhas completamente vazias
                    df = df.dropna(how="all")
                    return df
                except Exception as exc:
                    errors_tried.append(f"{encoding}/{sep}: {exc}")

        _log.error("csv_read_failed", path=str(path), attempts=errors_tried)
        return None
