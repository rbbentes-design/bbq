"""
MacroDesk Bloomberg Ecosystem — Data Normalizer
================================================

Responsabilidade única:
  Converter DataFrames de diferentes formatos Bloomberg para o formato
  canônico longo (ticker, field, date, value) pronto para gravação no banco.

Regras obrigatórias:
  - NaN pandas  → None Python → NULL SQLite (nunca string "NaN" ou "null")
  - Não preencher valores ausentes com valor anterior (sem forward-fill)
  - Não preencher com zero
  - Registrar toda ausência esperada em missing_records

Formato de saída (timeseries_records):
  {
    "ticker":              str,   # ticker Bloomberg ou alias
    "field":               str,   # nome do campo normalizado
    "date":                str,   # YYYY-MM-DD
    "value":               float | None,
    "frequency":           str,   # "D" padrão
    "source_file":         str,
    "ingestion_timestamp": str,   # ISO-8601
  }

Formato de saída (missing_records):
  {
    "source_file":      str,
    "row_number":       int | None,
    "ticker":           str | None,
    "field":            str | None,
    "reference_date":   str | None,
    "issue_type":       str,
    "issue_description": str,
    "detected_at":      str,
  }

Uso:
    norm = DataNormalizer()
    ts_records, missing = norm.normalize("fundamentals", df, "fundamentals_20260403_143015_a3f7.csv")
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Optional

from core.logger import get_logger

_log = get_logger("data_normalizer")

try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False


def _is_null(v: Any) -> bool:
    """True se o valor deve ser tratado como NULL (None, NaN, inf)."""
    if v is None:
        return True
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return True
    if isinstance(v, str) and v.strip().lower() in ("", "nan", "na", "n/a", "#n/a", "null", "none"):
        return True
    return False


def _to_float(v: Any) -> Optional[float]:
    """Converte para float ou retorna None se inválido."""
    if _is_null(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_str(v: Any) -> Optional[str]:
    """Converte para string ou None se nulo."""
    if _is_null(v):
        return None
    return str(v).strip()


def _extract_date_from_filename(filename: str) -> Optional[str]:
    """
    Extrai data YYYY-MM-DD do nome de arquivo como 'prices_20260403_*.csv'.
    Retorna None se não encontrar.
    """
    import re
    m = re.search(r"(\d{4})(\d{2})(\d{2})", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DataNormalizer:
    """
    Normaliza DataFrames Bloomberg para o formato canônico longo.
    """

    def normalize(
        self,
        dataset_type: str,
        df: "pd.DataFrame",
        source_file: str,
    ) -> tuple[list[dict], list[dict]]:
        """
        Converte DataFrame para registros de timeseries e log de ausências.

        Args:
            dataset_type: Tipo detectado pelo CsvParser (ex: "prices").
            df:           DataFrame lido pelo CsvParser (nomes de colunas já stripped).
            source_file:  Nome do arquivo renomeado (ex: prices_20260403_143015_a3f7.csv).

        Returns:
            (timeseries_records, missing_records)
        """
        if not _PANDAS_OK or df is None or df.empty:
            return [], []

        ingest_ts = _now_iso()
        # Data de referência padrão = data extraída do nome do arquivo
        default_date = _extract_date_from_filename(source_file) or ingest_ts[:10]

        dispatch = {
            "prices":        self._normalize_prices,
            "price_history": self._normalize_price_history,
            "fundamentals":  self._normalize_fundamentals,
            "options_iv":    self._normalize_options_iv,
            "gex_summary":   self._normalize_gex_summary,
            "gex_spx":       self._normalize_gex_spx,
            "letf_flows":    self._normalize_letf_flows,
            "macro_series":  self._normalize_macro_series,
            "macro_history": self._normalize_macro_history,
            "meta":          self._normalize_meta,
        }

        handler = dispatch.get(dataset_type)
        if handler is None:
            _log.warning("no_normalizer_for_type", dtype=dataset_type, file=source_file)
            return [], []

        ts_records, missing = handler(df, source_file, default_date, ingest_ts)
        _log.info(
            "normalization_done",
            dtype=dataset_type,
            file=source_file,
            ts_records=len(ts_records),
            missing=len(missing),
        )
        return ts_records, missing

    # ── Normalizadores por tipo ────────────────────────────────────────────────

    def _normalize_prices(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        prices_*.csv
        Aceita tanto o formato do bql_export.py (daily_return, ytd_return já calculados)
        quanto o formato legado (prev_price, price_w, price_ytd como preços brutos).
        Só grava campos que existem no CSV — campos ausentes são silenciados.
        """
        ts, missing = [], []
        # Campos numéricos aceitos — só os que existirem no CSV serão gravados
        all_price_fields = [
            "price", "prev_price", "price_w", "price_ytd",
            "daily_return", "ytd_return",
        ]

        cols = {c.lower(): c for c in df.columns}

        # Detecta coluna de ticker (bbg_ticker preferido, fallback yf_ticker)
        ticker_col = cols.get("bbg_ticker") or cols.get("yf_ticker") or cols.get("ticker")

        # Determina quais campos realmente existem neste CSV
        present_fields = [f for f in all_price_fields if f in cols]

        for idx, row in df.iterrows():
            row_num = int(idx) + 2  # +2 = header + 1-based

            ticker = _to_str(row.get(ticker_col)) if ticker_col else None
            if not ticker:
                missing.append(self._missing(source_file, row_num, None, None, default_date,
                                             "missing_ticker", f"Linha {row_num} sem ticker identificado"))
                continue

            for field in present_fields:
                col = cols[field]
                raw = row.get(col)
                val = _to_float(raw)

                if val is None and not _is_null(raw):
                    missing.append(self._missing(source_file, row_num, ticker, field, default_date,
                                                 "invalid_numeric", f"Linha {row_num} valor inválido para {field}: {raw!r}"))

                ts.append(self._record(ticker, field, default_date, val, source_file, ingest_ts))

        return ts, missing

    def _normalize_price_history(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        price_history_*.csv
        Colunas: date, yf_ticker, price
        Formato longo — uma linha por (ticker, date).
        """
        ts, missing = [], []
        cols = {c.lower(): c for c in df.columns}

        date_col   = cols.get("date")
        ticker_col = cols.get("yf_ticker") or cols.get("ticker") or cols.get("bbg_ticker")
        price_col  = cols.get("price") or cols.get("close")

        for idx, row in df.iterrows():
            row_num = int(idx) + 2

            ticker = _to_str(row.get(ticker_col)) if ticker_col else None
            date   = _to_str(row.get(date_col))   if date_col   else default_date
            price  = _to_float(row.get(price_col)) if price_col  else None

            if not ticker:
                missing.append(self._missing(source_file, row_num, None, "price", date,
                                             "missing_ticker", f"Linha {row_num} sem ticker"))
                continue
            if not date:
                missing.append(self._missing(source_file, row_num, ticker, "price", None,
                                             "missing_date", f"Linha {row_num} sem data"))
                continue

            ts.append(self._record(ticker, "price", date, price, source_file, ingest_ts))
            if price is None:
                missing.append(self._missing(source_file, row_num, ticker, "price", date,
                                             "missing_value", f"Linha {row_num} preço ausente"))

        return ts, missing

    def _normalize_fundamentals(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        fundamentals_*.csv
        Colunas: ticker, pe, mktcap_b, beta, profit_margin, debt_equity, roe,
                 dividend_yield, price, hi_52w, lo_52w, drawdown_52w
        """
        fundamental_fields = [
            "pe", "mktcap_b", "beta", "profit_margin", "debt_equity",
            "roe", "dividend_yield", "price", "hi_52w", "lo_52w", "drawdown_52w",
        ]
        return self._normalize_wide(df, source_file, default_date, ingest_ts,
                                    ticker_col_candidates=["ticker", "bbg_ticker"],
                                    fields=fundamental_fields)

    def _normalize_options_iv(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        options_iv_*.csv
        Colunas: ticker, atm_iv, skew_25d, pcr_oi
        """
        return self._normalize_wide(df, source_file, default_date, ingest_ts,
                                    ticker_col_candidates=["ticker", "bbg_ticker"],
                                    fields=["atm_iv", "skew_25d", "pcr_oi"])

    def _normalize_gex_summary(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        gex_summary_*.csv
        Colunas: date, spot, gex_total_bn, gex_call_bn, gex_put_bn, direction, gamma_regime, n_options
        Ticker virtual: SPX_GEX
        direction e gamma_regime são strings — gravamos como NULL em valor numérico mas
        registramos em missing_data_log com issue_type "string_field_skipped".
        """
        ts, missing = [], []
        cols    = {c.lower(): c for c in df.columns}
        numeric_fields = ["spot", "gex_total_bn", "gex_call_bn", "gex_put_bn", "n_options"]
        string_fields  = ["direction", "gamma_regime"]
        date_col       = cols.get("date")
        TICKER         = "SPX_GEX"

        for idx, row in df.iterrows():
            row_num = int(idx) + 2
            date    = _to_str(row.get(date_col)) if date_col else default_date
            date    = date or default_date

            for field in numeric_fields:
                col = cols.get(field)
                raw = row.get(col) if col else None
                val = _to_float(raw) if col else None
                ts.append(self._record(TICKER, field, date, val, source_file, ingest_ts))
                if val is None and col is not None:
                    missing.append(self._missing(source_file, row_num, TICKER, field, date,
                                                 "missing_value", f"Linha {row_num} {field} ausente"))

            # Campos string: gravar como texto em campo separado não existe no schema,
            # então codificamos como float (None) e registramos a string no missing_log
            # para rastreabilidade operacional.
            for field in string_fields:
                col = cols.get(field)
                raw = _to_str(row.get(col)) if col else None
                if raw:
                    missing.append(self._missing(source_file, row_num, TICKER, field, date,
                                                 "string_field_skipped",
                                                 f"Linha {row_num} {field}={raw!r} (string, não gravado em timeseries)"))

        return ts, missing

    def _normalize_gex_spx(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        gex_spx_*.csv
        Colunas: strike, put_call, open_int, gamma, gex_bn
        Ticker virtual: SPX_GEX_STRIKE_{strike}_{put_call}
        Preserva a estrutura por strike no banco.
        """
        ts, missing = [], []
        cols = {c.lower(): c for c in df.columns}

        strike_col  = cols.get("strike")
        pc_col      = cols.get("put_call")
        oi_col      = cols.get("open_int")
        gamma_col   = cols.get("gamma")
        gex_bn_col  = cols.get("gex_bn")

        for idx, row in df.iterrows():
            row_num = int(idx) + 2
            strike  = _to_str(row.get(strike_col))  if strike_col else None
            pc      = _to_str(row.get(pc_col))      if pc_col     else "x"

            if not strike:
                missing.append(self._missing(source_file, row_num, None, "gex_bn", default_date,
                                             "missing_value", f"Linha {row_num} sem strike"))
                continue

            ticker = f"SPX_GEX_{strike}_{pc}"

            for field, col in [("open_int", oi_col), ("gamma", gamma_col), ("gex_bn", gex_bn_col)]:
                val = _to_float(row.get(col)) if col else None
                ts.append(self._record(ticker, field, default_date, val, source_file, ingest_ts))

        return ts, missing

    def _normalize_letf_flows(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        letf_flows_*.csv
        Colunas: ticker, leverage, index, nav, aum_b
        """
        ts, missing = [], []
        cols = {c.lower(): c for c in df.columns}
        ticker_col = cols.get("ticker")
        numeric_fields = ["leverage", "nav", "aum_b"]

        for idx, row in df.iterrows():
            row_num = int(idx) + 2
            ticker  = _to_str(row.get(ticker_col)) if ticker_col else None

            if not ticker:
                missing.append(self._missing(source_file, row_num, None, None, default_date,
                                             "missing_ticker", f"Linha {row_num} sem ticker"))
                continue

            for field in numeric_fields:
                col = cols.get(field)
                val = _to_float(row.get(col)) if col else None
                ts.append(self._record(ticker, field, default_date, val, source_file, ingest_ts))

        return ts, missing

    def _normalize_macro_series(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        macro_series_*.csv
        Colunas: bbg_ticker, description, category, px_last, date
        Grava em bql_timeseries E em macro_series_latest via campo "_macro_snapshot_".
        """
        ts, missing = [], []
        cols = {c.lower(): c for c in df.columns}

        ticker_col = cols.get("bbg_ticker") or cols.get("ticker")
        value_col  = cols.get("px_last") or cols.get("value")
        date_col   = cols.get("date")
        desc_col   = cols.get("description")
        cat_col    = cols.get("category")

        for idx, row in df.iterrows():
            row_num = int(idx) + 2
            ticker  = _to_str(row.get(ticker_col)) if ticker_col else None
            date    = _to_str(row.get(date_col))   if date_col   else default_date
            val     = _to_float(row.get(value_col)) if value_col  else None
            desc    = _to_str(row.get(desc_col))    if desc_col   else ticker
            cat     = _to_str(row.get(cat_col))     if cat_col    else "other"

            if not ticker:
                missing.append(self._missing(source_file, row_num, None, "px_last", default_date,
                                             "missing_ticker", f"Linha {row_num} sem bbg_ticker"))
                continue

            # Grava em bql_timeseries (formato padrão)
            ts.append(self._record(ticker, "px_last", date or default_date, val,
                                   source_file, ingest_ts))

            # Grava metadados para reconstruir macro_series_latest
            # (campo especial com prefixo _meta_ para o db_writer reconhecer)
            ts.append(self._record(
                ticker, "_macro_meta_",
                date or default_date,
                None,          # valor não numérico — armazenado como NULL
                source_file, ingest_ts,
            ))

            if val is None:
                missing.append(self._missing(source_file, row_num, ticker, "px_last",
                                             date or default_date, "missing_value",
                                             f"Linha {row_num} px_last ausente para {ticker}"))

        return ts, missing

    def _normalize_macro_history(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        macro_history_*.csv
        Colunas: date, bbg_ticker, description, category, value
        Formato longo — histórico diário de séries macro.
        """
        ts, missing = [], []
        cols = {c.lower(): c for c in df.columns}

        date_col   = cols.get("date")
        ticker_col = cols.get("bbg_ticker") or cols.get("ticker")
        value_col  = cols.get("value") or cols.get("px_last")

        for idx, row in df.iterrows():
            row_num = int(idx) + 2
            ticker  = _to_str(row.get(ticker_col)) if ticker_col else None
            date    = _to_str(row.get(date_col))   if date_col   else default_date
            val     = _to_float(row.get(value_col)) if value_col  else None

            if not ticker:
                missing.append(self._missing(source_file, row_num, None, "value", date,
                                             "missing_ticker", f"Linha {row_num} sem bbg_ticker"))
                continue
            if not date:
                missing.append(self._missing(source_file, row_num, ticker, "value", None,
                                             "missing_date", f"Linha {row_num} sem data"))
                continue

            # Grava em bql_timeseries como série macro histórica
            ts.append(self._record(ticker, "macro_px_last", date, val, source_file, ingest_ts))

            if val is None:
                missing.append(self._missing(source_file, row_num, ticker, "macro_px_last", date,
                                             "missing_value", f"Linha {row_num} value ausente"))

        return ts, missing

    def _normalize_meta(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        meta_*.csv — timestamp do export Bloomberg.
        Aceita dois formatos:
          - bql_export.py: coluna única 'generated_at' com datetime ISO
          - legado: colunas 'key' e 'value'
        Gravamos como campo de sistema (ticker=SYSTEM) para rastreabilidade.
        """
        ts = []
        cols = {c.lower(): c for c in df.columns}

        # Formato bql_export.py: coluna 'generated_at'
        ga_col = cols.get("generated_at")
        if ga_col:
            ts.append(self._record("SYSTEM", "meta_generated_at", default_date, None,
                                   source_file, ingest_ts))
            return ts, []

        # Formato legado: colunas 'key' / 'value'
        key_col = cols.get("key")
        val_col = cols.get("value")
        if key_col and val_col:
            for _, row in df.iterrows():
                key = _to_str(row.get(key_col))
                if key:
                    ts.append(self._record("SYSTEM", f"meta_{key}", default_date, None,
                                           source_file, ingest_ts))

        return ts, []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _normalize_wide(
        self,
        df: "pd.DataFrame",
        source_file: str,
        default_date: str,
        ingest_ts: str,
        ticker_col_candidates: list[str],
        fields: list[str],
    ) -> tuple[list[dict], list[dict]]:
        """Helper genérico para CSVs no formato wide (uma linha por ticker)."""
        ts, missing = [], []
        cols = {c.lower(): c for c in df.columns}

        ticker_col = None
        for cand in ticker_col_candidates:
            if cand in cols:
                ticker_col = cols[cand]
                break

        for idx, row in df.iterrows():
            row_num = int(idx) + 2
            ticker  = _to_str(row.get(ticker_col)) if ticker_col else None

            if not ticker:
                missing.append(self._missing(source_file, row_num, None, None, default_date,
                                             "missing_ticker", f"Linha {row_num} sem ticker"))
                continue

            for field in fields:
                col = cols.get(field)
                raw = row.get(col) if col else None
                val = _to_float(raw) if col else None

                if val is None and col is not None and raw is not None and not _is_null(raw):
                    missing.append(self._missing(source_file, row_num, ticker, field, default_date,
                                                 "invalid_numeric",
                                                 f"Linha {row_num} valor inválido para {field}: {raw!r}"))
                elif val is None and col is not None:
                    missing.append(self._missing(source_file, row_num, ticker, field, default_date,
                                                 "missing_value", f"Linha {row_num} {field} ausente para {ticker}"))

                ts.append(self._record(ticker, field, default_date, val, source_file, ingest_ts))

        return ts, missing

    @staticmethod
    def _record(
        ticker: str, field: str, date: str, value: Optional[float],
        source_file: str, ingest_ts: str, freq: str = "D",
    ) -> dict:
        return {
            "ticker":              ticker,
            "field":               field,
            "date":                date,
            "value":               value,   # None → NULL no banco
            "frequency":           freq,
            "source_file":         source_file,
            "ingestion_timestamp": ingest_ts,
        }

    @staticmethod
    def _missing(
        source_file: str, row_number: Optional[int], ticker: Optional[str],
        field: Optional[str], reference_date: Optional[str],
        issue_type: str, issue_description: str,
    ) -> dict:
        return {
            "source_file":      source_file,
            "row_number":       row_number,
            "ticker":           ticker,
            "field":            field,
            "reference_date":   reference_date,
            "issue_type":       issue_type,
            "issue_description": issue_description,
            "detected_at":      _now_iso(),
        }
