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
            "prices":                     self._normalize_prices,
            "price_history":              self._normalize_price_history,
            "fundamentals":               self._normalize_fundamentals,
            "fundamentals_history":       self._normalize_fundamentals_history,
            "bond_etf_fundamentals":      self._normalize_bond_etf_fundamentals,
            "bond_etf_history":           self._normalize_bond_etf_history,
            "fx_etf_fundamentals":        self._normalize_fx_etf_fundamentals,
            "commodity_etf_fundamentals": self._normalize_commodity_etf_fundamentals,
            "options_iv":                 self._normalize_options_iv,
            "iv_history":                 self._normalize_iv_history,
            "iv_term":                    self._normalize_iv_term,
            "skew_tails":                 self._normalize_skew_tails,
            "gex_summary":                self._normalize_gex_summary,
            "gex_spx":                    self._normalize_gex_spx,
            "greeks_per_ticker":          self._normalize_greeks_per_ticker,
            "chain":                      self._normalize_chain,
            "volume_flows":               self._normalize_volume_flows,
            "borrow_rate":                self._normalize_borrow_rate,
            "earnings_calendar":          self._normalize_earnings_calendar,
            "dividends":                  self._normalize_dividends,
            "eps_revisions":              self._normalize_eps_revisions,
            "realized_vol":               self._normalize_realized_vol,
            "thematic_flow":              self._normalize_thematic_flow,
            "positioning_models":         self._normalize_positioning_models,
            "index_members":              self._normalize_index_members,
            "etf_holdings":               self._normalize_etf_holdings,
            "letf_flows":                 self._normalize_letf_flows,
            "macro_series":               self._normalize_macro_series,
            "macro_history":              self._normalize_macro_history,
            "meta":                       self._normalize_meta,
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
        Usa yf_ticker como chave primária quando disponível — evita duplicação BBG/YF no banco.
        Só grava campos que existem no CSV — campos ausentes são silenciados.
        """
        ts, missing = [], []
        # Campos numéricos aceitos — só os que existirem no CSV serão gravados
        all_price_fields = [
            "price", "prev_price", "price_w", "price_ytd",
            "daily_return", "ytd_return", "weekly_return",
        ]

        cols = {c.lower(): c for c in df.columns}

        # Usa yf_ticker como ticker primário (evita duplicação com bbg_ticker no banco)
        # Fallback para bbg_ticker (normalizado) se yf_ticker não estiver presente
        ticker_col = cols.get("yf_ticker") or cols.get("bbg_ticker") or cols.get("ticker")

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
        fundamentals_*.csv (sector ETFs + mega-caps individuais)
        Colunas: ticker, pe, pb, ps, mktcap_b, beta, profit_margin, roe,
                 dividend_yield, debt_equity, expense_ratio, aum_b, price,
                 hi_52w, lo_52w, drawdown_52w, sector
        """
        fundamental_fields = [
            "pe", "pb", "ps", "mktcap_b", "beta", "profit_margin", "roe",
            "dividend_yield", "debt_equity", "expense_ratio", "aum_b",
            "price", "hi_52w", "lo_52w", "drawdown_52w",
        ]
        return self._normalize_wide(df, source_file, default_date, ingest_ts,
                                    ticker_col_candidates=["ticker", "bbg_ticker"],
                                    fields=fundamental_fields)

    def _normalize_fundamentals_history(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        fundamentals_history_*.csv (252d de pe/pb/ps/dy/beta por ticker)
        Colunas: date, ticker, sector, pe, pb, ps, dividend_yield, beta
        Cada linha vira uma entrada timeseries por field.
        """
        ts, missing = [], []
        cols = {c.lower(): c for c in df.columns}
        date_col   = cols.get("date")
        ticker_col = cols.get("ticker") or cols.get("yf_ticker")
        fields = ["pe", "pb", "ps", "dividend_yield", "beta"]

        for idx, row in df.iterrows():
            row_num = int(idx) + 2
            ticker  = _to_str(row.get(ticker_col)) if ticker_col else None
            date    = _to_str(row.get(date_col))   if date_col   else default_date
            if not ticker or not date:
                missing.append(self._missing(source_file, row_num, ticker, "fund_hist", date,
                                             "missing_key", "Linha sem ticker/date"))
                continue
            for f in fields:
                col = cols.get(f)
                if not col:
                    continue
                val = _to_float(row.get(col))
                ts.append(self._record(ticker, f"fund_hist_{f}", date, val, source_file, ingest_ts))
                if val is None:
                    missing.append(self._missing(source_file, row_num, ticker, f, date,
                                                 "missing_value", f"{f} ausente"))
        return ts, missing

    def _normalize_bond_etf_fundamentals(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        bond_etf_fundamentals_*.csv
        Colunas: ticker, price, expense_ratio, aum_b, yield, ytd_return,
                 daily_return, label, category, duration
        Duration vem do mapa estático no script (RATES_DURATION).
        """
        return self._normalize_wide(
            df, source_file, default_date, ingest_ts,
            ticker_col_candidates=["ticker", "bbg_ticker"],
            fields=["price", "expense_ratio", "aum_b", "yield",
                    "ytd_return", "daily_return", "duration"],
        )

    def _normalize_bond_etf_history(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        bond_etf_history_*.csv (252d)
        Colunas: date, ticker, label, category, duration, price, yield
        """
        ts, missing = [], []
        cols = {c.lower(): c for c in df.columns}
        date_col   = cols.get("date")
        ticker_col = cols.get("ticker")
        for idx, row in df.iterrows():
            row_num = int(idx) + 2
            ticker  = _to_str(row.get(ticker_col)) if ticker_col else None
            date    = _to_str(row.get(date_col))   if date_col   else default_date
            if not ticker or not date:
                continue
            for f in ("price", "yield"):
                col = cols.get(f)
                if not col:
                    continue
                val = _to_float(row.get(col))
                ts.append(self._record(ticker, f"bond_{f}", date, val, source_file, ingest_ts))
                if val is None:
                    missing.append(self._missing(source_file, row_num, ticker, f, date,
                                                 "missing_value", f"{f} ausente"))
        return ts, missing

    def _normalize_fx_etf_fundamentals(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        fx_etf_fundamentals_*.csv
        Colunas: ticker, price, expense_ratio, aum_b, ytd_return, label, category
        """
        return self._normalize_wide(
            df, source_file, default_date, ingest_ts,
            ticker_col_candidates=["ticker", "bbg_ticker"],
            fields=["price", "expense_ratio", "aum_b", "ytd_return"],
        )

    def _normalize_commodity_etf_fundamentals(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        commodity_etf_fundamentals_*.csv
        Colunas: ticker, price, expense_ratio, aum_b, ytd_return, daily_return, label, category
        """
        return self._normalize_wide(
            df, source_file, default_date, ingest_ts,
            ticker_col_candidates=["ticker", "bbg_ticker"],
            fields=["price", "expense_ratio", "aum_b", "ytd_return", "daily_return"],
        )

    # ── COBERTURA COMPLETA — normalizers das novas exports ─────────────────

    def _normalize_iv_term(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """iv_term_*.csv: ticker, iv_30d, iv_60d, iv_90d, iv_180d, iv_360d, contango_*"""
        return self._normalize_wide(
            df, source_file, default_date, ingest_ts,
            ticker_col_candidates=["ticker"],
            fields=["iv_30d", "iv_60d", "iv_90d", "iv_180d", "iv_360d",
                    "contango_60_30", "contango_90_30", "contango_180_30"],
        )

    def _normalize_skew_tails(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        skew_tails_*.csv: 25 fields × 3 tenors (30D / 90D / 180D) por ticker.
        Ex: atm_30D, put25_90D, call_skew_180D, rr_25d_30D, tail_premium_180D, etc.
        """
        TENORS = ["30D", "90D", "180D"]
        BASE = ["atm", "put25", "call25", "put10", "call10",
                "call_skew", "put_skew", "skew_25d", "skew_10d", "rr_25d", "tail_premium"]
        fields = [f"{b}_{t}" for b in BASE for t in TENORS]
        return self._normalize_wide(
            df, source_file, default_date, ingest_ts,
            ticker_col_candidates=["ticker"],
            fields=fields,
        )

    def _normalize_greeks_per_ticker(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """greeks_per_ticker_*.csv: GEX/walls/flip por mega-cap."""
        return self._normalize_wide(
            df, source_file, default_date, ingest_ts,
            ticker_col_candidates=["ticker"],
            fields=["spot", "gex_total_bn", "gex_call_bn", "gex_put_bn",
                    "call_wall", "put_wall", "gamma_flip", "pc_oi",
                    "total_call_oi", "total_put_oi", "total_call_vol", "total_put_vol",
                    "n_contracts"],
        )

    def _normalize_chain(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        chain_TICKER_*.csv: chain raw por strike (não vai pra timeseries do banco
        — fica disponível pra consumidores diretos via leitura do CSV).
        """
        return [], []

    def _normalize_volume_flows(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """volume_flows_*.csv: volume + dollar volume + short interest + ETF flows."""
        return self._normalize_wide(
            df, source_file, default_date, ingest_ts,
            ticker_col_candidates=["ticker"],
            fields=["volume", "volume_avg_30d", "vol_ratio", "price",
                    "dollar_volume", "short_int_ratio", "short_int_pct",
                    "days_to_cover", "fund_flow_1d"],
        )

    def _normalize_borrow_rate(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """borrow_rate_*.csv: borrow rate + sec lending availability."""
        return self._normalize_wide(
            df, source_file, default_date, ingest_ts,
            ticker_col_candidates=["ticker"],
            fields=["borrow_rate", "sl_available"],
        )

    def _normalize_earnings_calendar(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """earnings_calendar_*.csv: ticker, next_earn_date, eps_estimate, rev_estimate, eps_growth_est."""
        return self._normalize_wide(
            df, source_file, default_date, ingest_ts,
            ticker_col_candidates=["ticker"],
            fields=["eps_estimate", "rev_estimate", "eps_growth_est"],
        )

    def _normalize_dividends(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """dividends_*.csv: ticker, next_div_date, ex_div_date, div_amount, div_yield."""
        return self._normalize_wide(
            df, source_file, default_date, ingest_ts,
            ticker_col_candidates=["ticker"],
            fields=["div_amount", "div_yield"],
        )

    def _normalize_eps_revisions(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """eps_revisions_*.csv: ticker, eps_est, eps_3m_ago, eps_rev_3m, eps_up_30d, eps_down_30d."""
        return self._normalize_wide(
            df, source_file, default_date, ingest_ts,
            ticker_col_candidates=["ticker"],
            fields=["eps_est", "eps_3m_ago", "eps_rev_3m", "eps_up_30d", "eps_down_30d"],
        )

    def _normalize_realized_vol(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """realized_vol_*.csv: ticker, rv_30d, rv_60d, rv_90d, rv_252d."""
        return self._normalize_wide(
            df, source_file, default_date, ingest_ts,
            ticker_col_candidates=["ticker"],
            fields=["rv_30d", "rv_60d", "rv_90d", "rv_252d"],
        )

    def _normalize_thematic_flow(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        thematic_flow_*.csv: ticker, theme, price, ret_1d, ret_5d, ret_21d, ret_63d, ret_ytd
        Substitui DeepVue scraping — performance por ETF temático.
        """
        return self._normalize_wide(
            df, source_file, default_date, ingest_ts,
            ticker_col_candidates=["ticker"],
            fields=["price", "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_ytd"],
        )

    def _normalize_positioning_models(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        positioning_models_*.csv: CTA + Vol Control + Risk Parity por ticker.
        ~17 fields cobrindo 3 modelos de positioning estilo BofA/GS.
        """
        return self._normalize_wide(
            df, source_file, default_date, ingest_ts,
            ticker_col_candidates=["ticker"],
            fields=[
                "price", "rv_5d", "rv_30d", "rv_60d",
                "cta_sig_20d", "cta_sig_60d", "cta_sig_120d",
                "cta_score", "cta_leverage", "cta_notional_b",
                "volctrl_score", "volctrl_leverage", "volctrl_notional_b",
                "rp_score", "rp_weight", "rp_notional_b",
                "flow_total_b",
            ],
        )

    def _normalize_index_members(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        index_members_*.csv: index, member, weight, mcap_b, price, chg_1d.
        Cada member vira ts records identificados como ticker do membro.
        """
        ts, missing = [], []
        cols = {c.lower(): c for c in df.columns}
        member_col = cols.get("member")
        index_col  = cols.get("index")
        if not member_col:
            return ts, missing
        for idx, row in df.iterrows():
            row_num = int(idx) + 2
            member  = _to_str(row.get(member_col))
            index_n = _to_str(row.get(index_col)) if index_col else ""
            if not member:
                continue
            for f in ("weight", "mcap_b", "price", "chg_1d"):
                col = cols.get(f)
                if not col:
                    continue
                val = _to_float(row.get(col))
                ts.append(self._record(member, f"index_{index_n}_{f}",
                                       default_date, val, source_file, ingest_ts))
                if val is None:
                    missing.append(self._missing(source_file, row_num, member, f, default_date,
                                                 "missing_value", f"{f} ausente"))
        return ts, missing

    def _normalize_etf_holdings(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        etf_holdings_*.csv: etf, holding, value (string).
        Estrutura/lista — não vira timeseries, fica para consumo direto.
        """
        return [], []

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

    def _normalize_iv_history(
        self, df: "pd.DataFrame", source_file: str, default_date: str, ingest_ts: str
    ) -> tuple[list[dict], list[dict]]:
        """
        iv_history_*.csv
        Colunas: date, yf_ticker, iv  (IV ATM 30d em decimal, ex: 0.28 = 28%)
        Armazena série histórica de IV para calcular iv_percentile downstream.
        """
        ts, missing = [], []
        cols = {c.lower(): c for c in df.columns}

        date_col   = cols.get("date")
        ticker_col = cols.get("yf_ticker") or cols.get("ticker")
        iv_col     = cols.get("iv") or cols.get("atm_iv") or cols.get("ivol_mid_atm")

        for idx, row in df.iterrows():
            row_num = int(idx) + 2
            ticker  = _to_str(row.get(ticker_col)) if ticker_col else None
            date    = _to_str(row.get(date_col))   if date_col   else default_date
            iv      = _to_float(row.get(iv_col))   if iv_col     else None

            if not ticker:
                missing.append(self._missing(source_file, row_num, None, "iv", date,
                                             "missing_ticker", f"Linha {row_num} sem ticker"))
                continue
            if not date:
                missing.append(self._missing(source_file, row_num, ticker, "iv", None,
                                             "missing_date", f"Linha {row_num} sem data"))
                continue

            ts.append(self._record(ticker, "iv_history", date, iv, source_file, ingest_ts))
            if iv is None:
                missing.append(self._missing(source_file, row_num, ticker, "iv", date,
                                             "missing_value", f"Linha {row_num} IV ausente"))

        return ts, missing

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
        numeric_fields = ["leverage", "nav", "nav_prev", "aum_b"]

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
                col = cols.get(field.lower())
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
