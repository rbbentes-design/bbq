"""
MacroDesk Bloomberg Ecosystem — Schema do Banco SQLite
======================================================

Cria todas as tabelas obrigatórias no banco macrodesk.db.
Idempotente: seguro para chamar múltiplas vezes (CREATE TABLE IF NOT EXISTS).

Tabelas:
  1. processed_zip_registry  — controle de zips já processados
  2. extracted_file_registry — controle de CSVs extraídos e renomeados
  3. bql_timeseries          — histórico de séries (ticker, field, date, value)
  4. bql_latest              — último valor por ticker/field
  5. ingestion_log           — log por execução do agente
  6. metadata_map            — mapeamento de colunas originais → normalizadas
  7. missing_data_log        — auditoria de dados ausentes ou inválidos
  8. macro_series_latest     — último valor de séries macro por ticker (curva de juros, spreads, VIX term)
  9. macro_series_history    — histórico diário de séries macro
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


# ── DDL de cada tabela ────────────────────────────────────────────────────────

_CREATE_PROCESSED_ZIP_REGISTRY = """
CREATE TABLE IF NOT EXISTS processed_zip_registry (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    zip_name        TEXT    NOT NULL,
    zip_path        TEXT    NOT NULL,
    zip_hash        TEXT    NOT NULL,           -- sha256 do arquivo
    zip_size        INTEGER NOT NULL,
    detected_at     TEXT    NOT NULL,           -- ISO-8601
    processed_at    TEXT,                       -- preenchido no final do processamento
    status          TEXT    NOT NULL DEFAULT 'pending',  -- pending | ok | error
    error_message   TEXT,
    UNIQUE (zip_name, zip_hash)                -- evita reprocessar o mesmo zip
);
"""

_CREATE_EXTRACTED_FILE_REGISTRY = """
CREATE TABLE IF NOT EXISTS extracted_file_registry (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    original_csv_name   TEXT    NOT NULL,
    renamed_csv_name    TEXT    NOT NULL,       -- nome com timestamp
    source_zip_name     TEXT    NOT NULL,
    saved_path          TEXT    NOT NULL,
    extracted_at        TEXT    NOT NULL,       -- ISO-8601
    file_hash           TEXT,                  -- sha256 do CSV extraído
    status              TEXT    NOT NULL DEFAULT 'ok',  -- ok | error
    UNIQUE (renamed_csv_name)                  -- nome único garantido por timestamp
);
"""

_CREATE_BQL_TIMESERIES = """
CREATE TABLE IF NOT EXISTS bql_timeseries (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT    NOT NULL,
    field               TEXT    NOT NULL,
    date                TEXT    NOT NULL,       -- YYYY-MM-DD
    value               REAL,                  -- NULL = dado ausente (não zero)
    frequency           TEXT    NOT NULL DEFAULT 'D',  -- D | W | M
    source_file         TEXT    NOT NULL,
    ingestion_timestamp TEXT    NOT NULL,       -- ISO-8601
    UNIQUE (ticker, field, date)               -- deduplicação por chave lógica
);
CREATE INDEX IF NOT EXISTS idx_ts_ticker_field ON bql_timeseries (ticker, field);
CREATE INDEX IF NOT EXISTS idx_ts_date         ON bql_timeseries (date);
"""

_CREATE_BQL_LATEST = """
CREATE TABLE IF NOT EXISTS bql_latest (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT    NOT NULL,
    field               TEXT    NOT NULL,
    latest_date         TEXT    NOT NULL,       -- YYYY-MM-DD da observação mais recente
    latest_value        REAL,                  -- NULL = dado ausente
    source_file         TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL,       -- ISO-8601 da última atualização
    UNIQUE (ticker, field)                     -- um registro por (ticker, field)
);
CREATE INDEX IF NOT EXISTS idx_latest_ticker ON bql_latest (ticker);
"""

_CREATE_INGESTION_LOG = """
CREATE TABLE IF NOT EXISTS ingestion_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL UNIQUE,    -- UUID gerado por execução
    started_at      TEXT    NOT NULL,
    finished_at     TEXT,
    zips_found      INTEGER DEFAULT 0,
    zips_processed  INTEGER DEFAULT 0,
    csvs_extracted  INTEGER DEFAULT 0,
    rows_ingested   INTEGER DEFAULT 0,
    rows_skipped    INTEGER DEFAULT 0,
    status          TEXT    NOT NULL DEFAULT 'running',  -- running | ok | error | partial
    error_message   TEXT
);
"""

_CREATE_METADATA_MAP = """
CREATE TABLE IF NOT EXISTS metadata_map (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    original_column     TEXT    NOT NULL,
    normalized_column   TEXT    NOT NULL,
    dataset_type        TEXT    NOT NULL,
    notes               TEXT,
    UNIQUE (original_column, dataset_type)
);
"""

_CREATE_MACRO_SERIES_LATEST = """
CREATE TABLE IF NOT EXISTS macro_series_latest (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    bbg_ticker          TEXT    NOT NULL UNIQUE,
    description         TEXT,
    category            TEXT,               -- rates_usd | credit_spread | volatility | fx | monetary | inflation | global_equity | commodity
    px_last             REAL,               -- último valor
    date                TEXT    NOT NULL,   -- YYYY-MM-DD da última observação
    updated_at          TEXT    NOT NULL    -- ISO-8601 da última ingestão
);
CREATE INDEX IF NOT EXISTS idx_macro_latest_ticker   ON macro_series_latest (bbg_ticker);
CREATE INDEX IF NOT EXISTS idx_macro_latest_category ON macro_series_latest (category);
"""

_CREATE_MACRO_SERIES_HISTORY = """
CREATE TABLE IF NOT EXISTS macro_series_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    bbg_ticker  TEXT    NOT NULL,
    description TEXT,
    category    TEXT,
    date        TEXT    NOT NULL,   -- YYYY-MM-DD
    value       REAL,               -- NULL = dado ausente
    source_file TEXT,
    ingestion_timestamp TEXT,
    UNIQUE (bbg_ticker, date)       -- um valor por (ticker, data)
);
CREATE INDEX IF NOT EXISTS idx_macro_history_ticker ON macro_series_history (bbg_ticker);
CREATE INDEX IF NOT EXISTS idx_macro_history_date   ON macro_series_history (date);
"""

_CREATE_IV_HISTORY = """
CREATE TABLE IF NOT EXISTS iv_history (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT    NOT NULL,
    date                TEXT    NOT NULL,       -- YYYY-MM-DD
    iv                  REAL,                  -- IV ATM 30d em decimal (0.28 = 28%)
    source_file         TEXT,
    ingestion_timestamp TEXT,
    UNIQUE (ticker, date)                      -- um valor por (ticker, data)
);
CREATE INDEX IF NOT EXISTS idx_iv_hist_ticker ON iv_history (ticker);
CREATE INDEX IF NOT EXISTS idx_iv_hist_date   ON iv_history (date);
"""

_CREATE_MISSING_DATA_LOG = """
CREATE TABLE IF NOT EXISTS missing_data_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT    NOT NULL,
    source_file         TEXT    NOT NULL,
    row_number          INTEGER,
    ticker              TEXT,
    field               TEXT,
    reference_date      TEXT,
    issue_type          TEXT    NOT NULL,   -- missing_value | missing_ticker | missing_field
                                            -- missing_date  | invalid_numeric | malformed_row
    issue_description   TEXT,
    detected_at         TEXT    NOT NULL    -- ISO-8601
);
CREATE INDEX IF NOT EXISTS idx_missing_run ON missing_data_log (run_id);
"""


# ── Valores iniciais de metadata_map ─────────────────────────────────────────
# Documenta o mapeamento de colunas originais dos CSVs Bloomberg para nomes normalizados.

_METADATA_MAP_SEED: list[tuple[str, str, str, str]] = [
    # (original_column, normalized_column, dataset_type, notes)
    ("bbg_ticker",     "ticker",          "prices",          "Bloomberg ticker"),
    ("yf_ticker",      "yf_ticker",       "prices",          "Yahoo Finance ticker"),
    ("price",          "price",           "prices",          "Último preço"),
    ("prev_price",     "prev_price",      "prices",          "Fechamento anterior"),
    ("price_w",        "price_w",         "prices",          "Preço 5d atrás"),
    ("price_ytd",      "price_ytd",       "prices",          "Preço 1-Jan"),
    ("pe",             "pe",              "fundamentals",    "P/E ratio"),
    ("mktcap_b",       "mktcap_b",        "fundamentals",    "Market cap em bilhões"),
    ("beta",           "beta",            "fundamentals",    "Beta"),
    ("profit_margin",  "profit_margin",   "fundamentals",    "Margem de lucro"),
    ("debt_equity",    "debt_equity",     "fundamentals",    "Dívida/Patrimônio"),
    ("roe",            "roe",             "fundamentals",    "Return on equity"),
    ("dividend_yield", "dividend_yield",  "fundamentals",    "Dividend yield"),
    ("hi_52w",         "hi_52w",          "fundamentals",    "Máxima 52 semanas"),
    ("lo_52w",         "lo_52w",          "fundamentals",    "Mínima 52 semanas"),
    ("drawdown_52w",   "drawdown_52w",    "fundamentals",    "Drawdown 52 semanas"),
    ("atm_iv",         "atm_iv",          "options_iv",      "IV ATM"),
    ("skew_25d",       "skew_25d",        "options_iv",      "Skew 25-delta"),
    ("pcr_oi",         "pcr_oi",          "options_iv",      "Put/Call OI ratio"),
    ("spot",           "spot",            "gex_summary",     "Spot SPX"),
    ("gex_total_bn",   "gex_total_bn",    "gex_summary",     "GEX total em bilhões"),
    ("gex_call_bn",    "gex_call_bn",     "gex_summary",     "GEX calls em bilhões"),
    ("gex_put_bn",     "gex_put_bn",      "gex_summary",     "GEX puts em bilhões"),
    ("direction",      "direction",       "gex_summary",     "Direção gamma"),
    ("gamma_regime",   "gamma_regime",    "gex_summary",     "Regime gamma"),
    ("n_options",      "n_options",       "gex_summary",     "Número de opções"),
    ("nav",            "nav",             "letf_flows",      "NAV do ETF"),
    ("aum_b",          "aum_b",           "letf_flows",      "AUM em bilhões"),
    ("leverage",       "leverage",        "letf_flows",      "Alavancagem"),
]


def create_schema(db_path: Path) -> None:
    """
    Cria todas as tabelas e índices no banco SQLite.
    Idempotente: pode ser chamado mais de uma vez com segurança.

    Args:
        db_path: Caminho completo para o arquivo .db
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")     # melhor concorrência
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")   # bom equilíbrio segurança/velocidade

        for ddl in [
            _CREATE_PROCESSED_ZIP_REGISTRY,
            _CREATE_EXTRACTED_FILE_REGISTRY,
            _CREATE_BQL_TIMESERIES,
            _CREATE_BQL_LATEST,
            _CREATE_INGESTION_LOG,
            _CREATE_METADATA_MAP,
            _CREATE_MISSING_DATA_LOG,
            _CREATE_MACRO_SERIES_LATEST,
            _CREATE_MACRO_SERIES_HISTORY,
            _CREATE_IV_HISTORY,
        ]:
            # Cada DDL pode ter múltiplos statements separados por ;
            for stmt in ddl.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)

        # Popula metadata_map com mapeamentos conhecidos (ignora duplicatas)
        conn.executemany(
            """
            INSERT OR IGNORE INTO metadata_map
                (original_column, normalized_column, dataset_type, notes)
            VALUES (?, ?, ?, ?)
            """,
            _METADATA_MAP_SEED,
        )

        conn.commit()


def get_connection(db_path: Path) -> sqlite3.Connection:
    """
    Abre conexão com o banco, garante schema criado, retorna conexão.
    Caller é responsável por fechar (use como context manager).

    Returns:
        sqlite3.Connection com row_factory = sqlite3.Row
    """
    create_schema(db_path)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


if __name__ == "__main__":
    import sys
    from pathlib import Path as _P

    _root = _P(__file__).parent.parent
    sys.path.insert(0, str(_root))
    from config.settings import DATABASE_PATH

    create_schema(DATABASE_PATH)
    print(f"Schema criado com sucesso em: {DATABASE_PATH}")
