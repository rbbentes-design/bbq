"""
BQL Database — acumula histórico das exportações BQuant em SQLite.
Cada vez que um novo zip é extraído, os CSVs são inseridos com timestamp.
"""

from __future__ import annotations

import csv
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

BQL_DATA_DIR = Path(r"C:\Users\rafael bentes\bbg\agente\bql_data")
DB_PATH      = BQL_DATA_DIR / "bql_history.db"

_CREATE = """
CREATE TABLE IF NOT EXISTS fundamentals (
    ts TEXT, ticker TEXT, pe REAL, mktcap_b REAL, beta REAL,
    profit_margin REAL, debt_equity REAL, roe REAL,
    dividend_yield REAL, price REAL, hi_52w REAL, lo_52w REAL,
    drawdown_52w REAL
);
CREATE TABLE IF NOT EXISTS options_iv (
    ts TEXT, ticker TEXT, atm_iv REAL, skew_25d REAL, pcr_oi REAL
);
CREATE TABLE IF NOT EXISTS gex_summary (
    ts TEXT, date TEXT, spot REAL, gex_total_bn REAL,
    gex_call_bn REAL, gex_put_bn REAL, direction TEXT,
    gamma_regime TEXT, n_options INTEGER
);
CREATE TABLE IF NOT EXISTS letf_flows (
    ts TEXT, ticker TEXT, nav REAL, aum_b REAL,
    leverage INTEGER, idx TEXT
);
"""


def _conn() -> sqlite3.Connection:
    BQL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(_CREATE)
    con.commit()
    return con


def _read_csv(prefix: str) -> list[dict[str, str]]:
    dated = sorted(BQL_DATA_DIR.glob(f"{prefix}_*.csv"), reverse=True)
    path  = dated[0] if dated else BQL_DATA_DIR / f"{prefix}.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(val: Any) -> float | None:
    try:
        return float(val) if val not in (None, "", "nan") else None
    except (ValueError, TypeError):
        return None


def append_from_csvs() -> int:
    """
    Lê os CSVs mais recentes e insere no banco com timestamp atual.
    Retorna número de linhas inseridas.
    """
    ts  = datetime.now().isoformat(timespec="seconds")
    con = _conn()
    total = 0

    # fundamentals
    rows = _read_csv("fundamentals")
    for r in rows:
        con.execute(
            "INSERT INTO fundamentals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ts, r.get("ticker"), _f(r.get("pe")), _f(r.get("mktcap_b")),
             _f(r.get("beta")), _f(r.get("profit_margin")), _f(r.get("debt_equity")),
             _f(r.get("roe")), _f(r.get("dividend_yield")), _f(r.get("price")),
             _f(r.get("hi_52w")), _f(r.get("lo_52w")), _f(r.get("drawdown_52w")))
        )
    total += len(rows)

    # options_iv
    rows = _read_csv("options_iv")
    for r in rows:
        con.execute(
            "INSERT INTO options_iv VALUES (?,?,?,?,?)",
            (ts, r.get("ticker"), _f(r.get("atm_iv")),
             _f(r.get("skew_25d")), _f(r.get("pcr_oi")))
        )
    total += len(rows)

    # gex_summary
    rows = _read_csv("gex_summary")
    for r in rows:
        con.execute(
            "INSERT INTO gex_summary VALUES (?,?,?,?,?,?,?,?,?)",
            (ts, r.get("date"), _f(r.get("spot")),
             _f(r.get("gex_total_bn")), _f(r.get("gex_call_bn")),
             _f(r.get("gex_put_bn")), r.get("direction"),
             r.get("gamma_regime"), int(r.get("n_options") or 0))
        )
    total += len(rows)

    # letf_flows
    rows = _read_csv("letf_flows")
    for r in rows:
        con.execute(
            "INSERT INTO letf_flows VALUES (?,?,?,?,?,?)",
            (ts, r.get("ticker"), _f(r.get("nav")), _f(r.get("aum_b")),
             int(r.get("leverage") or 0), r.get("index"))
        )
    total += len(rows)

    con.commit()
    con.close()
    return total


def query(sql: str, params: tuple = ()) -> list[dict]:
    """Executa query e retorna lista de dicts."""
    con = _conn()
    con.row_factory = sqlite3.Row
    rows = con.execute(sql, params).fetchall()
    con.close()
    return [dict(r) for r in rows]


def last_n(table: str, n: int = 10) -> list[dict]:
    return query(f"SELECT * FROM {table} ORDER BY ts DESC LIMIT ?", (n,))
