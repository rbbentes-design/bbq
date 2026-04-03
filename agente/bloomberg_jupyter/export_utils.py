"""
Bloomberg Jupyter — Export Utilities
======================================

Utilitários compartilhados por todos os scripts de coleta do BQuant.

Funções:
    bql_to_df(resp, field)       → DataFrame de um campo BQL
    safe_float(val)              → float ou None
    make_zip(csv_files, prefix)  → cria bql_data_{date}.zip e retorna path
    ensure_output_dir()          → cria pasta de saída, retorna Path
    log(msg)                     → print com timestamp
    now_str()                    → "2026-04-03T14:30:15"
    today_str()                  → "2026-04-03"

Compatível com:
    - Python 3.9+ (BQuant padrão)
    - bql library (disponível no ambiente BQuant/BQNT)

Uso:
    from export_utils import bql_to_df, make_zip, log
"""

from __future__ import annotations

import io
import math
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# ── Detecção de ambiente ────────────────────────────────────────────────────
def is_bquant() -> bool:
    """True se estiver rodando no ambiente Bloomberg BQuant."""
    try:
        import bql  # noqa: F401
        return True
    except ImportError:
        return False


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


# ── Pasta de saída ───────────────────────────────────────────────────────────

def ensure_output_dir() -> Path:
    """
    Detecta a pasta de saída correta dependendo do ambiente:
    - BQuant (Linux): ~/bql_exports/
    - Local (Windows): ~/Downloads/
    """
    if os.name == "nt":
        # Windows local
        base = Path.home() / "Downloads"
    else:
        # Linux BQuant
        base = Path.home() / "bql_exports"
    base.mkdir(parents=True, exist_ok=True)
    return base


# ── BQL helpers ──────────────────────────────────────────────────────────────

def bql_to_df(resp: Any, field_name: str) -> "Optional[pd.DataFrame]":
    """
    Converte resposta BQL de um campo para DataFrame pandas.

    Args:
        resp:       Objeto de resposta do bq.execute()
        field_name: Nome do campo BQL (ex: "px_last")

    Returns:
        DataFrame com índice = ticker e coluna = valor.
        None se falhar.
    """
    try:
        import pandas as pd
        df = bql.combined_df(resp)
        return df
    except Exception:
        pass
    try:
        # Alternativa: acessa o campo diretamente
        return resp[field_name].df()
    except Exception:
        return None


def bql_extract_values(resp: Any, field_name: str) -> dict[str, Any]:
    """
    Extrai {ticker: value} de uma resposta BQL single-field.
    Converte NaN para None.
    """
    result: dict[str, Any] = {}
    try:
        df = resp[field_name].df()
        for ticker, val in df[field_name].items():
            result[ticker] = safe_float(val)
    except Exception as exc:
        log(f"AVISO: bql_extract_values falhou para {field_name}: {exc}")
    return result


def bql_extract_timeseries(resp: Any, field_name: str) -> dict[str, list[tuple[str, float]]]:
    """
    Extrai séries temporais de uma resposta BQL historical.

    Returns:
        {ticker: [(date_str, value), ...]} ordenado por data asc
    """
    result: dict[str, list[tuple[str, float]]] = {}
    try:
        df = resp[field_name].df()
        # DataFrame com MultiIndex (ticker, date) ou colunas variadas
        if hasattr(df.index, "levels"):
            # MultiIndex: nível 0 = ticker, nível 1 = date
            for (ticker, date), row in df.iterrows():
                val = safe_float(row.get(field_name) or row.iloc[0])
                if val is not None:
                    result.setdefault(ticker, []).append((str(date)[:10], val))
        else:
            # Index simples (single ticker case)
            ticker = df.index.name or "unknown"
            for date, row in df.iterrows():
                val = safe_float(row.get(field_name) or row.iloc[0])
                if val is not None:
                    result.setdefault(ticker, []).append((str(date)[:10], val))
    except Exception as exc:
        log(f"AVISO: bql_extract_timeseries falhou para {field_name}: {exc}")
    # Ordena cada série por data
    for ticker in result:
        result[ticker].sort(key=lambda x: x[0])
    return result


def safe_float(val: Any) -> Optional[float]:
    """Converte para float ou retorna None se NaN/None/inválido."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


# ── CSV helpers ───────────────────────────────────────────────────────────────

def write_csv(
    output_dir: Path,
    prefix: str,
    rows: list[dict],
    fieldnames: list[str] | None = None,
) -> Path:
    """
    Grava um CSV na pasta de saída com sufixo de data.

    Args:
        output_dir: Pasta de destino.
        prefix:     Prefixo do arquivo (ex: "prices").
        rows:       Lista de dicts com os dados.
        fieldnames: Ordem das colunas. Se None, usa as chaves do primeiro registro.

    Returns:
        Path do arquivo gravado.
    """
    import csv as _csv

    if not rows:
        log(f"  AVISO: nenhuma linha para gravar em {prefix}_{today_str()}.csv")
        return output_dir / f"{prefix}_{today_str()}.csv"

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    filename = f"{prefix}_{today_str()}.csv"
    path = output_dir / filename

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    log(f"  CSV gravado: {filename} ({len(rows)} linhas)")
    return path


def write_meta_csv(output_dir: Path, extra: dict | None = None) -> Path:
    """Grava meta_{today}.csv com timestamp e info do export."""
    meta = {"key": "generated_at", "value": now_str()}
    rows = [meta]
    if extra:
        for k, v in extra.items():
            rows.append({"key": k, "value": str(v)})
    return write_csv(output_dir, "meta", rows, fieldnames=["key", "value"])


# ── ZIP ───────────────────────────────────────────────────────────────────────

def make_zip(csv_files: list[Path], output_dir: Path, zip_prefix: str = "bql_data") -> Path:
    """
    Cria um arquivo .zip com todos os CSVs fornecidos.
    Nome: bql_data_{YYYY-MM-DD}.zip

    Args:
        csv_files:  Lista de arquivos CSV para incluir no zip.
        output_dir: Pasta onde o zip será salvo.
        zip_prefix: Prefixo do nome do zip.

    Returns:
        Path do arquivo .zip criado.
    """
    zip_name = f"{zip_prefix}_{today_str()}.zip"
    zip_path = output_dir / zip_name

    # Se o zip do dia já existe, usa sufixo de hora para evitar sobrescrita
    if zip_path.exists():
        hour_str = datetime.now().strftime("%H%M%S")
        zip_name = f"{zip_prefix}_{today_str()}_{hour_str}.zip"
        zip_path = output_dir / zip_name

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for csv_path in csv_files:
            if csv_path.exists():
                zf.write(csv_path, csv_path.name)
                log(f"  + {csv_path.name}")

    size_kb = round(zip_path.stat().st_size / 1024, 1)
    log(f"ZIP criado: {zip_name} ({size_kb} KB)")
    return zip_path


# ── Ticker universo ───────────────────────────────────────────────────────────

# Universo de preços: (bbg_ticker, yf_ticker, nome_legível)
PRICE_UNIVERSE: list[tuple[str, str, str]] = [
    # Índices
    ("SPX Index",      "^GSPC",    "S&P 500"),
    ("NDX Index",      "^NDX",     "Nasdaq 100"),
    ("INDU Index",     "^DJI",     "Dow Jones"),
    ("RTY Index",      "^RUT",     "Russell 2000"),
    ("SX5E Index",     "FEZ",      "EuroStoxx 50"),
    ("NKY Index",      "EWJ",      "Nikkei 225"),
    ("MXEF Index",     "EEM",      "MSCI EM"),
    ("MXWO Index",     "URTH",     "MSCI World"),
    ("IBOV Index",     "EWZ",      "Ibovespa"),
    # Volatilidade
    ("VIX Index",      "^VIX",     "VIX"),
    ("VIX3M Index",    "^VIX",     "VIX 3M"),
    ("VVIX Index",     "^VVIX",    "VVIX"),
    # Juros
    ("USGG10YR Index", "^TNX",     "Treasury 10yr"),
    ("USGG2YR Index",  "^IRX",     "Treasury 2yr"),
    ("USGG30YR Index", "^TYX",     "Treasury 30yr"),
    # ETFs de renda fixa
    ("TLT US Equity",  "TLT",      "Treasury 20yr (TLT)"),
    ("HYG US Equity",  "HYG",      "High Yield (HYG)"),
    ("LQD US Equity",  "LQD",      "Investment Grade (LQD)"),
    # Moedas
    ("DXY Curncy",     "DX-Y.NYB", "US Dollar Index"),
    ("EURUSD Curncy",  "EURUSD=X", "EUR/USD"),
    ("USDJPY Curncy",  "JPY=X",    "USD/JPY"),
    ("GBPUSD Curncy",  "GBPUSD=X", "GBP/USD"),
    # Commodities
    ("XAU Curncy",     "GLD",      "Gold"),
    ("CO1 Comdty",     "BZ=F",     "Brent Crude"),
    ("CL1 Comdty",     "CL=F",     "WTI Crude"),
    ("NG1 Comdty",     "NG=F",     "Natural Gas"),
    # Crypto
    ("XBT Curncy",     "BTC-USD",  "Bitcoin"),
    ("ETH Curncy",     "ETH-USD",  "Ethereum"),
    # Mag7 + big caps
    ("AAPL US Equity", "AAPL",     "Apple"),
    ("MSFT US Equity", "MSFT",     "Microsoft"),
    ("NVDA US Equity", "NVDA",     "NVIDIA"),
    ("AMZN US Equity", "AMZN",     "Amazon"),
    ("GOOGL US Equity","GOOGL",    "Alphabet"),
    ("META US Equity", "META",     "Meta"),
    ("TSLA US Equity", "TSLA",     "Tesla"),
    ("AVGO US Equity", "AVGO",     "Broadcom"),
    ("JPM US Equity",  "JPM",      "JPMorgan"),
    ("LLY US Equity",  "LLY",      "Eli Lilly"),
]

# Universo de fundamentos (apenas equities com dados fundamentalistas)
FUNDAMENTAL_UNIVERSE: list[str] = [
    "AAPL US Equity", "MSFT US Equity", "NVDA US Equity", "AMZN US Equity",
    "GOOGL US Equity", "META US Equity", "TSLA US Equity", "AVGO US Equity",
    "JPM US Equity", "LLY US Equity", "UNH US Equity", "XOM US Equity",
    "COST US Equity", "V US Equity", "MA US Equity", "WMT US Equity",
    "NFLX US Equity", "JNJ US Equity", "PG US Equity", "BRK/B US Equity",
    "SPY US Equity", "QQQ US Equity", "GLD US Equity", "TLT US Equity",
    "HYG US Equity",
]

# Universo de macro series
MACRO_UNIVERSE: list[tuple[str, str, str]] = [
    # (bbg_ticker, campo_principal, descrição)
    # Curva de juros EUA
    ("USGG1M Index",    "px_last", "US Treasury 1M"),
    ("USGG3M Index",    "px_last", "US Treasury 3M"),
    ("USGG6M Index",    "px_last", "US Treasury 6M"),
    ("USGG1YR Index",   "px_last", "US Treasury 1yr"),
    ("USGG2YR Index",   "px_last", "US Treasury 2yr"),
    ("USGG5YR Index",   "px_last", "US Treasury 5yr"),
    ("USGG10YR Index",  "px_last", "US Treasury 10yr"),
    ("USGG30YR Index",  "px_last", "US Treasury 30yr"),
    # Spreads de crédito
    ("LUACOAS Index",   "px_last", "IG OAS Spread"),
    ("LF98OAS Index",   "px_last", "HY OAS Spread"),
    ("CDXIG Index",     "px_last", "CDX IG 5yr"),
    ("CDXHY Index",     "px_last", "CDX HY 5yr"),
    # Volatilidade
    ("VIX Index",       "px_last", "VIX Spot"),
    ("VIX9D Index",     "px_last", "VIX 9-day"),
    ("VIX3M Index",     "px_last", "VIX 3-month"),
    ("VVIX Index",      "px_last", "VVIX"),
    ("MOVE Index",      "px_last", "MOVE (bond vol)"),
    # DXY e FX
    ("DXY Curncy",      "px_last", "DXY"),
    # Liquidez / Fed
    ("FDFD Index",      "px_last", "Fed Funds Rate"),
    ("SOFRRATE Index",  "px_last", "SOFR"),
    # Inflação
    ("USSWIT10 Index",  "px_last", "US 10yr Breakeven"),
    ("USSWIT2 Index",   "px_last", "US 2yr Breakeven"),
    # Global
    ("MXEF Index",      "px_last", "MSCI EM"),
    ("MXWO Index",      "px_last", "MSCI World"),
    ("SX5E Index",      "px_last", "EuroStoxx 50"),
    # Commodities macro
    ("XAU Curncy",      "px_last", "Gold"),
    ("CO1 Comdty",      "px_last", "Brent Crude"),
]
