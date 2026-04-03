"""
MacroDesk Bloomberg Ecosystem — Query Layer
============================================

ÚNICA interface autorizada entre o MacroDesk e o banco de dados.

Regra absoluta:
  O MacroDesk NÃO lê CSV diretamente.
  O MacroDesk NÃO acessa Yahoo Finance.
  O MacroDesk NÃO acessa IBKR.
  O MacroDesk consulta APENAS este módulo.

Se o banco estiver vazio ou desatualizado, as funções retornam estruturas
vazias e is_data_available() retorna False — o MacroDesk deve exibir aviso
ao usuário em vez de tentar buscar dados de outra fonte.

Funções principais:
  get_latest_prices()        → dict compatível com market_prices.collect()
  get_fundamentals()         → dict de múltiplos por ticker
  get_options_iv()           → dict de volatilidade implícita por ticker
  get_gex_summary()          → dict de GEX do SPX
  get_letf_flows()           → lista de ETFs alavancados
  get_price_history()        → séries de preços por ticker
  get_last_ingestion_status() → info sobre última execução do agente
  is_data_available()        → bool — False = banco vazio ou > STALE_MINUTES

Uso:
    from app.query_layer import BloombergQueryLayer
    ql = BloombergQueryLayer()

    if not ql.is_data_available():
        print("Banco não atualizado. Execute o Bloomberg Agent primeiro.")
    else:
        prices = ql.get_latest_prices()
        funds  = ql.get_fundamentals()
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import sys
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import DATABASE_PATH, STALE_MINUTES


class BloombergQueryLayer:
    """
    Query layer Bloomberg-only para o MacroDesk.

    Args:
        db_path: Caminho para macrodesk.db. Se None, usa DATABASE_PATH de settings.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or DATABASE_PATH

    # ── Status e Disponibilidade ─────────────────────────────────────────────

    def is_data_available(self) -> bool:
        """
        Retorna True se o banco tem dados recentes (menos de STALE_MINUTES atrás).
        False = banco vazio, arquivo não existe, ou dados desatualizados.
        """
        status = self.get_last_ingestion_status()
        if not status or status.get("status") not in ("ok", "partial"):
            return False
        finished_at = status.get("finished_at") or ""
        if not finished_at:
            return False
        try:
            dt_finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
            age = datetime.now(timezone.utc) - dt_finished
            return age < timedelta(minutes=STALE_MINUTES)
        except Exception:
            return False

    def get_last_ingestion_status(self) -> dict[str, Any]:
        """
        Retorna informações sobre a última execução do agente.

        Returns:
            {
              "run_id": str,
              "started_at": str,
              "finished_at": str,
              "zips_processed": int,
              "rows_ingested": int,
              "status": str,
              "age_minutes": float,
            }
            Retorna {} se o banco não existir ou não tiver nenhuma execução.
        """
        try:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT run_id, started_at, finished_at,
                           zips_processed, rows_ingested, status
                    FROM ingestion_log
                    WHERE status IN ('ok', 'partial', 'no_new_data')
                    ORDER BY finished_at DESC
                    LIMIT 1
                    """
                ).fetchone()
                if not row:
                    return {}
                result = dict(row)
                # Calcula idade
                try:
                    dt_finished = datetime.fromisoformat(
                        (result.get("finished_at") or "").replace("Z", "+00:00")
                    )
                    result["age_minutes"] = round(
                        (datetime.now(timezone.utc) - dt_finished).total_seconds() / 60, 1
                    )
                except Exception:
                    result["age_minutes"] = None
                return result
        except Exception:
            return {}

    # ── Preços ───────────────────────────────────────────────────────────────

    def get_latest_prices(self) -> dict[str, dict[str, Any]]:
        """
        Retorna preços e retornos no formato compatível com market_prices.collect().

        Aceita dois formatos vindos do banco:
          - bql_export.py: 'price', 'daily_return', 'ytd_return' (já calculados)
          - legado: 'price', 'prev_price', 'price_w', 'price_ytd' (preços brutos)

        Returns:
            {
              "^GSPC": {
                "name":           "S&P 500",
                "price":          5100.23,
                "daily_return":   -0.0122,
                "ytd_return":     -0.085,
                "source":         "bloomberg",
              },
              ...
            }
            Retorna {} se não houver dados de preço no banco.
        """
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT ticker, field, latest_value
                    FROM bql_latest
                    WHERE field IN (
                        'price', 'daily_return', 'ytd_return',
                        'prev_price', 'price_w', 'price_ytd'
                    )
                    """
                ).fetchall()
        except Exception:
            return {}

        if not rows:
            return {}

        # Agrupa por ticker
        raw: dict[str, dict[str, Any]] = {}
        for row in rows:
            ticker, fld, val = row["ticker"], row["field"], row["latest_value"]
            raw.setdefault(ticker, {})[fld] = val

        # Monta resultado final
        result: dict[str, dict[str, Any]] = {}
        for ticker, vals in raw.items():
            price = vals.get("price")
            if price is None:
                continue

            price = float(price)
            entry: dict[str, Any] = {
                "name":   ticker,
                "price":  round(price, 4),
                "source": "bloomberg",
            }

            # Retorno diário: usa campo direto (bql_export.py) ou calcula de prev_price
            dr = vals.get("daily_return")
            prev_price = vals.get("prev_price")
            if dr is not None:
                entry["daily_return"] = round(float(dr), 6)
            elif prev_price and float(prev_price) != 0:
                entry["daily_return"] = round((price - float(prev_price)) / float(prev_price), 6)

            # Retorno YTD: usa campo direto (bql_export.py) ou calcula de price_ytd
            ytdr = vals.get("ytd_return")
            price_ytd = vals.get("price_ytd")
            if ytdr is not None:
                entry["ytd_return"] = round(float(ytdr), 6)
            elif price_ytd and float(price_ytd) != 0:
                entry["ytd_return"] = round((price - float(price_ytd)) / float(price_ytd), 6)

            # Retorno semanal: só no formato legado
            price_w = vals.get("price_w")
            if price_w and float(price_w) != 0:
                entry["weekly_return"] = round((price - float(price_w)) / float(price_w), 6)

            result[ticker] = entry

        # Tenta enriquecer com nomes amigáveis via node_registry
        try:
            sys.path.insert(0, str(_ROOT))
            from app.desk.node_registry import NODES
            for node in NODES.values():
                bbg = node.get("ticker") or ""
                yf  = node.get("meta", {}).get("yf_symbol") or ""
                label = node.get("label", "")
                for sym in [bbg, yf]:
                    if sym and sym in result:
                        result[sym]["name"] = label
        except Exception:
            pass

        return result

    # ── Fundamentos ──────────────────────────────────────────────────────────

    def get_fundamentals(self, ticker: str | None = None) -> dict[str, dict[str, Any]]:
        """
        Retorna múltiplos fundamentalistas por ticker.

        Args:
            ticker: Se fornecido, retorna apenas dados deste ticker.
                    Se None, retorna todos os tickers disponíveis.

        Returns:
            {
              "AAPL": {
                "pe": 32.5,
                "mktcap_b": 3741.0,
                "beta": 1.2,
                "profit_margin": 0.253,
                "debt_equity": 1.5,
                "roe": 1.7,
                "dividend_yield": 0.005,
                "price": 215.3,
                "hi_52w": 260.1,
                "lo_52w": 164.2,
                "drawdown_52w": -0.17,
              },
              ...
            }
        """
        fundamental_fields = [
            "pe", "mktcap_b", "beta", "profit_margin", "debt_equity",
            "roe", "dividend_yield", "price", "hi_52w", "lo_52w", "drawdown_52w",
        ]
        return self._get_latest_by_fields(fundamental_fields, ticker_filter=ticker)

    # ── Volatilidade Implícita ─────────────────────────────────────────────

    def get_options_iv(self, ticker: str | None = None) -> dict[str, dict[str, Any]]:
        """
        Retorna volatilidade implícita e skew por ticker.

        Returns:
            {
              "AAPL": {
                "atm_iv": 0.28,
                "skew_25d": 0.04,
                "pcr_oi": 0.8,
              },
              ...
            }
        """
        return self._get_latest_by_fields(["atm_iv", "skew_25d", "pcr_oi"], ticker_filter=ticker)

    # ── GEX ───────────────────────────────────────────────────────────────────

    def get_gex_summary(self) -> dict[str, Any]:
        """
        Retorna o resumo de Gamma Exposure do SPX.

        Returns:
            {
              "spot": 5200.0,
              "gex_total_bn": -1.5,
              "gex_call_bn": 2.0,
              "gex_put_bn": -3.5,
              "n_options": 45000,
              "date": "2026-04-03",
            }
        """
        gex_fields = ["spot", "gex_total_bn", "gex_call_bn", "gex_put_bn", "n_options"]
        data = self._get_latest_by_fields(gex_fields, ticker_filter="SPX_GEX")
        raw = data.get("SPX_GEX", {})
        if not raw:
            return {}
        # Adiciona a data mais recente disponível
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT latest_date FROM bql_latest WHERE ticker = 'SPX_GEX' ORDER BY latest_date DESC LIMIT 1"
                ).fetchone()
                if row:
                    raw["date"] = row[0]
        except Exception:
            pass
        return raw

    def get_gex_by_strike(self) -> list[dict[str, Any]]:
        """
        Retorna GEX por strike do SPX.

        Returns:
            [
              {"strike": 5200, "put_call": "call", "gex_bn": 0.5, "open_int": 12000, "gamma": 0.01},
              ...
            ]
            Ordenado por strike crescente.
        """
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT ticker, field, latest_value
                    FROM bql_latest
                    WHERE ticker LIKE 'SPX_GEX_%_%'
                    ORDER BY ticker
                    """
                ).fetchall()
        except Exception:
            return []

        # Reconstrói a estrutura por strike
        raw: dict[str, dict[str, Any]] = {}
        for row in rows:
            ticker, fld, val = row["ticker"], row["field"], row["latest_value"]
            # ticker formato: SPX_GEX_{strike}_{put_call}
            parts = ticker.split("_")
            if len(parts) >= 4:
                strike   = parts[2]
                put_call = parts[3]
                key      = f"{strike}_{put_call}"
                raw.setdefault(key, {"strike": strike, "put_call": put_call})[fld] = val

        result = sorted(raw.values(), key=lambda x: float(x.get("strike", 0)))
        return result

    # ── LETF Flows ────────────────────────────────────────────────────────────

    def get_letf_flows(self) -> list[dict[str, Any]]:
        """
        Retorna fluxos e AUM de ETFs alavancados.

        Returns:
            [
              {"ticker": "TQQQ", "leverage": 3, "nav": 52.3, "aum_b": 18.5},
              ...
            ]
        """
        letf_fields = ["leverage", "nav", "aum_b"]
        data = self._get_latest_by_fields(letf_fields)
        # Filtra tickers que têm pelo menos nav ou aum_b
        result = []
        for ticker, vals in data.items():
            if vals.get("nav") is not None or vals.get("aum_b") is not None:
                entry = {"ticker": ticker}
                entry.update(vals)
                result.append(entry)
        return sorted(result, key=lambda x: x.get("aum_b") or 0, reverse=True)

    # ── Histórico de Preços ────────────────────────────────────────────────

    def get_price_history(
        self,
        ticker: str | None = None,
        days: int = 252,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Retorna série histórica de preços de fechamento.

        Args:
            ticker: Ticker específico ou None para todos.
            days:   Quantos dias de histórico retornar.

        Returns:
            {
              "AAPL": [
                {"date": "2025-01-02", "price": 185.0},
                {"date": "2025-01-03", "price": 187.5},
                ...
              ],
              ...
            }
        """
        try:
            with self._connect() as conn:
                if ticker:
                    rows = conn.execute(
                        """
                        SELECT ticker, date, value
                        FROM bql_timeseries
                        WHERE ticker = ? AND field = 'price'
                        ORDER BY date DESC
                        LIMIT ?
                        """,
                        (ticker, days),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT ticker, date, value
                        FROM bql_timeseries
                        WHERE field = 'price'
                          AND date >= date('now', ? || ' days')
                        ORDER BY ticker, date
                        """,
                        (f"-{days}",),
                    ).fetchall()
        except Exception:
            return {}

        result: dict[str, list[dict]] = {}
        for row in rows:
            t, d, v = row["ticker"], row["date"], row["value"]
            result.setdefault(t, []).append({"date": d, "price": v})

        return result

    # ── Macro Series ──────────────────────────────────────────────────────────

    def get_macro_series(self, category: str | None = None) -> dict[str, dict[str, Any]]:
        """
        Retorna séries macroeconômicas do banco (curva de juros, spreads, VIX term, etc.).

        Args:
            category: Filtro opcional de categoria:
                      "rates_usd" | "credit_spread" | "volatility" | "fx" |
                      "monetary" | "inflation" | "global_equity" | "commodity" |
                      "rates_derived" | "volatility_derived"

        Returns:
            {
              "USGG10YR Index": {
                "description": "US Treasury 10yr",
                "category":    "rates_usd",
                "px_last":     4.25,
                "date":        "2026-04-03",
              },
              "US_2Y10Y_SPREAD": {
                "description": "US 2yr-10yr Spread (calculado)",
                "category":    "rates_derived",
                "px_last":     0.15,
                ...
              },
              ...
            }
        """
        try:
            sql = "SELECT bbg_ticker, description, category, px_last, date FROM macro_series_latest"
            params: list[Any] = []
            if category:
                sql += " WHERE category = ?"
                params.append(category)
            sql += " ORDER BY category, bbg_ticker"

            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()

            result: dict[str, dict[str, Any]] = {}
            for row in rows:
                result[row["bbg_ticker"]] = {
                    "description": row["description"],
                    "category":    row["category"],
                    "px_last":     row["px_last"],
                    "date":        row["date"],
                }
            return result
        except Exception:
            return {}

    def get_macro_history(
        self,
        bbg_ticker: str | None = None,
        category: str | None = None,
        days: int = 252,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Retorna histórico de séries macro.

        Returns:
            {
              "USGG10YR Index": [
                {"date": "2025-01-02", "value": 4.10},
                ...
              ],
            }
        """
        try:
            sql = f"""
                SELECT bbg_ticker, date, value
                FROM macro_series_history
                WHERE date >= date('now', '-{days} days')
            """
            params: list[Any] = []
            if bbg_ticker:
                sql += " AND bbg_ticker = ?"
                params.append(bbg_ticker)
            if category:
                sql += " AND category = ?"
                params.append(category)
            sql += " ORDER BY bbg_ticker, date"

            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()

            result: dict[str, list[dict]] = {}
            for row in rows:
                result.setdefault(row["bbg_ticker"], []).append({
                    "date": row["date"], "value": row["value"]
                })
            return result
        except Exception:
            return {}

    def get_yield_curve(self) -> dict[str, float | None]:
        """
        Retorna a curva de juros americana atual.

        Returns:
            {
              "1M": 4.32, "3M": 4.28, "6M": 4.20, "1Y": 4.10,
              "2Y": 4.05, "5Y": 4.12, "10Y": 4.25, "30Y": 4.45,
              "2Y10Y": 0.20,   # spread calculado
            }
        """
        macro = self.get_macro_series(category="rates_usd")
        derived = self.get_macro_series(category="rates_derived")

        TENOR_MAP = {
            "USGG1M Index":   "1M",
            "USGG3M Index":   "3M",
            "USGG6M Index":   "6M",
            "USGG1YR Index":  "1Y",
            "USGG2YR Index":  "2Y",
            "USGG5YR Index":  "5Y",
            "USGG10YR Index": "10Y",
            "USGG30YR Index": "30Y",
        }

        result: dict[str, float | None] = {}
        for bbg, tenor in TENOR_MAP.items():
            result[tenor] = macro.get(bbg, {}).get("px_last")

        # Derivados
        spread = derived.get("US_2Y10Y_SPREAD", {}).get("px_last")
        if spread is not None:
            result["2Y10Y"] = spread

        return result

    def get_vix_term_structure(self) -> dict[str, float | None]:
        """
        Retorna a estrutura a termo do VIX.

        Returns:
            {"spot": 18.5, "9d": 17.2, "3m": 19.8, "ratio_9d": 0.93, "ratio_3m": 1.07}
        """
        macro   = self.get_macro_series(category="volatility")
        derived = self.get_macro_series(category="volatility_derived")

        result = {
            "spot": macro.get("VIX Index",   {}).get("px_last"),
            "9d":   macro.get("VIX9D Index", {}).get("px_last"),
            "3m":   macro.get("VIX3M Index", {}).get("px_last"),
            "vvix": macro.get("VVIX Index",  {}).get("px_last"),
            "move": macro.get("MOVE Index",  {}).get("px_last"),
        }

        result["ratio_9d"] = derived.get("VIX_TERM_9D_SP", {}).get("px_last")
        result["ratio_3m"] = derived.get("VIX_TERM_3M_SP", {}).get("px_last")
        return result

    def get_credit_spreads(self) -> dict[str, float | None]:
        """
        Retorna spreads de crédito atuais.

        Returns:
            {"ig_oas": 92.3, "hy_oas": 320.5, "cdx_ig": 60.2, "cdx_hy": 310.0}
        """
        macro = self.get_macro_series(category="credit_spread")
        return {
            "ig_oas":  macro.get("LUACOAS Index", {}).get("px_last"),
            "hy_oas":  macro.get("LF98OAS Index", {}).get("px_last"),
            "cdx_ig":  macro.get("CDXIG Index",   {}).get("px_last"),
            "cdx_hy":  macro.get("CDXHY Index",   {}).get("px_last"),
        }

    # ── Privados ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Abre conexão com row_factory=sqlite3.Row."""
        if not self._db_path.exists():
            raise FileNotFoundError(
                f"Banco não encontrado: {self._db_path}\n"
                "Execute o Bloomberg Agent antes de usar o MacroDesk."
            )
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_latest_by_fields(
        self,
        fields: list[str],
        ticker_filter: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Busca os valores mais recentes de múltiplos campos na bql_latest.
        Retorna {ticker: {field: value, ...}}.
        """
        try:
            placeholders = ",".join("?" * len(fields))
            params: list[Any] = list(fields)

            sql = f"""
                SELECT ticker, field, latest_value
                FROM bql_latest
                WHERE field IN ({placeholders})
            """
            if ticker_filter:
                sql += " AND ticker = ?"
                params.append(ticker_filter)

            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()

        except Exception:
            return {}

        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            ticker, fld, val = row["ticker"], row["field"], row["latest_value"]
            result.setdefault(ticker, {})[fld] = val

        return result


# ── Instância global conveniente ──────────────────────────────────────────────
# Permite uso direto: from app.query_layer import ql
ql = BloombergQueryLayer()


# ── Mensagem padrão para MacroDesk sem dados ─────────────────────────────────
BANCO_VAZIO_HTML = """
<div style="background:#1a1a2e;border:1px solid #f59e0b;border-radius:8px;padding:24px;
            text-align:center;color:#f59e0b;font-family:monospace;max-width:600px;margin:40px auto">
    <div style="font-size:1.5rem;margin-bottom:12px">⚠ Banco Bloomberg não atualizado</div>
    <div style="color:#94a3b8;font-size:0.9rem;line-height:1.6">
        O banco de dados Bloomberg ainda não foi atualizado nesta sessão.<br>
        Execute o <strong>Bloomberg Agent</strong> antes de usar o MacroDesk.<br><br>
        <span style="color:#60a5fa">Clique em "Iniciar Bloomberg Agent" na área de trabalho</span><br>
        ou rode: <code style="color:#34d399">python -m core.bloomberg_main_agent</code>
    </div>
</div>
"""
