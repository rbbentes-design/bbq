"""
Provider: Interactive Brokers (ib_insync)

Backbone principal de market data:
  - Preços e retornos diários/semanais/YTD via histórico IBKR
  - Snapshots em tempo real (bid/ask/last/volume)
  - Option chains com greeks (Fase 4)

Conexão: TWS porta 7497 (paper) ou 4001 (IB Gateway)
"""

from __future__ import annotations

import time
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("providers.ibkr")

# ── Configuração de conexão ───────────────────────────────────────────────────
IBKR_HOST      = "127.0.0.1"
IBKR_PORT      = 7497
IBKR_CLIENT_ID = 20          # evitar conflito com TWS/outros clientes
CONNECT_TIMEOUT = 10         # segundos para aguardar conexão
REQUEST_DELAY   = 0.05       # throttle entre requisições (50ms)
BATCH_SIZE      = 40         # max contratos por lote de histData

# ── whatToShow por tipo de contrato ──────────────────────────────────────────
_WHAT_TO_SHOW: dict[str, str] = {
    "STK":      "TRADES",
    "IND":      "TRADES",
    "CASH":     "MIDPOINT",
    "FUT":      "TRADES",
    "CONTFUT":  "TRADES",
    "CRYPTO":   "TRADES",
    "ETF":      "TRADES",
}

# ── Mapeamento yfinance symbol → contrato IB ─────────────────────────────────
# Formato: ticker → (type, *args)
#   type 'stk'     → Stock(symbol, exchange, currency)
#   type 'index'   → Index(symbol, exchange)
#   type 'forex'   → Forex(pair)
#   type 'contfut' → ContFuture(symbol, exchange)
#   type 'crypto'  → Crypto(symbol, exchange, currency)
_CONTRACT_MAP: dict[str, tuple] = {
    # ── US Equity Indices ────────────────────────────────────────────────────
    "^GSPC":      ("index",   "SPX",        "CBOE"),
    "^NDX":       ("index",   "NDX",        "NASDAQ"),
    "^RUT":       ("index",   "RUT",        "RUSSELL"),
    "^VIX":       ("index",   "VIX",        "CBOE"),
    "^VIX9D":     ("index",   "VIX9D",      "CBOE"),
    "^VIX3M":     ("index",   "VIX3M",      "CBOE"),
    "^TNX":       ("index",   "TNX",        "CBOE"),   # 10Y yield
    "^TYX":       ("index",   "TYX",        "CBOE"),   # 30Y yield
    "^IRX":       ("index",   "IRX",        "CBOE"),   # 13-wk T-bill
    "^FVX":       ("index",   "FVX",        "CBOE"),   # 5Y yield
    # ── FX ───────────────────────────────────────────────────────────────────
    "EURUSD=X":   ("forex",   "EURUSD"),
    "USDJPY=X":   ("forex",   "USDJPY"),
    "GBPUSD=X":   ("forex",   "GBPUSD"),
    "USDCNH=X":   ("forex",   "USDCNH"),
    "USDBRL=X":   ("forex",   "USDBRL"),
    "USDMXN=X":   ("forex",   "USDMXN"),
    "AUDUSD=X":   ("forex",   "AUDUSD"),
    "USDCHF=X":   ("forex",   "USDCHF"),
    "DX-Y.NYB":   ("contfut", "DX",         "NYBOT"),  # DXY futures
    # ── Commodities (continuous futures) ─────────────────────────────────────
    "CL=F":       ("contfut", "CL",         "NYMEX"),  # WTI crude
    "NG=F":       ("contfut", "NG",         "NYMEX"),  # Natural gas
    "HG=F":       ("contfut", "HG",         "COMEX"),  # Copper
    "ZW=F":       ("contfut", "ZW",         "CBOT"),   # Wheat
    "ZC=F":       ("contfut", "ZC",         "CBOT"),   # Corn
    "ZS=F":       ("contfut", "ZS",         "CBOT"),   # Soybeans
    "GC=F":       ("contfut", "GC",         "COMEX"),  # Gold futures
    "SI=F":       ("contfut", "SI",         "COMEX"),  # Silver futures
    # ── Crypto ───────────────────────────────────────────────────────────────
    "BTC-USD":    ("crypto",  "BTC",        "PAXOS",   "USD"),
    "ETH-USD":    ("crypto",  "ETH",        "PAXOS",   "USD"),
    # ── EU Equity Indices ─────────────────────────────────────────────────────
    "^STOXX50E":  ("index",   "STOXX50E",   "DTB"),
    "^GDAXI":     ("index",   "DAX",        "DTB"),
    "^FCHI":      ("index",   "CAC40",      "MONEP"),
    "^FTSE":      ("index",   "FTSE100",    "LIFFE"),
    "^IBEX":      ("index",   "IBEX35",     "MEFF"),
    "^AEX":       ("index",   "AEX",        "FTA"),
    # ── Asia / EM Equity Indices ──────────────────────────────────────────────
    "^N225":      ("index",   "NI225",      "OSE.JPN"),
    "^HSI":       ("index",   "HSI",        "HKFE"),
    "^NSEI":      ("index",   "NIFTY50",    "NSE"),
    "^BVSP":      ("index",   "IBOV",       "BOVESPA"),
    "^AXJO":      ("index",   "AS51",       "ASX"),    # ASX 200
    "^TWII":      ("index",   "TWII",       "TWSE"),   # Taiwan
    "000300.SS":  ("index",   "CSI300",     "SZSE"),
}

# ETF fallback para índices globais sem subscrição de dados
_ETF_FALLBACK: dict[str, str] = {
    "^STOXX50E": "FEZ",   # Euro Stoxx 50
    "^GDAXI":    "EWG",   # Germany
    "^FCHI":     "EWQ",   # France
    "^FTSE":     "EWU",   # UK
    "^N225":     "EWJ",   # Japan
    "^HSI":      "EWH",   # Hong Kong
    "^NSEI":     "INDA",  # India
    "^BVSP":     "EWZ",   # Brazil
    "^AXJO":     "EWA",   # Australia
    "000300.SS": "ASHR",  # China A-shares
}


# ── Contract factory ──────────────────────────────────────────────────────────

def _make_contract(ticker: str):
    """Converte ticker formato yfinance → contrato ib_insync."""
    try:
        from ib_insync import Stock, Index, Forex, ContFuture, Crypto
    except ImportError:
        return None

    spec = _CONTRACT_MAP.get(ticker)
    if spec is None:
        # Default: US stock via SMART router
        clean = ticker.replace("^", "").replace("-USD", "").replace("=X", "")
        return Stock(clean, "SMART", "USD")

    kind = spec[0]
    if kind == "index":
        return Index(spec[1], spec[2])
    if kind == "forex":
        return Forex(spec[1])
    if kind == "contfut":
        return ContFuture(spec[1], spec[2])
    if kind == "crypto":
        return Crypto(spec[1], spec[2], spec[3])
    # stk
    return Stock(spec[1], spec[2] if len(spec) > 2 else "SMART",
                 spec[3] if len(spec) > 3 else "USD")


def _what_to_show(contract) -> str:
    sec = getattr(contract, "secType", "STK")
    return _WHAT_TO_SHOW.get(sec, "TRADES")


# ── Conexão ───────────────────────────────────────────────────────────────────

def _connect_ib():
    """Conecta ao TWS. Retorna IB instance ou None."""
    import asyncio
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    try:
        from ib_insync import IB, util
        util.logToConsole('ERROR')   # silencia logs internos do ib_insync
        ib = IB()
        ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID,
                   timeout=CONNECT_TIMEOUT, readonly=True)
        if ib.isConnected():
            _log.info("ibkr_connected", host=IBKR_HOST, port=IBKR_PORT)
            return ib
        _log.warning("ibkr_connect_failed")
        return None
    except Exception as exc:
        _log.warning("ibkr_connect_error", error=str(exc))
        return None


# ── Coleta de dados históricos ────────────────────────────────────────────────

def _fetch_historical(ib, contract, duration: str = "6 D",
                      bar_size: str = "1 day") -> list:
    """Busca barras históricas. Retorna lista de BarData."""
    try:
        what = _what_to_show(contract)
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what,
            useRTH=True,
            formatDate=1,
        )
        return bars or []
    except Exception as exc:
        _log.debug("hist_data_error",
                   sym=getattr(contract, "symbol", "?"), error=str(exc))
        return []


def _fetch_ytd_historical(ib, contract) -> list:
    """Busca barras desde 1-Jan do ano atual."""
    from datetime import date
    year = date.today().year
    try:
        what = _what_to_show(contract)
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=f"{date.today().timetuple().tm_yday + 5} D",
            barSizeSetting="1 day",
            whatToShow=what,
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            return []
        # Filtra barras do ano corrente
        return [b for b in bars if str(b.date)[:4] == str(year)]
    except Exception:
        return []


# ── Histórico de closes para análise de rede ──────────────────────────────────

def fetch_historical_closes(
    tickers: list[str],
    lookback_days: int = 90,
    client_id: int = 21,       # ID separado do live loop (que usa 20)
) -> dict[str, list[float]]:
    """
    Busca série de fechamentos diários via IBKR para os tickers fornecidos.
    Usado pela análise de rede (MST/RMT) como substituto do yfinance.

    Returns:
        {ticker: [close, close, ...]} — séries alinhadas pelo índice temporal mais curto
    """
    ib = None
    closes: dict[str, list[float]] = {}
    duration = f"{lookback_days} D"

    try:
        import asyncio
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        from ib_insync import IB, util
        util.logToConsole("ERROR")
        ib = IB()
        ib.connect(IBKR_HOST, IBKR_PORT, clientId=client_id,
                   timeout=CONNECT_TIMEOUT, readonly=True)
        if not ib.isConnected():
            _log.warning("ibkr_hist_connect_failed")
            return {}

        _log.info("ibkr_hist_start", tickers=len(tickers), lookback=lookback_days)

        for sym in tickers:
            try:
                contract = _make_contract(sym)
                if contract is None:
                    continue
                ib.qualifyContracts(contract)
                bars = _fetch_historical(ib, contract, duration=duration, bar_size="1 day")
                if len(bars) >= 20:
                    closes[sym] = [float(b.close) for b in bars]
                    _log.debug("ibkr_hist_ok", sym=sym, bars=len(bars))
                else:
                    _log.debug("ibkr_hist_short", sym=sym, bars=len(bars))
                import time as _time
                _time.sleep(REQUEST_DELAY)
            except Exception as exc:
                _log.debug("ibkr_hist_error", sym=sym, error=str(exc))

    except Exception as exc:
        _log.warning("ibkr_hist_failed", error=str(exc))
    finally:
        if ib and ib.isConnected():
            ib.disconnect()

    _log.info("ibkr_hist_done", collected=len(closes), of=len(tickers))
    return closes


# ── Coleta principal ──────────────────────────────────────────────────────────

def collect(
    tickers: dict[str, str] | None = None,
    include_ytd: bool = True,
) -> dict[str, Any]:
    """
    Coleta preços e retornos via IBKR TWS.

    Args:
        tickers  : {symbol: friendly_name}. None = todos do node_registry.
        include_ytd: se True, calcula retorno YTD.

    Returns:
        {ticker: {name, price, daily_return, weekly_return, ytd_return?, ...}}
        Formato idêntico ao market_prices.collect() para drop-in replacement.
    """
    # Monta mapa de tickers
    if tickers is None:
        tickers = _build_ticker_map()
    if not tickers:
        return {}

    ib = _connect_ib()
    if ib is None:
        return {}

    results: dict[str, Any] = {}

    try:
        syms = list(tickers.keys())
        total = len(syms)
        _log.info("ibkr_collect_start", tickers=total)

        for i, sym in enumerate(syms):
            contract = _make_contract(sym)
            if contract is None:
                continue

            # Qualifica contrato (preenche conId, exchange, etc.)
            try:
                ib.qualifyContracts(contract)
            except Exception:
                pass  # continua mesmo sem qualificar

            # Histórico semanal (últimas 6 sessões = daily + weekly)
            bars = _fetch_historical(ib, contract, duration="6 D")
            if len(bars) < 2:
                # Tenta ETF fallback para índices globais
                fallback = _ETF_FALLBACK.get(sym)
                if fallback:
                    from ib_insync import Stock
                    fb_contract = Stock(fallback, "SMART", "USD")
                    try:
                        ib.qualifyContracts(fb_contract)
                    except Exception:
                        pass
                    bars = _fetch_historical(ib, fb_contract, duration="6 D")

            if len(bars) < 2:
                continue

            price      = float(bars[-1].close)
            prev_close = float(bars[-2].close)
            week_start = float(bars[0].close)

            daily  = (price - prev_close) / prev_close if prev_close else 0.0
            weekly = (price - week_start) / week_start if week_start else 0.0

            entry: dict[str, Any] = {
                "name":          tickers[sym],
                "price":         round(price, 4),
                "daily_return":  round(daily, 4),
                "weekly_return": round(weekly, 4),
                "source":        "ibkr",
            }

            # YTD
            if include_ytd:
                ytd_bars = _fetch_ytd_historical(ib, contract)
                if len(ytd_bars) >= 2:
                    ytd_open  = float(ytd_bars[0].close)
                    ytd_close = float(ytd_bars[-1].close)
                    entry["ytd_return"] = round(
                        (ytd_close - ytd_open) / ytd_open if ytd_open else 0.0, 4
                    )

            results[sym] = entry
            _log.debug("ibkr_ticker_ok", sym=sym, price=price,
                       d1=f"{daily:+.2%}")

            # Throttle para não sobrecarregar IB
            if (i + 1) % BATCH_SIZE == 0:
                time.sleep(1.0)
            else:
                time.sleep(REQUEST_DELAY)

    except Exception as exc:
        _log.warning("ibkr_collect_error", error=str(exc))
    finally:
        try:
            ib.disconnect()
            _log.info("ibkr_disconnected", collected=len(results))
        except Exception:
            pass

    return results


# ── Snapshot em tempo real ────────────────────────────────────────────────────

def snapshot(tickers: list[str]) -> dict[str, dict]:
    """
    Snapshot de bid/ask/last/volume em tempo real.
    Requer subscrição de market data ativa no TWS.

    Returns:
        {ticker: {bid, ask, last, volume, open_interest}}
    """
    ib = _connect_ib()
    if ib is None:
        return {}

    results: dict[str, dict] = {}
    try:
        contracts = []
        valid = []
        for sym in tickers:
            c = _make_contract(sym)
            if c:
                contracts.append(c)
                valid.append(sym)

        try:
            ib.qualifyContracts(*contracts)
        except Exception:
            pass

        # Solicita snapshot para todos de uma vez
        ib_tickers = [
            ib.reqMktData(c, "", snapshot=True, regulatorySnapshot=False)
            for c in contracts
        ]
        ib.sleep(2)  # aguarda dados chegarem

        for sym, t in zip(valid, ib_tickers):
            bid  = t.bid  if t.bid  and t.bid  > 0 else None
            ask  = t.ask  if t.ask  and t.ask  > 0 else None
            last = t.last if t.last and t.last > 0 else None
            close = t.close if t.close and t.close > 0 else None
            results[sym] = {
                "bid":           bid,
                "ask":           ask,
                "last":          last or close,
                "volume":        t.volume or None,
                "open_interest": getattr(t, "openInterest", None),
            }
            ib.cancelMktData(t.contract)

    except Exception as exc:
        _log.warning("ibkr_snapshot_error", error=str(exc))
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

    return results


# ── Option chain (scaffold Fase 4) ───────────────────────────────────────────

def get_option_chain(underlying: str) -> dict[str, Any]:
    """
    Busca parâmetros de opções (expirations, strikes) para um underlying.
    Retorna estrutura base para a camada de opções (Fase 4).
    """
    ib = _connect_ib()
    if ib is None:
        return {}

    try:
        contract = _make_contract(underlying)
        ib.qualifyContracts(contract)

        chains = ib.reqSecDefOptParams(
            contract.symbol, "",
            contract.secType,
            contract.conId,
        )
        if not chains:
            return {}

        chain = chains[0]
        return {
            "underlying":   underlying,
            "exchange":     chain.exchange,
            "multiplier":   chain.multiplier,
            "expirations":  sorted(chain.expirations),
            "strikes":      sorted(chain.strikes),
            "n_expirations": len(chain.expirations),
            "n_strikes":    len(chain.strikes),
        }
    except Exception as exc:
        _log.warning("ibkr_options_error", underlying=underlying, error=str(exc))
        return {}
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_ticker_map() -> dict[str, str]:
    """Extrai todos os tickers do node_registry."""
    try:
        from app.desk.node_registry import NODES
        return {
            nd["ticker"]: nd.get("label", nid)
            for nid, nd in NODES.items()
            if nd.get("ticker")
        }
    except Exception:
        from app.providers.market_prices import DEFAULT_TICKERS
        return DEFAULT_TICKERS


def _ensure_event_loop() -> None:
    """Garante que existe um event loop — necessário para eventkit/ib_insync no Python 3.10+."""
    import asyncio
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def is_available() -> bool:
    """Verifica se ib_insync está instalado E TWS acessível na porta."""
    _ensure_event_loop()
    try:
        import asyncio
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        import ib_insync  # noqa: F401
    except (ImportError, Exception):
        return False
    import socket
    try:
        with socket.create_connection((IBKR_HOST, IBKR_PORT), timeout=3):
            return True
    except OSError:
        return False
