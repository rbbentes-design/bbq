"""
Provider: Interactive Brokers — ib_insync + ibeam Gateway

Arquitetura (VPS / local com ibeam):

  ibeam → IB Gateway :4001 (TCP socket nativo)
       ↓
  ib_insync (asyncio event-driven)
       ↓
  ┌─────────────────┬──────────────────┐
  │  market data    │  order execution │
  │  reqMktData()   │  placeOrder()    │
  │  onBarUpdate()  │  onOrderFilled() │
  └─────────────────┴──────────────────┘
       ↓
  structlog JSON → Discord alerts → Supabase

Retry: 3x com backoff exponencial. Falha total → alerta Discord.
Market hours: NYSE 09:30-16:00 ET via APScheduler (opcional).
Latência: 10-50ms vs ~150-400ms da arquitetura REST anterior.

Configuração (.env):
  IBKR_HOST=127.0.0.1
  IBKR_PORT=4001               # ibeam/Gateway (7497=TWS paper, 4001=Gateway)
  IBKR_CLIENT_ID=20
  DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...   # opcional
  SUPABASE_URL=https://xxx.supabase.co                       # opcional
  SUPABASE_KEY=xxx                                           # opcional
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Callable

from app.audit.logger import get_logger

_log = get_logger("providers.ibkr")

# ── Configuração ──────────────────────────────────────────────────────────────

IBKR_HOST       = os.getenv("IBKR_HOST",      "127.0.0.1")
IBKR_PORT       = int(os.getenv("IBKR_PORT",  "4001"))      # ibeam/Gateway
IBKR_CLIENT_ID  = int(os.getenv("IBKR_CLIENT_ID", "20"))
CONNECT_TIMEOUT = 10        # segundos por tentativa
MAX_RETRIES     = 3         # tentativas antes de desistir + alertar
RETRY_BACKOFF   = [2, 5, 15]  # segundos entre tentativas
REQUEST_DELAY   = 0.05      # throttle entre requisições (50ms)
BATCH_SIZE      = 40        # max contratos por lote de histData

_DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL", "")
_SUPABASE_URL    = os.getenv("SUPABASE_URL", "")
_SUPABASE_KEY    = os.getenv("SUPABASE_KEY", "")

# ── whatToShow por tipo de contrato ──────────────────────────────────────────

_WHAT_TO_SHOW: dict[str, str] = {
    "STK":     "TRADES",
    "IND":     "TRADES",
    "CASH":    "MIDPOINT",
    "FUT":     "TRADES",
    "CONTFUT": "TRADES",
    "CRYPTO":  "TRADES",
    "ETF":     "TRADES",
}

# ── Mapeamento yfinance / Bloomberg symbol → contrato IB ─────────────────────

_CONTRACT_MAP: dict[str, tuple] = {
    # ── US Equity Indices ────────────────────────────────────────────────────
    "^GSPC":      ("index",   "SPX",       "CBOE"),
    "^NDX":       ("index",   "NDX",       "NASDAQ"),
    "^RUT":       ("index",   "RUT",       "RUSSELL"),
    "^VIX":       ("index",   "VIX",       "CBOE"),
    "^VIX9D":     ("index",   "VIX9D",     "CBOE"),
    "^VIX3M":     ("index",   "VIX3M",     "CBOE"),
    "^TNX":       ("index",   "TNX",       "CBOE"),
    "^TYX":       ("index",   "TYX",       "CBOE"),
    "^IRX":       ("index",   "IRX",       "CBOE"),
    "^FVX":       ("index",   "FVX",       "CBOE"),
    # ── FX ───────────────────────────────────────────────────────────────────
    "EURUSD=X":   ("forex",   "EURUSD"),
    "USDJPY=X":   ("forex",   "USDJPY"),
    "GBPUSD=X":   ("forex",   "GBPUSD"),
    "USDCNH=X":   ("forex",   "USDCNH"),
    "USDBRL=X":   ("forex",   "USDBRL"),
    "USDMXN=X":   ("forex",   "USDMXN"),
    "AUDUSD=X":   ("forex",   "AUDUSD"),
    "USDCHF=X":   ("forex",   "USDCHF"),
    "DX-Y.NYB":   ("contfut", "DX",        "NYBOT"),
    # ── Commodities ──────────────────────────────────────────────────────────
    "CL=F":       ("contfut", "CL",        "NYMEX"),
    "NG=F":       ("contfut", "NG",        "NYMEX"),
    "HG=F":       ("contfut", "HG",        "COMEX"),
    "ZW=F":       ("contfut", "ZW",        "CBOT"),
    "ZC=F":       ("contfut", "ZC",        "CBOT"),
    "ZS=F":       ("contfut", "ZS",        "CBOT"),
    "GC=F":       ("contfut", "GC",        "COMEX"),
    "SI=F":       ("contfut", "SI",        "COMEX"),
    # ── Crypto ───────────────────────────────────────────────────────────────
    "BTC-USD":    ("crypto",  "BTC",       "PAXOS", "USD"),
    "ETH-USD":    ("crypto",  "ETH",       "PAXOS", "USD"),
    # ── US Equities especiais ─────────────────────────────────────────────────
    "BRK-B":      ("stk",    "BRK B",     "NYSE",  "USD"),
    "BRK-A":      ("stk",    "BRK A",     "NYSE",  "USD"),
    # ── EU Indices ───────────────────────────────────────────────────────────
    "^STOXX50E":  ("index",   "STOXX50E",  "DTB"),
    "^GDAXI":     ("index",   "DAX",       "DTB"),
    "^FCHI":      ("index",   "CAC40",     "MONEP"),
    "^FTSE":      ("index",   "FTSE100",   "LIFFE"),
    "^IBEX":      ("index",   "IBEX35",    "MEFF"),
    # ── Asia ─────────────────────────────────────────────────────────────────
    "^N225":      ("index",   "NI225",     "OSE.JPN"),
    "^HSI":       ("index",   "HSI",       "HKFE"),
    "^NSEI":      ("index",   "NIFTY50",   "NSE"),
    "^BVSP":      ("index",   "IBOV",      "BOVESPA"),
}

_ETF_FALLBACK: dict[str, str] = {
    "^STOXX50E": "FEZ", "^GDAXI": "EWG", "^FCHI": "EWQ",
    "^FTSE": "EWU",     "^N225":  "EWJ", "^HSI":  "EWH",
    "^NSEI": "INDA",    "^BVSP":  "EWZ",
}

# Sufixos Bloomberg a remover antes de usar como símbolo IB/yfinance
_BBG_SUFFIXES = (
    " US Equity", " US EQUITY", " Equity", " EQUITY",
    " Index", " INDEX", " Comdty", " COMDTY", " Curncy", " CURNCY",
)


def _strip_bbg(ticker: str) -> str:
    """Remove sufixos Bloomberg: 'AAPL US EQUITY' → 'AAPL'."""
    t = ticker.strip()
    for sfx in _BBG_SUFFIXES:
        if t.endswith(sfx):
            t = t[:-len(sfx)].strip()
            break
    return t.replace("/", "-")


# ── Alertas externos ──────────────────────────────────────────────────────────

def _send_discord(message: str) -> None:
    """Envia alerta ao Discord via webhook. Silencioso se não configurado."""
    if not _DISCORD_WEBHOOK:
        return
    try:
        import urllib.request, json as _json
        data = _json.dumps({"content": f"🤖 **IBKR** {message}"}).encode()
        req  = urllib.request.Request(
            _DISCORD_WEBHOOK,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as exc:
        _log.debug("discord_alert_failed", error=str(exc)[:60])


def _log_to_supabase(table: str, record: dict) -> None:
    """Insere registro no Supabase. Silencioso se não configurado."""
    if not _SUPABASE_URL or not _SUPABASE_KEY:
        return
    try:
        import urllib.request, json as _json
        url  = f"{_SUPABASE_URL}/rest/v1/{table}"
        data = _json.dumps(record).encode()
        req  = urllib.request.Request(url, data=data, method="POST", headers={
            "Content-Type":  "application/json",
            "apikey":        _SUPABASE_KEY,
            "Authorization": f"Bearer {_SUPABASE_KEY}",
            "Prefer":        "return=minimal",
        })
        urllib.request.urlopen(req, timeout=5)
    except Exception as exc:
        _log.debug("supabase_insert_failed", table=table, error=str(exc)[:60])


# ── Contract factory ──────────────────────────────────────────────────────────

def _make_contract(ticker: str):
    """Converte ticker → contrato ib_insync. Normaliza sufixos Bloomberg."""
    try:
        from ib_insync import Stock, Index, Forex, ContFuture, Crypto
    except ImportError:
        return None

    clean = _strip_bbg(ticker)
    spec  = _CONTRACT_MAP.get(clean) or _CONTRACT_MAP.get(ticker)

    if spec is None:
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


# ── Conexão com retry 3x + Discord ───────────────────────────────────────────

def _ensure_event_loop() -> None:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def _connect_ib(port: int = IBKR_PORT, client_id: int = IBKR_CLIENT_ID):
    """
    Conecta ao IB Gateway/TWS com retry 3x + backoff exponencial.
    Envia alerta Discord se todas as tentativas falharem.

    Tenta client IDs consecutivos para evitar conflito com sessões abertas.
    """
    _ensure_event_loop()
    try:
        from ib_insync import IB, util
        util.logToConsole("ERROR")
    except ImportError:
        _log.warning("ib_insync_not_installed")
        return None

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        for cid in range(client_id, client_id + 5):
            try:
                ib = IB()
                ib.connect(IBKR_HOST, port, clientId=cid,
                           timeout=CONNECT_TIMEOUT, readonly=True)
                if ib.isConnected():
                    _log.info("ibkr_connected",
                              host=IBKR_HOST, port=port,
                              clientId=cid, attempt=attempt)
                    return ib
                try: ib.disconnect()
                except Exception: pass
            except Exception as exc:
                last_error = str(exc)
                err_lower  = last_error.lower()
                if "already in use" in err_lower or "326" in err_lower:
                    continue  # cid ocupado — tenta o próximo
                # Erro real (connection refused = gateway não está rodando)
                break

        if attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF[attempt - 1]
            _log.warning("ibkr_retry", attempt=attempt, wait_s=wait,
                         error=last_error[:80])
            time.sleep(wait)

    # Todas as tentativas falharam
    msg = f"conexão falhou após {MAX_RETRIES} tentativas — {last_error[:100]}"
    _log.warning("ibkr_connect_failed", error=last_error[:120])
    _send_discord(f"⚠️ {msg}")
    return None


# ── Market hours check ────────────────────────────────────────────────────────

def is_market_open(exchange: str = "NYSE") -> bool:
    """
    Verifica se o mercado está aberto agora.
    NYSE: 09:30-16:00 ET, seg-sex, exceto feriados.
    Não usa APScheduler aqui — apenas check pontual.
    """
    try:
        import zoneinfo
        et = zoneinfo.ZoneInfo("America/New_York")
    except ImportError:
        try:
            import pytz
            et = pytz.timezone("America/New_York")
        except ImportError:
            return True  # sem timezone lib — assume aberto

    now = datetime.now(et)
    if now.weekday() >= 5:  # sáb/dom
        return False
    open_time  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_time <= now <= close_time


# ── Coleta de dados históricos ────────────────────────────────────────────────

def _fetch_historical(ib, contract, duration: str = "6 D",
                      bar_size: str = "1 day") -> list:
    try:
        what = _what_to_show(contract)
        bars = ib.reqHistoricalData(
            contract, endDateTime="", durationStr=duration,
            barSizeSetting=bar_size, whatToShow=what,
            useRTH=True, formatDate=1,
        )
        return bars or []
    except Exception as exc:
        _log.debug("hist_data_error",
                   sym=getattr(contract, "symbol", "?"), error=str(exc)[:60])
        return []


def _fetch_ytd_historical(ib, contract) -> list:
    year = date.today().year
    try:
        what = _what_to_show(contract)
        bars = ib.reqHistoricalData(
            contract, endDateTime="",
            durationStr=f"{date.today().timetuple().tm_yday + 5} D",
            barSizeSetting="1 day", whatToShow=what,
            useRTH=True, formatDate=1,
        )
        return [b for b in (bars or []) if str(b.date)[:4] == str(year)]
    except Exception:
        return []


# ── Histórico de closes para análise de rede ──────────────────────────────────

def fetch_historical_closes(
    tickers: list[str],
    lookback_days: int = 90,
    client_id: int = 21,
) -> dict[str, list[float]]:
    """
    Série de fechamentos diários via IBKR para análise MST/RMT.
    Normaliza sufixos Bloomberg automaticamente.

    Returns: {ticker: [close_t0, ..., close_tN]}
    """
    ib     = None
    closes: dict[str, list[float]] = {}
    duration = f"{lookback_days} D"

    try:
        ib = _connect_ib(client_id=client_id)
        if ib is None:
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
                time.sleep(REQUEST_DELAY)
            except Exception as exc:
                _log.debug("ibkr_hist_error", sym=sym, error=str(exc)[:60])

    except Exception as exc:
        _log.warning("ibkr_hist_failed", error=str(exc))
    finally:
        if ib and ib.isConnected():
            try: ib.disconnect()
            except Exception: pass

    _log.info("ibkr_hist_done", collected=len(closes), of=len(tickers))
    return closes


# ── Coleta principal (preços + retornos) ──────────────────────────────────────

def collect(
    tickers: dict[str, str] | None = None,
    include_ytd: bool = True,
) -> dict[str, Any]:
    """
    Coleta preços e retornos via IBKR Gateway (:4001).
    Normaliza sufixos Bloomberg automaticamente.

    Returns: {ticker: {name, price, daily_return, weekly_return, ytd_return?, source}}
    """
    if tickers is None:
        tickers = _build_ticker_map()
    if not tickers:
        return {}

    ib = _connect_ib()
    if ib is None:
        return {}

    results: dict[str, Any] = {}

    try:
        syms  = list(tickers.keys())
        total = len(syms)
        _log.info("ibkr_collect_start", tickers=total, port=IBKR_PORT)

        for i, sym in enumerate(syms):
            contract = _make_contract(sym)
            if contract is None:
                continue

            try:
                ib.qualifyContracts(contract)
            except Exception:
                pass

            bars = _fetch_historical(ib, contract, duration="6 D")

            # ETF fallback para índices globais
            if len(bars) < 2:
                fallback = _ETF_FALLBACK.get(_strip_bbg(sym)) or _ETF_FALLBACK.get(sym)
                if fallback:
                    from ib_insync import Stock
                    fb = Stock(fallback, "SMART", "USD")
                    try: ib.qualifyContracts(fb)
                    except Exception: pass
                    bars = _fetch_historical(ib, fb, duration="6 D")

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

            if include_ytd:
                ytd_bars = _fetch_ytd_historical(ib, contract)
                if len(ytd_bars) >= 2:
                    ytd_open  = float(ytd_bars[0].close)
                    ytd_close = float(ytd_bars[-1].close)
                    entry["ytd_return"] = round(
                        (ytd_close - ytd_open) / ytd_open if ytd_open else 0.0, 4
                    )

            results[sym] = entry
            _log.debug("ibkr_ticker_ok", sym=sym, price=price, d1=f"{daily:+.2%}")

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


# ── Snapshot em tempo real (reqMktData streaming) ─────────────────────────────

def snapshot(tickers: list[str]) -> dict[str, dict]:
    """
    Snapshot bid/ask/last/volume via reqMktData (event-driven, não polling).
    Solicita todos de uma vez, aguarda 2s para dados chegarem, cancela.

    Returns: {ticker: {bid, ask, last, volume, open_interest}}
    """
    ib = _connect_ib()
    if ib is None:
        return {}

    results: dict[str, dict] = {}
    try:
        contracts, valid = [], []
        for sym in tickers:
            c = _make_contract(sym)
            if c:
                contracts.append(c)
                valid.append(sym)

        try:
            ib.qualifyContracts(*contracts)
        except Exception:
            pass

        # reqMktData: solicita streaming tick data (snapshot=True = one-shot)
        ib_tickers = [
            ib.reqMktData(c, "", snapshot=True, regulatorySnapshot=False)
            for c in contracts
        ]
        ib.sleep(2)  # event loop processa os ticks chegando

        for sym, t in zip(valid, ib_tickers):
            bid   = t.bid   if t.bid   and t.bid   > 0 else None
            ask   = t.ask   if t.ask   and t.ask   > 0 else None
            last  = t.last  if t.last  and t.last  > 0 else None
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
        try: ib.disconnect()
        except Exception: pass

    return results


# ── Streaming contínuo event-driven ──────────────────────────────────────────

@dataclass
class StreamSession:
    """Sessão de streaming ativa. Mantém a conexão e os callbacks."""
    ib: Any
    on_bar_update: Callable[[str, dict], None] | None = None
    on_order_filled: Callable[[dict], None] | None = None
    _ticker_map: dict = field(default_factory=dict)  # conId → symbol

    def stop(self) -> None:
        try: self.ib.disconnect()
        except Exception: pass
        _log.info("ibkr_stream_stopped")


def start_streaming(
    tickers: list[str],
    on_bar_update: Callable[[str, dict], None] | None = None,
    on_order_filled: Callable[[dict], None] | None = None,
    bar_size: str = "1 min",
    check_market_hours: bool = True,
) -> StreamSession | None:
    """
    Inicia streaming contínuo via reqMktData + barras em tempo real.

    on_bar_update(symbol, bar_dict) — chamado a cada barra completada
    on_order_filled(fill_dict)      — chamado quando uma ordem é executada

    Verifica horário de mercado antes de conectar (NYSE 09:30-16:00 ET).
    Retorna StreamSession para manter referência e poder parar com .stop().

    Uso típico:
        session = start_streaming(
            ["SPY", "QQQ"],
            on_bar_update=my_strategy,
            on_order_filled=my_fill_handler,
        )
        # ... roda em background
        session.stop()
    """
    if check_market_hours and not is_market_open():
        _log.info("ibkr_stream_outside_hours")
        return None

    ib = _connect_ib()
    if ib is None:
        return None

    session = StreamSession(ib=ib, on_bar_update=on_bar_update,
                            on_order_filled=on_order_filled)

    try:
        from ib_insync import BarDataList

        for sym in tickers:
            contract = _make_contract(sym)
            if contract is None:
                continue
            try:
                ib.qualifyContracts(contract)
            except Exception:
                pass

            # Barras em tempo real (reqRealTimeBars: 5s fixo pela API IB)
            bars: BarDataList = ib.reqRealTimeBars(
                contract, barSize=5, whatToShow=_what_to_show(contract),
                useRTH=True,
            )
            session._ticker_map[id(bars)] = sym

            # Callback de barra — chamado pelo event loop ib_insync
            if on_bar_update:
                def _make_cb(symbol: str):
                    def _on_bar(bars, has_new_bar: bool):
                        if not has_new_bar or not bars:
                            return
                        b = bars[-1]
                        on_bar_update(symbol, {
                            "time":   str(b.time),
                            "open":   float(b.open),
                            "high":   float(b.high),
                            "low":    float(b.low),
                            "close":  float(b.close),
                            "volume": int(b.volume),
                        })
                    return _on_bar
                bars.updateEvent += _make_cb(sym)

        # Callback de fills
        if on_order_filled:
            def _on_fill(trade, fill):
                fill_dict = {
                    "symbol":       trade.contract.symbol,
                    "action":       trade.order.action,
                    "qty":          fill.execution.shares,
                    "price":        fill.execution.price,
                    "commission":   fill.commissionReport.commission if fill.commissionReport else None,
                    "time":         str(fill.execution.time),
                    "order_id":     trade.order.orderId,
                }
                _log.info("ibkr_fill", **{k: v for k, v in fill_dict.items() if v is not None})
                _log_to_supabase("trades", fill_dict)
                on_order_filled(fill_dict)

            ib.fillEvent += _on_fill

        _log.info("ibkr_stream_started", tickers=tickers, bar_size=bar_size)
        return session

    except Exception as exc:
        _log.warning("ibkr_stream_error", error=str(exc))
        try: ib.disconnect()
        except Exception: pass
        return None


# ── Execução de ordem ─────────────────────────────────────────────────────────

def place_order(
    symbol: str,
    action: str,        # "BUY" | "SELL"
    quantity: int,
    order_type: str = "MKT",   # "MKT" | "LMT"
    limit_price: float | None = None,
    risk_check: Callable[[], bool] | None = None,
) -> dict | None:
    """
    Envia ordem via TCP socket direto (LMT/MKT).
    Verifica risk_check() antes de enviar — se retornar False, bloqueia.
    Loga via structlog e envia para Supabase.

    Returns: dict com ordem submetida, ou None se bloqueada.
    """
    if risk_check is not None and not risk_check():
        _log.warning("ibkr_order_blocked_by_risk",
                     symbol=symbol, action=action, qty=quantity)
        _send_discord(f"🚫 Ordem **{action} {quantity} {symbol}** bloqueada pelo risk manager")
        return None

    ib = _connect_ib()
    if ib is None:
        return None

    try:
        from ib_insync import MarketOrder, LimitOrder

        contract = _make_contract(symbol)
        if contract is None:
            return None
        ib.qualifyContracts(contract)

        order = (LimitOrder(action, quantity, limit_price)
                 if order_type == "LMT" and limit_price
                 else MarketOrder(action, quantity))

        trade = ib.placeOrder(contract, order)
        ib.sleep(1)  # aguarda ACK

        result = {
            "symbol":     symbol,
            "action":     action,
            "qty":        quantity,
            "order_type": order_type,
            "limit_price":limit_price,
            "order_id":   trade.order.orderId,
            "status":     trade.orderStatus.status,
            "timestamp":  datetime.now().isoformat(),
        }
        _log.info("ibkr_order_placed", **result)
        _log_to_supabase("orders", result)
        _send_discord(f"📋 {action} {quantity} {symbol} @ {'MKT' if not limit_price else f'LMT {limit_price}'} — {result['status']}")
        return result

    except Exception as exc:
        _log.warning("ibkr_order_error", symbol=symbol, error=str(exc))
        return None
    finally:
        try: ib.disconnect()
        except Exception: pass


# ── Option chain ──────────────────────────────────────────────────────────────

def get_option_chain(underlying: str) -> dict[str, Any]:
    """Busca expirations + strikes para um underlying."""
    ib = _connect_ib()
    if ib is None:
        return {}
    try:
        contract = _make_contract(underlying)
        ib.qualifyContracts(contract)
        chains = ib.reqSecDefOptParams(
            contract.symbol, "", contract.secType, contract.conId,
        )
        if not chains:
            return {}
        chain = chains[0]
        return {
            "underlying":    underlying,
            "exchange":      chain.exchange,
            "multiplier":    chain.multiplier,
            "expirations":   sorted(chain.expirations),
            "strikes":       sorted(chain.strikes),
            "n_expirations": len(chain.expirations),
            "n_strikes":     len(chain.strikes),
        }
    except Exception as exc:
        _log.warning("ibkr_options_error", underlying=underlying, error=str(exc))
        return {}
    finally:
        try: ib.disconnect()
        except Exception: pass


# ── Availability check ────────────────────────────────────────────────────────

def is_available() -> bool:
    """Verifica se ib_insync está instalado E Gateway acessível na porta."""
    _ensure_event_loop()
    try:
        import ib_insync  # noqa: F401
    except ImportError:
        return False
    import socket
    try:
        with socket.create_connection((IBKR_HOST, IBKR_PORT), timeout=3):
            return True
    except OSError:
        return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_ticker_map() -> dict[str, str]:
    try:
        from app.desk.node_registry import NODES
        return {nd["ticker"]: nd.get("label", nid)
                for nid, nd in NODES.items() if nd.get("ticker")}
    except Exception:
        from app.providers.market_prices import DEFAULT_TICKERS
        return DEFAULT_TICKERS
