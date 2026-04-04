"""
Provider: Market Data Chain — camadas de fallback

Hierarquia para preços correntes (snapshot):
  1. Bloomberg DB         (query_layer — fonte oficial)
  2. IBKR snapshot        (ib_insync TWS/Gateway)
  3. Alpha Vantage        (ALPHA_VANTAGE_API_KEY)
  4. Twelve Data          (TWELVE_DATA_API_KEY)
  5. Finnhub              (FINNHUB_API_KEY)

Hierarquia para histórico OHLCV (MST / network analysis):
  1. Bloomberg CSV DB     (query_layer)
  2. IBKR historical      (ib_insync)
  3. Alpha Vantage        TIME_SERIES_DAILY_ADJUSTED
  4. Twelve Data          /time_series
  5. Finnhub              /stock/candle

Cada camada retorna apenas os tickers que conseguiu — as camadas seguintes
preenchem os tickers que ainda faltam. No final, o chamador recebe um dict
completo com tudo que foi possível coletar, com a chave "source" indicando
de onde cada ticker veio.

Configuração: chaves de API no .env
  ALPHA_VANTAGE_API_KEY=demo
  TWELVE_DATA_API_KEY=xxx
  FINNHUB_API_KEY=yyy
"""

from __future__ import annotations

import time
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("providers.market_data_chain")

# ── Normalização de ticker para cada API ──────────────────────────────────────
# Algumas APIs não aceitam sufixos como =X, =F, ^, -USD

def _to_av(ticker: str) -> str:
    """Alpha Vantage: EURUSD=X → EUR/USD, ^GSPC → ^GSPC (suportado), BTC-USD → BTC"""
    if ticker.endswith("=X") and len(ticker) == 8:   # forex pair AAABBB=X
        return f"{ticker[:3]}/{ticker[3:6]}"
    if ticker.endswith("-USD"):
        return ticker.replace("-USD", "")
    if ticker.endswith("=F"):
        return ticker[:-2]
    return ticker

def _to_td(ticker: str) -> str:
    """Twelve Data: EURUSD=X → EUR/USD, ^GSPC → SPX, etc."""
    _map = {
        "^GSPC": "SPX", "^NDX": "NDX", "^RUT": "RUT", "^VIX": "VIX",
        "^TNX": "TNX", "^TYX": "TYX", "^STOXX50E": "SX5E",
        "^GDAXI": "DAX", "^FCHI": "CAC40", "^FTSE": "FTSE100",
        "^N225": "N225", "^HSI": "HSI",
    }
    if ticker in _map:
        return _map[ticker]
    if ticker.endswith("=X") and len(ticker) == 8:
        return f"{ticker[:3]}/{ticker[3:6]}"
    if ticker.endswith("-USD"):
        return f"{ticker[:-4]}/USD"
    if ticker.endswith("=F"):
        return ticker[:-2]
    return ticker

def _to_fh(ticker: str) -> str | None:
    """Finnhub: só suporta US equities/ETFs e forex. Índices e futuros: None."""
    if ticker.startswith("^") or ticker.endswith("=F") or ticker.endswith(".SS") \
       or ticker.endswith(".HK") or ticker.endswith(".BO"):
        return None
    if ticker.endswith("=X") and len(ticker) == 8:
        return f"OANDA:{ticker[:3]}_{ticker[3:6]}"
    if ticker.endswith("-USD"):
        return f"BINANCE:{ticker[:-4]}USDT"
    return ticker


# ── Camada 3: Alpha Vantage ───────────────────────────────────────────────────

def _av_quote(tickers: list[str], api_key: str) -> dict[str, dict]:
    """Preços correntes via Alpha Vantage GLOBAL_QUOTE. Respeita rate-limit (75 req/min free)."""
    import urllib.request, json as _json

    results: dict[str, dict] = {}
    for raw in tickers:
        sym = _to_av(raw)
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol={sym}&apikey={api_key}"
        )
        try:
            with urllib.request.urlopen(url, timeout=8) as r:
                data = _json.loads(r.read())
            q = data.get("Global Quote", {})
            price = q.get("05. price")
            if not price:
                continue
            prev  = q.get("08. previous close")
            chg   = q.get("10. change percent", "0%").replace("%", "")
            results[raw] = {
                "price":        float(price),
                "daily_return": float(chg) / 100 if chg else None,
                "prev_close":   float(prev) if prev else None,
                "source":       "alpha_vantage",
            }
            time.sleep(0.8)  # free tier: ≤75 req/min
        except Exception as exc:
            _log.debug("av_quote_error", sym=raw, error=str(exc))
    return results


def _av_history(tickers: list[str], api_key: str,
                lookback_days: int = 60) -> dict[str, list[float]]:
    """Histórico diário via Alpha Vantage TIME_SERIES_DAILY_ADJUSTED."""
    import urllib.request, json as _json

    results: dict[str, list[float]] = {}
    for raw in tickers:
        sym = _to_av(raw)
        size = "compact" if lookback_days <= 100 else "full"
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_DAILY_ADJUSTED&symbol={sym}"
            f"&outputsize={size}&apikey={api_key}"
        )
        try:
            with urllib.request.urlopen(url, timeout=12) as r:
                data = _json.loads(r.read())
            series = data.get("Time Series (Daily)", {})
            if not series:
                continue
            closes = [
                float(v["5. adjusted close"])
                for v in list(series.values())[:lookback_days]
            ]
            closes.reverse()  # cronológico
            if len(closes) >= 10:
                results[raw] = closes
            time.sleep(0.8)
        except Exception as exc:
            _log.debug("av_history_error", sym=raw, error=str(exc))
    return results


# ── Camada 4: Twelve Data ─────────────────────────────────────────────────────

def _td_quote(tickers: list[str], api_key: str) -> dict[str, dict]:
    """Preços correntes via Twelve Data /quote (batch de até 120 tickers por call)."""
    import urllib.request, json as _json, urllib.parse

    BATCH = 120
    results: dict[str, dict] = {}
    for i in range(0, len(tickers), BATCH):
        batch_raw = tickers[i: i + BATCH]
        batch_sym = [_to_td(t) for t in batch_raw]
        syms = ",".join(batch_sym)
        url  = (
            f"https://api.twelvedata.com/quote"
            f"?symbol={urllib.parse.quote(syms)}&apikey={api_key}"
        )
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                data = _json.loads(r.read())
            # batch response is a dict keyed by symbol
            if isinstance(data, dict) and "close" in data:
                # single-symbol response wrapped in top-level dict
                data = {batch_sym[0]: data}
            for j, (raw, sym) in enumerate(zip(batch_raw, batch_sym)):
                q = data.get(sym, {})
                price = q.get("close")
                if not price:
                    continue
                chg = q.get("percent_change")
                results[raw] = {
                    "price":        float(price),
                    "daily_return": float(chg) / 100 if chg else None,
                    "source":       "twelve_data",
                }
        except Exception as exc:
            _log.debug("td_quote_error", error=str(exc))
        time.sleep(0.5)
    return results


def _td_history(tickers: list[str], api_key: str,
                lookback_days: int = 60) -> dict[str, list[float]]:
    """Histórico via Twelve Data /time_series."""
    import urllib.request, json as _json

    results: dict[str, list[float]] = {}
    outputsize = min(lookback_days + 5, 5000)
    for raw in tickers:
        sym = _to_td(raw)
        url = (
            f"https://api.twelvedata.com/time_series"
            f"?symbol={sym}&interval=1day"
            f"&outputsize={outputsize}&apikey={api_key}"
        )
        try:
            with urllib.request.urlopen(url, timeout=12) as r:
                data = _json.loads(r.read())
            values = data.get("values", [])
            if not values:
                continue
            closes = [float(v["close"]) for v in reversed(values)]
            if len(closes) >= 10:
                results[raw] = closes[-lookback_days:]
            time.sleep(0.3)
        except Exception as exc:
            _log.debug("td_history_error", sym=raw, error=str(exc))
    return results


# ── Camada 5: Finnhub ─────────────────────────────────────────────────────────

def _fh_quote(tickers: list[str], api_key: str) -> dict[str, dict]:
    """Preços via Finnhub /quote."""
    import urllib.request, json as _json

    results: dict[str, dict] = {}
    for raw in tickers:
        sym = _to_fh(raw)
        if not sym:
            continue
        url = f"https://finnhub.io/api/v1/quote?symbol={sym}&token={api_key}"
        try:
            with urllib.request.urlopen(url, timeout=8) as r:
                q = _json.loads(r.read())
            price = q.get("c")
            prev  = q.get("pc")
            if not price:
                continue
            daily_ret = (price - prev) / prev if prev and prev != 0 else None
            results[raw] = {
                "price":        float(price),
                "daily_return": daily_ret,
                "source":       "finnhub",
            }
            time.sleep(0.05)  # 60 req/min free
        except Exception as exc:
            _log.debug("fh_quote_error", sym=raw, error=str(exc))
    return results


def _fh_history(tickers: list[str], api_key: str,
                lookback_days: int = 60) -> dict[str, list[float]]:
    """Histórico via Finnhub /stock/candle."""
    import urllib.request, json as _json, math

    results: dict[str, list[float]] = {}
    now  = int(time.time())
    from_ts = now - lookback_days * 86400
    for raw in tickers:
        sym = _to_fh(raw)
        if not sym or ":" in sym:   # forex/crypto: Finnhub candles não cobrem
            continue
        url = (
            f"https://finnhub.io/api/v1/stock/candle"
            f"?symbol={sym}&resolution=D&from={from_ts}&to={now}&token={api_key}"
        )
        try:
            with urllib.request.urlopen(url, timeout=12) as r:
                data = _json.loads(r.read())
            if data.get("s") != "ok":
                continue
            closes = [float(v) for v in data.get("c", [])]
            if len(closes) >= 10:
                results[raw] = closes
            time.sleep(0.05)
        except Exception as exc:
            _log.debug("fh_history_error", sym=raw, error=str(exc))
    return results


# ── API pública ───────────────────────────────────────────────────────────────

def collect_prices(
    tickers: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Preços correntes com fallback em camadas:
      Bloomberg DB → IBKR → Alpha Vantage → Twelve Data → Finnhub

    Retorna dict {ticker: {price, daily_return, source, ...}}
    """
    from app.config.settings import settings

    remaining = list(tickers) if tickers else []
    combined: dict[str, dict] = {}

    # ── Camada 1: Bloomberg DB ────────────────────────────────────────────────
    try:
        from app.providers.market_prices import collect as bbg_collect
        bbg = bbg_collect()
        for t, d in bbg.items():
            combined[t] = d
        remaining = [t for t in remaining if t not in combined]
        if bbg:
            _log.info("chain_bbg_prices", got=len(bbg))
    except Exception as exc:
        _log.debug("chain_bbg_failed", error=str(exc))

    if not remaining:
        return combined

    # ── Camada 2: IBKR snapshot ───────────────────────────────────────────────
    try:
        from app.providers.ibkr import collect as ibkr_collect
        # ibkr.collect() aceita {ticker: name} — passa remaining como chaves
        ticker_map = {t: t for t in remaining}
        snap = ibkr_collect(tickers=ticker_map)
        for t, d in snap.items():
            if t not in combined:
                combined[t] = {**d, "source": "ibkr"}
        remaining = [t for t in remaining if t not in combined]
        if snap:
            _log.info("chain_ibkr_prices", got=len(snap))
    except Exception as exc:
        _log.debug("chain_ibkr_failed", error=str(exc))

    if not remaining:
        return combined

    # ── Camada 3: Alpha Vantage ───────────────────────────────────────────────
    av_key = settings.alpha_vantage_api_key
    if av_key:
        try:
            av = _av_quote(remaining, av_key)
            for t, d in av.items():
                if t not in combined:
                    combined[t] = d
            remaining = [t for t in remaining if t not in combined]
            if av:
                _log.info("chain_av_prices", got=len(av))
        except Exception as exc:
            _log.debug("chain_av_failed", error=str(exc))

    if not remaining:
        return combined

    # ── Camada 4: Twelve Data ─────────────────────────────────────────────────
    td_key = settings.twelve_data_api_key
    if td_key:
        try:
            td = _td_quote(remaining, td_key)
            for t, d in td.items():
                if t not in combined:
                    combined[t] = d
            remaining = [t for t in remaining if t not in combined]
            if td:
                _log.info("chain_td_prices", got=len(td))
        except Exception as exc:
            _log.debug("chain_td_failed", error=str(exc))

    if not remaining:
        return combined

    # ── Camada 5: Finnhub ─────────────────────────────────────────────────────
    fh_key = settings.finnhub_api_key
    if fh_key:
        try:
            fh = _fh_quote(remaining, fh_key)
            for t, d in fh.items():
                if t not in combined:
                    combined[t] = d
            remaining = [t for t in remaining if t not in combined]
            if fh:
                _log.info("chain_fh_prices", got=len(fh))
        except Exception as exc:
            _log.debug("chain_fh_failed", error=str(exc))

    if remaining:
        _log.warning("chain_prices_missing", tickers=remaining)

    _log.info("chain_prices_total", total=len(combined))
    return combined


def collect_historical(
    tickers: list[str],
    lookback_days: int = 60,
) -> dict[str, list[float]]:
    """
    Histórico de fechamentos diários (cronológico) com fallback em camadas:
      Bloomberg CSV → IBKR → Alpha Vantage → Twelve Data → Finnhub

    Retorna {ticker: [close_t0, close_t1, ..., close_tN]}  (mais antigo → mais recente)
    Cada série tem pelo menos 10 observações.
    """
    from app.config.settings import settings

    closes: dict[str, list[float]] = {}
    remaining = list(tickers)

    # ── Camada 1: Bloomberg CSV (BQuant export) ───────────────────────────────
    try:
        from app.providers.bql_csv import load_price_history
        bbg_hist = load_price_history()
        if bbg_hist:
            MIN_OBS = max(20, lookback_days // 4)
            for t, series in bbg_hist.items():
                if t in remaining and len(series) >= MIN_OBS:
                    closes[t] = series[-lookback_days:]
            remaining = [t for t in remaining if t not in closes]
            if closes:
                _log.info("chain_bbg_hist", got=len(closes))
    except Exception as exc:
        _log.debug("chain_bbg_hist_failed", error=str(exc))

    if not remaining:
        return closes

    # ── Camada 2: IBKR historical ─────────────────────────────────────────────
    try:
        from app.providers.ibkr import fetch_historical_closes
        ibkr_hist = fetch_historical_closes(remaining, lookback_days=lookback_days)
        for t, series in ibkr_hist.items():
            if t not in closes and len(series) >= 10:
                closes[t] = series
        remaining = [t for t in remaining if t not in closes]
        if ibkr_hist:
            _log.info("chain_ibkr_hist", got=len(ibkr_hist))
    except Exception as exc:
        _log.debug("chain_ibkr_hist_failed", error=str(exc))

    if not remaining:
        return closes

    # ── Camada 3: Alpha Vantage ───────────────────────────────────────────────
    av_key = settings.alpha_vantage_api_key
    if av_key:
        try:
            av = _av_history(remaining, av_key, lookback_days)
            for t, series in av.items():
                if t not in closes:
                    closes[t] = series
            remaining = [t for t in remaining if t not in closes]
            if av:
                _log.info("chain_av_hist", got=len(av))
        except Exception as exc:
            _log.debug("chain_av_hist_failed", error=str(exc))

    if not remaining:
        return closes

    # ── Camada 4: Twelve Data ─────────────────────────────────────────────────
    td_key = settings.twelve_data_api_key
    if td_key:
        try:
            td = _td_history(remaining, td_key, lookback_days)
            for t, series in td.items():
                if t not in closes:
                    closes[t] = series
            remaining = [t for t in remaining if t not in closes]
            if td:
                _log.info("chain_td_hist", got=len(td))
        except Exception as exc:
            _log.debug("chain_td_hist_failed", error=str(exc))

    if not remaining:
        return closes

    # ── Camada 5: Finnhub ─────────────────────────────────────────────────────
    fh_key = settings.finnhub_api_key
    if fh_key:
        try:
            fh = _fh_history(remaining, fh_key, lookback_days)
            for t, series in fh.items():
                if t not in closes:
                    closes[t] = series
            remaining = [t for t in remaining if t not in closes]
            if fh:
                _log.info("chain_fh_hist", got=len(fh))
        except Exception as exc:
            _log.debug("chain_fh_hist_failed", error=str(exc))

    if remaining:
        _log.warning("chain_hist_missing", tickers=remaining[:10], total_missing=len(remaining))

    _log.info("chain_hist_total", tickers=len(closes))
    return closes
