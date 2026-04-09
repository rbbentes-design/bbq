"""
TradingView MCP provider.

Calls the tradingview-mcp CLI (`tv` command) to extract technical indicators
from TradingView Desktop: VWAP, Value Area (VAH/VAL/POC), RSI, ATR, anchored VWAPs.

Requirements:
  - tradingview-mcp installed: git clone → npm install → npm link
  - TradingView Desktop running with: --remote-debugging-port=9222
  - Layout "ultimate profile" (or equivalent) saved in TradingView

All functions are resilient: return {} / [] if TV is not running or CLI unavailable.
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("providers.tradingview")

# ── CLI discovery ──────────────────────────────────────────────────────────────

# Try candidates in order; first one that works is used
_TV_CANDIDATES = ["tv", "tv.cmd", "npx tradingview-mcp"]
_TV_CMD: str | None = None
_SWITCH_DELAY = 2.5   # seconds after chart_set_symbol / chart_set_timeframe
_TOOL_TIMEOUT = 4     # seconds per tv CLI call (era 12 - perdia tempo demais)

# ── Indicator name aliases (matched to "ultimate profile" layout) ──────────────
# tv values returns: {"studies": [{"name": "...", "values": {"Key": "123.45"}}]}
# We flatten to a single dict and match by key name.

_VWAP_KEYS   = {
    "VWAP - Weekly", "VWAP - Daily", "VWAP - Monthly",
    "VWAP", "vwap", "Volume Weighted Average Price",
    "Previous VWAP - Daily",   # fallback: yesterday's session VWAP
}
_VAH_KEYS    = {
    "Value Area High", "Developing Value Area High",
    "VAH", "VA High",
}
_VAL_KEYS    = {
    "Value Area Low", "Developing Value Area Low",
    "VAL", "VA Low",
}
_POC_KEYS    = {
    "Point of Control", "Developing Point of Control",
    "POC", "POC Price",
}
_RSI_KEYS    = {"RSI", "RSI(14)", "Relative Strength Index", "RSI (14)"}
_ATR_KEYS    = {"ATR", "ATR(14)", "Average True Range", "ATR (14)"}
_MACD_KEYS   = {"MACD", "MACD Line", "MACD (12, 26, 9)"}
_SIGNAL_KEYS = {"Signal", "MACD Signal", "Signal Line"}

# LuxAlgo Oscillator Matrix
_HYPERWAVE_KEYS   = {"HyperWave"}
_MONEY_FLOW_KEYS  = {"Money Flow"}
_VENOM_SUP_KEYS   = {"Next Venom Support"}
_VENOM_RES_KEYS   = {"Next Venom Resistance"}


def _parse_float(s: str | float | None) -> float | None:
    """Parse a string like '369.37' or '−2.49' to float. Returns None on failure."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s) if s == s else None
    try:
        # Replace Unicode minus sign with ASCII minus
        cleaned = str(s).replace("\u2212", "-").replace(",", "").strip()
        v = float(cleaned)
        return v if v == v else None
    except (ValueError, TypeError):
        return None


def _flatten_studies(raw: dict | list) -> dict[str, float]:
    """
    Flatten the nested studies structure from `tv values` into a single dict.

    Input: {"studies": [{"name": "RSI", "values": {"RSI": "54.35", ...}}, ...]}
    Output: {"RSI": 54.35, "Value Area High": 383.19, ...}
    """
    flat: dict[str, float] = {}

    # Handle both raw response dict and already-extracted studies list
    if isinstance(raw, dict):
        studies = raw.get("studies") or raw.get("data", {}).get("studies") or []
        # Also try flat values at top level
        for k, v in raw.items():
            if k not in ("success", "studies", "study_count", "data"):
                parsed = _parse_float(v)
                if parsed is not None:
                    flat[k] = parsed
    elif isinstance(raw, list):
        studies = raw
    else:
        return flat

    for study in studies:
        if not isinstance(study, dict):
            continue
        values = study.get("values", {})
        if not isinstance(values, dict):
            continue
        for k, v in values.items():
            parsed = _parse_float(v)
            if parsed is not None:
                flat[k] = parsed

    return flat


def _find_value(data: dict[str, float], keys: set[str]) -> float | None:
    """Return first matching float value from a flattened indicator dict."""
    for k in keys:
        if k in data:
            v = data[k]
            if v == v:  # NaN guard
                return v
    return None


# ── Subprocess runner ──────────────────────────────────────────────────────────

def _resolve_tv_cmd() -> str | None:
    """Find the first working `tv` CLI candidate."""
    global _TV_CMD
    if _TV_CMD is not None:
        return _TV_CMD

    for candidate in _TV_CANDIDATES:
        try:
            parts = candidate.split()
            result = subprocess.run(
                parts + ["--help"],
                capture_output=True, text=True, timeout=8,
            )
            # --help exits 0 and prints usage
            if "Usage:" in result.stdout or "Commands:" in result.stdout:
                _TV_CMD = candidate
                _log.info("tv_cmd_found", cmd=candidate)
                return _TV_CMD
        except Exception:
            continue

    _log.warning("tv_cmd_not_found", tried=_TV_CANDIDATES)
    return None


def _run_tv(*args: str) -> dict[str, Any]:
    """
    Run `tv <command> [subcommand] [options]` and return parsed JSON.

    CLI command mapping (tradingview-mcp v2 syntax):
      status                → tv status
      symbol <SYM>          → tv symbol <SYM>
      timeframe <TF>        → tv timeframe <TF>
      values                → tv values
      data lines            → tv data lines
      data boxes            → tv data boxes
      quote                 → tv quote
      layout switch <name>  → tv layout switch <name>
      layout list           → tv layout list

    Returns {"success": False, "error": "..."} on any failure.
    """
    cmd_str = _resolve_tv_cmd()
    if cmd_str is None:
        return {"success": False, "error": "tv CLI not found"}

    try:
        full_cmd = cmd_str.split() + list(args)
        result = subprocess.run(
            full_cmd,
            capture_output=True, text=True, timeout=_TOOL_TIMEOUT,
        )
        raw = result.stdout.strip()
        if not raw:
            # Some commands write to stderr
            raw = result.stderr.strip()
        if not raw:
            return {"success": False, "error": "empty response"}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # CLI may output non-JSON on success for some commands
            return {"success": True, "data": raw}
        # Normalize: if there's no "success" key, treat non-empty data as success
        if "success" not in parsed:
            parsed["success"] = True
        return parsed
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Chart navigation ───────────────────────────────────────────────────────────

def _to_tv_symbol(ticker: str) -> str:
    """Convert agente ticker format to TradingView symbol."""
    # Strip Bloomberg suffixes
    for suffix in (" US Equity", " Index", " Comdty", " Curncy"):
        if ticker.endswith(suffix):
            ticker = ticker[: -len(suffix)]
    # Yahoo Finance special symbols
    ticker = ticker.replace("^GSPC", "SPX").replace("^NDX", "NDX").replace("^VIX", "VIX")
    return ticker.strip()


def _switch_to(ticker: str, timeframe: str = "D") -> bool:
    """Switch TradingView to a ticker/timeframe. Returns True on success."""
    tv_symbol = _to_tv_symbol(ticker)
    r1 = _run_tv("symbol", tv_symbol)
    if not r1.get("success"):
        _log.debug("tv_set_symbol_failed", ticker=ticker, error=r1.get("error"))
        return False
    time.sleep(_SWITCH_DELAY)
    r2 = _run_tv("timeframe", timeframe)
    if not r2.get("success"):
        _log.debug("tv_set_timeframe_failed", ticker=ticker, tf=timeframe, error=r2.get("error"))
    time.sleep(1.0)
    return True


# ── Data reading ───────────────────────────────────────────────────────────────

def _read_study_values() -> dict[str, float]:
    """Read and flatten all indicator values from the data window (tv values)."""
    resp = _run_tv("values")
    data = resp.get("data") or resp
    return _flatten_studies(data)


def _read_pine_lines() -> list[dict]:
    """Read horizontal Pine Script lines (anchored VWAPs, etc.)."""
    resp = _run_tv("data", "lines")
    data = resp.get("data") or []
    if isinstance(data, list):
        return data
    return []


def _read_pine_boxes() -> list[dict]:
    """Read Pine Script boxes (Value Area rectangles)."""
    resp = _run_tv("data", "boxes")
    data = resp.get("data") or []
    if isinstance(data, list):
        return data
    return []


def _read_quote() -> dict:
    """Read real-time price quote."""
    resp = _run_tv("quote")
    data = resp.get("data") or resp
    if isinstance(data, dict):
        return data
    return {}


# ── Snapshot assembly ──────────────────────────────────────────────────────────

def _derive_snapshot(price: float, sv: dict, pine_lines: list, pine_boxes: list) -> dict:
    """
    Derive a structured technical snapshot from raw TV data.

    Returns:
        {
          "price": float,
          "vwap": float | None,
          "vah": float | None,    # Value Area High
          "val": float | None,    # Value Area Low
          "poc": float | None,    # Point of Control
          "rsi": float | None,
          "atr": float | None,
          "macd": float | None,
          "macd_signal": float | None,
          "anchors": [float],     # anchored VWAP levels from Pine lines
          "zones": [{"vah": float, "val": float}],  # value zones from Pine boxes
          "price_vs_vwap": float,   # % above/below VWAP
          "price_vs_poc": float,    # % above/below POC
          "within_value_area": bool,
          "momentum_score": float,  # -1.0 to +1.0
          "regime": str,            # "bullish" | "bearish" | "neutral"
          "setup_quality": str,     # "strong" | "moderate" | "weak" | "no_data"
          "stop_tv": float | None,  # chart-structure-based stop price
        }
    """
    snap: dict[str, Any] = {"price": price}

    # ── Extract indicator values ───────────────────────────────────────────────
    vwap   = _find_value(sv, _VWAP_KEYS)
    vah    = _find_value(sv, _VAH_KEYS)
    val    = _find_value(sv, _VAL_KEYS)
    poc    = _find_value(sv, _POC_KEYS)
    rsi    = _find_value(sv, _RSI_KEYS)
    atr    = _find_value(sv, _ATR_KEYS)
    macd   = _find_value(sv, _MACD_KEYS)
    signal = _find_value(sv, _SIGNAL_KEYS)

    # Fall back to Pine boxes for VAH/VAL/POC if not in study values
    if (vah is None or val is None) and pine_boxes:
        # Use the most recent box (last in list)
        box = pine_boxes[-1]
        top = box.get("top_price") or box.get("top")
        bot = box.get("bottom_price") or box.get("bottom")
        if top and bot:
            vah = vah or float(top)
            val = val or float(bot)

    # Anchored VWAP levels from Pine lines
    anchors: list[float] = []
    for line in pine_lines:
        p = line.get("price")
        if p:
            try:
                anchors.append(round(float(p), 4))
            except (TypeError, ValueError):
                pass
    anchors.sort(reverse=True)

    # Value zones from Pine boxes
    zones: list[dict] = []
    for box in pine_boxes:
        top = box.get("top_price") or box.get("top")
        bot = box.get("bottom_price") or box.get("bottom")
        if top and bot:
            zones.append({"vah": round(float(top), 4), "val": round(float(bot), 4)})

    snap.update({
        "vwap": round(vwap, 4) if vwap else None,
        "vah":  round(vah, 4)  if vah  else None,
        "val":  round(val, 4)  if val  else None,
        "poc":  round(poc, 4)  if poc  else None,
        "rsi":  round(rsi, 2)  if rsi  else None,
        "atr":  round(atr, 4)  if atr  else None,
        "macd": round(macd, 4) if macd else None,
        "macd_signal": round(signal, 4) if signal else None,
        "anchors": anchors,
        "zones": zones,
    })

    # ── Derived metrics ────────────────────────────────────────────────────────
    snap["price_vs_vwap"] = round((price - vwap) / vwap, 4) if vwap and price else 0.0
    snap["price_vs_poc"]  = round((price - poc)  / poc,  4) if poc  and price else 0.0
    snap["within_value_area"] = bool(vah and val and val <= price <= vah)

    # Momentum score: sum of signals / total possible signals → [-1, +1]
    signals_raw: list[int] = []
    if vwap and price:
        signals_raw.append(1 if price > vwap else -1)
    if poc and price:
        signals_raw.append(1 if price > poc else -1)
    if rsi is not None:
        if rsi > 55:   signals_raw.append(1)
        elif rsi < 45: signals_raw.append(-1)
    if macd is not None and signal is not None:
        signals_raw.append(1 if macd > signal else -1)

    if signals_raw:
        momentum_score = round(sum(signals_raw) / len(signals_raw), 3)
    else:
        momentum_score = 0.0

    snap["momentum_score"] = momentum_score
    snap["regime"] = (
        "bullish" if momentum_score >  0.3 else
        "bearish" if momentum_score < -0.3 else
        "neutral"
    )

    # Setup quality: based on data completeness + signal clarity
    data_points = sum(1 for v in [vwap, vah, val, poc, rsi] if v is not None)
    if data_points >= 4 and abs(momentum_score) >= 0.5:
        snap["setup_quality"] = "strong"
    elif data_points >= 3:
        snap["setup_quality"] = "moderate"
    elif data_points >= 1:
        snap["setup_quality"] = "weak"
    else:
        snap["setup_quality"] = "no_data"

    # Stop suggestion based on chart structure
    stop_tv: float | None = None
    if price:
        # Long position stop: below VAL or below VWAP, whichever is higher (tighter)
        long_candidates: list[float] = []
        if val and price > val:
            long_candidates.append(round(val * 0.995, 4))
        if vwap and price > vwap:
            long_candidates.append(round(vwap * 0.99, 4))
        # Also consider nearest anchor below price
        anchors_below = [a for a in anchors if a < price]
        if anchors_below:
            long_candidates.append(round(anchors_below[0] * 0.997, 4))
        if long_candidates:
            stop_tv = max(long_candidates)  # tightest stop (highest floor)
        elif atr and price:
            stop_tv = round(price - 2.0 * atr, 4)

    snap["stop_tv"] = stop_tv
    return snap


def read_snapshot(ticker: str, timeframe: str = "D") -> dict:
    """
    Switch TradingView to `ticker` and read full technical snapshot.

    Returns the snapshot dict, or {} if TV is unavailable.
    """
    if not _switch_to(ticker, timeframe):
        return {}

    quote   = _read_quote()
    price   = float(quote.get("last_price") or quote.get("close") or 0)
    sv      = _read_study_values()
    lines   = _read_pine_lines()
    boxes   = _read_pine_boxes()

    if not price and not sv:
        return {}

    snap = _derive_snapshot(price, sv, lines, boxes)
    _log.debug(
        "tv_snapshot",
        ticker=ticker,
        regime=snap["regime"],
        rsi=snap.get("rsi"),
        vwap=snap.get("vwap"),
        val=snap.get("val"),
        stop_tv=snap.get("stop_tv"),
    )
    return snap


def collect_for_positions(
    tickers: list[str],
    layout: str = "ultimate profile",
    timeframe: str = "D",
) -> dict[str, dict]:
    """
    Load TradingView layout once, then collect technical snapshots for each ticker.

    Args:
        tickers: list of ticker symbols (agente format)
        layout: TradingView saved layout name
        timeframe: chart timeframe (default "D" = daily)

    Returns:
        {ticker: snapshot_dict} — tickers with no data are omitted
    """
    if not tickers:
        return {}

    # Check TV is running
    health = _run_tv("status")
    if not health.get("success"):
        _log.warning("tv_not_available", error=health.get("error", "?"))
        return {}

    # Load layout once
    if layout:
        r = _run_tv("layout", "switch", layout)
        if not r.get("success"):
            _log.warning("tv_layout_switch_failed", layout=layout, error=r.get("error"))
            # Continue anyway — use whatever is currently loaded
        else:
            _log.info("tv_layout_loaded", layout=layout)
            time.sleep(2.0)  # wait for layout to render

    results: dict[str, dict] = {}
    for ticker in tickers:
        try:
            snap = read_snapshot(ticker, timeframe)
            if snap:
                results[ticker] = snap
        except Exception as exc:
            _log.warning("tv_snapshot_failed", ticker=ticker, error=str(exc))

    _log.info(
        "tv_collect_done",
        requested=len(tickers),
        collected=len(results),
        timeframe=timeframe,
    )
    return results
