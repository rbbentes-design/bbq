"""
Technical TV enrichment.

Applies TradingView chart-structure data (VWAP, Value Area, RSI, ATR)
to PositionResult objects from the portfolio optimizer.

Refinements applied per position:
  - stop_loss: overridden if chart structure provides a tighter/better-anchored stop
  - stop_pct / risk_reward: recalculated after stop update
  - rationale: TV summary line appended
"""

from __future__ import annotations

from app.audit.logger import get_logger

_log = get_logger("analysis.technical_tv")

_REGIME_EMOJI = {
    "bullish": "🟢",
    "bearish": "🔴",
    "neutral": "⚪",
}

_QUALITY_COLOR = {
    "strong":   "strong",
    "moderate": "moderate",
    "weak":     "weak",
    "no_data":  "no_data",
}


def _rationale_line(ticker: str, snap: dict) -> str:
    """Build a single rationale string summarising the TV snapshot."""
    parts: list[str] = []

    regime = snap.get("regime", "neutral")
    parts.append(f"TV {_REGIME_EMOJI.get(regime, '')} {regime}")

    pvw = snap.get("price_vs_vwap")
    if pvw is not None:
        sign = "+" if pvw >= 0 else ""
        parts.append(f"vs VWAP {sign}{pvw*100:.1f}%")

    rsi = snap.get("rsi")
    if rsi is not None:
        parts.append(f"RSI {rsi:.0f}")

    poc = snap.get("poc")
    if poc:
        parts.append(f"POC ${poc:.2f}")

    vah = snap.get("vah")
    val = snap.get("val")
    if vah and val:
        parts.append(f"VA ${val:.2f}–${vah:.2f}")

    quality = snap.get("setup_quality", "")
    if quality and quality != "no_data":
        parts.append(f"setup: {quality}")

    stop_tv = snap.get("stop_tv")
    if stop_tv:
        parts.append(f"chart stop: ${stop_tv:.2f}")

    return " | ".join(parts)


def enrich_positions_with_tv(
    positions: list,
    tv_map: dict[str, dict],
) -> list:
    """
    Enrich PositionResult objects with TradingView chart-structure data.

    For each position where TV data is available:
      1. Optionally refines stop_loss (uses chart structure anchor if tighter for longs)
      2. Recalculates stop_pct and risk_reward
      3. Appends a TV summary line to rationale

    Does NOT modify direction, conviction, allocation, or expected return.

    Args:
        positions: list of PositionResult from portfolio optimizer
        tv_map: {ticker: snapshot_dict} from tradingview.collect_for_positions

    Returns:
        Same list with positions mutated in-place (also returned for chaining)
    """
    if not tv_map:
        return positions

    for pos in positions:
        snap = tv_map.get(pos.ticker)
        if not snap:
            continue

        price = pos.entry_price or snap.get("price") or 0
        if not price:
            continue

        direction = pos.direction or "long"
        stop_tv   = snap.get("stop_tv")

        # ── Refine stop loss ──────────────────────────────────────────────────
        if stop_tv and stop_tv > 0:
            if direction == "long":
                # For longs: prefer the higher stop (closer to current price = tighter risk)
                if pos.stop_loss and pos.stop_loss > 0:
                    # Only upgrade if chart stop is tighter (higher) AND below price
                    if stop_tv > pos.stop_loss and stop_tv < price:
                        _log.info(
                            "tv_stop_refined",
                            ticker=pos.ticker,
                            old=pos.stop_loss,
                            new=stop_tv,
                        )
                        pos.stop_loss = stop_tv
                elif stop_tv < price:
                    pos.stop_loss = stop_tv
            else:
                # Short: prefer lower stop (closer to price from above)
                if pos.stop_loss and pos.stop_loss > 0:
                    if stop_tv < pos.stop_loss and stop_tv > price:
                        _log.info(
                            "tv_stop_refined",
                            ticker=pos.ticker,
                            direction="short",
                            old=pos.stop_loss,
                            new=stop_tv,
                        )
                        pos.stop_loss = stop_tv
                elif stop_tv > price:
                    pos.stop_loss = stop_tv

        # ── Recalculate stop_pct and risk_reward ──────────────────────────────
        if pos.stop_loss and price:
            pos.stop_pct = abs(price - pos.stop_loss) / price
            if pos.take_profit and pos.stop_loss and pos.stop_loss != price:
                pos.risk_reward = abs(pos.take_profit - price) / abs(price - pos.stop_loss)

        # ── Append TV rationale ───────────────────────────────────────────────
        line = _rationale_line(pos.ticker, snap)
        if line:
            if not isinstance(pos.rationale, list):
                pos.rationale = []
            pos.rationale.append(line)

    return positions
