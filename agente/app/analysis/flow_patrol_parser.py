"""
FlowPatrol Parser — extrai dados estruturados do texto bruto do relatorio.

O PDF FlowPatrol contem tabelas delimitadas por `|` no texto bruto:
  - Index ETFs Largest Position Changes
  - Single Stocks Largest Position Changes
  - Asset ETFs Largest Position Changes
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("analysis.flow_patrol")


@dataclass
class FlowPatrolTrade:
    ticker: str
    strike: float
    option_type: str
    expiry: str
    iv_pct: float
    bto: int
    btc: int
    sto: int
    stc: int
    delta_change: float | None = None
    gamma_change: float | None = None
    vega_change: float | None = None
    stock_px: float | None = None
    open_int: int | None = None

    @property
    def net_volume(self) -> int:
        return self.bto + self.btc - self.sto - self.stc

    @property
    def total_volume(self) -> int:
        return self.bto + self.btc + self.sto + self.stc

    @property
    def dominant_action(self) -> str:
        vols = {"BTO": self.bto, "STO": self.sto, "BTC": self.btc, "STC": self.stc}
        return max(vols, key=vols.get)


@dataclass
class FlowPatrolSectorFlow:
    sector: str
    delta_usd_m: float
    percentile: float


@dataclass
class FlowPatrolHighlight:
    ticker: str
    metric: str
    value_usd_m: float
    percentile: float
    sentiment: str
    note: str


@dataclass
class FlowPatrolParsed:
    report_date: str = ""
    trades_index_etfs: list[FlowPatrolTrade] = field(default_factory=list)
    trades_single_stocks: list[FlowPatrolTrade] = field(default_factory=list)
    trades_asset_etfs: list[FlowPatrolTrade] = field(default_factory=list)
    top_delta: list[FlowPatrolTrade] = field(default_factory=list)
    top_gamma: list[FlowPatrolTrade] = field(default_factory=list)
    top_vega: list[FlowPatrolTrade] = field(default_factory=list)
    sectors: list[FlowPatrolSectorFlow] = field(default_factory=list)
    highlights: list[FlowPatrolHighlight] = field(default_factory=list)
    narrative: dict[str, str] = field(default_factory=dict)

    @property
    def all_trades(self) -> list[FlowPatrolTrade]:
        return (self.trades_index_etfs + self.trades_single_stocks +
                self.trades_asset_etfs)


_TABLE_HEADER_RE = re.compile(
    r"(Index ETFs|Single Stocks|Asset ETFs)\s+Largest Position Changes\s+"
    r"(\d{2}/\d{2}/\d{2,4})"
)
_TABLE_ROW_RE = re.compile(
    r"^\s*([A-Z][A-Z0-9.\-/]{0,6})\s*\|\s*"
    r"([\d.]+)\s*\|\s*"
    r"([CP])\s*\|\s*"
    r"(\d{1,2}/\d{1,2}/\d{2,4})\s*\|\s*"
    r"([\d.]+)%?\s*\|\s*"
    r"(\d[\d,]*)\s*\|\s*"
    r"(\d[\d,]*)\s*\|\s*"
    r"(\d[\d,]*)\s*\|\s*"
    r"(\d[\d,]*)\s*$",
    re.MULTILINE,
)


def _parse_int(s: str) -> int:
    try:
        return int(str(s).replace(",", "").strip())
    except (ValueError, AttributeError):
        return 0


def _parse_float(s: str) -> float:
    try:
        return float(str(s).replace(",", "").replace("$", "").replace("%", "").strip())
    except (ValueError, AttributeError):
        return 0.0


def _parse_table(text: str, section_start: int, section_name: str) -> list[FlowPatrolTrade]:
    next_section_re = re.compile(
        r"\n(### |Single Stock Positioning|Directional Positioning|"
        r"Gamma Positioning|Volatility Positioning|Sector Breakdown|"
        r"The information in this report)",
    )
    m = next_section_re.search(text, section_start + 50)
    end = m.start() if m else min(section_start + 3000, len(text))
    block = text[section_start:end]

    trades = []
    for match in _TABLE_ROW_RE.finditer(block):
        try:
            trade = FlowPatrolTrade(
                ticker=match.group(1).strip(),
                strike=_parse_float(match.group(2)),
                option_type=match.group(3).strip(),
                expiry=match.group(4).strip(),
                iv_pct=_parse_float(match.group(5)),
                bto=_parse_int(match.group(6)),
                btc=_parse_int(match.group(7)),
                sto=_parse_int(match.group(8)),
                stc=_parse_int(match.group(9)),
            )
            trades.append(trade)
        except Exception as exc:
            _log.debug("flow_patrol_row_parse_err", error=str(exc)[:80])
    return trades


def parse_flow_patrol(raw_text: str) -> FlowPatrolParsed:
    result = FlowPatrolParsed()
    if not raw_text:
        return result

    m = re.search(r"(\d{2}/\d{2}/\d{2,4})", raw_text[:500])
    if m:
        result.report_date = m.group(1)

    for m in _TABLE_HEADER_RE.finditer(raw_text):
        section = m.group(1)
        trades = _parse_table(raw_text, m.end(), section)
        if section == "Index ETFs":
            seen = {(t.ticker, t.strike, t.option_type, t.expiry) for t in result.trades_index_etfs}
            for t in trades:
                key = (t.ticker, t.strike, t.option_type, t.expiry)
                if key not in seen:
                    result.trades_index_etfs.append(t)
                    seen.add(key)
        elif section == "Single Stocks":
            seen = {(t.ticker, t.strike, t.option_type, t.expiry) for t in result.trades_single_stocks}
            for t in trades:
                key = (t.ticker, t.strike, t.option_type, t.expiry)
                if key not in seen:
                    result.trades_single_stocks.append(t)
                    seen.add(key)
        elif section == "Asset ETFs":
            seen = {(t.ticker, t.strike, t.option_type, t.expiry) for t in result.trades_asset_etfs}
            for t in trades:
                key = (t.ticker, t.strike, t.option_type, t.expiry)
                if key not in seen:
                    result.trades_asset_etfs.append(t)
                    seen.add(key)

    highlight_re = re.compile(
        r"([A-Z]{2,6})\s+(?:shows?|exhibits?|displays?|with|stands? out)[^.]*?"
        r"(extreme\s+|very\s+high\s+|high\s+|large\s+)?"
        r"(bearish|bullish)\s+(delta|gamma|vega)[^-$]*?"
        r"(-?\$?[\d.]+)M?\s*\(([\d.]+)(?:st|nd|rd|th)\s*%?'?ile\)",
        re.IGNORECASE,
    )
    for m in highlight_re.finditer(raw_text):
        ticker = m.group(1)
        modifier = (m.group(2) or "").strip().lower()
        bias = m.group(3).lower()
        metric = m.group(4).lower()
        value = _parse_float(m.group(5))
        pct = _parse_float(m.group(6))
        sentiment = f"extreme_{bias}" if "extreme" in modifier else bias
        result.highlights.append(FlowPatrolHighlight(
            ticker=ticker, metric=metric, value_usd_m=value,
            percentile=pct, sentiment=sentiment,
            note=f"{modifier} {bias} {metric}".strip(),
        ))

    sector_re = re.compile(
        r"(Bond ETF|Crypto|Financial ETF|Tech|Energy|Industrial|Consumer|"
        r"Healthcare|Real Estate|Utilities|Materials)[^-$]*?"
        r"(-?\$?[\d.]+)M?\s*\(([\d.]+)(?:st|nd|rd|th)?\s*%?'?ile\)",
        re.IGNORECASE,
    )
    seen_sectors = set()
    for m in sector_re.finditer(raw_text):
        sector = m.group(1).strip()
        if sector in seen_sectors:
            continue
        seen_sectors.add(sector)
        result.sectors.append(FlowPatrolSectorFlow(
            sector=sector,
            delta_usd_m=_parse_float(m.group(2)),
            percentile=_parse_float(m.group(3)),
        ))

    section_re = re.compile(
        r"###\s*([^\n]+)\n(.*?)(?=\n###|\Z)",
        re.DOTALL,
    )
    for m in section_re.finditer(raw_text):
        title = m.group(1).strip()
        content = m.group(2).strip()
        content = re.sub(r"^[A-Z0-9.\s|%/]+\|.*$", "", content, flags=re.MULTILINE)
        clean = re.sub(r"\s+", " ", content).strip()[:600]
        if clean:
            result.narrative[title] = clean

    _log.info("flow_patrol_parsed",
              index_trades=len(result.trades_index_etfs),
              single_trades=len(result.trades_single_stocks),
              etf_trades=len(result.trades_asset_etfs),
              highlights=len(result.highlights),
              sectors=len(result.sectors),
              narrative_sections=len(result.narrative))
    return result
