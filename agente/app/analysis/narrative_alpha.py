"""
Narrative Alpha — DeepVue Theme Tracker + X Sentiment

Gera sinais alpha por ticker a partir de:
  1. DeepVue theme tracker: momentum tematico (fluxo institucional)
  2. X tweets: confirmacao de narrativa (retail catching up = move ainda tem forca)

Logica principal (DeepVue + X pattern):
  - DeepVue = sinal primario (fluxo institucional / tematico)
  - X = confirmador de interesse: quando retail nota o tema, o move ainda nao acabou
  - Quando DeepVue decay + X confirmacao = short de alta convicao
  - Quando DeepVue force + X confirmacao = long de alta convicao

Referencia: HIMS short — DeepVue (GLP-1 decay, institutional outflow) confirmado no X
(narrativa de restricao da FDA atingindo retail = sweet spot, nao tarde demais)

Output: NarrativeAlphaResult com NarrativeSignal por ticker
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("analysis.narrative_alpha")


# ── Theme → Ticker mapping ────────────────────────────────────────────────────

_THEME_TICKERS: dict[str, list[str]] = {
    "Semiconductors":        ["NVDA", "AMD", "SMH", "MU", "AVGO", "TSM", "QCOM", "INTC", "SOXX"],
    "AI":                    ["NVDA", "MSFT", "GOOGL", "META", "AMD", "PLTR", "AI", "SMCI", "BOTZ"],
    "Software":              ["MSFT", "GOOGL", "META", "AMZN", "ORCL", "CRM", "NOW", "ADBE"],
    "Bitcoin Miners":        ["MARA", "RIOT", "HUT", "CLSK", "BTBT", "CIFR"],
    "Crypto":                ["MARA", "RIOT", "COIN", "MSTR", "BITO"],
    "Energy":                ["XLE", "CVX", "XOM", "COP", "OXY", "MPC", "VLO", "PSX"],
    "Oil & Gas":             ["XLE", "CVX", "XOM", "COP", "OXY", "HAL", "SLB", "BKR"],
    "Gold Miners":           ["GDX", "GDXJ", "NEM", "GOLD", "AEM", "WPM", "PAAS"],
    "Gold":                  ["GLD", "IAU", "GOLD", "NEM", "GDX"],
    "Silver":                ["SLV", "PAAS", "SILJ"],
    "Biotech":               ["XBI", "ARKG", "MRNA", "BNTX", "GILD", "BIIB", "REGN"],
    "Healthcare":            ["XLV", "UNH", "JNJ", "PFE", "ABBV", "LLY", "AMGN"],
    "Consumer Staples":      ["XLP", "PG", "KO", "WMT", "COST", "PM", "MO"],
    "Consumer Discretionary":["XLY", "AMZN", "TSLA", "HD", "NKE", "SBUX", "MCD"],
    "Utilities":             ["XLU", "NEE", "DUK", "SO", "AEP", "D"],
    "Real Estate":           ["VNQ", "O", "SPG", "PLD", "AMT", "EQIX"],
    "Banks":                 ["XLF", "JPM", "BAC", "GS", "MS", "C", "WFC"],
    "Aerospace & Defense":   ["ITA", "LMT", "RTX", "NOC", "GD", "KTOS", "AVAV", "HII"],
    "Airlines":              ["JETS", "DAL", "UAL", "AAL", "LUV"],
    "EV":                    ["TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI"],
    "Clean Energy":          ["ICLN", "ENPH", "FSLR", "NEE", "PLUG", "SEDG"],
    "Cloud":                 ["MSFT", "AMZN", "GOOGL", "CRM", "SNOW", "DDOG", "MDB"],
    "Cybersecurity":         ["HACK", "CRWD", "PANW", "ZS", "FTNT", "S", "OKTA"],
    "Growth":                ["QQQ", "ARKK", "NVDA", "TSLA", "META", "AMZN"],
    "Value":                 ["VTV", "BRK-B", "JPM", "JNJ"],
    "Small Caps":            ["IWM", "SLY", "IJR"],
    "Emerging Markets":      ["EEM", "EWZ", "FXI", "KWEB", "VWO"],
    "China Tech":            ["BABA", "JD", "BIDU", "KWEB", "PDD", "TCEHY"],
    "Memory":                ["MU", "WDC", "SNDK"],
    "Data Centers":          ["EQIX", "DLR", "SMCI", "NVDA", "IREN"],
    "Shipping":              ["ZIM", "DAC", "SBLK", "MATX", "GOGL"],
    "Materials":             ["XLB", "NEM", "FCX", "SCCO", "CLF", "AA"],
    "Industrials":           ["XLI", "GE", "HON", "CAT", "DE"],
    "Automation":            ["ROBO", "ABB", "FANUY", "IRBT", "ISRG"],
    "E-Commerce":            ["AMZN", "SHOP", "MELI", "ETSY", "EBAY"],
    "Telecom":               ["VZ", "T", "TMUS"],
    "Weight Loss":           ["HIMS", "LLY", "NVO", "AMGN"],
    "GLP-1":                 ["LLY", "NVO", "HIMS", "AMGN"],
    "Homebuilders":          ["XHB", "DHI", "LEN", "TOL", "PHM"],
    "Commodities":           ["DJP", "PDBC", "GSG", "XLE", "GLD", "SLV"],
}

# Fuzzy theme name matching: partial string -> canonical theme(s)
_THEME_FUZZY: list[tuple[str, list[str]]] = [
    ("semiconductor", ["Semiconductors"]),
    ("semis",         ["Semiconductors"]),
    (" ai ",          ["AI"]),
    ("artificial intelligence", ["AI"]),
    ("bitcoin",       ["Bitcoin Miners", "Crypto"]),
    ("crypto",        ["Crypto", "Bitcoin Miners"]),
    ("gold miner",    ["Gold Miners"]),
    ("gold",          ["Gold"]),
    ("silver",        ["Silver"]),
    ("oil",           ["Oil & Gas", "Energy"]),
    ("natural gas",   ["Energy"]),
    ("energy",        ["Energy"]),
    ("defense",       ["Aerospace & Defense"]),
    ("aerospace",     ["Aerospace & Defense"]),
    ("airline",       ["Airlines"]),
    ("bank",          ["Banks"]),
    ("financ",        ["Banks"]),
    ("biotech",       ["Biotech"]),
    ("healthcare",    ["Healthcare"]),
    ("health",        ["Healthcare"]),
    ("cloud",         ["Cloud"]),
    ("cyber",         ["Cybersecurity"]),
    ("electric vehicle", ["EV"]),
    (" ev ",          ["EV"]),
    ("china",         ["China Tech", "Emerging Markets"]),
    ("memory",        ["Memory"]),
    ("data center",   ["Data Centers"]),
    ("shipping",      ["Shipping"]),
    ("glp",           ["GLP-1", "Weight Loss"]),
    ("weight loss",   ["Weight Loss"]),
    ("homebuilder",   ["Homebuilders"]),
    ("material",      ["Materials"]),
    ("industrial",    ["Industrials"]),
    ("small cap",     ["Small Caps"]),
    ("emerging market", ["Emerging Markets"]),
]

# ── X author credibility ──────────────────────────────────────────────────────

_AUTHOR_WEIGHTS: dict[str, float] = {
    "SethCL":          2.0,
    "saxena_puru":     1.8,
    "SubuTrade":       1.5,
    "KobeissiLetter":  1.7,
    "zerohedge":       1.5,
    "MacroAlf":        1.8,
    "RaoulGMI":        1.7,
    "michaeljburry":   2.0,
    "TaviCosta":       1.8,
    "GunjanJS":        1.5,
    "LizAnnSonders":   1.6,
    "SoberLook":       1.5,
    "biancoresearch":  1.7,
    "elerianm":        1.8,
    "howardlindzon":   1.4,
    "DeepVue":         1.5,
    "SpotGamma":       1.5,
    "Spectra":         1.4,
}

# ── Sentiment keyword sets ────────────────────────────────────────────────────

_BEARISH_WORDS: frozenset[str] = frozenset({
    "crash", "collapse", "drop", "plunge", "fall", "tank", "dump", "sell",
    "short", "bear", "bearish", "decline", "down", "lower", "worse", "recession",
    "fear", "panic", "risk", "warning", "trouble", "bubble", "overvalued",
    "overbought", "overcrowded", "crowded", "losing", "loss", "weak", "weakness",
    "worsen", "deteriorat", "break", "broke", "broken", "fail", "failed",
    "restrict", "ban", "regulate", "shutdown", "exit", "sell-off", "selloff",
    "correction", "toppy", "rollover", "distribution", "fade",
})

_BULLISH_WORDS: frozenset[str] = frozenset({
    "rally", "surge", "pump", "rise", "soar", "jump", "buy", "long",
    "bull", "bullish", "upside", "higher", "better", "strong", "strength",
    "growth", "beat", "breakout", "momentum", "accumulate", "dip",
    "undervalued", "oversold", "opportunity", "catalyst", "bottom", "floor",
    "support", "bid", "squeeze", "rotation", "into", "buying", "inflow",
    "record", "ath", "all-time high", "outperform",
})


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class NarrativeSignal:
    ticker: str
    deepvue_score: float = 0.0         # [-1, 1] — tema decaindo = negativo
    deepvue_themes: list[str] = field(default_factory=list)
    x_sentiment_score: float = 0.0     # [-1, 1] — sentimento bruto X
    x_confirmation: bool = False        # X confirma direcao do DeepVue?
    x_mention_count: int = 0
    composite_narrative: float = 0.0   # score final [-1, 1]
    rationale: list[str] = field(default_factory=list)


@dataclass
class NarrativeAlphaResult:
    signals: dict[str, NarrativeSignal] = field(default_factory=dict)
    top_narrative_long: list[str] = field(default_factory=list)
    top_narrative_short: list[str] = field(default_factory=list)
    deepvue_themes_parsed: dict[str, float] = field(default_factory=dict)
    x_items_scored: int = 0


# ── DeepVue parser ────────────────────────────────────────────────────────────

def _parse_deepvue_themes(rss_items: list[Any]) -> dict[str, float]:
    """
    Extrai score por tema do DeepVue a partir dos rss_items.

    Usa o periodo "1W" como sinal primario (momentum de curto prazo institucional).
    Fallback: "Today" se 1W nao disponivel.

    Returns:
        dict[tema_name, score_pct_as_float] ex: {"Semiconductors": -0.082}
    """
    raw_text = ""
    for item in rss_items:
        sn = getattr(item, "source_name", "") or ""
        if "deepvue" in sn.lower() or "theme tracker" in sn.lower():
            raw_text = getattr(item, "summary", "") or ""
            break

    if not raw_text:
        return {}

    # Parseia secoes por periodo: [Today], [1W], [1M], etc.
    period_data: dict[str, dict[str, float]] = {}
    current_period = None

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Detecta cabecalho de periodo
        m = re.match(r"^\[(\w+)\]$", line)
        if m:
            current_period = m.group(1)
            period_data[current_period] = {}
            continue

        # Detecta linha de tema: "  Nome: +X.XX%"
        if current_period:
            m = re.match(r"^\s*(.+?):\s*([+-]?\d+\.?\d*)%\s*$", line)
            if m:
                theme_name = m.group(1).strip()
                try:
                    pct = float(m.group(2)) / 100.0
                except ValueError:
                    pct = 0.0
                period_data[current_period][theme_name] = pct

    # Prioridade de periodo: 1W > Today > 1M
    for period in ("1W", "Today", "1M"):
        if period_data.get(period):
            return period_data[period]

    return {}


def _theme_score_to_signal(pct: float) -> float:
    """
    Converte % de retorno do tema em signal [-1, 1].
    Threshold: +-5% por semana = sinal maximo.
    """
    return max(-1.0, min(1.0, pct / 0.05))


def _match_theme(theme_name: str) -> str | None:
    """Retorna o canonical theme name a partir de nome parcial/aproximado."""
    # Match exato primeiro
    if theme_name in _THEME_TICKERS:
        return theme_name

    # Fuzzy match via palavras-chave
    name_lower = f" {theme_name.lower()} "
    for keyword, themes in _THEME_FUZZY:
        if keyword in name_lower:
            return themes[0]

    return None


def _build_ticker_deepvue_scores(
    theme_scores: dict[str, float],
) -> dict[str, tuple[float, list[str]]]:
    """
    Para cada ticker, calcula score medio baseado nos temas ao qual pertence.

    Returns:
        dict[ticker, (avg_score, [themes_matched])]
    """
    ticker_scores: dict[str, list[float]] = {}
    ticker_themes: dict[str, list[str]] = {}

    for theme_raw, score in theme_scores.items():
        canonical = _match_theme(theme_raw)
        if canonical is None:
            continue

        tickers = _THEME_TICKERS.get(canonical, [])
        sig = _theme_score_to_signal(score)

        for t in tickers:
            ticker_scores.setdefault(t, []).append(sig)
            ticker_themes.setdefault(t, []).append(f"{canonical} {score:+.1%}")

    result: dict[str, tuple[float, list[str]]] = {}
    for ticker, scores in ticker_scores.items():
        avg = sum(scores) / len(scores)
        result[ticker] = (round(avg, 4), ticker_themes.get(ticker, []))

    return result


# ── X Sentiment parser ────────────────────────────────────────────────────────

def _score_text_sentiment(text: str) -> float:
    """
    Score de sentimento baseado em keyword matching [-1, 1].
    Positivo = bullish, negativo = bearish.
    """
    text_lower = text.lower()
    words = re.findall(r"\b\w+\b", text_lower)
    word_set = set(words)

    bull_hits = len(word_set & _BULLISH_WORDS)
    bear_hits = len(word_set & _BEARISH_WORDS)

    net = bull_hits - bear_hits
    total = bull_hits + bear_hits or 1
    return max(-1.0, min(1.0, net / max(total, 3)))


def _extract_ticker_mentions(text: str, known_tickers: set[str]) -> list[str]:
    """
    Extrai tickers mencionados no texto.
    Procura por $TICKER ou TICKER standalone em maiusculas.
    """
    # $TICKER pattern
    dollar_tickers = re.findall(r"\$([A-Z]{1,6})", text)

    # Standalone uppercase words que batem com universe
    words = re.findall(r"\b([A-Z]{2,6})\b", text)
    standalone = [w for w in words if w in known_tickers]

    return list(set(dollar_tickers + standalone))


def _build_x_sentiment(
    x_items: list[Any],
    known_tickers: set[str],
) -> dict[str, tuple[float, int]]:
    """
    Para cada ticker, computa score de sentimento dos X tweets.

    Ponderado por:
      - Credibilidade do autor (author_weight)
      - Engajamento (likes + reposts, cap em 10k)
      - Num de mencoes

    Returns:
        dict[ticker, (weighted_sentiment, mention_count)]
    """
    ticker_weighted_sum: dict[str, float] = {}
    ticker_weight_total: dict[str, float] = {}
    ticker_count: dict[str, int] = {}

    for item in x_items:
        text = getattr(item, "text", "") or ""
        author = getattr(item, "author", "") or ""
        eng = getattr(item, "engagement_info", None)

        if not text:
            continue

        # Sentimento do tweet
        sentiment = _score_text_sentiment(text)

        # Peso base = credibilidade do autor
        author_w = _AUTHOR_WEIGHTS.get(author, 1.0)

        # Boost por engajamento (normalizado, cap em 3x)
        likes = (getattr(eng, "likes", 0) or 0) if eng else 0
        reposts = (getattr(eng, "reposts", 0) or 0) if eng else 0
        engagement = likes + reposts * 2
        eng_boost = min(3.0, 1.0 + engagement / 5000.0)

        weight = author_w * eng_boost

        # Tickers mencionados neste tweet
        mentioned = _extract_ticker_mentions(text, known_tickers)

        # Se nenhum ticker explicito, tenta via temas no texto
        if not mentioned:
            text_lower = text.lower()
            for keyword, themes in _THEME_FUZZY:
                if keyword in f" {text_lower} ":
                    for theme in themes:
                        for t in _THEME_TICKERS.get(theme, []):
                            if t in known_tickers:
                                mentioned.append(t)
                    break

        for ticker in set(mentioned):
            if ticker not in known_tickers:
                continue
            ticker_weighted_sum[ticker] = ticker_weighted_sum.get(ticker, 0.0) + sentiment * weight
            ticker_weight_total[ticker] = ticker_weight_total.get(ticker, 0.0) + weight
            ticker_count[ticker] = ticker_count.get(ticker, 0) + 1

    result: dict[str, tuple[float, int]] = {}
    for ticker, total_w in ticker_weight_total.items():
        if total_w > 0:
            score = round(ticker_weighted_sum[ticker] / total_w, 4)
            result[ticker] = (score, ticker_count.get(ticker, 0))

    return result


# ── Main function ─────────────────────────────────────────────────────────────

def compute_narrative_alpha(
    bundle: Any,
    known_tickers: set[str] | None = None,
) -> NarrativeAlphaResult:
    """
    Computa sinais de narrative alpha a partir do bundle diario.

    Args:
        bundle: DailyIngestionBundle com rss_items e x_items
        known_tickers: set de tickers no universo (para filtrar mencoes no X)

    Returns:
        NarrativeAlphaResult com NarrativeSignal por ticker
    """
    result = NarrativeAlphaResult()

    rss_items = getattr(bundle, "rss_items", []) or []
    x_items   = getattr(bundle, "x_items", []) or []

    # ── 1. DeepVue: theme scores ─────────────────────────────────────────────
    theme_scores = _parse_deepvue_themes(rss_items)
    result.deepvue_themes_parsed = {k: round(v, 4) for k, v in theme_scores.items()}

    if theme_scores:
        _log.info("deepvue_themes_parsed", n=len(theme_scores),
                  top3_bull=[k for k, v in sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)[:3]],
                  top3_bear=[k for k, v in sorted(theme_scores.items(), key=lambda x: x[1])[:3]])
    else:
        _log.warning("deepvue_no_themes_found")

    ticker_dv = _build_ticker_deepvue_scores(theme_scores)

    # ── 2. X: sentiment per ticker ──────────────────────────────────────────
    # Build known tickers set: from DeepVue mapping + provided universe
    all_dv_tickers: set[str] = set()
    for tlist in _THEME_TICKERS.values():
        all_dv_tickers.update(tlist)
    if known_tickers:
        all_dv_tickers.update(known_tickers)

    ticker_x = _build_x_sentiment(x_items, all_dv_tickers)
    result.x_items_scored = len(x_items)

    if ticker_x:
        _log.info("x_sentiment_done", n_tickers=len(ticker_x), n_items=len(x_items))

    # ── 3. Composite per ticker ──────────────────────────────────────────────
    all_tickers = set(ticker_dv.keys()) | set(ticker_x.keys())

    for ticker in all_tickers:
        dv_score, dv_themes = ticker_dv.get(ticker, (0.0, []))
        x_score, x_count   = ticker_x.get(ticker, (0.0, 0))

        sig = NarrativeSignal(ticker=ticker)
        sig.deepvue_score       = dv_score
        sig.deepvue_themes      = dv_themes[:3]
        sig.x_sentiment_score   = x_score
        sig.x_mention_count     = x_count

        # X confirmation: both signals point same direction (and both meaningful)
        dv_strong = abs(dv_score) > 0.15
        x_strong  = abs(x_score)  > 0.10 and x_count >= 1
        sig.x_confirmation = (
            dv_strong and x_strong
            and (dv_score * x_score > 0)  # same sign
        )

        # Composite narrative score:
        # DeepVue is primary (weight 0.70)
        # X is confirmation multiplier (weight 0.30, boosted if confirmation)
        if sig.x_confirmation:
            # Both agree: DeepVue primary + X boosts conviction
            x_boost = 1.0 + 0.40 * min(1.0, x_count / 3.0)  # more mentions = stronger boost
            composite = (0.70 * dv_score + 0.30 * x_score) * x_boost
        elif dv_strong and not x_strong:
            # DeepVue only: still meaningful but lower conviction
            composite = 0.80 * dv_score
        elif x_strong and not dv_strong:
            # X only (no DeepVue theme): weaker signal
            composite = 0.40 * x_score
        else:
            composite = 0.70 * dv_score + 0.30 * x_score

        sig.composite_narrative = round(max(-1.0, min(1.0, composite)), 4)

        # Rationale
        if dv_themes:
            direction = "bull" if dv_score > 0 else "bear"
            sig.rationale.append(
                f"DeepVue {direction}: {', '.join(sig.deepvue_themes[:2])}"
            )
        if x_count > 0:
            x_dir = "bullish" if x_score > 0 else "bearish"
            sig.rationale.append(
                f"X sentiment {x_dir} ({x_count} tweet{'s' if x_count > 1 else ''})"
            )
        if sig.x_confirmation:
            sig.rationale.append("X confirma narrative DeepVue — sweet spot")

        result.signals[ticker] = sig

    # ── 4. Rankings ─────────────────────────────────────────────────────────
    sorted_tickers = sorted(
        result.signals.keys(),
        key=lambda t: result.signals[t].composite_narrative,
        reverse=True,
    )
    result.top_narrative_long  = [t for t in sorted_tickers if result.signals[t].composite_narrative > 0.15][:10]
    result.top_narrative_short = [t for t in reversed(sorted_tickers) if result.signals[t].composite_narrative < -0.15][:10]

    _log.info("narrative_alpha_done",
              n_tickers=len(result.signals),
              top_long=result.top_narrative_long[:3],
              top_short=result.top_narrative_short[:3])

    return result


def get_narrative_signal_for_ticker(
    ticker: str,
    narrative: NarrativeAlphaResult,
) -> float:
    """
    Retorna composite narrative score para uso no alpha_signals composite.
    Score [-1, 1].
    """
    sig = narrative.signals.get(ticker)
    return sig.composite_narrative if sig else 0.0
