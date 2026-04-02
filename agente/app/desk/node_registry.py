"""
MacroDesk — Node Registry

Define a hierarquia completa do universo de ativos:
  World → Asset Class → Region → Index → Sector → Asset → Internal Layers

Cada nó tem:
  - id único
  - label
  - level (0=World ... 7=Internal layer)
  - parent
  - children
  - metadata (ticker, gics, weight, cor de exibição)
"""

from __future__ import annotations
from typing import Any

# ── Paleta por classe de ativo ─────────────────────────────────────────────────
COLORS = {
    "world":        "#22d3ee",
    "asset_class":  "#60a5fa",
    "region":       "#818cf8",
    "index":        "#a78bfa",
    "sector":       "#34d399",
    "asset":        "#fbbf24",
    "vol":          "#f87171",
    "bond":         "#6ee7b7",
    "commodity":    "#f97316",
    "crypto":       "#c084fc",
    "fx":           "#e879f9",
    "macro":        "#38bdf8",
}

# ── Hierarquia estática ────────────────────────────────────────────────────────
# Cada entrada: id → {label, level, parent, children, color, ticker?, gics?, meta?}

NODES: dict[str, dict[str, Any]] = {

    # ── Nível 0: World ────────────────────────────────────────────────────────
    "world": {
        "label": "World", "level": 0, "parent": None, "color": COLORS["world"],
        "children": ["equities", "fixed_income", "commodities", "crypto", "fx", "macro_layer"],
    },

    # ── Nível 1: Asset Classes ────────────────────────────────────────────────
    "equities": {
        "label": "Equities", "level": 1, "parent": "world", "color": COLORS["asset_class"],
        "children": ["us_equities", "eu_equities", "em_equities", "jp_equities"],
    },
    "fixed_income": {
        "label": "Fixed Income", "level": 1, "parent": "world", "color": COLORS["bond"],
        "children": ["us_rates", "credit_ig", "credit_hy"],
    },
    "commodities": {
        "label": "Commodities", "level": 1, "parent": "world", "color": COLORS["commodity"],
        "children": ["energy_cmdy", "metals_cmdy"],
    },
    "crypto": {
        "label": "Crypto", "level": 1, "parent": "world", "color": COLORS["crypto"],
        "children": ["BTC-USD"],
    },
    "fx": {
        "label": "FX", "level": 1, "parent": "world", "color": COLORS["fx"],
        "children": ["DX-Y.NYB", "EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDBRL=X", "USDCNH=X"],
    },
    "macro_layer": {
        "label": "Macro / Liquidity", "level": 1, "parent": "world", "color": COLORS["macro"],
        "children": ["us_liquidity", "us_rates_macro"],
    },

    # ── Nível 2: Regions ──────────────────────────────────────────────────────
    "us_equities": {
        "label": "US Equities", "level": 2, "parent": "equities", "color": COLORS["region"],
        "children": ["sp500", "ndx100", "russell2000"],
    },
    "eu_equities": {
        "label": "EU Equities", "level": 2, "parent": "equities", "color": COLORS["region"],
        "children": ["eurostoxx50", "dax", "cac40", "ftse100"],
    },
    "em_equities": {
        "label": "EM Equities", "level": 2, "parent": "equities", "color": COLORS["region"],
        "children": ["eem_idx", "bovespa", "hang_seng", "nifty50", "csi300"],
    },
    "jp_equities": {
        "label": "JP Equities", "level": 2, "parent": "equities", "color": COLORS["region"],
        "children": ["nikkei225", "topix"],
    },
    "us_rates": {
        "label": "US Rates", "level": 2, "parent": "fixed_income", "color": COLORS["bond"],
        "children": ["TLT", "us_2y", "us_5y", "us_10y", "us_30y", "tips"],
    },
    "credit_hy": {
        "label": "HY Credit", "level": 2, "parent": "fixed_income", "color": COLORS["bond"],
        "children": ["HYG"],
    },
    "credit_ig": {
        "label": "IG Credit", "level": 2, "parent": "fixed_income", "color": COLORS["bond"],
        "children": ["LQD"],
    },
    "energy_cmdy": {
        "label": "Energy", "level": 2, "parent": "commodities", "color": COLORS["commodity"],
        "children": ["CL=F", "NG=F"],
    },
    "metals_cmdy": {
        "label": "Metals", "level": 2, "parent": "commodities", "color": COLORS["commodity"],
        "children": ["GLD", "SLV", "CPER"],
    },
    "us_liquidity": {
        "label": "US Liquidity", "level": 2, "parent": "macro_layer", "color": COLORS["macro"],
        "children": ["SHY", "BIL"],
    },
    "us_rates_macro": {
        "label": "Rates / Inflation", "level": 2, "parent": "macro_layer", "color": COLORS["macro"],
        "children": ["^VIX", "^VIX9D", "VIXY", "TIP"],
    },

    # ── Nível 3: Indices ──────────────────────────────────────────────────────
    "sp500": {
        "label": "S&P 500", "level": 3, "parent": "us_equities", "ticker": "^GSPC",
        "color": COLORS["index"], "liquidity_weight": 0.40,
        "children": [
            "sector_tech", "sector_health", "sector_fin", "sector_cd",
            "sector_comm", "sector_ind", "sector_cs", "sector_energy",
            "sector_mat", "sector_util", "sector_re",
        ],
    },
    "ndx100": {
        "label": "Nasdaq 100", "level": 3, "parent": "us_equities", "ticker": "^NDX",
        "color": COLORS["index"], "liquidity_weight": 0.20,
        "children": [
            "ndx_sector_tech", "ndx_sector_comm", "ndx_sector_cd",
            "ndx_sector_health", "ndx_sector_ind", "ndx_sector_cs",
            "ndx_sector_fin", "ndx_sector_util",
        ],
    },
    "russell2000": {
        "label": "Russell 2000", "level": 3, "parent": "us_equities", "ticker": "^RUT",
        "color": COLORS["index"], "liquidity_weight": 0.05,
        "children": [
            "rut_sector_health", "rut_sector_fin", "rut_sector_ind", "rut_sector_tech",
            "rut_sector_cd", "rut_sector_re", "rut_sector_energy",
            "rut_sector_mat", "rut_sector_comm", "rut_sector_cs", "rut_sector_util",
        ],
    },

    # ── Nível 3: EU Indices ───────────────────────────────────────────────────
    "eurostoxx50": {
        "label": "EuroStoxx 50", "level": 3, "parent": "eu_equities", "ticker": "FEZ",
        "color": COLORS["index"], "liquidity_weight": 0.06,
        "children": [], "meta": {"yf_symbol": "^STOXX50E", "etf_proxy": "FEZ"},
    },
    "dax": {
        "label": "DAX", "level": 3, "parent": "eu_equities", "ticker": "EWG",
        "color": COLORS["index"], "liquidity_weight": 0.04,
        "children": [], "meta": {"yf_symbol": "^GDAXI", "etf_proxy": "EWG"},
    },
    "cac40": {
        "label": "CAC 40", "level": 3, "parent": "eu_equities", "ticker": "EWQ",
        "color": COLORS["index"], "liquidity_weight": 0.03,
        "children": [], "meta": {"yf_symbol": "^FCHI", "etf_proxy": "EWQ"},
    },
    "ftse100": {
        "label": "FTSE 100", "level": 3, "parent": "eu_equities", "ticker": "EWU",
        "color": COLORS["index"], "liquidity_weight": 0.03,
        "children": [], "meta": {"yf_symbol": "^FTSE", "etf_proxy": "EWU"},
    },

    # ── Nível 3: EM Indices ───────────────────────────────────────────────────
    "eem_idx": {
        "label": "EM Equities", "level": 3, "parent": "em_equities", "ticker": "EEM",
        "color": COLORS["index"], "liquidity_weight": 0.05,
        "children": [],
    },
    "bovespa": {
        "label": "Bovespa", "level": 3, "parent": "em_equities", "ticker": "EWZ",
        "color": COLORS["index"], "liquidity_weight": 0.02,
        "children": [], "meta": {"yf_symbol": "^BVSP", "etf_proxy": "EWZ"},
    },
    "hang_seng": {
        "label": "Hang Seng", "level": 3, "parent": "em_equities", "ticker": "EWH",
        "color": COLORS["index"], "liquidity_weight": 0.02,
        "children": [], "meta": {"yf_symbol": "^HSI", "etf_proxy": "EWH"},
    },
    "nifty50": {
        "label": "Nifty 50", "level": 3, "parent": "em_equities", "ticker": "INDA",
        "color": COLORS["index"], "liquidity_weight": 0.02,
        "children": [], "meta": {"yf_symbol": "^NSEI", "etf_proxy": "INDA"},
    },
    "csi300": {
        "label": "CSI 300", "level": 3, "parent": "em_equities", "ticker": "MCHI",
        "color": COLORS["index"], "liquidity_weight": 0.02,
        "children": [], "meta": {"yf_symbol": "000300.SS", "etf_proxy": "MCHI"},
    },

    # ── Nível 3: JP Indices ───────────────────────────────────────────────────
    "nikkei225": {
        "label": "Nikkei 225", "level": 3, "parent": "jp_equities", "ticker": "EWJ",
        "color": COLORS["index"], "liquidity_weight": 0.04,
        "children": [], "meta": {"yf_symbol": "^N225", "etf_proxy": "EWJ"},
    },
    "topix": {
        "label": "TOPIX", "level": 3, "parent": "jp_equities", "ticker": "DBJP",
        "color": COLORS["index"], "liquidity_weight": 0.02,
        "children": [], "meta": {"yf_symbol": "^N300", "etf_proxy": "DBJP"},
    },

    # ── Nível 4: Sectors S&P 500 (GICS) — peso real no índice ────────────────
    # weights = SPX sector weights aproximados (Mar/2025)
    "sector_tech": {
        "label": "Technology", "level": 4, "parent": "sp500", "gics": "45",
        "color": COLORS["sector"], "weight": 0.31,
        "children": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "INTC", "TXN", "AMAT"],
    },
    "sector_health": {
        "label": "Healthcare", "level": 4, "parent": "sp500", "gics": "35",
        "color": COLORS["sector"], "weight": 0.12,
        "children": ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "ISRG", "AMGN"],
    },
    "sector_fin": {
        "label": "Financials", "level": 4, "parent": "sp500", "gics": "40",
        "color": COLORS["sector"], "weight": 0.13,
        "children": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "AXP"],
    },
    "sector_cd": {
        "label": "Cons. Discretionary", "level": 4, "parent": "sp500", "gics": "25",
        "color": COLORS["sector"], "weight": 0.10,
        "children": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "GM"],
    },
    "sector_comm": {
        "label": "Communication", "level": 4, "parent": "sp500", "gics": "50",
        "color": COLORS["sector"], "weight": 0.09,
        "children": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "EA", "WBD"],
    },
    "sector_ind": {
        "label": "Industrials", "level": 4, "parent": "sp500", "gics": "20",
        "color": COLORS["sector"], "weight": 0.09,
        "children": ["GE", "CAT", "HON", "UPS", "RTX", "LMT", "DE", "BA", "ETN", "MMM"],
    },
    "sector_cs": {
        "label": "Cons. Staples", "level": 4, "parent": "sp500", "gics": "30",
        "color": COLORS["sector"], "weight": 0.06,
        "children": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "GIS", "KMB"],
    },
    "sector_energy": {
        "label": "Energy", "level": 4, "parent": "sp500", "gics": "10",
        "color": COLORS["commodity"], "weight": 0.04,
        "children": ["XOM", "CVX", "COP", "EOG", "SLB", "OXY", "PSX", "VLO", "MPC", "KMI"],
    },
    "sector_mat": {
        "label": "Materials", "level": 4, "parent": "sp500", "gics": "15",
        "color": COLORS["sector"], "weight": 0.025,
        "children": ["LIN", "APD", "SHW", "FCX", "NEM", "CTVA", "ECL", "DOW", "NUE", "CF"],
    },
    "sector_util": {
        "label": "Utilities", "level": 4, "parent": "sp500", "gics": "55",
        "color": COLORS["sector"], "weight": 0.025,
        "children": ["NEE", "SO", "DUK", "AEP", "EXC", "XEL", "SRE", "PEG", "AWK", "ED"],
    },
    "sector_re": {
        "label": "Real Estate", "level": 4, "parent": "sp500", "gics": "60",
        "color": COLORS["sector"], "weight": 0.025,
        "children": ["AMT", "PLD", "EQIX", "CCI", "PSA", "WELL", "O", "DLR", "SPG", "VICI"],
    },

    # ── Nível 5: Assets macro (diretos no bundle) ─────────────────────────────
    "^VIX": {
        "label": "VIX", "level": 5, "parent": "us_rates_macro", "ticker": "^VIX",
        "color": COLORS["vol"], "children": [], "liquidity_weight": 0.01,
        "meta": {"type": "volatility"},
    },
    "TLT":     {"label": "TLT 20yr",  "level": 5, "parent": "us_rates",    "ticker": "TLT",      "color": COLORS["bond"],      "children": [], "liquidity_weight": 0.10},
    "HYG":     {"label": "HYG HY",    "level": 5, "parent": "credit_hy",   "ticker": "HYG",      "color": COLORS["bond"],      "children": [], "liquidity_weight": 0.05},
    "GLD":     {"label": "Gold",       "level": 5, "parent": "metals_cmdy", "ticker": "GLD",      "color": COLORS["commodity"], "children": [], "liquidity_weight": 0.07},
    "CL=F":    {"label": "WTI Crude",  "level": 5, "parent": "energy_cmdy", "ticker": "CL=F",     "color": COLORS["commodity"], "children": [], "liquidity_weight": 0.06},
    "BTC-USD": {"label": "Bitcoin",    "level": 5, "parent": "crypto",      "ticker": "BTC-USD",  "color": COLORS["crypto"],    "children": [], "liquidity_weight": 0.04},
    "DX-Y.NYB":{"label": "DXY",        "level": 5, "parent": "fx",          "ticker": "DX-Y.NYB", "color": COLORS["fx"],        "children": [], "liquidity_weight": 0.03},

    # ── Nível 5: FX pairs ─────────────────────────────────────────────────────
    "EURUSD=X": {"label": "EUR/USD", "level": 5, "parent": "fx", "ticker": "EURUSD=X", "color": COLORS["fx"], "children": [], "liquidity_weight": 0.04},
    "USDJPY=X": {"label": "USD/JPY", "level": 5, "parent": "fx", "ticker": "USDJPY=X", "color": COLORS["fx"], "children": [], "liquidity_weight": 0.03},
    "GBPUSD=X": {"label": "GBP/USD", "level": 5, "parent": "fx", "ticker": "GBPUSD=X", "color": COLORS["fx"], "children": [], "liquidity_weight": 0.02},
    "USDBRL=X": {"label": "USD/BRL", "level": 5, "parent": "fx", "ticker": "USDBRL=X", "color": COLORS["fx"], "children": [], "liquidity_weight": 0.01},
    "USDCNH=X": {"label": "USD/CNH", "level": 5, "parent": "fx", "ticker": "USDCNH=X", "color": COLORS["fx"], "children": [], "liquidity_weight": 0.02},

    # ── Nível 5: US Rates ─────────────────────────────────────────────────────
    "us_2y":  {"label": "2Y Tsy",  "level": 5, "parent": "us_rates", "ticker": "SHY",  "color": COLORS["bond"], "children": [], "liquidity_weight": 0.04, "meta": {"yf_symbol": "^IRX"}},
    "us_5y":  {"label": "5Y Tsy",  "level": 5, "parent": "us_rates", "ticker": "IEI",  "color": COLORS["bond"], "children": [], "liquidity_weight": 0.04, "meta": {"yf_symbol": "^FVX"}},
    "us_10y": {"label": "10Y Tsy", "level": 5, "parent": "us_rates", "ticker": "IEF",  "color": COLORS["bond"], "children": [], "liquidity_weight": 0.06, "meta": {"yf_symbol": "^TNX"}},
    "us_30y": {"label": "30Y Tsy", "level": 5, "parent": "us_rates", "ticker": "TLT",  "color": COLORS["bond"], "children": [], "liquidity_weight": 0.05, "meta": {"yf_symbol": "^TYX"}},
    "tips":   {"label": "TIPS",    "level": 5, "parent": "us_rates", "ticker": "TIP",   "color": COLORS["bond"], "children": [], "liquidity_weight": 0.03},

    # ── Nível 5: Credit ───────────────────────────────────────────────────────
    "LQD": {"label": "IG Credit (LQD)", "level": 5, "parent": "credit_ig", "ticker": "LQD", "color": COLORS["bond"], "children": [], "liquidity_weight": 0.05},

    # ── Nível 5: Commodities extras ───────────────────────────────────────────
    "NG=F":  {"label": "Nat. Gas",  "level": 5, "parent": "energy_cmdy", "ticker": "NG=F",  "color": COLORS["commodity"], "children": [], "liquidity_weight": 0.02},
    "SLV":   {"label": "Silver",    "level": 5, "parent": "metals_cmdy", "ticker": "SLV",   "color": COLORS["commodity"], "children": [], "liquidity_weight": 0.03},
    "CPER":  {"label": "Copper",    "level": 5, "parent": "metals_cmdy", "ticker": "CPER",  "color": COLORS["commodity"], "children": [], "liquidity_weight": 0.02},

    # ── Nível 5: Macro / Liquidez ─────────────────────────────────────────────
    "SHY":   {"label": "SHY 1-3yr", "level": 5, "parent": "us_liquidity",  "ticker": "SHY",  "color": COLORS["bond"],  "children": [], "liquidity_weight": 0.03},
    "BIL":   {"label": "T-Bills",   "level": 5, "parent": "us_liquidity",  "ticker": "BIL",  "color": COLORS["bond"],  "children": [], "liquidity_weight": 0.03},

    # ── Nível 5: Volatilidade / VIX term structure ────────────────────────────
    "^VIX9D": {"label": "VIX9D",  "level": 5, "parent": "us_rates_macro", "ticker": "^VIX9D", "color": COLORS["vol"], "children": [], "liquidity_weight": 0.005, "meta": {"type": "volatility"}},
    "VIXY":   {"label": "VIXY",   "level": 5, "parent": "us_rates_macro", "ticker": "VIXY",   "color": COLORS["vol"], "children": [], "liquidity_weight": 0.005, "meta": {"type": "volatility"}},
    "TIP":    {"label": "TIP",    "level": 5, "parent": "us_rates_macro", "ticker": "TIP",    "color": COLORS["bond"], "children": [], "liquidity_weight": 0.02},

    # ── Nível 5: Stocks S&P 500 — peso no SPX ─────────────────────────────────
    # Technology
    "AAPL": {"label": "Apple",      "level": 5, "parent": "sector_tech",   "ticker": "AAPL", "color": COLORS["asset"], "children": [], "weight": 0.075},
    "MSFT": {"label": "Microsoft",  "level": 5, "parent": "sector_tech",   "ticker": "MSFT", "color": COLORS["asset"], "children": [], "weight": 0.065},
    "NVDA": {"label": "Nvidia",     "level": 5, "parent": "sector_tech",   "ticker": "NVDA", "color": COLORS["asset"], "children": [], "weight": 0.060},
    "AVGO": {"label": "Broadcom",   "level": 5, "parent": "sector_tech",   "ticker": "AVGO", "color": COLORS["asset"], "children": [], "weight": 0.020},
    "ORCL": {"label": "Oracle",     "level": 5, "parent": "sector_tech",   "ticker": "ORCL", "color": COLORS["asset"], "children": [], "weight": 0.010},
    "CRM":  {"label": "Salesforce", "level": 5, "parent": "sector_tech",   "ticker": "CRM",  "color": COLORS["asset"], "children": [], "weight": 0.008},
    "AMD":  {"label": "AMD",        "level": 5, "parent": "sector_tech",   "ticker": "AMD",  "color": COLORS["asset"], "children": [], "weight": 0.009},
    "INTC": {"label": "Intel",      "level": 5, "parent": "sector_tech",   "ticker": "INTC", "color": COLORS["asset"], "children": [], "weight": 0.005},
    "TXN":  {"label": "Texas Instr","level": 5, "parent": "sector_tech",   "ticker": "TXN",  "color": COLORS["asset"], "children": [], "weight": 0.007},
    "AMAT": {"label": "Appl. Matls","level": 5, "parent": "sector_tech",   "ticker": "AMAT", "color": COLORS["asset"], "children": [], "weight": 0.007},
    # Healthcare
    "LLY":  {"label": "Eli Lilly",  "level": 5, "parent": "sector_health", "ticker": "LLY",  "color": COLORS["asset"], "children": [], "weight": 0.013},
    "UNH":  {"label": "UnitedHlth", "level": 5, "parent": "sector_health", "ticker": "UNH",  "color": COLORS["asset"], "children": [], "weight": 0.012},
    "JNJ":  {"label": "J&J",        "level": 5, "parent": "sector_health", "ticker": "JNJ",  "color": COLORS["asset"], "children": [], "weight": 0.010},
    "ABBV": {"label": "AbbVie",     "level": 5, "parent": "sector_health", "ticker": "ABBV", "color": COLORS["asset"], "children": [], "weight": 0.010},
    "MRK":  {"label": "Merck",      "level": 5, "parent": "sector_health", "ticker": "MRK",  "color": COLORS["asset"], "children": [], "weight": 0.009},
    "TMO":  {"label": "Thermo Fsh", "level": 5, "parent": "sector_health", "ticker": "TMO",  "color": COLORS["asset"], "children": [], "weight": 0.007},
    "ABT":  {"label": "Abbott",     "level": 5, "parent": "sector_health", "ticker": "ABT",  "color": COLORS["asset"], "children": [], "weight": 0.007},
    "DHR":  {"label": "Danaher",    "level": 5, "parent": "sector_health", "ticker": "DHR",  "color": COLORS["asset"], "children": [], "weight": 0.006},
    "ISRG": {"label": "Intuitive",  "level": 5, "parent": "sector_health", "ticker": "ISRG", "color": COLORS["asset"], "children": [], "weight": 0.007},
    "AMGN": {"label": "Amgen",      "level": 5, "parent": "sector_health", "ticker": "AMGN", "color": COLORS["asset"], "children": [], "weight": 0.007},
    # Financials
    "BRK-B":{"label": "Berkshire",  "level": 5, "parent": "sector_fin",    "ticker": "BRK-B","color": COLORS["asset"], "children": [], "weight": 0.018},
    "JPM":  {"label": "JPMorgan",   "level": 5, "parent": "sector_fin",    "ticker": "JPM",  "color": COLORS["asset"], "children": [], "weight": 0.014},
    "V":    {"label": "Visa",       "level": 5, "parent": "sector_fin",    "ticker": "V",    "color": COLORS["asset"], "children": [], "weight": 0.012},
    "MA":   {"label": "Mastercard", "level": 5, "parent": "sector_fin",    "ticker": "MA",   "color": COLORS["asset"], "children": [], "weight": 0.012},
    "BAC":  {"label": "Bank Amer",  "level": 5, "parent": "sector_fin",    "ticker": "BAC",  "color": COLORS["asset"], "children": [], "weight": 0.009},
    "WFC":  {"label": "Wells Fargo","level": 5, "parent": "sector_fin",    "ticker": "WFC",  "color": COLORS["asset"], "children": [], "weight": 0.008},
    "GS":   {"label": "Goldman",    "level": 5, "parent": "sector_fin",    "ticker": "GS",   "color": COLORS["asset"], "children": [], "weight": 0.007},
    "MS":   {"label": "Morgan Std", "level": 5, "parent": "sector_fin",    "ticker": "MS",   "color": COLORS["asset"], "children": [], "weight": 0.006},
    "SPGI": {"label": "S&P Global", "level": 5, "parent": "sector_fin",    "ticker": "SPGI", "color": COLORS["asset"], "children": [], "weight": 0.007},
    "AXP":  {"label": "Amex",       "level": 5, "parent": "sector_fin",    "ticker": "AXP",  "color": COLORS["asset"], "children": [], "weight": 0.007},
    # Consumer Discretionary
    "AMZN": {"label": "Amazon",     "level": 5, "parent": "sector_cd",     "ticker": "AMZN", "color": COLORS["asset"], "children": [], "weight": 0.035},
    "TSLA": {"label": "Tesla",      "level": 5, "parent": "sector_cd",     "ticker": "TSLA", "color": COLORS["asset"], "children": [], "weight": 0.017},
    "HD":   {"label": "Home Depot", "level": 5, "parent": "sector_cd",     "ticker": "HD",   "color": COLORS["asset"], "children": [], "weight": 0.009},
    "MCD":  {"label": "McDonald's", "level": 5, "parent": "sector_cd",     "ticker": "MCD",  "color": COLORS["asset"], "children": [], "weight": 0.007},
    "NKE":  {"label": "Nike",       "level": 5, "parent": "sector_cd",     "ticker": "NKE",  "color": COLORS["asset"], "children": [], "weight": 0.005},
    "LOW":  {"label": "Lowe's",     "level": 5, "parent": "sector_cd",     "ticker": "LOW",  "color": COLORS["asset"], "children": [], "weight": 0.006},
    "SBUX": {"label": "Starbucks",  "level": 5, "parent": "sector_cd",     "ticker": "SBUX", "color": COLORS["asset"], "children": [], "weight": 0.004},
    "TJX":  {"label": "TJX",        "level": 5, "parent": "sector_cd",     "ticker": "TJX",  "color": COLORS["asset"], "children": [], "weight": 0.006},
    "BKNG": {"label": "Booking",    "level": 5, "parent": "sector_cd",     "ticker": "BKNG", "color": COLORS["asset"], "children": [], "weight": 0.007},
    "GM":   {"label": "Gen. Motors","level": 5, "parent": "sector_cd",     "ticker": "GM",   "color": COLORS["asset"], "children": [], "weight": 0.004},
    # Communication Services
    "GOOGL":{"label": "Alphabet",   "level": 5, "parent": "sector_comm",   "ticker": "GOOGL","color": COLORS["asset"], "children": [], "weight": 0.025},
    "META": {"label": "Meta",       "level": 5, "parent": "sector_comm",   "ticker": "META", "color": COLORS["asset"], "children": [], "weight": 0.027},
    "NFLX": {"label": "Netflix",    "level": 5, "parent": "sector_comm",   "ticker": "NFLX", "color": COLORS["asset"], "children": [], "weight": 0.008},
    "DIS":  {"label": "Disney",     "level": 5, "parent": "sector_comm",   "ticker": "DIS",  "color": COLORS["asset"], "children": [], "weight": 0.005},
    "CMCSA":{"label": "Comcast",    "level": 5, "parent": "sector_comm",   "ticker": "CMCSA","color": COLORS["asset"], "children": [], "weight": 0.005},
    "T":    {"label": "AT&T",       "level": 5, "parent": "sector_comm",   "ticker": "T",    "color": COLORS["asset"], "children": [], "weight": 0.004},
    "VZ":   {"label": "Verizon",    "level": 5, "parent": "sector_comm",   "ticker": "VZ",   "color": COLORS["asset"], "children": [], "weight": 0.004},
    "TMUS": {"label": "T-Mobile",   "level": 5, "parent": "sector_comm",   "ticker": "TMUS", "color": COLORS["asset"], "children": [], "weight": 0.005},
    "EA":   {"label": "EA Games",   "level": 5, "parent": "sector_comm",   "ticker": "EA",   "color": COLORS["asset"], "children": [], "weight": 0.003},
    "WBD":  {"label": "Warner Bros","level": 5, "parent": "sector_comm",   "ticker": "WBD",  "color": COLORS["asset"], "children": [], "weight": 0.002},
    # Industrials
    "GE":   {"label": "GE Aero",    "level": 5, "parent": "sector_ind",    "ticker": "GE",   "color": COLORS["asset"], "children": [], "weight": 0.010},
    "CAT":  {"label": "Caterpillar","level": 5, "parent": "sector_ind",    "ticker": "CAT",  "color": COLORS["asset"], "children": [], "weight": 0.008},
    "HON":  {"label": "Honeywell",  "level": 5, "parent": "sector_ind",    "ticker": "HON",  "color": COLORS["asset"], "children": [], "weight": 0.007},
    "UPS":  {"label": "UPS",        "level": 5, "parent": "sector_ind",    "ticker": "UPS",  "color": COLORS["asset"], "children": [], "weight": 0.006},
    "RTX":  {"label": "Raytheon",   "level": 5, "parent": "sector_ind",    "ticker": "RTX",  "color": COLORS["asset"], "children": [], "weight": 0.007},
    "LMT":  {"label": "Lockheed",   "level": 5, "parent": "sector_ind",    "ticker": "LMT",  "color": COLORS["asset"], "children": [], "weight": 0.007},
    "DE":   {"label": "Deere",      "level": 5, "parent": "sector_ind",    "ticker": "DE",   "color": COLORS["asset"], "children": [], "weight": 0.006},
    "BA":   {"label": "Boeing",     "level": 5, "parent": "sector_ind",    "ticker": "BA",   "color": COLORS["asset"], "children": [], "weight": 0.005},
    "ETN":  {"label": "Eaton",      "level": 5, "parent": "sector_ind",    "ticker": "ETN",  "color": COLORS["asset"], "children": [], "weight": 0.007},
    "MMM":  {"label": "3M",         "level": 5, "parent": "sector_ind",    "ticker": "MMM",  "color": COLORS["asset"], "children": [], "weight": 0.004},
    # Consumer Staples
    "PG":   {"label": "P&G",        "level": 5, "parent": "sector_cs",     "ticker": "PG",   "color": COLORS["asset"], "children": [], "weight": 0.009},
    "KO":   {"label": "Coca-Cola",  "level": 5, "parent": "sector_cs",     "ticker": "KO",   "color": COLORS["asset"], "children": [], "weight": 0.007},
    "PEP":  {"label": "PepsiCo",    "level": 5, "parent": "sector_cs",     "ticker": "PEP",  "color": COLORS["asset"], "children": [], "weight": 0.007},
    "COST": {"label": "Costco",     "level": 5, "parent": "sector_cs",     "ticker": "COST", "color": COLORS["asset"], "children": [], "weight": 0.009},
    "WMT":  {"label": "Walmart",    "level": 5, "parent": "sector_cs",     "ticker": "WMT",  "color": COLORS["asset"], "children": [], "weight": 0.010},
    "PM":   {"label": "Philip Mor", "level": 5, "parent": "sector_cs",     "ticker": "PM",   "color": COLORS["asset"], "children": [], "weight": 0.005},
    "MO":   {"label": "Altria",     "level": 5, "parent": "sector_cs",     "ticker": "MO",   "color": COLORS["asset"], "children": [], "weight": 0.004},
    "CL":   {"label": "Colgate",    "level": 5, "parent": "sector_cs",     "ticker": "CL",   "color": COLORS["asset"], "children": [], "weight": 0.004},
    "GIS":  {"label": "Gen. Mills", "level": 5, "parent": "sector_cs",     "ticker": "GIS",  "color": COLORS["asset"], "children": [], "weight": 0.003},
    "KMB":  {"label": "Kimberly",   "level": 5, "parent": "sector_cs",     "ticker": "KMB",  "color": COLORS["asset"], "children": [], "weight": 0.003},
    # Energy
    "XOM":  {"label": "Exxon",      "level": 5, "parent": "sector_energy", "ticker": "XOM",  "color": COLORS["commodity"], "children": [], "weight": 0.014},
    "CVX":  {"label": "Chevron",    "level": 5, "parent": "sector_energy", "ticker": "CVX",  "color": COLORS["commodity"], "children": [], "weight": 0.010},
    "COP":  {"label": "ConocoPhil", "level": 5, "parent": "sector_energy", "ticker": "COP",  "color": COLORS["commodity"], "children": [], "weight": 0.005},
    "EOG":  {"label": "EOG Res.",   "level": 5, "parent": "sector_energy", "ticker": "EOG",  "color": COLORS["commodity"], "children": [], "weight": 0.004},
    "SLB":  {"label": "SLB",        "level": 5, "parent": "sector_energy", "ticker": "SLB",  "color": COLORS["commodity"], "children": [], "weight": 0.003},
    "OXY":  {"label": "Occidental", "level": 5, "parent": "sector_energy", "ticker": "OXY",  "color": COLORS["commodity"], "children": [], "weight": 0.003},
    "PSX":  {"label": "Phillips 66","level": 5, "parent": "sector_energy", "ticker": "PSX",  "color": COLORS["commodity"], "children": [], "weight": 0.003},
    "VLO":  {"label": "Valero",     "level": 5, "parent": "sector_energy", "ticker": "VLO",  "color": COLORS["commodity"], "children": [], "weight": 0.003},
    "MPC":  {"label": "Marathon",   "level": 5, "parent": "sector_energy", "ticker": "MPC",  "color": COLORS["commodity"], "children": [], "weight": 0.003},
    "KMI":  {"label": "Kinder Mor", "level": 5, "parent": "sector_energy", "ticker": "KMI",  "color": COLORS["commodity"], "children": [], "weight": 0.002},
    # Materials
    "LIN":  {"label": "Linde",      "level": 5, "parent": "sector_mat",    "ticker": "LIN",  "color": COLORS["sector"], "children": [], "weight": 0.006},
    "APD":  {"label": "Air Products","level": 5, "parent": "sector_mat",   "ticker": "APD",  "color": COLORS["sector"], "children": [], "weight": 0.004},
    "SHW":  {"label": "Sherwin-W",  "level": 5, "parent": "sector_mat",    "ticker": "SHW",  "color": COLORS["sector"], "children": [], "weight": 0.005},
    "FCX":  {"label": "Freeport",   "level": 5, "parent": "sector_mat",    "ticker": "FCX",  "color": COLORS["sector"], "children": [], "weight": 0.003},
    "NEM":  {"label": "Newmont",    "level": 5, "parent": "sector_mat",    "ticker": "NEM",  "color": COLORS["sector"], "children": [], "weight": 0.003},
    "CTVA": {"label": "Corteva",    "level": 5, "parent": "sector_mat",    "ticker": "CTVA", "color": COLORS["sector"], "children": [], "weight": 0.002},
    "ECL":  {"label": "Ecolab",     "level": 5, "parent": "sector_mat",    "ticker": "ECL",  "color": COLORS["sector"], "children": [], "weight": 0.003},
    "DOW":  {"label": "Dow Inc.",   "level": 5, "parent": "sector_mat",    "ticker": "DOW",  "color": COLORS["sector"], "children": [], "weight": 0.002},
    "NUE":  {"label": "Nucor",      "level": 5, "parent": "sector_mat",    "ticker": "NUE",  "color": COLORS["sector"], "children": [], "weight": 0.002},
    "CF":   {"label": "CF Indust.", "level": 5, "parent": "sector_mat",    "ticker": "CF",   "color": COLORS["sector"], "children": [], "weight": 0.002},
    # Utilities
    "NEE":  {"label": "NextEra",    "level": 5, "parent": "sector_util",   "ticker": "NEE",  "color": COLORS["sector"], "children": [], "weight": 0.007},
    "SO":   {"label": "Southern Co","level": 5, "parent": "sector_util",   "ticker": "SO",   "color": COLORS["sector"], "children": [], "weight": 0.004},
    "DUK":  {"label": "Duke Energy","level": 5, "parent": "sector_util",   "ticker": "DUK",  "color": COLORS["sector"], "children": [], "weight": 0.004},
    "AEP":  {"label": "Amer. Elec.","level": 5, "parent": "sector_util",   "ticker": "AEP",  "color": COLORS["sector"], "children": [], "weight": 0.003},
    "EXC":  {"label": "Exelon",     "level": 5, "parent": "sector_util",   "ticker": "EXC",  "color": COLORS["sector"], "children": [], "weight": 0.003},
    "XEL":  {"label": "Xcel Energy","level": 5, "parent": "sector_util",   "ticker": "XEL",  "color": COLORS["sector"], "children": [], "weight": 0.002},
    "SRE":  {"label": "Sempra",     "level": 5, "parent": "sector_util",   "ticker": "SRE",  "color": COLORS["sector"], "children": [], "weight": 0.003},
    "PEG":  {"label": "PSEG",       "level": 5, "parent": "sector_util",   "ticker": "PEG",  "color": COLORS["sector"], "children": [], "weight": 0.002},
    "AWK":  {"label": "Amer. Water","level": 5, "parent": "sector_util",   "ticker": "AWK",  "color": COLORS["sector"], "children": [], "weight": 0.002},
    "ED":   {"label": "Con Edison", "level": 5, "parent": "sector_util",   "ticker": "ED",   "color": COLORS["sector"], "children": [], "weight": 0.002},
    # Real Estate
    "AMT":  {"label": "Amer. Tower","level": 5, "parent": "sector_re",     "ticker": "AMT",  "color": COLORS["sector"], "children": [], "weight": 0.005},
    "PLD":  {"label": "Prologis",   "level": 5, "parent": "sector_re",     "ticker": "PLD",  "color": COLORS["sector"], "children": [], "weight": 0.005},
    "EQIX": {"label": "Equinix",    "level": 5, "parent": "sector_re",     "ticker": "EQIX", "color": COLORS["sector"], "children": [], "weight": 0.004},
    "CCI":  {"label": "Crown Castle","level": 5, "parent": "sector_re",    "ticker": "CCI",  "color": COLORS["sector"], "children": [], "weight": 0.003},
    "PSA":  {"label": "Pub. Storage","level": 5, "parent": "sector_re",    "ticker": "PSA",  "color": COLORS["sector"], "children": [], "weight": 0.003},
    "WELL": {"label": "Welltower",  "level": 5, "parent": "sector_re",     "ticker": "WELL", "color": COLORS["sector"], "children": [], "weight": 0.003},
    "O":    {"label": "Realty Inc.", "level": 5, "parent": "sector_re",    "ticker": "O",    "color": COLORS["sector"], "children": [], "weight": 0.003},
    "DLR":  {"label": "Digital Rty","level": 5, "parent": "sector_re",     "ticker": "DLR",  "color": COLORS["sector"], "children": [], "weight": 0.003},
    "SPG":  {"label": "Simon Prop.", "level": 5, "parent": "sector_re",    "ticker": "SPG",  "color": COLORS["sector"], "children": [], "weight": 0.003},
    "VICI": {"label": "VICI Prop.", "level": 5, "parent": "sector_re",     "ticker": "VICI", "color": COLORS["sector"], "children": [], "weight": 0.002},

    # ══════════════════════════════════════════════════════════════════════════
    # Nível 4: Setores Nasdaq 100 (GICS — pesos aproximados Mar/2025)
    # ══════════════════════════════════════════════════════════════════════════
    "ndx_sector_tech": {
        "label": "NDX Tech", "level": 4, "parent": "ndx100", "gics": "45",
        "color": COLORS["sector"], "weight": 0.51,
        "children": [
            "ndx_MSFT", "ndx_AAPL", "ndx_NVDA", "ndx_AVGO", "ndx_AMD",
            "ndx_ADBE", "ndx_QCOM", "ndx_INTC", "ndx_AMAT", "ndx_MU",
            "ndx_KLAC", "ndx_LRCX", "ndx_SNPS", "ndx_CDNS", "ndx_ON",
        ],
    },
    "ndx_sector_comm": {
        "label": "NDX Comm", "level": 4, "parent": "ndx100", "gics": "50",
        "color": COLORS["sector"], "weight": 0.16,
        "children": ["ndx_META", "ndx_GOOGL", "ndx_NFLX", "ndx_TMUS", "ndx_CHTR"],
    },
    "ndx_sector_cd": {
        "label": "NDX Cons.Disc", "level": 4, "parent": "ndx100", "gics": "25",
        "color": COLORS["sector"], "weight": 0.15,
        "children": ["ndx_AMZN", "ndx_TSLA", "ndx_BKNG", "ndx_SBUX", "ndx_ORLY", "ndx_ABNB", "ndx_LULU"],
    },
    "ndx_sector_health": {
        "label": "NDX Health", "level": 4, "parent": "ndx100", "gics": "35",
        "color": COLORS["sector"], "weight": 0.07,
        "children": ["ndx_AMGN", "ndx_VRTX", "ndx_GILD", "ndx_REGN", "ndx_ISRG", "ndx_DXCM", "ndx_IDXX", "ndx_BIIB"],
    },
    "ndx_sector_ind": {
        "label": "NDX Industrials", "level": 4, "parent": "ndx100", "gics": "20",
        "color": COLORS["sector"], "weight": 0.04,
        "children": ["ndx_HON", "ndx_CTAS", "ndx_FAST", "ndx_PAYX", "ndx_VRSK", "ndx_ODFL"],
    },
    "ndx_sector_cs": {
        "label": "NDX Cons.Staples", "level": 4, "parent": "ndx100", "gics": "30",
        "color": COLORS["sector"], "weight": 0.03,
        "children": ["ndx_COST", "ndx_PEP", "ndx_MDLZ", "ndx_KHC", "ndx_MNST"],
    },
    "ndx_sector_fin": {
        "label": "NDX Financials", "level": 4, "parent": "ndx100", "gics": "40",
        "color": COLORS["sector"], "weight": 0.02,
        "children": ["ndx_PYPL", "ndx_NDAQ"],
    },
    "ndx_sector_util": {
        "label": "NDX Utilities", "level": 4, "parent": "ndx100", "gics": "55",
        "color": COLORS["sector"], "weight": 0.01,
        "children": ["ndx_CEG", "ndx_XEL"],
    },

    # ── Nível 5: Holdings NDX100 — Technology (~51%) ──────────────────────────
    "ndx_MSFT": {"label": "Microsoft", "level": 5, "parent": "ndx_sector_tech",   "ticker": "MSFT", "color": COLORS["asset"], "children": [], "weight": 0.085},
    "ndx_AAPL": {"label": "Apple",     "level": 5, "parent": "ndx_sector_tech",   "ticker": "AAPL", "color": COLORS["asset"], "children": [], "weight": 0.082},
    "ndx_NVDA": {"label": "Nvidia",    "level": 5, "parent": "ndx_sector_tech",   "ticker": "NVDA", "color": COLORS["asset"], "children": [], "weight": 0.078},
    "ndx_AVGO": {"label": "Broadcom",  "level": 5, "parent": "ndx_sector_tech",   "ticker": "AVGO", "color": COLORS["asset"], "children": [], "weight": 0.028},
    "ndx_AMD":  {"label": "AMD",       "level": 5, "parent": "ndx_sector_tech",   "ticker": "AMD",  "color": COLORS["asset"], "children": [], "weight": 0.014},
    "ndx_ADBE": {"label": "Adobe",     "level": 5, "parent": "ndx_sector_tech",   "ticker": "ADBE", "color": COLORS["asset"], "children": [], "weight": 0.013},
    "ndx_QCOM": {"label": "Qualcomm",  "level": 5, "parent": "ndx_sector_tech",   "ticker": "QCOM", "color": COLORS["asset"], "children": [], "weight": 0.012},
    "ndx_INTC": {"label": "Intel",     "level": 5, "parent": "ndx_sector_tech",   "ticker": "INTC", "color": COLORS["asset"], "children": [], "weight": 0.008},
    "ndx_AMAT": {"label": "Appl.Matls","level": 5, "parent": "ndx_sector_tech",   "ticker": "AMAT", "color": COLORS["asset"], "children": [], "weight": 0.011},
    "ndx_MU":   {"label": "Micron",    "level": 5, "parent": "ndx_sector_tech",   "ticker": "MU",   "color": COLORS["asset"], "children": [], "weight": 0.010},
    "ndx_KLAC": {"label": "KLA Corp",  "level": 5, "parent": "ndx_sector_tech",   "ticker": "KLAC", "color": COLORS["asset"], "children": [], "weight": 0.009},
    "ndx_LRCX": {"label": "Lam Res.",  "level": 5, "parent": "ndx_sector_tech",   "ticker": "LRCX", "color": COLORS["asset"], "children": [], "weight": 0.008},
    "ndx_SNPS": {"label": "Synopsys",  "level": 5, "parent": "ndx_sector_tech",   "ticker": "SNPS", "color": COLORS["asset"], "children": [], "weight": 0.007},
    "ndx_CDNS": {"label": "Cadence",   "level": 5, "parent": "ndx_sector_tech",   "ticker": "CDNS", "color": COLORS["asset"], "children": [], "weight": 0.007},
    "ndx_ON":   {"label": "ON Semi",   "level": 5, "parent": "ndx_sector_tech",   "ticker": "ON",   "color": COLORS["asset"], "children": [], "weight": 0.005},
    # Communication Services (~16%)
    "ndx_META": {"label": "Meta",      "level": 5, "parent": "ndx_sector_comm",   "ticker": "META", "color": COLORS["asset"], "children": [], "weight": 0.040},
    "ndx_GOOGL":{"label": "Alphabet",  "level": 5, "parent": "ndx_sector_comm",   "ticker": "GOOGL","color": COLORS["asset"], "children": [], "weight": 0.035},
    "ndx_NFLX": {"label": "Netflix",   "level": 5, "parent": "ndx_sector_comm",   "ticker": "NFLX", "color": COLORS["asset"], "children": [], "weight": 0.015},
    "ndx_TMUS": {"label": "T-Mobile",  "level": 5, "parent": "ndx_sector_comm",   "ticker": "TMUS", "color": COLORS["asset"], "children": [], "weight": 0.012},
    "ndx_CHTR": {"label": "Charter",   "level": 5, "parent": "ndx_sector_comm",   "ticker": "CHTR", "color": COLORS["asset"], "children": [], "weight": 0.006},
    # Consumer Discretionary (~15%)
    "ndx_AMZN": {"label": "Amazon",    "level": 5, "parent": "ndx_sector_cd",     "ticker": "AMZN", "color": COLORS["asset"], "children": [], "weight": 0.048},
    "ndx_TSLA": {"label": "Tesla",     "level": 5, "parent": "ndx_sector_cd",     "ticker": "TSLA", "color": COLORS["asset"], "children": [], "weight": 0.025},
    "ndx_BKNG": {"label": "Booking",   "level": 5, "parent": "ndx_sector_cd",     "ticker": "BKNG", "color": COLORS["asset"], "children": [], "weight": 0.010},
    "ndx_SBUX": {"label": "Starbucks", "level": 5, "parent": "ndx_sector_cd",     "ticker": "SBUX", "color": COLORS["asset"], "children": [], "weight": 0.007},
    "ndx_ORLY": {"label": "O'Reilly",  "level": 5, "parent": "ndx_sector_cd",     "ticker": "ORLY", "color": COLORS["asset"], "children": [], "weight": 0.009},
    "ndx_ABNB": {"label": "Airbnb",    "level": 5, "parent": "ndx_sector_cd",     "ticker": "ABNB", "color": COLORS["asset"], "children": [], "weight": 0.007},
    "ndx_LULU": {"label": "Lululemon", "level": 5, "parent": "ndx_sector_cd",     "ticker": "LULU", "color": COLORS["asset"], "children": [], "weight": 0.006},
    # Healthcare (~7%)
    "ndx_AMGN": {"label": "Amgen",     "level": 5, "parent": "ndx_sector_health", "ticker": "AMGN", "color": COLORS["asset"], "children": [], "weight": 0.012},
    "ndx_VRTX": {"label": "Vertex",    "level": 5, "parent": "ndx_sector_health", "ticker": "VRTX", "color": COLORS["asset"], "children": [], "weight": 0.010},
    "ndx_GILD": {"label": "Gilead",    "level": 5, "parent": "ndx_sector_health", "ticker": "GILD", "color": COLORS["asset"], "children": [], "weight": 0.008},
    "ndx_REGN": {"label": "Regeneron", "level": 5, "parent": "ndx_sector_health", "ticker": "REGN", "color": COLORS["asset"], "children": [], "weight": 0.008},
    "ndx_ISRG": {"label": "Intuitive", "level": 5, "parent": "ndx_sector_health", "ticker": "ISRG", "color": COLORS["asset"], "children": [], "weight": 0.007},
    "ndx_DXCM": {"label": "Dexcom",    "level": 5, "parent": "ndx_sector_health", "ticker": "DXCM", "color": COLORS["asset"], "children": [], "weight": 0.004},
    "ndx_IDXX": {"label": "Idexx",     "level": 5, "parent": "ndx_sector_health", "ticker": "IDXX", "color": COLORS["asset"], "children": [], "weight": 0.004},
    "ndx_BIIB": {"label": "Biogen",    "level": 5, "parent": "ndx_sector_health", "ticker": "BIIB", "color": COLORS["asset"], "children": [], "weight": 0.004},
    # Industrials (~4%)
    "ndx_HON":  {"label": "Honeywell", "level": 5, "parent": "ndx_sector_ind",    "ticker": "HON",  "color": COLORS["asset"], "children": [], "weight": 0.008},
    "ndx_CTAS": {"label": "Cintas",    "level": 5, "parent": "ndx_sector_ind",    "ticker": "CTAS", "color": COLORS["asset"], "children": [], "weight": 0.007},
    "ndx_FAST": {"label": "Fastenal",  "level": 5, "parent": "ndx_sector_ind",    "ticker": "FAST", "color": COLORS["asset"], "children": [], "weight": 0.006},
    "ndx_PAYX": {"label": "Paychex",   "level": 5, "parent": "ndx_sector_ind",    "ticker": "PAYX", "color": COLORS["asset"], "children": [], "weight": 0.006},
    "ndx_VRSK": {"label": "Verisk",    "level": 5, "parent": "ndx_sector_ind",    "ticker": "VRSK", "color": COLORS["asset"], "children": [], "weight": 0.005},
    "ndx_ODFL": {"label": "Old Dom.",  "level": 5, "parent": "ndx_sector_ind",    "ticker": "ODFL", "color": COLORS["asset"], "children": [], "weight": 0.004},
    # Consumer Staples (~3%)
    "ndx_COST": {"label": "Costco",    "level": 5, "parent": "ndx_sector_cs",     "ticker": "COST", "color": COLORS["asset"], "children": [], "weight": 0.018},
    "ndx_PEP":  {"label": "PepsiCo",   "level": 5, "parent": "ndx_sector_cs",     "ticker": "PEP",  "color": COLORS["asset"], "children": [], "weight": 0.012},
    "ndx_MDLZ": {"label": "Mondelez",  "level": 5, "parent": "ndx_sector_cs",     "ticker": "MDLZ", "color": COLORS["asset"], "children": [], "weight": 0.005},
    "ndx_KHC":  {"label": "Kraft H.",  "level": 5, "parent": "ndx_sector_cs",     "ticker": "KHC",  "color": COLORS["asset"], "children": [], "weight": 0.003},
    "ndx_MNST": {"label": "Monster",   "level": 5, "parent": "ndx_sector_cs",     "ticker": "MNST", "color": COLORS["asset"], "children": [], "weight": 0.004},
    # Financials (~2%)
    "ndx_PYPL": {"label": "PayPal",    "level": 5, "parent": "ndx_sector_fin",    "ticker": "PYPL", "color": COLORS["asset"], "children": [], "weight": 0.005},
    "ndx_NDAQ": {"label": "Nasdaq Inc","level": 5, "parent": "ndx_sector_fin",    "ticker": "NDAQ", "color": COLORS["asset"], "children": [], "weight": 0.004},
    # Utilities (~1%)
    "ndx_CEG":  {"label": "Constell.", "level": 5, "parent": "ndx_sector_util",   "ticker": "CEG",  "color": COLORS["asset"], "children": [], "weight": 0.005},
    "ndx_XEL":  {"label": "Xcel",      "level": 5, "parent": "ndx_sector_util",   "ticker": "XEL",  "color": COLORS["asset"], "children": [], "weight": 0.003},

    # ══════════════════════════════════════════════════════════════════════════
    # Nível 4: Setores Russell 2000 (pesos aproximados Mar/2025)
    # ══════════════════════════════════════════════════════════════════════════
    "rut_sector_health": {
        "label": "RUT Health",    "level": 4, "parent": "russell2000", "gics": "35",
        "color": COLORS["sector"], "weight": 0.18,
        "children": ["rut_HIMS", "rut_ACAD", "rut_PRGO", "rut_ARWR", "rut_LGND", "rut_DNLI"],
    },
    "rut_sector_fin": {
        "label": "RUT Financials","level": 4, "parent": "russell2000", "gics": "40",
        "color": COLORS["sector"], "weight": 0.17,
        "children": ["rut_WSFS", "rut_CVBF", "rut_NBTB", "rut_BANF"],
    },
    "rut_sector_ind": {
        "label": "RUT Industrials","level": 4, "parent": "russell2000", "gics": "20",
        "color": COLORS["sector"], "weight": 0.16,
        "children": ["rut_AVAV", "rut_ATI", "rut_ESAB", "rut_ROLL", "rut_SSD", "rut_KTOS"],
    },
    "rut_sector_tech": {
        "label": "RUT Technology","level": 4, "parent": "russell2000", "gics": "45",
        "color": COLORS["sector"], "weight": 0.13,
        "children": ["rut_SANM", "rut_OSIS", "rut_KLIC", "rut_COHU", "rut_ACMR", "rut_SMTC"],
    },
    "rut_sector_cd": {
        "label": "RUT Cons.Disc", "level": 4, "parent": "russell2000", "gics": "25",
        "color": COLORS["sector"], "weight": 0.11,
        "children": ["rut_BOOT", "rut_SHAK", "rut_GMS", "rut_CHUY", "rut_PLAY"],
    },
    "rut_sector_re": {
        "label": "RUT Real Estate","level": 4, "parent": "russell2000", "gics": "60",
        "color": COLORS["sector"], "weight": 0.07,
        "children": ["rut_GTY", "rut_NXRT", "rut_BRT"],
    },
    "rut_sector_energy": {
        "label": "RUT Energy",    "level": 4, "parent": "russell2000", "gics": "10",
        "color": COLORS["commodity"], "weight": 0.06,
        "children": ["rut_MTDR", "rut_RES", "rut_SM"],
    },
    "rut_sector_mat": {
        "label": "RUT Materials", "level": 4, "parent": "russell2000", "gics": "15",
        "color": COLORS["sector"], "weight": 0.04,
        "children": ["rut_TREX", "rut_KALU", "rut_OMG", "rut_SXT"],
    },
    "rut_sector_comm": {
        "label": "RUT Comm.",     "level": 4, "parent": "russell2000", "gics": "50",
        "color": COLORS["sector"], "weight": 0.03,
        "children": ["rut_GOGO", "rut_AMCX", "rut_MSGS"],
    },
    "rut_sector_cs": {
        "label": "RUT Cons.Stpls","level": 4, "parent": "russell2000", "gics": "30",
        "color": COLORS["sector"], "weight": 0.03,
        "children": ["rut_INGR", "rut_CENT", "rut_HAIN"],
    },
    "rut_sector_util": {
        "label": "RUT Utilities", "level": 4, "parent": "russell2000", "gics": "55",
        "color": COLORS["sector"], "weight": 0.02,
        "children": ["rut_EE", "rut_MGEE", "rut_SPWH"],
    },

    # ── Nível 5: Holdings Russell 2000 (prefixo rut_) ─────────────────────────
    # Healthcare (~18%)
    "rut_HIMS":  {"label": "Hims&Hers",  "level": 5, "parent": "rut_sector_health", "ticker": "HIMS", "color": COLORS["asset"], "children": [], "weight": 0.005},
    "rut_ACAD":  {"label": "Acadia Ph.", "level": 5, "parent": "rut_sector_health", "ticker": "ACAD", "color": COLORS["asset"], "children": [], "weight": 0.004},
    "rut_PRGO":  {"label": "Perrigo",    "level": 5, "parent": "rut_sector_health", "ticker": "PRGO", "color": COLORS["asset"], "children": [], "weight": 0.004},
    "rut_ARWR":  {"label": "Arrowhead", "level": 5, "parent": "rut_sector_health",  "ticker": "ARWR", "color": COLORS["asset"], "children": [], "weight": 0.003},
    "rut_LGND":  {"label": "Ligand",     "level": 5, "parent": "rut_sector_health", "ticker": "LGND", "color": COLORS["asset"], "children": [], "weight": 0.003},
    "rut_DNLI":  {"label": "Denali",     "level": 5, "parent": "rut_sector_health", "ticker": "DNLI", "color": COLORS["asset"], "children": [], "weight": 0.002},
    # Financials (~17%)
    "rut_WSFS":  {"label": "WSFS Fin.", "level": 5, "parent": "rut_sector_fin",    "ticker": "WSFS", "color": COLORS["asset"], "children": [], "weight": 0.004},
    "rut_CVBF":  {"label": "CVB Fin.",  "level": 5, "parent": "rut_sector_fin",    "ticker": "CVBF", "color": COLORS["asset"], "children": [], "weight": 0.003},
    "rut_NBTB":  {"label": "NBT Bancp","level": 5, "parent": "rut_sector_fin",     "ticker": "NBTB", "color": COLORS["asset"], "children": [], "weight": 0.003},
    "rut_BANF":  {"label": "BancFirst", "level": 5, "parent": "rut_sector_fin",    "ticker": "BANF", "color": COLORS["asset"], "children": [], "weight": 0.003},
    # Industrials (~16%)
    "rut_AVAV":  {"label": "AeroVirmt", "level": 5, "parent": "rut_sector_ind",    "ticker": "AVAV", "color": COLORS["asset"], "children": [], "weight": 0.005},
    "rut_ATI":   {"label": "ATI Inc.",  "level": 5, "parent": "rut_sector_ind",    "ticker": "ATI",  "color": COLORS["asset"], "children": [], "weight": 0.005},
    "rut_ESAB":  {"label": "ESAB Corp","level": 5, "parent": "rut_sector_ind",     "ticker": "ESAB", "color": COLORS["asset"], "children": [], "weight": 0.004},
    "rut_ROLL":  {"label": "RBC Bear.", "level": 5, "parent": "rut_sector_ind",    "ticker": "ROLL", "color": COLORS["asset"], "children": [], "weight": 0.004},
    "rut_SSD":   {"label": "Simpson",   "level": 5, "parent": "rut_sector_ind",    "ticker": "SSD",  "color": COLORS["asset"], "children": [], "weight": 0.004},
    "rut_KTOS":  {"label": "Kratos",    "level": 5, "parent": "rut_sector_ind",    "ticker": "KTOS", "color": COLORS["asset"], "children": [], "weight": 0.003},
    # Technology (~13%)
    "rut_SANM":  {"label": "Sanmina",   "level": 5, "parent": "rut_sector_tech",   "ticker": "SANM", "color": COLORS["asset"], "children": [], "weight": 0.004},
    "rut_OSIS":  {"label": "OSI Sys.",  "level": 5, "parent": "rut_sector_tech",   "ticker": "OSIS", "color": COLORS["asset"], "children": [], "weight": 0.003},
    "rut_KLIC":  {"label": "Kulicke",   "level": 5, "parent": "rut_sector_tech",   "ticker": "KLIC", "color": COLORS["asset"], "children": [], "weight": 0.003},
    "rut_COHU":  {"label": "Cohu",      "level": 5, "parent": "rut_sector_tech",   "ticker": "COHU", "color": COLORS["asset"], "children": [], "weight": 0.002},
    "rut_ACMR":  {"label": "ACM Res.",  "level": 5, "parent": "rut_sector_tech",   "ticker": "ACMR", "color": COLORS["asset"], "children": [], "weight": 0.002},
    "rut_SMTC":  {"label": "Semtech",   "level": 5, "parent": "rut_sector_tech",   "ticker": "SMTC", "color": COLORS["asset"], "children": [], "weight": 0.002},
    # Consumer Discretionary (~11%)
    "rut_BOOT":  {"label": "Boot Barn", "level": 5, "parent": "rut_sector_cd",     "ticker": "BOOT", "color": COLORS["asset"], "children": [], "weight": 0.005},
    "rut_SHAK":  {"label": "Shake Shk","level": 5, "parent": "rut_sector_cd",      "ticker": "SHAK", "color": COLORS["asset"], "children": [], "weight": 0.003},
    "rut_GMS":   {"label": "GMS Inc.", "level": 5, "parent": "rut_sector_cd",      "ticker": "GMS",  "color": COLORS["asset"], "children": [], "weight": 0.004},
    "rut_CHUY":  {"label": "Chuy's",   "level": 5, "parent": "rut_sector_cd",      "ticker": "CHUY", "color": COLORS["asset"], "children": [], "weight": 0.002},
    "rut_PLAY":  {"label": "Dave&Bust","level": 5, "parent": "rut_sector_cd",      "ticker": "PLAY", "color": COLORS["asset"], "children": [], "weight": 0.003},
    # Real Estate (~7%)
    "rut_GTY":   {"label": "Getty Rty","level": 5, "parent": "rut_sector_re",      "ticker": "GTY",  "color": COLORS["asset"], "children": [], "weight": 0.003},
    "rut_NXRT":  {"label": "NexPoint", "level": 5, "parent": "rut_sector_re",      "ticker": "NXRT", "color": COLORS["asset"], "children": [], "weight": 0.002},
    "rut_BRT":   {"label": "BRT Rlty.", "level": 5, "parent": "rut_sector_re",     "ticker": "BRT",  "color": COLORS["asset"], "children": [], "weight": 0.002},
    # Energy (~6%)
    "rut_MTDR":  {"label": "Matador",   "level": 5, "parent": "rut_sector_energy", "ticker": "MTDR", "color": COLORS["commodity"], "children": [], "weight": 0.004},
    "rut_RES":   {"label": "RPC Inc.",  "level": 5, "parent": "rut_sector_energy", "ticker": "RES",  "color": COLORS["commodity"], "children": [], "weight": 0.002},
    "rut_SM":    {"label": "SM Energy", "level": 5, "parent": "rut_sector_energy", "ticker": "SM",   "color": COLORS["commodity"], "children": [], "weight": 0.003},
    # Materials (~4%)
    "rut_TREX":  {"label": "Trex",      "level": 5, "parent": "rut_sector_mat",    "ticker": "TREX", "color": COLORS["sector"], "children": [], "weight": 0.004},
    "rut_KALU":  {"label": "Kaiser Al.","level": 5, "parent": "rut_sector_mat",    "ticker": "KALU", "color": COLORS["sector"], "children": [], "weight": 0.003},
    "rut_OMG":   {"label": "OM Group", "level": 5, "parent": "rut_sector_mat",     "ticker": "OMG",  "color": COLORS["sector"], "children": [], "weight": 0.002},
    "rut_SXT":   {"label": "Sensient",  "level": 5, "parent": "rut_sector_mat",    "ticker": "SXT",  "color": COLORS["sector"], "children": [], "weight": 0.002},
    # Communication (~3%)
    "rut_GOGO":  {"label": "Gogo",      "level": 5, "parent": "rut_sector_comm",   "ticker": "GOGO", "color": COLORS["sector"], "children": [], "weight": 0.002},
    "rut_AMCX":  {"label": "AMC Ntwk.", "level": 5, "parent": "rut_sector_comm",   "ticker": "AMCX", "color": COLORS["sector"], "children": [], "weight": 0.002},
    "rut_MSGS":  {"label": "MSG Sports","level": 5, "parent": "rut_sector_comm",   "ticker": "MSGS", "color": COLORS["sector"], "children": [], "weight": 0.002},
    # Consumer Staples (~3%)
    "rut_INGR":  {"label": "Ingredion", "level": 5, "parent": "rut_sector_cs",     "ticker": "INGR", "color": COLORS["sector"], "children": [], "weight": 0.004},
    "rut_CENT":  {"label": "Central G.","level": 5, "parent": "rut_sector_cs",     "ticker": "CENT", "color": COLORS["sector"], "children": [], "weight": 0.003},
    "rut_HAIN":  {"label": "Hain Cel.", "level": 5, "parent": "rut_sector_cs",     "ticker": "HAIN", "color": COLORS["sector"], "children": [], "weight": 0.002},
    # Utilities (~2%)
    "rut_EE":    {"label": "Empire El.","level": 5, "parent": "rut_sector_util",   "ticker": "EE",   "color": COLORS["sector"], "children": [], "weight": 0.003},
    "rut_MGEE":  {"label": "MGE Energy","level": 5, "parent": "rut_sector_util",   "ticker": "MGEE", "color": COLORS["sector"], "children": [], "weight": 0.003},
    "rut_SPWH":  {"label": "Sportsman","level": 5, "parent": "rut_sector_util",    "ticker": "SPWH", "color": COLORS["sector"], "children": [], "weight": 0.002},
}

# ── Internal layers por ativo (Nível 7) ───────────────────────────────────────
INTERNAL_LAYERS = ["valuation", "volatility", "options", "risk", "flow", "macro_sens"]

INTERNAL_LAYER_LABELS = {
    "valuation":  "Valuation",
    "volatility": "Volatility / Vol Surface",
    "options":    "Options / Greeks",
    "risk":       "Risk / CVaR",
    "flow":       "Flow / Gamma",
    "macro_sens": "Macro Sensitivity",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_node(node_id: str) -> dict[str, Any] | None:
    return NODES.get(node_id)

def get_children(node_id: str) -> list[str]:
    node = NODES.get(node_id)
    return node.get("children", []) if node else []

def get_ancestors(node_id: str) -> list[str]:
    """Retorna caminho do nó até World."""
    path = []
    current = node_id
    while current:
        path.append(current)
        node = NODES.get(current)
        current = node.get("parent") if node else None
    return list(reversed(path))

def ticker_to_node_id(ticker: str) -> str | None:
    """Encontra node_id pelo ticker."""
    for nid, node in NODES.items():
        if node.get("ticker") == ticker:
            return nid
    return ticker if ticker in NODES else None

def nodes_at_level(level: int) -> list[str]:
    return [nid for nid, n in NODES.items() if n.get("level") == level]

def visible_subgraph(root_id: str, max_depth: int = 2) -> list[str]:
    """Retorna ids de nós visíveis a partir de um root, até max_depth níveis."""
    visited = []
    queue = [(root_id, 0)]
    while queue:
        nid, depth = queue.pop(0)
        if nid in visited:
            continue
        visited.append(nid)
        if depth < max_depth:
            for child in get_children(nid):
                if child not in visited:
                    queue.append((child, depth + 1))
    return visited
