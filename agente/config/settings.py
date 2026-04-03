"""
MacroDesk Bloomberg Ecosystem — Configuração Central
=====================================================

Todos os caminhos, constantes e parâmetros do ecossistema estão aqui.
Altere apenas este arquivo para ajustar caminhos ou comportamentos globais.
"""

from pathlib import Path

# ── Raiz do Projeto ───────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()          # .../bbg/agente/

# ── Pastas de Dados ───────────────────────────────────────────────────────────
BQL_DATA_DIR  = ROOT / "bql_data"                      # CSVs extraídos (trilha de auditoria)
DATABASE_DIR  = ROOT / "data" / "database"
DATABASE_PATH = DATABASE_DIR / "macrodesk.db"          # ÚNICA fonte de verdade
LOGS_DIR      = ROOT / "data" / "logs"

# ── Downloads (onde Bloomberg exporta os .zip) ───────────────────────────────
DOWNLOADS_DIR = Path.home() / "Downloads"
ZIP_PATTERN   = "bql_data_*.zip"                       # padrão dos exports do BQuant

# ── Controle de Estado ────────────────────────────────────────────────────────
# Arquivo que rastreia o último zip processado (compatibilidade com bql_unzip legado)
STATE_FILE    = BQL_DATA_DIR / ".last_zip"

# ── Freshness ─────────────────────────────────────────────────────────────────
STALE_MINUTES = 60    # avisar se ingestão mais antiga que este valor

# ── Datasets Bloomberg Conhecidos ─────────────────────────────────────────────
# Usado para detecção automática do tipo de CSV pelo prefixo do nome do arquivo.
KNOWN_DATASETS: dict[str, str] = {
    "prices":          "Preços e retornos por ativo",
    "price_history":   "Histórico de fechamento diário",
    "fundamentals":    "Múltiplos e métricas fundamentalistas",
    "options_iv":      "Volatilidade implícita e skew de opções",
    "gex_summary":     "Resumo de Gamma Exposure (GEX) do SPX",
    "gex_spx":         "GEX por strike do SPX",
    "letf_flows":      "Fluxos e AUM de ETFs alavancados",
    "meta":            "Metadados do export (timestamp BQuant)",
}

# ── Campos Bloomberg → label legível ─────────────────────────────────────────
FIELD_LABELS: dict[str, str] = {
    "price":           "Preço",
    "prev_price":      "Fechamento anterior",
    "price_w":         "Preço 5d atrás (referência semanal)",
    "price_ytd":       "Preço 1-Jan (referência YTD)",
    "pe":              "P/E",
    "mktcap_b":        "Market Cap (B)",
    "beta":            "Beta",
    "profit_margin":   "Margem de Lucro",
    "debt_equity":     "Dívida/Patrimônio",
    "roe":             "ROE",
    "dividend_yield":  "Dividend Yield",
    "hi_52w":          "Máxima 52 semanas",
    "lo_52w":          "Mínima 52 semanas",
    "drawdown_52w":    "Drawdown 52 semanas",
    "atm_iv":          "IV ATM",
    "skew_25d":        "Skew 25-Delta",
    "pcr_oi":          "Put/Call OI Ratio",
    "spot":            "Spot",
    "gex_total_bn":    "GEX Total (BN)",
    "gex_call_bn":     "GEX Call (BN)",
    "gex_put_bn":      "GEX Put (BN)",
    "direction":       "Direção GEX",
    "gamma_regime":    "Regime Gamma",
    "n_options":       "N Opções",
    "nav":             "NAV",
    "aum_b":           "AUM (B)",
    "leverage":        "Alavancagem",
}

# ── Índices Macro Principais (para Risk Radar / Monte Carlo no MacroDesk) ────
MACRO_INDICES: dict[str, str] = {
    "^GSPC":    "S&P 500",
    "^NDX":     "Nasdaq 100",
    "^DJI":     "Dow Jones",
    "^RUT":     "Russell 2000",
    "^VIX":     "VIX",
    "DX-Y.NYB": "DXY",
    "^TNX":     "Treasury 10yr",
    "TLT":      "Treasury 20yr (TLT)",
    "GLD":      "Gold",
    "CL=F":     "WTI Crude",
}
