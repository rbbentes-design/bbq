"""
Provider: Damodaran (NYU Stern)

Coleta dados de avaliação do Prof. Aswath Damodaran (NYU Stern).
Arquivos Excel atualizados na primeira semana de janeiro de cada ano.

Datasets coletados:
  - Implied ERP (Equity Risk Premium) — histórico anual + valor atual
  - Country Risk Premiums — CRP por país com rating Moody's
  - Cost of Capital por setor (EUA) — WACC, beta, spread de dívida

Site: https://pages.stern.nyu.edu/~adamodar/
"""

from __future__ import annotations

import io
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from app.audit.logger import get_logger
from app.config.settings import settings

_log = get_logger("providers.damodaran")

_BASE = "https://pages.stern.nyu.edu/~adamodar/pc/datasets"

# URLs dos datasets
_URLS: dict[str, str] = {
    "erp_implied":     f"{_BASE}/histimpl.xls",
    "country_risk":    f"{_BASE}/ctryprem.xlsx",
    "wacc_us":         f"{_BASE}/wacc.xls",
}

# Cache em disco — dados mudam 1x ao ano
_CACHE_MAX_AGE_DAYS = 30


def collect() -> dict[str, Any]:
    """
    Retorna dados consolidados do Damodaran:
      - erp_current: ERP implícito mais recente (S&P 500)
      - erp_history: série histórica de ERP implícito
      - country_risk: lista de países com CRP e rating
      - wacc_sectors: lista de setores com WACC, beta, cost of equity

    Returns:
        Dict com as três categorias. Seções ausentes ficam vazias se o
        download ou parsing falhar (não levanta exceção).
    """
    output: dict[str, Any] = {
        "erp_current": None,
        "erp_history": [],
        "country_risk": [],
        "wacc_sectors": [],
        "source_date": date.today().isoformat(),
    }

    try:
        import pandas as pd  # noqa: F401 — verificar disponibilidade
    except ImportError:
        _log.warning("damodaran_pandas_missing")
        return output

    # ── ERP Implícito ──────────────────────────────────────────────────────────
    try:
        erp_data = _fetch_erp_implied()
        output["erp_current"] = erp_data.get("current")
        output["erp_history"] = erp_data.get("history", [])
        _log.info("damodaran_erp_ok", current=output["erp_current"])
    except Exception as exc:
        _log.warning("damodaran_erp_error", error=str(exc))

    # ── Country Risk Premiums ──────────────────────────────────────────────────
    try:
        output["country_risk"] = _fetch_country_risk()
        _log.info("damodaran_country_risk_ok", countries=len(output["country_risk"]))
    except Exception as exc:
        _log.warning("damodaran_country_risk_error", error=str(exc))

    # ── WACC por Setor (EUA) ───────────────────────────────────────────────────
    try:
        output["wacc_sectors"] = _fetch_wacc_sectors()
        _log.info("damodaran_wacc_ok", sectors=len(output["wacc_sectors"]))
    except Exception as exc:
        _log.warning("damodaran_wacc_error", error=str(exc))

    return output


# ── Internos ───────────────────────────────────────────────────────────────────

def _download(url: str, cache_name: str) -> bytes:
    """Baixa arquivo com cache em disco."""
    cache_dir = settings.resolved_raw_dir() / "damodaran"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / cache_name

    # Usa cache se recente
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < _CACHE_MAX_AGE_DAYS * 86400:
            _log.debug("damodaran_cache_hit", file=cache_name)
            return cache_file.read_bytes()

    _log.info("damodaran_download", url=url)
    headers = {"User-Agent": "Mozilla/5.0 (educational research tool)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()

    cache_file.write_bytes(data)
    return data


def _fetch_erp_implied() -> dict[str, Any]:
    """
    Extrai ERP implícito do histimpl.xls.

    Usa 'Implied ERP (FCFE)' — medida padrão citada por Damodaran em relatórios.
    Aba: 'Historical Impl Premiums', header na linha 6.
    """
    import pandas as pd

    data = _download(_URLS["erp_implied"], "histimpl.xls")
    xl = pd.ExcelFile(io.BytesIO(data), engine="xlrd")

    # Aba principal conhecida; fallback para search
    target_sheet = "Historical Impl Premiums"
    if target_sheet not in xl.sheet_names:
        for sheet in xl.sheet_names:
            if any(kw in sheet.lower() for kw in ("implied", "historical", "premium")):
                target_sheet = sheet
                break
        else:
            target_sheet = xl.sheet_names[0]

    df = xl.parse(target_sheet, header=None)

    # Encontra linha de cabeçalho procurando "Year"
    header_row = None
    for i, row in df.iterrows():
        row_lower = [str(c).lower().strip() for c in row]
        if "year" in row_lower:
            header_row = i
            break

    if header_row is None:
        raise ValueError(f"Header 'Year' não encontrado em {target_sheet}")

    df.columns = [str(c).strip() for c in df.iloc[header_row]]
    df = df.iloc[header_row + 1:].copy()

    # Mantém somente linhas com Year numérico (1960–2030)
    def _is_year(v: Any) -> bool:
        try:
            y = int(float(str(v)))
            return 1960 <= y <= 2030
        except Exception:
            return False

    df = df[df["Year"].apply(_is_year)]

    # Colunas preferidas: FCFE é a medida padrão citada por Damodaran
    _FCFE_COL  = "Implied ERP (FCFE)"
    _DDM_COL   = "Implied Premium (DDM)"
    _TBOND_COL = "T.Bond Rate"

    erp_col = _FCFE_COL if _FCFE_COL in df.columns else _DDM_COL

    history = []
    for _, row in df.iterrows():
        year  = _safe_int(row.get("Year"))
        erp   = _safe_float(row.get(erp_col))
        tbond = _safe_float(row.get(_TBOND_COL))
        ddm   = _safe_float(row.get(_DDM_COL)) if erp_col != _DDM_COL else None

        if year is None or erp is None:
            continue

        entry: dict[str, Any] = {
            "year":          year,
            "erp_pct":       round(erp * 100, 2) if erp < 1 else round(erp, 2),
            "erp_method":    "FCFE",
            "t_bond_rate_pct": round(tbond * 100, 2) if tbond and tbond < 1 else (
                round(tbond, 2) if tbond else None
            ),
        }
        if ddm is not None:
            entry["erp_ddm_pct"] = round(ddm * 100, 2) if ddm < 1 else round(ddm, 2)
        history.append(entry)

    current = history[-1] if history else None
    return {"current": current, "history": history[-20:]}


def _fetch_country_risk() -> list[dict]:
    """
    Extrai Country Risk Premiums do ctryprem.xlsx.

    Aba: 'ERPs by country', header na linha 7.
    Colunas: Country, Region, Moody's Rating, Default Spread,
             Total ERP, Country Risk Premium, CDS spread, ...
    """
    import pandas as pd

    data = _download(_URLS["country_risk"], "ctryprem.xlsx")
    xl = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")

    # Aba principal conhecida
    target_sheet = "ERPs by country"
    if target_sheet not in xl.sheet_names:
        for sheet in xl.sheet_names:
            if "erp" in sheet.lower() or "country" in sheet.lower():
                target_sheet = sheet
                break
        else:
            target_sheet = xl.sheet_names[0]

    df = xl.parse(target_sheet, header=None)

    # Encontra cabeçalho procurando linha com "Country" e "Rating"
    header_row = _find_header_row(df, ["country", "rating", "spread", "premium"])
    if header_row is None:
        raise ValueError("Cabeçalho de country risk não encontrado")

    raw_cols = list(df.iloc[header_row])
    df = df.iloc[header_row + 1:].copy()
    df.columns = range(len(raw_cols))

    # Mapeamento por posição (estrutura conhecida do arquivo Jan 2026):
    # 0=Country, 1=Region, 2=Moody's, 3=Default Spread, 4=Total ERP, 5=CRP, ...
    COL_COUNTRY = 0
    COL_REGION  = 1
    COL_RATING  = 2
    COL_DEF_SPD = 3
    COL_TOTAL_ERP = 4
    COL_CRP     = 5

    results = []
    for _, row in df.iterrows():
        country = str(row.get(COL_COUNTRY, "")).strip()
        if not country or country.lower() in ("nan", "country", ""):
            continue
        # Pula linhas de header repetido ou total
        if any(kw in country.lower() for kw in ("country and equity", "enter the", "do you want")):
            continue

        entry: dict[str, Any] = {"country": country}

        region = str(row.get(COL_REGION, "")).strip()
        if region and region.lower() not in ("nan", ""):
            entry["region"] = region

        rating = str(row.get(COL_RATING, "")).strip()
        if rating and rating.lower() not in ("nan", ""):
            entry["moody_rating"] = rating

        total_erp = _safe_float_pct(row.get(COL_TOTAL_ERP))
        crp       = _safe_float_pct(row.get(COL_CRP))
        def_spd   = _safe_float_pct(row.get(COL_DEF_SPD))

        if total_erp is not None:
            entry["total_erp_pct"] = total_erp
        if crp is not None:
            entry["country_risk_premium_pct"] = crp
        if def_spd is not None:
            entry["default_spread_pct"] = def_spd

        # Só inclui países com algum dado de risco
        if crp is not None or total_erp is not None:
            results.append(entry)

    return results


def _fetch_wacc_sectors() -> list[dict]:
    """
    Extrai Cost of Capital por setor do wacc.xls (EUA).

    Colunas: Industry, # Firms, Beta, D/E Ratio, Cost of Equity,
             AT Cost of Debt, Tax Rate, WACC, EV/EBITDA, EV/Sales
    """
    import pandas as pd

    data = _download(_URLS["wacc_us"], "wacc.xls")
    xl = pd.ExcelFile(io.BytesIO(data), engine="xlrd")

    target_sheet = xl.sheet_names[0]
    for sheet in xl.sheet_names:
        if any(kw in sheet.lower() for kw in ("wacc", "capital", "industry")):
            target_sheet = sheet
            break

    df = xl.parse(target_sheet, header=None)

    header_row = _find_header_row(df, ["industry", "wacc", "beta", "firms"])
    if header_row is None:
        raise ValueError("Cabeçalho de WACC não encontrado")

    df.columns = [str(c).strip() for c in df.iloc[header_row]]
    df = df.iloc[header_row + 1:].copy()
    df = df.dropna(subset=[df.columns[0]])

    col_map = _normalize_columns(df.columns, {
        "industry":      ["industry", "industry name", "sector"],
        "n_firms":       ["number of firms", "# firms", "n", "firms"],
        "beta":          ["beta", "unlevered beta", "levered beta"],
        "de_ratio":      ["d/e ratio", "debt/equity", "d/e"],
        "cost_equity":   ["cost of equity", "ke"],
        "cost_debt_at":  ["after-tax cost of debt", "at cost of debt", "kd"],
        "tax_rate":      ["effective tax rate", "tax rate"],
        "wacc":          ["wacc", "cost of capital"],
        "ev_ebitda":     ["ev/ebitda", "enterprise value"],
    })

    results = []
    for _, row in df.iterrows():
        industry = str(row.get(col_map.get("industry", ""), "")).strip()
        if not industry or industry.lower() in ("nan", "industry name", "total market"):
            continue
        # Ignora linhas de total/aggregate
        if industry.lower().startswith("total") and "market" in industry.lower():
            continue

        entry: dict[str, Any] = {"industry": industry}

        for field, key in [
            ("n_firms", "n_firms"),
            ("beta", "beta"),
            ("de_ratio_pct", "de_ratio"),
            ("cost_equity_pct", "cost_equity"),
            ("cost_debt_at_pct", "cost_debt_at"),
            ("tax_rate_pct", "tax_rate"),
            ("wacc_pct", "wacc"),
            ("ev_ebitda", "ev_ebitda"),
        ]:
            raw = row.get(col_map.get(key, ""))
            if field == "n_firms":
                entry[field] = _safe_int(raw)
            elif field in ("beta", "ev_ebitda"):
                entry[field] = _safe_float(raw)
            else:
                entry[field] = _safe_float_pct(raw)

        if entry.get("wacc_pct") is not None:
            results.append(entry)

    return results


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_header_row(df: Any, keywords: list[str]) -> int | None:
    """Encontra índice da linha de cabeçalho buscando keywords."""
    for i, row in df.iterrows():
        row_lower = " ".join(str(c).lower() for c in row)
        if sum(1 for kw in keywords if kw in row_lower) >= 2:
            return i
    return None


def _normalize_columns(
    columns: Any,
    field_candidates: dict[str, list[str]],
) -> dict[str, str]:
    """
    Mapeia nome canônico → nome real da coluna.
    Retorna apenas os campos encontrados.
    """
    col_lower = {str(c).lower().strip(): str(c) for c in columns}
    result: dict[str, str] = {}
    for field, candidates in field_candidates.items():
        for cand in candidates:
            # Busca por substring
            for col_l, col_real in col_lower.items():
                if cand in col_l:
                    result[field] = col_real
                    break
            if field in result:
                break
    return result


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return None if f != f else f  # NaN check
    except (ValueError, TypeError):
        return None


def _safe_float_pct(v: Any) -> float | None:
    """Converte para percentual: 0.054 → 5.4, 5.4 → 5.4."""
    f = _safe_float(v)
    if f is None:
        return None
    # Damodaran usa decimal (0.054) para percentuais
    if abs(f) < 1.0 and f != 0:
        return round(f * 100, 2)
    return round(f, 2)


def _safe_int(v: Any) -> int | None:
    try:
        return int(float(str(v)))
    except (ValueError, TypeError):
        return None
