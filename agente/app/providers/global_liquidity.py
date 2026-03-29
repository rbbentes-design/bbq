"""
Provider: Global Liquidity

Rastreia liquidez global nos moldes do CapitalWars (Michael Howell):
  - Net Fed Liquidity (Fed BS - RRP - TGA) — o indicador mais observado
  - Money Market Funds (fluxo de caixa dos investidores)
  - Balanços dos principais bancos centrais: Fed, ECB, BoJ, PBoC
  - M2 global: EUA, Zona do Euro, Japão, China
  - Proxy de Liquidez Global (soma dos balanços em USD)

Fontes:
  - FRED (Federal Reserve St. Louis) — EUA + séries internacionais disponíveis
  - ECB Statistical Data Warehouse (SDW) — ECB / Zona do Euro
  - Bank of Japan API — BoJ

Referência: https://capitalwars.substack.com/p/global-liquidity
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.parse
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from app.audit.logger import get_logger
from app.config.settings import settings

_log = get_logger("providers.global_liquidity")

_FRED_BASE = "https://api.stlouisfed.org/fred"
_ECB_BASE  = "https://data-api.ecb.europa.eu/service/data"
_CACHE_MAX_AGE_DAYS = 1  # liquidez muda diariamente

# ── Séries FRED — Liquidez EUA ─────────────────────────────────────────────────

_FRED_US_LIQUIDITY: list[tuple[str, str, str, float]] = [
    # (series_id, label, unit_display, factor_to_billions)
    # WALCL e WTREGEN são reportados em Milhões no FRED → dividir por 1000
    ("WALCL",           "Fed Balance Sheet (Total Assets)",  "USD bi", 0.001),
    ("RRPONTSYD",       "Fed Overnight Reverse Repo (RRP)",  "USD bi", 1.0),
    ("WTREGEN",         "Treasury General Account (TGA)",    "USD bi", 0.001),
    ("M2SL",            "US M2 Money Stock",                 "USD bi", 1.0),
    ("BOGMBASE",        "US Monetary Base",                  "USD bi", 1.0),
    ("WRMFSL",          "Money Market Funds — Retail",       "USD bi", 1.0),
    ("WRMFNS",          "Money Market Funds — Institutional","USD bi", 1.0),
    ("TOTBKCR",         "Total Bank Credit",                 "USD bi", 1.0),
    ("DPSACBW027SBOG",  "Bank Deposits (total)",             "USD bi", 1.0),
]

# ── Séries FRED — Global ───────────────────────────────────────────────────────

_FRED_GLOBAL: list[tuple[str, str, str, float]] = [
    # (series_id, label, unit_original, factor_to_billions_local_currency)
    # M2 por país (em moeda local — convertido para USD adiante)
    ("MYAGM2CNM189N",   "China M2",         "CNY bi",  1.0),
    ("MABMM301JPM189S", "Japan M2",         "JPY bi",  1.0),
    ("MABMM301EZM189S", "Euro Area M2",     "EUR bi",  1.0),
    ("MABMM301GBM189S", "UK M2",            "GBP bi",  1.0),
    # Balanços BCx estrangeiros (via FRED — confirmados)
    ("ECBASSETSW",      "ECB Total Assets", "EUR bi",  0.001),  # FRED: milhões EUR → /1000 = bilhões EUR
    ("JPNASSETS",       "BoJ Total Assets", "JPY bi",  0.1),    # FRED: 億円 (100M JPY) → *0.1 = bilhões JPY
    # FX diários (todos em relação ao USD)
    ("DEXUSEU",         "EUR/USD",          "USD/EUR", 1.0),   # USD por 1 EUR
    ("DEXJPUS",         "JPY/USD",          "JPY/USD", 1.0),   # JPY por 1 USD
    ("DEXCHUS",         "CNY/USD",          "CNY/USD", 1.0),   # CNY por 1 USD
    ("DEXUSUK",         "GBP/USD",          "USD/GBP", 1.0),   # USD por 1 GBP
]

# ECB M3 via ECB SDW (apenas M3 — BSI dataset confirmado funcional)
_ECB_M3_KEY = "BSI.M.U2.N.V.M30.X.1.U2.2300.Z01.E"


# ── Public API ─────────────────────────────────────────────────────────────────

def collect(lookback_days: int = 365) -> dict[str, Any]:
    """
    Coleta todos os indicadores de liquidez global.

    Returns:
        {
          "us_liquidity": {...},     # indicadores FRED EUA
          "net_fed_liquidity": [...], # série computada: Fed BS - RRP - TGA
          "money_market": {...},     # MMF retail + institucional + total
          "ecb": {...},              # balanço ECB + M3
          "global_m2": {...},        # M2 por país + global proxy em USD
          "global_balance_sheets": {...},  # balanços BCx em USD
          "summary": {...},          # últimos valores computados
        }
    """
    fred_key = settings.fred_api_key
    output: dict[str, Any] = {
        "us_liquidity": {},
        "net_fed_liquidity": [],
        "money_market": {},
        "ecb": {},
        "global_m2": {},
        "global_balance_sheets": {},
        "summary": {},
    }

    start = (date.today() - timedelta(days=lookback_days)).isoformat()

    # ── 1. FRED — Liquidez EUA ─────────────────────────────────────────────────
    us_raw: dict[str, list[dict]] = {}
    if fred_key:
        for series_id, label, unit, factor in _FRED_US_LIQUIDITY:
            obs = _fred_observations(series_id, start, fred_key)
            if obs:
                # Normaliza para bilhões
                obs = [{"date": o["date"],
                        "value": round(o["value"] * factor, 2) if o["value"] is not None else None}
                       for o in obs]
                us_raw[series_id] = obs
                latest = obs[-1]
                prev   = obs[-2] if len(obs) >= 2 else None
                output["us_liquidity"][series_id] = {
                    "label":      label,
                    "unit":       unit,
                    "value":      latest["value"],
                    "date":       latest["date"],
                    "prev_value": prev["value"] if prev else None,
                    "change":     _diff(latest["value"], prev["value"] if prev else None),
                    "history":    obs[-52:],
                }
                _log.debug("fred_series_ok", series=series_id, value=latest["value"])
            else:
                _log.warning("fred_series_empty", series=series_id)
    else:
        _log.warning("fred_key_missing_for_liquidity")

    # ── 2. Net Fed Liquidity = WALCL - RRPONTSYD - TGA ───────────────────────
    walcl  = us_raw.get("WALCL", [])
    rrp    = us_raw.get("RRPONTSYD", [])
    tga    = us_raw.get("WTREGEN", [])

    if walcl and rrp and tga:
        output["net_fed_liquidity"] = _compute_net_liquidity(walcl, rrp, tga)
        if output["net_fed_liquidity"]:
            latest_nfl = output["net_fed_liquidity"][-1]
            prev_nfl   = output["net_fed_liquidity"][-2] if len(output["net_fed_liquidity"]) >= 2 else None
            output["summary"]["net_fed_liquidity"] = {
                "value":    latest_nfl["value"],
                "date":     latest_nfl["date"],
                "change_1w": _diff(latest_nfl["value"], prev_nfl["value"] if prev_nfl else None),
            }
            _log.info("net_fed_liquidity_ok",
                      value=round(latest_nfl["value"], 1),
                      date=latest_nfl["date"])

    # ── 3. Money Market Funds ─────────────────────────────────────────────────
    retail = us_raw.get("WRMFSL", [])
    inst   = us_raw.get("WRMFNS", [])
    if retail and inst:
        total_mmf = _add_series(retail, inst)
        output["money_market"] = {
            "retail":        _last_obs(retail),
            "institutional": _last_obs(inst),
            "total":         _last_obs(total_mmf) if total_mmf else None,
            "total_history": total_mmf[-52:] if total_mmf else [],
        }
        if total_mmf:
            output["summary"]["money_market_total"] = _last_obs(total_mmf)
            _log.info("mmf_ok", total=round(total_mmf[-1]["value"], 1))
    elif inst:
        # Só institucional disponível — usa como proxy do total
        output["money_market"] = {"institutional": _last_obs(inst)}
        output["summary"]["money_market_total"] = _last_obs(inst)
        _log.info("mmf_institutional_only", value=round(inst[-1]["value"], 1))
    elif retail:
        output["money_market"] = {"retail": _last_obs(retail)}
        output["summary"]["money_market_total"] = _last_obs(retail)

    # ── 4. FRED — M2 Global + Balanços BCx + FX ──────────────────────────────
    global_raw: dict[str, list[dict]] = {}
    if fred_key:
        for series_id, label, unit, factor in _FRED_GLOBAL:
            obs = _fred_observations(series_id, start, fred_key)
            if obs:
                if factor != 1.0:
                    obs = [{"date": o["date"],
                            "value": round(o["value"] * factor, 2) if o["value"] is not None else None}
                           for o in obs]
                global_raw[series_id] = obs
                output["global_m2"][series_id] = {
                    "label":   label,
                    "unit":    unit,
                    "value":   obs[-1]["value"],
                    "date":    obs[-1]["date"],
                    "history": obs[-24:],
                }

    # ── 5. FX rates (âncora de conversão para USD) ────────────────────────────
    # DEXUSEU : USD por 1 EUR  → local_eur * fx_eurusd = USD
    # DEXJPUS : JPY por 1 USD  → local_jpy / fx_jpyusd = USD
    # DEXCHUS : CNY por 1 USD  → local_cny / fx_cnyusd = USD
    # DEXUSUK : USD por 1 GBP  → local_gbp * fx_gbpusd = USD
    fx_eurusd = _latest_value(global_raw.get("DEXUSEU", []))
    fx_jpyusd = _latest_value(global_raw.get("DEXJPUS", []))
    fx_cnyusd = _latest_value(global_raw.get("DEXCHUS", []))
    fx_gbpusd = _latest_value(global_raw.get("DEXUSUK", []))

    def to_usd_eur(v: float | None) -> float | None:
        return round(v * fx_eurusd, 1) if (v and fx_eurusd) else None

    def to_usd_jpy(v: float | None) -> float | None:
        return round(v / fx_jpyusd, 1) if (v and fx_jpyusd and fx_jpyusd > 0) else None

    def to_usd_cny(v: float | None) -> float | None:
        return round(v / fx_cnyusd, 1) if (v and fx_cnyusd and fx_cnyusd > 0) else None

    def to_usd_gbp(v: float | None) -> float | None:
        return round(v * fx_gbpusd, 1) if (v and fx_gbpusd) else None

    # ── 6. M2 Global — tudo em USD ────────────────────────────────────────────
    m2_usd: dict[str, float | None] = {}

    us_m2 = _latest_value(us_raw.get("M2SL", []))
    if us_m2:
        m2_usd["US"] = round(us_m2, 1)

    ea_m2 = _latest_value(global_raw.get("MABMM301EZM189S", []))
    if ea_m2 and (v := to_usd_eur(ea_m2)):
        m2_usd["Euro Area"] = v

    jp_m2 = _latest_value(global_raw.get("MABMM301JPM189S", []))
    if jp_m2 and (v := to_usd_jpy(jp_m2)):
        m2_usd["Japan"] = v

    cn_m2 = _latest_value(global_raw.get("MYAGM2CNM189N", []))
    if cn_m2 and (v := to_usd_cny(cn_m2)):
        m2_usd["China"] = v

    uk_m2 = _latest_value(global_raw.get("MABMM301GBM189S", []))
    if uk_m2 and (v := to_usd_gbp(uk_m2)):
        m2_usd["UK"] = v

    if m2_usd:
        total_g5 = round(sum(v for v in m2_usd.values() if v), 0)
        output["summary"]["global_m2_usd"] = {
            "unit":       "USD bi",
            "components": {k: round(v, 0) for k, v in m2_usd.items() if v},
            "total_g5_usd_bi": total_g5,
            "fx_used": {
                "EURUSD": fx_eurusd,
                "JPYUSD": fx_jpyusd,
                "CNYUSD": fx_cnyusd,
                "GBPUSD": fx_gbpusd,
            },
            "date": date.today().isoformat(),
        }
        _log.info("global_m2_usd_ok", total_bi=total_g5, components=list(m2_usd.keys()))

    # ── 7. Balanços dos Bancos Centrais — todos em USD ────────────────────────
    # ECB assets via FRED (ECBASSETSW: já normalizado para bilhões EUR em coleta)
    ecb_bs_eur = _latest_value(global_raw.get("ECBASSETSW", []))
    ecb_bs_usd = to_usd_eur(ecb_bs_eur)
    ecb_bs_date = global_raw.get("ECBASSETSW", [{}])[-1].get("date") if global_raw.get("ECBASSETSW") else None

    # BoJ assets via FRED (JPNASSETS: bilhões JPY)
    boj_bs_jpy = _latest_value(global_raw.get("JPNASSETS", []))
    boj_bs_usd = to_usd_jpy(boj_bs_jpy)
    boj_bs_date = global_raw.get("JPNASSETS", [{}])[-1].get("date") if global_raw.get("JPNASSETS") else None

    # Fed assets (já em USD)
    fed_bs_usd = _latest_value(us_raw.get("WALCL", []))
    fed_bs_date = us_raw.get("WALCL", [{}])[-1].get("date") if us_raw.get("WALCL") else None

    bs_components: dict[str, dict] = {}
    if fed_bs_usd:
        bs_components["Fed"]  = {"value_usd_bi": round(fed_bs_usd, 0), "date": fed_bs_date}
    if ecb_bs_usd:
        bs_components["ECB"]  = {"value_usd_bi": round(ecb_bs_usd, 0), "date": ecb_bs_date,
                                  "value_eur_bi": round(ecb_bs_eur, 0) if ecb_bs_eur else None}
    if boj_bs_usd:
        bs_components["BoJ"]  = {"value_usd_bi": round(boj_bs_usd, 0), "date": boj_bs_date,
                                  "value_jpy_bi": round(boj_bs_jpy, 0) if boj_bs_jpy else None}

    if bs_components:
        total_bs = round(sum(c["value_usd_bi"] for c in bs_components.values()), 0)
        output["global_balance_sheets"] = {
            "unit":        "USD bi",
            "components":  bs_components,
            "total_g3_usd_bi": total_bs,
        }
        output["summary"]["global_balance_sheets"] = {
            "total_g3_usd_bi": total_bs,
            "components": {k: v["value_usd_bi"] for k, v in bs_components.items()},
        }
        _log.info("global_bs_ok", total_usd_bi=total_bs, components=list(bs_components.keys()))

    # ── 8. ECB M3 via ECB SDW API (opcional — fallback silencioso) ────────────
    try:
        ecb_m3 = _ecb_fetch(_ECB_M3_KEY, periods=36)
        if ecb_m3:
            ecb_m3_eur = ecb_m3[-1]["value"]
            ecb_m3_usd = to_usd_eur(ecb_m3_eur)
            output["ecb"]["m3"] = {
                "label":       "Euro Area M3",
                "value_eur_bi": ecb_m3_eur,
                "value_usd_bi": ecb_m3_usd,
                "date":        ecb_m3[-1]["date"],
                "change_eur":  _diff(ecb_m3_eur, ecb_m3[-2]["value"] if len(ecb_m3) >= 2 else None),
            }
            _log.info("ecb_m3_ok", eur_bi=ecb_m3_eur, usd_bi=ecb_m3_usd)
    except Exception as exc:
        _log.debug("ecb_m3_skipped", error=str(exc))

    _log.info("global_liquidity_done",
              us_series=len(output["us_liquidity"]),
              net_fed_points=len(output["net_fed_liquidity"]),
              ecb_ok=bool(output["ecb"]))
    return output


# ── Cálculos ───────────────────────────────────────────────────────────────────

def _compute_net_liquidity(
    walcl: list[dict],
    rrp: list[dict],
    tga: list[dict],
) -> list[dict]:
    """
    Net Fed Liquidity = Fed Balance Sheet - Reverse Repo - TGA.

    Alinha séries por data (a mais granular vence, as outras interpolam
    a última observação disponível).
    """
    # Indexa por data
    walcl_map = {o["date"]: o["value"] for o in walcl if o["value"] is not None}
    rrp_map   = {o["date"]: o["value"] for o in rrp   if o["value"] is not None}
    tga_map   = {o["date"]: o["value"] for o in tga   if o["value"] is not None}

    # Datas comuns (onde todas as três têm valor)
    common_dates = sorted(set(walcl_map) & set(rrp_map) & set(tga_map))

    result = []
    for d in common_dates:
        val = walcl_map[d] - rrp_map[d] - tga_map[d]
        result.append({"date": d, "value": round(val, 2)})

    return result


def _add_series(a: list[dict], b: list[dict]) -> list[dict]:
    """Soma duas séries alinhadas por data."""
    map_a = {o["date"]: o["value"] for o in a if o["value"] is not None}
    map_b = {o["date"]: o["value"] for o in b if o["value"] is not None}
    common = sorted(set(map_a) & set(map_b))
    return [{"date": d, "value": round(map_a[d] + map_b[d], 2)} for d in common]


def _last_obs(series: list[dict]) -> dict | None:
    if not series:
        return None
    return series[-1]


def _latest_value(series: list[dict]) -> float | None:
    obs = _last_obs(series)
    return obs["value"] if obs else None


def _diff(current: float | None, prev: float | None) -> float | None:
    if current is None or prev is None:
        return None
    return round(current - prev, 2)


# ── FRED helpers ───────────────────────────────────────────────────────────────

def _fred_observations(series_id: str, start: str, api_key: str) -> list[dict]:
    params = urllib.parse.urlencode({
        "series_id":          series_id,
        "api_key":            api_key,
        "observation_start":  start,
        "sort_order":         "asc",
        "file_type":          "json",
    })
    url = f"{_FRED_BASE}/series/observations?{params}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        return [
            {"date": o["date"], "value": _parse_float(o["value"])}
            for o in data.get("observations", [])
            if o.get("value") not in (".", "", None)
        ]
    except Exception as exc:
        _log.warning("fred_obs_error", series=series_id, error=str(exc))
        return []


# ── ECB SDW helpers ────────────────────────────────────────────────────────────

def _ecb_fetch(key: str, periods: int = 52) -> list[dict]:
    """
    Busca série do ECB Statistical Data Warehouse.
    Retorna lista [{date, value}] ordenada por data.

    API: https://data-api.ecb.europa.eu/service/data/{flow}/{key}
         ?format=jsondata&lastNObservations=N
    """
    # Extrai flow da key (primeira parte)
    flow = key.split(".")[0]
    url = (
        f"{_ECB_BASE}/{flow}/{key}"
        f"?format=jsondata&lastNObservations={periods}"
    )
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (research tool)",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())

    # Parse ECB JSON format
    # Structure: dataSets[0].series.{key}.observations.{period_index}: [value, ...]
    # dimensions.observation[0].values → period labels
    try:
        structure = data["structure"]
        time_dim  = structure["dimensions"]["observation"][0]["values"]
        dates     = [t["id"] for t in time_dim]

        datasets = data.get("dataSets", [])
        if not datasets:
            return []

        series_key = list(datasets[0]["series"].keys())[0]
        obs_raw = datasets[0]["series"][series_key]["observations"]

        result = []
        for idx_str, vals in obs_raw.items():
            idx = int(idx_str)
            val = vals[0] if vals else None
            if val is not None and idx < len(dates):
                result.append({"date": _ecb_date(dates[idx]), "value": round(float(val), 2)})

        return sorted(result, key=lambda x: x["date"])
    except Exception as exc:
        _log.warning("ecb_parse_error", key=key, error=str(exc))
        return []


def _ecb_date(raw: str) -> str:
    """Converte YYYY-WNN ou YYYY-MM para ISO YYYY-MM-DD (último dia do período)."""
    if "-W" in raw:
        # Weekly: 2024-W52 → último dia da semana
        import datetime
        year, week = raw.split("-W")
        d = datetime.datetime.strptime(f"{year}-W{week}-5", "%G-W%V-%u").date()
        return d.isoformat()
    if len(raw) == 7:  # YYYY-MM
        return raw + "-01"
    return raw


def _parse_float(v: str) -> float | None:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None
