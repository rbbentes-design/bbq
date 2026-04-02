"""
Provider: GEX + LETF Rebalancing Flow (Barbon et al.)

Calcula dois sinais mecânicos de pressão de fluxo no fim do dia:

  Γ^HP  — Gamma Hedging Pressure
           Dealers curtos em gamma precisam delta-hedge na direção do mercado,
           amplificando moves intraday (short gamma regime).

  Ω^LETF — LETF Rebalancing Flow
           ETFs alavancados (3x, 2x, inverso) precisam rebalancear diariamente
           para manter alavancagem constante. O fluxo é MECÂNICO e previsível:
             Rebal_$ = AUM × L × (L-1) × r / (1 + L×r)

Esses sinais são mais fortes perto do close (14:30-16:00 ET).

Roda via subprocess (bql_gex_flow.py) sob BQuant Python 3.11.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("providers.gex_letf")

_BQNT_PYTHON = Path("C:/blp/bqnt/environments/bqnt-3/python.exe")
_GEX_SCRIPT  = Path(__file__).parent.parent.parent / "scripts" / "bql_gex_flow.py"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class LETFFlowIndex:
    index:     str           # "spx" | "ndx" | "sox"
    flow_usd:  float = 0.0   # USD total de rebalanceamento (positivo=compra)
    flow_bn:   float = 0.0   # em bilhões
    direction: str   = "flat"  # "buy" | "sell" | "flat"
    ret:       float | None = None  # retorno do dia do underlying


@dataclass
class GEXData:
    gex_usd:      float = 0.0      # GEX total em USD
    gex_bn:       float = 0.0      # em bilhões
    gamma_regime: str   = "flat"   # "long" | "short" | "flat"
    flip_level:   float | None = None  # nível de preço onde GEX muda de sinal
    per_strike:   list  = field(default_factory=list)  # [[strike, gex_bn], ...]


@dataclass
class MemberFlow:
    ticker:        str
    letf_flow_usd: float = 0.0
    gex_flow_usd:  float = 0.0
    total_usd:     float = 0.0
    direction:     str   = "flat"


@dataclass
class FlowPrediction:
    """Resultado completo do modelo GEX + LETF."""
    # Sinais agregados por índice
    spx:    LETFFlowIndex = field(default_factory=lambda: LETFFlowIndex("spx"))
    ndx:    LETFFlowIndex = field(default_factory=lambda: LETFFlowIndex("ndx"))
    sox:    LETFFlowIndex = field(default_factory=lambda: LETFFlowIndex("sox"))

    # GEX do SPX
    gex:    GEXData = field(default_factory=GEXData)

    # Fluxo por ETF individual
    per_etf: dict[str, dict] = field(default_factory=dict)

    # Distribuição por membro
    per_member: dict[str, MemberFlow] = field(default_factory=dict)

    # Sinal líquido
    direction:     str   = "flat"
    magnitude_usd: float = 0.0
    magnitude_bn:  float = 0.0
    conviction:    str   = "low"
    summary:       str   = ""

    timestamp: str = ""
    source:    str = "bql"
    error:     str = ""


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_result(data: dict) -> FlowPrediction:
    pred = FlowPrediction(timestamp=data.get("timestamp", ""))

    letf = data.get("letf", {})
    for idx_key, attr in [("spx", "spx"), ("ndx", "ndx"), ("sox", "sox")]:
        raw = letf.get(idx_key, {})
        f_usd = raw.get("flow_usd", 0.0) or 0.0
        getattr(pred, attr).__class__  # type check
        setattr(pred, attr, LETFFlowIndex(
            index=idx_key,
            flow_usd=f_usd,
            flow_bn=round(f_usd / 1e9, 3),
            direction=raw.get("direction", "flat"),
            ret=letf.get(f"{idx_key}_r"),
        ))

    pred.per_etf = letf.get("per_etf", {})

    gex_raw = data.get("gex", {}).get("spx", {})
    pred.gex = GEXData(
        gex_usd      = gex_raw.get("gex_usd", 0.0) or 0.0,
        gex_bn       = gex_raw.get("gex_bn", 0.0) or 0.0,
        gamma_regime = gex_raw.get("gamma_regime", "flat"),
        flip_level   = gex_raw.get("flip_level"),
        per_strike   = gex_raw.get("per_strike", []),
    )

    for ticker, raw in data.get("flow_per_member", {}).items():
        pred.per_member[ticker] = MemberFlow(
            ticker        = ticker,
            letf_flow_usd = raw.get("letf_flow_usd", 0.0) or 0.0,
            gex_flow_usd  = raw.get("gex_flow_usd", 0.0) or 0.0,
            total_usd     = raw.get("total", 0.0) or 0.0,
            direction     = raw.get("direction", "flat"),
        )

    net = data.get("net_signal", {})
    pred.direction     = net.get("direction", "flat")
    pred.magnitude_usd = net.get("magnitude_usd", 0.0) or 0.0
    pred.magnitude_bn  = net.get("magnitude_bn", 0.0) or 0.0
    pred.conviction    = net.get("conviction", "low")
    pred.summary       = net.get("summary", "")

    return pred


# ── API pública ───────────────────────────────────────────────────────────────

def collect() -> FlowPrediction:
    """
    Executa bql_gex_flow.py via BQuant Python e retorna FlowPrediction.

    Retorna FlowPrediction vazio (com error preenchido) se BQL não disponível.
    """
    if not _BQNT_PYTHON.exists():
        _log.warning("bqnt_python_not_found", path=str(_BQNT_PYTHON))
        pred = FlowPrediction()
        pred.error = "BQuant Python não encontrado"
        return pred

    if not _GEX_SCRIPT.exists():
        _log.warning("gex_script_not_found", path=str(_GEX_SCRIPT))
        pred = FlowPrediction()
        pred.error = "Script bql_gex_flow.py não encontrado"
        return pred

    _log.info("gex_letf_start")
    try:
        proc = subprocess.run(
            [str(_BQNT_PYTHON), str(_GEX_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=300,  # opções chain demora mais
        )

        if proc.returncode != 0 and proc.stderr:
            _log.debug("gex_letf_stderr", stderr=proc.stderr[:400])

        if not proc.stdout.strip():
            _log.warning("gex_letf_empty_output")
            pred = FlowPrediction()
            pred.error = "Sem saída do script"
            return pred

        data = json.loads(proc.stdout.strip())
        pred = _parse_result(data)

        _log.info(
            "gex_letf_done",
            direction=pred.direction,
            magnitude_bn=pred.magnitude_bn,
            conviction=pred.conviction,
            gamma_regime=pred.gex.gamma_regime,
            spx_letf_bn=pred.spx.flow_bn,
        )
        return pred

    except subprocess.TimeoutExpired:
        _log.warning("gex_letf_timeout")
        pred = FlowPrediction()
        pred.error = "Timeout (>5min)"
        return pred
    except json.JSONDecodeError as e:
        _log.warning("gex_letf_json_error", error=str(e))
        pred = FlowPrediction()
        pred.error = f"JSON inválido: {e}"
        return pred
    except Exception as e:
        _log.warning("gex_letf_error", error=str(e))
        pred = FlowPrediction()
        pred.error = str(e)
        return pred


def get_member_flow(ticker: str, pred: FlowPrediction) -> dict[str, Any]:
    """
    Retorna {letf_flow, gex_flow, total, direction} para um ticker.
    Utilitário para uso no graph_engine e no render.
    """
    mf = pred.per_member.get(ticker)
    if mf is None:
        return {}
    return {
        "letf_flow_usd": mf.letf_flow_usd,
        "gex_flow_usd":  mf.gex_flow_usd,
        "total_usd":     mf.total_usd,
        "total_mn":      round(mf.total_usd / 1e6, 2),
        "direction":     mf.direction,
    }
