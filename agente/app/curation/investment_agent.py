"""
Investment Agent — Camada de Decisão.

Recebe os dados já organizados pelo pipeline (bundle + curation result)
e produz um diagnóstico de investimento acionável.

NÃO coleta dados. NÃO acessa Bloomberg. NÃO busca fontes externas.
Opera exclusivamente sobre os dados recebidos.

Arquitetura interna (5 motores):
  Motor 1 — Rational Engine      : macroeconomia, valuation, fundamento
  Motor 2 — Behavioral Engine    : fluxo, opções, positioning, mecânica
  Motor 3 — Entropy Engine       : desordem, divergências, fragilidade
  Motor 4 — Arbitration Engine   : qual motor pesa mais agora
  Motor 5 — Allocation Engine    : decisão prática de alocação
"""

from __future__ import annotations

import json as _json
import re as _re
from datetime import date
from pathlib import Path
from typing import Any

from app.audit.logger import get_logger
from app.curation.llm_client import call_claude
from app.curation.models import CurationResult
from app.models.daily_ingestion_bundle import DailyIngestionBundle

_log = get_logger("curation.investment_agent")
_MODEL = "claude-sonnet-4-6"


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM = """\
Você é um agente de diagnóstico macro, micro, fluxo, valuation e decisão de investimento.

Sua única função é receber dados já organizados e interpretá-los para tomar decisão.
Você não busca dados. Você não menciona Bloomberg. Você não mistura coleta com análise.

Você opera como árbitro entre três forças:

1. RACIONAL — leitura da economia e dos ativos: macroeconomia, microeconomia, \
econometria, crédito, valuation, reversão à média, prêmio de risco, produtividade, \
qualidade do crescimento, qualidade do lucro, saúde do sistema.

2. COMPORTAMENTAL — leitura mecânica do mercado: fluxo, opções, positioning, \
crowding, gamma, skew, volatilidade, breadth, CTA, vol control, concentração, \
reflexividade, narrativa versus preço.

3. ENTROPIA — leitura da desordem do sistema: dispersão entre sinais, quebra de \
correlações, instabilidade de regime, fragilidade de liquidez, divergência entre \
fundamento e preço, divergência entre fluxo e valuation, sinais de caos crescente.

═══ MOTORES INTERNOS (pense nesta ordem) ═══

Motor 1 — Rational Engine
Interprete os dados pela literatura clássica. O mundo real justifica esse preço?

Motor 2 — Behavioral Engine
Para onde o fluxo real está indo? Independentemente do fundamento, o mercado está
empurrando o preço para onde?

Motor 3 — Entropy Engine
O ambiente está ordenado, tensionado ou caótico? Qual o grau de desordem do sistema?

Motor 4 — Arbitration Engine
Compare os motores anteriores. O que está pesando mais agora: racional, comportamento
ou entropia?

Motor 5 — Allocation Engine
Transforme a leitura final em decisão prática. O que comprar, vender, evitar,
reduzir, proteger ou operar taticamente?

═══ SCORES INTERNOS (atribua antes de escrever) ═══

Rational Score      : -2 a +2  (fundamento vs. preço)
Behavioral Score    : -2 a +2  (fluxo, momentum, posicionamento)
Entropy Score       : -2 a +2  (invertido — entropia alta = pior para convicção)
Valuation Gap       : -2 a +2  (quão distante está o preço do fair value histórico)
Regime Confidence   : -2 a +2  (quão legível está o regime atual)

═══ BLOCOS DE ANÁLISE ═══

Organize sua leitura internamente nestes 10 blocos (nem todos precisam ter o mesmo
peso — priorize os que têm sinal mais forte):

1. Macro e ciclo
2. Inflação, juros e liquidez
3. Crédito e condições financeiras
4. Fiscal e sustentabilidade
5. Geopolítica e risco sistêmico
6. Fluxo e posicionamento
7. Opções e estrutura de mercado
8. Microeconomia, lucros e retorno sobre capital
9. Valuation e distância da média
10. Entropia, instabilidade e risco de caos

═══ REGRAS DE ARBITRAGEM ═══

- Racional forte + Comportamental fraco → mercado atrasado, oportunidade antecipada
- Racional fraco + Comportamental forte → rali mecânico, não estrutural
- Ambos fortes → favoreça risco
- Ambos fracos → reduza risco
- Entropia alta → reduza convicção em qualquer leitura simples
- Valuation muito esticado → exija mais confirmação
- Valuation muito deprimido + fluxo virando → procure assimetria contrária
- Conflito entre fundamento e fluxo → explicite o conflito e diga quem domina \
  no curto prazo e quem domina no médio prazo

═══ PRINCÍPIOS INVIOLÁVEIS ═══

- O dado tem prioridade sobre narrativa, consenso, manchete e mídia
- Nunca resuma dados sem interpretar o que significam
- Nunca fique em neutralidade vazia
- Nunca invente dados
- Sempre julgue a qualidade do movimento, não apenas sua direção
- Sempre compare preço atual contra histórico, assimetria e valuation
- Sempre conclua com ação prática
- Quando a entropia estiver alta, diga isso explicitamente e reduza confiança

═══ FORMATO DE SAÍDA ═══

PRIMEIRA LINHA obrigatória — JSON com os 5 scores (sem texto antes ou depois nessa linha):
SCORES: {"rational": X, "behavioral": X, "entropy": X, "valuation_gap": X, "regime_confidence": X}

Em seguida, o diagnóstico completo em português. Texto corrido, curto, denso, opinativo.
Tom de analista experiente, cético com narrativas rasas, focado em regime, fluxo, valuation
e assimetria.

O diagnóstico deve conter, mesmo sem títulos explícitos:

1. Diagnóstico do regime atual
2. Para onde o fluxo real está indo
3. Se o fundamento sustenta ou não o movimento
4. Grau de entropia ou fragilidade do sistema
5. Decisão prática — o que comprar, vender, evitar, reduzir ou proteger
6. Racional principal em poucas linhas

Sem burocracia. Sem linguagem genérica de IA. Termine sempre em ação prática.
Se a confiança estiver baixa por entropia alta ou conflito de sinais, diga isso.
"""


# ── Data context builder ───────────────────────────────────────────────────────

def _build_data_context(
    bundle: DailyIngestionBundle,
    curation: CurationResult | None,
) -> str:
    sections: list[str] = []

    # ── Curation signal ────────────────────────────────────────────────────────
    if curation:
        primary = curation.narrative.primary_signal
        secondaries = curation.narrative.secondary_signals or []
        lines = [
            "=== SINAL DE CURADORIA ===",
            f"Narrativa primária : {primary.label} (confiança {primary.confidence:.0%})",
            f"Status             : {primary.status}",
        ]
        if primary.description:
            lines.append(f"Descrição          : {primary.description[:200]}")
        if secondaries:
            lines.append("Secundárias        : " + " | ".join(s.label for s in secondaries[:3]))
        sections.append("\n".join(lines))

    # ── Market prices ──────────────────────────────────────────────────────────
    if bundle.market_prices:
        lines = ["=== PREÇOS DE MERCADO ==="]
        for ticker, info in list(bundle.market_prices.items())[:15]:
            if not isinstance(info, dict):
                continue
            name  = info.get("name", ticker)
            price = info.get("price")
            ret1d = info.get("return_1d")
            ret1w = info.get("return_1w")
            ytd   = info.get("return_ytd")
            if price is None:
                continue
            parts = [f"{name:<28} {price:>10.2f}"]
            if ret1d is not None:
                parts.append(f"  1d={ret1d:+.1f}%")
            if ret1w is not None:
                parts.append(f"  1w={ret1w:+.1f}%")
            if ytd is not None:
                parts.append(f"  YTD={ytd:+.1f}%")
            lines.append("  " + "".join(parts))
        sections.append("\n".join(lines))

    # ── Network / Econophysics ─────────────────────────────────────────────────
    if bundle.market_prices:
        try:
            from app.analysis.network import analyze as net_analyze, format_summary as net_fmt
            net_result = net_analyze(bundle.market_prices, lookback_days=90, lasso_alpha=0.1)
            net_summary = net_fmt(net_result, bundle.market_prices)
            if net_summary:
                sections.append(net_summary)
        except Exception:
            pass

    # ── FRED macro ────────────────────────────────────────────────────────────
    if bundle.fred_data:
        series = bundle.fred_data.get("series", {})
        calendar = bundle.fred_data.get("calendar", [])
        lines = ["=== MACRO (FRED) ==="]
        priority = ["Política Monetária", "Inflação", "Mercado de Trabalho",
                    "Crescimento", "Crédito e Condições Financeiras"]
        for cat in priority:
            cat_data = series.get(cat, [])
            if not cat_data:
                continue
            lines.append(f"\n{cat}:")
            for s in cat_data[:4]:
                val = s.get("value")
                chg = s.get("change")
                if val is None:
                    continue
                chg_str = f"  Δ{chg:+.3f}" if chg is not None else ""
                lines.append(f"  {s['label']:<40} {val:>8.2f} {s['unit']}{chg_str}")
        # HY implied default
        credit = series.get("Crédito e Condições Financeiras", [])
        hy = next((s for s in credit if s.get("series_id") == "BAMLH0A0HYM2"), None)
        if hy and hy.get("value"):
            oas_bps = hy["value"] * 100
            implied_dr = oas_bps / 0.70
            lines.append(f"\n  → HY OAS {oas_bps:.0f}bps → default implícito {implied_dr/100:.1f}%a.a. (recovery 30¢)")
        # Agenda próximos 7 dias
        _HIGH = ["employment", "nonfarm", "cpi", "fomc", "gdp", "jolts", "retail",
                 "pce", "ism", "ppi", "adp"]
        upcoming = [r for r in calendar if any(kw in r.get("release_name","").lower() for kw in _HIGH)]
        if upcoming:
            lines.append("\nAgenda próximos 10 dias (high-impact):")
            seen: set[str] = set()
            for r in upcoming[:8]:
                k = f"{r['date']}|{r['release_name']}"
                if k not in seen:
                    seen.add(k)
                    lines.append(f"  {r['date']}  {r['release_name']}")
        sections.append("\n".join(lines))

    # ── Damodaran ─────────────────────────────────────────────────────────────
    if bundle.damodaran_data:
        lines = ["=== VALUATION (DAMODARAN) ==="]
        erp = bundle.damodaran_data.get("erp_current")
        if erp:
            lines.append(
                f"  ERP implícito S&P 500 ({erp['year']}): {erp['erp_pct']:.2f}%  "
                f"| T-Bond: {erp.get('t_bond_rate_pct', '?'):.2f}%"
            )
        # Top 5 WACC mais alto (mais caro o capital)
        wacc = bundle.damodaran_data.get("wacc_sectors", [])
        if wacc:
            top5 = sorted([s for s in wacc if s.get("wacc_pct")],
                          key=lambda x: x["wacc_pct"], reverse=True)[:5]
            lines.append("  Setores com custo de capital mais alto:")
            for s in top5:
                lines.append(f"    {s['industry']:<40} WACC={s['wacc_pct']:.1f}%")
        sections.append("\n".join(lines))

    # ── Polymarket ────────────────────────────────────────────────────────────
    if bundle.polymarket_markets:
        lines = ["=== MERCADOS DE PREDIÇÃO (POLYMARKET) ==="]
        for m in bundle.polymarket_markets[:8]:
            title = m.get("title") or m.get("question", "")
            prob  = m.get("best_yes_probability") or m.get("probability")
            vol   = m.get("volume")
            if not title:
                continue
            prob_str = f"  {prob:.0%}" if prob is not None else ""
            vol_str  = f"  vol=${vol/1e6:.1f}M" if vol else ""
            lines.append(f"  {title[:70]}{prob_str}{vol_str}")
        sections.append("\n".join(lines))

    # ── SpotGamma / Flow ───────────────────────────────────────────────────────
    sg_reports = bundle.spotgamma_reports
    if sg_reports:
        lines = ["=== FLUXO / OPÇÕES (SPOTGAMMA) ==="]
        for r in sg_reports[:2]:
            if r.raw_text:
                lines.append(f"[{r.report_type}] {r.title}")
                lines.append(r.raw_text[:800])
        sections.append("\n".join(lines))

    # ── X Timeline top signals ─────────────────────────────────────────────────
    if bundle.x_items:
        lines = ["=== X TIMELINE (sinais recentes) ==="]
        for t in bundle.x_items[:10]:
            if t.text and len(t.text) > 40:
                lines.append(f"  [{t.author}] {t.text[:200]}")
        sections.append("\n".join(lines))

    # ── Global Liquidity ──────────────────────────────────────────────────────
    liq = getattr(bundle, "global_liquidity", {})
    if liq:
        summary = liq.get("summary", {})
        us      = liq.get("us_liquidity", {})
        ecb     = liq.get("ecb", {})
        lines   = ["=== LIQUIDEZ GLOBAL ==="]

        nfl = summary.get("net_fed_liquidity")
        if nfl:
            lines.append(f"  Net Fed Liquidity : {nfl['value']:.0f} USD bi  [{nfl['date']}]"
                         + (f"  Δ1w={nfl['change_1w']:+.0f}" if nfl.get("change_1w") else ""))

        for sid, short in [("WALCL","Fed BS"), ("RRPONTSYD","RRP"), ("WTREGEN","TGA")]:
            e = us.get(sid)
            if e:
                lines.append(f"  {short:<12}: {e['value']:.0f} USD bi  [{e['date']}]")

        mmf_total = summary.get("money_market_total")
        if mmf_total:
            lines.append(f"  MMF Total    : {mmf_total['value']:.0f} USD bi  [{mmf_total['date']}]")

        # Balanços BCx (Fed/ECB/BoJ em USD)
        gbs = liq.get("summary", {}).get("global_balance_sheets", {})
        if gbs:
            total_g3 = gbs.get("total_g3_usd_bi")
            if total_g3:
                lines.append(f"  G3 BS Total  : {total_g3:.0f} USD bi")
            for name, val in gbs.get("components", {}).items():
                if val:
                    lines.append(f"    {name}: {val:.0f} USD bi")

        g_m2 = summary.get("global_m2_usd")
        if g_m2:
            total = g_m2.get("total_g5_usd_bi") or g_m2.get("total_g4_usd_bi")
            if total:
                lines.append(f"  M2 G5 (USD)  : {total:.0f} USD bi")
            for c, v in g_m2.get("components", {}).items():
                if v:
                    lines.append(f"    {c}: {v:.0f}")

        sections.append("\n".join(lines))

    # ── ZeroHedge blocks ──────────────────────────────────────────────────────
    if bundle.market_ear_blocks:
        lines = ["=== ZEROHEDGE MARKET EAR ==="]
        for b in bundle.market_ear_blocks[:6]:
            content = (b.body_text or b.subtitle or b.title or "").strip()
            if content:
                title_str = f"[{b.title}] " if b.title else ""
                lines.append(f"  {title_str}{content[:300]}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


# ── Public API ─────────────────────────────────────────────────────────────────

def diagnose(
    bundle: DailyIngestionBundle,
    curation: CurationResult | None = None,
    focus: str | None = None,
) -> dict[str, Any]:
    """
    Executa o diagnóstico de investimento sobre o bundle do dia.

    Args:
        bundle   : DailyIngestionBundle já processado pelo pipeline
        curation : CurationResult opcional (narrativa + verificação)
        focus    : instrução específica do operador (ex: "foque em crédito HY")

    Returns:
        {
          "scores"   : {"rational": int, "behavioral": int, "entropy": int,
                        "valuation_gap": int, "regime_confidence": int}
          "narrative": str  — diagnóstico completo em português
          "run_date" : str
          "raw"      : str  — output bruto do LLM
        }
    """
    data_ctx = _build_data_context(bundle, curation)

    user_prompt = (
        f"DATA DE REFERÊNCIA: {bundle.run_date}\n\n"
        f"=== DADOS DO DIA ===\n{data_ctx}"
    )
    if focus:
        user_prompt += f"\n\nINSTRUÇÃO DO OPERADOR: {focus}"

    _log.info("investment_agent_start", run_date=str(bundle.run_date),
              data_chars=len(data_ctx))

    raw = call_claude(
        _SYSTEM,
        user_prompt,
        model=_MODEL,
        max_tokens=2000,
        temperature=0.3,
    )

    # ── Extrai scores da primeira linha ────────────────────────────────────────
    scores: dict[str, int] = {}
    narrative = raw
    m = _re.search(r"SCORES:\s*(\{[^}]+\})", raw)
    if m:
        try:
            scores = {k: int(v) for k, v in _json.loads(m.group(1)).items()}
        except Exception:
            pass
        # Remove a linha de scores do texto narrativo
        narrative = _re.sub(r"SCORES:\s*\{[^}]+\}\n?", "", raw).strip()

    _log.info("investment_agent_done", chars=len(raw), scores=scores)
    return {
        "scores":    scores,
        "narrative": narrative,
        "run_date":  str(bundle.run_date),
        "raw":       raw,
    }
