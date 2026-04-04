"""
Vol Options Regime — Motor de decisão de volatilidade via Greeks Dashboard (BQuant).

Interpreta os dados reais do Greeks Dashboard (GEX, IV/RV, Squeeze, Tail, Skew, Flow)
e converte em decisões acionáveis que reverberam em:
  1. Alocação — position_scalar, hedge_required, hedge_asset
  2. Estratégia de opções — sell vol / buy vol / hedge / neutral
  3. Brain (narrativa) — regime de amplificação vs amortecimento

Lógica central:
  GEX positivo  → dealers long gamma → spot AMORTECIDO → vender vol
  GEX negativo  → dealers short gamma → spot AMPLIFICADO → comprar proteção
  IV rica (IV-RV > 4pp) → premium alto → vender premium (IC, spreads)
  IV barata (IV-RV < -2pp) → comprar vol (straddles, calendars)
  Squeeze > 75 → combustível para move explosivo → hedge obrigatório
  Tail > 65 → risco de cauda elevado → reduzir posições e adicionar hedge
  Spot < Gamma Flip → short gamma regime → moves amplificados → cuidado
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from app.audit.logger import get_logger

if TYPE_CHECKING:
    from app.providers.options_store import OptionsSnapshot
    from app.analysis.vol_regime import VolRegimeResult

_log = get_logger("analysis.vol_options_regime")


# ── Enums / constantes ────────────────────────────────────────────────────────

class GexRegime:
    LONG_STRONG   = "long_gamma_strong"    # GEX > +3B: mercado muito amortecido
    LONG          = "long_gamma"           # GEX > +0.5B: leve amortecimento
    NEUTRAL       = "neutral_gamma"        # ±0.5B
    SHORT         = "short_gamma"          # GEX < -0.5B: amplificação
    SHORT_EXTREME = "short_gamma_extreme"  # GEX < -3B: tail risk alto


class IvRegime:
    VERY_RICH  = "iv_very_rich"    # IV-RV > +6pp: vender vol agressivo
    RICH       = "iv_rich"         # IV-RV > +3pp: vender vol
    FAIR       = "iv_fair"         # ±3pp: neutro
    CHEAP      = "iv_cheap"        # IV-RV < -2pp: comprar vol
    VERY_CHEAP = "iv_very_cheap"   # IV-RV < -5pp: comprar vol agressivo


class VolDecision:
    SELL_VOL = "sell_vol"   # vender premium — IV rica + GEX long + calmo
    BUY_VOL  = "buy_vol"    # comprar vol — IV barata ou GEX curto extremo
    HEDGE    = "hedge"      # adicionar proteção — tail/squeeze elevados
    NEUTRAL  = "neutral"    # aguardar / sem sinal claro


class SpotRegime:
    ABOVE_CALL_WALL   = "above_call_wall"     # resistência acima
    NEAR_CALL_WALL    = "near_call_wall"      # próximo teto
    BETWEEN_WALLS     = "between_walls"       # zona normal
    NEAR_PUT_WALL     = "near_put_wall"       # próximo suporte
    BELOW_PUT_WALL    = "below_put_wall"      # suporte rompido
    ABOVE_GAMMA_FLIP  = "above_gamma_flip"    # long gamma zone
    BELOW_GAMMA_FLIP  = "below_gamma_flip"    # short gamma zone


# ── Output ────────────────────────────────────────────────────────────────────

@dataclass
class VolOptionsRegime:
    # Classificações
    gex_regime:    str = GexRegime.NEUTRAL
    iv_regime:     str = IvRegime.FAIR
    spot_regime:   str = SpotRegime.BETWEEN_WALLS
    vol_decision:  str = VolDecision.NEUTRAL

    # Métricas-chave capturadas
    gex_net_bn:     float = 0.0
    iv_rv_pp:       float = 0.0
    squeeze_score:  float = 0.0
    tail_score:     float = 0.0
    flow_score:     float = 50.0
    skew_25d:       float = 0.0
    spot:           float = 0.0
    gamma_flip:     float = 0.0
    call_wall:      float = 0.0
    put_wall:       float = 0.0
    vix_bbg:        float = 0.0    # VIX do Greeks Dashboard (BBG, mais preciso)
    delta_bn:       float = 0.0
    vanna_bn:       float = 0.0

    # Decisões de alocação
    position_scalar_adj: float = 0.0    # ajuste ADICIONAL ao position_scalar do vol_regime
                                         # ex: -0.10 = reduz mais 10% nas posições
    hedge_required:      bool  = False
    hedge_asset:         str   = "SPTS"  # default: short SPX ETF
    hedge_alloc_pct:     float = 0.0     # % do portfolio para o hedge (0.02 = 2%)

    # Estratégia de opções de portfólio
    portfolio_options_strategy: str = ""   # "iron_condor" | "spy_put_spread" | "call_overwrite" | "straddle" | ""
    strategy_rationale: str = ""

    # Sinais granulares [0,1]
    sell_vol_signal:  float = 0.0   # convicção para vender vol
    buy_vol_signal:   float = 0.0   # convicção para comprar vol
    hedge_signal:     float = 0.0   # urgência de hedge

    # Para o brain / narrativa
    amplification_regime: bool = False   # GEX short = moves maiores, narrativas aceleram
    dampening_regime:     bool = False   # GEX long = mercado preso, reversão a média
    fragility_alert:      bool = False   # squeeze + tail elevados = fragilidade estrutural
    flow_positive:        bool = False   # flow score > 60 = demanda estrutural positiva

    # Texto explicativo
    rationale: str = ""
    level_notes: list[str] = field(default_factory=list)


# ── Motor principal ───────────────────────────────────────────────────────────

def compute_vol_options_regime(
    snapshot: "OptionsSnapshot | None",
    vol_regime: "VolRegimeResult | None" = None,
) -> VolOptionsRegime | None:
    """
    Interpreta o snapshot do Greeks Dashboard e retorna um VolOptionsRegime.
    Retorna None se não houver snapshot.
    """
    if snapshot is None:
        return None

    r = VolOptionsRegime(
        gex_net_bn    = snapshot.gex_net_bn,
        iv_rv_pp      = snapshot.iv_rv_pp,
        squeeze_score = snapshot.squeeze_score,
        tail_score    = snapshot.tail_score,
        flow_score    = snapshot.flow_score_total,
        skew_25d      = snapshot.skew_25d,
        spot          = snapshot.spot,
        gamma_flip    = snapshot.gamma_flip,
        call_wall     = snapshot.call_wall,
        put_wall      = snapshot.put_wall,
        vix_bbg       = snapshot.vix,
        delta_bn      = snapshot.delta_bn,
        vanna_bn      = snapshot.vanna_bn,
    )

    reasons: list[str] = []

    # ── 1. GEX Regime ─────────────────────────────────────────────────────────
    gex = snapshot.gex_net_bn
    if gex > 3.0:
        r.gex_regime = GexRegime.LONG_STRONG
        r.dampening_regime = True
        reasons.append(f"GEX={gex:+.1f}B — dealers long gamma forte; spot amortecido, range estreito")
    elif gex > 0.5:
        r.gex_regime = GexRegime.LONG
        r.dampening_regime = True
        reasons.append(f"GEX={gex:+.1f}B — dealers long gamma; bias de mean reversion")
    elif gex < -3.0:
        r.gex_regime = GexRegime.SHORT_EXTREME
        r.amplification_regime = True
        reasons.append(f"GEX={gex:+.1f}B — dealers SHORT gamma extremo; moves amplificados, TAIL RISK elevado")
    elif gex < -0.5:
        r.gex_regime = GexRegime.SHORT
        r.amplification_regime = True
        reasons.append(f"GEX={gex:+.1f}B — dealers short gamma; volatilidade pode se acelerar")
    else:
        r.gex_regime = GexRegime.NEUTRAL
        reasons.append(f"GEX={gex:+.1f}B — neutro")

    # ── 2. Spot vs Key Levels ─────────────────────────────────────────────────
    spot     = snapshot.spot
    gflip    = snapshot.gamma_flip
    cwall    = snapshot.call_wall
    pwall    = snapshot.put_wall

    if spot > 0 and gflip > 0:
        if spot > gflip * 1.001:
            r.spot_regime = SpotRegime.ABOVE_GAMMA_FLIP
            reasons.append(f"Spot {spot:,.0f} ACIMA do Gamma Flip {gflip:,.0f} — zona long gamma")
        else:
            r.spot_regime = SpotRegime.BELOW_GAMMA_FLIP
            reasons.append(f"Spot {spot:,.0f} ABAIXO do Gamma Flip {gflip:,.0f} — zona short gamma, moves amplificados")
            r.amplification_regime = True

    if cwall > 0 and pwall > 0:
        dist_cw = (cwall - spot) / spot if spot > 0 else 0
        dist_pw = (spot - pwall) / spot if spot > 0 else 0
        if dist_cw < 0.003:    # < 0.3% do Call Wall
            r.spot_regime = SpotRegime.NEAR_CALL_WALL
            reasons.append(f"Spot próximo ao Call Wall ({cwall:,.0f}) — resistência, potencial de rejeição")
            r.level_notes.append(f"⚠ Spot {dist_cw:.1%} do Call Wall {cwall:,.0f}")
        elif spot > cwall:
            r.spot_regime = SpotRegime.ABOVE_CALL_WALL
            reasons.append(f"Spot ACIMA do Call Wall {cwall:,.0f} — break ou armadilha")
            r.level_notes.append(f"🔴 Spot acima da resistência Call Wall {cwall:,.0f}")
        elif dist_pw < 0.005:  # < 0.5% do Put Wall
            if r.spot_regime == SpotRegime.ABOVE_GAMMA_FLIP:
                pass
            else:
                r.spot_regime = SpotRegime.NEAR_PUT_WALL
            reasons.append(f"Spot próximo ao Put Wall ({pwall:,.0f}) — suporte, bounce possível")
            r.level_notes.append(f"⚡ Spot {dist_pw:.1%} do Put Wall {pwall:,.0f} — zona de suporte de dealer")
        elif spot < pwall:
            r.spot_regime = SpotRegime.BELOW_PUT_WALL
            reasons.append(f"Spot ABAIXO do Put Wall {pwall:,.0f} — suporte perdido, risco de aceleração da queda")
            r.level_notes.append(f"🔴 Suporte Put Wall rompido — aceleração de queda provável")

    # ── 3. IV Regime ──────────────────────────────────────────────────────────
    iv_rv = snapshot.iv_rv_pp
    if iv_rv > 6:
        r.iv_regime = IvRegime.VERY_RICH
        reasons.append(f"IV-RV={iv_rv:+.1f}pp — premium muito rico; vender vol (iron condor / call spread)")
    elif iv_rv > 3:
        r.iv_regime = IvRegime.RICH
        reasons.append(f"IV-RV={iv_rv:+.1f}pp — premium rico; vender vol com spreads")
    elif iv_rv < -5:
        r.iv_regime = IvRegime.VERY_CHEAP
        reasons.append(f"IV-RV={iv_rv:+.1f}pp — IV MUITO BARATA vs realizada; comprar vol agressivo (straddles)")
    elif iv_rv < -2:
        r.iv_regime = IvRegime.CHEAP
        reasons.append(f"IV-RV={iv_rv:+.1f}pp — IV barata; comprar vol / calendars")
    else:
        r.iv_regime = IvRegime.FAIR
        reasons.append(f"IV-RV={iv_rv:+.1f}pp — premium justo; neutro")

    # ── 4. Squeeze & Tail ────────────────────────────────────────────────────
    sq = snapshot.squeeze_score
    tail = snapshot.tail_score
    skew = snapshot.skew_25d

    if sq > 80:
        r.fragility_alert = True
        reasons.append(f"Squeeze={sq:.0f}/100 — compressão extrema; move explosivo iminente")
    elif sq > 60:
        r.fragility_alert = True
        reasons.append(f"Squeeze={sq:.0f}/100 — compressão elevada; fragilidade crescente")

    if tail > 70:
        r.fragility_alert = True
        reasons.append(f"Tail Risk={tail:.0f}/100 — risco de cauda ALTO; hedge obrigatório")
    elif tail > 50:
        reasons.append(f"Tail Risk={tail:.0f}/100 — risco de cauda moderado")

    if skew > 5:
        reasons.append(f"Skew 25D={skew:+.1f}pp — put skew elevado; mercado pagando por proteção")
    elif skew < 1:
        reasons.append(f"Skew 25D={skew:+.1f}pp — skew baixo; proteção barata")

    # ── 5. Flow Score ────────────────────────────────────────────────────────
    flow = snapshot.flow_score_total
    r.flow_positive = flow > 60
    if flow > 70:
        reasons.append(f"Flow Score={flow:.0f}/100 — demanda estrutural forte (CTA + dealer positivos)")
    elif flow < 30:
        reasons.append(f"Flow Score={flow:.0f}/100 — fluxo estrutural NEGATIVO; pressão vendedora mecânica")
    elif flow < 40:
        reasons.append(f"Flow Score={flow:.0f}/100 — fluxo levemente negativo")

    # ── 6. Compute signals compostos [0,1] ───────────────────────────────────

    # Sell vol signal: GEX long + IV rich + calm + not squeeze
    sell_vol = 0.0
    if r.gex_regime in (GexRegime.LONG, GexRegime.LONG_STRONG):
        sell_vol += 0.35 if r.gex_regime == GexRegime.LONG_STRONG else 0.25
    if r.iv_regime in (IvRegime.RICH, IvRegime.VERY_RICH):
        sell_vol += 0.30 if r.iv_regime == IvRegime.VERY_RICH else 0.20
    if r.spot_regime in (SpotRegime.ABOVE_GAMMA_FLIP, SpotRegime.BETWEEN_WALLS):
        sell_vol += 0.10
    if skew > 3:
        sell_vol += 0.10   # mercado pagando bem pelas puts → vender put spread
    if sq > 60:
        sell_vol *= 0.5    # squeeeze cancela parcialmente sell vol
    if tail > 65:
        sell_vol *= 0.3
    r.sell_vol_signal = min(sell_vol, 1.0)

    # Buy vol signal: IV cheap OR GEX short extreme OR below gamma flip
    buy_vol = 0.0
    if r.iv_regime in (IvRegime.CHEAP, IvRegime.VERY_CHEAP):
        buy_vol += 0.35 if r.iv_regime == IvRegime.VERY_CHEAP else 0.20
    if r.gex_regime == GexRegime.SHORT_EXTREME:
        buy_vol += 0.35
    elif r.gex_regime == GexRegime.SHORT:
        buy_vol += 0.15
    if r.spot_regime == SpotRegime.BELOW_GAMMA_FLIP:
        buy_vol += 0.15
    if sq > 70:
        buy_vol += 0.15
    r.buy_vol_signal = min(buy_vol, 1.0)

    # Hedge signal: tail + squeeze + GEX extreme + spot below key levels
    hedge = 0.0
    hedge += min(tail / 100, 1.0) * 0.40
    hedge += min(sq / 100, 1.0) * 0.25
    if r.gex_regime == GexRegime.SHORT_EXTREME:
        hedge += 0.25
    elif r.gex_regime == GexRegime.SHORT:
        hedge += 0.10
    if r.spot_regime in (SpotRegime.BELOW_PUT_WALL, SpotRegime.BELOW_GAMMA_FLIP):
        hedge += 0.15
    if r.spot_regime == SpotRegime.NEAR_PUT_WALL:
        hedge += 0.05
    if flow < 35:
        hedge += 0.05
    r.hedge_signal = min(hedge, 1.0)

    # ── 7. Vol Decision final ────────────────────────────────────────────────
    if r.hedge_signal > 0.55 or tail > 65 or r.gex_regime == GexRegime.SHORT_EXTREME:
        r.vol_decision = VolDecision.HEDGE
    elif r.sell_vol_signal > 0.50 and r.buy_vol_signal < 0.30:
        r.vol_decision = VolDecision.SELL_VOL
    elif r.buy_vol_signal > 0.40 and r.sell_vol_signal < 0.35:
        r.vol_decision = VolDecision.BUY_VOL
    else:
        r.vol_decision = VolDecision.NEUTRAL

    # ── 8. Position scalar adjustment ────────────────────────────────────────
    adj = 0.0

    # GEX
    if r.gex_regime == GexRegime.SHORT_EXTREME:
        adj -= 0.15
    elif r.gex_regime == GexRegime.SHORT:
        adj -= 0.07
    elif r.gex_regime == GexRegime.LONG_STRONG:
        adj += 0.03   # mercado amortecido, ligeiramente mais risk-on

    # Tail
    if tail > 70:
        adj -= 0.12
    elif tail > 55:
        adj -= 0.06

    # Squeeze
    if sq > 80:
        adj -= 0.10
    elif sq > 65:
        adj -= 0.05

    # Flow negativo
    if flow < 30:
        adj -= 0.08
    elif flow < 40:
        adj -= 0.04

    # Spot abaixo de suportes
    if r.spot_regime == SpotRegime.BELOW_PUT_WALL:
        adj -= 0.08
    elif r.spot_regime == SpotRegime.BELOW_GAMMA_FLIP:
        adj -= 0.05

    # VIX do BBG (mais preciso): se muito alto, reduz mais
    vix_bbg = snapshot.vix
    if vix_bbg > 35:
        adj -= 0.10
    elif vix_bbg > 25:
        adj -= 0.05

    r.position_scalar_adj = max(-0.35, min(0.10, adj))  # limita entre -35% e +10%

    # ── 9. Hedge params ──────────────────────────────────────────────────────
    r.hedge_required = r.vol_decision == VolDecision.HEDGE or r.hedge_signal > 0.45

    if r.hedge_required:
        if r.gex_regime == GexRegime.SHORT_EXTREME or tail > 70:
            r.hedge_asset = "SPTS"           # ProShares Short S&P 500
            r.hedge_alloc_pct = 0.05 + min((tail - 50) / 100, 0.05)
        elif r.vol_decision == VolDecision.HEDGE:
            r.hedge_asset = "VXX"            # long vol
            r.hedge_alloc_pct = 0.03 + min((tail - 40) / 200, 0.04)
        else:
            r.hedge_asset = "BIL"            # cash/T-bills
            r.hedge_alloc_pct = 0.03

        r.hedge_alloc_pct = round(min(r.hedge_alloc_pct, 0.10), 4)

    # ── 10. Portfolio options strategy ───────────────────────────────────────
    if r.vol_decision == VolDecision.SELL_VOL:
        if r.iv_regime == IvRegime.VERY_RICH and r.gex_regime == GexRegime.LONG_STRONG:
            r.portfolio_options_strategy = "iron_condor"
            r.strategy_rationale = (
                f"IV-RV={iv_rv:+.1f}pp muito rico + GEX={gex:+.1f}B long gamma forte: "
                f"vender iron condor com strikes entre Put Wall ({pwall:,.0f}) e Call Wall ({cwall:,.0f})"
            )
        elif r.iv_regime == IvRegime.RICH:
            r.portfolio_options_strategy = "call_overwrite"
            r.strategy_rationale = (
                f"IV-RV={iv_rv:+.1f}pp rico: covered call overwrite nas posições long para capturar premium"
            )
    elif r.vol_decision == VolDecision.BUY_VOL:
        if r.iv_regime == IvRegime.VERY_CHEAP:
            r.portfolio_options_strategy = "straddle"
            r.strategy_rationale = (
                f"IV-RV={iv_rv:+.1f}pp muito barato + GEX={gex:+.1f}B: "
                f"straddle ATM para capturar move direcional sem aposta de lado"
            )
        else:
            r.portfolio_options_strategy = "calendar_spread"
            r.strategy_rationale = (
                f"IV-RV={iv_rv:+.1f}pp: calendar spread — comprar vol de prazo curto vs vender prazo longo"
            )
    elif r.vol_decision == VolDecision.HEDGE:
        r.portfolio_options_strategy = "spy_put_spread"
        r.strategy_rationale = (
            f"Tail={tail:.0f}/100 + Squeeze={sq:.0f}/100: "
            f"put spread de portfólio no SPX/SPY (Put Wall {pwall:,.0f} como strike vendido, "
            f"-5% como strike comprado). Hedge {r.hedge_alloc_pct:.0%} do portfolio."
        )

    # ── 11. Rationale final ───────────────────────────────────────────────────
    decision_labels = {
        VolDecision.SELL_VOL: "VENDER VOL",
        VolDecision.BUY_VOL:  "COMPRAR VOL",
        VolDecision.HEDGE:    "HEDGE",
        VolDecision.NEUTRAL:  "NEUTRO",
    }
    r.rationale = (
        f"[{decision_labels[r.vol_decision]}] GEX={gex:+.1f}B ({r.gex_regime}) | "
        f"IV-RV={iv_rv:+.1f}pp ({r.iv_regime}) | "
        f"Squeeze={sq:.0f} | Tail={tail:.0f} | Flow={flow:.0f} | "
        f"Position adj={r.position_scalar_adj:+.0%}. "
        + " ".join(reasons[:4])
    )

    _log.info(
        "vol_options_regime",
        decision=r.vol_decision,
        gex_regime=r.gex_regime,
        iv_regime=r.iv_regime,
        position_adj=round(r.position_scalar_adj, 3),
        hedge=r.hedge_required,
        hedge_alloc=r.hedge_alloc_pct,
        strategy=r.portfolio_options_strategy,
    )

    return r


# ── Helper para texto de regime no MacroDesk ──────────────────────────────────

def vol_decision_badge_html(r: VolOptionsRegime) -> str:
    """Retorna HTML de badge colorido para exibir no MacroDesk sidebar."""
    colors = {
        VolDecision.SELL_VOL: ("#22c55e", "VENDER VOL"),
        VolDecision.BUY_VOL:  ("#f59e0b", "COMPRAR VOL"),
        VolDecision.HEDGE:    ("#ef4444", "HEDGE"),
        VolDecision.NEUTRAL:  ("#6b7280", "NEUTRO"),
    }
    color, label = colors.get(r.vol_decision, ("#6b7280", "—"))

    gex_color = "#22c55e" if r.gex_net_bn > 0 else "#ef4444"
    iv_sign = "+" if r.iv_rv_pp >= 0 else ""

    return (
        f"<div style='background:rgba(0,0,0,.3);border:1px solid {color}40;"
        f"border-radius:6px;padding:8px 10px;margin-bottom:6px'>"
        f"<div style='font-size:10px;font-weight:700;color:{color};"
        f"letter-spacing:1px;margin-bottom:4px'>◈ {label}</div>"
        f"<div style='font-size:10px;color:#94a3b8;line-height:1.6'>"
        f"GEX <span style='color:{gex_color};font-weight:600'>{r.gex_net_bn:+.1f}B</span> · "
        f"IV-RV <span style='font-weight:600'>{iv_sign}{r.iv_rv_pp:.1f}pp</span> · "
        f"Sq <span style='color:{'#f59e0b' if r.squeeze_score>60 else '#94a3b8'}'>{r.squeeze_score:.0f}</span> · "
        f"Tail <span style='color:{'#ef4444' if r.tail_score>65 else '#94a3b8'}'>{r.tail_score:.0f}</span>"
        f"</div>"
        + (f"<div style='font-size:9px;color:{color};margin-top:3px'>{r.strategy_rationale[:80]}...</div>"
           if r.strategy_rationale else "")
        + "</div>"
    )
