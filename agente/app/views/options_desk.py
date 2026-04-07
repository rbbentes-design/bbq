"""
Options Desk — Aba "Opções ◈" do MacroDesk.

Sub-abas:
  1. Opções    — GEX, Key Levels, Greeks, Skew, Flow Score, JARVIS
  2. CTA       — Posicionamento CTA: crowding, momentum 1m/3m/6m/12m, squeeze candidates
  3. Dark Pool — Shadow Flow: dark pool score, volume ratio, unusual options
  4. Vol       — Regime de volatilidade, stress, VIX, IV rank por ativo
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.providers.options_store import OptionsSnapshot

# ── CSS ───────────────────────────────────────────────────────────────────────

_OD_CSS = """
<style>
.od-wrap {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace;
  color: #e2e8f0;
  padding: 20px 28px 60px;
  background: #020810;
  min-height: 100%;
  flex: 1;
  width: 100%;
  min-width: 0;
  box-sizing: border-box;
}
/* ── Sub-tabs ── */
.od-subtab-bar {
  display: flex; gap: 4px; margin-bottom: 20px;
  border-bottom: 1px solid rgba(0,212,232,.15); padding-bottom: 0;
}
.od-subtab {
  padding: 7px 18px; font-size: 11px; letter-spacing: 1px;
  text-transform: uppercase; border: none; background: transparent;
  color: rgba(0,212,232,.4); cursor: pointer; border-bottom: 2px solid transparent;
  margin-bottom: -1px; transition: color .2s, border-color .2s;
}
.od-subtab:hover { color: rgba(0,212,232,.8); }
.od-subtab.active { color: rgba(0,212,232,1); border-bottom-color: rgba(0,212,232,.8); }
.od-subpanel { display: none; }
.od-subpanel.active { display: block; }
/* ── Badges header ── */
.od-header-strip {
  display: flex; flex-wrap: wrap; gap: 8px;
  margin-bottom: 16px; align-items: center;
}
.od-badge {
  display: flex; flex-direction: column;
  background: rgba(0,212,232,.07);
  border: 1px solid rgba(0,212,232,.2);
  border-radius: 6px;
  padding: 8px 14px;
  min-width: 90px;
}
.od-badge-label {
  font-size: 9px; letter-spacing: 1.5px; color: rgba(0,212,232,.5);
  text-transform: uppercase; margin-bottom: 3px;
}
.od-badge-val {
  font-size: 18px; font-weight: 700; color: rgba(0,212,232,1);
  font-family: 'Share Tech Mono', monospace;
  letter-spacing: 1px;
}
.od-badge-val.od-up   { color: #22c55e; }
.od-badge-val.od-dn   { color: #ef4444; }
.od-badge-val.od-warn { color: #f59e0b; }
.od-ts {
  font-size: 10px; color: rgba(0,212,232,.3); margin-left: auto;
  align-self: flex-end; padding-bottom: 4px;
}
/* ── Grids ── */
.od-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.od-grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.od-grid-4 { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 16px; margin-bottom: 16px; }
/* ── Panel card ── */
.od-panel {
  background: rgba(0,6,18,.92);
  border: 1px solid rgba(0,212,232,.15);
  border-radius: 8px;
  padding: 16px 18px;
  position: relative;
  overflow: hidden;
}
.od-panel::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, rgba(0,212,232,.4), transparent);
}
.od-panel-title {
  font-size: 10px; letter-spacing: 1.5px; color: rgba(0,212,232,.5);
  text-transform: uppercase; margin-bottom: 12px;
  border-bottom: 1px solid rgba(0,212,232,.1); padding-bottom: 6px;
}
/* ── Level bar ── */
.od-level-bar {
  height: 4px; background: rgba(0,212,232,.08); border-radius: 2px;
  position: relative; margin: 6px 0 14px;
}
.od-level-bar-fill {
  position: absolute; top: 0; height: 100%;
  background: rgba(0,212,232,.5); border-radius: 2px;
  transition: width .3s;
}
.od-level-mark {
  position: absolute; top: -4px; width: 2px; height: 12px;
  border-radius: 1px;
}
.od-level-label {
  position: absolute; top: 14px; transform: translateX(-50%);
  font-size: 9px; color: rgba(0,212,232,.5); white-space: nowrap;
}
/* ── Z-score bar ── */
.od-z-row {
  display: flex; align-items: center; gap: 8px; margin-bottom: 6px;
}
.od-z-label {
  width: 80px; font-size: 10px; color: rgba(0,212,232,.6);
  text-transform: uppercase; letter-spacing: .5px; flex-shrink: 0;
}
.od-z-bar-wrap {
  flex: 1; height: 12px; background: rgba(0,212,232,.06); border-radius: 3px;
  position: relative; overflow: hidden;
}
.od-z-bar-fill {
  position: absolute; top: 0; height: 100%; border-radius: 3px;
  transition: width .3s;
}
.od-z-val {
  width: 38px; font-size: 10px; text-align: right;
  color: rgba(0,212,232,.8); font-family: monospace; flex-shrink: 0;
}
.od-score-total {
  text-align: center; padding: 10px 0;
  font-size: 28px; font-weight: 700; color: rgba(0,212,232,1);
  font-family: 'Share Tech Mono', monospace;
}
.od-score-label {
  text-align: center; font-size: 9px;
  color: rgba(0,212,232,.4); letter-spacing: 2px; text-transform: uppercase;
}
/* ── Greek row ── */
.od-greek-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 7px 0; border-bottom: 1px solid rgba(0,212,232,.06);
}
.od-greek-name { font-size: 11px; color: rgba(0,212,232,.7); letter-spacing: .5px; }
.od-greek-val  { font-size: 16px; font-weight: 700; font-family: monospace; }
/* ── CTA bar ── */
.od-cta-row {
  display: flex; align-items: center; gap: 8px; margin-bottom: 7px;
}
.od-cta-ticker {
  width: 52px; font-size: 11px; color: #e2e8f0; font-family: monospace;
  font-weight: 700; flex-shrink: 0;
}
.od-cta-name {
  width: 100px; font-size: 9px; color: rgba(0,212,232,.45);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex-shrink: 0;
}
.od-cta-bar-wrap {
  flex: 1; height: 10px; background: rgba(255,255,255,.06); border-radius: 3px;
  position: relative; overflow: hidden;
}
.od-cta-bar-fill {
  position: absolute; top: 0; height: 100%; border-radius: 3px;
}
.od-cta-score {
  width: 42px; font-size: 10px; text-align: right;
  font-family: monospace; flex-shrink: 0;
}
.od-cta-crowd {
  width: 80px; font-size: 9px; letter-spacing: .5px; text-align: right;
  flex-shrink: 0; text-transform: uppercase;
}
/* ── Momentum mini bars ── */
.od-mom-grid {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
  margin-bottom: 16px;
}
.od-mom-card {
  background: rgba(0,212,232,.04); border: 1px solid rgba(0,212,232,.1);
  border-radius: 6px; padding: 10px 12px; text-align: center;
}
.od-mom-label { font-size: 9px; color: rgba(0,212,232,.4); letter-spacing: 1px; margin-bottom: 6px; }
.od-mom-val   { font-size: 18px; font-weight: 700; font-family: monospace; }
/* ── Dark pool row ── */
.od-dp-row {
  display: flex; align-items: center; gap: 8px; margin-bottom: 6px; padding: 5px 0;
  border-bottom: 1px solid rgba(255,255,255,.04);
}
.od-dp-ticker { width: 52px; font-size: 11px; font-family: monospace; font-weight: 700; flex-shrink: 0; }
.od-dp-bar-wrap {
  flex: 1; height: 8px; background: rgba(255,255,255,.06); border-radius: 3px;
  position: relative; overflow: hidden;
}
.od-dp-bar-fill { position: absolute; top: 0; height: 100%; border-radius: 3px; }
.od-dp-score { width: 38px; font-size: 10px; text-align: right; font-family: monospace; flex-shrink: 0; }
.od-dp-src   { width: 52px; font-size: 9px; color: rgba(0,212,232,.3); text-align: right; flex-shrink: 0; }
/* ── Vol regime ── */
.od-regime-big {
  display: flex; align-items: center; gap: 16px; padding: 16px 0; margin-bottom: 12px;
  border-bottom: 1px solid rgba(0,212,232,.1);
}
.od-regime-badge {
  font-size: 24px; font-weight: 900; letter-spacing: 2px; font-family: monospace;
  padding: 8px 20px; border-radius: 8px; border: 1px solid;
}
.od-stress-bar-wrap {
  flex: 1; height: 14px; background: rgba(255,255,255,.06);
  border-radius: 4px; position: relative; overflow: hidden;
}
.od-stress-bar-fill {
  position: absolute; top: 0; left: 0; height: 100%; border-radius: 4px;
  transition: width .4s;
}
/* ── Empty ── */
.od-empty {
  text-align: center; padding: 80px 0;
  color: rgba(0,212,232,.25); font-size: 13px; letter-spacing: 1px;
}
.od-empty b { display: block; font-size: 20px; margin-bottom: 8px; }
/* ── JARVIS embed ── */
.od-jarvis-wrap {
  width: 100%; border-radius: 6px; overflow: hidden;
  border: 1px solid rgba(0,212,232,.15); margin-top: 12px;
}
.od-jarvis-wrap iframe { width: 100%; height: 900px; border: none; display: block; }
</style>
"""

_OD_JS = """
<script>
function odSwitchTab(name, btn) {
  document.querySelectorAll('.od-subtab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.od-subpanel').forEach(p => p.classList.remove('active'));
  if (btn) btn.classList.add('active');
  var el = document.getElementById('od-panel-' + name);
  if (el) el.classList.add('active');
}
</script>
"""


# ── Renderer principal ────────────────────────────────────────────────────────

def render_options_tab(
    snapshot: "OptionsSnapshot | None",
    jarvis_html: str | None = None,
    cta_result=None,        # CTAPositioningResult | None
    shadow_flow=None,       # ShadowFlowResult | None
    vol_regime=None,        # VolRegimeResult | None
    signals: dict | None = None,  # dict[str, AssetSignal]
) -> str:
    sigs = signals or {}

    subtab_bar = (
        "<div class='od-subtab-bar'>"
        "<button class='od-subtab active' onclick=\"odSwitchTab('options',this)\">Opções</button>"
        "<button class='od-subtab' onclick=\"odSwitchTab('cta',this)\">CTA</button>"
        "<button class='od-subtab' onclick=\"odSwitchTab('darkpool',this)\">Dark Pool</button>"
        "<button class='od-subtab' onclick=\"odSwitchTab('vol',this)\">Volatilidade</button>"
        "</div>"
    )

    panel_options  = _build_options_panel(snapshot, jarvis_html)
    panel_cta      = _build_cta_panel(cta_result, sigs)
    panel_darkpool = _build_darkpool_panel(shadow_flow, sigs)
    panel_vol      = _build_vol_panel(vol_regime, sigs, snapshot)

    return (
        f"{_OD_CSS}"
        "<div class='od-wrap'>"
        + subtab_bar
        + f"<div id='od-panel-options' class='od-subpanel active'>{panel_options}</div>"
        + f"<div id='od-panel-cta' class='od-subpanel'>{panel_cta}</div>"
        + f"<div id='od-panel-darkpool' class='od-subpanel'>{panel_darkpool}</div>"
        + f"<div id='od-panel-vol' class='od-subpanel'>{panel_vol}</div>"
        + "</div>"
        + _OD_JS
    )


# ═══════════════════════════════════════════════════════
# ABA 1 — OPÇÕES (conteúdo original)
# ═══════════════════════════════════════════════════════

def _build_options_panel(snapshot: "OptionsSnapshot | None", jarvis_html: str | None) -> str:
    if snapshot is None:
        return (
            "<div class='od-empty'>"
            "<b>◈ Sem dados de opções</b>"
            "Baixe o ZIP do BQuant e coloque em ~/Downloads.<br>"
            "<code>greeks_SPX_YYYYMMDD.zip</code> — auto-importado no próximo run."
            "</div>"
        )
    panels = [
        _header_strip(snapshot),
        f"<div class='od-grid-3'>{_panel_levels(snapshot)}{_panel_greeks(snapshot)}{_panel_skew(snapshot)}</div>",
        _panel_flow_score(snapshot),
    ]
    if jarvis_html:
        panels.append(_panel_jarvis(jarvis_html))
    return "".join(panels)


# ═══════════════════════════════════════════════════════
# ABA 2 — CTA POSITIONING
# ═══════════════════════════════════════════════════════

def _build_cta_panel(cta_result, signals: dict) -> str:
    if cta_result is None:
        return "<div class='od-empty'><b>CTA</b>Dados CTA não disponíveis neste run.</div>"

    # ── Aggregate scores ──────────────────────────────────────────────────────
    eq  = getattr(cta_result, "aggregate_equity_score",    0) or 0
    bd  = getattr(cta_result, "aggregate_bond_score",      0) or 0
    cmd = getattr(cta_result, "aggregate_commodity_score", 0) or 0
    ext_longs  = getattr(cta_result, "extreme_longs",  []) or []
    ext_shorts = getattr(cta_result, "extreme_shorts", []) or []
    squeezes   = getattr(cta_result, "squeeze_candidates", []) or []

    def agg_card(label: str, val: float, color: str) -> str:
        pct = (val + 1) / 2 * 100
        bar_color = "#22c55e" if val > 0.2 else ("#ef4444" if val < -0.2 else "#f59e0b")
        return (
            f"<div class='od-panel'>"
            f"<div class='od-panel-title'>{label}</div>"
            f"<div style='font-size:28px;font-weight:900;font-family:monospace;color:{bar_color};margin-bottom:8px'>"
            f"{val:+.2f}</div>"
            f"<div style='height:6px;background:rgba(255,255,255,.06);border-radius:3px;overflow:hidden'>"
            f"<div style='width:{pct:.0f}%;height:100%;background:{bar_color};border-radius:3px'></div>"
            f"</div>"
            f"<div style='font-size:9px;color:rgba(255,255,255,.3);margin-top:4px'>-1 = max short · +1 = max long</div>"
            f"</div>"
        )

    agg_row = (
        f"<div class='od-grid-3' style='margin-bottom:20px'>"
        + agg_card("CTA Equities", eq, "#22c55e")
        + agg_card("CTA Bonds", bd, "#38bdf8")
        + agg_card("CTA Commodities", cmd, "#f59e0b")
        + "</div>"
    )

    # ── Chips de extremos ─────────────────────────────────────────────────────
    def chips(tickers: list[str], color: str, label: str) -> str:
        if not tickers:
            return ""
        c = "".join(
            f"<span style='background:{color}22;border:1px solid {color}66;"
            f"border-radius:4px;padding:2px 8px;font-size:10px;font-family:monospace;color:{color}'>"
            f"{t}</span>"
            for t in tickers
        )
        return (
            f"<div style='margin-bottom:10px'>"
            f"<div style='font-size:9px;color:rgba(255,255,255,.3);letter-spacing:1px;"
            f"text-transform:uppercase;margin-bottom:4px'>{label}</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:5px'>{c}</div>"
            f"</div>"
        )

    chips_html = (
        f"<div class='od-panel' style='margin-bottom:16px'>"
        f"<div class='od-panel-title'>Posicionamento Extremo</div>"
        + chips(ext_longs,  "#22c55e", "Extreme Long — squeeze risk se virar")
        + chips(ext_shorts, "#ef4444", "Extreme Short — squeeze risk se subir")
        + chips(squeezes,   "#f59e0b", "Squeeze Candidates")
        + "</div>"
    )

    # ── Ranking por ativo ─────────────────────────────────────────────────────
    sigs_dict = getattr(cta_result, "signals", {}) or {}
    sorted_sigs = sorted(sigs_dict.values(), key=lambda s: abs(getattr(s, "cta_score", 0)), reverse=True)

    crowd_colors = {
        "extreme_long":  "#22c55e",
        "long":          "#86efac",
        "neutral":       "#6b7280",
        "short":         "#fca5a5",
        "extreme_short": "#ef4444",
    }

    rows_html = ""
    for sig in sorted_sigs[:20]:
        score  = getattr(sig, "cta_score", 0) or 0
        crowd  = getattr(sig, "crowding", "neutral") or "neutral"
        ticker = getattr(sig, "ticker", "?")
        name   = getattr(sig, "name", "")
        m1     = getattr(sig, "momentum_1m",  None)
        m3     = getattr(sig, "momentum_3m",  None)
        m6     = getattr(sig, "momentum_6m",  None)
        m12    = getattr(sig, "momentum_12m", None)

        bar_pct   = (score + 1) / 2 * 100
        bar_color = "#22c55e" if score > 0.2 else ("#ef4444" if score < -0.2 else "#6b7280")
        cc        = crowd_colors.get(crowd, "#6b7280")

        mom_cells = ""
        for lbl, val in [("1M", m1), ("3M", m3), ("6M", m6), ("12M", m12)]:
            vc = "#22c55e" if (val or 0) > 0 else "#ef4444"
            vs = f"{val:+.1%}" if val is not None else "—"
            mom_cells += (
                f"<span style='font-size:9px;color:{vc};font-family:monospace;"
                f"background:{vc}15;border-radius:3px;padding:1px 5px'>{lbl} {vs}</span>"
            )

        rows_html += (
            f"<div class='od-cta-row'>"
            f"<span class='od-cta-ticker'>{ticker}</span>"
            f"<span class='od-cta-name'>{name}</span>"
            f"<div class='od-cta-bar-wrap'>"
            f"<div style='position:absolute;top:0;left:50%;width:1px;height:100%;background:rgba(255,255,255,.1)'></div>"
            f"<div class='od-cta-bar-fill' style='left:{min(bar_pct,50):.1f}%;width:{abs(bar_pct-50):.1f}%;background:{bar_color}'></div>"
            f"</div>"
            f"<span class='od-cta-score' style='color:{bar_color}'>{score:+.2f}</span>"
            f"<span class='od-cta-crowd' style='color:{cc};font-size:8px'>{crowd.replace('_',' ')}</span>"
            f"</div>"
            f"<div style='padding-left:52px;display:flex;gap:5px;flex-wrap:wrap;margin-bottom:8px'>{mom_cells}</div>"
        )

    ranking = (
        f"<div class='od-panel'>"
        f"<div class='od-panel-title'>Ranking CTA por Ativo — Momentum Score</div>"
        + rows_html
        + "</div>"
    )

    return agg_row + chips_html + ranking


# ═══════════════════════════════════════════════════════
# ABA 3 — DARK POOL / SHADOW FLOW
# ═══════════════════════════════════════════════════════

def _build_darkpool_panel(shadow_flow, signals: dict) -> str:
    if shadow_flow is None:
        return "<div class='od-empty'><b>Dark Pool</b>Shadow Flow não disponível neste run.</div>"

    sigs_dict   = getattr(shadow_flow, "signals",     {}) or {}
    top_bullish = getattr(shadow_flow, "top_bullish", []) or []
    top_bearish = getattr(shadow_flow, "top_bearish", []) or []

    # ── Chips top bullish / bearish ───────────────────────────────────────────
    def chip_row(tickers: list[str], color: str, label: str) -> str:
        if not tickers:
            return ""
        chips = "".join(
            f"<span style='background:{color}20;border:1px solid {color}50;"
            f"border-radius:4px;padding:2px 8px;font-size:10px;font-family:monospace;color:{color}'>{t}</span>"
            for t in tickers[:8]
        )
        return (
            f"<div style='margin-bottom:8px'>"
            f"<div style='font-size:9px;color:rgba(255,255,255,.3);letter-spacing:1px;text-transform:uppercase;margin-bottom:4px'>{label}</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:5px'>{chips}</div>"
            f"</div>"
        )

    summary = (
        f"<div class='od-panel' style='margin-bottom:16px'>"
        f"<div class='od-panel-title'>Dark Pool — Fluxo Institucional</div>"
        + chip_row(top_bullish, "#22c55e", "Acumulação — Dark Pool Buy")
        + chip_row(top_bearish, "#ef4444", "Distribuição — Dark Pool Sell")
        + "</div>"
    )

    # ── Barras por ativo ──────────────────────────────────────────────────────
    sorted_dp = sorted(
        sigs_dict.values(),
        key=lambda s: abs(getattr(s, "dark_pool_score", 0)),
        reverse=True,
    )

    rows = ""
    for dp in sorted_dp[:25]:
        ticker = getattr(dp, "ticker", "?")
        score  = getattr(dp, "dark_pool_score", 0) or 0
        vol_r  = getattr(dp, "unusual_volume_ratio", 1) or 1
        opt_s  = getattr(dp, "options_sentiment", "neutral") or "neutral"
        sweeps = getattr(dp, "n_sweeps", 0) or 0
        source = getattr(dp, "source", "vol") or "vol"

        bar_color = "#22c55e" if score > 0 else "#ef4444"
        bar_pct   = abs(score) * 50  # max 50% of half
        bar_left  = 50 if score >= 0 else 50 - bar_pct

        os_color  = {"bullish": "#22c55e", "bearish": "#ef4444", "neutral": "#6b7280"}.get(opt_s, "#6b7280")
        vol_color = "#f59e0b" if vol_r > 2 else ("#22c55e" if vol_r > 1.3 else "#6b7280")

        rows += (
            f"<div class='od-dp-row'>"
            f"<span class='od-dp-ticker' style='color:{bar_color}'>{ticker}</span>"
            f"<div class='od-dp-bar-wrap'>"
            f"<div style='position:absolute;top:0;left:50%;width:1px;height:100%;background:rgba(255,255,255,.1)'></div>"
            f"<div class='od-dp-bar-fill' style='left:{bar_left:.1f}%;width:{bar_pct:.1f}%;background:{bar_color}'></div>"
            f"</div>"
            f"<span class='od-dp-score' style='color:{bar_color}'>{score:+.2f}</span>"
            f"<span style='width:36px;font-size:9px;text-align:right;color:{vol_color};font-family:monospace;flex-shrink:0'>{vol_r:.1f}x</span>"
            f"<span style='width:50px;font-size:9px;text-align:right;color:{os_color};text-transform:uppercase;flex-shrink:0'>{opt_s[:4]}</span>"
            + (f"<span style='width:30px;font-size:9px;text-align:right;color:#f59e0b;flex-shrink:0'>{sweeps}&uarr;</span>" if sweeps > 0 else "<span style='width:30px'></span>")
            + "</div>"
        )

    header_row = (
        "<div style='display:flex;gap:8px;font-size:9px;color:rgba(0,212,232,.3);"
        "letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;padding-bottom:4px;"
        "border-bottom:1px solid rgba(0,212,232,.08)'>"
        "<span style='width:52px'>Ticker</span>"
        "<span style='flex:1;text-align:center'>Dark Pool Score (-1 sell · +1 buy)</span>"
        "<span style='width:38px;text-align:right'>Score</span>"
        "<span style='width:36px;text-align:right'>Vol</span>"
        "<span style='width:50px;text-align:right'>Options</span>"
        "<span style='width:30px;text-align:right'>Swp</span>"
        "</div>"
    )

    ranking = (
        f"<div class='od-panel'>"
        f"<div class='od-panel-title'>Dark Pool Score por Ativo</div>"
        + header_row + rows
        + "</div>"
    )

    return summary + ranking


# ═══════════════════════════════════════════════════════
# ABA 4 — VOLATILIDADE
# ═══════════════════════════════════════════════════════

def _build_vol_panel(vol_regime, signals: dict, snapshot: "OptionsSnapshot | None") -> str:
    parts = []

    # ── Regime de vol ─────────────────────────────────────────────────────────
    if vol_regime is not None:
        regime     = getattr(vol_regime, "regime",       "unknown") or "unknown"
        stress     = getattr(vol_regime, "stress_score", 0)  or 0
        vix        = getattr(vol_regime, "vix_spot",     None)
        vix3m      = getattr(vol_regime, "vix3m",        None)
        rvol       = getattr(vol_regime, "realized_vol", None)
        ivol       = getattr(vol_regime, "implied_vol",  None)
        iv_pct     = getattr(vol_regime, "iv_percentile",None)
        term_slope = getattr(vol_regime, "term_slope",   None)
        drivers    = getattr(vol_regime, "drivers",      []) or []

        regime_colors = {
            "calm":   ("#065f46", "#4ade80"),
            "normal": ("#1e3a5f", "#38bdf8"),
            "stress": ("#7f1d1d", "#f87171"),
            "panic":  ("#581c87", "#c084fc"),
        }
        bg, fg = regime_colors.get(regime, ("#1f2937", "#9ca3af"))
        stress_color = "#22c55e" if stress < 0.3 else ("#f59e0b" if stress < 0.6 else "#ef4444")

        def vrow(label: str, val: Any, fmt: str = ".2f", suffix: str = "") -> str:
            vs = f"{val:{fmt}}{suffix}" if val is not None else "—"
            return (
                f"<div style='display:flex;justify-content:space-between;padding:6px 0;"
                f"border-bottom:1px solid rgba(255,255,255,.04)'>"
                f"<span style='font-size:10px;color:rgba(0,212,232,.5)'>{label}</span>"
                f"<span style='font-size:12px;font-weight:600;font-family:monospace;color:#e2e8f0'>{vs}</span>"
                f"</div>"
            )

        driver_chips = "".join(
            f"<span style='font-size:9px;background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.3);"
            f"border-radius:3px;padding:2px 7px;color:#fca5a5'>{d}</span>"
            for d in drivers[:6]
        )

        vol_block = (
            f"<div class='od-panel' style='margin-bottom:16px'>"
            f"<div class='od-panel-title'>Regime de Volatilidade</div>"
            f"<div class='od-regime-big'>"
            f"<div class='od-regime-badge' style='background:{bg};border-color:{fg};color:{fg}'>"
            f"{regime.upper()}</div>"
            f"<div style='flex:1'>"
            f"<div style='font-size:10px;color:rgba(255,255,255,.4);margin-bottom:4px'>Stress Score</div>"
            f"<div class='od-stress-bar-wrap'>"
            f"<div class='od-stress-bar-fill' style='width:{stress*100:.0f}%;background:{stress_color}'></div>"
            f"</div>"
            f"<div style='font-size:11px;font-family:monospace;color:{stress_color};margin-top:3px'>{stress:.2f}</div>"
            f"</div></div>"
            + (f"<div style='display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px'>{driver_chips}</div>" if driver_chips else "")
            + f"<div class='od-grid-2'>"
            + f"<div>" + vrow("VIX Spot", vix, ".2f") + vrow("VIX 3M", vix3m, ".2f") + vrow("Term Slope", term_slope, "+.3f") + "</div>"
            + f"<div>" + vrow("IV 30D", ivol, ".1%") + vrow("RV 30D", rvol, ".1%") + vrow("IV Percentile", iv_pct, ".0%") + "</div>"
            + "</div></div>"
        )
        parts.append(vol_block)

    # ── IV rank por ativo (dos alpha signals) ─────────────────────────────────
    iv_rows = []
    for ticker, sig in sorted(signals.items(), key=lambda x: (getattr(x[1], "iv_percentile", None) or 0), reverse=True):
        iv_pct_s = getattr(sig, "iv_percentile", None)
        skew_s   = getattr(sig, "skew_5pct",     None)
        if iv_pct_s is None:
            continue
        bar_color = "#22c55e" if iv_pct_s < 0.4 else ("#f59e0b" if iv_pct_s < 0.7 else "#ef4444")
        skew_color = "#22c55e" if (skew_s or 0) < -0.01 else ("#ef4444" if (skew_s or 0) > 0.02 else "#6b7280")
        skew_s_fmt = f"{skew_s:+.2%}" if skew_s is not None else "—"
        iv_rows.append(
            f"<div class='od-dp-row'>"
            f"<span class='od-dp-ticker' style='color:{bar_color}'>{ticker}</span>"
            f"<div class='od-dp-bar-wrap'>"
            f"<div class='od-dp-bar-fill' style='left:0;width:{iv_pct_s*100:.0f}%;background:{bar_color}'></div>"
            f"</div>"
            f"<span class='od-dp-score' style='color:{bar_color}'>{iv_pct_s:.0%}</span>"
            f"<span style='width:60px;font-size:9px;text-align:right;color:{skew_color};font-family:monospace;flex-shrink:0'>{skew_s_fmt}</span>"
            f"</div>"
        )

    if iv_rows:
        iv_header = (
            "<div style='display:flex;gap:8px;font-size:9px;color:rgba(0,212,232,.3);"
            "letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;padding-bottom:4px;"
            "border-bottom:1px solid rgba(0,212,232,.08)'>"
            "<span style='width:52px'>Ticker</span>"
            "<span style='flex:1'>IV Rank (verde=barata · vermelho=cara)</span>"
            "<span style='width:38px;text-align:right'>IV%</span>"
            "<span style='width:60px;text-align:right'>Skew 5%</span>"
            "</div>"
        )
        parts.append(
            f"<div class='od-panel'>"
            f"<div class='od-panel-title'>IV Rank por Ativo — Fonte: Alpha Signals</div>"
            + iv_header + "".join(iv_rows)
            + "</div>"
        )

    # ── GEX / squeeze da aba opções (se disponível) ───────────────────────────
    if snapshot is not None:
        sq    = snapshot.squeeze_score
        tail  = snapshot.tail_score
        gex   = snapshot.gex_net_bn
        sq_c  = "#ef4444" if sq > 80 else ("#f59e0b" if sq > 60 else "#22c55e")
        tl_c  = "#ef4444" if tail > 70 else ("#f59e0b" if tail > 50 else "#22c55e")
        gex_c = "#22c55e" if gex > 0 else "#ef4444"
        parts.append(
            f"<div class='od-panel'>"
            f"<div class='od-panel-title'>Indicadores de Risco — SPX Options</div>"
            f"<div class='od-grid-3'>"
            + _stat_card("Squeeze Score", f"{sq:.0f}/100", sq_c)
            + _stat_card("Tail Risk",     f"{tail:.0f}/100", tl_c)
            + _stat_card("GEX Net",       f"{gex:+.1f}B", gex_c)
            + "</div></div>"
        )

    if not parts:
        return "<div class='od-empty'><b>Volatilidade</b>Dados de regime de vol não disponíveis.</div>"

    return "".join(parts)


def _stat_card(label: str, val: str, color: str) -> str:
    return (
        f"<div style='text-align:center;padding:12px 0'>"
        f"<div style='font-size:9px;color:rgba(0,212,232,.4);letter-spacing:1px;text-transform:uppercase;margin-bottom:6px'>{label}</div>"
        f"<div style='font-size:26px;font-weight:900;font-family:monospace;color:{color}'>{val}</div>"
        f"</div>"
    )


# ═══════════════════════════════════════════════════════
# PAINÉIS INTERNOS DA ABA OPÇÕES (originais)
# ═══════════════════════════════════════════════════════

def _coleta_badge(snap: "OptionsSnapshot") -> str:
    from datetime import datetime
    ts_str = snap.ts or ""
    try:
        ts_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
        now = datetime.now()
        age_h = (now - ts_dt).total_seconds() / 3600
        if age_h > 36:
            age_label = f"{age_h/24:.0f}d atrás"; color = "#ef4444"; bg = "rgba(239,68,68,.12)"
        elif age_h > 8:
            age_label = f"{age_h:.0f}h atrás";    color = "#f59e0b"; bg = "rgba(245,158,11,.12)"
        else:
            age_label = f"{age_h:.0f}h atrás";    color = "#22c55e"; bg = "rgba(34,197,94,.12)"
        age_html = f" <span style='color:{color};font-size:9px'>({age_label})</span>"
    except Exception:
        bg = "rgba(0,212,232,.07)"; color = "rgba(0,212,232,.6)"; age_html = ""

    return (
        f"<div style='margin-left:auto;display:flex;flex-direction:column;"
        f"background:{bg};border:1px solid {color}40;border-radius:6px;"
        f"padding:6px 12px;align-self:center'>"
        f"<span style='font-size:9px;letter-spacing:1px;color:{color};text-transform:uppercase'>Coleta</span>"
        f"<span style='font-size:12px;font-weight:700;color:{color}'>{ts_str}{age_html}</span>"
        f"<span style='font-size:9px;color:{color}80'>{snap.ticker}</span>"
        f"</div>"
    )


def _header_strip(snap: "OptionsSnapshot") -> str:
    def badge(label: str, val: str, cls: str = "") -> str:
        return (
            f"<div class='od-badge'>"
            f"<span class='od-badge-label'>{label}</span>"
            f"<span class='od-badge-val {cls}'>{val}</span>"
            f"</div>"
        )
    gex    = snap.gex_net_bn
    gex_cls = "od-up" if gex > 0 else "od-dn"
    gex_s   = f"{'+' if gex >= 0 else ''}{gex:.1f}B"
    iv_rv   = snap.iv_rv_pp
    iv_rv_cls = "od-up" if iv_rv > 2 else ("od-dn" if iv_rv < -2 else "")
    sq  = snap.squeeze_score
    sq_cls  = "od-warn" if sq > 60 else ("od-dn" if sq > 80 else "")
    tail    = snap.tail_score
    tail_cls = "od-warn" if tail > 50 else ("od-dn" if tail > 70 else "")
    flip    = snap.gamma_flip
    spot    = snap.spot
    return (
        f"<div class='od-header-strip'>"
        + badge("SPOT",      f"{spot:,.0f}")
        + badge("GAMMA FLIP", f"{flip:,.0f}" if flip else "N/A",
                "od-warn" if flip and abs(spot - flip) < spot * 0.005 else "")
        + badge("GEX NET",   gex_s, gex_cls)
        + badge("P/C RATIO", f"{snap.pc_ratio:.2f}×")
        + badge("VIX",       f"{snap.vix:.2f}" if snap.vix else "—")
        + badge("IV 30D",    f"{snap.iv_30d*100:.2f}%")
        + badge("RV 30D",    f"{snap.rv_30d*100:.2f}%")
        + badge("IV − RV",   f"{iv_rv:+.1f}pp", iv_rv_cls)
        + badge("SQUEEZE",   f"{sq:.0f}/100", sq_cls)
        + badge("TAIL RISK", f"{tail:.0f}/100", tail_cls)
        + _coleta_badge(snap)
        + "</div>"
    )


def _panel_levels(snap: "OptionsSnapshot") -> str:
    spot = snap.spot; call_wall = snap.call_wall; put_wall = snap.put_wall; flip = snap.gamma_flip
    lo  = min(p for p in [put_wall, flip, spot] if p > 0) * 0.995
    hi  = max(p for p in [call_wall, flip, spot] if p > 0) * 1.005
    rng = hi - lo or 1
    def pct(v): return max(0, min(100, (v - lo) / rng * 100)) if v else 50
    def mark(v, color, label):
        if not v: return ""
        p = pct(v)
        return (
            f"<div class='od-level-mark' style='left:{p:.1f}%;background:{color}'></div>"
            f"<div class='od-level-label' style='left:{p:.1f}%'>{label}<br>{v:,.0f}</div>"
        )
    bar_html = (
        f"<div class='od-level-bar' style='margin-top:30px;margin-bottom:35px'>"
        + mark(put_wall, "#ef4444", "PUT WALL") + mark(flip, "#f59e0b", "G-FLIP")
        + mark(spot, "#00d4e8", "SPOT") + mark(call_wall, "#22c55e", "CALL WALL")
        + "</div>"
    )
    dist_flip = ((spot - flip) / flip * 100) if flip else 0
    dist_cw   = ((call_wall - spot) / spot * 100) if call_wall else 0
    dist_pw   = ((spot - put_wall) / spot * 100) if put_wall else 0
    def row(label, val, color=""):
        fc = color or "#e2e8f0"
        return (f"<tr><td style='color:rgba(0,212,232,.5);font-size:10px;padding:3px 8px 3px 0'>{label}</td>"
                f"<td style='font-size:12px;font-weight:600;color:{fc};font-family:monospace'>{val}</td></tr>")
    table = (
        "<table style='width:100%;border-collapse:collapse'>"
        + row("Spot",       f"{spot:,.0f}", "#00d4e8")
        + row("Call Wall",  f"{call_wall:,.0f}  (+{dist_cw:.1f}%)" if call_wall else "—", "#22c55e")
        + row("Put Wall",   f"{put_wall:,.0f}  (−{dist_pw:.1f}%)" if put_wall else "—", "#ef4444")
        + row("Gamma Flip", f"{flip:,.0f}  ({dist_flip:+.1f}%)" if flip else "—", "#f59e0b")
        + "</table>"
    )
    return f"<div class='od-panel'><div class='od-panel-title'>Key Levels</div>" + bar_html + table + "</div>"


def _panel_greeks(snap: "OptionsSnapshot") -> str:
    def row(name, val, unit="bn", color=""):
        cls = "od-up" if val > 0 else ("od-dn" if val < 0 else "")
        return (
            f"<div class='od-greek-row'>"
            f"<span class='od-greek-name'>{name}</span>"
            f"<span class='od-greek-val {cls}'>{val:+.2f} <small style='font-size:10px;opacity:.5'>{unit}</small></span>"
            f"</div>"
        )
    frag = getattr(snap, "fragility_score", None)
    frag_pct = (frag or 0) * 100
    frag_cls = "od-dn" if frag_pct > 60 else ("od-warn" if frag_pct > 30 else "od-up")
    return (
        f"<div class='od-panel'><div class='od-panel-title'>Greeks (Dealer Exposure)</div>"
        + row("Delta", snap.delta_bn)
        + row("Vanna", snap.vanna_bn)
        + row("Charm", snap.charm_bn)
        + (f"<div class='od-greek-row'><span class='od-greek-name'>Fragility</span>"
           f"<span class='od-greek-val {frag_cls}'>{frag_pct:.1f}"
           f"<small style='font-size:10px;opacity:.5'>/100</small></span></div>" if frag is not None else "")
        + "</div>"
    )


def _panel_skew(snap: "OptionsSnapshot") -> str:
    skew  = snap.skew_25d; iv_rv = snap.iv_rv_pp
    iv    = snap.iv_30d * 100; rv = snap.rv_30d * 100
    skew_cls  = "od-dn" if skew > 3 else ("od-up" if skew < 1 else "")
    ivrv_cls  = "od-up" if iv_rv > 3 else ("od-dn" if iv_rv < -1 else "")
    def metric(label, val, cls=""):
        return (
            f"<div style='padding:8px 0;border-bottom:1px solid rgba(0,212,232,.06)'>"
            f"<div style='font-size:9px;color:rgba(0,212,232,.45);letter-spacing:1px;text-transform:uppercase;margin-bottom:3px'>{label}</div>"
            f"<div class='od-greek-val {cls}' style='font-size:16px'>{val}</div>"
            f"</div>"
        )
    return (
        f"<div class='od-panel'><div class='od-panel-title'>Volatility / Skew</div>"
        + metric("IV 30D", f"{iv:.2f}%")
        + metric("RV 30D", f"{rv:.2f}%")
        + metric("IV − RV Premium", f"{iv_rv:+.1f}pp", ivrv_cls)
        + metric("25D Put Skew", f"{skew:+.1f}pp", skew_cls)
        + "</div>"
    )


def _panel_flow_score(snap: "OptionsSnapshot") -> str:
    z = snap.z_scores; w = snap.weights; total = snap.flow_score_total
    labels = {"cta":"CTA","dealer":"Dealer","volctrl":"Vol Ctrl","rp":"Risk Parity",
              "leveraged":"Leveraged","passive_etf":"Passive ETF","buyback":"Buyback","cot":"COT"}
    def z_bar(key):
        z_val = z.get(key, 0); w_val = w.get(key, 0); label = labels.get(key, key)
        normalized = max(-3, min(3, z_val)); center = 50
        bar_w = abs(normalized) / 3 * 50
        bar_left = center if normalized >= 0 else center - bar_w
        bar_color = "#22c55e" if normalized >= 0 else "#ef4444"
        z_color   = "#22c55e" if z_val > 0 else ("#ef4444" if z_val < 0 else "#94a3b8")
        return (
            f"<div class='od-z-row'>"
            f"<span class='od-z-label'>{label}</span>"
            f"<div class='od-z-bar-wrap'>"
            f"<div style='position:absolute;top:0;left:50%;width:1px;height:100%;background:rgba(0,212,232,.2)'></div>"
            f"<div class='od-z-bar-fill' style='left:{bar_left:.1f}%;width:{bar_w:.1f}%;background:{bar_color}'></div>"
            f"</div>"
            f"<span class='od-z-val' style='color:{z_color}'>{z_val:+.2f}</span>"
            f"</div>"
        )
    total_cls = "od-up" if total > 60 else ("od-dn" if total < 40 else "od-warn")
    bars = "".join(z_bar(k) for k in labels)
    return (
        f"<div class='od-panel'>"
        f"<div class='od-panel-title'>Flow Score por Participante</div>"
        f"<div style='display:grid;grid-template-columns:1fr 180px;gap:16px;align-items:start'>"
        f"<div>{bars}</div>"
        f"<div style='text-align:center;padding-top:10px'>"
        f"<div class='od-score-total {total_cls}'>{total:.0f}</div>"
        f"<div class='od-score-label'>Flow Score<br>Total</div>"
        f"<div style='font-size:9px;color:rgba(0,212,232,.25);margin-top:6px'>0 = bearish · 100 = bullish</div>"
        f"</div></div></div>"
    )


def _panel_jarvis(html: str) -> str:
    import re as _re
    # Remove audio / speech synthesis from the Bloomberg export
    html = _re.sub(r"<audio[^>]*>.*?</audio>", "", html, flags=_re.DOTALL | _re.IGNORECASE)
    html = _re.sub(r"speechSynthesis\s*\.\s*speak\s*\([^)]*\)\s*;?", "", html)
    html = _re.sub(r"window\.speechSynthesis[^;]*;?", "", html)
    html = _re.sub(r"new\s+SpeechSynthesisUtterance[^;]*;?", "", html)
    safe = html.replace("&", "&amp;").replace('"', "&quot;")
    return (
        f"<div class='od-panel'>"
        f"<div class='od-panel-title'>Greeks Dashboard (Bloomberg)</div>"
        f"<div class='od-jarvis-wrap'>"
        f'<iframe srcdoc="{safe}" sandbox="allow-scripts allow-same-origin"></iframe>'
        f"</div></div>"
    )
