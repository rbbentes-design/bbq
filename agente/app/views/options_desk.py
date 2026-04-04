"""
Options Desk — Aba "Opções ◈" do MacroDesk.

Consome OptionsSnapshot importado do Greeks Dashboard (BQuant ZIP).

Painéis:
  1. Header — métricas-chave em cards (GEX, Gamma Flip, IV/RV, VIX, Squeeze, Tail)
  2. Key Levels — Spot vs Call Wall / Put Wall / Gamma Flip
  3. Greeks — Delta, Vanna, Charm (bn)
  4. Flow Score — z-scores por participante + score total
  5. Skew — 25D skew e IV-RV premium
  6. JARVIS embed — iframe com jarvis.html (se disponível)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.providers.options_store import OptionsSnapshot

# ── CSS ───────────────────────────────────────────────────────────────────────

_OD_CSS = """
<style>
.od-wrap {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace;
  color: #e2e8f0;
  padding: 16px 20px 60px;
  background: #020810;
  min-height: 100%;
}
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
.od-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
.od-grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 12px; }
.od-panel {
  background: rgba(0,6,18,.92);
  border: 1px solid rgba(0,212,232,.15);
  border-radius: 6px;
  padding: 14px 16px;
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
  text-transform: uppercase; margin-bottom: 10px;
  border-bottom: 1px solid rgba(0,212,232,.1); padding-bottom: 5px;
}
/* Level bar */
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
/* z-score bar */
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
/* Greek row */
.od-greek-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 7px 0; border-bottom: 1px solid rgba(0,212,232,.06);
}
.od-greek-name {
  font-size: 11px; color: rgba(0,212,232,.7); letter-spacing: .5px;
}
.od-greek-val {
  font-size: 16px; font-weight: 700; font-family: monospace;
}
/* Empty state */
.od-empty {
  text-align: center; padding: 80px 0;
  color: rgba(0,212,232,.25); font-size: 13px; letter-spacing: 1px;
}
.od-empty b { display: block; font-size: 20px; margin-bottom: 8px; }
/* JARVIS embed */
.od-jarvis-wrap {
  width: 100%; border-radius: 6px; overflow: hidden;
  border: 1px solid rgba(0,212,232,.15); margin-top: 12px;
}
.od-jarvis-wrap iframe {
  width: 100%; height: 680px; border: none; display: block;
}
</style>
"""


# ── Renderer principal ────────────────────────────────────────────────────────

def render_options_tab(
    snapshot: "OptionsSnapshot | None",
    jarvis_html: str | None = None,
) -> str:
    """
    Gera HTML completo para a aba Opções ◈.

    snapshot: OptionsSnapshot importado do Greeks Dashboard
    jarvis_html: conteúdo do jarvis.html (se disponível) para embed
    """
    if snapshot is None:
        return (
            f"{_OD_CSS}"
            "<div class='od-wrap'>"
            "<div class='od-empty'>"
            "<b>◈ Sem dados de opções</b>"
            "Importe um snapshot com:<br>"
            "<code>agente options-import greeks_SPX_YYYYMMDD.zip</code>"
            "</div></div>"
        )

    m = snapshot.metrics

    panels = [
        _header_strip(snapshot),
        f"<div class='od-grid-3'>{_panel_levels(snapshot)}{_panel_greeks(snapshot)}{_panel_skew(snapshot)}</div>",
        _panel_flow_score(snapshot),
    ]

    if jarvis_html:
        panels.append(_panel_jarvis(jarvis_html))

    return (
        f"{_OD_CSS}"
        "<div class='od-wrap'>"
        + "".join(panels)
        + "</div>"
    )


# ── Painel 1 — Header strip de métricas ──────────────────────────────────────

def _header_strip(snap: "OptionsSnapshot") -> str:
    def badge(label: str, val: str, cls: str = "") -> str:
        return (
            f"<div class='od-badge'>"
            f"<span class='od-badge-label'>{label}</span>"
            f"<span class='od-badge-val {cls}'>{val}</span>"
            f"</div>"
        )

    gex = snap.gex_net_bn
    gex_cls = "od-up" if gex > 0 else "od-dn"
    gex_s = f"{'+' if gex >= 0 else ''}{gex:.1f}B"

    iv_rv = snap.iv_rv_pp
    iv_rv_cls = "od-up" if iv_rv > 2 else ("od-dn" if iv_rv < -2 else "")

    sq = snap.squeeze_score
    sq_cls = "od-warn" if sq > 60 else ("od-dn" if sq > 80 else "")

    tail = snap.tail_score
    tail_cls = "od-warn" if tail > 50 else ("od-dn" if tail > 70 else "")

    flip = snap.gamma_flip
    spot = snap.spot

    return (
        f"<div class='od-header-strip'>"
        + badge("SPOT", f"{spot:,.0f}")
        + badge("GAMMA FLIP", f"{flip:,.0f}" if flip else "N/A",
                "od-warn" if abs(spot - flip) < spot * 0.005 else "")
        + badge("GEX NET", gex_s, gex_cls)
        + badge("P/C RATIO", f"{snap.pc_ratio:.2f}×")
        + badge("VIX", f"{snap.vix:.2f}" if snap.vix else "—")
        + badge("IV 30D", f"{snap.iv_30d*100:.2f}%")
        + badge("RV 30D", f"{snap.rv_30d*100:.2f}%")
        + badge("IV − RV", f"{iv_rv:+.1f}pp", iv_rv_cls)
        + badge("SQUEEZE", f"{sq:.0f}/100", sq_cls)
        + badge("TAIL RISK", f"{tail:.0f}/100", tail_cls)
        + f"<span class='od-ts'>{snap.ts} · {snap.ticker}</span>"
        + "</div>"
    )


# ── Painel 2 — Key Levels ────────────────────────────────────────────────────

def _panel_levels(snap: "OptionsSnapshot") -> str:
    spot = snap.spot
    call_wall = snap.call_wall
    put_wall = snap.put_wall
    flip = snap.gamma_flip

    # Compute range for visualization
    lo = min(p for p in [put_wall, flip, spot] if p > 0) * 0.995
    hi = max(p for p in [call_wall, flip, spot] if p > 0) * 1.005
    rng = hi - lo or 1

    def pct(v: float) -> float:
        return max(0, min(100, (v - lo) / rng * 100)) if v else 50

    def mark(v: float, color: str, label: str) -> str:
        if not v:
            return ""
        p = pct(v)
        return (
            f"<div class='od-level-mark' style='left:{p:.1f}%;background:{color}'></div>"
            f"<div class='od-level-label' style='left:{p:.1f}%'>{label}<br>{v:,.0f}</div>"
        )

    bar_html = (
        f"<div class='od-level-bar' style='margin-top:30px;margin-bottom:35px'>"
        + mark(put_wall, "#ef4444", "PUT WALL")
        + mark(flip, "#f59e0b", "G-FLIP")
        + mark(spot, "#00d4e8", "SPOT")
        + mark(call_wall, "#22c55e", "CALL WALL")
        + "</div>"
    )

    dist_flip = ((spot - flip) / flip * 100) if flip else 0
    dist_cw = ((call_wall - spot) / spot * 100) if call_wall else 0
    dist_pw = ((spot - put_wall) / spot * 100) if put_wall else 0

    def row(label: str, val: str, color: str = "") -> str:
        style = f" style='color:{color}'" if color else ""
        fc = color or "#e2e8f0"
        return f"<tr><td style='color:rgba(0,212,232,.5);font-size:10px;padding:3px 8px 3px 0'>{label}</td><td{style} style='font-size:12px;font-weight:600;color:{fc};font-family:monospace'>{val}</td></tr>"

    table = (
        "<table style='width:100%;border-collapse:collapse'>"
        + row("Spot", f"{spot:,.0f}", "#00d4e8")
        + row("Call Wall", f"{call_wall:,.0f}  (+{dist_cw:.1f}%)" if call_wall else "—", "#22c55e")
        + row("Put Wall", f"{put_wall:,.0f}  (−{dist_pw:.1f}%)" if put_wall else "—", "#ef4444")
        + row("Gamma Flip", f"{flip:,.0f}  ({dist_flip:+.1f}%)" if flip else "—", "#f59e0b")
        + "</table>"
    )

    return (
        f"<div class='od-panel'>"
        f"<div class='od-panel-title'>Key Levels</div>"
        + bar_html
        + table
        + "</div>"
    )


# ── Painel 3 — Greeks ─────────────────────────────────────────────────────────

def _panel_greeks(snap: "OptionsSnapshot") -> str:
    def row(name: str, val: float, unit: str = "bn", color: str = "") -> str:
        cls = "od-up" if val > 0 else ("od-dn" if val < 0 else "")
        return (
            f"<div class='od-greek-row'>"
            f"<span class='od-greek-name'>{name}</span>"
            f"<span class='od-greek-val {cls}'>{val:+.2f} <small style='font-size:10px;opacity:.5'>{unit}</small></span>"
            f"</div>"
        )

    frag = snap.fragility
    frag_cls = "od-warn" if frag > 0.4 else ("od-dn" if frag > 0.7 else "od-up")
    frag_pct = frag * 100 if abs(frag) <= 1 else abs(frag)

    return (
        f"<div class='od-panel'>"
        f"<div class='od-panel-title'>Greeks (Market Maker)</div>"
        + row("Delta", snap.delta_bn)
        + row("Vanna", snap.vanna_bn)
        + row("Charm", snap.charm_bn)
        + f"<div class='od-greek-row' style='margin-top:6px'>"
        f"<span class='od-greek-name'>Fragility</span>"
        f"<span class='od-greek-val {frag_cls}'>{frag_pct:.1f} <small style='font-size:10px;opacity:.5'>/100</small></span>"
        + "</div>"
        + "</div>"
    )


# ── Painel 4 — Skew / Vol ─────────────────────────────────────────────────────

def _panel_skew(snap: "OptionsSnapshot") -> str:
    skew = snap.skew_25d
    iv_rv = snap.iv_rv_pp
    iv = snap.iv_30d * 100
    rv = snap.rv_30d * 100

    skew_cls = "od-dn" if skew > 3 else ("od-up" if skew < 1 else "")
    ivrv_cls = "od-up" if iv_rv > 3 else ("od-dn" if iv_rv < -1 else "")

    def metric(label: str, val: str, cls: str = "") -> str:
        return (
            f"<div style='padding:8px 0;border-bottom:1px solid rgba(0,212,232,.06)'>"
            f"<div style='font-size:9px;color:rgba(0,212,232,.45);letter-spacing:1px;text-transform:uppercase;margin-bottom:3px'>{label}</div>"
            f"<div class='od-greek-val {cls}' style='font-size:16px'>{val}</div>"
            f"</div>"
        )

    return (
        f"<div class='od-panel'>"
        f"<div class='od-panel-title'>Volatility / Skew</div>"
        + metric("IV 30D", f"{iv:.2f}%")
        + metric("RV 30D", f"{rv:.2f}%")
        + metric("IV − RV Premium", f"{iv_rv:+.1f}pp", ivrv_cls)
        + metric("25D Put Skew", f"{skew:+.1f}pp", skew_cls)
        + "</div>"
    )


# ── Painel 5 — Flow Score ─────────────────────────────────────────────────────

def _panel_flow_score(snap: "OptionsSnapshot") -> str:
    z = snap.z_scores
    w = snap.weights
    total = snap.flow_score_total

    # Participante labels
    labels = {
        "cta": "CTA",
        "dealer": "Dealer",
        "volctrl": "Vol Ctrl",
        "rp": "Risk Parity",
        "leveraged": "Leveraged",
        "passive_etf": "Passive ETF",
        "buyback": "Buyback",
        "cot": "COT",
    }

    def z_bar(key: str) -> str:
        z_val = z.get(key, 0)
        w_val = w.get(key, 0)
        label = labels.get(key, key)

        # Bar: 0 = center; range [-3, +3] → width 0-100%
        # Left half for negative, right half for positive (center-anchored bar)
        normalized = max(-3, min(3, z_val))  # clamp
        center = 50  # center %
        bar_w = abs(normalized) / 3 * 50  # max 50% of half
        if normalized >= 0:
            bar_left = center
            bar_color = "#22c55e"
        else:
            bar_left = center - bar_w
            bar_color = "#ef4444"

        z_color = "#22c55e" if z_val > 0 else ("#ef4444" if z_val < 0 else "#94a3b8")

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
        f"</div></div>"
        f"</div>"
    )


# ── Painel 6 — JARVIS embed ───────────────────────────────────────────────────

def _panel_jarvis(html: str) -> str:
    """Embeds the JARVIS HTML as a srcdoc iframe."""
    # Escape for srcdoc attribute
    safe = html.replace("&", "&amp;").replace('"', "&quot;")
    return (
        f"<div class='od-panel'>"
        f"<div class='od-panel-title'>JARVIS Greeks Dashboard (Bloomberg)</div>"
        f"<div class='od-jarvis-wrap'>"
        f'<iframe srcdoc="{safe}" sandbox="allow-scripts allow-same-origin"></iframe>'
        f"</div></div>"
    )
