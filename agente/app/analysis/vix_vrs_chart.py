"""
VIX × VRS Quadrant chart renderer (HTML + SVG, sem matplotlib).
Cores consistentes com desk dark theme.
"""

from __future__ import annotations

from app.analysis.vix_vrs_regime import VixVrsResult


_COLORS = {
    "r1": "#22c55e",  # verde — rally
    "r2": "#3b82f6",  # azul — entry
    "r3": "#ef4444",  # vermelho — bear
    "r4": "#a855f7",  # roxo — raro
    "bg": "#0a1628",
    "border": "#1e293b",
    "label": "#94a3b8",
    "value": "#e2e8f0",
    "trail": "#f59e0b",  # laranja (trail 60d)
    "current": "#ffffff",
}


def render_vix_vrs_panel(result: VixVrsResult) -> str:
    """Painel completo: regime badge + scatter + stats."""
    if not result or result.error or result.regime == 0:
        msg = result.error if result and result.error else "Historico VIX insuficiente"
        return (
            f'<div style="background:{_COLORS["bg"]};border:1px solid {_COLORS["border"]};'
            f'border-radius:8px;padding:16px;margin-bottom:12px">'
            f'<div style="font-size:11px;font-weight:700;color:{_COLORS["label"]};text-transform:uppercase;'
            f'letter-spacing:0.08em;margin-bottom:8px">VIX × VRS Quadrant (Krishnamurthy JOTA 71)</div>'
            f'<div style="color:#64748b;font-size:12px">{msg}</div></div>'
        )

    regime_color = _COLORS[f"r{result.regime}"]

    # Scatter SVG
    svg = _render_scatter_svg(result)

    # Stats row
    stats_html = f'''
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin-top:12px">
  <div style="padding:8px 10px;background:#061422;border-radius:6px;border-left:3px solid {regime_color}">
    <div style="color:{_COLORS["label"]};font-size:9px;text-transform:uppercase;letter-spacing:0.08em">Regime</div>
    <div style="color:{regime_color};font-weight:700;font-size:14px">R{result.regime}</div>
  </div>
  <div style="padding:8px 10px;background:#061422;border-radius:6px">
    <div style="color:{_COLORS["label"]};font-size:9px;text-transform:uppercase;letter-spacing:0.08em">VIX</div>
    <div style="color:{_COLORS["value"]};font-weight:700;font-size:14px">{result.vix:.1f}</div>
    <div style="color:{_COLORS["label"]};font-size:9px">thr {result.vix_threshold:.1f}</div>
  </div>
  <div style="padding:8px 10px;background:#061422;border-radius:6px">
    <div style="color:{_COLORS["label"]};font-size:9px;text-transform:uppercase;letter-spacing:0.08em">VRS</div>
    <div style="color:{_COLORS["value"]};font-weight:700;font-size:14px">{result.vrs:+.2f}</div>
    <div style="color:{_COLORS["label"]};font-size:9px">{result.horizon_label}</div>
  </div>
  <div style="padding:8px 10px;background:#061422;border-radius:6px">
    <div style="color:{_COLORS["label"]};font-size:9px;text-transform:uppercase;letter-spacing:0.08em">Spread</div>
    <div style="color:{_COLORS["value"]};font-weight:700;font-size:14px">{result.spread:+.2f}</div>
    <div style="color:{_COLORS["label"]};font-size:9px">ref {result.ref_mean:+.2f}</div>
  </div>
</div>
'''

    # Action + structures
    structures_html = "".join(
        f'<span style="display:inline-block;background:#061422;border:1px solid {regime_color}60;'
        f'color:{_COLORS["value"]};font-size:10px;padding:3px 8px;border-radius:4px;margin:2px">{s}</span>'
        for s in result.structures
    )

    action_html = f'''
<div style="margin-top:12px;padding:10px 12px;background:#061422;border-radius:6px;border-left:3px solid {regime_color}">
  <div style="color:{regime_color};font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px">
    {result.regime_label}
  </div>
  <div style="color:{_COLORS["value"]};font-size:12px;margin-bottom:6px">{result.interpretation}</div>
  <div style="color:{_COLORS["label"]};font-size:11px;margin-bottom:6px"><strong style="color:{_COLORS["value"]}">Action:</strong> {result.action}</div>
  <div style="margin-top:6px">{structures_html}</div>
</div>
'''

    return f'''
<div style="background:{_COLORS["bg"]};border:1px solid {_COLORS["border"]};border-radius:8px;padding:14px 16px;margin-bottom:14px">
  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px">
    <div style="font-size:13px;font-weight:700;color:{_COLORS["value"]}">
      🎯 VIX × VRS Quadrant — Krishnamurthy JOTA 71
    </div>
    <div style="color:{_COLORS["label"]};font-size:10px">CMT study — entry timing + regime scorecard</div>
  </div>
  {svg}
  {stats_html}
  {action_html}
</div>
'''


def _render_scatter_svg(result: VixVrsResult) -> str:
    """Renderiza scatter VIX x VRS com os 4 quadrantes + trail."""
    W, H = 700, 300
    PAD_L, PAD_R, PAD_T, PAD_B = 50, 20, 20, 35

    # Ranges
    all_vix = [p["vix"] for p in result.trail_60d] + [result.vix, result.vix_threshold]
    all_vrs = [p["vrs"] for p in result.trail_60d] + [result.vrs, 0]
    vix_min, vix_max = min(all_vix) - 2, max(all_vix) + 2
    vrs_min, vrs_max = min(all_vrs) - 1, max(all_vrs) + 1

    def x(v):
        return PAD_L + (v - vix_min) / (vix_max - vix_min) * (W - PAD_L - PAD_R)

    def y(v):
        return PAD_T + (vrs_max - v) / (vrs_max - vrs_min) * (H - PAD_T - PAD_B)

    # Quadrant backgrounds
    x_thr = x(result.vix_threshold)
    y_thr = y(0)
    xL, xR = PAD_L, W - PAD_R
    yT, yB = PAD_T, H - PAD_B

    quad_rects = (
        # R1: VIX<thr, VRS<0 (bottom-left) — verde
        f'<rect x="{xL}" y="{y_thr}" width="{x_thr-xL}" height="{yB-y_thr}" fill="{_COLORS["r1"]}" opacity="0.08"/>'
        # R2: VIX>thr, VRS<0 (bottom-right) — azul
        f'<rect x="{x_thr}" y="{y_thr}" width="{xR-x_thr}" height="{yB-y_thr}" fill="{_COLORS["r2"]}" opacity="0.08"/>'
        # R3: VIX>thr, VRS>0 (top-right) — vermelho
        f'<rect x="{x_thr}" y="{yT}" width="{xR-x_thr}" height="{y_thr-yT}" fill="{_COLORS["r3"]}" opacity="0.08"/>'
        # R4: VIX<thr, VRS>0 (top-left) — roxo
        f'<rect x="{xL}" y="{yT}" width="{x_thr-xL}" height="{y_thr-yT}" fill="{_COLORS["r4"]}" opacity="0.08"/>'
    )

    # Threshold lines
    lines = (
        f'<line x1="{x_thr}" y1="{yT}" x2="{x_thr}" y2="{yB}" stroke="{_COLORS["label"]}" stroke-dasharray="4 3" opacity="0.5"/>'
        f'<line x1="{xL}" y1="{y_thr}" x2="{xR}" y2="{y_thr}" stroke="{_COLORS["label"]}" stroke-dasharray="4 3" opacity="0.5"/>'
    )

    # Quadrant labels
    q_labels = (
        f'<text x="{xL+10}" y="{yT+18}" fill="{_COLORS["r4"]}" font-size="10" font-weight="700">R4 — RARO</text>'
        f'<text x="{xR-80}" y="{yT+18}" fill="{_COLORS["r3"]}" font-size="10" font-weight="700">R3 — BEAR</text>'
        f'<text x="{xL+10}" y="{yB-8}" fill="{_COLORS["r1"]}" font-size="10" font-weight="700">R1 — RALLY</text>'
        f'<text x="{xR-90}" y="{yB-8}" fill="{_COLORS["r2"]}" font-size="10" font-weight="700">R2 — ENTRY</text>'
    )

    # Trail polyline
    trail_points = " ".join(f"{x(p['vix']):.1f},{y(p['vrs']):.1f}" for p in result.trail_60d)
    trail_path = (
        f'<polyline points="{trail_points}" fill="none" stroke="{_COLORS["trail"]}" stroke-width="1.2" opacity="0.6"/>'
        if trail_points else ""
    )
    trail_dots = "".join(
        f'<circle cx="{x(p["vix"]):.1f}" cy="{y(p["vrs"]):.1f}" r="2" fill="{_COLORS["trail"]}" opacity="0.5"/>'
        for p in result.trail_60d
    )

    # Current point (large)
    cur_color = _COLORS[f"r{result.regime}"]
    current_dot = (
        f'<circle cx="{x(result.vix):.1f}" cy="{y(result.vrs):.1f}" r="8" '
        f'fill="{cur_color}" stroke="white" stroke-width="2"/>'
        f'<text x="{x(result.vix):.1f}" y="{y(result.vrs)-14:.1f}" fill="white" '
        f'font-size="10" font-weight="700" text-anchor="middle">HOJE R{result.regime}</text>'
    )

    # Axis labels
    axes = (
        f'<text x="{(xL+xR)/2}" y="{H-8}" fill="{_COLORS["label"]}" font-size="10" text-anchor="middle">VIX</text>'
        f'<text x="15" y="{(yT+yB)/2}" fill="{_COLORS["label"]}" font-size="10" '
        f'transform="rotate(-90 15 {(yT+yB)/2})" text-anchor="middle">VRS</text>'
        # Tick vix_threshold
        f'<text x="{x_thr}" y="{yT-4}" fill="{_COLORS["label"]}" font-size="9" text-anchor="middle">thr {result.vix_threshold:.1f}</text>'
    )

    return f'''
<svg width="100%" height="{H}" viewBox="0 0 {W} {H}" preserveAspectRatio="none"
     style="background:#041020;border:1px solid {_COLORS["border"]};border-radius:6px">
  {quad_rects}
  {lines}
  {q_labels}
  {trail_path}
  {trail_dots}
  {current_dot}
  {axes}
</svg>
'''
