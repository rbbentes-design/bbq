"""
Desk Radar — Renderer HTML da aba de inteligência de mercado

Consome DeskIntelligenceResult e dados auxiliares.
Gera 7 painéis auto-contidos em HTML/CSS/JS puro.

Painéis:
  1. Regime Atual (header com pesos + drivers)
  2. Narrativa dominante + Clusters temáticos
  3. RRG Turbinado (canvas com trilhas, IV→cor, skew→borda)
  4. Mapa de Contágio (hubs MST + vizinhos)
  5. Ranking Geral (tabela regime-adjusted)
  6. Hidden Opportunities + Fragility (side-by-side)
  7. Explicações textuais (accordion por ativo)
"""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.analysis.desk_intelligence import DeskIntelligenceResult
    from app.analysis.alpha_signals import AssetSignal
    from app.analysis.relative_strength import RRGResult


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _h(v: Any) -> str:
    """Escapa HTML básico."""
    return str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _score_bar(score: float, lo: float = -1.0, hi: float = 1.0) -> str:
    """Mini barra colorida inline."""
    pct = (score - lo) / (hi - lo) * 100 if hi != lo else 50
    pct = max(0.0, min(100.0, pct))
    color = "#22c55e" if score >= 0.15 else ("#ef4444" if score <= -0.15 else "#f59e0b")
    return (
        f'<div style="display:flex;align-items:center;gap:6px">'
        f'<div style="flex:1;height:4px;background:#1e293b;border-radius:2px">'
        f'<div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:2px"></div>'
        f'</div>'
        f'<span style="font-size:10px;color:{color};min-width:36px;text-align:right">'
        f'{score:+.2f}</span></div>'
    )


def _quad_color(q: str) -> str:
    return {"leading": "#22c55e", "improving": "#3b82f6",
            "weakening": "#f97316", "lagging": "#ef4444"}.get(q, "#6b7280")


def _quad_icon(q: str) -> str:
    return {"leading": "▲", "improving": "↑", "weakening": "↓", "lagging": "▼"}.get(q, "·")


def _conv_badge(c: str) -> str:
    styles = {
        "high":   "background:#14532d;color:#86efac",
        "medium": "background:#78350f;color:#fcd34d",
        "low":    "background:#1e293b;color:#6b7280",
    }
    s = styles.get(c, styles["low"])
    return f'<span style="{s};padding:1px 6px;border-radius:3px;font-size:10px;font-weight:700">{c.upper()}</span>'


# ── Painel 1: Regime ─────────────────────────────────────────────────────────────

def _panel_regime(result: "DeskIntelligenceResult") -> str:
    regime_label = result.market_regime_label or result.market_regime.replace("_", " ").title()
    conf         = result.regime_confidence
    weights      = result.regime_weights or {}
    drivers      = result.regime_drivers or []

    conf_pct = f"{conf:.0%}"
    conf_color = "#22c55e" if conf >= 0.65 else ("#f59e0b" if conf >= 0.45 else "#6b7280")

    # Badge principal
    _REGIME_BG = {
        "risk_on_momentum":   ("#065f46", "#4ade80"),
        "risk_off_defensive": ("#7f1d1d", "#f87171"),
        "vol_squeeze":        ("#1e3a5f", "#60a5fa"),
        "narrative_driven":   ("#4c1d95", "#c084fc"),
        "mechanical_passive": ("#0c4a6e", "#38bdf8"),
        "dispersed_rotation": ("#78350f", "#fcd34d"),
        "stress":             ("#450a0a", "#ef4444"),
    }
    bg, fg = _REGIME_BG.get(result.market_regime, ("#1f2937", "#9ca3af"))

    badge = (
        f'<div style="display:flex;align-items:center;gap:20px;background:{bg};'
        f'border:1px solid {fg}40;border-radius:12px;padding:18px 24px">'
        f'<div style="font-size:28px;font-weight:900;color:{fg};letter-spacing:-0.5px">'
        f'{_h(regime_label)}</div>'
        f'<div style="font-size:14px;color:{conf_color};font-weight:700;background:{conf_color}22;'
        f'padding:4px 12px;border-radius:20px;border:1px solid {conf_color}44">'
        f'Conf. {conf_pct}</div></div>'
    )

    # Barra de pesos segmentada
    _WEIGHT_COLORS = {
        "macro":     "#60a5fa",
        "technical": "#4ade80",
        "vol":       "#c084fc",
        "flow":      "#fbbf24",
        "narrative": "#f472b6",
    }
    _WEIGHT_LABELS = {
        "macro": "Macro", "technical": "Técnico",
        "vol": "Vol", "flow": "Fluxo", "narrative": "Narrativa",
    }
    weight_segs = ""
    weight_legend = ""
    for k in ("macro", "technical", "vol", "flow", "narrative"):
        v  = weights.get(k, 0.0)
        pct = v * 100
        col = _WEIGHT_COLORS[k]
        weight_segs  += f'<div style="flex:{v:.2f};background:{col};height:12px" title="{_WEIGHT_LABELS[k]}: {pct:.0f}%"></div>'
        weight_legend += (
            f'<div style="display:flex;align-items:center;gap:4px">'
            f'<div style="width:8px;height:8px;background:{col};border-radius:2px"></div>'
            f'<span style="font-size:10px;color:#9ca3af">{_WEIGHT_LABELS[k]} {pct:.0f}%</span>'
            f'</div>'
        )

    weights_html = (
        f'<div style="margin-top:10px">'
        f'<div style="font-size:10px;color:#6b7280;margin-bottom:4px">PESOS DO REGIME</div>'
        f'<div style="display:flex;border-radius:4px;overflow:hidden;height:12px">{weight_segs}</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:6px">{weight_legend}</div>'
        f'</div>'
    )

    # Chips de drivers
    chips = "".join(
        f'<span style="background:#1e293b;color:#94a3b8;border:1px solid #374151;'
        f'padding:2px 8px;border-radius:12px;font-size:10px">{_h(d)}</span>'
        for d in drivers[:6]
    )
    drivers_html = f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:10px">{chips}</div>' if chips else ""

    return (
        f'<div class="dr-panel">'
        f'{badge}{weights_html}{drivers_html}'
        f'</div>'
    )


# ── Painel 2: Narrativa + Clusters ───────────────────────────────────────────────

def _panel_narrative(
    result: "DeskIntelligenceResult",
    rrg_result: "RRGResult | None",
) -> str:
    dominant = result.dominant_narrative
    clusters = result.narrative_clusters or {}

    if not dominant and not clusters:
        return (
            '<div class="dr-panel">'
            '<div class="dr-panel-title">Narrativa Dominante</div>'
            '<div style="color:#6b7280;font-size:12px">Sem dados narrativos disponíveis.</div>'
            '</div>'
        )

    # Header do tema dominante
    dom_html = ""
    if dominant:
        dom_html = (
            f'<div style="background:#1e1b4b;border:1px solid #818cf880;border-radius:8px;'
            f'padding:12px 16px;margin-bottom:12px">'
            f'<div style="font-size:10px;color:#818cf8;font-weight:700;margin-bottom:4px">'
            f'NARRATIVA DOMINANTE</div>'
            f'<div style="font-size:18px;font-weight:800;color:#e0e7ff">{_h(dominant)}</div>'
            f'</div>'
        )

    rrg_sigs = {}
    if rrg_result and hasattr(rrg_result, "signals"):
        rrg_sigs = rrg_result.signals or {}

    # Grid de clusters
    cluster_cards = ""
    for theme, tickers in list(clusters.items())[:8]:
        ticker_chips = ""
        for t in tickers[:8]:
            rs  = rrg_sigs.get(t)
            q   = getattr(rs, "quadrant", "") if rs else ""
            col = _quad_color(q)
            ico = _quad_icon(q)
            ticker_chips += (
                f'<span style="background:{col}20;color:{col};border:1px solid {col}40;'
                f'padding:2px 7px;border-radius:10px;font-size:10px;font-weight:600">'
                f'{ico} {_h(t)}</span>'
            )
        cluster_cards += (
            f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:10px 12px">'
            f'<div style="font-size:11px;color:#94a3b8;font-weight:700;margin-bottom:6px">{_h(theme)}</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:4px">{ticker_chips}</div>'
            f'</div>'
        )

    clusters_html = (
        f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px;margin-top:8px">'
        f'{cluster_cards}</div>'
    ) if cluster_cards else ""

    # Quando não tem clusters — mostrar leaders/laggards dos RRG signals
    extra_html = ""
    if not cluster_cards and rrg_sigs:
        leading  = [t for t, s in rrg_sigs.items() if getattr(s, "quadrant", "") == "leading"][:8]
        improving = [t for t, s in rrg_sigs.items() if getattr(s, "quadrant", "") == "improving"][:8]
        weakening = [t for t, s in rrg_sigs.items() if getattr(s, "quadrant", "") == "weakening"][:6]
        lagging  = [t for t, s in rrg_sigs.items() if getattr(s, "quadrant", "") == "lagging"][:6]

        def _chip_row(tickers, color):
            return "".join(
                f'<span style="background:{color}18;color:{color};border:1px solid {color}35;'
                f'padding:3px 8px;border-radius:8px;font-size:11px;font-weight:600">{_h(t)}</span>'
                for t in tickers
            )

        rows = []
        if leading:
            rows.append(f'<div style="margin-bottom:10px"><div style="font-size:10px;color:#22c55e;font-weight:700;margin-bottom:5px">▲ LEADING</div><div style="display:flex;flex-wrap:wrap;gap:4px">{_chip_row(leading,"#22c55e")}</div></div>')
        if improving:
            rows.append(f'<div style="margin-bottom:10px"><div style="font-size:10px;color:#3b82f6;font-weight:700;margin-bottom:5px">↑ IMPROVING</div><div style="display:flex;flex-wrap:wrap;gap:4px">{_chip_row(improving,"#3b82f6")}</div></div>')
        if weakening:
            rows.append(f'<div style="margin-bottom:10px"><div style="font-size:10px;color:#f97316;font-weight:700;margin-bottom:5px">↓ WEAKENING</div><div style="display:flex;flex-wrap:wrap;gap:4px">{_chip_row(weakening,"#f97316")}</div></div>')
        if lagging:
            rows.append(f'<div style="margin-bottom:6px"><div style="font-size:10px;color:#ef4444;font-weight:700;margin-bottom:5px">▼ LAGGING</div><div style="display:flex;flex-wrap:wrap;gap:4px">{_chip_row(lagging,"#ef4444")}</div></div>')

        if rows:
            extra_html = f'<div style="margin-top:12px">{"".join(rows)}</div>'

    return (
        f'<div class="dr-panel">'
        f'<div class="dr-panel-title">Narrativa &amp; Clusters</div>'
        f'{dom_html}{clusters_html}{extra_html}'
        f'</div>'
    )


# ── Painel 3: RRG Turbinado ──────────────────────────────────────────────────────

def _panel_rrg_turbinado(
    result: "DeskIntelligenceResult",
    rrg_result: "RRGResult | None",
) -> str:
    rrg_meta = result.rrg_meta or {}

    # Serializa dados para JS
    rrg_js_data: list[dict] = []
    for ticker, meta in rrg_meta.items():
        if meta.get("rs_ratio") is None or meta.get("rs_momentum") is None:
            continue
        rrg_js_data.append({
            "ticker":       ticker,
            "rs_ratio":     meta.get("rs_ratio"),
            "rs_momentum":  meta.get("rs_momentum"),
            "rs_percentile": meta.get("rs_percentile"),
            "quadrant":     meta.get("quadrant", ""),
            "bubble_size":  meta.get("bubble_size", 14),
            "bubble_color": meta.get("bubble_color", "#6b7280"),
            "border_color": meta.get("border_color", "#6b7280"),
            "border_width": meta.get("border_width", 1),
            "opacity":      meta.get("opacity", 0.7),
            "label_extra":  meta.get("label_extra", ""),
            "tail_rs_ratio":    meta.get("tail_rs_ratio", []),
            "tail_rs_momentum": meta.get("tail_rs_momentum", []),
            "conviction":   meta.get("conviction", "low"),
            "composite":    meta.get("composite", 0),
            "iv_percentile": meta.get("iv_percentile"),
            "skew":         meta.get("skew"),
        })

    rrg_json = json.dumps(rrg_js_data, ensure_ascii=False)

    rrg_js = f"""
(function() {{
const RRG_DATA = {rrg_json};
const canvas  = document.getElementById('dr-rrg-canvas');
if (!canvas || !RRG_DATA.length) return;
const ctx = canvas.getContext('2d');

const W = canvas.width, H = canvas.height;
const PAD = 58;
const plotW = W - PAD * 2, plotH = H - PAD * 2;

// Range dinâmico
let minX = 94, maxX = 106, minY = 94, maxY = 106;
RRG_DATA.forEach(d => {{
  minX = Math.min(minX, d.rs_ratio   - 2); maxX = Math.max(maxX, d.rs_ratio   + 2);
  minY = Math.min(minY, d.rs_momentum - 2); maxY = Math.max(maxY, d.rs_momentum + 2);
}});

function toX(v) {{ return PAD + (v - minX) / (maxX - minX) * plotW; }}
function toY(v) {{ return H - PAD - (v - minY) / (maxY - minY) * plotH; }}

// Zoom/pan state
let scale = 1, ox = 0, oy = 0;
let isPan = false, panStart = {{x:0, y:0}};

// Draw arrowhead at end of trail
function drawArrow(x1, y1, x2, y2, color, alpha) {{
  const angle = Math.atan2(y2-y1, x2-x1);
  const len = 7;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x2 - len*Math.cos(angle-0.35), y2 - len*Math.sin(angle-0.35));
  ctx.lineTo(x2, y2);
  ctx.lineTo(x2 - len*Math.cos(angle+0.35), y2 - len*Math.sin(angle+0.35));
  ctx.stroke();
  ctx.restore();
}}

function redraw() {{
  ctx.clearRect(0, 0, W, H);

  // ── Axes labels (outside transform — fixed) ───────────────────────────────
  ctx.fillStyle = '#475569'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('RS-Ratio →', W/2, H-6);
  ctx.save(); ctx.translate(13, H/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('RS-Momentum →', 0, 0); ctx.restore();

  // ── Apply zoom transform ──────────────────────────────────────────────────
  ctx.save();
  ctx.translate(ox, oy);
  ctx.scale(scale, scale);

  const px100 = toX(100), py100 = toY(100);

  // Quadrant backgrounds (richer colors)
  const quads = [
    {{ x0:px100, y0:PAD,    x1:W-PAD, y1:py100, color:'#14532d65', label:'LEADING',   lx:W-PAD-6,  ly:PAD+16, align:'right' }},
    {{ x0:PAD,   y0:PAD,    x1:px100, y1:py100, color:'#1e3a8a55', label:'IMPROVING', lx:PAD+6,    ly:PAD+16, align:'left'  }},
    {{ x0:PAD,   y0:py100,  x1:px100, y1:H-PAD, color:'#7f1d1d65', label:'LAGGING',   lx:PAD+6,    ly:H-PAD-8, align:'left' }},
    {{ x0:px100, y0:py100,  x1:W-PAD, y1:H-PAD, color:'#78350f55', label:'WEAKENING', lx:W-PAD-6,  ly:H-PAD-8, align:'right' }},
  ];
  quads.forEach(q => {{
    ctx.fillStyle = q.color;
    ctx.fillRect(q.x0, q.y0, q.x1-q.x0, q.y1-q.y0);
  }});

  // Pivot lines — espessura constante na tela
  ctx.strokeStyle = '#2d4a6b'; ctx.lineWidth = 1.5 / scale;
  ctx.beginPath(); ctx.moveTo(PAD, py100); ctx.lineTo(W-PAD, py100); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(px100, PAD); ctx.lineTo(px100, H-PAD); ctx.stroke();

  // Pivot labels — tamanho constante na tela
  ctx.fillStyle = '#4a6380'; ctx.font = (9/scale)+'px sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('100', px100, H-PAD+14/scale);
  ctx.textAlign = 'right';
  ctx.fillText('100', PAD-6/scale, py100+3/scale);

  // Quadrant labels — tamanho constante
  quads.forEach(q => {{
    ctx.fillStyle = '#ffffff35';
    ctx.font = 'bold '+(10/scale)+'px monospace';
    ctx.textAlign = q.align;
    ctx.fillText(q.label, q.lx, q.ly);
  }});

  // Trails + bubbles
  RRG_DATA.forEach(d => {{
    const x = toX(d.rs_ratio), y = toY(d.rs_momentum);
    const tailR = d.tail_rs_ratio || [], tailM = d.tail_rs_momentum || [];

    // Tamanho da bolha constante na tela (dividido por scale)
    const r_px = Math.max(8, Math.min(30, d.bubble_size / 2));  // pixels na tela
    const r = r_px / scale;                                       // unidades lógicas

    // Trail
    if (tailR.length >= 2) {{
      const isDashed = (d.quadrant === 'improving' || d.quadrant === 'weakening');
      ctx.save();
      ctx.globalAlpha = 0.55;
      ctx.strokeStyle = d.bubble_color;
      ctx.lineWidth = (isDashed ? 1.5 : 2) / scale;
      if (isDashed) ctx.setLineDash([4/scale, 3/scale]);
      ctx.beginPath();
      for (let i=0; i<tailR.length; i++) {{
        const tx = toX(tailR[i]), ty = toY(tailM[i] != null ? tailM[i] : d.rs_momentum);
        i===0 ? ctx.moveTo(tx,ty) : ctx.lineTo(tx,ty);
      }}
      ctx.lineTo(x,y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
      // Arrow at the tip (tamanho constante)
      const prevX = toX(tailR[tailR.length-1]);
      const prevY = toY(tailM[tailM.length-1] != null ? tailM[tailM.length-1] : d.rs_momentum);
      const angle = Math.atan2(y-prevY, x-prevX);
      const aLen = 7/scale;
      ctx.save(); ctx.globalAlpha=0.85; ctx.strokeStyle=d.bubble_color; ctx.lineWidth=1.5/scale;
      ctx.beginPath();
      ctx.moveTo(x-aLen*Math.cos(angle-0.35), y-aLen*Math.sin(angle-0.35));
      ctx.lineTo(x, y);
      ctx.lineTo(x-aLen*Math.cos(angle+0.35), y-aLen*Math.sin(angle+0.35));
      ctx.stroke(); ctx.restore();
    }}

    // Outer glow ring for high conviction
    if (d.conviction === 'high') {{
      ctx.save();
      ctx.globalAlpha = 0.18;
      ctx.strokeStyle = d.bubble_color;
      ctx.lineWidth = 4 / scale;
      ctx.beginPath(); ctx.arc(x, y, r * 1.55, 0, Math.PI*2); ctx.stroke();
      ctx.restore();
    }}

    // Bubble fill gradient
    ctx.save();
    ctx.globalAlpha = d.opacity != null ? Math.max(0.55, d.opacity) : 0.85;
    const grad = ctx.createRadialGradient(x - r*0.35, y - r*0.35, r*0.05, x, y, r);
    grad.addColorStop(0, d.bubble_color + 'ff');
    grad.addColorStop(0.6, d.bubble_color + 'cc');
    grad.addColorStop(1, d.bubble_color + '44');
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI*2); ctx.fill();

    // Border (skew direction)
    ctx.strokeStyle = d.border_color;
    ctx.lineWidth = Math.max(1.5, d.border_width || 2) / scale;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI*2); ctx.stroke();
    ctx.restore();

    // Label — tamanho constante na tela
    const fontSize = Math.round((r_px > 15 ? 10 : 9) / scale);
    ctx.fillStyle = '#f8fafc';
    ctx.font = `bold ${{fontSize}}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.shadowColor = '#000'; ctx.shadowBlur = 3/scale;
    ctx.fillText(d.ticker, x, y + r + 12/scale);
    ctx.shadowBlur = 0;
    if (d.label_extra) {{
      ctx.fillStyle = '#94a3b8'; ctx.font = (8/scale)+'px sans-serif';
      ctx.fillText(d.label_extra, x, y + r + 22/scale);
    }}
  }});

  ctx.restore(); // end zoom transform
}}

redraw();

// ── Zoom (wheel) ─────────────────────────────────────────────────────────────
canvas.addEventListener('wheel', function(e) {{
  e.preventDefault();
  const factor = e.deltaY > 0 ? 0.85 : 1.18;
  const rect = canvas.getBoundingClientRect();
  const cx = (e.clientX - rect.left) * (W / rect.width);
  const cy = (e.clientY - rect.top)  * (H / rect.height);
  const newScale = Math.min(7, Math.max(0.25, scale * factor));
  ox = cx - (cx - ox) * (newScale / scale);
  oy = cy - (cy - oy) * (newScale / scale);
  scale = newScale;
  redraw();
}}, {{passive: false}});

// ── Pan (drag) ────────────────────────────────────────────────────────────────
canvas.addEventListener('mousedown', function(e) {{
  if (e.button !== 0) return;
  isPan = true;
  panStart = {{x: e.clientX - ox, y: e.clientY - oy}};
  canvas.style.cursor = 'grabbing';
}});
window.addEventListener('mousemove', function(e) {{
  if (isPan) {{
    ox = e.clientX - panStart.x;
    oy = e.clientY - panStart.y;
    redraw();
  }}
}});
window.addEventListener('mouseup', function() {{
  if (isPan) {{ isPan = false; canvas.style.cursor = 'crosshair'; }}
}});

// ── Tooltip ───────────────────────────────────────────────────────────────────
const tooltip = document.getElementById('dr-rrg-tooltip');
canvas.addEventListener('mousemove', function(e) {{
  if (isPan) return;
  const rect = canvas.getBoundingClientRect();
  const cx = (e.clientX - rect.left) * (W / rect.width);
  const cy = (e.clientY - rect.top)  * (H / rect.height);
  const mx = (cx - ox) / scale;
  const my = (cy - oy) / scale;
  let found = null, minDist = 35 / scale;
  RRG_DATA.forEach(d => {{
    if (d.rs_ratio == null) return;
    const r = Math.max(8, Math.min(30, d.bubble_size / 2)) / scale;
    const dx = toX(d.rs_ratio) - mx, dy = toY(d.rs_momentum) - my;
    const dist = Math.sqrt(dx*dx + dy*dy);
    if (dist < minDist + r * 0.5) {{ minDist = dist; found = d; }}
  }});
  if (found) {{
    const iv  = found.iv_percentile != null ? `IV: ${{(found.iv_percentile*100).toFixed(0)}}%ile` : '';
    const sk  = found.skew != null ? `Skew: ${{found.skew > 0 ? '+' : ''}}${{found.skew.toFixed(3)}}` : '';
    const rs  = `RS-Ratio: ${{found.rs_ratio?.toFixed(2)}} | Mom: ${{found.rs_momentum?.toFixed(2)}}`;
    const pct = found.rs_percentile != null ? `RS%ile: ${{found.rs_percentile.toFixed(0)}}°` : '';
    const quad_colors = {{leading:'#22c55e',improving:'#60a5fa',weakening:'#f97316',lagging:'#ef4444'}};
    const qc = quad_colors[found.quadrant] || '#94a3b8';
    tooltip.innerHTML = `<strong style="color:#f8fafc;font-size:13px">${{found.ticker}}</strong>
      <span style="color:${{qc}};font-size:10px;margin-left:6px">${{found.quadrant?.toUpperCase()}}</span><br>
      <span style="color:#94a3b8;font-size:10px">${{rs}}</span><br>
      <span style="color:#64748b;font-size:10px">${{pct}} &nbsp; ${{iv}} &nbsp; ${{sk}}</span><br>
      <span style="color:#64748b;font-size:10px">Conv: <span style="color:#e5e7eb">${{found.conviction}}</span> &nbsp;
      Score: <span style="color:${{found.composite >= 0 ? '#4ade80':'#f87171'}}">${{found.composite > 0?'+':''}}${{found.composite?.toFixed(2)}}</span></span>`;
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top  = (e.clientY - 36) + 'px';
  }} else {{
    tooltip.style.display = 'none';
  }}
}});
canvas.addEventListener('mouseleave', function() {{
  if (!isPan) tooltip.style.display = 'none';
}});

// ── Reset zoom button ─────────────────────────────────────────────────────────
const btnReset = document.getElementById('dr-rrg-reset');
if (btnReset) {{ btnReset.addEventListener('click', function() {{
  scale = 1; ox = 0; oy = 0; redraw();
}}); }}
}})();
"""

    legend = (
        '<div style="display:flex;flex-wrap:wrap;gap:14px;margin-top:10px;font-size:10px;color:#64748b;align-items:center">'
        '<div style="display:flex;align-items:center;gap:4px">'
        '<span style="display:inline-block;width:11px;height:11px;border-radius:50%;'
        'background:linear-gradient(135deg,#22c55e,#16a34a)"></span>IV barata</div>'
        '<div style="display:flex;align-items:center;gap:4px">'
        '<span style="display:inline-block;width:11px;height:11px;border-radius:50%;'
        'background:linear-gradient(135deg,#ef4444,#b91c1c)"></span>IV cara</div>'
        '<div style="display:flex;align-items:center;gap:4px">'
        '<span style="display:inline-block;width:11px;height:11px;border-radius:50%;'
        'background:transparent;border:2.5px solid #22c55e"></span>Call skew</div>'
        '<div style="display:flex;align-items:center;gap:4px">'
        '<span style="display:inline-block;width:11px;height:11px;border-radius:50%;'
        'background:transparent;border:2.5px solid #ef4444"></span>Put skew</div>'
        '<div style="color:#4b5563">Tamanho = convicção &nbsp;·&nbsp; Trilha = 5 períodos &nbsp;·&nbsp; ↑ = direção</div>'
        '<div style="margin-left:auto;font-size:9px;color:#374151">scroll=zoom · drag=pan</div>'
        '</div>'
    )

    return (
        f'<div class="dr-panel">'
        f'<div class="dr-panel-title" style="display:flex;justify-content:space-between;align-items:center">'
        f'<span>RRG Turbinado <span style="font-size:11px;color:#6b7280">'
        f'— IV rank · skew · conviction · trailing path</span></span>'
        f'<button id="dr-rrg-reset" style="font-size:10px;padding:2px 8px;border:1px solid #334155;'
        f'background:transparent;color:#64748b;border-radius:4px;cursor:pointer">⊙ Reset</button>'
        f'</div>'
        f'<canvas id="dr-rrg-canvas" width="1100" height="580" '
        f'style="width:100%;height:auto;min-height:520px;background:#060a12;border-radius:8px;'
        f'border:1px solid #1e293b;display:block;cursor:crosshair"></canvas>'
        f'{legend}'
        f'</div>'
        f'<script>{rrg_js}</script>'
    )


# ── Painel 4: Mapa de Contágio ───────────────────────────────────────────────────

def _panel_contagion(
    result: "DeskIntelligenceResult",
    signals: dict[str, "AssetSignal"],
    network_result: dict | None,
) -> str:
    cont_scores  = result.contagion_scores or {}
    top_contagion = sorted(cont_scores, key=lambda t: cont_scores[t], reverse=True)[:8]

    if not top_contagion:
        return (
            '<div class="dr-panel">'
            '<div class="dr-panel-title">Mapa de Contágio</div>'
            '<div style="color:#6b7280;font-size:12px">Sem dados de rede disponíveis.</div>'
            '</div>'
        )

    # Mapa de adjacência do MST
    mst_adj: dict[str, list[str]] = {}
    if network_result:
        for e in (network_result.get("mst", {}).get("edges", []) or []):
            if isinstance(e, dict):
                a, b = e.get("from", ""), e.get("to", "")
                rho  = e.get("correlation", 0.0)
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                a, b = e[0], e[1]
                rho  = e[2] if len(e) > 2 else 0.0
            else:
                continue
            if a and b:
                mst_adj.setdefault(a, []).append((b, rho))
                mst_adj.setdefault(b, []).append((a, rho))

    rows = ""
    for ticker in top_contagion:
        score = cont_scores[ticker]
        sig   = signals.get(ticker)
        comp  = (sig.composite or 0.0) if sig else 0.0
        col   = "#22c55e" if comp >= 0.0 else "#ef4444"

        # Vizinhos de 1° grau no MST
        neighbors = mst_adj.get(ticker, [])[:4]
        nb_html = ""
        for nb, rho in neighbors:
            rho_col = "#22c55e" if rho >= 0 else "#ef4444"
            nb_html += (
                f'<span style="background:{rho_col}20;color:{rho_col};border:1px solid {rho_col}30;'
                f'padding:1px 6px;border-radius:8px;font-size:9px">'
                f'{_h(nb)} {rho:+.2f}</span>'
            )

        bar_pct = score * 100
        rows += (
            f'<div style="border-bottom:1px solid #1e293b;padding:10px 0">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
            f'<span style="font-size:13px;font-weight:700;color:#e5e7eb">{_h(ticker)}</span>'
            f'<span style="font-size:11px;color:{col};font-weight:700">{score:.2f}</span>'
            f'</div>'
            f'<div style="width:100%;height:6px;background:#1e293b;border-radius:3px;margin-bottom:6px">'
            f'<div style="width:{bar_pct:.1f}%;height:100%;background:{col};border-radius:3px"></div></div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:4px">{nb_html}</div>'
            f'</div>'
        )

    return (
        f'<div class="dr-panel">'
        f'<div class="dr-panel-title">Mapa de Contágio — Hubs MST</div>'
        f'{rows}'
        f'</div>'
    )


# ── Painel 5: Ranking Geral ──────────────────────────────────────────────────────

def _panel_narrative_engine(result: "DeskIntelligenceResult") -> str:
    """Painel 5a: Motor de Narrativa — estados, assimetria, crowdedness, exaustão."""
    narr_states  = result.narrative_states   or {}
    asym_scores  = result.asymmetry_scores   or {}
    crowd_scores = result.crowdedness_scores or {}
    exhaust_sc   = result.exhaustion_scores  or {}
    convictions  = result.convictions        or {}
    rationales   = result.narrative_rationales or {}
    overloads    = result.consensus_overloads  or {}
    chart_reg    = result.chart_regimes        or {}
    wsb_top      = result.wsb_top_mentions     or []
    top_squeeze  = result.top_squeeze          or []

    if not narr_states:
        return ""

    # Cores por estado
    STATE_COLOR = {
        "pre_narrative": "#818cf8",
        "emerging":      "#22c55e",
        "mature":        "#f59e0b",
        "exhausted":     "#ef4444",
    }
    STATE_LABEL = {
        "pre_narrative": "Pré-Narrativa",
        "emerging":      "Emergente",
        "mature":        "Matura",
        "exhausted":     "Exaurida",
    }
    CONV_LABEL = {
        "compra_agressiva": ("▲▲", "#22c55e"),
        "compra_tatica":    ("▲",  "#84cc16"),
        "reduzir":          ("↓",  "#f59e0b"),
        "realizar":         ("↓↓", "#f97316"),
        "evitar":           ("—",  "#6b7280"),
        "short_tatico":     ("▼",  "#ef4444"),
    }

    def _mini_bar_100(val: float, color: str) -> str:
        pct = max(0, min(100, val))
        return (f'<div style="flex:1;height:5px;background:#1e293b;border-radius:2px">'
                f'<div style="width:{pct:.0f}%;height:100%;background:{color};border-radius:2px"></div>'
                f'</div><span style="font-size:9px;color:{color};min-width:26px;text-align:right">{val:.0f}</span>')

    # Resumo rápido: top assimetria, top short
    top_asym  = result.top_asymmetry    or []
    top_crowd = result.top_crowded      or []
    top_exh   = result.top_exhausted_narr or []
    emerging  = result.emerging_themes  or []
    shorts    = result.short_candidates or []

    # Cards de resumo
    def _summary_chips(label: str, tickers: list, color: str) -> str:
        if not tickers:
            return ""
        chips = "".join(
            f'<span style="background:{color}22;color:{color};border:1px solid {color}44;'
            f'border-radius:3px;padding:1px 6px;font-size:10px;font-weight:700">{_h(t)}</span>'
            for t in tickers[:8]
        )
        return (f'<div style="margin-bottom:8px">'
                f'<span style="font-size:10px;color:#6b7280;margin-right:6px">{label}</span>'
                f'<span style="display:inline-flex;flex-wrap:wrap;gap:4px">{chips}</span></div>')

    summary = (
        f'<div style="background:#060a12;border-radius:6px;padding:10px 12px;margin-bottom:10px">'
        + _summary_chips("Alta Assimetria", top_asym[:6], "#22c55e")
        + _summary_chips("Short Tático", shorts[:6], "#ef4444")
        + _summary_chips("Pré/Emergente", emerging[:6], "#818cf8")
        + _summary_chips("Mais Crowded", top_crowd[:5], "#f97316")
        + _summary_chips("Exauridos", top_exh[:5], "#ef4444")
        + (_summary_chips("WSB Top", wsb_top[:6], "#38bdf8") if wsb_top else "")
        + (_summary_chips("Squeeze", top_squeeze[:5], "#c084fc") if top_squeeze else "")
        + "</div>"
    )

    # Tabela principal — ordena por assimetria
    sorted_tickers = sorted(narr_states.keys(), key=lambda t: asym_scores.get(t, 0), reverse=True)

    cols = "32px 80px 90px 1fr 1fr 1fr 48px 80px"
    header = (
        f'<div style="display:grid;grid-template-columns:{cols};gap:4px;padding:5px 8px;'
        f'background:#0f172a;border-radius:6px 6px 0 0;font-size:9px;font-weight:700;color:#6b7280">'
        f'<div>#</div><div>Ativo</div><div>Estado</div>'
        f'<div>Assimetria</div><div>Crowded</div><div>Exaustão</div>'
        f'<div>Conv.</div><div>Regime Gráfico</div>'
        f'</div>'
    )

    rows = ""
    for i, ticker in enumerate(sorted_tickers, 1):
        state = narr_states.get(ticker, "pre_narrative")
        asym  = asym_scores.get(ticker, 50.0)
        crowd = crowd_scores.get(ticker, 0.0)
        exh   = exhaust_sc.get(ticker, 0.0)
        conv  = convictions.get(ticker, "evitar")
        chart = chart_reg.get(ticker, "")
        overl = overloads.get(ticker, False)

        sc    = STATE_COLOR.get(state, "#6b7280")
        sl    = STATE_LABEL.get(state, state)
        ci, cc = CONV_LABEL.get(conv, ("—", "#6b7280"))
        row_bg = "#0a1628" if i % 2 == 0 else "#060a12"

        # Alerta de consenso saturado
        alert = ""
        if overl:
            alert = ' <span style="color:#ef4444;font-size:9px" title="Consenso saturado">⚠</span>'

        # Chart regime badge curto
        chart_short = chart.replace("tendência_", "").replace("_", " ")[:18]

        rows += (
            f'<div style="display:grid;grid-template-columns:{cols};gap:4px;padding:5px 8px;'
            f'background:{row_bg};align-items:center;font-size:10px">'
            f'<div style="color:#6b7280;font-size:9px">{i}</div>'
            f'<div style="font-weight:700;color:#e5e7eb">{_h(ticker)}{alert}</div>'
            f'<div><span style="background:{sc}22;color:{sc};border:1px solid {sc}44;'
            f'border-radius:3px;padding:1px 5px;font-size:9px">{sl}</span></div>'
            f'<div style="display:flex;align-items:center;gap:4px">{_mini_bar_100(asym, "#22c55e")}</div>'
            f'<div style="display:flex;align-items:center;gap:4px">{_mini_bar_100(crowd, "#f97316")}</div>'
            f'<div style="display:flex;align-items:center;gap:4px">{_mini_bar_100(exh, "#ef4444")}</div>'
            f'<div style="color:{cc};font-weight:700;font-size:11px;text-align:center">{ci}</div>'
            f'<div style="color:#6b7280;font-size:9px">{_h(chart_short)}</div>'
            f'</div>'
        )

    return (
        f'<div class="dr-panel">'
        f'<div class="dr-panel-title">Motor de Narrativa — Assimetria · Crowdedness · Exaustão</div>'
        f'<div style="font-size:10px;color:#6b7280;margin-bottom:8px">'
        f'▲▲ compra agressiva &nbsp; ▲ compra tática &nbsp; ↓ reduzir &nbsp; ↓↓ realizar &nbsp; ▼ short tático &nbsp; — evitar'
        f'</div>'
        f'{summary}'
        f'{header}{rows}'
        f'</div>'
    )


def _panel_ranking(
    result: "DeskIntelligenceResult",
    signals: dict[str, "AssetSignal"],
    rrg_result: "RRGResult | None",
) -> str:
    ranked = result.ranked_assets or []
    if not ranked:
        return ""

    rrg_sigs = {}
    if rrg_result and hasattr(rrg_result, "signals"):
        rrg_sigs = rrg_result.signals or {}

    opp  = result.opportunity_scores or {}
    frag = result.fragility_scores   or {}
    adj  = result.regime_adj_scores  or {}
    asym = result.asymmetry_scores   or {}
    narr = result.narrative_states   or {}
    conv_map = result.convictions    or {}

    STATE_COLOR = {
        "pre_narrative": "#818cf8", "emerging": "#22c55e",
        "mature": "#f59e0b", "exhausted": "#ef4444",
    }

    cols = "32px 80px 1fr 52px 52px 52px 80px 60px 56px"
    header = (
        f'<div style="display:grid;grid-template-columns:{cols};'
        f'gap:4px;padding:6px 8px;background:#0f172a;border-radius:6px 6px 0 0;'
        f'font-size:10px;font-weight:700;color:#6b7280;margin-bottom:1px">'
        f'<div>#</div><div>Ativo</div><div>Score Ajustado</div>'
        f'<div>Asim.</div><div>Opp.</div><div>Frag.</div>'
        f'<div>Quadrante</div><div>Dir.</div><div>Conv.</div>'
        f'</div>'
    )

    rows = ""
    for i, ticker in enumerate(ranked, 1):
        sig   = signals.get(ticker)
        rs    = rrg_sigs.get(ticker)
        comp  = adj.get(ticker, 0.0)
        op    = opp.get(ticker, 0.0)
        fr    = frag.get(ticker, 0.0)
        as_   = asym.get(ticker, 50.0)
        state = narr.get(ticker, "")
        quad  = getattr(rs, "quadrant", "") if rs else ""
        direc = (sig.direction if sig else "neutral") or "neutral"
        conv  = (sig.conviction if sig else "low") or "low"
        op_conv = conv_map.get(ticker, "")

        rank_badge = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else str(i)))
        dir_color = {"long": "#22c55e", "short": "#ef4444"}.get(direc, "#6b7280")
        state_dot = ""
        if state:
            sc = STATE_COLOR.get(state, "#6b7280")
            state_dot = f'<span style="color:{sc};font-size:8px" title="{state}">●</span> '
        row_bg = "#0a1628" if i % 2 == 0 else "#060a12"

        # Assimetria colorida
        as_color = "#22c55e" if as_ > 65 else ("#f59e0b" if as_ > 45 else "#ef4444")

        rows += (
            f'<div style="display:grid;grid-template-columns:{cols};'
            f'gap:4px;padding:6px 8px;background:{row_bg};align-items:center;font-size:11px">'
            f'<div style="color:#6b7280;font-size:10px">{rank_badge}</div>'
            f'<div style="font-weight:700;color:#e5e7eb">{state_dot}{_h(ticker)}</div>'
            f'<div>{_score_bar(comp)}</div>'
            f'<div style="color:{as_color};font-size:10px;font-weight:700">{as_:.0f}</div>'
            f'<div style="color:{"#22c55e" if op > 0.20 else "#6b7280"};font-size:10px">{op:+.2f}</div>'
            f'<div style="color:{"#ef4444" if fr > 0.40 else "#6b7280"};font-size:10px">{fr:.2f}</div>'
            f'<div style="color:{_quad_color(quad)};font-size:10px">{_quad_icon(quad)} {_h(quad)}</div>'
            f'<div style="color:{dir_color};font-size:10px;font-weight:700">{_h(direc.upper())}</div>'
            f'<div>{_conv_badge(conv)}</div>'
            f'</div>'
        )

    return (
        f'<div class="dr-panel">'
        f'<div class="dr-panel-title">Ranking — Score Ajustado ao Regime</div>'
        f'{header}{rows}'
        f'</div>'
    )


# ── Painel 6: Hidden Opportunities + Fragility ──────────────────────────────────

def _panel_opportunities_fragility(
    result: "DeskIntelligenceResult",
    signals: dict[str, "AssetSignal"],
    rrg_result: "RRGResult | None",
) -> str:
    rrg_sigs = {}
    if rrg_result and hasattr(rrg_result, "signals"):
        rrg_sigs = rrg_result.signals or {}

    # Coluna esquerda: oportunidades
    opp_cards = ""
    for ticker in result.top_opportunities[:5]:
        score = result.opportunity_scores.get(ticker, 0.0)
        sig   = signals.get(ticker)
        rs    = rrg_sigs.get(ticker)
        quad  = getattr(rs, "quadrant", "") if rs else ""
        conv  = (sig.conviction if sig else "low") or "low"
        exp_lines = (result.explanations.get(ticker) or "").split(".")
        short_exp = exp_lines[1].strip() if len(exp_lines) > 1 else ""

        opp_cards += (
            f'<div style="background:#0a1628;border:1px solid #22c55e30;border-left:3px solid #22c55e;'
            f'border-radius:6px;padding:10px 12px;margin-bottom:8px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
            f'<span style="font-size:13px;font-weight:800;color:#e5e7eb">{_h(ticker)}</span>'
            f'<div style="display:flex;gap:6px;align-items:center">'
            f'<span style="font-size:11px;color:#22c55e;font-weight:700">{score:+.2f}</span>'
            f'{_conv_badge(conv)}</div></div>'
            f'<div style="font-size:10px;color:{_quad_color(quad)}">{_quad_icon(quad)} {_h(quad)}</div>'
            f'<div style="font-size:10px;color:#94a3b8;margin-top:4px">{_h(short_exp)}</div>'
            f'</div>'
        )

    # Coluna direita: frágeis
    frag_cards = ""
    for ticker in result.top_fragile[:5]:
        score = result.fragility_scores.get(ticker, 0.0)
        sig   = signals.get(ticker)
        rs    = rrg_sigs.get(ticker)
        quad  = getattr(rs, "quadrant", "") if rs else ""
        conv  = (sig.conviction if sig else "low") or "low"
        exp_lines = (result.explanations.get(ticker) or "").split(".")
        short_exp = exp_lines[1].strip() if len(exp_lines) > 1 else ""

        frag_cards += (
            f'<div style="background:#0a1628;border:1px solid #ef444430;border-left:3px solid #ef4444;'
            f'border-radius:6px;padding:10px 12px;margin-bottom:8px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
            f'<span style="font-size:13px;font-weight:800;color:#e5e7eb">{_h(ticker)}</span>'
            f'<div style="display:flex;gap:6px;align-items:center">'
            f'<span style="font-size:11px;color:#ef4444;font-weight:700">{score:.2f}</span>'
            f'{_conv_badge(conv)}</div></div>'
            f'<div style="font-size:10px;color:{_quad_color(quad)}">{_quad_icon(quad)} {_h(quad)}</div>'
            f'<div style="font-size:10px;color:#94a3b8;margin-top:4px">{_h(short_exp)}</div>'
            f'</div>'
        )

    return (
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">'
        f'<div class="dr-panel">'
        f'<div class="dr-panel-title" style="color:#22c55e">◈ Hidden Opportunities</div>'
        f'{opp_cards or "<div style=\'color:#6b7280;font-size:12px\'>Nenhuma oportunidade identificada.</div>"}'
        f'</div>'
        f'<div class="dr-panel">'
        f'<div class="dr-panel-title" style="color:#ef4444">⚠ Fragility</div>'
        f'{frag_cards or "<div style=\'color:#6b7280;font-size:12px\'>Nenhum ativo frágil identificado.</div>"}'
        f'</div>'
        f'</div>'
    )


# ── Painel 7: Explicações (accordion) ───────────────────────────────────────────

def _panel_explanations(
    result: "DeskIntelligenceResult",
    signals: dict[str, "AssetSignal"],
    rrg_result: "RRGResult | None",
) -> str:
    explanations = result.explanations or {}
    ranked = result.ranked_assets or list(explanations.keys())

    rrg_sigs = {}
    if rrg_result and hasattr(rrg_result, "signals"):
        rrg_sigs = rrg_result.signals or {}

    items = ""
    for ticker in ranked[:20]:
        exp  = explanations.get(ticker, "")
        if not exp:
            continue
        sig  = signals.get(ticker)
        rs   = rrg_sigs.get(ticker)
        quad = getattr(rs, "quadrant", "") if rs else ""
        adj  = result.regime_adj_scores.get(ticker, 0.0)
        col  = _quad_color(quad)
        uid  = ticker.replace("-", "_").replace("^", "X").replace("=", "E")

        items += (
            f'<div style="border-bottom:1px solid #1e293b">'
            f'<button onclick="drToggle(\'{uid}\')" style="width:100%;background:none;border:none;'
            f'padding:10px 12px;cursor:pointer;display:flex;justify-content:space-between;'
            f'align-items:center;text-align:left">'
            f'<div style="display:flex;align-items:center;gap:10px">'
            f'<span style="font-size:12px;font-weight:800;color:#e5e7eb">{_h(ticker)}</span>'
            f'<span style="font-size:10px;color:{col}">{_quad_icon(quad)} {_h(quad)}</span>'
            f'</div>'
            f'<div style="display:flex;align-items:center;gap:8px">'
            f'<span style="font-size:11px;color:{"#22c55e" if adj>=0 else "#ef4444"}">'
            f'{adj:+.2f}</span>'
            f'<span id="dr-ico-{uid}" style="color:#6b7280;font-size:14px">▶</span>'
            f'</div>'
            f'</button>'
            f'<div id="dr-exp-{uid}" style="display:none;padding:0 14px 12px;'
            f'font-size:11px;color:#94a3b8;line-height:1.6">{_h(exp)}</div>'
            f'</div>'
        )

    js_accordion = (
        '<script>'
        'function drToggle(id){'
        'const el=document.getElementById("dr-exp-"+id);'
        'const ico=document.getElementById("dr-ico-"+id);'
        'if(el){el.style.display=el.style.display==="none"?"block":"none";}'
        'if(ico){ico.textContent=el.style.display==="block"?"▼":"▶";}'
        '}'
        '</script>'
    )

    return (
        f'<div class="dr-panel">'
        f'<div class="dr-panel-title">Explicações por Ativo</div>'
        f'<div style="border:1px solid #1e293b;border-radius:6px;overflow:hidden">'
        f'{items}</div>'
        f'</div>'
        f'{js_accordion}'
    )


# ── CSS base da aba ──────────────────────────────────────────────────────────────

_RADAR_CSS = """
<style>
.dr-wrap {
  display: flex;
  flex-direction: column;
  gap: 20px;
  padding: 20px 28px;
  background: #060a12;
  min-height: 100%;
  flex: 1;
  width: 100%;
  min-width: 0;
  box-sizing: border-box;
  color: #e5e7eb;
  font-family: 'Inter', 'Segoe UI', sans-serif;
}
.dr-panel {
  background: #0a1628;
  border: 1px solid #1e293b;
  border-radius: 12px;
  padding: 18px 20px;
}
.dr-panel-title {
  font-size: 13px;
  font-weight: 700;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 14px;
}
#dr-rrg-tooltip {
  position: fixed;
  display: none;
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 6px;
  padding: 8px 10px;
  font-size: 11px;
  color: #cbd5e1;
  pointer-events: none;
  z-index: 9999;
  line-height: 1.5;
  max-width: 200px;
}
</style>
"""


# ── Painel TV Zones ──────────────────────────────────────────────────────────────

def _panel_tv_zones(
    signals: dict[str, "AssetSignal"] | None,
    result: "DeskIntelligenceResult | None",
) -> str:
    """
    Painel de Value Area / VWAP / Setup Quality por ativo.
    Extrai ZoneSignals armazenados em signal._zone_signal pelo pipeline.
    """
    # Coleta zone_signals de todos os sinais disponíveis
    zone_data: list[tuple[str, object]] = []
    for ticker, sig in (signals or {}).items():
        zs = getattr(sig, "_zone_signal", None)
        if zs is not None:
            zone_data.append((ticker, zs))

    if not zone_data:
        return ""  # TV não rodou — não exibe o painel

    # Ordena: ideal primeiro, depois acceptable, stretched, avoid
    _order = {"ideal": 0, "acceptable": 1, "stretched": 2, "avoid": 3, "no_data": 4}
    zone_data.sort(key=lambda x: (_order.get(getattr(x[1], "entry_quality", "no_data"), 4), x[0]))

    # Cores e ícones
    _eq_color = {
        "ideal":      ("#22c55e", "✅"),
        "acceptable": ("#f59e0b", "🟡"),
        "stretched":  ("#f97316", "⚠"),
        "avoid":      ("#ef4444", "🔴"),
        "no_data":    ("#6b7280", "⚪"),
    }
    _tv_regime_color = {
        "bullish": "#22c55e",
        "bearish": "#ef4444",
        "neutral": "#94a3b8",
    }

    def _bar(val: float, max_val: float, color: str, w: int = 80) -> str:
        pct = min(abs(val) / max(abs(max_val), 0.001), 1.0) * 100
        return (
            f"<div style='width:{w}px;height:5px;background:rgba(255,255,255,.06);"
            f"border-radius:2px;display:inline-block;vertical-align:middle'>"
            f"<div style='width:{pct:.0f}%;height:100%;background:{color};"
            f"border-radius:2px'></div></div>"
        )

    rows = []
    for ticker, zs in zone_data:
        eq   = getattr(zs, "entry_quality", "no_data")
        color, icon = _eq_color.get(eq, ("#6b7280", "⚪"))
        tv_reg = getattr(zs, "tv_regime", "neutral")
        tv_color = _tv_regime_color.get(tv_reg, "#94a3b8")
        price = getattr(zs, "price", 0)
        vwap  = getattr(zs, "vwap", None)
        vah   = getattr(zs, "vah", None)
        val_  = getattr(zs, "val", None)
        poc   = getattr(zs, "poc", None)
        rsi   = getattr(zs, "rsi", None)
        pvwap = getattr(zs, "price_vs_vwap", None)
        if pvwap is None and vwap and price:
            pvwap = (price - vwap) / vwap
        alloc_s = getattr(zs, "allocation_scalar", 1.0)
        refined_stop = getattr(zs, "refined_stop", None)
        first_target = getattr(zs, "first_target", None)

        within_va = getattr(zs, "within_value_area", False)
        va_badge = (
            f"<span style='background:#22c55e22;color:#22c55e;font-size:8px;"
            f"padding:1px 4px;border-radius:3px;margin-left:3px'>VA</span>"
            if within_va else ""
        )

        # Scalar badge
        scalar_color = "#22c55e" if alloc_s > 1.05 else ("#ef4444" if alloc_s < 0.85 else "#94a3b8")
        scalar_sign = "+" if alloc_s > 1.0 else ""
        scalar_text = f"{(alloc_s - 1) * 100:+.0f}%"

        # VWAP delta
        pvwap_text = f"{pvwap*100:+.1f}%" if pvwap is not None else "—"
        pvwap_color = "#22c55e" if (pvwap or 0) > 0 else "#ef4444"

        # VA range
        va_range = f"{val_:,.1f}–{vah:,.1f}" if (vah and val_) else "—"

        rows.append(
            f"<tr>"
            f"<td style='padding:5px 6px;white-space:nowrap'>"
            f"  <span style='font-weight:700;color:#e2e8f0;font-size:11px'>{ticker}</span>"
            f"  {va_badge}"
            f"</td>"
            f"<td style='padding:5px 6px;text-align:center'>"
            f"  <span style='color:{color};font-size:10px;font-weight:600'>{icon} {eq}</span>"
            f"</td>"
            f"<td style='padding:5px 6px;text-align:center'>"
            f"  <span style='color:{tv_color};font-size:10px'>{tv_reg}</span>"
            f"</td>"
            f"<td style='padding:5px 6px;text-align:right;color:{pvwap_color};font-size:10px;font-family:monospace'>"
            f"  {pvwap_text}"
            f"</td>"
            f"<td style='padding:5px 6px;font-size:10px;color:#94a3b8;font-family:monospace'>"
            f"  {va_range}"
            f"</td>"
            f"<td style='padding:5px 6px;text-align:center;font-size:10px;color:#94a3b8'>"
            f"  {f'{rsi:.0f}' if rsi else '—'}"
            f"</td>"
            f"<td style='padding:5px 6px;text-align:right'>"
            f"  <span style='color:{scalar_color};font-size:10px;font-weight:600'>{scalar_text}</span>"
            f"</td>"
            f"<td style='padding:5px 6px;font-size:9px;color:#64748b;font-family:monospace;white-space:nowrap'>"
            + (f"stop {refined_stop:,.2f}" if refined_stop else "")
            + (f" → {first_target:,.2f}" if first_target else "")
            + f"</td>"
            f"</tr>"
        )

    # Summary chips
    ideal_n   = sum(1 for _, z in zone_data if getattr(z, "entry_quality", "") == "ideal")
    accept_n  = sum(1 for _, z in zone_data if getattr(z, "entry_quality", "") == "acceptable")
    stretch_n = sum(1 for _, z in zone_data if getattr(z, "entry_quality", "") == "stretched")
    avoid_n   = sum(1 for _, z in zone_data if getattr(z, "entry_quality", "") == "avoid")

    chips = (
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px'>"
        f"<span style='background:#22c55e22;color:#22c55e;font-size:10px;padding:2px 8px;border-radius:4px'>✅ Ideal: {ideal_n}</span>"
        f"<span style='background:#f59e0b22;color:#f59e0b;font-size:10px;padding:2px 8px;border-radius:4px'>🟡 Accept.: {accept_n}</span>"
        f"<span style='background:#f9741622;color:#f97316;font-size:10px;padding:2px 8px;border-radius:4px'>⚠ Stretched: {stretch_n}</span>"
        f"<span style='background:#ef444422;color:#ef4444;font-size:10px;padding:2px 8px;border-radius:4px'>🔴 Avoid: {avoid_n}</span>"
        f"<span style='font-size:9px;color:#4b5563;margin-left:auto;align-self:center'>"
        f"Alloc. scalar = zona VA+VWAP×conviction | TradingView</span>"
        f"</div>"
    )

    table = (
        f"<div style='overflow-x:auto'>"
        f"<table style='width:100%;border-collapse:collapse;font-size:11px'>"
        f"<thead><tr style='border-bottom:1px solid #1e293b;color:#475569;font-size:9px;letter-spacing:.5px;text-transform:uppercase'>"
        f"<th style='padding:4px 6px;text-align:left'>Ativo</th>"
        f"<th style='padding:4px 6px;text-align:center'>Setup</th>"
        f"<th style='padding:4px 6px;text-align:center'>Regime TV</th>"
        f"<th style='padding:4px 6px;text-align:right'>vs VWAP</th>"
        f"<th style='padding:4px 6px'>Value Area</th>"
        f"<th style='padding:4px 6px;text-align:center'>RSI</th>"
        f"<th style='padding:4px 6px;text-align:right'>Δ Alloc</th>"
        f"<th style='padding:4px 6px'>Stop → Alvo</th>"
        f"</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        f"</table></div>"
    )

    return (
        f"<div class='dr-panel'>"
        f"<div class='dr-panel-title'>◦ TradingView — Value Area · VWAP · Setup Quality</div>"
        + chips
        + table
        + "</div>"
    )


# ── Main render function ─────────────────────────────────────────────────────────

def render_desk_radar_tab(
    result: "DeskIntelligenceResult | None",
    signals: dict[str, "AssetSignal"] | None = None,
    rrg_result: "RRGResult | None" = None,
    market_prices: dict | None = None,
    network_result: dict | None = None,
) -> str:
    """
    Gera o HTML completo da aba Desk Radar.
    Retorna string HTML auto-contida (sem dependências externas).
    """
    signals = signals or {}

    if result is None:
        return (
            f'{_RADAR_CSS}'
            f'<div class="dr-wrap">'
            f'<div class="dr-panel" style="text-align:center;padding:40px">'
            f'<div style="font-size:16px;color:#6b7280">Desk Intelligence não disponível.</div>'
            f'<div style="font-size:12px;color:#4b5563;margin-top:8px">'
            f'Execute o pipeline completo para gerar os scores de inteligência.</div>'
            f'</div></div>'
        )

    tooltip_div = '<div id="dr-rrg-tooltip"></div>'

    # Stats header rápido
    stats_html = (
        f'<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:2px">'
        f'<span style="font-size:11px;color:#6b7280">{result.n_assets} ativos</span>'
        f'<span style="font-size:11px;color:#6b7280">{result.n_with_rrg} com RRG</span>'
        f'<span style="font-size:11px;color:#6b7280">'
        f'{len(result.top_opportunities)} oportunidades</span>'
        f'<span style="font-size:11px;color:#6b7280">'
        f'{len(result.top_fragile)} frágeis</span>'
        f'</div>'
    )

    narrative_panel   = _panel_narrative(result, rrg_result)
    contagion_panel   = _panel_contagion(result, signals, network_result)
    opp_frag_panel    = _panel_opportunities_fragility(result, signals, rrg_result)
    narr_engine_panel = _panel_narrative_engine(result)
    ranking_panel     = _panel_ranking(result, signals, rrg_result)
    rrg_panel         = _panel_rrg_turbinado(result, rrg_result)
    tv_zones_panel    = _panel_tv_zones(signals, result)
    explanations_panel = _panel_explanations(result, signals, rrg_result)

    # Layout: regime (full) → 2-col sidebar → RRG (full) → rest
    sidebar_row = (
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start">'
        f'{narrative_panel}'
        f'{contagion_panel}'
        f'</div>'
    )

    return (
        f'{_RADAR_CSS}'
        f'{tooltip_div}'
        f'<div class="dr-wrap">'
        f'{stats_html}'
        f'{_panel_regime(result)}'
        f'{sidebar_row}'
        f'{rrg_panel}'
        f'{tv_zones_panel}'
        f'{narr_engine_panel}'
        f'{opp_frag_panel}'
        f'{ranking_panel}'
        f'{explanations_panel}'
        f'</div>'
    )
