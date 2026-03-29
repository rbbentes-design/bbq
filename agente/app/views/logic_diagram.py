"""
Views: Logic Diagram (Draw.io XML → HTML)

Gera diagrama de transmissão causal do sinal ISQ em formato
Draw.io XML renderizado como HTML standalone — sem dependências externas.

Baseado no alphaear-logic-visualizer (Awesome-finance-skills).

Uso:
    from app.views.logic_diagram import generate_diagram
    html_path = generate_diagram(isq_signal, output_path)
"""

from __future__ import annotations

import html as html_lib
from pathlib import Path
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("views.logic_diagram")

# Cores por impacto
_IMPACT_COLORS = {
    "positive": ("#d5e8d4", "#82b366"),  # (fill, stroke) — verde
    "negative": ("#f8cecc", "#b85450"),  # vermelho
    "neutral":  ("#dae8fc", "#6c8ebf"),  # azul
}

# Cores por direção do ticker
_DIR_COLORS = {
    "long":    ("#d5e8d4", "#82b366"),
    "short":   ("#f8cecc", "#b85450"),
    "neutral": ("#fff2cc", "#d6b656"),
}


def _xml_id(base: str, idx: int) -> str:
    return f"{base}_{idx}"


def build_drawio_xml(signal: Any) -> str:
    """
    Gera XML Draw.io para o diagrama de transmissão causal.

    Layout:
      - Linha superior: nós da cadeia de transmissão (da esquerda para direita)
      - Linha inferior: tickers impactados (ordenados por peso)
      - Setas conectam nós da cadeia entre si e os nós finais aos tickers
    """
    cells = []
    cell_id = 1

    def add_cell(xml: str) -> None:
        cells.append(xml)

    # ── Header (título do sinal) ──────────────────────────────────────────────
    title = html_lib.escape(signal.title[:100])
    add_cell(
        f'<mxCell id="{cell_id}" value="{title}" style="text;html=1;align=center;'
        f'verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;'
        f'fontSize=14;fontStyle=1;" vertex="1" parent="1">'
        f'<mxGeometry x="50" y="20" width="900" height="40" as="geometry"/></mxCell>'
    )
    cell_id += 1

    # ── Nós da cadeia de transmissão ──────────────────────────────────────────
    chain = signal.transmission_chain
    x_start = 50
    x_step = 200
    y_chain = 100
    node_ids: dict[int, int] = {}

    for i, node in enumerate(chain):
        fill, stroke = _IMPACT_COLORS.get(node.impact, ("#eeeeee", "#666666"))
        label = html_lib.escape(node.node)
        reasoning = html_lib.escape(node.reasoning[:80])
        x = x_start + i * x_step
        nid = cell_id
        node_ids[i] = nid
        add_cell(
            f'<mxCell id="{nid}" value="&lt;b&gt;{label}&lt;/b&gt;&lt;br/&gt;'
            f'&lt;font color=&quot;#555&quot;&gt;{reasoning}&lt;/font&gt;" '
            f'style="rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};" '
            f'vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y_chain}" width="160" height="70" as="geometry"/></mxCell>'
        )
        cell_id += 1

        # Seta para o próximo nó
        if i < len(chain) - 1:
            edge_color = stroke
            add_cell(
                f'<mxCell id="{cell_id}" style="edgeStyle=orthogonalEdgeStyle;'
                f'strokeColor={edge_color};exitX=1;exitY=0.5;entryX=0;entryY=0.5;" '
                f'edge="1" source="{nid}" target="{nid + 1}" parent="1">'
                f'<mxGeometry relative="1" as="geometry"/></mxCell>'
            )
            cell_id += 1

    # ── Tickers impactados ────────────────────────────────────────────────────
    tickers = sorted(signal.impact_tickers, key=lambda t: -t.weight)[:8]
    y_tickers = 240
    ticker_x_step = max(120, 950 // max(len(tickers), 1))
    ticker_ids: list[int] = []

    for i, ticker in enumerate(tickers):
        fill, stroke = _DIR_COLORS.get(ticker.direction, ("#eeeeee", "#666666"))
        sym = html_lib.escape(ticker.ticker)
        name = html_lib.escape(ticker.name or ticker.ticker)
        direction = ticker.direction.upper()
        weight_pct = int(ticker.weight * 100)
        x = 50 + i * ticker_x_step
        tid = cell_id
        ticker_ids.append(tid)
        add_cell(
            f'<mxCell id="{tid}" value="&lt;b&gt;{sym}&lt;/b&gt;&lt;br/&gt;{name}&lt;br/&gt;'
            f'&lt;font color=&quot;#555&quot;&gt;{direction} | {weight_pct}%&lt;/font&gt;" '
            f'style="rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};" '
            f'vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y_tickers}" width="110" height="65" as="geometry"/></mxCell>'
        )
        cell_id += 1

    # Seta do último nó da cadeia para cada ticker
    if node_ids and ticker_ids:
        last_node_id = node_ids[len(chain) - 1]
        for tid in ticker_ids:
            add_cell(
                f'<mxCell id="{cell_id}" style="edgeStyle=elbowEdgeStyle;strokeColor=#aaaaaa;'
                f'dashed=1;" edge="1" source="{last_node_id}" target="{tid}" parent="1">'
                f'<mxGeometry relative="1" as="geometry"/></mxCell>'
            )
            cell_id += 1

    # ── Monta XML completo ────────────────────────────────────────────────────
    cells_xml = "\n    ".join(cells)
    return f"""<mxGraphModel>
  <root>
    <mxCell id="0"/>
    <mxCell id="1" parent="0"/>
    {cells_xml}
  </root>
</mxGraphModel>"""


def generate_diagram(signal: Any, output_path: Path) -> Path | None:
    """
    Gera diagrama HTML standalone com o Draw.io viewer embutido.

    Args:
        signal:      ISQSignal com transmission_chain e impact_tickers.
        output_path: Caminho de saída .html.

    Returns:
        Path do HTML gerado, ou None se falhar.
    """
    try:
        xml = build_drawio_xml(signal)
        xml_escaped = html_lib.escape(xml)

        # HTML com Draw.io viewer via CDN (mxGraph)
        html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>ISQ Diagram — {html_lib.escape(signal.title[:60])}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 0; padding: 10px; background: #f5f5f5; }}
  h2 {{ font-size: 14px; color: #333; margin-bottom: 4px; }}
  .meta {{ font-size: 12px; color: #666; margin-bottom: 12px; }}
  .diagram-container {{ background: white; border: 1px solid #ddd; border-radius: 4px; padding: 8px; }}
  .sentiment {{ display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 11px; font-weight: bold; }}
  .pos {{ background: #d5e8d4; color: #2d6a2d; }}
  .neg {{ background: #f8cecc; color: #a00; }}
  .neu {{ background: #dae8fc; color: #1a4a8a; }}
</style>
</head>
<body>
<h2>Cadeia de Transmissão Causal — {html_lib.escape(signal.title)}</h2>
<div class="meta">
  Sentimento: <span class="sentiment {'pos' if signal.sentiment_score > 0.1 else 'neg' if signal.sentiment_score < -0.1 else 'neu'}">{
  'BULLISH' if signal.sentiment_score > 0.1 else 'BEARISH' if signal.sentiment_score < -0.1 else 'NEUTRO'
  } ({signal.sentiment_score:+.2f})</span>
  &nbsp;|&nbsp; Confiança: {signal.confidence:.0%}
  &nbsp;|&nbsp; Intensidade: {'⬛' * signal.intensity}{'⬜' * (5 - signal.intensity)}
</div>
<div class="diagram-container">
<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph='{{"highlight":"#0000ff","nav":true,"resize":true,"toolbar":"zoom layers tags lightbox","edit":"_blank","xml":"{xml_escaped}"}}'>
</div>
</div>
<br>
<details>
<summary style="cursor:pointer;font-size:12px;color:#666;">Análise qualitativa</summary>
<p style="font-size:12px;color:#444;max-width:900px;">{html_lib.escape(signal.reasoning)}</p>
</details>
<script type="text/javascript" src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>
</body>
</html>"""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding="utf-8")
        _log.info("logic_diagram_saved", path=str(output_path))
        return output_path

    except Exception as exc:
        _log.warning("logic_diagram_error", error=str(exc))
        return None
