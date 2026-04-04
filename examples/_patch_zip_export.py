# -*- coding: utf-8 -*-
"""
Adiciona botão "📦 Export ZIP" ao Greeks Dashboard.

ZIP exportado contém:
  ├── metrics.json   — GEX, IV, Squeeze, Tail, z-scores, etc.
  └── jarvis.html    — JARVIS HUD completo

Uso no BQuant (célula Jupyter após carregar greeks_dashboard.py):
    exec(open('_patch_zip_export.py').read())
"""

import ast
import pathlib

DASH_FILE = pathlib.Path(__file__).parent / 'greeks_dashboard.py'

src = DASH_FILE.read_text(encoding='utf-8')

# ── Âncoras reais encontradas no arquivo ─────────────────────────────────────
ANCHOR_BTN   = "export_btn.on_click(_on_export)"
ANCHOR_HBOX  = "wd.HBox([run_btn, spx_pred_w, flow_pred_w, disp_w, cta_weight_w, export_btn])"
ANCHOR_VBOX  = "    out_export,\n    out_main"

# ── Código ZIP a injetar ──────────────────────────────────────────────────────
ZIP_CODE = '''
# ── ZIP Export ────────────────────────────────────────────────────────────────
zip_btn = wd.Button(
    description='\U0001f4e6 Export ZIP',
    button_style='info',
    icon='download',
    layout={'width': '160px'},
)
out_zip = wd.Output()


def _on_export_zip(_):
    with out_zip:
        clear_output(wait=True)
        if not _snapshot.get('ts'):
            print("\u26a0\ufe0f Rode a an\\xe1lise primeiro.")
            return
        try:
            import io as _io
            import zipfile as _zf
            import json as _json
            import base64 as _b64

            ticker   = _snapshot['ticker']
            ts_safe  = str(_snapshot['ts']).replace(':', '').replace(' ', '_')
            zip_name = f"greeks_{ticker.replace(' ', '_')}_{ts_safe}.zip"

            buf = _io.BytesIO()
            with _zf.ZipFile(buf, 'w', _zf.ZIP_DEFLATED) as zf:
                # 1. metrics.json
                payload = {
                    'ticker':  ticker,
                    'spot':    _snapshot.get('spot'),
                    'ts':      _snapshot.get('ts'),
                    'metrics': _snapshot.get('metrics', {}),
                }
                zf.writestr('metrics.json',
                            _json.dumps(payload, indent=2, default=str))

                # 2. jarvis.html
                html = _export_dashboard_html()
                if html:
                    if isinstance(html, str):
                        html = html.encode('utf-8')
                    zf.writestr('jarvis.html', html)

            buf.seek(0)
            data = buf.read()
            b64  = _b64.b64encode(data).decode('ascii')

            js = (
                f"var a=document.createElement('a');"
                f"a.href='data:application/zip;base64,{b64}';"
                f"a.download='{zip_name}';"
                f"document.body.appendChild(a);a.click();"
                f"document.body.removeChild(a);"
            )
            display(HTML(f"<script>{js}</script>"))
            display(wd.HTML(
                f"<div class='mm-dash'><div class='mm-card'>"
                f"<p>\u2705 ZIP gerado: <b>{zip_name}</b></p>"
                f"<p><small>metrics.json + jarvis.html &middot; "
                f"{len(data)/1024:.0f} KB</small></p>"
                f"</div></div>"
            ))
        except Exception as exc:
            print(f"\u274c Erro ao gerar ZIP: {exc}")


zip_btn.on_click(_on_export_zip)
'''

# ── Aplicar patches ───────────────────────────────────────────────────────────
applied = 0
errors  = []

# 1. Injeta código ZIP logo depois de export_btn.on_click(_on_export)
if ANCHOR_BTN in src:
    src = src.replace(
        ANCHOR_BTN,
        ANCHOR_BTN + "\n" + ZIP_CODE,
        1,
    )
    applied += 1
    print("[1/3] ZIP export code injetado OK")
else:
    errors.append("[1/3] ERRO: ancora nao encontrada: export_btn.on_click(_on_export)")

# 2. Adiciona zip_btn no HBox de controles
if ANCHOR_HBOX in src:
    NEW_HBOX = ANCHOR_HBOX.replace("export_btn])", "export_btn, zip_btn])")
    src = src.replace(ANCHOR_HBOX, NEW_HBOX, 1)
    applied += 1
    print("[2/3] zip_btn adicionado ao HBox OK")
else:
    errors.append("[2/3] ERRO: HBox nao encontrado")

# 3. Adiciona out_zip no VBox de display
if ANCHOR_VBOX in src:
    src = src.replace(
        ANCHOR_VBOX,
        "    out_export,\n    out_zip,\n    out_main",
        1,
    )
    applied += 1
    print("[3/3] out_zip adicionado ao VBox OK")
else:
    errors.append("[3/3] ERRO: VBox final nao encontrado")

for e in errors:
    print(e)

print(f"\nAplicado {applied}/3 alterações")

if applied == 0:
    print("Nenhuma alteração feita — arquivo não modificado.")
else:
    # Verificação de sintaxe antes de salvar
    try:
        ast.parse(src)
        print("Sintaxe: OK")
    except SyntaxError as e:
        print(f"ERRO DE SINTAXE apos patch: {e}")
        print("Arquivo NAO salvo para evitar corrupcao.")
        raise SystemExit(1)

    DASH_FILE.write_text(src, encoding='utf-8')
    print(f"Salvo: {DASH_FILE}")
