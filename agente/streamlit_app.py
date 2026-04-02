"""
MacroDesk — Streamlit App

Arquitetura simples:
  - live.bat coleta dados (BQL, IBKR) e gera o HTML
  - Streamlit apenas lê e exibe o HTML gerado
  - Auto-refresh a cada N segundos

Uso:
    uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import pathlib
from datetime import date, datetime

import streamlit as st

st.set_page_config(
    page_title="MacroDesk",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

WORKSPACE = pathlib.Path(r"C:\Users\rafael bentes\agente-workspace\bundles")


def _find_latest_html() -> pathlib.Path | None:
    """Retorna o HTML desk_v2 mais recente do dia."""
    today = str(date.today())
    day_dir = WORKSPACE / today
    if not day_dir.exists():
        return None
    candidates = sorted(
        day_dir.glob("*_desk_v2*.html"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("MacroDesk")
    refresh_interval = st.slider("Auto-refresh (s)", 10, 300, 60, step=10)

    if st.button("Refresh agora"):
        st.rerun()

    st.divider()
    html_path = _find_latest_html()
    if html_path:
        mtime = datetime.fromtimestamp(html_path.stat().st_mtime)
        st.caption(f"Arquivo: `{html_path.name}`")
        st.caption(f"Gerado: {mtime.strftime('%H:%M:%S')}")
        st.caption(f"Data: {date.today()}")
    else:
        st.warning("Nenhum HTML encontrado.\nRode `live.bat` para gerar.")

    st.divider()
    st.caption("**Como usar:**")
    st.caption("1. Rode `live.bat` — gera e atualiza o HTML")
    st.caption("2. Streamlit exibe automaticamente")
    st.caption(f"3. Auto-refresh a cada {refresh_interval}s")

# ── Main ──────────────────────────────────────────────────────────────────────

if not html_path:
    st.info("Rode `live.bat` para gerar o dashboard, depois volte aqui.")
    time.sleep(10)
    st.rerun()

html = html_path.read_text(encoding="utf-8")

# Remove o meta-refresh do HTML (Streamlit faz o refresh)
html = html.replace(
    "setTimeout(()=>{const u=location.pathname;location.replace",
    "// setTimeout(()=>{const u=location.pathname;location.replace"
)

st.components.v1.html(html, height=960, scrolling=False)

# Sem auto-refresh automático — use o botão "Refresh agora" na sidebar
# (qualquer reload reseta o iframe e perde o tab ativo)
