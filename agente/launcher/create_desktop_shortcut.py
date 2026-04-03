"""
Cria um atalho "MacroDesk Bloomberg Agent" na área de trabalho do Windows.
Execute uma vez após configurar o projeto.

    python launcher/create_desktop_shortcut.py
"""

import os
import sys
from pathlib import Path


def create_shortcut() -> None:
    """Cria atalho .lnk na área de trabalho usando Windows Script Host (WSH)."""

    root     = Path(__file__).parent.parent
    bat_path = root / "launcher" / "run_bloomberg_agent.bat"
    desktop  = Path(os.path.expanduser("~")) / "Desktop"
    shortcut = desktop / "MacroDesk Bloomberg Agent.lnk"

    if not bat_path.exists():
        print(f"ERRO: launcher não encontrado em {bat_path}")
        sys.exit(1)

    if not desktop.exists():
        # Tenta OneDrive Desktop (PT-BR)
        desktop = Path(os.path.expanduser("~")) / "OneDrive" / "Área de Trabalho"
        if not desktop.exists():
            desktop = Path(os.path.expanduser("~")) / "OneDrive" / "Desktop"
        if not desktop.exists():
            print(f"AVISO: Pasta Desktop não encontrada. Crie o atalho manualmente para:\n  {bat_path}")
            return

    # Usa Windows Script Host via VBScript gerado dinamicamente
    icon_path = bat_path  # ícone default do .bat

    vbs = f"""
Set oWS = WScript.CreateObject("WScript.Shell")
sLinkFile = "{shortcut}"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = "{bat_path}"
oLink.WorkingDirectory = "{root}"
oLink.Description = "MacroDesk Bloomberg Agent — Ingestão de dados Bloomberg"
oLink.WindowStyle = 1
oLink.Save
""".strip()

    vbs_file = root / "launcher" / "_temp_shortcut.vbs"
    try:
        vbs_file.write_text(vbs, encoding="utf-8")
        ret = os.system(f'cscript //nologo "{vbs_file}"')
        if ret == 0 and shortcut.exists():
            print(f"Atalho criado: {shortcut}")
        else:
            print(f"AVISO: não foi possível criar o atalho automaticamente.")
            print(f"Crie manualmente: clique direito no Desktop → Novo → Atalho → {bat_path}")
    finally:
        if vbs_file.exists():
            vbs_file.unlink(missing_ok=True)


if __name__ == "__main__":
    create_shortcut()
