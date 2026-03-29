"""
Configura o Windows Task Scheduler para rodar o agente todo dia as 05:30.

Uso:
    python scripts/schedule_setup.py          # instala a tarefa
    python scripts/schedule_setup.py --remove  # remove a tarefa
    python scripts/schedule_setup.py --status  # mostra status
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TASK_NAME = "AgenteEditorialDiario"
SCHEDULE_TIME = "05:30"


def _venv_python() -> Path:
    here = Path(__file__).resolve().parent.parent
    for candidate in [
        here / ".venv" / "Scripts" / "python.exe",
        here / "venv" / "Scripts" / "python.exe",
    ]:
        if candidate.exists():
            return candidate
    sys.exit("ERROR: .venv nao encontrado. Rode 'pip install -e .' primeiro.")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def install() -> None:
    python = _venv_python()
    root = _project_root()
    run_script = root / "scripts" / "run_daily.bat"

    # Cria o .bat que o scheduler vai chamar
    bat_content = f"""@echo off
cd /d "{root}"
"{python}" -m app.cli.run --headless --no-open >> "{root / 'run_log.txt'}" 2>&1
"""
    run_script.write_text(bat_content, encoding="utf-8")
    print(f"Criado: {run_script}")

    # Cria a tarefa no Task Scheduler
    cmd = [
        "schtasks", "/Create", "/F",
        "/TN", TASK_NAME,
        "/TR", str(run_script),
        "/SC", "DAILY",
        "/ST", SCHEDULE_TIME,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Tarefa '{TASK_NAME}' criada com sucesso.")
        print(f"Horario: {SCHEDULE_TIME} todos os dias")
        print(f"Log: {root / 'run_log.txt'}")
    else:
        print(f"ERRO ao criar tarefa:\n{result.stderr}")
        sys.exit(1)


def remove() -> None:
    cmd = ["schtasks", "/Delete", "/F", "/TN", TASK_NAME]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Tarefa '{TASK_NAME}' removida.")
    else:
        print(f"Nao foi possivel remover: {result.stderr.strip()}")


def status() -> None:
    cmd = ["schtasks", "/Query", "/TN", TASK_NAME, "/FO", "LIST", "/V"]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="cp850")
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if any(k in line for k in ["Nome da tarefa", "Task To Run", "Proximo", "Next Run",
                                        "Status", "Ultimo", "Last Run"]):
                print(line)
    else:
        print(f"Tarefa '{TASK_NAME}' nao encontrada.")


if __name__ == "__main__":
    if "--remove" in sys.argv:
        remove()
    elif "--status" in sys.argv:
        status()
    else:
        install()
