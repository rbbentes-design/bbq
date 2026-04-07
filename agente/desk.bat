@echo off
chcp 65001 >/dev/null 2>&1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\rafael bentes\bbg\agente"
title MacroDesk
uv run python -m app.cli.run desk
pause
