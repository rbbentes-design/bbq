@echo off
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\rafael bentes\bbg\agente"
taskkill /f /fi "windowtitle eq MacroDesk Live*" >nul 2>&1
uv run python -m app.cli.run ingest --interval 60
pause
