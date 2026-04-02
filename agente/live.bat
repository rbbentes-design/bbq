@echo off
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\rafael bentes\bbg\agente"
taskkill /f /fi "windowtitle eq MacroDesk Live*" >nul 2>&1
uv run python -m app.cli.run live run --interval 60
pause
