@echo off
title MacroDesk Live
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\rafael bentes\bbg\agente"

echo.
echo  ============================================
echo   MACRO DESK
echo  ============================================
echo.

:: Mata instancia anterior se existir
taskkill /f /fi "windowtitle eq MacroDesk Live*" >nul 2>&1

:: Roda pipeline completo: coleta, HTML, abre browser, live loop
uv run python -m app.cli.run ingest --interval 60
