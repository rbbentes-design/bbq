@echo off
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\rafael bentes\bbg\agente"
title MacroDesk Morning

echo [%TIME%] Bloomberg ingest (se CSV novo disponivel)...
uv run python -c "
import sys
sys.path.insert(0, '.')
from core.bloomberg_main_agent import BloombergMainAgent
r = BloombergMainAgent().run()
print(f'Bloomberg DB: {r.rows_ingested} linhas ({r.status})')
" 2>&1

echo [%TIME%] Iniciando pipeline completo (ingest + writer + desk)...
uv run python -m app.cli.run all
