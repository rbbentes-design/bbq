@echo off
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\rafael bentes\bbg\agente"
title MacroDesk Morning

echo [%TIME%] 1/3 Bloomberg ingest...
uv run python -c "
import sys
sys.path.insert(0, '.')
from core.bloomberg_main_agent import BloombergMainAgent
r = BloombergMainAgent().run()
print(f'Bloomberg DB: {r.rows_ingested} linhas ({r.status})')
" 2>&1

echo [%TIME%] 2/3 Coleta fontes (X, ZeroHedge, SpotGamma...)
uv run python -m app.cli.run ingest --no-open 2>&1

echo [%TIME%] 3/3 MacroDesk (portfolio + HTML)...
uv run python -m app.cli.run desk
