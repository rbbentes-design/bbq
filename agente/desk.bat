@echo off
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\rafael bentes\bbg\agente"
echo Liberando porta 8502...
powershell -Command "Get-NetTCPConnection -LocalPort 8502 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
timeout /t 2 /nobreak >nul
echo Iniciando MacroDesk (Streamlit)...
uv run streamlit run streamlit_app.py --server.port 8502 --server.headless false
pause
