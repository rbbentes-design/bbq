@echo off
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\rafael bentes\agente-workspace\bundles\2026-04-02"
echo Servidor HTTP rodando em http://localhost:8765
echo Pressione Ctrl+C para parar
uv run --directory "C:\Users\rafael bentes\bbg\agente" python -m http.server 8765
pause
