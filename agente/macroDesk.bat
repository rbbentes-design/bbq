@echo off
setlocal
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\rafael bentes\bbg\agente"

echo === MacroDesk — iniciando ===

:: Mata processos antigos (live loop, Streamlit, HTTP server)
echo Parando processos antigos...
taskkill /F /IM python.exe /T 2>nul
taskkill /F /IM node.exe /T 2>nul
powershell -Command "Get-NetTCPConnection -LocalPort 8501,8502,8765 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }" 2>nul
timeout /t 2 /nobreak >nul

:: Data de hoje
for /f "delims=" %%d in ('powershell -NoLogo -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%d
set BUNDLE_DIR=C:\Users\rafael bentes\agente-workspace\bundles\%TODAY%

echo Data: %TODAY%
echo Bundle dir: %BUNDLE_DIR%

:: Verifica se existe bundle do dia
if not exist "%BUNDLE_DIR%" (
    echo.
    echo [ERRO] Sem bundle para hoje. Rode primeiro:
    echo   uv run agente run ingest
    echo.
    pause
    exit /b 1
)

:: HTTP server — serve toda a pasta de bundles (acesso por data/arquivo)
echo Iniciando HTTP server em http://localhost:8765 ...
start "MacroDesk-HTTP" /B cmd /c "uv run python -m http.server 8765 --directory \"%BUNDLE_DIR%\" 2>nul"

timeout /t 2 /nobreak >nul

:: Live loop — gera e atualiza o HTML a cada 60s
echo Iniciando live loop (interval=60s)...
start "MacroDesk-Live" cmd /k "uv run python -m app.cli.live --no-open --interval 60"

:: Aguarda o HTML ser gerado (live loop leva ~15s para build inicial)
echo Aguardando build inicial (35s — SPX network ~50 tickers)...
timeout /t 35 /nobreak >nul

:: Abre o browser no HTML fixo
echo Abrindo browser...
start "" "http://localhost:8765/macroDesk.html"

echo.
echo === MacroDesk ativo! ===
echo   Browser: http://localhost:8765/macroDesk.html
echo   Auto-refresh: a cada 90s (meta-refresh no HTML)
echo   Para parar: feche a janela "MacroDesk-Live"
echo.
