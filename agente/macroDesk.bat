@echo off
setlocal
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\rafael bentes\bbg\agente"

echo === MacroDesk — iniciando ===

:: Mata apenas processos nas portas usadas (nao mata todo Python)
echo Liberando portas 8765 e 8766...
powershell -NoProfile -Command ^
  "Get-NetTCPConnection -LocalPort 8765,8766 -ErrorAction SilentlyContinue ^
   | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
timeout /t 2 /nobreak >nul

:: Data de hoje
for /f "delims=" %%d in ('powershell -NoLogo -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%d
set BUNDLE_DIR=C:\Users\rafael bentes\agente-workspace\bundles\%TODAY%

echo Data: %TODAY%
echo Bundle dir: %BUNDLE_DIR%

:: Verifica bundle
if not exist "%BUNDLE_DIR%" (
    echo.
    echo [ERRO] Sem bundle para hoje. Rode primeiro:
    echo   uv run python -m app.cli.run ingest
    echo.
    pause
    exit /b 1
)

:: BQL Receiver — porta 8766
echo Iniciando BQL Receiver...
start "BQL-Receiver" /min cmd /c "uv run python scripts\bql_receiver.py"
timeout /t 2 /nobreak >nul

:: HTTP server — porta 8765
echo Iniciando HTTP server em http://localhost:8765 ...
start "MacroDesk-HTTP" /B cmd /c "uv run python -m http.server 8765 --directory \"%BUNDLE_DIR%\" 2>nul"
timeout /t 2 /nobreak >nul

:: Live loop
echo Iniciando live loop (interval=60s)...
start "MacroDesk-Live" cmd /k "uv run python -m app.cli.live --no-open --interval 60"

:: Aguarda build inicial
echo Aguardando build inicial (35s)...
timeout /t 35 /nobreak >nul

:: Abre browser
echo Abrindo browser...
start "" "http://localhost:8765/macroDesk.html"

echo.
echo === MacroDesk ativo! ===
echo   Browser: http://localhost:8765/macroDesk.html
echo   Para parar: feche a janela "MacroDesk-Live"
echo.
