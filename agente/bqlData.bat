@echo off
setlocal
set PYTHONIOENCODING=utf-8
set BQNT_PY=C:\blp\bqnt\environments\bqnt-3\python.exe
set SCRIPT=%~dp0scripts\bql_export.py

echo === BQL Data Export ===
echo Ambiente: %BQNT_PY%
echo Script:   %SCRIPT%
echo Saida:    C:\Users\rafael bentes\agente-workspace\bql_data\
echo.

if not exist "%BQNT_PY%" (
    echo [ERRO] BQuant Python nao encontrado em %BQNT_PY%
    echo Verifique se o Bloomberg Terminal esta aberto.
    pause
    exit /b 1
)

:: Modo loop: atualiza a cada 3 minutos automaticamente
echo Iniciando loop (Ctrl+C para parar)...
echo Os dados serao atualizados a cada 3 minutos.
echo O MacroDesk detecta automaticamente e renova o dashboard.
echo.

"%BQNT_PY%" "%SCRIPT%" --loop

pause
