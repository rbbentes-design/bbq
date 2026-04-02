@echo off
setlocal
set PYTHONIOENCODING=utf-8
set BQNT_PY=C:\blp\bqnt\environments\bqnt-3\python.exe
set SCRIPT=%~dp0scripts\bql_export.py

echo === BQL Data Export (local) ===
echo Python: %BQNT_PY%
echo Script: %SCRIPT%
echo Saida:  C:\Users\rafael bentes\bbg\agente\bql_data\
echo.

if not exist "%BQNT_PY%" (
    echo [ERRO] Bloomberg Python nao encontrado: %BQNT_PY%
    echo Verifique se o Bloomberg Terminal esta aberto.
    pause
    exit /b 1
)

echo Iniciando... Ctrl+C para parar.
echo.

"%BQNT_PY%" "%SCRIPT%"

pause
