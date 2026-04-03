@echo off
chcp 65001 > nul
title MacroDesk Bloomberg Agent

REM ────────────────────────────────────────────────────────────────────────────
REM  MacroDesk Bloomberg Agent — Launcher
REM
REM  Clique duplo neste arquivo para iniciar o agente.
REM  Ou crie um atalho na área de trabalho apontando para este .bat
REM
REM  O agente irá:
REM    1. Procurar arquivos .zip do Bloomberg em Downloads
REM    2. Extrair e renomear os CSVs com timestamp
REM    3. Salvar em bql_data/
REM    4. Normalizar e gravar no banco SQLite (macrodesk.db)
REM    5. Exibir resultado na tela
REM ────────────────────────────────────────────────────────────────────────────

REM Vai para a pasta raiz do projeto (dois níveis acima do launcher/)
cd /d "%~dp0.."

echo.
echo  ============================================================
echo   MacroDesk Bloomberg Agent
echo  ============================================================
echo.

REM ── Detecta o Python correto ────────────────────────────────────────────────
REM Tenta .venv primeiro (instalação padrão do projeto)
set PYTHON=

if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
    echo  Python: .venv\Scripts\python.exe
    goto :run
)

REM Tenta uv (gerenciador de pacotes alternativo)
where uv > nul 2>&1
if %ERRORLEVEL% == 0 (
    echo  Usando: uv run
    uv run python -m core.bloomberg_main_agent
    goto :done
)

REM Fallback: python do sistema
where python > nul 2>&1
if %ERRORLEVEL% == 0 (
    set PYTHON=python
    echo  Python: sistema
    goto :run
)

echo  ERRO: Python nao encontrado.
echo  Instale Python ou configure o .venv do projeto.
pause
exit /b 1

:run
REM ── Opções de execução ──────────────────────────────────────────────────────
REM  Modo 1: Interface Tkinter (padrão) — descomente para usar
REM  Modo 2: Linha de comando (headless) — padrão atual

REM Detecta se --headless foi passado como argumento
set HEADLESS=0
for %%a in (%*) do (
    if "%%a"=="--headless" set HEADLESS=1
    if "%%a"=="-h"         set HEADLESS=1
)

if "%HEADLESS%"=="1" (
    echo  Modo: headless (linha de comando)
    echo.
    "%PYTHON%" -m core.bloomberg_main_agent
) else (
    REM Verifica se tkinter está disponível
    "%PYTHON%" -c "import tkinter" > nul 2>&1
    if %ERRORLEVEL% == 0 (
        echo  Modo: interface grafica
        echo.
        "%PYTHON%" launcher\desktop_ui.py
    ) else (
        echo  Tkinter nao disponivel — modo headless
        echo.
        "%PYTHON%" -m core.bloomberg_main_agent
    )
)

:done
echo.
echo  ============================================================
echo   Execucao concluida.
echo  ============================================================
echo.

REM Mantém a janela aberta se rodou em modo headless
if "%HEADLESS%"=="1" (
    echo  Pressione qualquer tecla para fechar...
    pause > nul
)

exit /b 0
