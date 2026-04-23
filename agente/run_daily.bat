@echo off
REM Agente Daily Run - roda pipeline completo
REM Agendado via Windows Task Scheduler (unica task: "Agente Daily Pipeline")
REM
REM Seg (pre-market): ingest (X/RSS/SG novos) + writer (week_ahead) + desk
REM Ter-Sex:          all (ingest completo + writer + desk)
REM Sabado:           writer (podcast) + desk com bundle mais recente
REM Domingo:          tese_livre + desk com bundle mais recente

set PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%
set UV=%USERPROFILE%\.local\bin\uv.exe
set PYTHONIOENCODING=utf-8

cd /d "%USERPROFILE%\bbg\agente"

echo [%date% %time%] Inicio >> "%USERPROFILE%\bbg\agente\daily.log"

REM ── Bloomberg: extrai ZIPs + ingere DB ANTES do pipeline (dados sempre frescos) ──
echo [%date% %time%] Bloomberg extract + ingest >> "%USERPROFILE%\bbg\agente\daily.log"
"%UV%" run python -m app.cli.run bloomberg-ingest >> "%USERPROFILE%\bbg\agente\daily.log" 2>&1

for /f %%d in ('powershell -Command "(Get-Date).DayOfWeek.value__"') do set DOW=%%d

if %DOW%==0 goto sunday
if %DOW%==1 goto monday
if %DOW%==6 goto saturday

:weekday
REM Terca a Sexta: pipeline completo (market data fresco)
echo [%date% %time%] Dia util (ter-sex) - all >> "%USERPROFILE%\bbg\agente\daily.log"
"%UV%" run python -m app.cli.run all >> "%USERPROFILE%\bbg\agente\daily.log" 2>&1
goto jdesk

:monday
REM Segunda pre-market: ingest fontes + writer (week_ahead) + desk
echo [%date% %time%] Segunda - ingest + writer + desk >> "%USERPROFILE%\bbg\agente\daily.log"
"%UV%" run python -m app.cli.run ingest >> "%USERPROFILE%\bbg\agente\daily.log" 2>&1
"%UV%" run python -m app.cli.run writer >> "%USERPROFILE%\bbg\agente\daily.log" 2>&1
"%UV%" run python -m app.cli.run desk --no-open --no-watch >> "%USERPROFILE%\bbg\agente\daily.log" 2>&1
goto jdesk

:saturday
REM Sabado: podcast + desk com dados mais recentes
for /f %%f in ('powershell -Command "(Get-ChildItem '%USERPROFILE%\agente-workspace\bundles' -Directory | Sort-Object Name -Descending | Select-Object -First 1).Name"') do set LASTDAY=%%f
echo [%date% %time%] Sabado - writer (podcast) + desk (bundle: %LASTDAY%) >> "%USERPROFILE%\bbg\agente\daily.log"
"%UV%" run python -m app.cli.run writer --date %LASTDAY% >> "%USERPROFILE%\bbg\agente\daily.log" 2>&1
"%UV%" run python -m app.cli.run desk --date %LASTDAY% --no-open --no-watch >> "%USERPROFILE%\bbg\agente\daily.log" 2>&1
goto jdesk

:sunday
REM Domingo: tese_livre + desk
for /f %%f in ('powershell -Command "(Get-ChildItem '%USERPROFILE%\agente-workspace\bundles' -Directory | Sort-Object Name -Descending | Select-Object -First 1).Name"') do set LASTDAY=%%f
echo [%date% %time%] Domingo - writer (tese_livre) + desk (bundle: %LASTDAY%) >> "%USERPROFILE%\bbg\agente\daily.log"
"%UV%" run python -m app.cli.run writer --date %LASTDAY% >> "%USERPROFILE%\bbg\agente\daily.log" 2>&1
"%UV%" run python -m app.cli.run desk --date %LASTDAY% --no-open --no-watch >> "%USERPROFILE%\bbg\agente\daily.log" 2>&1
goto jdesk

:jdesk
echo [%date% %time%] Pipeline ok >> "%USERPROFILE%\bbg\agente\daily.log"

cd /d "%USERPROFILE%\bbg\j-desk"
if exist "C:\blp\bqnt\environments\bqnt-3\python.exe" (
    C:\blp\bqnt\environments\bqnt-3\python.exe -m jdesk.run --no-open >> "%USERPROFILE%\bbg\agente\daily.log" 2>&1
) else (
    "%UV%" run python -m jdesk.run --no-open >> "%USERPROFILE%\bbg\agente\daily.log" 2>&1
)

echo [%date% %time%] Concluido >> "%USERPROFILE%\bbg\agente\daily.log"
