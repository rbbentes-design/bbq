@echo off
REM Bloomberg Watcher - roda bloomberg-ingest a cada 15min
REM Importa ZIPs novos assim que aparecem em ~/Downloads

set PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%
set UV=%USERPROFILE%\.local\bin\uv.exe
set PYTHONIOENCODING=utf-8

cd /d "%USERPROFILE%\bbg\agente"

echo [%date% %time%] Watcher tick >> "%USERPROFILE%\bbg\agente\watcher.log"

"%UV%" run python -m app.cli.run bloomberg-ingest >> "%USERPROFILE%\bbg\agente\watcher.log" 2>&1

REM Se houve dados novos, regenera desk
powershell -Command "$last = Get-Content '%USERPROFILE%\bbg\agente\watcher.log' -Tail 30; if ($last -match 'Options imported|rows_ingested\"?:\s*[1-9]|BQL extracted') { exit 0 } else { exit 1 }"
if %ERRORLEVEL% NEQ 0 goto end

echo [%date% %time%] Rows ingested - regen desk >> "%USERPROFILE%\bbg\agente\watcher.log"

for /f %%f in ('powershell -Command "(Get-ChildItem '%USERPROFILE%\agente-workspace\bundles' -Directory | Sort-Object Name -Descending | Select-Object -First 1).Name"') do set LASTDAY=%%f
"%UV%" run python -m app.cli.run desk --date %LASTDAY% --no-open --no-watch >> "%USERPROFILE%\bbg\agente\watcher.log" 2>&1

:end
echo [%date% %time%] Watcher done >> "%USERPROFILE%\bbg\agente\watcher.log"
