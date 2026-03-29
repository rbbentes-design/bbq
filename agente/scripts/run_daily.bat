@echo off
cd /d "C:\Users\rafael bentes\bbg\agente"
"C:\Users\rafael bentes\bbg\agente\.venv\Scripts\python.exe" -m app.cli.run --headless --no-open >> "C:\Users\rafael bentes\bbg\agente\run_log.txt" 2>&1
