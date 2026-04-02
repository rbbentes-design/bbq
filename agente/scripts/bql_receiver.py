"""
BQL Receiver — servidor HTTP local porta 8766
Recebe CSVs do BQuant via POST e salva em bql_data/
Inicia automaticamente pelo macroDesk.bat
"""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from datetime import datetime

OUT = Path(r'C:\Users\rafael bentes\bbg\agente\bql_data')
OUT.mkdir(parents=True, exist_ok=True)
PORT = 8766


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length  = int(self.headers.get('Content-Length', 0))
            payload = json.loads(self.rfile.read(length))
            fname   = payload['filename']
            content = payload['content']
            (OUT / fname).write_text(content, encoding='utf-8')
            ts = datetime.now().strftime('%H:%M:%S')
            print(f'[{ts}] salvo: {fname} ({len(content)} chars)')
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK')
        except Exception as e:
            print(f'[ERRO] {e}')
            self.send_response(500)
            self.end_headers()

    def log_message(self, *a):
        pass  # silencia logs HTTP


if __name__ == '__main__':
    print(f'BQL Receiver ouvindo em 0.0.0.0:{PORT}')
    print(f'Salvando em: {OUT}')
    HTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
