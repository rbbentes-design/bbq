"""
BQL Gist Sync
Lê CSVs do GitHub Gist (publicado pelo BQuant) e salva localmente.
Roda automaticamente no loop do live.py.
"""

from __future__ import annotations

import csv
import io
from datetime import date
from pathlib import Path
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("providers.bql_gist")

# Config — lida de .bquant na raiz do projeto
_CONFIG_FILE = Path(__file__).parent.parent.parent / ".bquant"
_BQL_DATA_DIR = Path(r"C:\Users\rafael bentes\bbg\agente\bql_data")


def _load_config() -> dict[str, str]:
    cfg: dict[str, str] = {}
    if _CONFIG_FILE.exists():
        for line in _CONFIG_FILE.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                cfg[k.strip()] = v.strip()
    return cfg


def sync_from_gist() -> bool:
    """
    Baixa CSVs do Gist e salva em bql_data/.
    Retorna True se atualizou algum arquivo.
    """
    try:
        import requests
    except ImportError:
        return False

    cfg = _load_config()
    gist_id = cfg.get("GIST_ID", "")
    token   = cfg.get("GITHUB_TOKEN", "")

    if not gist_id or not token or token == "COLOQUE_NOVO_TOKEN_AQUI":
        return False

    try:
        r = requests.get(
            f"https://api.github.com/gists/{gist_id}",
            headers={"Authorization": f"token {token}"},
            timeout=10,
        )
        if r.status_code != 200:
            _log.warning("gist_sync_error", status=r.status_code)
            return False

        files = r.json().get("files", {})
        _BQL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        updated = False

        for fname, info in files.items():
            if not fname.endswith(".csv"):
                continue
            content = info.get("content", "")
            if not content:
                continue
            local = _BQL_DATA_DIR / fname
            if not local.exists() or local.read_text(encoding="utf-8") != content:
                local.write_text(content, encoding="utf-8")
                updated = True
                _log.info("gist_file_saved", file=fname)

        return updated

    except Exception as exc:
        _log.warning("gist_sync_failed", error=str(exc))
        return False
