"""
BQL Unzip — monitora Downloads por bql_data_*.zip e extrai pro bql_data/
Chamado automaticamente pelo live.py a cada ciclo.
"""

from pathlib import Path
import zipfile

DOWNLOADS  = Path.home() / "Downloads"
BQL_DATA   = Path(r"C:\Users\rafael bentes\bbg\agente\bql_data")
STATE_FILE = BQL_DATA / ".last_zip"


def _latest_zip() -> Path | None:
    zips = sorted(DOWNLOADS.glob("bql_data_*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


def _state() -> str:
    if STATE_FILE.exists():
        try:
            return STATE_FILE.read_text().strip()
        except Exception:
            pass
    return ""


def extract_if_new() -> bool:
    BQL_DATA.mkdir(parents=True, exist_ok=True)

    latest = _latest_zip()
    if not latest:
        return False

    # Compara mtime + tamanho — detecta tanto arquivo novo quanto sobrescrito
    stat = latest.stat()
    key  = f"{stat.st_mtime:.0f}_{stat.st_size}"
    if key == _state():
        return False

    with zipfile.ZipFile(latest, "r") as z:
        for member in z.namelist():
            p = Path(member)
            if p.suffix == ".csv" and len(p.parts) <= 2:
                (BQL_DATA / p.name).write_bytes(z.read(member))

    STATE_FILE.write_text(key)
    print(f"[bql_unzip] extraído: {latest.name} → {BQL_DATA}")

    # Acumula no banco histórico
    try:
        import sys, os
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from app.providers.bql_db import append_from_csvs
        n = append_from_csvs()
        print(f"[bql_unzip] {n} linhas inseridas no histórico")
    except Exception as e:
        print(f"[bql_unzip] db warn: {e}")

    return True


if __name__ == "__main__":
    if extract_if_new():
        print("Dados atualizados.")
    else:
        print("Nenhum zip novo encontrado.")
