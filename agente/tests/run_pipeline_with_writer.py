"""
Roda o pipeline completo (coleta + curadoria) e depois gera o texto editorial
no modo determinado pela data passada como argumento.

Uso:
    python tests/run_pipeline_with_writer.py              # modo do dia
    python tests/run_pipeline_with_writer.py 2026-03-27  # force sexta (week_recap)
"""
import sys
import json
from pathlib import Path
from datetime import date

from app.pipeline.ingestion import run_ingestion
from app.curation.writer import to_docx, write
from app.audit.logger import configure_logging

configure_logging("INFO")

write_date = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()

# 1. Pipeline completo
print(f"\nRodando pipeline completo... (modo editorial: {write_date})\n{'='*60}")
bundle = run_ingestion(headless=True)

# 2. Carrega curation result gerado pelo pipeline
run_id = bundle.run_id
run_date_str = bundle.run_date if hasattr(bundle, "run_date") else str(date.today())
curation_path = bundle.artifact_paths.get("curation")

if not curation_path or not Path(curation_path).exists():
    print("Curadoria não encontrada. Abortando escrita.")
    sys.exit(1)

from app.curation.models import CurationResult
result = CurationResult.model_validate(
    json.loads(Path(curation_path).read_text(encoding="utf-8"))
)

# Override da data para determinar o modo editorial
result.run_date = write_date

# 3. Escreve
print(f"\nGerando texto no modo {write_date}...\n{'='*60}")
output = write(result, bundle)

print(f"\n[MODO]: {output.mode}")
print(f"[FOCO]: {output.focus[:120]}")
print(f"\n{'='*60}\n")
print(output.text[:800], "...\n")

# 4. Salva .docx
out_dir = Path(curation_path).parent
out_path = out_dir / f"{run_id}_written_{output.mode}.docx"
try:
    docx_bytes = to_docx(output.text, output.mode, write_date)
    out_path.write_bytes(docx_bytes)
    print(f"Salvo: {out_path}")
except Exception as e:
    print(f"Erro docx ({e}), salvando .txt")
    out_path = out_path.with_suffix(".txt")
    out_path.write_text(f"# [{output.mode.upper()}] {write_date}\n\n{output.text}\n", encoding="utf-8")
    print(f"Salvo: {out_path}")
