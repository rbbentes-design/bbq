from __future__ import annotations

from pathlib import Path

from app.models.source_document import SourceDocument
from app.storage.paths import workspace
from app.utils.hashing import sha256_of_str
from app.utils.timestamps import new_ulid, utcnow


class RawStore:
    """
    Persiste o HTML bruto coletado e o SourceDocument associado.
    Separa artefato físico (HTML) de metadados (JSON).
    """

    def save_html(self, source: str, run_id: str, html: str) -> Path:
        path = workspace.raw_html_path(source, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        return path

    def save_document(self, doc: SourceDocument) -> Path:
        path = workspace.raw_document_path(doc.source_name, doc.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        return path

    def build_document(
        self,
        source_name: str,
        source_url: str,
        access_method: str,
        html: str,
        metadata: dict | None = None,
    ) -> tuple[SourceDocument, Path]:
        """
        Salva o HTML e cria o SourceDocument correspondente.
        Retorna (documento, caminho_do_html).
        """
        run_id = new_ulid()
        content_hash = sha256_of_str(html)
        html_path = self.save_html(source_name, run_id, html)

        doc = SourceDocument(
            id=run_id,
            source_name=source_name,
            source_url=source_url,
            collected_at=utcnow(),
            access_method=access_method,
            raw_content_path=str(html_path),
            content_hash=content_hash,
            metadata=metadata or {},
        )
        self.save_document(doc)
        return doc, html_path

    def load_html(self, source: str, run_id: str) -> str:
        return workspace.raw_html_path(source, run_id).read_text(encoding="utf-8")

    def load_document(self, source: str, run_id: str) -> SourceDocument:
        path = workspace.raw_document_path(source, run_id)
        return SourceDocument.model_validate_json(path.read_text(encoding="utf-8"))


raw_store = RawStore()
