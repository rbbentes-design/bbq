from __future__ import annotations

from pathlib import Path

from app.models.daily_ingestion_bundle import DailyIngestionBundle
from app.storage.paths import workspace


class BundleStore:
    """
    Persiste e carrega o DailyIngestionBundle completo em JSON.
    Um arquivo por execução, organizado por data.
    """

    def save(self, bundle: DailyIngestionBundle) -> Path:
        path = workspace.bundle_path(bundle.run_date, bundle.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load(self, run_date: object, run_id: str) -> DailyIngestionBundle:
        from datetime import date as Date
        path = workspace.bundle_path(run_date, run_id)  # type: ignore[arg-type]
        return DailyIngestionBundle.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    def list_bundles(self) -> list[Path]:
        """Lista todos os bundles salvos, ordenados do mais recente ao mais antigo."""
        return sorted(
            workspace.bundles.rglob("*.json"),
            reverse=True,
        )


bundle_store = BundleStore()
