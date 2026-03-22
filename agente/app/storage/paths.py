"""
Gerenciamento centralizado de caminhos do workspace.

Responsabilidades:
- Resolver todos os caminhos a partir das settings
- Criar diretórios que não existem (workspace setup)
- Prover helpers de nomenclatura de arquivos por run_id e data
- Nunca misturar caminhos do código com caminhos de dados

Os diretórios do workspace NÃO ficam no git.
O código (este arquivo) fica no git.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from app.config.settings import settings


class WorkspacePaths:
    """Resolve e inicializa os caminhos do workspace de dados."""

    def __init__(self) -> None:
        self._s = settings

    # ── Raízes ────────────────────────────────────────────────────────────────

    @property
    def workspace(self) -> Path:
        return self._s.workspace_dir

    @property
    def raw(self) -> Path:
        return self._s.resolved_raw_dir()

    @property
    def normalized(self) -> Path:
        return self._s.resolved_normalized_dir()

    @property
    def bundles(self) -> Path:
        return self._s.resolved_bundles_dir()

    @property
    def logs(self) -> Path:
        return self._s.resolved_logs_dir()

    @property
    def debug(self) -> Path:
        return self._s.resolved_debug_dir()

    @property
    def state(self) -> Path:
        return self._s.resolved_state_dir()

    @property
    def state_zerohedge(self) -> Path:
        return self.state / "zerohedge"

    @property
    def state_x(self) -> Path:
        return self.state / "x"

    # ── Arquivos de sessão ────────────────────────────────────────────────────

    @property
    def zerohedge_state_file(self) -> Path:
        return self._s.resolved_zerohedge_state_path()

    @property
    def zerohedge_profile_dir(self) -> Path:
        return self._s.resolved_zerohedge_profile_dir()

    @property
    def x_state_file(self) -> Path:
        return self._s.resolved_x_state_path()

    @property
    def x_profile_dir(self) -> Path:
        return self._s.resolved_x_profile_dir()

    # ── Helpers de nomenclatura ───────────────────────────────────────────────

    def raw_html_path(self, source: str, run_id: str) -> Path:
        """Caminho para HTML bruto de uma coleta."""
        return self.raw / source / f"{run_id}.html"

    def raw_document_path(self, source: str, run_id: str) -> Path:
        """Caminho para metadados JSON do SourceDocument."""
        return self.raw / source / f"{run_id}.json"

    def normalized_blocks_path(self, source: str, run_id: str) -> Path:
        """Caminho para blocos normalizados (JSONL)."""
        return self.normalized / source / f"{run_id}.jsonl"

    def bundle_path(self, run_date: date, run_id: str) -> Path:
        """Caminho para o bundle diário."""
        date_str = run_date.isoformat()
        return self.bundles / date_str / f"{run_id}.json"

    def markdown_report_path(self, run_date: date, run_id: str) -> Path:
        """Caminho para o relatório Markdown."""
        date_str = run_date.isoformat()
        return self.bundles / date_str / f"{run_id}_report.md"

    def json_report_path(self, run_date: date, run_id: str) -> Path:
        """Caminho para o resumo JSON."""
        date_str = run_date.isoformat()
        return self.bundles / date_str / f"{run_id}_summary.json"

    def audit_log_path(self, run_date: date) -> Path:
        """Caminho para o log de auditoria diário (JSONL)."""
        return self.logs / f"audit_{run_date.isoformat()}.jsonl"

    # ── Inicialização ─────────────────────────────────────────────────────────

    def ensure_all(self) -> None:
        """
        Cria todos os diretórios do workspace se não existirem.
        Chamado na inicialização do pipeline — nunca falha silenciosamente.
        """
        dirs = [
            self.raw / "zerohedge",
            self.raw / "x",
            self.normalized / "zerohedge",
            self.normalized / "x",
            self.bundles,
            self.logs,
            self.debug,
            self.state_zerohedge,
            self.state_x,
            self.zerohedge_profile_dir,
            self.x_profile_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# Instância singleton
workspace = WorkspacePaths()
