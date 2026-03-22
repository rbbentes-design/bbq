"""
Configuração central da aplicação.

Todas as variáveis de configuração são lidas do arquivo .env via Pydantic Settings.
Nenhum segredo é hardcoded aqui. Valores sensíveis nunca são expostos em logs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Ambiente ──────────────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ── Workspace externo ─────────────────────────────────────────────────────
    # Raiz do workspace de dados — deve ficar FORA do repositório git.
    # Configurável por .env para facilitar migração futura (ex: montar bucket S3).
    workspace_dir: Path = Field(
        default=Path.home() / "agente-workspace",
        description="Diretório raiz do workspace de dados (fora do git).",
    )

    # Subdirs derivados de workspace_dir. Podem ser sobrescritos individualmente.
    raw_dir: Path | None = None
    normalized_dir: Path | None = None
    bundles_dir: Path | None = None
    logs_dir: Path | None = None
    debug_dir: Path | None = None
    state_dir: Path | None = None

    # ── Playwright ────────────────────────────────────────────────────────────
    playwright_headless: bool = False
    auth_timeout_ms: int = 120_000
    debug_auth: bool = False

    # ── Chrome Profile ────────────────────────────────────────────────────────
    # Reutiliza perfil Chrome existente (ja autenticado) em vez de login manual.
    chrome_user_data_dir: Path = Field(
        default=Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "User Data",
        description="Caminho do User Data do Chrome (pasta que contem os perfis).",
    )
    chrome_profile: str = Field(
        default="Default",
        description="Nome do perfil Chrome a usar (ex: 'Default', 'Profile 1').",
    )

    # ── ZeroHedge ─────────────────────────────────────────────────────────────
    zerohedge_base_url: str = "https://www.zerohedge.com"
    zerohedge_market_ear_url: str = "https://www.zerohedge.com/the-market-ear"
    zerohedge_state_path: Path | None = None
    zerohedge_profile_dir: Path | None = None

    # ── X ─────────────────────────────────────────────────────────────────────
    x_provider_mode: Literal["browser", "api"] = "browser"
    x_timeline_limit: int = 50
    x_state_path: Path | None = None
    x_profile_dir: Path | None = None

    # ── Storage ───────────────────────────────────────────────────────────────
    enable_sqlite: bool = False
    sqlite_path: Path | None = None

    # ── Raiz do projeto (calculada em runtime) ────────────────────────────────
    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    # ── Resolução de caminhos derivados ───────────────────────────────────────
    def resolved_raw_dir(self) -> Path:
        return self.raw_dir or self.workspace_dir / "raw"

    def resolved_normalized_dir(self) -> Path:
        return self.normalized_dir or self.workspace_dir / "normalized"

    def resolved_bundles_dir(self) -> Path:
        return self.bundles_dir or self.workspace_dir / "bundles"

    def resolved_logs_dir(self) -> Path:
        return self.logs_dir or self.workspace_dir / "logs"

    def resolved_debug_dir(self) -> Path:
        return self.debug_dir or self.workspace_dir / "debug"

    def resolved_state_dir(self) -> Path:
        return self.state_dir or self.workspace_dir / "state"

    def resolved_zerohedge_state_path(self) -> Path:
        return self.zerohedge_state_path or self.resolved_state_dir() / "zerohedge" / "state.json"

    def resolved_zerohedge_profile_dir(self) -> Path:
        return self.zerohedge_profile_dir or self.resolved_state_dir() / "zerohedge" / "profile"

    def resolved_x_state_path(self) -> Path:
        return self.x_state_path or self.resolved_state_dir() / "x" / "state.json"

    def resolved_x_profile_dir(self) -> Path:
        return self.x_profile_dir or self.resolved_state_dir() / "x" / "profile"

    def resolved_sqlite_path(self) -> Path:
        return self.sqlite_path or self.workspace_dir / "agente.db"

    @field_validator("workspace_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: object) -> Path:
        return Path(str(v))


# Instância singleton — importar este objeto, não a classe.
settings = Settings()
