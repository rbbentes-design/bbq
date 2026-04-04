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


_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE if _ENV_FILE.exists() else None,
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

    # ── ZeroHedge ─────────────────────────────────────────────────────────────
    zerohedge_base_url: str = "https://www.zerohedge.com"
    zerohedge_market_ear_url: str = "https://www.zerohedge.com/the-market-ear"
    zerohedge_blocks_limit: int = 30
    zerohedge_state_path: Path | None = None
    zerohedge_profile_dir: Path | None = None

    # ── X ─────────────────────────────────────────────────────────────────────
    x_provider_mode: Literal["browser", "api"] = "browser"
    x_timeline_limit: int = 100
    x_state_path: Path | None = None
    x_profile_dir: Path | None = None

    # ── ElevenLabs TTS ────────────────────────────────────────────────────────
    elevenlabs_api_key: str = Field(default="", description="ElevenLabs API key")
    elevenlabs_voice_id: str = Field(default="XrMNSxvVxLkUlaSeEuLM", description="ElevenLabs voice ID")
    podcast_intro_path: Path = Field(
        default=Path.home() / "agente-workspace/podcast/começo.mpeg",
        description="Intro do podcast (mpeg/mp3)",
    )
    podcast_outro_path: Path = Field(
        default=Path.home() / "agente-workspace/podcast/final.mpeg",
        description="Outro do podcast (mpeg/mp3)",
    )

    # ── FRED ──────────────────────────────────────────────────────────────────
    fred_api_key: str = Field(default="", description="FRED API key")

    # ── Market Data (fallback chain) ──────────────────────────────────────────
    alpha_vantage_api_key: str = Field(default="", description="Alpha Vantage API key")
    twelve_data_api_key:   str = Field(default="", description="Twelve Data API key")
    finnhub_api_key:       str = Field(default="", description="Finnhub API key")

    # ── CME DataServices (OAuth) ──────────────────────────────────────────────
    cme_api_id:     str = Field(default="", description="CME OAuth API ID")
    cme_api_secret: str = Field(default="", description="CME OAuth secret")

    # ── Cboe All Access (OAuth) ───────────────────────────────────────────────
    cboe_api_key:    str = Field(default="", description="Cboe All Access client_id")
    cboe_api_secret: str = Field(default="", description="Cboe All Access client_secret")

    # ── Curation ──────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    rss_feeds: str = Field(default="", description="URLs de feeds RSS separadas por virgula")

    @property
    def resolved_anthropic_api_key(self) -> str:
        """Returns the API key, preferring the .env file over system env vars."""
        key = self.anthropic_api_key
        if key.startswith("sk-ant-"):
            return key
        # System env var is corrupt — read directly from .env file
        if _ENV_FILE.exists():
            for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val.startswith("sk-ant-"):
                        return val
        return key
    curation_enabled: bool = True
    curation_confidence_threshold: float = 0.65
    curation_max_evidence_iterations: int = 3

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
