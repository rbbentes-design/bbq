"""
Options Store — Importa e persiste snapshots do Greeks Dashboard (BQuant).

Fluxo:
  BQuant → "📦 Export ZIP" → greeks_TICKER_YYYYMMDD_HHMM.zip
    ├── metrics.json
    └── jarvis.html

  agente options-import greeks_SPX_20260404_1430.zip
    → salva em workspace/options/YYYY-MM-DD/
        ├── ULID_metrics.json
        └── ULID_jarvis.html   (se presente)

  MacroDesk → load_latest_options() → OptionsSnapshot → render_options_tab()

Campos de metrics capturados do Greek Dashboard:
  gamma_flip, gex_net_bn, pc_ratio, iv_rv_pp, iv_30d, rv_30d,
  squeeze_score, tail_score, call_wall, put_wall, daily_move, fragility,
  delta_bn, vanna_bn, charm_bn, skew_25d, flow_score_total, vix,
  z_cta, z_dealer, z_volctrl, z_rp, z_leveraged, z_passive_etf, z_buyback, z_cot,
  w_cta, w_dealer, w_volctrl, w_rp, w_leveraged, w_passive_etf, w_buyback, w_cot,
  squeeze_components
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.audit.logger import get_logger
from app.storage.paths import workspace
from app.utils.timestamps import new_ulid, utcnow

_log = get_logger("providers.options_store")


# ── Modelo ────────────────────────────────────────────────────────────────────

@dataclass
class OptionsSnapshot:
    run_id: str
    ticker: str
    spot: float
    ts: str                             # timestamp do Greeks Dashboard (ex: "2026-04-04 14:30")
    imported_at: str                    # UTC ISO quando foi importado
    metrics: dict[str, Any] = field(default_factory=dict)
    has_jarvis_html: bool = False       # True se jarvis.html foi incluído no zip

    # ── Accessors de conveniência ─────────────────────────────────────────────

    @property
    def gamma_flip(self) -> float:
        return float(self.metrics.get("gamma_flip") or 0)

    @property
    def gex_net_bn(self) -> float:
        return float(self.metrics.get("gex_net_bn") or 0)

    @property
    def iv_30d(self) -> float:
        return float(self.metrics.get("iv_30d") or 0)

    @property
    def rv_30d(self) -> float:
        return float(self.metrics.get("rv_30d") or 0)

    @property
    def iv_rv_pp(self) -> float:
        return float(self.metrics.get("iv_rv_pp") or 0)

    @property
    def pc_ratio(self) -> float:
        return float(self.metrics.get("pc_ratio") or 0)

    @property
    def squeeze_score(self) -> float:
        v = self.metrics.get("squeeze_score") or 0
        return float(v) if isinstance(v, (int, float)) else 0.0

    @property
    def tail_score(self) -> float:
        return float(self.metrics.get("tail_score") or 0)

    @property
    def call_wall(self) -> float:
        return float(self.metrics.get("call_wall") or 0)

    @property
    def put_wall(self) -> float:
        return float(self.metrics.get("put_wall") or 0)

    @property
    def vix(self) -> float:
        return float(self.metrics.get("vix") or 0)

    @property
    def skew_25d(self) -> float:
        return float(self.metrics.get("skew_25d") or 0)

    @property
    def flow_score_total(self) -> float:
        return float(self.metrics.get("flow_score_total") or 0)

    @property
    def delta_bn(self) -> float:
        return float(self.metrics.get("delta_bn") or 0)

    @property
    def vanna_bn(self) -> float:
        return float(self.metrics.get("vanna_bn") or 0)

    @property
    def charm_bn(self) -> float:
        return float(self.metrics.get("charm_bn") or 0)

    @property
    def fragility(self) -> float:
        return float(self.metrics.get("fragility") or 0)

    @property
    def daily_move(self) -> float:
        """Movimento diário implícito esperado (vol pts ou %)."""
        return float(self.metrics.get("daily_move") or 0)

    @property
    def squeeze_components(self) -> dict[str, dict]:
        """
        Decomposição do squeeze_score em sub-componentes.
        Cada componente tem: label, value (str), score (float), max (float), desc.
        Componentes típicos: gex, pc_ratio, flip_proximity, vol_premium.
        """
        c = self.metrics.get("squeeze_components") or {}
        return c if isinstance(c, dict) else {}

    @property
    def z_scores(self) -> dict[str, float]:
        keys = ["z_cta", "z_dealer", "z_volctrl", "z_rp", "z_leveraged", "z_passive_etf", "z_buyback", "z_cot"]
        return {k.replace("z_", ""): float(self.metrics.get(k) or 0) for k in keys}

    @property
    def weights(self) -> dict[str, float]:
        keys = ["w_cta", "w_dealer", "w_volctrl", "w_rp", "w_leveraged", "w_passive_etf", "w_buyback", "w_cot"]
        return {k.replace("w_", ""): float(self.metrics.get(k) or 0) for k in keys}


# ── Store ─────────────────────────────────────────────────────────────────────

class OptionsStore:

    def import_from_zip(self, zip_path: str | Path) -> OptionsSnapshot:
        """
        Importa um ZIP gerado pelo botão 'Export ZIP' do Greeks Dashboard.
        Salva metrics.json + jarvis.html (se presente) em workspace/options/YYYY-MM-DD/.
        Retorna o OptionsSnapshot criado.
        """
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP não encontrado: {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()

            if "metrics.json" not in names:
                raise ValueError(f"ZIP sem metrics.json: {names}")

            payload = json.loads(zf.read("metrics.json").decode("utf-8"))
            jarvis_html = zf.read("jarvis.html").decode("utf-8") if "jarvis.html" in names else None

        run_id = new_ulid()
        snap = OptionsSnapshot(
            run_id=run_id,
            ticker=payload.get("ticker", ""),
            spot=float(payload.get("spot") or 0),
            ts=payload.get("ts", ""),
            imported_at=utcnow().isoformat(),
            metrics=payload.get("metrics", {}),
            has_jarvis_html=jarvis_html is not None,
        )

        # Persiste
        out_dir = self._day_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = out_dir / f"{run_id}_metrics.json"
        metrics_path.write_text(
            json.dumps({
                "run_id":       snap.run_id,
                "ticker":       snap.ticker,
                "spot":         snap.spot,
                "ts":           snap.ts,
                "imported_at":  snap.imported_at,
                "has_jarvis":   snap.has_jarvis_html,
                "metrics":      snap.metrics,
            }, indent=2, default=str),
            encoding="utf-8",
        )

        if jarvis_html:
            html_path = out_dir / f"{run_id}_jarvis.html"
            html_path.write_text(jarvis_html, encoding="utf-8")

        _log.info("options_imported",
                  ticker=snap.ticker, spot=snap.spot,
                  ts=snap.ts, run_id=run_id,
                  path=str(metrics_path))
        return snap

    def save(self, snap: OptionsSnapshot, out_dir: Path | None = None) -> Path:
        """Salva snapshot já construído (usado internamente)."""
        d = out_dir or self._day_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{snap.run_id}_metrics.json"
        path.write_text(
            json.dumps({
                "run_id":      snap.run_id,
                "ticker":      snap.ticker,
                "spot":        snap.spot,
                "ts":          snap.ts,
                "imported_at": snap.imported_at,
                "has_jarvis":  snap.has_jarvis_html,
                "metrics":     snap.metrics,
            }, indent=2, default=str),
            encoding="utf-8",
        )
        return path

    def load_latest(self, for_date: date | None = None) -> OptionsSnapshot | None:
        """
        Carrega o snapshot mais recente (por data de importação).
        Procura em workspace/options/**/*_metrics.json, ordenado por mtime desc.
        """
        base = workspace.options
        if not base.exists():
            return None

        candidates = sorted(base.rglob("*_metrics.json"), reverse=True)
        if for_date:
            date_str = for_date.isoformat()
            candidates = [p for p in candidates if date_str in str(p)]

        for path in candidates:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                snap = OptionsSnapshot(
                    run_id=data.get("run_id", ""),
                    ticker=data.get("ticker", ""),
                    spot=float(data.get("spot") or 0),
                    ts=data.get("ts", ""),
                    imported_at=data.get("imported_at", ""),
                    metrics=data.get("metrics", {}),
                    has_jarvis_html=data.get("has_jarvis", False),
                )
                return snap
            except Exception as exc:
                _log.warning("options_load_error", path=str(path), error=str(exc))
                continue
        return None

    def load_jarvis_html(self, snap: OptionsSnapshot) -> str | None:
        """Carrega o jarvis.html correspondente ao snapshot, se existir."""
        if not snap.has_jarvis_html:
            return None
        # Procura na mesma pasta do metrics.json
        base = workspace.options
        for path in base.rglob(f"{snap.run_id}_jarvis.html"):
            try:
                return path.read_text(encoding="utf-8")
            except Exception:
                return None
        return None

    def list_snapshots(self) -> list[OptionsSnapshot]:
        """Lista todos os snapshots importados, do mais recente ao mais antigo."""
        base = workspace.options
        if not base.exists():
            return []
        results = []
        for path in sorted(base.rglob("*_metrics.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                results.append(OptionsSnapshot(
                    run_id=data.get("run_id", ""),
                    ticker=data.get("ticker", ""),
                    spot=float(data.get("spot") or 0),
                    ts=data.get("ts", ""),
                    imported_at=data.get("imported_at", ""),
                    metrics=data.get("metrics", {}),
                    has_jarvis_html=data.get("has_jarvis", False),
                ))
            except Exception:
                continue
        return results

    def _day_dir(self) -> Path:
        return workspace.options / date.today().isoformat()


options_store = OptionsStore()
