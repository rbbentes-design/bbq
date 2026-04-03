"""
MacroDesk Bloomberg Ecosystem — Desktop Launcher UI
=====================================================

Interface Tkinter simples para acionar o Bloomberg Agent com um clique.

Mostra:
  - Botão "Iniciar Bloomberg Agent"
  - Botão "Abrir MacroDesk" (quando disponível)
  - Status em tempo real
  - Contador de zips, CSVs e linhas ingeridas
  - Log scrollável da execução
  - Indicador de saúde do banco (última atualização)

Uso:
    python launcher/desktop_ui.py

Ou via atalho no desktop apontando para run_bloomberg_agent.bat
"""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import font, scrolledtext, ttk

# ── Garante root no sys.path ───────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Paleta de cores (tema escuro, alinhado ao MacroDesk) ──────────────────
COLORS = {
    "bg":           "#060a12",
    "bg_card":      "#0d1117",
    "bg_input":     "#161b22",
    "accent":       "#3b82f6",
    "accent_hover": "#2563eb",
    "success":      "#22c55e",
    "warning":      "#f59e0b",
    "error":        "#ef4444",
    "text":         "#e2e8f0",
    "text_dim":     "#64748b",
    "border":       "#1e293b",
}


class BloombergAgentUI:
    """Interface Tkinter para o Bloomberg Agent."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("MacroDesk — Bloomberg Agent")
        self.root.configure(bg=COLORS["bg"])
        self.root.resizable(True, True)
        self.root.minsize(540, 520)

        # Centralize na tela
        self.root.geometry("580x620")
        self._center_window()

        # Estado interno
        self._running = False
        self._agent_thread: threading.Thread | None = None

        self._build_ui()
        self._update_db_status()

    # ── Construção da UI ─────────────────────────────────────────────────

    def _build_ui(self) -> None:
        pad = {"padx": 20, "pady": 8}

        # ── Cabeçalho ─────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=COLORS["bg"], pady=16)
        header.pack(fill="x")

        tk.Label(
            header, text="MacroDesk Bloomberg Agent",
            bg=COLORS["bg"], fg=COLORS["text"],
            font=("Segoe UI", 15, "bold"),
        ).pack()

        tk.Label(
            header, text="Pipeline oficial de ingestão de dados Bloomberg",
            bg=COLORS["bg"], fg=COLORS["text_dim"],
            font=("Segoe UI", 9),
        ).pack()

        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=16)

        # ── Status do Banco ───────────────────────────────────────────────
        status_frame = tk.Frame(self.root, bg=COLORS["bg_card"], padx=16, pady=10)
        status_frame.pack(fill="x", padx=16, pady=(12, 4))

        tk.Label(
            status_frame, text="Status do Banco",
            bg=COLORS["bg_card"], fg=COLORS["text_dim"],
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w")

        self._db_status_label = tk.Label(
            status_frame, text="Verificando...",
            bg=COLORS["bg_card"], fg=COLORS["warning"],
            font=("Segoe UI", 9),
        )
        self._db_status_label.pack(anchor="w")

        # ── Métricas ──────────────────────────────────────────────────────
        metrics_frame = tk.Frame(self.root, bg=COLORS["bg"])
        metrics_frame.pack(fill="x", padx=16, pady=4)

        self._m_zips  = self._metric_card(metrics_frame, "Zips encontrados", "—")
        self._m_csvs  = self._metric_card(metrics_frame, "CSVs extraídos",   "—")
        self._m_rows  = self._metric_card(metrics_frame, "Linhas ingeridas",  "—")
        self._m_errs  = self._metric_card(metrics_frame, "Erros",             "—")

        for w in [self._m_zips, self._m_csvs, self._m_rows, self._m_errs]:
            w.pack(side="left", expand=True, fill="x", padx=4)

        # ── Botão Principal ───────────────────────────────────────────────
        btn_frame = tk.Frame(self.root, bg=COLORS["bg"])
        btn_frame.pack(fill="x", padx=16, pady=8)

        self._btn_start = tk.Button(
            btn_frame,
            text="▶  Iniciar Bloomberg Agent",
            command=self._start_agent,
            bg=COLORS["accent"], fg="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat", cursor="hand2",
            padx=20, pady=10,
            activebackground=COLORS["accent_hover"],
            activeforeground="white",
        )
        self._btn_start.pack(side="left", expand=True, fill="x", padx=(0, 6))

        self._btn_open = tk.Button(
            btn_frame,
            text="🌐 MacroDesk",
            command=self._open_macrodesk,
            bg=COLORS["bg_input"], fg=COLORS["text_dim"],
            font=("Segoe UI", 10),
            relief="flat", cursor="hand2",
            padx=12, pady=10,
            state="disabled",
        )
        self._btn_open.pack(side="right")

        # ── Status atual ──────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Pronto. Clique em Iniciar para processar novos dados Bloomberg.")
        status_label = tk.Label(
            self.root, textvariable=self._status_var,
            bg=COLORS["bg"], fg=COLORS["text_dim"],
            font=("Segoe UI", 9),
            wraplength=520, justify="left",
        )
        status_label.pack(padx=20, anchor="w")
        self._status_widget = status_label

        # ── Log scrollável ────────────────────────────────────────────────
        log_frame = tk.Frame(self.root, bg=COLORS["bg"])
        log_frame.pack(fill="both", expand=True, padx=16, pady=(4, 16))

        tk.Label(
            log_frame, text="Log de execução",
            bg=COLORS["bg"], fg=COLORS["text_dim"],
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w")

        self._log_text = scrolledtext.ScrolledText(
            log_frame,
            bg=COLORS["bg_card"], fg=COLORS["text"],
            font=("Consolas", 8),
            relief="flat",
            state="disabled",
            height=10,
            wrap="word",
            insertbackground=COLORS["text"],
        )
        self._log_text.pack(fill="both", expand=True)

        # Tags de cor no log
        self._log_text.tag_config("ok",      foreground=COLORS["success"])
        self._log_text.tag_config("warn",    foreground=COLORS["warning"])
        self._log_text.tag_config("err",     foreground=COLORS["error"])
        self._log_text.tag_config("dim",     foreground=COLORS["text_dim"])

    def _metric_card(self, parent: tk.Widget, label: str, value: str) -> tk.Frame:
        """Cria um card de métrica com label e valor."""
        card = tk.Frame(parent, bg=COLORS["bg_card"], padx=12, pady=8)
        tk.Label(card, text=label, bg=COLORS["bg_card"], fg=COLORS["text_dim"],
                 font=("Segoe UI", 7)).pack(anchor="w")
        val_label = tk.Label(card, text=value, bg=COLORS["bg_card"], fg=COLORS["text"],
                             font=("Segoe UI", 13, "bold"))
        val_label.pack(anchor="w")
        card._val_label = val_label   # type: ignore[attr-defined]
        return card

    def _set_metric(self, card: tk.Frame, value: str, color: str | None = None) -> None:
        card._val_label.config(  # type: ignore[attr-defined]
            text=value,
            fg=color or COLORS["text"],
        )

    # ── Lógica Principal ─────────────────────────────────────────────────

    def _start_agent(self) -> None:
        if self._running:
            return

        self._running = True
        self._btn_start.config(text="⏳ Executando...", state="disabled", bg=COLORS["bg_input"])
        self._clear_metrics()
        self._log_clear()
        self._set_status("Agente iniciado...", COLORS["text"])

        def _run() -> None:
            try:
                from core.bloomberg_main_agent import BloombergMainAgent
                agent = BloombergMainAgent()
                result = agent.run(on_progress=self._on_progress)

                # Atualiza métricas finais na UI thread
                self.root.after(0, lambda: self._on_agent_done(result))

            except Exception as exc:
                self.root.after(0, lambda: self._on_agent_error(str(exc)))

        self._agent_thread = threading.Thread(target=_run, daemon=True)
        self._agent_thread.start()

    def _on_progress(self, msg: str) -> None:
        """Chamado pelo agente a cada etapa — roda em thread do agente."""
        self.root.after(0, lambda m=msg: self._log_append(m))
        self.root.after(0, lambda m=msg: self._set_status(m, COLORS["text_dim"]))

    def _on_agent_done(self, result: Any) -> None:
        """Chamado na UI thread quando o agente termina."""
        self._running = False

        # Métricas
        self._set_metric(self._m_zips, str(result.zips_found))
        self._set_metric(self._m_csvs, str(result.csvs_extracted))
        self._set_metric(
            self._m_rows, str(result.rows_ingested),
            COLORS["success"] if result.rows_ingested > 0 else COLORS["text_dim"],
        )
        self._set_metric(
            self._m_errs, str(result.errors),
            COLORS["error"] if result.errors > 0 else COLORS["success"],
        )

        # Status
        status_map = {
            "ok":           ("Banco de dados atualizado com sucesso.",         COLORS["success"], "ok"),
            "no_new_data":  ("Nenhum novo arquivo Bloomberg encontrado.",      COLORS["warning"], "warn"),
            "partial":      ("Atualização parcial. Verifique o log de erros.", COLORS["warning"], "warn"),
            "error":        ("Falha na ingestão. Verifique o log.",            COLORS["error"],   "err"),
        }
        msg, color, tag = status_map.get(
            result.status, ("Ingestão finalizada.", COLORS["text"], "dim")
        )
        self._set_status(msg, color)
        self._log_append(f"\n{'='*40}", "dim")
        self._log_append(f"STATUS FINAL: {result.status.upper()}", tag)
        self._log_append(
            f"Zips: {result.zips_processed}/{result.zips_found} | "
            f"CSVs: {result.csvs_extracted} | "
            f"Linhas: {result.rows_ingested} | "
            f"Erros: {result.errors}",
            "dim",
        )

        # Reabilita botão
        self._btn_start.config(
            text="▶  Iniciar Bloomberg Agent",
            state="normal",
            bg=COLORS["accent"],
        )

        # Habilita botão MacroDesk se banco ok
        self._update_db_status()

    def _on_agent_error(self, error: str) -> None:
        self._running = False
        self._set_status(f"Erro fatal: {error}", COLORS["error"])
        self._log_append(f"ERRO FATAL: {error}", "err")
        self._btn_start.config(
            text="▶  Iniciar Bloomberg Agent",
            state="normal",
            bg=COLORS["accent"],
        )

    def _open_macrodesk(self) -> None:
        """Abre o HTML do MacroDesk mais recente no navegador."""
        import webbrowser
        try:
            from config.settings import ROOT
            # Procura HTML mais recente
            patterns = ["*_macro_desk_v2.html", "*_macro_desk*.html", "macro_desk*.html"]
            for pat in patterns:
                found = sorted(ROOT.glob(f"**/{pat}"), key=lambda p: p.stat().st_mtime, reverse=True)
                if found:
                    webbrowser.open(found[0].as_uri())
                    self._log_append(f"MacroDesk aberto: {found[0].name}", "ok")
                    return
            self._set_status("Nenhum arquivo MacroDesk encontrado. Rode 'agente run' primeiro.", COLORS["warning"])
        except Exception as exc:
            self._set_status(f"Erro ao abrir MacroDesk: {exc}", COLORS["error"])

    # ── Helpers de UI ─────────────────────────────────────────────────────

    def _update_db_status(self) -> None:
        """Atualiza label de status do banco e agendda próxima verificação."""
        try:
            from app.query_layer import BloombergQueryLayer
            ql = BloombergQueryLayer()
            status = ql.get_last_ingestion_status()

            if not status:
                self._db_status_label.config(
                    text="Banco não inicializado. Execute o agente para criar.",
                    fg=COLORS["warning"],
                )
                self._btn_open.config(state="disabled", fg=COLORS["text_dim"])
            else:
                age = status.get("age_minutes")
                rows = status.get("rows_ingested", 0)
                st   = status.get("status", "")
                age_str = f"{age:.0f} min atrás" if age is not None else "?"
                color   = COLORS["success"] if st == "ok" else COLORS["warning"]
                self._db_status_label.config(
                    text=f"{st.upper()} — {rows:,} linhas — atualizado {age_str}",
                    fg=color,
                )
                if st in ("ok", "partial"):
                    self._btn_open.config(state="normal", fg=COLORS["text"])
        except Exception:
            self._db_status_label.config(
                text="Banco não encontrado.",
                fg=COLORS["text_dim"],
            )

        # Reagenda verificação a cada 30 segundos
        self.root.after(30_000, self._update_db_status)

    def _set_status(self, msg: str, color: str) -> None:
        self._status_var.set(msg)
        self._status_widget.config(fg=color)

    def _log_append(self, msg: str, tag: str = "") -> None:
        self._log_text.config(state="normal")
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self._log_text.insert("end", line, tag or "")
        self._log_text.see("end")
        self._log_text.config(state="disabled")

    def _log_clear(self) -> None:
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")

    def _clear_metrics(self) -> None:
        for card in [self._m_zips, self._m_csvs, self._m_rows, self._m_errs]:
            self._set_metric(card, "—")

    def _center_window(self) -> None:
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"+{x}+{y}")

    # ── Run ───────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.root.mainloop()


# ── Type alias para lambda no _on_agent_done ──────────────────────────────
from typing import Any


def main() -> None:
    app = BloombergAgentUI()
    app.run()


if __name__ == "__main__":
    main()
