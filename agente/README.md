# Agente Editorial — Etapa 1

Pipeline de ingestão e curadoria de conteúdo editorial pessoal.

Coleta automaticamente posts do **The Market Ear (ZeroHedge)** e tweets do **X (Twitter)** via Playwright, normaliza os dados em modelos estruturados, gera relatórios em Markdown e JSON, e persiste tudo em workspace local.

---

## Visão Geral

```
ZeroHedge Market Ear ──┐
                        ├─► Pipeline ─► Bundle ─► Relatório (MD + JSON)
X Timeline ────────────┘
```

### O que é coletado

| Fonte | Dados | Volume típico |
|---|---|---|
| ZeroHedge / The Market Ear | Título + corpo + imagens de cada bloco editorial | ~15 blocos/dia |
| X Timeline | Autor, texto, URL, engajamento (likes/reposts/replies) | até 50 tweets/run |

---

## Estrutura do Projeto

```
agente/
├── app/
│   ├── audit/          # AuditLogger (JSONL) + factories de AuditRecord
│   ├── auth/           # Sessão Playwright + bootstrap de login interativo
│   ├── cli/            # Comandos Typer: auth, run, report
│   ├── config/         # Settings via Pydantic (lê .env)
│   ├── models/         # Pydantic: MarketEarBlock, XTimelineItem, DailyIngestionBundle
│   ├── pipeline/       # Orquestrador de ingestão (ingestion.py)
│   ├── providers/      # Scrapers: zerohedge.py, x_timeline.py
│   ├── storage/        # RawStore, NormalizedStore, BundleStore, WorkspacePaths
│   ├── utils/          # hashing, timestamps (ULID), text helpers
│   └── views/          # Gerador de relatórios (Markdown + JSON)
├── tests/              # Suite pytest — 74 testes (utils, audit, models, storage, views)
└── pyproject.toml
```

---

## Requisitos

- Python 3.11+
- Playwright Chromium (instalado via `playwright install chromium`)
- Contas ativas no ZeroHedge e no X (login manual na primeira vez)

---

## Instalação

```bash
# 1. Criar e ativar venv
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# 2. Instalar dependências
pip install -e ".[dev]"

# 3. Instalar navegador Playwright
playwright install chromium

# 4. Configurar workspace (opcional — padrão: ~/agente-workspace)
echo "WORKSPACE_DIR=C:/data/agente" > .env
```

---

## Uso

### 1. Login (primeira vez)

Abre o Chromium com perfil persistente para você fazer login manualmente:

```bash
# Login no ZeroHedge
python -m app.cli.auth login zerohedge

# Login no X
python -m app.cli.auth login x

# Verificar sessões
python -m app.cli.auth check
```

O perfil do browser fica salvo em `~/agente-workspace/state/browser/`. Nas próximas execuções o login é automático.

### 2. Rodar o pipeline

```bash
# Com janela do browser visível
python -m app.cli.run

# Modo headless (sem janela)
python -m app.cli.run --headless
```

Saída de exemplo:

```
Iniciando pipeline de ingestao...
┌─────────────────────────────────────────────────┐
│ Bundle 2026-03-22 | run_id=01KMC193...           │
├──────────────────────┬──────────────────────────┤
│ ZeroHedge blocos     │ 15                       │
│ X tweets             │ 50                       │
│ Erros                │ 0                        │
└──────────────────────┴──────────────────────────┘
Pipeline concluido sem erros.
```

### 3. Ver relatório

```bash
# Relatório Markdown do último run
python -m app.cli.report show

# Relatório de um run específico
python -m app.cli.report show --run-id 01KMC19...

# Resumo JSON
python -m app.cli.report show --format json

# Listar todos os bundles salvos
python -m app.cli.report list
```

---

## Workspace de Dados

Todos os artefatos ficam fora do repositório git, em `~/agente-workspace/` (configurável via `WORKSPACE_DIR`):

```
~/agente-workspace/
├── raw/
│   ├── zerohedge/<run_id>.html     # HTML bruto
│   └── x/<run_id>.html
├── normalized/
│   ├── zerohedge/<run_id>.jsonl    # Blocos normalizados (1 JSON por linha)
│   └── x/<run_id>.jsonl
├── bundles/
│   └── 2026-03-22/
│       ├── <run_id>.json           # DailyIngestionBundle completo
│       ├── <run_id>_report.md      # Relatório Markdown
│       └── <run_id>_summary.json   # Resumo JSON
├── logs/
│   └── audit_2026-03-22.jsonl      # Log de auditoria
└── state/
    └── browser/                    # Perfil Playwright persistente
```

---

## Testes

```bash
# Rodar todos os testes
python -m pytest tests/ -v

# Com cobertura
python -m pytest tests/ --cov=app --cov-report=term-missing
```

74 testes cobrindo: `utils`, `audit`, `models`, `storage`, `views`. Providers e pipeline não são testados (dependem de Playwright/browser).

---

## Configuração (.env)

| Variável | Padrão | Descrição |
|---|---|---|
| `WORKSPACE_DIR` | `~/agente-workspace` | Raiz dos dados (fora do git) |
| `PLAYWRIGHT_HEADLESS` | `false` | Browser sem janela |
| `X_TIMELINE_LIMIT` | `50` | Máximo de tweets por run |
| `LOG_LEVEL` | `INFO` | Nível de log (DEBUG/INFO/WARNING) |
| `APP_ENV` | `development` | Ambiente |

---

## Modelos de Dados

### `DailyIngestionBundle`
Bundle completo de um run diário. Contém:
- `run_id` — ULID único do run
- `run_date` — data do run
- `market_ear_blocks` — lista de `MarketEarBlock`
- `x_items` — lista de `XTimelineItem`
- `audit_summary` — contagem de registros e erros
- `artifact_paths` — caminhos de todos os artefatos gerados

### `MarketEarBlock`
Bloco editorial do The Market Ear:
- `title`, `body_text`, `image_refs`
- `published_at`, `position_index`
- `source_url`, `raw_source_document_id`

### `XTimelineItem`
Tweet do timeline:
- `author` (@handle), `text`, `url`, `created_at`
- `engagement_info` (likes, reposts, replies)
- `media_refs`, `raw_source_document_id`

---

## Arquitetura

O pipeline segue o fluxo:

```
Playwright (browser)
    └─► Provider (collect_html + parse)
            └─► RawStore (HTML bruto)
            └─► NormalizedStore (JSONL)
            └─► DailyIngestionBundle
                    └─► BundleStore (JSON)
                    └─► views/report (MD + JSON)
                    └─► AuditLogger (JSONL)
```

Cada camada é independente — storage não conhece providers, views não conhecem storage.
