"""
Portfolio Tracker — Trade Ledger

Rastreia o ciclo completo de cada posição:
  OPEN  → data/hora + preço de entrada + tamanho
  CLOSE → data/hora + preço de saída  + P&L realizado

Estrutura de arquivo:
  {workspace}/portfolio/active_portfolio.json   ← posições abertas
  {workspace}/portfolio/trade_log.json          ← ledger completo (open + closed)
  {workspace}/portfolio/history/{date}.json     ← arquivo diário

Campos de cada trade:
  trade_id        : UUID curto
  ticker          : str
  direction       : "long" | "short"
  status          : "open" | "closed"
  entry_date      : "2026-03-31"
  entry_time      : "10:16:28"
  entry_price     : float
  shares          : float
  target_usd      : float      ← alocação nominal
  exit_date       : str | None
  exit_time       : str | None
  exit_price      : float | None
  pnl_realized    : float | None   ← P&L após fechar
  pnl_unrealized  : float | None   ← mark-to-market enquanto aberto
  pnl_pct         : float | None   ← % sobre target_usd
  conviction      : str
  rationale       : list[str]
  asset_class     : str
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("analysis.portfolio_tracker")

_PORTFOLIO_DIR = Path.home() / "agente-workspace" / "portfolio"
_ACTIVE_FILE   = _PORTFOLIO_DIR / "active_portfolio.json"
_TRADE_LOG     = _PORTFOLIO_DIR / "trade_log.json"
_HISTORY_DIR   = _PORTFOLIO_DIR / "history"

_RF_DAILY = 0.053 / 252   # risk-free rate daily


def _ensure_dirs() -> None:
    _PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _new_trade_id() -> str:
    return str(uuid.uuid4())[:8].upper()


def _load_trade_log() -> list[dict]:
    if not _TRADE_LOG.exists():
        return []
    try:
        return json.loads(_TRADE_LOG.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_trade_log(trades: list[dict]) -> None:
    _TRADE_LOG.write_text(
        json.dumps(trades, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _load_active() -> dict | None:
    if not _ACTIVE_FILE.exists():
        return None
    try:
        return json.loads(_ACTIVE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_active(state: dict) -> None:
    _ACTIVE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── Open positions ─────────────────────────────────────────────────────────────

_BBG_SUFFIXES = (
    " US Equity", " US EQUITY", " Equity", " EQUITY",
    " Index", " INDEX", " Comdty", " COMDTY", " Curncy", " CURNCY",
)

def _norm(ticker: str) -> str:
    """Remove sufixos Bloomberg: 'AAPL US EQUITY' → 'AAPL'."""
    t = ticker.strip()
    for sfx in _BBG_SUFFIXES:
        if t.endswith(sfx):
            t = t[:-len(sfx)].strip()
            break
    return t.replace("/", "-")


def _fetch_live_price(ticker: str, market_prices: dict) -> float:
    """
    Busca o preço live do ticker no momento exato da abertura da posição.
    1. IBKR reqMktData snapshot (real-time, sem delay)
    2. yfinance fast_info (15min delay — fallback)
    3. market_prices do bundle (último recurso)
    """
    yf_ticker = _norm(ticker)

    # 1. IBKR
    try:
        from app.providers import ibkr
        if ibkr.is_available():
            raw = ibkr.snapshot([yf_ticker])
            d = raw.get(yf_ticker, {})
            price = float(d.get("last") or d.get("ask") or d.get("bid") or 0.0)
            if price > 0:
                _log.info("entry_price_live", ticker=ticker, price=price, source="ibkr")
                return price
    except Exception:
        pass

    # 2. yfinance (normalizado)
    try:
        import yfinance as yf
        fi = yf.Ticker(yf_ticker).fast_info
        price = float(getattr(fi, "last_price", None) or getattr(fi, "previous_close", None) or 0.0)
        if price > 0:
            _log.info("entry_price_live", ticker=ticker, price=price, source="yfinance")
            return price
    except Exception:
        pass

    # 3. bundle — tenta tanto com sufixo quanto sem
    price = float(
        (market_prices.get(ticker) or market_prices.get(yf_ticker) or {}).get("price") or 0.0
    )
    _log.info("entry_price_bundle", ticker=ticker, price=price, source="bundle")
    return price


def open_portfolio(
    portfolio,                       # PortfolioResult
    market_prices: dict[str, Any],
) -> dict:
    """
    Abre novas posições a partir de um PortfolioResult.
    Fecha posições antigas que não aparecem mais no novo portfolio.

    Returns: state dict
    """
    _ensure_dirs()
    now = datetime.now()
    today = str(date.today())

    trade_log = _load_trade_log()
    existing  = _load_active() or {}
    existing_open = {t["ticker"]: t for t in existing.get("trades", []) if t["status"] == "open"}

    new_tickers = {p.ticker for p in portfolio.positions}

    # ── Fecha posições que saíram do portfolio ─────────────────────────────
    closed_now: list[dict] = []
    new_directions = {p.ticker: p.direction for p in portfolio.positions}
    for ticker, old_trade in existing_open.items():
        direction_flipped = (
            ticker in new_directions and
            new_directions[ticker] != old_trade["direction"]
        )

        # Calcula P&L antes de decidir
        current_price = (market_prices.get(ticker) or {}).get("price") or old_trade["entry_price"]
        pnl = _calc_pnl(old_trade, current_price)
        target_abs = abs(old_trade["target_usd"]) if old_trade["target_usd"] else 1
        pnl_pct = pnl / target_abs

        # ── Regra de holding mínimo: não fecha no mesmo dia ──────────────
        # A menos que: direction flip | stop semântico (-8%) | take-profit (+15%)
        opened_today = old_trade.get("entry_date") == today
        hard_exit = (
            direction_flipped or
            pnl_pct < -0.08 or   # stop semântico
            pnl_pct > 0.15       # take-profit
        )
        if ticker not in new_tickers or direction_flipped:
            if opened_today and not hard_exit:
                # Mantém — posição aberta hoje, sem motivo forte para fechar
                _log.info("position_held_min_period",
                          ticker=ticker, entry_date=old_trade["entry_date"],
                          pnl_pct=f"{pnl_pct:+.1%}")
                continue

            # Raciocinio de saída
            if direction_flipped:
                why_closed = f"Direcao invertida: era {old_trade['direction']}, agora {new_directions[ticker]} — regime ou sinal mudou"
            elif pnl_pct < -0.08:
                why_closed = f"Stop-loss semantico: P&L = {pnl_pct:+.1%} — sinal deteriorou e ativo saiu do portfolio otimizado"
            elif pnl_pct > 0.15:
                why_closed = f"Take-profit via rebalanceo: P&L = {pnl_pct:+.1%} — ativo nao e mais a melhor oportunidade marginal"
            else:
                why_closed = f"Saiu do portfolio otimizado (P&L = {pnl_pct:+.1%}) — composite abaixo do threshold ou constraints de alocacao"

            old_trade["status"]       = "closed"
            old_trade["exit_date"]    = today
            old_trade["exit_time"]    = now.strftime("%H:%M:%S")
            old_trade["exit_price"]   = round(float(current_price), 4)
            old_trade["pnl_realized"] = round(pnl, 2)
            old_trade["pnl_pct"]      = round(pnl_pct, 6)
            old_trade["pnl_unrealized"] = None
            old_trade["why_closed"]   = why_closed
            closed_now.append(old_trade)
            _log.info("position_closed",
                      ticker=ticker,
                      pnl=round(pnl, 2),
                      pnl_pct=f"{pnl_pct:+.1%}",
                      entry=old_trade["entry_price"],
                      exit=current_price,
                      reason=why_closed)

    # ── Abre novas posições ────────────────────────────────────────────────
    opened_now: list[dict] = []
    for pos in portfolio.positions:
        # Mantém se já existe e está aberta — preserva entry_price original
        if pos.ticker in existing_open and pos.direction == existing_open[pos.ticker]["direction"]:
            existing_open[pos.ticker]["target_usd"] = round(pos.allocation_usd, 2)
            opened_now.append(existing_open[pos.ticker])
            continue

        # Preço live no momento exato da abertura
        entry_price = _fetch_live_price(pos.ticker, market_prices)
        shares = abs(pos.allocation_usd) / entry_price if entry_price > 0 else 0.0

        # Razao de entrada
        why_opened = (
            f"Selecionado por otimizador Max-Sharpe: "
            f"E[R]={pos.expected_return_ann:+.1%}, "
            f"Sharpe={pos.sharpe_implied:.2f}, "
            f"convicao={pos.conviction}. "
            + ('; '.join(pos.rationale[:2]) if pos.rationale else "")
        )

        trade: dict = {
            "trade_id":       _new_trade_id(),
            "ticker":         pos.ticker,
            "name":           pos.name,
            "direction":      pos.direction,
            "status":         "open",
            "entry_date":     today,
            "entry_time":     now.strftime("%H:%M:%S"),
            "entry_price":    round(float(entry_price), 4),
            "shares":         round(shares, 4),
            "target_usd":     round(abs(pos.allocation_usd), 2),
            "exit_date":      None,
            "exit_time":      None,
            "exit_price":     None,
            "pnl_realized":   None,
            "pnl_unrealized": 0.0,
            "pnl_pct":        0.0,
            "conviction":     pos.conviction,
            "asset_class":    pos.asset_class,
            "cluster_id":     pos.cluster_id,
            "expected_return_ann": pos.expected_return_ann,
            "rationale":      pos.rationale,
            "why_opened":     why_opened,
            "why_closed":     None,
        }
        opened_now.append(trade)
        _log.info("position_opened",
                  ticker=pos.ticker,
                  direction=pos.direction,
                  price=round(float(entry_price), 2),
                  shares=round(shares, 2),
                  usd=round(abs(pos.allocation_usd), 0))

    # ── Salva no trade log (closed + opened) ──────────────────────────────
    trade_log.extend(closed_now)
    # Atualiza entradas abertas existentes no log
    for t in opened_now:
        existing_ids = [i for i, x in enumerate(trade_log) if x["trade_id"] == t["trade_id"]]
        if existing_ids:
            trade_log[existing_ids[0]] = t
        else:
            trade_log.append(t)

    _save_trade_log(trade_log)

    # ── Salva estado ativo ─────────────────────────────────────────────────
    state = {
        "inception_date":   today,
        "last_updated":     now.strftime("%H:%M:%S"),
        "budget":           portfolio.budget,
        "regime_at_entry":  portfolio.regime_mode,
        "sharpe_at_entry":  portfolio.sharpe,
        "expected_return":  portfolio.expected_return_ann,
        "trades":           opened_now,
        "snapshots":        existing.get("snapshots", []),
    }
    _save_active(state)

    _log.info("portfolio_opened",
              opened=len(opened_now),
              closed=len(closed_now),
              budget=portfolio.budget)
    return state


def set_entry_price(ticker: str, price: float) -> bool:
    """
    Corrige o entry_price de uma posição aberta.
    Recalcula shares para manter o target_usd consistente.

    Uso: quando a call foi feita a um preço diferente do preço de mercado
    no momento em que o pipeline rodou.

    Returns True se encontrou e atualizou, False se ticker não estava aberto.
    """
    state = _load_active()
    if not state:
        return False

    updated = False
    for trade in state.get("trades", []):
        if trade["ticker"].upper() == ticker.upper() and trade["status"] == "open":
            old_price = trade["entry_price"]
            target_usd = trade.get("target_usd", 0.0)
            trade["entry_price"] = round(float(price), 4)
            if target_usd > 0 and price > 0:
                trade["shares"] = round(abs(target_usd) / price, 6)
            trade["entry_override"] = True
            trade["entry_override_note"] = f"Manual override: {old_price} → {price}"
            updated = True
            _log.info("entry_price_override",
                      ticker=ticker, old=old_price, new=price,
                      shares=trade["shares"])
            break

    if updated:
        _save_active(state)
        # Atualiza também o trade_log
        log = _load_trade_log()
        for t in log:
            if t["ticker"].upper() == ticker.upper() and t["status"] == "open":
                t["entry_price"] = round(float(price), 4)
                t["entry_override"] = True
                break
        _save_trade_log(log)

    return updated


def _calc_pnl(trade: dict, current_price: float) -> float:
    """P&L de uma posição (unrealized ou realized)."""
    entry = trade.get("entry_price", 0.0)
    shares = trade.get("shares", 0.0)
    direction = trade.get("direction", "long")
    if entry <= 0 or shares <= 0:
        return 0.0
    price_diff = current_price - entry
    return -shares * price_diff if direction == "short" else shares * price_diff


# ── Mark-to-market ──────────────────────────────────────────────────────────────

def update_snapshot(market_prices: dict[str, Any]) -> dict | None:
    """
    Atualiza P&L unrealized de todas as posições abertas.
    Adiciona snapshot ao histórico.
    """
    state = _load_active()
    if not state:
        return None

    now = datetime.now().strftime("%H:%M:%S")
    pnl_by_ticker: dict[str, float] = {}
    total_pnl = 0.0
    budget = state.get("budget", 100_000)

    for trade in state.get("trades", []):
        if trade["status"] != "open":
            continue
        ticker = trade["ticker"]
        current = (market_prices.get(ticker) or {}).get("price")
        if not current:
            continue

        pnl = _calc_pnl(trade, float(current))
        pnl = round(pnl, 2)
        trade["pnl_unrealized"] = pnl
        trade["pnl_pct"] = round(pnl / trade["target_usd"], 6) if trade["target_usd"] else 0
        pnl_by_ticker[ticker] = pnl
        total_pnl += pnl

    snapshot = {
        "timestamp":     now,
        "prices":        {t: (market_prices.get(t) or {}).get("price")
                          for t in pnl_by_ticker},
        "pnl_by_ticker": pnl_by_ticker,
        "total_pnl":     round(total_pnl, 2),
        "pnl_pct":       round(total_pnl / budget, 6),
    }

    state["snapshots"].append(snapshot)
    state["snapshots"] = state["snapshots"][-200:]
    state["last_updated"] = now
    _save_active(state)

    _log.info("portfolio_snapshot",
              total_pnl=snapshot["total_pnl"],
              pnl_pct=f"{snapshot['pnl_pct']:.2%}")
    return snapshot


# ── Close a specific position ──────────────────────────────────────────────────

def close_position(ticker: str, market_prices: dict[str, Any]) -> dict | None:
    """
    Fecha uma posição específica pelo ticker.
    Retorna o trade fechado ou None se não encontrado.
    """
    state = _load_active()
    if not state:
        return None

    now = datetime.now()
    trade_log = _load_trade_log()

    for trade in state["trades"]:
        if trade["ticker"] != ticker or trade["status"] != "open":
            continue

        current_price = (market_prices.get(ticker) or {}).get("price") or trade["entry_price"]
        pnl = _calc_pnl(trade, float(current_price))

        trade["status"]       = "closed"
        trade["exit_date"]    = str(date.today())
        trade["exit_time"]    = now.strftime("%H:%M:%S")
        trade["exit_price"]   = round(float(current_price), 4)
        trade["pnl_realized"] = round(pnl, 2)
        trade["pnl_pct"]      = round(pnl / trade["target_usd"], 6) if trade["target_usd"] else 0
        trade["pnl_unrealized"] = None

        # Atualiza trade log
        for i, t in enumerate(trade_log):
            if t["trade_id"] == trade["trade_id"]:
                trade_log[i] = trade
                break
        else:
            trade_log.append(trade)

        _save_trade_log(trade_log)

        # Remove das posições ativas
        state["trades"] = [t for t in state["trades"] if t["ticker"] != ticker]
        _save_active(state)

        _log.info("position_closed_manual",
                  ticker=ticker,
                  pnl=round(pnl, 2),
                  entry=trade["entry_price"],
                  exit=current_price)
        return trade

    return None


# ── Summary queries ────────────────────────────────────────────────────────────

def get_pnl_summary() -> dict | None:
    """Resumo P&L atual (posições abertas + histórico de snapshots)."""
    state = _load_active()
    if not state:
        return None

    snaps = state.get("snapshots", [])
    last = snaps[-1] if snaps else {}

    open_trades = [t for t in state.get("trades", []) if t["status"] == "open"]
    history = [{"timestamp": s["timestamp"], "pnl_pct": s["pnl_pct"]}
               for s in snaps[-60:]]

    # Realized P&L from closed trades (cumulative)
    trades = _load_trade_log()
    closed = [t for t in trades if t["status"] == "closed" and t.get("pnl_realized") is not None]
    realized_pnl = round(sum(t["pnl_realized"] for t in closed), 2)

    return {
        "total_pnl":       last.get("total_pnl", 0),
        "pnl_pct":         last.get("pnl_pct", 0),
        "pnl_by_ticker":   last.get("pnl_by_ticker", {}),
        "last_prices":     last.get("prices", {}),
        "inception_date":  state.get("inception_date"),
        "last_updated":    state.get("last_updated"),
        "regime_at_entry": state.get("regime_at_entry"),
        "positions":       open_trades,
        "history":         history,
        "realized_pnl":    realized_pnl,
    }


def get_trade_history() -> list[dict]:
    """Retorna todos os trades (open + closed) do ledger."""
    trades = _load_trade_log()
    # Ordena: abertos primeiro, depois closed por data desc
    open_t   = [t for t in trades if t["status"] == "open"]
    closed_t = sorted(
        [t for t in trades if t["status"] == "closed"],
        key=lambda x: (x.get("exit_date", ""), x.get("exit_time", "")),
        reverse=True,
    )
    return open_t + closed_t


def get_performance_summary() -> dict:
    """
    Resumo de performance histórica.
    Calcula: win rate, profit factor, avg P&L, total realized P&L.
    """
    trades = _load_trade_log()
    closed = [t for t in trades if t["status"] == "closed" and t.get("pnl_realized") is not None]

    if not closed:
        return {"total_trades": 0, "realized_pnl": 0, "win_rate": 0}

    wins   = [t for t in closed if t["pnl_realized"] > 0]
    losses = [t for t in closed if t["pnl_realized"] <= 0]
    realized_pnl = sum(t["pnl_realized"] for t in closed)
    gross_profit = sum(t["pnl_realized"] for t in wins)
    gross_loss   = abs(sum(t["pnl_realized"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "total_trades":   len(closed),
        "open_trades":    len([t for t in trades if t["status"] == "open"]),
        "win_trades":     len(wins),
        "loss_trades":    len(losses),
        "win_rate":       round(len(wins) / len(closed), 4),
        "realized_pnl":   round(realized_pnl, 2),
        "gross_profit":   round(gross_profit, 2),
        "gross_loss":     round(gross_loss, 2),
        "profit_factor":  round(profit_factor, 3),
        "avg_win":        round(gross_profit / len(wins), 2) if wins else 0,
        "avg_loss":       round(-gross_loss / len(losses), 2) if losses else 0,
        "best_trade":     max((t["pnl_realized"] for t in closed), default=0),
        "worst_trade":    min((t["pnl_realized"] for t in closed), default=0),
    }


def daily_archive() -> None:
    """Arquiva snapshot do dia no histórico."""
    state = _load_active()
    if not state:
        return
    _ensure_dirs()
    today = str(date.today())
    hist_path = _HISTORY_DIR / f"{today}.json"
    hist_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    _log.info("portfolio_archived", path=str(hist_path))


# ── CLI display helpers ────────────────────────────────────────────────────────

def print_open_positions(console=None) -> None:
    """Imprime tabela de posições abertas com P&L atual."""
    state = _load_active()
    if not state:
        if console:
            console.print("[dim]Sem portfolio ativo.[/dim]")
        return

    trades = [t for t in state.get("trades", []) if t["status"] == "open"]
    if not trades:
        if console:
            console.print("[dim]Sem posições abertas.[/dim]")
        return

    total_pnl = sum(t.get("pnl_unrealized", 0) or 0 for t in trades)
    budget    = state.get("budget", 100_000)

    if console:
        from rich.table import Table
        from rich.panel import Panel

        t_table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 1))
        t_table.add_column("ID", style="dim")
        t_table.add_column("Ticker")
        t_table.add_column("Dir.", justify="center")
        t_table.add_column("Entrada", justify="right")
        t_table.add_column("Data/Hora")
        t_table.add_column("Shares", justify="right")
        t_table.add_column("USD", justify="right")
        t_table.add_column("P&L Unreal.", justify="right")
        t_table.add_column("P&L%", justify="right")

        for tr in trades:
            pnl   = tr.get("pnl_unrealized", 0) or 0
            pnl_p = tr.get("pnl_pct", 0) or 0
            dir_c = "green" if tr["direction"] == "long" else "red"
            pnl_c = "green" if pnl >= 0 else "red"
            t_table.add_row(
                tr["trade_id"],
                f"[bold]{tr['ticker']}[/bold]",
                f"[{dir_c}]{'▲' if tr['direction']=='long' else '▼'}[/{dir_c}]",
                f"${tr['entry_price']:.2f}",
                f"{tr['entry_date']} {tr['entry_time']}",
                f"{tr['shares']:.2f}",
                f"${tr['target_usd']:,.0f}",
                f"[{pnl_c}]${pnl:+,.0f}[/{pnl_c}]",
                f"[{pnl_c}]{pnl_p:+.2%}[/{pnl_c}]",
            )

        pnl_color = "green" if total_pnl >= 0 else "red"
        console.print(Panel(
            t_table,
            title=f"[bold magenta]Posicoes Abertas[/bold magenta] — {len(trades)} posicoes | "
                  f"P&L: [{pnl_color}]${total_pnl:+,.0f} ({total_pnl/budget:+.2%})[/{pnl_color}]",
            border_style="magenta",
        ))
    else:
        print(f"{'ID':8} {'Ticker':12} {'Dir':6} {'Entrada':10} {'Data':10} {'Hora':8} {'USD':>10} {'P&L':>10} {'P&L%':>8}")
        print("-" * 80)
        for tr in trades:
            pnl = tr.get("pnl_unrealized", 0) or 0
            print(f"{tr['trade_id']:8} {tr['ticker']:12} {tr['direction']:6} "
                  f"${tr['entry_price']:.2f}  {tr['entry_date']} {tr['entry_time']} "
                  f"${tr['target_usd']:>9,.0f} ${pnl:>+9,.0f} {tr.get('pnl_pct',0):>+7.2%}")
        print("-" * 80)
        print(f"{'TOTAL':52} ${total_pnl:>+10,.0f} {total_pnl/budget:>+7.2%}")
