"""
decision_engine.py — 0DTE Intraday Trading Decision Engine
Integrates with greeks_dashboard.py for ES/SPX options.

Architecture:
  Layer 1  — Market state (intraday features)
  Layer 2  — Feature engineering (0DTE specific)
  Layer 3  — Predictive model (GBM / CatBoost baseline)
  Layer 4  — Strategy selector (0DTE structures)
  Layer 5  — Risk engine (capital / margin / sizing)
  Layer 6  — IB Execution (paper + live)
  Layer 7  — Trade journal
  Layer 8  — Backtest / walk-forward
"""

# ─── imports ─────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import json
import uuid
import time
import logging
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
from scipy.stats import norm

# ── garantir que o diretório deste arquivo está no sys.path ─────────────────
import sys as _sys
import os as _os
_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _THIS_DIR not in _sys.path:
    _sys.path.insert(0, _THIS_DIR)

# reuse from main dashboard
try:
    from greeks_dashboard import (
        calculate_all_greeks,
        black_scholes_price_vec,
        fetch_options_chain,
        compute_walls,
        compute_strike_exposures,
        fetch_market_data,
        TRADING_DAYS,
        FUTURES_MULTIPLIER,
    )
    _DASHBOARD_AVAILABLE = True
except Exception as _de_import_err:
    _DASHBOARD_AVAILABLE = False
    TRADING_DAYS = 252
    FUTURES_MULTIPLIER = 50
    # stubs para rodar standalone sem o dashboard
    def calculate_all_greeks(*a, **kw): return {}
    def black_scholes_price_vec(*a, **kw): return np.zeros(1)
    def fetch_options_chain(*a, **kw): return None
    def compute_walls(*a, **kw): return None, None
    def compute_strike_exposures(*a, **kw): return None
    def fetch_market_data(*a, **kw): return {}

# ─── logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('decision_engine.log', mode='a', encoding='utf-8'),
    ]
)
log = logging.getLogger('DecisionEngine')

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 0 — CONSTANTS & ENUMS
# ══════════════════════════════════════════════════════════════════════════════

ES_POINT_VALUE    = 50.0    # USD per point (ES full contract)
MES_POINT_VALUE   = 5.0     # USD per point (MES micro contract)
SPX_OPTION_MULT   = 100     # SPX/ES option multiplier
MARKET_OPEN_ET    = (9, 30)
MARKET_CLOSE_ET   = (16, 0)
FLATTEN_BEFORE_MIN = 15     # minutes before close to flatten all
LAST_ENTRY_MIN    = 30      # minutes before close to block new entries

class Regime(str, Enum):
    DIRECTIONAL_LONG  = 'directional_long'
    DIRECTIONAL_SHORT = 'directional_short'
    NEUTRAL           = 'neutral'
    LONG_VOL          = 'long_vol'
    SHORT_VOL         = 'short_vol'
    NO_TRADE          = 'no_trade'

class StructureType(str, Enum):
    # Directional
    LONG_CALL          = 'long_call'
    LONG_CALL_SPREAD   = 'long_call_spread'
    SHORT_CALL_SPREAD  = 'short_call_spread'
    LONG_PUT           = 'long_put'
    LONG_PUT_SPREAD    = 'long_put_spread'
    SHORT_PUT_SPREAD   = 'short_put_spread'
    # Neutral
    IRON_CONDOR        = 'iron_condor'
    CALL_FLY           = 'call_fly'
    IRON_BUTTERFLY     = 'iron_butterfly'
    PUT_FLY            = 'put_fly'
    # Vol
    LONG_STRADDLE      = 'long_straddle'
    # ES/MES futures
    ES_LONG            = 'es_long'
    ES_SHORT           = 'es_short'
    MES_LONG           = 'mes_long'
    MES_SHORT          = 'mes_short'
    # Disabled (not 0DTE compatible)
    CALL_CALENDAR      = 'call_calendar'
    PUT_CALENDAR       = 'put_calendar'
    CALL_DIAGONAL      = 'call_diagonal'
    PUT_DIAGONAL       = 'put_diagonal'
    NONE               = 'none'

DISABLED_STRUCTURES = {
    StructureType.CALL_CALENDAR:  'Requer vencimentos diferentes — incompatível com política 0DTE',
    StructureType.PUT_CALENDAR:   'Requer vencimentos diferentes — incompatível com política 0DTE',
    StructureType.CALL_DIAGONAL:  'Requer vencimentos diferentes — incompatível com política 0DTE',
    StructureType.PUT_DIAGONAL:   'Requer vencimentos diferentes — incompatível com política 0DTE',
}

# Regime → eligible structures (ordered by preference)
REGIME_STRUCTURES: Dict[Regime, List[StructureType]] = {
    Regime.DIRECTIONAL_LONG:  [StructureType.LONG_CALL_SPREAD,  StructureType.LONG_CALL,         StructureType.SHORT_PUT_SPREAD, StructureType.ES_LONG,  StructureType.MES_LONG],
    Regime.DIRECTIONAL_SHORT: [StructureType.LONG_PUT_SPREAD,   StructureType.LONG_PUT,           StructureType.SHORT_CALL_SPREAD,StructureType.ES_SHORT, StructureType.MES_SHORT],
    Regime.NEUTRAL:           [StructureType.IRON_CONDOR,        StructureType.IRON_BUTTERFLY,     StructureType.CALL_FLY,         StructureType.PUT_FLY],
    Regime.LONG_VOL:          [StructureType.LONG_STRADDLE,      StructureType.LONG_CALL_SPREAD,   StructureType.LONG_PUT_SPREAD],
    Regime.SHORT_VOL:         [StructureType.IRON_CONDOR,        StructureType.SHORT_PUT_SPREAD,   StructureType.SHORT_CALL_SPREAD,StructureType.IRON_BUTTERFLY],
    Regime.NO_TRADE:          [],
}

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 0 — DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RiskConfig:
    """Configurable risk parameters — all percentages of capital."""
    max_risk_per_trade_pct:      float = 1.0    # % of capital
    max_daily_loss_pct:          float = 3.0    # % of capital
    max_total_exposure_pct:      float = 15.0   # % of capital
    max_margin_usage_pct:        float = 50.0   # % of buying power
    reserve_cash_pct:            float = 10.0   # % of capital always kept
    max_positions_open:          int   = 4
    max_exposure_per_structure_pct: float = 5.0
    max_exposure_per_expiry_pct: float  = 8.0
    max_contracts_per_trade:     int   = 10
    min_cash_buffer:             float = 5000.0  # USD always kept
    min_confidence_to_trade:     float = 0.55
    last_entry_minutes_before_close: int = LAST_ENTRY_MIN
    flatten_minutes_before_close: int   = FLATTEN_BEFORE_MIN
    paper_mode:                  bool  = True    # ALWAYS default True

@dataclass
class AccountState:
    """Real-time account state (from IBKR or manual input)."""
    net_liquidation:    float = 100_000.0
    available_cash:     float = 100_000.0
    buying_power:       float = 200_000.0
    available_margin:   float = 100_000.0
    unrealized_pnl:     float = 0.0
    realized_pnl_day:   float = 0.0
    margin_used:        float = 0.0
    risk_used_day:      float = 0.0   # $ already at risk today
    source:             str   = 'manual'  # 'ibkr' or 'manual'
    timestamp:          str   = field(default_factory=lambda: datetime.utcnow().isoformat())

@dataclass
class Leg:
    """Single option or futures leg."""
    instrument:  str          # 'option' or 'futures'
    side:        int          # +1 long, -1 short
    option_type: Optional[str] = None   # 'C' or 'P'
    strike:      Optional[float] = None
    expiry:      Optional[str]   = None
    dte:         Optional[int]   = None
    iv:          Optional[float] = None
    delta:       Optional[float] = None
    gamma:       Optional[float] = None
    vega:        Optional[float] = None
    theta:       Optional[float] = None
    vanna:       Optional[float] = None
    charm:       Optional[float] = None
    px:          Optional[float] = None
    multiplier:  float = 100.0

@dataclass
class Structure:
    """Complete multi-leg options structure or futures trade."""
    structure_type:  StructureType
    legs:            List[Leg]
    net_debit:       float = 0.0   # positive = paid, negative = received
    max_loss:        float = 0.0   # always positive
    max_gain:        float = 0.0   # positive = finite, inf = unlimited credit
    breakeven_lower: Optional[float] = None
    breakeven_upper: Optional[float] = None
    net_delta:       float = 0.0
    net_gamma:       float = 0.0
    net_vega:        float = 0.0
    net_theta:       float = 0.0
    net_vanna:       float = 0.0
    net_charm:       float = 0.0
    estimated_margin: float = 0.0

@dataclass
class TradeDecision:
    """Final structured decision output."""
    decision_id:     str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:       str = field(default_factory=lambda: datetime.utcnow().isoformat())
    action:          str = 'no_trade'        # 'buy','sell','no_trade'
    instrument:      str = 'ES'              # 'ES','MES','SPX_OPT'
    structure:       StructureType = StructureType.NONE
    regime:          Regime = Regime.NO_TRADE
    confidence:      float = 0.0
    regime_proba:    Dict[str, float] = field(default_factory=dict)
    entry_price:     Optional[float] = None
    stop_loss:       Optional[float] = None
    take_profit:     Optional[float] = None
    quantity:        int = 0
    expiry:          Optional[str] = None
    strikes:         Dict[str, float] = field(default_factory=dict)
    rationale:       str = ''
    risk_metrics:    Dict[str, Any] = field(default_factory=dict)
    flatten_time:    Optional[str] = None
    execution_ready: bool = False
    block_reason:    Optional[str] = None
    # Capital
    capital_available:      float = 0.0
    margin_available:       float = 0.0
    risk_budget_trade:      float = 0.0
    risk_budget_day_remaining: float = 0.0
    estimated_trade_cost:   float = 0.0
    estimated_max_loss:     float = 0.0
    estimated_margin_usage: float = 0.0
    allowed_size:           int = 0
    size_block_reason:      Optional[str] = None
    # Structure
    structure_obj:   Optional[Structure] = None

@dataclass
class TradeRecord:
    """Persistent trade log entry."""
    trade_id:        str
    decision_id:     str
    timestamp_open:  str
    timestamp_close: Optional[str]
    ticker:          str
    structure:       str
    strikes:         Dict[str, float]
    expiry:          str
    quantity:        int
    entry_price:     float
    exit_price:      Optional[float]
    stop_loss:       float
    take_profit:     float
    pnl_open:        float
    pnl_closed:      float
    exit_reason:     str   # 'stop','target','time_stop','flatten','signal_flip'
    initial_greeks:  Dict[str, float]
    regime:          str
    confidence:      float
    features_snapshot: Dict[str, float]
    order_id:        Optional[str]
    fill_price:      Optional[float]

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — MARKET STATE (intraday features from options chain)
# ══════════════════════════════════════════════════════════════════════════════

class MarketState:
    """Reads and consolidates intraday market data for 0DTE decision."""

    def __init__(self, ticker: str = 'SPX Index', spot: float = 5000.0,
                 df: Optional[pd.DataFrame] = None):
        self.ticker = ticker
        self.spot   = spot
        self.df     = df        # options chain DataFrame
        self.ts     = datetime.utcnow()
        self.features: Dict[str, float] = {}

    def compute(self) -> Dict[str, float]:
        """
        Compute all Layer-1 and Layer-2 features from options chain.
        Returns flat dict of features for ML model.
        """
        f = {}
        df = self.df
        spot = self.spot

        if df is None or df.empty:
            log.warning('MarketState.compute: no options chain data')
            return f

        is_call = df['Type'] == 'Call'
        is_put  = df['Type'] == 'Put'
        K       = df['Strike'].values
        iv      = df['IV'].values
        oi      = df['OI'].values
        tte     = df['Tte'].values  # years

        # ── ATM IV 0DTE ──────────────────────────────────────────────────────
        # Filter to 0DTE only (Tte < 1 trading day)
        zero_dte_mask = tte < (1.0 / TRADING_DAYS)
        if zero_dte_mask.sum() > 0:
            df0 = df[zero_dte_mask].copy()
            K0  = df0['Strike'].values
            iv0 = df0['IV'].values
            atm_idx = np.argmin(np.abs(K0 - spot))
            f['atm_iv_0dte']     = float(iv0[atm_idx]) if len(iv0) > 0 else np.nan
            # Skew 0DTE: 25Δ put IV - 25Δ call IV
            calls0 = df0[df0['Type'] == 'Call']
            puts0  = df0[df0['Type'] == 'Put']
            if len(calls0) > 0 and len(puts0) > 0:
                g0 = calculate_all_greeks(spot, df0['Strike'].values, df0['IV'].values,
                                          df0['Tte'].values, df0['Type'].values) if _DASHBOARD_AVAILABLE else {}
                if g0:
                    c_mask = df0['Type'].values == 'Call'
                    p_mask = df0['Type'].values == 'Put'
                    c_deltas = np.abs(g0['delta'][c_mask])
                    p_deltas = np.abs(g0['delta'][p_mask])
                    # find closest to 0.25 delta
                    if len(c_deltas) > 0:
                        c25_idx = np.argmin(np.abs(c_deltas - 0.25))
                        f['iv_25d_call_0dte'] = float(df0[c_mask]['IV'].values[c25_idx])
                    if len(p_deltas) > 0:
                        p25_idx = np.argmin(np.abs(p_deltas - 0.25))
                        f['iv_25d_put_0dte']  = float(df0[p_mask]['IV'].values[p25_idx])
                    f['skew_0dte'] = f.get('iv_25d_put_0dte', 0) - f.get('iv_25d_call_0dte', 0)
            # Curvature 0DTE: 10Δ put + 10Δ call - 2×ATM IV
            f['atm_iv_0dte_level'] = f.get('atm_iv_0dte', 0)
        else:
            f['atm_iv_0dte'] = float(iv[np.argmin(np.abs(K - spot))]) if len(iv) > 0 else np.nan

        # ── OI Walls ──────────────────────────────────────────────────────────
        if _DASHBOARD_AVAILABLE:
            try:
                greeks_all = calculate_all_greeks(spot, K, iv, tte, df['Type'].values)
                agg = compute_strike_exposures(df, greeks_all, spot)
                call_wall, put_wall = compute_walls(agg)
                f['call_wall']          = float(call_wall) if call_wall else spot * 1.02
                f['put_wall']           = float(put_wall)  if put_wall  else spot * 0.98
                f['dist_to_call_wall']  = (f['call_wall'] - spot) / spot
                f['dist_to_put_wall']   = (spot - f['put_wall'])  / spot
                f['wall_range']         = (f['call_wall'] - f['put_wall']) / spot
                # Net GEX
                f['net_gex'] = float((agg.get('Call_gamma', pd.Series([0])).sum()
                                      - agg.get('Put_gamma', pd.Series([0])).sum()))
                # Vanna and Charm aggregate
                f['net_vanna'] = float((agg.get('Call_vanna', pd.Series([0])).sum()
                                        - agg.get('Put_vanna', pd.Series([0])).sum())) if 'Call_vanna' in agg.columns else 0.0
                f['net_charm'] = float(agg.get('Call_charm', pd.Series([0])).sum()
                                       + agg.get('Put_charm', pd.Series([0])).sum()) if 'Call_charm' in agg.columns else 0.0
            except Exception as e:
                log.warning(f'Wall/GEX compute failed: {e}')

        # ── OI concentration ─────────────────────────────────────────────────
        oi_calls = np.where(is_call, oi, 0)
        oi_puts  = np.where(is_put,  oi, 0)
        total_oi = oi.sum()
        f['total_oi']        = float(total_oi)
        f['oi_call_pct']     = float(oi_calls.sum() / (total_oi + 1e-9))
        f['pc_oi_ratio']     = float(oi_puts.sum() / (oi_calls.sum() + 1e-9))
        # OI concentration at ATM (±2 strikes)
        atm_mask = np.abs(K - spot) <= spot * 0.01
        f['atm_oi_pct']      = float(oi[atm_mask].sum() / (total_oi + 1e-9))

        # ── Time features ─────────────────────────────────────────────────────
        now = datetime.now()
        open_time  = now.replace(hour=MARKET_OPEN_ET[0], minute=MARKET_OPEN_ET[1], second=0)
        close_time = now.replace(hour=MARKET_CLOSE_ET[0], minute=MARKET_CLOSE_ET[1], second=0)
        elapsed_min  = max(0, (now - open_time).total_seconds() / 60)
        remaining_min= max(0, (close_time - now).total_seconds() / 60)
        total_min    = (close_time - open_time).total_seconds() / 60
        f['time_of_day_pct']  = float(elapsed_min / total_min)
        f['minutes_to_close'] = float(remaining_min)
        f['minutes_elapsed']  = float(elapsed_min)
        # Theta compression: gamma and theta explode in last 2 hours of 0DTE
        f['theta_compression'] = float(np.exp(-remaining_min / 60.0))  # 0→1 as close approaches

        self.features = f
        return f

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — PREDICTIVE MODEL (GBM baseline, CatBoost if available)
# ══════════════════════════════════════════════════════════════════════════════

class DecisionModel:
    """
    Gradient Boosting classifier for regime prediction.
    Features: market state + external scores (flow, squeeze, tail risk).
    CatBoost preferred; sklearn GBM as fallback.
    Architecture is ready for neural network upgrade.
    """

    FEATURE_NAMES = [
        'atm_iv_0dte', 'skew_0dte', 'pc_oi_ratio', 'atm_oi_pct',
        'dist_to_call_wall', 'dist_to_put_wall', 'wall_range',
        'net_gex', 'net_vanna', 'net_charm',
        'time_of_day_pct', 'minutes_to_close', 'theta_compression',
        'flow_score', 'squeeze_score', 'tail_score',
        'iv_rv_spread', 'skew_level',
    ]

    def __init__(self):
        self.model       = None
        self.is_fitted   = False
        self.model_name  = 'untrained'
        self.feature_importance: Dict[str, float] = {}
        self.classes_    = list(Regime)

    def _build_features(self, state_features: Dict[str, float],
                        external: Dict[str, float]) -> np.ndarray:
        merged = {**state_features, **external}
        return np.array([merged.get(k, 0.0) for k in self.FEATURE_NAMES])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train on historical feature matrix X and regime labels y."""
        try:
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier(
                iterations=300, depth=5, learning_rate=0.05,
                eval_metric='Accuracy', verbose=0, random_seed=42)
            self.model_name = 'CatBoost'
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                random_state=42, subsample=0.8)
            self.model_name = 'GBM-sklearn'
        self.model.fit(X, y)
        self.is_fitted = True
        # Feature importance
        try:
            imp = (self.model.get_feature_importance()
                   if self.model_name == 'CatBoost'
                   else self.model.feature_importances_)
            self.feature_importance = dict(zip(self.FEATURE_NAMES, imp.tolist()))
        except Exception:
            pass
        log.info(f'DecisionModel fitted: {self.model_name}, samples={len(y)}')

    def predict(self, state_features: Dict[str, float],
                external: Dict[str, float]) -> Tuple[Regime, float, Dict[str, float]]:
        """
        Returns (regime, confidence, proba_dict).
        Falls back to rule-based heuristic if model not fitted.
        """
        if self.is_fitted:
            x = self._build_features(state_features, external).reshape(1, -1)
            try:
                proba = self.model.predict_proba(x)[0]
                idx   = int(np.argmax(proba))
                conf  = float(proba[idx])
                # Map index to Regime
                regime_list = [Regime.DIRECTIONAL_LONG, Regime.DIRECTIONAL_SHORT,
                               Regime.NEUTRAL, Regime.LONG_VOL, Regime.SHORT_VOL,
                               Regime.NO_TRADE]
                regime = regime_list[idx] if idx < len(regime_list) else Regime.NO_TRADE
                proba_dict = {r.value: float(p)
                              for r, p in zip(regime_list, proba[:len(regime_list)])}
                return regime, conf, proba_dict
            except Exception as e:
                log.warning(f'Model predict failed: {e}')

        # ── Rule-based heuristic (no trained model) ──────────────────────────
        return self._heuristic(state_features, external)

    def _heuristic(self, f: Dict, ext: Dict) -> Tuple[Regime, float, Dict]:
        """HAR + flow + GEX heuristic when model not trained."""
        flow   = ext.get('flow_score', 50)
        squeeze= ext.get('squeeze_score', 0)
        tail   = ext.get('tail_score', 0)
        iv_rv  = ext.get('iv_rv_spread', 0)   # IV - RV; positive = IV rich
        skew   = f.get('skew_0dte', 0)         # put - call; positive = put bid
        gex    = f.get('net_gex', 0)
        min_cl = f.get('minutes_to_close', 390)

        scores = {
            Regime.DIRECTIONAL_LONG:  0.0,
            Regime.DIRECTIONAL_SHORT: 0.0,
            Regime.NEUTRAL:           0.0,
            Regime.LONG_VOL:          0.0,
            Regime.SHORT_VOL:         0.0,
            Regime.NO_TRADE:          0.0,
        }

        # Flow score drives directional bias (>60 bullish, <40 bearish)
        if flow > 65:   scores[Regime.DIRECTIONAL_LONG]  += 0.3
        elif flow < 35: scores[Regime.DIRECTIONAL_SHORT] += 0.3

        # GEX: positive GEX → market stabilized → neutral / short vol
        if gex > 0:
            scores[Regime.NEUTRAL]    += 0.2
            scores[Regime.SHORT_VOL]  += 0.15
        else:
            scores[Regime.LONG_VOL]   += 0.2

        # IV-RV spread: IV rich → short vol
        if iv_rv > 3:   scores[Regime.SHORT_VOL]  += 0.25
        elif iv_rv < -3: scores[Regime.LONG_VOL]  += 0.25

        # Squeeze score: high → buy vol / cautious
        if squeeze > 70: scores[Regime.LONG_VOL]  += 0.2
        if tail > 70:    scores[Regime.NO_TRADE]   += 0.4

        # Too close to close: no trade
        if min_cl < LAST_ENTRY_MIN:
            scores[Regime.NO_TRADE] += 1.0

        # Normalize
        total = sum(scores.values()) or 1.0
        proba = {k.value: v / total for k, v in scores.items()}
        best  = max(scores, key=scores.get)
        conf  = scores[best] / total
        return best, conf, proba

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — STRATEGY SELECTOR (0DTE structure builder)
# ══════════════════════════════════════════════════════════════════════════════

class StrategySelector:
    """
    Selects and constructs the best 0DTE structure for a given regime.
    Reuses black_scholes_price_vec and calculate_all_greeks from dashboard.
    """

    def __init__(self, df: pd.DataFrame, spot: float, rfr: float = 0.05,
                 minutes_to_close: float = 390):
        self.df              = df
        self.spot            = spot
        self.rfr             = rfr
        self.minutes_to_close= minutes_to_close
        # 0DTE chain only
        self.df0 = df[df['Tte'] < 1.0 / TRADING_DAYS * 2].copy() if df is not None else pd.DataFrame()

    # ── Strike helpers ─────────────────────────────────────────────────────────
    def _atm(self) -> float:
        K = self.df0['Strike'].values
        return float(K[np.argmin(np.abs(K - self.spot))]) if len(K) > 0 else self.spot

    def _nearest_strike(self, target: float) -> float:
        K = self.df0['Strike'].values
        return float(K[np.argmin(np.abs(K - target))]) if len(K) > 0 else target

    def _delta_strike(self, target_delta: float, option_type: str = 'C') -> float:
        """Find strike with delta closest to target_delta."""
        sub = self.df0[self.df0['Type'] == ('Call' if option_type == 'C' else 'Put')]
        if sub.empty:
            return self._atm()
        g = calculate_all_greeks(self.spot, sub['Strike'].values, sub['IV'].values,
                                  sub['Tte'].values, sub['Type'].values, r=self.rfr) \
            if _DASHBOARD_AVAILABLE else {'delta': np.zeros(len(sub))}
        deltas = np.abs(g['delta'])
        idx = np.argmin(np.abs(deltas - abs(target_delta)))
        return float(sub['Strike'].values[idx])

    def _get_iv(self, strike: float, opt_type: str) -> float:
        sub = self.df0[(np.abs(self.df0['Strike'] - strike) < 1e-3) &
                       (self.df0['Type'] == ('Call' if opt_type == 'C' else 'Put'))]
        return float(sub['IV'].values[0]) if not sub.empty else 0.15

    def _get_tte(self) -> float:
        if self.df0.empty: return 0.5 / TRADING_DAYS
        return float(self.df0['Tte'].mean())

    def _price_leg(self, strike: float, opt_type: str, sign: int) -> Leg:
        iv  = self._get_iv(strike, opt_type)
        tte = self._get_tte()
        px  = float(black_scholes_price_vec(
            self.spot, np.array([strike]), np.array([iv]),
            np.array([tte]), np.array([opt_type]), r=self.rfr)[0]) \
            if _DASHBOARD_AVAILABLE else 0.0
        g = calculate_all_greeks(self.spot, np.array([strike]), np.array([iv]),
                                  np.array([tte]), np.array([opt_type]), r=self.rfr) \
            if _DASHBOARD_AVAILABLE else {}
        return Leg(
            instrument='option', side=sign,
            option_type=opt_type, strike=strike, iv=iv, tte=tte, px=px,
            delta=float(g.get('delta', [0])[0])  if g else 0.0,
            gamma=float(g.get('gamma', [0])[0])  if g else 0.0,
            vega =float(g.get('vega',  [0])[0])  if g else 0.0,
            theta=float(g.get('theta', [0])[0])  if g else 0.0,
            vanna=float(g.get('vanna', [0])[0])  if g else 0.0,
            charm=float(g.get('charm', [0])[0])  if g else 0.0,
            multiplier=SPX_OPTION_MULT)

    def _build_structure(self, legs: List[Leg], stype: StructureType) -> Structure:
        net_debit = sum(l.side * (l.px or 0) * l.multiplier for l in legs)
        net_delta = sum(l.side * (l.delta or 0) for l in legs)
        net_gamma = sum(l.side * (l.gamma or 0) for l in legs)
        net_vega  = sum(l.side * (l.vega or 0)  for l in legs)
        net_theta = sum(l.side * (l.theta or 0) for l in legs)
        net_vanna = sum(l.side * (l.vanna or 0) for l in legs)
        net_charm = sum(l.side * (l.charm or 0) for l in legs)
        # Max loss calculation
        max_loss = self._calc_max_loss(legs, stype, net_debit)
        max_gain = self._calc_max_gain(legs, stype, net_debit)
        margin   = abs(max_loss)
        return Structure(
            structure_type=stype, legs=legs,
            net_debit=net_debit, max_loss=max_loss, max_gain=max_gain,
            net_delta=net_delta, net_gamma=net_gamma, net_vega=net_vega,
            net_theta=net_theta, net_vanna=net_vanna, net_charm=net_charm,
            estimated_margin=margin)

    def _calc_max_loss(self, legs: List[Leg], stype: StructureType, net_debit: float) -> float:
        """Calculate defined max loss for structure (always finite for 0DTE)."""
        if stype in (StructureType.LONG_CALL, StructureType.LONG_PUT,
                     StructureType.LONG_STRADDLE):
            return abs(net_debit)  # max loss = premium paid
        if stype in (StructureType.LONG_CALL_SPREAD, StructureType.LONG_PUT_SPREAD):
            return abs(net_debit)
        if stype in (StructureType.SHORT_CALL_SPREAD, StructureType.SHORT_PUT_SPREAD):
            strikes = sorted([l.strike for l in legs if l.strike])
            width   = abs(strikes[-1] - strikes[0]) * SPX_OPTION_MULT if len(strikes) >= 2 else 500
            return width - abs(net_debit)
        if stype == StructureType.IRON_CONDOR:
            call_legs = [l for l in legs if l.option_type == 'C']
            put_legs  = [l for l in legs if l.option_type == 'P']
            call_width = abs(call_legs[0].strike - call_legs[1].strike) * SPX_OPTION_MULT if len(call_legs) == 2 else 500
            put_width  = abs(put_legs[0].strike  - put_legs[1].strike)  * SPX_OPTION_MULT if len(put_legs)  == 2 else 500
            return max(call_width, put_width) - abs(net_debit)
        if stype in (StructureType.CALL_FLY, StructureType.PUT_FLY,
                     StructureType.IRON_BUTTERFLY):
            return abs(net_debit)
        return abs(net_debit) * 2  # conservative fallback

    def _calc_max_gain(self, legs: List[Leg], stype: StructureType, net_debit: float) -> float:
        if stype in (StructureType.SHORT_CALL_SPREAD, StructureType.SHORT_PUT_SPREAD,
                     StructureType.IRON_CONDOR, StructureType.IRON_BUTTERFLY):
            return abs(net_debit)  # credit received
        if stype in (StructureType.LONG_CALL_SPREAD, StructureType.LONG_PUT_SPREAD):
            strikes = sorted([l.strike for l in legs if l.strike])
            width   = abs(strikes[-1] - strikes[0]) * SPX_OPTION_MULT if len(strikes) >= 2 else 500
            return width - abs(net_debit)
        return float('inf')  # long options: theoretically unlimited

    def build(self, stype: StructureType) -> Optional[Structure]:
        """Build the structure for the given type. Returns None if disabled."""
        if stype in DISABLED_STRUCTURES:
            log.info(f'Structure {stype} disabled: {DISABLED_STRUCTURES[stype]}')
            return None

        atm = self._atm()
        sd  = (self.spot * self._get_iv(atm, 'C') *
               np.sqrt(max(self.minutes_to_close / 390.0, 0.01) / TRADING_DAYS))
        otm_c1 = self._nearest_strike(self.spot + sd * 0.5)
        otm_c2 = self._nearest_strike(self.spot + sd * 1.0)
        otm_p1 = self._nearest_strike(self.spot - sd * 0.5)
        otm_p2 = self._nearest_strike(self.spot - sd * 1.0)

        try:
            if stype == StructureType.LONG_CALL:
                legs = [self._price_leg(atm, 'C', +1)]
            elif stype == StructureType.LONG_PUT:
                legs = [self._price_leg(atm, 'P', +1)]
            elif stype == StructureType.LONG_STRADDLE:
                legs = [self._price_leg(atm, 'C', +1), self._price_leg(atm, 'P', +1)]
            elif stype == StructureType.LONG_CALL_SPREAD:
                legs = [self._price_leg(atm,   'C', +1), self._price_leg(otm_c1, 'C', -1)]
            elif stype == StructureType.LONG_PUT_SPREAD:
                legs = [self._price_leg(atm,   'P', +1), self._price_leg(otm_p1, 'P', -1)]
            elif stype == StructureType.SHORT_CALL_SPREAD:
                legs = [self._price_leg(otm_c1,'C', -1), self._price_leg(otm_c2, 'C', +1)]
            elif stype == StructureType.SHORT_PUT_SPREAD:
                legs = [self._price_leg(otm_p1,'P', -1), self._price_leg(otm_p2, 'P', +1)]
            elif stype == StructureType.IRON_CONDOR:
                legs = [self._price_leg(otm_p1,'P', -1), self._price_leg(otm_p2,'P', +1),
                        self._price_leg(otm_c1,'C', -1), self._price_leg(otm_c2,'C', +1)]
            elif stype == StructureType.IRON_BUTTERFLY:
                legs = [self._price_leg(otm_p1,'P', +1), self._price_leg(atm,'P', -1),
                        self._price_leg(atm,'C', -1),    self._price_leg(otm_c1,'C', +1)]
            elif stype == StructureType.CALL_FLY:
                legs = [self._price_leg(atm,'C', +1),    self._price_leg(otm_c1,'C', -2),
                        self._price_leg(otm_c2,'C', +1)]
            elif stype == StructureType.PUT_FLY:
                legs = [self._price_leg(atm,'P', +1),    self._price_leg(otm_p1,'P', -2),
                        self._price_leg(otm_p2,'P', +1)]
            else:
                return None
            return self._build_structure(legs, stype)
        except Exception as e:
            log.error(f'build_structure {stype}: {e}')
            return None

    def select_best(self, regime: Regime, confidence: float,
                    risk_config: RiskConfig) -> Tuple[Optional[Structure], StructureType]:
        """Select and build the best structure for the regime."""
        candidates = REGIME_STRUCTURES.get(regime, [])
        # Filter: too close to close → only simple structures
        if self.minutes_to_close < 60:
            simple = {StructureType.LONG_CALL, StructureType.LONG_PUT,
                      StructureType.LONG_CALL_SPREAD, StructureType.LONG_PUT_SPREAD,
                      StructureType.ES_LONG, StructureType.ES_SHORT,
                      StructureType.MES_LONG, StructureType.MES_SHORT}
            candidates = [c for c in candidates if c in simple]
        # Penalize complex structures with low confidence
        if confidence < 0.65:
            complex_structs = {StructureType.IRON_CONDOR, StructureType.CALL_FLY,
                               StructureType.PUT_FLY, StructureType.IRON_BUTTERFLY}
            candidates = [c for c in candidates if c not in complex_structs]
        for stype in candidates:
            if stype in (StructureType.ES_LONG, StructureType.ES_SHORT,
                         StructureType.MES_LONG, StructureType.MES_SHORT):
                # Futures: always buildable
                return None, stype
            struct = self.build(stype)
            if struct and struct.max_loss > 0:
                return struct, stype
        return None, StructureType.NONE

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — RISK ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class RiskEngine:
    """
    Validates capital, margin, sizing.
    All limits derived from account state + risk config.
    Smallest of all constraints wins for sizing.
    """

    def __init__(self, config: RiskConfig, account: AccountState):
        self.cfg = config
        self.acc = account

    @property
    def capital(self) -> float:
        return self.acc.net_liquidation

    @property
    def risk_budget_trade(self) -> float:
        return self.capital * self.cfg.max_risk_per_trade_pct / 100.0

    @property
    def risk_budget_day(self) -> float:
        return self.capital * self.cfg.max_daily_loss_pct / 100.0

    @property
    def risk_budget_day_remaining(self) -> float:
        return max(0.0, self.risk_budget_day - abs(min(0.0, self.acc.realized_pnl_day)))

    @property
    def max_exposure(self) -> float:
        return self.capital * self.cfg.max_total_exposure_pct / 100.0

    def _check_hard_blocks(self, stop_defined: bool, minutes_to_close: float,
                           structure: Optional[Structure]) -> Optional[str]:
        """Returns block reason string or None."""
        if self.acc.available_cash < self.cfg.min_cash_buffer:
            return f'Cash below buffer: ${self.acc.available_cash:,.0f} < ${self.cfg.min_cash_buffer:,.0f}'
        if self.acc.available_margin <= 0:
            return 'No available margin'
        if self.risk_budget_day_remaining <= 0:
            return f'Daily risk budget exhausted: realized P&L = ${self.acc.realized_pnl_day:,.0f}'
        if not stop_defined:
            return 'Stop loss not defined — trade blocked'
        if minutes_to_close < self.cfg.last_entry_minutes_before_close:
            return f'Too close to close: {minutes_to_close:.0f} min remaining (limit={self.cfg.last_entry_minutes_before_close})'
        if structure and structure.max_loss <= 0:
            return 'Max loss undefined — trade blocked'
        return None

    def size(self, structure: Optional[Structure], stop_pts: float,
             instrument: str = 'option', minutes_to_close: float = 390) -> Dict:
        """
        Calculate allowed size (contracts).
        Returns sizing dict with all constraints.
        """
        max_loss_per_lot = 0.0
        margin_per_lot   = 0.0
        cost_per_lot     = 0.0

        if instrument in ('ES', 'MES'):
            pv = ES_POINT_VALUE if instrument == 'ES' else MES_POINT_VALUE
            max_loss_per_lot = abs(stop_pts) * pv
            margin_per_lot   = max_loss_per_lot * 2
            cost_per_lot     = max_loss_per_lot
        elif structure:
            max_loss_per_lot = abs(structure.max_loss)
            margin_per_lot   = abs(structure.estimated_margin)
            cost_per_lot     = abs(structure.net_debit) if structure.net_debit > 0 else margin_per_lot

        if max_loss_per_lot <= 0:
            return {'allowed_size': 0, 'block_reason': 'max_loss_per_lot = 0'}

        # Size constraints
        n_risk_trade = int(self.risk_budget_trade         / max_loss_per_lot)
        n_risk_day   = int(self.risk_budget_day_remaining  / max_loss_per_lot)
        n_margin     = int(self.acc.available_margin       / (margin_per_lot + 1e-9))
        n_cash       = int((self.acc.available_cash - self.cfg.min_cash_buffer) / (cost_per_lot + 1e-9))
        n_max        = self.cfg.max_contracts_per_trade

        final = max(0, min(n_risk_trade, n_risk_day, n_margin, n_cash, n_max))

        block = None
        if final == 0:
            if n_risk_day   == 0: block = 'Daily risk budget exhausted'
            elif n_risk_trade == 0: block = 'Risk per trade limit: insufficient capital'
            elif n_margin   == 0: block = 'Insufficient margin'
            elif n_cash     == 0: block = 'Insufficient cash'

        return {
            'allowed_size':         final,
            'n_by_risk_trade':      n_risk_trade,
            'n_by_risk_day':        n_risk_day,
            'n_by_margin':          n_margin,
            'n_by_cash':            n_cash,
            'n_by_max_config':      n_max,
            'max_loss_per_lot':     max_loss_per_lot,
            'margin_per_lot':       margin_per_lot,
            'cost_per_lot':         cost_per_lot,
            'estimated_trade_cost': final * cost_per_lot,
            'estimated_max_loss':   final * max_loss_per_lot,
            'estimated_margin':     final * margin_per_lot,
            'block_reason':         block,
            'capital':              self.capital,
            'risk_budget_trade':    self.risk_budget_trade,
            'risk_budget_day_remaining': self.risk_budget_day_remaining,
            'available_margin':     self.acc.available_margin,
            'available_cash':       self.acc.available_cash,
        }

    def validate(self, decision: TradeDecision, structure: Optional[Structure],
                 minutes_to_close: float) -> TradeDecision:
        """Enrich decision with risk metrics and execution_ready flag."""
        stop_defined = decision.stop_loss is not None

        # Hard block checks
        block = self._check_hard_blocks(stop_defined, minutes_to_close, structure)
        if block:
            decision.execution_ready = False
            decision.block_reason    = block
            decision.allowed_size    = 0
            decision.size_block_reason = block
            return decision

        # Sizing
        sz = self.size(structure, stop_pts=abs((decision.entry_price or self.acc.available_cash) -
                                               (decision.stop_loss or 0)),
                       instrument=decision.instrument, minutes_to_close=minutes_to_close)
        decision.allowed_size            = sz['allowed_size']
        decision.size_block_reason       = sz.get('block_reason')
        decision.estimated_trade_cost    = sz['estimated_trade_cost']
        decision.estimated_max_loss      = sz['estimated_max_loss']
        decision.estimated_margin_usage  = sz['estimated_margin']
        decision.capital_available       = sz['capital']
        decision.margin_available        = sz['available_margin']
        decision.risk_budget_trade       = sz['risk_budget_trade']
        decision.risk_budget_day_remaining = sz['risk_budget_day_remaining']

        decision.execution_ready = (sz['allowed_size'] > 0
                                    and not self.cfg.paper_mode
                                    and stop_defined)
        if self.cfg.paper_mode and sz['allowed_size'] > 0:
            decision.block_reason = None  # paper mode: allow but flag

        # Flatten time
        now = datetime.now()
        close = now.replace(hour=MARKET_CLOSE_ET[0], minute=MARKET_CLOSE_ET[1], second=0)
        flatten_dt = close - timedelta(minutes=self.cfg.flatten_minutes_before_close)
        decision.flatten_time = flatten_dt.strftime('%H:%M ET')

        return decision

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 6 — IB EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

class IBExecution:
    """
    Interactive Brokers execution layer via ib_insync.
    Paper mode by default — live requires explicit enable + kill switch.
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 7497,
                 client_id: int = 1, paper_mode: bool = True):
        self.host       = host
        self.port       = port
        self.client_id  = client_id
        self.paper_mode = paper_mode
        self.ib         = None
        self.connected  = False
        self.kill_switch= False
        self._orders: Dict[str, Any] = {}

    def connect(self) -> bool:
        try:
            from ib_insync import IB
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id, readonly=False)
            self.connected = True
            log.info(f'IB connected: {self.host}:{self.port} client={self.client_id}')
            return True
        except Exception as e:
            log.error(f'IB connect failed: {e}')
            self.connected = False
            return False

    def disconnect(self):
        if self.ib and self.connected:
            self.ib.disconnect()
            self.connected = False
            log.info('IB disconnected')

    def get_account_state(self) -> AccountState:
        """Fetch real-time account values from IB."""
        if not self.connected:
            return AccountState(source='manual')
        try:
            vals = {v.tag: v.value for v in self.ib.accountValues()
                    if v.currency == 'USD'}
            return AccountState(
                net_liquidation = float(vals.get('NetLiquidation',         100_000)),
                available_cash  = float(vals.get('AvailableFunds',         100_000)),
                buying_power    = float(vals.get('BuyingPower',            200_000)),
                available_margin= float(vals.get('ExcessLiquidity',        100_000)),
                unrealized_pnl  = float(vals.get('UnrealizedPnL',               0)),
                realized_pnl_day= float(vals.get('RealizedPnL',                 0)),
                margin_used     = float(vals.get('InitMarginReq',                0)),
                source='ibkr',
                timestamp=datetime.utcnow().isoformat())
        except Exception as e:
            log.error(f'get_account_state failed: {e}')
            return AccountState(source='manual')

    def _validate_live(self, decision: TradeDecision) -> Optional[str]:
        """Guard: all conditions for live execution."""
        if self.kill_switch:        return 'Kill switch active'
        if self.paper_mode:         return 'Paper mode — live disabled'
        if not self.connected:      return 'IB not connected'
        if not decision.stop_loss:  return 'Stop loss missing'
        if decision.quantity <= 0:  return 'Invalid quantity'
        if not decision.expiry:     return 'Expiry undefined'
        if not decision.execution_ready: return 'Risk engine blocked execution'
        return None

    def submit(self, decision: TradeDecision, live: bool = False) -> Dict:
        """Submit order. live=False → paper log only."""
        if live:
            block = self._validate_live(decision)
            if block:
                log.warning(f'LIVE BLOCKED: {block}')
                return {'status': 'blocked', 'reason': block}
        result = {
            'decision_id': decision.decision_id,
            'mode':        'live' if live else 'paper',
            'timestamp':   datetime.utcnow().isoformat(),
            'structure':   decision.structure.value,
            'quantity':    decision.quantity,
            'entry':       decision.entry_price,
            'stop':        decision.stop_loss,
            'target':      decision.take_profit,
            'status':      'paper_logged' if not live else 'pending',
            'orders':      [],
        }
        if live and self.connected:
            result.update(self._submit_live(decision))
        log.info(f"Order [{result['mode']}]: {result['structure']} x{result['quantity']} "
                 f"entry={result['entry']} stop={result['stop']}")
        return result

    def _submit_live(self, decision: TradeDecision) -> Dict:
        from ib_insync import Stock, Future, Option, MarketOrder, LimitOrder, StopOrder, BracketOrder
        orders = []
        try:
            qty = decision.quantity
            struct = decision.structure
            if struct in (StructureType.ES_LONG, StructureType.ES_SHORT):
                contract = Future('ES', '', 'CME')
                self.ib.qualifyContracts(contract)
                side = 'BUY' if struct == StructureType.ES_LONG else 'SELL'
                bracket = self.ib.bracketOrder(side, qty,
                    limitPrice=decision.entry_price,
                    takeProfitPrice=decision.take_profit,
                    stopLossPrice=decision.stop_loss)
                for o in bracket:
                    trade = self.ib.placeOrder(contract, o)
                    orders.append({'order_id': trade.order.orderId, 'status': 'submitted'})
            elif struct in (StructureType.MES_LONG, StructureType.MES_SHORT):
                contract = Future('MES', '', 'CME')
                self.ib.qualifyContracts(contract)
                side = 'BUY' if struct == StructureType.MES_LONG else 'SELL'
                bracket = self.ib.bracketOrder(side, qty,
                    limitPrice=decision.entry_price,
                    takeProfitPrice=decision.take_profit,
                    stopLossPrice=decision.stop_loss)
                for o in bracket:
                    trade = self.ib.placeOrder(contract, o)
                    orders.append({'order_id': trade.order.orderId, 'status': 'submitted'})
            else:
                # Options: submit each leg
                if decision.structure_obj:
                    for leg in decision.structure_obj.legs:
                        exp_str = (decision.expiry or '').replace('-', '')
                        contract = Option('SPX', exp_str,
                                          int(leg.strike or 0),
                                          leg.option_type or 'C', 'CBOE')
                        self.ib.qualifyContracts(contract)
                        side = 'BUY' if leg.side > 0 else 'SELL'
                        order = LimitOrder(side, abs(qty),
                                           lmtPrice=round(leg.px or 0, 2))
                        trade = self.ib.placeOrder(contract, order)
                        orders.append({'order_id': trade.order.orderId,
                                       'strike': leg.strike, 'type': leg.option_type})
            return {'status': 'submitted', 'orders': orders}
        except Exception as e:
            log.error(f'_submit_live failed: {e}')
            return {'status': 'error', 'error': str(e), 'orders': orders}

    def flatten_all(self, live: bool = False):
        """Emergency: flatten all positions before close."""
        log.warning('FLATTEN ALL called')
        if not live or not self.connected:
            log.info('FLATTEN ALL — paper mode: no orders sent')
            return
        try:
            positions = self.ib.positions()
            for pos in positions:
                if pos.position == 0: continue
                side = 'SELL' if pos.position > 0 else 'BUY'
                order_type = MarketOrder(side, abs(pos.position))
                trade = self.ib.placeOrder(pos.contract, order_type)
                log.info(f'Flatten: {pos.contract.symbol} {side} {abs(pos.position)} → order {trade.order.orderId}')
        except Exception as e:
            log.error(f'flatten_all failed: {e}')

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 7 — TRADE JOURNAL
# ══════════════════════════════════════════════════════════════════════════════

class TradeJournal:
    """Persistent trade log: JSON lines format."""

    def __init__(self, path: str = 'trade_journal.jsonl'):
        self.path   = path
        self._cache: List[TradeRecord] = []

    def record_open(self, decision: TradeDecision, features: Dict) -> TradeRecord:
        rec = TradeRecord(
            trade_id        = str(uuid.uuid4())[:8],
            decision_id     = decision.decision_id,
            timestamp_open  = datetime.utcnow().isoformat(),
            timestamp_close = None,
            ticker          = decision.instrument,
            structure       = decision.structure.value,
            strikes         = decision.strikes,
            expiry          = decision.expiry or '',
            quantity        = decision.quantity,
            entry_price     = decision.entry_price or 0.0,
            exit_price      = None,
            stop_loss       = decision.stop_loss or 0.0,
            take_profit     = decision.take_profit or 0.0,
            pnl_open        = 0.0,
            pnl_closed      = 0.0,
            exit_reason     = '',
            initial_greeks  = decision.risk_metrics,
            regime          = decision.regime.value,
            confidence      = decision.confidence,
            features_snapshot = features,
            order_id        = None,
            fill_price      = None,
        )
        self._cache.append(rec)
        self._write(rec)
        return rec

    def record_close(self, trade_id: str, exit_price: float, exit_reason: str):
        for rec in self._cache:
            if rec.trade_id == trade_id:
                rec.timestamp_close = datetime.utcnow().isoformat()
                rec.exit_price      = exit_price
                rec.pnl_closed      = (exit_price - rec.entry_price) * rec.quantity
                rec.exit_reason     = exit_reason
                self._write(rec)
                return
        log.warning(f'TradeJournal.record_close: trade_id {trade_id} not found')

    def _write(self, rec: TradeRecord):
        try:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(rec)) + '\n')
        except Exception as e:
            log.error(f'Journal write failed: {e}')

    def load(self) -> pd.DataFrame:
        rows = []
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in f:
                    rows.append(json.loads(line))
        except FileNotFoundError:
            pass
        return pd.DataFrame(rows)

    def daily_pnl(self) -> float:
        df = self.load()
        if df.empty or 'pnl_closed' not in df.columns: return 0.0
        today = date.today().isoformat()
        mask  = df['timestamp_open'].str.startswith(today)
        return float(df[mask]['pnl_closed'].sum())

    def summary(self) -> Dict:
        df = self.load()
        if df.empty: return {}
        closed = df[df['exit_reason'].notna() & (df['exit_reason'] != '')]
        if closed.empty: return {'total_trades': 0}
        pnl = closed['pnl_closed']
        return {
            'total_trades':   len(closed),
            'win_rate':       float((pnl > 0).mean()),
            'avg_pnl':        float(pnl.mean()),
            'total_pnl':      float(pnl.sum()),
            'max_win':        float(pnl.max()),
            'max_loss':       float(pnl.min()),
            'sharpe_approx':  float(pnl.mean() / (pnl.std() + 1e-9)),
            'by_regime':      closed.groupby('regime')['pnl_closed'].mean().to_dict(),
            'by_structure':   closed.groupby('structure')['pnl_closed'].mean().to_dict(),
            'by_exit_reason': closed.groupby('exit_reason')['pnl_closed'].mean().to_dict(),
        }

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 8 — BACKTEST (walk-forward)
# ══════════════════════════════════════════════════════════════════════════════

class Backtester:
    """
    Walk-forward backtest on historical feature + P&L data.
    No random split — always chronological.
    """

    def __init__(self, model: DecisionModel, risk_config: RiskConfig):
        self.model  = model
        self.cfg    = risk_config
        self.results: List[Dict] = []

    def run(self, feature_df: pd.DataFrame, pnl_col: str = 'pnl',
            train_window: int = 60, step: int = 5,
            slippage_pct: float = 0.05, commission: float = 1.5) -> pd.DataFrame:
        """
        Walk-forward: train on [t-train_window:t], predict [t:t+step].
        feature_df must have columns = FEATURE_NAMES + pnl_col, sorted by date.
        """
        rows = []
        n = len(feature_df)
        feat_cols = [c for c in self.model.FEATURE_NAMES if c in feature_df.columns]

        for start in range(train_window, n - step, step):
            train = feature_df.iloc[start - train_window: start]
            test  = feature_df.iloc[start: start + step]

            X_tr = train[feat_cols].values
            # Simulate regime labels from realized pnl (positive = long, negative = short)
            y_tr = np.where(train[pnl_col].values > 0,
                            Regime.DIRECTIONAL_LONG.value,
                            Regime.DIRECTIONAL_SHORT.value)
            try:
                self.model.fit(X_tr, y_tr)
            except Exception:
                continue

            for _, row in test.iterrows():
                feat_dict = row[feat_cols].to_dict()
                regime, conf, _ = self.model.predict(feat_dict, {})
                actual_pnl = float(row.get(pnl_col, 0))
                # Apply slippage + commission
                net_pnl = actual_pnl * (1 - slippage_pct) - commission
                rows.append({
                    'date':       row.name,
                    'regime':     regime.value,
                    'confidence': conf,
                    'pnl_gross':  actual_pnl,
                    'pnl_net':    net_pnl,
                    'hit':        (regime == Regime.DIRECTIONAL_LONG and actual_pnl > 0) or
                                  (regime == Regime.DIRECTIONAL_SHORT and actual_pnl < 0),
                })

        if not rows:
            return pd.DataFrame()

        df_res = pd.DataFrame(rows)
        # Aggregate
        total_pnl  = df_res['pnl_net'].sum()
        win_rate   = float(df_res['hit'].mean())
        cum        = df_res['pnl_net'].cumsum()
        drawdown   = float((cum - cum.cummax()).min())
        sharpe     = float(df_res['pnl_net'].mean() / (df_res['pnl_net'].std() + 1e-9) * np.sqrt(252))

        log.info(f'Backtest complete: trades={len(df_res)} pnl=${total_pnl:,.0f} '
                 f'win={win_rate*100:.1f}% sharpe={sharpe:.2f} dd=${drawdown:,.0f}')
        self.results = rows
        return df_res

    def report(self) -> Dict:
        if not self.results: return {}
        df = pd.DataFrame(self.results)
        return {
            'total_trades':  len(df),
            'win_rate':      float(df['hit'].mean()),
            'total_pnl_net': float(df['pnl_net'].sum()),
            'sharpe':        float(df['pnl_net'].mean() / (df['pnl_net'].std() + 1e-9) * np.sqrt(252)),
            'max_drawdown':  float((df['pnl_net'].cumsum() - df['pnl_net'].cumsum().cummax()).min()),
            'avg_confidence':float(df['confidence'].mean()),
            'by_regime':     df.groupby('regime')['pnl_net'].sum().to_dict(),
            'hit_by_regime': df.groupby('regime')['hit'].mean().to_dict(),
        }

# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — main entry point
# ══════════════════════════════════════════════════════════════════════════════

class DecisionOrchestrator:
    """
    Top-level: runs all layers and returns a TradeDecision.
    Designed to be called once per signal cycle (e.g. every 5 min).

    Usage:
        engine = DecisionOrchestrator(ticker='ES1 Index', spot=5200)
        engine.update_account(available_cash=100_000, net_liquidation=100_000)
        decision = engine.run(df_chain, external_scores)
        print(decision)
    """

    def __init__(self, ticker: str = 'ES1 Index', spot: float = 5000.0,
                 rfr: float = 0.05, risk_config: Optional[RiskConfig] = None,
                 account: Optional[AccountState] = None):
        self.ticker  = ticker
        self.spot    = spot
        self.rfr     = rfr
        self.cfg     = risk_config or RiskConfig()
        self.account = account or AccountState()
        self.model   = DecisionModel()
        self.journal = TradeJournal()
        self.ib      = IBExecution(paper_mode=self.cfg.paper_mode)
        self._last_decision: Optional[TradeDecision] = None

    def update_account(self, **kwargs):
        """Update account state from keyword args or IB."""
        if self.ib.connected:
            self.account = self.ib.get_account_state()
        else:
            for k, v in kwargs.items():
                if hasattr(self.account, k):
                    setattr(self.account, k, v)

    def run(self, df: pd.DataFrame,
            external_scores: Optional[Dict] = None) -> TradeDecision:
        """
        Full decision cycle.
        external_scores: dict with flow_score, squeeze_score, tail_score,
                         iv_rv_spread, skew_level from greeks_dashboard.
        """
        ext = external_scores or {}
        now = datetime.now()
        open_t = now.replace(hour=MARKET_OPEN_ET[0], minute=MARKET_OPEN_ET[1], second=0)
        close_t= now.replace(hour=MARKET_CLOSE_ET[0], minute=MARKET_CLOSE_ET[1], second=0)
        min_to_close = max(0, (close_t - now).total_seconds() / 60)

        # Layer 1+2: market state features
        ms = MarketState(self.ticker, self.spot, df)
        features = ms.compute()
        features['minutes_to_close'] = min_to_close

        # Layer 3: predict regime
        regime, confidence, proba = self.model.predict(features, ext)
        log.info(f'Regime: {regime.value} conf={confidence:.2%}')

        decision = TradeDecision(
            regime       = regime,
            confidence   = confidence,
            regime_proba = proba,
        )

        # No trade conditions
        if regime == Regime.NO_TRADE or confidence < self.cfg.min_confidence_to_trade:
            decision.action = 'no_trade'
            decision.block_reason = (f'Confidence {confidence:.2%} < '
                                     f'threshold {self.cfg.min_confidence_to_trade:.2%}'
                                     if regime != Regime.NO_TRADE
                                     else 'Regime: no_trade')
            self._last_decision = decision
            return decision

        # Layer 4: select structure
        sel = StrategySelector(df, self.spot, self.rfr, min_to_close)
        struct, stype = sel.select_best(regime, confidence, self.cfg)

        decision.structure    = stype
        decision.structure_obj= struct
        decision.instrument   = 'ES' if stype in (StructureType.ES_LONG, StructureType.ES_SHORT) else \
                                'MES' if stype in (StructureType.MES_LONG, StructureType.MES_SHORT) else \
                                'SPX_OPT'
        decision.action = ('buy' if regime in (Regime.DIRECTIONAL_LONG,  Regime.LONG_VOL)  else
                           'sell' if regime in (Regime.DIRECTIONAL_SHORT, Regime.SHORT_VOL) else
                           'buy')  # neutral → buy the spread

        # Entry, stop, target
        if struct:
            entry_px = abs(struct.net_debit) / SPX_OPTION_MULT
            stop_px  = entry_px * 0.5 if struct.net_debit > 0 else entry_px + struct.max_loss / SPX_OPTION_MULT
            target_px= entry_px * 2.0 if struct.net_debit > 0 else entry_px - struct.max_gain / SPX_OPTION_MULT
            decision.entry_price  = round(entry_px,  2)
            decision.stop_loss    = round(stop_px,   2)
            decision.take_profit  = round(target_px, 2)
            decision.strikes      = {f'leg{i+1}': l.strike for i, l in enumerate(struct.legs) if l.strike}
            decision.risk_metrics = {
                'net_delta': struct.net_delta, 'net_gamma': struct.net_gamma,
                'net_vega':  struct.net_vega,  'net_theta': struct.net_theta,
            }
            decision.expiry = df[df['Tte'] < 2.0 / TRADING_DAYS]['Exp'].min().strftime('%Y-%m-%d') \
                              if (not df.empty and 'Exp' in df.columns and
                                  not df[df['Tte'] < 2.0 / TRADING_DAYS].empty) else ''
        elif stype in (StructureType.ES_LONG, StructureType.ES_SHORT, StructureType.MES_LONG, StructureType.MES_SHORT):
            pv      = ES_POINT_VALUE if 'ES' in stype.value else MES_POINT_VALUE
            stop_pts= features.get('atm_iv_0dte', 0.01) * self.spot * 0.5
            decision.entry_price = round(self.spot, 2)
            decision.stop_loss   = round(self.spot - stop_pts if 'LONG' in stype.value.upper()
                                         else self.spot + stop_pts, 2)
            decision.take_profit = round(self.spot + stop_pts * 2 if 'LONG' in stype.value.upper()
                                         else self.spot - stop_pts * 2, 2)

        # Rationale
        decision.rationale = (
            f"Regime={regime.value} conf={confidence:.1%} | "
            f"Flow={ext.get('flow_score',0):.0f} Squeeze={ext.get('squeeze_score',0):.0f} "
            f"Tail={ext.get('tail_score',0):.0f} | "
            f"ATM_IV={features.get('atm_iv_0dte',0)*100:.1f}% "
            f"Skew={features.get('skew_0dte',0)*100:.1f}pp "
            f"GEX={features.get('net_gex',0):.1e} | "
            f"Walls: call={features.get('call_wall',0):.0f} put={features.get('put_wall',0):.0f}"
        )

        # Layer 5: risk validation
        risk = RiskEngine(self.cfg, self.account)
        decision.quantity = min(decision.allowed_size or 1,
                                risk.size(struct, stop_pts=abs((decision.entry_price or 0) -
                                                               (decision.stop_loss or 0)),
                                          instrument=decision.instrument,
                                          minutes_to_close=min_to_close).get('allowed_size', 0))
        decision = risk.validate(decision, struct, min_to_close)

        log.info(f"Decision [{decision.decision_id}]: {decision.action} {decision.structure.value} "
                 f"x{decision.quantity} entry={decision.entry_price} "
                 f"stop={decision.stop_loss} target={decision.take_profit} "
                 f"exec_ready={decision.execution_ready} paper={self.cfg.paper_mode}")

        self._last_decision = decision
        return decision

    def execute_paper(self) -> Dict:
        """Execute last decision in paper mode."""
        if not self._last_decision:
            return {'status': 'no_decision'}
        return self.ib.submit(self._last_decision, live=False)

    def execute_live(self) -> Dict:
        """Execute last decision live — requires kill_switch=False and all validations."""
        if not self._last_decision:
            return {'status': 'no_decision'}
        return self.ib.submit(self._last_decision, live=True)

    def flatten_all(self, live: bool = False):
        """Flatten all positions (end of day)."""
        self.ib.flatten_all(live=live)


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: build_decision_engine_tab (for greeks_dashboard.py Tab 14)
# ══════════════════════════════════════════════════════════════════════════════

def build_decision_engine_tab(df, spot, rfr, ticker, external_scores=None):
    """
    Builds the ipywidgets tab for the Decision Engine.
    Called from greeks_dashboard.py main assembly.
    Reuses existing df, spot, rfr from the dashboard run.
    """
    import ipywidgets as wd
    from IPython.display import display as _disp, HTML

    ext = external_scores or {}

    # ── Config widgets ──────────────────────────────────────────────────────
    w_cash     = wd.FloatText(value=100_000, description='Cash ($):',
                              layout=wd.Layout(width='200px'), style={'description_width': '80px'})
    w_nlv      = wd.FloatText(value=100_000, description='NLV ($):',
                              layout=wd.Layout(width='200px'), style={'description_width': '80px'})
    w_bp       = wd.FloatText(value=200_000, description='Buying Power:',
                              layout=wd.Layout(width='220px'), style={'description_width': '100px'})
    w_margin   = wd.FloatText(value=100_000, description='Avail Margin:',
                              layout=wd.Layout(width='220px'), style={'description_width': '100px'})
    w_rpnl     = wd.FloatText(value=0,       description='Real PnL $:',
                              layout=wd.Layout(width='200px'), style={'description_width': '80px'})
    w_risk_pct = wd.FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1,
                                description='Risk/Trade %:', readout_format='.1f',
                                layout=wd.Layout(width='300px'), style={'description_width': '100px'})
    w_daily_pct= wd.FloatSlider(value=3.0, min=0.5, max=10.0, step=0.5,
                                description='Daily Loss %:', readout_format='.1f',
                                layout=wd.Layout(width='300px'), style={'description_width': '100px'})
    w_maxpos   = wd.IntSlider(value=4, min=1, max=10, description='Max Positions:',
                              layout=wd.Layout(width='280px'), style={'description_width': '100px'})
    w_paper    = wd.ToggleButton(value=True, description='PAPER MODE ON',
                                 button_style='warning', icon='shield',
                                 layout=wd.Layout(width='160px', height='36px'))
    w_run      = wd.Button(description='▶ Gerar Decisão', button_style='primary',
                           layout=wd.Layout(width='160px', height='36px'))
    w_paper_ex = wd.Button(description='📋 Paper Execute', button_style='warning',
                           layout=wd.Layout(width='150px', height='36px'))
    w_live_ex  = wd.Button(description='⚡ Live Execute', button_style='danger',
                           layout=wd.Layout(width='140px', height='36px'))
    w_flatten  = wd.Button(description='🚨 Flatten All', button_style='danger',
                           layout=wd.Layout(width='140px', height='36px'))

    out_decision = wd.Output()
    out_journal  = wd.Output()

    # Shared orchestrator
    orchestrator = [None]

    def _make_orchestrator():
        cfg = RiskConfig(
            max_risk_per_trade_pct = w_risk_pct.value,
            max_daily_loss_pct     = w_daily_pct.value,
            max_positions_open     = w_maxpos.value,
            paper_mode             = w_paper.value,
        )
        acc = AccountState(
            net_liquidation  = w_nlv.value,
            available_cash   = w_cash.value,
            buying_power     = w_bp.value,
            available_margin = w_margin.value,
            realized_pnl_day = w_rpnl.value,
        )
        return DecisionOrchestrator(ticker=ticker, spot=spot, rfr=rfr,
                                    risk_config=cfg, account=acc)

    def _render_decision(d: TradeDecision):
        block_color  = '#ff4444' if d.block_reason else '#00ff99'
        exec_label   = '✓ Pronto para executar' if d.execution_ready else '✗ Bloqueado'
        action_color = ('#00ff99' if d.action == 'buy' else
                        '#ff4444' if d.action == 'sell' else '#aaaaaa')
        conf_bar_w = int(d.confidence * 180)
        regime_bars = ''.join(
            f"<div style='display:flex;align-items:center;margin:2px 0;'>"
            f"<span style='color:#aaa;font-size:9px;width:140px;'>{k}</span>"
            f"<div style='background:#00d4e8;height:8px;width:{int(v*120)}px;border-radius:2px;'></div>"
            f"<span style='color:#aaa;font-size:9px;margin-left:4px;'>{v*100:.0f}%</span></div>"
            for k, v in sorted(d.regime_proba.items(), key=lambda x: -x[1]))
        strikes_html = ''.join(f"<b>{k}</b>: {v:.0f} &nbsp;" for k, v in d.strikes.items())
        html = f"""
<div style='background:#0d1520;border:1px solid rgba(0,212,232,.25);border-radius:8px;
            padding:16px;font-family:monospace;margin:8px 0;'>
  <div style='display:flex;gap:20px;flex-wrap:wrap;margin-bottom:12px;'>
    <!-- Action card -->
    <div style='background:#1a2035;border-left:4px solid {action_color};
                padding:12px 20px;border-radius:6px;min-width:180px;'>
      <div style='color:#aaa;font-size:9px;letter-spacing:.8px;'>DECISÃO</div>
      <div style='color:{action_color};font-size:22px;font-weight:bold;text-transform:uppercase;'>{d.action}</div>
      <div style='color:#aaa;font-size:10px;'>{d.instrument} · {d.structure.value}</div>
    </div>
    <!-- Confidence -->
    <div style='background:#1a2035;padding:12px 20px;border-radius:6px;min-width:180px;'>
      <div style='color:#aaa;font-size:9px;letter-spacing:.8px;'>CONFIANÇA</div>
      <div style='color:#fff;font-size:20px;font-weight:bold;'>{d.confidence*100:.1f}%</div>
      <div style='background:#333;height:8px;width:180px;border-radius:4px;margin-top:4px;'>
        <div style='background:#00d4e8;height:8px;width:{conf_bar_w}px;border-radius:4px;'></div>
      </div>
    </div>
    <!-- Prices -->
    <div style='background:#1a2035;padding:12px 20px;border-radius:6px;min-width:200px;'>
      <div style='color:#aaa;font-size:9px;letter-spacing:.8px;'>PREÇOS</div>
      <div style='font-size:11px;margin-top:4px;line-height:1.8;'>
        <span style='color:#aaa;'>Entrada:</span> <b style='color:#fff;'>{d.entry_price}</b><br>
        <span style='color:#aaa;'>Stop:</span> <b style='color:#ff4444;'>{d.stop_loss}</b><br>
        <span style='color:#aaa;'>Alvo:</span> <b style='color:#00ff99;'>{d.take_profit}</b>
      </div>
    </div>
    <!-- Size & Risk -->
    <div style='background:#1a2035;padding:12px 20px;border-radius:6px;min-width:200px;'>
      <div style='color:#aaa;font-size:9px;letter-spacing:.8px;'>SIZING & RISCO</div>
      <div style='font-size:11px;margin-top:4px;line-height:1.8;'>
        <span style='color:#aaa;'>Qtde:</span> <b style='color:#00d4e8;'>{d.quantity}</b><br>
        <span style='color:#aaa;'>Perda máx:</span> <b style='color:#ff6b35;'>${d.estimated_max_loss:,.0f}</b><br>
        <span style='color:#aaa;'>Custo est.:</span> <b>${d.estimated_trade_cost:,.0f}</b>
      </div>
    </div>
    <!-- Capital -->
    <div style='background:#1a2035;padding:12px 20px;border-radius:6px;min-width:200px;'>
      <div style='color:#aaa;font-size:9px;letter-spacing:.8px;'>CAPITAL</div>
      <div style='font-size:11px;margin-top:4px;line-height:1.8;'>
        <span style='color:#aaa;'>Disponível:</span> <b>${d.capital_available:,.0f}</b><br>
        <span style='color:#aaa;'>Margem:</span> <b>${d.margin_available:,.0f}</b><br>
        <span style='color:#aaa;'>Budget dia:</span> <b>${d.risk_budget_day_remaining:,.0f}</b>
      </div>
    </div>
    <!-- Execution status -->
    <div style='background:#1a2035;border-left:4px solid {block_color};
                padding:12px 20px;border-radius:6px;min-width:200px;'>
      <div style='color:#aaa;font-size:9px;letter-spacing:.8px;'>STATUS</div>
      <div style='color:{block_color};font-size:13px;font-weight:bold;'>{exec_label}</div>
      <div style='color:#aaa;font-size:10px;'>{d.block_reason or ("PAPER MODE" if True else "")}</div>
      <div style='color:#ffaa00;font-size:10px;'>Flatten: {d.flatten_time}</div>
    </div>
  </div>
  <!-- Strikes -->
  {'<div style="font-size:10px;color:#aaa;margin-bottom:8px;">Strikes: ' + strikes_html + '</div>' if d.strikes else ''}
  <!-- Regime probabilities -->
  <div style='margin-bottom:10px;'>
    <div style='color:#00d4e8;font-size:9px;letter-spacing:.8px;margin-bottom:4px;'>PROBABILIDADE POR REGIME</div>
    {regime_bars}
  </div>
  <!-- Rationale -->
  <div style='background:#0a0f1a;padding:8px 12px;border-radius:4px;font-size:10px;color:#aaa;'>
    {d.rationale}
  </div>
</div>
"""
        return html

    def _render_guide():
        return """
<div style='background:#0d1520;border:1px solid rgba(0,212,232,.15);border-radius:6px;
            padding:12px 16px;font-size:11px;font-family:monospace;line-height:1.8;margin:8px 0;'>
<span style='color:#00d4e8;font-size:10px;letter-spacing:.8px;'>COMO USAR O DECISION ENGINE</span><br><br>
<b style='color:#fff;'>① Configure a conta</b> — preencha Cash, NLV, Buying Power e Margem disponível com os valores reais da sua conta IB.<br>
<b style='color:#fff;'>② Ajuste os limites de risco</b> — Risk/Trade % = % do NLV que você aceita perder por operação (ex: 1% de $100k = $1.000 por trade).<br>
<b style='color:#fff;'>③ Clique "Gerar Decisão"</b> — o sistema lê o estado atual do mercado (opções chain, GEX, walls, IV) e produz a decisão.<br>
<b style='color:#fff;'>④ Leia o output</b>:<br>
&nbsp;&nbsp;· <span style='color:#00ff99;'>BUY</span> = entrar comprado (long call spread, long straddle, ES long)<br>
&nbsp;&nbsp;· <span style='color:#ff4444;'>SELL</span> = entrar vendido (short put spread, iron condor, ES short)<br>
&nbsp;&nbsp;· <span style='color:#aaa;'>NO_TRADE</span> = sem sinal claro ou bloqueio de risco<br>
<b style='color:#fff;'>⑤ Paper Execute</b> — registra no journal sem enviar ordens reais. Use para validar o fluxo.<br>
<b style='color:#fff;'>⑥ Live Execute</b> — só funciona se Paper Mode estiver DESLIGADO, IB conectado e risk engine liberar.<br>
<b style='color:#ff4444;'>⑦ Flatten All</b> — encerra TODAS as posições imediatamente. Use antes do fechamento ou em emergência.<br><br>
<b style='color:#ffaa00;'>⚠ Regra central:</b> o sistema opera SOMENTE day trade 0DTE. Toda posição é encerrada no mesmo dia.<br>
O horário limite de entrada é configurado em <code>LAST_ENTRY_MIN</code> minutos antes do fechamento.<br>
O sistema força flatten <code>FLATTEN_BEFORE_MIN</code> minutos antes do fechamento, independente do resultado.
</div>
"""

    def _on_run(_):
        orch = _make_orchestrator()
        orchestrator[0] = orch
        w_paper.description = 'PAPER MODE ON' if w_paper.value else 'LIVE MODE ATIVO'
        d = orch.run(df, external_scores=ext)
        with out_decision:
            out_decision.clear_output(wait=True)
            _disp(HTML(_render_decision(d)))
            _disp(HTML(_render_guide()))

    def _on_paper_exec(_):
        if orchestrator[0]:
            res = orchestrator[0].execute_paper()
            with out_journal:
                out_journal.clear_output(wait=True)
                _disp(HTML(f"<pre style='color:#00d4e8;font-size:10px;'>"
                           f"PAPER EXECUTE:\n{json.dumps(res, indent=2)}</pre>"))

    def _on_live_exec(_):
        if orchestrator[0]:
            res = orchestrator[0].execute_live()
            with out_journal:
                out_journal.clear_output(wait=True)
                _disp(HTML(f"<pre style='color:#ff6b35;font-size:10px;'>"
                           f"LIVE EXECUTE:\n{json.dumps(res, indent=2)}</pre>"))

    def _on_flatten(_):
        if orchestrator[0]:
            live = not w_paper.value
            orchestrator[0].flatten_all(live=live)
            with out_journal:
                out_journal.clear_output(wait=True)
                _disp(HTML("<div style='color:#ff4444;font-size:14px;font-weight:bold;'>"
                           "🚨 FLATTEN ALL enviado</div>"))

    def _on_paper_toggle(change):
        w_paper.description = 'PAPER MODE ON' if change['new'] else '⚠ LIVE MODE'
        w_paper.button_style = 'warning' if change['new'] else 'danger'

    w_run.on_click(_on_run)
    w_paper_ex.on_click(_on_paper_exec)
    w_live_ex.on_click(_on_live_exec)
    w_flatten.on_click(_on_flatten)
    w_paper.observe(_on_paper_toggle, names='value')

    header = wd.HTML(
        "<div style='padding:8px 0 4px;'>"
        "<h3 style='color:#00d4e8;margin:0 0 2px;font-size:15px;'>Decision Engine — 0DTE Intraday</h3>"
        "<p style='color:rgba(255,255,255,.3);font-size:10px;margin:0;'>"
        "Day trade only · 0DTE structures · ES / MES / SPX Options · "
        "Capital-constrained sizing · Flatten before close</p></div>")

    account_row = wd.VBox([
        wd.HTML("<p style='color:rgba(0,212,232,.5);font-size:9px;margin:4px 0 2px;"
                "letter-spacing:.8px;font-family:monospace;'>CONTA & CAPITAL</p>"),
        wd.HBox([w_cash, w_nlv, w_bp, w_margin, w_rpnl],
                layout=wd.Layout(flex_flow='row wrap', gap='8px')),
    ])
    risk_row = wd.VBox([
        wd.HTML("<p style='color:rgba(0,212,232,.5);font-size:9px;margin:8px 0 2px;"
                "letter-spacing:.8px;font-family:monospace;'>LIMITES DE RISCO</p>"),
        wd.HBox([w_risk_pct, w_daily_pct, w_maxpos],
                layout=wd.Layout(flex_flow='row wrap', gap='8px')),
    ])
    btn_row = wd.HBox(
        [w_paper, w_run, w_paper_ex, w_live_ex, w_flatten],
        layout=wd.Layout(gap='8px', margin='10px 0 6px 0', align_items='center'))

    # Initial render of guide
    with out_decision:
        _disp(HTML(_render_guide()))

    return wd.VBox([header, account_row, risk_row, btn_row, out_decision, out_journal])
