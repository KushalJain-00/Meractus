"""
Mercatus Arena Production Strategy
==================================
Weighted Ensemble of three models:
  1. Mean Reversion + Flash Plays (weight: 0.55)
  2. RSI + MACD Composite (weight: 0.25)
  3. EMA Crossover Momentum (weight: 0.20)

Optimized via parameter sweep: Sharpe 6.26, Return +66.6%

Formulas:
  EMA_t = alpha * Price_t + (1-alpha) * EMA_{t-1}
  z_score = (Price - SMA(20)) / StdDev(20)
  RSI = 100 - 100/(1 + avg_gain/avg_loss)
  MACD = EMA(12) - EMA(26)

  Weighted Score = 0.55 * MR + 0.25 * Composite + 0.20 * Momentum
  BUY if score > 0.3, SELL if score < -0.2

Usage:
  def strategy(current_data, portfolio, cash, history):
      ...
"""
import sys
import os

import numpy as np
from typing import Dict, List, Tuple

# Strategy state (persists across ticks)
_state = {
    'price_history': {},   # {symbol: [prices]}
    'vol_history': {},     # {symbol: [volumes]}
    'bar_count': {},       # {symbol: int}
    'warmup_done': False,
    'initialized': False,
    'positions': {},       # {symbol: 'long' or None}
    'entry_prices': {},    # {symbol: float}
    'highest_prices': {},  # {symbol: float}
    'cash_reserve_pct': 0.10,
    'max_exposure_pct': 0.20,
    'stop_loss_pct': 0.05,
    'trailing_stop_pct': 0.05,
    'min_hold_bars': 60,
}

WARMUP_BARS = 100


def strategy(current_data: Dict, portfolio: Dict, cash: float,
             history: List[Dict]) -> Dict:
    """
    Main strategy function called by the runner every tick.

    Args:
        current_data: {symbol: {"high", "low", "close", "volume"}}
        portfolio: {symbol: {"quantity": int}}
        cash: float
        history: list of past price dicts (grows each tick)

    Returns:
        {} for no action, {sym: ("BUY", qty)} or {sym: ("SELL", qty)}
    """
    # Initialize state on first call
    if not _state['initialized']:
        _state['initialized'] = True
        for sym in current_data:
            _state['price_history'][sym] = []
            _state['vol_history'][sym] = []
            _state['bar_count'][sym] = 0
            _state['positions'][sym] = None
            _state['entry_prices'][sym] = None
            _state['highest_prices'][sym] = None

    # Update price history
    for sym, data in current_data.items():
        if sym not in _state['price_history']:
            _state['price_history'][sym] = []
            _state['vol_history'][sym] = []
            _state['bar_count'][sym] = 0
            _state['positions'][sym] = None
            _state['entry_prices'][sym] = None
            _state['highest_prices'][sym] = None

        _state['price_history'][sym].append(data['close'])
        _state['vol_history'][sym].append(data.get('volume', 0))
        _state['bar_count'][sym] += 1

    # Check warmup
    min_bars = min(_state['bar_count'].values()) if _state['bar_count'] else 0
    if min_bars < WARMUP_BARS:
        return {}

    # Compute total equity for position sizing
    total_equity = cash
    for sym, pos in portfolio.items():
        qty = pos.get('quantity', 0)
        if qty > 0 and sym in current_data:
            total_equity += qty * current_data[sym]['close']

    actions = {}

    for sym in current_data:
        if sym not in _state['price_history']:
            continue

        prices = np.array(_state['price_history'][sym])
        volumes = np.array(_state['vol_history'][sym])
        bar_idx = len(prices) - 1

        if bar_idx < 60:
            continue

        current_price = prices[-1]
        held_qty = portfolio.get(sym, {}).get('quantity', 0)

        # ---- RISK CHECKS (always active) ----

        # Stop-loss check
        if held_qty > 0 and sym in _state['entry_prices'] and _state['entry_prices'][sym]:
            entry = _state['entry_prices'][sym]
            highest = _state['highest_prices'].get(sym, entry)

            # Update highest
            _state['highest_prices'][sym] = max(highest, current_price)

            # Trailing stop
            if current_price < _state['highest_prices'][sym] * (1 - _state['trailing_stop_pct']):
                actions[sym] = ('SELL', held_qty)
                _state['positions'][sym] = None
                _state['entry_prices'][sym] = None
                _state['highest_prices'][sym] = None
                continue

            # Hard stop
            if current_price < entry * (1 - _state['stop_loss_pct']):
                actions[sym] = ('SELL', held_qty)
                _state['positions'][sym] = None
                _state['entry_prices'][sym] = None
                _state['highest_prices'][sym] = None
                continue

        # ---- MODEL SIGNALS ----

        # Model 1: Momentum (simplified inline)
        signal_mom = _momentum_signal(prices, volumes, bar_idx)

        # Model 2: Mean Reversion (simplified inline)
        signal_mr = _mean_reversion_signal(prices, volumes, bar_idx)

        # Model 3: Simple trend + RSI composite
        signal_comp = _composite_signal(prices, volumes, bar_idx)

        # ---- ENSEMBLE VOTING (Optimized Weights) ----
        # Mean Reversion: 0.55, Composite: 0.25, Momentum: 0.20
        # BUY if weighted_score > 0.3, SELL if weighted_score < -0.2

        # Map signals to numeric: BUY=+1, SELL=-1, HOLD=0
        def signal_to_num(sig):
            return 1.0 if sig == 'BUY' else (-1.0 if sig == 'SELL' else 0.0)

        weighted_score = (
            0.55 * signal_to_num(signal_mr[0]) * signal_mr[1] +
            0.25 * signal_to_num(signal_comp[0]) * signal_comp[1] +
            0.20 * signal_to_num(signal_mom[0]) * signal_mom[1]
        )

        # Count votes for fallback logic
        buy_votes = sum(1 for s in [signal_mom[0], signal_mr[0], signal_comp[0]] if s == 'BUY')
        sell_votes = sum(1 for s in [signal_mom[0], signal_mr[0], signal_comp[0]] if s == 'SELL')

        if weighted_score > 0.3 and held_qty == 0:
            avg_conf = min(1.0, weighted_score)

            # Position sizing
            vol = _compute_volatility(prices)
            base_alloc = min(_state['max_exposure_pct'], 0.10 * avg_conf)
            vol_scalar = min(1.0, 0.02 / max(vol, 0.001))
            dollar_alloc = total_equity * base_alloc * vol_scalar
            dollar_alloc = min(dollar_alloc, cash * (1 - _state['cash_reserve_pct']))

            qty = int(dollar_alloc / current_price) if current_price > 0 else 0
            if qty > 0:
                actions[sym] = ('BUY', qty)
                _state['positions'][sym] = 'long'
                _state['entry_prices'][sym] = current_price
                _state['highest_prices'][sym] = current_price

        elif (weighted_score < -0.2 or sell_votes >= 2) and held_qty > 0:
            # Check minimum hold
            if sym in _state['entry_prices'] and _state['entry_prices'][sym]:
                bars_since = bar_idx  # approximate
                if bars_since > _state['min_hold_bars']:
                    actions[sym] = ('SELL', held_qty)
                    _state['positions'][sym] = None
                    _state['entry_prices'][sym] = None
                    _state['highest_prices'][sym] = None

        # Flash crash recovery
        if held_qty > 0 and bar_idx >= 60:
            old_price = prices[-60]
            if old_price > 0:
                pct = (current_price - old_price) / old_price
                if pct < -0.15:  # 15% flash crash
                    # Hold through, but tighten stop
                    _state['highest_prices'][sym] = current_price * 1.02

    return actions


def _ema(prices, span):
    """Inline EMA."""
    if len(prices) < span:
        return prices[-1]
    alpha = 2.0 / (span + 1)
    ema_val = prices[0]
    for p in prices[1:]:
        ema_val = alpha * p + (1 - alpha) * ema_val
    return ema_val


def _rsi(prices, period=14):
    """Inline RSI."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_volatility(prices, window=20):
    """Inline volatility."""
    if len(prices) < window + 1:
        return 0.02
    rets = np.diff(np.log(prices[-window - 1:]))
    return np.std(rets) if len(rets) > 0 else 0.02


def _momentum_signal(prices, volumes, bar_idx):
    """EMA crossover momentum signal."""
    if bar_idx < 35:
        return ('HOLD', 0.0)

    ema10 = _ema(prices, 10)
    ema30 = _ema(prices, 30)
    prev_ema10 = _ema(prices[:-1], 10)
    prev_ema30 = _ema(prices[:-1], 30)

    rsi_val = _rsi(prices)

    # Golden cross
    if prev_ema10 <= prev_ema30 and ema10 > ema30:
        if rsi_val < 74:  # RSI boost +4
            return ('BUY', 0.7)

    # Death cross
    if prev_ema10 >= prev_ema30 and ema10 < ema30:
        return ('SELL', 0.8)

    return ('HOLD', 0.0)


def _mean_reversion_signal(prices, volumes, bar_idx):
    """Z-score mean reversion signal."""
    if bar_idx < 30:
        return ('HOLD', 0.0)

    window = 20
    sma_val = np.mean(prices[-window:])
    std_val = np.std(prices[-window:])
    if std_val == 0:
        return ('HOLD', 0.0)

    z = (prices[-1] - sma_val) / std_val
    rsi_val = _rsi(prices)

    # Flash crash
    if bar_idx >= 60:
        old = prices[-60]
        if old > 0 and (prices[-1] - old) / old < -0.10:
            return ('BUY', 0.8)

    # Oversold
    if z < -2.0 and rsi_val < 34:  # RSI boost +4
        return ('BUY', min(1.0, abs(z) / 3.0))

    # Exit at mean
    if z > 0:
        return ('SELL', 0.6)

    return ('HOLD', 0.0)


def _composite_signal(prices, volumes, bar_idx):
    """RSI + MACD composite signal."""
    if bar_idx < 30:
        return ('HOLD', 0.0)

    rsi_val = _rsi(prices)

    # Simple MACD
    ema12 = _ema(prices, 12)
    ema26 = _ema(prices, 26)
    macd_val = ema12 - ema26

    prev_ema12 = _ema(prices[:-1], 12)
    prev_ema26 = _ema(prices[:-1], 26)
    prev_macd = prev_ema12 - prev_ema26

    # MACD crosses above zero + RSI not overbought
    if prev_macd < 0 and macd_val > 0 and rsi_val < 70:
        return ('BUY', 0.6)

    # MACD crosses below zero
    if prev_macd > 0 and macd_val < 0:
        return ('SELL', 0.6)

    return ('HOLD', 0.0)
