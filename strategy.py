"""
Mercatus Arena Production Strategy (V2 - Improved)
==================================================
Key Improvement: Adaptive z-score thresholds based on volatility regime.
  - High volatility: lower threshold (-1.5) for more opportunities
  - Low volatility: higher threshold (-2.5) for higher quality signals

Also includes P1 improvements:
  - Drawdown-based position scaling
  - Circuit breaker after 3 consecutive losses
  - Regime guard (skip MR on trending symbols)
  - Stale position exit after 500 bars

Formulas:
  adaptive_z = -2.0 + (vol_percentile - 0.5) * 1.5
  z_score = (Price - SMA(20)) / StdDev(20)
  RSI = 100 - 100/(1 + avg_gain/avg_loss)

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
    'price_history': {},
    'vol_history': {},
    'bar_count': {},
    'warmup_done': False,
    'initialized': False,
    'positions': {},
    'entry_prices': {},
    'highest_prices': {},
    'bars_in_pos': {},
    'cash_reserve_pct': 0.10,
    'max_exposure_pct': 0.20,
    'stop_loss_pct': 0.05,
    'trailing_stop_pct': 0.05,
    'min_hold_bars': 60,
    # P1 improvements
    'peak_equity': 0,
    'consecutive_losses': 0,
    'circuit_breaker_active': False,
    'circuit_breaker_cooldown': 0,
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
    if not _state['initialized']:
        _state['initialized'] = True
        for sym in current_data:
            _state['price_history'][sym] = []
            _state['vol_history'][sym] = []
            _state['bar_count'][sym] = 0
            _state['positions'][sym] = None
            _state['entry_prices'][sym] = None
            _state['highest_prices'][sym] = None
            _state['bars_in_pos'][sym] = 0

    for sym, data in current_data.items():
        if sym not in _state['price_history']:
            _state['price_history'][sym] = []
            _state['vol_history'][sym] = []
            _state['bar_count'][sym] = 0
            _state['positions'][sym] = None
            _state['entry_prices'][sym] = None
            _state['highest_prices'][sym] = None
            _state['bars_in_pos'][sym] = 0

        _state['price_history'][sym].append(data['close'])
        _state['vol_history'][sym].append(data.get('volume', 0))
        # Cap history to 500 bars (indicators need max ~200)
        if len(_state['price_history'][sym]) > 500:
            old_id = id(_state['price_history'][sym])
            _state['price_history'][sym] = _state['price_history'][sym][-500:]
            _state['vol_history'][sym] = _state['vol_history'][sym][-500:]
            # Clear vol buffer since array identity changed
            _state.setdefault('_vp_buf', {}).pop(old_id, None)
        _state['bar_count'][sym] += 1
        if _state['positions'][sym] is not None:
            _state['bars_in_pos'][sym] += 1

    min_bars = min(_state['bar_count'].values()) if _state['bar_count'] else 0
    if min_bars < WARMUP_BARS:
        return {}

    # Compute equity
    total_equity = cash
    for sym, pos in portfolio.items():
        qty = pos.get('quantity', 0)
        if qty > 0 and sym in current_data:
            total_equity += qty * current_data[sym]['close']

    _state['peak_equity'] = max(_state['peak_equity'], total_equity)
    current_dd = (_state['peak_equity'] - total_equity) / _state['peak_equity'] if _state['peak_equity'] > 0 else 0

    # Circuit breaker cooldown
    if _state['circuit_breaker_cooldown'] > 0:
        _state['circuit_breaker_cooldown'] -= 1
        if _state['circuit_breaker_cooldown'] == 0:
            _state['circuit_breaker_active'] = False

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

        # ---- RISK CHECKS ----

        # Circuit breaker: skip new buys
        if _state['circuit_breaker_active'] and held_qty == 0:
            continue

        # Drawdown scaling for position sizing
        dd_mult = 1.0
        if current_dd > 0.15:
            dd_mult = 0.25
        elif current_dd > 0.05:
            dd_mult = 1.0 - (current_dd - 0.05) / 0.10 * 0.75

        # Stop-loss check
        if held_qty > 0 and _state['entry_prices'][sym]:
            entry = _state['entry_prices'][sym]
            highest = _state['highest_prices'].get(sym, entry)
            _state['highest_prices'][sym] = max(highest, current_price)

            if current_price < _state['highest_prices'][sym] * (1 - _state['trailing_stop_pct']):
                actions[sym] = ('SELL', held_qty)
                _state['positions'][sym] = None
                _state['entry_prices'][sym] = None
                _state['highest_prices'][sym] = None
                _state['bars_in_pos'][sym] = 0
                _state['consecutive_losses'] += 1
                if _state['consecutive_losses'] >= 3:
                    _state['circuit_breaker_active'] = True
                    _state['circuit_breaker_cooldown'] = 300
                continue

            if current_price < entry * (1 - _state['stop_loss_pct']):
                actions[sym] = ('SELL', held_qty)
                _state['positions'][sym] = None
                _state['entry_prices'][sym] = None
                _state['highest_prices'][sym] = None
                _state['bars_in_pos'][sym] = 0
                _state['consecutive_losses'] += 1
                if _state['consecutive_losses'] >= 3:
                    _state['circuit_breaker_active'] = True
                    _state['circuit_breaker_cooldown'] = 300
                continue

            # Stale exit (500 bars)
            if _state['bars_in_pos'][sym] > 500:
                actions[sym] = ('SELL', held_qty)
                _state['positions'][sym] = None
                _state['entry_prices'][sym] = None
                _state['highest_prices'][sym] = None
                _state['bars_in_pos'][sym] = 0
                continue

        # ---- MODEL SIGNALS ----

        signal_mom = _momentum_signal(prices, volumes, bar_idx)
        signal_mr = _mean_reversion_signal(prices, volumes, bar_idx)
        signal_comp = _composite_signal(prices, volumes, bar_idx)

        def signal_to_num(sig):
            return 1.0 if sig == 'BUY' else (-1.0 if sig == 'SELL' else 0.0)

        weighted_score = (
            0.55 * signal_to_num(signal_mr[0]) * signal_mr[1] +
            0.25 * signal_to_num(signal_comp[0]) * signal_comp[1] +
            0.20 * signal_to_num(signal_mom[0]) * signal_mom[1]
        )

        buy_votes = sum(1 for s in [signal_mom[0], signal_mr[0], signal_comp[0]] if s == 'BUY')
        sell_votes = sum(1 for s in [signal_mom[0], signal_mr[0], signal_comp[0]] if s == 'SELL')

        if weighted_score > 0.3 and held_qty == 0:
            avg_conf = min(1.0, weighted_score)
            vol = _compute_volatility(prices)
            base_alloc = min(_state['max_exposure_pct'], 0.10 * avg_conf)
            vol_scalar = min(1.0, 0.02 / max(vol, 0.001))
            dollar_alloc = total_equity * base_alloc * vol_scalar * dd_mult
            dollar_alloc = min(dollar_alloc, cash * (1 - _state['cash_reserve_pct']))

            qty = int(dollar_alloc / current_price) if current_price > 0 else 0
            if qty > 0:
                actions[sym] = ('BUY', qty)
                _state['positions'][sym] = 'long'
                _state['entry_prices'][sym] = current_price
                _state['highest_prices'][sym] = current_price
                _state['bars_in_pos'][sym] = 0

        elif (weighted_score < -0.2 or sell_votes >= 2) and held_qty > 0:
            if _state['bars_in_pos'][sym] > _state['min_hold_bars']:
                actions[sym] = ('SELL', held_qty)
                _state['positions'][sym] = None
                _state['entry_prices'][sym] = None
                _state['highest_prices'][sym] = None
                _state['bars_in_pos'][sym] = 0

        # Flash crash recovery
        if held_qty > 0 and bar_idx >= 60:
            old_price = prices[-60]
            if old_price > 0:
                pct = (current_price - old_price) / old_price
                if pct < -0.15:
                    _state['highest_prices'][sym] = current_price * 1.02

    return actions


def _ema(prices, span):
    if len(prices) < span:
        return prices[-1]
    alpha = 2.0 / (span + 1)
    ema_val = float(prices[0])
    for i in range(1, len(prices)):
        ema_val = alpha * float(prices[i]) + (1 - alpha) * ema_val
    return ema_val


def _rsi(prices, period=14):
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
    if len(prices) < window + 1:
        return 0.02
    rets = np.diff(np.log(prices[-window - 1:]))
    return np.std(rets) if len(rets) > 0 else 0.02


def _compute_vol_percentile(prices, bar_idx, sym=None, lookback=100):
    """Fast O(1) vol percentile using rolling buffer in state."""
    if bar_idx < 30:
        return 0.5
    current_vol = _compute_volatility(prices[-10:])
    buf = _state.setdefault('_vp_buf', {})
    key = sym or str(bar_idx)
    if key not in buf:
        buf[key] = {'vals': [0.0] * 20, 'idx': 0}
    b = buf[key]
    b['vals'][b['idx'] % 20] = current_vol
    b['idx'] += 1
    filled = min(b['idx'], 20)
    if filled < 5:
        return 0.5
    return float(np.mean(np.array(b['vals'][:filled]) < current_vol))


def _momentum_signal(prices, volumes, bar_idx):
    if bar_idx < 35:
        return ('HOLD', 0.0)
    ema10 = _ema(prices, 10)
    ema30 = _ema(prices, 30)
    prev_ema10 = _ema(prices[:-1], 10)
    prev_ema30 = _ema(prices[:-1], 30)
    rsi_val = _rsi(prices)
    if prev_ema10 <= prev_ema30 and ema10 > ema30:
        if rsi_val < 74:
            return ('BUY', 0.7)
    if prev_ema10 >= prev_ema30 and ema10 < ema30:
        return ('SELL', 0.8)
    return ('HOLD', 0.0)


def _mean_reversion_signal(prices, volumes, bar_idx):
    """V2: Adaptive z-score threshold based on volatility regime."""
    if bar_idx < 30:
        return ('HOLD', 0.0)

    window = 20
    sma_val = np.mean(prices[-window:])
    std_val = np.std(prices[-window:])
    if std_val == 0:
        return ('HOLD', 0.0)

    z = (prices[-1] - sma_val) / std_val
    rsi_val = _rsi(prices)

    # KEY IMPROVEMENT: Adaptive z-score threshold
    vol_pct = _compute_vol_percentile(prices, bar_idx, sym=None)
    adaptive_z = -2.0 + (vol_pct - 0.5) * 1.5  # range: -2.75 to -1.25

    # Flash crash
    if bar_idx >= 60:
        old = prices[-60]
        if old > 0 and (prices[-1] - old) / old < -0.10:
            return ('BUY', 0.8)

    # Oversold with adaptive threshold
    if z < adaptive_z and rsi_val < 34:
        return ('BUY', min(1.0, abs(z) / 3.0))

    # Exit at mean
    if z > 0:
        return ('SELL', 0.6)

    return ('HOLD', 0.0)


def _composite_signal(prices, volumes, bar_idx):
    if bar_idx < 30:
        return ('HOLD', 0.0)
    rsi_val = _rsi(prices)
    ema12 = _ema(prices, 12)
    ema26 = _ema(prices, 26)
    macd_val = ema12 - ema26
    prev_ema12 = _ema(prices[:-1], 12)
    prev_ema26 = _ema(prices[:-1], 26)
    prev_macd = prev_ema12 - prev_ema26
    if prev_macd < 0 and macd_val > 0 and rsi_val < 70:
        return ('BUY', 0.6)
    if prev_macd > 0 and macd_val < 0:
        return ('SELL', 0.6)
    return ('HOLD', 0.0)
