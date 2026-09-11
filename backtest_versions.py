"""
Versioned backtesting engine.
Each improvement is a version. Models and results are saved at each step.
"""
import sys, os, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from features import ema, rsi, bollinger_bands, macd, detect_regime
from features import sma as sma_func


RESULTS_DIR = '/home/kushal_jain/Meractus/results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_version(version_name, results, metadata=None):
    """Save results and metadata for a version."""
    path = os.path.join(RESULTS_DIR, f'{version_name}.json')
    save_data = {k: v for k, v in results.items() if k not in ['equity_curve', 'trade_log']}
    if metadata:
        save_data['metadata'] = metadata
    with open(path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Saved: {path}")
    return path


def load_all_versions():
    """Load all version results for comparison."""
    versions = {}
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.endswith('.json'):
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                versions[fname.replace('.json', '')] = json.load(f)
    return versions


def print_comparison(versions):
    """Print comparison table of all versions."""
    print("\n" + "=" * 100)
    print(f"{'Version':<35} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'Trades':>8} {'WinRate':>10} {'Best5D':>10} {'Worst5D':>10}")
    print("-" * 100)
    for name, r in sorted(versions.items()):
        print(f"{name:<35} {r.get('total_return_pct', 0):>+9.2f}% {r.get('sharpe_ratio', 0):>8.3f} {r.get('max_drawdown_pct', 0):>9.2f}% {r.get('total_trades', 0):>8} {r.get('win_rate_pct', 0):>9.1f}% {r.get('best_5d_pct', 0):>+9.2f}% {r.get('worst_5d_pct', 0):>+9.2f}%")
    print("=" * 100)


# ============================================================
# V0: BASELINE (current mean reversion)
# ============================================================
def compute_signals_v0(price_arr, vol_arr, symbols):
    """Baseline: fixed z-score < -2.0, exit at z > 0, no regime filter."""
    n_bars, n_syms = price_arr.shape
    signals = np.zeros((n_bars, n_syms))
    confidences = np.zeros((n_bars, n_syms))
    positions_held = np.zeros(n_syms, dtype=int)  # track shares

    for s in range(n_syms):
        p = price_arr[:, s]
        window = 20

        roll_mean = np.full(n_bars, np.nan)
        roll_std = np.full(n_bars, np.nan)
        for i in range(window - 1, n_bars):
            roll_mean[i] = np.mean(p[i - window + 1:i + 1])
            roll_std[i] = np.std(p[i - window + 1:i + 1], ddof=1)
        z = np.where(roll_std > 0, (p - roll_mean) / roll_std, 0.0)

        rsi_vals = np.full(n_bars, 50.0)
        period = 14
        deltas = np.diff(p)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        if len(gains) >= period:
            ag = np.mean(gains[:period])
            al = np.mean(losses[:period])
            for i in range(period, len(deltas)):
                ag = (ag * (period - 1) + gains[i]) / period
                al = (al * (period - 1) + losses[i]) / period
                rsi_vals[i + 1] = 100.0 - 100.0 / (1.0 + ag / al) if al > 0 else 100.0

        for i in range(40, n_bars):
            if np.isnan(z[i]):
                continue
            # Flash crash
            if i >= 60 and p[i - 60] > 0 and (p[i] - p[i - 60]) / p[i - 60] < -0.10:
                signals[i, s] = 1.0
                confidences[i, s] = 0.8
            elif z[i] < -2.0 and rsi_vals[i] < 34:
                signals[i, s] = 1.0
                confidences[i, s] = min(1.0, abs(z[i]) / 3.0)
            elif z[i] > 0:
                signals[i, s] = -0.5
                confidences[i, s] = 0.5

    return signals, confidences


# ============================================================
# V1: ADAPTIVE Z-SCORE THRESHOLDS
# ============================================================
def compute_signals_v1(price_arr, vol_arr, symbols):
    """
    Improvement 1: Adaptive z-score thresholds based on volatility regime.
    High vol -> lower threshold (easier entry, more opportunities)
    Low vol -> higher threshold (stricter, higher quality)
    """
    n_bars, n_syms = price_arr.shape
    signals = np.zeros((n_bars, n_syms))
    confidences = np.zeros((n_bars, n_syms))

    for s in range(n_syms):
        p = price_arr[:, s]
        window = 20

        roll_mean = np.full(n_bars, np.nan)
        roll_std = np.full(n_bars, np.nan)
        for i in range(window - 1, n_bars):
            roll_mean[i] = np.mean(p[i - window + 1:i + 1])
            roll_std[i] = np.std(p[i - window + 1:i + 1], ddof=1)
        z = np.where(roll_std > 0, (p - roll_mean) / roll_std, 0.0)

        # Compute rolling volatility percentile
        vol_window = 100
        vol_series = np.full(n_bars, np.nan)
        for i in range(vol_window, n_bars):
            rets = np.diff(np.log(np.maximum(p[i - vol_window:i + 1], 0.01)))
            vol_series[i] = np.std(rets)

        # Adaptive threshold: high vol = easier entry (lower z threshold)
        # base_z = -2.0, adjusted by vol percentile
        adaptive_z = np.full(n_bars, -2.0)
        for i in range(vol_window + window, n_bars):
            if not np.isnan(vol_series[i]):
                # Compute percentile of current vol vs history
                hist_vol = vol_series[vol_window:i]
                hist_vol = hist_vol[~np.isnan(hist_vol)]
                if len(hist_vol) > 10:
                    pct = np.mean(hist_vol < vol_series[i])
                    # High vol (pct > 0.7) -> lower threshold (-1.5)
                    # Low vol (pct < 0.3) -> higher threshold (-2.5)
                    adaptive_z[i] = -2.0 + (pct - 0.5) * 1.5  # range: -2.75 to -1.25

        rsi_vals = np.full(n_bars, 50.0)
        period = 14
        deltas = np.diff(p)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        if len(gains) >= period:
            ag = np.mean(gains[:period])
            al = np.mean(losses[:period])
            for i in range(period, len(deltas)):
                ag = (ag * (period - 1) + gains[i]) / period
                al = (al * (period - 1) + losses[i]) / period
                rsi_vals[i + 1] = 100.0 - 100.0 / (1.0 + ag / al) if al > 0 else 100.0

        for i in range(40, n_bars):
            if np.isnan(z[i]):
                continue
            if i >= 60 and p[i - 60] > 0 and (p[i] - p[i - 60]) / p[i - 60] < -0.10:
                signals[i, s] = 1.0
                confidences[i, s] = 0.8
            elif z[i] < adaptive_z[i] and rsi_vals[i] < 34:
                signals[i, s] = 1.0
                confidences[i, s] = min(1.0, abs(z[i]) / 3.0)
            elif z[i] > 0:
                signals[i, s] = -0.5
                confidences[i, s] = 0.5

    return signals, confidences


# ============================================================
# V2: STAGED TAKE-PROFIT
# ============================================================
def compute_signals_v2(price_arr, vol_arr, symbols):
    """
    Improvement 2: Staged exits - partial profit taking.
    z > +0.5 -> sell 25%, z > +1.0 -> sell 50%, z > +2.0 -> sell 100%
    Uses signal strength encoding: -0.25, -0.5, -1.0
    """
    n_bars, n_syms = price_arr.shape
    signals = np.zeros((n_bars, n_syms))
    confidences = np.zeros((n_bars, n_syms))

    for s in range(n_syms):
        p = price_arr[:, s]
        window = 20

        roll_mean = np.full(n_bars, np.nan)
        roll_std = np.full(n_bars, np.nan)
        for i in range(window - 1, n_bars):
            roll_mean[i] = np.mean(p[i - window + 1:i + 1])
            roll_std[i] = np.std(p[i - window + 1:i + 1], ddof=1)
        z = np.where(roll_std > 0, (p - roll_mean) / roll_std, 0.0)

        # Adaptive threshold
        vol_window = 100
        vol_series = np.full(n_bars, np.nan)
        for i in range(vol_window, n_bars):
            rets = np.diff(np.log(np.maximum(p[i - vol_window:i + 1], 0.01)))
            vol_series[i] = np.std(rets)

        adaptive_z = np.full(n_bars, -2.0)
        for i in range(vol_window + window, n_bars):
            if not np.isnan(vol_series[i]):
                hist_vol = vol_series[vol_window:i]
                hist_vol = hist_vol[~np.isnan(hist_vol)]
                if len(hist_vol) > 10:
                    pct = np.mean(hist_vol < vol_series[i])
                    adaptive_z[i] = -2.0 + (pct - 0.5) * 1.5

        rsi_vals = np.full(n_bars, 50.0)
        period = 14
        deltas = np.diff(p)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        if len(gains) >= period:
            ag = np.mean(gains[:period])
            al = np.mean(losses[:period])
            for i in range(period, len(deltas)):
                ag = (ag * (period - 1) + gains[i]) / period
                al = (al * (period - 1) + losses[i]) / period
                rsi_vals[i + 1] = 100.0 - 100.0 / (1.0 + ag / al) if al > 0 else 100.0

        for i in range(40, n_bars):
            if np.isnan(z[i]):
                continue
            if i >= 60 and p[i - 60] > 0 and (p[i] - p[i - 60]) / p[i - 60] < -0.10:
                signals[i, s] = 1.0
                confidences[i, s] = 0.8
            elif z[i] < adaptive_z[i] and rsi_vals[i] < 34:
                signals[i, s] = 1.0
                confidences[i, s] = min(1.0, abs(z[i]) / 3.0)
            # Staged exits (instead of simple z > 0)
            elif z[i] > 2.0:
                signals[i, s] = -1.0  # full exit
                confidences[i, s] = 0.9
            elif z[i] > 1.0:
                signals[i, s] = -0.5  # partial exit
                confidences[i, s] = 0.7
            elif z[i] > 0.5:
                signals[i, s] = -0.25  # trim
                confidences[i, s] = 0.5

    return signals, confidences


# ============================================================
# V3: ATR-BASED STOPS
# ============================================================
def compute_atr(high, low, close, period=14):
    """Compute ATR."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period] = np.mean(tr[1:period + 1])
        for i in range(period + 1, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def compute_signals_v3(price_arr, vol_arr, symbols, high_arr=None, low_arr=None):
    """
    Improvement 3: ATR-based dynamic stops.
    Stop = entry - 2.0 * ATR
    Take profit = entry + 3.0 * ATR (reward:risk = 1.5:1)
    """
    n_bars, n_syms = price_arr.shape
    signals = np.zeros((n_bars, n_syms))
    confidences = np.zeros((n_bars, n_syms))

    for s in range(n_syms):
        p = price_arr[:, s]
        h = high_arr[:, s] if high_arr is not None else p
        l = low_arr[:, s] if low_arr is not None else p
        window = 20

        roll_mean = np.full(n_bars, np.nan)
        roll_std = np.full(n_bars, np.nan)
        for i in range(window - 1, n_bars):
            roll_mean[i] = np.mean(p[i - window + 1:i + 1])
            roll_std[i] = np.std(p[i - window + 1:i + 1], ddof=1)
        z = np.where(roll_std > 0, (p - roll_mean) / roll_std, 0.0)

        atr_vals = compute_atr(h, l, p, 14)

        vol_window = 100
        vol_series = np.full(n_bars, np.nan)
        for i in range(vol_window, n_bars):
            rets = np.diff(np.log(np.maximum(p[i - vol_window:i + 1], 0.01)))
            vol_series[i] = np.std(rets)

        adaptive_z = np.full(n_bars, -2.0)
        for i in range(vol_window + window, n_bars):
            if not np.isnan(vol_series[i]):
                hist_vol = vol_series[vol_window:i]
                hist_vol = hist_vol[~np.isnan(hist_vol)]
                if len(hist_vol) > 10:
                    pct = np.mean(hist_vol < vol_series[i])
                    adaptive_z[i] = -2.0 + (pct - 0.5) * 1.5

        rsi_vals = np.full(n_bars, 50.0)
        period = 14
        deltas = np.diff(p)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses_arr = np.where(deltas < 0, -deltas, 0.0)
        if len(gains) >= period:
            ag = np.mean(gains[:period])
            al = np.mean(losses_arr[:period])
            for i in range(period, len(deltas)):
                ag = (ag * (period - 1) + gains[i]) / period
                al = (al * (period - 1) + losses_arr[i]) / period
                rsi_vals[i + 1] = 100.0 - 100.0 / (1.0 + ag / al) if al > 0 else 100.0

        # Track ATR-based stops
        entry_atr_stop = np.full(n_bars, np.nan)
        entry_atr_tp = np.full(n_bars, np.nan)

        for i in range(40, n_bars):
            if np.isnan(z[i]):
                continue

            # Check ATR stop/TP if in position
            if not np.isnan(entry_atr_stop[i]) and p[i] < entry_atr_stop[i]:
                signals[i, s] = -1.0
                confidences[i, s] = 1.0
                continue
            if not np.isnan(entry_atr_tp[i]) and p[i] > entry_atr_tp[i]:
                signals[i, s] = -0.8
                confidences[i, s] = 0.9
                continue

            if i >= 60 and p[i - 60] > 0 and (p[i] - p[i - 60]) / p[i - 60] < -0.10:
                signals[i, s] = 1.0
                confidences[i, s] = 0.8
                if not np.isnan(atr_vals[i]):
                    entry_atr_stop[i + 1:] = p[i] - 2.0 * atr_vals[i]
                    entry_atr_tp[i + 1:] = p[i] + 3.0 * atr_vals[i]
            elif z[i] < adaptive_z[i] and rsi_vals[i] < 34:
                signals[i, s] = 1.0
                confidences[i, s] = min(1.0, abs(z[i]) / 3.0)
                if not np.isnan(atr_vals[i]):
                    # Propagate stop/TP forward
                    for j in range(i + 1, min(i + 500, n_bars)):
                        if np.isnan(entry_atr_stop[j]):
                            entry_atr_stop[j] = p[i] - 2.0 * atr_vals[i]
                            entry_atr_tp[j] = p[i] + 3.0 * atr_vals[i]
                        else:
                            break
            elif z[i] > 2.0:
                signals[i, s] = -1.0
                confidences[i, s] = 0.9
                entry_atr_stop[i:] = np.nan
                entry_atr_tp[i:] = np.nan
            elif z[i] > 1.0:
                signals[i, s] = -0.5
                confidences[i, s] = 0.7
            elif z[i] > 0.5:
                signals[i, s] = -0.25
                confidences[i, s] = 0.5

    return signals, confidences


# ============================================================
# SIMULATION ENGINE (shared across all versions)
# ============================================================
def simulate(signals, confidences, price_arr, vol_arr, timestamps, symbols,
             initial_capital=10_000_000, version_name='v0',
             use_atr_stops=False, high_arr=None, low_arr=None,
             use_dd_scaling=False, use_circuit_breaker=False,
             use_regime_guard=False, use_stale_exit=False,
             use_kelly=False, use_vol_confirmation=False):
    """Simulate trades from pre-computed signals."""
    n_bars, n_syms = price_arr.shape
    cash = initial_capital
    holdings = np.zeros(n_syms, dtype=int)
    entry_prices = np.full(n_syms, np.nan)
    highest_prices = np.full(n_syms, np.nan)
    bars_in_pos = np.zeros(n_syms, dtype=int)
    equity_curve = np.zeros(n_bars)
    trade_log = []

    # ATR for stops
    atr_vals = None
    if use_atr_stops and high_arr is not None and low_arr is not None:
        atr_cache = {}
        for s in range(n_syms):
            atr_cache[s] = compute_atr(high_arr[:, s], low_arr[:, s], price_arr[:, s], 14)

    # Regime cache
    regime_cache = {}
    if use_regime_guard:
        for s in range(n_syms):
            p = price_arr[:, s]
            if len(p) > 500:
                regime_cache[s] = detect_regime(p[:2000], window=500, threshold=0.3)
            else:
                regime_cache[s] = 'unknown'

    # Circuit breaker state
    consecutive_losses = 0
    circuit_breaker_active = False
    circuit_breaker_cooldown = 0

    # Drawdown tracking
    peak_equity = initial_capital

    # Win/loss tracking for Kelly
    trade_wins = []
    trade_losses = []

    warmup = 100
    stop_loss_pct = 0.05
    trailing_stop_pct = 0.05
    min_hold_bars = 60

    for i in range(n_bars):
        pos_value = np.sum(holdings * price_arr[i])
        equity = cash + pos_value
        equity_curve[i] = equity

        if i < warmup:
            continue

        peak_equity = max(peak_equity, equity)
        current_dd = (peak_equity - equity) / peak_equity

        # Circuit breaker cooldown
        if circuit_breaker_cooldown > 0:
            circuit_breaker_cooldown -= 1
            if circuit_breaker_cooldown == 0:
                circuit_breaker_active = False

        # Update highest prices and check stops
        for s in range(n_syms):
            if holdings[s] > 0:
                highest_prices[s] = max(highest_prices[s], price_arr[i, s])
                bars_in_pos[s] += 1

                # ATR stop
                if use_atr_stops and atr_vals is not None:
                    if not np.isnan(atr_vals[s][i]) and price_arr[i, s] < entry_prices[s] - 2.0 * atr_vals[s][i]:
                        sell_value = holdings[s] * price_arr[i, s]
                        cash += sell_value
                        pnl = (price_arr[i, s] - entry_prices[s]) * holdings[s]
                        trade_log.append({'timestamp': int(timestamps[i]), 'symbol': symbols[s],
                                          'action': 'SELL', 'price': float(price_arr[i, s]),
                                          'quantity': int(holdings[s]), 'pnl': float(pnl), 'reason': 'atr_stop'})
                        if pnl > 0:
                            trade_wins.append(pnl)
                            consecutive_losses = 0
                        else:
                            trade_losses.append(abs(pnl))
                            consecutive_losses += 1
                        holdings[s] = 0
                        entry_prices[s] = np.nan
                        highest_prices[s] = np.nan
                        bars_in_pos[s] = 0
                        continue

                # Trailing stop
                if current_dd < 0.15 and bars_in_pos[s] > min_hold_bars:
                    if price_arr[i, s] < highest_prices[s] * (1 - trailing_stop_pct):
                        sell_value = holdings[s] * price_arr[i, s]
                        cash += sell_value
                        pnl = (price_arr[i, s] - entry_prices[s]) * holdings[s]
                        trade_log.append({'timestamp': int(timestamps[i]), 'symbol': symbols[s],
                                          'action': 'SELL', 'price': float(price_arr[i, s]),
                                          'quantity': int(holdings[s]), 'pnl': float(pnl), 'reason': 'trailing_stop'})
                        if pnl > 0:
                            trade_wins.append(pnl)
                            consecutive_losses = 0
                        else:
                            trade_losses.append(abs(pnl))
                            consecutive_losses += 1
                        holdings[s] = 0
                        entry_prices[s] = np.nan
                        highest_prices[s] = np.nan
                        bars_in_pos[s] = 0

        # Process signals
        for s in range(n_syms):
            sig = signals[i, s]
            conf = confidences[i, s]
            current = price_arr[i, s]

            # Circuit breaker: skip new buys
            if circuit_breaker_active and sig > 0 and holdings[s] == 0:
                continue

            # Drawdown scaling
            dd_mult = 1.0
            if use_dd_scaling:
                if current_dd > 0.15:
                    dd_mult = 0.25
                elif current_dd > 0.05:
                    dd_mult = 1.0 - (current_dd - 0.05) / 0.10 * 0.75

            # Kelly sizing
            kelly_mult = 1.0
            if use_kelly and len(trade_wins) > 10 and len(trade_losses) > 0:
                p_win = len(trade_wins) / (len(trade_wins) + len(trade_losses))
                avg_win = np.mean(trade_wins[-50:]) if len(trade_wins) > 0 else 1
                avg_loss = np.mean(trade_losses[-50:]) if len(trade_losses) > 0 else 1
                b = avg_win / max(avg_loss, 0.01)
                raw_kelly = max(0, (p_win * b - (1 - p_win)) / b)
                kelly_mult = min(0.25, raw_kelly)  # fractional Kelly

            # Regime guard: skip mean reversion on trending symbols
            if use_regime_guard and sig > 0 and regime_cache.get(symbols[s]) == 'trending':
                continue

            # Volume confirmation
            if use_vol_confirmation and sig > 0:
                if i >= 20:
                    vol_sma = np.mean(vol_arr[i-20:i, s])
                    if vol_arr[i, s] < vol_sma * 1.2:
                        continue  # skip if volume not confirming

            if sig > 0.5 and holdings[s] == 0:
                # Position sizing
                vol = 0.02
                if i >= 20:
                    rets = np.diff(np.log(np.maximum(price_arr[i-20:i+1, s], 0.01)))
                    vol = np.std(rets) if len(rets) > 0 else 0.02

                base_alloc = min(0.20, 0.10 * conf)
                vol_scalar = min(1.0, 0.02 / max(vol, 0.001))
                dollar_alloc = equity * base_alloc * vol_scalar * dd_mult * kelly_mult
                available = cash - equity * 0.10
                dollar_alloc = min(dollar_alloc, max(available, 0))

                qty = int(dollar_alloc / current) if current > 0 else 0
                if qty > 0 and dollar_alloc <= cash * 0.90:
                    cash -= qty * current
                    holdings[s] = qty
                    entry_prices[s] = current
                    highest_prices[s] = current
                    bars_in_pos[s] = 0
                    trade_log.append({'timestamp': int(timestamps[i]), 'symbol': symbols[s],
                                      'action': 'BUY', 'price': float(current),
                                      'quantity': qty, 'confidence': float(conf), 'reason': 'signal'})

            elif sig < -0.5 and holdings[s] > 0 and bars_in_pos[s] > min_hold_bars:
                sell_value = holdings[s] * current
                cash += sell_value
                pnl = (current - entry_prices[s]) * holdings[s]
                trade_log.append({'timestamp': int(timestamps[i]), 'symbol': symbols[s],
                                  'action': 'SELL', 'price': float(current),
                                  'quantity': int(holdings[s]), 'pnl': float(pnl), 'reason': 'signal'})
                if pnl > 0:
                    trade_wins.append(pnl)
                    consecutive_losses = 0
                else:
                    trade_losses.append(abs(pnl))
                    consecutive_losses += 1
                    if consecutive_losses >= 3:
                        circuit_breaker_active = True
                        circuit_breaker_cooldown = 300  # pause 300 bars
                holdings[s] = 0
                entry_prices[s] = np.nan
                highest_prices[s] = np.nan
                bars_in_pos[s] = 0

            # Stale exit
            elif use_stale_exit and holdings[s] > 0 and bars_in_pos[s] > 500:
                sell_value = holdings[s] * current
                cash += sell_value
                pnl = (current - entry_prices[s]) * holdings[s]
                trade_log.append({'timestamp': int(timestamps[i]), 'symbol': symbols[s],
                                  'action': 'SELL', 'price': float(current),
                                  'quantity': int(holdings[s]), 'pnl': float(pnl), 'reason': 'stale_exit'})
                holdings[s] = 0
                entry_prices[s] = np.nan
                highest_prices[s] = np.nan
                bars_in_pos[s] = 0

    # Metrics
    final_equity = equity_curve[-1]
    returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1)
    returns = returns[np.isfinite(returns)]
    bars_per_day = 390
    if len(returns) > bars_per_day:
        n_days = len(returns) // bars_per_day
        daily_rets = np.array([np.mean(returns[i * bars_per_day:(i + 1) * bars_per_day]) for i in range(n_days)])
    else:
        daily_rets = returns

    sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0
    cummax = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - cummax) / np.maximum(cummax, 1)
    max_dd = np.min(dd)

    sell_trades = [t for t in trade_log if t['action'] == 'SELL']
    buy_trades = [t for t in trade_log if t['action'] == 'BUY']
    n_rt = min(len(buy_trades), len(sell_trades))
    wins = sum(1 for j in range(n_rt) if sell_trades[j]['price'] > buy_trades[j]['price'])
    win_rate = wins / n_rt if n_rt > 0 else 0

    sym_pnl = {}
    for t in trade_log:
        if 'pnl' in t:
            sym_pnl[t['symbol']] = sym_pnl.get(t['symbol'], 0) + t['pnl']

    window_size = 5 * 390
    best_5d = worst_5d = 0
    for start in range(0, len(equity_curve) - window_size, window_size):
        ret = (equity_curve[start + window_size] - equity_curve[start]) / equity_curve[start]
        best_5d = max(best_5d, ret)
        worst_5d = min(worst_5d, ret)

    return {
        'strategy': version_name,
        'final_equity': float(final_equity),
        'total_return_pct': float((final_equity - initial_capital) / initial_capital * 100),
        'sharpe_ratio': float(sharpe),
        'max_drawdown_pct': float(max_dd * 100),
        'total_trades': len(trade_log),
        'win_rate_pct': float(win_rate * 100),
        'symbol_pnl': sym_pnl,
        'best_5d_pct': float(best_5d * 100),
        'worst_5d_pct': float(worst_5d * 100),
        'profit_factor': float(sum(trade_wins) / max(sum(trade_losses), 1)),
    }


# ============================================================
# MAIN: Run all versions
# ============================================================
if __name__ == '__main__':
    csv_path = '/home/kushal_jain/Meractus/paper_dataset(1).csv'
    df = pd.read_csv(csv_path).sort_values(['symbol', 'timestamp'])

    pivot_price = df.pivot_table(index='timestamp', columns='symbol', values='price')
    pivot_vol = df.pivot_table(index='timestamp', columns='symbol', values='volume')
    pivot_high = df.pivot_table(index='timestamp', columns='symbol', values='high')
    pivot_low = df.pivot_table(index='timestamp', columns='symbol', values='low')

    timestamps = pivot_price.index.values
    symbols = pivot_price.columns.tolist()
    price_arr = pivot_price.values
    vol_arr = pivot_vol.values.astype(float)
    high_arr = pivot_high.values
    low_arr = pivot_low.values

    print(f"Data loaded: {len(symbols)} symbols, {len(timestamps)} bars")

    all_results = {}

    # V0: Baseline
    print("\n--- V0: Baseline (fixed z-score) ---")
    t0 = time.time()
    sig0, conf0 = compute_signals_v0(price_arr, vol_arr, symbols)
    r0 = simulate(sig0, conf0, price_arr, vol_arr, timestamps, symbols, version_name='v0_baseline')
    save_version('v0_baseline', r0, {'description': 'Fixed z-score < -2.0, exit z > 0'})
    all_results['v0_baseline'] = r0
    print(f"  Time: {time.time()-t0:.1f}s | Return: {r0['total_return_pct']:+.2f}% | Sharpe: {r0['sharpe_ratio']:.3f} | WinRate: {r0['win_rate_pct']:.1f}%")

    # V1: Adaptive z-score
    print("\n--- V1: Adaptive Z-Score Thresholds ---")
    t0 = time.time()
    sig1, conf1 = compute_signals_v1(price_arr, vol_arr, symbols)
    r1 = simulate(sig1, conf1, price_arr, vol_arr, timestamps, symbols, version_name='v1_adaptive_z')
    save_version('v1_adaptive_z', r1, {'description': 'Volatility-adaptive z-score thresholds'})
    all_results['v1_adaptive_z'] = r1
    print(f"  Time: {time.time()-t0:.1f}s | Return: {r1['total_return_pct']:+.2f}% | Sharpe: {r1['sharpe_ratio']:.3f} | WinRate: {r1['win_rate_pct']:.1f}%")

    # V2: Staged take-profit
    print("\n--- V2: Staged Take-Profit ---")
    t0 = time.time()
    sig2, conf2 = compute_signals_v2(price_arr, vol_arr, symbols)
    r2 = simulate(sig2, conf2, price_arr, vol_arr, timestamps, symbols, version_name='v2_staged_exit')
    save_version('v2_staged_exit', r2, {'description': 'Adaptive z + staged exits at z=+0.5/+1.0/+2.0'})
    all_results['v2_staged_exit'] = r2
    print(f"  Time: {time.time()-t0:.1f}s | Return: {r2['total_return_pct']:+.2f}% | Sharpe: {r2['sharpe_ratio']:.3f} | WinRate: {r2['win_rate_pct']:.1f}%")

    # V3: ATR-based stops
    print("\n--- V3: ATR-Based Stops ---")
    t0 = time.time()
    sig3, conf3 = compute_signals_v3(price_arr, vol_arr, symbols, high_arr, low_arr)
    r3 = simulate(sig3, conf3, price_arr, vol_arr, timestamps, symbols,
                  version_name='v3_atr_stops', use_atr_stops=True, high_arr=high_arr, low_arr=low_arr)
    save_version('v3_atr_stops', r3, {'description': 'Adaptive z + staged exits + ATR-based stops'})
    all_results['v3_atr_stops'] = r3
    print(f"  Time: {time.time()-t0:.1f}s | Return: {r3['total_return_pct']:+.2f}% | Sharpe: {r3['sharpe_ratio']:.3f} | WinRate: {r3['win_rate_pct']:.1f}%")

    # V4: All P0 + Drawdown scaling
    print("\n--- V4: + Drawdown Scaling ---")
    t0 = time.time()
    r4 = simulate(sig3, conf3, price_arr, vol_arr, timestamps, symbols,
                  version_name='v4_dd_scaling', use_atr_stops=True, high_arr=high_arr, low_arr=low_arr,
                  use_dd_scaling=True)
    save_version('v4_dd_scaling', r4, {'description': 'V3 + drawdown-based position scaling'})
    all_results['v4_dd_scaling'] = r4
    print(f"  Time: {time.time()-t0:.1f}s | Return: {r4['total_return_pct']:+.2f}% | Sharpe: {r4['sharpe_ratio']:.3f} | WinRate: {r4['win_rate_pct']:.1f}%")

    # V5: + Circuit breaker
    print("\n--- V5: + Circuit Breaker ---")
    t0 = time.time()
    r5 = simulate(sig3, conf3, price_arr, vol_arr, timestamps, symbols,
                  version_name='v5_circuit_breaker', use_atr_stops=True, high_arr=high_arr, low_arr=low_arr,
                  use_dd_scaling=True, use_circuit_breaker=True)
    save_version('v5_circuit_breaker', r5, {'description': 'V4 + circuit breaker after 3 losses'})
    all_results['v5_circuit_breaker'] = r5
    print(f"  Time: {time.time()-t0:.1f}s | Return: {r5['total_return_pct']:+.2f}% | Sharpe: {r5['sharpe_ratio']:.3f} | WinRate: {r5['win_rate_pct']:.1f}%")

    # V6: + Regime guard
    print("\n--- V6: + Regime Guard ---")
    t0 = time.time()
    r6 = simulate(sig3, conf3, price_arr, vol_arr, timestamps, symbols,
                  version_name='v6_regime_guard', use_atr_stops=True, high_arr=high_arr, low_arr=low_arr,
                  use_dd_scaling=True, use_circuit_breaker=True, use_regime_guard=True)
    save_version('v6_regime_guard', r6, {'description': 'V5 + skip MR on trending symbols'})
    all_results['v6_regime_guard'] = r6
    print(f"  Time: {time.time()-t0:.1f}s | Return: {r6['total_return_pct']:+.2f}% | Sharpe: {r6['sharpe_ratio']:.3f} | WinRate: {r6['win_rate_pct']:.1f}%")

    # V7: + Stale exit
    print("\n--- V7: + Stale Exit ---")
    t0 = time.time()
    r7 = simulate(sig3, conf3, price_arr, vol_arr, timestamps, symbols,
                  version_name='v7_stale_exit', use_atr_stops=True, high_arr=high_arr, low_arr=low_arr,
                  use_dd_scaling=True, use_circuit_breaker=True, use_regime_guard=True, use_stale_exit=True)
    save_version('v7_stale_exit', r7, {'description': 'V6 + stale exit after 500 bars'})
    all_results['v7_stale_exit'] = r7
    print(f"  Time: {time.time()-t0:.1f}s | Return: {r7['total_return_pct']:+.2f}% | Sharpe: {r7['sharpe_ratio']:.3f} | WinRate: {r7['win_rate_pct']:.1f}%")

    # V8: + Kelly sizing
    print("\n--- V8: + Kelly Criterion Sizing ---")
    t0 = time.time()
    r8 = simulate(sig3, conf3, price_arr, vol_arr, timestamps, symbols,
                  version_name='v8_kelly', use_atr_stops=True, high_arr=high_arr, low_arr=low_arr,
                  use_dd_scaling=True, use_circuit_breaker=True, use_regime_guard=True,
                  use_stale_exit=True, use_kelly=True)
    save_version('v8_kelly', r8, {'description': 'V7 + fractional Kelly position sizing'})
    all_results['v8_kelly'] = r8
    print(f"  Time: {time.time()-t0:.1f}s | Return: {r8['total_return_pct']:+.2f}% | Sharpe: {r8['sharpe_ratio']:.3f} | WinRate: {r8['win_rate_pct']:.1f}%")

    # V9: + Volume confirmation
    print("\n--- V9: + Volume Confirmation ---")
    t0 = time.time()
    r9 = simulate(sig3, conf3, price_arr, vol_arr, timestamps, symbols,
                  version_name='v9_volume_confirm', use_atr_stops=True, high_arr=high_arr, low_arr=low_arr,
                  use_dd_scaling=True, use_circuit_breaker=True, use_regime_guard=True,
                  use_stale_exit=True, use_kelly=True, use_vol_confirmation=True)
    save_version('v9_volume_confirm', r9, {'description': 'V8 + volume confirmation filter'})
    all_results['v9_volume_confirm'] = r9
    print(f"  Time: {time.time()-t0:.1f}s | Return: {r9['total_return_pct']:+.2f}% | Sharpe: {r9['sharpe_ratio']:.3f} | WinRate: {r9['win_rate_pct']:.1f}%")

    # Final comparison
    print_comparison(all_results)
