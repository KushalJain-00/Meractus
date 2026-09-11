"""
Fast vectorized backtesting engine.
Instead of calling strategy.signal() per tick per symbol (slow),
pre-compute all signals in batch using numpy/pandas vectorization.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from typing import Dict, List
import json
import time
from features import ema, rsi, bollinger_bands, macd, volatility, detect_regime, momentum_score


def fast_backtest(csv_path: str, initial_capital: float = 10_000_000.0,
                  strategy_name: str = 'momentum') -> Dict:
    """
    Vectorized backtest: pre-compute all signals, then simulate trades.
    ~100x faster than per-tick loop.
    """
    print(f"\n{'='*60}")
    print(f"Fast backtest: {strategy_name}")
    print(f"{'='*60}")

    t0 = time.time()

    # Load and pivot data
    df = pd.read_csv(csv_path)
    df = df.sort_values(['symbol', 'timestamp'])

    # Pivot to wide format
    pivot_price = df.pivot_table(index='timestamp', columns='symbol', values='price')
    pivot_vol = df.pivot_table(index='timestamp', columns='symbol', values='volume')
    pivot_high = df.pivot_table(index='timestamp', columns='symbol', values='high')
    pivot_low = df.pivot_table(index='timestamp', columns='symbol', values='low')

    timestamps = pivot_price.index.values
    symbols = pivot_price.columns.tolist()
    n_bars = len(timestamps)

    print(f"Loaded: {len(symbols)} symbols, {n_bars} timestamps")
    print(f"Load time: {time.time()-t0:.1f}s")

    t1 = time.time()

    # Pre-compute features for all symbols
    price_arr = pivot_price.values  # (n_bars, n_syms)
    vol_arr = pivot_vol.values.astype(float)
    high_arr = pivot_high.values
    low_arr = pivot_low.values

    # === STRATEGY 1: MOMENTUM ===
    if strategy_name == 'momentum':
        signals = _compute_momentum_signals(price_arr, vol_arr, symbols)

    # === STRATEGY 2: MEAN REVERSION ===
    elif strategy_name == 'mean_reversion':
        signals = _compute_mr_signals(price_arr, vol_arr, symbols)

    # === STRATEGY 3: COMPOSITE ===
    elif strategy_name == 'composite':
        signals = _compute_composite_signals(price_arr, vol_arr, symbols)

    print(f"Signal computation: {time.time()-t1:.1f}s")

    t2 = time.time()

    # === SIMULATE TRADES ===
    result = _simulate_trades(signals, price_arr, vol_arr, timestamps, symbols,
                              initial_capital, strategy_name)

    print(f"Simulation: {time.time()-t2:.1f}s")
    print(f"Total: {time.time()-t0:.1f}s")

    return result


def _ema_fast(arr, span):
    """Vectorized EMA across axis=0."""
    out = np.full_like(arr, np.nan, dtype=float)
    alpha = 2.0 / (span + 1)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _compute_momentum_signals(price_arr, vol_arr, symbols):
    """Compute momentum signals for all symbols at all timestamps."""
    n_bars, n_syms = price_arr.shape
    # Signal matrix: +1 = BUY, -1 = SELL, 0 = HOLD
    signals = np.zeros((n_bars, n_syms), dtype=float)
    confidences = np.zeros((n_bars, n_syms), dtype=float)

    for s in range(n_syms):
        p = price_arr[:, s]

        # EMAs
        ema_fast = np.full(n_bars, np.nan)
        ema_slow = np.full(n_bars, np.nan)
        alpha_f = 2.0 / 11  # span=10
        alpha_s = 2.0 / 31  # span=30

        ef = p[0]
        es = p[0]
        for i in range(n_bars):
            ef = alpha_f * p[i] + (1 - alpha_f) * ef
            es = alpha_s * p[i] + (1 - alpha_s) * es
            ema_fast[i] = ef
            ema_slow[i] = es

        # Regime: use first 2000 bars to classify, then cache
        regime = 'trending'
        if n_bars > 400:
            r = detect_regime(p[:2000], window=500, threshold=0.3)
            regime = r

        # RSI
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
                if al > 0:
                    rsi_vals[i + 1] = 100.0 - 100.0 / (1.0 + ag / al)
                else:
                    rsi_vals[i + 1] = 100.0

        # Generate signals
        warmup = 60
        for i in range(warmup, n_bars):
            if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
                continue

            # Previous values
            ef_prev = ema_fast[i - 1] if i > 0 else ema_fast[i]
            es_prev = ema_slow[i - 1] if i > 0 else ema_slow[i]

            # Golden cross
            if ef_prev <= es_prev and ema_fast[i] > ema_slow[i]:
                if regime == 'trending' and rsi_vals[i] < 74:
                    conf = min(1.0, abs(ema_fast[i] - ema_slow[i]) / ema_slow[i] * 100)
                    signals[i, s] = 1.0
                    confidences[i, s] = conf

            # Death cross
            if ef_prev >= es_prev and ema_fast[i] < ema_slow[i]:
                signals[i, s] = -1.0
                confidences[i, s] = 0.8

    return {'signals': signals, 'confidences': confidences, 'type': 'momentum'}


def _compute_mr_signals(price_arr, vol_arr, symbols):
    """Compute mean reversion signals."""
    n_bars, n_syms = price_arr.shape
    signals = np.zeros((n_bars, n_syms), dtype=float)
    confidences = np.zeros((n_bars, n_syms), dtype=float)

    for s in range(n_syms):
        p = price_arr[:, s]
        window = 20

        # Rolling mean and std
        roll_mean = np.full(n_bars, np.nan)
        roll_std = np.full(n_bars, np.nan)
        for i in range(window - 1, n_bars):
            roll_mean[i] = np.mean(p[i - window + 1:i + 1])
            roll_std[i] = np.std(p[i - window + 1:i + 1], ddof=1)

        # Z-scores
        z = np.where(roll_std > 0, (p - roll_mean) / roll_std, 0.0)

        # RSI
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
                if al > 0:
                    rsi_vals[i + 1] = 100.0 - 100.0 / (1.0 + ag / al)
                else:
                    rsi_vals[i + 1] = 100.0

        warmup = 40
        for i in range(warmup, n_bars):
            if np.isnan(z[i]):
                continue

            # Flash crash detection
            if i >= 60:
                old = p[i - 60]
                if old > 0 and (p[i] - old) / old < -0.10:
                    signals[i, s] = 1.0
                    confidences[i, s] = 0.8
                    continue

            # Oversold
            if z[i] < -2.0 and rsi_vals[i] < 34:
                conf = min(1.0, abs(z[i]) / 3.0)
                signals[i, s] = 1.0
                confidences[i, s] = conf

            # Exit at mean
            elif z[i] > 0:
                signals[i, s] = -0.5
                confidences[i, s] = 0.5

    return {'signals': signals, 'confidences': confidences, 'type': 'mean_reversion'}


def _compute_composite_signals(price_arr, vol_arr, symbols):
    """Compute composite RSI+MACD signals."""
    n_bars, n_syms = price_arr.shape
    signals = np.zeros((n_bars, n_syms), dtype=float)
    confidences = np.zeros((n_bars, n_syms), dtype=float)

    for s in range(n_syms):
        p = price_arr[:, s]

        # RSI
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
                if al > 0:
                    rsi_vals[i + 1] = 100.0 - 100.0 / (1.0 + ag / al)
                else:
                    rsi_vals[i + 1] = 100.0

        # MACD
        ema12 = np.full(n_bars, np.nan)
        ema26 = np.full(n_bars, np.nan)
        a12 = 2.0 / 13
        a26 = 2.0 / 27
        e12 = p[0]
        e26 = p[0]
        for i in range(n_bars):
            e12 = a12 * p[i] + (1 - a12) * e12
            e26 = a26 * p[i] + (1 - a26) * e26
            ema12[i] = e12
            ema26[i] = e26

        macd_line = ema12 - ema26

        warmup = 40
        for i in range(warmup, n_bars):
            if i < 1:
                continue
            prev_macd = macd_line[i - 1]
            curr_macd = macd_line[i]

            # MACD crosses above zero + RSI OK
            if prev_macd < 0 and curr_macd > 0 and rsi_vals[i] < 70:
                signals[i, s] = 1.0
                confidences[i, s] = 0.6

            # MACD crosses below zero
            if prev_macd > 0 and curr_macd < 0:
                signals[i, s] = -1.0
                confidences[i, s] = 0.6

    return {'signals': signals, 'confidences': confidences, 'type': 'composite'}


def _simulate_trades(signals_dict, price_arr, vol_arr, timestamps, symbols,
                     initial_capital, strategy_name):
    """
    Simulate trades using pre-computed signals.
    Uses trailing stops and position sizing.
    """
    n_bars, n_syms = price_arr.shape
    signals = signals_dict['signals']
    confidences = signals_dict['confidences']

    # State
    cash = initial_capital
    holdings = np.zeros(n_syms, dtype=int)  # shares per symbol
    entry_prices = np.full(n_syms, np.nan)
    highest_prices = np.full(n_syms, np.nan)
    cost_basis = np.full(n_syms, np.nan)  # total cost per position

    equity_curve = np.zeros(n_bars)
    trade_log = []
    warmup = 100

    # Strategy parameters
    stop_loss_pct = 0.05
    trailing_stop_pct = 0.05
    max_exposure_pct = 0.20
    cash_reserve_pct = 0.10
    min_hold_bars = 60
    bars_in_pos = np.zeros(n_syms, dtype=int)

    for i in range(n_bars):
        # Current portfolio value
        pos_value = np.sum(holdings * price_arr[i])
        equity = cash + pos_value
        equity_curve[i] = equity

        if i < warmup:
            continue

        # Update highest prices
        for s in range(n_syms):
            if holdings[s] > 0:
                highest_prices[s] = max(highest_prices[s], price_arr[i, s])
                bars_in_pos[s] += 1

        # Risk checks first: stop-loss and trailing stop
        for s in range(n_syms):
            if holdings[s] > 0 and not np.isnan(entry_prices[s]):
                current = price_arr[i, s]

                # Trailing stop
                if current < highest_prices[s] * (1 - trailing_stop_pct) and bars_in_pos[s] > min_hold_bars:
                    sell_value = holdings[s] * current
                    cash += sell_value
                    pnl = (current - entry_prices[s]) * holdings[s]
                    trade_log.append({
                        'timestamp': int(timestamps[i]), 'symbol': symbols[s],
                        'action': 'SELL', 'price': float(current),
                        'quantity': int(holdings[s]), 'pnl': float(pnl),
                        'reason': 'trailing_stop'
                    })
                    holdings[s] = 0
                    entry_prices[s] = np.nan
                    highest_prices[s] = np.nan
                    bars_in_pos[s] = 0
                    continue

                # Hard stop
                if current < entry_prices[s] * (1 - stop_loss_pct) and bars_in_pos[s] > min_hold_bars:
                    sell_value = holdings[s] * current
                    cash += sell_value
                    pnl = (current - entry_prices[s]) * holdings[s]
                    trade_log.append({
                        'timestamp': int(timestamps[i]), 'symbol': symbols[s],
                        'action': 'SELL', 'price': float(current),
                        'quantity': int(holdings[s]), 'pnl': float(pnl),
                        'reason': 'stop_loss'
                    })
                    holdings[s] = 0
                    entry_prices[s] = np.nan
                    highest_prices[s] = np.nan
                    bars_in_pos[s] = 0

        # Process signals
        for s in range(n_syms):
            sig = signals[i, s]
            conf = confidences[i, s]
            current = price_arr[i, s]

            if sig > 0.5 and holdings[s] == 0:
                # BUY
                # Position sizing
                vol = 0.02
                if i >= 20:
                    rets = np.diff(np.log(np.maximum(price_arr[i-20:i+1, s], 0.01)))
                    vol = np.std(rets) if len(rets) > 0 else 0.02

                base_alloc = min(max_exposure_pct, 0.10 * conf)
                vol_scalar = min(1.0, 0.02 / max(vol, 0.001))
                dollar_alloc = equity * base_alloc * vol_scalar
                available = cash - equity * cash_reserve_pct
                dollar_alloc = min(dollar_alloc, max(available, 0))

                qty = int(dollar_alloc / current) if current > 0 else 0
                if qty > 0:
                    cost = qty * current
                    if cost <= cash * (1 - cash_reserve_pct):
                        cash -= cost
                        holdings[s] = qty
                        entry_prices[s] = current
                        highest_prices[s] = current
                        bars_in_pos[s] = 0
                        trade_log.append({
                            'timestamp': int(timestamps[i]), 'symbol': symbols[s],
                            'action': 'BUY', 'price': float(current),
                            'quantity': qty, 'confidence': float(conf),
                            'reason': 'signal'
                        })

            elif sig < -0.5 and holdings[s] > 0 and bars_in_pos[s] > min_hold_bars:
                # SELL
                sell_value = holdings[s] * current
                cash += sell_value
                pnl = (current - entry_prices[s]) * holdings[s]
                trade_log.append({
                    'timestamp': int(timestamps[i]), 'symbol': symbols[s],
                    'action': 'SELL', 'price': float(current),
                    'quantity': int(holdings[s]), 'pnl': float(pnl),
                    'reason': 'signal'
                })
                holdings[s] = 0
                entry_prices[s] = np.nan
                highest_prices[s] = np.nan
                bars_in_pos[s] = 0

    # Final equity
    final_equity = equity_curve[-1]

    # Metrics
    returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1)
    returns = returns[~np.isnan(returns)]
    returns = returns[np.isfinite(returns)]

    # Daily returns (390 bars per day)
    bars_per_day = 390
    if len(returns) > bars_per_day:
        n_days = len(returns) // bars_per_day
        daily_rets = np.array([
            np.mean(returns[i * bars_per_day:(i + 1) * bars_per_day])
            for i in range(n_days)
        ])
    else:
        daily_rets = returns

    sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0

    # Max drawdown
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / np.maximum(cummax, 1)
    max_dd = np.min(drawdown)

    # Win rate
    sell_trades = [t for t in trade_log if t['action'] == 'SELL']
    buy_trades = [t for t in trade_log if t['action'] == 'BUY']
    n_roundtrips = min(len(buy_trades), len(sell_trades))
    wins = sum(1 for i in range(n_roundtrips) if sell_trades[i]['price'] > buy_trades[i]['price'])
    win_rate = wins / n_roundtrips if n_roundtrips > 0 else 0

    # Per-symbol PnL
    sym_pnl = {}
    for t in trade_log:
        if 'pnl' in t:
            sym = t['symbol']
            sym_pnl[sym] = sym_pnl.get(sym, 0) + t['pnl']

    # Best/worst 5-day window
    window_size = 5 * 390
    best_5d = -np.inf
    worst_5d = np.inf
    for start in range(0, len(equity_curve) - window_size, window_size):
        end = start + window_size
        ret = (equity_curve[end] - equity_curve[start]) / equity_curve[start]
        best_5d = max(best_5d, ret)
        worst_5d = min(worst_5d, ret)

    result = {
        'strategy': strategy_name,
        'final_equity': float(final_equity),
        'total_return_pct': float((final_equity - initial_capital) / initial_capital * 100),
        'sharpe_ratio': float(sharpe),
        'max_drawdown_pct': float(max_dd * 100),
        'total_trades': len(trade_log),
        'win_rate_pct': float(win_rate * 100),
        'symbol_pnl': sym_pnl,
        'best_5d_pct': float(best_5d * 100),
        'worst_5d_pct': float(worst_5d * 100),
        'equity_curve': equity_curve.tolist(),
        'trade_log': trade_log,
    }

    print(f"\n--- {strategy_name} Results ---")
    print(f"Final Equity:  ${final_equity:,.2f}")
    print(f"Total Return:  {result['total_return_pct']:+.2f}%")
    print(f"Sharpe Ratio:  {sharpe:.3f}")
    print(f"Max Drawdown:  {max_dd*100:.2f}%")
    print(f"Total Trades:  {len(trade_log)}")
    print(f"Win Rate:      {win_rate*100:.1f}%")
    if sym_pnl:
        best_sym = max(sym_pnl, key=sym_pnl.get)
        worst_sym = min(sym_pnl, key=sym_pnl.get)
        print(f"Best Symbol:   {best_sym} (${sym_pnl[best_sym]:+,.0f})")
        print(f"Worst Symbol:  {worst_sym} (${sym_pnl[worst_sym]:+,.0f})")

    return result


if __name__ == '__main__':
    csv_path = '/home/kushal_jain/Meractus/paper_dataset(1).csv'
    results = {}

    for strat in ['momentum', 'mean_reversion', 'composite']:
        results[strat] = fast_backtest(csv_path, strategy_name=strat)

    # Comparison table
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)
    print(f"{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'Trades':>8} {'WinRate':>10} {'Best5D':>10} {'Worst5D':>10}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{r['strategy']:<25} {r['total_return_pct']:>+9.2f}% {r['sharpe_ratio']:>8.3f} {r['max_drawdown_pct']:>9.2f}% {r['total_trades']:>8} {r['win_rate_pct']:>9.1f}% {r['best_5d_pct']:>+9.2f}% {r['worst_5d_pct']:>+9.2f}%")
    print("=" * 80)

    # Save
    save_data = {}
    for name, r in results.items():
        save_data[name] = {k: v for k, v in r.items() if k not in ['equity_curve', 'trade_log']}
    with open('/home/kushal_jain/Meractus/backtest_results.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print("\nResults saved to backtest_results.json")
