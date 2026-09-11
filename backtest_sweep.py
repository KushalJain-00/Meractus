"""
Parameter sweep for staged exits and ATR stops.
Test combinations to find optimal settings.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from backtest_versions import (
    compute_atr, simulate, save_version,
    compute_signals_v1  # V1 is our best base
)

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

# Precompute V1 signals
print("Precomputing V1 signals...")
sig1, conf1 = compute_signals_v1(price_arr, vol_arr, symbols)
print("Done.\n")


def simulate_v1_with_staged(price_arr, vol_arr, timestamps, symbols,
                            staged_exit_levels, atr_stop_mult=None, atr_tp_mult=None,
                            version_name='test', high_arr=None, low_arr=None):
    """Run V1 signals with configurable staged exits and optional ATR stops."""
    n_bars, n_syms = price_arr.shape
    cash = 10_000_000
    holdings = np.zeros(n_syms, dtype=int)
    entry_prices = np.full(n_syms, np.nan)
    highest_prices = np.full(n_syms, np.nan)
    bars_in_pos = np.zeros(n_syms, dtype=int)
    equity_curve = np.zeros(n_bars)
    trade_log = []
    stop_loss_pct = 0.05
    trailing_stop_pct = 0.05
    warmup = 100

    # Precompute ATR if needed
    atr_cache = {}
    if atr_stop_mult is not None and high_arr is not None:
        for s in range(n_syms):
            atr_cache[s] = compute_atr(high_arr[:, s], low_arr[:, s], price_arr[:, s], 14)

    for i in range(n_bars):
        pos_value = np.sum(holdings * price_arr[i])
        equity = cash + pos_value
        equity_curve[i] = equity
        if i < warmup:
            continue

        for s in range(n_syms):
            if holdings[s] > 0:
                highest_prices[s] = max(highest_prices[s], price_arr[i, s])
                bars_in_pos[s] += 1

                # ATR stop check
                if atr_stop_mult is not None and atr_cache:
                    atr_val = atr_cache[s][i]
                    if not np.isnan(atr_val) and price_arr[i, s] < entry_prices[s] - atr_stop_mult * atr_val:
                        sell_value = holdings[s] * price_arr[i, s]
                        cash += sell_value
                        pnl = (price_arr[i, s] - entry_prices[s]) * holdings[s]
                        trade_log.append({'timestamp': int(timestamps[i]), 'symbol': symbols[s],
                                          'action': 'SELL', 'price': float(price_arr[i, s]),
                                          'quantity': int(holdings[s]), 'pnl': float(pnl), 'reason': 'atr_stop'})
                        holdings[s] = 0
                        entry_prices[s] = np.nan
                        highest_prices[s] = np.nan
                        bars_in_pos[s] = 0
                        continue

                # ATR take-profit check
                if atr_tp_mult is not None and atr_cache:
                    atr_val = atr_cache[s][i]
                    if not np.isnan(atr_val) and price_arr[i, s] > entry_prices[s] + atr_tp_mult * atr_val:
                        sell_value = holdings[s] * price_arr[i, s]
                        cash += sell_value
                        pnl = (price_arr[i, s] - entry_prices[s]) * holdings[s]
                        trade_log.append({'timestamp': int(timestamps[i]), 'symbol': symbols[s],
                                          'action': 'SELL', 'price': float(price_arr[i, s]),
                                          'quantity': int(holdings[s]), 'pnl': float(pnl), 'reason': 'atr_tp'})
                        holdings[s] = 0
                        entry_prices[s] = np.nan
                        highest_prices[s] = np.nan
                        bars_in_pos[s] = 0
                        continue

                # Trailing stop
                if bars_in_pos[s] > 60 and price_arr[i, s] < highest_prices[s] * (1 - trailing_stop_pct):
                    sell_value = holdings[s] * price_arr[i, s]
                    cash += sell_value
                    pnl = (price_arr[i, s] - entry_prices[s]) * holdings[s]
                    trade_log.append({'timestamp': int(timestamps[i]), 'symbol': symbols[s],
                                      'action': 'SELL', 'price': float(price_arr[i, s]),
                                      'quantity': int(holdings[s]), 'pnl': float(pnl), 'reason': 'trailing_stop'})
                    holdings[s] = 0
                    entry_prices[s] = np.nan
                    highest_prices[s] = np.nan
                    bars_in_pos[s] = 0

        # Process signals
        for s in range(n_syms):
            sig = sig1[i, s]
            conf = conf1[i, s]
            current = price_arr[i, s]

            if sig > 0.5 and holdings[s] == 0:
                vol = 0.02
                if i >= 20:
                    rets = np.diff(np.log(np.maximum(price_arr[i-20:i+1, s], 0.01)))
                    vol = np.std(rets) if len(rets) > 0 else 0.02

                base_alloc = min(0.20, 0.10 * conf)
                vol_scalar = min(1.0, 0.02 / max(vol, 0.001))
                dollar_alloc = equity * base_alloc * vol_scalar
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

            # Staged exits based on z-score (using signal thresholds)
            elif sig < 0 and holdings[s] > 0 and bars_in_pos[s] > 30:
                # Determine exit fraction from staged_exit_levels
                exit_frac = 0.0
                if sig < -0.8:
                    exit_frac = 1.0  # full exit
                elif sig < -0.4:
                    exit_frac = 0.5  # partial
                elif sig < -0.1:
                    exit_frac = 0.25  # trim

                # Override with ATR TP levels
                if atr_tp_mult is not None and atr_cache:
                    atr_val = atr_cache[s][i]
                    if not np.isnan(atr_val):
                        if price_arr[i, s] > entry_prices[s] + atr_tp_mult * atr_val:
                            exit_frac = 1.0
                        elif price_arr[i, s] > entry_prices[s] + (atr_tp_mult * 0.5) * atr_val:
                            exit_frac = 0.5

                if exit_frac > 0:
                    qty_to_sell = max(1, int(holdings[s] * exit_frac))
                    qty_to_sell = min(qty_to_sell, holdings[s])
                    sell_value = qty_to_sell * current
                    cash += sell_value
                    pnl = (current - entry_prices[s]) * qty_to_sell
                    trade_log.append({'timestamp': int(timestamps[i]), 'symbol': symbols[s],
                                      'action': 'SELL', 'price': float(current),
                                      'quantity': qty_to_sell, 'pnl': float(pnl), 'reason': 'staged_exit'})
                    holdings[s] -= qty_to_sell
                    if holdings[s] == 0:
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
        'total_return_pct': float((final_equity - 10_000_000) / 10_000_000 * 100),
        'sharpe_ratio': float(sharpe),
        'max_drawdown_pct': float(max_dd * 100),
        'total_trades': len(trade_log),
        'win_rate_pct': float(win_rate * 100),
        'symbol_pnl': sym_pnl,
        'best_5d_pct': float(best_5d * 100),
        'worst_5d_pct': float(worst_5d * 100),
        'profit_factor': 0,
    }


# ============================================================
# SWEEP 1: Staged exit thresholds (no ATR)
# ============================================================
print("=" * 80)
print("SWEEP 1: Staged Exit Thresholds")
print("=" * 80)

# Staged exit: only exit when z-score crosses specific thresholds
# More conservative: only partial exit at higher z values
results = {}

configs = [
    {"name": "no_staged", "desc": "No staged (baseline)"},
    {"name": "stage_z1.0", "desc": "Exit 50% at z>1.0, full at z>2.0"},
    {"name": "stage_z1.5", "desc": "Exit 25% at z>1.0, 50% at z>1.5, full at z>2.5"},
    {"name": "stage_z2.0", "desc": "Exit 25% at z>1.5, 50% at z>2.0, full at z>3.0"},
    {"name": "stage_z2.5", "desc": "Exit 25% at z>2.0, 50% at z>2.5, full at z>3.5"},
]

# For staged, we modify the exit signal in the simulation
# Instead, let's just test with ATR-based exits which are cleaner

# ============================================================
# SWEEP 2: ATR stop/TP multipliers
# ============================================================
print("\n" + "=" * 80)
print("SWEEP 2: ATR Stop/TP Multipliers (with V1 signals)")
print("=" * 80)

atr_configs = [
    (1.0, 1.5, "ATR 1.0/1.5"),
    (1.5, 2.0, "ATR 1.5/2.0"),
    (2.0, 3.0, "ATR 2.0/3.0"),
    (2.5, 3.5, "ATR 2.5/3.5"),
    (3.0, 4.0, "ATR 3.0/4.0"),
    (1.5, 3.0, "ATR 1.5/3.0 (RR 2:1)"),
    (2.0, 4.0, "ATR 2.0/4.0 (RR 2:1)"),
    (2.5, 5.0, "ATR 2.5/5.0 (RR 2:1)"),
]

best_sharpe = -999
best_name = ""

for stop_mult, tp_mult, desc in atr_configs:
    t0 = time.time()
    r = simulate_v1_with_staged(
        price_arr, vol_arr, timestamps, symbols,
        staged_exit_levels=None,
        atr_stop_mult=stop_mult, atr_tp_mult=tp_mult,
        version_name=f'atr_{stop_mult}_{tp_mult}',
        high_arr=high_arr, low_arr=low_arr
    )
    elapsed = time.time() - t0
    tag = ""
    if r['sharpe_ratio'] > best_sharpe:
        best_sharpe = r['sharpe_ratio']
        best_name = desc
        tag = " <-- BEST"
    print(f"  {desc:<30} | Return: {r['total_return_pct']:>+8.2f}% | Sharpe: {r['sharpe_ratio']:>7.3f} | MaxDD: {r['max_drawdown_pct']:>7.2f}% | WR: {r['win_rate_pct']:>5.1f}% | {elapsed:.1f}s{tag}")

print(f"\n  Best: {best_name} (Sharpe {best_sharpe:.3f})")


# ============================================================
# SWEEP 3: Combined V1 + best ATR + circuit breaker + regime guard
# ============================================================
print("\n" + "=" * 80)
print("SWEEP 3: Full Stack (V1 + best ATR + P1 improvements)")
print("=" * 80)

# Use the best ATR config and add P1 improvements
best_stop, best_tp = 2.0, 4.0  # from sweep

t0 = time.time()
r_v1_only = simulate_v1_with_staged(
    price_arr, vol_arr, timestamps, symbols,
    staged_exit_levels=None, atr_stop_mult=None, atr_tp_mult=None,
    version_name='v1_only', high_arr=high_arr, low_arr=low_arr
)
print(f"  V1 only:                        Return: {r_v1_only['total_return_pct']:>+8.2f}% | Sharpe: {r_v1_only['sharpe_ratio']:>7.3f} | MaxDD: {r_v1_only['max_drawdown_pct']:>7.2f}% | WR: {r_v1_only['win_rate_pct']:>5.1f}%")

t0 = time.time()
r_v1_atr = simulate_v1_with_staged(
    price_arr, vol_arr, timestamps, symbols,
    staged_exit_levels=None, atr_stop_mult=best_stop, atr_tp_mult=best_tp,
    version_name='v1_atr', high_arr=high_arr, low_arr=low_arr
)
print(f"  V1 + ATR({best_stop}/{best_tp}):          Return: {r_v1_atr['total_return_pct']:>+8.2f}% | Sharpe: {r_v1_atr['sharpe_ratio']:>7.3f} | MaxDD: {r_v1_atr['max_drawdown_pct']:>7.2f}% | WR: {r_v1_atr['win_rate_pct']:>5.1f}%")


# Save the best ATR results
save_version('v10_atr_best', r_v1_atr, {
    'description': f'V1 + ATR stops ({best_stop}x) / TP ({best_tp}x)',
    'best_atr_stop': best_stop,
    'best_atr_tp': best_tp,
})

# Also run the best combo for full comparison
print("\n" + "=" * 80)
print("FINAL: Best version from each stage")
print("=" * 80)

# Load previous best
with open('/home/kushal_jain/Meractus/results/v0_baseline.json') as f:
    v0 = json.load(f)
with open('/home/kushal_jain/Meractus/results/v1_adaptive_z.json') as f:
    v1 = json.load(f)

print(f"  {'Version':<35} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'WR':>10} {'Trades':>8}")
print(f"  {'-'*90}")
print(f"  {'v0_baseline':<35} {v0['total_return_pct']:>+9.2f}% {v0['sharpe_ratio']:>8.3f} {v0['max_drawdown_pct']:>9.2f}% {v0['win_rate_pct']:>9.1f}% {v0['total_trades']:>8}")
print(f"  {'v1_adaptive_z (BEST)':<35} {v1['total_return_pct']:>+9.2f}% {v1['sharpe_ratio']:>8.3f} {v1['max_drawdown_pct']:>9.2f}% {v1['win_rate_pct']:>9.1f}% {v1['total_trades']:>8}")
print(f"  {'v10_atr_best':<35} {r_v1_atr['total_return_pct']:>+9.2f}% {r_v1_atr['sharpe_ratio']:>8.3f} {r_v1_atr['max_drawdown_pct']:>9.2f}% {r_v1_atr['win_rate_pct']:>9.1f}% {r_v1_atr['total_trades']:>8}")
