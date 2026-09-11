"""
Final comparison: V1 adaptive z in ensemble vs all versions.
Also tests a few more micro-improvements on V1.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from backtest_versions import (
    compute_atr, simulate, save_version,
    compute_signals_v1, compute_signals_v0
)
from backtest_sweep import simulate_v1_with_staged

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

print(f"Data: {len(symbols)} symbols, {len(timestamps)} bars\n")

# ============================================================
# Compute V1 signals (our winner)
# ============================================================
print("Computing V1 signals...")
sig1, conf1 = compute_signals_v1(price_arr, vol_arr, symbols)

# ============================================================
# Test: V1 with tighter RSI filter (RSI < 30 instead of 34)
# ============================================================
print("\n--- Micro-test: V1 with RSI < 30 ---")
sig_rsi30 = sig1.copy()
conf_rsi30 = conf1.copy()

# Recompute with stricter RSI
n_bars, n_syms = price_arr.shape
for s in range(n_syms):
    p = price_arr[:, s]
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

    # Override: only buy if RSI < 30 (stricter than 34)
    for i in range(40, n_bars):
        if sig_rsi30[i, s] > 0.5 and rsi_vals[i] >= 30:
            sig_rsi30[i, s] = 0
            conf_rsi30[i, s] = 0

r_rsi30 = simulate(sig_rsi30, conf_rsi30, price_arr, vol_arr, timestamps, symbols, version_name='v1_rsi30')
print(f"  Return: {r_rsi30['total_return_pct']:+.2f}% | Sharpe: {r_rsi30['sharpe_ratio']:.3f} | MaxDD: {r_rsi30['max_drawdown_pct']:.2f}% | WR: {r_rsi30['win_rate_pct']:.1f}% | Trades: {r_rsi30['total_trades']}")

# ============================================================
# Test: V1 with RSI < 28 (even stricter)
# ============================================================
sig_rsi28 = sig1.copy()
conf_rsi28 = conf1.copy()
for s in range(n_syms):
    p = price_arr[:, s]
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
        if sig_rsi28[i, s] > 0.5 and rsi_vals[i] >= 28:
            sig_rsi28[i, s] = 0
            conf_rsi28[i, s] = 0

r_rsi28 = simulate(sig_rsi28, conf_rsi28, price_arr, vol_arr, timestamps, symbols, version_name='v1_rsi28')
print(f"  Return: {r_rsi28['total_return_pct']:+.2f}% | Sharpe: {r_rsi28['sharpe_ratio']:.3f} | MaxDD: {r_rsi28['max_drawdown_pct']:.2f}% | WR: {r_rsi28['win_rate_pct']:.1f}% | Trades: {r_rsi28['total_trades']}")

# ============================================================
# Test: V1 with higher z-score requirement for buy (z < -2.5)
# ============================================================
print("\n--- Micro-test: V1 with z < -2.5 ---")
sig_tight = sig1.copy()
conf_tight = conf1.copy()
for s in range(n_syms):
    p = price_arr[:, s]
    window = 20
    roll_mean = np.full(n_bars, np.nan)
    roll_std = np.full(n_bars, np.nan)
    for i in range(window - 1, n_bars):
        roll_mean[i] = np.mean(p[i - window + 1:i + 1])
        roll_std[i] = np.std(p[i - window + 1:i + 1], ddof=1)
    z = np.where(roll_std > 0, (p - roll_mean) / roll_std, 0.0)

    vol_window = 100
    vol_series = np.full(n_bars, np.nan)
    for i in range(vol_window, n_bars):
        rets = np.diff(np.log(np.maximum(p[i - vol_window:i + 1], 0.01)))
        vol_series[i] = np.std(rets)

    adaptive_z = np.full(n_bars, -2.5)  # tighter base
    for i in range(vol_window + window, n_bars):
        if not np.isnan(vol_series[i]):
            hist_vol = vol_series[vol_window:i]
            hist_vol = hist_vol[~np.isnan(hist_vol)]
            if len(hist_vol) > 10:
                pct = np.mean(hist_vol < vol_series[i])
                adaptive_z[i] = -2.5 + (pct - 0.5) * 1.5  # range: -3.25 to -1.75

    for i in range(40, n_bars):
        if np.isnan(z[i]):
            continue
        if z[i] < adaptive_z[i]:
            sig_tight[i, s] = 1.0
            conf_tight[i, s] = min(1.0, abs(z[i]) / 3.5)
        elif z[i] > 0:
            sig_tight[i, s] = -0.5
            conf_tight[i, s] = 0.5

r_tight = simulate(sig_tight, conf_tight, price_arr, vol_arr, timestamps, symbols, version_name='v1_tight_z')
print(f"  Return: {r_tight['total_return_pct']:+.2f}% | Sharpe: {r_tight['sharpe_ratio']:.3f} | MaxDD: {r_tight['max_drawdown_pct']:.2f}% | WR: {r_tight['win_rate_pct']:.1f}% | Trades: {r_tight['total_trades']}")

# ============================================================
# Test: V1 with circuit breaker + drawdown scaling (P1 improvements on V1)
# ============================================================
print("\n--- V1 + P1 improvements ---")
# Reuse V1 simulation with circuit breaker and dd scaling
from backtest_versions import simulate as sim_full

r_v1_p1 = sim_full(sig1, conf1, price_arr, vol_arr, timestamps, symbols,
                    version_name='v1_p1', use_dd_scaling=True, use_circuit_breaker=True,
                    use_regime_guard=True, use_stale_exit=True)
print(f"  V1 + all P1:   Return: {r_v1_p1['total_return_pct']:+.2f}% | Sharpe: {r_v1_p1['sharpe_ratio']:.3f} | MaxDD: {r_v1_p1['max_drawdown_pct']:.2f}% | WR: {r_v1_p1['win_rate_pct']:.1f}%")

# ============================================================
# FINAL COMPARISON TABLE
# ============================================================
print("\n" + "=" * 110)
print("FINAL RESULTS - ALL VERSIONS")
print("=" * 110)

# Load all saved results
results_dir = '/home/kushal_jain/Meractus/results'
all_r = {}
for fname in sorted(os.listdir(results_dir)):
    if fname.endswith('.json'):
        with open(os.path.join(results_dir, fname)) as f:
            all_r[fname.replace('.json', '')] = json.load(f)

# Add non-saved results
all_r['v1_rsi30'] = r_rsi30
all_r['v1_rsi28'] = r_rsi28
all_r['v1_tight_z'] = r_tight
all_r['v1_p1'] = r_v1_p1

print(f"\n{'Version':<35} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'WinRate':>10} {'Trades':>8}")
print("-" * 95)
for name in ['v0_baseline', 'v1_adaptive_z', 'v1_rsi30', 'v1_rsi28', 'v1_tight_z', 'v1_p1', 'v2_staged_exit', 'v3_atr_stops', 'v10_atr_best']:
    if name in all_r:
        r = all_r[name]
        marker = " <-- WINNER" if name == 'v1_adaptive_z' else ""
        print(f"{name:<35} {r['total_return_pct']:>+9.2f}% {r['sharpe_ratio']:>8.3f} {r['max_drawdown_pct']:>9.2f}% {r['win_rate_pct']:>9.1f}% {r['total_trades']:>8}{marker}")

# Save the overall winner
save_version('WINNER_v1_adaptive_z', all_r['v1_adaptive_z'], {
    'description': 'Adaptive z-score thresholds based on volatility regime',
    'key_insight': 'Volatile periods use lower z threshold (-1.5), quiet periods use higher (-2.5)',
    'improvement_over_baseline': "+{:.1f}% return, +{:.3f} Sharpe".format(
        all_r['v1_adaptive_z']['total_return_pct'] - all_r['v0_baseline']['total_return_pct'],
        all_r['v1_adaptive_z']['sharpe_ratio'] - all_r['v0_baseline']['sharpe_ratio']
    )
})
print(f"\nWinner saved: WINNER_v1_adaptive_z.json")
