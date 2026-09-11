#!/usr/bin/env python3
"""
Ensemble voting strategy: combines momentum, mean-reversion, and composite signals
with weighted voting. Reuses backtest_fast.py signal functions and trade simulation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from typing import Dict
import json
import time

from backtest_fast import (
    _compute_momentum_signals,
    _compute_mr_signals,
    _compute_composite_signals,
    _simulate_trades,
)


def _combine_signals(mom, mr, comp, w_mr, w_comp, w_mom, buy_thresh, sell_thresh):
    """Vectorized weighted voting across pre-computed signal sets."""
    n_bars, n_syms = mr['signals'].shape
    tw = w_mr + w_comp + w_mom
    w_mr, w_comp, w_mom = w_mr/tw, w_comp/tw, w_mom/tw

    # Weighted score: signal * confidence * weight
    score = (w_mr * mr['signals'] * mr['confidences'] +
             w_comp * comp['signals'] * comp['confidences'] +
             w_mom * mom['signals'] * mom['confidences'])

    signals = np.zeros_like(score)
    confidences = np.zeros_like(score)

    buy_mask = score > buy_thresh
    sell_mask = score < sell_thresh

    # Any model strong-sell veto
    veto = ((mr['signals'] < -0.5) & (mr['confidences'] > 0.7) |
            (comp['signals'] < -0.5) & (comp['confidences'] > 0.7) |
            (mom['signals'] < -0.5) & (mom['confidences'] > 0.7))

    signals[buy_mask] = 1.0
    confidences[buy_mask] = np.minimum(1.0, np.abs(score[buy_mask]))

    signals[sell_mask] = -1.0
    confidences[sell_mask] = np.minimum(1.0, np.abs(score[sell_mask]))

    veto_mask = veto & ~buy_mask & ~sell_mask
    signals[veto_mask] = -0.8
    confidences[veto_mask] = 0.7

    return {'signals': signals, 'confidences': confidences, 'type': 'ensemble'}


def run_backtest(csv_path, initial_capital=10_000_000.0, strategy_name='ensemble',
                 w_mr=0.45, w_comp=0.35, w_mom=0.20,
                 buy_thresh=0.5, sell_thresh=-0.3):
    """Run a single backtest with given parameters."""
    print(f"\n{'='*60}")
    print(f"Backtest: {strategy_name}")
    if 'ensemble' in strategy_name:
        print(f"  Weights: MR={w_mr:.2f} Comp={w_comp:.2f} Mom={w_mom:.2f}")
        print(f"  Thresholds: buy>{buy_thresh:.2f} sell<{sell_thresh:.2f}")
    print(f"{'='*60}")

    t0 = time.time()
    df = pd.read_csv(csv_path)
    df = df.sort_values(['symbol', 'timestamp'])
    pivot_price = df.pivot_table(index='timestamp', columns='symbol', values='price')
    pivot_vol = df.pivot_table(index='timestamp', columns='symbol', values='volume')
    timestamps = pivot_price.index.values
    symbols = pivot_price.columns.tolist()
    price_arr = pivot_price.values
    vol_arr = pivot_vol.values.astype(float)
    print(f"Loaded: {len(symbols)} symbols, {len(timestamps)} timestamps")

    t1 = time.time()
    if strategy_name == 'momentum':
        signals = _compute_momentum_signals(price_arr, vol_arr, symbols)
    elif strategy_name == 'mean_reversion':
        signals = _compute_mr_signals(price_arr, vol_arr, symbols)
    elif strategy_name == 'composite':
        signals = _compute_composite_signals(price_arr, vol_arr, symbols)
    else:
        mom = _compute_momentum_signals(price_arr, vol_arr, symbols)
        mr = _compute_mr_signals(price_arr, vol_arr, symbols)
        comp = _compute_composite_signals(price_arr, vol_arr, symbols)
        signals = _combine_signals(mom, mr, comp, w_mr, w_comp, w_mom, buy_thresh, sell_thresh)
    print(f"Signal computation: {time.time()-t1:.1f}s")

    t2 = time.time()
    result = _simulate_trades(signals, price_arr, vol_arr, timestamps, symbols,
                              initial_capital, strategy_name)
    print(f"Simulation: {time.time()-t2:.1f}s")
    print(f"Total: {time.time()-t0:.1f}s")
    return result


def parameter_sweep(csv_path, initial_capital=10_000_000.0):
    """Sweep weight combinations and thresholds — vectorized, no re-loading."""
    print("\n" + "=" * 80)
    print("PARAMETER SWEEP — Ensemble Weight & Threshold Combinations")
    print("=" * 80)

    t_start = time.time()

    # Load data once
    df = pd.read_csv(csv_path)
    df = df.sort_values(['symbol', 'timestamp'])
    pivot_price = df.pivot_table(index='timestamp', columns='symbol', values='price')
    pivot_vol = df.pivot_table(index='timestamp', columns='symbol', values='volume')
    timestamps = pivot_price.index.values
    symbols = pivot_price.columns.tolist()
    price_arr = pivot_price.values
    vol_arr = pivot_vol.values.astype(float)

    # Pre-compute all signal sets once
    print("Pre-computing signal sets (this is the expensive part)...")
    t0 = time.time()
    mom = _compute_momentum_signals(price_arr, vol_arr, symbols)
    mr = _compute_mr_signals(price_arr, vol_arr, symbols)
    comp = _compute_composite_signals(price_arr, vol_arr, symbols)
    print(f"Signal pre-computation: {time.time()-t0:.1f}s")

    # Sweep — combine_signals is vectorized, _simulate_trades is the bottleneck
    weight_combos = [
        (0.45, 0.35, 0.20),  # default
        (0.50, 0.30, 0.20),  # MR heavy
        (0.33, 0.33, 0.34),  # equal
        (0.55, 0.25, 0.20),  # MR dominant
        (0.30, 0.50, 0.20),  # Comp dominant
        (0.40, 0.30, 0.30),  # boost momentum
        (0.60, 0.25, 0.15),  # MR very heavy
    ]
    buy_thresholds = [0.3, 0.5, 0.7]
    sell_thresholds = [-0.2, -0.3, -0.5]

    results = []
    total = len(weight_combos) * len(buy_thresholds) * len(sell_thresholds)
    count = 0

    for w_mr, w_comp, w_mom in weight_combos:
        for buy_t in buy_thresholds:
            for sell_t in sell_thresholds:
                count += 1
                sig_dict = _combine_signals(mom, mr, comp, w_mr, w_comp, w_mom, buy_t, sell_t)
                r = _simulate_trades(sig_dict, price_arr, vol_arr, timestamps, symbols,
                                     initial_capital, f"ens_{count}")
                r['params'] = {'w_mr': w_mr, 'w_comp': w_comp, 'w_mom': w_mom,
                               'buy_thresh': buy_t, 'sell_thresh': sell_t}
                results.append(r)
                if count % 10 == 0:
                    print(f"  [{count}/{total}] done... ({time.time()-t_start:.0f}s elapsed)")

    results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)

    print(f"\nSweep complete: {total} combinations in {time.time()-t_start:.1f}s")
    print(f"\nTop 10 by Sharpe Ratio:")
    print(f"{'Config':<60} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'Trades':>8} {'WinRate':>10}")
    print("-" * 106)
    for r in results[:10]:
        p = r['params']
        cfg = f"MR={p['w_mr']:.2f} C={p['w_comp']:.2f} M={p['w_mom']:.2f} B>{p['buy_thresh']:.1f} S<{p['sell_thresh']:.1f}"
        print(f"{cfg:<60} {r['total_return_pct']:>+9.2f}% {r['sharpe_ratio']:>8.3f} {r['max_drawdown_pct']:>9.2f}% {r['total_trades']:>8} {r['win_rate_pct']:>9.1f}%")

    return results


def main():
    csv_path = '/home/kushal_jain/Meractus/paper_dataset(1).csv'
    initial_capital = 10_000_000.0

    # Run 3 individual strategies + default ensemble
    results = {}
    for strat in ['momentum', 'mean_reversion', 'composite']:
        results[strat] = run_backtest(csv_path, initial_capital, strat)
    results['ensemble'] = run_backtest(csv_path, initial_capital, 'ensemble')

    # Parameter sweep
    sweep_results = parameter_sweep(csv_path, initial_capital)
    best_sweep = sweep_results[0] if sweep_results else None

    # If sweep found better, add it
    if best_sweep and best_sweep['sharpe_ratio'] > results['ensemble']['sharpe_ratio']:
        p = best_sweep['params']
        results['ensemble_optimized'] = run_backtest(
            csv_path, initial_capital, 'weighted_ensemble',
            w_mr=p['w_mr'], w_comp=p['w_comp'], w_mom=p['w_mom'],
            buy_thresh=p['buy_thresh'], sell_thresh=p['sell_thresh'],
        )

    # === COMPARISON TABLE ===
    print("\n" + "=" * 110)
    print("STRATEGY COMPARISON — All Strategies")
    print("=" * 110)
    print(f"{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'Trades':>8} {'WinRate':>10} {'Best5D':>10} {'Worst5D':>10}")
    print("-" * 110)
    for name in ['momentum', 'mean_reversion', 'composite', 'ensemble', 'ensemble_optimized']:
        if name in results:
            r = results[name]
            print(f"{r['strategy']:<25} {r['total_return_pct']:>+9.2f}% {r['sharpe_ratio']:>8.3f} {r['max_drawdown_pct']:>9.2f}% {r['total_trades']:>8} {r['win_rate_pct']:>9.1f}% {r['best_5d_pct']:>+9.2f}% {r['worst_5d_pct']:>+9.2f}%")
    print("=" * 110)

    # Save
    save_data = {}
    for name, r in results.items():
        save_data[name] = {k: v for k, v in r.items() if k not in ['equity_curve', 'trade_log']}
    with open('/home/kushal_jain/Meractus/backtest_ensemble_results.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print("\nResults saved to backtest_ensemble_results.json")


if __name__ == '__main__':
    main()
