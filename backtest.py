"""
Backtesting engine for Mercatus Arena strategies.
Replays CSV data tick-by-tick, runs each strategy independently, logs results.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from models.momentum import MomentumStrategy
from models.mean_reversion import MeanReversionStrategy
from models.ensemble import LightGBMStrategy
from risk import RiskManager
from features import volatility
import json
import time


class BacktestEngine:
    """Event-driven backtester."""

    def __init__(self, initial_capital: float = 10_000_000.0,
                 max_order_value_pct: float = 0.01):
        self.initial_capital = initial_capital
        self.max_order_value_pct = max_order_value_pct
        self.results = {}

    def _load_data(self, csv_path: str) -> Dict[str, pd.DataFrame]:
        """Load CSV and organize by symbol."""
        df = pd.read_csv(csv_path)
        df = df.sort_values(['symbol', 'timestamp'])
        symbols = {}
        for sym in df['symbol'].unique():
            sdf = df[df['symbol'] == sym].copy()
            sdf = sdf.sort_values('timestamp').reset_index(drop=True)
            symbols[sym] = sdf
        return symbols

    def run_strategy(self, strategy_name: str, csv_path: str,
                     strategy, risk_mgr: RiskManager,
                     warmup_bars: int = 100,
                     rebalance_interval: int = 390) -> Dict:
        """
        Run a single strategy against the CSV data.
        strategy.signal() is called per symbol per tick.
        """
        print(f"\n{'='*60}")
        print(f"Running backtest: {strategy_name}")
        print(f"{'='*60}")

        data = self._load_data(csv_path)
        symbols = sorted(data.keys())

        # Build aligned timestamps
        all_timestamps = set()
        for sym in symbols:
            all_timestamps.update(data[sym]['timestamp'].values)
        timestamps = sorted(all_timestamps)

        print(f"Symbols: {len(symbols)}, Timestamps: {len(timestamps)}")

        # State
        cash = self.initial_capital
        portfolio: Dict[str, Dict] = {s: {'quantity': 0} for s in symbols}
        trades = []
        equity_curve = []
        trade_log = []

        # Per-symbol price arrays (for strategy history)
        price_arrays = {s: data[s]['price'].values for s in symbols}
        volume_arrays = {s: data[s]['volume'].values for s in symbols}
        bar_counters = {s: 0 for s in symbols}

        # Map timestamp to index per symbol
        ts_to_idx = {}
        for sym in symbols:
            ts_to_idx[sym] = {row['timestamp']: i for i, row in data[sym].iterrows()}

        strategy.reset()
        risk_mgr.reset()

        t_start = time.time()
        last_log = 0

        for t_idx, ts in enumerate(timestamps):
            # Progress
            if t_idx - last_log > 10000:
                elapsed = time.time() - t_start
                pct = t_idx / len(timestamps) * 100
                print(f"  [{pct:.1f}%] t={t_idx}/{len(timestamps)} | equity=${cash + sum(portfolio[s]['quantity'] * price_arrays[s][ts_to_idx[s].get(ts, 0)] for s in symbols):,.0f} | elapsed={elapsed:.0f}s")
                last_log = t_idx

            # Get current prices
            current_prices = {}
            for sym in symbols:
                if ts in ts_to_idx[sym]:
                    idx = ts_to_idx[sym][ts]
                    current_prices[sym] = price_arrays[sym][idx]
                    bar_counters[sym] = idx

            # Warmup
            if t_idx < warmup_bars:
                equity_curve.append({
                    'timestamp': ts,
                    'equity': cash,
                    'cash': cash,
                    'positions_value': 0,
                })
                continue

            # Check stop-losses for all held positions
            for sym in symbols:
                if portfolio[sym]['quantity'] > 0 and sym in current_prices:
                    if risk_mgr.check_stop_loss(sym, current_prices[sym], portfolio):
                        # Force sell
                        qty = portfolio[sym]['quantity']
                        price = current_prices[sym]
                        cash += qty * price
                        portfolio[sym]['quantity'] = 0
                        risk_mgr.record_exit(sym)
                        trade_log.append({
                            'timestamp': ts, 'symbol': sym, 'action': 'SELL',
                            'price': price, 'quantity': qty, 'reason': 'stop_loss'
                        })
                        strategy.on_fill(sym, 'SELL', price, bar_counters[sym])

            # Generate signals
            for sym in symbols:
                if sym not in current_prices:
                    continue

                # Build price history up to current bar
                idx = ts_to_idx[sym].get(ts, None)
                if idx is None:
                    continue

                prices_slice = price_arrays[sym][:idx + 1]
                volumes_slice = volume_arrays[sym][:idx + 1]

                # Get signal
                if hasattr(strategy, 'signal') and 'LightGBM' in strategy.__class__.__name__:
                    action, confidence = strategy.signal(
                        sym, prices_slice, idx, volumes_slice,
                        all_prices=price_arrays, all_volumes=volume_arrays
                    )
                else:
                    action, confidence = strategy.signal(
                        sym, prices_slice, idx, volumes_slice
                    )

                if action == 'HOLD':
                    continue

                price = current_prices[sym]

                if action == 'BUY' and portfolio[sym]['quantity'] == 0:
                    # Position sizing
                    vol = 0.02
                    vol_arr = volatility(prices_slice, 20)
                    if not np.isnan(vol_arr[-1]):
                        vol = vol_arr[-1]

                    qty = risk_mgr.compute_position_size(
                        sym, price, cash, portfolio, current_prices,
                        signal_strength=confidence, volatility=vol
                    )
                    if qty > 0:
                        cost = qty * price
                        if cost <= cash * (1 - 0.10):  # 10% cash reserve
                            cash -= cost
                            portfolio[sym]['quantity'] = qty
                            risk_mgr.record_entry(sym, price)
                            trade_log.append({
                                'timestamp': ts, 'symbol': sym, 'action': 'BUY',
                                'price': price, 'quantity': qty,
                                'confidence': confidence, 'reason': 'signal'
                            })
                            strategy.on_fill(sym, 'BUY', price, idx)

                elif action == 'SELL' and portfolio[sym]['quantity'] > 0:
                    qty = portfolio[sym]['quantity']
                    cash += qty * price
                    portfolio[sym]['quantity'] = 0
                    risk_mgr.record_exit(sym)
                    trade_log.append({
                        'timestamp': ts, 'symbol': sym, 'action': 'SELL',
                        'price': price, 'quantity': qty, 'reason': 'signal'
                    })
                    strategy.on_fill(sym, 'SELL', price, idx)

            # Record equity
            positions_value = sum(
                portfolio[s]['quantity'] * current_prices.get(s, 0)
                for s in symbols
            )
            equity_curve.append({
                'timestamp': ts,
                'equity': cash + positions_value,
                'cash': cash,
                'positions_value': positions_value,
            })

        # Final equity
        final_equity = equity_curve[-1]['equity'] if equity_curve else self.initial_capital

        # Compute metrics
        equity_df = pd.DataFrame(equity_curve)
        equity_df['returns'] = equity_df['equity'].pct_change()

        total_return = (final_equity - self.initial_capital) / self.initial_capital
        daily_returns = equity_df['returns'].dropna()

        # Approximate daily returns (group by ~390 bars per day)
        bars_per_day = 390
        if len(daily_returns) > bars_per_day:
            daily_rets = daily_returns.values.reshape(-1, bars_per_day).mean(axis=1)
        else:
            daily_rets = daily_returns.values

        sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0

        # Max drawdown
        equity_arr = equity_df['equity'].values
        cummax = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - cummax) / cummax
        max_drawdown = np.min(drawdown)

        # Win rate
        buy_trades = [t for t in trade_log if t['action'] == 'BUY']
        sell_trades = [t for t in trade_log if t['action'] == 'SELL']
        n_trades = min(len(buy_trades), len(sell_trades))

        wins = 0
        for i in range(n_trades):
            buy_price = buy_trades[i]['price']
            sell_price = sell_trades[i]['price']
            if sell_price > buy_price:
                wins += 1
        win_rate = wins / n_trades if n_trades > 0 else 0

        # Per-symbol PnL
        symbol_pnl = {}
        for t in trade_log:
            sym = t['symbol']
            if sym not in symbol_pnl:
                symbol_pnl[sym] = 0
        # Simple approach: track buy/sell pairs
        positions_tracker = {}
        for t in trade_log:
            sym = t['symbol']
            if t['action'] == 'BUY':
                positions_tracker[sym] = {'qty': t['quantity'], 'cost': t['price']}
            elif t['action'] == 'SELL' and sym in positions_tracker:
                pnl = (t['price'] - positions_tracker[sym]['cost']) * t['quantity']
                symbol_pnl[sym] = symbol_pnl.get(sym, 0) + pnl
                del positions_tracker[sym]

        result = {
            'strategy': strategy_name,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': len(trade_log),
            'buy_trades': len([t for t in trade_log if t['action'] == 'BUY']),
            'sell_trades': len([t for t in trade_log if t['action'] == 'SELL']),
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'symbol_pnl': symbol_pnl,
            'equity_curve': equity_df,
            'trade_log': trade_log,
        }

        print(f"\n--- {strategy_name} Results ---")
        print(f"Final Equity:  ${final_equity:,.2f}")
        print(f"Total Return:  {total_return*100:+.2f}%")
        print(f"Sharpe Ratio:  {sharpe:.3f}")
        print(f"Max Drawdown:  {max_drawdown*100:.2f}%")
        print(f"Total Trades:  {len(trade_log)}")
        print(f"Win Rate:      {win_rate*100:.1f}%")
        print(f"Best Symbol:   {max(symbol_pnl, key=symbol_pnl.get) if symbol_pnl else 'N/A'} (${max(symbol_pnl.values()) if symbol_pnl else 0:,.0f})")
        print(f"Worst Symbol:  {min(symbol_pnl, key=symbol_pnl.get) if symbol_pnl else 'N/A'} (${min(symbol_pnl.values()) if symbol_pnl else 0:,.0f})")

        return result

    def run_all(self, csv_path: str) -> Dict[str, Dict]:
        """Run all three strategies and compare."""
        results = {}

        # Strategy 1: Momentum
        momentum = MomentumStrategy(
            fast_period=10, slow_period=30,
            stop_loss_pct=0.05, trailing_stop_pct=0.05
        )
        risk1 = RiskManager(initial_capital=self.initial_capital)
        results['momentum'] = self.run_strategy(
            'EMA Crossover Momentum', csv_path, momentum, risk1
        )

        # Strategy 2: Mean Reversion
        mean_rev = MeanReversionStrategy(
            lookback=20, entry_threshold=2.0,
            take_profit_pct=0.02, stop_loss_pct=0.05,
            rsi_entry=30.0, rsi_boost=4
        )
        risk2 = RiskManager(initial_capital=self.initial_capital)
        results['mean_reversion'] = self.run_strategy(
            'Mean Reversion + Flash Plays', csv_path, mean_rev, risk2
        )

        # Strategy 3: LightGBM
        try:
            lgbm = LightGBMStrategy(
                train_window=5 * 390, retrain_every=6 * 390,
                confidence_threshold=0.6, stop_loss_pct=0.05
            )
            risk3 = RiskManager(initial_capital=self.initial_capital)
            results['lgbm'] = self.run_strategy(
                'LightGBM Direction Predictor', csv_path, lgbm, risk3
            )
        except Exception as e:
            print(f"\nLightGBM strategy failed: {e}")
            print("Make sure lightgbm is installed: pip install lightgbm")

        return results


def compare_results(results: Dict[str, Dict]):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)
    print(f"{'Strategy':<35} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'Trades':>8} {'WinRate':>10}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{r['strategy']:<35} {r['total_return_pct']:>+9.2f}% {r['sharpe_ratio']:>8.3f} {r['max_drawdown_pct']:>9.2f}% {r['total_trades']:>8} {r['win_rate_pct']:>9.1f}%")
    print("=" * 80)


if __name__ == '__main__':
    csv_path = '/home/kushal_jain/Meractus/paper_dataset(1).csv'
    engine = BacktestEngine(initial_capital=10_000_000)
    results = engine.run_all(csv_path)
    compare_results(results)

    # Save results summary
    summary = {}
    for name, r in results.items():
        summary[name] = {
            'strategy': r['strategy'],
            'final_equity': r['final_equity'],
            'total_return_pct': r['total_return_pct'],
            'sharpe_ratio': r['sharpe_ratio'],
            'max_drawdown_pct': r['max_drawdown_pct'],
            'total_trades': r['total_trades'],
            'win_rate_pct': r['win_rate_pct'],
            'symbol_pnl': r['symbol_pnl'],
        }
    with open('/home/kushal_jain/Meractus/backtest_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nResults saved to backtest_results.json")
