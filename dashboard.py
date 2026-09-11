"""
Mercatus Arena Terminal Dashboard
Real-time terminal UI for monitoring trades, positions, and PnL.
"""
import time
import os
import json
from datetime import datetime
from collections import deque
from typing import Dict, List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.columns import Columns
from rich import box


class TradeLog:
    """Persistent trade log with file output."""

    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.trades: List[dict] = []
        self.log_file = os.path.join(log_dir, f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        self.summary_file = os.path.join(log_dir, 'trade_summary.json')

    def add(self, trade: dict):
        self.trades.append(trade)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(trade, default=str) + '\n')

    def save_summary(self, stats: dict, equity_curve: list):
        summary = {
            'total_trades': len(self.trades),
            'stats': stats,
            'equity_curve': equity_curve[-100:] if equity_curve else [],
            'trades': self.trades[-50:],
            'saved_at': datetime.now().isoformat(),
        }
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    def get_recent(self, n: int = 20) -> List[dict]:
        return self.trades[-n:]


class Dashboard:
    """Rich terminal dashboard."""

    def __init__(self):
        self.console = Console()
        self.trade_log = TradeLog()
        self.equity_curve: List[float] = []
        self.pnl_history: List[float] = []
        self.start_time = time.time()
        self.last_signals: Dict[str, dict] = {}

    def _make_header(self, stats: dict) -> Panel:
        elapsed = stats.get('elapsed', 0)
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)

        header = Text()
        header.append(" MERCATUS ARENA ", style="bold white on blue")
        header.append("  ")
        header.append(f"Tick: {stats.get('tick_count', 0):,}", style="cyan")
        header.append("  |  ")
        header.append(f"Rate: {stats.get('tick_rate', 0):.1f}/s", style="cyan")
        header.append("  |  ")
        header.append(f"Orders: {stats.get('orders_sent', 0)}", style="green")
        header.append("  |  ")
        header.append(f"Errors: {stats.get('errors', 0)}", style="red" if stats.get('errors', 0) > 0 else "dim")
        header.append("  |  ")
        header.append(f"Uptime: {h}h{m:02d}m{s:02d}s", style="yellow")
        header.append("  |  ")
        header.append(f"Symbols: {stats.get('symbols_tracked', 0)}", style="magenta")

        return Panel(header, style="bold blue", box=box.DOUBLE)

    def _make_portfolio_table(self, stats: dict, prices: dict) -> Table:
        table = Table(title="Portfolio & Positions", box=box.ROUNDED, show_lines=True)
        table.add_column("Symbol", style="bold cyan", width=8)
        table.add_column("Qty", justify="right", style="white")
        table.add_column("Entry", justify="right", style="dim")
        table.add_column("Current", justify="right", style="white")
        table.add_column("PnL", justify="right")
        table.add_column("PnL %", justify="right")
        table.add_column("Signal", justify="center")

        portfolio = stats.get('portfolio', {})
        cash = stats.get('cash', 10_000_000)

        # Cash row
        table.add_row(
            "CASH", "--", "--",
            f"${cash:,.0f}", "--", "--", "--",
            style="bold green"
        )

        total_pnl = 0
        for sym in sorted(set(list(portfolio.keys()) + list(prices.keys()))):
            pos = portfolio.get(sym, {})
            qty = pos.get('quantity', 0)
            if qty <= 0:
                continue

            entry = pos.get('avg_price', 0)
            current = prices.get(sym, {}).get('close', 0)
            if isinstance(current, dict):
                current = current.get('close', 0)

            pnl = (current - entry) * qty if entry > 0 else 0
            pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
            total_pnl += pnl

            signal = self.last_signals.get(sym, {})
            signal_str = signal.get('action', '--')
            signal_style = "green" if signal_str == "BUY" else "red" if signal_str == "SELL" else "dim"

            pnl_style = "green" if pnl >= 0 else "red"
            table.add_row(
                sym, f"{qty:,}", f"${entry:.2f}",
                f"${current:.2f}",
                f"${pnl:+,.0f}",
                f"{pnl_pct:+.2f}%",
                Text(signal_str, style=signal_style),
                style=pnl_style
            )

        # Total
        table.add_row(
            "TOTAL", "", "", "",
            f"${total_pnl:+,.0f}",
            f"{total_pnl / 10_000_000 * 100:+.2f}%",
            "", style="bold"
        )

        return table

    def _make_trade_log_table(self, trades: List[dict]) -> Table:
        table = Table(title="Recent Trades", box=box.ROUNDED, show_lines=True)
        table.add_column("Time", style="dim", width=10)
        table.add_column("Symbol", style="bold cyan", width=8)
        table.add_column("Action", width=6)
        table.add_column("Qty", justify="right", width=8)
        table.add_column("Price", justify="right", width=10)
        table.add_column("PnL", justify="right", width=12)
        table.add_column("Reason", style="dim", width=12)

        for trade in reversed(trades[-15:]):
            ts = trade.get('timestamp', 0)
            if isinstance(ts, float) and ts > 1e9:
                time_str = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            else:
                time_str = str(ts)[:10]

            action = trade.get('action', '')
            action_style = "bold green" if action == "BUY" else "bold red" if action == "SELL" else "white"

            pnl = trade.get('pnl')
            pnl_str = f"${pnl:+,.0f}" if pnl is not None else "--"
            pnl_style = "green" if pnl and pnl > 0 else "red" if pnl and pnl < 0 else "dim"

            table.add_row(
                time_str,
                trade.get('symbol', '?'),
                Text(action, style=action_style),
                f"{trade.get('quantity', 0):,}",
                f"${trade.get('price', 0):.2f}",
                Text(pnl_str, style=pnl_style),
                trade.get('reason', '--'),
            )

        return table

    def _make_stats_panel(self, stats: dict) -> Panel:
        cash = stats.get('cash', 10_000_000)
        equity = cash
        portfolio = stats.get('portfolio', {})
        for sym, pos in portfolio.items():
            qty = pos.get('quantity', 0)
            # Use last known price
            equity += qty * pos.get('avg_price', 0)

        total_return = (equity - 10_000_000) / 10_000_000 * 100

        stats_text = Text()
        stats_text.append("Performance\n", style="bold yellow")
        stats_text.append(f"  Equity:    ${equity:,.0f}\n", style="white")
        stats_text.append(f"  Cash:      ${cash:,.0f}\n", style="green")
        stats_text.append(f"  Return:    {total_return:+.2f}%\n",
                          style="green" if total_return >= 0 else "red")
        stats_text.append(f"  Positions: {sum(1 for p in portfolio.values() if p.get('quantity', 0) > 0)}\n", style="cyan")
        stats_text.append(f"  Trades:    {stats.get('orders_sent', 0)}\n", style="magenta")

        return Panel(stats_text, title="Stats", box=box.ROUNDED, style="bold")

    def _make_signals_panel(self, signals: dict) -> Table:
        table = Table(title="Active Signals", box=box.ROUNDED, show_lines=True)
        table.add_column("Symbol", style="bold cyan", width=8)
        table.add_column("Action", width=8)
        table.add_column("Confidence", justify="right", width=10)
        table.add_column("Z-Score", justify="right", width=10)
        table.add_column("RSI", justify="right", width=8)

        for sym, sig in sorted(signals.items()):
            action = sig.get('action', 'HOLD')
            action_style = "green" if action == "BUY" else "red" if action == "SELL" else "dim"
            table.add_row(
                sym,
                Text(action, style=action_style),
                f"{sig.get('confidence', 0):.2f}",
                f"{sig.get('z_score', 0):.2f}",
                f"{sig.get('rsi', 50):.1f}",
            )

        return table

    def update(self, stats: dict, prices: dict, trades: list, signals: dict = None):
        """Update the dashboard display."""
        self.last_signals = signals or {}
        equity = stats.get('cash', 10_000_000)
        portfolio = stats.get('portfolio', {})
        for sym, pos in portfolio.items():
            qty = pos.get('quantity', 0)
            price = prices.get(sym, {})
            if isinstance(price, dict):
                price = price.get('close', 0)
            equity += qty * price

        self.equity_curve.append(equity)
        if len(self.equity_curve) > 5000:
            self.equity_curve = self.equity_curve[-5000:]

        # Build layout
        self.console.clear()

        header = self._make_header(stats)
        self.console.print(header)

        portfolio_table = self._make_portfolio_table(stats, prices)
        trade_table = self._make_trade_log_table(trades)
        stats_panel = self._make_stats_panel(stats)

        self.console.print(Columns([portfolio_table, stats_panel], expand=True))
        self.console.print(trade_table)

        if signals:
            sig_table = self._make_signals_panel(signals)
            self.console.print(sig_table)

        # Footer
        progress = stats.get('progress', '')
        if progress:
            self.console.print(f"\n  Progress: {progress}  |  Press Ctrl+C to stop", style="dim")

    def log_trade(self, trade: dict):
        """Log a trade."""
        self.trade_log.add(trade)

    def save_state(self, stats: dict):
        """Save current state to disk."""
        self.trade_log.save_summary(stats, self.equity_curve)

    def generate_graphs(self, output_dir: str = 'logs'):
        """Generate performance graphs."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return

        os.makedirs(output_dir, exist_ok=True)

        if len(self.equity_curve) < 2:
            return

        # Equity curve
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        ax.plot(self.equity_curve, color='#00bcd4', linewidth=1.5)
        ax.fill_between(range(len(self.equity_curve)), self.equity_curve, alpha=0.1, color='#00bcd4')
        ax.set_xlabel('Tick', color='#e0e0e0', fontsize=12)
        ax.set_ylabel('Equity ($)', color='#e0e0e0', fontsize=12)
        ax.set_title('Live Equity Curve', color='#e0e0e0', fontsize=14, fontweight='bold')
        ax.tick_params(colors='#e0e0e0')
        ax.grid(True, alpha=0.2, color='#333355')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'equity_curve.png'), facecolor='#1a1a2e', dpi=150)
        plt.close()

        # Drawdown
        equity_arr = np.array(self.equity_curve)
        cummax = np.maximum.accumulate(equity_arr)
        dd = (equity_arr - cummax) / cummax * 100

        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        ax.fill_between(range(len(dd)), dd, 0, alpha=0.4, color='#f44336')
        ax.plot(dd, color='#f44336', linewidth=1)
        ax.set_xlabel('Tick', color='#e0e0e0')
        ax.set_ylabel('Drawdown (%)', color='#e0e0e0')
        ax.set_title('Drawdown', color='#e0e0e0', fontsize=14, fontweight='bold')
        ax.tick_params(colors='#e0e0e0')
        ax.grid(True, alpha=0.2, color='#333355')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'drawdown.png'), facecolor='#1a1a2e', dpi=150)
        plt.close()

        # Trade PnL distribution
        if self.trade_log.trades:
            pnls = [t.get('pnl', 0) for t in self.trade_log.trades if 'pnl' in t]
            if pnls:
                fig, ax = plt.subplots(figsize=(10, 5))
                fig.patch.set_facecolor('#1a1a2e')
                ax.set_facecolor('#1a1a2e')
                ax.hist(pnls, bins=30, color='#00bcd4', alpha=0.7, edgecolor='#1a1a2e')
                ax.axvline(x=0, color='#f44336', linestyle='--', linewidth=2)
                ax.axvline(x=np.mean(pnls), color='#4caf50', linestyle='-', linewidth=2,
                           label=f'Mean: ${np.mean(pnls):+,.0f}')
                ax.set_xlabel('PnL ($)', color='#e0e0e0')
                ax.set_ylabel('Frequency', color='#e0e0e0')
                ax.set_title('Trade PnL Distribution', color='#e0e0e0', fontsize=14, fontweight='bold')
                ax.legend(facecolor='#2a2a4a', edgecolor='#333355', labelcolor='#e0e0e0')
                ax.tick_params(colors='#e0e0e0')
                ax.grid(True, alpha=0.2, color='#333355')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'trade_pnl.png'), facecolor='#1a1a2e', dpi=150)
                plt.close()

        # Cumulative PnL by symbol
        sym_pnl = {}
        for t in self.trade_log.trades:
            if 'pnl' in t:
                sym = t['symbol']
                sym_pnl[sym] = sym_pnl.get(sym, 0) + t['pnl']

        if sym_pnl:
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#1a1a2e')
            symbols = sorted(sym_pnl.keys())
            pnls = [sym_pnl[s] / 1000 for s in symbols]
            colors = ['#4caf50' if p > 0 else '#f44336' for p in pnls]
            ax.barh(symbols, pnls, color=colors, alpha=0.85)
            ax.set_xlabel('PnL ($K)', color='#e0e0e0')
            ax.set_title('Per-Symbol PnL', color='#e0e0e0', fontsize=14, fontweight='bold')
            ax.tick_params(colors='#e0e0e0')
            ax.grid(True, alpha=0.2, color='#333355', axis='x')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'symbol_pnl.png'), facecolor='#1a1a2e', dpi=150)
            plt.close()

        print(f"  Graphs saved to {output_dir}/")
        print(f"    - equity_curve.png")
        print(f"    - drawdown.png")
        print(f"    - trade_pnl.png")
        print(f"    - symbol_pnl.png")
