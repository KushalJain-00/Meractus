"""
Mercatus Arena Live Runner
Connects API → Strategy → Orders with real-time dashboard.
"""
import asyncio
import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime

from strategy import strategy, _state
from api_client import MercatusClient, SimulatedClient
from dashboard import Dashboard

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/mercatus.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger('mercatus.runner')

# Global stop flag
_stop = False


def handle_signal(sig, frame):
    global _stop
    log.info("Shutdown signal received...")
    _stop = True


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


class LiveRunner:
    """Orchestrates API → Strategy → Orders with dashboard."""

    def __init__(self, client, dashboard: Dashboard, speed: float = 1.0):
        self.client = client
        self.dashboard = dashboard
        self.speed = speed
        self.running = False
        self.tick_count = 0

        # Wire up callbacks
        self.client.on_tick = self._on_tick
        self.client.on_trade = self._on_trade

    def _on_tick(self, tick_num: int, data: dict):
        """Called on each market data tick."""
        self.tick_count = tick_num

    def _on_trade(self, trade: dict):
        """Called when a trade is executed."""
        self.dashboard.log_trade(trade)

    async def run_live(self):
        """Run against live Mercatus server."""
        log.info("Connecting to live server...")
        connected = await self.client.connect()
        if not connected:
            log.error("Failed to connect!")
            return

        log.info("Connected! Starting strategy loop...")
        self.running = True

        # Start listening in background
        listen_task = asyncio.create_task(self.client.listen())

        # Main strategy loop
        try:
            while self.running and not _stop:
                await self._strategy_tick()
                await asyncio.sleep(0.1)  # 100ms strategy loop
        except asyncio.CancelledError:
            pass
        finally:
            await self.client.disconnect()
            self.dashboard.save_state(self.client.get_stats())
            self.dashboard.generate_graphs()

    async def run_simulation(self):
        """Run simulation with CSV data."""
        log.info("Starting simulation...")
        connected = self.client.connect()
        if not connected:
            log.error("Failed to load data!")
            return

        log.info(f"Data loaded. Running simulation...")
        self.running = True

        try:
            while self.running and not _stop:
                if not self.client.get_next_tick():
                    log.info("Simulation complete!")
                    break

                await self._strategy_tick()

                # Adaptive speed
                if self.speed > 0:
                    await asyncio.sleep(1.0 / self.speed)

                # Update dashboard every 100 ticks
                if self.tick_count % 100 == 0:
                    stats = self.client.get_stats()
                    self.dashboard.update(
                        stats,
                        self.client.current_data,
                        self.dashboard.trade_log.get_recent(),
                    )

        except asyncio.CancelledError:
            pass
        finally:
            stats = self.client.get_stats()
            self.dashboard.save_state(stats)
            self.dashboard.generate_graphs()
            self._print_final_summary(stats)

    async def _strategy_tick(self):
        """Run one tick of the strategy."""
        if not self.client.current_data:
            return

        try:
            # Call strategy
            actions = strategy(
                self.client.current_data,
                self.client.portfolio,
                self.client.cash,
                self.client.history,
            )

            # Send orders
            if actions:
                for symbol, (action, qty) in actions.items():
                    if qty > 0:
                        success = await self.client.send_order(symbol, action, qty)
                        if success:
                            log.info(f"ORDER: {action} {qty} {symbol}")

            # Drain pending trades from simulated client
            if hasattr(self.client, '_pending_trades'):
                while self.client._pending_trades:
                    trade = self.client._pending_trades.pop(0)
                    self.dashboard.log_trade(trade)
                    if self.on_trade:
                        self.client.on_trade(trade)

        except Exception as e:
            log.error(f"Strategy error: {e}", exc_info=True)
            self.client.errors += 1

    def _print_final_summary(self, stats: dict):
        """Print final performance summary."""
        cash = stats.get('cash', 10_000_000)
        equity = cash
        portfolio = stats.get('portfolio', {})
        for sym, pos in portfolio.items():
            qty = pos.get('quantity', 0)
            entry = pos.get('avg_price', 0)
            equity += qty * entry

        total_return = (equity - 10_000_000) / 10_000_000 * 100
        trades = self.dashboard.trade_log.trades
        pnls = [t.get('pnl', 0) for t in trades if 'pnl' in t]
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls) * 100 if pnls else 0

        print("\n" + "=" * 60)
        print("  SIMULATION COMPLETE")
        print("=" * 60)
        print(f"  Final Equity:  ${equity:,.0f}")
        print(f"  Total Return:  {total_return:+.2f}%")
        print(f"  Total Trades:  {len(trades)}")
        print(f"  Win Rate:      {win_rate:.1f}%")
        if pnls:
            print(f"  Avg Win:       ${sum(p for p in pnls if p > 0) / max(wins, 1):+,.0f}")
            print(f"  Avg Loss:      ${sum(p for p in pnls if p < 0) / max(len(pnls) - wins, 1):+,.0f}")
            print(f"  Profit Factor: {sum(p for p in pnls if p > 0) / abs(sum(p for p in pnls if p < 0)):.2f}")
        print(f"  Ticks:         {stats.get('tick_count', 0):,}")
        print(f"  Orders Sent:   {stats.get('orders_sent', 0)}")
        print(f"  Errors:        {stats.get('errors', 0)}")
        print("=" * 60)
        print(f"\n  Logs saved to: logs/")
        print(f"  Graphs saved to: logs/")
        print()


async def main():
    parser = argparse.ArgumentParser(description='Mercatus Arena Live Runner')
    parser.add_argument('--mode', choices=['live', 'simulate'], default='simulate',
                        help='Run mode: live (real server) or simulate (CSV data)')
    parser.add_argument('--server', default='ws://localhost:8765',
                        help='WebSocket server URL')
    parser.add_argument('--team', default='TeamAlpha', help='Team name')
    parser.add_argument('--code', default='abc123', help='Team code')
    parser.add_argument('--data', default='paper_dataset(1).csv',
                        help='CSV data file for simulation')
    parser.add_argument('--speed', type=float, default=10.0,
                        help='Simulation speed (ticks/sec)')
    parser.add_argument('--duration', type=int, default=0,
                        help='Run duration in seconds (0=forever)')
    args = parser.parse_args()

    dashboard = Dashboard()

    if args.mode == 'simulate':
        with open('dataset_meta.json') as f:
            meta = json.load(f)
        client = SimulatedClient(args.data, meta['symbols'])
        runner = LiveRunner(client, dashboard, speed=args.speed)
        log.info(f"Simulation mode: {args.data} at {args.speed}x speed")
        await runner.run_simulation()
    else:
        client = MercatusClient(args.server, args.team, args.code)
        runner = LiveRunner(client, dashboard)
        log.info(f"Live mode: {args.server} as {args.team}")
        await runner.run_live()


if __name__ == '__main__':
    asyncio.run(main())
