"""
Mercatus Arena Live Runner
Works with real server OR mock server for testing.
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
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger('mercatus.runner')

_stop = False
def _sig(s, f):
    global _stop
    _stop = True
signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM, _sig)


class LiveRunner:
    def __init__(self, client, dashboard: Dashboard, speed: float = 1.0):
        self.client = client
        self.dashboard = dashboard
        self.speed = speed
        self.running = False
        self.tick_count = 0

    def _on_tick(self, tick_num, data):
        self.tick_count = tick_num

    def _on_trade(self, trade):
        self.dashboard.log_trade(trade)

    async def run_live(self):
        """Run against live Mercatus server."""
        log.info("Connecting to live server...")
        if not self.client.login():
            log.error("Login failed!")
            return

        self.client.start_time = time.time()
        self.running = True

        connected = await self.client.connect_ws()
        if connected:
            log.info("WebSocket connected, listening for ticks...")
            reconnect_count = 0
            while self.running and not _stop:
                if not self.client.connected:
                    reconnect_count += 1
                    if reconnect_count > 5:
                        log.warning("Too many reconnects, falling back to REST")
                        break
                    log.info(f"WS disconnected, reconnecting ({reconnect_count}/5)...")
                    await asyncio.sleep(1)
                    await self.client.connect_ws()
                    continue
                reconnect_count = 0
                await self._strategy_tick()
                await asyncio.sleep(0.05)

        if not self.client.connected or not connected:
            log.info("Falling back to REST polling...")
            while self.running and not _stop:
                self.client.get_snapshot()
                self.client.get_portfolio()
                self.tick_count += 1
                await self._strategy_tick()
                await asyncio.sleep(1.0)

        self.dashboard.save_state(self.client.get_stats())
        self.dashboard.generate_graphs()

    async def run_simulation(self):
        """Run simulation with CSV data."""
        log.info("Starting simulation...")
        if not self.client.connect():
            log.error("Failed to load data!")
            return

        self.running = True
        log.info(f"Running {len(self.client._timestamps)} ticks...")

        try:
            while self.running and not _stop:
                if not self.client.get_next_tick():
                    break
                await self._strategy_tick()
                if self.speed > 0:
                    await asyncio.sleep(1.0 / self.speed)
                if self.tick_count % 100 == 0:
                    stats = self.client.get_stats()
                    self.dashboard.update(stats, self.client.current_data,
                                          self.dashboard.trade_log.get_recent())
        except asyncio.CancelledError:
            pass

        stats = self.client.get_stats()
        self.dashboard.save_state(stats)
        self.dashboard.generate_graphs()
        self._print_summary(stats)

    async def _strategy_tick(self):
        if not self.client.current_data:
            return
        try:
            actions = strategy(
                self.client.current_data,
                self.client.portfolio,
                self.client.cash,
                self.client.history,
            )
            if actions:
                for symbol, (action, qty) in actions.items():
                    if qty > 0:
                        success = self.client.send_order(symbol, action, qty)
                        if success:
                            log.info(f"ORDER: {action} {qty} {symbol}")

            while self.client._pending_trades:
                t = self.client._pending_trades.pop(0)
                self.dashboard.log_trade(t)

        except Exception as e:
            log.error(f"Strategy error: {e}", exc_info=True)
            self.client.errors += 1

    def _print_summary(self, stats: dict):
        cash = stats.get('cash', 10_000_000)
        equity = cash
        for sym, pos in stats.get('portfolio', {}).items():
            qty = pos.get('quantity', 0)
            price = self.client.last_prices.get(sym, pos.get('avg_price', 0))
            equity += qty * price

        total_return = (equity - 10_000_000) / 10_000_000 * 100
        trades = self.dashboard.trade_log.trades
        pnls = [t.get('pnl', 0) for t in trades if 'pnl' in t]
        wins = sum(1 for p in pnls if p > 0)

        print("\n" + "=" * 60)
        print("  SIMULATION COMPLETE")
        print("=" * 60)
        print(f"  Final Equity:  ${equity:,.0f}")
        print(f"  Total Return:  {total_return:+.2f}%")
        print(f"  Total Trades:  {len(trades)}")
        print(f"  Win Rate:      {wins / max(len(pnls), 1) * 100:.1f}%")
        if pnls:
            avg_win = sum(p for p in pnls if p > 0) / max(wins, 1)
            avg_loss = sum(p for p in pnls if p < 0) / max(len(pnls) - wins, 1)
            print(f"  Avg Win:       ${avg_win:+,.0f}")
            print(f"  Avg Loss:      ${avg_loss:+,.0f}")
        print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description='Mercatus Arena Runner')
    parser.add_argument('config', nargs='?', help='Config JSON file')
    parser.add_argument('--mode', choices=['live', 'simulate', 'mock'], default='simulate')
    parser.add_argument('--data', default='paper_dataset(1).csv')
    parser.add_argument('--speed', type=float, default=10.0)
    args = parser.parse_args()

    dashboard = Dashboard()

    if args.mode == 'simulate':
        with open(os.path.join(os.path.dirname(__file__), 'docs', 'dataset_meta.json')) as f:
            meta = json.load(f)
        client = SimulatedClient(args.data, meta['symbols'])
        runner = LiveRunner(client, dashboard, speed=args.speed)
        await runner.run_simulation()

    elif args.mode == 'mock':
        from mock_server import run_mock_server
        client = MercatusClient('http://localhost:4040', api_key='sk_mock')

        async def run():
            mock_task = asyncio.create_task(run_mock_server(args.data))
            await asyncio.sleep(1)
            runner = LiveRunner(client, dashboard, speed=50)
            runner.running = True
            while runner.running and not _stop:
                snap = client.get_snapshot()
                if snap:
                    client.tick_count += 1
                    client.history.append({'timestamp': time.time(), 'data': snap})
                await runner._strategy_tick()
                await asyncio.sleep(0.02)
            dashboard.save_state(client.get_stats())
            dashboard.generate_graphs()
            mock_task.cancel()

        await run()

    elif args.mode == 'live':
        if not args.config:
            print("Need config file for live mode: --mode live config.json")
            return
        with open(args.config) as f:
            cfg = json.load(f)
        client = MercatusClient(
            cfg['server']['base_url'],
            api_key=cfg['auth'].get('api_key', ''),
            email=cfg['auth'].get('email', ''),
            password=cfg['auth'].get('password', ''),
        )
        runner = LiveRunner(client, dashboard)
        await runner.run_live()


if __name__ == '__main__':
    asyncio.run(main())
