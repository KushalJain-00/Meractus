"""
Mercatus Arena WebSocket API Client
Connects to the Mercatus server, receives market data, sends orders.
"""
import asyncio
import json
import logging
import time
from typing import Dict, Optional, Callable, Any

try:
    import websockets
except ImportError:
    websockets = None

log = logging.getLogger('mercatus.api')


class MercatusClient:
    """WebSocket client for Mercatus Arena."""

    def __init__(self, server_url: str, team_name: str, team_code: str):
        self.server_url = server_url
        self.team_name = team_name
        self.team_code = team_code
        self.ws = None
        self.connected = False
        self.authenticated = False

        # State
        self.current_data: Dict[str, Dict] = {}
        self.portfolio: Dict[str, Dict] = {}
        self.cash: float = 10_000_000
        self.history: list = []
        self.last_prices: Dict[str, float] = {}

        # Callbacks
        self.on_tick: Optional[Callable] = None
        self.on_trade: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

        # Stats
        self.tick_count = 0
        self.start_time = None
        self.orders_sent = 0
        self.errors = 0

    async def connect(self):
        """Connect to Mercatus WebSocket server."""
        if websockets is None:
            raise ImportError("pip install websockets")

        log.info(f"Connecting to {self.server_url}...")
        try:
            self.ws = await websockets.connect(
                self.server_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            )
            self.connected = True
            self.start_time = time.time()
            log.info("Connected!")

            # Authenticate
            await self._authenticate()
            return True
        except Exception as e:
            log.error(f"Connection failed: {e}")
            self.errors += 1
            return False

    async def _authenticate(self):
        """Send authentication message."""
        auth_msg = {
            "type": "auth",
            "team_name": self.team_name,
            "team_code": self.team_code,
        }
        await self.ws.send(json.dumps(auth_msg))
        log.info(f"Authenticated as {self.team_name}")

    async def listen(self):
        """Listen for incoming messages."""
        if not self.connected or not self.ws:
            return

        try:
            async for message in self.ws:
                await self._handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            log.warning("Connection closed by server")
            self.connected = False
        except Exception as e:
            log.error(f"Listen error: {e}")
            self.errors += 1
            self.connected = False

    async def _handle_message(self, raw: str):
        """Parse and handle incoming message."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            log.warning(f"Invalid JSON: {raw[:100]}")
            return

        msg_type = msg.get('type', '')

        if msg_type == 'market_data':
            await self._handle_market_data(msg)
        elif msg_type == 'portfolio_update':
            await self._handle_portfolio(msg)
        elif msg_type == 'trade_ack':
            await self._handle_trade_ack(msg)
        elif msg_type == 'error':
            log.error(f"Server error: {msg.get('message', 'unknown')}")
            self.errors += 1
        elif msg_type == 'auth_ok':
            self.authenticated = True
            log.info("Authentication successful")
        elif msg_type == 'auth_fail':
            log.error("Authentication failed!")
        else:
            log.debug(f"Unknown message type: {msg_type}")

    async def _handle_market_data(self, msg: dict):
        """Process market data tick."""
        self.tick_count += 1
        data = msg.get('data', {})
        timestamp = msg.get('timestamp', time.time() * 1000)

        # Update current data for strategy
        for symbol, tick in data.items():
            if isinstance(tick, dict):
                self.current_data[symbol] = {
                    'high': tick.get('high', tick.get('price', 0)),
                    'low': tick.get('low', tick.get('price', 0)),
                    'close': tick.get('price', tick.get('close', 0)),
                    'volume': tick.get('volume', 0),
                }
                self.last_prices[symbol] = tick.get('price', tick.get('close', 0))

        # Add to history
        self.history.append({
            'timestamp': timestamp,
            'data': data.copy(),
        })

        # Keep history manageable (last 1000 ticks)
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

        # Callback
        if self.on_tick:
            self.on_tick(self.tick_count, self.current_data)

    async def _handle_portfolio(self, msg: dict):
        """Process portfolio update."""
        self.portfolio = msg.get('portfolio', {})
        self.cash = msg.get('cash', self.cash)

    async def _handle_trade_ack(self, msg: dict):
        """Process trade acknowledgment."""
        symbol = msg.get('symbol', '')
        action = msg.get('action', '')
        qty = msg.get('quantity', 0)
        price = msg.get('price', 0)
        status = msg.get('status', 'unknown')

        log.info(f"Trade ACK: {action} {qty} {symbol} @ {price} - {status}")

        if self.on_trade:
            self.on_trade({
                'symbol': symbol,
                'action': action,
                'quantity': qty,
                'price': price,
                'status': status,
                'timestamp': time.time(),
            })

    async def send_order(self, symbol: str, action: str, quantity: int) -> bool:
        """Send an order to the server."""
        if not self.connected or not self.ws:
            log.warning(f"Cannot send order: not connected")
            return False

        if quantity <= 0:
            return False

        order = {
            "type": "order",
            "symbol": symbol,
            "action": action,  # "BUY" or "SELL"
            "quantity": quantity,
        }

        try:
            await self.ws.send(json.dumps(order))
            self.orders_sent += 1
            log.info(f"Order sent: {action} {quantity} {symbol}")
            return True
        except Exception as e:
            log.error(f"Order send failed: {e}")
            self.errors += 1
            return False

    async def disconnect(self):
        """Disconnect from server."""
        if self.ws:
            await self.ws.close()
            self.connected = False
            log.info("Disconnected")

    def get_stats(self) -> dict:
        """Get connection statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'connected': self.connected,
            'tick_count': self.tick_count,
            'tick_rate': self.tick_count / max(elapsed, 1),
            'orders_sent': self.orders_sent,
            'errors': self.errors,
            'elapsed': elapsed,
            'cash': self.cash,
            'portfolio': self.portfolio,
            'symbols_tracked': len(self.current_data),
        }


class SimulatedClient:
    """Simulated client for offline testing with CSV data."""

    def __init__(self, csv_path: str, symbols: list):
        self.csv_path = csv_path
        self.symbols = symbols
        self.connected = False
        self.authenticated = False

        self.current_data: Dict[str, Dict] = {}
        self.portfolio: Dict[str, Dict] = {}
        self.cash: float = 10_000_000
        self.history: list = []
        self.last_prices: Dict[str, float] = {}

        self.on_tick: Optional[Callable] = None
        self.on_trade: Optional[Callable] = None

        self.tick_count = 0
        self.start_time = None
        self.orders_sent = 0
        self.errors = 0

        self._data = None
        self._timestamps = None
        self._idx = 0
        self._pending_trades = []

    def load_data(self):
        """Load CSV data for simulation."""
        import pandas as pd
        df = pd.read_csv(self.csv_path).sort_values(['symbol', 'timestamp'])

        pivot_price = df.pivot_table(index='timestamp', columns='symbol', values='price')
        pivot_vol = df.pivot_table(index='timestamp', columns='symbol', values='volume')

        self._timestamps = pivot_price.index.values
        self._prices = pivot_price.values
        self._vols = pivot_vol.values.astype(float)
        self._symbols = pivot_price.columns.tolist()
        self._idx = 0

        log.info(f"Loaded {len(self._timestamps)} bars for {len(self._symbols)} symbols")

    def connect(self):
        """Simulate connection."""
        self.connected = True
        self.authenticated = True
        self.start_time = time.time()
        self.load_data()
        return True

    def get_next_tick(self) -> bool:
        """Get next tick from CSV data. Returns False when done."""
        if self._idx >= len(self._timestamps):
            return False

        ts = self._timestamps[self._idx]
        for i, sym in enumerate(self._symbols):
            price = self._prices[self._idx, i]
            vol = self._vols[self._idx, i]
            self.current_data[sym] = {
                'high': price * 1.001,
                'low': price * 0.999,
                'close': price,
                'volume': vol,
            }
            self.last_prices[sym] = price

        self.history.append({
            'timestamp': ts,
            'data': self.current_data.copy(),
        })

        if len(self.history) > 1000:
            self.history = self.history[-1000:]

        self.tick_count += 1
        self._idx += 1

        if self.on_tick:
            self.on_tick(self.tick_count, self.current_data)

        return True

    def send_order(self, symbol: str, action: str, quantity: int) -> bool:
        """Simulate order with portfolio tracking."""
        if quantity <= 0:
            return False
        price = self.last_prices.get(symbol, 0)
        if price <= 0:
            return False

        if action == "BUY":
            cost = quantity * price
            if cost > self.cash * 0.95:
                return False
            self.cash -= cost
            pos = self.portfolio.get(symbol, {})
            old_qty = pos.get('quantity', 0)
            old_avg = pos.get('avg_price', 0)
            new_qty = old_qty + quantity
            new_avg = (old_avg * old_qty + price * quantity) / new_qty if new_qty > 0 else price
            self.portfolio[symbol] = {'quantity': new_qty, 'avg_price': new_avg}
            self._pending_trades.append({
                'symbol': symbol, 'action': action, 'quantity': quantity,
                'price': price, 'status': 'simulated', 'timestamp': time.time(),
            })
            self.orders_sent += 1
            return True

        elif action == "SELL":
            pos = self.portfolio.get(symbol, {})
            held = pos.get('quantity', 0)
            avg_entry = pos.get('avg_price', 0)
            qty = min(quantity, held)
            if qty <= 0:
                return False
            pnl = (price - avg_entry) * qty
            self.cash += qty * price
            new_qty = held - qty
            if new_qty <= 0:
                self.portfolio[symbol] = {'quantity': 0, 'avg_price': 0}
            else:
                self.portfolio[symbol] = {'quantity': new_qty, 'avg_price': avg_entry}

            self._pending_trades.append({
                'symbol': symbol,
                'action': action,
                'quantity': qty,
                'price': price,
                'pnl': pnl,
                'status': 'simulated',
                'timestamp': time.time(),
            })
            self.orders_sent += 1
            return True

        self.orders_sent += 1
        self._pending_trades.append({
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'status': 'simulated',
            'timestamp': time.time(),
        })
        return True

    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'connected': self.connected,
            'tick_count': self.tick_count,
            'tick_rate': self.tick_count / max(elapsed, 1),
            'orders_sent': self.orders_sent,
            'errors': self.errors,
            'elapsed': elapsed,
            'cash': self.cash,
            'portfolio': self.portfolio,
            'symbols_tracked': len(self.current_data),
            'progress': f"{self._idx}/{len(self._timestamps)}" if self._timestamps is not None else "0/0",
        }
