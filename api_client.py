"""
Mercatus Arena API Client
REST + WebSocket client matching the actual competition protocol.
"""
import asyncio
import json
import logging
import time
import urllib.request
import urllib.error
from typing import Dict, Optional, Callable

try:
    import websockets
except ImportError:
    websockets = None

log = logging.getLogger('mercatus.api')


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.tokens = max_per_minute
        self.last_refill = time.time()

    def acquire(self) -> bool:
        now = time.time()
        elapsed = now - self.last_refill
        if elapsed > 60:
            self.tokens = self.max_per_minute
            self.last_refill = now
        elif elapsed > 0:
            self.tokens = min(self.max_per_minute, self.tokens + elapsed * (self.max_per_minute / 60))
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    def wait_time(self) -> float:
        if self.tokens >= 1:
            return 0
        return (1 - self.tokens) * (60 / self.max_per_minute)


class MercatusClient:
    """REST + WebSocket client for Mercatus Arena."""

    def __init__(self, base_url: str, api_key: str = '', email: str = '', password: str = ''):
        self.base_url = base_url.rstrip('/')
        self.ws_url = self.base_url.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws'
        self.api_key = api_key
        self.email = email
        self.password = password
        self.jwt = None

        self.connected = False
        self.ws = None

        self.current_data: Dict[str, Dict] = {}
        self.portfolio: Dict[str, Dict] = {}
        self.cash: float = 10_000_000
        self.history: list = []
        self.last_prices: Dict[str, float] = {}
        self.market_active = False

        self.on_tick: Optional[Callable] = None
        self.on_trade: Optional[Callable] = None

        self.tick_count = 0
        self.start_time = None
        self.orders_sent = 0
        self.api_calls = 0
        self.errors = 0

        self._trade_limiter = RateLimiter(60)
        self._api_limiter = RateLimiter(300)
        self._pending_trades = []

    def _http_request(self, method: str, path: str, data: dict = None) -> dict:
        """Make HTTP request with rate limiting."""
        if not self._api_limiter.acquire():
            wait = self._api_limiter.wait_time()
            log.warning(f"API rate limit, waiting {wait:.1f}s")
            time.sleep(wait)
            self._api_limiter.acquire()

        url = self.base_url + path
        headers = {'Content-Type': 'application/json'}
        if self.jwt:
            headers['Authorization'] = f'Bearer {self.jwt}'
        if self.api_key:
            headers['X-API-Key'] = self.api_key

        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                self.api_calls += 1
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ''
            if e.code == 401 and not getattr(req, '_retried', False):
                log.warning("JWT expired, relogging...")
                self.jwt = None
                self.login()
                req.add_header('Authorization', f'Bearer {self.jwt}')
                req._retried = True
                try:
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        self.api_calls += 1
                        return json.loads(resp.read())
                except Exception:
                    pass
            if 'RATE_LIMITED' in body:
                log.warning("Rate limited, backing off 2s...")
                time.sleep(2)
            else:
                log.error(f"HTTP {e.code}: {body}")
            self.errors += 1
            raise
        except Exception as e:
            log.error(f"HTTP error: {e}")
            self.errors += 1
            raise

    def login(self) -> bool:
        """Login with email/password or API key."""
        try:
            if self.api_key:
                log.info(f"Using API key: {self.api_key[:8]}...")
                return True

            result = self._http_request('POST', '/api/auth/login', {
                'email': self.email,
                'password': self.password,
            })
            self.jwt = result.get('token') or result.get('jwt')
            log.info("Logged in successfully")
            return True
        except Exception as e:
            log.error(f"Login failed: {e}")
            return False

    def check_market(self) -> bool:
        """Check if market is active."""
        try:
            result = self._http_request('GET', '/api/market/status')
            state = result.get('state', '')
            self.market_active = state == 'ACTIVE_MARKET'
            log.info(f"Market state: {state}")
            return self.market_active
        except Exception as e:
            log.warning(f"Market status check failed: {e}")
            return False

    def get_snapshot(self) -> dict:
        """Get current market snapshot via REST."""
        try:
            result = self._http_request('GET', '/api/market/snapshot')
            data = result.get('data', result.get('prices', {}))
            for symbol, tick in data.items():
                if isinstance(tick, dict):
                    self.current_data[symbol] = {
                        'high': tick.get('high', tick.get('price', 0)),
                        'low': tick.get('low', tick.get('price', 0)),
                        'close': tick.get('price', tick.get('close', 0)),
                        'volume': tick.get('volume', 0),
                    }
                    self.last_prices[symbol] = tick.get('price', tick.get('close', 0))
            return self.current_data
        except Exception as e:
            log.warning(f"Snapshot failed: {e}")
            return {}

    def get_portfolio(self) -> dict:
        """Get current portfolio from server."""
        try:
            result = self._http_request('GET', '/api/team/portfolio')
            self.cash = result.get('cash', self.cash)
            raw_portfolio = result.get('portfolio', result.get('holdings', {}))
            for sym, pos in raw_portfolio.items():
                self.portfolio[sym] = {
                    'quantity': pos.get('quantity', 0),
                    'avg_price': pos.get('avg_price', pos.get('avgPrice', 0)),
                }
            return self.portfolio
        except Exception as e:
            log.warning(f"Portfolio fetch failed: {e}")
            return self.portfolio

    def send_order(self, symbol: str, action: str, quantity: int) -> bool:
        """Send order via REST."""
        if quantity <= 0:
            return False

        if not self._trade_limiter.acquire():
            wait = self._trade_limiter.wait_time()
            log.warning(f"Trade rate limit, waiting {wait:.1f}s")
            time.sleep(wait)
            self._trade_limiter.acquire()

        path = '/api/trade/buy' if action == 'BUY' else '/api/trade/sell'
        try:
            result = self._http_request('POST', path, {
                'symbol': symbol,
                'quantity': quantity,
            })
            self.orders_sent += 1
            log.info(f"ORDER OK: {action} {quantity} {symbol}")

            trade = {
                'symbol': symbol, 'action': action, 'quantity': quantity,
                'price': result.get('fill_price', self.last_prices.get(symbol, 0)),
                'status': result.get('status', 'filled'),
            }
            self._pending_trades.append(trade)
            return True
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ''
            if 'RATE_LIMITED' in body:
                log.warning("Rate limited, backing off...")
                time.sleep(2)
            elif 'INSUFFICIENT_POSITION' in body:
                log.warning(f"Cannot sell {symbol}: insufficient position")
            else:
                log.error(f"Order failed: {body}")
            self.errors += 1
            return False
        except Exception as e:
            log.error(f"Order error: {e}")
            self.errors += 1
            return False

    async def connect_ws(self) -> bool:
        """Connect WebSocket for live ticks."""
        if websockets is None:
            log.warning("websockets not installed, using REST polling")
            return False

        try:
            headers = {}
            if self.jwt:
                headers['Authorization'] = f'Bearer {self.jwt}'
            if self.api_key:
                headers['X-API-Key'] = self.api_key

            self.ws = await asyncio.wait_for(
                websockets.connect(self.ws_url, extra_headers=headers, ping_interval=20),
                timeout=10,
            )
            self.connected = True
            log.info(f"WebSocket connected to {self.ws_url}")
            return True
        except Exception as e:
            log.warning(f"WebSocket connect failed: {e}")
            return False

    async def listen_ws(self):
        """Listen for WebSocket messages."""
        if not self.ws:
            return
        try:
            async for message in self.ws:
                msg = json.loads(message)
                msg_type = msg.get('type', '')

                if msg_type in ('tick', 'market_data', 'price_update'):
                    data = msg.get('data', msg.get('prices', {}))
                    for symbol, tick in data.items():
                        if isinstance(tick, dict):
                            self.current_data[symbol] = {
                                'high': tick.get('high', tick.get('price', 0)),
                                'low': tick.get('low', tick.get('price', 0)),
                                'close': tick.get('price', tick.get('close', 0)),
                                'volume': tick.get('volume', 0),
                            }
                            self.last_prices[symbol] = tick.get('price', tick.get('close', 0))
                    self.tick_count += 1
                    self.history.append({'timestamp': msg.get('t', time.time()), 'data': data})
                    if len(self.history) > 1000:
                        self.history = self.history[-1000:]
                    if self.on_tick:
                        self.on_tick(self.tick_count, self.current_data)

                elif msg_type == 'fill':
                    if self.on_trade:
                        self.on_trade(msg)

                elif msg_type == 'portfolio_update':
                    self.portfolio = msg.get('portfolio', {})
                    self.cash = msg.get('cash', self.cash)

        except Exception as e:
            log.warning(f"WebSocket listen error: {e}")
            self.connected = False

    def poll_loop_sync(self, interval: float = 1.0):
        """Synchronous REST polling loop for simulation."""
        self.start_time = time.time()
        self.get_snapshot()
        self.get_portfolio()
        self.tick_count += 1
        self.history.append({'timestamp': time.time(), 'data': self.current_data.copy()})

    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'connected': self.connected,
            'tick_count': self.tick_count,
            'tick_rate': self.tick_count / max(elapsed, 1),
            'orders_sent': self.orders_sent,
            'api_calls': self.api_calls,
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

        self.current_data: Dict[str, Dict] = {}
        self.portfolio: Dict[str, Dict] = {}
        self.cash: float = 10_000_000
        self.history: list = []
        self.last_prices: Dict[str, float] = {}
        self.market_active = True

        self.on_tick: Optional[Callable] = None
        self.on_trade: Optional[Callable] = None

        self.tick_count = 0
        self.start_time = None
        self.orders_sent = 0
        self.api_calls = 0
        self.errors = 0

        self._trade_limiter = RateLimiter(60)
        self._api_limiter = RateLimiter(300)
        self._pending_trades = []

        self._prices = None
        self._vols = None
        self._timestamps = None
        self._idx = 0

    def load_data(self):
        import pandas as pd
        df = pd.read_csv(self.csv_path).sort_values(['symbol', 'timestamp'])
        pivot_price = df.pivot_table(index='timestamp', columns='symbol', values='price')
        pivot_vol = df.pivot_table(index='timestamp', columns='symbol', values='volume')
        self._timestamps = pivot_price.index.values
        self._prices = pivot_price.values
        self._vols = pivot_vol.values.astype(float)
        self._symbols = pivot_price.columns.tolist()

    def connect(self) -> bool:
        self.connected = True
        self.start_time = time.time()
        self.load_data()
        return True

    def login(self) -> bool:
        return True

    def check_market(self) -> bool:
        return True

    def get_snapshot(self) -> dict:
        if self._idx >= len(self._timestamps):
            return {}
        for i, sym in enumerate(self._symbols):
            price = self._prices[self._idx, i]
            vol = self._vols[self._idx, i]
            self.current_data[sym] = {
                'high': price * 1.001, 'low': price * 0.999,
                'close': price, 'volume': vol,
            }
            self.last_prices[sym] = price
        return self.current_data

    def get_portfolio(self) -> dict:
        return self.portfolio

    def get_next_tick(self) -> bool:
        if self._idx >= len(self._timestamps):
            return False
        self.get_snapshot()
        ts = self._timestamps[self._idx]
        self.history.append({'timestamp': ts, 'data': self.current_data.copy()})
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        self.tick_count += 1
        self._idx += 1
        if self.on_tick:
            self.on_tick(self.tick_count, self.current_data)
        return True

    def send_order(self, symbol: str, action: str, quantity: int) -> bool:
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
                'symbol': symbol, 'action': action, 'quantity': qty,
                'price': price, 'pnl': pnl, 'status': 'simulated', 'timestamp': time.time(),
            })
            self.orders_sent += 1
            return True

        return False

    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'connected': self.connected,
            'tick_count': self.tick_count,
            'tick_rate': self.tick_count / max(elapsed, 1),
            'orders_sent': self.orders_sent,
            'api_calls': self.api_calls,
            'errors': self.errors,
            'elapsed': elapsed,
            'cash': self.cash,
            'portfolio': self.portfolio,
            'symbols_tracked': len(self.current_data),
            'progress': f"{self._idx}/{len(self._timestamps)}" if self._timestamps is not None else "0/0",
        }
