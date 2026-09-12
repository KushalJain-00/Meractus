"""
Mock Mercatus Server for end-to-end testing.
Serves CSV data as live ticks via WebSocket + REST.
"""
import asyncio
import json
import time
import sys
import pandas as pd
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

try:
    import websockets
except ImportError:
    websockets = None


class MockDataFeed:
    """Loads CSV and serves ticks."""

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path).sort_values(['symbol', 'timestamp'])
        pivot_price = df.pivot_table(index='timestamp', columns='symbol', values='price')
        pivot_vol = df.pivot_table(index='timestamp', columns='symbol', values='volume')
        self.timestamps = pivot_price.index.values
        self.prices = pivot_price.values
        self.vols = pivot_vol.values.astype(float)
        self.symbols = pivot_price.columns.tolist()
        self.idx = 0
        self.current_data = {}
        self.start_time = time.time()
        print(f"Loaded {len(self.timestamps)} ticks for {len(self.symbols)} symbols")

    def get_tick(self) -> dict:
        if self.idx >= len(self.timestamps):
            return None
        data = {}
        for i, sym in enumerate(self.symbols):
            data[sym] = {
                'price': float(self.prices[self.idx, i]),
                'high': float(self.prices[self.idx, i] * 1.001),
                'low': float(self.prices[self.idx, i] * 0.999),
                'volume': int(self.vols[self.idx, i]),
            }
        ts = int(self.timestamps[self.idx])
        self.idx += 1
        self.current_data = data
        return {'t': ts, 'data': data}


class MockRESTHandler(BaseHTTPRequestHandler):
    """REST API mock."""

    feed = None
    portfolio = {}
    cash = 10_000_000

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == '/api/market/status':
            self._json_response({'state': 'ACTIVE_MARKET'})
        elif self.path == '/api/market/snapshot':
            if self.feed:
                tick = self.feed.get_tick()
                self._json_response({'data': tick['data'] if tick else {}})
            else:
                self._json_response({'data': {}})
        elif self.path == '/api/team/portfolio':
            self._json_response({'cash': self.cash, 'portfolio': self.portfolio})
        elif self.path == '/api/market/leaderboard':
            self._json_response({'rankings': []})
        else:
            self._json_response({'error': 'not found'}, 404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length)) if length > 0 else {}

        if self.path == '/api/auth/register':
            self._json_response({'token': 'mock_jwt_token', 'api_key': 'sk_mock_abc123'})
        elif self.path == '/api/auth/login':
            self._json_response({'token': 'mock_jwt_token'})
        elif self.path == '/api/trade/buy':
            symbol = body.get('symbol', '')
            qty = body.get('quantity', 0)
            price = self.feed.current_data.get(symbol, {}).get('price', 0) if self.feed else 0
            cost = qty * price
            if cost > self.cash * 0.95:
                self._json_response({'error': 'INSUFFICIENT_CASH'}, 400)
                return
            self.cash -= cost
            pos = self.portfolio.get(symbol, {})
            old_qty = pos.get('quantity', 0)
            new_qty = old_qty + qty
            new_avg = (pos.get('avg_price', 0) * old_qty + price * qty) / new_qty if new_qty > 0 else price
            self.portfolio[symbol] = {'quantity': new_qty, 'avg_price': new_avg}
            self._json_response({'status': 'filled', 'fill_price': price})
        elif self.path == '/api/trade/sell':
            symbol = body.get('symbol', '')
            qty = body.get('quantity', 0)
            pos = self.portfolio.get(symbol, {})
            held = pos.get('quantity', 0)
            if qty > held:
                self._json_response({'error': 'INSUFFICIENT_POSITION'}, 400)
                return
            price = self.feed.current_data.get(symbol, {}).get('price', 0) if self.feed else 0
            self.cash += qty * price
            new_qty = held - qty
            if new_qty <= 0:
                self.portfolio[symbol] = {'quantity': 0, 'avg_price': 0}
            else:
                self.portfolio[symbol] = {'quantity': new_qty, 'avg_price': pos.get('avg_price', price)}
            self._json_response({'status': 'filled', 'fill_price': price})
        else:
            self._json_response({'error': 'not found'}, 404)

    def _json_response(self, data, code=200):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())


class MockWebSocketServer:
    """WebSocket server that pushes ticks."""

    def __init__(self, feed: MockDataFeed):
        self.feed = feed
        self.clients = set()

    async def handler(self, ws, path=None):
        self.clients.add(ws)
        print(f"Client connected ({len(self.clients)} total)")
        try:
            while True:
                tick = self.feed.get_tick()
                if tick is None:
                    await asyncio.sleep(0.1)
                    continue
                msg = json.dumps({'type': 'tick', 't': tick['t'], 'data': tick['data']})
                await ws.send(msg)
                await asyncio.sleep(0.01)
        except Exception:
            pass
        finally:
            self.clients.discard(ws)
            print(f"Client disconnected ({len(self.clients)} total)")


async def run_mock_server(csv_path: str, rest_port: int = 4040, ws_port: int = 8080):
    """Run full mock server."""
    feed = MockDataFeed(csv_path)
    MockRESTHandler.feed = feed

    rest_server = HTTPServer(('0.0.0.0', rest_port), MockRESTHandler)
    rest_thread = Thread(target=rest_server.serve_forever, daemon=True)
    rest_thread.start()
    print(f"REST server on http://localhost:{rest_port}")

    if websockets:
        ws_server = MockWebSocketServer(feed)
        ws_handler = await websockets.serve(ws_server.handler, '0.0.0.0', ws_port)
        print(f"WebSocket server on ws://localhost:{ws_port}")
    else:
        print("websockets not installed, WebSocket server disabled")

    print("Mock server ready! Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        rest_server.shutdown()
        print("\nMock server stopped.")


if __name__ == '__main__':
    csv = sys.argv[1] if len(sys.argv) > 1 else 'paper_dataset(1).csv'
    asyncio.run(run_mock_server(csv))
