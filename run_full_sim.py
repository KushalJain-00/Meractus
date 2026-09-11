import json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_client import SimulatedClient
from strategy import strategy, _state
from dashboard import Dashboard

with open('dataset_meta.json') as f:
    meta = json.load(f)

client = SimulatedClient('paper_dataset(1).csv', meta['symbols'])
client.connect()
dashboard = Dashboard()

trades = []
tick = 0
t0 = time.time()
total_bars = len(client._timestamps)
equity_curve = []

while client.get_next_tick():
    tick += 1
    if tick < 100:
        equity_curve.append(10_000_000)
        continue

    actions = strategy(client.current_data, client.portfolio, client.cash, client.history)

    for sym, (action, qty) in actions.items():
        if qty > 0:
            ok = client.send_order(sym, action, qty)
            if ok:
                pass  # trades logged via _pending_trades below

    # Drain pending trades (with PnL computed by client)
    while client._pending_trades:
        t = client._pending_trades.pop(0)
        t['tick'] = tick
        trades.append(t)
        dashboard.log_trade(t)

    equity = client.cash
    for s, p in client.portfolio.items():
        qty = p.get('quantity', 0)
        price = client.current_data.get(s, {}).get('close', p.get('avg_price', 0))
        equity += qty * price
    equity_curve.append(equity)

    if tick % 5000 == 0:
        elapsed = time.time() - t0
        ret = (equity - 10e6) / 10e6 * 100
        print(f"{tick}/{total_bars} ({tick/total_bars*100:.0f}%) | {tick/elapsed:.0f}/s | Trades: {len(trades)} | ${equity:,.0f} ({ret:+.1f}%)")

elapsed = time.time() - t0
equity = client.cash
for s, p in client.portfolio.items():
    equity += p.get('quantity', 0) * client.last_prices.get(s, p.get('avg_price', 0))

total_return = (equity - 10e6) / 10e6 * 100
pnls = [t.get('pnl', 0) for t in trades if 'pnl' in t]
wins = sum(1 for p in pnls if p > 0)

print(f"\n=== DONE: {tick} ticks, {elapsed:.0f}s ===")
print(f"Equity: ${equity:,.0f} ({total_return:+.2f}%)")
print(f"Trades: {len(trades)} | PnL trades: {len(pnls)} | Winners: {wins}/{len(pnls)}")
if pnls:
    avg_win = sum(p for p in pnls if p > 0) / max(wins, 1)
    avg_loss = sum(p for p in pnls if p < 0) / max(len(pnls) - wins, 1)
    print(f"Avg Win: ${avg_win:+,.0f} | Avg Loss: ${avg_loss:+,.0f}")

dashboard.equity_curve = equity_curve
dashboard.save_state(client.get_stats())
dashboard.generate_graphs()
print("Graphs saved to logs/")
