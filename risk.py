"""
Risk management module for Mercatus Arena.
Handles position sizing, stop-losses, exposure limits, and portfolio-level constraints.
"""
import numpy as np
from typing import Dict, Tuple, Optional


class RiskManager:
    """Portfolio-level risk management."""

    def __init__(self, initial_capital: float = 10_000_000.0,
                 max_exposure_per_symbol: float = 0.20,
                 max_total_exposure: float = 0.85,
                 cash_reserve: float = 0.10,
                 stop_loss_pct: float = 0.05,
                 trailing_stop_pct: float = 0.05,
                 max_positions: int = 15,
                 flash_crash_recovery_pct: float = 0.30):
        self.initial_capital = initial_capital
        self.max_exposure_per_symbol = max_exposure_per_symbol
        self.max_total_exposure = max_total_exposure
        self.cash_reserve = cash_reserve
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_positions = max_positions
        self.flash_crash_recovery_pct = flash_crash_recovery_pct

        # State tracking
        self.entry_prices: Dict[str, float] = {}
        self.highest_prices: Dict[str, float] = {}
        self.position_values: Dict[str, float] = {}

    def reset(self):
        """Reset state for new run."""
        self.entry_prices.clear()
        self.highest_prices.clear()
        self.position_values.clear()

    def update_state(self, portfolio: Dict, current_prices: Dict):
        """Update internal state from portfolio and prices."""
        for sym, pos in portfolio.items():
            qty = pos.get('quantity', 0)
            if qty > 0:
                price = current_prices.get(sym, 0)
                self.position_values[sym] = qty * price
                if sym not in self.highest_prices:
                    self.highest_prices[sym] = price
                else:
                    self.highest_prices[sym] = max(self.highest_prices[sym], price)

    def can_buy(self, symbol: str, price: float, quantity: int,
                cash: float, portfolio: Dict, current_prices: Dict) -> bool:
        """Check if a buy order is allowed by risk rules."""
        total_equity = cash + sum(
            pos.get('quantity', 0) * current_prices.get(sym, 0)
            for sym, pos in portfolio.items()
        )

        # Cash reserve check
        order_value = price * quantity
        remaining_cash = cash - order_value
        if remaining_cash < total_equity * self.cash_reserve:
            return False

        # Per-symbol exposure check
        current_held = portfolio.get(symbol, {}).get('quantity', 0)
        current_val = current_held * current_prices.get(symbol, price)
        new_val = current_val + order_value
        if new_val > total_equity * self.max_exposure_per_symbol:
            return False

        # Total exposure check
        total_invested = sum(
            pos.get('quantity', 0) * current_prices.get(sym, 0)
            for sym, pos in portfolio.items()
        ) + order_value
        if total_invested > total_equity * self.max_total_exposure:
            return False

        # Max positions check
        active_positions = sum(1 for pos in portfolio.values() if pos.get('quantity', 0) > 0)
        if active_positions >= self.max_positions and symbol not in portfolio:
            return False

        return True

    def can_sell(self, symbol: str, quantity: int, portfolio: Dict) -> bool:
        """Check if a sell order is allowed."""
        held = portfolio.get(symbol, {}).get('quantity', 0)
        return held >= quantity

    def check_stop_loss(self, symbol: str, current_price: float, portfolio: Dict) -> bool:
        """Returns True if stop-loss triggered (should sell)."""
        if symbol not in self.entry_prices:
            return False
        entry = self.entry_prices[symbol]
        if current_price < entry * (1 - self.stop_loss_pct):
            return True
        # Trailing stop
        highest = self.highest_prices.get(symbol, entry)
        if current_price < highest * (1 - self.trailing_stop_pct):
            return True
        return False

    def check_flash_crash(self, symbol: str, current_price: float,
                          recent_prices: list, threshold: float = -0.10) -> bool:
        """Returns True if flash crash detected (buying opportunity if threshold positive)."""
        if len(recent_prices) < 60:
            return False
        old_price = recent_prices[-60]
        if old_price > 0:
            pct_change = (current_price - old_price) / old_price
            return pct_change < threshold
        return False

    def compute_position_size(self, symbol: str, price: float, cash: float,
                              portfolio: Dict, current_prices: Dict,
                              signal_strength: float = 1.0,
                              volatility: float = 0.02) -> int:
        """
        Compute optimal position size using volatility-adjusted sizing.
        signal_strength: 0-1 confidence in the trade
        volatility: annualized volatility of the symbol
        """
        total_equity = cash + sum(
            pos.get('quantity', 0) * current_prices.get(s, 0)
            for s, pos in portfolio.items()
        )

        # Base allocation: fraction of equity proportional to signal strength
        base_pct = min(self.max_exposure_per_symbol, 0.10 * signal_strength)

        # Volatility adjustment: higher vol = smaller position
        vol_scalar = min(1.0, 0.02 / max(volatility, 0.001))
        adjusted_pct = base_pct * vol_scalar

        # Dollar allocation
        dollar_alloc = total_equity * adjusted_pct

        # Cap by remaining cash (with reserve)
        available_cash = cash - total_equity * self.cash_reserve
        dollar_alloc = min(dollar_alloc, available_cash)

        # Convert to shares
        if price <= 0:
            return 0
        shares = int(dollar_alloc / price)

        # Ensure at least 1 share if we want to trade
        return max(shares, 0)

    def record_entry(self, symbol: str, price: float):
        """Record entry price for stop-loss tracking."""
        self.entry_prices[symbol] = price
        self.highest_prices[symbol] = price

    def record_exit(self, symbol: str):
        """Clean up on exit."""
        self.entry_prices.pop(symbol, None)
        self.highest_prices.pop(symbol, None)
        self.position_values.pop(symbol, None)

    def get_portfolio_stats(self, cash: float, portfolio: Dict,
                            current_prices: Dict) -> Dict:
        """Compute portfolio statistics."""
        equity = cash
        positions_value = 0
        for sym, pos in portfolio.items():
            qty = pos.get('quantity', 0)
            price = current_prices.get(sym, 0)
            val = qty * price
            positions_value += val
            equity += val

        exposure = positions_value / equity if equity > 0 else 0
        cash_pct = cash / equity if equity > 0 else 1

        return {
            'equity': equity,
            'cash': cash,
            'positions_value': positions_value,
            'exposure': exposure,
            'cash_pct': cash_pct,
            'num_positions': sum(1 for pos in portfolio.values() if pos.get('quantity', 0) > 0),
            'pnl': equity - self.initial_capital,
            'pnl_pct': (equity - self.initial_capital) / self.initial_capital,
        }
