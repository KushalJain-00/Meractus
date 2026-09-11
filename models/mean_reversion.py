"""
Strategy 2: Mean Reversion + Flash Crash Plays
----------------------------------------------
Core idea: Buy oversold dips (z-score < -2), sell at mean reversion.
Buy flash crashes for quick pop. Quick exits at +2% profit.

Formulas:
  z_score = (Price - SMA) / Rolling_Std
  Entry: z_score < -entry_threshold (buy oversold)
  Exit: z_score > exit_threshold OR profit > take_profit_pct
  Flash crash: buy if price drops > flash_crash_pct in flash_crash_window bars

RSI Boosting (from IIT KGP winner):
  Instead of RSI < 30, use RSI < 30 + rsi_boost (typically +3-4 points)
  Filters 40% false signals while keeping 95% true entries.
"""
import numpy as np
from typing import Dict, Tuple
from features import sma, rsi, bollinger_bands, volatility, flash_crash_detector, returns


class MeanReversionStrategy:
    """Z-score mean reversion with flash crash detection."""

    def __init__(self, lookback: int = 20, entry_threshold: float = 2.0,
                 exit_threshold: float = 0.0, take_profit_pct: float = 0.02,
                 stop_loss_pct: float = 0.05, rsi_entry: float = 30.0,
                 rsi_boost: int = 4, flash_crash_pct: float = -0.10,
                 flash_crash_window: int = 60, flash_crash_exit: float = 0.05,
                 min_hold_bars: int = 10):
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.rsi_entry = rsi_entry
        self.rsi_boost = rsi_boost
        self.flash_crash_pct = flash_crash_pct
        self.flash_crash_window = flash_crash_window
        self.flash_crash_exit = flash_crash_exit
        self.min_hold_bars = min_hold_bars

        # State
        self.entry_prices: Dict[str, float] = {}
        self.entry_bars: Dict[str, int] = {}
        self.highest_prices: Dict[str, float] = {}
        self.is_flash_play: Dict[str, bool] = {}

    def reset(self):
        self.entry_prices.clear()
        self.entry_bars.clear()
        self.highest_prices.clear()
        self.is_flash_play.clear()

    def signal(self, symbol: str, prices: np.ndarray, bar_idx: int,
               volumes: np.ndarray = None) -> Tuple[str, float]:
        """
        Generate mean reversion signal.
        Returns: (action, confidence)
        """
        if bar_idx < self.lookback + 14:
            return ('HOLD', 0.0)

        price_slice = prices[:bar_idx + 1]
        current_price = prices[bar_idx]

        # Compute z-score
        _, _, _, z_scores = bollinger_bands(price_slice, self.lookback, 2.0)
        z = z_scores[-1] if not np.isnan(z_scores[-1]) else 0.0

        # RSI with boost
        rsi_arr = rsi(price_slice, 14)
        rsi_val = rsi_arr[-1] if not np.isnan(rsi_arr[-1]) else 50.0

        # Flash crash detection
        is_flash = False
        if bar_idx >= self.flash_crash_window:
            old_price = prices[bar_idx - self.flash_crash_window]
            if old_price > 0:
                pct = (current_price - old_price) / old_price
                if pct < self.flash_crash_pct:
                    is_flash = True

        # Bars held
        bars_held = bar_idx - self.entry_bars.get(symbol, bar_idx) if symbol in self.entry_bars else 0

        # Current PnL if in position
        in_position = symbol in self.entry_prices
        if in_position:
            entry = self.entry_prices[symbol]
            pnl_pct = (current_price - entry) / entry
        else:
            pnl_pct = 0.0

        # Stop-loss check (always active)
        if in_position and bars_held > self.min_hold_bars:
            if pnl_pct < -self.stop_loss_pct:
                return ('SELL', 1.0)

        # Highest price tracking
        if symbol in self.highest_prices:
            self.highest_prices[symbol] = max(self.highest_prices[symbol], current_price)
        else:
            self.highest_prices[symbol] = current_price

        # Trailing stop
        if in_position and bars_held > self.min_hold_bars:
            if current_price < self.highest_prices[symbol] * 0.95:
                return ('SELL', 0.9)

        # SELL signals (mean reversion exit)
        if in_position:
            # Flash play exit: quick profit
            if self.is_flash_play.get(symbol, False):
                if pnl_pct > self.flash_crash_exit:
                    return ('SELL', 0.8)
                if pnl_pct < -self.stop_loss_pct:
                    return ('SELL', 1.0)

            # Normal mean reversion exit: z-score returns to mean
            if z > self.exit_threshold and bars_held > self.min_hold_bars:
                return ('SELL', 0.7)

            # Take profit
            if pnl_pct > self.take_profit_pct:
                return ('SELL', 0.8)

        # BUY signals
        if not in_position:
            # Flash crash buy: immediate entry
            if is_flash:
                return ('BUY', 0.9)

            # Mean reversion entry: oversold dip
            if z < -self.entry_threshold:
                # RSI confirmation with boost: RSI < 30 + boost
                if rsi_val < (self.rsi_entry + self.rsi_boost):
                    # Confidence based on how oversold
                    confidence = min(1.0, abs(z) / 3.0)
                    return ('BUY', confidence)

            # Secondary: buy on Bollinger lower band touch
            bb_upper, bb_middle, bb_lower, _ = bollinger_bands(price_slice, self.lookback, 2.0)
            if current_price < bb_lower[-1] * 1.001 and rsi_val < (40 + self.rsi_boost):
                return ('BUY', 0.5)

        return ('HOLD', 0.0)

    def on_fill(self, symbol: str, action: str, price: float, bar_idx: int):
        """Called when order is filled."""
        if action == 'BUY':
            self.entry_prices[symbol] = price
            self.entry_bars[symbol] = bar_idx
            self.highest_prices[symbol] = price
            # Check if this was a flash play
            self.is_flash_play[symbol] = True  # will be cleared if not
        elif action == 'SELL':
            self.entry_prices.pop(symbol, None)
            self.entry_bars.pop(symbol, None)
            self.highest_prices.pop(symbol, None)
            self.is_flash_play.pop(symbol, None)
