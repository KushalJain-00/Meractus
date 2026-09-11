"""
Strategy 1: EMA Crossover Momentum with Regime Filter
-----------------------------------------------------
Core idea: Buy when fast EMA crosses above slow EMA (golden cross),
sell when fast crosses below slow (death cross).
Only trades symbols classified as 'trending' by regime detector.

Formulas:
  EMA_t = alpha * Price_t + (1 - alpha) * EMA_{t-1}
  alpha = 2 / (span + 1)
  Signal: BUY if EMA_fast > EMA_slow AND regime == 'trending'
          SELL if EMA_fast < EMA_slow
  Stop-loss: trailing stop at -5% from highest price since entry
"""
import numpy as np
from typing import Dict, List, Tuple
from features import ema, detect_regime, rolling_sharpe, rsi, volatility, returns


class MomentumStrategy:
    """EMA crossover momentum with regime filter."""

    def __init__(self, fast_period: int = 10, slow_period: int = 30,
                 regime_window: int = 5 * 390, regime_threshold: float = 0.3,
                 stop_loss_pct: float = 0.05, trailing_stop_pct: float = 0.05,
                 min_hold_bars: int = 60, rsi_filter: bool = True,
                 rsi_boost: int = 4):
        """
        rsi_boost: delay RSI entry by N points (from IIT KGP winning strategy).
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.regime_window = regime_window
        self.regime_threshold = regime_threshold
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.min_hold_bars = min_hold_bars
        self.rsi_filter = rsi_filter
        self.rsi_boost = rsi_boost

        # State per symbol
        self.highest_prices: Dict[str, float] = {}
        self.entry_bars: Dict[str, int] = {}
        self.regime_cache: Dict[str, str] = {}
        self.last_regime_check: Dict[str, int] = {}

    def reset(self):
        self.highest_prices.clear()
        self.entry_bars.clear()
        self.regime_cache.clear()
        self.last_regime_check.clear()

    def _get_regime(self, symbol: str, prices: np.ndarray, bar_idx: int) -> str:
        """Get regime, cache it, re-check every 500 bars."""
        if symbol not in self.last_regime_check or bar_idx - self.last_regime_check[symbol] > 500:
            self.regime_cache[symbol] = detect_regime(
                prices[:bar_idx + 1],
                window=min(self.regime_window, bar_idx // 2),
                threshold=self.regime_threshold
            )
            self.last_regime_check[symbol] = bar_idx
        return self.regime_cache.get(symbol, 'unknown')

    def signal(self, symbol: str, prices: np.ndarray, bar_idx: int,
               volumes: np.ndarray = None) -> Tuple[str, float]:
        """
        Generate trading signal for a symbol at a given bar.
        Returns: (action, confidence)
            action: 'BUY', 'SELL', or 'HOLD'
            confidence: 0-1 signal strength
        """
        if bar_idx < self.slow_period + 10:
            return ('HOLD', 0.0)

        price_slice = prices[:bar_idx + 1]
        current_price = prices[bar_idx]

        # Compute EMAs
        ema_fast = ema(price_slice, self.fast_period)
        ema_slow = ema(price_slice, self.slow_period)

        if np.isnan(ema_fast[-1]) or np.isnan(ema_slow[-1]):
            return ('HOLD', 0.0)

        fast_val = ema_fast[-1]
        slow_val = ema_slow[-1]
        prev_fast = ema_fast[-2] if len(ema_fast) > 1 else fast_val
        prev_slow = ema_slow[-2] if len(ema_slow) > 1 else slow_val

        # Regime filter
        regime = self._get_regime(symbol, prices, bar_idx)

        # RSI filter (with boost: require RSI < 30+boost for oversold entry)
        rsi_val = 50.0
        if self.rsi_filter:
            rsi_arr = rsi(price_slice, 14)
            rsi_val = rsi_arr[-1] if not np.isnan(rsi_arr[-1]) else 50.0

        # Volatility for position sizing
        vol_arr = volatility(price_slice, 20)
        current_vol = vol_arr[-1] if not np.isnan(vol_arr[-1]) else 0.02

        # Golden cross: fast crosses above slow
        golden_cross = prev_fast <= prev_slow and fast_val > slow_val
        death_cross = prev_fast >= prev_slow and fast_val < slow_val

        # Holding trend: fast > slow
        uptrend = fast_val > slow_val
        downtrend = fast_val < slow_val

        # Stop-loss check
        if symbol in self.highest_prices:
            self.highest_prices[symbol] = max(self.highest_prices[symbol], current_price)
        else:
            self.highest_prices[symbol] = current_price

        # Minimum hold period check
        bars_held = bar_idx - self.entry_bars.get(symbol, bar_idx) if symbol in self.entry_bars else 0

        # SELL signals
        if death_cross and bars_held > self.min_hold_bars:
            return ('SELL', 0.9)
        if symbol in self.highest_prices and bars_held > self.min_hold_bars:
            if current_price < self.highest_prices[symbol] * (1 - self.trailing_stop_pct):
                return ('SELL', 1.0)  # Stop-loss always fires

        # BUY signals
        if regime == 'trending' and golden_cross:
            # RSI filter: don't buy overbought (with boost)
            if rsi_val < (70 + self.rsi_boost):
                confidence = min(1.0, abs(fast_val - slow_val) / slow_val * 100)
                return ('BUY', confidence)

        # Strong trend continuation: add to winners
        if regime == 'trending' and uptrend:
            # Price pullback to fast EMA = add opportunity
            pullback = (fast_val - current_price) / fast_val
            if 0 < pullback < 0.02 and rsi_val < (50 + self.rsi_boost):
                return ('BUY', 0.5)

        return ('HOLD', 0.0)

    def on_fill(self, symbol: str, action: str, price: float, bar_idx: int):
        """Called when an order is filled."""
        if action == 'BUY':
            self.entry_bars[symbol] = bar_idx
            self.highest_prices[symbol] = price
        elif action == 'SELL':
            self.entry_bars.pop(symbol, None)
            self.highest_prices.pop(symbol, None)
