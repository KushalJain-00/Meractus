"""
Feature engineering for Mercatus Arena trading strategies.
All functions operate on price/volume series and return numpy arrays or pandas Series.
"""
import numpy as np
import pandas as pd
from typing import Tuple


def ema(prices: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average. Returns array same length as prices (NaN-padded)."""
    out = np.full_like(prices, np.nan, dtype=float)
    if len(prices) < 1:
        return out
    alpha = 2.0 / (span + 1)
    out[0] = prices[0]
    for i in range(1, len(prices)):
        out[i] = alpha * prices[i] + (1 - alpha) * out[i - 1]
    return out


def sma(prices: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average."""
    out = np.full_like(prices, np.nan, dtype=float)
    if len(prices) < window:
        return out
    cumsum = np.cumsum(prices)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    out[window - 1:] = cumsum[window - 1:] / window
    return out


def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index (0-100)."""
    out = np.full_like(prices, np.nan, dtype=float)
    if len(prices) < period + 1:
        return out
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    # First valid RSI
    if avg_loss == 0:
        out[period] = 100.0
    else:
        first_avg_gain = np.mean(gains[:period])
        first_avg_loss = np.mean(losses[:period])
        if first_avg_loss > 0:
            out[period] = 100.0 - (100.0 / (1.0 + first_avg_gain / first_avg_loss))
        else:
            out[period] = 100.0
    return out


def bollinger_bands(prices: np.ndarray, window: int = 20, num_std: float = 2.0):
    """Returns (upper, middle, lower, z_score)."""
    middle = sma(prices, window)
    rolling_std = np.full_like(prices, np.nan, dtype=float)
    for i in range(window - 1, len(prices)):
        rolling_std[i] = np.std(prices[i - window + 1:i + 1], ddof=1)
    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std
    z_score = np.where(rolling_std > 0, (prices - middle) / rolling_std, 0.0)
    return upper, middle, lower, z_score


def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range."""
    out = np.full_like(close, np.nan, dtype=float)
    if len(close) < period + 1:
        return out
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    # Wilder smoothing
    out[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, len(close)):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average Directional Index."""
    out = np.full_like(close, np.nan, dtype=float)
    if len(close) < period * 2:
        return out
    tr = np.zeros(len(close))
    plus_dm = np.zeros(len(close))
    minus_dm = np.zeros(len(close))

    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0
        minus_dm[i] = down if (down > up and down > 0) else 0

    atr_val = np.mean(tr[1:period + 1])
    plus_dm_smooth = np.mean(plus_dm[1:period + 1])
    minus_dm_smooth = np.mean(minus_dm[1:period + 1])

    dx_vals = []
    for i in range(period, len(close)):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
        plus_dm_smooth = (plus_dm_smooth * (period - 1) + plus_dm[i]) / period
        minus_dm_smooth = (minus_dm_smooth * (period - 1) + minus_dm[i]) / period

        if atr_val > 0:
            plus_di = 100 * plus_dm_smooth / atr_val
            minus_di = 100 * minus_dm_smooth / atr_val
        else:
            plus_di = 0
            minus_di = 0

        di_sum = plus_di + minus_di
        if di_sum > 0:
            dx = 100 * abs(plus_di - minus_di) / di_sum
        else:
            dx = 0
        dx_vals.append(dx)

        if len(dx_vals) >= period:
            if len(dx_vals) == period:
                out[i] = np.mean(dx_vals)
            else:
                out[i] = (out[i - 1] * (period - 1) + dx) / period
    return out


def vwap(prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Volume Weighted Average Price (cumulative within session)."""
    cum_vol = np.cumsum(volumes)
    cum_pv = np.cumsum(prices * volumes)
    return np.where(cum_vol > 0, cum_pv / cum_vol, prices)


def returns(prices: np.ndarray, period: int = 1) -> np.ndarray:
    """Percentage returns."""
    out = np.full_like(prices, np.nan, dtype=float)
    if len(prices) > period:
        out[period:] = (prices[period:] - prices[:-period]) / prices[:-period]
    return out


def log_returns(prices: np.ndarray, period: int = 1) -> np.ndarray:
    """Log returns."""
    out = np.full_like(prices, np.nan, dtype=float)
    if len(prices) > period:
        out[period:] = np.log(prices[period:] / prices[:-period])
    return out


def volatility(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling volatility (annualized from returns)."""
    rets = returns(prices)
    out = np.full_like(prices, np.nan, dtype=float)
    for i in range(window, len(rets)):
        out[i] = np.std(rets[i - window + 1:i + 1], ddof=1)
    return out


def rolling_sharpe(prices: np.ndarray, window: int = 5, periods_per_day: int = 390) -> np.ndarray:
    """Rolling Sharpe ratio over window of returns."""
    rets = returns(prices)
    out = np.full_like(prices, np.nan, dtype=float)
    for i in range(window, len(rets)):
        segment = rets[i - window + 1:i + 1]
        mu = np.mean(segment)
        sigma = np.std(segment, ddof=1)
        if sigma > 0:
            out[i] = mu / sigma * np.sqrt(periods_per_day * 252)
    return out


def detect_regime(prices: np.ndarray, window: int = 5 * 390, threshold: float = 0.3) -> str:
    """
    Classify price series as 'trending' or 'mean_reverting'.
    Uses rolling Sharpe and Hurst exponent approximation.
    Returns regime string.
    """
    if len(prices) < window * 2:
        return 'unknown'

    # Rolling Sharpe approach
    sharpe = rolling_sharpe(prices, window=window)
    valid_sharpe = sharpe[~np.isnan(sharpe)]
    if len(valid_sharpe) == 0:
        return 'unknown'

    avg_sharpe = np.mean(np.abs(valid_sharpe[-window:]))

    # Hurst exponent approximation (rescaled range)
    rets = returns(prices)
    valid_rets = rets[~np.isnan(rets)]
    if len(valid_rets) < 100:
        return 'unknown'

    # Simple R/S analysis
    n = min(len(valid_rets), 500)
    segment = valid_rets[-n:]
    mean_r = np.mean(segment)
    deviations = np.cumsum(segment - mean_r)
    R = np.max(deviations) - np.min(deviations)
    S = np.std(segment, ddof=1)
    if S > 0:
        hurst = np.log(R / S) / np.log(n)
    else:
        hurst = 0.5

    if avg_sharpe > threshold or hurst > 0.55:
        return 'trending'
    elif hurst < 0.45:
        return 'mean_reverting'
    else:
        return 'random'


def flash_crash_detector(prices: np.ndarray, threshold: float = -0.10, window: int = 60) -> np.ndarray:
    """Detect flash crashes: price drops > threshold in window bars."""
    out = np.zeros(len(prices), dtype=bool)
    for i in range(window, len(prices)):
        pct_change = (prices[i] - prices[i - window]) / prices[i - window]
        if pct_change < threshold:
            out[i] = True
    return out


def momentum_score(prices: np.ndarray, windows: list = [5, 10, 20, 50]) -> np.ndarray:
    """Composite momentum score across multiple windows."""
    scores = np.zeros(len(prices))
    count = 0
    for w in windows:
        rets = returns(prices, w)
        valid = ~np.isnan(rets)
        scores[valid] += np.sign(rets[valid]) * np.minimum(np.abs(rets[valid]), 0.2)
        count += valid.astype(int)
    return np.where(count > 0, scores / count, 0.0)


def compute_all_features(prices: np.ndarray, volumes: np.ndarray,
                         highs: np.ndarray = None, lows: np.ndarray = None) -> dict:
    """Compute all features for a single symbol. Returns dict of feature arrays."""
    if highs is None:
        highs = prices
    if lows is None:
        lows = prices

    return {
        'ema_10': ema(prices, 10),
        'ema_30': ema(prices, 30),
        'ema_50': ema(prices, 50),
        'sma_20': sma(prices, 20),
        'rsi_14': rsi(prices, 14),
        'rsi_7': rsi(prices, 7),
        'macd': macd(prices)[0],
        'macd_signal': macd(prices)[1],
        'macd_hist': macd(prices)[2],
        'bb_upper': bollinger_bands(prices)[0],
        'bb_middle': bollinger_bands(prices)[1],
        'bb_lower': bollinger_bands(prices)[2],
        'bb_zscore': bollinger_bands(prices)[3],
        'atr_14': atr(highs, lows, prices, 14),
        'adx_14': adx(highs, lows, prices, 14),
        'returns_1': returns(prices, 1),
        'returns_5': returns(prices, 5),
        'returns_10': returns(prices, 10),
        'returns_20': returns(prices, 20),
        'volatility_20': volatility(prices, 20),
        'momentum': momentum_score(prices),
        'volume_sma_20': sma(volumes.astype(float), 20),
        'volume_ratio': volumes / np.maximum(sma(volumes.astype(float), 20), 1),
        'flash_crash': flash_crash_detector(prices),
    }
