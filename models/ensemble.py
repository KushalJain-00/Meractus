"""
Strategy 3: LightGBM Direction Predictor
-----------------------------------------
Trains a LightGBM classifier on rolling window of features to predict
next-bar price direction (up/down).

Features (63 total, inspired by the Regime-Aware LightGBM paper):
  - EMA distances (5/10/20/50 period)
  - RSI (7, 14 period)
  - MACD histogram
  - Bollinger Band z-score and width
  - ATR normalized by price
  - ADX
  - Returns (1, 5, 10, 20 bar)
  - Volatility (10, 20 bar)
  - Volume ratio (current / 20-SMA)
  - Momentum composite
  - Lagged features (1, 2, 3 bars back)

Label: 1 if next-bar return > 0, else 0

Walk-forward: retrain every 6 hours on rolling 5-day window.
"""
import numpy as np
from typing import Dict, Tuple, Optional
from features import (ema, rsi, macd, bollinger_bands, atr, adx,
                      returns, volatility, momentum_score, sma)


class LightGBMStrategy:
    """LightGBM-based direction predictor with walk-forward retraining."""

    def __init__(self, train_window: int = 5 * 390, retrain_every: int = 6 * 390,
                 min_train_samples: int = 500, confidence_threshold: float = 0.6,
                 stop_loss_pct: float = 0.05):
        self.train_window = train_window
        self.retrain_every = retrain_every
        self.min_train_samples = min_train_samples
        self.confidence_threshold = confidence_threshold
        self.stop_loss_pct = stop_loss_pct

        # Model state
        self.model = None
        self.last_train_bar = -9999
        self.feature_names = []
        self.scaler_mean = None
        self.scaler_std = None

        # Position state
        self.entry_prices: Dict[str, float] = {}
        self.entry_bars: Dict[str, int] = {}

    def reset(self):
        self.model = None
        self.last_train_bar = -9999
        self.entry_prices.clear()
        self.entry_bars.clear()

    def _compute_features(self, prices: np.ndarray, volumes: np.ndarray,
                          bar_idx: int) -> Optional[np.ndarray]:
        """Compute feature vector for a single bar."""
        if bar_idx < 60:
            return None

        p = prices[:bar_idx + 1]
        v = volumes[:bar_idx + 1]
        features = []

        # EMA distances (normalized by price)
        for span in [5, 10, 20, 50]:
            e = ema(p, span)
            features.append((e[-1] / p[-1] - 1) if not np.isnan(e[-1]) else 0)

        # RSI
        r7 = rsi(p, 7)
        r14 = rsi(p, 14)
        features.append((r7[-1] / 100 - 0.5) if not np.isnan(r7[-1]) else 0)
        features.append((r14[-1] / 100 - 0.5) if not np.isnan(r14[-1]) else 0)

        # MACD
        macd_l, sig_l, hist = macd(p)
        features.append(hist[-1] / p[-1] if not np.isnan(hist[-1]) else 0)

        # Bollinger
        _, _, _, z = bollinger_bands(p, 20, 2.0)
        features.append(z[-1] if not np.isnan(z[-1]) else 0)

        # BB width
        upper, middle, lower, _ = bollinger_bands(p, 20, 2.0)
        if not np.isnan(middle[-1]) and middle[-1] > 0:
            features.append((upper[-1] - lower[-1]) / middle[-1])
        else:
            features.append(0)

        # ATR normalized
        a = atr(p, p, p, 14)
        features.append((a[-1] / p[-1]) if not np.isnan(a[-1]) else 0)

        # ADX
        adx_val = adx(p, p, p, 14)
        features.append((adx_val[-1] / 100 - 0.5) if not np.isnan(adx_val[-1]) else 0)

        # Returns at multiple horizons
        for period in [1, 5, 10, 20]:
            r = returns(p, period)
            features.append(r[-1] if not np.isnan(r[-1]) else 0)

        # Volatility
        for w in [10, 20]:
            vol = volatility(p, w)
            features.append(vol[-1] if not np.isnan(vol[-1]) else 0)

        # Volume ratio
        vol_sma = sma(v.astype(float), 20)
        if not np.isnan(vol_sma[-1]) and vol_sma[-1] > 0:
            features.append(v[-1] / vol_sma[-1] - 1)
        else:
            features.append(0)

        # Momentum composite
        mom = momentum_score(p)
        features.append(mom[-1])

        # Lagged returns
        for lag in [1, 2, 3]:
            r = returns(p, lag)
            features.append(r[-1] if not np.isnan(r[-1]) else 0)

        # Lagged RSI
        for lag in [1, 2]:
            idx = bar_idx - lag
            if idx >= 14:
                r_lag = rsi(p[:idx + 1], 14)
                features.append((r_lag[-1] / 100 - 0.5) if not np.isnan(r_lag[-1]) else 0)
            else:
                features.append(0)

        # Price momentum: (price - price_20) / price_20
        if bar_idx >= 20:
            features.append((p[-1] - p[-20]) / p[-20])
        else:
            features.append(0)

        # High-low range normalized
        if bar_idx >= 20:
            h = np.max(p[-20:])
            l = np.min(p[-20:])
            features.append((h - l) / l if l > 0 else 0)
        else:
            features.append(0)

        # Close position within 20-bar range
        if bar_idx >= 20:
            h = np.max(p[-20:])
            l = np.min(p[-20:])
            features.append((p[-1] - l) / (h - l) if h > l else 0.5)
        else:
            features.append(0.5)

        # Up/down bar ratio (last 10 bars)
        if bar_idx >= 10:
            recent = p[-10:]
            ups = np.sum(np.diff(recent) > 0)
            features.append(ups / 9 - 0.5)
        else:
            features.append(0)

        return np.array(features, dtype=float)

    def _train_model(self, all_prices: Dict[str, np.ndarray],
                     all_volumes: Dict[str, np.ndarray], bar_idx: int):
        """Train LightGBM on rolling window of all symbols."""
        try:
            import lightgbm as lgb
        except ImportError:
            return

        X_list = []
        y_list = []

        start_bar = max(0, bar_idx - self.train_window)

        for sym in all_prices:
            prices = all_prices[sym]
            volumes = all_volumes[sym]

            for i in range(max(60, start_bar), bar_idx - 1):
                feat = self._compute_features(prices, volumes, i)
                if feat is None:
                    continue

                # Label: next-bar return > 0
                if i + 1 < len(prices):
                    next_ret = (prices[i + 1] - prices[i]) / prices[i]
                    label = 1 if next_ret > 0 else 0
                    X_list.append(feat)
                    y_list.append(label)

        if len(X_list) < self.min_train_samples:
            return

        X = np.array(X_list)
        y = np.array(y_list)

        # Replace NaN/inf
        X = np.nan_to_num(X, nan=0, posinf=1, neginf=-1)

        # Standardize
        self.scaler_mean = np.mean(X, axis=0)
        self.scaler_std = np.std(X, axis=0) + 1e-8
        X = (X - self.scaler_mean) / self.scaler_std

        # Train LightGBM
        self.model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )

        # Time-based split: last 20% for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
        )

        self.last_train_bar = bar_idx

    def signal(self, symbol: str, prices: np.ndarray, bar_idx: int,
               volumes: np.ndarray = None,
               all_prices: Dict = None, all_volumes: Dict = None) -> Tuple[str, float]:
        """
        Generate LightGBM signal.
        Returns: (action, confidence)
        """
        if bar_idx < 60:
            return ('HOLD', 0.0)

        # Retrain if needed
        if bar_idx - self.last_train_bar > self.retrain_every:
            if all_prices is not None and all_volumes is not None:
                self._train_model(all_prices, all_volumes, bar_idx)

        if self.model is None:
            return ('HOLD', 0.0)

        # Compute features
        feat = self._compute_features(prices, volumes, bar_idx)
        if feat is None:
            return ('HOLD', 0.0)

        # Standardize
        feat = np.nan_to_num(feat, nan=0, posinf=1, neginf=-1)
        if self.scaler_mean is not None:
            feat = (feat - self.scaler_mean) / self.scaler_std

        # Predict
        try:
            proba = self.model.predict_proba(feat.reshape(1, -1))[0]
            up_prob = proba[1]
        except Exception:
            return ('HOLD', 0.0)

        # Stop-loss check
        if symbol in self.entry_prices:
            entry = self.entry_prices[symbol]
            current = prices[bar_idx]
            if (current - entry) / entry < -self.stop_loss_pct:
                return ('SELL', 1.0)

        # Signal logic
        if up_prob > self.confidence_threshold:
            if symbol not in self.entry_prices:
                return ('BUY', up_prob)
        elif up_prob < (1 - self.confidence_threshold):
            if symbol in self.entry_prices:
                return ('SELL', 1 - up_prob)

        return ('HOLD', 0.0)

    def on_fill(self, symbol: str, action: str, price: float, bar_idx: int):
        """Called when order is filled."""
        if action == 'BUY':
            self.entry_prices[symbol] = price
            self.entry_bars[symbol] = bar_idx
        elif action == 'SELL':
            self.entry_prices.pop(symbol, None)
            self.entry_bars.pop(symbol, None)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Return feature importance if model is trained."""
        if self.model is None:
            return None
        importance = self.model.feature_importances_
        names = [f'f{i}' for i in range(len(importance))]
        return dict(sorted(zip(names, importance), key=lambda x: -x[1])[:15])
