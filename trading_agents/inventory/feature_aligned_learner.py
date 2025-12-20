"""Feature-Aligned Learning System.

KEY INSIGHT: Update frequency should match FEATURE TIMESCALE, not model complexity!

Features have natural timescales:
- FAST (changes every bar): momentum, volatility spikes, price action
- MEDIUM (changes daily): trend strength, support/resistance, daily vol
- SLOW (changes weekly+): regime, correlations, seasonal patterns

Each feature group can use ANY model complexity - the update frequency
is driven by HOW FAST THE FEATURE CHANGES, not how complex the model is.

This is how professional quant funds actually work:
- High-frequency signals → update models in real-time (even complex ones!)
- Low-frequency signals → update models less often (even simple ones!)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np
from collections import deque
from abc import ABC, abstractmethod

# Online models for fast updates
from .online_models import (
    OnlineLinearRegression,
    OnlineRidge,
    OnlineVolatility,
    OnlineRegimeDetector,
)


class FeatureTimescale(Enum):
    """Timescale categories for features."""
    FAST = "fast"      # Changes every bar (4H): momentum, vol spikes
    MEDIUM = "medium"  # Changes daily (~6 bars): trend, daily patterns
    SLOW = "slow"      # Changes weekly+ (~42 bars): regime, correlations


@dataclass
class FeatureSpec:
    """Specification for a feature."""
    name: str
    timescale: FeatureTimescale
    index: int  # Index in feature vector
    description: str = ""
    importance: float = 1.0  # Relative importance weight


# ============================================================================
# FEATURE DEFINITIONS FOR CRYPTO TRADING
# ============================================================================

CRYPTO_FEATURE_SPECS = [
    # FAST FEATURES (update every bar) - indices 0-4
    FeatureSpec("ret_1bar", FeatureTimescale.FAST, 0, "1-bar return", 1.0),
    FeatureSpec("ret_5bar", FeatureTimescale.FAST, 1, "5-bar return", 0.8),
    FeatureSpec("vol_intrabar", FeatureTimescale.FAST, 2, "Intrabar volatility", 0.9),
    FeatureSpec("volume_spike", FeatureTimescale.FAST, 3, "Volume vs avg ratio", 0.7),
    FeatureSpec("price_momentum", FeatureTimescale.FAST, 4, "Short-term momentum", 0.9),

    # MEDIUM FEATURES (update daily) - indices 5-7
    FeatureSpec("trend_strength", FeatureTimescale.MEDIUM, 5, "ADX/trend indicator", 0.8),
    FeatureSpec("daily_vol", FeatureTimescale.MEDIUM, 6, "Daily volatility", 0.7),
    FeatureSpec("sma_ratio", FeatureTimescale.MEDIUM, 7, "SMA5/SMA20 ratio", 0.6),

    # SLOW FEATURES (update weekly) - indices 8-9
    FeatureSpec("regime", FeatureTimescale.SLOW, 8, "Market regime (0/1/2)", 0.9),
    FeatureSpec("cross_corr", FeatureTimescale.SLOW, 9, "Cross-asset correlation", 0.5),
]


def get_feature_indices(timescale: FeatureTimescale) -> List[int]:
    """Get feature indices for a given timescale."""
    return [f.index for f in CRYPTO_FEATURE_SPECS if f.timescale == timescale]


def get_feature_names(timescale: FeatureTimescale) -> List[str]:
    """Get feature names for a given timescale."""
    return [f.name for f in CRYPTO_FEATURE_SPECS if f.timescale == timescale]


# ============================================================================
# FEATURE GROUP MODELS
# ============================================================================

class FeatureGroupModel(ABC):
    """Base class for models that handle a specific feature timescale group."""

    @abstractmethod
    def predict(self, features: np.ndarray) -> float:
        """Make prediction using features from this group."""
        pass

    @abstractmethod
    def update(self, features: np.ndarray, target: float):
        """Update model with new observation."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get serializable state."""
        pass


class FastFeatureModel(FeatureGroupModel):
    """
    Model for FAST features (momentum, vol spikes).

    Updates EVERY bar because these features change rapidly.
    Can use complex models since we're updating based on feature timescale,
    not model complexity!

    Features: ret_1bar, ret_5bar, vol_intrabar, volume_spike, price_momentum
    """

    def __init__(self, n_features: int = 5):
        self.n_features = n_features
        self.feature_indices = get_feature_indices(FeatureTimescale.FAST)

        # Use ensemble of online models for robustness
        self.linear = OnlineLinearRegression(n_features, learning_rate=0.002)
        self.ridge = OnlineRidge(n_features, forgetting_factor=0.98)

        # Momentum-specific: track recent momentum for signal
        self.momentum_ema = 0.0
        self.momentum_alpha = 0.3

        # Volatility spike detector
        self.vol_history = deque(maxlen=20)
        self.vol_threshold = 2.0  # Std devs for "spike"

        self.n_updates = 0

    def predict(self, features: np.ndarray) -> float:
        """Predict using fast features."""
        # Extract fast features
        fast_feats = self._extract_features(features)

        # Blend online models
        linear_pred = self.linear.predict(fast_feats)
        ridge_pred = self.ridge.predict(fast_feats)

        # Weight towards more stable ridge as updates increase
        alpha = min(0.7, 0.3 + self.n_updates / 500)
        model_pred = (1 - alpha) * linear_pred + alpha * ridge_pred

        # Incorporate momentum signal directly
        momentum = fast_feats[0] if len(fast_feats) > 0 else 0  # ret_1bar
        momentum_signal = 0.3 * momentum + 0.7 * self.momentum_ema

        # Blend model prediction with momentum signal
        # More weight to raw momentum (it's ground truth!)
        return 0.4 * model_pred + 0.6 * momentum_signal

    def update(self, features: np.ndarray, target: float):
        """Update fast feature model - called EVERY bar."""
        fast_feats = self._extract_features(features)

        # Update online models
        linear_pred = self.linear.predict(fast_feats)
        self.linear.update(fast_feats, target, linear_pred)

        ridge_pred = self.ridge.predict(fast_feats)
        self.ridge.update(fast_feats, target, ridge_pred)

        # Update momentum EMA
        if len(fast_feats) > 0:
            self.momentum_ema = (
                self.momentum_alpha * fast_feats[0] +
                (1 - self.momentum_alpha) * self.momentum_ema
            )

        # Track volatility for spike detection
        if len(fast_feats) > 2:
            self.vol_history.append(fast_feats[2])

        self.n_updates += 1

    def _extract_features(self, full_features: np.ndarray) -> np.ndarray:
        """Extract fast features from full feature vector."""
        if len(full_features) >= max(self.feature_indices) + 1:
            return np.array([full_features[i] for i in self.feature_indices])
        return full_features[:self.n_features]

    def is_vol_spike(self) -> bool:
        """Detect if current volatility is a spike."""
        if len(self.vol_history) < 10:
            return False
        mean_vol = np.mean(list(self.vol_history)[:-1])
        std_vol = np.std(list(self.vol_history)[:-1])
        current = self.vol_history[-1]
        return current > mean_vol + self.vol_threshold * std_vol

    def get_state(self) -> Dict[str, Any]:
        return {
            "n_updates": self.n_updates,
            "momentum_ema": self.momentum_ema,
            "linear": self.linear.get_state(),
            "ridge": self.ridge.get_state(),
        }


class MediumFeatureModel(FeatureGroupModel):
    """
    Model for MEDIUM features (trend, daily patterns).

    Updates every ~6 bars (daily at 4H timeframe) because these
    features don't change bar-to-bar.

    Features: trend_strength, daily_vol, sma_ratio
    """

    def __init__(self, n_features: int = 3):
        self.n_features = n_features
        self.feature_indices = get_feature_indices(FeatureTimescale.MEDIUM)

        # Buffer for batch updates
        self.feature_buffer = deque(maxlen=200)
        self.target_buffer = deque(maxlen=200)

        # Model: can be more complex since updated less often
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Trend state
        self.trend_direction = 0  # -1, 0, 1
        self.trend_strength = 0.0

        # ARIMA-like: track recent predictions for momentum
        self.prediction_history = deque(maxlen=10)

        self.n_updates = 0
        self.last_refit = 0
        self.refit_frequency = 6  # Every 6 bars

    def predict(self, features: np.ndarray) -> float:
        """Predict using medium features."""
        medium_feats = self._extract_features(features)

        # Linear prediction
        base_pred = np.dot(self.weights, medium_feats) + self.bias

        # Add trend persistence (trends tend to continue)
        trend_contrib = 0.2 * self.trend_direction * self.trend_strength

        return base_pred + trend_contrib

    def update(self, features: np.ndarray, target: float):
        """Buffer data and refit periodically."""
        medium_feats = self._extract_features(features)
        self.feature_buffer.append(medium_feats)
        self.target_buffer.append(target)

        self.n_updates += 1

        # Update trend state from features
        if len(medium_feats) > 0:
            self.trend_strength = abs(medium_feats[0]) if len(medium_feats) > 0 else 0
            sma_ratio = medium_feats[2] if len(medium_feats) > 2 else 0
            self.trend_direction = 1 if sma_ratio > 0.01 else (-1 if sma_ratio < -0.01 else 0)

        # Refit model periodically
        if (self.n_updates - self.last_refit) >= self.refit_frequency:
            self._refit()
            self.last_refit = self.n_updates

    def _refit(self):
        """Refit model on buffered data."""
        if len(self.feature_buffer) < 20:
            return

        X = np.array(list(self.feature_buffer))
        y = np.array(list(self.target_buffer))

        # Simple ridge regression
        try:
            lambda_reg = 1.0
            XtX = X.T @ X + lambda_reg * np.eye(X.shape[1])
            Xty = X.T @ y
            self.weights = np.linalg.solve(XtX, Xty)
            self.bias = np.mean(y) - np.dot(self.weights, np.mean(X, axis=0))
        except Exception:
            pass

    def _extract_features(self, full_features: np.ndarray) -> np.ndarray:
        """Extract medium features."""
        if len(full_features) >= max(self.feature_indices) + 1:
            return np.array([full_features[i] for i in self.feature_indices])
        return np.zeros(self.n_features)

    def get_state(self) -> Dict[str, Any]:
        return {
            "n_updates": self.n_updates,
            "weights": self.weights.tolist(),
            "trend_direction": self.trend_direction,
            "trend_strength": self.trend_strength,
        }


class SlowFeatureModel(FeatureGroupModel):
    """
    Model for SLOW features (regime, correlations).

    Updates weekly (~42 bars at 4H) because these structural
    features change slowly.

    Features: regime, cross_corr
    """

    def __init__(self, n_features: int = 2):
        self.n_features = n_features
        self.feature_indices = get_feature_indices(FeatureTimescale.SLOW)

        # Buffer for batch training
        self.feature_buffer = deque(maxlen=500)
        self.target_buffer = deque(maxlen=500)

        # Regime-specific models (one per regime)
        self.regime_means = {0: 0.0, 1: 0.0, 2: 0.0}  # Bear, Neutral, Bull
        self.regime_counts = {0: 1, 1: 1, 2: 1}

        # Correlation impact model
        self.corr_coefficient = 0.0

        # ML model (fitted less frequently)
        self.ml_model = None

        self.n_updates = 0
        self.last_refit = 0
        self.refit_frequency = 42  # Weekly

    def predict(self, features: np.ndarray) -> float:
        """Predict using slow features."""
        slow_feats = self._extract_features(features)

        # Get current regime
        regime = int(slow_feats[0]) if len(slow_feats) > 0 else 1
        regime = max(0, min(2, regime))

        # Regime-based prediction
        regime_pred = self.regime_means.get(regime, 0.0)

        # Correlation adjustment
        corr = slow_feats[1] if len(slow_feats) > 1 else 0
        corr_adj = self.corr_coefficient * corr

        # ML model if available
        if self.ml_model is not None:
            try:
                ml_pred = self.ml_model.predict(slow_feats.reshape(1, -1))[0]
                return 0.5 * regime_pred + 0.3 * ml_pred + 0.2 * corr_adj
            except Exception:
                pass

        return regime_pred + corr_adj

    def update(self, features: np.ndarray, target: float):
        """Buffer data and refit weekly."""
        slow_feats = self._extract_features(features)
        self.feature_buffer.append(slow_feats)
        self.target_buffer.append(target)

        self.n_updates += 1

        # Update regime means incrementally
        regime = int(slow_feats[0]) if len(slow_feats) > 0 else 1
        regime = max(0, min(2, regime))

        # Exponential moving average for regime mean
        alpha = 0.1
        self.regime_means[regime] = (
            alpha * target + (1 - alpha) * self.regime_means[regime]
        )
        self.regime_counts[regime] += 1

        # Refit ML model weekly
        if (self.n_updates - self.last_refit) >= self.refit_frequency:
            self._refit()
            self.last_refit = self.n_updates

    def _refit(self):
        """Refit ML model on buffered data."""
        if len(self.feature_buffer) < 100:
            return

        X = np.array(list(self.feature_buffer))
        y = np.array(list(self.target_buffer))

        try:
            from sklearn.ensemble import RandomForestRegressor

            self.ml_model = RandomForestRegressor(
                n_estimators=30,
                max_depth=4,
                min_samples_leaf=10,
                random_state=42,
            )
            self.ml_model.fit(X, y)

            # Update correlation coefficient
            if X.shape[1] > 1:
                corr_with_target = np.corrcoef(X[:, 1], y)[0, 1]
                if not np.isnan(corr_with_target):
                    self.corr_coefficient = corr_with_target * 0.01

        except ImportError:
            # sklearn not available
            pass
        except Exception:
            pass

    def _extract_features(self, full_features: np.ndarray) -> np.ndarray:
        """Extract slow features."""
        if len(full_features) >= max(self.feature_indices) + 1:
            return np.array([full_features[i] for i in self.feature_indices])
        return np.zeros(self.n_features)

    def get_state(self) -> Dict[str, Any]:
        return {
            "n_updates": self.n_updates,
            "regime_means": self.regime_means,
            "corr_coefficient": self.corr_coefficient,
            "ml_fitted": self.ml_model is not None,
        }


# ============================================================================
# MAIN FEATURE-ALIGNED LEARNER
# ============================================================================

@dataclass
class FeatureAlignedConfig:
    """Configuration for feature-aligned learning."""
    # Feature group settings
    fast_update_frequency: int = 1    # Every bar
    medium_update_frequency: int = 6  # Every 6 bars (~daily)
    slow_update_frequency: int = 42   # Every 42 bars (~weekly)

    # Blending weights (learned adaptively)
    initial_fast_weight: float = 0.5
    initial_medium_weight: float = 0.3
    initial_slow_weight: float = 0.2

    # Volatility gating
    reduce_slow_weight_in_high_vol: bool = True
    high_vol_threshold: float = 0.04


class FeatureAlignedLearner:
    """
    Feature-Aligned Learning System.

    Core principle: UPDATE FREQUENCY MATCHES FEATURE TIMESCALE

    - Fast features (momentum, vol) → update every bar
    - Medium features (trend, daily) → update daily
    - Slow features (regime, corr) → update weekly

    Each group can use ANY model complexity - we're not limiting
    complex models to slow updates!
    """

    def __init__(
        self,
        n_features: int = 10,
        config: Optional[FeatureAlignedConfig] = None,
    ):
        self.n_features = n_features
        self.config = config or FeatureAlignedConfig()

        # Feature group models
        self.fast_model = FastFeatureModel(n_features=5)
        self.medium_model = MediumFeatureModel(n_features=3)
        self.slow_model = SlowFeatureModel(n_features=2)

        # Adaptive blending weights
        self.weights = {
            "fast": self.config.initial_fast_weight,
            "medium": self.config.initial_medium_weight,
            "slow": self.config.initial_slow_weight,
        }

        # Performance tracking for weight adaptation
        self.group_errors = {"fast": deque(maxlen=50), "medium": deque(maxlen=50), "slow": deque(maxlen=50)}
        self.last_predictions = {"fast": 0.0, "medium": 0.0, "slow": 0.0}

        # Volatility for gating
        self.volatility = OnlineVolatility(alpha=0.06)

        # Regime detection
        self.regime = OnlineRegimeDetector(n_regimes=3)

        self.n_updates = 0

    def predict(self, features: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """
        Get trading signal using feature-aligned predictions.

        Returns:
            signal: "long", "short", or "hold"
            confidence: 0-1 confidence score
            details: Breakdown by feature group
        """
        details = {}

        # Update volatility and regime (these inform predictions)
        vol = self.volatility.predict(features)
        regime = int(self.regime.predict(features))

        # Get predictions from each feature group
        fast_pred = self.fast_model.predict(features)
        medium_pred = self.medium_model.predict(features)
        slow_pred = self.slow_model.predict(features)

        self.last_predictions = {
            "fast": fast_pred,
            "medium": medium_pred,
            "slow": slow_pred,
        }

        details["fast_pred"] = fast_pred
        details["medium_pred"] = medium_pred
        details["slow_pred"] = slow_pred
        details["volatility"] = vol
        details["regime"] = ["Bear", "Neutral", "Bull"][regime]

        # Adjust weights based on market conditions
        weights = self._get_adaptive_weights(vol, regime)
        details["weights"] = weights

        # Blend predictions
        blended = (
            weights["fast"] * fast_pred +
            weights["medium"] * medium_pred +
            weights["slow"] * slow_pred
        )
        details["blended"] = blended

        # Convert to signal
        signal, confidence = self._to_signal(blended, vol, regime)

        return signal, confidence, details

    def update(self, features: np.ndarray, actual_return: float):
        """
        Update models based on feature timescales.

        - Fast model: updated EVERY call
        - Medium model: updated every N calls (buffers data between)
        - Slow model: updated every M calls (buffers data between)
        """
        self.n_updates += 1

        # Track errors for weight adaptation
        for group, pred in self.last_predictions.items():
            error = abs(pred - actual_return)
            self.group_errors[group].append(error)

        # Update volatility and regime (always)
        vol_pred = self.volatility.predict(features)
        self.volatility.update(features, actual_return, vol_pred)

        regime_pred = self.regime.predict(features)
        self.regime.update(features, actual_return, regime_pred)

        # FAST: Update every bar (frequency = 1)
        self.fast_model.update(features, actual_return)

        # MEDIUM: Update every 6 bars (but always buffer)
        self.medium_model.update(features, actual_return)

        # SLOW: Update every 42 bars (but always buffer)
        self.slow_model.update(features, actual_return)

        # Adapt weights periodically
        if self.n_updates % 20 == 0:
            self._adapt_weights()

    def _get_adaptive_weights(
        self,
        volatility: float,
        regime: int,
    ) -> Dict[str, float]:
        """Get adaptive weights based on market conditions."""
        weights = dict(self.weights)

        # In high volatility: trust fast features more, slow features less
        if self.config.reduce_slow_weight_in_high_vol and volatility > self.config.high_vol_threshold:
            vol_factor = min(2.0, volatility / self.config.high_vol_threshold)
            weights["fast"] *= vol_factor
            weights["slow"] /= vol_factor

        # In neutral regime: reduce confidence in directional models
        if regime == 1:  # Neutral
            weights["medium"] *= 0.8

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _adapt_weights(self):
        """Adapt weights based on recent prediction errors."""
        maes = {}
        for group, errors in self.group_errors.items():
            if len(errors) > 10:
                maes[group] = np.mean(list(errors))

        if len(maes) < 3:
            return

        # Inverse MAE weighting
        inv_maes = {k: 1.0 / max(0.0001, v) for k, v in maes.items()}
        total = sum(inv_maes.values())

        # Slow adaptation
        alpha = 0.1
        for group in self.weights:
            if group in inv_maes:
                target = inv_maes[group] / total
                self.weights[group] = (1 - alpha) * self.weights[group] + alpha * target

    def _to_signal(
        self,
        prediction: float,
        volatility: float,
        regime: int,
    ) -> Tuple[str, float]:
        """Convert prediction to trading signal."""
        # Adaptive threshold based on volatility
        base_threshold = 0.003
        threshold = base_threshold * (1 + volatility * 10)

        # Vol spike detection from fast model
        vol_spike = self.fast_model.is_vol_spike()
        if vol_spike:
            threshold *= 1.5  # Need stronger signal during vol spikes

        if prediction > threshold:
            signal = "long"
            confidence = min(0.95, 0.5 + abs(prediction) / (threshold * 3))
        elif prediction < -threshold:
            signal = "short"
            confidence = min(0.95, 0.5 + abs(prediction) / (threshold * 3))
        else:
            signal = "hold"
            confidence = 0.3

        # Reduce confidence in neutral regime
        if regime == 1:
            confidence *= 0.85

        return signal, max(0.1, min(0.95, confidence))

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state."""
        return {
            "n_updates": self.n_updates,
            "weights": self.weights,
            "fast_model": self.fast_model.get_state(),
            "medium_model": self.medium_model.get_state(),
            "slow_model": self.slow_model.get_state(),
        }

    def get_summary(self) -> str:
        """Human-readable summary."""
        return f"""
FEATURE-ALIGNED LEARNER STATUS
==============================
Total Updates: {self.n_updates}

FAST Features (updated every bar):
  - Updates: {self.fast_model.n_updates}
  - Momentum EMA: {self.fast_model.momentum_ema:.4f}
  - Weight: {self.weights['fast']:.1%}

MEDIUM Features (updated every 6 bars):
  - Updates: {self.medium_model.n_updates}
  - Trend: {self.medium_model.trend_direction} (strength: {self.medium_model.trend_strength:.2f})
  - Weight: {self.weights['medium']:.1%}

SLOW Features (updated every 42 bars):
  - Updates: {self.slow_model.n_updates}
  - Regime means: {self.slow_model.regime_means}
  - ML fitted: {self.slow_model.ml_model is not None}
  - Weight: {self.weights['slow']:.1%}

Current Regime: {self.regime.get_regime_name()}
"""


def create_feature_aligned_learner(**kwargs) -> FeatureAlignedLearner:
    """Factory function."""
    config = FeatureAlignedConfig(**kwargs)
    return FeatureAlignedLearner(config=config)
