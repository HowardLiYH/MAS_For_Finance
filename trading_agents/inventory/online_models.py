"""Online Learning Models - Models that update with each new observation.

This module implements TRUE online learning where models update their
parameters after each new data point, similar to how hedge funds operate.

Key Concepts:
- Online Learning: Update model after EVERY observation
- Incremental Learning: Model improves without full retraining
- No Look-Ahead Bias: Only uses data available at prediction time
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from collections import deque


@dataclass
class OnlineModelState:
    """State that persists across predictions."""
    weights: Optional[np.ndarray] = None
    bias: float = 0.0
    learning_rate: float = 0.01
    n_updates: int = 0
    recent_errors: List[float] = field(default_factory=list)


class OnlineModel(ABC):
    """Base class for online learning models."""

    @abstractmethod
    def predict(self, features: np.ndarray) -> float:
        """Make prediction using current model state."""
        pass

    @abstractmethod
    def update(self, features: np.ndarray, target: float, prediction: float):
        """Update model with new observation."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get serializable state for persistence."""
        pass

    @abstractmethod
    def load_state(self, state: Dict[str, Any]):
        """Load state from persistence."""
        pass


class OnlineLinearRegression(OnlineModel):
    """Online Linear Regression using Stochastic Gradient Descent.

    Updates weights after each observation:
        w = w - lr * gradient
        gradient = (prediction - target) * features

    This is what many hedge funds use for real-time signal generation.
    """

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.001,
        l2_regularization: float = 0.0001,
    ):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.l2_reg = l2_regularization

        # Initialize weights with small random values
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

        # Tracking
        self.n_updates = 0
        self.recent_errors = deque(maxlen=100)

    def predict(self, features: np.ndarray) -> float:
        """Linear prediction: w·x + b"""
        return float(np.dot(self.weights, features) + self.bias)

    def update(self, features: np.ndarray, target: float, prediction: float):
        """SGD update after observing true target."""
        error = prediction - target
        self.recent_errors.append(abs(error))

        # Gradient descent step
        gradient = error * features + self.l2_reg * self.weights
        self.weights -= self.learning_rate * gradient
        self.bias -= self.learning_rate * error

        self.n_updates += 1

    def get_state(self) -> Dict[str, Any]:
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "n_updates": self.n_updates,
            "learning_rate": self.learning_rate,
            "recent_mae": np.mean(self.recent_errors) if self.recent_errors else 0,
        }

    def load_state(self, state: Dict[str, Any]):
        self.weights = np.array(state["weights"])
        self.bias = state["bias"]
        self.n_updates = state["n_updates"]


class OnlineRidge(OnlineModel):
    """Online Ridge Regression with recursive least squares.

    Uses Sherman-Morrison formula for efficient online updates.
    More stable than pure SGD for financial data.
    """

    def __init__(
        self,
        n_features: int,
        regularization: float = 1.0,
        forgetting_factor: float = 0.99,  # Emphasize recent data
    ):
        self.n_features = n_features
        self.reg = regularization
        self.forgetting = forgetting_factor

        # Initialize
        self.weights = np.zeros(n_features)
        self.P = np.eye(n_features) / regularization  # Inverse covariance

        self.n_updates = 0
        self.recent_errors = deque(maxlen=100)

    def predict(self, features: np.ndarray) -> float:
        return float(np.dot(self.weights, features))

    def update(self, features: np.ndarray, target: float, prediction: float):
        """Recursive least squares update."""
        error = target - prediction
        self.recent_errors.append(abs(error))

        # Sherman-Morrison update
        Px = self.P @ features
        denominator = self.forgetting + features @ Px
        K = Px / denominator  # Kalman gain

        self.weights += K * error
        self.P = (self.P - np.outer(K, Px)) / self.forgetting

        self.n_updates += 1

    def get_state(self) -> Dict[str, Any]:
        return {
            "weights": self.weights.tolist(),
            "P": self.P.tolist(),
            "n_updates": self.n_updates,
        }

    def load_state(self, state: Dict[str, Any]):
        self.weights = np.array(state["weights"])
        self.P = np.array(state["P"])
        self.n_updates = state["n_updates"]


class OnlineEMA(OnlineModel):
    """Exponential Moving Average predictor.

    Simple but effective: predicts next value as EMA of past values.
    Updates instantly with each new observation.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.ema = None
        self.n_updates = 0

    def predict(self, features: np.ndarray) -> float:
        """Predict as current EMA."""
        if self.ema is None:
            return float(features[-1]) if len(features) > 0 else 0.0
        return self.ema

    def update(self, features: np.ndarray, target: float, prediction: float):
        """Update EMA with new observation."""
        if self.ema is None:
            self.ema = target
        else:
            self.ema = self.alpha * target + (1 - self.alpha) * self.ema
        self.n_updates += 1

    def get_state(self) -> Dict[str, Any]:
        return {"ema": self.ema, "n_updates": self.n_updates}

    def load_state(self, state: Dict[str, Any]):
        self.ema = state["ema"]
        self.n_updates = state["n_updates"]


class OnlineVolatility(OnlineModel):
    """Online volatility estimation using exponential weighted variance.

    Critical for risk management - updates vol estimate with each return.
    """

    def __init__(self, alpha: float = 0.06):  # ~30 period effective window
        self.alpha = alpha
        self.mean = 0.0
        self.variance = 0.0001  # Initial variance
        self.n_updates = 0

    def predict(self, features: np.ndarray) -> float:
        """Return current volatility estimate."""
        return np.sqrt(self.variance)

    def update(self, features: np.ndarray, target: float, prediction: float):
        """Update volatility with new return observation."""
        # Welford's online algorithm with exponential weighting
        delta = target - self.mean
        self.mean += self.alpha * delta
        self.variance = (1 - self.alpha) * (self.variance + self.alpha * delta * delta)
        self.n_updates += 1

    def get_state(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "variance": self.variance,
            "n_updates": self.n_updates,
        }

    def load_state(self, state: Dict[str, Any]):
        self.mean = state["mean"]
        self.variance = state["variance"]
        self.n_updates = state["n_updates"]


class OnlineRegimeDetector(OnlineModel):
    """Online Hidden Markov Model for regime detection.

    Incrementally updates regime probabilities as new data arrives.
    v0.9.6: Fixed to be more responsive to regime changes.
    """

    def __init__(self, n_regimes: int = 3, learning_rate: float = 0.1):
        self.n_regimes = n_regimes
        self.learning_rate = learning_rate

        # Regime statistics (mean, variance of returns per regime)
        # Adjusted for typical crypto returns (4h bars)
        self.regime_means = np.array([-0.015, 0.0, 0.015])  # Bear, Neutral, Bull
        self.regime_vars = np.array([0.001, 0.0005, 0.001])  # Similar variance

        # Current regime probabilities
        self.regime_probs = np.ones(n_regimes) / n_regimes

        # Transition matrix - LESS sticky for faster regime detection
        self.transition = np.array([
            [0.7, 0.2, 0.1],   # Bear can transition
            [0.15, 0.7, 0.15], # Neutral balanced
            [0.1, 0.2, 0.7],   # Bull can transition
        ])

        self.n_updates = 0

    def predict(self, features: np.ndarray) -> float:
        """Return most likely regime (0=Bear, 1=Neutral, 2=Bull)."""
        return float(np.argmax(self.regime_probs))

    def update(self, features: np.ndarray, target: float, prediction: float):
        """Update regime probabilities with new return observation."""
        # Likelihood of observation under each regime
        # Add small epsilon to prevent numerical issues
        eps = 1e-10
        likelihoods = np.exp(-0.5 * ((target - self.regime_means) ** 2) / (self.regime_vars + eps))
        likelihoods /= np.sqrt(2 * np.pi * (self.regime_vars + eps))
        likelihoods = np.clip(likelihoods, eps, 1e10)

        # Bayesian update with stronger signal
        new_probs = self.regime_probs * likelihoods
        if new_probs.sum() > 0:
            new_probs /= new_probs.sum()
        else:
            new_probs = np.ones(self.n_regimes) / self.n_regimes

        # Apply transition matrix (but with reduced influence)
        # Mix: 70% observation-based, 30% transition-based
        transitioned = self.transition.T @ new_probs
        self.regime_probs = 0.7 * new_probs + 0.3 * transitioned

        # Update regime statistics (adaptive)
        most_likely = np.argmax(self.regime_probs)
        self.regime_means[most_likely] += self.learning_rate * (target - self.regime_means[most_likely])

        # Update variance with clipping to prevent collapse
        new_var = self.regime_vars[most_likely] + self.learning_rate * (
            (target - self.regime_means[most_likely])**2 - self.regime_vars[most_likely]
        )
        self.regime_vars[most_likely] = np.clip(new_var, 0.0001, 0.01)

        self.n_updates += 1

    def get_regime_name(self) -> str:
        regime = int(np.argmax(self.regime_probs))
        return ["Bear", "Neutral", "Bull"][regime]

    def get_state(self) -> Dict[str, Any]:
        return {
            "regime_probs": self.regime_probs.tolist(),
            "regime_means": self.regime_means.tolist(),
            "regime_vars": self.regime_vars.tolist(),
            "n_updates": self.n_updates,
        }

    def load_state(self, state: Dict[str, Any]):
        self.regime_probs = np.array(state["regime_probs"])
        self.regime_means = np.array(state["regime_means"])
        self.regime_vars = np.array(state["regime_vars"])
        self.n_updates = state["n_updates"]


class OnlineModelManager:
    """Manages a collection of online models for a trading agent.

    This is what makes our system behave like a hedge fund:
    - Each model updates after every observation
    - State persists across sessions
    - Models adapt to changing market conditions
    """

    def __init__(self, n_features: int = 10):
        self.n_features = n_features

        self.models = {
            "return_predictor": OnlineLinearRegression(n_features, learning_rate=0.001),
            "volatility": OnlineVolatility(alpha=0.06),
            "regime": OnlineRegimeDetector(n_regimes=3),
            "trend_ridge": OnlineRidge(n_features, forgetting_factor=0.99),
        }

        self.last_predictions: Dict[str, float] = {}

    def predict_all(self, features: np.ndarray) -> Dict[str, float]:
        """Get predictions from all models."""
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(features)
        self.last_predictions = predictions
        return predictions

    def update_all(self, features: np.ndarray, actual_return: float):
        """Update all models with observed outcome."""
        for name, model in self.models.items():
            pred = self.last_predictions.get(name, 0.0)
            model.update(features, actual_return, pred)

    def get_combined_signal(self, features: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """Get trading signal combining FEATURES and model predictions.

        v0.9.5: FIXED - Momentum is now the PRIMARY driver
        - Untrained model predictions should NOT override clear momentum
        - Model influence grows as it learns (based on n_updates)
        - Features are the ground truth; model just adds refinement

        Features expected:
        - features[0]: 1-bar return
        - features[1]: 5-bar return
        - features[2]: 10-bar return
        - features[3]: 5-bar volatility
        - features[6]: SMA ratio (trend)
        - features[7]: trend indicator (-1, 0, 1)

        Returns:
            signal: "long", "short", or "hold"
            confidence: 0-1 confidence score
            details: Breakdown from each model
        """
        preds = self.predict_all(features)

        # Get model predictions
        vol = max(0.001, preds["volatility"])
        regime = int(preds["regime"])

        # === MOMENTUM IS THE PRIMARY SIGNAL ===
        # These are ACTUAL returns - ground truth
        ret_1 = features[0] if len(features) > 0 else 0  # 1-bar return
        ret_5 = features[1] if len(features) > 1 else 0  # 5-bar return
        ret_10 = features[2] if len(features) > 2 else 0  # 10-bar return

        # NaN handling - replace NaN with 0 to prevent propagation
        if np.isnan(ret_1): ret_1 = 0.0
        if np.isnan(ret_5): ret_5 = 0.0
        if np.isnan(ret_10): ret_10 = 0.0

        # Momentum signal: weighted average of returns
        # This is the PRIMARY driver of trading decisions
        momentum = 0.5 * ret_1 + 0.3 * ret_5 + 0.2 * ret_10

        # Model influence: ONLY trust the model after it has learned
        # Start with 0% model influence, grow to max 30% after 500 updates
        return_predictor = self.models["return_predictor"]
        n_updates = return_predictor.n_updates
        model_weight = min(0.3, n_updates / 1500)  # Max 30% after 500 updates

        # Clip model prediction to reasonable range to prevent wild swings
        model_pred = preds["return_predictor"]
        model_pred_clipped = np.clip(model_pred, -0.1, 0.1)  # Max ±10% prediction

        # Combined signal: momentum-dominated, with optional model refinement
        # Momentum weight: 1 - model_weight (starts at 100%, min 70%)
        momentum_weight = 1.0 - model_weight
        combined_signal = momentum_weight * momentum + model_weight * model_pred_clipped

        # Regime adjustment (mild)
        regime_multiplier = [0.9, 1.0, 1.1][regime]

        # Scale to percentage for threshold comparison
        # momentum of 0.01 = 1% return, should be meaningful
        adjusted_signal = combined_signal * regime_multiplier * 100

        # Threshold: 0.5% momentum triggers a trade
        signal_threshold = 0.5

        if adjusted_signal > signal_threshold:
            signal = "long"
            confidence = min(0.95, 0.5 + abs(adjusted_signal) / 5.0)
        elif adjusted_signal < -signal_threshold:
            signal = "short"
            confidence = min(0.95, 0.5 + abs(adjusted_signal) / 5.0)
        else:
            signal = "hold"
            confidence = 0.3  # Low confidence when holding

        # Adjust confidence based on volatility (high vol = less confident)
        # Use FEATURE volatility (features[3]) if available, else model vol
        feature_vol = features[3] if len(features) > 3 and not np.isnan(features[3]) else vol
        if feature_vol > 0.03:  # 3% volatility threshold
            confidence *= 0.7  # Reduce confidence by 30%
        elif feature_vol > 0.05:  # 5% volatility
            confidence *= 0.5  # Reduce confidence by 50%

        # Ensure confidence is in valid range
        confidence = max(0.1, min(0.95, confidence))

        details = {
            "return_prediction": model_pred,
            "model_pred_clipped": model_pred_clipped,
            "momentum": momentum,
            "model_weight": model_weight,
            "n_updates": n_updates,
            "volatility": vol,
            "regime": ["Bear", "Neutral", "Bull"][regime],
            "combined_signal": combined_signal,
            "adjusted_signal": adjusted_signal,
            "signal_threshold": signal_threshold,
        }

        return signal, confidence, details

    def get_state(self) -> Dict[str, Any]:
        return {name: model.get_state() for name, model in self.models.items()}

    def load_state(self, state: Dict[str, Any]):
        for name, model_state in state.items():
            if name in self.models:
                self.models[name].load_state(model_state)


# Example usage showing the online learning loop
def demo_online_learning():
    """Demonstrate online learning with simulated data."""
    import random

    manager = OnlineModelManager(n_features=5)

    print("=" * 60)
    print("ONLINE LEARNING DEMO")
    print("=" * 60)

    # Simulate 20 trading bars
    for i in range(20):
        # Generate random features (in reality: OHLCV + indicators)
        features = np.random.randn(5)

        # Get prediction BEFORE seeing outcome
        signal, confidence, details = manager.get_combined_signal(features)

        # Simulate actual return (in reality: from next bar)
        actual_return = np.random.randn() * 0.02 + 0.001

        # Update models with observed outcome
        manager.update_all(features, actual_return)

        print(f"Bar {i+1:2d}: Signal={signal:5s} (conf={confidence:.2f}), "
              f"Predicted={details['return_prediction']:+.4f}, "
              f"Actual={actual_return:+.4f}, "
              f"Regime={details['regime']}")

    print("\n" + "=" * 60)
    print("Model states after 20 updates:")
    state = manager.get_state()
    print(f"  Return predictor: {state['return_predictor']['n_updates']} updates")
    print(f"  Volatility: {state['volatility']['n_updates']} updates")
    print(f"  Regime: {state['regime']['n_updates']} updates")


if __name__ == "__main__":
    demo_online_learning()
