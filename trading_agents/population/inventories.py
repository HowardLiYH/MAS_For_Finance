"""Extended method inventories for each agent role.

Each role has 12-15 methods available, but agents only select 3-4 at a time.
This creates selection pressure and allows agents to learn optimal combinations.

The inventories are designed to have:
- Diverse approaches (technical, statistical, ML-based)
- Different computational costs
- Methods that may be complementary or redundant
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any
from enum import Enum


# =============================================================================
# ANALYST INVENTORY (15 methods)
# =============================================================================

ANALYST_INVENTORY = [
    # Technical Analysis (5)
    "RSI",                    # Relative Strength Index
    "MACD",                   # Moving Average Convergence Divergence
    "BollingerBands",         # Bollinger Bands
    "ADX",                    # Average Directional Index
    "Stochastic",             # Stochastic Oscillator

    # Statistical Analysis (4)
    "Autocorrelation",        # Lagged correlation analysis
    "VolatilityClustering",   # GARCH-style volatility
    "MeanReversion",          # Mean reversion signals
    "Cointegration",          # Cross-asset cointegration

    # Decomposition Methods (3)
    "STL_Decomposition",      # Seasonal-Trend-Loess
    "WaveletTransform",       # Multi-scale analysis
    "FourierAnalysis",        # Frequency domain

    # Machine Learning (3)
    "HMM_Regime",             # Hidden Markov Model regimes
    "KalmanFilter",           # State space filtering
    "IsolationForest",        # Anomaly detection
]

ANALYST_METHOD_INFO = {
    "RSI": {
        "category": "technical",
        "description": "Momentum oscillator measuring speed of price changes",
        "compute_cost": 0.1,
        "parameters": {"period": 14, "overbought": 70, "oversold": 30},
    },
    "MACD": {
        "category": "technical",
        "description": "Trend-following momentum indicator",
        "compute_cost": 0.1,
        "parameters": {"fast": 12, "slow": 26, "signal": 9},
    },
    "BollingerBands": {
        "category": "technical",
        "description": "Volatility bands around moving average",
        "compute_cost": 0.1,
        "parameters": {"period": 20, "std_dev": 2},
    },
    "ADX": {
        "category": "technical",
        "description": "Trend strength indicator",
        "compute_cost": 0.1,
        "parameters": {"period": 14},
    },
    "Stochastic": {
        "category": "technical",
        "description": "Momentum comparing close to high-low range",
        "compute_cost": 0.1,
        "parameters": {"k_period": 14, "d_period": 3},
    },
    "Autocorrelation": {
        "category": "statistical",
        "description": "Lagged price correlation for mean reversion",
        "compute_cost": 0.2,
        "parameters": {"max_lag": 20},
    },
    "VolatilityClustering": {
        "category": "statistical",
        "description": "GARCH-style volatility regime detection",
        "compute_cost": 0.5,
        "parameters": {"p": 1, "q": 1},
    },
    "MeanReversion": {
        "category": "statistical",
        "description": "Half-life and mean reversion speed",
        "compute_cost": 0.3,
        "parameters": {"lookback": 60},
    },
    "Cointegration": {
        "category": "statistical",
        "description": "Cross-asset cointegration for pairs trading",
        "compute_cost": 0.4,
        "parameters": {"max_lag": 10},
    },
    "STL_Decomposition": {
        "category": "decomposition",
        "description": "Seasonal, trend, residual decomposition",
        "compute_cost": 0.3,
        "parameters": {"period": 24},
    },
    "WaveletTransform": {
        "category": "decomposition",
        "description": "Multi-scale time-frequency analysis",
        "compute_cost": 0.6,
        "parameters": {"wavelet": "db4", "levels": 4},
    },
    "FourierAnalysis": {
        "category": "decomposition",
        "description": "Dominant frequency extraction",
        "compute_cost": 0.4,
        "parameters": {"top_k": 5},
    },
    "HMM_Regime": {
        "category": "ml",
        "description": "Hidden Markov Model for regime detection",
        "compute_cost": 0.8,
        "parameters": {"n_states": 3},
    },
    "KalmanFilter": {
        "category": "ml",
        "description": "State space trend extraction",
        "compute_cost": 0.5,
        "parameters": {"process_noise": 0.01},
    },
    "IsolationForest": {
        "category": "ml",
        "description": "Anomaly detection for unusual patterns",
        "compute_cost": 0.6,
        "parameters": {"contamination": 0.1},
    },
}


# =============================================================================
# RESEARCHER INVENTORY (12 methods)
# =============================================================================

RESEARCHER_INVENTORY = [
    # Statistical Forecasting (4)
    "ARIMA",                  # AutoRegressive Integrated Moving Average
    "ExponentialSmoothing",   # Holt-Winters
    "VectorAutoregression",   # VAR for multi-asset
    "GARCH_Forecast",         # Volatility forecasting

    # Machine Learning (4)
    "RandomForest",           # Ensemble tree model
    "GradientBoosting",       # XGBoost-style
    "LSTM_Forecast",          # Recurrent neural network
    "TemporalFusion",         # Transformer-based TFT

    # Uncertainty Quantification (4)
    "BootstrapEnsemble",      # Bootstrap confidence intervals
    "QuantileRegression",     # Direct quantile estimation
    "BayesianInference",      # Posterior distribution
    "ConformalPrediction",    # Distribution-free intervals
]

RESEARCHER_METHOD_INFO = {
    "ARIMA": {
        "category": "statistical",
        "description": "Classic time series forecasting",
        "compute_cost": 0.3,
        "parameters": {"p": 5, "d": 1, "q": 0},
    },
    "ExponentialSmoothing": {
        "category": "statistical",
        "description": "Trend and seasonality smoothing",
        "compute_cost": 0.2,
        "parameters": {"trend": "add", "seasonal": None},
    },
    "VectorAutoregression": {
        "category": "statistical",
        "description": "Multi-asset joint forecasting",
        "compute_cost": 0.5,
        "parameters": {"max_lag": 5},
    },
    "GARCH_Forecast": {
        "category": "statistical",
        "description": "Volatility forecasting",
        "compute_cost": 0.4,
        "parameters": {"p": 1, "q": 1},
    },
    "RandomForest": {
        "category": "ml",
        "description": "Ensemble decision trees",
        "compute_cost": 0.6,
        "parameters": {"n_estimators": 100, "max_depth": 10},
    },
    "GradientBoosting": {
        "category": "ml",
        "description": "Gradient boosted trees",
        "compute_cost": 0.7,
        "parameters": {"n_estimators": 100, "learning_rate": 0.1},
    },
    "LSTM_Forecast": {
        "category": "ml",
        "description": "Long short-term memory RNN",
        "compute_cost": 1.0,
        "parameters": {"hidden_size": 64, "num_layers": 2},
    },
    "TemporalFusion": {
        "category": "ml",
        "description": "Temporal Fusion Transformer",
        "compute_cost": 1.2,
        "parameters": {"hidden_size": 32, "attention_heads": 4},
    },
    "BootstrapEnsemble": {
        "category": "uncertainty",
        "description": "Bootstrap confidence intervals",
        "compute_cost": 0.5,
        "parameters": {"n_bootstrap": 100},
    },
    "QuantileRegression": {
        "category": "uncertainty",
        "description": "Direct quantile estimation",
        "compute_cost": 0.4,
        "parameters": {"quantiles": [0.05, 0.25, 0.5, 0.75, 0.95]},
    },
    "BayesianInference": {
        "category": "uncertainty",
        "description": "Posterior distribution estimation",
        "compute_cost": 0.8,
        "parameters": {"prior": "normal", "n_samples": 1000},
    },
    "ConformalPrediction": {
        "category": "uncertainty",
        "description": "Distribution-free prediction intervals",
        "compute_cost": 0.3,
        "parameters": {"alpha": 0.1},
    },
}


# =============================================================================
# TRADER INVENTORY (10 methods)
# =============================================================================

TRADER_INVENTORY = [
    # Execution Styles (4)
    "AggressiveMarket",       # Immediate market orders
    "PassiveLimit",           # Patient limit orders
    "TWAP",                   # Time-weighted average price
    "VWAP",                   # Volume-weighted average price

    # Position Sizing (3)
    "KellyCriterion",         # Optimal fraction betting
    "FixedFractional",        # Fixed percentage per trade
    "VolatilityScaled",       # Size inversely to volatility

    # Entry Strategies (3)
    "MomentumEntry",          # Trend-following entries
    "ContrarianEntry",        # Mean-reversion entries
    "BreakoutEntry",          # Range breakout entries
]

TRADER_METHOD_INFO = {
    "AggressiveMarket": {
        "category": "execution",
        "description": "Immediate market orders for urgent execution",
        "compute_cost": 0.1,
        "parameters": {"slippage_tolerance": 0.002},
    },
    "PassiveLimit": {
        "category": "execution",
        "description": "Patient limit orders for better fills",
        "compute_cost": 0.1,
        "parameters": {"offset_bps": 10},
    },
    "TWAP": {
        "category": "execution",
        "description": "Time-weighted execution over period",
        "compute_cost": 0.2,
        "parameters": {"num_slices": 10, "interval_minutes": 30},
    },
    "VWAP": {
        "category": "execution",
        "description": "Volume-weighted execution",
        "compute_cost": 0.3,
        "parameters": {"participation_rate": 0.1},
    },
    "KellyCriterion": {
        "category": "sizing",
        "description": "Optimal growth rate position sizing",
        "compute_cost": 0.2,
        "parameters": {"kelly_fraction": 0.5},
    },
    "FixedFractional": {
        "category": "sizing",
        "description": "Fixed percentage of capital per trade",
        "compute_cost": 0.1,
        "parameters": {"fraction": 0.02},
    },
    "VolatilityScaled": {
        "category": "sizing",
        "description": "Size inversely proportional to volatility",
        "compute_cost": 0.2,
        "parameters": {"target_volatility": 0.15},
    },
    "MomentumEntry": {
        "category": "entry",
        "description": "Enter in direction of momentum",
        "compute_cost": 0.1,
        "parameters": {"momentum_period": 20},
    },
    "ContrarianEntry": {
        "category": "entry",
        "description": "Fade extreme moves",
        "compute_cost": 0.1,
        "parameters": {"zscore_threshold": 2.0},
    },
    "BreakoutEntry": {
        "category": "entry",
        "description": "Enter on range breakouts",
        "compute_cost": 0.1,
        "parameters": {"lookback": 20, "breakout_pct": 0.02},
    },
}


# =============================================================================
# RISK INVENTORY (10 methods)
# =============================================================================

RISK_INVENTORY = [
    # Position Limits (3)
    "MaxLeverage",            # Hard leverage cap
    "MaxPositionSize",        # Maximum position as % of capital
    "ConcentrationLimit",     # Single asset concentration

    # Loss Limits (3)
    "MaxDrawdown",            # Maximum drawdown before stop
    "DailyStopLoss",          # Daily loss limit
    "TrailingStop",           # Trailing stop loss

    # Risk Metrics (2)
    "VaRLimit",               # Value at Risk constraint
    "ExpectedShortfall",      # CVaR / Expected Shortfall

    # Dynamic Risk (2)
    "VolatilityAdjusted",     # Adjust limits by volatility
    "RegimeAware",            # Different limits per regime
]

RISK_METHOD_INFO = {
    "MaxLeverage": {
        "category": "position",
        "description": "Hard cap on leverage",
        "compute_cost": 0.1,
        "parameters": {"max_leverage": 5.0},
    },
    "MaxPositionSize": {
        "category": "position",
        "description": "Maximum position as percentage of capital",
        "compute_cost": 0.1,
        "parameters": {"max_size": 0.3},
    },
    "ConcentrationLimit": {
        "category": "position",
        "description": "Limit concentration in single asset",
        "compute_cost": 0.1,
        "parameters": {"max_concentration": 0.5},
    },
    "MaxDrawdown": {
        "category": "loss",
        "description": "Stop trading after max drawdown",
        "compute_cost": 0.1,
        "parameters": {"max_drawdown": 0.1},
    },
    "DailyStopLoss": {
        "category": "loss",
        "description": "Daily loss limit before stopping",
        "compute_cost": 0.1,
        "parameters": {"max_daily_loss": 0.03},
    },
    "TrailingStop": {
        "category": "loss",
        "description": "Trailing stop loss on positions",
        "compute_cost": 0.1,
        "parameters": {"trail_pct": 0.02},
    },
    "VaRLimit": {
        "category": "metric",
        "description": "Value at Risk constraint (95%)",
        "compute_cost": 0.3,
        "parameters": {"var_limit": 0.05, "confidence": 0.95},
    },
    "ExpectedShortfall": {
        "category": "metric",
        "description": "Expected Shortfall / CVaR constraint",
        "compute_cost": 0.3,
        "parameters": {"es_limit": 0.07, "confidence": 0.95},
    },
    "VolatilityAdjusted": {
        "category": "dynamic",
        "description": "Tighten limits when volatility is high",
        "compute_cost": 0.2,
        "parameters": {"vol_multiplier": 1.5},
    },
    "RegimeAware": {
        "category": "dynamic",
        "description": "Different limits for different regimes",
        "compute_cost": 0.2,
        "parameters": {"bull_leverage": 5.0, "bear_leverage": 2.0},
    },
}


# =============================================================================
# INVENTORY ACCESS FUNCTIONS
# =============================================================================

def get_inventory(role: str) -> List[str]:
    """Get the method inventory for a role."""
    inventories = {
        "analyst": ANALYST_INVENTORY,
        "researcher": RESEARCHER_INVENTORY,
        "trader": TRADER_INVENTORY,
        "risk": RISK_INVENTORY,
    }
    return inventories.get(role, [])


def get_method_info(role: str) -> Dict[str, Dict[str, Any]]:
    """Get method information for a role."""
    info_maps = {
        "analyst": ANALYST_METHOD_INFO,
        "researcher": RESEARCHER_METHOD_INFO,
        "trader": TRADER_METHOD_INFO,
        "risk": RISK_METHOD_INFO,
    }
    return info_maps.get(role, {})


def get_all_inventories() -> Dict[str, List[str]]:
    """Get all inventories."""
    return {
        "analyst": ANALYST_INVENTORY,
        "researcher": RESEARCHER_INVENTORY,
        "trader": TRADER_INVENTORY,
        "risk": RISK_INVENTORY,
    }


def get_inventory_sizes() -> Dict[str, int]:
    """Get the size of each inventory."""
    return {
        "analyst": len(ANALYST_INVENTORY),
        "researcher": len(RESEARCHER_INVENTORY),
        "trader": len(TRADER_INVENTORY),
        "risk": len(RISK_INVENTORY),
    }
