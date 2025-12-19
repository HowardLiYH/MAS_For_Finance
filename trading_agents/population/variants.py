"""Agent variants for each role in the population.

Each role (Analyst, Researcher, Trader, Risk) has 5 distinct variants
with different strategies, parameters, and approaches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum
import numpy as np
import pandas as pd

from .base import PopulationAgent, AgentRole, PopulationConfig, AgentPopulation


# =============================================================================
# ANALYST VARIANTS
# =============================================================================

class AnalystVariantType(Enum):
    """Types of analyst variants."""
    TECHNICAL = "technical"      # TALib-focused technical analysis
    STATISTICAL = "statistical"  # Statistical features (autocorr, volatility)
    MOMENTUM = "momentum"        # Momentum and trend indicators
    VOLATILITY = "volatility"    # Volatility-focused analysis
    HYBRID = "hybrid"            # Adaptive mix of approaches


@dataclass
class AnalystConfig:
    """Configuration for an analyst variant."""
    variant_type: AnalystVariantType
    lookback_periods: List[int]
    feature_weights: Dict[str, float]
    trend_sensitivity: float
    regime_threshold: float


class AnalystVariant(PopulationAgent):
    """An analyst agent variant for population-based learning."""

    DEFAULT_CONFIGS = {
        AnalystVariantType.TECHNICAL: AnalystConfig(
            variant_type=AnalystVariantType.TECHNICAL,
            lookback_periods=[14, 21, 50],
            feature_weights={"rsi": 0.3, "macd": 0.3, "bb": 0.2, "adx": 0.2},
            trend_sensitivity=0.6,
            regime_threshold=0.5,
        ),
        AnalystVariantType.STATISTICAL: AnalystConfig(
            variant_type=AnalystVariantType.STATISTICAL,
            lookback_periods=[20, 60, 120],
            feature_weights={"autocorr": 0.4, "volatility": 0.3, "skew": 0.15, "kurt": 0.15},
            trend_sensitivity=0.4,
            regime_threshold=0.6,
        ),
        AnalystVariantType.MOMENTUM: AnalystConfig(
            variant_type=AnalystVariantType.MOMENTUM,
            lookback_periods=[5, 10, 20],
            feature_weights={"roc": 0.4, "mom": 0.3, "trix": 0.2, "ppo": 0.1},
            trend_sensitivity=0.8,
            regime_threshold=0.4,
        ),
        AnalystVariantType.VOLATILITY: AnalystConfig(
            variant_type=AnalystVariantType.VOLATILITY,
            lookback_periods=[10, 20, 30],
            feature_weights={"atr": 0.35, "std": 0.25, "range": 0.2, "bb_width": 0.2},
            trend_sensitivity=0.3,
            regime_threshold=0.7,
        ),
        AnalystVariantType.HYBRID: AnalystConfig(
            variant_type=AnalystVariantType.HYBRID,
            lookback_periods=[14, 28, 56],
            feature_weights={"rsi": 0.2, "volatility": 0.2, "momentum": 0.2, "trend": 0.2, "volume": 0.2},
            trend_sensitivity=0.5,
            regime_threshold=0.5,
        ),
    }

    def __init__(self, variant_type: AnalystVariantType, config: Optional[AnalystConfig] = None):
        self.variant_type = variant_type
        self.config = config or self.DEFAULT_CONFIGS[variant_type]
        super().__init__(
            variant_name=variant_type.value,
            variant_config={"type": variant_type.value}
        )
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize learnable parameters."""
        self.parameters = {
            "lookback_periods": list(self.config.lookback_periods),
            "feature_weights": dict(self.config.feature_weights),
            "trend_sensitivity": self.config.trend_sensitivity,
            "regime_threshold": self.config.regime_threshold,
        }

    @property
    def role(self) -> AgentRole:
        return AgentRole.ANALYST

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)
        # Ensure weights sum to 1
        if "feature_weights" in parameters:
            weights = parameters["feature_weights"]
            total = sum(weights.values())
            if total > 0:
                self.parameters["feature_weights"] = {k: v/total for k, v in weights.items()}

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis on price data."""
        price_df = inputs.get("price_data")
        if price_df is None or price_df.empty:
            return {"features": {}, "trend": "neutral", "regime": "unknown"}

        features = self._extract_features(price_df)
        trend = self._detect_trend(features)
        regime = self._detect_regime(features)

        return {
            "features": features,
            "trend": trend,
            "regime": regime,
            "variant": self.variant_name,
            "agent_id": self.id,
        }

    def _extract_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract features based on variant type."""
        features = {}
        close = df["close"].values if "close" in df.columns else df.iloc[:, 0].values

        for period in self.parameters["lookback_periods"]:
            if len(close) > period:
                # Basic features
                features[f"return_{period}"] = (close[-1] / close[-period] - 1) if close[-period] != 0 else 0
                features[f"volatility_{period}"] = np.std(np.diff(np.log(close[-period:]))) if len(close) > period else 0
                features[f"momentum_{period}"] = close[-1] - close[-period] if len(close) > period else 0

        return features

    def _detect_trend(self, features: Dict[str, float]) -> str:
        """Detect market trend."""
        returns = [v for k, v in features.items() if "return" in k]
        if not returns:
            return "neutral"

        avg_return = np.mean(returns)
        threshold = self.parameters["trend_sensitivity"] * 0.01

        if avg_return > threshold:
            return "bullish"
        elif avg_return < -threshold:
            return "bearish"
        return "neutral"

    def _detect_regime(self, features: Dict[str, float]) -> str:
        """Detect market regime."""
        vols = [v for k, v in features.items() if "volatility" in k]
        if not vols:
            return "normal"

        avg_vol = np.mean(vols)
        threshold = self.parameters["regime_threshold"] * 0.02

        if avg_vol > threshold:
            return "high_volatility"
        elif avg_vol < threshold * 0.5:
            return "low_volatility"
        return "normal"


# =============================================================================
# RESEARCHER VARIANTS
# =============================================================================

class ResearcherVariantType(Enum):
    """Types of researcher variants."""
    STATISTICAL = "statistical"    # ARIMA, statistical forecasting
    ENSEMBLE = "ensemble"          # Ensemble methods
    BAYESIAN = "bayesian"          # Bayesian inference
    QUANTILE = "quantile"          # Quantile regression focus
    ADAPTIVE = "adaptive"          # Adaptive learning rate


@dataclass
class ResearcherConfig:
    """Configuration for a researcher variant."""
    variant_type: ResearcherVariantType
    forecast_horizons: List[int]
    confidence_method: str
    uncertainty_quantiles: List[float]
    calibration_window: int


class ResearcherVariant(PopulationAgent):
    """A researcher agent variant for population-based learning."""

    DEFAULT_CONFIGS = {
        ResearcherVariantType.STATISTICAL: ResearcherConfig(
            variant_type=ResearcherVariantType.STATISTICAL,
            forecast_horizons=[2, 6, 12],  # in 4h periods
            confidence_method="bootstrap",
            uncertainty_quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
            calibration_window=100,
        ),
        ResearcherVariantType.ENSEMBLE: ResearcherConfig(
            variant_type=ResearcherVariantType.ENSEMBLE,
            forecast_horizons=[2, 6, 12],
            confidence_method="ensemble_std",
            uncertainty_quantiles=[0.1, 0.5, 0.9],
            calibration_window=50,
        ),
        ResearcherVariantType.BAYESIAN: ResearcherConfig(
            variant_type=ResearcherVariantType.BAYESIAN,
            forecast_horizons=[2, 6, 12],
            confidence_method="posterior",
            uncertainty_quantiles=[0.05, 0.5, 0.95],
            calibration_window=200,
        ),
        ResearcherVariantType.QUANTILE: ResearcherConfig(
            variant_type=ResearcherVariantType.QUANTILE,
            forecast_horizons=[2, 6, 12],
            confidence_method="quantile",
            uncertainty_quantiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
            calibration_window=150,
        ),
        ResearcherVariantType.ADAPTIVE: ResearcherConfig(
            variant_type=ResearcherVariantType.ADAPTIVE,
            forecast_horizons=[1, 4, 8],
            confidence_method="adaptive",
            uncertainty_quantiles=[0.1, 0.5, 0.9],
            calibration_window=30,
        ),
    }

    def __init__(self, variant_type: ResearcherVariantType, config: Optional[ResearcherConfig] = None):
        self.variant_type = variant_type
        self.config = config or self.DEFAULT_CONFIGS[variant_type]
        super().__init__(
            variant_name=variant_type.value,
            variant_config={"type": variant_type.value}
        )
        self._init_parameters()

    def _init_parameters(self) -> None:
        self.parameters = {
            "forecast_horizons": list(self.config.forecast_horizons),
            "confidence_method": self.config.confidence_method,
            "uncertainty_quantiles": list(self.config.uncertainty_quantiles),
            "calibration_window": self.config.calibration_window,
            "base_confidence": 0.6,
            "momentum_weight": 0.3,
            "mean_reversion_weight": 0.3,
        }

    @property
    def role(self) -> AgentRole:
        return AgentRole.RESEARCHER

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate research summary with forecasts and uncertainty."""
        features = inputs.get("features", {})
        trend = inputs.get("trend", "neutral")
        price_data = inputs.get("price_data")

        forecast = self._generate_forecast(features, trend, price_data)
        confidence = self._calculate_confidence(features)
        risk = self._calculate_risk(features, price_data)
        recommendation = self._generate_recommendation(forecast, confidence, trend)

        return {
            "market_state": trend,
            "recommendation": recommendation,
            "confidence": confidence,
            "forecast": forecast,
            "risk": risk,
            "variant": self.variant_name,
            "agent_id": self.id,
        }

    def _generate_forecast(self, features: Dict, trend: str, price_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Generate price forecasts for different horizons."""
        base_return = 0.0

        # Use features to estimate expected return
        returns = [v for k, v in features.items() if "return" in k]
        if returns:
            momentum = np.mean(returns) * self.parameters["momentum_weight"]
            mean_rev = -np.mean(returns) * self.parameters["mean_reversion_weight"]
            base_return = momentum + mean_rev

        forecast = {}
        for horizon in self.parameters["forecast_horizons"]:
            # Scale by horizon with decay
            forecast[f"{horizon * 4}h"] = base_return * np.sqrt(horizon) * 0.5

        return forecast

    def _calculate_confidence(self, features: Dict) -> float:
        """Calculate forecast confidence."""
        base = self.parameters["base_confidence"]

        # Adjust based on feature consistency
        returns = [v for k, v in features.items() if "return" in k]
        if returns and len(returns) > 1:
            consistency = 1 - np.std(returns) / (np.abs(np.mean(returns)) + 0.01)
            base = base * (0.5 + 0.5 * np.clip(consistency, 0, 1))

        return np.clip(base, 0.1, 0.95)

    def _calculate_risk(self, features: Dict, price_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Calculate risk metrics."""
        vols = [v for k, v in features.items() if "volatility" in k]
        avg_vol = np.mean(vols) if vols else 0.02

        return {
            "q05": -2.0 * avg_vol,
            "q25": -0.5 * avg_vol,
            "q50": 0.0,
            "q75": 0.5 * avg_vol,
            "q95": 2.0 * avg_vol,
            "var_95": 1.65 * avg_vol,
        }

    def _generate_recommendation(self, forecast: Dict, confidence: float, trend: str) -> str:
        """Generate trading recommendation."""
        if not forecast:
            return "HOLD"

        avg_forecast = np.mean(list(forecast.values()))

        if confidence > 0.6:
            if avg_forecast > 0.005:
                return "BUY"
            elif avg_forecast < -0.005:
                return "SELL"

        return "HOLD"


# =============================================================================
# TRADER VARIANTS
# =============================================================================

class TraderVariantType(Enum):
    """Types of trader variants."""
    AGGRESSIVE = "aggressive"      # High risk, high reward
    CONSERVATIVE = "conservative"  # Low risk, steady
    MOMENTUM = "momentum"          # Follow trends
    CONTRARIAN = "contrarian"      # Fade moves
    ADAPTIVE = "adaptive"          # Adjust based on conditions


@dataclass
class TraderConfig:
    """Configuration for a trader variant."""
    variant_type: TraderVariantType
    base_position_size: float
    max_leverage: float
    risk_per_trade: float
    take_profit_multiplier: float
    stop_loss_multiplier: float


class TraderVariant(PopulationAgent):
    """A trader agent variant for population-based learning."""

    DEFAULT_CONFIGS = {
        TraderVariantType.AGGRESSIVE: TraderConfig(
            variant_type=TraderVariantType.AGGRESSIVE,
            base_position_size=0.4,
            max_leverage=8.0,
            risk_per_trade=0.03,
            take_profit_multiplier=3.0,
            stop_loss_multiplier=1.0,
        ),
        TraderVariantType.CONSERVATIVE: TraderConfig(
            variant_type=TraderVariantType.CONSERVATIVE,
            base_position_size=0.15,
            max_leverage=2.0,
            risk_per_trade=0.01,
            take_profit_multiplier=1.5,
            stop_loss_multiplier=0.5,
        ),
        TraderVariantType.MOMENTUM: TraderConfig(
            variant_type=TraderVariantType.MOMENTUM,
            base_position_size=0.3,
            max_leverage=5.0,
            risk_per_trade=0.02,
            take_profit_multiplier=2.5,
            stop_loss_multiplier=0.8,
        ),
        TraderVariantType.CONTRARIAN: TraderConfig(
            variant_type=TraderVariantType.CONTRARIAN,
            base_position_size=0.25,
            max_leverage=4.0,
            risk_per_trade=0.015,
            take_profit_multiplier=2.0,
            stop_loss_multiplier=1.0,
        ),
        TraderVariantType.ADAPTIVE: TraderConfig(
            variant_type=TraderVariantType.ADAPTIVE,
            base_position_size=0.25,
            max_leverage=5.0,
            risk_per_trade=0.02,
            take_profit_multiplier=2.0,
            stop_loss_multiplier=0.75,
        ),
    }

    def __init__(self, variant_type: TraderVariantType, config: Optional[TraderConfig] = None):
        self.variant_type = variant_type
        self.config = config or self.DEFAULT_CONFIGS[variant_type]
        super().__init__(
            variant_name=variant_type.value,
            variant_config={"type": variant_type.value}
        )
        self._init_parameters()

    def _init_parameters(self) -> None:
        self.parameters = {
            "base_position_size": self.config.base_position_size,
            "max_leverage": self.config.max_leverage,
            "risk_per_trade": self.config.risk_per_trade,
            "take_profit_multiplier": self.config.take_profit_multiplier,
            "stop_loss_multiplier": self.config.stop_loss_multiplier,
            "confidence_scaling": 1.0,
            "volatility_adjustment": 0.5,
        }

    @property
    def role(self) -> AgentRole:
        return AgentRole.TRADER

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)
        # Clamp values
        self.parameters["base_position_size"] = np.clip(self.parameters["base_position_size"], 0.05, 0.5)
        self.parameters["max_leverage"] = np.clip(self.parameters["max_leverage"], 1.0, 10.0)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading proposal."""
        research = inputs.get("research", {})
        news_digest = inputs.get("news_digest", {})
        current_price = inputs.get("current_price", 0)

        if current_price <= 0:
            return self._empty_proposal()

        direction = self._determine_direction(research, news_digest)
        position_size = self._calculate_position_size(research)
        leverage = self._calculate_leverage(research)
        tp, sl = self._calculate_exits(current_price, direction, research)

        return {
            "direction": direction,
            "position_size": position_size,
            "leverage": leverage,
            "order_type": "MARKET",
            "entry_price": current_price,
            "take_profit": tp,
            "stop_loss": sl,
            "variant": self.variant_name,
            "agent_id": self.id,
        }

    def _determine_direction(self, research: Dict, news: Dict) -> str:
        """Determine trade direction."""
        rec = research.get("recommendation", "HOLD")

        if self.variant_type == TraderVariantType.CONTRARIAN:
            # Fade the recommendation
            if rec == "BUY":
                return "SHORT"
            elif rec == "SELL":
                return "LONG"

        # Normal direction
        if rec == "BUY":
            return "LONG"
        elif rec == "SELL":
            return "SHORT"
        return "LONG"  # Default

    def _calculate_position_size(self, research: Dict) -> float:
        """Calculate position size based on confidence."""
        confidence = research.get("confidence", 0.5)
        base = self.parameters["base_position_size"]
        scaling = self.parameters["confidence_scaling"]

        size = base * (0.5 + 0.5 * confidence * scaling)
        return np.clip(size, 0.05, 0.5)

    def _calculate_leverage(self, research: Dict) -> float:
        """Calculate leverage."""
        confidence = research.get("confidence", 0.5)
        max_lev = self.parameters["max_leverage"]

        # Scale leverage with confidence
        lev = 1.0 + (max_lev - 1.0) * confidence * 0.8
        return np.clip(lev, 1.0, max_lev)

    def _calculate_exits(self, price: float, direction: str, research: Dict) -> tuple:
        """Calculate take profit and stop loss."""
        risk = research.get("risk", {})
        vol = abs(risk.get("var_95", 0.02))

        tp_mult = self.parameters["take_profit_multiplier"]
        sl_mult = self.parameters["stop_loss_multiplier"]

        if direction == "LONG":
            tp = price * (1 + vol * tp_mult)
            sl = price * (1 - vol * sl_mult)
        else:
            tp = price * (1 - vol * tp_mult)
            sl = price * (1 + vol * sl_mult)

        return tp, sl

    def _empty_proposal(self) -> Dict[str, Any]:
        return {
            "direction": "LONG",
            "position_size": 0.0,
            "leverage": 1.0,
            "order_type": "MARKET",
            "entry_price": 0,
            "take_profit": 0,
            "stop_loss": 0,
            "variant": self.variant_name,
            "agent_id": self.id,
        }


# =============================================================================
# RISK MANAGER VARIANTS
# =============================================================================

class RiskVariantType(Enum):
    """Types of risk manager variants."""
    STRICT = "strict"        # Very tight limits
    MODERATE = "moderate"    # Balanced approach
    DYNAMIC = "dynamic"      # Adjusts to conditions
    VAR_BASED = "var_based"  # VaR-focused
    DRAWDOWN = "drawdown"    # Drawdown-focused


@dataclass
class RiskConfig:
    """Configuration for a risk manager variant."""
    variant_type: RiskVariantType
    max_leverage: float
    max_position_size: float
    max_daily_loss: float
    max_drawdown: float
    margin_buffer: float


class RiskVariant(PopulationAgent):
    """A risk manager agent variant for population-based learning."""

    DEFAULT_CONFIGS = {
        RiskVariantType.STRICT: RiskConfig(
            variant_type=RiskVariantType.STRICT,
            max_leverage=3.0,
            max_position_size=0.2,
            max_daily_loss=0.02,
            max_drawdown=0.05,
            margin_buffer=0.3,
        ),
        RiskVariantType.MODERATE: RiskConfig(
            variant_type=RiskVariantType.MODERATE,
            max_leverage=5.0,
            max_position_size=0.35,
            max_daily_loss=0.03,
            max_drawdown=0.10,
            margin_buffer=0.2,
        ),
        RiskVariantType.DYNAMIC: RiskConfig(
            variant_type=RiskVariantType.DYNAMIC,
            max_leverage=6.0,
            max_position_size=0.4,
            max_daily_loss=0.04,
            max_drawdown=0.12,
            margin_buffer=0.15,
        ),
        RiskVariantType.VAR_BASED: RiskConfig(
            variant_type=RiskVariantType.VAR_BASED,
            max_leverage=5.0,
            max_position_size=0.3,
            max_daily_loss=0.025,
            max_drawdown=0.08,
            margin_buffer=0.25,
        ),
        RiskVariantType.DRAWDOWN: RiskConfig(
            variant_type=RiskVariantType.DRAWDOWN,
            max_leverage=4.0,
            max_position_size=0.25,
            max_daily_loss=0.02,
            max_drawdown=0.06,
            margin_buffer=0.25,
        ),
    }

    def __init__(self, variant_type: RiskVariantType, config: Optional[RiskConfig] = None):
        self.variant_type = variant_type
        self.config = config or self.DEFAULT_CONFIGS[variant_type]
        super().__init__(
            variant_name=variant_type.value,
            variant_config={"type": variant_type.value}
        )
        self._init_parameters()

    def _init_parameters(self) -> None:
        self.parameters = {
            "max_leverage": self.config.max_leverage,
            "max_position_size": self.config.max_position_size,
            "max_daily_loss": self.config.max_daily_loss,
            "max_drawdown": self.config.max_drawdown,
            "margin_buffer": self.config.margin_buffer,
            "hard_fail_multiplier": 1.5,  # Hard fail at 1.5x soft limits
        }

    @property
    def role(self) -> AgentRole:
        return AgentRole.RISK

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Review trading proposal for risk."""
        proposal = inputs.get("proposal", {})
        portfolio_state = inputs.get("portfolio_state", {})

        violations = []
        adjustments = {}

        # Check leverage
        lev = proposal.get("leverage", 1.0)
        max_lev = self.parameters["max_leverage"]
        hard_lev = max_lev * self.parameters["hard_fail_multiplier"]

        if lev > hard_lev:
            violations.append({"type": "leverage", "severity": "hard_fail", "value": lev, "limit": hard_lev})
        elif lev > max_lev:
            violations.append({"type": "leverage", "severity": "soft_fail", "value": lev, "limit": max_lev})
            adjustments["leverage"] = max_lev

        # Check position size
        size = proposal.get("position_size", 0)
        max_size = self.parameters["max_position_size"]
        hard_size = max_size * self.parameters["hard_fail_multiplier"]

        if size > hard_size:
            violations.append({"type": "position_size", "severity": "hard_fail", "value": size, "limit": hard_size})
        elif size > max_size:
            violations.append({"type": "position_size", "severity": "soft_fail", "value": size, "limit": max_size})
            adjustments["position_size"] = max_size

        # Check drawdown
        current_dd = portfolio_state.get("drawdown", 0)
        max_dd = self.parameters["max_drawdown"]

        if current_dd > max_dd:
            violations.append({"type": "drawdown", "severity": "hard_fail", "value": current_dd, "limit": max_dd})

        # Determine verdict
        hard_fails = [v for v in violations if v["severity"] == "hard_fail"]
        soft_fails = [v for v in violations if v["severity"] == "soft_fail"]

        if hard_fails:
            verdict = "hard_fail"
        elif soft_fails:
            verdict = "soft_fail"
        else:
            verdict = "pass"

        return {
            "verdict": verdict,
            "violations": violations,
            "adjustments": adjustments,
            "variant": self.variant_name,
            "agent_id": self.id,
        }


# =============================================================================
# POPULATION FACTORY FUNCTIONS
# =============================================================================

def create_analyst_population(size: int = 5) -> AgentPopulation[AnalystVariant]:
    """Create a population of analyst variants."""
    config = PopulationConfig(role=AgentRole.ANALYST, size=size)
    variants = list(AnalystVariantType)[:size]
    agents = [AnalystVariant(v) for v in variants]
    return AgentPopulation(config, agents)


def create_researcher_population(size: int = 5) -> AgentPopulation[ResearcherVariant]:
    """Create a population of researcher variants."""
    config = PopulationConfig(role=AgentRole.RESEARCHER, size=size)
    variants = list(ResearcherVariantType)[:size]
    agents = [ResearcherVariant(v) for v in variants]
    return AgentPopulation(config, agents)


def create_trader_population(size: int = 5) -> AgentPopulation[TraderVariant]:
    """Create a population of trader variants."""
    config = PopulationConfig(role=AgentRole.TRADER, size=size)
    variants = list(TraderVariantType)[:size]
    agents = [TraderVariant(v) for v in variants]
    return AgentPopulation(config, agents)


def create_risk_population(size: int = 5) -> AgentPopulation[RiskVariant]:
    """Create a population of risk manager variants."""
    config = PopulationConfig(role=AgentRole.RISK, size=size)
    variants = list(RiskVariantType)[:size]
    agents = [RiskVariant(v) for v in variants]
    return AgentPopulation(config, agents)
