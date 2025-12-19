"""Abstract interfaces for inventory methods."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


# ============= Analyst Methods =============

class FeatureMethod(ABC):
    """Base class for feature construction methods (Analyst A-B)."""
    name: str

    @abstractmethod
    def run(self, price_df: "pd.DataFrame", **kwargs) -> "pd.DataFrame":
        """
        Construct features from price data.

        Args:
            price_df: DataFrame with columns [open, high, low, close, volume]

        Returns:
            DataFrame with constructed features
        """
        ...


class TrendMethod(ABC):
    """Base class for trend detection methods (Analyst A-C)."""
    name: str

    @abstractmethod
    def run(self, price_df: "pd.DataFrame", **kwargs) -> "pd.DataFrame":
        """
        Detect trends and regime states from price data.

        Args:
            price_df: DataFrame with price data

        Returns:
            DataFrame with columns like [prob_up, prob_down, regime, slope]
        """
        ...


# ============= Researcher Methods =============

class ForecastMethod(ABC):
    """Base class for forecasting methods (Researcher R-A)."""
    name: str

    @abstractmethod
    def run(self, features: "pd.DataFrame", trend: "pd.DataFrame", **kwargs) -> Dict[str, float]:
        """
        Generate price forecasts at various horizons.

        Args:
            features: Feature DataFrame from Analyst
            trend: Trend DataFrame from Analyst

        Returns:
            Dict mapping horizon (e.g., "8h", "24h") to predicted % change
        """
        ...


class UncertaintyMethod(ABC):
    """Base class for uncertainty quantification methods (Researcher R-B)."""
    name: str

    @abstractmethod
    def run(self, features: "pd.DataFrame", trend: "pd.DataFrame",
            forecast: Dict[str, float], **kwargs) -> Dict[str, float]:
        """
        Quantify uncertainty/risk in forecasts.

        Args:
            features: Feature DataFrame
            trend: Trend DataFrame
            forecast: Current forecast dict

        Returns:
            Dict with uncertainty metrics (e.g., q05, q95, q10, q90)
        """
        ...


class CalibrationMethod(ABC):
    """Base class for probability calibration methods (Researcher R-C)."""
    name: str

    @abstractmethod
    def run(self, forecast: Dict[str, float], **kwargs) -> Dict[str, float]:
        """
        Calibrate forecast probabilities.

        Args:
            forecast: Raw forecast dict

        Returns:
            Calibrated forecast dict
        """
        ...


# ============= Trader Methods =============

class ExecutionStyleMethod(ABC):
    """Base class for execution style methods (Trader T-A)."""
    name: str

    @abstractmethod
    def choose(self, research_summary: Dict[str, Any], news: List[Any], **kwargs) -> str:
        """
        Choose execution style based on market context.

        Args:
            research_summary: Research summary dict from Researcher
            news: List of news items

        Returns:
            Execution style name (e.g., "Aggressive_Market")
        """
        ...


# ============= Risk Methods =============

class RiskCheckMethod(ABC):
    """Base class for risk check methods (Risk Manager M-A)."""
    name: str

    @abstractmethod
    def evaluate(self, execution: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate risk of proposed execution.

        Args:
            execution: Proposed execution dict
            context: Context with price_df, account_value, etc.

        Returns:
            Dict with verdict ("pass", "soft_fail", "hard_fail"), reasons, envelope
        """
        ...
