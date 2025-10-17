
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

class FeatureConstructionMethod(ABC):
    name: str
    @abstractmethod
    def run(self, price_df: pd.DataFrame, **kw) -> pd.DataFrame: ...

class TrendDetectionMethod(ABC):
    name: str
    @abstractmethod
    def run(self, price_df: pd.DataFrame, **kw) -> pd.DataFrame: ...

class ForecastingMethod(ABC):
    name: str
    @abstractmethod
    def run(self, features: pd.DataFrame, trend: pd.DataFrame, **kw) -> Dict[str, float]: ...

class UncertaintyMethod(ABC):
    name: str
    @abstractmethod
    def run(self, features: pd.DataFrame, trend: pd.DataFrame, forecast: Dict[str, float], **kw) -> Dict[str, float]: ...

class CalibrationMethod(ABC):
    name: str
    @abstractmethod
    def run(self, forecast: Dict[str, float], **kw) -> Dict[str, float]: ...

class ExecutionStyleMethod(ABC):
    name: str
    @abstractmethod
    def choose(self, summary: Dict[str, Any], news: list, **kw) -> str: ...

class RiskCheckMethod(ABC):
    name: str
    @abstractmethod
    def evaluate(self, exec_sum: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]: ...
