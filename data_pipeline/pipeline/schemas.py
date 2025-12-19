"""Data validation models (Pydantic v2).
- PriceBar: one OHLCV bar.
- NewsItem: one news article metadata.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator

class PriceBar(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @field_validator("high")
    @classmethod
    def high_not_below_low(cls, v, info):
        low = info.data.get("low")
        if low is not None and v < low:
            raise ValueError("high < low")
        return v

class NewsItem(BaseModel):
    source: str
    title: str
    url: str
    published_at: datetime
    tickers: list[str] = Field(default_factory=list)
    summary: str = ""

    @field_validator("published_at")
    @classmethod
    def ensure_tz(cls, v):
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)
