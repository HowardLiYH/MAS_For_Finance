"""Cross-asset feature generators for multi-asset trading.

Generates market-wide signals from multiple crypto assets to provide
context for trading decisions across all assets.
"""
from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np


@dataclass
class MarketContext:
    """
    Container for cross-asset market context features.

    Provides market-wide signals that can be used by all trading agents
    to understand the broader market environment.
    """
    # Core features DataFrame (indexed by timestamp)
    features: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Individual feature series for convenience
    btc_dominance: pd.Series = field(default_factory=pd.Series)
    altcoin_momentum: pd.Series = field(default_factory=pd.Series)
    eth_btc_ratio: pd.Series = field(default_factory=pd.Series)
    cross_oi_change: pd.Series = field(default_factory=pd.Series)
    aggregate_funding: pd.Series = field(default_factory=pd.Series)
    risk_on_off: pd.Series = field(default_factory=pd.Series)
    market_volatility: pd.Series = field(default_factory=pd.Series)
    cross_correlation: pd.Series = field(default_factory=pd.Series)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "btc_dominance": self.btc_dominance.to_dict() if not self.btc_dominance.empty else {},
            "altcoin_momentum": self.altcoin_momentum.to_dict() if not self.altcoin_momentum.empty else {},
            "eth_btc_ratio": self.eth_btc_ratio.to_dict() if not self.eth_btc_ratio.empty else {},
            "cross_oi_change": self.cross_oi_change.to_dict() if not self.cross_oi_change.empty else {},
            "aggregate_funding": self.aggregate_funding.to_dict() if not self.aggregate_funding.empty else {},
            "risk_on_off": self.risk_on_off.to_dict() if not self.risk_on_off.empty else {},
        }


def btc_dominance(
    btc_df: pd.DataFrame,
    all_dfs: Dict[str, pd.DataFrame],
    volume_weighted: bool = True,
) -> pd.Series:
    """
    Calculate BTC dominance proxy based on turnover/volume.

    Higher dominance = BTC-centric market (risk-off)
    Lower dominance = Altcoin rotation (risk-on)

    Args:
        btc_df: BTC DataFrame
        all_dfs: Dict of all asset DataFrames
        volume_weighted: Use turnover (True) or simple count

    Returns:
        Series with BTC dominance ratio (0-1)
    """
    if volume_weighted:
        # Use turnover if available, else volume * close
        btc_vol = btc_df.get("turnover", btc_df["volume"] * btc_df["close"])

        total_vol = btc_vol.copy()
        for symbol, df in all_dfs.items():
            if symbol != "BTC":
                vol = df.get("turnover", df["volume"] * df["close"])
                total_vol = total_vol + vol.reindex(total_vol.index, fill_value=0)

        dominance = btc_vol / (total_vol + 1e-9)
    else:
        # Simple equal-weight
        dominance = pd.Series(1.0 / len(all_dfs), index=btc_df.index)

    return dominance.clip(0, 1).rename("btc_dominance")


def altcoin_momentum(
    all_dfs: Dict[str, pd.DataFrame],
    exclude: List[str] = None,
    window: int = 6,  # 6 bars = 24h for 4h data
) -> pd.Series:
    """
    Calculate equal-weighted altcoin momentum (excluding BTC).

    Positive = Altcoins outperforming
    Negative = Altcoins underperforming

    Args:
        all_dfs: Dict of all asset DataFrames
        exclude: Symbols to exclude (default: ["BTC"])
        window: Lookback window for momentum

    Returns:
        Series with average altcoin return
    """
    exclude = exclude or ["BTC"]

    returns = []
    for symbol, df in all_dfs.items():
        if symbol not in exclude and "close" in df.columns:
            ret = np.log(df["close"]).diff(window)
            returns.append(ret)

    if not returns:
        return pd.Series(dtype=float, name="altcoin_momentum")

    # Equal-weighted average
    combined = pd.concat(returns, axis=1)
    momentum = combined.mean(axis=1)

    return momentum.rename("altcoin_momentum")


def eth_btc_ratio(
    eth_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    normalize: bool = True,
    window: int = 24,  # 4 days for 4h data
) -> pd.Series:
    """
    Calculate ETH/BTC ratio and its z-score.

    Rising = ETH strength / risk-on
    Falling = BTC strength / flight to quality

    Args:
        eth_df: ETH DataFrame
        btc_df: BTC DataFrame
        normalize: Return z-score (True) or raw ratio
        window: Lookback for z-score normalization

    Returns:
        Series with ETH/BTC ratio (or z-score)
    """
    if "close" not in eth_df.columns or "close" not in btc_df.columns:
        return pd.Series(dtype=float, name="eth_btc_ratio")

    # Align indices
    common_idx = eth_df.index.intersection(btc_df.index)
    ratio = eth_df.loc[common_idx, "close"] / btc_df.loc[common_idx, "close"]

    if normalize:
        mean = ratio.rolling(window, min_periods=1).mean()
        std = ratio.rolling(window, min_periods=1).std()
        ratio = (ratio - mean) / (std + 1e-9)

    return ratio.rename("eth_btc_ratio")


def cross_oi_delta(
    all_dfs: Dict[str, pd.DataFrame],
    window: int = 1,
) -> pd.Series:
    """
    Calculate total OI change across all assets.

    Rising OI = New money entering / conviction
    Falling OI = Deleveraging / uncertainty

    Args:
        all_dfs: Dict of all asset DataFrames
        window: Lookback for delta calculation

    Returns:
        Series with aggregate OI change (%)
    """
    oi_changes = []
    weights = []

    for symbol, df in all_dfs.items():
        oi_col = "oi_usd" if "oi_usd" in df.columns else "oi_btc"
        if oi_col in df.columns:
            oi = df[oi_col]
            oi_pct_change = oi.pct_change(window)
            oi_changes.append(oi_pct_change)
            # Weight by OI size
            weights.append(oi.rolling(window * 2, min_periods=1).mean())

    if not oi_changes:
        return pd.Series(dtype=float, name="cross_oi_delta")

    # Volume-weighted average OI change
    changes_df = pd.concat(oi_changes, axis=1)
    weights_df = pd.concat(weights, axis=1)

    # Normalize weights
    weights_norm = weights_df.div(weights_df.sum(axis=1), axis=0)

    weighted_change = (changes_df * weights_norm).sum(axis=1)

    return weighted_change.rename("cross_oi_delta")


def aggregate_funding(
    all_dfs: Dict[str, pd.DataFrame],
    volume_weighted: bool = True,
) -> pd.Series:
    """
    Calculate volume-weighted average funding rate across assets.

    High positive = Market very long / crowded
    High negative = Market very short / capitulation

    Args:
        all_dfs: Dict of all asset DataFrames
        volume_weighted: Weight by turnover (True) or equal-weight

    Returns:
        Series with aggregate funding rate
    """
    funding_rates = []
    weights = []

    for symbol, df in all_dfs.items():
        if "funding_rate" in df.columns:
            fr = df["funding_rate"]
            funding_rates.append(fr)

            if volume_weighted:
                w = df.get("turnover", df["volume"] * df["close"])
                weights.append(w)

    if not funding_rates:
        return pd.Series(dtype=float, name="aggregate_funding")

    funding_df = pd.concat(funding_rates, axis=1)

    if volume_weighted and weights:
        weights_df = pd.concat(weights, axis=1)
        weights_norm = weights_df.div(weights_df.sum(axis=1), axis=0)
        agg_funding = (funding_df * weights_norm).sum(axis=1)
    else:
        agg_funding = funding_df.mean(axis=1)

    return agg_funding.rename("aggregate_funding")


def risk_on_off(
    all_dfs: Dict[str, pd.DataFrame],
    btc_symbol: str = "BTC",
    window: int = 6,
) -> pd.Series:
    """
    Calculate risk-on/risk-off indicator based on altcoin beta.

    Measures how much altcoins are moving relative to BTC.
    High = Risk-on (altcoins amplifying BTC moves)
    Low = Risk-off (altcoins lagging)

    Args:
        all_dfs: Dict of all asset DataFrames
        btc_symbol: BTC symbol key
        window: Lookback for beta calculation

    Returns:
        Series with risk-on/off indicator (-1 to 1 scaled)
    """
    if btc_symbol not in all_dfs:
        return pd.Series(dtype=float, name="risk_on_off")

    btc_ret = np.log(all_dfs[btc_symbol]["close"]).diff()

    altcoin_rets = []
    for symbol, df in all_dfs.items():
        if symbol != btc_symbol and "close" in df.columns:
            ret = np.log(df["close"]).diff()
            altcoin_rets.append(ret)

    if not altcoin_rets:
        return pd.Series(dtype=float, name="risk_on_off")

    # Calculate rolling beta of altcoins to BTC
    avg_alt_ret = pd.concat(altcoin_rets, axis=1).mean(axis=1)

    # Rolling correlation as risk-on proxy
    correlation = avg_alt_ret.rolling(window, min_periods=2).corr(btc_ret)

    # Rolling beta = cov(alt, btc) / var(btc)
    cov = avg_alt_ret.rolling(window, min_periods=2).cov(btc_ret)
    var = btc_ret.rolling(window, min_periods=2).var()
    beta = cov / (var + 1e-9)

    # Combine: high correlation + high beta = risk-on
    risk_indicator = (correlation * beta).clip(-2, 2) / 2

    return risk_indicator.rename("risk_on_off")


def market_volatility(
    all_dfs: Dict[str, pd.DataFrame],
    window: int = 24,  # 4 days for 4h data
) -> pd.Series:
    """
    Calculate aggregate market volatility.

    Args:
        all_dfs: Dict of all asset DataFrames
        window: Lookback for volatility calculation

    Returns:
        Series with annualized volatility
    """
    vols = []
    weights = []

    for symbol, df in all_dfs.items():
        if "close" in df.columns:
            ret = np.log(df["close"]).diff()
            vol = ret.rolling(window, min_periods=2).std()
            vols.append(vol)
            weights.append(df.get("turnover", df["volume"] * df["close"]))

    if not vols:
        return pd.Series(dtype=float, name="market_volatility")

    vol_df = pd.concat(vols, axis=1)
    weights_df = pd.concat(weights, axis=1)
    weights_norm = weights_df.div(weights_df.sum(axis=1), axis=0)

    # Weighted average volatility, annualized (4h bars = 6 * 365 bars/year)
    avg_vol = (vol_df * weights_norm).sum(axis=1) * np.sqrt(6 * 365)

    return avg_vol.rename("market_volatility")


def cross_correlation(
    all_dfs: Dict[str, pd.DataFrame],
    window: int = 24,
) -> pd.Series:
    """
    Calculate average pairwise correlation across assets.

    High correlation = Macro-driven / risk-off
    Low correlation = Idiosyncratic / rotation

    Args:
        all_dfs: Dict of all asset DataFrames
        window: Lookback for correlation

    Returns:
        Series with average correlation (0-1)
    """
    returns = []
    for symbol, df in all_dfs.items():
        if "close" in df.columns:
            ret = np.log(df["close"]).diff()
            returns.append(ret.rename(symbol))

    if len(returns) < 2:
        return pd.Series(dtype=float, name="cross_correlation")

    returns_df = pd.concat(returns, axis=1)

    # Rolling correlation matrix
    avg_corr = []
    for i in range(len(returns_df)):
        if i < window:
            avg_corr.append(np.nan)
        else:
            window_df = returns_df.iloc[i-window:i]
            corr_matrix = window_df.corr()
            # Average off-diagonal correlations
            mask = ~np.eye(len(corr_matrix), dtype=bool)
            avg = corr_matrix.values[mask].mean()
            avg_corr.append(avg)

    result = pd.Series(avg_corr, index=returns_df.index, name="cross_correlation")
    return result


def generate_market_context(
    all_dfs: Dict[str, pd.DataFrame],
    btc_symbol: str = "BTC",
    eth_symbol: str = "ETH",
) -> MarketContext:
    """
    Generate complete market context from multi-asset data.

    Args:
        all_dfs: Dict of all asset DataFrames
        btc_symbol: BTC symbol key
        eth_symbol: ETH symbol key

    Returns:
        MarketContext with all cross-asset features
    """
    print("📈 Generating cross-asset market context...")

    ctx = MarketContext()

    # Get BTC and ETH DataFrames
    btc_df = all_dfs.get(btc_symbol, pd.DataFrame())
    eth_df = all_dfs.get(eth_symbol, pd.DataFrame())

    # Generate all features
    if not btc_df.empty:
        ctx.btc_dominance = btc_dominance(btc_df, all_dfs)
        print("  ✓ BTC dominance")

    ctx.altcoin_momentum = altcoin_momentum(all_dfs)
    print("  ✓ Altcoin momentum")

    if not btc_df.empty and not eth_df.empty:
        ctx.eth_btc_ratio = eth_btc_ratio(eth_df, btc_df)
        print("  ✓ ETH/BTC ratio")

    ctx.cross_oi_change = cross_oi_delta(all_dfs)
    print("  ✓ Cross OI delta")

    ctx.aggregate_funding = aggregate_funding(all_dfs)
    print("  ✓ Aggregate funding")

    ctx.risk_on_off = risk_on_off(all_dfs, btc_symbol)
    print("  ✓ Risk on/off")

    ctx.market_volatility = market_volatility(all_dfs)
    print("  ✓ Market volatility")

    ctx.cross_correlation = cross_correlation(all_dfs)
    print("  ✓ Cross correlation")

    # Combine into features DataFrame
    features = pd.concat([
        ctx.btc_dominance,
        ctx.altcoin_momentum,
        ctx.eth_btc_ratio,
        ctx.cross_oi_change,
        ctx.aggregate_funding,
        ctx.risk_on_off,
        ctx.market_volatility,
        ctx.cross_correlation,
    ], axis=1)

    ctx.features = features

    print(f"  Generated {len(features.columns)} market context features")

    return ctx


def add_market_context_to_asset(
    asset_df: pd.DataFrame,
    market_ctx: MarketContext,
    prefix: str = "mkt_",
) -> pd.DataFrame:
    """
    Add market context features to an individual asset DataFrame.

    Args:
        asset_df: Single asset DataFrame
        market_ctx: MarketContext from generate_market_context()
        prefix: Prefix for market context columns

    Returns:
        DataFrame with market context columns added
    """
    df = asset_df.copy()

    for col in market_ctx.features.columns:
        series = market_ctx.features[col].reindex(df.index)
        df[f"{prefix}{col}"] = series

    return df
