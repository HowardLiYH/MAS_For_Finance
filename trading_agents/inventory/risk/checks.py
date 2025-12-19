"""Risk check methods for Risk Manager Agent (Step M-A)."""
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from ..interfaces import RiskCheckMethod
from ..registry import register


@register("risk.checks", "var_safe_band")
class VaRSafeBand(RiskCheckMethod):
    """
    Check if proposed position keeps 1-day VaR within safe band.

    Returns:
        - pass: VaR within limit
        - soft_fail: VaR breach (suggests smaller size)
        - hard_fail: Extreme VaR breach (abort)
    """
    name = "var_safe_band"

    def evaluate(self, execution: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        price_df = context["price_df"]
        returns = price_df["close"].pct_change().dropna()
        volatility = float(returns.rolling(30, min_periods=10).std().iloc[-1] or 0.02)

        var99 = 2.33 * volatility  # 99% VaR (Gaussian)
        position_value = execution["position_size"] * execution["entry_price"]
        pnl_at_var = position_value * var99 * execution.get("leverage", 1.0)

        limit = float(context.get("var_limit", 0.02)) * execution["entry_price"]
        extreme_limit = limit * 3.0

        # Hard fail: extreme breach
        if abs(pnl_at_var) > extreme_limit:
            return {
                "verdict": "hard_fail",
                "envelope": {},
                "reasons": [f"Extreme VaR breach: {abs(pnl_at_var):.2f} > {extreme_limit:.2f} - ABORT"]
            }

        # Pass: within limit
        if abs(pnl_at_var) <= limit:
            return {"verdict": "pass", "envelope": {}, "reasons": ["VaR within band"]}

        # Soft fail: suggest smaller size
        shrink = max(0.1, limit / (abs(pnl_at_var) + 1e-9))
        max_size = float(execution["position_size"] * shrink)
        return {
            "verdict": "soft_fail",
            "envelope": {"max_size": max_size},
            "reasons": ["VaR breach: suggest smaller size"]
        }


@register("risk.checks", "leverage_position_limits")
class LeveragePositionLimits(RiskCheckMethod):
    """
    Enforce hard limits on leverage and position size.

    Returns:
        - pass: Within limits
        - soft_fail: Exceeds soft limits (suggests adjustment)
        - hard_fail: Exceeds hard limits (abort)
    """
    name = "leverage_position_limits"

    def evaluate(self, execution: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        max_leverage = float(context.get("max_leverage", 5.0))
        max_position = float(context.get("max_position", 1.0))

        extreme_leverage = max_leverage * 2.0
        extreme_position = max_position * 2.0

        # Hard fail: extreme leverage
        if execution["leverage"] > extreme_leverage:
            return {
                "verdict": "hard_fail",
                "envelope": {},
                "reasons": [f"Extreme leverage {execution['leverage']:.1f}x > {extreme_leverage:.1f}x - ABORT"]
            }

        # Hard fail: extreme position
        if abs(execution["position_size"]) > extreme_position:
            return {
                "verdict": "hard_fail",
                "envelope": {},
                "reasons": [f"Extreme position {execution['position_size']:.2f} > {extreme_position:.2f} - ABORT"]
            }

        # Soft fail: moderate leverage violation
        if execution["leverage"] > max_leverage:
            return {
                "verdict": "soft_fail",
                "envelope": {"max_leverage": max_leverage},
                "reasons": [f"Leverage > {max_leverage}x"]
            }

        # Soft fail: moderate position violation
        if abs(execution["position_size"]) > max_position:
            return {
                "verdict": "soft_fail",
                "envelope": {"max_size": max_position},
                "reasons": [f"Position > {max_position} units"]
            }

        return {"verdict": "pass", "envelope": {}, "reasons": ["Size & leverage OK"]}


@register("risk.checks", "liquidation_safety")
class LiquidationSafety(RiskCheckMethod):
    """
    Check if liquidation price is too close to current price.

    Returns:
        - hard_fail: Liquidation within 1% buffer (abort)
        - soft_fail: Liquidation within 2% (suggests lower leverage)
    """
    name = "liquidation_safety"

    def evaluate(self, execution: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        price_df = context["price_df"]
        current_price = float(price_df["close"].iloc[-1])
        liquidation_price = execution.get("liquidation_price")

        if liquidation_price is None:
            return {"verdict": "pass", "envelope": {}, "reasons": ["Liquidation price not set"]}

        # Calculate distance to liquidation
        if execution["direction"] == "LONG":
            distance_pct = (current_price - liquidation_price) / current_price
        else:
            distance_pct = (liquidation_price - current_price) / current_price

        min_buffer = float(context.get("min_liquidation_buffer", 0.01))

        # Hard fail: too close
        if distance_pct < min_buffer:
            return {
                "verdict": "hard_fail",
                "envelope": {},
                "reasons": [f"Liquidation too close: {distance_pct*100:.2f}% < {min_buffer*100:.1f}% - ABORT"]
            }

        # Soft fail: warning
        warning_buffer = min_buffer * 2.0
        if distance_pct < warning_buffer:
            return {
                "verdict": "soft_fail",
                "envelope": {"min_liquidation_buffer": warning_buffer},
                "reasons": [f"Liquidation close: {distance_pct*100:.2f}% - reduce leverage"]
            }

        return {"verdict": "pass", "envelope": {}, "reasons": ["Liquidation safety OK"]}


@register("risk.checks", "margin_call_risk")
class MarginCallRisk(RiskCheckMethod):
    """
    Check if order would cause margin call.

    Returns:
        - hard_fail: Order uses > 90% of account as margin (abort)
        - soft_fail: Order uses > 70% of account (suggests smaller position)
    """
    name = "margin_call_risk"

    def evaluate(self, execution: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        account_value = float(context.get("account_value", 10000.0))
        maintenance_margin_ratio = float(context.get("maintenance_margin_ratio", 0.5))

        position_value = execution["position_size"] * execution["entry_price"]
        required_margin = position_value / execution["leverage"]
        maintenance_margin = required_margin * maintenance_margin_ratio

        max_margin_usage = float(context.get("max_margin_usage", 0.9))

        # Hard fail: margin > 90% of account
        if required_margin > account_value * max_margin_usage:
            return {
                "verdict": "hard_fail",
                "envelope": {},
                "reasons": [f"Margin {required_margin:.2f} > {max_margin_usage*100:.0f}% of account - ABORT"]
            }

        # Hard fail: maintenance margin exceeds account
        if maintenance_margin > account_value:
            return {
                "verdict": "hard_fail",
                "envelope": {},
                "reasons": [f"Maintenance margin {maintenance_margin:.2f} > account value - ABORT"]
            }

        # Soft fail: high margin usage
        warning_usage = 0.7
        if required_margin > account_value * warning_usage:
            max_safe_size = (account_value * warning_usage * execution["leverage"]) / execution["entry_price"]
            return {
                "verdict": "soft_fail",
                "envelope": {"max_size": max_safe_size},
                "reasons": [f"High margin usage: {required_margin/account_value*100:.1f}%"]
            }

        return {"verdict": "pass", "envelope": {}, "reasons": ["Margin OK"]}


@register("risk.checks", "global_var_breach")
class GlobalVaRBreach(RiskCheckMethod):
    """
    Check if order would breach global portfolio VaR limit.

    Returns:
        - hard_fail: Total VaR exceeds limit (abort)
        - soft_fail: Approaching limit (suggests smaller position)
    """
    name = "global_var_breach"

    def evaluate(self, execution: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        price_df = context["price_df"]
        returns = price_df["close"].pct_change().dropna()
        volatility = float(returns.rolling(30, min_periods=10).std().iloc[-1] or 0.02)

        position_value = execution["position_size"] * execution["entry_price"]
        order_var = position_value * 2.33 * volatility * execution["leverage"]

        existing_var = float(context.get("existing_portfolio_var", 0.0))
        account_value = float(context.get("account_value", 10000.0))
        global_limit = float(context.get("global_var_limit", account_value * 0.05))

        total_var = existing_var + abs(order_var)

        # Hard fail: exceeds limit
        if total_var > global_limit:
            return {
                "verdict": "hard_fail",
                "envelope": {},
                "reasons": [f"Global VaR {total_var:.2f} > {global_limit:.2f} - ABORT"]
            }

        # Soft fail: approaching limit
        warning_threshold = global_limit * 0.8
        if total_var > warning_threshold:
            max_safe_var = global_limit - existing_var
            max_safe_size = max_safe_var / (2.33 * volatility * execution["leverage"] * execution["entry_price"])
            return {
                "verdict": "soft_fail",
                "envelope": {"max_size": max_safe_size},
                "reasons": [f"Approaching VaR limit: {total_var/global_limit*100:.1f}%"]
            }

        return {"verdict": "pass", "envelope": {}, "reasons": ["Global VaR OK"]}
