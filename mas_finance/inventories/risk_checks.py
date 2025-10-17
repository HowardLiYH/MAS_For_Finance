
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from .interfaces import RiskCheckMethod
from .registry import register

@register("risk.checks", "var_safe_band")
class VaRSafeBand(RiskCheckMethod):
    """Check if proposed size/leverage keeps 1-day VaR within safe band (proxy from recent volatility)."""
    name = "var_safe_band"
    def evaluate(self, exec_sum: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        price_df = context["price_df"]
        ret = price_df["close"].pct_change().dropna()
        vol = float(ret.rolling(30, min_periods=10).std().iloc[-1] or 0.02)
        var99 = 2.33 * vol  # Gaussian proxy
        pnl_at_var = exec_sum["position_size"] * exec_sum["entry_price"] * var99 * exec_sum.get("leverage",1.0)
        limit = float(context.get("var_limit", 0.02)) * exec_sum["entry_price"]  # 2% of notional as proxy
        if abs(pnl_at_var) <= limit:
            return {"verdict":"pass","envelope":{},"reasons":["VaR within band"]}
        else:
            # soft-fail with envelope to shrink size
            shrink = max(0.1, limit / (abs(pnl_at_var)+1e-9))
            max_size = float(exec_sum["position_size"] * shrink)
            return {"verdict":"soft_fail","envelope":{"max_size": max_size},"reasons":["VaR breach: suggest smaller size"]}

@register("risk.checks", "leverage_position_limits")
class LeverageAndSize(RiskCheckMethod):
    """Hard limits on leverage and absolute position size."""
    name = "leverage_position_limits"
    def evaluate(self, exec_sum: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        max_lev = float(context.get("max_leverage", 5.0))
        max_pos = float(context.get("max_position", 1.0))
        if exec_sum["leverage"] > max_lev:
            return {"verdict":"soft_fail", "envelope":{"max_leverage":max_lev}, "reasons":[f"Leverage>{max_lev}x"]}
        if abs(exec_sum["position_size"]) > max_pos:
            return {"verdict":"soft_fail", "envelope":{"max_size":max_pos}, "reasons":[f"Position>{max_pos} units"]}
        return {"verdict":"pass","envelope":{},"reasons":["Size & leverage OK"]}
