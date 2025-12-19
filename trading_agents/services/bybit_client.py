"""Bybit Testnet API client for paper trading."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
import hashlib
import hmac
import time
import json
import asyncio

# Optional imports - graceful fallback if not installed
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False


@dataclass
class OrderResponse:
    """Response from order placement."""
    order_id: str
    symbol: str
    side: str
    order_type: str
    qty: float
    price: Optional[float]
    status: str
    created_time: datetime
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """A trading position."""
    symbol: str
    side: str  # "Buy" or "Sell"
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: int
    liquidation_price: float

    @property
    def value_usd(self) -> float:
        return self.size * self.mark_price


@dataclass
class WalletBalance:
    """Wallet balance information."""
    coin: str
    equity: float
    available_balance: float
    used_margin: float
    unrealized_pnl: float


class BybitTestnetClient:
    """
    Async client for Bybit Testnet API.

    Supports:
    - Order placement (market/limit)
    - Order cancellation
    - Position queries
    - Wallet balance queries
    - WebSocket subscription for executions
    """

    # Testnet endpoints
    BASE_URL = "https://api-testnet.bybit.com"
    WS_URL = "wss://stream-testnet.bybit.com/v5/private"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        recv_window: int = 5000,
    ):
        """
        Initialize Bybit Testnet client.

        Args:
            api_key: Bybit testnet API key
            api_secret: Bybit testnet API secret
            recv_window: Request validity window in ms
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp is required for BybitTestnetClient. Install with: pip install aiohttp")

        self.api_key = api_key
        self.api_secret = api_secret
        self.recv_window = recv_window
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[Any] = None
        self._ws_callbacks: List[Callable] = []

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _generate_signature(self, params: Dict[str, Any], timestamp: int) -> str:
        """Generate HMAC signature for request."""
        param_str = str(timestamp) + self.api_key + str(self.recv_window)

        if params:
            param_str += json.dumps(params, separators=(',', ':'))

        return hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _get_headers(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Generate authenticated request headers."""
        timestamp = int(time.time() * 1000)
        signature = self._generate_signature(params, timestamp)

        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": str(timestamp),
            "X-BAPI-RECV-WINDOW": str(self.recv_window),
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make authenticated API request."""
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        headers = self._get_headers(params)

        if method == "GET":
            async with session.get(url, params=params, headers=headers) as resp:
                data = await resp.json()
        else:
            async with session.post(url, json=params, headers=headers) as resp:
                data = await resp.json()

        if data.get("retCode") != 0:
            raise Exception(f"Bybit API error: {data.get('retMsg', 'Unknown error')}")

        return data.get("result", {})

    async def place_order(
        self,
        symbol: str,
        side: str,  # "Buy" or "Sell"
        order_type: str,  # "Market" or "Limit"
        qty: float,
        price: Optional[float] = None,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
    ) -> OrderResponse:
        """
        Place an order on Bybit Testnet.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            side: "Buy" or "Sell"
            order_type: "Market" or "Limit"
            qty: Order quantity
            price: Limit price (required for limit orders)
            take_profit: Take profit price
            stop_loss: Stop loss price
            reduce_only: Whether this order reduces position only
            time_in_force: "GTC", "IOC", "FOK"

        Returns:
            OrderResponse with order details
        """
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "timeInForce": time_in_force,
        }

        if price and order_type == "Limit":
            params["price"] = str(price)

        if take_profit:
            params["takeProfit"] = str(take_profit)

        if stop_loss:
            params["stopLoss"] = str(stop_loss)

        if reduce_only:
            params["reduceOnly"] = True

        result = await self._request("POST", "/v5/order/create", params)

        return OrderResponse(
            order_id=result.get("orderId", ""),
            symbol=symbol,
            side=side,
            order_type=order_type,
            qty=qty,
            price=price,
            status=result.get("orderStatus", "Created"),
            created_time=datetime.now(timezone.utc),
            raw=result,
        )

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            symbol: Trading pair
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        params = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id,
        }

        try:
            await self._request("POST", "/v5/order/cancel", params)
            return True
        except Exception as e:
            print(f"[BybitClient] Cancel order failed: {e}")
            return False

    async def get_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Get order details."""
        params = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id,
        }

        result = await self._request("GET", "/v5/order/realtime", params)
        orders = result.get("list", [])
        return orders[0] if orders else {}

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders."""
        params = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol

        result = await self._request("GET", "/v5/order/realtime", params)
        return result.get("list", [])

    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Get current positions.

        Args:
            symbol: Optional filter by symbol

        Returns:
            List of Position objects
        """
        params = {"category": "linear", "settleCoin": "USDT"}
        if symbol:
            params["symbol"] = symbol

        result = await self._request("GET", "/v5/position/list", params)

        positions = []
        for pos in result.get("list", []):
            size = float(pos.get("size", 0))
            if size == 0:
                continue

            positions.append(Position(
                symbol=pos.get("symbol", ""),
                side=pos.get("side", ""),
                size=size,
                entry_price=float(pos.get("avgPrice", 0)),
                mark_price=float(pos.get("markPrice", 0)),
                unrealized_pnl=float(pos.get("unrealisedPnl", 0)),
                leverage=int(pos.get("leverage", 1)),
                liquidation_price=float(pos.get("liqPrice", 0)),
            ))

        return positions

    async def get_wallet_balance(self, coin: str = "USDT") -> WalletBalance:
        """
        Get wallet balance.

        Args:
            coin: Coin to query (default: USDT)

        Returns:
            WalletBalance object
        """
        params = {"accountType": "UNIFIED", "coin": coin}

        result = await self._request("GET", "/v5/account/wallet-balance", params)

        accounts = result.get("list", [])
        if not accounts:
            return WalletBalance(coin=coin, equity=0, available_balance=0, used_margin=0, unrealized_pnl=0)

        account = accounts[0]
        coins = account.get("coin", [])

        for c in coins:
            if c.get("coin") == coin:
                return WalletBalance(
                    coin=coin,
                    equity=float(c.get("equity", 0)),
                    available_balance=float(c.get("availableToWithdraw", 0)),
                    used_margin=float(c.get("totalPositionMM", 0)),
                    unrealized_pnl=float(c.get("unrealisedPnl", 0)),
                )

        return WalletBalance(coin=coin, equity=0, available_balance=0, used_margin=0, unrealized_pnl=0)

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol.

        Args:
            symbol: Trading pair
            leverage: Leverage value (1-100)

        Returns:
            True if successful
        """
        params = {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }

        try:
            await self._request("POST", "/v5/position/set-leverage", params)
            return True
        except Exception as e:
            print(f"[BybitClient] Set leverage failed: {e}")
            return False

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker for a symbol."""
        params = {"category": "linear", "symbol": symbol}
        result = await self._request("GET", "/v5/market/tickers", params)
        tickers = result.get("list", [])
        return tickers[0] if tickers else {}

    async def subscribe_executions(self, callback: Callable[[Dict], None]) -> None:
        """
        Subscribe to execution updates via WebSocket.

        Args:
            callback: Function to call with execution data
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets is required for WebSocket subscriptions. Install with: pip install websockets")

        self._ws_callbacks.append(callback)

        if self._ws is not None:
            return  # Already connected

        asyncio.create_task(self._ws_loop())

    async def _ws_loop(self) -> None:
        """WebSocket connection loop."""
        import websockets

        while True:
            try:
                async with websockets.connect(self.WS_URL) as ws:
                    self._ws = ws

                    # Authenticate
                    await self._ws_auth(ws)

                    # Subscribe to executions
                    await ws.send(json.dumps({
                        "op": "subscribe",
                        "args": ["execution"],
                    }))

                    # Listen for messages
                    async for message in ws:
                        data = json.loads(message)
                        if data.get("topic") == "execution":
                            for callback in self._ws_callbacks:
                                try:
                                    callback(data)
                                except Exception as e:
                                    print(f"[BybitClient] WS callback error: {e}")

            except Exception as e:
                print(f"[BybitClient] WebSocket error: {e}")
                await asyncio.sleep(5)  # Reconnect after delay

    async def _ws_auth(self, ws) -> None:
        """Authenticate WebSocket connection."""
        expires = int((time.time() + 10) * 1000)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            f"GET/realtime{expires}".encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        await ws.send(json.dumps({
            "op": "auth",
            "args": [self.api_key, expires, signature],
        }))

    async def close(self) -> None:
        """Close client connections."""
        if self._session and not self._session.closed:
            await self._session.close()
        if self._ws:
            await self._ws.close()
