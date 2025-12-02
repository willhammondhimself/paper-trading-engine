"""
Abstract interfaces for the unified trading system.

Defines contracts that all strategies and brokers must implement,
enabling seamless integration of different strategy types.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd

from paper_trading.core.types import Signal, UnifiedPosition


class StrategyInterface(ABC):
    """
    Universal strategy interface for all trading strategies.

    This abstract base class defines the contract that all strategy wrappers
    must implement. It's designed to work with both:
    - Pydantic-based strategies (volatility arbitrage)
    - DataFrame-based strategies (signal mapping, options sentiment)

    The interface is intentionally minimal to accommodate different
    strategy paradigms while providing a consistent API for the backtest engine.
    """

    @abstractmethod
    def generate_signals(
        self,
        timestamp: datetime,
        market_data: pd.DataFrame,
        positions: dict[str, UnifiedPosition],
    ) -> list[Signal]:
        """
        Generate trading signals for the current time step.

        This is the core strategy method called by the backtest engine
        on each time step. The strategy examines market data and current
        positions to produce trading signals.

        Args:
            timestamp: Current timestamp in the backtest
            market_data: Market data DataFrame with columns:
                - symbol: str
                - timestamp: datetime
                - open, high, low, close: float
                - volume: int
                Additional columns may be present depending on data source
            positions: Dictionary mapping symbol -> UnifiedPosition
                for all currently open positions

        Returns:
            List of Signal objects to execute.
            Empty list if no signals generated.

        Example:
            >>> def generate_signals(self, timestamp, market_data, positions):
            ...     signals = []
            ...     spy_data = market_data[market_data['symbol'] == 'SPY']
            ...     current_price = spy_data.iloc[-1]['close']
            ...
            ...     # Example: Buy if we don't have a position
            ...     if 'SPY' not in positions:
            ...         signals.append(Signal(
            ...             symbol='SPY',
            ...             action='buy',
            ...             quantity=100,
            ...             reason='Entry signal',
            ...             strategy_type=self.strategy_type,
            ...             timestamp=timestamp
            ...         ))
            ...     return signals
        """
        pass

    @abstractmethod
    def get_required_symbols(self) -> list[str]:
        """
        Return list of symbols required by this strategy.

        The backtest engine uses this to determine what data to fetch.
        For example, a volatility arbitrage strategy might return ['SPY']
        to indicate it needs SPY data and options.

        Returns:
            List of ticker symbols (e.g., ['SPY', 'QQQ'])

        Example:
            >>> def get_required_symbols(self):
            ...     return ['SPY']
        """
        pass

    def on_trade_executed(
        self, timestamp: datetime, signal: Signal, executed_price: float
    ) -> None:
        """
        Optional callback when a trade is executed.

        Strategies can override this to track their execution history,
        update internal state, or log trades. The default implementation
        does nothing.

        Args:
            timestamp: When the trade was executed
            signal: Original signal that triggered the trade
            executed_price: Actual execution price (may differ from signal due to slippage)

        Example:
            >>> def on_trade_executed(self, timestamp, signal, executed_price):
            ...     self.trade_history.append({
            ...         'timestamp': timestamp,
            ...         'signal': signal,
            ...         'price': executed_price
            ...     })
        """
        pass

    def on_backtest_start(self, start_date: datetime, end_date: datetime) -> None:
        """
        Optional callback when backtest begins.

        Use this to initialize strategy state, load models, or
        prepare data structures. The default implementation does nothing.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date

        Example:
            >>> def on_backtest_start(self, start_date, end_date):
            ...     self.trade_count = 0
            ...     self.entry_prices = {}
            ...     print(f"Starting backtest from {start_date} to {end_date}")
        """
        pass

    def on_backtest_end(self) -> None:
        """
        Optional callback when backtest completes.

        Use this for cleanup, final logging, or generating
        strategy-specific reports. The default implementation does nothing.

        Example:
            >>> def on_backtest_end(self):
            ...     print(f"Backtest complete. Total trades: {self.trade_count}")
            ...     self.generate_report()
        """
        pass


class BrokerInterface(ABC):
    """
    Abstract broker interface for order execution.

    Defines the contract for broker implementations (paper trading,
    live trading, etc.). This will be implemented in Phase 2 for
    Alpaca integration.

    For MVP, the UnifiedBacktestEngine handles execution directly
    without using a broker abstraction.
    """

    @abstractmethod
    def submit_order(
        self, symbol: str, action: str, quantity: int, order_type: str = "market"
    ) -> str:
        """
        Submit a trading order.

        Args:
            symbol: Instrument symbol
            action: "buy" or "sell"
            quantity: Number of shares/contracts
            order_type: Order type (default: "market")

        Returns:
            Order ID for tracking

        Raises:
            BrokerError: If order submission fails
        """
        pass

    @abstractmethod
    def get_account_value(self) -> float:
        """
        Get current account value (cash + positions).

        Returns:
            Total account value in dollars
        """
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[dict]:
        """
        Get current position for a symbol.

        Args:
            symbol: Instrument symbol

        Returns:
            Position dict or None if no position exists
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation succeeded, False otherwise
        """
        pass
