"""
Strategy adapter for wrapping volatility arbitrage strategy.

Provides a bridge between the vol arb strategy's native interface and
the unified StrategyInterface, handling all necessary format conversions.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

# Add vol arb to path
VOL_ARB_PATH = Path(__file__).parent.parent.parent / "strategies" / "volatility-arbitrage" / "src"
if str(VOL_ARB_PATH) not in sys.path:
    sys.path.insert(0, str(VOL_ARB_PATH))

from volatility_arbitrage.strategy.base import Signal as VolArbSignal
from volatility_arbitrage.strategy.base import Strategy as VolArbStrategy

from paper_trading.adapters.data_adapter import DataAdapter
from paper_trading.core.interfaces import StrategyInterface
from paper_trading.core.types import Signal, StrategyType, UnifiedPosition


class VolatilityArbitrageAdapter(StrategyInterface):
    """
    Adapter wrapping volatility arbitrage strategy to unified interface.

    This adapter enables the vol arb strategy to work within the unified
    backtesting framework by handling all format conversions:

    - UnifiedPosition → vol arb Position (for strategy calls)
    - vol arb Signal → unified Signal (for backtest engine)
    - DataFrame → Pydantic models (for data flow)

    The wrapped strategy is completely unaware it's being adapted,
    demonstrating the adapter pattern for systems integration.

    Example:
        >>> from strategies.volatility_arbitrage.src... import VolatilityArbitrageStrategy
        >>> vol_arb_config = {...}  # Vol arb configuration
        >>> vol_strategy = VolatilityArbitrageStrategy(**vol_arb_config)
        >>>
        >>> # Wrap strategy for unified system
        >>> adapter = VolatilityArbitrageAdapter(vol_strategy)
        >>>
        >>> # Now use with unified backtest engine
        >>> signals = adapter.generate_signals(timestamp, market_data, positions)
    """

    def __init__(self, wrapped_strategy: VolArbStrategy):
        """
        Initialize adapter with volatility arbitrage strategy.

        Args:
            wrapped_strategy: Instance of vol arb Strategy to wrap
        """
        self.wrapped_strategy = wrapped_strategy
        self.strategy_type = StrategyType.VOLATILITY_ARBITRAGE

    def generate_signals(
        self,
        timestamp: datetime,
        market_data: pd.DataFrame,
        positions: dict[str, UnifiedPosition],
    ) -> List[Signal]:
        """
        Generate trading signals by calling wrapped strategy.

        Performs the following conversions:
        1. UnifiedPosition dict → vol arb Position dict
        2. Call wrapped strategy's generate_signals()
        3. vol arb Signal list → unified Signal list

        Args:
            timestamp: Current backtest timestamp
            market_data: DataFrame with OHLCV data
            positions: Current positions (unified format)

        Returns:
            List of unified Signal objects

        Note:
            The wrapped strategy expects positions in its native format,
            so we convert before calling and convert signals after.
        """
        # Convert unified positions → vol arb positions
        vol_positions = self._convert_positions_to_vol_arb(positions)

        # Call wrapped strategy (uses native vol arb types)
        vol_signals = self.wrapped_strategy.generate_signals(
            timestamp=timestamp,
            market_data=market_data,
            positions=vol_positions,
        )

        # Convert vol arb signals → unified signals
        unified_signals = [
            self._convert_signal_from_vol_arb(s, timestamp) for s in vol_signals
        ]

        return unified_signals

    def get_required_symbols(self) -> List[str]:
        """
        Return symbols required by vol arb strategy.

        For volatility arbitrage, this is typically a single underlying
        symbol (e.g., "SPY") which is also used for options data.

        Returns:
            List with the underlying symbol

        Example:
            >>> adapter.get_required_symbols()
            ['SPY']
        """
        # Vol arb strategy should have a symbol attribute
        # If your vol arb implementation differs, adjust this
        if hasattr(self.wrapped_strategy, "symbol"):
            return [self.wrapped_strategy.symbol]
        elif hasattr(self.wrapped_strategy, "underlying_symbol"):
            return [self.wrapped_strategy.underlying_symbol]
        else:
            # Fallback: inspect strategy config or raise error
            raise AttributeError(
                "Vol arb strategy must have 'symbol' or 'underlying_symbol' attribute"
            )

    def on_trade_executed(
        self, timestamp: datetime, signal: Signal, executed_price: float
    ) -> None:
        """
        Forward trade execution callback to wrapped strategy.

        If the wrapped strategy implements on_trade_executed, this calls it
        with the appropriate format conversion.

        Args:
            timestamp: Trade execution time
            signal: Unified signal that was executed
            executed_price: Actual execution price
        """
        # Convert unified signal → vol arb signal for callback
        vol_signal = VolArbSignal(
            symbol=signal.symbol,
            action=signal.action,  # type: ignore
            quantity=signal.quantity,
            reason=signal.reason,
        )

        # Call wrapped strategy's callback if it exists
        if hasattr(self.wrapped_strategy, "on_trade_executed"):
            self.wrapped_strategy.on_trade_executed(
                timestamp=timestamp,
                signal=vol_signal,
                executed_price=executed_price,
            )

    def on_backtest_start(self, start_date: datetime, end_date: datetime) -> None:
        """
        Forward backtest start callback to wrapped strategy.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
        """
        if hasattr(self.wrapped_strategy, "on_backtest_start"):
            self.wrapped_strategy.on_backtest_start(start_date, end_date)

    def on_backtest_end(self) -> None:
        """Forward backtest end callback to wrapped strategy."""
        if hasattr(self.wrapped_strategy, "on_backtest_end"):
            self.wrapped_strategy.on_backtest_end()

    def _convert_positions_to_vol_arb(
        self, unified_positions: dict[str, UnifiedPosition]
    ) -> dict[str, object]:
        """
        Convert unified positions to vol arb Position objects.

        Args:
            unified_positions: Dict mapping symbol → UnifiedPosition

        Returns:
            Dict mapping symbol → vol arb Position

        Note:
            Only converts positions from this strategy type to avoid
            mixing positions from different strategies.
        """
        vol_positions = {}

        for symbol, unified_pos in unified_positions.items():
            # Only convert positions from this strategy
            if unified_pos.strategy_type == StrategyType.VOLATILITY_ARBITRAGE:
                vol_positions[symbol] = DataAdapter.unified_to_vol_arb_position(
                    unified_pos
                )

        return vol_positions

    def _convert_signal_from_vol_arb(
        self, vol_signal: VolArbSignal, timestamp: datetime
    ) -> Signal:
        """
        Convert vol arb Signal to unified Signal.

        Args:
            vol_signal: Vol arb Signal object
            timestamp: Signal generation timestamp

        Returns:
            Unified Signal object
        """
        return Signal(
            symbol=vol_signal.symbol,
            action=vol_signal.action,
            quantity=vol_signal.quantity,
            reason=vol_signal.reason,
            strategy_type=self.strategy_type,
            timestamp=timestamp,
        )
