"""
Integration tests for volatility arbitrage adapter.

Tests end-to-end integration of vol arb strategy through the unified system:
- Adapter instantiation
- Data conversion fidelity
- Signal generation
- Position tracking
"""

import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

# Add vol arb to path
VOL_ARB_PATH = (
    Path(__file__).parent.parent / "strategies" / "volatility-arbitrage" / "src"
)
sys.path.insert(0, str(VOL_ARB_PATH))

from volatility_arbitrage.strategy.base import BuyAndHoldStrategy
from volatility_arbitrage.strategy.volatility_arbitrage import (
    VolatilityArbitrageConfig,
    VolatilityArbitrageStrategy,
)

from paper_trading.adapters.strategy_adapter import VolatilityArbitrageAdapter
from paper_trading.core.types import Signal, StrategyType, UnifiedPosition


class TestVolArbAdapterInstantiation:
    """Test that we can create and wrap the vol arb strategy."""

    def test_wrap_buy_and_hold_strategy(self) -> None:
        """Test wrapping simple buy-and-hold strategy as baseline."""
        # Use the simple BuyAndHoldStrategy from vol arb
        base_strategy = BuyAndHoldStrategy(symbol="SPY", quantity=100)

        # Wrap it with adapter
        adapter = VolatilityArbitrageAdapter(base_strategy)

        assert adapter.wrapped_strategy == base_strategy
        assert adapter.strategy_type == StrategyType.VOLATILITY_ARBITRAGE
        assert adapter.get_required_symbols() == ["SPY"]

    def test_wrap_vol_arb_strategy(self) -> None:
        """Test wrapping actual volatility arbitrage strategy."""
        # Create vol arb config with minimal settings
        config = VolatilityArbitrageConfig(
            entry_threshold_pct=Decimal("5.0"),
            exit_threshold_pct=Decimal("2.0"),
            min_days_to_expiry=14,
            max_days_to_expiry=60,
            position_size_pct=Decimal("5.0"),
            use_regime_detection=False,  # Disable for simplicity
        )

        # Create strategy
        vol_strategy = VolatilityArbitrageStrategy(config=config)

        # Wrap with adapter
        adapter = VolatilityArbitrageAdapter(vol_strategy)

        assert adapter.wrapped_strategy == vol_strategy
        assert adapter.strategy_type == StrategyType.VOLATILITY_ARBITRAGE


class TestVolArbDataConversion:
    """Test data format conversions for vol arb integration."""

    def test_generate_signals_with_simple_data(self) -> None:
        """Test signal generation with minimal market data."""
        # Create simple buy-and-hold strategy
        base_strategy = BuyAndHoldStrategy(symbol="SPY", quantity=100)
        adapter = VolatilityArbitrageAdapter(base_strategy)

        # Create sample market data
        df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, 9, 30)],
            'symbol': ['SPY'],
            'open': [450.0],
            'high': [451.0],
            'low': [449.0],
            'close': [450.5],
            'volume': [1000000]
        })

        # Generate signals (no positions)
        signals = adapter.generate_signals(
            timestamp=datetime(2024, 1, 1, 9, 30),
            market_data=df,
            positions={}
        )

        # Buy-and-hold should generate one buy signal on first call
        assert len(signals) == 1
        assert signals[0].symbol == "SPY"
        assert signals[0].action == "buy"
        assert signals[0].quantity == 100
        assert signals[0].strategy_type == StrategyType.VOLATILITY_ARBITRAGE

    def test_signals_respect_existing_positions(self) -> None:
        """Test that strategy respects existing positions."""
        base_strategy = BuyAndHoldStrategy(symbol="SPY", quantity=100)
        adapter = VolatilityArbitrageAdapter(base_strategy)

        # Create market data
        df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, 9, 30)],
            'symbol': ['SPY'],
            'open': [450.0],
            'high': [451.0],
            'low': [449.0],
            'close': [450.5],
            'volume': [1000000]
        })

        # Create existing position
        existing_position = UnifiedPosition(
            symbol="SPY",
            quantity=100,
            avg_entry_price=Decimal("450.0"),
            current_price=Decimal("450.5"),
            strategy_type=StrategyType.VOLATILITY_ARBITRAGE,
            last_update=datetime(2024, 1, 1, 9, 30)
        )

        # Generate signals with existing position
        signals = adapter.generate_signals(
            timestamp=datetime(2024, 1, 1, 9, 30),
            market_data=df,
            positions={"SPY": existing_position}
        )

        # Should not generate signals if position exists
        assert len(signals) == 0

    def test_signal_format_conversion(self) -> None:
        """Test that vol arb signals are correctly converted to unified format."""
        base_strategy = BuyAndHoldStrategy(symbol="QQQ", quantity=50)
        adapter = VolatilityArbitrageAdapter(base_strategy)

        df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1)],
            'symbol': ['QQQ'],
            'close': [380.0],
            'volume': [500000]
        })

        signals = adapter.generate_signals(
            timestamp=datetime(2024, 1, 1),
            market_data=df,
            positions={}
        )

        # Verify signal structure
        assert len(signals) == 1
        signal = signals[0]

        # Check all required fields
        assert isinstance(signal, Signal)
        assert signal.symbol == "QQQ"
        assert signal.action in ["buy", "sell"]
        assert signal.quantity > 0
        assert isinstance(signal.reason, str)
        assert signal.strategy_type == StrategyType.VOLATILITY_ARBITRAGE
        assert isinstance(signal.timestamp, datetime)


class TestVolArbPositionTracking:
    """Test position tracking through adapter."""

    def test_position_conversion_roundtrip(self) -> None:
        """Test that positions convert correctly between formats."""
        base_strategy = BuyAndHoldStrategy(symbol="SPY", quantity=100)
        adapter = VolatilityArbitrageAdapter(base_strategy)

        # Create unified position
        unified_pos = UnifiedPosition(
            symbol="SPY",
            quantity=100,
            avg_entry_price=Decimal("450.0"),
            current_price=Decimal("455.0"),
            strategy_type=StrategyType.VOLATILITY_ARBITRAGE,
            last_update=datetime(2024, 1, 1)
        )

        # Convert to vol arb format (internal method test)
        vol_positions = adapter._convert_positions_to_vol_arb({"SPY": unified_pos})

        # Should have one position
        assert len(vol_positions) == 1
        assert "SPY" in vol_positions

        # Verify converted position
        vol_pos = vol_positions["SPY"]
        assert vol_pos.symbol == "SPY"
        assert vol_pos.quantity == 100
        assert vol_pos.avg_entry_price == Decimal("450.0")
        assert vol_pos.current_price == Decimal("455.0")

    def test_position_pnl_preserved(self) -> None:
        """Test that P&L calculations are preserved through conversion."""
        unified_pos = UnifiedPosition(
            symbol="SPY",
            quantity=100,
            avg_entry_price=Decimal("450.0"),
            current_price=Decimal("455.0"),
            strategy_type=StrategyType.VOLATILITY_ARBITRAGE,
            last_update=datetime(2024, 1, 1)
        )

        # Check unified P&L
        assert unified_pos.unrealized_pnl == Decimal("500.0")  # (455-450)*100

        # Convert to vol arb format
        adapter = VolatilityArbitrageAdapter(
            BuyAndHoldStrategy(symbol="SPY", quantity=100)
        )
        vol_positions = adapter._convert_positions_to_vol_arb({"SPY": unified_pos})
        vol_pos = vol_positions["SPY"]

        # Check vol arb P&L matches
        assert vol_pos.unrealized_pnl == Decimal("500.0")

    def test_filters_positions_by_strategy_type(self) -> None:
        """Test that adapter only converts positions from its strategy."""
        adapter = VolatilityArbitrageAdapter(
            BuyAndHoldStrategy(symbol="SPY", quantity=100)
        )

        # Create positions from different strategies
        vol_arb_pos = UnifiedPosition(
            symbol="SPY",
            quantity=100,
            avg_entry_price=Decimal("450.0"),
            current_price=Decimal("455.0"),
            strategy_type=StrategyType.VOLATILITY_ARBITRAGE,
            last_update=datetime(2024, 1, 1)
        )

        signal_mapping_pos = UnifiedPosition(
            symbol="QQQ",
            quantity=50,
            avg_entry_price=Decimal("380.0"),
            current_price=Decimal("385.0"),
            strategy_type=StrategyType.SIGNAL_MAPPING,
            last_update=datetime(2024, 1, 1)
        )

        # Convert positions
        vol_positions = adapter._convert_positions_to_vol_arb({
            "SPY": vol_arb_pos,
            "QQQ": signal_mapping_pos
        })

        # Should only convert vol arb positions
        assert len(vol_positions) == 1
        assert "SPY" in vol_positions
        assert "QQQ" not in vol_positions


class TestVolArbCallbacks:
    """Test lifecycle callbacks."""

    def test_on_trade_executed_callback(self) -> None:
        """Test trade execution callback."""
        base_strategy = BuyAndHoldStrategy(symbol="SPY", quantity=100)
        adapter = VolatilityArbitrageAdapter(base_strategy)

        # Create a signal
        signal = Signal(
            symbol="SPY",
            action="buy",
            quantity=100,
            reason="Test trade",
            strategy_type=StrategyType.VOLATILITY_ARBITRAGE,
            timestamp=datetime(2024, 1, 1)
        )

        # Call callback (should not raise error)
        adapter.on_trade_executed(
            timestamp=datetime(2024, 1, 1, 9, 30),
            signal=signal,
            executed_price=450.5
        )

        # No assertion needed - just verify it doesn't crash

    def test_on_backtest_start_callback(self) -> None:
        """Test backtest start callback."""
        base_strategy = BuyAndHoldStrategy(symbol="SPY", quantity=100)
        adapter = VolatilityArbitrageAdapter(base_strategy)

        # Call callback
        adapter.on_backtest_start(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31)
        )

        # No assertion needed - just verify it doesn't crash

    def test_on_backtest_end_callback(self) -> None:
        """Test backtest end callback."""
        base_strategy = BuyAndHoldStrategy(symbol="SPY", quantity=100)
        adapter = VolatilityArbitrageAdapter(base_strategy)

        # Call callback
        adapter.on_backtest_end()

        # No assertion needed - just verify it doesn't crash


class TestVolArbEndToEnd:
    """End-to-end smoke tests."""

    def test_multi_day_simulation(self) -> None:
        """Test adapter with multiple days of data."""
        base_strategy = BuyAndHoldStrategy(symbol="SPY", quantity=100)
        adapter = VolatilityArbitrageAdapter(base_strategy)

        # Create 5 days of market data
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'symbol': ['SPY'] * 5,
            'open': [449.0, 450.0, 451.0, 452.0, 453.0],
            'high': [451.0, 452.0, 453.0, 454.0, 455.0],
            'low': [448.0, 449.0, 450.0, 451.0, 452.0],
            'close': [450.0, 451.0, 452.0, 453.0, 454.0],
            'volume': [1000000] * 5
        })

        positions = {}
        all_signals = []

        # Simulate 5 days
        for date in dates:
            daily_data = df[df['timestamp'] == date]

            signals = adapter.generate_signals(
                timestamp=date,
                market_data=daily_data,
                positions=positions
            )

            all_signals.extend(signals)

            # Simulate position creation from first signal
            if signals and not positions:
                positions["SPY"] = UnifiedPosition(
                    symbol="SPY",
                    quantity=signals[0].quantity,
                    avg_entry_price=Decimal("450.0"),
                    current_price=Decimal(str(daily_data.iloc[0]['close'])),
                    strategy_type=StrategyType.VOLATILITY_ARBITRAGE,
                    last_update=date
                )

        # Should generate one buy signal on first day
        assert len(all_signals) == 1
        assert all_signals[0].action == "buy"
        assert all_signals[0].symbol == "SPY"

    def test_adapter_with_empty_market_data(self) -> None:
        """Test that adapter handles empty DataFrames gracefully."""
        base_strategy = BuyAndHoldStrategy(symbol="SPY", quantity=100)
        adapter = VolatilityArbitrageAdapter(base_strategy)

        # Empty DataFrame
        df = pd.DataFrame(columns=['timestamp', 'symbol', 'close', 'volume'])

        # Should return empty signals, not crash
        signals = adapter.generate_signals(
            timestamp=datetime(2024, 1, 1),
            market_data=df,
            positions={}
        )

        assert isinstance(signals, list)
