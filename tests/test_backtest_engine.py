"""
Unit tests for BacktestEngine.

Tests the unified backtest engine with various scenarios:
- Buy-and-hold strategy
- No signals generated
- Respecting existing positions
- Insufficient funds handling
- Multi-symbol support
"""

import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

# Add vol arb to path
VOL_ARB_PATH = Path(__file__).parent.parent / "strategies" / "volatility-arbitrage" / "src"
sys.path.insert(0, str(VOL_ARB_PATH))

from volatility_arbitrage.strategy.base import BuyAndHoldStrategy

from paper_trading.adapters.strategy_adapter import VolatilityArbitrageAdapter
from paper_trading.backtest.engine import BacktestEngine
from paper_trading.core.interfaces import StrategyInterface
from paper_trading.core.types import Signal, StrategyType


class TestBacktestEngineSingleStrategy:
    """Test backtest engine with buy-and-hold strategy."""

    def test_backtest_engine_single_buy_and_hold(self) -> None:
        """Test basic buy-and-hold execution with rising prices."""
        # Create strategy
        base_strategy = BuyAndHoldStrategy(symbol="SPY", quantity=100)
        adapter = VolatilityArbitrageAdapter(base_strategy)

        # Market data: 5 days, rising prices (450 -> 454)
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        market_data = pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": ["SPY"] * 5,
                "open": [449.0, 450.0, 451.0, 452.0, 453.0],
                "high": [451.0, 452.0, 453.0, 454.0, 455.0],
                "low": [448.0, 449.0, 450.0, 451.0, 452.0],
                "close": [450.0, 451.0, 452.0, 453.0, 454.0],
                "volume": [1000000] * 5,
            }
        )

        # Run backtest
        engine = BacktestEngine(strategy=adapter, market_data=market_data, initial_cash=100_000.0)
        result = engine.run()

        # Verify trade execution
        assert result.num_trades == 1
        assert result.trades[0].action == "buy"
        assert result.trades[0].symbol == "SPY"
        assert result.trades[0].quantity == 100
        assert result.trades[0].price == Decimal("450.0")

        # Verify final position
        assert len(result.equity_curve) == 5
        # Note: final_positions not exposed in BacktestResult - check via engine state
        assert engine.positions["SPY"].quantity == 100

        # Verify returns
        # Cost: 100 * 450 = 45,000
        # Final value: 100 * 454 = 45,400
        # Return: 400
        assert result.final_capital > result.initial_capital
        assert result.total_return == Decimal("400.0")

    def test_backtest_engine_tracks_equity_curve(self) -> None:
        """Test that equity curve tracks position value correctly."""
        base_strategy = BuyAndHoldStrategy(symbol="SPY", quantity=100)
        adapter = VolatilityArbitrageAdapter(base_strategy)

        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        market_data = pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": ["SPY"] * 3,
                "close": [450.0, 455.0, 460.0],
                "volume": [1000000] * 3,
            }
        )

        engine = BacktestEngine(strategy=adapter, market_data=market_data, initial_cash=100_000.0)
        result = engine.run()

        equity_curve = result.equity_curve

        # Day 1: Buy 100 @ 450 = 45,000 spent
        # Cash: 55,000, Position: 45,000, Total: 100,000
        assert equity_curve.iloc[0]["total_equity"] == 100_000.0

        # Day 2: Position worth 100 * 455 = 45,500
        # Cash: 55,000, Position: 45,500, Total: 100,500
        assert equity_curve.iloc[1]["total_equity"] == 100_500.0

        # Day 3: Position worth 100 * 460 = 46,000
        # Cash: 55,000, Position: 46,000, Total: 101,000
        assert equity_curve.iloc[2]["total_equity"] == 101_000.0


class TestBacktestEngineNoSignals:
    """Test backtest engine when no signals are generated."""

    def test_backtest_engine_no_signals(self) -> None:
        """Test that engine handles strategies that don't generate signals."""

        # Dummy strategy that never signals
        class NoSignalStrategy(StrategyInterface):
            def generate_signals(self, timestamp, market_data, positions):
                return []

            def get_required_symbols(self):
                return ["SPY"]

        strategy = NoSignalStrategy()
        strategy.strategy_type = StrategyType.VOLATILITY_ARBITRAGE  # Add for result

        market_data = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "symbol": ["SPY", "SPY"],
                "close": [450.0, 451.0],
                "volume": [1000000, 1000000],
            }
        )

        engine = BacktestEngine(strategy=strategy, market_data=market_data, initial_cash=100_000.0)
        result = engine.run()

        # Verify no trades
        assert result.num_trades == 0
        assert len(result.trades) == 0

        # Verify capital unchanged
        assert result.final_capital == result.initial_capital
        assert result.total_return == 0

        # Equity curve should still be tracked
        assert len(result.equity_curve) == 2
        assert all(result.equity_curve["total_equity"] == 100_000.0)


class TestBacktestEnginePositionRespect:
    """Test that engine respects existing positions."""

    def test_backtest_engine_respects_existing_positions(self) -> None:
        """Test that strategy sees positions and doesn't re-enter."""
        base_strategy = BuyAndHoldStrategy(symbol="SPY", quantity=100)
        adapter = VolatilityArbitrageAdapter(base_strategy)

        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        market_data = pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": ["SPY"] * 3,
                "close": [450.0, 451.0, 452.0],
                "volume": [1000000] * 3,
            }
        )

        engine = BacktestEngine(strategy=adapter, market_data=market_data, initial_cash=100_000.0)
        result = engine.run()

        # Should buy on day 1, then no more signals
        assert result.num_trades == 1

        # Check equity curve shows position holding
        equity_df = result.equity_curve
        assert len(equity_df) == 3

        # Equity increases with price
        assert equity_df.iloc[0]["total_equity"] == 100_000.0  # Buy day
        assert equity_df.iloc[1]["total_equity"] == 100_100.0  # +100
        assert equity_df.iloc[2]["total_equity"] == 100_200.0  # +200


class TestBacktestEngineInsufficientFunds:
    """Test handling of insufficient funds."""

    def test_backtest_engine_insufficient_funds(self) -> None:
        """Test that trades are skipped when insufficient cash."""
        # Try to buy $450k worth with only $100k
        base_strategy = BuyAndHoldStrategy(symbol="SPY", quantity=1000)
        adapter = VolatilityArbitrageAdapter(base_strategy)

        market_data = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "symbol": ["SPY"],
                "close": [450.0],
                "volume": [1000000],
            }
        )

        engine = BacktestEngine(strategy=adapter, market_data=market_data, initial_cash=100_000.0)
        result = engine.run()

        # Should skip the trade
        assert result.num_trades == 0
        assert result.final_capital == result.initial_capital

    def test_backtest_engine_partial_buy_allowed(self) -> None:
        """Test that smaller quantities can still be bought."""
        # Buy 200 shares at $450 = $90k (affordable)
        base_strategy = BuyAndHoldStrategy(symbol="SPY", quantity=200)
        adapter = VolatilityArbitrageAdapter(base_strategy)

        market_data = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "symbol": ["SPY"],
                "close": [450.0],
                "volume": [1000000],
            }
        )

        engine = BacktestEngine(strategy=adapter, market_data=market_data, initial_cash=100_000.0)
        result = engine.run()

        # Should execute
        assert result.num_trades == 1
        assert result.trades[0].quantity == 200


class TestBacktestEngineMultiSymbol:
    """Test multi-symbol support."""

    def test_backtest_engine_multi_symbol(self) -> None:
        """Test handling multiple symbols in market_data."""

        class MultiSymbolStrategy(StrategyInterface):
            def __init__(self):
                self.strategy_type = StrategyType.VOLATILITY_ARBITRAGE

            def generate_signals(self, timestamp, market_data, positions):
                if not positions:  # Buy both on first call
                    return [
                        Signal(
                            symbol="SPY",
                            action="buy",
                            quantity=100,
                            reason="Entry SPY",
                            strategy_type=StrategyType.VOLATILITY_ARBITRAGE,
                            timestamp=timestamp,
                        ),
                        Signal(
                            symbol="QQQ",
                            action="buy",
                            quantity=50,
                            reason="Entry QQQ",
                            strategy_type=StrategyType.VOLATILITY_ARBITRAGE,
                            timestamp=timestamp,
                        ),
                    ]
                return []

            def get_required_symbols(self):
                return ["SPY", "QQQ"]

        dates = [datetime(2024, 1, 1)]
        market_data = pd.DataFrame(
            {
                "timestamp": dates * 2,
                "symbol": ["SPY", "QQQ"],
                "close": [450.0, 380.0],
                "volume": [1000000, 500000],
            }
        )

        engine = BacktestEngine(
            strategy=MultiSymbolStrategy(),
            market_data=market_data,
            initial_cash=100_000.0,
        )
        result = engine.run()

        # Should execute both trades
        assert result.num_trades == 2

        # Check symbols
        symbols_traded = {t.symbol for t in result.trades}
        assert symbols_traded == {"SPY", "QQQ"}

        # Check final positions
        assert len(engine.positions) == 2
        assert engine.positions["SPY"].quantity == 100
        assert engine.positions["QQQ"].quantity == 50

        # Cost: (100 * 450) + (50 * 380) = 45,000 + 19,000 = 64,000
        # Cash left: 36,000
        # Position value: 64,000
        # Total: 100,000
        assert result.final_capital == result.initial_capital


class TestBacktestEngineEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_market_data(self) -> None:
        """Test with empty DataFrame."""

        class DummyStrategy(StrategyInterface):
            def __init__(self):
                self.strategy_type = StrategyType.VOLATILITY_ARBITRAGE

            def generate_signals(self, timestamp, market_data, positions):
                return []

            def get_required_symbols(self):
                return []

        market_data = pd.DataFrame(columns=["timestamp", "symbol", "close", "volume"])

        engine = BacktestEngine(
            strategy=DummyStrategy(), market_data=market_data, initial_cash=100_000.0
        )
        result = engine.run()

        assert result.num_trades == 0
        assert result.final_capital == result.initial_capital

    def test_sell_without_position(self) -> None:
        """Test selling without holding a position."""

        class SellStrategy(StrategyInterface):
            def __init__(self):
                self.strategy_type = StrategyType.VOLATILITY_ARBITRAGE

            def generate_signals(self, timestamp, market_data, positions):
                # Try to sell without buying first
                return [
                    Signal(
                        symbol="SPY",
                        action="sell",
                        quantity=100,
                        reason="Sell signal",
                        strategy_type=StrategyType.VOLATILITY_ARBITRAGE,
                        timestamp=timestamp,
                    )
                ]

            def get_required_symbols(self):
                return ["SPY"]

        market_data = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "symbol": ["SPY"],
                "close": [450.0],
                "volume": [1000000],
            }
        )

        engine = BacktestEngine(
            strategy=SellStrategy(), market_data=market_data, initial_cash=100_000.0
        )
        result = engine.run()

        # Should skip the sell (no position to sell)
        assert result.num_trades == 0
        assert result.final_capital == result.initial_capital

    def test_max_drawdown_calculation(self) -> None:
        """Test that max drawdown is calculated correctly."""
        base_strategy = BuyAndHoldStrategy(symbol="SPY", quantity=100)
        adapter = VolatilityArbitrageAdapter(base_strategy)

        # Create drawdown scenario: rise, peak, drop
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        market_data = pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": ["SPY"] * 5,
                "close": [450.0, 460.0, 470.0, 450.0, 440.0],  # Peak at 470, drop to 440
                "volume": [1000000] * 5,
            }
        )

        engine = BacktestEngine(strategy=adapter, market_data=market_data, initial_cash=100_000.0)
        result = engine.run()

        # Peak equity: 55,000 (cash) + 47,000 (100 * 470) = 102,000
        # Trough equity: 55,000 + 44,000 (100 * 440) = 99,000
        # Drawdown: (102,000 - 99,000) / 102,000 ≈ 2.94%
        assert result.max_drawdown > Decimal("0.02")  # > 2%
        assert result.max_drawdown < Decimal("0.04")  # < 4%
