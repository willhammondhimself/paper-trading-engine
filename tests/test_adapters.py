"""
Unit tests for data adapters.

Tests conversion between Pydantic models and DataFrames,
ensuring type safety and data integrity across formats.
"""

import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

# Add vol arb to path
VOL_ARB_PATH = (
    Path(__file__).parent.parent / "strategies" / "volatility-arbitrage" / "src"
)
sys.path.insert(0, str(VOL_ARB_PATH))

from volatility_arbitrage.core.types import OptionType, Position, TickData

from paper_trading.adapters.data_adapter import DataAdapter
from paper_trading.core.types import StrategyType, UnifiedPosition


class TestTickDataConversions:
    """Test TickData ↔ DataFrame conversions."""

    def test_dataframe_to_tick_data_basic(self) -> None:
        """Test basic DataFrame → TickData conversion."""
        df = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30)],
                "close": [450.0],
                "volume": [1000000],
            }
        )

        ticks = DataAdapter.dataframe_to_tick_data(df, "SPY")

        assert len(ticks) == 1
        assert ticks[0].symbol == "SPY"
        assert ticks[0].price == Decimal("450.0")
        assert ticks[0].volume == 1000000
        assert ticks[0].timestamp == datetime(2024, 1, 1, 9, 30)

    def test_dataframe_to_tick_data_with_bid_ask(self) -> None:
        """Test DataFrame → TickData with bid/ask."""
        df = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30)],
                "close": [450.0],
                "volume": [1000000],
                "bid": [449.95],
                "ask": [450.05],
            }
        )

        ticks = DataAdapter.dataframe_to_tick_data(df, "SPY")

        assert ticks[0].bid == Decimal("449.95")
        assert ticks[0].ask == Decimal("450.05")
        assert ticks[0].mid_price == Decimal("450.0")

    def test_dataframe_to_tick_data_multiple_rows(self) -> None:
        """Test converting multiple DataFrame rows."""
        df = pd.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 9, 30),
                    datetime(2024, 1, 1, 10, 30),
                    datetime(2024, 1, 1, 11, 30),
                ],
                "close": [450.0, 451.0, 452.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        ticks = DataAdapter.dataframe_to_tick_data(df, "SPY")

        assert len(ticks) == 3
        assert ticks[0].price == Decimal("450.0")
        assert ticks[1].price == Decimal("451.0")
        assert ticks[2].price == Decimal("452.0")

    def test_tick_data_to_dataframe_basic(self) -> None:
        """Test TickData → DataFrame conversion."""
        tick = TickData(
            timestamp=datetime(2024, 1, 1, 9, 30),
            symbol="SPY",
            price=Decimal("450.0"),
            volume=1000000,
        )

        df = DataAdapter.tick_data_to_dataframe([tick])

        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "SPY"
        assert df.iloc[0]["price"] == 450.0
        assert df.iloc[0]["volume"] == 1000000
        assert df.iloc[0]["timestamp"] == datetime(2024, 1, 1, 9, 30)

    def test_tick_data_to_dataframe_with_bid_ask(self) -> None:
        """Test TickData → DataFrame with bid/ask."""
        tick = TickData(
            timestamp=datetime(2024, 1, 1, 9, 30),
            symbol="SPY",
            price=Decimal("450.0"),
            volume=1000000,
            bid=Decimal("449.95"),
            ask=Decimal("450.05"),
        )

        df = DataAdapter.tick_data_to_dataframe([tick])

        assert df.iloc[0]["bid"] == 449.95
        assert df.iloc[0]["ask"] == 450.05

    def test_roundtrip_conversion(self) -> None:
        """Test DataFrame → TickData → DataFrame roundtrip."""
        original_df = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30)],
                "close": [450.0],
                "volume": [1000000],
                "bid": [449.95],
                "ask": [450.05],
            }
        )

        # Convert to TickData and back
        ticks = DataAdapter.dataframe_to_tick_data(original_df, "SPY")
        result_df = DataAdapter.tick_data_to_dataframe(ticks)

        # Check values match (allowing for column name difference: close → price)
        assert result_df.iloc[0]["price"] == 450.0
        assert result_df.iloc[0]["volume"] == 1000000
        assert result_df.iloc[0]["bid"] == 449.95
        assert result_df.iloc[0]["ask"] == 450.05


class TestPositionConversions:
    """Test UnifiedPosition ↔ vol arb Position conversions."""

    def test_unified_to_vol_arb_position(self) -> None:
        """Test UnifiedPosition → vol arb Position."""
        unified = UnifiedPosition(
            symbol="SPY",
            quantity=100,
            avg_entry_price=Decimal("450.0"),
            current_price=Decimal("455.0"),
            strategy_type=StrategyType.VOLATILITY_ARBITRAGE,
            last_update=datetime(2024, 1, 1),
        )

        vol_pos = DataAdapter.unified_to_vol_arb_position(unified)

        assert vol_pos.symbol == "SPY"
        assert vol_pos.quantity == 100
        assert vol_pos.avg_entry_price == Decimal("450.0")
        assert vol_pos.current_price == Decimal("455.0")
        assert vol_pos.last_update == datetime(2024, 1, 1)

    def test_vol_arb_to_unified_position(self) -> None:
        """Test vol arb Position → UnifiedPosition."""
        vol_pos = Position(
            symbol="SPY",
            quantity=100,
            avg_entry_price=Decimal("450.0"),
            current_price=Decimal("455.0"),
            last_update=datetime(2024, 1, 1),
        )

        unified = DataAdapter.vol_arb_to_unified_position(
            vol_pos, StrategyType.VOLATILITY_ARBITRAGE
        )

        assert unified.symbol == "SPY"
        assert unified.quantity == 100
        assert unified.avg_entry_price == Decimal("450.0")
        assert unified.current_price == Decimal("455.0")
        assert unified.strategy_type == StrategyType.VOLATILITY_ARBITRAGE
        assert unified.last_update == datetime(2024, 1, 1)

    def test_position_roundtrip_conversion(self) -> None:
        """Test UnifiedPosition → vol arb → UnifiedPosition roundtrip."""
        original = UnifiedPosition(
            symbol="QQQ",
            quantity=50,
            avg_entry_price=Decimal("380.0"),
            current_price=Decimal("385.0"),
            strategy_type=StrategyType.VOLATILITY_ARBITRAGE,
            last_update=datetime(2024, 1, 1, 15, 0),
        )

        # Convert to vol arb and back
        vol_pos = DataAdapter.unified_to_vol_arb_position(original)
        result = DataAdapter.vol_arb_to_unified_position(
            vol_pos, StrategyType.VOLATILITY_ARBITRAGE
        )

        # Check all fields match
        assert result.symbol == original.symbol
        assert result.quantity == original.quantity
        assert result.avg_entry_price == original.avg_entry_price
        assert result.current_price == original.current_price
        assert result.strategy_type == original.strategy_type
        assert result.last_update == original.last_update

    def test_position_preserves_pnl_calculations(self) -> None:
        """Test that P&L calculations work after conversion."""
        unified = UnifiedPosition(
            symbol="SPY",
            quantity=100,
            avg_entry_price=Decimal("450.0"),
            current_price=Decimal("455.0"),
            strategy_type=StrategyType.VOLATILITY_ARBITRAGE,
            last_update=datetime(2024, 1, 1),
        )

        # Convert and check vol arb position calculations
        vol_pos = DataAdapter.unified_to_vol_arb_position(unified)
        assert vol_pos.unrealized_pnl == Decimal("500.0")  # (455-450) * 100

        # Convert back and check unified calculations
        back_to_unified = DataAdapter.vol_arb_to_unified_position(
            vol_pos, StrategyType.VOLATILITY_ARBITRAGE
        )
        assert back_to_unified.unrealized_pnl == Decimal("500.0")
        assert back_to_unified.unrealized_pnl_pct == Decimal("1.111111111111111111111111111")  # 500/45000 * 100


class TestOptionContractCreation:
    """Test OptionContract creation from dictionaries."""

    def test_create_option_contract_call(self) -> None:
        """Test creating a call option contract."""
        data = {
            "symbol": "SPY",
            "option_type": "call",
            "strike": 450.0,
            "expiry": datetime(2024, 2, 1),
            "price": 5.50,
        }

        contract = DataAdapter.create_option_contract_from_dict(data)

        assert contract.symbol == "SPY"
        assert contract.option_type == OptionType.CALL
        assert contract.strike == Decimal("450.0")
        assert contract.expiry == datetime(2024, 2, 1)
        assert contract.price == Decimal("5.50")

    def test_create_option_contract_put(self) -> None:
        """Test creating a put option contract."""
        data = {
            "symbol": "SPY",
            "option_type": "put",
            "strike": 450.0,
            "expiry": datetime(2024, 2, 1),
            "price": 4.25,
        }

        contract = DataAdapter.create_option_contract_from_dict(data)

        assert contract.option_type == OptionType.PUT
        assert contract.price == Decimal("4.25")

    def test_create_option_contract_with_greeks(self) -> None:
        """Test creating option contract with full data."""
        data = {
            "symbol": "SPY",
            "option_type": "call",
            "strike": 450.0,
            "expiry": datetime(2024, 2, 1),
            "price": 5.50,
            "bid": 5.45,
            "ask": 5.55,
            "volume": 1000,
            "open_interest": 5000,
            "implied_volatility": 0.20,
        }

        contract = DataAdapter.create_option_contract_from_dict(data)

        assert contract.bid == Decimal("5.45")
        assert contract.ask == Decimal("5.55")
        assert contract.volume == 1000
        assert contract.open_interest == 5000
        assert contract.implied_volatility == Decimal("0.20")


class TestOptionChainCreation:
    """Test OptionChain creation from DataFrame."""

    def test_create_option_chain_basic(self) -> None:
        """Test creating option chain from DataFrame."""
        df = pd.DataFrame(
            {
                "option_type": ["call", "call", "put", "put"],
                "strike": [445.0, 450.0, 445.0, 450.0],
                "price": [8.00, 5.50, 3.50, 4.25],
                "expiry": [datetime(2024, 2, 1)] * 4,
            }
        )

        chain = DataAdapter.create_option_chain_from_dataframe(
            df=df,
            symbol="SPY",
            timestamp=datetime(2024, 1, 1),
            expiry=datetime(2024, 2, 1),
            underlying_price=Decimal("450.0"),
        )

        assert chain.symbol == "SPY"
        assert chain.underlying_price == Decimal("450.0")
        assert len(chain.calls) == 2
        assert len(chain.puts) == 2
        assert chain.timestamp == datetime(2024, 1, 1)
        assert chain.expiry == datetime(2024, 2, 1)

    def test_option_chain_separates_calls_puts(self) -> None:
        """Test that calls and puts are properly separated."""
        df = pd.DataFrame(
            {
                "option_type": ["call", "put"],
                "strike": [450.0, 450.0],
                "price": [5.50, 4.25],
                "expiry": [datetime(2024, 2, 1), datetime(2024, 2, 1)],
            }
        )

        chain = DataAdapter.create_option_chain_from_dataframe(
            df=df,
            symbol="SPY",
            timestamp=datetime(2024, 1, 1),
            expiry=datetime(2024, 2, 1),
            underlying_price=Decimal("450.0"),
        )

        # Verify calls
        assert len(chain.calls) == 1
        assert chain.calls[0].option_type == OptionType.CALL
        assert chain.calls[0].price == Decimal("5.50")

        # Verify puts
        assert len(chain.puts) == 1
        assert chain.puts[0].option_type == OptionType.PUT
        assert chain.puts[0].price == Decimal("4.25")

    def test_option_chain_methods(self) -> None:
        """Test that OptionChain methods work after creation."""
        df = pd.DataFrame(
            {
                "option_type": ["call", "put"],
                "strike": [450.0, 450.0],
                "price": [5.50, 4.25],
                "expiry": [datetime(2024, 2, 1), datetime(2024, 2, 1)],
            }
        )

        chain = DataAdapter.create_option_chain_from_dataframe(
            df=df,
            symbol="SPY",
            timestamp=datetime(2024, 1, 1),
            expiry=datetime(2024, 2, 1),
            underlying_price=Decimal("450.0"),
        )

        # Test time_to_expiry calculation
        tte = chain.time_to_expiry
        assert tte > 0  # Should be positive (31 days)

        # Test get_atm_strike (should return 450 since underlying is 450)
        atm_strike = chain.get_atm_strike()
        assert atm_strike == Decimal("450.0")
