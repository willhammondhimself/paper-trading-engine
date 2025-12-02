"""
Data format adapters for converting between Pydantic models and DataFrames.

Bridges the gap between:
- Vol arb strategy: Uses Pydantic models (TickData, OptionChain, Position)
- Signal strategies: Use pandas DataFrames
- Unified system: Uses both depending on strategy needs

This adapter layer enables seamless integration without modifying original codebases.
"""

import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Add vol arb strategy to Python path for imports
VOL_ARB_PATH = Path(__file__).parent.parent.parent / "strategies" / "volatility-arbitrage" / "src"
if str(VOL_ARB_PATH) not in sys.path:
    sys.path.insert(0, str(VOL_ARB_PATH))

from volatility_arbitrage.core.types import (
    OptionChain,
    OptionContract,
    OptionType,
    Position,
    TickData,
    Trade,
)

from paper_trading.core.types import UnifiedPosition


class DataAdapter:
    """
    Bidirectional conversion between Pydantic models and DataFrames.

    This adapter handles format conversions needed for strategy integration:
    - DataFrame → Pydantic models (for vol arb strategy)
    - Pydantic models → DataFrame (for analysis/storage)
    - UnifiedPosition ↔ vol arb Position (for strategy calls)
    """

    @staticmethod
    def dataframe_to_tick_data(df: pd.DataFrame, symbol: str) -> List[TickData]:
        """
        Convert DataFrame rows to TickData objects.

        Args:
            df: DataFrame with columns: timestamp, close, volume
                Optional: open, high, low, bid, ask
            symbol: Ticker symbol for all ticks

        Returns:
            List of TickData objects

        Example:
            >>> df = pd.DataFrame({
            ...     'timestamp': [datetime(2024, 1, 1)],
            ...     'close': [450.0],
            ...     'volume': [1000000]
            ... })
            >>> ticks = DataAdapter.dataframe_to_tick_data(df, 'SPY')
            >>> ticks[0].price
            Decimal('450.0')
        """
        ticks = []
        for _, row in df.iterrows():
            ticks.append(
                TickData(
                    timestamp=row["timestamp"] if isinstance(row["timestamp"], datetime) else pd.to_datetime(row["timestamp"]),
                    symbol=symbol,
                    price=Decimal(str(row["close"])),
                    volume=int(row["volume"]) if "volume" in row else 0,
                    bid=Decimal(str(row["bid"])) if "bid" in row and pd.notna(row["bid"]) else None,
                    ask=Decimal(str(row["ask"])) if "ask" in row and pd.notna(row["ask"]) else None,
                )
            )
        return ticks

    @staticmethod
    def tick_data_to_dataframe(ticks: List[TickData]) -> pd.DataFrame:
        """
        Convert TickData objects to DataFrame.

        Args:
            ticks: List of TickData objects

        Returns:
            DataFrame with columns: timestamp, symbol, price, volume, bid, ask

        Example:
            >>> tick = TickData(
            ...     timestamp=datetime(2024, 1, 1),
            ...     symbol='SPY',
            ...     price=Decimal('450.0'),
            ...     volume=1000000
            ... )
            >>> df = DataAdapter.tick_data_to_dataframe([tick])
            >>> df.iloc[0]['price']
            450.0
        """
        return pd.DataFrame(
            [
                {
                    "timestamp": t.timestamp,
                    "symbol": t.symbol,
                    "price": float(t.price),
                    "volume": t.volume,
                    "bid": float(t.bid) if t.bid else None,
                    "ask": float(t.ask) if t.ask else None,
                }
                for t in ticks
            ]
        )

    @staticmethod
    def unified_to_vol_arb_position(unified_pos: UnifiedPosition) -> Position:
        """
        Convert UnifiedPosition to vol arb Position.

        Enables calling vol arb strategy's generate_signals method
        with positions from the unified backtest engine.

        Args:
            unified_pos: UnifiedPosition from backtest engine

        Returns:
            Vol arb Position object

        Example:
            >>> from paper_trading.core.types import UnifiedPosition, StrategyType
            >>> unified = UnifiedPosition(
            ...     symbol='SPY',
            ...     quantity=100,
            ...     avg_entry_price=Decimal('450.0'),
            ...     current_price=Decimal('455.0'),
            ...     strategy_type=StrategyType.VOLATILITY_ARBITRAGE,
            ...     last_update=datetime(2024, 1, 1)
            ... )
            >>> vol_pos = DataAdapter.unified_to_vol_arb_position(unified)
            >>> vol_pos.quantity
            100
        """
        return Position(
            symbol=unified_pos.symbol,
            quantity=unified_pos.quantity,
            avg_entry_price=unified_pos.avg_entry_price,
            current_price=unified_pos.current_price,
            last_update=unified_pos.last_update,
        )

    @staticmethod
    def vol_arb_to_unified_position(
        vol_pos: Position, strategy_type: "StrategyType"
    ) -> UnifiedPosition:
        """
        Convert vol arb Position to UnifiedPosition.

        Args:
            vol_pos: Vol arb Position object
            strategy_type: Strategy type to tag the position with

        Returns:
            UnifiedPosition for backtest engine

        Example:
            >>> from volatility_arbitrage.core.types import Position
            >>> from paper_trading.core.types import StrategyType
            >>> vol_pos = Position(
            ...     symbol='SPY',
            ...     quantity=100,
            ...     avg_entry_price=Decimal('450.0'),
            ...     current_price=Decimal('455.0'),
            ...     last_update=datetime(2024, 1, 1)
            ... )
            >>> unified = DataAdapter.vol_arb_to_unified_position(
            ...     vol_pos, StrategyType.VOLATILITY_ARBITRAGE
            ... )
            >>> unified.unrealized_pnl
            Decimal('500.0')
        """
        return UnifiedPosition(
            symbol=vol_pos.symbol,
            quantity=vol_pos.quantity,
            avg_entry_price=vol_pos.avg_entry_price,
            current_price=vol_pos.current_price,
            strategy_type=strategy_type,
            last_update=vol_pos.last_update,
        )

    @staticmethod
    def create_option_contract_from_dict(data: dict) -> OptionContract:
        """
        Create OptionContract from dictionary.

        Useful for converting DataFrame rows or API responses into
        vol arb's OptionContract type.

        Args:
            data: Dictionary with keys: symbol, option_type, strike, expiry, price
                Optional: bid, ask, volume, open_interest, implied_volatility

        Returns:
            OptionContract object

        Example:
            >>> contract_data = {
            ...     'symbol': 'SPY',
            ...     'option_type': 'call',
            ...     'strike': 450.0,
            ...     'expiry': datetime(2024, 2, 1),
            ...     'price': 5.50
            ... }
            >>> contract = DataAdapter.create_option_contract_from_dict(contract_data)
            >>> contract.strike
            Decimal('450.0')
        """
        return OptionContract(
            symbol=data["symbol"],
            option_type=OptionType(data["option_type"].lower()),
            strike=Decimal(str(data["strike"])),
            expiry=data["expiry"],
            price=Decimal(str(data["price"])),
            bid=Decimal(str(data["bid"])) if "bid" in data and data["bid"] is not None else None,
            ask=Decimal(str(data["ask"])) if "ask" in data and data["ask"] is not None else None,
            volume=int(data.get("volume", 0)),
            open_interest=int(data.get("open_interest", 0)),
            implied_volatility=Decimal(str(data["implied_volatility"]))
            if "implied_volatility" in data and data["implied_volatility"] is not None
            else None,
        )

    @staticmethod
    def create_option_chain_from_dataframe(
        df: pd.DataFrame,
        symbol: str,
        timestamp: datetime,
        expiry: datetime,
        underlying_price: Decimal,
        risk_free_rate: Decimal = Decimal("0.05"),
    ) -> OptionChain:
        """
        Create OptionChain from DataFrame.

        Args:
            df: DataFrame with columns: option_type, strike, price, etc.
            symbol: Underlying symbol
            timestamp: Chain snapshot time
            expiry: Options expiration date
            underlying_price: Current underlying price
            risk_free_rate: Risk-free rate (default: 5%)

        Returns:
            OptionChain object with calls and puts

        Example:
            >>> df = pd.DataFrame({
            ...     'option_type': ['call', 'put'],
            ...     'strike': [450.0, 450.0],
            ...     'price': [5.50, 4.25],
            ...     'expiry': [datetime(2024, 2, 1), datetime(2024, 2, 1)]
            ... })
            >>> chain = DataAdapter.create_option_chain_from_dataframe(
            ...     df, 'SPY', datetime(2024, 1, 1),
            ...     datetime(2024, 2, 1), Decimal('450.0')
            ... )
            >>> len(chain.calls)
            1
        """
        calls = []
        puts = []

        for _, row in df.iterrows():
            contract_data = row.to_dict()
            contract_data["symbol"] = symbol
            contract = DataAdapter.create_option_contract_from_dict(contract_data)

            if contract.option_type == OptionType.CALL:
                calls.append(contract)
            else:
                puts.append(contract)

        return OptionChain(
            symbol=symbol,
            timestamp=timestamp,
            expiry=expiry,
            underlying_price=underlying_price,
            calls=calls,
            puts=puts,
            risk_free_rate=risk_free_rate,
        )
