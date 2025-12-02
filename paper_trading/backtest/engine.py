"""
Unified backtest engine for single-strategy backtesting.

Provides a minimal, interview-ready implementation (~220 lines) demonstrating:
- Event-driven simulation architecture
- Position tracking as single source of truth
- Multi-symbol support
- Equity curve generation
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

import pandas as pd

from paper_trading.core.interfaces import StrategyInterface
from paper_trading.core.types import BacktestResult, Signal, StrategyType, Trade, UnifiedPosition


class BacktestEngine:
    """
    Event-driven backtesting engine for strategy evaluation.

    Simulates trading by looping over timestamps, generating signals,
    and executing trades with immediate fills at close prices.

    Example:
        >>> strategy = VolatilityArbitrageAdapter(vol_arb_strategy)
        >>> engine = BacktestEngine(strategy, market_data, initial_cash=100_000)
        >>> result = engine.run()
        >>> print(f"Total return: {result.total_return_pct}%")
    """

    def __init__(
        self,
        strategy: StrategyInterface,
        market_data: pd.DataFrame,
        initial_cash: float = 100_000.0,
        commission_rate: float = 0.0,
    ) -> None:
        """
        Initialize backtest engine.

        Args:
            strategy: Strategy implementing StrategyInterface
            market_data: DataFrame with columns: timestamp, symbol, close, volume
            initial_cash: Starting cash balance
            commission_rate: Commission as fraction (e.g., 0.001 = 0.1%)
        """
        self.strategy = strategy
        self.market_data = market_data.sort_values("timestamp").reset_index(drop=True)
        self.initial_cash = Decimal(str(initial_cash))
        self.commission_rate = Decimal(str(commission_rate))

        # State tracking
        self.cash = self.initial_cash
        self.positions: dict[str, UnifiedPosition] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[dict] = []

    def run(self) -> BacktestResult:
        """
        Run backtest simulation.

        Returns:
            BacktestResult with trades, equity curve, and metrics
        """
        if self.market_data.empty:
            return self._create_empty_result()

        # Get date range
        timestamps = self.market_data["timestamp"].unique()
        start_date = timestamps[0]
        end_date = timestamps[-1]

        # Notify strategy of backtest start
        self.strategy.on_backtest_start(start_date, end_date)

        # Main event loop: process each timestamp
        for timestamp in timestamps:
            self._process_timestamp(timestamp)

        # Notify strategy of backtest end
        self.strategy.on_backtest_end()

        # Calculate final metrics
        return self._calculate_metrics(start_date, end_date)

    def _process_timestamp(self, timestamp: datetime) -> None:
        """Process a single timestamp: update positions, generate signals, execute trades."""
        # Get market data for this timestamp
        daily_data = self.market_data[self.market_data["timestamp"] == timestamp]

        # Update existing positions with current prices
        self._update_positions(daily_data, timestamp)

        # Generate signals from strategy
        signals = self.strategy.generate_signals(timestamp, daily_data, self.positions)

        # Execute each signal
        for signal in signals:
            self._execute_signal(signal, daily_data, timestamp)

        # Record equity snapshot
        self._record_equity(timestamp)

    def _update_positions(self, market_data: pd.DataFrame, timestamp: datetime) -> None:
        """Mark positions to market using current prices."""
        for symbol, position in self.positions.items():
            symbol_data = market_data[market_data["symbol"] == symbol]
            if not symbol_data.empty:
                current_price = Decimal(str(symbol_data.iloc[0]["close"]))
                position.update_price(current_price, timestamp)

    def _execute_signal(
        self, signal: Signal, market_data: pd.DataFrame, timestamp: datetime
    ) -> None:
        """
        Execute a trading signal.

        Handles buy/sell with position averaging, insufficient funds,
        and missing market data gracefully.
        """
        # Get price from market data
        symbol_data = market_data[market_data["symbol"] == signal.symbol]
        if symbol_data.empty:
            return  # Skip if no data for this symbol

        price = Decimal(str(symbol_data.iloc[0]["close"]))

        if signal.action == "buy":
            self._execute_buy(signal, price, timestamp)
        elif signal.action == "sell":
            self._execute_sell(signal, price, timestamp)

    def _execute_buy(self, signal: Signal, price: Decimal, timestamp: datetime) -> None:
        """Execute buy order with position averaging."""
        # Calculate cost including commission
        cost = price * Decimal(signal.quantity) * (Decimal("1") + self.commission_rate)

        # Check sufficient funds
        if cost > self.cash:
            return  # Insufficient funds, skip

        # Deduct cash
        self.cash -= cost

        # Create or update position
        if signal.symbol in self.positions:
            # Average into existing position
            pos = self.positions[signal.symbol]
            total_qty = pos.quantity + signal.quantity
            new_avg_price = (
                pos.avg_entry_price * Decimal(pos.quantity) + price * Decimal(signal.quantity)
            ) / Decimal(total_qty)

            pos.quantity = total_qty
            pos.avg_entry_price = new_avg_price
            pos.current_price = price
            pos.last_update = timestamp
        else:
            # Create new position
            self.positions[signal.symbol] = UnifiedPosition(
                symbol=signal.symbol,
                quantity=signal.quantity,
                avg_entry_price=price,
                current_price=price,
                strategy_type=signal.strategy_type,
                last_update=timestamp,
            )

        # Record trade
        commission = cost - (price * Decimal(signal.quantity))
        self.trades.append(
            Trade(
                timestamp=timestamp,
                symbol=signal.symbol,
                action="buy",
                quantity=signal.quantity,
                price=price,
                commission=commission,
                strategy_type=signal.strategy_type,
                signal_reason=signal.reason,
            )
        )

        # Notify strategy
        self.strategy.on_trade_executed(timestamp, signal, float(price))

    def _execute_sell(self, signal: Signal, price: Decimal, timestamp: datetime) -> None:
        """Execute sell order, reducing or closing position."""
        # Check if we have a position to sell
        if signal.symbol not in self.positions:
            return  # No position to sell, skip

        pos = self.positions[signal.symbol]

        # Determine quantity to sell (can't sell more than we have)
        sell_qty = min(signal.quantity, pos.quantity)

        # Calculate proceeds (subtract commission)
        proceeds = price * Decimal(sell_qty)
        commission = proceeds * self.commission_rate
        net_proceeds = proceeds - commission

        # Add cash
        self.cash += net_proceeds

        # Update or close position
        if sell_qty == pos.quantity:
            # Close entire position
            del self.positions[signal.symbol]
        else:
            # Reduce position
            pos.quantity -= sell_qty
            pos.current_price = price
            pos.last_update = timestamp

        # Record trade
        self.trades.append(
            Trade(
                timestamp=timestamp,
                symbol=signal.symbol,
                action="sell",
                quantity=sell_qty,
                price=price,
                commission=commission,
                strategy_type=signal.strategy_type,
                signal_reason=signal.reason,
            )
        )

        # Notify strategy
        self.strategy.on_trade_executed(timestamp, signal, float(price))

    def _record_equity(self, timestamp: datetime) -> None:
        """Record equity curve snapshot."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        total_equity = self.cash + positions_value

        self.equity_curve.append(
            {
                "timestamp": timestamp,
                "cash": float(self.cash),
                "positions_value": float(positions_value),
                "total_equity": float(total_equity),
            }
        )

    def _calculate_max_drawdown(self) -> Decimal:
        """Calculate maximum peak-to-trough drawdown."""
        if not self.equity_curve:
            return Decimal("0")

        equity_values = [Decimal(str(e["total_equity"])) for e in self.equity_curve]

        peak = equity_values[0]
        max_dd = Decimal("0")

        for value in equity_values:
            if value > peak:
                peak = value

            if peak > 0:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)

        return max_dd

    def _calculate_metrics(self, start_date: datetime, end_date: datetime) -> BacktestResult:
        """Calculate backtest metrics and create result."""
        final_equity = self.cash + sum(pos.market_value for pos in self.positions.values())

        total_return = final_equity - self.initial_cash
        total_return_pct = (total_return / self.initial_cash) * Decimal("100")

        max_drawdown = self._calculate_max_drawdown()

        # Trade statistics (simple version for MVP)
        num_trades = len(self.trades)
        # For MVP: defer win rate calculation (requires matching buys/sells)
        num_winning_trades = 0
        num_losing_trades = 0
        win_rate = Decimal("0")

        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)

        return BacktestResult(
            strategy_type=self.strategy.strategy_type
            if hasattr(self.strategy, "strategy_type")
            else StrategyType.VOLATILITY_ARBITRAGE,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_cash,
            final_capital=final_equity,
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=None,  # Defer for MVP
            max_drawdown=max_drawdown,
            num_trades=num_trades,
            num_winning_trades=num_winning_trades,
            num_losing_trades=num_losing_trades,
            win_rate=win_rate,
            trades=self.trades,
            equity_curve=equity_df,
        )

    def _create_empty_result(self) -> BacktestResult:
        """Create empty result for edge case of no data."""
        return BacktestResult(
            strategy_type=self.strategy.strategy_type
            if hasattr(self.strategy, "strategy_type")
            else StrategyType.VOLATILITY_ARBITRAGE,
            start_date=datetime.now(),
            end_date=datetime.now(),
            initial_capital=self.initial_cash,
            final_capital=self.initial_cash,
            total_return=Decimal("0"),
            total_return_pct=Decimal("0"),
            sharpe_ratio=None,
            max_drawdown=Decimal("0"),
            num_trades=0,
            num_winning_trades=0,
            num_losing_trades=0,
            win_rate=Decimal("0"),
            trades=[],
            equity_curve=pd.DataFrame(
                columns=["timestamp", "cash", "positions_value", "total_equity"]
            ),
        )
