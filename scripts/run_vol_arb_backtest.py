#!/usr/bin/env python3
"""
Simple demo script for unified backtest engine.

Runs a buy-and-hold strategy on SPY as a proof-of-concept for the
volatility arbitrage adapter and backtest engine integration.
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# Add project root and vol arb to path
PROJECT_ROOT = Path(__file__).parent.parent
VOL_ARB_PATH = PROJECT_ROOT / "strategies" / "volatility-arbitrage" / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VOL_ARB_PATH))

from volatility_arbitrage.strategy.base import BuyAndHoldStrategy

from paper_trading.adapters.strategy_adapter import VolatilityArbitrageAdapter
from paper_trading.backtest.engine import BacktestEngine


def fetch_market_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical market data using yfinance.

    Args:
        symbol: Ticker symbol (e.g., "SPY")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume
    """
    print(f"Fetching {symbol} data from {start_date} to {end_date}...")

    # Download data
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)

    if df.empty:
        raise ValueError(f"No data found for {symbol}")

    # Rename columns to match backtest engine format
    df = df.reset_index()
    df = df.rename(
        columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    # Add symbol column
    df["symbol"] = symbol

    # Select and reorder columns
    df = df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]

    print(f"Fetched {len(df)} rows of data")

    return df


def print_backtest_summary(result):
    """
    Print backtest results summary.

    Args:
        result: BacktestResult instance
    """
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Period: {result.start_date.date()} to {result.end_date.date()}")
    print(f"Duration: {(result.end_date - result.start_date).days} days")
    print()
    print(f"Initial Capital: ${result.initial_capital:,.2f}")
    print(f"Final Capital:   ${result.final_capital:,.2f}")
    print(f"Total Return:    ${result.total_return:,.2f} ({result.total_return_pct:.2f}%)")
    print()
    print(f"Max Drawdown:    {result.max_drawdown:.2%}")
    print(f"Sharpe Ratio:    {result.sharpe_ratio if result.sharpe_ratio else 'N/A'}")
    print()
    print(f"Total Trades:    {result.num_trades}")
    print(f"Winning Trades:  {result.num_winning_trades}")
    print(f"Losing Trades:   {result.num_losing_trades}")
    print(f"Win Rate:        {result.win_rate:.1f}%")
    print("=" * 60)

    if result.num_trades > 0:
        print("\nTrade History:")
        print("-" * 60)
        for i, trade in enumerate(result.trades, 1):
            print(
                f"{i}. {trade.timestamp.date()} | {trade.action.upper()} {trade.quantity} "
                f"{trade.symbol} @ ${trade.price:.2f}"
            )
            print(f"   Reason: {trade.signal_reason}")


def plot_equity_curve(result, save_path="backtest_equity_curve.png"):
    """
    Plot equity curve from backtest results.

    Args:
        result: BacktestResult instance
        save_path: Path to save the plot (optional)
    """
    equity_df = result.equity_curve

    if equity_df.empty:
        print("No equity data to plot")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot total equity
    ax1.plot(equity_df["timestamp"], equity_df["total_equity"], label="Total Equity", linewidth=2)
    ax1.axhline(
        y=float(result.initial_capital),
        color="gray",
        linestyle="--",
        label="Initial Capital",
        alpha=0.7,
    )
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.set_title("Backtest Equity Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Format y-axis with thousands separator
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # Plot cash vs positions
    ax2.plot(equity_df["timestamp"], equity_df["cash"], label="Cash", linewidth=2)
    ax2.plot(
        equity_df["timestamp"], equity_df["positions_value"], label="Positions Value", linewidth=2
    )
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Value ($)")
    ax2.set_title("Cash vs Positions")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nEquity curve saved to: {save_path}")
    else:
        plt.show()


def main():
    """Run backtest demonstration."""
    # Configuration
    SYMBOL = "SPY"
    START_DATE = "2023-01-01"
    END_DATE = "2024-12-01"
    INITIAL_CASH = 100_000.0
    BUY_QUANTITY = 100

    print("=" * 60)
    print("UNIFIED BACKTEST ENGINE DEMONSTRATION")
    print("=" * 60)
    print(f"Strategy: Buy-and-hold {BUY_QUANTITY} shares of {SYMBOL}")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Initial Capital: ${INITIAL_CASH:,.2f}")
    print()

    # Fetch market data
    market_data = fetch_market_data(SYMBOL, START_DATE, END_DATE)

    # Create strategy
    # Using BuyAndHoldStrategy as a simple example
    # The vol arb adapter works with any strategy implementing the base Strategy interface
    base_strategy = BuyAndHoldStrategy(symbol=SYMBOL, quantity=BUY_QUANTITY)

    # Wrap in adapter
    strategy = VolatilityArbitrageAdapter(base_strategy)

    print(f"Strategy initialized: {strategy.__class__.__name__}")
    print(f"Required symbols: {strategy.get_required_symbols()}")
    print()

    # Run backtest
    print("Running backtest...")
    engine = BacktestEngine(
        strategy=strategy,
        market_data=market_data,
        initial_cash=INITIAL_CASH,
        commission_rate=0.001,  # 0.1% commission
    )

    result = engine.run()

    # Print results
    print_backtest_summary(result)

    # Plot equity curve
    print("\nGenerating equity curve plot...")
    plot_equity_curve(result)


if __name__ == "__main__":
    main()
