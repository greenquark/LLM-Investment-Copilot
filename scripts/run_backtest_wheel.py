import asyncio
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml

from core.data.factory import create_data_engine_from_config
from core.data.wheel_view import WheelDataView
from core.backtest.scheduler import DecisionScheduler
from core.backtest.engine import BacktestEngine
from core.backtest.backtest_utils import (
    load_backtest_config,
    get_backtest_symbol,
    get_backtest_timeframe,
    parse_backtest_dates,
    print_backtest_header,
)
from core.strategy.wheel import WheelStrategy, WheelStrategyConfig
from core.utils.logging import Logger
from core.utils.config_loader import load_config_with_secrets

async def main(
    ticker: Optional[str] = None,
    timeframe: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: Optional[int] = None,
):
    """
    Main backtest function.
    
    Args:
        ticker: Optional ticker symbol (overrides config)
        timeframe: Optional timeframe (overrides config, default: 15m for wheel strategy)
        start_date: Optional start date in YYYY-MM-DD format (overrides config)
        end_date: Optional end date in YYYY-MM-DD format (overrides config)
        days: Optional number of calendar days for backtest period (e.g., 365 for one year)
    """
    # Load backtest configuration using shared utilities
    env, strat_cfg_raw, bt_cfg = load_backtest_config(
        project_root=project_root,
        strategy_config_file="strategy.wheel.yaml",
    )
    
    # Get symbol with CLI priority
    symbol = get_backtest_symbol(bt_cfg, strat_cfg_raw, cli_symbol=ticker)
    
    # Wheel strategy uses 15m by default, but allow override
    timeframe = get_backtest_timeframe(bt_cfg, strat_cfg_raw, default="15m", cli_timeframe=timeframe)
    
    # Create data engine from config (supports multiple data sources)
    # This will use the 'data.historical_source' setting from config
    data_engine = create_data_engine_from_config(
        env_config=env,
        use_for="historical",  # Use historical source for backtesting
    )
    wheel_view = WheelDataView(data_engine, symbol)
    strat_cfg = WheelStrategyConfig(strat_cfg_raw)
    
    logger = Logger(prefix="[BACKTEST]")
    scheduler = DecisionScheduler(interval_minutes=15)  # Wheel strategy uses 15m intervals
    
    strategy = WheelStrategy(symbol, strat_cfg, wheel_view)
    engine = BacktestEngine(data_engine, scheduler, logger)
    
    # Parse dates with CLI priority
    start, end, initial_cash = parse_backtest_dates(bt_cfg, cli_start_date=start_date, cli_end_date=end_date, cli_days=days)
    
    # Print backtest header using shared utility
    print_backtest_header(symbol, start, end, initial_cash, timeframe)
    
    logger.log(f"Starting backtest for {symbol} from {start} to {end} with ${initial_cash:,.2f}")
    logger.log("Press Ctrl-C to stop the backtest early and see partial results")
    
    try:
        result = await engine.run(symbol, strategy, start, end, initial_cash)
        print("\n=== Backtest Results ===")
        print(f"Equity curve points: {len(result.equity_curve)}")
        print(f"\nPerformance Metrics:")
        print(f"  Total Return: {result.metrics.total_return:.2%}")
        print(f"  CAGR: {result.metrics.cagr:.2%}")
        print(f"  Volatility: {result.metrics.volatility:.2%}")
        print(f"  Sharpe Ratio: {result.metrics.sharpe:.2f}")
        print(f"  Max Drawdown: {result.metrics.max_drawdown:.2%}")
        if result.metrics.max_drawdown_start:
            print(f"  Max DD Period: {result.metrics.max_drawdown_start.date()} to {result.metrics.max_drawdown_end.date() if result.metrics.max_drawdown_end else 'N/A'}")
    except KeyboardInterrupt:
        logger.log("\nBacktest interrupted by user (Ctrl-C)")
        # Try to get partial results if available
        try:
            result = engine.get_partial_result()
            if result and result.equity_curve:
                print("\n=== Partial Backtest Results (Interrupted) ===")
                print(f"Equity curve points: {len(result.equity_curve)}")
                print(f"\nPerformance Metrics (Partial):")
                print(f"  Total Return: {result.metrics.total_return:.2%}")
                print(f"  CAGR: {result.metrics.cagr:.2%}")
                print(f"  Volatility: {result.metrics.volatility:.2%}")
                print(f"  Sharpe Ratio: {result.metrics.sharpe:.2f}")
                print(f"  Max Drawdown: {result.metrics.max_drawdown:.2%}")
                if result.metrics.max_drawdown_start:
                    print(f"  Max DD Period: {result.metrics.max_drawdown_start.date()} to {result.metrics.max_drawdown_end.date() if result.metrics.max_drawdown_end else 'N/A'}")
            else:
                print("\nNo partial results available yet.")
        except Exception as e:
            logger.log(f"Could not retrieve partial results: {e}")
        sys.exit(0)
    except Exception as e:
        logger.log(f"Error during backtest: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Wheel Strategy backtest")
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Ticker symbol to use (overrides config files). Example: --ticker SPY",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Timeframe to use (overrides config files). Example: --timeframe 15m",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for backtest (YYYY-MM-DD, overrides config). Example: --start-date 2024-01-01",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for backtest (YYYY-MM-DD, overrides config). Example: --end-date 2024-12-31",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of calendar days for backtest period (e.g., 365 for one year). End date defaults to today if not specified. Example: --days 365",
    )
    args = parser.parse_args()
    
    try:
        asyncio.run(main(
            ticker=args.ticker,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            days=args.days,
        ))
    except KeyboardInterrupt:
        print("\nBacktest interrupted. Exiting...")
        sys.exit(0)
