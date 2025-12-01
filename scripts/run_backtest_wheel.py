import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml

from core.data.factory import create_data_engine_from_config
from core.data.wheel_view import WheelDataView
from core.backtest.scheduler import DecisionScheduler
from core.backtest.engine import BacktestEngine
from core.strategy.wheel import WheelStrategy, WheelStrategyConfig
from core.utils.logging import Logger
from core.utils.config_loader import load_config_with_secrets

async def main():
    # Use absolute paths for config files
    config_dir = project_root / "config"
    env_file = config_dir / "env.backtest.yaml"
    strategy_file = config_dir / "strategy.wheel.yaml"
    
    if not env_file.exists():
        raise FileNotFoundError(f"Config file not found: {env_file}")
    if not strategy_file.exists():
        raise FileNotFoundError(f"Config file not found: {strategy_file}")
    
    # Load configs with secrets merged in
    env = load_config_with_secrets(env_file)
    strat_cfg_raw = load_config_with_secrets(strategy_file)
    
    # Validate required config keys
    if "backtest" not in env:
        raise ValueError("Missing 'backtest' section in config")
    if "symbol" not in strat_cfg_raw:
        raise ValueError("Missing 'symbol' in strategy config")
    
    symbol = strat_cfg_raw["symbol"]
    bt_cfg = env["backtest"]
    
    # Validate backtest config
    required_bt_keys = ["start", "end", "initial_cash"]
    for key in required_bt_keys:
        if key not in bt_cfg:
            raise ValueError(f"Missing 'backtest.{key}' in config")
    
    # Create data engine from config (supports multiple data sources)
    # This will use the 'data.historical_source' setting from config
    data_engine = create_data_engine_from_config(
        env_config=env,
        use_for="historical",  # Use historical source for backtesting
    )
    wheel_view = WheelDataView(data_engine, symbol)
    strat_cfg = WheelStrategyConfig(strat_cfg_raw)
    
    logger = Logger(prefix="[BACKTEST]")
    scheduler = DecisionScheduler(interval_minutes=15)
    
    strategy = WheelStrategy(symbol, strat_cfg, wheel_view)
    engine = BacktestEngine(data_engine, scheduler, logger)
    
    start = datetime.fromisoformat(bt_cfg["start"])
    end = datetime.fromisoformat(bt_cfg["end"])
    initial_cash = bt_cfg["initial_cash"]
    
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
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBacktest interrupted. Exiting...")
        sys.exit(0)
