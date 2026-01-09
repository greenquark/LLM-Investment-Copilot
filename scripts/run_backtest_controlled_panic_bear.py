"""
Backtest script for Controlled Panic Bear Strategy.

This strategy enters bearish positions during panic conditions:
- High VIX (volatility spike)
- CNN Fear & Greed Index in panic range (5-25)
- Price below SMA20 with negative 3-day return

Uses options strategies: put spreads, diagonals, and lotto puts.
"""

import asyncio
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

import yaml

from core.data.factory import create_data_engine_from_config
from core.data.trading_calendar import get_trading_calendar
from core.backtest.scheduler import DecisionScheduler
from core.backtest.engine import BacktestEngine
from core.backtest.benchmarks import run_buy_and_hold
from core.strategy.controlled_panic_bear import (
    ControlledPanicBearStrategy,
    ControlledPanicBearConfig,
)
from core.utils.logging import Logger
from core.utils.config_loader import load_config_with_secrets
from core.visualization import PlotlyChartVisualizer
from core.visualization.chart_config import get_chart_config


async def main(use_local_chart: bool = False):
    """Main backtest function."""
    # Use absolute paths for config files
    config_dir = project_root / "config"
    env_file = config_dir / "env.backtest.yaml"
    strategy_file = config_dir / "strategy.controlled_panic_bear.yaml"
    
    if not env_file.exists():
        raise FileNotFoundError(f"Config file not found: {env_file}")
    
    # Load environment config
    env = load_config_with_secrets(env_file)
    
    # Load strategy config if exists, otherwise use defaults
    if strategy_file.exists():
        strat_cfg_raw = load_config_with_secrets(strategy_file)
    else:
        print(f"Strategy config not found: {strategy_file}")
        print("Using default configuration")
        strat_cfg_raw = {}
    
    # Validate required config keys
    if "backtest" not in env:
        raise ValueError("Missing 'backtest' section in config")
    
    bt_cfg = env["backtest"]
    
    # Validate backtest config
    required_bt_keys = ["start", "end", "initial_cash"]
    for key in required_bt_keys:
        if key not in bt_cfg:
            raise ValueError(f"Missing 'backtest.{key}' in config")
    
    # Create data engine from config
    data_engine = create_data_engine_from_config(
        env_config=env,
        use_for="historical",
    )
    
    # Create strategy config
    timeframe = bt_cfg.get("timeframe") or strat_cfg_raw.get("timeframe", "1D")
    symbol = strat_cfg_raw.get("symbol", "SPY")
    vix_symbol = strat_cfg_raw.get("vix_symbol", "^VIX")
    
    strategy_config = ControlledPanicBearConfig.from_dict(strat_cfg_raw)
    
    # Create strategy
    strategy = ControlledPanicBearStrategy(
        symbol=symbol,
        config=strategy_config,
        data_engine=data_engine,
        vix_symbol=vix_symbol,
    )
    
    logger = Logger(prefix="[ControlledPanicBear]")
    
    # Create scheduler (daily decisions)
    scheduler = DecisionScheduler(interval_minutes=24 * 60)
    
    # Create backtest engine
    engine = BacktestEngine(data_engine, scheduler, logger)
    
    # Parse dates
    start = datetime.fromisoformat(bt_cfg["start"])
    end_original = datetime.fromisoformat(bt_cfg["end"])
    initial_cash = bt_cfg["initial_cash"]
    
    # Adjust end date to the last trading day on or before the end date
    try:
        calendar = get_trading_calendar("NYSE")
        end_date_original = end_original.date()
        
        # Check if the end date itself is a trading day
        if calendar.is_trading_day(end_date_original):
            # End date is a trading day, use it
            end = end_original
        else:
            # End date is not a trading day, get the previous trading day
            previous_trading_day = calendar.previous_trading_day(end_date_original)
            if previous_trading_day:
                # Update end datetime to use the previous trading day at end of day
                end = datetime.combine(previous_trading_day, datetime.max.time())
                logger.log(
                    f"Adjusted end date from {end_date_original} to {end.date()} "
                    f"(last trading day on or before end date)"
                )
            else:
                # If we can't get previous trading day, use original end date
                end = end_original
                logger.log(
                    f"Could not determine previous trading day for {end_date_original}, "
                    f"using original end date"
                )
    except Exception as e:
        # If trading calendar fails, use original end date
        end = end_original
        logger.log(
            f"Trading calendar unavailable, using original end date {end_date_original}: {e}"
        )
    
    # Print backtest date range
    print("\n" + "=" * 80)
    print("BACKTEST DATE RANGE")
    print("=" * 80)
    print(f"Start Date: {start.date()}")
    print(f"End Date:   {end.date()}")
    print(f"Initial Cash: ${initial_cash:,.2f}")
    print(f"Timeframe: {timeframe}")
    print("=" * 80)
    print()
    
    logger.log(
        f"Starting Controlled Panic Bear Backtest for {symbol} "
        f"{start.date()} → {end.date()} | initial_cash=${initial_cash:,.2f}"
    )
    logger.log(
        f"Config: VIX_min={strategy_config.vix_min}, "
        f"CNN_range=[{strategy_config.cnn_min}, {strategy_config.cnn_max}], "
        f"ret3d_min={strategy_config.ret3d_min:.2%}"
    )
    
    # Run backtest
    result = await engine.run(
        symbol, strategy, start, end, initial_cash, timeframe=timeframe
    )
    
    print("\n=== Controlled Panic Bear Backtest Results ===")
    
    equity_items = sorted(result.equity_curve.items(), key=lambda x: x[0])
    if equity_items:
        initial_equity = equity_items[0][1]
        final_equity = equity_items[-1][1]
    else:
        initial_equity = final_equity = initial_cash
    
    pnl = final_equity - initial_equity
    pnl_pct = (final_equity / initial_equity - 1) * 100 if initial_equity > 0 else 0.0
    
    print(f"Initial: ${initial_equity:,.2f}")
    print(f"Final:   ${final_equity:,.2f}")
    print(f"P&L:     ${pnl:,.2f} ({pnl_pct:+.2f}%)")
    
    print(f"\n=== Performance Metrics ===")
    print(f"  Total Return: {result.metrics.total_return:.2%}")
    print(f"  CAGR: {result.metrics.cagr:.2%}")
    print(f"  Volatility: {result.metrics.volatility:.2%}")
    print(f"  Sharpe Ratio: {result.metrics.sharpe:.2f}")
    print(f"  Max Drawdown: {result.metrics.max_drawdown:.2%}")
    
    # Calculate Buy & Hold benchmark
    print(f"\n=== Calculating Buy & Hold Benchmark ({symbol}) ===")
    try:
        bh_result = await run_buy_and_hold(
            data_engine, symbol, start, end, initial_cash, timeframe=timeframe
        )
        
        strategy_metrics = result.metrics
        bh_metrics = bh_result.metrics
        
        print(f"\n=== Strategy vs Buy & Hold Comparison ===")
        print(f"{'Metric':<20} {'Strategy':>15} {'Buy & Hold':>15} {'Difference':>15}")
        print("-" * 65)
        
        ret_diff = strategy_metrics.total_return - bh_metrics.total_return
        print(f"{'Total Return':<20} {strategy_metrics.total_return:>14.2%} {bh_metrics.total_return:>14.2%} {ret_diff:>+14.2%}")
        
        cagr_diff = strategy_metrics.cagr - bh_metrics.cagr
        print(f"{'CAGR':<20} {strategy_metrics.cagr:>14.2%} {bh_metrics.cagr:>14.2%} {cagr_diff:>+14.2%}")
        
        sharpe_diff = strategy_metrics.sharpe - bh_metrics.sharpe
        print(f"{'Sharpe Ratio':<20} {strategy_metrics.sharpe:>14.2f} {bh_metrics.sharpe:>14.2f} {sharpe_diff:>+14.2f}")
        
    except Exception as e:
        logger.log(f"Failed to calculate Buy & Hold benchmark: {e}")
        print(f"\n⚠️  Could not calculate Buy & Hold benchmark: {e}")
    
    # Print intents summary
    intents = strategy.get_intents()
    if intents:
        print(f"\n=== Trading Intents Generated ===")
        print(f"Total intents: {len(intents)}")
        
        intent_types = {}
        for intent in intents:
            intent_type = intent.get("type", "UNKNOWN")
            intent_types[intent_type] = intent_types.get(intent_type, 0) + 1
        
        for intent_type, count in sorted(intent_types.items()):
            print(f"  {intent_type}: {count}")
        
        print(f"\nFinal exposure: {strategy.get_exposure()}")
    
    # Generate visualization
    print(f"\n=== Starting Chart Visualization ===")
    try:
        bars = await data_engine.get_bars(symbol, start, end, timeframe)
        
        if bars:
            metrics_dict = {
                "total_return": result.metrics.total_return,
                "cagr": result.metrics.cagr,
                "volatility": result.metrics.volatility,
                "sharpe": result.metrics.sharpe,
                "max_drawdown": result.metrics.max_drawdown,
            }
            
            visualizer = PlotlyChartVisualizer(theme="tradingview", figsize=(1400, 900))
            chart_config = get_chart_config("controlled_panic_bear")
            visualizer.build_chart(
                bars=bars,
                signals=None,  # No signals for this strategy yet
                indicator_data=None,
                equity_curve=result.equity_curve,
                metrics=metrics_dict,
                symbol=symbol,
                show_equity=True,
                chart_config=chart_config,
                strategy_name="controlled_panic_bear",
            )
            
            print("\n📊 Displaying interactive chart...")
            visualizer.show(renderer="browser")
            
            # Optionally save to HTML
            save_html = input("\nSave chart to HTML file? (y/n): ").strip().lower()
            if save_html == 'y':
                output_file = project_root / "charts" / f"controlled_panic_bear_{start.date()}_to_{end.date()}.html"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                visualizer.to_html(str(output_file))
                print(f"✅ Chart saved to: {output_file}")
    
    except Exception as e:
        logger.log(f"Failed to generate chart: {e}")
        print(f"\n⚠️  Could not generate chart: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Controlled Panic Bear Strategy backtest")
    parser.add_argument(
        "--local-chart",
        action="store_true",
        help="Use local Python chart instead of web chart",
    )
    args = parser.parse_args()
    
    try:
        asyncio.run(main(use_local_chart=args.local_chart))
    except KeyboardInterrupt:
        print("\n\n⚠️  Backtest interrupted by user (Ctrl-C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
