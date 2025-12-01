import asyncio
import sys
import os
import subprocess
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging to show INFO level messages (for cache/MarketData logs)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

import yaml

from core.data.factory import create_data_engine_from_config
from core.backtest.scheduler import DecisionScheduler
from core.backtest.engine import BacktestEngine
from core.backtest.benchmarks import run_buy_and_hold
from core.strategy.mystic_pulse import MysticPulseStrategy, MysticPulseConfig
from core.utils.logging import Logger
from core.visualization import (
    get_web_chart_server,
    LocalChartVisualizer,
    PlotlyChartVisualizer,
)

async def main(use_local_chart: bool = False):
    # Use absolute paths for config files
    config_dir = project_root / "config"
    env_file = config_dir / "env.backtest.yaml"
    strategy_file = config_dir / "strategy.mystic_pulse.yaml"
    
    if not env_file.exists():
        raise FileNotFoundError(f"Config file not found: {env_file}")
    if not strategy_file.exists():
        # Create default config if it doesn't exist
        default_config = {
            "symbol": "AAPL",
            "adx_length": 9,
            "smoothing_factor": 1,
            "collect_length": 100,
            "contrast_gamma_bars": 0.7,
            "contrast_gamma_plots": 0.8,
            "min_trend_score": 5,
            "timeframe": "1D"  # Allowed: 15m, 30m, 1H, 2H, 4H, 1D
        }
        strategy_file.parent.mkdir(parents=True, exist_ok=True)
        with open(strategy_file, 'w') as f:
            yaml.dump(default_config, f)
        print(f"Created default config at {strategy_file}")
    
    with open(env_file) as f:
        env = yaml.safe_load(f)
    with open(strategy_file) as f:
        strat_cfg_raw = yaml.safe_load(f)
    
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
    
    # Create strategy config
    # Timeframe can be set in env.backtest.yaml (takes precedence) or strategy.mystic_pulse.yaml
    timeframe = bt_cfg.get("timeframe") or strat_cfg_raw.get("timeframe", "1D")
    
    # Log which timeframe is being used and from where
    if "timeframe" in bt_cfg:
        print(f"Using timeframe '{timeframe}' from env.backtest.yaml")
    else:
        print(f"Using timeframe '{timeframe}' from strategy.mystic_pulse.yaml (env.backtest.yaml timeframe not set)")
    
    strategy_config = MysticPulseConfig(
        adx_length=strat_cfg_raw.get("adx_length", 9),
        smoothing_factor=strat_cfg_raw.get("smoothing_factor", 1),
        collect_length=strat_cfg_raw.get("collect_length", 100),
        contrast_gamma_bars=strat_cfg_raw.get("contrast_gamma_bars", 0.7),
        contrast_gamma_plots=strat_cfg_raw.get("contrast_gamma_plots", 0.8),
        min_trend_score=strat_cfg_raw.get("min_trend_score", 5),
        timeframe=timeframe,
        capital_deployment_pct=strat_cfg_raw.get("capital_deployment_pct", 1.0),
    )
    
    strategy = MysticPulseStrategy(symbol, strategy_config, data_engine)
    
    logger = Logger(prefix="[BACKTEST]")
    
    # Calculate scheduler interval based on timeframe
    # Only allowed timeframes: 15m, 30m, 1H, 2H, 4H, 1D
    timeframe_upper = timeframe.upper()
    if timeframe_upper in ("1D", "D"):
        # Daily: 1 day
        scheduler = DecisionScheduler(interval_minutes=24 * 60)
    elif timeframe_upper.endswith("H"):
        # Hourly: parse number of hours (1H, 2H, 4H)
        try:
            hours = int(timeframe_upper[:-1]) if timeframe_upper != "H" else 1
        except ValueError:
            hours = 1
        scheduler = DecisionScheduler(interval_minutes=hours * 60)
    elif timeframe_upper.endswith("M") or timeframe_upper.endswith("m"):
        # Minutely: parse number of minutes (15m, 30m)
        try:
            minutes = int(timeframe_upper[:-1])
        except ValueError:
            minutes = 15
        scheduler = DecisionScheduler(interval_minutes=minutes)
    else:
        # Default to 15 minutes if parsing fails
        scheduler = DecisionScheduler(interval_minutes=15)
    
    engine = BacktestEngine(data_engine, scheduler, logger)
    
    start = datetime.fromisoformat(bt_cfg["start"])
    end = datetime.fromisoformat(bt_cfg["end"])
    initial_cash = bt_cfg["initial_cash"]
    
    logger.log(f"Starting Revised MP2.0 backtest for {symbol} from {start} to {end} with ${initial_cash:,.2f}")
    logger.log(f"Strategy config: ADX={strategy_config.adx_length}, Smoothing={strategy_config.smoothing_factor}, MinScore={strategy_config.min_trend_score}")
    logger.log(f"Capital deployment: {strategy_config.capital_deployment_pct*100:.0f}% per trade")
    logger.log("Press Ctrl-C to stop the backtest early and see partial results")
    
    try:
        result = await engine.run(symbol, strategy, start, end, initial_cash, timeframe=strategy_config.timeframe)
        print("\n=== Backtest Results ===")
        print(f"Equity curve points: {len(result.equity_curve)}")
        
        # Calculate final portfolio value and P&L
        if result.equity_curve:
            equity_items = sorted(result.equity_curve.items(), key=lambda x: x[0])
            final_equity = equity_items[-1][1]
            initial_equity = equity_items[0][1] if equity_items else initial_cash
        else:
            final_equity = initial_cash
            initial_equity = initial_cash
        
        # Calculate P&L using equity curve values for accuracy
        total_pnl = final_equity - initial_equity
        total_pnl_pct = (final_equity / initial_equity - 1.0) * 100 if initial_equity > 0 else 0.0
        
        print(f"\n=== Profit & Loss Summary ===")
        print(f"  Initial Capital: ${initial_cash:,.2f}")
        print(f"  Initial Equity: ${initial_equity:,.2f}")
        print(f"  Final Equity: ${final_equity:,.2f}")
        print(f"  Total P&L: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)")
        
        # Calculate realized vs unrealized P&L if we have access to portfolio
        # Note: This would require tracking closed trades, which we can add if needed
        
        print(f"\n=== Performance Metrics ===")
        print(f"  Total Return: {result.metrics.total_return:.2%}")
        print(f"  CAGR: {result.metrics.cagr:.2%}")
        print(f"  Volatility: {result.metrics.volatility:.2%}")
        print(f"  Sharpe Ratio: {result.metrics.sharpe:.2f}")
        print(f"  Max Drawdown: {result.metrics.max_drawdown:.2%}")
        if result.metrics.max_drawdown_start:
            print(f"  Max DD Period: {result.metrics.max_drawdown_start.date()} to {result.metrics.max_drawdown_end.date() if result.metrics.max_drawdown_end else 'N/A'}")
        
        # Calculate Buy & Hold benchmark for comparison (using same timeframe as strategy)
        print(f"\n=== Calculating Buy & Hold Benchmark ===")
        try:
            bh_result = await run_buy_and_hold(data_engine, symbol, start, end, initial_cash, timeframe=strategy_config.timeframe)
            
            strategy_metrics = result.metrics
            bh_metrics = bh_result.metrics
            
            print(f"\n=== Strategy vs Buy & Hold Comparison ===")
            print(f"{'Metric':<20} {'Strategy':>15} {'Buy & Hold':>15} {'Difference':>15}")
            print("-" * 65)
            
            # Total Return
            ret_diff = strategy_metrics.total_return - bh_metrics.total_return
            print(f"{'Total Return':<20} {strategy_metrics.total_return:>14.2%} {bh_metrics.total_return:>14.2%} {ret_diff:>+14.2%}")
            
            # CAGR
            cagr_diff = strategy_metrics.cagr - bh_metrics.cagr
            print(f"{'CAGR':<20} {strategy_metrics.cagr:>14.2%} {bh_metrics.cagr:>14.2%} {cagr_diff:>+14.2%}")
            
            # Volatility
            vol_diff = strategy_metrics.volatility - bh_metrics.volatility
            print(f"{'Volatility':<20} {strategy_metrics.volatility:>14.2%} {bh_metrics.volatility:>14.2%} {vol_diff:>+14.2%}")
            
            # Sharpe Ratio
            sharpe_diff = strategy_metrics.sharpe - bh_metrics.sharpe
            print(f"{'Sharpe Ratio':<20} {strategy_metrics.sharpe:>14.2f} {bh_metrics.sharpe:>14.2f} {sharpe_diff:>+14.2f}")
            
            # Max Drawdown
            dd_diff = strategy_metrics.max_drawdown - bh_metrics.max_drawdown
            print(f"{'Max Drawdown':<20} {strategy_metrics.max_drawdown:>14.2%} {bh_metrics.max_drawdown:>14.2%} {dd_diff:>+14.2%}")
            
            # Final Equity Comparison
            if result.equity_curve and bh_result.equity_curve:
                strategy_final = sorted(result.equity_curve.items())[-1][1]
                bh_final = sorted(bh_result.equity_curve.items())[-1][1]
                equity_diff = strategy_final - bh_final
                equity_diff_pct = ((strategy_final / bh_final) - 1.0) * 100 if bh_final > 0 else 0.0
                print(f"\n{'Final Equity':<20} ${strategy_final:>14,.2f} ${bh_final:>14,.2f} ${equity_diff:>+14,.2f} ({equity_diff_pct:+.2f}%)")
            
            # Outperformance summary
            print(f"\n=== Outperformance Summary ===")
            if ret_diff > 0:
                print(f"✅ Strategy outperformed Buy & Hold by {ret_diff:.2%} in total return")
            elif ret_diff < 0:
                print(f"❌ Strategy underperformed Buy & Hold by {abs(ret_diff):.2%} in total return")
            else:
                print(f"➖ Strategy matched Buy & Hold performance")
            
            if cagr_diff > 0:
                print(f"✅ Strategy CAGR is {cagr_diff:.2%} higher than Buy & Hold")
            elif cagr_diff < 0:
                print(f"❌ Strategy CAGR is {abs(cagr_diff):.2%} lower than Buy & Hold")
            
            if abs(sharpe_diff) > 0.1:
                if sharpe_diff > 0:
                    print(f"✅ Strategy has better risk-adjusted returns (Sharpe: {sharpe_diff:+.2f})")
                else:
                    print(f"❌ Buy & Hold has better risk-adjusted returns (Sharpe: {sharpe_diff:+.2f})")
            
        except Exception as e:
            logger.log(f"Failed to calculate Buy & Hold benchmark: {e}")
            print(f"\n⚠️  Could not calculate Buy & Hold benchmark: {e}")
        
        # Display cache statistics
        if hasattr(data_engine, 'get_cache_stats'):
            cache_stats = data_engine.get_cache_stats()
            print("\n=== Cache Statistics ===")
            print(f"  Total Data Requests: {cache_stats['total_requests']}")
            print(f"  Cache Hits (100%):   {cache_stats['cache_hits']}")
            print(f"  Cache Partial Hits:  {cache_stats['cache_partial_hits']}")
            print(f"  API Calls:           {cache_stats['api_calls']}")
            
            # Calculate cache hit rates
            # Full cache hit rate: requests fully served from cache (no API call needed)
            full_hit_rate = (cache_stats['cache_hits'] / cache_stats['total_requests']) * 100 if cache_stats['total_requests'] > 0 else 0.0
            # Overall cache effectiveness: requests that used cache (full or partial)
            total_with_cache = cache_stats['cache_hits'] + cache_stats['cache_partial_hits']
            overall_cache_rate = (total_with_cache / cache_stats['total_requests']) * 100 if cache_stats['total_requests'] > 0 else 0.0
            
            print(f"  Full Cache Hit Rate: {full_hit_rate:.1f}%")
            print(f"  Cache Used Rate:     {overall_cache_rate:.1f}% (includes partial hits)")
            print(f"  Bars from Cache:     {cache_stats['total_bars_from_cache']}")
            print(f"  Bars from API:       {cache_stats['total_bars_from_api']}")
        
        # Generate visualization
        chart_type = "Local (Python)" if use_local_chart else "Web (Flask)"
        print(f"\n=== Starting {chart_type} Chart ===")
        try:
            # Get all bars used in backtest (full range for charting)
            print(f"Fetching bars for chart: {symbol} from {start} to {end} with timeframe {strategy_config.timeframe}")
            bars = await data_engine.get_bars(symbol, start, end, strategy_config.timeframe)
            signals = strategy.get_signals()
            indicator_data = strategy.get_indicator_history()
            
            print(f"Retrieved {len(bars)} bars for charting")
            if bars:
                print(f"Bar timestamp range: {min(b.timestamp for b in bars)} to {max(b.timestamp for b in bars)}")
                print(f"First 5 bars: {[b.timestamp.strftime('%Y-%m-%d %H:%M:%S') for b in bars[:5]]}")
                print(f"Last 5 bars: {[b.timestamp.strftime('%Y-%m-%d %H:%M:%S') for b in bars[-5:]]}")
            
            # Filter indicator data to match bar timestamps exactly
            # Create a set of normalized bar timestamps for efficient lookup
            if strategy_config.timeframe.upper().endswith("H") or strategy_config.timeframe.upper() == "H":
                # Hourly: normalize both to hour level (remove minutes/seconds/microseconds)
                bar_timestamps_normalized = {
                    b.timestamp.replace(minute=0, second=0, microsecond=0) for b in bars
                }
                filtered_indicator_data = []
                seen_timestamps = set()
                for ind in indicator_data:
                    ind_ts_normalized = ind.timestamp.replace(minute=0, second=0, microsecond=0)
                    if ind_ts_normalized in bar_timestamps_normalized and ind_ts_normalized not in seen_timestamps:
                        filtered_indicator_data.append(ind)
                        seen_timestamps.add(ind_ts_normalized)
            else:
                # Other timeframes: normalize to seconds
                bar_timestamps_normalized = {
                    b.timestamp.replace(microsecond=0) for b in bars
                }
                filtered_indicator_data = []
                seen_timestamps = set()
                for ind in indicator_data:
                    ind_ts_normalized = ind.timestamp.replace(microsecond=0)
                    if ind_ts_normalized in bar_timestamps_normalized and ind_ts_normalized not in seen_timestamps:
                        filtered_indicator_data.append(ind)
                        seen_timestamps.add(ind_ts_normalized)
            
            # Sort filtered indicator data by timestamp to match bar order
            filtered_indicator_data.sort(key=lambda ind: ind.timestamp)
            
            print(f"Retrieved {len(bars)} bars, {len(signals)} signals, {len(indicator_data)} indicator data points (filtered to {len(filtered_indicator_data)} matching bars)")
            if len(bars) != len(filtered_indicator_data):
                print(f"Warning: Bar count ({len(bars)}) doesn't match filtered indicator count ({len(filtered_indicator_data)})")
                if bars and indicator_data:
                    print(f"Bar timestamp range: {min(b.timestamp for b in bars)} to {max(b.timestamp for b in bars)}")
                    print(f"Indicator timestamp range: {min(ind.timestamp for ind in indicator_data)} to {max(ind.timestamp for ind in indicator_data)}")
                    print(f"First 3 bar timestamps: {[b.timestamp.strftime('%Y-%m-%d %H:%M:%S') for b in bars[:3]]}")
                    print(f"First 3 filtered indicator timestamps: {[ind.timestamp.strftime('%Y-%m-%d %H:%M:%S') for ind in filtered_indicator_data[:3]]}")
            
            if bars:
                # Convert metrics to dict
                metrics_dict = {
                    "total_return": result.metrics.total_return,
                    "cagr": result.metrics.cagr,
                    "volatility": result.metrics.volatility,
                    "sharpe": result.metrics.sharpe,
                    "max_drawdown": result.metrics.max_drawdown,
                }
                
                # Choose charting method (default: Plotly)
                if use_local_chart:
                    # Use matplotlib (legacy)
                    print("\n=== Starting Local Chart Visualization (Matplotlib) ===")
                    chart = LocalChartVisualizer(style="dark_background", figsize=(16, 12))
                    chart.set_data(
                        bars=bars,
                        signals=signals,
                        indicator_data=filtered_indicator_data,
                        equity_curve=result.equity_curve,
                        metrics=metrics_dict,
                        symbol=symbol
                    )
                    chart.show(block=True)
                    chart.close()
                else:
                    # Use Plotly (recommended)
                    print("\n=== Starting Plotly Chart Visualization ===")
                    visualizer = PlotlyChartVisualizer(theme="tradingview", figsize=(1400, 900))
                    visualizer.build_chart(
                        bars=bars,
                        signals=signals,
                        indicator_data=filtered_indicator_data,
                        equity_curve=result.equity_curve,
                        metrics=metrics_dict,
                        symbol=symbol,
                        show_equity=True,
                    )
                    print("Opening interactive chart in browser...")
                    visualizer.show(renderer="browser")
                    
                    # Optionally save to HTML
                    save_html = input("\nSave chart to HTML? (y/n): ").lower().strip()
                    if save_html == 'y':
                        filename = f"chart_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        visualizer.to_html(filename)
                        print(f"Chart saved to {filename}")
            else:
                print("No bars available for charting")
        except Exception as e:
            print(f"ERROR starting chart: {e}")
            logger.log(f"Error starting chart: {e}")
            import traceback
            traceback.print_exc()
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
                
                # Try to start chart with partial results
                try:
                    bars = await data_engine.get_bars(symbol, start, end, strategy_config.timeframe)
                    signals = strategy.get_signals()
                    indicator_data = strategy.get_indicator_history()
                    
                    # Filter indicator data to match bar timestamps
                    if strategy_config.timeframe.upper().endswith("H") or strategy_config.timeframe.upper() == "H":
                        # Hourly: normalize both to hour level
                        bar_timestamps_normalized = {
                            b.timestamp.replace(minute=0, second=0, microsecond=0) for b in bars
                        }
                        filtered_indicator_data = [
                            ind for ind in indicator_data
                            if ind.timestamp.replace(minute=0, second=0, microsecond=0) in bar_timestamps_normalized
                        ]
                    else:
                        # Other timeframes: normalize to seconds
                        bar_timestamps_normalized = {
                            b.timestamp.replace(microsecond=0) for b in bars
                        }
                        filtered_indicator_data = [
                            ind for ind in indicator_data
                            if ind.timestamp.replace(microsecond=0) in bar_timestamps_normalized
                        ]
                    
                    if bars:
                        metrics_dict = {
                            "total_return": result.metrics.total_return,
                            "cagr": result.metrics.cagr,
                            "volatility": result.metrics.volatility,
                            "sharpe": result.metrics.sharpe,
                            "max_drawdown": result.metrics.max_drawdown,
                        }
                        
                        if use_local_chart:
                            chart = LocalChartVisualizer(style="dark_background", figsize=(16, 12))
                            chart.set_data(
                                bars=bars,
                                signals=signals,
                                indicator_data=filtered_indicator_data,
                                equity_curve=result.equity_curve,
                                metrics=metrics_dict,
                                symbol=symbol
                            )
                            chart.show(block=True)
                            chart.close()
                        else:
                            WebChartServer = get_web_chart_server()
                            server = WebChartServer(port=5000)
                            server.set_data(
                                bars=bars,
                                signals=signals,
                                indicator_data=filtered_indicator_data,
                                equity_curve=result.equity_curve,
                                metrics=metrics_dict,
                                symbol=symbol
                            )
                            server.start_server(open_browser=True)
                            print("\nPartial chart server is running. Press Enter to stop...")
                            input()
                except Exception as chart_err:
                    logger.log(f"Could not start partial chart: {chart_err}")
            else:
                print("\nNo partial results available yet.")
        except Exception as e:
            logger.log(f"Could not retrieve partial results: {e}")
        sys.exit(0)
    except Exception as e:
        logger.log(f"Error during backtest: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Revised MP2.0 Strategy Backtest")
    parser.add_argument(
        "--local-chart",
        action="store_true",
        help="Use local Matplotlib chart (legacy) instead of Plotly."
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="tradingview",
        choices=["moomoo", "tradingview", "dark", "light"],
        help="Chart theme for Plotly visualization (default: tradingview)"
    )
    args = parser.parse_args()
    
    try:
        asyncio.run(main(use_local_chart=args.local_chart))
    except KeyboardInterrupt:
        print("\nBacktest interrupted. Exiting...")
        sys.exit(0)

