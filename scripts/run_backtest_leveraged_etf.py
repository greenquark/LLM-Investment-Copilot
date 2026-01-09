"""
Backtest script for Leveraged ETF Volatility Swing Trading Strategy.

This script runs a backtest of the Leveraged ETF Volatility Swing strategy,
which exploits daily and weekly price fluctuations using:
- Trend filters (regime detection via underlying index)
- Volatility-based mean reversion
- Probability-driven targets
- Laddered execution

This strategy works with any bull/bear leveraged ETF pair. Examples:
- SOXL/SOXS (semiconductors, regime: SOXX)
- TQQQ/SQQQ (NASDAQ, regime: QQQ)
- UPRO/SPXU (S&P 500, regime: SPY)
- TNA/TZA (small caps, regime: IWM)
- LABU/LABD (biotech, regime: XBI)
"""

import asyncio
import sys
import os
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
from core.strategy.leveraged_etf_vol_swing import LeveragedETFVolSwingStrategy, LeveragedETFVolSwingConfig
from core.utils.logging import Logger
from core.utils.config_loader import load_config_with_secrets
from core.visualization import PlotlyChartVisualizer
from core.visualization.chart_config import get_chart_config


async def main(use_local_chart: bool = False):
    """Main backtest function."""
    # Use absolute paths for config files
    config_dir = project_root / "config"
    env_file = config_dir / "env.backtest.yaml"
    strategy_file = config_dir / "strategy.leveraged_etf_vol_swing.yaml"
    
    if not env_file.exists():
        raise FileNotFoundError(f"Config file not found: {env_file}")
    if not strategy_file.exists():
        raise FileNotFoundError(f"Strategy config file not found: {strategy_file}")
    
    # Load configs with secrets merged in
    env = load_config_with_secrets(env_file)
    strat_cfg_raw = load_config_with_secrets(strategy_file)
    
    # Validate required config keys
    if "backtest" not in env:
        raise ValueError("Missing 'backtest' section in config")
    
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
    # Timeframe can be set in env.backtest.yaml (takes precedence) or strategy config
    timeframe = bt_cfg.get("timeframe") or strat_cfg_raw.get("timeframe", "1D")
    
    # Log which timeframe is being used and from where
    if "timeframe" in bt_cfg:
        print(f"Using timeframe '{timeframe}' from env.backtest.yaml")
    else:
        print(f"Using timeframe '{timeframe}' from strategy.leveraged_etf_vol_swing.yaml (env.backtest.yaml timeframe not set)")
    
    strategy_config = LeveragedETFVolSwingConfig(
        regime_symbol=strat_cfg_raw.get("regime_symbol", "SOXX"),
        ema_fast=strat_cfg_raw.get("ema_fast", 20),
        ema_slow=strat_cfg_raw.get("ema_slow", 50),
        bull_etf_symbol=strat_cfg_raw.get("bull_etf_symbol", "SOXL"),
        bear_etf_symbol=strat_cfg_raw.get("bear_etf_symbol", "SOXS"),
        bb_length=strat_cfg_raw.get("bb_length", 20),
        bb_std=strat_cfg_raw.get("bb_std", 2.0),
        atr_length=strat_cfg_raw.get("atr_length", 14),
        rsi_fast=strat_cfg_raw.get("rsi_fast", 3),
        rsi_slow=strat_cfg_raw.get("rsi_slow", 14),
        volume_ma_length=strat_cfg_raw.get("volume_ma_length", 20),
        rsi_fast_threshold=strat_cfg_raw.get("rsi_fast_threshold", 10.0),
        rsi_slow_threshold=strat_cfg_raw.get("rsi_slow_threshold", 45.0),
        volume_threshold=strat_cfg_raw.get("volume_threshold", 0.7),
        atr_support_range_min=strat_cfg_raw.get("atr_support_range_min", 0.5),
        atr_support_range_max=strat_cfg_raw.get("atr_support_range_max", 0.8),
        entry_ladder_1_pct=strat_cfg_raw.get("entry_ladder_1_pct", 0.6),
        entry_ladder_1_atr_offset=strat_cfg_raw.get("entry_ladder_1_atr_offset", 0.3),
        entry_ladder_2_pct=strat_cfg_raw.get("entry_ladder_2_pct", 0.4),
        entry_ladder_2_atr_offset=strat_cfg_raw.get("entry_ladder_2_atr_offset", 0.7),
        stop_atr_multiple=strat_cfg_raw.get("stop_atr_multiple", 1.0),
        target_1_atr_multiple=strat_cfg_raw.get("target_1_atr_multiple", 1.0),
        target_2_atr_multiple=strat_cfg_raw.get("target_2_atr_multiple", 1.5),
        max_holding_days=strat_cfg_raw.get("max_holding_days", 5),
        risk_per_trade_pct=strat_cfg_raw.get("risk_per_trade_pct", 0.75),
        kelly_fraction=strat_cfg_raw.get("kelly_fraction", 0.25),
        target_probability=strat_cfg_raw.get("target_probability", 0.55),
        weekly_ma_length=strat_cfg_raw.get("weekly_ma_length", 10),
        timeframe=timeframe,
        capital_deployment_pct=strat_cfg_raw.get("capital_deployment_pct", 1.0),
    )
    
    # Create strategy instance
    strategy = LeveragedETFVolSwingStrategy(
        bull_etf_symbol=strategy_config.bull_etf_symbol,
        bear_etf_symbol=strategy_config.bear_etf_symbol,
        regime_symbol=strategy_config.regime_symbol,
        config=strategy_config,
        data_engine=data_engine,
    )
    
    logger = Logger(prefix="[BACKTEST]")
    
    # Calculate scheduler interval based on timeframe
    # Strategy uses daily bars
    timeframe_upper = timeframe.upper()
    if timeframe_upper in ("1D", "D"):
        scheduler = DecisionScheduler(interval_minutes=24 * 60)
    else:
        # Default to daily if not specified
        scheduler = DecisionScheduler(interval_minutes=24 * 60)
    
    engine = BacktestEngine(data_engine, scheduler, logger)
    
    start = datetime.fromisoformat(bt_cfg["start"])
    end = datetime.fromisoformat(bt_cfg["end"])
    initial_cash = bt_cfg["initial_cash"]
    
    # Print backtest date range
    print("\n" + "=" * 80)
    print("BACKTEST DATE RANGE")
    print("=" * 80)
    print(f"Start Date: {start.date()}")
    print(f"End Date:   {end.date()}")
    print(f"Initial Cash: ${initial_cash:,.2f}")
    print(f"Timeframe: {strategy_config.timeframe}")
    print("=" * 80)
    print()
    
    logger.log(f"Starting Leveraged ETF Volatility Swing backtest from {start} to {end} with ${initial_cash:,.2f}")
    logger.log(f"Strategy config:")
    logger.log(f"  Regime symbol: {strategy_config.regime_symbol} (underlying index)")
    logger.log(f"  Bull ETF symbol: {strategy_config.bull_etf_symbol} (3x long)")
    logger.log(f"  Bear ETF symbol: {strategy_config.bear_etf_symbol} (3x short)")
    logger.log(f"  Risk per trade: {strategy_config.risk_per_trade_pct}%")
    logger.log(f"  Capital deployment: {strategy_config.capital_deployment_pct*100:.0f}% per trade")
    logger.log("Press Ctrl-C to stop the backtest early and see partial results")
    
    try:
        # Note: The strategy manages multiple symbols (bull_etf, bear_etf, regime_symbol)
        # For now, we'll use bull_etf as the primary symbol for the engine
        # The strategy will fetch data for all three symbols internally
        result = await engine.run(
            symbol=strategy_config.bull_etf_symbol,  # Primary symbol for engine
            strategy=strategy,
            start=start,
            end=end,
            initial_cash=initial_cash,
            timeframe=strategy_config.timeframe,
        )
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
        
        print(f"\n=== Performance Metrics ===")
        print(f"  Total Return: {result.metrics.total_return:.2%}")
        print(f"  CAGR: {result.metrics.cagr:.2%}")
        print(f"  Volatility: {result.metrics.volatility:.2%}")
        print(f"  Sharpe Ratio: {result.metrics.sharpe:.2f}")
        print(f"  Max Drawdown: {result.metrics.max_drawdown:.2%}")
        if result.metrics.max_drawdown_start:
            print(f"  Max DD Period: {result.metrics.max_drawdown_start.date()} to {result.metrics.max_drawdown_end.date() if result.metrics.max_drawdown_end else 'N/A'}")
        
        # Calculate Buy & Hold benchmark for bull ETF (primary instrument)
        print(f"\n=== Calculating Buy & Hold Benchmark ({strategy_config.bull_etf_symbol}) ===")
        try:
            bh_result = await run_buy_and_hold(
                data_engine,
                strategy_config.bull_etf_symbol,
                start,
                end,
                initial_cash,
                timeframe=strategy_config.timeframe,
            )
            
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
            full_hit_rate = (cache_stats['cache_hits'] / cache_stats['total_requests']) * 100 if cache_stats['total_requests'] > 0 else 0.0
            total_with_cache = cache_stats['cache_hits'] + cache_stats['cache_partial_hits']
            overall_cache_rate = (total_with_cache / cache_stats['total_requests']) * 100 if cache_stats['total_requests'] > 0 else 0.0
            
            print(f"  Full Cache Hit Rate: {full_hit_rate:.1f}%")
            print(f"  Cache Used Rate:     {overall_cache_rate:.1f}% (includes partial hits)")
            print(f"  Bars from Cache:     {cache_stats['total_bars_from_cache']}")
            print(f"  Bars from API:       {cache_stats['total_bars_from_api']}")
        
        # Generate visualization
        print(f"\n=== Starting Chart Visualization ===")
        try:
            # Get bars for all symbols used in strategy
            print(f"Fetching bars for chart: {strategy_config.bull_etf_symbol}, {strategy_config.bear_etf_symbol}, {strategy_config.regime_symbol}")
            bull_etf_bars = await data_engine.get_bars(
                strategy_config.bull_etf_symbol, start, end, strategy_config.timeframe
            )
            bear_etf_bars = await data_engine.get_bars(
                strategy_config.bear_etf_symbol, start, end, strategy_config.timeframe
            )
            regime_bars = await data_engine.get_bars(
                strategy_config.regime_symbol, start, end, strategy_config.timeframe
            )
            
            # Use bull ETF bars as primary for chart (can be enhanced to show all three)
            bars = bull_etf_bars
            signals = strategy.get_signals()
            indicator_data = strategy.get_indicator_history()
            
            print(f"Retrieved {len(bars)} bars for charting ({strategy_config.bull_etf_symbol})")
            print(f"Retrieved {len(signals)} trade signals")
            print(f"Retrieved {len(indicator_data)} indicator data points")
            
            if bars:
                # Log date range for debugging
                bar_dates = sorted([b.timestamp.date() for b in bars])
                print(f"Bar date range: {bar_dates[0]} to {bar_dates[-1]}")
                print(f"Last 5 bar dates: {[str(d) for d in bar_dates[-5:]]}")
                if indicator_data:
                    indicator_dates = sorted([ind.timestamp.date() for ind in indicator_data])
                    print(f"Indicator date range: {indicator_dates[0]} to {indicator_dates[-1]}")
                    print(f"Last 5 indicator dates: {[str(d) for d in indicator_dates[-5:]]}")
            
            if bars:
                # Create chart visualizer
                visualizer = PlotlyChartVisualizer(theme="tradingview", figsize=(1400, 900))
                
                # Build chart with data
                metrics_dict = {
                    "total_return": result.metrics.total_return,
                    "cagr": result.metrics.cagr,
                    "volatility": result.metrics.volatility,
                    "sharpe": result.metrics.sharpe,
                    "max_drawdown": result.metrics.max_drawdown,
                }
                
                chart_config = get_chart_config("leveraged_etf")
                chart = visualizer.build_chart(
                    bars=bars,
                    signals=signals,
                    indicator_data=indicator_data,
                    equity_curve=result.equity_curve,
                    metrics=metrics_dict,
                    symbol=strategy_config.bull_etf_symbol,
                    chart_config=chart_config,
                    strategy_name="leveraged_etf",
                    show_equity=True,
                )
                
                # Show chart
                print("\n📊 Displaying interactive chart...")
                visualizer.show(renderer="browser")
                
                # Optionally save to HTML
                save_html = input("\nSave chart to HTML file? (y/n): ").strip().lower()
                if save_html == 'y':
                    output_file = project_root / "charts" / f"leveraged_etf_backtest_{start.date()}_to_{end.date()}.html"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    visualizer.to_html(str(output_file))
                    print(f"✅ Chart saved to: {output_file}")
        
        except Exception as e:
            logger.log(f"Failed to generate chart: {e}")
            print(f"\n⚠️  Could not generate chart: {e}")
            import traceback
            traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Backtest interrupted by user (Ctrl-C)")
        print("Retrieving partial results...")
        partial_result = engine.get_partial_result()
        if partial_result:
            print("\n=== Partial Backtest Results ===")
            if partial_result.equity_curve:
                equity_items = sorted(partial_result.equity_curve.items(), key=lambda x: x[0])
                if equity_items:
                    final_equity = equity_items[-1][1]
                    initial_equity = equity_items[0][1]
                    total_pnl = final_equity - initial_equity
                    total_pnl_pct = (final_equity / initial_equity - 1.0) * 100 if initial_equity > 0 else 0.0
                    print(f"  Initial Equity: ${initial_equity:,.2f}")
                    print(f"  Final Equity: ${final_equity:,.2f}")
                    print(f"  Total P&L: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)")
        else:
            print("No partial results available")
    except Exception as e:
        logger.log(f"Backtest failed: {e}")
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Leveraged ETF Volatility Swing Strategy backtest")
    parser.add_argument(
        "--local-chart",
        action="store_true",
        help="Use local Python chart instead of web chart",
    )
    args = parser.parse_args()
    
    asyncio.run(main(use_local_chart=args.local_chart))

