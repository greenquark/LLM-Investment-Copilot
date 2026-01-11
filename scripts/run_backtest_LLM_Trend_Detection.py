import asyncio
import sys
from datetime import datetime
from pathlib import Path
import logging
import argparse
import yaml
from typing import List, Dict, Optional, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.models.fill import Fill

from core.data.factory import create_data_engine_from_config
from core.backtest.scheduler import DecisionScheduler
from core.backtest.engine import BacktestEngine
from core.backtest.benchmarks import run_buy_and_hold
from core.backtest.performance import evaluate_performance
from core.utils.logging import Logger
from core.utils.config_loader import load_config_with_secrets
from core.visualization import (
    LocalChartVisualizer,
    PlotlyChartVisualizer,
)
from core.visualization.chart_config import get_chart_config
from core.backtest.backtest_utils import (
    load_backtest_config,
    get_backtest_symbol,
    get_backtest_timeframe,
    parse_backtest_dates,
    print_backtest_header,
    create_scheduler_from_timeframe,
)

from core.strategy.llm_trend_detection import (
    LLMTrendDetectionStrategy,
    LLMTrendDetectionConfig,
)


def _analyze_trend_periods(regime_history: List[Dict]) -> List[Dict]:
    """
    Analyze regime history to find distinct trend periods.
    
    A trend period begins when the regime changes to a new value
    and ends when it changes to a different value.
    
    Args:
        regime_history: List of regime decision dictionaries with 'timestamp' and 'regime' keys
    
    Returns:
        List of dictionaries with period information:
        - regime: The regime type (TREND_UP, TREND_DOWN, RANGE)
        - start_date: When the period began
        - end_date: When the period ended
        - duration_days: Number of days in the period
        - start_price: Price at start (if available)
        - end_price: Price at end (if available)
    """
    if not regime_history:
        return []
    
    periods = []
    current_regime = None
    period_start = None
    period_start_price = None
    previous_timestamp = None
    previous_price = None
    
    for decision in regime_history:
        timestamp = decision.get("timestamp")
        regime = decision.get("regime", "UNKNOWN")
        price = decision.get("price")
        
        # Convert timestamp to datetime if it's a string
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif not isinstance(timestamp, datetime):
            continue  # Skip invalid timestamps
        
        # If regime changed, end previous period and start new one
        if regime != current_regime:
            # End previous period if it exists
            if current_regime is not None and period_start is not None and previous_timestamp is not None:
                periods.append({
                    "regime": current_regime,
                    "start_date": period_start,
                    "end_date": previous_timestamp,
                    "duration_days": (previous_timestamp - period_start).days + 1,
                    "start_price": period_start_price,
                    "end_price": previous_price,
                })
            
            # Start new period
            current_regime = regime
            period_start = timestamp
            period_start_price = price
        
        # Track the most recent timestamp and price for the current period
        previous_timestamp = timestamp
        previous_price = price
    
    # Don't forget the last period
    if current_regime is not None and period_start is not None and previous_timestamp is not None:
        periods.append({
            "regime": current_regime,
            "start_date": period_start,
            "end_date": previous_timestamp,
            "duration_days": (previous_timestamp - period_start).days + 1,
            "start_price": period_start_price,
            "end_price": previous_price,
        })
    
    return periods


async def _extract_trades(engine, symbol: str, data_engine, end: datetime, timeframe: str) -> List[Dict]:
    """
    Extract all trades from the backtest execution engine.
    
    Returns a list of trade dictionaries with entry/exit dates and prices.
    """
    # Access fills from the execution engine
    if not hasattr(engine, '_exec_engine') or engine._exec_engine is None:
        return []
    
    fills = engine._exec_engine._fills
    if not fills:
        return []
    
    # Sort fills by timestamp
    fills_sorted = sorted(fills, key=lambda f: f.timestamp)
    
    # Group fills into trades (entry/exit pairs)
    # Track entry fills with remaining quantities (don't modify original Fill objects)
    trades: List[Dict] = []
    entry_fills: List[Tuple[Fill, int]] = []  # (fill, remaining_quantity)
    
    for fill in fills_sorted:
        if fill.symbol != symbol:
            continue
        
        if fill.quantity > 0:
            # Buy - add to entry fills with full quantity
            entry_fills.append((fill, fill.quantity))
        else:
            # Sell - match with entry fills (FIFO)
            remaining_sell_qty = abs(fill.quantity)
            
            while remaining_sell_qty > 0 and entry_fills:
                entry_fill, entry_remaining_qty = entry_fills[0]
                
                if entry_remaining_qty <= remaining_sell_qty:
                    # Close entire entry
                    trade_qty = entry_remaining_qty
                    entry_fills.pop(0)
                    remaining_sell_qty -= entry_remaining_qty
                else:
                    # Partial close
                    trade_qty = remaining_sell_qty
                    entry_fills[0] = (entry_fill, entry_remaining_qty - remaining_sell_qty)
                    remaining_sell_qty = 0
                
                # Calculate P&L for this trade
                pnl = (fill.price - entry_fill.price) * trade_qty
                pnl_pct = ((fill.price / entry_fill.price) - 1) * 100 if entry_fill.price > 0 else 0.0
                
                trades.append({
                    'entry_date': entry_fill.timestamp,
                    'exit_date': fill.timestamp,
                    'entry_price': entry_fill.price,
                    'exit_price': fill.price,
                    'quantity': trade_qty,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                })
    
    # Handle any remaining open positions (not closed)
    # Get final price for unrealized positions to calculate P&L
    from core.utils.price_utils import get_final_price
    final_price = await get_final_price(
        data_engine=data_engine,
        symbol=symbol,
        target_date=end.date(),
        timeframe=timeframe,
        equity_curve=None,
        final_shares=0.0,
        fallback_to_equity_curve=False,
    )
    
    # Fallback: try to get price from engine's last market price if available
    if final_price == 0.0 and hasattr(engine, '_exec_engine') and engine._exec_engine:
        if hasattr(engine._exec_engine, '_market_prices') and symbol in engine._exec_engine._market_prices:
            final_price = engine._exec_engine._market_prices[symbol]
    
    for entry_fill, remaining_qty in entry_fills:
        # Calculate unrealized P&L as if sold at final price
        if final_price > 0:
            unrealized_pnl = (final_price - entry_fill.price) * remaining_qty
            unrealized_pnl_pct = ((final_price / entry_fill.price) - 1) * 100 if entry_fill.price > 0 else 0.0
        else:
            unrealized_pnl = None
            unrealized_pnl_pct = None
        
        trades.append({
            'entry_date': entry_fill.timestamp,
            'exit_date': None,  # Still open - will show as "UNREALIZED"
            'entry_price': entry_fill.price,
            'exit_price': final_price if final_price > 0 else None,  # Use final price for unrealized
            'quantity': remaining_qty,
            'pnl': unrealized_pnl,  # Calculate P&L as if sold at final price
            'pnl_pct': unrealized_pnl_pct,
        })
    
    return trades


async def _print_trades(engine, symbol: str, data_engine, end: datetime, timeframe: str) -> None:
    """
    Print all trades from the backtest execution engine.
    
    For unrealized positions, calculates P&L as if sold at the end date price
    to fairly compare against buy & hold.
    
    Args:
        engine: BacktestEngine instance (after run completes)
        symbol: Trading symbol
        data_engine: Data engine to fetch final price for unrealized positions
        end: End datetime of the backtest
        timeframe: Timeframe string (e.g., "1D")
    """
    # Extract trades using shared function
    trades = await _extract_trades(engine, symbol, data_engine, end, timeframe)
    
    # Print trades
    print("\n=== Trades ===")
    if not trades:
        print("No trades executed")
        return
    
    print(f"\nTotal Trades: {len(trades)}")
    print(f"{'#':<4} {'Entry Date':<12} {'Exit Date':<12} {'Qty':<8} {'Entry $':<10} {'Exit $':<10} {'P&L $':<12} {'P&L %':<10}")
    print("-" * 90)
    
    total_pnl = 0.0
    closed_trades = 0
    open_trades = 0
    
    for i, trade in enumerate(trades, 1):
        entry_date_str = trade['entry_date'].strftime('%Y-%m-%d')
        # Show "UNREALIZED" for exit date if position is still open
        exit_date_str = trade['exit_date'].strftime('%Y-%m-%d') if trade['exit_date'] else 'UNREALIZED'
        qty = trade['quantity']
        entry_price = trade['entry_price']
        # For unrealized positions, show the exit price (final price) if available
        exit_price = trade['exit_price'] if trade['exit_price'] is not None else 'N/A'
        
        if trade['pnl'] is not None:
            # P&L is calculated (either realized or unrealized)
            pnl_str = f"${trade['pnl']:+,.2f}"
            pnl_pct_str = f"{trade['pnl_pct']:+.2f}%"
            total_pnl += trade['pnl']
            if trade['exit_date'] is not None:
                closed_trades += 1
            else:
                open_trades += 1
        else:
            # P&L couldn't be calculated (shouldn't happen with our changes, but keep as fallback)
            pnl_str = "UNREALIZED"
            pnl_pct_str = "N/A"
            open_trades += 1
        
        # Format exit price - show as float if it's a number, otherwise show as string
        if isinstance(exit_price, (int, float)) and exit_price > 0:
            exit_price_str = f"${exit_price:.2f}"
        else:
            exit_price_str = str(exit_price)
        
        print(f"{i:<4} {entry_date_str:<12} {exit_date_str:<12} {qty:<8} "
              f"${entry_price:<9.2f} {exit_price_str:<9} {pnl_str:<12} {pnl_pct_str:<10}")
    
    print("-" * 90)
    print(f"\nClosed Trades: {closed_trades}")
    print(f"Open Trades: {open_trades}")
    
    # Calculate realized vs total P&L
    realized_pnl = sum(trade['pnl'] for trade in trades if trade['exit_date'] is not None and trade['pnl'] is not None)
    unrealized_pnl = sum(trade['pnl'] for trade in trades if trade['exit_date'] is None and trade['pnl'] is not None)
    
    if closed_trades > 0:
        avg_realized_pnl = realized_pnl / closed_trades
        print(f"Total Realized P&L: ${realized_pnl:+,.2f}")
        print(f"Average P&L per Closed Trade: ${avg_realized_pnl:+,.2f}")
    
    if open_trades > 0 and unrealized_pnl != 0:
        print(f"Total Unrealized P&L (at end price): ${unrealized_pnl:+,.2f}")
    
    if total_pnl != 0:
        print(f"Total P&L (Realized + Unrealized): ${total_pnl:+,.2f}")

# Custom formatter to indent INFO messages for better readability
class IndentedInfoFormatter(logging.Formatter):
    """Formatter that indents INFO level messages."""
    def format(self, record):
        # Indent INFO messages with 2 spaces
        if record.levelno == logging.INFO:
            record.msg = f"  {record.msg}"
        return super().format(record)

# Configure logging with indented INFO messages
handler = logging.StreamHandler()
handler.setFormatter(IndentedInfoFormatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
logging.basicConfig(
    level=logging.INFO,
    handlers=[handler],
)


async def run_backtest_llm_trend_detection(
    use_local_chart: bool = False,
    ticker: Optional[str] = None,
    timeframe: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: Optional[int] = None,
):
    """
    Run backtest using LLM_Trend_Detection strategy.
    
    Args:
        use_local_chart: Use local chart instead of web chart
        ticker: Optional ticker symbol (overrides config)
        timeframe: Optional timeframe (overrides config)
        start_date: Optional start date in YYYY-MM-DD format (overrides config)
        end_date: Optional end date in YYYY-MM-DD format (overrides config)
        days: Optional number of calendar days for backtest period (e.g., 365 for one year)
    """
    # Load backtest configuration using shared utilities
    env, strat_cfg_raw, bt_cfg = load_backtest_config(
        project_root=project_root,
        strategy_config_file="strategy.llm_trend_detection.yaml",
        strategy_name="llm_trend_detection",
    )
    
    # Get symbol and timeframe with proper priority (CLI > config)
    symbol = get_backtest_symbol(bt_cfg, strat_cfg_raw, cli_symbol=ticker)
    timeframe = get_backtest_timeframe(bt_cfg, strat_cfg_raw, default="1D", cli_timeframe=timeframe)
    strat_cfg_raw["timeframe"] = timeframe
    
    # Parse dates with CLI priority
    start, end, initial_cash = parse_backtest_dates(bt_cfg, cli_start_date=start_date, cli_end_date=end_date, cli_days=days)
    
    # Create data engine
    data_engine = create_data_engine_from_config(env_config=env, use_for="historical")

    cfg = LLMTrendDetectionConfig(
        timeframe=strat_cfg_raw.get("timeframe", "1D"),
        lookback_bars=strat_cfg_raw.get("lookback_bars", 250),
        slope_window=strat_cfg_raw.get("slope_window", 60),
        ma_short=strat_cfg_raw.get("ma_short", 20),
        ma_medium=strat_cfg_raw.get("ma_medium", 50),
        ma_long=strat_cfg_raw.get("ma_long", 200),
        rsi_length=strat_cfg_raw.get("rsi_length", 14),
        llm_model=strat_cfg_raw.get("llm_model", "gpt-5-mini"),
        llm_temperature=strat_cfg_raw.get("llm_temperature", 0.0),
        llm_timeout=strat_cfg_raw.get("llm_timeout", 180.0),
        use_llm=strat_cfg_raw.get("use_llm", True),
        openai_api_key=strat_cfg_raw.get("openai_api_key"),  # Read from config
        enable_trading=strat_cfg_raw.get("enable_trading", True),  # Enable trading for performance testing
        capital_deployment_pct=strat_cfg_raw.get("capital_deployment_pct", 1.0),  # Deploy 100% of capital
    )

    strategy = LLMTrendDetectionStrategy(symbol, cfg, data_engine)
    logger = Logger(prefix="[LLM_Trend_Backtest]")

    # Create scheduler from timeframe using shared utility
    scheduler = create_scheduler_from_timeframe(timeframe)
    engine = BacktestEngine(data_engine, scheduler, logger)

    # Print backtest header using shared utility
    print_backtest_header(symbol, start, end, initial_cash, cfg.timeframe)

    logger.log(
        f"Starting LLM_Trend_Detection Backtest for {symbol} "
        f"{start} → {end} | initial_cash=${initial_cash:,.2f}"
    )
    logger.log(
        f"Config(timeframe={cfg.timeframe}, lookback={cfg.lookback_bars}, use_llm={cfg.use_llm})"
    )

    result = await engine.run(
        symbol, strategy, start, end, initial_cash, timeframe=cfg.timeframe
    )

    # Recalculate metrics from equity curve (engine returns dummy metrics)
    result.metrics = evaluate_performance(result.equity_curve)

    print("\n=== LLM_Trend_Detection Backtest Results ===")
    print("\n=== NOTE - Trend signals are not intended to be trading signals. It is intended to be used for trend detection and analysis. This is for debugging only===")

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

    print("\n=== Buy & Hold Benchmark ===")
    bh = await run_buy_and_hold(
        data_engine, symbol, start, end, initial_cash, timeframe=cfg.timeframe
    )
    print(f"Strategy Return: {result.metrics.total_return:.2%}")
    print(f"Buy&Hold Return: {bh.metrics.total_return:.2%}")

    # Print trades for debugging
    await _print_trades(engine, symbol, data_engine, end, cfg.timeframe)
    
    # Extract trades for chart visualization
    trades = await _extract_trades(engine, symbol, data_engine, end, cfg.timeframe)
    
    # Convert trades to TradeSignal objects for chart
    from core.visualization.models import TradeSignal
    trade_signals = []
    for trade in trades:
        # Add buy signal at entry
        trade_signals.append(TradeSignal(
            timestamp=trade['entry_date'],
            price=trade['entry_price'],
            side="BUY",
            trend_score=0,  # Not used for trade signals
            di_plus=None,
            di_minus=None,
        ))
        # Add sell signal at exit (if not unrealized)
        if trade['exit_date'] is not None:
            trade_signals.append(TradeSignal(
                timestamp=trade['exit_date'],
                price=trade['exit_price'] if trade['exit_price'] is not None else trade['entry_price'],
                side="SELL",
                trend_score=0,  # Not used for trade signals
                di_plus=None,
                di_minus=None,
            ))

    # Print regime decision summary
    regime_history = strategy.get_regime_history()
    if regime_history:
        print("\n=== Trend Decision Summary ===")
        regime_counts = {}
        for decision in regime_history:
            regime = decision.get("regime", "UNKNOWN")
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        total_decisions = len(regime_history)
        print(f"Total Decisions: {total_decisions}")
        for regime, count in sorted(regime_counts.items()):
            pct = (count / total_decisions) * 100 if total_decisions > 0 else 0
            print(f"  {regime}: {count} ({pct:.1f}%)")
        
        # Analyze trend periods (beginning and ending dates)
        print("\n=== Trend Periods ===")
        trend_periods = _analyze_trend_periods(regime_history)
        
        if trend_periods:
            print(f"\nFound {len(trend_periods)} trend period(s):\n")
            for i, period in enumerate(trend_periods, 1):
                regime = period["regime"]
                start_date = period["start_date"]
                end_date = period["end_date"]
                duration_days = period["duration_days"]
                start_price = period.get("start_price", "N/A")
                end_price = period.get("end_price", "N/A")
                
                print(f"Period {i}: {regime}")
                print(f"  Start:  {start_date.strftime('%Y-%m-%d')} (Price: ${start_price:.2f})" if isinstance(start_price, (int, float)) else f"  Start:  {start_date.strftime('%Y-%m-%d')} (Price: {start_price})")
                print(f"  End:    {end_date.strftime('%Y-%m-%d')} (Price: ${end_price:.2f})" if isinstance(end_price, (int, float)) else f"  End:    {end_date.strftime('%Y-%m-%d')} (Price: {end_price})")
                print(f"  Duration: {duration_days} day(s)")
                
                # Calculate price change if available
                if isinstance(start_price, (int, float)) and isinstance(end_price, (int, float)) and start_price > 0:
                    price_change = end_price - start_price
                    price_change_pct = (price_change / start_price) * 100
                    print(f"  Price Change: ${price_change:+.2f} ({price_change_pct:+.2f}%)")
                print()
        else:
            print("No distinct trend periods found (all decisions are the same regime)")
        
        # Show latest state
        latest_state = strategy.get_latest_state()
        if latest_state:
            print(f"\nLatest Regime: {latest_state.regime_final}")
            print(f"  Trend Strength: {latest_state.trend_strength:.2f}")
            print(f"  Range Strength: {latest_state.range_strength:.2f}")
            if latest_state.summary_for_user:
                print(f"  Summary: {latest_state.summary_for_user}")

    print("\n=== Chart ===")
    bars = await data_engine.get_bars(symbol, start, end, cfg.timeframe)
    
    # Diagnostic: Check for date gaps in bars
    if bars:
        from datetime import date
        bar_dates = sorted([b.timestamp.date() for b in bars])
        print(f"Retrieved {len(bars)} bars for charting")
        print(f"Bar date range: {bar_dates[0]} to {bar_dates[-1]}")
        print(f"Expected range: {start.date()} to {end.date()}")
        
        # Check for gaps in August 2024
        aug_bars = [d for d in bar_dates if d.month == 8 and d.year == 2024]
        if aug_bars:
            print(f"\nAugust 2024 bars: {len(aug_bars)} dates")
            # Check for gaps around Aug 5-12
            aug_5_12 = [d for d in aug_bars if date(2024, 8, 5) <= d <= date(2024, 8, 12)]
            if aug_5_12:
                print(f"Aug 5-12 bars: {[str(d) for d in aug_5_12]}")
                if len(aug_5_12) < 6:
                    print(f"WARNING: Expected 6 trading days (Aug 5-9, 12), found {len(aug_5_12)}")
                    from core.data import get_trading_days
                    expected_days = get_trading_days(date(2024, 8, 5), date(2024, 8, 12))
                    missing = [d for d in expected_days if d not in aug_5_12]
                    if missing:
                        print(f"Missing dates: {[str(d) for d in missing]}")
    
    signals = strategy.get_signals()
    # Use LLM indicator history which includes BB and RSI
    indicator_data = strategy.get_llm_indicator_history()
    # Fallback to old indicator history if new one is empty (backward compatibility)
    if not indicator_data:
        indicator_data = strategy.get_indicator_history()

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
            indicator_data=indicator_data,
            equity_curve=result.equity_curve,
            metrics=metrics_dict,
            symbol=symbol,
        )
        chart.show(block=True)
        chart.close()
    else:
        visualizer = PlotlyChartVisualizer(theme="tradingview", figsize=(1400, 900))
        chart_config = get_chart_config("llm_trend", use_regime_history=regime_history is not None)
        visualizer.build_chart(
            bars=bars,
            signals=signals,
            indicator_data=indicator_data,
            equity_curve=result.equity_curve,
            metrics=metrics_dict,
            symbol=symbol,
            show_equity=True,
            regime_history=regime_history,  # Pass regime history for trend indicator
            chart_config=chart_config,
            strategy_name="llm_trend",
            trade_signals=trade_signals,  # Pass trade signals for overlay
        )
        print("Opening interactive chart in browser...")
        visualizer.show(renderer="browser")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM_Trend_Detection Backtest")
    parser.add_argument(
        "--local-chart",
        action="store_true",
        help="Use local Matplotlib chart instead of Plotly.",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Ticker symbol to use (overrides config files). Example: --ticker TQQQ",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Timeframe to use (overrides config files). Example: --timeframe 1D",
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
        asyncio.run(run_backtest_llm_trend_detection(
            use_local_chart=args.local_chart,
            ticker=args.ticker,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            days=args.days,
        ))
    except KeyboardInterrupt:
        print("\nBacktest interrupted.")
        sys.exit(0)
