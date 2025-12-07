import asyncio
import sys
from datetime import datetime
from pathlib import Path
import logging
import argparse
import yaml
from typing import List, Dict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data.factory import create_data_engine_from_config
from core.backtest.scheduler import DecisionScheduler
from core.backtest.engine import BacktestEngine
from core.backtest.benchmarks import run_buy_and_hold
from core.utils.logging import Logger
from core.utils.config_loader import load_config_with_secrets
from core.visualization import (
    LocalChartVisualizer,
    PlotlyChartVisualizer,
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


async def run_backtest_llm_trend_detection(use_local_chart: bool = False):
    """
    Run backtest using LLM_Trend_Detection strategy.
    """
    config_dir = project_root / "config"
    env_file = config_dir / "env.backtest.yaml"
    strategy_file = config_dir / "strategy.llm_trend_detection.yaml"

    if not env_file.exists():
        raise FileNotFoundError(f"Config file not found: {env_file}")
    if not strategy_file.exists():
        raise FileNotFoundError(f"Strategy config file not found: {strategy_file}")

    # Load configs with secrets merged in
    env = load_config_with_secrets(env_file)
    strat_cfg_raw = load_config_with_secrets(strategy_file, strategy_name="llm_trend_detection")

    if "backtest" not in env:
        raise ValueError("Missing 'backtest' in env.backtest.yaml")
    if "symbol" not in strat_cfg_raw:
        raise ValueError("Missing 'symbol' in strategy.llm_trend_detection.yaml")

    symbol = strat_cfg_raw["symbol"]
    bt_cfg = env["backtest"]

    for key in ("start", "end", "initial_cash"):
        if key not in bt_cfg:
            raise ValueError(f"Missing 'backtest.{key}' in env.backtest.yaml")

    data_engine = create_data_engine_from_config(env_config=env, use_for="historical")

    timeframe = bt_cfg.get("timeframe") or strat_cfg_raw.get("timeframe", "1D")
    strat_cfg_raw["timeframe"] = timeframe

    if "timeframe" in bt_cfg:
        print(f"[LLM_Trend] Using timeframe '{timeframe}' from env.backtest.yaml")
    else:
        print(
            f"[LLM_Trend] Using timeframe '{timeframe}' "
            f"from strategy.llm_trend_detection.yaml"
        )

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
        use_llm=strat_cfg_raw.get("use_llm", True),
        openai_api_key=strat_cfg_raw.get("openai_api_key"),  # Read from config
        enable_trading=strat_cfg_raw.get("enable_trading", True),  # Enable trading for performance testing
        capital_deployment_pct=strat_cfg_raw.get("capital_deployment_pct", 1.0),  # Deploy 100% of capital
    )

    strategy = LLMTrendDetectionStrategy(symbol, cfg, data_engine)
    logger = Logger(prefix="[LLM_Trend_Backtest]")

    t = timeframe.upper()
    if t in ("1D", "D"):
        scheduler = DecisionScheduler(interval_minutes=24 * 60)
    elif t.endswith("H"):
        hours = int(t[:-1]) if t != "H" else 1
        scheduler = DecisionScheduler(interval_minutes=hours * 60)
    elif t.endswith("M") or t.endswith("m"):
        minutes = int(t[:-1])
        scheduler = DecisionScheduler(interval_minutes=minutes)
    else:
        scheduler = DecisionScheduler(interval_minutes=15)

    engine = BacktestEngine(data_engine, scheduler, logger)

    start = datetime.fromisoformat(bt_cfg["start"])
    end = datetime.fromisoformat(bt_cfg["end"])
    initial_cash = bt_cfg["initial_cash"]

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

    print("\n=== LLM_Trend_Detection Backtest Results ===")

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
        visualizer.build_chart(
            bars=bars,
            signals=signals,
            indicator_data=indicator_data,
            equity_curve=result.equity_curve,
            metrics=metrics_dict,
            symbol=symbol,
            show_equity=True,
            regime_history=regime_history,  # Pass regime history for trend indicator
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
    args = parser.parse_args()

    try:
        asyncio.run(run_backtest_llm_trend_detection(use_local_chart=args.local_chart))
    except KeyboardInterrupt:
        print("\nBacktest interrupted.")
        sys.exit(0)
