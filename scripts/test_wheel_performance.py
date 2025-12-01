import asyncio
from datetime import datetime

import yaml

from core.data.marketdata_app import MarketDataAppAdapter
from core.data.wheel_view import WheelDataView
from core.backtest.scheduler import DecisionScheduler
from core.backtest.engine import BacktestEngine
from core.backtest.benchmarks import run_buy_and_hold
from core.strategy.wheel import WheelStrategy, WheelStrategyConfig
from core.utils.logging import Logger
from core.utils.config_loader import load_config_with_secrets

async def main():
    from pathlib import Path
    config_dir = Path("config")
    env_file = config_dir / "env.backtest.yaml"
    strategy_file = config_dir / "strategy.wheel.yaml"
    
    # Load configs with secrets merged in
    env = load_config_with_secrets(env_file)
    strat_cfg_raw = load_config_with_secrets(strategy_file)

    symbol = strat_cfg_raw["symbol"]
    # Use new data_sources structure
    mkt_cfg = env.get("data_sources", {}).get("marketdata_app", {})
    bt_cfg = env["backtest"]

    data_engine = MarketDataAppAdapter(api_token=mkt_cfg.get("api_token", ""))
    wheel_view = WheelDataView(data_engine, symbol)
    strat_cfg = WheelStrategyConfig(strat_cfg_raw)

    logger = Logger(prefix="[PERF]")
    scheduler = DecisionScheduler(interval_minutes=15)
    strategy = WheelStrategy(symbol, strat_cfg, wheel_view)
    engine = BacktestEngine(data_engine, scheduler, logger)

    start = datetime.fromisoformat(bt_cfg["start"])
    end = datetime.fromisoformat(bt_cfg["end"])
    initial_cash = bt_cfg["initial_cash"]

    wheel_result = await engine.run(symbol, strategy, start, end, initial_cash)
    bh_result = await run_buy_and_hold(data_engine, symbol, start, end, initial_cash)

    w = wheel_result.metrics
    b = bh_result.metrics

    print("=== Wheel vs Buy-and-Hold Performance ===")
    print(f"Symbol: {symbol}")
    print()
    print("Wheel strategy:")
    print(f"  Total return : {w.total_return:.2%}")
    print(f"  CAGR         : {w.cagr:.2%}")
    print(f"  Volatility   : {w.volatility:.2%}")
    print(f"  Sharpe       : {w.sharpe:.2f}")
    print(f"  Max drawdown : {w.max_drawdown:.2%}")
    print()
    print("Buy-and-hold:")
    print(f"  Total return : {b.total_return:.2%}")
    print(f"  CAGR         : {b.cagr:.2%}")
    print(f"  Volatility   : {b.volatility:.2%}")
    print(f"  Sharpe       : {b.sharpe:.2f}")
    print(f"  Max drawdown : {b.max_drawdown:.2%}")

if __name__ == "__main__":
    asyncio.run(main())
