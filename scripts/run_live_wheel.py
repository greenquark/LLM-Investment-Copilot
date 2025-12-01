import asyncio
import yaml

from core.data.moomoo_data import MoomooDataAdapter
from core.data.wheel_view import WheelDataView
from core.execution.moomoo_exec import MoomooExecutionEngine
from core.portfolio.portfolio import Portfolio, PortfolioState
from core.strategy.wheel import WheelStrategy, WheelStrategyConfig
from core.live.engine import LiveEngine, LiveEngineConfig
from core.utils.logging import Logger
from core.utils.config_loader import load_config_with_secrets

async def main():
    from pathlib import Path
    config_dir = Path("config")
    env_file = config_dir / "env.live.yaml"
    strategy_file = config_dir / "strategy.wheel.yaml"
    
    # Load configs with secrets merged in
    env = load_config_with_secrets(env_file)
    strat_cfg_raw = load_config_with_secrets(strategy_file)

    symbol = strat_cfg_raw["symbol"]
    moo_cfg = env["moomoo"]
    live_cfg_raw = env["live"]

    data_engine = MoomooDataAdapter(host=moo_cfg["host"], port=moo_cfg["port"])
    exec_engine = MoomooExecutionEngine(
        host=moo_cfg["host"],
        port=moo_cfg["port"],
        account_id=moo_cfg["account_id"],
    )
    wheel_view = WheelDataView(data_engine, symbol)
    strat_cfg = WheelStrategyConfig(strat_cfg_raw)

    portfolio = Portfolio(PortfolioState(cash=live_cfg_raw["initial_cash"]))
    logger = Logger(prefix="[LIVE]")
    strategy = WheelStrategy(symbol, strat_cfg, wheel_view)

    live_cfg = LiveEngineConfig(interval_minutes=live_cfg_raw["interval_minutes"])
    engine = LiveEngine(data_engine, exec_engine, portfolio, strategy, logger, live_cfg)

    stop = asyncio.Event()
    try:
        await engine.run(symbol, stop)
    except KeyboardInterrupt:
        stop.set()

if __name__ == "__main__":
    asyncio.run(main())
