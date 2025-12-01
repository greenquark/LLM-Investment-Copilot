"""
Example Strategy Skeleton

This is a template/skeleton file showing how to implement a custom strategy.
Copy this file and rename it to implement your own strategy logic.

To use this skeleton:
1. Copy this file to a new file (e.g., `my_strategy.py`)
2. Rename the class from `ExampleStrategy` to your strategy name
3. Implement the abstract methods: `on_start`, `on_decision`, `on_end`
4. Add your strategy-specific logic
5. Create a corresponding config class if needed
6. Update .gitignore to exclude your implementation file
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from core.strategy.base import Strategy, Context
from core.data.base import DataEngine
from core.models.bar import Bar


@dataclass
class ExampleStrategyConfig:
    """
    Configuration for the example strategy.
    
    Add your strategy-specific configuration parameters here.
    This config is typically loaded from a YAML file.
    """
    # Example parameters - customize for your strategy
    lookback_period: int = 20
    threshold: float = 0.05
    max_position_size: float = 1.0  # Fraction of portfolio
    
    @classmethod
    def from_dict(cls, config: dict) -> "ExampleStrategyConfig":
        """
        Create config from dictionary (typically loaded from YAML).
        
        Args:
            config: Dictionary with configuration values
            
        Returns:
            ExampleStrategyConfig instance
        """
        return cls(
            lookback_period=config.get("lookback_period", 20),
            threshold=config.get("threshold", 0.05),
            max_position_size=config.get("max_position_size", 1.0),
        )


class ExampleStrategy(Strategy):
    """
    Example strategy implementation.
    
    This class demonstrates the structure of a strategy:
    - Inherits from Strategy base class
    - Takes symbol, config, and data_engine in __init__
    - Implements on_start, on_decision, and on_end methods
    - Uses Context to access portfolio, execution, and logging
    """
    
    def __init__(
        self,
        symbol: str,
        config: ExampleStrategyConfig,
        data_engine: DataEngine,
    ):
        """
        Initialize the strategy.
        
        Args:
            symbol: Trading symbol (e.g., "AAPL", "TQQQ")
            config: Strategy configuration
            data_engine: Data engine for fetching market data
        """
        self.symbol = symbol
        self.config = config
        self.data_engine = data_engine
        
        # Strategy state - add your own state variables here
        self._initialized = False
        self._bars: list[Bar] = []
        # Add more state variables as needed
    
    async def on_start(self, ctx: Context) -> None:
        """
        Called once when the strategy starts (backtest or live trading).
        
        Use this to:
        - Initialize indicators
        - Load historical data
        - Set up any required state
        
        Args:
            ctx: Strategy context with portfolio, execution, symbol, and log
        """
        ctx.log(f"Starting {self.__class__.__name__} strategy for {self.symbol}")
        ctx.log(f"Config: lookback={self.config.lookback_period}, threshold={self.config.threshold}")
        
        # Example: Fetch initial bars for indicator calculation
        # end_time = datetime.now()
        # start_time = ...  # Calculate based on lookback_period
        # self._bars = await self.data_engine.get_bars(
        #     self.symbol, start_time, end_time, timeframe="1D"
        # )
        
        self._initialized = True
        ctx.log("Strategy initialized")
    
    async def on_decision(self, ctx: Context, now: datetime) -> None:
        """
        Called at each decision interval (e.g., every 15 minutes for live, or at each bar for backtest).
        
        This is where your main trading logic goes:
        1. Fetch current market data
        2. Calculate indicators
        3. Make trading decisions
        4. Execute trades via ctx.execution
        
        Args:
            ctx: Strategy context with portfolio, execution, symbol, and log
            now: Current timestamp
        """
        if not self._initialized:
            ctx.log("Warning: Strategy not initialized, skipping decision")
            return
        
        # Example: Fetch latest bars
        # end_time = now
        # start_time = ...  # Calculate based on lookback_period
        # bars = await self.data_engine.get_bars(
        #     self.symbol, start_time, end_time, timeframe="1D"
        # )
        
        # Example: Get current position
        position = ctx.portfolio.get_position(self.symbol)
        current_shares = position.shares if position else 0
        
        ctx.log(f"Decision time: {now}, Current position: {current_shares} shares")
        
        # TODO: Implement your trading logic here
        # Example structure:
        # 1. Calculate indicators/signals
        #    signal = self._calculate_signal(bars)
        #
        # 2. Determine target position
        #    target_shares = self._calculate_target_position(signal, ctx.portfolio)
        #
        # 3. Execute trades if needed
        #    if target_shares != current_shares:
        #        if target_shares > current_shares:
        #            # Buy
        #            shares_to_buy = target_shares - current_shares
        #            await ctx.execution.buy(self.symbol, shares_to_buy)
        #        else:
        #            # Sell
        #            shares_to_sell = current_shares - target_shares
        #            await ctx.execution.sell(self.symbol, shares_to_sell)
        
        # Placeholder: No trading logic implemented
        ctx.log("No trading action (skeleton implementation)")
    
    async def on_end(self, ctx: Context) -> None:
        """
        Called once when the strategy ends (backtest completes or live trading stops).
        
        Use this to:
        - Close out positions if needed
        - Generate final reports
        - Clean up resources
        
        Args:
            ctx: Strategy context with portfolio, execution, symbol, and log
        """
        ctx.log(f"Ending {self.__class__.__name__} strategy for {self.symbol}")
        
        # Example: Close any remaining positions
        # position = ctx.portfolio.get_position(self.symbol)
        # if position and position.shares > 0:
        #     await ctx.execution.sell(self.symbol, position.shares)
        #     ctx.log(f"Closed position: {position.shares} shares")
        
        # Log final portfolio state
        portfolio_value = ctx.portfolio.get_total_value()
        ctx.log(f"Final portfolio value: ${portfolio_value:,.2f}")
    
    # Helper methods - add your own helper methods here
    
    def _calculate_signal(self, bars: list[Bar]) -> float:
        """
        Calculate trading signal from bars.
        
        This is a placeholder - implement your signal calculation logic.
        
        Args:
            bars: List of bars for analysis
            
        Returns:
            Signal value (e.g., -1 for sell, 0 for hold, 1 for buy)
        """
        # TODO: Implement signal calculation
        # Example: Simple moving average crossover
        # if len(bars) < self.config.lookback_period:
        #     return 0
        # 
        # prices = [bar.close for bar in bars]
        # sma = sum(prices[-self.config.lookback_period:]) / self.config.lookback_period
        # current_price = bars[-1].close
        # 
        # if current_price > sma * (1 + self.config.threshold):
        #     return 1  # Buy signal
        # elif current_price < sma * (1 - self.config.threshold):
        #     return -1  # Sell signal
        # return 0  # Hold
        
        return 0  # Placeholder
    
    def _calculate_target_position(
        self,
        signal: float,
        portfolio_value: float,
        current_price: float,
    ) -> int:
        """
        Calculate target position size based on signal and portfolio.
        
        Args:
            signal: Trading signal (-1, 0, or 1)
            portfolio_value: Current portfolio value
            current_price: Current price of the symbol
            
        Returns:
            Target number of shares to hold
        """
        # TODO: Implement position sizing logic
        # Example:
        # if signal == 0:
        #     return 0  # No position
        # 
        # # Calculate position size as fraction of portfolio
        # target_value = portfolio_value * self.config.max_position_size
        # target_shares = int(target_value / current_price)
        # 
        # return target_shares if signal > 0 else 0
        
        return 0  # Placeholder

